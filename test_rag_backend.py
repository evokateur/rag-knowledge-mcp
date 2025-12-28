#!/usr/bin/env python3
"""
Test RAG Backend Implementation

Tests the complete backend functionality including read and write operations.
Uses TEST_* environment variables from .env for test database configuration.
"""

import shutil
import pytest
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from config import create_rag_backend

# Load test environment variables
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_knowledge_base():
    """
    Create test markdown files in ./test-knowledge-base/.

    Does NOT ingest - just creates the files.
    Tests can call ingest_directory() or add_document() as needed.
    """
    kb_dir = Path("./test-knowledge-base")
    kb_dir.mkdir(exist_ok=True)

    # Create test markdown files
    (kb_dir / "ml_basics.md").write_text(
        """# Machine Learning Basics

Machine learning is a subset of artificial intelligence that focuses on
building systems that can learn from data. Common techniques include
supervised learning, unsupervised learning, and reinforcement learning.
Neural networks are a key component of deep learning approaches.

## Key Concepts

- Supervised learning uses labeled data
- Unsupervised learning finds patterns in unlabeled data
- Reinforcement learning learns through rewards
"""
    )

    (kb_dir / "python_intro.md").write_text(
        """# Python Programming

Python is a high-level programming language known for its readability
and versatility. It's widely used in data science, web development,
and automation. Popular frameworks include Django, Flask, and FastAPI.
Python's extensive library ecosystem makes it ideal for rapid development.

## Popular Uses

- Data science and machine learning
- Web development
- Automation and scripting
"""
    )

    (kb_dir / "vector_db_guide.md").write_text(
        """# Vector Database Guide

Vector databases store high-dimensional vectors and enable similarity search.
They are essential for RAG systems, enabling semantic search over large
document collections. Popular options include Chroma, Pinecone, and Qdrant.
Embedding models convert text into vectors for storage and retrieval.

## Common Vector Databases

- ChromaDB - open source, embedded
- Pinecone - managed cloud service
- Qdrant - high performance, self-hosted
"""
    )

    yield kb_dir

    # Cleanup
    shutil.rmtree(kb_dir, ignore_errors=True)


@pytest.fixture
async def backend():
    """Create and initialize a backend instance for testing."""
    backend = create_rag_backend()
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def backend_with_data(backend, test_knowledge_base):
    """
    Backend with pre-ingested test data from test-knowledge-base/.

    Uses ingest_directory() to load all test markdown files.
    Tests that use this fixture assume ingestion works correctly.
    """
    await backend.ingest_directory(directory=str(test_knowledge_base))
    return backend


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.asyncio
async def test_backend_initialization(backend):
    """Test that backend initializes successfully."""
    stats = await backend.get_stats()
    assert "total_documents" in stats
    assert "total_chunks" in stats
    assert "embedding_model" in stats
    assert "vector_dimension" in stats


@pytest.mark.asyncio
async def test_config_loaded(backend):
    """Test that backend has config loaded."""
    assert backend.config is not None
    assert hasattr(backend.config, "persist_dir")
    assert hasattr(backend.config, "collection")
    assert hasattr(backend.config, "embedding_model")


# ============================================================================
# Ingestion Tests (explicitly test write operations)
# ============================================================================


@pytest.mark.asyncio
async def test_ingest_directory(backend, test_knowledge_base):
    """Test bulk directory ingestion workflow."""
    stats = await backend.ingest_directory(directory=str(test_knowledge_base))

    assert stats["documents_processed"] == 3
    assert stats["total_chunks"] > 0
    assert len(stats["files_processed"]) == 3

    # Verify documents were actually ingested
    backend_stats = await backend.get_stats()
    assert backend_stats["total_documents"] == 3


@pytest.mark.asyncio
async def test_ingest_directory_always_rebuilds(backend, test_knowledge_base):
    """Test that ingestion always rebuilds (no duplicates on re-run)."""
    # First ingestion
    await backend.ingest_directory(directory=str(test_knowledge_base))

    stats1 = await backend.get_stats()
    doc_count1 = stats1["total_documents"]

    # Second ingestion (should rebuild, not add duplicates)
    await backend.ingest_directory(directory=str(test_knowledge_base))

    stats2 = await backend.get_stats()
    doc_count2 = stats2["total_documents"]

    # Should have same count (not doubled)
    assert doc_count2 == doc_count1


@pytest.mark.asyncio
async def test_add_document(backend):
    """Test adding a single document."""
    result = await backend.add_document(
        content="Test document about artificial intelligence and machine learning.",
        metadata={
            "source": "test.md",
            "author": "Test Author",
            "created_at": datetime.now().isoformat(),
        },
        chunk_size=50,
        chunk_overlap=10,
    )

    assert "document_id" in result
    assert "chunks_created" in result
    assert result["chunks_created"] > 0

    # Verify it was added
    doc = await backend.get_document(result["document_id"])
    assert doc is not None
    assert doc["id"] == result["document_id"]


# ============================================================================
# Search Tests (use backend_with_data - assumes ingestion works)
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_search(backend_with_data):
    """Test semantic search functionality."""
    test_cases = [
        ("What is machine learning?", 3),
        ("Tell me about Python programming", 2),
        ("How do vector databases work?", 2),
    ]

    for query, top_k in test_cases:
        results = await backend_with_data.search(
            query=query, top_k=top_k, score_threshold=0.0
        )

        assert isinstance(results, list)
        assert len(results) <= top_k

        # Verify result structure
        if results:
            for result in results:
                assert "id" in result
                assert "content" in result
                assert "score" in result
                assert "metadata" in result
                assert 0.0 <= result["score"] <= 1.0


@pytest.mark.asyncio
async def test_search_with_score_threshold(backend_with_data):
    """Test search with score threshold filtering."""
    # Search with high threshold
    results_high = await backend_with_data.search(
        query="machine learning", top_k=10, score_threshold=0.8
    )

    # Search with low threshold
    results_low = await backend_with_data.search(
        query="machine learning", top_k=10, score_threshold=0.0
    )

    # High threshold should return fewer or equal results
    assert len(results_high) <= len(results_low)


@pytest.mark.asyncio
async def test_search_relevance(backend_with_data):
    """Test that search returns relevant results."""
    # Query about machine learning should find ML document
    results = await backend_with_data.search(
        query="supervised learning neural networks", top_k=5, score_threshold=0.3
    )

    assert len(results) > 0
    # Top result should mention ML-related terms
    top_result = results[0]
    content_lower = top_result["content"].lower()
    assert any(
        term in content_lower
        for term in ["machine learning", "supervised", "neural", "learning"]
    )


# ============================================================================
# Document Retrieval Tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_documents(backend_with_data):
    """Test listing documents with pagination."""
    doc_list = await backend_with_data.list_documents(limit=10, offset=0)

    assert "total" in doc_list
    assert "count" in doc_list
    assert "offset" in doc_list
    assert "documents" in doc_list
    assert "has_more" in doc_list

    assert doc_list["total"] == 3
    assert doc_list["count"] == 3
    assert len(doc_list["documents"]) == 3

    # Verify document structure
    for doc in doc_list["documents"]:
        assert "id" in doc
        assert "source" in doc
        assert "created_at" in doc


@pytest.mark.asyncio
async def test_get_document(backend_with_data):
    """Test retrieving a specific document by ID."""
    # List documents to get a real ID
    doc_list = await backend_with_data.list_documents(limit=1, offset=0)
    assert doc_list["count"] > 0

    doc_id = doc_list["documents"][0]["id"]

    # Retrieve the document
    doc = await backend_with_data.get_document(doc_id)

    assert doc is not None
    assert doc["id"] == doc_id
    assert "source" in doc
    assert "content" in doc
    assert "chunk_count" in doc
    assert doc["chunk_count"] > 0


@pytest.mark.asyncio
async def test_get_nonexistent_document(backend):
    """Test retrieving a document that doesn't exist."""
    doc = await backend.get_document("nonexistent_id_12345")
    assert doc is None


@pytest.mark.asyncio
async def test_get_stats(backend_with_data):
    """Test retrieving knowledge base statistics."""
    stats = await backend_with_data.get_stats()

    assert stats["total_documents"] == 3
    assert stats["total_chunks"] > 0
    assert isinstance(stats["embedding_model"], str)
    assert isinstance(stats["vector_dimension"], int)
    assert stats["vector_dimension"] > 0


# ============================================================================
# Document Deletion Tests
# ============================================================================


@pytest.mark.asyncio
async def test_delete_document(backend):
    """Test deleting a document and its chunks."""
    # Add a document
    result = await backend.add_document(
        content="Test document for deletion.",
        metadata={"source": "delete_test.md", "created_at": datetime.now().isoformat()},
        chunk_size=50,
        chunk_overlap=10,
    )
    doc_id = result["document_id"]

    # Verify it exists
    doc = await backend.get_document(doc_id)
    assert doc is not None

    # Delete it
    deleted = await backend.delete_document(doc_id)
    assert deleted is True

    # Verify it's gone
    doc = await backend.get_document(doc_id)
    assert doc is None


@pytest.mark.asyncio
async def test_delete_nonexistent_document(backend):
    """Test deleting a document that doesn't exist."""
    deleted = await backend.delete_document("nonexistent_id_12345")
    assert deleted is False
