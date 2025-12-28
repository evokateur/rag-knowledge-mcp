"""
End-to-end tests for the RAG Knowledge MCP Server.

Tests the actual MCP server tools (rag_search_knowledge, rag_list_documents, etc.)
rather than just the backend. This validates the complete integration:
FastMCP → Tools → Backend → Response formatting.
"""

import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from contextlib import asynccontextmanager

from rag_knowledge_mcp import (
    app_lifespan,
    search_knowledge,
    list_documents,
    get_document,
    get_stats,
    SearchKnowledgeInput,
    ListDocumentsInput,
    GetDocumentInput,
    ResponseFormat,
)


@pytest.fixture
def test_knowledge_base():
    """Create test markdown files for ingestion."""
    kb_dir = Path("./test-knowledge-base")
    kb_dir.mkdir(exist_ok=True)

    (kb_dir / "ml_basics.md").write_text(
        """# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed.

Key concepts include:
- Supervised learning with labeled data
- Unsupervised learning for pattern discovery
- Neural networks and deep learning
- Model training and evaluation
"""
    )

    (kb_dir / "python_intro.md").write_text(
        """# Python Programming

Python is a high-level, interpreted programming language known for its simplicity
and readability. It's widely used in data science, web development, and automation.

Core features:
- Dynamic typing and duck typing
- Extensive standard library
- Rich ecosystem of third-party packages
- Support for multiple programming paradigms
"""
    )

    (kb_dir / "vector_db_guide.md").write_text(
        """# Vector Database Guide

Vector databases are specialized systems for storing and querying high-dimensional
vectors. They enable semantic search and similarity matching for AI applications.

Popular vector databases:
- ChromaDB for local development
- Pinecone for production use
- Weaviate for hybrid search
- Qdrant for performance-critical applications
"""
    )

    yield kb_dir

    shutil.rmtree(kb_dir, ignore_errors=True)


@pytest.fixture
async def mcp_context(test_knowledge_base):
    """
    Create MCP lifespan context with initialized backend and test data.

    This simulates the actual MCP server lifecycle, initializing the backend
    via the app_lifespan context manager and ingesting test data.
    """
    async with app_lifespan(None) as lifespan_state:
        rag_backend = lifespan_state["rag_backend"]

        await rag_backend.ingest_directory(directory=str(test_knowledge_base))

        mock_request_context = Mock()
        mock_request_context.lifespan_state = lifespan_state

        mock_ctx = Mock()
        mock_ctx.request_context = mock_request_context
        mock_ctx.log_info = Mock()
        mock_ctx.log_error = Mock()

        yield mock_ctx


@pytest.mark.asyncio
async def test_search_knowledge_markdown(mcp_context):
    """Test rag_search_knowledge tool with markdown output."""
    params = SearchKnowledgeInput(
        query="machine learning concepts",
        top_k=3,
        score_threshold=0.0,
        include_metadata=True,
        response_format=ResponseFormat.MARKDOWN,
    )

    result = await search_knowledge(params, mcp_context)

    assert isinstance(result, str)
    assert "Search Results" in result
    assert "Score:" in result
    assert "machine learning" in result.lower() or "supervised" in result.lower()
    assert "Metadata:" in result
    assert "Source:" in result


@pytest.mark.asyncio
async def test_search_knowledge_json(mcp_context):
    """Test rag_search_knowledge tool with JSON output."""
    params = SearchKnowledgeInput(
        query="Python programming language",
        top_k=5,
        score_threshold=0.0,
        include_metadata=True,
        response_format=ResponseFormat.JSON,
    )

    result = await search_knowledge(params, mcp_context)

    assert isinstance(result, str)
    data = json.loads(result)

    assert "query" in data
    assert data["query"] == "Python programming language"
    assert "results_count" in data
    assert "results" in data
    assert isinstance(data["results"], list)
    assert data["results_count"] > 0

    first_result = data["results"][0]
    assert "content" in first_result
    assert "score" in first_result
    assert "metadata" in first_result
    assert isinstance(first_result["score"], (int, float))


@pytest.mark.asyncio
async def test_search_knowledge_score_threshold(mcp_context):
    """Test search with score threshold filtering."""
    params_high_threshold = SearchKnowledgeInput(
        query="completely unrelated quantum physics topic",
        top_k=10,
        score_threshold=0.9,
        response_format=ResponseFormat.JSON,
    )

    result = await search_knowledge(params_high_threshold, mcp_context)
    data = json.loads(result)

    assert data["results_count"] >= 0
    for res in data["results"]:
        assert res["score"] >= 0.9


@pytest.mark.asyncio
async def test_search_knowledge_empty_results(mcp_context):
    """Test search that returns no results above threshold."""
    params = SearchKnowledgeInput(
        query="xyz",
        top_k=5,
        score_threshold=0.99,
        response_format=ResponseFormat.MARKDOWN,
    )

    result = await search_knowledge(params, mcp_context)

    assert isinstance(result, str)
    assert "No results found" in result or "0 found" in result


@pytest.mark.asyncio
async def test_list_documents_markdown(mcp_context):
    """Test rag_list_documents tool with markdown output."""
    params = ListDocumentsInput(
        limit=20, offset=0, response_format=ResponseFormat.MARKDOWN
    )

    result = await list_documents(params, mcp_context)

    assert isinstance(result, str)
    assert "Documents" in result
    assert "ID:" in result
    assert result.count("##") >= 3


@pytest.mark.asyncio
async def test_list_documents_json(mcp_context):
    """Test rag_list_documents tool with JSON output."""
    params = ListDocumentsInput(
        limit=10, offset=0, response_format=ResponseFormat.JSON
    )

    result = await list_documents(params, mcp_context)

    assert isinstance(result, str)
    data = json.loads(result)

    assert "total" in data
    assert "count" in data
    assert "offset" in data
    assert "documents" in data
    assert "has_more" in data

    assert data["total"] == 3
    assert data["count"] == 3
    assert data["offset"] == 0
    assert len(data["documents"]) == 3
    assert data["has_more"] is False

    for doc in data["documents"]:
        assert "id" in doc
        assert "source" in doc
        assert "created_at" in doc


@pytest.mark.asyncio
async def test_list_documents_pagination(mcp_context):
    """Test document list pagination."""
    params_page1 = ListDocumentsInput(
        limit=2, offset=0, response_format=ResponseFormat.JSON
    )

    result_page1 = await list_documents(params_page1, mcp_context)
    data_page1 = json.loads(result_page1)

    assert data_page1["count"] == 2
    assert data_page1["total"] == 3
    assert data_page1["has_more"] is True
    assert data_page1["next_offset"] == 2

    params_page2 = ListDocumentsInput(
        limit=2, offset=2, response_format=ResponseFormat.JSON
    )

    result_page2 = await list_documents(params_page2, mcp_context)
    data_page2 = json.loads(result_page2)

    assert data_page2["count"] == 1
    assert data_page2["total"] == 3
    assert data_page2["has_more"] is False


@pytest.mark.asyncio
async def test_get_document_markdown(mcp_context):
    """Test rag_get_document tool with markdown output."""
    list_params = ListDocumentsInput(
        limit=1, offset=0, response_format=ResponseFormat.JSON
    )
    list_result = await list_documents(list_params, mcp_context)
    list_data = json.loads(list_result)

    document_id = list_data["documents"][0]["id"]

    get_params = GetDocumentInput(
        document_id=document_id, response_format=ResponseFormat.MARKDOWN
    )

    result = await get_document(get_params, mcp_context)

    assert isinstance(result, str)
    assert "ID:" in result
    assert "Created:" in result
    assert "Content" in result
    assert document_id in result


@pytest.mark.asyncio
async def test_get_document_json(mcp_context):
    """Test rag_get_document tool with JSON output."""
    list_params = ListDocumentsInput(
        limit=1, offset=0, response_format=ResponseFormat.JSON
    )
    list_result = await list_documents(list_params, mcp_context)
    list_data = json.loads(list_result)

    document_id = list_data["documents"][0]["id"]

    get_params = GetDocumentInput(
        document_id=document_id, response_format=ResponseFormat.JSON
    )

    result = await get_document(get_params, mcp_context)

    assert isinstance(result, str)
    data = json.loads(result)

    assert "id" in data
    assert data["id"] == document_id
    assert "source" in data
    assert "content" in data
    assert "created_at" in data
    assert "chunk_count" in data
    assert isinstance(data["chunk_count"], int)
    assert data["chunk_count"] > 0


@pytest.mark.asyncio
async def test_get_document_not_found(mcp_context):
    """Test get_document with non-existent document ID."""
    get_params = GetDocumentInput(
        document_id="nonexistent_doc_id", response_format=ResponseFormat.JSON
    )

    result = await get_document(get_params, mcp_context)

    assert isinstance(result, str)
    data = json.loads(result)

    assert "success" in data
    assert data["success"] is False
    assert "error" in data
    assert "not found" in data["error"].lower()


@pytest.mark.asyncio
async def test_get_stats(mcp_context):
    """Test rag_get_stats tool."""
    result = await get_stats(mcp_context)

    assert isinstance(result, str)
    data = json.loads(result)

    assert "total_documents" in data
    assert "total_chunks" in data
    assert "embedding_model" in data
    assert "vector_dimension" in data
    assert "collection_name" in data
    assert "persist_directory" in data

    assert data["total_documents"] == 3
    assert data["total_chunks"] > 0
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["vector_dimension"] == 384
    assert isinstance(data["collection_name"], str)
    assert len(data["collection_name"]) > 0


@pytest.mark.asyncio
async def test_input_validation():
    """Test Pydantic input validation for tool parameters."""
    with pytest.raises(Exception):
        SearchKnowledgeInput(
            query="",
            top_k=5,
        )

    with pytest.raises(Exception):
        SearchKnowledgeInput(
            query="valid query", top_k=100
        )

    with pytest.raises(Exception):
        SearchKnowledgeInput(
            query="valid query", top_k=5, score_threshold=1.5
        )

    with pytest.raises(Exception):
        ListDocumentsInput(limit=0)

    with pytest.raises(Exception):
        ListDocumentsInput(limit=20, offset=-1)

    with pytest.raises(Exception):
        GetDocumentInput(document_id="")


@pytest.mark.asyncio
async def test_search_with_metadata_filtering(mcp_context):
    """Test search includes metadata in results."""
    params = SearchKnowledgeInput(
        query="vector database",
        top_k=3,
        include_metadata=True,
        response_format=ResponseFormat.JSON,
    )

    result = await search_knowledge(params, mcp_context)
    data = json.loads(result)

    assert data["results_count"] > 0

    for res in data["results"]:
        assert "metadata" in res
        metadata = res["metadata"]
        assert "source" in metadata
        assert "parent_doc" in metadata
        assert "chunk_index" in metadata


@pytest.mark.asyncio
async def test_complete_workflow(mcp_context):
    """
    Test complete workflow: stats → search → list → get document.

    This simulates a typical MCP client interaction.
    """
    stats_result = await get_stats(mcp_context)
    stats_data = json.loads(stats_result)
    assert stats_data["total_documents"] == 3

    search_params = SearchKnowledgeInput(
        query="Python", top_k=5, response_format=ResponseFormat.JSON
    )
    search_result = await search_knowledge(search_params, mcp_context)
    search_data = json.loads(search_result)
    assert search_data["results_count"] > 0

    list_params = ListDocumentsInput(
        limit=10, offset=0, response_format=ResponseFormat.JSON
    )
    list_result = await list_documents(list_params, mcp_context)
    list_data = json.loads(list_result)
    assert list_data["count"] == 3

    document_id = list_data["documents"][0]["id"]
    get_params = GetDocumentInput(
        document_id=document_id, response_format=ResponseFormat.JSON
    )
    get_result = await get_document(get_params, mcp_context)
    get_data = json.loads(get_result)
    assert get_data["id"] == document_id
    assert "content" in get_data
