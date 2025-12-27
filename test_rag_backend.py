#!/usr/bin/env python3
"""
Test script for RAG Knowledge MCP Server

This script helps you test your RAG backend implementation
independently of the MCP layer.

Usage:
    python test_rag_backend.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


async def test_rag_backend():
    """Test the RAG backend implementation."""

    print("=" * 60)
    print("RAG Backend Test Suite")
    print("=" * 60)
    print()

    # Import your RAG backend
    # Replace with your actual implementation
    try:
        from chroma_backend import RagBackend

        print("✓ Successfully imported RagBackend")
    except ImportError as e:
        print(f"✗ Failed to import RagBackend: {e}")
        print("\nMake sure you've installed dependencies:")
        print("  uv sync")
        return

    # Initialize backend
    print("\n1. Initializing RAG backend...")
    print("-" * 60)
    try:
        backend = RagBackend(
            persist_directory=os.getenv("TEST_RAG_PERSIST_DIR", "./test_chroma_db"),
            collection_name=os.getenv("TEST_RAG_COLLECTION", "knowledge_base"),
            embedding_model=os.getenv("TEST_RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        await backend.initialize()
        print("✓ Backend initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return

    # Test 1: Add sample documents
    print("\n2. Adding sample documents...")
    print("-" * 60)

    sample_docs = [
        {
            "content": """
            Machine learning is a subset of artificial intelligence that focuses on 
            building systems that can learn from data. Common techniques include 
            supervised learning, unsupervised learning, and reinforcement learning.
            Neural networks are a key component of deep learning approaches.
            """,
            "metadata": {
                "source": "ml_basics.txt",
                "author": "AI Researcher",
                "tags": "machine-learning, ai, basics",
                "created_at": datetime.now().isoformat(),
            },
        },
        {
            "content": """
            Python is a high-level programming language known for its readability
            and versatility. It's widely used in data science, web development,
            and automation. Popular frameworks include Django, Flask, and FastAPI.
            Python's extensive library ecosystem makes it ideal for rapid development.
            """,
            "metadata": {
                "source": "python_intro.txt",
                "author": "Developer",
                "tags": "python, programming, development",
                "created_at": datetime.now().isoformat(),
            },
        },
        {
            "content": """
            Vector databases store high-dimensional vectors and enable similarity search.
            They are essential for RAG systems, enabling semantic search over large
            document collections. Popular options include Chroma, Pinecone, and Qdrant.
            Embedding models convert text into vectors for storage and retrieval.
            """,
            "metadata": {
                "source": "vector_db_guide.txt",
                "author": "Database Expert",
                "tags": "vector-database, rag, embeddings",
                "created_at": datetime.now().isoformat(),
            },
        },
    ]

    doc_ids = []
    for i, doc in enumerate(sample_docs, 1):
        try:
            result = await backend.add_document(
                content=doc["content"],
                metadata=doc["metadata"],
                chunk_size=50,  # Small chunks for testing
                chunk_overlap=10,
            )
            doc_ids.append(result["document_id"])
            print(
                f"✓ Document {i} added: {result['document_id']} ({result['chunks_created']} chunks)"
            )
        except Exception as e:
            print(f"✗ Failed to add document {i}: {e}")

    # Test 2: Get statistics
    print("\n3. Checking knowledge base statistics...")
    print("-" * 60)
    try:
        stats = await backend.get_stats()
        print("✓ Statistics retrieved:")
        print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Total chunks: {stats['total_chunks']}")
        print(f"  - Embedding model: {stats['embedding_model']}")
        print(f"  - Vector dimension: {stats['vector_dimension']}")
    except Exception as e:
        print(f"✗ Failed to get statistics: {e}")

    # Test 3: Semantic search
    print("\n4. Testing semantic search...")
    print("-" * 60)

    test_queries = [
        ("What is machine learning?", 3),
        ("Tell me about Python programming", 2),
        ("How do vector databases work?", 2),
    ]

    for query, top_k in test_queries:
        print(f"\nQuery: '{query}' (top_k={top_k})")
        try:
            results = await backend.search(
                query=query, top_k=top_k, score_threshold=0.3
            )

            if results:
                print(f"✓ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n  Result {i} (score: {result['score']:.3f}):")
                    content_preview = result["content"][:100].replace("\n", " ")
                    print(f"    Content: {content_preview}...")
                    print(f"    Source: {result['metadata'].get('source', 'Unknown')}")
            else:
                print("  ⚠ No results found above threshold")

        except Exception as e:
            print(f"✗ Search failed: {e}")

    # Test 4: List documents
    print("\n5. Listing documents...")
    print("-" * 60)
    try:
        doc_list = await backend.list_documents(limit=10, offset=0)
        print(f"✓ Found {doc_list['total']} documents:")
        for doc in doc_list["documents"]:
            print(f"  - {doc['id']} ({doc['source']})")
    except Exception as e:
        print(f"✗ Failed to list documents: {e}")

    # Test 5: Get specific document
    print("\n6. Retrieving specific document...")
    print("-" * 60)
    if doc_ids:
        try:
            doc = await backend.get_document(doc_ids[0])
            if doc:
                print(f"✓ Retrieved document: {doc['id']}")
                print(f"  Source: {doc['source']}")
                print(f"  Author: {doc.get('author', 'N/A')}")
                print(f"  Chunks: {doc['chunk_count']}")
                print(f"  Content preview: {doc['content'][:100]}...")
            else:
                print("✗ Document not found")
        except Exception as e:
            print(f"✗ Failed to get document: {e}")

    # Test 6: Delete document
    print("\n7. Testing document deletion...")
    print("-" * 60)
    if doc_ids:
        try:
            deleted = await backend.delete_document(doc_ids[0])
            if deleted:
                print(f"✓ Document {doc_ids[0]} deleted successfully")

                # Verify deletion
                doc = await backend.get_document(doc_ids[0])
                if doc is None:
                    print("✓ Confirmed: Document no longer exists")
                else:
                    print("⚠ Warning: Document still exists after deletion")
            else:
                print("✗ Document not found for deletion")
        except Exception as e:
            print(f"✗ Failed to delete document: {e}")

    # Cleanup
    print("\n8. Cleaning up...")
    print("-" * 60)
    try:
        await backend.close()
        print("✓ Backend closed successfully")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If all tests passed, your backend is ready!")
    print("2. Configure Claude Desktop/Code to use your MCP server")
    print("3. Test with actual queries through Claude")
    print("\nTo test the MCP server directly:")
    print("  npx @modelcontextprotocol/inspector python rag_knowledge_mcp.py")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(test_rag_backend())
