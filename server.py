#!/usr/bin/env python3
"""
RAG Knowledge Base MCP Server

A Model Context Protocol server that provides semantic search capabilities
over a RAG (Retrieval-Augmented Generation) knowledge base. This server
enables LLMs to query, retrieve, and manage documents using vector similarity
search with configurable chunking and embedding strategies.

The actual RAG implementation (chunking, embedding, vector DB) is pluggable,
allowing you to integrate your preferred tools and models.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Configure logging to stderr (not stdout for stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Goes to stderr by default
)
logger = logging.getLogger(__name__)

# ============================================================================
# Response Format Enum
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


# ============================================================================
# RAG Backend Interface (Pluggable)
# ============================================================================

class RAGBackend:
    """
    Abstract interface for RAG backend implementation.
    
    Implement this interface with your actual vector database,
    embedding model, and chunking strategy.
    """
    
    async def initialize(self):
        """Initialize vector database connections and load models."""
        logger.info("Initializing RAG backend...")
        # TODO: Connect to your vector database (Chroma, Pinecone, FAISS, etc.)
        # TODO: Load your embedding model (OpenAI, Sentence Transformers, etc.)
        # This is a placeholder - replace with actual implementation
        pass
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over the knowledge base.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0-1.0)
            filters: Optional metadata filters
            
        Returns:
            List of search results with score, content, and metadata
        """
        # TODO: Implement actual semantic search
        # 1. Encode query with your embedding model
        # 2. Query vector database
        # 3. Filter by score threshold
        # 4. Return formatted results
        
        # Placeholder response
        return [
            {
                "id": "doc_001",
                "content": "This is a placeholder chunk. Replace with actual RAG implementation.",
                "score": 0.95,
                "metadata": {
                    "source": "example.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "created_at": "2024-01-15T10:30:00Z"
                }
            }
        ]
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document text content
            metadata: Document metadata (source, author, etc.)
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            
        Returns:
            Document ID and statistics
        """
        # TODO: Implement document ingestion
        # 1. Chunk the document with your strategy
        # 2. Generate embeddings for each chunk
        # 3. Store in vector database
        # 4. Return document ID and stats
        
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return {
            "document_id": doc_id,
            "chunks_created": 5,  # Placeholder
            "metadata": metadata
        }
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deleted, False if not found
        """
        # TODO: Implement document deletion
        return True
    
    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            Paginated list of documents with metadata
        """
        # TODO: Implement document listing
        return {
            "total": 0,
            "count": 0,
            "offset": offset,
            "documents": [],
            "has_more": False,
            "next_offset": None
        }
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            Document data or None if not found
        """
        # TODO: Implement document retrieval
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics including document count, chunk count, etc.
        """
        # TODO: Implement statistics retrieval
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_model": "placeholder-model",
            "vector_dimension": 768
        }
    
    async def close(self):
        """Clean up connections and resources."""
        logger.info("Closing RAG backend connections...")
        # TODO: Close vector database connections
        pass


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def app_lifespan():
    """
    Manage resources that persist for the server's lifetime.
    
    Initializes the RAG backend on startup and cleans up on shutdown.
    """
    rag_backend = RAGBackend()
    await rag_backend.initialize()
    
    yield {"rag_backend": rag_backend}
    
    await rag_backend.close()


# ============================================================================
# Initialize MCP Server
# ============================================================================

mcp = FastMCP("rag_knowledge_mcp", lifespan=app_lifespan)


# ============================================================================
# Pydantic Input Models
# ============================================================================

class SearchKnowledgeInput(BaseModel):
    """Input parameters for semantic knowledge base search."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    query: str = Field(
        ...,
        description="Search query text for semantic similarity matching",
        min_length=1,
        max_length=1000
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of results to return (1-50)",
        ge=1,
        le=50
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score threshold (0.0-1.0). Results below this are filtered out.",
        ge=0.0,
        le=1.0
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include document metadata in results"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class AddDocumentInput(BaseModel):
    """Input parameters for adding a document to the knowledge base."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    content: str = Field(
        ...,
        description="Full text content of the document to add",
        min_length=1
    )
    source: str = Field(
        ...,
        description="Source identifier (e.g., filename, URL, or document title)",
        min_length=1,
        max_length=500
    )
    author: Optional[str] = Field(
        default=None,
        description="Document author name",
        max_length=200
    )
    tags: Optional[List[str]] = Field(
        default_factory=list,
        description="List of tags or categories for the document",
        max_items=20
    )
    chunk_size: int = Field(
        default=512,
        description="Size of text chunks for embedding (128-2048 tokens)",
        ge=128,
        le=2048
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between consecutive chunks (0-512 tokens)",
        ge=0,
        le=512
    )


class DeleteDocumentInput(BaseModel):
    """Input parameters for deleting a document."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    document_id: str = Field(
        ...,
        description="Unique identifier of the document to delete",
        min_length=1,
        max_length=200
    )


class ListDocumentsInput(BaseModel):
    """Input parameters for listing documents."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    limit: int = Field(
        default=20,
        description="Maximum number of documents to return per page (1-100)",
        ge=1,
        le=100
    )
    offset: int = Field(
        default=0,
        description="Number of documents to skip for pagination",
        ge=0
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class GetDocumentInput(BaseModel):
    """Input parameters for retrieving a specific document."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    document_id: str = Field(
        ...,
        description="Unique identifier of the document to retrieve",
        min_length=1,
        max_length=200
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def format_search_results_markdown(results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
    """
    Format search results as human-readable Markdown.
    
    Args:
        results: List of search result dictionaries
        include_metadata: Whether to include metadata in output
        
    Returns:
        Formatted Markdown string
    """
    if not results:
        return "No results found matching your query."
    
    lines = [f"# Search Results ({len(results)} found)\n"]
    
    for i, result in enumerate(results, 1):
        lines.append(f"## Result {i} (Score: {result['score']:.3f})\n")
        lines.append(f"{result['content']}\n")
        
        if include_metadata and 'metadata' in result:
            metadata = result['metadata']
            lines.append("**Metadata:**")
            lines.append(f"- Source: {metadata.get('source', 'Unknown')}")
            if 'page' in metadata:
                lines.append(f"- Page: {metadata['page']}")
            if 'created_at' in metadata:
                lines.append(f"- Created: {metadata['created_at']}")
            lines.append("")
        
        lines.append("---\n")
    
    return "\n".join(lines)


def format_document_list_markdown(doc_data: Dict[str, Any]) -> str:
    """
    Format document list as human-readable Markdown.
    
    Args:
        doc_data: Document list data with pagination info
        
    Returns:
        Formatted Markdown string
    """
    if doc_data['count'] == 0:
        return "No documents found in the knowledge base."
    
    lines = [
        f"# Documents ({doc_data['count']} of {doc_data['total']})\n",
        f"Showing items {doc_data['offset'] + 1}-{doc_data['offset'] + doc_data['count']}\n"
    ]
    
    for doc in doc_data['documents']:
        lines.append(f"## {doc.get('source', 'Untitled')}")
        lines.append(f"- **ID:** {doc['id']}")
        if 'author' in doc:
            lines.append(f"- **Author:** {doc['author']}")
        if 'tags' in doc and doc['tags']:
            lines.append(f"- **Tags:** {', '.join(doc['tags'])}")
        lines.append(f"- **Created:** {doc.get('created_at', 'Unknown')}")
        lines.append("")
    
    if doc_data['has_more']:
        lines.append(f"\n*More results available. Use offset={doc_data['next_offset']} to see next page.*")
    
    return "\n".join(lines)


def format_document_markdown(doc: Dict[str, Any]) -> str:
    """
    Format single document as human-readable Markdown.
    
    Args:
        doc: Document data dictionary
        
    Returns:
        Formatted Markdown string
    """
    lines = [
        f"# {doc.get('source', 'Untitled Document')}\n",
        f"**ID:** {doc['id']}",
        f"**Created:** {doc.get('created_at', 'Unknown')}\n"
    ]
    
    if 'author' in doc:
        lines.append(f"**Author:** {doc['author']}")
    
    if 'tags' in doc and doc['tags']:
        lines.append(f"**Tags:** {', '.join(doc['tags'])}\n")
    
    lines.append("## Content\n")
    lines.append(doc.get('content', 'No content available'))
    
    return "\n".join(lines)


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="rag_search_knowledge",
    annotations={
        "title": "Search Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def search_knowledge(params: SearchKnowledgeInput, ctx: Context) -> str:
    """
    Search the knowledge base using semantic similarity.
    
    Performs vector similarity search over indexed documents to find
    the most relevant content matching the query. Results are ranked
    by similarity score and can be filtered by a minimum threshold.
    
    Args:
        params (SearchKnowledgeInput): Search parameters including:
            - query (str): Search query text
            - top_k (int): Maximum results to return (default: 5)
            - score_threshold (float): Minimum similarity score (default: 0.0)
            - include_metadata (bool): Include document metadata (default: True)
            - response_format (str): Output format ('markdown' or 'json')
    
    Returns:
        str: Search results in requested format (Markdown or JSON)
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info(f"Searching knowledge base: '{params.query}' (top_k={params.top_k})")
        
        # Perform semantic search
        results = await rag_backend.search(
            query=params.query,
            top_k=params.top_k,
            score_threshold=params.score_threshold
        )
        
        # Filter by score threshold (defensive, backend should handle this)
        results = [r for r in results if r['score'] >= params.score_threshold]
        
        ctx.log_info(f"Found {len(results)} results above threshold {params.score_threshold}")
        
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "query": params.query,
                "results_count": len(results),
                "results": results
            }, indent=2)
        else:
            return format_search_results_markdown(results, params.include_metadata)
            
    except Exception as e:
        error_msg = f"Error searching knowledge base: {str(e)}"
        ctx.log_error(error_msg)
        return f"Error: {error_msg}\n\nPlease check your query and try again."


@mcp.tool(
    name="rag_add_document",
    annotations={
        "title": "Add Document to Knowledge Base",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def add_document(params: AddDocumentInput, ctx: Context) -> str:
    """
    Add a new document to the knowledge base.
    
    Processes the document by chunking the text, generating embeddings,
    and storing in the vector database for future semantic search.
    
    Args:
        params (AddDocumentInput): Document parameters including:
            - content (str): Full document text
            - source (str): Source identifier (filename, URL, etc.)
            - author (Optional[str]): Document author
            - tags (Optional[List[str]]): Document tags/categories
            - chunk_size (int): Size of chunks for embedding (default: 512)
            - chunk_overlap (int): Overlap between chunks (default: 50)
    
    Returns:
        str: JSON response with document ID and ingestion statistics
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info(f"Adding document: {params.source}")
        ctx.report_progress(0.1, "Preparing document for ingestion...")
        
        # Prepare metadata
        metadata = {
            "source": params.source,
            "created_at": datetime.now().isoformat()
        }
        if params.author:
            metadata["author"] = params.author
        if params.tags:
            metadata["tags"] = params.tags
        
        ctx.report_progress(0.3, "Chunking and embedding document...")
        
        # Add document to knowledge base
        result = await rag_backend.add_document(
            content=params.content,
            metadata=metadata,
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap
        )
        
        ctx.report_progress(1.0, "Document successfully added")
        ctx.log_info(f"Document added: {result['document_id']} ({result['chunks_created']} chunks)")
        
        return json.dumps({
            "success": True,
            "document_id": result["document_id"],
            "chunks_created": result["chunks_created"],
            "metadata": result["metadata"]
        }, indent=2)
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        ctx.log_error(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2)


@mcp.tool(
    name="rag_delete_document",
    annotations={
        "title": "Delete Document from Knowledge Base",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def delete_document(params: DeleteDocumentInput, ctx: Context) -> str:
    """
    Delete a document from the knowledge base.
    
    Removes the document and all its associated chunks from the vector database.
    This operation is destructive and cannot be undone.
    
    Args:
        params (DeleteDocumentInput): Deletion parameters including:
            - document_id (str): Unique identifier of document to delete
    
    Returns:
        str: JSON response indicating success or failure
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info(f"Deleting document: {params.document_id}")
        
        # Delete document
        deleted = await rag_backend.delete_document(params.document_id)
        
        if deleted:
            ctx.log_info(f"Document deleted: {params.document_id}")
            return json.dumps({
                "success": True,
                "message": f"Document '{params.document_id}' successfully deleted"
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Document '{params.document_id}' not found"
            }, indent=2)
            
    except Exception as e:
        error_msg = f"Error deleting document: {str(e)}"
        ctx.log_error(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2)


@mcp.tool(
    name="rag_list_documents",
    annotations={
        "title": "List Documents in Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_documents(params: ListDocumentsInput, ctx: Context) -> str:
    """
    List all documents in the knowledge base with pagination.
    
    Returns a paginated list of documents with their metadata.
    Use the offset parameter to navigate through pages of results.
    
    Args:
        params (ListDocumentsInput): List parameters including:
            - limit (int): Maximum documents per page (default: 20)
            - offset (int): Number of documents to skip (default: 0)
            - response_format (str): Output format ('markdown' or 'json')
    
    Returns:
        str: Document list in requested format with pagination metadata
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info(f"Listing documents (limit={params.limit}, offset={params.offset})")
        
        # Get document list
        doc_data = await rag_backend.list_documents(
            limit=params.limit,
            offset=params.offset
        )
        
        ctx.log_info(f"Found {doc_data['count']} documents (total: {doc_data['total']})")
        
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(doc_data, indent=2)
        else:
            return format_document_list_markdown(doc_data)
            
    except Exception as e:
        error_msg = f"Error listing documents: {str(e)}"
        ctx.log_error(error_msg)
        return f"Error: {error_msg}\n\nPlease try again."


@mcp.tool(
    name="rag_get_document",
    annotations={
        "title": "Get Document by ID",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_document(params: GetDocumentInput, ctx: Context) -> str:
    """
    Retrieve a specific document by its ID.
    
    Returns the complete document content and metadata.
    
    Args:
        params (GetDocumentInput): Retrieval parameters including:
            - document_id (str): Unique identifier of document
            - response_format (str): Output format ('markdown' or 'json')
    
    Returns:
        str: Document content in requested format, or error if not found
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info(f"Retrieving document: {params.document_id}")
        
        # Get document
        doc = await rag_backend.get_document(params.document_id)
        
        if not doc:
            return json.dumps({
                "success": False,
                "error": f"Document '{params.document_id}' not found"
            }, indent=2)
        
        ctx.log_info(f"Document retrieved: {params.document_id}")
        
        # Format response
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(doc, indent=2)
        else:
            return format_document_markdown(doc)
            
    except Exception as e:
        error_msg = f"Error retrieving document: {str(e)}"
        ctx.log_error(error_msg)
        return f"Error: {error_msg}\n\nPlease check the document ID and try again."


@mcp.tool(
    name="rag_get_stats",
    annotations={
        "title": "Get Knowledge Base Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_stats(ctx: Context) -> str:
    """
    Get statistics about the knowledge base.
    
    Returns information about the total number of documents, chunks,
    embedding model configuration, and other metadata.
    
    Returns:
        str: JSON-formatted statistics
    """
    try:
        rag_backend: RAGBackend = ctx.request_context.lifespan_state["rag_backend"]
        
        ctx.log_info("Retrieving knowledge base statistics")
        
        # Get stats
        stats = await rag_backend.get_stats()
        
        ctx.log_info(f"Stats retrieved: {stats['total_documents']} documents, {stats['total_chunks']} chunks")
        
        return json.dumps(stats, indent=2)
        
    except Exception as e:
        error_msg = f"Error retrieving statistics: {str(e)}"
        ctx.log_error(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg
        }, indent=2)


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with stdio transport (default for local MCP integrations)
    # To use streamable HTTP instead: mcp.run(transport="streamable_http", port=8000)
    mcp.run()
