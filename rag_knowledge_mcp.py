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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Goes to stderr by default
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
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def app_lifespan():
    """
    Manage resources that persist for the server's lifetime.

    Initializes the RAG backend on startup and cleans up on shutdown.
    Configuration is loaded from config.py (which reads from .env if present).
    The backend implementation is determined by RAG_BACKEND_CLASS.
    """
    from config import create_rag_backend

    rag_backend = create_rag_backend()
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
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(
        ...,
        description="Search query text for semantic similarity matching",
        min_length=1,
        max_length=1000,
    )
    top_k: int = Field(
        default=5, description="Maximum number of results to return (1-50)", ge=1, le=50
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score threshold (0.0-1.0). Results below this are filtered out.",
        ge=0.0,
        le=1.0,
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include document metadata in results"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class ListDocumentsInput(BaseModel):
    """Input parameters for listing documents."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    limit: int = Field(
        default=20,
        description="Maximum number of documents to return per page (1-100)",
        ge=1,
        le=100,
    )
    offset: int = Field(
        default=0, description="Number of documents to skip for pagination", ge=0
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


class GetDocumentInput(BaseModel):
    """Input parameters for retrieving a specific document."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    document_id: str = Field(
        ...,
        description="Unique identifier of the document to retrieve",
        min_length=1,
        max_length=200,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable",
    )


# ============================================================================
# Helper Functions
# ============================================================================


def format_search_results_markdown(
    results: List[Dict[str, Any]], include_metadata: bool = True
) -> str:
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

        if include_metadata and "metadata" in result:
            metadata = result["metadata"]
            lines.append("**Metadata:**")
            lines.append(f"- Source: {metadata.get('source', 'Unknown')}")
            if "page" in metadata:
                lines.append(f"- Page: {metadata['page']}")
            if "created_at" in metadata:
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
    if doc_data["count"] == 0:
        return "No documents found in the knowledge base."

    lines = [
        f"# Documents ({doc_data['count']} of {doc_data['total']})\n",
        f"Showing items {doc_data['offset'] + 1}-{doc_data['offset'] + doc_data['count']}\n",
    ]

    for doc in doc_data["documents"]:
        lines.append(f"## {doc.get('source', 'Untitled')}")
        lines.append(f"- **ID:** {doc['id']}")
        if "author" in doc:
            lines.append(f"- **Author:** {doc['author']}")
        if "tags" in doc and doc["tags"]:
            lines.append(f"- **Tags:** {', '.join(doc['tags'])}")
        lines.append(f"- **Created:** {doc.get('created_at', 'Unknown')}")
        lines.append("")

    if doc_data["has_more"]:
        lines.append(
            f"\n*More results available. Use offset={doc_data['next_offset']} to see next page.*"
        )

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
        f"**Created:** {doc.get('created_at', 'Unknown')}\n",
    ]

    if "author" in doc:
        lines.append(f"**Author:** {doc['author']}")

    if "tags" in doc and doc["tags"]:
        lines.append(f"**Tags:** {', '.join(doc['tags'])}\n")

    lines.append("## Content\n")
    lines.append(doc.get("content", "No content available"))

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
        "openWorldHint": False,
    },
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
        rag_backend: RagBackend = ctx.request_context.lifespan_state["rag_backend"]

        ctx.log_info(
            f"Searching knowledge base: '{params.query}' (top_k={params.top_k})"
        )

        # Perform semantic search
        results = await rag_backend.search(
            query=params.query,
            top_k=params.top_k,
            score_threshold=params.score_threshold,
        )

        # Filter by score threshold (defensive, backend should handle this)
        results = [r for r in results if r["score"] >= params.score_threshold]

        ctx.log_info(
            f"Found {len(results)} results above threshold {params.score_threshold}"
        )

        # Format response
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(
                {
                    "query": params.query,
                    "results_count": len(results),
                    "results": results,
                },
                indent=2,
            )
        else:
            return format_search_results_markdown(results, params.include_metadata)

    except Exception as e:
        error_msg = f"Error searching knowledge base: {str(e)}"
        ctx.log_error(error_msg)
        return f"Error: {error_msg}\n\nPlease check your query and try again."


@mcp.tool(
    name="rag_list_documents",
    annotations={
        "title": "List Documents in Knowledge Base",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
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
        rag_backend: RagBackend = ctx.request_context.lifespan_state["rag_backend"]

        ctx.log_info(
            f"Listing documents (limit={params.limit}, offset={params.offset})"
        )

        # Get document list
        doc_data = await rag_backend.list_documents(
            limit=params.limit, offset=params.offset
        )

        ctx.log_info(
            f"Found {doc_data['count']} documents (total: {doc_data['total']})"
        )

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
        "openWorldHint": False,
    },
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
        rag_backend: RagBackend = ctx.request_context.lifespan_state["rag_backend"]

        ctx.log_info(f"Retrieving document: {params.document_id}")

        # Get document
        doc = await rag_backend.get_document(params.document_id)

        if not doc:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Document '{params.document_id}' not found",
                },
                indent=2,
            )

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
        "openWorldHint": False,
    },
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
        rag_backend: RagBackend = ctx.request_context.lifespan_state["rag_backend"]

        ctx.log_info("Retrieving knowledge base statistics")

        # Get stats
        stats = await rag_backend.get_stats()

        ctx.log_info(
            f"Stats retrieved: {stats['total_documents']} documents, {stats['total_chunks']} chunks"
        )

        return json.dumps(stats, indent=2)

    except Exception as e:
        error_msg = f"Error retrieving statistics: {str(e)}"
        ctx.log_error(error_msg)
        return json.dumps({"success": False, "error": error_msg}, indent=2)


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with stdio transport (default for local MCP integrations)
    # To use streamable HTTP instead: mcp.run(transport="streamable_http", port=8000)
    mcp.run()
