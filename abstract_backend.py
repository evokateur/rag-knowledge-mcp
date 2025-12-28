"""
Abstract RAG Backend

This is the abstract base class that all RAG backend implementations must inherit from.
It defines the interface contract that both read operations (exposed via MCP) and
write operations (used by ingestion) must implement.

To create a new backend implementation:
1. Inherit from AbstractRagBackend
2. Override _create_config() to return your custom config (optional)
3. Implement all abstract methods
4. Update RAG_BACKEND_CLASS in .env to point to your implementation

See chroma_backend.py for a complete working example.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AbstractRagBackend(ABC):
    """
    Abstract base class for RAG backend implementations.

    All backend implementations must inherit from this class and implement
    all abstract methods. Configuration is available via self.config after
    initialize() is called.
    """

    def __init__(self):
        """Initialize backend (config will be set during initialize())."""
        self.config = None

    def _create_config(self):
        """
        Create configuration object for this backend.

        Override in subclass to return a custom config model that extends
        BackendConfig. Default implementation returns base BackendConfig.

        Returns:
            BackendConfig or subclass: Configuration object
        """
        from config import BackendConfig

        return BackendConfig()

    async def initialize(self):
        """
        Initialize vector database connections and load embedding models.

        This method is called once when the MCP server starts up. It loads
        configuration and then calls _initialize_backend() for implementation-
        specific initialization.

        Subclasses should override _initialize_backend() instead of this method.
        """
        # Load configuration
        self.config = self._create_config()
        logger.info(f"Loaded configuration: {self.config.model_dump()}")

        # Call implementation-specific initialization
        await self._initialize_backend()

    @abstractmethod
    async def _initialize_backend(self):
        """
        Initialize backend-specific resources.

        Override this method to connect to your vector database and load models.
        Configuration is available via self.config.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
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
            Format: [
                {
                    "id": "chunk_id",
                    "content": "text content",
                    "score": 0.95,  # 0.0-1.0, higher is better
                    "metadata": {"source": "doc.pdf", ...}
                },
                ...
            ]
        """
        pass

    @abstractmethod
    async def list_documents(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        List documents in the knowledge base with pagination.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            Paginated list of documents with metadata
            Format: {
                "total": 100,
                "count": 20,
                "offset": 0,
                "documents": [
                    {
                        "id": "doc_id",
                        "source": "filename.pdf",
                        "author": "John Doe",
                        "tags": ["tag1", "tag2"],
                        "created_at": "2024-01-15T10:30:00Z",
                        "chunk_count": 5
                    },
                    ...
                ],
                "has_more": True,
                "next_offset": 20
            }
        """
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.

        Args:
            document_id: ID of document to retrieve

        Returns:
            Document data or None if not found
            Format: {
                "id": "doc_id",
                "source": "filename.pdf",
                "author": "John Doe",
                "tags": ["tag1", "tag2"],
                "created_at": "2024-01-15T10:30:00Z",
                "chunk_count": 5,
                "content": "full reconstructed text..."
            }
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics including document count, chunk count, etc.
            Format: {
                "total_documents": 100,
                "total_chunks": 500,
                "embedding_model": "model-name",
                "vector_dimension": 768,
                "collection_name": "knowledge_base",
                "persist_directory": "/path/to/db"
            }
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up connections and resources."""
        pass

    # ========================================================================
    # Write Methods (for ingestion, not exposed via MCP)
    # ========================================================================

    @abstractmethod
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Dict[str, Any]:
        """
        Add a single document to the knowledge base.

        Args:
            content: Full document text
            metadata: Document metadata (must include 'source')
            chunk_size: Size of chunks (implementation-specific units)
            chunk_overlap: Overlap between chunks

        Returns:
            Document ID and statistics
            Format: {
                "document_id": "unique_id",
                "chunks_created": 5,
                "metadata": {...}
            }
        """
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the knowledge base.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def ingest_directory(
        self,
        directory: str,
        rebuild: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory into the knowledge base.

        This is a bulk operation that efficiently processes multiple documents.
        The default implementation looks for .md files, but you can adapt to
        your needs (PDF, txt, etc.).

        Args:
            directory: Path to directory containing documents
            rebuild: If True, delete existing collection first
            chunk_size: Size of chunks (implementation-specific units)
            chunk_overlap: Overlap between chunks

        Returns:
            Statistics about the ingestion
            Format: {
                "documents_processed": 10,
                "total_chunks": 50,
                "files_processed": ["file1.md", "file2.md", ...]
            }
        """
        pass
