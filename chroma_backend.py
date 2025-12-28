"""
RAG Backend Implementation using Chroma + Sentence Transformers

This backend provides complete RAG functionality with ChromaDB vector database.

Features:
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Semantic search with metadata filtering
- Document reconstruction from chunks
- Bulk directory ingestion with batch embeddings
- Proper async patterns
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field

from config import BackendConfig
from abstract_backend import AbstractRagBackend

logger = logging.getLogger(__name__)


class ChromaConfig(BackendConfig):
    """
    ChromaDB-specific configuration extending base BackendConfig.

    Adds chunking strategy parameters specific to the Chroma implementation.
    """

    chunk_size: int = Field(
        default=500, description="Size of text chunks in characters"
    )

    chunk_overlap: int = Field(
        default=100, description="Overlap between chunks in characters"
    )

    chunk_separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separators for recursive text splitting (paragraph -> sentence -> word)",
    )


class RagBackend(AbstractRagBackend):
    """
    Chroma + Sentence Transformers implementation of RAG backend.

    Configuration is loaded from environment variables into ChromaConfig.
    """

    def __init__(self):
        """Initialize RAG backend."""
        super().__init__()
        self.client = None
        self.collection = None
        self.model = None

    def _create_config(self) -> ChromaConfig:
        """Create ChromaDB-specific configuration."""
        return ChromaConfig()

    async def _initialize_backend(self):
        """Initialize Chroma client and load embedding model."""
        try:
            # Initialize Chroma with persistence
            self.client = chromadb.PersistentClient(
                path=self.config.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "description": "RAG knowledge base for MCP server",
                },
            )

            # Load embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)

            doc_count = self.collection.count()
            logger.info(f"RAG backend initialized with {doc_count} documents")
            logger.info(
                f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize RAG backend: {e}")
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Chroma.

        Args:
            query: Search query text
            top_k: Maximum number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            filters: Optional Chroma metadata filters

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()

            # Query Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []

            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score (1 - normalized_distance)
                    # Cosine distance is in [0, 2], so we normalize to [0, 1]
                    distance = results["distances"][0][i]
                    score = 1.0 - (distance / 2.0)

                    # Apply score threshold
                    if score >= score_threshold:
                        formatted_results.append(
                            {
                                "id": results["ids"][0][i],
                                "content": results["documents"][0][i],
                                "score": round(score, 4),
                                "metadata": results["metadatas"][0][i] or {},
                            }
                        )

            logger.info(
                f"Search completed: {len(formatted_results)} results above threshold"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def list_documents(self, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        List documents with pagination.

        Args:
            limit: Maximum documents to return
            offset: Number of documents to skip

        Returns:
            Paginated document list
        """
        try:
            # Get all documents (we'll group by parent_doc)
            # Note: This is simplified - for production, you'd want a separate documents table
            results = self.collection.get(
                include=["metadatas"],
                limit=limit * 10,  # Get extra to account for chunks
            )

            # Group chunks by parent document
            docs_map = {}
            for metadata in results["metadatas"]:
                parent_doc = metadata.get("parent_doc", "unknown")
                if parent_doc not in docs_map:
                    docs_map[parent_doc] = {
                        "id": parent_doc,
                        "source": metadata.get("source", "Unknown"),
                        "author": metadata.get("author"),
                        "tags": metadata.get("tags", []),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "chunk_count": metadata.get("chunk_count", 0),
                    }

            # Convert to list and apply pagination
            all_docs = list(docs_map.values())
            total = len(all_docs)

            # Sort by created_at descending
            all_docs.sort(key=lambda x: x["created_at"], reverse=True)

            # Apply pagination
            paginated_docs = all_docs[offset : offset + limit]

            has_more = (offset + len(paginated_docs)) < total
            next_offset = offset + len(paginated_docs) if has_more else None

            return {
                "total": total,
                "count": len(paginated_docs),
                "offset": offset,
                "documents": paginated_docs,
                "has_more": has_more,
                "next_offset": next_offset,
            }

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.

        Args:
            document_id: Document ID to retrieve

        Returns:
            Document data or None if not found
        """
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"parent_doc": document_id}, include=["documents", "metadatas"]
            )

            if not results["ids"]:
                return None

            # Sort chunks by index
            chunks_data = list(
                zip(results["ids"], results["documents"], results["metadatas"])
            )
            chunks_data.sort(key=lambda x: x[2].get("chunk_index", 0))

            # Reconstruct full content
            full_content = " ".join([doc for _, doc, _ in chunks_data])

            # Get metadata from first chunk
            metadata = chunks_data[0][2]

            return {
                "id": document_id,
                "source": metadata.get("source", "Unknown"),
                "author": metadata.get("author"),
                "tags": metadata.get("tags", []),
                "created_at": metadata.get("created_at", "Unknown"),
                "chunk_count": len(chunks_data),
                "content": full_content,
            }

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics dictionary
        """
        try:
            # Get total chunk count
            total_chunks = self.collection.count()

            # Count unique documents
            results = self.collection.get(include=["metadatas"], limit=total_chunks)

            unique_docs = set()
            for metadata in results["metadatas"]:
                parent_doc = metadata.get("parent_doc")
                if parent_doc:
                    unique_docs.add(parent_doc)

            return {
                "total_documents": len(unique_docs),
                "total_chunks": total_chunks,
                "embedding_model": self.config.embedding_model,
                "vector_dimension": self.model.get_sentence_embedding_dimension(),
                "collection_name": self.config.collection,
                "persist_directory": self.config.persist_dir,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        logger.info("Closing RAG backend...")
        # Chroma client handles cleanup automatically
        self.client = None
        self.collection = None
        self.model = None

    # ========================================================================
    # Write Methods (for ingestion, not exposed via MCP)
    # ========================================================================

    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Dict[str, Any]:
        """
        Add a single document to the knowledge base.

        Args:
            content: Full document text
            metadata: Document metadata (must include 'source')
            chunk_size: Size of chunks in characters (defaults to config.chunk_size)
            chunk_overlap: Overlap between chunks in characters (defaults to config.chunk_overlap)

        Returns:
            Document ID and statistics
        """
        try:
            # Use config defaults if not specified
            chunk_size = chunk_size or self.config.chunk_size
            chunk_overlap = chunk_overlap or self.config.chunk_overlap

            # Generate unique document ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            source_slug = (
                metadata.get("source", "doc").replace("/", "_").replace(".", "_")[:50]
            )
            doc_id = f"{source_slug}_{timestamp}"

            # Chunk the document using LangChain
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=self.config.chunk_separators,
            )
            chunks = text_splitter.split_text(content)

            logger.info(f"Created {len(chunks)} chunks for document {doc_id}")

            # Generate embeddings
            embeddings = self.model.encode(chunks, convert_to_tensor=False).tolist()

            # Create chunk IDs and metadata
            chunk_ids = [f"{doc_id}_chunk_{i:04d}" for i in range(len(chunks))]
            chunk_metadatas = [
                {
                    **metadata,
                    "chunk_index": i,
                    "parent_doc": doc_id,
                    "chunk_count": len(chunks),
                }
                for i in range(len(chunks))
            ]

            # Add to Chroma
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadatas,
            )

            logger.info(
                f"Document {doc_id} added successfully with {len(chunks)} chunks"
            )

            return {
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the knowledge base.

        Args:
            document_id: ID of document to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            # Query for all chunks belonging to this document
            results = self.collection.get(where={"parent_doc": document_id}, include=[])

            if not results["ids"]:
                logger.warning(f"Document {document_id} not found")
                return False

            # Delete all chunks
            self.collection.delete(ids=results["ids"])

            logger.info(
                f"Deleted document {document_id} ({len(results['ids'])} chunks)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    async def ingest_directory(
        self,
        directory: str,
        rebuild: bool = False,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Dict[str, Any]:
        """
        Ingest all markdown files from a directory into the knowledge base.

        This is a bulk operation that efficiently processes multiple documents.

        Args:
            directory: Path to directory containing markdown files
            rebuild: If True, delete existing collection first
            chunk_size: Size of chunks in characters (defaults to config.chunk_size)
            chunk_overlap: Overlap between chunks in characters (defaults to config.chunk_overlap)

        Returns:
            Statistics about the ingestion
        """
        try:
            # Use config defaults if not specified
            chunk_size = chunk_size or self.config.chunk_size
            chunk_overlap = chunk_overlap or self.config.chunk_overlap

            directory_path = Path(directory)

            if not directory_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

            # Handle rebuild
            if rebuild:
                try:
                    self.client.delete_collection(name=self.config.collection)
                    logger.info(
                        f"Deleted existing collection: {self.config.collection}"
                    )

                    # Recreate collection
                    self.collection = self.client.get_or_create_collection(
                        name=self.config.collection,
                        metadata={
                            "hnsw:space": "cosine",
                            "description": "RAG knowledge base for MCP server",
                        },
                    )
                except Exception:
                    logger.debug(
                        f"No existing collection to delete: {self.config.collection}"
                    )

            # Find all markdown files
            markdown_files = list(directory_path.rglob("*.md"))

            if not markdown_files:
                logger.warning(f"No markdown files found in {directory}")
                return {
                    "documents_processed": 0,
                    "total_chunks": 0,
                    "files_processed": [],
                }

            logger.info(f"Found {len(markdown_files)} markdown files")

            # Prepare chunker
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=self.config.chunk_separators,
            )

            # Process all documents and collect chunks
            all_chunk_ids = []
            all_chunks = []
            all_metadatas = []
            files_processed = []

            for file_path in markdown_files:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    relative_path = file_path.relative_to(directory_path)

                    # Generate document ID
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    source_slug = (
                        str(relative_path).replace("/", "_").replace(".", "_")[:50]
                    )
                    doc_id = f"{source_slug}_{timestamp}"

                    # Chunk document
                    chunks = text_splitter.split_text(content)

                    # Create chunk metadata
                    for i, chunk_text in enumerate(chunks):
                        all_chunk_ids.append(f"{doc_id}_chunk_{i:04d}")
                        all_chunks.append(chunk_text)
                        all_metadatas.append(
                            {
                                "source": str(relative_path),
                                "parent_doc": doc_id,
                                "chunk_index": i,
                                "chunk_count": len(chunks),
                                "created_at": datetime.now().isoformat(),
                            }
                        )

                    files_processed.append(str(relative_path))
                    logger.debug(f"Processed {relative_path}: {len(chunks)} chunks")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

            # Batch generate embeddings for ALL chunks
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.model.encode(
                all_chunks,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=32,
            ).tolist()

            # Store all chunks in one operation
            logger.info("Storing embeddings in ChromaDB...")
            self.collection.add(
                ids=all_chunk_ids,
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
            )

            logger.info(
                f"✓ Ingestion complete: {len(files_processed)} documents, {len(all_chunks)} chunks"
            )

            return {
                "documents_processed": len(files_processed),
                "total_chunks": len(all_chunks),
                "files_processed": files_processed,
            }

        except Exception as e:
            logger.error(f"Failed to ingest directory: {e}")
            raise
