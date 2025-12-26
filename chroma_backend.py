"""
Example RAG Backend Implementation using Chroma + Sentence Transformers

This is a complete, working implementation that you can use as-is or
adapt for your specific needs. It demonstrates:

- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Simple word-based chunking
- Metadata filtering
- Proper async patterns

To use this implementation:
1. uv sync
2. Replace the RAGBackend class in rag_knowledge_mcp.py with this one
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RAGBackend:
    """
    Chroma + Sentence Transformers implementation of RAG backend.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG backend configuration.
        
        Args:
            persist_directory: Path to store Chroma database
            collection_name: Name of the Chroma collection
            embedding_model: Sentence Transformers model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        self.client = None
        self.collection = None
        self.model = None
    
    async def initialize(self):
        """Initialize Chroma client and load embedding model."""
        try:
            # Initialize Chroma with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "description": "RAG knowledge base for MCP server"
                }
            )
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            
            doc_count = self.collection.count()
            logger.info(f"RAG backend initialized with {doc_count} documents")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG backend: {e}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
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
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Convert distance to similarity score (1 - normalized_distance)
                    # Cosine distance is in [0, 2], so we normalize to [0, 1]
                    distance = results['distances'][0][i]
                    score = 1.0 - (distance / 2.0)
                    
                    # Apply score threshold
                    if score >= score_threshold:
                        formatted_results.append({
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "score": round(score, 4),
                            "metadata": results['metadatas'][0][i] or {}
                        })
            
            logger.info(f"Search completed: {len(formatted_results)} results above threshold")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Add document to Chroma with chunking and embedding.
        
        Args:
            content: Full document text
            metadata: Document metadata
            chunk_size: Size of chunks in words
            chunk_overlap: Overlap between chunks in words
            
        Returns:
            Document ID and statistics
        """
        try:
            # Generate unique document ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            source_slug = metadata.get('source', 'doc').replace(' ', '_').replace('/', '_')[:50]
            doc_id = f"{source_slug}_{timestamp}"
            
            # Chunk the document
            chunks = self._chunk_text(content, chunk_size, chunk_overlap)
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
                    "chunk_count": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to Chroma
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Document {doc_id} added successfully with {len(chunks)} chunks")
            
            return {
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document and all its chunks from Chroma.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Query for all chunks belonging to this document
            results = self.collection.get(
                where={"parent_doc": document_id},
                include=[]
            )
            
            if not results['ids']:
                logger.warning(f"Document {document_id} not found")
                return False
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            logger.info(f"Deleted document {document_id} ({len(results['ids'])} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise
    
    async def list_documents(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
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
                limit=limit * 10  # Get extra to account for chunks
            )
            
            # Group chunks by parent document
            docs_map = {}
            for metadata in results['metadatas']:
                parent_doc = metadata.get('parent_doc', 'unknown')
                if parent_doc not in docs_map:
                    docs_map[parent_doc] = {
                        'id': parent_doc,
                        'source': metadata.get('source', 'Unknown'),
                        'author': metadata.get('author'),
                        'tags': metadata.get('tags', []),
                        'created_at': metadata.get('created_at', 'Unknown'),
                        'chunk_count': metadata.get('chunk_count', 0)
                    }
            
            # Convert to list and apply pagination
            all_docs = list(docs_map.values())
            total = len(all_docs)
            
            # Sort by created_at descending
            all_docs.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Apply pagination
            paginated_docs = all_docs[offset:offset + limit]
            
            has_more = (offset + len(paginated_docs)) < total
            next_offset = offset + len(paginated_docs) if has_more else None
            
            return {
                "total": total,
                "count": len(paginated_docs),
                "offset": offset,
                "documents": paginated_docs,
                "has_more": has_more,
                "next_offset": next_offset
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
                where={"parent_doc": document_id},
                include=["documents", "metadatas"]
            )
            
            if not results['ids']:
                return None
            
            # Sort chunks by index
            chunks_data = list(zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            ))
            chunks_data.sort(key=lambda x: x[2].get('chunk_index', 0))
            
            # Reconstruct full content
            full_content = ' '.join([doc for _, doc, _ in chunks_data])
            
            # Get metadata from first chunk
            metadata = chunks_data[0][2]
            
            return {
                'id': document_id,
                'source': metadata.get('source', 'Unknown'),
                'author': metadata.get('author'),
                'tags': metadata.get('tags', []),
                'created_at': metadata.get('created_at', 'Unknown'),
                'chunk_count': len(chunks_data),
                'content': full_content
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
            results = self.collection.get(
                include=["metadatas"],
                limit=total_chunks
            )
            
            unique_docs = set()
            for metadata in results['metadatas']:
                parent_doc = metadata.get('parent_doc')
                if parent_doc:
                    unique_docs.add(parent_doc)
            
            return {
                "total_documents": len(unique_docs),
                "total_chunks": total_chunks,
                "embedding_model": self.embedding_model_name,
                "vector_dimension": self.model.get_sentence_embedding_dimension(),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
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
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Simple word-based text chunking.
        
        For production, consider using:
        - LangChain's RecursiveCharacterTextSplitter
        - Semantic chunking based on embeddings
        - Document-structure-aware chunking (headers, paragraphs)
        
        Args:
            text: Text to chunk
            chunk_size: Size in words
            chunk_overlap: Overlap in words
            
        Returns:
            List of text chunks
        """
        # Split into words
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        step_size = chunk_size - chunk_overlap
        
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:  # Don't add empty chunks
                chunk = ' '.join(chunk_words)
                chunks.append(chunk)
        
        return chunks
