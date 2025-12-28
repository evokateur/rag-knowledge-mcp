# RAG Knowledge Base MCP Server

A Model Context Protocol (MCP) server that provides semantic search capabilities over a RAG (Retrieval-Augmented Generation) knowledge base. This server enables Claude and other LLMs to query and retrieve documents using vector similarity search with a read-only interface.

## Overview

This MCP server provides a clean interface for RAG functionality while keeping the actual implementation (chunking strategies, embedding models, vector databases) completely pluggable. You can integrate any RAG stack you prefer.

### Features

- **Semantic Search**: Query your knowledge base using natural language
- **Read-Only Interface**: Safe, non-destructive access to your knowledge base
- **Flexible Formats**: Returns results in Markdown or JSON format
- **Pagination**: Efficient handling of large document collections
- **Pluggable Backend**: Easy integration with your existing RAG infrastructure
- **Type-Safe Configuration**: Pydantic models with validation and IDE autocomplete

### Tools Provided

1. **rag_search_knowledge** - Semantic search over knowledge base
2. **rag_list_documents** - Browse available documents
3. **rag_get_document** - Retrieve specific documents by ID
4. **rag_get_stats** - View knowledge base statistics

Document ingestion is handled separately via `ingest_knowledge.py` script.

## Architecture

```
                    READ-ONLY MCP INTERFACE
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (Claude)                      │
│                 (Desktop App / Claude Code)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ stdio transport
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastMCP Server (rag_knowledge_mcp.py)           │
│  - 4 read-only tools: search, list, get, stats              │
│  - Request/response formatting (Markdown/JSON)               │
│  - Error handling & logging                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ read-only access
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            AbstractRagBackend (ABC Interface)                │
│  - Pydantic configuration (BackendConfig hierarchy)          │
│  - Read methods: search, list, get, stats                    │
│  - Write methods: add, delete, ingest (not exposed to MCP)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         ChromaBackend (chroma_backend.py)                    │
│  - ChromaConfig: chunking strategy, embedding model          │
│  - ChromaDB PersistentClient (cosine similarity)             │
│  - Sentence Transformers embeddings                          │
│  - LangChain RecursiveCharacterTextSplitter                  │
└─────────────────────────────────────────────────────────────┘

                    INGESTION WORKFLOW
┌─────────────────────────────────────────────────────────────┐
│           ingest_knowledge.py (CLI script)                   │
│  - Reads markdown files from knowledge-base/                 │
│  - Calls backend.ingest_directory()                          │
│  - Batch embedding generation for efficiency                 │
│  - --rebuild flag to recreate collection                     │
└────────────────────────┬────────────────────────────────────┘
                         │ write access
                         ▼
              (Uses same AbstractRagBackend)
```

## Installation

### 1. Install Dependencies

```bash
# Install core dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev
```

### 2. Add Your RAG Dependencies

The default ChromaDB backend is already configured. Edit `pyproject.toml` if you want to add dependencies for your chosen stack. For example:

```toml
dependencies = [
    "mcp>=1.1.0",
    "chromadb>=0.4.0",
    "sentence-transformers>=2.3.0",
    # Add your stack here
    # "pinecone-client>=3.0.0",
    # "openai>=1.0.0",
]
```

Then run `uv sync` to install.

### 3. Implement Your RAG Backend (Optional)

The default ChromaDB backend (`chroma_backend.py`) is ready to use. To create a custom backend:

**Step 1: Create a Pydantic configuration class**

```python
from config import BackendConfig
from pydantic import Field

class MyBackendConfig(BackendConfig):
    """Extend base config with backend-specific settings."""
    # Inherits: knowledge_dir, persist_dir, collection, embedding_model
    api_key: str = Field(default_factory=lambda: os.getenv("MY_API_KEY", ""))
    custom_param: int = Field(default=42)
```

**Step 2: Inherit from AbstractRagBackend**

```python
from abstract_backend import AbstractRagBackend

class MyBackend(AbstractRagBackend):
    def _create_config(self) -> MyBackendConfig:
        """Return your config class."""
        return MyBackendConfig()

    async def _initialize_backend(self):
        """Initialize your vector database and embedding model."""
        # Access config via self.config
        self.client = MyVectorDB(api_key=self.config.api_key)
        self.model = MyEmbeddingModel(self.config.embedding_model)

    async def search(self, query, top_k, score_threshold, filters):
        # 1. Encode query with embedding model
        # 2. Query vector database
        # 3. Return ranked results
        pass

    # ... implement all other abstract methods
```

**Step 3: Configure dependency injection**

Set `RAG_BACKEND_CLASS` in your `.env` file:

```bash
RAG_BACKEND_CLASS=my_backend.MyBackend
```

See `abstract_backend.py` for the complete interface contract and `chroma_backend.py` for a full working example.

## Configuration

### Configure Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/your/project",
        "python",
        "rag_knowledge_mcp.py"
      ],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Configure Claude Code

Create or update `~/.config/claude-code/mcp_config.json`:

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/your/project",
        "python",
        "rag_knowledge_mcp.py"
      ],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Environment Variables (Optional)

You can configure your RAG backend using environment variables:

```bash
export VECTOR_DB_URL="http://localhost:6333"  # Qdrant
export EMBEDDING_MODEL="all-MiniLM-L6-v2"     # Sentence Transformers
export OPENAI_API_KEY="sk-..."                # OpenAI embeddings
```

## Usage Examples

### Ingesting Documents (One-Time Setup)

Before using the MCP server, populate your knowledge base:

```bash
# Place markdown files in ./knowledge-base/ directory
# Then run ingestion script

python ingest_knowledge.py

# Or rebuild from scratch
python ingest_knowledge.py --rebuild
```

The ingestion script:
- Reads all `.md` files from `RAG_KNOWLEDGE_DIR` (default: `./knowledge-base/`)
- Chunks documents using RecursiveCharacterTextSplitter
- Generates embeddings in batches for efficiency
- Stores in vector database at `RAG_PERSIST_DIR` (default: `./chroma_db/`)

### Searching the Knowledge Base (Via MCP)

Once ingested, Claude can search using natural language:

```python
# Claude can now use these tools directly:
"""
Search for information about machine learning optimization techniques
"""

# Calls: rag_search_knowledge(
#   query="machine learning optimization techniques",
#   top_k=5,
#   score_threshold=0.7
# )
```

### Listing Documents

```python
# Browse available documents
"""
Show me all documents in the knowledge base
"""

# Calls: rag_list_documents(limit=20, offset=0)
```

### Retrieving Specific Documents

```python
# Get full document content
"""
Retrieve the document about neural networks
"""

# Claude will first search/list to find the document ID, then:
# Calls: rag_get_document(document_id="...")
```

## Implementation Guide

### Example: Chroma + Sentence Transformers (Default Backend)

The included `chroma_backend.py` provides a complete working implementation:

```python
import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import Field
from config import BackendConfig
from abstract_backend import AbstractRagBackend

class ChromaConfig(BackendConfig):
    """ChromaDB-specific configuration extending base BackendConfig."""
    chunk_size: int = Field(default=500, description="Size of text chunks in characters")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in characters")
    chunk_separators: list[str] = Field(
        default=["\n\n", "\n", ". ", " ", ""],
        description="Separators for recursive text splitting"
    )

class RagBackend(AbstractRagBackend):
    def __init__(self):
        super().__init__()
        self.client = None
        self.collection = None
        self.model = None

    def _create_config(self) -> ChromaConfig:
        """Create ChromaDB-specific configuration."""
        return ChromaConfig()

    async def _initialize_backend(self):
        # Initialize Chroma client using self.config
        self.client = chromadb.PersistentClient(path=self.config.persist_dir)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection,
            metadata={"hnsw:space": "cosine"}
        )

        # Load embedding model
        self.model = SentenceTransformer(self.config.embedding_model)
        logger.info(f"RAG backend initialized with {self.collection.count()} documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Encode query
        query_embedding = self.model.encode(query).tolist()
        
        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]  # Convert distance to similarity
            if score >= score_threshold:
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "score": score,
                    "metadata": results['metadatas'][0][i]
                })
        
        return formatted_results
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> Dict[str, Any]:
        # Use config defaults if not specified
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap

        # Chunk using LangChain RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.config.chunk_separators,
        )
        chunks = text_splitter.split_text(content)

        # Generate embeddings
        embeddings = self.model.encode(chunks, convert_to_tensor=False).tolist()

        # Generate IDs
        doc_id = f"{metadata.get('source', 'doc')}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        chunk_ids = [f"{doc_id}_chunk_{i:04d}" for i in range(len(chunks))]

        # Add chunk metadata
        chunk_metadatas = [
            {**metadata, "chunk_index": i, "parent_doc": doc_id, "chunk_count": len(chunks)}
            for i in range(len(chunks))
        ]

        # Add to Chroma
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas
        )

        return {
            "document_id": doc_id,
            "chunks_created": len(chunks),
            "metadata": metadata
        }

    # See chroma_backend.py for complete implementation...
```

### Example: Pinecone + OpenAI

Here's how you'd implement a Pinecone backend with the Pydantic configuration pattern:

```python
import pinecone
from openai import AsyncOpenAI
from pydantic import Field
from config import BackendConfig
from abstract_backend import AbstractRagBackend

class PineconeConfig(BackendConfig):
    """Pinecone-specific configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    environment: str = Field(default_factory=lambda: os.getenv("PINECONE_ENV", "us-west1-gcp"))
    index_name: str = Field(default="knowledge-base")

class PineconeBackend(AbstractRagBackend):
    def _create_config(self) -> PineconeConfig:
        return PineconeConfig()

    async def _initialize_backend(self):
        # Initialize Pinecone using self.config
        pinecone.init(
            api_key=self.config.api_key,
            environment=self.config.environment
        )

        # Connect to index
        self.index = pinecone.Index(self.config.index_name)

        # Initialize OpenAI client
        self.openai = AsyncOpenAI()

    async def search(self, query: str, top_k: int = 5, **kwargs):
        # Generate embedding with OpenAI
        response = await self.openai.embeddings.create(
            model=self.config.embedding_model,  # Use config
            input=query
        )
        query_embedding = response.data[0].embedding

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format and return results...

    # Implement all other abstract methods...
```

Then set in `.env`:
```bash
RAG_BACKEND_CLASS=pinecone_backend.PineconeBackend
PINECONE_API_KEY=your-api-key
PINECONE_ENV=us-west1-gcp
```

## Testing

### Run Automated Tests

The project includes comprehensive pytest tests:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_abc.py              # Test abstract base class
pytest test_rag_backend.py      # Test backend implementation

# Run specific test
pytest test_rag_backend.py::test_semantic_search

# Run with coverage (if pytest-cov installed)
pytest --cov=. --cov-report=html
```

Tests automatically use separate test database (`./test_chroma_db/`) via `TEST_*` environment variables.

### Test Server Startup

```bash
# Verify server can start
uv run python rag_knowledge_mcp.py --help
```

### Test with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python rag_knowledge_mcp.py
```

This opens a web interface where you can test all tools interactively.

### Test Individual Tools

Use the inspector to call tools manually:

```json
// Test search
{
  "query": "machine learning",
  "top_k": 3,
  "score_threshold": 0.5
}

// Test list documents
{
  "limit": 10,
  "offset": 0
}

// Test get document
{
  "document_id": "ml_basics_md_20241227_123456_789012"
}
```

## Best Practices

### Chunking Strategies

The ChromaDB backend uses LangChain's RecursiveCharacterTextSplitter by default:

- **Recursive chunking** (default): Splits by paragraphs → sentences → words for natural boundaries
- **Character-based**: Uses character count (not tokens) for consistent sizing
- **Configurable separators**: Customize via `ChromaConfig.chunk_separators`
- **Alternative approaches**: Semantic chunking (use embeddings), fixed-size, sentence-based
- **Consider your use case**: Technical docs benefit from smaller chunks, narratives from larger

### Embedding Models

- **Lightweight**: `all-MiniLM-L6-v2` (384 dim) - fast, good for general use
- **Balanced**: `all-mpnet-base-v2` (768 dim) - better quality
- **Domain-specific**: Fine-tune on your data
- **Commercial**: OpenAI `text-embedding-3-small/large`, Cohere

### Vector Databases

- **Chroma**: Easy to use, good for prototypes, local-first
- **Pinecone**: Managed, scalable, good for production
- **Qdrant**: Fast, feature-rich, self-hostable
- **FAISS**: Facebook's library, very fast, lower-level

### Performance Tips

1. **Batch ingestion**: Use `ingest_directory()` for bulk operations (generates embeddings in batches)
2. **Async all the way**: Backend uses async/await for I/O operations
3. **Connection pooling**: MCP server uses FastMCP lifespan to reuse connections
4. **Optimize chunking**: Tune `chunk_size` and `chunk_overlap` for your use case
5. **Index optimization**: Tune your vector DB parameters (e.g., HNSW settings)
6. **Embedding caching**: Consider caching embeddings for frequently queried terms

## Troubleshooting

### Server won't start

```bash
# Check Python version (3.10+)
python --version

# Verify all dependencies installed
uv sync

# Check for import errors
uv run python -c "from mcp.server.fastmcp import FastMCP"
```

### Claude can't connect

1. Check config file path and JSON syntax
2. Verify Python path in config
3. Check Claude logs (Help > Show Logs)
4. Test server directly: `python rag_knowledge_mcp.py`

### Search returns no results

1. Verify documents are indexed: Use `rag_list_documents` tool or check stats
2. Run ingestion if knowledge base is empty: `python ingest_knowledge.py`
3. Check score threshold (try 0.0 first to see all results)
4. Test embedding model separately
5. Verify vector DB connection and collection name

### Memory issues with large documents

1. Adjust chunking in `.env`: Set smaller `RAG_CHUNK_SIZE` (e.g., 300)
2. Use `--rebuild` flag with ingestion to recreate with new settings
3. Process documents incrementally (add a few at a time)
4. Consider document filtering/sampling before ingestion

## Advanced Features

### Adding Reranking

Improve search quality by reranking initial results:

```python
from sentence_transformers import CrossEncoder

async def search(self, query, top_k=5, ...):
    # Initial retrieval (get more candidates)
    candidates = await self._initial_search(query, top_k * 3)
    
    # Rerank with cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc['content']] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by reranker scores and return top_k
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_k]]
```

### Adding Metadata Filtering

```python
# Allow filtering by source, date, tags, etc.
await rag_backend.search(
    query="machine learning",
    filters={
        "tags": {"$contains": "neural-networks"},
        "created_at": {"$gte": "2024-01-01"}
    }
)
```

### Hybrid Search (Vector + Keyword)

Combine semantic search with traditional keyword search:

```python
async def search(self, query, top_k=5, hybrid_alpha=0.5):
    # Vector search
    vector_results = await self._vector_search(query, top_k * 2)
    
    # Keyword search (BM25)
    keyword_results = await self._keyword_search(query, top_k * 2)
    
    # Combine scores with weighted average
    combined = self._merge_results(
        vector_results,
        keyword_results,
        alpha=hybrid_alpha
    )
    
    return combined[:top_k]
```

## Extending the Server

This server is designed to be customized for your specific needs. Common enhancements:

**Backend Extensions:**
- [ ] Alternative vector databases (Pinecone, Qdrant, FAISS)
- [ ] Different embedding models (OpenAI, Cohere, custom fine-tuned)
- [ ] Hybrid search (vector + keyword/BM25)
- [ ] Reranking with cross-encoders
- [ ] Metadata filtering and faceted search

**Ingestion Pipeline:**
- [ ] Document preprocessing (PDF, DOCX, HTML parsing)
- [ ] Metadata extraction (title, author, date)
- [ ] Multi-modal support (images, tables)
- [ ] Incremental updates (track document changes)
- [ ] Background ingestion jobs

**Query Enhancement:**
- [ ] Query expansion and refinement
- [ ] Caching layer for frequent queries
- [ ] Monitoring and analytics
- [ ] A/B testing different retrieval strategies

All enhancements can be implemented in your backend class while keeping the MCP interface unchanged.

## License

MIT License - feel free to use and modify for your projects.

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [FastMCP GitHub](https://github.com/modelcontextprotocol/python-sdk)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Pinecone Documentation](https://docs.pinecone.io/)

## Support

For issues with:

- **MCP protocol**: See [MCP docs](https://modelcontextprotocol.io/)
- **This server**: File an issue or adapt the code as needed
- **Backend implementation**: See `abstract_backend.py` for interface contract
- **ChromaDB**: See [Chroma docs](https://docs.trychroma.com/)
- **Pydantic configuration**: See [Pydantic docs](https://docs.pydantic.dev/)
- **Your custom RAG implementation**: Consult your vector DB/embedding model docs
