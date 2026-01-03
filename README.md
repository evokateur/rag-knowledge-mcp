# RAG Knowledge Base MCP Server

This is an MCP server I use to connect my knowledge base to Claude

## Setup

```bash
uv sync --extra dev # installs pytest tests
```

### Ingesting Documents

Default configuration assumes knowledge base is in `knowledge-base` in the project root. Since this is git-ignored I create a symlink to a directory with my knowledge base markdown files. The directory can also be configured in `.env` (see [`.env.example`](https://github.com/evokateur/rag-knowledge-mcp/blob/main/.env.example))

Once the documents are in place:

```bash
uv run python ingest_knowledge.py 
```

## Configuration

### Claude Desktop

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "/absolute/path/to/uv",
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

Note: Claude Desktop did not seem to have `uv` in its path so I used the absolute path from `which uv`

### Claude Code

https://code.claude.com/docs/en/mcp

What I did:

```bash
claude mcp add-from-claude-desktop
```


