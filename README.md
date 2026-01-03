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
        "/absolute/path/to/this/project",
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

Note: Claude Desktop did not seem to have `uv` in its path so I used the
path returned from `which uv`

### Claude Code

<https://code.claude.com/docs/en/mcp>

I created a wrapper script (`~/.bin/rag-knowledge-mcp`) then ran

```bash
claude mcp add --transport stdio rag-knowledge ~/.bin/rag-knowledge-mcp
```

This is the contents of `wrapper-example.sh`, which can be copied and modified:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/absolute/path/to/this/project"
UV="/absolute/path/to/uv"

export LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$PROJECT_DIR"
exec "$UV" run python rag_knowledge_mcp.py
```
