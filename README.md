# RAG Knowledge Base MCP Server

This is an MCP server I use to connect my knowledge base to Claude

## Setup

```bash
uv sync --extra dev # installs pytest tests
```

### Ingesting Documents

Default configuration assumes the knowledge base docs are in `knowledge-base` in the project root. 
Since this is git-ignored one can create a symlink named `knowledge-base` pointing elsewhere.

The directory can also be configured by copying `.env.example` to `.env` and...

```
RAG_KNOWLEDGE_DIR=./knowledge-base # <--- changing this to something else
```

Once the documents are in place:

```bash
uv run python ingest.py
```

## Client Configuration

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
absolute path returned from `which uv`

### Claude Code

<https://code.claude.com/docs/en/mcp>

I created a wrapper script (`~/.bin/rag-knowledge-mcp`) then ran

```bash
claude mcp add --transport stdio rag-knowledge ~/.bin/rag-knowledge-mcp
```

The contents of `wrapper-example.sh` can be copied and modified

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/absolute/path/to/this/project"
UV="/absolute/path/to/uv"

export LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$PROJECT_DIR"
exec "$UV" run python rag_knowledge_mcp.py
```
