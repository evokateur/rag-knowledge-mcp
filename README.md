# RAG Knowledge Base MCP Server

This is an MCP server I use to connect a RAG knowledge base to Claude Desktop, Claude Code, and [other things](https://github.com/evokateur/cv-joint).

## Setup

With `pytest` tests:

```bash
uv sync --extra dev
```

### Ingesting Documents

Embedding and retrieval are done with Chroma (other types of embedding/retrieval are possible by implementing `AbstractRagBackend`)

Default configuration assumes the KB docs will be at the project root in `./knowledge-base` – which can be a symlink.

What my directory looks like (more or less):

```sh
knowledge-base
├── companies
│   └── frobozz-co.md
│   └── acme.md
├── developers
│   └── wesley-hinkle.md
└── projects
|   ├── magic-api-gateway.md
|   ├── zork-legacy-cms.md
|   ├── torch-saas.md
|   ├── grue-detector.md
|   ├── zorkmid-sdk.md
|   ├── hello-footpad.md
|   ├── anvil.md
└── skills-mapping.md
```

Once the documents are linked up/in place:

```sh
uv run python ingest.py
```

or just

```sh
make embeddings
```

## Client Configuration

### Claude Desktop

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "uv",
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

`uv` may need to be an absolute path, depending on how it's installed.

### Claude Code

<https://code.claude.com/docs/en/mcp>

Installs with `--scope local` by default (`claude` in CWD)

```sh
claude mcp add rag-knowledge -- uv run --directory /absolute/path/to/this/project python rag_knowledge_mcp.py
```

Add `--scope user` for `claude` in any directory:

```sh
claude mcp add --scope user rag-knowledge -- uv run --directory /absolute/path/to/this/project python rag_knowledge_mcp.py
```
