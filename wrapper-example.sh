#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/absolute/path/to/this/project"
UV="/absolute/path/to/uv"

export LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$PROJECT_DIR"
exec "$UV" run python rag_knowledge_mcp.py
