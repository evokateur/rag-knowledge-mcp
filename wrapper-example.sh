#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/code/rag-knowledge-mcp"
UV="$(which uv)"

echo "$PROJECT_DIR"
echo "$UV"

export LOG_LEVEL="${LOG_LEVEL:-INFO}"

cd "$PROJECT_DIR"
exec "$UV" run python rag_knowledge_mcp.py
