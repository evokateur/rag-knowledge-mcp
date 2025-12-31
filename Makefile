.PHONY: embeddings test test-e2e

embeddings:
	uv run python ingest_knowledge.py

test:
	uv run pytest

test-e2e:
	uv run pytest -m e2e 

