.PHONY: embeddings test test-e2e

embeddings:
	uv run python ingest.py

test:
	uv run pytest

test-e2e:
	uv run pytest -m e2e 

