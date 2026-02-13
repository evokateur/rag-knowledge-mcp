#!/usr/bin/env python3
"""Test configuration loading from environment variables."""

import pytest

from config import BackendConfig, _get_config, PROJECT_ROOT
from chroma_backend import ChromaConfig


# ============================================================================
# Default values (when no environment variables are set)
# ============================================================================

BACKEND_DEFAULTS = {
    "knowledge_dir": str((PROJECT_ROOT / "knowledge-base").absolute()),
    "persist_dir": str((PROJECT_ROOT / "chroma_db").absolute()),
    "collection": "knowledge_base",
    "embedding_model": "all-MiniLM-L6-v2",
}

CHROMA_DEFAULTS = {
    **BACKEND_DEFAULTS,
    "chunk_size": 500,
    "chunk_overlap": 100,
    "chunk_separators": ["\n\n", "\n", ". ", " ", ""],
}


# ============================================================================
# _get_config
# ============================================================================


def test_get_config_returns_default(monkeypatch):
    """_get_config returns the default when the env var is unset."""
    monkeypatch.delenv("RAG_COLLECTION", raising=False)
    monkeypatch.delenv("TEST_RAG_COLLECTION", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert _get_config("RAG_COLLECTION", "knowledge_base") == "knowledge_base"


def test_get_config_reads_env_var(monkeypatch):
    """_get_config returns the env var value when set."""
    monkeypatch.setenv("RAG_COLLECTION", "custom_collection")
    monkeypatch.delenv("TEST_RAG_COLLECTION", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert _get_config("RAG_COLLECTION", "knowledge_base") == "custom_collection"


def test_get_config_prefers_test_prefix_in_test_mode(monkeypatch):
    """_get_config prefers TEST_ prefixed var when running under pytest."""
    monkeypatch.setenv("RAG_COLLECTION", "production")
    monkeypatch.setenv("TEST_RAG_COLLECTION", "test_collection")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_config.py::test_example")

    assert _get_config("RAG_COLLECTION", "knowledge_base") == "test_collection"


def test_get_config_falls_back_to_unprefixed_in_test_mode(monkeypatch):
    """_get_config falls back to unprefixed var when TEST_ is not set."""
    monkeypatch.setenv("RAG_COLLECTION", "production")
    monkeypatch.delenv("TEST_RAG_COLLECTION", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_config.py::test_example")

    assert _get_config("RAG_COLLECTION", "knowledge_base") == "production"


# ============================================================================
# BackendConfig defaults
# ============================================================================


def test_backend_config_defaults(monkeypatch):
    """BackendConfig fields match expected defaults when no env vars are set."""
    for key in ["RAG_KNOWLEDGE_DIR", "RAG_PERSIST_DIR", "RAG_COLLECTION", "RAG_EMBEDDING_MODEL"]:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(f"TEST_{key}", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = BackendConfig()

    assert config.knowledge_dir == BACKEND_DEFAULTS["knowledge_dir"]
    assert config.persist_dir == BACKEND_DEFAULTS["persist_dir"]
    assert config.collection == BACKEND_DEFAULTS["collection"]
    assert config.embedding_model == BACKEND_DEFAULTS["embedding_model"]


def test_backend_config_from_env(monkeypatch):
    """BackendConfig fields are overridden by environment variables."""
    monkeypatch.setenv("RAG_COLLECTION", "custom_collection")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "all-mpnet-base-v2")
    monkeypatch.delenv("TEST_RAG_COLLECTION", raising=False)
    monkeypatch.delenv("TEST_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = BackendConfig()

    assert config.collection == "custom_collection"
    assert config.embedding_model == "all-mpnet-base-v2"


# ============================================================================
# ChromaConfig defaults (includes chunking strategy)
# ============================================================================


def test_chroma_config_defaults(monkeypatch):
    """ChromaConfig fields match expected defaults when no env vars are set."""
    all_keys = [
        "RAG_KNOWLEDGE_DIR", "RAG_PERSIST_DIR", "RAG_COLLECTION",
        "RAG_EMBEDDING_MODEL", "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP",
        "RAG_CHUNK_SEPARATORS",
    ]
    for key in all_keys:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(f"TEST_{key}", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = ChromaConfig()

    assert config.chunk_size == CHROMA_DEFAULTS["chunk_size"]
    assert config.chunk_overlap == CHROMA_DEFAULTS["chunk_overlap"]
    assert config.chunk_separators == CHROMA_DEFAULTS["chunk_separators"]
    # Inherited base fields still work
    assert config.collection == CHROMA_DEFAULTS["collection"]


def test_chroma_config_from_env(monkeypatch):
    """ChromaConfig chunking fields are overridden by environment variables."""
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1000")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "200")
    monkeypatch.setenv("RAG_CHUNK_SEPARATORS", '["\\n\\n", " "]')
    monkeypatch.delenv("TEST_RAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("TEST_RAG_CHUNK_OVERLAP", raising=False)
    monkeypatch.delenv("TEST_RAG_CHUNK_SEPARATORS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    config = ChromaConfig()

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.chunk_separators == ["\n\n", " "]
