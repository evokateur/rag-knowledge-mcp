"""
Shared configuration for RAG Knowledge MCP Server

Configuration is loaded from environment variables (via .env file if present).
This module provides a single source of truth for all configuration values.
"""

import os
import importlib
from pathlib import Path
from typing import TYPE_CHECKING
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from abstract_backend import AbstractRagBackend

# Project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent

# Load environment variables from .env file if it exists
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)


# Helper to get config value with optional TEST_ prefix in test mode
def _get_config(key: str, default: str) -> str:
    """
    Get configuration value with TEST_ prefix in test mode.

    Checks for PYTEST_CURRENT_TEST environment variable at runtime
    (not import time) to detect test mode, ensuring proper test isolation.
    """
    # Check test mode dynamically (pytest sets PYTEST_CURRENT_TEST during test execution)
    is_test_mode = "PYTEST_CURRENT_TEST" in os.environ

    if is_test_mode:
        test_value = os.getenv(f"TEST_{key}")
        if test_value is not None:
            return test_value
    return os.getenv(key, default)


# Helper to convert relative paths to absolute paths relative to project root
def _absolute_path(path_str: str) -> str:
    """
    Convert path to absolute path, resolving relative paths from project root.

    This ensures that relative paths in .env are always interpreted
    relative to the project directory (where config.py is), not relative
    to the current working directory.

    Args:
        path_str: Path string (can be absolute or relative)

    Returns:
        Absolute path as string

    Example:
        "./chroma_db" -> "/absolute/path/to/project/chroma_db"
        "/tmp/db" -> "/tmp/db" (unchanged)
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.absolute())


# ============================================================================
# Pydantic Configuration Models
# ============================================================================


class BackendConfig(BaseModel):
    """
    Base configuration for all RAG backends.

    Loads from environment variables with optional TEST_ prefix in test mode.
    Subclass this to add backend-specific configuration.
    """

    model_config = ConfigDict(frozen=True)  # Make config immutable after creation

    knowledge_dir: str = Field(
        default_factory=lambda: _absolute_path(_get_config("RAG_KNOWLEDGE_DIR", "./knowledge-base")),
        description="Knowledge base source directory (input for ingestion)",
    )

    persist_dir: str = Field(
        default_factory=lambda: _absolute_path(_get_config("RAG_PERSIST_DIR", "./chroma_db")),
        description="Vector database directory (output/storage)",
    )

    collection: str = Field(
        default_factory=lambda: _get_config("RAG_COLLECTION", "knowledge_base"),
        description="Collection name in the vector database",
    )

    embedding_model: str = Field(
        default_factory=lambda: _get_config(
            "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        ),
        description="Sentence Transformers embedding model name",
    )


# ============================================================================
# Legacy module-level exports (for backwards compatibility)
# ============================================================================

# Knowledge base source directory (input for ingestion)
RAG_KNOWLEDGE_DIR = _absolute_path(_get_config("RAG_KNOWLEDGE_DIR", "./knowledge-base"))

# Vector database directory (output of ingestion, input for queries)
RAG_PERSIST_DIR = _absolute_path(_get_config("RAG_PERSIST_DIR", "./chroma_db"))

# Collection name in the vector database
RAG_COLLECTION = _get_config("RAG_COLLECTION", "knowledge_base")

# Sentence Transformers embedding model
# Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (quality), multi-qa-mpnet-base-dot-v1 (Q&A)
RAG_EMBEDDING_MODEL = _get_config("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Backend class to use (fully qualified path: module.ClassName)
RAG_BACKEND_CLASS = _get_config("RAG_BACKEND_CLASS", "chroma_backend.RagBackend")


def create_rag_backend() -> "AbstractRagBackend":
    """
    Factory function to create the configured RAG backend.

    The backend class is specified by RAG_BACKEND_CLASS environment variable
    in the format "module_name.ClassName" (e.g., "chroma_backend.RagBackend").

    This allows swapping vector database implementations without changing code.
    Just set RAG_BACKEND_CLASS in your .env file to point to a different backend.

    Returns:
        AbstractRagBackend: An instance of the configured backend class

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the class doesn't exist in the module
        ValueError: If RAG_BACKEND_CLASS format is invalid

    Example:
        backend = create_rag_backend()
        await backend.initialize()
    """
    if "." not in RAG_BACKEND_CLASS:
        raise ValueError(
            f"RAG_BACKEND_CLASS must be in format 'module.ClassName', got: {RAG_BACKEND_CLASS}"
        )

    # Split module and class name
    module_name, class_name = RAG_BACKEND_CLASS.rsplit(".", 1)

    # Dynamic import
    module = importlib.import_module(module_name)
    BackendClass = getattr(module, class_name)

    # Instantiate and return
    return BackendClass()
