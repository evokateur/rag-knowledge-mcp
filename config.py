"""
Shared configuration for RAG Knowledge MCP Server.

Configuration is loaded from layered YAML files and merged (later wins):

    settings.yaml                       committed defaults (required)
    ~/.rag-knowledge-mcp/settings.yaml  user override (optional)
    settings.local.yaml                 local override (optional, gitignored)

Under pytest the layers are instead settings.yaml -> settings.test.yaml, so
tests run against an isolated database without touching production settings.
"""

import os
import importlib
import copy
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from abstract_backend import AbstractRagBackend

# Project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent

# Optional per-user override file
USER_CONFIG_FILE = Path.home() / ".rag-knowledge-mcp" / "settings.yaml"

# Path-valued keys resolved to absolute paths against the project root
_PATH_KEYS = ("knowledge_dir", "persist_dir")


def _read_yaml_file(path: Path, *, required: bool = False) -> dict:
    """Read a YAML file and return a dict, or {} if it is absent/empty."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required settings file not found: {path}")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def _absolute_path(path_str: str) -> str:
    """Resolve a path to absolute, expanding ~ and rooting relatives at PROJECT_ROOT."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.absolute())


def _build_merged(
    project_root: Path = PROJECT_ROOT,
    user_config_file: Path = USER_CONFIG_FILE,
    under_test: Optional[bool] = None,
) -> dict:
    """Assemble the merged configuration dict from the YAML layers.

    Parameterized so tests can point it at a temporary directory.
    """
    if under_test is None:
        under_test = "PYTEST_CURRENT_TEST" in os.environ

    merged = _read_yaml_file(project_root / "settings.yaml", required=True)

    if under_test:
        deep_merge(merged, _read_yaml_file(project_root / "settings.test.yaml"))
    else:
        deep_merge(merged, _read_yaml_file(user_config_file))
        deep_merge(merged, _read_yaml_file(project_root / "settings.local.yaml"))

    for key in _PATH_KEYS:
        if key in merged:
            merged[key] = _absolute_path(merged[key])

    return merged


def get_merged_config() -> dict:
    """Return a fresh copy of the merged runtime configuration."""
    return copy.deepcopy(_build_merged())


# ============================================================================
# Pydantic Configuration Models
# ============================================================================


class BackendConfig(BaseModel):
    """Base configuration for all RAG backends.

    Populate via ``BackendConfig.model_validate(get_merged_config())``. Extra
    keys in the merged dict (e.g. backend_class, backend-specific fields) are
    ignored. Subclass this to add backend-specific configuration.
    """

    model_config = ConfigDict(frozen=True)  # immutable after creation

    knowledge_dir: str = Field(
        description="Knowledge base source directory (input for ingestion)",
    )
    persist_dir: str = Field(
        description="Vector database directory (output/storage)",
    )
    collection: str = Field(
        description="Collection name in the vector database",
    )
    embedding_model: str = Field(
        description="Sentence Transformers embedding model name",
    )


def create_rag_backend() -> "AbstractRagBackend":
    """Instantiate the RAG backend named by ``backend_class`` in settings.yaml.

    The value is a fully qualified path "module.ClassName" (e.g.
    "chroma_backend.RagBackend"), letting you swap vector database
    implementations without changing code.

    Example:
        backend = create_rag_backend()
        await backend.initialize()
    """
    backend_class = get_merged_config()["backend_class"]

    if "." not in backend_class:
        raise ValueError(
            f"backend_class must be in format 'module.ClassName', got: {backend_class}"
        )

    module_name, class_name = backend_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    BackendClass = getattr(module, class_name)
    return BackendClass()
