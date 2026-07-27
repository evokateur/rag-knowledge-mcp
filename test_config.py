#!/usr/bin/env python3
"""Test layered YAML configuration loading."""

import textwrap
from pathlib import Path

from config import (
    BackendConfig,
    PROJECT_ROOT,
    _absolute_path,
    _build_merged,
    deep_merge,
)
from chroma_backend import ChromaConfig


DEFAULTS_YAML = textwrap.dedent(
    """
    backend_class: chroma_backend.RagBackend
    knowledge_dir: ./knowledge-base
    persist_dir: ./chroma_db
    collection: knowledge_base
    embedding_model: all-MiniLM-L6-v2
    chunk_size: 500
    chunk_overlap: 100
    chunk_separators: ["\\n\\n", "\\n", ". ", " ", ""]
    """
)

TEST_YAML = textwrap.dedent(
    """
    knowledge_dir: ./test-knowledge-base
    persist_dir: ./test_chroma_db
    collection: test_knowledge_base
    """
)


def _write(path: Path, content: str) -> None:
    path.write_text(content)


# ============================================================================
# deep_merge
# ============================================================================


def test_deep_merge_overrides_scalars():
    base = {"a": 1, "b": 2}
    deep_merge(base, {"b": 3})
    assert base == {"a": 1, "b": 3}


def test_deep_merge_is_recursive():
    base = {"nested": {"a": 1, "b": 2}}
    deep_merge(base, {"nested": {"b": 3, "c": 4}})
    assert base == {"nested": {"a": 1, "b": 3, "c": 4}}


# ============================================================================
# _absolute_path
# ============================================================================


def test_absolute_path_resolves_relative_from_project_root():
    assert _absolute_path("./chroma_db") == str((PROJECT_ROOT / "chroma_db").absolute())


def test_absolute_path_expands_tilde():
    assert _absolute_path("~/db") == str((Path.home() / "db").absolute())


def test_absolute_path_leaves_absolute_unchanged():
    assert _absolute_path("/tmp/db") == "/tmp/db"


# ============================================================================
# _build_merged layering
# ============================================================================


def test_defaults_only(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)

    merged = _build_merged(project_root=tmp_path, under_test=False)

    assert merged["collection"] == "knowledge_base"
    assert merged["backend_class"] == "chroma_backend.RagBackend"
    # relative paths resolve against the real project root, not the yaml location
    assert merged["knowledge_dir"] == str((PROJECT_ROOT / "knowledge-base").absolute())


def test_user_then_local_override(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)
    user_file = tmp_path / "user.yaml"
    _write(user_file, "collection: from_user\nembedding_model: from_user\n")
    _write(tmp_path / "settings.local.yaml", "collection: from_local\n")

    merged = _build_merged(
        project_root=tmp_path, user_config_file=user_file, under_test=False
    )

    # local wins over user, user wins over defaults
    assert merged["collection"] == "from_local"
    assert merged["embedding_model"] == "from_user"


def test_under_test_merges_test_layer_and_ignores_user_local(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)
    _write(tmp_path / "settings.test.yaml", TEST_YAML)
    user_file = tmp_path / "user.yaml"
    _write(user_file, "collection: from_user\n")
    _write(tmp_path / "settings.local.yaml", "collection: from_local\n")

    merged = _build_merged(
        project_root=tmp_path, user_config_file=user_file, under_test=True
    )

    # test layer wins; user/local are not consulted in test mode
    assert merged["collection"] == "test_knowledge_base"
    assert merged["persist_dir"] == str((PROJECT_ROOT / "test_chroma_db").absolute())


def test_paths_resolved_to_absolute(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)

    merged = _build_merged(project_root=tmp_path, under_test=False)

    assert Path(merged["knowledge_dir"]).is_absolute()
    assert Path(merged["persist_dir"]).is_absolute()


# ============================================================================
# Pydantic models validate the merged dict
# ============================================================================


def test_backend_config_from_merged(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)
    merged = _build_merged(project_root=tmp_path, under_test=False)

    config = BackendConfig.model_validate(merged)

    assert config.collection == "knowledge_base"
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_chroma_config_from_merged(tmp_path):
    _write(tmp_path / "settings.yaml", DEFAULTS_YAML)
    merged = _build_merged(project_root=tmp_path, under_test=False)

    config = ChromaConfig.model_validate(merged)

    assert config.chunk_size == 500
    assert config.chunk_overlap == 100
    assert config.chunk_separators == ["\n\n", "\n", ". ", " ", ""]
    # inherited base field still present
    assert config.collection == "knowledge_base"
