#!/usr/bin/env python3
"""Test Abstract Base Class implementation."""

import inspect
import pytest
from abstract_backend import AbstractRagBackend
from config import create_rag_backend


def test_abstract_backend_import():
    """Verify AbstractRagBackend can be imported."""
    assert AbstractRagBackend is not None


def test_abstract_backend_is_abstract():
    """Verify AbstractRagBackend is actually abstract."""
    assert inspect.isabstract(AbstractRagBackend)


def test_cannot_instantiate_directly():
    """Verify AbstractRagBackend cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractRagBackend()


def test_has_required_abstract_methods():
    """Verify all required abstract methods are defined."""
    expected_methods = {
        "_initialize_backend",  # initialize() is concrete, _initialize_backend() is abstract
        "search",
        "list_documents",
        "get_document",
        "get_stats",
        "close",
        "add_document",
        "delete_document",
        "ingest_directory",
    }

    abstract_methods = {
        name
        for name, method in inspect.getmembers(AbstractRagBackend)
        if getattr(method, "__isabstractmethod__", False)
    }

    assert abstract_methods == expected_methods


def test_factory_returns_abstract_backend_instance():
    """Verify create_rag_backend returns an AbstractRagBackend instance."""
    backend = create_rag_backend()
    assert isinstance(backend, AbstractRagBackend)
