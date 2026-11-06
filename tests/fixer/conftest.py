"""Shared pytest fixtures for fixer dataflow and closure-cell tests."""

from __future__ import annotations

from typing import Iterator

import pytest

from pydoppelgangerhunt.fixer import clear_dataflow_cache


@pytest.fixture(autouse=True)
def _reset_dataflow_caches() -> Iterator[None]:
    """Clears module-level AST and dataflow caches before and after every test."""
    clear_dataflow_cache()
    yield
    clear_dataflow_cache()
