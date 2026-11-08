"""Shared pytest fixtures for fixer dataflow and closure-cell tests."""

from __future__ import annotations

from typing import Iterator

import pytest

from pydoppelgangerhunt.fixer import clear_dataflow_cache
from tests.fixer._closure_test_helpers import (
    _extract_symtable_scope_info,
    _run_differential_refactoring_case,
    locate_ast_stmt_bounds,
    locate_marked_line_bounds,
)

__all__ = [
    "_extract_symtable_scope_info",
    "_run_differential_refactoring_case",
    "locate_ast_stmt_bounds",
    "locate_marked_line_bounds",
]


@pytest.fixture(autouse=True)
def _reset_dataflow_caches() -> Iterator[None]:
    """Clears module-level AST and dataflow caches before and after every test."""
    clear_dataflow_cache()
    yield
    clear_dataflow_cache()
