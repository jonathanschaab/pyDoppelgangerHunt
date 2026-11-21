"""Pytest fixtures and configuration for pyDoppelgangerHunt tests."""

from __future__ import annotations

from typing import Generator

import pytest

from pydoppelgangerhunt.fixer import clear_dataflow_cache
from pydoppelgangerhunt.fixer._df_cache import _debug_ast_immutability


@pytest.fixture(autouse=True)
def auto_reset_downstream_reads_cache() -> Generator[None, None, None]:
    """Resets the dataflow cache and restores AST immutability mode around each test."""
    prev_debug = _debug_ast_immutability[0]
    _debug_ast_immutability[0] = False
    clear_dataflow_cache()
    try:
        yield
    finally:
        clear_dataflow_cache()
        _debug_ast_immutability[0] = prev_debug
