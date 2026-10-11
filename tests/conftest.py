"""Pytest fixtures and configuration for pyDoppelgangerHunt tests."""

from __future__ import annotations

from typing import Generator

import pytest

from pydoppelgangerhunt.fixer import clear_dataflow_cache


@pytest.fixture(autouse=True)
def auto_reset_downstream_reads_cache() -> Generator[None, None, None]:
    """Resets the dataflow cache before and after each test."""
    clear_dataflow_cache()
    yield
    clear_dataflow_cache()
