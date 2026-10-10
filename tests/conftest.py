"""Pytest fixtures and configuration for pyDoppelgangerHunt tests."""

from __future__ import annotations

from typing import Generator

import pytest

from pydoppelgangerhunt.fixer.dataflow import _clear_downstream_reads_cache


@pytest.fixture(autouse=True)
def auto_reset_downstream_reads_cache() -> Generator[None, None, None]:
    """Resets the downstream reads cache before and after each test."""
    _clear_downstream_reads_cache()
    yield
    _clear_downstream_reads_cache()
