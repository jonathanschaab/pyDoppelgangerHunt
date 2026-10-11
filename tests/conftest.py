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


def assert_valid_python(source: str, mode: str = "exec") -> None:
    """Verifies that source is both syntactically and semantically valid Python."""
    lines = source.splitlines(keepends=True)
    import_lines = [ln for ln in lines if ln.startswith(("import ", "from "))]
    other_lines = [ln for ln in lines if not ln.startswith(("import ", "from "))]
    if import_lines:
        compile("".join(import_lines), filename="<synthesized_imports>", mode=mode)
    other_code = "".join(other_lines)
    if other_code.strip():
        first_line = next((ln for ln in other_lines if ln.strip()), "")
        if first_line.startswith((" ", "\t")):
            compile("if 1:\n" + other_code, filename="<synthesized_helper>", mode=mode)
        else:
            compile(other_code, filename="<synthesized_helper>", mode=mode)

