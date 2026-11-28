"""Unit tests for single-boundary UnitSchema normalization."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict

import pytest

from pydoppelgangerhunt import (
    NormalizedCoordinates,
    NormalizedSourceBuffer,
    NormalizedUnit,
    SourceLinesOrigin,
    normalize_unit_boundary,
    resolve_unit_source_buffer,
)


class _CustomPathLike:
    """Minimal os.PathLike implementation whose __str__ differs from __fspath__."""

    def __init__(self, target: str) -> None:
        self._target = target

    def __fspath__(self) -> str:
        return self._target

    def __str__(self) -> str:
        return "<opaque-path-repr>"


@pytest.mark.parametrize(
    ("unit_payload", "caller_strictness", "default_strictness", "expected_mode"),
    [
        pytest.param({}, None, "strict", "strict", id="unset_defaults_to_strict"),
        pytest.param({}, None, "pedantic", "pedantic", id="unset_defaults_to_pedantic"),
        pytest.param(
            {"closure_strictness": "lenient"},
            None,
            "pedantic",
            "lenient",
            id="unit_strictness_overrides_default",
        ),
        pytest.param(
            {"closure_strictness": "lenient"},
            "pedantic",
            "strict",
            "pedantic",
            id="caller_strictness_overrides_unit",
        ),
    ],
)
def test_normalize_unit_boundary_resolves_coords_pathlike_anchor_and_policy(
    unit_payload: Dict[str, Any],
    caller_strictness: str | None,
    default_strictness: str,
    expected_mode: str,
) -> None:
    """Verifies normalize_unit_boundary normalizes PathLike, aliases, anchors, and policy once."""
    unit: Dict[str, Any] = {
        "file": _CustomPathLike("notebooks/demo.ipynb#cell_3"),
        "name": "cell_fn",
        "type": "compound_block",
        "lineno": 10,
        "end_lineno": 20,
        "col_offset": 4,
        "end_col_offset": 18,
        "source_lines_is_sliced": True,
        "source_lines_is_col_sliced": False,
        "allow_snippet_only": True,
        **unit_payload,
    }
    norm = normalize_unit_boundary(
        unit,
        closure_strictness=caller_strictness,
        default_strictness=default_strictness,
    )
    assert isinstance(norm, NormalizedUnit)
    assert norm.file_path == "notebooks/demo.ipynb#cell_3"
    assert norm.notebook_anchor == ("notebooks/demo.ipynb", 2)
    assert norm.name == "cell_fn"
    assert norm.kind == "compound_block"
    assert norm.coords == NormalizedCoordinates(start=10, end=20, start_col=4, end_col=18)
    assert norm.policy.mode == expected_mode
    assert norm.source_lines_is_sliced is True
    assert norm.source_lines_is_col_sliced is False
    assert norm.allow_snippet_only is True
    with pytest.raises(FrozenInstanceError):
        norm.name = "mutated"  # type: ignore[misc]


def test_normalize_unit_boundary_handles_non_numeric_coords_and_non_bool_slice_flags() -> None:
    """Verifies non-numeric coordinates and non-bool slice flags degrade safely to None."""
    norm = normalize_unit_boundary(
        {
            "file": "plain.py",
            "start": "not_an_int",
            "end": None,
            "source_lines_is_sliced": "yes",
            "source_lines_is_col_sliced": 1,
        }
    )
    assert norm.coords == NormalizedCoordinates(
        start=None, end=None, start_col=None, end_col=None
    )
    assert norm.has_malformed_coords is True
    assert norm.notebook_anchor is None
    assert norm.source_lines_is_sliced is None
    assert norm.source_lines_is_col_sliced is None


def test_resolve_unit_source_buffer_respects_allow_disk_read_and_include_lines(
    tmp_path: Path,
) -> None:
    """Verifies allow_disk_read=False produces placeholder lines and include_lines=False skips line slicing."""
    mod_file = tmp_path / "sample.py"
    mod_file.write_text("x = 1\ny = 2\n", encoding="utf-8")

    no_disk = resolve_unit_source_buffer(
        {"file": str(mod_file), "name": "block", "start": 1, "end": 2},
        repo_root=str(tmp_path),
        allow_disk_read=False,
    )
    assert no_disk.origin is SourceLinesOrigin.PLACEHOLDER
    assert no_disk.lines == ["# Source for block lines 1-2\n"]

    full_only = resolve_unit_source_buffer(
        {"file": str(mod_file), "start": 1, "end": 2, "source_text": "a = 1\nb = 2\n"},
        include_lines=False,
    )
    assert full_only.lines == []
    assert full_only.full_text == "a = 1\nb = 2\n"
    assert full_only.is_invalid_notebook is False


def test_resolve_unit_source_buffer_reads_disk_and_inline_with_canonical_origin(
    tmp_path: Path,
) -> None:
    """Verifies resolve_unit_source_buffer resolves disk files and inline buffers identically."""
    mod_file = tmp_path / "sample.py"
    src = "def f(x):\n    y = x + 1\n    return y\n"
    mod_file.write_text(src, encoding="utf-8")

    disk_buf = resolve_unit_source_buffer(
        {"file": _CustomPathLike(str(mod_file)), "start": 2, "end": 3},
        repo_root=str(tmp_path),
    )
    assert isinstance(disk_buf, NormalizedSourceBuffer)
    assert disk_buf.origin is SourceLinesOrigin.DISK
    assert disk_buf.lines == ["    y = x + 1\n", "    return y\n"]
    assert disk_buf.full_text is None
    assert disk_buf.is_invalid_notebook is False
    assert disk_buf.file_path == str(mod_file)

    inline_buf = resolve_unit_source_buffer(
        {
            "file": "analysis.ipynb#cell_1",
            "start": 1,
            "end": 1,
            "source_text": "val = 42\n",
        }
    )
    assert inline_buf.origin is SourceLinesOrigin.FULL_BUFFER
    assert inline_buf.lines == ["val = 42\n"]
    assert inline_buf.full_text == "val = 42\n"
    assert inline_buf.notebook_anchor == ("analysis.ipynb", 0)
    assert inline_buf.is_invalid_notebook is False
