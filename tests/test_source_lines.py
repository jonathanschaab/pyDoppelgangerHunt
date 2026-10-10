"""Unit tests for unit source lines coordinate parsing, bounds checking, and column resolution."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from pydoppelgangerhunt import (
    check_units_overlap,
    compute_unit_spans,
)
from pydoppelgangerhunt.fixer import analyze_unit_variable_scope
from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access
    _extract_unit_end_col,
    _get_valid_unit_bounds,
    _resolve_unit_ast_end_col,
    collect_downstream_read_names,
)
from pydoppelgangerhunt.matcher import (  # pylint: disable=protected-access
    _check_column_bounds_relationship,
    _unit_sloc,
    _update_merged_unit_columns,
)
from pydoppelgangerhunt.reporters import (  # pylint: disable=protected-access
    _unit_line_bounds,
    format_github_annotations,
    format_sarif_report,
)
from pydoppelgangerhunt.source_lines import parse_unit_coord


def test_resolve_unit_ast_end_col_multiline_unit_statements() -> None:
    """Verifies that multi-line units ending on a line with multiple statements include all statements up through the line end."""

    code = (
        "def worker():\n"
        "    a = 1\n"
        "    total = 10; count = 20\n"
        "    print(res)\n"
    )
    # Multi-line unit covering lines 2 to 3
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "count", "res"})
    assert reads is not None
    # total and count are on line 3 (end line of multi-line unit), so they are NOT downstream reads
    assert "total" not in reads
    assert "count" not in reads


def test_resolve_unit_ast_end_col_bounds_validation() -> None:
    """Verifies bounds validation in _resolve_unit_ast_end_col."""
    tree = ast.parse("x = 10\ny = 20\n")
    # Inverted start and end coordinates
    assert _resolve_unit_ast_end_col(tree, {"start": 5, "end": 2}) is None
    # Zero or negative start
    assert _resolve_unit_ast_end_col(tree, {"start": 0, "end": 2}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": -1, "end": 2}) is None
    # Zero or negative end
    assert _resolve_unit_ast_end_col(tree, {"start": 1, "end": 0}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": 1, "end": -1}) is None


def test_inspect_unit_scope_coordinate_string_with_column_offset() -> None:
    """Verifies that _inspect_unit_scope parses coordinate strings with column offsets."""
    full_text = (
        "async def async_worker(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item\n"
        "    return total\n"
    )
    unit = {
        "start": "3:4",
        "end": "4:20",
        "source_lines": ["    for item in items:\n", "        total += item\n"],
        "source_text": full_text,
    }
    scope = analyze_unit_variable_scope(unit)
    assert scope["is_async"] is True


def test_resolve_unit_ast_end_col_single_line_semicolon_with_and_without_start_col() -> None:
    """Verifies that _resolve_unit_ast_end_col returns None when multiple statements share
    a line without column bounds, but accurately selects subsequent statements when start_col
    is supplied."""
    code = "def f(): x = 1; y = 2\n"
    tree = ast.parse(code)
    scope_fn = tree.body[0]

    # Without start_col: ambiguous multiple statements fail closed by returning None
    unit_no_col = {"start": 1, "end": 1}
    end_col_first = _resolve_unit_ast_end_col(scope_fn, unit_no_col)
    assert end_col_first is None

    # With start_col: accurately matches second statement y = 2
    unit_with_col = {"start": 1, "end": 1, "start_col": len("def f(): x = 1; ")}
    end_col_second = _resolve_unit_ast_end_col(scope_fn, unit_with_col)
    assert end_col_second == len("def f(): x = 1; y = 2")


def test_resolve_unit_ast_end_col_multiline_semicolon_closing_line() -> None:
    """Verifies that for multi-line units ending on a line with multiple statements,
    _resolve_unit_ast_end_col fails closed and returns None when columns are not given."""
    code = (
        "def f():\n"
        "    x = 1\n"
        "    y = 2; z = 3\n"
    )
    tree = ast.parse(code)
    scope_fn = tree.body[0]

    unit = {"start": 2, "end": 3}
    end_col = _resolve_unit_ast_end_col(scope_fn, unit)
    assert end_col is None


def test_extract_and_resolve_unit_col_coord_strings() -> None:
    """Verifies that column offset helpers safely parse coordinate strings with offsets."""
    u_end_str = {"end_col": "25:0"}
    assert _extract_unit_end_col(u_end_str) == 25

    u_end_offset_str = {"end_col_offset": "40:5"}
    assert _extract_unit_end_col(u_end_offset_str) == 40

    tree = ast.parse("x = 10; y = 20\n")
    scope_fn = tree
    u_start_str = {"start": 1, "end": 1, "start_col": "8:0"}
    end_col = _resolve_unit_ast_end_col(scope_fn, u_start_str)
    assert end_col == len("x = 10; y = 20")


def test_parse_unit_coord_overload_and_default_none() -> None:
    """Verifies parse_unit_coord behavior with default=None and formatted coordinates."""
    assert parse_unit_coord({"col": None}, "col", default=None) is None
    assert parse_unit_coord({}, "col", default=None) is None
    assert parse_unit_coord({"col": ""}, "col", default=None) is None
    assert parse_unit_coord({"col": "   "}, "col", default=None) is None
    assert parse_unit_coord({"col": ":0"}, "col", default=None) is None
    assert parse_unit_coord({"col": ""}, "col", default=1) == 1
    assert parse_unit_coord({"col": "   "}, "col", default=1) == 1
    assert parse_unit_coord({"col": ":0"}, "col", default=1) == 1
    assert parse_unit_coord({"col": 42}, "col", default=None) == 42
    assert parse_unit_coord({"col": "42:0"}, "col", default=None) == 42
    assert parse_unit_coord({"col": "42.0"}, "col") == 42
    assert parse_unit_coord({"col": "42.0:0"}, "col") == 42
    assert parse_unit_coord({"col": 42.0}, "col") == 42
    assert parse_unit_coord({"col": 42}, "col") == 42
    assert parse_unit_coord({}, "col", default=10) == 10
    with pytest.raises(ValueError):
        parse_unit_coord({"col": "invalid"}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": float("inf")}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": "inf"}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": float("nan")}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": "nan"}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": [10]}, "col")
    with pytest.raises(ValueError):
        parse_unit_coord({"col": {"nested": 1}}, "col")


@pytest.mark.parametrize(
    ("unit_dict", "expected"),
    [
        ({"start": "invalid"}, None),
        ({"start": "10", "end": "invalid"}, None),
        ({"start": None}, None),
        ({"start": 0}, None),
        ({"start": 5, "end": 2}, None),
        ({"start": "2", "end": "5"}, (2, 5)),
    ],
)
def test_get_valid_unit_bounds_malformed_coordinates(
    unit_dict: Dict[str, Any], expected: Optional[Tuple[int, int]]
) -> None:
    """Verifies that _get_valid_unit_bounds handles malformed coordinates defensively."""
    assert _get_valid_unit_bounds(unit_dict) == expected


def test_resolve_unit_ast_end_col_single_line_compound_block() -> None:
    """Verifies that single-line compound blocks with multiple trailing statements
    fail closed and return None without column bounds."""
    code = (
        "def process(items):\n"
        "    for x in items: total += x; yield x\n"
    )
    tree = ast.parse(code)
    scope_fn = tree.body[0]
    unit = {"start": 2, "end": 2, "kind": "compound_block"}
    end_col = _resolve_unit_ast_end_col(scope_fn, unit)
    assert end_col is None


def test_coordinate_parsing_colon_formatted_columns_across_subsystems(
    tmp_path: Path,
) -> None:
    """Verifies that colon-formatted columns (e.g. '8:0') parse safely across all subsystems."""
    u1 = {"file": "mod.py", "start": 1, "end": 1, "start_col": "8:0", "end_col": "15:0"}
    u2 = {"file": "mod.py", "start": 1, "end": 1, "start_col": "16:0", "end_col": "22:0"}

    # 1. check_units_overlap
    assert check_units_overlap(u1, u2) is False

    # 2. compute_unit_spans
    src = "        val = compute()  # code\n"
    span = compute_unit_spans(src, u1)
    assert span.start_col_char == 8
    assert span.end_col_char == 15

    # 3. Matcher column bounds comparison & propagation
    enclosed, _ = _check_column_bounds_relationship(1, 1, 1, 1, u1, u2)
    assert enclosed is False
    target = {"start_col": "8:0", "end_col": "20:0"}
    donor = {"start_col": "4:0", "end_col": "25:0"}
    _update_merged_unit_columns(target, donor, 1, 1, 1, 1)
    assert target["start_col"] == 4
    assert target["end_col"] == 25

    # 4. Reporters
    sarif = format_sarif_report([(0.9, u1, u2)], target=".", threshold=0.8)
    assert "startColumn" in json.dumps(sarif)
    ann = format_github_annotations([(0.9, u1, u2)])
    assert any("col=9" in a for a in ann)


def test_coordinate_parsing_colon_formatted_lines_across_subsystems() -> None:
    """Verifies that colon-formatted lines (e.g. '10:0') parse safely across all subsystems."""
    from pydoppelgangerhunt.clustering import cluster_clone_families, unit_key
    from pydoppelgangerhunt.coverage import compute_unit_coverage
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap
    from pydoppelgangerhunt.matcher import compute_priority_score
    from pydoppelgangerhunt.fixer.source import is_valid_unit_coordinates

    u1 = {
        "file": "mod.py",
        "name": "f1",
        "start": "10:0",
        "end": "20:0",
        "kind": "function",
        "token_count": 30,
        "complexity": 2,
    }
    u2 = {
        "file": "mod.py",
        "name": "f2",
        "start": "30:0",
        "end": "40:0",
        "kind": "function",
        "token_count": 30,
        "complexity": 2,
    }

    # 1. unit_key
    k1 = unit_key(u1)
    assert "10-20:f1" in k1

    # 2. compute_priority_score
    p_score = compute_priority_score(0.9, u1, u2)
    assert p_score > 0

    # 3. cluster_clone_families
    clones = [(0.9, u1, u2)]
    fams = cluster_clone_families(clones)
    assert len(fams) == 1
    assert fams[0]["total_lines"] == 22

    # 4. compute_unit_coverage
    cov_data = {"mod.py": {10, 11, 12}}
    cov = compute_unit_coverage(u1, cov_data)
    assert 0.0 < cov < 1.0

    # 5. compute_unit_diff_overlap
    overlap_count, ratio = compute_unit_diff_overlap(
        u1, {"mod.py": [(10, 15)]}
    )
    assert overlap_count == 6
    assert ratio > 0.0

    # 6. check_units_overlap
    assert check_units_overlap(u1, u2) is False

    # 7. is_valid_unit_coordinates
    assert is_valid_unit_coordinates(u1, strict=False) is True
    assert is_valid_unit_coordinates(u1) is True
    assert is_valid_unit_coordinates(u1, strict=True) is False
    assert is_valid_unit_coordinates({"start": "invalid:foo"}) is False


def test_parse_unit_coord_non_integral_floats_rejected() -> None:
    """Verifies that non-integral floats and float strings truncate in default mode
    and are rejected in strict mode."""
    # Integral floats and strings are accepted
    assert parse_unit_coord({"start": 10.0}, "start") == 10
    assert parse_unit_coord({"start": "10.0"}, "start") == 10
    assert parse_unit_coord({"start": "10.0:0"}, "start") == 10

    # Non-integral floats truncate in lenient/default mode
    assert parse_unit_coord({"start": 10.5}, "start") == 10
    assert parse_unit_coord({"start": "12.7"}, "start") == 12
    assert parse_unit_coord({"start": "12.9:0"}, "start") == 12

    # In strict mode, float coordinates are rejected
    with pytest.raises(ValueError, match="float coordinate not allowed in strict mode"):
        parse_unit_coord({"start": 10.5}, "start", strict=True)

    with pytest.raises(ValueError, match="cannot convert to integer"):
        parse_unit_coord({"start": "12.7"}, "start", strict=True)


def test_coordinate_clamping_debug_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that clamping corrupt/inverted coordinates logs at DEBUG level."""
    with caplog.at_level(logging.DEBUG):
        unit_inverted = {"file": "test.py", "name": "foo", "start": 10, "end": 5}
        sloc = _unit_sloc(unit_inverted)
        assert sloc == 1
        assert "Clamped invalid unit coordinates" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        unit_zero = {"file": "test.py", "name": "bar", "start": 0, "end": 0}
        bounds = _unit_line_bounds(unit_zero)
        assert bounds == (1, 1)
        assert "Clamped invalid unit coordinates" in caplog.text


def test_is_valid_unit_coordinates_strict_mode() -> None:
    """Verifies that is_valid_unit_coordinates strictly rejects blank, colon, float, or missing
    coordinates in explicit strict mode, but permits colon-formatted and float coordinates in
    default lenient mode."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        is_valid_unit_coordinates,
    )

    assert is_valid_unit_coordinates({"start": 1, "end": 2}) is True
    # Colon coordinates rejected in strict mode, accepted in lenient mode
    colon_unit = {"start": "10:0", "end": "20:0"}
    assert is_valid_unit_coordinates(colon_unit, strict=True) is False
    assert is_valid_unit_coordinates(colon_unit, strict=False) is True
    # Blank start rejected in both
    assert is_valid_unit_coordinates({"start": ""}) is False
    assert is_valid_unit_coordinates({"start": ""}, strict=False) is False
    # Missing start rejected
    assert is_valid_unit_coordinates({"end": 5}) is False
    # Missing end rejected in strict mode
    assert is_valid_unit_coordinates({"start": 5}, strict=True) is False
    # Inverted coordinates rejected in both strict and lenient modes
    assert is_valid_unit_coordinates({"start": 5, "end": 2}, strict=True) is False
    assert is_valid_unit_coordinates({"start": 5, "end": 2}, strict=False) is False
    # Zero or negative end rejected
    assert is_valid_unit_coordinates({"start": 5, "end": 0}) is False
    assert is_valid_unit_coordinates({"start": 5, "end": -1}) is False
    # Inverted same-line column coordinates rejected
    assert is_valid_unit_coordinates({"start": 1, "end": 1, "start_col": 10, "end_col": 5}) is False
    # Float rejected in strict mode, accepted in default/lenient mode
    assert is_valid_unit_coordinates({"start": 12.0}, strict=True) is False
    assert is_valid_unit_coordinates({"start": 12.0}) is True
    assert is_valid_unit_coordinates({"start": 12.0}, strict=False) is True


def test_parse_unit_coord_strict_rejects_bool() -> None:
    """Verifies that parse_unit_coord with strict=True rejects boolean literals."""
    with pytest.raises(ValueError, match="boolean coordinate not allowed in strict mode"):
        parse_unit_coord({"start": True}, "start", strict=True)

    with pytest.raises(ValueError, match="boolean coordinate not allowed in strict mode"):
        parse_unit_coord({"start": False}, "start", strict=True)

    # In lenient mode, int(True) evaluates to 1
    assert parse_unit_coord({"start": True}, "start", strict=False) == 1
    # Valid int in strict mode succeeds
    assert parse_unit_coord({"start": 42}, "start", strict=True) == 42


def test_parse_unit_coord_strict_rejects_signed_prefix() -> None:
    """Verifies that parse_unit_coord with strict=True rejects signed prefix strings."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        is_valid_unit_coordinates,
    )

    with pytest.raises(ValueError, match="signed prefix not allowed in strict mode"):
        parse_unit_coord({"start": "+1"}, "start", strict=True)

    with pytest.raises(ValueError, match="signed prefix not allowed in strict mode"):
        parse_unit_coord({"start": "-1"}, "start", strict=True)

    # In lenient mode, signed strings parse to their integer values
    assert parse_unit_coord({"start": "+1"}, "start", strict=False) == 1
    assert parse_unit_coord({"start": "-1"}, "start", strict=False) == -1

    # is_valid_unit_coordinates with strict=True rejects signed prefixes
    assert is_valid_unit_coordinates({"start": "+1", "end": "5"}, strict=True) is False
    assert is_valid_unit_coordinates({"start": "+1", "end": "5"}, strict=False) is True

