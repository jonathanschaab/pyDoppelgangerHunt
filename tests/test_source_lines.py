"""Unit tests for unit source lines coordinate parsing, bounds checking, and column resolution."""

from __future__ import annotations

import ast
import json
import logging
import sys
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
    """Verifies that multi-line units ending on a line with multiple statements
    without column coordinates fail closed, treating loads on the end line as downstream."""
    code = (
        "def worker():\n"
        "    a = 1\n"
        "    use(x); use(y)\n"
        "    print(res)\n"
    )
    # Multi-line unit covering lines 2 to 3 without column bounds fails closed
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"x", "y", "res"})
    assert reads is not None
    # Because end_col is None for ambiguous trailing statements,
    # fail-closed treats them as downstream reads
    assert "x" in reads
    assert "y" in reads
    assert "res" in reads


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
    assert scope["is_async"] is False
    assert "items" in scope["inputs"]
    assert "total" in scope["outputs"]


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


def test_parse_unit_coord_non_integral_floats_truncated_or_rejected() -> None:
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
    assert is_valid_unit_coordinates(
        {"start": 1, "end": 1, "start_col_offset": 10, "end_col_offset": 5}
    ) is False
    assert is_valid_unit_coordinates(
        {"start": 1, "end": 1, "start_col": 10, "end_col_offset": 5}
    ) is False
    assert is_valid_unit_coordinates({"start": 1, "end": 2, "start_col_offset": -1}) is False
    assert is_valid_unit_coordinates({"start": 1, "end": 2, "end_col_offset": -1}) is False
    assert is_valid_unit_coordinates(
        {"start": 1, "end": 2, "start_col_offset": 0, "end_col_offset": 10}
    ) is True
    assert is_valid_unit_coordinates(
        {"start": 1, "end": 2, "start_col": 0, "end_col_offset": 10}
    ) is True
    assert is_valid_unit_coordinates(
        {"start": 1, "end": 2, "start_col_offset": 0, "end_col": 10}
    ) is True
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


def test_parse_unit_coord_strict_rejects_negative_integers() -> None:
    """Verifies that parse_unit_coord with strict=True rejects negative integers."""
    with pytest.raises(ValueError, match="negative integer not allowed in strict mode"):
        parse_unit_coord({"start": -1}, "start", strict=True)

    with pytest.raises(ValueError, match="negative integer not allowed in strict mode"):
        parse_unit_coord({"start": -42}, "start", strict=True)

    # 0 is a valid offset (e.g. col_offset)
    assert parse_unit_coord({"col": 0}, "col", strict=True) == 0

    # In lenient mode, negative integers pass through
    assert parse_unit_coord({"start": -1}, "start", strict=False) == -1


def test_resolve_safe_unit_file_path_symlinked_ancestor_root(tmp_path: Path) -> None:
    """Verifies resolve_safe_unit_file_path handles symlinked ancestor roots
    (e.g., /var -> /private/var)."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_safe_unit_file_path,
    )

    private_dir = tmp_path / "private"
    real_var = private_dir / "var"
    real_var.mkdir(parents=True)
    real_repo = real_var / "my_project"
    real_repo.mkdir()
    target_py = real_repo / "main.py"
    target_py.write_text("print('symlink ancestor test')\n", encoding="utf-8")

    sym_var = tmp_path / "var"
    try:
        sym_var.symlink_to(real_var, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support directory symlinks or lacks privileges")

    sym_repo_file = sym_var / "my_project" / "main.py"
    # Even though sym_var in parent chain is a symlink, target is an ancestor of repo root
    resolved = resolve_safe_unit_file_path(
        {"file": str(sym_repo_file)}, repo_root=str(real_repo)
    )
    assert resolved == target_py.resolve()


def test_resolve_unit_line_bounds_defensive_non_dict() -> None:
    """Verifies that resolve_unit_line_bounds handles non-dict objects gracefully."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_unit_line_bounds,
    )

    assert resolve_unit_line_bounds(None) == (1, 1)  # type: ignore[arg-type]
    assert resolve_unit_line_bounds([1, 2, 3]) == (1, 1)  # type: ignore[arg-type]
    assert resolve_unit_line_bounds("not_a_dict") == (1, 1)  # type: ignore[arg-type]
    assert resolve_unit_line_bounds(123) == (1, 1)  # type: ignore[arg-type]


def test_resolve_safe_unit_file_path_relative_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies resolve_safe_unit_file_path handles relative repo_root paths."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_safe_unit_file_path,
    )

    repo_dir = tmp_path / "my_project"
    repo_dir.mkdir()
    target_py = repo_dir / "main.py"
    target_py.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    resolved = resolve_safe_unit_file_path(
        {"file": "main.py"}, repo_root="my_project"
    )
    assert resolved == target_py.resolve()


def test_get_valid_unit_bounds_defensive_non_dict() -> None:
    """Verifies _get_valid_unit_bounds handles None and non-dict inputs defensively."""
    assert _get_valid_unit_bounds(None) is None  # type: ignore[arg-type]
    assert _get_valid_unit_bounds("invalid") is None  # type: ignore[arg-type]
    assert _get_valid_unit_bounds([1, 2, 3]) is None  # type: ignore[arg-type]
    assert _get_valid_unit_bounds(123) is None  # type: ignore[arg-type]


def test_resolve_safe_unit_file_path_symlinked_root_in_repo_symlink(tmp_path: Path) -> None:
    """Verifies resolve_safe_unit_file_path rejects in-repo symlinks even when
    accessed via a symlinked root alias."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_safe_unit_file_path,
    )

    private_dir = tmp_path / "private"
    real_var = private_dir / "var"
    real_var.mkdir(parents=True)
    real_repo = real_var / "my_project"
    real_repo.mkdir()
    sub_dir = real_repo / "sub"
    sub_dir.mkdir()
    target_py = sub_dir / "target.py"
    target_py.write_text("print('target')\n", encoding="utf-8")

    sym_var = tmp_path / "var"
    try:
        sym_var.symlink_to(real_var, target_is_directory=True)
        link_sub = real_repo / "link_sub"
        link_sub.symlink_to(sub_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support directory symlinks or lacks privileges")

    # In-repo symlink accessed via symlink ancestor prefix: must return None
    aliased_file = sym_var / "my_project" / "link_sub" / "target.py"
    resolved = resolve_safe_unit_file_path(
        {"file": str(aliased_file)}, repo_root=str(real_repo)
    )
    assert resolved is None


def test_resolve_safe_unit_file_path_hop_back_symlink_rejected(tmp_path: Path) -> None:
    """Verifies resolve_safe_unit_file_path rejects in-repo symlinks that hop out
    and back to target_root."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_safe_unit_file_path,
    )

    real_repo = tmp_path / "repo"
    real_repo.mkdir()
    target_py = real_repo / "target.py"
    target_py.write_text("print('hello')\n", encoding="utf-8")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    hop_back = outside_dir / "hop_back"

    try:
        hop_back.symlink_to(real_repo, target_is_directory=True)
        link_out = real_repo / "link_out"
        link_out.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support directory symlinks or lacks privileges")

    hop_path = real_repo / "link_out" / "hop_back" / "target.py"
    resolved = resolve_safe_unit_file_path(
        {"file": str(hop_path)}, repo_root=str(real_repo)
    )
    assert resolved is None


def test_resolve_unit_line_bounds_strict_mode() -> None:
    """Verifies resolve_unit_line_bounds raises ValueError on malformed or inverted coordinates when strict=True."""
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        resolve_unit_line_bounds,
    )

    assert resolve_unit_line_bounds({"start": 2, "end": 5}, strict=True) == (2, 5)
    assert resolve_unit_line_bounds({"start": 3, "end": 3}, strict=True) == (3, 3)

    for bad_unit in (
        None,
        "not_a_dict",
        {},
        {"start": 3},
        {"start": None, "end": 2},
        {"start": 0, "end": 2},
        {"start": -1, "end": 2},
        {"start": 2, "end": 0},
        {"start": 2, "end": -3},
        {"start": 5, "end": 2},
        {"start": "2:0", "end": 5},
        {"start": 2.0, "end": 5},
        {"start": "invalid", "end": 5},
    ):
        with pytest.raises(ValueError):
            resolve_unit_line_bounds(bad_unit, strict=True)  # type: ignore[arg-type]


def test_is_sliced_unit_source_lines_rejects_malformed_and_inverted_coordinates() -> None:
    """Verifies is_sliced_unit_source_lines returns False instead of clamping to (1, 1) on invalid bounds."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _is_sliced_excerpt_for_file,
    )
    from pydoppelgangerhunt.source_lines import (  # pylint: disable=import-outside-toplevel
        is_sliced_unit_source_lines,
    )

    one_line = ["x = 1\n"]
    # Previously, clamping to (1, 1) caused 1-line source_lines to falsely match span == 1
    assert is_sliced_unit_source_lines({}, one_line) is False
    assert is_sliced_unit_source_lines({"start": None, "end": None}, one_line) is False
    assert is_sliced_unit_source_lines({"start": 0, "end": 0}, one_line) is False
    assert is_sliced_unit_source_lines({"start": -3, "end": -1}, one_line) is False
    assert is_sliced_unit_source_lines({"start": 10, "end": 5}, one_line) is False
    assert is_sliced_unit_source_lines({"start": "invalid", "end": 1}, one_line) is False
    assert is_sliced_unit_source_lines({"start": 1, "end": "invalid"}, one_line) is False
    assert is_sliced_unit_source_lines({"start": 1, "end": 1}, one_line) is True
    assert is_sliced_unit_source_lines({"start": 4, "end": 5}, ["a = 1\n", "b = 2\n"]) is True

    # _is_sliced_excerpt_for_file fails closed (treats malformed coordinates as a sliced excerpt)
    assert (
        _is_sliced_excerpt_for_file(
            {"start": 10, "end": 5, "source_lines": one_line},
            one_line,
        )
        is True
    )


def test_resolve_unit_ast_end_col_caches_stmts_by_start_lineno() -> None:
    """Verifies _get_scope_stmts_by_start_lineno indexes statements by start line and evicts cleanly."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access,import-outside-toplevel
        _evict_tree_scopes,
        _get_scope_stmts_by_start_lineno,
        _scope_stmts_by_end,
        _scope_stmts_by_start,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()
    code = (
        "def big_scope():\n"
        "    a = 1\n"
        "    b = 2; c = 3\n"
        "    d = (\n"
        "        4\n"
        "    )\n"
    )
    tree = ast.parse(code)
    fn_scope = tree.body[0]

    by_start = _get_scope_stmts_by_start_lineno(fn_scope)
    assert fn_scope in _scope_stmts_by_start
    assert len(by_start[2]) == 1
    assert len(by_start[3]) == 2
    assert len(by_start[4]) == 1

    # Single-statement line 2 resolves end_col; multi-statement line 3 without start_col fails closed
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 2}) == len("    a = 1")
    assert fn_scope in _scope_stmts_by_end
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 3}) is None
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 6}) == len("    )")

    _evict_tree_scopes(tree)
    assert fn_scope not in _scope_stmts_by_start
    assert fn_scope not in _scope_stmts_by_end


def test_extract_unit_body_lines_col0_comment_and_multiline_string() -> None:
    """Verifies _extract_unit_body_lines wraps in 'if 1:' when leading col-0 comment precedes indented def."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access,import-outside-toplevel
        _extract_unit_body_lines,
    )

    raw_lines = [
        "# leading comment at col 0",
        "    def method(self):",
        '        """Docstring."""',
        '        msg = """line1',
        'col0_continuation"""',
        "        return msg",
    ]
    extracted = _extract_unit_body_lines({"kind": "method"}, raw_lines)
    assert not any("def method" in ln for ln in extracted)
    assert not any("Docstring" in ln for ln in extracted)
    assert any("return msg" in ln for ln in extracted)


def test_resolve_unit_ast_end_col_multistatement_with_start_col_and_comp_target_order() -> None:
    """Verifies multi-statement start_col resolution and comprehension tuple target free-name extraction."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access,import-outside-toplevel
        _extract_helper_symbols,
    )

    code = (
        "def f():\n"
        "    x = 1\n"
        "    y = 2\n"
    )
    fn_scope = ast.parse(code).body[0]
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 2, "start_col": 4}) == len("    x = 1")
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 3, "start_col": 4}) == len("    y = 2")
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 3, "start_col": 8}) is None

    comp_bound = ast.parse("def _shared(items, a):\n    return [x for (x, a[x]) in items]\n")
    free_bound, _, _ = _extract_helper_symbols(comp_bound)
    assert "x" not in free_bound

    comp_free = ast.parse("def _shared(items, a):\n    return [x for (a[x], x) in items]\n")
    free_unbound, _, _ = _extract_helper_symbols(comp_free)
    assert "x" in free_unbound


def test_scope_intervals_col_offset_sig_and_lru_shrink_and_decorated_pragma() -> None:
    """Verifies col_offset signature invalidation, dynamic LRU shrinkage, and decorated pragma leading scan."""
    from collections import OrderedDict  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access,import-outside-toplevel
        _debug_ast_immutability,
        _evict_tree_scopes,
        _store_lru_cache_entry,
    )
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access,import-outside-toplevel
        _unit_has_no_refactor_pragma,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access,import-outside-toplevel
        _get_tree_scope_intervals,
    )

    tree = ast.parse("def f1(): pass\ndef f2(): pass\n")
    tree.body[0].lineno = tree.body[1].lineno = 1
    tree.body[0].end_lineno = tree.body[1].end_lineno = 1
    tree.body[0].col_offset = 0
    tree.body[1].col_offset = 15
    from pydoppelgangerhunt.fixer._df_cache import _debug_ast_immutability, evict_tree_scopes

    # Production mode refreshes intervals on AST mutation rather than raising
    intervals_prod = _get_tree_scope_intervals(tree)
    assert intervals_prod[0][3] is tree.body[1]
    tree.body[0].col_offset = 30
    intervals_prod_refreshed = _get_tree_scope_intervals(tree)
    assert intervals_prod_refreshed[0][3] is tree.body[0]
    evict_tree_scopes(tree)

    # Debug mode raises RuntimeError when a cached AST node is mutated in place
    tree.body[0].col_offset = 0
    _debug_ast_immutability[0] = True
    try:
        intervals1 = _get_tree_scope_intervals(tree)
        assert intervals1[0][3] is tree.body[1]
        tree.body[0].col_offset = 30
        with pytest.raises(RuntimeError, match="Cached AST node was mutated in-place"):
            _get_tree_scope_intervals(tree)
    finally:
        _debug_ast_immutability[0] = False
        evict_tree_scopes(tree)
    prev_dbg = _debug_ast_immutability[0]
    _debug_ast_immutability[0] = False
    try:
        assert _get_tree_scope_intervals(tree)[0][3] is tree.body[0]
    finally:
        _debug_ast_immutability[0] = prev_dbg
    _evict_tree_scopes(tree)
    intervals2 = _get_tree_scope_intervals(tree)
    assert intervals2[0][3] is tree.body[0]

    cache: OrderedDict[str, int] = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    _store_lru_cache_entry(cache, "d", 4, max_size=2)
    assert list(cache.keys()) == ["c", "d"]

    decorated_src = (
        "# pragma: no-refactor\n"
        "\n"
        "@dec\n"
        "def fn():\n"
        "    return '# pragma: no-refactor'\n"
    )
    assert _unit_has_no_refactor_pragma({"start": 4, "end": 5}, None, orig_text=decorated_src) is False
    decorated_pragma_src = (
        "# pragma: no-refactor\n"
        "@dec\n"
        "def fn():\n"
        "    return 1\n"
    )
    assert _unit_has_no_refactor_pragma({"start": 3, "end": 4}, None, orig_text=decorated_pragma_src) is True


def test_comp_tuple_target_left_to_right_and_patch_receiver_kind(tmp_path: Any) -> None:
    """Verifies left-to-right comprehension tuple target unpacking and receiver_kind in generate_refactoring_patch."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access,import-outside-toplevel
        _extract_unit_live_in_names,
    )
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    fn1_tree = ast.parse("def f(items, arr):\n    return [1 for (arr[x], x) in items]\n")
    fn2_tree = ast.parse("def f(items, arr):\n    return [1 for (x, arr[x]) in items]\n")
    assert "x" in _extract_unit_live_in_names(fn1_tree.body[0], 2, 2)
    assert "x" not in _extract_unit_live_in_names(fn2_tree.body[0], 2, 2)

    s_before = inspect_single_unit_scope({"start": 1, "end": 1, "source_text": "[1 for (arr[x], x) in items]\n"})
    s_after = inspect_single_unit_scope({"start": 1, "end": 1, "source_text": "[1 for (x, arr[x]) in items]\n"})
    assert "x" in s_before["inputs"]
    assert "x" not in s_after["inputs"]

    cls_src = (
        "class Service:\n"
        "    @classmethod\n"
        "    def first(mcls, val: int) -> int:\n"
        "        base = mcls.DEFAULT_BASE + val\n"
        "        scaled = base * 2\n"
        "        return scaled + 1\n"
        "\n"
        "    @classmethod\n"
        "    def second(mcls, val: int) -> int:\n"
        "        base = mcls.DEFAULT_BASE + val\n"
        "        scaled = base * 2\n"
        "        return scaled + 1\n"
    )
    f_path = tmp_path / "svc.py"
    f_path.write_text(cls_src, encoding="utf-8")
    u1 = {"file": "svc.py", "start": 3, "end": 6, "name": "Service.first", "kind": "method"}
    u2 = {"file": "svc.py", "start": 9, "end": 12, "name": "Service.second", "kind": "method"}
    patch_text = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path))
    assert "@classmethod" in patch_text


def test_sliced_method_type_params_and_toml_fallback(tmp_path: Any, monkeypatch: Any) -> None:
    """Verifies sliced method has_class_scoped_type_params, _strip_stmt_target_lines, and TOML fallback."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.config import (  # pylint: disable=import-outside-toplevel
        load_toml_section,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access,import-outside-toplevel
        _strip_stmt_target_lines,
    )

    if sys.version_info >= (3, 12):
        sliced_method = {
            "start": 10,
            "end": 11,
            "kind": "method",
            "source_lines": [
                "    def compute[U: BoundT](self, x: U) -> U:\n",
                "        return x\n",
            ],
        }
        scope = inspect_single_unit_scope(sliced_method)
        assert scope["has_class_scoped_type_params"] is True

    assert _strip_stmt_target_lines(["res = a + b"]) == ["a + b"]

    toml_file = tmp_path / "cfg.toml"
    toml_file.write_text(
        '[tool.pydoppelgangerhunt]\n"closure_strictness" = "lenient"\nskip_pre_unit_closures = true\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    parsed = load_toml_section(toml_file, "pydoppelgangerhunt")
    assert parsed["closure_strictness"] == "lenient"
    assert parsed["skip_pre_unit_closures"] is True


def test_precomputed_scope_in_synthesize_and_nested_nonlocal_shadowing() -> None:
    """Verifies precomputed scope reuse in synthesize_shared_helper_code and nested nonlocal store scoping."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access,import-outside-toplevel
        _build_unit_delegation_call,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        synthesize_shared_helper_code,
    )

    shadowed = inspect_single_unit_scope(
        {
            "start": 1,
            "end": 4,
            "source_text": (
                "nonlocal x\n"
                "def inner():\n"
                "    x = 1\n"
                "    return x\n"
            ),
        }
    )
    assert "x" not in shadowed["outputs"]

    bound_inner = inspect_single_unit_scope(
        {
            "start": 1,
            "end": 5,
            "source_text": (
                "def inner():\n"
                "    nonlocal x\n"
                "    x = 1\n"
                "    return x\n"
                "inner()\n"
            ),
        }
    )
    assert "x" in bound_inner["outputs"]

    u1 = {"start": 1, "end": 2, "source_text": "def f1(a):\n    return a + 1\n"}
    u2 = {"start": 1, "end": 2, "source_text": "def f2(a):\n    return a + 1\n"}
    s1 = inspect_single_unit_scope(u1)
    s2 = inspect_single_unit_scope(u2)
    merged = analyze_unit_variable_scope(u1, u2, scope1=s1, scope2=s2)
    res = synthesize_shared_helper_code(u1, u2, scope1=s1, scope2=s2, scope=merged)
    assert "def " in res

    expr_src = "out = [x + 1 for x in items]  # type: ignore\n"
    expr_unit = {"start": 1, "end": 1, "kind": "comprehension"}
    rep_without = _build_unit_delegation_call(
        expr_unit,
        expr_src,
        [expr_src],
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["items"],
        outputs=[],
        scope={"param_details": []},
        preserve_boundary_pragmas=False,
    )
    rep_with = _build_unit_delegation_call(
        expr_unit,
        expr_src,
        [expr_src],
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["items"],
        outputs=[],
        scope={"param_details": []},
        preserve_boundary_pragmas=True,
    )
    assert "# type: ignore" not in rep_without
    assert "# type: ignore" in rep_with


def test_pep695_typing_imports_class_scope_isolation_and_multiline_indent_error() -> None:
    """Verifies PEP 695 typing imports, class body lexical scope isolation, and multiline string IndentationError retry."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access,import-outside-toplevel
        _find_multiline_string_continuation_lines,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access,import-outside-toplevel
        _extract_required_typing_imports,
    )

    if sys.version_info >= (3, 12):
        imports = _extract_required_typing_imports(
            "def _shared_fn[T: Sequence[int], Sequence: int](x: T) -> T:\n    return x\n"
        )
        assert imports == []
        imports2 = _extract_required_typing_imports(
            "def _shared_fn[T: Sequence[int]](x: T) -> T:\n    return x\n"
        )
        assert imports2 == ["Sequence"]

    cls_unit = {
        "start": 1,
        "end": 6,
        "source_text": (
            "class HelperBox:\n"
            "    ext_fn = 1\n"
            "    direct_use = ext_fn + 1\n"
            "    def run(self):\n"
            "        return ext_fn(self)\n"
        ),
    }
    cls_scope = inspect_single_unit_scope(cls_unit)
    assert "ext_fn" in cls_scope["inputs"]
    assert "direct_use" not in cls_scope["inputs"]

    unaligned = (
        "    if cond:\n"
        "        x = 1\n"
        '  y = """first\n'
        "    second\n"
        'third"""\n'
    )
    assert _find_multiline_string_continuation_lines(unaligned) == {4, 5}


def test_lhs_receiver_pruning_and_precomputed_scope_pruning() -> None:
    """Verifies whole-line expression LHS receiver pruning and precomputed scope LHS pruning."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access,import-outside-toplevel
        _order_inputs_by_rhs_loads,
        synthesize_shared_helper_code,
    )

    # 1.2: _order_inputs_by_rhs_loads pins receivers to index 0 when receiver_names is provided
    rhs_tree = ast.parse("fn(self)").body[0].value  # type: ignore[attr-defined]
    assert _order_inputs_by_rhs_loads(["fn", "self"], rhs_tree, ("self",)) == [
        "self",
        "fn",
    ]
    assert _order_inputs_by_rhs_loads(["fn", "self"], rhs_tree, ()) == [
        "fn",
        "self",
    ]

    # 1.1: Asymmetrical receiver binding after LHS pruning rejected in patch generation
    src = (
        "class Box:\n"
        "    def m1(self, fn):\n"
        "        self.res1 = [fn(self) for _ in range(3)]\n"
        "    def m2(self, fn, other):\n"
        "        self.res2 = [fn(other) for _ in range(3)]\n"
    )
    u1 = {
        "kind": "comprehension",
        "start": 3,
        "end": 3,
        "enclosing_class": "Box",
        "receiver_param": "self",
        "receiver_kind": "instance",
        "source_text": src,
    }
    u2 = {
        "kind": "comprehension",
        "start": 5,
        "end": 5,
        "enclosing_class": "Box",
        "receiver_param": "self",
        "receiver_kind": "instance",
        "source_text": src,
    }
    assert (
        generate_refactoring_patch([(0.95, u1, u2)], method_binding="auto") == ""
    )
    assert (
        generate_refactoring_patch([(0.95, u1, u2)], method_binding="method") == ""
    )

    # 1.2: Symmetrical non-attribute RHS receiver argument does not swap or drop arguments
    sym_src = (
        "class Box:\n"
        "    def m1(self, fn):\n"
        "        self.res1 = [fn(self) for _ in range(3)]\n"
        "    def m2(self, fn):\n"
        "        self.res2 = [fn(self) for _ in range(3)]\n"
    )
    su1 = dict(u1, file="box.py", name="m1_comp", source_text=sym_src)
    su2 = dict(u2, file="box.py", name="m2_comp", source_text=sym_src)
    sym_helper_mod = synthesize_shared_helper_code(su1, su2, method_binding="module")
    assert "def _shared_m1_comp_m2_comp(self: Any, fn: Any) -> Any:" in sym_helper_mod
    sym_helper_meth = synthesize_shared_helper_code(su1, su2, method_binding="method")
    assert "def _shared_m1_comp_m2_comp(self, fn: Any) -> Any:" in sym_helper_meth

    # 2.3: Precomputed unpruned scopes passed directly to synthesize_shared_helper_code
    comp_src1 = "def f1(d1, items):\n    d1[0] = [x * 2 for x in items]\n"
    comp_src2 = "def f2(d2, items):\n    d2[0] = [x * 2 for x in items]\n"
    cu1 = {
        "kind": "comprehension",
        "start": 2,
        "end": 2,
        "source_text": comp_src1,
    }
    cu2 = {
        "kind": "comprehension",
        "start": 2,
        "end": 2,
        "source_text": comp_src2,
    }
    s1 = inspect_single_unit_scope(cu1)
    s2 = inspect_single_unit_scope(cu2)
    merged = analyze_unit_variable_scope(cu1, cu2)
    helper = synthesize_shared_helper_code(
        cu1, cu2, scope1=s1, scope2=s2, scope=merged
    )
    assert "def _shared_helper(items: Any) -> Any:" in helper
    assert "d1" not in helper and "d2" not in helper


def test_class_scope_headers_comprehensions_helper_symbols_and_global_depth() -> None:
    """Verifies class scope header defaults, comprehension iterators, helper symbol resolution, and nested globals."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access,import-outside-toplevel
        _resolve_unit_ast_end_col,
    )
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access,import-outside-toplevel
        _ClassScopeVisitor,
    )
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access,import-outside-toplevel
        _extract_helper_symbols,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access,import-outside-toplevel
        _find_multiline_string_continuation_lines,
        is_subroutine_unit,
    )

    # 1.2: Class attribute used in method default is not an escaping free read
    cls_tree = ast.parse(
        "class Counter:\n"
        "    total = 0\n"
        "    def reset(self, val=total):\n"
        "        return val\n"
    )
    cls_vis = _ClassScopeVisitor()
    cls_vis.visit(cls_tree.body[0])
    assert "total" not in cls_vis.free_reads

    # 1.3: Class body comprehension outermost iter sees class attributes
    cls_comp_unit = {
        "start": 1,
        "end": 3,
        "source_text": (
            "class Config:\n"
            "    ITEMS = [1, 2, 3]\n"
            "    DOUBLED = [x * 2 for x in ITEMS]\n"
        ),
    }
    comp_scope = inspect_single_unit_scope(cls_comp_unit)
    assert "ITEMS" not in comp_scope["inputs"]

    # 2.1: _extract_helper_symbols does not shadow method calls with class attributes
    helper_code = (
        "def _shared_fn():\n"
        "    class Box:\n"
        "        foo = 1\n"
        "        bar = foo + 1\n"
        "        def m(self):\n"
        "            return foo()\n"
        "    return Box()\n"
    )
    free_names, _, _ = _extract_helper_symbols(ast.parse(helper_code))
    assert "foo" in free_names

    if sys.version_info >= (3, 12):
        ta_code = (
            "type TopAlias[T] = list[T]\n"
            "def _shared_ta():\n"
            "    type InnerAlias[U] = dict[str, ExtType]\n"
            "    return InnerAlias\n"
        )
        ta_free, ta_defined, _ = _extract_helper_symbols(ast.parse(ta_code))
        assert "ExtType" in ta_free
        assert "TopAlias" in ta_defined
        assert "InnerAlias" in ta_defined

    # 2.2: Nested global declaration does not leak to outer unit or nested function
    nested_global_unit = {
        "start": 1,
        "end": 4,
        "source_text": (
            "def inner():\n"
            "    global x\n"
            "    x = 1\n"
            "x = 2\n"
        ),
    }
    ng_scope = inspect_single_unit_scope(nested_global_unit)
    assert "x" in ng_scope["outputs"]

    outer_global_unit = {
        "start": 1,
        "end": 5,
        "source_text": (
            "global y\n"
            "def inner():\n"
            "    y = 2\n"
            "    return y\n"
            "z = inner()\n"
        ),
    }
    og_scope = inspect_single_unit_scope(outer_global_unit)
    assert "y" not in og_scope["inputs"]

    # 2.4: is_subroutine_unit honors pre-parsed tree without source_text
    parsed_fn = ast.parse("def f():\n    x = 1\n")
    assert is_subroutine_unit({"start": 2, "end": 2}, tree=parsed_fn) is True

    # _resolve_unit_ast_end_col handles preceding multi-line statement ending on u_start
    ml_tree = ast.parse("x = 0; a = (\n    1\n); b = 2\n")
    assert (
        _resolve_unit_ast_end_col(ml_tree, {"start": 3, "end": 3, "start_col": 3})
        == 8
    )
    assert (
        _resolve_unit_ast_end_col(ml_tree, {"start": 3, "end": 3, "start_col": 0})
        is None
    )

    # 3.1: Single-line f-string with multi-line or unclosed interpolation expression
    fstr_src = 'msg = f"val: {\n    compute(x)\n}"\n'
    assert _find_multiline_string_continuation_lines(fstr_src) == set()
    unclosed_single_fstr = 'msg = f"val: {x\ny = 1\n'
    assert _find_multiline_string_continuation_lines(unclosed_single_fstr) == set()


def test_positional_meta2_internal_nonlocal_and_global_type_param_bounds() -> None:
    """Verifies positional meta2 lookup, internal nonlocal scoping, enclosing global PEP 695 bounds, and multiline end_col."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access,import-outside-toplevel
        _evict_tree_scopes,
        _get_or_parse_tree,
        _resolve_unit_ast_end_col,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access,import-outside-toplevel
        _get_tree_scope_intervals,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access,import-outside-toplevel
        _format_helper_parameters,
    )

    # 1.1: Positional parameter metadata does not fall back to meta2[var] when inputs2[idx] != var
    params = _format_helper_parameters(
        inputs=["a", "b"],
        meta1={"a": {"type": "int", "kind": "pos"}, "b": {"type": "int", "kind": "pos"}},
        meta2={"a": {"type": "list[int]", "kind": "pos"}, "b": {"type": "int", "kind": "pos"}},
        effective_binding="module",
        is_static_clone=False,
        is_class_receiver=False,
        type_merge_strategy="union",
        inputs2=["ext_val", "b"],
        positional_target_map=True,
    )
    assert params == ["a: int", "b: int"]

    # 1.2: Nested nonlocal referencing outer local in unit does not escape as unit nonlocal
    fn_unit = {
        "kind": "function",
        "start": 1,
        "end": 7,
        "source_text": (
            "def compute(a: int) -> int:\n"
            "    acc = a\n"
            "    def step(d: int) -> None:\n"
            "        nonlocal acc\n"
            "        acc += d\n"
            "    step(5)\n"
            "    return acc\n"
        ),
    }
    fn_scope = inspect_single_unit_scope(fn_unit)
    assert fn_scope["nonlocals"] == []
    assert fn_scope["inputs"] == ["a"]

    # 1.3: Enclosing function global declaration does not trigger PEP 695 type_param_closure_capture
    if sys.version_info >= (3, 12):
        tp_src = (
            "BoundType = object\n"
            "def outer():\n"
            "    global BoundType\n"
            "    BoundType = int\n"
            "    def inner[T: BoundType](x: T) -> T:\n"
            "        return x\n"
            "    return inner\n"
        )
        tp_unit = {
            "kind": "closure",
            "start": 5,
            "end": 6,
            "source_text": tp_src,
        }
        tp_scope = inspect_single_unit_scope(tp_unit)
        assert "type_param_closure_capture" not in tp_scope["control_flow_hazards"]

    # 2.1 & 2.2: _get_tree_scope_intervals invalidation and multi-line statement _resolve_unit_ast_end_col
    cached_tree = _get_or_parse_tree("def f1(): pass\ndef f2(): pass\n")
    assert cached_tree is not None
    intervals1 = _get_tree_scope_intervals(cached_tree)
    orig_col = cached_tree.body[0].col_offset  # type: ignore[attr-defined]
    try:
        cached_tree.body[0].col_offset = 99  # type: ignore[attr-defined]
        _evict_tree_scopes(cached_tree)
        intervals2 = _get_tree_scope_intervals(cached_tree)
        assert intervals2 is not intervals1
    finally:
        cached_tree.body[0].col_offset = orig_col  # type: ignore[attr-defined]
        _evict_tree_scopes(cached_tree)

    multi_stmt_tree = ast.parse("if True:\n    x = (\n        1 + 2\n    )\n")
    assert (
        _resolve_unit_ast_end_col(
            multi_stmt_tree, {"start": 2, "end": 4, "start_col": 4}
        )
        == 5
    )


def test_multilevel_nonlocal_harvested_expr_receiver_and_delegation_indent(
    tmp_path: Path,
) -> None:
    """Verifies multi-level nested nonlocals, harvested expr receiver_kind=None, downstream cache key, and delegation indent."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        collect_downstream_read_names,
    )
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access,import-outside-toplevel
        _build_unit_delegation_call,
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        synthesize_shared_helper_code,
    )

    # 1.1: Multi-level nested nonlocal does not leak intermediate function local into outer unit stores/deleted_names
    ml_nonlocal_unit = {
        "kind": "compound_block",
        "start": 1,
        "end": 9,
        "source_text": (
            "def level1():\n"
            "    acc = 0\n"
            "    def level2():\n"
            "        nonlocal acc\n"
            "        acc += 1\n"
            "        del acc\n"
            "    level2()\n"
            "    return 1\n"
            "res = level1()\n"
        ),
    }
    ml_scope = inspect_single_unit_scope(ml_nonlocal_unit)
    assert ml_scope["outputs"] == ["level1", "res"]
    ml_del_unit = {
        "kind": "compound_block",
        "start": 1,
        "end": 9,
        "source_text": (
            "acc = 5\n"
            "def level1():\n"
            "    acc = 0\n"
            "    def level2():\n"
            "        nonlocal acc\n"
            "        del acc\n"
            "    level2()\n"
            "    return 1\n"
            "res = level1()\n"
        ),
    }
    assert "acc" in inspect_single_unit_scope(ml_del_unit)["outputs"]

    # 1.2: Harvested comprehension units with receiver_kind=None inside same class with method_binding="method"
    f = tmp_path / "harvested_comp.py"
    f.write_text(
        "class Worker:\n"
        "    def a(self, items: list[int]) -> None:\n"
        "        self.r1 = [x * 2 for x in items]\n"
        "    def b(self, items: list[int]) -> None:\n"
        "        self.r2 = [x * 2 for x in items]\n",
        encoding="utf-8",
    )
    hu1 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "a_comp",
        "kind": "comprehension",
        "receiver_kind": None,
        "receiver_param": None,
    }
    hu2 = {
        "file": str(f),
        "start": 5,
        "end": 5,
        "name": "b_comp",
        "kind": "comprehension",
        "receiver_kind": None,
        "receiver_param": None,
    }
    h_code = synthesize_shared_helper_code(
        hu1, hu2, method_binding="method", repo_root=str(tmp_path)
    )
    assert "def _shared_a_comp_b_comp(self, items: Any) -> Any:" in h_code
    patch = generate_refactoring_patch(
        [(0.98, hu1, hu2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert "+    def _shared_a_comp_b_comp(self, items: Any) -> Any:" in patch
    assert "+        self.r1 = self._shared_a_comp_b_comp(items)" in patch

    # 2.1: collect_downstream_read_names distinguishes closure vs coinciding subroutine block
    coincide_src = (
        "def outer():\n"
        "    def inner():\n"
        "        val = 1\n"
        "        return val\n"
        "    return inner()\n"
    )
    closure_reads = collect_downstream_read_names(
        coincide_src, {"kind": "closure", "start": 2, "end": 4}, {"inner"}
    )
    sub_reads = collect_downstream_read_names(
        coincide_src, {"kind": "compound_block", "start": 2, "end": 4}, {"inner"}
    )
    assert closure_reads == set()
    assert sub_reads == {"inner"}

    # 2.2: Leading col-0 comment on multi-line unit preserves statement indentation in _build_unit_delegation_call
    comment_src = (
        "def run(x):\n"
        "# col-0 note\n"
        "    y = x + 1\n"
        "    return y\n"
    )
    call_str = _build_unit_delegation_call(
        {"kind": "compound_block", "start": 2, "end": 3},
        comment_src,
        comment_src.splitlines(keepends=True),
        effective_binding="module",
        helper_name="_shared_run",
        inputs=["x"],
        outputs=["y"],
        scope={"param_details": []},
    )
    assert call_str.startswith("    y = _shared_run(x)")


@pytest.mark.parametrize(
    ("source", "target_fn_only", "unit", "expected"),
    [
        pytest.param("if cond:\n    x = 1\n    y = 2\n", False, {"start": 1, "end": 3, "start_col": 0}, 9, id="compound_if_end_col"),
        pytest.param("if cond:\n    x = 1\n    y = 2\n", False, {"start": 1, "end": 3, "start_col": 4}, None, id="compound_if_mismatched_col"),
        pytest.param("def f(x):\n    a = x + 1\n    b = a * 2\n", True, {"start": 2, "end": 3, "start_col": 4}, 13, id="multiline_sub_range"),
        pytest.param("def f():\n    a = 1; b = (\n        2\n    )\n", True, {"start": 2, "end": 4, "start_col": 4}, 5, id="semicolon_multiline_tail"),
    ],
)
def test_resolve_unit_ast_end_col_multiline_and_semicolon_cases(
    source: str,
    target_fn_only: bool,
    unit: Dict[str, Any],
    expected: Optional[int],
) -> None:
    """Verify _resolve_unit_ast_end_col handles multi-line blocks, sub-ranges, and semicolon continuations."""
    tree = ast.parse(source)
    target: ast.AST = tree.body[0] if target_fn_only else tree
    assert _resolve_unit_ast_end_col(target, unit) == expected


def test_extract_single_node_local_names_class_vs_function_walrus_and_bare_annotations() -> None:
    """Verify ClassDef isolates comprehension walrus and bare AnnAssign while FunctionDef binds them."""
    from pydoppelgangerhunt.fixer._df_closures import _extract_single_node_local_names

    cls_locals = _extract_single_node_local_names(
        ast.parse("class C:\n    vals = [w := v for v in items]\n").body[0]
    )
    assert "vals" in cls_locals
    assert "w" not in cls_locals

    fn_locals = _extract_single_node_local_names(
        ast.parse("def f(items):\n    vals = [w := v for v in items]\n").body[0]
    )
    assert "vals" in fn_locals
    assert "w" in fn_locals

    ann_locals = _extract_single_node_local_names(
        ast.parse("class C:\n    bare_x: int\n    assigned_y: int = 1\n").body[0]
    )
    assert "bare_x" not in ann_locals
    assert "assigned_y" in ann_locals


def test_toml_config_top_validation_and_headerless_multiline_arrays(tmp_path: Path) -> None:
    """Verify CLI top config validation and zero-dependency TOML multiline array parsing."""
    from pydoppelgangerhunt.cli import main as cli_main
    from pydoppelgangerhunt.config import load_toml_section, load_tool_config

    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text("top = 5\n", encoding="utf-8")
    assert load_tool_config(config_path=str(cfg_file)).get("top") == 5

    for invalid_top in ("0", "2.5"):
        bad_cfg = tmp_path / "bad_top.toml"
        bad_cfg.write_text(f"[tool.pydoppelgangerhunt]\ntop = {invalid_top}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            cli_main([str(tmp_path), "--config", str(bad_cfg)])

    hdrless_toml = tmp_path / "headerless_arr.toml"
    hdrless_toml.write_text('exclude = [\n  ["a", "b"]\n]\nthreshold = 0.85\n', encoding="utf-8")
    sec_toml = tmp_path / "sec_multiline.toml"
    sec_toml.write_text(
        '[tool.other]\nmatrix = [\n  "[tool.pydoppelgangerhunt]"\n]\n'
        '[tool.pydoppelgangerhunt]\nexclude = [\n  ["x", "y"]\n]\nthreshold = 0.91\n',
        encoding="utf-8",
    )
    orig_tomllib = sys.modules.get("tomllib")
    orig_tomli = sys.modules.get("tomli")
    try:
        sys.modules["tomllib"] = None  # type: ignore[assignment]
        sys.modules["tomli"] = None  # type: ignore[assignment]
        parsed_hdrless = load_toml_section(str(hdrless_toml), "pydoppelgangerhunt")
        parsed_sec = load_toml_section(str(sec_toml), "pydoppelgangerhunt")
    finally:
        if orig_tomllib is not None:
            sys.modules["tomllib"] = orig_tomllib
        else:
            sys.modules.pop("tomllib", None)
        if orig_tomli is not None:
            sys.modules["tomli"] = orig_tomli
        else:
            sys.modules.pop("tomli", None)

    assert parsed_hdrless.get("threshold") == 0.85
    assert parsed_hdrless.get("exclude") == [["a", "b"]]
    assert parsed_sec.get("exclude") == [["x", "y"]]
    assert parsed_sec.get("threshold") == 0.91


def test_extract_helper_symbols_class_walrus_globals_wildcards_and_del() -> None:
    """Verify helper symbol extraction across class scoping, comprehension walrus, global, wildcard *, and del."""
    from pydoppelgangerhunt.fixer.patch import _extract_helper_symbols

    cls_shadow_code = (
        "class _H:\n"
        "    x = outer_sym\n"
        "    outer_sym = 1\n"
        "    @deco\n"
        "    def deco(self, a: outer_sym) -> None:\n"
        "        pass\n"
    )
    free_cls, def_cls, _ = _extract_helper_symbols(ast.parse(cls_shadow_code))
    assert "outer_sym" in free_cls
    assert "deco" in free_cls
    assert {"x", "outer_sym", "deco"} <= def_cls

    comp_walrus_code = "def _h(items):\n    vals = [w := v + 1 for v in items]\n    return w, vals\n"
    free_cw, _, _ = _extract_helper_symbols(ast.parse(comp_walrus_code))
    assert "w" not in free_cw

    global_helper_src = (
        "def _shared_helper():\n"
        "    def inner():\n"
        "        return X\n"
        "    global X\n"
        "    X = 1\n"
        "    return inner()\n"
    )
    g_free, g_def, _ = _extract_helper_symbols(ast.parse(global_helper_src))
    assert "X" in g_free
    assert "X" not in g_def

    star_free, star_def, star_imp = _extract_helper_symbols(
        ast.parse("def _shared_fn():\n    from math import *\n    return 1\n")
    )
    assert "*" not in star_free
    assert "*" not in star_def
    assert "*" not in star_imp

    del_free, _, _ = _extract_helper_symbols(ast.parse("def _shared_del():\n    del ExternalSym\n"))
    assert "ExternalSym" in del_free


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires match/case statement syntax")
def test_extract_helper_symbols_match_case_bindings_and_wildcards() -> None:
    """Verify helper symbol extraction handles match/case bindings, '_' wildcard targets, and bare AnnAssign."""
    from pydoppelgangerhunt.fixer.patch import _extract_helper_symbols

    match_helper_src = (
        "class C:\n"
        "    bare_attr: OuterType\n"
        "    val = bare_attr\n"
        "    match cmd:\n"
        "        case [first, *rest]:\n"
        "            out = first + len(rest)\n"
        "        case {'k': mapped, **extra}:\n"
        "            out = mapped + len(extra)\n"
    )
    m_free, _, _ = _extract_helper_symbols(ast.parse(match_helper_src))
    for bound in ("first", "rest", "mapped", "extra"):
        assert bound not in m_free
    assert "bare_attr" in m_free
    assert "OuterType" in m_free

    match_wc_tree = ast.parse(
        "def _shared_match(val):\n"
        "    match val:\n"
        "        case {'k': x, **rest}:\n"
        "            return _\n"
    )
    match_node = match_wc_tree.body[0].body[0]  # type: ignore[attr-defined]
    match_node.cases[0].pattern.rest = "_"
    wc_free, wc_def, _ = _extract_helper_symbols(match_wc_tree)
    assert "_" in wc_free
    assert "_" not in wc_def


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Requires PEP 695 type parameter syntax")
def test_extract_helper_symbols_pep695_generic_method_annotations() -> None:
    """Verify class-scoped annotations and PEP 695 type params are bound inside generic methods."""
    from pydoppelgangerhunt.fixer.patch import _extract_helper_symbols

    generic_cls_code = (
        "class _Helper:\n"
        "    Attr = int\n"
        "    def m[T: Attr](self, x: Attr) -> T:\n"
        "        return x\n"
    )
    free_syms, _, _ = _extract_helper_symbols(ast.parse(generic_cls_code))
    assert "Attr" not in free_syms
    assert "T" not in free_syms


def test_outer_expr_visitor_skips_walrus_store_targets() -> None:
    """Verify _OuterExprVisitor visits walrus values without recording store targets as external reads."""
    from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor

    outer_vis = _OuterExprVisitor()
    outer_vis.visit(ast.parse("(w := val + 1)", mode="eval"))
    assert outer_vis.ordered_names == ["val"]

    walrus_expr_v = _OuterExprVisitor()
    walrus_expr_v.visit(ast.parse("((w := transform(x)) + w for x in items)").body[0])
    assert "w" not in walrus_expr_v.names
    assert walrus_expr_v.ordered_names == ["items", "transform"]


@pytest.mark.parametrize(
    ("source", "unit_span", "candidate_names"),
    [
        pytest.param(
            "def outer():\n    x = 1\n    class C:\n        a = x + 1\n        x = 99\n"
            "        def m(self):\n            return x\n    return C\n",
            {"kind": "compound_block", "start": 4, "end": 4},
            {"x", "a"},
            id="class_attr_kill_before_method",
        ),
        pytest.param(
            "def outer():\n    x = 1\n    class C:\n        a = x + 1\n        x = 99\n"
            "        vals = [x for _ in range(1)]\n    return C\n",
            {"kind": "compound_block", "start": 4, "end": 4},
            {"x", "a"},
            id="class_attr_kill_before_comp",
        ),
        pytest.param(
            "def outer(x):\n    for _ in range(2):\n        class C:\n            x += 1\n"
            "            x = 99\n            vals = [x for _ in range(1)]\n    return None\n",
            {"kind": "compound_block", "start": 4, "end": 4},
            {"x"},
            id="outer_param_class_loop",
        ),
        pytest.param(
            "def outer(x):\n    for _ in range(2):\n        class C:\n            vals = [x for _ in range(1)]\n"
            "            x = 99\n        x += 1\n",
            {"kind": "compound_block", "start": 6, "end": 6},
            {"x"},
            id="loop_carried_class_read",
        ),
        pytest.param(
            "def outer(x):\n    x = x + 1\n    class C:\n        for x in x:\n            pass\n    return C\n",
            {"start": 2, "end": 2},
            {"x"},
            id="class_for_iter_reads_outer",
        ),
        pytest.param(
            "def outer(x):\n    x = x + 1\n    class C:\n        x: int\n        z = x\n    return C\n",
            {"start": 2, "end": 2},
            {"x"},
            id="class_bare_annotation_not_kill",
        ),
    ],
)
def test_downstream_read_visitor_isolates_class_attributes_and_loops(
    source: str,
    unit_span: Dict[str, Any],
    candidate_names: set[str],
) -> None:
    """Verify downstream read tracking isolates class-attribute kills from outer/method/comprehension reads."""
    ds_reads = collect_downstream_read_names(
        source, unit_span, candidate_names, tree=ast.parse(source)
    )
    assert ds_reads is not None and "x" in ds_reads


def test_branch_aware_deleted_outputs_and_pair_resolution(tmp_path: Path) -> None:
    """Verify branch-symmetric del propagation marks deleted_output and blocks unsafe pairing."""
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    del_branch_a = (
        "def f(cond):\n    if cond:\n        x = 1\n        del x\n    else:\n        x = 2\n    return 0\n"
    )
    del_branch_b = (
        "def f(cond):\n    if not cond:\n        x = 2\n    else:\n        x = 1\n        del x\n    return 0\n"
    )
    del_file_a = tmp_path / "del_a.py"
    del_file_b = tmp_path / "del_b.py"
    del_file_a.write_text(del_branch_a, encoding="utf-8")
    del_file_b.write_text(del_branch_b, encoding="utf-8")
    u_del_a = {"kind": "if_block", "start": 2, "end": 6, "file": str(del_file_a)}
    u_del_b = {"kind": "if_block", "start": 2, "end": 6, "file": str(del_file_b)}
    sc_del_a = inspect_single_unit_scope(u_del_a, repo_root=str(tmp_path), tree=ast.parse(del_branch_a))
    sc_del_b = inspect_single_unit_scope(u_del_b, repo_root=str(tmp_path), tree=ast.parse(del_branch_b))
    assert sc_del_a["outputs"] == sc_del_b["outputs"] == ["x"]
    assert sc_del_a["conditional_outputs"] == sc_del_b["conditional_outputs"] == ["x"]
    assert "deleted_output" in sc_del_a["control_flow_hazards"]
    assert "deleted_output" in sc_del_b["control_flow_hazards"]
    assert not sc_del_a["is_control_flow_safe"]
    assert not sc_del_b["is_control_flow_safe"]
    assert resolve_clone_pair_outputs(u_del_a, u_del_b, sc_del_a, sc_del_b, del_branch_a, del_branch_b) is None

    del_input_a = "def g(x, cond):\n    if cond:\n        del x\n    else:\n        x = 2\n"
    del_input_b = "def g(x, cond):\n    if not cond:\n        x = 2\n    else:\n        del x\n"
    del_in_fa = tmp_path / "del_in_a.py"
    del_in_fb = tmp_path / "del_in_b.py"
    del_in_fa.write_text(del_input_a, encoding="utf-8")
    del_in_fb.write_text(del_input_b, encoding="utf-8")
    sc_in_a = inspect_single_unit_scope(
        {"kind": "if_block", "start": 2, "end": 5, "file": str(del_in_fa)},
        repo_root=str(tmp_path),
        tree=ast.parse(del_input_a),
    )
    sc_in_b = inspect_single_unit_scope(
        {"kind": "if_block", "start": 2, "end": 5, "file": str(del_in_fb)},
        repo_root=str(tmp_path),
        tree=ast.parse(del_input_b),
    )
    assert "x" in sc_in_a["inputs"] and "x" in sc_in_b["inputs"]
    assert "x" in sc_in_a["conditional_outputs"] and "x" in sc_in_b["conditional_outputs"]
    assert not sc_in_a["is_control_flow_safe"] and not sc_in_b["is_control_flow_safe"]


def test_scope_visitor_nonlocals_annotations_loops_and_read_before_write(tmp_path: Path) -> None:
    """Verify scope analysis for class walrus, nested/forward nonlocals, bare annotations, for iter, and AugAssign."""
    from pydoppelgangerhunt.fixer._df_visitors import _extract_nested_scope_free_reads_with_nodes
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    cls_unit_src = "class Box:\n    vals = [w := v for v in items]\n    after = w\n"
    box_file = tmp_path / "box.py"
    box_file.write_text(cls_unit_src, encoding="utf-8")
    cls_scope = inspect_single_unit_scope(
        {"kind": "class", "start": 1, "end": 3, "file": str(box_file)},
        repo_root=str(tmp_path),
        tree=ast.parse(cls_unit_src),
    )
    assert "w" in cls_scope["inputs"]

    nested_nl_src = (
        "def f1():\n"
        "    def inner():\n"
        "        def deep():\n"
        "            nonlocal x\n"
        "            x = 2\n"
        "        x = 1\n"
        "        deep()\n"
        "        return x\n"
        "    return inner()\n"
    )
    nl_file = tmp_path / "nested_nl.py"
    nl_file.write_text(nested_nl_src, encoding="utf-8")
    nl_scope = inspect_single_unit_scope(
        {"kind": "function", "start": 1, "end": 9, "file": str(nl_file)},
        repo_root=str(tmp_path),
        tree=ast.parse(nested_nl_src),
    )
    assert "x" not in nl_scope["nonlocals"]

    for_self_ref_src = "def walk(node):\n    for node in node.children:\n        process(node)\n"
    for_scope = inspect_single_unit_scope(
        {
            "kind": "loop_body",
            "start": 2,
            "end": 3,
            "source_text": for_self_ref_src,
            "source_code": "for node in node.children:\n    process(node)\n",
        },
        tree=ast.parse(for_self_ref_src),
    )
    assert "node" in for_scope["inputs"]

    bare_attr_sub_src = "self.value: int\narr[idx]: str\n"
    bare_attr_scope = inspect_single_unit_scope(
        {"kind": "compound_block", "start": 1, "end": 2, "source_text": bare_attr_sub_src, "source_code": bare_attr_sub_src},
        tree=ast.parse(bare_attr_sub_src),
    )
    assert {"self", "arr", "idx"} <= set(bare_attr_scope["inputs"])
    assert bare_attr_scope["binding_kind"] == "instance"

    fwd_nl_fn_src = (
        "def compute(a: int) -> int:\n"
        "    def step(d: int) -> None:\n"
        "        nonlocal acc\n"
        "        acc += d\n"
        "    acc = a\n"
        "    step(5)\n"
        "    return acc\n"
    )
    fwd_nl_fn_scope = inspect_single_unit_scope(
        {"kind": "function", "start": 1, "end": 7, "source_text": fwd_nl_fn_src, "source_code": fwd_nl_fn_src},
        tree=ast.parse(fwd_nl_fn_src),
    )
    assert fwd_nl_fn_scope["nonlocals"] == []
    assert fwd_nl_fn_scope["inputs"] == ["a"]

    fwd_nl_sub_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": fwd_nl_fn_src,
            "source_code": "def step(d: int) -> None:\n    nonlocal acc\n    acc += d\nacc = a\nstep(5)\n",
        },
        tree=ast.parse(fwd_nl_fn_src),
    )
    assert fwd_nl_sub_scope["nonlocals"] == []
    assert fwd_nl_sub_scope["inputs"] == ["a"]

    cls_rbw_src = "def build() -> type:\n    class C:\n        val = x + 1\n    x = 0\n    return C\n"
    cls_rbw_scope = inspect_single_unit_scope(
        {"kind": "closure", "start": 1, "end": 5, "source_text": cls_rbw_src, "source_code": cls_rbw_src},
        tree=ast.parse(cls_rbw_src),
    )
    assert "x" in cls_rbw_scope["inputs"]

    bare_nl_fn_src = (
        "def compute(a: int) -> int:\n"
        "    acc: int\n"
        "    def init() -> None:\n"
        "        nonlocal acc\n"
        "        acc = a\n"
        "    init()\n"
        "    return acc\n"
    )
    bare_nl_fn_scope = inspect_single_unit_scope(
        {"kind": "function", "start": 1, "end": 7, "source_text": bare_nl_fn_src, "source_code": bare_nl_fn_src},
        tree=ast.parse(bare_nl_fn_src),
    )
    assert bare_nl_fn_scope["nonlocals"] == []
    assert bare_nl_fn_scope["inputs"] == ["a"]

    bare_nl_sub_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": bare_nl_fn_src,
            "source_code": "acc: int\ndef init() -> None:\n    nonlocal acc\n    acc = a\ninit()\n",
        },
        tree=ast.parse(bare_nl_fn_src),
    )
    assert bare_nl_sub_scope["nonlocals"] == []
    assert bare_nl_sub_scope["inputs"] == ["a"]
    assert "acc" in bare_nl_sub_scope["outputs"]

    nl_aug_fn = ast.parse(
        "def inner():\n    nonlocal z_val, a_val\n    z_val += 1\n    a_val += 2\n"
    ).body[0]
    assert isinstance(nl_aug_fn, ast.FunctionDef)
    nl_aug_reads, nl_aug_nodes = _extract_nested_scope_free_reads_with_nodes(nl_aug_fn)
    assert nl_aug_reads == {"z_val", "a_val"}
    assert [n.id for n in nl_aug_nodes] == ["z_val", "a_val"]


def test_scope_conditional_live_in_and_pre_bound_isolation() -> None:
    """Verify conditional live-in promotion requires prior bindings and isolates inner scopes/dels/TypeAlias."""
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _find_enclosing_pre_bound_names,
        inspect_single_unit_scope,
    )

    cond_reset_src = (
        "def gen(acc, reset_batch, step):\n"
        "    if reset_batch:\n"
        "        acc = 0\n"
        "    acc += step\n"
        "    yield acc\n"
    )
    cond_reset_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 5,
            "source_text": cond_reset_src,
            "source_code": "if reset_batch:\n    acc = 0\nacc += step\nyield acc\n",
        },
        tree=ast.parse(cond_reset_src),
    )
    assert "acc" in cond_reset_scope["inputs"]

    loop_cond_src = (
        "def run_steps(steps, acc):\n"
        "    for reset_batch, step in steps:\n"
        "        if reset_batch:\n"
        "            acc = 0\n"
        "        acc = acc + step\n"
        "    return 0\n"
    )
    loop_cond_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 3,
            "end": 5,
            "source_text": loop_cond_src,
            "source_code": "if reset_batch:\n    acc = 0\nacc = acc + step\n",
        },
        tree=ast.parse(loop_cond_src),
    )
    assert "acc" in loop_cond_scope["inputs"]
    loop_cond_reads = collect_downstream_read_names(
        loop_cond_src,
        {"start": 3, "end": 5},
        {"acc"},
        tree=ast.parse(loop_cond_src),
        unit_inputs=set(loop_cond_scope["inputs"]),
    )
    assert loop_cond_reads is not None and "acc" in loop_cond_reads

    leak_src = (
        "class Box:\n"
        "    CLASS_CONST = 1\n"
        "    def process(self, items, flag):\n"
        "        squares = [k * k for k in items]\n"
        "        fn = lambda l_arg: l_arg + 1\n"
        "        try:\n"
        "            risky()\n"
        "        except Exception as err:\n"
        "            pass\n"
        "        if flag:\n"
        "            k = 10\n"
        "            err = 20\n"
        "            CLASS_CONST = 30\n"
        "        res = (k, err, CLASS_CONST)\n"
        "        return res\n"
    )
    leak_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 10,
            "end": 14,
            "source_text": leak_src,
            "source_code": (
                "if flag:\n    k = 10\n    err = 20\n    CLASS_CONST = 30\nres = (k, err, CLASS_CONST)\n"
            ),
        },
        tree=ast.parse(leak_src),
    )
    assert "k" not in leak_scope["inputs"]
    assert "err" not in leak_scope["inputs"]
    assert "CLASS_CONST" not in leak_scope["inputs"]
    assert {"k", "err", "CLASS_CONST"} <= set(leak_scope["conditional_outputs"])

    nested_cond_full = (
        "def outer(x: int, flag: bool) -> int:\n"
        "    if flag:\n"
        "        x = 0\n"
        "    def helper(y: int) -> int:\n"
        "        return x + y\n"
        "    return helper(1)\n"
    )
    nested_cond_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": nested_cond_full,
            "source_code": "if flag:\n    x = 0\ndef helper(y: int) -> int:\n    return x + y\nreturn helper(1)\n",
        },
        tree=ast.parse(nested_cond_full),
    )
    assert "x" in nested_cond_scope["inputs"]

    uninit_def_src = (
        "def compute(flag: bool) -> tuple[int, int]:\n"
        "    if flag:\n"
        "        x = 10\n"
        "    y = x if flag else 0\n"
        "    x = 20\n"
        "    return y, x\n"
    )
    uninit_def_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 5,
            "source_text": uninit_def_src,
            "source_code": "if flag:\n    x = 10\ny = x if flag else 0\nx = 20\n",
        },
        tree=ast.parse(uninit_def_src),
    )
    assert "x" not in uninit_def_scope["inputs"]
    assert "x" in uninit_def_scope["outputs"]

    uninit_loop_src = (
        "def process(items: list[int], flag: bool) -> int:\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        if flag:\n"
        "            k = item * 2\n"
        "        total += k if flag else 1\n"
        "    return total\n"
    )
    uninit_loop_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 4,
            "end": 6,
            "source_text": uninit_loop_src,
            "source_code": "if flag:\n    k = item * 2\ntotal += k if flag else 1\n",
        },
        tree=ast.parse(uninit_loop_src),
    )
    assert "k" not in uninit_loop_scope["inputs"]
    assert "k" in uninit_loop_scope["conditional_outputs"]

    caller_callee_src = (
        "def outer(x):\n"
        "    def caller():\n"
        "        return callee()\n"
        "    def callee():\n"
        "        return x\n"
        "    res = caller()\n"
        "    x = 10\n"
        "    return res, x\n"
    )
    caller_callee_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 7,
            "source_text": caller_callee_src,
            "source_code": "def caller():\n    return callee()\ndef callee():\n    return x\nres = caller()\nx = 10\n",
        },
        tree=ast.parse(caller_callee_src),
    )
    assert "x" in caller_callee_scope["inputs"]

    del_star_src = (
        "from math import *\n"
        "def f(flag):\n"
        "    x = 1\n"
        "    del x\n"
        "    if flag:\n"
        "        x = 2\n"
        "    y = x if flag else 0\n"
        "    return y\n"
    )
    pre_bound_del, _ = _find_enclosing_pre_bound_names(ast.parse(del_star_src), 5, 7)
    assert "*" not in pre_bound_del
    assert "x" not in pre_bound_del
    del_star_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 5,
            "end": 7,
            "source_text": del_star_src,
            "source_code": "if flag:\n    x = 2\ny = x if flag else 0\n",
        },
        tree=ast.parse(del_star_src),
    )
    assert "x" not in del_star_scope["inputs"]
    assert "x" in del_star_scope["conditional_outputs"]

    for_def, for_cond = _analyze_block_assignment(ast.parse("for arr[idx] in items:\n    pass\n").body)
    assert "arr" not in for_cond
    assert "idx" not in for_cond
    assert "arr" not in for_def

    branch_del_src = (
        "def f(flag, cond):\n"
        "    x = 1\n"
        "    if cond:\n"
        "        del x\n"
        "    if flag:\n"
        "        x = 2\n"
        "    y = x if flag else 0\n"
        "    return y\n"
    )
    pre_bound_branch_del, _ = _find_enclosing_pre_bound_names(ast.parse(branch_del_src), 5, 7)
    assert "x" in pre_bound_branch_del

    if sys.version_info >= (3, 12):
        ta_pre_src = (
            "def f(flag):\n"
            "    type MyAlias[T: (lazy_w := int)] = list[T]\n"
            "    if flag:\n"
            "        lazy_w = 1\n"
            "    return lazy_w\n"
        )
        pre_bound_ta, _ = _find_enclosing_pre_bound_names(ast.parse(ta_pre_src), 3, 5)
        assert "MyAlias" in pre_bound_ta
        assert "lazy_w" not in pre_bound_ta


def test_source_continuation_lines_and_boundary_pragmas() -> None:
    """Verify multiline string continuations, pragma preservation, cast probe idents, recursion guard, and super() pruning."""
    from pydoppelgangerhunt.fixer.patch import (
        _build_unit_delegation_call,
        _validate_helper_compilation,
        refactor_module_units,
    )
    from pydoppelgangerhunt.fixer.source import _find_multiline_string_continuation_lines
    from pydoppelgangerhunt.fixer.synthesis import (
        _extract_probe_active_idents,
        _prune_unreferenced_lhs_scope,
    )

    cast_idents = _extract_probe_active_idents(
        'def _probe(x: Any) -> Any:\n    return typing.cast("T", x)\n', "Any"
    )
    assert "T" in cast_idents

    expr_src = "# leading note\nres = [x + 1 for x in items]  # type: ignore\n"
    expr_call = _build_unit_delegation_call(
        {"kind": "comprehension", "start": 2, "end": 2},
        expr_src,
        expr_src.splitlines(keepends=True),
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["items"],
        outputs=[],
        scope={"param_details": []},
        preserve_boundary_pragmas=True,
    )
    assert expr_call == "res = _shared_comp(items)  # type: ignore\n"

    assert _find_multiline_string_continuation_lines('x = "unclosed\ny = 1\nz = 2\n') == set()
    assert _find_multiline_string_continuation_lines('x = """unclosed\ny = 1\nz = 2\n') == {2, 3}

    expr_pragma_src = (
        "def compute(items):\n"
        "    res = [\n"
        "        x + 1  # type: ignore[operator]\n"
        "        for x in items\n"
        "    ]\n"
        "    return res\n"
    )
    deleg_call = _build_unit_delegation_call(
        {"kind": "comprehension", "start": 2, "end": 5},
        expr_pragma_src,
        expr_pragma_src.splitlines(keepends=True),
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["items"],
        outputs=[],
        scope={"param_details": []},
    )
    assert "# type: ignore[operator]" in deleg_call
    refactored_expr = refactor_module_units(
        expr_pragma_src,
        [({"kind": "comprehension", "start": 2, "end": 5, "start_col": 10, "end_col": 5}, "_shared_comp(items)")],
    )
    assert "# type: ignore[operator]" in refactored_expr

    deep_expr = "(" * 5000 + "1" + ")" * 5000
    assert _validate_helper_compilation(deep_expr) is False
    assert _validate_helper_compilation("    " + deep_expr) is False
    assert _validate_helper_compilation("import os\n\n\n" + deep_expr) is False

    super_unit: Dict[str, Any] = {
        "kind": "comprehension",
        "inputs": ["self", "items"],
        "receiver_param": "self",
        "receiver_kind": "instance",
    }
    super_scope: Dict[str, Any] = {
        "inputs": ["self", "items"],
        "free_vars": ["self", "items"],
        "attrs_read": [],
        "attrs_written": ["self.res"],
        "instance_attrs": ["self.res"],
        "class_attrs": [],
        "has_super": True,
        "has_mangled_names": False,
        "has_instance_binding": True,
        "has_class_binding": False,
        "has_receiver_access": True,
        "binding_kind": "instance",
    }
    _prune_unreferenced_lhs_scope(
        super_unit, super_scope, ["self.res = [super().transform(x) for x in items]\n"]
    )
    assert "self" in super_scope["inputs"]
    assert super_scope["has_instance_binding"] is True
    assert super_scope["binding_kind"] == "instance"


def test_sequence_source_lines_without_slice_support_and_string_controls(
    tmp_path: Path,
) -> None:
    """Verifies arbitrary Sequence[str] objects work without slice support and strings are rejected."""
    from collections import UserList
    from collections.abc import Sequence

    from pydoppelgangerhunt.fixer.source import is_subroutine_unit, validate_unit_schema
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
        is_non_string_sequence,
    )

    class _IndexOnlyLines(Sequence[str]):
        def __init__(self, items: list[str]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int | slice) -> Any:
            if isinstance(index, slice):
                raise TypeError("slicing is intentionally unsupported")
            return self._items[index]

    sample_lines = ["def add(a, b):\n", "    return a + b\n"]
    seq_variants: tuple[Sequence[str], ...] = (
        tuple(sample_lines),
        UserList(sample_lines),
        _IndexOnlyLines(sample_lines),
    )
    for seq in seq_variants:
        assert is_non_string_sequence(seq) is True
        unit: dict[str, Any] = {
            "file": str(tmp_path / "mod.py"),
            "start": 2,
            "end": 2,
            "source_lines": seq,
        }
        validate_unit_schema(unit)
        assert is_subroutine_unit(unit) is True
        lines, origin = extract_unit_source_lines_with_state(unit)
        assert origin == SourceLinesOrigin.FULL_BUFFER
        assert lines == [sample_lines[1]]

    for bad_seq in ("def f(): pass\n", b"def f(): pass\n", bytearray(b"abc")):
        assert is_non_string_sequence(bad_seq) is False
        with pytest.raises(ValueError, match="ordered sequence of str"):
            validate_unit_schema(
                {"file": "a.py", "start": 1, "end": 1, "source_lines": bad_seq}
            )


def test_extract_unit_source_lines_disk_out_of_range_and_lf_normalization(
    tmp_path: Path,
) -> None:
    """Verifies disk out-of-range slices return PLACEHOLDER origin and in-range slices normalize LF."""
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    target = tmp_path / "no_trailing_nl.py"
    target.write_text("a = 1\nb = 2", encoding="utf-8")

    valid_lines, valid_origin = extract_unit_source_lines_with_state(
        {"file": str(target), "start": 1, "end": 2, "kind": "block"},
        repo_root=str(tmp_path),
    )
    assert valid_origin == SourceLinesOrigin.DISK
    assert valid_lines == ["a = 1\n", "b = 2\n"]

    oob_lines, oob_origin = extract_unit_source_lines_with_state(
        {"file": str(target), "start": 10, "end": 12, "kind": "block"},
        repo_root=str(tmp_path),
    )
    assert oob_origin == SourceLinesOrigin.PLACEHOLDER
    assert oob_lines == ["# Source for unit lines 10-12\n"]


def test_non_str_source_lines_fail_closed_to_placeholder(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies non-string items inside source_lines are rejected by validate_unit_schema and fail closed to PLACEHOLDER."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    bad_units = [
        {"name": "bad_int", "start": 1, "end": 2, "source_lines": [123, "x = 1\n"]},
        {"name": "bad_none", "start": 1, "end": 2, "source_lines": [None]},
    ]
    for bad_unit in bad_units:
        with pytest.raises(ValueError, match="ordered sequence of str"):
            validate_unit_schema(bad_unit)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            lines, origin = extract_unit_source_lines_with_state(bad_unit)
        assert origin is SourceLinesOrigin.PLACEHOLDER
        assert origin.is_placeholder is True
        assert origin.has_real_source is False
        assert lines == [f"# Source for {bad_unit['name']} lines 1-2\n"]
        assert "Rejecting non-string source_lines entry" in caplog.text


def test_notebook_cache_recursion_error_and_lru_eviction_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies read_notebook_cell_lines_cached handles RecursionError and enforces bounded LRU reuse/eviction."""
    import pydoppelgangerhunt.source_lines as sl_mod
    from pydoppelgangerhunt.source_lines import (
        _MAX_NOTEBOOK_CACHE_ENTRIES,
        _NOTEBOOK_CELLS_CACHE,
        clear_notebook_cache,
        read_notebook_cell_lines_cached,
    )

    clear_notebook_cache()
    deep_nb = tmp_path / "deep.ipynb"
    deep_nb.write_text('{"cells": [{"cell_type": "code", "source": ["x = 1"]}]}', encoding="utf-8")
    with monkeypatch.context() as mp:
        mp.setattr(sl_mod.json, "loads", lambda *_a, **_kw: (_ for _ in ()).throw(RecursionError("deep json")))
        assert read_notebook_cell_lines_cached(deep_nb, 0) is None

    clear_notebook_cache()
    parse_calls = 0
    real_loads = json.loads

    def _counting_loads(text: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return real_loads(text, *args, **kwargs)

    nb_paths: list[Path] = []
    for idx in range(_MAX_NOTEBOOK_CACHE_ENTRIES + 2):
        nb_p = tmp_path / f"nb_{idx}.ipynb"
        nb_p.write_text(
            json.dumps({"cells": [{"cell_type": "code", "source": [f"val_{idx} = {idx}\n"]}]}),
            encoding="utf-8",
        )
        nb_paths.append(nb_p)

    with monkeypatch.context() as mp:
        mp.setattr(sl_mod.json, "loads", _counting_loads)
        # Fill cache cleanly
        for idx in range(_MAX_NOTEBOOK_CACHE_ENTRIES):
            assert read_notebook_cell_lines_cached(nb_paths[idx], 0) == [f"val_{idx} = {idx}\n"]
        assert parse_calls == _MAX_NOTEBOOK_CACHE_ENTRIES

        # Repeated read of nb_0 reuses cache (moves to MRU) without calling json.loads again
        assert read_notebook_cell_lines_cached(nb_paths[0], 0) == ["val_0 = 0\n"]
        assert parse_calls == _MAX_NOTEBOOK_CACHE_ENTRIES

        # Adding 2 more notebooks evicts nb_1 and nb_2 (the LRU entries), keeping cache size bounded
        assert read_notebook_cell_lines_cached(nb_paths[-2], 0) == [
            f"val_{_MAX_NOTEBOOK_CACHE_ENTRIES} = {_MAX_NOTEBOOK_CACHE_ENTRIES}\n"
        ]
        assert read_notebook_cell_lines_cached(nb_paths[-1], 0) == [
            f"val_{_MAX_NOTEBOOK_CACHE_ENTRIES + 1} = {_MAX_NOTEBOOK_CACHE_ENTRIES + 1}\n"
        ]
        assert len(_NOTEBOOK_CELLS_CACHE) == _MAX_NOTEBOOK_CACHE_ENTRIES
        cached_paths = {key[0] for key in _NOTEBOOK_CELLS_CACHE}
        assert str(nb_paths[0]) in cached_paths
        assert str(nb_paths[1]) not in cached_paths
        assert str(nb_paths[2]) not in cached_paths
    clear_notebook_cache()


@pytest.mark.parametrize(
    ("case_id", "unit_overrides"),
    [
        pytest.param(
            "malformed_cell_0_with_source_lines",
            {
                "file": "analysis.ipynb#cell_0",
                "start": 1,
                "end": 2,
                "source_lines": ["def example():\n", "    return 42\n"],
            },
            id="malformed_cell_0_with_source_lines",
        ),
        pytest.param(
            "malformed_negative_cell_with_source_text",
            {
                "file": "analysis.ipynb#cell_-1",
                "start": 1,
                "end": 2,
                "source_text": "def example():\n    return 42\n",
            },
            id="malformed_negative_cell_with_source_text",
        ),
        pytest.param(
            "malformed_non_cell_fragment_with_file_source",
            {
                "file": "analysis.ipynb#bad_anchor",
                "start": 1,
                "end": 2,
                "file_source": "def example():\n    return 42\n",
            },
            id="malformed_non_cell_fragment_with_file_source",
        ),
        pytest.param(
            "malformed_empty_fragment_with_source_lines",
            {
                "file": "analysis.ipynb#",
                "start": 1,
                "end": 1,
                "source_lines": ["x = 1\n"],
            },
            id="malformed_empty_fragment_with_source_lines",
        ),
        pytest.param(
            "out_of_range_cell_999_against_nbformat_file_source_even_with_source_lines",
            {
                "file": "analysis.ipynb#cell_999",
                "start": 1,
                "end": 2,
                "source_lines": ["def example():\n", "    return 42\n"],
                "file_source": json.dumps(
                    {
                        "nbformat": 4,
                        "cells": [{"cell_type": "code", "source": ["def a():\n", "    return 1\n"]}],
                    }
                ),
            },
            id="out_of_range_cell_999_with_nbformat_file_source",
        ),
        pytest.param(
            "out_of_range_cell_999_against_nbformat_source_text",
            {
                "file": "analysis.ipynb#cell_999",
                "start": 1,
                "end": 2,
                "source_text": json.dumps(
                    {"cells": [{"cell_type": "code", "source": ["def a():\n", "    return 1\n"]}]}
                ),
            },
            id="out_of_range_cell_999_with_nbformat_source_text",
        ),
        pytest.param(
            "markdown_cell_in_nbformat_file_source",
            {
                "file": "analysis.ipynb#cell_1",
                "start": 1,
                "end": 1,
                "file_source": json.dumps(
                    {
                        "cells": [
                            {"cell_type": "markdown", "source": ["# Heading\n"]},
                            {"cell_type": "code", "source": ["def c():\n", "    return 2\n"]},
                        ]
                    }
                ),
            },
            id="markdown_cell_in_nbformat_file_source",
        ),
        pytest.param(
            "corrupt_json_in_nb_file_source",
            {
                "file": "analysis.ipynb#cell_1",
                "start": 1,
                "end": 1,
                "source_lines": ["x = 1\n"],
                "file_source": "{corrupt json",
            },
            id="corrupt_json_in_nb_file_source",
        ),
        pytest.param(
            "corrupt_nbformat_json_in_source_text",
            {
                "file": "analysis.ipynb#cell_1",
                "start": 1,
                "end": 1,
                "source_text": '{"cells": [{"cell_type": "code"',
            },
            id="corrupt_nbformat_json_in_source_text",
        ),
        pytest.param(
            "raw_nbformat_json_in_source_lines_wrong_cell",
            {
                "file": "analysis.ipynb#cell_2",
                "start": 1,
                "end": 1,
                "source_lines": [
                    '{"cells": [{"cell_type": "code", "source": ["x = 1\\n"]}]}'
                ],
            },
            id="raw_nbformat_json_in_source_lines_wrong_cell",
        ),
    ],
)
def test_extract_unit_source_lines_in_memory_notebook_anchors_fail_closed(
    case_id: str, unit_overrides: dict[str, Any]
) -> None:
    """Verifies malformed anchors and invalid/out-of-range in-memory nbformat buffers fail closed as PLACEHOLDER."""
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    unit = {"name": case_id, **unit_overrides}
    lines, origin = extract_unit_source_lines_with_state(unit)
    assert lines == [], case_id
    assert origin is SourceLinesOrigin.PLACEHOLDER, case_id


def test_extract_unit_source_lines_in_memory_nbformat_and_unsaved_cell_buffers(
    tmp_path: Path,
) -> None:
    """Verifies in-memory nbformat extraction vs authoritative plain-Python unsaved notebook cell buffers."""
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    nb_json = json.dumps(
        {
            "nbformat": 4,
            "cells": [
                {"cell_type": "markdown", "source": ["# Notes\n"]},
                {"cell_type": "code", "source": ["def cell_two(x):\n", "    return x + 2\n"]},
            ],
        }
    )

    # 1. Valid nbformat JSON in file_source extracts the anchored code cell's Python lines
    lines_fs, origin_fs = extract_unit_source_lines_with_state(
        {"file": "nb.ipynb#cell_2", "start": 1, "end": 2, "file_source": nb_json}
    )
    assert lines_fs == ["def cell_two(x):\n", "    return x + 2\n"]
    assert origin_fs is SourceLinesOrigin.FULL_BUFFER

    # 2. Out-of-range line bounds inside a valid JSON code cell fail closed
    lines_oob, origin_oob = extract_unit_source_lines_with_state(
        {"file": "nb.ipynb#cell_2", "start": 10, "end": 12, "source_text": nb_json}
    )
    assert lines_oob == []
    assert origin_oob is SourceLinesOrigin.PLACEHOLDER

    # 3. Plain-Python cell buffers with syntactically valid #cell_N override disk .ipynb lookup
    disk_nb = tmp_path / "nb.ipynb"
    disk_nb.write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": ["disk_only = 0\n"]}]}),
        encoding="utf-8",
    )
    lines_unsaved, origin_unsaved = extract_unit_source_lines_with_state(
        {
            "file": "nb.ipynb#cell_2",
            "start": 1,
            "end": 2,
            "source_lines": ["def unsaved():\n", "    return 7\n"],
        },
        repo_root=str(tmp_path),
    )
    assert lines_unsaved == ["def unsaved():\n", "    return 7\n"]
    assert origin_unsaved is SourceLinesOrigin.PRE_SLICED

    # 4. Plain-Python dict literal in source_text is not misclassified as nbformat JSON
    lines_dict, origin_dict = extract_unit_source_lines_with_state(
        {"file": "nb.ipynb#cell_1", "start": 1, "end": 1, "source_text": '{"k": 1}\n'}
    )
    assert lines_dict == ['{"k": 1}\n']
    assert origin_dict is SourceLinesOrigin.FULL_BUFFER

    # 5. Plain-Python unsaved source_lines overrides valid nbformat file_source when cell anchor exists
    lines_edit, origin_edit = extract_unit_source_lines_with_state(
        {
            "file": "nb.ipynb#cell_2",
            "start": 1,
            "end": 2,
            "source_lines": ["def edited():\n", "    return 99\n"],
            "file_source": nb_json,
        }
    )
    assert lines_edit == ["def edited():\n", "    return 99\n"]
    assert origin_edit is SourceLinesOrigin.PRE_SLICED
