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
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 3, "start_col": 4}) is None
    assert _resolve_unit_ast_end_col(fn_scope, {"start": 2, "end": 3, "start_col": 8}) is None

    comp_tree = ast.parse("def _shared(items, a):\n    return [x for (x, a[x]) in items]\n")
    free_names, _, _ = _extract_helper_symbols(comp_tree)
    assert "x" in free_names


def test_scope_intervals_col_offset_sig_and_lru_shrink_and_decorated_pragma() -> None:
    """Verifies col_offset signature invalidation, dynamic LRU shrinkage, and decorated pragma leading scan."""
    from collections import OrderedDict  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access,import-outside-toplevel
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
    intervals1 = _get_tree_scope_intervals(tree)
    assert intervals1[0][3] is tree.body[1]
    tree.body[0].col_offset = 30
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
