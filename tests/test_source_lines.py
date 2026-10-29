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
        intervals2 = _get_tree_scope_intervals(cached_tree)
        assert intervals2 is not intervals1
    finally:
        cached_tree.body[0].col_offset = orig_col  # type: ignore[attr-defined]

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


def test_review_edge_cases_multiline_end_col_class_walrus_and_top_config(
    tmp_path: Any,
) -> None:
    """Verify multi-line start_col end_col resolution, class walrus scope, and config top validation."""
    import ast
    import sys
    from pydoppelgangerhunt.cli import main as cli_main
    from pydoppelgangerhunt.config import load_tool_config
    from pydoppelgangerhunt.fixer._df_cache import _resolve_unit_ast_end_col
    from pydoppelgangerhunt.fixer._df_closures import _extract_single_node_local_names

    # 1. Multi-line compound statement with start_col resolves leaf end_col_offset
    compound_src = "if cond:\n    x = 1\n    y = 2\n"
    compound_tree = ast.parse(compound_src)
    assert (
        _resolve_unit_ast_end_col(
            compound_tree, {"start": 1, "end": 3, "start_col": 0}
        )
        == 9
    )
    assert (
        _resolve_unit_ast_end_col(
            compound_tree, {"start": 1, "end": 3, "start_col": 4}
        )
        is None
    )

    # 2. Class-body comprehension walrus target does not leak into ClassDef local bindings
    cls_tree = ast.parse("class C:\n    vals = [w := v for v in items]\n")
    cls_locals = _extract_single_node_local_names(cls_tree.body[0])
    assert "vals" in cls_locals
    assert "w" not in cls_locals

    fn_tree = ast.parse("def f(items):\n    vals = [w := v for v in items]\n")
    fn_locals = _extract_single_node_local_names(fn_tree.body[0])
    assert "vals" in fn_locals
    assert "w" in fn_locals

    # 3. 'top' in _CONFIG_KEY_NAMES and validated in main() when loaded from config (including float rejection)
    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text("top = 5\n", encoding="utf-8")
    loaded = load_tool_config(config_path=str(cfg_file))
    assert loaded.get("top") == 5

    for invalid_top in ("0", "2.5"):
        bad_cfg = tmp_path / "bad_top.toml"
        bad_cfg.write_text(
            f"[tool.pydoppelgangerhunt]\ntop = {invalid_top}\n", encoding="utf-8"
        )
        with pytest.raises(SystemExit):
            cli_main([str(tmp_path), "--config", str(bad_cfg)])

    # 4. Class-scoped annotations on PEP 695 generic methods in _extract_helper_symbols
    if sys.version_info >= (3, 12):
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

    # 5. _OuterExprVisitor.visit_NamedExpr visits value without recording store target
    from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor

    outer_vis = _OuterExprVisitor()
    outer_vis.visit(ast.parse("(w := val + 1)", mode="eval"))
    assert outer_vis.ordered_names == ["val"]

    # 6. Finding 1: Incremental class scope binding & comprehension walrus in _extract_helper_symbols
    from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names
    from pydoppelgangerhunt.fixer.patch import (
        _build_unit_delegation_call,
        _extract_helper_symbols,
    )

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

    comp_walrus_code = (
        "def _h(items):\n"
        "    vals = [w := v + 1 for v in items]\n"
        "    return w, vals\n"
    )
    free_cw, _, _ = _extract_helper_symbols(ast.parse(comp_walrus_code))
    assert "w" not in free_cw

    # 7. Finding 2: _DownstreamReadVisitor isolates class-attribute kills from method reads
    cls_downstream_src = (
        "def outer():\n"
        "    x = 1\n"
        "    class C:\n"
        "        a = x + 1\n"
        "        x = 99\n"
        "        def m(self):\n"
        "            return x\n"
        "    return C\n"
    )
    ds_reads = collect_downstream_read_names(
        cls_downstream_src, {"kind": "compound_block", "start": 4, "end": 4}, {"x", "a"}
    )
    assert ds_reads is not None and "x" in ds_reads

    # 8. Finding 3: _extract_probe_active_idents retains cast("T", ...) type parameter
    from pydoppelgangerhunt.fixer.synthesis import _extract_probe_active_idents

    cast_idents = _extract_probe_active_idents(
        'def _probe(x: Any) -> Any:\n    return typing.cast("T", x)\n', "Any"
    )
    assert "T" in cast_idents

    # 9. Finding 4: _build_unit_delegation_call on whole-line expr preserves inline pragma without duplicating leading comments
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

    # 10. Finding 5: Class-body comprehension walrus target does not mask subsequent class-body loads in _ScopeVisitor
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    cls_unit_src = (
        "class Box:\n"
        "    vals = [w := v for v in items]\n"
        "    after = w\n"
    )
    box_file = tmp_path / "box.py"
    box_file.write_text(cls_unit_src, encoding="utf-8")
    cls_scope = inspect_single_unit_scope(
        {"kind": "class", "start": 1, "end": 3, "file": str(box_file)},
        repo_root=str(tmp_path),
        tree=ast.parse(cls_unit_src),
    )
    assert "w" in cls_scope["inputs"]

    # 11. Class-body comprehension reads outer local after same-named class attribute store
    cls_comp_src = (
        "def outer():\n"
        "    x = 1\n"
        "    class C:\n"
        "        a = x + 1\n"
        "        x = 99\n"
        "        vals = [x for _ in range(1)]\n"
        "    return C\n"
    )
    ds_comp_reads = collect_downstream_read_names(
        cls_comp_src, {"kind": "compound_block", "start": 4, "end": 4}, {"x", "a"}
    )
    assert ds_comp_reads is not None and "x" in ds_comp_reads

    # 12. Nested nonlocal declared before assignment in intermediate nested function
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

    # 13. Single-line unclosed string does not mark remaining lines as multiline continuations
    from pydoppelgangerhunt.fixer.source import _find_multiline_string_continuation_lines

    unclosed_single = 'x = "unclosed\ny = 1\nz = 2\n'
    assert _find_multiline_string_continuation_lines(unclosed_single) == set()
    unclosed_triple = 'x = """unclosed\ny = 1\nz = 2\n'
    assert _find_multiline_string_continuation_lines(unclosed_triple) == {2, 3}

    # 14. Headerless TOML fallback with multiline array element starting with '['
    from pydoppelgangerhunt.config import load_toml_section

    hdrless_toml = tmp_path / "headerless_arr.toml"
    hdrless_toml.write_text(
        'exclude = [\n  ["a", "b"]\n]\nthreshold = 0.85\n', encoding="utf-8"
    )
    orig_tomllib = sys.modules.get("tomllib")
    orig_tomli = sys.modules.get("tomli")
    try:
        sys.modules["tomllib"] = None  # type: ignore[assignment]
        sys.modules["tomli"] = None  # type: ignore[assignment]
        parsed_hdrless = load_toml_section(str(hdrless_toml), "pydoppelgangerhunt")
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

    # 15. Finding 1.1: collect_downstream_read_names initializes outer_bound across outer parameters,
    # enclosing functions, and loop-carried passes.
    outer_param_cls_src = (
        "def outer(x):\n"
        "    for _ in range(2):\n"
        "        class C:\n"
        "            x += 1\n"
        "            x = 99\n"
        "            vals = [x for _ in range(1)]\n"
        "    return None\n"
    )
    ds_outer_param = collect_downstream_read_names(
        outer_param_cls_src, {"kind": "compound_block", "start": 4, "end": 4}, {"x"}
    )
    assert ds_outer_param is not None and "x" in ds_outer_param

    loop_carried_cls_src = (
        "def outer(x):\n"
        "    for _ in range(2):\n"
        "        class C:\n"
        "            vals = [x for _ in range(1)]\n"
        "            x = 99\n"
        "        x += 1\n"
    )
    ds_loop_carried = collect_downstream_read_names(
        loop_carried_cls_src, {"kind": "compound_block", "start": 6, "end": 6}, {"x"}
    )
    assert ds_loop_carried is not None and "x" in ds_loop_carried

    # 16. Finding 1.2: Branch-aware deleted_names symmetry and deleted_output hazard
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_pair_outputs

    del_branch_a = (
        "def f(cond):\n"
        "    if cond:\n"
        "        x = 1\n"
        "        del x\n"
        "    else:\n"
        "        x = 2\n"
        "    return 0\n"
    )
    del_branch_b = (
        "def f(cond):\n"
        "    if not cond:\n"
        "        x = 2\n"
        "    else:\n"
        "        x = 1\n"
        "        del x\n"
        "    return 0\n"
    )
    del_file_a = tmp_path / "del_a.py"
    del_file_b = tmp_path / "del_b.py"
    del_file_a.write_text(del_branch_a, encoding="utf-8")
    del_file_b.write_text(del_branch_b, encoding="utf-8")
    u_del_a = {"kind": "if_block", "start": 2, "end": 6, "file": str(del_file_a)}
    u_del_b = {"kind": "if_block", "start": 2, "end": 6, "file": str(del_file_b)}
    sc_del_a = inspect_single_unit_scope(
        u_del_a, repo_root=str(tmp_path), tree=ast.parse(del_branch_a)
    )
    sc_del_b = inspect_single_unit_scope(
        u_del_b, repo_root=str(tmp_path), tree=ast.parse(del_branch_b)
    )
    assert sc_del_a["outputs"] == sc_del_b["outputs"] == ["x"]
    assert sc_del_a["conditional_outputs"] == sc_del_b["conditional_outputs"] == ["x"]
    assert "deleted_output" in sc_del_a["control_flow_hazards"]
    assert "deleted_output" in sc_del_b["control_flow_hazards"]
    assert not sc_del_a["is_control_flow_safe"]
    assert not sc_del_b["is_control_flow_safe"]
    assert (
        resolve_clone_pair_outputs(
            u_del_a, u_del_b, sc_del_a, sc_del_b, del_branch_a, del_branch_b
        )
        is None
    )

    del_input_a = (
        "def g(x, cond):\n"
        "    if cond:\n"
        "        del x\n"
        "    else:\n"
        "        x = 2\n"
    )
    del_input_b = (
        "def g(x, cond):\n"
        "    if not cond:\n"
        "        x = 2\n"
        "    else:\n"
        "        del x\n"
    )
    del_in_fa = tmp_path / "del_in_a.py"
    del_in_fb = tmp_path / "del_in_b.py"
    del_in_fa.write_text(del_input_a, encoding="utf-8")
    del_in_fb.write_text(del_input_b, encoding="utf-8")
    u_in_a = {"kind": "if_block", "start": 2, "end": 5, "file": str(del_in_fa)}
    u_in_b = {"kind": "if_block", "start": 2, "end": 5, "file": str(del_in_fb)}
    sc_in_a = inspect_single_unit_scope(
        u_in_a, repo_root=str(tmp_path), tree=ast.parse(del_input_a)
    )
    sc_in_b = inspect_single_unit_scope(
        u_in_b, repo_root=str(tmp_path), tree=ast.parse(del_input_b)
    )
    assert "x" in sc_in_a["inputs"] and "x" in sc_in_b["inputs"]
    assert "x" in sc_in_a["conditional_outputs"] and "x" in sc_in_b["conditional_outputs"]
    assert not sc_in_a["is_control_flow_safe"] and not sc_in_b["is_control_flow_safe"]

    # 17. Finding 1.3: _extract_helper_symbols handles match/case bindings and bare AnnAssign
    if sys.version_info >= (3, 10):
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
        assert "first" not in m_free
        assert "rest" not in m_free
        assert "mapped" not in m_free
        assert "extra" not in m_free
        assert "bare_attr" in m_free
        assert "OuterType" in m_free

    # 18. Finding 1.4: Interior line pragmas preserved on multi-line expression units
    from pydoppelgangerhunt.fixer.patch import (
        _build_unit_delegation_call,
        refactor_module_units,
    )

    expr_pragma_src = (
        "def compute(items):\n"
        "    res = [\n"
        "        x + 1  # type: ignore[operator]\n"
        "        for x in items\n"
        "    ]\n"
        "    return res\n"
    )
    expr_unit_full = {"kind": "comprehension", "start": 2, "end": 5}
    deleg_call = _build_unit_delegation_call(
        expr_unit_full,
        expr_pragma_src,
        expr_pragma_src.splitlines(keepends=True),
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["items"],
        outputs=[],
        scope={"param_details": []},
    )
    assert "# type: ignore[operator]" in deleg_call
    expr_unit_col = {
        "kind": "comprehension",
        "start": 2,
        "end": 5,
        "start_col": 10,
        "end_col": 5,
    }
    refactored_expr = refactor_module_units(
        expr_pragma_src, [(expr_unit_col, "_shared_comp(items)")]
    )
    assert "# type: ignore[operator]" in refactored_expr

    # 19. Finding 1.1: ast.For / ast.AsyncFor visits iter before target in _ScopeVisitor and _ClassScopeVisitor
    for_self_ref_src = (
        "def walk(node):\n"
        "    for node in node.children:\n"
        "        process(node)\n"
    )
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

    cls_for_src = (
        "def outer(x):\n"
        "    x = x + 1\n"
        "    class C:\n"
        "        for x in x:\n"
        "            pass\n"
        "    return C\n"
    )
    cls_for_reads = collect_downstream_read_names(
        cls_for_src, {"start": 2, "end": 2}, {"x"}, tree=ast.parse(cls_for_src)
    )
    assert cls_for_reads is not None and "x" in cls_for_reads

    # 20. Finding 1.2: Live-in reads after conditional stores in _inspect_unit_scope and collect_downstream_read_names
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

    # 21. Finding 1.3: Bare ast.AnnAssign in ClassDef not treated as runtime store
    from pydoppelgangerhunt.fixer._df_closures import _extract_single_node_local_names

    cls_ann_tree = ast.parse("class C:\n    bare_x: int\n    assigned_y: int = 1\n")
    cls_node = cls_ann_tree.body[0]
    cls_locals = _extract_single_node_local_names(cls_node)
    assert "bare_x" not in cls_locals
    assert "assigned_y" in cls_locals

    cls_ann_ds_src = (
        "def outer(x):\n"
        "    x = x + 1\n"
        "    class C:\n"
        "        x: int\n"
        "        z = x\n"
        "    return C\n"
    )
    cls_ann_ds_reads = collect_downstream_read_names(
        cls_ann_ds_src, {"start": 2, "end": 2}, {"x"}, tree=ast.parse(cls_ann_ds_src)
    )
    assert cls_ann_ds_reads is not None and "x" in cls_ann_ds_reads

    # 22. Finding 1.4: Bare non-Name AnnAssign targets visited in _ScopeVisitor
    bare_attr_sub_src = "self.value: int\narr[idx]: str\n"
    bare_attr_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 1,
            "end": 2,
            "source_text": bare_attr_sub_src,
            "source_code": bare_attr_sub_src,
        },
        tree=ast.parse(bare_attr_sub_src),
    )
    assert "self" in bare_attr_scope["inputs"]
    assert "arr" in bare_attr_scope["inputs"]
    assert "idx" in bare_attr_scope["inputs"]
    assert bare_attr_scope["binding_kind"] == "instance"

    # 23. Finding 1.5: _extract_helper_symbols respects global/nonlocal in _collect_direct_bindings
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

    # 24. Finding 1.6: _validate_helper_compilation catches RecursionError
    from pydoppelgangerhunt.fixer.patch import _validate_helper_compilation

    deep_expr = "(" * 5000 + "1" + ")" * 5000
    assert _validate_helper_compilation(deep_expr) is False
    assert _validate_helper_compilation("    " + deep_expr) is False
    assert _validate_helper_compilation("import os\n\n\n" + deep_expr) is False

    # 25. Finding 1.1: Comprehension targets, lambda locals, closed ExceptHandlers, and enclosing class attrs
    # do not leak into _find_enclosing_pre_bound_names.
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
                "if flag:\n"
                "    k = 10\n"
                "    err = 20\n"
                "    CLASS_CONST = 30\n"
                "res = (k, err, CLASS_CONST)\n"
            ),
        },
        tree=ast.parse(leak_src),
    )
    assert "k" not in leak_scope["inputs"]
    assert "err" not in leak_scope["inputs"]
    assert "CLASS_CONST" not in leak_scope["inputs"]
    assert {"k", "err", "CLASS_CONST"} <= set(leak_scope["conditional_outputs"])

    # 26. Finding 1.2: Zero-dependency TOML fallback tracks multiline arrays across non-target sections
    assert parsed_hdrless.get("exclude") == [["a", "b"]]
    sec_toml = tmp_path / "sec_multiline.toml"
    sec_toml.write_text(
        '[tool.other]\nmatrix = [\n  "[tool.pydoppelgangerhunt]"\n]\n'
        '[tool.pydoppelgangerhunt]\nexclude = [\n  ["x", "y"]\n]\nthreshold = 0.91\n',
        encoding="utf-8",
    )
    try:
        sys.modules["tomllib"] = None  # type: ignore[assignment]
        sys.modules["tomli"] = None  # type: ignore[assignment]
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
    assert parsed_sec.get("exclude") == [["x", "y"]]
    assert parsed_sec.get("threshold") == 0.91

    # 27. Finding 1: Forward-referenced nonlocal bindings in whole functions and subroutines
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
        {
            "kind": "function",
            "start": 1,
            "end": 7,
            "source_text": fwd_nl_fn_src,
            "source_code": fwd_nl_fn_src,
        },
        tree=ast.parse(fwd_nl_fn_src),
    )
    assert fwd_nl_fn_scope["nonlocals"] == []
    assert fwd_nl_fn_scope["inputs"] == ["a"]

    fwd_nl_sub_code = (
        "def step(d: int) -> None:\n"
        "    nonlocal acc\n"
        "    acc += d\n"
        "acc = a\n"
        "step(5)\n"
    )
    fwd_nl_sub_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": fwd_nl_fn_src,
            "source_code": fwd_nl_sub_code,
        },
        tree=ast.parse(fwd_nl_fn_src),
    )
    assert fwd_nl_sub_scope["nonlocals"] == []
    assert fwd_nl_sub_scope["inputs"] == ["a"]

    # 28. Finding 2: Immediate ClassDef body execution records read_before_write before outer store
    cls_rbw_src = (
        "def build() -> type:\n"
        "    class C:\n"
        "        val = x + 1\n"
        "    x = 0\n"
        "    return C\n"
    )
    cls_rbw_scope = inspect_single_unit_scope(
        {
            "kind": "closure",
            "start": 1,
            "end": 5,
            "source_text": cls_rbw_src,
            "source_code": cls_rbw_src,
        },
        tree=ast.parse(cls_rbw_src),
    )
    assert "x" in cls_rbw_scope["inputs"]

    # 29. Finding 3: Intra-expression walrus target reads in _OuterExprVisitor
    from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor

    walrus_expr_v = _OuterExprVisitor()
    walrus_expr_v.visit(ast.parse("((w := transform(x)) + w for x in items)").body[0])
    assert "w" not in walrus_expr_v.names
    assert walrus_expr_v.ordered_names == ["items", "transform"]

    # 30. Finding 4: Multi-statement multi-line units with start_col in _resolve_unit_ast_end_col
    from pydoppelgangerhunt.fixer._df_cache import _resolve_unit_ast_end_col

    ms_tree = ast.parse("def f(x):\n    a = x + 1\n    b = a * 2\n")
    assert (
        _resolve_unit_ast_end_col(
            ms_tree.body[0], {"start": 2, "end": 3, "start_col": 4}
        )
        == 13
    )

    # 31. Issue 2.1: Nested function body free reads after conditional store in subroutine
    nested_cond_full = (
        "def outer(x: int, flag: bool) -> int:\n"
        "    if flag:\n"
        "        x = 0\n"
        "    def helper(y: int) -> int:\n"
        "        return x + y\n"
        "    return helper(1)\n"
    )
    nested_cond_sub = (
        "if flag:\n"
        "    x = 0\n"
        "def helper(y: int) -> int:\n"
        "    return x + y\n"
        "return helper(1)\n"
    )
    nested_cond_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": nested_cond_full,
            "source_code": nested_cond_sub,
        },
        tree=ast.parse(nested_cond_full),
    )
    assert "x" in nested_cond_scope["inputs"]

    # 32. Finding 1: Conditional live-in promotion requires prior binding (no false promotion from def_assigned or loop)
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

    # 33. Finding 2: Transitive deferred live-in tracking for caller-before-callee nested functions
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
    caller_callee_sub = (
        "def caller():\n"
        "    return callee()\n"
        "def callee():\n"
        "    return x\n"
        "res = caller()\n"
        "x = 10\n"
    )
    caller_callee_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 7,
            "source_text": caller_callee_src,
            "source_code": caller_callee_sub,
        },
        tree=ast.parse(caller_callee_src),
    )
    assert "x" in caller_callee_scope["inputs"]

    # 34. Finding 3: _find_enclosing_pre_bound_names discards prior del and skips wildcard "*" imports
    from pydoppelgangerhunt.fixer.scope import _find_enclosing_pre_bound_names

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

    # 35. Finding 4: _extract_helper_symbols ignores "*" on wildcard imports
    star_helper_tree = ast.parse("def _shared_fn():\n    from math import *\n    return 1\n")
    star_free, star_def, star_imp = _extract_helper_symbols(star_helper_tree)
    assert "*" not in star_free
    assert "*" not in star_def
    assert "*" not in star_imp

    # 36. Multi-statement semicolon line ending in a multi-line statement in _resolve_unit_ast_end_col
    semi_ml_tree = ast.parse("def f():\n    a = 1; b = (\n        2\n    )\n")
    assert (
        _resolve_unit_ast_end_col(
            semi_ml_tree.body[0], {"start": 2, "end": 4, "start_col": 4}
        )
        == 5
    )

    # 37. Wildcard "_" pattern target ignored in _extract_helper_symbols
    if sys.version_info >= (3, 10):
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

    # 38. Finding 1: ast.For target traversal in _analyze_block_assignment only records Store names
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

    for_sub_stmts = ast.parse("for arr[idx] in items:\n    pass\n").body
    for_def, for_cond = _analyze_block_assignment(for_sub_stmts)
    assert "arr" not in for_cond
    assert "idx" not in for_cond
    assert "arr" not in for_def

    # 39. Finding 2: Branch-nested del in _find_enclosing_pre_bound_names keeps pre-bound name
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
    pre_bound_branch_del, _ = _find_enclosing_pre_bound_names(
        ast.parse(branch_del_src), 5, 7
    )
    assert "x" in pre_bound_branch_del

    # 40. Finding 3: PEP 695 TypeAlias binding in _find_enclosing_pre_bound_names isolates lazy walrus
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

    # 41. Finding 4: _extract_helper_symbols records ast.Del target reads for unbound del
    del_free, _, _ = _extract_helper_symbols(
        ast.parse("def _shared_del():\n    del ExternalSym\n")
    )
    assert "ExternalSym" in del_free

    # 42. Finding 1: Bare ast.AnnAssign initialized via nested nonlocal in whole functions and subroutines
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
        {
            "kind": "function",
            "start": 1,
            "end": 7,
            "source_text": bare_nl_fn_src,
            "source_code": bare_nl_fn_src,
        },
        tree=ast.parse(bare_nl_fn_src),
    )
    assert bare_nl_fn_scope["nonlocals"] == []
    assert bare_nl_fn_scope["inputs"] == ["a"]

    bare_nl_sub_code = (
        "acc: int\n"
        "def init() -> None:\n"
        "    nonlocal acc\n"
        "    acc = a\n"
        "init()\n"
    )
    bare_nl_sub_scope = inspect_single_unit_scope(
        {
            "kind": "compound_block",
            "start": 2,
            "end": 6,
            "source_text": bare_nl_fn_src,
            "source_code": bare_nl_sub_code,
        },
        tree=ast.parse(bare_nl_fn_src),
    )
    assert bare_nl_sub_scope["nonlocals"] == []
    assert bare_nl_sub_scope["inputs"] == ["a"]
    assert "acc" in bare_nl_sub_scope["outputs"]

    # 43. Finding 2: _FuncScopeVisitor.visit_AugAssign records nonlocal target load_nodes in source order
    from pydoppelgangerhunt.fixer._df_visitors import (
        _extract_nested_scope_free_reads_with_nodes,
    )

    nl_aug_fn = ast.parse(
        "def inner():\n"
        "    nonlocal z_val, a_val\n"
        "    z_val += 1\n"
        "    a_val += 2\n"
    ).body[0]
    assert isinstance(nl_aug_fn, ast.FunctionDef)
    nl_aug_reads, nl_aug_nodes = _extract_nested_scope_free_reads_with_nodes(nl_aug_fn)
    assert nl_aug_reads == {"z_val", "a_val"}
    assert [n.id for n in nl_aug_nodes] == ["z_val", "a_val"]

    # 44. Finding 3: _prune_unreferenced_lhs_scope preserves receiver binding when RHS uses super()
    from pydoppelgangerhunt.fixer.synthesis import _prune_unreferenced_lhs_scope

    super_rhs_lines = ["self.res = [super().transform(x) for x in items]\n"]
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
    _prune_unreferenced_lhs_scope(super_unit, super_scope, super_rhs_lines)
    assert "self" in super_scope["inputs"]
    assert super_scope["has_instance_binding"] is True
    assert super_scope["binding_kind"] == "instance"


_CLOSURE_CELL_SCENARIOS = [
    # --- Positive cases (hazard raised) ---
    (
        "post_unit_mutated_cell",
        (
            "def build_post(items):\n"
            "    scale = 1\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        callbacks.append(lambda x, item=item: x * scale + item)\n"
            "    scale = 10\n"
            "    return callbacks\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    callbacks.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "loop_mutated_cell",
        (
            "def build_loop(items):\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        def cb(x):\n"
            "            return x + item\n"
            "        callbacks.append(cb)\n"
            "    return callbacks\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": "def cb(x):\n    return x + item\ncallbacks.append(cb)\n",
        },
        True,
    ),
    (
        "intra_unit_nonlocal_mutator_with_escaping_cell",
        (
            "def outer_inc():\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    inc()\n"
            "    return count\n"
        ),
        {
            "start": 2,
            "end": 5,
            "kind": "compound_block",
            "source_code": "count = 0\ndef inc():\n    nonlocal count\n    count += 1\n",
        },
        True,
    ),
    (
        "pre_unit_sibling_closure_sharing_mutated_cell",
        (
            "def outer_sib(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(get_acc())\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(get_acc())\n",
        },
        True,
    ),
    (
        "pre_unit_closure_mutating_nonlocal_cell_passed_to_unit",
        (
            "def outer_nl_sib(items):\n"
            "    acc = 0\n"
            "    def bump():\n"
            "        nonlocal acc\n"
            "        acc += 1\n"
            "    for x in items:\n"
            "        bump()\n"
            "        use(acc + x)\n"
            "    return acc\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    bump()\n    use(acc + x)\n",
        },
        True,
    ),
    (
        "multilevel_intermediate_nonlocal_mutation",
        (
            "def outer_inter(items):\n"
            "    scale = 1\n"
            "    def inner_inter():\n"
            "        nonlocal scale\n"
            "        cbs = []\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "        scale = 20\n"
            "        return cbs\n"
            "    return inner_inter\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "whole_closure_post_unit_cell_mutation",
        (
            "def make_handlers(items):\n"
            "    scale = 1\n"
            "    def handle(x):\n"
            "        total = 0\n"
            "        for item in items:\n"
            "            total += (x + item) * scale\n"
            "        return total\n"
            "    scale = 10\n"
            "    return handle\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def handle(x):\n"
                "    total = 0\n"
                "    for item in items:\n"
                "        total += (x + item) * scale\n"
                "    return total\n"
            ),
        },
        True,
    ),
    (
        "whole_closure_nonlocal_store",
        (
            "def make_counter():\n"
            "    count = 0\n"
            "    def step(delta):\n"
            "        nonlocal count\n"
            "        count += delta\n"
            "        return count\n"
            "    return step\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def step(delta):\n"
                "    nonlocal count\n"
                "    count += delta\n"
                "    return count\n"
            ),
        },
        True,
    ),
    (
        "assigned_lambda_sibling_closure_in_check4",
        (
            "def outer_lam(items):\n"
            "    bias = 0\n"
            "    get_bias = lambda: bias\n"
            "    out = []\n"
            "    for x in items:\n"
            "        bias += x\n"
            "        out.append(get_bias())\n"
            "    return bias, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    bias += x\n    out.append(get_bias())\n",
        },
        True,
    ),
    (
        "intra_unit_mutator_escaping_via_container_to_pre_unit_reader",
        (
            "def outer_escape(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    callbacks.append(inc)\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "callbacks.append(inc)\n"
            ),
        },
        True,
    ),
    (
        "classdef_namespace_isolation_shadowing",
        (
            "def outer_cls(items):\n"
            "    class C:\n"
            "        scale = 1\n"
            "        def m(self):\n"
            "            cbs = []\n"
            "            for item in items:\n"
            "                cbs.append(lambda x, item=item: x * scale + item)\n"
            "            return cbs\n"
            "    scale = 10\n"
            "    return C\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "pre_bound_outer_var_mutated_by_intra_unit_closure",
        (
            "def outer_inc():\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    inc()\n"
            "    return count\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": "def inc():\n    nonlocal count\n    count += 1\ninc()\n",
        },
        True,
    ),
    (
        "nested_closure_unit_with_inner_nonlocal_mutation",
        (
            "def outer_mid():\n"
            "    count = 0\n"
            "    def middle():\n"
            "        def inner():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        inner()\n"
            "    return middle, count\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_code": (
                "def middle():\n"
                "    def inner():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "    inner()\n"
            ),
        },
        True,
    ),
    (
        "pre_unit_anon_lambda_and_decorator_escape",
        (
            "def outer_anon(callbacks, readers):\n"
            "    readers.append(lambda: count)\n"
            "    count = 0\n"
            "    @callbacks.append\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
        ),
        {
            "start": 3,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "@callbacks.append\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
            ),
        },
        True,
    ),
    (
        "conditional_delete_in_enclosing_if",
        (
            "def build_cdel(items, cond):\n"
            "    scale = 1\n"
            "    if cond:\n"
            "        del scale\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "conditional_store_before_span_in_if",
        (
            "def build_cstore(items, cond):\n"
            "    if cond:\n"
            "        scale = 1\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "conditional_store_before_span_in_suppressing_with",
        (
            "def build_with_suppress(items, d):\n"
            "    with suppress(KeyError):\n"
            "        scale = d['x']\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "transitive_sibling_closure_calls",
        (
            "def outer_trans(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    def step():\n"
            "        return get_acc() * 2\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(step())\n"
            "    return acc, out\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(step())\n",
        },
        True,
    ),
    (
        "lazy_genexp_walrus_mutator",
        (
            "def run_genexp_walrus(items):\n"
            "    acc = 0\n"
            "    if items:\n"
            "        gen = ((acc := acc + x) for x in items)\n"
            "    else:\n"
            "        gen = iter(())\n"
            "    list(gen)\n"
            "    return acc\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    gen = ((acc := acc + x) for x in items)\n"
                "else:\n"
                "    gen = iter(())\n"
            ),
        },
        True,
    ),
    (
        "except_as_target_implicit_exit_delete",
        (
            "def run_except_as(items, fn):\n"
            "    cbs = []\n"
            "    try:\n"
            "        fn()\n"
            "    except ValueError as exc:\n"
            "        for x in items:\n"
            "            cbs.append(lambda x=x: (x, exc))\n"
            "    return cbs\n"
        ),
        {
            "start": 6,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (x, exc))\n",
        },
        True,
    ),
    (
        "self_name_rebind_in_outer_scope",
        (
            "def outer_rebind_fn(items):\n"
            "    def scale():\n"
            "        cbs = []\n"
            "        for x in items:\n"
            "            cbs.append(lambda x=x: (x, scale))\n"
            "        return cbs\n"
            "    res = scale()\n"
            "    scale = 10\n"
            "    return res\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (x, scale))\n",
        },
        True,
    ),
    (
        "async_coroutine_nonlocal_mutation_across_await",
        (
            "def outer_async(items, checkpoint):\n"
            "    total = 0\n"
            "    async def worker():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "            await checkpoint()\n"
            "    return worker, lambda: total\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for x in items:\n    total += x\n    await checkpoint()\n",
        },
        True,
    ),
    (
        "readonly_generator_capturing_mutable_outer_cell",
        (
            "def outer_ro_gen(items):\n"
            "    scale = 2\n"
            "    def set_scale(v):\n"
            "        nonlocal scale\n"
            "        scale = v\n"
            "    def gen():\n"
            "        for x in items:\n"
            "            yield scale * x\n"
            "    return gen, set_scale\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": "for x in items:\n    yield scale * x\n",
        },
        True,
    ),
    (
        "aliased_sibling_closure_callback",
        (
            "def outer_alias_container(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    reader = get_acc\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(reader())\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(reader())\n",
        },
        True,
    ),
    (
        "container_stored_sibling_closure_callback",
        (
            "def outer_container_cb(items):\n"
            "    acc = 0\n"
            "    readers = [lambda: acc]\n"
            "    out = []\n"
            "    for y in items:\n"
            "        acc += y\n"
            "        out.append(readers[0]())\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": "for y in items:\n    acc += y\n    out.append(readers[0]())\n",
        },
        True,
    ),
    (
        "nested_def_default_walrus_in_closure",
        (
            "def outer_def_walrus(items):\n"
            "    acc = 0\n"
            "    def make_fn():\n"
            "        nonlocal acc\n"
            "        def inner(val=(acc := acc + 1)):\n"
            "            return val\n"
            "        return inner\n"
            "    out = []\n"
            "    for x in items:\n"
            "        fn = make_fn()\n"
            "        out.append(fn() + acc + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    fn = make_fn()\n"
                "    out.append(fn() + acc + x)\n"
            ),
        },
        True,
    ),
    # --- Negative cases (no hazard raised) ---
    (
        "neg_closure_capturing_never_reassigned_parameter",
        (
            "def build_param_only(items, factor):\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v: v * factor)\n"
            "    return cbs\n"
        ),
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_code": "for x in items:\n    cbs.append(lambda v: v * factor)\n",
        },
        False,
    ),
    (
        "neg_item_default_binding_with_constant_cell",
        (
            "def build_safe(items, scale):\n"
            "    callbacks = []\n"
            "    for item in items:\n"
            "        callbacks.append(lambda x, item=item: x * scale + item)\n"
            "    return callbacks\n"
        ),
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    callbacks.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_nonlocal_mutation_in_unit_with_no_escaping_readers",
        (
            "def compute_local_nl(items):\n"
            "    if items:\n"
            "        count = 0\n"
            "        def bump(step):\n"
            "            nonlocal count\n"
            "            count += step\n"
            "        for x in items:\n"
            "            bump(x)\n"
            "    return len(items)\n"
        ),
        {
            "start": 2,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    count = 0\n"
                "    def bump(step):\n"
                "        nonlocal count\n"
                "        count += step\n"
                "    for x in items:\n"
                "        bump(x)\n"
            ),
        },
        False,
    ),
    (
        "neg_eagerly_consumed_genexp_capturing_later_mutated_var",
        (
            "def compute_eager_genexp(items):\n"
            "    scale = 2\n"
            "    if items:\n"
            "        total = sum(v * scale for v in items)\n"
            "    else:\n"
            "        total = 0\n"
            "    scale = 10\n"
            "    return total + scale\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    total = sum(v * scale for v in items)\n"
                "else:\n"
                "    total = 0\n"
            ),
        },
        False,
    ),
    (
        "neg_eagerly_consumed_genexp_walrus_not_lazy_mutator",
        (
            "def compute_eager_walrus(items):\n"
            "    acc = 0\n"
            "    if items:\n"
            "        total = sum((acc := acc + v) for v in items)\n"
            "    else:\n"
            "        total = 0\n"
            "    return total + acc\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    total = sum((acc := acc + v) for v in items)\n"
                "else:\n"
                "    total = 0\n"
            ),
        },
        False,
    ),
    (
        "neg_call_result_passed_to_container_append_not_closure_alias",
        (
            "def run_no_alias(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    out = []\n"
            "    out.append(get_acc())\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    out.append(x)\n",
        },
        False,
    ),
    (
        "neg_binding_in_same_enclosing_with_block_is_pre_bound",
        (
            "def build_in_same_with(items, ctx):\n"
            "    cbs = []\n"
            "    with ctx:\n"
            "        scale = 2\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_lazy_genexp_first_iterable_evaluated_eagerly",
        (
            "def build_gen_first_iter(items):\n"
            "    seq = list(items)\n"
            "    if seq:\n"
            "        gen = (x * 2 for x in seq)\n"
            "    else:\n"
            "        gen = iter(())\n"
            "    seq = []\n"
            "    return list(gen)\n"
        ),
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if seq:\n"
                "    gen = (x * 2 for x in seq)\n"
                "else:\n"
                "    gen = iter(())\n"
            ),
        },
        False,
    ),
    (
        "multilevel_intermediate_pre_unit_nonlocal_mutation",
        (
            "def outer_inter_pre(items):\n"
            "    scale = 1\n"
            "    def inner_inter():\n"
            "        nonlocal scale\n"
            "        scale += 1\n"
            "        cbs = []\n"
            "        for item in items:\n"
            "            cbs.append(lambda x, item=item: x * scale + item)\n"
            "        return cbs\n"
            "    return inner_inter\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "pre_unit_closure_factory_assigned_to_variable",
        (
            "def outer_factory(items):\n"
            "    acc = 0\n"
            "    def make_closure():\n"
            "        def read_acc():\n"
            "            return acc\n"
            "        return read_acc\n"
            "    fn = make_closure()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(fn() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(fn() + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_exhaustive_if_else_branch_initialization_before_unit",
        (
            "def build_if_else_init(items, cond):\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        scale = 1\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_enclosing_local_class_method_synchronous_nonlocal_mutation",
        (
            "def outer_cls_method(log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def run(self, items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(x)\n"
            "    return Worker\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "outer_global_captured_by_intra_unit_closure",
        (
            "def outer_glob_cap(items):\n"
            "    global shared_g\n"
            "    shared_g = 1\n"
            "    out = []\n"
            "    for x in items:\n"
            "        out.append(lambda y, x=x: y + shared_g + x)\n"
            "    shared_g = 2\n"
            "    return out\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    out.append(lambda y, x=x: y + shared_g + x)\n"
            ),
        },
        True,
    ),
    (
        "multilevel_outer_loop_escaping_nonlocal_unit_mutation",
        (
            "def outer_loop_nl(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    for grp in groups:\n"
            "        def inner_step(x):\n"
            "            nonlocal total\n"
            "            if x > 0:\n"
            "                total += x\n"
            "                cbs.append(lambda y: y + total + x)\n"
            "        inner_step(grp)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "if x > 0:\n"
                "    total += x\n"
                "    cbs.append(lambda y: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "eager_comprehension_aliased_sibling_closure",
        (
            "def outer_comp_alias(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    readers = [get_acc for _ in range(3)]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(readers[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(readers[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "nested_class_body_nonlocal_unbound_conditional_mutation",
        (
            "def outer_cls_nl(items):\n"
            "    def mid():\n"
            "        class Box:\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        return Box\n"
            "    total = 0\n"
            "    return mid\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total = x * 2\n"
            ),
        },
        True,
    ),
    (
        "generator_mutating_local_cell_captured_by_sibling_closure",
        (
            "def outer_gen_sib(items, callbacks):\n"
            "    acc = 0\n"
            "    callbacks.append(lambda: acc)\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        yield x\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc += x\n    yield x\n",
        },
        True,
    ),
    (
        "multicall_nested_fn_mutating_and_capturing_nonlocal_cell",
        (
            "def outer_mc():\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    inner_step([1])\n"
            "    inner_step([2])\n"
            "    return cbs\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_exhaustive_if_else_after_del_rebinds_before_unit",
        (
            "def build_del_rebind(items, cond):\n"
            "    scale = 0\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        del scale\n"
            "        scale = 1\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "exhaustive_if_else_with_nested_single_branch_del_not_definite",
        (
            "def build_nested_del(items, cond, sub):\n"
            "    cbs = []\n"
            "    if cond:\n"
            "        scale = 1\n"
            "        if sub:\n"
            "            del scale\n"
            "    else:\n"
            "        scale = 2\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "closure_factory_returning_sibling_closure_and_pop",
        (
            "def outer_ret_sib(items):\n"
            "    acc = 0\n"
            "    def read_acc():\n"
            "        return acc\n"
            "    def get_reader():\n"
            "        return read_acc\n"
            "    cbs = [get_reader()]\n"
            "    fn = cbs.pop()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(fn() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 10,
            "end": 12,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(fn() + x)\n"
            ),
        },
        True,
    ),
    (
        "generator_factory_assigned_to_variable",
        (
            "def outer_gen_fac(items):\n"
            "    acc = 0\n"
            "    def make_gen():\n"
            "        yield acc\n"
            "    g = make_gen()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(g) + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(g) + x)\n"
            ),
        },
        True,
    ),
    (
        "intra_unit_mutator_self_registering_in_body",
        (
            "def outer_self_reg(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "        callbacks.append(inc)\n"
            "    inc()\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "    callbacks.append(inc)\n"
                "inc()\n"
            ),
        },
        True,
    ),
    (
        "intra_unit_nested_factory_returning_mutator",
        (
            "def outer_nested_fac(callbacks):\n"
            "    def get_count():\n"
            "        return count\n"
            "    count = 0\n"
            "    def make_inc():\n"
            "        def inc():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        return inc\n"
            "    callbacks.append(make_inc())\n"
            "    return get_count\n"
        ),
        {
            "start": 4,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def make_inc():\n"
                "    def inc():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "    return inc\n"
                "callbacks.append(make_inc())\n"
            ),
        },
        True,
    ),
    (
        "neg_sync_subroutine_nonlocal_mut_with_post_def_outer_binding",
        (
            "def outer_post_def(log):\n"
            "    total = 0\n"
            "    def worker(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x * 2\n"
            "            log(x)\n"
            "    worker([1, 2])\n"
            "    total += 1\n"
            "    return total\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "neg_len_of_closure_container_is_not_closure_alias",
        (
            "def outer_len_cbs(items):\n"
            "    acc = 0\n"
            "    callbacks = [lambda: acc]\n"
            "    n = len(callbacks)\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(n + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(n + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_method_inside_class_does_not_inherit_class_body_nonlocal",
        (
            "def outer_cls_method_iso(items, log):\n"
            "    total = 0\n"
            "    class Box:\n"
            "        nonlocal total\n"
            "        total = 1\n"
            "        def run(self):\n"
            "            total = 0\n"
            "            for x in items:\n"
            "                total += x\n"
            "                log(x)\n"
            "            return total\n"
            "    return Box, total\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "eager_listcomp_walrus_body_is_conditional_not_pre_bound",
        (
            "def build_comp_walrus(items, empty_seq):\n"
            "    cbs = []\n"
            "    _ = [(scale := v) for v in empty_seq]\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "neg_lazy_genexp_first_iterable_walrus_is_eager_not_lazy_mutator",
        (
            "def build_gen_hdr_walrus(items):\n"
            "    if items:\n"
            "        gen = (x * 2 for x in (seq := list(items)))\n"
            "    else:\n"
            "        seq = []\n"
            "        gen = iter(())\n"
            "    return list(gen), seq\n"
        ),
        {
            "start": 2,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "if items:\n"
                "    gen = (x * 2 for x in (seq := list(items)))\n"
                "else:\n"
                "    seq = []\n"
                "    gen = iter(())\n"
            ),
        },
        False,
    ),
    (
        "comprehension_append_sibling_closure_carrier",
        (
            "def outer_comp_append(items):\n"
            "    acc = 0\n"
            "    readers = []\n"
            "    _ = [readers.append(lambda: acc) for _ in range(2)]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(readers[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(readers[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "multilevel_sibling_closure_nonlocal_deletion_in_inner_scope",
        (
            "def outer_ml_del(items):\n"
            "    total = 5\n"
            "    def inner():\n"
            "        def clear():\n"
            "            nonlocal total\n"
            "            del total\n"
            "        clear()\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total = x * 2\n"
            "    return inner\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total = x * 2\n"
            ),
        },
        True,
    ),
    (
        "conditional_augassign_on_unbound_nonlocal_cell",
        (
            "def outer_unb_aug():\n"
            "    def inner_unb(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "    inner_unb([-1, -2])\n"
            "    total = 0\n"
            "    return inner_unb\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    if x > 0:\n"
                "        total += x * 2\n"
            ),
        },
        True,
    ),
    (
        "neg_len_of_inline_lambda_comprehension_is_not_closure_alias",
        (
            "def outer_len_inline_comp(items):\n"
            "    acc = 0\n"
            "    n = len([lambda: acc for _ in range(3)])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(n + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(n + x)\n"
            ),
        },
        False,
    ),
    (
        "generator_loop_initialized_cell_captured_by_pre_unit_sibling_closure",
        (
            "def outer_gen_loop_init(items, callbacks):\n"
            "    callbacks.append(lambda: acc)\n"
            "    for x in items:\n"
            "        acc = x * 2\n"
            "        yield x\n"
        ),
        {
            "start": 3,
            "end": 5,
            "kind": "compound_block",
            "source_code": "for x in items:\n    acc = x * 2\n    yield x\n",
        },
        True,
    ),
    (
        "multicall_enclosed_fn_invoked_in_comprehension",
        (
            "def outer_comp_call(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    _ = [inner_step(g) for g in groups]\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "multicall_enclosed_fn_escaping_via_return",
        (
            "def outer_ret_fn():\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "    return inner_step, cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "lazy_genexp_first_iterable_closure_carrier_alias",
        (
            "def outer_lazy_hdr_carrier(items):\n"
            "    acc = 0\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    gen = (f() for f in [get_acc])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(gen) + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(gen) + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_helper_with_eager_sum_genexp_is_not_closure_factory",
        (
            "def outer_sum_gen(items):\n"
            "    acc = 0\n"
            "    def compute_initial():\n"
            "        return sum(x * acc for x in items)\n"
            "    init_val = compute_initial()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(init_val + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(init_val + x)\n"
            ),
        },
        False,
    ),
    (
        "conditional_walrus_in_comprehension_first_iterable_not_pre_bound",
        (
            "def build_nested_hdr_walrus(items, empty_seq):\n"
            "    cbs = []\n"
            "    _ = [x for x in [(scale := v) for v in empty_seq]]\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        True,
    ),
    (
        "neg_comprehension_target_shadowing_outer_cell_in_lambda",
        (
            "def outer_shadow_target(items):\n"
            "    acc = [1, 2]\n"
            "    cbs = [lambda: acc for acc in acc]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc = [x]\n"
            "        out.append(cbs[0]() + x)\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc = [x]\n"
                "    out.append(cbs[0]() + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_pre_bound_var_deleted_and_rebound_in_one_if_branch",
        (
            "def build_del_rebind_branch(items, cond):\n"
            "    scale = 1\n"
            "    if cond:\n"
            "        del scale\n"
            "        scale = 2\n"
            "    else:\n"
            "        pass\n"
            "    cbs = []\n"
            "    for item in items:\n"
            "        cbs.append(lambda x, item=item: x * scale + item)\n"
            "    return cbs\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for item in items:\n"
                "    cbs.append(lambda x, item=item: x * scale + item)\n"
            ),
        },
        False,
    ),
    (
        "neg_multilevel_shadowed_sibling_closure_cell_no_sharing",
        (
            "def outer_shadow_sib(items):\n"
            "    acc = 100\n"
            "    def get_acc():\n"
            "        return acc\n"
            "    def inner():\n"
            "        acc = 0\n"
            "        out = []\n"
            "        for x in items:\n"
            "            acc += x\n"
            "            out.append(get_acc() + acc)\n"
            "        return acc, out\n"
            "    return inner\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(get_acc() + acc)\n"
            ),
        },
        False,
    ),
    (
        "multicall_enclosed_fn_defined_in_local_class",
        (
            "def outer_cls_method(groups):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    class Runner:\n"
            "        def step(self, items):\n"
            "            nonlocal total\n"
            "            total += 1\n"
            "            for x in items:\n"
            "                cbs.append(lambda y, x=x: y + total + x)\n"
            "    return Runner, cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "class_body_mutating_carrier_call_captures_shared_cell",
        (
            "def outer_cls_carrier(items):\n"
            "    acc = 0\n"
            "    cbs = []\n"
            "    class Box:\n"
            "        cbs.append(lambda: acc)\n"
            "        out = []\n"
            "        for x in items:\n"
            "            acc += x\n"
            "            out.append(cbs[0]() + x)\n"
            "    return Box, acc\n"
        ),
        {
            "start": 7,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(cbs[0]() + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_sibling_class_shadowed_attribute_not_post_read",
        (
            "def outer_sib_cls_attr(items):\n"
            "    count = 0\n"
            "    def inc():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    for x in items:\n"
            "        inc()\n"
            "    class Sibling:\n"
            "        count = 10\n"
            "        doubled = count * 2\n"
            "    return Sibling\n"
        ),
        {
            "start": 2,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "count = 0\n"
                "def inc():\n"
                "    nonlocal count\n"
                "    count += 1\n"
                "for x in items:\n"
                "    inc()\n"
            ),
        },
        False,
    ),
    (
        "multicall_enclosed_fn_self_escaping_in_own_body",
        (
            "def outer_self_rec(callbacks):\n"
            "    total = 0\n"
            "    cbs = []\n"
            "    def inner_step(items):\n"
            "        nonlocal total\n"
            "        total += 1\n"
            "        for x in items:\n"
            "            cbs.append(lambda y, x=x: y + total + x)\n"
            "        callbacks.append(inner_step)\n"
            "    inner_step([1])\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda y, x=x: y + total + x)\n"
            ),
        },
        True,
    ),
    (
        "subroutine_unit_containing_direct_nonlocal_statement",
        (
            "def outer_direct_nl(items):\n"
            "    total = 0\n"
            "    def inner():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            total += x\n"
            "    return inner, total\n"
        ),
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "nonlocal total\n"
                "for x in items:\n"
                "    total += x\n"
            ),
        },
        True,
    ),
    (
        "try_except_earlier_handler_binding_not_definite_in_later_handler",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    except TypeError:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
            "        return callback\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_in_loop_earlier_handler_binding_captured_in_later_handler",
        (
            "def outer(batches, items):\n"
            "    for batch in batches:\n"
            "        try:\n"
            "            perform_operation(batch)\n"
            "        except ValueError:\n"
            "            scale = 10\n"
            "        except TypeError:\n"
            "            for x in items:\n"
            "                callback = lambda: scale\n"
            "            return callback\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_handler_binding_not_definite_in_else_block",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    else:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
            "        return callback\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "try_except_handler_and_else_binding_not_definite_in_finally_block",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        scale = 10\n"
            "    else:\n"
            "        scale = 20\n"
            "    finally:\n"
            "        for x in items:\n"
            "            callback = lambda: scale\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    callback = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "neg_try_except_own_handler_target_and_prior_stmt_definitely_bound",
        (
            "def outer(items):\n"
            "    try:\n"
            "        perform_operation()\n"
            "    except ValueError:\n"
            "        other = 1\n"
            "    except TypeError as err:\n"
            "        scale = 10\n"
            "        for x in items:\n"
            "            msg = str(err)\n"
            "            callback = lambda: scale\n"
            "        return callback, msg\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    msg = str(err)\n"
                "    callback = lambda: scale\n"
            ),
        },
        False,
    ),
    (
        "builtin_call_arg_invoking_closure_factory_and_mutating_outer_container",
        (
            "def outer(items):\n"
            "    value = 1\n"
            "    saved = []\n"
            "    def make_callback():\n"
            "        callback = lambda: value\n"
            "        saved.append(callback)\n"
            "        return callback\n"
            "    result = len([make_callback()])\n"
            "    for x in items:\n"
            "        value = x + result\n"
            "    return saved, value\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    value = x + result\n"
            ),
        },
        True,
    ),
    (
        "builtin_call_arg_outer_container_read_by_unit",
        (
            "def outer(items):\n"
            "    value = 1\n"
            "    saved = []\n"
            "    def make_callback():\n"
            "        callback = lambda: value\n"
            "        saved.append(callback)\n"
            "        return callback\n"
            "    result = len([make_callback()])\n"
            "    for x in items:\n"
            "        value = x + len(saved)\n"
            "    return result, value\n"
        ),
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    value = x + len(saved)\n"
            ),
        },
        True,
    ),
    (
        "multiline_decorator_unit_capturing_uninitialized_decorated_fn",
        (
            "def outer(items, register):\n"
            "    @register(\n"
            "        items,\n"
            "        [lambda x: x * inner() for _ in items],\n"
            "    )\n"
            "    def inner():\n"
            "        return 2\n"
            "    return inner\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 8,
            "end_col": 46,
            "kind": "comprehension",
            "source_code": "[lambda x: x * inner() for _ in items]",
        },
        True,
    ),
    (
        "same_line_decorator_unit_capturing_late_bound_variable",
        (
            "def outer(items, dec):\n"
            "    @dec([lambda x: x * scale for _ in items])\n"
            "    def inner():\n"
            "        return scale\n"
            "    scale = 10\n"
            "    return inner\n"
        ),
        {
            "start": 2,
            "end": 2,
            "start_col": 9,
            "end_col": 45,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        True,
    ),
    (
        "nested_decorated_defs_unit_inside_inner_body_captures_outer_late_var",
        (
            "def outer(items, dec1, dec2):\n"
            "    @dec1(\n"
            "        1,\n"
            "    )\n"
            "    @dec2\n"
            "    def mid():\n"
            "        for x in items:\n"
            "            cb = lambda: scale\n"
            "        return cb\n"
            "    scale = 10\n"
            "    return mid\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cb = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "decorated_first_stmt_closure_unit_captures_late_bound_var",
        (
            "def o(d):\n"
            "    @d\n"
            "    def i():\n"
            "        return scale\n"
            "    scale = 1\n"
            "    return i\n"
        ),
        {
            "start": 2,
            "end": 4,
            "start_col": 4,
            "kind": "closure",
            "source_code": (
                "@d\n"
                "def i():\n"
                "    return scale\n"
            ),
        },
        True,
    ),
    (
        "enumerate_carrier_closure_reads_unit_mutated_var",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    pairs = enumerate([lambda: acc])\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(next(pairs)[1]())\n"
            "    return acc, out\n"
        ),
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(next(pairs)[1]())\n"
            ),
        },
        True,
    ),
    (
        "enclosing_local_class_method_calls_sibling_peek_reading_nonlocal",
        (
            "def outer(items, log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def peek(self):\n"
            "            return total\n"
            "        def run(self, items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(self.peek())\n"
            "    return Worker\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(self.peek())\n"
            ),
        },
        True,
    ),
    (
        "neg_with_as_target_definitely_pre_bound_before_unit",
        (
            "def outer(ctx, items):\n"
            "    with ctx() as fh:\n"
            "        pass\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: (fh, x))\n"
            "    return cbs\n"
        ),
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda x=x: (fh, x))\n"
            ),
        },
        False,
    ),
    (
        "neg_if_return_else_fallthrough_binding_before_unit",
        (
            "def outer(cond, items):\n"
            "    if cond:\n"
            "        y = 1\n"
            "        return []\n"
            "    else:\n"
            "        scale = 2\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_inner_fn_mutating_own_local_container_not_outer_carrier",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    holder = []\n"
            "    def helper():\n"
            "        holder = []\n"
            "        holder.append(lambda: acc)\n"
            "        return len(holder)\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        holder.append(acc)\n"
            "    return acc, holder\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    holder.append(acc)\n"
            ),
        },
        False,
    ),
    (
        "neg_factory_with_bare_annotation_shadows_outer_closure_name",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    cb = lambda: acc\n"
            "    def get_val():\n"
            "        cb: object\n"
            "        return 1\n"
            "    out = []\n"
            "    for x in items:\n"
            "        acc += x\n"
            "        out.append(get_val())\n"
            "    return acc, out, cb\n"
        ),
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    acc += x\n"
                "    out.append(get_val())\n"
            ),
        },
        False,
    ),
    (
        "neg_same_line_pre_bound_with_none_start_col_inferred",
        (
            "def outer(items):\n"
            "    scale = 10; cbs = [lambda x: x * scale for _ in items]\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        False,
    ),
    (
        "same_line_post_mutated_with_none_start_col_inferred",
        (
            "def outer(items):\n"
            "    cbs = [lambda x: x * scale for _ in items]; scale = 20\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_code": "[lambda x: x * scale for _ in items]",
        },
        True,
    ),
    (
        "unit_starting_on_comment_line_before_first_body_stmt",
        (
            "def outer(items):\n"
            "    # leading comment before first body statement\n"
            "    for x in items:\n"
            "        cb = lambda: scale\n"
            "    scale = 10\n"
            "    return cb\n"
        ),
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_code": (
                "# leading comment before first body statement\n"
                "for x in items:\n"
                "    cb = lambda: scale\n"
            ),
        },
        True,
    ),
    (
        "unit_in_multiline_class_base_header_not_treated_as_class_body",
        (
            "def outer(items, make_base):\n"
            "    scale = 1\n"
            "    class Box(\n"
            "        make_base([lambda x, item=item: x * scale + item for item in items]),\n"
            "    ):\n"
            "        nonlocal scale\n"
            "        v = Box\n"
            "    return Box\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 18,
            "end_col": 75,
            "kind": "comprehension",
            "source_code": "[lambda x, item=item: x * scale + item for item in items]",
        },
        False,
    ),
    (
        "unit_in_multiline_class_base_header_captures_class_name_bound_after_header",
        (
            "def outer(items, make_base):\n"
            "    class Box(\n"
            "        make_base([lambda x, item=item: (Box, x, item) for item in items]),\n"
            "    ):\n"
            "        pass\n"
            "    return Box\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 18,
            "end_col": 73,
            "kind": "comprehension",
            "source_code": "[lambda x, item=item: (Box, x, item) for item in items]",
        },
        True,
    ),
    (
        "comprehension_walrus_mutates_sibling_captured_cell",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    cb = lambda: acc\n"
            "    res = [(acc := acc + x) for x in items]\n"
            "    return res, cb\n"
        ),
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 43,
            "kind": "comprehension",
            "source_code": "[(acc := acc + x) for x in items]",
        },
        True,
    ),
    (
        "comprehension_walrus_mutates_enclosing_nonlocal_cell",
        (
            "def outer(items):\n"
            "    acc = 0\n"
            "    def inner():\n"
            "        nonlocal acc\n"
            "        return [(acc := acc + x) for x in items]\n"
            "    return inner, acc\n"
        ),
        {
            "start": 5,
            "end": 5,
            "start_col": 15,
            "end_col": 48,
            "kind": "comprehension",
            "source_code": "[(acc := acc + x) for x in items]",
        },
        True,
    ),
    (
        "neg_local_class_staticmethod_first_param_not_aliased_as_self",
        (
            "def outer(log):\n"
            "    total = 0\n"
            "    class Worker:\n"
            "        def peek(self):\n"
            "            return total\n"
            "        @staticmethod\n"
            "        def run(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                total += x * 2\n"
            "                log(x)\n"
            "    return Worker\n"
        ),
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    total += x * 2\n"
                "    log(x)\n"
            ),
        },
        False,
    ),
    (
        "if_branch_with_suppress_body_store_not_definite",
        (
            "def outer(cond, suppress, d, items):\n"
            "    if cond:\n"
            "        with suppress(KeyError):\n"
            "            scale = d['x']\n"
            "    else:\n"
            "        scale = 2\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        True,
    ),
    (
        "neg_single_branch_if_del_and_return_preserves_fallthrough_binding",
        (
            "def outer(cond, items):\n"
            "    scale = 2\n"
            "    if cond:\n"
            "        del scale\n"
            "        return []\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_try_finally_without_handlers_definitely_binds_on_fallthrough",
        (
            "def outer(cleanup, items):\n"
            "    try:\n"
            "        scale = 2\n"
            "    finally:\n"
            "        cleanup()\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda v, x=x: v * scale + x)\n"
            "    return cbs\n"
        ),
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda v, x=x: v * scale + x)\n"
            ),
        },
        False,
    ),
    (
        "neg_same_line_identifier_boundary_in_unit_cols_fallback",
        (
            "def outer(items):\n"
            "    prefix_cbs = [lambda x: x * scale for _ in items]; scale = 10; cbs = [lambda x: x * scale for _ in items]\n"
            "    return cbs\n"
        ),
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_code": "cbs = [lambda x: x * scale for _ in items]",
        },
        False,
    ),
]


@pytest.mark.parametrize(
    "scenario_id,src,unit_spec,expect_hazard",
    _CLOSURE_CELL_SCENARIOS,
    ids=[s[0] for s in _CLOSURE_CELL_SCENARIOS],
)
def test_severed_closure_cell_scenarios(
    scenario_id: str,
    src: str,
    unit_spec: dict[str, object],
    expect_hazard: bool,
) -> None:
    """Verifies positive and negative severed_closure_cell scenarios individually."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope

    _ = scenario_id
    unit = dict(unit_spec)
    unit["source_text"] = src
    scope = inspect_single_unit_scope(unit, tree=ast.parse(src))
    if expect_hazard:
        assert "severed_closure_cell" in scope["control_flow_hazards"]
        assert scope["is_control_flow_safe"] is False
    else:
        assert "severed_closure_cell" not in scope["control_flow_hazards"]
        assert scope["is_control_flow_safe"] is True


def test_closure_cell_binding_and_refactoring_integration(tmp_path: Path) -> None:
    """Verifies end-to-end refactoring, skip reasons, and nonlocal promotion."""
    from pydoppelgangerhunt.fixer import (
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_pair_outputs,
        synthesize_shared_helper_code,
    )

    # 1. Generator subroutine inside nested closure with outer nonlocal declaration
    gen_nl_src = (
        "def outer_a(items):\n"
        "    total = 0\n"
        "    def gen_a():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            yield x\n"
        "    return gen_a, lambda: total\n"
        "\n"
        "def outer_b(items):\n"
        "    total = 10\n"
        "    def gen_b():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            yield x\n"
        "    return gen_b, lambda: total\n"
    )
    f_gen_nl = tmp_path / "gen_nl.py"
    f_gen_nl.write_text(gen_nl_src, encoding="utf-8")
    u_gen1 = {
        "file": str(f_gen_nl),
        "start": 5,
        "end": 7,
        "name": "gen_a:For",
        "kind": "compound_block",
    }
    u_gen2 = {
        "file": str(f_gen_nl),
        "start": 14,
        "end": 16,
        "name": "gen_b:For",
        "kind": "compound_block",
    }
    assert collect_downstream_read_names(gen_nl_src, u_gen1, candidates={"total", "x"}) == {"total"}
    ds_nl_all = collect_downstream_read_names(gen_nl_src, u_gen1, candidates=None)
    assert ds_nl_all is not None and "total" in ds_nl_all
    sc_gen_nl = inspect_single_unit_scope(u_gen1, repo_root=str(tmp_path))
    assert "severed_closure_cell" in sc_gen_nl["control_flow_hazards"]
    assert (
        generate_refactoring_patch(
            [(1.0, u_gen1, u_gen2)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )

    # Synchronous nonlocal mutation without suspension safely refactors
    sync_nl_src = (
        "def outer_a(items):\n"
        "    total = 0\n"
        "    def worker_a():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            log(x)\n"
        "    return worker_a, lambda: total\n"
        "\n"
        "def outer_b(items):\n"
        "    total = 10\n"
        "    def worker_b():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total += x * 2\n"
        "            log(x)\n"
        "    return worker_b, lambda: total\n"
    )
    f_sync_nl = tmp_path / "sync_nl.py"
    f_sync_nl.write_text(sync_nl_src, encoding="utf-8")
    u_sync1 = {
        "file": str(f_sync_nl),
        "start": 5,
        "end": 7,
        "name": "worker_a:For",
        "kind": "compound_block",
    }
    u_sync2 = {
        "file": str(f_sync_nl),
        "start": 14,
        "end": 16,
        "name": "worker_b:For",
        "kind": "compound_block",
    }
    sc_sync1 = inspect_single_unit_scope(u_sync1, repo_root=str(tmp_path))
    assert "severed_closure_cell" not in sc_sync1["control_flow_hazards"]
    patch_sync_nl = generate_refactoring_patch(
        [(1.0, u_sync1, u_sync2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "total = _shared_" in patch_sync_nl

    # 2. Conditional store to pre-bound nonlocal promotes variable to input instead of None
    cond_nl_src = (
        "def run_a(items):\n"
        "    total = 5\n"
        "    def worker():\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            if x > 0:\n"
        "                total = x * 2\n"
        "    worker()\n"
        "    return total\n"
    )
    sc_cond_nl = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": cond_nl_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(cond_nl_src),
    )
    assert "total" in sc_cond_nl["inputs"]
    assert "total" in sc_cond_nl["outputs"]
    assert "total" not in sc_cond_nl["conditional_outputs"]
    assert "severed_closure_cell" not in sc_cond_nl["control_flow_hazards"]

    # Unbound nonlocal is not promoted to input and triggers severed_closure_cell
    unbound_nl_src = (
        "def outer_unb():\n"
        "    def inner_unb(items):\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            if x > 0:\n"
        "                total = x * 2\n"
        "    total = 0\n"
        "    return inner_unb\n"
    )
    sc_unbound_nl = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": unbound_nl_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(unbound_nl_src),
    )
    assert "total" not in sc_unbound_nl["inputs"]
    assert "total" in sc_unbound_nl["conditional_outputs"]
    assert "severed_closure_cell" in sc_unbound_nl["control_flow_hazards"]

    # Branch-only or deleted pre-unit nonlocal (including post-def del or sibling del) is not promoted to input
    for bad_init_stmt, post_stmt in (
        ("if flag:\n        total = 5", "return inner_bd"),
        ("total = 5\n    del total", "return inner_bd"),
        ("total = 5", "del total\n    return inner_bd"),
        (
            "total = 5\n    def clear():\n        nonlocal total\n        del total",
            "return inner_bd, clear",
        ),
    ):
        branch_or_del_nl_src = (
            "def outer_bd(items, flag):\n"
            f"    {bad_init_stmt}\n"
            "    def inner_bd():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total = x * 2\n"
            f"    {post_stmt}\n"
        )
        bd_tree = ast.parse(branch_or_del_nl_src)
        for_node = next(n for n in ast.walk(bd_tree) if isinstance(n, ast.For))
        sc_bd_nl = inspect_single_unit_scope(
            {
                "start": for_node.lineno,
                "end": for_node.end_lineno,
                "kind": "compound_block",
                "source_text": branch_or_del_nl_src,
                "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
            },
            tree=bd_tree,
        )
        assert "total" not in sc_bd_nl["inputs"]
        assert "total" in sc_bd_nl["conditional_outputs"]
        assert "severed_closure_cell" in sc_bd_nl["control_flow_hazards"]

    # Lexical shadowing (mid local unbound at inner) and global in mid do not promote outer cell
    for shadow_or_glob_src in (
        (
            "def outer_sg():\n"
            "    total = 5\n"
            "    def mid():\n"
            "        def inner(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        total = 0\n"
            "        return inner\n"
            "    return mid\n"
        ),
        (
            "def outer_sg():\n"
            "    def mid():\n"
            "        global total\n"
            "        total = 10\n"
            "        def inner(items):\n"
            "            nonlocal total\n"
            "            for x in items:\n"
            "                if x > 0:\n"
            "                    total = x * 2\n"
            "        return inner\n"
            "    total = 5\n"
            "    return mid\n"
        ),
    ):
        sg_tree = ast.parse(shadow_or_glob_src)
        sg_for = next(n for n in ast.walk(sg_tree) if isinstance(n, ast.For))
        sc_sg = inspect_single_unit_scope(
            {
                "start": sg_for.lineno,
                "end": sg_for.end_lineno,
                "kind": "compound_block",
                "source_text": shadow_or_glob_src,
                "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
            },
            tree=sg_tree,
        )
        assert "total" not in sc_sg["inputs"]
        assert "severed_closure_cell" in sc_sg["control_flow_hazards"]

    # Post-span del inside enclosing loop unbinds nonlocal for subsequent iterations
    loop_post_del_src = (
        "def outer_loop_del():\n"
        "    total = 5\n"
        "    def inner(groups):\n"
        "        nonlocal total\n"
        "        for items in groups:\n"
        "            for x in items:\n"
        "                if x > 0:\n"
        "                    total = x * 2\n"
        "            del total\n"
        "    return inner\n"
    )
    sc_lpd = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": loop_post_del_src,
            "source_code": "for x in items:\n    if x > 0:\n        total = x * 2\n",
        },
        tree=ast.parse(loop_post_del_src),
    )
    assert "total" not in sc_lpd["inputs"]
    assert "severed_closure_cell" in sc_lpd["control_flow_hazards"]

    # Conditional live-in read/AugAssign followed by unconditional write or branch-only pre_bound
    for cond_live_nl_src, u_s, u_e, u_code in (
        (
            "def outer_late_nl():\n"
            "    def inner(items):\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "        total = 0\n"
            "    total = 10\n"
            "    return inner\n",
            4,
            7,
            "for x in items:\n    if x > 0:\n        total += x * 2\ntotal = 0\n",
        ),
        (
            "def outer_branch_aug(items, flag):\n"
            "    if flag:\n"
            "        total = 5\n"
            "    def inner():\n"
            "        nonlocal total\n"
            "        for x in items:\n"
            "            if x > 0:\n"
            "                total += x * 2\n"
            "    return inner\n",
            6,
            8,
            "for x in items:\n    if x > 0:\n        total += x * 2\n",
        ),
    ):
        sc_cl = inspect_single_unit_scope(
            {
                "start": u_s,
                "end": u_e,
                "kind": "compound_block",
                "source_text": cond_live_nl_src,
                "source_code": u_code,
            },
            tree=ast.parse(cond_live_nl_src),
        )
        assert "total" not in sc_cl["inputs"]
        assert "severed_closure_cell" in sc_cl["control_flow_hazards"]

    # synthesize_shared_helper_code(closure_strictness='lenient') does not mutate caller unit dicts
    lenient_u1 = dict(u_sync1)
    lenient_u2 = dict(u_sync2)
    _ = synthesize_shared_helper_code(
        lenient_u1,
        lenient_u2,
        repo_root=str(tmp_path),
        closure_strictness="lenient",
    )
    assert "skip_pre_unit_closures" not in lenient_u1
    assert "skip_pre_unit_closures" not in lenient_u2

    # Precomputed scopes respect changed closure_strictness and skip_pre_unit_closures
    pre_clos_gen_src = (
        "def gen_pre_a(items):\n"
        "    step = 1\n"
        "    cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return cb\n"
        "\n"
        "def gen_pre_b(items):\n"
        "    step = 2\n"
        "    cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return cb\n"
    )
    f_pre_clos = tmp_path / "pre_clos_gen.py"
    f_pre_clos.write_text(pre_clos_gen_src, encoding="utf-8")
    u_pre_g1 = {
        "file": str(f_pre_clos),
        "start": 4,
        "end": 6,
        "name": "gen_pre_a:For",
        "kind": "compound_block",
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    u_pre_g2 = {
        "file": str(f_pre_clos),
        "start": 12,
        "end": 14,
        "name": "gen_pre_b:For",
        "kind": "compound_block",
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    strict_sc1 = inspect_single_unit_scope(u_pre_g1, repo_root=str(tmp_path))
    strict_sc2 = inspect_single_unit_scope(u_pre_g2, repo_root=str(tmp_path))
    strict_combined = analyze_unit_variable_scope(
        u_pre_g1, u_pre_g2, repo_root=str(tmp_path), scope1=strict_sc1, scope2=strict_sc2
    )
    assert strict_sc1["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_sc1["control_flow_hazards"]

    # Lenient synthesis and output resolution with strict precomputed scopes matches uncached lenient
    synth_lenient_no_pre = synthesize_shared_helper_code(
        u_pre_g1, u_pre_g2, repo_root=str(tmp_path), closure_strictness="lenient"
    )
    synth_lenient_with_pre = synthesize_shared_helper_code(
        u_pre_g1,
        u_pre_g2,
        repo_root=str(tmp_path),
        scope1=strict_sc1,
        scope2=strict_sc2,
        scope=strict_combined,
        closure_strictness="lenient",
    )
    assert synth_lenient_no_pre != ""
    assert synth_lenient_with_pre == synth_lenient_no_pre

    lenient_sc1 = inspect_single_unit_scope(
        {**u_pre_g1, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    lenient_sc2 = inspect_single_unit_scope(
        {**u_pre_g2, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    assert lenient_sc1["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in lenient_sc1["control_flow_hazards"]

    res_lenient_matching_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        lenient_sc1,
        lenient_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=True,
        return_details=True,
    )
    res_lenient_with_strict_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        strict_sc1,
        strict_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=True,
        return_details=True,
    )
    assert res_lenient_matching_pre.outputs is not None
    assert res_lenient_with_strict_pre == res_lenient_matching_pre

    # Strict resolution and synthesis with lenient precomputed scopes matches uncached strict
    strict_combined_from_lenient = analyze_unit_variable_scope(
        u_pre_g1,
        u_pre_g2,
        repo_root=str(tmp_path),
        scope1=lenient_sc1,
        scope2=lenient_sc2,
    )
    assert strict_combined_from_lenient["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_combined_from_lenient["control_flow_hazards"]
    res_strict_with_lenient_pre = resolve_clone_pair_outputs(
        u_pre_g1,
        u_pre_g2,
        lenient_sc1,
        lenient_sc2,
        repo_root=str(tmp_path),
        skip_pre_unit_closures=False,
        return_details=True,
    )
    assert res_strict_with_lenient_pre.outputs is None
    assert res_strict_with_lenient_pre.skip_reason == "severed closure cell"
    assert (
        synthesize_shared_helper_code(
            u_pre_g1,
            u_pre_g2,
            repo_root=str(tmp_path),
            scope1=lenient_sc1,
            scope2=lenient_sc2,
            closure_strictness="strict",
        )
        == ""
    )
    assert (
        generate_refactoring_patch(
            [
                (
                    1.0,
                    {**u_pre_g1, "skip_pre_unit_closures": True},
                    {**u_pre_g2, "skip_pre_unit_closures": True},
                )
            ],
            repo_root=str(tmp_path),
            replace_clones=True,
            closure_strictness="strict",
        )
        == ""
    )

    # ClassDef immediate scope with nonlocal declaration in collect_downstream_read_names
    cls_nl_src = (
        "def outer_cls_nl(items):\n"
        "    total = 0\n"
        "    class Box:\n"
        "        nonlocal total\n"
        "        for x in items:\n"
        "            total = x * 2\n"
        "    return Box, total\n"
    )
    assert collect_downstream_read_names(
        cls_nl_src,
        {"start": 5, "end": 6},
        candidates={"total", "x"},
    ) == {"total"}

    # 3. Late-bound closure cell end-to-end skip reason and helper synthesis block
    late_cell_src = (
        "def build_a(items):\n"
        "    callbacks = []\n"
        "    for item in items:\n"
        "        def cb(x, item=item):\n"
        "            return x * scale + item\n"
        "        callbacks.append(cb)\n"
        "    scale = 10\n"
        "    return [f(2) for f in callbacks]\n"
        "\n"
        "def build_b(items):\n"
        "    callbacks = []\n"
        "    for item in items:\n"
        "        def cb(x, item=item):\n"
        "            return x * scale + item\n"
        "        callbacks.append(cb)\n"
        "    scale = 20\n"
        "    return [f(3) for f in callbacks]\n"
    )
    f_late = tmp_path / "late_cell.py"
    f_late.write_text(late_cell_src, encoding="utf-8")
    u_late1 = {
        "file": str(f_late),
        "start": 3,
        "end": 6,
        "name": "build_a:For",
        "kind": "compound_block",
    }
    u_late2 = {
        "file": str(f_late),
        "start": 12,
        "end": 15,
        "name": "build_b:For",
        "kind": "compound_block",
    }
    sc_late1 = inspect_single_unit_scope(u_late1, repo_root=str(tmp_path))
    sc_late2 = inspect_single_unit_scope(u_late2, repo_root=str(tmp_path))
    assert "severed_closure_cell" in sc_late1["control_flow_hazards"]
    assert synthesize_shared_helper_code(u_late1, u_late2, repo_root=str(tmp_path)) == ""
    res_late = resolve_clone_pair_outputs(
        u_late1, u_late2, sc_late1, sc_late2, repo_root=str(tmp_path), return_details=True
    )
    assert res_late.outputs is None
    assert res_late.skip_reason == "severed closure cell"

    # 8. Enclosing with cleanup vs severed closure cell skip reasons
    global_with_src = (
        "def run_glob(ctx, items):\n"
        "    global shared_total\n"
        "    with ctx:\n"
        "        for x in items:\n"
        "            shared_total += x\n"
    )
    u_glob_with = {
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_text": global_with_src,
        "source_code": "for x in items:\n    shared_total += x\n",
    }
    sc_glob = inspect_single_unit_scope(u_glob_with, tree=ast.parse(global_with_src))
    assert "severed_closure_cell" in sc_glob["control_flow_hazards"]
    res_glob = resolve_clone_pair_outputs(
        u_glob_with,
        u_glob_with,
        sc_glob,
        sc_glob,
        source_text1=global_with_src,
        source_text2=global_with_src,
        return_details=True,
    )
    assert res_glob.outputs is None
    assert res_glob.skip_reason == "severed closure cell"

    nl_with_src = (
        "def outer_with(ctx, items):\n"
        "    shared_total = 0\n"
        "    def run_nl():\n"
        "        nonlocal shared_total\n"
        "        with ctx:\n"
        "            for x in items:\n"
        "                shared_total += x\n"
        "    return run_nl\n"
    )
    u_nl_with = {
        "start": 6,
        "end": 7,
        "kind": "compound_block",
        "source_text": nl_with_src,
        "source_code": "for x in items:\n    shared_total += x\n",
    }
    sc_nl_with = inspect_single_unit_scope(u_nl_with, tree=ast.parse(nl_with_src))
    assert "severed_closure_cell" not in sc_nl_with["control_flow_hazards"]
    res_nl_with = resolve_clone_pair_outputs(
        u_nl_with,
        u_nl_with,
        sc_nl_with,
        sc_nl_with,
        source_text1=nl_with_src,
        source_text2=nl_with_src,
        return_details=True,
    )
    assert res_nl_with.outputs is None
    assert res_nl_with.skip_reason == "enclosing try or with cleanup hazard"

    # Multi-level generator with severed closure cell prioritizes severed closure cell reason
    multilevel_cell_src = (
        "def outer_chain(items):\n"
        "    scale = 1\n"
        "    def inner_chain():\n"
        "        cbs = []\n"
        "        for item in items:\n"
        "            cbs.append(lambda x, item=item: x * scale + item)\n"
        "            yield item\n"
        "    scale = 20\n"
        "    return inner_chain\n"
    )
    u_ml = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": multilevel_cell_src,
        "source_code": (
            "for item in items:\n"
            "    cbs.append(lambda x, item=item: x * scale + item)\n"
            "    yield item\n"
        ),
    }
    sc_ml = inspect_single_unit_scope(u_ml, tree=ast.parse(multilevel_cell_src))
    assert "severed_closure_cell" in sc_ml["control_flow_hazards"]
    res_ml_gen = resolve_clone_pair_outputs(
        u_ml,
        u_ml,
        sc_ml,
        sc_ml,
        source_text1=multilevel_cell_src,
        source_text2=multilevel_cell_src,
        return_details=True,
    )
    assert res_ml_gen.outputs is None
    assert res_ml_gen.skip_reason == "severed closure cell"


def test_df_cell_hazards_numbered_checks_unit() -> None:
    """Directly tests each numbered check function (1&2, 3, 4) in _df_cell_hazards."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _analyze_enclosing_fn_cells,
        _check_captured_cell_mutations,
        _check_intra_unit_nonlocal_mutators,
        _check_sibling_closure_sharing,
        _collect_unit_nonlocal_mutators,
        _extract_closure_nonlocal_mutations,
        _extract_expr_scope_free_reads,
        _find_enclosing_function_chain,
    )

    # Check 1 & 2: _check_captured_cell_mutations
    src_c12 = (
        "def outer(items, param):\n"
        "    pre = 1\n"
        "    for x in items:\n"
        "        f = lambda: x + pre + late + param\n"
        "    late = 2\n"
        "    return f\n"
    )
    tree_c12 = ast.parse(src_c12)
    chain_c12 = _find_enclosing_function_chain(tree_c12, 4, 4)
    assert len(chain_c12) == 1
    info_c12 = _analyze_enclosing_fn_cells(
        chain_c12[0][2], (4, 4, None, None), unit_mutations={"f"}
    )
    # Capturing only pre and param is safe
    assert (
        _check_captured_cell_mutations(
            {"pre", "param"}, {"pre", "param", "items"}, [info_c12]
        )
        is False
    )
    # Capturing loop-stored 'x' triggers Check 1
    assert (
        _check_captured_cell_mutations({"x"}, {"items"}, [info_c12])
        is True
    )
    # Capturing 'late' (bound after span) triggers Check 2
    assert (
        _check_captured_cell_mutations({"late"}, {"late", "items"}, [info_c12])
        is True
    )

    # Check 3: _check_intra_unit_nonlocal_mutators
    src_c3 = (
        "def outer():\n"
        "    count = 0\n"
        "    def inc():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    inc()\n"
        "    return count\n"
    )
    tree_c3 = ast.parse(src_c3)
    chain_c3 = _find_enclosing_function_chain(tree_c3, 2, 5)
    info_c3 = _analyze_enclosing_fn_cells(chain_c3[0][2], (2, 5, None, None))
    unit_stmts_c3 = ast.parse("count = 0\ndef inc():\n    nonlocal count\n    count += 1\n").body
    mutators_c3 = _collect_unit_nonlocal_mutators(unit_stmts_c3, mod_tree=tree_c3)
    assert (
        _check_intra_unit_nonlocal_mutators(
            unit_stmts_c3,
            mutators_c3,
            {"count", "inc"},
            set(),
            [info_c3],
            is_subroutine=True,
        )
        is True
    )
    # When count is local to unit and not read after unit, Check 3 is False
    src_c3_safe = (
        "def outer():\n"
        "    count = 0\n"
        "    def inc():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    return 42\n"
    )
    tree_c3_safe = ast.parse(src_c3_safe)
    chain_c3_safe = _find_enclosing_function_chain(tree_c3_safe, 2, 5)
    info_c3_safe = _analyze_enclosing_fn_cells(chain_c3_safe[0][2], (2, 5, None, None))
    assert (
        _check_intra_unit_nonlocal_mutators(
            unit_stmts_c3,
            mutators_c3,
            {"count", "inc"},
            set(),
            [info_c3_safe],
            is_subroutine=True,
        )
        is False
    )

    # Check 4: _check_sibling_closure_sharing
    src_c4 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(get_acc())\n"
        "    return acc, out\n"
    )
    tree_c4 = ast.parse(src_c4)
    chain_c4 = _find_enclosing_function_chain(tree_c4, 6, 8)
    info_c4 = _analyze_enclosing_fn_cells(chain_c4[0][2], (6, 8, None, None))
    assert (
        _check_sibling_closure_sharing(
            {"items", "acc", "out", "get_acc"},
            {"acc", "x"},
            [info_c4],
        )
        is True
    )
    # When get_acc is not an input to the unit, Check 4 is False
    assert (
        _check_sibling_closure_sharing(
            {"items", "acc", "out"},
            {"acc", "x"},
            [info_c4],
        )
        is False
    )

    # _extract_closure_nonlocal_mutations scans nested def headers (e.g. walrus in default)
    fn_with_nested_hdr = ast.parse(
        "def make():\n"
        "    nonlocal acc\n"
        "    def inner(v=(acc := acc + 1)):\n"
        "        return v\n"
        "    return inner\n"
    ).body[0]
    assert isinstance(fn_with_nested_hdr, ast.FunctionDef)
    assert "acc" in _extract_closure_nonlocal_mutations(fn_with_nested_hdr)

    # _extract_expr_scope_free_reads excludes first comprehension iterable when include_header=False
    gen_expr = ast.parse("(x * scale for x in seq)").body[0].value  # type: ignore[attr-defined]
    assert _extract_expr_scope_free_reads(gen_expr, include_header=False) == {"scale"}
    assert _extract_expr_scope_free_reads(gen_expr, include_header=True) == {"scale", "seq"}

    # Ephemeral unit modules in _collect_unit_nonlocal_mutators do not pollute caches
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_parent_maps,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()
    _collect_unit_nonlocal_mutators(ast.parse("x = 1\n").body, mod_tree=None)
    assert not any(
        isinstance(k, tuple) and any(isinstance(el, ast.Module) for el in k)
        for k in _scope_cell_info_cache
    )
    assert not any(isinstance(k, ast.Module) for k in _scope_parent_maps)

    # _evict_tree_scopes removes tuple keys when an AST node sits at an interior index (e.g. index 3)
    subtree_fn = ast.parse("def sub_fn():\n    return 1\n").body[0]
    sentinel_key: tuple[object, ...] = ("prefix", 1, frozenset({"x"}), subtree_fn, "suffix")
    _scope_cell_info_cache[sentinel_key] = "cached_val"
    _evict_tree_scopes(subtree_fn)
    assert sentinel_key not in _scope_cell_info_cache

    # _find_enclosing_function_chain excludes inner functions at greater col_offset on u_start
    same_line_tree = ast.parse(
        "def outer(cond):\n"
        "    if cond:\n"
        "        def inner():\n"
        "            return 1\n"
    )
    chain_same_line = _find_enclosing_function_chain(
        same_line_tree, 3, 4, u_start_col=4
    )
    assert [getattr(fn, "name", None) for _, _, fn in chain_same_line] == ["outer"]

    # _find_enclosing_function_chain excludes decorated inner function when unit is inside decorator
    dec_tree = ast.parse(
        "def outer(items):\n"
        "    @dec([x * 2 for x in items])\n"
        "    def inner():\n"
        "        return 1\n"
    )
    chain_dec = _find_enclosing_function_chain(dec_tree, 2, 2)
    assert [getattr(fn, "name", None) for _, _, fn in chain_dec] == ["outer"]

    # _find_enclosing_function_chain excludes inner function when unit lies inside its multi-line header
    hdr_tree = ast.parse(
        "def outer(items):\n"
        "    scale = 1\n"
        "    def inner(\n"
        "        default_cbs=[lambda x, item=item: x * scale + item for item in items],\n"
        "    ):\n"
        "        scale = 2\n"
        "        return default_cbs\n"
    )
    chain_hdr = _find_enclosing_function_chain(hdr_tree, 4, 4)
    assert [getattr(fn, "name", None) for _, _, fn in chain_hdr] == ["outer"]

    # _node_earliest_start_pos and multiline decorator span_in_hdr tracking
    from pydoppelgangerhunt.fixer._df_cell_hazards import _node_earliest_start_pos

    mdec_tree = ast.parse(
        "def outer(items, reg):\n"
        "    @reg(\n"
        "        [x for x in items],\n"
        "    )\n"
        "    def inner():\n"
        "        return post_var\n"
    )
    outer_fn = mdec_tree.body[0]
    inner_fn = outer_fn.body[0]  # type: ignore[attr-defined]
    assert _node_earliest_start_pos(inner_fn) == (2, 4)
    info_mdec = _analyze_enclosing_fn_cells(outer_fn, (3, 3, 8, 26))
    assert "inner" not in info_mdec.pre_bound
    assert "inner" in info_mdec.post_mutated
    assert "post_var" in info_mdec.post_reads

    # Decorated def as first statement of enclosing function starts at @ column
    first_dec_tree = ast.parse(
        "def o():\n"
        "    @d\n"
        "    def i():\n"
        "        return 1\n"
        "    return i\n"
    )
    chain_first_dec = _find_enclosing_function_chain(
        first_dec_tree, 2, 4, u_start_col=4
    )
    assert [getattr(fn, "name", None) for _, _, fn in chain_first_dec] == ["o"]

    # PEP 695 type_params on inner function are included in info.fn_locals
    import sys  # pylint: disable=import-outside-toplevel

    if sys.version_info >= (3, 12):
        tp_src = (
            "def outer(items):\n"
            "    T = 1\n"
            "    def inner[T]():\n"
            "        for x in items:\n"
            "            f = lambda: T\n"
            "        return f\n"
            "    return inner, T\n"
        )
        tp_tree = ast.parse(tp_src)
        tp_chain = _find_enclosing_function_chain(tp_tree, 4, 5)
        tp_inner_info = _analyze_enclosing_fn_cells(
            tp_chain[-1][2], (4, 5, None, None)
        )
        assert "T" in tp_inner_info.fn_locals


def test_expr_traversal_subscript_compare_unary_fstring_and_ifexp() -> None:
    """Verifies _extract_expr_closure_and_load_info and hazard checks across nested expressions."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _extract_expr_closure_and_load_info,
    )

    expr_cases = [
        ("callbacks[index_cb()]", {"callbacks"}, {"index_cb"}),
        ("items[index]", {"items", "index"}, set()),
        ("left_cb < right_cb()", {"left_cb"}, {"right_cb"}),
        ("not unary_cb(arg_cb)", {"arg_cb"}, {"unary_cb"}),
        ("f'{fmt_cb(val_cb):{spec_cb}}'", {"val_cb", "spec_cb"}, {"fmt_cb"}),
        ("body_cb if cond_cb() else alt_cb", {"body_cb", "alt_cb"}, {"cond_cb"}),
    ]
    for expr_src, expected_loads, expected_calls in expr_cases:
        expr_node = ast.parse(expr_src, mode="eval").body
        called: set[str] = set()
        _, _, loaded = _extract_expr_closure_and_load_info(
            [expr_node], called_names_out=called
        )
        assert expected_loads <= loaded, expr_src
        assert expected_calls <= called, expr_src

    # Closure references and calls nested inside Subscript slice, Compare, UnaryOp, and JoinedStr
    nested_hazard_cases = [
        (
            "def outer(items, callbacks):\n"
            "    idx = 0\n"
            "    def get_idx():\n"
            "        return idx\n"
            "    def pick():\n"
            "        return callbacks[get_idx()]\n"
            "    out = []\n"
            "    for x in items:\n"
            "        idx += x\n"
            "        out.append(pick())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    idx += x\n    out.append(pick())\n",
        ),
        (
            "def outer(items):\n"
            "    limit = 0\n"
            "    def get_limit():\n"
            "        return limit\n"
            "    def check(v):\n"
            "        return v < get_limit()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        limit += x\n"
            "        out.append(check(x))\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    limit += x\n    out.append(check(x))\n",
        ),
        (
            "def outer(items):\n"
            "    flag = False\n"
            "    def get_flag():\n"
            "        return flag\n"
            "    def inverted():\n"
            "        return not get_flag()\n"
            "    out = []\n"
            "    for x in items:\n"
            "        flag = bool(x)\n"
            "        out.append(inverted())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    flag = bool(x)\n    out.append(inverted())\n",
        ),
        (
            "def outer(items):\n"
            "    tag = 'a'\n"
            "    def get_tag():\n"
            "        return tag\n"
            "    def render():\n"
            "        return f'[{get_tag()}]'\n"
            "    out = []\n"
            "    for x in items:\n"
            "        tag = str(x)\n"
            "        out.append(render())\n"
            "    return out\n",
            8,
            10,
            "for x in items:\n    tag = str(x)\n    out.append(render())\n",
        ),
    ]
    for src, s_line, e_line, u_code in nested_hazard_cases:
        unit = {
            "start": s_line,
            "end": e_line,
            "kind": "compound_block",
            "source_text": src,
            "source_code": u_code,
        }
        scope = inspect_single_unit_scope(unit, tree=ast.parse(src))
        assert "severed_closure_cell" in scope["control_flow_hazards"]


def _run_differential_refactoring_case(
    tmp_path: Path,
    case_name: str,
    fn_src: str,
    u1_bounds: tuple[int, int],
    expect_patch: bool,
    call_args_list: list[tuple[object, ...]],
) -> None:
    """Executes original and refactored modules and verifies identical return values and side effects."""
    import subprocess  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import generate_refactoring_patch

    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / ".git").mkdir(exist_ok=True)
    fn_b_src = fn_src.replace("def entry(", "def entry_b(", 1)
    line_offset = len(fn_src.splitlines()) + 1
    combined_src = f"{fn_src}\n{fn_b_src}"
    mod_file = case_dir / "mod.py"
    mod_file.write_text(combined_src, encoding="utf-8")

    u1 = {
        "file": str(mod_file),
        "start": u1_bounds[0],
        "end": u1_bounds[1],
        "name": "entry:block",
        "kind": "compound_block",
    }
    u2 = {
        "file": str(mod_file),
        "start": u1_bounds[0] + line_offset,
        "end": u1_bounds[1] + line_offset,
        "name": "entry_b:block",
        "kind": "compound_block",
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(case_dir), replace_clones=False
    )
    if expect_patch:
        assert patch != "", f"Expected refactoring patch for {case_name}"
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn"],
            input=patch,
            text=True,
            cwd=str(case_dir),
            check=True,
            capture_output=True,
        )
        refactored_src = mod_file.read_text(encoding="utf-8")
        assert "_shared_" in refactored_src
    else:
        assert patch == "", f"Expected hazard to block patch for {case_name}"
        refactored_src = combined_src

    for args in call_args_list:
        ns_orig: dict[str, Any] = {}
        ns_ref: dict[str, Any] = {}
        exec(compile(combined_src, "<orig>", "exec"), ns_orig)  # pylint: disable=exec-used
        exec(compile(refactored_src, "<ref>", "exec"), ns_ref)  # pylint: disable=exec-used
        res_orig = ns_orig["entry"](*args)
        res_ref = ns_ref["entry"](*args)
        res_ref_b = ns_ref["entry_b"](*args)
        assert res_orig == res_ref, f"Mismatch in {case_name} for args={args}"
        assert res_orig == res_ref_b, f"Mismatch in {case_name} (entry_b) for args={args}"


def test_differential_cell_mutations_vs_ordinary_bindings(tmp_path: Path) -> None:
    """Differential execution tests distinguishing cell mutations, object mutation, locals, branches, and deletion."""
    # 1a. Rebinding a variable captured by a nested function (post/loop rebinding blocked)
    rebind_hazard_a = (
        "def entry(items):\n"
        "    scale = 1\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda v: v + scale)\n"
        "        z = x * 2\n"
        "    scale = 10\n"
        "    return [cb(5) for cb in cbs], z\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "rebind_captured_blocked",
        rebind_hazard_a,
        (4, 6),
        expect_patch=False,
        call_args_list=[([1, 2, 3],)],
    )

    # 1b. Pre-bound unchanging variable captured by nested function (refactors and matches)
    rebind_safe_a = (
        "def entry(items):\n"
        "    scale = 10\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        step = x * scale\n"
        "        cbs.append(lambda v, s=step: v + scale + s)\n"
        "    return [cb(1) for cb in cbs]\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "prebound_captured_safe",
        rebind_safe_a,
        (4, 6),
        expect_patch=True,
        call_args_list=[([1, 2, 3],), ([],)],
    )

    # 2. Mutating an object referenced by a captured variable without rebinding the variable
    obj_mut_a = (
        "def entry(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        history.append(x * 2)\n"
        "        total += x\n"
        "    return total, snap(), history\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "object_mutation_without_rebinding",
        obj_mut_a,
        (5, 7),
        expect_patch=True,
        call_args_list=[([2, 4, 6],), ([],)],
    )

    # 3. Rebinding a name local to a nested function rather than to the enclosing function
    inner_local_a = (
        "def entry(items):\n"
        "    scale = 5\n"
        "    def helper_local(v):\n"
        "        scale = v * 10\n"
        "        return scale + 1\n"
        "    out = []\n"
        "    for x in items:\n"
        "        cb = lambda y, x=x: y + scale + x\n"
        "        out.append(cb(2))\n"
        "    return out, helper_local(3), scale\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "inner_function_local_rebinding",
        inner_local_a,
        (7, 9),
        expect_patch=True,
        call_args_list=[([1, 2],)],
    )

    # 4. Conditional assignment inside try/except that may raise before completing
    cond_try_a = (
        "def entry(should_fail):\n"
        "    def risky():\n"
        "        if should_fail:\n"
        "            raise ValueError('boom')\n"
        "        return 99\n"
        "    value = 1\n"
        "    try:\n"
        "        value = risky()\n"
        "    except Exception:\n"
        "        pass\n"
        "    getter = lambda: value\n"
        "    return getter(), value\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "conditional_try_preserve_prebound",
        cond_try_a,
        (7, 10),
        expect_patch=True,
        call_args_list=[(True,), (False,)],
    )

    # 5. Deleting a binding, including exception-handler targets cleared by Python
    exc_clear_a = (
        "def entry(items):\n"
        "    err = 'initial'\n"
        "    try:\n"
        "        raise RuntimeError('fail')\n"
        "    except RuntimeError as err:\n"
        "        pass\n"
        "    err = 'recovered'\n"
        "    out = []\n"
        "    for x in items:\n"
        "        cb = lambda v, x=x: f'{err}:{v + x}'\n"
        "        out.append(cb(1))\n"
        "    return out\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "except_handler_target_rebound",
        exc_clear_a,
        (9, 11),
        expect_patch=True,
        call_args_list=[([10, 20],)],
    )


def test_nested_scope_reads_do_not_falsely_escape_mutator() -> None:
    """Verifies unexecuted or synchronous intra-unit references to a nonlocal mutator do not count as escaping."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _extract_shallow_loads_and_calls,
    )

    # 1. Reviewer's exact scenario: unused nested function calling increment()
    unused_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def unused():\n"
        "            increment()\n"
        "    return len(items)\n"
    )
    u_unused = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": unused_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def unused():\n"
            "        increment()\n"
        ),
    }
    sc_unused = inspect_single_unit_scope(
        u_unused, tree=ast.parse(unused_caller_src)
    )
    assert "severed_closure_cell" not in sc_unused["control_flow_hazards"]

    # 2. Synchronous nested caller invoked inside unit, where neither count nor increment escapes
    sync_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def run_once():\n"
        "            increment()\n"
        "        run_once()\n"
        "    return len(items)\n"
    )
    u_sync = {
        "start": 2,
        "end": 9,
        "kind": "compound_block",
        "source_text": sync_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def run_once():\n"
            "        increment()\n"
            "    run_once()\n"
        ),
    }
    sc_sync = inspect_single_unit_scope(u_sync, tree=ast.parse(sync_caller_src))
    assert "severed_closure_cell" not in sc_sync["control_flow_hazards"]

    # 3. Escaping nested caller (run_once returned/used after unit while count is also read) -> hazard
    escaped_caller_src = (
        "def outer(items):\n"
        "    if items:\n"
        "        count = 0\n"
        "        def increment():\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "        def run_once():\n"
        "            increment()\n"
        "    return run_once, count\n"
    )
    u_escaped = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": escaped_caller_src,
        "source_code": (
            "if items:\n"
            "    count = 0\n"
            "    def increment():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "    def run_once():\n"
            "        increment()\n"
        ),
    }
    sc_escaped = inspect_single_unit_scope(
        u_escaped, tree=ast.parse(escaped_caller_src)
    )
    assert "severed_closure_cell" in sc_escaped["control_flow_hazards"]

    # 4. _extract_shallow_loads_and_calls stops at nested def body but inspects default/decorator headers
    stmts = ast.parse(
        "def inner(cb=header_ref):\n"
        "    body_ref()\n"
        "    return body_val\n"
    ).body
    non_call, _ = _extract_shallow_loads_and_calls(stmts)
    assert "header_ref" in non_call
    assert "body_ref" not in non_call
    assert "body_val" not in non_call


def test_partial_column_inference_all_four_combinations_and_utf8() -> None:
    """Verifies _unit_cols and _resolve_unit_col_bounds across all 4 (start_col, end_col) combinations and UTF-8."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer.scope import _unit_cols

    # 1. Multiple statements on a single ASCII line
    ascii_line = "    x = 1; y = x + 2; z = y * 3\n"
    target_snippet = "y = x + 2"
    exp_s = len("    x = 1; ".encode("utf-8"))
    exp_e = len("    x = 1; y = x + 2".encode("utf-8"))

    for s_in, e_in in [
        (None, None),
        (None, exp_e),
        (exp_s, None),
        (exp_s, exp_e),
    ]:
        unit_dict: dict[str, Any] = {"source_code": target_snippet}
        if s_in is not None:
            unit_dict["start_col"] = s_in
        if e_in is not None:
            unit_dict["end_col"] = e_in
        got_s, got_e = _unit_cols(unit_dict, [ascii_line])
        assert (got_s, got_e) == (exp_s, exp_e), (s_in, e_in)

    # 2. _resolve_unit_col_bounds with identical AST statements on the same line
    dup_src = "def f(a):\n    a = a + 1; a = a + 1\n    return a\n"
    dup_tree = ast.parse(dup_src)
    cand_stmt = ast.parse("a = a + 1\n").body
    # First statement is cols 4..13, second statement is cols 15..24; ambiguous when both cols are None
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, None, None) == (
        None,
        None,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 4, None) == (
        4,
        13,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 15, None) == (
        15,
        24,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, None, 24) == (
        15,
        24,
    )
    assert _resolve_unit_col_bounds(cand_stmt, dup_tree, 2, 2, 4, 13) == (
        4,
        13,
    )

    # 3. Non-ASCII UTF-8 multi-statement line across all 4 coordinate combinations
    utf8_src = (
        "def calc():\n"
        "    café = '☕'; résumé = café + '!'; return lambda: résumé\n"
    )
    utf8_line = "    café = '☕'; résumé = café + '!'; return lambda: résumé\n"
    utf8_snippet = "résumé = café + '!'"
    pos = utf8_line.index(utf8_snippet)
    u8_s = len(utf8_line[:pos].encode("utf-8"))
    u8_e = len(utf8_line[: pos + len(utf8_snippet)].encode("utf-8"))

    for s_in, e_in in [
        (None, None),
        (None, u8_e),
        (u8_s, None),
        (u8_s, u8_e),
    ]:
        u_utf8: dict[str, Any] = {
            "start": 2,
            "end": 2,
            "kind": "sliding_window",
            "source_text": utf8_src,
            "source_code": utf8_snippet,
        }
        if s_in is not None:
            u_utf8["start_col"] = s_in
        if e_in is not None:
            u_utf8["end_col"] = e_in
        sc = inspect_single_unit_scope(u_utf8, tree=ast.parse(utf8_src))
        assert sc["inputs"] == ["café"]
        assert sc["outputs"] == ["résumé"]


@pytest.mark.parametrize(
    "case_id,stmts_src,initial_bound,expected_bound,expected_term",
    [
        (
            "raise_in_try_preserves_initial_bound",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except Exception:\n"
                "    pass\n"
            ),
            frozenset({"value"}),
            {"value"},
            False,
        ),
        (
            "raise_in_try_without_initial_bound_not_definite",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except Exception:\n"
                "    pass\n"
            ),
            frozenset(),
            set(),
            False,
        ),
        (
            "try_and_all_handlers_bind_same_var",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except ValueError:\n"
                "    value = 10\n"
                "except Exception:\n"
                "    value = 20\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_else_and_all_handlers_bind_var",
            (
                "try:\n"
                "    step = risky_operation()\n"
                "except Exception:\n"
                "    value = -1\n"
                "else:\n"
                "    value = 1\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_handler_raises_so_try_else_binding_survives_fallthrough",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except Exception:\n"
                "    raise\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_finally_overrides_and_binds_unconditionally",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except Exception:\n"
                "    pass\n"
                "finally:\n"
                "    final_val = 42\n"
            ),
            frozenset(),
            {"final_val"},
            False,
        ),
        (
            "if_return_in_one_branch_binds_via_fallthrough_branch",
            (
                "if cond:\n"
                "    return 0\n"
                "else:\n"
                "    value = 7\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "if_raise_in_both_branches_terminates_block",
            (
                "if cond:\n"
                "    raise ValueError('a')\n"
                "else:\n"
                "    return 1\n"
            ),
            frozenset(),
            set(),
            True,
        ),
        (
            "del_in_one_non_terminating_branch_removes_initial_binding",
            (
                "if cond:\n"
                "    del value\n"
            ),
            frozenset({"value"}),
            set(),
            False,
        ),
        (
            "del_followed_by_rebind_in_same_branch_preserves_binding",
            (
                "if cond:\n"
                "    del value\n"
                "    value = 2\n"
            ),
            frozenset({"value"}),
            {"value"},
            False,
        ),
        (
            "except_handler_target_implicitly_deleted_on_exit",
            (
                "try:\n"
                "    risky_operation()\n"
                "except Exception as value:\n"
                "    pass\n"
            ),
            frozenset({"value"}),
            set(),
            False,
        ),
    ],
)
def test_definite_bindings_exceptional_control_flow_table(
    case_id: str,
    stmts_src: str,
    initial_bound: frozenset[str],
    expected_bound: set[str],
    expected_term: bool,
) -> None:
    """Table-driven verification of _extract_block_definite_bindings_and_term across exceptional paths."""
    import textwrap  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _analyze_enclosing_fn_cells,
        _extract_block_definite_bindings_and_term,
    )

    _ = case_id
    mod = ast.parse(f"def _wrap():\n{textwrap.indent(stmts_src, '    ')}\n")
    fn_node = mod.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    bound, term = _extract_block_definite_bindings_and_term(
        fn_node.body,
        active_nonlocals=None,
        initial_bound=initial_bound,
        fn_globals=frozenset(),
    )
    assert bound == expected_bound
    assert term is expected_term

    # Also verify pre-unit try/except exhaustive bindings in _analyze_enclosing_fn_cells
    if not initial_bound and not expected_term:
        outer_src = (
            "def outer(cond, risky_operation):\n"
            f"{textwrap.indent(stmts_src, '    ')}"
            "    return lambda: 1\n"
        )
        outer_mod = ast.parse(outer_src)
        outer_fn = outer_mod.body[0]
        last_line = len(outer_src.splitlines())
        info = _analyze_enclosing_fn_cells(
            outer_fn, (last_line, last_line, None, None)
        )
        for v_name in expected_bound:
            assert v_name in info.pre_bound


def test_2d_coordinate_enclosure_and_decorator_start_pos() -> None:
    """Verifies 2D coordinate enclosure and decorator (line, col) coupling."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _SpanBounds,
        _node_earliest_start_pos,
        _strictly_encloses_span,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _resolve_immediate_scope_in_fn,
    )
    from pydoppelgangerhunt.fixer.scope import _find_enclosing_pre_bound_names

    # 1. Disjoint node to the left on the same line does not enclose span
    if_left = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=2,
        end_lineno=2,
        col_offset=0,
        end_col_offset=15,
    )
    assert not _strictly_encloses_span(if_left, _SpanBounds(2, 2, 20, 35))
    assert not _strictly_encloses_span(if_left, _SpanBounds(2, 2, 20, None))

    # 2. Disjoint node to the right on the same line does not enclose span
    if_right = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=2,
        end_lineno=2,
        col_offset=20,
        end_col_offset=35,
    )
    assert not _strictly_encloses_span(if_right, _SpanBounds(2, 2, 0, 15))
    assert not _strictly_encloses_span(if_right, _SpanBounds(2, 2, None, 15))

    # 3. Multi-line node ending on line 2 at col 10 does not enclose unit at [15, 25] on line 2
    multiline_prior = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=1,
        end_lineno=2,
        col_offset=0,
        end_col_offset=10,
    )
    assert not _strictly_encloses_span(
        multiline_prior, _SpanBounds(2, 2, 15, 25)
    )
    assert not _strictly_encloses_span(
        multiline_prior, _SpanBounds(2, 2, 15, None)
    )

    # 4. True single-line compound enclosure: `if cond: target_unit()`
    single_if_mod = ast.parse("if cond: target_unit()\n")
    single_if = single_if_mod.body[0]
    assert _strictly_encloses_span(single_if, _SpanBounds(1, 1, 9, 22))
    assert not _strictly_encloses_span(single_if, _SpanBounds(1, 1, 0, 22))

    # 5. _resolve_immediate_scope_in_fn ignores disjoint class ending before or starting after unit on same line
    cls_tree = ast.parse(
        "def outer():\n"
        "    class Box(\n"
        "        Base\n"
        "    ): pass\n"
    )
    outer_fn = cls_tree.body[0]
    box_cls = outer_fn.body[0]  # type: ignore[attr-defined]
    box_cls.end_col_offset = 11
    assert (
        _resolve_immediate_scope_in_fn(
            outer_fn, 4, 4, u_start_col=13, u_end_col=18
        )
        is outer_fn
    )
    box_cls.lineno = 4
    box_cls.col_offset = 20
    box_cls.end_col_offset = 35
    assert (
        _resolve_immediate_scope_in_fn(
            outer_fn, 4, 4, u_start_col=4, u_end_col=15
        )
        is outer_fn
    )

    # 6. _find_enclosing_pre_bound_names sets in_loop=False when loop is disjoint on boundary line
    loop_tree = ast.parse(
        "def outer(items):\n"
        "    for a in (\n"
        "        items\n"
        "    ): pass\n"
    )
    for_stmt = loop_tree.body[0].body[0]  # type: ignore[attr-defined]
    for_stmt.end_col_offset = 11
    _, in_loop = _find_enclosing_pre_bound_names(
        loop_tree, 4, 4, u_start_col=13, u_end_col=18
    )
    assert in_loop is False
    for_stmt.lineno = 4
    for_stmt.col_offset = 20
    for_stmt.end_col_offset = 35
    _, in_loop_right = _find_enclosing_pre_bound_names(
        loop_tree, 4, 4, u_start_col=4, u_end_col=15
    )
    assert in_loop_right is False

    # 7. Decorator line/col tracking uses decorator's '@' column and handles loose forms
    dec_fn = ast.FunctionDef(
        name="inner",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass(lineno=4, end_lineno=4, col_offset=8, end_col_offset=12)],
        decorator_list=[
            ast.Name(
                id="dec",
                ctx=ast.Load(),
                lineno=2,
                end_lineno=2,
                col_offset=3,
                end_col_offset=6,
            )
        ],
        lineno=3,
        end_lineno=4,
        col_offset=4,
        end_col_offset=12,
    )
    assert _node_earliest_start_pos(dec_fn) == (2, 2)
    wrap_mod = ast.Module(body=[dec_fn], type_ignores=[])
    closures = _collect_scope_closures(wrap_mod)
    assert closures[0][0] == (2, 2)


def test_duplicate_cols_decorator_snapping_and_cell_info_immutability() -> None:
    """Covers duplicate _unit_cols disambiguation, loose decorators, nonlocal shadowing, and immutable cache."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _analyze_enclosing_fn_cells,
        _extract_fn_outer_carrier_targets,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
    )
    from pydoppelgangerhunt.fixer._df_pairing import (
        resolve_clone_generator_subroutine_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        ensure_unit_scope_closure_mode,
        inspect_single_unit_scope,
        scope_matches_closure_mode,
    )

    # 1. _unit_cols returns (None, None) when duplicate snippets on a line are ambiguous
    dup_line = "    a = f(); a = f()\n"
    assert _unit_cols({"source_code": "a = f()"}, [dup_line]) == (None, None)
    assert _unit_cols(
        {"source_code": "a = f()", "start_col": 4}, [dup_line]
    ) == (4, 11)
    assert _unit_cols(
        {"source_code": "a = f()", "start_col": 13}, [dup_line]
    ) == (13, 20)
    assert _unit_cols(
        {"source_code": "a = f()", "end_col": 20}, [dup_line]
    ) == (13, 20)

    # 2. Loose decorator forms (@dec, @ dec, @(dec)) at top-level and indented
    for dec_text in ("@dec", "@ dec", "@(dec)"):
        top_mod = ast.parse(f"{dec_text}\ndef fn():\n    return x\n")
        assert _node_earliest_start_pos(top_mod.body[0]) == (1, 0)
        assert _collect_scope_closures(top_mod)[0][0] == (1, 0)

        ind_mod = ast.parse(
            f"def outer(x):\n    {dec_text}\n    def fn():\n        return x\n"
        )
        inner_fn = ind_mod.body[0].body[0]  # type: ignore[attr-defined]
        assert _node_earliest_start_pos(inner_fn) == (2, 4)

    # 3. _extract_fn_outer_carrier_targets does not report shadowed nonlocal in lambda or comprehension
    shadow_mod = ast.parse(
        "def outer():\n"
        "    x = []\n"
        "    y = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        f = lambda x: x.append(lambda: y)\n"
        "        g = [x.append(lambda: y) for x in ([],)]\n"
        "        return f, g\n"
    )
    mid_fn = shadow_mod.body[0].body[2]  # type: ignore[attr-defined]
    assert _extract_fn_outer_carrier_targets(mid_fn) == set()

    unshadowed_mod = ast.parse(
        "def outer():\n"
        "    x = []\n"
        "    y = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        f = lambda z: x.append(lambda: y)\n"
        "        return f\n"
    )
    mid_unshadowed = unshadowed_mod.body[0].body[2]  # type: ignore[attr-defined]
    assert _extract_fn_outer_carrier_targets(mid_unshadowed) == {"x"}

    # 4. _analyze_enclosing_fn_cells returns immutable containers and reuses cached statement deltas
    multi_unit_mod = ast.parse(
        "def big_fn(a):\n"
        "    b = a + 1\n"
        "    def helper():\n"
        "        return b\n"
        "    u1 = b + 2\n"
        "    u2 = u1 + 3\n"
        "    return helper(), u2\n"
    )
    big_fn = multi_unit_mod.body[0]
    info1 = _analyze_enclosing_fn_cells(
        big_fn, (5, 5, 4, 14), mod_tree=multi_unit_mod
    )
    info2 = _analyze_enclosing_fn_cells(
        big_fn, (6, 6, 4, 15), mod_tree=multi_unit_mod
    )
    assert isinstance(info1.fn_locals, frozenset)
    assert isinstance(info1.pre_bound, frozenset)
    assert isinstance(info1.sibling_closures, tuple)
    assert isinstance(info1.alias_edges, tuple)
    assert "b" in info1.pre_bound
    assert "u1" in info2.pre_bound

    # 5. Centralized closure mode helpers and generator subroutine strictness normalization
    gen_src = (
        "def gen(items):\n"
        "    cb = lambda: acc\n"
        "    acc = 0\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        yield acc\n"
        "    return acc, cb\n"
    )
    gen_tree = ast.parse(gen_src)
    u_gen = {
        "start": 3,
        "end": 6,
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "acc = 0\nfor x in items:\n    acc += x\n    yield acc\n",
    }
    strict_sc = inspect_single_unit_scope(u_gen, tree=gen_tree)
    assert scope_matches_closure_mode(strict_sc, False) is True
    assert scope_matches_closure_mode(strict_sc, True) is False
    lenient_sc = ensure_unit_scope_closure_mode(
        u_gen, strict_sc, True, source_text=gen_src, tree=gen_tree
    )
    assert scope_matches_closure_mode(lenient_sc, True) is True
    assert (
        resolve_clone_generator_subroutine_outputs(
            u_gen,
            dict(u_gen),
            strict_sc,
            dict(strict_sc),
            source_text1=gen_src,
            source_text2=gen_src,
            tree1=gen_tree,
            tree2=gen_tree,
            closure_strictness="lenient",
        )
        == (["acc"], ["acc"])
    )


def test_closure_factories_return_refs_and_class_body_units() -> None:
    """Verifies closure factories, return refs, class-body units, shadowing, col inference, and post-unit lenient expression hazards."""
    import ast
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        inspect_single_unit_scope,
    )

    # 1. Local non-escaping lambda inside helper does not taint plain return value as closure factory
    src_item1 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def compute_initial(seq):\n"
        "        s = sorted(seq, key=lambda v: v)\n"
        "        return sum(s) + acc\n"
        "    init_val = compute_initial(items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(init_val + x)\n"
        "    return acc, out\n"
    )
    tree_item1 = ast.parse(src_item1)
    unit_item1 = {
        "start": 8,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_item1,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(init_val + x)\n"
        ),
    }
    scope_item1 = inspect_single_unit_scope(unit_item1, tree=tree_item1)
    assert "severed_closure_cell" not in scope_item1["control_flow_hazards"]
    assert scope_item1["is_control_flow_safe"] is True

    # 2. Helper calling a sibling closure in return (ret_calls) is not conflated with returning callable (ret_loads)
    src_item2 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    def compute_initial():\n"
        "        return get_acc() + 1\n"
        "    init_val = compute_initial()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(init_val + x)\n"
        "    return acc, out\n"
    )
    tree_item2 = ast.parse(src_item2)
    unit_item2 = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_item2,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(init_val + x)\n"
        ),
    }
    scope_item2 = inspect_single_unit_scope(unit_item2, tree=tree_item2)
    assert "severed_closure_cell" not in scope_item2["control_flow_hazards"]
    assert scope_item2["is_control_flow_safe"] is True

    # Verify helper actually returning get_acc (ret_loads) or transitive call to factory STILL triggers hazard
    src_item2_pos = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    def get_reader():\n"
        "        return get_acc\n"
        "    def get_reader_2():\n"
        "        return get_reader()\n"
        "    reader = get_reader_2()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(reader() + x)\n"
        "    return acc, out\n"
    )
    tree_item2_pos = ast.parse(src_item2_pos)
    unit_item2_pos = {
        "start": 11,
        "end": 13,
        "kind": "compound_block",
        "source_text": src_item2_pos,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(reader() + x)\n"
        ),
    }
    scope_item2_pos = inspect_single_unit_scope(
        unit_item2_pos, tree=tree_item2_pos
    )
    assert "severed_closure_cell" in scope_item2_pos["control_flow_hazards"]

    # 3A. Class-body unit capturing enclosing class name Box before class body finishes
    src_item3a = (
        "def outer(items):\n"
        "    cbs = []\n"
        "    class Box:\n"
        "        for x in items:\n"
        "            cbs.append(lambda x=x: (Box, x))\n"
        "    return Box, cbs\n"
    )
    tree_item3a = ast.parse(src_item3a)
    unit_item3a = {
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_text": src_item3a,
        "source_code": (
            "for x in items:\n"
            "    cbs.append(lambda x=x: (Box, x))\n"
        ),
    }
    scope_item3a = inspect_single_unit_scope(unit_item3a, tree=tree_item3a)
    assert "severed_closure_cell" in scope_item3a["control_flow_hazards"]

    # 3B. Class-body unit conditionally assigning v without nonlocal does not promote outer local v
    src_item3b = (
        "def outer(flag):\n"
        "    v = 10\n"
        "    class Box:\n"
        "        if flag:\n"
        "            v = 20\n"
        "        w = 30\n"
        "    return Box, v\n"
    )
    tree_item3b = ast.parse(src_item3b)
    unit_item3b = {
        "start": 4,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_item3b,
        "source_code": "if flag:\n    v = 20\nw = 30\n",
    }
    scope_item3b = inspect_single_unit_scope(unit_item3b, tree=tree_item3b)
    assert "v" not in scope_item3b["inputs"]
    assert "v" in scope_item3b["conditional_outputs"]

    # 4. Lexical shadowing of outer local in inner function does not leak into pre_bound / cond_live_ins
    src_item4 = (
        "def outer(flag):\n"
        "    v = 10\n"
        "    def inner():\n"
        "        if flag:\n"
        "            v = 20\n"
        "        res = v if flag else 0\n"
        "        return res, v\n"
        "    return inner()\n"
    )
    tree_item4 = ast.parse(src_item4)
    unit_item4 = {
        "start": 4,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_item4,
        "source_code": "if flag:\n    v = 20\nres = v if flag else 0\n",
    }
    scope_item4 = inspect_single_unit_scope(unit_item4, tree=tree_item4)
    assert "v" not in scope_item4["inputs"]

    # 5. _unit_cols / _infer_missing_col_from_line_ast disambiguates comprehension vs parent Assign/BinOp
    comp_assign_line = ["    res = [x * 2 for x in items]"]
    assert _unit_cols(
        {"kind": "comprehension", "end_col": 32}, comp_assign_line
    ) == (10, 32)
    comp_binop_line = ["    [x * 2 for x in items] + [1]"]
    assert _unit_cols(
        {"kind": "comprehension", "start_col": 4}, comp_binop_line
    ) == (4, 26)

    # 6. Post-unit sibling closure capturing walrus-mutated var in expression unit caught in lenient mode
    src_item6 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    vals = [(acc := acc + x) for x in items]\n"
        "    cb = lambda: acc\n"
        "    return vals, cb\n"
    )
    tree_item6 = ast.parse(src_item6)
    unit_item6 = {
        "start": 3,
        "end": 3,
        "start_col": 11,
        "end_col": 44,
        "kind": "comprehension",
        "skip_pre_unit_closures": True,
        "source_text": src_item6,
        "source_code": "[(acc := acc + x) for x in items]",
    }
    scope_item6 = inspect_single_unit_scope(unit_item6, tree=tree_item6)
    assert "severed_closure_cell" in scope_item6["control_flow_hazards"]


def test_unseeded_conditional_stores_and_nested_mutator_shadowing() -> None:
    """Verifies unseeded cond store in unit closure, nested fn shadowing in mutators, outer carrier value filtering, nested comp target shadowing, lambda default args, and class method nonlocal del."""
    import ast
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    # 1. Unseeded conditional store captured by an intra-unit closure with pre-unit conditional binding
    src_r1 = (
        "def outer(items, init_flag):\n"
        "    if init_flag:\n"
        "        y = 10\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            y = x * 2\n"
        "        cbs.append(lambda: y)\n"
        "    return cbs\n"
    )
    tree_r1 = ast.parse(src_r1)
    unit_r1 = {
        "start": 5,
        "end": 8,
        "kind": "compound_block",
        "source_text": src_r1,
        "source_code": (
            "for x in items:\n"
            "    if x > 0:\n"
            "        y = x * 2\n"
            "    cbs.append(lambda: y)\n"
        ),
    }
    scope_r1 = inspect_single_unit_scope(unit_r1, tree=tree_r1)
    assert "severed_closure_cell" in scope_r1["control_flow_hazards"]

    # 2. Intra-unit nested function locals shadowed in _collect_unit_nonlocal_mutators
    src_r2 = (
        "def outer(items):\n"
        "    count = 100\n"
        "    if items:\n"
        "        def middle():\n"
        "            count = 0\n"
        "            def inner():\n"
        "                nonlocal count\n"
        "                count += 1\n"
        "            inner()\n"
        "            gen = ((count := count + x) for x in items)\n"
        "            return count + sum(gen)\n"
        "        res = middle()\n"
        "    else:\n"
        "        res = 0\n"
        "    return count + res\n"
    )
    tree_r2 = ast.parse(src_r2)
    unit_r2 = {
        "start": 3,
        "end": 14,
        "kind": "compound_block",
        "source_text": src_r2,
        "source_code": (
            "if items:\n"
            "    def middle():\n"
            "        count = 0\n"
            "        def inner():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        inner()\n"
            "        gen = ((count := count + x) for x in items)\n"
            "        return count + sum(gen)\n"
            "    res = middle()\n"
            "else:\n"
            "    res = 0\n"
        ),
    }
    scope_r2 = inspect_single_unit_scope(unit_r2, tree=tree_r2)
    assert "severed_closure_cell" not in scope_r2["control_flow_hazards"]
    assert scope_r2["is_control_flow_safe"] is True

    # 3. _extract_fn_outer_carrier_targets_uncached ignores non-closure outer attribute calls
    src_r3 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    saved = []\n"
        "    def helper(seq):\n"
        "        s = sorted(seq, key=lambda v: v)\n"
        "        saved.append(len(s) + acc)\n"
        "    helper(items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(len(saved) + x)\n"
        "    return acc, out\n"
    )
    tree_r3 = ast.parse(src_r3)
    unit_r3 = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_r3,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(len(saved) + x)\n"
        ),
    }
    scope_r3 = inspect_single_unit_scope(unit_r3, tree=tree_r3)
    assert "severed_closure_cell" not in scope_r3["control_flow_hazards"]
    assert scope_r3["is_control_flow_safe"] is True

    # 4. Comprehension target shadowing accumulated across nested comprehensions
    src_r4 = (
        "def outer(items):\n"
        "    acc = [1, 2]\n"
        "    cbs = [[lambda: acc for acc in acc] for _ in range(2)]\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc = [x]\n"
        "        out.append(len(cbs) + x)\n"
        "    return acc, out\n"
    )
    tree_r4 = ast.parse(src_r4)
    unit_r4 = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": src_r4,
        "source_code": (
            "for x in items:\n"
            "    acc = [x]\n"
            "    out.append(len(cbs) + x)\n"
        ),
    }
    scope_r4 = inspect_single_unit_scope(unit_r4, tree=tree_r4)
    assert "severed_closure_cell" not in scope_r4["control_flow_hazards"]
    assert scope_r4["is_control_flow_safe"] is True

    # 5. Lambda default arguments inspected in _extract_expr_closure_and_load_info
    src_r5 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    reader = lambda cb=get_acc: cb()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(reader() + x)\n"
        "    return acc, out\n"
    )
    tree_r5 = ast.parse(src_r5)
    unit_r5 = {
        "start": 7,
        "end": 9,
        "kind": "compound_block",
        "source_text": src_r5,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(reader() + x)\n"
        ),
    }
    scope_r5 = inspect_single_unit_scope(unit_r5, tree=tree_r5)
    assert "severed_closure_cell" in scope_r5["control_flow_hazards"]

    # 6. Sibling method nonlocal deletion inside an enclosing ClassDef marks variable pre_unbound
    src_r6 = (
        "def outer(items):\n"
        "    total = 10\n"
        "    cbs = []\n"
        "    class Box:\n"
        "        def clear(self):\n"
        "            nonlocal total\n"
        "            del total\n"
        "        def build(self):\n"
        "            for x in items:\n"
        "                cbs.append(lambda x=x: total + x)\n"
        "    return Box, cbs\n"
    )
    tree_r6 = ast.parse(src_r6)
    unit_r6 = {
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_r6,
        "source_code": (
            "for x in items:\n"
            "    cbs.append(lambda x=x: total + x)\n"
        ),
    }
    scope_r6 = inspect_single_unit_scope(unit_r6, tree=tree_r6)
    assert "severed_closure_cell" in scope_r6["control_flow_hazards"]


def test_exported_conditional_stores_scope_headers_and_exhaustive_match() -> None:
    """Verifies unseeded cond store exported and captured, header exclusion in _find_enclosing_pre_bound_names, exhaustive match definite bindings, and conditional class methods."""
    import ast
    import sys
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        inspect_single_unit_scope,
    )

    # Finding 1: Unseeded conditional store with no pre-unit binding captured by intra-unit closure and exported
    src_f1 = (
        "def outer(x):\n"
        "    cbs = []\n"
        "    if x > 0:\n"
        "        y = x * 2\n"
        "    cbs.append(lambda: y)\n"
        "    return cbs, (y if x > 0 else 0)\n"
    )
    tree_f1 = ast.parse(src_f1)
    unit_f1 = {
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "source_text": src_f1,
        "source_code": "if x > 0:\n    y = x * 2\ncbs.append(lambda: y)\n",
    }
    scope_f1 = inspect_single_unit_scope(unit_f1, tree=tree_f1)
    assert "severed_closure_cell" in scope_f1["control_flow_hazards"]

    # Finding 2: _find_enclosing_pre_bound_names excludes inner scope when unit span is in inner scope's header
    src_f2 = (
        "def outer():\n"
        "    x = 1\n"
        "    def inner(\n"
        "        a=x,\n"
        "    ):\n"
        "        x = 2\n"
        "        return a + x\n"
        "    return inner\n"
    )
    tree_f2 = ast.parse(src_f2)
    pre_bound_hdr, _ = _find_enclosing_pre_bound_names(
        tree_f2, 4, 4, u_start_col=8, u_end_col=11
    )
    assert "x" in pre_bound_hdr
    assert "inner" not in pre_bound_hdr
    assert "a" not in pre_bound_hdr

    # Finding 3A: Exhaustive match/case before unit marks variable definitely pre_bound
    if sys.version_info >= (3, 10):
        src_f3a = (
            "def outer(cmd, items):\n"
            "    match cmd:\n"
            "        case 'add':\n"
            "            scale = 2\n"
            "        case _:\n"
            "            scale = 1\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: x * scale)\n"
            "    return cbs\n"
        )
        tree_f3a = ast.parse(src_f3a)
        unit_f3a = {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f3a,
            "source_code": (
                "for x in items:\n"
                "    cbs.append(lambda x=x: x * scale)\n"
            ),
        }
        scope_f3a = inspect_single_unit_scope(unit_f3a, tree=tree_f3a)
        assert "severed_closure_cell" not in scope_f3a["control_flow_hazards"]
        assert scope_f3a["is_control_flow_safe"] is True

    # Finding 3B: Conditionally defined class method aliases self to sibling method closures
    src_f3b = (
        "def outer(items, fast):\n"
        "    acc = 0\n"
        "    class Worker:\n"
        "        if fast:\n"
        "            def get_acc(self):\n"
        "                return acc\n"
        "            def run(self):\n"
        "                out = []\n"
        "                for x in items:\n"
        "                    acc_local = self.get_acc() + x\n"
        "                    out.append(acc_local)\n"
        "                return out\n"
        "    acc += 1\n"
        "    return Worker, acc\n"
    )
    tree_f3b = ast.parse(src_f3b)
    unit_f3b = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_f3b,
        "source_code": (
            "for x in items:\n"
            "    acc_local = self.get_acc() + x\n"
            "    out.append(acc_local)\n"
        ),
    }
    scope_f3b = inspect_single_unit_scope(unit_f3b, tree=tree_f3b)
    assert "self" in scope_f3b["inputs"]

    src_f3b_mut = (
        "def outer(items, fast):\n"
        "    acc = 0\n"
        "    class Worker:\n"
        "        if fast:\n"
        "            def get_acc(self):\n"
        "                return acc\n"
        "            def run(self):\n"
        "                nonlocal acc\n"
        "                out = []\n"
        "                for x in items:\n"
        "                    acc += x\n"
        "                    out.append(self.get_acc())\n"
        "                return out\n"
        "    return Worker\n"
    )
    tree_f3b_mut = ast.parse(src_f3b_mut)
    unit_f3b_mut = {
        "start": 10,
        "end": 12,
        "kind": "compound_block",
        "source_text": src_f3b_mut,
        "source_code": (
            "for x in items:\n"
            "    acc += x\n"
            "    out.append(self.get_acc())\n"
        ),
    }
    scope_f3b_mut = inspect_single_unit_scope(
        unit_f3b_mut, tree=tree_f3b_mut
    )
    assert "severed_closure_cell" in scope_f3b_mut["control_flow_hazards"]


def test_utf8_column_slicing_finally_exits_and_except_handler_bindings(
    tmp_path: Path,
) -> None:
    """Covers UTF-8 prefix/multiline/semicolon slicing, finally conditional exits, exception-handler bindings, and 8-area matrix."""
    from pydoppelgangerhunt.fixer.dataflow import (
        clear_dataflow_cache,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.patch import refactor_module_units
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (
        _extract_unit_body_lines,
        _slice_unit_token_lines,
    )

    # --- Finding 2: UTF-8 coordinates before unit, semicolon-separated lines, multiline expressions, and pre-sliced lines ---
    # 2A. Semicolon-separated single-line with multi-byte UTF-8 before the unit
    utf8_semi_src = (
        "def compute(items, a, b):\n"
        '    tag = "café_αβγ_🚀"; target = [x * 2 for x in items]; res = a + b; tail = 0\n'
        "    return target, res\n"
    )
    utf8_semi_line = utf8_semi_src.splitlines(keepends=True)[1]
    comp_snip = "[x * 2 for x in items]"
    s_b, e_b = _unit_cols(
        {"kind": "comprehension", "source_code": comp_snip}, [utf8_semi_line]
    )
    assert s_b is not None and e_b is not None
    # Verify partial column inference (start_col only & end_col only) agrees with full inference
    assert _unit_cols(
        {"kind": "comprehension", "start_col": s_b}, [utf8_semi_line]
    ) == (s_b, e_b)
    assert _unit_cols(
        {"kind": "comprehension", "end_col": e_b}, [utf8_semi_line]
    ) == (s_b, e_b)
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": s_b, "end_col": e_b},
        [utf8_semi_line],
    ) == [comp_snip]
    # Pre-sliced source_lines must not be double-sliced when original start_col/end_col are present
    assert _slice_unit_token_lines(
        {
            "kind": "comprehension",
            "start_col": s_b,
            "end_col": e_b,
            "source_code": comp_snip,
        },
        [comp_snip],
    ) == [comp_snip]

    # Semicolon-separated statement unit after UTF-8 prefix
    stmt_snip = "res = a + b"
    stmt_s, stmt_e = _unit_cols(
        {"kind": "basic_block", "source_code": stmt_snip}, [utf8_semi_line]
    )
    assert stmt_s is not None and stmt_e is not None
    semi_stmt_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "compound_block",
            "source_text": utf8_semi_src,
            "source_code": stmt_snip,
        }
    )
    assert semi_stmt_scope["inputs"] == ["a", "b"]
    assert semi_stmt_scope["outputs"] == ["res"]

    # 2B. Multiline expression with non-ASCII prefix before unit on first line
    utf8_multi_src = (
        "def build(items, scale):\n"
        '    header = "résumé_🚀", [\n'
        "        x * scale\n"
        "        for x in items\n"
        "    ] + [0]\n"
        "    return header\n"
    )
    multi_lines = utf8_multi_src.splitlines(keepends=True)[1:5]
    multi_snip = "[\n        x * scale\n        for x in items\n    ]"
    ms_col, me_col = _unit_cols(
        {"kind": "comprehension", "source_code": multi_snip}, multi_lines
    )
    assert ms_col == len('    header = "résumé_🚀", '.encode("utf-8"))
    assert me_col == len("    ]".encode("utf-8"))
    assert _unit_cols(
        {"kind": "comprehension", "start_col": ms_col}, multi_lines
    ) == (ms_col, me_col)
    assert _unit_cols(
        {"kind": "comprehension", "end_col": me_col}, multi_lines
    ) == (ms_col, me_col)
    sliced_multi = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": ms_col, "end_col": me_col},
        multi_lines,
    )
    assert "".join(sliced_multi) == multi_snip
    multi_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "kind": "comprehension",
            "source_text": utf8_multi_src,
            "source_code": multi_snip,
        }
    )
    assert set(multi_scope["inputs"]) == {"items", "scale"}

    # 2C. Multiline semicolon-separated statement with UTF-8 prefix
    utf8_multi_semi_src = (
        "def calc_pair(a, b):\n"
        '    tag = "café_🚀"; total = (\n'
        "        a + b\n"
        "    ); trailing = 99\n"
        "    return total\n"
    )
    multi_semi_snip = "total = (\n        a + b\n    )"
    multi_semi_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": utf8_multi_semi_src,
            "source_code": multi_semi_snip,
        }
    )
    assert multi_semi_scope["inputs"] == ["a", "b"]
    assert multi_semi_scope["outputs"] == ["total"]

    # 2D. Single-line function body extraction with UTF-8 in signature header
    one_line_fn = 'def greet(prefix="café_🚀"): return prefix + "!"'
    assert _extract_unit_body_lines(
        {"kind": "function"}, [one_line_fn]
    ) == ['return prefix + "!"']

    # --- Finding 4: Conditional exits in finally (return, raise, nested branches) + differential execution ---
    # 4A. Reviewer exact reproduction: try/finally with `if flag: return` then `value = 2`
    src_f4_ret = (
        "def outer(flag, items):\n"
        "    try:\n"
        "        value = 1\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_ret = {
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_f4_ret,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        sc_f4_ret = inspect_single_unit_scope(u_f4_ret)
    assert "severed_closure_cell" not in sc_f4_ret["control_flow_hazards"]
    assert sc_f4_ret["is_control_flow_safe"] is True

    # 4B. Conditional raise and nested branches in finally where all continuing paths bind `value`
    src_f4_nested = (
        "def outer(flag, mode, items):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
        "    finally:\n"
        "        if flag:\n"
        "            if mode == 1:\n"
        "                value = 10\n"
        "            else:\n"
        "                return []\n"
        "        else:\n"
        "            if mode == 2:\n"
        "                raise ValueError('abort')\n"
        "            value = 20\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_nested = {
        "start": 17,
        "end": 18,
        "kind": "compound_block",
        "source_text": src_f4_nested,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        sc_f4_nested = inspect_single_unit_scope(u_f4_nested)
    assert "severed_closure_cell" not in sc_f4_nested["control_flow_hazards"]

    # 4C. Continuing path in finally does NOT bind `value` -> hazard MUST be raised
    src_f4_unsafe = (
        "def outer(flag, mode, items):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
        "    finally:\n"
        "        if flag:\n"
        "            if mode == 1:\n"
        "                value = 10\n"
        "        else:\n"
        "            value = 20\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_unsafe = {
        "start": 13,
        "end": 14,
        "kind": "compound_block",
        "source_text": src_f4_unsafe,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    sc_f4_unsafe = inspect_single_unit_scope(u_f4_unsafe)
    assert "severed_closure_cell" in sc_f4_unsafe["control_flow_hazards"]

    # 4D. Differential execution of refactored try/finally with conditional return
    mod_f4_src = (
        "def f1(flag, items):\n"
        "    try:\n"
        "        value = 1\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n\n"
        "def f2(flag, items):\n"
        "    try:\n"
        "        value = 5\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    f4_file = tmp_path / "mod_f4.py"
    f4_file.write_text(mod_f4_src, encoding="utf-8")
    u_f4_1 = {
        "file": str(f4_file),
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "name": "f1_loop",
    }
    u_f4_2 = {
        "file": str(f4_file),
        "start": 21,
        "end": 22,
        "kind": "compound_block",
        "name": "f2_loop",
    }
    refactored_f4 = (
        "def _shared_f4_loop(cbs, items, value):\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n\n"
        + refactor_module_units(
            mod_f4_src,
            [
                (u_f4_1, "    _shared_f4_loop(cbs, items, value)"),
                (u_f4_2, "    _shared_f4_loop(cbs, items, value)"),
            ],
        )
    )
    ns_orig_f4: dict = {}
    ns_ref_f4: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(mod_f4_src, ns_orig_f4)  # pylint: disable=exec-used
        exec(refactored_f4, ns_ref_f4)  # pylint: disable=exec-used
    assert ns_orig_f4["f1"](True, [1, 2]) == ns_ref_f4["f1"](True, [1, 2]) == []
    assert [cb() for cb in ns_orig_f4["f1"](False, [3, 4])] == [
        cb() for cb in ns_ref_f4["f1"](False, [3, 4])
    ] == [5, 6]

    # --- Finding 5: Exception-handler binding analysis & differential execution ---
    # 5A. Reviewer example: conditional del in try before operation(), closure in except handler
    src_f5_del_in_try = (
        "def outer(cond, op, recover):\n"
        "    value = 10\n"
        "    try:\n"
        "        if cond():\n"
        "            del value\n"
        "        op()\n"
        "    except Exception:\n"
        "        callback = lambda: value\n"
        "        recover()\n"
        "        return callback\n"
        "    return lambda: value\n"
    )
    u_f5_in_handler = {
        "start": 8,
        "end": 9,
        "kind": "compound_block",
        "source_text": src_f5_del_in_try,
        "source_code": "callback = lambda: value\nrecover()\n",
    }
    sc_f5_in_handler = inspect_single_unit_scope(u_f5_in_handler)
    assert "severed_closure_cell" in sc_f5_in_handler["control_flow_hazards"]
    assert sc_f5_in_handler["is_control_flow_safe"] is False

    # Verify why extracting u_f5_in_handler as `_shared(recover, value)` would break original runtime semantics:
    # When cond() is True and op() raises, original outer() runs recover() and returns callback without raising
    # UnboundLocalError during handler execution, whereas passing `value` eagerly raises UnboundLocalError!
    ns_f5: dict = {}
    exec(src_f5_del_in_try, ns_f5)  # pylint: disable=exec-used
    recovered_calls: list = []
    cb_res = ns_f5["outer"](
        lambda: True,
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda: recovered_calls.append("ok"),
    )
    assert recovered_calls == ["ok"]
    with pytest.raises(NameError):
        cb_res()

    # 5B. Exception raised before assignment in try, unit after try/except
    src_f5_pre_assign = (
        "def outer(op, recover, items):\n"
        "    try:\n"
        "        op()\n"
        "        value = 10\n"
        "    except Exception:\n"
        "        recover()\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    sc_f5_pre_assign = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f5_pre_assign,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert "severed_closure_cell" in sc_f5_pre_assign["control_flow_hazards"]

    # 5C. `except Exception as value:` implicitly deletes `value` at handler exit
    src_f5_except_as = (
        "def outer(op, items):\n"
        "    value = 10\n"
        "    try:\n"
        "        op()\n"
        "    except Exception as value:\n"
        "        pass\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: (value, x))\n"
        "    return cbs\n"
    )
    sc_f5_except_as = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f5_except_as,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (value, x))\n",
        }
    )
    assert "severed_closure_cell" in sc_f5_except_as["control_flow_hazards"]

    # 5D. Assignment in `finally` restores definite binding after `del value` in `try` and `except ... as value`
    src_f5_finally_rebind = (
        "def outer(op, items):\n"
        "    value = 10\n"
        "    try:\n"
        "        del value\n"
        "        op()\n"
        "    except Exception as value:\n"
        "        pass\n"
        "    finally:\n"
        "        value = 42\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    sc_f5_finally_rebind = inspect_single_unit_scope(
        {
            "start": 11,
            "end": 12,
            "kind": "compound_block",
            "source_text": src_f5_finally_rebind,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert (
        "severed_closure_cell"
        not in sc_f5_finally_rebind["control_flow_hazards"]
    )

    # --- 8-Area Review Validation Matrix ---
    # Area 1: Enclosing-function detection with decorators, multiline headers, and multiple expressions on one line
    src_area1 = (
        "def outer(dec1, dec2, items):\n"
        "    scale = 3\n"
        "    @dec1(lambda v: v + scale)\n"
        "    @dec2\n"
        "    def inner(\n"
        "        a=[x * scale for x in items],\n"
        "    ):\n"
        "        scale = 99\n"
        "        return [x + 1 for x in a]; [x + 2 for x in a]\n"
        "    return inner\n"
    )
    sc_area1_hdr = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "comprehension",
            "source_text": src_area1,
            "source_code": "[x * scale for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_area1_hdr["control_flow_hazards"]

    # Area 2: Three-level nested functions with same-name locals, nonlocal declarations, and del
    src_area2 = (
        "def level0(items):\n"
        "    acc = 1\n"
        "    def level1():\n"
        "        acc = 2\n"
        "        def level2():\n"
        "            nonlocal acc\n"
        "            del acc\n"
        "        return level2\n"
        "    cbs = [lambda x=x: x + acc for x in items]\n"
        "    return level1, cbs\n"
    )
    sc_area2 = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "comprehension",
            "source_text": src_area2,
            "source_code": "[lambda x=x: x + acc for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_area2["control_flow_hazards"]

    # Area 3: Sibling closure propagation with multiple same-name definitions, recursive closures, and alias cycles
    src_area3 = (
        "def outer(flag, items):\n"
        "    acc = 0\n"
        "    if flag:\n"
        "        def helper(n):\n"
        "            return helper(n - 1) if n > 0 else 0\n"
        "    else:\n"
        "        def helper(n):\n"
        "            return acc + n\n"
        "    a = helper\n"
        "    b = a\n"
        "    a = b\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(b(x))\n"
        "    return acc, out\n"
    )
    sc_area3 = inspect_single_unit_scope(
        {
            "start": 13,
            "end": 15,
            "kind": "compound_block",
            "source_text": src_area3,
            "source_code": "for x in items:\n    acc += x\n    out.append(b(x))\n",
        }
    )
    assert "severed_closure_cell" in sc_area3["control_flow_hazards"]

    # Area 4: Definite bindings with unconditional walrus in `if (scale := ...):` and `with`
    src_area4 = (
        "def outer(x, items):\n"
        "    if (scale := x + 1) > 0:\n"
        "        offset = 1\n"
        "    else:\n"
        "        offset = 2\n"
        "    cbs = []\n"
        "    for item in items:\n"
        "        cbs.append(lambda item=item: item * scale + offset)\n"
        "    return cbs\n"
    )
    sc_area4 = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_area4,
            "source_code": "for item in items:\n    cbs.append(lambda item=item: item * scale + offset)\n",
        }
    )
    assert "severed_closure_cell" not in sc_area4["control_flow_hazards"]

    # Area 5: Generator expressions with shadowed built-in `sum` (treated as lazy/deferred)
    src_area5 = (
        "def outer(sum, items):\n"
        "    acc = 0\n"
        "    deferred = sum(acc + x for x in items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(deferred(x))\n"
        "    return deferred, acc, out\n"
    )
    sc_area5 = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5,
            "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
        }
    )
    assert "severed_closure_cell" in sc_area5["control_flow_hazards"]

    # Contrast: unshadowed `sum(acc + x for x in items)` is eager and safe
    src_area5_eager = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    total = sum(acc + x for x in items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(total + x)\n"
        "    return total, acc, out\n"
    )
    sc_area5_eager = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5_eager,
            "source_code": "for x in items:\n    acc += x\n    out.append(total + x)\n",
        }
    )
    assert "severed_closure_cell" not in sc_area5_eager["control_flow_hazards"]

    # Area 6: Cache correctness across clear_dataflow_cache and source changes
    clear_dataflow_cache()
    sc_area5_after_clear = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5,
            "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
        }
    )
    assert (
        sc_area5_after_clear["control_flow_hazards"]
        == sc_area5["control_flow_hazards"]
    )

    # Area 7: Clone-output resolution rejects severed_closure_cell in both normal and generator-subroutine paths
    u_unsafe_pair = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": src_area5,
        "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
    }
    assert (
        resolve_clone_pair_outputs(
            u_unsafe_pair,
            dict(u_unsafe_pair),
            sc_area5,
            dict(sc_area5),
            source_text1=src_area5,
            source_text2=src_area5,
        )
        is None
    )

    src_area7_gen = (
        "def gen(items):\n"
        "    cb = lambda: acc\n"
        "    acc = 0\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        yield acc\n"
        "    return acc, cb\n"
    )
    u_area7_gen = {
        "start": 3,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_area7_gen,
        "source_code": "acc = 0\nfor x in items:\n    acc += x\n    yield acc\n",
    }
    sc_area7_gen = inspect_single_unit_scope(u_area7_gen)
    assert (
        resolve_clone_generator_subroutine_outputs(
            u_area7_gen,
            dict(u_area7_gen),
            sc_area7_gen,
            dict(sc_area7_gen),
            source_text1=src_area7_gen,
            source_text2=src_area7_gen,
            closure_strictness="strict",
        )
        is None
    )


def test_same_stmt_evaluation_order_comp_walrus_and_decorator_alignment() -> None:
    """Verifies same-stmt RHS->LHS order (A), multiline ';' indent (B), comp walrus deferred reads (C), non-escaping nonlocal mutator (D), and decorator column snapping (E)."""
    import ast
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _analyze_enclosing_fn_cells,
        _extract_comprehension_deferred_closure_info,
    )
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        inspect_single_unit_scope,
    )

    # Finding A1: Same-statement Assign LHS target is post-mutated (not pre-bound) when unit is on RHS
    src_a1 = (
        "def outer(items):\n"
        "    cbs = [lambda y, x=x: (cbs, x, y) for x in items]\n"
        "    return cbs\n"
    )
    tree_a1 = ast.parse(src_a1)
    bound_a1, in_loop_a1 = _find_enclosing_pre_bound_names(
        tree_a1, 2, 2, u_start_col=10, u_end_col=53
    )
    assert "cbs" not in bound_a1
    assert not in_loop_a1
    info_a1 = _analyze_enclosing_fn_cells(
        tree_a1.body[0], (2, 2, 10, 53), mod_tree=tree_a1
    )
    assert "cbs" not in info_a1.pre_bound
    assert "cbs" in info_a1.post_mutated
    sc_a1 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 10,
            "end_col": 53,
            "kind": "comprehension",
            "source_text": src_a1,
            "source_code": "[lambda y, x=x: (cbs, x, y) for x in items]",
        },
        tree=tree_a1,
    )
    assert "severed_closure_cell" in sc_a1["control_flow_hazards"]

    # Finding A2: Same-statement Assign with prior binding of LHS variable still records LHS as post_mutated
    src_a2 = (
        "def outer(items):\n"
        "    cbs = None\n"
        "    cbs = [lambda y, x=x: (cbs, x, y) for x in items]\n"
        "    return cbs\n"
    )
    tree_a2 = ast.parse(src_a2)
    sc_a2 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 53,
            "kind": "comprehension",
            "source_text": src_a2,
            "source_code": "[lambda y, x=x: (cbs, x, y) for x in items]",
        },
        tree=tree_a2,
    )
    assert "severed_closure_cell" in sc_a2["control_flow_hazards"]

    # Finding A3: For-loop iter expression unit treats loop target as post_mutated + loop_mutated (not pre_bound)
    src_a3 = (
        "def outer(items):\n"
        "    out = []\n"
        "    for x in [lambda y=i: (x, y) for i in items]:\n"
        "        out.append(x)\n"
        "    return out\n"
    )
    tree_a3 = ast.parse(src_a3)
    bound_a3, in_loop_a3 = _find_enclosing_pre_bound_names(
        tree_a3, 3, 3, u_start_col=13, u_end_col=48
    )
    assert "x" not in bound_a3
    assert not in_loop_a3
    info_a3 = _analyze_enclosing_fn_cells(
        tree_a3.body[0], (3, 3, 13, 48), mod_tree=tree_a3
    )
    assert "x" not in info_a3.pre_bound
    assert "x" in info_a3.post_mutated
    assert "x" in info_a3.loop_mutated
    sc_a3 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 13,
            "end_col": 48,
            "kind": "comprehension",
            "source_text": src_a3,
            "source_code": "[lambda y=i: (x, y) for i in items]",
        },
        tree=tree_a3,
    )
    assert "severed_closure_cell" in sc_a3["control_flow_hazards"]

    # Finding B: Multiline semicolon-prefixed multi-statement unit preserves line-0 base indentation
    src_b = (
        "def compute(x):\n"
        "    prefix = 1; a = (\n"
        "        x + 1\n"
        "    )\n"
        "    b = a * 2; suffix = 3\n"
        "    return b + prefix + suffix\n"
    )
    sc_b = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "start_col": 16,
            "end_col": 13,
            "kind": "compound_block",
            "source_text": src_b,
            "source_code": "a = (\n    x + 1\n)\nb = a * 2\n",
        },
        tree=ast.parse(src_b),
    )
    assert sc_b["inputs"] == ["x"]
    assert sc_b["outputs"] == ["a", "b"]
    assert sc_b["is_control_flow_safe"] is True

    # Finding C: Comprehension walrus target captured by escaping closure inside comprehension is kept in deferred_reads
    comp_c = ast.parse("[(w := v, lambda: w) for v in items]", mode="eval").body
    c_reads, _ = _extract_comprehension_deferred_closure_info(comp_c)  # type: ignore[arg-type]
    assert "w" in c_reads

    comp_c_gen = ast.parse(
        "[(w := v, (w for _ in range(1))) for v in items]", mode="eval"
    ).body
    c_gen_reads, _ = _extract_comprehension_deferred_closure_info(comp_c_gen)  # type: ignore[arg-type]
    assert "w" in c_gen_reads

    src_c = (
        "def outer(items):\n"
        "    pairs = [(w := v, lambda: w) for v in items]\n"
        "    w = 99\n"
        "    res = pairs[0][1]() + w\n"
        "    return res\n"
    )
    sc_c = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_c,
            "source_code": "w = 99\nres = pairs[0][1]() + w\n",
        },
        tree=ast.parse(src_c),
    )
    assert "severed_closure_cell" in sc_c["control_flow_hazards"]

    sc_c_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 12,
            "end_col": 48,
            "kind": "comprehension",
            "source_text": src_c,
            "source_code": "[(w := v, lambda: w) for v in items]",
        },
        tree=ast.parse(src_c),
    )
    assert "severed_closure_cell" in sc_c_comp["control_flow_hazards"]

    # Finding D: Non-escaping intra-unit nonlocal mutator on top-level unit local is safe unless outer closure shares cell
    src_d_safe = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    x = x * 2\n"
        "    def bump():\n"
        "        nonlocal x\n"
        "        x += 10\n"
        "    bump()\n"
        "    return x\n"
    )
    sc_d_safe = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_d_safe,
            "source_code": (
                "x = x * 2\n"
                "def bump():\n"
                "    nonlocal x\n"
                "    x += 10\n"
                "bump()\n"
            ),
        },
        tree=ast.parse(src_d_safe),
    )
    assert "severed_closure_cell" not in sc_d_safe["control_flow_hazards"]

    src_d_unsafe_sib = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    obs = lambda: x\n"
        "    x = x * 2\n"
        "    def bump():\n"
        "        nonlocal x\n"
        "        x += 10\n"
        "    bump()\n"
        "    return x, obs\n"
    )
    sc_d_unsafe_sib = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_d_unsafe_sib,
            "source_code": (
                "x = x * 2\n"
                "def bump():\n"
                "    nonlocal x\n"
                "    x += 10\n"
                "bump()\n"
            ),
        },
        tree=ast.parse(src_d_unsafe_sib),
    )
    assert "severed_closure_cell" in sc_d_unsafe_sib["control_flow_hazards"]

    # Finding E: _node_earliest_start_pos snaps @foo, @ foo, and @  foo to def col and preserves unaligned earlier col
    tree_e_aligned = ast.parse(
        "def outer():\n"
        "    @  dec\n"
        "    def inner():\n"
        "        return 1\n"
    )
    inner_aligned = tree_e_aligned.body[0].body[0]  # type: ignore[attr-defined]
    assert _node_earliest_start_pos(inner_aligned) == (2, 4)
    inner_aligned.decorator_list[0].col_offset = 9
    assert _node_earliest_start_pos(inner_aligned) == (2, 4)
    inner_aligned.decorator_list[0].col_offset = 2
    assert _node_earliest_start_pos(inner_aligned) == (2, 1)


def test_lexical_def_escape_identity_genexp_matrix_and_utf8_ambiguity() -> None:
    """Covers lexical def escape identity, column-aware shortcuts, mod_tree cache isolation, genexp matrix, runtime definite-binding validation, and UTF-8 single-boundary ambiguity."""
    import sys
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _SpanBounds,
        _analyze_enclosing_fn_cells,
        _collect_escaped_intra_unit_defs,
        _collect_escaped_intra_unit_names,
        _extract_block_definite_bindings_and_term,
        _find_enclosing_function_chain,
        _get_fn_scope_context,
        _make_empty_traversal_buffers,
        _try_apply_cached_stmt_delta,
    )
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        _unit_cols,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    # 1. Item 1: Lexical-scope definition identity in _collect_escaped_intra_unit_defs
    src_item1_safe = (
        "def entry(a):\n"
        "    value = a + 10\n"
        "    value = value * 2\n"
        "    def helper():\n"
        "        def callback():\n"
        "            nonlocal value\n"
        "            value += 5\n"
        "        callback()\n"
        "        return 0\n"
        "    helper()\n"
        "    def other():\n"
        "        def callback():\n"
        "            return 1\n"
        "        return callback\n"
        "    escaped_cb = other()\n"
        "    return value, escaped_cb\n"
    )
    tree_item1_safe = ast.parse(src_item1_safe)
    unit_code_item1 = (
        "value = value * 2\n"
        "def helper():\n"
        "    def callback():\n"
        "        nonlocal value\n"
        "        value += 5\n"
        "    callback()\n"
        "    return 0\n"
        "helper()\n"
        "def other():\n"
        "    def callback():\n"
        "        return 1\n"
        "    return callback\n"
        "escaped_cb = other()\n"
    )
    cand_item1 = ast.parse(unit_code_item1).body
    helper_cb_node = cand_item1[1].body[0]  # type: ignore[attr-defined]
    other_cb_node = cand_item1[3].body[0]  # type: ignore[attr-defined]
    esc_names_1, esc_ids_1 = _collect_escaped_intra_unit_defs(
        cand_item1, {"value", "escaped_cb", "callback"}
    )
    assert "escaped_cb" in esc_names_1
    assert id(helper_cb_node) not in esc_ids_1
    assert id(other_cb_node) in esc_ids_1
    assert "callback" not in _collect_escaped_intra_unit_names(
        cand_item1, {"value", "escaped_cb"}
    )
    sc_item1_safe = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 15,
            "kind": "compound_block",
            "source_text": src_item1_safe,
            "source_code": unit_code_item1,
        },
        tree=tree_item1_safe,
    )
    assert "severed_closure_cell" not in sc_item1_safe["control_flow_hazards"]

    # Ambiguous same-scope branch definitions with same name conservatively mark both as escaping
    src_item1_ambig = (
        "def entry(a, flag):\n"
        "    value = a + 10\n"
        "    value = value * 2\n"
        "    if flag:\n"
        "        def callback():\n"
        "            nonlocal value\n"
        "            value += 5\n"
        "    else:\n"
        "        def callback():\n"
        "            return 1\n"
        "    return value, callback\n"
    )
    sc_item1_ambig = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 10,
            "kind": "compound_block",
            "source_text": src_item1_ambig,
            "source_code": (
                "value = value * 2\n"
                "if flag:\n"
                "    def callback():\n"
                "        nonlocal value\n"
                "        value += 5\n"
                "else:\n"
                "    def callback():\n"
                "        return 1\n"
            ),
        },
        tree=ast.parse(src_item1_ambig),
    )
    assert "severed_closure_cell" in sc_item1_ambig["control_flow_hazards"]

    # 2. Item 2: Column-aware _try_apply_cached_stmt_delta and same-line scope boundaries
    same_line_src = (
        "def f(items):\n"
        "    a = 1; b = [lambda: a for _ in items]; a = 2\n"
        "    return b, a\n"
    )
    same_line_tree = ast.parse(same_line_src)
    fn_same_line = same_line_tree.body[0]
    assert isinstance(fn_same_line, ast.FunctionDef)
    ctx_sl = _get_fn_scope_context(fn_same_line, mod_tree=same_line_tree)
    bufs_sl = _make_empty_traversal_buffers(frozenset(), ctx_sl)
    deltas_sl: dict = {}
    span_sl = _SpanBounds(2, 2, 11, 41)
    assert (
        _try_apply_cached_stmt_delta(
            fn_same_line.body[0], span_sl, bufs_sl, deltas_sl
        )
        is True
    )
    assert "a" in bufs_sl.state.pre_bound
    assert (
        _try_apply_cached_stmt_delta(
            fn_same_line.body[2], span_sl, bufs_sl, deltas_sl
        )
        is True
    )
    assert "a" in bufs_sl.state.post_mutated

    # _find_enclosing_function_chain and _find_enclosing_pre_bound_names with end_col in header vs body
    hdr_body_src = (
        "def outer(items):\n"
        "    x = 1\n"
        "    def inner(d=[i + x for i in items]): return d\n"
        "    return inner\n"
    )
    hdr_body_tree = ast.parse(hdr_body_src)
    chain_hdr = _find_enclosing_function_chain(
        hdr_body_tree, 3, 3, u_start_col=16, u_end_col=38
    )
    assert [n.name for _, _, n in chain_hdr] == ["outer"]  # type: ignore[attr-defined]
    chain_body = _find_enclosing_function_chain(
        hdr_body_tree, 3, 3, u_start_col=41, u_end_col=49
    )
    assert [n.name for _, _, n in chain_body] == ["outer", "inner"]  # type: ignore[attr-defined]
    pre_hdr, _ = _find_enclosing_pre_bound_names(
        hdr_body_tree, 3, 3, u_start_col=16, u_end_col=38
    )
    assert "x" in pre_hdr and "d" not in pre_hdr
    pre_body, _ = _find_enclosing_pre_bound_names(
        hdr_body_tree, 3, 3, u_start_col=41, u_end_col=49
    )
    assert "d" in pre_body

    # 3. Item 3: Closure analysis cache isolation across distinct mod_tree contexts and eviction
    shared_fn_src = (
        "def worker(items):\n"
        "    v = 1\n"
        "    seq = list(v + x for x in items)\n"
        "    v = 99\n"
        "    res = v + 1\n"
        "    return seq, res\n"
    )
    mod_clean = ast.parse("SCALE = 1\n" + shared_fn_src)
    mod_shadow = ast.parse("list = custom_list\n" + shared_fn_src)
    shared_fn_node = mod_clean.body[1]
    ctx_clean = _get_fn_scope_context(shared_fn_node, mod_tree=mod_clean)
    ctx_shadow = _get_fn_scope_context(shared_fn_node, mod_tree=mod_shadow)
    assert isinstance(ctx_clean.fn_locals, frozenset)
    assert isinstance(ctx_clean.shadowed_names, frozenset)
    assert "list" not in ctx_clean.shadowed_names
    assert "list" in ctx_shadow.shadowed_names
    info_clean = _analyze_enclosing_fn_cells(
        shared_fn_node, (5, 6, None, None), mod_tree=mod_clean
    )
    info_shadow = _analyze_enclosing_fn_cells(
        shared_fn_node, (5, 6, None, None), mod_tree=mod_shadow
    )
    assert not any("v" in r for _, r, _ in info_clean.sibling_closures)
    assert any("v" in r for _, r, _ in info_shadow.sibling_closures)
    _evict_tree_scopes(mod_shadow)
    assert not any(
        isinstance(k, tuple) and mod_shadow in k for k in _scope_cell_info_cache
    )
    clear_dataflow_cache()
    assert len(_scope_cell_info_cache) == 0

    # 4. Item 4: Generator-expression 6-path interaction matrix
    # 4.1 Lazy genexp consumed AFTER rebinding vs BEFORE rebinding
    src_gen_after = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    value = 10\n"
        "    return list(gen)\n"
    )
    sc_gen_after = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 36,
            "kind": "comprehension",
            "source_text": src_gen_after,
            "source_code": "(value + x for x in items)",
        },
        tree=ast.parse(src_gen_after),
    )
    assert "severed_closure_cell" in sc_gen_after["control_flow_hazards"]

    src_gen_before = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    out = list(gen)\n"
        "    value = 10\n"
        "    res = value + 1\n"
        "    return out, res\n"
    )
    sc_gen_before = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_text": src_gen_before,
            "source_code": "value = 10\nres = value + 1\n",
        },
        tree=ast.parse(src_gen_before),
    )
    assert "severed_closure_cell" not in sc_gen_before["control_flow_hazards"]

    # 4.2 Generator expression returned from unit to caller before outer rebinding
    src_gen_ret = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    res = value + 2\n"
        "    value = 10\n"
        "    return gen, res\n"
    )
    sc_gen_ret = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_gen_ret,
            "source_code": "gen = (value + x for x in items)\nres = value + 2\n",
        },
        tree=ast.parse(src_gen_ret),
    )
    assert "severed_closure_cell" in sc_gen_ret["control_flow_hazards"]

    # 4.3 Shadowed sum/tuple consumer treated as lazy
    src_gen_shadow_sum = (
        "def outer(items, sum):\n"
        "    value = 1\n"
        "    total = sum(value + x for x in items)\n"
        "    value = 10\n"
        "    return total\n"
    )
    sc_gen_shadow_sum = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 15,
            "end_col": 41,
            "kind": "comprehension",
            "source_text": src_gen_shadow_sum,
            "source_code": "(value + x for x in items)",
        },
        tree=ast.parse(src_gen_shadow_sum),
    )
    assert "severed_closure_cell" in sc_gen_shadow_sum["control_flow_hazards"]

    # 4.4 Generator expression nested in list comprehension
    src_gen_in_listcomp = (
        "def outer(matrix):\n"
        "    value = 1\n"
        "    gens = [(value + x for x in row) for row in matrix]\n"
        "    value = 10\n"
        "    return [list(g) for g in gens]\n"
    )
    sc_gen_in_listcomp = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 11,
            "end_col": 55,
            "kind": "comprehension",
            "source_text": src_gen_in_listcomp,
            "source_code": "[(value + x for x in row) for row in matrix]",
        },
        tree=ast.parse(src_gen_in_listcomp),
    )
    assert "severed_closure_cell" in sc_gen_in_listcomp["control_flow_hazards"]

    # 4.5 Walrus in first iterable (eager) vs in body (lazy)
    src_walrus_iter = (
        "def outer(items):\n"
        "    w = 0\n"
        "    gen = (x + 1 for x in (w := items))\n"
        "    y = w\n"
        "    z = len(y)\n"
        "    return list(gen), z\n"
    )
    sc_walrus_iter = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_walrus_iter,
            "source_code": "y = w\nz = len(y)\n",
        },
        tree=ast.parse(src_walrus_iter),
    )
    assert "severed_closure_cell" not in sc_walrus_iter["control_flow_hazards"]

    src_walrus_body = (
        "def outer(items):\n"
        "    w = 0\n"
        "    gen = ((w := x) for x in items)\n"
        "    w = 10\n"
        "    res = list(gen) + [w]\n"
        "    return res\n"
    )
    sc_walrus_body = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_walrus_body,
            "source_code": "w = 10\nres = list(gen) + [w]\n",
        },
        tree=ast.parse(src_walrus_body),
    )
    assert "severed_closure_cell" in sc_walrus_body["control_flow_hazards"]

    # 4.6 Nested generator expressions where inner target shadows outer variable
    src_gen_shadow_target = (
        "def outer(matrix):\n"
        "    value = 1\n"
        "    gen = (sum(value + 1 for value in row) for row in matrix)\n"
        "    value = 10\n"
        "    return list(gen), value\n"
    )
    sc_gen_shadow_target = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 61,
            "kind": "comprehension",
            "source_text": src_gen_shadow_target,
            "source_code": "(sum(value + 1 for value in row) for row in matrix)",
        },
        tree=ast.parse(src_gen_shadow_target),
    )
    assert "value" not in sc_gen_shadow_target["inputs"]
    assert "severed_closure_cell" not in sc_gen_shadow_target["control_flow_hazards"]

    # 5. Item 5: Definite-binding validation against Python runtime execution & refactoring decision
    prog_try_finally = (
        "def run_case(flag):\n"
        "    try:\n"
        "        if flag:\n"
        "            a = 1\n"
        "        else:\n"
        "            a = 2\n"
        "    finally:\n"
        "        if flag == 'abort':\n"
        "            raise RuntimeError('stop')\n"
        "        b = a + 10\n"
        "    return a, b\n"
    )
    tree_tf = ast.parse(prog_try_finally)
    fn_tf = tree_tf.body[0]
    assert isinstance(fn_tf, ast.FunctionDef)
    def_bound_tf, all_term_tf = _extract_block_definite_bindings_and_term(
        fn_tf.body[:1], None
    )
    assert def_bound_tf == {"a", "b"}
    assert all_term_tf is False
    ns_tf: dict = {}
    exec(prog_try_finally, ns_tf)  # nosec B102
    assert ns_tf["run_case"](True) == (1, 11)
    assert ns_tf["run_case"](False) == (2, 12)

    # Validate unseeded conditional store with sibling closure: both scope fields & resolve_clone_pair_outputs
    src_unseeded = (
        "def outer(flag):\n"
        "    if flag:\n"
        "        value = 10\n"
        "    def read():\n"
        "        return value\n"
        "    if not flag:\n"
        "        value = 20\n"
        "    return read()\n"
    )
    u_unseeded = {
        "start": 2,
        "end": 3,
        "kind": "if_block",
        "source_text": src_unseeded,
        "source_code": "if flag:\n    value = 10\n",
    }
    sc_unseeded = inspect_single_unit_scope(
        u_unseeded, tree=ast.parse(src_unseeded)
    )
    assert sc_unseeded["definite_stores"] == []
    assert sc_unseeded["outputs"] == ["value"]
    assert sc_unseeded["conditional_outputs"] == ["value"]
    assert "severed_closure_cell" in sc_unseeded["control_flow_hazards"]
    assert (
        resolve_clone_pair_outputs(
            u_unseeded,
            dict(u_unseeded),
            sc_unseeded,
            dict(sc_unseeded),
            source_text1=src_unseeded,
            source_text2=src_unseeded,
        )
        is None
    )

    if sys.version_info >= (3, 10):
        prog_match = (
            "def eval_match(x):\n"
            "    match x:\n"
            "        case 1 if x > 0:\n"
            "            m = 10\n"
            "        case _:\n"
            "            m = 20\n"
            "    return m\n"
        )
        tree_m = ast.parse(prog_match)
        def_m, term_m = _extract_block_definite_bindings_and_term(
            tree_m.body[0].body[:1], None  # type: ignore[attr-defined]
        )
        assert def_m == {"m"} and not term_m
        ns_m: dict = {}
        exec(prog_match, ns_m)  # nosec B102
        assert ns_m["eval_match"](1) == 10
        assert ns_m["eval_match"](2) == 20

    # 6. Item 6: UTF-8 single-boundary ambiguity preservation and round-trip verification
    ambig_expr_line = ['    tag = "αβγ_🚀"; res = a + b + c\n']
    res_s_col = len('    tag = "αβγ_🚀"; res = '.encode("utf-8"))
    # Starting at `a` in `a + b + c`, both `a + b` and `a + b + c` are compound ast.expr nodes -> ambiguous!
    assert _unit_cols(
        {"kind": "complex_expr", "start_col": res_s_col}, ambig_expr_line
    ) == (res_s_col, None)
    # Ending at `c` in `a + b + c`, only `(a + b) + c` is a non-leaf ast.expr ending there -> unambiguous!
    res_e_col = len('    tag = "αβγ_🚀"; res = a + b + c'.encode("utf-8"))
    assert _unit_cols(
        {"kind": "complex_expr", "end_col": res_e_col}, ambig_expr_line
    ) == (res_s_col, res_e_col)

    # Single-line compound statement `if flag: a = 1; b = 2` with only end_col has both If and Assign ending at end_col -> ambiguous!
    ambig_stmt_line = ['    if flag: a = "café"; b = "🚀"\n']
    stmt_end_col = len(ambig_stmt_line[0].rstrip("\r\n").encode("utf-8"))
    assert _unit_cols(
        {"kind": "basic_block", "end_col": stmt_end_col}, ambig_stmt_line
    ) == (None, stmt_end_col)

    # Round-trip check across multibyte prefixes and unique expressions
    for prefix in ('    p = "αβγ"; ', '    p = "日本語_🚀_✨"; ', "    "):
        for expr_str, kind_str in (
            ("[x * 2 for x in items]", "comprehension"),
            ("a + b", "complex_expr"),
            ("{k: v for k, v in pairs}", "comprehension"),
        ):
            full_ln = f"{prefix}target = {expr_str}; tail = 1 # comment 🚀\n"
            exp_s = len(f"{prefix}target = ".encode("utf-8"))
            exp_e = exp_s + len(expr_str.encode("utf-8"))
            assert _unit_cols(
                {"kind": kind_str, "source_code": expr_str}, [full_ln]
            ) == (exp_s, exp_e)
            assert _unit_cols(
                {"kind": kind_str, "start_col": exp_s}, [full_ln]
            ) == (exp_s, exp_e)
            assert _unit_cols(
                {"kind": kind_str, "end_col": exp_e}, [full_ln]
            ) == (exp_s, exp_e)
            assert _slice_utf8_line(full_ln, exp_s, exp_e) == expr_str
            assert _slice_unit_token_lines(
                {"kind": kind_str, "start_col": exp_s, "end_col": exp_e},
                [full_ln],
            ) == [expr_str]


def test_chained_walrus_augassign_trystar_and_slice_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers chained-comparison walrus short-circuiting, conditional live-in hazards, x += unit post-reads, TryStar definite bindings, UTF-8 slice edge cases, NamedTuple mutators, and star-import caching."""
    import copy
    import pickle
    import sys
    from pydoppelgangerhunt.fixer import (
        _is_irrefutable_case,
        _is_irrefutable_pattern,
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.fixer._df_cache import (
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _IntraUnitMutator,
        _collect_unit_nonlocal_mutators,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
        _extract_block_definite_bindings_and_term,
        _extract_unconditional_walrus_names,
    )
    import pydoppelgangerhunt.fixer._df_closures as df_closures_mod
    from pydoppelgangerhunt.fixer._df_hazards import (
        _scope_reads_outputs_after_line,
    )
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    # 1. Chained comparison short-circuiting in _extract_unconditional_walrus_names and cell traversal
    chain_cmp_expr = ast.parse("a < b < (y := f())", mode="eval").body
    assert _extract_unconditional_walrus_names(chain_cmp_expr) == set()
    first_cmp_expr = ast.parse(
        "(x := g()) < (y := f()) < (z := h())", mode="eval"
    ).body
    assert _extract_unconditional_walrus_names(first_cmp_expr) == {"x", "y"}

    src_chain_short = (
        "def outer(a, b, f, items):\n"
        "    ok = a < b < (y := f())\n"
        "    cbs = [lambda x=x: x + y for x in items]\n"
        "    return ok, cbs\n"
    )
    tree_chain_short = ast.parse(src_chain_short)
    info_chain_short = _analyze_enclosing_fn_cells(
        tree_chain_short.body[0], (3, 3, 10, 44), mod_tree=tree_chain_short
    )
    assert "y" not in info_chain_short.pre_bound
    assert "y" in info_chain_short.pre_mutated
    sc_chain_short = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": src_chain_short,
            "source_code": "[lambda x=x: x + y for x in items]",
        },
        tree=tree_chain_short,
    )
    assert "severed_closure_cell" in sc_chain_short["control_flow_hazards"]

    src_chain_first = (
        "def outer(a, c, f, items):\n"
        "    ok = a < (y := f()) < c\n"
        "    cbs = [lambda x=x: x + y for x in items]\n"
        "    return ok, cbs\n"
    )
    tree_chain_first = ast.parse(src_chain_first)
    sc_chain_first = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": src_chain_first,
            "source_code": "[lambda x=x: x + y for x in items]",
        },
        tree=tree_chain_first,
    )
    assert "severed_closure_cell" not in sc_chain_first["control_flow_hazards"]

    # 2. Dropped inputs for conditional live-ins (v in pre_bound raises hazard; purely intra-unit stays safe)
    src_cond_live_in_pre = (
        "def entry(flag1, flag2):\n"
        "    if flag1:\n"
        "        x = 10\n"
        "    if flag2:\n"
        "        x = 20\n"
        "        y = x + 1\n"
        "    else:\n"
        "        y = x + 2\n"
        "    return y\n"
    )
    sc_cond_pre = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_cond_live_in_pre,
            "source_code": (
                "if flag2:\n"
                "    x = 20\n"
                "    y = x + 1\n"
                "else:\n"
                "    y = x + 2\n"
            ),
        },
        tree=ast.parse(src_cond_live_in_pre),
    )
    assert "x" not in sc_cond_pre["inputs"]
    assert "severed_closure_cell" in sc_cond_pre["control_flow_hazards"]
    assert sc_cond_pre["is_control_flow_safe"] is False
    _run_differential_refactoring_case(
        tmp_path,
        "cond_live_in_pre_bound_hazard",
        src_cond_live_in_pre,
        (4, 8),
        expect_patch=False,
        call_args_list=[(True, False), (False, True), (True, True)],
    )

    src_cond_intra_only = (
        "def entry(flag):\n"
        "    if flag:\n"
        "        x = 10\n"
        "    y = x + 1 if flag else 0\n"
        "    return y\n"
    )
    sc_cond_intra = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_cond_intra_only,
            "source_code": "if flag:\n    x = 10\ny = x + 1 if flag else 0\n",
        },
        tree=ast.parse(src_cond_intra_only),
    )
    assert "x" not in sc_cond_intra["inputs"]
    assert "severed_closure_cell" not in sc_cond_intra["control_flow_hazards"]

    # 3. Scope closure mode recomputation and non-mutation of caller u1/u2 dicts in patch.py
    src_patch_mode = (
        "def f1(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n\n"
        "def f2(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n"
    )
    patch_file = tmp_path / "mode_mod.py"
    patch_file.write_text(src_patch_mode, encoding="utf-8")
    u1_orig: dict[str, Any] = {
        "file": str(patch_file),
        "start": 2,
        "end": 3,
        "name": "f1:blk",
        "kind": "compound_block",
    }
    u2_orig: dict[str, Any] = {
        "file": str(patch_file),
        "start": 7,
        "end": 8,
        "name": "f2:blk",
        "kind": "compound_block",
    }
    _ = generate_refactoring_patch(
        [(1.0, u1_orig, u2_orig)],
        repo_root=str(tmp_path),
        replace_clones=False,
        closure_strictness="lenient",
    )
    assert "skip_pre_unit_closures" not in u1_orig
    assert "skip_pre_unit_closures" not in u2_orig
    recomputed_sc = ensure_unit_scope_closure_mode(
        u1_orig,
        {"_skip_pre_unit_closures": False, "inputs": ["a"]},
        True,
        source_text=src_patch_mode,
    )
    assert recomputed_sc["_skip_pre_unit_closures"] is True
    assert "skip_pre_unit_closures" not in u1_orig

    # 4. Performance: module star-import and single-node-locals caching across many units
    clear_dataflow_cache()
    star_walk_calls = 0
    orig_uncached_star = df_closures_mod._node_has_star_import_uncached

    def _counting_star_import(root: ast.AST) -> bool:
        nonlocal star_walk_calls
        if isinstance(root, ast.Module) and len(root.body) > 50:
            star_walk_calls += 1
        return orig_uncached_star(root)

    monkeypatch.setattr(
        df_closures_mod, "_node_has_star_import_uncached", _counting_star_import
    )
    large_lines = [f"CONST_{i} = {i}\n" for i in range(200)]
    for i in range(30):
        large_lines.append(
            f"def fn_{i}(items):\n"
            f"    scale = {i}\n"
            f"    tot = sum(x + scale for x in items)\n"
            f"    res = tot + scale\n"
            f"    return res\n"
        )
    large_src = "".join(large_lines)
    large_tree = ast.parse(large_src)
    for i in range(30):
        base_ln = 200 + i * 5 + 1
        sc_unit = inspect_single_unit_scope(
            {
                "start": base_ln + 2,
                "end": base_ln + 3,
                "kind": "compound_block",
                "source_text": large_src,
                "source_code": "tot = sum(x + scale for x in items)\nres = tot + scale\n",
            },
            tree=large_tree,
        )
        assert sc_unit["is_control_flow_safe"] is True
    assert star_walk_calls == 1
    assert (large_tree, "has_star_import") in _scope_cell_info_cache
    assert (large_tree, "single_node_locals") in _scope_cell_info_cache

    # 5. _slice_utf8_line and _slice_unit_token_lines edge cases
    assert _slice_utf8_line("é = foo(1)\n", 4, 4) == ""
    assert _slice_utf8_line("é = foo(1)\n", 5, 2) == "foo(1)\n"
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 4, "end_col": 4},
        ["    [x for x in y]\n"],
    ) == [""]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 4, "end_col": 0},
        ["    [\n", "        x for x in y\n", "    ]\n"],
    ) == ["[\n", "        x for x in y\n", ""]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 50, "end_col": 70},
        ["[x for x in y]"],
    ) == ["[x for x in y]"]

    # 6. `x += <unit>` records post-read and post-mutation of `x`
    src_aug = (
        "def outer(items):\n"
        "    x = 1\n"
        "    x += sum([lambda: x for _ in items][0]() for _ in (1,))\n"
        "    return x\n"
    )
    tree_aug = ast.parse(src_aug)
    info_aug = _analyze_enclosing_fn_cells(
        tree_aug.body[0], (3, 3, 13, 39), mod_tree=tree_aug
    )
    assert "x" in info_aug.post_reads
    assert "x" in info_aug.post_mutated
    sc_aug = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 13,
            "end_col": 39,
            "kind": "comprehension",
            "source_text": src_aug,
            "source_code": "[lambda: x for _ in items]",
        },
        tree=tree_aug,
    )
    assert "severed_closure_cell" in sc_aug["control_flow_hazards"]

    # 7. TryStar / except* definite-binding and closure-cell hazard analysis (Python 3.11+)
    if sys.version_info >= (3, 11):
        src_trystar = (
            "def outer(op, items):\n"
            "    keep = 1\n"
            "    victim = 2\n"
            "    eg = 3\n"
            "    try:\n"
            "        del victim\n"
            "        op()\n"
            "    except* ValueError as eg:\n"
            "        pass\n"
            "    finally:\n"
            "        final_var = 4\n"
            "    cbs = [lambda x=x: (keep, final_var, x) for x in items]\n"
            "    return cbs\n"
        )
        tree_ts = ast.parse(src_trystar)
        fn_ts = tree_ts.body[0]
        assert isinstance(fn_ts, ast.FunctionDef)
        ts_bound, ts_term = _extract_block_definite_bindings_and_term(
            fn_ts.body[:4], None
        )
        assert ts_bound == {"keep", "final_var"}
        assert ts_term is False
        sc_ts_safe = inspect_single_unit_scope(
            {
                "start": 12,
                "end": 12,
                "start_col": 10,
                "end_col": 59,
                "kind": "comprehension",
                "source_text": src_trystar,
                "source_code": "[lambda x=x: (keep, final_var, x) for x in items]",
            },
            tree=tree_ts,
        )
        assert "severed_closure_cell" not in sc_ts_safe["control_flow_hazards"]

        src_trystar_unsafe = src_trystar.replace(
            "(keep, final_var, x)", "(eg, x)"
        )
        sc_ts_unsafe = inspect_single_unit_scope(
            {
                "start": 12,
                "end": 12,
                "start_col": 10,
                "end_col": 46,
                "kind": "comprehension",
                "source_text": src_trystar_unsafe,
                "source_code": "[lambda x=x: (eg, x) for x in items]",
            },
            tree=ast.parse(src_trystar_unsafe),
        )
        assert "severed_closure_cell" in sc_ts_unsafe["control_flow_hazards"]

    # 8. _is_irrefutable_pattern(None), _IntraUnitMutator(NamedTuple), and _scope_reads_outputs_after_line list outputs
    assert _is_irrefutable_pattern(None) is False
    assert _is_irrefutable_case(ast.AST()) is False

    gen_muts = _collect_unit_nonlocal_mutators(
        ast.parse("g = ((w := x) for x in items)\n").body
    )
    assert len(gen_muts) == 1
    assert isinstance(gen_muts[0], _IntraUnitMutator)
    assert isinstance(gen_muts[0].node, ast.GeneratorExp)
    copied_mut = copy.copy(gen_muts[0])
    pickled_mut = pickle.loads(pickle.dumps(gen_muts[0]))
    assert copied_mut.node is gen_muts[0].node
    assert isinstance(pickled_mut.node, ast.GeneratorExp)

    scope_fn_tree = ast.parse(
        "def f():\n"
        "    nonlocal shared\n"
        "    shared = 1\n"
    ).body[0]
    assert (
        _scope_reads_outputs_after_line(
            scope_fn_tree,
            3,
            ["shared"],  # type: ignore[arg-type]
            end_col=14,
            start_line=3,
            start_col=4,
        )
        is True
    )

    # 9. PEP 227 class scope isolation and nested tuple cache eviction
    pep227_tree = ast.parse(
        "def outer():\n"
        "    def inner():\n"
        "        x = 1\n"
        "        class C:\n"
        "            global x\n"
        "            x = 99\n"
        "            def m(self):\n"
        "                nonlocal x\n"
        "                x = 2\n"
        "        return C\n"
        "    return inner\n"
    )
    outer_fn = pep227_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    cls_c = inner_fn.body[1]
    assert isinstance(cls_c, ast.ClassDef)
    cls_muts = _collect_unit_nonlocal_mutators([cls_c], mod_tree=pep227_tree)
    assert any(m.name == "C" and "x" in m.mutated_vars for m in cls_muts)
    assert any(m.name == "m" and "x" in m.mutated_vars for m in cls_muts)
    inner_muts = _collect_unit_nonlocal_mutators([inner_fn], mod_tree=pep227_tree)
    assert not any("x" in m.mutated_vars for m in inner_muts)

    from pydoppelgangerhunt.fixer._df_cache import _evict_tree_scopes

    nested_key = (("deep", (pep227_tree, "nested_tag")), 42)
    _scope_cell_info_cache[nested_key] = "sentinel"
    _evict_tree_scopes(pep227_tree)
    assert nested_key not in _scope_cell_info_cache


def test_builtin_call_loads_target_rebinding_vs_mutation_and_hazard_matrix(
    tmp_path: Path,
) -> None:
    """Covers built-in call load/closure preservation, rebound vs mutated targets, and review matrix."""
    import subprocess

    from pydoppelgangerhunt.fixer import (
        _slice_unit_token_lines,
        clear_dataflow_cache,
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        scope_matches_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import _slice_utf8_line
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _FnScopeContext,
        _SpanBounds,
        _analyze_enclosing_fn_cells,
        _extract_cached_scope_free_reads,
        _extract_closure_carrier_mutation_targets_and_values,
        _extract_closure_carrier_targets_and_values,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_expr_closure_and_load_info,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _extract_mutated_target_names,
        _extract_root_target_names,
        _find_enclosing_function_chain,
        _resolve_unit_col_bounds,
    )

    # 1. Built-in call handling preserves loaded_names, inline_reads, inline_muts, and called_names_out
    len_expr = ast.parse("len(items)", mode="eval").body
    called_unshadowed: set[str] = set()
    r_len, m_len, l_len = _extract_expr_closure_and_load_info(
        [len_expr], called_names_out=called_unshadowed
    )
    assert r_len == set()
    assert m_len == set()
    assert l_len == {"items"}
    assert called_unshadowed == set()

    nested_builtin_expr = ast.parse(
        "any((lambda: captured)() for _ in seq) or sum([len(other)], start=init_cb())",
        mode="eval",
    ).body
    called_nested: set[str] = set()
    r_nb, m_nb, l_nb = _extract_expr_closure_and_load_info(
        [nested_builtin_expr], called_names_out=called_nested
    )
    assert "captured" in r_nb
    assert m_nb == set()
    assert {"seq", "other"} <= l_nb
    assert called_nested == {"init_cb"}

    shadowed_ctx = _FnScopeContext(
        fn_locals=frozenset({"len"}),
        parent_map={},
        shadowed_names=frozenset({"len"}),
        has_star_import=False,
    )
    called_shadowed: set[str] = set()
    _, _, l_shad = _extract_expr_closure_and_load_info(
        [len_expr], shadowed_ctx, called_names_out=called_shadowed
    )
    assert l_shad == {"items"}
    assert called_shadowed == {"len"}

    # 2. Rebound target names vs mutated attribute/subscript target names
    stmt_rebind = ast.parse("cb = make_callback()\n").body[0]
    stmt_destruct = ast.parse("first, (second, *rest) = make_callbacks()\n").body[0]
    stmt_attr = ast.parse("obj.callback = make_callback()\n").body[0]
    stmt_sub = ast.parse("mapping[key] = make_callback()\n").body[0]
    stmt_mixed = ast.parse("bound_var, obj.callback = f1, f2\n").body[0]
    assert isinstance(stmt_rebind, ast.Assign)
    assert isinstance(stmt_destruct, ast.Assign)
    assert isinstance(stmt_attr, ast.Assign)
    assert isinstance(stmt_sub, ast.Assign)
    assert isinstance(stmt_mixed, ast.Assign)

    assert _extract_root_target_names(stmt_rebind.targets[0]) == {"cb"}
    assert _extract_mutated_target_names(stmt_rebind.targets[0]) == set()
    assert _extract_root_target_names(stmt_destruct.targets[0]) == {
        "first",
        "second",
        "rest",
    }
    assert _extract_mutated_target_names(stmt_destruct.targets[0]) == set()
    assert _extract_root_target_names(stmt_attr.targets[0]) == set()
    assert _extract_mutated_target_names(stmt_attr.targets[0]) == {"obj"}
    assert _extract_root_target_names(stmt_sub.targets[0]) == set()
    assert _extract_mutated_target_names(stmt_sub.targets[0]) == {"mapping"}
    assert _extract_root_target_names(stmt_mixed.targets[0]) == {"bound_var"}
    assert _extract_mutated_target_names(stmt_mixed.targets[0]) == {"obj"}

    assert _extract_closure_carrier_targets_and_values(stmt_attr) is None
    assert _extract_closure_carrier_targets_and_values(stmt_sub) is None
    attr_mut_carrier = _extract_closure_carrier_mutation_targets_and_values(
        stmt_attr
    )
    sub_mut_carrier = _extract_closure_carrier_mutation_targets_and_values(
        stmt_sub
    )
    assert attr_mut_carrier is not None and attr_mut_carrier[0] == {"obj"}
    assert sub_mut_carrier is not None and sub_mut_carrier[0] == {"mapping"}

    alias_src = (
        "def outer(obj, mapping, key, source_cb):\n"
        "    obj.callback = source_cb\n"
        "    mapping[key] = source_cb\n"
        "    direct_alias = source_cb\n"
        "    left_a, *right_b = source_cb, source_cb\n"
        "    return direct_alias, left_a, right_b\n"
    )
    alias_tree = ast.parse(alias_src)
    alias_info = _analyze_enclosing_fn_cells(
        alias_tree.body[0], (6, 6, None, None), mod_tree=alias_tree
    )
    alias_dict = dict(alias_info.alias_edges)
    assert "obj" not in alias_dict
    assert "mapping" not in alias_dict
    assert alias_dict.get("direct_alias") == frozenset({"source_cb"})
    assert alias_dict.get("left_a") == frozenset({"source_cb"})
    assert alias_dict.get("right_b") == frozenset({"source_cb"})

    # 3. Paired positive (rebind captured cell) vs negative (mutate referenced object) hazard checks
    src_rebind_captured = (
        "def outer(items):\n"
        "    value = 1\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    value = 2\n"
        "    return cbs\n"
    )
    sc_rebind = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_rebind_captured,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert "severed_closure_cell" in sc_rebind["control_flow_hazards"]

    src_mutate_captured_obj = (
        "def outer(items):\n"
        "    value = []\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: len(value) + x)\n"
        "    value.append(2)\n"
        "    return cbs\n"
    )
    sc_mutate_obj = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_mutate_captured_obj,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: len(value) + x)\n",
        }
    )
    assert "severed_closure_cell" not in sc_mutate_obj["control_flow_hazards"]
    assert sc_mutate_obj["is_control_flow_safe"] is True

    # 4. Cache eviction across NamedTuple/frozenset keys, statement nodes, and defensive copy immutability
    cache_tree = ast.parse(
        "def outer(x):\n"
        "    def inner():\n"
        "        nonlocal x\n"
        "        x += 1\n"
        "        return x\n"
        "    return inner\n"
    )
    outer_node = cache_tree.body[0]
    assert isinstance(outer_node, ast.FunctionDef)
    inner_node = outer_node.body[0]
    assert isinstance(inner_node, ast.FunctionDef)

    mut_s, del_s = _extract_closure_nonlocal_mutations_and_deletions(inner_node)
    mut_s.add("corrupted")
    del_s.add("corrupted")
    mut_s2, del_s2 = _extract_closure_nonlocal_mutations_and_deletions(inner_node)
    assert "corrupted" not in mut_s2
    assert "corrupted" not in del_s2

    free_s = _extract_cached_scope_free_reads(inner_node)
    free_s.add("corrupted")
    assert "corrupted" not in _extract_cached_scope_free_reads(inner_node)

    ret_l, ret_c = _extract_fn_returned_closure_refs(outer_node)
    ret_l.add("corrupted")
    ret_c.add("corrupted")
    ret_l2, ret_c2 = _extract_fn_returned_closure_refs(outer_node)
    assert "corrupted" not in ret_l2
    assert "corrupted" not in ret_c2

    carriers_s = _extract_fn_outer_carrier_targets(outer_node)
    carriers_s.add("corrupted")
    assert "corrupted" not in _extract_fn_outer_carrier_targets(outer_node)

    nt_key = (_SpanBounds(1, 2), frozenset({inner_node.body[0]}))
    _scope_cell_info_cache[nt_key] = "nt_val"
    _evict_tree_scopes(cache_tree)
    assert nt_key not in _scope_cell_info_cache

    # 5. Column bounds resolution: two statements on one line, Unicode prefix, and ambiguous fail-closed
    two_stmt_src = (
        "def f(xs, ys):\n"
        "    a = [x + 1 for x in xs]; b = [y + 2 for y in ys]\n"
        "    return a, b\n"
    )
    two_stmt_tree = ast.parse(two_stmt_src)
    cand_b = ast.parse("[y + 2 for y in ys]\n").body
    s_col_b, e_col_b = _resolve_unit_col_bounds(
        cand_b, two_stmt_tree, 2, 2, None, None
    )
    assert s_col_b == 33
    assert e_col_b == 52

    ambig_src = (
        "def f(xs):\n"
        "    return ([x for x in xs], [x for x in xs])\n"
    )
    ambig_tree = ast.parse(ambig_src)
    cand_ambig = ast.parse("[x for x in xs]\n").body
    s_ambig, e_ambig = _resolve_unit_col_bounds(
        cand_ambig, ambig_tree, 2, 2, None, None
    )
    assert s_ambig is None
    assert e_ambig is None

    dec_src = (
        "@dec1(\n"
        "    arg1\n"
        ")\n"
        "def f(xs):\n"
        "    return [x for x in xs]\n"
    )
    dec_tree = ast.parse(dec_src)
    assert _find_enclosing_function_chain(dec_tree, 2, 2) == []
    assert len(_find_enclosing_function_chain(dec_tree, 5, 5)) == 1

    # 6. Lexical scope: intermediate function declaring `global x` blocks nested `nonlocal x`
    global_barrier_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        global x\n"
        "        x = 2\n"
        "    return mid\n"
    )
    outer_gb = global_barrier_tree.body[0]
    assert isinstance(outer_gb, ast.FunctionDef)
    mid_gb = outer_gb.body[1]
    assert isinstance(mid_gb, ast.FunctionDef)
    # Synthetically attach a child function mutating nonlocal x inside `mid` (which has `global x`)
    synth_inner = ast.parse(
        "def inner():\n    nonlocal x\n    x = 3\n"
    ).body[0]
    mid_gb.body.append(synth_inner)
    clear_dataflow_cache()
    mid_muts, _ = _extract_closure_nonlocal_mutations_and_deletions(mid_gb)
    assert "x" not in mid_muts

    # 7. UTF-8 byte offsets and _slice_unit_token_lines with accented and CJK characters and CRLF
    line_accent = '    result = "é" + value\r\n'
    line_cjk = '    result = "界" + value\r\n'
    tree_accent = ast.parse(line_accent.strip())
    rhs_accent = tree_accent.body[0].value  # type: ignore[attr-defined]
    tree_cjk = ast.parse(line_cjk.strip())
    rhs_cjk = tree_cjk.body[0].value  # type: ignore[attr-defined]
    assert (
        _slice_utf8_line(
            line_accent,
            4 + rhs_accent.col_offset,
            4 + rhs_accent.end_col_offset,
        )
        == '"é" + value'
    )
    assert (
        _slice_utf8_line(
            line_cjk,
            4 + rhs_cjk.col_offset,
            4 + rhs_cjk.end_col_offset,
        )
        == '"界" + value'
    )
    multiline_utf8 = [
        'prefix = "é界" + (\r\n',
        "    value_a + value_b\r\n",
        ') + "é界"\r\n',
    ]
    start_b = len('prefix = "é界" + '.encode("utf-8"))
    end_b = len(")".encode("utf-8"))
    sliced_multi = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": start_b, "end_col": end_b},
        multiline_utf8,
    )
    assert sliced_multi == ["(\r\n", "    value_a + value_b\r\n", ")"]

    # 8 & 9. Closure-strictness mode transitions and hazard propagation across refactoring entry points
    lenient_src = (
        "def outer(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "    return items\n"
    )
    unit_base = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": lenient_src,
        "source_code": "for x in items:\n    acc = x * 2\n",
    }
    sc_strict = ensure_unit_scope_closure_mode(
        unit_base,
        None,
        skip_pre_unit_closures=False,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_strict, False) is True
    assert scope_matches_closure_mode(sc_strict, True) is False
    assert "severed_closure_cell" in sc_strict["control_flow_hazards"]

    sc_lenient = ensure_unit_scope_closure_mode(
        unit_base,
        sc_strict,
        skip_pre_unit_closures=True,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_lenient, True) is True
    assert "severed_closure_cell" not in sc_lenient["control_flow_hazards"]

    sc_back_to_strict = ensure_unit_scope_closure_mode(
        unit_base,
        sc_lenient,
        skip_pre_unit_closures=False,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_back_to_strict, False) is True
    assert "severed_closure_cell" in sc_back_to_strict["control_flow_hazards"]

    # Returned closure hazard cannot be bypassed even when skip_pre_unit_closures=True
    ret_hazard_src = (
        "def outer1(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "        acc += 1\n"
        "    return cbs\n\n"
        "def outer2(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "        acc += 1\n"
        "    return cbs\n"
    )
    ret_file = tmp_path / "ret_hazard.py"
    ret_file.write_text(ret_hazard_src, encoding="utf-8")
    u1_ret = {
        "file": str(ret_file),
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "source_text": ret_hazard_src,
        "source_code": "for x in items:\n    acc = x * 2\n    acc += 1\n",
        "skip_pre_unit_closures": True,
    }
    u2_ret = {
        "file": str(ret_file),
        "start": 10,
        "end": 12,
        "kind": "compound_block",
        "source_text": ret_hazard_src,
        "source_code": "for x in items:\n    acc = x * 2\n    acc += 1\n",
        "skip_pre_unit_closures": True,
    }
    sc_u1_ret = inspect_single_unit_scope(u1_ret, repo_root=str(tmp_path))
    sc_u2_ret = inspect_single_unit_scope(u2_ret, repo_root=str(tmp_path))
    assert resolve_clone_pair_outputs(u1_ret, u2_ret, sc_u1_ret, sc_u2_ret) is None
    assert (
        resolve_clone_generator_subroutine_outputs(
            u1_ret, u2_ret, sc_u1_ret, sc_u2_ret
        )
        is None
    )
    patch_res = generate_refactoring_patch(
        [(1.0, u1_ret, u2_ret)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch_res == ""

    # 10. Differential execution verifying captured mutable side effects and pre/post callback observations
    safe_mut_src = (
        "def run_first(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    before = snap()\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            history.append(x * 3)\n"
        "    after = snap()\n"
        "    return before, after, snap\n\n"
        "def run_second(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    before = snap()\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            history.append(x * 3)\n"
        "    after = snap()\n"
        "    return before, after, snap\n"
    )
    safe_file = tmp_path / "safe_mut.py"
    safe_file.write_text(safe_mut_src, encoding="utf-8")
    (tmp_path / ".git").mkdir(exist_ok=True)
    u1_safe = {
        "file": str(safe_file),
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": safe_mut_src,
        "source_code": "for x in items:\n    if x > 0:\n        history.append(x * 3)\n",
    }
    u2_safe = {
        "file": str(safe_file),
        "start": 15,
        "end": 17,
        "kind": "compound_block",
        "source_text": safe_mut_src,
        "source_code": "for x in items:\n    if x > 0:\n        history.append(x * 3)\n",
    }
    safe_patch = generate_refactoring_patch(
        [(1.0, u1_safe, u2_safe)], repo_root=str(tmp_path), replace_clones=True
    )
    assert safe_patch != ""
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=safe_patch,
        text=True,
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    patched_src = safe_file.read_text(encoding="utf-8")
    ns_orig: dict[str, Any] = {}
    ns_ref: dict[str, Any] = {}
    exec(  # pylint: disable=exec-used
        compile(safe_mut_src, "<orig>", "exec"), ns_orig
    )  # nosec B102
    exec(  # pylint: disable=exec-used
        compile(patched_src, "<ref>", "exec"), ns_ref
    )  # nosec B102
    b_o, a_o, snap_o = ns_orig["run_first"]([-1, 2, 4])
    b_r, a_r, snap_r = ns_ref["run_first"]([-1, 2, 4])
    assert (b_o, a_o, snap_o(), snap_o()) == (b_r, a_r, snap_r(), snap_r())
    assert a_r == [6, 12]

