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
            "        gen = ((seq := list(items)), (x * 2 for x in seq))[1]\n"
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
                "    gen = ((seq := list(items)), (x * 2 for x in seq))[1]\n"
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
            "    _ = [(scale := v) for x in [1] for v in empty_seq]\n"
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
    (
        "pos_complex_expr_walrus_mutating_outer_local_read_after_unit",
        (
            "def outer(a, b):\n"
            "    acc = 0\n"
            "    res = (acc := a + b) * 2\n"
            "    return res + acc\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 28,
            "kind": "complex_expr",
            "source_code": "(acc := a + b) * 2",
        },
        True,
    ),
    (
        "neg_complex_expr_capturing_stable_outer_local",
        (
            "def outer(a, b):\n"
            "    scale = 10\n"
            "    fn_pair = (lambda x: x * scale, a + b)\n"
            "    return fn_pair\n"
        ),
        {
            "start": 3,
            "end": 3,
            "start_col": 14,
            "end_col": 42,
            "kind": "complex_expr",
            "source_code": "(lambda x: x * scale, a + b)",
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
    """Verifies positive/negative severed_closure_cell scenarios and lenient->strict monotonicity."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope

    _ = scenario_id
    compile(src, f"<{scenario_id}>", "exec")
    parsed_tree = ast.parse(src)
    unit = dict(unit_spec)
    unit["source_text"] = src
    scope = inspect_single_unit_scope(unit, tree=parsed_tree)
    if expect_hazard:
        assert "severed_closure_cell" in scope["control_flow_hazards"]
        assert scope["is_control_flow_safe"] is False
    else:
        assert "severed_closure_cell" not in scope["control_flow_hazards"]

    unit_lenient = dict(unit, skip_pre_unit_closures=True)
    scope_lenient = inspect_single_unit_scope(unit_lenient, tree=parsed_tree)
    if "severed_closure_cell" in scope_lenient["control_flow_hazards"]:
        assert "severed_closure_cell" in scope["control_flow_hazards"]


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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_unit_nonlocal_mutators,
        _extract_closure_nonlocal_mutations,
        _extract_expr_scope_free_reads,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _check_captured_cell_mutations,
        _check_intra_unit_nonlocal_mutators,
        _check_sibling_closure_sharing,
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
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
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos

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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
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
    import shutil  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import generate_refactoring_patch

    if shutil.which("git") is None:
        pytest.skip("git executable is required for differential patch application")

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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
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
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
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
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _SpanBounds,
        _strictly_encloses_span,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_fn_outer_carrier_targets,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_comprehension_deferred_closure_info,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
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
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_intra_unit_defs,
        _collect_escaped_intra_unit_names,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _SpanBounds,
        _analyze_enclosing_fn_cells,
        _extract_block_definite_bindings_and_term,
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
    shadow_nodes = set(ast.walk(mod_shadow))
    assert any(
        isinstance(k, _ScopeCacheKey)
        and (k.owner in shadow_nodes or mod_shadow in k.aux_ast_nodes)
        for k in _scope_cell_info_cache
    )
    assert any(
        _key_references_scope_nodes(k, shadow_nodes)
        for k in _scope_cell_info_cache
    )
    _evict_tree_scopes(mod_shadow)
    assert not any(
        isinstance(k, _ScopeCacheKey)
        and (k.owner in shadow_nodes or mod_shadow in k.aux_ast_nodes)
        for k in _scope_cell_info_cache
    )
    assert not any(
        _key_references_scope_nodes(k, shadow_nodes)
        for k in _scope_cell_info_cache
    )
    with pytest.raises(ValueError, match="aux_ast_nodes"):
        _ScopeCacheKey(shared_fn_node, "bad_params", params=(mod_shadow,))
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
        "    _, gen = (w := items), (x + 1 for x in w)\n"
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
    assert any(
        getattr(k, "owner", None) is large_tree
        and getattr(k, "kind", None) == "has_star_import"
        for k in _scope_cell_info_cache
    )
    assert any(
        getattr(k, "owner", None) is large_tree
        and getattr(k, "kind", None) == "single_node_locals"
        for k in _scope_cell_info_cache
    )

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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _SpanBounds,
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
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
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
        "any((lambda: captured)() for _ in seq) or int(str(len(other)), base=init_cb())",
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

    # 6. Lexical scope: intermediate function declaring `global x` is a PEP 3104 SyntaxError
    # when nested `inner` declares `nonlocal x`, and fails closed if constructed synthetically.
    invalid_lexical_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        global x\n"
        "        x = 2\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x = 3\n"
        "    f = lambda: x\n"
        "    return f, mid\n"
    )
    with pytest.raises(SyntaxError):
        compile(invalid_lexical_src, "<invalid_lexical>", "exec")
    global_barrier_tree = ast.parse(invalid_lexical_src)
    outer_gb = global_barrier_tree.body[0]
    assert isinstance(outer_gb, ast.FunctionDef)
    mid_gb = outer_gb.body[1]
    assert isinstance(mid_gb, ast.FunctionDef)
    clear_dataflow_cache()
    mid_muts, _ = _extract_closure_nonlocal_mutations_and_deletions(mid_gb)
    assert "x" in mid_muts
    sc_invalid_lex = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "closure",
            "source_text": invalid_lexical_src,
            "source_code": "f = lambda: x\n",
        },
        tree=global_barrier_tree,
    )
    assert "severed_closure_cell" in sc_invalid_lex["control_flow_hazards"]
    assert sc_invalid_lex["is_control_flow_safe"] is False

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
    import shutil  # pylint: disable=import-outside-toplevel

    if shutil.which("git") is None:
        pytest.skip("git executable is required for patch application")
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


@pytest.mark.parametrize(
    "u_start_col,u_end_col,expected_in_bounds",
    [
        (None, None, [True, True, True]),
        (4, 9, [True, False, False]),
        (11, 20, [False, True, False]),
        (22, 31, [False, False, True]),
    ],
)
def test_is_ast_node_within_unit_bounds_and_visitors_column_filtering(
    u_start_col: Optional[int],
    u_end_col: Optional[int],
    expected_in_bounds: list[bool],
) -> None:
    """Verifies _is_ast_node_within_unit_bounds, _UnitLiveInVisitor, and _DownstreamReadVisitor column bounds."""
    from pydoppelgangerhunt.fixer._df_visitors import (
        _DownstreamReadVisitor,
        _UnitLiveInVisitor,
        _is_ast_node_within_unit_bounds,
    )

    src = "def f(a, b, c):\n    x = a; y = x + b; z = y + c\n"
    fn_node = ast.parse(src).body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    stmts = fn_node.body
    assert len(stmts) == 3

    actual = [
        _is_ast_node_within_unit_bounds(
            s, 2, 2, u_end_col=u_end_col, u_start_col=u_start_col
        )
        for s in stmts
    ]
    assert actual == expected_in_bounds
    assert _is_ast_node_within_unit_bounds(stmts[0], 2, 2) is True

    # Middle statement on line 2: `y = x + b` (col 11..20)
    s_mid = stmts[1]
    mid_start = s_mid.col_offset
    mid_end = s_mid.end_col_offset
    live_in_vis = _UnitLiveInVisitor(
        2, 2, u_end_col=mid_end, u_start_col=mid_start, scope_node=fn_node
    )
    live_in_vis.visit(fn_node)
    assert live_in_vis.live_ins == {"x", "b"}

    ds_vis = _DownstreamReadVisitor(
        2, 2, u_end_col=mid_end, u_start_col=mid_start
    )
    for s in stmts:
        ds_vis.visit(s)
    assert "y" in ds_vis.loaded
    assert "c" in ds_vis.loaded
    assert "a" not in ds_vis.loaded


def test_review_fixes_scope_preservation_carrier_shadowing_and_concurrency() -> None:
    """Covers hand-built scope preservation, returned closure shadowing, escaped defs, and cache concurrency."""
    import threading
    import time
    from types import MappingProxyType

    from pydoppelgangerhunt.fixer import clear_dataflow_cache
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_intra_unit_defs,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
        _build_enclosing_chain_cell_infos,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        ensure_unit_scope_closure_mode,
    )

    # 1. Hand-built scopes and pathological_ast fallback scopes are preserved by ensure_unit_scope_closure_mode
    hand_built: dict[str, Any] = {
        "inputs": ["custom_in"],
        "outputs": ["custom_out"],
        "stores": ["custom_out"],
        "free_vars": ["custom_in"],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
        "control_flow_hazards": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_return": False,
        "local_imports": [],
        "yield_expr_names": [],
    }
    marked_hand_built = {**hand_built, "_skip_pre_unit_closures": True}
    assert (
        ensure_unit_scope_closure_mode(
            {"start": 1, "end": 1}, marked_hand_built, True
        )
        is marked_hand_built
    )
    unmarked_fallback = ensure_unit_scope_closure_mode(
        {"start": 1, "end": 1}, hand_built, True
    )
    assert unmarked_fallback is not hand_built
    assert unmarked_fallback["_skip_pre_unit_closures"] is True
    assert unmarked_fallback["is_control_flow_safe"] is False
    assert "severed_closure_cell" in unmarked_fallback["control_flow_hazards"]
    pathological_scope: dict[str, Any] = {
        **hand_built,
        "_skip_pre_unit_closures": False,
        "control_flow_hazards": ["pathological_ast"],
        "is_control_flow_safe": False,
    }
    assert (
        ensure_unit_scope_closure_mode(
            {"start": 1, "end": 1}, pathological_scope, True
        )
        is pathological_scope
    )
    combined_scope = analyze_unit_variable_scope(
        {"start": 1, "end": 1, "skip_pre_unit_closures": True},
        {"start": 2, "end": 2, "skip_pre_unit_closures": False},
        scope1={**hand_built, "_skip_pre_unit_closures": True},
        scope2={**hand_built, "_skip_pre_unit_closures": False},
    )
    assert isinstance(combined_scope["_skip_pre_unit_closures"], bool)
    assert combined_scope["_skip_pre_unit_closures"] is False

    # 2. _FnScopeContext.parent_map is wrapped in immutable MappingProxyType
    ctx_tree = ast.parse("def f(x):\n    return x + 1\n")
    fn_ctx = _get_fn_scope_context(ctx_tree.body[0], mod_tree=ctx_tree)
    assert isinstance(fn_ctx.parent_map, MappingProxyType)
    with pytest.raises(TypeError):
        fn_ctx.parent_map[ctx_tree] = ctx_tree  # type: ignore[index]

    # 3. _extract_fn_returned_closure_refs respects local parameter shadowing built-in `len`
    shad_ret_tree = ast.parse(
        "def outer(cb):\n"
        "    def helper(len):\n"
        "        return len(cb)\n"
        "    return helper\n"
    )
    outer_fn = shad_ret_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    helper_fn = outer_fn.body[0]
    assert isinstance(helper_fn, ast.FunctionDef)
    outer_ctx = _get_fn_scope_context(outer_fn, mod_tree=shad_ret_tree)
    ret_loads, _ = _extract_fn_returned_closure_refs(helper_fn, outer_ctx)
    assert "cb" in ret_loads

    # 4. _collect_escaped_intra_unit_defs: sibling nonlocal rebind order-independence and global nested defs
    order_a = ast.parse(
        "def target():\n"
        "    return 1\n"
        "def caller():\n"
        "    return target()\n"
        "def rebinder():\n"
        "    nonlocal target\n"
        "    def target():\n"
        "        return 2\n"
        "    return target\n"
        "caller()\n"
        "rebinder()\n"
    ).body
    order_b = ast.parse(
        "def target():\n"
        "    return 1\n"
        "def rebinder():\n"
        "    nonlocal target\n"
        "    def target():\n"
        "        return 2\n"
        "    return target\n"
        "def caller():\n"
        "    return target()\n"
        "caller()\n"
        "rebinder()\n"
    ).body
    names_a, ids_a = _collect_escaped_intra_unit_defs(order_a, frozenset())
    names_b, ids_b = _collect_escaped_intra_unit_defs(order_b, frozenset())
    assert names_a == names_b
    assert "target" in names_a
    assert len(ids_a) == len(ids_b) == 2

    global_def_stmts = ast.parse(
        "def builder():\n"
        "    global exported_fn\n"
        "    def exported_fn():\n"
        "        return 42\n"
        "builder()\n"
    ).body
    g_names, g_ids = _collect_escaped_intra_unit_defs(
        global_def_stmts, frozenset()
    )
    assert "exported_fn" in g_names
    assert len(g_ids) == 1

    # 5. _extract_fn_outer_carrier_targets: class body attribute vs nonlocal & lambda walrus local
    cls_carrier_tree = ast.parse(
        "def outer():\n"
        "    slot = None\n"
        "    lam_slot = None\n"
        "    def worker(y):\n"
        "        nonlocal slot, lam_slot\n"
        "        class LocalAttrOnly:\n"
        "            slot = lambda: y\n"
        "        f = lambda: (lam_slot := (lambda: y))\n"
        "        return LocalAttrOnly, f\n"
    )
    worker_fn = cls_carrier_tree.body[0].body[2]  # type: ignore[attr-defined]
    assert isinstance(worker_fn, ast.FunctionDef)
    assert _extract_fn_outer_carrier_targets(worker_fn) == set()

    cls_nl_tree = ast.parse(
        "def outer():\n"
        "    slot = None\n"
        "    def worker(y):\n"
        "        nonlocal slot\n"
        "        class WithMethodNonlocal:\n"
        "            def set_slot(self):\n"
        "                nonlocal slot\n"
        "                slot = lambda: y\n"
        "        return WithMethodNonlocal\n"
    )
    worker_nl_fn = cls_nl_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert isinstance(worker_nl_fn, ast.FunctionDef)
    assert _extract_fn_outer_carrier_targets(worker_nl_fn) == {"slot"}

    # 6. _unit_cols requires exact match when one endpoint is already provided
    line_dup = ["    x = foo(1); y = foo(1)\n"]
    s_exact, e_exact = _unit_cols(
        {"source_code": "foo(1)", "start_col": 8}, line_dup
    )
    assert (s_exact, e_exact) == (8, 14)

    # 7. Eviction work check on 5,000 composite keys with 300-name frozensets
    clear_dataflow_cache()
    evict_tree = ast.parse("def target_fn(x):\n    return x + 1\n")
    other_tree = ast.parse("def keep_fn(x):\n    return x + 2\n")
    big_names = frozenset(f"var_name_{i}" for i in range(300))
    for i in range(2500):
        _scope_cell_info_cache[
            (evict_tree.body[0], "outer_carriers", big_names, i)
        ] = frozenset()
        _scope_cell_info_cache[
            (other_tree.body[0], "outer_carriers", big_names, i)
        ] = frozenset()
    assert len(_scope_cell_info_cache) == 5000
    _evict_tree_scopes(evict_tree)
    assert len(_scope_cell_info_cache) == 2500
    assert all(k[0] is other_tree.body[0] for k in _scope_cell_info_cache)

    # 8. Multi-threaded concurrency across _analyze_enclosing_fn_cells, _build_enclosing_chain_cell_infos, and eviction
    clear_dataflow_cache()
    conc_src = (
        "def outer(items):\n"
        "    scale = 2\n"
        "    class Box:\n"
        "        def compute(self, val):\n"
        "            cb = lambda x: (x + scale + val)\n"
        "            return [cb(i) for i in items]\n"
        "    for k in items:\n"
        "        tmp = k * scale\n"
        "    return Box, tmp\n"
    )
    conc_tree = ast.parse(conc_src)
    outer_c = conc_tree.body[0]
    assert isinstance(outer_c, ast.FunctionDef)
    box_c = outer_c.body[1]
    assert isinstance(box_c, ast.ClassDef)
    comp_c = box_c.body[0]
    assert isinstance(comp_c, ast.FunctionDef)
    chain = [(1, 9, outer_c), (4, 6, comp_c)]

    def _worker(tid: int) -> bool:
        for step in range(25):
            infos = _build_enclosing_chain_cell_infos(
                chain,
                6,
                6,
                19,
                41,
                frozenset(),
                mod_tree=conc_tree,
            )
            info_loop = _analyze_enclosing_fn_cells(
                outer_c, (8, 8, 8, 23), mod_tree=conc_tree
            )
            if tid == 0 and step % 10 == 0:
                _evict_tree_scopes(ast.parse("def ephemeral(): pass\n"))
            if "scale" not in infos[0].pre_bound or "scale" not in info_loop.pre_bound:
                return False
        return True

    results: list[bool] = [False] * 6
    threads = [
        threading.Thread(target=lambda idx=i: results.__setitem__(idx, _worker(idx)))
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results)


def test_p1_p2_and_deep_correctness_review_matrix(tmp_path: Path) -> None:
    """Comprehensive tests for P1/P2 findings: heterogeneous frozenset cache eviction, fail-closed spans, PEP 3104/227 lexical rules, lazy/eager genexps without scope_ctx, C-enforced scalar built-ins vs custom dunders, carrier mutations vs rebinding, and intermediate _EnclosingFnCellInfo summaries."""
    import symtable

    from pydoppelgangerhunt.fixer import (
        analyze_unit_variable_scope,
        clear_dataflow_cache,
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_comprehension_deferred_closure_info,
        _extract_lazy_genexp_walrus_names,
        _has_invalid_lexical_scope_declarations,
        _has_unresolved_span_bounds,
        _is_lazy_genexp,
        _is_node_inside_span,
        _is_non_closure_scalar_expr,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _UnitClosureHazardContext,
        _has_severed_closure_cell_hazard,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

    # --- P1-1 & P2-2: Heterogeneous frozenset eviction, nested AST nodes, and _ScopeCacheKey ---
    clear_dataflow_cache()
    evict_mod = ast.parse(
        "def outer(items):\n"
        "    def inner(y):\n"
        "        return [ (lambda z: z + y)(x) for x in items ]\n"
        "    return inner\n"
    )
    keep_mod = ast.parse("def survivor(items):\n    return [x for x in items]\n")
    outer_fn = evict_mod.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    ret_stmt = inner_fn.body[0]
    assert isinstance(ret_stmt, ast.Return)
    listcomp_node = ret_stmt.value
    assert isinstance(listcomp_node, ast.ListComp)
    call_node = listcomp_node.elt
    assert isinstance(call_node, ast.Call)
    lambda_node = call_node.func
    assert isinstance(lambda_node, ast.Lambda)

    # Heterogeneous frozensets with string/int metadata and AST nodes in various hash positions
    het_keys = [
        frozenset({"metadata", outer_fn}),
        frozenset({"alpha", "beta", "gamma", 42, inner_fn}),
        ("composite", frozenset({"tag_a", "tag_b", lambda_node})),
        ("nested_comp", (_SpanBounds(1, 4), frozenset({"meta", listcomp_node}))),
        _ScopeCacheKey(
            owner=keep_mod,
            kind="cross_ref",
            params=("param",),
            aux_ast_nodes=(inner_fn,),
        ),
    ]
    keep_key = _ScopeCacheKey(
        owner=keep_mod,
        kind="survivor_key",
        params=(frozenset({"only_strings", "safe"}),),
    )
    for idx, hk in enumerate(het_keys):
        _scope_cell_info_cache[hk] = f"evict_{idx}"
        _scope_closures_cache[hk] = [(1, set())]
    _scope_cell_info_cache[keep_key] = "keep_val"

    tree_nodes = set(ast.walk(evict_mod))
    for hk in het_keys:
        assert _key_references_scope_nodes(hk, tree_nodes) is True
    assert _key_references_scope_nodes(keep_key, tree_nodes) is False

    _evict_tree_scopes(evict_mod)
    for hk in het_keys:
        assert hk not in _scope_cell_info_cache
        assert hk not in _scope_closures_cache
    assert _scope_cell_info_cache.get(keep_key) == "keep_val"

    # Repeated analysis after eviction produces identical results
    info_before = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    _evict_tree_scopes(evict_mod)
    info_after = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    assert info_before == info_after

    # --- P1-2: Fail-closed policy for unresolved source spans & incomplete AST metadata ---
    # 1. Two identical statements on one line without column bounds -> unresolved -> fail closed
    dup_line_src = (
        "def f(xs):\n"
        "    a = 1; a = 1\n"
        "    return [lambda: a for _ in xs]\n"
    )
    dup_line_tree = ast.parse(dup_line_src)
    cand_assign = ast.parse("a = 1\n").body
    assert _resolve_unit_col_bounds(
        cand_assign, dup_line_tree, 2, 2, None, None
    ) == (None, None)
    assert (
        _has_unresolved_span_bounds(
            dup_line_tree.body[0], 2, 2, None, None
        )
        is True
    )
    sc_dup_line = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "subroutine",
            "source_text": dup_line_src,
            "source_code": "a = 1\n",
        },
        tree=dup_line_tree,
    )
    assert "severed_closure_cell" in sc_dup_line["control_flow_hazards"]
    assert sc_dup_line["is_control_flow_safe"] is False

    # 2. Single ast.Expr sub-expression candidate on a non-Expr line does not fall back to full statement
    subexpr_line_src = (
        "def f(xs):\n"
        "    res = [x + 1 for x in xs] + [x + 1 for x in xs]\n"
        "    return res\n"
    )
    subexpr_tree = ast.parse(subexpr_line_src)
    cand_comp_expr = ast.parse("[x + 1 for x in xs]\n").body
    assert _resolve_unit_col_bounds(
        cand_comp_expr, subexpr_tree, 2, 2, None, None
    ) == (None, None)
    sc_ambig_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_text": subexpr_line_src,
            "source_code": "[x + 1 for x in xs]",
        },
        tree=subexpr_tree,
    )
    assert "severed_closure_cell" in sc_ambig_comp["control_flow_hazards"]

    # 3. Incomplete AST metadata (missing end_lineno or end_col_offset) fails closed
    corrupt_tree = ast.parse(
        "def f(xs):\n"
        "    a = 1\n"
        "    b = a + 2\n"
        "    return [lambda: b for _ in xs]\n"
    )
    corrupt_fn = corrupt_tree.body[0]
    assert isinstance(corrupt_fn, ast.FunctionDef)
    corrupt_stmt = corrupt_fn.body[1]
    corrupt_stmt.end_lineno = None
    assert _is_node_inside_span(corrupt_stmt, _SpanBounds(3, 3, 4, 13)) is False
    assert (
        _has_unresolved_span_bounds(corrupt_fn, 3, 3, 4, 13) is True
    )
    assert (
        _has_severed_closure_cell_hazard(
            _UnitClosureHazardContext(
                candidate_stmts=ast.parse("b = a + 2\n").body,
                mod_tree=corrupt_tree,
                u_start=3,
                u_end=3,
                u_start_col=4,
                u_end_col=13,
                is_subroutine=True,
                unit_kind="subroutine",
                inputs=["a"],
                outputs=["b"],
                stores=["b"],
                deleted_names=frozenset(),
                unit_top_locals=frozenset({"b"}),
                cols_resolved=True,
            )
        )
        is True
    )

    corrupt_col_tree = ast.parse(
        "def f(xs):\n"
        "    a = 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    corrupt_col_fn = corrupt_col_tree.body[0]
    assert isinstance(corrupt_col_fn, ast.FunctionDef)
    corrupt_col_fn.body[1].end_col_offset = None
    assert (
        _is_node_inside_span(
            corrupt_col_fn.body[1], _SpanBounds(3, 3, 4, 13)
        )
        is False
    )
    assert _has_unresolved_span_bounds(corrupt_col_fn, 3, 3, 4, 13) is True

    # --- P1-3: PEP 3104 & PEP 227 lexical-scope rules (invalid vs valid multi-level variants) ---
    # Invalid variants rejected by symtable/compile and fail-closed in scope/hazard analysis
    invalid_variants = [
        # Enclosing global blocks nested nonlocal
        "def outer():\n    x = 0\n    def mid():\n        global x\n        def inner():\n            nonlocal x\n            x += 1\n",
        # Same scope nonlocal and global
        "def outer():\n    x = 0\n    def inner():\n        nonlocal x\n        global x\n        x = 1\n",
        # Parameter declared nonlocal
        "def outer():\n    x = 0\n    def inner(x):\n        nonlocal x\n        x = 1\n",
        # Parameter declared global
        "def outer(x):\n    global x\n    x = 1\n",
        # Module-rooted nonlocal with no binding in enclosing function
        "def outer():\n    def inner():\n        nonlocal missing_var\n        missing_var = 1\n",
    ]
    for inv_src in invalid_variants:
        with pytest.raises(SyntaxError):
            symtable.symtable(inv_src, "<sym>", "exec")
        inv_tree = ast.parse(inv_src)
        assert (
            _has_invalid_lexical_scope_declarations(
                inv_tree.body, mod_tree=inv_tree
            )
            is True
        )
        sc_inv = inspect_single_unit_scope(
            {
                "start": 1,
                "end": len(inv_src.splitlines()),
                "kind": "function",
                "source_text": inv_src,
                "source_code": inv_src,
            },
            tree=inv_tree,
        )
        assert "severed_closure_cell" in sc_inv["control_flow_hazards"]
        assert sc_inv["is_control_flow_safe"] is False

    # Valid multi-level variant 1: intermediate function has its own local binding `x = 2`
    valid_mid_local_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        x = 2\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x = 3\n"
        "        inner()\n"
        "        return x\n"
        "    y = x + 10\n"
        "    return y, mid()\n"
    )
    symtable.symtable(valid_mid_local_src, "<valid1>", "exec")
    v1_tree = ast.parse(valid_mid_local_src)
    v1_outer = v1_tree.body[0]
    assert isinstance(v1_outer, ast.FunctionDef)
    v1_mid = v1_outer.body[1]
    assert isinstance(v1_mid, ast.FunctionDef)
    v1_inner = v1_mid.body[1]
    assert isinstance(v1_inner, ast.FunctionDef)
    assert _has_invalid_lexical_scope_declarations(
        v1_tree.body, mod_tree=v1_tree
    ) is False
    assert _extract_closure_nonlocal_mutations_and_deletions(v1_inner) == (
        {"x"},
        set(),
    )
    assert _extract_closure_nonlocal_mutations_and_deletions(v1_mid) == (
        set(),
        set(),
    )
    sc_v1 = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 10,
            "kind": "subroutine",
            "source_text": valid_mid_local_src,
            "source_code": "y = x + 10\n",
        },
        tree=v1_tree,
    )
    assert "severed_closure_cell" not in sc_v1["control_flow_hazards"]

    # Valid multi-level variant 2: intermediate `mid` has `global x`, closer `between` binds local `x = 2`, `inner` has `nonlocal x`
    valid_between_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        global x\n"
        "        def between():\n"
        "            x = 2\n"
        "            def inner():\n"
        "                nonlocal x\n"
        "                x = 3\n"
        "            inner()\n"
        "            return x\n"
        "        return between()\n"
        "    return x, mid()\n"
    )
    symtable.symtable(valid_between_src, "<valid2>", "exec")
    v2_tree = ast.parse(valid_between_src)
    assert (
        _has_invalid_lexical_scope_declarations(v2_tree.body, mod_tree=v2_tree)
        is False
    )

    # Valid multi-level variant 3: chained `nonlocal x` across `mid` and `inner` mutates `outer`'s `x`
    valid_chained_nl_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x += 5\n"
        "        return inner\n"
        "    cb = lambda: x\n"
        "    return cb, mid\n"
    )
    symtable.symtable(valid_chained_nl_src, "<valid3>", "exec")
    v3_tree = ast.parse(valid_chained_nl_src)
    v3_mid = v3_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert _extract_closure_nonlocal_mutations_and_deletions(v3_mid) == (
        {"x"},
        set(),
    )
    sc_v3 = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "closure",
            "source_text": valid_chained_nl_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=v3_tree,
    )
    assert "severed_closure_cell" in sc_v3["control_flow_hazards"]

    # Valid multi-level variant 4 (PEP 227): class body with `global x; x = 99` bypassed by method `nonlocal x`
    valid_cls_global_src = (
        "x = 0\n"
        "def outer():\n"
        "    x = 1\n"
        "    class Mid:\n"
        "        global x\n"
        "        x = 99\n"
        "        def bump(self):\n"
        "            nonlocal x\n"
        "            x += 10\n"
        "    cb = lambda: x\n"
        "    return cb, Mid\n"
    )
    symtable.symtable(valid_cls_global_src, "<valid4>", "exec")
    v4_tree = ast.parse(valid_cls_global_src)
    assert (
        _has_invalid_lexical_scope_declarations(v4_tree.body, mod_tree=v4_tree)
        is False
    )
    v4_cls = v4_tree.body[1].body[1]  # type: ignore[attr-defined]
    assert _extract_closure_nonlocal_mutations_and_deletions(v4_cls) == (
        {"x"},
        set(),
    )
    sc_v4 = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 10,
            "kind": "closure",
            "source_text": valid_cls_global_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=v4_tree,
    )
    assert "severed_closure_cell" in sc_v4["control_flow_hazards"]

    # --- P2-1: _is_lazy_genexp & comprehension helpers with and without scope_ctx ---
    detached_gen = ast.parse("(x + a for x in items)", mode="eval").body
    assert isinstance(detached_gen, ast.GeneratorExp)
    assert _is_lazy_genexp(detached_gen, None) is True

    # Eager consumer inside a comprehension when scope_ctx is omitted (None)
    nested_eager_comp = ast.parse(
        "[sum((w := x) + a for x in row) for row in rows]", mode="eval"
    ).body
    assert isinstance(nested_eager_comp, ast.ListComp)
    assert _extract_lazy_genexp_walrus_names(nested_eager_comp, None) == set()
    c_reads_eager, c_muts_eager = _extract_comprehension_deferred_closure_info(
        nested_eager_comp, None
    )
    assert c_reads_eager == set()
    assert c_muts_eager == set()

    top_eager_sum = ast.parse(
        "sum((w := x) + a for x in items)", mode="eval"
    ).body
    assert _extract_lazy_genexp_walrus_names(top_eager_sum, None) == set()

    # Lazy generator expression inside an outer list comprehension when scope_ctx is None
    nested_lazy_comp = ast.parse(
        "[((w := x) + a for x in row) for row in rows]", mode="eval"
    ).body
    assert isinstance(nested_lazy_comp, ast.ListComp)
    assert _extract_lazy_genexp_walrus_names(nested_lazy_comp, None) == {"w"}
    c_reads_lazy, c_muts_lazy = _extract_comprehension_deferred_closure_info(
        nested_lazy_comp, None
    )
    assert "a" in c_reads_lazy
    assert "w" in c_muts_lazy

    # --- P2-3: C-enforced scalar built-ins vs unconstrained custom dunders ---
    for scalar_code in (
        "len(items)",
        "int(x)",
        "bool(x)",
        "str(x)",
        "ord(c)",
        "isinstance(x, int)",
        "1 + 2",
        "'a' == 'b'",
        "not custom_obj",
        "-len(items)",
        "len(items) + 1",
        "len(items) == 0",
    ):
        expr_node = ast.parse(scalar_code, mode="eval").body
        assert _is_non_closure_scalar_expr(expr_node) is True, scalar_code

    for non_scalar_code in (
        "abs(custom_obj)",
        "round(custom_obj)",
        "divmod(custom_obj, 2)",
        "pow(custom_obj, 2)",
        "sum(items)",
        "-custom_obj",
        "custom_obj + 1",
        "len(items) + custom_obj",
        "custom_obj == other_obj",
    ):
        expr_node = ast.parse(non_scalar_code, mode="eval").body
        assert _is_non_closure_scalar_expr(expr_node) is False, non_scalar_code

    # Custom dunder (__abs__ / __eq__) returning a closure from a factory object is tracked as a carrier
    dunder_factory_src = (
        "def outer(items):\n"
        "    scale = 1\n"
        "    class AbsFactory:\n"
        "        def __abs__(self):\n"
        "            return lambda v: v + scale\n"
        "    fac = AbsFactory()\n"
        "    cb = abs(fac)\n"
        "    for x in items:\n"
        "        scale = x * 2\n"
        "        res = cb(x)\n"
        "    return res\n"
    )
    sc_dunder = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_text": dunder_factory_src,
            "source_code": "for x in items:\n    scale = x * 2\n    res = cb(x)\n",
        }
    )
    assert "severed_closure_cell" in sc_dunder["control_flow_hazards"]

    # --- Sections 2, 3, & 4: Carrier mutation vs rebinding, comprehension iterables, and _EnclosingFnCellInfo ---
    # Mutating attributes/subscripts on a captured object without rebinding the cell is safe
    for mut_stmt in (
        "box.attr = 2",
        "box[0] = 2",
        "box.attr += 2",
        "box[0] += 2",
        "box.custom_method(2)",
    ):
        obj_mut_src = (
            "def outer(items, box):\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: box[0] + x)\n"
            f"    {mut_stmt}\n"
            "    return cbs\n"
        )
        sc_obj_mut = inspect_single_unit_scope(
            {
                "start": 3,
                "end": 4,
                "kind": "compound_block",
                "source_text": obj_mut_src,
                "source_code": "for x in items:\n    cbs.append(lambda x=x: box[0] + x)\n",
            }
        )
        assert "severed_closure_cell" not in sc_obj_mut["control_flow_hazards"], mut_stmt

    # Storing a closure into an attribute or custom method on `holder` tracks `holder` as a closure carrier
    for carrier_stmt in (
        "holder.cb = lambda: scale",
        "holder.register(lambda: scale)",
    ):
        carrier_mut_src = (
            "def outer(items, holder):\n"
            f"    {carrier_stmt}\n"
            "    for x in items:\n"
            "        scale = x + 1\n"
            "    return holder\n"
        )
        sc_carrier_mut = inspect_single_unit_scope(
            {
                "start": 3,
                "end": 4,
                "kind": "compound_block",
                "source_text": carrier_mut_src,
                "source_code": "for x in items:\n    scale = x + 1\n",
            }
        )
        assert "severed_closure_cell" in sc_carrier_mut["control_flow_hazards"], carrier_stmt

    # Outermost comprehension iterable is evaluated eagerly; rebinding `items` after unit is safe
    eager_iter_src = (
        "def outer(items):\n"
        "    cbs = [lambda x=x: x + 1 for x in items]\n"
        "    items = []\n"
        "    return cbs\n"
    )
    sc_eager_iter = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": eager_iter_src,
            "source_code": "[lambda x=x: x + 1 for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_eager_iter["control_flow_hazards"]

    # Intermediate _EnclosingFnCellInfo control-flow summary verification
    cf_summary_src = (
        "def outer(flag, items):\n"
        "    uncond = 1\n"
        "    if flag:\n"
        "        both = 10\n"
        "        only_then = 20\n"
        "    else:\n"
        "        both = 11\n"
        "    try:\n"
        "        may_del = 30\n"
        "    except RuntimeError as err_var:\n"
        "        del may_del\n"
        "    for item in items:\n"
        "        unit_marker = both + uncond\n"
        "        loop_var = item\n"
        "    after_var = unit_marker\n"
        "    return after_var, loop_var\n"
    )
    cf_tree = ast.parse(cf_summary_src)
    cf_info = _analyze_enclosing_fn_cells(
        cf_tree.body[0], (13, 13, 8, 35), mod_tree=cf_tree
    )
    assert {"flag", "items", "uncond", "both", "item"} <= cf_info.pre_bound
    assert "only_then" not in cf_info.pre_bound
    assert "only_then" in cf_info.pre_mutated
    assert "may_del" not in cf_info.pre_bound
    assert "err_var" in cf_info.pre_unbound
    assert "loop_var" in cf_info.loop_mutated
    assert "after_var" in cf_info.post_mutated
    assert "unit_marker" in cf_info.post_reads

    # End-to-end consistency across analyze_unit_variable_scope, resolve_clone_*, and generate_refactoring_patch
    e2e_hazard_src = (
        "def f1(items):\n"
        "    cb = lambda v: v + scale\n"
        "    for x in items:\n"
        "        scale = x + 1\n"
        "    return cb, scale\n\n"
        "def f2(items):\n"
        "    cb = lambda v: v + scale\n"
        "    for x in items:\n"
        "        scale = x + 1\n"
        "    return cb, scale\n"
    )
    e2e_file = tmp_path / "e2e_hazard.py"
    e2e_file.write_text(e2e_hazard_src, encoding="utf-8")
    u1_e2e = {
        "file": str(e2e_file),
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": e2e_hazard_src,
        "source_code": "for x in items:\n    scale = x + 1\n",
    }
    u2_e2e = {
        "file": str(e2e_file),
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": e2e_hazard_src,
        "source_code": "for x in items:\n    scale = x + 1\n",
    }
    sc1_e2e = inspect_single_unit_scope(u1_e2e, repo_root=str(tmp_path))
    sc2_e2e = inspect_single_unit_scope(u2_e2e, repo_root=str(tmp_path))
    comb_e2e = analyze_unit_variable_scope(
        u1_e2e, u2_e2e, repo_root=str(tmp_path), scope1=sc1_e2e, scope2=sc2_e2e
    )
    assert "severed_closure_cell" in comb_e2e["control_flow_hazards"]
    assert comb_e2e["is_control_flow_safe"] is False
    res_pair = resolve_clone_pair_outputs(
        u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True
    )
    assert res_pair.outputs is None
    assert res_pair.skip_reason == "severed closure cell"
    res_gen = resolve_clone_generator_subroutine_outputs(
        u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True
    )
    assert res_gen.outputs is None
    assert res_gen.skip_reason == "severed closure cell"
    assert (
        generate_refactoring_patch(
            [(1.0, u1_e2e, u2_e2e)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )


def test_unmarked_scope_recomputation_slice_truncation_and_sections_2_1_to_2_6(
    tmp_path: Any,
) -> None:
    """Covers unmarked precomputed scopes, multiline slice end_col truncation, and review sections 2.1-2.6."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (
        _is_irrefutable_case,
        _is_irrefutable_pattern,
        clear_dataflow_cache,
        inspect_single_unit_scope,
        resolve_clone_generator_subroutine_outputs,
        synthesize_shared_helper_code,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _propagate_sibling_closures,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _is_eagerly_consumed_genexp,
    )
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        scope_matches_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    clear_dataflow_cache()

    # [P1] Unmarked scopes never match strict or lenient mode and are recomputed when unit has source
    gen_src = (
        "def gen_a(items):\n"
        "    pre_cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return len(items)\n\n"
        "def gen_b(items):\n"
        "    pre_cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return len(items)\n"
    )
    gen_file = tmp_path / "unmarked_modes.py"
    gen_file.write_text(gen_src, encoding="utf-8")
    u_g1 = {
        "file": str(gen_file),
        "start": 3,
        "end": 5,
        "name": "gen_a:For",
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    u_g2 = {
        "file": str(gen_file),
        "start": 10,
        "end": 12,
        "name": "gen_b:For",
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }

    strict_sc1 = inspect_single_unit_scope(u_g1, repo_root=str(tmp_path))
    strict_sc2 = inspect_single_unit_scope(u_g2, repo_root=str(tmp_path))
    assert scope_matches_closure_mode(strict_sc1, False) is True
    assert scope_matches_closure_mode(strict_sc1, True) is False

    unmarked_strict1 = dict(strict_sc1)
    unmarked_strict2 = dict(strict_sc2)
    unmarked_strict1.pop("_skip_pre_unit_closures", None)
    unmarked_strict2.pop("_skip_pre_unit_closures", None)
    assert scope_matches_closure_mode(unmarked_strict1, False) is False
    assert scope_matches_closure_mode(unmarked_strict1, True) is False
    assert scope_matches_closure_mode(
        {**unmarked_strict1, "_skip_pre_unit_closures": None}, False
    ) is False

    # Passing unmarked strict scopes into lenient mode recomputes to lenient
    recomputed_lenient = ensure_unit_scope_closure_mode(
        u_g1, unmarked_strict1, True, repo_root=str(tmp_path)
    )
    assert recomputed_lenient["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in recomputed_lenient["control_flow_hazards"]
    res_gen_lenient = resolve_clone_generator_subroutine_outputs(
        u_g1,
        u_g2,
        unmarked_strict1,
        unmarked_strict2,
        repo_root=str(tmp_path),
        closure_strictness="lenient",
        return_details=True,
    )
    assert res_gen_lenient.outputs is not None
    unmarked_comb_strict = analyze_unit_variable_scope(
        u_g1, u_g2, repo_root=str(tmp_path)
    )
    unmarked_comb_strict.pop("_skip_pre_unit_closures", None)
    synth_from_unmarked_strict = synthesize_shared_helper_code(
        u_g1,
        u_g2,
        repo_root=str(tmp_path),
        scope1=unmarked_strict1,
        scope2=unmarked_strict2,
        scope=unmarked_comb_strict,
        closure_strictness="lenient",
    )
    assert "def " in synth_from_unmarked_strict

    # Passing unmarked lenient scopes into strict mode recomputes to strict
    lenient_sc1 = inspect_single_unit_scope(
        {**u_g1, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    lenient_sc2 = inspect_single_unit_scope(
        {**u_g2, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    unmarked_lenient1 = dict(lenient_sc1)
    unmarked_lenient2 = dict(lenient_sc2)
    unmarked_lenient1.pop("_skip_pre_unit_closures", None)
    unmarked_lenient2.pop("_skip_pre_unit_closures", None)
    recomputed_strict = ensure_unit_scope_closure_mode(
        u_g1, unmarked_lenient1, False, repo_root=str(tmp_path)
    )
    assert recomputed_strict["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in recomputed_strict["control_flow_hazards"]
    res_gen_strict = resolve_clone_generator_subroutine_outputs(
        u_g1,
        u_g2,
        unmarked_lenient1,
        unmarked_lenient2,
        repo_root=str(tmp_path),
        closure_strictness="strict",
        return_details=True,
    )
    assert res_gen_strict.outputs is None
    assert res_gen_strict.skip_reason == "severed closure cell"
    unmarked_comb_lenient = analyze_unit_variable_scope(
        {**u_g1, "skip_pre_unit_closures": True},
        {**u_g2, "skip_pre_unit_closures": True},
        repo_root=str(tmp_path),
    )
    unmarked_comb_lenient.pop("_skip_pre_unit_closures", None)
    assert (
        synthesize_shared_helper_code(
            u_g1,
            u_g2,
            repo_root=str(tmp_path),
            scope1=unmarked_lenient1,
            scope2=unmarked_lenient2,
            scope=unmarked_comb_lenient,
            closure_strictness="strict",
        )
        == ""
    )

    # [P2] _slice_unit_token_lines truncates end_col even when line 0 is pre-sliced or empty
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 24, "end_col": 5},
        ["[\n", "        x * 2 for x in items\n", "    ] + [99]\n"],
    ) == ["[\n", "        x * 2 for x in items\n", "    ]"]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 5},
        ["\n", "        x * 2 for x in items\n", "    ] + [99]\n"],
    ) == ["\n", "        x * 2 for x in items\n", "    ]"]
    utf8_last = '    ] + "é界🚀"\n'
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 30, "end_col": 5},
        ["[\n", "    x for x in items\n", utf8_last],
    ) == ["[\n", "    x for x in items\n", "    ]"]
    utf8_empty_slice = "é = "
    utf8_len = len(utf8_empty_slice.encode("utf-8"))
    assert _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": utf8_len, "end_col": utf8_len},
        [utf8_empty_slice],
    ) == [""]

    # Section 2.1: Closure mutation propagation through classes
    cls_prop_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    z = 3\n"
        "    w = 4\n"
        "    class OuterCls:\n"
        "        x = 99\n"
        "        nonlocal z\n"
        "        z = 30\n"
        "        global g_var\n"
        "        g_var = 100\n"
        "        if True:\n"
        "            def branch_m(self):\n"
        "                nonlocal y\n"
        "                y += 1\n"
        "        class InnerCls:\n"
        "            def deep_m(self):\n"
        "                class MethodCls:\n"
        "                    def leaf(self):\n"
        "                        nonlocal x, w\n"
        "                        x += 1\n"
        "                        del w\n"
        "                return MethodCls\n"
    )
    outer_cls_node = cls_prop_tree.body[0].body[4]  # type: ignore[attr-defined]
    muts_2_1, dels_2_1 = _extract_closure_nonlocal_mutations_and_deletions(
        outer_cls_node
    )
    assert muts_2_1 == {"x", "y", "z", "w"}
    assert dels_2_1 == {"w"}
    assert "g_var" not in muts_2_1

    # Class attribute assignment without nonlocal does not report nonlocal mutation
    cls_shadow_only = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    class ShadowOnly:\n"
        "        x = 2\n"
        "        def read_attr(self):\n"
        "            return self.x\n"
    )
    assert _extract_closure_nonlocal_mutations_and_deletions(
        cls_shadow_only.body[0].body[1]  # type: ignore[attr-defined]
    ) == (set(), set())

    # Section 2.2: Comprehension and generator-expression evaluation timing
    lazy_walrus_src = (
        "def f1(items):\n"
        "    x = 0\n"
        "    gen = ((x := x + item) for item in items)\n"
        "    x = 100\n"
        "    return list(gen)\n"
    )
    sc_lazy_walrus = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "kind": "compound_block",
            "source_text": lazy_walrus_src,
            "source_code": "gen = ((x := x + item) for item in items)\n",
        }
    )
    assert "severed_closure_cell" in sc_lazy_walrus["control_flow_hazards"]
    ns_lazy: dict[str, Any] = {}
    exec(lazy_walrus_src, ns_lazy)  # pylint: disable=exec-used
    assert ns_lazy["f1"]([1, 2, 3]) == [101, 103, 106]

    comp_target_shadow_src = (
        "def f1(items):\n"
        "    x = 0\n"
        "    for step in items:\n"
        "        x += step\n"
        "    funcs = [(lambda: x) for x in items]\n"
        "    return x, [fn() for fn in funcs]\n\n"
        "def f2(items):\n"
        "    x = 0\n"
        "    for step in items:\n"
        "        x += step\n"
        "    funcs = [(lambda: x) for x in items]\n"
        "    return x, [fn() for fn in funcs]\n"
    )
    u_cts1 = {
        "start": 2,
        "end": 4,
        "kind": "compound_block",
        "source_text": comp_target_shadow_src,
        "source_code": "x = 0\nfor step in items:\n    x += step\n",
    }
    sc_cts1 = inspect_single_unit_scope(u_cts1)
    assert "severed_closure_cell" not in sc_cts1["control_flow_hazards"]

    shadowed_sum_tree = ast.parse(
        "def outer(items, custom_sum):\n"
        "    sum = custom_sum\n"
        "    return sum(x for x in items)\n"
    )
    sum_fn_node = shadowed_sum_tree.body[0]
    sum_ctx = _get_fn_scope_context(sum_fn_node, mod_tree=shadowed_sum_tree)
    genexp_node = sum_fn_node.body[1].value.args[0]  # type: ignore[attr-defined]
    assert (
        _is_eagerly_consumed_genexp(
            genexp_node,
            sum_ctx.parent_map,
            set(sum_ctx.shadowed_names),
            sum_ctx.has_star_import,
        )
        is False
    )

    # Section 2.3: Precision of _is_closure_factory_scope for ClassDef and AsyncFunctionDef
    factory_prec_tree = ast.parse(
        "def outer(x):\n"
        "    class PureRecord:\n"
        "        a = 1\n"
        "    class PureService:\n"
        "        def compute(self, v):\n"
        "            return v * 2\n"
        "    class CapturingCls:\n"
        "        def get(self):\n"
        "            return x\n"
        "    class MutatingCls:\n"
        "        def inc(self):\n"
        "            nonlocal x\n"
        "            x += 1\n"
        "    class FactoryCls:\n"
        "        def make(self, a):\n"
        "            return lambda: a\n"
        "    class CarrierCls:\n"
        "        def __init__(self):\n"
        "            reg.append(lambda: x)\n"
        "    async def pure_async(a, b):\n"
        "        return a + b\n"
        "    async def capturing_async():\n"
        "        return x\n"
        "    async def mutating_async():\n"
        "        nonlocal x\n"
        "        x += 1\n"
        "    async def async_gen():\n"
        "        yield 1\n"
    )
    f_prec_fn = factory_prec_tree.body[0]
    f_prec_ctx = _get_fn_scope_context(f_prec_fn, mod_tree=factory_prec_tree)
    body_nodes = {
        getattr(n, "name", ""): n for n in f_prec_fn.body  # type: ignore[attr-defined]
    }
    assert _is_closure_factory_scope(body_nodes["PureRecord"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["PureService"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["CapturingCls"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["MutatingCls"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["FactoryCls"], f_prec_ctx) is True
    assert _extract_fn_outer_carrier_targets(
        body_nodes["CarrierCls"], frozenset(), f_prec_ctx
    ) == {"reg"}
    assert _is_closure_factory_scope(body_nodes["pure_async"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["capturing_async"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["mutating_async"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["async_gen"], f_prec_ctx) is True

    # Section 2.4: Transitive sibling-closure propagation (cyclic aliases, rebound aliases, factory chains)
    cyclic_closures: list[tuple[str | None, set[str], set[str]]] = [
        ("seed_cb", {"x"}, {"y"}),
    ]
    cyclic_edges: list[tuple[str, set[str]]] = [
        ("alias_a", {"seed_cb", "alias_b"}),
        ("alias_b", {"alias_a"}),
        ("alias_c", {"alias_b"}),
    ]
    post_reads_cyc: set[str] = {"alias_c"}
    prop_cyc = _propagate_sibling_closures(
        cyclic_closures, post_reads=post_reads_cyc, alias_edges=cyclic_edges
    )
    by_name = {name: (r, m) for name, r, m in prop_cyc if name is not None}
    assert by_name["alias_a"] == ({"x"}, {"y"})
    assert by_name["alias_b"] == ({"x"}, {"y"})
    assert by_name["alias_c"] == ({"x"}, {"y"})
    assert "x" in post_reads_cyc

    chain_factory_tree = ast.parse(
        "def outer(flag):\n"
        "    x = 1\n"
        "    def raw_cb():\n"
        "        return x\n"
        "    def f1():\n"
        "        return raw_cb\n"
        "    def f2():\n"
        "        return f1()\n"
        "    picked = f2() if flag else raw_cb\n"
        "    x = 2\n"
        "    return picked()\n"
    )
    cf_info_2_4 = _analyze_enclosing_fn_cells(
        chain_factory_tree.body[0],
        _SpanBounds(10, 10),
        unit_mutations=frozenset({"x"}),
        mod_tree=chain_factory_tree,
    )
    assert "x" in cf_info_2_4.post_reads

    # Section 2.5: Exception-handler implicit deletion and pattern-binding analysis
    assert _is_irrefutable_pattern(None) is False
    if sys.version_info >= (3, 10):
        match_tree_2_5 = ast.parse(
            "match subj:\n"
            "    case _ if cond:\n"
            "        a = 1\n"
            "    case [only]:\n"
            "        a = 2\n"
            "    case 1 | _:\n"
            "        a = 3\n"
        )
        m_cases = match_tree_2_5.body[0].cases  # type: ignore[attr-defined]
        assert _is_irrefutable_case(m_cases[0]) is False
        assert _is_irrefutable_case(m_cases[1]) is False
        assert _is_irrefutable_case(m_cases[2]) is True

    except_del_input_src = (
        "def f(err):\n"
        "    try:\n"
        "        x = err + 1\n"
        "    except Exception as err:\n"
        "        x = 0\n"
        "    return x\n"
    )
    sc_except_in = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "kind": "compound_block",
            "source_text": except_del_input_src,
            "source_code": "try:\n    x = err + 1\nexcept Exception as err:\n    x = 0\n",
        }
    )
    assert "err" in sc_except_in["inputs"]
    assert "deleted_output" in sc_except_in["control_flow_hazards"]
    assert sc_except_in["is_control_flow_safe"] is False

    try_finally_del_tree = ast.parse(
        "def outer():\n"
        "    surviving = 1\n"
        "    deleted_in_handler = 2\n"
        "    rebound_in_finally = 3\n"
        "    try:\n"
        "        risky()\n"
        "        only_in_try = 4\n"
        "        raise RuntimeError()\n"
        "    except Exception as rebound_in_finally:\n"
        "        del deleted_in_handler\n"
        "        raise\n"
        "    finally:\n"
        "        rebound_in_finally = 30\n"
        "    target = 0\n"
    )
    tf_info = _analyze_enclosing_fn_cells(
        try_finally_del_tree.body[0],
        _SpanBounds(14, 14),
        mod_tree=try_finally_del_tree,
    )
    assert "surviving" in tf_info.pre_bound
    assert "rebound_in_finally" in tf_info.pre_bound
    assert "deleted_in_handler" not in tf_info.pre_bound
    assert "only_in_try" not in tf_info.pre_bound

    # Section 2.6: Concurrent closure and suspension analysis (generators and async coroutines)
    async_coro_src = (
        "async def run_task(items):\n"
        "    x = 1\n"
        "    async def async_reader():\n"
        "        return x\n"
        "    coro = async_reader()\n"
        "    for item in items:\n"
        "        x = x + item\n"
        "        res = await coro\n"
        "    return res\n"
    )
    sc_async_coro = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": async_coro_src,
            "source_code": "for item in items:\n    x = x + item\n    res = await coro\n",
        }
    )
    assert "severed_closure_cell" in sc_async_coro["control_flow_hazards"]

    pure_async_src = (
        "async def run_pure(items):\n"
        "    x = 1\n"
        "    async def pure_helper(v):\n"
        "        return v + 1\n"
        "    coro = pure_helper(10)\n"
        "    for item in items:\n"
        "        x = x + item\n"
        "        res = await coro\n"
        "    return x + res\n"
    )
    sc_pure_async = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": pure_async_src,
            "source_code": "for item in items:\n    x = x + item\n    res = await coro\n",
        }
    )
    assert "severed_closure_cell" not in sc_pure_async["control_flow_hazards"]

    multi_gen_src = (
        "def dual_generators(items):\n"
        "    shared = 0\n"
        "    def gen_peer():\n"
        "        nonlocal shared\n"
        "        shared += 10\n"
        "        yield shared\n"
        "    peer = gen_peer()\n"
        "    for item in items:\n"
        "        shared += item\n"
        "        yield shared\n"
        "    return next(peer)\n"
    )
    sc_multi_gen = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_text": multi_gen_src,
            "source_code": "for item in items:\n    shared += item\n    yield shared\n",
        }
    )
    assert "severed_closure_cell" in sc_multi_gen["control_flow_hazards"]


def test_intra_unit_global_escapes_detached_shadowing_suspended_local_closures_and_review_areas(
    tmp_path: Any,
) -> None:
    """Covers [P1] intra-unit global escapes, [P1] detached unit enclosing local shadowing,
    [P2] suspended units with local closures, [P2] unmarked minimal precomputed scopes,
    and review areas A-E.
    """
    from pydoppelgangerhunt.fixer import (
        analyze_unit_variable_scope,
        clear_dataflow_cache,
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_unit_nonlocal_mutators,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures_uncached,
        _extract_scope_local_names,
        _has_escaped_intra_unit_global_declarations,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        ensure_unit_scope_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    clear_dataflow_cache()

    # --- [P1] #1: Global declarations inside extracted unit escaping to outer statements ---
    global_escape_assign_src = (
        "x = 0\n"
        "def process():\n"
        "    global x\n"
        "    x = 1\n"
        "    x = 2\n"
        "    return x\n"
    )
    ge_tree = ast.parse(global_escape_assign_src)
    assert (
        _has_escaped_intra_unit_global_declarations(
            ge_tree.body[1], 3, 4
        )
        is True
    )
    sc_global_escape = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "basic_block",
            "source_text": global_escape_assign_src,
            "source_code": "global x\nx = 1\n",
        },
        tree=ge_tree,
    )
    assert "severed_closure_cell" in sc_global_escape["control_flow_hazards"]
    assert sc_global_escape["is_control_flow_safe"] is False

    # Pre-unit statement + in-unit global + post-unit write/read -> escaping global hazard
    global_escape_pre_local_src = (
        "x = 0\n"
        "def process():\n"
        "    a = 1\n"
        "    global x\n"
        "    y = x + a\n"
        "    x = 2\n"
        "    return x, y\n"
    )
    sc_global_pre = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "basic_block",
            "source_text": global_escape_pre_local_src,
            "source_code": "global x\ny = x + a\n",
        }
    )
    assert "severed_closure_cell" in sc_global_pre["control_flow_hazards"]

    # Enclosing local `x = 0` shadowed if `def middle` loses `global x` when `return x` remains outside unit
    global_escape_enclosing_cell_src = (
        "x = 100\n"
        "def outer():\n"
        "    x = 5\n"
        "    def middle():\n"
        "        global x\n"
        "        y = 1\n"
        "        return x + y\n"
        "    return middle\n"
    )
    sc_global_enc_cell = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "basic_block",
            "source_text": global_escape_enclosing_cell_src,
            "source_code": "global x\ny = 1\n",
        }
    )
    assert "severed_closure_cell" in sc_global_enc_cell["control_flow_hazards"]

    # Safe case 1: `global x` is ONLY used inside the extracted unit, never touched/read outside
    global_self_contained_src = (
        "x = 0\n"
        "def process(a):\n"
        "    global x\n"
        "    x = a + 1\n"
        "    res = a * 2\n"
        "    return res\n"
    )
    sc_global_self = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "basic_block",
            "source_text": global_self_contained_src,
            "source_code": "global x\nx = a + 1\n",
        }
    )
    assert "severed_closure_cell" not in sc_global_self["control_flow_hazards"]

    # Safe case 2: `global x` is also declared outside the unit (`outer_globals`), and unit only reads `x`
    global_redundant_read_src = (
        "x = 10\n"
        "def process(a):\n"
        "    global x\n"
        "    global x\n"
        "    y = x + a\n"
        "    x = 20\n"
        "    return y\n"
    )
    sc_global_redundant = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "basic_block",
            "source_text": global_redundant_read_src,
            "source_code": "global x\ny = x + a\n",
        }
    )
    assert "severed_closure_cell" not in sc_global_redundant["control_flow_hazards"]

    # --- [P1] #2: Detached unit modules retain enclosing function locals for shadowed builtins ---
    detached_shadow_src = (
        "def outer(items, custom_list):\n"
        "    list = custom_list\n"
        "    value = 1\n"
        "    seq = list(value + item for item in items)\n"
        "    value = 2\n"
        "    return seq\n"
    )
    ds_tree = ast.parse(detached_shadow_src)
    ds_outer = ds_tree.body[0]
    assert isinstance(ds_outer, ast.FunctionDef)
    detached_mod = ast.Module(body=[ds_outer.body[2]], type_ignores=[])
    assert "list" in _extract_scope_local_names(
        detached_mod, module_tree=ds_tree
    )
    detached_closures = _collect_scope_closures_uncached(
        detached_mod,
        module_tree=ds_tree,
        genexp_include_header=False,
        cache_parent_map=False,
    )
    assert any("value" in reads for _, reads in detached_closures)
    sc_detached_shadow = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": detached_shadow_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=ds_tree,
    )
    assert "severed_closure_cell" in sc_detached_shadow["control_flow_hazards"]

    # Detached unit with walrus inside genexp passed to shadowed `list`
    detached_walrus_shadow_src = (
        "def outer(items, custom_list):\n"
        "    list = custom_list\n"
        "    w = 0\n"
        "    seq = list((w := item) for item in items)\n"
        "    return seq, w\n"
    )
    dw_tree = ast.parse(detached_walrus_shadow_src)
    dw_outer = dw_tree.body[0]
    assert isinstance(dw_outer, ast.FunctionDef)
    dw_muts = _collect_unit_nonlocal_mutators(
        [dw_outer.body[2]], mod_tree=dw_tree
    )
    assert any("w" in m.mutated_vars for m in dw_muts)
    sc_detached_walrus = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": detached_walrus_shadow_src,
            "source_code": "seq = list((w := item) for item in items)\n",
        },
        tree=dw_tree,
    )
    assert "severed_closure_cell" in sc_detached_walrus["control_flow_hazards"]

    # PEP 227 class scope isolation: class-body `list = custom_list` applies directly inside ClassDef body,
    # NOT inside a method `def m(self, items)`!
    cls_isolation_src = (
        "def outer(custom_list):\n"
        "    class Holder:\n"
        "        list = custom_list\n"
        "        def compute(self, items):\n"
        "            value = 1\n"
        "            seq = list(value + item for item in items)\n"
        "            value = 2\n"
        "            return seq, value\n"
        "    return Holder\n"
    )
    ci_tree = ast.parse(cls_isolation_src)
    sc_cls_method_safe = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "basic_block",
            "source_text": cls_isolation_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=ci_tree,
    )
    assert "severed_closure_cell" not in sc_cls_method_safe["control_flow_hazards"]

    # Directly inside ClassDef body (not inside a method), `list = custom_list` DOES shadow `list`
    cls_body_shadow_src = (
        "def outer(items, custom_list):\n"
        "    value = 1\n"
        "    class Holder:\n"
        "        list = custom_list\n"
        "        seq = list(value + item for item in items)\n"
        "    value = 2\n"
        "    return Holder\n"
    )
    cbs_tree = ast.parse(cls_body_shadow_src)
    sc_cls_body_shadow = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": cls_body_shadow_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=cbs_tree,
    )
    assert "severed_closure_cell" in sc_cls_body_shadow["control_flow_hazards"]

    # --- [P2] #3: Suspended units allow purely unit-local nonlocal mutators while rejecting escaping ones ---
    suspended_local_closure_src = (
        "def worker_a(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n\n"
        "def worker_b(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n"
    )
    slc_tree = ast.parse(suspended_local_closure_src)
    u_slc1 = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": suspended_local_closure_src,
        "source_code": (
            "for item in items:\n"
            "    value = item\n"
            "    def increment():\n"
            "        nonlocal value\n"
            "        value += 1\n"
            "    increment()\n"
            "    yield value\n"
        ),
    }
    sc_slc1 = inspect_single_unit_scope(u_slc1, tree=slc_tree)
    assert "severed_closure_cell" not in sc_slc1["control_flow_hazards"]
    assert sc_slc1["is_control_flow_safe"] is True

    # Suspended unit where inner closure mutates a variable read after the unit IS rejected
    suspended_escaping_var_src = (
        "def worker(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n"
        "    return value\n"
    )
    sc_susp_escape = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 8,
            "kind": "compound_block",
            "source_text": suspended_escaping_var_src,
            "source_code": u_slc1["source_code"],
        }
    )
    assert "severed_closure_cell" in sc_susp_escape["control_flow_hazards"]

    # Suspended unit where inner closure mutates an outer-scope variable IS rejected
    suspended_outer_var_src = (
        "def worker(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        def add_item():\n"
        "            nonlocal total\n"
        "            total += item\n"
        "        add_item()\n"
        "        yield item\n"
    )
    sc_susp_outer = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 8,
            "kind": "compound_block",
            "source_text": suspended_outer_var_src,
            "source_code": (
                "for item in items:\n"
                "    def add_item():\n"
                "        nonlocal total\n"
                "        total += item\n"
                "    add_item()\n"
                "    yield item\n"
            ),
        }
    )
    assert "severed_closure_cell" in sc_susp_outer["control_flow_hazards"]

    # --- [P2] #4: Minimal unmarked precomputed scopes are recomputed when tree/source is available ---
    min_src = (
        "def outer(items):\n"
        "    cb = lambda: y\n"
        "    for x in items:\n"
        "        y = x + 1\n"
        "    return cb()\n"
    )
    min_tree = ast.parse(min_src)
    min_unit = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": min_src,
        "source_code": "for x in items:\n    y = x + 1\n",
    }
    minimal_unmarked_scope = {"inputs": ["items"], "outputs": ["y"]}
    recomputed_min = ensure_unit_scope_closure_mode(
        min_unit, minimal_unmarked_scope, False, tree=min_tree
    )
    assert "_skip_pre_unit_closures" in recomputed_min
    assert recomputed_min["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in recomputed_min["control_flow_hazards"]

    # --- Review Area E: Exception-handler termination & unexported conditional rebinding ---
    # 1. Terminating handler (`except Exception as e: return 0`) does NOT demote pre-try definite binding `e = 1`
    term_handler_stmts = ast.parse(
        "e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    return 0\n"
    ).body
    def_th: set[str] = set()
    cond_th: set[str] = set()
    _analyze_block_assignment(term_handler_stmts, def_th, cond_th)
    assert "e" in def_th
    assert "e" not in cond_th

    # 2. Non-terminating handler (`except Exception as e: x = 0`) DOES demote pre-try `e = 1` to conditional
    nonterm_handler_stmts = ast.parse(
        "e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    x = 0\n"
    ).body
    def_nth: set[str] = set()
    cond_nth: set[str] = set()
    _analyze_block_assignment(nonterm_handler_stmts, def_nth, cond_nth)
    assert "e" not in def_nth
    assert "e" in cond_nth

    # 3. Handler that deletes and rebinds `v` inside handler body (or rebinds `e` in finally) preserves definite binding,
    # whereas `e = 99` inside `except Exception as e:` is still cleared by Python's implicit handler-exit `del e`.
    rebind_handler_stmts = ast.parse(
        "v = 1\n"
        "e = 1\n"
        "rebound_e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    del v\n"
        "    v = 99\n"
        "    e = 99\n"
        "except RuntimeError as rebound_e:\n"
        "    x = 3\n"
        "finally:\n"
        "    rebound_e = 100\n"
    ).body
    def_rh: set[str] = set()
    cond_rh: set[str] = set()
    _analyze_block_assignment(rebind_handler_stmts, def_rh, cond_rh)
    assert "v" in def_rh
    assert "v" not in cond_rh
    assert "rebound_e" in def_rh
    assert "e" not in def_rh
    assert "e" in cond_rh

    # 4. In-unit `except Exception as e:` where `e` was pre-bound before unit and read after unit -> hazard
    except_clobber_prebound_src = (
        "def f(a):\n"
        "    e = a + 1\n"
        "    try:\n"
        "        res = a * 2\n"
        "    except Exception as e:\n"
        "        res = 0\n"
        "    return res + e\n"
    )
    sc_except_clobber = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_text": except_clobber_prebound_src,
            "source_code": (
                "try:\n"
                "    res = a * 2\n"
                "except Exception as e:\n"
                "    res = 0\n"
            ),
        }
    )
    assert "severed_closure_cell" in sc_except_clobber["control_flow_hazards"]
    assert sc_except_clobber["is_control_flow_safe"] is False

    # --- Review Area D: UTF-8 slicing edge cases (CRLF, astral emoji, CJK, accented) ---
    astral_line = '    msg = "🚀✨café日本語"; val = (a + b) * c\r\n'
    val_expr = "(a + b) * c"
    s_utf8 = len('    msg = "🚀✨café日本語"; val = '.encode("utf-8"))
    e_utf8 = s_utf8 + len(val_expr.encode("utf-8"))
    assert _slice_utf8_line(astral_line, s_utf8, e_utf8) == val_expr
    assert _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": s_utf8, "end_col": e_utf8},
        [astral_line],
    ) == [val_expr]


def test_multilevel_concurrent_sibling_with_definite_bindings_and_class_escape() -> None:
    """Covers P1 (multi-level concurrent sibling), P2 (with definite bindings, class escape, context cache), and Areas A-E."""
    import sys
    from types import MappingProxyType
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cache import (
        _get_scope_parent_map,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _collect_escaped_and_executed_intra_unit_defs,
        _collect_unit_nonlocal_mutators,
        _extract_closure_nonlocal_mutations_and_deletions_uncached,
        _extract_fn_outer_carrier_targets,
        _extract_fn_outer_carrier_targets_uncached,
        _extract_fn_returned_closure_refs,
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
        _is_closure_factory_scope,
        _propagate_sibling_closures,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _check_sibling_closure_sharing,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _build_enclosing_chain_cell_infos,
        _extract_block_definite_bindings_and_term,
        _find_enclosing_chain_definite_pre_bound,
    )

    # =========================================================================
    # 1. [P1] Multi-level concurrent sibling detection (outer -> middle -> inner -> generator)
    # =========================================================================
    p1_src = (
        "def outer():\n"
        "    value = 0\n"
        "    def sibling():\n"
        "        return value\n"
        "    def middle():\n"
        "        def inner():\n"
        "            def generator():\n"
        "                nonlocal value\n"
        "                value += 1\n"
        "                yield value\n"
        "            return generator()\n"
        "        return inner()\n"
        "    return middle(), sibling()\n"
    )
    p1_tree = ast.parse(p1_src)
    p1_unit = {
        "start": 9,
        "end": 10,
        "kind": "linear_block",
        "source_text": p1_src,
        "source_code": "value += 1\nyield value\n",
    }
    p1_scope = inspect_single_unit_scope(p1_unit, tree=p1_tree)
    assert "severed_closure_cell" in p1_scope["control_flow_hazards"]
    assert p1_scope["is_control_flow_safe"] is False
    p1_chain = _find_enclosing_function_chain(p1_tree, 9, 10)
    assert len(p1_chain) == 4  # outer -> middle -> inner -> generator
    p1_chain_infos = _build_enclosing_chain_cell_infos(
        p1_chain, 9, 10, None, None, frozenset({"value"}), mod_tree=p1_tree
    )
    assert (
        _check_sibling_closure_sharing(
            frozenset(),
            frozenset({"value"}),
            p1_chain_infos,
            is_suspended=True,
        )
        is True
    )

    # Shadowed case: middle defines its own local `value = 10` without any sibling closure in middle/inner
    p1_shadowed_src = (
        "def outer():\n"
        "    value = 0\n"
        "    def sibling():\n"
        "        return value\n"
        "    def middle():\n"
        "        value = 10\n"
        "        for step in (1, 2):\n"
        "            value += step\n"
        "            yield value\n"
        "    return list(middle()), sibling()\n"
    )
    p1_shadowed_tree = ast.parse(p1_shadowed_src)
    p1_shadowed_unit = {
        "start": 7,
        "end": 9,
        "kind": "compound_block",
        "source_text": p1_shadowed_src,
        "source_code": (
            "for step in (1, 2):\n"
            "    value += step\n"
            "    yield value\n"
        ),
    }
    p1_shadowed_scope = inspect_single_unit_scope(
        p1_shadowed_unit, tree=p1_shadowed_tree
    )
    assert "severed_closure_cell" not in p1_shadowed_scope["control_flow_hazards"]
    p1_sh_chain = _find_enclosing_function_chain(p1_shadowed_tree, 7, 9)
    p1_sh_infos = _build_enclosing_chain_cell_infos(
        p1_sh_chain, 7, 9, None, None, frozenset({"value"}), mod_tree=p1_shadowed_tree
    )
    assert (
        _check_sibling_closure_sharing(
            frozenset(),
            frozenset({"value"}),
            p1_sh_infos,
            is_suspended=True,
        )
        is False
    )

    # =========================================================================
    # 2. [P2] Definite-binding analysis inside `with` and `async with` bodies
    # =========================================================================
    # 2a. Unconditional assignment in pre-unit `with` populates definite pre_bound
    with_definite_src = (
        "def process(cm, items):\n"
        "    with cm() as resource:\n"
        "        scale = resource.factor\n"
        "    out = [x * scale for x in items]\n"
        "    return out\n"
    )
    with_def_tree = ast.parse(with_definite_src)
    with_def_chain = _find_enclosing_function_chain(with_def_tree, 4, 4)
    pre_bound_with = _find_enclosing_chain_definite_pre_bound(
        with_def_chain, 4, 4, mod_tree=with_def_tree
    )
    assert {"cm", "items", "resource", "scale"} <= pre_bound_with
    sc_with_def = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "linear_block",
            "source_text": with_definite_src,
            "source_code": "out = [x * scale for x in items]\n",
        },
        tree=with_def_tree,
    )
    assert "severed_closure_cell" not in sc_with_def["control_flow_hazards"]

    # 2b. Conditional assignment inside `with`, multiple items, return/break/continue, delete, async with
    with_cases: list[tuple[str, set[str], bool]] = [
        (
            "with cm1() as a, cm2() as b:\n    c = a + b\n",
            {"a", "b", "c"},
            False,
        ),
        (
            "with cm():\n    if cond:\n        c = 1\n",
            set(),
            False,
        ),
        (
            "with cm():\n    if cond:\n        c = 1\n    else:\n        c = 2\n",
            {"c"},
            False,
        ),
        (
            "with cm():\n    if cond:\n        return 0\n    c = 1\n",
            {"c"},
            False,
        ),
        (
            "with cm():\n    c = 1\n    return c\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    break\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    continue\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    del c\n",
            set(),
            False,
        ),
        (
            "c = 1\nwith cm():\n    del c\n",
            set(),
            False,
        ),
        (
            "async def _af():\n    async with acm() as a:\n        b = a + 1\n",
            {"a", "b"},
            False,
        ),
        (
            "with suppress(KeyError):\n    c = 1\n",
            {"c"},
            False,
        ),
        (
            "with contextlib.suppress(KeyError) as sup:\n    c = 1\n",
            {"sup", "c"},
            False,
        ),
        (
            "with suppress(KeyError):\n    c = d['k']\n",
            set(),
            False,
        ),
    ]
    for code_str, expected_bound, expected_term in with_cases:
        parsed = ast.parse(code_str)
        stmts = (
            parsed.body[0].body
            if isinstance(parsed.body[0], ast.AsyncFunctionDef)
            else parsed.body
        )
        got_bound, got_term = _extract_block_definite_bindings_and_term(
            stmts, None
        )
        assert got_bound == expected_bound, f"Mismatch for:\n{code_str}"
        assert got_term is expected_term, f"Term mismatch for:\n{code_str}"

    # =========================================================================
    # 3. [P2] Class definitions escape classification (Cases 1-5)
    # =========================================================================
    # Case 1: Unreferenced local class with uncalled method mutating unit-local `count`
    cls_unreferenced_src = (
        "def outer(items):\n"
        "    def reader():\n"
        "        return count\n"
        "    count = 0\n"
        "    class LocalHelper:\n"
        "        def mutate(self):\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "    res = count + len(items)\n"
        "    return count, res, reader()\n"
    )
    cls_unref_tree = ast.parse(cls_unreferenced_src)
    cls_unref_unit = {
        "start": 4,
        "end": 9,
        "kind": "linear_block",
        "source_text": cls_unreferenced_src,
        "source_code": (
            "count = 0\n"
            "class LocalHelper:\n"
            "    def mutate(self):\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "res = count + len(items)\n"
        ),
    }
    muts_unref = _collect_unit_nonlocal_mutators(
        ast.parse(str(cls_unref_unit["source_code"])).body
    )
    assert any(
        m.name == "LocalHelper" and m.is_escaped_def is False
        for m in muts_unref
    )
    sc_cls_unref = inspect_single_unit_scope(
        cls_unref_unit, tree=cls_unref_tree
    )
    assert "severed_closure_cell" not in sc_cls_unref["control_flow_hazards"]

    # Cases 2-5: Escaping/executed class definitions MUST be classified as escaped mutators
    escaping_cls_tails = [
        # Case 2: Assigned to an escaping binding read after the unit
        ("escaped = LocalHelper\n", "    return count, escaped\n"),
        # Case 3a: Passed to an unknown function call
        ("register(LocalHelper)\n", "    return count\n"),
        # Case 3b: Instantiated inside the unit
        ("inst = LocalHelper()\n", "    return count\n"),
        # Case 4: Referenced by an escaping closure
        ("cb = lambda: LocalHelper\n", "    return count, cb\n"),
    ]
    for extra_stmt, ret_line in escaping_cls_tails:
        esc_src = (
            "def outer(items, register):\n"
            "    count = 0\n"
            "    class LocalHelper:\n"
            "        def mutate(self):\n"
            "            nonlocal count\n"
            "            count += 1\n"
            f"    {extra_stmt}"
            f"{ret_line}"
        )
        esc_unit_code = (
            "count = 0\n"
            "class LocalHelper:\n"
            "    def mutate(self):\n"
            "        nonlocal count\n"
            "        count += 1\n"
            f"{extra_stmt}"
        )
        sc_esc = inspect_single_unit_scope(
            {
                "start": 2,
                "end": 7,
                "kind": "linear_block",
                "source_text": esc_src,
                "source_code": esc_unit_code,
            },
            tree=ast.parse(esc_src),
        )
        assert "severed_closure_cell" in sc_esc["control_flow_hazards"], (
            f"Expected hazard for: {extra_stmt}"
        )

    # Case 5: Decorated class, base class, keyword metaclass, or direct class-body nonlocal mutation
    for cls_header_or_body in [
        "@dec\nclass LocalHelper:\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(Base):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(metaclass=Meta):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper:\n    nonlocal count\n    count += 1\n",
    ]:
        c5_stmts = ast.parse("count = 0\n" + cls_header_or_body).body
        c5_muts = _collect_unit_nonlocal_mutators(c5_stmts)
        assert any(
            m.name == "LocalHelper" and m.is_escaped_def is True
            for m in c5_muts
        )
        _, c5_esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(
            c5_stmts, frozenset()
        )
        assert any(
            isinstance(s, ast.ClassDef) and id(s) in c5_esc_ids
            for s in c5_stmts
        )

    # =========================================================================
    # 4. [P2] Cached closure metadata context validation on shared AST nodes
    # =========================================================================
    clear_dataflow_cache()
    shared_fn_tree = ast.parse(
        "def factory(items):\n"
        "    res = any(x + outer_val for x in items)\n"
        "    box.append(any(x + outer_val for x in items))\n"
        "    return any(cb_ref for x in items)\n"
    )
    shared_fn = shared_fn_tree.body[0]
    assert isinstance(shared_fn, ast.FunctionDef)
    parent_map_shared = MappingProxyType(_get_scope_parent_map(shared_fn_tree))
    owner_a = ast.Module(body=[], type_ignores=[])
    owner_b = ast.Module(body=[], type_ignores=[])
    # Context A: `any` is unshadowed -> `any(...)` is eagerly consumed and scalar
    ctx_a = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items"}),
        has_star_import=False,
        scope_owner=owner_a,
    )
    # Context B: `any` is shadowed -> `any(...)` is an unknown call with a lazy genexp carrier!
    ctx_b = _FnScopeContext(
        fn_locals=frozenset({"items", "any"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items", "any"}),
        has_star_import=False,
        scope_owner=owner_b,
    )
    # Context C: same shadowed_names as A, but has_star_import=True -> genexp is lazy!
    ctx_c = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items"}),
        has_star_import=True,
        scope_owner=owner_a,
    )
    assert _is_closure_factory_scope(shared_fn, ctx_a) is False
    assert _is_closure_factory_scope(shared_fn, ctx_b) is True
    assert _is_closure_factory_scope(shared_fn, ctx_c) is True
    ctx_detached = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset({"items"}),
        has_star_import=False,
        scope_owner=owner_a,
    )
    assert _is_closure_factory_scope(shared_fn, ctx_detached) is False
    assert _extract_fn_outer_carrier_targets(shared_fn, frozenset(), ctx_a) == set()
    assert _extract_fn_outer_carrier_targets(shared_fn, frozenset(), ctx_b) == {"box"}
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_a) == (set(), set())
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_b) == (
        set(),
        {"any"},
    )

    # =========================================================================
    # 5. [P2] Table-driven definite-binding runtime oracle comparison & Areas A-E
    # =========================================================================
    oracle_cases: list[tuple[str, dict[str, object], set[str]]] = [
        (
            "if flag:\n    a = 1\n    b = 2\nelse:\n    a = 3\n    c = 4\n",
            {"flag": True},
            {"a"},
        ),
        (
            "try:\n    a = 1\nexcept ValueError:\n    a = 2\nfinally:\n    f = 3\n",
            {},
            {"a", "f"},
        ),
        (
            "if (w := val) and (u := w + 1):\n    z = 1\nelse:\n    z = 2\n",
            {"val": 0},
            {"w", "z"},
        ),
        (
            "for i in (it := [1, 2]):\n    loop_v = i\n",
            {},
            {"it"},
        ),
    ]
    if sys.version_info >= (3, 10):
        oracle_cases.append(
            (
                "match val:\n    case 1:\n        m = 10\n    case _:\n        m = 20\n",
                {"val": 5},
                {"m"},
            )
        )
    if sys.version_info >= (3, 11):
        oracle_cases.append(
            (
                "try:\n    a = 1\nexcept* ValueError:\n    a = 2\nfinally:\n    f = 3\n",
                {},
                {"f"},
            )
        )
    for code_str, env_init, expected_def in oracle_cases:
        stmts = ast.parse(code_str).body
        static_bound, _ = _extract_block_definite_bindings_and_term(stmts, None)
        assert static_bound == expected_def
        ns: dict[str, object] = dict(env_init)
        exec(compile(code_str, "<oracle>", "exec"), {}, ns)  # noqa: S102
        assert static_bound <= set(ns.keys())

    # --- Area A: _has_invalid_lexical_scope_declarations & class scopes ---
    area_a_cases: list[tuple[str, bool]] = [
        # Valid: class attribute `value = 2` is bypassed by nested method's `nonlocal value`
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Example:\n"
            "        value = 2\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            False,
        ),
        # Valid: class-level `global value` does not block method `nonlocal value` when `outer` binds `value`
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Example:\n"
            "        global value\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            False,
        ),
        # Invalid: `outer` declares `global value`, so method `nonlocal value` has no lexical cell
        (
            "def outer():\n"
            "    global value\n"
            "    class Example:\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            True,
        ),
        # Valid: nested classes `OuterCls -> InnerCls -> mutate` resolving `nonlocal value` to `outer`
        (
            "def outer():\n"
            "    value = 1\n"
            "    class OuterCls:\n"
            "        class InnerCls:\n"
            "            def mutate(self):\n"
            "                nonlocal value\n"
            "                value = 4\n",
            False,
        ),
        # Invalid: intervening function `mid` inside class declares `global value`
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Cls:\n"
            "        def mid(self):\n"
            "            global value\n"
            "            def inner():\n"
            "                nonlocal value\n"
            "                value = 5\n",
            True,
        ),
    ]
    for src_a, expected_invalid in area_a_cases:
        tree_a = ast.parse(src_a)
        got_invalid = _has_invalid_lexical_scope_declarations(
            tree_a.body, mod_tree=tree_a
        )
        assert got_invalid is expected_invalid
        try:
            compile(src_a, "<area_a>", "exec")
            compiled_ok = True
        except SyntaxError:
            compiled_ok = False
        assert compiled_ok is (not expected_invalid)

    # --- Area B: _extract_closure_nonlocal_mutations_and_deletions_uncached rebinding matrix ---
    area_b_fn = ast.parse(
        "def holder():\n"
        "    nonlocal a, b, c, d, e, f, g, h\n"
        "    a = 1\n"
        "    del b\n"
        "    _ = [x := 1 for c in (1, 2) if (d := c)]\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        pass\n"
        "    class Sub:\n"
        "        h = 99\n"
        "        def sub_m(self):\n"
        "            nonlocal h\n"
        "            h = 100\n"
    ).body[0]
    assert isinstance(area_b_fn, ast.FunctionDef)
    if sys.version_info >= (3, 10):
        area_b_fn.body.append(
            ast.parse("match a:\n    case f:\n        pass\n").body[0]
        )
    else:
        synth_match_as_cls = type(
            "MatchAs", (ast.stmt,), {"_fields": ("pattern", "name")}
        )
        area_b_fn.body.append(synth_match_as_cls(pattern=None, name="f"))
    if sys.version_info >= (3, 12):
        area_b_fn.body.append(ast.parse("type g = int\n").body[0])
    else:
        synth_ta_cls = type(
            "TypeAlias",
            (ast.stmt,),
            {"_fields": ("name", "type_params", "value")},
        )
        area_b_fn.body.append(
            synth_ta_cls(
                name=ast.Name(id="g", ctx=ast.Store()),
                type_params=[],
                value=ast.Name(id="int", ctx=ast.Load()),
            )
        )
    b_muts, b_dels = _extract_closure_nonlocal_mutations_and_deletions_uncached(
        area_b_fn
    )
    assert b_muts == frozenset({"a", "b", "d", "e", "f", "g", "h"})
    assert b_dels == frozenset({"b", "e"})

    # --- Area C: _propagate_sibling_closures edge cases ---
    prop_res = _propagate_sibling_closures(
        [
            ("self_fn", frozenset({"self_fn", "x"}), frozenset({"m1"})),
            ("mut_a", frozenset({"mut_b", "y"}), frozenset()),
            ("mut_b", frozenset({"mut_a", "z"}), frozenset({"m2"})),
            ("dup", frozenset({"d1"}), frozenset()),
            ("dup", frozenset({"d2"}), frozenset({"dm"})),
        ],
        alias_edges=[
            ("alias1", frozenset({"mut_a"})),
            ("alias2", frozenset({"alias1"})),
            ("alias1", frozenset({"alias2"})),
        ],
    )
    prop_dict = {name: (r, m) for name, r, m in prop_res if name is not None}
    assert prop_dict["self_fn"] == ({"self_fn", "x"}, {"m1"})
    assert prop_dict["mut_a"] == ({"mut_a", "mut_b", "y", "z"}, {"m2"})
    assert prop_dict["alias2"] == ({"mut_a", "mut_b", "y", "z"}, {"m2"})
    assert prop_dict["dup"] == ({"d1", "d2"}, {"dm"})

    # --- Area D: _find_enclosing_function_chain and _resolve_unit_col_bounds ---
    area_d_src = (
        "@dec\n"
        "def outer(a=lambda: 1):\n"
        "    def single_line(): x = 1; y = 2; return x + y\n"
        "    return single_line()\n"
    )
    area_d_tree = ast.parse(area_d_src)
    # Span inside decorator/header (line 1 or line 2 default arg) is NOT enclosed by outer body
    assert _find_enclosing_function_chain(area_d_tree, 1, 1) == []
    # Span on line 3 inside single_line body
    chain_d = _find_enclosing_function_chain(
        area_d_tree, 3, 3, u_start_col=23, u_end_col=35
    )
    assert [getattr(n, "name", "") for _, _, n in chain_d] == [
        "outer",
        "single_line",
    ]
    outer_fn_d = area_d_tree.body[0]
    assert isinstance(outer_fn_d, ast.FunctionDef)
    single_fn_d = outer_fn_d.body[0]
    assert isinstance(single_fn_d, ast.FunctionDef)
    cand_d = [single_fn_d.body[0]]
    assert _resolve_unit_col_bounds(cand_d, area_d_tree, 3, 3, None, None) == (
        23,
        28,
    )

    # --- Area E: _extract_fn_outer_carrier_targets_uncached ---
    area_e_fn = ast.parse(
        "def carrier_fn():\n"
        "    global glob_box\n"
        "    nonlocal nl_var\n"
        "    local_list = []\n"
        "    local_list.append(lambda: captured)\n"
        "    glob_box.append(lambda: captured)\n"
        "    outer_Attr.cb = lambda: captured\n"
        "    outer_sub[0] = lambda: captured\n"
        "    outer_nested.items[0].append(lambda: captured)\n"
        "    fn_ref = lambda: captured\n"
        "    fn_alias = fn_ref\n"
        "    outer_via_alias.register(fn_alias)\n"
        "    nl_var = fn_alias\n"
    ).body[0]
    assert isinstance(area_e_fn, ast.FunctionDef)
    e_targets = _extract_fn_outer_carrier_targets_uncached(area_e_fn)
    assert e_targets == frozenset(
        {
            "outer_Attr",
            "outer_sub",
            "outer_nested",
            "outer_via_alias",
            "nl_var",
        }
    )


def test_review_round_pep227_trydel_definite_equivalence_and_cache_regressions() -> None:
    """Verifies PEP 227 class-level global isolation, try-del-return handler fallthrough,
    definite-binding engine equivalence, statement-delta equivalence, and cache isolation.
    """
    import ast  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
        clear_dataflow_cache,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        _scope_cell_info_cache,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (  # pylint: disable=import-outside-toplevel
        _SpanBounds,
        _extract_cached_scope_free_reads,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _extract_fn_outer_carrier_targets_uncached,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _is_self_escaping_def_node,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (  # pylint: disable=import-outside-toplevel
        _analyze_enclosing_fn_cells_uncached,
        _extract_block_definite_bindings_and_term,
        _iter_class_body_methods,
    )
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=import-outside-toplevel
        _ClassScopeVisitor,
        _extract_nested_scope_free_reads_with_nodes,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        _analyze_block_assignment,
        ensure_unit_scope_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _slice_unit_token_lines,
    )

    # --- Test A: Class-level `global` does not propagate into nested methods/lambdas/comprehensions/classes ---
    pep227_src = (
        "def outer(xs):\n"
        "    items = []\n"
        "    cell_v = 10\n"
        "    class C:\n"
        "        global items, cell_v\n"
        "        direct_read = cell_v\n"
        "        comp_read = [cell_v + x for x in xs]\n"
        "        lam_read = lambda self: cell_v\n"
        "        def register(self, cb):\n"
        "            items.append(cb)\n"
        "        def capture_and_store(self):\n"
        "            items.append(lambda: cell_v)\n"
        "        class Inner:\n"
        "            def use(self):\n"
        "                return cell_v\n"
        "    c = C()\n"
        "    c.register(lambda: cell_v)\n"
        "    cell_v = 20\n"
        "    return c, items\n"
    )
    compile(pep227_src, "<pep227_src>", "exec")
    pep227_tree = ast.parse(pep227_src)
    outer_fn = pep227_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    cls_c = outer_fn.body[2]
    assert isinstance(cls_c, ast.ClassDef)
    outer_ctx = _get_fn_scope_context(outer_fn, mod_tree=pep227_tree)
    c_targets = _extract_fn_outer_carrier_targets_uncached(
        cls_c, frozenset(), outer_ctx
    )
    assert "items" in c_targets
    cls_vis = _ClassScopeVisitor()
    for s_node in cls_c.body:
        cls_vis.visit(s_node)
    assert "cell_v" in cls_vis.globals
    assert "cell_v" in cls_vis.nested_free_reads
    nested_reads, _ = _extract_nested_scope_free_reads_with_nodes(cls_c)
    assert "cell_v" in nested_reads

    # Sibling method carrier inside enclosing class body (`enclosing_cls=True`)
    cls_sib_carrier_src = (
        "def outer(xs):\n"
        "    box = []\n"
        "    val = 1\n"
        "    class Handler:\n"
        "        global box\n"
        "        def save_cb(self):\n"
        "            box.append(lambda: val)\n"
        "        def work(self):\n"
        "            nonlocal val\n"
        "            for x in xs:\n"
        "                val += x + box[0]()\n"
        "            return val\n"
        "    return Handler, box\n"
    )
    compile(cls_sib_carrier_src, "<cls_sib_carrier>", "exec")
    sc_cls_sib = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 11,
            "kind": "compound_block",
            "source_text": cls_sib_carrier_src,
            "source_code": "for x in xs:\n    val += x + box[0]()\n",
        },
        tree=ast.parse(cls_sib_carrier_src),
    )
    assert "severed_closure_cell" in sc_cls_sib["control_flow_hazards"]

    # --- Test B: Unknown strictness mode without source fails closed ---
    unmarked_precomputed: dict[str, Any] = {
        "inputs": ["a"],
        "outputs": ["b"],
        "stores": ["b"],
        "definite_stores": ["b"],
        "free_vars": ["a"],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
        "control_flow_hazards": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_return": False,
        "local_imports": [],
        "yield_expr_names": [],
    }
    strict_unknown_res = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        unmarked_precomputed,
        skip_pre_unit_closures=False,
    )
    assert strict_unknown_res["is_control_flow_safe"] is False
    assert "severed_closure_cell" in strict_unknown_res["control_flow_hazards"]
    assert strict_unknown_res["_skip_pre_unit_closures"] is False

    # --- Test C: Same-name definitions in conditional branches ---
    same_name_defs_src = (
        "def outer(flag, items):\n"
        "    x = 0\n"
        "    if flag:\n"
        "        def callback():\n"
        "            return x\n"
        "    else:\n"
        "        def callback():\n"
        "            return 1\n"
        "    for item in items:\n"
        "        x += item + callback()\n"
        "    return callback, x\n"
    )
    compile(same_name_defs_src, "<same_name_defs>", "exec")
    sc_same_name = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_text": same_name_defs_src,
            "source_code": "for item in items:\n    x += item + callback()\n",
        },
        tree=ast.parse(same_name_defs_src),
    )
    assert "severed_closure_cell" in sc_same_name["control_flow_hazards"]

    # --- Test D: Rebound aliases and Test E: Conditional closure creation ---
    cond_closure_unreturned_src = (
        "def outer(flag, items):\n"
        "    if flag:\n"
        "        callback = lambda: x\n"
        "    for item in items:\n"
        "        x = item * 2\n"
        "    return items\n"
    )
    compile(cond_closure_unreturned_src, "<cond_closure>", "exec")
    sc_cond_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": cond_closure_unreturned_src,
            "source_code": "for item in items:\n    x = item * 2\n",
            "skip_pre_unit_closures": False,
        },
        tree=ast.parse(cond_closure_unreturned_src),
    )
    sc_cond_lenient = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": cond_closure_unreturned_src,
            "source_code": "for item in items:\n    x = item * 2\n",
            "skip_pre_unit_closures": True,
        },
        tree=ast.parse(cond_closure_unreturned_src),
    )
    assert "severed_closure_cell" in sc_cond_strict["control_flow_hazards"]
    assert "severed_closure_cell" not in sc_cond_lenient["control_flow_hazards"]

    # --- Star import shadowing `len` prevents treating `len(cb)` as pure built-in ---
    star_len_src = (
        "from pkg import *\n"
        "def outer(items):\n"
        "    secret = 1\n"
        "    n = len(lambda: secret)\n"
        "    for x in items:\n"
        "        secret += x + n()\n"
        "    return n, secret\n"
    )
    compile(star_len_src, "<star_len>", "exec")
    sc_star_len = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_text": star_len_src,
            "source_code": "for x in items:\n    secret += x + n()\n",
        },
        tree=ast.parse(star_len_src),
    )
    assert "severed_closure_cell" in sc_star_len["control_flow_hazards"]

    # --- Comprehension filter mutating outer cell via walrus when primary iterable may be empty ---
    comp_filter_mut_src = (
        "def outer(empty_seq):\n"
        "    acc = 0\n"
        "    cb = lambda: acc\n"
        "    res = [x for x in empty_seq if (acc := acc + x) > 0]\n"
        "    return res, cb\n"
    )
    compile(comp_filter_mut_src, "<comp_filter_mut>", "exec")
    sc_comp_filter = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 56,
            "kind": "comprehension",
            "source_text": comp_filter_mut_src,
            "source_code": "[x for x in empty_seq if (acc := acc + x) > 0]",
        },
        tree=ast.parse(comp_filter_mut_src),
    )
    assert "severed_closure_cell" in sc_comp_filter["control_flow_hazards"]

    # --- Test E: `async for` target is conditional (not unconditionally pre-bound) ---
    async_for_src = (
        "async def outer(aiter, items):\n"
        "    async for scale in aiter:\n"
        "        pass\n"
        "    cbs = [lambda x, i=i: x * scale + i for i in items]\n"
        "    return cbs\n"
    )
    compile(async_for_src, "<async_for>", "exec")
    sc_async_for = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 55,
            "kind": "comprehension",
            "source_text": async_for_src,
            "source_code": "[lambda x, i=i: x * scale + i for i in items]",
        },
        tree=ast.parse(async_for_src),
    )
    assert "severed_closure_cell" in sc_async_for["control_flow_hazards"]

    # --- Test F: 3-level nested closure chain with middle class definition ---
    three_level_cls_src = (
        "def outer(cond, items):\n"
        "    if cond:\n"
        "        factor = 3\n"
        "    class Middle:\n"
        "        def build(self):\n"
        "            cbs = [lambda x, i=i: x * factor + i for i in items]\n"
        "            return cbs\n"
        "    return Middle\n"
    )
    compile(three_level_cls_src, "<three_level_cls>", "exec")
    sc_three_cls = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "start_col": 18,
            "end_col": 64,
            "kind": "comprehension",
            "source_text": three_level_cls_src,
            "source_code": "[lambda x, i=i: x * factor + i for i in items]",
        },
        tree=ast.parse(three_level_cls_src),
    )
    assert "severed_closure_cell" in sc_three_cls["control_flow_hazards"]

    # --- Test G & [P1]: `try: del x; might_raise(); return 1 except ValueError: pass` ---
    try_del_ret_src = (
        "def outer(might_raise, items):\n"
        "    x = 1\n"
        "    try:\n"
        "        del x\n"
        "        might_raise()\n"
        "        return []\n"
        "    except ValueError:\n"
        "        pass\n"
        "    cbs = [lambda v, i=i: v + x + i for i in items]\n"
        "    return cbs\n"
    )
    compile(try_del_ret_src, "<try_del_ret>", "exec")
    try_del_tree = ast.parse(try_del_ret_src)
    try_del_fn = try_del_tree.body[0]
    assert isinstance(try_del_fn, ast.FunctionDef)
    may_d: set[str] = set()
    def_s, cond_s = _analyze_block_assignment(
        try_del_fn.body[:2], may_deleted=may_d
    )
    assert "x" not in def_s
    assert "x" in cond_s
    assert "x" in may_d
    sc_try_del_ret = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "start_col": 10,
            "end_col": 51,
            "kind": "comprehension",
            "source_text": try_del_ret_src,
            "source_code": "[lambda v, i=i: v + x + i for i in items]",
        },
        tree=try_del_tree,
    )
    assert "severed_closure_cell" in sc_try_del_ret["control_flow_hazards"]

    # --- _iter_class_body_methods across nested control flow and nested classes ---
    cls_methods_src = (
        "class Complex:\n"
        "    if True:\n"
        "        def m_if(self): pass\n"
        "    else:\n"
        "        async def m_else(self): pass\n"
        "    try:\n"
        "        def m_try(self): pass\n"
        "    except Exception:\n"
        "        def m_except(self): pass\n"
        "    finally:\n"
        "        def m_finally(self): pass\n"
        "    with ctx():\n"
        "        def m_with(self): pass\n"
        "    for _ in range(1):\n"
        "        def m_for(self): pass\n"
        "    class Nested:\n"
        "        def m_nested_only(self): pass\n"
    )
    if sys.version_info >= (3, 10):
        cls_methods_src += (
            "    match 1:\n"
            "        case 1:\n"
            "            def m_match(self): pass\n"
        )
    cls_methods_node = ast.parse(cls_methods_src).body[0]
    assert isinstance(cls_methods_node, ast.ClassDef)
    found_method_names = [m.name for m in _iter_class_body_methods(cls_methods_node)]
    expected_methods = [
        "m_if",
        "m_else",
        "m_try",
        "m_except",
        "m_finally",
        "m_with",
        "m_for",
    ]
    if sys.version_info >= (3, 10):
        expected_methods.append("m_match")
    assert found_method_names == expected_methods
    assert "m_nested_only" not in found_method_names

    # --- Statement-delta cached vs. uncached (`use_stmt_deltas=False`) differential test ---
    delta_diff_src = (
        "def complex_fn(items, cond, mgr):\n"
        "    a = 1\n"
        "    b = []\n"
        "    cb1 = lambda: a + len(b)\n"
        "    if cond:\n"
        "        c = 2\n"
        "    else:\n"
        "        c = 3\n"
        "    for x in items:\n"
        "        b.append(x + c)\n"
        "        try:\n"
        "            d = x * 2\n"
        "        except ValueError:\n"
        "            d = 0\n"
        "    with mgr:\n"
        "        e = a + c\n"
        "    cb2 = lambda: e\n"
        "    return cb1, cb2, b, d\n"
    )
    delta_tree = ast.parse(delta_diff_src)
    delta_fn = delta_tree.body[0]
    assert isinstance(delta_fn, ast.FunctionDef)
    for u_s, u_e in ((5, 8), (9, 14), (15, 16)):
        span_u = _SpanBounds(u_s, u_e)
        info_cached = _analyze_enclosing_fn_cells_uncached(
            delta_fn,
            span_u,
            mod_tree=delta_tree,
            use_stmt_deltas=True,
        )
        info_uncached = _analyze_enclosing_fn_cells_uncached(
            delta_fn,
            span_u,
            mod_tree=delta_tree,
            use_stmt_deltas=False,
        )
        assert info_cached == info_uncached

    # --- Shared-corpus equivalence between `_analyze_block_assignment` and `_extract_block_definite_bindings_and_term` ---
    shared_definite_corpus: list[tuple[str, set[str]]] = [
        ("x = 1\ny = 2\n", {"x", "y"}),
        ("x = 1\ndel x\n", set()),
        (
            "if cond:\n    a = 1\n    b = 2\nelse:\n    a = 3\n",
            {"a"},
        ),
        (
            "if cond:\n    a = 1\n    return 0\nelse:\n    b = 2\n",
            {"b"},
        ),
        (
            "x = 1\ntry:\n    del x\n    op()\n    return 1\nexcept ValueError:\n    pass\n",
            set(),
        ),
        (
            "x = 1\ntry:\n    del x\n    op()\n    return 1\nexcept ValueError:\n    x = 2\n",
            {"x"},
        ),
        (
            "with suppress(ValueError):\n    a = 1\n",
            {"a"},
        ),
        (
            "with suppress(ValueError):\n    a = d['k']\n",
            set(),
        ),
        (
            "with open_res() as r:\n    a = 1\n",
            {"r", "a"},
        ),
        (
            "if (w := get_v()):\n    pass\n",
            {"w"},
        ),
    ]
    if sys.version_info >= (3, 10):
        shared_definite_corpus.append(
            (
                "match v:\n    case 1:\n        m = 1\n    case _:\n        m = 2\n",
                {"m"},
            )
        )
    if sys.version_info >= (3, 11):
        shared_definite_corpus.append(
            (
                "try:\n    a = 1\nexcept* ValueError:\n    a = 2\nfinally:\n    f = 3\n",
                {"f"},
            )
        )
    for snippet, expected_def_set in shared_definite_corpus:
        stmts = ast.parse(snippet).body
        def_a, _ = _analyze_block_assignment(stmts)
        def_b, _ = _extract_block_definite_bindings_and_term(stmts, None)
        assert def_a == expected_def_set, f"_analyze_block_assignment mismatch on:\n{snippet}"
        assert def_b == expected_def_set, f"_extract_block_definite_bindings_and_term mismatch on:\n{snippet}"
        assert def_a == def_b

    # --- Multi-line `_slice_unit_token_lines` boundary-equality test ---
    ml_lines = ["    x = (\n", "        1 + 2\n", "    )\n"]
    first_bytes_len = len(ml_lines[0].rstrip("\r\n").encode("utf-8"))
    last_bytes_len = len(ml_lines[-1].rstrip("\r\n").encode("utf-8"))
    # 1. Start column strictly inside first line (Slices prefix)
    sliced_inside = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": 8, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_inside[0] == "(\n"
    assert sliced_inside[-1] == "    )"
    # 2. Start column exactly at first line's byte length (Retained unchanged)
    sliced_exact_first = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": first_bytes_len, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_exact_first[0] == ml_lines[0]
    # 3. Start column beyond first line's byte length (Retained unchanged)
    sliced_beyond_first = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": first_bytes_len + 10, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_beyond_first[0] == ml_lines[0]
    # 4. Start column immediately after a multibyte UTF-8 character
    utf8_ml = ['    "\u00e9\u754c" + (\n', "        1 + 2\n", "    )\n"]
    after_mb_col = len('    "\u00e9\u754c" + '.encode("utf-8"))
    sliced_mb = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": after_mb_col, "end_col": 5},
        utf8_ml,
    )
    assert sliced_mb[0] == "(\n"
    # 5. Pre-sliced vs unsliced input matching `source_code`
    presliced_ml = ["(\n", "        1 + 2\n", "    )\n"]
    assert (
        _slice_unit_token_lines(
            {
                "kind": "complex_expr",
                "start_col": 8,
                "end_col": last_bytes_len,
                "source_code": "".join(presliced_ml),
            },
            presliced_ml,
        )
        == presliced_ml
    )

    # --- Test F (Review): Multiple `except*` clauses and `finally` assignments/deletions ---
    if sys.version_info >= (3, 11):
        multi_trystar_src = (
            "keep = 1\n"
            " victim = 2\n"
            "try:\n"
            "    a = 10\n"
            "except* ValueError as eg1:\n"
            "    a = 20\n"
            "    del victim\n"
            "except* TypeError as eg2:\n"
            "    a = 30\n"
            "finally:\n"
            "    fin = 40\n"
        ).replace("\n victim", "\nvictim")
        ts_stmts = ast.parse(multi_trystar_src).body
        ts_def_a, ts_cond_a = _analyze_block_assignment(ts_stmts)
        ts_def_b, _ = _extract_block_definite_bindings_and_term(ts_stmts, None)
        assert ts_def_a == {"keep", "fin"}
        assert ts_def_b == {"keep", "fin"}
        assert {"a", "victim"} <= ts_cond_a

    # --- Test G (Review): Cached (`use_cache=True`) vs uncached (`use_cache=False`) helper equivalence ---
    clear_dataflow_cache()
    helper_eq_tree = ast.parse(
        "def outer(box, items):\n"
        "    x = 1\n"
        "    @dec\n"
        "    def inner(v):\n"
        "        nonlocal x\n"
        "        x += v\n"
        "        box.append(lambda: x + v)\n"
        "        return lambda: x\n"
        "    return inner\n"
    )
    outer_eq = helper_eq_tree.body[0]
    assert isinstance(outer_eq, ast.FunctionDef)
    inner_eq = outer_eq.body[1]
    assert isinstance(inner_eq, ast.FunctionDef)
    eq_ctx = _get_fn_scope_context(outer_eq, mod_tree=helper_eq_tree)
    assert _extract_closure_nonlocal_mutations_and_deletions(
        inner_eq, use_cache=True
    ) == _extract_closure_nonlocal_mutations_and_deletions(
        inner_eq, use_cache=False
    )
    assert _extract_cached_scope_free_reads(
        inner_eq, use_cache=True
    ) == _extract_cached_scope_free_reads(inner_eq, use_cache=False)
    assert _is_self_escaping_def_node(
        inner_eq, use_cache=True
    ) == _is_self_escaping_def_node(inner_eq, use_cache=False)
    assert _is_closure_factory_scope(
        inner_eq, eq_ctx, use_cache=True
    ) == _is_closure_factory_scope(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_returned_closure_refs(
        inner_eq, eq_ctx, use_cache=True
    ) == _extract_fn_returned_closure_refs(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=True
    ) == _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=False
    )

    # --- Multi-path runtime oracle intersection vs both static definite-binding engines ---
    multipath_snippet = (
        "def _probe(cond, fail):\n"
        "    always = 1\n"
        "    survives = 2\n"
        "    killed_on_err = 3\n"
        "    if cond:\n"
        "        branch_both = 10\n"
        "        only_true = 11\n"
        "    else:\n"
        "        branch_both = 20\n"
        "    try:\n"
        "        del killed_on_err\n"
        "        if fail:\n"
        "            raise ValueError('boom')\n"
        "        return ('early', always, survives, branch_both)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    return set(locals().keys())\n"
    )
    ns_mp: dict[str, Any] = {}
    exec(compile(multipath_snippet, "<multipath>", "exec"), ns_mp)  # noqa: S102
    probe_fn = ns_mp["_probe"]
    fallthrough_key_sets: list[set[str]] = []
    for c_val in (True, False):
        for f_val in (True, False):
            res_mp = probe_fn(c_val, f_val)
            if isinstance(res_mp, set):
                fallthrough_key_sets.append(res_mp - {"cond", "fail"})
    runtime_definite = set.intersection(*fallthrough_key_sets)
    probe_body = ast.parse(multipath_snippet).body[0].body[:-1]  # type: ignore[attr-defined]
    static_def_1, _ = _analyze_block_assignment(probe_body)
    static_def_2, _ = _extract_block_definite_bindings_and_term(probe_body, None)
    assert static_def_1 == runtime_definite == {"always", "survives", "branch_both"}
    assert static_def_2 == runtime_definite

    # --- Negative control: forcible pass-by-value extraction past hazard diverges at runtime ---
    orig_hazard_fn_src = (
        "def run_orig(items):\n"
        "    scale = 2\n"
        "    cbs = [lambda x, i=i: x * scale + i for i in items]\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    forced_extracted_src = (
        "def _forced_helper(items, scale):\n"
        "    return [lambda x, i=i: x * scale + i for i in items]\n"
        "def run_forced(items):\n"
        "    scale = 2\n"
        "    cbs = _forced_helper(items, scale)\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    ns_neg_ctrl: dict[str, Any] = {}
    exec(compile(orig_hazard_fn_src + "\n" + forced_extracted_src, "<neg_ctrl>", "exec"), ns_neg_ctrl)  # noqa: S102
    assert ns_neg_ctrl["run_orig"]([1, 2]) == [31, 32]
    assert ns_neg_ctrl["run_forced"]([1, 2]) == [7, 8]
    assert ns_neg_ctrl["run_orig"]([1, 2]) != ns_neg_ctrl["run_forced"]([1, 2])

    # --- Stale `unit["source_code"]` fails closed when real source disagrees and cols are None ---
    real_mod_src = "def f(a):\n    x = a + 1\n    return x\n"
    sc_stale = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_text": real_mod_src,
            "source_code": "completely_unrelated_var = 999\n",
        },
        tree=ast.parse(real_mod_src),
    )
    assert "pathological_ast" in sc_stale["control_flow_hazards"]
    assert sc_stale["is_control_flow_safe"] is False

    # --- Detached unit AST nodes do not pollute `_scope_cell_info_cache` or `_scope_parent_maps` ---
    clear_dataflow_cache()
    unit_with_nested_src = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        def helper(v):\n"
        "            return v + 1\n"
        "        total += helper(x)\n"
        "    return total\n"
    )
    mod_tree_clean = ast.parse(unit_with_nested_src)
    mod_nodes = set(ast.walk(mod_tree_clean))
    _ = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_text": unit_with_nested_src,
            "source_code": (
                "for x in items:\n"
                "    def helper(v):\n"
                "        return v + 1\n"
                "    total += helper(x)\n"
            ),
        },
        tree=mod_tree_clean,
    )
    for p_key in _scope_parent_maps:
        assert p_key in mod_nodes
    for c_key in _scope_cell_info_cache:
        if isinstance(c_key[0], ast.AST):
            assert c_key[0] in mod_nodes

    # --- `analyze_unit_variable_scope` explicit `skip_pre_unit_closures` parameter ---
    lenient_unit_src = (
        "def f(items):\n"
        "    cb = lambda: items\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "    return cb, acc\n"
    )
    u_len = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": lenient_unit_src,
        "source_code": "for x in items:\n    acc = x * 2\n",
    }
    comb_lenient = analyze_unit_variable_scope(
        u_len, dict(u_len), skip_pre_unit_closures=True
    )
    assert comb_lenient["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in comb_lenient["control_flow_hazards"]


def test_suppress_with_termination_and_definite_bindings() -> None:
    """Verifies Items 2 and 3: control-flow termination vs exception suppression and non-raising definite bindings in with suppress(...)."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # Item 2: return inside with suppress(Exception) CANNOT be suppressed by __exit__
    ret_in_suppress_src = (
        "with suppress(Exception):\n"
        "    return 42\n"
        "x = 1\n"
    )
    ret_stmts = ast.parse(ret_in_suppress_src).body
    assert _block_terminates([ret_stmts[0]]) is True
    assert _block_terminates(ret_stmts) is True
    def_ret, cond_ret = _analyze_block_assignment(ret_stmts)
    bound_ret, term_ret = _extract_block_definite_bindings_and_term(
        ret_stmts, None
    )
    assert def_ret == set()
    assert "x" not in cond_ret
    assert bound_ret == set()
    assert term_ret is True

    # Conversely, raise inside with suppress(Exception) CAN be suppressed by __exit__
    raise_in_suppress_src = (
        "with suppress(Exception):\n"
        "    raise ValueError('boom')\n"
        "x = 1\n"
    )
    raise_stmts = ast.parse(raise_in_suppress_src).body
    assert _block_terminates([raise_stmts[0]]) is False
    assert _block_terminates(raise_stmts) is False
    def_raise, _ = _analyze_block_assignment(raise_stmts)
    bound_raise, term_raise = _extract_block_definite_bindings_and_term(
        raise_stmts, None
    )
    assert def_raise == {"x"}
    assert bound_raise == {"x"}
    assert term_raise is False

    # Return after a potentially-raising statement inside with suppress(...) does NOT guarantee termination
    risky_before_ret_src = (
        "with suppress(KeyError):\n"
        "    scale = d['x']\n"
        "    return 42\n"
        "x = 1\n"
    )
    risky_ret_stmts = ast.parse(risky_before_ret_src).body
    assert _block_terminates([risky_ret_stmts[0]]) is False
    def_rret, cond_rret = _analyze_block_assignment(risky_ret_stmts)
    bound_rret, term_rret = _extract_block_definite_bindings_and_term(
        risky_ret_stmts, None
    )
    assert def_rret == {"x"}
    assert "scale" in cond_rret
    assert bound_rret == {"x"}
    assert term_rret is False

    # If/else inside with suppress(...) where test is safe and both branches return constants terminates
    if_ret_suppress_src = (
        "with suppress(Exception):\n"
        "    flag = True\n"
        "    if flag:\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
        "x = 1\n"
    )
    if_ret_stmts = ast.parse(if_ret_suppress_src).body
    assert _block_terminates([if_ret_stmts[0]]) is True
    assert (
        _extract_block_definite_bindings_and_term(if_ret_stmts, None)
        == ({"flag"}, True)
    )

    # Item 3: Non-raising prefix statements inside with suppress(...) bind definitely
    mixed_suppress_src = (
        "with suppress(KeyError):\n"
        "    value = 42\n"
        "    pair_a, pair_b = 1, 2\n"
        "    scale = d['x']\n"
        "    after_risky = 99\n"
    )
    mixed_stmts = ast.parse(mixed_suppress_src).body
    def_mix, cond_mix = _analyze_block_assignment(mixed_stmts)
    bound_mix, term_mix = _extract_block_definite_bindings_and_term(
        mixed_stmts, None
    )
    assert def_mix == {"value", "pair_a", "pair_b"}
    assert bound_mix == {"value", "pair_a", "pair_b"}
    assert {"scale", "after_risky"} <= cond_mix
    assert term_mix is False


def test_scope_cache_key_permitted_parameter_types() -> None:
    """Verifies Item 4: _ScopeCacheKey enforces permitted immutable AST-free parameter schema."""
    import pytest
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _make_trusted_scope_cache_key,
    )

    mod = ast.parse("x = 1\n")
    stmt = mod.body[0]

    # Permitted scalar and immutable container types succeed
    valid_key = _ScopeCacheKey(
        mod,
        "test_kind",
        params=(
            "str_val",
            42,
            3.14,
            True,
            b"bytes",
            None,
            ("nested_tuple", frozenset({"a", "b", 1})),
        ),
        aux_ast_nodes=(stmt,),
    )
    assert valid_key.owner is mod
    assert valid_key.aux_ast_nodes == (stmt,)

    # Public make_trusted is not exposed on _ScopeCacheKey; _make_trusted_scope_cache_key is private
    assert not hasattr(_ScopeCacheKey, "make_trusted")
    trusted_key = _make_trusted_scope_cache_key(
        mod, "trusted_kind", params=("ok", 1)
    )
    assert trusted_key.kind == "trusted_kind"

    # AST nodes anywhere in params (including inside list/dict/set/tuple/frozenset) raise ValueError
    for bad_ast_params in (
        (stmt,),
        ((stmt,),),
        (frozenset({stmt}),),
        ([stmt],),
        ({stmt},),
        ({"k": stmt},),
        ({stmt: "v"},),
    ):
        with pytest.raises(ValueError, match="must not contain ast.AST"):
            _ScopeCacheKey(mod, "bad_ast", params=bad_ast_params)

    # Mutable containers or arbitrary objects in params raise TypeError
    for bad_type_params in (
        (["a", "b"],),
        ({"k": "v"},),
        ({"a", "b"},),
        (object(),),
    ):
        with pytest.raises(
            TypeError, match="only permits immutable"
        ):
            _ScopeCacheKey(mod, "bad_type", params=bad_type_params)


def test_closure_escape_analysis_edge_cases_5_1_to_5_3() -> None:
    """Verifies Item 5 (5.1, 5.2, 5.3): name collisions, class vs lexical namespaces, and definition vs invocation execution."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_and_executed_intra_unit_defs,
    )

    # 5.1(a): Multiple definitions sharing a name in mutually exclusive branches
    branch_defs_src = (
        "if cond:\n"
        "    def f():\n"
        "        return a\n"
        "else:\n"
        "    def f():\n"
        "        return b\n"
        "res = f()\n"
    )
    branch_stmts = ast.parse(branch_defs_src).body
    if_node = branch_stmts[0]
    assert isinstance(if_node, ast.If)
    f_then = if_node.body[0]
    f_else = if_node.orelse[0]
    _, esc_ids_br, exec_ids_br = _collect_escaped_and_executed_intra_unit_defs(
        branch_stmts, set()
    )
    assert id(f_then) in exec_ids_br and id(f_else) in exec_ids_br
    assert id(f_then) not in esc_ids_br and id(f_else) not in esc_ids_br

    # 5.1(b): Definition followed by ordinary assignment (rebound after alias vs rebound without escape)
    rebound_alias_stmts = ast.parse(
        "def f():\n    return a\nsaved = f\nf = 123\n"
    ).body
    _, esc_ids_ra, exec_ids_ra = _collect_escaped_and_executed_intra_unit_defs(
        rebound_alias_stmts, {"saved"}
    )
    assert id(rebound_alias_stmts[0]) in esc_ids_ra
    assert id(rebound_alias_stmts[0]) in exec_ids_ra

    # 5.1(c): Definitions inside nested classes
    nested_cls_stmts = ast.parse(
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            return x\n"
    ).body
    outer_cls = nested_cls_stmts[0]
    assert isinstance(outer_cls, ast.ClassDef)
    inner_cls = outer_cls.body[0]
    assert isinstance(inner_cls, ast.ClassDef)
    inner_method = inner_cls.body[0]
    # Unescaped Outer: Outer and Inner bodies execute at definition time, but method does not
    _, esc_unesc, exec_unesc = _collect_escaped_and_executed_intra_unit_defs(
        nested_cls_stmts, set()
    )
    assert id(outer_cls) in exec_unesc
    assert id(inner_cls) in exec_unesc
    assert id(inner_method) not in exec_unesc
    # Escaped Outer: Outer, Inner, and method all escape
    _, esc_esc, exec_esc = _collect_escaped_and_executed_intra_unit_defs(
        nested_cls_stmts, {"Outer"}
    )
    assert {id(outer_cls), id(inner_cls), id(inner_method)} <= esc_esc
    assert {id(outer_cls), id(inner_cls), id(inner_method)} <= exec_esc

    # 5.1(d) & 5.3: Functions declaring a nested definition global or nonlocal (executed vs unexecuted)
    nl_rebind_stmts = ast.parse(
        "def dead_binder():\n"
        "    nonlocal outer_cb\n"
        "    def outer_cb():\n"
        "        return x\n"
        "def live_binder():\n"
        "    nonlocal live_cb\n"
        "    def live_cb():\n"
        "        return y\n"
        "live_binder()\n"
    ).body
    dead_fn = nl_rebind_stmts[0]
    live_fn = nl_rebind_stmts[1]
    assert isinstance(dead_fn, ast.FunctionDef) and isinstance(
        live_fn, ast.FunctionDef
    )
    dead_cb = dead_fn.body[1]
    live_cb = live_fn.body[1]
    esc_names_nl, esc_ids_nl, exec_ids_nl = (
        _collect_escaped_and_executed_intra_unit_defs(nl_rebind_stmts, set())
    )
    assert id(dead_cb) not in esc_ids_nl
    assert id(dead_cb) not in exec_ids_nl
    assert id(live_cb) in esc_ids_nl
    assert id(live_cb) in exec_ids_nl
    assert "live_cb" in esc_names_nl

    # 5.2: Class body referencing an outer function before defining a same-named method
    cls_shadow_order_stmts = ast.parse(
        "def helper():\n"
        "    return x\n"
        "class C:\n"
        "    val = helper()\n"
        "    def helper(self):\n"
        "        return 1\n"
    ).body
    outer_helper = cls_shadow_order_stmts[0]
    _, _, exec_cls_order = _collect_escaped_and_executed_intra_unit_defs(
        cls_shadow_order_stmts, set()
    )
    assert id(outer_helper) in exec_cls_order

    # 5.2: Nested class inside class and method referencing outer cell vs self.attr / ClassName.attr
    pep227_src = (
        "def outer(items):\n"
        "    shared = 10\n"
        "    class Container:\n"
        "        shared = 99\n"
        "        class Inner:\n"
        "            def read_outer(self):\n"
        "                return shared + self.shared + Container.shared\n"
        "    inst = Container.Inner()\n"
        "    shared = 20\n"
        "    return inst.read_outer()\n"
    )
    sc_pep227 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 8,
            "kind": "compound_block",
            "source_text": pep227_src,
            "source_code": (
                "class Container:\n"
                "    shared = 99\n"
                "    class Inner:\n"
                "        def read_outer(self):\n"
                "            return shared + self.shared + Container.shared\n"
                "inst = Container.Inner()\n"
            ),
        },
        tree=ast.parse(pep227_src),
    )
    assert "severed_closure_cell" in sc_pep227["control_flow_hazards"]

    # 5.3: Decorator on uncalled function executes at definition time
    dec_exec_stmts = ast.parse(
        "def my_dec(fn):\n"
        "    return fn\n"
        "@my_dec\n"
        "def uncalled():\n"
        "    return 1\n"
    ).body
    dec_fn = dec_exec_stmts[0]
    uncalled_fn = dec_exec_stmts[1]
    _, esc_dec, exec_dec = _collect_escaped_and_executed_intra_unit_defs(
        dec_exec_stmts, set()
    )
    assert id(dec_fn) in exec_dec
    assert id(uncalled_fn) in esc_dec


def test_generator_expression_eager_vs_lazy_table_and_nested_walrus() -> None:
    """Verifies Item 6: all 10 eager vs lazy generator expression cases and nested walrus generators."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_comprehension_deferred_closure_info,
        _extract_lazy_genexp_walrus_names,
        _get_fn_scope_context,
        _is_lazy_genexp,
    )

    # All 10 cases from the review table:
    # (func_code_body, expected_is_lazy)
    table_cases: list[tuple[str, bool]] = [
        ("return sum(x for x in values)", False),
        ("return list(x for x in values)", False),
        ("return tuple(x for x in values)", False),
        ("return sorted(x for x in values)", False),
        ("return any(x for x in values)", False),
        ("return all(x for x in values)", False),
        ("return custom(x for x in values)", True),
        ("list = custom_fn\n    return list(x for x in values)", True),
        ("return list((x for x in values))", False),
        ("return iter(x for x in values)", True),
    ]
    for body_line, expected_lazy in table_cases:
        fn_src = f"def f(values, custom, custom_fn):\n    {body_line}\n"
        mod = ast.parse(fn_src)
        fn_node = mod.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        ctx = _get_fn_scope_context(fn_node, mod_tree=mod)
        genexp_nodes = [
            n for n in ast.walk(fn_node) if isinstance(n, ast.GeneratorExp)
        ]
        assert len(genexp_nodes) == 1
        assert _is_lazy_genexp(genexp_nodes[0], ctx) is expected_lazy, body_line

    # Walrus case 1: gen = ((count := count + x) for x in values)
    walrus_gen1_src = (
        "def f(values):\n"
        "    count = 0\n"
        "    gen = ((count := count + x) for x in values)\n"
        "    res = list(gen)\n"
        "    return res, count\n"
    )
    mod1 = ast.parse(walrus_gen1_src)
    fn1 = mod1.body[0]
    assert isinstance(fn1, ast.FunctionDef)
    ctx1 = _get_fn_scope_context(fn1, mod_tree=mod1)
    assign1 = fn1.body[1]
    assert isinstance(assign1, ast.Assign) and isinstance(
        assign1.value, ast.GeneratorExp
    )
    assert _extract_lazy_genexp_walrus_names(assign1, ctx1) == {"count"}
    _, def_muts1 = _extract_comprehension_deferred_closure_info(
        assign1.value, ctx1
    )
    assert "count" in def_muts1
    sc_w1 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "basic_block",
            "source_text": walrus_gen1_src,
            "source_code": (
                "count = 0\n"
                "gen = ((count := count + x) for x in values)\n"
            ),
        },
        tree=mod1,
    )
    assert "severed_closure_cell" in sc_w1["control_flow_hazards"]

    # Walrus case 2: gen = (sum((count := count + x) for x in values) for _ in range(1))
    # Outer generator is lazy; inner generator is eagerly consumed WHEN outer generator is advanced!
    walrus_gen2_src = (
        "def f(values):\n"
        "    count = 0\n"
        "    gen = (sum((count := count + x) for x in values) for _ in range(1))\n"
        "    res = list(gen)\n"
        "    return res, count\n"
    )
    mod2 = ast.parse(walrus_gen2_src)
    fn2 = mod2.body[0]
    assert isinstance(fn2, ast.FunctionDef)
    ctx2 = _get_fn_scope_context(fn2, mod_tree=mod2)
    assign2 = fn2.body[1]
    assert isinstance(assign2, ast.Assign)
    outer_gen = assign2.value
    assert isinstance(outer_gen, ast.GeneratorExp)
    assert isinstance(outer_gen.elt, ast.Call)
    inner_gen = outer_gen.elt.args[0]
    assert isinstance(inner_gen, ast.GeneratorExp)
    assert _is_lazy_genexp(outer_gen, ctx2) is True
    assert _is_lazy_genexp(inner_gen, ctx2) is False
    assert _extract_lazy_genexp_walrus_names(assign2, ctx2) == {"count"}
    _, def_muts2 = _extract_comprehension_deferred_closure_info(
        outer_gen, ctx2
    )
    assert "count" in def_muts2
    sc_w2 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "basic_block",
            "source_text": walrus_gen2_src,
            "source_code": (
                "count = 0\n"
                "gen = (sum((count := count + x) for x in values) for _ in range(1))\n"
            ),
        },
        tree=mod2,
    )
    assert "severed_closure_cell" in sc_w2["control_flow_hazards"]


def test_span_line_index_performance_and_scale() -> None:
    """Verifies Item 7: _SpanLineIndex scales cleanly across 10,000-statement modules, deep nesting, and multi-unit lines."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _has_unresolved_span_bounds,
        _resolve_unit_col_bounds,
    )

    clear_dataflow_cache()

    # 1. Module with 10,000 statements
    large_mod_lines = [f"v_{i} = {i}\n" for i in range(10000)]
    large_mod_tree = ast.parse("".join(large_mod_lines))
    cand_5000 = ast.parse("v_5000 = 5000\n").body
    cand_9999 = ast.parse("v_9999 = 9999\n").body
    assert _resolve_unit_col_bounds(
        cand_5000, large_mod_tree, 5001, 5001, None, None
    ) == (0, len("v_5000 = 5000"))
    assert _resolve_unit_col_bounds(
        cand_9999, large_mod_tree, 10000, 10000, None, None
    ) == (0, len("v_9999 = 9999"))
    assert (
        _has_unresolved_span_bounds(
            large_mod_tree, 5001, 5001, 0, len("v_5000 = 5000"), cand_5000
        )
        is False
    )

    # 2. Deeply nested functions (15 levels deep)
    deep_lines = ["def f_0(x):\n"]
    for depth in range(1, 15):
        indent = "    " * depth
        deep_lines.append(f"{indent}def f_{depth}(x):\n")
    leaf_indent = "    " * 15
    deep_lines.append(f"{leaf_indent}target = x + 1\n")
    deep_lines.append(f"{leaf_indent}return target\n")
    for depth in range(14, 0, -1):
        indent = "    " * depth
        deep_lines.append(f"{indent}return f_{depth}(x)\n")
    deep_tree = ast.parse("".join(deep_lines))
    cand_leaf = ast.parse("target = x + 1\n").body
    expected_s_col = len(leaf_indent)
    expected_e_col = len(leaf_indent) + len("target = x + 1")
    assert _resolve_unit_col_bounds(
        cand_leaf, deep_tree, 16, 16, None, None
    ) == (expected_s_col, expected_e_col)

    # 3. Many candidate units on the same source line (20 semicolon-separated statements)
    semi_parts = [f"s_{i} = {i}" for i in range(20)]
    semi_line = "    " + "; ".join(semi_parts) + "\n"
    semi_tree = ast.parse(f"def multi():\n{semi_line}")
    for i in (0, 7, 19):
        part_cand = ast.parse(f"s_{i} = {i}\n").body
        pos = semi_line.index(f"s_{i} = {i}")
        assert _resolve_unit_col_bounds(
            part_cand, semi_tree, 2, 2, None, None
        ) == (pos, pos + len(f"s_{i} = {i}"))
    assert (
        _has_unresolved_span_bounds(semi_tree.body[0], 2, 2, None, None)
        is True
    )

    # 4. Hundreds of candidate units within one large function + repeated inspection of same AST tree
    big_fn_lines = ["def big_fn(x):\n"] + [
        f"    step_{i} = x + {i}\n" for i in range(300)
    ]
    big_fn_tree = ast.parse("".join(big_fn_lines))
    for i in range(0, 300, 3):
        c_stmt = [big_fn_tree.body[0].body[i]]  # type: ignore[attr-defined]
        ln = i + 2
        s_c, e_c = _resolve_unit_col_bounds(
            c_stmt, big_fn_tree, ln, ln, None, None
        )
        assert (s_c, e_c) == (4, 4 + len(f"step_{i} = x + {i}"))
        assert (
            _has_unresolved_span_bounds(
                big_fn_tree.body[0], ln, ln, s_c, e_c, c_stmt
            )
            is False
        )

    assert any(
        k.owner is big_fn_tree and k.kind == "span_line_index"
        for k in _scope_cell_info_cache
    )
    _evict_tree_scopes(big_fn_tree)
    assert not any(
        k.owner is big_fn_tree
        for k in _scope_cell_info_cache
    )


def test_ensure_unit_scope_closure_mode_all_six_cases() -> None:
    """Verifies Item 8: all 6 scope-mode and missing-file / source_text fallback cases."""
    from pydoppelgangerhunt.fixer import (
        inspect_single_unit_scope,
        synthesize_shared_helper_code,
    )
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    src_with_pre_closure = (
        "def proc(items):\n"
        "    cb = lambda: acc\n"
        "    for x in items:\n"
        "        acc = x + 1\n"
        "    return 1\n"
    )
    base_unit = {
        "file": "missing_virtual_file_12345.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": src_with_pre_closure,
        "source_code": "for x in items:\n    acc = x + 1\n",
    }
    strict_scope = inspect_single_unit_scope(
        {**base_unit, "skip_pre_unit_closures": False}
    )
    lenient_scope = inspect_single_unit_scope(
        {**base_unit, "skip_pre_unit_closures": True}
    )
    assert strict_scope["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_scope["control_flow_hazards"]
    assert lenient_scope["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in lenient_scope["control_flow_hazards"]

    # 1. Strict scope reused in lenient mode -> recomputes to lenient
    res_1 = ensure_unit_scope_closure_mode(base_unit, strict_scope, True)
    assert res_1["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in res_1["control_flow_hazards"]

    # 2. Lenient scope reused in strict mode -> recomputes to strict
    res_2 = ensure_unit_scope_closure_mode(base_unit, lenient_scope, False)
    assert res_2["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in res_2["control_flow_hazards"]

    # 3. Scope missing _skip_pre_unit_closures -> recomputes and attaches flag
    unmarked = {
        k: v for k, v in lenient_scope.items() if k != "_skip_pre_unit_closures"
    }
    res_3 = ensure_unit_scope_closure_mode(base_unit, unmarked, False)
    assert res_3["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in res_3["control_flow_hazards"]

    # 4. Pathological AST scope reused in either mode without recomputing
    pathological_scope = {
        **strict_scope,
        "control_flow_hazards": ["pathological_ast"],
        "is_control_flow_safe": False,
    }
    assert (
        ensure_unit_scope_closure_mode(base_unit, pathological_scope, True)
        is pathological_scope
    )
    assert (
        ensure_unit_scope_closure_mode(base_unit, pathological_scope, False)
        is pathological_scope
    )

    # 5. Missing source file with precomputed scope and no source_text -> fails closed
    no_src_unit = {
        "file": "nonexistent_on_disk_98765.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
    }
    res_5 = ensure_unit_scope_closure_mode(no_src_unit, lenient_scope, False)
    assert res_5["is_control_flow_safe"] is False
    assert "severed_closure_cell" in res_5["control_flow_hazards"]
    assert res_5["_skip_pre_unit_closures"] is False

    # 6. Valid source_text supplied alongside a missing file (even if unit has source_text: None or source_lines: [])
    unit_with_none_src = {
        "file": "nonexistent_on_disk_98765.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": None,
        "source_lines": [],
    }
    res_6 = ensure_unit_scope_closure_mode(
        unit_with_none_src,
        lenient_scope,
        True,
        source_text=src_with_pre_closure,
    )
    assert res_6["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in res_6["control_flow_hazards"]
    helper_code = synthesize_shared_helper_code(
        unit_with_none_src,
        dict(unit_with_none_src),
        source_text1=src_with_pre_closure,
        source_text2=src_with_pre_closure,
        scope1=lenient_scope,
        scope2=dict(lenient_scope),
        skip_pre_unit_closures=True,
    )
    assert "def " in helper_code and "acc = x + 1" in helper_code


def test_utf8_source_slicing_multibyte_and_idempotence() -> None:
    """Verifies Item 9: UTF-8 byte-offset slicing across multibyte identifiers/literals and pre-sliced idempotence."""
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    # 1. café = "hello"; result = café + "world"
    line1 = 'café = "hello"; result = café + "world"\n'
    expr1 = 'café + "world"'
    s1 = len('café = "hello"; result = '.encode("utf-8"))
    e1 = s1 + len(expr1.encode("utf-8"))
    assert _slice_utf8_line(line1, s1, e1) == expr1
    unit1 = {"kind": "complex_expr", "start_col": s1, "end_col": e1}
    sliced1 = _slice_unit_token_lines(unit1, [line1])
    assert sliced1 == [expr1]
    # Idempotence on already-sliced lines
    assert _slice_unit_token_lines(unit1, sliced1) == [expr1]

    # 2. result = "é"; another = "世界"; final = 42
    line2 = 'result = "é"; another = "世界"; final = 42\n'
    for target in ('"é"', '"世界"', "42"):
        char_idx = line2.index(target)
        s_b = len(line2[:char_idx].encode("utf-8"))
        e_b = len(line2[: char_idx + len(target)].encode("utf-8"))
        assert _slice_utf8_line(line2, s_b, e_b) == target
        u_t = {"kind": "complex_expr", "start_col": s_b, "end_col": e_b}
        out_t = _slice_unit_token_lines(u_t, [line2])
        assert out_t == [target]
        assert _slice_unit_token_lines(u_t, out_t) == [target]

    # 3. Pre-sliced single-line expression where s_col < len(sliced) and source_code is omitted
    line3 = "x = [i + 1 for i in items]\n"
    expr3 = "[i + 1 for i in items]"
    u3 = {"kind": "comprehension", "start_col": 4, "end_col": 4 + len(expr3)}
    sliced3 = _slice_unit_token_lines(u3, [line3])
    assert sliced3 == [expr3]
    assert _slice_unit_token_lines(u3, sliced3) == [expr3]

    # 4. Multiline UTF-8 expression slicing and pre-sliced idempotence
    ml_utf8 = [
        '    résultat = (\n',
        '        café + "世界"\n',
        '    )\n',
    ]
    s_ml = len("    résultat = ".encode("utf-8"))
    e_ml = len("    )".encode("utf-8"))
    u_ml = {"kind": "complex_expr", "start_col": s_ml, "end_col": e_ml}
    sliced_ml = _slice_unit_token_lines(u_ml, ml_utf8)
    assert sliced_ml == ['(\n', '        café + "世界"\n', '    )']
    assert _slice_unit_token_lines(u_ml, sliced_ml) == sliced_ml


def test_property_invariants_cache_isolation_monotonicity_equivalence() -> None:
    """Verifies Item 10: all 7 property-oriented invariants across a diverse AST corpus."""
    import symtable
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_parent_maps,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _has_invalid_lexical_scope_declarations,
    )

    corpus_programs = [
        (
            "def f1(xs):\n"
            "    a = 1\n"
            "    res = [x + a for x in xs]\n"
            "    return res\n",
            3,
            3,
            "comprehension",
            "res = [x + a for x in xs]\n",
        ),
        (
            "def f2(xs):\n"
            "    a = 1\n"
            "    cb = lambda: a\n"
            "    for x in xs:\n"
            "        a = x + 2\n"
            "    return cb(), a\n",
            4,
            5,
            "compound_block",
            "for x in xs:\n    a = x + 2\n",
        ),
        (
            "def f3(d):\n"
            "    with suppress(KeyError):\n"
            "        v = 10\n"
            "    out = v + 1\n"
            "    return out\n",
            4,
            4,
            "basic_block",
            "out = v + 1\n",
        ),
    ]

    for src, u_s, u_e, kind, snippet in corpus_programs:
        tree = ast.parse(src)
        tree_nodes = set(ast.walk(tree))
        line_text = src.splitlines()[u_s - 1]
        s_col = len(line_text) - len(line_text.lstrip())
        end_line_text = src.splitlines()[u_e - 1]
        e_col = len(end_line_text.encode("utf-8"))

        unit_inferred = {
            "start": u_s,
            "end": u_e,
            "kind": kind,
            "source_text": src,
            "source_code": snippet,
        }
        unit_explicit = {
            **unit_inferred,
            "start_col": s_col,
            "end_col": e_col,
        }

        # 1. Cache equivalence (uncached fresh vs cached warm)
        clear_dataflow_cache()
        res_cold = inspect_single_unit_scope(dict(unit_explicit), tree=tree)
        res_warm = inspect_single_unit_scope(dict(unit_explicit), tree=tree)
        assert res_cold == res_warm

        # 3. Defensive copies (mutating returned dict/lists does not corrupt cache)
        res_warm["inputs"].append("__mutated_input__")
        res_warm["control_flow_hazards"].append("__mutated_hazard__")
        res_after_mut = inspect_single_unit_scope(
            dict(unit_explicit), tree=tree
        )
        assert "__mutated_input__" not in res_after_mut["inputs"]
        assert (
            "__mutated_hazard__" not in res_after_mut["control_flow_hazards"]
        )

        # 4. AST isolation (only nodes belonging to `tree` are retained in caches)
        for p_node in _scope_parent_maps:
            assert p_node in tree_nodes
        for c_key in _scope_cell_info_cache:
            for ast_n in (c_key.owner, *c_key.aux_ast_nodes):
                assert ast_n in tree_nodes

        # 5. Strictness monotonicity (safe in strict -> safe in lenient)
        sc_strict = inspect_single_unit_scope(
            {**unit_explicit, "skip_pre_unit_closures": False}, tree=tree
        )
        sc_lenient = inspect_single_unit_scope(
            {**unit_explicit, "skip_pre_unit_closures": True}, tree=tree
        )
        if sc_strict["is_control_flow_safe"]:
            assert sc_lenient["is_control_flow_safe"] is True
        assert set(sc_lenient["control_flow_hazards"]).issubset(
            set(sc_strict["control_flow_hazards"])
        )

        # 6. Span equivalence (explicit column bounds vs inferred column bounds)
        sc_inferred = inspect_single_unit_scope(
            {**unit_inferred, "skip_pre_unit_closures": False}, tree=tree
        )
        assert sc_inferred == sc_strict

        # 2. Cache eviction (evicting tree removes all entries referencing tree_nodes)
        _evict_tree_scopes(tree)
        assert not any(p in tree_nodes for p in _scope_parent_maps)
        assert not any(
            _key_references_scope_nodes(k, tree_nodes)
            for k in _scope_cell_info_cache
        )

    # 7. Lexical equivalence against Python's symtable/compile
    lexical_samples = [
        ("def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2\n", False),
        ("def f():\n    global x\n    def g():\n        nonlocal x\n        x = 2\n", True),
        ("def f(x):\n    nonlocal x\n", True),
        ("def f():\n    class C:\n        global x\n        def m(self):\n            return x\n", False),
    ]
    for lex_src, expected_invalid in lexical_samples:
        lex_tree = ast.parse(lex_src)
        got_invalid = _has_invalid_lexical_scope_declarations(
            lex_tree.body, mod_tree=lex_tree
        )
        symtable_rejected = False
        try:
            symtable.symtable(lex_src, "<lex_test>", "exec")
            compile(lex_src, "<lex_test>", "exec")
        except SyntaxError:
            symtable_rejected = True
        assert got_invalid is expected_invalid is symtable_rejected


def test_sibling_closure_ordinary_store_followed_by_call_or_mutation() -> None:
    """Verifies Finding 1: ordinary stores in extracted subroutines are checked against sibling closures."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope

    # 1. Ordinary store followed by sibling closure invocation reading the rebound local
    store_then_call_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def callback():\n"
        "        return x\n"
        "    x = 1\n"
        "    result = callback()\n"
        "    return result\n"
    )
    tree1 = ast.parse(store_then_call_src)
    for skip_pre in (False, True):
        sc1 = inspect_single_unit_scope(
            {
                "start": 5,
                "end": 7,
                "kind": "basic_block",
                "is_subroutine": True,
                "skip_pre_unit_closures": skip_pre,
                "source_text": store_then_call_src,
                "source_code": "x = 1\nresult = callback()\nreturn result\n",
            },
            tree=tree1,
        )
        assert "severed_closure_cell" in sc1["control_flow_hazards"]
        assert sc1["is_control_flow_safe"] is False

    # 2. Ordinary store followed by sibling closure mutation (both returning and output-unpacking)
    store_then_mutate_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def mutate():\n"
        "        nonlocal x\n"
        "        x = 2\n"
        "    x = 1\n"
        "    mutate()\n"
        "    y = 3\n"
        "    return x, y\n"
    )
    tree2 = ast.parse(store_then_mutate_src)
    for skip_pre in (False, True):
        sc2 = inspect_single_unit_scope(
            {
                "start": 6,
                "end": 8,
                "kind": "basic_block",
                "is_subroutine": True,
                "skip_pre_unit_closures": skip_pre,
                "source_text": store_then_mutate_src,
                "source_code": "x = 1\nmutate()\ny = 3\n",
            },
            tree=tree2,
        )
        assert "severed_closure_cell" in sc2["control_flow_hazards"]
        assert sc2["is_control_flow_safe"] is False

    # 3. Contrast: closure unit with its own local `x = 1` (not nonlocal) does not mutate outer `x`
    closure_own_local_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def callback():\n"
        "        return x\n"
        "    def inner():\n"
        "        x = 1\n"
        "        return callback() + x\n"
        "    return inner()\n"
    )
    sc_clos = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_text": closure_own_local_src,
            "source_code": "def inner():\n    x = 1\n    return callback() + x\n",
        },
        tree=ast.parse(closure_own_local_src),
    )
    assert "severed_closure_cell" not in sc_clos["control_flow_hazards"]


def test_suppress_with_name_errors_and_custom_bool_dunder() -> None:
    """Verifies Finding 2: ast.Name and truthiness checks inside suppress(...) are not assumed non-raising."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _extract_block_definite_bindings_and_term,
        _find_enclosing_chain_definite_pre_bound,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # 1. Assignment from unbound local inside suppress(NameError)
    unbound_local_src = (
        "from contextlib import suppress\n"
        "def example(flag):\n"
        "    with suppress(NameError):\n"
        "        x = missing_local\n"
        "    cb = lambda: x\n"
        "    if flag:\n"
        "        missing_local = 1\n"
        "    return cb()\n"
    )
    t_ul = ast.parse(unbound_local_src)
    chain_ul = _find_enclosing_function_chain(t_ul, 5, 5)
    pre_ul = _find_enclosing_chain_definite_pre_bound(chain_ul, 5, 5, mod_tree=t_ul)
    assert "x" not in pre_ul
    sc_ul = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "closure",
            "source_text": unbound_local_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=t_ul,
    )
    assert "severed_closure_cell" in sc_ul["control_flow_hazards"]

    # 2. Assignment from undefined global inside suppress(NameError)
    undef_global_src = (
        "from contextlib import suppress\n"
        "def example():\n"
        "    with suppress(NameError):\n"
        "        x = missing_global\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    t_ug = ast.parse(undef_global_src)
    chain_ug = _find_enclosing_function_chain(t_ug, 5, 5)
    pre_ug = _find_enclosing_chain_definite_pre_bound(chain_ug, 5, 5, mod_tree=t_ug)
    assert "x" not in pre_ug

    # 3. Return of undefined name inside suppress(NameError) does not terminate
    ret_undef_stmts = ast.parse(
        "with suppress(NameError):\n"
        "    return missing_name\n"
    ).body
    assert _block_terminates(ret_undef_stmts) is False
    _, term_undef = _extract_block_definite_bindings_and_term(ret_undef_stmts, None)
    assert term_undef is False

    # 4. `not x` where `x` can have a user-defined __bool__ that raises
    not_x_stmts = ast.parse(
        "with suppress(RuntimeError):\n"
        "    y = not x\n"
        "    return y\n"
    ).body
    assert _block_terminates(not_x_stmts) is False
    b_not_x, term_not_x = _extract_block_definite_bindings_and_term(
        not_x_stmts, None, initial_bound=frozenset({"x"})
    )
    assert "y" not in b_not_x
    assert term_not_x is False
    def_not_x, cond_not_x = _analyze_block_assignment(not_x_stmts)
    assert "y" not in def_not_x
    assert "y" in cond_not_x

    # Contrast: `a = 0; y = not a; return y` (known built-in constant `a`) is non-raising
    safe_bool_stmts = ast.parse(
        "with suppress(RuntimeError):\n"
        "    a = 0\n"
        "    y = not a\n"
        "    return y\n"
    ).body
    assert _block_terminates(safe_bool_stmts) is True
    b_safe_bool, term_safe_bool = _extract_block_definite_bindings_and_term(
        safe_bool_stmts, None
    )
    assert {"a", "y"}.issubset(b_safe_bool)
    assert term_safe_bool is True


def test_class_namespace_nested_deletions_and_ordered_redefinitions() -> None:
    """Verifies Finding 3: class-body definition analysis ignores nested deletions and respects statement order."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_and_executed_intra_unit_defs,
        _extract_class_body_unconditional_def_names,
    )

    # 1. Nested method deleting another class method's name does not affect class-level availability
    nested_del_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    def callback():\n"
        "        return 42\n"
        "    def another():\n"
        "        del callback\n"
        "    result = callback()\n"
    ).body
    outer_cb_1 = nested_del_stmts[0]
    cls_1 = nested_del_stmts[1]
    assert isinstance(cls_1, ast.ClassDef)
    uncond_1 = _extract_class_body_unconditional_def_names(cls_1)
    assert "callback" in uncond_1
    _, esc_ids_1, exec_ids_1 = _collect_escaped_and_executed_intra_unit_defs(
        nested_del_stmts, frozenset()
    )
    assert id(outer_cb_1) not in esc_ids_1
    assert id(outer_cb_1) not in exec_ids_1

    # 2. Class-body deletion followed by a later definition restores class-level availability
    del_then_def_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    del callback\n"
        "    def callback():\n"
        "        return 42\n"
        "    result = callback()\n"
    ).body
    outer_cb_2 = del_then_def_stmts[0]
    cls_2 = del_then_def_stmts[1]
    assert isinstance(cls_2, ast.ClassDef)
    uncond_2 = _extract_class_body_unconditional_def_names(cls_2)
    assert "callback" in uncond_2
    _, esc_ids_2, exec_ids_2 = _collect_escaped_and_executed_intra_unit_defs(
        del_then_def_stmts, frozenset()
    )
    assert id(outer_cb_2) not in esc_ids_2
    assert id(outer_cb_2) not in exec_ids_2

    # 3. Class-body definition followed by deletion before use falls back to outer scope
    def_then_del_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    def callback():\n"
        "        return 42\n"
        "    del callback\n"
        "    result = callback()\n"
    ).body
    outer_cb_3 = def_then_del_stmts[0]
    cls_3 = def_then_del_stmts[1]
    assert isinstance(cls_3, ast.ClassDef)
    uncond_3 = _extract_class_body_unconditional_def_names(cls_3)
    assert "callback" not in uncond_3
    _, _, exec_ids_3 = _collect_escaped_and_executed_intra_unit_defs(
        def_then_del_stmts, frozenset()
    )
    assert id(outer_cb_3) in exec_ids_3


def test_cyclic_mutable_containers_in_scope_cache_key() -> None:
    """Verifies Finding 4: self-referential and mutually recursive containers do not hang validation."""
    from pydoppelgangerhunt.fixer._df_cache import _ScopeCacheKey

    owner = ast.parse("x = 1")

    # 1. Self-referential list
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_list,))

    # 2. Self-referential dict
    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_dict,))

    # 3. Mutually recursive containers
    mut_a: list[Any] = []
    mut_b: dict[str, Any] = {"a": mut_a}
    mut_a.append(mut_b)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=((mut_a, mut_b),))

    # 4. Cyclic container that also contains an AST node still raises ValueError
    cyclic_with_ast: list[Any] = []
    cyclic_with_ast.append(cyclic_with_ast)
    cyclic_with_ast.append(owner.body[0])
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _ScopeCacheKey(owner, "test", params=(cyclic_with_ast,))


def test_indexed_cache_eviction_multi_owner_node_reuse_and_central_invariant() -> None:
    """Verifies Finding 5: indexed eviction, multi-owner keys, AST node reuse, and the central cache invariant."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_owner_keys_index,
        _scope_parent_maps,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context

    clear_dataflow_cache()

    # 1. Eviction of entries with multiple auxiliary AST owners
    tree_a = ast.parse("def fa():\n    return 1\n")
    tree_b = ast.parse("def fb():\n    return 2\n")
    tree_c = ast.parse("def fc():\n    return 3\n")
    multi_key = _ScopeCacheKey(
        owner=tree_a.body[0],
        kind="multi_aux",
        params=("tag", 1),
        aux_ast_nodes=(tree_b.body[0], tree_c),
    )
    _scope_cell_info_cache[multi_key] = ("cached_val",)
    assert multi_key in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_b.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_c, set())

    # Evicting tree_b (an auxiliary owner) must evict multi_key and clean up its index entries
    _evict_tree_scopes(tree_b)
    assert multi_key not in _scope_cell_info_cache
    assert multi_key not in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key not in _scope_owner_keys_index.get(tree_c, set())

    # 2. Reuse of an AST node after its enclosing tree is evicted
    reused_fn = tree_a.body[0]
    assert isinstance(reused_fn, ast.FunctionDef)
    ctx_before = _get_fn_scope_context(reused_fn, mod_tree=tree_a)
    assert ctx_before.mod_tree is tree_a
    _evict_tree_scopes(tree_a)
    assert not any(
        _key_references_scope_nodes(k, {reused_fn, tree_a})
        for k in _scope_cell_info_cache
    )
    new_wrapper_tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="math", names=[ast.alias(name="*")], level=0
            ),
            reused_fn,
        ],
        type_ignores=[],
    )
    ctx_after = _get_fn_scope_context(reused_fn, mod_tree=new_wrapper_tree)
    assert ctx_after.mod_tree is new_wrapper_tree
    assert ctx_after.has_star_import is True

    # 3. Central cache invariant across all supported cache key shapes
    clear_dataflow_cache()
    trees = [
        ast.parse(f"def fn_{i}(x):\n    y = x + {i}\n    return y\n")
        for i in range(4)
    ]
    tree_node_sets = [set(ast.walk(t)) for t in trees]
    all_supported_keys: list[Any] = []
    for idx, t in enumerate(trees):
        fn_node = t.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        stmt_node = fn_node.body[0]
        next_t = trees[(idx + 1) % len(trees)]
        k_ast = fn_node
        k_sck_single = _ScopeCacheKey(
            owner=fn_node, kind="single", params=(idx, ("nested", True))
        )
        k_sck_multi = _ScopeCacheKey(
            owner=stmt_node,
            kind="multi",
            params=(frozenset({"a", "b"}),),
            aux_ast_nodes=(t, next_t.body[0]),
        )
        k_tuple = ("custom_tuple", (fn_node, idx), frozenset({"meta"}))
        k_frozenset = frozenset({"heterogeneous_label", stmt_node, idx})
        for k in (k_ast, k_sck_single, k_sck_multi, k_tuple, k_frozenset):
            _scope_cell_info_cache[k] = (idx, "cell")
            _scope_closures_cache[k] = [(None, {f"v_{idx}"})]
            all_supported_keys.append(k)

    # Evict trees[1] and verify the exact invariant for every key
    evicted_nodes = tree_node_sets[1]
    _evict_tree_scopes(trees[1])
    for k in all_supported_keys:
        refs_evicted = _key_references_scope_nodes(k, evicted_nodes)
        if refs_evicted:
            assert k not in _scope_cell_info_cache
            assert k not in _scope_closures_cache
        else:
            assert k in _scope_cell_info_cache
            assert k in _scope_closures_cache







