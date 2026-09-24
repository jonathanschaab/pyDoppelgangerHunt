"""Unit tests for grandfathered baseline recording, loading, pruning, and drift immunity."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest import mock

import pytest

import pydoppelgangerhunt

from pydoppelgangerhunt import (
    UnionFind,
    analyze_unit_variable_scope,
    check_units_overlap,
    cluster_clone_families,
    compute_repository_dry_stats,
    extract_unit_source_code,
    filter_clones_by_baseline,
    filter_clones_by_git_diff,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    generate_refactoring_patch,
    load_baseline,
    parse_git_diff_hunks,
    prune_baseline,
    record_baseline,
    refactor_module_units,
    replace_unit_in_source,
    scan_target,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _analyze_block_assignment,
    _block_terminates,
    _populate_unit_receiver_metadata,
    _walrus_assignment_in_expr,
)


def test_modular_pydoppelgangerhunt_exports() -> None:
    """Test that pydoppelgangerhunt package exports clean public API and metadata."""
    assert pydoppelgangerhunt.__version__ == "1.0.0"
    assert callable(pydoppelgangerhunt.main)
    assert callable(pydoppelgangerhunt.scan_target)
    assert callable(pydoppelgangerhunt.cluster_clone_families)
    assert callable(pydoppelgangerhunt.record_baseline)
    assert callable(pydoppelgangerhunt.load_baseline)
    assert callable(pydoppelgangerhunt.pure_structural_fingerprint)
    assert callable(pydoppelgangerhunt.extract_unit_namespace)
    assert callable(pydoppelgangerhunt.namespaced_structural_fingerprint)
    assert callable(pydoppelgangerhunt.prune_baseline)
    assert isinstance(pydoppelgangerhunt.DEFAULT_STOP_SHINGLES, set)
    assert callable(pydoppelgangerhunt.get_boilerplate_stop_shingles)
    assert callable(pydoppelgangerhunt.compute_unit_diff_overlap)
    assert pydoppelgangerhunt.MAJOR_POLICY_THRESHOLD == 0.50
    assert pydoppelgangerhunt.NEW_POLICY_THRESHOLD == 0.80
    assert callable(pydoppelgangerhunt.replace_unit_in_source)
    assert callable(pydoppelgangerhunt.check_units_overlap)
    assert callable(pydoppelgangerhunt.filter_overlapping_clone_units)
    assert callable(pydoppelgangerhunt.refactor_module_units)

def test_baseline_drift_immunity_across_line_shifts(tmp_path: Path) -> None:
    """Test that grandfathered baseline matches via structural hash despite line position shifts."""
    pkg = tmp_path / "baseline_pkg"
    pkg.mkdir()

    file_a = pkg / "worker_a.py"
    file_b = pkg / "worker_b.py"

    code_body = (
        "def run_computation(x, y):\n"
        "    acc = 0\n"
        "    for val in range(x):\n"
        "        acc += val * y\n"
        "    return acc\n"
    )
    file_a.write_text(code_body, encoding="utf-8")
    file_b.write_text(code_body, encoding="utf-8")

    initial_clones = scan_target(str(pkg), min_lines=3, min_tokens=10, threshold=0.90)
    assert len(initial_clones) >= 1
    baseline_path = tmp_path / "baseline.json"
    record_baseline(initial_clones, str(baseline_path), target=str(pkg), threshold=0.90)
    fps = load_baseline(str(baseline_path))

    # Shift lines in file_a by inserting 15 comment lines at top
    shifted_code = ("# Upstream edit shifting lines down\n" * 15) + code_body
    file_a.write_text(shifted_code, encoding="utf-8")

    shifted_clones = scan_target(str(pkg), min_lines=3, min_tokens=10, threshold=0.90)
    assert len(shifted_clones) >= 1
    _, u1, u2 = shifted_clones[0]
    # Verify line numbers indeed shifted
    assert u1["start"] >= 16 or u2["start"] >= 16

    # Verify structural fingerprint suppresses the clone despite the line shift
    remaining, suppressed = filter_clones_by_baseline(shifted_clones, fps)
    assert suppressed >= 1
    assert len(remaining) == 0

def test_fixer_scope_analysis_and_signature_synthesis(tmp_path: Path) -> None:
    """Test AST variable scope analysis and concrete parameter signature synthesis."""
    file_a = tmp_path / "math_a.py"
    file_b = tmp_path / "math_b.py"

    file_a.write_text(
        "def compute_coords(x: float, y: float) -> float:\n"
        "    scaled = (x * 2.0) + (y * 3.0)\n"
        "    return scaled\n",
        encoding="utf-8",
    )
    file_b.write_text(
        "def compute_coords_v2(x: float, y: float) -> float:\n"
        "    scaled = (x * 2.0) + (y * 3.0)\n"
        "    return scaled\n",
        encoding="utf-8",
    )

    u1 = {"file": str(file_a), "start": 1, "end": 3, "name": "compute_coords"}
    u2 = {"file": str(file_b), "start": 1, "end": 3, "name": "compute_coords_v2"}

    scope = analyze_unit_variable_scope(u1, u2, repo_root=str(tmp_path))
    assert scope["inputs"] == ["x", "y"]
    assert "scaled" in scope["outputs"]
    assert "scaled" in scope["locals"]

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_compute_coords" in helper
    assert "x: float, y: float" in helper
    assert "-> float:" in helper
    assert "*args" not in helper

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch
    assert "def _shared_compute_coords" in patch

    # Test single-unit scope analysis
    scope_single = analyze_unit_variable_scope(u1, repo_root=str(tmp_path))
    assert scope_single["inputs"] == ["x", "y"]
    assert scope_single["return_type"] == "float"

    # Test async function with *args and **kwargs
    file_async = tmp_path / "async_service.py"
    file_async.write_text(
        "async def fetch_record(record_id: int, *extra_args: Any, **options: Any) -> int:\n"
        "    return record_id\n",
        encoding="utf-8",
    )
    u_async = {"file": str(file_async), "start": 1, "end": 2, "name": "fetch_record"}
    scope_async = analyze_unit_variable_scope(u_async, repo_root=str(tmp_path))
    assert "record_id" in scope_async["inputs"]
    assert "extra_args" in scope_async["inputs"]
    assert "options" in scope_async["inputs"]
    assert "record_id" in scope_async["outputs"]
    assert scope_async["return_type"] == "int"

    # Test closure-captured variables from outer lexical scopes
    file_closure = tmp_path / "closure_service.py"
    file_closure.write_text(
        "def make_multiplier(factor: int):\n"
        "    def inner_mult(x: int) -> int:\n"
        "        return x * factor\n"
        "    return inner_mult\n",
        encoding="utf-8",
    )
    u_closure = {"file": str(file_closure), "start": 2, "end": 3, "name": "inner_mult"}
    scope_closure = analyze_unit_variable_scope(u_closure, repo_root=str(tmp_path))
    assert "x" in scope_closure["inputs"]
    assert "factor" in scope_closure["inputs"]
    assert "factor" in scope_closure["free_vars"]
    assert scope_closure["return_type"] == "int"

    # Test nonlocal and global variable declarations
    file_scope_vars = tmp_path / "scope_vars.py"
    file_scope_vars.write_text(
        "def stateful_step():\n"
        "    global global_cache\n"
        "    nonlocal outer_counter\n"
        "    outer_counter += 1\n"
        "    global_cache = outer_counter\n"
        "    return outer_counter\n",
        encoding="utf-8",
    )
    u_state = {"file": str(file_scope_vars), "start": 1, "end": 6, "name": "stateful_step"}
    scope_state = analyze_unit_variable_scope(u_state, repo_root=str(tmp_path))
    assert "outer_counter" in scope_state["nonlocals"]
    assert "global_cache" in scope_state["globals"]
    assert "outer_counter" in scope_state["inputs"]
    assert "outer_counter" in scope_state["outputs"]

    # Test class instance attributes (self.x, self.y)
    file_cls = tmp_path / "point_cls.py"
    file_cls.write_text(
        "class Point:\n"
        "    def translate(self, dx: float, dy: float = 0.0) -> None:\n"
        "        self.x += dx\n"
        "        self.y += dy\n",
        encoding="utf-8",
    )
    u_cls = {"file": str(file_cls), "start": 2, "end": 4, "name": "translate"}
    scope_cls = analyze_unit_variable_scope(u_cls, repo_root=str(tmp_path))
    assert scope_cls["inputs"][0] == "self"
    assert "self.x" in scope_cls["attrs_read"]
    assert "self.x" in scope_cls["attrs_written"]
    assert "self.y" in scope_cls["attrs_read"]
    assert "self.y" in scope_cls["attrs_written"]
    assert scope_cls["return_type"] == "None"

    # Test keyword-only parameters and defaults preservation
    file_kw = tmp_path / "kw_service.py"
    file_kw.write_text(
        "def fetch_api(url: str, timeout: float = 5.0, *, retries: int = 3) -> dict:\n"
        "    return {'url': url}\n",
        encoding="utf-8",
    )
    u_kw = {"file": str(file_kw), "start": 1, "end": 2, "name": "fetch_api"}
    scope_kw = analyze_unit_variable_scope(u_kw, repo_root=str(tmp_path))
    assert "url" in scope_kw["inputs"]
    assert "timeout" in scope_kw["inputs"]
    assert "retries" in scope_kw["inputs"]
    assert scope_kw["return_type"] == "dict"

    helper_kw = synthesize_shared_helper_code(u_kw, u_kw, repo_root=str(tmp_path))
    assert "url: str" in helper_kw
    assert "timeout: float = 5.0" in helper_kw
    assert "*, retries: int = 3" in helper_kw
    assert "-> dict:" in helper_kw

    # Test conflicting defaults and conflicting types between clone units
    file_conf1 = tmp_path / "conf1.py"
    file_conf2 = tmp_path / "conf2.py"
    file_conf1.write_text(
        "def process_data(val: int, timeout: float = 10.0) -> int:\n"
        "    return val + int(timeout)\n",
        encoding="utf-8",
    )
    file_conf2.write_text(
        "def process_data_v2(val: float, timeout: float = 20.0) -> str:\n"
        "    return str(val + timeout)\n",
        encoding="utf-8",
    )
    u_c1 = {"file": str(file_conf1), "start": 1, "end": 2, "name": "process_data"}
    u_c2 = {"file": str(file_conf2), "start": 1, "end": 2, "name": "process_data_v2"}
    helper_conf = synthesize_shared_helper_code(u_c1, u_c2, repo_root=str(tmp_path))
    sig_conf = helper_conf.split("\n")[0]
    # Conflicting types val: int vs float fallback to Any
    assert "val: Any" in sig_conf
    # Conflicting defaults 10.0 vs 20.0 are safely omitted from signature
    assert "timeout: float," in sig_conf or "timeout: float)" in sig_conf
    assert "= 10.0" not in sig_conf
    assert "= 20.0" not in sig_conf
    # Test nested function handling (top-level function guard)
    file_nested = tmp_path / "nested_func.py"
    file_nested.write_text(
        "def outer_calc(base: int) -> int:\n"
        "    def inner_step(step: int) -> int:\n"
        "        return base + step\n"
        "    return inner_step(5)\n",
        encoding="utf-8",
    )
    u_nested = {"file": str(file_nested), "start": 1, "end": 4, "name": "outer_calc"}
    scope_nested = analyze_unit_variable_scope(u_nested, repo_root=str(tmp_path))
    assert scope_nested["inputs"] == ["base"]
    assert scope_nested["return_type"] == "int"

    # Test classmethod with cls parameter ordering
    file_cm = tmp_path / "cm_service.py"
    file_cm.write_text(
        "class Builder:\n"
        "    @classmethod\n"
        "    def build(cls, tag: str = 'item') -> str:\n"
        "        return f'{tag}'\n",
        encoding="utf-8",
    )
    u_cm = {"file": str(file_cm), "start": 2, "end": 4, "name": "build"}
    scope_cm = analyze_unit_variable_scope(u_cm, repo_root=str(tmp_path))
    assert scope_cm["inputs"][0] == "cls"
    helper_cm = synthesize_shared_helper_code(u_cm, u_cm, repo_root=str(tmp_path))
    assert (
        "def _shared_build(cls, tag: str = 'item') -> str:" in helper_cm
        or "def _shared_build(cls: Any, tag: str = 'item') -> str:" in helper_cm
    )

    # Test positional default order invalidation (clearing preceding defaults)
    file_ord1 = tmp_path / "ord1.py"
    file_ord2 = tmp_path / "ord2.py"
    file_ord1.write_text(
        "def order_func(a: int = 1, b: int = 2) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    file_ord2.write_text(
        "def order_func_v2(a: int = 1, b: int = 3) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    u_o1 = {"file": str(file_ord1), "start": 1, "end": 2, "name": "order_func"}
    u_o2 = {"file": str(file_ord2), "start": 1, "end": 2, "name": "order_func_v2"}
    helper_ord = synthesize_shared_helper_code(u_o1, u_o2, repo_root=str(tmp_path))
    sig_ord = helper_ord.split("\n")[0]
    assert "a: int, b: int" in sig_ord
    assert "= 1" not in sig_ord

    # Test type_merge_strategy="union"
    helper_union = synthesize_shared_helper_code(u_c1, u_c2, type_merge_strategy="union", repo_root=str(tmp_path))
    sig_union = helper_union.split("\n")[0]
    assert "Union[int, float]" in sig_union
    assert "Union[int, str]" in sig_union

    # Test read-only nonlocal variable (in inputs, but NOT in outputs)
    file_ro_nl = tmp_path / "ro_nonlocal.py"
    file_ro_nl.write_text(
        "def read_only_step(x: int) -> int:\n"
        "    nonlocal scale_factor\n"
        "    return x * scale_factor\n",
        encoding="utf-8",
    )
    u_ro_nl = {"file": str(file_ro_nl), "start": 1, "end": 3, "name": "read_only_step"}
    scope_ro_nl = analyze_unit_variable_scope(u_ro_nl, repo_root=str(tmp_path))
    assert "scale_factor" in scope_ro_nl["inputs"]
    assert "scale_factor" in scope_ro_nl["nonlocals"]
    # Conservative check: scale_factor was only read, never stored, so NOT in outputs
    assert "scale_factor" not in scope_ro_nl["outputs"]

    # Test canonical argument kind ordering (pos -> vararg -> kwonly -> kwarg)
    file_full_args = tmp_path / "full_args.py"
    file_full_args.write_text(
        "def complex_sig(self, a: int, *args: Any, b: int = 1, **kwargs: Any) -> bool:\n"
        "    return True\n",
        encoding="utf-8",
    )
    u_fa = {"file": str(file_full_args), "start": 1, "end": 2, "name": "complex_sig"}
    helper_fa = synthesize_shared_helper_code(u_fa, u_fa, repo_root=str(tmp_path))
    sig_fa = helper_fa.split("\n")[0]
    assert "self: Any, a: int, *args: Any, b: int = 1, **kwargs: Any" in sig_fa
    assert "-> bool:" in sig_fa


    # Test empty / invalid source unit
    scope_empty = analyze_unit_variable_scope({"file": "non_existent.py", "start": 1, "end": 1, "name": "bad"})
    assert scope_empty["inputs"] == []

    # Test vector fallback and invalid baseline JSON
    from pydoppelgangerhunt.baseline import compute_unit_structural_hash  # pylint: disable=import-outside-toplevel
    assert compute_unit_structural_hash({"structural_hash": "existing12345678"}) == "existing12345678"
    assert len(compute_unit_structural_hash({"tokens": ["alpha", "beta"]})) == 16
    h_vec = compute_unit_structural_hash({"vector": {"token_x": 3, "token_y": 1}})
    assert len(h_vec) == 16

    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("NOT_JSON_DATA", encoding="utf-8")
    assert load_baseline(str(corrupt_json)) == set()
    assert load_baseline(str(tmp_path / "absent.json")) == set()

    # Test fixer with no common lines fallback and non-existent patch targets
    file_diff1 = tmp_path / "diff1.py"
    file_diff2 = tmp_path / "diff2.py"
    file_diff1.write_text("def unique_one():\n    return 1\n", encoding="utf-8")
    file_diff2.write_text("def unique_two():\n    return 2\n", encoding="utf-8")
    u_d1 = {"file": str(file_diff1), "start": 1, "end": 2, "name": "unique_one"}
    u_d2 = {"file": str(file_diff2), "start": 1, "end": 2, "name": "unique_two"}
    helper_uncommon = synthesize_shared_helper_code(u_d1, u_d2, repo_root=str(tmp_path))
    assert "_shared_unique_one_unique_two" in helper_uncommon

    u_missing = {"file": "no_such_file_on_disk.py", "start": 1, "end": 2, "name": "miss"}
    assert generate_refactoring_patch([(0.9, u_missing, u_d2)]) == ""

    # Test cli risk warnings helper
    from pydoppelgangerhunt import cli  # pylint: disable=import-outside-toplevel
    warn_u1 = {"file": "mod1.py", "start": 1, "end": 10, "name": "f1"}
    warn_u2 = {"file": "mod2.py", "start": 1, "end": 10, "name": "f2"}
    risk_lines = cli._audit_clone_risk_warnings(  # pylint: disable=protected-access
        warn_u1,
        warn_u2,
        audit_blame=True,
        cov_data={"mod1.py": {1, 2}, "mod2.py": set(range(1, 10))},
        use_color=False,
    )
    assert isinstance(risk_lines, list)

def test_path_independent_structural_fingerprint_rename_and_fallback(tmp_path: Path) -> None:
    """Test pure structural fingerprinting across file renames and legacy baseline fallback."""
    u1 = {"file": "dir_a/worker.py", "name": "do_work", "structural_hash": "hash_worker_01"}
    u2 = {"file": "dir_b/worker.py", "name": "do_work", "structural_hash": "hash_worker_02"}

    # Pure structural fingerprint is independent of file paths and order-invariant
    pure_fp = pydoppelgangerhunt.pure_structural_fingerprint(u1, u2)
    pure_fp_rev = pydoppelgangerhunt.pure_structural_fingerprint(u2, u1)
    assert pure_fp == pure_fp_rev
    assert "hash_worker_01" in pure_fp and "hash_worker_02" in pure_fp
    assert "dir_a" not in pure_fp and "dir_b" not in pure_fp

    # Simulate baseline recording
    baseline_path = tmp_path / "baseline_v120.json"
    pydoppelgangerhunt.record_baseline([(0.95, u1, u2)], str(baseline_path), "test_target", 0.90)

    # Move/rename dir_a/worker.py to dir_renamed/worker.py
    u1_renamed = {"file": "dir_renamed/worker.py", "name": "do_work", "structural_hash": "hash_worker_01"}
    active_clones = [(0.95, u1_renamed, u2)]

    # Old fingerprints (fp, sfp) won't match because path changed, but pure_sfp will match!
    loaded_fps = pydoppelgangerhunt.load_baseline(str(baseline_path))
    remaining, suppressed = pydoppelgangerhunt.filter_clones_by_baseline(active_clones, loaded_fps)
    assert suppressed == 1
    assert len(remaining) == 0

    # Test backward-compatibility with v1.0/v1.1 legacy baseline JSON lacking pure_structural_fingerprint
    legacy_baseline = tmp_path / "baseline_legacy.json"
    legacy_data = {
        "version": "1.1.0",
        "fingerprints": [
            {
                "fingerprint": "old_dir/a.py:calc <===> old_dir/b.py:calc",
                "structural_fingerprint": "old_dir/a.py#hash_c1 <===> old_dir/b.py#hash_c2",
                "hash_a": "hash_c1",
                "hash_b": "hash_c2",
            }
        ]
    }
    legacy_baseline.write_text(json.dumps(legacy_data), encoding="utf-8")
    loaded_legacy_fps = pydoppelgangerhunt.load_baseline(str(legacy_baseline))
    # Check that synthesized pure structural fingerprint is present
    expected_pure = "hash_c1 <===> hash_c2"
    assert expected_pure in loaded_legacy_fps

    # Verify a renamed clone matching hash_c1 and hash_c2 is suppressed using legacy baseline
    u_renamed_a = {"file": "new_dir/a.py", "name": "calc", "structural_hash": "hash_c1"}
    u_renamed_b = {"file": "new_dir/b.py", "name": "calc", "structural_hash": "hash_c2"}
    rem_legacy, supp_legacy = pydoppelgangerhunt.filter_clones_by_baseline(
        [(0.92, u_renamed_a, u_renamed_b)], loaded_legacy_fps
    )
    assert supp_legacy == 1
    assert len(rem_legacy) == 0

def test_baseline_pruning_and_cli(tmp_path: Path) -> None:
    """Test prune_baseline removing orphaned entries, synchronizing renames, and CLI integration."""
    u1 = {"file": "src/active_a.py", "name": "active_a", "structural_hash": "hash_act_a"}
    u2 = {"file": "src/active_b.py", "name": "active_b", "structural_hash": "hash_act_b"}
    u_dead1 = {"file": "src/dead_a.py", "name": "dead_a", "structural_hash": "hash_dead_a"}
    u_dead2 = {"file": "src/dead_b.py", "name": "dead_b", "structural_hash": "hash_dead_b"}
    u_moved1 = {"file": "src/old_loc.py", "name": "calc", "structural_hash": "hash_moved"}
    u_moved2 = {"file": "src/fixed_loc.py", "name": "calc", "structural_hash": "hash_fixed"}

    baseline_file = tmp_path / "prune_test_baseline.json"
    initial_clones = [
        (0.95, u1, u2),
        (0.92, u_dead1, u_dead2),
        (0.90, u_moved1, u_moved2),
    ]
    pydoppelgangerhunt.record_baseline(initial_clones, str(baseline_file), "test_pkg", 0.90)

    # Now simulate u_dead1/u_dead2 being deleted/refactored (not active)
    # and u_moved1 being moved to src/new_loc.py
    u_moved1_new = {"file": "src/new_loc.py", "name": "calc", "structural_hash": "hash_moved"}
    current_active = [
        (0.95, u1, u2),
        (0.90, u_moved1_new, u_moved2),
    ]

    pruned_count, retained_count = pydoppelgangerhunt.prune_baseline(str(baseline_file), current_active)
    assert pruned_count == 1
    assert retained_count == 2

    # Verify pruned baseline content
    content = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert content["clone_count"] == 2
    assert len(content["fingerprints"]) == 2
    # Verify moved clone was retained and synchronized to new file path
    fps_files = [(item["file_a"], item["file_b"]) for item in content["fingerprints"]]
    assert any("src/new_loc.py" in (f1, f2) for f1, f2 in fps_files)

    # Test edge cases: non-existent file, corrupt file, invalid JSON structure
    assert pydoppelgangerhunt.prune_baseline(str(tmp_path / "missing.json"), current_active) == (0, 0)
    corrupt_f = tmp_path / "corrupt_baseline.json"
    corrupt_f.write_text("invalid json content", encoding="utf-8")
    assert pydoppelgangerhunt.prune_baseline(str(corrupt_f), current_active) == (0, 0)
    not_dict_f = tmp_path / "not_dict.json"
    not_dict_f.write_text("[1, 2, 3]", encoding="utf-8")
    assert pydoppelgangerhunt.prune_baseline(str(not_dict_f), current_active) == (0, 0)

    # CLI test: --prune-baseline without baseline specified
    code_missing_base = pydoppelgangerhunt.main([str(tmp_path), "--prune-baseline"])
    assert code_missing_base == 1

    # CLI test: --prune-baseline with non-existent baseline file
    code_not_found = pydoppelgangerhunt.main([str(tmp_path), "--baseline", str(tmp_path / "no_such_file.json"), "--prune-baseline"])
    assert code_not_found == 1

    # CLI test: valid execution with --prune-baseline
    src_dir = tmp_path / "cli_src"
    src_dir.mkdir()
    f1 = src_dir / "mod1.py"
    f2 = src_dir / "mod2.py"
    shared_code = "def compute_val(x, y):\n    res = x * y\n    print(res)\n    return res + 1\n"
    f1.write_text(shared_code, encoding="utf-8")
    f2.write_text(shared_code, encoding="utf-8")

    cli_base = tmp_path / "cli_baseline.json"
    # First record baseline
    record_ret = pydoppelgangerhunt.main([str(src_dir), "--threshold", "0.70", "--min-lines", "3", "--min-tokens", "5", "--record-baseline", str(cli_base)])
    assert record_ret == 0
    assert cli_base.exists()

    # Now prune baseline (nothing pruned, 1 retained, clone suppressed)
    prune_ret = pydoppelgangerhunt.main([str(src_dir), "--threshold", "0.70", "--min-lines", "3", "--min-tokens", "5", "--baseline", str(cli_base), "--prune-baseline"])
    assert prune_ret == 0

def test_cross_namespace_collision_prevention_and_granular_tracking(tmp_path: Path) -> None:
    """Test that identical boilerplate across different modules is disambiguated and tracked granularly."""
    bp_hash = "hash_boilerplate_common"
    u_auth1 = {"file": "auth/views.py", "name": "to_dict", "structural_hash": bp_hash}
    u_auth2 = {"file": "auth/helpers.py", "name": "to_dict", "structural_hash": bp_hash}

    u_bill1 = {"file": "billing/views.py", "name": "to_dict", "structural_hash": bp_hash}
    u_bill2 = {"file": "billing/helpers.py", "name": "to_dict", "structural_hash": bp_hash}

    # Verify extract_unit_namespace
    assert pydoppelgangerhunt.extract_unit_namespace("auth/views.py") == "auth"
    assert pydoppelgangerhunt.extract_unit_namespace("billing/helpers.py") == "billing"
    assert pydoppelgangerhunt.extract_unit_namespace("standalone.py") == "."
    assert pydoppelgangerhunt.extract_unit_namespace("./nested/dir/mod.py") == "nested/dir"

    # Verify namespaced structural fingerprints
    ns_auth = pydoppelgangerhunt.namespaced_structural_fingerprint(u_auth1, u_auth2)
    ns_bill = pydoppelgangerhunt.namespaced_structural_fingerprint(u_bill1, u_bill2)
    assert ns_auth != ns_bill
    assert "auth#hash_boilerplate_common" in ns_auth
    assert "billing#hash_boilerplate_common" in ns_bill

    # Pure structural fingerprint is identical
    pure_auth = pydoppelgangerhunt.pure_structural_fingerprint(u_auth1, u_auth2)
    pure_bill = pydoppelgangerhunt.pure_structural_fingerprint(u_bill1, u_bill2)
    assert pure_auth == pure_bill

    # Record baseline grandfathering ONLY the auth/ boilerplate pair
    base_file = tmp_path / "baseline_disambig.json"
    pydoppelgangerhunt.record_baseline([(0.98, u_auth1, u_auth2)], str(base_file), "test_ns", 0.90)

    loaded_base = pydoppelgangerhunt.load_baseline(str(base_file))

    # Test 1: Scan both clone pairs -> auth is suppressed, billing is NOT (detected as new)
    active_clones = [
        (0.98, u_auth1, u_auth2),
        (0.98, u_bill1, u_bill2),
    ]
    new_clones, suppressed = pydoppelgangerhunt.filter_clones_by_baseline(active_clones, loaded_base)
    assert suppressed == 1
    assert len(new_clones) == 1
    assert new_clones[0][1]["file"] == "billing/views.py"

    # Test 2: Even if auth/ pair is absent, billing/ cannot hijack auth's grandfathered record
    active_bill_only = [(0.98, u_bill1, u_bill2)]
    new_clones_bill, suppressed_bill = pydoppelgangerhunt.filter_clones_by_baseline(active_bill_only, loaded_base)
    assert suppressed_bill == 0
    assert len(new_clones_bill) == 1

    # Test 3: Granular 1-to-1 matching within the same namespace
    u_auth3 = {"file": "auth/extra.py", "name": "to_dict", "structural_hash": bp_hash}
    u_auth4 = {"file": "auth/other.py", "name": "to_dict", "structural_hash": bp_hash}
    two_auth_pairs = [
        (0.98, u_auth1, u_auth2),
        (0.98, u_auth3, u_auth4),
    ]
    new_two, supp_two = pydoppelgangerhunt.filter_clones_by_baseline(two_auth_pairs, loaded_base)
    assert supp_two == 1
    assert len(new_two) == 1
    assert new_two[0][1]["file"] == "auth/extra.py"

    # Test 4: File rename resilience within the same namespace
    u_auth1_renamed = {"file": "auth/auth_views.py", "name": "to_dict", "structural_hash": bp_hash}
    renamed_active = [(0.98, u_auth1_renamed, u_auth2)]
    new_renamed, supp_renamed = pydoppelgangerhunt.filter_clones_by_baseline(renamed_active, loaded_base)
    assert supp_renamed == 1
    assert len(new_renamed) == 0

def test_stale_unstaged_diff_protection_during_baseline_pruning(tmp_path: Path) -> None:
    """Test that prune_baseline protects inactive entries touching unstaged git changes."""
    u_auth1 = {"file": "src/auth/views.py", "name": "login", "structural_hash": "hash_auth_login"}
    u_auth2 = {"file": "src/auth/helpers.py", "name": "login", "structural_hash": "hash_auth_login"}
    u_clean1 = {"file": "src/calc/math.py", "name": "add", "structural_hash": "hash_clean_add"}
    u_clean2 = {"file": "src/calc/utils.py", "name": "add", "structural_hash": "hash_clean_add"}

    base_file = tmp_path / "prune_dirty_baseline.json"
    pydoppelgangerhunt.record_baseline(
        [(0.95, u_auth1, u_auth2), (0.95, u_clean1, u_clean2)],
        str(base_file),
        "test_prune",
        0.90,
    )

    active_clones: List[Any] = []
    dirty_unstaged = {"src/auth/views.py": [(10, 20)]}

    prune_res = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_clones,
        unstaged_modified_ranges=dirty_unstaged,
    )
    assert isinstance(prune_res, tuple)
    pruned, retained = prune_res
    assert pruned == 1
    assert retained == 1
    assert prune_res.skipped_dirty_count == 1
    assert prune_res.pruned_count == 1
    assert prune_res.retained_count == 1

    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert data["clone_count"] == 1
    assert data["fingerprints"][0]["file_a"] == "src/auth/views.py"

    # Clean worktree prunes the remaining entry
    prune_res_clean = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_clones,
        unstaged_modified_ranges={},
    )
    assert prune_res_clean.pruned_count == 1
    assert prune_res_clean.retained_count == 0
    assert prune_res_clean.skipped_dirty_count == 0

    data_clean = json.loads(base_file.read_text(encoding="utf-8"))
    assert data_clean["clone_count"] == 0
    assert len(data_clean["fingerprints"]) == 0

def test_cli_prune_baseline_with_unstaged_diff(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """Test CLI output for --prune-baseline when unstaged git changes protect entries."""
    src_dir = tmp_path / "cli_dirty"
    src_dir.mkdir()
    f1 = src_dir / "mod1.py"
    f2 = src_dir / "mod2.py"
    code = "def compute_val(x, y):\n    res = x * y\n    print(res)\n    return res + 1\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    cli_base = tmp_path / "cli_prune_test.json"
    pydoppelgangerhunt.main([
        str(src_dir),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--record-baseline", str(cli_base),
    ])
    assert cli_base.exists()

    # Now mod2 has different function, mod1 has unstaged changes
    f2.write_text("def different_function():\n    return 42\n", encoding="utf-8")
    norm_f1 = str(f1.resolve()).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff.get_git_modified_line_ranges",
        lambda since_ref=None, repo_root=None: {norm_f1: [(1, 5)]},
    )

    exit_code = pydoppelgangerhunt.main([
        str(src_dir),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--baseline", str(cli_base),
        "--prune-baseline",
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "skipped due to unstaged git changes" in captured

def test_defensive_unit_start_end_none_handling(tmp_path: Path) -> None:
    """Verifies that units with None start or end lines do not crash with TypeError."""
    from pydoppelgangerhunt.baseline import clone_pair_fingerprint, record_baseline
    from pydoppelgangerhunt.clustering import unit_key
    from pydoppelgangerhunt.fixer import check_units_overlap, filter_overlapping_clone_units, refactor_module_units

    u_null_bounds: Dict[str, Any] = {
        "file": str(tmp_path / "mod.py"),
        "start": None,
        "end": None,
        "name": None,
    }
    u_null_bounds2: Dict[str, Any] = {
        "file": str(tmp_path / "mod.py"),
        "start": None,
        "end": None,
        "name": None,
    }

    # Should not raise TypeError: int() argument must be ... not 'NoneType'
    key = unit_key(u_null_bounds)
    assert ":1-1:unit" in key

    assert check_units_overlap(u_null_bounds, u_null_bounds2) is True

    retained = filter_overlapping_clone_units([u_null_bounds, u_null_bounds2])
    assert len(retained) == 1

    fp = clone_pair_fingerprint(u_null_bounds, u_null_bounds2)
    assert "unit" in fp

    base_file = str(tmp_path / "base.json")
    saved = record_baseline([(1.0, u_null_bounds, u_null_bounds2)], base_file, "target", 0.8)
    assert Path(saved).is_file()

    out = refactor_module_units("x = 1\n", [])
    assert out == "x = 1\n"

def test_batch_32_exhaustive_review_hardening(tmp_path: Path) -> None:
    """Tests Batch 32 edge case fixes across fixer, diff, baseline, and reporters."""
    from pydoppelgangerhunt.fixer import (
        _block_terminates,
        _analyze_block_assignment,
        _walrus_assignment_in_expr,
    )
    from pydoppelgangerhunt.git_diff import get_git_blame_info
    from pydoppelgangerhunt.baseline import _match_clone_record, prune_baseline
    from pydoppelgangerhunt.reporters import generate_html_report
    import unittest.mock as mock

    # 1. _block_terminates on try-except-else terminating branches
    tree_term = ast.parse(
        "try:\n    x = 1\nexcept Exception:\n    return 2\nelse:\n    return 3\n"
    )
    assert _block_terminates(tree_term.body) is True

    tree_nonterm1 = ast.parse(
        "try:\n    return 1\nexcept Exception:\n    pass\n"
    )
    assert _block_terminates(tree_nonterm1.body) is False

    tree_nonterm2 = ast.parse(
        "try:\n    x = 1\nexcept Exception:\n    return 2\nelse:\n    pass\n"
    )
    assert _block_terminates(tree_nonterm2.body) is False

    # 2. _analyze_block_assignment with orelse deletions and unconditional deletion discarding
    # 2a. Deletion in else clause demoted to conditional
    tree_del_else = ast.parse(
        "x = 10\ntry:\n    pass\nexcept Exception:\n    pass\nelse:\n    del x\n"
    )
    d_else, c_else = _analyze_block_assignment(tree_del_else.body)
    assert "x" not in d_else
    assert "x" in c_else

    # 2b. Unconditional deletion in both try and except discarded from definite and conditional
    tree_del_both = ast.parse(
        "x = 10\ntry:\n    del x\nexcept Exception:\n    del x\n"
    )
    d_both, c_both = _analyze_block_assignment(tree_del_both.body)
    assert "x" not in d_both
    assert "x" not in c_both

    # 2c. Unconditional deletion in try with terminating except handler
    tree_del_term = ast.parse(
        "x = 10\ntry:\n    del x\nexcept Exception:\n    return 1\n"
    )
    d_term, c_term = _analyze_block_assignment(tree_del_term.body)
    assert "x" not in d_term
    assert "x" not in c_term

    # 3. _walrus_assignment_in_expr in AST slice bounds
    tree_slice = ast.parse("y = a[(b := 1) : (c := 2)]")
    assert isinstance(tree_slice.body[0], ast.Assign)
    d_sl, c_sl = _walrus_assignment_in_expr(tree_slice.body[0].value)
    assert "b" in (d_sl | c_sl)
    assert "c" in (d_sl | c_sl)

    # 4. get_git_blame_info with out-of-order headers without author line
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command") as mock_git:
        mock_git.return_value = (
            "0000000000000000000000000000000000000001 1 1 1\n"
            "author-time 1700000000\n"
            "summary Initial commit\n"
        )
        blame = get_git_blame_info("test.py", 1, 5)
        assert blame["author"] == "Unknown"
        assert blame["timestamp"] == 1700000000

    # 5. _match_clone_record and prune_baseline with None name_a and name_b values
    c_keys = {
        "fp": "fp",
        "sfp": "sfp",
        "ns_sfp": "ns_sfp",
        "pure_sfp": "pure_sfp",
        "namespaces": ["ns_a", "ns_b"],
        "names": ["name_a", "name_b"],
    }
    unconsumed = [
        {
            "pure_structural_fingerprint": "pure_sfp",
            "hash_a": "h1",
            "hash_b": "h2",
            "name_a": None,
            "name_b": None,
        }
    ]
    # Verify no TypeError '<' not supported between instances of 'NoneType'
    matched = _match_clone_record(c_keys, unconsumed)
    assert matched is not None

    b_file = tmp_path / "baseline.json"
    b_file.write_text(
        '{"fingerprints": [{"pure_structural_fingerprint": "pure_sfp", "hash_a": "h1", "hash_b": "h2", "name_a": null, "name_b": null}]}',
        encoding="utf-8",
    )
    pruned_cnt, retained_cnt = prune_baseline(str(b_file), [])
    assert pruned_cnt == 1
    assert retained_cnt == 0

    # 6. HTML entity escaping in generate_html_report
    u_special_1 = {
        "file": "pkg/foo.py",
        "start": 1,
        "end": 2,
        "name": "<lambda>",
    }
    u_special_2 = {
        "file": "pkg/bar.py",
        "start": 1,
        "end": 2,
        "name": "<listcomp>",
    }
    html_out = generate_html_report([(0.9, u_special_1, u_special_2)], "My <Package>", 0.8)
    assert "&lt;lambda&gt;" in html_out
    assert "&lt;listcomp&gt;" in html_out
    assert "My &lt;Package&gt;" in html_out

def test_batch_38_windows_path_case_insensitivity_and_unicode_resilience(tmp_path: Any, monkeypatch: Any) -> None:
    """Tests Batch 38: Windows case-insensitive path matching, UnicodeDecodeError resilience, and markdown table escaping."""
    import sys
    from pydoppelgangerhunt.baseline import load_baseline, prune_baseline
    from pydoppelgangerhunt.config import (
        find_matching_path_value,
        init_tool_configuration,
        paths_match_boundary,
    )
    from pydoppelgangerhunt.fixer import (
        _is_same_file_path,
        _populate_unit_receiver_metadata,
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.reporters import format_markdown_summary

    # 1. Windows case-insensitive boundary path matching
    monkeypatch.setattr(sys, "platform", "win32")
    assert paths_match_boundary("C:/Project/Sub/Engine.py", "c:/project/sub/engine.py")
    assert paths_match_boundary("C:/Project/Sub/Engine.py", "sub/engine.py")
    assert paths_match_boundary("sub/engine.py", "C:/Project/Sub/Engine.py")
    assert not paths_match_boundary("C:/Project/Sub/Other.py", "sub/engine.py")

    # 2. find_matching_path_value with Windows case insensitivity
    data_map = {"c:/repo/sub/module.py": 42}
    val = find_matching_path_value("C:/Repo/Sub/Module.py", data_map)
    assert val == 42
    val_rel = find_matching_path_value("sub/module.py", data_map)
    assert val_rel == 42
    assert find_matching_path_value("experiments/run#1.ipynb#cell_3", {"experiments/run#1.ipynb": 100}) == 100

    # 3. _is_same_file_path using boundary matching
    assert _is_same_file_path("./Sub/Module.ipynb#cell1", "sub/module.ipynb")
    assert _is_same_file_path("C:/Repo/Sub/Module.py", "sub/module.py")

    # 4. UnicodeDecodeError resilience across file reading
    bad_file = tmp_path / "corrupted.py"
    bad_file.write_bytes(b"\x80\xff\xfe\x00\x01not-utf8")
    u_corrupt = {"file": str(bad_file), "start": 1, "end": 2, "name": "bad"}

    # _populate_unit_receiver_metadata does not raise
    _populate_unit_receiver_metadata(u_corrupt, repo_root=str(tmp_path))
    assert u_corrupt.get("receiver_kind") is None

    # generate_refactoring_patch skips corrupted files gracefully
    patch = generate_refactoring_patch([(1.0, u_corrupt, u_corrupt)], repo_root=str(tmp_path))
    assert patch == ""

    # load_baseline and prune_baseline with non-UTF8 baseline file
    bad_baseline = tmp_path / "bad_baseline.json"
    bad_baseline.write_bytes(b"\xff\xfe\x00corrupt")
    base_fps = load_baseline(str(bad_baseline))
    assert len(base_fps) == 0

    pruned = prune_baseline(str(bad_baseline), [])
    assert pruned.pruned_count == 0

    # init_tool_configuration with non-UTF8 pyproject.toml leaves file untouched and uses standalone config
    bad_pyproject = tmp_path / "pyproject.toml"
    bad_bytes = b"\x80\xff[project]\nname='bad'\n"
    bad_pyproject.write_bytes(bad_bytes)
    res_cfg = init_tool_configuration(str(tmp_path))
    assert bad_pyproject.read_bytes() == bad_bytes
    assert ".pydoppelgangerhunt.toml" in res_cfg

    # 5. Markdown table escaping for pipe characters in package names
    stats_with_pipe = {
        "dry_score": 95.0,
        "grade": "A",
        "sloc": 1000,
        "dloc": 50,
        "duplication_pct": 5.0,
        "clone_pairs": 1,
        "clone_families": 1,
        "package_sloc": {"pkg|with|pipes": 500, "normal_pkg": 500},
    }
    summary = format_markdown_summary(stats_with_pipe, "test_repo")
    assert "| `pkg\\|with\\|pipes` | 500 |" in summary
    assert "| `normal_pkg` | 500 |" in summary

def test_batch_40_notebook_source_code_and_matcher_path_robustness(tmp_path: Any) -> None:
    """Tests Batch 40: notebook cell source extraction, clone diffing, and matcher/baseline path equivalence."""
    import json
    from pydoppelgangerhunt.baseline import (
        clone_pair_fingerprint,
        clone_pair_structural_fingerprint,
        extract_unit_namespace,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import merge_adjacent_clones, suppress_subclones
    from pydoppelgangerhunt.reporters import (
        extract_unit_source_code,
        format_sarif_report,
        generate_clone_diff,
    )

    # 1. extract_unit_source_code and generate_clone_diff from Jupyter Notebook code cells
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["def calc(x):\n", "    y = x * 2\n", "    return y\n"],
            },
            {
                "cell_type": "code",
                "source": ["def calc(x):\n", "    y = x * 3\n", "    return y\n"],
            },
        ]
    }
    nb_path = tmp_path / "notebook.ipynb"
    nb_path.write_text(json.dumps(nb_content), encoding="utf-8")

    u1 = {"file": f"{nb_path}#cell_1", "start": 1, "end": 3, "name": "calc"}
    u2 = {"file": f"{nb_path}#cell_2", "start": 1, "end": 3, "name": "calc"}

    lines1 = extract_unit_source_code(u1, repo_root=str(tmp_path))
    assert lines1 == ["def calc(x):\n", "    y = x * 2\n", "    return y\n"]

    diff_text = generate_clone_diff(u1, u2, repo_root=str(tmp_path))
    assert "-    y = x * 2" in diff_text
    assert "+    y = x * 3" in diff_text

    # Out-of-bounds cell index fallback
    u_oob = {"file": f"{nb_path}#cell_99", "start": 1, "end": 3, "name": "oob"}
    lines_oob = extract_unit_source_code(u_oob, repo_root=str(tmp_path))
    assert lines_oob == []


    # 2. format_sarif_report normalizes leading dot-slash while retaining cell anchors
    u_dot_cell = {"file": "./notebook.ipynb#cell_1", "start": 1, "end": 3, "name": "calc"}
    sarif = format_sarif_report([(1.0, u_dot_cell, u_dot_cell)], "repo", 0.9)
    uri = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri == "notebook.ipynb#cell_1"

    # 3. merge_adjacent_clones consolidates hits across dot-slash file representations
    u1_w1 = {"file": "./mod.py", "name": "f:w1", "start": 10, "end": 15, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}
    u2_w1 = {"file": "mod.py", "name": "g:w1", "start": 30, "end": 35, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}
    u1_w2 = {"file": "mod.py", "name": "f:w2", "start": 11, "end": 16, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}
    u2_w2 = {"file": "./mod.py", "name": "g:w2", "start": 31, "end": 36, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}

    merged = merge_adjacent_clones([(1.0, u1_w1, u2_w1), (1.0, u1_w2, u2_w2)], line_tolerance=2)
    assert len(merged) == 1
    assert merged[0][1]["start"] == 10
    assert merged[0][1]["end"] == 16

    # 4. suppress_subclones eliminates redundant sub-clones across dot-slash file representations
    u_parent1 = {"file": "./worker.py", "name": "worker", "start": 1, "end": 50, "token_count": 100}
    u_parent2 = {"file": "worker.py", "name": "worker2", "start": 1, "end": 50, "token_count": 100}
    u_child1 = {"file": "worker.py", "name": "worker:w1", "start": 10, "end": 25, "token_count": 30}
    u_child2 = {"file": "./worker.py", "name": "worker2:w1", "start": 10, "end": 25, "token_count": 30}

    suppressed = suppress_subclones([(1.0, u_parent1, u_parent2), (0.98, u_child1, u_child2)])
    assert len(suppressed) == 1
    assert suppressed[0][1]["name"] == "worker"

    # 5. baseline fingerprint and recording path normalization
    u_b1 = {"file": "./core/engine.py", "start": 5, "end": 15, "name": "run", "tokens": ["a", "b"]}
    u_b2 = {"file": "core/engine.py", "start": 5, "end": 15, "name": "run", "tokens": ["a", "b"]}
    assert clone_pair_fingerprint(u_b1, u_b1) == clone_pair_fingerprint(u_b2, u_b2)
    assert clone_pair_structural_fingerprint(u_b1, u_b1) == clone_pair_structural_fingerprint(u_b2, u_b2)
    assert extract_unit_namespace("./core/engine.py") == "core"

    base_json = tmp_path / "baseline.json"
    record_baseline([(1.0, u_b1, u_b2)], str(base_json), "repo", 0.9)
    base_data = json.loads(base_json.read_text(encoding="utf-8"))
    assert base_data["fingerprints"][0]["file_a"] == "core/engine.py"
    assert base_data["fingerprints"][0]["file_b"] == "core/engine.py"

def test_batch_47_git_diff_metrics_matcher_clustering(tmp_path: Path) -> None:
    """Batch 47: Test git diff hunk parsing, blame parsing, UnionFind rank orders,

    metrics relative package path resolution, matcher edge cases, and coverage readers.
    """
    # pylint: disable=protected-access,import-outside-toplevel
    import sqlite3
    from pydoppelgangerhunt.baseline import clone_pair_fingerprint, prune_baseline
    from pydoppelgangerhunt.coverage import _read_sqlite_coverage, _read_xml_coverage
    from pydoppelgangerhunt.git_diff import (
        _run_git_command,
        get_git_blame_info,
        get_git_modified_line_ranges,
    )
    from pydoppelgangerhunt.matcher import (
        _split_exemption_endpoint,
        call_sequence_similarity,
        compute_pair_similarity,
        jaccard_similarity,
        merge_adjacent_clones,
        tfidf_multiset_jaccard_similarity,
    )

    # 1. git_diff: parse_git_diff_hunks single-line hunks and deleted-file bleed prevention
    diff_data = (
        "--- a/file_one.py\n"
        "+++ b/file_one.py\n"
        "@@ -10 +10 @@\n"
        "+added_single_line\n"
        "--- a/file_deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,5 +0,0 @@\n"
        "-deleted_line\n"
        "--- a/file_two.py\n"
        "+++ b/file_two.py\n"
        "@@ -20,2 +20,2 @@\n"
        "+line20\n+line21\n"
    )
    parsed_hunks = parse_git_diff_hunks(diff_data)
    assert parsed_hunks.get("file_one.py") == [(10, 10)]
    assert "file_deleted.py" not in parsed_hunks
    assert parsed_hunks.get("file_two.py") == [(20, 21)]

    # 2. git_diff: _run_git_command exception safety and range queries
    with mock.patch("subprocess.run", side_effect=OSError("binary failure")):
        assert _run_git_command(["status"]) is None

    mock_diff = "+++ b/target_mod.py\n@@ -5,3 +5,3 @@\n"
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=mock_diff):
        ranges = get_git_modified_line_ranges(since_ref="HEAD~1")
        assert "target_mod.py" in ranges

    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=None):
        assert get_git_modified_line_ranges() == {}

    # 3. git_diff: filter_clones_by_git_diff empty ranges & both_units flag
    unit_alpha = {"file": "mod_a.py", "start": 5, "end": 15}
    unit_beta = {"file": "mod_b.py", "start": 5, "end": 15}
    clone_candidate = [(0.92, unit_alpha, unit_beta)]
    assert filter_clones_by_git_diff(clone_candidate, {}) == []

    matched_ranges = {"mod_a.py": [(5, 10)], "mod_b.py": [(5, 10)]}
    both_filtered = filter_clones_by_git_diff(
        clone_candidate, matched_ranges, both_units=True
    )
    assert len(both_filtered) == 1

    # 4. git_diff: get_git_blame_info empty file and invalid timestamp handling
    blame_empty = get_git_blame_info("", 1, 5)
    assert blame_empty["author"] == "Unknown"

    porcelain_sample = (
        "fedcba9876543210fedcba9876543210fedcba98 1 1 1\n"
        "author Dev Master\n"
        "author-time not_a_valid_integer\n"
        "summary Patch update\n"
        "\tval = 42\n"
    )
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=porcelain_sample):
        blame_res = get_git_blame_info("mod_a.py", 1, 1)
        assert blame_res["commit"] == "fedcba98"
        assert blame_res["author"] == "Dev Master"
        assert blame_res["timestamp"] == 0

    # 5. UnionFind: rank orders and self-union branch coverage
    uf = UnionFind()
    assert uf.find("item_solo") == "item_solo"
    root_1 = uf.union("elem_a", "elem_b")
    root_2 = uf.union("elem_c", "elem_d")
    assert uf.union("elem_e", "elem_c") == root_2
    assert uf.union("elem_c", "elem_f") == root_2
    assert uf.union("elem_a", "elem_a") == root_1

    # 6. clustering: tolerance-floor rejection and medoid sub-threshold rejection
    cand_1 = {"file": "c1.py", "name": "f1", "start": 1, "end": 10}
    cand_2 = {"file": "c2.py", "name": "f2", "start": 1, "end": 10}
    fams_tol = cluster_clone_families(
        [(0.77, cand_1, cand_2)],
        linkage="complete",
        min_similarity_floor=0.80,
        linkage_tolerance=0.05,
    )
    assert len(fams_tol) == 0

    cand_3 = {"file": "c3.py", "name": "f3", "start": 1, "end": 10}
    # In medoid linkage, cand_3 cannot merge with {cand_1, cand_2} because sim(cand_2, cand_3) < floor
    fams_med = cluster_clone_families(
        [(0.95, cand_1, cand_2), (0.75, cand_2, cand_3)],
        linkage="medoid",
        min_similarity_floor=0.80,
    )
    assert len(fams_med) == 1
    assert fams_med[0]["member_count"] == 2

    # 7. metrics: package_sloc relative to target_dir and OSError handling
    stat_root = tmp_path / "repo_metrics"
    svc_sub = stat_root / "services_pkg"
    svc_sub.mkdir(parents=True)
    (svc_sub / "handler.py").write_text("def run_job():\n    return 100\n", encoding="utf-8")
    (stat_root / "entrypoint.py").write_text("import sys\n", encoding="utf-8")

    stats = compute_repository_dry_stats(str(stat_root), [])
    assert "services_pkg" in stats["package_sloc"]

    with mock.patch("builtins.open", side_effect=OSError("Disk read failure")):
        stats_error = compute_repository_dry_stats(str(stat_root), [])
        assert stats_error["sloc"] == 0

    # 8. matcher: empty fallbacks, exemptions, and alternative similarity modes
    assert jaccard_similarity(set(), set()) == 0.0
    assert call_sequence_similarity([], []) == 0.0
    assert tfidf_multiset_jaccard_similarity({}, {}, {}) == 0.0
    assert merge_adjacent_clones([]) == []

    assert _split_exemption_endpoint("no_colon_path") == ("no_colon_path", None)
    assert _split_exemption_endpoint("file.py:sym") == ("file.py", "sym")

    unit_trace_1 = {"calls": ["step_a", "step_b", "step_c"]}
    unit_trace_2 = {"calls": ["step_a", "step_b", "step_c"]}
    assert compute_pair_similarity(unit_trace_1, unit_trace_2, call_sequences=True) == 1.0

    unit_vec_1 = {"vector": {"tok_x": 3, "tok_y": 1}, "shingles": set()}
    unit_vec_2 = {"vector": {"tok_x": 3, "tok_y": 1}, "shingles": set()}
    idf_map = {"tok_x": 1.2, "tok_y": 1.8}
    assert compute_pair_similarity(
        unit_vec_1, unit_vec_2, tfidf=True, bag_of_tokens=True, idf_weights=idf_map
    ) == 1.0

    # 9. baseline: plain set consumable suppression and prune error recovery
    u_base_1 = {"file": "b1.py", "name": "fn1", "tokens": ["tok_a", "tok_b"], "start": 1, "end": 10}
    u_base_2 = {"file": "b2.py", "name": "fn2", "tokens": ["tok_a", "tok_b"], "start": 1, "end": 10}
    fp_base = clone_pair_fingerprint(u_base_1, u_base_2)
    plain_fp_set = {fp_base}
    rem, supp = filter_clones_by_baseline([(0.95, u_base_1, u_base_2)], plain_fp_set)
    assert supp == 1
    assert len(rem) == 0

    with mock.patch(
        "pydoppelgangerhunt.git_diff.get_git_modified_line_ranges",
        side_effect=RuntimeError("git failure"),
    ):
        base_doc = {"fingerprints": [99999, "legacy_raw_str", {"fingerprint": "dict_raw"}]}
        base_file = tmp_path / "mock_baseline.json"
        base_file.write_text(json.dumps(base_doc), encoding="utf-8")
        prune_res = prune_baseline(str(base_file), active_clones=[])
        assert isinstance(prune_res, tuple)

    # 10. coverage: SQLite bit bounds & unmapped IDs, and XML parse exceptions
    db_cov = tmp_path / ".coverage_bit_bounds"
    db_conn = sqlite3.connect(str(db_cov))
    db_c = db_conn.cursor()
    db_c.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    db_c.execute("CREATE TABLE line_bits (file_id INTEGER, num_bits INTEGER, bits BLOB)")
    db_c.execute("INSERT INTO file VALUES (1, 'recorded.py')")
    db_c.execute("INSERT INTO line_bits VALUES (1, 2, ?)", (b"\xff",))
    db_c.execute("INSERT INTO line_bits VALUES (888, 5, ?)", (b"\xff",))
    db_conn.commit()
    db_conn.close()

    cov_result = _read_sqlite_coverage(str(db_cov))
    assert cov_result.get("recorded.py") == {1, 2}

    xml_cov = tmp_path / "cov_malformed.xml"
    xml_cov.write_text(
        "<coverage><project><package><classes><class filename='m.py'>"
        "<lines><line number='not_a_num' hits='1'/></lines></class></classes></package></project></coverage>",
        encoding="utf-8",
    )
    assert _read_xml_coverage(str(xml_cov)) == {"m.py": set()}
    xml_cov.write_text("<<<NOT_XML>>>", encoding="utf-8")
    assert _read_xml_coverage(str(xml_cov)) == {}

def test_batch_63_multiline_toml_and_baseline_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests multi-line TOML array parsing in zero-dep fallback, baseline structural reconstruction, and record disambiguation."""
    # pylint: disable=import-outside-toplevel
    import builtins
    from pydoppelgangerhunt.baseline import (
        _match_clone_record,
        _record_matches_names,
        load_baseline,
        prune_baseline,
    )
    from pydoppelgangerhunt.config import _parse_toml_array_value, load_toml_section

    # 1. Direct _parse_toml_array_value validation
    assert _parse_toml_array_value("['alpha', 'beta']") == ["alpha", "beta"]
    assert _parse_toml_array_value("[ 10, 2.5, true, false, 'text' ]") == [
        10,
        2.5,
        True,
        False,
        "text",
    ]
    assert not _parse_toml_array_value("")

    # 2. Multi-line TOML array parsing via zero-dependency line fallback
    multiline_toml = tmp_path / "multiline.toml"
    multiline_toml.write_text(
        """[tool.pydoppelgangerhunt]
threshold = 0.88 # inline comment
exclude = [
    "venv", # comment 1
    ".venv",
    "build",
    "dist",
]
flags = [
    true,
    false,
]
values = [ 10, 20, 30 ]
""",
        encoding="utf-8",
    )
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomli", "tomllib"):
            raise ImportError("mocked no toml parser")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    cfg = load_toml_section(multiline_toml, "pydoppelgangerhunt")
    assert cfg.get("threshold") == 0.88
    assert cfg.get("exclude") == ["venv", ".venv", "build", "dist"]
    assert cfg.get("flags") == [True, False]
    assert cfg.get("values") == [10, 20, 30]
    monkeypatch.undo()

    # 3. Baseline structural fingerprint reconstruction in load_baseline and prune_baseline
    base_json = tmp_path / "legacy_base.json"
    base_json.write_text(
        json.dumps({
            "version": "1.3.0",
            "fingerprints": [
                {
                    "file_a": "pkg/mod_a.py",
                    "name_a": "compute",
                    "hash_a": "aaaa111122223333",
                    "file_b": "pkg/mod_b.py",
                    "name_b": "calculate",
                    "hash_b": "bbbb111122223333",
                }
            ],
        }),
        encoding="utf-8",
    )
    loaded_base = load_baseline(str(base_json))
    assert len(loaded_base.records) == 1
    rec = loaded_base.records[0]
    expected_sfp = "pkg/mod_a.py#aaaa111122223333 <===> pkg/mod_b.py#bbbb111122223333"
    assert rec.get("structural_fingerprint") == expected_sfp
    assert expected_sfp in loaded_base

    u1 = {"file": "pkg/mod_a.py", "name": "new_compute", "structural_hash": "aaaa111122223333"}
    u2 = {"file": "pkg/mod_b.py", "name": "new_calc", "structural_hash": "bbbb111122223333"}
    prune_res = prune_baseline(str(base_json), [(0.95, u1, u2)], unstaged_modified_ranges={})
    assert prune_res.retained_count == 1
    assert prune_res.pruned_count == 0

    # 4. _match_clone_record Pass 1 disambiguation prioritizing matching structural hash
    rec_stale = {
        "fingerprint": "service.py:worker <===> client.py:worker",
        "structural_fingerprint": "service.py#old1 <===> client.py#old2",
        "name_a": "worker",
        "name_b": "worker",
    }
    rec_fresh = {
        "fingerprint": "service.py:worker <===> client.py:worker",
        "structural_fingerprint": "service.py#new1 <===> client.py#new2",
        "name_a": "worker",
        "name_b": "worker",
    }
    keys_p1 = {
        "fp": "service.py:worker <===> client.py:worker",
        "sfp": "service.py#new1 <===> client.py#new2",
        "ns_sfp": "dummy",
        "pure_sfp": "dummy",
        "namespaces": [".", "."],
        "names": ["worker", "worker"],
    }
    matched_p1 = _match_clone_record(keys_p1, [rec_stale, rec_fresh])
    assert matched_p1 is rec_fresh

    # 5. _match_clone_record Pass 3/4 disambiguation prioritizing symbol names
    rec_diff_names = {
        "namespaced_structural_fingerprint": "pkg#h1 <===> pkg#h2",
        "pure_structural_fingerprint": "h1 <===> h2",
        "namespace_a": "pkg",
        "namespace_b": "pkg",
        "name_a": "alpha",
        "name_b": "beta",
    }
    rec_same_names = {
        "namespaced_structural_fingerprint": "pkg#h1 <===> pkg#h2",
        "pure_structural_fingerprint": "h1 <===> h2",
        "namespace_a": "pkg",
        "namespace_b": "pkg",
        "name_a": "worker",
        "name_b": "worker",
    }
    assert _record_matches_names(rec_same_names, ["worker", "worker"]) is True
    assert _record_matches_names(rec_diff_names, ["worker", "worker"]) is False

    keys_ns = {
        "fp": "nomatch",
        "sfp": "nomatch",
        "ns_sfp": "pkg#h1 <===> pkg#h2",
        "pure_sfp": "h1 <===> h2",
        "namespaces": ["pkg", "pkg"],
        "names": ["worker", "worker"],
    }
    matched_ns = _match_clone_record(keys_ns, [rec_diff_names, rec_same_names])
    assert matched_ns is rec_same_names

def test_batch_64_control_flow_hazards_and_baseline_pruning(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch declines units with naked loop controls,

    and prune_baseline correctly uses sfp_to_clone and name disambiguation across identical hashes.
    """
    from pydoppelgangerhunt.baseline import prune_baseline  # pylint: disable=import-outside-toplevel

    # 1. Test generate_refactoring_patch with naked break and naked continue
    loop_code = (
        "def process_items(items: list) -> None:\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            break\n"
        "        if x == 0:\n"
        "            continue\n"
        "        print(x)\n"
    )
    src_file = tmp_path / "proc.py"
    src_file.write_text(loop_code, encoding="utf-8")

    u_break = {
        "file": str(src_file),
        "start": 3,
        "end": 4,
        "name": "process_items:If_break",
        "kind": "compound_block",
    }
    u_continue = {
        "file": str(src_file),
        "start": 5,
        "end": 6,
        "name": "process_items:If_continue",
        "kind": "compound_block",
    }

    # Patch generation should decline (return empty string) for naked_break and naked_continue
    patch_break = generate_refactoring_patch([(0.95, u_break, u_break)], repo_root=str(tmp_path))
    assert patch_break == ""

    patch_continue = generate_refactoring_patch([(0.95, u_continue, u_continue)], repo_root=str(tmp_path))
    assert patch_continue == ""

    # 2. Test prune_baseline sfp_to_clone and name-disambiguated ns_sfp_to_clones
    base_json = tmp_path / "baseline_multi.json"
    data = {
        "version": "1.3.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "target": ".",
        "threshold": 0.8,
        "clone_count": 2,
        "fingerprints": [
            {
                "file_a": "pkg/mod1.py",
                "name_a": "alpha",
                "hash_a": "1111222233334444",
                "file_b": "pkg/mod1.py",
                "name_b": "beta",
                "hash_b": "1111222233334444",
            },
            {
                "file_a": "pkg/mod2.py",
                "name_a": "gamma",
                "hash_a": "1111222233334444",
                "file_b": "pkg/mod2.py",
                "name_b": "delta",
                "hash_b": "1111222233334444",
            },
        ],
    }
    base_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Active clones: pair in mod1 was renamed from beta -> beta_renamed, while mod2 pair is active as-is
    u_m1_a = {"file": "pkg/mod1.py", "name": "alpha", "structural_hash": "1111222233334444"}
    u_m1_b = {"file": "pkg/mod1.py", "name": "beta_renamed", "structural_hash": "1111222233334444"}
    u_m2_a = {"file": "pkg/mod2.py", "name": "gamma", "structural_hash": "1111222233334444"}
    u_m2_b = {"file": "pkg/mod2.py", "name": "delta", "structural_hash": "1111222233334444"}

    active = [
        (0.95, u_m1_a, u_m1_b),
        (0.95, u_m2_a, u_m2_b),
    ]
    res = prune_baseline(str(base_json), active, unstaged_modified_ranges={})
    assert res.retained_count == 2
    assert res.pruned_count == 0

    saved_data = json.loads(base_json.read_text(encoding="utf-8"))
    recs = saved_data["fingerprints"]
    # mod1 record should be updated to beta_renamed, without adopting mod2's file path
    m1_rec = next(r for r in recs if r["file_a"] == "pkg/mod1.py")
    assert m1_rec["name_b"] == "beta_renamed"
    assert m1_rec["file_b"] == "pkg/mod1.py"

    m2_rec = next(r for r in recs if r["file_a"] == "pkg/mod2.py")
    assert m2_rec["name_b"] == "delta"
    assert m2_rec["file_b"] == "pkg/mod2.py"

def test_batch_82_baseline_hash_unpacking_and_prune_synchronization(tmp_path: Path) -> None:
    """Verifies that load_baseline unpacks pure structural hashes and prune_baseline synchronizes self-healed records."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        load_baseline,
        prune_baseline,
    )

    # 1. Baseline file with pure_structural_fingerprint but missing explicit hash_a and hash_b
    b_file = tmp_path / "baseline_legacy.json"
    b_file.write_text(
        json.dumps({
            "version": "1.2.0",
            "fingerprints": [
                {
                    "fingerprint": "a.py:f1 <===> b.py:f2",
                    "pure_structural_fingerprint": "1111222233334444 <===> 5555666677778888",
                }
            ],
        }),
        encoding="utf-8",
    )
    loaded = load_baseline(str(b_file))
    rec = loaded.records[0]
    assert rec["hash_a"] == "1111222233334444"
    assert rec["hash_b"] == "5555666677778888"

    # 2. Self-healing in prune_baseline synchronizes pure_structural_fingerprint, hash_a, and hash_b
    from pydoppelgangerhunt.baseline import clone_pair_structural_fingerprint  # pylint: disable=import-outside-toplevel

    u1 = {"file": "mod1.py", "name": "f1_renamed", "tokens": ["x", "+", "1"]}
    u2 = {"file": "mod2.py", "name": "f2_renamed", "tokens": ["y", "+", "2"]}
    sfp_orig = clone_pair_structural_fingerprint(u1, u2)
    b_file2 = tmp_path / "baseline_healing.json"
    b_file2.write_text(
        json.dumps({
            "version": "1.2.0",
            "fingerprints": [
                {
                    "fingerprint": "old1.py:f1 <===> old2.py:f2",
                    "structural_fingerprint": sfp_orig,
                }
            ],
        }),
        encoding="utf-8",
    )
    active_clones = [(1.0, u1, u2)]

    res = prune_baseline(str(b_file2), active_clones, unstaged_modified_ranges={})
    assert res.retained_count == 1
    updated_data = json.loads(b_file2.read_text(encoding="utf-8"))
    updated_rec = updated_data["fingerprints"][0]
    assert updated_rec["file_a"] == "mod1.py"
    assert "pure_structural_fingerprint" in updated_rec
    assert "hash_a" in updated_rec
    assert "hash_b" in updated_rec


def test_corpus_calibration_compute_and_serialization(tmp_path: Path) -> None:
    """Verifies that compute_corpus_calibration computes accurate distributions and round-trips via baseline v1.4.0."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    # Construct synthetic units with known shingle frequencies
    units: List[Dict[str, Any]] = [
        {"file": f"f{i}.py", "name": f"fn_{i}", "shingles": {("Common", "A"), ("Domain", "B")}}
        for i in range(10)
    ]
    # Add a ubiquitous shingle in 8/10 units (80% > 25%)
    for i in range(8):
        sh_set = units[i]["shingles"]
        assert isinstance(sh_set, set)
        sh_set.add(("Ubiquitous", "Boilerplate"))
    # Add a rare shingle in 2/10 units (20% <= 25%)
    for i in range(2):
        sh_set = units[i]["shingles"]
        assert isinstance(sh_set, set)
        sh_set.add(("Rare", "Shingle"))

    calib = compute_corpus_calibration(units, max_index_frequency=0.25, min_corpus_size=4)
    assert calib["total_units"] == 10
    assert calib["max_index_frequency"] == 0.25
    assert ("Ubiquitous", "Boilerplate") in calib["global_stop_shingles"]
    assert ("Common", "A") in calib["global_stop_shingles"]  # 10/10 = 100% > 25%
    assert ("Rare", "Shingle") not in calib["global_stop_shingles"]  # 2/10 = 20% <= 25%
    assert calib["shingle_frequencies"][("Ubiquitous", "Boilerplate")] == 8
    assert calib["shingle_frequencies"][("Rare", "Shingle")] == 2

    # Round-trip via record_baseline and load_baseline
    base_file = tmp_path / "baseline_calib.json"
    u1 = {"file": "mod.py", "name": "f1", "tokens": ["a"]}
    u2 = {"file": "mod.py", "name": "f2", "tokens": ["b"]}
    saved = record_baseline([(1.0, u1, u2)], str(base_file), "target", 0.90, corpus_calibration=calib)
    assert Path(saved).is_file()

    raw_json = json.loads(base_file.read_text(encoding="utf-8"))
    assert raw_json["version"] in ("1.4.0", "1.5.0")
    assert "corpus_calibration" in raw_json
    assert raw_json["corpus_calibration"]["total_units"] == 10

    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["total_units"] == 10
    assert ("Ubiquitous", "Boilerplate") in loaded_calib["global_stop_shingles"]
    assert loaded_calib["shingle_frequencies"][("Ubiquitous", "Boilerplate")] == 8
    assert loaded_calib["shingle_frequencies"][("Rare", "Shingle")] == 2


def test_differential_scan_calibrated_pruning_prevents_false_negatives(tmp_path: Path) -> None:
    """Verifies that corpus calibration prevents false-negative pruning of domain shingles in small diff runs.

    In a small diff scan of 8 units, 3 units sharing a domain shingle represent 37.5% (> 25%) locally.
    Without calibration, the shingle exceeds max_posting_len = 2 and is falsely pruned, dropping clone candidates.
    With global calibration (1000 units), the combined frequency is 3/1008 (0.3%), preserving the clone candidates.
    """
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    repo.mkdir()

    # Create 8 files, 3 of which share an identical domain function
    shared_code = (
        "def domain_logic_clone(a, b, c):\n"
        "    result = 0\n"
        "    for x in range(a):\n"
        "        if x > b:\n"
        "            result += x * c\n"
        "        else:\n"
        "            result -= x\n"
        "    return result\n"
    )

    (repo / "f1.py").write_text(shared_code, encoding="utf-8")
    (repo / "f2.py").write_text(shared_code, encoding="utf-8")
    (repo / "f3.py").write_text(shared_code, encoding="utf-8")

    # 5 files with distinct un-cloned functions to total 8 units
    for i in range(4, 9):
        code = f"def distinct_func_{i}(val):\n" + "\n".join(f"    v_{j} = val + {j}" for j in range(i)) + "\n    return val\n"
        (repo / f"f{i}.py").write_text(code, encoding="utf-8")

    # Run WITHOUT calibration: with 8 units and max_index_frequency=0.25 (min_corpus_size=4),
    # max_posting_len is max(2, ceil(8 * 0.25)) = 2.
    # The domain shingles in f1, f2, f3 appear 3 times (> 2) and are all pruned as stop shingles!
    clones_uncalibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=6,
        max_index_frequency=0.25,
        min_corpus_size=4,
    )
    assert len(clones_uncalibrated) == 0, "Expected 0 clones due to false-negative local frequency skew"

    # Run WITH global corpus calibration (e.g. 1000 units in full repo)
    calib = {
        "total_units": 1000,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    clones_calibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=6,
        max_index_frequency=0.25,
        min_corpus_size=4,
        corpus_calibration=calib,
    )
    assert len(clones_calibrated) >= 3, f"Expected at least 3 clone pairs detected with calibration, got {len(clones_calibrated)}"


def test_differential_scan_calibrated_pruning_suppresses_global_boilerplate(tmp_path: Path) -> None:
    """Verifies that calibrated global stop shingles suppress ubiquitous repository boilerplate even in small diffs."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_boiler"
    repo.mkdir()

    # Two units that share boilerplate
    boilerplate_code = (
        "def handle_service_call(req):\n"
        "    try:\n"
        "        logger.info('Processing %s', req)\n"
        "        return req.process()\n"
        "    except Exception as exc:\n"
        "        logger.error('Failed: %s', exc)\n"
        "        raise\n"
    )
    (repo / "s1.py").write_text(boilerplate_code, encoding="utf-8")
    (repo / "s2.py").write_text(boilerplate_code, encoding="utf-8")

    # 1. Uncalibrated scan on s1.py and s2.py finds 1 clone pair
    clones_uncalib = scan_target(str(repo), threshold=0.90, min_lines=5)
    assert len(clones_uncalib) == 1

    # Harvest units to collect the boilerplate shingles
    _, calib_harvested = scan_target(str(repo), threshold=0.90, min_lines=5, return_calibration=True)
    all_boilerplate_shingles = set(calib_harvested["shingle_frequencies"].keys())
    assert len(all_boilerplate_shingles) > 0

    # Calibrate with all boilerplate shingles designated as global stop shingles
    calib = {
        "total_units": 500,
        "max_index_frequency": 0.25,
        "min_lines": 5,
        "global_stop_shingles": all_boilerplate_shingles,
        "shingle_frequencies": {sh: 300 for sh in all_boilerplate_shingles},
    }

    # Verify that passing this calibration suppresses candidate pair generation and yields 0 clones
    clones_calibrated = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=5,
        corpus_calibration=calib,
    )
    assert len(clones_calibrated) == 0, "Expected boilerplate clone pair to be completely suppressed by calibrated stop shingles"


def test_baseline_backward_compatibility_v10_to_v13(tmp_path: Path) -> None:
    """Verifies that legacy baseline files (v1.0.0 - v1.3.0) load cleanly with corpus_calibration=None."""
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel

    for version in ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]:
        legacy_file = tmp_path / f"baseline_{version}.json"
        legacy_file.write_text(
            json.dumps({
                "version": version,
                "target": "pkg",
                "threshold": 0.90,
                "fingerprints": [
                    "pkg/a.py:foo <===> pkg/b.py:bar",
                    {
                        "fingerprint": "pkg/c.py:c1 <===> pkg/d.py:d1",
                        "structural_fingerprint": "pkg/c.py#111 <===> pkg/d.py#222",
                    },
                ],
            }),
            encoding="utf-8",
        )
        loaded = load_baseline(str(legacy_file))
        assert loaded.corpus_calibration is None
        assert "pkg/a.py:foo <===> pkg/b.py:bar" in loaded
        assert "pkg/c.py#111 <===> pkg/d.py#222" in loaded


def test_baseline_calibration_edge_cases_and_security(tmp_path: Path) -> None:
    """Verifies security resilience against malformed baseline JSON and edge-case serialization."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        _deserialize_shingle_key,
        load_baseline,
        record_baseline,
    )

    # 1. Non-dict JSON payloads do not crash load_baseline
    for malformed_content in [b"[1, 2, 3]", b'"a string"', b"42", b"true"]:
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_bytes(malformed_content)
        fps = load_baseline(str(malformed_file))
        assert len(fps) == 0
        assert fps.corpus_calibration is None

    # 2. Corrupted fields in corpus_calibration do not crash load_baseline
    corrupted_file = tmp_path / "corrupted_calib.json"
    corrupted_file.write_text(
        json.dumps({
            "version": "1.4.0",
            "corpus_calibration": {
                "total_units": "not_an_int",
                "max_index_frequency": "not_a_float",
                "global_stop_shingles": "not_a_list",
                "shingle_frequencies": {"k": "not_an_int"},
            },
        }),
        encoding="utf-8",
    )
    loaded_corrupt = load_baseline(str(corrupted_file))
    assert loaded_corrupt.corpus_calibration is not None
    assert loaded_corrupt.corpus_calibration["total_units"] == 0
    assert loaded_corrupt.corpus_calibration["max_index_frequency"] == 0.25

    # 3. max_index_frequency=None round-trip
    calib_none_max = {
        "total_units": 100,
        "max_index_frequency": None,
        "global_stop_shingles": {("A", "B"), ("C", "D")},
        "shingle_frequencies": {("A", "B"): 5, ("C", "D"): 10},
    }
    none_max_file = tmp_path / "none_max.json"
    record_baseline([], str(none_max_file), "target", 0.90, corpus_calibration=calib_none_max)
    loaded_none = load_baseline(str(none_max_file))
    assert loaded_none.corpus_calibration is not None
    assert loaded_none.corpus_calibration["max_index_frequency"] is None

    # 4. Nested shingle structure deserialization produces hashable tuples
    nested_res = _deserialize_shingle_key("[[1, 2], [3, 4]]")
    assert isinstance(nested_res, tuple)
    assert isinstance(nested_res[0], tuple)
    assert hash(nested_res) is not None

    # 5. Deterministic sorting in serialized output
    raw_text = none_max_file.read_text(encoding="utf-8")
    loaded_json = json.loads(raw_text)
    stops = loaded_json["corpus_calibration"]["global_stop_shingles"]
    freq_keys = list(loaded_json["corpus_calibration"]["shingle_frequencies"].keys())
    assert stops == sorted(stops, key=str)
    assert freq_keys == sorted(freq_keys)


def test_scan_target_calibration_edge_cases(tmp_path: Path) -> None:
    """Verifies that scan_target correctly handles raw list stop shingles, string keys, and small corpus bounds."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "calib_edge_repo"
    repo.mkdir()

    sample_func = (
        "def duplicated_logic(x, y):\n"
        "    val = x + 1\n"
        "    acc = val * 2\n"
        "    diff = acc - y\n"
        "    total = diff + 4\n"
        "    return total\n"
    )
    (repo / "f1.py").write_text(sample_func, encoding="utf-8")
    (repo / "f2.py").write_text(sample_func, encoding="utf-8")

    # 1. Uncalibrated scan on 2 files finds 1 clone pair
    uncalib_clones = scan_target(str(repo), min_lines=5, threshold=0.90)
    assert len(uncalib_clones) == 1

    # 2. Calibrated scan on small corpus (< effective_min_corpus=30) must NOT falsely prune clones
    small_calib: Dict[str, Any] = {
        "total_units": 2,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    calib_clones = scan_target(str(repo), min_lines=5, threshold=0.90, corpus_calibration=small_calib)
    assert len(calib_clones) == 1, "Small corpus with calibration must not prune valid clones"

    # 3. Handling max_index_frequency=None in corpus_calibration does not raise TypeError
    none_freq_calib: Dict[str, Any] = {
        "total_units": 2,
        "max_index_frequency": None,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }
    none_clones = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        max_index_frequency=None,
        corpus_calibration=none_freq_calib,
    )
    assert len(none_clones) == 1

    # 4. Raw list-of-lists in global_stop_shingles does not raise unhashable type: 'list'
    list_stops_calib: Dict[str, Any] = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "global_stop_shingles": [["Name", "Assign"]],
        "shingle_frequencies": {'["Name", "Assign"]': 50},
    }
    list_stop_clones = scan_target(str(repo), min_lines=5, threshold=0.90, corpus_calibration=list_stops_calib)
    assert isinstance(list_stop_clones, list)


def test_prune_baseline_preserves_corpus_calibration(tmp_path: Path) -> None:
    """Verifies that prune_baseline preserves corpus_calibration metadata in the baseline file."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        load_baseline,
        prune_baseline,
        record_baseline,
    )

    base_file = tmp_path / "baseline_for_prune.json"
    calib = {
        "total_units": 42,
        "max_index_frequency": 0.20,
        "global_stop_shingles": {("Stop", "A")},
        "shingle_frequencies": {("Stop", "A"): 15},
    }
    u1 = {"file": "mod1.py", "name": "fn1", "tokens": ["x", "1"]}
    u2 = {"file": "mod2.py", "name": "fn2", "tokens": ["y", "2"]}
    record_baseline([(1.0, u1, u2)], str(base_file), "target", 0.90, corpus_calibration=calib)

    # Prune with active clones retaining the entry
    prune_res = prune_baseline(str(base_file), [(1.0, u1, u2)], unstaged_modified_ranges={})
    assert prune_res.retained_count == 1

    # Load baseline and verify corpus_calibration is intact
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] == 42
    assert loaded.corpus_calibration["max_index_frequency"] == 0.20
    assert ("Stop", "A") in loaded.corpus_calibration["global_stop_shingles"]
    assert loaded.corpus_calibration["shingle_frequencies"][("Stop", "A")] == 15


def test_prune_baseline_synchronizes_calibration_scope_and_hash(tmp_path: Path) -> None:
    """Verifies that prune_baseline synchronizes corpus_calibration scope and recomputes config_hash."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_calibration_config_hash,
        load_baseline,
        prune_baseline,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "a.py"
    f2 = src / "b.py"
    f1.write_text("def run():\n    pass\n", encoding="utf-8")
    f2.write_text("def run():\n    pass\n", encoding="utf-8")

    u1 = {"file": "a.py", "name": "run"}
    u2 = {"file": "b.py", "name": "run"}
    base_file = tmp_path / "pruned_calib.json"

    # Baseline file where corpus_calibration initially lacked scope
    initial_calib: Dict[str, Any] = {
        "total_units": 10,
        "max_index_frequency": 0.25,
        "scope": None,
        "target_repo_relative": None,
    }
    initial_hash = compute_calibration_config_hash(initial_calib)
    initial_calib["config_hash"] = initial_hash

    raw_data = {
        "version": "1.4.0",
        "target": str(src),
        "target_repo_relative": "src",
        "path_basis": "target_relative",
        "clone_count": 1,
        "fingerprints": [
            {
                "file_a": "a.py",
                "file_b": "b.py",
                "name_a": "run",
                "name_b": "run",
                "fingerprint": "fp1",
                "structural_fingerprint": "sfp1",
            }
        ],
        "corpus_calibration": initial_calib,
        "config_hash": initial_hash,
    }
    base_file.write_text(json.dumps(raw_data, indent=2), encoding="utf-8")

    # Prune with active clones
    prune_res = prune_baseline(
        str(base_file),
        [(1.0, u1, u2)],
        unstaged_modified_ranges={},
        target=str(src),
        repo_root=str(repo),
    )
    assert prune_res.retained_count == 1

    # Check persisted file on disk directly
    saved_data = json.loads(base_file.read_text(encoding="utf-8"))
    saved_calib = saved_data["corpus_calibration"]
    assert saved_calib["scope"] == "src"
    assert saved_calib["target_repo_relative"] == "src"
    expected_hash = compute_calibration_config_hash(saved_calib)
    assert saved_calib["config_hash"] == expected_hash
    assert saved_data["config_hash"] == expected_hash
    assert saved_calib["config_hash"] != initial_hash

    # Check loaded baseline
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["scope"] == "src"
    assert _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope="src")


def test_red_team_baseline_and_matcher_hardening(tmp_path: Path) -> None:
    """Verifies robustness fixes for recursion limits, unhashable stops, and negative bounds."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        _deep_tuple,
        _deserialize_shingle_key,
        load_baseline,
    )
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # 1. _deep_tuple depth limit and cycle safety
    deep_val: Any = [1]
    for _ in range(25):
        deep_val = [deep_val]
    with pytest.raises(ValueError, match="Exceeded maximum nesting depth"):
        _deep_tuple(deep_val)

    # Key deserialization handles exceeded recursion depth without crashing
    deep_json = json.dumps(deep_val)
    deserialized = _deserialize_shingle_key(deep_json)
    assert deserialized == deep_json

    # Cyclic list raises ValueError in _deep_tuple
    cyclic: List[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Cyclic container detected"):
        _deep_tuple(cyclic)

    # Mutual/indirect cycle raises ValueError
    cycle_a: List[Any] = []
    cycle_b: List[Any] = [cycle_a]
    cycle_a.append(cycle_b)
    with pytest.raises(ValueError, match="Cyclic container detected"):
        _deep_tuple(cycle_a)

    # Non-cyclic DAG (diamond graph) succeeds
    shared = [1, 2]
    dag = [shared, shared]
    assert _deep_tuple(dag) == ((1, 2), (1, 2))

    # 2. Unhashable items in stop-shingles do not discard valid stop shingles in load_baseline
    calib_json = tmp_path / "mixed_stops.json"
    calib_json.write_text(
        json.dumps({
            "version": "1.4.0",
            "corpus_calibration": {
                "total_units": -10,  # Negative units clamped to 0
                "max_index_frequency": None,  # Explicitly None
                "global_stop_shingles": [
                    {"dict": "not_hashable"},
                    ["valid", "stop"],
                    "scalar_stop",
                ],
                "shingle_frequencies": {
                    '["valid", "stop"]': 25,
                },
            },
        }),
        encoding="utf-8",
    )
    loaded = load_baseline(str(calib_json))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] == 0  # Clamped to >= 0
    assert loaded.corpus_calibration["max_index_frequency"] is None
    stops = loaded.corpus_calibration["global_stop_shingles"]
    assert ("valid", "stop") in stops
    assert "scalar_stop" in stops
    assert len(stops) == 2

    # 3. Matcher scan_target with tfidf=True and negative total_units does not raise ValueError
    repo = tmp_path / "red_repo"
    repo.mkdir()
    code = (
        "def compute_score(a, b):\n"
        "    v1 = a + 1\n"
        "    v2 = v1 * 2\n"
        "    v3 = v2 + b\n"
        "    v4 = v3 - 5\n"
        "    return v4\n"
    )
    (repo / "r1.py").write_text(code, encoding="utf-8")
    (repo / "r2.py").write_text(code, encoding="utf-8")

    clones = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        tfidf=True,
        corpus_calibration={
            "total_units": -100,
            "max_index_frequency": None,
            "global_stop_shingles": [{"unhashable": "obj"}, ["valid", "token"]],
            "shingle_frequencies": {},
        },
    )
    assert len(clones) == 1

    # 4. Non-finite values (NaN / inf / -inf / <= 0) and overflow in matcher do not crash
    for invalid_freq in [float("nan"), float("inf"), float("-inf"), 0.0, -0.5, 1e308]:
        nan_calib = {
            "total_units": 50,
            "max_index_frequency": invalid_freq,
            "global_stop_shingles": set(),
            "shingle_frequencies": {},
        }
        res_calib = scan_target(str(repo), min_lines=5, threshold=0.90, corpus_calibration=nan_calib)
        assert len(res_calib) == 1
        res_uncalib = scan_target(str(repo), min_lines=5, threshold=0.90, max_index_frequency=invalid_freq)
        assert len(res_uncalib) == 1

    # 5. Non-finite values and negative frequencies in baseline JSON
    non_finite_file = tmp_path / "non_finite_calib.json"
    non_finite_file.write_text(
        json.dumps({
            "version": "1.4.0",
            "corpus_calibration": {
                "total_units": 100,
                "max_index_frequency": "NaN",
                "global_stop_shingles": ["stop1"],
                "shingle_frequencies": {
                    "stop1": -5,  # Negative frequency filtered out
                    "stop2": 10,
                },
            },
        }),
        encoding="utf-8",
    )
    loaded_nf = load_baseline(str(non_finite_file))
    assert loaded_nf.corpus_calibration is not None
    assert loaded_nf.corpus_calibration["max_index_frequency"] == 0.25  # Fallback on NaN
    assert "stop1" not in loaded_nf.corpus_calibration["shingle_frequencies"]
    assert loaded_nf.corpus_calibration["shingle_frequencies"]["stop2"] == 10


def test_diff_aware_candidate_pruning_and_stats(tmp_path: Path) -> None:
    """Verifies that diff_files feeds diff-aware unit statistics into candidate pruning."""
    repo = tmp_path / "diff_repo"
    repo.mkdir()

    code_clone_1 = """
def process_data_alpha(items):
    res = []
    for x in items:
        if x > 10:
            res.append(x * 2)
        else:
            res.append(x + 1)
    return res
"""

    code_clone_2 = """
def process_data_beta(items):
    out = []
    for val in items:
        if val > 10:
            out.append(val * 2)
        else:
            out.append(val + 1)
    return out
"""

    code_unmod_pair = """
def internal_worker_gamma(elements):
    acc = []
    for e in elements:
        if e % 2 == 0:
            acc.append(e // 2)
        else:
            acc.append(e * 3)
    return acc
"""

    (repo / "mod.py").write_text(code_clone_1, encoding="utf-8")
    (repo / "unmod_a.py").write_text(code_clone_2, encoding="utf-8")
    (repo / "unmod_b.py").write_text(code_unmod_pair, encoding="utf-8")
    (repo / "unmod_c.py").write_text(code_unmod_pair, encoding="utf-8")

    # 1. Full scan without diff_files finds both clone pairs
    full_clones = scan_target(str(repo), min_lines=6, threshold=0.90)
    assert len(full_clones) == 2

    # 2. Diff scan with diff_files=["mod.py"]: only pairs touching mod.py are returned
    diff_clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["mod.py"],
    )
    assert len(diff_clones) == 1
    pair = diff_clones[0]
    files_in_clone = {Path(pair[1]["file"]).name, Path(pair[2]["file"]).name}
    assert files_in_clone == {"mod.py", "unmod_a.py"}

    # 3. Diff scan where diff_files has no modified files in target: returns [] immediately
    empty_diff_clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["non_existent_file.py"],
    )
    assert empty_diff_clones == []

    # 4. Diff-aware statistics prevent small-diff false pruning with corpus_calibration
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel
    mod_units = harvest_file_units(str(repo / "mod.py"), str(repo), min_lines=6)
    assert len(mod_units) >= 1
    shared_shingle = list(mod_units[0]["shingles"])[0]

    # Baseline total_units = 100, max_index_frequency = 0.25 (pruning cutoff is ceil(101 * 0.25) = 26).
    # If df_global = 25 and df_local in diff = 1 -> combined_df = 26 <= 26 (retained).
    # If df_local counted all local units (>= 2) -> combined_df >= 27 > 26 (would be falsely pruned).
    calib = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {
            shared_shingle if isinstance(shared_shingle, str) else json.dumps(list(shared_shingle)): 25,
        },
    }

    calib_clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["mod.py"],
        corpus_calibration=calib,
    )
    assert len(calib_clones) == 1
    pair_calib = calib_clones[0]
    assert {Path(pair_calib[1]["file"]).name, Path(pair_calib[2]["file"]).name} == {"mod.py", "unmod_a.py"}


def test_record_baseline_calibration_hardening(tmp_path: Path) -> None:
    """Test record_baseline robustly sanitizes non-finite floats, overflow, and invalid counts."""
    from pydoppelgangerhunt.baseline import load_baseline, record_baseline  # pylint: disable=import-outside-toplevel

    u1 = {"file": "mod1.py", "name": "fn1", "structural_hash": "hash1"}
    u2 = {"file": "mod2.py", "name": "fn2", "structural_hash": "hash2"}
    clones = [(0.95, u1, u2)]

    # 1. Non-finite max_index_frequency (NaN, Inf) and negative/invalid counts
    calib_malformed = {
        "total_units": "invalid_int",
        "max_index_frequency": float("nan"),
        "global_stop_shingles": ["stop_a", ("nested", "shingle"), 12345],
        "shingle_frequencies": {
            "valid_key": 42,
            "zero_key": 0,
            "negative_key": -5,
            "invalid_count": "not_an_int",
        },
    }

    out_file = tmp_path / "baseline_hardened.json"
    record_baseline(clones, str(out_file), "target_repo", 0.90, corpus_calibration=calib_malformed)

    raw_json = json.loads(out_file.read_text(encoding="utf-8"))
    calib_data = raw_json.get("corpus_calibration")
    assert calib_data is not None
    assert calib_data["total_units"] == 0
    assert calib_data["max_index_frequency"] is None
    assert calib_data["shingle_frequencies"] == {"valid_key": 42}
    assert "stop_a" in calib_data["global_stop_shingles"]

    # 2. Inf and out-of-range floats
    calib_inf = {
        "total_units": 50,
        "max_index_frequency": float("inf"),
        "shingle_frequencies": {"k": 10},
    }
    out_inf = tmp_path / "baseline_inf.json"
    record_baseline(clones, str(out_inf), "target_repo", 0.90, corpus_calibration=calib_inf)
    data_inf = json.loads(out_inf.read_text(encoding="utf-8"))["corpus_calibration"]
    assert data_inf["max_index_frequency"] is None
    assert data_inf["total_units"] == 50

    # 3. Valid float should be cleanly preserved
    calib_valid = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "shingle_frequencies": {"k": 10},
    }
    out_valid = tmp_path / "baseline_valid.json"
    record_baseline(clones, str(out_valid), "target_repo", 0.90, corpus_calibration=calib_valid)
    data_valid = json.loads(out_valid.read_text(encoding="utf-8"))["corpus_calibration"]
    assert data_valid["max_index_frequency"] == 0.25
    assert data_valid["total_units"] == 100

    loaded = load_baseline(str(out_valid))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] == 100
    assert loaded.corpus_calibration["max_index_frequency"] == 0.25


def test_diff_scan_uncalibrated_pruning(tmp_path: Path) -> None:
    """Test diff-only scan without calibration prunes ubiquitous boilerplate across repository units."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_pruning"
    repo.mkdir()

    boilerplate_code = (
        "def common_logging_handler(event_name, payload, context):\n"
        "    tag = 'AUDIT'\n"
        "    msg = f'[{tag}] {event_name}: {payload}'\n"
        "    print(msg)\n"
        "    log_line = {'event': event_name, 'ctx': context}\n"
        "    return log_line\n"
    )

    for i in range(34):
        (repo / f"unmod_{i}.py").write_text(boilerplate_code, encoding="utf-8")

    mod_code = (
        boilerplate_code + "\n\n"
        "def calculate_tax_for_eu_region(amount, rate, country_code):\n"
        "    subtotal = amount * rate\n"
        "    tax = subtotal * 0.20\n"
        "    total = subtotal + tax\n"
        "    status = 'EU_OK'\n"
        "    return {'total': total, 'tax': tax, 'status': status}\n"
    )
    (repo / "mod.py").write_text(mod_code, encoding="utf-8")

    target_code = (
        "def calculate_tax_for_eu_region_duplicate(amount, rate, country_code):\n"
        "    subtotal = amount * rate\n"
        "    tax = subtotal * 0.20\n"
        "    total = subtotal + tax\n"
        "    status = 'EU_OK'\n"
        "    return {'total': total, 'tax': tax, 'status': status}\n"
    )
    (repo / "clone_target.py").write_text(target_code, encoding="utf-8")

    diff_clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["mod.py"],
        max_index_frequency=0.25,
        min_corpus_size=30,
        corpus_calibration=None,
    )

    assert len(diff_clones) == 1
    pair = diff_clones[0]
    matched_files = {Path(pair[1]["file"]).name, Path(pair[2]["file"]).name}
    assert matched_files == {"mod.py", "clone_target.py"}
    assert pair[1]["name"] in ("calculate_tax_for_eu_region", "calculate_tax_for_eu_region_duplicate")


def test_diff_scan_avoids_pruning_novel_shingles_missing_from_calibration(tmp_path: Path) -> None:
    """Verifies that shingles missing from calibration frequencies in differential mode are not falsely pruned."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_stale_baseline"
    repo.mkdir()

    def generate_fn(num: int) -> str:
        return (
            f"def custom_domain_service_action_{num}(payload, context, options):\n"
            "    transformed = [item.strip() for item in payload if item]\n"
            "    result_map = {idx: val.upper() for idx, val in enumerate(transformed)}\n"
            f"    audit_tag = 'ACTION_{num}'\n"
            "    status_flag = len(result_map) > 0\n"
            "    return {'status': status_flag, 'tag': audit_tag, 'data': result_map}\n"
        )

    diff_code_1 = "\n\n".join(generate_fn(i) for i in range(4))
    diff_code_2 = "\n\n".join(generate_fn(i) for i in range(4))
    (repo / "new_service_a.py").write_text(diff_code_1, encoding="utf-8")
    (repo / "new_service_b.py").write_text(diff_code_2, encoding="utf-8")

    # Stale/small baseline calibration with total_units=1 and empty shingle_frequencies:
    # total_corpus_units = 1 + 8 = 9.
    # With max_index_frequency=0.25 and effective_min_corpus=4, max_posting_len = max(2, ceil(9 * 0.25)) = 3.
    # The novel domain shingles occur in 8 units (or 4 per file).
    # If uncalibrated df_local=8 (or 4) were compared with max_posting_len=3, it would be falsely pruned!
    # Because it is missing from calibration frequencies, the posting-length check is bypassed.
    calib = {
        "total_units": 1,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["new_service_a.py", "new_service_b.py"],
        corpus_calibration=calib,
    )

    cross_file_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in clones
        if Path(u1["file"]).name != Path(u2["file"]).name
    ]
    assert len(cross_file_clones) >= 4
    for _, u1, u2 in cross_file_clones:
        f1 = Path(u1["file"]).name
        f2 = Path(u2["file"]).name
        assert {f1, f2} == {"new_service_a.py", "new_service_b.py"}

    # Verify that global_stop_shingles is STILL honored for novel shingles if present in global_stop_shingles
    units = harvest_file_units(str(repo / "new_service_a.py"), str(repo), min_lines=6)
    assert len(units) >= 1

    calib_all_stopped = {
        "total_units": 1,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(units[0]["shingles"]),
        "shingle_frequencies": {},
    }
    clones_all_stopped = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.90,
        diff_files=["new_service_a.py", "new_service_b.py"],
        corpus_calibration=calib_all_stopped,
    )
    names = {u1["name"] for _, u1, _ in clones_all_stopped}
    assert "custom_domain_service_action_0" not in names


def test_corpus_calibration_malformed_types_and_unserializable_shingles(tmp_path: Path) -> None:
    """Verifies that non-dict calibration, non-dict shingle_frequencies, and un-serializable objects do not crash."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import record_baseline  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "sample.py"
    code_file.write_text(
        "def compute_something(a, b):\n"
        "    x = a * 2 + b\n"
        "    y = x ** 2 - 1\n"
        "    return y if y > 0 else 0\n\n"
        "def compute_something_dup(a, b):\n"
        "    x = a * 2 + b\n"
        "    y = x ** 2 - 1\n"
        "    return y if y > 0 else 0\n",
        encoding="utf-8",
    )

    # 1. Non-dict corpus_calibration passed to scan_target
    clones = scan_target(str(tmp_path), min_lines=3, corpus_calibration="not_a_dict")  # type: ignore[arg-type]
    assert len(clones) >= 1

    # 2. corpus_calibration with shingle_frequencies=None under tfidf=True
    clones_tfidf = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={"shingle_frequencies": None, "total_units": 5},
    )
    assert len(clones_tfidf) >= 1

    # 3. corpus_calibration with non-dict shingle_frequencies
    clones_bad_freqs = scan_target(
        str(tmp_path),
        min_lines=3,
        corpus_calibration={"shingle_frequencies": "not_a_dict", "total_units": 5},
    )
    assert len(clones_bad_freqs) >= 1

    # 4. record_baseline with non-dict corpus_calibration
    base_file = tmp_path / "base1.json"
    record_baseline(
        clones,
        str(base_file),
        str(tmp_path),
        0.9,
        corpus_calibration="not_a_dict",  # type: ignore[arg-type]
    )
    assert base_file.exists()

    # 5. record_baseline with un-JSON-serializable objects in global_stop_shingles
    base_file_unserializable = tmp_path / "base2.json"
    record_baseline(
        clones,
        str(base_file_unserializable),
        str(tmp_path),
        0.9,
        corpus_calibration={
            "total_units": 10,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [({1, 2}, [object()])],
            "shingle_frequencies": {"shingle_key_1": 5},
        },
    )
    assert base_file_unserializable.exists()

    # 6. tfidf=True with non-finite and negative frequencies in corpus_calibration
    clones_nonfinite = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 5,
            "shingle_frequencies": {
                "compute_something": float("inf"),
                "var_key": float("-inf"),
                "other_key": float("nan"),
                "neg_key": -999,
                "str_inf": "inf",
            },
        },
    )
    assert len(clones_nonfinite) >= 1

    # 7. candidate pairing with non-finite and negative frequencies in diff and normal mode
    clones_diff_nonfinite = scan_target(
        str(tmp_path),
        min_lines=3,
        diff_files=["sample.py"],
        corpus_calibration={
            "total_units": 5,
            "max_index_frequency": 0.25,
            "shingle_frequencies": {
                "compute_something": float("inf"),
                "neg_key": -50,
            },
        },
    )
    assert len(clones_diff_nonfinite) >= 1


def test_type_tagged_shingle_keys_and_oversized_unit_bounds(tmp_path: Path) -> None:
    """Verifies type-tagged shingle keys, legacy key backward compatibility, and oversized unit bounds."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        MAX_CALIBRATION_UNITS,
        _deserialize_shingle_key,
        _safe_total_units,
        _serialize_shingle_key,
        load_baseline as load_base,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import _lookup_calib_freq, scan_target  # pylint: disable=import-outside-toplevel

    # 1. Type-tagged key serialization & deserialization fidelity
    # Tuples
    tup = ("Module", "If", "Compare")
    assert _serialize_shingle_key(tup) == 't:["Module", "If", "Compare"]'
    assert _deserialize_shingle_key('t:["Module", "If", "Compare"]') == tup

    # Scalar strings (including ones that look like JSON lists)
    s_like_list = '["Module", "If"]'
    assert _serialize_shingle_key(s_like_list) == 's:["Module", "If"]'
    assert _deserialize_shingle_key('s:["Module", "If"]') == s_like_list
    assert isinstance(_deserialize_shingle_key('s:["Module", "If"]'), str)

    # Strings with prefixes
    s_prefix = "t:custom_tag"
    assert _serialize_shingle_key(s_prefix) == "s:t:custom_tag"
    assert _deserialize_shingle_key("s:t:custom_tag") == s_prefix

    # Numbers and booleans
    assert _serialize_shingle_key(42) == "i:42"
    assert _deserialize_shingle_key("i:42") == 42
    assert _serialize_shingle_key(3.14) == "f:3.14"
    assert _deserialize_shingle_key("f:3.14") == 3.14
    assert _serialize_shingle_key(True) == "b:1"
    assert _deserialize_shingle_key("b:1") is True

    # Legacy untagged format compatibility
    assert _deserialize_shingle_key('["Legacy", "Shingle"]') == ("Legacy", "Shingle")
    assert _deserialize_shingle_key("plain_legacy_str") == "plain_legacy_str"

    # 2. _safe_total_units bounds
    assert _safe_total_units(100) == 100
    assert _safe_total_units(0) == 0
    assert _safe_total_units(-50) == 0
    assert _safe_total_units(None) == 0
    assert _safe_total_units("invalid") == 0
    assert _safe_total_units(10**400) == MAX_CALIBRATION_UNITS
    assert _safe_total_units(MAX_CALIBRATION_UNITS + 500) == MAX_CALIBRATION_UNITS

    # 3. _lookup_calib_freq multi-representation lookup
    freqs = {
        ("Exact", "Tuple"): 10,
        't:["Tagged", "Tuple"]': 20,
        '["Legacy", "Tuple"]': 30,
        "s:tagged_string": 40,
        "exact_string": 50,
        "i:99": 60,
    }
    assert _lookup_calib_freq(freqs, ("Exact", "Tuple")) == 10
    assert _lookup_calib_freq(freqs, ("Tagged", "Tuple")) == 20
    assert _lookup_calib_freq(freqs, ("Legacy", "Tuple")) == 30
    assert _lookup_calib_freq(freqs, "tagged_string") == 40
    assert _lookup_calib_freq(freqs, "exact_string") == 50
    assert _lookup_calib_freq(freqs, 99) == 60
    assert _lookup_calib_freq(freqs, "missing") is None

    # 4. TF-IDF and candidate pairing with oversized total_units
    code_file = tmp_path / "oversized_sample.py"
    code_file.write_text(
        "def oversized_test_fn(x, y):\n"
        "    res = x * 3 + y\n"
        "    res2 = res ** 2 + 10\n"
        "    return res2 if res2 > 5 else 0\n\n"
        "def oversized_test_fn_dup(x, y):\n"
        "    res = x * 3 + y\n"
        "    res2 = res ** 2 + 10\n"
        "    return res2 if res2 > 5 else 0\n",
        encoding="utf-8",
    )

    clones_oversized = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 10**400,
            "shingle_frequencies": {},
        },
    )
    assert len(clones_oversized) >= 1

    # 5. Global frequencies validated and clamped to total_units
    clones_freq_clamped = scan_target(
        str(tmp_path),
        min_lines=3,
        tfidf=True,
        corpus_calibration={
            "total_units": 10,
            "shingle_frequencies": {
                "oversized_test_fn": 99999,  # Exceeds total_units=10
            },
        },
    )
    assert len(clones_freq_clamped) >= 1

    # 6. Candidate pairing with total_units=0 and frequencies > 0 does not prune candidates
    clones_zero_units = scan_target(
        str(tmp_path),
        min_lines=3,
        corpus_calibration={
            "total_units": 0,
            "max_index_frequency": 0.25,
            "shingle_frequencies": {
                "oversized_test_fn": 100,
            },
        },
    )
    assert len(clones_zero_units) >= 1

    # 7. record_baseline and load_baseline round-trip with type-tagged keys and oversized bounds
    base_file = tmp_path / "tagged_baseline.json"
    record_baseline(
        clones_oversized,
        str(base_file),
        str(tmp_path),
        0.90,
        corpus_calibration={
            "total_units": 10**400,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [("A", "B"), '["string_that_looks_like_tuple"]', 123],
            "shingle_frequencies": {
                ("A", "B"): 5,
                '["string_that_looks_like_tuple"]': 10,
                123: 15,
            },
        },
    )
    raw_saved = json.loads(base_file.read_text(encoding="utf-8"))
    calib_saved = raw_saved["corpus_calibration"]
    assert calib_saved["total_units"] == MAX_CALIBRATION_UNITS
    assert 't:["A", "B"]' in calib_saved["shingle_frequencies"]
    assert 's:["string_that_looks_like_tuple"]' in calib_saved["shingle_frequencies"]
    assert "i:123" in calib_saved["shingle_frequencies"]

    loaded = load_base(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["total_units"] == MAX_CALIBRATION_UNITS
    assert loaded_calib["shingle_frequencies"][("A", "B")] == 5
    assert loaded_calib["shingle_frequencies"]['["string_that_looks_like_tuple"]'] == 10
    assert loaded_calib["shingle_frequencies"][123] == 15


def test_diff_scan_prunes_uncalibrated_shingles_matching_many_unmodified_units(tmp_path: Path) -> None:
    """Verifies that uncalibrated shingles matching many unmodified units are bounded/pruned to prevent pair explosion."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_explosion_defense"
    repo.mkdir()

    # Create 10 unmodified files, each containing a common helper with identical ubiquitous structure
    common_code = (
        "def common_repository_helper(context, data):\n"
        "    if not context:\n"
        "        return None\n"
        "    res = {}\n"
        "    for k, v in data.items():\n"
        "        res[k] = str(v).strip()\n"
        "    return res\n"
    )
    for i in range(10):
        (repo / f"unmodified_{i}.py").write_text(
            f"# File unmodified {i}\n" + common_code,
            encoding="utf-8",
        )

    # Create 1 changed file in the diff containing the same common helper plus a unique function
    changed_code = (
        common_code
        + "\n\ndef unique_changed_feature():\n"
        + "    return 'UNIQUE_DIFF_FEATURE_ALPHA_BETA'\n"
    )
    (repo / "changed.py").write_text(changed_code, encoding="utf-8")

    # Stale/partial calibration missing common_repository_helper shingles entirely:
    # 11 total units in repo (10 unmodified + 1 changed).
    # calib specifies total_units=10, max_index_frequency=0.25, min_corpus_size=4.
    # Because common_repository_helper occurs in 10 unmodified units, df_unmodified = 10 > max_posting_len (which is ceil(11 * 0.25) = 3).
    # The common shingle MUST be pruned from pairing against the 10 unmodified units!
    calib = {
        "total_units": 10,
        "max_index_frequency": 0.25,
        "min_lines": 6,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    clones = scan_target(
        str(repo),
        min_lines=6,
        threshold=0.85,
        diff_files=["changed.py"],
        corpus_calibration=calib,
    )

    # The changed file should NOT be paired with all 10 unmodified files across the common helper
    changed_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in clones
        if "changed.py" in (Path(u1["file"]).name, Path(u2["file"]).name)
    ]
    assert len(changed_clones) == 0

    # Also verify that if two diff files share a novel domain clone (df_unmodified == 0), it IS detected
    novel_clone_code = (
        "def novel_domain_handler(request, response):\n"
        "    data_items = [x for x in request if x]\n"
        "    mapped = {i: str(v).lower() for i, v in enumerate(data_items)}\n"
        "    audit = 'DOMAIN_HANDLER_SECURE_TAG'\n"
        "    return {'audit': audit, 'mapped': mapped}\n"
    )
    (repo / "changed_a.py").write_text(novel_clone_code, encoding="utf-8")
    (repo / "changed_b.py").write_text(novel_clone_code, encoding="utf-8")

    novel_clones = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=["changed_a.py", "changed_b.py"],
        corpus_calibration=dict(calib, min_lines=5),
    )
    cross_diff_clones = [
        (sim, u1, u2)
        for sim, u1, u2 in novel_clones
        if {Path(u1["file"]).name, Path(u2["file"]).name} == {"changed_a.py", "changed_b.py"}
    ]
    assert len(cross_diff_clones) >= 1


def test_min_corpus_size_malformed_fallback(tmp_path: Path) -> None:
    """Verifies that non-integer, negative, or invalid min_corpus_size safely falls back without raising errors."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "min_corpus_sample.py"
    code_file.write_text(
        "def helper_sample_one(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n\n"
        "def helper_sample_two(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n",
        encoding="utf-8",
    )

    # 1. Invalid min_corpus_size in corpus_calibration
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration={"min_corpus_size": "invalid", "total_units": 10},  # type: ignore[dict-item]
    )
    assert len(clones) >= 1

    # 2. Negative min_corpus_size in corpus_calibration
    clones_neg = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration={"min_corpus_size": -50, "total_units": 10},
    )
    assert len(clones_neg) >= 1

    # 3. Direct invalid min_corpus_size parameter in scan_target
    clones_arg = scan_target(
        str(tmp_path),
        min_lines=2,
        min_corpus_size="invalid",  # type: ignore[arg-type]
    )
    assert len(clones_arg) >= 1

    # 4. compute_corpus_calibration with invalid min_corpus_size
    calib = compute_corpus_calibration([], min_corpus_size="invalid")  # type: ignore[arg-type]
    assert isinstance(calib, dict)


def test_small_positive_max_index_frequency_preserved_without_lossy_rounding(tmp_path: Path) -> None:
    """Verifies that small positive max_index_frequency values (e.g. 4e-7) are not rounded to 0.0 or reset to 0.25."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    base_path = tmp_path / "small_freq_baseline.json"
    calib = compute_corpus_calibration([], max_index_frequency=4e-7)
    assert calib["max_index_frequency"] == 4e-7

    u1 = {"file": "a.py", "name": "fn1", "tokens": ["a"]}
    u2 = {"file": "b.py", "name": "fn2", "tokens": ["b"]}
    record_baseline([(1.0, u1, u2)], str(base_path), "repo", 0.85, corpus_calibration=calib)

    # Raw JSON must retain small positive float
    raw_data = json.loads(base_path.read_text(encoding="utf-8"))
    assert raw_data["corpus_calibration"]["max_index_frequency"] == 4e-7

    # Loaded baseline must retain exact 4e-7 and not fall back to 0.25
    loaded = load_baseline(str(base_path))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["max_index_frequency"] == 4e-7


def test_corpus_calibration_feature_mode_validation_and_rejection(tmp_path: Path) -> None:
    """Verifies that feature space modes (bag_of_tokens, call_sequences) are tracked and mismatched calibration is skipped."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "feature_mode_sample.py"
    code_file.write_text(
        "def func_alpha(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n\n"
        "def func_beta(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n",
        encoding="utf-8",
    )

    # 1. Feature flags are properly recorded
    calib_bot = compute_corpus_calibration([], bag_of_tokens=True)
    assert calib_bot["bag_of_tokens"] is True
    assert calib_bot["call_sequences"] is False

    calib_calls = compute_corpus_calibration([], call_sequences=True)
    assert calib_calls["bag_of_tokens"] is False
    assert calib_calls["call_sequences"] is True

    calib_default = compute_corpus_calibration([])
    assert calib_default["bag_of_tokens"] is False
    assert calib_default["call_sequences"] is False

    # 2. Incompatible calibration mode is skipped in scan_target without corrupting AST clone detection
    # Passing bag_of_tokens=True calibration to a standard AST shingle scan
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        corpus_calibration=calib_bot,
        bag_of_tokens=False,
        call_sequences=False,
    )
    # The clone must be detected and not falsely pruned by incompatible calibration lookups
    assert len(clones) >= 1


def test_malformed_crafted_calibration_metadata(tmp_path: Path) -> None:
    """Verifies that non-boolean and crafted strings in calibration metadata are safely coerced and sanitized."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        _attach_calibration_flags,
        _safe_bool,
        load_baseline,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    # 1. _safe_bool coercion checks
    assert _safe_bool(True) is True
    assert _safe_bool(False) is False
    assert _safe_bool(1) is True
    assert _safe_bool(0) is False
    assert _safe_bool("true") is True
    assert _safe_bool("True") is True
    assert _safe_bool("1") is True
    assert _safe_bool("yes") is True
    assert _safe_bool("false") is False
    assert _safe_bool("False") is False
    assert _safe_bool("0") is False
    assert _safe_bool("no") is False
    assert _safe_bool(["invalid"]) is False
    assert _safe_bool({"key": "val"}) is False
    assert _safe_bool(None) is False

    # 2. _attach_calibration_flags with malformed types
    target_dict: Dict[str, Any] = {}
    _attach_calibration_flags(target_dict, "not_a_dict")  # type: ignore[arg-type]
    assert not target_dict

    crafted_source = {
        "bag_of_tokens": "false",
        "call_sequences": "0",
        "filter_stop_shingles": ["invalid"],
        "min_corpus_size": "crafted_string",
    }
    _attach_calibration_flags(target_dict, crafted_source)
    assert target_dict["bag_of_tokens"] is False
    assert target_dict["call_sequences"] is False
    assert target_dict["filter_stop_shingles"] is False
    assert target_dict["min_corpus_size"] is None

    # Negative min_corpus_size sanitized to None
    _attach_calibration_flags(target_dict, {"min_corpus_size": -42})
    assert target_dict["min_corpus_size"] is None

    # 3. Compatibility checking on malformed calib objects
    assert _is_calibration_mode_compatible("not_a_dict", bag_of_tokens=False, call_sequences=False) is False  # type: ignore[arg-type]
    assert _is_calibration_mode_compatible({"bag_of_tokens": "false"}, bag_of_tokens=False, call_sequences=False) is True
    assert _is_calibration_mode_compatible({"bag_of_tokens": "true"}, bag_of_tokens=True, call_sequences=False) is True

    # 4. JSON loading with spoofed fields
    base_file = tmp_path / "spoofed_baseline.json"
    spoofed_json = {
        "version": "1.4.0",
        "target": "repo",
        "threshold": 0.85,
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 50,
            "max_index_frequency": 0.25,
            "bag_of_tokens": "false",
            "call_sequences": ["crafted"],
            "min_corpus_size": "invalid_size",
        },
    }
    base_file.write_text(json.dumps(spoofed_json), encoding="utf-8")
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["bag_of_tokens"] is False
    assert loaded.corpus_calibration["call_sequences"] is False
    assert loaded.corpus_calibration["min_corpus_size"] is None


def test_subnormal_and_extreme_floats_index_frequency() -> None:
    """Verifies that subnormal and extreme float frequencies are handled safely without exceptions."""
    from pydoppelgangerhunt.baseline import _safe_index_frequency  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _compute_max_posting_len  # pylint: disable=import-outside-toplevel

    # Subnormal and very small positive floats
    assert _safe_index_frequency(1e-300) == 1e-300
    assert _safe_index_frequency(1e-320) == 1e-320

    # Underflow to 0.0 or negative/inf/nan rejected
    assert _safe_index_frequency(0.0) is None
    assert _safe_index_frequency(-1e-5) is None
    assert _safe_index_frequency(1.00001) is None
    assert _safe_index_frequency(float("nan")) is None
    assert _safe_index_frequency(float("inf")) is None

    # Extremely small positive float in posting calculation floors to max(2, ...)
    posting_len = _compute_max_posting_len(500, 1e-300, 10)
    assert posting_len == 2


def test_stop_shingles_calibration_compatibility_and_isolation(tmp_path: Path) -> None:
    """Verifies that configuration-derived stops are kept out of calibration and unconfigured scans reject stop-shingle calibration."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        DEFAULT_STOP_SHINGLES,
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. compute_corpus_calibration does not inject DEFAULT_STOP_SHINGLES into global_stop_shingles
    calib = compute_corpus_calibration([], filter_stop_shingles=True, stop_shingles={("Custom", "Stop")})
    assert len(calib["global_stop_shingles"]) == 0
    for s in DEFAULT_STOP_SHINGLES:
        assert s not in calib["global_stop_shingles"]
    assert ("Custom", "Stop") not in calib["global_stop_shingles"]
    assert calib["filter_stop_shingles"] is True

    # 2. _is_calibration_mode_compatible rejects filter_stop_shingles true-to-false transition
    assert _is_calibration_mode_compatible(calib, bag_of_tokens=False, call_sequences=False, filter_stop_shingles=True) is True
    assert _is_calibration_mode_compatible(calib, bag_of_tokens=False, call_sequences=False, filter_stop_shingles=False) is False

    # 3. scan_target skips calibration recorded with filter_stop_shingles=True when current scan has filter_stop_shingles=False
    code_file = tmp_path / "stop_shingle_test.py"
    code_file.write_text(
        "def fn_a(a, b):\n"
        "    x = a + b\n"
        "    y = x * 2\n"
        "    return y\n\n"
        "def fn_b(a, b):\n"
        "    x = a + b\n"
        "    y = x * 2\n"
        "    return y\n",
        encoding="utf-8",
    )
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        min_tokens=3,
        corpus_calibration=calib,
        filter_stop_shingles=False,
    )
    assert len(clones) >= 1


def test_harvesting_modes_calibration_persistence_and_compatibility(tmp_path: Path) -> None:
    """Verifies that AST unit-shaping and harvesting modes are persisted in calibration and validated for compatibility."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        HARVEST_BOOLEAN_MODES,
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. compute_corpus_calibration attaches all default harvesting boolean modes and int bounds
    calib_default = compute_corpus_calibration([])
    for flag, default_val in HARVEST_BOOLEAN_MODES:
        assert flag in calib_default
        assert calib_default[flag] is default_val
    assert calib_default["window_size"] == 5
    assert calib_default["min_expr_complexity"] == 4

    # 2. compute_corpus_calibration overrides harvesting flags when explicitly provided
    calib_custom = compute_corpus_calibration(
        [],
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=8,
    )
    assert calib_custom["blind_literals"] is True
    assert calib_custom["strip_annotations"] is False
    assert calib_custom["idioms"] is True
    assert calib_custom["commutative"] is True
    assert calib_custom["filter_boilerplate"] is True
    assert calib_custom["sliding_window"] is True
    assert calib_custom["window_size"] == 10
    assert calib_custom["complex_expressions"] is True
    assert calib_custom["min_expr_complexity"] == 8

    # 3. Round-trip baseline persistence through record_baseline and load_baseline
    base_file = tmp_path / "harvest_baseline.json"
    record_baseline([], str(base_file), "target", 0.90, corpus_calibration=calib_custom)
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    loaded_calib = loaded.corpus_calibration
    assert loaded_calib["blind_literals"] is True
    assert loaded_calib["strip_annotations"] is False
    assert loaded_calib["idioms"] is True
    assert loaded_calib["commutative"] is True
    assert loaded_calib["filter_boilerplate"] is True
    assert loaded_calib["sliding_window"] is True
    assert loaded_calib["window_size"] == 10
    assert loaded_calib["complex_expressions"] is True
    assert loaded_calib["min_expr_complexity"] == 8

    # 4. _is_calibration_mode_compatible rejects mismatched harvesting configurations
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=8,
    ) is True

    # Mismatched blind_literals
    assert _is_calibration_mode_compatible(calib_custom, blind_literals=False) is False
    assert _is_calibration_mode_compatible(calib_default, blind_literals=True) is False

    # Mismatched strip_annotations
    assert _is_calibration_mode_compatible(calib_custom, strip_annotations=True) is False
    assert _is_calibration_mode_compatible(calib_default, strip_annotations=False) is False

    # Mismatched idioms
    assert _is_calibration_mode_compatible(calib_custom, idioms=False) is False
    assert _is_calibration_mode_compatible(calib_default, idioms=True) is False

    # Mismatched commutative
    assert _is_calibration_mode_compatible(calib_custom, commutative=False) is False
    assert _is_calibration_mode_compatible(calib_default, commutative=True) is False

    # Mismatched filter_boilerplate
    assert _is_calibration_mode_compatible(calib_custom, filter_boilerplate=False) is False
    assert _is_calibration_mode_compatible(calib_default, filter_boilerplate=True) is False

    # Mismatched window_size when sliding_window is active
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=5,
        complex_expressions=True,
        min_expr_complexity=8,
    ) is False

    # Mismatched min_expr_complexity when complex_expressions is active
    assert _is_calibration_mode_compatible(
        calib_custom,
        blind_literals=True,
        strip_annotations=False,
        idioms=True,
        commutative=True,
        filter_boilerplate=True,
        sliding_window=True,
        window_size=10,
        complex_expressions=True,
        min_expr_complexity=4,
    ) is False

    # 5. Integration: scan_target skips incompatible calibration
    code_file = tmp_path / "diff_test.py"
    code_file.write_text(
        "def service_one(a, b):\n"
        "    v1 = a * 10\n"
        "    v2 = b + 20\n"
        "    return v1 + v2\n\n"
        "def service_two(a, b):\n"
        "    v1 = a * 10\n"
        "    v2 = b + 20\n"
        "    return v1 + v2\n",
        encoding="utf-8",
    )
    clones = scan_target(
        str(tmp_path),
        min_lines=2,
        min_tokens=3,
        corpus_calibration=calib_custom,
        blind_literals=False,
    )
    assert len(clones) >= 1


def test_calibration_bounds_and_frequency_cutoff_compatibility(tmp_path: Path) -> None:
    """Verifies that min_lines, min_tokens, and max_index_frequency are persisted and validated for compatibility."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        _is_calibration_mode_compatible,
        scan_target,
    )

    # 1. Default and custom bounds in compute_corpus_calibration
    c_default = compute_corpus_calibration([])
    assert c_default["min_lines"] == 8
    assert c_default["min_tokens"] == 15
    assert c_default["max_index_frequency"] == 0.25

    c_custom = compute_corpus_calibration(
        [],
        min_lines=6,
        min_tokens=10,
        max_index_frequency=0.40,
    )
    assert c_custom["min_lines"] == 6
    assert c_custom["min_tokens"] == 10
    assert c_custom["max_index_frequency"] == 0.40

    # 2. Round-trip baseline persistence
    base_file = tmp_path / "bounds_baseline.json"
    record_baseline([], str(base_file), "target", 0.90, corpus_calibration=c_custom)
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["min_lines"] == 6
    assert loaded.corpus_calibration["min_tokens"] == 10
    assert loaded.corpus_calibration["max_index_frequency"] == 0.40

    # 3. Direct compatibility check for min_lines and min_tokens
    assert _is_calibration_mode_compatible(c_custom, min_lines=6, min_tokens=10, max_index_frequency=0.40) is True
    assert _is_calibration_mode_compatible(c_custom, min_lines=8, min_tokens=10, max_index_frequency=0.40) is False
    assert _is_calibration_mode_compatible(c_custom, min_lines=6, min_tokens=15, max_index_frequency=0.40) is False
    assert _is_calibration_mode_compatible(c_default, min_lines=6, min_tokens=15, max_index_frequency=0.25) is False
    assert _is_calibration_mode_compatible(c_default, min_lines=8, min_tokens=10, max_index_frequency=0.25) is False

    # 4. Direct compatibility check for max_index_frequency
    assert _is_calibration_mode_compatible(c_default, max_index_frequency=0.50) is False
    assert _is_calibration_mode_compatible(c_default, max_index_frequency=None) is False

    c_none = compute_corpus_calibration([], max_index_frequency=None)
    assert c_none["max_index_frequency"] is None
    assert _is_calibration_mode_compatible(c_none, max_index_frequency=None) is True
    assert _is_calibration_mode_compatible(c_none, max_index_frequency=0.25) is False

    # 5. Integration: Lowered min_lines threshold in diff scan skips calibration
    repo_dir = tmp_path / "threshold_repo"
    repo_dir.mkdir()
    fn_7_lines = (
        "def micro_service_step(val_x, val_y):\n"
        "    t1 = val_x * 3\n"
        "    t2 = val_y * 7\n"
        "    t3 = t1 + t2\n"
        "    t4 = t3 * 2\n"
        "    t5 = t4 - 1\n"
        "    return t5 if t5 > 0 else 0\n"
    )
    (repo_dir / "worker_a.py").write_text(fn_7_lines, encoding="utf-8")
    (repo_dir / "worker_b.py").write_text(fn_7_lines, encoding="utf-8")

    # Baseline recorded with default min_lines=8
    calib_base_8 = compute_corpus_calibration([], min_lines=8)
    # Scanning with min_lines=6 lowers the threshold compared to calib_base_8
    clones_lowered = scan_target(
        str(repo_dir),
        min_lines=6,
        min_tokens=5,
        threshold=0.90,
        corpus_calibration=calib_base_8,
    )
    # Incompatible calibration is bypassed and the 7-line clone is detected
    assert len(clones_lowered) == 1
    # When scanned with matching min_lines=8, the 7-line function is naturally filtered out
    clones_strict = scan_target(
        str(repo_dir),
        min_lines=8,
        threshold=0.90,
        corpus_calibration=calib_base_8,
    )
    assert len(clones_strict) == 0

    # 6. min_corpus_size compatibility and scan-argument authoritativeness
    c_mcs_4 = compute_corpus_calibration([], min_corpus_size=4)
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=100) is False
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=4) is True
    assert _is_calibration_mode_compatible(c_mcs_4, min_corpus_size=None) is True

    clones_mcs_override = scan_target(
        str(repo_dir),
        min_lines=6,
        min_tokens=5,
        min_corpus_size=100,
        corpus_calibration=c_mcs_4,
    )
    assert len(clones_mcs_override) == 1


def test_diff_files_canonical_path_matching_prevents_nested_basename_collisions(tmp_path: Path) -> None:
    """Verifies that diff_files path resolution prevents nested files with identical basenames from matching."""
    repo = tmp_path / "repo_nested_diff"
    repo.mkdir()
    pkg = repo / "pkg"
    pkg.mkdir()

    root_code = (
        "def compute_root_metrics(data_items, factor_val):\n"
        "    accumulator = 0\n"
        "    for item in data_items:\n"
        "        weighted = item * factor_val + 17\n"
        "        accumulator += weighted\n"
        "    return accumulator\n"
    )
    other1_code = (
        "def compute_root_metrics(data_items, factor_val):\n"
        "    accumulator = 0\n"
        "    for item in data_items:\n"
        "        weighted = item * factor_val + 17\n"
        "        accumulator += weighted\n"
        "    return accumulator\n"
    )

    nested_code = (
        "def process_nested_payload(records, scale_num):\n"
        "    output_sum = 100\n"
        "    for entry in records:\n"
        "        temp_res = (entry + scale_num) * 3\n"
        "        output_sum += temp_res\n"
        "    return output_sum\n"
    )
    other2_code = (
        "def process_nested_payload(records, scale_num):\n"
        "    output_sum = 100\n"
        "    for entry in records:\n"
        "        temp_res = (entry + scale_num) * 3\n"
        "        output_sum += temp_res\n"
        "    return output_sum\n"
    )

    (repo / "foo.py").write_text(root_code, encoding="utf-8")
    (repo / "other1.py").write_text(other1_code, encoding="utf-8")
    (pkg / "foo.py").write_text(nested_code, encoding="utf-8")
    (repo / "other2.py").write_text(other2_code, encoding="utf-8")

    # 1. Full scan finds both clone pairs
    all_clones = scan_target(str(repo), min_lines=5, threshold=0.90)
    assert len(all_clones) == 2

    # 2. Diff scan specifying ONLY root "foo.py" must NOT match "pkg/foo.py"
    diff_root_only = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=["foo.py"],
    )
    assert len(diff_root_only) == 1
    files_touched_root = {Path(diff_root_only[0][1]["file"]).name, Path(diff_root_only[0][2]["file"]).name}
    assert files_touched_root == {"foo.py", "other1.py"}
    unit_paths_root = {diff_root_only[0][1]["file"].replace("\\", "/"), diff_root_only[0][2]["file"].replace("\\", "/")}
    assert "pkg/foo.py" not in unit_paths_root

    # 3. Diff scan specifying ONLY nested "pkg/foo.py" must NOT match root "foo.py"
    diff_nested_only = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=["pkg/foo.py"],
    )
    assert len(diff_nested_only) == 1
    files_touched_nested = {Path(diff_nested_only[0][1]["file"]).name, Path(diff_nested_only[0][2]["file"]).name}
    assert files_touched_nested == {"foo.py", "other2.py"}
    unit_paths_nested = {diff_nested_only[0][1]["file"].replace("\\", "/"), diff_nested_only[0][2]["file"].replace("\\", "/")}
    assert "pkg/foo.py" in unit_paths_nested
    assert "foo.py" not in unit_paths_nested

    # 4. Absolute paths also resolve unambiguously
    diff_abs_root = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=[str((repo / "foo.py").resolve())],
    )
    assert len(diff_abs_root) == 1
    assert "pkg/foo.py" not in {diff_abs_root[0][1]["file"].replace("\\", "/"), diff_abs_root[0][2]["file"].replace("\\", "/")}

    diff_abs_nested = scan_target(
        str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=[str((pkg / "foo.py").resolve())],
    )
    assert len(diff_abs_nested) == 1
    assert "pkg/foo.py" in {diff_abs_nested[0][1]["file"].replace("\\", "/"), diff_abs_nested[0][2]["file"].replace("\\", "/")}

    # 5. Scanning subdirectory with diff_files outside that directory returns empty
    diff_sub_mismatch = scan_target(
        str(pkg),
        repo_root=str(repo),
        min_lines=5,
        threshold=0.90,
        diff_files=["foo.py"],
    )
    assert len(diff_sub_mismatch) == 0


def test_baseline_matching_preserves_target_relative_subdirectory_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that subdirectory baselines preserve target-relative identity and cross-root baselines match."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "src"
    pkg = src / "pkg"
    pkg.mkdir(parents=True)

    foo_code = (
        "def compute_summary(data_list, factor):\n"
        "    accum = 0\n"
        "    for val in data_list:\n"
        "        accum += val * factor + 5\n"
        "    return accum\n"
    )
    (pkg / "worker.py").write_text(foo_code, encoding="utf-8")
    (pkg / "worker_clone.py").write_text(foo_code, encoding="utf-8")

    sub_baseline = tmp_path / "sub_baseline.json"
    root_baseline = tmp_path / "root_baseline.json"

    # 1. Record baseline targeting subdirectory repo/src
    rec_args = [
        "pydoppelgangerhunt",
        str(src),
        "--record-baseline",
        str(sub_baseline),
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", rec_args)
    assert main() == 0
    assert sub_baseline.exists()

    # Loaded records must be relative to src (pkg/worker.py), NOT src/pkg/worker.py
    loaded_sub = load_baseline(str(sub_baseline))
    assert len(loaded_sub.records) == 1
    rec = loaded_sub.records[0]
    assert rec["file_a"].replace("\\", "/").startswith("pkg/")
    assert not rec["file_a"].replace("\\", "/").startswith("src/")

    # 2. Scanning repo/src with --baseline sub_baseline must suppress the grandfathered clone (Exit 0)
    check_args = [
        "pydoppelgangerhunt",
        str(src),
        "--baseline",
        str(sub_baseline),
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", check_args)
    assert main() == 0

    # 3. Record baseline targeting full repository repo
    rec_root_args = [
        "pydoppelgangerhunt",
        str(repo),
        "--record-baseline",
        str(root_baseline),
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", rec_root_args)
    assert main() == 0
    assert root_baseline.exists()

    loaded_root = load_baseline(str(root_baseline))
    assert len(loaded_root.records) == 1
    root_rec = loaded_root.records[0]
    assert root_rec["file_a"].replace("\\", "/").startswith("src/pkg/")

    # 4. Scanning repo/src with --baseline root_baseline must ALSO suppress the grandfathered clone via Pass 6
    cross_args = [
        "pydoppelgangerhunt",
        str(src),
        "--baseline",
        str(root_baseline),
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", cross_args)
    assert main() == 0

    # 5. Pruning root_baseline while targeting repo/src must retain the clone via boundary matching
    prune_args = [
        "pydoppelgangerhunt",
        str(src),
        "--baseline",
        str(root_baseline),
        "--prune-baseline",
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", prune_args)
    assert main() == 0

    reloaded_root = load_baseline(str(root_baseline))
    assert len(reloaded_root.records) == 1
    assert reloaded_root.records[0]["file_a"].replace("\\", "/").startswith("src/pkg/")


def test_calibrated_differential_tfidf_weights_unchanged_candidate_partner_shingles(
    tmp_path: Path,
) -> None:
    """Verifies calibrated differential TF-IDF weights incorporate global frequencies for unchanged partner shingles."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    f_changed = tmp_path / "changed.py"
    f_unchanged = tmp_path / "unchanged.py"

    code_changed = (
        "def run_pipeline(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    return x + a + b + c + d\n"
    )
    # Extra statement in unchanged creates shingles present ONLY in unchanged.py
    code_unchanged = (
        "def run_pipeline(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    extra_unique_val = 99999 + 88888\n"
        "    return x + a + b + c + d\n"
    )

    f_changed.write_text(code_changed, encoding="utf-8")
    f_unchanged.write_text(code_unchanged, encoding="utf-8")

    # 1. Harvest units to find shingles unique to unchanged
    units_raw = scan_target(str(tmp_path), min_lines=5, threshold=0.60)
    assert len(units_raw) == 1

    # 2. Build calibration with high global frequency for all shingles
    all_units, calib_dict = scan_target(
        str(tmp_path), min_lines=5, threshold=0.60, return_calibration=True
    )
    assert isinstance(calib_dict, dict)
    assert len(all_units) == 1

    # Modify calibration to simulate a large corpus where shingles have moderate global count (10% < 25%)
    calib_dict["total_units"] = 500
    freq_map = calib_dict.get("shingle_frequencies", {})
    for sh in list(freq_map.keys()):
        freq_map[sh] = 50

    # 3. Differential scan where only changed.py is in diff_files
    diff_clones = scan_target(
        str(tmp_path),
        min_lines=5,
        threshold=0.60,
        tfidf=True,
        diff_files=["changed.py"],
        corpus_calibration=calib_dict,
    )
    assert len(diff_clones) == 1
    sim, u1, u2 = diff_clones[0]
    assert sim >= 0.60
    assert {u1["file"], u2["file"]} == {"changed.py", "unchanged.py"}


def test_calibration_config_hash_determinism_and_sensitivity() -> None:
    """Verifies that compute_calibration_config_hash is deterministic and sensitive to all parameters."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    cfg1: Dict[str, Any] = {"bag_of_tokens": False, "min_lines": 8, "max_index_frequency": 0.25, "idioms": True}
    cfg2: Dict[str, Any] = {"idioms": True, "max_index_frequency": 0.25, "min_lines": 8, "bag_of_tokens": False}
    hash1 = compute_calibration_config_hash(cfg1)
    hash2 = compute_calibration_config_hash(cfg2)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Sensitive to bag_of_tokens
    cfg_bot = dict(cfg1)
    cfg_bot["bag_of_tokens"] = True
    assert compute_calibration_config_hash(cfg_bot) != hash1

    # Sensitive to min_lines
    cfg_lines = dict(cfg1)
    cfg_lines["min_lines"] = 10
    assert compute_calibration_config_hash(cfg_lines) != hash1

    # Sensitive to max_index_frequency
    cfg_freq = dict(cfg1)
    cfg_freq["max_index_frequency"] = 0.50
    assert compute_calibration_config_hash(cfg_freq) != hash1

    # Sensitive to max_index_frequency=None
    cfg_none = dict(cfg1)
    cfg_none["max_index_frequency"] = None
    assert compute_calibration_config_hash(cfg_none) != hash1

    # Invariant to absent vs explicit None min_corpus_size
    assert compute_calibration_config_hash({}) == compute_calibration_config_hash({"min_corpus_size": None})
    assert compute_calibration_config_hash({"min_corpus_size": 100}) != compute_calibration_config_hash({"min_corpus_size": None})
    # Fallback to min_corpus_units when min_corpus_size is absent or None
    assert compute_calibration_config_hash({"min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 100})
    assert compute_calibration_config_hash({"min_corpus_size": None, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 100})
    # min_corpus_size takes precedence over min_corpus_units when both present
    assert compute_calibration_config_hash({"min_corpus_size": 50, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 50})
    # Preserves explicit min_corpus_size: 0
    assert compute_calibration_config_hash({"min_corpus_size": 0, "min_corpus_units": 100}) == compute_calibration_config_hash({"min_corpus_size": 0})
    assert compute_calibration_config_hash({"min_corpus_size": 0}) != compute_calibration_config_hash({"min_corpus_size": 100})

    from pydoppelgangerhunt.baseline import _extract_calibration_settings  # pylint: disable=import-outside-toplevel
    assert _extract_calibration_settings({"min_corpus_units": 150})["min_corpus_size"] == 150
    assert _extract_calibration_settings({"min_corpus_size": 0, "min_corpus_units": 150})["min_corpus_size"] == 0
    # Fallback to exclude when excludes is absent or None
    assert _extract_calibration_settings({"exclude": ["vendor/*"]})["excludes"] == ["vendor/*"]
    assert _extract_calibration_settings({"excludes": None, "exclude": ["vendor/*"]})["excludes"] == ["vendor/*"]
    assert _extract_calibration_settings({"excludes": ["build/*"], "exclude": ["vendor/*"]})["excludes"] == ["build/*"]

    # Defensively handle explicit None in calls and vector during calibration
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    calib_calls_none = compute_corpus_calibration([{"calls": None}], call_sequences=True)
    assert calib_calls_none["total_units"] == 1
    calib_vector_none = compute_corpus_calibration([{"vector": None}], bag_of_tokens=True)
    assert calib_vector_none["total_units"] == 1
    # Defensively handle collection without .keys() in vector when bag_of_tokens is unset
    calib_vector_list = compute_corpus_calibration([{"vector": ["t1", "t2"]}], bag_of_tokens=False)
    assert calib_vector_list["total_units"] == 1
    assert "t1" in calib_vector_list["shingle_frequencies"]



def test_baseline_config_hash_and_recorded_commit_roundtrip(tmp_path: Path) -> None:
    """Verifies that record_baseline and load_baseline preserve config_hash and recorded_commit."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_corpus_calibration,
        load_baseline,
        record_baseline,
    )

    baseline_file = str(tmp_path / "baseline_calib_hash.json")
    calib = compute_corpus_calibration([], min_lines=8, bag_of_tokens=True)
    assert "config_hash" in calib
    assert len(calib["config_hash"]) == 64

    saved_path = record_baseline([], baseline_file, str(tmp_path), 0.90, corpus_calibration=calib)
    assert Path(saved_path).exists()

    data = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert "config_hash" in data
    assert data["config_hash"] == calib["config_hash"]
    assert "corpus_calibration" in data
    assert data["corpus_calibration"]["config_hash"] == calib["config_hash"]

    loaded = load_baseline(baseline_file)
    assert loaded.config_hash == calib["config_hash"]
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration.get("config_hash") == calib["config_hash"]


def test_legacy_baseline_loads_without_config_hash_or_recorded_commit(tmp_path: Path) -> None:
    """Verifies that legacy baselines lacking config_hash or recorded_commit load safely."""
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel

    legacy_file = tmp_path / "legacy_v14.json"
    legacy_data = {
        "version": "1.4.0",
        "threshold": 0.85,
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 100,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
        },
    }
    legacy_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    loaded = load_baseline(str(legacy_file))
    assert loaded.corpus_calibration is not None
    assert "config_hash" in loaded.corpus_calibration
    assert loaded.recorded_commit is None


def test_get_git_head_commit_behavior(tmp_path: Path) -> None:
    """Verifies get_git_head_commit resolution in git repos and outside git repos."""
    from pydoppelgangerhunt.git_diff import get_git_head_commit  # pylint: disable=import-outside-toplevel

    # Inside current git repo
    head_commit = get_git_head_commit()
    assert head_commit is not None
    assert len(head_commit) in (40, 64)
    assert all(c in "0123456789abcdef" for c in head_commit)

    # Outside git repo
    empty_dir = tmp_path / "not_a_git_repo"
    empty_dir.mkdir()
    assert get_git_head_commit(repo_root=empty_dir) is None


def test_baseline_metadata_type_validation_with_crafted_values(tmp_path: Path) -> None:
    """Verifies that non-string/numeric config_hash and recorded_commit degrade safely without raising in load_baseline."""
    crafted_path = tmp_path / "crafted_baseline.json"
    crafted_data = {
        "version": "1.5.0",
        "fingerprints": [],
        "config_hash": 123456789,
        "recorded_commit": 987654321,
        "corpus_calibration": {
            "total_units": 100,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
            "config_hash": 123456789,
            "recorded_commit": 987654321,
        },
    }
    crafted_path.write_text(json.dumps(crafted_data), encoding="utf-8")

    loaded = load_baseline(str(crafted_path))
    assert isinstance(loaded.config_hash, str)
    assert len(loaded.config_hash) == 64
    assert loaded.config_hash[:8]  # Can slice without TypeError
    assert loaded.recorded_commit is None  # Degraded safely from numeric value
    assert loaded.corpus_calibration is not None
    assert isinstance(loaded.corpus_calibration.get("config_hash"), str)
    assert loaded.corpus_calibration.get("recorded_commit") is None


def test_compute_calibration_config_hash_lossless_float() -> None:
    """Verifies that compute_calibration_config_hash hashes floats without lossy 6-decimal rounding."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    hash_a = compute_calibration_config_hash({"max_index_frequency": 0.25})
    hash_b = compute_calibration_config_hash({"max_index_frequency": 0.2500004})
    assert hash_a != hash_b


def test_compute_calibration_config_hash_malformed_frequency_fallback() -> None:
    """Verifies that unparseable max_index_frequency values fall back to 0.25 in config hash."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel

    hash_default = compute_calibration_config_hash({"max_index_frequency": 0.25})
    hash_invalid_str = compute_calibration_config_hash({"max_index_frequency": "invalid_frequency"})
    hash_invalid_bool = compute_calibration_config_hash({"max_index_frequency": True})
    hash_invalid_neg = compute_calibration_config_hash({"max_index_frequency": -0.5})
    assert hash_invalid_str == hash_default
    assert hash_invalid_bool == hash_default
    assert hash_invalid_neg == hash_default

    # Explicit None remains distinct
    hash_none = compute_calibration_config_hash({"max_index_frequency": None})
    assert hash_none != hash_default


def test_matches_boundary_and_structural_hashes_no_suffix_fallback(tmp_path: Path) -> None:
    """Verifies that _matches_boundary_and_structural_hashes disables suffix fallback when resolver is provided."""
    from pydoppelgangerhunt.canonical_path import CanonicalPathResolver  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import _matches_boundary_and_structural_hashes  # pylint: disable=import-outside-toplevel

    resolver = CanonicalPathResolver(target_root=tmp_path / "pkg", repo_root=tmp_path)
    # Different files: worker.py in target vs pkg/worker.py in repo
    res = _matches_boundary_and_structural_hashes(
        "worker.py", "other.py", "hash1", "hash2",
        "pkg/other_pkg/worker.py", "other.py", "hash1", "hash2",
        resolver=resolver,
    )
    assert not res


def test_calibration_compatibility_audit_tests_and_include_notebooks() -> None:
    """Verifies that audit_tests and include_notebooks are validated in calibration compatibility."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    calib_std = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "audit_tests": False,
        "include_notebooks": False,
        "config_hash": compute_calibration_config_hash({"audit_tests": False, "include_notebooks": False}),
    }
    assert _is_calibration_mode_compatible(calib_std, audit_tests=False, include_notebooks=False)
    assert not _is_calibration_mode_compatible(calib_std, audit_tests=True, include_notebooks=False)
    assert not _is_calibration_mode_compatible(calib_std, audit_tests=False, include_notebooks=True)


def test_prune_baseline_legacy_upgrade_adds_path_basis(tmp_path: Path) -> None:
    """Verifies that pruning a legacy v1.0–v1.4 baseline infers and preserves path_basis when upgrading to v1.5.0."""
    from pydoppelgangerhunt.baseline import load_baseline, prune_baseline  # pylint: disable=import-outside-toplevel

    legacy_file_repo = tmp_path / "legacy_v1_repo_rel.json"
    legacy_data_repo = {
        "version": "1.0.0",
        "created_at": "2026-01-01T00:00:00Z",
        "target": "src",
        "threshold": 0.90,
        "clone_count": 1,
        "fingerprints": [
            {
                "fingerprint": "src/a.py:f1 <===> src/b.py:f2",
                "file_a": "src/a.py",
                "file_b": "src/b.py",
                "name_a": "f1",
                "name_b": "f2",
                "similarity": 0.95,
            }
        ],
    }
    legacy_file_repo.write_text(json.dumps(legacy_data_repo), encoding="utf-8")

    prune_res = prune_baseline(str(legacy_file_repo), active_clones=[], unstaged_modified_ranges={})
    assert prune_res.pruned_count == 1

    raw_saved = json.loads(legacy_file_repo.read_text(encoding="utf-8"))
    assert raw_saved["version"] == "1.5.0"
    assert raw_saved["path_basis"] == "repo_relative"

    loaded = load_baseline(str(legacy_file_repo))
    assert loaded.path_basis == "repo_relative"

    legacy_file_target = tmp_path / "legacy_v1_target_rel.json"
    legacy_data_target = {
        "version": "1.0.0",
        "created_at": "2026-01-01T00:00:00Z",
        "target": "src",
        "threshold": 0.90,
        "clone_count": 1,
        "fingerprints": [
            {
                "fingerprint": "a.py:f1 <===> b.py:f2",
                "file_a": "a.py",
                "file_b": "b.py",
                "name_a": "f1",
                "name_b": "f2",
                "similarity": 0.95,
            }
        ],
    }
    legacy_file_target.write_text(json.dumps(legacy_data_target), encoding="utf-8")

    prune_res_target = prune_baseline(str(legacy_file_target), active_clones=[], unstaged_modified_ranges={})
    assert prune_res_target.pruned_count == 1

    raw_saved_target = json.loads(legacy_file_target.read_text(encoding="utf-8"))
    assert raw_saved_target["version"] == "1.5.0"
    assert raw_saved_target["path_basis"] == "target_relative"

    loaded_target = load_baseline(str(legacy_file_target))
    assert loaded_target.path_basis == "target_relative"


def test_prune_baseline_preserves_legacy_repo_relative_records_across_runs(tmp_path: Path) -> None:
    """Verifies that pruning a legacy repo-relative baseline preserves coordinates without false pruning on future loads."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_unit_structural_hash,
        filter_clones_by_baseline,
        load_baseline,
        prune_baseline,
    )

    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    file_a = src_dir / "worker.py"
    file_b = src_dir / "helper.py"
    code_a = "def worker(x):\n    return x * 2 + 1\n"
    code_b = "def helper(y):\n    return y * 2 + 1\n"
    file_a.write_text(code_a, encoding="utf-8")
    file_b.write_text(code_b, encoding="utf-8")

    u1 = {"file": "worker.py", "name": "worker", "code": code_a}
    u2 = {"file": "helper.py", "name": "helper", "code": code_b}
    u1["structural_hash"] = compute_unit_structural_hash(u1)
    u2["structural_hash"] = compute_unit_structural_hash(u2)
    active_clones = [(1.0, u1, u2)]

    # Legacy baseline v1.0.0 with repo-relative paths (src/worker.py, src/helper.py) and no path_basis
    legacy_file = repo_dir / "baseline.json"
    legacy_data = {
        "version": "1.0.0",
        "created_at": "2026-01-01T00:00:00Z",
        "target": "src",
        "threshold": 0.90,
        "clone_count": 1,
        "fingerprints": [
            {
                "file_a": "src/worker.py",
                "file_b": "src/helper.py",
                "name_a": "worker",
                "name_b": "helper",
                "hash_a": u1["structural_hash"],
                "hash_b": u2["structural_hash"],
                "similarity": 1.0,
            }
        ],
    }
    legacy_file.write_text(json.dumps(legacy_data), encoding="utf-8")

    # Run 1: prune baseline with active target-relative clones
    prune_res_1 = prune_baseline(
        str(legacy_file),
        active_clones=active_clones,
        unstaged_modified_ranges={},
        repo_root=str(repo_dir),
        target="src",
    )
    assert prune_res_1.pruned_count == 0
    assert prune_res_1.retained_count == 1

    raw_saved = json.loads(legacy_file.read_text(encoding="utf-8"))
    assert raw_saved["version"] == "1.5.0"
    assert raw_saved["path_basis"] == "repo_relative"

    # Verify that future loads interpret coordinates correctly and suppress active clones
    loaded = load_baseline(str(legacy_file))
    assert loaded.path_basis == "repo_relative"
    new_clones, suppressed_count = filter_clones_by_baseline(
        active_clones,
        loaded,
        repo_root=str(repo_dir),
        target="src",
    )
    assert suppressed_count == 1
    assert len(new_clones) == 0

    # Run 2: prune baseline again - live record must NOT be pruned
    prune_res_2 = prune_baseline(
        str(legacy_file),
        active_clones=active_clones,
        unstaged_modified_ranges={},
        repo_root=str(repo_dir),
        target="src",
    )
    assert prune_res_2.pruned_count == 0
    assert prune_res_2.retained_count == 1


def test_scan_target_full_scan_with_calibration_avoids_double_counting(tmp_path: Path) -> None:
    """Verifies that full scan with corpus_calibration does not double-count corpus units or frequencies."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    shared_code = (
        "def common_worker(x):\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    c = b + 3\n"
        "    d = c + 4\n"
        "    e = d + 5\n"
        "    f = e + 6\n"
        "    g = f + 7\n"
        "    return g\n"
    )
    for i in range(8):
        (tmp_path / f"shared_{i}.py").write_text(shared_code, encoding="utf-8")
    for i in range(22):
        unique_code = (
            f"def unique_{i}():\n"
            f"    alpha_{i} = 'hello'\n"
            f"    beta_{i} = 'world'\n"
            f"    gamma_{i} = 'foo'\n"
            f"    delta_{i} = 'bar'\n"
            f"    epsilon_{i} = 'baz'\n"
            f"    zeta_{i} = 'qux'\n"
            f"    eta_{i} = 'test'\n"
            f"    return alpha_{i} + beta_{i} + gamma_{i} + delta_{i} + epsilon_{i} + zeta_{i} + eta_{i}\n"
        )
        (tmp_path / f"unique_{i}.py").write_text(unique_code, encoding="utf-8")

    _, calib = scan_target(str(tmp_path), max_index_frequency=0.28, return_calibration=True)
    assert calib["total_units"] == 30

    clones_full = scan_target(
        str(tmp_path),
        max_index_frequency=0.28,
        corpus_calibration=calib,
        diff_files=None,
    )
    clones_uncalib = scan_target(
        str(tmp_path),
        max_index_frequency=0.28,
        corpus_calibration=None,
        diff_files=None,
    )
    assert len(clones_full) == 28
    assert len(clones_uncalib) == 28


def test_cross_scope_baseline_matching_prevents_false_suppression(tmp_path: Path) -> None:
    """Verifies that a baseline recorded under a subdirectory does not falsely suppress same-named root files."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        BaselineFingerprints,
        filter_clones_by_baseline,
    )

    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)

    # Baseline recorded under src/ with target_repo_relative="src"
    base_record = {
        "fingerprint": "service.py:run <===> worker.py:work",
        "structural_fingerprint": "service.py#hash_srv <===> worker.py#hash_wrk",
        "namespaced_structural_fingerprint": ".#hash_srv <===> .#hash_wrk",
        "pure_structural_fingerprint": "hash_srv <===> hash_wrk",
        "file_a": "service.py",
        "name_a": "run",
        "hash_a": "hash_srv",
        "file_b": "worker.py",
        "name_b": "work",
        "hash_b": "hash_wrk",
    }
    base_fps = BaselineFingerprints(
        fps={base_record["fingerprint"]},
        records=[base_record],
        target=str(src_dir),
        target_repo_relative="src",
    )

    # Active scan at repo root produces clones at repo root (outside src)
    u_root_1 = {"file": "service.py", "name": "run", "structural_hash": "hash_srv"}
    u_root_2 = {"file": "worker.py", "name": "work", "structural_hash": "hash_wrk"}
    root_clones = [(1.0, u_root_1, u_root_2)]

    remaining, supp = filter_clones_by_baseline(
        root_clones, base_fps, repo_root=str(repo_dir), target=str(repo_dir)
    )
    # The clone outside src must NOT be suppressed
    assert supp == 0
    assert len(remaining) == 1

    # But a clone at src/service.py and src/worker.py in the root scan MUST be suppressed
    u_in_src_1 = {"file": "src/service.py", "name": "run", "structural_hash": "hash_srv"}
    u_in_src_2 = {"file": "src/worker.py", "name": "work", "structural_hash": "hash_wrk"}
    src_clones = [(1.0, u_in_src_1, u_in_src_2)]

    remaining_src, supp_src = filter_clones_by_baseline(
        src_clones, base_fps, repo_root=str(repo_dir), target=str(repo_dir)
    )
    assert supp_src == 1
    assert len(remaining_src) == 0


def test_calibration_compatibility_with_excludes() -> None:
    """Verifies that excludes are preserved and compared during calibration compatibility checks."""
    from pydoppelgangerhunt.baseline import compute_corpus_calibration  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    unit = {
        "file": "foo.py",
        "name": "bar",
        "shingles": ["sh1", "sh2"],
        "tokens": ["def", "bar"],
    }
    calib = compute_corpus_calibration([unit], excludes=["vendor", "tests"])
    assert calib.get("excludes") == ["tests", "vendor"]

    # Compatible when active excludes match
    assert _is_calibration_mode_compatible(calib, excludes=["vendor", "tests"])
    assert _is_calibration_mode_compatible(calib, excludes=["tests", "vendor"])

    # Incompatible when active excludes differ or are empty
    assert not _is_calibration_mode_compatible(calib, excludes=None)
    assert not _is_calibration_mode_compatible(calib, excludes=["vendor"])
    assert not _is_calibration_mode_compatible(calib, excludes=["other"])


def test_canonicalize_endpoint_path_with_path_basis() -> None:
    """Verifies that _canonicalize_endpoint_path relies on path_basis rather than prefix heuristics."""
    from pydoppelgangerhunt.baseline import _canonicalize_endpoint_path  # pylint: disable=import-outside-toplevel

    # Target-relative path coincidentally starting with offset name must be prepended
    res = _canonicalize_endpoint_path("src/module.py", "src", path_basis="target_relative")
    assert res == "src/src/module.py"

    # Repo-relative path must not be prepended even if offset is provided
    res_repo = _canonicalize_endpoint_path("src/module.py", "src", path_basis="repo_relative")
    assert res_repo == "src/module.py"

    res_worktree = _canonicalize_endpoint_path("src/module.py", "src", path_basis="worktree_relative")
    assert res_worktree == "src/module.py"

    # Normal target-relative path without coincident prefix
    res_normal = _canonicalize_endpoint_path("module.py", "src", path_basis="target_relative")
    assert res_normal == "src/module.py"

    # Empty offset
    res_no_off = _canonicalize_endpoint_path("src/module.py", None, path_basis="target_relative")
    assert res_no_off == "src/module.py"


def test_same_target_worktree_baseline_suppression(tmp_path: Path) -> None:
    """Verifies that recording and filtering a baseline on the same subdirectory in a git repo suppresses clones."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    f1 = src_dir / "a.py"
    f2 = src_dir / "b.py"
    code = "def worker():\n    x = 1\n    y = 2\n    z = x + y\n    print(z)\n    return z\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    # Record baseline targeting src
    base_file = repo_dir / "baseline.json"
    u1 = {"file": "a.py", "name": "worker", "structural_hash": "h1"}
    u2 = {"file": "b.py", "name": "worker", "structural_hash": "h2"}
    clones = [(1.0, u1, u2)]

    pydoppelgangerhunt.record_baseline(
        clones,
        str(base_file),
        str(src_dir),
        0.80,
    )

    base_fps = load_baseline(str(base_file))
    assert getattr(base_fps, "path_basis", None) == "target_relative"

    # Active scan also targeting src
    # CLI resolves git_worktree_root as repo_dir and passes target=src_dir
    filtered, suppressed = filter_clones_by_baseline(
        clones,
        base_fps,
        repo_root=str(repo_dir),
        target=str(src_dir),
    )
    assert suppressed == 1
    assert len(filtered) == 0


def test_matches_boundary_and_structural_hashes_strict_repo_coordinates() -> None:
    """Verifies that _matches_boundary_and_structural_hashes does not conflate root and subdirectory files."""
    from pydoppelgangerhunt.baseline import _matches_boundary_and_structural_hashes  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.canonical_path import CanonicalPathResolver  # pylint: disable=import-outside-toplevel

    resolver = CanonicalPathResolver(target_root="/workspace/src", repo_root="/workspace")

    # False: Root file "foo.py" vs Subdirectory file "src/foo.py" even with identical structural hash
    assert not _matches_boundary_and_structural_hashes(
        r_fa="foo.py",
        r_fb="bar.py",
        r_ha="hash_123",
        r_hb="hash_456",
        c_fa="src/foo.py",
        c_fb="src/bar.py",
        c_ha="hash_123",
        c_hb="hash_456",
        resolver=resolver,
    )

    # True: Matching repo-relative coordinates
    assert _matches_boundary_and_structural_hashes(
        r_fa="src/foo.py",
        r_fb="src/bar.py",
        r_ha="hash_123",
        r_hb="hash_456",
        c_fa="src/foo.py",
        c_fb="src/bar.py",
        c_ha="hash_123",
        c_hb="hash_456",
        resolver=resolver,
    )

    # True: Symmetric endpoints reversed
    assert _matches_boundary_and_structural_hashes(
        r_fa="src/foo.py",
        r_fb="src/bar.py",
        r_ha="hash_123",
        r_hb="hash_456",
        c_fa="src/bar.py",
        c_fb="src/foo.py",
        c_ha="hash_456",
        c_hb="hash_123",
        resolver=resolver,
    )

    # True: Identical hashes reversed
    assert _matches_boundary_and_structural_hashes(
        r_fa="src/foo.py",
        r_fb="src/bar.py",
        r_ha="hash_same",
        r_hb="hash_same",
        c_fa="src/bar.py",
        c_fb="src/foo.py",
        c_ha="hash_same",
        c_hb="hash_same",
        resolver=resolver,
    )


def test_prune_baseline_cross_scope_isolation(tmp_path: Path) -> None:
    """Verifies prune_baseline isolates scopes and prunes subdirectory items when only root file matches."""
    repo_dir = tmp_path / "scope_repo"
    repo_dir.mkdir()
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    # Record baseline for src target with worker.py
    base_file = repo_dir / "baseline.json"
    u1 = {"file": "worker.py", "name": "worker_func", "structural_hash": "h_same"}
    u2 = {"file": "worker_clone.py", "name": "worker_func", "structural_hash": "h_same"}
    clones = [(1.0, u1, u2)]

    pydoppelgangerhunt.record_baseline(
        clones,
        str(base_file),
        str(src_dir),
        0.80,
    )

    # Active scan on repo root has worker.py at ROOT (outside src/), with same name & hash
    root_u1 = {"file": "worker.py", "name": "worker_func", "structural_hash": "h_same"}
    root_u2 = {"file": "worker_clone.py", "name": "worker_func", "structural_hash": "h_same"}
    active_root_clones = [(1.0, root_u1, root_u2)]

    # Pruning baseline against root scan should NOT retain the src/ baseline item
    res = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_root_clones,
        repo_root=str(repo_dir),
        target=str(repo_dir),
        unstaged_modified_ranges={},
    )
    assert res.pruned_count == 1
    assert res.retained_count == 0


def test_prune_baseline_cross_root_preserves_path_coordinates(tmp_path: Path) -> None:
    """Verifies prune_baseline does not rewrite target-relative paths to repo paths on cross-root prune."""
    repo_dir = tmp_path / "cross_repo"
    repo_dir.mkdir()
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    base_file = repo_dir / "baseline.json"
    u1 = {"file": "worker.py", "name": "worker_func", "structural_hash": "h_orig"}
    u2 = {"file": "worker_clone.py", "name": "worker_func", "structural_hash": "h_orig"}
    clones = [(1.0, u1, u2)]

    pydoppelgangerhunt.record_baseline(
        clones,
        str(base_file),
        str(src_dir),
        0.80,
    )

    # Active scan at repo root matches src/worker.py via boundary match
    active_u1 = {"file": "src/worker.py", "name": "worker_func", "structural_hash": "h_orig"}
    active_u2 = {"file": "src/worker_clone.py", "name": "worker_func", "structural_hash": "h_orig"}
    active_clones = [(1.0, active_u1, active_u2)]

    res = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_clones,
        repo_root=str(repo_dir),
        target=str(repo_dir),
        unstaged_modified_ranges={},
    )
    assert res.retained_count == 1
    # Check that file_a remains target-relative ("worker.py") and was not rewritten to "src/worker.py"
    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert data["fingerprints"][0]["file_a"] == "worker.py"


def test_cli_default_target_baseline_suppression(tmp_path: Path) -> None:
    """Verifies _apply_baseline_and_diff_filters correctly uses target when args.target is None."""
    import argparse  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import _apply_baseline_and_diff_filters  # pylint: disable=import-outside-toplevel

    repo_dir = tmp_path / "default_repo"
    repo_dir.mkdir()
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    base_file = repo_dir / "baseline.json"
    u1 = {"file": "a.py", "name": "func", "structural_hash": "h1"}
    u2 = {"file": "b.py", "name": "func", "structural_hash": "h1"}
    clones = [(1.0, u1, u2)]

    pydoppelgangerhunt.record_baseline(
        clones,
        str(base_file),
        str(src_dir),
        0.80,
    )

    # Simulated args where args.target is None (omitted on CLI)
    args = argparse.Namespace(
        target=None,
        baseline=str(base_file),
        prune_baseline=False,
        diff_only=False,
        since=None,
        format="text",
    )

    # When target=str(src_dir) is passed, baseline suppresses the clone
    filtered, exit_code = _apply_baseline_and_diff_filters(
        clones,
        args,
        tool_cfg={},
        target_repo_root=str(src_dir),
        target=str(src_dir),
    )
    assert exit_code is None
    assert len(filtered) == 0


def test_compute_path_offset_single_file_target(tmp_path: Path) -> None:
    """Verifies that _compute_path_offset and _derive_target_offsets normalize file targets to their parent directory."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        _compute_path_offset,
        _derive_target_offsets,
    )

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    src = repo / "src"
    src.mkdir(parents=True)
    file_a = src / "worker.py"
    file_a.write_text("def run():\n    pass\n", encoding="utf-8")

    # _compute_path_offset normalizes file target to parent directory
    offset = _compute_path_offset(file_a, repo)
    assert offset == "src"

    # _derive_target_offsets derives scan_offset="src" for a single-file target
    base_off, scan_off = _derive_target_offsets(
        base_target=src,
        base_target_rel="src",
        repo_root=repo,
        target=file_a,
    )
    assert base_off == "src"
    assert scan_off == "src"

    # Clone recorded with target_repo_relative="src" is suppressed when scanning single file file_a
    base_file = repo / "baseline.json"
    u1 = {"file": "worker.py", "name": "run", "structural_hash": "hash_a"}
    u2 = {"file": "worker.py", "name": "run2", "structural_hash": "hash_a"}
    clones = [(1.0, u1, u2)]

    pydoppelgangerhunt.record_baseline(
        clones,
        str(base_file),
        str(file_a),
        0.80,
        repo_root=repo,
    )
    loaded_base = pydoppelgangerhunt.load_baseline(str(base_file))
    assert loaded_base.target_repo_relative == "src"

    filtered, suppressed = pydoppelgangerhunt.filter_clones_by_baseline(
        clones,
        loaded_base,
        repo_root=str(repo),
        target=str(file_a),
    )
    assert suppressed == 1
    assert len(filtered) == 0


def test_calibration_scope_compatibility_and_rejection() -> None:
    """Verifies that corpus calibration scope is validated and rejected when target scopes differ."""
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    # Base calibration recorded for "src"
    calib_src = {
        "total_units": 100,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "scope": "src",
        "target_repo_relative": "src",
    }
    calib_src["config_hash"] = compute_calibration_config_hash(calib_src)

    # Active scan on root repo (target_scope=None) must be rejected
    assert not _is_calibration_mode_compatible(calib_src, target_scope=None)
    assert not _is_calibration_mode_compatible(calib_src, target_scope="")

    # Active scan on "tests" must be rejected
    assert not _is_calibration_mode_compatible(calib_src, target_scope="tests")

    # Active scan on "src" matches
    assert _is_calibration_mode_compatible(calib_src, target_scope="src")

    # Base calibration recorded for root repo (scope=None)
    calib_root = {
        "total_units": 500,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "scope": None,
    }
    calib_root["config_hash"] = compute_calibration_config_hash(calib_root)

    # Active scan on "src" must be rejected
    assert not _is_calibration_mode_compatible(calib_root, target_scope="src")

    # Active scan on root repo matches
    assert _is_calibration_mode_compatible(calib_root, target_scope=None)
    assert _is_calibration_mode_compatible(calib_root, target_scope="")

    # Config hashes differ between scopes
    hash_src = compute_calibration_config_hash({"scope": "src"})
    hash_root = compute_calibration_config_hash({"scope": None})
    assert hash_src != hash_root


def test_calibration_persisted_fields_validated_against_config_hash() -> None:
    """Verifies that _is_calibration_mode_compatible does not accept calibrations whose stored fields differ from config_hash."""
    from typing import Dict, Any  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_calibration_config_hash  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    valid_calib: Dict[str, Any] = {
        "total_units": 50,
        "max_index_frequency": 0.25,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
        "bag_of_tokens": False,
        "call_sequences": False,
        "filter_stop_shingles": False,
        "scope": "src",
    }
    valid_calib["config_hash"] = compute_calibration_config_hash(valid_calib)

    # 1. Matches when persisted fields match config_hash and active scan matches
    assert _is_calibration_mode_compatible(valid_calib, bag_of_tokens=False, target_scope="src")

    # 2. Desynchronized field: config_hash was for bag_of_tokens=False, but stored field was edited to True
    desync_calib = dict(valid_calib)
    desync_calib["bag_of_tokens"] = True
    # Fast-path must NOT accept desync_calib even if active scan has bag_of_tokens=False (matching config_hash)
    assert not _is_calibration_mode_compatible(desync_calib, bag_of_tokens=False, target_scope="src")

    # 3. Desynchronized scope: config_hash was for scope="src", but stored field was edited to "tests"
    desync_scope = dict(valid_calib)
    desync_scope["scope"] = "tests"
    assert not _is_calibration_mode_compatible(desync_scope, bag_of_tokens=False, target_scope="src")


def test_detect_clone_path_basis() -> None:
    """Verifies that _detect_clone_path_basis distinguishes repo-relative from target-relative clone endpoints."""
    from pydoppelgangerhunt.baseline import _detect_clone_path_basis  # pylint: disable=import-outside-toplevel

    # 1. Explicit basis overrides
    assert _detect_clone_path_basis([], "src", explicit_basis="repo_relative") == "repo_relative"
    assert _detect_clone_path_basis([], "src", explicit_basis="repo") == "repo_relative"
    assert _detect_clone_path_basis([], "src", explicit_basis="worktree_relative") == "repo_relative"
    assert _detect_clone_path_basis([], "src", explicit_basis="target_relative") == "target_relative"
    assert _detect_clone_path_basis([], "src", explicit_basis="target") == "target_relative"

    # 2. No scan offset -> default target_relative
    assert _detect_clone_path_basis([], None) == "target_relative"
    assert _detect_clone_path_basis([], "") == "target_relative"

    # 3. Tuple/list clones
    c_repo = (
        0.95,
        {"file": "src/worker.py", "name": "work"},
        {"file": "src/helper.py", "name": "help"},
    )
    c_target = (
        0.95,
        {"file": "worker.py", "name": "work"},
        {"file": "helper.py", "name": "help"},
    )
    assert _detect_clone_path_basis([c_repo], "src") == "repo_relative"
    assert _detect_clone_path_basis([c_target], "src") == "target_relative"

    # 4. Dict clones
    d_repo = {
        "unit_a": {"file": "src/worker.py", "name": "work"},
        "unit_b": {"file": "src/helper.py", "name": "help"},
    }
    d_target = {
        "unit_a": {"file": "worker.py", "name": "work"},
        "unit_b": {"file": "helper.py", "name": "help"},
    }
    assert _detect_clone_path_basis([d_repo], "src") == "repo_relative"
    assert _detect_clone_path_basis([d_target], "src") == "target_relative"


def test_filter_clones_by_baseline_with_repo_relative_clones(tmp_path: Path) -> None:
    """Verifies filter_clones_by_baseline suppresses clones whether active endpoints are repo- or target-relative."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        filter_clones_by_baseline,
        record_baseline,
        load_baseline,
    )

    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)

    u1_t = {"file": "worker.py", "name": "work", "structural_hash": "hash1"}
    u2_t = {"file": "helper.py", "name": "help", "structural_hash": "hash2"}
    target_clones = [(0.95, u1_t, u2_t)]

    # Record baseline for src targeting directory
    bp_path = str(repo_dir / ".baseline.json")
    record_baseline(
        target_clones,
        bp_path,
        str(src_dir),
        0.90,
        repo_root=str(repo_dir),
    )
    loaded_base = load_baseline(bp_path)

    # Scenario A: Clones harvested relative to repo_root (src/worker.py)
    u1_r = {"file": "src/worker.py", "name": "work", "structural_hash": "hash1"}
    u2_r = {"file": "src/helper.py", "name": "help", "structural_hash": "hash2"}
    repo_clones = [(0.95, u1_r, u2_r)]

    rem_repo, supp_repo = filter_clones_by_baseline(
        repo_clones,
        loaded_base,
        repo_root=str(repo_dir),
        target=str(src_dir),
    )
    assert supp_repo == 1
    assert len(rem_repo) == 0

    # Scenario B: Clones harvested relative to target (worker.py)
    rem_target, supp_target = filter_clones_by_baseline(
        target_clones,
        loaded_base,
        repo_root=str(repo_dir),
        target=str(src_dir),
    )
    assert supp_target == 1
    assert len(rem_target) == 0

    # Scenario C: Explicit clone_basis override
    rem_exp, supp_exp = filter_clones_by_baseline(
        repo_clones,
        loaded_base,
        repo_root=str(repo_dir),
        target=str(src_dir),
        clone_basis="repo_relative",
    )
    assert supp_exp == 1
    assert len(rem_exp) == 0


def test_prune_baseline_with_repo_relative_clones(tmp_path: Path) -> None:
    """Verifies prune_baseline correctly matches and retains records when active clones are repo-relative."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        prune_baseline,
        record_baseline,
        load_baseline,
    )

    repo_dir = tmp_path / "repo_prune"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)

    u1_t = {"file": "worker.py", "name": "work", "structural_hash": "hash1"}
    u2_t = {"file": "helper.py", "name": "help", "structural_hash": "hash2"}
    u3_t = {"file": "dead.py", "name": "old_func", "structural_hash": "hash_dead"}
    initial_clones = [(0.95, u1_t, u2_t), (0.90, u1_t, u3_t)]

    bp_path = str(repo_dir / ".baseline.json")
    record_baseline(
        initial_clones,
        bp_path,
        str(src_dir),
        0.90,
        repo_root=str(repo_dir),
    )

    # Active clones only contain the active pair, but formatted as repo-relative: "src/worker.py"
    u1_r = {"file": "src/worker.py", "name": "work", "structural_hash": "hash1"}
    u2_r = {"file": "src/helper.py", "name": "help", "structural_hash": "hash2"}
    active_repo_clones = [(0.95, u1_r, u2_r)]

    prune_res = prune_baseline(
        bp_path,
        active_repo_clones,
        repo_root=str(repo_dir),
        target=str(src_dir),
    )
    # The active clone must be retained, and the orphaned clone (dead.py) must be pruned
    assert prune_res.retained_count == 1
    assert prune_res.pruned_count == 1

    reloaded = load_baseline(bp_path)
    assert len(reloaded.records) == 1
    # File paths in baseline must remain target_relative if base_basis was target_relative
    assert reloaded.records[0]["file_a"] in ("worker.py", "helper.py")


def test_record_baseline_single_file_in_git_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that record_baseline captures recorded_commit for a single-file target in a Git worktree."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        record_baseline,
        load_baseline,
    )

    repo_dir = tmp_path / "git_single_file_repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    target_file = src_dir / "worker.py"
    target_file.write_text("def work(): pass\n", encoding="utf-8")

    norm_repo = str(repo_dir).replace("\\", "/")
    head_hash = "abcdef0123456789abcdef0123456789abcdef01"

    def mock_run_git(args: Any, cwd: Any = None) -> Any:
        if cwd is not None and not Path(cwd).is_dir():
            raise NotADirectoryError(f"Cwd must be a directory, got: {cwd}")
        if args == ["rev-parse", "--show-toplevel"]:
            return norm_repo + "\n"
        if args == ["rev-parse", "HEAD"]:
            return head_hash + "\n"
        return None

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_git)

    u1 = {"file": "worker.py", "name": "work", "structural_hash": "h1"}
    u2 = {"file": "worker.py", "name": "work2", "structural_hash": "h2"}
    clones = [(0.95, u1, u2)]

    bp_path = str(repo_dir / ".single_baseline.json")
    # Target is the single file path target_file, with repo_root=None (probes from target)
    record_baseline(
        clones,
        bp_path,
        str(target_file),
        0.90,
    )

    loaded = load_baseline(bp_path)
    assert loaded.recorded_commit == head_hash


def test_cli_record_baseline_subdirectory_persists_target_repo_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that CLI --record-baseline on a subdirectory records target_repo_relative."""
    import json  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo_dir = tmp_path / "cli_git_repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    f1 = src_dir / "a.py"
    f2 = src_dir / "b.py"
    code = "def shared():\n    x = 10\n    y = 20\n    return x + y\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code.replace("shared", "shared2"), encoding="utf-8")

    norm_repo = str(repo_dir).replace("\\", "/")
    head_hash = "1234567890abcdef1234567890abcdef12345678"

    def mock_run_git(args: Any, cwd: Any = None) -> Any:
        if cwd is not None and not Path(cwd).is_dir():
            raise NotADirectoryError(f"Cwd must be a directory, got: {cwd}")
        if args == ["rev-parse", "--show-toplevel"]:
            return norm_repo + "\n"
        if args == ["rev-parse", "HEAD"]:
            return head_hash + "\n"
        return None

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_git)

    baseline_out = repo_dir / "sub_baseline.json"
    exit_code = main([
        str(src_dir),
        "--record-baseline", str(baseline_out),
        "--min-lines", "3",
        "--min-tokens", "5",
        "--threshold", "0.80",
    ])
    assert exit_code == 0
    assert baseline_out.exists()

    raw_data = json.loads(baseline_out.read_text(encoding="utf-8"))
    assert raw_data.get("target_repo_relative") == "src"
    assert raw_data.get("recorded_commit") == head_hash


def test_run_git_command_normalizes_file_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _run_git_command normalizes cwd when pointing to an existing file."""
    import subprocess  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from typing import List, Optional  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import _run_git_command  # pylint: disable=import-outside-toplevel

    f = tmp_path / "dummy.py"
    f.write_text("print('hello')", encoding="utf-8")

    captured_cwds: List[Optional[str]] = []

    def mock_subprocess_run(*args: Any, **kwargs: Any) -> Any:
        captured_cwds.append(kwargs.get("cwd"))
        proc = mock.MagicMock()
        proc.returncode = 0
        proc.stdout = "ok\n"
        return proc

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    res = _run_git_command(["status"], cwd=str(f))
    assert res == "ok\n"
    assert len(captured_cwds) == 1
    # Effective cwd must be the directory containing dummy.py, not dummy.py itself
    assert captured_cwds[0] == str(tmp_path)


def test_record_baseline_normalizes_repo_relative_clones_to_target_relative(tmp_path: Path) -> None:
    """Verifies that record_baseline normalizes repo-relative clones to target-relative endpoints."""
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "foo.py").write_text("def f():\n    return 42\n", encoding="utf-8")
    (src / "bar.py").write_text("def g():\n    return 42\n", encoding="utf-8")

    baseline_file = repo / "baseline.json"
    u1 = {"file": "src/foo.py", "name": "f", "structural_hash": "a1b2c3d4e5f60718"}
    u2 = {"file": "src/bar.py", "name": "g", "structural_hash": "a1b2c3d4e5f60718"}
    clones = [(0.95, u1, u2)]

    bp = record_baseline(
        clones,
        str(baseline_file),
        target=str(src),
        threshold=0.90,
        repo_root=str(repo),
    )
    assert Path(bp).exists()
    data = json.loads(Path(bp).read_text(encoding="utf-8"))
    assert data["path_basis"] == "target_relative"
    assert data["target_repo_relative"] == "src"
    rec = data["fingerprints"][0]
    # Endpoints in baseline must be target-relative
    assert rec["file_a"] == "foo.py"
    assert rec["file_b"] == "bar.py"
    assert rec["namespace_a"] == "."
    assert rec["namespace_b"] == "."

    # 1. Verify suppression when caller provides repo-relative clones (e.g. from scan_target with git root)
    base_fps = load_baseline(str(baseline_file))
    rem_clones, suppressed = filter_clones_by_baseline(
        clones,
        base_fps,
        repo_root=str(repo),
        target=str(src),
    )
    assert suppressed == 1
    assert len(rem_clones) == 0

    # 2. Verify suppression when caller provides target-relative clones (e.g. from CLI scan)
    target_clones = [
        (0.95, {"file": "foo.py", "name": "f", "structural_hash": "a1b2c3d4e5f60718"},
               {"file": "bar.py", "name": "g", "structural_hash": "a1b2c3d4e5f60718"})
    ]
    rem_target, supp_target = filter_clones_by_baseline(
        target_clones,
        base_fps,
        repo_root=str(repo),
        target=str(src),
        clone_basis="target_relative",
    )
    assert supp_target == 1
    assert len(rem_target) == 0


def test_detect_clone_path_basis_filesystem_disambiguation(tmp_path: Path) -> None:
    """Verifies that _detect_clone_path_basis disambiguates ambiguous prefixes via filesystem probes."""
    from pydoppelgangerhunt.baseline import _detect_clone_path_basis  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    nested = src / "src" / "foo.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("x = 1\n", encoding="utf-8")

    # Candidate unit file is 'src/foo.py'. Under target repo/src, this refers to repo/src/src/foo.py
    # Under repo, repo/src/foo.py does NOT exist.
    clones_target = [
        (0.90, {"file": "src/foo.py", "name": "u1"}, {"file": "src/foo.py", "name": "u2"})
    ]
    basis_detected = _detect_clone_path_basis(
        clones_target,
        scan_offset="src",
        repo_root=str(repo),
        target=str(src),
    )
    assert basis_detected == "target_relative"

    # Now create real repo/src/real.py
    real_file = src / "real.py"
    real_file.write_text("y = 2\n", encoding="utf-8")
    # repo/src/real.py exists under repo, but src/src/real.py does not exist under src
    clones_repo = [
        (0.90, {"file": "src/real.py", "name": "v1"}, {"file": "src/real.py", "name": "v2"})
    ]
    basis_repo_detected = _detect_clone_path_basis(
        clones_repo,
        scan_offset="src",
        repo_root=str(repo),
        target=str(src),
    )
    assert basis_repo_detected == "repo_relative"


def test_detect_clone_path_basis_ambiguous_probes_retains_target_relative(tmp_path: Path) -> None:
    """Verifies that _detect_clone_path_basis retains target-relative convention when both probes succeed."""
    from pydoppelgangerhunt.baseline import _detect_clone_path_basis  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_ambig"
    src = repo / "src"
    # Create BOTH repo/src/ambig.py AND repo/src/src/ambig.py
    top_file = src / "ambig.py"
    top_file.parent.mkdir(parents=True, exist_ok=True)
    top_file.write_text("a = 1\n", encoding="utf-8")

    nested_file = src / "src" / "ambig.py"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("b = 2\n", encoding="utf-8")

    clones = [
        (0.95, {"file": "src/ambig.py", "name": "u1"}, {"file": "src/ambig.py", "name": "u2"})
    ]

    # Without explicit basis: ambiguous filesystem probes retain target-relative harvest convention
    ambig_basis = _detect_clone_path_basis(
        clones,
        scan_offset="src",
        repo_root=str(repo),
        target=str(src),
    )
    assert ambig_basis == "target_relative"

    # With explicit basis: explicit basis is honored
    assert _detect_clone_path_basis(
        clones,
        scan_offset="src",
        explicit_basis="repo_relative",
        repo_root=str(repo),
        target=str(src),
    ) == "repo_relative"
    assert _detect_clone_path_basis(
        clones,
        scan_offset="src",
        explicit_basis="target_relative",
        repo_root=str(repo),
        target=str(src),
    ) == "target_relative"


def test_detect_clone_path_basis_notebook_anchors(tmp_path: Path) -> None:
    """Verifies that _detect_clone_path_basis strips notebook anchors before probing filesystem."""
    from pydoppelgangerhunt.baseline import _detect_clone_path_basis  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_nb"
    src = repo / "src"
    nb = src / "analysis.ipynb"
    nb.parent.mkdir(parents=True)
    nb.write_text("{}", encoding="utf-8")

    # Clones with #cell_1 fragment anchor referencing repo-relative path repo/src/analysis.ipynb
    clones = [
        (0.90, {"file": "src/analysis.ipynb#cell_1", "name": "u1"}, {"file": "src/analysis.ipynb#cell_2", "name": "u2"})
    ]
    basis = _detect_clone_path_basis(
        clones,
        scan_offset="src",
        repo_root=str(repo),
        target=str(src),
    )
    assert basis == "repo_relative"


def test_detect_clone_path_basis_nonexistent_directory_target(tmp_path: Path) -> None:
    """Verifies that _detect_clone_path_basis preserves non-existent directory targets without collapsing to parent."""
    from pydoppelgangerhunt.baseline import _detect_clone_path_basis  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_virtual"
    repo.mkdir()
    target_virtual_dir = repo / "virtual_subpkg"
    # virtual_subpkg does NOT exist on disk as a directory
    assert not target_virtual_dir.exists()

    clones = [
        (0.90, {"file": "mod.py", "name": "u1"}, {"file": "mod.py", "name": "u2"})
    ]
    basis = _detect_clone_path_basis(
        clones,
        scan_offset="virtual_subpkg",
        repo_root=str(repo),
        target=str(target_virtual_dir),
    )
    assert basis == "target_relative"


def test_probe_directory_normalization_case_insensitive(tmp_path: Path) -> None:
    """Verifies that probe directory and offset resolution normalize non-existent files with uppercase extensions."""
    from pydoppelgangerhunt.baseline import _compute_path_offset, _derive_target_offsets  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_case"
    repo.mkdir()
    # Non-existent files with uppercase extensions
    sub_file = str(repo / "src" / "worker.PY")
    offset = _compute_path_offset(sub_file, str(repo))
    assert offset == "src"

    b_offset, s_offset = _derive_target_offsets(
        base_target=str(repo / "src" / "mod.IPYNB"),
        base_target_rel=None,
        repo_root=str(repo),
        target=str(repo / "src" / "other.PY"),
    )
    assert b_offset == "src"
    assert s_offset == "src"


def test_record_baseline_resolves_absolute_endpoints(tmp_path: Path) -> None:
    """Verifies record_baseline resolves absolute file paths to target-relative paths in baseline."""
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "a.py"
    f2 = src / "b.py"
    f1.write_text("def foo():\n    pass\n", encoding="utf-8")
    f2.write_text("def foo():\n    pass\n", encoding="utf-8")

    u1 = {"file": str(f1.resolve()), "name": "foo", "start": 1, "end": 2, "source": "def foo(): pass"}
    u2 = {"file": str(f2.resolve()), "name": "foo", "start": 1, "end": 2, "source": "def foo(): pass"}
    clones = [(1.0, u1, u2)]

    base_file = tmp_path / "baseline.json"
    record_baseline(
        clones,
        str(base_file),
        target=str(src),
        repo_root=str(repo),
        threshold=0.9,
    )

    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["path_basis"] == "target_relative"
    rec = data["fingerprints"][0]
    assert rec["file_a"] == "a.py"
    assert rec["file_b"] == "b.py"

    loaded = load_baseline(str(base_file))
    suppressed, count = filter_clones_by_baseline(
        clones,
        loaded,
        target=str(src),
        repo_root=str(repo),
    )
    assert count == 1
    assert len(suppressed) == 0


def test_record_baseline_finalizes_calibration_scope_and_hash(tmp_path: Path) -> None:
    """Verifies record_baseline applies final target scope before hashing and recomputes config_hash."""
    from pydoppelgangerhunt.baseline import (  # pylint: disable=import-outside-toplevel
        compute_calibration_config_hash,
        compute_corpus_calibration,
    )
    from pydoppelgangerhunt.matcher import _is_calibration_mode_compatible  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "a.py"
    f2 = src / "b.py"
    f1.write_text("def run():\n    pass\n", encoding="utf-8")
    f2.write_text("def run():\n    pass\n", encoding="utf-8")

    u1 = {"file": "a.py", "name": "run", "shingles": ["run"]}
    u2 = {"file": "b.py", "name": "run", "shingles": ["run"]}
    clones = [(1.0, u1, u2)]

    # 1. Create corpus calibration without explicit scope (scope defaults to None)
    calib_raw = compute_corpus_calibration([u1, u2])
    assert calib_raw.get("scope") is None
    initial_hash = calib_raw.get("config_hash")
    assert initial_hash is not None

    # 2. Record baseline for subdirectory src/ within repo
    base_file = tmp_path / "baseline_calib.json"
    record_baseline(
        clones,
        str(base_file),
        target=str(src),
        repo_root=str(repo),
        threshold=0.9,
        corpus_calibration=calib_raw,
    )

    with open(base_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Scope in calibration entry must be finalized to "src" and not overwritten to None
    persisted_calib = data["corpus_calibration"]
    assert persisted_calib["scope"] == "src"
    assert persisted_calib["target_repo_relative"] == "src"

    # config_hash must be recomputed from the finalized entry (different from initial_hash with scope=None)
    final_expected_hash = compute_calibration_config_hash(persisted_calib)
    assert persisted_calib["config_hash"] == final_expected_hash
    assert data["config_hash"] == final_expected_hash
    assert persisted_calib["config_hash"] != initial_hash

    # 3. Reload baseline and verify compatibility
    loaded = load_baseline(str(base_file))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["scope"] == "src"
    assert loaded.corpus_calibration["config_hash"] == final_expected_hash

    # Must be compatible with active subdirectory scan (target_scope="src")
    assert _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope="src")
    # Must reject incompatible target_scope (e.g. root scan target_scope=None)
    assert not _is_calibration_mode_compatible(loaded.corpus_calibration, target_scope=None)


def test_load_baseline_preserves_empty_shingle_frequencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that load_baseline preserves an empty shingle_frequencies dictionary as {} rather than None."""
    from pydoppelgangerhunt import matcher  # pylint: disable=import-outside-toplevel

    base_file_empty = tmp_path / "baseline_empty_freqs.json"
    raw_data_empty: Dict[str, Any] = {
        "version": "1.5.0",
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 0,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "shingle_frequencies": {},
            "min_lines": 4,
            "min_tokens": 10,
        },
    }
    base_file_empty.write_text(json.dumps(raw_data_empty), encoding="utf-8")

    loaded_empty = load_baseline(str(base_file_empty))
    assert loaded_empty.corpus_calibration is not None
    assert isinstance(loaded_empty.corpus_calibration["shingle_frequencies"], dict)
    assert loaded_empty.corpus_calibration["shingle_frequencies"] == {}

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    code1 = (
        "def sample_fn_one(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n"
    )
    code2 = (
        "def sample_fn_two(a, b, c, d):\n"
        "    x = (a * 2) + (b * 3)\n"
        "    y = (c * 4) + (d * 5)\n"
        "    return x + y if x > y else y - x\n"
    )
    (src_dir / "mod1.py").write_text(code1, encoding="utf-8")
    (src_dir / "mod2.py").write_text(code2, encoding="utf-8")

    # In modern empty calibration, novel shingles are budget-bounded
    monkeypatch.setattr(matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 0)
    clones_bounded = scan_target(
        str(src_dir),
        min_lines=4,
        min_tokens=10,
        threshold=0.80,
        corpus_calibration=loaded_empty.corpus_calibration,
    )
    # Budget of 0 rejects novel candidate pairs
    assert len(clones_bounded) == 0

    # Contrast with legacy baseline where shingle_frequencies is omitted (None)
    base_file_legacy = tmp_path / "baseline_legacy.json"
    raw_data_legacy: Dict[str, Any] = {
        "version": "1.4.0",
        "clone_count": 0,
        "fingerprints": [],
        "corpus_calibration": {
            "total_units": 10,
            "max_index_frequency": 0.25,
            "global_stop_shingles": [],
            "min_lines": 4,
            "min_tokens": 10,
        },
    }
    base_file_legacy.write_text(json.dumps(raw_data_legacy), encoding="utf-8")
    loaded_legacy = load_baseline(str(base_file_legacy))
    assert loaded_legacy.corpus_calibration is not None
    assert loaded_legacy.corpus_calibration["shingle_frequencies"] is None

    # In legacy calibration, novel budget capping is bypassed
    clones_legacy = scan_target(
        str(src_dir),
        min_lines=4,
        min_tokens=10,
        threshold=0.80,
        corpus_calibration=loaded_legacy.corpus_calibration,
    )
    assert len(clones_legacy) == 1


def test_prune_baseline_inactive_first_record_no_unbound_local(tmp_path: Path) -> None:
    """Verifies that prune_baseline does not raise UnboundLocalError when the first record is inactive and dirty."""
    from pydoppelgangerhunt.baseline import prune_baseline  # pylint: disable=import-outside-toplevel

    base_json = tmp_path / "baseline.json"
    base_json.write_text(
        json.dumps({
            "version": "1.5.0",
            "clone_count": 2,
            "fingerprints": [
                {
                    "file_a": "pkg/inactive_a.py",
                    "file_b": "pkg/inactive_b.py",
                    "name_a": "fn_a",
                    "name_b": "fn_b",
                    "fingerprint": "pkg/inactive_a.py:fn_a <===> pkg/inactive_b.py:fn_b",
                    "structural_fingerprint": "pkg/inactive_a.py#hash1 <===> pkg/inactive_b.py#hash2",
                },
                {
                    "file_a": "pkg/active_a.py",
                    "file_b": "pkg/active_b.py",
                    "name_a": "fn_active_a",
                    "name_b": "fn_active_b",
                    "fingerprint": "pkg/active_a.py:fn_active_a <===> pkg/active_b.py:fn_active_b",
                    "structural_fingerprint": "pkg/active_a.py#hash_act_1 <===> pkg/active_b.py#hash_act_2",
                },
            ],
        }),
        encoding="utf-8",
    )
    # The first record is NOT in active_clones, but pkg/inactive_a.py is dirty in unstaged_modified_ranges
    u1 = {"file": "pkg/active_a.py", "name": "fn_active_a", "structural_hash": "hash_act_1"}
    u2 = {"file": "pkg/active_b.py", "name": "fn_active_b", "structural_hash": "hash_act_2"}
    prune_res = prune_baseline(
        str(base_json),
        [(1.0, u1, u2)],
        unstaged_modified_ranges={"pkg/inactive_a.py": [(1, 10)]},
    )
    # The inactive dirty record is retained (skipped dirty), and active record is retained
    assert prune_res.retained_count == 2
    assert prune_res.pruned_count == 0

    # Also test legacy string format as first record
    base_json_legacy = tmp_path / "baseline_legacy_str.json"
    base_json_legacy.write_text(
        json.dumps({
            "version": "1.4.0",
            "clone_count": 1,
            "fingerprints": [
                "pkg/legacy_a.py:fn_leg_a <===> pkg/legacy_b.py:fn_leg_b",
            ],
        }),
        encoding="utf-8",
    )
    prune_res_legacy = prune_baseline(
        str(base_json_legacy),
        [],
        unstaged_modified_ranges={"pkg/legacy_a.py": [(1, 10)]},
    )
    assert prune_res_legacy.retained_count == 1
    assert prune_res_legacy.pruned_count == 0


def test_prune_baseline_upgrades_legacy_target_repo_relative(tmp_path: Path) -> None:
    """Verifies that prune_baseline correctly populates target_repo_relative for legacy baselines."""
    repo_dir = tmp_path / "myrepo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    base_file = repo_dir / "baseline.json"

    base_file.write_text(
        json.dumps({
            "version": "1.4.0",
            "target": "src",
            "clone_count": 1,
            "fingerprints": [{
                "file_a": "service.py",
                "file_b": "worker.py",
                "name_a": "fn_a",
                "name_b": "fn_b",
                "structural_hash_a": "hash1",
                "structural_hash_b": "hash2",
                "structural_fingerprint": "service.py#hash1 <===> worker.py#hash2",
            }],
        }),
        encoding="utf-8",
    )

    active = [(1.0, {
        "file": "service.py",
        "name": "fn_a",
        "structural_hash": "hash1",
    }, {
        "file": "worker.py",
        "name": "fn_b",
        "structural_hash": "hash2",
    })]

    res = prune_baseline(
        str(base_file),
        active,
        repo_root=str(repo_dir),
        target=str(src_dir),
    )
    assert res.retained_count == 1
    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert data.get("target_repo_relative") == "src"


def test_match_clone_record_preserves_empty_root_namespace() -> None:
    """Verifies that _match_clone_record does not drop empty string root namespaces in Pass 4."""
    from pydoppelgangerhunt.baseline import _match_clone_record

    rec = {
        "file_a": "main.py",
        "file_b": "pkg/mod.py",
        "namespace_a": "",
        "namespace_b": "pkg",
        "pure_structural_fingerprint": "h1 <===> h2",
        "name_a": "run",
        "name_b": "worker",
    }
    unconsumed = [rec]
    c_keys = {
        "file_a": "renamed_main.py",
        "file_b": "pkg/renamed_mod.py",
        "name_a": "run",
        "name_b": "worker",
        "hash_a": "h1",
        "hash_b": "h2",
        "pure_sfp": "h1 <===> h2",
    }
    matched = _match_clone_record(c_keys, unconsumed)
    assert matched is rec


def test_numeric_sanitizers_reject_booleans() -> None:
    """Verifies that _safe_total_units, _safe_int, _safe_index_frequency, and _sanitize_shingle_frequency_dict reject booleans."""
    from pydoppelgangerhunt.baseline import (
        _safe_index_frequency,
        _safe_int,
        _safe_total_units,
        _sanitize_shingle_frequency_dict,
    )

    assert _safe_total_units(True) == 0
    assert _safe_total_units(False) == 0
    assert _safe_int(True, min_val=1) is None
    assert _safe_int(False, min_val=0) is None
    assert _safe_index_frequency(True) is None
    assert _safe_index_frequency(False) is None

    # Test _sanitize_shingle_frequency_dict explicitly ignores booleans
    raw_freqs = {"valid": 5, "bool_t": True, "bool_f": False, "str_num": "4", "neg": -2, "zero": 0}
    sanitized = _sanitize_shingle_frequency_dict(raw_freqs, key_transform=lambda k: k)
    assert sanitized == {"valid": 5, "str_num": 4}


def test_extract_record_endpoint_data_structural_hash_fallbacks() -> None:
    """Verifies that _extract_record_endpoint_data extracts structural_hash_a/b and structural_hash."""
    from pydoppelgangerhunt.baseline import _extract_record_endpoint_data

    # structural_hash_a and structural_hash_b
    rec1 = {"file_a": "a.py", "file_b": "b.py", "structural_hash_a": "sha", "structural_hash_b": "shb"}
    assert _extract_record_endpoint_data(rec1) == ("a.py", "", "sha", "b.py", "", "shb")

    # structural_hash common fallback
    rec2 = {"file_a": "a.py", "file_b": "b.py", "structural_hash": "sh_common"}
    assert _extract_record_endpoint_data(rec2) == ("a.py", "", "sh_common", "b.py", "", "sh_common")

    # hash_a/hash_b priority over structural_hash_a/b
    rec3 = {
        "file_a": "a.py",
        "file_b": "b.py",
        "hash_a": "ha",
        "hash_b": "hb",
        "structural_hash_a": "sha",
        "structural_hash_b": "shb",
    }
    assert _extract_record_endpoint_data(rec3) == ("a.py", "", "ha", "b.py", "", "hb")


def test_load_baseline_failure_diagnostic_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that load_baseline logs diagnostic debug message on load failure."""
    import logging
    invalid_file = tmp_path / "corrupt_baseline.json"
    invalid_file.write_text("{invalid json", encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        base = load_baseline(str(invalid_file))
        assert len(base) == 0

    assert any("Failed to load baseline at" in record.message for record in caplog.records)


def test_matches_boundary_and_structural_hashes_identical_hashes_swapped_endpoints() -> None:
    """Verifies that _matches_boundary_and_structural_hashes matches swapped endpoints with identical hashes when resolver is None."""
    from pydoppelgangerhunt.baseline import _matches_boundary_and_structural_hashes  # pylint: disable=import-outside-toplevel

    # Both units share identical structural hash 'common_hash'
    # Baseline recorded as r_fa='dir1/mod.py', r_fb='dir2/mod.py'
    # Candidate scan harvested as c_fa='dir2/mod.py', c_fb='dir1/mod.py' (swapped orientation)
    assert _matches_boundary_and_structural_hashes(
        r_fa="dir1/mod.py",
        r_fb="dir2/mod.py",
        r_ha="common_hash",
        r_hb="common_hash",
        c_fa="dir2/mod.py",
        c_fb="dir1/mod.py",
        c_ha="common_hash",
        c_hb="common_hash",
        resolver=None,
    )


def test_extract_record_endpoint_data_intra_file_clone_hash_order() -> None:
    """Verifies that _extract_record_endpoint_data does not invert hashes for intra-file clone pairs."""
    from pydoppelgangerhunt.baseline import _extract_record_endpoint_data  # pylint: disable=import-outside-toplevel

    # Intra-file clone where file_a == file_b == "worker.py"
    # Structural fingerprint formatted as "worker.py#hash_a <===> worker.py#hash_b"
    item = {
        "file_a": "worker.py",
        "file_b": "worker.py",
        "structural_fingerprint": "worker.py#hash_a <===> worker.py#hash_b",
    }
    fa, _, ha, fb, _, hb = _extract_record_endpoint_data(item)
    assert fa == "worker.py"
    assert fb == "worker.py"
    assert ha == "hash_a"
    assert hb == "hash_b"

    # Also verify when file_a and file_b are not provided upfront
    item_no_files = {
        "structural_fingerprint": "worker.py#hash_a <===> worker.py#hash_b",
    }
    fa2, _, ha2, fb2, _, hb2 = _extract_record_endpoint_data(item_no_files)
    assert fa2 == "worker.py"
    assert fb2 == "worker.py"
    assert ha2 == "hash_a"
    assert hb2 == "hash_b"


def test_is_absolute_path_str_drive_strictness() -> None:
    """Verifies that _is_absolute_path_str distinguishes drive-absolute from drive-relative paths."""
    from pydoppelgangerhunt.baseline import _is_absolute_path_str  # pylint: disable=import-outside-toplevel

    assert _is_absolute_path_str("C:/foo.py") is True
    assert _is_absolute_path_str(r"C:\foo.py") is True
    assert _is_absolute_path_str("/foo.py") is True
    assert _is_absolute_path_str(r"\foo.py") is True
    assert _is_absolute_path_str("//server/share/foo.py") is True
    assert _is_absolute_path_str("C:foo.py") is False
    assert _is_absolute_path_str("d:sub/foo.py") is False
    assert _is_absolute_path_str("foo.py") is False
    assert _is_absolute_path_str("") is False



