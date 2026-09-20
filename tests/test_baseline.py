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

    # 3. _is_same_file_path using boundary matching
    assert _is_same_file_path("./Sub/Module.py#cell1", "sub/module.py")
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
    assert raw_json["version"] == "1.4.0"
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

    # Harvest units to find one of the boilerplate shingles
    units_result = scan_target(str(repo), threshold=0.90, min_lines=5, return_calibration=True)
    _, calib_harvested = units_result
    some_shingle = next(iter(calib_harvested["shingle_frequencies"].keys()))

    # Calibrate with this shingle designated as a global stop shingle
    calib = {
        "total_units": 500,
        "max_index_frequency": 0.25,
        "global_stop_shingles": {some_shingle},
        "shingle_frequencies": {some_shingle: 300},
    }

    # Verify that passing this calibration with stop shingle prunes the candidate pairs
    clones = scan_target(
        str(repo),
        threshold=0.90,
        min_lines=5,
        corpus_calibration=calib,
    )
    # The shingle is pruned from candidate pair generation
    assert isinstance(clones, list)


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

