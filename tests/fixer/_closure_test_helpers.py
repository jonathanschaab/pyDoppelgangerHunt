"""Shared test helpers for closure-cell dataflow, symtable, and differential refactoring tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import symtable
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from pydoppelgangerhunt.fixer import (
    generate_refactoring_patch,
    inspect_single_unit_scope,
)
from pydoppelgangerhunt.fixer._df_cache import _suppress_syntax_warnings

__all__ = [
    "_ensure_shared_git_repo",
    "_extract_symtable_scope_info",
    "_run_differential_refactoring_case",
]


def _ensure_shared_git_repo(tmp_path: Path) -> Path:
    """Initializes and returns a reusable git repository root inside tmp_path."""
    repo_dir = tmp_path / "_shared_git_repo"
    if not (repo_dir / ".git").exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "-q"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
        )
    return repo_dir


def _extract_symtable_scope_info(
    source_text: str,
) -> Dict[Tuple[str, int], Dict[str, frozenset[str]]]:
    """Computes C-level symtable locals, frees, globals, and cells keyed by (scope_name, lineno)."""
    try:
        with _suppress_syntax_warnings():
            root_table = symtable.symtable(source_text, "<symtable>", "exec")
    except (SyntaxError, ValueError):
        return {}
    result: Dict[Tuple[str, int], Dict[str, frozenset[str]]] = {}
    stack: List[symtable.SymbolTable] = [root_table]
    while stack:
        tbl = stack.pop()
        if isinstance(tbl, symtable.Function):
            descendant_frees: Set[str] = set()
            desc_stack = list(tbl.get_children())
            while desc_stack:
                child_tbl = desc_stack.pop()
                if isinstance(child_tbl, symtable.Function):
                    descendant_frees.update(child_tbl.get_frees())
                desc_stack.extend(child_tbl.get_children())
            local_set = frozenset(tbl.get_locals())
            global_set = frozenset(tbl.get_globals())
            cells = (local_set - global_set) & frozenset(descendant_frees)
            result[(tbl.get_name(), tbl.get_lineno())] = {
                "locals": local_set,
                "frees": frozenset(tbl.get_frees()),
                "globals": global_set,
                "parameters": frozenset(tbl.get_parameters()),
                "cells": cells,
            }
        stack.extend(tbl.get_children())
    return result


def _run_differential_refactoring_case(
    tmp_path: Path,
    case_name: str,
    fn_src: str,
    u1_bounds: tuple[int, int],
    expect_patch: bool,
    call_args_list: list[tuple[object, ...]],
    *,
    expected_hazard_reason: Optional[str] = None,
) -> None:
    """Executes original and refactored modules and verifies identical return values and hazard reasons."""
    if shutil.which("git") is None:
        pytest.skip("git executable is required for differential patch application")

    case_dir = _ensure_shared_git_repo(tmp_path)
    fn_b_src = fn_src.replace("def entry(", "def entry_b(", 1)
    line_offset = len(fn_src.splitlines()) + 1
    combined_src = f"{fn_src}\n{fn_b_src}"
    mod_file = case_dir / f"{case_name}_mod.py"
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
        [(1.0, u1, u2)], repo_root=str(case_dir), replace_clones=True
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
        for args in call_args_list:
            ns_orig: dict[str, Any] = {}
            ns_ref: dict[str, Any] = {}
            exec(compile(combined_src, "<orig>", "exec"), ns_orig)  # pylint: disable=exec-used  # nosec B102
            exec(compile(refactored_src, "<ref>", "exec"), ns_ref)  # pylint: disable=exec-used  # nosec B102
            res_orig = ns_orig["entry"](*args)
            res_ref = ns_ref["entry"](*args)
            res_ref_b = ns_ref["entry_b"](*args)
            assert res_orig == res_ref, f"Mismatch in {case_name} for args={args}"
            assert res_orig == res_ref_b, f"Mismatch in {case_name} (entry_b) for args={args}"
    else:
        assert patch == "", f"Expected hazard to block patch for {case_name}"
        sc1 = inspect_single_unit_scope(u1, repo_root=str(case_dir))
        assert (
            "severed_closure_cell" in sc1["control_flow_hazards"]
            or not sc1["is_control_flow_safe"]
        ), f"Expected control-flow hazard in scope for {case_name}: {sc1}"
        if "severed_closure_cell" in sc1["control_flow_hazards"]:
            reasons = sc1.get("closure_hazard_reasons", [])
            assert reasons, f"Expected non-empty closure_hazard_reasons for {case_name}"
            assert "unresolved_span_bounds" not in reasons, (
                f"Blocked case {case_name} hit unresolved_span_bounds instead of semantic hazard"
            )
            if expected_hazard_reason is not None:
                assert expected_hazard_reason in reasons
        for args in call_args_list:
            ns_orig_only: dict[str, Any] = {}
            exec(compile(combined_src, "<orig>", "exec"), ns_orig_only)  # pylint: disable=exec-used  # nosec B102
            res_a = ns_orig_only["entry"](*args)
            res_b = ns_orig_only["entry_b"](*args)
            assert res_a == res_b
