"""Shared pytest fixtures and helpers for fixer dataflow and closure-cell tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterator

import pytest

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    generate_refactoring_patch,
)


@pytest.fixture(autouse=True)
def _reset_dataflow_caches() -> Iterator[None]:
    """Clears module-level AST and dataflow caches before and after every test."""
    clear_dataflow_cache()
    yield
    clear_dataflow_cache()


def _run_differential_refactoring_case(
    tmp_path: Path,
    case_name: str,
    fn_src: str,
    u1_bounds: tuple[int, int],
    expect_patch: bool,
    call_args_list: list[tuple[object, ...]],
) -> None:
    """Executes original and refactored modules and verifies identical return values and side effects."""
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

