"""Unit tests for refactoring patch pipeline, overlap handling, rollback safety, and git apply."""

from __future__ import annotations

import ast
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    analyze_unit_variable_scope,
    check_units_overlap,
    extract_unit_source_code,
    generate_refactoring_patch,
    scan_target,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (
    patch as patch_mod,
    resolve_closure_strictness_mode,
)
from pydoppelgangerhunt.fixer.source import (  # pylint: disable=protected-access
    _load_unit_file_text,
)
from pydoppelgangerhunt.fixer.depgraph import build_module_graph
from tests.helpers import assert_valid_python

requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git CLI not available"
)

def test_audit_tests_parametrize_candidate_detection(tmp_path: Path) -> None:
    """Test detection of copy-pasted test functions as @pytest.mark.parametrize candidates."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_example.py"
    test_file.write_text(
        """
def test_calc_tier_low():
    val = 10.0
    res = val * 2.0
    assert res == 20.0

def test_calc_tier_high():
    val = 50.0
    res = val * 2.0
    assert res == 100.0
""",
        encoding="utf-8",
    )
    clones = scan_target(
        str(test_dir),
        threshold=0.80,
        min_lines=3,
        min_tokens=5,
        audit_tests=True,
        blind_literals=True,
    )
    assert len(clones) >= 1
    u1, u2 = clones[0][1], clones[0][2]
    assert u1["name"] == "test_calc_tier_low" or u2["name"] == "test_calc_tier_low"

def test_comment_and_formatting_preservation(tmp_path: Path) -> None:
    """Test that replace_unit_in_source and generate_refactoring_patch preserve comments and formatting."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch, replace_unit_in_source  # pylint: disable=import-outside-toplevel

    source_code = (
        "# Top header license comment\n"
        "# type: ignore[module-attr]\n"
        "\n"
        "def calculate_tax(income: float) -> float:  # inline comment\n"
        "    # Step 1: Base deduction\n"
        "    rate = 0.20\n"
        "    return income * rate\n"
        "\n"
        "# Section divider comment\n"
        "def other_worker():\n"
        "    # worker notes\n"
        "    pass  # noqa: E501\n"
    )
    src_file = tmp_path / "tax_calc.py"
    src_file.write_text(source_code, encoding="utf-8")

    u_calc = {"file": str(src_file), "start": 4, "end": 7, "name": "calculate_tax", "kind": "function"}

    # 1. Test replace_unit_in_source
    call_stmt = "def calculate_tax(income: float) -> float:\n    return _shared_calculate_tax(income)\n"
    replaced = replace_unit_in_source(source_code, u_calc, call_stmt)

    assert "# Top header license comment\n" in replaced
    assert "# type: ignore[module-attr]\n" in replaced
    assert "# Section divider comment\n" in replaced
    assert "# worker notes\n" in replaced
    assert "pass  # noqa: E501\n" in replaced
    assert "return _shared_calculate_tax(income)" in replaced

    # 2. Test generate_refactoring_patch with replace_clones=True
    patch = generate_refactoring_patch([(0.95, u_calc, u_calc)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_calculate_tax" in patch
    assert "# Clone Pair" in patch

def test_generate_refactoring_patch_same_class_compound_blocks(tmp_path: Path) -> None:
    """Verifies refactoring compound blocks within class methods invokes self._shared_helper."""
    f = tmp_path / "processor.py"
    code = (
        "class DataProcessor:\n"
        "    def process_one(self, items: list) -> int:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            total += x\n"
        "        return total\n"
        "\n"
        "    def process_two(self, items: list) -> int:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            total += x\n"
        "        return total * 2\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "process_one:For",
        "kind": "compound_block",
        "enclosing_class": "DataProcessor",
    }
    u2 = {
        "file": str(f),
        "start": 10,
        "end": 11,
        "name": "process_two:For",
        "kind": "compound_block",
        "enclosing_class": "DataProcessor",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Private method in class body
    assert "+    def _shared_process_one_process_two(" in patch
    # Invocation via self._shared_helper omitting self from args
    assert "self._shared_process_one_process_two(" in patch

def test_generate_refactoring_patch_does_not_mutate_unit_is_static(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not mutate caller unit dicts with is_static."""
    code = (
        "class Handler:\n"
        "    @staticmethod\n"
        "    def s_work(x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    def i_work(self, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
    )
    f = tmp_path / "mutate_check.py"
    f.write_text(code, encoding="utf-8")
    u_inst = {"file": str(f), "start": 8, "end": 9, "name": "i_work", "kind": "function"}
    u_stat = {"file": str(f), "start": 3, "end": 4, "name": "s_work", "kind": "function"}

    _ = generate_refactoring_patch(
        [(0.95, u_inst, u_stat)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "is_static" not in u_inst

def test_generate_refactoring_patch_slash_normalization(tmp_path: Path) -> None:
    """Verifies that differing path slash formats normalize to the same file during patch generation."""
    code = (
        "class Normalizer:\n"
        "    def worker_one(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
        "\n"
        "    def worker_two(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
    )
    sub = tmp_path / "pkg"
    sub.mkdir()
    f = sub / "norm.py"
    f.write_text(code, encoding="utf-8")

    # One unit with POSIX slash, one with Windows backslash
    u1 = {"file": "pkg/norm.py", "start": 2, "end": 4, "name": "worker_one", "kind": "function", "enclosing_class": "Normalizer"}
    u2 = {"file": "pkg\\norm.py", "start": 6, "end": 8, "name": "worker_two", "kind": "function", "enclosing_class": "Normalizer"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    # Both units must be replaced in the single file and delegate to the helper
    assert patch.count("+        return self._shared_worker_one_worker_two(x)") == 2

def test_check_units_overlap_same_line_column_bounded() -> None:
    """Verifies that distinct non-overlapping expressions on the same line are not marked as overlapping."""
    u1 = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 4,
        "end_col": 15,
    }
    u2 = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 18,
        "end_col": 29,
    }
    assert check_units_overlap(u1, u2) is False

    u_overlap = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 10,
        "end_col": 20,
    }
    assert check_units_overlap(u1, u_overlap) is True

def test_generate_refactoring_patch_mixed_static_instance_does_not_mark_static(tmp_path: Path) -> None:
    """Verifies that a mixed static/instance clone pair does not evaluate to is_static=True."""
    code = (
        "class Handler:\n"
        "    @staticmethod\n"
        "    def static_calc(a: int, b: int) -> int:\n"
        "        return a * 10 + b\n"
        "\n"
        "    def inst_calc(self, a: int, b: int) -> int:\n"
        "        return a * 10 + b\n"
    )
    f = tmp_path / "mixed.py"
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 3,
        "end": 4,
        "name": "static_calc",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "inst_calc",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "@staticmethod\n    def _shared" not in patch
    assert "@staticmethod\ndef _shared" not in patch
    assert "def _shared_static_calc_inst_calc(a: int, b: int) -> int:" in patch

def test_extracted_helper_short_circuit_walrus_initialization(tmp_path: Path) -> None:
    """Verifies that short-circuit walrus inside extracted clone initializes output = None."""
    code = (
        "def worker_a(cond: bool, data: int) -> int:\n"
        "    if cond and (res := data * 2):\n"
        "        pass\n"
        "    return res\n"
        "\n"
        "def worker_b(cond: bool, data: int) -> int:\n"
        "    if cond and (res := data * 2):\n"
        "        pass\n"
        "    return res\n"
    )
    f = tmp_path / "walrus_sub.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "worker_a:block_L2", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "worker_b:block_L7", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "res = None" in patch

def test_extracted_helper_unconditional_walrus_no_redundant_init(tmp_path: Path) -> None:
    """Verifies that unconditionally evaluated walrus inside extracted clone does not initialize with None."""
    code = (
        "def worker_a(data: int) -> int:\n"
        "    total = (res := data * 2)\n"
        "    return total + res\n"
        "\n"
        "def worker_b(data: int) -> int:\n"
        "    total = (res := data * 2)\n"
        "    return total + res\n"
    )
    f = tmp_path / "walrus_uncond.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 2, "name": "worker_a:block_L2", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 6, "end": 6, "name": "worker_b:block_L6", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "res = None" not in patch

def test_refactoring_patch_global_and_nonlocal_not_assigned_at_call_site(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not generate call-site assignments for globals and declines nonlocals."""
    code_glob = (
        "total = 0\n"
        "def outer():\n"
        "    def inner1():\n"
        "        global total\n"
        "        total += 1\n"
        "    def inner2():\n"
        "        global total\n"
        "        total += 1\n"
    )
    f = tmp_path / "mod_glob.py"
    f.write_text(code_glob, encoding="utf-8")
    u1 = {"file": str(f), "start": 4, "end": 5, "name": "outer:inner1", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "outer:inner2", "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "total = _shared" not in patch
    assert "_shared" in patch

    # Nonlocal-dependent units must decline patch generation to avoid compile-time SyntaxError
    code_nl = (
        "def outer():\n"
        "    acc = 10\n"
        "    def inner1():\n"
        "        nonlocal acc\n"
        "        acc += 2\n"
        "    def inner2():\n"
        "        nonlocal acc\n"
        "        acc += 2\n"
    )
    f_nl = tmp_path / "mod_nl.py"
    f_nl.write_text(code_nl, encoding="utf-8")
    u1_nl = {"file": str(f_nl), "start": 4, "end": 5, "name": "outer:inner1", "kind": "compound_block"}
    u2_nl = {"file": str(f_nl), "start": 7, "end": 8, "name": "outer:inner2", "kind": "compound_block"}
    patch_nl = generate_refactoring_patch(
        [(1.0, u1_nl, u2_nl)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch_nl == ""

def test_generate_refactoring_patch_multi_clone_same_file_git_apply(tmp_path: Path) -> None:
    """Verifies that multiple clone pairs in the same file generate a single unified diff that applies cleanly via git apply."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "def alpha(x: int) -> int:\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    return b\n\n"
        "def beta(y: int) -> int:\n"
        "    a = y * 2\n"
        "    b = a + 1\n"
        "    return b\n\n"
        "def gamma(z: int) -> int:\n"
        "    c = z * 3\n"
        "    d = c + 5\n"
        "    return d\n\n"
        "def delta(w: int) -> int:\n"
        "    c = w * 3\n"
        "    d = c + 5\n"
        "    return d\n"
    )
    mod_file = tmp_path / "multi_mod.py"
    mod_file.write_text(src, encoding="utf-8")

    u_alpha = {"name": "alpha", "file": "multi_mod.py", "start": 1, "end": 4, "kind": "function", "receiver_kind": "none"}
    u_beta = {"name": "beta", "file": "multi_mod.py", "start": 6, "end": 9, "kind": "function", "receiver_kind": "none"}
    u_gamma = {"name": "gamma", "file": "multi_mod.py", "start": 11, "end": 14, "kind": "function", "receiver_kind": "none"}
    u_delta = {"name": "delta", "file": "multi_mod.py", "start": 16, "end": 19, "kind": "function", "receiver_kind": "none"}

    patch = generate_refactoring_patch(
        [(0.95, u_alpha, u_beta), (0.95, u_gamma, u_delta)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    # 1. Exactly one unified diff header for multi_mod.py
    assert patch.count("--- a/multi_mod.py") == 1
    assert patch.count("+++ b/multi_mod.py") == 1
    # 2. Both clone pairs reflected and helpers synthesized
    assert "# Clone Pair (95.0%): multi_mod.py <===> multi_mod.py" in patch
    assert "def _shared_alpha_beta(x: int) -> int:" in patch
    assert "def _shared_gamma_delta(z: int) -> int:" in patch
    assert "return _shared_alpha_beta(x)" in patch
    assert "return _shared_gamma_delta(z)" in patch

    # 3. Verify clean git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    # Verify execution
    run_proc = subprocess.run(
        ["python", "-c", "from multi_mod import alpha, beta, gamma, delta; print(alpha(3), beta(3), gamma(2), delta(2))"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "7 7 11 11"

def test_generate_refactoring_patch_overlapping_units_filtered(tmp_path: Path) -> None:
    """Verifies that overlapping candidate units are filtered without crashing ValueError."""
    src = (
        "def outer(x: int) -> int:\n"
        "    y = x * 2\n"
        "    z = y + 1\n"
        "    return z\n"
    )
    f = tmp_path / "overlap.py"
    f.write_text(src, encoding="utf-8")

    u_outer = {"name": "outer", "file": "overlap.py", "start": 1, "end": 4, "kind": "function"}
    u_inner = {"name": "outer:inner", "file": "overlap.py", "start": 2, "end": 3, "kind": "compound_block"}

    # Pair proposing outer and inner concurrently
    patch = generate_refactoring_patch(
        [(0.90, u_outer, u_outer), (0.85, u_inner, u_inner)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def outer(x: int) -> int:" in patch

def test_multi_class_method_helper_line_adjustment(tmp_path: Path) -> None:
    """Verifies that line shifts from earlier method refactorings do not corrupt helper insertion in later classes."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class ServiceFirst:\n"
        "    def action_one(self, x: int) -> int:\n"
        "        a = x * 2\n"
        "        b = a + 3\n"
        "        return b\n"
        "\n"
        "    def action_two(self, x: int) -> int:\n"
        "        a = x * 2\n"
        "        b = a + 3\n"
        "        return b\n"
        "\n"
        "class ServiceSecond:\n"
        "    def execute_one(self, y: int) -> int:\n"
        "        m = y * 5\n"
        "        n = m + 7\n"
        "        return n\n"
        "\n"
        "    def execute_two(self, y: int) -> int:\n"
        "        m = y * 5\n"
        "        n = m + 7\n"
        "        return n\n"
    )
    f = tmp_path / "services.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "action_one", "file": "services.py", "start": 2, "end": 5, "kind": "function", "enclosing_class": "ServiceFirst"}
    u2 = {"name": "action_two", "file": "services.py", "start": 7, "end": 10, "kind": "function", "enclosing_class": "ServiceFirst"}
    u3 = {"name": "execute_one", "file": "services.py", "start": 13, "end": 16, "kind": "function", "enclosing_class": "ServiceSecond"}
    u4 = {"name": "execute_two", "file": "services.py", "start": 18, "end": 21, "kind": "function", "enclosing_class": "ServiceSecond"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2), (1.0, u3, u4)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from services import ServiceFirst, ServiceSecond; "
            "s1 = ServiceFirst(); s2 = ServiceSecond(); "
            "print(s1.action_one(4), s2.execute_one(3))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "11 22"

def test_super_call_rejected_for_module_binding(tmp_path: Path) -> None:
    """Verifies that methods containing super() are rejected when effective binding is module."""
    src = (
        "class Base:\n"
        "    def run(self) -> str:\n"
        "        return 'base'\n"
        "\n"
        "class ChildA(Base):\n"
        "    def execute(self) -> str:\n"
        "        val = super().run() + '_a'\n"
        "        return val\n"
        "\n"
        "class ChildB(Base):\n"
        "    def execute(self) -> str:\n"
        "        val = super().run() + '_b'\n"
        "        return val\n"
    )
    f = tmp_path / "super_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "execute", "file": "super_mod.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "ChildA"}
    u2 = {"name": "execute", "file": "super_mod.py", "start": 11, "end": 13, "kind": "function", "enclosing_class": "ChildB"}

    # Cross-class clones resolve to module binding -> should reject
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

    # Explicit module binding on same class -> should also reject
    u3 = {"name": "execute", "file": "super_mod.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "ChildA"}
    patch_module = generate_refactoring_patch(
        [(1.0, u1, u3)],
        repo_root=str(tmp_path),
        method_binding="module",
        replace_clones=True,
    )
    assert patch_module == ""

def test_cross_module_global_rejected(tmp_path: Path) -> None:
    """Verifies that clone pairs using global variables across different files are rejected."""
    pkg = tmp_path / "state_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "COUNTER = 0\n"
        "\n"
        "def increment_counter() -> int:\n"
        "    global COUNTER\n"
        "    COUNTER += 1\n"
        "    return COUNTER\n"
    )
    src2 = (
        "COUNTER = 100\n"
        "\n"
        "def advance_counter() -> int:\n"
        "    global COUNTER\n"
        "    COUNTER += 1\n"
        "    return COUNTER\n"
    )
    (pkg / "mod_a.py").write_text(src1, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "increment_counter", "file": "state_pkg/mod_a.py", "start": 3, "end": 6, "kind": "function"}
    u2 = {"name": "advance_counter", "file": "state_pkg/mod_b.py", "start": 3, "end": 6, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_partially_overlapping_inputs_subprocess_execution(tmp_path: Path) -> None:
    """Verifies runtime correctness of refactored Type-2 clones with partially overlapping parameters."""
    code1 = (
        "def eval_math(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def eval_math_alt(a: int, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "math1.py"
    f2 = tmp_path / "math2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "math1.py", "start": 1, "end": 2, "name": "eval_math", "kind": "function"}
    u2 = {"file": "math2.py", "start": 1, "end": 2, "name": "eval_math_alt", "kind": "function"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""

    patch_file = tmp_path / "refactor.patch"
    patch_file.write_text(patch, encoding="utf-8")

    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_file)],
        cwd=tmp_path,
        check=True,
    )

    runner_code = (
        "from math1 import eval_math\n"
        "from math2 import eval_math_alt\n"
        "r1 = eval_math(3, 4)\n"
        "r2 = eval_math_alt(5, 6)\n"
        "assert r1 == 10, f'Expected 10, got {r1}'\n"
        "assert r2 == 16, f'Expected 16, got {r2}'\n"
        "print('PARTIAL_INPUTS_SUCCESS')\n"
    )
    runner_file = tmp_path / "run_test.py"
    runner_file.write_text(runner_code, encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(runner_file)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Execution failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "PARTIAL_INPUTS_SUCCESS" in res.stdout

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_refactoring_subprocess_execution(tmp_path: Path) -> None:
    """End-to-end integration test verifying that refactored pattern matching methods execute cleanly via subprocess."""
    code1 = (
        "def evaluate_code_v1(payload: dict) -> str:\n"
        "    match payload:\n"
        "        case {'status': 'success', 'code': c}:\n"
        "            res = f'OK:{c}'\n"
        "        case _ as fallback:\n"
        "            res = f'FAIL:{fallback.get(\"status\")}'\n"
        "    return res\n"
    )
    code2 = (
        "def evaluate_code_v2(payload: dict) -> str:\n"
        "    match payload:\n"
        "        case {'status': 'success', 'code': c}:\n"
        "            res = f'OK:{c}'\n"
        "        case _ as fallback:\n"
        "            res = f'FAIL:{fallback.get(\"status\")}'\n"
        "    return res\n"
    )
    entry_script = (
        "from mod1 import evaluate_code_v1\n"
        "from mod2 import evaluate_code_v2\n"
        "assert evaluate_code_v1({'status': 'success', 'code': 200}) == 'OK:200'\n"
        "assert evaluate_code_v1({'status': 'error'}) == 'FAIL:error'\n"
        "assert evaluate_code_v2({'status': 'success', 'code': 200}) == 'OK:200'\n"
        "assert evaluate_code_v2({'status': 'error'}) == 'FAIL:error'\n"
        "print('ALL_EVAL_TESTS_PASSED')\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    main_py = tmp_path / "run_eval.py"

    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    main_py.write_text(entry_script, encoding="utf-8")

    u1 = {"file": "mod1.py", "start": 1, "end": 7, "name": "evaluate_code_v1", "kind": "function"}
    u2 = {"file": "mod2.py", "start": 1, "end": 7, "name": "evaluate_code_v2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch != ""

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    proc = subprocess.run(
        [sys.executable, str(main_py)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ALL_EVAL_TESTS_PASSED" in proc.stdout

def test_batch_83_generator_subunit_terminal_return_syntax() -> None:
    """Verifies that _build_unit_delegation_call generates valid Python syntax 'return (yield from ...)' for generator subunits."""
    from pydoppelgangerhunt.fixer.patch import _build_unit_delegation_call  # pylint: disable=import-outside-toplevel

    code = (
        "def gen(items):\n"
        "    for item in items:\n"
        "        yield item\n"
        "    return 42\n"
    )
    lines = code.splitlines(keepends=True)
    target_unit = {"kind": "block", "start": 2, "end": 4}
    scope = {
        "inputs": ["items"],
        "outputs": [],
        "param_details": [{"name": "items"}],
        "has_yield": True,
    }

    call_code = _build_unit_delegation_call(
        target_unit,
        code,
        lines,
        effective_binding="module",
        helper_name="_shared_gen",
        inputs=["items"],
        outputs=[],
        scope=scope,
    )

    assert "return (yield from _shared_gen(items))" in call_code
    assert_valid_python(f"def wrapper():\n    {call_code}\n")
    parsed = ast.parse(f"def wrapper():\n    {call_code}\n")
    assert isinstance(parsed, ast.Module)

def test_generate_refactoring_patch_permuted_outputs_alignment(tmp_path: Path) -> None:
    """Verifies that when clone 2 assigns output variables in permuted order relative to
    clone 1, patch generation fails closed by default to prevent variable swapping."""
    src1 = (
        "def compute_1(a: int, b: int) -> tuple:\n"
        "    x = a * 10\n"
        "    y = b * 20\n"
        "    return x, y\n"
    )
    src2 = (
        "def compute_2(a: int, b: int) -> tuple:\n"
        "    y = b * 20\n"
        "    x = a * 10\n"
        "    return x, y\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {
        "file": "mod1.py",
        "start": 2,
        "end": 3,
        "name": "compute_1:block",
        "kind": "compound_block",
    }
    u2 = {
        "file": "mod2.py",
        "start": 2,
        "end": 3,
        "name": "compute_2:block",
        "kind": "compound_block",
    }

    # Under default allow_permuted_outputs=False, permuted outputs fail closed
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

    # Conflicting / ambiguous orderings with differing variables fail closed
    src3 = (
        "def compute_3(a: int, b: int, c: int) -> tuple:\n"
        "    x = a * 10\n"
        "    y = b * 20\n"
        "    z = c * 30\n"
        "    return x, y, z\n"
    )
    src4 = (
        "def compute_4(a: int, b: int, c: int) -> tuple:\n"
        "    y = b * 20\n"
        "    x = a * 10\n"
        "    w = c * 30\n"
        "    return y, x, w\n"
    )
    f3 = tmp_path / "mod3.py"
    f4 = tmp_path / "mod4.py"
    f3.write_text(src3, encoding="utf-8")
    f4.write_text(src4, encoding="utf-8")
    u3 = {"file": "mod3.py", "start": 2, "end": 4, "name": "c3", "kind": "compound_block"}
    u4 = {"file": "mod4.py", "start": 2, "end": 4, "name": "c4", "kind": "compound_block"}
    patch_conflicting = generate_refactoring_patch(
        [(1.0, u3, u4)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch_conflicting == ""

def test_generate_refactoring_patch_cross_module_auto_creates_common_module(tmp_path: Path) -> None:
    """Verifies that auto cross-module strategy synthesizes _common.py and decouples siblings."""
    pkg_dir = tmp_path / "service_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "def compute_alpha(x: int) -> int:\n"
        "    step1 = x * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    src_b = (
        "def compute_beta(v: int) -> int:\n"
        "    step1 = v * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    f_a = pkg_dir / "srv_a.py"
    f_b = pkg_dir / "srv_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_alpha", "file": "service_pkg/srv_a.py", "start": 1, "end": 4, "kind": "function"}
    u_b = {"name": "compute_beta", "file": "service_pkg/srv_b.py", "start": 1, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # 1. Diff includes both files plus the new shared utility module
    assert "--- a/service_pkg/srv_a.py" in patch
    assert "--- a/service_pkg/srv_b.py" in patch
    assert "diff --git a/service_pkg/_common.py b/service_pkg/_common.py" in patch
    assert "new file mode 100644" in patch
    assert "--- /dev/null" in patch
    assert "+++ b/service_pkg/_common.py" in patch

    # 2. Both files import from service_pkg._common
    assert "from service_pkg._common import _shared_compute_alpha_compute_beta" in patch
    # Sibling modules do not import each other
    assert "from service_pkg.srv_a import" not in patch
    assert "from service_pkg.srv_b import" not in patch

    # 3. Verify git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"
    assert (pkg_dir / "_common.py").is_file()

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from service_pkg.srv_a import compute_alpha; from service_pkg.srv_b import compute_beta; print(compute_alpha(5), compute_beta(5))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "57 57"

def test_generate_refactoring_patch_cross_module_auto_preserves_existing_common_module(tmp_path: Path) -> None:
    """Verifies that auto strategy cleanly appends to an already-existing _common.py without overwriting."""
    pkg_dir = tmp_path / "service_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    existing_common = (
        "def existing_utility() -> str:\n"
        "    return 'preserved'\n"
    )
    (pkg_dir / "_common.py").write_text(existing_common, encoding="utf-8")

    src_a = (
        "def do_calc_a(x: int) -> int:\n"
        "    res = x * 3 + 1\n"
        "    return res\n"
    )
    src_b = (
        "def do_calc_b(x: int) -> int:\n"
        "    res = x * 3 + 1\n"
        "    return res\n"
    )
    (pkg_dir / "calc_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "calc_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "do_calc_a", "file": "service_pkg/calc_a.py", "start": 1, "end": 3, "kind": "function"}
    u_b = {"name": "do_calc_b", "file": "service_pkg/calc_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # Modified existing _common.py (not new file mode)
    assert "new file mode 100644" not in patch
    assert "--- a/service_pkg/_common.py" in patch
    assert "+++ b/service_pkg/_common.py" in patch

    # Apply git diff
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from service_pkg._common import existing_utility; "
            "from service_pkg.calc_a import do_calc_a; "
            "from service_pkg.calc_b import do_calc_b; "
            "print(existing_utility(), do_calc_a(10), do_calc_b(10))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "preserved 31 31"

def test_generate_refactoring_patch_cross_module_host_strategy_success(tmp_path: Path) -> None:
    """Verifies cross-module clone consolidation when cross_file_strategy='host_module' with no cycle."""
    pkg_dir = tmp_path / "host_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "def compute_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    src_caller = (
        "def compute_other(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    (pkg_dir / "host.py").write_text(src_host, encoding="utf-8")
    (pkg_dir / "caller.py").write_text(src_caller, encoding="utf-8")

    u1 = {"name": "compute_data", "file": "host_pkg/host.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_other", "file": "host_pkg/caller.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # host.py defines the helper
    assert "--- a/host_pkg/host.py" in patch
    assert "def _shared_compute_data_compute_other" in patch

    # caller.py imports helper from host_pkg.host
    assert "--- a/host_pkg/caller.py" in patch
    assert "from host_pkg.host import _shared_compute_data_compute_other" in patch

    # Apply diff and verify runtime
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import host_pkg.host; import host_pkg.caller; "
            "print(host_pkg.host.compute_data(4), host_pkg.caller.compute_other(4))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "13 13"

def test_generate_refactoring_patch_cross_module_host_strategy_circular_detection(
    tmp_path: Path,
) -> None:
    """Verifies that host_module strategy detects transitive circular import and logs cycle diagnostic."""
    pkg_dir = tmp_path / "cycle_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    # Cycle chain: host -> mid -> caller
    # If caller imports host, it closes caller -> host -> mid -> caller
    src_caller = (
        "def compute_c(x: int) -> int:\n"
        "    return x + 10\n"
    )
    src_mid = (
        "import cycle_pkg.caller\n\n"
        "def compute_m(x: int) -> int:\n"
        "    return cycle_pkg.caller.compute_c(x)\n"
    )
    src_host = (
        "import cycle_pkg.mid\n\n"
        "def compute_h(x: int) -> int:\n"
        "    return x + 10\n"
    )
    (pkg_dir / "caller.py").write_text(src_caller, encoding="utf-8")
    (pkg_dir / "mid.py").write_text(src_mid, encoding="utf-8")
    (pkg_dir / "host.py").write_text(src_host, encoding="utf-8")

    u_host = {"name": "compute_h", "file": "cycle_pkg/host.py", "start": 3, "end": 4, "kind": "function"}
    u_caller = {"name": "compute_c", "file": "cycle_pkg/caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # Cycle detected: cycle: cycle_pkg.caller -> cycle_pkg.host -> cycle_pkg.mid -> cycle_pkg.caller
    assert (
        "Circular import or unresolvable module path "
        "(cycle: cycle_pkg.caller -> cycle_pkg.host -> cycle_pkg.mid -> cycle_pkg.caller)"
    ) in patch
    # caller.py must NOT be edited with a circular import
    assert "--- a/cycle_pkg/caller.py" not in patch
    assert "from cycle_pkg.host import" not in patch

def test_generate_refactoring_patch_cross_module_missing_caller_file(tmp_path: Path) -> None:
    """Verifies that missing caller file gracefully records refactoring commentary without crash."""
    pkg_dir = tmp_path / "single_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "def compute_alone(x: int) -> int:\n"
        "    return x * 3\n"
    )
    (pkg_dir / "file1.py").write_text(src_f1, encoding="utf-8")

    u1 = {"name": "compute_alone", "file": "single_pkg/file1.py", "start": 1, "end": 2, "kind": "function"}
    u2_missing = {"name": "compute_alone", "file": "single_pkg/nonexistent.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2_missing)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="host_module",
    )

    assert "--- a/single_pkg/file1.py" in patch
    assert "Complete refactoring by importing the helper into single_pkg/nonexistent.py" in patch

def test_generate_refactoring_patch_rejects_out_of_root_caller_path(
    tmp_path: Path,
) -> None:
    """Verifies that an out-of-root caller path is rejected without reading outside files."""
    repo_dir = tmp_path / "safe_repo"
    pkg_dir = repo_dir / "target_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    secret_file = tmp_path / "outside_confidential.py"
    secret_file.write_text("CONFIDENTIAL_TOKEN = 12345\n", encoding="utf-8")

    src_f1 = (
        "def safe_worker(x: int) -> int:\n"
        "    return x + 10\n"
    )
    (pkg_dir / "worker.py").write_text(src_f1, encoding="utf-8")

    u1 = {
        "name": "safe_worker",
        "file": "target_pkg/worker.py",
        "start": 1,
        "end": 2,
        "kind": "function",
    }
    u2_outside = {
        "name": "safe_worker",
        "file": "../outside_confidential.py",
        "start": 1,
        "end": 2,
        "kind": "function",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2_outside)],
        repo_root=str(repo_dir),
        replace_clones=False,
        cross_file_strategy="host_module",
    )

    assert patch == ""
    assert "outside_confidential" not in patch
    assert "CONFIDENTIAL_TOKEN" not in patch

    extracted = extract_unit_source_code(u2_outside, repo_root=str(repo_dir))
    assert not any("CONFIDENTIAL_TOKEN" in line for line in extracted)
    assert extracted == ["# Source for safe_worker lines 1-2\n"]

def test_generate_refactoring_patch_rejects_symlink_caller_path(
    tmp_path: Path,
) -> None:
    """Verifies that a symlinked caller path within root is rejected and skipped."""
    repo_dir = tmp_path / "sym_repo"
    pkg_dir = repo_dir / "inner_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    outside_file = tmp_path / "external_target.py"
    outside_file.write_text("OUTSIDE_SYM_DATA = 999\n", encoding="utf-8")

    symlink_file = pkg_dir / "sym_link.py"
    try:
        symlink_file.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform/privilege level")

    src_f1 = (
        "def process_item(item: int) -> int:\n"
        "    return item * 2\n"
    )
    (pkg_dir / "host.py").write_text(src_f1, encoding="utf-8")

    u1 = {
        "name": "process_item",
        "file": "inner_pkg/host.py",
        "start": 1,
        "end": 2,
        "kind": "function",
    }
    u2_sym = {
        "name": "process_item",
        "file": "inner_pkg/sym_link.py",
        "start": 1,
        "end": 2,
        "kind": "function",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2_sym)],
        repo_root=str(repo_dir),
        replace_clones=False,
        cross_file_strategy="host_module",
    )

    assert patch == ""
    assert "OUTSIDE_SYM_DATA" not in patch

    extracted = extract_unit_source_code(u2_sym, repo_root=str(repo_dir))
    assert not any("OUTSIDE_SYM_DATA" in line for line in extracted)
    assert extracted == ["# Source for process_item lines 1-2\n"]

def test_host_module_no_ghost_edge_when_replace_clones_is_false(tmp_path: Path) -> None:
    """Verifies that replace_clones=False does not insert phantom edges into the dependency graph."""
    pkg = tmp_path / "ghost_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_a = "def run_a(x: int) -> int:\n    return x + 1\n"
    src_b = "def run_b(x: int) -> int:\n    return x + 1\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "run_a", "file": "ghost_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "run_b", "file": "ghost_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    # With replace_clones=False, mod_b is NOT wired to import mod_a
    patch_dry = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="host_module",
    )
    assert "from ghost_pkg.mod_a import" not in patch_dry
    assert "Complete refactoring by importing the helper into ghost_pkg/mod_b.py" in patch_dry

def test_auto_strategy_falls_back_to_host_when_separate_packages(tmp_path: Path) -> None:
    """Verifies that auto strategy uses host_module instead of polluting repo root when files share no common package."""
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    pkg_a.mkdir()
    pkg_b.mkdir()
    (pkg_a / "__init__.py").write_text("", encoding="utf-8")
    (pkg_b / "__init__.py").write_text("", encoding="utf-8")

    src_a = "def process(x: int) -> int:\n    return x * 2 + 10\n"
    src_b = "def handle(x: int) -> int:\n    return x * 2 + 10\n"
    (pkg_a / "srv.py").write_text(src_a, encoding="utf-8")
    (pkg_b / "srv.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "process", "file": "pkg_a/srv.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "handle", "file": "pkg_b/srv.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # In auto mode, separate packages fall back to host_module rather than repo root _common.py
    assert "_common.py" not in patch
    assert "--- a/pkg_a/srv.py" in patch
    assert "--- a/pkg_b/srv.py" in patch
    assert "from pkg_a.srv import _shared_process_handle" in patch

def test_register_plan_dependencies_init_package_context(tmp_path: Path) -> None:
    """Verifies that _register_plan_dependencies_in_graph correctly identifies __init__.py as package context."""
    from pydoppelgangerhunt.fixer.patch import _register_plan_dependencies_in_graph  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer.depgraph import ModuleDependencyGraph  # pylint: disable=import-outside-toplevel

    dg = ModuleDependencyGraph(tmp_path)
    pkg_init = tmp_path / "my_package" / "__init__.py"
    pkg_init.parent.mkdir()
    pkg_init.write_text("", encoding="utf-8")

    deps_file = tmp_path / "my_package" / "deps.py"
    deps_file.write_text("", encoding="utf-8")
    dg.add_module("my_package.deps", deps_file)

    # When registered for __init__.py with 'from .deps import X', it should resolve to my_package.deps
    _register_plan_dependencies_in_graph(dg, "my_package", pkg_init, ["from .deps import X"])

    # dg should have edge my_package -> my_package.deps
    assert "my_package.deps" in dg.get_dependencies("my_package")

def test_host_module_rejects_unresolved_nonbuiltin_dependency(tmp_path: Path) -> None:
    """Verifies that a helper referencing a type only defined in clone 2 file rejects host module extraction."""
    pkg = tmp_path / "pkg_host_unresolved"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def compute_1(x: DepOnlyInTwo) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "class DepOnlyInTwo:\n"
        "    pass\n\n"
        "def compute_2(x: DepOnlyInTwo) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "compute_2", "file": str(f2), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "unresolved symbol 'DepOnlyInTwo' in host module" in patch
    assert "skipping extraction" in patch

def test_generate_refactoring_patch_when_target_is_package_directory(tmp_path: Path) -> None:
    """Verifies that when repo_root is a package directory, imports retain enclosing package context."""
    project_root = tmp_path / "my_project"
    project_root.mkdir()
    top_pkg = project_root / "my_package"
    top_pkg.mkdir()
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")
    sub_pkg = top_pkg / "sub"
    sub_pkg.mkdir()
    (sub_pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def calc_one(x: int) -> int:\n    return x * 2 + 1\n"
    src2 = "def calc_two(x: int) -> int:\n    return x * 2 + 1\n"
    f1 = sub_pkg / "mod1.py"
    f2 = sub_pkg / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc_one", "file": "sub/mod1.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "calc_two", "file": "sub/mod2.py", "start": 1, "end": 2, "kind": "function"}

    # Target scan is passed as the package directory itself (top_pkg)
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(top_pkg),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from my_package.sub._common import _shared_calc_one_calc_two" in patch
    assert "from sub._common import" not in patch
    assert "diff --git a/my_package/sub/_common.py b/my_package/sub/_common.py" in patch
    assert "--- a/my_package/sub/mod1.py" in patch
    assert "+++ b/my_package/sub/mod1.py" in patch
    assert "--- a/my_package/sub/mod2.py" in patch
    assert "+++ b/my_package/sub/mod2.py" in patch

    # Initialize a git repository at project_root and verify that the generated patch
    # applies cleanly from the project root.
    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=project_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=project_root, check=True, capture_output=True)

    patch_file = project_root / "refactor.patch"
    patch_file.write_text(patch, encoding="utf-8")
    apply_res = subprocess.run(["git", "apply", "refactor.patch"], cwd=project_root, capture_output=True, text=True)
    assert apply_res.returncode == 0
    assert (sub_pkg / "_common.py").is_file()
    assert "_shared_calc_one_calc_two" in (sub_pkg / "_common.py").read_text(encoding="utf-8")

def test_generate_refactoring_patch_src_layout_uses_project_root_patch_paths(
    tmp_path: Path,
) -> None:
    """Verifies src-layout package roots keep project-root patch paths while preserving imports."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )

    src_root = project_root / "src"
    top_pkg = src_root / "my_package"
    sub_pkg = top_pkg / "sub"
    sub_pkg.mkdir(parents=True)
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub_pkg / "__init__.py").write_text("", encoding="utf-8")

    src = "def util(a: int) -> int:\n    return a * 2\n"
    (sub_pkg / "a.py").write_text(src, encoding="utf-8")
    (sub_pkg / "b.py").write_text(src, encoding="utf-8")

    u1 = {"name": "util", "file": "sub/a.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "util", "file": "sub/b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(top_pkg),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from my_package.sub._common import _shared_util" in patch
    assert "def _shared_util(a: int) -> int:" in patch
    assert "return a * 2" in patch
    assert (
        "diff --git a/src/my_package/sub/_common.py "
        "b/src/my_package/sub/_common.py"
    ) in patch
    assert "--- a/src/my_package/sub/a.py" in patch
    assert "+++ b/src/my_package/sub/a.py" in patch
    assert "--- a/src/my_package/sub/b.py" in patch
    assert "+++ b/src/my_package/sub/b.py" in patch

    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )

    patch_file = project_root / "refactor.patch"
    patch_file.write_text(patch, encoding="utf-8")
    apply_res = subprocess.run(
        ["git", "apply", "refactor.patch"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert apply_res.returncode == 0
    assert (sub_pkg / "_common.py").is_file()

def test_generate_refactoring_patch_file_repo_root_uses_project_root_patch_paths(
    tmp_path: Path,
) -> None:
    """Verifies file-path repo roots walk up to the project root for patch paths."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )

    lib_root = project_root / "lib"
    top_pkg = lib_root / "my_package"
    top_pkg.mkdir(parents=True)
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")

    src = "def util(a: int) -> int:\n    return a * 2\n"
    module_a = top_pkg / "a.py"
    module_b = top_pkg / "b.py"
    module_a.write_text(src, encoding="utf-8")
    module_b.write_text(src, encoding="utf-8")

    u1 = {"name": "util", "file": "a.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "util", "file": "b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(module_a),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from my_package._common import _shared_util" in patch
    assert "def _shared_util(a: int) -> int:" in patch
    assert "return a * 2" in patch
    assert "diff --git a/lib/my_package/_common.py b/lib/my_package/_common.py" in patch
    assert "--- a/lib/my_package/a.py" in patch
    assert "+++ b/lib/my_package/a.py" in patch
    assert "--- a/lib/my_package/b.py" in patch
    assert "+++ b/lib/my_package/b.py" in patch

    subprocess.run(["git", "init"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )

    patch_file = project_root / "refactor.patch"
    patch_file.write_text(patch, encoding="utf-8")
    apply_res = subprocess.run(
        ["git", "apply", "refactor.patch"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    assert apply_res.returncode == 0
    assert (top_pkg / "_common.py").is_file()

def test_generate_refactoring_patch_strategy_normalization(tmp_path: Path) -> None:
    """Verifies that strategy arguments handle whitespace, casing, and unknown fallback gracefully."""
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("def fn():\n    return 42\n", encoding="utf-8")
    f2.write_text("def fn2():\n    return 42\n", encoding="utf-8")
    u1 = {"name": "fn", "file": "a.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": "b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        cross_file_strategy="  UNKNOWN_STRATEGY  ",
        type_merge_strategy=" STRICT ",
        method_binding=" MODULE ",
    )
    assert "--- a/a.py" in patch
    assert "+++ b/a.py" in patch

def test_generate_refactoring_patch_nested_subdirectory_without_init(tmp_path: Path) -> None:
    """Verifies scanning a nested directory inside a package produces fully-qualified import paths."""
    repo_root = tmp_path / "project"
    pkg = repo_root / "my_pkg"
    tools = pkg / "tools"
    tools.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = tools / "a.py"
    f2 = tools / "b.py"
    f1.write_text("def run(v: int) -> int:\n    return v * 3\n", encoding="utf-8")
    f2.write_text("def run2(v: int) -> int:\n    return v * 3\n", encoding="utf-8")

    u1 = {"name": "run", "file": "a.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "run2", "file": "b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tools),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )
    # Import must be fully qualified under my_pkg, not bare from _common
    assert "from my_pkg.tools._common import _shared_run_run2" in patch
    assert "from _common" not in patch
    # Diff paths must be relative to the enclosing project root (my_pkg/tools/...)
    assert "--- a/my_pkg/tools/a.py" in patch
    assert "+++ b/my_pkg/tools/a.py" in patch
    assert "--- a/my_pkg/tools/b.py" in patch
    assert "+++ b/my_pkg/tools/b.py" in patch
    assert "+++ b/my_pkg/tools/_common.py" in patch

def test_clone_differing_local_builtin_binding_rejected(tmp_path: Path) -> None:
    """Verifies that extraction is rejected when one clone uses a builtin while another binds the name locally."""
    pkg = tmp_path / "builtin_shadow_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "custom_utils.py").write_text("def len(x): return 999\n", encoding="utf-8")

    src1 = (
        "def count_items_1(items: list) -> int:\n"
        "    return len(items)\n"
    )
    src2 = (
        "def count_items_2(items: list) -> int:\n"
        "    from builtin_shadow_pkg.custom_utils import len\n"
        "    return len(items)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "count_items_1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "count_items_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'len' across clone sources" in patch
    assert "skipping extraction" in patch

def test_clone_differing_local_builtin_binding_rejected_same_file(tmp_path: Path) -> None:
    """Verifies that same-file clones where one uses a builtin and another shadows it locally are rejected."""
    f = tmp_path / "same_file_shadow.py"
    src = (
        "def count_items_1(items: list) -> int:\n"
        "    return len(items)\n\n"
        "def count_items_2(items: list) -> int:\n"
        "    from math import isqrt as len\n"
        "    return len(items)\n"
    )
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "count_items_1", "file": str(f), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "count_items_2", "file": str(f), "start": 4, "end": 6, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    assert "conflicting imported symbol 'len' across clone sources" in patch
    assert "skipping extraction" in patch

def test_extract_helper_symbols_scope_resolution() -> None:
    """Verifies that _extract_helper_symbols accurately identifies free, defined, and local names."""
    from pydoppelgangerhunt.fixer.patch import _extract_helper_symbols  # pylint: disable=import-outside-toplevel

    code = """
def my_helper(param1: ExternalType, default_arg=ExternalDefault) -> ReturnType:
    import local_mod
    from local_pkg import local_func
    with external_ctx() as (ctx_a, ctx_b):
        pass
    try:
        risky_op()
    except ExternalError as err:
        handle_err(err)
    class LocalClass(ExternalBase):
        def method(self):
            return self.attr
    walrus_res = (walrus_bound := calc())
    items = [comp_item for comp_item in external_seq]
    return local_func(local_mod.run(ctx_a, LocalClass(), walrus_res, items))
"""
    tree = ast.parse(code)
    free, defined, local_imps = _extract_helper_symbols(tree)
    # Free names should include external dependencies
    assert "ExternalType" in free
    assert "ExternalDefault" in free
    assert "ReturnType" in free
    assert "external_ctx" in free
    assert "risky_op" in free
    assert "ExternalError" in free
    assert "handle_err" in free
    assert "ExternalBase" in free
    assert "calc" in free
    assert "external_seq" in free

    # Locally bound names should NOT be in free names
    assert "param1" not in free
    assert "default_arg" not in free
    assert "ctx_a" not in free
    assert "ctx_b" not in free
    assert "err" not in free
    assert "LocalClass" not in free
    assert "walrus_bound" not in free
    assert "comp_item" not in free

    # Local imports
    assert "local_mod" in local_imps
    assert "local_func" in local_imps
    assert "my_helper" in defined

    if sys.version_info >= (3, 12):
        pep695_tree = ast.parse("def _shared_fn[T: BoundSym](x: T = T, *, k: T = KwSym) -> T:\n    return x\n")
        p_free, p_def, _ = _extract_helper_symbols(pep695_tree)
        assert "BoundSym" in p_free
        assert "T" in p_free
        assert "KwSym" in p_free
        assert "_shared_fn" in p_def

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_extract_helper_symbols_pattern_matching() -> None:
    """Verifies that match-case pattern bindings are recognized as local bindings on Python 3.10+."""
    from pydoppelgangerhunt.fixer.patch import _extract_helper_symbols  # pylint: disable=import-outside-toplevel

    code = """
def match_helper(external_val: Any) -> Any:
    match external_val:
        case [head, *tail]:
            return process(head, tail)
"""
    tree = ast.parse(code)
    free, defined, _ = _extract_helper_symbols(tree)
    assert "external_val" not in free
    assert "process" in free
    assert "head" not in free
    assert "tail" not in free
    assert "match_helper" in defined

def test_generate_refactoring_patch_rejects_internal_symlink_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a clone referencing an internal symlink is rejected rather than modifying the target file."""
    repo = tmp_path / "symlink_repo"
    repo.mkdir()
    (repo / "__init__.py").write_text("", encoding="utf-8")

    real_file = repo / "real.py"
    real_file.write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    link_file = repo / "link.py"

    try:
        link_file.symlink_to(real_file)
    except (OSError, NotImplementedError):
        link_file.write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
        orig_is_symlink = Path.is_symlink

        def mock_is_symlink(self: Path) -> bool:
            if self.name == "link.py":
                return True
            return orig_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    u1 = {"name": "helper", "file": "link.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "helper", "file": "real.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(repo),
        replace_clones=True,
    )

    # Because u1 is link.py, it must be rejected and must NOT generate modifications against real.py
    assert "real.py" not in patch
    assert patch == ""

    # Inverted pair: u2 is real.py (valid) and u1 is link.py (symlink to real.py).
    # Must be rejected before same-file detection and must NOT modify real.py.
    patch_inverted = generate_refactoring_patch(
        [(1.0, u2, u1)],
        repo_root=str(repo),
        replace_clones=True,
    )
    assert "real.py" not in patch_inverted
    assert patch_inverted == ""

def test_generate_refactoring_patch_rejects_absolute_path_outside_repo_root(
    tmp_path: Path,
) -> None:
    """Verifies that an absolute path outside repo_root is rejected even if inside cwd or parent."""
    project = tmp_path / "proj"
    project.mkdir()
    sub_repo = project / "sub_repo"
    sub_repo.mkdir()
    (sub_repo / "__init__.py").write_text("", encoding="utf-8")

    outside_file = project / "outside.py"
    outside_file.write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    inside_file = sub_repo / "inside.py"
    inside_file.write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    u_inside = {"name": "helper", "file": "inside.py", "start": 1, "end": 2, "kind": "function"}
    u_outside = {"name": "helper", "file": str(outside_file), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_inside, u_outside)],
        repo_root=str(sub_repo),
        replace_clones=True,
    )
    assert patch == ""
    assert "outside.py" not in patch
    assert "inside.py" not in patch

def test_resolve_safe_clone_file_path_filesystem_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that _resolve_safe_clone_file_path gracefully catches filesystem errors."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _resolve_safe_clone_file_path,
    )

    repo = tmp_path / "err_repo"
    repo.mkdir()
    f = repo / "target.py"
    f.write_text("x = 1\n", encoding="utf-8")

    orig_is_file = Path.is_file

    def mock_is_file_oserror(self: Path) -> bool:
        if self.name == "target.py":
            raise OSError("Access denied")
        return orig_is_file(self)

    monkeypatch.setattr(Path, "is_file", mock_is_file_oserror)
    res = _resolve_safe_clone_file_path("target.py", [repo])
    assert res is None

def test_resolve_safe_clone_file_path_rejects_sentinel_file_on_disk(
    tmp_path: Path,
) -> None:
    """Verifies that out-of-root paths are rejected even if the sentinel file exists on disk."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _resolve_safe_clone_file_path,
    )

    repo = tmp_path / "sentinel_repo"
    repo.mkdir()
    sentinel_dir = repo / ".git" / ".pydoppelgangerhunt-invalid-path"
    sentinel_dir.mkdir(parents=True)
    sentinel_file = sentinel_dir / "__outside_root__"
    sentinel_file.write_text("# fake sentinel on disk\n", encoding="utf-8")

    res = _resolve_safe_clone_file_path("../outside.py", [repo])
    assert res is None

def test_host_module_cycle_detection_skips_finalization_transactionally(
    tmp_path: Path,
) -> None:
    """Verifies that when a caller-to-host cycle is detected, no code hunks are emitted in patch."""
    pkg = tmp_path / "cyc_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "import cyc_pkg.caller\n\n"
        "def work_host(x: int) -> int:\n"
        "    return x * 2 + 1\n"
    )
    src_caller = (
        "def work_caller(x: int) -> int:\n"
        "    return x * 2 + 1\n"
    )
    (pkg / "host.py").write_text(src_host, encoding="utf-8")
    (pkg / "caller.py").write_text(src_caller, encoding="utf-8")

    u_host = {"name": "work_host", "file": "cyc_pkg/host.py", "start": 3, "end": 4, "kind": "function"}
    u_caller = {"name": "work_caller", "file": "cyc_pkg/caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    assert "--- a/cyc_pkg/caller.py" not in patch
    assert "--- a/cyc_pkg/host.py" not in patch
    assert "Circular import or unresolvable module path" in patch

def test_helper_requires_future_annotations_vararg_and_kwarg() -> None:
    """Verifies that *args and **kwargs annotations with PEP 604 unions trigger future annotations."""
    code_var = "def helper(*items: int | str) -> None:\n    pass\n"
    assert patch_mod._helper_requires_future_annotations(code_var) is True

    code_kw = "def helper(**kwargs: int | float) -> None:\n    pass\n"
    assert patch_mod._helper_requires_future_annotations(code_kw) is True

    code_none = "def helper(*items: int, **kwargs: str) -> None:\n    pass\n"
    assert patch_mod._helper_requires_future_annotations(code_none) is False

def test_extract_module_defined_names_compound_blocks() -> None:
    """Verifies that classes, functions, and variables in top-level if/try/with are discovered."""
    src = (
        "try:\n"
        "    class FastWorker:\n"
        "        pass\n"
        "except ImportError:\n"
        "    def fallback_worker():\n"
        "        pass\n"
        "if True:\n"
        "    CONFIG_VAL: int = 10\n"
        "    def inner_fn():\n"
        "        nested_var = 1\n"
        "        class NestedClass:\n"
        "            pass\n"
    )
    defs = patch_mod._extract_module_defined_names(src)
    assert "FastWorker" in defs
    assert "fallback_worker" in defs
    assert "CONFIG_VAL" in defs
    assert "inner_fn" in defs
    assert "nested_var" not in defs
    assert "NestedClass" not in defs

def test_has_future_annotations_value_error_handling() -> None:
    """Verifies that _has_future_annotations gracefully handles source strings triggering ValueError."""
    assert patch_mod._has_future_annotations("def foo():\n    pass\x00") is False

def test_extract_module_defined_names_ignores_non_store_contexts() -> None:
    """Verifies that attribute and subscript assignment targets do not register load-context names as defined."""
    src = (
        "config.value = 1\n"
        "data[key] = 2\n"
        "registry.items['id'] = 3\n"
        "a, (b, c) = (10, (20, 30))\n"
        "for item in data:\n"
        "    pass\n"
        "with open('foo') as f:\n"
        "    pass\n"
    )
    defined = patch_mod._extract_module_defined_names(src)
    assert "config" not in defined
    assert "data" not in defined
    assert "key" not in defined
    assert "registry" not in defined
    assert "a" in defined
    assert "b" in defined
    assert "c" in defined
    assert "item" in defined
    assert "f" in defined

def test_generate_refactoring_patch_subdirectory_scan_detects_project_cycle(tmp_path: Path) -> None:
    """Verifies that scanning a non-package subdirectory detects circular imports through project-level modules."""
    project = tmp_path / "my_project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")

    tools = project / "tools"
    tools.mkdir()
    script_a = tools / "script_a.py"
    script_b = tools / "script_b.py"

    pkg = project / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    service = pkg / "service.py"

    # script_a defines compute_a and imports pkg.service
    script_a.write_text(
        "import pkg.service\n\n"
        "def compute_a(x: int) -> int:\n"
        "    return x + 42\n",
        encoding="utf-8",
    )
    # script_b defines compute_b (clone of compute_a)
    script_b.write_text(
        "def compute_b(x: int) -> int:\n"
        "    return x + 42\n",
        encoding="utf-8",
    )
    # service imports tools.script_b
    service.write_text(
        "import tools.script_b\n\n"
        "def run_svc() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )

    u1 = {"name": "compute_a", "file": str(script_a), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_b", "file": str(script_b), "start": 1, "end": 2, "kind": "function"}

    # Scanning from the subdirectory 'tools'
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tools),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # A circular import is detected through pkg.service, so host extraction is rejected with advisory comment
    assert "Circular import or unresolvable module path" in patch
    # Verify the cycle path was detected through pkg.service
    assert "pkg.service" in patch
    # script_b must not have an emitted replacement or import
    assert "from script_a import" not in patch
    assert "from tools.script_a import" not in patch

def test_extract_defined_symbols_comprehensive_constructs() -> None:
    """Verifies symbol extraction correctly discovers while loops, match mapping rest, and type aliases."""
    while_code = "while active:\n    x = 10\n"
    names = patch_mod._extract_module_defined_names(while_code)
    assert "x" in names

    if sys.version_info >= (3, 10):
        match_code = (
            "match payload:\n"
            "    case {'key': val, **rest_kwargs}:\n"
            "        pass\n"
        )
        match_names = patch_mod._extract_module_defined_names(match_code)
        assert "val" in match_names
        assert "rest_kwargs" in match_names

    if sys.version_info >= (3, 12):
        alias_code = "type CustomInt = int\n"
        alias_names = patch_mod._extract_module_defined_names(alias_code)
        assert "CustomInt" in alias_names

def test_extract_helper_symbols_nested_walrus_isolation() -> None:
    """Verifies walrus assignments inside nested functions do not leak into helper defined names."""
    code = (
        "def outer(val: int) -> int:\n"
        "    def inner() -> int:\n"
        "        if (inner_walrus := val > 0):\n"
        "            return 1\n"
        "        return 0\n"
        "    return inner()\n"
    )
    tree = ast.parse(code)
    _, defined, _ = patch_mod._extract_helper_symbols(tree.body[0])
    assert "outer" in defined or "inner" in defined
    assert "inner_walrus" not in defined

def test_generate_refactoring_patch_deduplicates_standalone_advisory_comments(
    tmp_path: Path,
) -> None:
    """Verifies duplicate standalone advisory comments are deduplicated across file plans without diffs."""
    pkg = tmp_path / "dedup_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "file1.py"
    f2 = pkg / "file2.py"

    # Both files already import each other causing cycle
    f1.write_text("import dedup_pkg.file2\ndef helper(): return 999\n", encoding="utf-8")
    f2.write_text("import dedup_pkg.file1\ndef helper(): return 999\n", encoding="utf-8")

    u1 = {"name": "helper", "file": str(f1), "start": 2, "end": 2, "kind": "function"}
    u2 = {"name": "helper", "file": str(f2), "start": 2, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # Standalone comment should only appear once in patch output
    assert patch.count("# Note: Cross-module clone pair; helper extraction") == 1

def test_helper_lambda_parameter_scoping_does_not_abort_extraction(tmp_path: Path) -> None:
    """Verifies that lambda parameters are properly scoped and not treated as unresolved symbols."""
    pkg = tmp_path / "lambda_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    f1 = pkg / "mod1.py"
    f2 = pkg / "mod2.py"
    c1 = "def sort_first(items):\n    return sorted(items, key=lambda x: x.val)\n"
    c2 = "def sort_second(items):\n    return sorted(items, key=lambda x: x.val)\n"
    f1.write_text(c1, encoding="utf-8")
    f2.write_text(c2, encoding="utf-8")
    u1 = {"name": "sort_first", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "sort_second", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        cross_file_strategy="shared_module",
    )
    assert "unresolved symbol 'x'" not in patch
    assert "_common.py" in patch

def test_adjust_line_for_replacements_exact_parity() -> None:
    """Verifies that _adjust_line_for_replacements produces exact line numbers matching refactor_module_units."""
    from pydoppelgangerhunt.fixer._patch_delegation import _adjust_line_for_replacements
    from pydoppelgangerhunt.fixer import refactor_module_units

    orig_text = (
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "line 5\n"
        "line 6\n"
        "line 7\n"
        "line 8\n"
        "line 9\n"
        "line 10\n"
    )
    # Edge cases
    assert _adjust_line_for_replacements(0, [], orig_text) == 1
    assert _adjust_line_for_replacements(1, [], orig_text) == 1
    assert _adjust_line_for_replacements(5, [], orig_text) == 5

    # Case 1: Line expansion upstream (lines 3-4 [2 lines] replaced with 5 lines -> +3 delta)
    u_exp = {"file": "t.py", "start": 3, "end": 4}
    r_exp = (u_exp, "r1\nr2\nr3\nr4\nr5\n")
    # Target line 7 should shift by +3 to line 10
    adjusted = _adjust_line_for_replacements(7, [r_exp], orig_text)
    assert adjusted == 10

    # Verify against actual refactored text
    refactored = refactor_module_units(orig_text, [r_exp])
    ref_lines = refactored.splitlines(keepends=True)
    # The original line 7 ("line 7\n") should now be at 1-indexed line 10
    assert ref_lines[adjusted - 1] == "line 7\n"

    # Case 2: Line contraction upstream (lines 2-5 [4 lines] replaced with 1 line -> -3 delta)
    u_con = {"file": "t.py", "start": 2, "end": 5}
    r_con = (u_con, "single_replacement\n")
    # Target line 8 should shift by -3 to line 5
    adjusted_con = _adjust_line_for_replacements(8, [r_con], orig_text)
    assert adjusted_con == 5
    refactored_con = refactor_module_units(orig_text, [r_con])
    ref_con_lines = refactored_con.splitlines(keepends=True)
    assert ref_con_lines[adjusted_con - 1] == "line 8\n"

def test_generate_refactoring_patch_same_file_multiple_methods_and_helpers(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch coordinates reverse-order replacements and method helper insertion."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch

    code = (
        "class ProcessingEngine:\n"
        "    def step_one(self, data: list) -> list:\n"
        "        cleaned = [x.strip() for x in data if x]\n"
        "        validated = [c for c in cleaned if len(c) > 3]\n"
        "        return validated\n"
        "\n"
        "    def step_two(self, data: list) -> list:\n"
        "        cleaned = [x.strip() for x in data if x]\n"
        "        validated = [c for c in cleaned if len(c) > 3]\n"
        "        return validated\n"
    )
    f = tmp_path / "engine.py"
    f.write_text(code, encoding="utf-8")

    tree = ast.parse(code)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    fns.sort(key=lambda n: n.lineno)
    fn1, fn2 = fns[0], fns[1]

    # Target the inner body statements (lines 3-4 and lines 8-9)
    u1 = {
        "name": "ProcessingEngine.step_one",
        "file": str(f),
        "start": fn1.body[0].lineno,
        "end": fn1.body[1].end_lineno,
        "kind": "statement_sequence",
        "enclosing_class": "ProcessingEngine",
        "enclosing_class_start": 1,
    }
    u2 = {
        "name": "ProcessingEngine.step_two",
        "file": str(f),
        "start": fn2.body[0].lineno,
        "end": fn2.body[1].end_lineno,
        "kind": "statement_sequence",
        "enclosing_class": "ProcessingEngine",
        "enclosing_class_start": 1,
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch != ""
    assert "--- a/engine.py" in patch
    assert "+++ b/engine.py" in patch
    assert "def _shared_" in patch
    assert "self._shared_" in patch
    assert "-        cleaned = [x.strip() for x in data if x]" in patch

def test_edge_case_robustness_non_dict_and_invalid_units() -> None:
    """Verifies that non-dict or malformed units raise descriptive errors rather than silently defaulting."""
    from pydoppelgangerhunt.fixer import (
        check_units_overlap,
        compute_unit_char_offsets,
        compute_unit_replacement_span,
        compute_unit_spans,
    )
    from pydoppelgangerhunt.fixer._patch_delegation import _adjust_line_for_replacements
    from pydoppelgangerhunt.fixer.source import _compute_unit_spans

    # Non-dict units raise TypeError
    with pytest.raises(TypeError, match="Units must be dictionaries"):
        check_units_overlap(None, {"file": "a.py", "start": 1, "end": 2})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Units must be dictionaries"):
        check_units_overlap({"file": "a.py", "start": 1, "end": 2}, "invalid")  # type: ignore[arg-type]

    # Malformed unit types in span calculations raise TypeError
    with pytest.raises(TypeError, match="Unit must be a dictionary"):
        compute_unit_spans("line 1\n", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unit must be a dictionary"):
        with pytest.deprecated_call(match="_compute_unit_spans is deprecated"):
            _compute_unit_spans("line 1\n", None)  # type: ignore[arg-type]

    # Replacement span with invalid unit raises TypeError
    with pytest.raises(TypeError, match="Unit must be a dictionary"):
        compute_unit_replacement_span("line 1\n", None, "rep")  # type: ignore[arg-type]

    # Malformed line numbers raise ValueError rather than silently returning false defaults
    with pytest.raises(ValueError, match="Malformed unit"):
        check_units_overlap({"file": "a.py", "start": "bad", "end": 2}, {"file": "a.py", "start": 1, "end": 2})
    with pytest.raises(ValueError, match="Malformed unit"):
        compute_unit_char_offsets("a\nb\nc\n", {"file": "a.py", "start": "bad", "end": 2})

    # Inverted lines in unit
    inv_unit = {"file": "a.py", "start": 5, "end": 2}
    assert compute_unit_char_offsets("a\nb\nc\n", inv_unit) == (6, 6)

    # _adjust_line_for_replacements with invalid target_line or non-dict unit
    adj = _adjust_line_for_replacements("invalid", [], "line 1\n")  # type: ignore[arg-type]
    assert adj == 1
    with pytest.raises(TypeError, match="Unit must be a dictionary"):
        _adjust_line_for_replacements(2, [(None, "rep")], "line 1\n")  # type: ignore[arg-type,list-item]

def test_edge_case_robustness_columns_and_empty_inputs() -> None:
    """Verifies handling of invalid column types, inverted column offsets, and empty inputs."""
    from pydoppelgangerhunt.fixer import (
        check_units_overlap,
        col_offset_to_char_offset,
        compute_unit_byte_offsets,
        refactor_module_units,
    )

    # col_offset_to_char_offset with string or non-numeric offsets
    assert col_offset_to_char_offset("hello world", "5") == 5  # type: ignore[arg-type]
    assert col_offset_to_char_offset("hello world", -10) == 0
    with pytest.raises(ValueError, match="Malformed column offset"):
        col_offset_to_char_offset("hello world", "invalid")  # type: ignore[arg-type]

    # Malformed column offsets in check_units_overlap raise ValueError
    with pytest.raises(ValueError, match="Malformed unit"):
        check_units_overlap(
            {"file": "a.py", "start": 1, "end": 1, "start_col": "invalid"},
            {"file": "a.py", "start": 1, "end": 1, "start_col": 0, "end_col": 10},
        )

    # Inverted column bounds on a single line: end_byte must not precede start_byte
    inv_col_unit = {"file": "a.py", "start": 1, "end": 1, "start_col": 15, "end_col": 5}
    s_b, e_b = compute_unit_byte_offsets("01234567890123456789\n", inv_col_unit)
    assert s_b <= e_b

    # check_units_overlap with string columns
    u_str1 = {"file": "a.py", "start": 1, "end": 1, "start_col": "5", "end_col": "10"}
    u_str2 = {"file": "a.py", "start": 1, "end": 1, "start_col": "8", "end_col": "15"}
    assert check_units_overlap(u_str1, u_str2)

    # Empty source text and empty replacement in refactor_module_units
    assert refactor_module_units("", []) == ""
    assert refactor_module_units("content", []) == "content"
    del_unit = {"file": "a.py", "start": 1, "end": 1}
    assert refactor_module_units("remove this\nkeep this\n", [(del_unit, "")]) == "keep this\n"

def test_dual_tier_overlap_conservative_missing_column_rejection() -> None:
    """Verifies intentional dual-tier overlap behavior: conservative whole-line rejection vs exact byte collision."""
    from pydoppelgangerhunt.fixer import check_units_overlap, refactor_module_units

    # Tier 1: When one unit omits column offsets, check_units_overlap conservatively assumes whole-line conflict
    unit_with_cols = {"file": "mod.py", "start": 3, "end": 3, "start_col": 0, "end_col": 10}
    unit_without_cols = {"file": "mod.py", "start": 3, "end": 3}
    assert check_units_overlap(unit_with_cols, unit_without_cols)
    sample_text = "line1\nline2\ncall_one() and call_two()\nline4\n"
    with pytest.raises(ValueError, match="Overlapping unit collision detected"):
        refactor_module_units(sample_text, [(unit_with_cols, "new_one()"), (unit_without_cols, "new_stmt()")])

    # Tier 1 Acceptance: Disjoint column ranges on the same line are accepted
    unit_left = {"file": "mod.py", "start": 3, "end": 3, "start_col": 0, "end_col": 10}
    unit_right = {"file": "mod.py", "start": 3, "end": 3, "start_col": 15, "end_col": 25}
    assert not check_units_overlap(unit_left, unit_right)
    success_refactor = refactor_module_units(sample_text, [(unit_left, "first_rep"), (unit_right, "second_rep")])
    assert "first_rep and second_rep\n" in success_refactor

    # Tier 2 Catch: Pragma slice expansion extending to line end collides with subsequent unit on same line
    src_expand = "x = 1  # type: ignore\ny = calc() + tail\n"
    unit_calc = {"file": "mod.py", "start": 1, "end": 2, "start_col": 0, "end_col": 10, "kind": "complex_expr"}
    unit_tail = {"file": "mod.py", "start": 2, "end": 2, "start_col": 13, "end_col": 17, "kind": "complex_expr"}
    # Tier 1 allows because line 2 column bounds (10 vs 13) are disjoint:
    assert not check_units_overlap(unit_calc, unit_tail)
    # Tier 2 catches collision because unit_calc extends slice to end of line 2 to attach pragma:
    with pytest.raises(ValueError, match="Overlapping unit collision detected"):
        refactor_module_units(src_expand, [(unit_calc, "new_y"), (unit_tail, "new_tail")])

def test_check_units_overlap_multiline_end_boundary_single_line() -> None:
    """Verifies column-aware overlap detection when a single-line unit shares a multi-line unit's end line."""
    from pydoppelgangerhunt.fixer import check_units_overlap, refactor_module_units

    # Reviewer test case: multi-line unit 1 (lines 1..3, cols 0..10) and single-line unit 2 (line 3, cols 20..30)
    u_multi = {"file": "pkg/mod.py", "start": 1, "end": 3, "start_col": 0, "end_col": 10}
    u_single = {"file": "pkg/mod.py", "start": 3, "end": 3, "start_col": 20, "end_col": 30}
    assert not check_units_overlap(u_multi, u_single)
    assert not check_units_overlap(u_single, u_multi)

    # Overlapping columns on the shared end boundary line
    u_single_overlap = {"file": "pkg/mod.py", "start": 3, "end": 3, "start_col": 5, "end_col": 15}
    assert check_units_overlap(u_multi, u_single_overlap)
    assert check_units_overlap(u_single_overlap, u_multi)

    # Multi-line units overlapping across interior lines (lines 1..3 vs lines 2..3)
    u_interior = {"file": "pkg/mod.py", "start": 2, "end": 3, "start_col": 20, "end_col": 30}
    assert check_units_overlap(u_multi, u_interior)
    assert check_units_overlap(u_interior, u_multi)

    # End-to-end refactoring with non-overlapping multi-line and single-line end-boundary units
    src_code = "first_stmt = 1\nsecond_stmt = 2\nres = calc() + tail\n"
    unit_m = {"file": "pkg/mod.py", "start": 1, "end": 3, "start_col": 0, "end_col": 12, "kind": "complex_expr"}
    unit_s = {"file": "pkg/mod.py", "start": 3, "end": 3, "start_col": 15, "end_col": 19, "kind": "complex_expr"}
    assert not check_units_overlap(unit_m, unit_s)
    refactored = refactor_module_units(src_code, [(unit_m, "fast_res"), (unit_s, "offset")])
    assert "fast_res + offset\n" in refactored

def test_check_units_overlap_multiline_start_boundary_single_line() -> None:
    """Verifies column-aware overlap detection when a single-line unit shares a multi-line unit's start line."""
    from pydoppelgangerhunt.fixer import check_units_overlap

    u_multi = {
        "file": "pkg/mod.py",
        "start": 1,
        "end": 3,
        "start_col": 20,
        "end_col": 30,
    }
    # Single-line unit before multi-line unit's start_col on line 1: disjoint ([5, 15) and [20, EOL))
    u_single_before = {
        "file": "pkg/mod.py",
        "start": 1,
        "end": 1,
        "start_col": 5,
        "end_col": 15,
    }

    assert not check_units_overlap(u_multi, u_single_before)
    assert not check_units_overlap(u_single_before, u_multi)

    # Single-line unit intersecting multi-line unit's start_col on line 1: overlapping ([15, 25) and [20, EOL))
    u_single_overlap = {
        "file": "pkg/mod.py",
        "start": 1,
        "end": 1,
        "start_col": 15,
        "end_col": 25,
    }

    assert check_units_overlap(u_multi, u_single_overlap)
    assert check_units_overlap(u_single_overlap, u_multi)

def test_unit_span_namedtuple_and_compute_unit_spans() -> None:
    """Verifies that UnitSpan carries character/byte bounds and line/col metadata without recomputation."""
    from pydoppelgangerhunt.fixer import UnitSpan, compute_unit_spans, compute_unit_replacement_span

    source = "header = '🚀'; data = [x for x in items]\n"
    unit = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 25,
        "end_col": 42,
        "kind": "comprehension",
    }
    span = compute_unit_spans(source, unit)
    assert isinstance(span, UnitSpan)
    assert span.start_char == 22
    assert span.end_char == 39
    assert span.start_byte == 25
    assert span.end_byte == 42
    assert span.is_column_bounded is True
    assert span.start_line == 1
    assert span.end_line == 1
    assert span.start_col_char == 22
    assert span.end_col_char == 39

    # Verify threading into compute_unit_replacement_span avoids recomputation
    s_c, e_c, rep = compute_unit_replacement_span(source, unit, "helper(items)", unit_span=span)
    assert s_c == 22
    assert e_c == 39
    assert rep == "helper(items)"

def test_adjust_line_for_replacements_additive_deltas() -> None:
    """Verifies that _compute_replacement_line_deltas and _adjust_line_for_replacements compute exact line shifts."""
    from pydoppelgangerhunt.fixer._patch_delegation import (
        _adjust_line_for_replacements,
        _compute_replacement_line_deltas,
    )
    from pydoppelgangerhunt.fixer import refactor_module_units

    orig_text = (
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "line 5\n"
        "line 6\n"
        "line 7\n"
        "line 8\n"
    )
    # Unit 1: lines 2-3 (2 lines) replaced with 4 lines (+2 delta)
    u1 = {"file": "mod.py", "start": 2, "end": 3}
    # Unit 2: lines 5-6 (2 lines) replaced with 1 line (-1 delta)
    u2 = {"file": "mod.py", "start": 5, "end": 6}
    reps = [(u1, "a\nb\nc\nd\n"), (u2, "single\n")]

    deltas = _compute_replacement_line_deltas(reps, orig_text)
    assert deltas == [(3, 2), (6, -1)]

    # Target line 4 (after u1, before u2): should shift by +2 from line 4 to line 6
    adj4 = _adjust_line_for_replacements(4, reps, orig_text)
    assert adj4 == 6

    # Target line 7 (after u1 and u2): net shift is +2 - 1 = +1, so line 7 shifts to line 8
    adj7 = _adjust_line_for_replacements(7, reps, orig_text)
    assert adj7 == 8

    # Parity check against actual refactor_module_units
    refactored = refactor_module_units(orig_text, reps)
    ref_lines = refactored.splitlines(keepends=True)
    assert ref_lines[adj4 - 1] == "line 4\n"
    assert ref_lines[adj7 - 1] == "line 7\n"

def test_check_units_overlap_malformed_inputs_raise_errors() -> None:
    """Verifies that check_units_overlap raises descriptive errors rather than silently returning False on invalid units."""
    from pydoppelgangerhunt.fixer import check_units_overlap

    u_valid = {"file": "mod.py", "start": 1, "end": 5}

    # Non-dictionary units raise TypeError
    with pytest.raises(TypeError, match="Units must be dictionaries"):
        check_units_overlap(123, u_valid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Units must be dictionaries"):
        check_units_overlap(u_valid, "invalid")  # type: ignore[arg-type]

    # Malformed start/end lines raise ValueError
    with pytest.raises(ValueError, match="Malformed unit: invalid line boundary"):
        check_units_overlap({"file": "mod.py", "start": "bad_int", "end": 5}, u_valid)
    with pytest.raises(ValueError, match="Malformed unit: invalid line boundary"):
        check_units_overlap(u_valid, {"file": "mod.py", "start": 1, "end": "bad_int"})

    # Malformed column offsets raise ValueError
    with pytest.raises(ValueError, match="Malformed unit: invalid column offset"):
        check_units_overlap({"file": "mod.py", "start": 1, "end": 5, "start_col": "invalid"}, u_valid)
    with pytest.raises(ValueError, match="Malformed unit: invalid column offset"):
        check_units_overlap(u_valid, {"file": "mod.py", "start": 1, "end": 5, "end_col": "invalid"})

def test_adjust_line_for_replacements_boundary_coincidence() -> None:
    """Verifies that _adjust_line_for_replacements does not shift a target line when a unit ends on that target line."""
    from pydoppelgangerhunt.fixer._patch_delegation import _adjust_line_for_replacements

    orig_text = (
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "line 5\n"
        "line 6\n"
    )
    # Unit 1: lines 1-3 replaced with 1 line (delta = -2)
    u1 = {"file": "mod.py", "start": 1, "end": 3}
    # Unit 2: lines 4-5 replaced with 3 lines (delta = +1)
    u2 = {"file": "mod.py", "start": 4, "end": 5}
    reps = [(u1, "single_1_to_3\n"), (u2, "exp_4_1\nexp_4_2\nexp_4_3\n")]

    # Target line 3: coincides with end of u1. Since end_line < target_line is 3 < 3 (False),
    # u1's delta is NOT added to target line 3. Result remains 3.
    assert _adjust_line_for_replacements(3, reps, orig_text) == 3

    # Target line 4: strictly downstream of u1 (3 < 4), but coincides with start of u2.
    # u1 applies (delta -2), u2 does not apply (5 < 4 is False). Result is 4 + (-2) = 2.
    assert _adjust_line_for_replacements(4, reps, orig_text) == 2

    # Target line 5: coincides with end of u2. u1 applies (3 < 5, delta -2),
    # but u2 ending on line 5 does NOT apply (5 < 5 is False). Result is 5 + (-2) = 3.
    assert _adjust_line_for_replacements(5, reps, orig_text) == 3

    # Target line 6: downstream of both u1 and u2 (3 < 6 and 5 < 6).
    # Both deltas apply: -2 + 1 = -1. Result is 6 + (-1) = 5.
    assert _adjust_line_for_replacements(6, reps, orig_text) == 5

def test_check_units_overlap_multiline_inverted_column_bounds() -> None:
    """Verifies that check_units_overlap handles boundary column disjointness and interior overlaps."""
    from pydoppelgangerhunt.fixer import check_units_overlap

    # Multi-line unit with start_col > end_col across lines 1 to 3
    u_multi = {"file": "mod.py", "start": 1, "end": 3, "start_col": 30, "end_col": 10}

    # Single-line unit on line 1 before start_col (disjoint)
    u_single_disjoint = {"file": "mod.py", "start": 1, "end": 1, "start_col": 15, "end_col": 25}
    assert check_units_overlap(u_multi, u_single_disjoint) is False
    assert check_units_overlap(u_single_disjoint, u_multi) is False

    # Single-line unit on line 1 intersecting start_col (overlapping: [25, 35) and [30, EOL))
    u_single_overlap_start = {"file": "mod.py", "start": 1, "end": 1, "start_col": 25, "end_col": 35}
    assert check_units_overlap(u_multi, u_single_overlap_start) is True
    assert check_units_overlap(u_single_overlap_start, u_multi) is True

    # Single-line unit on line 2 (interior line: unconditionally overlapping)
    u_single_interior = {"file": "mod.py", "start": 2, "end": 2, "start_col": 5, "end_col": 15}
    assert check_units_overlap(u_multi, u_single_interior) is True
    assert check_units_overlap(u_single_interior, u_multi) is True

    # Single-line unit on line 3 intersecting end_col (overlapping: [5, 15) and [0, 10))
    u_single_overlap_end = {"file": "mod.py", "start": 3, "end": 3, "start_col": 5, "end_col": 15}
    assert check_units_overlap(u_multi, u_single_overlap_end) is True
    assert check_units_overlap(u_single_overlap_end, u_multi) is True

    # Single-line unit on line 3 after end_col (disjoint: [15, 25) and [0, 10))
    u_single_disjoint_end = {"file": "mod.py", "start": 3, "end": 3, "start_col": 15, "end_col": 25}
    assert check_units_overlap(u_multi, u_single_disjoint_end) is False
    assert check_units_overlap(u_single_disjoint_end, u_multi) is False

    # Two multi-line units sharing the same lines 1-3
    u_multi_same = {"file": "mod.py", "start": 1, "end": 3, "start_col": 30, "end_col": 10}
    assert check_units_overlap(u_multi, u_multi_same) is True

    # Single-line unit with inverted column bounds on same line represents an empty range (disjoint)
    u_empty = {"file": "mod.py", "start": 1, "end": 1, "start_col": 20, "end_col": 10}
    assert check_units_overlap(u_multi, u_empty) is False
    assert check_units_overlap(u_empty, u_multi) is False

    # Single-line unit with zero-width column bounds (start_col == end_col) represents an empty range (disjoint)
    u_zero_start = {"file": "mod.py", "start": 1, "end": 1, "start_col": 35, "end_col": 35}
    assert check_units_overlap(u_multi, u_zero_start) is False
    assert check_units_overlap(u_zero_start, u_multi) is False

    u_zero_end = {"file": "mod.py", "start": 3, "end": 3, "start_col": 5, "end_col": 5}
    assert check_units_overlap(u_multi, u_zero_end) is False
    assert check_units_overlap(u_zero_end, u_multi) is False

    # Sequential touch with zero-width single unit represents an empty range (disjoint)
    u_prev = {"file": "mod.py", "start": 1, "end": 2, "start_col": 0, "end_col": 20}
    u_zero_touch = {"file": "mod.py", "start": 2, "end": 2, "start_col": 10, "end_col": 10}
    assert check_units_overlap(u_prev, u_zero_touch) is False
    assert check_units_overlap(u_zero_touch, u_prev) is False

    # Coordinate clamping parity: start: 0 and end: 0 clamped to line >= 1
    u_zero_line1 = {"file": "mod.py", "start": 0, "end": 0, "start_col": 0, "end_col": 10}
    u_line1 = {"file": "mod.py", "start": 1, "end": 1, "start_col": 5, "end_col": 15}
    assert check_units_overlap(u_zero_line1, u_line1) is True

def test_unit_desc_strict_validation_and_parsing() -> None:
    """Verifies that _unit_desc in refactor_module_units strictly raises TypeError and ValueError."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    # Non-dictionary unit raises TypeError
    with pytest.raises(TypeError, match="Unit must be a dictionary"):
        refactor_module_units("x = 1\n", [("invalid", "y = 2\n")])  # type: ignore[list-item]

    # Non-integer start line raises ValueError
    with pytest.raises(ValueError, match="Malformed unit: invalid 'start' line"):
        refactor_module_units("x = 1\n", [({"file": "m.py", "start": "bad", "end": 1}, "y = 2\n")])

    # Overlapping collision error formatting with valid units
    u1 = {"file": "m.py", "start": 1, "end": 2, "name": "fn1"}
    u2 = {"file": "m.py", "start": 1, "end": 2, "name": "fn2"}
    with pytest.raises(ValueError, match="Overlapping unit collision detected between 'fn1' \\(1-2\\) and 'fn2' \\(1-2\\)"):
        refactor_module_units("x = 1\ny = 2\n", [(u1, "x = 10\n"), (u2, "y = 20\n")])

def test_is_valid_unit_coordinates_scenarios() -> None:
    """Verifies that is_valid_unit_coordinates, parse_unit_coord, and compute_line_offsets validate and extract coordinates."""
    from pydoppelgangerhunt.fixer.patch import (
        compute_line_offsets as compute_line_offsets_pub,
        is_valid_unit_coordinates as is_valid_pub,
        parse_unit_coord as parse_unit_coord_pub,
    )
    from pydoppelgangerhunt.fixer._unit_schema import _parse_unit_coord
    from pydoppelgangerhunt.fixer.source import (
        _compute_line_offsets,
        _is_valid_unit_coordinates,
        compute_line_offsets,
        is_valid_unit_coordinates,
        parse_unit_coord,
    )
    import pydoppelgangerhunt as pdgh
    import pydoppelgangerhunt.fixer as pdgh_fixer

    # Test correctness and behavioural equivalence across aliases
    test_cases: List[Tuple[Any, bool]] = [
        (None, False),
        ("string_unit", False),
        (["list_unit"], False),
        ({}, False),
        ({"start": 1, "end": 5}, True),
        ({"start": "1", "end": "5"}, True),
        ({"start": "12:0", "end": "15:0"}, True),
        ({"start": "12:0", "end": "15:0", "start_col": "4:0", "end_col": "10:0"}, True),
        ({"start": "not_an_int"}, False),
        ({"start": 1, "end": [2]}, False),
        ({"start": 1, "end": 5, "start_col": "invalid"}, False),
        ({"start": 1, "end": 5, "end_col": {}}, False),
        ({"start": ""}, False),
        ({"start": 1, "end": 5, "start_col": ""}, False),
        ({"start": float("inf"), "end": 5}, False),
        ({"start": "inf", "end": 5}, False),
        ({"start": 1, "end": float("nan")}, False),
        ({"start": 5, "start_col": 20, "end_col": 10}, False),
    ]
    for tc, expected in test_cases:
        assert is_valid_unit_coordinates(tc) is expected
        assert is_valid_pub(tc) is expected
        assert _is_valid_unit_coordinates(tc) is expected
        assert pdgh_fixer.is_valid_unit_coordinates(tc) is expected
        assert pdgh.is_valid_unit_coordinates(tc) is expected

    coord_cases: List[Tuple[Dict[str, Any], str, int, int]] = [
        ({"start": 10}, "start", 1, 10),
        ({"start": "15"}, "start", 1, 15),
        ({"start": "10:0"}, "start", 1, 10),
        ({"start_col": "8:2"}, "start_col", 0, 8),
        ({"start": ""}, "start", 1, 1),
        ({"start": "   "}, "start", 1, 1),
        ({"start": ":0"}, "start", 1, 1),
        ({}, "start", 42, 42),
        ({"end": 99}, "end", 0, 99),
    ]
    for u_data, key, def_val, expected_val in coord_cases:
        assert parse_unit_coord(u_data, key, default=def_val) == expected_val
        assert parse_unit_coord_pub(u_data, key, default=def_val) == expected_val
        assert _parse_unit_coord(u_data, key, default=def_val) == expected_val
        assert pdgh_fixer.parse_unit_coord(u_data, key, default=def_val) == expected_val
        assert pdgh.parse_unit_coord(u_data, key, default=def_val) == expected_val

    line_cases: List[Tuple[List[str], Tuple[List[int], List[int]]]] = [
        (["hello\n", "world 🚀\n"], ([0, 6, 14], [0, 6, 17])),
        ([], ([0], [0])),
        (["line1\n", "line2\n"], ([0, 6, 12], [0, 6, 12])),
    ]
    for lines_input, (exp_c, exp_b) in line_cases:
        res = compute_line_offsets(lines_input)
        assert res == (exp_c, exp_b)
        assert compute_line_offsets_pub(lines_input) == (exp_c, exp_b)
        assert _compute_line_offsets(lines_input) == (exp_c, exp_b)
        assert pdgh_fixer.compute_line_offsets(lines_input) == (exp_c, exp_b)
        assert pdgh.compute_line_offsets(lines_input) == (exp_c, exp_b)

def test_compute_replacement_line_deltas_raises_on_overlapping_units() -> None:
    """Verifies that _compute_replacement_line_deltas raises ValueError on overlapping units."""
    from pydoppelgangerhunt.fixer.patch import _compute_replacement_line_deltas

    orig_text = "def f1():\n    return 1\n\ndef f2():\n    return 2\n"
    u1 = {"file": "sample.py", "name": "fn1", "start": 1, "end": 3}
    u2 = {"file": "sample.py", "name": "fn2", "start": 2, "end": 5}

    with pytest.raises(
        ValueError,
        match=r"Overlapping unit collision detected in line delta computation between 'fn1' \(1-3\) and 'fn2' \(2-5\) in sample\.py",
    ):
        _compute_replacement_line_deltas([(u1, "# rep 1\n"), (u2, "# rep 2\n")], orig_text)

def test_generate_refactoring_patch_rejects_malformed_unit_schema_at_entry(
    tmp_path: Path,
) -> None:
    """Verifies generate_refactoring_patch fails fast on invalid unit schemas before processing valid pairs."""
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text("def run_a(val: int) -> int:\n    return val * 2\n", encoding="utf-8")
    f2.write_text("def run_b(val: int) -> int:\n    return val * 2\n", encoding="utf-8")

    # Pair 1: Malformed coordinates rejected by entry-point schema validation
    bad_u1 = {"file": str(f1), "name": "run_a", "start": "not_an_int", "end": 2}
    bad_u2 = {"file": str(f2), "name": "run_b", "start": 1, "end": 2}

    # Pair 2: Valid clone pair in the same batch
    good_u1 = {"file": str(f1), "name": "run_a", "start": 1, "end": 2, "kind": "function"}
    good_u2 = {"file": str(f2), "name": "run_b", "start": 1, "end": 2, "kind": "function"}

    with pytest.raises(ValueError, match="start"):
        generate_refactoring_patch(
            [(0.9, bad_u1, bad_u2), (1.0, good_u1, good_u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )

def test_generate_refactoring_patch_skips_syntax_error_pair_without_aborting_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an unexpected SyntaxError during clone pair processing does not abort valid refactorings."""
    import pydoppelgangerhunt.fixer.patch as patch_mod

    f1 = tmp_path / "mod_valid1.py"
    f2 = tmp_path / "mod_valid2.py"
    f1.write_text("def process_a(val: int) -> int:\n    return val + 10\n", encoding="utf-8")
    f2.write_text("def process_b(val: int) -> int:\n    return val + 10\n", encoding="utf-8")

    good_u1 = {"file": str(f1), "name": "process_a", "start": 1, "end": 2, "kind": "function"}
    good_u2 = {"file": str(f2), "name": "process_b", "start": 1, "end": 2, "kind": "function"}

    # Pair 1: clone unit that triggers SyntaxError during scope analysis
    bad_u1 = {"file": str(f1), "name": "syntax_err", "start": 1, "end": 2, "kind": "function"}
    bad_u2 = {"file": str(f2), "name": "syntax_err", "start": 1, "end": 2, "kind": "function"}

    original_scope = patch_mod.analyze_unit_variable_scope
    called = [False]

    def mock_scope(*args: Any, **kwargs: Any) -> Any:
        if not called[0]:
            called[0] = True
            raise SyntaxError("Unexpected syntax error in scope analysis")
        return original_scope(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "analyze_unit_variable_scope", mock_scope)

    patch = generate_refactoring_patch(
        [(0.9, bad_u1, bad_u2), (1.0, good_u1, good_u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    # Valid pair should have successfully generated patch despite SyntaxError on first pair
    assert "--- a/mod_valid1.py" in patch
    assert "--- a/mod_valid2.py" in patch
    assert "_shared_process_a_process_b" in patch

def test_compute_unit_spans_unpadded_line_length_and_no_double_newline() -> None:
    """Verifies that compute_unit_spans does not leave stray double newlines on statement replacements."""
    from pydoppelgangerhunt.fixer import compute_unit_spans, refactor_module_units

    # Whole-line indented statement
    src = "def foo():\n    x = 1\n    return x\n"
    unit = {"file": "test.py", "start": 2, "end": 2, "start_col": 4, "end_col": 9}
    span = compute_unit_spans(src, unit)
    # Because prefix is only whitespace and suffix is empty (no trailing code), it is whole-line
    assert span.is_column_bounded is False

    # Replacement should replace whole line including trailing newline, with no double blank lines
    result = refactor_module_units(src, [(unit, "    x = 2\n")])
    assert result == "def foo():\n    x = 2\n    return x\n"
    assert "\n\n" not in result

    # Multi-byte UTF-8 line where byte offset differs from character length
    src_emoji = "def foo():\n    msg = '🚀 rocket'\n    return msg\n"
    # AST end_col_offset in UTF-8 bytes for "    msg = '🚀 rocket'" is 24 bytes
    unit_emoji = {"file": "test.py", "start": 2, "end": 2, "start_col": 4, "end_col": 24}
    span_emoji = compute_unit_spans(src_emoji, unit_emoji)
    assert span_emoji.is_column_bounded is False
    res_emoji = refactor_module_units(src_emoji, [(unit_emoji, "    msg = '🌟 star'\n")])
    assert res_emoji == "def foo():\n    msg = '🌟 star'\n    return msg\n"
    assert "\n\n" not in res_emoji

    # Mid-line expression with trailing code on the same line MUST be column-bounded
    src_mid = "x = 1; y = 2\n"
    unit_mid = {"file": "test.py", "start": 1, "end": 1, "start_col": 0, "end_col": 5}
    span_mid = compute_unit_spans(src_mid, unit_mid)
    assert span_mid.is_column_bounded is True

def test_render_file_patch_plan_omits_unit_with_missing_or_invalid_end() -> None:
    """Verifies that _render_file_patch_plan fails closed and omits units whose end
    coordinate is omitted or inverted."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\nc = 3\n",
        is_new_file=False,
    )
    u_no_end = {"file": "test.py", "start": 1}
    u_inv = {"file": "test.py", "start": 3, "end": 1}
    plan.replacements.append((u_no_end, "# replaced line 1\n"))
    plan.replacements.append((u_inv, "# replaced line 3\n"))

    diff = _render_file_patch_plan(plan, replace_clones=True)
    assert "# replaced line 1" not in diff
    assert "# replaced line 3" not in diff

def test_render_file_patch_plan_mixed_valid_and_colon_invalid_units() -> None:
    """Verifies that _render_file_patch_plan safely filters valid units and ignores
    colon-formatted invalid units without raising ValueError during retained candidate
    cross-checking."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\nc = 3\n",
        is_new_file=False,
    )
    u_valid = {"file": "test.py", "start": 1, "end": 1}
    u_colon = {"file": "test.py", "start": "2:0", "end": "2:5"}
    plan.replacements.append((u_valid, "a = 99\n"))
    plan.replacements.append((u_colon, "b = 100\n"))

    diff = _render_file_patch_plan(plan, replace_clones=True)
    assert "+a = 99" in diff
    assert "b = 100" not in diff

def test_generate_refactoring_patch_no_state_leakage_on_skipped_pair(tmp_path: Path) -> None:
    """Verifies that helper names and comments are not leaked when clone extraction is skipped."""
    f1 = tmp_path / "m1.py"
    f2 = tmp_path / "m2.py"
    # m1 imports m2 and m2 imports m1 (circular dependency)
    f1.write_text("import m2\ndef fn():\n    return 42\n", encoding="utf-8")
    f2.write_text("import m1\ndef fn():\n    return 42\n", encoding="utf-8")

    u1 = {"file": str(f1), "name": "fn", "start": 2, "end": 3, "kind": "function"}
    u2 = {"file": str(f2), "name": "fn", "start": 2, "end": 3, "kind": "function"}

    # Circular import rejection should leave comments and plans clean without orphan helper registrations
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    # Patch should contain circular dependency rejection notice
    assert "Circular import or unresolvable module path" in patch

def test_generate_refactoring_patch_rolls_back_partial_mutations_on_deep_failure(
    tmp_path: Path,
) -> None:
    """Verifies that when a deep failure occurs during candidate pair processing,

    transactional rollback restores snapshotted plans and depgraph, leaving no leaked state.
    """
    from unittest.mock import patch as mock_patch
    from pydoppelgangerhunt.fixer.patch import _build_unit_delegation_call

    module_a = tmp_path / "rollback_mod_a.py"
    module_b = tmp_path / "rollback_mod_b.py"
    module_a.write_text(
        "def compute_cube(val: int) -> int:\n    return val ** 3\n\n"
        "def compute_square(num: int) -> int:\n    return num ** 2\n",
        encoding="utf-8",
    )
    module_b.write_text(
        "def compute_cube_replica(val: int) -> int:\n    return val ** 3\n\n"
        "def compute_square_replica(num: int) -> int:\n    return num ** 2\n",
        encoding="utf-8",
    )

    pair_failing_u1 = {"file": str(module_a), "name": "compute_cube", "start": 1, "end": 2, "kind": "function"}
    pair_failing_u2 = {"file": str(module_b), "name": "compute_cube_replica", "start": 1, "end": 2, "kind": "function"}

    pair_success_u1 = {"file": str(module_a), "name": "compute_square", "start": 4, "end": 5, "kind": "function"}
    pair_success_u2 = {"file": str(module_b), "name": "compute_square_replica", "start": 4, "end": 5, "kind": "function"}

    invocation_count = 0
    original_delegation_builder = _build_unit_delegation_call

    def mocked_delegation_builder(*args: Any, **kwargs: Any) -> str:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            raise ValueError("Simulated deep mid-pipeline AST delegation failure")
        return original_delegation_builder(*args, **kwargs)

    with mock_patch(
        "pydoppelgangerhunt.fixer._patch_delegation._build_unit_delegation_call",
        side_effect=mocked_delegation_builder,
    ):
        patch_result = generate_refactoring_patch(
            [(0.92, pair_failing_u1, pair_failing_u2), (1.0, pair_success_u1, pair_success_u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )

    # First pair must be completely rolled back: no helper or comments referencing compute_cube
    assert "Clone Pair (92.0%)" not in patch_result
    assert "_shared_compute_cube" not in patch_result
    assert "return _shared_compute_cube" not in patch_result

    # Second pair must succeed cleanly
    assert "Clone Pair (100.0%)" in patch_result
    assert "_shared_compute_square" in patch_result
    assert "--- a/rollback_mod_a.py" in patch_result
    assert "--- a/rollback_mod_b.py" in patch_result

def test_compute_unit_spans_whitespace_parity_with_form_feed() -> None:
    """Verifies that compute_unit_spans correctly classifies whitespace with form-feeds (\\f)."""
    from pydoppelgangerhunt.fixer import compute_unit_spans

    # Form feed in indentation before statement: \f    x = 10\n
    source_with_ff = "def fn():\n\f    x = 10\n    return x\n"
    unit_indented_stmt = {"file": "ff_test.py", "start": 2, "end": 2, "start_col": 5, "end_col": 11}
    span_ff = compute_unit_spans(source_with_ff, unit_indented_stmt)
    # The prefix \f    is pure whitespace, suffix is empty, so whole-line statement is NOT column bounded
    assert not span_ff.is_column_bounded

    # Non-whitespace prefix on the same line:
    source_code_prefix = "def fn():\n    flag = x = 10\n    return x\n"
    unit_sub_expr = {"file": "ff_test.py", "start": 2, "end": 2, "start_col": 11, "end_col": 17}
    span_sub = compute_unit_spans(source_code_prefix, unit_sub_expr)
    assert span_sub.is_column_bounded

def test_column_bounded_replacement_no_double_newline_on_clean_line_end() -> None:
    """Verifies that column-bounded replacements on clean line ends do not produce duplicate double newlines."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    code_buffer = (
        "def compute(elements):\n"
        "    # type: ignore\n"
        "    result = [e * 2 for e in elements]\n"
        "    return result\n"
    )
    # Unit covering line 2 to line 3 with clean line end on line 3
    expr_unit = {
        "file": "calc.py",
        "name": "comp_expr",
        "start": 2,
        "end": 3,
        "start_col": 4,
        "end_col": 38,
        "kind": "comprehension",
    }
    # Replacement string ending in newline (e.g. from helper or multi-line expansion)
    rep_text = "transform_elements(elements)\n"
    repaired_code = refactor_module_units(code_buffer, [(expr_unit, rep_text)])

    # Must contain single newline, not double newline (\n\n) before 'return result'
    assert "\n\n" not in repaired_code
    assert repaired_code == (
        "def compute(elements):\n"
        "    transform_elements(elements)  # type: ignore\n"
        "    return result\n"
    )
    assert_valid_python(repaired_code)

def test_check_units_overlap_inverted_line_ranges_are_disjoint() -> None:
    """Verifies that check_units_overlap treats inverted line ranges (start > end) as empty and disjoint."""
    from pydoppelgangerhunt.fixer import check_units_overlap

    valid_unit = {"file": "mod.py", "start": 2, "end": 4}
    inv_u1 = {"file": "mod.py", "start": 5, "end": 2}
    inv_u2 = {"file": "mod.py", "start": 3, "end": 1}

    # Inverted ranges represent empty sets; they cannot collide with valid or inverted units
    assert check_units_overlap(inv_u1, valid_unit) is False
    assert check_units_overlap(valid_unit, inv_u1) is False
    assert check_units_overlap(inv_u1, inv_u2) is False

def test_resolve_unit_replacement_crlf_line_endings_no_dangling_carriage_return() -> None:
    """Verifies that CRLF line endings (\r\n) do not produce orphaned \r carriage returns."""
    from pydoppelgangerhunt.fixer import resolve_unit_replacement

    # 1. Column-bounded unit with boundary pragma on line 1, ending on line 3 with suffix code
    crlf_col_code = "x = calc(  # type: ignore\r\n    data\r\n) + tail\r\n"
    col_unit = {
        "file": "calc.py",
        "name": "calc",
        "start": 1,
        "end": 3,
        "start_col": 4,
        "end_col": 1,
        "kind": "complex_expr",
    }
    rep_item = resolve_unit_replacement(crlf_col_code, col_unit, "fast_calc(data)")
    assert rep_item.consumes_line_suffix is True
    assert "\r " not in rep_item.final_rep
    assert "\r\r" not in rep_item.final_rep
    assert rep_item.final_rep.endswith("\r\n")
    assert "# type: ignore" in rep_item.final_rep
    assert "+ tail" in rep_item.final_rep

    # 2. Column-bounded unit with boundary pragma on line 1, ending on line 2 without suffix code
    col_unit_clean = {
        "file": "calc.py",
        "name": "calc",
        "start": 1,
        "end": 2,
        "start_col": 4,
        "end_col": 8,
        "kind": "complex_expr",
    }
    clean_rep_item = resolve_unit_replacement(crlf_col_code, col_unit_clean, "fast_calc(data)")
    assert clean_rep_item.consumes_line_suffix is True
    assert "\r " not in clean_rep_item.final_rep
    assert "\r\r" not in clean_rep_item.final_rep
    assert clean_rep_item.final_rep.endswith("\r\n")
    assert "# type: ignore" in clean_rep_item.final_rep

    # 3. Whole-line replacement with pragma on start boundary line (line 1)
    crlf_fn_code = "def calc():  # type: ignore\r\n    return 42\r\n"
    fn_unit = {"file": "calc.py", "name": "calc", "start": 1, "end": 2, "kind": "function"}
    whole_rep_item = resolve_unit_replacement(
        crlf_fn_code, fn_unit, "def calc():\r\n    return 100\r\n"
    )
    assert "\r " not in whole_rep_item.final_rep
    assert "\r\r" not in whole_rep_item.final_rep
    assert whole_rep_item.final_rep.endswith("\r\n")
    assert "# type: ignore" in whole_rep_item.final_rep

def test_column_bounded_replacement_clean_line_end_no_duplicate_newline() -> None:
    """Verifies that a column-bounded expression at clean line-end without pragmas does not duplicate newlines."""
    from pydoppelgangerhunt.fixer import refactor_module_units, resolve_unit_replacement

    code = "val = [x for x in data]\nprint(val)\n"
    unit = {
        "file": "test_mod.py",
        "name": "comp",
        "start": 1,
        "end": 1,
        "start_col": 6,
        "end_col": 23,
        "kind": "comprehension",
    }
    item = resolve_unit_replacement(code, unit, "helper(data)\n")
    assert item.consumes_line_suffix is True
    assert item.final_rep == "helper(data)\n"

    result = refactor_module_units(code, [(unit, "helper(data)\n")])
    assert result == "val = helper(data)\nprint(val)\n"

def test_compute_unit_spans_deprecation_warning() -> None:
    """Verifies that _compute_unit_spans emits a runtime DeprecationWarning with stacklevel=2."""
    from pydoppelgangerhunt.fixer.source import _compute_unit_spans

    code = "def sample():\n    return 42\n"
    unit = {"file": "sample.py", "name": "sample", "start": 1, "end": 2}
    with pytest.deprecated_call(match="_compute_unit_spans is deprecated"):
        result = _compute_unit_spans(code, unit)

    assert result == ((0, 28), (0, 28), False)

def test_generate_refactoring_patch_unexpected_exception_keyerror_rollback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that unexpected KeyError during clone pair synthesis triggers rollback and WARNING log."""
    from unittest.mock import patch as mock_patch

    src_file_x = tmp_path / "trans_keyerr_x.py"
    src_file_y = tmp_path / "trans_keyerr_y.py"
    src_file_x.write_text(
        "def transform_alpha(item: str) -> str:\n    return item.strip().lower()\n\n"
        "def transform_beta(data: bytes) -> bytes:\n    return data.lstrip()\n",
        encoding="utf-8",
    )
    src_file_y.write_text(
        "def transform_alpha_replica(item: str) -> str:\n    return item.strip().lower()\n\n"
        "def transform_beta_replica(data: bytes) -> bytes:\n    return data.lstrip()\n",
        encoding="utf-8",
    )

    pair_err_u1 = {"file": str(src_file_x), "name": "transform_alpha", "start": 1, "end": 2, "kind": "function"}
    pair_err_u2 = {"file": str(src_file_y), "name": "transform_alpha_replica", "start": 1, "end": 2, "kind": "function"}

    pair_ok_u1 = {"file": str(src_file_x), "name": "transform_beta", "start": 4, "end": 5, "kind": "function"}
    pair_ok_u2 = {"file": str(src_file_y), "name": "transform_beta_replica", "start": 4, "end": 5, "kind": "function"}

    real_scope_analyzer = analyze_unit_variable_scope

    def _exploding_scope_analyzer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        for arg in args:
            if isinstance(arg, dict) and "alpha" in arg.get("name", ""):
                raise KeyError("simulated_corrupt_symbol_key")
        return real_scope_analyzer(*args, **kwargs)

    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.fixer.patch"):
        with mock_patch(
            "pydoppelgangerhunt.fixer.patch.analyze_unit_variable_scope",
            side_effect=_exploding_scope_analyzer,
        ):
            patch_out = generate_refactoring_patch(
                [(0.95, pair_err_u1, pair_err_u2), (0.98, pair_ok_u1, pair_ok_u2)],
                repo_root=str(tmp_path),
                replace_clones=True,
            )

    # First pair with KeyError was skipped and rolled back
    assert "Clone Pair (95.0%)" not in patch_out
    assert "_shared_transform_alpha" not in patch_out
    # Second pair succeeded
    assert "Clone Pair (98.0%)" in patch_out
    assert "_shared_transform_beta" in patch_out

    # Unexpected error was logged loudly at WARNING with exc_info
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "Unexpected failure processing clone pair" in warning_records[0].message
    assert warning_records[0].exc_info is not None

def test_generate_refactoring_patch_programming_bugs_fail_loudly(
    tmp_path: Path,
) -> None:
    """Verifies that internal developer bugs (AttributeError, NameError, RecursionError) fail loudly."""
    from unittest.mock import patch as mock_patch

    f = tmp_path / "bug_loud.py"
    f.write_text("def work():\n    return 1\n", encoding="utf-8")
    u1 = {"file": str(f), "name": "work", "start": 1, "end": 2, "kind": "function"}
    u2 = {"file": str(f), "name": "work", "start": 1, "end": 2, "kind": "function"}

    # AttributeError
    with mock_patch(
        "pydoppelgangerhunt.fixer.patch.analyze_unit_variable_scope",
        side_effect=AttributeError("internal attribute missing due to developer typo"),
    ):
        with pytest.raises(AttributeError, match="developer typo"):
            generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path))

    # NameError
    with mock_patch(
        "pydoppelgangerhunt.fixer.patch.analyze_unit_variable_scope",
        side_effect=NameError("internal name not defined"),
    ):
        with pytest.raises(NameError, match="internal name not defined"):
            generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path))

    # RecursionError
    with mock_patch(
        "pydoppelgangerhunt.fixer.patch.analyze_unit_variable_scope",
        side_effect=RecursionError("maximum recursion depth exceeded"),
    ):
        with pytest.raises(RecursionError, match="maximum recursion depth exceeded"):
            generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path))

def test_plan_snapshot_typed_dict_and_restore(tmp_path: Path) -> None:
    """Verifies that _FilePatchPlan.snapshot() conforms to _PlanSnapshot and restores state cleanly."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _PlanSnapshot

    f = tmp_path / "plan_snap.py"
    f.write_text("x = 1\n", encoding="utf-8")
    plan = _FilePatchPlan(
        file_path=f,
        rel_path="plan_snap.py",
        orig_text="x = 1\n",
    )
    plan.replacements.append(({"file": str(f), "start": 1, "end": 1}, "x = 2\n"))
    plan.module_helpers.append("def helper(): pass\n")
    plan.method_helpers.append((1, "def method(): pass\n"))
    plan.missing_imports.append("import math")
    plan.comments.append("# comment\n")
    plan.used_helper_names.add("_shared_x")
    plan.claimed_units.append({"file": str(f), "name": "x"})

    snap: _PlanSnapshot = plan.snapshot()
    # Validate TypedDict keys match expected annotations
    assert set(snap.keys()) == set(_PlanSnapshot.__annotations__.keys())

    # Mutate plan state
    plan.replacements.clear()
    plan.module_helpers.clear()
    plan.used_helper_names.clear()

    # Restore from snapshot
    plan.restore(snap)
    assert len(plan.replacements) == 1
    assert len(plan.module_helpers) == 1
    assert "_shared_x" in plan.used_helper_names
    assert len(plan.claimed_units) == 1

def test_compute_unit_spans_start_col_exceeds_line_length_clamping() -> None:
    """Verifies that start_col > len(line_bytes) safely clamps to the line length boundary."""
    from pydoppelgangerhunt.fixer import compute_unit_spans

    # ASCII line (length 10: "x = 42\n" -> 7 bytes)
    code_ascii = "x = 42\n"
    unit_ascii = {"file": "a.py", "start": 1, "end": 1, "start_col": 999, "end_col": 1200}
    span_ascii = compute_unit_spans(code_ascii, unit_ascii)
    # Line length is 7 bytes (7 characters)
    assert span_ascii.start_byte == 7
    assert span_ascii.end_byte == 7
    assert span_ascii.start_char == 7
    assert span_ascii.end_char == 7

    # Multibyte Unicode line: "🚀 = 'rocket'\n" (rocket emoji: 4 bytes, 1 char)
    code_unicode = "🚀 = 'rocket'\n"
    # Line is 14 bytes, 11 characters
    unit_unicode = {"file": "u.py", "start": 1, "end": 1, "start_col": 500, "end_col": 600}
    span_unicode = compute_unit_spans(code_unicode, unit_unicode)
    assert span_unicode.start_byte == len(code_unicode.encode("utf-8"))
    assert span_unicode.end_byte == len(code_unicode.encode("utf-8"))
    assert span_unicode.start_char == len(code_unicode)
    assert span_unicode.end_char == len(code_unicode)

def test_apply_line_deltas_edge_cases() -> None:
    """Verifies that _apply_line_deltas handles boundary conditions and non-integer inputs."""
    from pydoppelgangerhunt.fixer.patch import _apply_line_deltas

    # Empty deltas
    assert _apply_line_deltas(10, []) == 10

    # Target line <= 1 returns max(1, t_line)
    assert _apply_line_deltas(0, [(5, 2)]) == 1
    assert _apply_line_deltas(-5, [(5, 2)]) == 1
    assert _apply_line_deltas(1, [(5, 2)]) == 1

    # Malformed target line input
    assert _apply_line_deltas("invalid", [(5, 2)]) == 1  # type: ignore[arg-type]

    # Precomputed deltas: (end_line, delta)
    deltas = [(3, 2), (7, -1), (12, 4)]
    # Target line 3: strictly before end_line 3 (end_line < target_line is False for 3 < 3)
    assert _apply_line_deltas(3, deltas) == 3
    # Target line 5: after end_line 3, before 7 (+2)
    assert _apply_line_deltas(5, deltas) == 7
    # Target line 10: after 3 and 7 (+2 - 1 = +1)
    assert _apply_line_deltas(10, deltas) == 11
    # Target line 15: after 3, 7, and 12 (+2 - 1 + 4 = +5)
    assert _apply_line_deltas(15, deltas) == 20

def test_split_source_lines_physical_newlines_and_form_feed() -> None:
    """Verifies that split_source_lines splits strictly on physical newlines and preserves form-feed characters."""
    import io
    from pydoppelgangerhunt.fixer import split_source_lines

    assert split_source_lines("") == []
    assert split_source_lines("x = 1") == ["x = 1"]
    assert split_source_lines("x = 1\n") == ["x = 1\n"]
    assert split_source_lines("x = 1\r\ny = 2\rz = 3\n") == ["x = 1\r\n", "y = 2\r", "z = 3\n"]
    assert split_source_lines("\n\n") == ["\n", "\n"]

    # Form-feed (\f) and vertical tab (\v) are kept within the line, matching Python AST
    ff_code = "def fn():\n\x0c    x = 10\n    return x\n"
    lines = split_source_lines(ff_code)
    assert len(lines) == 3
    assert lines[0] == "def fn():\n"
    assert lines[1] == "\x0c    x = 10\n"
    assert lines[2] == "    return x\n"

    # Parity with io.StringIO readlines
    for s in ["", "a", "a\n", "a\r\nb\nc", "def fn():\n\x0c    pass\n"]:
        assert split_source_lines(s) == io.StringIO(s, newline="").readlines()

def test_empty_source_buffer_replacement_is_noop() -> None:
    """Verifies that replacements targeting an empty source buffer return empty string without inserting text."""
    from pydoppelgangerhunt.fixer import (
        compute_unit_replacement_span,
        refactor_module_units,
        replace_unit_in_source,
        resolve_unit_replacement,
    )

    unit = {"file": "empty.py", "start": 1, "end": 1}

    # resolve_unit_replacement
    item = resolve_unit_replacement("", unit, "x = 1\n")
    assert item.final_rep == ""
    assert item.start_char == 0
    assert item.end_char == 0

    # compute_unit_replacement_span
    start_c, end_c, final_rep = compute_unit_replacement_span("", unit, "x = 1\n")
    assert (start_c, end_c, final_rep) == (0, 0, "")

    # replace_unit_in_source
    assert replace_unit_in_source("", unit, "x = 1\n") == ""

    # refactor_module_units
    assert refactor_module_units("", [(unit, "x = 1\n")]) == ""

def test_package_reexport_split_source_lines() -> None:
    """Verifies that split_source_lines is re-exported from package root and fixer subpackages."""
    import pydoppelgangerhunt as pdgh
    import pydoppelgangerhunt.fixer as pdgh_fixer

    assert hasattr(pdgh, "split_source_lines")
    assert "split_source_lines" in pdgh.__all__
    assert callable(pdgh.split_source_lines)

    assert hasattr(pdgh_fixer, "split_source_lines")
    assert "split_source_lines" in pdgh_fixer.__all__
    assert callable(pdgh_fixer.split_source_lines)

def test_compute_unit_spans_non_positive_line_clamping() -> None:
    """Verifies that compute_unit_spans clamps non-positive line numbers to line 1 matching check_units_overlap."""
    from pydoppelgangerhunt.fixer import check_units_overlap, compute_unit_spans, refactor_module_units

    src = "x = 1\ny = 2\nz = 3\n"
    unit_zero = {"file": "test.py", "start": 0, "end": 0}

    # compute_unit_spans clamps start and end to line 1
    span = compute_unit_spans(src, unit_zero)
    assert span.start_line == 1
    assert span.end_line == 1
    assert span.start_char == 0
    assert span.end_char == 6  # length of "x = 1\n"

    # check_units_overlap harmoniously treats start:0, end:0 as line 1
    assert check_units_overlap(unit_zero, {"file": "test.py", "start": 1, "end": 1})
    assert not check_units_overlap(unit_zero, {"file": "test.py", "start": 2, "end": 2})

    # refactor_module_units fails closed on non-positive coordinates
    with pytest.raises(ValueError, match="invalid coordinates for refactoring"):
        refactor_module_units(src, [(unit_zero, "x = 99\n")])

    # Negative coordinates clamp to line 1
    unit_neg = {"file": "test.py", "start": -10, "end": -5}
    span_neg = compute_unit_spans(src, unit_neg)
    assert span_neg.start_line == 1
    assert span_neg.end_line == 1

    # Inverted coordinates with non-positive end (start=3, end=0) clamp to end=1 (start > end)
    unit_inverted = {"file": "test.py", "start": 3, "end": 0}
    span_inv = compute_unit_spans(src, unit_inverted)
    assert span_inv.start_line == 3
    assert span_inv.end_line == 1
    assert span_inv.start_char == span_inv.end_char == len(src)

def test_render_file_patch_plan_preserves_crlf_in_helpers() -> None:
    """Verifies that _render_file_patch_plan uses CRLF line endings when splicing helpers into CRLF files."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    crlf_source = "class MyClass:\r\n    def target(self):\r\n        return 1\r\n"
    plan = _FilePatchPlan(
        file_path=Path("crlf_sample.py"),
        rel_path="crlf_sample.py",
        orig_text=crlf_source,
        is_new_file=False,
    )
    plan.method_helpers.append((2, "    def helper(self):\r\n        return 42"))
    plan.module_helpers.append("def global_helper():\r\n    return 99")

    diff = _render_file_patch_plan(plan, replace_clones=False)
    assert "--- a/crlf_sample.py" in diff
    assert "+    def helper(self):\r\n" in diff
    assert "+def global_helper():\r\n" in diff
    # Ensure no bare \n line endings within the inserted helper diff lines
    for line in diff.splitlines(keepends=True):
        if line.startswith("+") and not line.startswith("+++"):
            assert line.endswith("\r\n"), f"Expected CRLF line ending in hunk line: {line!r}"

def test_split_source_lines_finditer_avoid_intermediate_allocation() -> None:
    """Verifies that split_source_lines with finditer produces identical results across all line ending types."""
    import io
    from pydoppelgangerhunt.fixer import split_source_lines

    cases = [
        "",
        "single line",
        "single line\n",
        "single line\r\n",
        "single line\r",
        "line 1\r\nline 2\r\nline 3\r\n",
        "line 1\nline 2\nline 3\n",
        "line 1\rline 2\rline 3\r",
        "mixed\r\nline 2\nline 3\rline 4",
        "\n\n\n",
        "\r\n\r\n",
        "def fn():\n\x0c    pass\n",
    ]

    for case in cases:
        result = split_source_lines(case)
        expected = io.StringIO(case, newline="").readlines()
        assert result == expected, f"Mismatch on {case!r}: {result!r} != {expected!r}"
        assert all(isinstance(ln, str) and ln for ln in result)

def test_count_physical_newlines_and_detect_line_ending() -> None:
    """Verifies physical newline counting and line ending detection across LF, CRLF, and lone CR."""
    from pydoppelgangerhunt.fixer import count_physical_newlines, detect_line_ending

    # count_physical_newlines
    assert count_physical_newlines("") == 0
    assert count_physical_newlines("no newline") == 0
    assert count_physical_newlines("a\nb\n") == 2
    assert count_physical_newlines("a\r\nb\r\n") == 2
    assert count_physical_newlines("a\rb\r") == 2
    assert count_physical_newlines("a\r\nb\nc\rd") == 3

    # detect_line_ending
    assert detect_line_ending() == "\n"
    assert detect_line_ending("") == "\n"
    assert detect_line_ending("x = 1\n") == "\n"
    assert detect_line_ending("x = 1\r\n") == "\r\n"
    assert detect_line_ending("x = 1\r") == "\r"
    assert detect_line_ending("x = 1", "y = 2\r\n") == "\r\n"
    assert detect_line_ending("x = 1", "y = 2\r") == "\r"

def test_compute_replacement_line_deltas_lone_cr_source() -> None:
    """Verifies that _compute_replacement_line_deltas accurately computes line delta for lone-CR sources."""
    from pydoppelgangerhunt.fixer.patch import _compute_replacement_line_deltas

    # 3 lines of CR-only source
    src_cr = "def foo():\r    x = 1\r    return x\r"
    # Replace lines 1-2 with 1 line -> delta must be -1
    u = {"file": "cr_test.py", "start": 1, "end": 2}
    deltas = _compute_replacement_line_deltas([(u, "def foo():\r")], src_cr)
    assert deltas == [(2, -1)]

def test_resolve_unit_replacement_lone_cr_source() -> None:
    """Verifies that resolve_unit_replacement and replace_unit_in_source preserve lone-CR line terminators."""
    from pydoppelgangerhunt.fixer import replace_unit_in_source

    # Column-bounded replacement consuming line suffix with boundary pragma on CR source
    src_cr = "x = calc()  # type: ignore\ry = 1\r"
    unit_col = {"file": "cr_test.py", "start": 1, "end": 1, "start_col": 0, "end_col": 10}
    res_col = replace_unit_in_source(src_cr, unit_col, "x = new_calc()")
    # Must preserve \r before y = 1\r without introducing \n
    assert res_col == "x = new_calc()  # type: ignore\ry = 1\r"
    assert "\n" not in res_col

    # Whole-line replacement on CR source
    src_cr_line = "x = 1\ry = 2\r"
    unit_line = {"file": "cr_test.py", "start": 1, "end": 1}
    res_line = replace_unit_in_source(src_cr_line, unit_line, "x = 10")
    assert res_line == "x = 10\ry = 2\r"
    assert "\n" not in res_line

def test_render_file_patch_plan_lone_cr_helpers() -> None:
    """Verifies that _render_file_patch_plan preserves lone-CR line terminators when splicing helpers."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    cr_source = "class MyClass:\r    def target(self):\r        return 1\r"
    plan = _FilePatchPlan(
        file_path=Path("cr_sample.py"),
        rel_path="cr_sample.py",
        orig_text=cr_source,
        is_new_file=False,
    )
    plan.method_helpers.append((2, "    def helper(self):\r        return 42"))
    plan.module_helpers.append("def global_helper():\r    return 99")

    diff = _render_file_patch_plan(plan, replace_clones=False)
    assert "--- a/cr_sample.py" in diff
    # Ensure all added helper lines terminate with \r (and not \r\n or bare \n)
    for line in diff.splitlines(keepends=True):
        if line.startswith("+") and not line.startswith("+++"):
            assert line.endswith("\r"), f"Expected lone-CR line ending in hunk line: {line!r}"
            assert not line.endswith("\r\n"), f"Did not expect CRLF in CR hunk line: {line!r}"

def test_package_reexport_newline_helpers() -> None:
    """Verifies that count_physical_newlines and detect_line_ending are re-exported from package roots."""
    import pydoppelgangerhunt as pdgh
    import pydoppelgangerhunt.fixer as pdgh_fixer

    for name in ("count_physical_newlines", "detect_line_ending"):
        assert hasattr(pdgh, name)
        assert name in pdgh.__all__
        assert callable(getattr(pdgh, name))
        assert hasattr(pdgh_fixer, name)
        assert name in pdgh_fixer.__all__
        assert callable(getattr(pdgh_fixer, name))

def test_detect_line_ending_sample_boundary_and_straddle() -> None:
    """Verifies that detect_line_ending handles boundary-straddling CRLF and expanded sample buffers."""
    from pydoppelgangerhunt.fixer import detect_line_ending

    # Case 1: CRLF at legacy 1024-byte sample boundary without EOF newline
    text_1024 = "x" * 1023 + "\r\n" + "y = 1"
    assert detect_line_ending(text_1024) == "\r\n"

    # Case 2: CRLF straddles the 4096-byte sample boundary (char 4095 is \r, 4096 is \n)
    text_4096_crlf = "a" * 4095 + "\r\n" + "b = 2"
    assert detect_line_ending(text_4096_crlf) == "\r\n"

    # Case 3: Lone CR at char 4095 followed by non-\n
    text_4096_cr = "a" * 4095 + "\r" + "b = 2"
    assert detect_line_ending(text_4096_cr) == "\r"

def test_detect_line_ending_true_majority_vote_resilience() -> None:
    """Verifies that detect_line_ending performs a true majority vote across all lines and resists stray terminators."""
    from pydoppelgangerhunt.fixer import detect_line_ending
    from pydoppelgangerhunt.fixer.source import _insert_imports_into_module

    # Overwhelmingly LF with stray CRLF early in file
    lf_with_stray_crlf = ["import os\n", "stray = 1\r\n"] + ["x = 2\n"] * 50
    assert detect_line_ending(*lf_with_stray_crlf) == "\n"
    assert detect_line_ending("".join(lf_with_stray_crlf)) == "\n"

    # Verify import insertion into overwhelmingly LF module with stray CRLF uses \n
    inserted_lines = _insert_imports_into_module(lf_with_stray_crlf, ["import sys"])
    assert any("import sys\n" == ln for ln in inserted_lines)
    assert not any("import sys\r\n" == ln for ln in inserted_lines)

    # Overwhelmingly CRLF with stray LF early in file
    crlf_with_stray_lf = ["import os\r\n", "stray = 1\n"] + ["x = 2\r\n"] * 50
    assert detect_line_ending(*crlf_with_stray_lf) == "\r\n"
    assert detect_line_ending("".join(crlf_with_stray_lf)) == "\r\n"

    # Overwhelmingly lone CR with stray LF
    cr_with_stray_lf = ["import os\r", "stray = 1\n"] + ["x = 2\r"] * 50
    assert detect_line_ending(*cr_with_stray_lf) == "\r"
    assert detect_line_ending("".join(cr_with_stray_lf)) == "\r"

    # Tie-breaking priority among non-zero counts (LF -> CRLF -> CR)
    assert detect_line_ending("a\r\nb\n") == "\n"
    assert detect_line_ending("a\rb\n") == "\n"
    assert detect_line_ending("a\r\nb\r") == "\r\n"
    assert detect_line_ending("a\r\nb\nc\r") == "\n"

def test_resolve_unit_replacement_form_feed_in_final_rep() -> None:
    """Verifies that resolve_unit_replacement uses physical line splitting for final_rep containing form feeds."""
    from pydoppelgangerhunt.fixer import resolve_unit_replacement

    source = "x = 1  # type: ignore\ny = 2\n"
    unit = {"file": "test.py", "start": 1, "end": 1}
    rep_with_ff = "\x0cx = 2\x0c"
    item = resolve_unit_replacement(source, unit, rep_with_ff, preserve_boundary_pragmas=True)
    assert item.final_rep == "\x0cx = 2\x0c  # type: ignore\n"

def test_render_file_patch_plan_form_feed_in_helpers() -> None:
    """Verifies that _render_file_patch_plan preserves form feeds in method and module helpers."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="class A:\n    pass\n",
        is_new_file=False,
    )
    plan.method_helpers.append((2, "    def helper(self):\x0c        return 1\n"))
    plan.module_helpers.append("def top_helper():\x0c    return 2\n")

    diff = _render_file_patch_plan(plan, replace_clones=True)
    assert "+    def helper(self):\x0c        return 1" in diff
    assert "+def top_helper():\x0c    return 2" in diff

    new_plan = _FilePatchPlan(
        file_path=Path("new.py"),
        rel_path="new.py",
        orig_text="",
        is_new_file=True,
    )
    new_plan.module_helpers.append("def new_helper():\x0c    return 3\n")
    diff_new = _render_file_patch_plan(new_plan, replace_clones=True)
    assert "+def new_helper():\x0c    return 3" in diff_new

def test_detect_line_ending_iterable_and_mixed_arguments() -> None:
    """Verifies that detect_line_ending accepts iterables, generators, sequences, and None values."""
    from pydoppelgangerhunt.fixer import detect_line_ending

    # Passing list of lines directly
    lines = ["first\r\n", "second\r\n", "third\n"]
    assert detect_line_ending(lines) == "\r\n"

    # Generator expression
    assert detect_line_ending(l for l in lines) == "\r\n"

    # Mixed arguments
    assert detect_line_ending("prefix\n", ["nested\n", "other\n"], None, "stray\r\n") == "\n"

    # Empty / none
    assert detect_line_ending() == "\n"
    assert detect_line_ending([]) == "\n"
    assert detect_line_ending(None) == "\n"

def test_intervals_overlap_public_helper() -> None:
    """Verifies intervals_overlap on disjoint, overlapping, touching, and zero-width intervals."""
    from pydoppelgangerhunt.fixer import intervals_overlap

    # Disjoint intervals
    assert intervals_overlap(0, 5, 5, 10) is False
    assert intervals_overlap(0, 5, 6, 10) is False

    # Overlapping intervals
    assert intervals_overlap(0, 5, 4, 10) is True
    assert intervals_overlap(2, 8, 1, 4) is True

    # Zero-width interval inside range
    assert intervals_overlap(0, 5, 3, 3) is True
    assert intervals_overlap(3, 3, 0, 5) is True

    # Zero-width interval at boundary (touching does not overlap)
    assert intervals_overlap(0, 5, 0, 0) is False
    assert intervals_overlap(0, 5, 5, 5) is False

    # Two zero-width intervals
    assert intervals_overlap(5, 5, 5, 5) is True
    assert intervals_overlap(5, 5, 6, 6) is False

def test_validate_module_unit_replacements_and_tier_toggles() -> None:
    """Verifies validate_module_unit_replacements and tier1/tier2 toggles in refactor_module_units."""
    from pydoppelgangerhunt.fixer import (
        UnitCollisionError,
        refactor_module_units,
        validate_module_unit_replacements,
    )

    code = "x = 1\ny = 2\nz = 3\n"
    u1 = {"file": "test.py", "start": 1, "end": 1, "start_col": 0, "end_col": 5}
    u2 = {"file": "test.py", "start": 2, "end": 2, "start_col": 0, "end_col": 5}

    # Non-colliding replacements pass validation
    validate_module_unit_replacements(code, [(u1, "x = 10"), (u2, "y = 20")])

    # Colliding replacements raise UnitCollisionError
    u1_collide = {"file": "test.py", "start": 1, "end": 1, "start_col": 2, "end_col": 5}
    with pytest.raises(UnitCollisionError):
        validate_module_unit_replacements(code, [(u1, "x = 10"), (u1_collide, "100")])

    # Tier 1 vs Tier 2 toggling
    # With tier1=False and tier2=False, collisions are not checked
    res = refactor_module_units(
        code,
        [(u1, "x = 10"), (u2, "y = 20")],
        tier1=False,
        tier2=False,
    )
    assert "x = 10" in res

def test_extract_unit_body_lines_and_synthesis_form_feed(tmp_path: Path) -> None:
    """Verifies that _extract_unit_body_lines and synthesize_shared_helper_code preserve form feeds."""
    from pydoppelgangerhunt.fixer.source import _extract_unit_body_lines
    from pydoppelgangerhunt.fixer.synthesis import synthesize_shared_helper_code

    raw_fn = [
        "def helper(x):\n",
        "\x0c    res = x + 1\n",
        "    return res\n",
    ]
    unit = {"file": "m.py", "start": 1, "end": 3, "kind": "function"}
    body_lines = _extract_unit_body_lines(unit, raw_fn)
    assert any("\x0c" in ln for ln in body_lines)

    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    code = "def calc(x):\n\x0c    res = x + 10\n    return res\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 3, "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 3, "kind": "function"}
    h_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "\x0c" in h_code

def test_negative_column_clamping_in_check_units_overlap() -> None:
    """Verifies that negative column offsets clamp to 0 and do not produce erroneous collisions."""
    # Negative start column clamps to 0: [0, 5) and [5, 10) touch but do not overlap
    u_neg1 = {"file": "mod.py", "start": 1, "end": 1, "start_col": -10, "end_col": 5}
    u_pos = {"file": "mod.py", "start": 1, "end": 1, "start_col": 5, "end_col": 10}
    assert check_units_overlap(u_neg1, u_pos) is False

    # Negative start column clamps to 0: [0, 6) and [5, 10) overlap at [5, 6)
    u_neg2 = {"file": "mod.py", "start": 1, "end": 1, "start_col": -10, "end_col": 6}
    assert check_units_overlap(u_neg2, u_pos) is True

    # Both negative clamp to 0: [0, 0) is empty zero-width, so no collision
    u_neg_zero = {"file": "mod.py", "start": 1, "end": 1, "start_col": -10, "end_col": -2}
    assert check_units_overlap(u_neg_zero, u_pos) is False

def test_reporters_precompiled_physical_line_regex_notebook_cells(tmp_path: Path) -> None:
    """Verifies that extract_unit_source_code uses _PHYSICAL_LINE_RE and preserves form feeds."""
    from pydoppelgangerhunt.source_lines import _PHYSICAL_LINE_RE

    assert _PHYSICAL_LINE_RE.pattern == r"[^\r\n]*(?:\r\n|\r|\n|$)"

    nb_data = {
        "cells": [
            {"cell_type": "code", "source": "val = 1\n\x0cres = val + 2\n"}
        ]
    }
    nb_file = tmp_path / "notebook.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    unit = {"file": str(nb_file) + "#cell_1", "start": 1, "end": 2}
    lines = extract_unit_source_code(unit, repo_root=str(tmp_path))
    assert lines == ["val = 1\n", "\x0cres = val + 2\n"]
    assert "\x0c" in lines[1]

@requires_git
def test_generate_refactoring_patch_sync_generator_with_outputs_and_return(tmp_path: Path) -> None:
    """Verifies that refactoring sync generators with outputs generates return in helper and captures via yield from."""
    src1 = (
        "def produce_and_sum1(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    src2 = (
        "def produce_and_sum2(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    f1 = tmp_path / "ps1.py"
    f2 = tmp_path / "ps2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "produce_and_sum1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 5, "name": "produce_and_sum2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "return total" in patch
    assert "return total, x" not in patch
    assert "total = (yield from _shared_produce_and_sum1_produce_and_sum2(items))" in patch
    assert "-> Generator[" in patch

    # Semantic contract validation: execute original code, apply git patch, execute refactored code, and assert after == before
    eval_code = (
        "import ps1, ps2\n"
        "for mod_name, fn in [('ps1', ps1.produce_and_sum1), ('ps2', ps2.produce_and_sum2)]:\n"
        "    for test_items in [[], [42], [1, 2, 3, 4]]:\n"
        "        gen = fn(test_items)\n"
        "        yielded = []\n"
        "        while True:\n"
        "            try:\n"
        "                yielded.append(next(gen))\n"
        "            except StopIteration as e:\n"
        "                ret = e.value\n"
        "                break\n"
        "        print(f'{mod_name}:{test_items}:{yielded}:{ret}')\n"
    )
    before_proc = subprocess.run(
        [sys.executable, "-c", eval_code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert before_proc.returncode == 0, f"Original execution failed: {before_proc.stderr}"

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    after_proc = subprocess.run(
        [sys.executable, "-c", eval_code],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after_proc.returncode == 0, f"Refactored execution failed: {after_proc.stderr}"
    assert after_proc.stdout == before_proc.stdout

@requires_git
def test_generate_refactoring_patch_sync_generator_renamed_downstream_outputs(tmp_path: Path) -> None:
    """Verifies that generator clones with distinct renamed outputs map outputs 1-to-1 without returning unassigned variables."""
    src1 = (
        "def f1(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    src2 = (
        "def f2(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += x\n"
        "        yield x\n"
        "    return count\n"
    )
    f1 = tmp_path / "ren1.py"
    f2 = tmp_path / "ren2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    eval_script = (
        "import ren1, ren2\n"
        "for mod_name, fn in [('ren1', ren1.f1), ('ren2', ren2.f2)]:\n"
        "    for items in [[], [42], [1, 2, 3, 4]]:\n"
        "        gen = fn(items)\n"
        "        yielded = []\n"
        "        while True:\n"
        "            try:\n"
        "                yielded.append(next(gen))\n"
        "            except StopIteration as e:\n"
        "                ret = e.value\n"
        "                break\n"
        "        print(f'{mod_name}:{items}:{yielded}:{ret}')\n"
    )
    before_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert before_proc.returncode == 0, f"Original execution failed: {before_proc.stderr}"

    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 5, "name": "f2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch([(0.90, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "return total" in patch
    assert "return total, count" not in patch
    assert "return total, x" not in patch
    assert "count = (yield from _shared_f1_f2(items))" in patch
    assert "total = (yield from _shared_f1_f2(items))" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    after_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after_proc.returncode == 0, f"Refactored execution failed: {after_proc.stderr}"
    assert after_proc.stdout == before_proc.stdout

def test_generate_refactoring_patch_sync_generator_unequal_output_counts_rejected(tmp_path: Path) -> None:
    """Verifies that generator clones with unequal output counts and un-counterparted needed outputs are rejected."""
    src1 = (
        "def f1(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    src2 = (
        "def f2(items: list[int]):\n"
        "    count = 0\n"
        "    status = 0\n"
        "    for x in items:\n"
        "        count += x\n"
        "        status += 1\n"
        "        yield x\n"
        "    return count\n"
    )
    f1 = tmp_path / "u_out1.py"
    f2 = tmp_path / "u_out2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 7, "name": "f2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch([(0.90, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""

@requires_git
def test_generate_refactoring_patch_sync_generator_classdef_downstream_execution(tmp_path: Path) -> None:
    """Verifies that class bodies executing immediately after a generator unit retain the observed variable."""
    src1 = (
        "def f1(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    class C:\n"
        "        value = x\n"
        "    return C\n"
    )
    src2 = (
        "def f2(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    class C:\n"
        "        value = x\n"
        "    return C\n"
    )
    f1 = tmp_path / "cg1.py"
    f2 = tmp_path / "cg2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    eval_script = (
        "import cg1, cg2\n"
        "for mod_name, fn in [('cg1', cg1.f1), ('cg2', cg2.f2)]:\n"
        "    for test_items in [[1, 2, 3], []]:\n"
        "        gen = fn(test_items)\n"
        "        while True:\n"
        "            try:\n"
        "                next(gen)\n"
        "            except StopIteration as e:\n"
        "                cls_res = e.value\n"
        "                break\n"
        "        print(f'{mod_name}:{cls_res.value}')\n"
    )
    before_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert before_proc.returncode == 0, f"Original execution failed: {before_proc.stderr}"
    assert before_proc.stdout.strip() == "cg1:3\ncg1:0\ncg2:3\ncg2:0"

    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 4, "name": "f2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "return x" in patch
    assert "x = (yield from _shared_f1_f2(items))" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    after_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after_proc.returncode == 0, f"Refactored execution failed: {after_proc.stderr}"
    assert after_proc.stdout == before_proc.stdout

@requires_git
def test_generate_refactoring_patch_sync_generator_closure_capture(tmp_path: Path) -> None:
    """Verifies that closure captures downstream retain captured variables in generator return."""
    src1 = (
        "def f1(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    def get_x():\n"
        "        return x\n"
        "    return get_x\n"
    )
    src2 = (
        "def f2(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    def get_x():\n"
        "        return x\n"
        "    return get_x\n"
    )
    f1 = tmp_path / "clo1.py"
    f2 = tmp_path / "clo2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    eval_script = (
        "import clo1, clo2\n"
        "for mod_name, fn in [('clo1', clo1.f1), ('clo2', clo2.f2)]:\n"
        "    for test_items in [[10, 20, 30], []]:\n"
        "        gen = fn(test_items)\n"
        "        while True:\n"
        "            try:\n"
        "                next(gen)\n"
        "            except StopIteration as e:\n"
        "                getter = e.value\n"
        "                break\n"
        "        print(f'{mod_name}:{getter()}')\n"
    )
    before_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert before_proc.returncode == 0, f"Original execution failed: {before_proc.stderr}"
    assert before_proc.stdout.strip() == "clo1:30\nclo1:0\nclo2:30\nclo2:0"

    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 4, "name": "f2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "return x" in patch
    assert "x = (yield from _shared_f1_f2(items))" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    after_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after_proc.returncode == 0, f"Refactored execution failed: {after_proc.stderr}"
    assert after_proc.stdout == before_proc.stdout

def test_generate_refactoring_patch_symmetric_global_filtering(tmp_path: Path) -> None:
    """Verifies generate_refactoring_patch filters globals/nonlocals from u1/u2 outputs."""
    code = (
        "global_var = 0\n"
        "def fn1():\n"
        "    global global_var\n"
        "    global_var = 100\n"
        "    common_calc = 10\n"
        "    local_var = common_calc + 1\n"
        "def fn2():\n"
        "    global global_var\n"
        "    global_var = 200\n"
        "    common_calc = 10\n"
        "    local_var = common_calc + 1\n"
    )
    f = tmp_path / "mod.py"
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 6,
        "name": "fn1",
        "kind": "function",
        "outputs": ["local_var", "global_var"],
    }
    u2 = {
        "file": str(f),
        "start": 7,
        "end": 11,
        "name": "fn2",
        "kind": "function",
        "outputs": ["local_var", "global_var"],
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "return local_var" in patch
    assert "return local_var, global_var" not in patch

def test_generate_refactoring_patch_sync_generator_indefinite_loop_var_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that generator clones where downstream reads depend on a loop variable that
    is not definitely assigned inside the unit (initialization is outside the unit) are rejected."""
    src1 = (
        "def f1(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    class C:\n"
        "        value = x\n"
        "    return C\n"
    )
    src2 = (
        "def f2(items: list[int]):\n"
        "    x = 0\n"
        "    for x in items:\n"
        "        yield x\n"
        "    class C:\n"
        "        value = x\n"
        "    return C\n"
    )
    f1 = tmp_path / "indef1.py"
    f2 = tmp_path / "indef2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    # Unit starts at line 3 (for x in items: ...), so line 2 (x = 0) is outside the unit!
    u1 = {"file": str(f1), "start": 3, "end": 4, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 3, "end": 4, "name": "f2:for", "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

@requires_git
def test_generate_refactoring_patch_sync_generator_with_union_yield_type(
    tmp_path: Path,
) -> None:
    """Verifies that generator clones yielding union types inject Union into typing imports
    and successfully apply via git."""
    src1 = (
        "def produce1(items: tuple[int, str]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        if isinstance(x, int):\n"
        "            total += x\n"
        "    yield from items\n"
        "    return total\n"
    )
    src2 = (
        "def produce2(items: tuple[int, str]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        if isinstance(x, int):\n"
        "            total += x\n"
        "    yield from items\n"
        "    return total\n"
    )
    f1 = tmp_path / "u1.py"
    f2 = tmp_path / "u2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 7, "name": "produce1", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 7, "name": "produce2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "Union[int, str]" in patch
    assert "from typing import" in patch
    assert "Union" in patch
    assert "Generator" in patch

    eval_script = (
        "import u1, u2\n"
        "for mod_name, fn in [('u1', u1.produce1), ('u2', u2.produce2)]:\n"
        "    gen = fn((10, 'hello'))\n"
        "    yielded = []\n"
        "    while True:\n"
        "        try:\n"
        "            yielded.append(next(gen))\n"
        "        except StopIteration as e:\n"
        "            ret = e.value\n"
        "            break\n"
        "    print(f'{mod_name}:{yielded}:{ret}')\n"
    )
    before_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert before_proc.returncode == 0, f"Original execution failed: {before_proc.stderr}"

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True
    )
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "init"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    apply_proc = subprocess.run(
        ["git", "apply"],
        input=patch,
        text=True,
        cwd=str(tmp_path),
        capture_output=True,
        check=False,
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    after_proc = subprocess.run(
        [sys.executable, "-c", eval_script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert after_proc.returncode == 0, f"Refactored execution failed: {after_proc.stderr}"
    assert after_proc.stdout == before_proc.stdout

def test_generate_refactoring_patch_non_generator_mismatched_kinds_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that non-generator clone pairs where one unit is a block and the other is a
    whole function are unconditionally rejected."""
    src1 = (
        "def f1(items: list[int]) -> int:\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n"
    )
    src2 = (
        "def f2(items: list[int]) -> int:\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n"
    )
    f1 = tmp_path / "ng1.py"
    f2 = tmp_path / "ng2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 1, "end": 5, "name": "f2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generate_refactoring_patch_closure_strictness_knob(tmp_path: Path) -> None:
    """Verifies closure_strictness knob controls pre-unit closure isolation in patch generation."""
    src1 = (
        "def g1(items: list[int]):\n"
        "    cb = lambda: extra\n"
        "    for x in items:\n"
        "        yield x\n"
        "        extra = x\n"
        "    return 0\n"
    )
    src2 = (
        "def g2(items: list[int]):\n"
        "    for y in items:\n"
        "        yield y\n"
        "        extra = y\n"
        "    return 0\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 3, "end": 5, "name": "g1:block", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 4, "name": "g2:block", "kind": "compound_block"}

    # 1. Default strict mode: fails closed because cb = lambda: extra captures extra
    # before extra is definitely assigned prior to unit start
    patch_strict = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="strict",
    )
    assert patch_strict == ""

    # 2. Lenient mode: bypasses pre-unit closure scan, successfully synthesizing patch
    patch_lenient = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="lenient",
    )
    assert patch_lenient != ""
    assert "yield from" in patch_lenient

@pytest.mark.parametrize(
    ("mode_arg", "default_mode", "expected"),
    [
        ("lenient", "strict", "lenient"),
        ("strict", "strict", "strict"),
        ("pedantic", "strict", "pedantic"),
        (None, "strict", "strict"),
        (None, "pedantic", "pedantic"),
        ("fast", "strict", "strict"),
        ("skip", "strict", "strict"),
        ("fail_closed", "strict", "strict"),
        ("unknown_mode", "pedantic", "pedantic"),
        ("unknown_mode", "strict", "strict"),
    ],
)
def test_resolve_closure_strictness_mode_aliases(
    mode_arg: Optional[str],
    default_mode: str,
    expected: str,
) -> None:
    """Verifies resolve_closure_strictness_mode maps canonical modes and falls back to default on unknown values."""
    assert resolve_closure_strictness_mode(mode_arg, default=default_mode) == expected

def test_generate_refactoring_patch_replace_clones_false_emits_helper_with_differing_outputs(
    tmp_path: Path,
) -> None:
    """Verifies that suggestion mode (replace_clones=False) emits shared helper
    despite output difference."""
    f = tmp_path / "single.py"
    f.write_text(
        "def run1():\n"
        "    a = 1\n"
        "    return a\n"
        "\n"
        "def run2():\n"
        "    b = 1\n"
        "    c = 2\n"
        "    return b + c\n",
        encoding="utf-8",
    )
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 2,
        "name": "run1:block",
        "kind": "compound_block",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "run2:block",
        "kind": "compound_block",
    }

    # When replacing clones, arity mismatch skips delegation
    patch_replace = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch_replace == ""

    # When replace_clones=False, helper-only suggestion mode emits the shared helper definition
    patch_preview = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=False
    )
    assert "def _shared_run1_run2" in patch_preview

def test_pre_unit_generator_expression_lazy_read_captured(tmp_path: Path) -> None:
    """Verifies that pre-unit lazy generator expressions capturing unit variables fail closed."""
    src1 = (
        "def g1(items: list[int]):\n"
        "    g = (extra for _ in items)\n"
        "    for x in items:\n"
        "        yield x\n"
        "        extra = x\n"
        "    return 0\n"
    )
    src2 = (
        "def g2(items: list[int]):\n"
        "    for y in items:\n"
        "        yield y\n"
        "        extra = y\n"
        "    return 0\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 3, "end": 5, "name": "g1:block", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 4, "name": "g2:block", "kind": "compound_block"}

    # In strict mode (default), lazy genexp capturing 'extra' causes rejection
    patch_strict = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="strict",
    )
    assert patch_strict == ""

    # In lenient mode, pre-unit closures are bypassed
    patch_lenient = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="lenient",
    )
    assert patch_lenient != ""

def test_pre_unit_class_generator_expression_lazy_read_captured(tmp_path: Path) -> None:
    """Verifies that pre-unit class-body generator expressions capturing unit
    variables fail closed."""
    src1 = (
        "def g1(items: list[int]):\n"
        "    class C:\n"
        "        gen = (extra for _ in items)\n"
        "    for x in items:\n"
        "        yield x\n"
        "        extra = x\n"
        "    return 0\n"
    )
    src2 = (
        "def g2(items: list[int]):\n"
        "    for y in items:\n"
        "        yield y\n"
        "        extra = y\n"
        "    return 0\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 4, "end": 6, "name": "g1:block", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 4, "name": "g2:block", "kind": "compound_block"}

    # In strict mode (default), class genexp capturing 'extra' causes rejection
    patch_strict = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="strict",
    )
    assert patch_strict == ""

    # In lenient mode, pre-unit closures are bypassed
    patch_lenient = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="lenient",
    )
    assert patch_lenient != ""

def test_load_unit_file_text_truncated_at_start_line_fails_closed() -> None:
    """Verifies that _load_unit_file_text returns None when source_lines has ambiguous length."""
    unit = {
        "file": "dummy.py",
        "start": 1,
        "end": 3,
        "source_lines": ["x = 1\n", "y = 2\n", "z = 3\n"],
    }
    # source_lines_is_sliced is None and len(source_lines) == (end - start + 1) -> fails closed
    assert _load_unit_file_text(unit) is None

    # When explicitly marked not sliced, it returns the content
    unit_full = dict(unit, source_lines_is_sliced=False)
    assert _load_unit_file_text(unit_full) == "x = 1\ny = 2\nz = 3\n"

def test_render_file_patch_plan_path_normalization_span_matching() -> None:
    """Verifies that _render_file_patch_plan normalizes file paths when cross-checking
    retained candidate spans, ensuring path drift does not cause replacement drops."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _FilePatchPlan,
        _render_file_patch_plan,
    )

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\n",
        is_new_file=False,
    )
    u = {"file": "./test.py", "start": 1, "end": 1}
    plan.replacements.append((u, "a = 99\n"))

    # Reconstructed unit copy with canonical path 'test.py'
    u_norm_copy = {"file": "test.py", "start": 1, "end": 1}
    with mock.patch(
        "pydoppelgangerhunt.fixer.patch.filter_overlapping_clone_units",
        return_value=[u_norm_copy],
    ):
        diff = _render_file_patch_plan(plan, replace_clones=True)
        assert "+a = 99" in diff

def test_render_file_patch_plan_non_standard_unit_span_defensive_recovery() -> None:
    """Verifies that _render_file_patch_plan defends against non-standard or malformed
    retained candidate dicts without raising ValueError."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _FilePatchPlan,
        _render_file_patch_plan,
    )

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\n",
        is_new_file=False,
    )
    u_valid = {"file": "test.py", "start": 1, "end": 1}
    plan.replacements.append((u_valid, "a = 99\n"))

    u_malformed = {"file": "test.py", "start": "bad_coord", "end": None}
    with mock.patch(
        "pydoppelgangerhunt.fixer.patch.filter_overlapping_clone_units",
        return_value=[u_malformed],
    ):
        diff = _render_file_patch_plan(plan, replace_clones=True)
        assert isinstance(diff, str)

def test_generate_refactoring_patch_unreadable_second_file_skips_pair(
    tmp_path: Path,
) -> None:
    """Verifies that an unreadable second file in a cross-file clone pair safely skips
    the pair and does not perform a partial refactor on the first file."""
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    f1.write_text("def f():\n    return 42\n", encoding="utf-8")
    f2.write_text("def g():\n    return 42\n", encoding="utf-8")

    u1 = {"name": "f", "file": "f1.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "g", "file": "f2.py", "start": 1, "end": 2, "kind": "function"}

    orig_read_text = Path.read_text

    def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "f2.py":
            raise OSError("Simulated disk read failure")
        return orig_read_text(self, *args, **kwargs)

    with mock.patch.object(Path, "read_text", autospec=True, side_effect=mock_read_text):
        patch = generate_refactoring_patch(
            [(1.0, u1, u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
    # Patch must be empty (pair skipped without refactoring u1)
    assert patch == ""

def test_record_skip_comment_same_file_plan_no_duplicate(tmp_path: Path) -> None:
    """Verifies that _record_skip_comment does not duplicate comments when f1 and f2
    alias the same file patch plan."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _FilePatchPlan,
        _record_skip_comment,
    )

    f = tmp_path / "same.py"
    plan = _FilePatchPlan(f, str(f.relative_to(tmp_path)), "x = 1\n")
    skip_reasons: Dict[str, int] = {}
    _record_skip_comment(plan, plan, skip_reasons, "test_skip", "# skip notice\n")
    assert plan.comments == ["# skip notice\n"]
    assert skip_reasons["test_skip"] == 1

def test_unit_start_module_scope_helper() -> None:
    """Verifies that _unit_start extracts and clamps unit start line coordinates."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _unit_start,
    )

    assert _unit_start({"start": 10}) == 10
    assert _unit_start({"start": 0}) == 1
    assert _unit_start({"start": -5}) == 1
    assert _unit_start({}) == 1

def test_validate_module_unit_replacements_strict_parameter() -> None:
    """Verifies that validate_module_unit_replacements respects the strict parameter."""
    from pydoppelgangerhunt.fixer.patch import validate_module_unit_replacements

    code = "def foo():\n    return 42\n"
    # Colon-delimited coordinate string (e.g. '1:0'): valid when strict=False,
    # raises ValueError when strict=True
    unit_colon = {"start": "1:0", "end": 2, "kind": "function", "file": "foo.py"}
    replacements = [(unit_colon, "def foo():\n    return 100\n")]

    validate_module_unit_replacements(code, replacements, strict=False)

    with pytest.raises(ValueError):
        validate_module_unit_replacements(code, replacements, strict=True)

def test_render_file_patch_plan_exact_key_disambiguates_same_line_units() -> None:
    """Verifies that _render_file_patch_plan disambiguates same-line units using exact keys
    including column coordinates, preventing dropped candidates from preempting winning units."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _FilePatchPlan,
        _render_file_patch_plan,
    )
    from pathlib import Path

    code = "    x = compute_inner_expr(val)\n"
    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text=code,
    )

    # u_outer spans cols 4..32, prioritized by filter_overlapping_clone_units
    u_outer = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 4,
        "end_col": 32,
        "name": "outer_assign",
        "kind": "statement",
    }
    # u_inner is a sub-expression on the same line (cols 8..32), dropped
    # by filter_overlapping_clone_units
    u_inner = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 8,
        "end_col": 32,
        "name": "inner_call",
        "kind": "call",
    }

    # Pass a copy of u_outer so its object id differs from the one retained
    # in filter_overlapping_clone_units
    u_outer_copy = dict(u_outer)
    # Place u_inner first in replacements
    plan.replacements = [
        (u_inner, "    inner_replacement()\n"),
        (u_outer_copy, "    outer_replacement()\n"),
    ]

    patch = _render_file_patch_plan(
        plan,
        replace_clones=True,
    )
    # The patch must apply u_outer (the winning unit) and NOT preempt it with u_inner
    assert "outer_replacement()" in patch
    assert "inner_replacement()" not in patch

def test_file_patch_plan_parsed_tree_uses_get_or_parse_tree(tmp_path: Path) -> None:
    """Verifies _FilePatchPlan.parsed_tree delegates to _get_or_parse_tree."""
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access
        _parsed_tree_cache,
    )
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan  # pylint: disable=protected-access

    p = tmp_path / "plan_test.py"
    text = "def hello(): pass\n"
    p.write_text(text, encoding="utf-8")
    plan = _FilePatchPlan(file_path=p, rel_path="plan_test.py", orig_text=text)
    tree1 = plan.parsed_tree
    assert tree1 is not None
    key = (plan.content_digest, len(text))
    assert key in _parsed_tree_cache
    assert _parsed_tree_cache[key][0] is tree1

def test_synthesize_shared_helper_code_unmapped_stmt_fails_closed(
    tmp_path: Path,
) -> None:
    """Verifies that synthesize_shared_helper_code fails closed immediately for unmapped
    statement types (such as ast.Assert or bare ast.AnnAssign) in complex_expr units."""
    src = (
        "def verify(a: int, b: int) -> None:\n"
        "    assert a > b\n"
        "    val: int\n"
    )
    f1 = tmp_path / "unmapped1.py"
    f2 = tmp_path / "unmapped2.py"
    f1.write_text(src, encoding="utf-8")
    f2.write_text(src, encoding="utf-8")

    u_assert1 = {"file": "unmapped1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u_assert2 = {"file": "unmapped2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}
    assert synthesize_shared_helper_code(u_assert1, u_assert2, repo_root=str(tmp_path)) == ""

    u_ann1 = {"file": "unmapped1.py", "start": 3, "end": 3, "name": "u1", "kind": "complex_expr"}
    u_ann2 = {"file": "unmapped2.py", "start": 3, "end": 3, "name": "u2", "kind": "complex_expr"}
    assert synthesize_shared_helper_code(u_ann1, u_ann2, repo_root=str(tmp_path)) == ""

def test_asymmetric_expr_kind_and_chained_assign_outputs_cleared(
    tmp_path: Path,
) -> None:
    """Verifies that synthesize_shared_helper_code handles kind-less u1 paired with
    kind='complex_expr' u2 (emitting return ... and -> Any), rejects conflicting explicit
    kinds, and clears outputs on chained whole-line assignments (a = b = expr)."""
    src1 = (
        "def calc_one(a: int, b: int) -> int:\n"
        "    x = y = (a + b) * (a - b)\n"
        "    return x + y\n"
    )
    src2 = (
        "def calc_two(a: int, b: int) -> int:\n"
        "    z = (a + b) * (a - b)\n"
        "    return z\n"
    )
    f1 = tmp_path / "chain1.py"
    f2 = tmp_path / "chain2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    # 1. u1 missing 'kind' while u2 has kind='complex_expr'
    u1_no_kind = {"file": "chain1.py", "start": 2, "end": 2, "name": "u1"}
    u2_expr = {
        "file": "chain2.py",
        "start": 2,
        "end": 2,
        "name": "u2",
        "kind": "complex_expr",
    }
    helper = synthesize_shared_helper_code(u1_no_kind, u2_expr, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in helper
    assert "return (a + b) * (a - b)" in helper

    # 2. Conflicting explicit kinds ('function' vs 'complex_expr') fail closed
    u1_conflict = {
        "file": "chain1.py",
        "start": 2,
        "end": 2,
        "name": "u1",
        "kind": "function",
    }
    assert synthesize_shared_helper_code(u1_conflict, u2_expr, repo_root=str(tmp_path)) == ""

    # 3. Chained assignment with pre-populated outputs on u1 (['x', 'y']) vs u2 (['z'])
    u1_chained = {
        "file": "chain1.py",
        "start": 2,
        "end": 2,
        "name": "u1",
        "kind": "complex_expr",
        "outputs": ["x", "y"],
    }
    u2_single = {
        "file": "chain2.py",
        "start": 2,
        "end": 2,
        "name": "u2",
        "kind": "complex_expr",
        "outputs": ["z"],
    }
    patch = generate_refactoring_patch(
        [(0.98, u1_chained, u2_single)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+    x = y = _shared_u1_u2(a, b)" in patch
    assert "+    z = _shared_u1_u2(a, b)" in patch

    # 4. Kind-less u1 paired with kind='complex_expr' u2 in generate_refactoring_patch
    patch_kindless = generate_refactoring_patch(
        [(0.98, dict(u1_no_kind), dict(u2_expr))],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+    x = y = _shared_u1_u2(a, b)" in patch_kindless
    assert "+    z = _shared_u1_u2(a, b)" in patch_kindless

def test_generate_refactoring_patch_transactional_rollback_multi_file_same_file_and_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies per-pair snapshot rollback leaves zero partial edits on multi-file, same-file, and cycle failures."""
    # 1. Multi-file pair-2 exception after Pair 1 succeeded and Pair 2 started modifying files:
    #    Pair 1 edits remain intact; Pair 2 edits and new module entries are completely rolled back.
    a1 = tmp_path / "a1.py"
    a2 = tmp_path / "a2.py"
    b1 = tmp_path / "b1.py"
    b2 = tmp_path / "b2.py"
    pair1_fn = (
        "def calc_one(x):\n"
        "    t = x + 10\n"
        "    return t * 2\n"
    )
    pair2_fn = (
        "def calc_two(y):\n"
        "    t = y + 20\n"
        "    return t * 3\n"
    )
    a1.write_text(pair1_fn, encoding="utf-8")
    a2.write_text(pair1_fn, encoding="utf-8")
    b1.write_text(pair2_fn, encoding="utf-8")
    b2.write_text(pair2_fn, encoding="utf-8")

    p1_u1 = {"file": str(a1), "start": 1, "end": 3, "name": "calc_one", "kind": "function"}
    p1_u2 = {"file": str(a2), "start": 1, "end": 3, "name": "calc_one", "kind": "function"}
    p2_u1 = {"file": str(b1), "start": 1, "end": 3, "name": "calc_two", "kind": "function"}
    p2_u2 = {"file": str(b2), "start": 1, "end": 3, "name": "calc_two", "kind": "function"}

    import pydoppelgangerhunt.fixer._patch_delegation as patch_deleg_mod

    real_replace = patch_deleg_mod.resolve_unit_replacement
    replace_calls = 0

    def _fail_second_pair_second_edit(*args: Any, **kwargs: Any) -> Any:
        nonlocal replace_calls
        replace_calls += 1
        # Fail after the 3rd replacement (middle of pair 2 after pair 2 touched its first file)
        if replace_calls == 4:
            raise RuntimeError("simulated mid-pair failure after first-file modification")
        return real_replace(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(
            patch_deleg_mod, "resolve_unit_replacement", _fail_second_pair_second_edit
        )
        patch_partial = generate_refactoring_patch(
            [(0.99, p1_u1, p1_u2), (0.99, p2_u1, p2_u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
    assert "calc_one" in patch_partial
    assert "b1.py" not in patch_partial
    assert "b2.py" not in patch_partial

    # 2. Same-file clone pair failure after first replacement leaves empty patch
    same_file = tmp_path / "same.py"
    same_file.write_text(
        "def left(x):\n    v = x + 1\n    return v * 2\n\n"
        "def right(x):\n    v = x + 1\n    return v * 2\n",
        encoding="utf-8",
    )
    sf_u1 = {"file": str(same_file), "start": 1, "end": 3, "name": "left", "kind": "function"}
    sf_u2 = {"file": str(same_file), "start": 5, "end": 7, "name": "right", "kind": "function"}
    sf_calls = 0

    def _fail_same_file_midway(*args: Any, **kwargs: Any) -> Any:
        nonlocal sf_calls
        sf_calls += 1
        if sf_calls == 2:
            raise RuntimeError("simulated same-file failure")
        return real_replace(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(
            patch_deleg_mod, "resolve_unit_replacement", _fail_same_file_midway
        )
        same_patch = generate_refactoring_patch(
            [(0.99, sf_u1, sf_u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
    assert same_patch == ""

    # 3. Circular import dependency across modules is rejected cleanly without mutating callers
    cyc_a = tmp_path / "cyc_a.py"
    cyc_b = tmp_path / "cyc_b.py"
    cyc_a.write_text(
        "import cyc_b\n\n"
        "def f1(x):\n"
        "    val = x + 1\n"
        "    return val * 2\n",
        encoding="utf-8",
    )
    cyc_b.write_text(
        "def f2(x):\n"
        "    val = x + 1\n"
        "    return val * 2\n",
        encoding="utf-8",
    )
    ca_u = {"file": str(cyc_a), "start": 3, "end": 5, "name": "f1", "kind": "function"}
    cb_u = {"file": str(cyc_b), "start": 1, "end": 3, "name": "f2", "kind": "function"}
    dg = build_module_graph(str(tmp_path))
    cyc_patch = generate_refactoring_patch(
        [(0.99, ca_u, cb_u)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="inline_first",
        depgraph=dg,
    )
    assert "cyc_b" in cyc_patch
    assert "from cyc_a import" not in cyc_patch
    assert "Circular import" in cyc_patch or "circular" in cyc_patch.lower()

def test_generate_refactoring_patch_uses_unit_source_code_when_second_unit_lacks_file_source(
    tmp_path: Path,
) -> None:
    """Verifies preview patches synthesize from u2 source_code without placeholder comments when disk source is absent."""
    mod_a = tmp_path / "mod_a.py"
    mod_a.write_text(
        "def compute_a(x, y):\n    val = (x * 2 + y)\n    return val\n",
        encoding="utf-8",
    )
    line_text = "    val = (x * 2 + y)"
    expr_start_col = line_text.index("x * 2 + y")
    expr_end_col = expr_start_col + len("x * 2 + y")

    u1 = {
        "file": str(mod_a),
        "name": "compute_a:expr",
        "start": 2,
        "end": 2,
        "start_col": expr_start_col,
        "end_col": expr_end_col,
        "kind": "complex_expr",
    }
    u2_with_src = {
        "file": str(tmp_path / "missing_mod_b.py"),
        "name": "compute_b:expr",
        "start": 7,
        "end": 7,
        "start_col": 14,
        "end_col": 23,
        "kind": "complex_expr",
        "allow_snippet_only": True,
        "source_code": "x * 2 + y",
    }
    patch_out = generate_refactoring_patch(
        [(0.99, u1, u2_with_src)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert "# Source for " not in patch_out
    assert "x * 2 + y" in patch_out
    assert "+def " in patch_out

    # Also verify whole-function u2 with source_code in default strict mode
    u1_fn = {
        "file": str(mod_a),
        "name": "compute_a",
        "start": 1,
        "end": 3,
        "kind": "function",
    }
    u2_fn_src = {
        "file": str(tmp_path / "missing_mod_b.py"),
        "name": "compute_b",
        "start": 10,
        "end": 12,
        "kind": "function",
        "source_code": "def compute_b(x, y):\n    val = (x * 2 + y)\n    return val\n",
    }
    fn_patch = generate_refactoring_patch(
        [(0.99, u1_fn, u2_fn_src)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert "# Source for " not in fn_patch
    assert "x * 2 + y" in fn_patch
    assert "+def " in fn_patch

    # Negative control: missing disk file and no source_code never emits placeholder helper lines
    u1_missing = {
        "file": str(tmp_path / "missing_mod_a.py"),
        "name": "compute_a",
        "start": 1,
        "end": 2,
        "kind": "function",
    }
    u2_missing = {
        "file": str(tmp_path / "missing_mod_b.py"),
        "name": "compute_b",
        "start": 1,
        "end": 2,
        "kind": "function",
    }
    empty_patch = generate_refactoring_patch(
        [(0.99, u1_missing, u2_missing)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert empty_patch == ""
    assert "# Source for " not in empty_patch

@pytest.mark.parametrize(
    ("exc_kind", "expect_diag_error", "expect_warning"),
    [
        pytest.param(
            "type_error",
            True,
            True,
            id="internal_type_error_triggers_diagnostic_error_and_warning",
        ),
        pytest.param(
            "value_error",
            True,
            True,
            id="internal_value_error_triggers_diagnostic_error_and_warning",
        ),
        pytest.param(
            "syntax_error",
            False,
            False,
            id="control_syntax_error_skips_at_debug",
        ),
        pytest.param(
            "unit_collision",
            False,
            False,
            id="control_unit_collision_error_skips_at_debug",
        ),
    ],
)
def test_generate_refactoring_patch_reports_internal_value_and_type_errors_as_diagnostic_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc_kind: str,
    expect_diag_error: bool,
    expect_warning: bool,
) -> None:
    """Verifies unexpected TypeError/ValueError invoke on_diagnostic_error and warn while SyntaxError/UnitCollisionError skip at DEBUG."""
    import pydoppelgangerhunt.fixer.patch as patch_mod  # pylint: disable=import-outside-toplevel

    exc_map: Dict[str, Exception] = {
        "type_error": TypeError("internal synthesis type bug"),
        "value_error": ValueError("internal synthesis value bug"),
        "syntax_error": SyntaxError("unparseable snippet"),
        "unit_collision": patch_mod.UnitCollisionError("unit overlap"),
    }
    src = (
        "def f1(a, b):\n    total = a + b\n    return total * 2\n\n"
        "def f2(a, b):\n    total = a + b\n    return total * 2\n"
    )
    mod_file = tmp_path / "diag_err.py"
    mod_file.write_text(src, encoding="utf-8")
    u1 = {"file": str(mod_file), "name": "f1", "kind": "function", "start": 1, "end": 3}
    u2 = {"file": str(mod_file), "name": "f2", "kind": "function", "start": 5, "end": 7}

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise exc_map[exc_kind]

    monkeypatch.setattr(patch_mod, "analyze_unit_variable_scope", _boom)

    diag_calls = 0

    def _on_diag() -> None:
        nonlocal diag_calls
        diag_calls += 1

    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.fixer.patch"):
        out = generate_refactoring_patch(
            [(0.99, u1, u2)],
            repo_root=str(tmp_path),
            on_diagnostic_error=_on_diag,
        )

    assert out == ""
    assert diag_calls == (1 if expect_diag_error else 0)
    has_warn = any(
        rec.levelno >= logging.WARNING
        and "Unexpected failure processing clone pair" in rec.getMessage()
        for rec in caplog.records
    )
    assert has_warn is expect_warning

@pytest.mark.parametrize(
    ("mode", "allowlist", "skip_pair", "expected_marker"),
    [
        pytest.param(
            "strict",
            None,
            True,
            "CustomTx",
            id="skipped_heuristic_pair_emits_no_summary_warning",
        ),
        pytest.param(
            "pedantic",
            ["*.atomic"],
            True,
            "*.atomic",
            id="skipped_wildcard_pair_emits_no_summary_warning",
        ),
        pytest.param(
            "strict",
            None,
            False,
            "CustomTx",
            id="control_committed_heuristic_pair_emits_summary_warning",
        ),
        pytest.param(
            "pedantic",
            ["*.atomic"],
            False,
            "*.atomic",
            id="control_committed_wildcard_pair_emits_summary_warning",
        ),
    ],
)
def test_generate_refactoring_patch_skipped_pairs_do_not_emit_heuristic_or_wildcard_summary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    allowlist: Optional[List[str]],
    skip_pair: bool,
    expected_marker: str,
) -> None:
    """Verifies skipped clone pairs do not pollute session-level heuristic or wildcard Summary warnings."""
    from pydoppelgangerhunt.config import (  # pylint: disable=import-outside-toplevel
        clear_closure_warning_cache,
    )

    mgr_expr = "CustomTx()" if allowlist is None else "db.atomic()"
    f1 = tmp_path / "m1.py"
    f2 = tmp_path / "m2.py"
    fn_a = (
        f"def run_a(db, step):\n"
        f"    with {mgr_expr}:\n"
        f"        val = step()\n"
        f"    cb = lambda: val\n"
        f"    res = cb() + 1\n"
        f"    return res\n"
    )
    fn_b = (
        f"def run_b(db, step):\n"
        f"    with {mgr_expr}:\n"
        f"        val = step()\n"
        f"    cb = lambda: val\n"
        f"    res = cb() + 1\n"
        f"    return res\n"
    )
    f1.write_text(fn_a, encoding="utf-8")
    f2.write_text(fn_b, encoding="utf-8")
    u1 = {
        "file": str(f1),
        "name": "run_a:block",
        "kind": "sliding_window",
        "start": 4,
        "end": 5,
    }
    u2 = {
        "file": str(f2),
        "name": "run_b:block",
        "kind": "sliding_window",
        "start": 4,
        "end": 5,
    }

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        patch_out = generate_refactoring_patch(
            [(0.99, u1, u2)],
            repo_root=str(tmp_path),
            cross_file_strategy="skip" if skip_pair else "host_module",
            closure_strictness=mode,
            non_suppressing_ctx_managers=allowlist,
        )

    summary_warnings = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "Summary:" in rec.getMessage()
    ]
    if skip_pair:
        assert patch_out == ""
        assert not summary_warnings
    else:
        assert "def _shared_" in patch_out
        assert any(expected_marker in msg for msg in summary_warnings), summary_warnings
