"""Unit tests for unit delegation calls, whole-method/async rewriting, expression pruning, and pragmas."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    check_units_overlap,
    generate_refactoring_patch,
    refactor_module_units,
    replace_unit_in_source,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import patch as patch_mod
from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access
    _build_whole_method_delegation,
)
from tests.helpers import assert_valid_python


def test_replace_unit_in_source_subline_and_token_slicing() -> None:
    """Verifies sub-line and token-range replacement preserving surrounding code and pragmas."""
    from pydoppelgangerhunt import replace_unit_in_source  # pylint: disable=import-outside-toplevel

    # 1. Statement sharing a line with prefix code and trailing pragma
    code1 = "x = 1; y = (a + b)  # type: ignore[operator]\n"
    unit_subline = {
        "file": "sub.py",
        "start": 1,
        "end": 1,
        "start_col": 7,
        "end_col": 18,
        "name": "y_assign",
    }
    replaced1 = replace_unit_in_source(code1, unit_subline, "y = helper(a, b)")
    assert replaced1 == "x = 1; y = helper(a, b)  # type: ignore[operator]\n"

    # 2. Multi-line unit with boundary pragma preservation
    code2 = (
        "def run_job():\n"
        "    val = compute_step()\n"
        "    return val  # type: ignore[no-any-return]\n"
    )
    unit_whole = {"file": "job.py", "start": 2, "end": 3, "name": "run_job_body"}
    replaced2 = replace_unit_in_source(code2, unit_whole, "    return _shared_compute_step()")
    assert "return _shared_compute_step()  # type: ignore[no-any-return]\n" in replaced2

    # 3. Whole-line with preserve_boundary_pragmas=False
    replaced3 = replace_unit_in_source(
        code2, unit_whole, "    return _shared_compute_step()", preserve_boundary_pragmas=False
    )
    assert "return _shared_compute_step()\n" in replaced3
    assert "# type: ignore" not in replaced3

def test_replace_unit_in_source_harvested_block_keeps_indentation(tmp_path: Path) -> None:
    """Verifies harvested indented multiline blocks are replaced as whole lines."""
    from pydoppelgangerhunt import replace_unit_in_source  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    code = (
        "def run(value: int) -> int:\n"
        "    if value > 0:\n"
        "        result = value + 1\n"
        "        return result\n"
        "    fallback = 0\n"
        "    return fallback\n"
    )
    src = tmp_path / "sample.py"
    src.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(src), str(tmp_path), min_lines=2, min_tokens=3)
    block_unit = next(unit for unit in units if unit.get("kind") == "compound_block")

    replaced = replace_unit_in_source(code, block_unit, "    helper_result = shared(value)\n    return helper_result\n")

    assert replaced == (
        "def run(value: int) -> int:\n"
        "    helper_result = shared(value)\n"
        "    return helper_result\n"
        "    fallback = 0\n"
        "    return fallback\n"
    )

def test_generate_refactoring_patch_same_class_whole_methods(tmp_path: Path) -> None:
    """Verifies that refactoring whole methods in the same class preserves headers and delegates."""
    f = tmp_path / "bank.py"
    code = (
        "class BankAccount:\n"
        "    def calc_interest_a(self, principal: float) -> float:\n"
        "        rate = 0.05\n"
        "        return principal * rate\n"
        "\n"
        "    def calc_interest_b(self, principal: float) -> float:\n"
        "        rate = 0.05\n"
        "        return principal * rate\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "calc_interest_a",
        "kind": "function",
        "enclosing_class": "BankAccount",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "calc_interest_b",
        "kind": "function",
        "enclosing_class": "BankAccount",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Private helper should be inserted inside BankAccount
    assert "+    def _shared_calc_interest_a_calc_interest_b(self, principal: float) -> float:" in patch
    # Signatures preserved and bodies delegate via self._shared_helper
    assert "+        return self._shared_calc_interest_a_calc_interest_b(principal)" in patch
    assert "-        rate = 0.05" in patch

    # Check replace_clones=False inserting helper inside class
    patch_preview = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=False)
    assert patch_preview
    assert "+    def _shared_calc_interest_a_calc_interest_b(self, principal: float) -> float:" in patch_preview
    assert "-        rate = 0.05" not in patch_preview

def test_generate_refactoring_patch_cross_class_receiver_injection(tmp_path: Path) -> None:
    """Verifies cross-class clones extract to module-level helper with explicit receiver injection."""
    f = tmp_path / "workers.py"
    code = (
        "class WorkerA:\n"
        "    def work(self, speed: int) -> int:\n"
        "        return speed * 10\n"
        "\n"
        "class WorkerB:\n"
        "    def work(self, speed: int) -> int:\n"
        "        return speed * 10\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "work",
        "kind": "function",
        "enclosing_class": "WorkerA",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "work",
        "kind": "function",
        "enclosing_class": "WorkerB",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Module-level helper with self: Any
    assert "+def _shared_work(self: Any, speed: int) -> int:" in patch
    # Call site delegates with self passed explicitly
    assert "+        return _shared_work(self, speed)" in patch

def test_build_whole_method_delegation_sync_and_async_generators() -> None:
    """Verifies whole-method delegation syntax for sync and async generators."""
    src = (
        "class Streamer:\n"
        "    def gen(self, count: int):\n"
        "        \"\"\"Docstring.\"\"\"\n"
        "        for i in range(count):\n"
        "            yield i\n"
    )
    unit = {"name": "gen", "start": 2, "end": 5}
    sync_del = _build_whole_method_delegation(
        src,
        unit,
        call_prefix="self.",
        helper_name="_shared_gen",
        args_str="count",
        has_yield=True,
        is_async=False,
        has_return=False,
    )
    assert "yield from self._shared_gen(count)" in sync_del

    sync_del_ret = _build_whole_method_delegation(
        src,
        unit,
        call_prefix="self.",
        helper_name="_shared_gen",
        args_str="count",
        has_yield=True,
        is_async=False,
        has_return=True,
    )
    assert "return (yield from self._shared_gen(count))" in sync_del_ret

    async_src = (
        "class AsyncStreamer:\n"
        "    async def stream(self, count: int):\n"
        "        for i in range(count):\n"
        "            yield i\n"
    )
    async_unit = {"name": "stream", "start": 2, "end": 4}
    async_del = _build_whole_method_delegation(
        async_src,
        async_unit,
        call_prefix="self.",
        helper_name="_shared_stream",
        args_str="count",
        has_yield=True,
        is_async=True,
    )
    assert "async for _item in self._shared_stream(count):\n            yield _item" in async_del

def test_two_space_async_generator_delegation(tmp_path: Path) -> None:
    """Verifies that 2-space indented files use 2-space step for async generator delegation."""
    code = (
        "async def gen1(src):\n"
        "  async for item in src:\n"
        "    yield item\n"
        "\n"
        "async def gen2(src):\n"
        "  async for item in src:\n"
        "    yield item\n"
    )
    f = tmp_path / "two_gen.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "gen1:asyncfor", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 6, "end": 7, "name": "gen2:asyncfor", "kind": "compound_block"}
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    assert "+  async for _item in _shared_gen1_gen2(src):\n+    yield _item\n" in patch

def test_cross_class_clone_call_site_receiver_arguments(tmp_path: Path) -> None:
    """Verifies that cross-class method clones preserve receiver arguments at both call sites."""
    code = (
        "class WorkerA:\n"
        "    def execute(self, payload: str) -> str:\n"
        "        return payload.strip().lower()\n"
        "\n"
        "class WorkerB:\n"
        "    def run_task(self, payload: str) -> str:\n"
        "        return payload.strip().lower()\n"
    )
    f = tmp_path / "workers.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": "workers.py", "start": 2, "end": 3, "name": "execute", "kind": "function"}
    u2 = {"file": "workers.py", "start": 6, "end": 7, "name": "run_task", "kind": "function"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_execute_run_task(self: Any, payload: str)" in patch
    assert "+        return _shared_execute_run_task(self, payload)" in patch
    # Verify WorkerB also passes self, not just WorkerA
    count_self_call = patch.count("_shared_execute_run_task(self, payload)")
    assert count_self_call == 2

def test_standalone_comprehension_replacement_preserves_indentation() -> None:
    """Verifies that an expression unit occupying its own line retains its leading indentation."""
    source = "def foo():\n    [x for x in data]\n"
    unit = {
        "kind": "comprehension",
        "start": 2,
        "end": 2,
        "start_col": 4,
        "end_col": 21,
    }
    rep = "_shared(data)"
    result = replace_unit_in_source(source, unit, rep)
    assert result == "def foo():\n    _shared(data)\n"

def test_refactor_module_units_same_line_column_order() -> None:
    """Verifies that refactor_module_units applies same-line replacements from right to left."""
    source = "val = [a for a in b] + [c for c in d]\n"
    u1 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 6,
        "end_col": 20,
    }
    u2 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 23,
        "end_col": 37,
    }
    replacements = [(u1, "helper(b)"), (u2, "helper(d)")]
    refactored = refactor_module_units(source, replacements)
    assert refactored == "val = helper(b) + helper(d)\n"

def test_build_whole_method_delegation_single_line_function() -> None:
    """Verifies that _build_whole_method_delegation cleanly handles single-line functions without IndentationError."""
    code = "def add(x: int, y: int) -> int: return x + y\n"
    unit = {"file": "t.py", "start": 1, "end": 1, "name": "add", "kind": "function"}
    res = _build_whole_method_delegation(code, unit, "", "_shared_add", "x, y")
    assert "def add(x: int, y: int) -> int:\n" in res
    assert "return _shared_add(x, y)" in res
    assert_valid_python(res)
    parsed = ast.parse(res)
    assert len(parsed.body) == 1
    assert isinstance(parsed.body[0], ast.FunctionDef)

def test_build_whole_method_delegation_decorated_single_line() -> None:
    """Verifies that _build_whole_method_delegation retains decorators on single-line functions."""
    code = "@decorator\ndef compute(x: int) -> int: return x * 2\n"
    unit = {"file": "t.py", "start": 1, "end": 2, "name": "compute", "kind": "function"}
    res = _build_whole_method_delegation(code, unit, "", "_shared_compute", "x")
    assert "@decorator\n" in res
    assert "def compute(x: int) -> int:\n" in res
    assert "return _shared_compute(x)" in res
    assert_valid_python(res)
    parsed = ast.parse(res)
    assert len(parsed.body) == 1
    fn_def = parsed.body[0]
    assert isinstance(fn_def, ast.FunctionDef)
    assert len(fn_def.decorator_list) == 1

def test_local_node_traversal_and_delegation_tie_breaking(tmp_path: Path) -> None:
    """Verifies _iter_local_nodes isolation and delegation tie-breaking."""
    from pydoppelgangerhunt.parser import harvest_file_units
    from pydoppelgangerhunt.fixer.patch import _build_whole_method_delegation

    code = (
        "class Handler:\n"
        "    def handle(self, req):\n"
        "        def validate(data):\n"
        "            if not data:\n"
        "                a = 1\n"
        "                b = 2\n"
        "                c = 3\n"
        "                d = 4\n"
        "                e = 5\n"
        "                return False\n"
        "            return True\n"
        "        return validate(req)\n"
    )
    src_file = tmp_path / "handler.py"
    src_file.write_text(code, encoding="utf-8")

    units_with_closures = harvest_file_units(
        str(src_file),
        repo_root=str(tmp_path),
        min_lines=4,
        min_tokens=5,
        harvest_closures=True,
    )
    names = [u["name"] for u in units_with_closures]
    assert "handle:If" not in names
    assert "Handler.handle:If" not in names
    assert any("validate:If" in name for name in names)

    validate_if = next(u for u in units_with_closures if "validate:If" in u["name"])
    assert validate_if.get("receiver_kind") is None
    assert validate_if.get("is_static") is False

    clause_units = harvest_file_units(
        str(src_file),
        repo_root=str(tmp_path),
        min_lines=4,
        min_tokens=5,
        clause_level=True,
        harvest_closures=True,
    )
    c_names = [u["name"] for u in clause_units]
    assert "handle:if_branch" not in c_names

    delegation_code = (
        "class A:\n"
        "    def method(self):\n"
        "        return 42\n"
    )
    u_delegation = {
        "file": str(src_file),
        "start": 2,
        "end": 3,
        "name": "method",
        "kind": "function",
    }
    del_res = _build_whole_method_delegation(
        delegation_code,
        u_delegation,
        call_prefix="self.",
        helper_name="_shared_method",
        args_str="",
    )
    assert "def method(self):" in del_res
    assert "return self._shared_method()" in del_res

def test_init_method_delegation_suppresses_return(tmp_path: Path) -> None:
    """Verifies that __init__ method refactoring never emits return in delegation calls."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class ModelA:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.val = val * 10\n"
        "        self.active = True\n"
        "\n"
        "class ModelB:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.val = val * 10\n"
        "        self.active = True\n"
    )
    f = tmp_path / "models.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "__init__", "file": "models.py", "start": 2, "end": 4, "kind": "function", "enclosing_class": "ModelA"}
    u2 = {"name": "__init__", "file": "models.py", "start": 7, "end": 9, "kind": "function", "enclosing_class": "ModelB"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    lines = [
        line for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    init_delegations = [line for line in lines if "_shared_init" in line]
    assert len(init_delegations) >= 2
    for line in init_delegations:
        assert "return " not in line

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
            "from models import ModelA, ModelB; "
            "a = ModelA(5); b = ModelB(7); "
            "print(a.val, a.active, b.val, b.active)",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "50 True 70 True"

def test_super_call_accepted_for_same_class_method_binding(tmp_path: Path) -> None:
    """Verifies that methods containing super() within the same class generate a valid class helper method."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class BaseHandler:\n"
        "    def perform(self, x: int) -> int:\n"
        "        return x * 10\n"
        "\n"
        "class CustomHandler(BaseHandler):\n"
        "    def handle_primary(self, x: int) -> int:\n"
        "        base_val = super().perform(x)\n"
        "        return base_val + 5\n"
        "\n"
        "    def handle_secondary(self, x: int) -> int:\n"
        "        base_val = super().perform(x)\n"
        "        return base_val + 5\n"
    )
    f = tmp_path / "handlers.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "handle_primary", "file": "handlers.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "CustomHandler"}
    u2 = {"name": "handle_secondary", "file": "handlers.py", "start": 10, "end": 12, "kind": "function", "enclosing_class": "CustomHandler"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "def _shared_handle_primary" in patch
    assert "super().perform(x)" in patch

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
            "from handlers import CustomHandler; "
            "h = CustomHandler(); "
            "print(h.handle_primary(3), h.handle_secondary(7))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "35 75"

def test_init_colon_notation_no_return_delegation(tmp_path: Path) -> None:
    """Verifies that __init__ methods with colon-prefixed names delegate without return statements."""
    src = (
        "class ConfigRecord:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.key = key\n"
        "        self.active = True\n"
        "\n"
        "class DataRecord:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.key = key\n"
        "        self.active = True\n"
    )
    f = tmp_path / "records.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "ConfigRecord:__init__",
        "file": "records.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ConfigRecord",
    }
    u2 = {
        "name": "DataRecord:__init__",
        "file": "records.py",
        "start": 7,
        "end": 9,
        "kind": "function",
        "enclosing_class": "DataRecord",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch != ""
    assert "return _shared" not in patch
    assert "_shared_ConfigRecord_DataRecord(self, key)" in patch

def test_generate_refactoring_patch_custom_receiver_paired_with_standalone(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch supports replacing clones between a custom receiver method and a function."""
    src1 = (
        "class Runner:\n"
        "    def execute(this, val: int) -> int:\n"
        "        return val * 2\n"
    )
    src2 = (
        "def execute_standalone(val: int) -> int:\n"
        "    return val * 2\n"
    )
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "a.py", "start": 2, "end": 3, "name": "execute"}
    u2 = {"file": "b.py", "start": 1, "end": 2, "name": "execute_standalone"}
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/a.py" in patch
    assert "--- a/b.py" in patch
    assert "_shared_execute_execute_standalone" in patch

def test_generate_refactoring_patch_custom_receiver_different_attrs_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing different attributes on custom receivers are not falsely merged."""
    src1 = (
        "class A:\n"
        "    def get_val(this) -> int:\n"
        "        return this.alpha\n"
    )
    src2 = (
        "class B:\n"
        "    def get_val(this) -> int:\n"
        "        return this.beta\n"
    )
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "a.py", "start": 2, "end": 3, "name": "get_val"}
    u2 = {"file": "b.py", "start": 2, "end": 3, "name": "get_val"}
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""

def test_generate_refactoring_patch_cross_receiver_attrs_matching(tmp_path: Path) -> None:
    """Verifies that methods accessing the same attribute under different receiver names (self vs this) are merged."""
    src1 = (
        "class A:\n"
        "    def get_val(self) -> int:\n"
        "        return self.total\n"
    )
    src2 = (
        "class B:\n"
        "    def get_val(this) -> int:\n"
        "        return this.total\n"
    )
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "a.py", "start": 2, "end": 3, "name": "get_val"}
    u2 = {"file": "b.py", "start": 2, "end": 3, "name": "get_val"}
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/a.py" in patch
    assert "--- a/b.py" in patch
    assert "_shared_get_val" in patch

def test_canonicalize_helper_preserves_comments_and_pragmas() -> None:
    """Verifies that canonicalizing relative imports preserves inline comments, pragmas, and formatting."""
    from pydoppelgangerhunt.fixer.patch import _canonicalize_helper_relative_imports  # pylint: disable=import-outside-toplevel

    helper = (
        "def helper_func(x: int) -> int:\n"
        "    # Important internal logic comment\n"
        "    from .utils import convert  # type: ignore[import-untyped]\n"
        "    # Another inline remark\n"
        "    return convert(x)  # noqa: E501\n"
    )
    canonicalized = _canonicalize_helper_relative_imports(helper, [("my_pkg.sub.mod", False)])
    assert "from my_pkg.sub.utils import convert  # type: ignore[import-untyped]" in canonicalized
    assert "# Important internal logic comment" in canonicalized
    assert "# Another inline remark" in canonicalized
    assert "# noqa: E501" in canonicalized

def test_extract_module_defined_names_augassign_and_walrus() -> None:
    """Verifies that AugAssign store targets and module-level walrus expressions are extracted as defined."""
    src = (
        "total = 0\n"
        "total += 10\n"
        "state.count += 5\n"
        "if (module_val := get_config()):\n"
        "    pass\n"
        "def helper_fn():\n"
        "    if (nested_val := 99):\n"
        "        pass\n"
    )
    defined = patch_mod._extract_module_defined_names(src)
    assert "total" in defined
    assert "module_val" in defined
    assert "state" not in defined
    assert "count" not in defined
    assert "nested_val" not in defined

def test_col_offset_to_char_offset_ascii_and_multibyte() -> None:
    """Verifies that col_offset_to_char_offset translates UTF-8 byte offsets to character indices."""
    from pydoppelgangerhunt.fixer import col_offset_to_char_offset

    # Edge cases
    assert col_offset_to_char_offset("", None) == 0
    assert col_offset_to_char_offset("abc", 0) == 0
    assert col_offset_to_char_offset("abc", -5) == 0
    assert col_offset_to_char_offset("abc", 100) == 3

    # ASCII line: byte offset equals character index
    ascii_line = "alpha = beta + gamma\n"
    assert col_offset_to_char_offset(ascii_line, 8) == 8
    assert col_offset_to_char_offset(ascii_line, 15) == 15

    # Multi-byte line with emojis (4 bytes each in UTF-8, 1 char in str)
    # "msg = '🚀🚀' + str(val)\n"
    # len("msg = '") = 7 bytes / 7 chars
    # '🚀🚀' = 8 bytes / 2 chars
    # "' + str(" = 8 bytes / 8 chars
    # Total byte offset of 'val' = 7 + 8 + 8 = 23 bytes
    # Total char index of 'val' = 7 + 2 + 8 = 17 chars
    emoji_line = "msg = '\U0001F680\U0001F680' + str(val)\n"
    assert col_offset_to_char_offset(emoji_line, 23) == 17
    assert emoji_line[17:20] == "val"

    # Multi-byte line with Chinese characters (3 bytes each in UTF-8, 1 char in str)
    # "title = '中文测试' + name\n"
    # len("title = '") = 9 bytes / 9 chars
    # '中文测试' = 12 bytes / 4 chars
    # "' + " = 4 bytes / 4 chars
    # Total byte offset of 'name' = 9 + 12 + 4 = 25 bytes
    # Total char index of 'name' = 9 + 4 + 4 = 17 chars
    cjk_line = "title = '中文测试' + name\n"
    assert col_offset_to_char_offset(cjk_line, 25) == 17
    assert cjk_line[17:21] == "name"

def test_compute_unit_byte_and_char_offsets() -> None:
    """Verifies that compute_unit_byte_offsets and compute_unit_char_offsets compute exact spans."""
    from pydoppelgangerhunt.fixer import (
        compute_unit_byte_offsets,
        compute_unit_char_offsets,
    )

    source = (
        "def compute(data):\n"
        "    msg = '\U0001F680' + str([x for x in data])\n"
        "    return msg\n"
    )
    # Unit for the list comprehension on line 2
    # In UTF-8: line 2 starts at byte 19 ("def compute(data):\n" is 19 bytes)
    # On line 2: "    msg = '\U0001F680' + str(" is 4 + 7 + 4 + 8 = 23 bytes
    # [x for x in data] is 17 bytes -> end col 40
    tree = ast.parse(source)
    comp_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ListComp))

    unit = {
        "file": "test.py",
        "start": comp_node.lineno,
        "end": comp_node.end_lineno,
        "start_col": comp_node.col_offset,
        "end_col": comp_node.end_col_offset,
        "kind": "comprehension",
    }

    char_span = compute_unit_char_offsets(source, unit)
    byte_span = compute_unit_byte_offsets(source, unit)

    # Character slice must match AST source segment
    assert source[char_span[0]:char_span[1]] == "[x for x in data]"
    # Byte slice in UTF-8 bytes must match
    source_bytes = source.encode("utf-8")
    assert source_bytes[byte_span[0]:byte_span[1]] == b"[x for x in data]"

def test_compute_unit_replacement_span_pragmas_and_whole_line() -> None:
    """Verifies that compute_unit_replacement_span correctly formats replacements and retains boundary pragmas."""
    from pydoppelgangerhunt.fixer import compute_unit_replacement_span

    # 1. Whole-line function replacement ensures trailing newline
    source = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    u_foo = {"file": "test.py", "start": 1, "end": 2, "kind": "function"}
    s_c, e_c, rep = compute_unit_replacement_span(source, u_foo, "def foo():\n    return 42")
    assert rep.endswith("\n")
    assert source[s_c:e_c] == "def foo():\n    return 1\n"

    # 2. Boundary pragma on line is retained
    source_pragma = "def calc():\n    val = 10  # type: ignore\n    return val\n"
    u_stmt = {"file": "test.py", "start": 2, "end": 2, "kind": "statement"}
    s_c2, e_c2, rep2 = compute_unit_replacement_span(
        source_pragma, u_stmt, "    val = compute_shared()", preserve_boundary_pragmas=True
    )
    assert "# type: ignore" in rep2
    assert source_pragma[s_c2:e_c2] == "    val = 10  # type: ignore\n"

def test_refactor_module_units_descending_offset_shuffled_order() -> None:
    """Verifies that refactor_module_units sorts replacements by descending offset regardless of input order."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    source = (
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
        "\n"
        "def third():\n"
        "    return 3\n"
    )
    u1 = {"file": "t.py", "start": 1, "end": 2, "name": "first"}
    u2 = {"file": "t.py", "start": 4, "end": 5, "name": "second"}
    u3 = {"file": "t.py", "start": 7, "end": 8, "name": "third"}

    r1 = (u1, "def first():\n    return 10\n")
    r2 = (u2, "def second():\n    return 20\n")
    r3 = (u3, "def third():\n    return 30\n")

    # Pass in shuffled order [r2, r1, r3]
    result = refactor_module_units(source, [r2, r1, r3])
    expected = (
        "def first():\n"
        "    return 10\n"
        "\n"
        "def second():\n"
        "    return 20\n"
        "\n"
        "def third():\n"
        "    return 30\n"
    )
    assert result == expected
    assert_valid_python(result)

def test_refactor_module_units_same_line_multiline_expansion_no_drift() -> None:
    """Verifies that a downstream multi-line replacement on the same line preserves upstream coordinates."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    source = "val = [a for a in b] + [c for c in d]\n"
    u1 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 6,
        "end_col": 20,
        "kind": "comprehension",
    }
    u2 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 23,
        "end_col": 37,
        "kind": "comprehension",
    }
    # Downstream replacement is multi-line with indentation
    rep2 = "helper(\n    d,\n    extra=True,\n)"
    rep1 = "helper(b)"

    # Pass in forward order: [u1, u2]
    refactored = refactor_module_units(source, [(u1, rep1), (u2, rep2)])
    expected = (
        "val = helper(b) + helper(\n"
        "    d,\n"
        "    extra=True,\n"
        ")\n"
    )
    assert refactored == expected
    assert_valid_python(refactored)

def test_refactor_module_units_touching_line_boundaries_no_drift() -> None:
    """Verifies that units touching line boundaries are refactored without line truncation or coordinate drift."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    source = (
        "# calculation\n"
        "res = (\n"
        "    compute_a(items) +\n"
        "    compute_b(items)\n"
        ")\n"
    )
    # Unit 1 is compute_a(items) on line 3, cols 4..20
    # Unit 2 is compute_b(items) on line 4, cols 4..20
    u1 = {
        "file": "test.py",
        "start": 3,
        "end": 3,
        "start_col": 4,
        "end_col": 20,
        "kind": "complex_expr",
    }
    u2 = {
        "file": "test.py",
        "start": 4,
        "end": 4,
        "start_col": 4,
        "end_col": 20,
        "kind": "complex_expr",
    }
    rep1 = "shared_a(\n        items,\n        mode='fast',\n    )"
    rep2 = "shared_b(\n        items,\n        mode='slow',\n    )"

    refactored = refactor_module_units(source, [(u1, rep1), (u2, rep2)])
    expected = (
        "# calculation\n"
        "res = (\n"
        "    shared_a(\n"
        "        items,\n"
        "        mode='fast',\n"
        "    ) +\n"
        "    shared_b(\n"
        "        items,\n"
        "        mode='slow',\n"
        "    )\n"
        ")\n"
    )
    assert refactored == expected
    assert_valid_python(refactored)

def test_refactor_module_units_unicode_multibyte_source_code() -> None:
    """Verifies that source code containing multi-byte UTF-8 emojis and symbols refactors cleanly."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    source = (
        "# Banner: \U0001F680 Space Rocket Launch \U0001F680\n"
        "title = 'Rocket: \U0001F680' + str([x for x in telemetry])\n"
        "notes = 'Details: \U0001F31F' + str([y for y in status])\n"
    )
    tree = ast.parse(source)
    comps = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]
    comps.sort(key=lambda n: n.lineno)

    u1 = {
        "file": "rocket.py",
        "start": comps[0].lineno,
        "end": comps[0].end_lineno,
        "start_col": comps[0].col_offset,
        "end_col": comps[0].end_col_offset,
        "kind": "comprehension",
    }
    u2 = {
        "file": "rocket.py",
        "start": comps[1].lineno,
        "end": comps[1].end_lineno,
        "start_col": comps[1].col_offset,
        "end_col": comps[1].end_col_offset,
        "kind": "comprehension",
    }

    rep1 = "process_telemetry(telemetry)"
    rep2 = "process_status(status)"

    refactored = refactor_module_units(source, [(u1, rep1), (u2, rep2)])
    assert "process_telemetry(telemetry)" in refactored
    assert "process_status(status)" in refactored
    assert "\U0001F680" in refactored
    assert "\U0001F31F" in refactored
    assert_valid_python(refactored)

def test_column_bounded_replacement_preserves_suffix_and_pragma() -> None:
    """Verifies that column-bounded replacements preserve trailing code and place boundary pragmas at line end."""
    from pydoppelgangerhunt.fixer import (
        compute_unit_replacement_span,
        replace_unit_in_source,
        resolve_unit_replacement,
    )

    # Case 1: Mid-line expression followed by binary operation and trailing type ignore
    source1 = "result = expensive_call() + other_expression  # type: ignore\n"
    u_call = {"file": "test_mod.py", "start": 1, "end": 1, "start_col": 9, "end_col": 25, "kind": "complex_expr"}
    refactored1 = replace_unit_in_source(source1, u_call, "shared_call()")
    assert refactored1 == "result = shared_call() + other_expression  # type: ignore\n"
    assert_valid_python(refactored1)

    # Case 2: compute_unit_replacement_span with boundary pragma attached to unit on line with trailing code
    source2 = "val = calculate() + offset  # type: ignore\n"
    u_calc = {"file": "test_mod.py", "start": 1, "end": 1, "start_col": 6, "end_col": 17, "kind": "complex_expr"}
    s_c, e_c, final_rep = compute_unit_replacement_span(source2, u_calc, "new_calc()", preserve_boundary_pragmas=True)
    assembled = source2[:s_c] + final_rep + source2[e_c:]
    assert assembled == "val = new_calc() + offset  # type: ignore\n"
    assert_valid_python(assembled)

    # Case 3: Trailing code without terminal newline at EOF
    source3 = "out = compute_val() + 10  # noqa"
    u_comp = {"file": "test_mod.py", "start": 1, "end": 1, "start_col": 6, "end_col": 19, "kind": "complex_expr"}
    refactored3 = replace_unit_in_source(source3, u_comp, "fast_val()")
    assert refactored3 == "out = fast_val() + 10  # noqa"
    assert_valid_python(refactored3)

    # Case 4: Attached boundary pragma (# type: ignore) on multi-line unit with different pragma in suffix (# noqa)
    source4 = "res = calculate(  # type: ignore\n    data\n)  # noqa: E501\n"
    u_multi = {"file": "test_mod.py", "start": 1, "end": 3, "start_col": 6, "end_col": 1, "kind": "complex_expr"}
    item4 = resolve_unit_replacement(source4, u_multi, "fast_calc(data)", preserve_boundary_pragmas=True)
    assert item4.consumes_line_suffix is True
    refactored4 = replace_unit_in_source(source4, u_multi, "fast_calc(data)", preserve_boundary_pragmas=True)
    assert refactored4 == "res = fast_calc(data)  # noqa: E501  # type: ignore\n"
    assert_valid_python(refactored4)

    # Case 5: Attached boundary pragma on multi-line unit with regular comment in suffix
    source5 = "res = calculate(  # type: ignore\n    data\n)  # compute offset\n"
    item5 = resolve_unit_replacement(source5, u_multi, "fast_calc(data)", preserve_boundary_pragmas=True)
    assert item5.consumes_line_suffix is True
    refactored5 = replace_unit_in_source(source5, u_multi, "fast_calc(data)", preserve_boundary_pragmas=True)
    assert refactored5 == "res = fast_calc(data)  # compute offset  # type: ignore\n"
    assert_valid_python(refactored5)

def test_refactor_module_units_same_line_emoji_prefix_replacement() -> None:
    """Verifies that AST byte offsets are translated to character offsets when replacing code units preceded by emojis."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    # Single emoji before list comprehension on the same line
    emoji_source = "prefix = '🚀'; value = [x for x in data]\n"
    tree1 = ast.parse(emoji_source)
    comp_node = next(n for n in ast.walk(tree1) if isinstance(n, ast.ListComp))
    assert comp_node.col_offset == 25  # In UTF-8 bytes ('🚀' is 4 bytes vs 1 char)
    u_comp = {
        "file": "emoji_mod.py",
        "start": comp_node.lineno,
        "end": comp_node.end_lineno,
        "start_col": comp_node.col_offset,
        "end_col": comp_node.end_col_offset,
        "kind": "comprehension",
    }
    refactored_emoji = refactor_module_units(emoji_source, [(u_comp, "helper(data)")])
    assert refactored_emoji == "prefix = '🚀'; value = helper(data)\n"
    assert_valid_python(refactored_emoji)

    # Multiple multi-byte characters and trailing statements
    multi_emoji_source = "tag = '🎉✨🔥'; items = [i * 2 for i in raw_items]; done = True\n"
    tree2 = ast.parse(multi_emoji_source)
    comp_node2 = next(n for n in ast.walk(tree2) if isinstance(n, ast.ListComp))
    u_comp2 = {
        "file": "emoji_mod.py",
        "start": comp_node2.lineno,
        "end": comp_node2.end_lineno,
        "start_col": comp_node2.col_offset,
        "end_col": comp_node2.end_col_offset,
        "kind": "comprehension",
    }
    refactored_multi = refactor_module_units(multi_emoji_source, [(u_comp2, "transform(raw_items)")])
    assert refactored_multi == "tag = '🎉✨🔥'; items = transform(raw_items); done = True\n"
    assert_valid_python(refactored_multi)

def test_col_offset_to_char_offset_clamps_malformed_utf8_offset() -> None:
    """Verifies that col_offset_to_char_offset clamps to the nearest valid character boundary when offset lands mid-sequence."""
    from pydoppelgangerhunt.fixer import col_offset_to_char_offset

    line = "header = '\U0001F680'; data = 42\n"
    # '\U0001F680' begins at byte 10 and ends at byte 14 (UTF-8 bytes: 10, 11, 12, 13)
    # Byte offset 10 points to the emoji character (char index 10)
    assert col_offset_to_char_offset(line, 10) == 10
    # Byte offsets 11, 12, 13 land in the middle of '\U0001F680' and clamp to char index 10
    assert col_offset_to_char_offset(line, 11) == 10
    assert col_offset_to_char_offset(line, 12) == 10
    assert col_offset_to_char_offset(line, 13) == 10
    # Byte offset 14 points past the emoji (char index 11)
    assert col_offset_to_char_offset(line, 14) == 11

def test_multi_unit_expansion_with_boundary_pragma_line_suffix_extension() -> None:
    """Verifies multi-unit replacements where both units expand and one has a trailing boundary pragma."""
    from pydoppelgangerhunt.fixer import (
        ReplacementItem,
        refactor_module_units,
        resolve_unit_replacement,
    )

    source_text = (
        "x = 1  # type: ignore\n"
        "y = [1, 2, 3] ; extra = 10\n"
        "z = [4, 5, 6]\n"
        "out = None\n"
    )
    # Unit 1: spans lines 1 to 2, ending at column 13 on line 2 (cols 0-13)
    u1 = {
        "file": "calc.py",
        "name": "u1",
        "start": 1,
        "end": 2,
        "start_col": 0,
        "end_col": 13,
        "kind": "complex_expr",
    }
    # Unit 2: 'z = [4, 5, 6]' on line 3 (cols 0-13)
    u2 = {
        "file": "calc.py",
        "name": "u2",
        "start": 3,
        "end": 3,
        "start_col": 0,
        "end_col": 13,
        "kind": "comprehension",
    }

    rep1 = "x = 1\ny = (\n    1,\n    2,\n    3,\n)"
    rep2 = "z = (\n    4,\n    5,\n    6,\n)"

    # Test resolve_unit_replacement directly
    item1 = resolve_unit_replacement(source_text, u1, rep1, preserve_boundary_pragmas=True)
    assert isinstance(item1, ReplacementItem)
    assert item1.consumes_line_suffix is True
    line1_len = len("x = 1  # type: ignore\n")
    line2_len = len("y = [1, 2, 3] ; extra = 10\n")
    assert item1.end_char == line1_len + line2_len
    assert "# type: ignore" in item1.final_rep
    assert "; extra = 10" in item1.final_rep

    # Test refactor_module_units applying both multi-line expansions
    reps = [(u1, rep1), (u2, rep2)]
    refactored = refactor_module_units(source_text, reps)

    assert "y = (\n    1,\n    2,\n    3,\n) ; extra = 10  # type: ignore\n" in refactored
    assert "z = (\n    4,\n    5,\n    6,\n)" in refactored
    assert "out = None\n" in refactored

def test_col_offset_to_char_offset_utf8_multibyte_clamping_and_exact_boundary() -> None:
    """Verifies that col_offset_to_char_offset accurately handles multi-byte UTF-8 boundaries and clamps."""
    from pydoppelgangerhunt.fixer import (
        col_offset_to_char_offset,
        compute_unit_spans,
    )

    # Greek characters: each 'α', 'β', 'γ' is 2 UTF-8 bytes.
    # Total bytes: 2 ('α') + 2 ('β') + 2 ('γ') + 1 ('\n') = 7 bytes.
    line = "αβγ\n"
    assert line.encode("utf-8") == b"\xce\xb1\xce\xb2\xce\xb3\n"

    # Landing exactly at byte 0: char 0 ("")
    assert col_offset_to_char_offset(line, 0) == 0
    # Landing exactly at byte 2 (end of 'α'): char 1 ("α")
    assert col_offset_to_char_offset(line, 2) == 1
    # Landing at byte 3 (middle of 'β' byte sequence): clamped safely to preceding valid char boundary (char 1)
    assert col_offset_to_char_offset(line, 3) == 1
    # Landing exactly at byte 4 (end of 'β'): char 2 ("αβ")
    assert col_offset_to_char_offset(line, 4) == 2
    # Landing at byte 5 (middle of 'γ'): clamped safely to preceding char boundary (char 2)
    assert col_offset_to_char_offset(line, 5) == 2
    # Landing exactly at byte 6 (end of 'γ'): char 3 ("αβγ")
    assert col_offset_to_char_offset(line, 6) == 3

    # Test through compute_unit_spans with exact and clamped column bounds
    u_exact = {"file": "greek.py", "start": 1, "end": 1, "start_col": 0, "end_col": 4, "kind": "complex_expr"}
    span_exact = compute_unit_spans(line, u_exact)
    assert span_exact.start_char == 0
    assert span_exact.end_char == 2
    assert span_exact.start_byte == 0
    assert span_exact.end_byte == 4

    u_clamped = {"file": "greek.py", "start": 1, "end": 1, "start_col": 0, "end_col": 3, "kind": "complex_expr"}
    span_clamped = compute_unit_spans(line, u_clamped)
    assert span_clamped.start_char == 0
    assert span_clamped.end_char == 1
    assert span_clamped.start_byte == 0
    assert span_clamped.end_byte == 3

def test_col_offset_to_char_offset_ascii_fast_path_equivalence() -> None:
    """Verifies that the ASCII fast-path in col_offset_to_char_offset matches UTF-8 decoding."""
    from pydoppelgangerhunt.fixer import col_offset_to_char_offset

    line = "def calculate_total(price, tax):"
    assert line.isascii() is True

    for c in [-5, 0, 3, 10, len(line), len(line) + 10]:
        assert col_offset_to_char_offset(line, c) == max(0, min(len(line), c))

def test_package_reexport_col_offset_to_char_offset() -> None:
    """Verifies that col_offset_to_char_offset is exported in the root pydoppelgangerhunt package."""
    import pydoppelgangerhunt

    assert hasattr(pydoppelgangerhunt, "col_offset_to_char_offset")
    assert "col_offset_to_char_offset" in pydoppelgangerhunt.__all__
    assert callable(pydoppelgangerhunt.col_offset_to_char_offset)

def test_refactor_module_units_inverted_line_range_is_noop() -> None:
    """Verifies that inverted line ranges (start > end) fail closed with ValueError
    in mutation APIs."""
    from pydoppelgangerhunt.fixer import refactor_module_units, replace_unit_in_source

    original_code = "line1\nline2\nline3\n"
    inverted_unit = {"file": "sample.py", "name": "bad_inv", "start": 5, "end": 2}

    # refactor_module_units fails closed with ValueError
    with pytest.raises(ValueError, match="invalid coordinates for refactoring"):
        refactor_module_units(original_code, [(inverted_unit, "INVERTED_REPLACEMENT\n")])

    # replace_unit_in_source fails closed with ValueError
    with pytest.raises(ValueError, match="invalid coordinates for replacement"):
        replace_unit_in_source(original_code, inverted_unit, "INVERTED_REPLACEMENT\n")

    # Out of bounds start line
    oob_unit = {"file": "sample.py", "name": "bad_oob", "start": 99, "end": 100}
    result_oob = refactor_module_units(original_code, [(oob_unit, "OOB_REPLACEMENT\n")])
    assert result_oob == original_code

def test_col_offset_to_char_offset_surrogate_code_points() -> None:
    """Verifies that col_offset_to_char_offset and compute_unit_spans handle lone surrogates without error."""
    from pydoppelgangerhunt.fixer import col_offset_to_char_offset
    from pydoppelgangerhunt.fixer.source import compute_unit_spans

    # Lone surrogate character (U+D800) encoded in UTF-8 as 3 bytes (0xED 0xA0 0x80)
    surrogate_line = "val = '\ud800'"
    # Offset 0..7: 'val = \'' (7 ASCII characters, bytes 0..7)
    assert col_offset_to_char_offset(surrogate_line, 6) == 6
    assert col_offset_to_char_offset(surrogate_line, 7) == 7
    # Offset 8 & 9: mid-sequence bytes within the 3-byte surrogate; clamps back to 7
    assert col_offset_to_char_offset(surrogate_line, 8) == 7
    assert col_offset_to_char_offset(surrogate_line, 9) == 7
    # Offset 10: boundary after surrogate character (character index 8)
    assert col_offset_to_char_offset(surrogate_line, 10) == 8
    # Offset 11: boundary after closing quote (character index 9)
    assert col_offset_to_char_offset(surrogate_line, 11) == 9

    # compute_unit_spans with surrogate code points
    source_with_surrogate = "val = '\ud800'\nprint(val)\n"
    unit = {"file": "surrogate.py", "start": 1, "end": 1, "start_col": 0, "end_col": 10}
    span = compute_unit_spans(source_with_surrogate, unit)
    assert span.start_char == 0
    assert span.start_byte == 0
    assert span.end_char > 0
    assert span.end_byte > 0

def test_refactor_module_units_missing_file_key_defaults_to_module() -> None:
    """Verifies that refactor_module_units defaults missing 'file' keys to <module> for Tier 1 validation."""
    from pydoppelgangerhunt.fixer import UnitCollisionError, refactor_module_units

    code = "x = 1\ny = 2\nz = 3\n"
    # Units without 'file' keys that overlap in line ranges
    u1 = {"name": "u1", "start": 1, "end": 2}
    u2 = {"name": "u2", "start": 2, "end": 3}

    with pytest.raises(UnitCollisionError) as exc_info:
        refactor_module_units(code, [(u1, "rep1\n"), (u2, "rep2\n")])
    assert "<module>" in str(exc_info.value)
    assert "Overlapping unit collision detected" in str(exc_info.value)

def test_refactor_module_units_dry_run_validation() -> None:
    """Verifies that refactor_module_units with dry_run=True validates collisions without mutating source."""
    from pydoppelgangerhunt.fixer import UnitCollisionError, refactor_module_units

    code = "def f1():\n    return 1\n\ndef f2():\n    return 2\n"
    u1 = {"file": "sample.py", "start": 1, "end": 2}
    u2 = {"file": "sample.py", "start": 4, "end": 5}

    # Dry run with valid non-overlapping replacements returns original source_text unchanged
    res = refactor_module_units(
        code,
        [(u1, "def f1_new():\n    return 10\n"), (u2, "def f2_new():\n    return 20\n")],
        dry_run=True,
    )
    assert res == code

    # Dry run with colliding replacements still raises UnitCollisionError
    u_collide = {"file": "sample.py", "start": 2, "end": 4}
    with pytest.raises(UnitCollisionError):
        refactor_module_units(
            code,
            [(u1, "def f1_new():\n    return 10\n"), (u_collide, "pass\n")],
            dry_run=True,
        )

def test_refactor_module_units_form_feed_line_replacement() -> None:
    """Verifies that refactor_module_units replaces whole statements with form-feed indentation cleanly."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    source_with_ff = "def fn():\n\x0c    x = 10\n    return x\n"
    unit_stmt = {"file": "ff_test.py", "start": 2, "end": 2, "start_col": 5, "end_col": 11}
    # Whole-statement replacement must replace the entire line including \f, not just \f
    res = refactor_module_units(source_with_ff, [(unit_stmt, "    x = 20\n")])
    assert res == "def fn():\n    x = 20\n    return x\n"

    # Sub-expression replacement inside a form-feed line
    source_ff_expr = "def fn():\n\x0c    flag = x = 10\n    return x\n"
    # Column offset 12 is 'x = 10' (\f is 1 byte, 4 spaces is 4 bytes, 'flag = ' is 7 bytes: 1+4+7=12)
    unit_sub = {"file": "ff_test.py", "start": 2, "end": 2, "start_col": 12, "end_col": 18}
    res_sub = refactor_module_units(source_ff_expr, [(unit_sub, "y = 99")])
    assert res_sub == "def fn():\n\x0c    flag = y = 99\n    return x\n"

def test_refactor_module_units_boundary_touching_zero_width_insertion_order() -> None:
    """Verifies deterministic application order for zero-width insertions touching replacement boundaries."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    code = "abcdefghij\n"
    u_rep = {"file": "test.py", "start": 1, "end": 1, "start_col": 2, "end_col": 5}

    # Zero-width insertion at start boundary [2, 2) is applied before replacement text [2, 5)
    u_ins_start = {"file": "test.py", "start": 1, "end": 1, "start_col": 2, "end_col": 2}
    res_start = refactor_module_units(code, [(u_rep, "XYZ"), (u_ins_start, "INS_")])
    assert res_start == "abINS_XYZfghij\n"

    # Zero-width insertion at end boundary [5, 5) is applied after replacement text [2, 5)
    u_ins_end = {"file": "test.py", "start": 1, "end": 1, "start_col": 5, "end_col": 5}
    res_end = refactor_module_units(code, [(u_rep, "XYZ"), (u_ins_end, "_INS")])
    assert res_end == "abXYZ_INSfghij\n"

def test_resolve_unit_replacement_last_line_offset_indexing() -> None:
    """Verifies resolve_unit_replacement on the last line uses offset arrays without string re-encoding."""
    from pydoppelgangerhunt.fixer import resolve_unit_replacement
    from pydoppelgangerhunt.fixer.source import _compute_line_offsets, split_source_lines

    # Case 1: Clean line end with replacement ending in newline
    code1 = "x = 1\ny = 2 + 3\n"
    lines1 = split_source_lines(code1)
    char_offs1, byte_offs1 = _compute_line_offsets(lines1)
    unit1 = {"file": "test.py", "start": 2, "end": 2, "start_col": 4, "end_col": 9}
    item1 = resolve_unit_replacement(
        code1,
        unit1,
        "42\n",
        preserve_boundary_pragmas=True,
        lines=lines1,
        line_char_offsets=char_offs1,
        line_byte_offsets=byte_offs1,
    )
    assert item1.consumes_line_suffix is True
    assert item1.end_char == char_offs1[2]
    assert item1.end_byte == byte_offs1[2]
    assert item1.final_rep == "42\n"

    # Case 2: Unclean suffix with missing pragmas appending to line end
    code2 = "a = 1  # type: ignore\ny = (2 + 3) * 5\n"
    lines2 = split_source_lines(code2)
    char_offs2, byte_offs2 = _compute_line_offsets(lines2)
    unit2 = {"file": "test.py", "start": 1, "end": 2, "start_col": 0, "end_col": 11}
    item2 = resolve_unit_replacement(
        code2,
        unit2,
        "res = 42",
        preserve_boundary_pragmas=True,
        lines=lines2,
        line_char_offsets=char_offs2,
        line_byte_offsets=byte_offs2,
    )
    assert item2.consumes_line_suffix is True
    assert item2.end_char == char_offs2[2]
    assert item2.end_byte == byte_offs2[2]
    assert item2.final_rep == "res = 42 * 5  # type: ignore\n"

def test_dual_tier_rejection_and_pragma_expanded_catch() -> None:
    """Verifies both Tier 1 coordinate rejection and Tier 2 pragma-expanded collision catch."""
    from pydoppelgangerhunt.fixer.patch import UnitCollisionError

    # 1. Tier 1 Semantic Overlap Rejection
    u_t1_a = {"name": "fn1", "file": "mod.py", "start": 1, "end": 2}
    u_t1_b = {"name": "fn2", "file": "mod.py", "start": 2, "end": 3}
    assert check_units_overlap(u_t1_a, u_t1_b) is True
    with pytest.raises(UnitCollisionError, match="Overlapping unit collision detected"):
        refactor_module_units("x = 1\ny = 2\nz = 3\n", [(u_t1_a, "pass\n"), (u_t1_b, "pass\n")], tier1=True, tier2=False)

    # 2. Tier 2 Catch of Pragma-Expanded Collision
    src = "a = 1  # type: ignore\nb = 2\n"
    # u1: line 1, col 0..5 (statement part). Because line 1 has a trailing pragma,
    # resolve_unit_replacement expands it to whole line [0, line_bytes[1]) to preserve pragma.
    u_p1 = {"name": "u1", "file": "mod.py", "start": 1, "end": 1, "start_col": 0, "end_col": 5}
    # u2: line 1, col 7..21 (pointing into trailing comment/token on same line).
    u_p2 = {"name": "u2", "file": "mod.py", "start": 1, "end": 1, "start_col": 7, "end_col": 21, "kind": "complex_expr"}

    # In Tier 1, column intervals [0, 5) and [7, 21) do not overlap
    assert check_units_overlap(u_p1, u_p2) is False

    # Tier 1 alone allows it without error
    out = refactor_module_units(src, [(u_p1, "a = 10\n"), (u_p2, "comment")], tier1=True, tier2=False)
    assert out is not None

    # Tier 2 catches the physical interval collision caused by pragma whole-line expansion
    with pytest.raises(UnitCollisionError, match="Overlapping unit collision detected between 'u1' \\(1-1\\) and 'u2' \\(1-1\\)"):
        refactor_module_units(src, [(u_p1, "a = 10\n"), (u_p2, "comment")], tier1=True, tier2=True)

def test_generate_refactoring_patch_async_generator_try_cleanup_rejected(tmp_path: Path) -> None:
    """Verifies that clone pairs of async generators with cleanup blocks are rejected."""
    src1 = (
        "async def agen1(items: list[int]):\n"
        "    try:\n"
        "        for x in items:\n"
        "            yield x\n"
        "    finally:\n"
        "        pass\n"
    )
    src2 = (
        "async def agen2(items: list[int]):\n"
        "    try:\n"
        "        for x in items:\n"
        "            yield x\n"
        "    finally:\n"
        "        pass\n"
    )
    f1 = tmp_path / "as1.py"
    f2 = tmp_path / "as2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    assert_valid_python(src1)
    assert_valid_python(src2)

    u1 = {"file": str(f1), "start": 1, "end": 6, "name": "agen1", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 6, "name": "agen2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generate_refactoring_patch_async_generator_with_bare_return_accepted(tmp_path: Path) -> None:
    """Verifies that clone pairs of async generators with bare return statements are accepted and refactored."""
    src1 = (
        "async def agen1(items: list[int]):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            return\n"
        "        yield x\n"
    )
    src2 = (
        "async def agen2(items: list[int]):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            return\n"
        "        yield x\n"
    )
    f1 = tmp_path / "as1.py"
    f2 = tmp_path / "as2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 5, "name": "agen1", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 5, "name": "agen2", "kind": "function"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "async def _shared_agen1_agen2" in patch
    assert "-> AsyncIterator[" in patch

def test_generate_refactoring_patch_async_generator_with_downstream_outputs_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that async generator clones requiring downstream outputs are rejected."""
    src1 = (
        "async def af1(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    print(total)\n"
    )
    src2 = (
        "async def af2(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += x\n"
        "        yield x\n"
        "    print(count)\n"
    )
    f1 = tmp_path / "ag_out1.py"
    f2 = tmp_path / "ag_out2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    assert_valid_python(src1)
    assert_valid_python(src2)

    u1 = {
        "file": str(f1), "start": 2, "end": 5,
        "name": "af1:for", "kind": "compound_block", "is_async": True,
    }
    u2 = {
        "file": str(f2), "start": 2, "end": 5,
        "name": "af2:for", "kind": "compound_block", "is_async": True,
    }

    patch = generate_refactoring_patch(
        [(0.90, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generate_refactoring_patch_mixed_async_sync_rejected(tmp_path: Path) -> None:
    """Verifies that pairing an async function or generator with a sync counterpart is rejected in synthesis and patch."""
    src1 = (
        "async def compute1(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n"
    )
    src2 = (
        "def compute2(items: list[int]):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n"
    )
    f1 = tmp_path / "as1.py"
    f2 = tmp_path / "as2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 5, "name": "compute1", "kind": "function", "is_async": True}
    u2 = {"file": str(f2), "start": 1, "end": 5, "name": "compute2", "kind": "function", "is_async": False}

    # 1. Direct synthesis check: verify synthesis rejects mixed async parity
    mixed_helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert mixed_helper == ""

    # 2. Control check: matching sync parity synthesizes cleanly
    sync_helper = synthesize_shared_helper_code(u2, u2, repo_root=str(tmp_path))
    assert sync_helper != ""
    assert "def _shared_compute2" in sync_helper

    # 3. Patch generation check: verify patch generation rejects mixed async parity
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

    # 4. Mixed async generator vs sync generator: verify rejection
    src_ag = (
        "async def ag1(items: list[int]):\n"
        "    for x in items:\n"
        "        yield x\n"
    )
    src_sg = (
        "def sg2(items: list[int]):\n"
        "    for x in items:\n"
        "        yield x\n"
    )
    f_ag = tmp_path / "ag.py"
    f_sg = tmp_path / "sg.py"
    f_ag.write_text(src_ag, encoding="utf-8")
    f_sg.write_text(src_sg, encoding="utf-8")
    u_ag = {
        "file": str(f_ag),
        "start": 1,
        "end": 3,
        "name": "ag1",
        "kind": "function",
        "is_async": True,
    }
    u_sg = {
        "file": str(f_sg),
        "start": 1,
        "end": 3,
        "name": "sg2",
        "kind": "function",
        "is_async": False,
    }

    assert synthesize_shared_helper_code(u_ag, u_sg, repo_root=str(tmp_path)) == ""
    assert generate_refactoring_patch(
        [(1.0, u_ag, u_sg)], repo_root=str(tmp_path), replace_clones=True
    ) == ""

def test_generate_refactoring_patch_mismatched_subroutine_kinds_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that clone pairs where one unit is a block and the other is a
    whole function are rejected."""
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
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    f1 = tmp_path / "m1.py"
    f2 = tmp_path / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    # u1 is a compound block ("f1:for"), u2 is a whole function ("f2")
    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "f1:for", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 1, "end": 6, "name": "f2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generate_refactoring_patch_subroutine_unpaired_outputs_rejected(tmp_path: Path) -> None:
    """Verifies that non-generator subroutines with unpaired outputs fail closed safely."""
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    f1.write_text("def run1():\n    a = 1\n    return a\n", encoding="utf-8")
    f2.write_text("def run2():\n    b = 1\n    c = 2\n    return b + c\n", encoding="utf-8")

    u1 = {
        "file": str(f1),
        "start": 2,
        "end": 2,
        "name": "run1:block",
        "kind": "compound_block",
        "outputs": ["a", "a"],
    }
    u2 = {
        "file": str(f2),
        "start": 2,
        "end": 3,
        "name": "run2:block",
        "kind": "compound_block",
        "outputs": ["b", "c"],
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generator_subroutine_enclosing_try_finally_read_rejected(tmp_path: Path) -> None:
    """Verifies that generator subroutine extraction is rejected when the unit is lexically
    inside a try block whose finally reads a needed output, guarding against abnormal exit loss."""
    src1 = (
        "def g1(items):\n"
        "    try:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            yield x\n"
        "            total += x\n"
        "    finally:\n"
        "        print(total)\n"
    )
    src2 = (
        "def g2(items):\n"
        "    try:\n"
        "        total = 0\n"
        "        for y in items:\n"
        "            yield y\n"
        "            total += y\n"
        "    finally:\n"
        "        print(total)\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 3, "end": 6, "name": "g1:block", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 3, "end": 6, "name": "g2:block", "kind": "compound_block"}

    # Should be rejected because total is needed and read in enclosing try cleanup
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generator_subroutine_enclosing_try_swallowing_handler_post_try_read_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that generator subroutine extraction is rejected when the unit is lexically
    inside a try block whose handler can fall through and a needed output is read post-try."""
    src1 = (
        "def g1(items):\n"
        "    try:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            yield x\n"
        "            total += x\n"
        "    except GeneratorExit:\n"
        "        pass\n"
        "    print(total)\n"
    )
    src2 = (
        "def g2(items):\n"
        "    try:\n"
        "        total = 0\n"
        "        for y in items:\n"
        "            yield y\n"
        "            total += y\n"
        "    except GeneratorExit:\n"
        "        pass\n"
        "    print(total)\n"
    )
    f1 = tmp_path / "g1_swallow.py"
    f2 = tmp_path / "g2_swallow.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 3, "end": 6, "name": "g1:block", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 3, "end": 6, "name": "g2:block", "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""

def test_generate_refactoring_patch_skips_mismatched_subroutine_kinds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that clone pairs with mismatched subroutine kinds immediately skip."""
    import logging  # pylint: disable=import-outside-toplevel

    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text("def fn():\n    x = 42\n    y = x + 1\n", encoding="utf-8")
    f2.write_text("def fn2():\n    x = 42\n    y = x + 1\n", encoding="utf-8")

    # u1 is a function (not a subroutine), u2 is a subroutine compound_block
    u1 = {"file": str(f1), "start": 1, "end": 3, "name": "fn", "kind": "function"}
    u2 = {"file": str(f2), "start": 2, "end": 3, "name": "fn2:sub", "kind": "compound_block"}

    with caplog.at_level(logging.DEBUG, logger="pydoppelgangerhunt.fixer.patch"):
        with mock.patch(
            "pydoppelgangerhunt.fixer.patch.synthesize_shared_helper_code"
        ) as mock_synth, mock.patch(
            "pydoppelgangerhunt.fixer.patch.evaluate_clone_pair_scopes"
        ) as mock_scope:
            patch = generate_refactoring_patch(
                [(1.0, u1, u2)],
                repo_root=str(tmp_path),
                replace_clones=True,
            )
            assert patch == ""
            # Early exit: hoisted check prevents running scope analysis or synthesis
            mock_synth.assert_not_called()
            mock_scope.assert_not_called()

    assert any(
        "mismatched subroutine kinds (is_sub1=False, is_sub2=True)" in r.message
        for r in caplog.records
    )

def test_replace_unit_and_refactor_missing_coordinates_raise_value_error() -> None:
    """Verifies that missing 'start' or 'end' line coordinates raise ValueError in
    replace_unit_in_source and refactor_module_units."""
    code = "x = 1\ny = 2\n"
    # Missing end
    with pytest.raises(ValueError, match="missing required 'end' line coordinate"):
        replace_unit_in_source(code, {"start": 1}, "x = 10\n")

    with pytest.raises(ValueError, match="missing required 'end' line coordinate"):
        refactor_module_units(code, [({"start": 1}, "x = 10\n")])

    # Missing start
    with pytest.raises(ValueError, match="missing required 'start' line coordinate"):
        replace_unit_in_source(code, {"end": 1}, "x = 10\n")

    with pytest.raises(ValueError, match="missing required 'start' line coordinate"):
        refactor_module_units(code, [({"end": 1}, "x = 10\n")])

def test_same_file_delegation_preserves_renamed_inputs_and_outputs_on_clone_2(
    tmp_path: Path,
) -> None:
    """Verifies that same-file subroutine delegation uses clone 2's renamed inputs
    and outputs instead of incorrectly copying clone 1's variable names."""
    f = tmp_path / "calc.py"
    code = (
        "def compute_alpha(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "    return total\n"
        "\n"
        "def compute_beta(data):\n"
        "    count = 0\n"
        "    for y in data:\n"
        "        count += y\n"
        "    return count\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 3,
        "end": 4,
        "name": "compute_alpha:for",
        "kind": "compound_block",
        "inputs": ["items", "total"],
        "outputs": ["total"],
    }
    u2 = {
        "file": str(f),
        "start": 9,
        "end": 10,
        "name": "compute_beta:for",
        "kind": "compound_block",
        "inputs": ["data", "count"],
        "outputs": ["count"],
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    # Clone 1 delegates using total and items
    assert "+    total = _shared" in patch
    # Clone 2 must delegate using count and data, NOT total and items
    assert "+    count = _shared" in patch

def test_pragma_no_refactor_harvested_but_skipped_in_patch(tmp_path: Path) -> None:
    """Verifies that units with # pragma: no-refactor are harvested for clone reporting
    but skipped during patch generation."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch
    from pydoppelgangerhunt.parser import harvest_file_units

    code1 = (
        "def compute_first(a, b):\n"
        "    # pragma: no-refactor\n"
        "    res = a * 2 + b * 3\n"
        "    return res\n"
    )
    code2 = (
        "def compute_second(a, b):\n"
        "    res = a * 2 + b * 3\n"
        "    return res\n"
    )
    p1 = tmp_path / "mod1.py"
    p2 = tmp_path / "mod2.py"
    p1.write_text(code1, encoding="utf-8")
    p2.write_text(code2, encoding="utf-8")

    units1 = harvest_file_units(str(p1), str(tmp_path), min_lines=3, min_tokens=5)
    assert len(units1) > 0, "Pragma: no-refactor must not suppress unit harvesting"

    units2 = harvest_file_units(str(p2), str(tmp_path), min_lines=3, min_tokens=5)
    assert len(units2) > 0

    u1 = dict(units1[0], file=str(p1.relative_to(tmp_path)))
    u2 = dict(units2[0], file=str(p2.relative_to(tmp_path)))
    clones = [(1.0, u1, u2)]

    patch = generate_refactoring_patch(clones, repo_root=str(tmp_path), replace_clones=True)
    assert "# Note: Clone pair" in patch
    assert "inline '# pragma: no-refactor' present" in patch

def test_complex_expr_whole_line_preserves_assignment_target_apply_and_run(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line complex_expr clones preserve assignment targets at call sites
    and execute correctly after applying the refactoring patch."""
    import importlib.util
    import subprocess
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    f = tmp_path / "calc_pair.py"
    code = (
        "def get_pair(x: int, y: int):\n"
        "    return x + 1, y * 2\n"
        "\n"
        "def worker1(x: int, y: int):\n"
        "    a, b = get_pair(x, y)\n"
        "    return a + b\n"
        "\n"
        "def worker2(x: int, y: int):\n"
        "    c, d = get_pair(x, y)\n"
        "    return c + d\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": "calc_pair.py",
        "start": 5,
        "end": 5,
        "name": "worker1:assign",
        "kind": "complex_expr",
    }
    u2 = {
        "file": "calc_pair.py",
        "start": 9,
        "end": 9,
        "name": "worker2:assign",
        "kind": "complex_expr",
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "(a, b) = _shared_" in patch
    assert "(c, d) = _shared_" in patch

    # Initialize temporary git repository and apply the patch
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tester"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True
    )
    proc = subprocess.run(
        ["git", "apply"],
        input=patch.encode("utf-8"),
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    assert proc.returncode == 0

    # Load and execute the refactored module to verify runtime semantics
    spec = importlib.util.spec_from_file_location("calc_pair_patched", str(f))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.worker1(10, 20) == 51
    assert mod.worker2(10, 20) == 51

def test_pragma_no_refactor_in_string_literal_does_not_suppress_refactoring(
    tmp_path: Path,
) -> None:
    """Verifies that '# pragma: no-refactor' appearing inside a string literal
    does not suppress clone refactoring."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    code1 = (
        "def emit_message1(val):\n"
        "    msg = '# pragma: no-refactor'\n"
        "    return f'{msg}: {val}'\n"
    )
    code2 = (
        "def emit_message2(val):\n"
        "    msg = '# pragma: no-refactor'\n"
        "    return f'{msg}: {val}'\n"
    )
    p1 = tmp_path / "msg1.py"
    p2 = tmp_path / "msg2.py"
    p1.write_text(code1, encoding="utf-8")
    p2.write_text(code2, encoding="utf-8")

    u1 = {"file": "msg1.py", "start": 1, "end": 3, "name": "emit_message1", "kind": "function"}
    u2 = {"file": "msg2.py", "start": 1, "end": 3, "name": "emit_message2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "pragma: no-refactor' present" not in patch
    assert "def _shared_emit_message1_emit_message2" in patch

def test_pragma_no_refactor_on_preceding_statement_does_not_suppress_unit(
    tmp_path: Path,
) -> None:
    """Verifies that a trailing pragma on a preceding statement does not suppress an
    unrelated subsequent unit."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    code1 = (
        "x = 1  # pragma: no-refactor\n"
        "def compute_logic1(a, b):\n"
        "    return a * 2 + b * 3\n"
    )
    code2 = (
        "x = 1  # pragma: no-refactor\n"
        "def compute_logic2(a, b):\n"
        "    return a * 2 + b * 3\n"
    )
    p1 = tmp_path / "logic1.py"
    p2 = tmp_path / "logic2.py"
    p1.write_text(code1, encoding="utf-8")
    p2.write_text(code2, encoding="utf-8")

    u1 = {"file": "logic1.py", "start": 2, "end": 3, "name": "compute_logic1", "kind": "function"}
    u2 = {"file": "logic2.py", "start": 2, "end": 3, "name": "compute_logic2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "pragma: no-refactor' present" not in patch
    assert "def _shared_compute_logic1_compute_logic2" in patch

def test_complex_expr_indented_multiline_assignment_dedents_cleanly(
    tmp_path: Path,
) -> None:
    """Verifies that indented multi-line assignments in complex_expr units
    are cleanly dedented without IndentationError when extracting targets."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    f = tmp_path / "mod_indent.py"
    code = (
        "def compute_total(a: int, b: int):\n"
        "    total = (\n"
        "        a + b\n"
        "    )\n"
        "    return total\n"
        "\n"
        "def compute_sum(x: int, y: int):\n"
        "    total = (\n"
        "        x + y\n"
        "    )\n"
        "    return total\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": "mod_indent.py",
        "start": 2,
        "end": 4,
        "name": "compute_total:assign",
        "kind": "complex_expr",
    }
    u2 = {
        "file": "mod_indent.py",
        "start": 8,
        "end": 10,
        "name": "compute_sum:assign",
        "kind": "complex_expr",
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "total = _shared_" in patch

def test_refactor_module_units_honors_offset_aliases() -> None:
    """Verifies that refactor_module_units respects start_col_offset and end_col_offset
    aliases when start_col and end_col are absent, preserving same-line neighbours."""
    from pydoppelgangerhunt.fixer import refactor_module_units

    src = "x = 1; y = 2; z = 3\n"
    # Unit replacing only 'y = 2' using alias keys
    unit = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col_offset": 7,
        "end_col_offset": 12,
    }
    result = refactor_module_units(src, [(unit, "y = 20")])
    assert result == "x = 1; y = 20; z = 3\n"

def test_generate_refactoring_patch_whole_line_return_complex_expr(tmp_path: Path) -> None:
    """Verifies that whole-line complex_expr for Return statements retains the return keyword
    in call site delegation rather than emitting a bare helper call."""
    code1 = (
        "def get_val_1(x: int, y: int) -> int:\n"
        "    return compute_val(x, y)\n"
    )
    code2 = (
        "def get_val_2(x: int, y: int) -> int:\n"
        "    return compute_val(x, y)\n"
    )
    f1 = tmp_path / "ret1.py"
    f2 = tmp_path / "ret2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    u1 = {
        "file": "ret1.py",
        "start": 2,
        "end": 2,
        "name": "get_val_1:ret",
        "kind": "complex_expr",
    }
    u2 = {
        "file": "ret2.py",
        "start": 2,
        "end": 2,
        "name": "get_val_2:ret",
        "kind": "complex_expr",
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "return _shared_get_val_1_get_val_2(compute_val, x, y)" in patch

def test_complex_expr_multiline_boundary_pragma_preserved(tmp_path: Path) -> None:
    """Verifies that boundary pragmas attached to whole-line multi-line complex_expr
    units are preserved via compute_unit_replacement_span."""
    f = tmp_path / "mod_pragma.py"
    code = (
        "def compute_total(a: int, b: int):\n"
        "    total: int = (  # type: ignore[assignment]\n"
        "        a + b\n"
        "    )\n"
        "    return total\n"
        "\n"
        "def compute_sum(x: int, y: int):\n"
        "    total: int = (  # type: ignore[assignment]\n"
        "        x + y\n"
        "    )\n"
        "    return total\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": "mod_pragma.py",
        "start": 2,
        "end": 4,
        "name": "compute_total:assign",
        "kind": "complex_expr",
    }
    u2 = {
        "file": "mod_pragma.py",
        "start": 8,
        "end": 10,
        "name": "compute_sum:assign",
        "kind": "complex_expr",
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "# type: ignore[assignment]" in patch

def test_build_unit_delegation_call_complex_expr_boundary_pragma_preserved() -> None:
    """Verifies that _build_unit_delegation_call preserves boundary pragmas on
    whole-line complex_expr units."""
    from pydoppelgangerhunt.fixer.patch import _build_unit_delegation_call

    code = (
        "def run(a: int, b: int):\n"
        "    res = (  # noqa: E501\n"
        "        a + b\n"
        "    )\n"
        "    return res\n"
    )
    lines = code.splitlines(keepends=True)
    unit = {
        "file": "test.py",
        "start": 2,
        "end": 4,
        "kind": "complex_expr",
    }
    call = _build_unit_delegation_call(
        unit,
        code,
        lines,
        effective_binding="function",
        helper_name="_shared_add",
        inputs=["a", "b"],
        outputs=["res"],
        scope={"kind": "function"},
    )
    assert "# noqa: E501" in call
    assert "res = _shared_add(a, b)" in call

def test_infer_helper_return_type_generator_send_type_strict_strategy() -> None:
    """Verifies that type_merge_strategy='strict' causes generator send type to
    fall back to 'object' instead of 'Any' when has_yield_assignment is True."""
    from pydoppelgangerhunt.fixer._synthesis_types import _infer_helper_return_type

    scope_with_assign = {
        "has_yield": True,
        "has_yield_assignment": True,
        "yield_expr_names": [("literal", "int")],
    }
    # Default fallback_any should return Any
    ret_default = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope_with_assign,
        {},
        {},
        type_merge_strategy="fallback_any",
    )
    assert "Generator[int, Any, None]" in ret_default

    # Strict strategy should return object
    ret_strict = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope_with_assign,
        {},
        {},
        type_merge_strategy="strict",
    )
    assert "Generator[int, object, None]" in ret_strict

    # Async generator with strict strategy
    ret_async_strict = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope_with_assign,
        {},
        {},
        type_merge_strategy="strict",
        is_async=True,
    )
    assert "AsyncGenerator[int, object]" in ret_async_strict

def test_whole_line_complex_expr_unmapped_statement_fails_closed(tmp_path: Path) -> None:
    """Verifies that whole-line complex_expr with unmapped statements (assert, del, raise)
    fails closed rather than emitting a bare helper call that drops statement semantics."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _build_unit_delegation_call,
        generate_refactoring_patch,
    )

    code = (
        "def check(x):\n"
        "    assert validate(x)\n"
        "    return x\n"
    )
    lines = code.splitlines(keepends=True)
    unit_assert = {
        "file": "test.py",
        "start": 2,
        "end": 2,
        "name": "check",
        "kind": "complex_expr",
    }
    # Direct check of _build_unit_delegation_call returning ""
    rep = _build_unit_delegation_call(
        unit_assert,
        code,
        lines,
        effective_binding="module",
        helper_name="_shared_validate",
        inputs=["x"],
        outputs=[],
        scope={"inputs": ["x"], "outputs": []},
    )
    assert rep == ""

    # Check that generate_refactoring_patch rolls back cleanly and does not emit bare call
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")
    u1 = {
        "file": "mod1.py",
        "start": 2,
        "end": 2,
        "name": "check",
        "kind": "complex_expr",
        "inputs": ["x"],
        "outputs": [],
    }
    u2 = {
        "file": "mod2.py",
        "start": 2,
        "end": 2,
        "name": "check",
        "kind": "complex_expr",
        "inputs": ["x"],
        "outputs": [],
    }
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    # The patch must NOT replace 'assert validate(x)' with a bare '_shared_validate(x)'
    assert "-    assert validate(x)" not in patch

def test_complex_expr_augassign_delegation_prunes_unused_target(tmp_path: Path) -> None:
    """Verifies that whole-line complex_expr with AugAssign generates a delegation call
    that prunes the unreferenced target from call arguments."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _build_unit_delegation_call,
        generate_refactoring_patch,
    )

    code1 = (
        "def run_one(total, a, b):\n"
        "    total += a + b\n"
    )
    code2 = (
        "def run_two(counter, a, b):\n"
        "    counter += a + b\n"
    )
    lines1 = code1.splitlines(keepends=True)
    u1 = {
        "file": "mod1.py",
        "start": 2,
        "end": 2,
        "name": "run_one",
        "kind": "complex_expr",
        "inputs": ["total", "a", "b"],
        "outputs": [],
    }
    # Direct test of _build_unit_delegation_call:
    rep = _build_unit_delegation_call(
        u1,
        code1,
        lines1,
        effective_binding="module",
        helper_name="_shared_add",
        inputs=["a", "b"],
        outputs=[],
        scope={"inputs": ["total", "a", "b"], "outputs": []},
        target_inputs=["total", "a", "b"],
    )
    assert "total += _shared_add(a, b)" in rep
    assert "_shared_add(total" not in rep

    # Full patch generation test
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    u2 = {
        "file": "mod2.py",
        "start": 2,
        "end": 2,
        "name": "run_two",
        "kind": "complex_expr",
        "inputs": ["counter", "a", "b"],
        "outputs": [],
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "+    total += _shared_run_one_run_two(a, b)" in patch
    assert "+    counter += _shared_run_one_run_two(a, b)" in patch

def test_complex_expr_augassign_target_pruning_different_param_orders(tmp_path: Path) -> None:
    """Verifies that whole-line complex_expr AugAssign units with different parameter
    orderings do not swap arguments at call sites."""
    code1 = (
        "def run_one(total, a, b):\n"
        "    total += a - b\n"
        "    return total\n"
    )
    code2 = (
        "def run_two(a, b, counter):\n"
        "    counter += a - b\n"
        "    return counter\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "mod1.py",
        "start": 2,
        "end": 2,
        "name": "run_one",
        "kind": "complex_expr",
        "inputs": ["total", "a", "b"],
        "outputs": [],
    }
    u2 = {
        "file": "mod2.py",
        "start": 2,
        "end": 2,
        "name": "run_two",
        "kind": "complex_expr",
        "inputs": ["a", "b", "counter"],
        "outputs": [],
    }
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "+    counter += _shared_run_one_run_two(a, b)" in patch
    assert "(b, a)" not in patch

def test_complex_expr_augassign_asymmetric_rhs_not_pruned(tmp_path: Path) -> None:
    """Verifies that if one unit references its AugAssign target on RHS while the other
    does not, neither has its parameter pruned to prevent arity mismatch."""
    code1 = (
        "def f1(total, a, b):\n"
        "    total += a + b\n"
    )
    code2 = (
        "def f2(counter, a, b):\n"
        "    counter += counter + a + b\n"
    )
    f1 = tmp_path / "m1.py"
    f2 = tmp_path / "m2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "m1.py", "start": 2, "end": 2, "name": "f1", "kind": "complex_expr"}
    u2 = {"file": "m2.py", "start": 2, "end": 2, "name": "f2", "kind": "complex_expr"}
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    if helper:
        assert "total: Any" in helper or "counter: Any" in helper

def test_build_unit_delegation_call_asymmetric_rhs_equal_arity() -> None:
    """Verifies that _build_unit_delegation_call does not prune target when
    len(effective_targets) == len(inputs), preventing arity mismatch."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access
        _build_unit_delegation_call,
    )

    unit = {
        "kind": "complex_expr",
        "start": 1,
        "end": 1,
    }
    orig_lines = ["total += a + a\n"]
    call_str = _build_unit_delegation_call(
        unit,
        "total += a + a\n",
        orig_lines,
        effective_binding="module",
        helper_name="_shared_helper",
        inputs=["total", "a"],
        outputs=[],
        scope={"inputs": ["total", "a"]},
        target_inputs=["total", "a"],
        await_prefix="",
    )
    assert "_shared_helper(total, a)" in call_str

def test_subline_complex_expr_not_treated_as_augassign_target(tmp_path: Path) -> None:
    """Verifies that subline complex_expr units with column offsets do not extract line's target."""
    src = (
        "def run(a, b):\n"
        "    total += compute(a, b)\n"
        "    return total\n"
    )
    f = tmp_path / "subline_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "file": "subline_mod.py",
        "start": 2,
        "end": 2,
        "start_col": 13,
        "end_col": 26,
        "name": "u1",
        "kind": "complex_expr",
    }
    u2 = dict(u1)
    u2["name"] = "u2"

    helper_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "total" not in helper_code

def test_augassign_target_pruning_same_name_across_clones(tmp_path: Path) -> None:
    """Verifies that when both clones use the same unreferenced AugAssign target name,
    the target is pruned from both inputs and inputs2 without desynchronization."""
    src1 = (
        "def func1(total: int, a: int, b: int) -> int:\n"
        "    total += a - b\n"
        "    return total\n"
    )
    src2 = (
        "def func2(total: int, a: int, b: int) -> int:\n"
        "    total += a - b\n"
        "    return total\n"
    )
    f1 = tmp_path / "same_aug1.py"
    f2 = tmp_path / "same_aug2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "same_aug1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "same_aug2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    helper_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in helper_code
    assert "total" not in helper_code

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in patch
    assert "+    total += _shared_u1_u2(a, b)" in patch

def test_whole_line_complex_expr_semicolon_multiple_statements_fails_closed(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line complex_expr units containing semicolon-separated statements
    fail closed rather than silently dropping trailing statements."""
    src = (
        "def run(a: int, b: int) -> int:\n"
        "    x = a + b; side_effect()\n"
        "    return x\n"
    )
    f1 = tmp_path / "semi1.py"
    f2 = tmp_path / "semi2.py"
    f1.write_text(src, encoding="utf-8")
    f2.write_text(src, encoding="utf-8")

    u1 = {"file": "semi1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "semi2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    assert synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path)) == ""
    assert (
        generate_refactoring_patch(
            [(0.95, u1, u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
        == ""
    )

def test_whole_line_comprehension_preserves_assignment_target_and_indentation(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line comprehension units preserve their assignment target,
    leading indentation, and trailing newline when replaced at call sites."""
    src1 = (
        "def f1(items: list[int]) -> list[int]:\n"
        "    res = [x * 2 for x in items if x > 0]\n"
        "    return res\n"
    )
    src2 = (
        "def f2(items: list[int]) -> list[int]:\n"
        "    out = [x * 2 for x in items if x > 0]\n"
        "    return out\n"
    )
    f1 = tmp_path / "comp_whole1.py"
    f2 = tmp_path / "comp_whole2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {
        "file": "comp_whole1.py",
        "start": 2,
        "end": 2,
        "name": "u1",
        "kind": "comprehension",
    }
    u2 = {
        "file": "comp_whole2.py",
        "start": 2,
        "end": 2,
        "name": "u2",
        "kind": "comprehension",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(items: Any) -> Any:" in patch
    assert "+    return [x * 2 for x in items if x > 0]" in patch
    assert "+    res = _shared_u1_u2(items)" in patch
    assert "+    out = _shared_u1_u2(items)" in patch

def test_whole_line_comprehension_and_complex_expr_subscript_attribute_lhs_pruned(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line comprehension and complex_expr units with subscript or
    attribute assignment targets prune LHS-only load identifiers and receiver attributes."""
    src1 = (
        "def update_slot(arr: list[list[int]], i: int, items: list[int]) -> None:\n"
        "    arr[i] = [x * 2 for x in items if x > 0]\n"
    )
    src2 = (
        "def compute_list(items: list[int]) -> list[int]:\n"
        "    res = [x * 2 for x in items if x > 0]\n"
        "    return res\n"
    )
    f1 = tmp_path / "lhs_sub1.py"
    f2 = tmp_path / "lhs_sub2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "lhs_sub1.py", "start": 2, "end": 2, "name": "u1", "kind": "comprehension"}
    u2 = {"file": "lhs_sub2.py", "start": 2, "end": 2, "name": "u2", "kind": "comprehension"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(items: Any) -> Any:" in patch
    assert "+    arr[i] = _shared_u1_u2(items)" in patch
    assert "+    res = _shared_u1_u2(items)" in patch

    # Also test self.cache[k] = compute(a, b) paired with res = compute(a, b)
    src_attr1 = (
        "class Worker:\n"
        "    def store(self, k: str, a: int, b: int) -> None:\n"
        "        self.cache[k] = (a + b) * (a - b)\n"
    )
    src_attr2 = (
        "class Worker2:\n"
        "    def calc(self, a: int, b: int) -> int:\n"
        "        res = (a + b) * (a - b)\n"
        "        return res\n"
    )
    fa1 = tmp_path / "lhs_attr1.py"
    fa2 = tmp_path / "lhs_attr2.py"
    fa1.write_text(src_attr1, encoding="utf-8")
    fa2.write_text(src_attr2, encoding="utf-8")

    ua1 = {"file": "lhs_attr1.py", "start": 3, "end": 3, "name": "u1", "kind": "complex_expr"}
    ua2 = {"file": "lhs_attr2.py", "start": 3, "end": 3, "name": "u2", "kind": "complex_expr"}

    patch_attr = generate_refactoring_patch(
        [(0.95, ua1, ua2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in patch_attr
    assert "+        self.cache[k] = _shared_u1_u2(a, b)" in patch_attr
    assert "+        res = _shared_u1_u2(a, b)" in patch_attr

def test_unit_has_no_refactor_pragma_fallback_inline_and_multiline_leading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that _unit_has_no_refactor_pragma fallback detects inline trailing
    # pragma: no-refactor comments and multi-line contiguous leading comment blocks."""
    from pydoppelgangerhunt.fixer import patch as patch_mod
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access
        _unit_has_no_refactor_pragma,
    )

    def _raise_runtime_error(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated tokenizer failure")

    monkeypatch.setattr(patch_mod, "extract_unit_comments_and_pragmas", _raise_runtime_error)

    # 1. Inline trailing comment on header line
    lines_inline = [
        "def foo():  # pragma: no-refactor\n",
        "    return 42\n",
    ]
    u_inline = {"start": 1, "end": 2}
    assert _unit_has_no_refactor_pragma(u_inline, lines_inline) is True

    # 2. Multi-line contiguous leading comment block above unit
    lines_multiline = [
        "# pragma: no-refactor\n",
        "# Intentional duplicate implementation\n",
        "def bar():\n",
        "    return 99\n",
    ]
    u_multiline = {"start": 3, "end": 4}
    assert _unit_has_no_refactor_pragma(u_multiline, lines_multiline) is True

def test_whole_line_complex_expr_lhs_receiver_pruned_across_method_and_function(
    tmp_path: Path,
) -> None:
    """Verifies that when self appears only on the LHS of an instance method's whole-line
    complex_expr, receiver binding flags are recomputed so it pairs cleanly with a
    top-level function or staticmethod."""
    src_method = (
        "class Worker:\n"
        "    def store(self, k: str, a: int, b: int) -> None:\n"
        "        self.cache[k] = (a + b) * (a - b)\n"
    )
    src_func = (
        "def standalone_calc(a: int, b: int) -> int:\n"
        "    res = (a + b) * (a - b)\n"
        "    return res\n"
    )
    f1 = tmp_path / "rec_method.py"
    f2 = tmp_path / "rec_func.py"
    f1.write_text(src_method, encoding="utf-8")
    f2.write_text(src_func, encoding="utf-8")

    u1 = {"file": "rec_method.py", "start": 3, "end": 3, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "rec_func.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in helper

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in patch
    assert "+        self.cache[k] = _shared_u1_u2(a, b)" in patch
    assert "+    res = _shared_u1_u2(a, b)" in patch

def test_asymmetric_augassign_vs_assign_whole_line_complex_expr_refactored(
    tmp_path: Path,
) -> None:
    """Verifies that a whole-line complex_expr with AugAssign (total += a + b) paired
    with a regular Assign (res = a + b) prunes the unreferenced AugAssign target and
    refactors both call sites."""
    src1 = (
        "def accumulate(total: int, a: int, b: int) -> int:\n"
        "    total += (a + b) * 2\n"
        "    return total\n"
    )
    src2 = (
        "def compute_once(a: int, b: int) -> int:\n"
        "    res = (a + b) * 2\n"
        "    return res\n"
    )
    f1 = tmp_path / "asym_aug1.py"
    f2 = tmp_path / "asym_aug2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "asym_aug1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "asym_aug2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in helper

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(a: Any, b: Any) -> Any:" in patch
    assert "+    total += _shared_u1_u2(a, b)" in patch
    assert "+    res = _shared_u1_u2(a, b)" in patch

def test_custom_classmethod_receiver_preserved_after_lhs_pruning() -> None:
    """Verifies that _prune_unreferenced_lhs_scope preserves class binding for a
    @classmethod with a custom receiver name (e.g., klass) when LHS loads are pruned."""
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access
        _prune_unreferenced_lhs_scope,
    )

    lines = ["registry[idx] = klass.create(a, b)"]
    unit = {
        "name": "build_entry",
        "kind": "complex_expr",
        "start": 1,
        "end": 1,
        "receiver_param": "klass",
        "receiver_kind": "class",
        "source_text": lines[0],
    }
    scope = inspect_single_unit_scope(unit)
    _prune_unreferenced_lhs_scope(unit, scope, lines)

    assert "registry" not in scope["inputs"]
    assert "idx" not in scope["inputs"]
    assert "klass" in scope["inputs"]
    assert scope["has_class_binding"] is True
    assert scope["has_instance_binding"] is False
    assert scope["has_receiver_access"] is True
    assert scope["binding_kind"] == "class"

def test_whole_line_comprehension_distinct_lhs_attrs_with_shared_rhs_attr(
    tmp_path: Path,
) -> None:
    """Verifies that two methods assigning to distinct LHS attributes (self.res1 vs self.res2)
    while reading a shared attribute on the RHS (self.factor) prune the LHS attribute writes
    and refactor cleanly."""
    src = (
        "class Processor:\n"
        "    def __init__(self, factor: int) -> None:\n"
        "        self.factor = factor\n"
        "        self.res1 = []\n"
        "        self.res2 = []\n"
        "\n"
        "    def run_first(self, items: list) -> None:\n"
        "        self.res1 = [x * self.factor + 1 for x in items if x > 0]\n"
        "\n"
        "    def run_second(self, items: list) -> None:\n"
        "        self.res2 = [x * self.factor + 1 for x in items if x > 0]\n"
    )
    f = tmp_path / "proc.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "file": "proc.py",
        "start": 8,
        "end": 8,
        "name": "Processor.run_first:<comp>",
        "kind": "comprehension",
        "enclosing_class": "Processor",
        "receiver_param": "self",
    }
    u2 = {
        "file": "proc.py",
        "start": 11,
        "end": 11,
        "name": "Processor.run_second:<comp>",
        "kind": "comprehension",
        "enclosing_class": "Processor",
        "receiver_param": "self",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_" in helper
    assert "return [x * self.factor + 1 for x in items if x > 0]" in helper

    patch = generate_refactoring_patch(
        [(0.98, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+        self.res1 = _shared_run_first_run_second(self, items)" in patch
    assert "+        self.res2 = _shared_run_first_run_second(self, items)" in patch

def test_async_function_and_async_method_kinds_whole_body_delegation(
    tmp_path: Path,
) -> None:
    """Verifies that units with kind='async_function' or kind='async_method' strip their
    async def header during body extraction, synthesize a valid async def helper, and emit
    whole-method delegation calls."""
    src = (
        "async def fetch_alpha(client, key: str) -> int:\n"
        "    raw = await client.get(key)\n"
        "    val = int(raw.strip())\n"
        "    return val + 1\n"
        "\n"
        "async def fetch_beta(client, key: str) -> int:\n"
        "    raw = await client.get(key)\n"
        "    val = int(raw.strip())\n"
        "    return val + 1\n"
    )
    f = tmp_path / "async_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "file": "async_mod.py",
        "start": 1,
        "end": 4,
        "name": "fetch_alpha",
        "kind": "async_function",
    }
    u2 = {
        "file": "async_mod.py",
        "start": 6,
        "end": 9,
        "name": "fetch_beta",
        "kind": "async_function",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper.startswith("async def _shared_fetch_alpha_fetch_beta(")
    assert helper.count("async def ") == 1

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+async def _shared_fetch_alpha_fetch_beta(" in patch
    assert "+    return await _shared_fetch_alpha_fetch_beta(client, key)" in patch

def test_lhs_and_augassign_pruning_ignores_bound_comprehension_target_on_rhs(
    tmp_path: Path,
) -> None:
    """Verifies that _node_references_name ignores bound comprehension targets on the RHS
    so LHS subscript indices (arr[x] = [x * 2 for x in items]) and AugAssign targets
    (total += sum(total for total in items)) are properly pruned."""
    src1 = (
        "def fill_slot(arr: list, x: int, items: list) -> None:\n"
        "    arr[x] = [x * 2 + 1 for x in items if x > 0]\n"
    )
    src2 = (
        "def compute_list(items: list) -> list:\n"
        "    res = [x * 2 + 1 for x in items if x > 0]\n"
        "    return res\n"
    )
    f1 = tmp_path / "comp_lhs1.py"
    f2 = tmp_path / "comp_lhs2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "comp_lhs1.py", "start": 2, "end": 2, "name": "u1", "kind": "comprehension"}
    u2 = {"file": "comp_lhs2.py", "start": 2, "end": 2, "name": "u2", "kind": "comprehension"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(items: Any) -> Any:" in helper

    patch = generate_refactoring_patch(
        [(0.98, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+    arr[x] = _shared_u1_u2(items)" in patch
    assert "+    res = _shared_u1_u2(items)" in patch

    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access
        _extract_augassign_unreferenced_target,
    )

    assert (
        _extract_augassign_unreferenced_target(["total += sum(total for total in items)"])
        == "total"
    )

def test_top_level_async_comprehension_and_nested_class_lambda_async_scoping() -> None:
    """Verifies that top-level async comprehensions set unit['is_async'] = True in
    _populate_unit_receiver_metadata and that _ScopeHierarchyVisitor resets async_stack
    to False inside ClassDef and Lambda bodies nested inside async def."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=protected-access
        _populate_unit_receiver_metadata,
    )
    from pydoppelgangerhunt.parser import (  # pylint: disable=protected-access
        _ScopeHierarchyVisitor,
    )

    top_src = "res = [await fetch(x) for x in items]\n"
    u_top: dict[str, Any] = {
        "start": 1,
        "end": 1,
        "kind": "comprehension",
        "is_subroutine": True,
    }
    _populate_unit_receiver_metadata(u_top, source_text=top_src)
    assert u_top["is_async"] is True

    nested_src = (
        "async def outer(items):\n"
        "    class Inner:\n"
        "        vals = [x for x in range(5)]\n"
        "    fn = lambda seq=[await f(i) for i in items]: [y for y in seq]\n"
    )
    tree = ast.parse(nested_src)
    scoper = _ScopeHierarchyVisitor()
    scoper.visit(tree)

    outer_fn = tree.body[0]
    inner_cls = outer_fn.body[0]  # type: ignore[attr-defined]
    cls_comp = inner_cls.body[0].value
    lam_node = outer_fn.body[1].value  # type: ignore[attr-defined]
    lam_default_comp = lam_node.args.defaults[0]
    lam_body_comp = lam_node.body

    assert scoper.func_is_async[id(cls_comp)] is False
    assert scoper.func_is_async[id(lam_default_comp)] is True
    assert scoper.func_is_async[id(lam_body_comp)] is False

def test_kindless_subline_expr_slicing_and_mixed_assign_return_alignment(
    tmp_path: Path,
) -> None:
    """Verifies that synthesize_shared_helper_code aligns expression kinds before slicing
    sub-line token columns and unwraps ast.Return alongside ast.Assign in _strip_stmt_target_lines."""
    # 1. Kind-less sub-line unit with start_col/end_col paired with explicit complex_expr unit
    sub_src1 = "def f1(a, b, c):\n    out = (a + b) * (b + c)\n    return out\n"
    sub_src2 = "def f2(a, b, c):\n    val = (a + b) * (b + c)\n    return val\n"
    sf1 = tmp_path / "sub1.py"
    sf2 = tmp_path / "sub2.py"
    sf1.write_text(sub_src1, encoding="utf-8")
    sf2.write_text(sub_src2, encoding="utf-8")
    u_sub1 = {
        "file": "sub1.py",
        "start": 2,
        "end": 2,
        "start_col": 10,
        "end_col": 27,
        "name": "s1",
    }
    u_sub2 = {
        "file": "sub2.py",
        "start": 2,
        "end": 2,
        "start_col": 10,
        "end_col": 27,
        "name": "s2",
        "kind": "complex_expr",
    }
    sub_helper = synthesize_shared_helper_code(u_sub1, u_sub2, repo_root=str(tmp_path))
    assert "return (a + b) * (b + c)" in sub_helper
    assert "out =" not in sub_helper

    # 2. Mixed whole-line Assign (res = <expr>) and Return (return <expr>) multiline complex_expr
    mix_src1 = (
        "def m1(a, b, c, d):\n"
        "    res = (\n"
        "        (a + b)\n"
        "        * (c + d)\n"
        "    )\n"
        "    return res\n"
    )
    mix_src2 = (
        "def m2(a, b, c, d):\n"
        "    return (\n"
        "        (a + b)\n"
        "        * (c + d)\n"
        "    )\n"
    )
    mf1 = tmp_path / "mix1.py"
    mf2 = tmp_path / "mix2.py"
    mf1.write_text(mix_src1, encoding="utf-8")
    mf2.write_text(mix_src2, encoding="utf-8")
    u_mix1 = {
        "file": "mix1.py",
        "start": 2,
        "end": 5,
        "name": "m1_expr",
        "kind": "complex_expr",
    }
    u_mix2 = {
        "file": "mix2.py",
        "start": 2,
        "end": 5,
        "name": "m2_expr",
        "kind": "complex_expr",
    }
    mix_helper = synthesize_shared_helper_code(u_mix1, u_mix2, repo_root=str(tmp_path))
    assert "return (a + b) * (c + d)" in mix_helper
    mix_patch = generate_refactoring_patch(
        [(0.95, u_mix1, u_mix2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "+    res = _shared_m1_expr_m2_expr(a, b, c, d)" in mix_patch
    assert "+    return _shared_m1_expr_m2_expr(a, b, c, d)" in mix_patch

def test_same_class_whole_line_expr_lhs_only_self_runtime_safety(
    tmp_path: Path,
) -> None:
    """Verifies that same-class whole-line comprehension/complex_expr clones where self
    is only referenced on the LHS do not emit mismatched receiver signatures."""
    cls_src = (
        "class Processor:\n"
        "    def m1(self, items):\n"
        "        self.res1 = [x * 2 for x in items]\n"
        "        return self.res1\n"
        "\n"
        "    def m2(self, items):\n"
        "        self.res2 = [x * 2 for x in items]\n"
        "        return self.res2\n"
    )
    f_cls = tmp_path / "proc.py"
    f_cls.write_text(cls_src, encoding="utf-8")
    u1_c = {"file": "proc.py", "start": 3, "end": 3, "name": "m1_c"}
    u2_c = {
        "file": "proc.py",
        "start": 7,
        "end": 7,
        "name": "m2_c",
        "kind": "comprehension",
    }
    patch_str = generate_refactoring_patch(
        [(0.99, u1_c, u2_c)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "@staticmethod" not in patch_str
    assert "+def _shared_m1_c_m2_c(items: Any) -> Any:" in patch_str
    assert "+        self.res1 = _shared_m1_c_m2_c(items)" in patch_str
    assert "+        self.res2 = _shared_m1_c_m2_c(items)" in patch_str

def test_whole_line_complex_expr_leading_comment_and_col0_multiline_string(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line complex_expr units starting with a comment and containing
    a column-0 multiline string literal parse and refactor cleanly."""
    src = (
        "def f1(cache, k, suffix):\n"
        "    # leading comment 1\n"
        "    cache[k] = (\n"
        '        """header\n'
        "col0_line\n"
        '""" + suffix\n'
        "    )\n"
        "    return cache\n"
        "\n"
        "def f2(store, key, suffix):\n"
        "    # leading comment 2\n"
        "    store[key] = (\n"
        '        """header\n'
        "col0_line\n"
        '""" + suffix\n'
        "    )\n"
        "    return store\n"
    )
    p = tmp_path / "col0_expr.py"
    p.write_text(src, encoding="utf-8")
    u1 = {"file": "col0_expr.py", "start": 2, "end": 7, "name": "f1_expr", "kind": "complex_expr"}
    u2 = {"file": "col0_expr.py", "start": 11, "end": 16, "name": "f2_expr", "kind": "complex_expr"}
    patch_str = generate_refactoring_patch(
        [(0.99, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "+def _shared_f1_expr_f2_expr(suffix: Any) -> Any:" in patch_str
    assert "+    cache[k] = _shared_f1_expr_f2_expr(suffix)" in patch_str
    assert "+    store[key] = _shared_f1_expr_f2_expr(suffix)" in patch_str

def test_out_of_bounds_pragma_and_col0_multiline_string_subroutine_synthesis(
    tmp_path: Path,
) -> None:
    """Verifies that _unit_has_no_refactor_pragma handles out-of-bounds u_start without
    IndexError and that subroutine synthesis with column-0 multiline strings preserves
    exact string literal content alongside synthesized preambles and returns."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _unit_has_no_refactor_pragma,
    )

    pragma_lines = ["# pragma: no-refactor\n", "x = 1\n"]
    assert (
        _unit_has_no_refactor_pragma(
            {"start": 50, "end": 55}, pragma_lines, "".join(pragma_lines)
        )
        is False
    )
    dec_pragma_lines = [
        "# pragma: no-refactor\n",
        "@my_decorator(\n",
        "    arg=True,\n",
        ")\n",
        "def compute(x):\n",
        "    return x + 1\n",
    ]
    assert (
        _unit_has_no_refactor_pragma(
            {"start": 5, "end": 6}, dec_pragma_lines, "".join(dec_pragma_lines)
        )
        is True
    )

    src = (
        "class Formatter:\n"
        "    def build_a(self, flag: bool, suffix: str) -> str:\n"
        "        if flag:\n"
        "            tag = 'ON'\n"
        '        banner = """header\n'
        'col0_body_line"""\n'
        "        res = f'{banner}:{tag}:{suffix}'\n"
        "        return res\n"
        "\n"
        "    def build_b(self, flag: bool, suffix: str) -> str:\n"
        "        if flag:\n"
        "            tag = 'ON'\n"
        '        banner = """header\n'
        'col0_body_line"""\n'
        "        res = f'{banner}:{tag}:{suffix}'\n"
        "        return res\n"
    )
    f_sub = tmp_path / "fmt.py"
    f_sub.write_text(src, encoding="utf-8")
    u1 = {"file": "fmt.py", "start": 3, "end": 7, "name": "build_a_sub", "kind": "subroutine"}
    u2 = {"file": "fmt.py", "start": 11, "end": 15, "name": "build_b_sub", "kind": "subroutine"}
    helper_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "\ncol0_body_line\"\"\"\n" in helper_code
    assert "    col0_body_line" not in helper_code
    assert "        tag = None\n" in helper_code
    assert "        return tag, banner, res\n" in helper_code
    patch_str = generate_refactoring_patch(
        [(0.99, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "    col0_body_line" not in patch_str
    assert "+        return tag, banner, res\n" in patch_str

@pytest.mark.parametrize(
    ("hazard_case", "expected_reason"),
    [
        pytest.param("invalid_cols", "invalid unit columns", id="invalid_unit_columns"),
        pytest.param("stale_source", "stale source code", id="stale_source_code"),
        pytest.param("pathological_ast", "pathological ast", id="pathological_ast"),
    ],
)
def test_generate_refactoring_patch_prioritizes_blocking_hazards_on_complex_expr(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    hazard_case: str,
    expected_reason: str,
) -> None:
    """Verifies complex_expr pairs report blocking source/AST skip reasons rather than expression pruning errors."""
    expr_file = tmp_path / "exprs.py"
    if hazard_case == "pathological_ast":
        deep_expr = "+" * 600 + "x"
        src = (
            f"def f1(x):\n    val = {deep_expr}\n    return val\n\n"
            f"def f2(x):\n    val = {deep_expr}\n    return val\n"
        )
        line1 = f"    val = {deep_expr}"
        expr_text = deep_expr
    else:
        src = (
            "def f1(x):\n    val = (x + 1)\n    return val\n\n"
            "def f2(x):\n    val = (x + 1)\n    return val\n"
        )
        line1 = "    val = (x + 1)"
        expr_text = "x + 1"
    expr_file.write_text(src, encoding="utf-8")
    col_start = line1.index(expr_text)
    col_end = col_start + len(expr_text)

    u1: Dict[str, Any] = {
        "file": str(expr_file),
        "name": "f1:expr",
        "start": 2,
        "end": 2,
        "start_col": col_start,
        "end_col": col_end,
        "kind": "complex_expr",
    }
    u2: Dict[str, Any] = {
        "file": str(expr_file),
        "name": "f2:expr",
        "start": 6,
        "end": 6,
        "start_col": col_start,
        "end_col": col_end,
        "kind": "complex_expr",
    }
    if hazard_case == "invalid_cols":
        u1["start_col"] = 500
        u1["end_col"] = 510
    elif hazard_case == "stale_source":
        u1["source_code"] = "stale_expr * 999"

    with caplog.at_level(logging.INFO, logger="pydoppelgangerhunt"):
        patch_out = generate_refactoring_patch(
            [(0.99, u1, u2)], repo_root=str(tmp_path), replace_clones=True
        )
    assert patch_out == ""
    assert expected_reason in caplog.text
    assert "incompatible inputs or outputs" not in caplog.text

def test_generate_refactoring_patch_expression_unit_avoids_redundant_disk_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies generate_refactoring_patch infers expression column bounds from loaded plan lines without re-reading disk."""
    import builtins as py_builtins  # pylint: disable=import-outside-toplevel
    import pydoppelgangerhunt.fixer.patch as patch_mod  # pylint: disable=import-outside-toplevel

    src = (
        "def f1(x):\n    val = (x + 1) * 2\n    return val\n\n"
        "def f2(x):\n    val = (x + 1) * 2\n    return val\n"
    )
    expr_file = tmp_path / "expr_mod.py"
    expr_file.write_text(src, encoding="utf-8")
    u1 = {
        "file": str(expr_file),
        "name": "f1:expr",
        "kind": "complex_expr",
        "start": 2,
        "end": 2,
        "source_code": "(x + 1) * 2",
    }
    u2 = {
        "file": str(expr_file),
        "name": "f2:expr",
        "kind": "complex_expr",
        "start": 6,
        "end": 6,
        "source_code": "(x + 1) * 2",
    }

    real_open = py_builtins.open
    real_read_text = Path.read_text
    real_resolve_cols = patch_mod.resolve_unit_inferred_col_bounds
    open_count = 0
    read_text_count = 0
    col_bounds_used_preloaded: List[bool] = []
    target_resolved = expr_file.resolve()

    def _counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal open_count
        try:
            if Path(file).resolve() == target_resolved:
                open_count += 1
        except (OSError, TypeError, ValueError):
            pass
        return real_open(file, *args, **kwargs)

    def _counting_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal read_text_count
        try:
            if self.resolve() == target_resolved:
                read_text_count += 1
        except (OSError, TypeError, ValueError):
            pass
        return real_read_text(self, *args, **kwargs)

    def _spy_resolve_cols(*args: Any, **kwargs: Any) -> Any:
        col_bounds_used_preloaded.append(
            kwargs.get("unit_code_lines") is not None
            and kwargs.get("is_placeholder_origin") is not None
        )
        return real_resolve_cols(*args, **kwargs)

    monkeypatch.setattr(py_builtins, "open", _counting_open)
    monkeypatch.setattr(Path, "read_text", _counting_read_text)
    monkeypatch.setattr(patch_mod, "resolve_unit_inferred_col_bounds", _spy_resolve_cols)

    patch_out = generate_refactoring_patch(
        [(0.99, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_" in patch_out
    assert read_text_count == 1
    assert open_count == 0
    assert col_bounds_used_preloaded == [True, True]


def test_build_whole_method_delegation_handles_multibyte_utf8_single_line_def() -> None:
    """Verifies single-line function delegation converts UTF-8 byte col_offset to character index."""
    from pydoppelgangerhunt.fixer._patch_delegation import (  # pylint: disable=import-outside-toplevel
        _build_whole_method_delegation,
    )

    src = 'def café(x="é"): return x + "!"\n'
    unit = {"file": "m.py", "name": "café", "start": 1, "end": 1, "kind": "function"}
    delegated = _build_whole_method_delegation(
        source_text=src,
        unit=unit,
        call_prefix="",
        helper_name="_shared_cafe",
        args_str="x",
        await_prefix="",
        has_return=True,
        is_async=False,
        has_yield=False,
    )
    first_line = delegated.splitlines()[0]
    assert first_line == 'def café(x="é"):'
    compile(delegated, "<utf8_delegation>", "exec")
