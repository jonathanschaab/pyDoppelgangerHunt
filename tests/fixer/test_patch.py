"""Unit tests for fixer refactoring patch generation, delegation calls, overlap collision, and git apply."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    check_units_overlap,
    extract_unit_source_code,
    generate_refactoring_patch,
    refactor_module_units,
    replace_unit_in_source,
    scan_target,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _build_whole_method_delegation,
    patch as patch_mod,
)
from pydoppelgangerhunt.parser import harvest_file_units


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

def test_inter_block_overlap_collision_detection_and_reverse_offset_refactoring(tmp_path: Path) -> None:
    """Verifies overlap collision detection, maximal subset filtering, and reverse-order refactoring."""
    import pytest  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        check_units_overlap,
        filter_overlapping_clone_units,
        refactor_module_units,
    )

    u_outer = {"file": "mod.py", "start": 10, "end": 30, "name": "outer_block"}
    u_inner = {"file": "mod.py", "start": 15, "end": 25, "name": "inner_block"}
    u_disjoint = {"file": "mod.py", "start": 40, "end": 50, "name": "disjoint_block"}
    u_other_file = {"file": "other.py", "start": 10, "end": 30, "name": "other_outer"}

    # 1. Test check_units_overlap
    assert check_units_overlap(u_outer, u_inner) is True
    assert check_units_overlap(u_inner, u_outer) is True
    assert check_units_overlap(u_outer, u_disjoint) is False
    assert check_units_overlap(u_outer, u_other_file) is False

    # 2. Test filter_overlapping_clone_units
    # Should prioritize u_outer (21 lines) over u_inner (11 lines), while keeping u_disjoint
    filtered = filter_overlapping_clone_units([u_inner, u_outer, u_disjoint])
    assert len(filtered) == 2
    names = [u["name"] for u in filtered]
    assert "outer_block" in names
    assert "disjoint_block" in names
    assert "inner_block" not in names

    # Empty list edge case
    assert filter_overlapping_clone_units([]) == []

    # 3. Test refactor_module_units overlap collision exception
    source_sample = (
        "line 1\nline 2\nline 3\nline 4\nline 5\n"
        "line 6\nline 7\nline 8\nline 9\nline 10\n"
    )
    rep_col1 = ({"file": "test.py", "start": 2, "end": 5, "name": "c1"}, "NEW_C1\n")
    rep_col2 = ({"file": "test.py", "start": 4, "end": 7, "name": "c2"}, "NEW_C2\n")

    with pytest.raises(ValueError, match="Overlapping unit collision detected"):
        refactor_module_units(source_sample, [rep_col1, rep_col2])

    # 4. Test refactor_module_units reverse line order application (no offset drift)
    # Block A: lines 2-4 (replaced with 1 line)
    # Block B: lines 7-9 (replaced with 1 line)
    rep_a = ({"file": "test.py", "start": 2, "end": 4, "name": "block_a"}, "REPLACED_A\n")
    rep_b = ({"file": "test.py", "start": 7, "end": 9, "name": "block_b"}, "REPLACED_B\n")

    # Pass in forward order: refactor_module_units should internally sort reverse
    refactored = refactor_module_units(source_sample, [rep_a, rep_b])
    expected_lines = [
        "line 1\n",
        "REPLACED_A\n",
        "line 5\n",
        "line 6\n",
        "REPLACED_B\n",
        "line 10\n",
    ]
    assert refactored == "".join(expected_lines)

    # Empty replacements edge case
    assert refactor_module_units(source_sample, []) == source_sample

    # 5. Test generate_refactoring_patch with overlapping clone units in the same file
    f_multi = tmp_path / "multi_clone.py"
    f_multi.write_text(
        "def first_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n"
        "\n"
        "def second_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n",
        encoding="utf-8",
    )
    u_t1 = {"file": str(f_multi), "start": 1, "end": 4, "name": "first_task"}
    u_t2 = {"file": str(f_multi), "start": 6, "end": 9, "name": "second_task"}

    # Non-overlapping clones in the same file are both safely refactored in reverse order
    patch = generate_refactoring_patch([(0.95, u_t1, u_t2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_first_task_second_task" in patch
    assert "_shared_first_task_second_task(x)" in patch

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

def test_generate_refactoring_patch_cross_file_receiver_difference(tmp_path: Path) -> None:
    """Verifies that cross-file pairs with differing receiver kinds decline replacement if receiver is referenced."""
    code1 = (
        "class ServiceA:\n"
        "    def run(self, x: int) -> int:\n"
        "        y = x\n"
        "        return self.factor + y\n"
    )
    code2 = (
        "class ServiceB:\n"
        "    @classmethod\n"
        "    def run(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return cls.factor + y\n"
    )
    f1 = tmp_path / "srv_a.py"
    f2 = tmp_path / "srv_b.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "run", "kind": "function", "enclosing_class": "ServiceA", "receiver_kind": "instance"}
    u2 = {"file": str(f2), "start": 3, "end": 5, "name": "run", "kind": "function", "enclosing_class": "ServiceB", "receiver_kind": "class"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

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

def test_backslash_relative_path_normalization_on_unit(tmp_path: Path) -> None:
    """Verifies that units carrying Windows-style backslashes resolve correctly across platforms."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    f = pkg_dir / "service.py"
    f.write_text("def run_a(x: int) -> int:\n    return x + 10\n\ndef run_b(x: int) -> int:\n    return x + 10\n", encoding="utf-8")

    u1 = {"file": "pkg\\service.py", "start": 1, "end": 2, "name": "run_a", "kind": "function"}
    u2 = {"file": "pkg/service.py", "start": 4, "end": 5, "name": "run_b", "kind": "function"}

    extracted = extract_unit_source_code(u1, repo_root=str(tmp_path))
    assert len(extracted) == 2
    assert "def run_a" in extracted[0]

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/pkg/service.py" in patch
    assert "+++ b/pkg/service.py" in patch

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

def test_build_whole_method_delegation_single_line_function() -> None:
    """Verifies that _build_whole_method_delegation cleanly handles single-line functions without IndentationError."""
    code = "def add(x: int, y: int) -> int: return x + y\n"
    unit = {"file": "t.py", "start": 1, "end": 1, "name": "add", "kind": "function"}
    res = _build_whole_method_delegation(code, unit, "", "_shared_add", "x, y")
    assert "def add(x: int, y: int) -> int:\n" in res
    assert "return _shared_add(x, y)" in res
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
    parsed = ast.parse(res)
    assert len(parsed.body) == 1
    fn_def = parsed.body[0]
    assert isinstance(fn_def, ast.FunctionDef)
    assert len(fn_def.decorator_list) == 1

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

def test_batch_34_local_node_traversal_and_delegation_hardening(tmp_path: Path) -> None:
    """Tests Batch 34: _iter_local_nodes isolation and delegation tie-breaking."""
    from pydoppelgangerhunt.parser import harvest_file_units
    from pydoppelgangerhunt.fixer import _build_whole_method_delegation

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

def test_generate_refactoring_patch_cross_module_with_import_and_execution(tmp_path: Path) -> None:
    """Verifies that cross-module clone refactoring injects the import into file 2 and both run."""
    import subprocess  # pylint: disable=import-outside-toplevel

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
        cross_file_strategy="host",
    )

    # 1. Diff includes both files
    assert "--- a/service_pkg/srv_a.py" in patch
    assert "--- a/service_pkg/srv_b.py" in patch
    # 2. File 2 imports helper from file 1
    assert "from service_pkg.srv_a import _shared_compute_alpha_compute_beta" in patch

    # 3. Verify git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        ["python", "-c", "from service_pkg.srv_a import compute_alpha; from service_pkg.srv_b import compute_beta; print(compute_alpha(5), compute_beta(5))"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "57 57"

def test_generate_refactoring_patch_cross_module_circular_import_safety(tmp_path: Path) -> None:
    """Verifies that circular module imports are detected and prevented during cross-module patch generation."""
    pkg_dir = tmp_path / "cyclic_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "import cyclic_pkg.mod_b\n\n"
        "def work_a(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    src_b = (
        "def work_b(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    f_a = pkg_dir / "mod_a.py"
    f_b = pkg_dir / "mod_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "work_a", "file": "cyclic_pkg/mod_a.py", "start": 3, "end": 5, "kind": "function"}
    u_b = {"name": "work_b", "file": "cyclic_pkg/mod_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    # File 2 is NOT refactored with an unsafe circular import
    assert "--- a/cyclic_pkg/mod_b.py" not in patch
    assert "Circular import or unresolvable module path" in patch

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

def test_cross_module_helper_name_collision_deduplication(tmp_path: Path) -> None:
    """Verifies that if File 2 already defines the candidate helper name, an indexed suffix is generated."""
    pkg = tmp_path / "coll_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def handle_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    src2 = (
        "_shared_handle_data_process_data = 'existing_symbol'\n"
        "\n"
        "def process_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    (pkg / "f1.py").write_text(src1, encoding="utf-8")
    (pkg / "f2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "handle_data", "file": "coll_pkg/f1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "process_data", "file": "coll_pkg/f2.py", "start": 3, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )
    assert "_shared_handle_data_process_data_2" in patch
    assert "from coll_pkg.f1 import _shared_handle_data_process_data_2" in patch

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
    parsed = ast.parse(f"def wrapper():\n    {call_code}\n")
    assert isinstance(parsed, ast.Module)


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


def test_generate_refactoring_patch_permuted_outputs_alignment(tmp_path: Path) -> None:
    """Verifies that when clone 2 assigns variables in permuted order, the unpacking tuple aligns with the helper's return."""
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

    u1 = {"file": "mod1.py", "start": 2, "end": 3, "name": "compute_1:block", "kind": "compound_block"}
    u2 = {"file": "mod2.py", "start": 2, "end": 3, "name": "compute_2:block", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/mod1.py" in patch
    assert "--- a/mod2.py" in patch
    # Both call sites should unpack x, y in canonical helper return order
    assert "+    x, y = _shared_compute_1" in patch
    assert "y, x = _shared_compute_1" not in patch


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


def test_generate_refactoring_patch_cross_module_auto_prevents_circular_import(tmp_path: Path) -> None:
    """Verifies that synthesizing _common.py eliminates sibling circular dependencies even when imports exist."""
    pkg_dir = tmp_path / "cyclic_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "import cyclic_pkg.mod_b\n\n"
        "def work_a(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    src_b = (
        "def work_b(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "work_a", "file": "cyclic_pkg/mod_a.py", "start": 3, "end": 5, "kind": "function"}
    u_b = {"name": "work_b", "file": "cyclic_pkg/mod_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # In auto mode, _common.py is synthesized and neither sibling imports the other for the helper
    assert "--- a/cyclic_pkg/mod_a.py" in patch
    assert "--- a/cyclic_pkg/mod_b.py" in patch
    assert "from cyclic_pkg._common import _shared_work_a_work_b" in patch

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
            "import cyclic_pkg.mod_a; import cyclic_pkg.mod_b; "
            "print(cyclic_pkg.mod_a.work_a(5), cyclic_pkg.mod_b.work_b(5))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "12 12"


@pytest.mark.parametrize("reverse_order", [False, True])
def test_generate_refactoring_patch_cross_module_when_one_file_is_already_shared_module(
    tmp_path: Path, reverse_order: bool
) -> None:
    """Verifies consolidating when either duplicate unit is already located in the shared module."""
    pkg_dir = tmp_path / "shared_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_worker = (
        "def compute_task(x: int) -> int:\n"
        "    r = x * 4 + 3\n"
        "    return r\n"
    )
    src_common = (
        "def helper_seed() -> int:\n"
        "    return 1\n"
        "\n"
        "def compute_common(x: int) -> int:\n"
        "    r = x * 4 + 3\n"
        "    return r\n"
    )
    (pkg_dir / "worker.py").write_text(src_worker, encoding="utf-8")
    (pkg_dir / "_common.py").write_text(src_common, encoding="utf-8")

    u_worker = {"name": "compute_task", "file": "shared_pkg/worker.py", "start": 1, "end": 3, "kind": "function"}
    u_common = {"name": "compute_common", "file": "shared_pkg/_common.py", "start": 4, "end": 6, "kind": "function"}

    pair = (1.0, u_common, u_worker) if reverse_order else (1.0, u_worker, u_common)

    patch = generate_refactoring_patch(
        [pair],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    helper_name = "_shared_compute_common_compute_task" if reverse_order else "_shared_compute_task_compute_common"
    # worker.py imports from shared_pkg._common
    assert "--- a/shared_pkg/worker.py" in patch
    assert f"from shared_pkg._common import {helper_name}" in patch
    # _common.py defines the helper
    assert "--- a/shared_pkg/_common.py" in patch
    assert f"def {helper_name}" in patch

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
            "import shared_pkg.worker; import shared_pkg._common; "
            "print(shared_pkg.worker.compute_task(7), shared_pkg._common.compute_common(7))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "31 31"


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


def test_generate_refactoring_patch_shared_module_helper_name_collision_avoidance(
    tmp_path: Path,
) -> None:
    """Verifies that two distinct clone pairs sharing the same base name do not collide in _common.py."""
    pkg_dir = tmp_path / "collision_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "def compute_val(x: int) -> int:\n"
        "    return x * 10\n"
        "\n"
        "def calculate_val(x: int) -> int:\n"
        "    return x + 99\n"
    )
    src_b = (
        "def compute_val(y: int) -> int:\n"
        "    return y * 10\n"
        "\n"
        "def calculate_val(y: int) -> int:\n"
        "    return y + 99\n"
    )
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a1 = {"name": "compute_val", "file": "collision_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b1 = {"name": "compute_val", "file": "collision_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    u_a2 = {"name": "calculate_val", "file": "collision_pkg/mod_a.py", "start": 4, "end": 5, "kind": "function"}
    u_b2 = {"name": "calculate_val", "file": "collision_pkg/mod_b.py", "start": 4, "end": 5, "kind": "function"}

    # Pre-existing _common.py that already contains _shared_compute_val
    src_existing_common = (
        "def _shared_compute_val(x: int) -> int:\n"
        "    return x * 999\n"
    )
    (pkg_dir / "_common.py").write_text(src_existing_common, encoding="utf-8")

    patch = generate_refactoring_patch(
        [(1.0, u_a1, u_b1), (1.0, u_a2, u_b2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # Pre-existing _shared_compute_val was in _common.py, so helper for u_a1/u_b1 gets _shared_compute_val_2
    assert "def _shared_compute_val_2" in patch
    assert "def _shared_calculate_val" in patch
    assert "from collision_pkg._common import _shared_compute_val_2" in patch


def test_generate_refactoring_patch_auto_detects_existing_shared_module_cycle(
    tmp_path: Path,
) -> None:
    """Verifies that auto strategy detects when existing _common.py already imports caller and avoids cycle."""
    pkg_dir = tmp_path / "precycle_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    # _common.py already imports mod_a
    src_common = (
        "import precycle_pkg.mod_a\n\n"
        "def existing_seed() -> int:\n"
        "    return 1\n"
    )
    src_a = (
        "def run_task(x: int) -> int:\n"
        "    return x * 5\n"
    )
    src_b = (
        "def run_other(x: int) -> int:\n"
        "    return x * 5\n"
    )
    (pkg_dir / "_common.py").write_text(src_common, encoding="utf-8")
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "run_task", "file": "precycle_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "run_other", "file": "precycle_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # Circular import detected for mod_a -> _common -> mod_a
    assert (
        "Circular import detected "
        "(cycle: precycle_pkg.mod_a -> precycle_pkg._common -> precycle_pkg.mod_a)"
    ) in patch
    # mod_a must NOT be patched with a circular import
    assert "--- a/precycle_pkg/mod_a.py" not in patch
    # mod_b (which has no cycle) safely imports from _common
    assert "--- a/precycle_pkg/mod_b.py" in patch
    assert "from precycle_pkg._common import _shared_run_task_run_other" in patch


def test_generate_refactoring_patch_lazy_depgraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that build_module_graph is evaluated lazily only when needed."""
    mock_build = mock.MagicMock()
    monkeypatch.setattr(patch_mod, "build_module_graph", mock_build)

    f = tmp_path / "single.py"
    f.write_text("def a(): return 1\ndef b(): return 1\n", encoding="utf-8")
    u1 = {"name": "a", "file": str(f), "start": 1, "end": 1, "kind": "function"}
    u2 = {"name": "b", "file": str(f), "start": 2, "end": 2, "kind": "function"}

    # Intra-file patch generation does not consult or build the graph
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert "def _shared_a_b" in patch
    mock_build.assert_not_called()


def test_evolving_depgraph_detects_multi_pair_cycles(tmp_path: Path) -> None:
    """Verifies that accepted cross-module dependencies update the evolving graph, catching multi-pair cycles."""
    pkg = tmp_path / "cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod_a.py").write_text(
        "def a1(x: int) -> int:\n    return x + 1\ndef a2(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    (pkg / "mod_b.py").write_text(
        "def b1(x: int) -> int:\n    return x + 1\ndef b2(z: int) -> int:\n    return z * 3\n",
        encoding="utf-8",
    )
    (pkg / "mod_c.py").write_text(
        "def c1(z: int) -> int:\n    return z * 3\ndef c2(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )

    u_a1 = {"name": "a1", "file": "cycle_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b1 = {"name": "b1", "file": "cycle_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}
    u_b2 = {"name": "b2", "file": "cycle_pkg/mod_b.py", "start": 3, "end": 4, "kind": "function"}
    u_c1 = {"name": "c1", "file": "cycle_pkg/mod_c.py", "start": 1, "end": 2, "kind": "function"}
    u_c2 = {"name": "c2", "file": "cycle_pkg/mod_c.py", "start": 3, "end": 4, "kind": "function"}
    u_a2 = {"name": "a2", "file": "cycle_pkg/mod_a.py", "start": 3, "end": 4, "kind": "function"}

    # Pair 1: a1(host) <-> b1(caller) -> b imports a (b -> a)
    # Pair 2: b2(host) <-> c1(caller) -> c imports b (c -> b)
    # Pair 3: c2(host) <-> a2(caller) -> a imports c (a -> c), which closes a -> c -> b -> a cycle!
    clones = [
        (1.0, u_a1, u_b1),
        (1.0, u_b2, u_c1),
        (1.0, u_c2, u_a2),
    ]

    patch = generate_refactoring_patch(
        clones,
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # Pair 1 & Pair 2 succeed and are wired
    assert "from cycle_pkg.mod_a import _shared_a1_b1" in patch
    assert "from cycle_pkg.mod_b import _shared_b2_c1" in patch

    # Pair 3 is detected as closing a cycle: cycle_pkg.mod_a -> cycle_pkg.mod_c -> cycle_pkg.mod_b -> cycle_pkg.mod_a
    assert "Circular import or unresolvable module path" in patch
    assert "cycle_pkg.mod_a -> cycle_pkg.mod_c -> cycle_pkg.mod_b -> cycle_pkg.mod_a" in patch
    # mod_a must NOT import from mod_c
    assert "from cycle_pkg.mod_c import" not in patch


def test_shared_module_computes_required_imports_for_host(tmp_path: Path) -> None:
    """Verifies that shared modules receive typing and source dependencies even if already imported in f1."""
    pkg = tmp_path / "imports_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "from typing import List\n"
        "\n"
        "def calc_roots(vals: List[float]) -> float:\n"
        "    import math\n"
        "    return math.sqrt(sum(vals))\n"
    )
    src_f2 = (
        "from typing import List\n"
        "\n"
        "def compute_roots(items: List[float]) -> float:\n"
        "    import math\n"
        "    return math.sqrt(sum(items))\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "calc_roots", "file": "imports_pkg/f1.py", "start": 3, "end": 5, "kind": "function"}
    u2 = {"name": "compute_roots", "file": "imports_pkg/f2.py", "start": 3, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Verify that the synthesized _common.py patch contains typing imports and preserves local math import
    assert "diff --git a/imports_pkg/_common.py b/imports_pkg/_common.py" in patch
    assert "+from typing import List" in patch
    assert "+    import math" in patch
    assert "\n+import math\n" not in patch


def test_unreadable_existing_shared_module_skips_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an existing shared module that fails to read does not emit new-file patch."""
    pkg = tmp_path / "unread_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "f1.py").write_text("def run_a(x):\n    return x * 2\n", encoding="utf-8")
    (pkg / "f2.py").write_text("def run_b(x):\n    return x * 2\n", encoding="utf-8")
    shared_file = pkg / "_common.py"
    shared_file.write_text("# existing unreadable\n", encoding="utf-8")

    orig_read_text = Path.read_text

    def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() == shared_file.resolve():
            raise OSError("Permission denied: unreadable")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    u1 = {"name": "run_a", "file": "unread_pkg/f1.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "run_b", "file": "unread_pkg/f2.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Must NOT emit a new-file patch for _common.py
    assert "new file mode 100644" not in patch
    assert "--- /dev/null" not in patch
    # Must emit advisory notice about existing unreadable file
    expected_note = (
        "shared module file unread_pkg/_common.py exists but could not be read; skipping extraction."
    )
    assert expected_note in patch


def test_relative_imports_preserved_and_translated_in_shared_module(tmp_path: Path) -> None:
    """Verifies that relative imports in clone files are translated to canonical imports in _common.py."""
    pkg = tmp_path / "rel_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    src_f1 = (
        "from .helpers import CustomType\n"
        "\n"
        "def compute_val(v: CustomType) -> int:\n"
        "    return 42\n"
    )
    src_f2 = (
        "from .helpers import CustomType\n"
        "\n"
        "def compute_other(v: CustomType) -> int:\n"
        "    return 42\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "compute_val", "file": "rel_pkg/f1.py", "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_other", "file": "rel_pkg/f2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py should receive translated canonical import: from rel_pkg.helpers import CustomType
    assert "diff --git a/rel_pkg/_common.py b/rel_pkg/_common.py" in patch
    assert "+from rel_pkg.helpers import CustomType" in patch


def test_shared_module_host_imports_cycle_detection(tmp_path: Path) -> None:
    """Verifies that when _common.py gains dependencies on a caller module, cycle is detected and prevented."""
    pkg = tmp_path / "host_cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "class HelperDep:\n"
        "    pass\n"
        "\n"
        "def clone_a(x: HelperDep) -> int:\n"
        "    return 42\n"
    )
    src_f2 = (
        "from host_cycle_pkg.f1 import HelperDep\n"
        "\n"
        "def clone_b(x: HelperDep) -> int:\n"
        "    return 42\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "clone_a", "file": "host_cycle_pkg/f1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "clone_b", "file": "host_cycle_pkg/f2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Because helper uses HelperDep from host_cycle_pkg.f1, _common.py imports host_cycle_pkg.f1.
    # Therefore, f1 importing _common.py would form f1 -> _common -> f1 cycle!
    assert "Circular import detected" in patch
    assert "host_cycle_pkg.f1 -> host_cycle_pkg._common -> host_cycle_pkg.f1" in patch


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


def test_future_annotations_propagation_in_shared_module(tmp_path: Path) -> None:
    """Verifies that from __future__ import annotations is carried over to synthesized shared module."""
    pkg = tmp_path / "future_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "from __future__ import annotations\n"
        "from typing import List\n"
        "\n"
        "def process_models(items: List[str] | None) -> int:\n"
        "    return len(items) if items else 0\n"
    )
    src_f2 = (
        "from __future__ import annotations\n"
        "from typing import List\n"
        "\n"
        "def handle_models(items: List[str] | None) -> int:\n"
        "    return len(items) if items else 0\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "process_models", "file": "future_pkg/f1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "handle_models", "file": "future_pkg/f2.py", "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/future_pkg/_common.py b/future_pkg/_common.py" in patch
    assert "+from __future__ import annotations" in patch
    # Verify from __future__ import annotations comes before other imports in the diff
    future_idx = patch.find("+from __future__ import annotations")
    typing_idx = patch.find("+from typing import List")
    assert future_idx != -1
    assert typing_idx != -1
    assert future_idx < typing_idx


def test_shared_module_dry_run_no_ghost_edge_or_caller_import(tmp_path: Path) -> None:
    """Verifies that replace_clones=False with shared_module does not emit caller imports or ghost edges."""
    pkg = tmp_path / "dry_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_a = "def compute_a(x: int) -> int:\n    return x * 10 + 1\n"
    src_b = "def compute_b(x: int) -> int:\n    return x * 10 + 1\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_a", "file": "dry_shared_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "compute_b", "file": "dry_shared_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    # _common.py is synthesized with the helper
    assert "diff --git a/dry_shared_pkg/_common.py b/dry_shared_pkg/_common.py" in patch
    assert "+def _shared_compute_a_compute_b(x: int) -> int:" in patch

    # Callers receive the proposed import from _common
    assert "+from dry_shared_pkg._common import _shared_compute_a_compute_b" in patch
    # Function bodies are NOT replaced because replace_clones=False
    assert "return _shared_compute_a_compute_b" not in patch
    # Callers receive advisory comments explaining where helper was extracted
    assert "Complete refactoring by replacing the clone with a call in dry_shared_pkg/mod_a.py" in patch
    assert "Complete refactoring by replacing the clone with a call in dry_shared_pkg/mod_b.py" in patch


def test_shared_module_existing_directory_skipped(tmp_path: Path) -> None:
    """Verifies that when the target shared module path is an existing directory, extraction skips gracefully."""
    pkg = tmp_path / "dir_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Create a directory named _common.py
    dir_conflict = pkg / "_common.py"
    dir_conflict.mkdir()

    src_a = "def run_a(x: int) -> int:\n    return x + 5\n"
    src_b = "def run_b(x: int) -> int:\n    return x + 5\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "run_a", "file": "dir_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "run_b", "file": "dir_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "is an existing directory; skipping extraction" in patch
    assert "new file mode 100644" not in patch


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


def test_host_module_detects_cycle_from_synthesized_host_imports(tmp_path: Path) -> None:
    """Verifies that host_module detects cycle when host gains an import pointing back to caller."""
    pkg = tmp_path / "host_cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # mod_b has a helper function used by the clone in mod_b
    src_b = (
        "def helper_b(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "def compute_b(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    src_a = (
        "def helper_b(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "def compute_a(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_a", "file": "host_cycle_pkg/mod_a.py", "start": 4, "end": 6, "kind": "function"}
    u_b = {"name": "compute_b", "file": "host_cycle_pkg/mod_b.py", "start": 4, "end": 6, "kind": "function"}

    # Pre-wire mod_a to import mod_b so mod_a -> mod_b
    src_a_with_import = (
        "from host_cycle_pkg.mod_b import helper_b\n"
        "\n"
        "def compute_a(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    (pkg / "mod_a.py").write_text(src_a_with_import, encoding="utf-8")
    u_a["start"] = 3
    u_a["end"] = 5

    # If mod_b were to import mod_a, cycle mod_b -> mod_a -> mod_b would form
    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    assert "Circular import detected" in patch or "Circular import or unresolvable module path" in patch
    assert "from host_cycle_pkg.mod_a import _shared_compute_a_compute_b" not in patch


def test_host_module_ignores_type_checking_guarded_direct_import(tmp_path: Path) -> None:
    """Verifies that TYPE_CHECKING-only imports do not block safe host-module extraction."""
    pkg_dir = tmp_path / "guard_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    import guard_pkg.caller\n\n"
        "def compute_host(x: int) -> int:\n"
        "    return x * 2 + 5\n"
    )
    src_caller = (
        "def compute_caller(x: int) -> int:\n"
        "    return x * 2 + 5\n"
    )
    (pkg_dir / "host.py").write_text(src_host, encoding="utf-8")
    (pkg_dir / "caller.py").write_text(src_caller, encoding="utf-8")

    u_host = {"name": "compute_host", "file": "guard_pkg/host.py", "start": 6, "end": 7, "kind": "function"}
    u_caller = {"name": "compute_caller", "file": "guard_pkg/caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    assert "Circular import or unresolvable module path" not in patch
    assert "from guard_pkg.host import _shared_compute_host_compute_caller" in patch


def test_shared_module_translates_local_relative_imports(tmp_path: Path) -> None:
    """Verifies that relative function-local imports in clones are translated to canonical absolute paths."""
    pkg = tmp_path / "local_rel_pkg"
    sub1 = pkg / "sub1"
    sub2 = pkg / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "__init__.py").write_text("", encoding="utf-8")

    (pkg / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 42\n", encoding="utf-8")

    src_1 = (
        "def compute_1(x: int) -> int:\n"
        "    from ..helpers import transform\n"
        "    return transform(x)\n"
    )
    src_2 = (
        "def compute_2(x: int) -> int:\n"
        "    from ..helpers import transform\n"
        "    return transform(x)\n"
    )
    (sub1 / "mod1.py").write_text(src_1, encoding="utf-8")
    (sub2 / "mod2.py").write_text(src_2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": "local_rel_pkg/sub1/mod1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_2", "file": "local_rel_pkg/sub2/mod2.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In local_rel_pkg/_common.py, the import should be canonicalized to from local_rel_pkg.helpers import transform
    assert "from local_rel_pkg.helpers import transform" in patch
    # Should NOT have verbatim 'from ..helpers import transform' added in the shared module
    assert "+from ..helpers import transform" not in patch
    assert "+    from ..helpers import transform" not in patch


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


def test_shared_module_translates_local_relative_imports_package_init(tmp_path: Path) -> None:
    """Verifies that relative function-local imports in pkg/__init__.py are translated with package context."""
    pkg = tmp_path / "pkg_init_test"
    pkg.mkdir()
    (pkg / "deps.py").write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    src_init = (
        "def compute_init(x: int) -> int:\n"
        "    from .deps import helper\n"
        "    return helper(x)\n"
    )
    src_mod = (
        "def compute_mod(x: int) -> int:\n"
        "    from .deps import helper\n"
        "    return helper(x)\n"
    )
    (pkg / "__init__.py").write_text(src_init, encoding="utf-8")
    (pkg / "mod.py").write_text(src_mod, encoding="utf-8")

    u1 = {"name": "compute_init", "file": "pkg_init_test/__init__.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_mod", "file": "pkg_init_test/mod.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In pkg_init_test/_common.py, the import should be translated to from pkg_init_test.deps import helper
    assert "from pkg_init_test.deps import helper" in patch
    # Should NOT have resolved to from deps import helper (which would be missing pkg_init_test.)
    assert "+from deps import helper" not in patch
    assert "+    from deps import helper" not in patch


def test_existing_shared_module_gains_future_annotations_from_source(tmp_path: Path) -> None:
    """Verifies that an existing shared module gains future annotations when clone source has them."""
    pkg = tmp_path / "future_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # Pre-existing _common.py without future annotations
    (pkg / "_common.py").write_text("def existing_helper() -> None:\n    pass\n", encoding="utf-8")

    src1 = (
        "from __future__ import annotations\n"
        "from dep_pkg import Later\n\n"
        "def run_calc(x: Later) -> Later:\n"
        "    return x\n"
    )
    src2 = (
        "from __future__ import annotations\n"
        "from dep_pkg import Later\n\n"
        "def run_calc2(x: Later) -> Later:\n"
        "    return x\n"
    )
    (pkg / "mod1.py").write_text(src1, encoding="utf-8")
    (pkg / "mod2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "run_calc", "file": "future_pkg/mod1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "run_calc2", "file": "future_pkg/mod2.py", "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py should receive from __future__ import annotations
    assert "+from __future__ import annotations" in patch


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


def test_collect_host_missing_imports_preserves_aliased_typing(tmp_path: Path) -> None:
    """Verifies that aliased typing imports (e.g. from typing import List as MyList) are preserved."""
    pkg = tmp_path / "alias_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from typing import List as MyList\n\n"
        "def process(items: MyList[int]) -> int:\n"
        "    return len(items)\n"
    )
    src2 = (
        "from typing import List as MyList\n\n"
        "def process2(items: MyList[int]) -> int:\n"
        "    return len(items)\n"
    )
    (pkg / "mod1.py").write_text(src1, encoding="utf-8")
    (pkg / "mod2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "process", "file": "alias_pkg/mod1.py", "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "process2", "file": "alias_pkg/mod2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In _common.py, the aliased typing import should be present
    assert "from typing import List as MyList" in patch


def test_generate_patch_relative_repo_root_no_path_doubling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that passing a relative repo_root does not duplicate path components."""
    monkeypatch.chdir(tmp_path)
    rel_root = Path("sub_proj")
    pkg = rel_root / "my_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src = "def util(a: int) -> int:\n    return a * 2\n"
    (pkg / "a.py").write_text(src, encoding="utf-8")
    (pkg / "b.py").write_text(src, encoding="utf-8")

    u1 = {"name": "util", "file": str(pkg / "a.py"), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "util", "file": str(pkg / "b.py"), "start": 1, "end": 2, "kind": "function"}

    # Pass genuinely relative repo_root
    rel_repo_root = "sub_proj"
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=rel_repo_root,
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "sub_proj/sub_proj" not in patch.replace("\\", "/")
    assert "my_pkg/_common.py" in patch.replace("\\", "/")


def test_generate_patch_conflicting_import_symbols_skips_extraction(tmp_path: Path) -> None:
    """Verifies that clones binding the same symbol to different modules skip extraction."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    sub1 = pkg / "sub1"
    sub1.mkdir()
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    sub2 = pkg / "sub2"
    sub2.mkdir()
    (sub2 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    src1 = (
        "from .helpers import CustomType\n\n"
        "def compute(item: CustomType) -> int:\n"
        "    return len(item)\n"
    )
    src2 = (
        "from .helpers import CustomType\n\n"
        "def compute(item: CustomType) -> int:\n"
        "    return len(item)\n"
    )
    f1 = sub1 / "worker.py"
    f2 = sub2 / "worker.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType'" in patch
    assert "skipping extraction" in patch
    assert "_common.py" not in patch


def test_generate_patch_skips_guarded_imports_in_shared_module(tmp_path: Path) -> None:
    """Verifies that imports inside conditional or guarded blocks are not emitted into shared modules."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from typing import List\n"
        "if True:\n"
        "    from typing_extensions import Buffer\n\n"
        "def run_job(x: List[int]) -> int:\n"
        "    return len(x)\n"
    )
    src2 = (
        "from typing import List\n"
        "try:\n"
        "    import optional_dep\n"
        "except ImportError:\n"
        "    pass\n\n"
        "def run_job(x: List[int]) -> int:\n"
        "    return len(x)\n"
    )
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_job", "file": str(f1), "start": 5, "end": 6, "kind": "function"}
    u2 = {"name": "run_job", "file": str(f2), "start": 7, "end": 8, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    common_section = patch.split("diff --git a/pkg/_common.py")[-1]
    assert "from typing import List" in common_section
    assert "Buffer" not in common_section
    assert "optional_dep" not in common_section


def test_generate_patch_wildcard_import_unresolved_symbol(tmp_path: Path) -> None:
    """Verifies that an unresolved symbol in a helper where source used wildcard import skips extraction."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import *\n\n"
        "def calc(x: CustomType) -> float:\n"
        "    return float(len(str(x)))\n"
    )
    src2 = (
        "from math import *\n\n"
        "def calc(x: CustomType) -> float:\n"
        "    return float(len(str(x)))\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "potentially relies on wildcard import" in patch
    assert "skipping extraction" in patch


def test_cross_file_wildcard_with_explicit_import_rejected(tmp_path: Path) -> None:
    """Verifies that cross-file extraction is rejected when one clone uses wildcard and another has explicit import."""
    pkg = tmp_path / "pkg_wildcard_mixed"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep_a import *\n\n"
        "def process(x: TransformType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "from dep_b import TransformType\n\n"
        "def process(x: TransformType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "process", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "process", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "potentially relies on wildcard import" in patch
    assert "skipping extraction" in patch
    assert "from dep_b import TransformType" not in patch


def test_same_file_wildcard_import_allowed(tmp_path: Path) -> None:
    """Verifies that same-file clones sharing a module-level wildcard import are extracted."""
    pkg = tmp_path / "pkg_same_file_wildcard"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src = (
        "from math import *\n\n"
        "def run_a(x: int) -> float:\n"
        "    return float(x) * 2.0\n\n"
        "def run_b(x: int) -> float:\n"
        "    return float(x) * 2.0\n"
    )
    f = pkg / "mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "run_a", "file": str(f), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "run_b", "file": str(f), "start": 6, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    assert "potentially relies on wildcard import" not in patch
    assert "def _shared_run_a" in patch


def test_shared_module_rejects_unresolved_nonbuiltin_dependency(tmp_path: Path) -> None:
    """Verifies that a helper referencing an unimported module-local type rejects shared module extraction."""
    pkg = tmp_path / "pkg_unresolved"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def compute_1(x: LocalDep) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def compute_2(x: LocalDep) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": str(f1), "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "compute_2", "file": str(f2), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "unresolved symbol 'LocalDep' in shared module" in patch
    assert "skipping extraction" in patch


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


def test_shared_module_rejects_conflicting_local_imports(tmp_path: Path) -> None:
    """Verifies that clones with conflicting local imports for the same symbol are rejected."""
    pkg = tmp_path / "pkg_conflicting_locals"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def calc_1(x: float) -> float:\n"
        "    from math import sin as f\n"
        "    return f(x)\n"
    )
    src2 = (
        "def calc_2(x: float) -> float:\n"
        "    from cmath import sin as f\n"
        "    return f(x).real\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc_1", "file": str(f1), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "calc_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting local import symbol 'f' across clone sources" in patch
    assert "skipping extraction" in patch


def test_host_module_detects_conflicting_imported_symbol_with_host(tmp_path: Path) -> None:
    """Verifies that extraction is skipped if host module already imports the symbol from another package."""
    pkg = tmp_path / "pkg_host_conflict"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from pkg_a import transform\n\n"
        "def run_1(x: int) -> int:\n"
        "    from pkg_b import transform\n"
        "    return transform(x)\n"
    )
    src2 = (
        "def run_2(x: int) -> int:\n"
        "    from pkg_b import transform\n"
        "    return transform(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_1", "file": str(f1), "start": 3, "end": 5, "kind": "function"}
    u2 = {"name": "run_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting imported symbol 'transform' across clone sources" in patch
    assert "skipping extraction" in patch


def test_canonicalize_helper_relative_imports_with_nonstandard_whitespace() -> None:
    """Verifies that relative imports with irregular whitespace are parsed and canonicalized."""
    from pydoppelgangerhunt.fixer.patch import _canonicalize_helper_relative_imports  # pylint: disable=import-outside-toplevel

    code = (
        "def helper(x: int) -> int:\n"
        "    from   .helpers   import   transform\n"
        "    return transform(x)\n"
    )
    res = _canonicalize_helper_relative_imports(code, [("my_pkg.mod", False)])
    assert "from my_pkg.helpers import transform" in res
    assert "from   ." not in res


def test_shared_module_rejects_conflicting_local_relative_imports(tmp_path: Path) -> None:
    """Verifies that clones in different subpackages using conflicting relative imports are rejected."""
    pkg = tmp_path / "conflict_rel_pkg"
    sub1 = pkg / "sub1"
    sub2 = pkg / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "__init__.py").write_text("", encoding="utf-8")

    (sub1 / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    (sub2 / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 2\n", encoding="utf-8")

    src_1 = (
        "def compute_1(x: int) -> int:\n"
        "    from .helpers import transform\n"
        "    return transform(x)\n"
    )
    src_2 = (
        "def compute_2(x: int) -> int:\n"
        "    from .helpers import transform\n"
        "    return transform(x)\n"
    )
    (sub1 / "mod1.py").write_text(src_1, encoding="utf-8")
    (sub2 / "mod2.py").write_text(src_2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": "conflict_rel_pkg/sub1/mod1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_2", "file": "conflict_rel_pkg/sub2/mod2.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert (
        "conflicting relative import in helper across clone sources" in patch
        or "conflicting local import symbol 'transform' across clone sources" in patch
    )
    assert "from conflict_rel_pkg.sub1.helpers import transform" not in patch


def test_same_file_conflicting_local_imports_skips_without_corrupting_plan(tmp_path: Path) -> None:
    """Verifies same-file clones with conflicting local imports skip extraction cleanly without dirtying plan."""
    f = tmp_path / "same_file_mod.py"
    f.write_text(
        "def foo(x: int) -> int:\n"
        "    from math import sqrt as f\n"
        "    return f(x)\n\n"
        "def bar(x: int) -> int:\n"
        "    from cmath import sqrt as f\n"
        "    return f(x).real\n",
        encoding="utf-8",
    )
    u1 = {"name": "foo", "file": str(f), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "bar", "file": str(f), "start": 5, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    assert "conflicting local import symbol 'f' across clone sources" in patch
    assert "skipping extraction" in patch
    assert "_shared_foo_bar" not in patch


def test_local_imports_not_hoisted_to_module_scope(tmp_path: Path) -> None:
    """Verifies that lazy/guarded local imports stay inside the helper body and are not hoisted to module scope."""
    pkg = tmp_path / "lazy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def do_calc1(x: int) -> int:\n"
        "    import math\n"
        "    return math.isqrt(x)\n"
    )
    src2 = (
        "def do_calc2(x: int) -> int:\n"
        "    import math\n"
        "    return math.isqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "do_calc1", "file": str(f1), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "do_calc2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/lazy_pkg/_common.py b/lazy_pkg/_common.py" in patch
    assert "+    import math" in patch
    assert "\n+import math\n" not in patch


def test_host_existing_relative_imports_canonicalized_without_false_conflict(tmp_path: Path) -> None:
    """Verifies that existing relative imports in host module match canonicalized clone imports without conflict."""
    pkg = tmp_path / "rel_host_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    host_src = (
        "from .helpers import CustomType\n\n"
        "def run_host(val: CustomType) -> CustomType:\n"
        "    return val\n"
    )
    caller_src = (
        "from rel_host_pkg.helpers import CustomType\n\n"
        "def run_caller(val: CustomType) -> CustomType:\n"
        "    return val\n"
    )

    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(host_src, encoding="utf-8")
    f_caller.write_text(caller_src, encoding="utf-8")

    u1 = {"name": "run_host", "file": str(f_host), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "run_caller", "file": str(f_caller), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting imported symbol" not in patch
    assert "skipping extraction" not in patch
    assert "from rel_host_pkg.host_mod import _shared_run_host_run_caller" in patch


def test_shared_module_resolution_failure_adds_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if resolve_shared_module_file raises ValueError, patch adds skip advisory comment."""
    pkg = tmp_path / "unsafe_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 1\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    from pydoppelgangerhunt.fixer import patch as patch_mod  # pylint: disable=import-outside-toplevel

    def mock_resolve_shared(*args: Any, **kwargs: Any) -> Path:
        raise ValueError("Shared module path resolves outside common package directory")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_shared)

    patch = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Shared module path resolves outside common package directory" in patch
    assert "skipping extraction" in patch
    assert "_common.py" not in patch


def test_shared_module_resolution_filesystem_error_adds_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if resolve_shared_module_file raises OSError or RuntimeError, patch adds skip advisory comment."""
    pkg = tmp_path / "oserr_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 1\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    from pydoppelgangerhunt.fixer import patch as patch_mod  # pylint: disable=import-outside-toplevel

    def mock_resolve_oserror(*args: Any, **kwargs: Any) -> Path:
        raise OSError("Permission denied: cannot access directory")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_oserror)

    patch_os = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Permission denied: cannot access directory" in patch_os
    assert "skipping extraction" in patch_os
    assert "_common.py" not in patch_os

    def mock_resolve_runtime_err(*args: Any, **kwargs: Any) -> Path:
        raise RuntimeError("Symlink loop detected")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_runtime_err)

    patch_rt = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Symlink loop detected" in patch_rt
    assert "skipping extraction" in patch_rt
    assert "_common.py" not in patch_rt


def test_clone_sources_conflicting_definition_and_import_rejected(tmp_path: Path) -> None:
    """Verifies that if one clone imports a symbol while another defines it locally, extraction is rejected."""
    pkg = tmp_path / "conflict_def_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep import CustomType\n\n"
        "def calc1(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "class CustomType:\n"
        "    pass\n\n"
        "def calc2(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType' across clone sources" in patch
    assert "skipping extraction" in patch


def test_clone_sources_inconsistent_binding_missing_import_rejected(tmp_path: Path) -> None:
    """Verifies that if one clone imports a symbol while another has no binding for it, extraction is rejected."""
    pkg = tmp_path / "inconsistent_bind_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep import CustomType\n\n"
        "def calc1(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "def calc2(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType' across clone sources" in patch
    assert "skipping extraction" in patch


def test_helper_retains_module_import_when_local_import_not_selected(tmp_path: Path) -> None:
    """Verifies that module-level import is hoisted when clone 2 had local import not present in clone 1 helper."""
    pkg = tmp_path / "retained_local_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import sqrt\n\n"
        "def calc1(x: int) -> float:\n"
        "    return sqrt(x)\n"
    )
    src2 = (
        "def calc2(x: int) -> float:\n"
        "    from math import sqrt\n"
        "    return sqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    assert "from math import sqrt" in patch
    assert "skipping extraction" not in patch
    assert "diff --git a/retained_local_pkg/_common.py b/retained_local_pkg/_common.py" in patch


def test_nested_function_parameters_do_not_shadow_outer_helper_imports(tmp_path: Path) -> None:
    """Verifies that nested function parameters in helper do not prevent module-level imports from hoisting."""
    pkg = tmp_path / "nested_shadow_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import sqrt as transform\n\n"
        "def runner1(x: int) -> float:\n"
        "    res = transform(x)\n"
        "    def inner(transform: int) -> int:\n"
        "        return transform + 1\n"
        "    return res + inner(1)\n"
    )
    src2 = (
        "from math import sqrt as transform\n\n"
        "def runner2(x: int) -> float:\n"
        "    res = transform(x)\n"
        "    def inner(transform: int) -> int:\n"
        "        return transform + 1\n"
        "    return res + inner(1)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "runner1", "file": str(f1), "start": 3, "end": 7, "kind": "function"}
    u2 = {"name": "runner2", "file": str(f2), "start": 3, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from math import sqrt as transform" in patch
    assert "skipping extraction" not in patch


def test_symlink_shared_module_target_rejected_with_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if target shared module is a symlink, extraction is skipped with advisory comment."""
    pkg = tmp_path / "sym_target_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def f1(x: int) -> int:\n    return x + 1\n"
    src2 = "def f2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "f1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "f2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    orig_is_symlink = Path.is_symlink

    def mock_is_symlink(self: Path) -> bool:
        if self.name == "_common.py":
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "is an existing symlink" in patch
    assert "skipping extraction" in patch
    assert "diff --git" not in patch


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


def test_generate_refactoring_patch_with_package_root_depgraph_preserves_cycle_detection(
    tmp_path: Path,
) -> None:
    """Verifies externally built depgraphs from package roots use the same module namespace."""
    from pydoppelgangerhunt.fixer.depgraph import build_module_graph  # pylint: disable=import-outside-toplevel

    project_root = tmp_path / "project"
    project_root.mkdir()
    top_pkg = project_root / "my_package"
    top_pkg.mkdir()
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")
    (top_pkg / "caller.py").write_text(
        "def compute_c(x: int) -> int:\n"
        "    return x + 10\n",
        encoding="utf-8",
    )
    (top_pkg / "mid.py").write_text(
        "import my_package.caller\n\n"
        "def compute_m(x: int) -> int:\n"
        "    return my_package.caller.compute_c(x)\n",
        encoding="utf-8",
    )
    (top_pkg / "host.py").write_text(
        "import my_package.mid\n\n"
        "def compute_h(x: int) -> int:\n"
        "    return x + 10\n",
        encoding="utf-8",
    )

    depgraph = build_module_graph(top_pkg)
    u_host = {"name": "compute_h", "file": "host.py", "start": 3, "end": 4, "kind": "function"}
    u_caller = {"name": "compute_c", "file": "caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(top_pkg),
        replace_clones=True,
        cross_file_strategy="host_module",
        depgraph=depgraph,
    )

    assert (
        "Circular import or unresolvable module path "
        "(cycle: my_package.caller -> my_package.host -> my_package.mid -> my_package.caller)"
    ) in patch
    assert "from my_package.host import _shared_compute_h_compute_c" not in patch


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


def test_collect_host_missing_imports_handles_shadowed_builtins(tmp_path: Path) -> None:
    """Verifies that shadowed builtins are hoisted when consistent or rejected when conflicting."""
    pkg = tmp_path / "shadow_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Consistent shadowing: both modules import a custom `len`
    src1_consistent = (
        "from custom_utils import len\n\n"
        "def count_items1(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    src2_consistent = (
        "from custom_utils import len\n\n"
        "def count_items2(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    f1 = pkg / "c1.py"
    f2 = pkg / "c2.py"
    f1.write_text(src1_consistent, encoding="utf-8")
    f2.write_text(src2_consistent, encoding="utf-8")

    u1 = {"name": "count_items1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "count_items2", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch_consistent = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from custom_utils import len" in patch_consistent
    assert "from shadow_pkg._common import _shared_count_items1_count_items2" in patch_consistent

    # Inconsistent shadowing: one clone imports custom `len`, the other uses builtin `len`
    src2_inconsistent = (
        "def count_items2(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    f2.write_text(src2_inconsistent, encoding="utf-8")
    u2_inconsistent = {"name": "count_items2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch_inconsistent = generate_refactoring_patch(
        [(1.0, u1, u2_inconsistent)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'len'" in patch_inconsistent
    assert "skipping extraction" in patch_inconsistent


def test_generate_refactoring_patch_cross_file_strategy_skip(tmp_path: Path) -> None:
    """Verifies that cross_file_strategy='skip' bypasses cross-file clones while allowing same-file clones."""
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text("def a():\n    return 1\ndef b():\n    return 1\n", encoding="utf-8")
    f2.write_text("def c():\n    return 1\n", encoding="utf-8")

    u_a = {"name": "a", "file": "mod1.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "b", "file": "mod1.py", "start": 3, "end": 4, "kind": "function"}
    u_c = {"name": "c", "file": "mod2.py", "start": 1, "end": 2, "kind": "function"}

    # Pure cross-file pair with skip should produce no patch
    patch_cross = generate_refactoring_patch(
        [(1.0, u_a, u_c)],
        repo_root=str(tmp_path),
        cross_file_strategy="skip",
    )
    assert patch_cross == ""

    # Same-file pair should still be extracted even when cross_file_strategy='skip'
    patch_same = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        cross_file_strategy="skip",
    )
    assert "def _shared_a_b" in patch_same


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


def test_nested_imports_do_not_suppress_outer_helper_imports(tmp_path: Path) -> None:
    """Verifies that imports in nested functions/classes do not suppress outer helper imports."""
    # pylint: disable=protected-access
    code = (
        "def helper(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    tree = ast.parse(code)
    free_names, defined_names, local_imports = patch_mod._extract_helper_symbols(tree)
    assert "transform" in free_names
    assert "transform" not in local_imports
    assert "helper" in defined_names
    assert "x" in defined_names
    assert "inner" in defined_names
    assert "y" not in defined_names

    # End-to-end patch generation test:
    # Outer helper uses `transform`, while inner function locally imports a different `transform`.
    # Ensure `from mymod import transform` is hoisted to the shared module header.
    pkg = tmp_path / "nested_import_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from mymod import transform\n\n"
        "def run_step1(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    src2 = (
        "from mymod import transform\n\n"
        "def run_step2(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    f1 = pkg / "s1.py"
    f2 = pkg / "s2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_step1", "file": str(f1), "start": 3, "end": 8, "kind": "function"}
    u2 = {"name": "run_step2", "file": str(f2), "start": 3, "end": 8, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from mymod import transform" in patch
    assert "from nested_import_pkg._common import _shared_run_step1_run_step2" in patch


def test_type_checking_guarded_import_does_not_suppress_runtime_helper_typing(
    tmp_path: Path,
) -> None:
    """Verifies that imports under 'if TYPE_CHECKING:' do not suppress runtime helper typing imports."""
    pkg = tmp_path / "type_check_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Existing _common.py contains guarded TYPE_CHECKING import
    common_file = pkg / "_common.py"
    common_file.write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from typing import List\n",
        encoding="utf-8",
    )

    src1 = (
        "def compute_items1(vals: List[int]) -> int:\n"
        "    return sum(vals)\n"
    )
    src2 = (
        "def compute_items2(vals: List[int]) -> int:\n"
        "    return sum(vals)\n"
    )
    f1 = pkg / "c1.py"
    f2 = pkg / "c2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_items1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "compute_items2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py patch MUST contain unconditional 'from typing import List'
    assert "+from typing import List" in patch


def test_cross_file_host_local_definition_conflict_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that cross-file extraction into host module is rejected when callers define conflicting local dependencies."""
    # pylint: disable=protected-access
    pkg = tmp_path / "host_conflict_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "class LocalService:\n"
        "    pass\n\n"
        "def process_service1(svc: LocalService) -> int:\n"
        "    return 42\n"
    )
    src_caller = (
        "class LocalService:\n"
        "    pass\n\n"
        "def process_service2(svc: LocalService) -> int:\n"
        "    return 42\n"
    )
    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(src_host, encoding="utf-8")
    f_caller.write_text(src_caller, encoding="utf-8")

    u_host = {"name": "process_service1", "file": str(f_host), "start": 4, "end": 5, "kind": "function"}
    u_caller = {"name": "process_service2", "file": str(f_caller), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting local definition 'LocalService' across clone sources" in patch
    # Extraction must be safely skipped, no helper inserted
    assert "_shared_process_service1_process_service2" not in patch

    # Direct collector verification for conflicting and unresolved symbols
    host_plan = patch_mod._FilePatchPlan(f_host, src_host, "host_mod.py")
    helper_code = "def _shared(svc: LocalService) -> int:\n    return 42\n"
    with pytest.raises(ValueError, match="conflicting local definition 'LocalService'"):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope={},
            source_texts=[(src_host, "host_mod", False), (src_caller, "caller_mod", False)],
            host_mod="host_mod",
        )

    with pytest.raises(ValueError, match="unresolved symbol 'LocalService' in caller module"):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope={},
            source_texts=[(src_host, "host_mod", False), ("# no defs\n", "caller_mod", False)],
            host_mod="host_mod",
        )


def test_host_local_definition_omits_self_import_when_caller_imports_from_host(
    tmp_path: Path,
) -> None:
    """Verifies that when a caller imports a dependency from host, host omits self-imports of its own local definition."""
    # pylint: disable=protected-access
    pkg = tmp_path / "self_import_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    host_src = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def process_item1(item: LocalDep) -> int:\n"
        "    return 42\n"
    )
    caller_src = (
        "from self_import_pkg.host_mod import LocalDep\n\n"
        "def process_item2(item: LocalDep) -> int:\n"
        "    return 42\n"
    )
    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(host_src, encoding="utf-8")
    f_caller.write_text(caller_src, encoding="utf-8")

    # 1. Direct collector check
    host_plan = patch_mod._FilePatchPlan(f_host, host_src, "self_import_pkg/host_mod.py")
    helper_code = "def _shared_process_item1_process_item2(item: LocalDep) -> int:\n    return 42\n"
    imports = patch_mod._collect_host_missing_imports(
        host_plan=host_plan,
        helper_code=helper_code,
        scope={},
        source_texts=[
            (host_src, "self_import_pkg.host_mod", False),
            (caller_src, "self_import_pkg.caller_mod", False),
        ],
        host_mod="self_import_pkg.host_mod",
    )
    # Must NOT contain self-import from self_import_pkg.host_mod
    assert not any("self_import_pkg.host_mod" in imp for imp in imports)
    assert not any("LocalDep" in imp for imp in imports)

    # 2. End-to-end patch generation check
    u1 = {"name": "process_item1", "file": str(f_host), "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "process_item2", "file": str(f_caller), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    # Host module diff must NOT contain a self-import
    assert "+from self_import_pkg.host_mod import LocalDep" not in patch
    assert "+import self_import_pkg.host_mod" not in patch
    # Host module must receive the extracted helper
    assert "def _shared_process_item1_process_item2(item: LocalDep) -> int:" in patch
    # Caller module must import the extracted helper from host
    assert "from self_import_pkg.host_mod import _shared_process_item1_process_item2" in patch


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


def test_clones_with_matching_local_imports_hoisted_to_shared_module(tmp_path: Path) -> None:
    """Verifies that when helper lacks a local import, an agreed local import from clones is safely hoisted."""
    pkg = tmp_path / "shared_local_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import isqrt\n\n"
        "def compute_1(x: int) -> int:\n"
        "    return isqrt(x)\n"
    )
    src2 = (
        "def compute_2(x: int) -> int:\n"
        "    from math import isqrt\n"
        "    return isqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/shared_local_pkg/_common.py b/shared_local_pkg/_common.py" in patch
    assert "+from math import isqrt" in patch
    assert "def _shared_compute_1_compute_2(x: int) -> int:" in patch


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


def test_shared_module_rejects_subsequent_pair_creating_cycle_with_earlier_caller(
    tmp_path: Path,
) -> None:
    """Verifies that a subsequent clone pair cannot inject dependencies into a shared module

    that create a circular dependency with an earlier pair's caller.
    """
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # caller_a defines CustomType and has clone 1
    src_a = (
        "class CustomType:\n"
        "    pass\n\n"
        "def clone_one_a(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    # caller_b has clone 1
    src_b = (
        "def clone_one_b(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    # caller_c imports CustomType from caller_a and has clone 2
    src_c = (
        "from pkg.caller_a import CustomType\n\n"
        "def clone_two_c(item: CustomType) -> int:\n"
        "    a = 10\n"
        "    b = a + 20\n"
        "    return b\n"
    )
    # caller_d imports CustomType from caller_a and has clone 2
    src_d = (
        "from pkg.caller_a import CustomType\n\n"
        "def clone_two_d(item: CustomType) -> int:\n"
        "    a = 10\n"
        "    b = a + 20\n"
        "    return b\n"
    )

    fa = pkg / "caller_a.py"
    fb = pkg / "caller_b.py"
    fc = pkg / "caller_c.py"
    fd = pkg / "caller_d.py"
    fa.write_text(src_a, encoding="utf-8")
    fb.write_text(src_b, encoding="utf-8")
    fc.write_text(src_c, encoding="utf-8")
    fd.write_text(src_d, encoding="utf-8")

    u1a = {"name": "clone_one_a", "file": str(fa), "start": 4, "end": 7, "kind": "function"}
    u1b = {"name": "clone_one_b", "file": str(fb), "start": 1, "end": 4, "kind": "function"}
    u2c = {"name": "clone_two_c", "file": str(fc), "start": 3, "end": 6, "kind": "function"}
    u2d = {"name": "clone_two_d", "file": str(fd), "start": 3, "end": 6, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1a, u1b), (1.0, u2c, u2d)],
        repo_root=str(root),
        cross_file_strategy="shared_module",
        replace_clones=True,
    )

    assert patch is not None
    # Pair 1 succeeded in creating _common.py
    assert "pkg/_common.py" in patch
    # Pair 1 callers were patched
    assert "pkg/caller_a.py" in patch
    assert "pkg/caller_b.py" in patch

    # _common.py must NOT import caller_a (which would cause a cycle)
    assert "+from pkg.caller_a import" not in patch

    # Pair 2 was rejected due to circular dependency
    assert "rejected due to circular dependency" in patch


def test_cross_module_shared_module_directory_collision_rejected(tmp_path: Path) -> None:
    """Verifies that if the shared module path is an existing directory, extraction is safely skipped."""
    pkg = tmp_path / "pkg_dir_collision"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 10\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 10\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    # Create directory at _common.py
    (pkg / "_common.py").mkdir()

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        cross_file_strategy="shared_module",
        replace_clones=True,
    )

    assert "existing directory" in patch
    assert "skipping extraction" in patch


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


