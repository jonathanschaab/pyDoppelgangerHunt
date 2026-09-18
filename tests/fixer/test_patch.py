"""Unit tests for fixer refactoring patch generation, delegation calls, overlap collision, and git apply."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

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
