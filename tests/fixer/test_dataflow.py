"""Unit tests for fixer dataflow analysis, downstream reads, and generator subroutines."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    generate_refactoring_patch,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access
    GeneratorCloneSideData,
    _DownstreamReadVisitor,
    _VisitorPassMode,
    _clear_downstream_reads_cache,
    _collect_pre_unit_closures,
    _collect_scope_closures,
    _downstream_cache_lock,
    _downstream_reads_cache,
    _enclosing_try_reads_outputs,
    _extract_effective_unit_outputs,
    _extract_nested_scope_free_reads,
    _find_enclosing_loops,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _get_scope_stmts_by_end_lineno,
    _get_scope_try_and_with_blocks,
    _load_unit_file_text,
    _pair_clone_outputs,
    _scope_closures_cache,
    _scope_loops_cache,
    _scope_parent_maps,
    _scope_reads_outputs_after_line,
    _scope_stmts_by_end,
    _scope_try_with_cache,
    collect_downstream_read_names,
    is_async_generator_with_return_value,
    resolve_clone_generator_subroutine_outputs,
    resolve_generator_subroutine_outputs,
)


def test_collect_downstream_read_names_scope_and_closure_capture() -> None:
    """Verifies that collect_downstream_read_names captures free variables and immediate class body reads, while respecting shadowed locals."""
    # Case 1: Immediately executed class body (class C: value = x) and closure capture (def inner(): return x)
    code = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        return x\n"
        "\n"
        "    class C:\n"
        "        value = x\n"
        "\n"
        "    print(total)\n"
    )
    unit = {"start": 3, "end": 5}

    # With candidates: both 'total' and 'x' are captured downstream
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads == {"total", "x"}

    # Without candidates: all loaded names in outer scope and escaping nested scopes are found
    all_reads = collect_downstream_read_names(code, unit)
    assert all_reads == {"total", "x", "print"}

    # Case 2: Shadowed variables in nested functions/classes do NOT escape to outer scope
    code_shadowed = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "\n"
        "    def inner_param(x):\n"
        "        return x\n"
        "\n"
        "    def inner_local():\n"
        "        x = 10\n"
        "        return x\n"
        "\n"
        "    class ClassShadowed:\n"
        "        x = 10\n"
        "        value = x\n"
        "\n"
        "    print(total)\n"
    )
    reads_shadowed = collect_downstream_read_names(code_shadowed, unit, candidates={"total", "x"})
    assert reads_shadowed == {"total"}


def test_collect_downstream_read_names_positional_and_expression_boundaries() -> None:
    """Verifies same-line boundaries, same-expression, enclosing-expression, and multiline expression reads."""
    # 1. Same-line unit boundary: statement after semicolon
    code_sameline = (
        "def f(items):\n"
        "    total = 0\n"
        "    for x in items: yield x; print(total)\n"
    )
    u_sameline = {"start": 3, "end": 3, "end_col": 27}
    reads_sameline = collect_downstream_read_names(code_sameline, u_sameline, candidates={"total", "x"})
    assert reads_sameline == {"total"}

    # 2. Output used later in the same expression: (yield x) + x
    code_same_expr = (
        "def f(x):\n"
        "    res = (yield x) + x\n"
        "    return res\n"
    )
    u_same_expr = {"start": 2, "end": 2, "end_col": 19}
    reads_same_expr = collect_downstream_read_names(code_same_expr, u_same_expr, candidates={"x"})
    assert reads_same_expr == {"x"}

    # 3. Output used in an enclosing call expression: func((yield x), x)
    code_call_expr = (
        "def f(x):\n"
        "    res = func((yield x), x)\n"
        "    return res\n"
    )
    u_call_expr = {"start": 2, "end": 2, "end_col": 25}
    reads_call_expr = collect_downstream_read_names(code_call_expr, u_call_expr, candidates={"x"})
    assert reads_call_expr == {"x"}

    # 4. Multiline expression: argument on subsequent line
    code_multiline = (
        "def f(x):\n"
        "    res = func(\n"
        "        (yield x),\n"
        "        x,\n"
        "    )\n"
        "    return res\n"
    )
    u_multiline = {"start": 3, "end": 3, "end_col": 17}
    reads_multiline = collect_downstream_read_names(code_multiline, u_multiline, candidates={"x"})
    assert reads_multiline == {"x"}

    # 5. Output used BEFORE the unit on the same line: not downstream
    code_prior = (
        "def f(x):\n"
        "    res = x + (yield x)\n"
        "    return res\n"
    )
    u_prior = {"start": 2, "end": 2, "end_col": 23}
    reads_prior = collect_downstream_read_names(code_prior, u_prior, candidates={"x"})
    assert reads_prior == set()


def test_resolve_generator_subroutine_outputs_preserves_per_side_necessity_and_safe_fallback(
) -> None:
    """Verifies that resolve_generator_subroutine_outputs pairs renamed outputs and guards
    against partial knowledge."""
    u1_outs = ["total", "x"]
    u2_outs = ["count", "x"]

    # 1. Both sides known: only the mapped total/count slot is needed, x is discarded
    res1 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}, definite={"total"}),
        GeneratorCloneSideData(u2_outs, downstream={"count"}, definite={"count"}),
    )
    assert res1 is not None
    out1, out2 = res1
    assert out1 == ["total"]
    assert out2 == ["count"]

    # 2. Unknown fallback with definite sets: fails closed when candidate lacks definite store
    res2 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}, definite={"total"}),
        GeneratorCloneSideData(u2_outs, downstream=None, definite={"count"}),
    )
    assert res2 is None

    res2_def = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}, definite={"total", "x"}),
        GeneratorCloneSideData(u2_outs, downstream=None, definite={"count", "x"}),
    )
    assert res2_def is not None
    fb_def1, fb_def2 = res2_def
    assert fb_def1 == ["total", "x"]
    assert fb_def2 == ["count", "x"]

    # 3. Unknown fallback without definite sets: fails closed on non-definite outputs
    res3 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}),
        GeneratorCloneSideData(u2_outs, downstream=None),
    )
    assert res3 is None

    # 4. Neither side needs outputs
    res4 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream=set()),
        GeneratorCloneSideData(u2_outs, downstream=set()),
    )
    assert res4 is not None
    none_out1, none_out2 = res4
    assert not none_out1
    assert not none_out2

    # 5. Unequal output counts where needed output has no counterpart: fails closed (returns None)
    res_unequal = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(["total"], downstream={"total"}),
        GeneratorCloneSideData(["count", "status"], downstream={"count"}),
    )
    assert res_unequal is None

    # 6. Downstream explicitly requires output assigned in loop: fails closed if not definite
    res_needed = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(["x"], downstream={"x"}, definite=set()),
        GeneratorCloneSideData(["x"], downstream={"x"}, definite=set()),
    )
    assert res_needed is None


def test_collect_downstream_read_names_comprehensions_and_walrus() -> None:
    """Verifies that comprehension targets do not pollute enclosing scope and walrus bindings are scoped correctly."""

    # 1. Comprehension inside downstream closure: target variable 'x' is comprehension-local,
    # so 'return x' correctly reads outer 'x' as a free variable
    code_closure_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        _ = [x for x in items]\n"
        "        return x\n"
        "\n"
        "    return inner\n"
    )
    u_closure_comp = {"start": 2, "end": 3}
    reads_closure = collect_downstream_read_names(code_closure_comp, u_closure_comp, candidates={"x"})
    assert reads_closure == {"x"}

    # 2. Walrus operator (:=) inside comprehension: binds to enclosing function scope per PEP 572
    code_walrus = (
        "def outer(items):\n"
        "    for w in items:\n"
        "        yield w\n"
        "\n"
        "    def inner():\n"
        "        _ = [(w := y) for y in items]\n"
        "        return w\n"
        "\n"
        "    return inner\n"
    )
    u_walrus = {"start": 2, "end": 3}
    reads_walrus = collect_downstream_read_names(code_walrus, u_walrus, candidates={"w"})
    # 'w' is bound locally inside 'inner' by the walrus expression, not read from outer
    assert reads_walrus == set()

    # 3. Direct comprehension downstream: comprehension target is not a read of generator variable
    code_direct_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    res = [x for x in items]\n"
        "    return res\n"
    )
    u_direct = {"start": 2, "end": 3}
    reads_direct = collect_downstream_read_names(code_direct_comp, u_direct, candidates={"x"})
    assert reads_direct == set()

    # 4. Direct comprehension referencing outer variable in if-filter
    code_filter_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    res = [y for y in items if y == x]\n"
        "    return res\n"
    )
    reads_filter = collect_downstream_read_names(code_filter_comp, u_direct, candidates={"x"})
    assert reads_filter == {"x"}

    # 5. Comprehension inside downstream ClassDef body: target is not class attribute, iter is free read
    code_class_comp = (
        "def outer(data):\n"
        "    for x in data:\n"
        "        yield x\n"
        "    class C:\n"
        "        items = [x for x in data]\n"
        "    return C\n"
    )
    reads_class = collect_downstream_read_names(code_class_comp, u_direct, candidates={"x", "data"})
    assert reads_class == {"data"}


def test_generator_clone_with_nested_genexpr(tmp_path: Path) -> None:
    """Verifies that nested generator expressions inside generator units do not leak comprehension targets into scope."""
    code1 = (
        "def process_matrix(matrix: list[list[int]]):\n"
        "    grand_total = 0\n"
        "    for row in matrix:\n"
        "        row_sum = sum(x * 2 for x in row)\n"
        "        grand_total += row_sum\n"
        "        yield row_sum\n"
        "    return grand_total\n"
    )
    code2 = (
        "def process_matrix_alt(matrix: list[list[int]]):\n"
        "    grand_total = 0\n"
        "    for row in matrix:\n"
        "        row_sum = sum(y * 2 for y in row)\n"
        "        grand_total += row_sum\n"
        "        yield row_sum\n"
        "    return grand_total\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "mod1.py",
        "start": 3,
        "end": 6,
        "name": "process_matrix",
        "kind": "compound_block",
    }
    u2 = {
        "file": "mod2.py",
        "start": 3,
        "end": 6,
        "name": "process_matrix_alt",
        "kind": "compound_block",
    }

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        source_text1=code1,
        source_text2=code2,
    )
    assert helper != ""
    assert "def _shared" in helper
    assert "yield row_sum" in helper
    assert "return grand_total" in helper
    # Comprehension targets x and y must not be parameters of the synthesized helper
    sig_line = helper.splitlines()[0]
    params_part = sig_line[sig_line.index("(") + 1 : sig_line.rindex(")")]
    param_names = [p.split(":")[0].strip() for p in params_part.split(",") if p.strip()]
    assert "x" not in param_names and "y" not in param_names
    assert "grand_total" in param_names

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "_shared" in patch


def test_generator_clone_with_yield_from_and_returns(tmp_path: Path) -> None:
    """Verifies that generator clones containing yield from delegate return values and propagate outputs correctly."""
    code1 = (
        "def stream_blocks(blocks: list[list[int]]):\n"
        "    total_count = 0\n"
        "    for b in blocks:\n"
        "        yield from b\n"
        "        total_count += len(b)\n"
        "    return total_count\n"
    )
    code2 = (
        "def stream_blocks_alt(blocks: list[list[int]]):\n"
        "    total_count = 0\n"
        "    for b in blocks:\n"
        "        yield from b\n"
        "        total_count += len(b)\n"
        "    return total_count\n"
    )
    f1 = tmp_path / "stream1.py"
    f2 = tmp_path / "stream2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "stream1.py",
        "start": 3,
        "end": 5,
        "name": "stream_blocks",
        "kind": "compound_block",
    }
    u2 = {
        "file": "stream2.py",
        "start": 3,
        "end": 5,
        "name": "stream_blocks_alt",
        "kind": "compound_block",
    }

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        source_text1=code1,
        source_text2=code2,
    )
    assert helper != ""
    assert "yield from b" in helper
    assert "return total_count" in helper

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "(yield from _shared" in patch


def test_collect_downstream_read_names_nonlocal_and_global_declarations() -> None:
    """Verifies that downstream nonlocal declarations are captured as escaping free reads, while globals are excluded."""

    code_nonlocal = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        nonlocal x\n"
        "        return x\n"
        "    return inner\n"
    )
    u = {"start": 2, "end": 3}
    reads_nonlocal = collect_downstream_read_names(code_nonlocal, u, candidates={"x"})
    assert reads_nonlocal == {"x"}

    code_global = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        global x\n"
        "        return x\n"
        "    return inner\n"
    )
    reads_global = collect_downstream_read_names(code_global, u, candidates={"x"})
    # 'global x' accesses module global x, not outer's local x
    assert reads_global == set()

    # Pre-parsed tree parameter equivalence
    tree = ast.parse(code_nonlocal)
    reads_with_tree = collect_downstream_read_names(code_nonlocal, u, candidates={"x"}, tree=tree)
    assert reads_with_tree == {"x"}


def test_generator_clone_side_data_interface() -> None:
    """Verifies that GeneratorCloneSideData cleanly encapsulates per-side clone context."""
    side1 = GeneratorCloneSideData(
        outputs=["total", "x"],
        downstream={"total"},
        definite={"total"},
    )
    side2 = GeneratorCloneSideData(
        outputs=["count", "x"],
        downstream={"count"},
        definite={"count"},
    )

    resolved = resolve_generator_subroutine_outputs(side1, side2)
    assert resolved == (["total"], ["count"])

    # Fail closed on unequal output counts when counterpart is missing
    side_unpaired = GeneratorCloneSideData(
        outputs=["count", "extra", "status"],
        downstream={"count", "extra"},
        definite={"count", "extra"},
    )
    assert resolve_generator_subroutine_outputs(side1, side_unpaired) is None


def test_is_async_generator_with_return_value_policy() -> None:
    """Verifies that is_async_generator_with_return_value strictly enforces PEP 525 constraints."""
    # Clean sync generator with return value (allowed in Python 3.3+)
    sync_gen = {"has_yield": True, "is_async": False, "has_return_value": True}
    assert not is_async_generator_with_return_value(sync_gen)

    # Clean async function with return value (allowed)
    async_fn = {"has_yield": False, "is_async": True, "has_return_value": True}
    assert not is_async_generator_with_return_value(async_fn)

    # Async generator without return value (allowed)
    async_gen_no_ret = {"has_yield": True, "is_async": True, "has_return_value": False}
    assert not is_async_generator_with_return_value(async_gen_no_ret)

    # Async generator with explicit return value (illegal under PEP 525)
    async_gen_with_ret = {"has_yield": True, "is_async": True, "has_return_value": True}
    assert is_async_generator_with_return_value(async_gen_with_ret)

    # Multi-scope evaluation: if the merged helper combines yield, async, and return value, it is rejected
    assert is_async_generator_with_return_value(async_gen_no_ret, async_gen_with_ret)
    assert is_async_generator_with_return_value(async_gen_no_ret, sync_gen)

    # When no scope contains return values and has_outputs is False, not rejected
    clean_async = {"has_yield": False, "is_async": True, "has_return_value": False}
    sync_gen_no_ret = {"has_yield": True, "is_async": False, "has_return_value": False}
    assert not is_async_generator_with_return_value(clean_async, sync_gen_no_ret)

    # Async generator attempting to propagate downstream outputs (illegal under PEP 525)
    assert is_async_generator_with_return_value(async_gen_no_ret, has_outputs=True)
    assert not is_async_generator_with_return_value(sync_gen, has_outputs=True)


def test_downstream_read_visitor_aug_assign() -> None:
    """Verifies that augmented assignments downstream of a unit are captured as reads."""

    code = (
        "def compute_running_total(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    total += x\n"
        "    counts[0] += 1\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads == {"total", "x"}


def test_pair_clone_outputs_duplicate_names() -> None:
    """Verifies that _pair_clone_outputs does not raise KeyError on duplicate output names."""
    # Equal lengths with duplicates on one side (len(dedup) differs -> unequal arities fail closed)
    pairs1 = _pair_clone_outputs(["a", "b"], ["a", "a"])
    assert pairs1 == []

    # Equal raw lengths with duplicate on one side and disjoint names (returns empty without KeyError)
    pairs2 = _pair_clone_outputs(["x", "y"], ["z", "z"])
    assert pairs2 == []

    # Equal deduplicated lengths with duplicates on both sides
    pairs3 = _pair_clone_outputs(["a", "b", "a"], ["c", "d", "c"])
    assert pairs3 == [("a", "c"), ("b", "d")]

    # Unequal raw lengths with duplicates (unequal arities fail closed)
    pairs4 = _pair_clone_outputs(["a", "b", "a"], ["a", "a"])
    assert pairs4 == []


def test_extract_nested_scope_free_reads_type_annotations() -> None:
    """Verifies that type annotations on parameters and return types in nested functions are captured as free reads."""

    code = (
        "def outer():\n"
        "    class MyType:\n"
        "        pass\n"
        "    class ReturnType:\n"
        "        pass\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "    def nested(param: MyType) -> ReturnType:\n"
        "        return None\n"
    )
    unit = {"start": 6, "end": 7}
    reads = collect_downstream_read_names(code, unit, candidates={"MyType", "ReturnType", "x"})
    assert reads is not None
    assert "MyType" in reads
    assert "ReturnType" in reads


def test_pair_clone_outputs_positional_alignment() -> None:
    """Verifies that _pair_clone_outputs pairs outputs by position and fails closed on conflicts."""
    # Colliding variable names across different semantic positions fail closed
    pairs_conflict = _pair_clone_outputs(["a", "b"], ["b", "c"])
    assert pairs_conflict == []

    # Partial common names with conflicting indices fail closed to prevent variable swapping
    pairs_scrambled = _pair_clone_outputs(["a", "b", "c"], ["x", "a", "b"])
    assert pairs_scrambled == []

    # Common names at matching positions succeed with 1-to-1 positional pairing
    pairs_aligned = _pair_clone_outputs(["a", "b", "c"], ["a", "x", "c"])
    assert pairs_aligned == [("a", "a"), ("b", "x"), ("c", "c")]

    # Swapped variable names with identical name set fail closed to prevent miscompilation
    pairs_swapped = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_swapped == []


def test_extract_nested_scope_free_reads_outer_scope_shadowing() -> None:
    """Verifies that defaults, decorators, and annotations in nested functions are not wiped by inner local stores."""

    code = (
        "def outer():\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "    def inner_default(a=x):\n"
        "        x = 1\n"
        "        return a + x\n"
        "    def dec(fn):\n"
        "        return fn\n"
        "    @dec(x)\n"
        "    def inner_dec():\n"
        "        x = 2\n"
        "        return x\n"
        "    def inner_ann() -> x:\n"
        "        x = 3\n"
        "        return x\n"
        "    class InnerClass(x):\n"
        "        x = 4\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"x", "dec"})
    assert reads is not None
    assert "x" in reads


def test_downstream_reads_in_enclosing_class() -> None:
    """Verifies that collect_downstream_read_names traverses statements and methods in an enclosing class."""

    code = (
        "class ProcessingEngine:\n"
        "    x = 10\n"
        "    total = x * 2\n"
        "    downstream_attr = total + 5\n"
        "    def get_total(self):\n"
        "        return total\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads


def test_downstream_read_visitor_reaching_definitions_reassigned_variable() -> None:
    """Verifies that reaching definitions clear variables killed by unconditional assignments before reads."""

    # Case 1: Unconditional reassignment kills reaching definition
    code1 = (
        "def worker():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    x = 10\n"
        "    print(x)\n"
    )
    unit1 = {"start": 2, "end": 3}
    reads1 = collect_downstream_read_names(code1, unit1, candidates={"x", "y"})
    assert reads1 is not None
    assert "x" not in reads1
    assert "y" not in reads1

    # Case 2: Read in RHS before kill retains the read
    code2 = (
        "def worker():\n"
        "    x = 1\n"
        "    x = x + 10\n"
    )
    unit2 = {"start": 2, "end": 2}
    reads2 = collect_downstream_read_names(code2, unit2, candidates={"x"})
    assert reads2 is not None
    assert "x" in reads2

    # Case 3: Conditional If without else does NOT kill reaching definition
    code3 = (
        "def worker():\n"
        "    x = 1\n"
        "    if condition:\n"
        "        x = 10\n"
        "    print(x)\n"
    )
    unit3 = {"start": 2, "end": 2}
    reads3 = collect_downstream_read_names(code3, unit3, candidates={"x"})
    assert reads3 is not None
    assert "x" in reads3

    # Case 4: Conditional If with else in both branches kills reaching definition
    code4 = (
        "def worker():\n"
        "    x = 1\n"
        "    if condition:\n"
        "        x = 10\n"
        "    else:\n"
        "        x = 20\n"
        "    print(x)\n"
    )
    unit4 = {"start": 2, "end": 2}
    reads4 = collect_downstream_read_names(code4, unit4, candidates={"x"})
    assert reads4 is not None
    assert "x" not in reads4

    # Case 5: AnnAssign with value kills, but without value does not kill
    code5_with_val = (
        "def worker():\n"
        "    x = 1\n"
        "    x: int = 10\n"
        "    print(x)\n"
    )
    assert "x" not in collect_downstream_read_names(code5_with_val, {"start": 2, "end": 2}, candidates={"x"})  # type: ignore[operator]

    code5_no_val = (
        "def worker():\n"
        "    x = 1\n"
        "    x: int\n"
        "    print(x)\n"
    )
    assert "x" in collect_downstream_read_names(code5_no_val, {"start": 2, "end": 2}, candidates={"x"})  # type: ignore[operator]


def test_same_line_unit_boundaries_semicolon_downstream_resolution() -> None:
    """Verifies that same-line units without column offsets correctly resolve downstream statements after semicolons."""

    code = (
        "def runner():\n"
        "    yield x; print(total)\n"
    )
    # Unit on line 2 with NO start_col or end_col
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    # total is downstream of yield x on the same line
    assert "total" in reads
    # x is inside the unit, so it should NOT be reported as a downstream read
    assert "x" not in reads


def test_downstream_read_visitor_try_except_else_kills() -> None:
    """Verifies that exception handler kills do not leak into node.orelse in visit_Try."""

    code = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as total:\n"
        "        pass\n"
        "    else:\n"
        "        print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_read_visitor_try_finally_unconditional_kills() -> None:
    """Verifies that unconditional assignments in finally: kill reaching definitions downstream."""

    code_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        do_something()\n"
        "    finally:\n"
        "        total = 99\n"
        "    print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code_killed, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads

    code_retained = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        do_something()\n"
        "    finally:\n"
        "        pass\n"
        "    print(total)\n"
    )
    reads_retained = collect_downstream_read_names(code_retained, unit, candidates={"total"})
    assert reads_retained is not None
    assert "total" in reads_retained


def test_downstream_read_visitor_try_except_else_joint_kills() -> None:
    """Verifies that reaching definitions are killed only when all try/except/else branches assign."""

    # All branches assign total -> killed downstream
    code_all_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    except ValueError:\n"
        "        total = 2\n"
        "    except TypeError:\n"
        "        total = 3\n"
        "    print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code_all_killed, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads

    # One except branch does not assign total -> total reaches downstream
    code_partial = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    except ValueError:\n"
        "        pass\n"
        "    except TypeError:\n"
        "        total = 3\n"
        "    print(total)\n"
    )
    reads_partial = collect_downstream_read_names(code_partial, unit, candidates={"total"})
    assert reads_partial is not None
    assert "total" in reads_partial

    # Try body does not assign, but else and all except assign -> killed downstream
    code_else_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"
        "        total = 2\n"
        "    else:\n"
        "        total = 1\n"
        "    print(total)\n"
    )
    reads_else = collect_downstream_read_names(code_else_killed, unit, candidates={"total"})
    assert reads_else is not None
    assert "total" not in reads_else

    # Exception name is deleted at except block exit in Python 3 -> reaches downstream
    code_as_name = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as total:\n"
        "        pass\n"
        "    print(total)\n"
    )
    reads_as_name = collect_downstream_read_names(code_as_name, unit, candidates={"total"})
    assert reads_as_name is not None
    assert "total" in reads_as_name


def test_downstream_read_visitor_chained_with_context_managers() -> None:
    """Verifies that earlier context manager targets kill prior reaching definitions for later items."""

    code = (
        "def worker():\n"
        "    x = 10\n"
        "    with open_mgr() as x, use_mgr(x) as y:\n"
        "        pass\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"x"})
    assert reads is not None
    # x was killed by open_mgr() as x before use_mgr(x) was evaluated
    assert "x" not in reads

    # Also verify intra-item evaluation: open_mgr(x) evaluates BEFORE as x kills it
    code_intra = (
        "def worker():\n"
        "    x = 10\n"
        "    with open_mgr(x) as x:\n"
        "        pass\n"
    )
    reads_intra = collect_downstream_read_names(code_intra, unit, candidates={"x"})
    assert reads_intra is not None
    assert "x" in reads_intra


def test_class_scope_visitor_global_and_nonlocal() -> None:
    """Verifies that global declarations in class bodies are excluded from free reads and class stores."""
    code_global = (
        "class MyClass:\n"
        "    global g_val\n"
        "    g_val = 100\n"
        "    x = g_val\n"
        "    y = outer_read\n"
    )
    tree_g = ast.parse(code_global)
    class_node = tree_g.body[0]
    escaped_g = _extract_nested_scope_free_reads(class_node)  # type: ignore[arg-type]
    assert "g_val" not in escaped_g
    assert "outer_read" in escaped_g

    code_nonlocal = (
        "def outer():\n"
        "    n_val = 1\n"
        "    class Inner:\n"
        "        nonlocal n_val\n"
        "        n_val = 2\n"
        "        z = n_val\n"
    )
    tree_nl = ast.parse(code_nonlocal)
    func_node = tree_nl.body[0]
    inner_class_node = func_node.body[1]  # type: ignore[attr-defined]
    escaped_nl = _extract_nested_scope_free_reads(inner_class_node)  # type: ignore[arg-type]
    # n_val is nonlocal to Inner, so it escapes Inner as a free read referencing outer scope
    assert "n_val" in escaped_nl


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type_params syntax requires Python 3.12+")
def test_extract_nested_scope_free_reads_pep695_type_params() -> None:
    """Verifies that PEP 695 type parameter scopes do not treat type vars as free reads."""
    code = (
        "def inner[T: BoundType](val: T) -> T:\n"
        "    return val\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    escaped = _extract_nested_scope_free_reads(fn_node)  # type: ignore[arg-type]
    assert "T" not in escaped
    assert "BoundType" in escaped


def test_resolve_clone_generator_subroutine_outputs_precomputed_and_fallback(
    tmp_path: Path,
) -> None:
    """Verifies generator subroutine output resolution with precomputed outputs and fallback."""
    # Case 1: Precomputed outputs present on u1 and u2
    u1_pre = {"outputs": ["a", "b"], "source_text": "def f():\n    a = 1\n"}
    u2_pre = {"outputs": ["x", "y"], "source_text": "def g():\n    x = 1\n"}
    s1 = {"definite_stores": ["a"], "globals": ["b"], "nonlocals": []}
    s2 = {"definite_stores": ["x"], "globals": [], "nonlocals": ["y"]}
    res = resolve_clone_generator_subroutine_outputs(u1_pre, u2_pre, s1, s2)
    assert res is not None
    outs1, outs2 = res
    assert outs1 == ["a"]
    assert outs2 == ["x"]

    # Case 2: On-disk file resolution and downstream reads
    src1 = (
        "def gen1():\n"
        "    total = 0\n"
        "    yield total\n"
        "    print(total)\n"
    )
    src2 = (
        "def gen2():\n"
        "    total = 0\n"
        "    yield total\n"
        "    print(total)\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    unit1 = {"file": str(f1), "start": 2, "end": 3}
    unit2 = {"file": str(f2), "start": 2, "end": 3}
    scope1 = {"outputs": ["total"], "definite_stores": ["total"], "inputs": []}
    scope2 = {"outputs": ["total"], "definite_stores": ["total"], "inputs": []}
    res_disk = resolve_clone_generator_subroutine_outputs(
        unit1, unit2, scope1, scope2, repo_root=str(tmp_path)
    )
    assert res_disk is not None
    assert res_disk == (["total"], ["total"])

    # Case 3: Fail-closed counterpart mismatch on unequal arity with needed downstream output
    src3_extra = (
        "def gen1():\n"
        "    total = 0\n"
        "    extra = 1\n"
        "    yield total\n"
        "    print(total, extra)\n"
    )
    scope1_extra = {
        "outputs": ["total", "extra"],
        "definite_stores": ["total", "extra"],
        "inputs": [],
    }
    scope2_single = {"outputs": ["other"], "definite_stores": ["other"], "inputs": []}
    res_mismatch = resolve_clone_generator_subroutine_outputs(
        unit1, unit2, scope1_extra, scope2_single, source_text1=src3_extra, source_text2=src2
    )
    assert res_mismatch is None


def test_downstream_read_visitor_record_downstream_read_and_comprehensions() -> None:
    """Verifies that _record_downstream_read respects comprehension scoping and killed variables."""
    code = (
        "def runner():\n"
        "    data = [1, 2, 3]\n"
        "    res = [item for item in data]\n"
        "    def nested():\n"
        "        return [val for val in data]\n"
        "    class Helper:\n"
        "        stored = [elem for elem in data]\n"
        "    total = 0\n"
        "    total += 5\n"
        "    return total\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(
        code, unit, candidates={"data", "item", "val", "elem", "total"}
    )
    assert reads is not None
    assert "data" in reads
    assert "item" not in reads
    assert "val" not in reads
    assert "elem" not in reads
    assert "total" not in reads


def test_class_body_comprehension_free_reads() -> None:
    """Verifies that class-body comprehensions do not resolve class attributes (PEP 227)."""
    code = (
        "def outer():\n"
        "    x = 10\n"
        "    class C:\n"
        "        x = 100\n"
        "        values = [x for _ in range(1)]\n"
        "    return C\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"x"})
    assert reads is not None
    assert "x" in reads


def test_conditional_named_expr_does_not_kill_variables() -> None:
    """Verifies that conditionally evaluated named expressions do not kill reaching definitions."""
    code = (
        "def compute(flag, items):\n"
        "    total = 10\n"
        "    if flag and (total := 20):\n"
        "        pass\n"
        "    return total\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_pair_clone_outputs_swapped_names_and_positional_roles() -> None:
    """Verifies positional pairing across renamed variable roles and identity on permutations."""
    # Equal arity with different names: preserves positional sequence
    pairs_pos = _pair_clone_outputs(["x", "y"], ["b", "a"])
    assert pairs_pos == [("x", "b"), ("y", "a")]

    # Identical name sets with conflicting order fail closed to prevent miscompilation
    pairs_ident = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_ident == []

    # Mismatched arity fails closed
    pairs_mismatched = _pair_clone_outputs(["a", "b"], ["b"])
    assert pairs_mismatched == []


def test_except_handler_name_not_treated_as_escaping_free_read() -> None:
    """Verifies ast.ExceptHandler.name is recorded as local store and not an escaping read."""
    # Nested function with except handler
    func_src = (
        "def nested():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as err:\n"
        "        return str(err)\n"
    )
    node = ast.parse(func_src).body[0]
    assert isinstance(node, ast.FunctionDef)
    free = _extract_nested_scope_free_reads(node)
    assert "err" not in free

    # Nested class with except handler in body
    cls_src = (
        "class NestedClass:\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as err:\n"
        "        err_msg = str(err)\n"
    )
    cls_node = ast.parse(cls_src).body[0]
    assert isinstance(cls_node, ast.ClassDef)
    cls_free = _extract_nested_scope_free_reads(cls_node)
    assert "err" not in cls_free


def test_resolve_clone_generator_subroutine_outputs_requires_both_outputs() -> None:
    """Verifies resolve_clone_generator_subroutine_outputs requires both sides to have outputs."""
    u1 = {"outputs": ["a"], "start": 2, "end": 2}
    u2 = {"start": 2, "end": 2}  # No precomputed outputs
    scope1 = {"outputs": ["a"], "definite_stores": ["a"], "inputs": []}
    scope2 = {"outputs": ["b"], "definite_stores": ["b"], "inputs": []}

    src1 = "def f1():\n    a = 1\n    return a\n"
    src2 = "def f2():\n    b = 2\n    return b\n"
    res = resolve_clone_generator_subroutine_outputs(
        u1, u2, scope1, scope2, source_text1=src1, source_text2=src2
    )
    assert res is not None
    outs1, outs2 = res
    assert outs1 == ["a"]
    assert outs2 == ["b"]


def test_downstream_reads_preserve_exception_alias_reassigned_in_same_handler() -> None:
    """Verifies reassigning an exception alias in handler preserves downstream reads."""
    code = (
        "def process():\n"
        "    try:\n"
        "        do_work()\n"
        "    except ValueError as err:\n"
        "        err = 'custom: ' + str(err)\n"
        "        log(err)\n"
    )
    # Unit is line 5: "err = 'custom: ' + str(err)"
    unit = {"start": 5, "end": 5}
    reads = collect_downstream_read_names(code, unit, candidates={"err"})
    assert reads is not None
    assert "err" in reads

    # Conversely, if unit is inside try body, the subsequent except handler DOES kill err
    code_try = (
        "def process_try():\n"
        "    try:\n"
        "        err = 'initial'\n"
        "    except ValueError as err:\n"
        "        log(err)\n"
    )
    unit_try = {"start": 3, "end": 3}
    reads_try = collect_downstream_read_names(code_try, unit_try, candidates={"err"})
    assert reads_try is not None
    assert "err" not in reads_try


def test_collect_downstream_read_names_loop_carried_dependence() -> None:
    """Verifies that reads earlier in an enclosing loop body are captured as downstream reads."""
    code = (
        "def process_batches(batches):\n"
        "    total = 0\n"
        "    for batch in batches:\n"
        "        log(total)\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield x\n"
    )
    # Unit is lines 5 to 7: inner loop where total is updated
    unit = {"start": 5, "end": 7}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_collect_downstream_read_names_try_finally_exceptional_path() -> None:
    """Verifies that try body stores do not kill variables in finally block across
    exceptional paths."""
    code = (
        "def run_transaction():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    finally:\n"
        "        print(total)\n"
    )
    # Unit is line 2: "total = 10"
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    # print(total) in finally could run after an exception in try before total = 1,
    # so line 2's total must be considered read downstream.
    assert "total" in reads


def test_downstream_reads_pre_unit_loop_assign_does_not_kill_post_unit() -> None:
    """Verifies that pre-unit assignments in enclosing loops do not kill post-unit reads."""
    src = (
        "for batch in batches:\n"
        "    total = 0\n"
        "    for x in batch:\n"
        "        total += x\n"
        "        yield x\n"
        "    print(total)\n"
    )
    unit = {"file": "mod.py", "start": 3, "end": 5, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_recursive_closure_no_infinite_recursion() -> None:
    """Verifies that recursive pre-unit closures do not cause infinite recursion."""
    src = (
        "def helper(n):\n"
        "    return helper(n - 1) if n else 0\n"
        "\n"
        "x = 1\n"
        "y = 2\n"
        "helper(3)\n"
    )
    unit = {"file": "mod.py", "start": 4, "end": 5, "name": "unit", "kind": "block"}
    reads = collect_downstream_read_names(src, unit, candidates={"helper", "x", "y"})
    assert reads is not None
    assert "helper" in reads


def test_downstream_reads_mutually_recursive_closures_no_recursion_error() -> None:
    """Verifies that mutually recursive pre-unit closures do not cause recursion errors.

    Under the blanket pre-unit closure rule, all pre-unit closures' free variables are
    unioned directly into the candidate read set. Asserting both 'foo' and 'bar' proves
    that mutual definitions are safely traversed without infinite recursion, though
    reachability is satisfied directly by the union rather than a transitive walk.
    """
    src = (
        "def foo(n):\n"
        "    return bar(n - 1) if n else 0\n"
        "def bar(n):\n"
        "    return foo(n - 1) if n else 0\n"
        "\n"
        "x = 1\n"
        "y = 2\n"
        "foo(5)\n"
    )
    unit = {"file": "mod.py", "start": 6, "end": 7, "name": "unit", "kind": "block"}
    reads = collect_downstream_read_names(src, unit, candidates={"foo", "bar", "x", "y"})
    assert reads is not None
    assert "foo" in reads
    assert "bar" in reads


def test_downstream_reads_named_expr_target_is_not_recorded_as_read() -> None:
    """Verifies that walrus expression targets in downstream code are not recorded as reads."""
    src = (
        "z = 1\n"
        "if (z := 42):\n"
        "    pass\n"
    )
    unit = {"file": "mod.py", "start": 1, "end": 1, "name": "unit", "kind": "block"}
    reads = collect_downstream_read_names(src, unit, candidates={"z"})
    assert reads is not None
    assert "z" not in reads


def test_downstream_reads_del_bare_name_is_read_not_killed() -> None:
    """Verifies that bare 'del name' downstream is counted as a read and not added to killed."""
    src = (
        "def fn(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    del total\n"
    )
    unit = {"file": "mod.py", "start": 3, "end": 5, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_pre_unit_def_in_if_block() -> None:
    """Verifies that pre-unit closures defined in nested blocks (if, try, for) are discovered."""
    src = (
        "def fn(cond, items):\n"
        "    if cond:\n"
        "        def helper():\n"
        "            return total\n"
        "    for x in items:\n"
        "        total = x * 2\n"
        "        yield x\n"
        "    helper()\n"
    )
    unit = {"file": "mod.py", "start": 5, "end": 7, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_pre_unit_lambda_assignment() -> None:
    """Verifies that pre-unit lambda assignments (cb = lambda: total) have their captured
    reads preserved under the blanket pre-unit rule without requiring alias tracking or
    downstream call resolution."""
    src = (
        "def fn(items):\n"
        "    cb = lambda: total\n"
        "    for x in items:\n"
        "        total = x * 2\n"
        "        yield x\n"
        "    cb()\n"
    )
    unit = {"file": "mod.py", "start": 3, "end": 5, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_pre_unit_loop_assign_inside_def_exercises_innermost_node() -> None:
    """Verifies that pre-unit loop assignments inside a function do not kill post-unit reads,
    exercising _find_innermost_enclosing_node."""
    src = (
        "def process(batches):\n"
        "    for batch in batches:\n"
        "        total = 0\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield x\n"
        "        print(total)\n"
    )
    unit = {"file": "mod.py", "start": 4, "end": 6, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_class_method_deferred_read() -> None:
    """Verifies that pre-unit class methods capturing candidate variables have their free
    reads preserved directly under the blanket pre-unit rule."""
    src = (
        "class K:\n"
        "    def m(self):\n"
        "        return total\n"
        "\n"
        "for x in items:\n"
        "    total = x\n"
        "    yield x\n"
        "K().m()\n"
    )
    unit = {"file": "mod.py", "start": 5, "end": 7, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_class_instance_alias_deferred_read() -> None:
    """Verifies that pre-unit class methods capturing candidate variables are preserved
    by the blanket pre-unit rule regardless of instance aliasing (obj = K())."""
    src = (
        "class K:\n"
        "    def m(self):\n"
        "        return total\n"
        "\n"
        "obj = K()\n"
        "for x in items:\n"
        "    total = x\n"
        "    yield x\n"
        "obj.m()\n"
    )
    unit = {"file": "mod.py", "start": 6, "end": 8, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_pre_unit_closure_redefinition_in_branches() -> None:
    """Verifies that closures redefined across conditional branches merge free variable sets."""
    src = (
        "if flag:\n"
        "    def h():\n"
        "        return total\n"
        "else:\n"
        "    def h():\n"
        "        return other\n"
        "\n"
        "for x in items:\n"
        "    total = x\n"
        "    yield x\n"
        "h()\n"
    )
    unit = {"file": "mod.py", "start": 8, "end": 10, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total", "other"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_escaping_closure_registered_before_unit() -> None:
    """Verifies that pre-unit closures capturing candidate variables are treated as live
    even when invoked indirectly through pre-registered handlers."""
    src = (
        "def cb():\n"
        "    return total\n"
        "\n"
        "register(cb)\n"
        "for x in items:\n"
        "    total = x\n"
        "    yield x\n"
        "fire()\n"
    )
    unit = {"file": "mod.py", "start": 5, "end": 7, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_compound_statement_lambda_after_unit_not_pre_captured() -> None:
    """Verifies that lambdas occurring after u_start inside an enclosing compound statement
    are not prematurely captured into pre_unit_captured_reads."""
    src = (
        "if condition:\n"
        "    a = 1\n"
        "    for x in items:\n"
        "        fn = lambda: total\n"
        "        total = x * 2\n"
        "        yield x\n"
    )
    unit = {"file": "mod.py", "start": 3, "end": 6, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads


def test_collect_downstream_read_names_inverted_coordinates() -> None:
    """Verifies that inverted unit coordinates (start > end) safely return None."""
    src = "x = 1\ny = 2\nz = x + y\n"
    unit = {"file": "mod.py", "start": 3, "end": 1, "name": "inverted"}
    reads = collect_downstream_read_names(src, unit, candidates={"x", "y", "z"})
    assert reads is None


def test_resolve_clone_generator_subroutine_outputs_duplicate_names_fails_closed() -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs returns ([], []) when outputs
    are unneeded downstream, and fails closed (returns None) when downstream reads outputs and
    precomputed outputs contain duplicate names causing arity collapse."""
    u1 = {"file": "mod1.py", "start": 1, "end": 2, "outputs": ["a", "b"]}
    u2 = {"file": "mod2.py", "start": 1, "end": 2, "outputs": ["c", "c"]}
    scope1 = {"inputs": [], "definite_stores": ["a", "b"]}
    scope2 = {"inputs": [], "definite_stores": ["c"]}
    res_unneeded = resolve_clone_generator_subroutine_outputs(
        u1, u2, scope1, scope2, source_text1="yield 1", source_text2="yield 1"
    )
    assert res_unneeded == ([], [])

    res_needed = resolve_clone_generator_subroutine_outputs(
        u1,
        u2,
        scope1,
        scope2,
        source_text1="yield 1\nyield 2\nprint(a, b)",
        source_text2="yield 1\nyield 2\nprint(c)",
    )
    assert res_needed is None


def test_load_unit_file_text_source_lines_fallback() -> None:
    """Verifies that _load_unit_file_text falls back to in-memory source_lines
    when file is not on disk."""
    unit = {
        "file": "virtual/unwritten.py",
        "start": 1,
        "end": 2,
        "source_lines": ["def foo():\n", "    pass\n"],
        "source_lines_is_sliced": False,
    }
    loaded = _load_unit_file_text(unit)
    assert loaded == "def foo():\n    pass\n"


def test_extract_nested_scope_free_reads_class_assigned_nonlocal() -> None:
    """Verifies that _extract_nested_scope_free_reads unions assigned nonlocal variables
    in class bodies into escaped free reads for parity with function scopes."""
    code = (
        "class C:\n"
        "    nonlocal captured_var\n"
        "    captured_var = 42\n"
    )
    class_node = ast.parse(code).body[0]
    assert isinstance(class_node, ast.ClassDef)
    free_reads = _extract_nested_scope_free_reads(class_node)
    assert "captured_var" in free_reads


def test_load_unit_file_text_prioritizes_disk_file_over_sliced_source_lines(
    tmp_path: Path,
) -> None:
    """Verifies that _load_unit_file_text prefers reading the full file from disk when it exists,
    falling back to sliced source_lines only when the file is not on disk."""
    full_file = tmp_path / "worker.py"
    disk_content = (
        "async def worker():\n"
        "    for i in range(10):\n"
        "        yield i\n"
    )
    full_file.write_text(disk_content, encoding="utf-8")
    unit = {
        "file": str(full_file),
        "start": 2,
        "end": 3,
        "source_lines": ["    for i in range(10):\n", "        yield i\n"],
    }
    loaded = _load_unit_file_text(unit, repo_root=str(tmp_path))
    assert loaded == disk_content


def test_load_unit_file_text_prioritizes_repo_root_over_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that _load_unit_file_text checks repo_root before cwd for relative paths,
    avoiding path shadowing when the process working directory is not the repository root."""
    cwd_dir = tmp_path / "cwd_dir"
    repo_dir = tmp_path / "repo_dir"
    cwd_dir.mkdir()
    repo_dir.mkdir()

    (cwd_dir / "mod.py").write_text("# cwd version\n", encoding="utf-8")
    (repo_dir / "mod.py").write_text("# repo version\n", encoding="utf-8")

    monkeypatch.chdir(cwd_dir)
    unit = {"file": "mod.py", "start": 1, "end": 1}
    # When repo_root is specified, the file in repo_root must take precedence over cwd
    text_repo = _load_unit_file_text(unit, repo_root=str(repo_dir))
    assert text_repo == "# repo version\n"

    # When repo_root is not specified, it falls back to cwd
    text_cwd = _load_unit_file_text(unit)
    assert text_cwd == "# cwd version\n"


def test_load_unit_file_text_explicit_empty_string_source_text(tmp_path: Path) -> None:
    """Verifies that an explicit empty string in source_text or file_source does not fall
    through to disk."""
    f = tmp_path / "on_disk.py"
    f.write_text("# disk content\n", encoding="utf-8")

    # Explicit empty string in source_text must return "" and not read disk
    unit_empty = {"file": str(f), "source_text": ""}
    assert _load_unit_file_text(unit_empty) == ""

    # Explicit empty string in file_source when source_text is None
    unit_empty_file_source = {"file": str(f), "file_source": ""}
    assert _load_unit_file_text(unit_empty_file_source) == ""


def test_loop_orelse_reaching_definitions_for_and_while() -> None:
    """Verifies that reads in loop orelse blocks see reaching definitions entering the loop.

    When the loop body does not execute (e.g. empty sequence for For, or immediately False
    condition for While), assignments inside the loop body must not mask reaching definitions
    needed by the orelse clause.
    """
    src_for = (
        "def f(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total = 10\n"
        "    else:\n"
        "        print(total)\n"
    )
    unit_for = {"start": 2, "end": 2}
    reads_for = collect_downstream_read_names(src_for, unit_for, candidates={"total"})
    assert reads_for == {"total"}

    src_while = (
        "def f(cond):\n"
        "    total = 0\n"
        "    while cond:\n"
        "        total = 10\n"
        "    else:\n"
        "        print(total)\n"
    )
    unit_while = {"start": 2, "end": 2}
    reads_while = collect_downstream_read_names(src_while, unit_while, candidates={"total"})
    assert reads_while == {"total"}


def test_module_level_unit_prior_function_free_reads() -> None:
    """Verifies that module-level units treat free reads from prior top-level functions as live."""
    src = (
        "def log_value():\n"
        "    return g_total\n"
        "\n"
        "g_total = 100\n"
        "print('done')\n"
    )
    unit = {"start": 4, "end": 5}
    reads = collect_downstream_read_names(src, unit, candidates={"g_total"})
    assert reads == {"g_total"}


def test_collect_downstream_read_names_empty_candidates_early_out() -> None:
    """Verifies that empty candidates set triggers early-out returning empty set."""
    src = "x = 1\ny = 2\n"
    unit = {"start": 1, "end": 1}
    assert collect_downstream_read_names(src, unit, candidates=set()) == set()


def test_collect_downstream_read_names_skip_pre_unit_closures() -> None:
    """Verifies that skip_pre_unit_closures=True bypasses pre-unit closure read extraction."""
    src = (
        "def outer():\n"
        "    cb = lambda: val\n"
        "    val = 10\n"
        "    print('done')\n"
    )
    unit = {"start": 3, "end": 3}
    reads_default = collect_downstream_read_names(src, unit, candidates={"val"})
    assert reads_default == {"val"}

    reads_skipped = collect_downstream_read_names(
        src, unit, candidates={"val"}, skip_pre_unit_closures=True
    )
    assert reads_skipped == set()


def test_extract_nested_scope_free_reads_pure_ast_immutability() -> None:
    """Verifies that _extract_nested_scope_free_reads does not mutate AST nodes."""
    code = (
        "def compute(a):\n"
        "    return a + b + c\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    free_reads = _extract_nested_scope_free_reads(fn_node)  # type: ignore[arg-type]
    assert free_reads == {"b", "c"}
    assert not hasattr(fn_node, "_free_reads_cache")


def test_extract_nested_scope_free_reads_generator_scope_isolation() -> None:
    """Verifies that generator expressions isolate inner comprehensions and lambdas
    without leaking loop targets or inner parameters as free reads."""
    code1 = "([z for z in row] for row in matrix)"
    gen_node1 = ast.parse(code1, mode="eval").body
    free1 = _extract_nested_scope_free_reads(gen_node1)  # type: ignore[arg-type]
    assert free1 == {"matrix"}

    code2 = "(((lambda p: p + offset)(item)) for item in items)"
    gen_node2 = ast.parse(code2, mode="eval").body
    free2 = _extract_nested_scope_free_reads(gen_node2)  # type: ignore[arg-type]
    assert free2 == {"items", "offset"}


def test_collect_downstream_read_names_lru_caching() -> None:
    """Verifies that collect_downstream_read_names uses an LRU cache across repeated calls."""
    _clear_downstream_reads_cache()
    code = (
        "def run():\n"
        "    cb = lambda: extra\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "        extra = x\n"
        "    print(extra)\n"
    )
    tree = ast.parse(code)
    unit = {"file": "mod.py", "start": 3, "end": 5}
    reads1 = collect_downstream_read_names(code, unit, candidates={"extra"}, tree=tree)
    assert reads1 == {"extra"}

    # Subsequent call hits the cache and returns a fresh copy of the set
    reads2 = collect_downstream_read_names(code, unit, candidates={"extra"}, tree=tree)
    assert reads2 == {"extra"}
    assert reads1 is not reads2

    reads1.add("mutated")
    reads3 = collect_downstream_read_names(code, unit, candidates={"extra"}, tree=tree)
    assert reads3 == {"extra"}


def test_collect_pre_unit_closures_deeply_nested_compound_statements() -> None:
    """Verifies that _collect_pre_unit_closures discovers pre-unit closures across nested
    compound statements (if/for/try/with) without missing headers or corrupting scopes."""
    from pydoppelgangerhunt.fixer.dataflow import _collect_pre_unit_closures

    code = (
        "if (lambda: cond_captured)():\n"
        "    for item in (x for x in iter_captured):\n"
        "        try:\n"
        "            with ctx_captured as c:\n"
        "                cb = lambda: inner_captured\n"
        "        except Exception:\n"
        "            handler_cb = lambda: err_captured\n"
        "# Line 8: Unit starts here\n"
        "result = 123\n"
    )
    tree = ast.parse(code)
    captured = _collect_pre_unit_closures(tree, u_start=8)
    expected = {
        "cond_captured",
        "iter_captured",
        "inner_captured",
        "err_captured",
    }
    assert expected.issubset(captured)


def test_collect_downstream_read_names_source_digest_optimization() -> None:
    """Verifies that collect_downstream_read_names uses precomputed source_digest in cache key."""
    from pydoppelgangerhunt.fixer.dataflow import (
        _clear_downstream_reads_cache,
        collect_downstream_read_names,
    )

    _clear_downstream_reads_cache()
    code = (
        "def run():\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "        extra = x\n"
        "    print(extra)\n"
    )
    unit = {"file": "mod.py", "start": 3, "end": 4}
    digest = "fixed_hex_digest"

    reads = collect_downstream_read_names(
        code, unit, candidates={"extra"}, source_digest=digest
    )
    assert reads == {"extra"}

    cached_reads = collect_downstream_read_names(
        code, unit, candidates={"extra"}, source_digest=digest
    )
    assert cached_reads == {"extra"}


def test_collect_downstream_read_names_malformed_coordinates() -> None:
    """Verifies that collect_downstream_read_names handles malformed coordinates gracefully."""
    assert collect_downstream_read_names("x = 1\n", {"start": "invalid"}) is None


def test_extract_effective_unit_outputs_set_handling() -> None:
    """Verifies that multi-item set outputs are rejected to prevent unpositional pairing,
    while single-item sets are supported."""
    assert _extract_effective_unit_outputs({"outputs": {"z", "a", "m"}}, {}) == []
    assert _extract_effective_unit_outputs({"outputs": {"single"}}, {}) == ["single"]


def test_collect_downstream_read_names_cache_isolation_on_content_change() -> None:
    """Verifies that downstream read cache isolates entries when source content changes."""
    _clear_downstream_reads_cache()
    code1 = "def f():\n    x = 1\n    return x\n"
    code2 = "def f():\n    y = 2\n    return y\n"
    unit = {"file": "f.py", "start": 2, "end": 2}
    reads1 = collect_downstream_read_names(code1, unit)
    reads2 = collect_downstream_read_names(code2, unit)
    assert reads1 == {"x"}
    assert reads2 == {"y"}


def test_load_unit_file_text_resilience_to_non_utf8_bytes(tmp_path: Path) -> None:
    """Verifies that _load_unit_file_text uses errors='replace' on non-UTF-8 bytes
    and enforces containment."""
    latin1_file = tmp_path / "latin1.py"
    latin1_file.write_bytes(b"def compute():\n    # Legacy comment \xe9\n    return 42\n")
    loaded = _load_unit_file_text({"file": str(latin1_file)}, repo_root=str(tmp_path))
    assert loaded is not None
    assert "return 42" in loaded
    assert "\ufffd" in loaded

    # When repo_root is omitted, files in global tempdir outside CWD are rejected fail-closed
    assert _load_unit_file_text({"file": str(latin1_file)}, repo_root=None) is None


def test_collect_downstream_read_names_mtime_independent_cache_sharing() -> None:
    """Verifies that differing mtimes for identical content share cache entries via digest
    deduplication."""
    _clear_downstream_reads_cache()
    code = "def f():\n    x = 1\n    return x\n"
    unit1 = {"file": "f.py", "start": 2, "end": 2, "mtime": 100}
    unit2 = {"file": "f.py", "start": 2, "end": 2, "mtime": 200}
    reads1 = collect_downstream_read_names(code, unit1)
    reads2 = collect_downstream_read_names(code, unit2)
    assert reads1 == {"x"}
    assert reads2 == {"x"}
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        _downstream_reads_cache,
    )
    # Content digest deduplication: identical content with differing timestamps shares the cache
    assert len(_downstream_reads_cache) == 1


def test_collect_downstream_read_names_digest_invalidation(tmp_path: Path) -> None:
    """Verifies that differing content digest produces distinct cache entries for
    downstream reads."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        _downstream_reads_cache,
    )

    _clear_downstream_reads_cache()
    f_path = tmp_path / "target.py"
    code = "def f():\n    x = 1\n    return x\n"
    f_path.write_text(code, encoding="utf-8")
    unit1 = {"file": str(f_path), "start": 2, "end": 2, "source_digest": "hash1"}
    reads1 = collect_downstream_read_names(code, unit1)
    assert reads1 == {"x"}
    assert len(_downstream_reads_cache) == 1

    unit2 = {"file": str(f_path), "start": 2, "end": 2, "source_digest": "hash2"}
    reads2 = collect_downstream_read_names(code, unit2)
    assert reads2 == {"x"}
    assert len(_downstream_reads_cache) == 2


def test_collect_downstream_read_names_no_disk_stat_call() -> None:
    """Verifies that collect_downstream_read_names avoids redundant disk stat() calls."""
    _clear_downstream_reads_cache()
    code = "def f():\n    x = 1\n    return x\n"
    unit = {"file": "virtual_module.py", "start": 2, "end": 2}
    with mock.patch.object(Path, "stat") as mock_stat:
        reads = collect_downstream_read_names(code, unit)
        assert reads == {"x"}
        mock_stat.assert_not_called()


def test_resolve_clone_generator_subroutine_outputs_precomputed_definite_stores() -> None:
    """Verifies precomputed outputs are validated against definite stores when available."""
    u1 = {"outputs": ["x", "unassigned"], "source_text": "def f():\n    x = 1\n"}
    u2 = {"outputs": ["a", "b"], "source_text": "def g():\n    a = 1\n"}
    # Scope 1 only definitely assigns x, leaving unassigned lacking definite store
    s1 = {"definite_stores": ["x"], "inputs": []}
    s2 = {"definite_stores": ["a", "b"], "inputs": []}
    res_rejected = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res_rejected is None

    # When both sides definitely assign all precomputed outputs, pairing succeeds
    s1_valid = {"definite_stores": ["x", "unassigned"], "inputs": []}
    res_valid = resolve_clone_generator_subroutine_outputs(u1, u2, s1_valid, s2)
    assert res_valid == (["x", "unassigned"], ["a", "b"])


def test_resolve_clone_generator_subroutine_outputs_inverted_and_zero_coords() -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs safely clamps inverted
    or zero coordinates when inspecting enclosing try cleanup."""
    code1 = (
        "try:\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "finally:\n"
        "    cleanup = total\n"
    )
    code2 = (
        "try:\n"
        "    for y in range(10):\n"
        "        yield y\n"
        "finally:\n"
        "    cleanup = count\n"
    )
    u1 = {
        "file": "mod1.py",
        "start": 3,
        "end": 2,
        "source_text": code1,
        "outputs": ["total"],
    }
    u2 = {
        "file": "mod2.py",
        "start": 0,
        "end": 0,
        "source_text": code2,
        "outputs": ["count"],
    }
    s1 = {"definite_stores": ["total"], "inputs": []}
    s2 = {"definite_stores": ["count"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_load_unit_file_text_honors_sliced_lines() -> None:
    """Verifies that _load_unit_file_text returns None when source_lines is marked as sliced."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        _load_unit_file_text,
    )

    unit_sliced = {
        "file": "virtual.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = 1\n", "    y = 2\n", "    z = 3\n"],
        "source_lines_is_sliced": True,
    }
    assert _load_unit_file_text(unit_sliced) is None

    unit_heuristic = {
        "file": "virtual.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = 1\n", "    y = 2\n", "    z = 3\n"],
    }
    assert _load_unit_file_text(unit_heuristic) is None

    unit_full = {
        "file": "virtual.py",
        "start": 1,
        "end": 3,
        "source_lines": ["x = 1\n", "y = 2\n", "z = 3\n"],
        "source_lines_is_sliced": False,
    }
    assert _load_unit_file_text(unit_full) == "x = 1\ny = 2\nz = 3\n"

    unit_eof = {
        "file": "virtual_eof.py",
        "start": 2,
        "end": 4,
        "source_lines": ["a\n", "b\n", "c\n", "d\n"],
    }
    assert _load_unit_file_text(unit_eof) == "a\nb\nc\nd\n"

    unit_no_newlines = {
        "file": "virtual_no_nl.py",
        "start": 1,
        "end": 2,
        "source_lines": ["x = 1", "y = 2", "z = 3"],
        "source_lines_is_sliced": False,
    }
    assert _load_unit_file_text(unit_no_newlines) == "x = 1\ny = 2\nz = 3\n"


def test_downstream_reads_non_ascii_same_line_semicolon() -> None:
    """Verifies downstream read analysis with non-ASCII characters preceding same-line unit."""
    code = (
        "def f():\n"
        "    tag = 'café'; total = 1; print(total)\n"
    )
    unit = {
        "file": "test_ascii.py",
        "start": 2,
        "end": 2,
        "start_col": 19,
        "end_col": 28,
        "kind": "compound_block",
    }
    reads = collect_downstream_read_names(code, unit, candidates={"total", "tag"})
    assert reads == {"total"}

    # Fallback end_col resolution without end_col in unit
    unit_no_end_col = {
        "file": "test_ascii.py",
        "start": 2,
        "end": 2,
        "start_col": 19,
        "kind": "compound_block",
    }
    reads_fallback = collect_downstream_read_names(
        code, unit_no_end_col, candidates={"total", "tag"}
    )
    assert reads_fallback == {"total"}


def test_find_enclosing_loops_explicit_none_end_lineno() -> None:
    """Verifies that _find_enclosing_loops handles AST nodes with end_lineno explicitly None."""
    code = (
        "for i in range(10):\n"
        "    x = i\n"
    )
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Explicitly set end_lineno attribute to None
            node.end_lineno = None  # type: ignore[assignment]
    # Should not raise TypeError when end_lineno is None
    loops = _find_enclosing_loops(tree, u_start=2, u_end=2)
    assert loops == []


def test_downstream_cache_update_prevents_premature_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that updating an existing key in _downstream_reads_cache does not evict,
    and exceeding capacity evicts the oldest item."""
    monkeypatch.setattr("pydoppelgangerhunt.fixer.dataflow._MAX_DOWNSTREAM_CACHE_SIZE", 5)
    _clear_downstream_reads_cache()
    try:
        with _downstream_cache_lock:
            # Fill cache with 5 items
            for i in range(5):
                _downstream_reads_cache[(f"key_{i}",)] = frozenset([f"var_{i}"])

        # Create a unit and code to populate cache
        code = (
            "def worker():\n"
            "    total = 0\n"
            "    yield total\n"
            "    print(total)\n"
        )
        unit = {"file": "worker.py", "start": 3, "end": 3}
        # First call: adds a new key when cache is at capacity, evicting key_0
        reads1 = collect_downstream_read_names(code, unit, candidates={"total"})
        assert reads1 == {"total"}

        with _downstream_cache_lock:
            assert len(_downstream_reads_cache) == 5
            assert ("key_0",) not in _downstream_reads_cache
            keys_before = list(_downstream_reads_cache.keys())

        # Calling again on the same unit updates/hits without shrinking or premature eviction
        reads2 = collect_downstream_read_names(code, unit, candidates={"total"})
        assert reads2 == {"total"}

        with _downstream_cache_lock:
            assert len(_downstream_reads_cache) == 5
            # Most recently updated/accessed key moved to the end
            assert list(_downstream_reads_cache.keys())[-1] == keys_before[-1]
    finally:
        _clear_downstream_reads_cache()


def test_resolve_clone_generator_subroutine_outputs_missing_source_fails_closed() -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs fails closed when outputs
    are needed but source text and AST trees are unavailable to verify try-cleanup safety."""
    u1 = {"outputs": ["x"]}
    u2 = {"outputs": ["a"]}
    s1 = {"definite_stores": ["x"], "inputs": []}
    s2 = {"definite_stores": ["a"], "inputs": []}
    # No source_text, file, or tree is provided
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_enclosing_try_reads_outputs_in_else_and_except_finally_reads() -> None:
    """Verifies that _enclosing_try_reads_outputs flags units in try/else/except when
    finally reads outputs."""
    # 1. Unit inside else: block whose finally reads outputs
    code_else = (
        "def run_else(items):\n"
        "    try:\n"
        "        open_res()\n"
        "    except Exception:\n"
        "        handle_err()\n"
        "    else:\n"
        "        for x in items:\n"
        "            yield x\n"
        "    finally:\n"
        "        cleanup(out_var)\n"
    )
    tree_else = ast.parse(code_else)
    scope_else = tree_else.body[0]
    # Lines 7 to 8 are the unit inside else:
    assert _enclosing_try_reads_outputs(scope_else, 7, 8, {"out_var"}) is True
    # If finally does not read the output, returns False
    assert _enclosing_try_reads_outputs(scope_else, 7, 8, {"other_var"}) is False

    # 2. Unit inside except: block whose finally reads outputs
    code_except = (
        "def run_except(fallback):\n"
        "    try:\n"
        "        open_res()\n"
        "    except IOError:\n"
        "        for x in fallback:\n"
        "            yield x\n"
        "    finally:\n"
        "        cleanup(out_var)\n"
    )
    tree_except = ast.parse(code_except)
    scope_except = tree_except.body[0]
    # Lines 5 to 6 are the unit inside except:
    assert _enclosing_try_reads_outputs(scope_except, 5, 6, {"out_var"}) is True
    assert _enclosing_try_reads_outputs(scope_except, 5, 6, {"other_var"}) is False

    # 3. Unit inside else: rejected by resolve_clone_generator_subroutine_outputs
    u1 = {
        "file": "m1.py",
        "start": 7,
        "end": 8,
        "source_text": code_else,
        "outputs": ["out_var"],
    }
    u2 = {
        "file": "m2.py",
        "start": 7,
        "end": 8,
        "source_text": code_else,
        "outputs": ["out_var"],
    }
    s1 = {"definite_stores": ["out_var"], "inputs": []}
    s2 = {"definite_stores": ["out_var"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_downstream_read_visitor_trailing_same_line_without_col() -> None:
    """Verifies that trailing nodes on the same line are detected as downstream when
    u_end_col is None."""
    code = "x = 1; y = x + 1"
    tree = ast.parse(code)
    # Unit occupies line 1, end line 1, u_end_col is None (unresolved).
    visitor = _DownstreamReadVisitor(
        1,
        1,
        None,
        candidates={"x"},
        pass_mode=_VisitorPassMode.AFTER_UNIT,
    )
    visitor.visit(tree)
    # The read of x in 'y = x + 1' must be detected as downstream because u_end_col is None.
    assert "x" in visitor.loaded


def test_enclosing_swallowing_with_block_rejects_outputs() -> None:
    """Verifies that units enclosed in contextlib.suppress blocks with downstream reads
    are rejected."""
    code = (
        "import contextlib\n"
        "def f():\n"
        "    with contextlib.suppress(GeneratorExit):\n"
        "        total = 10\n"
        "        yield total\n"
        "    print(total)\n"
    )
    tree = ast.parse(code)
    scope_node = tree.body[1]
    # Unit lines 4 to 5 inside with block:
    assert _enclosing_try_reads_outputs(scope_node, 4, 5, {"total"}) is True
    assert _enclosing_try_reads_outputs(scope_node, 4, 5, {"other"}) is False

    u1 = {
        "file": "m1.py",
        "start": 4,
        "end": 5,
        "source_text": code,
        "outputs": ["total"],
    }
    u2 = {
        "file": "m2.py",
        "start": 4,
        "end": 5,
        "source_text": code,
        "outputs": ["total"],
    }
    s1 = {"definite_stores": ["total"], "inputs": []}
    s2 = {"definite_stores": ["total"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_generator_subroutine_outputs_mocked_scope_fails_closed() -> None:
    """Verifies that mocked/hand-built scopes lacking definite_stores and inputs fail closed."""
    u1 = {"outputs": ["val"], "source_text": "def f():\n    val = 1\n"}
    u2 = {"outputs": ["val"], "source_text": "def g():\n    val = 2\n"}
    # Scope dict missing both definite_stores and inputs
    s1: Dict[str, Any] = {"globals": [], "nonlocals": []}
    s2: Dict[str, Any] = {"globals": [], "nonlocals": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_downstream_dynamic_reads_fail_closed() -> None:
    """Verifies that downstream calls to dynamic read functions (locals, vars, eval, exec, dir)
    mark all candidate outputs as loaded fail-closed."""
    code_locals = (
        "def f():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    data = locals()\n"
        "    return data\n"
    )
    unit = {"file": "mod.py", "start": 2, "end": 3}
    reads = collect_downstream_read_names(code_locals, unit, candidates={"x", "y"})
    assert reads == {"x", "y"}

    code_eval = (
        "def g():\n"
        "    x = 10\n"
        "    eval('x + 1')\n"
    )
    unit_g = {"file": "mod.py", "start": 2, "end": 2}
    reads_g = collect_downstream_read_names(code_eval, unit_g, candidates={"x"})
    assert reads_g == {"x"}


def test_enclosing_with_reads_fail_closed() -> None:
    """Verifies that an enclosing with statement (custom or non-suppress) whose outputs are read
    downstream causes generator subroutine extraction to fail closed."""
    code = (
        "def f():\n"
        "    with custom_manager():\n"
        "        total = 10\n"
        "        yield total\n"
        "    print(total)\n"
    )
    tree = ast.parse(code)
    scope_node = tree.body[0]
    assert _enclosing_try_reads_outputs(scope_node, 3, 4, {"total"}) is True
    assert _enclosing_try_reads_outputs(scope_node, 3, 4, {"other"}) is False


def test_collect_scope_closures_method_param_genexp_does_not_leak() -> None:
    """Verifies that method parameters used in generator expressions inside pre-unit classes
    do not escape as free reads into enclosing scope closures."""
    code = (
        "outer_var = 10\n"
        "class PreUnitHelper:\n"
        "    def calculate(self, total: int):\n"
        "        return (x + outer_var for x in range(total))\n"
        "\n"
        "# Line 6: Unit starts here\n"
        "total = 100\n"
        "res = total + 1\n"
    )
    tree = ast.parse(code)
    captured = _collect_pre_unit_closures(tree, u_start=6)
    assert "total" not in captured
    assert "outer_var" in captured


def test_scope_reads_outputs_after_line_boundary_precision() -> None:
    """Verifies that _scope_reads_outputs_after_line evaluates reads strictly post-line."""
    code_no_post = (
        "def run():\n"
        "    with open('f') as fp:\n"
        "        total = 42\n"
        "        yield total\n"
    )
    tree_no_post = ast.parse(code_no_post)
    func_node_no_post = tree_no_post.body[0]
    # Line 4 is 'yield total', which is the end of the with-block
    # Reads on line 4 itself must NOT be treated as occurring downstream after line 4
    assert not _scope_reads_outputs_after_line(func_node_no_post, 4, {"total"})

    code_with_post = (
        "def run():\n"
        "    with open('f') as fp:\n"
        "        total = 42\n"
        "        yield total\n"
        "    print(total)\n"
    )
    tree_with_post = ast.parse(code_with_post)
    func_node_with_post = tree_with_post.body[0]
    # Reads on line 5 (after the with-block on line 4) MUST be detected
    assert _scope_reads_outputs_after_line(func_node_with_post, 4, {"total"})


def test_collect_scope_closures_class_compound_blocks() -> None:
    """Verifies that _collect_scope_closures inspects methods inside compound blocks in classes."""
    code = (
        "def outer():\n"
        "    outer_var = 10\n"
        "    class C:\n"
        "        if True:\n"
        "            def helper(self):\n"
        "                return outer_var\n"
        "        try:\n"
        "            def helper_try(self):\n"
        "                return outer_var + 1\n"
        "        except Exception:\n"
        "            def helper_except(self):\n"
        "                return outer_var + 2\n"
        "    x = 1\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    closures = _collect_scope_closures(func_node)
    captured_vars = set()
    for _, names in closures:
        captured_vars.update(names)
    assert "outer_var" in captured_vars


def test_enclosing_try_reads_outputs_empty_body_nodes_mock() -> None:
    """Verifies that _enclosing_try_reads_outputs gracefully handles empty body nodes."""
    # Synthetic With node with empty body
    empty_with = ast.With(items=[], body=[])
    dummy_module = ast.Module(body=[empty_with], type_ignores=[])
    assert not _enclosing_try_reads_outputs(dummy_module, 1, 2, {"out"})

    # Synthetic Try node with empty body
    empty_try = ast.Try(body=[], handlers=[], orelse=[], finalbody=[])
    dummy_module_try = ast.Module(body=[empty_try], type_ignores=[])
    assert not _enclosing_try_reads_outputs(dummy_module_try, 1, 2, {"out"})


def test_get_scope_stmts_by_end_lineno_none_end_lineno() -> None:
    """Verifies _get_scope_stmts_by_end_lineno gracefully handles nodes with end_lineno=None."""
    from pydoppelgangerhunt.fixer.dataflow import _get_scope_stmts_by_end_lineno

    stmt = ast.Pass()
    stmt.lineno = 5
    stmt.end_lineno = None  # Explicitly None
    mod = ast.Module(body=[stmt], type_ignores=[])
    res = _get_scope_stmts_by_end_lineno(mod)
    assert 5 in res
    assert res[5] == [stmt]


def test_enclosing_try_reads_outputs_fallback_b_end() -> None:
    """Verifies _enclosing_try_reads_outputs fallback b_end when end_lineno is None."""
    code = (
        "def test():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception:\n"
        "        pass\n"
        "    finally:\n"
        "        pass\n"
        "    return x\n"
    )
    tree = ast.parse(code)
    try_node = tree.body[0].body[0]  # type: ignore[attr-defined]
    try_node.end_lineno = None  # Force fallback calculation of b_end
    # With b_end spanning through finally (line 7), line 8 (return x) is recognized as downstream!
    assert _enclosing_try_reads_outputs(tree.body[0], 3, 3, {"x"})


def test_collect_scope_closures_decorator_none_lineno() -> None:
    """Verifies that _collect_scope_closures safely handles decorator AST nodes with
    explicit None for lineno without raising TypeError."""
    code = """
def outer():
    @dec
    def inner():
        return x
"""
    tree = ast.parse(code)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    inner_func = func.body[0]
    assert isinstance(inner_func, ast.FunctionDef)
    setattr(inner_func.decorator_list[0], "lineno", None)

    closures = _collect_scope_closures(func)
    assert len(closures) == 1
    assert closures[0][0] == inner_func.lineno
    assert "x" in closures[0][1]


def test_set_valued_outputs_inverting_semantic_order_rejected() -> None:
    """Verifies that set-valued outputs whose alphabetical ordering would invert semantic
    store order are rejected to prevent unpositional pairing."""
    # Semantic store order: side 1 stores (total, count), side 2 stores (sum_val, n_val)
    # Alphabetical order: count < total, n_val < sum_val
    outs1 = {"total", "count"}
    outs2 = {"sum_val", "n_val"}

    # 1. Direct output pairing rejects sets
    assert _pair_clone_outputs(outs1, outs2) == []  # type: ignore[arg-type]

    # 2. Generator subroutine resolution rejects set-valued outputs with len > 1
    u1 = {"file": "m1.py", "start": 2, "end": 4, "outputs": outs1}
    u2 = {"file": "m2.py", "start": 2, "end": 4, "outputs": outs2}
    s1 = {"definite_stores": ["total", "count"], "inputs": []}
    s2 = {"definite_stores": ["sum_val", "n_val"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None

    # 3. Helper synthesis fails closed on multi-item set outputs
    assert synthesize_shared_helper_code(u1, u2) == ""


def test_resolve_clone_generator_subroutine_outputs_side2_zero_coords_rejected() -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs rejects when side 1 has clean
    coordinates but side 2 has (0, 0) coordinates."""
    code1 = (
        "try:\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "finally:\n"
        "    cleanup = total\n"
    )
    code2 = (
        "try:\n"
        "    for y in range(10):\n"
        "        yield y\n"
        "finally:\n"
        "    cleanup = count\n"
    )
    # Side 1 is clean and valid (lines 2-3)
    u1 = {
        "file": "mod1.py",
        "start": 2,
        "end": 3,
        "source_text": code1,
        "outputs": ["total"],
    }
    # Side 2 has invalid zero coordinates (0, 0)
    u2 = {
        "file": "mod2.py",
        "start": 0,
        "end": 0,
        "source_text": code2,
        "outputs": ["count"],
    }
    s1 = {"definite_stores": ["total"], "inputs": []}
    s2 = {"definite_stores": ["count"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_collect_downstream_read_names_wrong_source_digest_isolation() -> None:
    """Verifies that a caller-supplied wrong or colliding source_digest does not cause stale
    cache sharing across differing source texts."""
    _clear_downstream_reads_cache()
    code1 = "def f():\n    var_a = 1\n    return var_a\n"
    code2 = "def f():\n    var_b = 2\n    return var_b\n"
    unit1 = {"file": "mod.py", "start": 2, "end": 2}
    unit2 = {"file": "mod.py", "start": 2, "end": 2}

    # Both calls supply the same source_digest, but have different source_text
    reads1 = collect_downstream_read_names(code1, unit1, source_digest="colliding_digest")
    reads2 = collect_downstream_read_names(code2, unit2, source_digest="colliding_digest")

    assert reads1 == {"var_a"}
    assert reads2 == {"var_b"}


def test_scope_metadata_caching_and_cache_clear() -> None:
    """Verifies that scope metadata is cached in WeakKeyDictionaries and purged on clear."""
    code = (
        "def f(x):\n"
        "    for i in range(10):\n"
        "        try:\n"
        "            y = lambda: x\n"
        "        except Exception:\n"
        "            pass\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]

    parent_map = _get_scope_parent_map(func_node)
    assert func_node in _scope_parent_maps
    assert _get_scope_parent_map(func_node) is parent_map

    stmts_by_end = _get_scope_stmts_by_end_lineno(func_node)
    assert func_node in _scope_stmts_by_end
    assert _get_scope_stmts_by_end_lineno(func_node) is stmts_by_end

    closures = _collect_scope_closures(func_node)
    assert func_node in _scope_closures_cache
    assert _collect_scope_closures(func_node) is closures

    loops = _find_enclosing_loops(func_node, 3, 4)
    assert func_node in _scope_loops_cache
    assert _find_enclosing_loops(func_node, 3, 4) == loops

    try_with = _get_scope_try_and_with_blocks(func_node)
    assert func_node in _scope_try_with_cache
    assert _get_scope_try_and_with_blocks(func_node) is try_with

    _clear_downstream_reads_cache()

    assert func_node not in _scope_parent_maps
    assert func_node not in _scope_stmts_by_end
    assert func_node not in _scope_closures_cache
    assert func_node not in _scope_loops_cache
    assert func_node not in _scope_try_with_cache


def test_scope_metadata_caching_non_weakref_object_fallback() -> None:
    """Verifies that non-weakreferenceable objects fall back without raising TypeError."""
    import weakref  # pylint: disable=import-outside-toplevel

    class _MockNonWeakNode(ast.AST):
        __slots__ = ()

    node = _MockNonWeakNode()
    cache: weakref.WeakKeyDictionary[ast.AST, str] = weakref.WeakKeyDictionary()

    result = _get_or_compute_scope_cached(cache, node, lambda: "computed_val")
    assert result == "computed_val"


