"""Unit tests for fixer dataflow analysis, downstream reads, and generator subroutines."""

from __future__ import annotations

import ast
from collections import OrderedDict
import concurrent.futures
import gc
import hashlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    generate_refactoring_patch,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import dataflow
from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access
    GeneratorCloneSideData,
    _ClosurePos,
    _DownstreamReadVisitor,
    _MAX_DOWNSTREAM_CACHE_SIZE,
    _MAX_SCOPE_CACHE_SIZE,
    _VisitorPassMode,
    _build_downstream_cache_key,
    _classify_closure_position,
    _clear_downstream_reads_cache,
    _collect_pre_unit_closures,
    _collect_scope_closures,
    _downstream_cache_lock,
    _downstream_reads_cache,
    _enclosing_try_reads_outputs,
    _execute_read_pass,
    _extract_effective_unit_outputs,
    _extract_nested_scope_free_reads,
    _extract_scope_local_names,
    _extract_single_node_local_names,
    _extract_unit_live_in_names,
    _find_enclosing_loops,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _get_scope_stmts_by_end_lineno,
    _get_scope_try_and_with_blocks,
    _is_ast_node_within_unit_bounds,
    _is_eagerly_consumed_genexp,
    _load_unit_file_text,
    _pair_clone_outputs,
    _resolve_unit_ast_end_col,
    _scope_closures_cache,
    _scope_loops_cache,
    _scope_parent_maps,
    _scope_reads_outputs_after_line,
    _scope_stmts_by_end,
    _scope_try_with_cache,
    _stmts_enclose_unit,
    _warned_lenient_closure,
    clear_dataflow_cache,
    collect_downstream_read_names,
    has_async_generator_delegation_hazard,
    is_async_generator_with_return_value,
    resolve_clone_generator_subroutine_outputs,
    resolve_clone_pair_outputs,
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
    assert is_async_generator_with_return_value is has_async_generator_delegation_hazard

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

    # Differing variable orders (permuted outputs) fail closed when allow_permuted_outputs is False
    pairs_swapped = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_swapped == []
    assert _pair_clone_outputs(["x", "y"], ["y", "x"], allow_permuted_outputs=True) == [
        ("x", "y"),
        ("y", "x"),
    ]


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
    """Verifies that same-line units without column offsets fail closed and treat
    same-line statements as downstream when end column is ambiguous."""

    code = (
        "def runner():\n"
        "    yield x; print(total)\n"
    )
    # Unit on line 2 with NO start_col or end_col: fails closed and treats same-line reads as downstream
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads
    assert "x" in reads

    # When start_col / end_col are provided, disambiguation distinguishes in-unit vs downstream reads
    unit_col = {"start": 2, "end": 2, "start_col": 4, "end_col": 11}
    reads_col = collect_downstream_read_names(code, unit_col, candidates={"total", "x"})
    assert reads_col is not None
    assert "total" in reads_col
    assert "x" not in reads_col


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
    u1_pre = {
        "outputs": ["a", "b"],
        "source_text": "def f():\n    a = 1\n    print(a)\n",
        "start": 2,
        "end": 2,
    }
    u2_pre = {
        "outputs": ["x", "y"],
        "source_text": "def g():\n    x = 1\n    print(x)\n",
        "start": 2,
        "end": 2,
    }
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

    # Identical name sets with conflicting order fail closed when allow_permuted_outputs is False
    pairs_ident = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_ident == []
    assert _pair_clone_outputs(["x", "y"], ["y", "x"], allow_permuted_outputs=True) == [
        ("x", "y"),
        ("y", "x"),
    ]

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


def test_collect_downstream_read_names_loop_carried_unit_inputs_without_preceding_read() -> None:
    """Verifies that unit inputs updated in an enclosing loop are captured as live across
    iterations even when there are no statements preceding the unit in the loop body."""
    code = (
        "def process_batches(batches):\n"
        "    total = 0\n"
        "    for batch in batches:\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield total\n"
    )
    # Unit is lines 4 to 6: inner loop where total is updated
    unit = {"start": 4, "end": 6}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_collect_downstream_read_names_loop_carried_killed_before_unit() -> None:
    """Verifies that variables reassigned before the unit inside the enclosing loop
    are not captured as live across iterations."""
    code = (
        "def process_batches(batches):\n"
        "    for batch in batches:\n"
        "        total = 0\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield total\n"
    )
    # Unit is lines 4 to 6: inner loop where total is updated
    unit = {"start": 4, "end": 6}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads


def test_collect_downstream_read_names_loop_carried_conditional_reset() -> None:
    """Verifies that conditional resets before the unit inside an enclosing loop do not
    unconditionally kill loop-carried dependencies."""
    code = (
        "def process_batches(batches):\n"
        "    total = 0\n"
        "    for batch in batches:\n"
        "        if batch.reset:\n"
        "            total = 0\n"
        "        for x in batch.items:\n"
        "            total += x\n"
        "            yield total\n"
    )
    # Unit is lines 6 to 8: inner loop where total is updated
    unit = {"start": 6, "end": 8}
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
        "def process_batches(batches):\n"
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
        "def f(items):\n"
        "    class K:\n"
        "        def m(self):\n"
        "            return total\n"
        "\n"
        "    for x in items:\n"
        "        total = x\n"
        "        yield x\n"
        "    K().m()\n"
    )
    unit = {"file": "mod.py", "start": 6, "end": 8, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_class_instance_alias_deferred_read() -> None:
    """Verifies that pre-unit class methods capturing candidate variables are preserved
    by the blanket pre-unit rule regardless of instance aliasing (obj = K())."""
    src = (
        "def f(items):\n"
        "    class K:\n"
        "        def m(self):\n"
        "            return total\n"
        "\n"
        "    obj = K()\n"
        "    for x in items:\n"
        "        total = x\n"
        "        yield x\n"
        "    obj.m()\n"
    )
    unit = {"file": "mod.py", "start": 7, "end": 9, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_pre_unit_closure_redefinition_in_branches() -> None:
    """Verifies that closures redefined across conditional branches merge free variable sets."""
    src = (
        "def f(flag, items):\n"
        "    if flag:\n"
        "        def h():\n"
        "            return total\n"
        "    else:\n"
        "        def h():\n"
        "            return other\n"
        "\n"
        "    for x in items:\n"
        "        total = x\n"
        "        yield x\n"
        "    h()\n"
    )
    unit = {"file": "mod.py", "start": 9, "end": 11, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total", "other"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_escaping_closure_registered_before_unit() -> None:
    """Verifies that pre-unit closures capturing candidate variables are treated as live
    even when invoked indirectly through pre-registered handlers."""
    src = (
        "def f(items):\n"
        "    def cb():\n"
        "        return total\n"
        "\n"
        "    register(cb)\n"
        "    for x in items:\n"
        "        total = x\n"
        "        yield x\n"
        "    fire()\n"
    )
    unit = {"file": "mod.py", "start": 6, "end": 8, "name": "unit", "kind": "compound_block"}
    reads = collect_downstream_read_names(src, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_downstream_reads_compound_statement_lambda_after_unit_not_pre_captured() -> None:
    """Verifies that lambdas occurring after u_start inside an enclosing compound statement
    are not prematurely captured into pre_unit_captured_reads."""
    src = (
        "def f(condition, items):\n"
        "    if condition:\n"
        "        a = 1\n"
        "        for x in items:\n"
        "            fn = lambda: total\n"
        "            total = x * 2\n"
        "            yield x\n"
    )
    unit = {"file": "mod.py", "start": 4, "end": 7, "name": "unit", "kind": "compound_block"}
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
        "def outer():\n"
        "    captured_var = 0\n"
        "    class C:\n"
        "        nonlocal captured_var\n"
        "        captured_var = 42\n"
    )
    class_node = ast.parse(code).body[0].body[1]  # type: ignore[attr-defined]
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


def test_collect_downstream_read_names_content_digest_cache_sharing() -> None:
    """Verifies that identical content shares cache entries across distinct unit dicts."""
    _clear_downstream_reads_cache()
    code = "def f():\n    x = 1\n    return x\n"
    unit1 = {"file": "f.py", "start": 2, "end": 2, "tag": "first"}
    unit2 = {"file": "f.py", "start": 2, "end": 2, "tag": "second"}
    reads1 = collect_downstream_read_names(code, unit1)
    reads2 = collect_downstream_read_names(code, unit2)
    assert reads1 == {"x"}
    assert reads2 == {"x"}
    # Content digest deduplication: identical content shares the cache
    assert len(_downstream_reads_cache) == 1


def test_collect_downstream_read_names_digest_invalidation(tmp_path: Path) -> None:
    """Verifies that differing content digest produces distinct cache entries for
    downstream reads."""
    _clear_downstream_reads_cache()
    f_path = tmp_path / "target.py"
    code = "def f():\n    x = 1\n    return x\n"
    f_path.write_text(code, encoding="utf-8")
    unit1 = {"file": str(f_path), "start": 2, "end": 2}
    reads1 = collect_downstream_read_names(code, unit1, source_digest="hash1")
    assert reads1 == {"x"}
    assert len(_downstream_reads_cache) == 1

    unit2 = {"file": str(f_path), "start": 2, "end": 2}
    reads2 = collect_downstream_read_names(code, unit2, source_digest="hash2")
    assert reads2 == {"x"}
    assert len(_downstream_reads_cache) == 2


def test_collect_downstream_read_names_in_memory_no_disk_dependency() -> None:
    """Verifies that collect_downstream_read_names executes purely in-memory from source_text."""
    _clear_downstream_reads_cache()
    code = "def f():\n    x = 1\n    return x\n"
    unit = {"file": "virtual/nonexistent_path/mod.py", "start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit)
    assert reads == {"x"}


def test_resolve_clone_generator_subroutine_outputs_precomputed_definite_stores() -> None:
    """Verifies precomputed outputs are validated against definite stores when available."""
    u1 = {
        "outputs": ["x", "unassigned"],
        "source_text": "def f():\n    x = 1\n    print(x, unassigned)\n",
        "start": 2,
        "end": 2,
    }
    u2 = {
        "outputs": ["a", "b"],
        "source_text": "def g():\n    a = 1\n    print(a, b)\n",
        "start": 2,
        "end": 2,
    }
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
        "def gen1():\n"
        "    try:\n"
        "        for x in range(10):\n"
        "            yield x\n"
        "    finally:\n"
        "        cleanup = total\n"
    )
    code2 = (
        "def gen2():\n"
        "    try:\n"
        "        for y in range(10):\n"
        "            yield y\n"
        "    finally:\n"
        "        cleanup = count\n"
    )
    u1 = {
        "file": "mod1.py",
        "start": 4,
        "end": 3,
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
    """Verifies that _load_unit_file_text returns None when source_lines is marked as sliced,
    and falls back to in-memory source_lines for virtual units when not explicitly sliced."""
    unit_sliced = {
        "file": "virtual.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = 1\n", "    y = 2\n", "    z = 3\n"],
        "source_lines_is_sliced": True,
    }
    assert _load_unit_file_text(unit_sliced) is None

    unit_fallback = {
        "file": "virtual.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = 1\n", "    y = 2\n", "    z = 3\n"],
    }
    assert _load_unit_file_text(unit_fallback) is None

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
    monkeypatch.setattr(
        "pydoppelgangerhunt.fixer._df_hazards._MAX_DOWNSTREAM_CACHE_SIZE", 5
    )
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


def test_scope_reads_outputs_after_line_same_line_semicolon() -> None:
    """Verifies that _scope_reads_outputs_after_line flags same-line reads when end_col provided."""
    code = (
        "def run():\n"
        "    with mgr: total = yield x; print(total)\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    # Unit 'total = yield x' ends at col 29 on line 2
    # With end_col=29, print(total) starting at col 31 MUST be detected
    assert _scope_reads_outputs_after_line(func_node, 2, {"total"}, end_col=29)

    # Without end_col, reads on line 2 itself are not treated as post-line
    assert not _scope_reads_outputs_after_line(func_node, 2, {"total"})


def test_enclosing_try_reads_outputs_same_line_with_semicolon() -> None:
    """Verifies that _enclosing_try_reads_outputs permits intra-with reads,
    but detects reads downstream of the with block."""
    code = (
        "def run():\n"
        "    with mgr: total = yield x; print(total)\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    # Unit 'total = yield x' on line 2: intra-with read is permitted
    assert _enclosing_try_reads_outputs(func_node, 2, 2, {"total"}) is False

    code_post = (
        "def run():\n"
        "    with mgr: total = yield x\n"
        "    print(total)\n"
    )
    tree_post = ast.parse(code_post)
    func_post = tree_post.body[0]
    # Post-with read is detected as a cleanup hazard
    assert _enclosing_try_reads_outputs(func_post, 2, 2, {"total"}) is True


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
    assert closures[0][0].line == inner_func.lineno
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
        "def gen1():\n"
        "    try:\n"
        "        for x in range(10):\n"
        "            yield x\n"
        "    finally:\n"
        "        cleanup = total\n"
    )
    code2 = (
        "def gen2():\n"
        "    try:\n"
        "        for y in range(10):\n"
        "            yield y\n"
        "    finally:\n"
        "        cleanup = count\n"
    )
    # Side 1 is clean and valid (lines 3-4)
    u1 = {
        "file": "mod1.py",
        "start": 3,
        "end": 4,
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


def test_collect_downstream_read_names_source_digest_avoids_rehash() -> None:
    """Verifies that providing source_digest uses it directly and avoids rehashing source_text."""
    _clear_downstream_reads_cache()
    code1 = "def f():\n    var_a = 1\n    return var_a\n"
    code2 = "def f():\n    var_b = 2\n    return var_b\n"
    unit1 = {"file": "mod.py", "start": 2, "end": 2}
    unit2 = {"file": "mod.py", "start": 2, "end": 2}

    with mock.patch("pydoppelgangerhunt.fixer.dataflow.hashlib.sha256") as mock_sha:
        reads1 = collect_downstream_read_names(code1, unit1, source_digest="digest_a")
        mock_sha.assert_not_called()
        assert reads1 == {"var_a"}

        reads2 = collect_downstream_read_names(code2, unit2, source_digest="digest_b")
        mock_sha.assert_not_called()
        assert reads2 == {"var_b"}

    # Without source_digest, hashlib.sha256 is called to hash source_text
    with mock.patch(
        "pydoppelgangerhunt.fixer.dataflow.hashlib.sha256", wraps=hashlib.sha256
    ) as mock_sha_wrap:
        reads3 = collect_downstream_read_names(code1, unit1)
        mock_sha_wrap.assert_called()
        assert reads3 == {"var_a"}


def test_stmts_enclose_unit_with_explicit_none_lineno() -> None:
    """Verifies that _stmts_enclose_unit handles statements where lineno is explicitly None."""
    node = ast.Pass()
    node.lineno = None  # type: ignore[assignment]
    node.end_lineno = None  # type: ignore[assignment]
    assert _stmts_enclose_unit([node], 1, 2) is False


def test_scope_metadata_caching_and_cache_clear() -> None:
    """Verifies that scope metadata is cached in bounded OrderedDicts (LRU) and purged on clear."""
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


def test_scope_metadata_caching_unhashable_object_fallback() -> None:
    """Verifies that unhashable objects fall back gracefully without raising TypeError."""
    class _MockUnhashableNode(ast.AST):
        __hash__ = None  # type: ignore[assignment]

    node = _MockUnhashableNode()
    cache: OrderedDict[ast.AST, str] = OrderedDict()

    result = _get_or_compute_scope_cached(cache, node, lambda: "computed_val")
    assert result == "computed_val"
    assert len(cache) == 0


def test_scope_metadata_caching_lru_eviction() -> None:
    """Verifies that scope cache evicts oldest entries when capacity is exceeded."""
    cache: OrderedDict[ast.AST, str] = OrderedDict()
    nodes = [ast.Pass() for _ in range(_MAX_SCOPE_CACHE_SIZE + 5)]
    for idx, n in enumerate(nodes):
        val_str = f"val_{idx}"
        _get_or_compute_scope_cached(cache, n, lambda: val_str)
    assert len(cache) == _MAX_SCOPE_CACHE_SIZE
    assert nodes[0] not in cache
    assert nodes[-1] in cache


def test_extract_nested_scope_free_reads_generator_shadowed_builtin() -> None:
    """Verifies that generator expressions preserve free variables shadowing builtins."""
    code = "(id for _ in items)"
    gen_node = ast.parse(code, mode="eval").body
    assert _extract_nested_scope_free_reads(gen_node) == {"id", "items"}  # type: ignore[arg-type]
    assert _extract_nested_scope_free_reads(
        gen_node, candidates={"id"}  # type: ignore[arg-type]
    ) == {"id"}
    assert _extract_nested_scope_free_reads(
        gen_node, candidates={"items"}  # type: ignore[arg-type]
    ) == {"items"}
    assert _extract_nested_scope_free_reads(
        gen_node, candidates={"other"}  # type: ignore[arg-type]
    ) == set()


def test_collect_downstream_read_names_pre_unit_genexp_shadowed_builtin() -> None:
    """Verifies that pre-unit generator expressions capturing shadowed builtins are detected."""
    _clear_downstream_reads_cache()
    code = (
        "def compute(items):\n"
        "    id = 42\n"
        "    g = (id for _ in items)\n"
        "    id = 100\n"
    )
    unit = {"file": "test_id.py", "start": 4, "end": 4}
    reads = collect_downstream_read_names(
        code, unit, candidates={"id"}, skip_pre_unit_closures=False
    )
    assert reads == {"id"}

    reads_lenient = collect_downstream_read_names(
        code, unit, candidates={"id"}, skip_pre_unit_closures=True
    )
    assert reads_lenient == set()


def test_stmts_enclose_unit_synthetic_or_unlocated_nodes() -> None:
    """Verifies that synthetic or unlocated nodes (lineno=None or 0) do not cause underflow."""
    synthetic = ast.Pass()
    real_stmt = ast.parse("x = 1").body[0]
    real_stmt.lineno = 10
    real_stmt.end_lineno = 15

    stmts = [synthetic, real_stmt]
    # Unit at lines 5-8 is before real statements starting at 10: must not be enclosed
    assert not _stmts_enclose_unit(stmts, 5, 8)
    # Unit at lines 11-14 is enclosed
    assert _stmts_enclose_unit(stmts, 11, 14)
    # Sequence of only synthetic statements with no lineno
    assert not _stmts_enclose_unit([synthetic], 1, 5)


def test_resolve_clone_pair_outputs_clean_pair() -> None:
    """Verifies that resolve_clone_pair_outputs successfully pairs matching outputs."""
    u1 = {"kind": "function", "outputs": ["a", "b"]}
    u2 = {"kind": "function", "outputs": ["x", "y"]}
    s1: Dict[str, Any] = {"inputs": [], "definite_stores": ["a", "b"]}
    s2: Dict[str, Any] = {"inputs": [], "definite_stores": ["x", "y"]}

    res = resolve_clone_pair_outputs(u1, u2, s1, s2)
    assert res == (["a", "b"], ["x", "y"])


def test_resolve_clone_pair_outputs_mismatched_subroutine_kinds() -> None:
    """Verifies that resolve_clone_pair_outputs rejects mismatched subroutine kinds."""
    u1 = {"kind": "compound_block"}
    u2 = {"kind": "function"}
    s1: Dict[str, Any] = {"inputs": []}
    s2: Dict[str, Any] = {"inputs": []}

    assert resolve_clone_pair_outputs(u1, u2, s1, s2) is None


def test_resolve_clone_pair_outputs_set_valued_rejection() -> None:
    """Verifies that resolve_clone_pair_outputs rejects set-valued outputs with multiple items."""
    u1 = {"kind": "function", "outputs": {"a", "b"}}
    u2 = {"kind": "function", "outputs": ["x", "y"]}
    s1: Dict[str, Any] = {"inputs": []}
    s2: Dict[str, Any] = {"inputs": []}

    assert resolve_clone_pair_outputs(u1, u2, s1, s2) is None


def test_collect_scope_closures_class_header_expressions() -> None:
    """Verifies that _collect_scope_closures inspects class decorators, bases, and keywords."""
    code = (
        "def outer(val1, val2, val3):\n"
        "    @dec(lambda: val1)\n"
        "    class MyClass(base(lambda: val2), metaclass=meta(lambda: val3)):\n"
        "        pass\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    closures = _collect_scope_closures(func_node)
    captured = set()
    for _, names in closures:
        captured.update(names)
    assert "val1" in captured
    assert "val2" in captured
    assert "val3" in captured


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type_params syntax requires Python 3.12+"
)
def test_collect_scope_closures_class_type_params() -> None:
    """Verifies that _collect_scope_closures inspects class type parameters in Python 3.12+."""
    code_gen = (
        "def outer(val4):\n"
        "    class GenericClass[T: (lambda: val4)]:\n"
        "        pass\n"
    )
    tree_gen = ast.parse(code_gen)
    closures_gen = _collect_scope_closures(tree_gen.body[0])
    captured_gen = set()
    for _, names in closures_gen:
        captured_gen.update(names)
    assert "val4" in captured_gen


def test_collect_pre_unit_closures_same_line_semicolon() -> None:
    """Verifies that pre-unit closures defined before a semicolon on the same line are captured."""
    code = "cb = lambda: val; total = [x for x in items]\n"
    tree = ast.parse(code)
    pre_closures = _collect_pre_unit_closures(
        tree, u_start=1, candidates={"val"}, u_start_col=18
    )
    assert "val" in pre_closures

    pre_closures_prior = _collect_pre_unit_closures(
        tree, u_start=1, candidates={"val"}, u_start_col=0
    )
    assert "val" not in pre_closures_prior


def test_closure_pos_attributes_and_iteration() -> None:
    """Verifies that _ClosurePos behaves as a NamedTuple with line, col, and tuple unpacking."""
    pos = _ClosurePos(12, 4)
    assert isinstance(pos, tuple)
    assert pos.line == 12
    assert pos.col == 4
    assert pos[0] == 12
    assert pos[1] == 4
    assert len(pos) == 2
    line_val, col_val = pos
    assert line_val == 12 and col_val == 4
    assert pos.line < 15


def test_collect_scope_closures_falsy_col_offset_preserved() -> None:
    """Verifies that decorator and subnode column offset 0 is preserved and does not fall back."""
    fn = ast.FunctionDef(
        name="decorated",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[ast.Pass(lineno=3, col_offset=4)],
        decorator_list=[
            ast.Call(
                func=ast.Name(id="dec", ctx=ast.Load(), lineno=2, col_offset=0),
                args=[
                    ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            vararg=None,
                            kwonlyargs=[],
                            kw_defaults=[],
                            kwarg=None,
                            defaults=[],
                        ),
                        body=ast.Name(id="captured_val", ctx=ast.Load(), lineno=2, col_offset=5),
                        lineno=2,
                        col_offset=0,
                    )
                ],
                keywords=[],
                lineno=2,
                col_offset=0,
            )
        ],
        lineno=2,
        col_offset=8,
    )
    mod = ast.Module(body=[fn], type_ignores=[])
    closures = _collect_scope_closures(mod)
    assert len(closures) == 1
    pos, names = closures[0]
    assert "captured_val" in names
    assert pos.line == 2
    assert getattr(pos, "line", None) == 2
    assert getattr(pos, "col", None) == 0  # Preserved as 0, not overridden by fn.col_offset (8)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type_params syntax requires Python 3.12+"
)
def test_extract_nested_scope_free_reads_inner_type_params() -> None:
    """Verifies that inner function and type alias type params do not escape to outer scope."""
    # Python 3.12+ generic function syntax
    code = (
        "def outer():\n"
        "    def inner[T](val: T) -> T:\n"
        "        return val + external_var\n"
        "    return inner(1)\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    reads = _extract_nested_scope_free_reads(func_node)
    # external_var should be read, but T must NOT escape
    assert "external_var" in reads
    assert "T" not in reads

    # Inner TypeAlias syntax
    code_alias = (
        "def outer_alias():\n"
        "    type MyList[U] = list[U]\n"
        "    return external_var\n"
    )
    tree_alias = ast.parse(code_alias)
    func_alias = tree_alias.body[0]
    assert isinstance(func_alias, ast.FunctionDef)
    reads_alias = _extract_nested_scope_free_reads(func_alias)
    assert "external_var" in reads_alias
    assert "U" not in reads_alias


def test_resolve_unit_ast_end_col_single_line_multiple_semicolons() -> None:
    """Verifies that single-line units with multiple trailing semicolons fail closed by returning
    None when columns are ambiguous, but resolve accurately when uniquely matched."""
    code = "for x in items: a = 1; b = 2; c = 3\n"
    tree = ast.parse(code)
    # col_offset of a is 16, b is 23, c is 30. end_col_offset of c is 35.
    unit_ambiguous = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 23,
        "is_subroutine": True,
    }
    # Multiple statements (b = 2 and c = 3) match: fails closed to prevent swallowing downstream
    end_col = _resolve_unit_ast_end_col(tree, unit_ambiguous)
    assert end_col is None

    # Single matching statement (c = 3) accurately resolves to 35
    unit_unique = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 30,
        "is_subroutine": True,
    }
    assert _resolve_unit_ast_end_col(tree, unit_unique) == 35


def test_frozenset_outputs_defensive_handling() -> None:
    """Verifies that frozenset outputs are handled defensively across output pairing routines."""
    # _pair_clone_outputs rejects frozenset
    assert _pair_clone_outputs(frozenset(["a", "b"]), ["x", "y"]) == []
    assert _pair_clone_outputs(["a", "b"], frozenset(["x", "y"])) == []

    # _extract_effective_unit_outputs: multi-item frozenset returns []
    u_multi = {"outputs": frozenset(["a", "b"])}
    assert _extract_effective_unit_outputs(u_multi, {}) == []

    # Single-item frozenset returns list of 1 element
    u_single = {"outputs": frozenset(["a"])}
    assert _extract_effective_unit_outputs(u_single, {}) == ["a"]

    # resolve_clone_generator_subroutine_outputs rejects multi-item frozenset
    assert (
        resolve_clone_generator_subroutine_outputs(
            {"outputs": frozenset(["a", "b"])},
            {"outputs": ["x", "y"]},
            {},
            {},
        )
        is None
    )

    # resolve_clone_pair_outputs rejects multi-item frozenset
    assert (
        resolve_clone_pair_outputs(
            {"outputs": frozenset(["a", "b"])},
            {"outputs": ["x", "y"]},
            {},
            {},
        )
        is None
    )


def test_scope_parent_map_excludes_scope_node_and_stays_bounded() -> None:
    """Verifies that _scope_parent_maps excludes scope_node as key to prevent circular
    references, maps direct children to scope_node, and that cache size stays bounded."""
    # 1. Verify parent_map maps children to scope_node and does not assign a parent to scope_node
    code = "def sample():\n    x = 1\n    return x\n"
    tree = ast.parse(code)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    parent_map = _get_scope_parent_map(fn)
    assert fn not in parent_map
    assert parent_map[fn.body[0]] is fn

    # 2. Stress test: collect_downstream_read_names across 2,000 synthetic AST modules
    for i in range(2000):
        src = f"def f_{i}():\n    x = {i}\n    print(x)\n"
        unit = {"file": f"mod_{i}.py", "start": 2, "end": 2}
        collect_downstream_read_names(src, unit)

    # Cache must stay bounded to _MAX_SCOPE_CACHE_SIZE (1024)
    assert len(_scope_parent_maps) <= 1024
    gc.collect()


def test_load_unit_file_text_fails_closed_on_notebook(tmp_path: Path) -> None:
    """Verifies that _load_unit_file_text restricts disk fallback to .py files and fails closed
    (returns None) for Jupyter notebooks (.ipynb) to prevent unsafe patch synthesis."""
    nb_file = tmp_path / "notebook.ipynb"
    nb_file.write_text('{"cells": []}', encoding="utf-8")

    unit = {"file": str(nb_file), "start": 1, "end": 5}
    loaded = _load_unit_file_text(unit, repo_root=str(tmp_path))
    assert loaded is None


def test_collect_downstream_read_names_multithreaded_cache_concurrency() -> None:
    """Verifies that concurrent invocations across worker threads execute safely,
    maintain cache boundedness, and return fresh mutable copies preventing cross-thread mutation."""
    def _worker(thread_idx: int) -> None:
        for i in range(100):
            var_name = f"val_{thread_idx}_{i}"
            src = f"def worker_{thread_idx}_{i}():\n    {var_name} = 1\n    print({var_name})\n"
            unit = {"file": f"thread_{thread_idx}_mod_{i}.py", "start": 2, "end": 2}
            res = collect_downstream_read_names(src, unit)
            assert res is not None
            assert var_name in res
            # Mutating the returned set must not affect any cache entry
            res.add("SENTINEL_MUTATION")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, tid) for tid in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(_downstream_reads_cache) <= _MAX_DOWNSTREAM_CACHE_SIZE
    # Ensure no cache entry contains SENTINEL_MUTATION
    for cached_val in _downstream_reads_cache.values():
        assert "SENTINEL_MUTATION" not in cached_val


def test_subroutine_outputs_duplicate_stores_deduplicated_and_paired() -> None:
    """Verifies that multiple assignments to the same root variable in subroutines do not produce
    duplicate outputs or cause unwarranted rejection in resolve_clone_pair_outputs."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    code1 = (
        "def outer1(step):\n"
        "    total = 0\n"
        "    total += step\n"
        "    print(total)\n"
    )
    code2 = (
        "def outer2(delta):\n"
        "    count = 0\n"
        "    count += delta\n"
        "    print(count)\n"
    )

    u1 = {
        "kind": "compound_block",
        "start": 2,
        "end": 3,
        "file": "f1.py",
        "source_text": code1,
    }
    u2 = {
        "kind": "compound_block",
        "start": 2,
        "end": 3,
        "file": "f2.py",
        "source_text": code2,
    }

    s1 = inspect_single_unit_scope(u1)
    s2 = inspect_single_unit_scope(u2)

    # Scopes must deduplicate outputs in order of first store
    assert s1["outputs"] == ["total"]
    assert s2["outputs"] == ["count"]

    # resolve_clone_pair_outputs must successfully pair them
    res = resolve_clone_pair_outputs(u1, u2, s1, s2)
    assert res == (["total"], ["count"])

    # Even if precomputed outputs contain duplicates, it must safely deduplicate and pair
    u1_dup = dict(u1, outputs=["total", "total"])
    u2_dup = dict(u2, outputs=["count", "count"])
    res_dup = resolve_clone_pair_outputs(u1_dup, u2_dup, s1, s2)
    assert res_dup == (["total"], ["count"])


def test_clear_downstream_reads_cache_resets_warned_lenient_closure() -> None:
    """Verifies that _clear_downstream_reads_cache resets _warned_lenient_closure flag."""
    _warned_lenient_closure[0] = True
    assert _warned_lenient_closure[0] is True
    _clear_downstream_reads_cache()
    assert _warned_lenient_closure[0] is False


def test_collect_downstream_read_names_cache_hit_with_and_without_end_col() -> None:
    """Verifies that cache keys do not diverge whether end_col is pre-provided or AST-resolved."""
    _clear_downstream_reads_cache()
    src = (
        "def compute():\n"
        "    x = 10\n"
        "    return x + 5\n"
    )
    unit_no_end_col = {"file": "mod.py", "start": 2, "end": 2}
    res1 = collect_downstream_read_names(src, unit_no_end_col)
    assert res1 == {"x"}
    assert len(_downstream_reads_cache) == 1

    # Call with explicit end_col_offset matching AST resolution
    cached_key = next(iter(_downstream_reads_cache.keys()))
    resolved_end_col = (
        cached_key.end_col if hasattr(cached_key, "end_col") else cached_key[4]
    )
    assert resolved_end_col is not None

    unit_with_end_col = {
        "file": "mod.py",
        "start": 2,
        "end": 2,
        "end_col_offset": resolved_end_col,
    }
    res2 = collect_downstream_read_names(src, unit_with_end_col)
    assert res2 == {"x"}
    # Cache size must still be 1 (cache hit, no key divergence!)
    assert len(_downstream_reads_cache) == 1


def test_resolve_clone_generator_subroutine_outputs_closure_pos_tuple_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs safely normalizes
    (line, col) tuple instances from _collect_scope_closures without TypeError."""
    code1 = (
        "def g1():\n"
        "    val = 1\n"
        "    yield val\n"
        "    print(val)\n"
    )
    code2 = (
        "def g2():\n"
        "    val = 2\n"
        "    yield val\n"
        "    print(val)\n"
    )

    u1 = {"start": 2, "end": 3, "file": "mod1.py", "source_text": code1}
    u2 = {"start": 2, "end": 3, "file": "mod2.py", "source_text": code2}
    s1 = {"has_yield": True, "inputs": [], "outputs": ["val"], "definite_stores": ["val"]}
    s2 = {"has_yield": True, "inputs": [], "outputs": ["val"], "definite_stores": ["val"]}

    # Without closures, resolution succeeds and returns (["val"], ["val"])
    res_clean = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res_clean == (["val"], ["val"])

    def _mock_closures(scope_node: Any, **kwargs: Any) -> list[tuple[Any, set[str]]]:
        return [((2, 4), {"val"})]

    monkeypatch.setattr(
        "pydoppelgangerhunt.fixer._df_pairing._collect_scope_closures", _mock_closures
    )

    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None


def test_collect_pre_unit_closures_same_line_missing_col_fails_closed() -> None:
    """Verifies that same-line closures fail closed when column info is missing."""
    code = "cb = lambda: total; total = 1\n"
    tree = ast.parse(code)
    # With u_start_col=None, c_line == u_start is safely treated as pre-unit
    pre_closures = _collect_pre_unit_closures(
        tree, u_start=1, candidates={"total"}, u_start_col=None
    )
    assert "total" in pre_closures


def test_precomputed_outputs_empty_list_precedence() -> None:
    """Verifies that an explicitly empty precomputed_outputs list is honored over outputs."""
    from pydoppelgangerhunt.fixer.scope import _inspect_unit_scope

    code = (
        "def worker():\n"
        "    total = 0\n"
        "    yield total\n"
    )
    unit = {
        "file": "mod.py",
        "start": 2,
        "end": 3,
        "kind": "compound_block",
        "source_text": code,
        "precomputed_outputs": [],
        "outputs": ["total"],
    }
    scope = _inspect_unit_scope(unit)
    assert scope["outputs"] == []


def test_downstream_read_crlf_and_lone_cr_line_handling() -> None:
    """Verifies that collect_downstream_read_names handles CRLF and lone-CR line endings."""
    crlf_code = "def f():\r\n    x = 1\r\n    y = x + 1\r\n    return y\r\n"
    lone_cr_code = "def f():\r    x = 1\r    y = x + 1\r    return y\r"

    u_crlf = {"file": "crlf.py", "source_text": crlf_code, "start": 2, "end": 2}
    reads_crlf = collect_downstream_read_names(crlf_code, u_crlf, {"x"})
    assert reads_crlf is not None and "x" in reads_crlf

    u_cr = {"file": "cr.py", "source_text": lone_cr_code, "start": 2, "end": 2}
    reads_cr = collect_downstream_read_names(lone_cr_code, u_cr, {"x"})
    assert reads_cr is not None and "x" in reads_cr


def test_resolve_unit_ast_end_col_invalid_coords_fails_closed() -> None:
    """Verifies that _resolve_unit_ast_end_col returns None on invalid coordinates."""
    tree = ast.parse("def f():\n    x = 1\n")
    assert _resolve_unit_ast_end_col(tree, {"start": "invalid"}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": 1, "end": "invalid"}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": -1}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": 5, "end": 2}) is None


def test_clear_downstream_reads_cache_resets_closure_warning_cache() -> None:
    """Verifies that _clear_downstream_reads_cache resets closure warning cache."""
    from pydoppelgangerhunt.config import (  # pylint: disable=import-outside-toplevel
        _warned_closure_strictness_values,
        clear_closure_warning_cache,
        resolve_closure_strictness_mode,
    )

    clear_closure_warning_cache()
    resolve_closure_strictness_mode("unrecognized_mode_foo")
    assert "unrecognized_mode_foo" in _warned_closure_strictness_values

    _clear_downstream_reads_cache()
    assert len(_warned_closure_strictness_values) == 0


def test_outer_expr_visitor_multi_clause_comprehension_scoping() -> None:
    """Verifies that multi-clause comprehensions evaluate first iter in outer scope

    and subsequent iters with prior comprehension targets in scope.
    """
    # 1. First iter evaluates in outer scope (matrix), while subsequent clauses
    # and body read targets from earlier comprehension clauses.
    code_multi = (
        "def outer():\n"
        "    matrix = [[1, 2], [3, 4]]\n"
        "    row = [9, 9]\n"
        "    x = 42\n"
        "    yield 1\n"
        "    g = (x for row in matrix for x in row)\n"
    )
    u_multi = {"start": 5, "end": 5}
    reads_multi = collect_downstream_read_names(
        code_multi, u_multi, candidates={"matrix", "row", "x"}
    )
    assert reads_multi == {"matrix"}

    # 2. When the first generator's iter reads 'row', it is captured as an outer read.
    code_first_iter_read = (
        "def outer():\n"
        "    row = [[1, 2], [3, 4]]\n"
        "    x = 42\n"
        "    yield 1\n"
        "    g = (x for row in row for x in row)\n"
    )
    u_first = {"start": 4, "end": 4}
    reads_first = collect_downstream_read_names(
        code_first_iter_read, u_first, candidates={"row", "x"}
    )
    assert reads_first == {"row"}


def test_eagerly_consumed_genexp_not_treated_as_escaping_closure() -> None:
    """Verifies that generator expressions directly consumed by eager built-ins
    (e.g., sum) are not treated as escaping closures."""
    code = (
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
        "    print(total)\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    closures = _collect_scope_closures(func_node)
    # The eagerly consumed genexp in sum(...) should not be in closures
    assert len(closures) == 0

    code_escaping = (
        "def compute_escaping():\n"
        "    data = [1, 2, 3]\n"
        "    gen = (x * 2 for x in data)\n"
        "    yield gen\n"
    )
    tree_esc = ast.parse(code_escaping)
    closures_esc = _collect_scope_closures(tree_esc.body[0])
    assert len(closures_esc) == 1

    # Verify that resolve_clone_generator_subroutine_outputs succeeds for the eager case
    u1 = {"start": 2, "end": 4, "file": "f1.py", "source_text": code}
    u2 = {"start": 2, "end": 4, "file": "f2.py", "source_text": code}
    s1 = {"has_yield": True, "inputs": [], "outputs": ["total"], "definite_stores": ["total"]}
    s2 = {"has_yield": True, "inputs": [], "outputs": ["total"], "definite_stores": ["total"]}
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res == (["total"], ["total"])


def test_async_generator_try_without_yield_not_flagged_as_hazard() -> None:
    """Verifies that an async generator with try/except around an await
    (no yield inside the try) is not flagged as a control flow hazard."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )

    code_safe = (
        "async def fetch_stream():\n"
        "    try:\n"
        "        data = await fetch()\n"
        "    except Exception:\n"
        "        data = []\n"
        "    yield data\n"
    )
    unit_safe = {
        "file": "f.py",
        "start": 1,
        "end": 6,
        "kind": "function",
        "source_text": code_safe,
    }
    scope_safe = inspect_single_unit_scope(unit_safe)
    assert scope_safe["has_yield"] is True
    assert scope_safe["has_try_or_with"] is False
    assert has_async_generator_delegation_hazard(scope_safe) is False

    code_hazard = (
        "async def fetch_stream_hazard():\n"
        "    try:\n"
        "        yield 1\n"
        "    finally:\n"
        "        cleanup()\n"
    )
    unit_hazard = {
        "file": "f.py",
        "start": 1,
        "end": 5,
        "kind": "function",
        "source_text": code_hazard,
    }
    scope_hazard = inspect_single_unit_scope(unit_hazard)
    assert scope_hazard["has_yield"] is True
    assert scope_hazard["has_try_or_with"] is True
    assert has_async_generator_delegation_hazard(scope_hazard) is True


def test_handbuilt_unit_without_kind_treated_as_subroutine_mid_function() -> None:
    """Verifies that hand-built units lacking kind or is_subroutine are detected
    structurally as subroutines when located mid-function to prevent silent miscompilation."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        is_subroutine_unit,
    )

    code = (
        "def worker(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    c = b + 3\n"
        "    return c\n"
    )
    # Unit covering only lines 2-3 (statement block mid-function)
    unit_mid = {"file": "worker.py", "start": 2, "end": 3, "source_text": code}
    assert is_subroutine_unit(unit_mid, source_text=code) is True

    scope = inspect_single_unit_scope(unit_mid)
    # Because it is treated as a subroutine, variable assignments are captured as outputs
    assert "b" in scope["outputs"]
    assert "a" in scope["inputs"] or "x" in scope["inputs"]


def test_concurrent_generate_refactoring_patch_calls(tmp_path: Path) -> None:
    """Verifies that concurrent calls to generate_refactoring_patch do not corrupt
    or wipe shared caches or interfere with each other."""
    import concurrent.futures
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        generate_refactoring_patch,
    )

    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    code1 = (
        "def compute1(x: int) -> int:\n"
        "    a = x * 2\n"
        "    b = a + 3\n"
        "    return b\n"
    )
    code2 = (
        "def compute2(y: int) -> int:\n"
        "    a = y * 2\n"
        "    b = a + 3\n"
        "    return b\n"
    )
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    clones = [
        (
            1.0,
            {
                "file": str(f1),
                "start": 1,
                "end": 4,
                "name": "compute1",
                "kind": "function",
                "source_text": code1,
            },
            {
                "file": str(f2),
                "start": 1,
                "end": 4,
                "name": "compute2",
                "kind": "function",
                "source_text": code2,
            },
        )
    ]

    def _run_worker(worker_id: int) -> tuple[int, str | None]:
        patch_text = generate_refactoring_patch(
            clones,
            repo_root=str(tmp_path),
        )
        return worker_id, patch_text

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_run_worker, i) for i in range(8)]
        results = [f.result() for f in futures]

    assert len(results) == 8
    first_patch_text = results[0][1]
    assert first_patch_text is not None
    assert len(first_patch_text) > 0
    for _, patch_text in results[1:]:
        assert patch_text == first_patch_text


def test_clear_dataflow_cache_public_export() -> None:
    """Verifies that clear_dataflow_cache is publicly exported and resets caches."""
    import pydoppelgangerhunt.fixer as fixer_mod
    import pydoppelgangerhunt.fixer.dataflow as dataflow_mod

    assert "clear_dataflow_cache" in fixer_mod.__all__
    assert "clear_dataflow_cache" in dataflow_mod.__all__
    assert callable(fixer_mod.clear_dataflow_cache)
    assert fixer_mod.clear_dataflow_cache is dataflow_mod.clear_dataflow_cache

    # Seed caches
    with _downstream_cache_lock:
        _downstream_reads_cache[("dummy",)] = frozenset({"x"})
        _scope_closures_cache[ast.Pass()] = []

    assert len(_downstream_reads_cache) > 0
    assert len(_scope_closures_cache) > 0

    clear_dataflow_cache()

    assert len(_downstream_reads_cache) == 0
    assert len(_scope_closures_cache) == 0

    # Ensure backwards compatibility alias works identically
    with _downstream_cache_lock:
        _downstream_reads_cache[("dummy",)] = frozenset({"y"})
    _clear_downstream_reads_cache()
    assert len(_downstream_reads_cache) == 0


def test_extract_scope_local_names() -> None:
    """Verifies extraction of parameters and local stores without comprehension leakage."""
    code = (
        "def outer(pos_a, /, pos_b, *vargs, kw_a, **kwargs):\n"
        "    x = 10\n"
        "    y += 5\n"
        "    (z := 20)\n"
        "    for item in [1, 2]:\n"
        "        pass\n"
        "    with open('f') as f_obj:\n"
        "        pass\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as err:\n"
        "        pass\n"
        "    def inner_fn():\n"
        "        inner_leak = 99\n"
        "    class InnerClass:\n"
        "        class_attr = 1\n"
        "    import math as custom_math\n"
        "    from os import path as custom_path\n"
        "    comp_res = [comp_var for comp_var in [1, 2]]\n"
        "    gen_res = (gen_var for gen_var in [3, 4])\n"
        "    walrus_comp = [(walrus_leak := v) for v in [5, 6]]\n"
    )
    tree = ast.parse(code)
    func_node = tree.body[0]
    names = _extract_scope_local_names(func_node)

    # Parameters
    assert {"pos_a", "pos_b", "vargs", "kw_a", "kwargs"}.issubset(names)
    # Outer local stores
    assert {"x", "y", "z", "item", "f_obj", "err", "comp_res", "gen_res", "walrus_comp", "walrus_leak"}.issubset(names)
    # Inner definitions
    assert "inner_fn" in names
    assert "InnerClass" in names
    assert "custom_math" in names
    assert "custom_path" in names
    # Function itself
    assert "outer" in names

    # Comprehension iteration variables and inner function variables MUST NOT leak
    assert "comp_var" not in names
    assert "gen_var" not in names
    assert "v" not in names
    assert "inner_leak" not in names
    assert "class_attr" not in names
    assert "math" not in names
    assert "path" not in names


def test_eager_builtin_scope_shadowing_scenarios() -> None:
    """Verifies that shadowed built-in eager consumers treat generator
    expressions as escaping closures."""
    # 1. Shadowed by local function definition
    code_fn = (
        "def compute():\n"
        "    def sum(gen):\n"
        "        return lambda: next(gen)\n"
        "    data = [1, 2, 3]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_fn = ast.parse(code_fn)
    closures_fn = _collect_scope_closures(tree_fn.body[0])
    # Expect 2 closures: inner def sum and the escaping genexp passed to sum(...)
    assert len(closures_fn) == 2

    # 2. Shadowed by function parameter
    code_param = (
        "def compute(sum, data):\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_param = ast.parse(code_param)
    closures_param = _collect_scope_closures(tree_param.body[0])
    assert len(closures_param) == 1

    # 3. Shadowed by local variable assignment
    code_assign = (
        "def compute():\n"
        "    sum = lambda g: list(g)\n"
        "    data = [1, 2, 3]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_assign = ast.parse(code_assign)
    closures_assign = _collect_scope_closures(tree_assign.body[0])
    # Lambda in sum assignment + escaping genexp
    assert len(closures_assign) == 2

    # 4. Shadowed by import
    code_import = (
        "def compute():\n"
        "    from my_math import sum\n"
        "    data = [1, 2, 3]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_import = ast.parse(code_import)
    closures_import = _collect_scope_closures(tree_import.body[0])
    assert len(closures_import) == 1

    # 5. Comprehension dummy variable does NOT shadow outer built-in call
    code_no_leak = (
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    # min is used inside comprehension, but should not shadow subsequent min()\n"
        "    dummy = sum(min for min in data)\n"
        "    total = min(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_no_leak = ast.parse(code_no_leak)
    closures_no_leak = _collect_scope_closures(tree_no_leak.body[0])
    # Neither sum nor min are shadowed in compute scope -> 0 closures
    assert len(closures_no_leak) == 0

    # 6. Shadowed by walrus operator inside comprehension per PEP 572
    code_walrus_comp = (
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    dummy = [(sum := (lambda g: list(g))) for _ in data]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_walrus_comp = ast.parse(code_walrus_comp)
    closures_walrus = _collect_scope_closures(tree_walrus_comp.body[0])
    # Lambda in sum assignment + escaping genexp
    assert len(closures_walrus) == 2

    # 7. Wildcard star import dynamically shadows eager built-in
    code_star = (
        "from numpy import *\n"
        "\n"
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    total = sum(x * 2 for x in data)\n"
        "    yield total\n"
    )
    tree_star = ast.parse(code_star)
    closures_star = _collect_scope_closures(tree_star.body[1], module_tree=tree_star)
    # Expect 1 closure because sum is treated as potentially shadowed/escaping
    assert len(closures_star) == 1


def test_module_level_shadowing_of_eager_builtins() -> None:
    """Verifies that module-level symbols shadowing eager built-ins prevent
    inner generator expressions from being treated as eagerly consumed."""
    # Module-level import shadows sum
    code_mod_import = (
        "from numpy import sum\n"
        "\n"
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    res = sum(x * 2 for x in data)\n"
        "    yield res\n"
    )
    tree_mod_import = ast.parse(code_mod_import)
    fn_node = tree_mod_import.body[1]
    # Without module_tree: sum looks like unshadowed built-in, 0 closures
    closures_without = _collect_scope_closures(fn_node)
    assert len(closures_without) == 0

    # With module_tree: module-level sum is detected, genexp escapes as closure
    closures_with = _collect_scope_closures(fn_node, module_tree=tree_mod_import)
    assert len(closures_with) == 1

    # Module-level function definition shadows any
    code_mod_fn = (
        "def any(iterable):\n"
        "    return list(iterable)\n"
        "\n"
        "def check():\n"
        "    items = [False, True]\n"
        "    ok = any(x for x in items)\n"
        "    yield ok\n"
    )
    tree_mod_fn = ast.parse(code_mod_fn)
    fn_check = tree_mod_fn.body[1]
    closures_check = _collect_scope_closures(fn_check, module_tree=tree_mod_fn)
    assert len(closures_check) == 1

    # Module-level wildcard star import shadows sum
    code_mod_star = (
        "from numpy import *\n"
        "\n"
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    res = sum(x * 2 for x in data)\n"
        "    yield res\n"
    )
    tree_mod_star = ast.parse(code_mod_star)
    fn_star = tree_mod_star.body[1]
    closures_mod_star = _collect_scope_closures(fn_star, module_tree=tree_mod_star)
    assert len(closures_mod_star) == 1

    # Module-level wildcard star import nested inside a compound statement (e.g. if/try)
    code_compound_star = (
        "if True:\n"
        "    from numpy import *\n"
        "\n"
        "def compute():\n"
        "    data = [1, 2, 3]\n"
        "    res = sum(x * 2 for x in data)\n"
        "    yield res\n"
    )
    tree_compound_star = ast.parse(code_compound_star)
    fn_compound = tree_compound_star.body[1]
    closures_compound_star = _collect_scope_closures(fn_compound, module_tree=tree_compound_star)
    assert len(closures_compound_star) == 1


def test_has_wildcard_star_import_in_module_compound_stmt() -> None:
    """Verifies that _has_wildcard_star_import detects wildcard imports nested in
    module-level compound statements like if, try, etc."""
    from pydoppelgangerhunt.fixer._df_closures import _has_wildcard_star_import

    code = (
        "try:\n"
        "    from fast_math import *\n"
        "except ImportError:\n"
        "    from math import *\n"
        "\n"
        "def compute():\n"
        "    pass\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[1]
    assert _has_wildcard_star_import(fn_node, module_tree=tree) is True



def test_eager_consumed_genexp_non_str_receiver_join() -> None:
    """Verifies that .join() calls only count as eager consumption when the
    receiver is a confirmed string type, not arbitrary objects like Thread.join."""
    # 1. Non-string receiver (e.g. thread.join(genexp)) should NOT be eager
    code_thread = (
        "def worker():\n"
        "    import threading\n"
        "    t = threading.Thread()\n"
        "    t.join(x for x in [1, 2])\n"
        "    yield t\n"
    )
    tree_thread = ast.parse(code_thread)
    closures_thread = _collect_scope_closures(tree_thread.body[0])
    # Escaping genexp is captured
    assert len(closures_thread) == 1

    # 2. Confirmed string literal receiver: eager, NOT captured
    code_str_lit = (
        "def worker():\n"
        "    s = ', '.join(str(x) for x in [1, 2])\n"
        "    yield s\n"
    )
    tree_str_lit = ast.parse(code_str_lit)
    closures_str_lit = _collect_scope_closures(tree_str_lit.body[0])
    assert len(closures_str_lit) == 0

    # 3. Confirmed str.join or builtins.str.join: eager, NOT captured
    code_str_call = (
        "def worker():\n"
        "    import builtins\n"
        "    s1 = str.join(', ', (str(x) for x in [1, 2]))\n"
        "    s2 = builtins.str.join(', ', (str(y) for y in [3, 4]))\n"
        "    yield s1 + s2\n"
    )
    tree_str_call = ast.parse(code_str_call)
    closures_str_call = _collect_scope_closures(tree_str_call.body[0])
    assert len(closures_str_call) == 0


def test_resolve_clone_pair_outputs_forwards_source_text_and_tree() -> None:
    """Verifies resolve_clone_pair_outputs passes source_text and tree to is_subroutine_unit."""
    src1 = (
        "def outer1():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    src2 = (
        "def outer2():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    u1 = {"file": "mod1.py", "start": 2, "end": 3}
    u2 = {"file": "mod2.py", "start": 2, "end": 3}
    s1 = {"stores": ["x", "y"], "inputs": []}
    s2 = {"stores": ["x", "y"], "inputs": []}
    t1 = ast.parse(src1)
    t2 = ast.parse(src2)
    res = resolve_clone_pair_outputs(
        u1,
        u2,
        s1,
        s2,
        source_text1=src1,
        source_text2=src2,
        tree1=t1,
        tree2=t2,
    )
    assert res is not None


def test_is_eagerly_consumed_genexp_call_arg_validation() -> None:
    """Verifies that _is_eagerly_consumed_genexp checks that genexp is actually an argument."""
    # 1. Genexp called directly as func: (x for x in data)()
    genexp = ast.GeneratorExp(
        elt=ast.Name(id="x", ctx=ast.Load()),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="data", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        ],
    )
    call_as_func = ast.Call(func=genexp, args=[], keywords=[])
    parent_map: Dict[ast.AST, ast.AST] = {genexp: call_as_func}
    assert _is_eagerly_consumed_genexp(genexp, parent_map) is False

    # 2. Genexp passed as join argument: ",".join(x for x in data)
    join_call = ast.Call(
        func=ast.Attribute(value=ast.Constant(value=","), attr="join", ctx=ast.Load()),
        args=[genexp],
        keywords=[],
    )
    parent_map_join: Dict[ast.AST, ast.AST] = {genexp: join_call}
    assert _is_eagerly_consumed_genexp(genexp, parent_map_join) is True

    # 3. Genexp calling join: (x for x in data).join(",")
    join_call_wrong_pos = ast.Call(
        func=ast.Attribute(value=genexp, attr="join", ctx=ast.Load()),
        args=[ast.Constant(value=",")],
        keywords=[],
    )
    parent_map_wrong: Dict[ast.AST, ast.AST] = {genexp: join_call_wrong_pos}
    assert _is_eagerly_consumed_genexp(genexp, parent_map_wrong) is False


def test_is_confirmed_str_receiver_shadowed_builtins() -> None:
    """Verifies that _is_confirmed_str_receiver rejects builtins.str when builtins is shadowed."""
    from pydoppelgangerhunt.fixer._df_closures import _is_confirmed_str_receiver

    node_builtins = ast.Attribute(
        value=ast.Name(id="builtins", ctx=ast.Load()),
        attr="str",
        ctx=ast.Load(),
    )
    assert _is_confirmed_str_receiver(node_builtins, shadowed_names=None) is True
    assert _is_confirmed_str_receiver(node_builtins, shadowed_names={"builtins"}) is False

    node_dunder = ast.Attribute(
        value=ast.Name(id="__builtins__", ctx=ast.Load()),
        attr="str",
        ctx=ast.Load(),
    )
    assert _is_confirmed_str_receiver(node_dunder, shadowed_names=None) is True
    assert _is_confirmed_str_receiver(node_dunder, shadowed_names={"__builtins__"}) is False


def test_eager_consumed_genexp_shadowed_builtins_rejects_join() -> None:
    """Verifies that when builtins is shadowed by local assignment, builtins.str.join
    is not treated as an eager consumer."""
    from pydoppelgangerhunt.fixer._df_closures import _collect_scope_closures

    code = (
        "def worker():\n"
        "    builtins = object()\n"
        "    s = builtins.str.join(', ', (str(y) for y in [3, 4]))\n"
        "    yield s\n"
    )
    tree = ast.parse(code)
    closures = _collect_scope_closures(tree.body[0])
    assert len(closures) == 1


def test_parsed_tree_cache_key_without_redundant_source_text() -> None:
    """Verifies that _parsed_tree_cache keys on (content_hash, length) without full source text."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _get_or_parse_tree,
        _parsed_tree_cache,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()
    src = "def sample():\n    return 1 + 2\n"
    tree = _get_or_parse_tree(src)
    assert tree is not None
    assert len(_parsed_tree_cache) == 1
    key = next(iter(_parsed_tree_cache.keys()))
    assert len(key) == 2
    assert isinstance(key[0], str)
    assert isinstance(key[1], int)
    assert key[1] == len(src)
    clear_dataflow_cache()
    assert len(_parsed_tree_cache) == 0


def test_load_unit_file_text_synthetic_in_memory_fallback() -> None:
    """Verifies that _load_unit_file_text falls back to in-memory lines for virtual units."""
    unit = {
        "file": "nonexistent_virtual_module.py",
        "start": 1,
        "end": 2,
        "source_lines": ["def foo():\n", "    return 42\n"],
        "source_lines_is_sliced": False,
    }
    # When file does not exist on disk, fallback to in-memory source_lines
    text = _load_unit_file_text(unit)
    assert text == "def foo():\n    return 42\n"


def test_build_downstream_cache_key_content_digest_isolation() -> None:
    """Verifies that _build_downstream_cache_key shares entries across file paths and unit names."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        _build_downstream_cache_key,
    )

    code = (
        "def enclosing():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    return a + b\n"
    )
    unit1 = {"file": "file1.py", "name": "block1", "kind": "compound_block", "start": 2, "end": 3}
    unit2 = {"file": "file2.py", "name": "block2", "kind": "sliding_window", "start": 2, "end": 3}
    key1 = _build_downstream_cache_key(
        source_text=code,
        unit=unit1,
        u_start=2,
        u_end=3,
        candidates={"a", "b"},
        skip_pre_unit_closures=False,
    )
    key2 = _build_downstream_cache_key(
        source_text=code,
        unit=unit2,
        u_start=2,
        u_end=3,
        candidates={"a", "b"},
        skip_pre_unit_closures=False,
    )
    # Identical content digest and coordinates share cache entries regardless of file/name/kind
    assert key1 == key2


def test_extract_single_node_local_names_lambda_non_list_body() -> None:
    """Verifies that _extract_single_node_local_names safely processes non-list scope bodies."""
    expr = ast.parse("lambda x: (y := x + 1)").body[0].value  # type: ignore[attr-defined]
    names = _extract_single_node_local_names(expr)
    assert names == {"x", "y"}


def test_extract_single_node_local_names_wildcard_import_filtered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that _extract_single_node_local_names ignores '*' from wildcard star imports."""
    with caplog.at_level(logging.DEBUG):
        fn_def = ast.parse(
            "def func():\n"
            "    from math import *\n"
            "    import os\n"
            "    from sys import path as sys_path\n"
            "    val = 1\n"
        ).body[0]
        names = _extract_single_node_local_names(fn_def)
    assert "*" not in names
    assert names == {"func", "os", "sys_path", "val"}
    assert "Wildcard import detected ('from math import *')" in caplog.text


def test_subroutine_same_line_semicolon_in_with_without_end_col() -> None:
    """Verifies resolve_clone_generator_subroutine_outputs when a compound unit is followed
    by a semicolon statement on the same line inside a with block without explicit end_col."""
    src = (
        "def run():\n"
        "    with mgr: total = yield x; print(total)\n"
    )
    u = {"file": "m.py", "start": 2, "end": 2, "kind": "compound_block"}
    # Must resolve end_col via AST and detect print(total) as downstream read
    reads = collect_downstream_read_names(src, u, candidates={"total"})
    assert reads is not None
    assert "total" in reads

    u1 = dict(u, outputs=["total"])
    u2 = dict(u, outputs=["total"])
    scope = {"definite_stores": ["total"], "inputs": []}
    res = resolve_clone_generator_subroutine_outputs(
        u1, u2, scope, scope, source_text1=src, source_text2=src
    )
    # Intra-with reads are permitted; total is safely paired as an output
    assert res == (["total"], ["total"])


def test_resolve_clone_generator_subroutine_outputs_unflagged_sliced_lines() -> None:
    """Verifies that unflagged sliced source_lines fails closed when resolving
    subroutine outputs.
    """
    u1 = {
        "file": "virtual1.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = yield\n", "    total += x\n", "    yield total\n"],
    }
    u2 = {
        "file": "virtual2.py",
        "start": 10,
        "end": 12,
        "source_lines": ["    x = yield\n", "    total += x\n", "    yield total\n"],
    }
    scope1 = {"inputs": ["total"], "outputs": ["total"], "definite_stores": ["total"]}
    scope2 = {"inputs": ["total"], "outputs": ["total"], "definite_stores": ["total"]}
    # Sliced lines with len(source_lines) < end fail closed because full file text is unavailable
    res = resolve_clone_generator_subroutine_outputs(u1, u2, scope1, scope2)
    assert res is None


def test_downstream_cache_key_differentiation_for_unit_inputs() -> None:
    """Verifies that unit_inputs are normalized into the downstream cache key."""
    clear_dataflow_cache()
    src = (
        "def process(batches):\n"
        "    total = 0\n"
        "    for batch in batches:\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield total\n"
    )
    u_with_inputs = {
        "file": "calc.py",
        "start": 4,
        "end": 6,
        "kind": "for",
        "inputs": ["total"],
    }
    u_without_inputs = {
        "file": "calc.py",
        "start": 4,
        "end": 6,
        "kind": "for",
    }
    u_empty_inputs = {
        "file": "calc.py",
        "start": 4,
        "end": 6,
        "kind": "for",
        "inputs": [],
    }
    key1 = _build_downstream_cache_key(
        src, u_with_inputs, 4, 6, candidates={"total"}, skip_pre_unit_closures=False
    )
    key2 = _build_downstream_cache_key(
        src, u_without_inputs, 4, 6, candidates={"total"}, skip_pre_unit_closures=False
    )
    key3 = _build_downstream_cache_key(
        src, u_empty_inputs, 4, 6, candidates={"total"}, skip_pre_unit_closures=False
    )
    assert key1.unit_inputs == frozenset({"total"})
    assert key2.unit_inputs is None
    assert key3.unit_inputs == frozenset()
    assert key1 != key2
    assert key1 != key3

    # collect_downstream_read_names differentiates between populated and empty inputs
    reads1 = collect_downstream_read_names(src, u_with_inputs, candidates={"total"})
    reads2 = collect_downstream_read_names(src, u_empty_inputs, candidates={"total"})
    assert reads1 is not None and "total" in reads1
    assert reads2 is not None and "total" not in reads2


def test_build_downstream_cache_key_unhashable_and_coercion_guards() -> None:
    """Verifies that _build_downstream_cache_key safely coerces and filters unhashable inputs."""
    src = "def f(x):\n    return x\n"
    u_unhashable = {
        "file": "test.py",
        "start": 1,
        "end": 2,
        "inputs": [["nested", "list"], {"unhashable": "dict"}, "valid_var", 99],
    }
    key = _build_downstream_cache_key(
        src,
        u_unhashable,
        1,
        2,
        candidates={"valid_var"},
        skip_pre_unit_closures=False,
    )
    assert key.unit_inputs == frozenset({"valid_var", "99"})

    # Also test unhashable or non-string candidate elements
    key_cands = _build_downstream_cache_key(
        src,
        {"file": "test.py", "start": 1, "end": 2},
        1,
        2,
        candidates=[["bad_cand"], "good_cand", 10],  # type: ignore[arg-type]
        skip_pre_unit_closures=False,
    )
    assert key_cands.candidates == frozenset({"good_cand", "10"})


def test_downstream_cache_lock_reentrancy() -> None:
    """Verifies that _downstream_cache_lock is re-entrant to prevent thread deadlocks."""
    assert isinstance(_downstream_cache_lock, type(threading.RLock()))
    # Must support nested acquisition on the current thread without blocking
    with _downstream_cache_lock:
        with _downstream_cache_lock:
            pass


def test_store_lru_cache_entry_double_checked_locking() -> None:
    """Verifies that _store_lru_cache_entry returns previously cached value on race condition."""
    from collections import OrderedDict
    from pydoppelgangerhunt.fixer._df_cache import _store_lru_cache_entry

    cache: OrderedDict[str, str] = OrderedDict()
    res1 = _store_lru_cache_entry(cache, "k1", "val1", 10)
    assert res1 == "val1"
    assert cache["k1"] == "val1"

    # Simulate another thread having stored "val1" before our thread stores "val1_duplicate"
    res2 = _store_lru_cache_entry(cache, "k1", "val1_duplicate", 10)
    assert res2 == "val1"
    assert cache["k1"] == "val1"


def test_is_ast_node_within_unit_bounds_unresolved_column_fallback() -> None:
    """Verifies that _is_ast_node_within_unit_bounds falls back to whole-line containment
    when u_end_col is None, on both single-line and multi-line units."""
    sample_node = ast.Name(id="x", ctx=ast.Load())
    sample_node.lineno = 5
    sample_node.col_offset = 12

    # Single-line unit: lines 5 to 5 without column information
    assert _is_ast_node_within_unit_bounds(sample_node, 5, 5, u_end_col=None) is True
    # Single-line unit with column bound after node
    assert _is_ast_node_within_unit_bounds(sample_node, 5, 5, u_end_col=15) is True
    # Single-line unit with column bound before node
    assert _is_ast_node_within_unit_bounds(sample_node, 5, 5, u_end_col=10) is False

    # Multi-line unit: node on final line without column information
    assert _is_ast_node_within_unit_bounds(sample_node, 2, 5, u_end_col=None) is True
    # Multi-line unit: node strictly outside line bounds
    assert _is_ast_node_within_unit_bounds(sample_node, 6, 8, u_end_col=None) is False
    assert _is_ast_node_within_unit_bounds(sample_node, 1, 4, u_end_col=None) is False

    # Node missing lineno attribute
    nodeless = ast.Pass()
    assert _is_ast_node_within_unit_bounds(nodeless, 5, 5, u_end_col=None) is False


def test_single_line_unit_unresolved_end_col_live_in_and_loop_carried() -> None:
    """Verifies that single-line units without end column or precomputed inputs discover
    live-in variables and accurately propagate loop-carried downstream reads."""
    src = (
        "def run_loop(records):\n"
        "    running_sum = 0\n"
        "    for rec in records:\n"
        "        running_sum = running_sum + rec.val\n"
        "    return running_sum\n"
    )
    unit_def = {
        "file": "engine.py",
        "start": 4,
        "end": 4,
        "kind": "for",
    }
    tree = ast.parse(src)

    # 1. Live-in extraction directly discovers running_sum and rec
    live_ins = _extract_unit_live_in_names(tree, 4, 4, u_end_col=None)
    assert "running_sum" in live_ins
    assert "rec" in live_ins

    # 2. collect_downstream_read_names captures loop-carried accumulator read
    reads = collect_downstream_read_names(
        src, unit_def, candidates={"running_sum"}, tree=tree
    )
    assert reads is not None
    assert "running_sum" in reads


def test_classify_closure_position_boundary_at_end_col() -> None:
    """Verifies that a closure starting exactly at u_end_col is classified as is_post
    and not as intra-unit."""
    # Closure at line 4, col 20. Unit is line 2 to line 4, end_col 20.
    is_pre, is_intra, is_post = _classify_closure_position(
        (4, 20), u_start=2, u_end=4, u_end_col=20
    )
    assert is_post is True
    assert is_intra is False
    assert is_pre is False

    # Closure before end_col (col 19) is intra-unit
    is_pre, is_intra, is_post = _classify_closure_position(
        (4, 19), u_start=2, u_end=4, u_end_col=20
    )
    assert is_post is False
    assert is_intra is True


def test_enclosing_try_reads_outputs_same_line_unit_accumulator_in_with() -> None:
    """Verifies that intra-with reads do not trigger enclosing with cleanup rejection,
    while downstream reads after the with block exit fail closed."""
    src = (
        "with lock:\n"
        "    init_step = 1; acc = acc + 1; cleanup_flag = True\n"
    )
    tree = ast.parse(src)
    # Unit is 'acc = acc + 1' on line 2, outputs={'acc'}
    result = _enclosing_try_reads_outputs(
        tree, u_start=2, u_end=2, outputs={"acc"}
    )
    assert result is False

    # Intra-with subsequent reads on line 2 do not trigger cleanup rejection
    src_read = (
        "with lock:\n"
        "    init_step = 1; acc = acc + 1; print(acc)\n"
    )
    tree_read = ast.parse(src_read)
    result_read = _enclosing_try_reads_outputs(
        tree_read, u_start=2, u_end=2, outputs={"acc"}
    )
    assert result_read is False

    # Reads downstream after the with block exit fail closed
    src_after = (
        "with lock:\n"
        "    init_step = 1; acc = acc + 1; cleanup_flag = True\n"
        "print(acc)\n"
    )
    tree_after = ast.parse(src_after)
    result_after = _enclosing_try_reads_outputs(
        tree_after, u_start=2, u_end=2, outputs={"acc"}
    )
    assert result_after is True


def test_extract_unit_end_col_precedence() -> None:
    """Verifies that _extract_unit_end_col prioritizes canonical end_col over end_col_offset."""
    from pydoppelgangerhunt.fixer.dataflow import _extract_unit_end_col

    # end_col takes precedence over end_col_offset
    unit_both = {"end_col": 25, "end_col_offset": 10}
    assert _extract_unit_end_col(unit_both) == 25

    # Fallback to end_col_offset when end_col is missing
    unit_offset_only = {"end_col_offset": 15}
    assert _extract_unit_end_col(unit_offset_only) == 15


def test_enclosing_with_loop_carried_read_at_loop_top_rejected() -> None:
    """Verifies that a unit inside an enclosing with block whose output is read at
    the top of an enclosing loop is identified as reading outputs downstream."""
    from pydoppelgangerhunt.fixer.dataflow import _enclosing_try_reads_outputs

    src = (
        "def run(items):\n"
        "    for x in items:\n"
        "        print(total)\n"
        "        with suppress_context:\n"
        "            total = yield x\n"
    )
    tree = ast.parse(src)
    # Unit is line 5 ('total = yield x'), with is lines 4-5, loop is lines 2-5
    # Print at line 3 is before the with statement but inside the enclosing loop
    fn_node = tree.body[0]
    reads = _enclosing_try_reads_outputs(
        fn_node, u_start=5, u_end=5, outputs={"total"}
    )
    assert reads is True


def test_pair_clone_outputs_allow_permuted_outputs() -> None:
    """Verifies that _pair_clone_outputs rejects permuted outputs when disabled
    and pairs positionally when enabled."""
    from pydoppelgangerhunt.fixer.dataflow import _pair_clone_outputs

    u1_outs = ["x", "y"]
    u2_outs = ["y", "x"]

    # When allow_permuted_outputs is False, differing order fails closed
    assert _pair_clone_outputs(u1_outs, u2_outs, allow_permuted_outputs=False) == []

    # Enabled allows positional 1-to-1 pairing for permuted outputs
    assert _pair_clone_outputs(u1_outs, u2_outs, allow_permuted_outputs=True) == [
        ("x", "y"),
        ("y", "x"),
    ]

    # Partial conflicting names fail closed when allow_permuted_outputs is False
    assert _pair_clone_outputs(["x", "y"], ["y", "z"], allow_permuted_outputs=False) == []

    # Enabled allows positional 1-to-1 pairing
    assert _pair_clone_outputs(["x", "y"], ["y", "z"], allow_permuted_outputs=True) == [
        ("x", "y"),
        ("y", "z"),
    ]


def test_execute_read_pass_and_collect_closures_scalar_body() -> None:
    """Verifies that _execute_read_pass and _collect_scope_closures handle expression
    scopes with scalar bodies defensibly without raising TypeError."""
    lambda_node = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x", lineno=1, col_offset=7)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=ast.Name(id="x", ctx=ast.Load(), lineno=1, col_offset=10),
        lineno=1,
        col_offset=0,
    )

    # 1. _execute_read_pass on lambda node does not raise TypeError
    reads = _execute_read_pass(
        lambda_node,
        _DownstreamReadVisitor,
        u_start=1,
        u_end=1,
        u_end_col=None,
        candidates={"x"},
        enclosing_loops=[],
        pass_mode=_VisitorPassMode.AFTER_UNIT,
        is_module=False,
    )
    assert isinstance(reads, set)

    # 2. _collect_scope_closures on lambda node does not raise TypeError
    closures = _collect_scope_closures(lambda_node)
    assert isinstance(closures, list)


def test_extract_unit_live_in_names_prunes_nested_scope_params_and_comprehensions() -> None:
    """Verifies _extract_unit_live_in_names ignores parameters of nested functions and
    lambdas, as well as comprehension targets, while preserving true live-in variables."""
    code = (
        "def outer():\n"
        "    f = lambda p: p + a\n"
        "    g = (x for x in items)\n"
        "    l = [i * 2 for i in numbers]\n"
        "    s = {k: v for k, v in pairs}\n"
        "    def inner(q):\n"
        "        temp = q + 1\n"
        "        return temp + outer_val\n"
        "    res = inner(5)\n"
    )
    tree = ast.parse(code)
    # Unit covering the statements inside outer() (lines 2 to 9)
    live_ins = _extract_unit_live_in_names(tree, 2, 9, candidates=None)
    assert "p" not in live_ins
    assert "x" not in live_ins
    assert "i" not in live_ins
    assert "k" not in live_ins
    assert "v" not in live_ins
    assert "q" not in live_ins
    assert "temp" not in live_ins
    assert "inner" not in live_ins
    assert live_ins == {"a", "items", "numbers", "pairs", "outer_val"}


def test_downstream_read_visitor_augassign_unresolved_boundary_does_not_kill() -> None:
    """Verifies that an AugAssign on u_end does not kill variable when u_end_col is None."""
    from pydoppelgangerhunt.fixer._df_visitors import _DownstreamReadVisitor, _VisitorPassMode

    code = "x += 1\ny = x\n"
    tree = ast.parse(code)

    # 1. Unit ends at line 1 with u_end_col=None (unresolved boundary: x is not killed)
    visitor1 = _DownstreamReadVisitor(
        u_start=1,
        u_end=1,
        u_end_col=None,
        candidates={"x"},
        enclosing_loops=[],
        pass_mode=_VisitorPassMode.AFTER_UNIT,
        is_module_scope=True,
    )
    visitor1.visit(tree)
    assert "x" not in visitor1.killed

    # 2. Unit ends before line 1 (x += 1 is clearly post-unit: x is killed)
    visitor2 = _DownstreamReadVisitor(
        u_start=0,
        u_end=0,
        u_end_col=None,
        candidates={"x"},
        enclosing_loops=[],
        pass_mode=_VisitorPassMode.AFTER_UNIT,
        is_module_scope=True,
    )
    visitor2.visit(tree)
    assert "x" in visitor2.killed

    # 3. Unit ends at line 1 with resolved u_end_col=0 (x is killed)
    visitor3 = _DownstreamReadVisitor(
        u_start=1,
        u_end=1,
        u_end_col=0,
        candidates={"x"},
        enclosing_loops=[],
        pass_mode=_VisitorPassMode.AFTER_UNIT,
        is_module_scope=True,
    )
    visitor3.visit(tree)
    assert "x" in visitor3.killed


def test_unit_live_in_visitor_flow_sensitivity_and_non_name_targets() -> None:
    """Verifies flow sensitivity for branches, bare AnnAssign, and non-Name AugAssign."""
    from pydoppelgangerhunt.fixer._df_visitors import _extract_unit_live_in_names

    # 1. Conditional store followed by in-place update: acc is live-in
    code1 = (
        "def f(c, x):\n"
        "    if c:\n"
        "        acc = 0\n"
        "    acc += x\n"
    )
    tree1 = ast.parse(code1)
    live_ins1 = _extract_unit_live_in_names(tree1, 2, 4, candidates={"acc", "x", "c"})
    assert "acc" in live_ins1
    assert "x" in live_ins1
    assert "c" in live_ins1

    # 2. Non-Name AugAssign target: counts and i are live-in
    code2 = (
        "def g(counts, i):\n"
        "    counts[i] += 1\n"
    )
    tree2 = ast.parse(code2)
    live_ins2 = _extract_unit_live_in_names(tree2, 2, 2, candidates={"counts", "i"})
    assert "counts" in live_ins2
    assert "i" in live_ins2

    # 3. Bare AnnAssign does not mark variable as definitely stored
    code3 = (
        "def h():\n"
        "    val: int\n"
        "    res = val + 1\n"
    )
    tree3 = ast.parse(code3)
    live_ins3 = _extract_unit_live_in_names(tree3, 2, 3, candidates={"val", "res"})
    assert "val" in live_ins3
    assert "res" not in live_ins3


def test_pair_clone_outputs_role_swapped_rejection_and_positional_pairing() -> None:
    """Verifies that role-swapped clones ([x, y] vs [y, x]) are rejected under
    allow_permuted_outputs=False, but pair positionally when allow_permuted_outputs=True,
    while alpha-renamed outputs ([x, y] vs [a, b]) pair positionally by default."""
    from pydoppelgangerhunt.fixer.dataflow import _pair_clone_outputs

    # Role-swapped clones sharing the same variable names in reverse order
    # Clone 1: x = a * 10; y = b * 20 -> outputs: [x, y]
    # Clone 2: y = a * 10; x = b * 20 -> outputs: [y, x]
    # Under allow_permuted_outputs=False, must reject to prevent silent miscompile
    assert _pair_clone_outputs(["x", "y"], ["y", "x"], allow_permuted_outputs=False) == []

    # When allow_permuted_outputs=True, pairs positionally
    assert _pair_clone_outputs(["x", "y"], ["y", "x"], allow_permuted_outputs=True) == [
        ("x", "y"),
        ("y", "x"),
    ]

    # Pure alpha-renaming with disjoint names pairs positionally by default
    assert _pair_clone_outputs(["x", "y"], ["a", "b"], allow_permuted_outputs=False) == [
        ("x", "a"),
        ("y", "b"),
    ]


def test_collect_downstream_reads_wildcard_import_module_scope() -> None:
    """Verifies that downstream wildcard imports in module scope treat candidates
    as dynamically read, failing closed."""
    from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names

    code = (
        "x = 10\n"
        "from math import *\n"
        "y = 20\n"
    )
    unit = {"start": 1, "end": 1, "inputs": []}
    candidates = {"x"}

    reads = collect_downstream_read_names(code, unit, candidates=candidates)
    assert reads is not None
    assert "x" in reads

    # A module-level wildcard star import does not treat function-local variables as read
    code_fn = (
        "from math import *\n"
        "\n"
        "def compute():\n"
        "    x = 10\n"
        "    y = 20\n"
        "    return y\n"
    )
    unit_fn = {"start": 4, "end": 4, "inputs": []}
    reads_fn = collect_downstream_read_names(code_fn, unit_fn, candidates={"x"})
    assert reads_fn is not None
    assert "x" not in reads_fn


def test_downstream_read_visitor_wildcard_import_dynamic_read() -> None:
    """Verifies that _DownstreamReadVisitor marks has_dynamic_read and records
    reads for all candidates on encountering a downstream wildcard import."""
    from pydoppelgangerhunt.fixer._df_visitors import _DownstreamReadVisitor

    code = (
        "x = 1\n"
        "from util import *\n"
    )
    tree = ast.parse(code)
    visitor = _DownstreamReadVisitor(
        u_start=1,
        u_end=1,
        u_end_col=None,
        candidates={"x", "y"},
        is_module_scope=True,
    )
    for stmt in tree.body:
        visitor.visit(stmt)

    assert visitor.has_dynamic_read is True
    assert "x" in visitor.loaded
    assert "y" in visitor.loaded


def test_get_or_compute_scope_cached_canonical_instance_return() -> None:
    """Verifies that _get_or_compute_scope_cached returns the canonical cached instance
    when another thread populates the cache concurrently during computation."""
    from collections import OrderedDict
    from pydoppelgangerhunt.fixer._df_cache import _get_or_compute_scope_cached

    cache: OrderedDict[str, list[str]] = OrderedDict()
    canonical_list: list[str] = ["canonical"]

    def compute_fn() -> list[str]:
        # Simulate another thread storing the canonical entry during our compute_fn
        cache["scope_node"] = canonical_list
        return ["duplicate_computed"]

    result = _get_or_compute_scope_cached(cache, "scope_node", compute_fn)
    assert result is canonical_list


def test_check_enclosing_try_cleanup_hazard_precomputed_scope_and_bounds() -> None:
    """Verifies that _check_enclosing_try_cleanup_hazard correctly accepts precomputed
    scope_node and bounds without redundant lookups."""
    from pydoppelgangerhunt.fixer._df_pairing import (
        _check_enclosing_try_cleanup_hazard,
        _validate_try_cleanup_hazard,
    )
    from pydoppelgangerhunt.fixer._df_hazards import _resolve_downstream_scope_node
    from pydoppelgangerhunt.fixer._df_cache import _get_valid_unit_bounds

    code = (
        "def run():\n"
        "    try:\n"
        "        x = 10\n"
        "    finally:\n"
        "        print(x)\n"
    )
    tree = ast.parse(code)
    unit = {"start": 3, "end": 3, "inputs": []}
    scope_node = _resolve_downstream_scope_node(code, unit, tree)
    bounds = _get_valid_unit_bounds(unit)

    # Precomputed scope_node and bounds passed directly
    has_hazard = _check_enclosing_try_cleanup_hazard(
        unit, ["x"], code, tree, scope_node=scope_node, bounds=bounds
    )
    assert has_hazard is True

    # Validate try cleanup hazard returns rejection
    resolution = _validate_try_cleanup_hazard(
        unit, ["x"], code, tree, side_idx=0, context="test clone"
    )
    assert resolution is not None
    assert resolution.skip_reason == "enclosing try or with cleanup hazard"


def test_evict_tree_scopes_purges_scope_roots_and_retains_unrelated_entries() -> None:
    """Verifies that _evict_tree_scopes accurately purges scope roots and leaves unrelated entries intact."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_parent_maps,
        _scope_stmts_by_end,
        _scope_closures_cache,
        _scope_loops_cache,
        _scope_try_with_cache,
        _get_scope_parent_map,
        _get_scope_stmts_by_end_lineno,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()

    # None and non-AST inputs do not raise
    _evict_tree_scopes(None)
    _evict_tree_scopes("not_an_ast")  # type: ignore[arg-type]

    code1 = "def f1(a):\n    return a + 1\n"
    code2 = "def f2(b):\n    return b + 2\n"

    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)

    fn1 = tree1.body[0]
    fn2 = tree2.body[0]

    # Populate scope metadata for tree1 and tree2
    _get_scope_parent_map(fn1)
    _get_scope_stmts_by_end_lineno(fn1)
    _scope_closures_cache[fn1] = []
    _scope_closures_cache[(fn1, tree1)] = []
    _scope_loops_cache[fn1] = []
    _scope_try_with_cache[fn1] = []

    _get_scope_parent_map(fn2)
    _get_scope_stmts_by_end_lineno(fn2)
    _scope_closures_cache[fn2] = []
    _scope_closures_cache[(fn2, tree2)] = []
    _scope_loops_cache[fn2] = []
    _scope_try_with_cache[fn2] = []

    assert fn1 in _scope_parent_maps
    assert fn2 in _scope_parent_maps

    # Evict tree1
    _evict_tree_scopes(tree1)

    # tree1 scope entries are purged
    assert fn1 not in _scope_parent_maps
    assert fn1 not in _scope_stmts_by_end
    assert fn1 not in _scope_closures_cache
    assert (fn1, tree1) not in _scope_closures_cache
    assert fn1 not in _scope_loops_cache
    assert fn1 not in _scope_try_with_cache

    # tree2 scope entries remain untouched
    assert fn2 in _scope_parent_maps
    assert fn2 in _scope_stmts_by_end
    assert fn2 in _scope_closures_cache
    assert (fn2, tree2) in _scope_closures_cache
    assert fn2 in _scope_loops_cache
    assert fn2 in _scope_try_with_cache

    clear_dataflow_cache()


def test_extract_scope_local_names_module_tree_single_visit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that _extract_scope_local_names does not redundantly re-extract
    module_tree names when traversing up to the module root."""
    import pydoppelgangerhunt.fixer._df_closures as df_closures

    code = (
        "GLOBAL_MOD_VAR = 100\n"
        "def outer_fn():\n"
        "    outer_var = 1\n"
        "    def inner_fn():\n"
        "        inner_var = 2\n"
        "        return inner_var\n"
    )
    tree = ast.parse(code)
    outer = tree.body[1]
    assert isinstance(outer, ast.FunctionDef)
    inner = outer.body[1]

    calls: list[ast.AST] = []
    orig_fn = df_closures._extract_single_node_local_names

    def _spy_extract(node: ast.AST) -> set[str]:
        calls.append(node)
        return orig_fn(node)

    monkeypatch.setattr(df_closures, "_extract_single_node_local_names", _spy_extract)

    names = df_closures._extract_scope_local_names(inner, module_tree=tree)

    assert "inner_var" in names
    assert "outer_var" in names
    assert "GLOBAL_MOD_VAR" in names

    # module_tree (tree) must be visited exactly once
    mod_visits = [n for n in calls if n is tree]
    assert len(mod_visits) == 1


def test_find_enclosing_loops_synthetic_node_none_lineno() -> None:
    """Verifies that _find_enclosing_loops handles synthetic AST nodes with
    explicit lineno=None without raising TypeError."""
    from pydoppelgangerhunt.fixer._df_hazards import (  # pylint: disable=import-outside-toplevel
        _find_enclosing_loops,
    )

    tree = ast.parse("for x in items:\n    pass\n")
    loop_node = tree.body[0]
    # Simulate a synthetic AST node where lineno/end_lineno are explicitly None
    loop_node.lineno = None  # type: ignore[assignment]
    loop_node.end_lineno = None  # type: ignore[assignment]

    loops = _find_enclosing_loops(tree, u_start=1, u_end=2)
    assert loops == []


def test_ast_node_bounds_synthetic_node_none_col_offset() -> None:
    """Verifies that _is_ast_node_within_unit_bounds and _is_node_after_unit
    handle synthetic nodes with explicit col_offset=None without raising TypeError."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=import-outside-toplevel
        _DownstreamReadVisitor,
        _is_ast_node_within_unit_bounds,
    )

    node = ast.Name(id="x", ctx=ast.Load())
    node.lineno = 5
    node.col_offset = None  # type: ignore[assignment]

    # Within bounds on line 5: falls back to True when col_offset is None
    assert _is_ast_node_within_unit_bounds(node, u_start=1, u_end=5, u_end_col=10)

    # After unit visitor: fails closed and treats same-line node on u_end as downstream
    visitor = _DownstreamReadVisitor(u_start=1, u_end=5, u_end_col=10)
    assert visitor._is_node_after_unit(node)


def test_u_start_col_loop_carried_read_and_unit_entry_killed() -> None:
    """Verifies that u_start_col prevents statements preceding the unit start on u_start
    from being misclassified as inside the unit."""
    from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names
    from pydoppelgangerhunt.fixer._df_visitors import (
        _is_ast_node_within_unit_bounds,
    )

    # 1. Direct boundary check on u_start with col_offset < u_start_col
    node = ast.Name(id="total", ctx=ast.Load())
    node.lineno = 2
    node.col_offset = 4
    assert not _is_ast_node_within_unit_bounds(
        node, u_start=2, u_end=2, u_end_col=30, u_start_col=15
    )
    node.col_offset = 18
    assert _is_ast_node_within_unit_bounds(
        node, u_start=2, u_end=2, u_end_col=30, u_start_col=15
    )

    # 2. Semicolon line with preceding read in an enclosing loop:
    code = (
        "def loop_gen(items):\n"
        "    for x in items:\n"
        "        use(total); total = x; yield total\n"
    )
    tree = ast.parse(code)
    unit = {
        "file": "test.py",
        "start": 3,
        "end": 3,
        "start_col": 20,
        "end_col": 42,
        "inputs": ["x"],
    }
    reads = collect_downstream_read_names(
        code, unit, candidates={"total", "x"}, tree=tree
    )
    assert reads is not None
    assert "total" in reads


def test_unit_live_in_loop_try_flow_sensitive() -> None:
    """Verifies that _UnitLiveInVisitor treats loops, try blocks, with blocks,
    and match statements flow-sensitively, preserving live-in detection for subsequent reads."""
    from pydoppelgangerhunt.fixer._df_visitors import _extract_unit_live_in_names

    # Inside a unit: a loop stores 'a', and then 'a' is read afterwards.
    # Since the loop may execute 0 times, 'a' must be marked as a live-in!
    code = (
        "def compute(items):\n"
        "    for x in items:\n"
        "        a = x\n"
        "    return a\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(
        fn_node, u_start=2, u_end=4, candidates={"a", "items"}
    )
    assert "a" in live_ins
    assert "items" in live_ins

    # Try block: 'b' is stored in body, but body may raise; read in finalbody or after
    code_try = (
        "def run():\n"
        "    try:\n"
        "        b = 10\n"
        "    except Exception:\n"
        "        pass\n"
        "    return b\n"
    )
    tree_try = ast.parse(code_try)
    fn_try = tree_try.body[0]
    live_ins_try = _extract_unit_live_in_names(
        fn_try, u_start=2, u_end=6, candidates={"b"}
    )
    assert "b" in live_ins_try

    # With block: 'c' is stored in with-body, but block may raise; read after
    code_with = (
        "def run_with(ctx):\n"
        "    with ctx:\n"
        "        c = 42\n"
        "    return c\n"
    )
    tree_with = ast.parse(code_with)
    fn_with = tree_with.body[0]
    live_ins_with = _extract_unit_live_in_names(
        fn_with, u_start=2, u_end=4, candidates={"c", "ctx"}
    )
    assert "c" in live_ins_with


def test_generator_subroutine_pair_preserves_try_hazard_skip_reason() -> None:
    """Verifies that resolve_clone_generator_subroutine_outputs returns the exact
    hazard resolution skip reason instead of a generic message."""
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_generator_subroutine_outputs

    code1 = (
        "def gen1():\n"
        "    try:\n"
        "        x = 1\n"
        "        yield x\n"
        "    finally:\n"
        "        use(x)\n"
    )
    code2 = (
        "def gen2():\n"
        "    x = 1\n"
        "    yield x\n"
    )
    u1 = {
        "file": "m1.py",
        "start": 3,
        "end": 4,
        "kind": "generator",
        "inputs": [],
        "outputs": ["x"],
    }
    u2 = {
        "file": "m2.py",
        "start": 2,
        "end": 3,
        "kind": "generator",
        "inputs": [],
        "outputs": ["x"],
    }
    scope1 = {"inputs": [], "outputs": ["x"], "definite_stores": ["x"]}
    scope2 = {"inputs": [], "outputs": ["x"], "definite_stores": ["x"]}
    res = resolve_clone_generator_subroutine_outputs(
        u1,
        u2,
        scope1,
        scope2,
        source_text1=code1,
        source_text2=code2,
        return_details=True,
    )
    assert res is not None
    assert res.outputs is None
    assert res.skip_reason == "enclosing try or with cleanup hazard"


def test_unit_live_in_visitor_try_orelse_no_handler_leak() -> None:
    """Verifies that variables stored in except handlers do not leak into try-else,
    correctly capturing live-ins in orelse."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code1 = (
        "def fn():\n"
        "    try:\n"
        "        do_something()\n"
        "    except Exception:\n"
        "        val = 1\n"
        "    else:\n"
        "        use(val)\n"
    )
    tree1 = ast.parse(code1)
    live_ins1 = _extract_unit_live_in_names(
        tree1.body[0], u_start=2, u_end=7, candidates={"val", "use", "do_something"}
    )
    assert "val" in live_ins1

    code2 = (
        "def fn():\n"
        "    try:\n"
        "        val = 1\n"
        "    except Exception:\n"
        "        pass\n"
        "    else:\n"
        "        use(val)\n"
    )
    tree2 = ast.parse(code2)
    live_ins2 = _extract_unit_live_in_names(
        tree2.body[0], u_start=2, u_end=7, candidates={"val", "use"}
    )
    assert "val" not in live_ins2


def test_unit_live_in_visitor_loop_orelse_resets_stored() -> None:
    """Verifies that loop orelse stores do not leak past the loop,
    and body stores do not falsely kill orelse live-ins."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    # 1. Body stores do not kill orelse live-ins because loop body might never execute
    code1 = (
        "def fn(items):\n"
        "    for x in items:\n"
        "        val = 1\n"
        "    else:\n"
        "        use(val)\n"
    )
    tree1 = ast.parse(code1)
    live_ins1 = _extract_unit_live_in_names(
        tree1.body[0], u_start=2, u_end=6, candidates={"val", "use", "items"}
    )
    assert "val" in live_ins1

    # 2. Orelse stores do not kill subsequent reads because loop might break before orelse
    code2 = (
        "def fn(items):\n"
        "    for x in items:\n"
        "        if test(x):\n"
        "            break\n"
        "    else:\n"
        "        val = 2\n"
        "    use(val)\n"
    )
    tree2 = ast.parse(code2)
    live_ins2 = _extract_unit_live_in_names(
        tree2.body[0], u_start=2, u_end=8, candidates={"val", "use", "items", "test"}
    )
    assert "val" in live_ins2


def test_nested_enclosing_loops_kills_do_not_leak_into_inner_loop() -> None:
    """Verifies that kills in outer enclosing loops do not prune variables
    accumulating across iterations of inner enclosing loops."""
    code = (
        "def process_groups(groups):\n"
        "    for group in groups:\n"
        "        total = 0\n"
        "        for batch in group:\n"
        "            for x in batch:\n"
        "                total += x\n"
        "                yield total\n"
    )
    unit = {
        "file": "test.py",
        "start": 5,
        "end": 7,
        "name": "unit",
        "kind": "compound_block",
    }
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_unit_live_in_visitor_except_handler_type_visited() -> None:
    """Verifies that dynamically bound exception handler types are captured as live-ins."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def fn():\n"
        "    try:\n"
        "        do_work()\n"
        "    except allowed_exceptions as err:\n"
        "        handle(err)\n"
    )
    tree = ast.parse(code)
    live_ins = _extract_unit_live_in_names(
        tree.body[0],
        u_start=2,
        u_end=5,
        candidates={"allowed_exceptions", "do_work", "handle"},
    )
    assert "allowed_exceptions" in live_ins


def test_unit_live_in_visitor_conditional_walrus_does_not_kill_subsequent_reads() -> None:
    """Verifies that walrus expressions in short-circuiting or ternary expressions
    do not kill subsequent live-in reads."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    # 1. Short-circuiting BoolOp: if flag and (total := 0):
    code1 = (
        "def fn(flag):\n"
        "    if flag and (total := 0):\n"
        "        pass\n"
        "    use(total)\n"
    )
    tree1 = ast.parse(code1)
    live_ins1 = _extract_unit_live_in_names(
        tree1.body[0], u_start=2, u_end=4, candidates={"total", "use", "flag"}
    )
    assert "total" in live_ins1

    # 2. Ternary IfExp: x = 1 if cond else (total := 0)
    code2 = (
        "def fn(cond):\n"
        "    x = 1 if cond else (total := 0)\n"
        "    use(total)\n"
    )
    tree2 = ast.parse(code2)
    live_ins2 = _extract_unit_live_in_names(
        tree2.body[0], u_start=2, u_end=3, candidates={"total", "use", "cond"}
    )
    assert "total" in live_ins2


def test_extract_nested_scope_free_reads_inner_generic_does_not_mask_outer_capture() -> None:
    """Verifies that inner generic definitions do not mask outer captures of the same name."""
    import sys
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_nested_scope_free_reads,
    )

    if sys.version_info < (3, 12):
        pytest.skip("PEP 695 type parameter syntax requires Python 3.12+")

    code = (
        "def outer():\n"
        "    def inner(default_val=T):\n"
        "        def helper[T](x: T) -> T:\n"
        "            return x\n"
        "        return helper(default_val)\n"
    )
    tree = ast.parse(code)
    inner_fn = tree.body[0].body[0]
    free = _extract_nested_scope_free_reads(inner_fn)
    assert "T" in free


def test_extract_nested_scope_free_reads_type_alias_preserves_external_reads() -> None:
    """Verifies that type alias type params do not leak and external type reads are captured."""
    import sys
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_nested_scope_free_reads,
    )

    if sys.version_info < (3, 12):
        pytest.skip("PEP 695 type parameter syntax requires Python 3.12+")

    code = (
        "def outer():\n"
        "    def inner():\n"
        "        type MyList[T] = list[T] | ExternalType\n"
        "        return MyList\n"
    )
    tree = ast.parse(code)
    inner_fn = tree.body[0].body[0]
    free = _extract_nested_scope_free_reads(inner_fn)
    assert "ExternalType" in free
    assert "T" not in free


def test_is_eagerly_consumed_genexp_slot_validation() -> None:
    """Verifies that generator expressions in non-iterated argument slots are not marked as eager."""
    from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=protected-access
        _get_scope_parent_map,
        _is_eagerly_consumed_genexp,
    )

    code = (
        "d = dict(gen=(x for x in items))\n"
        "m1 = min(items, default=(x for x in data))\n"
        "m2 = min((x for x in g1), (y for y in g2))\n"
        "s = sum(items, (x for x in data))\n"
        "j1 = ', '.join((x for x in words))\n"
        "j2 = str.join(',', (x for x in words))\n"
    )
    tree = ast.parse(code)
    parent_map = _get_scope_parent_map(tree)
    genexps = sorted(
        [n for n in ast.walk(tree) if isinstance(n, ast.GeneratorExp)],
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )
    assert len(genexps) == 7

    # d = dict(gen=(x for x in items)) -> non-iterated keyword argument
    assert _is_eagerly_consumed_genexp(genexps[0], parent_map) is False

    # m1 = min(items, default=(x for x in data)) -> default keyword argument
    assert _is_eagerly_consumed_genexp(genexps[1], parent_map) is False

    # m2 = min(g1, g2) -> multi-argument comparison
    assert _is_eagerly_consumed_genexp(genexps[2], parent_map) is False
    assert _is_eagerly_consumed_genexp(genexps[3], parent_map) is False

    # s = sum(items, genexp) -> start argument (2nd positional)
    assert _is_eagerly_consumed_genexp(genexps[4], parent_map) is False

    # j1 = ', '.join(...) -> 1st positional arg on str literal
    assert _is_eagerly_consumed_genexp(genexps[5], parent_map) is True

    # j2 = str.join(',', ...) -> 2nd positional arg on str class
    assert _is_eagerly_consumed_genexp(genexps[6], parent_map) is True


def test_scope_reads_outputs_after_line_reentering_compound_with_and_try() -> None:
    """Verifies that loop-carried re-entry reads inside with/try blocks detect downstream reads."""
    from pydoppelgangerhunt.fixer._df_hazards import (  # pylint: disable=protected-access
        _scope_reads_outputs_after_line,
    )

    code = (
        "for batch in batches:\n"
        "    with lock:\n"
        "        curr = acc\n"
        "        acc = curr + 1\n"
    )
    tree = ast.parse(code)
    reads_after = _scope_reads_outputs_after_line(
        tree, line=4, outputs={"acc"}, start_line=2
    )
    assert reads_after is True


def test_extract_unit_live_in_names_enclosing_compound_branch_isolation() -> None:
    """Verifies that live-in extraction preserves branch isolation when enclosing compound starts before unit."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def process(items, flag):\n"
        "    for item in items:\n"
        "        if flag:\n"
        "            x = 1\n"
        "        else:\n"
        "            y = 2\n"
        "        total = x + y\n"
        "        return total\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=3, u_end=7)
    assert "x" in live_ins
    assert "y" in live_ins
    assert "flag" in live_ins


def test_extract_unit_live_in_names_scope_node_at_unit_start() -> None:
    """Verifies that scope_node starting on u_start is not treated as a nested closure."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = "def compute(a, b):\n    return a + b\n"
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=1, u_end=2)
    assert "a" in live_ins
    assert "b" in live_ins


def test_extract_unit_live_in_names_outer_except_handler() -> None:
    """Verifies that an exception variable bound by an enclosing except header before u_start
    is recognized as a live-in when read inside the unit."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def handle(action):\n"
        "    try:\n"
        "        action()\n"
        "    except RuntimeError as err:\n"
        "        msg = str(err)\n"
        "        return msg\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=5, u_end=6)
    assert "err" in live_ins
    assert "msg" not in live_ins


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match syntax requires Python 3.10+")
def test_extract_unit_live_in_names_match_pattern_bindings() -> None:
    """Verifies that structural pattern bindings inside a unit are recorded as stores
    and not misclassified as live-in variables."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def parse_event(event, bonus):\n"
        "    match event:\n"
        "        case [head, *tail]:\n"
        "            res = head + len(tail) + bonus\n"
        "        case {'key': val, **rest}:\n"
        "            res = val + len(rest) + bonus\n"
        "    return res\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=2, u_end=7)
    assert "event" in live_ins
    assert "bonus" in live_ins
    assert "head" not in live_ins
    assert "tail" not in live_ins
    assert "val" not in live_ins
    assert "rest" not in live_ins


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type alias requires Python 3.12+")
def test_extract_single_node_local_names_type_alias_excludes_type_params() -> None:
    """Verifies that PEP 695 TypeAlias statements record the alias name in local names
    without leaking their lexical type_params into the enclosing scope."""
    from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=protected-access
        _extract_single_node_local_names,
    )

    code = (
        "def compute(items):\n"
        "    type MySeq[sum] = list[sum]\n"
        "    return sum(x for x in items)\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    local_names = _extract_single_node_local_names(fn_node)
    assert "MySeq" in local_names
    assert "items" in local_names
    assert "sum" not in local_names


def test_has_wildcard_star_import_detached_vs_attached_scope() -> None:
    """Verifies _has_wildcard_star_import across attached module trees and detached subtrees."""
    from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=protected-access
        _has_wildcard_star_import,
    )

    mod_clean = ast.parse("def f():\n    return 1\n")
    fn_clean = mod_clean.body[0]
    assert _has_wildcard_star_import(fn_clean, mod_clean) is False

    detached_star = ast.parse("def g():\n    from math import *\n    return 1\n").body[0]
    assert _has_wildcard_star_import(detached_star, mod_clean) is True


def test_collect_downstream_read_names_single_line_enclosing_loop() -> None:
    """Verifies that a single-line loop enclosing a sub-line unit is retained by
    _find_enclosing_loops and detects loop-carried reads prior to the unit."""
    _clear_downstream_reads_cache()
    code = (
        "def gen(items):\n"
        "    for x in items: use(total); total = x; yield total\n"
    )
    # On line 2, 'total = x; yield total' starts at col 32
    unit = {
        "file": "single_line_loop.py",
        "start": 2,
        "end": 2,
        "start_col": 32,
        "end_col": 54,
        "kind": "compound_block",
    }
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads


def test_extract_unit_live_in_names_annassign_annotations_and_bare_subscript() -> None:
    """Verifies that _UnitLiveInVisitor visits AnnAssign annotations and bare subscript/attribute
    targets while recording assigned Name targets as stores."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def process(arr, idx, obj):\n"
        "    x: MyType = 1\n"
        "    y: BareType\n"
        "    arr[idx]: SubType\n"
        "    obj.attr: AttrType\n"
        "    return x\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=2, u_end=5)
    assert "MyType" in live_ins
    assert "BareType" in live_ins
    assert "SubType" in live_ins
    assert "AttrType" in live_ins
    assert "arr" in live_ins
    assert "idx" in live_ins
    assert "obj" in live_ins
    assert "x" not in live_ins
    assert "y" not in live_ins


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type alias requires Python 3.12+")
def test_extract_unit_live_in_names_pep695_type_alias() -> None:
    """Verifies that _UnitLiveInVisitor isolates PEP 695 TypeAlias type_params from live-ins
    while recording the alias name as a store and external references in the value as live-ins."""
    from pydoppelgangerhunt.fixer._df_visitors import (  # pylint: disable=protected-access
        _extract_unit_live_in_names,
    )

    code = (
        "def make_types():\n"
        "    type MyMap[K, V: BoundType] = OuterContainer[K, V]\n"
        "    return MyMap\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=2, u_end=3)
    assert "BoundType" in live_ins
    assert "OuterContainer" in live_ins
    assert "MyMap" not in live_ins
    assert "K" not in live_ins
    assert "V" not in live_ins


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 syntax requires Python 3.12+")
def test_extract_single_node_local_names_includes_own_type_params() -> None:
    """Verifies that _extract_single_node_local_names includes scope_node's own PEP 695
    type_params so generic type parameters shadowing built-ins are detected."""
    from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=protected-access
        _extract_single_node_local_names,
    )

    code = (
        "def compute[sum](items):\n"
        "    return sum(x for x in items)\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    local_names = _extract_single_node_local_names(fn_node)
    assert "sum" in local_names
    assert "items" in local_names


def test_collect_downstream_read_names_lenient_closure_logs_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that collect_downstream_read_names with skip_pre_unit_closures=True
    logs at DEBUG rather than WARNING."""
    import logging

    _clear_downstream_reads_cache()
    code = (
        "def f():\n"
        "    x = 1\n"
        "    return x\n"
    )
    unit = {"file": "mod.py", "start": 2, "end": 2}
    with caplog.at_level(logging.DEBUG, logger="pydoppelgangerhunt"):
        reads = collect_downstream_read_names(
            code, unit, candidates={"x"}, skip_pre_unit_closures=True
        )
    assert reads == {"x"}
    warning_msgs = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert not any("Lenient closure mode active" in m for m in warning_msgs)
    debug_msgs = [
        r.message for r in caplog.records if r.levelno == logging.DEBUG
    ]
    assert any("Lenient closure mode active" in m for m in debug_msgs)


def test_nested_lambda_and_comprehension_closure_no_false_positive_free_reads() -> None:
    """Verifies that nested lambdas (lambda x: (lambda: x)) and lambdas/scopes inside
    comprehensions ([lambda: x for x in items]) do not falsely record outer lambda
    parameters or comprehension targets as free reads of the enclosing function scope."""
    code = (
        "def process(items):\n"
        "    fn = lambda x: (lambda: x)\n"
        "    callbacks = [lambda: x for x in items]\n"
        "    eager = sum((lambda: x + captured_var)() for x in items)\n"
        "    x = 42\n"
        "    captured_var = 99\n"
        "    return x, captured_var, fn, callbacks, eager\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]

    # Pre-unit closures before line 5 (where x and captured_var are assigned):
    # 'x' is bound by the outer lambda on line 2 and by the ListComp/GeneratorExp on lines 3-4,
    # whereas 'captured_var' is genuinely unbound in line 4's inner lambda.
    pre_closures = _collect_pre_unit_closures(fn_node, u_start=5)
    assert "x" not in pre_closures
    assert "captured_var" in pre_closures

    # Inspect all recorded closures in scope: none should record 'x' as a free read
    scope_closures = _collect_scope_closures(fn_node)
    all_reads = set().union(*(reads for _, reads in scope_closures))
    assert "x" not in all_reads
    assert "captured_var" in all_reads

    # _BaseScopeVisitor comprehension target filtering on nested lambda inside a nested function
    nested_code = (
        "def outer(items):\n"
        "    def inner():\n"
        "        return [lambda: x + ext_val for x in items]\n"
        "    return inner\n"
    )
    nested_tree = ast.parse(nested_code)
    inner_fn = nested_tree.body[0].body[0]  # type: ignore[attr-defined]
    free_reads = _extract_nested_scope_free_reads(inner_fn)
    assert "x" not in free_reads
    assert "ext_val" in free_reads
    assert "items" in free_reads


def test_find_enclosing_loops_whole_loop_with_col_metadata_not_self_enclosing() -> None:
    """Verifies that a whole-loop unit with start_col == loop.col_offset or end_col metadata
    is not treated as enclosing itself, while sub-line units starting after loop.col_offset are."""
    _clear_downstream_reads_cache()
    code = (
        "def run_loop(items):\n"
        "    acc = 0\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        yield acc\n"
        "    return 0\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]

    # Whole-loop unit spanning lines 3-5 with start_col=4 (matching the 'for' keyword)
    assert _find_enclosing_loops(fn_node, 3, 5, u_start_col=4) == []
    assert _find_enclosing_loops(fn_node, 3, 5, u_start_col=None) == []

    unit_whole_loop = {
        "file": "loop.py",
        "start": 3,
        "end": 5,
        "start_col": 4,
        "end_col": 17,
        "kind": "compound_block",
    }
    # 'acc' is dead after the loop (line 6 returns 0); must not be marked live via self-enclosure
    reads = collect_downstream_read_names(code, unit_whole_loop, candidates={"acc"})
    assert reads == set()


def test_extract_single_node_local_names_lambda_root_comprehension_and_nested_lambda() -> None:
    """Verifies that _extract_single_node_local_names on an ast.Lambda does not leak
    comprehension loop targets or inner lambda walrus stores when the lambda body is itself
    a comprehension or nested lambda."""
    lam_comp = ast.parse("lambda items: [sum for sum in items]").body[0].value  # type: ignore[attr-defined]
    comp_locals = _extract_single_node_local_names(lam_comp)
    assert "items" in comp_locals
    assert "sum" not in comp_locals

    lam_nested = ast.parse("lambda x: (lambda y: (sum := y))").body[0].value  # type: ignore[attr-defined]
    nested_locals = _extract_single_node_local_names(lam_nested)
    assert "x" in nested_locals
    assert "sum" not in nested_locals

    lam_walrus = ast.parse("lambda items: [(w := x) for x in items]").body[0].value  # type: ignore[attr-defined]
    walrus_locals = _extract_single_node_local_names(lam_walrus)
    assert "items" in walrus_locals
    assert "w" in walrus_locals
    assert "x" not in walrus_locals


def test_downstream_read_visitor_for_and_with_subscript_attribute_targets() -> None:
    """Verifies that _DownstreamReadVisitor visits Load sub-expressions inside For.target
    and With.items[*].optional_vars (e.g. for arr[idx] in items:, with cm() as obj.slot[k]:)."""
    _clear_downstream_reads_cache()
    code = (
        "def process(items, cm):\n"
        "    arr = [0, 0]\n"
        "    idx = 0\n"
        "    obj = make_holder()\n"
        "    k = 1\n"
        "    for arr[idx] in items:\n"
        "        pass\n"
        "    with cm() as obj.slot[k]:\n"
        "        pass\n"
    )
    unit = {"file": "for_with_targets.py", "start": 2, "end": 5}
    reads = collect_downstream_read_names(
        code, unit, candidates={"arr", "idx", "obj", "k"}
    )
    assert reads == {"arr", "idx", "obj", "k"}


def test_unit_live_in_visitor_multiline_assign_starting_before_u_start() -> None:
    """Verifies that _UnitLiveInVisitor visits Assign.value before Assign.targets even when
    the multi-line Assign node starts on u_start - 1."""
    code = (
        "def compute(val):\n"
        "    (\n"
        "        val,\n"
        "    ) = (\n"
        "        val + 1,\n"
        "    )\n"
        "    return val\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    # Assign starts on line 2; unit starts on line 3 (covering target on line 3 and value on line 5)
    live_ins = _extract_unit_live_in_names(fn_node, u_start=3, u_end=6)
    assert "val" in live_ins


def test_unit_live_in_visitor_subline_boolop_ifexp_namedexpr_evaluation_order() -> None:
    """Verifies that _UnitLiveInVisitor preserves conditional depth and value-before-target
    order when a sub-line unit starts inside an enclosing BoolOp, IfExp, or NamedExpr."""
    code_boolop = (
        "def f(cond, y):\n"
        "    res = cond and (x := y) or x\n"
    )
    tree_b = ast.parse(code_boolop)
    # Unit starts at '(x := y) or x' (col 19..32), after 'cond and '
    live_b = _extract_unit_live_in_names(
        tree_b.body[0], u_start=2, u_end=2, u_start_col=19, u_end_col=32
    )
    assert "x" in live_b
    assert "y" in live_b

    code_walrus = (
        "def g(x):\n"
        "    if (x := x + 1) > 0:\n"
        "        return x\n"
    )
    tree_w = ast.parse(code_walrus)
    # Sub-line unit starts at 'x + 1' inside '(x := x + 1)' (col 13..18)
    live_w = _extract_unit_live_in_names(
        tree_w.body[0], u_start=2, u_end=2, u_start_col=13, u_end_col=18
    )
    assert "x" in live_w


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match/case requires Python 3.10+")
def test_pattern_binding_visits_value_and_key_loads_before_store() -> None:
    """Verifies that case x.CONST as x: and case {x.k: 1, **x}: record the attribute load of x
    before binding/killing x in both _UnitLiveInVisitor and _DownstreamReadVisitor."""
    _clear_downstream_reads_cache()
    code = (
        "def matcher(subj, x):\n"
        "    y = 1\n"
        "    match subj:\n"
        "        case x.CONST as x:\n"
        "            return x\n"
        "        case {x.k: 1, **x}:\n"
        "            return y\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=3, u_end=7)
    assert "x" in live_ins

    unit = {"file": "match_order.py", "start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"x", "y"})
    assert reads == {"x", "y"}


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12+")
def test_pep695_generic_class_type_params_excluded_from_method_closures() -> None:
    """Verifies that PEP 695 class type_params (class C[T]:) are excluded from closures
    inside C's bases, keywords, and body, while decorators on C still capture free reads."""
    code = (
        "def make_cls(x, dec):\n"
        "    @dec(lambda: U)\n"
        "    class Box[T]:\n"
        "        def get(self, val: T) -> T:\n"
        "            return x\n"
        "    type MyAlias[K] = tuple[K, (lambda: (K, V))()]\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    closures = _collect_scope_closures(fn_node)
    captured = set().union(*(names for _, names in closures))
    assert "x" in captured
    assert "U" in captured
    assert "V" in captured
    assert "T" not in captured
    assert "K" not in captured


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match/case requires Python 3.10+")
def test_collect_scope_closures_match_case_pattern_bindings() -> None:
    """Verifies that _collect_scope_closures excludes variables bound by a match case
    pattern from closures in that case's guard and body, while preserving outer reads."""
    code = (
        "def handle(val):\n"
        "    match (lambda: subj_ext)():\n"
        "        case [x, *rest] if (lambda: x + len(rest) + guard_ext)():\n"
        "            cb = lambda: (x, rest, body_ext)\n"
        "        case {'k': v, **kw}:\n"
        "            cb2 = lambda: (v, kw)\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    closures = _collect_scope_closures(fn_node)
    captured = set().union(*(names for _, names in closures))
    assert "subj_ext" in captured
    assert "guard_ext" in captured
    assert "body_ext" in captured
    assert "x" not in captured
    assert "rest" not in captured
    assert "v" not in captured
    assert "kw" not in captured


def test_scope_hierarchy_visitor_header_expressions_async_context() -> None:
    """Verifies that _ScopeHierarchyVisitor records enclosing async context for header
    expressions (decorators, defaults, bases, keywords) before pushing inner scope."""
    from pydoppelgangerhunt.parser import _ScopeHierarchyVisitor  # pylint: disable=protected-access

    code = (
        "async def outer(dec, Base):\n"
        "    @dec([x for x in range(3)])\n"
        "    def sync_inner(a=[y for y in range(3)], *, b=[z for z in range(3)]):\n"
        "        return [w for w in range(3)]\n"
        "    class Inner([u for u in range(3)], meta=[v for v in range(3)]):\n"
        "        attr = [q for q in range(3)]\n"
    )
    tree = ast.parse(code)
    func_is_async: Dict[int, bool] = {}
    _ScopeHierarchyVisitor(func_is_async=func_is_async).visit(tree)
    comps = sorted(
        [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)],
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )
    assert len(comps) == 7
    # Header comprehensions on sync_inner (decorator, default, kw_default) and Inner (base, keyword)
    # evaluate in outer's async context (True)
    for header_comp in comps[:3] + comps[4:6]:
        assert func_is_async.get(id(header_comp)) is True
    # Body comprehensions inside sync_inner and Inner evaluate in sync context (False)
    assert func_is_async.get(id(comps[3])) is False
    assert func_is_async.get(id(comps[6])) is False


def test_unit_live_in_visitor_intra_unit_delete() -> None:
    """Verifies that _UnitLiveInVisitor records del targets as live-ins when not yet stored
    and discards deleted identifiers from self.stored so subsequent reads are captured."""
    code = (
        "def process(pre_existing, arr):\n"
        "    del pre_existing\n"
        "    local_var = 10\n"
        "    del (local_var, arr[local_var])\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    live_ins = _extract_unit_live_in_names(fn_node, u_start=2, u_end=4)
    assert "pre_existing" in live_ins
    assert "local_var" in live_ins
    assert "arr" in live_ins



