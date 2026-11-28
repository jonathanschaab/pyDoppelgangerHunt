"""Unit tests for clone output pairing, generator subroutine output resolution, and CloneOutputResolution contracts."""

from __future__ import annotations
import ast
from pathlib import Path
from typing import (
    Any,
    Dict,
)
import pytest
from pydoppelgangerhunt import SCOPE_SCHEMA_VERSION
from pydoppelgangerhunt.fixer._df_pairing import (  # pylint: disable=protected-access
    _extract_effective_unit_outputs,
    _pair_clone_outputs,
)
from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access
    GeneratorCloneSideData,
    has_async_generator_delegation_hazard,
    is_async_generator_with_return_value,
    resolve_clone_generator_subroutine_outputs,
    resolve_clone_pair_outputs,
    resolve_generator_subroutine_outputs,
)


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
    s1 = {
        "definite_stores": ["a"],
        "globals": ["b"],
        "nonlocals": [],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    s2 = {
        "definite_stores": ["x"],
        "globals": [],
        "nonlocals": ["y"],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
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

def test_extract_effective_unit_outputs_set_handling() -> None:
    """Verifies that multi-item set outputs are rejected to prevent unpositional pairing,
    while single-item sets are supported."""
    assert _extract_effective_unit_outputs({"outputs": {"z", "a", "m"}}, {}) == []
    assert _extract_effective_unit_outputs({"outputs": {"single"}}, {}) == ["single"]

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
    s1 = {
        "definite_stores": ["x"],
        "inputs": [],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    s2 = {
        "definite_stores": ["a", "b"],
        "inputs": [],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    res_rejected = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res_rejected is None

    # When both sides definitely assign all precomputed outputs, pairing succeeds
    s1_valid = {
        "definite_stores": ["x", "unassigned"],
        "inputs": [],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
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

def test_generator_subroutine_outputs_mocked_scope_fails_closed() -> None:
    """Verifies that mocked/hand-built scopes lacking definite_stores and inputs fail closed."""
    u1 = {"outputs": ["val"], "source_text": "def f():\n    val = 1\n"}
    u2 = {"outputs": ["val"], "source_text": "def g():\n    val = 2\n"}
    # Scope dict missing both definite_stores and inputs
    s1: Dict[str, Any] = {
        "globals": [],
        "nonlocals": [],
    }
    s2: Dict[str, Any] = {
        "globals": [],
        "nonlocals": [],
    }
    res = resolve_clone_generator_subroutine_outputs(u1, u2, s1, s2)
    assert res is None

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

def test_resolve_clone_pair_outputs_clean_pair() -> None:
    """Verifies that resolve_clone_pair_outputs successfully pairs matching outputs."""
    u1 = {"kind": "function", "outputs": ["a", "b"]}
    u2 = {"kind": "function", "outputs": ["x", "y"]}
    s1: Dict[str, Any] = {
        "inputs": [],
        "definite_stores": ["a", "b"],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    s2: Dict[str, Any] = {
        "inputs": [],
        "definite_stores": ["x", "y"],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }

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
    scope1 = {
        "inputs": ["total"],
        "outputs": ["total"],
        "definite_stores": ["total"],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    scope2 = {
        "inputs": ["total"],
        "outputs": ["total"],
        "definite_stores": ["total"],
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    # Sliced lines with len(source_lines) < end fail closed because full file text is unavailable
    res = resolve_clone_generator_subroutine_outputs(u1, u2, scope1, scope2)
    assert res is None

def test_pair_clone_outputs_allow_permuted_outputs() -> None:
    """Verifies that _pair_clone_outputs rejects permuted outputs when disabled
    and pairs positionally when enabled."""
    from pydoppelgangerhunt.fixer._df_pairing import _pair_clone_outputs

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

def test_pair_clone_outputs_role_swapped_rejection_and_positional_pairing() -> None:
    """Verifies that role-swapped clones ([x, y] vs [y, x]) are rejected under
    allow_permuted_outputs=False, but pair positionally when allow_permuted_outputs=True,
    while alpha-renamed outputs ([x, y] vs [a, b]) pair positionally by default."""
    from pydoppelgangerhunt.fixer._df_pairing import _pair_clone_outputs

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
