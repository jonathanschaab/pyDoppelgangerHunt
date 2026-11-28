"""Tests for context-manager suppression rules, allowlists, suspicious stems, and exit-protocol patching."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    generate_refactoring_patch,
    inspect_single_unit_scope,
)
from tests.fixer._prov_test_helpers import (
    _PROVEN_EXIT_CLASS,
    _WITH_MGR_RUN,
    _WITH_NULLCONTEXT_RUN,
    _late_bound_capture_safety_by_mode,
)


def test_shadowed_eval_parameter_preserves_proven_with_safety() -> None:
    """Verifies unrelated local eval parameter no longer withdraws non-suppression proofs on Mgr."""
    src = (
        "def configure(eval: bool = False):\n"
        "    flag = eval\n"
        "    return flag\n"
        "\n"
        "class SafeMgr:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n"
        "\n"
        "def worker(step):\n"
        "    mgr = SafeMgr()\n"
        "    with mgr:\n"
        "        out = step()\n"
        "    return out\n"
    )
    clear_dataflow_cache()
    tree = ast.parse(src)
    lines = src.splitlines()
    start_line = next(i + 1 for i, ln in enumerate(lines) if "with mgr:" in ln)
    end_line = next(i + 1 for i, ln in enumerate(lines) if "out = step()" in ln)
    for mode in ("strict", "pedantic"):
        sc = inspect_single_unit_scope(
            {
                "start": start_line,
                "end": end_line,
                "kind": "compound_block",
                "source_text": src,
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert sc["is_control_flow_safe"] is True, mode

def test_suppress_provenance_set_algebra_and_with_classification_modes() -> None:
    """Verifies _SuppressProvenance structured operations, heuristic vs pedantic unknown-with classification, and extra_non_suppressing_ctx_managers."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

    p1 = _SuppressProvenance(
        suppressors=frozenset({"s1", "s2"}),
        non_suppressors=frozenset({"ns1"}),
        stdlib_modules=frozenset({("ctx1", "contextlib")}),
        shadowed_builtins=frozenset({"open"}),
        has_star_import=False,
    )
    p2 = _SuppressProvenance(
        suppressors=frozenset({"s2", "s3"}),
        non_suppressors=frozenset({"ns1", "ns2"}),
        stdlib_modules=frozenset({("ctx1", "contextlib"), ("ctx2", "io")}),
        shadowed_builtins=frozenset({"builtins"}),
        has_star_import=True,
    )
    assert not isinstance(p1, AbstractSet)
    p_merged = p1.merge_branch_conservative(p2)
    assert isinstance(p_merged, _SuppressProvenance)
    assert p_merged.suppress_names == frozenset({"s1", "s2", "s3"})
    assert p_merged.non_suppress_names == frozenset({"ns1"})
    assert p_merged.stdlib_modules == frozenset({("ctx1", "contextlib")})
    assert p_merged.has_star_import is True

    p_cleared = p1.clear_name("s1")
    assert isinstance(p_cleared, _SuppressProvenance)
    assert p_cleared.suppress_names == frozenset({"s2"})
    assert p_cleared.non_suppress_names == frozenset({"ns1"})

    prov_src = (
        "from contextlib import suppress\n"
        "def example(flag, items):\n"
        "    if flag:\n"
        "        cm = suppress\n"
        "    else:\n"
        "        cm = suppress\n"
        "    with cm(TypeError):\n"
        "        n = len(items)\n"
        "    with open('path.txt'):\n"
        "        m = compute_value()\n"
        "    def open(v):\n"
        "        return v\n"
        "    return n, m\n"
    )
    prov_tree = ast.parse(prov_src)
    prov_fn = prov_tree.body[1]
    assert isinstance(prov_fn, ast.FunctionDef)
    prov = _seed_scope_suppress_aliases(prov_fn, prov_tree, u_start=7)
    assert isinstance(prov, _SuppressProvenance)
    assert "cm" in prov.suppress_names
    assert "open" in prov.shadowed_builtins
    bound_prov, _ = _extract_block_definite_bindings_and_term(
        prov_fn.body[:-2],
        None,
        suppress_aliases=prov,
        strict_with_suppression=True,
    )
    assert "n" not in bound_prov
    assert "m" not in bound_prov

    custom_safe_stmts = ast.parse(
        "with CustomTransaction():\n" "    committed = compute_value()\n"
    ).body
    # Default heuristic mode treats non-suppress-named managers as non-suppressing
    def_heur, cond_heur = _analyze_block_assignment(
        custom_safe_stmts, strict_with_suppression=False
    )
    assert "committed" in def_heur
    assert "committed" not in cond_heur

    # Pedantic mode treats unknown managers as potentially suppressing unless whitelisted
    def_ped_without, cond_ped_without = _analyze_block_assignment(
        custom_safe_stmts, strict_with_suppression=True
    )
    def_ped_with, cond_ped_with = _analyze_block_assignment(
        custom_safe_stmts,
        extra_non_suppressing_ctx_managers=frozenset({"CustomTransaction"}),
        strict_with_suppression=True,
    )
    assert "committed" not in def_ped_without
    assert "committed" in cond_ped_without
    assert "committed" in def_ped_with
    assert "committed" not in cond_ped_with

def test_aliased_suppress_provenance_cannot_be_overridden_by_allowlist() -> None:
    """Verifies benign-named local aliases of suppress, ExitStack, and custom suppressing classes remain suppressing in strict mode."""
    alias_src = (
        "from contextlib import suppress, ExitStack\n"
        "class CustomSwallower:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return True\n"
        "def outer(f):\n"
        "    benign_sup = suppress\n"
        "    benign_stack = ExitStack()\n"
        "    benign_custom = CustomSwallower()\n"
        "    with benign_sup(ValueError):\n"
        "        a = f()\n"
        "    with benign_stack:\n"
        "        b = f()\n"
        "    with benign_custom:\n"
        "        c = f()\n"
        "    return 0\n"
    )
    alias_tree = ast.parse(alias_src)
    sc_alias_strict = inspect_single_unit_scope(
        {
            "start": 11,
            "end": 16,
            "kind": "compound_block",
            "source_text": alias_src,
            "closure_strictness": "strict",
            "non_suppressing_ctx_managers": [
                "benign_sup",
                "benign_stack",
                "benign_custom",
            ],
        },
        tree=alias_tree,
    )
    assert set(sc_alias_strict["definite_stores"]) == set()
    assert {"a", "b", "c"} <= set(sc_alias_strict["conditional_outputs"])

def test_stdlib_allowlist_and_unqualified_catch_warnings() -> None:
    """Verifies stdlib non-suppressing allowlist and catch_warnings without visible import in strict mode."""
    stdlib_cm_src = (
        "import warnings\n"
        "import threading\n"
        "import decimal\n"
        "import unittest.mock\n"
        "from warnings import catch_warnings\n"
        "def run_stdlib(compute):\n"
        "    with warnings.catch_warnings(), catch_warnings(), threading.Lock(), decimal.localcontext(), unittest.mock.patch('os.sep'):\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    stdlib_tree = ast.parse(stdlib_cm_src)
    for mode in ("strict", "pedantic"):
        sc_std = inspect_single_unit_scope(
            {
                "start": 9,
                "end": 9,
                "kind": "basic_block",
                "source_text": stdlib_cm_src,
                "source_code": "    cb = lambda: val\n",
                "closure_strictness": mode,
            },
            tree=stdlib_tree,
        )
        assert "severed_closure_cell" not in sc_std["control_flow_hazards"], mode

    unimported_cw_src = (
        "def run_unimported(warnings_mod, compute):\n"
        "    with warnings_mod.catch_warnings(), catch_warnings():\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    unimported_tree = ast.parse(unimported_cw_src)
    sc_unimp_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": unimported_cw_src,
            "source_code": "    cb = lambda: val\n",
            "closure_strictness": "strict",
        },
        tree=unimported_tree,
    )
    assert "severed_closure_cell" not in sc_unimp_strict["control_flow_hazards"]

@pytest.mark.parametrize(
    "import_stmt",
    [
        "from evil import open",
        "import evil as open",
    ],
)
def test_imported_open_from_non_stdlib_module_remains_shadowed(
    import_stmt: str,
) -> None:
    """Verifies importing open from a non-stdlib source never un-shadows builtin open in unit or pre-span code."""
    in_unit_src = (
        "def run_in_unit(f, compute):\n"
        f"    {import_stmt}\n"
        "    with open(f):\n"
        "        x = compute()\n"
        "    return 0\n"
    )
    in_unit_tree = ast.parse(in_unit_src)
    sc_in_unit = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": in_unit_src,
            "closure_strictness": "strict",
        },
        tree=in_unit_tree,
    )
    assert "x" not in sc_in_unit["definite_stores"]
    assert "x" in sc_in_unit["conditional_outputs"]

    pre_span_src = (
        "def run_pre_span(f, compute):\n"
        f"    {import_stmt}\n"
        "    with open(f):\n"
        "        x = compute()\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    pre_span_tree = ast.parse(pre_span_src)
    sc_pre_span = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": pre_span_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "strict",
        },
        tree=pre_span_tree,
    )
    assert "severed_closure_cell" in sc_pre_span["control_flow_hazards"]
    assert sc_pre_span["is_control_flow_safe"] is False

@pytest.mark.parametrize(
    ("cls_src", "expect_proven_safe", "expect_may_suppress"),
    [
        (
            "class CleanExit:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        helper = lambda: True\n"
            "        return False\n",
            True,
            False,
        ),
        (
            "class MetaExit(metaclass=CustomMeta):\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n",
            False,
            False,
        ),
        (
            "class CustomNewExit:\n"
            "    def __new__(cls):\n        return other()\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n",
            False,
            False,
        ),
        (
            "class CustomGetattrExit:\n"
            "    def __getattribute__(self, name):\n        return object.__getattribute__(self, name)\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n",
            False,
            False,
        ),
        (
            "class ConditionalExit:\n"
            "    if flag:\n"
            "        def __exit__(self, exc_type, exc, tb):\n            return True\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n",
            False,
            True,
        ),
        (
            "class DuplicateExit:\n"
            "    def __exit__(self, exc_type, exc, tb):\n        return True\n"
            "    def __exit__(self, exc_type, exc, tb):\n        return False\n",
            False,
            True,
        ),
        (
            "class GeneratorExit:\n"
            "    def __exit__(self, exc_type, exc, tb):\n"
            "        yield False\n"
            "        return False\n",
            False,
            True,
        ),
        (
            "class AsyncDefSyncExit:\n"
            "    async def __exit__(self, exc_type, exc, tb):\n"
            "        return False\n",
            False,
            True,
        ),
        (
            "class AssignedLambdaExit:\n" "    __exit__ = lambda self, *a: True\n",
            False,
            True,
        ),
    ],
)
def test_class_def_exit_adversarial_bypasses_and_shallow_scope_walk(
    cls_src: str,
    expect_proven_safe: bool,
    expect_may_suppress: bool,
) -> None:
    """Verifies class __exit__ proof rejects metaclass/descriptor/generator/async/conditional bypasses."""
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance
    from pydoppelgangerhunt.fixer._df_with_rules import (
        _class_def_is_proven_non_suppressing_in_exit,
        _class_def_may_suppress_in_exit,
    )

    cls_node = ast.parse(cls_src).body[0]
    assert isinstance(cls_node, ast.ClassDef)
    prov = _SuppressProvenance()
    assert (
        _class_def_is_proven_non_suppressing_in_exit(cls_node, prov)
        is expect_proven_safe
    )
    assert _class_def_may_suppress_in_exit(cls_node) is expect_may_suppress

def test_func_def_generator_contextmanager_try_and_with_around_yield() -> None:
    """Verifies _func_def_may_return_suppressor detects yield wrapped in try or with blocks and local imports."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import _func_def_may_return_suppressor
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

    prov = _SuppressProvenance(
        suppressors=frozenset({"suppress"}),
        non_suppressors=frozenset({"nullcontext"}),
    )

    gen_try_src = (
        "def cm_try():\n"
        "    try:\n"
        "        yield\n"
        "    except Exception:\n"
        "        pass\n"
    )
    fn_try = ast.parse(gen_try_src).body[0]
    assert isinstance(fn_try, ast.FunctionDef)
    assert _func_def_may_return_suppressor(fn_try, prov) is True

    gen_with_local_import_src = (
        "def cm_local_suppress():\n"
        "    from contextlib import suppress as local_sup\n"
        "    with local_sup(ValueError):\n"
        "        yield\n"
    )
    fn_local = ast.parse(gen_with_local_import_src).body[0]
    assert isinstance(fn_local, ast.FunctionDef)
    assert _func_def_may_return_suppressor(fn_local, prov) is True

    gen_clean_src = "def cm_clean():\n" "    setup = 1\n" "    yield setup\n"
    fn_clean = ast.parse(gen_clean_src).body[0]
    assert isinstance(fn_clean, ast.FunctionDef)
    assert _func_def_may_return_suppressor(fn_clean, prov) is False

@pytest.mark.parametrize(
    "cls_header,expected_suppressor",
    [
        ("class LocalMgr(contextlib.suppress):\n    pass\n", True),
        ("class LocalMgr(suppress):\n    pass\n", True),
        (
            "class BaseSup:\n    def __exit__(self, *a):\n        return True\n"
            "class LocalMgr(BaseSup):\n    pass\n",
            True,
        ),
        ("class LocalMgr(contextlib.nullcontext):\n    pass\n", False),
    ],
)
def test_local_subclass_of_suppressor_tracked_in_with_provenance(
    cls_header: str,
    expected_suppressor: bool,
) -> None:
    """Verifies local classes inheriting from suppressing context managers are tracked as suppressors."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    indented_cls = "".join(
        f"    {line}" for line in cls_header.splitlines(keepends=True)
    )
    src = (
        "import contextlib\n"
        "from contextlib import suppress\n"
        "def run(d):\n"
        f"{indented_cls}"
        "    with LocalMgr():\n"
        "        val = d['k']\n"
        "    return val\n"
    )
    mod_tree = ast.parse(src)
    fn_node = mod_tree.body[2]
    assert isinstance(fn_node, ast.FunctionDef)
    prov = _seed_scope_suppress_aliases(
        fn_node, mod_tree, u_start=len(src.splitlines())
    )
    assert ("LocalMgr" in prov.suppressors) is expected_suppressor

    bound, _ = _extract_block_definite_bindings_and_term(
        fn_node.body,
        None,
        frozenset({"d"}),
        strict_with_suppression=False,
        suppress_aliases=_seed_scope_suppress_aliases(fn_node, mod_tree),
    )
    assert ("val" not in bound) is expected_suppressor

def test_local_class_dynamic_namespace_mutation_not_proven_non_suppressing() -> None:
    """Verifies local classes with dynamic namespace mutation (locals().update, exec, setattr) are not proven non-suppressing."""
    from pydoppelgangerhunt.fixer._df_with_rules import (
        _class_def_is_proven_non_suppressing_in_exit,
        _class_def_may_suppress_in_exit,
    )

    tricky_cls_src = (
        "class TrickyCM:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, *args):\n"
        "        return False\n"
        "    locals().update({'__exit__': lambda self, *a: True})\n"
    )
    tricky_cls = ast.parse(tricky_cls_src).body[0]
    assert isinstance(tricky_cls, ast.ClassDef)
    assert _class_def_may_suppress_in_exit(tricky_cls) is True
    assert _class_def_is_proven_non_suppressing_in_exit(tricky_cls) is False

    exec_cls_src = (
        "class ExecCM:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, *args):\n"
        "        return False\n"
        "    exec('def __exit__(self, *a): return True')\n"
    )
    exec_cls = ast.parse(exec_cls_src).body[0]
    assert isinstance(exec_cls, ast.ClassDef)
    assert _class_def_may_suppress_in_exit(exec_cls) is True
    assert _class_def_is_proven_non_suppressing_in_exit(exec_cls) is False

    full_src = (
        "def run_tricky(flag):\n"
        "    class TrickyCM:\n"
        "        def __enter__(self):\n"
        "            return self\n"
        "        def __exit__(self, *args):\n"
        "            return False\n"
        "        locals().update({'__exit__': lambda self, *a: True})\n"
        "    with TrickyCM():\n"
        "        if flag:\n"
        "            raise ValueError('boom')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n"
    )
    for mode in ("strict", "pedantic"):
        sc = inspect_single_unit_scope(
            {
                "start": 12,
                "end": 13,
                "kind": "compound_block",
                "source_text": full_src,
                "source_code": "    cb = lambda: x\n    out = cb() + 1\n",
                "closure_strictness": mode,
            }
        )
        assert "severed_closure_cell" in sc["control_flow_hazards"]
        assert "conditional_live_in" in sc.get("closure_hazard_reasons", ())

@pytest.mark.parametrize(
    "shadow_src,expect_hazard",
    [
        (
            "def f(SafeTx, flag):\n"
            "    with SafeTx():\n"
            "        if flag:\n"
            "            raise ValueError('boom')\n"
            "        x = 2\n"
            "    cb = lambda: x\n"
            "    return cb()\n",
            True,
        ),
        (
            "def f(custom_cm, flag):\n"
            "    SafeTx = custom_cm\n"
            "    with SafeTx():\n"
            "        if flag:\n"
            "            raise ValueError('boom')\n"
            "        x = 2\n"
            "    cb = lambda: x\n"
            "    return cb()\n",
            True,
        ),
        (
            "from unknown_pkg import *\n"
            "def f(flag):\n"
            "    with SafeTx():\n"
            "        if flag:\n"
            "            raise ValueError('boom')\n"
            "        x = 2\n"
            "    cb = lambda: x\n"
            "    return cb()\n",
            True,
        ),
        (
            "import contextlib\n"
            "def f(flag):\n"
            "    sup = contextlib.suppress\n"
            "    with sup.atomic():\n"
            "        if flag:\n"
            "            raise ValueError('boom')\n"
            "        x = 2\n"
            "    cb = lambda: x\n"
            "    return cb()\n",
            True,
        ),
        (
            "def f(flag):\n"
            "    with SafeTx():\n"
            "        if flag:\n"
            "            raise ValueError('boom')\n"
            "        x = 2\n"
            "    cb = lambda: x\n"
            "    return cb()\n",
            False,
        ),
    ],
)
def test_extra_non_suppressing_allowlist_shadowing_and_suppressor_wildcard_receiver(
    shadow_src: str,
    expect_hazard: bool,
) -> None:
    """Verifies non_suppressing_ctx_managers rejects locally shadowed names, star imports, and suppressor wildcard receivers."""
    lines = shadow_src.splitlines(keepends=True)
    cb_line = next(i for i, ln in enumerate(lines, 1) if "cb = lambda: x" in ln)
    sc = inspect_single_unit_scope(
        {
            "start": cb_line,
            "end": cb_line,
            "kind": "basic_block",
            "source_text": shadow_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": ["SafeTx", "*.atomic", ".atomic"],
        },
        tree=ast.parse(shadow_src),
    )
    assert ("severed_closure_cell" in sc["control_flow_hazards"]) is expect_hazard

def test_for_loop_retry_yielded_context_manager_and_relative_vs_wildcard_allowlist() -> (
    None
):
    """Verifies for-loop yielded retry/attempt context managers and distinct .Name vs *.attr allowlist semantics."""
    retry_src = (
        "def run_retry(Retrying, flag):\n"
        "    for step in Retrying():\n"
        "        with step:\n"
        "            if flag:\n"
        "                raise ValueError('transient')\n"
        "            x = 10\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    sc_retry = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 7,
            "kind": "basic_block",
            "source_text": retry_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(retry_src),
    )
    assert "severed_closure_cell" in sc_retry["control_flow_hazards"]
    assert "conditional_live_in" in sc_retry["closure_hazard_reasons"]

    attr_src = (
        "def run_attr(client, flag):\n"
        "    with client.atomic():\n"
        "        if flag:\n"
        "            raise ValueError('err')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    sc_dot_only = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "basic_block",
            "source_text": attr_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": [".atomic"],
        },
        tree=ast.parse(attr_src),
    )
    assert "severed_closure_cell" in sc_dot_only["control_flow_hazards"]

    sc_star_dot = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "basic_block",
            "source_text": attr_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": ["*.atomic"],
        },
        tree=ast.parse(attr_src),
    )
    assert "severed_closure_cell" not in sc_star_dot["control_flow_hazards"]

    rel_import_src = (
        "from . import atomic\n"
        "def run_rel(flag):\n"
        "    with atomic():\n"
        "        if flag:\n"
        "            raise ValueError('err')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    sc_rel = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 7,
            "kind": "basic_block",
            "source_text": rel_import_src,
            "source_code": "    cb = lambda: x\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": [".atomic"],
        },
        tree=ast.parse(rel_import_src),
    )
    assert "severed_closure_cell" not in sc_rel["control_flow_hazards"]

def test_multi_item_with_heuristic_telemetry_and_overflow_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies heuristic context-manager telemetry is recorded across multi-item with headers and overflow summary distinguishes retained vs omitted counts."""
    import logging
    from pydoppelgangerhunt.config import (
        _MAX_HEURISTIC_CTX_MANAGERS,
        clear_closure_warning_cache,
        collect_heuristic_ctx_managers,
        log_heuristic_ctx_manager_summary,
        peek_heuristic_non_suppressing_ctx_managers,
        record_heuristic_non_suppressing_ctx_manager,
    )

    clear_closure_warning_cache()
    multi_with_src = (
        "import contextlib\n"
        "def run_multi(compute):\n"
        "    with UnknownBefore(), contextlib.suppress(ValueError), UnknownAfter():\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    multi_tree = ast.parse(multi_with_src)
    with collect_heuristic_ctx_managers() as collected:
        sc = inspect_single_unit_scope(
            {
                "start": 5,
                "end": 5,
                "kind": "basic_block",
                "source_text": multi_with_src,
                "source_code": "    cb = lambda: val\n",
                "closure_strictness": "strict",
            },
            tree=multi_tree,
        )
    assert "severed_closure_cell" in sc["control_flow_hazards"]
    assert {"UnknownBefore", "UnknownAfter"} <= collected
    assert {"UnknownBefore", "UnknownAfter"} <= set(
        peek_heuristic_non_suppressing_ctx_managers()
    )

    # Verify overflow reporting in log_heuristic_ctx_manager_summary distinguishes retained names vs overflow occurrences
    clear_closure_warning_cache()
    for idx in range(_MAX_HEURISTIC_CTX_MANAGERS + 5):
        record_heuristic_non_suppressing_ctx_manager(f"CustomMgr_{idx:04d}")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        retained = log_heuristic_ctx_manager_summary()
    assert len(retained) == _MAX_HEURISTIC_CTX_MANAGERS
    assert any(
        rec.levelno == logging.WARNING
        and f"Summary: {_MAX_HEURISTIC_CTX_MANAGERS} unproven context manager(s)"
        in rec.message
        and "5 additional occurrence(s) omitted beyond cap" in rec.message
        for rec in caplog.records
    )

def test_pathlib_path_and_uncalled_factories_not_classified_as_non_suppressing(
    tmp_path: Path,
) -> None:
    """Verifies pathlib.Path and bare uncalled context-manager factories are never treated as non-suppressing."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import _STDLIB_NON_SUPPRESSING_MODULES
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
    from pydoppelgangerhunt.fixer._df_provenance_walk import _classify_with_suppression

    assert "pathlib" not in _STDLIB_NON_SUPPRESSING_MODULES

    cases = [
        (
            "pathlib_call",
            "from pathlib import Path\n",
            "def run_a(x):\n"
            "    with Path('example.txt'):\n"
            "        val = x + 1\n"
            "    snap = lambda: val\n"
            "    return snap() + x\n",
        ),
        (
            "bare_open",
            "",
            "def run_a(x):\n"
            "    with open:\n"
            "        val = x + 1\n"
            "    snap = lambda: val\n"
            "    return snap() + x\n",
        ),
        (
            "bare_nullcontext_attr",
            "import contextlib\n",
            "def run_a(x):\n"
            "    with contextlib.nullcontext:\n"
            "        val = x + 1\n"
            "    snap = lambda: val\n"
            "    return snap() + x\n",
        ),
        (
            "bare_nullcontext_name",
            "from contextlib import nullcontext\n",
            "def run_a(x):\n"
            "    with nullcontext:\n"
            "        val = x + 1\n"
            "    snap = lambda: val\n"
            "    return snap() + x\n",
        ),
    ]
    for label, hdr_src, fn_src in cases:
        mod_src = hdr_src + fn_src + "\n" + fn_src.replace("def run_a(", "def run_b(")
        mod_file = tmp_path / f"{label}.py"
        mod_file.write_text(mod_src, encoding="utf-8")
        tree = ast.parse(mod_src)
        run_a = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_a"
        )
        run_b = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_b"
        )
        with_a = next(n for n in run_a.body if isinstance(n, ast.With))
        prov_a = _seed_scope_suppress_aliases(
            run_a, mod_tree=tree, u_start=with_a.lineno
        )
        assert _classify_with_suppression(with_a, prov_a) != "non_suppressing", label

        # Extracting the with-block must see `val` as conditionally assigned (not definite),
        # and extracting the post-with closure block must reject `val` as an unbound cell.
        u1_with = {
            "name": "run_a:with",
            "file": str(mod_file),
            "start": with_a.lineno,
            "end": with_a.end_lineno,
            "kind": "compound_block",
            "source_text": mod_src,
        }
        sc_with = inspect_single_unit_scope(u1_with, tree=tree)
        assert "val" not in sc_with["definite_stores"], label
        assert "val" in sc_with["conditional_outputs"], label

        u1_post = {
            "name": "run_a:post",
            "file": str(mod_file),
            "start": run_a.body[1].lineno,
            "end": run_a.body[1].end_lineno,
            "kind": "basic_block",
            "source_text": mod_src,
        }
        u2_post = {
            "name": "run_b:post",
            "file": str(mod_file),
            "start": run_b.body[1].lineno,
            "end": run_b.body[1].end_lineno,
            "kind": "basic_block",
            "source_text": mod_src,
        }
        sc_post = inspect_single_unit_scope(u1_post, tree=tree)
        assert sc_post["is_control_flow_safe"] is False, label
        assert "severed_closure_cell" in sc_post["control_flow_hazards"], label
        patch = generate_refactoring_patch(
            [(1.0, u1_post, u2_post)],
            repo_root=str(tmp_path),
            replace_clones=True,
            closure_strictness="strict",
        )
        assert patch == "", label

    # Conversely, bound instance provenance (cm = contextlib.nullcontext(); with cm:) IS non_suppressing
    inst_src = (
        "import contextlib\n"
        "def valid_inst():\n"
        "    cm = contextlib.nullcontext()\n"
        "    with cm:\n"
        "        val = 1\n"
        "    return val\n"
    )
    inst_tree = ast.parse(inst_src)
    valid_fn = next(
        n
        for n in inst_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "valid_inst"
    )
    with_inst = next(n for n in valid_fn.body if isinstance(n, ast.With))
    prov_inst = _seed_scope_suppress_aliases(
        valid_fn, mod_tree=inst_tree, u_start=with_inst.lineno
    )
    assert _classify_with_suppression(with_inst, prov_inst) == "non_suppressing"

@pytest.mark.parametrize(
    "mgr_expr",
    [
        "ignoring(ValueError)",
        "tolerant()",
        "skip_errors()",
        "mute()",
        "muting()",
        "maybe()",
        "hush()",
        "discard()",
        "forgiving()",
        "squelch()",
        "quench()",
        "mask_errors()",
        "no_op()",
        "stack.enter_async_context(cm)",
    ],
)
def test_suspicious_suppress_substrings_stems_flagged_in_strict_mode(
    mgr_expr: str,
) -> None:
    """Verifies expanded suppression-idiom stems are classified as potentially suppressing in strict mode."""
    src = (
        "def run_case(compute, stack, cm):\n"
        f"    with {mgr_expr}:\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    sc = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: val\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(src),
    )
    assert "severed_closure_cell" in sc["control_flow_hazards"], mgr_expr

@pytest.mark.parametrize(
    "neutral_chain_expr",
    [
        "self.retry_budget.acquire()",
        "client.fallback_pool.connection()",
        "svc.skip_list.lock()",
        "self.mutex",
        "app.attempt_counter.track()",
    ],
)
def test_suspicious_suppress_substrings_negative_attribute_chain_and_mutex_controls(
    neutral_chain_expr: str,
) -> None:
    """Verifies intermediate receiver attributes and 'mutex' are not falsely flagged as suppressing in strict mode."""
    src = (
        "def run_chain(self, client, svc, app, compute):\n"
        f"    with {neutral_chain_expr}:\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    sc = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: val\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(src),
    )
    assert "severed_closure_cell" not in sc["control_flow_hazards"], neutral_chain_expr

    # Adversarial control: proven suppressor root on an attribute call IS still flagged
    sup_root_src = (
        "import contextlib\n"
        "def run_sup_root(compute):\n"
        "    sup_mod = contextlib\n"
        "    with sup_mod.suppress(ValueError):\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    sc_sup = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "basic_block",
            "source_text": sup_root_src,
            "source_code": "    cb = lambda: val\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(sup_root_src),
    )
    assert "severed_closure_cell" in sc_sup["control_flow_hazards"]

@pytest.mark.parametrize(
    "benign_mgr",
    [
        "masked_array()",
        "MaskedArray()",
        "commute()",
        "CommuteSession()",
        "Mutex()",
        "AsyncMutex()",
        "bootstrap()",
        "BootstrapContext()",
        "tolerance_ctx()",
        "FloatTolerance()",
        "batch_ctx()",
        "dispatch_scope()",
    ],
)
def test_suspicious_suppress_token_boundary_avoids_substring_false_positives(
    benign_mgr: str,
) -> None:
    """Verifies snake_case and CamelCase token boundaries prevent substring false positives on unrelated identifiers."""
    src = (
        "def work(compute):\n"
        f"    with {benign_mgr}:\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    sc = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(src),
    )
    assert sc["is_control_flow_safe"] is True, benign_mgr
    assert "severed_closure_cell" not in sc["control_flow_hazards"], benign_mgr

@pytest.mark.parametrize(
    "suspicious_mgr",
    [
        "mask_errors()",
        "ErrorMute()",
        "CancelScope()",
        "tolerate_errors()",
        "ErrorTolerant()",
        "MoveOnAfter()",
        "SUPPRESS_ERRORS()",
        "HTTPFallbackGuard()",
        "TrapExceptions()",
        "SkipOnError()",
    ],
)
def test_suspicious_suppress_token_boundary_matches_snake_and_camel_case_suppressors(
    suspicious_mgr: str,
) -> None:
    """Verifies snake_case, CamelCase, UPPER_SNAKE, and compound suppressor tokens are flagged in strict mode."""
    src = (
        "def work(compute):\n"
        f"    with {suspicious_mgr}:\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    sc = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(src),
    )
    assert sc["is_control_flow_safe"] is False, suspicious_mgr
    assert "severed_closure_cell" in sc["control_flow_hazards"], suspicious_mgr

def test_strict_heuristic_classification_never_pollutes_pedantic_cache_on_shared_ast() -> None:
    """Verifies heuristic non-suppression in strict mode never caches permanent proof for pedantic mode on the same AST."""
    from pydoppelgangerhunt.config import collect_heuristic_ctx_managers

    clear_dataflow_cache()
    src = (
        "def run_tx(compute):\n"
        "    with Transaction():\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    shared_tree = ast.parse(src)

    with collect_heuristic_ctx_managers() as strict_session_mgrs:
        sc_strict = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    cb = lambda: result\n",
                "closure_strictness": "strict",
            },
            tree=shared_tree,
        )
    assert sc_strict["is_control_flow_safe"] is True
    assert sc_strict["heuristic_ctx_managers"] == ["Transaction"]
    assert strict_session_mgrs == {"Transaction"}

    # Re-inspecting the identical AST object in pedantic mode without cache eviction must reject Transaction()
    with collect_heuristic_ctx_managers() as pedantic_session_mgrs:
        sc_pedantic = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    cb = lambda: result\n",
                "closure_strictness": "pedantic",
            },
            tree=shared_tree,
        )
    assert sc_pedantic["is_control_flow_safe"] is False
    assert "severed_closure_cell" in sc_pedantic["control_flow_hazards"]
    assert sc_pedantic["heuristic_ctx_managers"] == []
    assert pedantic_session_mgrs == set()

    # Supplying explicit allowlist proof on the same AST object succeeds in pedantic mode
    sc_allowlisted = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": ["Transaction"],
        },
        tree=shared_tree,
    )
    assert sc_allowlisted["is_control_flow_safe"] is True

@pytest.mark.parametrize(
    "module_src",
    [
        pytest.param(
            _PROVEN_EXIT_CLASS + "Mgr.__exit__ = lambda self, *a: True\n" + _WITH_MGR_RUN,
            id="module_level_exit_assignment",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "setattr(Mgr, '__exit__', lambda self, *a: True)\n"
            + _WITH_MGR_RUN,
            id="setattr_exit_literal",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "type.__setattr__(Mgr, '__exit__', lambda self, *a: True)\n"
            + _WITH_MGR_RUN,
            id="type_dunder_setattr_exit",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "setattr(Mgr, ATTR_NAME, lambda self, *a: True)\n"
            + _WITH_MGR_RUN,
            id="setattr_non_literal_attribute",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS + "delattr(Mgr, '__exit__')\n" + _WITH_MGR_RUN,
            id="delattr_exit",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS + "exec(PATCH_CODE)\n" + _WITH_MGR_RUN,
            id="dynamic_exec",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "def install_patch():\n"
            "    Mgr.__exit__ = lambda self, *a: True\n"
            + _WITH_MGR_RUN,
            id="patch_inside_helper_function",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "Alias = Mgr\n"
            "Alias.__exit__ = lambda self, *a: True\n"
            + _WITH_MGR_RUN,
            id="patch_through_class_alias",
        ),
        pytest.param(
            "import contextlib\n"
            "contextlib.nullcontext.__exit__ = lambda self, *a: True\n"
            + _WITH_NULLCONTEXT_RUN,
            id="stdlib_nullcontext_exit_patch",
        ),
        pytest.param(
            "def install_patch(manager_cls):\n"
            "    manager_cls.__exit__ = lambda self, *a: True\n"
            "def run(compute):\n"
            "    class Mgr:\n"
            "        def __enter__(self):\n"
            "            return self\n"
            "        def __exit__(self, *exc_info):\n"
            "            return False\n"
            "    install_patch(Mgr)\n"
            "    with Mgr():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            id="function_local_class_patched_by_parameterized_helper",
        ),
    ],
)
def test_runtime_exit_protocol_patch_withdraws_positive_non_suppression_proofs(
    module_src: str,
) -> None:
    """Verifies any runtime __exit__ patch or dynamic exec withdraws class and stdlib non-suppression proofs in pedantic mode."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is False

@pytest.mark.parametrize(
    ("module_src", "expected_strict_safe"),
    [
        pytest.param(
            _PROVEN_EXIT_CLASS + "Mgr.__exit__ = lambda self, *a: True\n" + _WITH_MGR_RUN,
            False,
            id="directly_patched_name_becomes_suspicious_in_strict",
        ),
        pytest.param(
            _PROVEN_EXIT_CLASS + "exec(PATCH_CODE)\n" + _WITH_MGR_RUN,
            True,
            id="untargeted_patch_falls_back_to_strict_name_heuristic",
        ),
    ],
)
def test_runtime_exit_protocol_patch_strict_mode_classification(
    module_src: str, expected_strict_safe: bool
) -> None:
    """Verifies strict mode distrusts directly patched names while untargeted patches defer to the documented heuristic."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["strict"] is expected_strict_safe

@pytest.mark.parametrize(
    ("module_src", "expected_pedantic_safe"),
    [
        pytest.param(_PROVEN_EXIT_CLASS + _WITH_MGR_RUN, True, id="unpatched_local_class"),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "setattr(config, 'name', 1)\n"
            "holder.value = 1\n"
            "holder.__setattr__('name', 2)\n"
            + _WITH_MGR_RUN,
            True,
            id="unrelated_attribute_mutations",
        ),
        pytest.param("import contextlib\n" + _WITH_NULLCONTEXT_RUN, True, id="unpatched_stdlib"),
        pytest.param(
            _PROVEN_EXIT_CLASS
            + "Mgr.__exit__ = lambda self, *a: True\n"
            "def run(compute, path):\n"
            "    with open(path):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="builtin_open_remains_proven_despite_patch",
        ),
        pytest.param(
            "exec('pass')\n"
            "def run(compute, path):\n"
            "    with open(path):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="builtin_open_remains_proven_despite_exec",
        ),
        pytest.param(
            "type(inst).__exit__ = lambda self, *a: True\n"
            "setattr(get_cls(), '__exit__', lambda self, *a: True)\n"
            "def run(compute, path):\n"
            "    with open(path):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="builtin_open_remains_proven_despite_untargeted_exit_patch",
        ),
        pytest.param(
            "import builtins\n"
            "builtins.open = custom_open\n"
            "def run(compute, path):\n"
            "    with open(path):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="neg_rebound_builtins_open_fails_closed",
        ),
        pytest.param(
            "def run(compute, path, open=custom_open):\n"
            "    with open(path):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="neg_shadowed_open_param_fails_closed",
        ),
    ],
)
def test_exit_protocol_patch_negative_controls_keep_positive_proofs(
    module_src: str, expected_pedantic_safe: bool
) -> None:
    """Verifies unrelated attribute mutations and untargeted exit patches keep builtin open proven while explicit open rebinding/shadowing fails closed."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expected_pedantic_safe

@pytest.mark.parametrize(
    ("snippet", "expected_names", "expected_patch"),
    [
        ("Mgr.__exit__ = f\n", {"Mgr"}, True),
        ("del Mgr.__aexit__\n", {"Mgr"}, True),
        ("inst.__class__ = Other\n", {"inst"}, True),
        ("setattr(Mgr, '__exit__', f)\n", {"Mgr"}, True),
        ("setattr(Mgr, name, f)\n", {"Mgr"}, True),
        ("setattr(*args)\n", set(), True),
        ("type.__setattr__(Mgr, '__aexit__', f)\n", {"Mgr"}, True),
        ("Mgr.__setattr__('__exit__', f)\n", {"Mgr"}, True),
        ("type(inst).__exit__ = f\n", set(), True),
        ("eval(code)\n", set(), True),
        ("setattr(Mgr, 'label', f)\n", set(), False),
        ("obj.__setattr__('label', 1)\n", set(), False),
        ("Mgr.label = f\n", set(), False),
        ("value = Mgr.__exit__\n", set(), False),
    ],
)
def test_scan_exit_protocol_patches_classifies_patch_constructs(
    snippet: str, expected_names: set[str], expected_patch: bool
) -> None:
    """Verifies the exit-protocol scan records patched receivers and ignores unrelated attribute traffic."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(snippet))
    assert scan.patched_names == frozenset(expected_names)
    assert scan.has_patch is expected_patch

@pytest.mark.parametrize(
    ("cls_src", "expected_pedantic_safe"),
    [
        pytest.param(_PROVEN_EXIT_CLASS, True, id="direct_falsy_exit"),
        pytest.param(
            "class Mgr:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        pass\n",
            True,
            id="implicit_none_exit",
        ),
        pytest.param(
            "class Safe:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n"
            "Mgr = Safe\n",
            True,
            id="alias_of_proven_class",
        ),
        pytest.param(
            "class Base:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n"
            "class Mgr(Base):\n    pass\n",
            False,
            id="inherited_exit_is_unproven",
        ),
        pytest.param(
            "class Base:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return True\n"
            "class Mgr(Base):\n    pass\n",
            False,
            id="inherited_suppressing_exit",
        ),
        pytest.param(
            "class Base:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n"
            "class Mgr(Base):\n"
            "    def __exit__(self, *exc_info):\n        return True\n",
            False,
            id="override_with_suppressing_exit",
        ),
        pytest.param(
            "class First:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n"
            "class Second:\n"
            "    def __exit__(self, *exc_info):\n        return True\n"
            "class Mgr(First, Second):\n    pass\n",
            False,
            id="multiple_inheritance",
        ),
        pytest.param(
            "from vendor import Base\n"
            "class Mgr(Base):\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n",
            False,
            id="imported_base_class",
        ),
        pytest.param(
            "class Base:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n"
            "Base = make_suppressing_base()\n"
            "class Mgr(Base):\n    pass\n",
            False,
            id="rebound_base_class_name",
        ),
        pytest.param(
            "class Mgr(metaclass=Meta):\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n",
            False,
            id="custom_metaclass",
        ),
        pytest.param(
            "class Mgr:\n"
            "    def __enter__(self):\n        return self\n"
            "    @swallow_errors\n"
            "    def __exit__(self, *exc_info):\n        return False\n",
            False,
            id="decorated_exit_method",
        ),
        pytest.param(
            "@swallow_errors\n"
            "class Mgr:\n"
            "    def __enter__(self):\n        return self\n"
            "    def __exit__(self, *exc_info):\n        return False\n",
            False,
            id="decorated_class",
        ),
        pytest.param(
            "class Outer:\n"
            "    class Mgr:\n"
            "        def __enter__(self):\n            return self\n"
            "        def __exit__(self, *exc_info):\n            return True\n"
            "Mgr = Outer.Mgr\n",
            False,
            id="nested_suppressing_class_alias",
        ),
        pytest.param(
            "class Mgr:\n"
            "    def __enter__(self):\n        return self\n"
            "    __exit__ = Mixin.__exit__\n",
            False,
            id="exit_bound_from_foreign_class",
        ),
    ],
)
def test_class_exit_proof_incomplete_cases_remain_unproven_in_pedantic_mode(
    cls_src: str, expected_pedantic_safe: bool
) -> None:
    """Verifies only complete class-body __exit__ proofs yield non-suppression; incomplete proofs stay unknown."""
    safety = _late_bound_capture_safety_by_mode(
        cls_src + _WITH_MGR_RUN, "    cb = lambda: v"
    )
    assert safety["pedantic"] is expected_pedantic_safe

@pytest.mark.parametrize(
    ("module_src", "expected_safe"),
    [
        pytest.param(
            "import contextlib\n"
            "def suppress_all(*a): return True\n"
            "contextlib.nullcontext = suppress_all\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="direct_contextlib_nullcontext_rebinding",
        ),
        pytest.param(
            "import contextlib as cl\n"
            "def suppress_all(*a): return True\n"
            "cl.nullcontext = suppress_all\n"
            "def run(compute):\n"
            "    with cl.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="aliased_contextlib_nullcontext_rebinding",
        ),
        pytest.param(
            "import contextlib as cl\n"
            "from contextlib import nullcontext\n"
            "def suppress_all(*a): return True\n"
            "cl.nullcontext = suppress_all\n"
            "def run(compute):\n"
            "    with nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="from_import_invalidated_by_module_alias_rebinding",
        ),
        pytest.param(
            "import builtins\n"
            "def suppress_all(*a): return True\n"
            "builtins.open = suppress_all\n"
            "def run(compute):\n"
            "    with builtins.open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="builtins_open_rebinding",
        ),
        pytest.param(
            "import builtins\n"
            "def suppress_all(*a): return True\n"
            "builtins.open = suppress_all\n"
            "def run(compute):\n"
            "    with open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="bare_open_invalidated_by_builtins_open_rebinding",
        ),
        pytest.param(
            "import io\n"
            "def suppress_all(*a): return True\n"
            "io.open = suppress_all\n"
            "def run(compute):\n"
            "    with io.open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="io_open_rebinding",
        ),
        pytest.param(
            "import contextlib\n"
            "def suppress_all(*a): return True\n"
            "setattr(contextlib, 'nullcontext', suppress_all)\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="setattr_contextlib_nullcontext",
        ),
        pytest.param(
            "import contextlib\n"
            "delattr(contextlib, 'nullcontext')\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="delattr_contextlib_nullcontext",
        ),
        pytest.param(
            "import contextlib\n"
            "def patch(name, fn):\n"
            "    setattr(contextlib, name, fn)\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="setattr_nonliteral_attr_on_contextlib",
        ),
        pytest.param(
            "class Door:\n"
            "    def close(self):\n"
            "        self.open = False\n"
            "def run(compute):\n"
            "    with open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="unrelated_self_open_does_not_invalidate_builtin_open",
        ),
    ],
)
def test_stdlib_factory_rebinding_invalidates_non_suppression_proof(
    module_src: str, expected_safe: bool
) -> None:
    """Verifies module-attribute rebinding (direct, aliased, setattr, delattr, nonliteral) invalidates stdlib factory proofs."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expected_safe, "pedantic": expected_safe}

@pytest.mark.parametrize(
    ("mgr_expr", "expect_definite"),
    [
        ("handle_errors()", False),
        ("ErrorHandler()", False),
        ("safe_run()", False),
        ("failsafe()", False),
        ("Rescue()", False),
        ("recover()", False),
        ("allow_failure()", False),
        ("ContinueOnError()", False),
        ("Masker()", False),
        ("Muter()", False),
        ("CustomTransaction()", True),
        ("masked_array()", True),
        ("tolerance_ctx()", True),
    ],
)
def test_suspicious_suppress_name_stems_and_neutral_controls(
    mgr_expr: str, expect_definite: bool
) -> None:
    """Verifies suppression-idiom stems match common error-handling managers while preserving neutral controls."""
    import ast

    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    stmts = ast.parse(f"with {mgr_expr}:\n    committed = compute_value()\n").body
    definite, conditional = _analyze_block_assignment(
        stmts, strict_with_suppression=False
    )
    assert ("committed" in definite) is expect_definite
    assert ("committed" in conditional) is (not expect_definite)

@pytest.mark.parametrize(
    ("stmt", "expect_has_patch", "expect_rebound_attrs"),
    [
        ("model.eval()", False, False),
        ('df.eval("a + b")', False, False),
        ("self.conn.connect = fn", False, False),
        ("import builtins as b\nb.eval('1')", True, True),
        ("import contextlib\ncontextlib.nullcontext = fn", False, True),
    ],
)
def test_exit_protocol_scanner_avoids_non_builtin_eval_and_nested_attr_stores(
    stmt: str, expect_has_patch: bool, expect_rebound_attrs: bool
) -> None:
    """Verifies non-builtin .eval() calls and nested attribute stores do not trigger exit-protocol invalidation."""
    import ast

    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    scan = scan_exit_protocol_patches(ast.parse(stmt))
    assert scan.has_patch is expect_has_patch
    assert bool(scan.rebound_attrs) is expect_rebound_attrs

def test_exit_protocol_shadowed_eval_and_benign_alias_keep_proven_ctx_manager() -> None:
    """Verifies shadowed local builtins and benign setattr calls do not withdraw proven class non-suppression in pedantic mode."""
    src = (
        _PROVEN_EXIT_CLASS
        + "def configure(eval: bool = False):\n"
        "    if eval:\n"
        "        return 1\n"
        "    return 0\n"
        "def label_item(item):\n"
        "    s = setattr\n"
        "    s(item, 'tag', 'ok')\n"
        + _WITH_MGR_RUN
    )
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["pedantic"] is True
    # Negative control: patching Mgr.__exit__ withdraws the non-suppression proof on the same capture unit
    patched_src = _PROVEN_EXIT_CLASS + "Mgr.__exit__ = lambda self, *a: True\n" + _WITH_MGR_RUN
    patched_safety = _late_bound_capture_safety_by_mode(patched_src, "    cb = lambda: v")
    assert patched_safety["pedantic"] is False

@pytest.mark.parametrize(
    ("module_src", "expected_safe"),
    [
        pytest.param(
            "from contextlib import nullcontext\n"
            "def install():\n"
            "    import contextlib as cl\n"
            "    cl.nullcontext = lambda *a: True\n"
            "def run(compute):\n"
            "    with nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="helper_scoped_cl_alias_invalidates_from_import_nullcontext",
        ),
        pytest.param(
            "import contextlib\n"
            "def install():\n"
            "    import contextlib as cl\n"
            "    cl.nullcontext = lambda *a: True\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="helper_scoped_cl_alias_invalidates_contextlib_attr_nullcontext",
        ),
        pytest.param(
            "import contextlib\n"
            "def run(compute):\n"
            "    with contextlib.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n"
            "import contextlib as cl\n"
            "cl.nullcontext = lambda *a: True\n",
            False,
            id="post_unit_module_alias_rebinding",
        ),
        pytest.param(
            "def install():\n"
            "    import builtins as b\n"
            "    b.open = lambda *a: True\n"
            "def run(compute):\n"
            "    with open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="helper_scoped_builtins_alias_open_rebinding",
        ),
        pytest.param(
            "import io\n"
            "def patch(name, fn):\n"
            "    import io as iomod\n"
            "    setattr(iomod, name, fn)\n"
            "def run(compute):\n"
            "    with io.open('x'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="helper_scoped_io_dynamic_setattr_rebinding",
        ),
        pytest.param(
            "import unittest\n"
            "unittest.mock.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="unittest_import_dotted_mock_patch_rebinding",
        ),
        pytest.param(
            "import unittest.mock\n"
            "unittest.mock.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="unittest_mock_import_dotted_patch_rebinding",
        ),
        pytest.param(
            "import unittest as ut\n"
            "ut.mock.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="unittest_alias_dotted_mock_patch_rebinding",
        ),
        pytest.param(
            "def install():\n"
            "    from unittest import mock\n"
            "    mock.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    import unittest.mock\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="from_unittest_import_mock_patch_rebinding",
        ),
        pytest.param(
            "import unittest.mock\n"
            "class C:\n"
            "    def update(self):\n"
            "        self.conn.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="control_nested_self_conn_patch_assignment",
        ),
        pytest.param(
            "import unittest.mock\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="control_benign_unittest_mock_patch",
        ),
        pytest.param(
            "from unittest import mock\n"
            "def run(compute):\n"
            "    with mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="control_benign_from_unittest_import_mock_patch",
        ),
    ],
)
def test_stdlib_scoped_aliases_and_dotted_chains_provenance_safety(
    module_src: str, expected_safe: bool
) -> None:
    """Verifies scoped/post-unit/chained stdlib aliases and dotted attribute chains withdraw non-suppression proofs."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expected_safe, "pedantic": expected_safe}

@pytest.mark.parametrize(
    ("prefix_src", "ctx_expr", "expect_safe", "expect_rebound"),
    [
        pytest.param(
            "def trigger(code):\n    exec(code)\n",
            "open(path)",
            {"strict": True, "pedantic": True},
            True,
            id="exec_keeps_builtin_open_trusted",
        ),
        pytest.param(
            "def trigger(code):\n    return eval(code)\n",
            "builtins.open(path)",
            {"strict": True, "pedantic": True},
            True,
            id="eval_keeps_builtins_open_trusted",
        ),
        pytest.param(
            "def trigger(code):\n    exec(code)\n",
            "contextlib.nullcontext()",
            {"strict": False, "pedantic": False},
            True,
            id="exec_still_withdraws_and_invalidates_contextlib_nullcontext",
        ),
        pytest.param(
            "def trigger(get_cls, fn):\n    setattr(get_cls(), '__exit__', fn)\n",
            "open(path)",
            {"strict": True, "pedantic": True},
            False,
            id="setattr_dynamic_receiver_exit_keeps_open_trusted",
        ),
        pytest.param(
            "builtins.open = lambda *a: True\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_direct_builtins_open_rebind_invalidates_open",
        ),
        pytest.param(
            "import builtins as b\nb.open = lambda *a: True\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_aliased_builtins_open_rebind_invalidates_open",
        ),
        pytest.param(
            "def patch(attr_name, fn):\n    setattr(builtins, attr_name, fn)\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_setattr_builtins_dynamic_attr_invalidates_open",
        ),
        pytest.param(
            "io.open = lambda *a: True\n",
            "open(path)",
            {"strict": True, "pedantic": True},
            True,
            id="io_open_rebind_leaves_bare_builtin_open_trusted",
        ),
        pytest.param(
            "io.open = lambda *a: True\n",
            "io.open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_io_open_rebind_invalidates_io_open",
        ),
    ],
)
def test_builtin_open_trusted_despite_exec_eval_and_untargeted_patches(
    prefix_src: str,
    ctx_expr: str,
    expect_safe: dict[str, bool],
    expect_rebound: bool,
) -> None:
    """Verifies builtin open remains trusted on exec/eval/untargeted patches unless open/builtins itself is rebound."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = (
        "import builtins\n"
        "import contextlib\n"
        "import io\n"
        + _PROVEN_EXIT_CLASS
        + prefix_src
        + "def run(path, compute):\n"
        + f"    with {ctx_expr}:\n"
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert bool(scan.rebound_attrs) is expect_rebound
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == expect_safe

@pytest.mark.parametrize(
    ("module_src", "unit_line", "extra_non_suppressing", "expected_safety"),
    [
        pytest.param(
            (
                "def my_transaction():\n"
                "    return ExternalTx()\n"
                "def run(compute):\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": True},
            id="module_def_allowlisted_helper_trusted_in_pedantic",
        ),
        pytest.param(
            (
                "from contextlib import contextmanager\n"
                "@contextmanager\n"
                "def my_transaction():\n"
                "    yield\n"
                "def run(compute):\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": True},
            id="module_generator_contextmanager_allowlisted_trusted_in_pedantic",
        ),
        pytest.param(
            (
                "async def my_transaction():\n"
                "    return ExternalTx()\n"
                "async def run(compute):\n"
                "    async with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": True},
            id="module_async_def_allowlisted_trusted_in_pedantic",
        ),
        pytest.param(
            (
                "def run(compute):\n"
                "    def my_transaction():\n"
                "        return ExternalTx()\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": True},
            id="local_def_allowlisted_helper_before_with_trusted_in_pedantic",
        ),
        pytest.param(
            (
                "def my_transaction():\n"
                "    return ExternalTx()\n"
                "def run(compute):\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset(),
            {"strict": True, "pedantic": False},
            id="negative_unallowlisted_helper_fails_closed_in_pedantic",
        ),
        pytest.param(
            (
                "def my_transaction():\n"
                "    return ExternalTx()\n"
                "def run(my_transaction, compute):\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": False},
            id="negative_param_shadowing_allowlisted_def_stays_rejected",
        ),
        pytest.param(
            (
                "def my_transaction():\n"
                "    return ExternalTx()\n"
                "def run(flag, unknown_cm, compute):\n"
                "    if flag:\n"
                "        def my_transaction():\n"
                "            return unknown_cm\n"
                "    with my_transaction():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"my_transaction"}),
            {"strict": True, "pedantic": False},
            id="negative_conditional_def_rebind_invalidates_allowlisted_helper",
        ),
        pytest.param(
            (
                "def open(path):\n"
                "    return ExternalTx(path)\n"
                "def run(path, compute):\n"
                "    with open(path):\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"open"}),
            {"strict": False, "pedantic": False},
            id="negative_def_open_marks_shadowable_builtin_unsafe",
        ),
        pytest.param(
            (
                "def get_cm():\n"
                "    return SwallowErrors()\n"
                "def run(compute):\n"
                "    with get_cm():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset(),
            {"strict": False, "pedantic": False},
            id="def_returning_suspicious_manager_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "def get_cm():\n"
                "    cm = SwallowErrors()\n"
                "    return cm\n"
                "def run(compute):\n"
                "    with get_cm():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset(),
            {"strict": False, "pedantic": False},
            id="def_returning_indirect_suspicious_alias_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def get_cm():\n"
                "    return contextlib.suppress(RuntimeError)\n"
                "def run(compute):\n"
                "    with get_cm():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            frozenset({"get_cm"}),
            {"strict": False, "pedantic": False},
            id="negative_allowlisted_name_returning_explicit_suppress_stays_unsafe",
        ),
    ],
)
def test_local_ctx_manager_function_allowlist_and_suspicious_return(
    module_src: str,
    unit_line: str,
    extra_non_suppressing: frozenset[str],
    expected_safety: dict[str, bool],
) -> None:
    """Verifies local helper def/async def allowlist unshadowing and suspicious/suppressor return propagation."""
    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == unit_line
    )
    tree = ast.parse(module_src)
    got: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": line_num,
                "end": line_num,
                "kind": "basic_block",
                "source_text": module_src,
                "source_code": unit_line + "\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": set(extra_non_suppressing),
            },
            tree=tree,
        )
        got[mode] = bool(sc["is_control_flow_safe"])
    assert got == expected_safety

@pytest.mark.parametrize(
    ("src", "min_py", "expect_has_patch"),
    [
        pytest.param(
            (
                "def f(obj):\n"
                "    from builtins import setattr\n"
                "    sa = setattr\n"
                "    sa(obj, 'safe_label', 1)\n"
            ),
            (3, 9),
            False,
            id="from_builtins_import_setattr_local_pure_alias_benign",
        ),
        pytest.param(
            (
                "def swallow(*a):\n"
                "    return True\n"
                "def f(cls):\n"
                "    from mod import eval as other\n"
                "    return eval('1')\n"
            ),
            (3, 9),
            True,
            id="import_eval_as_other_does_not_shadow_builtin_eval",
        ),
        pytest.param(
            (
                "def swallow(*a):\n"
                "    return True\n"
                "def f(cls):\n"
                "    from mod import setattr as sa\n"
                "    setattr(cls, '__exit__', swallow)\n"
            ),
            (3, 9),
            True,
            id="import_setattr_as_sa_does_not_shadow_builtin_setattr",
        ),
        pytest.param(
            (
                "def outer():\n"
                "    def inner[eval](x: eval):\n"
                "        return x\n"
                "    return eval('1')\n"
            ),
            (3, 12),
            True,
            id="pep695_type_param_named_eval_does_not_shadow_enclosing_eval",
        ),
        pytest.param(
            (
                "type Alias[eval] = int\n"
                "eval('1')\n"
            ),
            (3, 12),
            True,
            id="pep695_type_alias_type_param_named_eval_does_not_shadow_module_eval",
        ),
        pytest.param(
            (
                "type eval = int\n"
                "x = eval\n"
            ),
            (3, 12),
            False,
            id="pep695_type_alias_named_eval_shadows_module_eval",
        ),
    ],
)
def test_exit_protocol_import_alias_type_alias_and_pep695_type_params(
    src: str,
    min_py: tuple[int, int],
    expect_has_patch: bool,
) -> None:
    """Verifies ast.alias children, TypeAlias, and PEP 695 type parameters are handled accurately in exit-protocol scans."""
    import sys
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    if sys.version_info < min_py:
        pytest.skip(f"Requires Python {min_py[0]}.{min_py[1]}+ syntax")
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch

@pytest.mark.parametrize(
    ("module_src", "expect_safety"),
    [
        pytest.param(
            (
                "def contextlib():\n"
                "    return object()\n"
                "def run(step):\n"
                "    with contextlib.nullcontext():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": True, "pedantic": False},
            id="local_def_contextlib_does_not_trust_implicit_nullcontext",
        ),
        pytest.param(
            (
                "import math as contextlib\n"
                "def run(step):\n"
                "    with contextlib.nullcontext():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": True, "pedantic": False},
            id="unrelated_stdlib_aliased_as_contextlib_fails_closed",
        ),
        pytest.param(
            (
                "class AbstractContextManager:\n"
                "    def __exit__(self, *a):\n"
                "        return True\n"
                "class CustomCM(AbstractContextManager):\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "def run(step):\n"
                "    with CustomCM():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": False, "pedantic": False},
            id="local_class_shadowing_abstract_context_manager_fails_closed",
        ),
        pytest.param(
            (
                "from evil_pkg import AbstractContextManager\n"
                "class CustomCM(AbstractContextManager):\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "def run(step):\n"
                "    with CustomCM():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": True, "pedantic": False},
            id="non_stdlib_import_abstract_context_manager_fails_closed",
        ),
        pytest.param(
            (
                "import contextlib as cl\n"
                "class SafeCM(cl.AbstractContextManager):\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "def run(step):\n"
                "    with SafeCM():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": True, "pedantic": True},
            id="aliased_stdlib_contextlib_base_remains_trusted",
        ),
        pytest.param(
            (
                "def property():\n"
                "    return lambda fn: fn\n"
                "class CustomCM:\n"
                "    @property\n"
                "    def helper(self):\n"
                "        return True\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "def run(step):\n"
                "    with CustomCM():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            {"strict": True, "pedantic": False},
            id="shadowed_property_decorator_fails_closed_for_custom_cm",
        ),
    ],
)
def test_shadowed_or_non_stdlib_decorators_bases_and_factories_fail_closed(
    module_src: str,
    expect_safety: dict[str, bool],
) -> None:
    """Verifies non-stdlib imports or defs shadowing contextlib, base classes, and decorators are never positively proven."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == expect_safety

@pytest.mark.parametrize(
    ("src", "expect_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "Mgr.__exit__ = swallow\n",
            True,
            frozenset({"Mgr"}),
            id="direct_exit_attr_assignment",
        ),
        pytest.param(
            "del Mgr.__aexit__\n",
            True,
            frozenset({"Mgr"}),
            id="direct_aexit_attr_deletion",
        ),
        pytest.param(
            "setattr(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="direct_setattr_exit_protocol",
        ),
        pytest.param(
            "delattr(Mgr, '__class__')\n",
            True,
            frozenset({"Mgr"}),
            id="direct_delattr_class_protocol",
        ),
        pytest.param(
            "type.__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="direct_type_dunder_setattr_exit",
        ),
        pytest.param(
            "Mgr.__setattr__('__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="direct_bound_dunder_setattr_exit",
        ),
        pytest.param(
            "exec('pass')\n",
            True,
            frozenset(),
            id="direct_exec_call",
        ),
        pytest.param(
            "eval('1 + 1')\n",
            True,
            frozenset(),
            id="direct_eval_call",
        ),
        pytest.param(
            "setattr(Mgr, 'benign_label', 1)\n",
            False,
            frozenset(),
            id="control_direct_setattr_non_exit_attr_ignored",
        ),
        pytest.param(
            "Mgr.__setattr__('benign_label', 1)\n",
            False,
            frozenset(),
            id="control_direct_dunder_setattr_non_exit_attr_ignored",
        ),
    ],
)
def test_finite_syntactic_exit_protocol_guard_matrix(
    src: str,
    expect_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies the finite syntactic exit-protocol guard detects direct mutations and ignores benign attributes."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch
    assert scan.patched_names == expected_patched_names

@pytest.mark.parametrize(
    ("module_src", "allowlist", "expect_pedantic", "expect_strict"),
    [
        pytest.param(
            (
                "from pkg.tx import atomic\n"
                "exec(code)\n"
                "def run(compute):\n"
                "    with atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.tx.atomic"],
            True,
            True,
            id="allowlisted_from_imported_name_stays_trusted_under_exec",
        ),
        pytest.param(
            (
                "import pkg.tx\n"
                "exec(code)\n"
                "def run(compute):\n"
                "    with pkg.tx.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.tx.atomic"],
            True,
            True,
            id="allowlisted_attribute_chain_stays_trusted_under_exec",
        ),
        pytest.param(
            (
                "from pkg.tx import atomic\n"
                "eval(code)\n"
                "def run(compute):\n"
                "    with atomic:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.tx.atomic"],
            True,
            True,
            id="allowlisted_instance_name_stays_trusted_under_eval",
        ),
        pytest.param(
            (
                "from pkg.tx import atomic\n"
                "atomic.__exit__ = swallow\n"
                "def run(compute):\n"
                "    with atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.tx.atomic"],
            False,
            False,
            id="control_directly_patched_allowlisted_name_fails_closed",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "from contextlib import nullcontext\n"
                "contextlib.nullcontext = bad_cm\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["contextlib.nullcontext"],
            False,
            False,
            id="control_targeted_factory_rebind_revokes_allowlisted_name",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "exec(code)\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            [],
            False,
            False,
            id="control_unallowlisted_imported_factory_revoked_under_exec",
        ),
    ],
)
def test_explicit_allowlist_name_remains_trusted_under_exec_eval_and_fails_on_targeted_rebind(
    module_src: str,
    allowlist: list[str],
    expect_pedantic: bool,
    expect_strict: bool,
) -> None:
    """Verifies ast.Name and ast.Attribute explicit allowlist entries stay trusted under untargeted exec/eval while targeted patches fail closed."""
    unit_line = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == "    cb = lambda: v"
    )
    tree = ast.parse(module_src)
    for mode, expected in (("strict", expect_strict), ("pedantic", expect_pedantic)):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": module_src,
                "source_code": "    cb = lambda: v\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": allowlist,
            },
            tree=tree,
        )
        assert bool(sc["is_control_flow_safe"]) is expected, (mode, sc)

@pytest.mark.parametrize(
    ("mutation_stmt", "expect_has_patch", "expect_pedantic_safe"),
    [
        pytest.param(
            "Mgr.__bases__ = (Suppressor,)\n",
            True,
            False,
            id="direct_bases_assignment",
        ),
        pytest.param(
            "setattr(Mgr, '__bases__', (Suppressor,))\n",
            True,
            False,
            id="setattr_bases_mutation",
        ),
        pytest.param(
            "delattr(Mgr, '__bases__')\n",
            True,
            False,
            id="delattr_bases_mutation",
        ),
        pytest.param(
            "setattr(Mgr, 'label', 1)\n",
            False,
            True,
            id="control_benign_setattr_non_protocol_attr",
        ),
    ],
)
def test_exit_protocol_bases_mutation_withdraws_non_suppression_proofs(
    mutation_stmt: str,
    expect_has_patch: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies runtime __bases__ mutations trigger exit-protocol patch detection and withdraw class non-suppression proofs."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    src = _PROVEN_EXIT_CLASS + mutation_stmt + _WITH_MGR_RUN
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe, safety

@pytest.mark.parametrize(
    ("module_src", "allowlist", "expect_strict", "expect_pedantic"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "contextlib.nullcontext = bad_cm\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["contextlib.nullcontext"],
            False,
            False,
            id="rebound_contextlib_attr_overrides_exact_allowlist",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "contextlib.nullcontext = bad_cm\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["contextlib.nullcontext"],
            False,
            False,
            id="rebound_contextlib_attr_instance_overrides_exact_allowlist",
        ),
        pytest.param(
            (
                "import pkg\n"
                "from pkg import atomic\n"
                "pkg.atomic = swallow\n"
                "def run(compute):\n"
                "    with atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.atomic"],
            False,
            False,
            id="rebound_non_stdlib_import_overrides_from_imported_allowlist",
        ),
        pytest.param(
            (
                "import pkg\n"
                "pkg.atomic = swallow\n"
                "def run(compute):\n"
                "    with pkg.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.atomic"],
            False,
            False,
            id="rebound_non_stdlib_import_overrides_attribute_allowlist",
        ),
        pytest.param(
            (
                "import pkg as p\n"
                "from pkg import atomic\n"
                "p.atomic = swallow\n"
                "def run(compute):\n"
                "    with atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.atomic"],
            False,
            False,
            id="rebound_aliased_non_stdlib_import_overrides_allowlist",
        ),
        pytest.param(
            (
                "import pkg.tx\n"
                "pkg.tx.atomic = swallow\n"
                "def run(compute):\n"
                "    with pkg.tx.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.tx.atomic"],
            False,
            False,
            id="rebound_nested_non_stdlib_import_overrides_allowlist",
        ),
        pytest.param(
            (
                "import pkg\n"
                "setattr(pkg, 'atomic', swallow)\n"
                "def run(compute):\n"
                "    with pkg.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.atomic"],
            False,
            False,
            id="setattr_rebound_non_stdlib_import_overrides_allowlist",
        ),
        pytest.param(
            (
                "class Mgr:\n"
                "    pass\n"
                "Mgr.__exit__ = swallow\n"
                "def run(compute):\n"
                "    with Mgr.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["*.atomic"],
            False,
            False,
            id="directly_patched_receiver_blocked_from_wildcard_allowlist",
        ),
        pytest.param(
            (
                "from unknown_pkg import *\n"
                "def run(db, compute):\n"
                "    with db.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["*.atomic"],
            True,
            False,
            id="star_import_disables_wildcard_allowlist_in_pedantic",
        ),
        pytest.param(
            (
                "import pkg\n"
                "def run(compute):\n"
                "    with pkg.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["pkg.atomic"],
            True,
            True,
            id="control_unrebound_non_stdlib_attribute_allowlist_stays_trusted",
        ),
        pytest.param(
            (
                "def run(db, compute):\n"
                "    with db.atomic():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            ["*.atomic"],
            True,
            True,
            id="control_unpatched_param_receiver_matches_wildcard_allowlist",
        ),
    ],
)
def test_attribute_allowlist_respects_targeted_factory_rebind_and_patched_wildcard_receivers(
    module_src: str,
    allowlist: list[str],
    expect_strict: bool,
    expect_pedantic: bool,
) -> None:
    """Verifies attribute allowlists fail closed on targeted stdlib/non-stdlib factory rebinds and patched receivers."""
    unit_line = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == "    cb = lambda: v"
    )
    tree = ast.parse(module_src)
    for mode, expected in (("strict", expect_strict), ("pedantic", expect_pedantic)):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": module_src,
                "source_code": "    cb = lambda: v\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": allowlist,
            },
            tree=tree,
        )
        assert bool(sc["is_control_flow_safe"]) is expected, (mode, sc)

@pytest.mark.parametrize(
    ("setup_src", "with_expr", "expect_safe"),
    [
        pytest.param(
            "from contextlib import suppress\nsup = suppress\n",
            "sup().atomic()",
            False,
            id="call_chain_suppressor_root_rejected_by_wildcard",
        ),
        pytest.param(
            "from contextlib import suppress\nsup = suppress\n",
            "sup()[0].atomic()",
            False,
            id="subscript_call_chain_suppressor_root_rejected_by_wildcard",
        ),
        pytest.param(
            "",
            "pkg.suppress().atomic()",
            False,
            id="intermediate_suppress_attr_in_call_chain_rejected_by_wildcard",
        ),
        pytest.param(
            "from contextlib import suppress\nsup = suppress\n",
            "(sup if flag else db).atomic()",
            False,
            id="conditional_expr_suppressor_branch_rejected_by_wildcard",
        ),
        pytest.param(
            "import mypkg\nmypkg.get_tx = swallow\n",
            "mypkg.get_tx().atomic()",
            False,
            id="rebound_intermediate_factory_call_chain_rejected_by_wildcard",
        ),
        pytest.param(
            "",
            "get_db().atomic()",
            True,
            id="control_benign_factory_call_chain_matches_wildcard",
        ),
        pytest.param(
            "",
            "(tx := get_db()).atomic()",
            True,
            id="control_benign_walrus_call_chain_matches_wildcard",
        ),
    ],
)
def test_wildcard_ctx_manager_rejects_suppressor_and_rebound_call_chains(
    setup_src: str,
    with_expr: str,
    expect_safe: bool,
) -> None:
    """Verifies *.attr wildcard matching peels receiver call chains and rejects suppressor/rebound roots."""
    src = (
        setup_src
        + "def run(flag, db, pkg, get_db, compute):\n"
        + f"    with {with_expr}:\n"
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    unit_line = next(
        idx
        for idx, ln in enumerate(src.splitlines(), start=1)
        if ln == "    cb = lambda: v"
    )
    tree = ast.parse(src)
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": unit_line,
            "end": unit_line,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "    cb = lambda: v\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": ["*.atomic"],
        },
        tree=tree,
    )
    assert bool(sc["is_control_flow_safe"]) is expect_safe, sc

def test_try_finally_auxiliary_probes_isolate_heuristic_and_wildcard_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies auxiliary _run_finally probes do not leak heuristic or wildcard telemetry into active collectors."""
    import pydoppelgangerhunt.fixer._df_block_flow as wp_mod
    from pydoppelgangerhunt.config import (  # pylint: disable=import-outside-toplevel
        clear_closure_warning_cache,
        collect_heuristic_ctx_managers,
        collect_wildcard_ctx_managers,
    )

    src = (
        "def run(items, step):\n"
        "    for _ in items:\n"
        "        try:\n"
        "            x = step()\n"
        "            break\n"
        "        finally:\n"
        "            with NeutralFinallyMgr():\n"
        "                del x\n"
        "                break\n"
    )
    # Verify primary _run_finally pass populates outer collectors cleanly, while
    # clearing immediately after the primary pass proves all auxiliary probes
    # (abrupt_live, f_jumps, f_standalone) are isolated and do not re-populate them.
    for strict_mode, extra_mgrs, use_wc in (
        (False, frozenset(), False),
        (True, frozenset({"*.finally_mgr"}), True),
    ):
        case_src = src.replace("NeutralFinallyMgr()", "svc.finally_mgr()") if use_wc else src
        fn_node = ast.parse(case_src).body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        loop_node = fn_node.body[0]
        assert isinstance(loop_node, ast.For)
        try_node = loop_node.body[0]
        assert isinstance(try_node, ast.Try)
        f_body = try_node.finalbody

        clear_closure_warning_cache()
        orig_analyze = wp_mod._analyze_block_assignment_impl
        f_body_passes = 0
        with collect_heuristic_ctx_managers() as outer_h, collect_wildcard_ctx_managers() as outer_wc:

            def _make_spy(
                target_body: Any,
                orig_fn: Any,
                expect_wc: bool,
                heur_bucket: set[str],
                wc_bucket: set[tuple[str, str]],
            ) -> Any:
                def _spy(stmts: Any, *args: Any, **kwargs: Any) -> Any:
                    nonlocal f_body_passes
                    res = orig_fn(stmts, *args, **kwargs)
                    if stmts is target_body or stmts == target_body:
                        f_body_passes += 1
                        if f_body_passes == 1:
                            if expect_wc:
                                assert ("*.finally_mgr", "svc.finally_mgr") in wc_bucket
                                wc_bucket.clear()
                            else:
                                assert "NeutralFinallyMgr" in heur_bucket
                                heur_bucket.clear()
                    return res

                return _spy

            monkeypatch.setattr(
                wp_mod,
                "_analyze_block_assignment_impl",
                _make_spy(f_body, orig_analyze, use_wc, outer_h, outer_wc),
            )
            wp_mod._extract_block_definite_bindings_and_term(
                fn_node.body,
                active_nonlocals=None,
                strict_with_suppression=strict_mode,
                extra_non_suppressing_ctx_managers=extra_mgrs,
            )
            monkeypatch.setattr(wp_mod, "_analyze_block_assignment_impl", orig_analyze)
        assert f_body_passes >= 3
        assert outer_h == set()
        assert outer_wc == set()

@pytest.mark.parametrize(
    ("with_header", "expect_cm_invalidated"),
    [
        pytest.param(
            "with Transaction():\n",
            True,
            id="strict_neutral_unproven_with_invalidates_cross_scope_mutated_on_exit",
        ),
        pytest.param(
            "with contextlib.nullcontext():\n",
            False,
            id="control_proven_nullcontext_preserves_cross_scope_mutated_on_exit",
        ),
    ],
)
def test_neutral_with_exit_still_invalidates_cross_scope_mutated_provenance(
    with_header: str,
    expect_cm_invalidated: bool,
) -> None:
    """Verifies neutral unproven with managers still trigger cross-scope exit invalidation in strict mode."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import _update_prov_and_term_from_stmts
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

    src = (
        "import contextlib\n"
        "cm = contextlib.nullcontext\n"
        "def mutate():\n"
        "    nonlocal cm\n"
        "    cm = contextlib.suppress\n"
        + with_header
        + "    pass\n"
    )
    stmts = ast.parse(src).body
    res = _update_prov_and_term_from_stmts(
        stmts,
        _EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
        strict_with_suppression=False,
    )
    assert ("cm" not in res.prov.non_suppressors) is expect_cm_invalidated
    assert ("cm" in res.prov.suppressors) is expect_cm_invalidated
