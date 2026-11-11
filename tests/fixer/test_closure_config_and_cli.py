"""Tests for closure strictness precedence, heuristic session summaries, patch skip telemetry, and --suggest."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
from pathlib import Path
import sys
import threading
from types import MappingProxyType
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    analyze_unit_variable_scope,
    clear_dataflow_cache,
    generate_refactoring_patch,
    inspect_single_unit_scope,
    resolve_clone_generator_subroutine_outputs,
    resolve_clone_pair_outputs,
    synthesize_shared_helper_code,
)


def test_closure_strictness_precedence_ctx_managers_and_global_pure_builtins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies fail_closed->pedantic mapping, strictness precedence, non_suppressing_ctx_managers plumbing, and pure builtin calls with global reads."""
    import logging
    from pydoppelgangerhunt.config import (
        normalize_non_suppressing_ctx_managers,
        resolve_closure_strictness_mode,
        resolve_strict_with_suppression,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _is_ordered_contiguous_source_match,
    )
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    # 1. fail_closed maps to pedantic and explicit closure_strictness overrides conflicting strict_with_suppression
    assert resolve_closure_strictness_mode("fail_closed") == (
        "pedantic",
        False,
    )
    assert resolve_strict_with_suppression(closure_strictness="fail_closed") is True
    with caplog.at_level(logging.WARNING):
        resolved_conflict = resolve_strict_with_suppression(
            closure_strictness="strict",
            strict_with_suppression=True,
        )
    assert resolved_conflict is False
    assert "Conflicting configuration: closure_strictness" in caplog.text

    assert normalize_non_suppressing_ctx_managers("SafeTx, db.tx , ") == (
        "SafeTx",
        "db.tx",
    )

    # 2. subTest, ExitStack, AsyncExitStack, enter_context are treated as potentially suppressing in default strict mode
    for mgr_expr in (
        "self.subTest(i=1)",
        "ExitStack()",
        "AsyncExitStack()",
        "stack.enter_context(cm)",
    ):
        stmts = ast.parse(f"with {mgr_expr}:\n    v = compute()\n").body
        from pydoppelgangerhunt.fixer._df_with_provenance import (
            _analyze_block_assignment,
        )

        defs, conds = _analyze_block_assignment(stmts, strict_with_suppression=False)
        assert "v" not in defs
        assert "v" in conds

    # 3. End-to-end non_suppressing_ctx_managers plumbing in pedantic mode
    pedantic_src = (
        "class Service:\n"
        "    def run_a(self, x: int) -> int:\n"
        "        with CustomAuditScope():\n"
        "            val = self.compute(x)\n"
        "        res = val + 1\n"
        "        return res\n"
        "\n"
        "    def run_b(self, x: int) -> int:\n"
        "        with CustomAuditScope():\n"
        "            val = self.compute(x)\n"
        "        res = val + 1\n"
        "        return res\n"
    )
    f_ped = tmp_path / "ped_service.py"
    f_ped.write_text(pedantic_src, encoding="utf-8")
    u_ped1 = {
        "file": "ped_service.py",
        "start": 3,
        "end": 4,
        "name": "run_a_with",
        "kind": "compound_block",
    }
    u_ped2 = {
        "file": "ped_service.py",
        "start": 9,
        "end": 10,
        "name": "run_b_with",
        "kind": "compound_block",
    }
    sc_ped_blocked = analyze_unit_variable_scope(
        u_ped1,
        u_ped2,
        repo_root=str(tmp_path),
        closure_strictness="pedantic",
    )
    assert "val" in sc_ped_blocked["conditional_outputs"]
    sc_ped_allowed = analyze_unit_variable_scope(
        u_ped1,
        u_ped2,
        repo_root=str(tmp_path),
        closure_strictness="pedantic",
        non_suppressing_ctx_managers=["CustomAuditScope"],
    )
    assert "val" in sc_ped_allowed["definite_stores"]
    assert "val" not in sc_ped_allowed["conditional_outputs"]

    patch_ped = generate_refactoring_patch(
        [(0.99, u_ped1, u_ped2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="pedantic",
        non_suppressing_ctx_managers=["CustomAuditScope"],
    )
    assert "+    def _shared_" in patch_ped

    # 4. Enclosing global read with unshadowed pure builtin call is safe; impure call is rejected
    global_pure_src = (
        "GLOBAL_ITEMS = [1, 2, 3]\n"
        "def f_pure(scale):\n"
        "    global GLOBAL_ITEMS\n"
        "    n = len(GLOBAL_ITEMS) * scale\n"
        "    return n\n"
        "\n"
        "def f_impure(scale):\n"
        "    global GLOBAL_ITEMS\n"
        "    n = custom_mutator(GLOBAL_ITEMS) * scale\n"
        "    return n\n"
    )
    gp_tree = ast.parse(global_pure_src)
    sc_pure = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": global_pure_src,
            "source_code": "    n = len(GLOBAL_ITEMS) * scale\n",
        },
        tree=gp_tree,
    )
    assert "severed_closure_cell" not in sc_pure["control_flow_hazards"]

    sc_impure = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "basic_block",
            "source_text": global_pure_src,
            "source_code": "    n = custom_mutator(GLOBAL_ITEMS) * scale\n",
        },
        tree=gp_tree,
    )
    assert "severed_closure_cell" in sc_impure["control_flow_hazards"]
    assert "global_escape" in sc_impure["closure_hazard_reasons"]

    # 5. _is_ordered_contiguous_source_match rejects partial identifier/literal substring matches
    assert not _is_ordered_contiguous_source_match(["a = 1"], ["data = 100"])
    assert not _is_ordered_contiguous_source_match(["a = 1"], ["a = 10"])
    assert not _is_ordered_contiguous_source_match(["a = 1"], ["a = 1 + 2"])
    assert _is_ordered_contiguous_source_match(["a = 1"], ["x = 0; a = 1; y = 2"])
    assert _is_ordered_contiguous_source_match(["(a + b)"], ["res = (a + b) * 2"])

    # 6. _slice_unit_token_lines honors explicit is_pre_sliced=False even on column-0 expressions
    col0_lines = ["(a + b) + c\n"]
    sliced_col0 = _slice_unit_token_lines(
        {
            "start": 1,
            "end": 1,
            "start_col": 0,
            "end_col": 7,
            "kind": "complex_expr",
        },
        col0_lines,
        is_pre_sliced=False,
    )
    assert sliced_col0 == ["(a + b)"]
    presliced_col0 = _slice_unit_token_lines(
        {
            "start": 1,
            "end": 1,
            "start_col": 0,
            "end_col": 7,
            "kind": "complex_expr",
            "source_lines_is_col_sliced": True,
        },
        ["(a + b)\n"],
    )
    assert presliced_col0 == ["(a + b)\n"]


def test_granular_closure_hazard_skip_reasons_and_symtable_fuzzer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies granular closure_hazard skip reasons in patch generation and symtable consistency across generated AST programs."""
    import logging
    import random
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _extract_scope_local_names,
    )
    from pydoppelgangerhunt.fixer.scope import SCOPE_SCHEMA_VERSION
    from tests.fixer._closure_test_helpers import _extract_symtable_scope_info

    # 1. Unavailable source placeholder sets unavailable_source_placeholder
    sc_missing = inspect_single_unit_scope(
        {
            "file": str(tmp_path / "missing.py"),
            "start": 1,
            "end": 2,
            "kind": "subroutine",
        }
    )
    assert "severed_closure_cell" in sc_missing["control_flow_hazards"]
    assert "unavailable_source_placeholder" in sc_missing["closure_hazard_reasons"]

    # 2. Patch generation logs specific closure_hazard sub-reason in skip summary
    hazard_src = (
        "GLOBAL_STATE = [1, 2]\n"
        "def f1(scale):\n"
        "    global GLOBAL_STATE\n"
        "    n = custom_mutator(GLOBAL_STATE) * scale\n"
        "    m = n + 1\n"
        "    return m\n"
        "\n"
        "def f2(scale):\n"
        "    global GLOBAL_STATE\n"
        "    n = custom_mutator(GLOBAL_STATE) * scale\n"
        "    m = n + 1\n"
        "    return m\n"
    )
    hf = tmp_path / "haz.py"
    hf.write_text(hazard_src, encoding="utf-8")
    uh1 = {
        "file": "haz.py",
        "start": 4,
        "end": 5,
        "name": "f1_b",
        "kind": "compound_block",
    }
    uh2 = {
        "file": "haz.py",
        "start": 10,
        "end": 11,
        "name": "f2_b",
        "kind": "compound_block",
    }
    with caplog.at_level(logging.DEBUG):
        patch_out = generate_refactoring_patch(
            [(0.99, uh1, uh2)], repo_root=str(tmp_path), replace_clones=True
        )
    assert patch_out == ""
    assert any(
        rec.levelno == logging.INFO and "1 severed closure cell" in rec.message
        and "severed closure cell (" not in rec.message
        for rec in caplog.records
    )
    assert any(
        rec.levelno == logging.DEBUG
        and "severed closure cell (global escape)" in rec.message
        for rec in caplog.records
    )

    # 3. Randomized variable-permutation table test cross-checking scope local extraction against Python's symtable
    rng = random.Random(20261107)
    var_pool = ["v0", "v1", "v2", "v3", "v4"]
    for idx in range(25):
        p1, p2, loc1, loc2, g1 = rng.sample(var_pool, 5)
        lines = [
            f"def generated_{idx}({p1}, {p2}):",
            f"    global {g1}",
            f"    {loc1} = {p1} + 1",
            f"    if {p1} > 0:",
            f"        {loc2} = [(w_leaked := z + {loc1}) for z in ({p1}, {p2})]",
            "    else:",
            "        with suppress(ValueError):",
            f"            {loc2} = [{p2}]",
            "    def nested_cb(arg_n):",
            f"        return arg_n + {loc1} + {g1}",
            f"    return nested_cb({p2}), {loc2}",
        ]
        prog_src = "\n".join(lines) + "\n"
        tree = ast.parse(prog_src)
        sym_map = _extract_symtable_scope_info(prog_src)
        assert sym_map is not None
        outer_fn = tree.body[0]
        assert isinstance(outer_fn, ast.FunctionDef)
        ast_locals = _extract_scope_local_names(outer_fn, module_tree=tree)
        sym_outer = sym_map[(f"generated_{idx}", 1)]
        sym_nested = sym_map[("nested_cb", 9)]
        # Exclude PEP 709 inlined comprehension loop target 'z' while keeping PEP 572 walrus 'w_leaked'
        expected_locals = (
            (sym_outer["locals"] | sym_outer["parameters"])
            - sym_outer["globals"]
            - {"z"}
        )
        assert expected_locals <= ast_locals
        assert "w_leaked" in ast_locals
        assert "z" not in ast_locals
        assert loc1 in sym_outer["cells"]
        assert sym_nested["frees"] == frozenset({loc1})
        assert g1 not in (sym_outer["locals"] - sym_outer["globals"])
        info_cached = _analyze_enclosing_fn_cells_uncached(
            outer_fn, _SpanBounds(4, 8), mod_tree=tree, use_stmt_deltas=True
        )
        info_uncached = _analyze_enclosing_fn_cells_uncached(
            outer_fn, _SpanBounds(4, 8), mod_tree=tree, use_stmt_deltas=False
        )
        assert info_cached == info_uncached
        unit_sc = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 8,
                "kind": "compound_block",
                "source_text": prog_src,
            },
            tree=tree,
        )
        assert unit_sc["scope_schema_version"] == SCOPE_SCHEMA_VERSION


def test_unit_closure_strictness_override_and_ctx_manager_merging() -> None:
    """Verifies unit closure_strictness does not override explicit mode requests and non_suppressing_ctx_managers merges consistently."""
    from pydoppelgangerhunt.fixer.dataflow import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        scope_matches_closure_mode,
    )
    from pydoppelgangerhunt.fixer.synthesis import (
        synthesize_shared_helper_code,
    )

    src = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    cb = lambda: x\n"
        "    with CustomCtx(), UnitCtx():\n"
        "        x = a + 2\n"
        "        y = x * 3\n"
        "    return cb(), x, y\n"
    )
    tree = ast.parse(src)
    unit_with_strictness = {
        "start": 4,
        "end": 6,
        "kind": "compound_block",
        "source_text": src,
        "closure_strictness": "pedantic",
        "non_suppressing_ctx_managers": ["UnitCtx"],
    }

    # Explicit mode flags passed to ensure_unit_scope_closure_mode must override unit["closure_strictness"]
    sc_lenient = ensure_unit_scope_closure_mode(
        unit_with_strictness,
        None,
        True,
        strict_with_suppression=False,
        non_suppressing_ctx_managers=["GlobalCtx"],
        tree=tree,
    )
    assert sc_lenient["skip_pre_unit_closures"] is True
    assert sc_lenient["strict_with_suppression"] is False
    assert sc_lenient["non_suppressing_ctx_managers"] == [
        "GlobalCtx",
        "UnitCtx",
    ]
    assert scope_matches_closure_mode(
        sc_lenient,
        True,
        strict_with_suppression=False,
        non_suppressing_ctx_managers=["GlobalCtx", "UnitCtx"],
    )

    # Precomputed scope with _lhs_pruned and merged managers must be preserved without recomputation
    sc_pedantic = ensure_unit_scope_closure_mode(
        unit_with_strictness,
        None,
        False,
        strict_with_suppression=True,
        non_suppressing_ctx_managers=["CustomCtx"],
        tree=tree,
    )
    assert sc_pedantic["strict_with_suppression"] is True
    assert sc_pedantic["non_suppressing_ctx_managers"] == [
        "CustomCtx",
        "UnitCtx",
    ]
    assert "y" in sc_pedantic["definite_stores"]

    sc_marked = {**sc_pedantic, "_lhs_pruned": True}
    sc_rechecked = ensure_unit_scope_closure_mode(
        unit_with_strictness,
        sc_marked,
        False,
        strict_with_suppression=True,
        non_suppressing_ctx_managers=["CustomCtx"],
        tree=tree,
    )
    assert sc_rechecked is sc_marked
    assert sc_rechecked.get("_lhs_pruned") is True

    # Pairing and synthesis also merge global + unit non_suppressing_ctx_managers
    u1 = dict(unit_with_strictness, file="a.py", name="b1")
    u2 = dict(unit_with_strictness, file="b.py", name="b2")
    paired = resolve_clone_pair_outputs(
        u1,
        u2,
        sc_marked,
        dict(sc_marked),
        source_text1=src,
        source_text2=src,
        tree1=tree,
        tree2=tree,
        closure_strictness="pedantic",
        non_suppressing_ctx_managers=["CustomCtx"],
    )
    assert paired == (["x", "y"], ["x", "y"])

    helper_code = synthesize_shared_helper_code(
        u1,
        u2,
        scope1=sc_marked,
        scope2=dict(sc_marked),
        tree1=tree,
        tree2=tree,
        closure_strictness="pedantic",
        non_suppressing_ctx_managers=["CustomCtx"],
    )
    assert "with CustomCtx(), UnitCtx():" in helper_code


def test_pedantic_expr_unit_lhs_pruning_preserved_during_synthesis() -> None:
    """Verifies pedantic synthesize_shared_helper_code preserves LHS-pruned expression scopes."""
    from pydoppelgangerhunt.fixer.synthesis import (
        synthesize_shared_helper_code,
    )

    expr_mod_src = (
        "def run_a(items, offset):\n"
        "    res = 99\n"
        "    res = [x + offset for x in items if x > 0]\n"
        "    return res\n"
        "\n"
        "def run_b(items, offset):\n"
        "    res = 99\n"
        "    res = [x + offset for x in items if x > 0]\n"
        "    return res\n"
    )
    u1_expr = {
        "name": "comp_a",
        "kind": "comprehension",
        "start": 3,
        "end": 3,
        "source_text": expr_mod_src,
        "source_code": "    res = [x + offset for x in items if x > 0]\n",
    }
    u2_expr = {
        "name": "comp_b",
        "kind": "comprehension",
        "start": 8,
        "end": 8,
        "source_text": expr_mod_src,
        "source_code": "    res = [x + offset for x in items if x > 0]\n",
    }
    helper_ped = synthesize_shared_helper_code(
        u1_expr, u2_expr, closure_strictness="pedantic"
    )
    assert "def _shared_comp_a_comp_b(" in helper_ped
    assert "res" not in helper_ped.split("):", 1)[0]


def test_per_unit_closure_strictness_honored_when_top_level_none() -> None:
    """Verifies per-unit closure_strictness is honored when top-level closure_strictness is None."""
    from pydoppelgangerhunt.fixer.dataflow import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer.scope import (
        _inspect_unit_scope,
        analyze_unit_variable_scope,
    )

    unk_ctx_src = (
        "def fn_a(items, custom_mgr):\n"
        "    with custom_mgr():\n"
        "        val = sum(items)\n"
        "    step = lambda x: x + val\n"
        "    out = step(1)\n"
        "    return out\n"
    )
    u1_ped_unit = {
        "name": "sub_a",
        "kind": "block",
        "start": 4,
        "end": 5,
        "source_text": unk_ctx_src,
        "closure_strictness": "pedantic",
    }
    u2_ped_unit = dict(u1_ped_unit, name="sub_b")
    sc_ped = _inspect_unit_scope(u1_ped_unit)
    assert sc_ped["strict_with_suppression"] is True
    assert "severed_closure_cell" in sc_ped["control_flow_hazards"]
    var_info_ped = analyze_unit_variable_scope(u1_ped_unit, u2_ped_unit)
    assert var_info_ped["is_control_flow_safe"] is False

    u1_strict_unit = dict(u1_ped_unit, closure_strictness="strict")
    sc_strict = _inspect_unit_scope(u1_strict_unit)
    assert sc_strict["strict_with_suppression"] is False
    assert "severed_closure_cell" not in sc_strict["control_flow_hazards"]
    res_ped = resolve_clone_pair_outputs(u1_ped_unit, u2_ped_unit, sc_strict, sc_strict)
    assert res_ped is None


def test_neutral_named_suppressor_strict_vs_pedantic_and_summary_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies neutral-named suppressors in strict vs pedantic mode and end-of-run summary logging."""
    import logging
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        log_heuristic_ctx_manager_summary,
    )

    neutral_src = (
        "def run_neutral(compute):\n"
        "    with Guard(), SpanContext(), Transaction():\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    neutral_tree = ast.parse(neutral_src)
    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        sc_strict = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": neutral_src,
                "source_code": "    cb = lambda: result\n",
                "closure_strictness": "strict",
            },
            tree=neutral_tree,
        )
    assert "severed_closure_cell" not in sc_strict["control_flow_hazards"]
    assert any(
        "unproven context manager" in rec.message.lower()
        and "closure_strictness='pedantic'" in rec.message
        for rec in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        recorded = log_heuristic_ctx_manager_summary()
    assert set(recorded) == {"Guard", "SpanContext", "Transaction"}
    assert any(
        "3 unproven context manager(s)" in rec.message
        and "were treated as non-suppressing" in rec.message
        for rec in caplog.records
    )

    suspicious_src = (
        "def run_suspicious(compute):\n"
        "    with Swallow(), tolerate():\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    sc_suspicious = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": suspicious_src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(suspicious_src),
    )
    assert "severed_closure_cell" in sc_suspicious["control_flow_hazards"]

    sc_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": neutral_src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "pedantic",
        },
        tree=neutral_tree,
    )
    assert "severed_closure_cell" in sc_pedantic["control_flow_hazards"]


def test_synthesize_refactoring_suggestion_unconditional_delegation_and_hazard_warning(
    tmp_path: Any,
) -> None:
    """Verifies synthesize_refactoring_suggestion delegates with default args and emits a hazard warning when rejected."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    safe_src = (
        "def calc_a(x, y):\n"
        "    step = x * 2 + y\n"
        "    out = step + 10\n"
        "    return out\n\n"
        "def calc_b(x, y):\n"
        "    step = x * 2 + y\n"
        "    out = step + 10\n"
        "    return out\n"
    )
    safe_file = tmp_path / "safe_mod.py"
    safe_file.write_text(safe_src, encoding="utf-8")
    u_safe_1 = {
        "name": "calc_a:block",
        "file": str(safe_file),
        "start": 2,
        "end": 3,
        "kind": "compound_block",
        "source_text": safe_src,
    }
    u_safe_2 = {
        "name": "calc_b:block",
        "file": str(safe_file),
        "start": 7,
        "end": 8,
        "kind": "compound_block",
        "source_text": safe_src,
    }
    sug_default = synthesize_refactoring_suggestion(
        u_safe_1, u_safe_2, repo_root=str(tmp_path)
    )
    assert "def _shared_block(" in sug_default
    assert "return step, out" in sug_default
    assert "(*args, **kwargs)" not in sug_default

    hazard_src = (
        "def run_a(items):\n"
        "    scale = 2\n"
        "    cbs = [lambda v: v * scale for _ in items]\n"
        "    out = len(cbs)\n"
        "    scale = 10\n"
        "    return cbs[0](out)\n\n"
        "def run_b(items):\n"
        "    scale = 2\n"
        "    cbs = [lambda v: v * scale for _ in items]\n"
        "    out = len(cbs)\n"
        "    scale = 10\n"
        "    return cbs[0](out)\n"
    )
    hazard_file = tmp_path / "hazard_mod.py"
    hazard_file.write_text(hazard_src, encoding="utf-8")
    u_haz_1 = {
        "name": "run_a:block",
        "file": str(hazard_file),
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": hazard_src,
    }
    u_haz_2 = {
        "name": "run_b:block",
        "file": str(hazard_file),
        "start": 10,
        "end": 11,
        "kind": "compound_block",
        "source_text": hazard_src,
    }
    sug_hazard = synthesize_refactoring_suggestion(
        u_haz_1, u_haz_2, repo_root=str(tmp_path)
    )
    assert "rejected as unsafe" in sug_hazard
    assert "WARNING: Automatic helper synthesis was skipped" in sug_hazard
    assert "(*args, **kwargs)" not in sug_hazard


def test_resolve_unit_and_pair_closure_flags_conflict_warning_and_explicit_override(
    caplog: Any,
) -> None:
    """Verifies resolve_unit_closure_flags logs conflict warnings and resolve_pair_closure_flags honors explicit overrides."""
    import logging
    from pydoppelgangerhunt.config import (
        resolve_pair_closure_flags,
        resolve_unit_closure_flags,
    )

    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        skip_flag, strict_with_flag = resolve_unit_closure_flags(
            {"start": 1, "end": 2},
            skip_pre_unit_closures=True,
            strict_with_suppression=True,
        )
    assert (skip_flag, strict_with_flag) == (True, False)
    assert any("Conflicting configuration" in rec.message for rec in caplog.records)

    u1 = {"start": 1, "end": 2, "skip_pre_unit_closures": True}
    u2 = {"start": 3, "end": 4, "skip_pre_unit_closures": False}
    preserved = resolve_pair_closure_flags(u1, u2)
    assert preserved.unit1.skip_pre_unit_closures is True
    assert preserved.unit2.skip_pre_unit_closures is False

    overridden = resolve_pair_closure_flags(u1, u2, skip_pre_unit_closures=False)
    assert overridden.unit1.skip_pre_unit_closures is False
    assert overridden.unit2.skip_pre_unit_closures is False
    assert overridden.effective_skip_pre_unit_closures is False


def test_unit_level_closure_strictness_lenient_survives_generate_refactoring_patch(
    tmp_path: Any,
) -> None:
    """Verifies unit-level closure_strictness='lenient' is preserved by generate_refactoring_patch and synthesize_refactoring_suggestion when no explicit override is passed."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    src = (
        "def gen_a(items):\n"
        "    pre_cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return len(items)\n\n"
        "def gen_b(items):\n"
        "    pre_cb = lambda: step\n"
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
        "    return len(items)\n"
    )
    target_file = tmp_path / "unit_lenient.py"
    target_file.write_text(src, encoding="utf-8")
    unit_code = "    for x in items:\n" "        step = x + 1\n" "        yield step\n"
    u1_strict: dict[str, Any] = {
        "file": str(target_file),
        "start": 3,
        "end": 5,
        "name": "gen_a:For",
        "kind": "compound_block",
        "source_text": src,
        "source_code": unit_code,
    }
    u2_strict: dict[str, Any] = {
        "file": str(target_file),
        "start": 10,
        "end": 12,
        "name": "gen_b:For",
        "kind": "compound_block",
        "source_text": src,
        "source_code": unit_code,
    }
    patch_default = generate_refactoring_patch(
        [(1.0, u1_strict, u2_strict)],
        repo_root=str(tmp_path),
    )
    assert patch_default == ""

    u1_lenient = {**u1_strict, "closure_strictness": "lenient"}
    u2_lenient = {**u2_strict, "closure_strictness": "lenient"}
    patch_lenient = generate_refactoring_patch(
        [(1.0, u1_lenient, u2_lenient)],
        repo_root=str(tmp_path),
    )
    assert "def _shared_" in patch_lenient

    sug_lenient = synthesize_refactoring_suggestion(
        u1_lenient,
        u2_lenient,
        repo_root=str(tmp_path),
    )
    assert "def _shared_" in sug_lenient


def test_synthesize_refactoring_suggestion_contains_helper_synthesis_exception(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies synthesize_refactoring_suggestion catches unexpected exceptions from synthesize_shared_helper_code and returns a safe warning comment."""
    import pydoppelgangerhunt.fixer.synthesis as synth_mod
    import pydoppelgangerhunt.reporters as reporters_mod

    src = (
        "def f1(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n\n"
        "def f2(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n"
    )
    target_file = tmp_path / "synth_exc.py"
    target_file.write_text(src, encoding="utf-8")
    u1 = {
        "file": str(target_file),
        "start": 2,
        "end": 3,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    b = a + 1\n    c = b * 2\n",
    }
    u2 = {
        "file": str(target_file),
        "start": 7,
        "end": 8,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    b = a + 1\n    c = b * 2\n",
    }

    def _expected_syntax_failure(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("simulated malformed helper syntax")

    monkeypatch.setattr(synth_mod, "synthesize_shared_helper_code", _expected_syntax_failure)
    suggestion = reporters_mod.synthesize_refactoring_suggestion(
        u1, u2, repo_root=str(tmp_path)
    )
    assert "skipped as structurally incompatible" in suggestion
    assert "WARNING: Automatic helper synthesis was skipped" in suggestion

    def _unexpected_bug(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("unexpected internal defect")

    monkeypatch.setattr(synth_mod, "synthesize_shared_helper_code", _unexpected_bug)
    with pytest.raises(RuntimeError, match="unexpected internal defect"):
        reporters_mod.synthesize_refactoring_suggestion(
            u1, u2, repo_root=str(tmp_path)
        )


@pytest.mark.parametrize(
    "unit_dict,strictness,skip_closures,strict_with,expected",
    [
        ({}, None, None, None, (False, False)),
        ({}, "strict", None, None, (False, False)),
        ({}, "pedantic", None, None, (False, True)),
        ({}, "lenient", None, None, (True, False)),
        ({}, "pedantic", True, None, (False, True)),
        ({}, "lenient", None, True, (True, False)),
        ({"closure_strictness": "pedantic"}, None, None, None, (False, True)),
        ({"closure_strictness": "lenient"}, None, None, None, (True, False)),
        ({"closure_strictness": "lenient"}, "pedantic", None, None, (False, True)),
        ({"skip_pre_unit_closures": True}, None, None, None, (True, False)),
        ({"strict_with_suppression": True}, None, None, None, (False, True)),
        (
            {"closure_strictness": "pedantic", "skip_pre_unit_closures": True},
            None,
            None,
            None,
            (False, True),
        ),
        (
            {"skip_pre_unit_closures": True},
            None,
            False,
            True,
            (False, True),
        ),
        (
            {"strict_with_suppression": True},
            None,
            False,
            None,
            (False, True),
        ),
        (
            {"skip_pre_unit_closures": True},
            None,
            None,
            False,
            (True, False),
        ),
    ],
)
def test_resolve_unit_closure_flags_truth_table(
    unit_dict: dict[str, Any],
    strictness: Any,
    skip_closures: Any,
    strict_with: Any,
    expected: tuple[bool, bool],
) -> None:
    """Verifies resolve_unit_closure_flags 3-tier precedence truth table."""
    from pydoppelgangerhunt.config import resolve_unit_closure_flags

    base_unit = {"start": 1, "end": 2, **unit_dict}
    assert (
        resolve_unit_closure_flags(
            base_unit,
            closure_strictness=strictness,
            skip_pre_unit_closures=skip_closures,
            strict_with_suppression=strict_with,
        )
        == expected
    )


def test_synthesize_shared_helper_code_does_not_mutate_unit_allowlist_across_calls() -> (
    None
):
    """Verifies synthesize_shared_helper_code does not mutate caller u1/u2 dictionaries with transient allowlists."""
    import copy

    src = (
        "def f1(flag):\n"
        "    with custom_cm():\n"
        "        if flag:\n"
        "            raise ValueError('boom')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n\n"
        "def f2(flag):\n"
        "    with custom_cm():\n"
        "        if flag:\n"
        "            raise ValueError('boom')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n"
    )
    u1: dict[str, Any] = {
        "name": "f1:block",
        "file": "mod.py",
        "start": 6,
        "end": 7,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    cb = lambda: x\n    out = cb() + 1\n",
    }
    u2: dict[str, Any] = {
        "name": "f2:block",
        "file": "mod.py",
        "start": 15,
        "end": 16,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    cb = lambda: x\n    out = cb() + 1\n",
    }
    u1_before = copy.deepcopy(u1)
    u2_before = copy.deepcopy(u2)

    allowed_helper = synthesize_shared_helper_code(
        u1,
        u2,
        closure_strictness="pedantic",
        non_suppressing_ctx_managers=["custom_cm"],
    )
    assert "def _shared_" in allowed_helper
    assert u1 == u1_before
    assert u2 == u2_before
    assert "non_suppressing_ctx_managers" not in u1
    assert "non_suppressing_ctx_managers" not in u2

    rejected_helper = synthesize_shared_helper_code(
        u1,
        u2,
        closure_strictness="pedantic",
    )
    assert rejected_helper == ""
    sc1_pedantic = inspect_single_unit_scope(
        {**u1, "closure_strictness": "pedantic"},
    )
    assert "severed_closure_cell" in sc1_pedantic["control_flow_hazards"]


def test_heuristic_ctx_manager_summary_logged_once_at_batch_boundary(
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies per-pair helpers do not flush heuristic summary mid-run and batch entrypoints log one consolidated summary."""
    import logging
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        log_heuristic_ctx_manager_summary,
    )
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    src_a = (
        "def fa1(x):\n"
        "    with CustomMgrAlpha():\n"
        "        v = x + 1\n"
        "    out = v * 2\n"
        "    return out\n\n"
        "def fa2(x):\n"
        "    with CustomMgrAlpha():\n"
        "        v = x + 1\n"
        "    out = v * 2\n"
        "    return out\n"
    )
    src_b = (
        "def fb1(x):\n"
        "    with CustomMgrBeta():\n"
        "        v = x + 3\n"
        "    out = v * 4\n"
        "    return out\n\n"
        "def fb2(x):\n"
        "    with CustomMgrBeta():\n"
        "        v = x + 3\n"
        "    out = v * 4\n"
        "    return out\n"
    )
    file_a = tmp_path / "mod_a.py"
    file_b = tmp_path / "mod_b.py"
    file_a.write_text(src_a, encoding="utf-8")
    file_b.write_text(src_b, encoding="utf-8")

    ua1 = {
        "name": "fa1:block",
        "file": str(file_a),
        "start": 2,
        "end": 4,
        "kind": "compound_block",
        "source_text": src_a,
    }
    ua2 = {
        "name": "fa2:block",
        "file": str(file_a),
        "start": 8,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_a,
    }
    ub1 = {
        "name": "fb1:block",
        "file": str(file_b),
        "start": 2,
        "end": 4,
        "kind": "compound_block",
        "source_text": src_b,
    }
    ub2 = {
        "name": "fb2:block",
        "file": str(file_b),
        "start": 8,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_b,
    }

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        synthesize_refactoring_suggestion(ua1, ua2, repo_root=str(tmp_path))
        synthesize_refactoring_suggestion(ub1, ub2, repo_root=str(tmp_path))
        assert not [r for r in caplog.records if "Summary:" in r.message]
        flushed = log_heuristic_ctx_manager_summary()
    assert set(flushed) == {"CustomMgrAlpha", "CustomMgrBeta"}
    summary_records = [r for r in caplog.records if "Summary:" in r.message]
    assert len(summary_records) == 1
    assert "2 unproven context manager(s)" in summary_records[0].message

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        patch = generate_refactoring_patch(
            [(1.0, dict(ua1), dict(ua2)), (1.0, dict(ub1), dict(ub2))],
            repo_root=str(tmp_path),
        )
    assert "def _shared_" in patch
    patch_summaries = [r for r in caplog.records if "Summary:" in r.message]
    assert len(patch_summaries) == 1
    assert "2 unproven context manager(s)" in patch_summaries[0].message


@pytest.mark.parametrize("ctrl_stmt", ["break", "continue"])
def test_synthesize_refactoring_suggestion_rejects_naked_break_and_continue_helpers(
    ctrl_stmt: str,
) -> None:
    """Verifies synthesize_refactoring_suggestion compiles helpers and emits unsafe-extraction diagnostics on naked break/continue."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    src = (
        "def f1(items):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        f"            {ctrl_stmt}\n"
        "        y = x + 1\n"
        "    return 0\n\n"
        "def f2(items):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        f"            {ctrl_stmt}\n"
        "        y = x + 1\n"
        "    return 0\n"
    )
    u1 = {
        "name": "f1:block",
        "file": "loop_mod.py",
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "source_text": src,
        "source_code": f"        if x < 0:\n            {ctrl_stmt}\n        y = x + 1\n",
    }
    u2 = {
        "name": "f2:block",
        "file": "loop_mod.py",
        "start": 10,
        "end": 12,
        "kind": "compound_block",
        "source_text": src,
        "source_code": f"        if x < 0:\n            {ctrl_stmt}\n        y = x + 1\n",
    }
    suggestion = synthesize_refactoring_suggestion(u1, u2)
    assert "rejected as unsafe" in suggestion
    assert "def _shared_" not in suggestion


def test_unit_level_closure_strictness_downstream_alias_and_mixed_mode_pair_symmetry(
    tmp_path: Any,
) -> None:
    """Verifies unit-level closure_strictness is honored in downstream alias scans and mixed-mode pair entrypoints."""
    if sys.version_info >= (3, 12):
        alias_src = (
            "def run_alias(x):\n"
            "    cb = lambda: AliasT\n"
            "    type AliasT = int\n"
            "    val = x + 1\n"
            "    return val\n"
        )
        alias_tree = ast.parse(alias_src)
        u_lenient_alias = {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": alias_src,
            "source_code": "    type AliasT = int\n    val = x + 1\n",
            "closure_strictness": "lenient",
        }
        u_strict_alias = {
            **u_lenient_alias,
            "closure_strictness": "strict",
        }
        sc_lenient = inspect_single_unit_scope(u_lenient_alias, tree=alias_tree)
        sc_strict = inspect_single_unit_scope(u_strict_alias, tree=alias_tree)
        assert "AliasT" not in sc_lenient["outputs"]
        assert "AliasT" in sc_strict["outputs"]

    pair_src = (
        "def f1(flag):\n"
        "    with UnknownMgr():\n"
        "        if flag:\n"
        "            raise ValueError('err')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n\n"
        "def f2(flag):\n"
        "    with UnknownMgr():\n"
        "        if flag:\n"
        "            raise ValueError('err')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n"
    )
    mod_file = tmp_path / "mixed_pair.py"
    mod_file.write_text(pair_src, encoding="utf-8")
    u1_strict = {
        "name": "f1:block",
        "file": str(mod_file),
        "start": 6,
        "end": 7,
        "kind": "compound_block",
        "source_text": pair_src,
        "source_code": "    cb = lambda: x\n    out = cb() + 1\n",
        "closure_strictness": "strict",
    }
    u2_pedantic = {
        "name": "f2:block",
        "file": str(mod_file),
        "start": 15,
        "end": 16,
        "kind": "compound_block",
        "source_text": pair_src,
        "source_code": "    cb = lambda: x\n    out = cb() + 1\n",
        "closure_strictness": "pedantic",
    }
    comb_sc = analyze_unit_variable_scope(u1_strict, u2_pedantic)
    assert comb_sc["is_control_flow_safe"] is False
    assert "severed_closure_cell" in comb_sc["control_flow_hazards"]

    assert synthesize_shared_helper_code(u1_strict, u2_pedantic) == ""
    assert (
        generate_refactoring_patch(
            [(1.0, u1_strict, u2_pedantic)], repo_root=str(tmp_path)
        )
        == ""
    )


def test_cached_heuristic_ctx_manager_replay_and_per_hunk_patch_comment(
    tmp_path: Path,
) -> None:
    """Verifies cached delta/scope replay of heuristic managers and per-pair patch comment scoping."""
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        format_heuristic_ctx_manager_comment,
        peek_heuristic_non_suppressing_ctx_managers,
    )
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    src = (
        "def f1(compute, a):\n"
        "    with CustomTx():\n"
        "        x = compute(a)\n"
        "        y = x + 1\n"
        "    cb = lambda: y\n"
        "    return cb(), x\n\n"
        "def f2(compute, a):\n"
        "    with CustomTx():\n"
        "        x = compute(a)\n"
        "        y = x + 1\n"
        "    cb = lambda: y\n"
        "    return cb(), x\n\n"
        "def clean1(b):\n"
        "    z = b * 2\n"
        "    w = z + 3\n"
        "    return w\n\n"
        "def clean2(b):\n"
        "    z = b * 2\n"
        "    w = z + 3\n"
        "    return w\n"
    )
    tree = ast.parse(src)
    u_tx = {
        "start": 5,
        "end": 5,
        "kind": "basic_block",
        "source_text": src,
        "source_code": "    cb = lambda: y\n",
        "closure_strictness": "strict",
    }
    sc1 = inspect_single_unit_scope(u_tx, tree=tree)
    assert sc1["heuristic_ctx_managers"] == ["CustomTx"]
    assert "CustomTx" in peek_heuristic_non_suppressing_ctx_managers()

    # Clearing warning cache WITHOUT clearing AST/dataflow cache still replays CustomTx on warm cache hit
    clear_closure_warning_cache()
    assert peek_heuristic_non_suppressing_ctx_managers() == ()
    sc2 = inspect_single_unit_scope(u_tx, tree=tree)
    assert sc2["heuristic_ctx_managers"] == ["CustomTx"]
    assert "CustomTx" in peek_heuristic_non_suppressing_ctx_managers()

    # In generate_refactoring_patch, if the pair with CustomTx is rejected (e.g. output arity mismatch)
    # and only the clean pair is emitted, the patch must NOT contain the heuristic advisory comment.
    mod_file = tmp_path / "mod_heur.py"
    mod_file.write_text(src, encoding="utf-8")
    u_rej_1 = {
        "file": str(mod_file),
        "start": 2,
        "end": 4,
        "name": "f1:tx",
        "kind": "compound_block",
        "outputs": ["x"],
    }
    u_rej_2 = {
        "file": str(mod_file),
        "start": 9,
        "end": 11,
        "name": "f2:tx",
        "kind": "compound_block",
        "outputs": ["x", "y"],
    }
    u_clean_1 = {
        "file": str(mod_file),
        "start": 16,
        "end": 17,
        "name": "clean1:blk",
        "kind": "compound_block",
    }
    u_clean_2 = {
        "file": str(mod_file),
        "start": 21,
        "end": 22,
        "name": "clean2:blk",
        "kind": "compound_block",
    }
    patch_clean_only = generate_refactoring_patch(
        [(1.0, u_rej_1, u_rej_2), (1.0, u_clean_1, u_clean_2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "_shared_" in patch_clean_only
    assert format_heuristic_ctx_manager_comment(["CustomTx"]) not in patch_clean_only


@pytest.mark.parametrize(
    ("mode", "allowlist", "expect_safe"),
    [
        ("strict", (), True),
        ("pedantic", (), False),
        ("pedantic", ("NeutralTx",), True),
        ("lenient", (), True),
    ],
)
def test_public_entrypoints_configuration_consistency_matrix(
    tmp_path: Path,
    mode: str,
    allowlist: tuple[str, ...],
    expect_safe: bool,
) -> None:
    """Verifies all public fixer entry points agree across closure_strictness and non_suppressing_ctx_managers."""
    from pydoppelgangerhunt import synthesize_refactoring_suggestion
    from pydoppelgangerhunt.fixer import (
        analyze_unit_variable_scope,
        generate_refactoring_patch,
        inspect_single_unit_scope,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
        synthesize_shared_helper_code,
    )

    src = (
        "def work_a(items, compute):\n"
        "    with NeutralTx():\n"
        "        seed = compute(items)\n"
        "    cb = lambda x: x + seed\n"
        "    total = cb(10) + len(items)\n"
        "    return total\n\n"
        "def work_b(items, compute):\n"
        "    with NeutralTx():\n"
        "        seed = compute(items)\n"
        "    cb = lambda x: x + seed\n"
        "    total = cb(10) + len(items)\n"
        "    return total\n"
    )
    mod_file = tmp_path / "matrix_mod.py"
    mod_file.write_text(src, encoding="utf-8")
    tree = ast.parse(src)

    u1: dict[str, Any] = {
        "name": "work_a:block",
        "file": str(mod_file),
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    cb = lambda x: x + seed\n    total = cb(10) + len(items)\n",
        "closure_strictness": mode,
        "non_suppressing_ctx_managers": list(allowlist),
    }
    u2: dict[str, Any] = {
        "name": "work_b:block",
        "file": str(mod_file),
        "start": 11,
        "end": 12,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    cb = lambda x: x + seed\n    total = cb(10) + len(items)\n",
        "closure_strictness": mode,
        "non_suppressing_ctx_managers": list(allowlist),
    }

    sc1 = inspect_single_unit_scope(u1, tree=tree)
    sc2 = inspect_single_unit_scope(u2, tree=tree)
    assert sc1["is_control_flow_safe"] is expect_safe
    assert sc2["is_control_flow_safe"] is expect_safe

    comb = analyze_unit_variable_scope(u1, u2, tree1=tree, tree2=tree)
    assert comb["is_control_flow_safe"] is expect_safe

    pair_out = resolve_clone_pair_outputs(
        u1,
        u2,
        sc1,
        sc2,
        source_text1=src,
        source_text2=src,
        tree1=tree,
        tree2=tree,
        return_details=True,
    )
    if expect_safe:
        assert pair_out.outputs is not None
    else:
        assert pair_out.outputs is None

    gen_u1 = {**u1, "has_yield": True}
    gen_u2 = {**u2, "has_yield": True}
    gen_out = resolve_clone_generator_subroutine_outputs(
        gen_u1,
        gen_u2,
        sc1,
        sc2,
        source_text1=src,
        source_text2=src,
        tree1=tree,
        tree2=tree,
        return_details=True,
    )
    if expect_safe:
        assert gen_out.outputs is not None
    else:
        assert gen_out.outputs is None

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        helper_name="_shared_matrix_work",
        scope=comb,
        outputs=["total"],
        outputs2=["total"],
        source_text1=src,
        source_text2=src,
        tree1=tree,
        tree2=tree,
    )
    patch = generate_refactoring_patch(
        [(1.0, dict(u1), dict(u2))],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness=mode,
        non_suppressing_ctx_managers=list(allowlist),
    )
    suggestion = synthesize_refactoring_suggestion(dict(u1), dict(u2))
    if expect_safe:
        assert "def _shared_matrix_work(" in helper
        assert "_shared_" in patch
        assert "def _shared_" in suggestion
    else:
        assert helper == ""
        assert patch == ""
        assert "rejected as unsafe" in suggestion.lower()
        assert "severed closure cell" in suggestion.lower()


def test_docs_sync_with_suspicious_suppress_substrings_and_local_accumulator_strict_vs_pedantic() -> (
    None
):
    """Verifies README/CHANGELOG match _SUSPICIOUS_SUPPRESS_SUBSTRINGS, local primitive accumulators work in strict mode, and ephemeral AST tracking uses weakrefs."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _is_ephemeral_unit_ast,
        _mark_ephemeral_unit_ast,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SUSPICIOUS_SUPPRESS_SUBSTRINGS,
    )

    repo_root = Path(__file__).resolve().parents[2]
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    changelog_text = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert len(_SUSPICIOUS_SUPPRESS_SUBSTRINGS) == 34
    for sub in _SUSPICIOUS_SUPPRESS_SUBSTRINGS:
        assert f"`{sub}`" in readme_text, sub
        assert f"`{sub}`" in changelog_text, sub

    # Local primitive-initialized accumulator in a global-reading unit is permitted in strict mode and rejected in pedantic mode
    acc_src = (
        "SCALE = 3\n"
        "def sum_scaled(items):\n"
        "    global SCALE\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item * SCALE\n"
        "    return total\n"
    )
    acc_tree = ast.parse(acc_src)
    sc_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": acc_src,
            "closure_strictness": "strict",
        },
        tree=acc_tree,
    )
    assert sc_strict["is_control_flow_safe"] is True
    assert "global_escape" not in sc_strict["closure_hazard_reasons"]

    sc_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": acc_src,
            "closure_strictness": "pedantic",
        },
        tree=acc_tree,
    )
    assert sc_pedantic["is_control_flow_safe"] is False
    assert "global_escape" in sc_pedantic["closure_hazard_reasons"]

    # Ephemeral AST tracking uses weakref set rather than setting ad hoc attributes on ast.AST
    ephem_mod = ast.Module(body=[], type_ignores=[])
    assert _is_ephemeral_unit_ast(ephem_mod) is False
    _mark_ephemeral_unit_ast(ephem_mod)
    assert _is_ephemeral_unit_ast(ephem_mod) is True
    assert not hasattr(ephem_mod, "_is_ephemeral_unit_mod")


def test_concurrent_heuristic_ctx_manager_summary_per_thread_isolation() -> None:
    """Verifies concurrent threads have isolated heuristic context-manager summaries and defer depths."""
    import concurrent.futures
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        defer_heuristic_ctx_manager_summary,
        log_heuristic_ctx_manager_summary,
        peek_heuristic_non_suppressing_ctx_managers,
        record_heuristic_non_suppressing_ctx_manager,
    )

    clear_closure_warning_cache()

    def _thread_run(
        tid: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        mgr_a = f"ThreadMgrA_{tid}"
        mgr_b = f"ThreadMgrB_{tid}"
        with defer_heuristic_ctx_manager_summary():
            record_heuristic_non_suppressing_ctx_manager(mgr_b)
            record_heuristic_non_suppressing_ctx_manager(mgr_a)
            deferred_peek = log_heuristic_ctx_manager_summary(force=False)
            still_present = peek_heuristic_non_suppressing_ctx_managers()
        drained = log_heuristic_ctx_manager_summary(force=False)
        after_drain = peek_heuristic_non_suppressing_ctx_managers()
        assert after_drain == ()
        return deferred_peek, still_present, drained

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {tid: executor.submit(_thread_run, tid) for tid in range(16)}
        for tid, fut in futures.items():
            expected = (f"ThreadMgrA_{tid}", f"ThreadMgrB_{tid}")
            deferred_peek, still_present, drained = fut.result()
            assert deferred_peek == expected
            assert still_present == expected
            assert drained == expected


def test_monotone_unit_closure_flags_and_pyproject_strict_with_suppression(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies caller skip_pre_unit_closures=True cannot weaken unit pedantic mode and pyproject strict_with_suppression takes effect."""
    import logging
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        load_tool_config,
        resolve_unit_closure_flags,
    )

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        flags = resolve_unit_closure_flags(
            {"start": 1, "end": 2, "closure_strictness": "pedantic"},
            skip_pre_unit_closures=True,
        )
    assert flags == (False, True)
    assert any(
        "Conflicting configuration" in rec.message
        and "(--skip-pre-unit-closures)" not in rec.message
        for rec in caplog.records
    )

    # Strict unit policy also resists weakening by caller skip_pre_unit_closures=True
    flags_strict_unit = resolve_unit_closure_flags(
        {"start": 1, "end": 2, "closure_strictness": "strict"},
        skip_pre_unit_closures=True,
    )
    assert flags_strict_unit == (False, False)

    # pyproject.toml with strict_with_suppression = true (and no closure_strictness) activates pedantic without conflict
    proj_dir = tmp_path / "strict_with_proj"
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text(
        "[tool.pydoppelgangerhunt]\nstrict_with_suppression = true\n",
        encoding="utf-8",
    )
    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        cfg = load_tool_config(repo_root=str(proj_dir))
        eff_flags = resolve_unit_closure_flags(
            {"start": 1, "end": 2},
            closure_strictness=cfg.get("closure_strictness"),
            strict_with_suppression=cfg.get("strict_with_suppression"),
        )
    assert eff_flags == (False, True)
    assert not [r for r in caplog.records if "Conflicting configuration" in r.message]


def test_cli_text_report_clean_stdout_and_session_worker_heuristic_harvesting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies CLI text report keeps stdout/--output clean of '# Note:', qualifies manager names, and harvests worker threads."""
    import concurrent.futures
    import logging
    from pydoppelgangerhunt.cli import main
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        heuristic_ctx_manager_session,
        log_heuristic_ctx_manager_summary,
        record_heuristic_non_suppressing_ctx_manager,
    )

    repo = tmp_path / "cli_clean_repo"
    repo.mkdir()
    mod_src = (
        "def handle_order_a(db, self, items):\n"
        "    with db.atomic(), self.lock:\n"
        "        total = 0\n"
        "        for item in items:\n"
        "            if item > 0:\n"
        "                total += item * 2\n"
        "            else:\n"
        "                total -= item\n"
        "    cb = lambda: total\n"
        "    return cb()\n\n"
        "def handle_order_b(db, self, items):\n"
        "    with db.atomic(), self.lock:\n"
        "        total = 0\n"
        "        for item in items:\n"
        "            if item > 0:\n"
        "                total += item * 2\n"
        "            else:\n"
        "                total -= item\n"
        "    cb = lambda: total\n"
        "    return cb()\n"
    )
    (repo / "orders.py").write_text(mod_src, encoding="utf-8")
    out_report = tmp_path / "report.txt"

    # Seed a stale leftover before CLI session to verify session entry discards prior leftovers
    clear_closure_warning_cache()
    record_heuristic_non_suppressing_ctx_manager("StaleLeftoverMgr")
    capsys.readouterr()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt.config"):
        rc = main(
            [
                str(repo),
                "--min-lines",
                "6",
                "--suggest",
                "--sliding-window",
                "--output",
                str(out_report),
            ]
        )
    captured = capsys.readouterr()
    report_text = out_report.read_text(encoding="utf-8")
    assert rc in (0, 1)
    assert "# Note:" not in captured.out
    assert "unproven context manager" not in captured.out
    assert "# Note:" not in report_text
    assert "StaleLeftoverMgr" not in captured.err
    assert not any("StaleLeftoverMgr" in r.message for r in caplog.records)

    # Verify qualified manager names on direct scope inspection
    sc = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "basic_block",
            "source_text": mod_src,
            "source_code": "    cb = lambda: total\n",
            "closure_strictness": "strict",
        },
        tree=ast.parse(mod_src),
    )
    assert set(sc["heuristic_ctx_managers"]) == {"db.atomic", "self.lock"}

    # Verify heuristic_ctx_manager_session harvests worker threads spawned inside the session
    clear_closure_warning_cache()
    record_heuristic_non_suppressing_ctx_manager("PriorUnrelatedThreadMgr")
    with heuristic_ctx_manager_session():
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(
                record_heuristic_non_suppressing_ctx_manager, "worker.db.atomic"
            )
            f2 = pool.submit(
                record_heuristic_non_suppressing_ctx_manager, "worker.cache.lock"
            )
            f1.result()
            f2.result()
        harvested = log_heuristic_ctx_manager_summary()
    assert harvested == ("worker.cache.lock", "worker.db.atomic")


def test_synthesize_refactoring_suggestion_pair_level_structural_vs_safety_reasons(
    tmp_path: Path,
) -> None:
    """Verifies --suggest reports pair-level output/cleanup reasons and distinguishes structural incompatibility from safety hazards."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    arity_src = (
        "def f1(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return a + b\n\n"
        "def f2(x):\n"
        "    a = x + 1\n"
        "    return a\n"
    )
    arity_file = tmp_path / "arity.py"
    arity_file.write_text(arity_src, encoding="utf-8")
    sug_arity = synthesize_refactoring_suggestion(
        {
            "name": "f1:blk",
            "file": str(arity_file),
            "start": 2,
            "end": 3,
            "kind": "compound_block",
            "source_text": arity_src,
        },
        {
            "name": "f2:blk",
            "file": str(arity_file),
            "start": 7,
            "end": 7,
            "kind": "compound_block",
            "source_text": arity_src,
        },
        repo_root=str(tmp_path),
    )
    assert "skipped as structurally incompatible" in sug_arity
    assert "safety hazard" not in sug_arity
    assert "output arity mismatch" in sug_arity

    cleanup_src = (
        "def g1(x, log):\n"
        "    out = 0\n"
        "    try:\n"
        "        out = x + 1\n"
        "        step = out * 2\n"
        "    finally:\n"
        "        log.append(out)\n"
        "    return step\n\n"
        "def g2(x, log):\n"
        "    out = 0\n"
        "    try:\n"
        "        out = x + 1\n"
        "        step = out * 2\n"
        "    finally:\n"
        "        log.append(out)\n"
        "    return step\n"
    )
    cleanup_file = tmp_path / "cleanup.py"
    cleanup_file.write_text(cleanup_src, encoding="utf-8")
    sug_cleanup = synthesize_refactoring_suggestion(
        {
            "name": "g1:try_body",
            "file": str(cleanup_file),
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": cleanup_src,
        },
        {
            "name": "g2:try_body",
            "file": str(cleanup_file),
            "start": 13,
            "end": 14,
            "kind": "compound_block",
            "source_text": cleanup_src,
        },
        repo_root=str(tmp_path),
    )
    assert "rejected as unsafe" in sug_cleanup
    assert "enclosing try or with cleanup hazard" in sug_cleanup
