"""Tests for dataflow cache invariants, indexed eviction, lexical scope rules, and architectural guarantees."""
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


def test_heterogeneous_frozenset_eviction_and_fail_closed_spans(
    tmp_path: Path,
) -> None:
    """Covers heterogeneous frozenset cache eviction, fail-closed spans, PEP 3104/227 lexical rules, lazy/eager genexps without scope_ctx, C-enforced scalar built-ins vs custom dunders, carrier mutations vs rebinding, and intermediate _EnclosingFnCellInfo summaries."""
    import symtable

    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _SpanBounds,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_comprehension_deferred_closure_info,
        _extract_lazy_genexp_walrus_names,
        _has_invalid_lexical_scope_declarations,
        _has_unresolved_span_bounds,
        _is_lazy_genexp,
        _is_node_inside_span,
        _is_non_closure_scalar_expr,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _UnitClosureHazardContext,
        _has_severed_closure_cell_hazard,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

    # Heterogeneous frozenset eviction, nested AST nodes, and _ScopeCacheKey
    clear_dataflow_cache()
    evict_mod = ast.parse(
        "def outer(items):\n"
        "    def inner(y):\n"
        "        return [ (lambda z: z + y)(x) for x in items ]\n"
        "    return inner\n"
    )
    keep_mod = ast.parse("def survivor(items):\n    return [x for x in items]\n")
    outer_fn = evict_mod.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    ret_stmt = inner_fn.body[0]
    assert isinstance(ret_stmt, ast.Return)
    listcomp_node = ret_stmt.value
    assert isinstance(listcomp_node, ast.ListComp)
    call_node = listcomp_node.elt
    assert isinstance(call_node, ast.Call)
    lambda_node = call_node.func
    assert isinstance(lambda_node, ast.Lambda)

    het_keys = [
        frozenset({"metadata", outer_fn}),
        frozenset({"alpha", "beta", "gamma", 42, inner_fn}),
        ("composite", frozenset({"tag_a", "tag_b", lambda_node})),
        ("nested_comp", (_SpanBounds(1, 4), frozenset({"meta", listcomp_node}))),
        _ScopeCacheKey(
            owner=keep_mod,
            kind="cross_ref",
            params=("param",),
            aux_ast_nodes=(inner_fn,),
        ),
    ]
    keep_key = _ScopeCacheKey(
        owner=keep_mod,
        kind="survivor_key",
        params=(frozenset({"only_strings", "safe"}),),
    )
    for idx, hk in enumerate(het_keys):
        _scope_cell_info_cache[hk] = f"evict_{idx}"
        _scope_closures_cache[hk] = [(1, set())]
    _scope_cell_info_cache[keep_key] = "keep_val"

    tree_nodes = set(ast.walk(evict_mod))
    for hk in het_keys:
        assert _key_references_scope_nodes(hk, tree_nodes) is True
    assert _key_references_scope_nodes(keep_key, tree_nodes) is False

    _evict_tree_scopes(evict_mod)
    for hk in het_keys:
        assert hk not in _scope_cell_info_cache
        assert hk not in _scope_closures_cache
    assert _scope_cell_info_cache.get(keep_key) == "keep_val"

    info_before = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    _evict_tree_scopes(evict_mod)
    info_after = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    assert info_before == info_after

    # Fail-closed policy for unresolved source spans & incomplete AST metadata
    dup_line_src = (
        "def f(xs):\n"
        "    a = 1; a = 1\n"
        "    return [lambda: a for _ in xs]\n"
    )
    dup_line_tree = ast.parse(dup_line_src)
    cand_assign = ast.parse("a = 1\n").body
    assert _resolve_unit_col_bounds(
        cand_assign, dup_line_tree, 2, 2, None, None
    ) == (None, None)
    assert (
        _has_unresolved_span_bounds(
            dup_line_tree.body[0], 2, 2, None, None
        )
        is True
    )
    sc_dup_line = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "subroutine",
            "source_text": dup_line_src,
            "source_code": "a = 1\n",
        },
        tree=dup_line_tree,
    )
    assert "severed_closure_cell" in sc_dup_line["control_flow_hazards"]
    assert sc_dup_line["is_control_flow_safe"] is False

    subexpr_line_src = (
        "def f(xs):\n"
        "    res = [x + 1 for x in xs] + [x + 1 for x in xs]\n"
        "    return res\n"
    )
    subexpr_tree = ast.parse(subexpr_line_src)
    cand_comp_expr = ast.parse("[x + 1 for x in xs]\n").body
    assert _resolve_unit_col_bounds(
        cand_comp_expr, subexpr_tree, 2, 2, None, None
    ) == (None, None)
    sc_ambig_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "comprehension",
            "source_text": subexpr_line_src,
            "source_code": "[x + 1 for x in xs]",
        },
        tree=subexpr_tree,
    )
    assert "severed_closure_cell" in sc_ambig_comp["control_flow_hazards"]

    corrupt_tree = ast.parse(
        "def f(xs):\n"
        "    a = 1\n"
        "    b = a + 2\n"
        "    return [lambda: b for _ in xs]\n"
    )
    corrupt_fn = corrupt_tree.body[0]
    assert isinstance(corrupt_fn, ast.FunctionDef)
    corrupt_stmt = corrupt_fn.body[1]
    corrupt_stmt.end_lineno = None
    assert _is_node_inside_span(corrupt_stmt, _SpanBounds(3, 3, 4, 13)) is False
    assert (
        _has_unresolved_span_bounds(corrupt_fn, 3, 3, 4, 13) is True
    )
    assert (
        _has_severed_closure_cell_hazard(
            _UnitClosureHazardContext(
                candidate_stmts=ast.parse("b = a + 2\n").body,
                mod_tree=corrupt_tree,
                u_start=3,
                u_end=3,
                u_start_col=4,
                u_end_col=13,
                is_subroutine=True,
                unit_kind="subroutine",
                inputs=["a"],
                outputs=["b"],
                stores=["b"],
                deleted_names=frozenset(),
                unit_top_locals=frozenset({"b"}),
                cols_resolved=True,
            )
        )
        is True
    )

    corrupt_col_tree = ast.parse(
        "def f(xs):\n"
        "    a = 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    corrupt_col_fn = corrupt_col_tree.body[0]
    assert isinstance(corrupt_col_fn, ast.FunctionDef)
    corrupt_col_fn.body[1].end_col_offset = None
    assert (
        _is_node_inside_span(
            corrupt_col_fn.body[1], _SpanBounds(3, 3, 4, 13)
        )
        is False
    )
    assert _has_unresolved_span_bounds(corrupt_col_fn, 3, 3, 4, 13) is True

    # PEP 3104 & PEP 227 lexical-scope rules (invalid vs valid multi-level variants)
    invalid_variants = [
        "def outer():\n    x = 0\n    def mid():\n        global x\n        def inner():\n            nonlocal x\n            x += 1\n",
        "def outer():\n    x = 0\n    def inner():\n        nonlocal x\n        global x\n        x = 1\n",
        "def outer():\n    x = 0\n    def inner(x):\n        nonlocal x\n        x = 1\n",
        "def outer(x):\n    global x\n    x = 1\n",
        "def outer():\n    def inner():\n        nonlocal missing_var\n        missing_var = 1\n",
    ]
    for inv_src in invalid_variants:
        with pytest.raises(SyntaxError):
            symtable.symtable(inv_src, "<sym>", "exec")
        inv_tree = ast.parse(inv_src)
        assert (
            _has_invalid_lexical_scope_declarations(
                inv_tree.body, mod_tree=inv_tree
            )
            is True
        )
        sc_inv = inspect_single_unit_scope(
            {
                "start": 1,
                "end": len(inv_src.splitlines()),
                "kind": "function",
                "source_text": inv_src,
                "source_code": inv_src,
            },
            tree=inv_tree,
        )
        assert "severed_closure_cell" in sc_inv["control_flow_hazards"]
        assert sc_inv["is_control_flow_safe"] is False

    valid_mid_local_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        x = 2\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x = 3\n"
        "        inner()\n"
        "        return x\n"
        "    y = x + 10\n"
        "    return y, mid()\n"
    )
    symtable.symtable(valid_mid_local_src, "<valid1>", "exec")
    v1_tree = ast.parse(valid_mid_local_src)
    v1_outer = v1_tree.body[0]
    assert isinstance(v1_outer, ast.FunctionDef)
    v1_mid = v1_outer.body[1]
    assert isinstance(v1_mid, ast.FunctionDef)
    v1_inner = v1_mid.body[1]
    assert isinstance(v1_inner, ast.FunctionDef)
    assert _has_invalid_lexical_scope_declarations(
        v1_tree.body, mod_tree=v1_tree
    ) is False
    assert _extract_closure_nonlocal_mutations_and_deletions(v1_inner) == (
        {"x"},
        set(),
    )
    assert _extract_closure_nonlocal_mutations_and_deletions(v1_mid) == (
        set(),
        set(),
    )
    sc_v1 = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 10,
            "kind": "subroutine",
            "source_text": valid_mid_local_src,
            "source_code": "y = x + 10\n",
        },
        tree=v1_tree,
    )
    assert "severed_closure_cell" not in sc_v1["control_flow_hazards"]

    valid_between_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        global x\n"
        "        def between():\n"
        "            x = 2\n"
        "            def inner():\n"
        "                nonlocal x\n"
        "                x = 3\n"
        "            inner()\n"
        "            return x\n"
        "        return between()\n"
        "    return x, mid()\n"
    )
    symtable.symtable(valid_between_src, "<valid2>", "exec")
    v2_tree = ast.parse(valid_between_src)
    assert (
        _has_invalid_lexical_scope_declarations(v2_tree.body, mod_tree=v2_tree)
        is False
    )

    valid_chained_nl_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x += 5\n"
        "        return inner\n"
        "    cb = lambda: x\n"
        "    return cb, mid\n"
    )
    symtable.symtable(valid_chained_nl_src, "<valid3>", "exec")
    v3_tree = ast.parse(valid_chained_nl_src)
    v3_mid = v3_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert _extract_closure_nonlocal_mutations_and_deletions(v3_mid) == (
        {"x"},
        set(),
    )
    sc_v3 = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "closure",
            "source_text": valid_chained_nl_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=v3_tree,
    )
    assert "severed_closure_cell" in sc_v3["control_flow_hazards"]

    valid_cls_global_src = (
        "x = 0\n"
        "def outer():\n"
        "    x = 1\n"
        "    class Mid:\n"
        "        global x\n"
        "        x = 99\n"
        "        def bump(self):\n"
        "            nonlocal x\n"
        "            x += 10\n"
        "    cb = lambda: x\n"
        "    return cb, Mid\n"
    )
    symtable.symtable(valid_cls_global_src, "<valid4>", "exec")
    v4_tree = ast.parse(valid_cls_global_src)
    assert (
        _has_invalid_lexical_scope_declarations(v4_tree.body, mod_tree=v4_tree)
        is False
    )
    v4_cls = v4_tree.body[1].body[1]  # type: ignore[attr-defined]
    assert _extract_closure_nonlocal_mutations_and_deletions(v4_cls) == (
        {"x"},
        set(),
    )
    sc_v4 = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 10,
            "kind": "closure",
            "source_text": valid_cls_global_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=v4_tree,
    )
    assert "severed_closure_cell" in sc_v4["control_flow_hazards"]

    # _is_lazy_genexp & comprehension helpers with and without scope_ctx
    detached_gen = ast.parse("(x + a for x in items)", mode="eval").body
    assert isinstance(detached_gen, ast.GeneratorExp)
    assert _is_lazy_genexp(detached_gen, None) is True

    nested_eager_comp = ast.parse(
        "[sum((w := x) + a for x in row) for row in rows]", mode="eval"
    ).body
    assert isinstance(nested_eager_comp, ast.ListComp)
    assert _extract_lazy_genexp_walrus_names(nested_eager_comp, None) == set()
    c_reads_eager, c_muts_eager = _extract_comprehension_deferred_closure_info(
        nested_eager_comp, None
    )
    assert c_reads_eager == set()
    assert c_muts_eager == set()

    top_eager_sum = ast.parse(
        "sum((w := x) + a for x in items)", mode="eval"
    ).body
    assert _extract_lazy_genexp_walrus_names(top_eager_sum, None) == set()

    nested_lazy_comp = ast.parse(
        "[((w := x) + a for x in row) for row in rows]", mode="eval"
    ).body
    assert isinstance(nested_lazy_comp, ast.ListComp)
    assert _extract_lazy_genexp_walrus_names(nested_lazy_comp, None) == {"w"}
    c_reads_lazy, c_muts_lazy = _extract_comprehension_deferred_closure_info(
        nested_lazy_comp, None
    )
    assert "a" in c_reads_lazy
    assert "w" in c_muts_lazy

    # C-enforced scalar built-ins vs unconstrained custom dunders
    unshadowed_scope_ctx = _FnScopeContext(
        fn_locals=frozenset(),
        parent_map={},
        shadowed_names=frozenset(),
        has_star_import=False,
    )
    assert (
        _is_non_closure_scalar_expr(
            ast.parse("len(items)", mode="eval").body, None
        )
        is False
    )
    for scalar_code in (
        "len(items)",
        "int(x)",
        "bool(x)",
        "str(x)",
        "ord(c)",
        "isinstance(x, int)",
        "1 + 2",
        "'a' == 'b'",
        "not custom_obj",
        "-len(items)",
        "len(items) + 1",
        "len(items) == 0",
    ):
        expr_node = ast.parse(scalar_code, mode="eval").body
        assert (
            _is_non_closure_scalar_expr(expr_node, unshadowed_scope_ctx) is True
        ), scalar_code

    for non_scalar_code in (
        "abs(custom_obj)",
        "round(custom_obj)",
        "divmod(custom_obj, 2)",
        "pow(custom_obj, 2)",
        "sum(items)",
        "-custom_obj",
        "custom_obj + 1",
        "len(items) + custom_obj",
        "custom_obj == other_obj",
    ):
        expr_node = ast.parse(non_scalar_code, mode="eval").body
        assert (
            _is_non_closure_scalar_expr(expr_node, unshadowed_scope_ctx) is False
        ), non_scalar_code

    dunder_factory_src = (
        "def outer(items):\n"
        "    scale = 1\n"
        "    class AbsFactory:\n"
        "        def __abs__(self):\n"
        "            return lambda v: v + scale\n"
        "    fac = AbsFactory()\n"
        "    cb = abs(fac)\n"
        "    for x in items:\n"
        "        scale = x * 2\n"
        "        res = cb(x)\n"
        "    return res\n"
    )
    sc_dunder = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_text": dunder_factory_src,
            "source_code": "for x in items:\n    scale = x * 2\n    res = cb(x)\n",
        }
    )
    assert "severed_closure_cell" in sc_dunder["control_flow_hazards"]

    # Carrier mutation vs rebinding, comprehension iterables, and _EnclosingFnCellInfo
    for mut_stmt in (
        "box.attr = 2",
        "box[0] = 2",
        "box.attr += 2",
        "box[0] += 2",
        "box.custom_method(2)",
    ):
        obj_mut_src = (
            "def outer(items, box):\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: box[0] + x)\n"
            f"    {mut_stmt}\n"
            "    return cbs\n"
        )
        sc_obj_mut = inspect_single_unit_scope(
            {
                "start": 3,
                "end": 4,
                "kind": "compound_block",
                "source_text": obj_mut_src,
                "source_code": "for x in items:\n    cbs.append(lambda x=x: box[0] + x)\n",
            }
        )
        assert "severed_closure_cell" not in sc_obj_mut["control_flow_hazards"], mut_stmt

    for carrier_stmt in (
        "holder.cb = lambda: scale",
        "holder.register(lambda: scale)",
    ):
        carrier_mut_src = (
            "def outer(items, holder):\n"
            f"    {carrier_stmt}\n"
            "    for x in items:\n"
            "        scale = x + 1\n"
            "    return holder\n"
        )
        sc_carrier_mut = inspect_single_unit_scope(
            {
                "start": 3,
                "end": 4,
                "kind": "compound_block",
                "source_text": carrier_mut_src,
                "source_code": "for x in items:\n    scale = x + 1\n",
            }
        )
        assert "severed_closure_cell" in sc_carrier_mut["control_flow_hazards"], carrier_stmt

    eager_iter_src = (
        "def outer(items):\n"
        "    cbs = [lambda x=x: x + 1 for x in items]\n"
        "    items = []\n"
        "    return cbs\n"
    )
    sc_eager_iter = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": eager_iter_src,
            "source_code": "[lambda x=x: x + 1 for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_eager_iter["control_flow_hazards"]

    cf_summary_src = (
        "def outer(flag, items):\n"
        "    uncond = 1\n"
        "    if flag:\n"
        "        both = 10\n"
        "        only_then = 20\n"
        "    else:\n"
        "        both = 11\n"
        "    try:\n"
        "        may_del = 30\n"
        "    except RuntimeError as err_var:\n"
        "        del may_del\n"
        "    for item in items:\n"
        "        unit_marker = both + uncond\n"
        "        loop_var = item\n"
        "    after_var = unit_marker\n"
        "    return after_var, loop_var\n"
    )
    cf_tree = ast.parse(cf_summary_src)
    cf_info = _analyze_enclosing_fn_cells(
        cf_tree.body[0], (13, 13, 8, 35), mod_tree=cf_tree
    )
    assert {"flag", "items", "uncond", "both", "item"} <= cf_info.pre_bound
    assert "only_then" not in cf_info.pre_bound
    assert "only_then" in cf_info.pre_mutated
    assert "may_del" not in cf_info.pre_bound
    assert "err_var" in cf_info.pre_unbound
    assert "loop_var" in cf_info.loop_mutated
    assert "after_var" in cf_info.post_mutated
    assert "unit_marker" in cf_info.post_reads

    e2e_hazard_src = (
        "def f1(items):\n"
        "    cb = lambda v: v + scale\n"
        "    for x in items:\n"
        "        scale = x + 1\n"
        "    return cb, scale\n\n"
        "def f2(items):\n"
        "    cb = lambda v: v + scale\n"
        "    for x in items:\n"
        "        scale = x + 1\n"
        "    return cb, scale\n"
    )
    e2e_file = tmp_path / "e2e_hazard.py"
    e2e_file.write_text(e2e_hazard_src, encoding="utf-8")
    u1_e2e = {
        "file": str(e2e_file),
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": e2e_hazard_src,
        "source_code": "for x in items:\n    scale = x + 1\n",
    }
    u2_e2e = {
        "file": str(e2e_file),
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": e2e_hazard_src,
        "source_code": "for x in items:\n    scale = x + 1\n",
    }
    sc1_e2e = inspect_single_unit_scope(u1_e2e, repo_root=str(tmp_path))
    sc2_e2e = inspect_single_unit_scope(u2_e2e, repo_root=str(tmp_path))
    comb_e2e = analyze_unit_variable_scope(
        u1_e2e, u2_e2e, repo_root=str(tmp_path), scope1=sc1_e2e, scope2=sc2_e2e
    )
    assert "severed_closure_cell" in comb_e2e["control_flow_hazards"]
    assert comb_e2e["is_control_flow_safe"] is False
    res_pair = resolve_clone_pair_outputs(
        u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True
    )
    assert res_pair.outputs is None
    assert res_pair.skip_reason == "severed closure cell"
    res_gen = resolve_clone_generator_subroutine_outputs(
        u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True
    )
    assert res_gen.outputs is None
    assert res_gen.skip_reason == "severed closure cell"
    assert (
        generate_refactoring_patch(
            [(1.0, u1_e2e, u2_e2e)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )


def test_multilevel_concurrent_sibling_with_definite_bindings_and_class_escape() -> None:
    """Covers multi-level concurrent sibling detection, with definite bindings, class escape, and context cache."""
    from pydoppelgangerhunt.fixer._df_cache import _get_scope_parent_map
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _collect_escaped_and_executed_intra_unit_defs,
        _collect_unit_nonlocal_mutators,
        _extract_closure_nonlocal_mutations_and_deletions_uncached,
        _extract_fn_outer_carrier_targets,
        _extract_fn_outer_carrier_targets_uncached,
        _extract_fn_returned_closure_refs,
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
        _is_closure_factory_scope,
        _propagate_sibling_closures,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _check_sibling_closure_sharing,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _build_enclosing_chain_cell_infos,
        _extract_block_definite_bindings_and_term,
        _find_enclosing_chain_definite_pre_bound,
    )

    # 1. Multi-level concurrent sibling detection (outer -> middle -> inner -> generator)
    p1_src = (
        "def outer():\n"
        "    value = 0\n"
        "    def sibling():\n"
        "        return value\n"
        "    def middle():\n"
        "        def inner():\n"
        "            def generator():\n"
        "                nonlocal value\n"
        "                value += 1\n"
        "                yield value\n"
        "            return generator()\n"
        "        return inner()\n"
        "    return middle(), sibling()\n"
    )
    p1_tree = ast.parse(p1_src)
    p1_unit = {
        "start": 9,
        "end": 10,
        "kind": "linear_block",
        "source_text": p1_src,
        "source_code": "value += 1\nyield value\n",
    }
    p1_scope = inspect_single_unit_scope(p1_unit, tree=p1_tree)
    assert "severed_closure_cell" in p1_scope["control_flow_hazards"]
    assert p1_scope["is_control_flow_safe"] is False
    p1_chain = _find_enclosing_function_chain(p1_tree, 9, 10)
    assert len(p1_chain) == 4
    p1_chain_infos = _build_enclosing_chain_cell_infos(
        p1_chain, 9, 10, None, None, frozenset({"value"}), mod_tree=p1_tree
    )
    assert (
        _check_sibling_closure_sharing(
            frozenset(),
            frozenset({"value"}),
            p1_chain_infos,
            is_suspended=True,
        )
        is True
    )

    p1_shadowed_src = (
        "def outer():\n"
        "    value = 0\n"
        "    def sibling():\n"
        "        return value\n"
        "    def middle():\n"
        "        value = 10\n"
        "        for step in (1, 2):\n"
        "            value += step\n"
        "            yield value\n"
        "    return list(middle()), sibling()\n"
    )
    p1_shadowed_tree = ast.parse(p1_shadowed_src)
    p1_shadowed_unit = {
        "start": 7,
        "end": 9,
        "kind": "compound_block",
        "source_text": p1_shadowed_src,
        "source_code": (
            "for step in (1, 2):\n"
            "    value += step\n"
            "    yield value\n"
        ),
    }
    p1_shadowed_scope = inspect_single_unit_scope(
        p1_shadowed_unit, tree=p1_shadowed_tree
    )
    assert "severed_closure_cell" not in p1_shadowed_scope["control_flow_hazards"]
    p1_sh_chain = _find_enclosing_function_chain(p1_shadowed_tree, 7, 9)
    p1_sh_infos = _build_enclosing_chain_cell_infos(
        p1_sh_chain, 7, 9, None, None, frozenset({"value"}), mod_tree=p1_shadowed_tree
    )
    assert (
        _check_sibling_closure_sharing(
            frozenset(),
            frozenset({"value"}),
            p1_sh_infos,
            is_suspended=True,
        )
        is False
    )

    # 2. Definite-binding analysis inside `with` and `async with` bodies
    with_definite_src = (
        "import contextlib\n"
        "def process(cm, items):\n"
        "    with contextlib.nullcontext(cm) as resource:\n"
        "        scale = resource.factor\n"
        "    out = [x * scale for x in items]\n"
        "    return out\n"
    )
    with_def_tree = ast.parse(with_definite_src)
    with_def_chain = _find_enclosing_function_chain(with_def_tree, 5, 5)
    pre_bound_with = _find_enclosing_chain_definite_pre_bound(
        with_def_chain, 5, 5, mod_tree=with_def_tree
    )
    assert {"cm", "items", "resource", "scale"} <= pre_bound_with
    sc_with_def = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "linear_block",
            "source_text": with_definite_src,
            "source_code": "out = [x * scale for x in items]\n",
        },
        tree=with_def_tree,
    )
    assert "severed_closure_cell" not in sc_with_def["control_flow_hazards"]

    with_cases: list[tuple[str, set[str], bool]] = [
        (
            "with cm1() as a, cm2() as b:\n    c = 1\n",
            {"a", "b", "c"},
            False,
        ),
        (
            "with cm1() as a, cm2() as b:\n    c = a + b\n",
            {"a", "b"},
            False,
        ),
        (
            "with cm():\n    if cond:\n        c = 1\n",
            set(),
            False,
        ),
        (
            "cond = True\nwith cm():\n    if cond is not None:\n        c = 1\n    else:\n        c = 2\n",
            {"cond", "c"},
            False,
        ),
        (
            "cond = False\nwith cm():\n    if cond is None:\n        return 0\n    c = 1\n",
            {"cond", "c"},
            False,
        ),
        (
            "with cm():\n    c = 1\n    return c\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    break\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    continue\n    dead = 2\n",
            {"c"},
            True,
        ),
        (
            "with cm():\n    c = 1\n    del c\n",
            set(),
            False,
        ),
        (
            "c = 1\nwith cm():\n    del c\n",
            set(),
            False,
        ),
        (
            "async def _af():\n    async with acm() as a:\n        b = 1\n",
            {"a", "b"},
            False,
        ),
        (
            "with suppress(KeyError):\n    c = 1\n",
            {"c"},
            False,
        ),
        (
            "with contextlib.suppress(KeyError) as sup:\n    c = 1\n",
            {"sup", "c"},
            False,
        ),
        (
            "with suppress(KeyError):\n    c = d['k']\n",
            set(),
            False,
        ),
    ]
    for code_str, expected_bound, expected_term in with_cases:
        parsed = ast.parse(code_str)
        stmts = (
            parsed.body[0].body
            if isinstance(parsed.body[0], ast.AsyncFunctionDef)
            else parsed.body
        )
        got_bound, got_term = _extract_block_definite_bindings_and_term(
            stmts, None
        )
        assert got_bound == expected_bound, f"Mismatch for:\n{code_str}"
        assert got_term is expected_term, f"Term mismatch for:\n{code_str}"

    # 3. Class definitions escape classification
    cls_unreferenced_src = (
        "def outer(items):\n"
        "    def reader():\n"
        "        return count\n"
        "    count = 0\n"
        "    class LocalHelper:\n"
        "        def mutate(self):\n"
        "            nonlocal count\n"
        "            count += 1\n"
        "    res = count + len(items)\n"
        "    return count, res, reader()\n"
    )
    cls_unref_tree = ast.parse(cls_unreferenced_src)
    cls_unref_unit = {
        "start": 4,
        "end": 9,
        "kind": "linear_block",
        "source_text": cls_unreferenced_src,
        "source_code": (
            "count = 0\n"
            "class LocalHelper:\n"
            "    def mutate(self):\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "res = count + len(items)\n"
        ),
    }
    muts_unref = _collect_unit_nonlocal_mutators(
        ast.parse(str(cls_unref_unit["source_code"])).body
    )
    assert any(
        m.name == "LocalHelper" and m.is_escaped_def is False
        for m in muts_unref
    )
    sc_cls_unref = inspect_single_unit_scope(
        cls_unref_unit, tree=cls_unref_tree
    )
    assert "severed_closure_cell" not in sc_cls_unref["control_flow_hazards"]

    escaping_cls_tails = [
        ("escaped = LocalHelper\n", "    return count, escaped\n"),
        ("register(LocalHelper)\n", "    return count\n"),
        ("inst = LocalHelper()\n", "    return count\n"),
        ("cb = lambda: LocalHelper\n", "    return count, cb\n"),
    ]
    for extra_stmt, ret_line in escaping_cls_tails:
        esc_src = (
            "def outer(items, register):\n"
            "    count = 0\n"
            "    class LocalHelper:\n"
            "        def mutate(self):\n"
            "            nonlocal count\n"
            "            count += 1\n"
            f"    {extra_stmt}"
            f"{ret_line}"
        )
        esc_unit_code = (
            "count = 0\n"
            "class LocalHelper:\n"
            "    def mutate(self):\n"
            "        nonlocal count\n"
            "        count += 1\n"
            f"{extra_stmt}"
        )
        sc_esc = inspect_single_unit_scope(
            {
                "start": 2,
                "end": 7,
                "kind": "linear_block",
                "source_text": esc_src,
                "source_code": esc_unit_code,
            },
            tree=ast.parse(esc_src),
        )
        assert "severed_closure_cell" in sc_esc["control_flow_hazards"], (
            f"Expected hazard for: {extra_stmt}"
        )

    for cls_header_or_body in [
        "@dec\nclass LocalHelper:\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(Base):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(metaclass=Meta):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper:\n    nonlocal count\n    count += 1\n",
    ]:
        c5_stmts = ast.parse("count = 0\n" + cls_header_or_body).body
        c5_muts = _collect_unit_nonlocal_mutators(c5_stmts)
        assert any(
            m.name == "LocalHelper" and m.is_escaped_def is True
            for m in c5_muts
        )
        _, c5_esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(
            c5_stmts, frozenset()
        )
        assert any(
            isinstance(s, ast.ClassDef) and id(s) in c5_esc_ids
            for s in c5_stmts
        )

    # 4. Cached closure metadata context validation on shared AST nodes
    clear_dataflow_cache()
    shared_fn_tree = ast.parse(
        "def factory(items):\n"
        "    res = any(x + outer_val for x in items)\n"
        "    box.append(any(x + outer_val for x in items))\n"
        "    return any(cb_ref for x in items)\n"
    )
    shared_fn = shared_fn_tree.body[0]
    assert isinstance(shared_fn, ast.FunctionDef)
    parent_map_shared = MappingProxyType(_get_scope_parent_map(shared_fn_tree))
    owner_a = ast.Module(body=[], type_ignores=[])
    owner_b = ast.Module(body=[], type_ignores=[])
    ctx_a = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items"}),
        has_star_import=False,
        scope_owner=owner_a,
    )
    ctx_b = _FnScopeContext(
        fn_locals=frozenset({"items", "any"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items", "any"}),
        has_star_import=False,
        scope_owner=owner_b,
    )
    ctx_c = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=parent_map_shared,
        shadowed_names=frozenset({"items"}),
        has_star_import=True,
        scope_owner=owner_a,
    )
    assert _is_closure_factory_scope(shared_fn, ctx_a) is False
    assert _is_closure_factory_scope(shared_fn, ctx_b) is True
    assert _is_closure_factory_scope(shared_fn, ctx_c) is True
    ctx_detached = _FnScopeContext(
        fn_locals=frozenset({"items"}),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset({"items"}),
        has_star_import=False,
        scope_owner=owner_a,
    )
    assert _is_closure_factory_scope(shared_fn, ctx_detached) is False
    assert _extract_fn_outer_carrier_targets(shared_fn, frozenset(), ctx_a) == set()
    assert _extract_fn_outer_carrier_targets(shared_fn, frozenset(), ctx_b) == {"box"}
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_a) == (set(), set())
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_b) == (
        set(),
        {"any"},
    )

    # 5. Table-driven definite-binding runtime oracle comparison
    oracle_cases: list[tuple[str, dict[str, object], set[str]]] = [
        (
            "if flag:\n    a = 1\n    b = 2\nelse:\n    a = 3\n    c = 4\n",
            {"flag": True},
            {"a"},
        ),
        (
            "try:\n    a = 1\nexcept ValueError:\n    a = 2\nfinally:\n    f = 3\n",
            {},
            {"a", "f"},
        ),
        (
            "if (w := val) and (u := w + 1):\n    z = 1\nelse:\n    z = 2\n",
            {"val": 0},
            {"w", "z"},
        ),
        (
            "for i in (it := [1, 2]):\n    loop_v = i\n",
            {},
            {"it"},
        ),
    ]
    if sys.version_info >= (3, 10):
        oracle_cases.append(
            (
                "match val:\n    case 1:\n        m = 10\n    case _:\n        m = 20\n",
                {"val": 5},
                {"m"},
            )
        )
    if sys.version_info >= (3, 11):
        oracle_cases.append(
            (
                "try:\n    a = 1\nexcept* ValueError:\n    a = 2\nfinally:\n    f = 3\n",
                {},
                {"f"},
            )
        )
    for code_str, env_init, expected_def in oracle_cases:
        stmts = ast.parse(code_str).body
        static_bound, _ = _extract_block_definite_bindings_and_term(stmts, None)
        assert static_bound == expected_def
        ns: dict[str, object] = dict(env_init)
        exec(compile(code_str, "<oracle>", "exec"), {}, ns)  # noqa: S102
        assert static_bound <= set(ns.keys())

    # _has_invalid_lexical_scope_declarations & class scopes
    lex_cases: list[tuple[str, bool]] = [
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Example:\n"
            "        value = 2\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            False,
        ),
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Example:\n"
            "        global value\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            False,
        ),
        (
            "def outer():\n"
            "    global value\n"
            "    class Example:\n"
            "        def mutate(self):\n"
            "            nonlocal value\n"
            "            value = 3\n",
            True,
        ),
        (
            "def outer():\n"
            "    value = 1\n"
            "    class OuterCls:\n"
            "        class InnerCls:\n"
            "            def mutate(self):\n"
            "                nonlocal value\n"
            "                value = 4\n",
            False,
        ),
        (
            "def outer():\n"
            "    value = 1\n"
            "    class Cls:\n"
            "        def mid(self):\n"
            "            global value\n"
            "            def inner():\n"
            "                nonlocal value\n"
            "                value = 5\n",
            True,
        ),
    ]
    for src_a, expected_invalid in lex_cases:
        tree_a = ast.parse(src_a)
        got_invalid = _has_invalid_lexical_scope_declarations(
            tree_a.body, mod_tree=tree_a
        )
        assert got_invalid is expected_invalid
        try:
            compile(src_a, "<lex_case>", "exec")
            compiled_ok = True
        except SyntaxError:
            compiled_ok = False
        assert compiled_ok is (not expected_invalid)

    # _extract_closure_nonlocal_mutations_and_deletions_uncached rebinding matrix
    rebinding_fn = ast.parse(
        "def holder():\n"
        "    nonlocal a, b, c, d, e, f, g, h\n"
        "    a = 1\n"
        "    del b\n"
        "    _ = [x := 1 for c in (1, 2) if (d := c)]\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        pass\n"
        "    class Sub:\n"
        "        h = 99\n"
        "        def sub_m(self):\n"
        "            nonlocal h\n"
        "            h = 100\n"
    ).body[0]
    assert isinstance(rebinding_fn, ast.FunctionDef)
    if sys.version_info >= (3, 10):
        rebinding_fn.body.append(
            ast.parse("match a:\n    case f:\n        pass\n").body[0]
        )
    else:
        synth_match_as_cls = type(
            "MatchAs", (ast.stmt,), {"_fields": ("pattern", "name")}
        )
        rebinding_fn.body.append(synth_match_as_cls(pattern=None, name="f"))
    if sys.version_info >= (3, 12):
        rebinding_fn.body.append(ast.parse("type g = int\n").body[0])
    else:
        synth_ta_cls = type(
            "TypeAlias",
            (ast.stmt,),
            {"_fields": ("name", "type_params", "value")},
        )
        rebinding_fn.body.append(
            synth_ta_cls(
                name=ast.Name(id="g", ctx=ast.Store()),
                type_params=[],
                value=ast.Name(id="int", ctx=ast.Load()),
            )
        )
    b_muts, b_dels = _extract_closure_nonlocal_mutations_and_deletions_uncached(
        rebinding_fn
    )
    assert b_muts == frozenset({"a", "b", "d", "e", "f", "g", "h"})
    assert b_dels == frozenset({"b", "e"})

    # _propagate_sibling_closures edge cases
    prop_res = _propagate_sibling_closures(
        [
            ("self_fn", frozenset({"self_fn", "x"}), frozenset({"m1"})),
            ("mut_a", frozenset({"mut_b", "y"}), frozenset()),
            ("mut_b", frozenset({"mut_a", "z"}), frozenset({"m2"})),
            ("dup", frozenset({"d1"}), frozenset()),
            ("dup", frozenset({"d2"}), frozenset({"dm"})),
        ],
        alias_edges=[
            ("alias1", frozenset({"mut_a"})),
            ("alias2", frozenset({"alias1"})),
            ("alias1", frozenset({"alias2"})),
        ],
    )
    prop_dict = {name: (r, m) for name, r, m in prop_res if name is not None}
    assert prop_dict["self_fn"] == ({"self_fn", "x"}, {"m1"})
    assert prop_dict["mut_a"] == ({"mut_a", "mut_b", "y", "z"}, {"m2"})
    assert prop_dict["alias2"] == ({"mut_a", "mut_b", "y", "z"}, {"m2"})
    assert prop_dict["dup"] == ({"d1", "d2"}, {"dm"})

    # _find_enclosing_function_chain and _resolve_unit_col_bounds
    chain_col_src = (
        "@dec\n"
        "def outer(a=lambda: 1):\n"
        "    def single_line(): x = 1; y = 2; return x + y\n"
        "    return single_line()\n"
    )
    chain_col_tree = ast.parse(chain_col_src)
    assert _find_enclosing_function_chain(chain_col_tree, 1, 1) == []
    chain_d = _find_enclosing_function_chain(
        chain_col_tree, 3, 3, u_start_col=23, u_end_col=35
    )
    assert [getattr(n, "name", "") for _, _, n in chain_d] == [
        "outer",
        "single_line",
    ]
    outer_fn_d = chain_col_tree.body[0]
    assert isinstance(outer_fn_d, ast.FunctionDef)
    single_fn_d = outer_fn_d.body[0]
    assert isinstance(single_fn_d, ast.FunctionDef)
    cand_d = [single_fn_d.body[0]]
    assert _resolve_unit_col_bounds(cand_d, chain_col_tree, 3, 3, None, None) == (
        23,
        28,
    )

    # _extract_fn_outer_carrier_targets_uncached
    carrier_fn = ast.parse(
        "def carrier_fn():\n"
        "    global glob_box\n"
        "    nonlocal nl_var\n"
        "    local_list = []\n"
        "    local_list.append(lambda: captured)\n"
        "    glob_box.append(lambda: captured)\n"
        "    outer_Attr.cb = lambda: captured\n"
        "    outer_sub[0] = lambda: captured\n"
        "    outer_nested.items[0].append(lambda: captured)\n"
        "    fn_ref = lambda: captured\n"
        "    fn_alias = fn_ref\n"
        "    outer_via_alias.register(fn_alias)\n"
        "    nl_var = fn_alias\n"
    ).body[0]
    assert isinstance(carrier_fn, ast.FunctionDef)
    e_targets = _extract_fn_outer_carrier_targets_uncached(carrier_fn)
    assert e_targets == frozenset(
        {
            "outer_Attr",
            "outer_sub",
            "outer_nested",
            "outer_via_alias",
            "nl_var",
        }
    )


def test_pep227_trydel_definite_equivalence_and_cache_isolation() -> None:
    """Verifies PEP 227 class-level global isolation, try-del-return handler fallthrough, definite-binding engine equivalence, statement-delta equivalence, and cache isolation."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _scope_cell_info_cache,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _extract_cached_scope_free_reads,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _extract_fn_outer_carrier_targets_uncached,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _is_self_escaping_def_node,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
        _extract_block_definite_bindings_and_term,
        _iter_class_body_methods,
    )
    from pydoppelgangerhunt.fixer._df_visitors import (
        _ClassScopeVisitor,
        _extract_nested_scope_free_reads_with_nodes,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        ensure_unit_scope_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
    )

    # Class-level `global` does not propagate into nested methods/lambdas/comprehensions/classes
    pep227_src = (
        "def outer(xs):\n"
        "    items = []\n"
        "    cell_v = 10\n"
        "    class C:\n"
        "        global items, cell_v\n"
        "        direct_read = cell_v\n"
        "        comp_read = [cell_v + x for x in xs]\n"
        "        lam_read = lambda self: cell_v\n"
        "        def register(self, cb):\n"
        "            items.append(cb)\n"
        "        def capture_and_store(self):\n"
        "            items.append(lambda: cell_v)\n"
        "        class Inner:\n"
        "            def use(self):\n"
        "                return cell_v\n"
        "    c = C()\n"
        "    c.register(lambda: cell_v)\n"
        "    cell_v = 20\n"
        "    return c, items\n"
    )
    compile(pep227_src, "<pep227_src>", "exec")
    pep227_tree = ast.parse(pep227_src)
    outer_fn = pep227_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    cls_c = outer_fn.body[2]
    assert isinstance(cls_c, ast.ClassDef)
    outer_ctx = _get_fn_scope_context(outer_fn, mod_tree=pep227_tree)
    c_targets = _extract_fn_outer_carrier_targets_uncached(
        cls_c, frozenset(), outer_ctx
    )
    assert "items" in c_targets
    cls_vis = _ClassScopeVisitor()
    for s_node in cls_c.body:
        cls_vis.visit(s_node)
    assert "cell_v" in cls_vis.globals
    assert "cell_v" in cls_vis.nested_free_reads
    nested_reads, _ = _extract_nested_scope_free_reads_with_nodes(cls_c)
    assert "cell_v" in nested_reads

    cls_sib_carrier_src = (
        "def outer(xs):\n"
        "    box = []\n"
        "    val = 1\n"
        "    class Handler:\n"
        "        global box\n"
        "        def save_cb(self):\n"
        "            box.append(lambda: val)\n"
        "        def work(self):\n"
        "            nonlocal val\n"
        "            for x in xs:\n"
        "                val += x + box[0]()\n"
        "            return val\n"
        "    return Handler, box\n"
    )
    compile(cls_sib_carrier_src, "<cls_sib_carrier>", "exec")
    sc_cls_sib = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 11,
            "kind": "compound_block",
            "source_text": cls_sib_carrier_src,
            "source_code": "for x in xs:\n    val += x + box[0]()\n",
        },
        tree=ast.parse(cls_sib_carrier_src),
    )
    assert "severed_closure_cell" in sc_cls_sib["control_flow_hazards"]

    # Unknown strictness mode without source fails closed
    unmarked_precomputed: dict[str, Any] = {
        "inputs": ["a"],
        "outputs": ["b"],
        "stores": ["b"],
        "definite_stores": ["b"],
        "free_vars": ["a"],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
        "control_flow_hazards": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_return": False,
        "local_imports": [],
        "yield_expr_names": [],
    }
    strict_unknown_res = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        unmarked_precomputed,
        skip_pre_unit_closures=False,
    )
    assert strict_unknown_res["is_control_flow_safe"] is False
    assert "severed_closure_cell" in strict_unknown_res["control_flow_hazards"]
    assert strict_unknown_res["_skip_pre_unit_closures"] is False

    # Same-name definitions in conditional branches
    same_name_defs_src = (
        "def outer(flag, items):\n"
        "    x = 0\n"
        "    if flag:\n"
        "        def callback():\n"
        "            return x\n"
        "    else:\n"
        "        def callback():\n"
        "            return 1\n"
        "    for item in items:\n"
        "        x += item + callback()\n"
        "    return callback, x\n"
    )
    compile(same_name_defs_src, "<same_name_defs>", "exec")
    sc_same_name = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 10,
            "kind": "compound_block",
            "source_text": same_name_defs_src,
            "source_code": "for item in items:\n    x += item + callback()\n",
        },
        tree=ast.parse(same_name_defs_src),
    )
    assert "severed_closure_cell" in sc_same_name["control_flow_hazards"]

    # Conditional closure creation
    cond_closure_unreturned_src = (
        "def outer(flag, items):\n"
        "    if flag:\n"
        "        callback = lambda: x\n"
        "    for item in items:\n"
        "        x = item * 2\n"
        "    return items\n"
    )
    compile(cond_closure_unreturned_src, "<cond_closure>", "exec")
    sc_cond_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": cond_closure_unreturned_src,
            "source_code": "for item in items:\n    x = item * 2\n",
            "skip_pre_unit_closures": False,
        },
        tree=ast.parse(cond_closure_unreturned_src),
    )
    sc_cond_lenient = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": cond_closure_unreturned_src,
            "source_code": "for item in items:\n    x = item * 2\n",
            "skip_pre_unit_closures": True,
        },
        tree=ast.parse(cond_closure_unreturned_src),
    )
    assert "severed_closure_cell" in sc_cond_strict["control_flow_hazards"]
    assert "severed_closure_cell" not in sc_cond_lenient["control_flow_hazards"]

    # Star import shadowing `len` prevents treating `len(cb)` as pure built-in
    star_len_src = (
        "from pkg import *\n"
        "def outer(items):\n"
        "    secret = 1\n"
        "    n = len(lambda: secret)\n"
        "    for x in items:\n"
        "        secret += x + n()\n"
        "    return n, secret\n"
    )
    compile(star_len_src, "<star_len>", "exec")
    sc_star_len = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_text": star_len_src,
            "source_code": "for x in items:\n    secret += x + n()\n",
        },
        tree=ast.parse(star_len_src),
    )
    assert "severed_closure_cell" in sc_star_len["control_flow_hazards"]

    # Comprehension filter mutating outer cell via walrus when primary iterable may be empty
    comp_filter_mut_src = (
        "def outer(empty_seq):\n"
        "    acc = 0\n"
        "    cb = lambda: acc\n"
        "    res = [x for x in empty_seq if (acc := acc + x) > 0]\n"
        "    return res, cb\n"
    )
    compile(comp_filter_mut_src, "<comp_filter_mut>", "exec")
    sc_comp_filter = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 56,
            "kind": "comprehension",
            "source_text": comp_filter_mut_src,
            "source_code": "[x for x in empty_seq if (acc := acc + x) > 0]",
        },
        tree=ast.parse(comp_filter_mut_src),
    )
    assert "severed_closure_cell" in sc_comp_filter["control_flow_hazards"]

    # `async for` target is conditional (not unconditionally pre-bound)
    async_for_src = (
        "async def outer(aiter, items):\n"
        "    async for scale in aiter:\n"
        "        pass\n"
        "    cbs = [lambda x, i=i: x * scale + i for i in items]\n"
        "    return cbs\n"
    )
    compile(async_for_src, "<async_for>", "exec")
    sc_async_for = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "start_col": 10,
            "end_col": 55,
            "kind": "comprehension",
            "source_text": async_for_src,
            "source_code": "[lambda x, i=i: x * scale + i for i in items]",
        },
        tree=ast.parse(async_for_src),
    )
    assert "severed_closure_cell" in sc_async_for["control_flow_hazards"]

    # 3-level nested closure chain with middle class definition
    three_level_cls_src = (
        "def outer(cond, items):\n"
        "    if cond:\n"
        "        factor = 3\n"
        "    class Middle:\n"
        "        def build(self):\n"
        "            cbs = [lambda x, i=i: x * factor + i for i in items]\n"
        "            return cbs\n"
        "    return Middle\n"
    )
    compile(three_level_cls_src, "<three_level_cls>", "exec")
    sc_three_cls = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "start_col": 18,
            "end_col": 64,
            "kind": "comprehension",
            "source_text": three_level_cls_src,
            "source_code": "[lambda x, i=i: x * factor + i for i in items]",
        },
        tree=ast.parse(three_level_cls_src),
    )
    assert "severed_closure_cell" in sc_three_cls["control_flow_hazards"]

    # `try: del x; might_raise(); return 1 except ValueError: pass`
    try_del_ret_src = (
        "def outer(might_raise, items):\n"
        "    x = 1\n"
        "    try:\n"
        "        del x\n"
        "        might_raise()\n"
        "        return []\n"
        "    except ValueError:\n"
        "        pass\n"
        "    cbs = [lambda v, i=i: v + x + i for i in items]\n"
        "    return cbs\n"
    )
    compile(try_del_ret_src, "<try_del_ret>", "exec")
    try_del_tree = ast.parse(try_del_ret_src)
    try_del_fn = try_del_tree.body[0]
    assert isinstance(try_del_fn, ast.FunctionDef)
    may_d: set[str] = set()
    def_s, cond_s = _analyze_block_assignment(
        try_del_fn.body[:2], may_deleted=may_d
    )
    assert "x" not in def_s
    assert "x" in cond_s
    assert "x" in may_d
    sc_try_del_ret = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "start_col": 10,
            "end_col": 51,
            "kind": "comprehension",
            "source_text": try_del_ret_src,
            "source_code": "[lambda v, i=i: v + x + i for i in items]",
        },
        tree=try_del_tree,
    )
    assert "severed_closure_cell" in sc_try_del_ret["control_flow_hazards"]

    # _iter_class_body_methods across nested control flow and nested classes
    cls_methods_src = (
        "class Complex:\n"
        "    if True:\n"
        "        def m_if(self): pass\n"
        "    else:\n"
        "        async def m_else(self): pass\n"
        "    try:\n"
        "        def m_try(self): pass\n"
        "    except Exception:\n"
        "        def m_except(self): pass\n"
        "    finally:\n"
        "        def m_finally(self): pass\n"
        "    with ctx():\n"
        "        def m_with(self): pass\n"
        "    for _ in range(1):\n"
        "        def m_for(self): pass\n"
        "    class Nested:\n"
        "        def m_nested_only(self): pass\n"
    )
    if sys.version_info >= (3, 10):
        cls_methods_src += (
            "    match 1:\n"
            "        case 1:\n"
            "            def m_match(self): pass\n"
        )
    cls_methods_node = ast.parse(cls_methods_src).body[0]
    assert isinstance(cls_methods_node, ast.ClassDef)
    found_method_names = [m.name for m in _iter_class_body_methods(cls_methods_node)]
    expected_methods = [
        "m_if",
        "m_else",
        "m_try",
        "m_except",
        "m_finally",
        "m_with",
        "m_for",
    ]
    if sys.version_info >= (3, 10):
        expected_methods.append("m_match")
    assert found_method_names == expected_methods
    assert "m_nested_only" not in found_method_names

    # Statement-delta cached vs. uncached (`use_stmt_deltas=False`) differential test
    delta_diff_src = (
        "def complex_fn(items, cond, mgr):\n"
        "    a = 1\n"
        "    b = []\n"
        "    cb1 = lambda: a + len(b)\n"
        "    if cond:\n"
        "        c = 2\n"
        "    else:\n"
        "        c = 3\n"
        "    for x in items:\n"
        "        b.append(x + c)\n"
        "        try:\n"
        "            d = x * 2\n"
        "        except ValueError:\n"
        "            d = 0\n"
        "    with mgr:\n"
        "        e = a + c\n"
        "    cb2 = lambda: e\n"
        "    return cb1, cb2, b, d\n"
    )
    delta_tree = ast.parse(delta_diff_src)
    delta_fn = delta_tree.body[0]
    assert isinstance(delta_fn, ast.FunctionDef)
    for u_s, u_e in ((5, 8), (9, 14), (15, 16)):
        span_u = _SpanBounds(u_s, u_e)
        info_cached = _analyze_enclosing_fn_cells_uncached(
            delta_fn,
            span_u,
            mod_tree=delta_tree,
            use_stmt_deltas=True,
        )
        info_uncached = _analyze_enclosing_fn_cells_uncached(
            delta_fn,
            span_u,
            mod_tree=delta_tree,
            use_stmt_deltas=False,
        )
        assert info_cached == info_uncached

    # Shared-corpus equivalence between `_analyze_block_assignment` and `_extract_block_definite_bindings_and_term`
    shared_definite_corpus: list[tuple[str, set[str]]] = [
        ("x = 1\ny = 2\n", {"x", "y"}),
        ("x = 1\ndel x\n", set()),
        (
            "if cond:\n    a = 1\n    b = 2\nelse:\n    a = 3\n",
            {"a"},
        ),
        (
            "if cond:\n    a = 1\n    return 0\nelse:\n    b = 2\n",
            {"b"},
        ),
        (
            "x = 1\ntry:\n    del x\n    op()\n    return 1\nexcept ValueError:\n    pass\n",
            set(),
        ),
        (
            "x = 1\ntry:\n    del x\n    op()\n    return 1\nexcept ValueError:\n    x = 2\n",
            {"x"},
        ),
        (
            "with suppress(ValueError):\n    a = 1\n",
            {"a"},
        ),
        (
            "with suppress(ValueError):\n    a = d['k']\n",
            set(),
        ),
        (
            "with open_res() as r:\n    a = 1\n",
            {"r", "a"},
        ),
        (
            "if (w := get_v()):\n    pass\n",
            {"w"},
        ),
    ]
    if sys.version_info >= (3, 10):
        shared_definite_corpus.append(
            (
                "match v:\n    case 1:\n        m = 1\n    case _:\n        m = 2\n",
                {"m"},
            )
        )
    if sys.version_info >= (3, 11):
        shared_definite_corpus.append(
            (
                "try:\n    a = 1\nexcept* ValueError:\n    a = 2\nfinally:\n    f = 3\n",
                {"f"},
            )
        )
    for snippet, expected_def_set in shared_definite_corpus:
        stmts = ast.parse(snippet).body
        def_a, _ = _analyze_block_assignment(stmts)
        def_b, _ = _extract_block_definite_bindings_and_term(stmts, None)
        assert def_a == expected_def_set, f"_analyze_block_assignment mismatch on:\n{snippet}"
        assert def_b == expected_def_set, f"_extract_block_definite_bindings_and_term mismatch on:\n{snippet}"
        assert def_a == def_b

    # Multi-line `_slice_unit_token_lines` boundary-equality test
    ml_lines = ["    x = (\n", "        1 + 2\n", "    )\n"]
    first_bytes_len = len(ml_lines[0].rstrip("\r\n").encode("utf-8"))
    last_bytes_len = len(ml_lines[-1].rstrip("\r\n").encode("utf-8"))
    sliced_inside = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": 8, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_inside[0] == "(\n"
    assert sliced_inside[-1] == "    )"
    sliced_exact_first = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": first_bytes_len, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_exact_first[0] == ml_lines[0]
    sliced_beyond_first = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": first_bytes_len + 10, "end_col": last_bytes_len},
        ml_lines,
    )
    assert sliced_beyond_first[0] == ml_lines[0]
    utf8_ml = ['    "\u00e9\u754c" + (\n', "        1 + 2\n", "    )\n"]
    after_mb_col = len('    "\u00e9\u754c" + '.encode("utf-8"))
    sliced_mb = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": after_mb_col, "end_col": 5},
        utf8_ml,
    )
    assert sliced_mb[0] == "(\n"
    presliced_ml = ["(\n", "        1 + 2\n", "    )\n"]
    assert (
        _slice_unit_token_lines(
            {
                "kind": "complex_expr",
                "start_col": 8,
                "end_col": last_bytes_len,
                "source_code": "".join(presliced_ml),
            },
            presliced_ml,
        )
        == presliced_ml
    )

    if sys.version_info >= (3, 11):
        multi_trystar_src = (
            "keep = 1\n"
            "victim = 2\n"
            "try:\n"
            "    a = 10\n"
            "except* ValueError as eg1:\n"
            "    a = 20\n"
            "    del victim\n"
            "except* TypeError as eg2:\n"
            "    a = 30\n"
            "finally:\n"
            "    fin = 40\n"
        )
        ts_stmts = ast.parse(multi_trystar_src).body
        ts_def_a, ts_cond_a = _analyze_block_assignment(ts_stmts)
        ts_def_b, _ = _extract_block_definite_bindings_and_term(ts_stmts, None)
        assert ts_def_a == {"keep", "fin"}
        assert ts_def_b == {"keep", "fin"}
        assert {"a", "victim"} <= ts_cond_a

    # Cached (`use_cache=True`) vs uncached (`use_cache=False`) helper equivalence
    clear_dataflow_cache()
    helper_eq_tree = ast.parse(
        "def outer(box, items):\n"
        "    x = 1\n"
        "    @dec\n"
        "    def inner(v):\n"
        "        nonlocal x\n"
        "        x += v\n"
        "        box.append(lambda: x + v)\n"
        "        return lambda: x\n"
        "    return inner\n"
    )
    outer_eq = helper_eq_tree.body[0]
    assert isinstance(outer_eq, ast.FunctionDef)
    inner_eq = outer_eq.body[1]
    assert isinstance(inner_eq, ast.FunctionDef)
    eq_ctx = _get_fn_scope_context(outer_eq, mod_tree=helper_eq_tree)
    assert _extract_closure_nonlocal_mutations_and_deletions(
        inner_eq, use_cache=True
    ) == _extract_closure_nonlocal_mutations_and_deletions(
        inner_eq, use_cache=False
    )
    assert _extract_cached_scope_free_reads(
        inner_eq, use_cache=True
    ) == _extract_cached_scope_free_reads(inner_eq, use_cache=False)
    assert _is_self_escaping_def_node(
        inner_eq, use_cache=True
    ) == _is_self_escaping_def_node(inner_eq, use_cache=False)
    assert _is_closure_factory_scope(
        inner_eq, eq_ctx, use_cache=True
    ) == _is_closure_factory_scope(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_returned_closure_refs(
        inner_eq, eq_ctx, use_cache=True
    ) == _extract_fn_returned_closure_refs(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=True
    ) == _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=False
    )

    # Multi-path runtime oracle intersection vs both static definite-binding engines
    multipath_snippet = (
        "def _probe(cond, fail):\n"
        "    always = 1\n"
        "    survives = 2\n"
        "    killed_on_err = 3\n"
        "    if cond:\n"
        "        branch_both = 10\n"
        "        only_true = 11\n"
        "    else:\n"
        "        branch_both = 20\n"
        "    try:\n"
        "        del killed_on_err\n"
        "        if fail:\n"
        "            raise ValueError('boom')\n"
        "        return ('early', always, survives, branch_both)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    return set(locals().keys())\n"
    )
    ns_mp: dict[str, Any] = {}
    exec(compile(multipath_snippet, "<multipath>", "exec"), ns_mp)  # noqa: S102
    probe_fn = ns_mp["_probe"]
    fallthrough_key_sets: list[set[str]] = []
    for c_val in (True, False):
        for f_val in (True, False):
            res_mp = probe_fn(c_val, f_val)
            if isinstance(res_mp, set):
                fallthrough_key_sets.append(res_mp - {"cond", "fail"})
    runtime_definite = set.intersection(*fallthrough_key_sets)
    probe_body = ast.parse(multipath_snippet).body[0].body[:-1]  # type: ignore[attr-defined]
    static_def_1, _ = _analyze_block_assignment(probe_body)
    static_def_2, _ = _extract_block_definite_bindings_and_term(probe_body, None)
    assert static_def_1 == runtime_definite == {"always", "survives", "branch_both"}
    assert static_def_2 == runtime_definite

    # Negative control: forcible pass-by-value extraction past hazard diverges at runtime
    orig_hazard_fn_src = (
        "def run_orig(items):\n"
        "    scale = 2\n"
        "    cbs = [lambda x, i=i: x * scale + i for i in items]\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    forced_extracted_src = (
        "def _forced_helper(items, scale):\n"
        "    return [lambda x, i=i: x * scale + i for i in items]\n"
        "def run_forced(items):\n"
        "    scale = 2\n"
        "    cbs = _forced_helper(items, scale)\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    ns_neg_ctrl: dict[str, Any] = {}
    exec(compile(orig_hazard_fn_src + "\n" + forced_extracted_src, "<neg_ctrl>", "exec"), ns_neg_ctrl)  # noqa: S102
    assert ns_neg_ctrl["run_orig"]([1, 2]) == [31, 32]
    assert ns_neg_ctrl["run_forced"]([1, 2]) == [7, 8]
    assert ns_neg_ctrl["run_orig"]([1, 2]) != ns_neg_ctrl["run_forced"]([1, 2])

    # Stale `unit["source_code"]` fails closed when real source disagrees and cols are None
    real_mod_src = "def f(a):\n    x = a + 1\n    return x\n"
    sc_stale = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_text": real_mod_src,
            "source_code": "completely_unrelated_var = 999\n",
        },
        tree=ast.parse(real_mod_src),
    )
    assert "pathological_ast" in sc_stale["control_flow_hazards"]
    assert sc_stale["is_control_flow_safe"] is False

    # Detached unit AST nodes (including aux_ast_nodes for nested defs/classes) do not pollute caches
    clear_dataflow_cache()
    unit_with_nested_src = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        def helper(v):\n"
        "            def sub_inner(z):\n"
        "                return z + v\n"
        "            return sub_inner(1)\n"
        "        class LocalBox:\n"
        "            def get(self):\n"
        "                return x\n"
        "        total += helper(x) + LocalBox().get()\n"
        "    return total\n"
    )
    mod_tree_clean = ast.parse(unit_with_nested_src)
    mod_nodes = set(ast.walk(mod_tree_clean))
    _ = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 11,
            "kind": "compound_block",
            "source_text": unit_with_nested_src,
            "source_code": (
                "for x in items:\n"
                "    def helper(v):\n"
                "        def sub_inner(z):\n"
                "            return z + v\n"
                "        return sub_inner(1)\n"
                "    class LocalBox:\n"
                "        def get(self):\n"
                "            return x\n"
                "    total += helper(x) + LocalBox().get()\n"
            ),
        },
        tree=mod_tree_clean,
    )
    for p_key in _scope_parent_maps:
        assert p_key in mod_nodes
    for c_key in _scope_cell_info_cache:
        if isinstance(c_key, _ScopeCacheKey):
            for ast_n in (c_key.owner, *c_key.aux_ast_nodes):
                assert ast_n in mod_nodes
        elif isinstance(c_key[0], ast.AST):
            assert c_key[0] in mod_nodes

    # `analyze_unit_variable_scope` explicit `skip_pre_unit_closures` parameter
    lenient_unit_src = (
        "def f(items):\n"
        "    cb = lambda: items\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "    return cb, acc\n"
    )
    u_len = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": lenient_unit_src,
        "source_code": "for x in items:\n    acc = x * 2\n",
    }
    comb_lenient = analyze_unit_variable_scope(
        u_len, dict(u_len), skip_pre_unit_closures=True
    )
    assert comb_lenient["_skip_pre_unit_closures"] is True
    assert comb_lenient["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in comb_lenient["control_flow_hazards"]


def test_scope_cache_key_permitted_parameter_types() -> None:
    """Verifies _ScopeCacheKey enforces permitted immutable AST-free parameter schema."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _make_trusted_scope_cache_key,
    )

    mod = ast.parse("x = 1\n")
    stmt = mod.body[0]

    valid_key = _ScopeCacheKey(
        mod,
        "test_kind",
        params=(
            "str_val",
            42,
            3.14,
            True,
            b"bytes",
            None,
            ("nested_tuple", frozenset({"a", "b", 1})),
        ),
        aux_ast_nodes=(stmt,),
    )
    assert valid_key.owner is mod
    assert valid_key.aux_ast_nodes == (stmt,)

    assert not hasattr(_ScopeCacheKey, "make_trusted")
    trusted_key = _make_trusted_scope_cache_key(
        mod, "trusted_kind", params=("ok", 1)
    )
    assert trusted_key.kind == "trusted_kind"

    for bad_ast_params in (
        (stmt,),
        ((stmt,),),
        (frozenset({stmt}),),
        ([stmt],),
        ({stmt},),
        ({"k": stmt},),
        ({stmt: "v"},),
    ):
        with pytest.raises(ValueError, match="must not contain ast.AST"):
            _ScopeCacheKey(mod, "bad_ast", params=bad_ast_params)

    for bad_type_params in (
        (["a", "b"],),
        ({"k": "v"},),
        ({"a", "b"},),
        (object(),),
    ):
        with pytest.raises(
            TypeError, match="only permits immutable"
        ):
            _ScopeCacheKey(mod, "bad_type", params=bad_type_params)


def test_property_invariants_cache_isolation_monotonicity_equivalence() -> None:
    """Verifies all 7 property-oriented invariants across a diverse AST corpus including nested definitions."""
    import symtable
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _has_invalid_lexical_scope_declarations,
    )

    corpus_programs = [
        (
            "def f1(xs):\n"
            "    a = 1\n"
            "    res = [x + a for x in xs]\n"
            "    return res\n",
            3,
            3,
            "comprehension",
            "res = [x + a for x in xs]\n",
        ),
        (
            "def f2(xs):\n"
            "    a = 1\n"
            "    cb = lambda: a\n"
            "    for x in xs:\n"
            "        a = x + 2\n"
            "    return cb(), a\n",
            4,
            5,
            "compound_block",
            "for x in xs:\n    a = x + 2\n",
        ),
        (
            "def f3(d):\n"
            "    with suppress(KeyError):\n"
            "        v = 10\n"
            "    out = v + 1\n"
            "    return out\n",
            4,
            4,
            "basic_block",
            "out = v + 1\n",
        ),
        (
            "def f4(xs):\n"
            "    acc = 0\n"
            "    for x in xs:\n"
            "        def step_fn(v):\n"
            "            return v + 1\n"
            "        class StepHolder:\n"
            "            def val(self):\n"
            "                return x\n"
            "        acc += step_fn(x) + StepHolder().val()\n"
            "    return acc\n",
            3,
            9,
            "compound_block",
            (
                "for x in xs:\n"
                "    def step_fn(v):\n"
                "        return v + 1\n"
                "    class StepHolder:\n"
                "        def val(self):\n"
                "            return x\n"
                "    acc += step_fn(x) + StepHolder().val()\n"
            ),
        ),
    ]

    for src, u_s, u_e, kind, snippet in corpus_programs:
        tree = ast.parse(src)
        tree_nodes = set(ast.walk(tree))
        line_text = src.splitlines()[u_s - 1]
        s_col = len(line_text) - len(line_text.lstrip())
        end_line_text = src.splitlines()[u_e - 1]
        e_col = len(end_line_text.encode("utf-8"))

        unit_inferred = {
            "start": u_s,
            "end": u_e,
            "kind": kind,
            "source_text": src,
            "source_code": snippet,
        }
        unit_explicit = {
            **unit_inferred,
            "start_col": s_col,
            "end_col": e_col,
        }

        # 1. Cache equivalence (uncached fresh vs cached warm)
        clear_dataflow_cache()
        res_cold = inspect_single_unit_scope(dict(unit_explicit), tree=tree)
        res_warm = inspect_single_unit_scope(dict(unit_explicit), tree=tree)
        assert res_cold == res_warm

        # 3. Defensive copies (mutating returned dict/lists does not corrupt cache)
        res_warm["inputs"].append("__mutated_input__")
        res_warm["control_flow_hazards"].append("__mutated_hazard__")
        res_after_mut = inspect_single_unit_scope(
            dict(unit_explicit), tree=tree
        )
        assert "__mutated_input__" not in res_after_mut["inputs"]
        assert (
            "__mutated_hazard__" not in res_after_mut["control_flow_hazards"]
        )

        # 4. AST isolation (only nodes belonging to `tree` are retained in caches)
        for p_node in _scope_parent_maps:
            assert p_node in tree_nodes
        for c_key in _scope_cell_info_cache:
            for ast_n in (c_key.owner, *c_key.aux_ast_nodes):
                assert ast_n in tree_nodes

        # 5. Strictness monotonicity (safe in strict -> safe in lenient)
        sc_strict = inspect_single_unit_scope(
            {**unit_explicit, "skip_pre_unit_closures": False}, tree=tree
        )
        sc_lenient = inspect_single_unit_scope(
            {**unit_explicit, "skip_pre_unit_closures": True}, tree=tree
        )
        if sc_strict["is_control_flow_safe"]:
            assert sc_lenient["is_control_flow_safe"] is True
        assert set(sc_lenient["control_flow_hazards"]).issubset(
            set(sc_strict["control_flow_hazards"])
        )

        # 6. Span equivalence (explicit column bounds vs inferred column bounds)
        sc_inferred = inspect_single_unit_scope(
            {**unit_inferred, "skip_pre_unit_closures": False}, tree=tree
        )
        assert sc_inferred == sc_strict

        # 2. Cache eviction (evicting tree removes all entries referencing tree_nodes)
        _evict_tree_scopes(tree)
        assert not any(p in tree_nodes for p in _scope_parent_maps)
        assert not any(
            _key_references_scope_nodes(k, tree_nodes)
            for k in _scope_cell_info_cache
        )

    # 7. Lexical equivalence against Python's symtable/compile
    lexical_samples = [
        ("def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2\n", False),
        ("def f():\n    global x\n    def g():\n        nonlocal x\n        x = 2\n", True),
        ("def f(x):\n    nonlocal x\n", True),
        ("def f():\n    class C:\n        global x\n        def m(self):\n            return x\n", False),
    ]
    for lex_src, expected_invalid in lexical_samples:
        lex_tree = ast.parse(lex_src)
        got_invalid = _has_invalid_lexical_scope_declarations(
            lex_tree.body, mod_tree=lex_tree
        )
        symtable_rejected = False
        try:
            symtable.symtable(lex_src, "<lex_test>", "exec")
            compile(lex_src, "<lex_test>", "exec")
        except SyntaxError:
            symtable_rejected = True
        assert got_invalid is expected_invalid is symtable_rejected


def test_cyclic_mutable_containers_in_scope_cache_key() -> None:
    """Verifies self-referential and mutually recursive containers do not hang validation."""
    from pydoppelgangerhunt.fixer._df_cache import _ScopeCacheKey

    owner = ast.parse("x = 1")

    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_list,))

    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_dict,))

    mut_a: list[Any] = []
    mut_b: dict[str, Any] = {"a": mut_a}
    mut_a.append(mut_b)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=((mut_a, mut_b),))

    cyclic_with_ast: list[Any] = []
    cyclic_with_ast.append(cyclic_with_ast)
    cyclic_with_ast.append(owner.body[0])
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _ScopeCacheKey(owner, "test", params=(cyclic_with_ast,))


def test_indexed_cache_eviction_multi_owner_node_reuse_and_central_invariant() -> None:
    """Verifies indexed eviction, multi-owner keys, AST node reuse, and the central cache invariant."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context

    clear_dataflow_cache()

    tree_a = ast.parse("def fa():\n    return 1\n")
    tree_b = ast.parse("def fb():\n    return 2\n")
    tree_c = ast.parse("def fc():\n    return 3\n")
    multi_key = _ScopeCacheKey(
        owner=tree_a.body[0],
        kind="multi_aux",
        params=("tag", 1),
        aux_ast_nodes=(tree_b.body[0], tree_c),
    )
    _scope_cell_info_cache[multi_key] = ("cached_val",)
    assert multi_key in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_b.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_c, set())

    _evict_tree_scopes(tree_b)
    assert multi_key not in _scope_cell_info_cache
    assert multi_key not in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key not in _scope_owner_keys_index.get(tree_c, set())

    reused_fn = tree_a.body[0]
    assert isinstance(reused_fn, ast.FunctionDef)
    ctx_before = _get_fn_scope_context(reused_fn, mod_tree=tree_a)
    assert ctx_before.mod_tree is tree_a
    _evict_tree_scopes(tree_a)
    assert not any(
        _key_references_scope_nodes(k, {reused_fn, tree_a})
        for k in _scope_cell_info_cache
    )
    new_wrapper_tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="math", names=[ast.alias(name="*")], level=0
            ),
            reused_fn,
        ],
        type_ignores=[],
    )
    ctx_after = _get_fn_scope_context(reused_fn, mod_tree=new_wrapper_tree)
    assert ctx_after.mod_tree is new_wrapper_tree
    assert ctx_after.has_star_import is True

    clear_dataflow_cache()
    trees = [
        ast.parse(f"def fn_{i}(x):\n    y = x + {i}\n    return y\n")
        for i in range(4)
    ]
    tree_node_sets = [set(ast.walk(t)) for t in trees]
    all_supported_keys: list[Any] = []
    for idx, t in enumerate(trees):
        fn_node = t.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        stmt_node = fn_node.body[0]
        next_t = trees[(idx + 1) % len(trees)]
        k_ast = fn_node
        k_sck_single = _ScopeCacheKey(
            owner=fn_node, kind="single", params=(idx, ("nested", True))
        )
        k_sck_multi = _ScopeCacheKey(
            owner=stmt_node,
            kind="multi",
            params=(frozenset({"a", "b"}),),
            aux_ast_nodes=(t, next_t.body[0]),
        )
        k_tuple = ("custom_tuple", (fn_node, idx), frozenset({"meta"}))
        k_frozenset = frozenset({"heterogeneous_label", stmt_node, idx})
        for k in (k_ast, k_sck_single, k_sck_multi, k_tuple, k_frozenset):
            _scope_cell_info_cache[k] = (idx, "cell")
            _scope_closures_cache[k] = [(None, {f"v_{idx}"})]
            all_supported_keys.append(k)

    evicted_nodes = tree_node_sets[1]
    _evict_tree_scopes(trees[1])
    for k in all_supported_keys:
        refs_evicted = _key_references_scope_nodes(k, evicted_nodes)
        if refs_evicted:
            assert k not in _scope_cell_info_cache
            assert k not in _scope_closures_cache
        else:
            assert k in _scope_cell_info_cache
            assert k in _scope_closures_cache


def test_scope_cache_key_kind_validation_cross_cache_eviction_and_detached_subtree() -> None:
    """Covers kind validation, cross-cache removal orders, central invariant, and detached subtree eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _extract_key_ast_nodes,
        _extract_value_ast_nodes,
        _make_trusted_scope_cache_key,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_owner_keys_index,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context
    from pydoppelgangerhunt.fixer._df_closures import _collect_scope_closures

    def _assert_central_index_invariant() -> None:
        all_cached_keys = set(_scope_closures_cache.keys()) | set(
            _scope_cell_info_cache.keys()
        )
        for cache in (_scope_closures_cache, _scope_cell_info_cache):
            for k, val in cache.items():
                for node in (
                    *_extract_key_ast_nodes(k),
                    *_extract_value_ast_nodes(val),
                ):
                    assert k in _scope_owner_keys_index.get(node, set())
                    assert node in _scope_key_nodes_index.get(k, set())
        for node, keys in _scope_owner_keys_index.items():
            assert isinstance(node, ast.AST)
            assert keys, "Empty key set must be popped from _scope_owner_keys_index"
            for k in keys:
                assert k in all_cached_keys
        for k, nodes in _scope_key_nodes_index.items():
            assert k in all_cached_keys
            assert nodes

    mod = ast.parse("def f():\n    return 1\n")
    other_node = mod.body[0]
    with pytest.raises(TypeError, match="kind must be a string"):
        _ScopeCacheKey(owner=mod, kind=other_node)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="kind must be a string"):
        _make_trusted_scope_cache_key(owner=mod, kind=other_node)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        _ScopeCacheKey(owner=mod, kind="")
    with pytest.raises(ValueError, match="non-empty string"):
        _make_trusted_scope_cache_key(owner=mod, kind="")

    clear_dataflow_cache()
    shared_tree = ast.parse("def shared_fn():\n    return 42\n")
    shared_fn = shared_tree.body[0]
    shared_key = _ScopeCacheKey(owner=shared_fn, kind="shared_entry")

    # Order A
    _scope_closures_cache[shared_key] = [(None, {"a"})]
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _assert_central_index_invariant()
    del _scope_closures_cache[shared_key]
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    del _scope_cell_info_cache[shared_key]
    assert shared_fn not in _scope_owner_keys_index
    assert shared_key not in _scope_key_nodes_index
    _assert_central_index_invariant()

    # Order B
    _scope_closures_cache[shared_key] = [(None, {"a"})]
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.pop(shared_key)
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.pop(shared_key)
    assert shared_fn not in _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order C
    _scope_closures_cache[shared_key] = [(None, {"a"})]
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_closures_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_cell_info_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order D
    _scope_closures_cache[shared_key] = [(None, {"a"})]
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order E
    _scope_closures_cache[shared_key] = [(None, {"a"})]
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _evict_tree_scopes(shared_tree)
    assert shared_key not in _scope_closures_cache
    assert shared_key not in _scope_cell_info_cache
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Eviction of a detached AST subtree invalidates ancestor cached values
    clear_dataflow_cache()
    nested_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    def inner():\n"
        "        return x\n"
        "    return inner\n"
    )
    outer_fn = nested_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[1]
    assert isinstance(inner_fn, ast.FunctionDef)

    ctx_outer = _get_fn_scope_context(outer_fn, mod_tree=nested_tree)
    closures_outer = _collect_scope_closures(outer_fn, module_tree=nested_tree)
    assert inner_fn in ctx_outer.parent_map
    assert len(closures_outer) == 1
    assert outer_fn in _scope_parent_maps
    _assert_central_index_invariant()

    outer_fn.body.remove(inner_fn)
    _evict_tree_scopes(inner_fn)
    _assert_central_index_invariant()
    assert outer_fn not in _scope_parent_maps
    assert not any(
        inner_fn in _extract_value_ast_nodes(v)
        for v in (
            *_scope_cell_info_cache.values(),
            *_scope_closures_cache.values(),
        )
    )


def test_fn_scope_context_parent_map_immutability_and_enclosing_scope_key_isolation() -> None:
    """Covers MappingProxyType parent_map immutability and non-collision of enclosing_scopes in _make_scope_ctx_cache_key."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
        _make_nested_fn_scope_context,
        _make_scope_ctx_cache_key,
    )

    clear_dataflow_cache()
    src = (
        "def enc_a():\n"
        "    x = 1\n"
        "    def target():\n"
        "        return x\n"
        "    return target\n"
        "\n"
        "def enc_b():\n"
        "    x = 1\n"
        "    def target():\n"
        "        return x\n"
        "    return target\n"
    )
    tree = ast.parse(src)
    enc_a = tree.body[0]
    enc_b = tree.body[1]
    assert isinstance(enc_a, ast.FunctionDef)
    assert isinstance(enc_b, ast.FunctionDef)
    target_a = enc_a.body[1]
    target_b = enc_b.body[1]
    assert isinstance(target_a, ast.FunctionDef)
    assert isinstance(target_b, ast.FunctionDef)

    ctx_a = _get_fn_scope_context(enc_a, mod_tree=tree)
    ctx_b = _get_fn_scope_context(enc_b, mod_tree=tree)

    with pytest.raises(TypeError):
        ctx_a.parent_map[target_a] = enc_b  # type: ignore[index]
    nested_ctx_a = _make_nested_fn_scope_context(
        target_a, frozenset(), scope_ctx=ctx_a
    )
    nested_ctx_b = _make_nested_fn_scope_context(
        target_b, frozenset(), scope_ctx=ctx_b
    )
    with pytest.raises(TypeError):
        nested_ctx_a.parent_map[target_a] = enc_b  # type: ignore[index]

    assert nested_ctx_a.cache_fingerprint() == nested_ctx_b.cache_fingerprint()
    assert nested_ctx_a.enclosing_scopes != nested_ctx_b.enclosing_scopes
    shared_expr = target_a.body[0]
    key_a = _make_scope_ctx_cache_key(
        shared_expr, "expr_free_reads", nested_ctx_a
    )
    key_b = _make_scope_ctx_cache_key(
        shared_expr, "expr_free_reads", nested_ctx_b
    )
    assert key_a != key_b
    assert hash(key_a) != hash(key_b) or key_a.aux_ast_nodes != key_b.aux_ast_nodes


def test_async_subroutine_without_await_suspension_precision_and_monotonicity() -> None:
    """Distinguishes sync/async generators and awaiting coroutines from async functions without suspension points."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _UnitClosureHazardContext,
        _has_severed_closure_cell_hazard,
    )

    base_enclosing_src = (
        "async def outer():\n"
        "    shared = 0\n"
        "    def concurrent_reader():\n"
        "        return shared\n"
        "    r = concurrent_reader\n"
        "    shared = 1\n"
        "    out = shared + 1\n"
        "    return r(), out\n"
    )
    mod_tree = ast.parse(base_enclosing_src)
    sync_stmts = ast.parse("shared = 1\nout = shared + 1\n").body
    await_stmts = ast.parse("shared = await fetch()\nout = shared + 1\n").body
    yield_stmts = ast.parse("shared = 1\nyield shared\n").body

    ctx_async_no_await = _UnitClosureHazardContext(
        candidate_stmts=sync_stmts,
        mod_tree=mod_tree,
        u_start=6,
        u_end=7,
        u_start_col=4,
        u_end_col=20,
        is_subroutine=True,
        unit_kind="linear_block",
        inputs=(),
        outputs=("shared", "out"),
        stores=("shared", "out"),
        deleted_names=frozenset(),
        unit_top_locals=frozenset({"shared", "out"}),
        has_yield=False,
        is_async=True,
        skip_pre_unit_closures=False,
    )
    assert _has_severed_closure_cell_hazard(ctx_async_no_await) is False

    ctx_async_with_await = ctx_async_no_await._replace(
        candidate_stmts=await_stmts
    )
    assert _has_severed_closure_cell_hazard(ctx_async_with_await) is True

    ctx_sync_gen = ctx_async_no_await._replace(
        candidate_stmts=yield_stmts, has_yield=True, is_async=False
    )
    assert _has_severed_closure_cell_hazard(ctx_sync_gen) is True

    ctx_async_gen = ctx_async_no_await._replace(
        candidate_stmts=yield_stmts, has_yield=True, is_async=True
    )
    assert _has_severed_closure_cell_hazard(ctx_async_gen) is True

    for strict_ctx in (
        ctx_async_no_await,
        ctx_async_with_await,
        ctx_sync_gen,
        ctx_async_gen,
    ):
        lenient_ctx = strict_ctx._replace(skip_pre_unit_closures=True)
        strict_res = _has_severed_closure_cell_hazard(strict_ctx)
        lenient_res = _has_severed_closure_cell_hazard(lenient_ctx)
        if lenient_res:
            assert strict_res is True


def test_span_line_index_detached_subtree_eviction_and_recomputation() -> None:
    """Verifies _SpanLineIndex AST node indexing across all 4 maps, detached subtree eviction, and fresh rebuild."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_span_line_index,
        _resolve_unit_col_bounds,
    )

    clear_dataflow_cache()
    src = (
        "def keep_fn(x):\n"
        "    return x + 1\n"
        "\n"
        "def detach_fn(y):\n"
        "    if y > 0:\n"
        "        z = y * 2\n"
        "    return y\n"
    )
    tree = ast.parse(src)
    keep_fn = tree.body[0]
    detach_fn = tree.body[1]
    assert isinstance(detach_fn, ast.FunctionDef)
    inner_if = detach_fn.body[0]
    assert isinstance(inner_if, ast.If)
    inner_assign = inner_if.body[0]

    idx1 = _get_span_line_index(tree)
    span_key = _ScopeCacheKey(tree, "span_line_index")
    assert span_key in _scope_cell_info_cache

    cached_nodes = set(idx1.cached_ast_nodes())
    assert keep_fn in cached_nodes
    assert detach_fn in cached_nodes
    assert inner_if in cached_nodes
    assert inner_assign in cached_nodes
    for node in (keep_fn, detach_fn, inner_if, inner_assign):
        assert span_key in _scope_owner_keys_index.get(node, set())

    tree.body.pop(1)
    _evict_tree_scopes(detach_fn)

    assert span_key not in _scope_cell_info_cache
    assert detach_fn not in _scope_owner_keys_index
    assert inner_if not in _scope_owner_keys_index
    assert inner_assign not in _scope_owner_keys_index

    idx2 = _get_span_line_index(tree)
    assert idx2 is not idx1
    rebuilt_nodes = set(idx2.cached_ast_nodes())
    assert keep_fn in rebuilt_nodes
    assert detach_fn not in rebuilt_nodes
    assert inner_if not in rebuilt_nodes
    assert inner_assign not in rebuilt_nodes
    assert 4 not in idx2.by_start_line
    assert 7 not in idx2.by_end_line
    assert 6 not in idx2.stmts_by_start_line
    assert 6 not in idx2.stmts_by_end_line

    assert isinstance(keep_fn, ast.FunctionDef)
    cols = _resolve_unit_col_bounds([keep_fn.body[0]], tree, 2, 2, None, None)
    assert cols == (4, 16)


def test_detached_ast_line1_enclosing_scope_recovery_and_builtin_shadowing() -> None:
    """Verifies detached AST enclosing-scope recovery for line-1 nodes across sync/async/class/shadowed scopes."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context_uncached
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures_uncached,
        _extract_scope_local_names,
        _resolve_detached_enclosing_scopes,
    )

    mod_one_line = ast.parse(
        "def outer(items): sum = custom_sum; return sum(x for x in items)\n"
    )
    detached_ret = ast.parse("return sum(x for x in items)\n")
    enc_one_line = _resolve_detached_enclosing_scopes(detached_ret, mod_one_line)
    assert mod_one_line.body[0] in enc_one_line
    locals_one_line = _extract_scope_local_names(detached_ret, module_tree=mod_one_line)
    assert "sum" in locals_one_line
    closures_one_line = _collect_scope_closures_uncached(
        detached_ret, module_tree=mod_one_line
    )
    assert any("items" in reads for _, reads in closures_one_line)

    mod_async_one_line = ast.parse(
        "async def outer(items): sum = custom_sum; return sum(x for x in items)\n"
    )
    detached_async_ret = ast.parse("return sum(x for x in items)\n")
    enc_async = _resolve_detached_enclosing_scopes(
        detached_async_ret, mod_async_one_line
    )
    assert mod_async_one_line.body[0] in enc_async
    locals_async = _extract_scope_local_names(
        detached_async_ret, module_tree=mod_async_one_line
    )
    assert "sum" in locals_async
    closures_async = _collect_scope_closures_uncached(
        detached_async_ret, module_tree=mod_async_one_line
    )
    assert any("items" in reads for _, reads in closures_async)

    mod_multiline = ast.parse(
        "def outer(items, custom_sum):\n"
        "    sum = custom_sum\n"
        "    result = sum(x for x in items)\n"
        "    return result\n"
    )
    detached_stmt = ast.parse("result = sum(x for x in items)\n")
    assert detached_stmt.body[0].lineno == 1
    locals_multi = _extract_scope_local_names(detached_stmt, module_tree=mod_multiline)
    assert "sum" in locals_multi
    ctx_multi = _get_fn_scope_context_uncached(detached_stmt, mod_tree=mod_multiline)
    assert "sum" in ctx_multi.shadowed_names
    closures_multi = _collect_scope_closures_uncached(
        detached_stmt, module_tree=mod_multiline
    )
    assert any("items" in reads for _, reads in closures_multi)

    mod_nested_cls = ast.parse(
        "def outer(items, custom_sum, custom_max):\n"
        "    sum = custom_sum\n"
        "    class Inner:\n"
        "        max = custom_max\n"
        "        r1 = sum(x for x in items)\n"
        "        r2 = max(y for y in items)\n"
        "    return Inner\n"
    )
    detached_cls_unit = ast.parse(
        "r1 = sum(x for x in items)\nr2 = max(y for y in items)\n"
    )
    locals_cls = _extract_scope_local_names(
        detached_cls_unit, module_tree=mod_nested_cls
    )
    assert "sum" in locals_cls
    assert "max" in locals_cls
    closures_cls = _collect_scope_closures_uncached(
        detached_cls_unit, module_tree=mod_nested_cls
    )
    assert len(closures_cls) == 2

    detached_modified = ast.parse("other = sum(z for z in items)\n")
    locals_ambig = _extract_scope_local_names(
        detached_modified, module_tree=mod_multiline
    )
    assert "sum" in locals_ambig

    mod_with_top = ast.parse(
        "def helper(custom_sum):\n"
        "    sum = custom_sum\n"
        "    return sum\n"
        "top_res = sum(x for x in top_items)\n"
    )
    detached_top = ast.parse("top_res = sum(x for x in top_items)\n")
    locals_top = _extract_scope_local_names(detached_top, module_tree=mod_with_top)
    assert "sum" not in locals_top


def test_tracked_scope_cache_stmt_deltas_incremental_mutation_indexing() -> None:
    """Verifies that incremental mutations to cached stmt_deltas dictionaries update _scope_owner_keys_index."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _evict_tree_scopes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells

    clear_dataflow_cache()
    src = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    y = x + 2\n"
        "    def sub():\n"
        "        return x + y\n"
        "    return sub()\n"
    )
    tree = ast.parse(src)
    outer_fn = tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    stmt_x = outer_fn.body[0]
    stmt_y = outer_fn.body[1]

    _analyze_enclosing_fn_cells(
        outer_fn, (4, 5, 4, 16), mod_tree=tree
    )
    deltas_key = _ScopeCacheKey(
        owner=outer_fn, kind="stmt_deltas", aux_ast_nodes=(tree,)
    )
    assert deltas_key in _scope_cell_info_cache
    cached_deltas = _scope_cell_info_cache[deltas_key]
    assert isinstance(cached_deltas, _TrackedCachedDict)
    assert (stmt_x, True) in cached_deltas
    assert (stmt_y, True) in cached_deltas

    assert deltas_key in _scope_owner_keys_index[stmt_x]
    assert deltas_key in _scope_owner_keys_index[stmt_y]

    cached_deltas.pop((stmt_x, True))
    # O(1) incremental index does not rescan on pop; index is pruned at eviction time
    assert deltas_key in _scope_owner_keys_index.get(stmt_x, set())
    assert deltas_key in _scope_owner_keys_index[stmt_y]
    _evict_tree_scopes(tree)
    assert deltas_key not in _scope_owner_keys_index.get(stmt_x, set())
    assert deltas_key not in _scope_owner_keys_index.get(stmt_y, set())

    tree2 = ast.parse("def f(z):\n    s1 = z + 1\n    s2 = s1 + 1\n")
    fn2 = tree2.body[0]
    assert isinstance(fn2, ast.FunctionDef)
    s1, s2 = fn2.body[0], fn2.body[1]
    manual_key = _ScopeCacheKey(owner=fn2, kind="stmt_deltas")
    manual_deltas: dict[tuple[ast.AST, bool], Any] = _get_or_compute_scope_cached(
        _scope_cell_info_cache, manual_key, dict
    )
    assert s1 not in _scope_owner_keys_index
    manual_deltas.setdefault((s1, False), None)
    assert manual_key in _scope_owner_keys_index[s1]
    manual_deltas[(s2, True)] = None
    assert manual_key in _scope_owner_keys_index[s2]

    tree3 = ast.parse("def g(w):\n    st = w * 2\n")
    fn3 = tree3.body[0]
    assert isinstance(fn3, ast.FunctionDef)
    st = fn3.body[0]
    plain_key = _ScopeCacheKey(owner=fn3, kind="stmt_deltas")
    plain_dict: dict[tuple[ast.AST, bool], Any] = {}
    _scope_cell_info_cache[plain_key] = plain_dict
    _scope_cell_info_cache[plain_key][(st, False)] = None
    assert plain_key in _scope_owner_keys_index[st]

    _evict_tree_scopes(st)
    assert plain_key not in _scope_cell_info_cache


def test_extract_unit_source_code_empty_and_whitespace_source_text_fallback(
    tmp_path: Path,
) -> None:
    """Verifies that extract_unit_source_code falls back to file_source when source_text is empty/whitespace."""
    from pydoppelgangerhunt.reporters import extract_unit_source_code

    unit_empty_src = {
        "name": "example",
        "start": 1,
        "end": 2,
        "source_text": "",
        "file_source": "def example():\n    return 42\n",
    }
    assert extract_unit_source_code(unit_empty_src) == [
        "def example():\n",
        "    return 42\n",
    ]

    unit_ws_src = {
        "name": "example",
        "start": 1,
        "end": 2,
        "source_text": "   \n\t  ",
        "file_source": "def example():\n    return 42\n",
    }
    assert extract_unit_source_code(unit_ws_src) == [
        "def example():\n",
        "    return 42\n",
    ]

    disk_file = tmp_path / "disk_mod.py"
    disk_file.write_text("def on_disk():\n    return 99\n", encoding="utf-8")
    unit_explicit_empty = {
        "name": "on_disk",
        "file": str(disk_file),
        "start": 1,
        "end": 2,
        "source_text": "",
    }
    assert extract_unit_source_code(unit_explicit_empty, repo_root=str(tmp_path)) == [
        "# Source for on_disk lines 1-2\n"
    ]


def test_trusted_scope_cache_key_immutability_and_enclosing_chain_and_modes() -> None:
    """Verifies trusted key validation, same-line/multiline-header enclosing chains, and mode propagation."""
    from pydoppelgangerhunt.fixer._df_cache import _make_trusted_scope_cache_key
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _find_enclosing_function_chain,
        _make_scope_ctx_cache_key,
    )
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    dummy = ast.parse("x = 1").body[0]

    with pytest.raises(TypeError, match="immutable"):
        _make_trusted_scope_cache_key(dummy, "kind", params=([1, 2],))
    with pytest.raises(TypeError, match="immutable"):
        _make_scope_ctx_cache_key(dummy, "kind", None, extra_params=({"a": 1},))
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _make_trusted_scope_cache_key(dummy, "kind", params=(dummy,))

    chain_src = (
        "@(\n"
        "    dec\n"
        ")\n"
        "def outer(\n"
        "    a: int,\n"
        "    b: int,\n"
        ") -> int:\n"
        "    def mid(c: int) -> int:\n"
        "        def inner(d: int) -> int: return a + b + c + d\n"
        "        return inner(c)\n"
        "    return mid(a)\n"
    )
    chain_tree = ast.parse(chain_src)
    chain = _find_enclosing_function_chain(
        chain_tree, 9, 9, u_start_col=34, u_end_col=54
    )
    assert [getattr(n, "name", "") for _, _, n in chain] == [
        "outer",
        "mid",
        "inner",
    ]
    hdr_chain = _find_enclosing_function_chain(chain_tree, 5, 6)
    assert hdr_chain == []

    mode_src = (
        "def f1(n):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    x = n + 1\n"
        "    yield x\n"
        "    return x\n"
        "\n"
        "def f2(n):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    x = n + 1\n"
        "    yield x\n"
        "    return x\n"
    )
    u1 = {
        "name": "u1",
        "kind": "compound_block",
        "start": 4,
        "end": 5,
        "source_text": mode_src,
    }
    u2 = {
        "name": "u2",
        "kind": "compound_block",
        "start": 12,
        "end": 13,
        "source_text": mode_src,
    }

    s1_strict = inspect_single_unit_scope({**u1, "skip_pre_unit_closures": False})
    s2_strict = inspect_single_unit_scope({**u2, "skip_pre_unit_closures": False})
    s1_lenient = ensure_unit_scope_closure_mode(u1, s1_strict, True)
    assert s1_strict["_skip_pre_unit_closures"] is False
    assert s1_lenient["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" in s1_strict["control_flow_hazards"]
    assert "severed_closure_cell" not in s1_lenient["control_flow_hazards"]

    s2_lenient = inspect_single_unit_scope({**u2, "skip_pre_unit_closures": True})
    pair_strict = analyze_unit_variable_scope(
        u1, u2, scope1=s1_lenient, scope2=s2_lenient, skip_pre_unit_closures=False
    )
    assert pair_strict["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in pair_strict["control_flow_hazards"]

    assert (
        resolve_clone_pair_outputs(
            u1, u2, scope1=s1_lenient, scope2=s2_lenient, closure_strictness="strict"
        )
        is None
    )
    outs_lenient = resolve_clone_pair_outputs(
        u1, u2, scope1=s1_strict, scope2=s2_strict, closure_strictness="lenient"
    )
    assert outs_lenient == (["x"], ["x"])

    helper_strict = synthesize_shared_helper_code(
        u1,
        u2,
        helper_name="_helper",
        scope1=s1_lenient,
        scope2=s2_lenient,
        closure_strictness="strict",
    )
    assert helper_strict == ""

    pair_lenient = analyze_unit_variable_scope(
        u1, u2, scope1=s1_lenient, scope2=s2_lenient, skip_pre_unit_closures=True
    )
    helper_lenient = synthesize_shared_helper_code(
        u1,
        u2,
        helper_name="_helper",
        scope=pair_lenient,
        scope1=s1_lenient,
        scope2=s2_lenient,
        outputs=["x"],
        outputs2=["x"],
        closure_strictness="lenient",
    )
    assert "def _helper(" in helper_lenient

    assert (
        resolve_clone_generator_subroutine_outputs(
            u1,
            u2,
            s1_lenient,
            s2_lenient,
            source_text1=mode_src,
            source_text2=mode_src,
            closure_strictness="strict",
        )
        is None
    )
    gen_outs_lenient = resolve_clone_generator_subroutine_outputs(
        u1,
        u2,
        s1_strict,
        s2_strict,
        source_text1=mode_src,
        source_text2=mode_src,
        closure_strictness="lenient",
    )
    assert gen_outs_lenient == (["x"], ["x"])


def test_high_and_medium_severity_architectural_invariants() -> None:
    """Directly exercises cache-key isolation, O(|T|) auxiliary cache eviction, lock reentrancy, context-sensitive closure caching, _SuppressProvenance, and unified definite-binding analysis."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedAuxScopeCache,
        _aux_node_to_scope_owners,
        _aux_scope_owner_to_nodes,
        _build_downstream_cache_key,
        _downstream_cache_lock,
        _evict_tree_scopes,
        _get_scope_parent_map,
        _get_scope_stmts_by_end_lineno,
        _get_scope_stmts_by_start_lineno,
        _is_reentrant_cache_lock,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_parent_maps,
        _scope_stmts_by_end,
        _scope_stmts_by_start,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _collect_unit_nonlocal_mutators,
        _extract_expr_closure_and_load_info,
        _get_fn_scope_context,
        _is_non_closure_builtin_call,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
        _resolve_detached_enclosing_scopes,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _analyze_block_assignment,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    clear_dataflow_cache()

    # 1. _collect_unit_nonlocal_mutators with nested defs and real mod_tree never leaks ephemeral ast.Module
    nested_unit_src = (
        "def outer(items):\n"
        "    state = 0\n"
        "    for x in items:\n"
        "        def outer_helper(v):\n"
        "            nonlocal state\n"
        "            state += v\n"
        "            def inner_helper():\n"
        "                return state\n"
        "            return inner_helper\n"
        "        class Box:\n"
        "            def run(self):\n"
        "                nonlocal state\n"
        "                state += 1\n"
        "        outer_helper(x)()\n"
        "    return state\n"
    )
    mod_tree = ast.parse(nested_unit_src)
    mod_nodes = set(ast.walk(mod_tree))
    outer_fn = mod_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    for_stmt = outer_fn.body[1]
    assert isinstance(for_stmt, ast.For)
    mutators = _collect_unit_nonlocal_mutators(
        [for_stmt], mod_tree=mod_tree
    )
    assert any(m.name == "outer_helper" for m in mutators)
    for c_key in _scope_cell_info_cache:
        for ast_n in (c_key.owner, *c_key.aux_ast_nodes):
            assert ast_n in mod_nodes

    # Repeat call hits warm cache without adding new keys
    keys_before = set(_scope_cell_info_cache.keys())
    mutators_warm = _collect_unit_nonlocal_mutators(
        [for_stmt], mod_tree=mod_tree
    )
    assert mutators_warm == mutators
    assert set(_scope_cell_info_cache.keys()) == keys_before

    # 2. _TrackedAuxScopeCache & O(|T|) child-subtree eviction for parent_maps and stmt line maps
    clear_dataflow_cache()
    aux_tree = ast.parse(
        "def top_fn(x):\n"
        "    def child_fn(y):\n"
        "        return y + 1\n"
        "    return child_fn(x)\n"
    )
    top_fn = aux_tree.body[0]
    assert isinstance(top_fn, ast.FunctionDef)
    child_fn = top_fn.body[0]
    assert isinstance(child_fn, ast.FunctionDef)
    assert isinstance(_scope_parent_maps, _TrackedAuxScopeCache)
    _ = _get_scope_parent_map(aux_tree)
    _ = _get_scope_stmts_by_start_lineno(top_fn)
    _ = _get_scope_stmts_by_end_lineno(top_fn)
    assert aux_tree in _scope_parent_maps
    assert top_fn in _scope_stmts_by_start
    assert top_fn in _scope_stmts_by_end
    assert child_fn in _aux_node_to_scope_owners
    assert top_fn in _aux_scope_owner_to_nodes

    # Detach child_fn and evict ONLY child_fn: owning aux caches (aux_tree, top_fn) are evicted in O(|T|)
    top_fn.body.remove(child_fn)
    _evict_tree_scopes(child_fn)
    assert aux_tree not in _scope_parent_maps
    assert top_fn not in _scope_stmts_by_start
    assert top_fn not in _scope_stmts_by_end
    assert child_fn not in _aux_node_to_scope_owners
    assert not _aux_scope_owner_to_nodes

    # 3. Runtime lock reentrancy verification
    assert _is_reentrant_cache_lock(_downstream_cache_lock) is True
    assert _is_reentrant_cache_lock(threading.RLock()) is True
    assert _is_reentrant_cache_lock(threading.Lock()) is False

    # 4. _is_non_closure_builtin_call fails closed when scope_ctx is None & callees stay in loaded_names
    len_call = ast.parse("len(x)", mode="eval").body
    assert isinstance(len_call, ast.Call)
    assert _is_non_closure_builtin_call(len_call, None) is False
    unshadowed_ctx = _FnScopeContext(
        fn_locals=frozenset(),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset(),
        has_star_import=False,
    )
    assert _is_non_closure_builtin_call(len_call, unshadowed_ctx) is True

    direct_call = ast.parse("custom_callee(arg_val)", mode="eval").body
    _, _, loads_without_out = _extract_expr_closure_and_load_info(
        [direct_call], called_names_out=None
    )
    assert {"custom_callee", "arg_val"} <= loads_without_out

    # 5. Enclosing-scope sensitivity in _get_fn_scope_context and _collect_scope_closures, plus _DetachedDumpIndex
    clear_dataflow_cache()
    enc_src = (
        "def outer_shadow(list):\n"
        "    def inner(items):\n"
        "        return list(x for x in items)\n"
        "    return inner\n"
        "\n"
        "def outer_clean():\n"
        "    def inner(items):\n"
        "        return list(x for x in items)\n"
        "    return inner\n"
    )
    enc_tree = ast.parse(enc_src)
    outer_sh = enc_tree.body[0]
    outer_cl = enc_tree.body[1]
    assert isinstance(outer_sh, ast.FunctionDef)
    assert isinstance(outer_cl, ast.FunctionDef)
    shared_inner = outer_sh.body[0]
    assert isinstance(shared_inner, ast.FunctionDef)

    ctx_sh = _get_fn_scope_context(
        shared_inner, mod_tree=None, enclosing_scopes=(outer_sh,)
    )
    ctx_cl = _get_fn_scope_context(
        shared_inner, mod_tree=None, enclosing_scopes=(outer_cl,)
    )
    assert "list" in ctx_sh.shadowed_names
    assert "list" not in ctx_cl.shadowed_names

    clos_sh = _collect_scope_closures(
        shared_inner, module_tree=None, enclosing_scopes=(outer_sh,)
    )
    clos_cl = _collect_scope_closures(
        shared_inner, module_tree=None, enclosing_scopes=(outer_cl,)
    )
    assert len(clos_sh) == 1  # shadowed list -> lazy genexp closure
    assert len(clos_cl) == 0  # unshadowed list -> eager genexp

    detached_inner = ast.parse("return list(x for x in items)\n")
    _ = _resolve_detached_enclosing_scopes(detached_inner, enc_tree)
    dump_key = _ScopeCacheKey(enc_tree, "detached_dump_index")
    assert dump_key in _scope_cell_info_cache
    _evict_tree_scopes(enc_tree)
    assert dump_key not in _scope_cell_info_cache
    assert not _scope_closures_cache
    assert not _scope_key_nodes_index

    # 6. _SuppressProvenance dataclass & compound-statement / post-with shadowed builtin detection
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
    assert isinstance(prov, AbstractSet)
    assert "cm" in prov.suppress_names
    assert "open" in prov.shadowed_builtins
    bound_prov, _ = _extract_block_definite_bindings_and_term(
        prov_fn.body[:-2], None, suppress_aliases=prov
    )
    assert "n" not in bound_prov
    assert "m" not in bound_prov

    # 7. Unified definite-binding engine with extra_non_suppressing_ctx_managers
    custom_safe_stmts = ast.parse(
        "with SafeTransaction():\n"
        "    committed = compute_value()\n"
    ).body
    def_without, cond_without = _analyze_block_assignment(custom_safe_stmts)
    def_with, cond_with = _analyze_block_assignment(
        custom_safe_stmts,
        extra_non_suppressing_ctx_managers=frozenset({"SafeTransaction"}),
    )
    assert "committed" not in def_without
    assert "committed" in cond_without
    assert "committed" in def_with
    assert "committed" not in cond_with

    # 8. _FnScopeContext.cache_fingerprint distinguishes empty vs populated parent_map
    fp_empty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    fp_nonempty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({prov_fn: prov_tree}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    assert fp_empty != fp_nonempty
    assert fp_empty[-1] is False
    assert fp_nonempty[-1] is True

    # 9. _build_downstream_cache_key includes unit_source_hash when columns are None
    dk_1 = _build_downstream_cache_key(
        prov_src,
        {"start": 2, "end": 2, "source_code": "a = 1"},
        2,
        2,
        {"a"},
        False,
    )
    dk_2 = _build_downstream_cache_key(
        prov_src,
        {"start": 2, "end": 2, "source_code": "b = 2"},
        2,
        2,
        {"a"},
        False,
    )
    assert dk_1 != dk_2

    # 10. _node_earliest_start_pos locates leading `@` line across multi-line decorator expressions
    ml_dec_src = (
        "x = 1\n"
        "@(\n"
        "    decorator_factory\n"
        ")\n"
        "def decorated_fn():\n"
        "    return 1\n"
    )
    ml_dec_tree = ast.parse(ml_dec_src)
    dec_fn_node = ml_dec_tree.body[1]
    assert (
        _node_earliest_start_pos(
            dec_fn_node, source_lines=ml_dec_src.splitlines()
        )
        == (2, 0)
    )
