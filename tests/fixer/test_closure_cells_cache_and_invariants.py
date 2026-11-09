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


def test_heterogeneous_frozenset_eviction_and_scope_cache_key() -> None:
    """Covers heterogeneous frozenset cache eviction, nested AST nodes, and _ScopeCacheKey."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

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
        (
            "nested_comp",
            (_SpanBounds(1, 4), frozenset({"meta", listcomp_node})),
        ),
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

    info_before = _analyze_enclosing_fn_cells(outer_fn, (2, 3, 4, 54), mod_tree=evict_mod)
    _evict_tree_scopes(evict_mod)
    info_after = _analyze_enclosing_fn_cells(outer_fn, (2, 3, 4, 54), mod_tree=evict_mod)
    assert info_before == info_after


def test_fail_closed_policy_for_unresolved_source_spans() -> None:
    """Covers fail-closed policy for unresolved source spans and incomplete AST metadata."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _is_node_inside_span,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _UnitClosureHazardContext,
        _has_severed_closure_cell_hazard,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _has_unresolved_span_bounds,
        _resolve_unit_col_bounds,
    )

    dup_line_src = "def f(xs):\n" "    a = 1; a = 1\n" "    return [lambda: a for _ in xs]\n"
    dup_line_tree = ast.parse(dup_line_src)
    cand_assign = ast.parse("a = 1\n").body
    assert _resolve_unit_col_bounds(cand_assign, dup_line_tree, 2, 2, None, None) == (None, None)
    assert _has_unresolved_span_bounds(dup_line_tree.body[0], 2, 2, None, None) is True
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
        "def f(xs):\n" "    res = [x + 1 for x in xs] + [x + 1 for x in xs]\n" "    return res\n"
    )
    subexpr_tree = ast.parse(subexpr_line_src)
    cand_comp_expr = ast.parse("[x + 1 for x in xs]\n").body
    assert _resolve_unit_col_bounds(cand_comp_expr, subexpr_tree, 2, 2, None, None) == (None, None)
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
        "def f(xs):\n" "    a = 1\n" "    b = a + 2\n" "    return [lambda: b for _ in xs]\n"
    )
    corrupt_fn = corrupt_tree.body[0]
    assert isinstance(corrupt_fn, ast.FunctionDef)
    corrupt_stmt = corrupt_fn.body[1]
    corrupt_stmt.end_lineno = None
    assert _is_node_inside_span(corrupt_stmt, _SpanBounds(3, 3, 4, 13)) is False
    assert _has_unresolved_span_bounds(corrupt_fn, 3, 3, 4, 13) is True
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

    corrupt_col_tree = ast.parse("def f(xs):\n" "    a = 1\n" "    b = a + 2\n" "    return b\n")
    corrupt_col_fn = corrupt_col_tree.body[0]
    assert isinstance(corrupt_col_fn, ast.FunctionDef)
    corrupt_col_fn.body[1].end_col_offset = None
    assert _is_node_inside_span(corrupt_col_fn.body[1], _SpanBounds(3, 3, 4, 13)) is False
    assert _has_unresolved_span_bounds(corrupt_col_fn, 3, 3, 4, 13) is True


def test_pep3104_and_pep227_lexical_scope_rules() -> None:
    """Covers PEP 3104 and PEP 227 lexical-scope rules across invalid and valid multi-level variants."""
    import symtable

    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_closure_nonlocal_mutations_and_deletions,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _has_invalid_lexical_scope_declarations,
    )

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
        assert _has_invalid_lexical_scope_declarations(inv_tree.body, mod_tree=inv_tree) is True
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
    assert _has_invalid_lexical_scope_declarations(v1_tree.body, mod_tree=v1_tree) is False
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
    assert _has_invalid_lexical_scope_declarations(v2_tree.body, mod_tree=v2_tree) is False

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
    assert _has_invalid_lexical_scope_declarations(v4_tree.body, mod_tree=v4_tree) is False
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


def test_lazy_genexp_scalar_builtins_and_carrier_mutation_summaries(
    tmp_path: Path,
) -> None:
    """Covers lazy/eager genexps without scope_ctx, scalar built-ins vs custom dunders, carrier mutations vs rebinding, and _EnclosingFnCellInfo summaries."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _extract_comprehension_deferred_closure_info,
        _extract_lazy_genexp_walrus_names,
        _is_lazy_genexp,
        _is_non_closure_scalar_expr,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

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

    top_eager_sum = ast.parse("sum((w := x) + a for x in items)", mode="eval").body
    assert _extract_lazy_genexp_walrus_names(top_eager_sum, None) == set()

    nested_lazy_comp = ast.parse("[((w := x) + a for x in row) for row in rows]", mode="eval").body
    assert isinstance(nested_lazy_comp, ast.ListComp)
    assert _extract_lazy_genexp_walrus_names(nested_lazy_comp, None) == {"w"}
    c_reads_lazy, c_muts_lazy = _extract_comprehension_deferred_closure_info(nested_lazy_comp, None)
    assert "a" in c_reads_lazy
    assert "w" in c_muts_lazy

    # C-enforced scalar built-ins vs unconstrained custom dunders
    unshadowed_scope_ctx = _FnScopeContext(
        fn_locals=frozenset(),
        parent_map={},
        shadowed_names=frozenset(),
        has_star_import=False,
    )
    assert _is_non_closure_scalar_expr(ast.parse("len(items)", mode="eval").body, None) is False
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
        assert _is_non_closure_scalar_expr(expr_node, unshadowed_scope_ctx) is True, scalar_code

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
    cf_info = _analyze_enclosing_fn_cells(cf_tree.body[0], (13, 13, 8, 35), mod_tree=cf_tree)
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
    res_pair = resolve_clone_pair_outputs(u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True)
    assert res_pair.outputs is None
    assert res_pair.skip_reason == "severed closure cell"
    res_gen = resolve_clone_generator_subroutine_outputs(
        u1_e2e, u2_e2e, sc1_e2e, sc2_e2e, return_details=True
    )
    assert res_gen.outputs is None
    assert res_gen.skip_reason == "severed closure cell"
    assert (
        generate_refactoring_patch(
            [(1.0, u1_e2e, u2_e2e)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
        == ""
    )


def test_multilevel_concurrent_sibling_closure_sharing() -> None:
    """Covers multi-level concurrent sibling closure cell sharing detection."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _check_sibling_closure_sharing,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _build_enclosing_chain_cell_infos,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )

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
        "source_code": ("for step in (1, 2):\n" "    value += step\n" "    yield value\n"),
    }
    p1_shadowed_scope = inspect_single_unit_scope(p1_shadowed_unit, tree=p1_shadowed_tree)
    assert "severed_closure_cell" not in p1_shadowed_scope["control_flow_hazards"]
    p1_sh_chain = _find_enclosing_function_chain(p1_shadowed_tree, 7, 9)
    p1_sh_infos = _build_enclosing_chain_cell_infos(
        p1_sh_chain,
        7,
        9,
        None,
        None,
        frozenset({"value"}),
        mod_tree=p1_shadowed_tree,
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


def test_with_and_async_with_definite_bindings() -> None:
    """Covers definite-binding analysis inside with and async with bodies."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _find_enclosing_chain_definite_pre_bound,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )

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
            "with open('f1') as a, open('f2') as b:\n    c = 1\n",
            {"a", "b", "c"},
            False,
        ),
        (
            "with cm1() as a, cm2() as b:\n    c = 1\n",
            {"a"},
            False,
        ),
        (
            "with cm1() as a, cm2() as b:\n    c = a + b\n",
            {"a"},
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
            parsed.body[0].body if isinstance(parsed.body[0], ast.AsyncFunctionDef) else parsed.body
        )
        got_bound, got_term = _extract_block_definite_bindings_and_term(
            stmts, None, strict_with_suppression=True
        )
        assert got_bound == expected_bound, f"Mismatch for:\n{code_str}"
        assert got_term is expected_term, f"Term mismatch for:\n{code_str}"


def test_class_definitions_escape_and_shared_ast_context_cache() -> None:
    """Covers class definition escape classification and cached closure metadata context validation on shared AST nodes."""
    from pydoppelgangerhunt.fixer._df_cache import _get_scope_parent_map
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _collect_escaped_and_executed_intra_unit_defs,
        _collect_unit_nonlocal_mutators,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _is_closure_factory_scope,
    )

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
    muts_unref = _collect_unit_nonlocal_mutators(ast.parse(str(cls_unref_unit["source_code"])).body)
    assert any(m.name == "LocalHelper" and m.is_escaped_def is False for m in muts_unref)
    sc_cls_unref = inspect_single_unit_scope(cls_unref_unit, tree=cls_unref_tree)
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
        assert (
            "severed_closure_cell" in sc_esc["control_flow_hazards"]
        ), f"Expected hazard for: {extra_stmt}"

    for cls_header_or_body in [
        "@dec\nclass LocalHelper:\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(Base):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper(metaclass=Meta):\n    def mutate(self):\n        nonlocal count\n        count += 1\n",
        "class LocalHelper:\n    nonlocal count\n    count += 1\n",
    ]:
        c5_stmts = ast.parse("count = 0\n" + cls_header_or_body).body
        c5_muts = _collect_unit_nonlocal_mutators(c5_stmts)
        assert any(m.name == "LocalHelper" and m.is_escaped_def is True for m in c5_muts)
        _, c5_esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(c5_stmts, frozenset())
        assert any(isinstance(s, ast.ClassDef) and id(s) in c5_esc_ids for s in c5_stmts)

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
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_a) == (
        set(),
        set(),
    )
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_b) == (
        set(),
        {"any"},
    )


def test_definite_binding_oracle_and_rebinding_matrix() -> None:
    """Covers table-driven definite-binding runtime oracle comparison, lexical class scopes, rebinding matrix, and carrier targets."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_closure_nonlocal_mutations_and_deletions_uncached,
        _extract_fn_outer_carrier_targets_uncached,
        _propagate_sibling_closures,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
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
        got_invalid = _has_invalid_lexical_scope_declarations(tree_a.body, mod_tree=tree_a)
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
    expected_rebind_muts = {"a", "b", "d", "e", "h"}
    if sys.version_info >= (3, 10):
        rebinding_fn.body.append(ast.parse("match a:\n    case f:\n        pass\n").body[0])
        expected_rebind_muts.add("f")
    if sys.version_info >= (3, 12):
        rebinding_fn.body.append(ast.parse("type g = int\n").body[0])
        expected_rebind_muts.add("g")
    b_muts, b_dels = _extract_closure_nonlocal_mutations_and_deletions_uncached(rebinding_fn)
    assert b_muts == frozenset(expected_rebind_muts)
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
    chain_d = _find_enclosing_function_chain(chain_col_tree, 3, 3, u_start_col=23, u_end_col=35)
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


def test_pep227_class_level_global_isolation_and_sibling_carriers() -> None:
    """Verifies PEP 227 class-level global isolation and sibling class method carriers."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_fn_outer_carrier_targets_uncached,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_visitors import (
        _ClassScopeVisitor,
        _extract_nested_scope_free_reads_with_nodes,
    )

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
    c_targets = _extract_fn_outer_carrier_targets_uncached(cls_c, frozenset(), outer_ctx)
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


def test_ensure_unit_scope_closure_mode_legacy_and_explicit_contracts() -> None:
    """Verifies schema-versioned precomputed scopes, fail-closed unmarked scopes without source, and source_text=None handling."""
    from pydoppelgangerhunt.fixer.scope import (
        SCOPE_SCHEMA_VERSION,
        ensure_unit_scope_closure_mode,
    )

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
    # Unmarked legacy scope without loadable source must fail closed
    unmarked_no_source = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        unmarked_precomputed,
        skip_pre_unit_closures=False,
    )
    assert unmarked_no_source["is_control_flow_safe"] is False
    assert "severed_closure_cell" in unmarked_no_source["control_flow_hazards"]
    assert "unavailable_source_for_closure_mode" in unmarked_no_source["closure_hazard_reasons"]
    assert unmarked_no_source["scope_schema_version"] == SCOPE_SCHEMA_VERSION

    # Empty source_lines: [] must not count as having source
    empty_lines_res = ensure_unit_scope_closure_mode(
        {
            "file": "nonexistent_unavailable.py",
            "start": 1,
            "end": 2,
            "source_lines": [],
        },
        unmarked_precomputed,
        skip_pre_unit_closures=False,
    )
    assert empty_lines_res["is_control_flow_safe"] is False
    assert "severed_closure_cell" in empty_lines_res["control_flow_hazards"]

    # Unit with source_text: None and valid string source_code must not raise AttributeError
    none_source_text_res = ensure_unit_scope_closure_mode(
        {
            "file": "nonexistent_unavailable.py",
            "start": 1,
            "end": 1,
            "source_text": None,
            "source_code": "b = a + 1\n",
        },
        {
            **unmarked_precomputed,
            "skip_pre_unit_closures": True,
            "scope_schema_version": SCOPE_SCHEMA_VERSION,
        },
        skip_pre_unit_closures=False,
    )
    assert none_source_text_res["is_control_flow_safe"] is True
    assert none_source_text_res["skip_pre_unit_closures"] is False
    assert none_source_text_res["scope_schema_version"] == SCOPE_SCHEMA_VERSION

    # Properly stamped v2 scope with matching mode is preserved even without source
    stamped_strict = {
        **unmarked_precomputed,
        "skip_pre_unit_closures": False,
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    preserved = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        stamped_strict,
        skip_pre_unit_closures=False,
    )
    assert preserved["is_control_flow_safe"] is True
    assert "severed_closure_cell" not in preserved["control_flow_hazards"]

    marked_lenient = {
        **unmarked_precomputed,
        "skip_pre_unit_closures": True,
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    strict_mismatch_res = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        marked_lenient,
        skip_pre_unit_closures=False,
    )
    assert strict_mismatch_res["is_control_flow_safe"] is False
    assert "severed_closure_cell" in strict_mismatch_res["control_flow_hazards"]
    assert strict_mismatch_res["skip_pre_unit_closures"] is False


def test_same_name_definitions_and_conditional_closures_and_star_imports() -> None:
    """Verifies same-name definitions across branches, conditional closures, and star-import shadowing."""
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


def test_comprehension_filter_walrus_async_for_and_three_level_class_closures() -> None:
    """Verifies comprehension filter walrus mutations, async-for targets, and 3-level class closures."""
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


def test_try_del_return_fallthrough_and_iter_class_body_methods() -> None:
    """Verifies try-del-return handler fallthrough and _iter_class_body_methods."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _iter_class_body_methods,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

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
    def_s, cond_s = _analyze_block_assignment(try_del_fn.body[:2], may_deleted=may_d)
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
            "    match 1:\n" "        case 1:\n" "            def m_match(self): pass\n"
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


def test_stmt_delta_equivalence_and_shared_definite_binding_corpus() -> None:
    """Verifies statement-delta equivalence and definite-binding corpus parity."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

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
        assert (
            def_b == expected_def_set
        ), f"_extract_block_definite_bindings_and_term mismatch on:\n{snippet}"
        assert def_a == def_b

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


def test_slice_unit_token_lines_multiline_and_utf8_boundaries() -> None:
    """Verifies multi-line _slice_unit_token_lines boundary and UTF-8 handling."""
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

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
        {
            "kind": "complex_expr",
            "start_col": first_bytes_len,
            "end_col": last_bytes_len,
        },
        ml_lines,
    )
    assert sliced_exact_first[0] == ml_lines[0]
    sliced_beyond_first = _slice_unit_token_lines(
        {
            "kind": "complex_expr",
            "start_col": first_bytes_len + 10,
            "end_col": last_bytes_len,
        },
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


def test_cached_vs_uncached_helper_equivalence_and_multipath_runtime_oracle() -> None:
    """Verifies cached vs uncached helper equivalence and multi-path runtime oracle intersection."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _is_self_escaping_def_node,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

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
    ) == _extract_closure_nonlocal_mutations_and_deletions(inner_eq, use_cache=False)
    assert _extract_cached_scope_free_reads(
        inner_eq, use_cache=True
    ) == _extract_cached_scope_free_reads(inner_eq, use_cache=False)
    assert _is_self_escaping_def_node(inner_eq, use_cache=True) == _is_self_escaping_def_node(
        inner_eq, use_cache=False
    )
    assert _is_closure_factory_scope(inner_eq, eq_ctx, use_cache=True) == _is_closure_factory_scope(
        inner_eq, eq_ctx, use_cache=False
    )
    assert _extract_fn_returned_closure_refs(
        inner_eq, eq_ctx, use_cache=True
    ) == _extract_fn_returned_closure_refs(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=True
    ) == _extract_fn_outer_carrier_targets(inner_eq, frozenset(), eq_ctx, use_cache=False)

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
    exec(
        compile(
            orig_hazard_fn_src + "\n" + forced_extracted_src,
            "<neg_ctrl>",
            "exec",
        ),
        ns_neg_ctrl,
    )  # noqa: S102
    assert ns_neg_ctrl["run_orig"]([1, 2]) == [31, 32]
    assert ns_neg_ctrl["run_forced"]([1, 2]) == [7, 8]
    assert ns_neg_ctrl["run_orig"]([1, 2]) != ns_neg_ctrl["run_forced"]([1, 2])


def test_detached_unit_ast_isolation_and_stale_source_code_fail_closed() -> None:
    """Verifies stale source_code fails closed and detached unit AST nodes never pollute caches."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _scope_cell_info_cache,
        _scope_parent_maps,
    )

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
    assert "stale_source_code" in sc_stale["control_flow_hazards"]
    assert sc_stale["is_control_flow_safe"] is False

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
    comb_lenient = analyze_unit_variable_scope(u_len, dict(u_len), skip_pre_unit_closures=True)
    assert comb_lenient["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in comb_lenient["control_flow_hazards"]


def test_scope_cache_key_permitted_parameter_types() -> None:
    """Verifies _ScopeCacheKey enforces permitted immutable AST-free parameter schema, bool/int non-collision, and hidden AST detection."""
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
    trusted_key = _make_trusted_scope_cache_key(mod, "trusted_kind", params=("ok", 1))
    assert trusted_key.kind == "trusted_kind"
    assert trusted_key == _ScopeCacheKey(mod, "trusted_kind", params=("ok", 1))

    # 1 and True (and 0 and False) must not collide in key params
    key_int_1 = _ScopeCacheKey(mod, "bool_int", params=(1,))
    key_bool_true = _ScopeCacheKey(mod, "bool_int", params=(True,))
    key_int_0 = _ScopeCacheKey(mod, "bool_int", params=(0,))
    key_bool_false = _ScopeCacheKey(mod, "bool_int", params=(False,))
    assert key_int_1 != key_bool_true
    assert key_int_0 != key_bool_false
    assert len({key_int_1, key_bool_true, key_int_0, key_bool_false}) == 4

    for bad_ast_params in (
        (stmt,),
        ((stmt,),),
        (frozenset({stmt}),),
        ([stmt],),
        ({stmt},),
        ({"k": stmt},),
        ({stmt: "v"},),
        (object(), stmt),
        ([object(), (stmt,)],),
    ):
        with pytest.raises(ValueError, match="must not contain ast.AST"):
            _ScopeCacheKey(mod, "bad_ast", params=bad_ast_params)

    for bad_type_params in (
        (["a", "b"],),
        ({"k": "v"},),
        ({"a", "b"},),
        (object(),),
    ):
        with pytest.raises(TypeError, match="only permits immutable"):
            _ScopeCacheKey(mod, "bad_type", params=bad_type_params)


def test_structural_invariants_across_deterministic_corpus() -> None:
    """Verifies all 7 structural invariants across a deterministic AST corpus including nested definitions."""
    import symtable
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _has_invalid_lexical_scope_declarations,
    )

    corpus_programs = [
        (
            "def f1(xs):\n" "    a = 1\n" "    res = [x + a for x in xs]\n" "    return res\n",
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
        res_after_mut = inspect_single_unit_scope(dict(unit_explicit), tree=tree)
        assert "__mutated_input__" not in res_after_mut["inputs"]
        assert "__mutated_hazard__" not in res_after_mut["control_flow_hazards"]

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
        assert not any(_key_references_scope_nodes(k, tree_nodes) for k in _scope_cell_info_cache)

    # 7. Lexical equivalence against Python's symtable/compile
    lexical_samples = [
        (
            "def f():\n    x = 1\n    def g():\n        nonlocal x\n        x = 2\n",
            False,
        ),
        (
            "def f():\n    global x\n    def g():\n        nonlocal x\n        x = 2\n",
            True,
        ),
        ("def f(x):\n    nonlocal x\n", True),
        (
            "def f():\n    class C:\n        global x\n        def m(self):\n            return x\n",
            False,
        ),
    ]
    for lex_src, expected_invalid in lexical_samples:
        lex_tree = ast.parse(lex_src)
        got_invalid = _has_invalid_lexical_scope_declarations(lex_tree.body, mod_tree=lex_tree)
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
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import _FrozenClosureList

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
        _key_references_scope_nodes(k, {reused_fn, tree_a}) for k in _scope_cell_info_cache
    )
    new_wrapper_tree = ast.Module(
        body=[
            ast.ImportFrom(module="math", names=[ast.alias(name="*")], level=0),
            reused_fn,
        ],
        type_ignores=[],
    )
    ctx_after = _get_fn_scope_context(reused_fn, mod_tree=new_wrapper_tree)
    assert ctx_after.mod_tree is new_wrapper_tree
    assert ctx_after.has_star_import is True

    clear_dataflow_cache()
    trees = [ast.parse(f"def fn_{i}(x):\n    y = x + {i}\n    return y\n") for i in range(4)]
    tree_node_sets = [set(ast.walk(t)) for t in trees]
    all_supported_keys: list[Any] = []
    for idx, t in enumerate(trees):
        fn_node = t.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        stmt_node = fn_node.body[0]
        next_t = trees[(idx + 1) % len(trees)]
        k_ast = fn_node
        k_sck_single = _ScopeCacheKey(owner=fn_node, kind="single", params=(idx, ("nested", True)))
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
            _scope_closures_cache[k] = _FrozenClosureList([(None, frozenset({f"v_{idx}"}))])
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
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_owner_keys_index,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _FrozenClosureList,
        _collect_scope_closures,
    )

    def _assert_central_index_invariant() -> None:
        all_cached_keys = set(_scope_closures_cache.keys()) | set(_scope_cell_info_cache.keys())
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
    with pytest.raises(ValueError, match="non-empty string"):
        _ScopeCacheKey(owner=mod, kind="")

    clear_dataflow_cache()
    shared_tree = ast.parse("def shared_fn():\n    return 42\n")
    shared_fn = shared_tree.body[0]
    shared_key = _ScopeCacheKey(owner=shared_fn, kind="shared_entry")
    frozen_closures = _FrozenClosureList([(None, frozenset({"a"}))])

    # Order A
    _scope_closures_cache[shared_key] = frozen_closures
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
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.pop(shared_key)
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.pop(shared_key)
    assert shared_fn not in _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order C
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_closures_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_cell_info_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order D
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order E
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _evict_tree_scopes(shared_tree)
    assert shared_key not in _scope_closures_cache
    assert shared_key not in _scope_cell_info_cache
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Eviction of a detached AST subtree invalidates ancestor cached values (including value-less entries)
    clear_dataflow_cache()
    nested_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    def inner():\n"
        "        return x + ext_var\n"
        "    return inner\n"
    )
    outer_fn = nested_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[1]
    assert isinstance(inner_fn, ast.FunctionDef)

    ctx_outer = _get_fn_scope_context(outer_fn, mod_tree=nested_tree)
    closures_outer = _collect_scope_closures(outer_fn, module_tree=nested_tree)
    free_reads_outer = _extract_cached_scope_free_reads(outer_fn, use_cache=True)
    assert "ext_var" in free_reads_outer
    free_reads_key = _ScopeCacheKey(outer_fn, "free_reads")
    assert free_reads_key in _scope_cell_info_cache
    assert inner_fn in ctx_outer.parent_map
    assert len(closures_outer) == 1
    assert outer_fn in _scope_parent_maps
    _assert_central_index_invariant()

    outer_fn.body.remove(inner_fn)
    _evict_tree_scopes(inner_fn)
    _assert_central_index_invariant()
    assert outer_fn not in _scope_parent_maps
    assert free_reads_key not in _scope_cell_info_cache
    assert not any(
        inner_fn in _extract_value_ast_nodes(v)
        for v in (
            *_scope_cell_info_cache.values(),
            *_scope_closures_cache.values(),
        )
    )


def test_extract_value_ast_nodes_mandatory_protocol_and_tracked_dict_copy() -> None:
    """Verifies mandatory cached_ast_nodes() protocol and independent _TrackedCachedDict copies."""
    import copy
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _evict_tree_scopes,
        _extract_value_ast_nodes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
    )

    class _UnindexedCustomValue:
        def __init__(self, node: ast.AST) -> None:
            self.node = node

    class _ProperCustomValue:
        def __init__(self, node: ast.AST) -> None:
            self._node = node

        def cached_ast_nodes(self) -> tuple[ast.AST, ...]:
            return (self._node,)

    mod = ast.parse("x = 1\n")
    stmt = mod.body[0]
    bad_val = _UnindexedCustomValue(stmt)
    with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
        _extract_value_ast_nodes(bad_val)
    with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
        _scope_cell_info_cache[_ScopeCacheKey(mod, "bad_val")] = bad_val
    with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _ScopeCacheKey(mod, "bad_val_graceful"),
            lambda: bad_val,
        )
    assert _ScopeCacheKey(mod, "bad_val_graceful") not in _scope_cell_info_cache

    good_val = _ProperCustomValue(stmt)
    assert list(_extract_value_ast_nodes(good_val)) == [stmt]

    clear_dataflow_cache()
    td_key = _ScopeCacheKey(mod, "tracked_dict_test")
    _scope_cell_info_cache[td_key] = {"init": 1}
    tracked = _scope_cell_info_cache[td_key]
    assert isinstance(tracked, _TrackedCachedDict)
    cp1 = tracked.copy()
    cp2 = copy.copy(tracked)
    assert isinstance(cp1, _TrackedCachedDict)
    assert isinstance(cp2, _TrackedCachedDict)
    other_mod = ast.parse("y = 2\n")
    other_stmt = other_mod.body[0]
    cp1[other_stmt] = 99
    cp2[other_stmt] = 100
    _evict_tree_scopes(other_mod)
    assert td_key in _scope_cell_info_cache


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
    nested_ctx_a = _make_nested_fn_scope_context(target_a, frozenset(), scope_ctx=ctx_a)
    nested_ctx_b = _make_nested_fn_scope_context(target_b, frozenset(), scope_ctx=ctx_b)
    with pytest.raises(TypeError):
        nested_ctx_a.parent_map[target_a] = enc_b  # type: ignore[index]

    assert nested_ctx_a.cache_fingerprint() == nested_ctx_b.cache_fingerprint()
    assert nested_ctx_a.enclosing_scopes != nested_ctx_b.enclosing_scopes
    shared_expr = target_a.body[0]
    key_a = _make_scope_ctx_cache_key(shared_expr, "expr_free_reads", nested_ctx_a)
    key_b = _make_scope_ctx_cache_key(shared_expr, "expr_free_reads", nested_ctx_b)
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

    ctx_async_with_await = ctx_async_no_await._replace(candidate_stmts=await_stmts)
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
    from pydoppelgangerhunt.fixer._df_span_index import (
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

    # _SpanLineIndex indexes only the root via _ScopeCacheKey.owner to avoid per-node memory overhead
    assert idx1.cached_ast_nodes() == ()
    assert span_key in _scope_owner_keys_index.get(tree, set())
    assert detach_fn not in _scope_owner_keys_index

    tree.body.pop(1)
    _evict_tree_scopes(detach_fn)

    assert span_key not in _scope_cell_info_cache
    assert detach_fn not in _scope_owner_keys_index
    assert inner_if not in _scope_owner_keys_index
    assert inner_assign not in _scope_owner_keys_index

    idx2 = _get_span_line_index(tree)
    assert idx2 is not idx1
    assert idx2.cached_ast_nodes() == ()
    assert 4 not in idx2.by_start_line
    assert 7 not in idx2.by_end_line
    assert 6 not in idx2.stmts_by_start_line
    assert 6 not in idx2.stmts_by_end_line

    assert isinstance(keep_fn, ast.FunctionDef)
    cols = _resolve_unit_col_bounds([keep_fn.body[0]], tree, 2, 2, None, None)
    assert cols == (4, 16)


def test_coordinate_enclosing_scope_recovery_and_symtable_cross_check() -> None:
    """Verifies coordinate-based enclosing-scope resolution, explicit enclosing_scopes propagation, and symtable scope analysis."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context_uncached,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures_uncached,
        _extract_scope_local_names,
        _resolve_enclosing_scopes_by_coords,
    )
    from tests.fixer._closure_test_helpers import _extract_symtable_scope_info

    mod_one_line = ast.parse("def outer(items): sum = custom_sum; return sum(x for x in items)\n")
    outer_one = mod_one_line.body[0]
    assert isinstance(outer_one, ast.FunctionDef)
    ret_one = outer_one.body[1]
    enc_one_line = _resolve_enclosing_scopes_by_coords(ret_one, mod_one_line)
    assert outer_one in enc_one_line
    locals_one_line = _extract_scope_local_names(ret_one, module_tree=mod_one_line)
    assert "sum" in locals_one_line
    closures_one_line = _collect_scope_closures_uncached(outer_one, module_tree=mod_one_line)
    assert any("items" in reads for _, reads in closures_one_line)

    mod_async_one_line = ast.parse(
        "async def outer(items): sum = custom_sum; return sum(x for x in items)\n"
    )
    outer_async = mod_async_one_line.body[0]
    assert isinstance(outer_async, ast.AsyncFunctionDef)
    ret_async = outer_async.body[1]
    enc_async = _resolve_enclosing_scopes_by_coords(ret_async, mod_async_one_line)
    assert outer_async in enc_async
    locals_async = _extract_scope_local_names(ret_async, module_tree=mod_async_one_line)
    assert "sum" in locals_async
    closures_async = _collect_scope_closures_uncached(outer_async, module_tree=mod_async_one_line)
    assert any("items" in reads for _, reads in closures_async)

    mod_multiline = ast.parse(
        "def outer(items, custom_sum):\n"
        "    sum = custom_sum\n"
        "    result = sum(x for x in items)\n"
        "    return result\n"
    )
    outer_multi = mod_multiline.body[0]
    assert isinstance(outer_multi, ast.FunctionDef)
    detached_stmt = ast.parse("result = sum(x for x in items)\n")
    locals_multi = _extract_scope_local_names(
        detached_stmt,
        module_tree=mod_multiline,
        enclosing_scopes=(outer_multi,),
    )
    assert "sum" in locals_multi
    ctx_multi = _get_fn_scope_context_uncached(
        detached_stmt,
        mod_tree=mod_multiline,
        enclosing_scopes=(outer_multi,),
    )
    assert "sum" in ctx_multi.shadowed_names
    closures_multi = _collect_scope_closures_uncached(
        detached_stmt,
        module_tree=mod_multiline,
        enclosing_scopes=(outer_multi,),
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
    outer_cls_fn = mod_nested_cls.body[0]
    assert isinstance(outer_cls_fn, ast.FunctionDef)
    inner_cls = outer_cls_fn.body[1]
    assert isinstance(inner_cls, ast.ClassDef)
    detached_cls_unit = ast.parse("r1 = sum(x for x in items)\nr2 = max(y for y in items)\n")
    locals_cls = _extract_scope_local_names(
        detached_cls_unit,
        module_tree=mod_nested_cls,
        enclosing_scopes=(outer_cls_fn, inner_cls),
        include_classes=True,
    )
    assert "sum" in locals_cls
    assert "max" in locals_cls
    closures_cls = _collect_scope_closures_uncached(
        detached_cls_unit,
        module_tree=mod_nested_cls,
        enclosing_scopes=(outer_cls_fn, inner_cls),
    )
    assert len(closures_cls) == 2

    sym_info = _extract_symtable_scope_info(
        "def outer(a):\n"
        "    x = a + 1\n"
        "    global g_var\n"
        "    def inner(b):\n"
        "        nonlocal x\n"
        "        return x + b + g_var\n"
        "    return inner\n"
    )
    assert sym_info
    outer_sym = sym_info[("outer", 1)]
    inner_sym = sym_info[("inner", 4)]
    assert "x" in outer_sym["locals"]
    assert "a" in outer_sym["parameters"]
    assert "g_var" in outer_sym["globals"]
    assert "x" in inner_sym["frees"]
    assert "b" in inner_sym["parameters"]
    assert not _extract_symtable_scope_info("def broken(:\n")


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
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

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

    _analyze_enclosing_fn_cells(outer_fn, (4, 5, 4, 16), mod_tree=tree)
    deltas_key = _ScopeCacheKey(owner=outer_fn, kind="stmt_deltas", aux_ast_nodes=(tree,))
    assert deltas_key in _scope_cell_info_cache
    cached_deltas = _scope_cell_info_cache[deltas_key]
    assert isinstance(cached_deltas, _TrackedCachedDict)
    assert (stmt_x, True) in cached_deltas
    assert (stmt_y, True) in cached_deltas

    assert deltas_key in _scope_owner_keys_index[stmt_x]
    assert deltas_key in _scope_owner_keys_index[stmt_y]

    cached_deltas.pop((stmt_x, True))
    assert deltas_key not in _scope_owner_keys_index.get(stmt_x, set())
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
    """Verifies that extract_unit_source_code falls back to file_source or disk when source_text is empty/whitespace."""
    from pydoppelgangerhunt.reporters import (
        extract_unit_source_code,
        extract_unit_source_lines_with_state,
    )

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
    lines_with_state, is_pre_sliced = extract_unit_source_lines_with_state(unit_empty_src)
    assert lines_with_state == ["def example():\n", "    return 42\n"]
    assert bool(is_pre_sliced) is False
    assert is_pre_sliced.is_placeholder is False

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
        "def on_disk():\n",
        "    return 99\n",
    ]

    unit_missing_disk = {
        "name": "missing_fn",
        "file": str(tmp_path / "nonexistent.py"),
        "start": 1,
        "end": 2,
        "source_text": "",
    }
    assert extract_unit_source_code(unit_missing_disk, repo_root=str(tmp_path)) == [
        "# Source for missing_fn lines 1-2\n"
    ]


def test_trusted_scope_cache_key_immutability_and_enclosing_chain_and_modes() -> None:
    """Verifies trusted key construction, same-line/multiline-header enclosing chains, and mode propagation."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _make_trusted_scope_cache_key,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _make_scope_ctx_cache_key,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    dummy = ast.parse("x = 1").body[0]

    # _ScopeCacheKey validates params, whereas _make_trusted_scope_cache_key skips validation on hot internal paths
    with pytest.raises(TypeError, match="immutable"):
        _ScopeCacheKey(dummy, "kind", params=([1, 2],))
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _ScopeCacheKey(dummy, "kind", params=(dummy,))
    fast_key = _make_trusted_scope_cache_key(dummy, "kind", params=("a", 1))
    ctx_key = _make_scope_ctx_cache_key(dummy, "kind", None, extra_params=("a", 1))
    assert fast_key == _ScopeCacheKey(dummy, "kind", params=("a", 1))
    assert ctx_key.kind == "kind"

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
    chain = _find_enclosing_function_chain(chain_tree, 9, 9, u_start_col=34, u_end_col=54)
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
    assert s1_strict["skip_pre_unit_closures"] is False
    assert s1_lenient["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" in s1_strict["control_flow_hazards"]
    assert "severed_closure_cell" not in s1_lenient["control_flow_hazards"]

    s2_lenient = inspect_single_unit_scope({**u2, "skip_pre_unit_closures": True})
    pair_strict = analyze_unit_variable_scope(
        u1,
        u2,
        scope1=s1_lenient,
        scope2=s2_lenient,
        skip_pre_unit_closures=False,
    )
    assert pair_strict["skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in pair_strict["control_flow_hazards"]

    assert (
        resolve_clone_pair_outputs(
            u1,
            u2,
            scope1=s1_lenient,
            scope2=s2_lenient,
            closure_strictness="strict",
        )
        is None
    )
    outs_lenient = resolve_clone_pair_outputs(
        u1,
        u2,
        scope1=s1_strict,
        scope2=s2_strict,
        closure_strictness="lenient",
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
        u1,
        u2,
        scope1=s1_lenient,
        scope2=s2_lenient,
        skip_pre_unit_closures=True,
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


def test_collect_unit_nonlocal_mutators_ephemeral_module_isolation() -> None:
    """Verifies _collect_unit_nonlocal_mutators with nested defs and real mod_tree never leaks ephemeral ast.Module."""
    from pydoppelgangerhunt.fixer._df_cache import _scope_cell_info_cache
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_unit_nonlocal_mutators,
    )

    clear_dataflow_cache()
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
    mutators = _collect_unit_nonlocal_mutators([for_stmt], mod_tree=mod_tree)
    assert any(m.name == "outer_helper" for m in mutators)
    for c_key in _scope_cell_info_cache:
        for ast_n in (c_key.owner, *c_key.aux_ast_nodes):
            assert ast_n in mod_nodes

    keys_before = set(_scope_cell_info_cache.keys())
    mutators_warm = _collect_unit_nonlocal_mutators([for_stmt], mod_tree=mod_tree)
    assert mutators_warm == mutators
    assert set(_scope_cell_info_cache.keys()) == keys_before


def test_parent_map_and_stmt_line_map_detached_child_eviction() -> None:
    """Verifies child-subtree eviction for parent_maps and statement line maps via ancestor walk."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _get_scope_parent_map,
        _get_scope_stmts_by_end_lineno,
        _get_scope_stmts_by_start_lineno,
        _scope_parent_maps,
        _scope_stmts_by_end,
        _scope_stmts_by_start,
    )

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
    _ = _get_scope_parent_map(aux_tree)
    _ = _get_scope_stmts_by_start_lineno(top_fn)
    _ = _get_scope_stmts_by_end_lineno(top_fn)
    assert aux_tree in _scope_parent_maps
    assert top_fn in _scope_stmts_by_start
    assert top_fn in _scope_stmts_by_end

    top_fn.body.remove(child_fn)
    _evict_tree_scopes(child_fn)
    assert aux_tree not in _scope_parent_maps
    assert top_fn not in _scope_stmts_by_start
    assert top_fn not in _scope_stmts_by_end


def test_cache_lock_reentrancy_and_builtin_call_shadowing() -> None:
    """Verifies lock reentrancy and _is_non_closure_builtin_call fail-closed behavior."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _downstream_cache_lock,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _extract_expr_closure_and_load_info,
        _is_non_closure_builtin_call,
    )

    assert isinstance(_downstream_cache_lock, type(threading.RLock()))

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


def test_enclosing_scope_sensitivity_and_parent_map_reverse_index_eviction() -> None:
    """Verifies enclosing-scope sensitivity in _get_fn_scope_context and _collect_scope_closures, plus O(1) reverse-index parent-map eviction and sibling isolation."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _get_scope_parent_map,
        _node_to_parent_map_roots,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
    )

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

    ctx_sh = _get_fn_scope_context(shared_inner, mod_tree=None, enclosing_scopes=(outer_sh,))
    ctx_cl = _get_fn_scope_context(shared_inner, mod_tree=None, enclosing_scopes=(outer_cl,))
    assert "list" in ctx_sh.shadowed_names
    assert "list" not in ctx_cl.shadowed_names
    # _FnScopeContext.cached_ast_nodes() only indexes cache_aux_nodes, not every AST node in parent_map
    assert set(ctx_sh.cached_ast_nodes()) == set(ctx_sh.cache_aux_nodes())

    clos_sh = _collect_scope_closures(shared_inner, module_tree=None, enclosing_scopes=(outer_sh,))
    clos_cl = _collect_scope_closures(shared_inner, module_tree=None, enclosing_scopes=(outer_cl,))
    assert len(clos_sh) == 1  # shadowed list -> lazy genexp closure
    assert len(clos_cl) == 0  # unshadowed list -> eager genexp

    _ = _get_scope_parent_map(enc_tree)
    _ = _get_scope_parent_map(outer_sh)
    _ = _get_scope_parent_map(outer_cl)
    clean_inner = outer_cl.body[0]
    assert isinstance(clean_inner, ast.FunctionDef)
    assert enc_tree in _node_to_parent_map_roots.get(shared_inner, set())
    assert outer_sh in _node_to_parent_map_roots.get(shared_inner, set())
    assert outer_cl in _node_to_parent_map_roots.get(clean_inner, set())

    # Detaching a statement inside outer_cl evicts outer_cl and enc_tree parent maps,
    # while preserving unrelated sibling outer_sh!
    outer_cl.body.remove(clean_inner)
    _evict_tree_scopes(clean_inner)
    assert outer_cl not in _scope_parent_maps
    assert enc_tree not in _scope_parent_maps
    assert outer_sh in _scope_parent_maps

    _evict_tree_scopes(enc_tree)
    assert not _scope_closures_cache
    assert not _scope_cell_info_cache
    assert not _scope_key_nodes_index
    assert not _scope_parent_maps
    assert not _node_to_parent_map_roots


def test_suppress_provenance_set_algebra_and_with_classification_modes() -> None:
    """Verifies _SuppressProvenance structured operations, heuristic vs pedantic unknown-with classification, and extra_non_suppressing_ctx_managers."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _analyze_block_assignment,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

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
        "with SafeTransaction():\n" "    committed = compute_value()\n"
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
        extra_non_suppressing_ctx_managers=frozenset({"SafeTransaction"}),
        strict_with_suppression=True,
    )
    assert "committed" not in def_ped_without
    assert "committed" in cond_ped_without
    assert "committed" in def_ped_with
    assert "committed" not in cond_ped_with


def test_fn_scope_context_fingerprint_downstream_cache_key_and_multiline_decorator_start() -> None:
    """Verifies _FnScopeContext.cache_fingerprint, _build_downstream_cache_key, and _node_earliest_start_pos."""
    from pydoppelgangerhunt.fixer._df_cache import _build_downstream_cache_key
    from pydoppelgangerhunt.fixer._df_cell_carriers import _FnScopeContext
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos

    ml_dec_src = (
        "x = 1\n" "@(\n" "    decorator_factory\n" ")\n" "def decorated_fn():\n" "    return 1\n"
    )
    ml_dec_tree = ast.parse(ml_dec_src)
    dec_fn_node = ml_dec_tree.body[1]
    assert _node_earliest_start_pos(dec_fn_node, source_lines=ml_dec_src.splitlines()) == (2, 0)

    fp_empty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    fp_nonempty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({dec_fn_node: ml_dec_tree}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    assert fp_empty != fp_nonempty
    assert fp_empty[-1] is False
    assert fp_nonempty[-1] is True

    dk_1 = _build_downstream_cache_key(
        ml_dec_src,
        {"start": 2, "end": 2, "source_code": "a = 1"},
        2,
        2,
        {"a"},
        False,
    )
    dk_2 = _build_downstream_cache_key(
        ml_dec_src,
        {"start": 2, "end": 2, "source_code": "b = 2"},
        2,
        2,
        {"a"},
        False,
    )
    assert dk_1 != dk_2


def test_extract_scope_local_names_class_visibility_flag() -> None:
    """Verifies _extract_scope_local_names obeys include_classes per PEP 227 class scope rules."""
    from pydoppelgangerhunt.fixer._df_closures import _extract_scope_local_names

    cls_src = (
        "def enclosing():\n"
        "    outer_v = 10\n"
        "    class Outer:\n"
        "        cls_attr = 1\n"
        "        def method(self):\n"
        "            local_v = 2\n"
        "            return local_v\n"
        "    return outer_v\n"
    )
    cls_tree = ast.parse(cls_src)
    fn_node = cls_tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    cls_node = fn_node.body[1]
    assert isinstance(cls_node, ast.ClassDef)
    method_node = cls_node.body[1]
    assert isinstance(method_node, ast.FunctionDef)
    assert "cls_attr" not in _extract_scope_local_names(
        method_node, module_tree=cls_tree, include_classes=False
    )
    assert "cls_attr" in _extract_scope_local_names(
        method_node, module_tree=cls_tree, include_classes=True
    )


def test_get_scope_parent_map_excludes_expr_context_tokens() -> None:
    """Verifies _get_scope_parent_map skips shared ast.expr_context singleton tokens."""
    from pydoppelgangerhunt.fixer._df_cache import _get_scope_parent_map

    cls_tree = ast.parse("def enclosing(x):\n    y = x + 1\n    return y\n")
    fn_node = cls_tree.body[0]
    pmap = _get_scope_parent_map(cls_tree)
    assert not any(isinstance(k, ast.expr_context) for k in pmap)
    assert fn_node in pmap


def test_is_irrefutable_pattern_rejects_none_and_scope_exports_match() -> None:
    """Verifies _is_irrefutable_pattern(None) returns False and scope re-exports match _df_closures."""
    import pydoppelgangerhunt.fixer.scope as scope_mod
    from pydoppelgangerhunt.fixer._df_closures import (
        _is_irrefutable_case,
        _is_irrefutable_pattern,
    )

    assert _is_irrefutable_pattern(None) is False
    assert scope_mod._is_irrefutable_pattern is _is_irrefutable_pattern
    assert scope_mod._is_irrefutable_case is _is_irrefutable_case


def test_branch_merged_suppress_provenance_and_with_splitting_strict_names() -> None:
    """Verifies branch-conservative _SuppressProvenance merging across compound statements and strict_names propagation."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _analyze_block_assignment,
        _block_terminates,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
        _split_with_body_by_mode,
        _update_suppress_provenance_from_stmt,
    )

    # 1. If branch rebinds suppressor only on one path -> provenance retains suppressor conservatively
    if_rebind_src = (
        "from contextlib import suppress\n"
        "def f(cond, items):\n"
        "    cm = suppress\n"
        "    if cond:\n"
        "        cm = nullcontext\n"
        "    with cm(ValueError):\n"
        "        x = compute(items)\n"
        "    return x\n"
    )
    if_tree = ast.parse(if_rebind_src)
    if_fn = if_tree.body[1]
    assert isinstance(if_fn, ast.FunctionDef)
    prov_seeded = _seed_scope_suppress_aliases(if_fn, if_tree, u_start=5)
    assert "cm" in prov_seeded.suppress_names
    def_if, cond_if = _analyze_block_assignment(
        if_fn.body[:-1],
        suppress_aliases=_seed_scope_suppress_aliases(if_fn, if_tree, u_start=2),
        strict_with_suppression=False,
    )
    assert "x" not in def_if
    assert "x" in cond_if
    bound_if, _ = _extract_block_definite_bindings_and_term(
        if_fn.body[:-1],
        None,
        suppress_aliases=_seed_scope_suppress_aliases(if_fn, if_tree, u_start=2),
        strict_with_suppression=False,
    )
    assert "x" not in bound_if

    # 2. If/else where BOTH branches overwrite suppressor -> suppressor is cleared
    both_clear_stmts = ast.parse(
        "if cond:\n" "    cm = nullcontext\n" "else:\n" "    cm = closing\n"
    ).body
    prov_init = _SuppressProvenance(
        suppressors=frozenset({"cm"}),
        non_suppressors=frozenset({"nullcontext", "closing"}),
    )
    prov_after_both = _update_suppress_provenance_from_stmt(both_clear_stmts[0], prov_init)
    assert "cm" not in prov_after_both.suppress_names
    assert "cm" in prov_after_both.non_suppress_names

    # 3. While / For / Try / With / Match branch merges preserve pre-loop/pre-branch suppressor
    compound_branch_src = (
        "for item in items:\n"
        "    cm1 = other_factory()\n"
        "while cond:\n"
        "    cm2 = other_factory()\n"
        "try:\n"
        "    cm3 = other_factory()\n"
        "except Exception:\n"
        "    pass\n"
        "with lock:\n"
        "    cm4 = other_factory()\n"
    )
    comp_stmts = ast.parse(compound_branch_src).body
    prov_comp = _SuppressProvenance(suppressors=frozenset({"cm1", "cm2", "cm3", "cm4"}))
    for st in comp_stmts:
        prov_comp = _update_suppress_provenance_from_stmt(st, prov_comp)
    assert {"cm1", "cm2", "cm3", "cm4"} <= prov_comp.suppress_names

    # 4. _split_with_body_by_mode forwards strict_names for proven suppressor aliases and rejects bare strings
    with_suppress_unknown_name = ast.parse(
        "with custom_suppressor(NameError):\n" "    y = possibly_unbound_name\n" "    return 1\n"
    ).body[0]
    assert isinstance(with_suppress_unknown_name, ast.With)
    sup_prov = _SuppressProvenance(suppressors=frozenset({"custom_suppressor"}))
    with pytest.raises(TypeError, match="not a bare str"):
        _split_with_body_by_mode(
            with_suppress_unknown_name,
            "suppressing",  # type: ignore[arg-type]
            safe_names=set(),
            strict_names=True,
        )
    prefix_strict, rem_strict = _split_with_body_by_mode(
        with_suppress_unknown_name,
        sup_prov,
        safe_names=set(),
        strict_names=True,
    )
    prefix_lax, _ = _split_with_body_by_mode(
        with_suppress_unknown_name,
        sup_prov,
        safe_names=set(),
        strict_names=False,
    )
    assert not prefix_strict
    assert len(rem_strict) == 2
    assert len(prefix_lax) == 2
    assert (
        _block_terminates(
            [with_suppress_unknown_name],
            suppress_aliases=sup_prov,
            safe_names=set(),
        )
        is False
    )
    assert (
        _block_terminates(
            [with_suppress_unknown_name],
            suppress_aliases=sup_prov,
            safe_names={"possibly_unbound_name"},
        )
        is True
    )


def test_form_feed_source_lines_ifexp_walrus_and_stmt_delta_with_suppress() -> None:
    """Verifies form-feed line splitting parity, IfExp walrus definite intersection, and cached stmt_deltas with suppress."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _get_or_parse_tree,
        _split_source_lines_stripped,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
        _walrus_assignment_in_expr,
    )

    # 1. Form-feed \x0c in source does not desynchronize _split_source_lines_stripped from AST lineno
    ff_src = (
        "header = 'page1\x0cpage2'\n"
        "@(\n"
        "    my_decorator\n"
        ")\n"
        "def target_fn():\n"
        "    return header\n"
    )
    ff_tree = _get_or_parse_tree(ff_src)
    assert isinstance(ff_tree, ast.Module)
    ff_lines = _split_source_lines_stripped(ff_src)
    assert len(ff_lines) == 6
    dec_fn = ff_tree.body[1]
    assert _node_earliest_start_pos(dec_fn, source_lines=ff_lines) == (2, 0)

    # 2. _walrus_assignment_in_expr for ast.IfExp intersects definite bindings from both branches
    ifexp_both = ast.parse("(x := 1) if flag else (x := 2)", mode="eval").body
    def_both, cond_both = _walrus_assignment_in_expr(ifexp_both)
    assert def_both == {"x"}
    assert cond_both == set()

    ifexp_one = ast.parse("(x := 1) if flag else (y := 2)", mode="eval").body
    def_one, cond_one = _walrus_assignment_in_expr(ifexp_one)
    assert def_one == set()
    assert cond_one == {"x", "y"}

    ifexp_stmt = ast.parse("res = (x := 1) if flag else (x := 2)\n").body
    def_stmt, cond_stmt = _analyze_block_assignment(ifexp_stmt)
    assert {"res", "x"} <= def_stmt
    assert "x" not in cond_stmt

    # 3. Cached statement deltas agree with uncached path when `with suppress` reads pre_bound local
    suppress_pre_bound_src = (
        "from contextlib import suppress\n"
        "def outer(flag):\n"
        "    a = 10\n"
        "    with suppress(NameError):\n"
        "        x = a\n"
        "    def reader():\n"
        "        return x\n"
        "    out = x + 2\n"
        "    return reader(), out\n"
    )
    sp_tree = ast.parse(suppress_pre_bound_src)
    sp_fn = sp_tree.body[1]
    assert isinstance(sp_fn, ast.FunctionDef)
    clear_dataflow_cache()
    info_cached_1 = _analyze_enclosing_fn_cells(sp_fn, (8, 8, 4, 17), mod_tree=sp_tree)
    info_cached_2 = _analyze_enclosing_fn_cells(sp_fn, (8, 8, 4, 17), mod_tree=sp_tree)
    info_uncached = _analyze_enclosing_fn_cells(sp_fn, (8, 8, 4, 17), mod_tree=None)
    assert info_cached_1.pre_bound == info_cached_2.pre_bound
    assert info_cached_1.pre_bound == info_uncached.pre_bound
    assert "x" in info_cached_1.pre_bound


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
            "source_lines_is_sliced": True,
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
    with caplog.at_level(logging.INFO):
        patch_out = generate_refactoring_patch(
            [(0.99, uh1, uh2)], repo_root=str(tmp_path), replace_clones=True
        )
    assert patch_out == ""
    assert "severed closure cell (global escape)" in caplog.text

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
            (sym_outer["locals"] | sym_outer["parameters"]) - sym_outer["globals"] - {"z"}
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


def test_multi_item_with_subsequent_targets_and_walrus_after_suppressor() -> None:
    """Verifies that in a multi-item with statement, items after a suppressing or unknown item bind targets and walrus names only conditionally."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

    # 1. Intra-unit _analyze_block_assignment: item 1 suppresses, item 2 has `as res` and `(w := ...)`
    unit_src = "with suppress(ValueError), (w := acquire()) as res:\n" "    body_var = 1\n"
    stmts = ast.parse(unit_src).body
    def_assigned, cond_assigned = _analyze_block_assignment(stmts)
    assert "res" not in def_assigned
    assert "w" not in def_assigned
    assert "body_var" not in def_assigned
    assert {"res", "w", "body_var"} <= cond_assigned

    # 2. Item 1 is safe (open), item 2 suppresses, item 3 is after suppressor: items 1 & 2 bind definitely, item 3 and raising body bind conditionally
    unit_src2 = (
        "with open((w1 := 'f')) as safe_res, suppress((w2 := get_err())) as sup_res, open((w3 := 'g')) as post_sup_res:\n"
        "    body_var = may_raise()\n"
    )
    stmts2 = ast.parse(unit_src2).body
    def2, cond2 = _analyze_block_assignment(stmts2)
    assert {"w1", "safe_res", "w2", "sup_res"} <= def2
    assert "w3" not in def2
    assert "post_sup_res" not in def2
    assert "body_var" not in def2
    assert {"w3", "post_sup_res", "body_var"} <= cond2

    # 3. Enclosing cell traversal: pre-unit multi-item `with` where item 1 suppresses and item 2 binds `x` via `as` and `w`/`u` via walrus
    enc_src = (
        "def outer(f, g, d):\n"
        "    with suppress(RuntimeError), (w := f()) as (x, d[(u := g())]):\n"
        "        z = 1\n"
        "    target = 10\n"
        "    return x, w, u, z, target\n"
    )
    enc_tree = ast.parse(enc_src)
    fn_node = enc_tree.body[0]
    info = _analyze_enclosing_fn_cells(
        fn_node,
        _SpanBounds(4, 4, 4, 15),
        mod_tree=enc_tree,
    )
    assert "x" not in info.pre_bound
    assert "w" not in info.pre_bound
    assert "u" not in info.pre_bound
    assert "z" not in info.pre_bound
    assert {"x", "w", "u", "z"} <= info.pre_mutated
    assert {"x", "w", "u", "z"} <= (info.fn_locals - info.pre_bound)


def test_enclosing_global_read_pure_builtin_allowlist_and_star_import_rejection() -> None:
    """Verifies units reading an enclosing global allow only _PURE_GLOBAL_SAFE_BUILTINS and reject excluded builtins or wildcard imports."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _unit_has_global_mutating_call,
    )

    # 1. Positive control: pure builtins in _PURE_GLOBAL_SAFE_BUILTINS do not trigger global_escape
    for pure_call in ("len(g)", "bool(g)", "int(g)"):
        pure_src = (
            "g = [1, 2, 3]\n"
            "def outer():\n"
            "    global g\n"
            f"    x = {pure_call}\n"
            "    return x\n"
        )
        sc_pure = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": pure_src,
            },
            tree=ast.parse(pure_src),
        )
        assert "severed_closure_cell" not in sc_pure["control_flow_hazards"]
        assert "global_escape" not in sc_pure["closure_hazard_reasons"]
        assert (
            _unit_has_global_mutating_call(ast.parse(f"x = {pure_call}\n").body, frozenset())
            is False
        )

    # 2. Negative control: excluded builtins (dir, vars, print, eval, sum, repr) trigger global_escape
    for impure_call in (
        "dir()",
        "vars()",
        "print(g)",
        "eval('1')",
        "sum(g)",
        "repr(g)",
    ):
        impure_src = (
            "g = [1, 2, 3]\n"
            "def outer():\n"
            "    global g\n"
            f"    x = (g, {impure_call})\n"
            "    return x\n"
        )
        sc_impure = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": impure_src,
            },
            tree=ast.parse(impure_src),
        )
        assert "severed_closure_cell" in sc_impure["control_flow_hazards"]
        assert "global_escape" in sc_impure["closure_hazard_reasons"]
        assert (
            _unit_has_global_mutating_call(ast.parse(f"x = (g, {impure_call})\n").body, frozenset())
            is True
        )

    # 3. Wildcard import in module or enclosing scope shadows all builtins and triggers global_escape even for len(g)
    star_src = (
        "from math import *\n"
        "g = [1, 2, 3]\n"
        "def outer():\n"
        "    global g\n"
        "    x = len(g)\n"
        "    return x\n"
    )
    sc_star = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": star_src,
        },
        tree=ast.parse(star_src),
    )
    assert "severed_closure_cell" in sc_star["control_flow_hazards"]
    assert "global_escape" in sc_star["closure_hazard_reasons"]
    assert (
        _unit_has_global_mutating_call(
            ast.parse("x = len(g)\n").body, frozenset(), has_star_import=True
        )
        is True
    )


def test_normalized_source_code_equivalence_and_stale_source_code_hazard(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies comment-stripped/formatted source_code is accepted while genuinely stale source_code reports stale_source_code."""
    import logging
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch

    mod_src = (
        "def f1(a):\n"
        "    x = a + 1  # inline comment\n"
        "    y = x * 2  # another comment\n"
        "    return y\n\n"
        "def f2(a):\n"
        "    x = a + 1  # comment b\n"
        "    y = x * 2\n"
        "    return y\n"
    )
    tree = ast.parse(mod_src)

    # Comment-stripped and whitespace-normalized source_code matches raw_lines without hazard
    sc_norm = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "basic_block",
            "source_text": mod_src,
            "source_code": "x=a+1\ny=x*2\n",
        },
        tree=tree,
    )
    assert sc_norm["is_control_flow_safe"] is True
    assert "stale_source_code" not in sc_norm["control_flow_hazards"]
    assert "pathological_ast" not in sc_norm["control_flow_hazards"]
    assert sc_norm["inputs"] == ["a"]

    # Genuinely stale source_code reports stale_source_code (and not pathological_ast)
    f_path = tmp_path / "stale_mod.py"
    f_path.write_text(mod_src, encoding="utf-8")
    u_stale1 = {
        "file": str(f_path),
        "start": 2,
        "end": 3,
        "name": "u1",
        "kind": "basic_block",
        "source_code": "unrelated_stale = 123\nother_stale = 456\n",
    }
    u_stale2 = {
        "file": str(f_path),
        "start": 7,
        "end": 8,
        "name": "u2",
        "kind": "basic_block",
        "source_code": "unrelated_stale = 123\nother_stale = 456\n",
    }
    with caplog.at_level(logging.INFO):
        patch_res = generate_refactoring_patch(
            [(0.99, u_stale1, u_stale2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
    assert patch_res == ""
    assert "stale source code" in caplog.text


def test_slice_unit_token_lines_and_compile_warning_suppression() -> None:
    """Verifies _slice_unit_token_lines pre-sliced handling and _has_invalid_lexical_scope_declarations SyntaxWarning suppression."""
    import warnings
    from pydoppelgangerhunt.fixer._df_span_index import (
        _has_invalid_lexical_scope_declarations,
    )
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    # 1. Explicit is_pre_sliced=False on a 2-line file where unit is lines 1..2
    full_two_lines = ["a = 1\n", "b = 2\n"]
    sliced_full = _slice_unit_token_lines(
        {"start": 1, "end": 2},
        full_two_lines,
        is_pre_sliced=False,
    )
    assert sliced_full == ["a = 1\n", "b = 2\n"]

    # 2. Line-sliced source_lines (without source_lines_is_sliced) still undergoes column slicing
    sc_col = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "start_col": 11,
            "end_col": 16,
            "kind": "complex_expr",
            "source_lines": ["    a = 1; b + a\n"],
        }
    )
    assert sc_col["inputs"] == ["b", "a"] or set(sc_col["inputs"]) == {
        "a",
        "b",
    }

    # 3. Pre-sliced multi-line snippet with start_col/end_col is not double-sliced
    pre_sliced_multi = ["x = (\n", "    a + 1\n", ")"]
    res_multi = _slice_unit_token_lines(
        {"start": 10, "end": 12, "start_col": 4, "end_col": 5},
        pre_sliced_multi,
        is_pre_sliced=True,
        explicit_col_sliced=True,
    )
    assert res_multi == pre_sliced_multi

    # 4. _has_invalid_lexical_scope_declarations suppresses SyntaxWarning from invalid escape sequences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warn_mod = ast.parse('def f():\n    s = "\\d+"\n    return 1 is 1\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        is_invalid = _has_invalid_lexical_scope_declarations([warn_mod.body[0]], mod_tree=warn_mod)
    assert is_invalid is False
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]


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


def test_deep_nested_with_early_return_propagation() -> None:
    """Verifies 3-level nested with + early return propagation across known and unknown context managers."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    deep_sup_src = (
        "from contextlib import suppress\n"
        "def outer(f1, tx_lock, compute):\n"
        "    with open(f1):\n"
        "        with tx_lock():\n"
        "            with suppress(ValueError):\n"
        "                x = compute()\n"
        "                return x\n"
        "    after = 1\n"
        "    return after\n"
    )
    deep_sup_tree = ast.parse(deep_sup_src)
    deep_sup_fn = deep_sup_tree.body[1]
    prov_sup = _seed_scope_suppress_aliases(deep_sup_fn, mod_tree=deep_sup_tree)
    for strict_with in (False, True):
        assert (
            _block_terminates(
                [deep_sup_fn.body[0]],  # type: ignore[attr-defined]
                suppress_aliases=prov_sup,
                strict_with_suppression=strict_with,
            )
            is False
        )
        bound_ds, term_ds = _extract_block_definite_bindings_and_term(
            deep_sup_fn.body[:2],  # type: ignore[attr-defined]
            None,
            suppress_aliases=prov_sup,
            strict_with_suppression=strict_with,
        )
        assert "after" in bound_ds
        assert "x" not in bound_ds
        assert term_ds is False

    deep_unk_src = (
        "def outer(f1, f2, compute):\n"
        "    with open(f1):\n"
        "        with tx_lock():\n"
        "            with open(f2):\n"
        "                x = compute()\n"
        "                return x\n"
        "    after = 1\n"
        "    return after\n"
    )
    deep_unk_tree = ast.parse(deep_unk_src)
    deep_unk_fn = deep_unk_tree.body[0]
    prov_unk = _seed_scope_suppress_aliases(deep_unk_fn, mod_tree=deep_unk_tree)
    assert (
        _block_terminates(
            [deep_unk_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_unk,
            strict_with_suppression=False,
        )
        is True
    )
    assert (
        _block_terminates(
            [deep_unk_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_unk,
            strict_with_suppression=True,
        )
        is False
    )
    assert (
        _block_terminates(
            [deep_unk_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_unk,
            strict_with_suppression=True,
            extra_non_suppressing_ctx_managers=frozenset({"tx_lock"}),
        )
        is True
    )


def test_nested_raises_and_nested_safe_return_inside_suppress() -> None:
    """Verifies nested pytest.raises vs non-raising variable assignment inside suppress blocks."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )

    raises_in_sup_src = (
        "with suppress(Exception):\n"
        "    with pytest.raises(ValueError):\n"
        "        x = 10\n"
        "        return x\n"
        "after = 2\n"
    )
    raises_in_sup_stmts = ast.parse(raises_in_sup_src).body
    assert _block_terminates([raises_in_sup_stmts[0]]) is False
    bound_ris, term_ris = _extract_block_definite_bindings_and_term(raises_in_sup_stmts, None)
    assert bound_ris == {"after"}
    assert term_ris is False

    nested_safe_ret_src = (
        "with suppress(ValueError):\n"
        "    with suppress(KeyError):\n"
        "        x = 10\n"
        "    return x\n"
        "    risky_unreachable()\n"
    )
    nested_safe_ret_stmts = ast.parse(nested_safe_ret_src).body
    assert _block_terminates([nested_safe_ret_stmts[0]]) is True
    bound_nsr, term_nsr = _extract_block_definite_bindings_and_term(nested_safe_ret_stmts, None)
    assert bound_nsr == {"x"}
    assert term_nsr is True


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
    helper_ped = synthesize_shared_helper_code(u1_expr, u2_expr, closure_strictness="pedantic")
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


def test_conditional_store_promoted_to_input_only_when_read_downstream() -> None:
    """Verifies write-only conditional store is promoted to unit inputs only when read downstream."""
    from pydoppelgangerhunt.fixer.scope import _inspect_unit_scope

    dead_cond_src = (
        "def work(items, flag):\n"
        "    tmp = 0\n"
        "    acc = 1\n"
        "    if flag:\n"
        "        tmp = 2\n"
        "    acc = acc + len(items)\n"
        "    return acc\n"
    )
    u_dead_cond = {
        "name": "dead_cond_blk",
        "kind": "block",
        "start": 4,
        "end": 6,
        "source_text": dead_cond_src,
    }
    sc_dead = _inspect_unit_scope(u_dead_cond)
    assert "tmp" not in sc_dead["inputs"]
    assert "tmp" not in sc_dead["free_vars"]

    live_cond_src = (
        "def work_live(items, flag):\n"
        "    tmp = 0\n"
        "    acc = 1\n"
        "    if flag:\n"
        "        tmp = 2\n"
        "    acc = acc + len(items)\n"
        "    return acc + tmp\n"
    )
    u_live_cond = {
        "name": "live_cond_blk",
        "kind": "block",
        "start": 4,
        "end": 6,
        "source_text": live_cond_src,
    }
    sc_live = _inspect_unit_scope(u_live_cond)
    assert "tmp" in sc_live["inputs"]
    assert "tmp" in sc_live["free_vars"]


def test_syntax_warning_suppressed_across_parse_and_inference_paths() -> None:
    """Verifies SyntaxWarning is suppressed in tree parsing, symtable, column inference, and normalized dumps."""
    import warnings
    from pydoppelgangerhunt.fixer.scope import (
        _get_or_parse_tree,
        _infer_missing_col_from_line_ast,
        _inspect_unit_scope,
        _parse_normalized_stmt_dumps,
    )

    escape_src = (
        'PAT = "\\d+\\s+"\n' "def match_fn(x):\n" '    y = ("\\d+" + str(x))\n' "    return y\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        esc_tree = _get_or_parse_tree(escape_src)
        assert esc_tree is not None
        u_esc = {
            "name": "match_fn",
            "kind": "function",
            "start": 2,
            "end": 4,
            "source_text": escape_src,
        }
        sc_esc = _inspect_unit_scope(u_esc, file_tree=esc_tree)
        assert sc_esc["is_control_flow_safe"] is True
        col, ecol = _infer_missing_col_from_line_ast(
            '    y = ("\\d+" + str(x))',
            None,
            24,
            allow_expr_fallback=True,
        )
        assert col == 4 and ecol == 24
        dumps = _parse_normalized_stmt_dumps('PAT = "\\d+\\s+"\n')
        assert dumps is not None and len(dumps) == 1


def test_imported_suppress_and_nullcontext_provenance() -> None:
    """Verifies imported/aliased suppress and nullcontext provenance."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    imp_sup_src = (
        "from contextlib import suppress as sup, nullcontext as nc\n"
        "def f():\n"
        "    with sup(ValueError):\n"
        "        with nc():\n"
        "            x = 1\n"
        "            return x\n"
        "    after = 2\n"
    )
    imp_sup_tree = ast.parse(imp_sup_src)
    imp_sup_fn = imp_sup_tree.body[1]
    prov_imp = _seed_scope_suppress_aliases(imp_sup_fn, mod_tree=imp_sup_tree)
    assert "sup" not in prov_imp.shadowed_names
    assert "nc" not in prov_imp.shadowed_names
    assert ("sup", "contextlib.suppress") in prov_imp.imported_symbols
    assert ("nc", "contextlib.nullcontext") in prov_imp.imported_symbols
    assert (
        _block_terminates(
            [imp_sup_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_imp,
        )
        is True
    )
    bound_imp, term_imp = _extract_block_definite_bindings_and_term(
        imp_sup_fn.body,  # type: ignore[attr-defined]
        None,
        suppress_aliases=prov_imp,
    )
    assert bound_imp == {"x"}
    assert term_imp is True


def test_try_except_else_finally_positive_provenance_and_with_item_shadowing() -> None:
    """Verifies positive provenance across try/else/except/finally and with-item target shadowing."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    try_prov_src = (
        "def g(f1, f2, f3, f4):\n"
        "    try:\n"
        "        cm_safe = open\n"
        "        may_raise()\n"
        "        cm_late = open\n"
        "    except RuntimeError:\n"
        "        with cm_safe(f1):\n"
        "            ex_ok = 1\n"
        "        return ex_ok\n"
        "    else:\n"
        "        with cm_late(f2):\n"
        "            el_ok = 2\n"
        "    finally:\n"
        "        with cm_safe(f3):\n"
        "            fin_ok = 3\n"
        "    with cm_late(f4):\n"
        "        post_ok = 4\n"
    )
    try_prov_tree = ast.parse(try_prov_src)
    try_prov_fn = try_prov_tree.body[0]
    prov_g = _seed_scope_suppress_aliases(try_prov_fn, mod_tree=try_prov_tree)
    bound_try, _ = _extract_block_definite_bindings_and_term(
        try_prov_fn.body,  # type: ignore[attr-defined]
        None,
        suppress_aliases=prov_g,
        strict_with_suppression=True,
    )
    assert {"cm_safe", "cm_late", "el_ok", "fin_ok", "post_ok"} <= bound_try

    shadow_as_src = (
        "from contextlib import suppress\n"
        "def h(custom_factory):\n"
        "    with suppress(ValueError) as suppress:\n"
        "        with suppress(KeyError):\n"
        "            x = 1\n"
        "            return x\n"
        "    after = 2\n"
    )
    shadow_as_tree = ast.parse(shadow_as_src)
    shadow_as_fn = shadow_as_tree.body[1]
    prov_h = _seed_scope_suppress_aliases(shadow_as_fn, mod_tree=shadow_as_tree)
    assert (
        _block_terminates(
            [shadow_as_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_h,
        )
        is False
    )
    bound_h, term_h = _extract_block_definite_bindings_and_term(
        shadow_as_fn.body,  # type: ignore[attr-defined]
        None,
        suppress_aliases=prov_h,
    )
    assert "after" in bound_h
    assert "x" not in bound_h
    assert term_h is False

    rebind_in_body_src = (
        "from contextlib import suppress\n"
        "def k(other_cm):\n"
        "    with suppress(ValueError):\n"
        "        suppress = other_cm\n"
        "        with suppress(KeyError):\n"
        "            x = 1\n"
        "            return x\n"
        "    after = 2\n"
    )
    rebind_tree = ast.parse(rebind_in_body_src)
    rebind_fn = rebind_tree.body[1]
    prov_k = _seed_scope_suppress_aliases(rebind_fn, mod_tree=rebind_tree)
    assert (
        _block_terminates(
            [rebind_fn.body[0]],  # type: ignore[attr-defined]
            suppress_aliases=prov_k,
        )
        is False
    )


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
        "    with Swallow(), tolerate(), Transaction():\n"
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
    assert set(recorded) == {"Swallow", "Transaction", "tolerate"}
    assert any(
        "3 unproven context manager(s)" in rec.message
        and "were treated as non-suppressing" in rec.message
        for rec in caplog.records
    )

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


def test_local_class_allowlist_and_nested_function_seeding() -> None:
    """Verifies local/relative allowlist matching, __exit__ proof, and nested function seeding."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
        _seed_scope_suppress_aliases,
    )

    local_and_rel_src = (
        "from .tx import RelativeTx\n"
        "class SafeTransaction:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n"
        "class DynamicTransaction(BaseTx):\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        self._cleanup(exc_type)\n"
        "def run_tx(compute):\n"
        "    with SafeTransaction():\n"
        "        auto_proven = compute()\n"
        "    with DynamicTransaction(), RelativeTx():\n"
        "        allowlisted = compute()\n"
        "    return auto_proven, allowlisted\n"
    )
    local_rel_tree = ast.parse(local_and_rel_src)
    tx_fn = local_rel_tree.body[3]
    assert isinstance(tx_fn, ast.FunctionDef)
    prov_tx = _seed_scope_suppress_aliases(tx_fn, mod_tree=local_rel_tree)
    assert "SafeTransaction" in prov_tx.non_suppressors
    assert "SafeTransaction" not in prov_tx.shadowed_names
    assert "DynamicTransaction" not in prov_tx.shadowed_names
    assert "RelativeTx" not in prov_tx.shadowed_names

    defs_no_allow, conds_no_allow = _analyze_block_assignment(
        tx_fn.body[:-1],
        suppress_aliases=prov_tx,
        strict_with_suppression=True,
    )
    assert "auto_proven" in defs_no_allow
    assert "allowlisted" not in defs_no_allow
    assert "allowlisted" in conds_no_allow

    defs_with_allow, conds_with_allow = _analyze_block_assignment(
        tx_fn.body[:-1],
        suppress_aliases=prov_tx,
        extra_non_suppressing_ctx_managers=frozenset({"DynamicTransaction", ".RelativeTx"}),
        strict_with_suppression=True,
    )
    assert {"auto_proven", "allowlisted"} <= defs_with_allow
    assert "allowlisted" not in conds_with_allow

    nested_seed_src = (
        "from contextlib import nullcontext as nc\n"
        "def outer():\n"
        "    def inner(compute):\n"
        "        with nc():\n"
        "            x = compute()\n"
        "        return x\n"
        "    return inner\n"
        "nc = unknown_rebind()\n"
    )
    nested_seed_tree = ast.parse(nested_seed_src)
    outer_seed_fn = nested_seed_tree.body[1]
    assert isinstance(outer_seed_fn, ast.FunctionDef)
    inner_seed_fn = outer_seed_fn.body[0]
    assert isinstance(inner_seed_fn, ast.FunctionDef)
    prov_outer = _seed_scope_suppress_aliases(outer_seed_fn, mod_tree=nested_seed_tree)
    prov_inner = _seed_scope_suppress_aliases(inner_seed_fn, mod_tree=nested_seed_tree)
    assert prov_inner.non_suppressors == prov_outer.non_suppressors
    assert "nc" not in prov_inner.non_suppressors


def test_column_stale_source_detection_and_comment_stripping() -> None:
    """Covers stale-source detection when columns are present, expression equality/kwarg matching, and quote-aware comment stripping."""
    from pydoppelgangerhunt.fixer.scope import (
        _is_ordered_contiguous_source_match,
        _strip_inline_comment_line,
        ensure_unit_scope_closure_mode,
    )

    mod_src = "def f(a, b):\n    res = (a + b) * 2\n    return res\n"
    mod_tree = ast.parse(mod_src)
    sc_col_stale = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 10,
            "end_col": 21,
            "kind": "complex_expr",
            "source_text": mod_src,
            "source_code": "(completely_stale + 999)",
        },
        tree=mod_tree,
    )
    assert "stale_source_code" in sc_col_stale["control_flow_hazards"]
    assert sc_col_stale["is_control_flow_safe"] is False

    for expr_target, raw_line in (
        ("a == 1", "if a == 1 and b == 2:"),
        ("a <= 10", "flag = a <= 10 or b >= 20"),
        ("f(a=1)", "res = f(a=1) + g(b=2)"),
    ):
        assert _is_ordered_contiguous_source_match([expr_target], [raw_line]) is True

    unclosed_line = 'x = call("#not_a_comment", [\n'
    stripped_line = _strip_inline_comment_line(unclosed_line)
    assert '"#not_a_comment"' in stripped_line

    unclosed_with_real_comment = 'x = call("#keep_me", [  # real comment\n'
    stripped_with_comment = _strip_inline_comment_line(unclosed_with_real_comment)
    assert '"#keep_me"' in stripped_with_comment
    assert "real comment" not in stripped_with_comment

    hand_built_scope: dict[str, Any] = {
        "inputs": ["bogus_caller_input"],
        "outputs": ["bogus_caller_output"],
    }
    recomputed_full = ensure_unit_scope_closure_mode(
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_text": mod_src,
            "source_code": "    res = (a + b) * 2\n",
        },
        hand_built_scope,
        False,
        tree=mod_tree,
    )
    assert "bogus_caller_input" not in recomputed_full["inputs"]
    assert set(recomputed_full["inputs"]) == {"a", "b"}
    assert recomputed_full["outputs"] == ["res"]


def test_validate_unit_schema_accepts_parser_keys_and_rejects_non_mapping() -> None:
    """Verifies validate_unit_schema accepts parser-emitted keys and raises TypeError on non-mapping inputs."""
    from pydoppelgangerhunt.fixer import validate_unit_schema

    assert (
        validate_unit_schema(
            {
                "name": "u1",
                "kind": "basic_block",
                "start": 2,
                "end": 3,
                "start_col": 4,
                "end_col": 12,
                "closure_strictness": "pedantic",
                "non_suppressing_ctx_managers": ["SafeTx"],
                "enclosing_class": "MyService",
                "enclosing_is_async": False,
                "enclosing_is_classmethod": False,
                "enclosing_is_staticmethod": False,
                "self_param": "self",
                "cls_param": None,
                "tokens": ["a", "b"],
                "norm_tokens": ["ID", "ID"],
                "structural_hash": "deadbeef",
            },
            strict=True,
        )
        == set()
    )
    assert validate_unit_schema({"start": 1, "unknown_field": 123}) == {"unknown_field"}
    with pytest.raises(ValueError, match="Unrecognized keys"):
        validate_unit_schema({"start": 1, "unknown_field": 123}, strict=True)
    with pytest.raises(TypeError, match="Expected a unit mapping"):
        validate_unit_schema("not_a_dict", strict=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected a unit mapping"):
        validate_unit_schema("not_a_dict", strict=True)  # type: ignore[arg-type]


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


def test_infer_missing_col_multiple_statements_with_end_col_only() -> None:
    """Verifies _infer_missing_col_from_line_ast does not fall through to expression matching on statement units when multiple statements match."""
    from pydoppelgangerhunt.fixer.scope import _infer_missing_col_from_line_ast

    raw_line = "    if flag: a = f(1)"
    col_stmt, ecol_stmt = _infer_missing_col_from_line_ast(
        raw_line,
        None,
        21,
        allow_expr_fallback=False,
    )
    assert (col_stmt, ecol_stmt) == (None, 21)

    col_expr, ecol_expr = _infer_missing_col_from_line_ast(
        raw_line,
        None,
        21,
        unit_kind="complex_expr",
    )
    assert (col_expr, ecol_expr) == (17, 21)


def test_pre_try_and_pre_match_bindings_survive_loop_control_termination() -> None:
    """Verifies pre-try and pre-match definite bindings survive when try or match terminates via loop control."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
    )

    try_loop_src = "x = 1\n" "try:\n" "    break\n" "except RuntimeError:\n" "    continue\n"
    defs_try, conds_try, term_try, _ = _analyze_block_assignment_impl(
        ast.parse(try_loop_src).body,
        loop_control_terminates=True,
    )
    assert term_try is True
    assert "x" in defs_try
    assert "x" not in conds_try

    match_loop_src = (
        "x = 1\n"
        "match status:\n"
        "    case 200:\n"
        "        break\n"
        "    case _:\n"
        "        continue\n"
    )
    defs_match, conds_match, term_match, _ = _analyze_block_assignment_impl(
        ast.parse(match_loop_src).body,
        loop_control_terminates=True,
    )
    assert term_match is True
    assert "x" in defs_match
    assert "x" not in conds_match


def test_placeholder_origin_vs_real_comment_and_source_code_only_unit() -> None:
    """Verifies real comments starting with '# Source for ' are not treated as placeholders and source_code + tree units are not falsely flagged stale."""
    mod_with_comment = (
        "def f(a, b):\n" "    # Source for custom step\n" "    res = a + b\n" "    return res\n"
    )
    tree_comment = ast.parse(mod_with_comment)
    sc_comment_mismatch = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "kind": "basic_block",
            "source_text": mod_with_comment,
            "source_code": "    # Source for completely different line\n",
        },
        tree=tree_comment,
    )
    assert "stale_source_code" in sc_comment_mismatch["control_flow_hazards"]

    mod_simple = "def g(a, b):\n    res = a + b\n    return res\n"
    tree_simple = ast.parse(mod_simple)
    sc_code_and_tree = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_code": "    res = a + b\n",
        },
        tree=tree_simple,
    )
    assert "stale_source_code" not in sc_code_and_tree["control_flow_hazards"]
    assert sc_code_and_tree["is_control_flow_safe"] is True


def test_tracked_cached_dict_prunes_reverse_index_and_missing_cached_ast_nodes_raises() -> None:
    """Verifies _TrackedCachedDict removes stale reverse index entries on overwrite/pop/del and missing cached_ast_nodes() raises TypeError."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )

    clear_dataflow_cache()
    mod = ast.parse("def f():\n    a = 1\n    b = 2\n")
    fn = mod.body[0]
    n1 = fn.body[0]  # type: ignore[attr-defined]
    n2 = fn.body[1]  # type: ignore[attr-defined]
    ck = _ScopeCacheKey(fn, "unit_test_tracked_dict")
    _scope_cell_info_cache[ck] = {}
    tracked = _scope_cell_info_cache[ck]
    assert isinstance(tracked, _TrackedCachedDict)
    tracked["k1"] = n1
    assert ck in _scope_owner_keys_index.get(n1, set())
    tracked["k1"] = n2
    assert ck not in _scope_owner_keys_index.get(n1, set())
    assert ck in _scope_owner_keys_index.get(n2, set())
    popped = tracked.pop("k1")
    assert popped is n2
    assert ck not in _scope_owner_keys_index.get(n2, set())

    class _UnindexedCustomBox:
        pass

    with pytest.raises(TypeError, match="cached_ast_nodes"):
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _ScopeCacheKey(fn, "unit_test_bad_box_cache"),
            _UnindexedCustomBox,
        )


def test_source_lines_origin_equality_and_hash_contract() -> None:
    """Verifies SourceLinesOrigin preserves reflexive, symmetric, transitive equality and hash consistency."""
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    origins = (
        SourceLinesOrigin.PRE_SLICED,
        SourceLinesOrigin.FULL_BUFFER,
        SourceLinesOrigin.DISK,
        SourceLinesOrigin.PLACEHOLDER,
    )
    assert len(set(origins)) == 4
    for a in origins:
        assert a == a
        assert hash(a) == hash(a)
        assert a != a.name
        assert a != bool(a)
        assert a != True  # noqa: E712
        assert a != False  # noqa: E712
        for b in origins:
            if a is b:
                assert a == b
                assert hash(a) == hash(b)
            else:
                assert a != b

    mixed_set = {
        SourceLinesOrigin.PRE_SLICED,
        SourceLinesOrigin.PLACEHOLDER,
        True,
        "PRE_SLICED",
        "PLACEHOLDER",
    }
    assert len(mixed_set) == 5
    assert SourceLinesOrigin.PRE_SLICED.is_pre_sliced is True
    assert SourceLinesOrigin.PLACEHOLDER.is_placeholder is True
    assert SourceLinesOrigin.DISK.is_pre_sliced is False
    assert SourceLinesOrigin.DISK.is_placeholder is False


def test_walrus_left_to_right_order_and_compare_short_circuit_provenance() -> None:
    """Verifies left-to-right walrus evaluation order and ast.Compare short-circuit provenance merging."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _update_suppress_provenance_from_stmt,
    )

    # 1. Left-to-right sibling walrus: (x := suppress, y := x) -> both x and y are suppressors
    prov_base = _SuppressProvenance(
        suppressors=frozenset({"suppress"}),
        non_suppressors=frozenset({"nullcontext"}),
    )
    stmt_ltr = ast.parse("_ = (x := suppress, y := x)\n").body[0]
    prov_ltr = _update_suppress_provenance_from_stmt(stmt_ltr, prov_base)
    assert "x" in prov_ltr.suppressors
    assert "y" in prov_ltr.suppressors

    # 2. Reverse order when x starts as nullcontext: (y := x, x := suppress)
    # y reads x before x is rebound to suppress
    prov_with_x_safe = _SuppressProvenance(
        suppressors=frozenset({"suppress"}),
        non_suppressors=frozenset({"nullcontext", "x"}),
    )
    stmt_rtl = ast.parse("_ = (y := x, x := suppress)\n").body[0]
    prov_rtl = _update_suppress_provenance_from_stmt(stmt_rtl, prov_with_x_safe)
    assert "y" in prov_rtl.non_suppressors
    assert "y" not in prov_rtl.suppressors
    assert "x" in prov_rtl.suppressors
    assert "x" not in prov_rtl.non_suppressors

    # 3. ast.Compare: first comparator executes unconditionally, subsequent comparators may short-circuit
    stmt_cmp = ast.parse("_ = a < (first_c := nullcontext) < (second_c := nullcontext)\n").body[0]
    prov_cmp = _update_suppress_provenance_from_stmt(stmt_cmp, prov_base)
    assert "first_c" in prov_cmp.non_suppressors
    assert "second_c" not in prov_cmp.non_suppressors
    assert "second_c" in prov_cmp.shadowed_names

    # 4. Short-circuited comparator rebinding an existing non-suppressor to an unknown call invalidates it
    stmt_cmp_rebind = ast.parse("_ = a < b < (x := unknown_factory())\n").body[0]
    prov_cmp_rebind = _update_suppress_provenance_from_stmt(stmt_cmp_rebind, prov_with_x_safe)
    assert "x" not in prov_cmp_rebind.non_suppressors
    assert "x" in prov_cmp_rebind.shadowed_names


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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _class_def_is_proven_non_suppressing_in_exit,
        _class_def_may_suppress_in_exit,
    )

    cls_node = ast.parse(cls_src).body[0]
    assert isinstance(cls_node, ast.ClassDef)
    prov = _SuppressProvenance()
    assert _class_def_is_proven_non_suppressing_in_exit(cls_node, prov) is expect_proven_safe
    assert _class_def_may_suppress_in_exit(cls_node) is expect_may_suppress


def test_func_def_generator_contextmanager_try_and_with_around_yield() -> None:
    """Verifies _func_def_may_return_suppressor detects yield wrapped in try or with blocks and local imports."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _func_def_may_return_suppressor,
    )

    prov = _SuppressProvenance(
        suppressors=frozenset({"suppress"}),
        non_suppressors=frozenset({"nullcontext"}),
    )

    gen_try_src = (
        "def cm_try():\n" "    try:\n" "        yield\n" "    except Exception:\n" "        pass\n"
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


def test_walrus_side_effects_in_non_raising_expressions_inside_suppress() -> None:
    """Verifies walrus bindings in non-raising expressions inside suppress() propagate safe/nonzero state."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
    )

    src_safe_walrus = (
        "with suppress(ZeroDivisionError):\n"
        "    a = (w_safe := 10, w_bool := True)\n"
        "    b = (w_safe, not w_bool, 1 if w_bool else 2)\n"
    )
    defs_safe, conds_safe = _analyze_block_assignment(ast.parse(src_safe_walrus).body)
    assert {"a", "w_safe", "w_bool", "b"} <= defs_safe
    assert not ({"a", "w_safe", "w_bool", "b"} & conds_safe)

    src_short_circuit_walrus = (
        "with suppress(ZeroDivisionError):\n"
        "    _ = True or (w_short := True)\n"
        "    b = (not w_short, 1 if w_short else 2)\n"
    )
    defs_short, conds_short = _analyze_block_assignment(ast.parse(src_short_circuit_walrus).body)
    assert "b" not in defs_short
    assert "b" in conds_short


def test_cross_cache_reverse_index_sync_and_aux_ast_node_eviction() -> None:
    """Verifies sibling cache reverse-index synchronization and aux_ast_nodes eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_owner_keys_index,
    )

    clear_dataflow_cache()
    mod_a = ast.parse("def fa():\n    x = 1\n    y = 2\n")
    mod_b = ast.parse("def fb():\n    z = 3\n")
    fn_a = mod_a.body[0]
    stmt_x = fn_a.body[0]  # type: ignore[attr-defined]
    stmt_y = fn_a.body[1]  # type: ignore[attr-defined]
    fn_b = mod_b.body[0]

    shared_key = _ScopeCacheKey(fn_a, "shared_cross_cache_test")
    _scope_closures_cache[shared_key] = (stmt_x,)
    _scope_cell_info_cache[shared_key] = {}
    tracked_dict = _scope_cell_info_cache[shared_key]
    tracked_dict["node"] = stmt_y

    assert shared_key in _scope_owner_keys_index.get(stmt_x, set())
    assert shared_key in _scope_owner_keys_index.get(stmt_y, set())

    # Deleting from _scope_cell_info_cache must keep stmt_x indexed for _scope_closures_cache
    del _scope_cell_info_cache[shared_key]
    assert shared_key in _scope_owner_keys_index.get(stmt_x, set())
    assert shared_key not in _scope_owner_keys_index.get(stmt_y, set())

    # Key owned by mod_a with aux_ast_nodes in mod_b must be evicted when mod_b is evicted
    aux_key = _ScopeCacheKey(fn_a, "aux_evict_test", aux_ast_nodes=(fn_b,))
    _scope_cell_info_cache[aux_key] = ("ok",)
    assert aux_key in _scope_owner_keys_index.get(fn_b, set())
    _evict_tree_scopes(mod_b)
    assert aux_key not in _scope_cell_info_cache
    assert aux_key not in _scope_owner_keys_index.get(fn_a, set())


def test_scope_cache_key_frozenset_and_nested_int_float_bool_non_collision() -> None:
    """Verifies _ScopeCacheKey distinguishes int, float, and bool inside frozensets and nested tuples."""
    from pydoppelgangerhunt.fixer._df_cache import _ScopeCacheKey

    mod = ast.parse("x = 1\n")
    k_int = _ScopeCacheKey(mod, "fs", params=(frozenset({2}),))
    k_float = _ScopeCacheKey(mod, "fs", params=(frozenset({2.0}),))
    k_nested_int = _ScopeCacheKey(mod, "fs", params=(frozenset({(1, "a")}),))
    k_nested_bool = _ScopeCacheKey(mod, "fs", params=(frozenset({(True, "a")}),))

    assert k_int != k_float
    assert k_nested_int != k_nested_bool
    assert len({k_int, k_float, k_nested_int, k_nested_bool}) == 4


def test_unit_has_global_mutating_call_enclosing_class_and_nested_shadowing() -> None:
    """Verifies _unit_has_global_mutating_call detects pure-builtin shadowing in class bodies, comprehensions, and walrus targets."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _unit_has_global_mutating_call,
    )

    unshadowed_src = (
        "g_val = 1\n"
        "def run_pure(items):\n"
        "    global g_val\n"
        "    n = len(items)\n"
        "    out = n + g_val\n"
        "    return out\n"
    )
    unshadowed_tree = ast.parse(unshadowed_src)
    sc_unshadow = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": unshadowed_src,
            "source_code": "    n = len(items)\n    out = n + g_val\n",
        },
        tree=unshadowed_tree,
    )
    assert "global_escape" not in sc_unshadow.get("closure_hazard_reasons", ())

    comp_shadow_src = (
        "g_val = 1\n"
        "def run_comp(items):\n"
        "    global g_val\n"
        "    res = [len(items) for len in items]\n"
        "    out = g_val\n"
        "    return res, out\n"
    )
    comp_tree = ast.parse(comp_shadow_src)
    sc_comp_shadow = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": comp_shadow_src,
            "source_code": ("    res = [len(items) for len in items]\n" "    out = g_val\n"),
        },
        tree=comp_tree,
    )
    assert "global_escape" in sc_comp_shadow.get("closure_hazard_reasons", ())

    walrus_shadow_src = (
        "g_val = 1\n"
        "def run_walrus(cb, items):\n"
        "    global g_val\n"
        "    res = ((len := cb), len(items))\n"
        "    out = g_val\n"
        "    return res, out\n"
    )
    walrus_tree = ast.parse(walrus_shadow_src)
    sc_walrus_shadow = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": walrus_shadow_src,
            "source_code": ("    res = ((len := cb), len(items))\n" "    out = g_val\n"),
        },
        tree=walrus_tree,
    )
    assert "global_escape" in sc_walrus_shadow.get("closure_hazard_reasons", ())

    cls_shadow_src = (
        "g_val = 1\n"
        "def outer(items):\n"
        "    class LocalHolder:\n"
        "        global g_val\n"
        "        def len(seq):\n"
        "            return 0\n"
        "        n = len(items)\n"
        "        out = n + g_val\n"
        "    return LocalHolder.out\n"
    )
    cls_tree = ast.parse(cls_shadow_src)
    sc_cls_shadow = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_text": cls_shadow_src,
            "source_code": "        n = len(items)\n        out = n + g_val\n",
        },
        tree=cls_tree,
    )
    assert "global_escape" in sc_cls_shadow.get("closure_hazard_reasons", ())

    lambda_stmts = ast.parse("res = (lambda len: len(items))(cb)\n").body
    assert _unit_has_global_mutating_call(lambda_stmts, set()) is True


def test_clause_header_source_code_normalization_and_multiline_string_comments() -> None:
    """Verifies leading elif/else/except/finally headers and multiline string '#' literals do not trigger stale_source_code."""
    from pydoppelgangerhunt.fixer.scope import _strip_comments_from_lines

    elif_mod_src = (
        "def classify(x):\n"
        "    if x > 10:\n"
        "        out = 'high'\n"
        "    elif x > 0:  # positive branch\n"
        "        out = 'pos'\n"
        "    else:\n"
        "        out = 'neg'\n"
        "    return out\n"
    )
    elif_tree = ast.parse(elif_mod_src)
    sc_elif = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "clause",
            "source_text": elif_mod_src,
            "source_code": "elif x > 0:\n    out = 'pos'\n",
        },
        tree=elif_tree,
    )
    assert "stale_source_code" not in sc_elif["control_flow_hazards"]

    multiline_lines = [
        'msg = """first line\n',
        "# interior hash inside multiline string\n",
        'last line"""  # trailing comment\n',
    ]
    stripped_ml = _strip_comments_from_lines(multiline_lines)
    assert "# interior hash inside multiline string" in stripped_ml[1]
    assert "trailing comment" not in stripped_ml[2]
    assert 'last line"""' in stripped_ml[2]


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
    sug_default = synthesize_refactoring_suggestion(u_safe_1, u_safe_2, repo_root=str(tmp_path))
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
    sug_hazard = synthesize_refactoring_suggestion(u_haz_1, u_haz_2, repo_root=str(tmp_path))
    assert "rejected as unsafe" in sug_hazard
    assert "WARNING: Automatic helper synthesis was skipped" in sug_hazard
    assert "(*args, **kwargs)" not in sug_hazard


def test_concurrent_suppress_syntax_warnings_and_stmt_deltas_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies lock-serialized _suppress_syntax_warnings concurrency with catch_warnings and _MAX_STMT_DELTAS_PER_FN LRU reuse."""
    import concurrent.futures
    import warnings
    import pydoppelgangerhunt.fixer._df_cell_traversal as trav_mod
    from pydoppelgangerhunt.fixer._df_cache import (
        _MAX_STMT_DELTAS_PER_FN,
        _ScopeCacheKey,
        _scope_cell_info_cache,
        _suppress_syntax_warnings,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )

    pre_test_snapshot = list(warnings.filters)

    def _suppress_worker() -> None:
        for _ in range(10):
            with _suppress_syntax_warnings():
                ast.parse(r"s = '\d+'")

    def _catch_warnings_worker() -> None:
        for _ in range(10):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", UserWarning)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_suppress_worker) for _ in range(3)] + [
            pool.submit(_catch_warnings_worker) for _ in range(3)
        ]
        for f in futs:
            f.result(timeout=10.0)

    assert list(warnings.filters) == pre_test_snapshot

    clear_dataflow_cache()
    many_lines = (
        ["def big_fn(x):\n"]
        + [f"    v_{i} = x + {i}\n" for i in range(_MAX_STMT_DELTAS_PER_FN + 20)]
        + ["    return v_0\n"]
    )
    big_tree = ast.parse("".join(many_lines))
    big_fn = big_tree.body[0]
    assert isinstance(big_fn, ast.FunctionDef)
    info = _analyze_enclosing_fn_cells_uncached(
        big_fn,
        _SpanBounds(2, 5),
        mod_tree=big_tree,
        use_stmt_deltas=True,
    )
    assert info is not None
    stmt_deltas = _scope_cell_info_cache.get(
        _ScopeCacheKey(big_fn, "stmt_deltas", aux_ast_nodes=(big_tree,))
    )
    assert isinstance(stmt_deltas, dict)
    assert len(stmt_deltas) == _MAX_STMT_DELTAS_PER_FN

    orig_apply = trav_mod._try_apply_cached_stmt_delta
    pass2_hits = 0

    def _counting_apply(*args: Any, **kwargs: Any) -> Any:
        nonlocal pass2_hits
        res = orig_apply(*args, **kwargs)
        if res is not None:
            pass2_hits += 1
        return res

    monkeypatch.setattr(trav_mod, "_try_apply_cached_stmt_delta", _counting_apply)
    info2 = _analyze_enclosing_fn_cells_uncached(
        big_fn,
        _SpanBounds(2, 6),
        mod_tree=big_tree,
        use_stmt_deltas=True,
    )
    assert info2 is not None
    assert len(stmt_deltas) == _MAX_STMT_DELTAS_PER_FN
    assert pass2_hits >= _MAX_STMT_DELTAS_PER_FN - 10


def test_validate_unit_schema_strict_accepts_strict_with_suppression() -> None:
    """Verifies UnitSchema and validate_unit_schema(strict=True) accept strict_with_suppression."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    unit_with_flag = {
        "start": 1,
        "end": 2,
        "kind": "compound_block",
        "strict_with_suppression": True,
        "skip_pre_unit_closures": False,
        "closure_strictness": "pedantic",
    }
    assert validate_unit_schema(unit_with_flag, strict=True) == set()

    with pytest.raises(ValueError, match="Unrecognized keys"):
        validate_unit_schema(
            {**unit_with_flag, "unrecognized_extra_field": 123},
            strict=True,
        )


def test_module_compound_statement_suppressor_alias_propagation_and_linear_seed() -> None:
    """Verifies suppressor imports inside module-level compound blocks propagate into nested functions."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _ModuleSuppressSeed,
        _build_module_suppress_seed,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    nested_mod_src = (
        "if True:\n"
        "    from contextlib import suppress as sup\n"
        "    def target_fn(d, items):\n"
        "        with sup(KeyError):\n"
        "            val = d['k']\n"
        "        return [val for _ in items]\n"
    )
    mod_tree = ast.parse(nested_mod_src)
    if_stmt = mod_tree.body[0]
    assert isinstance(if_stmt, ast.If)
    target_fn = if_stmt.body[1]
    assert isinstance(target_fn, ast.FunctionDef)

    seeded_prov = _seed_scope_suppress_aliases(target_fn, mod_tree, u_start=target_fn.lineno)
    assert "sup" in seeded_prov.suppressors

    bound, _ = _extract_block_definite_bindings_and_term(
        target_fn.body[:1],
        None,
        strict_with_suppression=False,
        suppress_aliases=seeded_prov,
    )
    assert "val" not in bound

    many_mod_src = "".join(
        [
            "from contextlib import suppress as s0\n",
            *[f"s{i} = s{i - 1}\n" for i in range(1, 50)],
            "def f(x):\n    return x\n",
            "from contextlib import suppress as post_sup\n",
        ]
    )
    many_tree = ast.parse(many_mod_src)
    seed = _build_module_suppress_seed(many_tree)
    f_stmt = many_tree.body[-2]
    assert len(seed.by_top_stmt_id) == len(many_tree.body)
    assert "s49" in seed.prefix_by_top_stmt_id[id(f_stmt)].suppressors
    assert "post_sup" in seed.suffix_delta_by_top_stmt_id[id(f_stmt)].suppressors
    assert {"s49", "post_sup"} <= seed.by_top_stmt_id[id(f_stmt)].suppressors

    from types import MappingProxyType

    default_seed_a = _ModuleSuppressSeed(seeded_prov, MappingProxyType({}), False)
    default_seed_b = _ModuleSuppressSeed(seeded_prov, MappingProxyType({}), False)
    assert default_seed_a.prefix_by_top_stmt_id == {}
    assert default_seed_b.suffix_delta_by_top_stmt_id == {}
    with pytest.raises(TypeError):
        default_seed_a.prefix_by_top_stmt_id[1] = seeded_prov  # type: ignore[index]
    with pytest.raises(TypeError):
        default_seed_b.suffix_delta_by_top_stmt_id[1] = seeded_prov  # type: ignore[index]


@pytest.mark.parametrize(
    "assign_src,cond_var,expected_x_definite",
    [
        ("pair_a, pair_b = (1, custom_obj)", "pair_a", True),
        ("pair_a, pair_b = (1, custom_obj)", "pair_b", False),
        ("pair_a, pair_b = [True, custom_obj]", "pair_a", True),
        ("pair_a, pair_b = [True, custom_obj]", "pair_b", False),
        ("a, b = (1, a)", "b", False),
    ],
)
def test_unpacking_assignment_elementwise_bool_safety_in_suppressing_with(
    assign_src: str,
    cond_var: str,
    expected_x_definite: bool,
) -> None:
    """Verifies tuple/list unpacking inside a suppressing with block only marks proven bool-safe elements."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    fn_src = (
        "from contextlib import suppress\n"
        "def run(custom_obj, a):\n"
        "    with suppress(Exception):\n"
        f"        {assign_src}\n"
        f"        if {cond_var}:\n"
        "            y = 1\n"
        "        x = 1\n"
    )
    mod_tree = ast.parse(fn_src)
    fn_node = mod_tree.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree)
    bound, _ = _extract_block_definite_bindings_and_term(
        fn_node.body,
        None,
        frozenset({"custom_obj", "a"}),
        strict_with_suppression=True,
        suppress_aliases=prov,
    )
    assert ("x" in bound) is expected_x_definite


def test_subroutine_early_return_prunes_dead_conditional_outputs() -> None:
    """Verifies a subroutine with an early return queries downstream reads for pre-bound write-only conditional stores."""
    src = (
        "def compute(abort, cond, x):\n"
        "    dead_tmp = 0\n"
        "    live_tmp = 0\n"
        "    if abort:\n"
        "        return 99\n"
        "    if cond:\n"
        "        dead_tmp = x + 1\n"
        "        live_tmp = x + 2\n"
        "    res = x * 3\n"
        "    return res + live_tmp\n"
    )
    unit_src = (
        "    if abort:\n"
        "        return 99\n"
        "    if cond:\n"
        "        dead_tmp = x + 1\n"
        "        live_tmp = x + 2\n"
        "    res = x * 3\n"
    )
    scope = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 9,
            "kind": "compound_block",
            "source_text": src,
            "source_code": unit_src,
        },
        tree=ast.parse(src),
    )
    assert "dead_tmp" not in scope["inputs"]
    assert "live_tmp" in scope["inputs"]
    assert "res" in scope["outputs"]


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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    indented_cls = "".join(f"    {line}" for line in cls_header.splitlines(keepends=True))
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
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=len(src.splitlines()))
    assert ("LocalMgr" in prov.suppressors) is expected_suppressor

    bound, _ = _extract_block_definite_bindings_and_term(
        fn_node.body,
        None,
        frozenset({"d"}),
        strict_with_suppression=False,
        suppress_aliases=_seed_scope_suppress_aliases(fn_node, mod_tree),
    )
    assert ("val" not in bound) is expected_suppressor


def test_seed_scope_suppress_aliases_clears_function_locals_at_entry() -> None:
    """Verifies function-local assignments shadow module-level non-suppressor imports even when u_start=0."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

    src = (
        "from contextlib import nullcontext\n"
        "def run(custom_cm, d):\n"
        "    nullcontext = custom_cm\n"
        "    with nullcontext():\n"
        "        val = d['k']\n"
        "    from contextlib import suppress as late_sup\n"
        "    return val, late_sup\n"
    )
    mod_tree = ast.parse(src)
    fn_node = mod_tree.body[1]
    assert isinstance(fn_node, ast.FunctionDef)

    prov_zero = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=0)
    assert "nullcontext" not in prov_zero.non_suppressors
    assert "nullcontext" in prov_zero.shadowed_names

    prov_mid = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=3)
    assert "nullcontext" not in prov_mid.non_suppressors
    assert "late_sup" in prov_mid.suppressors

    bound, _ = _extract_block_definite_bindings_and_term(
        fn_node.body[:2],
        None,
        frozenset({"custom_cm", "d"}),
        strict_with_suppression=True,
        suppress_aliases=prov_zero,
    )
    assert "val" not in bound


def test_suppress_syntax_warnings_copy_on_write_and_nan_cache_key_equality() -> None:
    """Verifies _suppress_syntax_warnings uses catch_warnings copy-on-write and _ScopeCacheKey handles NaN."""
    import warnings
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _suppress_syntax_warnings,
    )

    outer_filters_ref = warnings.filters
    outer_snapshot = list(outer_filters_ref)
    with _suppress_syntax_warnings():
        inner_filters_ref = warnings.filters
        assert inner_filters_ref is not outer_filters_ref
        assert outer_filters_ref == outer_snapshot
        with _suppress_syntax_warnings():
            assert warnings.filters is inner_filters_ref
        assert warnings.filters is inner_filters_ref
    assert list(warnings.filters) == outer_snapshot
    assert outer_filters_ref == outer_snapshot

    owner = ast.parse("x = 1\n")
    k_nan_1 = _ScopeCacheKey(owner, "nan_test", (float("nan"), ("nested", float("nan"))))
    k_nan_2 = _ScopeCacheKey(owner, "nan_test", (float("nan"), ("nested", float("nan"))))
    assert k_nan_1 == k_nan_2
    assert hash(k_nan_1) == hash(k_nan_2)
    cache_probe = {k_nan_1: "hit"}
    assert cache_probe.get(k_nan_2) == "hit"


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
    unit_code = (
        "    for x in items:\n"
        "        step = x + 1\n"
        "        yield step\n"
    )
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

    def _boom(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("simulated helper synthesis failure")

    monkeypatch.setattr(synth_mod, "synthesize_shared_helper_code", _boom)
    suggestion = reporters_mod.synthesize_refactoring_suggestion(
        u1, u2, repo_root=str(tmp_path)
    )
    assert "rejected as unsafe" in suggestion
    assert "WARNING: Automatic helper synthesis was skipped" in suggestion


@pytest.mark.parametrize(
    "fn_kw,stmt_lines",
    [
        ("def", ["    with ctx:\n", "        v = G + 1\n"]),
        ("async def", ["    async with ctx:\n", "        v = G + 1\n"]),
        ("async def", ["    await coro\n", "    v = G + 1\n"]),
        ("def", ["    yield 1\n", "    v = G + 1\n"]),
        ("def", ["    yield from seq\n", "    v = G + 1\n"]),
        ("async def", ["    async for item in aiter:\n", "        v = G + 1\n"]),
        ("def", ["    tmp = [1]\n", "    del tmp\n", "    v = G + 1\n"]),
    ],
)
def test_global_escape_rejects_with_await_yield_async_for_and_del(
    fn_kw: str,
    stmt_lines: list[str],
) -> None:
    """Verifies _unit_has_global_mutating_call marks with/async-with/await/yield/async-for/del as potential global mutators."""
    body_str = "".join(stmt_lines)
    end_ln = 2 + len(stmt_lines)
    src = (
        f"{fn_kw} run(ctx, coro, seq, aiter):\n"
        "    global G\n"
        f"{body_str}"
        "    return v\n"
    )
    scope = inspect_single_unit_scope(
        {
            "start": 3,
            "end": end_ln,
            "kind": "compound_block",
            "source_text": src,
            "source_code": body_str,
        },
        tree=ast.parse(src),
    )
    assert "severed_closure_cell" in scope["control_flow_hazards"]
    assert "global_escape" in scope["closure_hazard_reasons"]
    assert scope["is_control_flow_safe"] is False


def test_invalid_lexical_declarations_cache_key_and_use_before_declaration() -> None:
    """Verifies _scope_has_invalid_lexical_declarations detects use-before-global/nonlocal in both AST-only and symtable modes and caches them under distinct keys."""
    from pydoppelgangerhunt.fixer._df_span_index import (
        _scope_has_invalid_lexical_declarations,
        _walk_invalid_lexical_scope_declarations,
    )

    use_before_global_src = "def f():\n    a = 1\n    global a\n"
    tree_g = ast.parse(use_before_global_src)
    fn_g = tree_g.body[0]
    assert (
        _walk_invalid_lexical_scope_declarations(
            [(fn_g, frozenset(), frozenset(), False)]
        )
        is True
    )
    assert _scope_has_invalid_lexical_declarations(fn_g, True, None) is True
    assert (
        _scope_has_invalid_lexical_declarations(
            fn_g, True, use_before_global_src.splitlines(keepends=True)
        )
        is True
    )

    read_before_nonlocal_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def inner():\n"
        "        y = x + 1\n"
        "        nonlocal x\n"
        "        x = y\n"
    )
    tree_nl = ast.parse(read_before_nonlocal_src)
    outer_fn = tree_nl.body[0]
    assert (
        _walk_invalid_lexical_scope_declarations(
            [(outer_fn, frozenset(), frozenset(), False)]
        )
        is True
    )
    assert _scope_has_invalid_lexical_declarations(outer_fn, True, None) is True
    assert (
        _scope_has_invalid_lexical_declarations(
            outer_fn, True, read_before_nonlocal_src.splitlines(keepends=True)
        )
        is True
    )


def test_multiline_parenthesised_decorator_earliest_start_pos_across_pipeline() -> None:
    """Verifies _node_earliest_start_pos uses cached tree source_lines to locate multiline parenthesised `@(` decorators even when source_lines is omitted."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
    )

    src = (
        "def outer(dec):\n"
        "    @(\n"
        "        dec\n"
        "    )\n"
        "    def inner():\n"
        "        return x\n"
        "    x = 1\n"
        "    return inner\n"
    )
    src_lines = src.splitlines(keepends=True)
    mod_tree = ast.parse(src)
    outer_fn = mod_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)

    # First call registers tree_source_lines via mod_tree
    closures = _collect_scope_closures(
        outer_fn,
        module_tree=mod_tree,
        source_lines=src_lines,
    )
    assert any(
        (pos.line, pos.col) == (2, 4) and "x" in reads
        for pos, reads in closures
    )
    # Subsequent call to _node_earliest_start_pos without explicit source_lines resolves line 2 via parent_map + cached tree_source_lines
    assert _node_earliest_start_pos(inner_fn) == (2, 4)

    cell_info = _analyze_enclosing_fn_cells(
        outer_fn,
        _SpanBounds(7, 7),
        mod_tree=mod_tree,
    )
    assert any("x" in reads for name, reads, _ in cell_info.sibling_closures if name == "inner")


def test_harvest_file_units_passes_strict_unit_schema_and_scope_inspection(
    tmp_path: Any,
) -> None:
    """Verifies every unit dict produced by harvest_file_units passes validate_unit_schema(strict=True) and inspect_single_unit_scope without stale_source_code."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema
    from pydoppelgangerhunt.parser import harvest_file_units

    sample_src = (
        "class Widget:\n"
        "    TABLE = [\n"
        "        (1, 'a', True),\n"
        "        (2, 'b', False),\n"
        "        (3, 'c', True),\n"
        "        (4, 'd', False),\n"
        "    ]\n"
        "    def compute(self, items, scale):\n"
        "        def scale_it(v):\n"
        "            return v * scale + 1\n"
        "        vals = [scale_it(x) for x in items if x > 0]\n"
        "        if len(vals) > 2:\n"
        "            total = sum(vals) + (scale * 2 + 3) * (scale - 1)\n"
        "            avg = total / len(vals)\n"
        "            return avg\n"
        "        else:\n"
        "            fallback = (scale + 10) * (scale + 20) - 5\n"
        "            return fallback\n"
    )
    sample_file = tmp_path / "harvest_all_kinds.py"
    sample_file.write_text(sample_src, encoding="utf-8")

    units = harvest_file_units(
        str(sample_file),
        str(tmp_path),
        min_lines=2,
        min_tokens=4,
        functions_only=False,
        sliding_window=True,
        window_size=2,
        complex_expressions=True,
        min_expr_complexity=3,
        clause_level=True,
        data_tables=True,
        class_level=True,
        harvest_closures=True,
        comprehensions=True,
    )
    assert len(units) >= 5
    for u in units:
        assert validate_unit_schema(u, strict=True) == set()
        sc = inspect_single_unit_scope(u, repo_root=str(tmp_path))
        assert "stale_source_code" not in sc["control_flow_hazards"]


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


def test_source_lines_origin_enum_properties_and_hash_contract() -> None:
    """Verifies SourceLinesOrigin is an Enum with transitive equality, consistent hashing, and boolean properties."""
    import enum
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    assert issubclass(SourceLinesOrigin, enum.Enum)
    assert SourceLinesOrigin.PRE_SLICED.is_pre_sliced is True
    assert SourceLinesOrigin.PRE_SLICED.is_placeholder is False
    assert SourceLinesOrigin.PRE_SLICED.has_real_source is True
    assert bool(SourceLinesOrigin.PRE_SLICED) is True

    assert SourceLinesOrigin.PLACEHOLDER.is_pre_sliced is True
    assert SourceLinesOrigin.PLACEHOLDER.is_placeholder is True
    assert SourceLinesOrigin.PLACEHOLDER.has_real_source is False
    assert bool(SourceLinesOrigin.PLACEHOLDER) is True

    assert SourceLinesOrigin.FULL_BUFFER.is_pre_sliced is False
    assert SourceLinesOrigin.FULL_BUFFER.has_real_source is True
    assert bool(SourceLinesOrigin.FULL_BUFFER) is False

    assert SourceLinesOrigin.DISK.is_pre_sliced is False
    assert SourceLinesOrigin.DISK.has_real_source is True
    assert bool(SourceLinesOrigin.DISK) is False

    assert SourceLinesOrigin.PRE_SLICED != "PRE_SLICED"
    assert SourceLinesOrigin.PRE_SLICED != True  # noqa: E712
    assert len(set(SourceLinesOrigin)) == 4


def test_synthesize_shared_helper_code_does_not_mutate_unit_allowlist_across_calls() -> None:
    """Verifies synthesize_shared_helper_code does not mutate caller u1/u2 dictionaries with transient allowlists."""
    import copy

    src = (
        "def f1(custom_cm, flag):\n"
        "    with custom_cm():\n"
        "        if flag:\n"
        "            raise ValueError('boom')\n"
        "        x = 2\n"
        "    cb = lambda: x\n"
        "    out = cb() + 1\n"
        "    return out\n\n"
        "def f2(custom_cm, flag):\n"
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


def test_local_class_dynamic_namespace_mutation_not_proven_non_suppressing() -> None:
    """Verifies local classes with dynamic namespace mutation (locals().update, exec, setattr) are not proven non-suppressing."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
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


def test_global_escape_dunder_builtins_heuristic_vs_pedantic_mode() -> None:
    """Verifies dunder-invoking builtins (len, bool, int, isinstance) are allowed in strict mode but rejected as global_escape in pedantic mode."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _HEURISTIC_GLOBAL_SAFE_BUILTINS,
        _PROVEN_PURE_GLOBAL_SAFE_BUILTINS,
        _PURE_GLOBAL_SAFE_BUILTINS,
        _unit_has_global_mutating_call,
    )

    assert _PROVEN_PURE_GLOBAL_SAFE_BUILTINS == frozenset({"callable", "chr", "id", "ord"})
    assert "len" in _HEURISTIC_GLOBAL_SAFE_BUILTINS
    assert "len" not in _PROVEN_PURE_GLOBAL_SAFE_BUILTINS
    assert _PURE_GLOBAL_SAFE_BUILTINS == (
        _PROVEN_PURE_GLOBAL_SAFE_BUILTINS | _HEURISTIC_GLOBAL_SAFE_BUILTINS
    )

    len_stmts = ast.parse("n = len(obj)\nout = n + G\n").body
    id_stmts = ast.parse("n = id(obj)\nout = n + G\n").body
    assert _unit_has_global_mutating_call(len_stmts, set(), allow_heuristic_builtins=True) is False
    assert _unit_has_global_mutating_call(len_stmts, set(), allow_heuristic_builtins=False) is True
    assert _unit_has_global_mutating_call(id_stmts, set(), allow_heuristic_builtins=False) is False

    dunder_global_src = (
        "G = 1\n"
        "class Mutator:\n"
        "    def __len__(self):\n"
        "        global G\n"
        "        G = 100\n"
        "        return 5\n"
        "def f(m):\n"
        "    global G\n"
        "    a = G\n"
        "    b = len(m)\n"
        "    c = G\n"
        "    return a, b, c\n"
    )
    tree = ast.parse(dunder_global_src)
    base_unit = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": dunder_global_src,
        "source_code": "    a = G\n    b = len(m)\n    c = G\n",
    }
    sc_strict = inspect_single_unit_scope({**base_unit, "closure_strictness": "strict"}, tree=tree)
    sc_pedantic = inspect_single_unit_scope(
        {**base_unit, "closure_strictness": "pedantic"}, tree=tree
    )
    assert "global_escape" not in sc_strict.get("closure_hazard_reasons", ())
    assert "global_escape" in sc_pedantic.get("closure_hazard_reasons", ())
    assert "severed_closure_cell" in sc_pedantic["control_flow_hazards"]


def test_public_evict_tree_scopes_invalidates_mutated_ast_caches() -> None:
    """Verifies public evict_tree_scopes invalidates cached scope intervals after in-place AST coordinate mutation."""
    from pydoppelgangerhunt import evict_tree_scopes as top_evict_tree_scopes
    from pydoppelgangerhunt.fixer import evict_tree_scopes
    from pydoppelgangerhunt.fixer.source import _get_tree_scope_intervals

    assert top_evict_tree_scopes is evict_tree_scopes
    clear_dataflow_cache()
    tree = ast.parse("def alpha():\n    return 1\n")
    intervals_before = _get_tree_scope_intervals(tree)
    assert len(intervals_before) == 1
    assert intervals_before[0][:2] == (1, 2)

    fn_node = tree.body[0]
    fn_node.lineno = 10
    fn_node.end_lineno = 20
    evict_tree_scopes(tree)
    intervals_after = _get_tree_scope_intervals(tree)
    assert intervals_after[0][:2] == (10, 20)


def test_ensure_unit_scope_closure_mode_preserves_terminal_pathological_or_stale_scope() -> None:
    """Verifies ensure_unit_scope_closure_mode preserves terminal pathological_ast/stale_source_code scopes without re-parsing."""
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    stale_scope = {
        "inputs": [],
        "outputs": [],
        "control_flow_hazards": ["stale_source_code"],
        "is_control_flow_safe": False,
        "scope_schema_version": 1,
        "skip_pre_unit_closures": False,
        "strict_with_suppression": False,
        "non_suppressing_ctx_managers": [],
    }
    preserved = ensure_unit_scope_closure_mode(
        {"start": 1, "end": 2},
        stale_scope,
        False,
        strict_with_suppression=True,
        non_suppressing_ctx_managers=["safe_cm"],
    )
    assert preserved is stale_scope
    assert preserved["is_control_flow_safe"] is False
    assert "stale_source_code" in preserved["control_flow_hazards"]


@pytest.mark.parametrize(
    "stmts_src,in_branch,expected_sup,expected_non_sup,expected_shadowed",
    [
        # A. Branch merging (if/else)
        (
            "if cond:\n"
            "    from contextlib import suppress as cm, nullcontext as nc\n"
            "else:\n"
            "    from contextlib import nullcontext as cm, nullcontext as nc\n",
            False,
            {"cm"},
            {"nc"},
            set(),
        ),
        # B. Assignment invalidation
        (
            "from contextlib import nullcontext as safe_cm\n" "safe_cm = unknown_factory()\n",
            False,
            set(),
            set(),
            {"safe_cm"},
        ),
        # C. Deletion invalidation
        (
            "from contextlib import nullcontext as safe_cm, suppress as sup_cm\n"
            "del safe_cm, sup_cm\n",
            False,
            set(),
            set(),
            {"safe_cm", "sup_cm"},
        ),
        # E. Aliasing after rebinding
        (
            "from contextlib import suppress as s1, nullcontext as n1\n"
            "s2 = s1\n"
            "n2 = n1\n"
            "n1 = unknown_factory()\n",
            False,
            {"s1", "s2"},
            {"n2"},
            {"n1"},
        ),
        # F. try/except/else/finally normal vs exceptional flow
        (
            "from contextlib import nullcontext as pre_nc\n"
            "try:\n"
            "    try_nc = pre_nc\n"
            "    may_raise()\n"
            "    post_raise_nc = pre_nc\n"
            "except Exception:\n"
            "    post_raise_nc = unknown_cm()\n"
            "else:\n"
            "    else_nc = try_nc\n"
            "finally:\n"
            "    from contextlib import suppress as fin_sup\n",
            False,
            {"fin_sup"},
            {"pre_nc", "try_nc"},
            {"post_raise_nc"},
        ),
    ],
)
def test_suppress_provenance_dataflow_transitions_truth_table(
    stmts_src: str,
    in_branch: bool,
    expected_sup: set[str],
    expected_non_sup: set[str],
    expected_shadowed: set[str],
) -> None:
    """Verifies _SuppressProvenance transitions across branches, rebinding, del, aliasing, and try/except/else/finally."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _EMPTY_SUPPRESS_PROVENANCE,
        _seed_scope_suppress_aliases,
        _update_suppress_provenance_from_stmts,
    )

    stmts = ast.parse(stmts_src).body
    prov = _update_suppress_provenance_from_stmts(
        stmts, _EMPTY_SUPPRESS_PROVENANCE, in_branch=in_branch
    )
    assert expected_sup <= set(prov.suppress_names)
    assert expected_non_sup <= set(prov.non_suppress_names)
    assert "else_nc" not in prov.non_suppress_names
    for name in expected_shadowed:
        assert name not in prov.non_suppress_names
        assert name in prov.shadowed_names

    # D. Function parameter shadowing of imported symbol
    param_shadow_tree = ast.parse(
        "from contextlib import nullcontext, suppress\n"
        "def inner(nullcontext, open):\n"
        "    return nullcontext, open, suppress\n"
    )
    inner_fn = param_shadow_tree.body[1]
    assert isinstance(inner_fn, ast.FunctionDef)
    seeded = _seed_scope_suppress_aliases(inner_fn, mod_tree=param_shadow_tree)
    assert "nullcontext" not in seeded.non_suppress_names
    assert "nullcontext" in seeded.shadowed_names
    assert "open" in seeded.shadowed_builtins
    assert "suppress" in seeded.suppress_names


@pytest.mark.parametrize(
    "with_src,strict_with,expected_defs,expected_conds",
    [
        # 1. (suppress, nullcontext): item 2 may be skipped if item 1's __enter__ or body raises
        (
            "from contextlib import suppress, nullcontext\n"
            "with suppress(ValueError) as a, nullcontext() as b:\n"
            "    may_raise()\n"
            "    body_v = 1\n",
            False,
            {"a"},
            {"b", "body_v"},
        ),
        # 2. (nullcontext, suppress): item 1 is proven non-suppressing, so `a` and `b` are definite, body after raising call is conditional
        (
            "from contextlib import suppress, nullcontext\n"
            "with nullcontext() as a, suppress(ValueError) as b:\n"
            "    may_raise()\n"
            "    body_v = 1\n",
            False,
            {"a", "b"},
            {"body_v"},
        ),
        # 3. (suppress, unknown_manager): item 1 suppresses, so `b` and `body_v` are conditional
        (
            "from contextlib import suppress\n"
            "with suppress(ValueError) as a, custom_mgr() as b:\n"
            "    may_raise()\n"
            "    body_v = 1\n",
            False,
            {"a"},
            {"b", "body_v"},
        ),
        # 4. (unknown_manager, nullcontext): in pedantic mode (strict_with=True), item 1 may suppress so `b` and `body_v` are conditional
        (
            "from contextlib import nullcontext\n"
            "with custom_mgr() as a, nullcontext() as b:\n"
            "    may_raise()\n"
            "    body_v = 1\n",
            True,
            {"a"},
            {"b", "body_v"},
        ),
        # 5. Walrus inside multi-item with: `manager` in item 1 is definite; `other` in item 2 after suppress is conditional
        (
            "from contextlib import suppress\n"
            "with (manager := suppress(ValueError)) as a, (other := custom_mgr()) as b:\n"
            "    may_raise()\n"
            "    body_v = 1\n",
            False,
            {"manager", "a"},
            {"other", "b", "body_v"},
        ),
    ],
)
def test_multi_item_with_provenance_and_walrus_transitions(
    with_src: str,
    strict_with: bool,
    expected_defs: set[str],
    expected_conds: set[str],
) -> None:
    """Verifies multi-item with statement definite vs conditional assignments and walrus provenance."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
    )

    stmts = ast.parse(with_src).body
    defs, conds = _analyze_block_assignment(
        stmts, strict_with_suppression=strict_with
    )
    assert expected_defs <= defs
    assert not (expected_conds & defs)
    assert expected_conds <= conds


def test_suppress_syntax_warnings_exception_and_nested_thread_restoration() -> None:
    """Verifies _suppress_syntax_warnings restores filters and active count across nested calls and exceptions."""
    import warnings
    from pydoppelgangerhunt.fixer._df_cache import (
        _suppress_syntax_warnings,
        _syntax_warning_active_count,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("always", UserWarning)
        baseline_filters = list(warnings.filters)

        with pytest.raises(RuntimeError, match="simulated parse failure"):
            with _suppress_syntax_warnings():
                assert _syntax_warning_active_count[0] == 1
                with _suppress_syntax_warnings():
                    assert _syntax_warning_active_count[0] == 2
                    raise RuntimeError("simulated parse failure")

        assert _syntax_warning_active_count[0] == 0
        assert list(warnings.filters) == baseline_filters
