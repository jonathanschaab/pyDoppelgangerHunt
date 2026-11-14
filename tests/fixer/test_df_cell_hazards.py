"""Tests for _df_cell_hazards, lexical scope rules, sibling carriers, and global-escape protocols."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
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
)


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
        assert (
            _has_invalid_lexical_scope_declarations(inv_tree.body, mod_tree=inv_tree)
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
    assert (
        _has_invalid_lexical_scope_declarations(v1_tree.body, mod_tree=v1_tree) is False
    )
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
        _has_invalid_lexical_scope_declarations(v2_tree.body, mod_tree=v2_tree) is False
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
        _has_invalid_lexical_scope_declarations(v4_tree.body, mod_tree=v4_tree) is False
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
        _is_non_closure_scalar_expr(ast.parse("len(items)", mode="eval").body, None)
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
        assert (
            "severed_closure_cell" not in sc_obj_mut["control_flow_hazards"]
        ), mut_stmt

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
        assert (
            "severed_closure_cell" in sc_carrier_mut["control_flow_hazards"]
        ), carrier_stmt

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
        "source_code": (
            "for step in (1, 2):\n" "    value += step\n" "    yield value\n"
        ),
    }
    p1_shadowed_scope = inspect_single_unit_scope(
        p1_shadowed_unit, tree=p1_shadowed_tree
    )
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
    muts_unref = _collect_unit_nonlocal_mutators(
        ast.parse(str(cls_unref_unit["source_code"])).body
    )
    assert any(
        m.name == "LocalHelper" and m.is_escaped_def is False for m in muts_unref
    )
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
        assert any(
            m.name == "LocalHelper" and m.is_escaped_def is True for m in c5_muts
        )
        _, c5_esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(
            c5_stmts, frozenset()
        )
        assert any(
            isinstance(s, ast.ClassDef) and id(s) in c5_esc_ids for s in c5_stmts
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
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_a) == (
        set(),
        set(),
    )
    assert _extract_fn_returned_closure_refs(shared_fn, ctx_b) == (
        set(),
        {"any"},
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
        assert not any(
            _key_references_scope_nodes(k, tree_nodes) for k in _scope_cell_info_cache
        )

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


def test_enclosing_global_read_pure_builtin_allowlist_and_star_import_rejection() -> (
    None
):
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
            _unit_has_global_mutating_call(
                ast.parse(f"x = {pure_call}\n").body, frozenset()
            )
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
            _unit_has_global_mutating_call(
                ast.parse(f"x = (g, {impure_call})\n").body, frozenset()
            )
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
            "source_code": (
                "    res = [len(items) for len in items]\n" "    out = g_val\n"
            ),
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
            "source_code": (
                "    res = ((len := cb), len(items))\n" "    out = g_val\n"
            ),
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


def test_global_escape_dunder_builtins_heuristic_vs_pedantic_mode() -> None:
    """Verifies dunder-invoking builtins (len, chr, bool, int, isinstance) are allowed in strict mode but rejected as global_escape in pedantic mode."""
    from pydoppelgangerhunt.fixer._df_cell_hazards import (
        _HEURISTIC_GLOBAL_SAFE_BUILTINS,
        _PROVEN_PURE_GLOBAL_SAFE_BUILTINS,
        _PURE_GLOBAL_SAFE_BUILTINS,
        _unit_has_global_mutating_call,
    )

    assert _PROVEN_PURE_GLOBAL_SAFE_BUILTINS == frozenset({"callable", "id", "ord"})
    assert "len" in _HEURISTIC_GLOBAL_SAFE_BUILTINS
    assert "chr" in _HEURISTIC_GLOBAL_SAFE_BUILTINS
    assert "len" not in _PROVEN_PURE_GLOBAL_SAFE_BUILTINS
    assert "chr" not in _PROVEN_PURE_GLOBAL_SAFE_BUILTINS
    assert _PURE_GLOBAL_SAFE_BUILTINS == (
        _PROVEN_PURE_GLOBAL_SAFE_BUILTINS | _HEURISTIC_GLOBAL_SAFE_BUILTINS
    )

    len_stmts = ast.parse("n = len(obj)\nout = n + G\n").body
    chr_stmts = ast.parse("b = chr(m)\nout = G\n").body
    id_stmts = ast.parse("n = id(obj)\nout = (n, G)\n").body
    id_radd_stmts = ast.parse("n = id(obj)\nout = n + G\n").body
    assert (
        _unit_has_global_mutating_call(len_stmts, set(), allow_heuristic_builtins=True)
        is False
    )
    assert (
        _unit_has_global_mutating_call(len_stmts, set(), allow_heuristic_builtins=False)
        is True
    )
    assert (
        _unit_has_global_mutating_call(chr_stmts, set(), allow_heuristic_builtins=True)
        is False
    )
    assert (
        _unit_has_global_mutating_call(chr_stmts, set(), allow_heuristic_builtins=False)
        is True
    )
    assert (
        _unit_has_global_mutating_call(id_stmts, set(), allow_heuristic_builtins=False)
        is False
    )
    assert (
        _unit_has_global_mutating_call(
            id_radd_stmts, set(), allow_heuristic_builtins=False
        )
        is True
    )

    for call_expr in ("len(m)", "chr(m)"):
        dunder_global_src = (
            "G = 1\n"
            "class Mutator:\n"
            "    def __len__(self):\n"
            "        global G\n"
            "        G = 100\n"
            "        return 5\n"
            "    def __index__(self):\n"
            "        global G\n"
            "        G = 100\n"
            "        return 65\n"
            "def f(m):\n"
            "    global G\n"
            "    a = G\n"
            f"    b = {call_expr}\n"
            "    c = G\n"
            "    return a, b, c\n"
        )
        tree = ast.parse(dunder_global_src)
        base_unit = {
            "start": 13,
            "end": 15,
            "kind": "compound_block",
            "source_text": dunder_global_src,
            "source_code": f"    a = G\n    b = {call_expr}\n    c = G\n",
        }
        sc_strict = inspect_single_unit_scope(
            {**base_unit, "closure_strictness": "strict"}, tree=tree
        )
        sc_pedantic = inspect_single_unit_scope(
            {**base_unit, "closure_strictness": "pedantic"}, tree=tree
        )
        assert "global_escape" not in sc_strict.get("closure_hazard_reasons", ())
        assert "global_escape" in sc_pedantic.get("closure_hazard_reasons", ())
        assert "severed_closure_cell" in sc_pedantic["control_flow_hazards"]


def test_except_handler_and_import_shadowing_of_dunder_builtin_in_global_reading_unit() -> (
    None
):
    """Verifies except-as and import-as bindings of dunder builtins inside global-reading units are caught in strict and pedantic modes."""
    except_shadow_src = (
        "G = 1\n"
        "def run_except(a):\n"
        "    global G\n"
        "    try:\n"
        "        v = G\n"
        "    except Exception as len:\n"
        "        v = len(a)\n"
        "    return v\n"
    )
    import_shadow_src = (
        "G = 1\n"
        "def run_import(a):\n"
        "    global G\n"
        "    import custom_pkg as id\n"
        "    v = id(a) + G\n"
        "    return v\n"
    )
    for mode in ("strict", "pedantic"):
        sc_exc = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 7,
                "kind": "compound_block",
                "source_text": except_shadow_src,
                "source_code": (
                    "    try:\n"
                    "        v = G\n"
                    "    except Exception as len:\n"
                    "        v = len(a)\n"
                ),
                "closure_strictness": mode,
            },
            tree=ast.parse(except_shadow_src),
        )
        assert "global_escape" in sc_exc["closure_hazard_reasons"]

        sc_imp = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 5,
                "kind": "compound_block",
                "source_text": import_shadow_src,
                "source_code": "    import custom_pkg as id\n    v = id(a) + G\n",
                "closure_strictness": mode,
            },
            tree=ast.parse(import_shadow_src),
        )
        assert "global_escape" in sc_imp["closure_hazard_reasons"]


@pytest.mark.parametrize(
    "mutating_stmt",
    [
        "holder.value = 1",
        "mapping['k'] = 1",
        "counter += 1",
        "del holder.value",
        "del mapping['k']",
    ],
)
def test_global_reading_unit_rejects_attribute_subscript_and_augassign_mutations(
    mutating_stmt: str,
) -> None:
    """Verifies global-reading units reject attribute, subscript, and augmented-assignment mutations as global_escape."""
    src = (
        "STATE = 1\n"
        "def run_global(holder, mapping, counter):\n"
        "    global STATE\n"
        "    before = STATE\n"
        f"    {mutating_stmt}\n"
        "    after = STATE\n"
        "    return before, after, counter\n"
    )
    tree = ast.parse(src)
    for mode in ("strict", "pedantic"):
        sc = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 6,
                "kind": "linear_block",
                "source_text": src,
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert "severed_closure_cell" in sc["control_flow_hazards"], (
            mutating_stmt,
            mode,
        )
        assert "global_escape" in sc["closure_hazard_reasons"], (mutating_stmt, mode)
        assert sc["is_control_flow_safe"] is False


def test_global_reading_unit_permits_deferred_nested_function_and_lambda_bodies() -> (
    None
):
    """Verifies deferred calls inside nested def/lambda bodies do not trigger false-positive global_escape."""
    deferred_safe_src = (
        "STATE = 10\n"
        "def make_callbacks(offset):\n"
        "    global STATE\n"
        "    snap = (STATE, offset)\n"
        "    def callback():\n"
        "        print(snap)\n"
        "        return snap\n"
        "    lam = lambda: print(snap)\n"
        "    return callback, lam\n"
    )
    deferred_tree = ast.parse(deferred_safe_src)
    for mode in ("strict", "pedantic"):
        sc_safe = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 8,
                "kind": "compound_block",
                "source_text": deferred_safe_src,
                "closure_strictness": mode,
            },
            tree=deferred_tree,
        )
        assert "global_escape" not in sc_safe["closure_hazard_reasons"], mode
        assert sc_safe["is_control_flow_safe"] is True, mode

    # Binary operator on unproven parameter in global-reading unit is permitted in strict and rejected in pedantic
    binop_src = deferred_safe_src.replace("(STATE, offset)", "STATE + offset")
    binop_tree = ast.parse(binop_src)
    sc_binop_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": binop_src,
            "closure_strictness": "strict",
        },
        tree=binop_tree,
    )
    assert sc_binop_strict["is_control_flow_safe"] is True
    sc_binop_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": binop_src,
            "closure_strictness": "pedantic",
        },
        tree=binop_tree,
    )
    assert sc_binop_pedantic["is_control_flow_safe"] is False
    assert "global_escape" in sc_binop_pedantic["closure_hazard_reasons"]

    # Positive control: call inside a nested function's default argument IS executed at def time
    eager_default_src = (
        "STATE = 10\n"
        "def make_eager_default(impure_fn):\n"
        "    global STATE\n"
        "    snap = STATE\n"
        "    def callback(v=impure_fn()):\n"
        "        return snap + v\n"
        "    return callback\n"
    )
    eager_tree = ast.parse(eager_default_src)
    sc_eager = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": eager_default_src,
            "closure_strictness": "strict",
        },
        tree=eager_tree,
    )
    assert "severed_closure_cell" in sc_eager["control_flow_hazards"]
    assert "global_escape" in sc_eager["closure_hazard_reasons"]


@pytest.mark.parametrize(
    "cond_stmt",
    [
        "if trigger:\n            tag = 1",
        "while trigger:\n            tag = 1\n            break",
        "assert trigger\n        tag = 1",
        "tag = 1 if (trigger and True) else 0",
        "tag = 1 if (trigger or False) else 0",
        "tag = 1 if (not trigger) else 0",
        "tag = 1 if trigger else 0",
        "tag = [1 for _ in (1,) if trigger]",
    ],
)
def test_pedantic_global_escape_rejects_implicit_truth_value_evaluation(
    tmp_path: Path,
    cond_stmt: str,
) -> None:
    """Verifies pedantic mode rejects implicit truth-value evaluation (__bool__/__len__) in global-reading units."""
    src = (
        "STATE = 10\n"
        "class BoolMutator:\n"
        "    def __bool__(self):\n"
        "        global STATE\n"
        "        STATE = 999\n"
        "        return True\n"
        "class LenMutator:\n"
        "    def __len__(self):\n"
        "        global STATE\n"
        "        STATE = 777\n"
        "        return 0\n"
        "def outer():\n"
        "    global STATE\n"
        "    def op_a(trigger):\n"
        "        first = STATE\n"
        f"        {cond_stmt}\n"
        "        second = STATE\n"
        "        return first, second, tag\n"
        "    def op_b(trigger):\n"
        "        first = STATE\n"
        f"        {cond_stmt}\n"
        "        second = STATE\n"
        "        return first, second, tag\n"
        "    return op_a, op_b\n"
    )
    f_path = tmp_path / "implicit_bool_global.py"
    f_path.write_text(src, encoding="utf-8")
    tree = ast.parse(src)
    outer_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "outer"
    )
    op_a = next(
        n for n in outer_fn.body if isinstance(n, ast.FunctionDef) and n.name == "op_a"
    )
    op_b = next(
        n for n in outer_fn.body if isinstance(n, ast.FunctionDef) and n.name == "op_b"
    )
    u1 = {
        "name": "outer.op_a:block",
        "file": str(f_path),
        "start": op_a.body[0].lineno,
        "end": op_a.body[-2].end_lineno,
        "kind": "compound_block",
        "source_text": src,
    }
    u2 = {
        "name": "outer.op_b:block",
        "file": str(f_path),
        "start": op_b.body[0].lineno,
        "end": op_b.body[-2].end_lineno,
        "kind": "compound_block",
        "source_text": src,
    }

    sc_pedantic = inspect_single_unit_scope(
        dict(u1, closure_strictness="pedantic"),
        tree=tree,
    )
    assert sc_pedantic["is_control_flow_safe"] is False
    assert "global_escape" in sc_pedantic["closure_hazard_reasons"]
    patch_pedantic = generate_refactoring_patch(
        [(1.0, dict(u1), dict(u2))],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="pedantic",
    )
    assert patch_pedantic == ""


def test_class_body_descriptor_set_name_and_accumulator_dunder_hazards(
    tmp_path: Path,
) -> None:
    """Verifies class-body descriptor (__set_name__) and @= accumulator hazards are rejected in global-reading units."""
    cls_src = (
        "STATE = 1\n"
        "class Descriptor:\n"
        "    def __set_name__(self, owner, name):\n"
        "        global STATE\n"
        "        STATE = 42\n"
        "def outer(descriptor):\n"
        "    global STATE\n"
        "    def build_a():\n"
        "        before = STATE\n"
        "        class Container:\n"
        "            value = descriptor\n"
        "        after = STATE\n"
        "        return before, after, Container\n"
        "    def build_b():\n"
        "        before = STATE\n"
        "        class Container:\n"
        "            value = descriptor\n"
        "        after = STATE\n"
        "        return before, after, Container\n"
        "    return build_a, build_b\n"
    )
    cls_file = tmp_path / "cls_descriptor.py"
    cls_file.write_text(cls_src, encoding="utf-8")
    cls_tree = ast.parse(cls_src)
    u1_cls = {
        "name": "build_a:block",
        "file": str(cls_file),
        "start": 9,
        "end": 12,
        "kind": "compound_block",
        "source_text": cls_src,
    }
    u2_cls = {
        "name": "build_b:block",
        "file": str(cls_file),
        "start": 15,
        "end": 18,
        "kind": "compound_block",
        "source_text": cls_src,
    }
    for mode in ("strict", "pedantic"):
        sc_cls = inspect_single_unit_scope(
            dict(u1_cls, closure_strictness=mode),
            tree=cls_tree,
        )
        assert sc_cls["is_control_flow_safe"] is False
        assert "global_escape" in sc_cls["closure_hazard_reasons"]
        patch_cls = generate_refactoring_patch(
            [(1.0, dict(u1_cls), dict(u2_cls))],
            repo_root=str(tmp_path),
            replace_clones=True,
            closure_strictness=mode,
        )
        assert patch_cls == ""

    # Static non-descriptor class body in a global-reading unit remains permitted
    static_cls_src = (
        "STATE = 1\n"
        "def outer():\n"
        "    global STATE\n"
        "    def build():\n"
        "        before = STATE\n"
        "        class StaticSpec:\n"
        "            TAG = 'v1'\n"
        "        after = STATE\n"
        "        return before, after, StaticSpec\n"
    )
    static_sc = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 8,
            "kind": "compound_block",
            "source_text": static_cls_src,
            "closure_strictness": "pedantic",
        },
        tree=ast.parse(static_cls_src),
    )
    assert static_sc["is_control_flow_safe"] is True

    # @= on a primitive-initialized accumulator must be rejected in both strict and pedantic modes
    matmul_src = (
        "STATE = 1\n"
        "def outer(rhs):\n"
        "    global STATE\n"
        "    total = 0\n"
        "    first = STATE\n"
        "    total @= rhs\n"
        "    second = STATE\n"
        "    return first, second, total\n"
    )
    matmul_tree = ast.parse(matmul_src)
    for mode in ("strict", "pedantic"):
        sc_mm = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 7,
                "kind": "compound_block",
                "source_text": matmul_src,
                "closure_strictness": mode,
            },
            tree=matmul_tree,
        )
        assert sc_mm["is_control_flow_safe"] is False
        assert "global_escape" in sc_mm["closure_hazard_reasons"]


@pytest.mark.parametrize(
    "dunder_stmt",
    [
        "val = STATE + x",
        "val = -x",
        "val = x[0]",
        "val = x.attr",
        "val = 0\n    for _ in x:\n        val = 1",
        "val = 0\n    for a, b in [x]:\n        val = a",
        "val = f'{x}'",
        "val = (x == 1)",
        "val = (x in (1, 2))",
        "val = {x}",
        "val = {x: 1}",
        "a, b = x\n    val = a",
        "import math as _m\n    val = 1",
        "raise x",
        "val = (x, 1)[0] + 2",
        "val = 0\n    if True:\n        val = x\n    out = val + 1",
        "pre = x_in\n    x_in = 0\n    out = x_in + 1",
        "for x in ():\n        pass\n    out = x + 1",
        "val = 0\n    items = [val + 1 for val in (x,)]",
        "val = 0\n    class LocalSpec:\n        val + 1",
    ],
)
def test_pedantic_global_escape_rejects_unproven_implicit_dunders(
    dunder_stmt: str,
) -> None:
    """Verifies pedantic mode rejects unproven implicit-dunder operations inside global-reading units."""
    stmt_lines = dunder_stmt.splitlines()
    end_line = 4 + len(stmt_lines)
    src = (
        "STATE = 10\n"
        "def run_op(x, x_in=0):\n"
        "    global STATE\n"
        "    first = STATE\n"
        f"    {dunder_stmt}\n"
        "    second = STATE\n"
        "    return first, second\n"
    )
    tree = ast.parse(src)
    sc_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": end_line + 1,
            "kind": "compound_block",
            "source_text": src,
            "closure_strictness": "pedantic",
        },
        tree=tree,
    )
    assert sc_pedantic["is_control_flow_safe"] is False, dunder_stmt
    assert "global_escape" in sc_pedantic["closure_hazard_reasons"], dunder_stmt


def test_pedantic_global_escape_radd_runtime_divergence_control() -> None:
    """Demonstrates runtime divergence when an __radd__ operand rebinds a global read before and after binary addition."""
    orig_src = (
        "STATE = 10\n"
        "class RAddMutator:\n"
        "    def __radd__(self, left):\n"
        "        global STATE\n"
        "        STATE = 99\n"
        "        return left + 5\n"
        "def compute(x):\n"
        "    global STATE\n"
        "    first = STATE\n"
        "    mid = first + x\n"
        "    second = STATE\n"
        "    return first, mid, second\n"
    )
    naive_extracted_src = (
        "STATE = 10\n"
        "class RAddMutator:\n"
        "    def __radd__(self, left):\n"
        "        global STATE\n"
        "        STATE = 99\n"
        "        return left + 5\n"
        "def _naive_helper(STATE_snap, x):\n"
        "    first = STATE_snap\n"
        "    mid = first + x\n"
        "    second = STATE_snap\n"
        "    return first, mid, second\n"
        "def compute(x):\n"
        "    global STATE\n"
        "    return _naive_helper(STATE, x)\n"
    )
    ns_orig: dict[str, Any] = {}
    ns_naive: dict[str, Any] = {}
    exec(compile(orig_src, "<orig_radd>", "exec"), ns_orig)  # pylint: disable=exec-used  # nosec B102
    exec(compile(naive_extracted_src, "<naive_radd>", "exec"), ns_naive)  # pylint: disable=exec-used  # nosec B102
    assert ns_orig["compute"](ns_orig["RAddMutator"]()) == (10, 15, 99)
    assert ns_naive["compute"](ns_naive["RAddMutator"]()) == (10, 15, 10)


@pytest.mark.parametrize(
    "safe_stmt",
    [
        "acc = 1\n    val = acc + 2",
        "a, b = (1, 2)\n    val = a + b",
        "val = 0\n    for i in (1, 2, 3):\n        val = val + i",
        "tag = 'ok'\n    val = f'{tag}:{42}'",
        "val = (x is None)",
        "val = (id(x), callable(x), ord('a'))",
        "arr = (10, 20)\n    val = arr[0]",
        "val = (1 in (1, 2, 3))",
    ],
)
def test_pedantic_global_escape_permits_proven_primitive_operations(
    safe_stmt: str,
) -> None:
    """Verifies pedantic mode permits operations on statically proven primitive literals and identity/C-builtin checks."""
    stmt_lines = safe_stmt.splitlines()
    end_line = 4 + len(stmt_lines)
    src = (
        "STATE = 10\n"
        "def run_safe(x):\n"
        "    global STATE\n"
        "    snap = STATE\n"
        f"    {safe_stmt}\n"
        "    return snap, val\n"
    )
    tree = ast.parse(src)
    sc_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": end_line,
            "kind": "compound_block",
            "source_text": src,
            "closure_strictness": "pedantic",
        },
        tree=tree,
    )
    assert sc_pedantic["is_control_flow_safe"] is True, safe_stmt
    assert "global_escape" not in sc_pedantic["closure_hazard_reasons"], safe_stmt


@pytest.mark.parametrize(
    "async_comp_expr",
    [
        "[item async for item in stream()]",
        "{item async for item in stream()}",
        "{item: 1 async for item in stream()}",
        "[val async for group in stream() async for val in group]",
    ],
)
@pytest.mark.parametrize("strictness", ["strict", "pedantic"])
def test_async_comprehension_in_global_reading_unit_rejected_in_strict_and_pedantic(
    async_comp_expr: str,
    strictness: str,
) -> None:
    """Verifies async comprehensions inside global-reading units trigger global_escape in both strict and pedantic modes."""
    src = (
        "STATE = 10\n"
        "async def run_async(stream):\n"
        "    global STATE\n"
        "    first = STATE\n"
        f"    collected = {async_comp_expr}\n"
        "    second = STATE\n"
        "    return first, second, collected\n"
    )
    tree = ast.parse(src)
    sc = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": src,
            "closure_strictness": strictness,
        },
        tree=tree,
    )
    assert sc["is_control_flow_safe"] is False
    assert "severed_closure_cell" in sc["control_flow_hazards"]
    assert "global_escape" in sc["closure_hazard_reasons"]


def test_async_comprehension_anext_mutating_global_runtime_divergence() -> None:
    """Demonstrates runtime divergence when an async comprehension suspends at __anext__ and mutates an enclosing global."""
    import asyncio

    ns: dict[str, Any] = {}
    exec(
        "STATE = 10\n"
        "class MutatingAsyncStream:\n"
        "    def __init__(self):\n"
        "        self._done = False\n"
        "    def __aiter__(self):\n"
        "        return self\n"
        "    async def __anext__(self):\n"
        "        global STATE\n"
        "        if self._done:\n"
        "            raise StopAsyncIteration\n"
        "        self._done = True\n"
        "        STATE = 77\n"
        "        return 1\n"
        "async def run_inplace():\n"
        "    global STATE\n"
        "    STATE = 10\n"
        "    first = STATE\n"
        "    items = [x async for x in MutatingAsyncStream()]\n"
        "    second = STATE\n"
        "    return first, second, items\n"
        "async def _extracted_snapshot(state_snapshot):\n"
        "    first = state_snapshot\n"
        "    items = [x async for x in MutatingAsyncStream()]\n"
        "    second = state_snapshot\n"
        "    return first, second, items\n"
        "async def run_extracted():\n"
        "    global STATE\n"
        "    STATE = 10\n"
        "    return await _extracted_snapshot(STATE)\n",
        ns,
    )
    inplace_res = asyncio.run(ns["run_inplace"]())
    extracted_res = asyncio.run(ns["run_extracted"]())
    assert inplace_res == (10, 77, [1])
    assert extracted_res == (10, 10, [1])
    assert inplace_res != extracted_res


@pytest.mark.parametrize(
    "call_expr,dunder_def",
    [
        (
            "len(probe)",
            "    def __len__(self):\n        global STATE\n        STATE = 99\n        return 3\n",
        ),
        (
            "chr(probe)",
            "    def __index__(self):\n        global STATE\n        STATE = 99\n        return 65\n",
        ),
    ],
)
def test_strict_builtin_allowlist_dunder_len_and_index_runtime_divergence_controls(
    call_expr: str,
    dunder_def: str,
) -> None:
    """Pins the documented strict-mode allowlist behavior vs pedantic rejection for __len__ and __index__ global mutators."""
    ns: dict[str, Any] = {}
    runtime_src = (
        "STATE = 10\n"
        "class DunderMutator:\n"
        f"{dunder_def}"
        "def run_inplace(probe):\n"
        "    global STATE\n"
        "    STATE = 10\n"
        "    first = STATE\n"
        f"    _ = {call_expr}\n"
        "    second = STATE\n"
        "    return first, second\n"
        "def _helper_snapshot(probe, state_snap):\n"
        "    first = state_snap\n"
        f"    _ = {call_expr}\n"
        "    second = state_snap\n"
        "    return first, second\n"
        "def run_snapshot(probe):\n"
        "    global STATE\n"
        "    STATE = 10\n"
        "    return _helper_snapshot(probe, STATE)\n"
    )
    exec(runtime_src, ns)
    probe_obj = ns["DunderMutator"]()
    assert ns["run_inplace"](probe_obj) == (10, 99)
    assert ns["run_snapshot"](probe_obj) == (10, 10)

    analysis_src = (
        "STATE = 10\n"
        "def inspect_target(probe):\n"
        "    global STATE\n"
        "    first = STATE\n"
        f"    _ = {call_expr}\n"
        "    second = STATE\n"
        "    return first, second\n"
    )
    analysis_tree = ast.parse(analysis_src)
    sc_strict = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": analysis_src,
            "closure_strictness": "strict",
        },
        tree=analysis_tree,
    )
    assert sc_strict["is_control_flow_safe"] is True
    assert "global_escape" not in sc_strict["closure_hazard_reasons"]

    sc_pedantic = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": analysis_src,
            "closure_strictness": "pedantic",
        },
        tree=analysis_tree,
    )
    assert sc_pedantic["is_control_flow_safe"] is False
    assert "global_escape" in sc_pedantic["closure_hazard_reasons"]
