"""Tests for closure-cell carriers, scope modes, UTF-8 slicing, and span indexing."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

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


def test_builtin_call_loads_target_rebinding_vs_mutation_and_hazard_matrix(
    tmp_path: Path,
) -> None:
    """Covers built-in call load/closure preservation, rebound vs mutated targets, and differential execution."""
    import subprocess

    from pydoppelgangerhunt.fixer import _slice_unit_token_lines
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        scope_matches_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import _slice_utf8_line
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _SpanBounds,
        _extract_cached_scope_free_reads,
        _extract_closure_carrier_mutation_targets_and_values,
        _extract_closure_carrier_targets_and_values,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_expr_closure_and_load_info,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _extract_mutated_target_names,
        _extract_root_target_names,
        _find_enclosing_function_chain,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

    # 1. Built-in call handling preserves loaded_names, inline_reads, inline_muts, and called_names_out
    unshadowed_ctx = _FnScopeContext(
        fn_locals=frozenset(),
        parent_map={},
        shadowed_names=frozenset(),
        has_star_import=False,
    )
    len_expr = ast.parse("len(items)", mode="eval").body
    called_unshadowed: set[str] = set()
    r_len, m_len, l_len = _extract_expr_closure_and_load_info(
        [len_expr], unshadowed_ctx, called_names_out=called_unshadowed
    )
    assert r_len == set()
    assert m_len == set()
    assert l_len == {"items"}
    assert called_unshadowed == set()

    called_no_ctx: set[str] = set()
    _, _, l_no_ctx = _extract_expr_closure_and_load_info(
        [len_expr], None, called_names_out=called_no_ctx
    )
    assert l_no_ctx == {"items"}
    assert called_no_ctx == {"len"}

    nested_builtin_expr = ast.parse(
        "any((lambda: captured)() for _ in seq) or int(str(len(other)), base=init_cb())",
        mode="eval",
    ).body
    called_nested: set[str] = set()
    r_nb, m_nb, l_nb = _extract_expr_closure_and_load_info(
        [nested_builtin_expr], unshadowed_ctx, called_names_out=called_nested
    )
    assert "captured" in r_nb
    assert m_nb == set()
    assert {"seq", "other"} <= l_nb
    assert called_nested == {"init_cb"}

    shadowed_ctx = _FnScopeContext(
        fn_locals=frozenset({"len"}),
        parent_map={},
        shadowed_names=frozenset({"len"}),
        has_star_import=False,
    )
    called_shadowed: set[str] = set()
    _, _, l_shad = _extract_expr_closure_and_load_info(
        [len_expr], shadowed_ctx, called_names_out=called_shadowed
    )
    assert l_shad == {"items"}
    assert called_shadowed == {"len"}

    # 2. Rebound target names vs mutated attribute/subscript target names
    stmt_rebind = ast.parse("cb = make_callback()\n").body[0]
    stmt_destruct = ast.parse("first, (second, *rest) = make_callbacks()\n").body[0]
    stmt_attr = ast.parse("obj.callback = make_callback()\n").body[0]
    stmt_sub = ast.parse("mapping[key] = make_callback()\n").body[0]
    stmt_mixed = ast.parse("bound_var, obj.callback = f1, f2\n").body[0]
    assert isinstance(stmt_rebind, ast.Assign)
    assert isinstance(stmt_destruct, ast.Assign)
    assert isinstance(stmt_attr, ast.Assign)
    assert isinstance(stmt_sub, ast.Assign)
    assert isinstance(stmt_mixed, ast.Assign)

    assert _extract_root_target_names(stmt_rebind.targets[0]) == {"cb"}
    assert _extract_mutated_target_names(stmt_rebind.targets[0]) == set()
    assert _extract_root_target_names(stmt_destruct.targets[0]) == {
        "first",
        "second",
        "rest",
    }
    assert _extract_mutated_target_names(stmt_destruct.targets[0]) == set()
    assert _extract_root_target_names(stmt_attr.targets[0]) == set()
    assert _extract_mutated_target_names(stmt_attr.targets[0]) == {"obj"}
    assert _extract_root_target_names(stmt_sub.targets[0]) == set()
    assert _extract_mutated_target_names(stmt_sub.targets[0]) == {"mapping"}
    assert _extract_root_target_names(stmt_mixed.targets[0]) == {"bound_var"}
    assert _extract_mutated_target_names(stmt_mixed.targets[0]) == {"obj"}

    assert _extract_closure_carrier_targets_and_values(stmt_attr) is None
    assert _extract_closure_carrier_targets_and_values(stmt_sub) is None
    attr_mut_carrier = _extract_closure_carrier_mutation_targets_and_values(
        stmt_attr
    )
    sub_mut_carrier = _extract_closure_carrier_mutation_targets_and_values(
        stmt_sub
    )
    assert attr_mut_carrier is not None and attr_mut_carrier[0] == {"obj"}
    assert sub_mut_carrier is not None and sub_mut_carrier[0] == {"mapping"}

    alias_src = (
        "def outer(obj, mapping, key, source_cb):\n"
        "    obj.callback = source_cb\n"
        "    mapping[key] = source_cb\n"
        "    direct_alias = source_cb\n"
        "    left_a, *right_b = source_cb, source_cb\n"
        "    return direct_alias, left_a, right_b\n"
    )
    alias_tree = ast.parse(alias_src)
    alias_info = _analyze_enclosing_fn_cells(
        alias_tree.body[0], (6, 6, None, None), mod_tree=alias_tree
    )
    alias_dict = dict(alias_info.alias_edges)
    assert "obj" not in alias_dict
    assert "mapping" not in alias_dict
    assert alias_dict.get("direct_alias") == frozenset({"source_cb"})
    assert alias_dict.get("left_a") == frozenset({"source_cb"})
    assert alias_dict.get("right_b") == frozenset({"source_cb"})

    # 3. Paired positive (rebind captured cell) vs negative (mutate referenced object) hazard checks
    src_rebind_captured = (
        "def outer(items):\n"
        "    value = 1\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    value = 2\n"
        "    return cbs\n"
    )
    sc_rebind = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_rebind_captured,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert "severed_closure_cell" in sc_rebind["control_flow_hazards"]

    src_mutate_captured_obj = (
        "def outer(items):\n"
        "    value = []\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: len(value) + x)\n"
        "    value.append(2)\n"
        "    return cbs\n"
    )
    sc_mutate_obj = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_mutate_captured_obj,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: len(value) + x)\n",
        }
    )
    assert "severed_closure_cell" not in sc_mutate_obj["control_flow_hazards"]
    assert sc_mutate_obj["is_control_flow_safe"] is True

    # 4. Cache eviction across NamedTuple/frozenset keys, statement nodes, and defensive copy immutability
    cache_tree = ast.parse(
        "def outer(x):\n"
        "    def inner():\n"
        "        nonlocal x\n"
        "        x += 1\n"
        "        return x\n"
        "    return inner\n"
    )
    outer_node = cache_tree.body[0]
    assert isinstance(outer_node, ast.FunctionDef)
    inner_node = outer_node.body[0]
    assert isinstance(inner_node, ast.FunctionDef)

    mut_s, del_s = _extract_closure_nonlocal_mutations_and_deletions(inner_node)
    mut_s.add("corrupted")
    del_s.add("corrupted")
    mut_s2, del_s2 = _extract_closure_nonlocal_mutations_and_deletions(inner_node)
    assert "corrupted" not in mut_s2
    assert "corrupted" not in del_s2

    free_s = _extract_cached_scope_free_reads(inner_node)
    free_s.add("corrupted")
    assert "corrupted" not in _extract_cached_scope_free_reads(inner_node)

    ret_l, ret_c = _extract_fn_returned_closure_refs(outer_node)
    ret_l.add("corrupted")
    ret_c.add("corrupted")
    ret_l2, ret_c2 = _extract_fn_returned_closure_refs(outer_node)
    assert "corrupted" not in ret_l2
    assert "corrupted" not in ret_c2

    carriers_s = _extract_fn_outer_carrier_targets(outer_node)
    carriers_s.add("corrupted")
    assert "corrupted" not in _extract_fn_outer_carrier_targets(outer_node)

    nt_key = (_SpanBounds(1, 2), frozenset({inner_node.body[0]}))
    _scope_cell_info_cache[nt_key] = "nt_val"
    _evict_tree_scopes(cache_tree)
    assert nt_key not in _scope_cell_info_cache

    # 5. Column bounds resolution: two statements on one line, Unicode prefix, and ambiguous fail-closed
    two_stmt_src = (
        "def f(xs, ys):\n"
        "    a = [x + 1 for x in xs]; b = [y + 2 for y in ys]\n"
        "    return a, b\n"
    )
    two_stmt_tree = ast.parse(two_stmt_src)
    cand_b = ast.parse("[y + 2 for y in ys]\n").body
    s_col_b, e_col_b = _resolve_unit_col_bounds(
        cand_b, two_stmt_tree, 2, 2, None, None
    )
    assert s_col_b == 33
    assert e_col_b == 52

    ambig_src = (
        "def f(xs):\n"
        "    return ([x for x in xs], [x for x in xs])\n"
    )
    ambig_tree = ast.parse(ambig_src)
    cand_ambig = ast.parse("[x for x in xs]\n").body
    s_ambig, e_ambig = _resolve_unit_col_bounds(
        cand_ambig, ambig_tree, 2, 2, None, None
    )
    assert s_ambig is None
    assert e_ambig is None

    dec_src = (
        "@dec1(\n"
        "    arg1\n"
        ")\n"
        "def f(xs):\n"
        "    return [x for x in xs]\n"
    )
    dec_tree = ast.parse(dec_src)
    assert _find_enclosing_function_chain(dec_tree, 2, 2) == []
    assert len(_find_enclosing_function_chain(dec_tree, 5, 5)) == 1

    # 6. Lexical scope: intermediate function declaring `global x` is a PEP 3104 SyntaxError
    # when nested `inner` declares `nonlocal x`, and fails closed if constructed synthetically.
    invalid_lexical_src = (
        "def outer():\n"
        "    x = 1\n"
        "    def mid():\n"
        "        global x\n"
        "        x = 2\n"
        "        def inner():\n"
        "            nonlocal x\n"
        "            x = 3\n"
        "    f = lambda: x\n"
        "    return f, mid\n"
    )
    with pytest.raises(SyntaxError):
        compile(invalid_lexical_src, "<invalid_lexical>", "exec")
    global_barrier_tree = ast.parse(invalid_lexical_src)
    outer_gb = global_barrier_tree.body[0]
    assert isinstance(outer_gb, ast.FunctionDef)
    mid_gb = outer_gb.body[1]
    assert isinstance(mid_gb, ast.FunctionDef)
    clear_dataflow_cache()
    mid_muts, _ = _extract_closure_nonlocal_mutations_and_deletions(mid_gb)
    assert "x" in mid_muts
    sc_invalid_lex = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "closure",
            "source_text": invalid_lexical_src,
            "source_code": "f = lambda: x\n",
        },
        tree=global_barrier_tree,
    )
    assert "severed_closure_cell" in sc_invalid_lex["control_flow_hazards"]
    assert sc_invalid_lex["is_control_flow_safe"] is False

    # 7. UTF-8 byte offsets and _slice_unit_token_lines with accented and CJK characters and CRLF
    line_accent = '    result = "é" + value\r\n'
    line_cjk = '    result = "界" + value\r\n'
    tree_accent = ast.parse(line_accent.strip())
    rhs_accent = tree_accent.body[0].value  # type: ignore[attr-defined]
    tree_cjk = ast.parse(line_cjk.strip())
    rhs_cjk = tree_cjk.body[0].value  # type: ignore[attr-defined]
    assert (
        _slice_utf8_line(
            line_accent,
            4 + rhs_accent.col_offset,
            4 + rhs_accent.end_col_offset,
        )
        == '"é" + value'
    )
    assert (
        _slice_utf8_line(
            line_cjk,
            4 + rhs_cjk.col_offset,
            4 + rhs_cjk.end_col_offset,
        )
        == '"界" + value'
    )
    multiline_utf8 = [
        'prefix = "é界" + (\r\n',
        "    value_a + value_b\r\n",
        ') + "é界"\r\n',
    ]
    start_b = len('prefix = "é界" + '.encode("utf-8"))
    end_b = len(")".encode("utf-8"))
    sliced_multi = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": start_b, "end_col": end_b},
        multiline_utf8,
    )
    assert sliced_multi == ["(\r\n", "    value_a + value_b\r\n", ")"]

    # 8 & 9. Closure-strictness mode transitions and hazard propagation across refactoring entry points
    lenient_src = (
        "def outer(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "    return items\n"
    )
    unit_base = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": lenient_src,
        "source_code": "for x in items:\n    acc = x * 2\n",
    }
    sc_strict = ensure_unit_scope_closure_mode(
        unit_base,
        None,
        skip_pre_unit_closures=False,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_strict, False) is True
    assert scope_matches_closure_mode(sc_strict, True) is False
    assert "severed_closure_cell" in sc_strict["control_flow_hazards"]

    sc_lenient = ensure_unit_scope_closure_mode(
        unit_base,
        sc_strict,
        skip_pre_unit_closures=True,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_lenient, True) is True
    assert "severed_closure_cell" not in sc_lenient["control_flow_hazards"]

    sc_back_to_strict = ensure_unit_scope_closure_mode(
        unit_base,
        sc_lenient,
        skip_pre_unit_closures=False,
        source_text=lenient_src,
    )
    assert scope_matches_closure_mode(sc_back_to_strict, False) is True
    assert "severed_closure_cell" in sc_back_to_strict["control_flow_hazards"]

    # Returned closure hazard cannot be bypassed even when skip_pre_unit_closures=True
    ret_hazard_src = (
        "def outer1(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "        acc += 1\n"
        "    return cbs\n\n"
        "def outer2(items):\n"
        "    cbs = [lambda: acc]\n"
        "    for x in items:\n"
        "        acc = x * 2\n"
        "        acc += 1\n"
        "    return cbs\n"
    )
    ret_file = tmp_path / "ret_hazard.py"
    ret_file.write_text(ret_hazard_src, encoding="utf-8")
    u1_ret = {
        "file": str(ret_file),
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "source_text": ret_hazard_src,
        "source_code": "for x in items:\n    acc = x * 2\n    acc += 1\n",
        "skip_pre_unit_closures": True,
    }
    u2_ret = {
        "file": str(ret_file),
        "start": 10,
        "end": 12,
        "kind": "compound_block",
        "source_text": ret_hazard_src,
        "source_code": "for x in items:\n    acc = x * 2\n    acc += 1\n",
        "skip_pre_unit_closures": True,
    }
    sc_u1_ret = inspect_single_unit_scope(u1_ret, repo_root=str(tmp_path))
    sc_u2_ret = inspect_single_unit_scope(u2_ret, repo_root=str(tmp_path))
    assert resolve_clone_pair_outputs(u1_ret, u2_ret, sc_u1_ret, sc_u2_ret) is None
    assert (
        resolve_clone_generator_subroutine_outputs(
            u1_ret, u2_ret, sc_u1_ret, sc_u2_ret
        )
        is None
    )
    patch_res = generate_refactoring_patch(
        [(1.0, u1_ret, u2_ret)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch_res == ""

    # 10. Differential execution verifying captured mutable side effects and pre/post callback observations
    safe_mut_src = (
        "def run_first(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    before = snap()\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            history.append(x * 3)\n"
        "    after = snap()\n"
        "    return before, after, snap\n\n"
        "def run_second(items):\n"
        "    history = []\n"
        "    snap = lambda: list(history)\n"
        "    before = snap()\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            history.append(x * 3)\n"
        "    after = snap()\n"
        "    return before, after, snap\n"
    )
    safe_file = tmp_path / "safe_mut.py"
    safe_file.write_text(safe_mut_src, encoding="utf-8")
    (tmp_path / ".git").mkdir(exist_ok=True)
    u1_safe = {
        "file": str(safe_file),
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": safe_mut_src,
        "source_code": "for x in items:\n    if x > 0:\n        history.append(x * 3)\n",
    }
    u2_safe = {
        "file": str(safe_file),
        "start": 15,
        "end": 17,
        "kind": "compound_block",
        "source_text": safe_mut_src,
        "source_code": "for x in items:\n    if x > 0:\n        history.append(x * 3)\n",
    }
    safe_patch = generate_refactoring_patch(
        [(1.0, u1_safe, u2_safe)], repo_root=str(tmp_path), replace_clones=True
    )
    assert safe_patch != ""
    import shutil  # pylint: disable=import-outside-toplevel

    if shutil.which("git") is None:
        pytest.skip("git executable is required for patch application")
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=safe_patch,
        text=True,
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    patched_src = safe_file.read_text(encoding="utf-8")
    ns_orig: dict[str, Any] = {}
    ns_ref: dict[str, Any] = {}
    exec(  # pylint: disable=exec-used
        compile(safe_mut_src, "<orig>", "exec"), ns_orig
    )  # nosec B102
    exec(  # pylint: disable=exec-used
        compile(patched_src, "<ref>", "exec"), ns_ref
    )  # nosec B102
    b_o, a_o, snap_o = ns_orig["run_first"]([-1, 2, 4])
    b_r, a_r, snap_r = ns_ref["run_first"]([-1, 2, 4])
    assert (b_o, a_o, snap_o(), snap_o()) == (b_r, a_r, snap_r(), snap_r())
    assert a_r == [6, 12]


@pytest.mark.parametrize(
    "u_start_col,u_end_col,expected_in_bounds",
    [
        (None, None, [True, True, True]),
        (4, 9, [True, False, False]),
        (11, 20, [False, True, False]),
        (22, 31, [False, False, True]),
    ],
)
def test_is_ast_node_within_unit_bounds_and_visitors_column_filtering(
    u_start_col: Optional[int],
    u_end_col: Optional[int],
    expected_in_bounds: list[bool],
) -> None:
    """Verifies _is_ast_node_within_unit_bounds, _UnitLiveInVisitor, and _DownstreamReadVisitor column bounds."""
    from pydoppelgangerhunt.fixer._df_visitors import (
        _DownstreamReadVisitor,
        _UnitLiveInVisitor,
        _is_ast_node_within_unit_bounds,
    )

    src = "def f(a, b, c):\n    x = a; y = x + b; z = y + c\n"
    fn_node = ast.parse(src).body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    stmts = fn_node.body
    assert len(stmts) == 3

    actual = [
        _is_ast_node_within_unit_bounds(
            s, 2, 2, u_end_col=u_end_col, u_start_col=u_start_col
        )
        for s in stmts
    ]
    assert actual == expected_in_bounds
    assert _is_ast_node_within_unit_bounds(stmts[0], 2, 2) is True

    # Middle statement on line 2: `y = x + b` (col 11..20)
    s_mid = stmts[1]
    mid_start = s_mid.col_offset
    mid_end = s_mid.end_col_offset
    live_in_vis = _UnitLiveInVisitor(
        2, 2, u_end_col=mid_end, u_start_col=mid_start, scope_node=fn_node
    )
    live_in_vis.visit(fn_node)
    assert live_in_vis.live_ins == {"x", "b"}

    ds_vis = _DownstreamReadVisitor(
        2, 2, u_end_col=mid_end, u_start_col=mid_start
    )
    for s in stmts:
        ds_vis.visit(s)
    assert "y" in ds_vis.loaded
    assert "c" in ds_vis.loaded
    assert "a" not in ds_vis.loaded


def test_scope_preservation_carrier_shadowing_and_concurrency() -> None:
    """Covers hand-built scope preservation, returned closure shadowing, escaped defs, and cache concurrency."""
    import threading
    from types import MappingProxyType

    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_intra_unit_defs,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
        _build_enclosing_chain_cell_infos,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        ensure_unit_scope_closure_mode,
    )

    # 1. Hand-built scopes and pathological_ast fallback scopes are preserved by ensure_unit_scope_closure_mode
    hand_built: dict[str, Any] = {
        "inputs": ["custom_in"],
        "outputs": ["custom_out"],
        "stores": ["custom_out"],
        "free_vars": ["custom_in"],
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
    marked_hand_built = {**hand_built, "_skip_pre_unit_closures": True}
    assert (
        ensure_unit_scope_closure_mode(
            {"start": 1, "end": 1}, marked_hand_built, True
        )
        is marked_hand_built
    )
    unmarked_fallback = ensure_unit_scope_closure_mode(
        {"start": 1, "end": 1}, hand_built, True
    )
    assert unmarked_fallback is not hand_built
    assert unmarked_fallback["_skip_pre_unit_closures"] is True
    assert unmarked_fallback["is_control_flow_safe"] is False
    assert "severed_closure_cell" in unmarked_fallback["control_flow_hazards"]
    pathological_scope: dict[str, Any] = {
        **hand_built,
        "_skip_pre_unit_closures": False,
        "control_flow_hazards": ["pathological_ast"],
        "is_control_flow_safe": False,
    }
    assert (
        ensure_unit_scope_closure_mode(
            {"start": 1, "end": 1}, pathological_scope, True
        )
        is pathological_scope
    )
    combined_scope = analyze_unit_variable_scope(
        {"start": 1, "end": 1, "skip_pre_unit_closures": True},
        {"start": 2, "end": 2, "skip_pre_unit_closures": False},
        scope1={**hand_built, "_skip_pre_unit_closures": True},
        scope2={**hand_built, "_skip_pre_unit_closures": False},
    )
    assert isinstance(combined_scope["_skip_pre_unit_closures"], bool)
    assert combined_scope["_skip_pre_unit_closures"] is False

    # 2. _FnScopeContext.parent_map is wrapped in immutable MappingProxyType
    ctx_tree = ast.parse("def f(x):\n    return x + 1\n")
    fn_ctx = _get_fn_scope_context(ctx_tree.body[0], mod_tree=ctx_tree)
    assert isinstance(fn_ctx.parent_map, MappingProxyType)
    with pytest.raises(TypeError):
        fn_ctx.parent_map[ctx_tree] = ctx_tree  # type: ignore[index]

    # 3. _extract_fn_returned_closure_refs respects local parameter shadowing built-in `len`
    shad_ret_tree = ast.parse(
        "def outer(cb):\n"
        "    def helper(len):\n"
        "        return len(cb)\n"
        "    return helper\n"
    )
    outer_fn = shad_ret_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    helper_fn = outer_fn.body[0]
    assert isinstance(helper_fn, ast.FunctionDef)
    outer_ctx = _get_fn_scope_context(outer_fn, mod_tree=shad_ret_tree)
    ret_loads, _ = _extract_fn_returned_closure_refs(helper_fn, outer_ctx)
    assert "cb" in ret_loads

    # 4. _collect_escaped_intra_unit_defs: sibling nonlocal rebind order-independence and global nested defs
    order_a = ast.parse(
        "def target():\n"
        "    return 1\n"
        "def caller():\n"
        "    return target()\n"
        "def rebinder():\n"
        "    nonlocal target\n"
        "    def target():\n"
        "        return 2\n"
        "    return target\n"
        "caller()\n"
        "rebinder()\n"
    ).body
    order_b = ast.parse(
        "def target():\n"
        "    return 1\n"
        "def rebinder():\n"
        "    nonlocal target\n"
        "    def target():\n"
        "        return 2\n"
        "    return target\n"
        "def caller():\n"
        "    return target()\n"
        "caller()\n"
        "rebinder()\n"
    ).body
    names_a, ids_a = _collect_escaped_intra_unit_defs(order_a, frozenset())
    names_b, ids_b = _collect_escaped_intra_unit_defs(order_b, frozenset())
    assert names_a == names_b
    assert "target" in names_a
    assert len(ids_a) == len(ids_b) == 2

    global_def_stmts = ast.parse(
        "def builder():\n"
        "    global exported_fn\n"
        "    def exported_fn():\n"
        "        return 42\n"
        "builder()\n"
    ).body
    g_names, g_ids = _collect_escaped_intra_unit_defs(
        global_def_stmts, frozenset()
    )
    assert "exported_fn" in g_names
    assert len(g_ids) == 1

    # 5. _extract_fn_outer_carrier_targets: class body attribute vs nonlocal & lambda walrus local
    cls_carrier_tree = ast.parse(
        "def outer():\n"
        "    slot = None\n"
        "    lam_slot = None\n"
        "    def worker(y):\n"
        "        nonlocal slot, lam_slot\n"
        "        class LocalAttrOnly:\n"
        "            slot = lambda: y\n"
        "        f = lambda: (lam_slot := (lambda: y))\n"
        "        return LocalAttrOnly, f\n"
    )
    worker_fn = cls_carrier_tree.body[0].body[2]  # type: ignore[attr-defined]
    assert isinstance(worker_fn, ast.FunctionDef)
    assert _extract_fn_outer_carrier_targets(worker_fn) == set()

    cls_nl_tree = ast.parse(
        "def outer():\n"
        "    slot = None\n"
        "    def worker(y):\n"
        "        nonlocal slot\n"
        "        class WithMethodNonlocal:\n"
        "            def set_slot(self):\n"
        "                nonlocal slot\n"
        "                slot = lambda: y\n"
        "        return WithMethodNonlocal\n"
    )
    worker_nl_fn = cls_nl_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert isinstance(worker_nl_fn, ast.FunctionDef)
    assert _extract_fn_outer_carrier_targets(worker_nl_fn) == {"slot"}

    # 6. _unit_cols requires exact match when one endpoint is already provided
    line_dup = ["    x = foo(1); y = foo(1)\n"]
    s_exact, e_exact = _unit_cols(
        {"source_code": "foo(1)", "start_col": 8}, line_dup
    )
    assert (s_exact, e_exact) == (8, 14)

    # 7. Eviction work check on 5,000 composite keys with 300-name frozensets
    clear_dataflow_cache()
    evict_tree = ast.parse("def target_fn(x):\n    return x + 1\n")
    other_tree = ast.parse("def keep_fn(x):\n    return x + 2\n")
    big_names = frozenset(f"var_name_{i}" for i in range(300))
    for i in range(2500):
        _scope_cell_info_cache[
            (evict_tree.body[0], "outer_carriers", big_names, i)
        ] = frozenset()
        _scope_cell_info_cache[
            (other_tree.body[0], "outer_carriers", big_names, i)
        ] = frozenset()
    assert len(_scope_cell_info_cache) == 5000
    _evict_tree_scopes(evict_tree)
    assert len(_scope_cell_info_cache) == 2500
    assert all(k[0] is other_tree.body[0] for k in _scope_cell_info_cache)

    # 8. Multi-threaded concurrency across _analyze_enclosing_fn_cells, _build_enclosing_chain_cell_infos, and eviction
    clear_dataflow_cache()
    conc_src = (
        "def outer(items):\n"
        "    scale = 2\n"
        "    class Box:\n"
        "        def compute(self, val):\n"
        "            cb = lambda x: (x + scale + val)\n"
        "            return [cb(i) for i in items]\n"
        "    for k in items:\n"
        "        tmp = k * scale\n"
        "    return Box, tmp\n"
    )
    conc_tree = ast.parse(conc_src)
    outer_c = conc_tree.body[0]
    assert isinstance(outer_c, ast.FunctionDef)
    box_c = outer_c.body[1]
    assert isinstance(box_c, ast.ClassDef)
    comp_c = box_c.body[0]
    assert isinstance(comp_c, ast.FunctionDef)
    chain = [(1, 9, outer_c), (4, 6, comp_c)]

    def _worker(tid: int) -> bool:
        for step in range(25):
            infos = _build_enclosing_chain_cell_infos(
                chain,
                6,
                6,
                19,
                41,
                frozenset(),
                mod_tree=conc_tree,
            )
            info_loop = _analyze_enclosing_fn_cells(
                outer_c, (8, 8, 8, 23), mod_tree=conc_tree
            )
            if tid == 0 and step % 10 == 0:
                _evict_tree_scopes(ast.parse("def ephemeral(): pass\n"))
            if "scale" not in infos[0].pre_bound or "scale" not in info_loop.pre_bound:
                return False
        return True

    results: list[bool] = [False] * 6
    threads = [
        threading.Thread(target=lambda idx=i: results.__setitem__(idx, _worker(idx)))
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results)


def test_unmarked_scope_recomputation_slice_truncation_and_scope_modes(
    tmp_path: Any,
) -> None:
    """Covers unmarked precomputed scopes, multiline slice end_col truncation, and class/closure propagation."""
    import sys  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (
        _is_irrefutable_case,
        _is_irrefutable_pattern,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _propagate_sibling_closures,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _is_eagerly_consumed_genexp,
    )
    from pydoppelgangerhunt.fixer.scope import (
        ensure_unit_scope_closure_mode,
        scope_matches_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    clear_dataflow_cache()

    # Unmarked scopes never match strict or lenient mode and are recomputed when unit has source
    gen_src = (
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
    gen_file = tmp_path / "unmarked_modes.py"
    gen_file.write_text(gen_src, encoding="utf-8")
    u_g1 = {
        "file": str(gen_file),
        "start": 3,
        "end": 5,
        "name": "gen_a:For",
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }
    u_g2 = {
        "file": str(gen_file),
        "start": 10,
        "end": 12,
        "name": "gen_b:For",
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "for x in items:\n    step = x + 1\n    yield step\n",
    }

    strict_sc1 = inspect_single_unit_scope(u_g1, repo_root=str(tmp_path))
    strict_sc2 = inspect_single_unit_scope(u_g2, repo_root=str(tmp_path))
    assert scope_matches_closure_mode(strict_sc1, False) is True
    assert scope_matches_closure_mode(strict_sc1, True) is False

    unmarked_strict1 = dict(strict_sc1)
    unmarked_strict2 = dict(strict_sc2)
    unmarked_strict1.pop("_skip_pre_unit_closures", None)
    unmarked_strict1.pop("skip_pre_unit_closures", None)
    unmarked_strict2.pop("_skip_pre_unit_closures", None)
    unmarked_strict2.pop("skip_pre_unit_closures", None)
    assert scope_matches_closure_mode(unmarked_strict1, False) is False
    assert scope_matches_closure_mode(unmarked_strict1, True) is False
    assert scope_matches_closure_mode(
        {**unmarked_strict1, "_skip_pre_unit_closures": None}, False
    ) is False

    # Passing unmarked strict scopes into lenient mode recomputes to lenient
    recomputed_lenient = ensure_unit_scope_closure_mode(
        u_g1, unmarked_strict1, True, repo_root=str(tmp_path)
    )
    assert recomputed_lenient["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in recomputed_lenient["control_flow_hazards"]
    res_gen_lenient = resolve_clone_generator_subroutine_outputs(
        u_g1,
        u_g2,
        unmarked_strict1,
        unmarked_strict2,
        repo_root=str(tmp_path),
        closure_strictness="lenient",
        return_details=True,
    )
    assert res_gen_lenient.outputs is not None
    unmarked_comb_strict = analyze_unit_variable_scope(
        u_g1, u_g2, repo_root=str(tmp_path)
    )
    unmarked_comb_strict.pop("_skip_pre_unit_closures", None)
    unmarked_comb_strict.pop("skip_pre_unit_closures", None)
    synth_from_unmarked_strict = synthesize_shared_helper_code(
        u_g1,
        u_g2,
        repo_root=str(tmp_path),
        scope1=unmarked_strict1,
        scope2=unmarked_strict2,
        scope=unmarked_comb_strict,
        closure_strictness="lenient",
    )
    assert "def " in synth_from_unmarked_strict

    # Passing unmarked lenient scopes into strict mode recomputes to strict
    lenient_sc1 = inspect_single_unit_scope(
        {**u_g1, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    lenient_sc2 = inspect_single_unit_scope(
        {**u_g2, "skip_pre_unit_closures": True}, repo_root=str(tmp_path)
    )
    unmarked_lenient1 = dict(lenient_sc1)
    unmarked_lenient2 = dict(lenient_sc2)
    unmarked_lenient1.pop("_skip_pre_unit_closures", None)
    unmarked_lenient1.pop("skip_pre_unit_closures", None)
    unmarked_lenient2.pop("_skip_pre_unit_closures", None)
    unmarked_lenient2.pop("skip_pre_unit_closures", None)
    recomputed_strict = ensure_unit_scope_closure_mode(
        u_g1, unmarked_lenient1, False, repo_root=str(tmp_path)
    )
    assert recomputed_strict["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in recomputed_strict["control_flow_hazards"]
    res_gen_strict = resolve_clone_generator_subroutine_outputs(
        u_g1,
        u_g2,
        unmarked_lenient1,
        unmarked_lenient2,
        repo_root=str(tmp_path),
        closure_strictness="strict",
        return_details=True,
    )
    assert res_gen_strict.outputs is None
    assert res_gen_strict.skip_reason == "severed closure cell"
    unmarked_comb_lenient = analyze_unit_variable_scope(
        {**u_g1, "skip_pre_unit_closures": True},
        {**u_g2, "skip_pre_unit_closures": True},
        repo_root=str(tmp_path),
    )
    unmarked_comb_lenient.pop("_skip_pre_unit_closures", None)
    unmarked_comb_lenient.pop("skip_pre_unit_closures", None)
    assert (
        synthesize_shared_helper_code(
            u_g1,
            u_g2,
            repo_root=str(tmp_path),
            scope1=unmarked_lenient1,
            scope2=unmarked_lenient2,
            scope=unmarked_comb_lenient,
            closure_strictness="strict",
        )
        == ""
    )

    # _slice_unit_token_lines truncates end_col even when line 0 is pre-sliced or empty
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 24, "end_col": 5},
        ["[\n", "        x * 2 for x in items\n", "    ] + [99]\n"],
    ) == ["[\n", "        x * 2 for x in items\n", "    ]"]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 5},
        ["\n", "        x * 2 for x in items\n", "    ] + [99]\n"],
    ) == ["\n", "        x * 2 for x in items\n", "    ]"]
    utf8_last = '    ] + "é界🚀"\n'
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 30, "end_col": 5},
        ["[\n", "    x for x in items\n", utf8_last],
    ) == ["[\n", "    x for x in items\n", "    ]"]
    utf8_empty_slice = "é = "
    utf8_len = len(utf8_empty_slice.encode("utf-8"))
    assert _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": utf8_len, "end_col": utf8_len},
        [utf8_empty_slice],
    ) == [""]

    # Closure mutation propagation through classes
    cls_prop_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    z = 3\n"
        "    w = 4\n"
        "    class OuterCls:\n"
        "        x = 99\n"
        "        nonlocal z\n"
        "        z = 30\n"
        "        global g_var\n"
        "        g_var = 100\n"
        "        if True:\n"
        "            def branch_m(self):\n"
        "                nonlocal y\n"
        "                y += 1\n"
        "        class InnerCls:\n"
        "            def deep_m(self):\n"
        "                class MethodCls:\n"
        "                    def leaf(self):\n"
        "                        nonlocal x, w\n"
        "                        x += 1\n"
        "                        del w\n"
        "                return MethodCls\n"
    )
    outer_cls_node = cls_prop_tree.body[0].body[4]  # type: ignore[attr-defined]
    muts_cls, dels_cls = _extract_closure_nonlocal_mutations_and_deletions(
        outer_cls_node
    )
    assert muts_cls == {"x", "y", "z", "w"}
    assert dels_cls == {"w"}
    assert "g_var" not in muts_cls

    # Class attribute assignment without nonlocal does not report nonlocal mutation
    cls_shadow_only = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    class ShadowOnly:\n"
        "        x = 2\n"
        "        def read_attr(self):\n"
        "            return self.x\n"
    )
    assert _extract_closure_nonlocal_mutations_and_deletions(
        cls_shadow_only.body[0].body[1]  # type: ignore[attr-defined]
    ) == (set(), set())

    # Comprehension and generator-expression evaluation timing
    lazy_walrus_src = (
        "def f1(items):\n"
        "    x = 0\n"
        "    gen = ((x := x + item) for item in items)\n"
        "    x = 100\n"
        "    return list(gen)\n"
    )
    sc_lazy_walrus = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "kind": "compound_block",
            "source_text": lazy_walrus_src,
            "source_code": "gen = ((x := x + item) for item in items)\n",
        }
    )
    assert "severed_closure_cell" in sc_lazy_walrus["control_flow_hazards"]
    ns_lazy: dict[str, Any] = {}
    exec(lazy_walrus_src, ns_lazy)  # pylint: disable=exec-used
    assert ns_lazy["f1"]([1, 2, 3]) == [101, 103, 106]

    comp_target_shadow_src = (
        "def f1(items):\n"
        "    x = 0\n"
        "    for step in items:\n"
        "        x += step\n"
        "    funcs = [(lambda: x) for x in items]\n"
        "    return x, [fn() for fn in funcs]\n\n"
        "def f2(items):\n"
        "    x = 0\n"
        "    for step in items:\n"
        "        x += step\n"
        "    funcs = [(lambda: x) for x in items]\n"
        "    return x, [fn() for fn in funcs]\n"
    )
    u_cts1 = {
        "start": 2,
        "end": 4,
        "kind": "compound_block",
        "source_text": comp_target_shadow_src,
        "source_code": "x = 0\nfor step in items:\n    x += step\n",
    }
    sc_cts1 = inspect_single_unit_scope(u_cts1)
    assert "severed_closure_cell" not in sc_cts1["control_flow_hazards"]

    shadowed_sum_tree = ast.parse(
        "def outer(items, custom_sum):\n"
        "    sum = custom_sum\n"
        "    return sum(x for x in items)\n"
    )
    sum_fn_node = shadowed_sum_tree.body[0]
    sum_ctx = _get_fn_scope_context(sum_fn_node, mod_tree=shadowed_sum_tree)
    genexp_node = sum_fn_node.body[1].value.args[0]  # type: ignore[attr-defined]
    assert (
        _is_eagerly_consumed_genexp(
            genexp_node,
            sum_ctx.parent_map,
            set(sum_ctx.shadowed_names),
            sum_ctx.has_star_import,
        )
        is False
    )

    # Precision of _is_closure_factory_scope for ClassDef and AsyncFunctionDef
    factory_prec_tree = ast.parse(
        "def outer(x):\n"
        "    class PureRecord:\n"
        "        a = 1\n"
        "    class PureService:\n"
        "        def compute(self, v):\n"
        "            return v * 2\n"
        "    class CapturingCls:\n"
        "        def get(self):\n"
        "            return x\n"
        "    class MutatingCls:\n"
        "        def inc(self):\n"
        "            nonlocal x\n"
        "            x += 1\n"
        "    class FactoryCls:\n"
        "        def make(self, a):\n"
        "            return lambda: a\n"
        "    class CarrierCls:\n"
        "        def __init__(self):\n"
        "            reg.append(lambda: x)\n"
        "    async def pure_async(a, b):\n"
        "        return a + b\n"
        "    async def capturing_async():\n"
        "        return x\n"
        "    async def mutating_async():\n"
        "        nonlocal x\n"
        "        x += 1\n"
        "    async def async_gen():\n"
        "        yield 1\n"
    )
    f_prec_fn = factory_prec_tree.body[0]
    f_prec_ctx = _get_fn_scope_context(f_prec_fn, mod_tree=factory_prec_tree)
    body_nodes = {
        getattr(n, "name", ""): n for n in f_prec_fn.body  # type: ignore[attr-defined]
    }
    assert _is_closure_factory_scope(body_nodes["PureRecord"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["PureService"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["CapturingCls"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["MutatingCls"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["FactoryCls"], f_prec_ctx) is True
    assert _extract_fn_outer_carrier_targets(
        body_nodes["CarrierCls"], frozenset(), f_prec_ctx
    ) == {"reg"}
    assert _is_closure_factory_scope(body_nodes["pure_async"], f_prec_ctx) is False
    assert _is_closure_factory_scope(body_nodes["capturing_async"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["mutating_async"], f_prec_ctx) is True
    assert _is_closure_factory_scope(body_nodes["async_gen"], f_prec_ctx) is True

    # Transitive sibling-closure propagation (cyclic aliases, rebound aliases, factory chains)
    cyclic_closures: list[tuple[str | None, set[str], set[str]]] = [
        ("seed_cb", {"x"}, {"y"}),
    ]
    cyclic_edges: list[tuple[str, set[str]]] = [
        ("alias_a", {"seed_cb", "alias_b"}),
        ("alias_b", {"alias_a"}),
        ("alias_c", {"alias_b"}),
    ]
    post_reads_cyc: set[str] = {"alias_c"}
    prop_cyc = _propagate_sibling_closures(
        cyclic_closures, post_reads=post_reads_cyc, alias_edges=cyclic_edges
    )
    by_name = {name: (r, m) for name, r, m in prop_cyc if name is not None}
    assert by_name["alias_a"] == ({"x"}, {"y"})
    assert by_name["alias_b"] == ({"x"}, {"y"})
    assert by_name["alias_c"] == ({"x"}, {"y"})
    assert "x" in post_reads_cyc

    chain_factory_tree = ast.parse(
        "def outer(flag):\n"
        "    x = 1\n"
        "    def raw_cb():\n"
        "        return x\n"
        "    def f1():\n"
        "        return raw_cb\n"
        "    def f2():\n"
        "        return f1()\n"
        "    picked = f2() if flag else raw_cb\n"
        "    x = 2\n"
        "    return picked()\n"
    )
    cf_info_chain = _analyze_enclosing_fn_cells(
        chain_factory_tree.body[0],
        _SpanBounds(10, 10),
        unit_mutations=frozenset({"x"}),
        mod_tree=chain_factory_tree,
    )
    assert "x" in cf_info_chain.post_reads

    # Exception-handler implicit deletion and pattern-binding analysis
    assert _is_irrefutable_pattern(None) is False
    if sys.version_info >= (3, 10):
        match_tree = ast.parse(
            "match subj:\n"
            "    case _ if cond:\n"
            "        a = 1\n"
            "    case [only]:\n"
            "        a = 2\n"
            "    case 1 | _:\n"
            "        a = 3\n"
        )
        m_cases = match_tree.body[0].cases  # type: ignore[attr-defined]
        assert _is_irrefutable_case(m_cases[0]) is False
        assert _is_irrefutable_case(m_cases[1]) is False
        assert _is_irrefutable_case(m_cases[2]) is True

    except_del_input_src = (
        "def f(err):\n"
        "    try:\n"
        "        x = err + 1\n"
        "    except Exception as err:\n"
        "        x = 0\n"
        "    return x\n"
    )
    sc_except_in = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "kind": "compound_block",
            "source_text": except_del_input_src,
            "source_code": "try:\n    x = err + 1\nexcept Exception as err:\n    x = 0\n",
        }
    )
    assert "err" in sc_except_in["inputs"]
    assert "deleted_output" in sc_except_in["control_flow_hazards"]
    assert sc_except_in["is_control_flow_safe"] is False

    try_finally_del_tree = ast.parse(
        "def outer():\n"
        "    surviving = 1\n"
        "    deleted_in_handler = 2\n"
        "    rebound_in_finally = 3\n"
        "    try:\n"
        "        risky()\n"
        "        only_in_try = 4\n"
        "        raise RuntimeError()\n"
        "    except Exception as rebound_in_finally:\n"
        "        del deleted_in_handler\n"
        "        raise\n"
        "    finally:\n"
        "        rebound_in_finally = 30\n"
        "    target = 0\n"
    )
    tf_info = _analyze_enclosing_fn_cells(
        try_finally_del_tree.body[0],
        _SpanBounds(14, 14),
        mod_tree=try_finally_del_tree,
    )
    assert "surviving" in tf_info.pre_bound
    assert "rebound_in_finally" in tf_info.pre_bound
    assert "deleted_in_handler" not in tf_info.pre_bound
    assert "only_in_try" not in tf_info.pre_bound

    # Concurrent closure and suspension analysis (generators and async coroutines)
    async_coro_src = (
        "async def run_task(items):\n"
        "    x = 1\n"
        "    async def async_reader():\n"
        "        return x\n"
        "    coro = async_reader()\n"
        "    for item in items:\n"
        "        x = x + item\n"
        "        res = await coro\n"
        "    return res\n"
    )
    sc_async_coro = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": async_coro_src,
            "source_code": "for item in items:\n    x = x + item\n    res = await coro\n",
        }
    )
    assert "severed_closure_cell" in sc_async_coro["control_flow_hazards"]

    pure_async_src = (
        "async def run_pure(items):\n"
        "    x = 1\n"
        "    async def pure_helper(v):\n"
        "        return v + 1\n"
        "    coro = pure_helper(10)\n"
        "    for item in items:\n"
        "        x = x + item\n"
        "        res = await coro\n"
        "    return x + res\n"
    )
    sc_pure_async = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 8,
            "kind": "compound_block",
            "source_text": pure_async_src,
            "source_code": "for item in items:\n    x = x + item\n    res = await coro\n",
        }
    )
    assert "severed_closure_cell" not in sc_pure_async["control_flow_hazards"]

    multi_gen_src = (
        "def dual_generators(items):\n"
        "    shared = 0\n"
        "    def gen_peer():\n"
        "        nonlocal shared\n"
        "        shared += 10\n"
        "        yield shared\n"
        "    peer = gen_peer()\n"
        "    for item in items:\n"
        "        shared += item\n"
        "        yield shared\n"
        "    return next(peer)\n"
    )
    sc_multi_gen = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 10,
            "kind": "compound_block",
            "source_text": multi_gen_src,
            "source_code": "for item in items:\n    shared += item\n    yield shared\n",
        }
    )
    assert "severed_closure_cell" in sc_multi_gen["control_flow_hazards"]


def test_intra_unit_global_escapes_detached_shadowing_and_suspended_local_closures() -> None:
    """Covers intra-unit global escapes, detached unit enclosing local shadowing, and suspended units with local closures."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_unit_nonlocal_mutators,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures_uncached,
        _extract_scope_local_names,
        _has_escaped_intra_unit_global_declarations,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        ensure_unit_scope_closure_mode,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    clear_dataflow_cache()

    # Global declarations inside extracted unit escaping to outer statements
    global_escape_assign_src = (
        "x = 0\n"
        "def process():\n"
        "    global x\n"
        "    x = 1\n"
        "    x = 2\n"
        "    return x\n"
    )
    ge_tree = ast.parse(global_escape_assign_src)
    assert (
        _has_escaped_intra_unit_global_declarations(
            ge_tree.body[1], 3, 4
        )
        is True
    )
    sc_global_escape = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "basic_block",
            "source_text": global_escape_assign_src,
            "source_code": "global x\nx = 1\n",
        },
        tree=ge_tree,
    )
    assert "severed_closure_cell" in sc_global_escape["control_flow_hazards"]
    assert sc_global_escape["is_control_flow_safe"] is False

    # Pre-unit statement + in-unit global + post-unit write/read -> escaping global hazard
    global_escape_pre_local_src = (
        "x = 0\n"
        "def process():\n"
        "    a = 1\n"
        "    global x\n"
        "    y = x + a\n"
        "    x = 2\n"
        "    return x, y\n"
    )
    sc_global_pre = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "basic_block",
            "source_text": global_escape_pre_local_src,
            "source_code": "global x\ny = x + a\n",
        }
    )
    assert "severed_closure_cell" in sc_global_pre["control_flow_hazards"]

    # Enclosing local `x = 0` shadowed if `def middle` loses `global x` when `return x` remains outside unit
    global_escape_enclosing_cell_src = (
        "x = 100\n"
        "def outer():\n"
        "    x = 5\n"
        "    def middle():\n"
        "        global x\n"
        "        y = 1\n"
        "        return x + y\n"
        "    return middle\n"
    )
    sc_global_enc_cell = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "basic_block",
            "source_text": global_escape_enclosing_cell_src,
            "source_code": "global x\ny = 1\n",
        }
    )
    assert "severed_closure_cell" in sc_global_enc_cell["control_flow_hazards"]

    # Safe case 1: `global x` is ONLY used inside the extracted unit, never touched/read outside
    global_self_contained_src = (
        "x = 0\n"
        "def process(a):\n"
        "    global x\n"
        "    x = a + 1\n"
        "    res = a * 2\n"
        "    return res\n"
    )
    sc_global_self = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "basic_block",
            "source_text": global_self_contained_src,
            "source_code": "global x\nx = a + 1\n",
        }
    )
    assert "severed_closure_cell" not in sc_global_self["control_flow_hazards"]

    # Safe case 2: `global x` is also declared outside the unit (`outer_globals`), and unit only reads `x`
    global_redundant_read_src = (
        "x = 10\n"
        "def process(a):\n"
        "    global x\n"
        "    global x\n"
        "    y = x + a\n"
        "    x = 20\n"
        "    return y\n"
    )
    sc_global_redundant = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "basic_block",
            "source_text": global_redundant_read_src,
            "source_code": "global x\ny = x + a\n",
        }
    )
    assert "severed_closure_cell" not in sc_global_redundant["control_flow_hazards"]

    # Detached unit modules retain enclosing function locals for shadowed builtins
    detached_shadow_src = (
        "def outer(items, custom_list):\n"
        "    list = custom_list\n"
        "    value = 1\n"
        "    seq = list(value + item for item in items)\n"
        "    value = 2\n"
        "    return seq\n"
    )
    ds_tree = ast.parse(detached_shadow_src)
    ds_outer = ds_tree.body[0]
    assert isinstance(ds_outer, ast.FunctionDef)
    detached_mod = ast.Module(body=[ds_outer.body[2]], type_ignores=[])
    assert "list" in _extract_scope_local_names(
        detached_mod, module_tree=ds_tree
    )
    detached_closures = _collect_scope_closures_uncached(
        detached_mod,
        module_tree=ds_tree,
        genexp_include_header=False,
        cache_parent_map=False,
    )
    assert any("value" in reads for _, reads in detached_closures)
    sc_detached_shadow = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": detached_shadow_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=ds_tree,
    )
    assert "severed_closure_cell" in sc_detached_shadow["control_flow_hazards"]

    # Detached unit with walrus inside genexp passed to shadowed `list`
    detached_walrus_shadow_src = (
        "def outer(items, custom_list):\n"
        "    list = custom_list\n"
        "    w = 0\n"
        "    seq = list((w := item) for item in items)\n"
        "    return seq, w\n"
    )
    dw_tree = ast.parse(detached_walrus_shadow_src)
    dw_outer = dw_tree.body[0]
    assert isinstance(dw_outer, ast.FunctionDef)
    dw_muts = _collect_unit_nonlocal_mutators(
        [dw_outer.body[2]], mod_tree=dw_tree
    )
    assert any("w" in m.mutated_vars for m in dw_muts)
    sc_detached_walrus = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 4,
            "kind": "basic_block",
            "source_text": detached_walrus_shadow_src,
            "source_code": "seq = list((w := item) for item in items)\n",
        },
        tree=dw_tree,
    )
    assert "severed_closure_cell" in sc_detached_walrus["control_flow_hazards"]

    # PEP 227 class scope isolation: class-body `list = custom_list` applies directly inside ClassDef body,
    # NOT inside a method `def m(self, items)`!
    cls_isolation_src = (
        "def outer(custom_list):\n"
        "    class Holder:\n"
        "        list = custom_list\n"
        "        def compute(self, items):\n"
        "            value = 1\n"
        "            seq = list(value + item for item in items)\n"
        "            value = 2\n"
        "            return seq, value\n"
        "    return Holder\n"
    )
    ci_tree = ast.parse(cls_isolation_src)
    sc_cls_method_safe = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "basic_block",
            "source_text": cls_isolation_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=ci_tree,
    )
    assert "severed_closure_cell" not in sc_cls_method_safe["control_flow_hazards"]

    # Directly inside ClassDef body (not inside a method), `list = custom_list` DOES shadow `list`
    cls_body_shadow_src = (
        "def outer(items, custom_list):\n"
        "    value = 1\n"
        "    class Holder:\n"
        "        list = custom_list\n"
        "        seq = list(value + item for item in items)\n"
        "    value = 2\n"
        "    return Holder\n"
    )
    cbs_tree = ast.parse(cls_body_shadow_src)
    sc_cls_body_shadow = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": cls_body_shadow_src,
            "source_code": "seq = list(value + item for item in items)\n",
        },
        tree=cbs_tree,
    )
    assert "severed_closure_cell" in sc_cls_body_shadow["control_flow_hazards"]

    # Suspended units allow purely unit-local nonlocal mutators while rejecting escaping ones
    suspended_local_closure_src = (
        "def worker_a(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n\n"
        "def worker_b(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n"
    )
    slc_tree = ast.parse(suspended_local_closure_src)
    u_slc1 = {
        "start": 2,
        "end": 8,
        "kind": "compound_block",
        "source_text": suspended_local_closure_src,
        "source_code": (
            "for item in items:\n"
            "    value = item\n"
            "    def increment():\n"
            "        nonlocal value\n"
            "        value += 1\n"
            "    increment()\n"
            "    yield value\n"
        ),
    }
    sc_slc1 = inspect_single_unit_scope(u_slc1, tree=slc_tree)
    assert "severed_closure_cell" not in sc_slc1["control_flow_hazards"]
    assert sc_slc1["is_control_flow_safe"] is True

    # Suspended unit where inner closure mutates a variable read after the unit IS rejected
    suspended_escaping_var_src = (
        "def worker(items):\n"
        "    for item in items:\n"
        "        value = item\n"
        "        def increment():\n"
        "            nonlocal value\n"
        "            value += 1\n"
        "        increment()\n"
        "        yield value\n"
        "    return value\n"
    )
    sc_susp_escape = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 8,
            "kind": "compound_block",
            "source_text": suspended_escaping_var_src,
            "source_code": u_slc1["source_code"],
        }
    )
    assert "severed_closure_cell" in sc_susp_escape["control_flow_hazards"]

    # Suspended unit where inner closure mutates an outer-scope variable IS rejected
    suspended_outer_var_src = (
        "def worker(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        def add_item():\n"
        "            nonlocal total\n"
        "            total += item\n"
        "        add_item()\n"
        "        yield item\n"
    )
    sc_susp_outer = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 8,
            "kind": "compound_block",
            "source_text": suspended_outer_var_src,
            "source_code": (
                "for item in items:\n"
                "    def add_item():\n"
                "        nonlocal total\n"
                "        total += item\n"
                "    add_item()\n"
                "    yield item\n"
            ),
        }
    )
    assert "severed_closure_cell" in sc_susp_outer["control_flow_hazards"]

    # Minimal unmarked precomputed scopes are recomputed when tree/source is available
    min_src = (
        "def outer(items):\n"
        "    cb = lambda: y\n"
        "    for x in items:\n"
        "        y = x + 1\n"
        "    return cb()\n"
    )
    min_tree = ast.parse(min_src)
    min_unit = {
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": min_src,
        "source_code": "for x in items:\n    y = x + 1\n",
    }
    minimal_unmarked_scope = {"inputs": ["items"], "outputs": ["y"]}
    recomputed_min = ensure_unit_scope_closure_mode(
        min_unit, minimal_unmarked_scope, False, tree=min_tree
    )
    assert "_skip_pre_unit_closures" in recomputed_min
    assert recomputed_min["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in recomputed_min["control_flow_hazards"]

    # Exception-handler termination & unexported conditional rebinding
    term_handler_stmts = ast.parse(
        "e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    return 0\n"
    ).body
    def_th: set[str] = set()
    cond_th: set[str] = set()
    _analyze_block_assignment(term_handler_stmts, def_th, cond_th)
    assert "e" in def_th
    assert "e" not in cond_th

    nonterm_handler_stmts = ast.parse(
        "e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    x = 0\n"
    ).body
    def_nth: set[str] = set()
    cond_nth: set[str] = set()
    _analyze_block_assignment(nonterm_handler_stmts, def_nth, cond_nth)
    assert "e" not in def_nth
    assert "e" in cond_nth

    rebind_handler_stmts = ast.parse(
        "v = 1\n"
        "e = 1\n"
        "rebound_e = 1\n"
        "try:\n"
        "    x = 2\n"
        "except Exception as e:\n"
        "    del v\n"
        "    v = 99\n"
        "    e = 99\n"
        "except RuntimeError as rebound_e:\n"
        "    x = 3\n"
        "finally:\n"
        "    rebound_e = 100\n"
    ).body
    def_rh: set[str] = set()
    cond_rh: set[str] = set()
    _analyze_block_assignment(rebind_handler_stmts, def_rh, cond_rh)
    assert "v" in def_rh
    assert "v" not in cond_rh
    assert "rebound_e" in def_rh
    assert "e" not in def_rh
    assert "e" in cond_rh

    except_clobber_prebound_src = (
        "def f(a):\n"
        "    e = a + 1\n"
        "    try:\n"
        "        res = a * 2\n"
        "    except Exception as e:\n"
        "        res = 0\n"
        "    return res + e\n"
    )
    sc_except_clobber = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 6,
            "kind": "compound_block",
            "source_text": except_clobber_prebound_src,
            "source_code": (
                "try:\n"
                "    res = a * 2\n"
                "except Exception as e:\n"
                "    res = 0\n"
            ),
        }
    )
    assert "severed_closure_cell" in sc_except_clobber["control_flow_hazards"]
    assert sc_except_clobber["is_control_flow_safe"] is False

    # UTF-8 slicing edge cases (CRLF, astral emoji, CJK, accented)
    astral_line = '    msg = "🚀✨café日本語"; val = (a + b) * c\r\n'
    val_expr = "(a + b) * c"
    s_utf8 = len('    msg = "🚀✨café日本語"; val = '.encode("utf-8"))
    e_utf8 = s_utf8 + len(val_expr.encode("utf-8"))
    assert _slice_utf8_line(astral_line, s_utf8, e_utf8) == val_expr
    assert _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": s_utf8, "end_col": e_utf8},
        [astral_line],
    ) == [val_expr]


def test_suppress_with_termination_and_definite_bindings() -> None:
    """Verifies control-flow termination vs exception suppression and non-raising definite bindings in with suppress(...)."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # return inside with suppress(Exception) CANNOT be suppressed by __exit__
    ret_in_suppress_src = (
        "with suppress(Exception):\n"
        "    return 42\n"
        "x = 1\n"
    )
    ret_stmts = ast.parse(ret_in_suppress_src).body
    assert _block_terminates([ret_stmts[0]]) is True
    assert _block_terminates(ret_stmts) is True
    def_ret, cond_ret = _analyze_block_assignment(ret_stmts)
    bound_ret, term_ret = _extract_block_definite_bindings_and_term(
        ret_stmts, None
    )
    assert def_ret == set()
    assert "x" not in cond_ret
    assert bound_ret == set()
    assert term_ret is True

    # Conversely, raise inside with suppress(Exception) CAN be suppressed by __exit__
    raise_in_suppress_src = (
        "with suppress(Exception):\n"
        "    raise ValueError('boom')\n"
        "x = 1\n"
    )
    raise_stmts = ast.parse(raise_in_suppress_src).body
    assert _block_terminates([raise_stmts[0]]) is False
    assert _block_terminates(raise_stmts) is False
    def_raise, _ = _analyze_block_assignment(raise_stmts)
    bound_raise, term_raise = _extract_block_definite_bindings_and_term(
        raise_stmts, None
    )
    assert def_raise == {"x"}
    assert bound_raise == {"x"}
    assert term_raise is False

    # Return after a potentially-raising statement inside with suppress(...) does NOT guarantee termination
    risky_before_ret_src = (
        "with suppress(KeyError):\n"
        "    scale = d['x']\n"
        "    return 42\n"
        "x = 1\n"
    )
    risky_ret_stmts = ast.parse(risky_before_ret_src).body
    assert _block_terminates([risky_ret_stmts[0]]) is False
    def_rret, cond_rret = _analyze_block_assignment(risky_ret_stmts)
    bound_rret, term_rret = _extract_block_definite_bindings_and_term(
        risky_ret_stmts, None
    )
    assert def_rret == {"x"}
    assert "scale" in cond_rret
    assert bound_rret == {"x"}
    assert term_rret is False

    # If/else inside with suppress(...) where test is safe and both branches return constants terminates
    if_ret_suppress_src = (
        "with suppress(Exception):\n"
        "    flag = True\n"
        "    if flag:\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
        "x = 1\n"
    )
    if_ret_stmts = ast.parse(if_ret_suppress_src).body
    assert _block_terminates([if_ret_stmts[0]]) is True
    assert (
        _extract_block_definite_bindings_and_term(if_ret_stmts, None)
        == ({"flag"}, True)
    )

    # Non-raising prefix statements inside with suppress(...) bind definitely
    mixed_suppress_src = (
        "with suppress(KeyError):\n"
        "    value = 42\n"
        "    pair_a, pair_b = 1, 2\n"
        "    scale = d['x']\n"
        "    after_risky = 99\n"
    )
    mixed_stmts = ast.parse(mixed_suppress_src).body
    def_mix, cond_mix = _analyze_block_assignment(mixed_stmts)
    bound_mix, term_mix = _extract_block_definite_bindings_and_term(
        mixed_stmts, None
    )
    assert def_mix == {"value", "pair_a", "pair_b"}
    assert bound_mix == {"value", "pair_a", "pair_b"}
    assert {"scale", "after_risky"} <= cond_mix
    assert term_mix is False


def test_closure_escape_analysis_edge_cases() -> None:
    """Verifies name collisions, class vs lexical namespaces, and definition vs invocation execution."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_and_executed_intra_unit_defs,
    )

    # Multiple definitions sharing a name in mutually exclusive branches
    branch_defs_src = (
        "if cond:\n"
        "    def f():\n"
        "        return a\n"
        "else:\n"
        "    def f():\n"
        "        return b\n"
        "res = f()\n"
    )
    branch_stmts = ast.parse(branch_defs_src).body
    if_node = branch_stmts[0]
    assert isinstance(if_node, ast.If)
    f_then = if_node.body[0]
    f_else = if_node.orelse[0]
    _, esc_ids_br, exec_ids_br = _collect_escaped_and_executed_intra_unit_defs(
        branch_stmts, set()
    )
    assert id(f_then) in exec_ids_br and id(f_else) in exec_ids_br
    assert id(f_then) not in esc_ids_br and id(f_else) not in esc_ids_br

    # Definition followed by ordinary assignment (rebound after alias vs rebound without escape)
    rebound_alias_stmts = ast.parse(
        "def f():\n    return a\nsaved = f\nf = 123\n"
    ).body
    _, esc_ids_ra, exec_ids_ra = _collect_escaped_and_executed_intra_unit_defs(
        rebound_alias_stmts, {"saved"}
    )
    assert id(rebound_alias_stmts[0]) in esc_ids_ra
    assert id(rebound_alias_stmts[0]) in exec_ids_ra

    # Definitions inside nested classes
    nested_cls_stmts = ast.parse(
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            return x\n"
    ).body
    outer_cls = nested_cls_stmts[0]
    assert isinstance(outer_cls, ast.ClassDef)
    inner_cls = outer_cls.body[0]
    assert isinstance(inner_cls, ast.ClassDef)
    inner_method = inner_cls.body[0]
    _, esc_unesc, exec_unesc = _collect_escaped_and_executed_intra_unit_defs(
        nested_cls_stmts, set()
    )
    assert id(outer_cls) in exec_unesc
    assert id(inner_cls) in exec_unesc
    assert id(inner_method) not in exec_unesc
    _, esc_esc, exec_esc = _collect_escaped_and_executed_intra_unit_defs(
        nested_cls_stmts, {"Outer"}
    )
    assert {id(outer_cls), id(inner_cls), id(inner_method)} <= esc_esc
    assert {id(outer_cls), id(inner_cls), id(inner_method)} <= exec_esc

    # Functions declaring a nested definition global or nonlocal (executed vs unexecuted)
    nl_rebind_stmts = ast.parse(
        "def dead_binder():\n"
        "    nonlocal outer_cb\n"
        "    def outer_cb():\n"
        "        return x\n"
        "def live_binder():\n"
        "    nonlocal live_cb\n"
        "    def live_cb():\n"
        "        return y\n"
        "live_binder()\n"
    ).body
    dead_fn = nl_rebind_stmts[0]
    live_fn = nl_rebind_stmts[1]
    assert isinstance(dead_fn, ast.FunctionDef) and isinstance(
        live_fn, ast.FunctionDef
    )
    dead_cb = dead_fn.body[1]
    live_cb = live_fn.body[1]
    esc_names_nl, esc_ids_nl, exec_ids_nl = (
        _collect_escaped_and_executed_intra_unit_defs(nl_rebind_stmts, set())
    )
    assert id(dead_cb) not in esc_ids_nl
    assert id(dead_cb) not in exec_ids_nl
    assert id(live_cb) in esc_ids_nl
    assert id(live_cb) in exec_ids_nl
    assert "live_cb" in esc_names_nl

    # Class body referencing an outer function before defining a same-named method
    cls_shadow_order_stmts = ast.parse(
        "def helper():\n"
        "    return x\n"
        "class C:\n"
        "    val = helper()\n"
        "    def helper(self):\n"
        "        return 1\n"
    ).body
    outer_helper = cls_shadow_order_stmts[0]
    _, _, exec_cls_order = _collect_escaped_and_executed_intra_unit_defs(
        cls_shadow_order_stmts, set()
    )
    assert id(outer_helper) in exec_cls_order

    # Nested class inside class and method referencing outer cell vs self.attr / ClassName.attr
    pep227_src = (
        "def outer(items):\n"
        "    shared = 10\n"
        "    class Container:\n"
        "        shared = 99\n"
        "        class Inner:\n"
        "            def read_outer(self):\n"
        "                return shared + self.shared + Container.shared\n"
        "    inst = Container.Inner()\n"
        "    shared = 20\n"
        "    return inst.read_outer()\n"
    )
    sc_pep227 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 8,
            "kind": "compound_block",
            "source_text": pep227_src,
            "source_code": (
                "class Container:\n"
                "    shared = 99\n"
                "    class Inner:\n"
                "        def read_outer(self):\n"
                "            return shared + self.shared + Container.shared\n"
                "inst = Container.Inner()\n"
            ),
        },
        tree=ast.parse(pep227_src),
    )
    assert "severed_closure_cell" in sc_pep227["control_flow_hazards"]

    # Decorator on uncalled function executes at definition time
    dec_exec_stmts = ast.parse(
        "def my_dec(fn):\n"
        "    return fn\n"
        "@my_dec\n"
        "def uncalled():\n"
        "    return 1\n"
    ).body
    dec_fn = dec_exec_stmts[0]
    uncalled_fn = dec_exec_stmts[1]
    _, esc_dec, exec_dec = _collect_escaped_and_executed_intra_unit_defs(
        dec_exec_stmts, set()
    )
    assert id(dec_fn) in exec_dec
    assert id(uncalled_fn) in esc_dec


def test_generator_expression_eager_vs_lazy_table_and_nested_walrus() -> None:
    """Verifies all 10 eager vs lazy generator expression cases and nested walrus generators."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_comprehension_deferred_closure_info,
        _extract_lazy_genexp_walrus_names,
        _get_fn_scope_context,
        _is_lazy_genexp,
    )

    table_cases: list[tuple[str, bool]] = [
        ("return sum(x for x in values)", False),
        ("return list(x for x in values)", False),
        ("return tuple(x for x in values)", False),
        ("return sorted(x for x in values)", False),
        ("return any(x for x in values)", False),
        ("return all(x for x in values)", False),
        ("return custom(x for x in values)", True),
        ("list = custom_fn\n    return list(x for x in values)", True),
        ("return list((x for x in values))", False),
        ("return iter(x for x in values)", True),
    ]
    for body_line, expected_lazy in table_cases:
        fn_src = f"def f(values, custom, custom_fn):\n    {body_line}\n"
        mod = ast.parse(fn_src)
        fn_node = mod.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        ctx = _get_fn_scope_context(fn_node, mod_tree=mod)
        genexp_nodes = [
            n for n in ast.walk(fn_node) if isinstance(n, ast.GeneratorExp)
        ]
        assert len(genexp_nodes) == 1
        assert _is_lazy_genexp(genexp_nodes[0], ctx) is expected_lazy, body_line

    # Walrus case 1: gen = ((count := count + x) for x in values)
    walrus_gen1_src = (
        "def f(values):\n"
        "    count = 0\n"
        "    gen = ((count := count + x) for x in values)\n"
        "    res = list(gen)\n"
        "    return res, count\n"
    )
    mod1 = ast.parse(walrus_gen1_src)
    fn1 = mod1.body[0]
    assert isinstance(fn1, ast.FunctionDef)
    ctx1 = _get_fn_scope_context(fn1, mod_tree=mod1)
    assign1 = fn1.body[1]
    assert isinstance(assign1, ast.Assign) and isinstance(
        assign1.value, ast.GeneratorExp
    )
    assert _extract_lazy_genexp_walrus_names(assign1, ctx1) == {"count"}
    _, def_muts1 = _extract_comprehension_deferred_closure_info(
        assign1.value, ctx1
    )
    assert "count" in def_muts1
    sc_w1 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "basic_block",
            "source_text": walrus_gen1_src,
            "source_code": (
                "count = 0\n"
                "gen = ((count := count + x) for x in values)\n"
            ),
        },
        tree=mod1,
    )
    assert "severed_closure_cell" in sc_w1["control_flow_hazards"]

    # Walrus case 2: gen = (sum((count := count + x) for x in values) for _ in range(1))
    walrus_gen2_src = (
        "def f(values):\n"
        "    count = 0\n"
        "    gen = (sum((count := count + x) for x in values) for _ in range(1))\n"
        "    res = list(gen)\n"
        "    return res, count\n"
    )
    mod2 = ast.parse(walrus_gen2_src)
    fn2 = mod2.body[0]
    assert isinstance(fn2, ast.FunctionDef)
    ctx2 = _get_fn_scope_context(fn2, mod_tree=mod2)
    assign2 = fn2.body[1]
    assert isinstance(assign2, ast.Assign)
    outer_gen = assign2.value
    assert isinstance(outer_gen, ast.GeneratorExp)
    assert isinstance(outer_gen.elt, ast.Call)
    inner_gen = outer_gen.elt.args[0]
    assert isinstance(inner_gen, ast.GeneratorExp)
    assert _is_lazy_genexp(outer_gen, ctx2) is True
    assert _is_lazy_genexp(inner_gen, ctx2) is False
    assert _extract_lazy_genexp_walrus_names(assign2, ctx2) == {"count"}
    _, def_muts2 = _extract_comprehension_deferred_closure_info(
        outer_gen, ctx2
    )
    assert "count" in def_muts2
    sc_w2 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "basic_block",
            "source_text": walrus_gen2_src,
            "source_code": (
                "count = 0\n"
                "gen = (sum((count := count + x) for x in values) for _ in range(1))\n"
            ),
        },
        tree=mod2,
    )
    assert "severed_closure_cell" in sc_w2["control_flow_hazards"]


def test_span_line_index_performance_and_scale() -> None:
    """Verifies _SpanLineIndex scales cleanly across 10,000-statement modules, deep nesting, and multi-unit lines within a strict time budget."""
    import time
    from pydoppelgangerhunt.fixer import _df_span_index
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _has_unresolved_span_bounds,
        _resolve_unit_col_bounds,
    )

    clear_dataflow_cache()
    t0 = time.perf_counter()

    # 1. Module with 10,000 statements
    large_mod_lines = [f"v_{i} = {i}\n" for i in range(10000)]
    large_mod_tree = ast.parse("".join(large_mod_lines))
    cand_5000 = ast.parse("v_5000 = 5000\n").body
    cand_9999 = ast.parse("v_9999 = 9999\n").body

    orig_build_index = _df_span_index._SpanLineIndex
    build_count = 0

    class _CountingSpanLineIndex(_df_span_index._SpanLineIndex):
        def __init__(self, root: ast.AST) -> None:
            nonlocal build_count
            build_count += 1
            super().__init__(root)

    setattr(_df_span_index, "_SpanLineIndex", _CountingSpanLineIndex)
    try:
        assert _resolve_unit_col_bounds(
            cand_5000, large_mod_tree, 5001, 5001, None, None
        ) == (0, len("v_5000 = 5000"))
        assert _resolve_unit_col_bounds(
            cand_9999, large_mod_tree, 10000, 10000, None, None
        ) == (0, len("v_9999 = 9999"))
        assert (
            _has_unresolved_span_bounds(
                large_mod_tree, 5001, 5001, 0, len("v_5000 = 5000"), cand_5000
            )
            is False
        )
        # Index for large_mod_tree must be built once and reused across queries
        assert build_count == 1
    finally:
        setattr(_df_span_index, "_SpanLineIndex", orig_build_index)

    # 2. Deeply nested functions (15 levels deep)
    deep_lines = ["def f_0(x):\n"]
    for depth in range(1, 15):
        indent = "    " * depth
        deep_lines.append(f"{indent}def f_{depth}(x):\n")
    leaf_indent = "    " * 15
    deep_lines.append(f"{leaf_indent}target = x + 1\n")
    deep_lines.append(f"{leaf_indent}return target\n")
    for depth in range(14, 0, -1):
        indent = "    " * depth
        deep_lines.append(f"{indent}return f_{depth}(x)\n")
    deep_tree = ast.parse("".join(deep_lines))
    cand_leaf = ast.parse("target = x + 1\n").body
    expected_s_col = len(leaf_indent)
    expected_e_col = len(leaf_indent) + len("target = x + 1")
    assert _resolve_unit_col_bounds(
        cand_leaf, deep_tree, 16, 16, None, None
    ) == (expected_s_col, expected_e_col)

    # 3. Many candidate units on the same source line (20 semicolon-separated statements)
    semi_parts = [f"s_{i} = {i}" for i in range(20)]
    semi_line = "    " + "; ".join(semi_parts) + "\n"
    semi_tree = ast.parse(f"def multi():\n{semi_line}")
    for i in (0, 7, 19):
        part_cand = ast.parse(f"s_{i} = {i}\n").body
        pos = semi_line.index(f"s_{i} = {i}")
        assert _resolve_unit_col_bounds(
            part_cand, semi_tree, 2, 2, None, None
        ) == (pos, pos + len(f"s_{i} = {i}"))
    assert (
        _has_unresolved_span_bounds(semi_tree.body[0], 2, 2, None, None)
        is True
    )

    # 4. Hundreds of candidate units within one large function + repeated inspection of same AST tree
    big_fn_lines = ["def big_fn(x):\n"] + [
        f"    step_{i} = x + {i}\n" for i in range(300)
    ]
    big_fn_tree = ast.parse("".join(big_fn_lines))
    for i in range(0, 300, 3):
        c_stmt = [big_fn_tree.body[0].body[i]]  # type: ignore[attr-defined]
        ln = i + 2
        s_c, e_c = _resolve_unit_col_bounds(
            c_stmt, big_fn_tree, ln, ln, None, None
        )
        assert (s_c, e_c) == (4, 4 + len(f"step_{i} = x + {i}"))
        assert (
            _has_unresolved_span_bounds(
                big_fn_tree.body[0], ln, ln, s_c, e_c, c_stmt
            )
            is False
        )

    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Span index scale test took too long: {elapsed:.2f}s"

    assert any(
        k.owner is big_fn_tree and k.kind == "span_line_index"
        for k in _scope_cell_info_cache
    )
    _evict_tree_scopes(big_fn_tree)
    assert not any(
        k.owner is big_fn_tree
        for k in _scope_cell_info_cache
    )


def test_ensure_unit_scope_closure_mode_all_six_cases() -> None:
    """Verifies all 6 scope-mode and missing-file / source_text fallback cases."""
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    src_with_pre_closure = (
        "def proc(items):\n"
        "    cb = lambda: acc\n"
        "    for x in items:\n"
        "        acc = x + 1\n"
        "    return 1\n"
    )
    base_unit = {
        "file": "missing_virtual_file_12345.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": src_with_pre_closure,
        "source_code": "for x in items:\n    acc = x + 1\n",
    }
    strict_scope = inspect_single_unit_scope(
        {**base_unit, "skip_pre_unit_closures": False}
    )
    lenient_scope = inspect_single_unit_scope(
        {**base_unit, "skip_pre_unit_closures": True}
    )
    assert strict_scope["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in strict_scope["control_flow_hazards"]
    assert lenient_scope["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in lenient_scope["control_flow_hazards"]

    # 1. Strict scope reused in lenient mode -> recomputes to lenient
    res_1 = ensure_unit_scope_closure_mode(base_unit, strict_scope, True)
    assert res_1["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in res_1["control_flow_hazards"]

    # 2. Lenient scope reused in strict mode -> recomputes to strict
    res_2 = ensure_unit_scope_closure_mode(base_unit, lenient_scope, False)
    assert res_2["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in res_2["control_flow_hazards"]

    # 3. Scope missing _skip_pre_unit_closures -> recomputes and attaches flag
    unmarked = {
        k: v
        for k, v in lenient_scope.items()
        if k not in ("_skip_pre_unit_closures", "skip_pre_unit_closures")
    }
    res_3 = ensure_unit_scope_closure_mode(base_unit, unmarked, False)
    assert res_3["_skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in res_3["control_flow_hazards"]

    # 4. Pathological AST scope reused in either mode without recomputing
    pathological_scope = {
        **strict_scope,
        "control_flow_hazards": ["pathological_ast"],
        "is_control_flow_safe": False,
    }
    assert (
        ensure_unit_scope_closure_mode(base_unit, pathological_scope, True)
        is pathological_scope
    )
    assert (
        ensure_unit_scope_closure_mode(base_unit, pathological_scope, False)
        is pathological_scope
    )

    # 5. Missing source file with precomputed scope and no source_text -> fails closed
    no_src_unit = {
        "file": "nonexistent_on_disk_98765.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
    }
    res_5 = ensure_unit_scope_closure_mode(no_src_unit, lenient_scope, False)
    assert res_5["is_control_flow_safe"] is False
    assert "severed_closure_cell" in res_5["control_flow_hazards"]
    assert res_5["_skip_pre_unit_closures"] is False

    # 6. Valid source_text supplied alongside a missing file (even if unit has source_text: None or source_lines: [])
    unit_with_none_src = {
        "file": "nonexistent_on_disk_98765.py",
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "source_text": None,
        "source_lines": [],
    }
    res_6 = ensure_unit_scope_closure_mode(
        unit_with_none_src,
        lenient_scope,
        True,
        source_text=src_with_pre_closure,
    )
    assert res_6["_skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in res_6["control_flow_hazards"]
    helper_code = synthesize_shared_helper_code(
        unit_with_none_src,
        dict(unit_with_none_src),
        source_text1=src_with_pre_closure,
        source_text2=src_with_pre_closure,
        scope1=lenient_scope,
        scope2=dict(lenient_scope),
        skip_pre_unit_closures=True,
    )
    assert "def " in helper_code and "acc = x + 1" in helper_code


def test_utf8_source_slicing_multibyte_and_idempotence() -> None:
    """Verifies UTF-8 byte-offset slicing across multibyte identifiers/literals and pre-sliced idempotence."""
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    # 1. café = "hello"; result = café + "world"
    line1 = 'café = "hello"; result = café + "world"\n'
    expr1 = 'café + "world"'
    s1 = len('café = "hello"; result = '.encode("utf-8"))
    e1 = s1 + len(expr1.encode("utf-8"))
    assert _slice_utf8_line(line1, s1, e1) == expr1
    unit1 = {"kind": "complex_expr", "start_col": s1, "end_col": e1}
    sliced1 = _slice_unit_token_lines(unit1, [line1])
    assert sliced1 == [expr1]
    assert _slice_unit_token_lines(unit1, sliced1) == [expr1]

    # 2. result = "é"; another = "世界"; final = 42
    line2 = 'result = "é"; another = "世界"; final = 42\n'
    for target in ('"é"', '"世界"', "42"):
        char_idx = line2.index(target)
        s_b = len(line2[:char_idx].encode("utf-8"))
        e_b = len(line2[: char_idx + len(target)].encode("utf-8"))
        assert _slice_utf8_line(line2, s_b, e_b) == target
        u_t = {"kind": "complex_expr", "start_col": s_b, "end_col": e_b}
        out_t = _slice_unit_token_lines(u_t, [line2])
        assert out_t == [target]
        assert _slice_unit_token_lines(u_t, out_t) == [target]

    # 3. Pre-sliced single-line expression where s_col < len(sliced) and source_code is omitted
    line3 = "x = [i + 1 for i in items]\n"
    expr3 = "[i + 1 for i in items]"
    u3 = {"kind": "comprehension", "start_col": 4, "end_col": 4 + len(expr3)}
    sliced3 = _slice_unit_token_lines(u3, [line3])
    assert sliced3 == [expr3]
    assert _slice_unit_token_lines(u3, sliced3) == [expr3]

    # 4. Multiline UTF-8 expression slicing and pre-sliced idempotence
    ml_utf8 = [
        '    résultat = (\n',
        '        café + "世界"\n',
        '    )\n',
    ]
    s_ml = len("    résultat = ".encode("utf-8"))
    e_ml = len("    )".encode("utf-8"))
    u_ml = {"kind": "complex_expr", "start_col": s_ml, "end_col": e_ml}
    sliced_ml = _slice_unit_token_lines(u_ml, ml_utf8)
    assert sliced_ml == ['(\n', '        café + "世界"\n', '    )']
    assert _slice_unit_token_lines(u_ml, sliced_ml) == sliced_ml


def test_sibling_closure_ordinary_store_followed_by_call_or_mutation() -> None:
    """Verifies ordinary stores in extracted subroutines are checked against sibling closures."""
    # 1. Ordinary store followed by sibling closure invocation reading the rebound local
    store_then_call_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def callback():\n"
        "        return x\n"
        "    x = 1\n"
        "    result = callback()\n"
        "    return result\n"
    )
    tree1 = ast.parse(store_then_call_src)
    for skip_pre in (False, True):
        sc1 = inspect_single_unit_scope(
            {
                "start": 5,
                "end": 7,
                "kind": "basic_block",
                "is_subroutine": True,
                "skip_pre_unit_closures": skip_pre,
                "source_text": store_then_call_src,
                "source_code": "x = 1\nresult = callback()\nreturn result\n",
            },
            tree=tree1,
        )
        assert "severed_closure_cell" in sc1["control_flow_hazards"]
        assert sc1["is_control_flow_safe"] is False

    # 2. Ordinary store followed by sibling closure mutation (both returning and output-unpacking)
    store_then_mutate_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def mutate():\n"
        "        nonlocal x\n"
        "        x = 2\n"
        "    x = 1\n"
        "    mutate()\n"
        "    y = 3\n"
        "    return x, y\n"
    )
    tree2 = ast.parse(store_then_mutate_src)
    for skip_pre in (False, True):
        sc2 = inspect_single_unit_scope(
            {
                "start": 6,
                "end": 8,
                "kind": "basic_block",
                "is_subroutine": True,
                "skip_pre_unit_closures": skip_pre,
                "source_text": store_then_mutate_src,
                "source_code": "x = 1\nmutate()\ny = 3\n",
            },
            tree=tree2,
        )
        assert "severed_closure_cell" in sc2["control_flow_hazards"]
        assert sc2["is_control_flow_safe"] is False

    # 3. Contrast: closure unit with its own local `x = 1` (not nonlocal) does not mutate outer `x`
    closure_own_local_src = (
        "def outer():\n"
        "    x = 0\n"
        "    def callback():\n"
        "        return x\n"
        "    def inner():\n"
        "        x = 1\n"
        "        return callback() + x\n"
        "    return inner()\n"
    )
    sc_clos = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "closure",
            "is_subroutine": False,
            "source_text": closure_own_local_src,
            "source_code": "def inner():\n    x = 1\n    return callback() + x\n",
        },
        tree=ast.parse(closure_own_local_src),
    )
    assert "severed_closure_cell" not in sc_clos["control_flow_hazards"]


def test_class_namespace_nested_deletions_and_ordered_redefinitions() -> None:
    """Verifies class-body definition analysis ignores nested deletions and respects statement order."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_and_executed_intra_unit_defs,
        _extract_class_body_unconditional_def_names,
    )

    # 1. Nested method deleting another class method's name does not affect class-level availability
    nested_del_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    def callback():\n"
        "        return 42\n"
        "    def another():\n"
        "        del callback\n"
        "    result = callback()\n"
    ).body
    outer_cb_1 = nested_del_stmts[0]
    cls_1 = nested_del_stmts[1]
    assert isinstance(cls_1, ast.ClassDef)
    uncond_1 = _extract_class_body_unconditional_def_names(cls_1)
    assert "callback" in uncond_1
    _, esc_ids_1, exec_ids_1 = _collect_escaped_and_executed_intra_unit_defs(
        nested_del_stmts, frozenset()
    )
    assert id(outer_cb_1) not in esc_ids_1
    assert id(outer_cb_1) not in exec_ids_1

    # 2. Class-body deletion followed by a later definition restores class-level availability
    del_then_def_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    del callback\n"
        "    def callback():\n"
        "        return 42\n"
        "    result = callback()\n"
    ).body
    outer_cb_2 = del_then_def_stmts[0]
    cls_2 = del_then_def_stmts[1]
    assert isinstance(cls_2, ast.ClassDef)
    uncond_2 = _extract_class_body_unconditional_def_names(cls_2)
    assert "callback" in uncond_2
    _, esc_ids_2, exec_ids_2 = _collect_escaped_and_executed_intra_unit_defs(
        del_then_def_stmts, frozenset()
    )
    assert id(outer_cb_2) not in esc_ids_2
    assert id(outer_cb_2) not in exec_ids_2

    # 3. Class-body definition followed by deletion before use falls back to outer scope
    def_then_del_stmts = ast.parse(
        "def callback():\n"
        "    nonlocal state\n"
        "    state += 1\n"
        "class Example:\n"
        "    def callback():\n"
        "        return 42\n"
        "    del callback\n"
        "    result = callback()\n"
    ).body
    outer_cb_3 = def_then_del_stmts[0]
    cls_3 = def_then_del_stmts[1]
    assert isinstance(cls_3, ast.ClassDef)
    uncond_3 = _extract_class_body_unconditional_def_names(cls_3)
    assert "callback" not in uncond_3
    _, _, exec_ids_3 = _collect_escaped_and_executed_intra_unit_defs(
        def_then_del_stmts, frozenset()
    )
    assert id(outer_cb_3) in exec_ids_3


def test_span_line_index_missing_end_positions_and_duplicate_boundary_statements() -> None:
    """Covers missing end_lineno before unit, same-line boundary deduplication, and ambiguous identical boundary statements."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_span_line_index,
        _has_unresolved_span_bounds,
        _resolve_unit_col_bounds,
    )

    clear_dataflow_cache()

    # 1. Synthetic AST node with missing end_lineno well before extracted unit
    tree_missing = ast.parse(
        "def outer():\n"
        "    early = 1\n"
        "    mid = 2\n"
        "    unit_var = 3\n"
        "    return unit_var\n"
    )
    outer_fn = tree_missing.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    early_stmt = outer_fn.body[0]
    early_stmt.end_lineno = None
    assert (
        _has_unresolved_span_bounds(
            outer_fn, 4, 4, 4, 16, [outer_fn.body[2]]
        )
        is True
    )

    # 2. iter_boundary_entries deduplicates by node identity when u_start == u_end
    clear_dataflow_cache()
    same_line_tree = ast.parse("def f():\n    a = 1; b = a + 2\n")
    idx = _get_span_line_index(same_line_tree)
    entries = idx.iter_boundary_entries(2, 2)
    node_ids = [id(e.node) for e in entries]
    assert len(node_ids) == len(set(node_ids))

    # 3. Multiple identical boundary statements on the same line reject ambiguous column inference
    ambig_src = "def f():\n    x = 1; x = 1\n    return x\n"
    ambig_tree = ast.parse(ambig_src)
    ambig_fn = ambig_tree.body[0]
    assert isinstance(ambig_fn, ast.FunctionDef)
    cand_stmt = [ast.parse("x = 1\n").body[0]]
    s_col, e_col = _resolve_unit_col_bounds(
        cand_stmt, ambig_tree, 2, 2, None, None
    )
    assert (s_col, e_col) == (None, None)
    assert (
        _has_unresolved_span_bounds(
            ambig_fn, 2, 2, s_col, e_col, cand_stmt
        )
        is True
    )
