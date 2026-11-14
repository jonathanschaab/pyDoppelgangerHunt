"""Definite-binding, with-provenance, control-flow termination, and span traversal unit tests."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from tests.fixer._closure_test_helpers import (
    _run_differential_refactoring_case,
)


@pytest.mark.parametrize(
    "case_id,stmts_src,initial_bound,expected_bound,expected_term",
    [
        (
            "raise_in_try_preserves_initial_bound",
            ("try:\n" "    value = risky_operation()\n" "except Exception:\n" "    pass\n"),
            frozenset({"value"}),
            {"value"},
            False,
        ),
        (
            "raise_in_try_without_initial_bound_not_definite",
            ("try:\n" "    value = risky_operation()\n" "except Exception:\n" "    pass\n"),
            frozenset(),
            set(),
            False,
        ),
        (
            "try_and_all_handlers_bind_same_var",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except ValueError:\n"
                "    value = 10\n"
                "except Exception:\n"
                "    value = 20\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_else_and_all_handlers_bind_var",
            (
                "try:\n"
                "    step = risky_operation()\n"
                "except Exception:\n"
                "    value = -1\n"
                "else:\n"
                "    value = 1\n"
            ),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_handler_raises_so_try_else_binding_survives_fallthrough",
            ("try:\n" "    value = risky_operation()\n" "except Exception:\n" "    raise\n"),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "try_finally_overrides_and_binds_unconditionally",
            (
                "try:\n"
                "    value = risky_operation()\n"
                "except Exception:\n"
                "    pass\n"
                "finally:\n"
                "    final_val = 42\n"
            ),
            frozenset(),
            {"final_val"},
            False,
        ),
        (
            "if_return_in_one_branch_binds_via_fallthrough_branch",
            ("if cond:\n" "    return 0\n" "else:\n" "    value = 7\n"),
            frozenset(),
            {"value"},
            False,
        ),
        (
            "if_raise_in_both_branches_terminates_block",
            ("if cond:\n" "    raise ValueError('a')\n" "else:\n" "    return 1\n"),
            frozenset(),
            set(),
            True,
        ),
        (
            "del_in_one_non_terminating_branch_removes_initial_binding",
            ("if cond:\n" "    del value\n"),
            frozenset({"value"}),
            set(),
            False,
        ),
        (
            "del_followed_by_rebind_in_same_branch_preserves_binding",
            ("if cond:\n" "    del value\n" "    value = 2\n"),
            frozenset({"value"}),
            {"value"},
            False,
        ),
        (
            "except_handler_target_implicitly_deleted_on_exit",
            ("try:\n" "    risky_operation()\n" "except Exception as value:\n" "    pass\n"),
            frozenset({"value"}),
            set(),
            False,
        ),
    ],
)
def test_definite_bindings_exceptional_control_flow_table(
    case_id: str,
    stmts_src: str,
    initial_bound: frozenset[str],
    expected_bound: set[str],
    expected_term: bool,
) -> None:
    """Table-driven verification of _extract_block_definite_bindings_and_term across exceptional paths."""
    import textwrap  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )

    _ = case_id
    mod = ast.parse(f"def _wrap():\n{textwrap.indent(stmts_src, '    ')}\n")
    fn_node = mod.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    bound, term = _extract_block_definite_bindings_and_term(
        fn_node.body,
        active_nonlocals=None,
        initial_bound=initial_bound,
        fn_globals=frozenset(),
    )
    assert bound == expected_bound
    assert term is expected_term

    # Also verify pre-unit try/except exhaustive bindings in _analyze_enclosing_fn_cells
    if not initial_bound and not expected_term:
        outer_src = (
            "def outer(cond, risky_operation):\n"
            f"{textwrap.indent(stmts_src, '    ')}"
            "    return lambda: 1\n"
        )
        outer_mod = ast.parse(outer_src)
        outer_fn = outer_mod.body[0]
        last_line = len(outer_src.splitlines())
        info = _analyze_enclosing_fn_cells(outer_fn, (last_line, last_line, None, None))
        for v_name in expected_bound:
            assert v_name in info.pre_bound


def test_2d_coordinate_enclosure_and_decorator_start_pos() -> None:
    """Verifies 2D coordinate enclosure and decorator (line, col) coupling."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _SpanBounds,
        _strictly_encloses_span,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
        _resolve_immediate_scope_in_fn,
    )
    from pydoppelgangerhunt.fixer.scope import _find_enclosing_pre_bound_names

    # 1. Disjoint node to the left on the same line does not enclose span
    if_left = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=2,
        end_lineno=2,
        col_offset=0,
        end_col_offset=15,
    )
    assert not _strictly_encloses_span(if_left, _SpanBounds(2, 2, 20, 35))
    assert not _strictly_encloses_span(if_left, _SpanBounds(2, 2, 20, None))

    # 2. Disjoint node to the right on the same line does not enclose span
    if_right = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=2,
        end_lineno=2,
        col_offset=20,
        end_col_offset=35,
    )
    assert not _strictly_encloses_span(if_right, _SpanBounds(2, 2, 0, 15))
    assert not _strictly_encloses_span(if_right, _SpanBounds(2, 2, None, 15))

    # 3. Multi-line node ending on line 2 at col 10 does not enclose unit at [15, 25] on line 2
    multiline_prior = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        simple=1,
        lineno=1,
        end_lineno=2,
        col_offset=0,
        end_col_offset=10,
    )
    assert not _strictly_encloses_span(multiline_prior, _SpanBounds(2, 2, 15, 25))
    assert not _strictly_encloses_span(multiline_prior, _SpanBounds(2, 2, 15, None))

    # 4. True single-line compound enclosure: `if cond: target_unit()`
    single_if_mod = ast.parse("if cond: target_unit()\n")
    single_if = single_if_mod.body[0]
    assert _strictly_encloses_span(single_if, _SpanBounds(1, 1, 9, 22))
    assert not _strictly_encloses_span(single_if, _SpanBounds(1, 1, 0, 22))

    # 5. _resolve_immediate_scope_in_fn ignores disjoint class ending before or starting after unit on same line
    from pydoppelgangerhunt.fixer._df_cache import evict_tree_scopes

    cls_tree = ast.parse("def outer():\n" "    class Box(\n" "        Base\n" "    ): pass\n")
    outer_fn = cls_tree.body[0]
    box_cls = outer_fn.body[0]  # type: ignore[attr-defined]
    box_cls.end_col_offset = 11
    assert _resolve_immediate_scope_in_fn(outer_fn, 4, 4, u_start_col=13, u_end_col=18) is outer_fn
    box_cls.lineno = 4
    box_cls.col_offset = 20
    box_cls.end_col_offset = 35
    evict_tree_scopes(cls_tree)
    assert _resolve_immediate_scope_in_fn(outer_fn, 4, 4, u_start_col=4, u_end_col=15) is outer_fn

    # 6. _find_enclosing_pre_bound_names sets in_loop=False when loop is disjoint on boundary line
    loop_tree = ast.parse(
        "def outer(items):\n" "    for a in (\n" "        items\n" "    ): pass\n"
    )
    for_stmt = loop_tree.body[0].body[0]  # type: ignore[attr-defined]
    for_stmt.end_col_offset = 11
    _, in_loop = _find_enclosing_pre_bound_names(loop_tree, 4, 4, u_start_col=13, u_end_col=18)
    assert in_loop is False
    for_stmt.lineno = 4
    for_stmt.col_offset = 20
    for_stmt.end_col_offset = 35
    evict_tree_scopes(loop_tree)
    _, in_loop_right = _find_enclosing_pre_bound_names(loop_tree, 4, 4, u_start_col=4, u_end_col=15)
    assert in_loop_right is False

    # 7. Decorator line/col tracking uses decorator's '@' column and handles loose forms
    dec_fn = ast.FunctionDef(
        name="inner",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Pass(lineno=4, end_lineno=4, col_offset=8, end_col_offset=12)],
        decorator_list=[
            ast.Name(
                id="dec",
                ctx=ast.Load(),
                lineno=2,
                end_lineno=2,
                col_offset=3,
                end_col_offset=6,
            )
        ],
        lineno=3,
        end_lineno=4,
        col_offset=4,
        end_col_offset=12,
    )
    assert _node_earliest_start_pos(dec_fn) == (2, 4)
    wrap_mod = ast.Module(body=[dec_fn], type_ignores=[])
    closures = _collect_scope_closures(wrap_mod)
    assert closures[0][0] == (2, 4)


def test_duplicate_cols_decorator_snapping_and_cell_info_immutability() -> None:
    """Covers duplicate _unit_cols disambiguation, loose decorators, nonlocal shadowing, and immutable cache."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_fn_outer_carrier_targets,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _node_earliest_start_pos,
    )
    from pydoppelgangerhunt.fixer._df_pairing import (
        resolve_clone_generator_subroutine_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        ensure_unit_scope_closure_mode,
        inspect_single_unit_scope,
        scope_matches_closure_mode,
    )

    # 1. _unit_cols returns (None, None) when duplicate snippets on a line are ambiguous
    dup_line = "    a = f(); a = f()\n"
    assert _unit_cols({"source_code": "a = f()"}, [dup_line]) == (None, None)
    assert _unit_cols({"source_code": "a = f()", "start_col": 4}, [dup_line]) == (4, 11)
    assert _unit_cols({"source_code": "a = f()", "start_col": 13}, [dup_line]) == (13, 20)
    assert _unit_cols({"source_code": "a = f()", "end_col": 20}, [dup_line]) == (13, 20)

    # 2. Loose decorator forms (@dec, @ dec, @(dec)) at top-level and indented
    for dec_text in ("@dec", "@ dec", "@(dec)"):
        top_mod = ast.parse(f"{dec_text}\ndef fn():\n    return x\n")
        assert _node_earliest_start_pos(top_mod.body[0]) == (1, 0)
        assert _collect_scope_closures(top_mod)[0][0] == (1, 0)

        ind_mod = ast.parse(f"def outer(x):\n    {dec_text}\n    def fn():\n        return x\n")
        inner_fn = ind_mod.body[0].body[0]  # type: ignore[attr-defined]
        assert _node_earliest_start_pos(inner_fn) == (2, 4)

    # 3. _extract_fn_outer_carrier_targets does not report shadowed nonlocal in lambda or comprehension
    shadow_mod = ast.parse(
        "def outer():\n"
        "    x = []\n"
        "    y = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        f = lambda x: x.append(lambda: y)\n"
        "        g = [x.append(lambda: y) for x in ([],)]\n"
        "        return f, g\n"
    )
    mid_fn = shadow_mod.body[0].body[2]  # type: ignore[attr-defined]
    assert _extract_fn_outer_carrier_targets(mid_fn) == set()

    unshadowed_mod = ast.parse(
        "def outer():\n"
        "    x = []\n"
        "    y = 1\n"
        "    def mid():\n"
        "        nonlocal x\n"
        "        f = lambda z: x.append(lambda: y)\n"
        "        return f\n"
    )
    mid_unshadowed = unshadowed_mod.body[0].body[2]  # type: ignore[attr-defined]
    assert _extract_fn_outer_carrier_targets(mid_unshadowed) == {"x"}

    # 4. _analyze_enclosing_fn_cells returns immutable containers and reuses cached statement deltas
    multi_unit_mod = ast.parse(
        "def big_fn(a):\n"
        "    b = a + 1\n"
        "    def helper():\n"
        "        return b\n"
        "    u1 = b + 2\n"
        "    u2 = u1 + 3\n"
        "    return helper(), u2\n"
    )
    big_fn = multi_unit_mod.body[0]
    info1 = _analyze_enclosing_fn_cells(big_fn, (5, 5, 4, 14), mod_tree=multi_unit_mod)
    info2 = _analyze_enclosing_fn_cells(big_fn, (6, 6, 4, 15), mod_tree=multi_unit_mod)
    assert isinstance(info1.fn_locals, frozenset)
    assert isinstance(info1.pre_bound, frozenset)
    assert isinstance(info1.sibling_closures, tuple)
    assert isinstance(info1.alias_edges, tuple)
    assert "b" in info1.pre_bound
    assert "u1" in info2.pre_bound

    # 5. Centralized closure mode helpers and generator subroutine strictness normalization
    gen_src = (
        "def gen(items):\n"
        "    cb = lambda: acc\n"
        "    acc = 0\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        yield acc\n"
        "    return acc, cb\n"
    )
    gen_tree = ast.parse(gen_src)
    u_gen = {
        "start": 3,
        "end": 6,
        "kind": "compound_block",
        "source_text": gen_src,
        "source_code": "acc = 0\nfor x in items:\n    acc += x\n    yield acc\n",
    }
    strict_sc = inspect_single_unit_scope(u_gen, tree=gen_tree)
    assert scope_matches_closure_mode(strict_sc, False) is True
    assert scope_matches_closure_mode(strict_sc, True) is False
    lenient_sc = ensure_unit_scope_closure_mode(
        u_gen, strict_sc, True, source_text=gen_src, tree=gen_tree
    )
    assert scope_matches_closure_mode(lenient_sc, True) is True
    assert resolve_clone_generator_subroutine_outputs(
        u_gen,
        dict(u_gen),
        strict_sc,
        dict(strict_sc),
        source_text1=gen_src,
        source_text2=gen_src,
        tree1=gen_tree,
        tree2=gen_tree,
        closure_strictness="lenient",
    ) == (["acc"], ["acc"])


def test_closure_factories_return_refs_and_class_body_units() -> None:
    """Verifies closure factories, return refs, class-body units, shadowing, col inference, and post-unit lenient expression hazards."""
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        inspect_single_unit_scope,
    )

    # 1. Local non-escaping lambda inside helper does not taint plain return value as closure factory
    src_item1 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def compute_initial(seq):\n"
        "        s = sorted(seq, key=lambda v: v)\n"
        "        return sum(s) + acc\n"
        "    init_val = compute_initial(items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(init_val + x)\n"
        "    return acc, out\n"
    )
    tree_item1 = ast.parse(src_item1)
    unit_item1 = {
        "start": 8,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_item1,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(init_val + x)\n"),
    }
    scope_item1 = inspect_single_unit_scope(unit_item1, tree=tree_item1)
    assert "severed_closure_cell" not in scope_item1["control_flow_hazards"]
    assert scope_item1["is_control_flow_safe"] is True

    # 2. Helper calling a sibling closure in return (ret_calls) is not conflated with returning callable (ret_loads)
    src_item2 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    def compute_initial():\n"
        "        return get_acc() + 1\n"
        "    init_val = compute_initial()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(init_val + x)\n"
        "    return acc, out\n"
    )
    tree_item2 = ast.parse(src_item2)
    unit_item2 = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_item2,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(init_val + x)\n"),
    }
    scope_item2 = inspect_single_unit_scope(unit_item2, tree=tree_item2)
    assert "severed_closure_cell" not in scope_item2["control_flow_hazards"]
    assert scope_item2["is_control_flow_safe"] is True

    # Verify helper actually returning get_acc (ret_loads) or transitive call to factory STILL triggers hazard
    src_item2_pos = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    def get_reader():\n"
        "        return get_acc\n"
        "    def get_reader_2():\n"
        "        return get_reader()\n"
        "    reader = get_reader_2()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(reader() + x)\n"
        "    return acc, out\n"
    )
    tree_item2_pos = ast.parse(src_item2_pos)
    unit_item2_pos = {
        "start": 11,
        "end": 13,
        "kind": "compound_block",
        "source_text": src_item2_pos,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(reader() + x)\n"),
    }
    scope_item2_pos = inspect_single_unit_scope(unit_item2_pos, tree=tree_item2_pos)
    assert "severed_closure_cell" in scope_item2_pos["control_flow_hazards"]

    # 3A. Class-body unit capturing enclosing class name Box before class body finishes
    src_item3a = (
        "def outer(items):\n"
        "    cbs = []\n"
        "    class Box:\n"
        "        for x in items:\n"
        "            cbs.append(lambda x=x: (Box, x))\n"
        "    return Box, cbs\n"
    )
    tree_item3a = ast.parse(src_item3a)
    unit_item3a = {
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_text": src_item3a,
        "source_code": ("for x in items:\n" "    cbs.append(lambda x=x: (Box, x))\n"),
    }
    scope_item3a = inspect_single_unit_scope(unit_item3a, tree=tree_item3a)
    assert "severed_closure_cell" in scope_item3a["control_flow_hazards"]

    # 3B. Class-body unit conditionally assigning v without nonlocal does not promote outer local v
    src_item3b = (
        "def outer(flag):\n"
        "    v = 10\n"
        "    class Box:\n"
        "        if flag:\n"
        "            v = 20\n"
        "        w = 30\n"
        "    return Box, v\n"
    )
    tree_item3b = ast.parse(src_item3b)
    unit_item3b = {
        "start": 4,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_item3b,
        "source_code": "if flag:\n    v = 20\nw = 30\n",
    }
    scope_item3b = inspect_single_unit_scope(unit_item3b, tree=tree_item3b)
    assert "v" not in scope_item3b["inputs"]
    assert "v" in scope_item3b["conditional_outputs"]

    # 4. Lexical shadowing of outer local in inner function does not leak into pre_bound / cond_live_ins
    src_item4 = (
        "def outer(flag):\n"
        "    v = 10\n"
        "    def inner():\n"
        "        if flag:\n"
        "            v = 20\n"
        "        res = v if flag else 0\n"
        "        return res, v\n"
        "    return inner()\n"
    )
    tree_item4 = ast.parse(src_item4)
    unit_item4 = {
        "start": 4,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_item4,
        "source_code": "if flag:\n    v = 20\nres = v if flag else 0\n",
    }
    scope_item4 = inspect_single_unit_scope(unit_item4, tree=tree_item4)
    assert "v" not in scope_item4["inputs"]

    # 5. _unit_cols / _infer_missing_col_from_line_ast disambiguates comprehension vs parent Assign/BinOp
    comp_assign_line = ["    res = [x * 2 for x in items]"]
    assert _unit_cols({"kind": "comprehension", "end_col": 32}, comp_assign_line) == (10, 32)
    comp_binop_line = ["    [x * 2 for x in items] + [1]"]
    assert _unit_cols({"kind": "comprehension", "start_col": 4}, comp_binop_line) == (4, 26)

    # 6. Post-unit sibling closure capturing walrus-mutated var in expression unit caught in lenient mode
    src_item6 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    vals = [(acc := acc + x) for x in items]\n"
        "    cb = lambda: acc\n"
        "    return vals, cb\n"
    )
    tree_item6 = ast.parse(src_item6)
    unit_item6 = {
        "start": 3,
        "end": 3,
        "start_col": 11,
        "end_col": 44,
        "kind": "comprehension",
        "skip_pre_unit_closures": True,
        "source_text": src_item6,
        "source_code": "[(acc := acc + x) for x in items]",
    }
    scope_item6 = inspect_single_unit_scope(unit_item6, tree=tree_item6)
    assert "severed_closure_cell" in scope_item6["control_flow_hazards"]


def test_unseeded_conditional_stores_and_nested_mutator_shadowing() -> None:
    """Verifies unseeded cond store in unit closure, nested fn shadowing in mutators, outer carrier value filtering, nested comp target shadowing, lambda default args, and class method nonlocal del."""
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    # 1. Unseeded conditional store captured by an intra-unit closure with pre-unit conditional binding
    src_r1 = (
        "def outer(items, init_flag):\n"
        "    if init_flag:\n"
        "        y = 10\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        if x > 0:\n"
        "            y = x * 2\n"
        "        cbs.append(lambda: y)\n"
        "    return cbs\n"
    )
    tree_r1 = ast.parse(src_r1)
    unit_r1 = {
        "start": 5,
        "end": 8,
        "kind": "compound_block",
        "source_text": src_r1,
        "source_code": (
            "for x in items:\n"
            "    if x > 0:\n"
            "        y = x * 2\n"
            "    cbs.append(lambda: y)\n"
        ),
    }
    scope_r1 = inspect_single_unit_scope(unit_r1, tree=tree_r1)
    assert "severed_closure_cell" in scope_r1["control_flow_hazards"]

    # 2. Intra-unit nested function locals shadowed in _collect_unit_nonlocal_mutators
    src_r2 = (
        "def outer(items):\n"
        "    count = 100\n"
        "    if items:\n"
        "        def middle():\n"
        "            count = 0\n"
        "            def inner():\n"
        "                nonlocal count\n"
        "                count += 1\n"
        "            inner()\n"
        "            gen = ((count := count + x) for x in items)\n"
        "            return count + sum(gen)\n"
        "        res = middle()\n"
        "    else:\n"
        "        res = 0\n"
        "    return count + res\n"
    )
    tree_r2 = ast.parse(src_r2)
    unit_r2 = {
        "start": 3,
        "end": 14,
        "kind": "compound_block",
        "source_text": src_r2,
        "source_code": (
            "if items:\n"
            "    def middle():\n"
            "        count = 0\n"
            "        def inner():\n"
            "            nonlocal count\n"
            "            count += 1\n"
            "        inner()\n"
            "        gen = ((count := count + x) for x in items)\n"
            "        return count + sum(gen)\n"
            "    res = middle()\n"
            "else:\n"
            "    res = 0\n"
        ),
    }
    scope_r2 = inspect_single_unit_scope(unit_r2, tree=tree_r2)
    assert "severed_closure_cell" not in scope_r2["control_flow_hazards"]
    assert scope_r2["is_control_flow_safe"] is True

    # 3. _extract_fn_outer_carrier_targets_uncached ignores non-closure outer attribute calls
    src_r3 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    saved = []\n"
        "    def helper(seq):\n"
        "        s = sorted(seq, key=lambda v: v)\n"
        "        saved.append(len(s) + acc)\n"
        "    helper(items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(len(saved) + x)\n"
        "    return acc, out\n"
    )
    tree_r3 = ast.parse(src_r3)
    unit_r3 = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_r3,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(len(saved) + x)\n"),
    }
    scope_r3 = inspect_single_unit_scope(unit_r3, tree=tree_r3)
    assert "severed_closure_cell" not in scope_r3["control_flow_hazards"]
    assert scope_r3["is_control_flow_safe"] is True

    # 4. Comprehension target shadowing accumulated across nested comprehensions
    src_r4 = (
        "def outer(items):\n"
        "    acc = [1, 2]\n"
        "    cbs = [[lambda: acc for acc in acc] for _ in range(2)]\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc = [x]\n"
        "        out.append(len(cbs) + x)\n"
        "    return acc, out\n"
    )
    tree_r4 = ast.parse(src_r4)
    unit_r4 = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": src_r4,
        "source_code": ("for x in items:\n" "    acc = [x]\n" "    out.append(len(cbs) + x)\n"),
    }
    scope_r4 = inspect_single_unit_scope(unit_r4, tree=tree_r4)
    assert "severed_closure_cell" not in scope_r4["control_flow_hazards"]
    assert scope_r4["is_control_flow_safe"] is True

    # 5. Lambda default arguments inspected in _extract_expr_closure_and_load_info
    src_r5 = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    def get_acc():\n"
        "        return acc\n"
        "    reader = lambda cb=get_acc: cb()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(reader() + x)\n"
        "    return acc, out\n"
    )
    tree_r5 = ast.parse(src_r5)
    unit_r5 = {
        "start": 7,
        "end": 9,
        "kind": "compound_block",
        "source_text": src_r5,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(reader() + x)\n"),
    }
    scope_r5 = inspect_single_unit_scope(unit_r5, tree=tree_r5)
    assert "severed_closure_cell" in scope_r5["control_flow_hazards"]

    # 6. Sibling method nonlocal deletion inside an enclosing ClassDef marks variable pre_unbound
    src_r6 = (
        "def outer(items):\n"
        "    total = 10\n"
        "    cbs = []\n"
        "    class Box:\n"
        "        def clear(self):\n"
        "            nonlocal total\n"
        "            del total\n"
        "        def build(self):\n"
        "            for x in items:\n"
        "                cbs.append(lambda x=x: total + x)\n"
        "    return Box, cbs\n"
    )
    tree_r6 = ast.parse(src_r6)
    unit_r6 = {
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_r6,
        "source_code": ("for x in items:\n" "    cbs.append(lambda x=x: total + x)\n"),
    }
    scope_r6 = inspect_single_unit_scope(unit_r6, tree=tree_r6)
    assert "severed_closure_cell" in scope_r6["control_flow_hazards"]


def test_exported_conditional_stores_scope_headers_and_exhaustive_match() -> None:
    """Verifies unseeded cond store exported and captured, header exclusion in _find_enclosing_pre_bound_names, exhaustive match definite bindings, and conditional class methods."""
    import sys
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        inspect_single_unit_scope,
    )

    # 1. Unseeded conditional store with no pre-unit binding captured by intra-unit closure and exported
    src_f1 = (
        "def outer(x):\n"
        "    cbs = []\n"
        "    if x > 0:\n"
        "        y = x * 2\n"
        "    cbs.append(lambda: y)\n"
        "    return cbs, (y if x > 0 else 0)\n"
    )
    tree_f1 = ast.parse(src_f1)
    unit_f1 = {
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "source_text": src_f1,
        "source_code": "if x > 0:\n    y = x * 2\ncbs.append(lambda: y)\n",
    }
    scope_f1 = inspect_single_unit_scope(unit_f1, tree=tree_f1)
    assert "severed_closure_cell" in scope_f1["control_flow_hazards"]

    # 2. _find_enclosing_pre_bound_names excludes inner scope when unit span is in inner scope's header
    src_f2 = (
        "def outer():\n"
        "    x = 1\n"
        "    def inner(\n"
        "        a=x,\n"
        "    ):\n"
        "        x = 2\n"
        "        return a + x\n"
        "    return inner\n"
    )
    tree_f2 = ast.parse(src_f2)
    pre_bound_hdr, _ = _find_enclosing_pre_bound_names(tree_f2, 4, 4, u_start_col=8, u_end_col=11)
    assert "x" in pre_bound_hdr
    assert "inner" not in pre_bound_hdr
    assert "a" not in pre_bound_hdr

    # 3A. Exhaustive match/case before unit marks variable definitely pre_bound
    if sys.version_info >= (3, 10):
        src_f3a = (
            "def outer(cmd, items):\n"
            "    match cmd:\n"
            "        case 'add':\n"
            "            scale = 2\n"
            "        case _:\n"
            "            scale = 1\n"
            "    cbs = []\n"
            "    for x in items:\n"
            "        cbs.append(lambda x=x: x * scale)\n"
            "    return cbs\n"
        )
        tree_f3a = ast.parse(src_f3a)
        unit_f3a = {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f3a,
            "source_code": ("for x in items:\n" "    cbs.append(lambda x=x: x * scale)\n"),
        }
        scope_f3a = inspect_single_unit_scope(unit_f3a, tree=tree_f3a)
        assert "severed_closure_cell" not in scope_f3a["control_flow_hazards"]
        assert scope_f3a["is_control_flow_safe"] is True

    # 3B. Conditionally defined class method aliases self to sibling method closures
    src_f3b = (
        "def outer(items, fast):\n"
        "    acc = 0\n"
        "    class Worker:\n"
        "        if fast:\n"
        "            def get_acc(self):\n"
        "                return acc\n"
        "            def run(self):\n"
        "                out = []\n"
        "                for x in items:\n"
        "                    acc_local = self.get_acc() + x\n"
        "                    out.append(acc_local)\n"
        "                return out\n"
        "    acc += 1\n"
        "    return Worker, acc\n"
    )
    tree_f3b = ast.parse(src_f3b)
    unit_f3b = {
        "start": 9,
        "end": 11,
        "kind": "compound_block",
        "source_text": src_f3b,
        "source_code": (
            "for x in items:\n" "    acc_local = self.get_acc() + x\n" "    out.append(acc_local)\n"
        ),
    }
    scope_f3b = inspect_single_unit_scope(unit_f3b, tree=tree_f3b)
    assert "self" in scope_f3b["inputs"]

    src_f3b_mut = (
        "def outer(items, fast):\n"
        "    acc = 0\n"
        "    class Worker:\n"
        "        if fast:\n"
        "            def get_acc(self):\n"
        "                return acc\n"
        "            def run(self):\n"
        "                nonlocal acc\n"
        "                out = []\n"
        "                for x in items:\n"
        "                    acc += x\n"
        "                    out.append(self.get_acc())\n"
        "                return out\n"
        "    return Worker\n"
    )
    tree_f3b_mut = ast.parse(src_f3b_mut)
    unit_f3b_mut = {
        "start": 10,
        "end": 12,
        "kind": "compound_block",
        "source_text": src_f3b_mut,
        "source_code": ("for x in items:\n" "    acc += x\n" "    out.append(self.get_acc())\n"),
    }
    scope_f3b_mut = inspect_single_unit_scope(unit_f3b_mut, tree=tree_f3b_mut)
    assert "severed_closure_cell" in scope_f3b_mut["control_flow_hazards"]


def test_utf8_semicolon_and_multiline_column_slicing() -> None:
    """Covers UTF-8 prefix, multiline, and semicolon-separated column slicing and body extraction."""
    from pydoppelgangerhunt.fixer.scope import (
        _unit_cols,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (
        _extract_unit_body_lines,
        _slice_unit_token_lines,
    )

    utf8_semi_src = (
        "def compute(items, a, b):\n"
        '    tag = "café_αβγ_🚀"; target = [x * 2 for x in items]; res = a + b; tail = 0\n'
        "    return target, res\n"
    )
    utf8_semi_line = utf8_semi_src.splitlines(keepends=True)[1]
    comp_snip = "[x * 2 for x in items]"
    s_b, e_b = _unit_cols({"kind": "comprehension", "source_code": comp_snip}, [utf8_semi_line])
    assert s_b is not None and e_b is not None
    assert _unit_cols({"kind": "comprehension", "start_col": s_b}, [utf8_semi_line]) == (s_b, e_b)
    assert _unit_cols({"kind": "comprehension", "end_col": e_b}, [utf8_semi_line]) == (s_b, e_b)
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": s_b, "end_col": e_b},
        [utf8_semi_line],
    ) == [comp_snip]
    assert _slice_unit_token_lines(
        {
            "kind": "comprehension",
            "start_col": s_b,
            "end_col": e_b,
            "source_code": comp_snip,
        },
        [comp_snip],
    ) == []
    assert _slice_unit_token_lines(
        {
            "kind": "comprehension",
            "start_col": s_b,
            "end_col": e_b,
            "source_code": comp_snip,
            "source_lines_is_col_sliced": True,
        },
        [comp_snip],
    ) == [comp_snip]

    stmt_snip = "res = a + b"
    stmt_s, stmt_e = _unit_cols({"kind": "basic_block", "source_code": stmt_snip}, [utf8_semi_line])
    assert stmt_s is not None and stmt_e is not None
    semi_stmt_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "compound_block",
            "source_text": utf8_semi_src,
            "source_code": stmt_snip,
        }
    )
    assert semi_stmt_scope["inputs"] == ["a", "b"]
    assert semi_stmt_scope["outputs"] == ["res"]

    utf8_multi_src = (
        "def build(items, scale):\n"
        '    header = "résumé_🚀", [\n'
        "        x * scale\n"
        "        for x in items\n"
        "    ] + [0]\n"
        "    return header\n"
    )
    multi_lines = utf8_multi_src.splitlines(keepends=True)[1:5]
    multi_snip = "[\n        x * scale\n        for x in items\n    ]"
    ms_col, me_col = _unit_cols({"kind": "comprehension", "source_code": multi_snip}, multi_lines)
    assert ms_col == len('    header = "résumé_🚀", '.encode("utf-8"))
    assert me_col == len("    ]".encode("utf-8"))
    assert _unit_cols({"kind": "comprehension", "start_col": ms_col}, multi_lines) == (
        ms_col,
        me_col,
    )
    assert _unit_cols({"kind": "comprehension", "end_col": me_col}, multi_lines) == (ms_col, me_col)
    sliced_multi = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": ms_col, "end_col": me_col},
        multi_lines,
    )
    assert "".join(sliced_multi) == multi_snip
    multi_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "kind": "comprehension",
            "source_text": utf8_multi_src,
            "source_code": multi_snip,
        }
    )
    assert set(multi_scope["inputs"]) == {"items", "scale"}

    utf8_multi_semi_src = (
        "def calc_pair(a, b):\n"
        '    tag = "café_🚀"; total = (\n'
        "        a + b\n"
        "    ); trailing = 99\n"
        "    return total\n"
    )
    multi_semi_snip = "total = (\n        a + b\n    )"
    multi_semi_scope = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": utf8_multi_semi_src,
            "source_code": multi_semi_snip,
        }
    )
    assert multi_semi_scope["inputs"] == ["a", "b"]
    assert multi_semi_scope["outputs"] == ["total"]

    one_line_fn = 'def greet(prefix="café_🚀"): return prefix + "!"'
    assert _extract_unit_body_lines({"kind": "function"}, [one_line_fn]) == ['return prefix + "!"']


def test_try_finally_conditional_exits_and_definite_bindings(
    tmp_path: Path,
) -> None:
    """Covers try/finally conditional return/raise paths and differential execution."""
    import warnings

    from pydoppelgangerhunt.fixer.patch import refactor_module_units
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    src_f4_ret = (
        "def outer(flag, items):\n"
        "    try:\n"
        "        value = 1\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_ret = {
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "source_text": src_f4_ret,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        sc_f4_ret = inspect_single_unit_scope(u_f4_ret)
    assert "severed_closure_cell" not in sc_f4_ret["control_flow_hazards"]
    assert sc_f4_ret["is_control_flow_safe"] is True

    src_f4_nested = (
        "def outer(flag, mode, items):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
        "    finally:\n"
        "        if flag:\n"
        "            if mode == 1:\n"
        "                value = 10\n"
        "            else:\n"
        "                return []\n"
        "        else:\n"
        "            if mode == 2:\n"
        "                raise ValueError('abort')\n"
        "            value = 20\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_nested = {
        "start": 17,
        "end": 18,
        "kind": "compound_block",
        "source_text": src_f4_nested,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        sc_f4_nested = inspect_single_unit_scope(u_f4_nested)
    assert "severed_closure_cell" not in sc_f4_nested["control_flow_hazards"]

    src_f4_unsafe = (
        "def outer(flag, mode, items):\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
        "    finally:\n"
        "        if flag:\n"
        "            if mode == 1:\n"
        "                value = 10\n"
        "        else:\n"
        "            value = 20\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    u_f4_unsafe = {
        "start": 13,
        "end": 14,
        "kind": "compound_block",
        "source_text": src_f4_unsafe,
        "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
    }
    sc_f4_unsafe = inspect_single_unit_scope(u_f4_unsafe)
    assert "severed_closure_cell" in sc_f4_unsafe["control_flow_hazards"]

    mod_f4_src = (
        "def f1(flag, items):\n"
        "    try:\n"
        "        value = 1\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n\n"
        "def f2(flag, items):\n"
        "    try:\n"
        "        value = 5\n"
        "    finally:\n"
        "        if flag:\n"
        "            return []\n"
        "        value = 2\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    f4_file = tmp_path / "mod_f4.py"
    f4_file.write_text(mod_f4_src, encoding="utf-8")
    u_f4_1 = {
        "file": str(f4_file),
        "start": 9,
        "end": 10,
        "kind": "compound_block",
        "name": "f1_loop",
    }
    u_f4_2 = {
        "file": str(f4_file),
        "start": 21,
        "end": 22,
        "kind": "compound_block",
        "name": "f2_loop",
    }
    refactored_f4 = (
        "def _shared_f4_loop(cbs, items, value):\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n\n"
        + refactor_module_units(
            mod_f4_src,
            [
                (u_f4_1, "    _shared_f4_loop(cbs, items, value)"),
                (u_f4_2, "    _shared_f4_loop(cbs, items, value)"),
            ],
        )
    )
    ns_orig_f4: dict = {}
    ns_ref_f4: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(mod_f4_src, ns_orig_f4)  # pylint: disable=exec-used
        exec(refactored_f4, ns_ref_f4)  # pylint: disable=exec-used
    assert ns_orig_f4["f1"](True, [1, 2]) == ns_ref_f4["f1"](True, [1, 2]) == []
    assert (
        [cb() for cb in ns_orig_f4["f1"](False, [3, 4])]
        == [cb() for cb in ns_ref_f4["f1"](False, [3, 4])]
        == [5, 6]
    )


def test_try_except_handler_bindings_and_implicit_deletion() -> None:
    """Covers conditional del in try, pre-assignment exceptions, except-as implicit deletion, and finally rebinding."""
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    src_f5_del_in_try = (
        "def outer(cond, op, recover):\n"
        "    value = 10\n"
        "    try:\n"
        "        if cond():\n"
        "            del value\n"
        "        op()\n"
        "    except Exception:\n"
        "        callback = lambda: value\n"
        "        recover()\n"
        "        return callback\n"
        "    return lambda: value\n"
    )
    u_f5_in_handler = {
        "start": 8,
        "end": 9,
        "kind": "compound_block",
        "source_text": src_f5_del_in_try,
        "source_code": "callback = lambda: value\nrecover()\n",
    }
    sc_f5_in_handler = inspect_single_unit_scope(u_f5_in_handler)
    assert "severed_closure_cell" in sc_f5_in_handler["control_flow_hazards"]
    assert sc_f5_in_handler["is_control_flow_safe"] is False

    ns_f5: dict = {}
    exec(src_f5_del_in_try, ns_f5)  # pylint: disable=exec-used
    recovered_calls: list = []
    cb_res = ns_f5["outer"](
        lambda: True,
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda: recovered_calls.append("ok"),
    )
    assert recovered_calls == ["ok"]
    with pytest.raises(NameError):
        cb_res()

    src_f5_pre_assign = (
        "def outer(op, recover, items):\n"
        "    try:\n"
        "        op()\n"
        "        value = 10\n"
        "    except Exception:\n"
        "        recover()\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    sc_f5_pre_assign = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f5_pre_assign,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert "severed_closure_cell" in sc_f5_pre_assign["control_flow_hazards"]

    src_f5_except_as = (
        "def outer(op, items):\n"
        "    value = 10\n"
        "    try:\n"
        "        op()\n"
        "    except Exception as value:\n"
        "        pass\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: (value, x))\n"
        "    return cbs\n"
    )
    sc_f5_except_as = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 9,
            "kind": "compound_block",
            "source_text": src_f5_except_as,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: (value, x))\n",
        }
    )
    assert "severed_closure_cell" in sc_f5_except_as["control_flow_hazards"]

    src_f5_finally_rebind = (
        "def outer(op, items):\n"
        "    value = 10\n"
        "    try:\n"
        "        del value\n"
        "        op()\n"
        "    except Exception as value:\n"
        "        pass\n"
        "    finally:\n"
        "        value = 42\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda x=x: x + value)\n"
        "    return cbs\n"
    )
    sc_f5_finally_rebind = inspect_single_unit_scope(
        {
            "start": 11,
            "end": 12,
            "kind": "compound_block",
            "source_text": src_f5_finally_rebind,
            "source_code": "for x in items:\n    cbs.append(lambda x=x: x + value)\n",
        }
    )
    assert "severed_closure_cell" not in sc_f5_finally_rebind["control_flow_hazards"]


def test_enclosing_function_detection_nested_nonlocals_and_sibling_aliases() -> None:
    """Covers decorator/header enclosing-function detection, 3-level nonlocal del scoping, and sibling closure alias cycles."""
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    src_area1 = (
        "def outer(dec1, dec2, items):\n"
        "    scale = 3\n"
        "    @dec1(lambda v: v + scale)\n"
        "    @dec2\n"
        "    def inner(\n"
        "        a=[x * scale for x in items],\n"
        "    ):\n"
        "        scale = 99\n"
        "        return [x + 1 for x in a]; [x + 2 for x in a]\n"
        "    return inner\n"
    )
    sc_area1_hdr = inspect_single_unit_scope(
        {
            "start": 6,
            "end": 6,
            "kind": "comprehension",
            "source_text": src_area1,
            "source_code": "[x * scale for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_area1_hdr["control_flow_hazards"]

    src_area2 = (
        "def level0(items):\n"
        "    acc = 1\n"
        "    def level1():\n"
        "        acc = 2\n"
        "        def level2():\n"
        "            nonlocal acc\n"
        "            del acc\n"
        "        return level2\n"
        "    cbs = [lambda x=x: x + acc for x in items]\n"
        "    return level1, cbs\n"
    )
    sc_area2 = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "comprehension",
            "source_text": src_area2,
            "source_code": "[lambda x=x: x + acc for x in items]",
        }
    )
    assert "severed_closure_cell" not in sc_area2["control_flow_hazards"]

    src_area3 = (
        "def outer(flag, items):\n"
        "    acc = 0\n"
        "    if flag:\n"
        "        def helper(n):\n"
        "            return helper(n - 1) if n > 0 else 0\n"
        "    else:\n"
        "        def helper(n):\n"
        "            return acc + n\n"
        "    a = helper\n"
        "    b = a\n"
        "    a = b\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(b(x))\n"
        "    return acc, out\n"
    )
    sc_area3 = inspect_single_unit_scope(
        {
            "start": 13,
            "end": 15,
            "kind": "compound_block",
            "source_text": src_area3,
            "source_code": "for x in items:\n    acc += x\n    out.append(b(x))\n",
        }
    )
    assert "severed_closure_cell" in sc_area3["control_flow_hazards"]


def test_walrus_definite_bindings_shadowed_sum_genexp_and_clone_output_rejection() -> None:
    """Covers unconditional walrus in if condition, shadowed vs unshadowed sum genexp, and clone-output hazard rejection."""
    from pydoppelgangerhunt.fixer.dataflow import (
        clear_dataflow_cache,
        resolve_clone_generator_subroutine_outputs,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    src_area4 = (
        "def outer(x, items):\n"
        "    if (scale := x + 1) > 0:\n"
        "        offset = 1\n"
        "    else:\n"
        "        offset = 2\n"
        "    cbs = []\n"
        "    for item in items:\n"
        "        cbs.append(lambda item=item: item * scale + offset)\n"
        "    return cbs\n"
    )
    sc_area4 = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_area4,
            "source_code": "for item in items:\n    cbs.append(lambda item=item: item * scale + offset)\n",
        }
    )
    assert "severed_closure_cell" not in sc_area4["control_flow_hazards"]

    src_area5 = (
        "def outer(sum, items):\n"
        "    acc = 0\n"
        "    deferred = sum(acc + x for x in items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(deferred(x))\n"
        "    return deferred, acc, out\n"
    )
    sc_area5 = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5,
            "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
        }
    )
    assert "severed_closure_cell" in sc_area5["control_flow_hazards"]

    src_area5_eager = (
        "def outer(items):\n"
        "    acc = 0\n"
        "    total = sum(acc + x for x in items)\n"
        "    out = []\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        out.append(total + x)\n"
        "    return total, acc, out\n"
    )
    sc_area5_eager = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5_eager,
            "source_code": "for x in items:\n    acc += x\n    out.append(total + x)\n",
        }
    )
    assert "severed_closure_cell" not in sc_area5_eager["control_flow_hazards"]

    clear_dataflow_cache()
    sc_area5_after_clear = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_area5,
            "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
        }
    )
    assert sc_area5_after_clear["control_flow_hazards"] == sc_area5["control_flow_hazards"]

    u_unsafe_pair = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": src_area5,
        "source_code": "for x in items:\n    acc += x\n    out.append(deferred(x))\n",
    }
    assert (
        resolve_clone_pair_outputs(
            u_unsafe_pair,
            dict(u_unsafe_pair),
            sc_area5,
            dict(sc_area5),
            source_text1=src_area5,
            source_text2=src_area5,
        )
        is None
    )

    src_area7_gen = (
        "def gen(items):\n"
        "    cb = lambda: acc\n"
        "    acc = 0\n"
        "    for x in items:\n"
        "        acc += x\n"
        "        yield acc\n"
        "    return acc, cb\n"
    )
    u_area7_gen = {
        "start": 3,
        "end": 6,
        "kind": "compound_block",
        "source_text": src_area7_gen,
        "source_code": "acc = 0\nfor x in items:\n    acc += x\n    yield acc\n",
    }
    sc_area7_gen = inspect_single_unit_scope(u_area7_gen)
    assert (
        resolve_clone_generator_subroutine_outputs(
            u_area7_gen,
            dict(u_area7_gen),
            sc_area7_gen,
            dict(sc_area7_gen),
            source_text1=src_area7_gen,
            source_text2=src_area7_gen,
            closure_strictness="strict",
        )
        is None
    )


def test_same_stmt_evaluation_order_comp_walrus_and_decorator_alignment() -> None:
    """Verifies same-stmt RHS->LHS evaluation order, multiline ';' indent, comp walrus deferred reads, non-escaping nonlocal mutator, and decorator column snapping."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_comprehension_deferred_closure_info,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        inspect_single_unit_scope,
    )

    # 1. Same-statement Assign LHS target is post-mutated (not pre-bound) when unit is on RHS
    src_a1 = (
        "def outer(items):\n"
        "    cbs = [lambda y, x=x: (cbs, x, y) for x in items]\n"
        "    return cbs\n"
    )
    tree_a1 = ast.parse(src_a1)
    bound_a1, in_loop_a1 = _find_enclosing_pre_bound_names(
        tree_a1, 2, 2, u_start_col=10, u_end_col=53
    )
    assert "cbs" not in bound_a1
    assert not in_loop_a1
    info_a1 = _analyze_enclosing_fn_cells(tree_a1.body[0], (2, 2, 10, 53), mod_tree=tree_a1)
    assert "cbs" not in info_a1.pre_bound
    assert "cbs" in info_a1.post_mutated
    sc_a1 = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 10,
            "end_col": 53,
            "kind": "comprehension",
            "source_text": src_a1,
            "source_code": "[lambda y, x=x: (cbs, x, y) for x in items]",
        },
        tree=tree_a1,
    )
    assert "severed_closure_cell" in sc_a1["control_flow_hazards"]

    # 2. Same-statement Assign with prior binding of LHS variable still records LHS as post_mutated
    src_a2 = (
        "def outer(items):\n"
        "    cbs = None\n"
        "    cbs = [lambda y, x=x: (cbs, x, y) for x in items]\n"
        "    return cbs\n"
    )
    tree_a2 = ast.parse(src_a2)
    sc_a2 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 53,
            "kind": "comprehension",
            "source_text": src_a2,
            "source_code": "[lambda y, x=x: (cbs, x, y) for x in items]",
        },
        tree=tree_a2,
    )
    assert "severed_closure_cell" in sc_a2["control_flow_hazards"]

    # 3. For-loop iter expression unit treats loop target as post_mutated + loop_mutated (not pre_bound)
    src_a3 = (
        "def outer(items):\n"
        "    out = []\n"
        "    for x in [lambda y=i: (x, y) for i in items]:\n"
        "        out.append(x)\n"
        "    return out\n"
    )
    tree_a3 = ast.parse(src_a3)
    bound_a3, in_loop_a3 = _find_enclosing_pre_bound_names(
        tree_a3, 3, 3, u_start_col=13, u_end_col=48
    )
    assert "x" not in bound_a3
    assert not in_loop_a3
    info_a3 = _analyze_enclosing_fn_cells(tree_a3.body[0], (3, 3, 13, 48), mod_tree=tree_a3)
    assert "x" not in info_a3.pre_bound
    assert "x" in info_a3.post_mutated
    assert "x" in info_a3.loop_mutated
    sc_a3 = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 13,
            "end_col": 48,
            "kind": "comprehension",
            "source_text": src_a3,
            "source_code": "[lambda y=i: (x, y) for i in items]",
        },
        tree=tree_a3,
    )
    assert "severed_closure_cell" in sc_a3["control_flow_hazards"]

    # 4. Multiline semicolon-prefixed multi-statement unit preserves line-0 base indentation
    src_b = (
        "def compute(x):\n"
        "    prefix = 1; a = (\n"
        "        x + 1\n"
        "    )\n"
        "    b = a * 2; suffix = 3\n"
        "    return b + prefix + suffix\n"
    )
    sc_b = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 5,
            "start_col": 16,
            "end_col": 13,
            "kind": "compound_block",
            "source_text": src_b,
            "source_code": "a = (\n    x + 1\n)\nb = a * 2\n",
        },
        tree=ast.parse(src_b),
    )
    assert sc_b["inputs"] == ["x"]
    assert sc_b["outputs"] == ["a", "b"]
    assert sc_b["is_control_flow_safe"] is True

    # 5. Comprehension walrus target captured by escaping closure inside comprehension is kept in deferred_reads
    comp_c = ast.parse("[(w := v, lambda: w) for v in items]", mode="eval").body
    c_reads, _ = _extract_comprehension_deferred_closure_info(comp_c)  # type: ignore[arg-type]
    assert "w" in c_reads

    comp_c_gen = ast.parse("[(w := v, (w for _ in range(1))) for v in items]", mode="eval").body
    c_gen_reads, _ = _extract_comprehension_deferred_closure_info(comp_c_gen)  # type: ignore[arg-type]
    assert "w" in c_gen_reads

    src_c = (
        "def outer(items):\n"
        "    pairs = [(w := v, lambda: w) for v in items]\n"
        "    w = 99\n"
        "    res = pairs[0][1]() + w\n"
        "    return res\n"
    )
    sc_c = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_c,
            "source_code": "w = 99\nres = pairs[0][1]() + w\n",
        },
        tree=ast.parse(src_c),
    )
    assert "severed_closure_cell" in sc_c["control_flow_hazards"]

    sc_c_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 12,
            "end_col": 48,
            "kind": "comprehension",
            "source_text": src_c,
            "source_code": "[(w := v, lambda: w) for v in items]",
        },
        tree=ast.parse(src_c),
    )
    assert "severed_closure_cell" in sc_c_comp["control_flow_hazards"]

    # 6. Non-escaping intra-unit nonlocal mutator on top-level unit local is safe unless outer closure shares cell
    src_d_safe = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    x = x * 2\n"
        "    def bump():\n"
        "        nonlocal x\n"
        "        x += 10\n"
        "    bump()\n"
        "    return x\n"
    )
    sc_d_safe = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 7,
            "kind": "compound_block",
            "source_text": src_d_safe,
            "source_code": (
                "x = x * 2\n" "def bump():\n" "    nonlocal x\n" "    x += 10\n" "bump()\n"
            ),
        },
        tree=ast.parse(src_d_safe),
    )
    assert "severed_closure_cell" not in sc_d_safe["control_flow_hazards"]

    src_d_unsafe_sib = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    obs = lambda: x\n"
        "    x = x * 2\n"
        "    def bump():\n"
        "        nonlocal x\n"
        "        x += 10\n"
        "    bump()\n"
        "    return x, obs\n"
    )
    sc_d_unsafe_sib = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_d_unsafe_sib,
            "source_code": (
                "x = x * 2\n" "def bump():\n" "    nonlocal x\n" "    x += 10\n" "bump()\n"
            ),
        },
        tree=ast.parse(src_d_unsafe_sib),
    )
    assert "severed_closure_cell" in sc_d_unsafe_sib["control_flow_hazards"]

    # 7. _node_earliest_start_pos snaps @foo, @ foo, and @  foo to def col and handles multiline decorator
    tree_e_aligned = ast.parse(
        "def outer():\n" "    @  dec\n" "    def inner():\n" "        return 1\n"
    )
    inner_aligned = tree_e_aligned.body[0].body[0]  # type: ignore[attr-defined]
    assert _node_earliest_start_pos(inner_aligned) == (2, 4)
    tree_e_multiline_dec = ast.parse(
        "def outer():\n"
        "    @(\n"
        "        dec\n"
        "    )\n"
        "    def inner():\n"
        "        return 1\n"
    )
    inner_ml_dec = tree_e_multiline_dec.body[0].body[0]  # type: ignore[attr-defined]
    assert _node_earliest_start_pos(inner_ml_dec) == (3, 4)


def test_lexical_def_escape_identity_genexp_matrix_and_utf8_ambiguity() -> None:
    """Covers lexical def escape identity, column-aware shortcuts, mod_tree cache isolation, genexp matrix, runtime definite-binding validation, and UTF-8 single-boundary ambiguity."""
    import sys
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_intra_unit_defs,
        _collect_escaped_intra_unit_names,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
        _make_empty_traversal_buffers,
        _try_apply_cached_stmt_delta,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer.scope import (
        _find_enclosing_pre_bound_names,
        _unit_cols,
        inspect_single_unit_scope,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    # 1. Lexical-scope definition identity in _collect_escaped_intra_unit_defs
    src_item1_safe = (
        "def entry(a):\n"
        "    value = a + 10\n"
        "    value = value * 2\n"
        "    def helper():\n"
        "        def callback():\n"
        "            nonlocal value\n"
        "            value += 5\n"
        "        callback()\n"
        "        return 0\n"
        "    helper()\n"
        "    def other():\n"
        "        def callback():\n"
        "            return 1\n"
        "        return callback\n"
        "    escaped_cb = other()\n"
        "    return value, escaped_cb\n"
    )
    tree_item1_safe = ast.parse(src_item1_safe)
    unit_code_item1 = (
        "value = value * 2\n"
        "def helper():\n"
        "    def callback():\n"
        "        nonlocal value\n"
        "        value += 5\n"
        "    callback()\n"
        "    return 0\n"
        "helper()\n"
        "def other():\n"
        "    def callback():\n"
        "        return 1\n"
        "    return callback\n"
        "escaped_cb = other()\n"
    )
    cand_item1 = ast.parse(unit_code_item1).body
    helper_cb_node = cand_item1[1].body[0]  # type: ignore[attr-defined]
    other_cb_node = cand_item1[3].body[0]  # type: ignore[attr-defined]
    esc_names_1, esc_ids_1 = _collect_escaped_intra_unit_defs(
        cand_item1, {"value", "escaped_cb", "callback"}
    )
    assert "escaped_cb" in esc_names_1
    assert id(helper_cb_node) not in esc_ids_1
    assert id(other_cb_node) in esc_ids_1
    assert "callback" not in _collect_escaped_intra_unit_names(cand_item1, {"value", "escaped_cb"})
    sc_item1_safe = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 15,
            "kind": "compound_block",
            "source_text": src_item1_safe,
            "source_code": unit_code_item1,
        },
        tree=tree_item1_safe,
    )
    assert "severed_closure_cell" not in sc_item1_safe["control_flow_hazards"]

    # Ambiguous same-scope branch definitions with same name conservatively mark both as escaping
    src_item1_ambig = (
        "def entry(a, flag):\n"
        "    value = a + 10\n"
        "    value = value * 2\n"
        "    if flag:\n"
        "        def callback():\n"
        "            nonlocal value\n"
        "            value += 5\n"
        "    else:\n"
        "        def callback():\n"
        "            return 1\n"
        "    return value, callback\n"
    )
    sc_item1_ambig = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 10,
            "kind": "compound_block",
            "source_text": src_item1_ambig,
            "source_code": (
                "value = value * 2\n"
                "if flag:\n"
                "    def callback():\n"
                "        nonlocal value\n"
                "        value += 5\n"
                "else:\n"
                "    def callback():\n"
                "        return 1\n"
            ),
        },
        tree=ast.parse(src_item1_ambig),
    )
    assert "severed_closure_cell" in sc_item1_ambig["control_flow_hazards"]

    # 2. Column-aware _try_apply_cached_stmt_delta and same-line scope boundaries
    same_line_src = (
        "def f(items):\n" "    a = 1; b = [lambda: a for _ in items]; a = 2\n" "    return b, a\n"
    )
    same_line_tree = ast.parse(same_line_src)
    fn_same_line = same_line_tree.body[0]
    assert isinstance(fn_same_line, ast.FunctionDef)
    ctx_sl = _get_fn_scope_context(fn_same_line, mod_tree=same_line_tree)
    bufs_sl = _make_empty_traversal_buffers(frozenset(), ctx_sl)
    deltas_sl: dict = {}
    span_sl = _SpanBounds(2, 2, 11, 41)
    assert _try_apply_cached_stmt_delta(fn_same_line.body[0], span_sl, bufs_sl, deltas_sl) is True
    assert "a" in bufs_sl.state.pre_bound
    assert _try_apply_cached_stmt_delta(fn_same_line.body[2], span_sl, bufs_sl, deltas_sl) is True
    assert "a" in bufs_sl.state.post_mutated

    # _find_enclosing_function_chain and _find_enclosing_pre_bound_names with end_col in header vs body
    hdr_body_src = (
        "def outer(items):\n"
        "    x = 1\n"
        "    def inner(d=[i + x for i in items]): return d\n"
        "    return inner\n"
    )
    hdr_body_tree = ast.parse(hdr_body_src)
    chain_hdr = _find_enclosing_function_chain(hdr_body_tree, 3, 3, u_start_col=16, u_end_col=38)
    assert [n.name for _, _, n in chain_hdr] == ["outer"]  # type: ignore[attr-defined]
    chain_body = _find_enclosing_function_chain(hdr_body_tree, 3, 3, u_start_col=41, u_end_col=49)
    assert [n.name for _, _, n in chain_body] == ["outer", "inner"]  # type: ignore[attr-defined]
    pre_hdr, _ = _find_enclosing_pre_bound_names(hdr_body_tree, 3, 3, u_start_col=16, u_end_col=38)
    assert "x" in pre_hdr and "d" not in pre_hdr
    pre_body, _ = _find_enclosing_pre_bound_names(hdr_body_tree, 3, 3, u_start_col=41, u_end_col=49)
    assert "d" in pre_body

    # 3. Closure analysis cache isolation across distinct mod_tree contexts and eviction
    shared_fn_src = (
        "def worker(items):\n"
        "    v = 1\n"
        "    seq = list(v + x for x in items)\n"
        "    v = 99\n"
        "    res = v + 1\n"
        "    return seq, res\n"
    )
    mod_clean = ast.parse("SCALE = 1\n" + shared_fn_src)
    mod_shadow = ast.parse("list = custom_list\n" + shared_fn_src)
    shared_fn_node = mod_clean.body[1]
    ctx_clean = _get_fn_scope_context(shared_fn_node, mod_tree=mod_clean)
    ctx_shadow = _get_fn_scope_context(shared_fn_node, mod_tree=mod_shadow)
    assert isinstance(ctx_clean.fn_locals, frozenset)
    assert isinstance(ctx_clean.shadowed_names, frozenset)
    assert "list" not in ctx_clean.shadowed_names
    assert "list" in ctx_shadow.shadowed_names
    info_clean = _analyze_enclosing_fn_cells(shared_fn_node, (5, 6, None, None), mod_tree=mod_clean)
    info_shadow = _analyze_enclosing_fn_cells(
        shared_fn_node, (5, 6, None, None), mod_tree=mod_shadow
    )
    assert not any("v" in r for _, r, _ in info_clean.sibling_closures)
    assert any("v" in r for _, r, _ in info_shadow.sibling_closures)
    shadow_nodes = set(ast.walk(mod_shadow))
    assert any(
        isinstance(k, _ScopeCacheKey) and (k.owner in shadow_nodes or mod_shadow in k.aux_ast_nodes)
        for k in _scope_cell_info_cache
    )
    assert any(_key_references_scope_nodes(k, shadow_nodes) for k in _scope_cell_info_cache)
    _evict_tree_scopes(mod_shadow)
    assert not any(
        isinstance(k, _ScopeCacheKey) and (k.owner in shadow_nodes or mod_shadow in k.aux_ast_nodes)
        for k in _scope_cell_info_cache
    )
    assert not any(_key_references_scope_nodes(k, shadow_nodes) for k in _scope_cell_info_cache)
    with pytest.raises(ValueError, match="aux_ast_nodes"):
        _ScopeCacheKey(shared_fn_node, "bad_params", params=(mod_shadow,))
    clear_dataflow_cache()
    assert len(_scope_cell_info_cache) == 0

    # 4. Generator-expression 6-path interaction matrix
    # 4.1 Lazy genexp consumed AFTER rebinding vs BEFORE rebinding
    src_gen_after = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    value = 10\n"
        "    return list(gen)\n"
    )
    sc_gen_after = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 36,
            "kind": "comprehension",
            "source_text": src_gen_after,
            "source_code": "(value + x for x in items)",
        },
        tree=ast.parse(src_gen_after),
    )
    assert "severed_closure_cell" in sc_gen_after["control_flow_hazards"]

    src_gen_before = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    out = list(gen)\n"
        "    value = 10\n"
        "    res = value + 1\n"
        "    return out, res\n"
    )
    sc_gen_before = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 6,
            "kind": "compound_block",
            "source_text": src_gen_before,
            "source_code": "value = 10\nres = value + 1\n",
        },
        tree=ast.parse(src_gen_before),
    )
    assert "severed_closure_cell" not in sc_gen_before["control_flow_hazards"]

    # 4.2 Generator expression returned from unit to caller before outer rebinding
    src_gen_ret = (
        "def outer(items):\n"
        "    value = 1\n"
        "    gen = (value + x for x in items)\n"
        "    res = value + 2\n"
        "    value = 10\n"
        "    return gen, res\n"
    )
    sc_gen_ret = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_gen_ret,
            "source_code": "gen = (value + x for x in items)\nres = value + 2\n",
        },
        tree=ast.parse(src_gen_ret),
    )
    assert "severed_closure_cell" in sc_gen_ret["control_flow_hazards"]

    # 4.3 Shadowed sum/tuple consumer treated as lazy
    src_gen_shadow_sum = (
        "def outer(items, sum):\n"
        "    value = 1\n"
        "    total = sum(value + x for x in items)\n"
        "    value = 10\n"
        "    return total\n"
    )
    sc_gen_shadow_sum = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 15,
            "end_col": 41,
            "kind": "comprehension",
            "source_text": src_gen_shadow_sum,
            "source_code": "(value + x for x in items)",
        },
        tree=ast.parse(src_gen_shadow_sum),
    )
    assert "severed_closure_cell" in sc_gen_shadow_sum["control_flow_hazards"]

    # 4.4 Generator expression nested in list comprehension
    src_gen_in_listcomp = (
        "def outer(matrix):\n"
        "    value = 1\n"
        "    gens = [(value + x for x in row) for row in matrix]\n"
        "    value = 10\n"
        "    return [list(g) for g in gens]\n"
    )
    sc_gen_in_listcomp = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 11,
            "end_col": 55,
            "kind": "comprehension",
            "source_text": src_gen_in_listcomp,
            "source_code": "[(value + x for x in row) for row in matrix]",
        },
        tree=ast.parse(src_gen_in_listcomp),
    )
    assert "severed_closure_cell" in sc_gen_in_listcomp["control_flow_hazards"]

    # 4.5 Walrus in first iterable (eager) vs in body (lazy)
    src_walrus_iter = (
        "def outer(items):\n"
        "    w = 0\n"
        "    _, gen = (w := items), (x + 1 for x in w)\n"
        "    y = w\n"
        "    z = len(y)\n"
        "    return list(gen), z\n"
    )
    sc_walrus_iter = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_walrus_iter,
            "source_code": "y = w\nz = len(y)\n",
        },
        tree=ast.parse(src_walrus_iter),
    )
    assert "severed_closure_cell" not in sc_walrus_iter["control_flow_hazards"]

    src_walrus_body = (
        "def outer(items):\n"
        "    w = 0\n"
        "    gen = ((w := x) for x in items)\n"
        "    w = 10\n"
        "    res = list(gen) + [w]\n"
        "    return res\n"
    )
    sc_walrus_body = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 5,
            "kind": "compound_block",
            "source_text": src_walrus_body,
            "source_code": "w = 10\nres = list(gen) + [w]\n",
        },
        tree=ast.parse(src_walrus_body),
    )
    assert "severed_closure_cell" in sc_walrus_body["control_flow_hazards"]

    # 4.6 Nested generator expressions where inner target shadows outer variable
    src_gen_shadow_target = (
        "def outer(matrix):\n"
        "    value = 1\n"
        "    gen = (sum(value + 1 for value in row) for row in matrix)\n"
        "    value = 10\n"
        "    return list(gen), value\n"
    )
    sc_gen_shadow_target = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 61,
            "kind": "comprehension",
            "source_text": src_gen_shadow_target,
            "source_code": "(sum(value + 1 for value in row) for row in matrix)",
        },
        tree=ast.parse(src_gen_shadow_target),
    )
    assert "value" not in sc_gen_shadow_target["inputs"]
    assert "severed_closure_cell" not in sc_gen_shadow_target["control_flow_hazards"]

    # 5. Definite-binding validation against Python runtime execution & refactoring decision
    prog_try_finally = (
        "def run_case(flag):\n"
        "    try:\n"
        "        if flag:\n"
        "            a = 1\n"
        "        else:\n"
        "            a = 2\n"
        "    finally:\n"
        "        if flag == 'abort':\n"
        "            raise RuntimeError('stop')\n"
        "        b = a + 10\n"
        "    return a, b\n"
    )
    tree_tf = ast.parse(prog_try_finally)
    fn_tf = tree_tf.body[0]
    assert isinstance(fn_tf, ast.FunctionDef)
    def_bound_tf, all_term_tf = _extract_block_definite_bindings_and_term(fn_tf.body[:1], None)
    assert def_bound_tf == {"a", "b"}
    assert all_term_tf is False
    ns_tf: dict = {}
    exec(prog_try_finally, ns_tf)  # nosec B102
    assert ns_tf["run_case"](True) == (1, 11)
    assert ns_tf["run_case"](False) == (2, 12)

    # Validate unseeded conditional store with sibling closure: both scope fields & resolve_clone_pair_outputs
    src_unseeded = (
        "def outer(flag):\n"
        "    if flag:\n"
        "        value = 10\n"
        "    def read():\n"
        "        return value\n"
        "    if not flag:\n"
        "        value = 20\n"
        "    return read()\n"
    )
    u_unseeded = {
        "start": 2,
        "end": 3,
        "kind": "if_block",
        "source_text": src_unseeded,
        "source_code": "if flag:\n    value = 10\n",
    }
    sc_unseeded = inspect_single_unit_scope(u_unseeded, tree=ast.parse(src_unseeded))
    assert sc_unseeded["definite_stores"] == []
    assert sc_unseeded["outputs"] == ["value"]
    assert sc_unseeded["conditional_outputs"] == ["value"]
    assert "severed_closure_cell" in sc_unseeded["control_flow_hazards"]
    assert (
        resolve_clone_pair_outputs(
            u_unseeded,
            dict(u_unseeded),
            sc_unseeded,
            dict(sc_unseeded),
            source_text1=src_unseeded,
            source_text2=src_unseeded,
        )
        is None
    )

    if sys.version_info >= (3, 10):
        prog_match = (
            "def eval_match(x):\n"
            "    match x:\n"
            "        case 1 if x > 0:\n"
            "            m = 10\n"
            "        case _:\n"
            "            m = 20\n"
            "    return m\n"
        )
        tree_m = ast.parse(prog_match)
        def_m, term_m = _extract_block_definite_bindings_and_term(
            tree_m.body[0].body[:1], None  # type: ignore[attr-defined]
        )
        assert def_m == {"m"} and not term_m
        ns_m: dict = {}
        exec(prog_match, ns_m)  # nosec B102
        assert ns_m["eval_match"](1) == 10
        assert ns_m["eval_match"](2) == 20

    # 6. UTF-8 single-boundary ambiguity preservation and round-trip verification
    ambig_expr_line = ['    tag = "αβγ_🚀"; res = a + b + c\n']
    res_s_col = len('    tag = "αβγ_🚀"; res = '.encode("utf-8"))
    # Starting at `a` in `a + b + c`, both `a + b` and `a + b + c` are compound ast.expr nodes -> ambiguous!
    assert _unit_cols({"kind": "complex_expr", "start_col": res_s_col}, ambig_expr_line) == (
        res_s_col,
        None,
    )
    # Ending at `c` in `a + b + c`, only `(a + b) + c` is a non-leaf ast.expr ending there -> unambiguous!
    res_e_col = len('    tag = "αβγ_🚀"; res = a + b + c'.encode("utf-8"))
    assert _unit_cols({"kind": "complex_expr", "end_col": res_e_col}, ambig_expr_line) == (
        res_s_col,
        res_e_col,
    )

    # Single-line compound statement `if flag: a = 1; b = 2` with only end_col has both If and Assign ending at end_col -> ambiguous!
    ambig_stmt_line = ['    if flag: a = "café"; b = "🚀"\n']
    stmt_end_col = len(ambig_stmt_line[0].rstrip("\r\n").encode("utf-8"))
    assert _unit_cols({"kind": "basic_block", "end_col": stmt_end_col}, ambig_stmt_line) == (
        None,
        stmt_end_col,
    )

    # Round-trip check across multibyte prefixes and unique expressions
    for prefix in ('    p = "αβγ"; ', '    p = "日本語_🚀_✨"; ', "    "):
        for expr_str, kind_str in (
            ("[x * 2 for x in items]", "comprehension"),
            ("a + b", "complex_expr"),
            ("{k: v for k, v in pairs}", "comprehension"),
        ):
            full_ln = f"{prefix}target = {expr_str}; tail = 1 # comment 🚀\n"
            exp_s = len(f"{prefix}target = ".encode("utf-8"))
            exp_e = exp_s + len(expr_str.encode("utf-8"))
            assert _unit_cols({"kind": kind_str, "source_code": expr_str}, [full_ln]) == (
                exp_s,
                exp_e,
            )
            assert _unit_cols({"kind": kind_str, "start_col": exp_s}, [full_ln]) == (exp_s, exp_e)
            assert _unit_cols({"kind": kind_str, "end_col": exp_e}, [full_ln]) == (exp_s, exp_e)
            assert _slice_utf8_line(full_ln, exp_s, exp_e) == expr_str
            assert _slice_unit_token_lines(
                {"kind": kind_str, "start_col": exp_s, "end_col": exp_e},
                [full_ln],
            ) == [expr_str]


def test_chained_comparison_walrus_short_circuiting() -> None:
    """Verifies chained comparison short-circuiting in _extract_unconditional_walrus_names and cell traversal."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells
    from pydoppelgangerhunt.fixer._df_with_provenance import _extract_unconditional_walrus_names
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    chain_cmp_expr = ast.parse("a < b < (y := f())", mode="eval").body
    assert _extract_unconditional_walrus_names(chain_cmp_expr) == set()
    first_cmp_expr = ast.parse("(x := g()) < (y := f()) < (z := h())", mode="eval").body
    assert _extract_unconditional_walrus_names(first_cmp_expr) == {"x", "y"}

    src_chain_short = (
        "def outer(a, b, f, items):\n"
        "    ok = a < b < (y := f())\n"
        "    cbs = [lambda x=x: x + y for x in items]\n"
        "    return ok, cbs\n"
    )
    tree_chain_short = ast.parse(src_chain_short)
    info_chain_short = _analyze_enclosing_fn_cells(
        tree_chain_short.body[0], (3, 3, 10, 44), mod_tree=tree_chain_short
    )
    assert "y" not in info_chain_short.pre_bound
    assert "y" in info_chain_short.pre_mutated
    sc_chain_short = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": src_chain_short,
            "source_code": "[lambda x=x: x + y for x in items]",
        },
        tree=tree_chain_short,
    )
    assert "severed_closure_cell" in sc_chain_short["control_flow_hazards"]

    src_chain_first = (
        "def outer(a, c, f, items):\n"
        "    ok = a < (y := f()) < c\n"
        "    cbs = [lambda x=x: x + y for x in items]\n"
        "    return ok, cbs\n"
    )
    tree_chain_first = ast.parse(src_chain_first)
    sc_chain_first = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 10,
            "end_col": 44,
            "kind": "comprehension",
            "source_text": src_chain_first,
            "source_code": "[lambda x=x: x + y for x in items]",
        },
        tree=tree_chain_first,
    )
    assert "severed_closure_cell" not in sc_chain_first["control_flow_hazards"]


def test_conditional_live_in_pre_bound_hazard_vs_intra_unit_safe(
    tmp_path: Path,
) -> None:
    """Verifies conditional live-ins with pre-unit binding raise severed_closure_cell while purely intra-unit conditional live-ins stay safe."""
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    src_cond_live_in_pre = (
        "def entry(flag1, flag2):\n"
        "    if flag1:\n"
        "        x = 10\n"
        "    if flag2:\n"
        "        x = 20\n"
        "        y = x + 1\n"
        "    else:\n"
        "        y = x + 2\n"
        "    return y\n"
    )
    sc_cond_pre = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 8,
            "kind": "compound_block",
            "source_text": src_cond_live_in_pre,
            "source_code": (
                "if flag2:\n" "    x = 20\n" "    y = x + 1\n" "else:\n" "    y = x + 2\n"
            ),
        },
        tree=ast.parse(src_cond_live_in_pre),
    )
    assert "x" not in sc_cond_pre["inputs"]
    assert "severed_closure_cell" in sc_cond_pre["control_flow_hazards"]
    assert sc_cond_pre["is_control_flow_safe"] is False
    _run_differential_refactoring_case(
        tmp_path,
        "cond_live_in_pre_bound_hazard",
        src_cond_live_in_pre,
        (4, 8),
        expect_patch=False,
        call_args_list=[(True, False), (False, True), (True, True)],
    )

    src_cond_intra_only = (
        "def entry(flag):\n"
        "    if flag:\n"
        "        x = 10\n"
        "    y = x + 1 if flag else 0\n"
        "    return y\n"
    )
    sc_cond_intra = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": src_cond_intra_only,
            "source_code": "if flag:\n    x = 10\ny = x + 1 if flag else 0\n",
        },
        tree=ast.parse(src_cond_intra_only),
    )
    assert "x" not in sc_cond_intra["inputs"]
    assert "severed_closure_cell" not in sc_cond_intra["control_flow_hazards"]


def test_patch_closure_mode_recomputation_does_not_mutate_caller_units(
    tmp_path: Path,
) -> None:
    """Verifies generate_refactoring_patch and ensure_unit_scope_closure_mode do not mutate caller unit dictionaries."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    src_patch_mode = (
        "def f1(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n\n"
        "def f2(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    return c\n"
    )
    patch_file = tmp_path / "mode_mod.py"
    patch_file.write_text(src_patch_mode, encoding="utf-8")
    u1_orig: dict[str, Any] = {
        "file": str(patch_file),
        "start": 2,
        "end": 3,
        "name": "f1:blk",
        "kind": "compound_block",
    }
    u2_orig: dict[str, Any] = {
        "file": str(patch_file),
        "start": 7,
        "end": 8,
        "name": "f2:blk",
        "kind": "compound_block",
    }
    _ = generate_refactoring_patch(
        [(1.0, u1_orig, u2_orig)],
        repo_root=str(tmp_path),
        replace_clones=False,
        closure_strictness="lenient",
    )
    assert "skip_pre_unit_closures" not in u1_orig
    assert "skip_pre_unit_closures" not in u2_orig
    recomputed_sc = ensure_unit_scope_closure_mode(
        u1_orig,
        {"skip_pre_unit_closures": False, "inputs": ["a"]},
        True,
        source_text=src_patch_mode,
    )
    assert recomputed_sc["skip_pre_unit_closures"] is True
    assert "skip_pre_unit_closures" not in u1_orig


def test_module_star_import_and_single_node_locals_cached_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies module star-import and single_node_locals checks are cached across multiple units in the same module."""
    import pydoppelgangerhunt.fixer._df_closures as df_closures_mod
    from pydoppelgangerhunt.fixer._df_cache import (
        _scope_cell_info_cache,
        clear_dataflow_cache,
    )
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    clear_dataflow_cache()
    large_lines = [f"CONST_{i} = {i}\n" for i in range(200)]
    for i in range(30):
        large_lines.append(
            f"def fn_{i}(items):\n"
            f"    scale = {i}\n"
            f"    tot = sum(x + scale for x in items)\n"
            f"    res = tot + scale\n"
            f"    return res\n"
        )
    large_src = "".join(large_lines)
    large_tree = ast.parse(large_src)

    star_walk_calls = 0
    orig_uncached_star = df_closures_mod._node_has_star_import_uncached

    def _counting_star_import(root: ast.AST) -> bool:
        nonlocal star_walk_calls
        if root is large_tree:
            star_walk_calls += 1
        return orig_uncached_star(root)

    monkeypatch.setattr(
        df_closures_mod,
        "_node_has_star_import_uncached",
        _counting_star_import,
    )
    for i in range(30):
        base_ln = 200 + i * 5 + 1
        sc_unit = inspect_single_unit_scope(
            {
                "start": base_ln + 2,
                "end": base_ln + 3,
                "kind": "compound_block",
                "source_text": large_src,
                "source_code": "tot = sum(x + scale for x in items)\nres = tot + scale\n",
            },
            tree=large_tree,
        )
        assert sc_unit["is_control_flow_safe"] is True
    assert star_walk_calls == 1
    assert any(
        getattr(k, "owner", None) is large_tree and getattr(k, "kind", None) == "has_star_import"
        for k in _scope_cell_info_cache
    )
    assert any(
        getattr(k, "owner", None) is large_tree and getattr(k, "kind", None) == "single_node_locals"
        for k in _scope_cell_info_cache
    )


def test_slice_utf8_line_and_augassign_post_read_and_trystar_bindings() -> None:
    """Verifies UTF-8 slice boundaries, AugAssign post-read/mutation tracking, TryStar bindings, and NamedTuple mutator copying."""
    import copy
    import pickle
    import sys
    from pydoppelgangerhunt.fixer import (
        _is_irrefutable_case,
        _is_irrefutable_pattern,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _IntraUnitMutator,
        _collect_unit_nonlocal_mutators,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells
    from pydoppelgangerhunt.fixer._df_hazards import _scope_reads_outputs_after_line
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        _slice_utf8_line,
    )

    assert _slice_utf8_line("é = foo(1)\n", 4, 4) == ""
    assert _slice_utf8_line("é = foo(1)\n", 5, 2) == "foo(1)\n"
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 4, "end_col": 4},
        ["    [x for x in y]\n"],
    ) == [""]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 4, "end_col": 0},
        ["    [\n", "        x for x in y\n", "    ]\n"],
    ) == ["[\n", "        x for x in y\n", ""]
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 50, "end_col": 70},
        ["[x for x in y]"],
    ) == []
    assert _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 50, "end_col": 70},
        ["[x for x in y]"],
        explicit_col_sliced=True,
    ) == ["[x for x in y]"]

    src_aug = (
        "def outer(items):\n"
        "    x = 1\n"
        "    x += sum([lambda: x for _ in items][0]() for _ in (1,))\n"
        "    return x\n"
    )
    tree_aug = ast.parse(src_aug)
    info_aug = _analyze_enclosing_fn_cells(tree_aug.body[0], (3, 3, 13, 39), mod_tree=tree_aug)
    assert "x" in info_aug.post_reads
    assert "x" in info_aug.post_mutated
    sc_aug = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "start_col": 13,
            "end_col": 39,
            "kind": "comprehension",
            "source_text": src_aug,
            "source_code": "[lambda: x for _ in items]",
        },
        tree=tree_aug,
    )
    assert "severed_closure_cell" in sc_aug["control_flow_hazards"]

    if sys.version_info >= (3, 11):
        src_trystar = (
            "def outer(op, items):\n"
            "    keep = 1\n"
            "    victim = 2\n"
            "    eg = 3\n"
            "    try:\n"
            "        del victim\n"
            "        op()\n"
            "    except* ValueError as eg:\n"
            "        pass\n"
            "    finally:\n"
            "        final_var = 4\n"
            "    cbs = [lambda x=x: (keep, final_var, x) for x in items]\n"
            "    return cbs\n"
        )
        tree_ts = ast.parse(src_trystar)
        fn_ts = tree_ts.body[0]
        assert isinstance(fn_ts, ast.FunctionDef)
        ts_bound, ts_term = _extract_block_definite_bindings_and_term(fn_ts.body[:4], None)
        assert ts_bound == {"keep", "final_var"}
        assert ts_term is False
        sc_ts_safe = inspect_single_unit_scope(
            {
                "start": 12,
                "end": 12,
                "start_col": 10,
                "end_col": 59,
                "kind": "comprehension",
                "source_text": src_trystar,
                "source_code": "[lambda x=x: (keep, final_var, x) for x in items]",
            },
            tree=tree_ts,
        )
        assert "severed_closure_cell" not in sc_ts_safe["control_flow_hazards"]

        src_trystar_unsafe = src_trystar.replace("(keep, final_var, x)", "(eg, x)")
        sc_ts_unsafe = inspect_single_unit_scope(
            {
                "start": 12,
                "end": 12,
                "start_col": 10,
                "end_col": 46,
                "kind": "comprehension",
                "source_text": src_trystar_unsafe,
                "source_code": "[lambda x=x: (eg, x) for x in items]",
            },
            tree=ast.parse(src_trystar_unsafe),
        )
        assert "severed_closure_cell" in sc_ts_unsafe["control_flow_hazards"]

    assert _is_irrefutable_pattern(None) is False
    assert _is_irrefutable_case(ast.AST()) is False

    gen_muts = _collect_unit_nonlocal_mutators(ast.parse("g = ((w := x) for x in items)\n").body)
    assert len(gen_muts) == 1
    assert isinstance(gen_muts[0], _IntraUnitMutator)
    assert isinstance(gen_muts[0].node, ast.GeneratorExp)
    copied_mut = copy.copy(gen_muts[0])
    pickled_mut = pickle.loads(pickle.dumps(gen_muts[0]))
    assert copied_mut.node is gen_muts[0].node
    assert isinstance(pickled_mut.node, ast.GeneratorExp)

    scope_fn_tree = ast.parse("def f():\n" "    nonlocal shared\n" "    shared = 1\n").body[0]
    assert (
        _scope_reads_outputs_after_line(
            scope_fn_tree,
            3,
            ["shared"],  # type: ignore[arg-type]
            end_col=14,
            start_line=3,
            start_col=4,
        )
        is True
    )


def test_pep227_class_scope_isolation_and_nested_tuple_cache_eviction() -> None:
    """Verifies PEP 227 class scope isolation for global/nonlocal declarations and nested tuple cache eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _collect_unit_nonlocal_mutators

    pep227_tree = ast.parse(
        "def outer():\n"
        "    def inner():\n"
        "        x = 1\n"
        "        class C:\n"
        "            global x\n"
        "            x = 99\n"
        "            def m(self):\n"
        "                nonlocal x\n"
        "                x = 2\n"
        "        return C\n"
        "    return inner\n"
    )
    outer_fn = pep227_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    cls_c = inner_fn.body[1]
    assert isinstance(cls_c, ast.ClassDef)
    cls_muts = _collect_unit_nonlocal_mutators([cls_c], mod_tree=pep227_tree)
    assert any(m.name == "C" and "x" in m.mutated_vars for m in cls_muts)
    assert any(m.name == "m" and "x" in m.mutated_vars for m in cls_muts)
    inner_muts = _collect_unit_nonlocal_mutators([inner_fn], mod_tree=pep227_tree)
    assert not any("x" in m.mutated_vars for m in inner_muts)

    nested_key = (("deep", (pep227_tree, "nested_tag")), 42)
    _scope_cell_info_cache[nested_key] = "sentinel"
    _evict_tree_scopes(pep227_tree)
    assert nested_key not in _scope_cell_info_cache


def test_suppress_with_name_errors_and_custom_bool_dunder() -> None:
    """Verifies ast.Name and truthiness checks inside suppress(...) are not assumed non-raising."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _find_enclosing_chain_definite_pre_bound,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # 1. Assignment from unbound local inside suppress(NameError)
    unbound_local_src = (
        "from contextlib import suppress\n"
        "def example(flag):\n"
        "    with suppress(NameError):\n"
        "        x = missing_local\n"
        "    cb = lambda: x\n"
        "    if flag:\n"
        "        missing_local = 1\n"
        "    return cb()\n"
    )
    t_ul = ast.parse(unbound_local_src)
    chain_ul = _find_enclosing_function_chain(t_ul, 5, 5)
    pre_ul = _find_enclosing_chain_definite_pre_bound(chain_ul, 5, 5, mod_tree=t_ul)
    assert "x" not in pre_ul
    sc_ul = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "closure",
            "source_text": unbound_local_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=t_ul,
    )
    assert "severed_closure_cell" in sc_ul["control_flow_hazards"]

    # 2. Assignment from undefined global inside suppress(NameError)
    undef_global_src = (
        "from contextlib import suppress\n"
        "def example():\n"
        "    with suppress(NameError):\n"
        "        x = missing_global\n"
        "    cb = lambda: x\n"
        "    return cb()\n"
    )
    t_ug = ast.parse(undef_global_src)
    chain_ug = _find_enclosing_function_chain(t_ug, 5, 5)
    pre_ug = _find_enclosing_chain_definite_pre_bound(chain_ug, 5, 5, mod_tree=t_ug)
    assert "x" not in pre_ug

    # 3. Return of undefined name inside suppress(NameError) does not terminate
    ret_undef_stmts = ast.parse("with suppress(NameError):\n" "    return missing_name\n").body
    assert _block_terminates(ret_undef_stmts) is False
    _, term_undef = _extract_block_definite_bindings_and_term(ret_undef_stmts, None)
    assert term_undef is False

    # 4. `not x` where `x` can have a user-defined __bool__ that raises
    not_x_stmts = ast.parse(
        "with suppress(RuntimeError):\n" "    y = not x\n" "    return y\n"
    ).body
    assert _block_terminates(not_x_stmts) is False
    b_not_x, term_not_x = _extract_block_definite_bindings_and_term(
        not_x_stmts, None, initial_bound=frozenset({"x"})
    )
    assert "y" not in b_not_x
    assert term_not_x is False
    def_not_x, cond_not_x = _analyze_block_assignment(not_x_stmts)
    assert "y" not in def_not_x
    assert "y" in cond_not_x

    # Contrast: `a = 0; y = not a; return y` (known built-in constant `a`) is non-raising
    safe_bool_stmts = ast.parse(
        "with suppress(RuntimeError):\n" "    a = 0\n" "    y = not a\n" "    return y\n"
    ).body
    assert _block_terminates(safe_bool_stmts) is True
    b_safe_bool, term_safe_bool = _extract_block_definite_bindings_and_term(safe_bool_stmts, None)
    assert {"a", "y"}.issubset(b_safe_bool)
    assert term_safe_bool is True


def test_unknown_and_aliased_and_nested_suppressing_context_managers() -> None:
    """Covers 3-state context manager classification, custom/unknown/aliased suppressors, and nested suppressors."""
    from pydoppelgangerhunt.fixer import inspect_single_unit_scope
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _classify_with_suppression,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # 1. Three-state classification: non_suppressing, suppressing, unknown
    w_non_sup = ast.parse("with open('data.txt') as f:\n    pass\n").body[0]
    w_sup = ast.parse("with suppress(ValueError):\n    pass\n").body[0]
    w_unk = ast.parse("with ExternalIgnoreValueError():\n    pass\n").body[0]
    assert isinstance(w_non_sup, ast.With)
    assert isinstance(w_sup, ast.With)
    assert isinstance(w_unk, ast.With)
    assert _classify_with_suppression(w_non_sup) == "non_suppressing"
    assert _classify_with_suppression(w_sup) == "suppressing"
    assert _classify_with_suppression(w_unk) == "unknown"

    # Known non-suppressing with + raise DOES terminate
    non_sup_raise = ast.parse("with open('data.txt') as f:\n    raise ValueError()\nx = 1\n").body
    assert _block_terminates([non_sup_raise[0]]) is True
    assert _extract_block_definite_bindings_and_term(non_sup_raise, None)[1] is True

    # 2. Custom / unknown context manager containing raise must NOT be treated as terminating
    custom_cm_src = (
        "class IgnoreValueError:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc_value, traceback):\n"
        "        return exc_type is ValueError\n"
        "\n"
        "def example(cond):\n"
        "    if cond:\n"
        "        with IgnoreValueError():\n"
        "            raise ValueError()\n"
        "    else:\n"
        "        x = 10\n"
        "    cb = lambda: x\n"
        "    return cb\n"
    )
    custom_tree = ast.parse(custom_cm_src)
    example_fn = custom_tree.body[1]
    assert isinstance(example_fn, ast.FunctionDef)
    assert _block_terminates(example_fn.body) is True
    assert _block_terminates([example_fn.body[0]]) is False
    bound_custom, term_custom = _extract_block_definite_bindings_and_term(
        [example_fn.body[0]], None
    )
    assert "x" not in bound_custom
    assert term_custom is False

    # Because `with IgnoreValueError(): raise ValueError()` can fall through with `x` unbound,
    # extracting `cb = lambda: x` at line 13 must be flagged as an unbound captured cell hazard!
    sc_custom = inspect_single_unit_scope(
        {
            "start": 13,
            "end": 13,
            "kind": "linear_block",
            "source_text": custom_cm_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=custom_tree,
    )
    assert "severed_closure_cell" in sc_custom["control_flow_hazards"]
    assert sc_custom["is_control_flow_safe"] is False

    # Same check when the context manager class is imported/external ("unknown" state)
    ext_cm_src = (
        "def example(cond):\n"
        "    if cond:\n"
        "        with ExternalIgnoreValueError():\n"
        "            raise ValueError()\n"
        "    else:\n"
        "        x = 10\n"
        "    cb = lambda: x\n"
        "    return cb\n"
    )
    ext_tree = ast.parse(ext_cm_src)
    sc_ext = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 7,
            "kind": "linear_block",
            "source_text": ext_cm_src,
            "source_code": "cb = lambda: x\n",
        },
        tree=ext_tree,
    )
    assert "severed_closure_cell" in sc_ext["control_flow_hazards"]

    # 3. Aliased contextlib.suppress (via assignment and via import ... as ...)
    aliased_assign_src = (
        "my_suppress = contextlib.suppress\n"
        "with my_suppress(KeyError):\n"
        "    before = 1\n"
        "    risky = d['missing']\n"
        "    after = 2\n"
        "    raise KeyError()\n"
        "tail = 3\n"
    )
    aliased_stmts = ast.parse(aliased_assign_src).body
    assert _block_terminates(aliased_stmts[:-1]) is False
    def_al, cond_al = _analyze_block_assignment(aliased_stmts)
    bound_al, term_al = _extract_block_definite_bindings_and_term(aliased_stmts, None)
    assert def_al == {"my_suppress", "before", "tail"}
    assert {"risky", "after"} <= cond_al
    assert bound_al == {"my_suppress", "before", "tail"}
    assert term_al is False

    aliased_import_src = (
        "from contextlib import suppress as sup\n"
        "with sup(ValueError):\n"
        "    risky = int('bad')\n"
        "    after = 2\n"
        "tail = 3\n"
    )
    imp_stmts = ast.parse(aliased_import_src).body
    def_imp, cond_imp = _analyze_block_assignment(imp_stmts)
    bound_imp, term_imp = _extract_block_definite_bindings_and_term(imp_stmts, None)
    assert def_imp == {"sup", "tail"}
    assert {"risky", "after"} <= cond_imp
    assert bound_imp == {"sup", "tail"}
    assert term_imp is False

    # 4. Nested suppressing context managers: termination propagation
    nested_non_term_src = (
        "with suppress(ValueError):\n"
        "    with suppress(KeyError):\n"
        "        raise ValueError()\n"
        "    return 1\n"
        "after = 2\n"
    )
    nested_non_term_stmts = ast.parse(nested_non_term_src).body
    assert _block_terminates([nested_non_term_stmts[0]]) is False
    bound_nnt, term_nnt = _extract_block_definite_bindings_and_term(nested_non_term_stmts, None)
    assert bound_nnt == {"after"}
    assert term_nnt is False

    nested_term_src = (
        "with suppress(ValueError):\n"
        "    with suppress(KeyError):\n"
        "        x = 10\n"
        "        return x\n"
        "after = 2\n"
    )
    nested_term_stmts = ast.parse(nested_term_src).body
    assert _block_terminates([nested_term_stmts[0]]) is True
    bound_nt, term_nt = _extract_block_definite_bindings_and_term(nested_term_stmts, None)
    assert bound_nt == {"x"}
    assert term_nt is True


def test_try_apply_cached_stmt_delta_self_rebinding_with_cm_as_cm() -> None:
    """Verifies _try_apply_cached_stmt_delta evaluates with pre-statement provenance on 'with cm() as cm:'."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _make_empty_traversal_buffers,
        _try_apply_cached_stmt_delta,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
    )

    mod = ast.parse(
        "def f():\n"
        "    with cm(ValueError) as cm:\n"
        "        x = may_fail()\n"
        "    with cm():\n"
        "        y = may_fail()\n"
    )
    fn_node = mod.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    ctx = _get_fn_scope_context(fn_node, mod)
    prov_init = _SuppressProvenance(suppressors=frozenset({"cm"}))
    bufs = _make_empty_traversal_buffers(
        frozenset(),
        ctx,
        initial_suppress_aliases=prov_init,
    )
    bounds = _SpanBounds(10, 10, None, None)
    stmt_deltas: dict[tuple[Any, ...], Any] = {}

    # First statement `with cm(ValueError) as cm:` uses pre-statement `cm` (suppressing)
    # and then clears `cm` from provenance for subsequent statements.
    assert _try_apply_cached_stmt_delta(fn_node.body[0], bounds, bufs, stmt_deltas) is True
    assert "x" not in bufs.state.pre_bound
    assert bufs.state.suppress_aliases is not None
    assert "cm" not in bufs.state.suppress_aliases.suppressors
    assert len(stmt_deltas) == 1

    # Re-applying cached delta with identical pre-statement provenance hits the cache
    bufs2 = _make_empty_traversal_buffers(
        frozenset(),
        ctx,
        initial_suppress_aliases=prov_init,
    )
    assert _try_apply_cached_stmt_delta(fn_node.body[0], bounds, bufs2, stmt_deltas) is True
    assert "x" not in bufs2.state.pre_bound
    assert bufs2.state.suppress_aliases is not None
    assert "cm" not in bufs2.state.suppress_aliases.suppressors
