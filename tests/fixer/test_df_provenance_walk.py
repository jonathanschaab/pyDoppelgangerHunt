"""Tests for forward suppression provenance walks across walrus, unpacking, with, try, match, and loops."""

from __future__ import annotations

import ast
import sys
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    inspect_single_unit_scope,
)
from tests.fixer._prov_test_helpers import (
    _PROVEN_EXIT_CLASS,
    _late_bound_capture_safety_by_mode,
)

@pytest.mark.parametrize(
    ("scenario_id", "setup_src", "mode", "expect_safe"),
    [
        pytest.param(
            "class_body_nonlocal_rebind_invalidates_cm",
            (
                "    class Trigger:\n"
                "        nonlocal cm\n"
                "        cm = SuppressErrors\n"
            ),
            "pedantic",
            False,
            id="class_body_nonlocal_rebind_invalidates_cm",
        ),
        pytest.param(
            "class_body_mutator_call_invalidates_cm",
            (
                "    def mutate():\n"
                "        nonlocal cm\n"
                "        cm = unknown_cm\n"
                "    class Trigger:\n"
                "        mutate()\n"
            ),
            "pedantic",
            False,
            id="class_body_mutator_call_invalidates_cm",
        ),
        pytest.param(
            "nested_fn_class_body_nonlocal_rebind_invalidates_cm",
            (
                "    def mutate():\n"
                "        class InnerTrigger:\n"
                "            nonlocal cm\n"
                "            cm = unknown_cm\n"
                "    mutate()\n"
            ),
            "pedantic",
            False,
            id="nested_fn_class_body_nonlocal_rebind_invalidates_cm",
        ),
        pytest.param(
            "comprehension_walrus_cross_scope_mutation_invalidates_cm",
            (
                "    def mutate():\n"
                "        nonlocal cm\n"
                "        return [(cm := unknown_cm) for _ in (1,)]\n"
                "    mutate()\n"
            ),
            "pedantic",
            False,
            id="comprehension_walrus_cross_scope_mutation_invalidates_cm",
        ),
        pytest.param(
            "negative_benign_class_body_preserves_proven_cm",
            (
                "    class SafeHelper:\n"
                "        TAG = 'v1'\n"
                "        @staticmethod\n"
                "        def noop():\n"
                "            return 1\n"
            ),
            "pedantic",
            True,
            id="negative_benign_class_body_preserves_proven_cm",
        ),
    ],
)
def test_immediate_class_body_and_comprehension_walrus_provenance(
    scenario_id: str,
    setup_src: str,
    mode: str,
    expect_safe: bool,
) -> None:
    """Verifies class bodies execute eagerly for provenance/cross-scope mutation and comprehension walruses propagate."""
    src = (
        "from contextlib import nullcontext\n"
        "def outer(SuppressErrors, unknown_cm, step, items):\n"
        "    cm = nullcontext\n"
        f"{setup_src}"
        "    with cm():\n"
        "        scale = step()\n"
        "    out = [(lambda: scale + x)() for x in items]\n"
        "    return out\n"
    )
    clear_dataflow_cache()
    tree = ast.parse(src)
    lines = src.splitlines()
    start_line = next(i + 1 for i, ln in enumerate(lines) if "out = [" in ln)
    sc = inspect_single_unit_scope(
        {
            "start": start_line,
            "end": start_line,
            "kind": "linear_block",
            "source_text": src,
            "closure_strictness": mode,
        },
        tree=tree,
    )
    assert sc["is_control_flow_safe"] is expect_safe, scenario_id

def test_branch_merged_suppress_provenance_and_with_splitting_strict_names() -> None:
    """Verifies branch-conservative _SuppressProvenance merging across compound statements and strict_names propagation."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment,
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _split_with_body_by_mode,
        _update_suppress_provenance_from_stmt,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

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
    prov_after_both = _update_suppress_provenance_from_stmt(
        both_clear_stmts[0], prov_init
    )
    assert "cm" not in prov_after_both.suppress_names
    assert "cm" in prov_after_both.non_suppress_names

    # 3. While / For / Try / With branch merges preserve pre-loop/pre-branch suppressor
    compound_branch_src = (
        "for item in items:\n"
        "    cm1 = other_factory()\n"
        "while cond:\n"
        "    cm2 = other_factory()\n"
        "try:\n"
        "    cm3 = other_factory()\n"
        "except Exception:\n"
        "    pass\n"
        "with suppress_errors():\n"
        "    cm4 = other_factory()\n"
    )
    comp_stmts = ast.parse(compound_branch_src).body
    prov_comp = _SuppressProvenance(suppressors=frozenset({"cm1", "cm2", "cm3", "cm4"}))
    for st in comp_stmts:
        prov_comp = _update_suppress_provenance_from_stmt(st, prov_comp)
    assert {"cm1", "cm2", "cm3", "cm4"} <= prov_comp.suppress_names
    neutral_with_stmt = ast.parse("with lock:\n    cm4 = other_factory()\n").body[0]
    prov_neutral_strict = _update_suppress_provenance_from_stmt(
        neutral_with_stmt,
        _SuppressProvenance(suppressors=frozenset({"cm4"})),
        strict_with_suppression=False,
    )
    assert "cm4" not in prov_neutral_strict.suppress_names
    prov_neutral_pedantic = _update_suppress_provenance_from_stmt(
        neutral_with_stmt,
        _SuppressProvenance(suppressors=frozenset({"cm4"})),
        strict_with_suppression=True,
    )
    assert "cm4" in prov_neutral_pedantic.suppress_names

    # 4. _split_with_body_by_mode forwards strict_names for proven suppressor aliases and rejects bare strings
    with_suppress_unknown_name = ast.parse(
        "with custom_suppressor(NameError):\n"
        "    y = possibly_unbound_name\n"
        "    return 1\n"
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

def test_multi_item_with_subsequent_targets_and_walrus_after_suppressor() -> None:
    """Verifies that in a multi-item with statement, items after a suppressing or unknown item bind targets and walrus names only conditionally."""
    from pydoppelgangerhunt.fixer._df_scope_ctx import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

    # 1. Intra-unit _analyze_block_assignment: item 1 suppresses, item 2 has `as res` and `(w := ...)`
    unit_src = (
        "with suppress(ValueError), (w := acquire()) as res:\n" "    body_var = 1\n"
    )
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

def test_imported_suppress_and_nullcontext_provenance() -> None:
    """Verifies imported/aliased suppress and nullcontext provenance."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

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
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

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

def test_walrus_left_to_right_order_and_compare_short_circuit_provenance() -> None:
    """Verifies left-to-right walrus evaluation order and ast.Compare short-circuit provenance merging."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmt,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

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
    stmt_cmp = ast.parse(
        "_ = a < (first_c := nullcontext) < (second_c := nullcontext)\n"
    ).body[0]
    prov_cmp = _update_suppress_provenance_from_stmt(stmt_cmp, prov_base)
    assert "first_c" in prov_cmp.non_suppressors
    assert "second_c" not in prov_cmp.non_suppressors
    assert "second_c" in prov_cmp.shadowed_names

    # 4. Short-circuited comparator rebinding an existing non-suppressor to an unknown call invalidates it
    stmt_cmp_rebind = ast.parse("_ = a < b < (x := unknown_factory())\n").body[0]
    prov_cmp_rebind = _update_suppress_provenance_from_stmt(
        stmt_cmp_rebind, prov_with_x_safe
    )
    assert "x" not in prov_cmp_rebind.non_suppressors
    assert "x" in prov_cmp_rebind.shadowed_names

def test_walrus_side_effects_in_non_raising_expressions_inside_suppress() -> None:
    """Verifies walrus bindings in non-raising expressions inside suppress() propagate safe/nonzero state."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

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
    defs_short, conds_short = _analyze_block_assignment(
        ast.parse(src_short_circuit_walrus).body
    )
    assert "b" not in defs_short
    assert "b" in conds_short

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
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

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
            "from contextlib import nullcontext as safe_cm\n"
            "safe_cm = unknown_factory()\n",
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
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

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
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    stmts = ast.parse(with_src).body
    defs, conds = _analyze_block_assignment(stmts, strict_with_suppression=strict_with)
    assert expected_defs <= defs
    assert not (expected_conds & defs)
    assert expected_conds <= conds

def test_with_item_provenance_advances_across_chained_walrus_and_as_target_shadowing() -> (
    None
):
    """Verifies _iter_with_item_provenance_steps advances item_prov across chained items."""
    # 1. Chained safe walrus manager: (manager := contextlib.nullcontext()) as resource, manager
    chained_safe_src = (
        "import contextlib\n"
        "def run_chained(calculate):\n"
        "    with (manager := contextlib.nullcontext()) as resource, manager:\n"
        "        result = calculate()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    chained_safe_tree = ast.parse(chained_safe_src)
    sc_chained_safe = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": chained_safe_src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "pedantic",
        },
        tree=chained_safe_tree,
    )
    assert "severed_closure_cell" not in sc_chained_safe["control_flow_hazards"]
    assert sc_chained_safe["is_control_flow_safe"] is True

    # 2. Chained walrus suppressor alias: (alias := contextlib.suppress)(ValueError), alias(KeyError)
    chained_sup_src = (
        "import contextlib\n"
        "def run_chained_sup(risky):\n"
        "    with (alias := contextlib.suppress)(ValueError), alias(KeyError):\n"
        "        result = risky()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    chained_sup_tree = ast.parse(chained_sup_src)
    sc_chained_sup = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 4,
            "kind": "compound_block",
            "source_text": chained_sup_src,
            "closure_strictness": "strict",
        },
        tree=chained_sup_tree,
    )
    assert "result" not in sc_chained_sup["definite_stores"]
    assert "result" in sc_chained_sup["conditional_outputs"]

    # 3. Earlier item `as open` shadows builtin open in subsequent item `open("path")`
    shadow_open_src = (
        "import contextlib\n"
        "def run_shadow_open( custom_cm, compute ):\n"
        "    with custom_cm() as open, open('path'):\n"
        "        result = compute()\n"
        "    cb = lambda: result\n"
        "    return cb()\n"
    )
    shadow_open_tree = ast.parse(shadow_open_src)
    sc_shadow_open = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "basic_block",
            "source_text": shadow_open_src,
            "source_code": "    cb = lambda: result\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": ["custom_cm"],
        },
        tree=shadow_open_tree,
    )
    assert "severed_closure_cell" in sc_shadow_open["control_flow_hazards"]

def test_async_with_nested_nullcontext_provenance_and_definite_assignment() -> None:
    """Verifies ast.AsyncWith with contextlib.nullcontext is recognized across provenance and definite-assignment helpers."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment,
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    async_src = (
        "import contextlib\n"
        "async def run_async(compute):\n"
        "    with contextlib.suppress(ValueError):\n"
        "        async with contextlib.nullcontext():\n"
        "            x = 1\n"
        "            return x\n"
        "    after = 2\n"
        "    async with contextlib.nullcontext():\n"
        "        y = compute()\n"
        "    cb = lambda: y\n"
        "    return cb()\n"
    )
    async_tree = ast.parse(async_src)
    async_fn = async_tree.body[1]
    assert isinstance(async_fn, ast.AsyncFunctionDef)
    prov_async = _seed_scope_suppress_aliases(async_fn, mod_tree=async_tree)

    assert (
        _block_terminates(
            [async_fn.body[0]],
            suppress_aliases=prov_async,
            strict_with_suppression=True,
        )
        is True
    )
    defs_a, conds_a = _analyze_block_assignment(
        async_fn.body[:1],
        suppress_aliases=prov_async,
        strict_with_suppression=True,
    )
    assert "x" in defs_a
    assert "x" not in conds_a

    bound_a, term_a = _extract_block_definite_bindings_and_term(
        async_fn.body[:1],
        None,
        suppress_aliases=prov_async,
        strict_with_suppression=True,
    )
    assert bound_a == {"x"}
    assert term_a is True

    sc_async_cb = inspect_single_unit_scope(
        {
            "start": 10,
            "end": 10,
            "kind": "basic_block",
            "source_text": async_src,
            "source_code": "    cb = lambda: y\n",
            "closure_strictness": "pedantic",
        },
        tree=async_tree,
    )
    assert "severed_closure_cell" not in sc_async_cb["control_flow_hazards"]

def test_comprehension_walrus_bindings_and_provenance_updates() -> None:
    """Verifies comprehension walrus expressions bind conditionally and update context-manager provenance."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
    from pydoppelgangerhunt.fixer._df_prov_exprs import (
        _update_suppress_provenance_from_expr_walrus,
    )
    from pydoppelgangerhunt.fixer._df_provenance_walk import _classify_with_suppression
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

    comp_stmts = ast.parse("vals = [y := x + 1 for x in items]\n").body
    def_assigned, cond_assigned = _analyze_block_assignment(comp_stmts)
    assert "vals" in def_assigned
    assert "y" not in def_assigned
    assert "y" in cond_assigned

    # Walrus inside comprehension assigning a known suppressor marks the alias as suppressing
    sup_src = (
        "import contextlib\n"
        "def f(items):\n"
        "    _ = [(cm := contextlib.suppress(ValueError)) for _ in items]\n"
        "    with cm:\n"
        "        val = 1\n"
    )
    sup_tree = ast.parse(sup_src)
    sup_fn = sup_tree.body[1]
    assert isinstance(sup_fn, ast.FunctionDef)
    prov_clean = _SuppressProvenance(
        stdlib_modules=frozenset({("contextlib", "contextlib")})
    )
    assert "cm" not in prov_clean.suppressors
    comp_expr = sup_fn.body[0].value  # type: ignore[attr-defined]
    prov_after_sup = _update_suppress_provenance_from_expr_walrus(
        (comp_expr,), prov_clean
    )
    assert "cm" in prov_after_sup.suppressors
    with_stmt = sup_fn.body[1]
    assert isinstance(with_stmt, ast.With)
    assert _classify_with_suppression(with_stmt, prov_after_sup) == "suppressing"

    # Positive control without comprehension walrus keeps nullcontext() non-suppressing proof
    safe_src = (
        "import contextlib\n"
        "def g_safe():\n"
        "    cm = contextlib.nullcontext()\n"
        "    with cm:\n"
        "        val = 1\n"
    )
    safe_tree = ast.parse(safe_src)
    safe_fn = safe_tree.body[1]
    assert isinstance(safe_fn, ast.FunctionDef)
    prov_safe = _seed_scope_suppress_aliases(safe_fn, mod_tree=safe_tree, u_start=4)
    assert "cm" in prov_safe.non_suppressor_instances
    with_safe = safe_fn.body[1]
    assert isinstance(with_safe, ast.With)
    assert _classify_with_suppression(with_safe, prov_safe) == "non_suppressing"

    # Adversarial control: walrus inside comprehension rebinding a previously proven non-suppressor clears proof
    rebind_src = (
        "import contextlib\n"
        "def g(items, CustomUnknown):\n"
        "    cm = contextlib.nullcontext()\n"
        "    _ = [(cm := CustomUnknown()) for _ in items]\n"
        "    with cm:\n"
        "        val = 1\n"
    )
    rebind_tree = ast.parse(rebind_src)
    rebind_fn = rebind_tree.body[1]
    assert isinstance(rebind_fn, ast.FunctionDef)
    prov_rebind = _seed_scope_suppress_aliases(rebind_fn, mod_tree=rebind_tree, u_start=5)
    assert "cm" not in prov_rebind.non_suppressor_instances
    with_rebind = rebind_fn.body[2]
    assert isinstance(with_rebind, ast.With)
    assert _classify_with_suppression(with_rebind, prov_rebind) == "unknown"

def test_lambda_body_walrus_excluded_while_lambda_default_walrus_binds() -> None:
    """Verifies lambda body walruses do not leak into enclosing assignments while parameter defaults do."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    body_walrus_stmts = ast.parse("fn = lambda: (leaked := 42)\n").body
    defs_body, conds_body = _analyze_block_assignment(body_walrus_stmts)
    assert "fn" in defs_body
    assert "leaked" not in defs_body
    assert "leaked" not in conds_body

    default_walrus_stmts = ast.parse("fn = lambda a=(eager_default := 42): a\n").body
    defs_default, conds_default = _analyze_block_assignment(default_walrus_stmts)
    assert "fn" in defs_default
    assert "eager_default" in defs_default
    assert "eager_default" not in conds_default

@pytest.mark.parametrize("use_callable_alias", [False, True])
def test_sequential_local_ctx_manager_rebinding_invalidates_prior_provenance(
    use_callable_alias: bool,
) -> None:
    """Verifies sequential local rebinding of a context-manager variable invalidates prior provenance."""
    first_bind = "cm = Transaction" if use_callable_alias else "cm = Transaction()"
    first_with = "cm()" if use_callable_alias else "cm"
    second_bind = (
        "cm = SuppressErrors" if use_callable_alias else "cm = SuppressErrors()"
    )
    second_with = "cm()" if use_callable_alias else "cm"
    src = (
        "def run_pipeline(compute):\n"
        f"    {first_bind}\n"
        f"    with {first_with}:\n"
        "        safe_val = compute()\n"
        f"    {second_bind}\n"
        f"    with {second_with}:\n"
        "        risky_val = compute()\n"
        "    safe_cb = lambda: safe_val\n"
        "    risky_cb = lambda: risky_val\n"
        "    return safe_cb(), risky_cb()\n"
    )
    tree = ast.parse(src)
    for mode, allowlist in (("strict", []), ("pedantic", ["Transaction"])):
        sc_safe = inspect_single_unit_scope(
            {
                "start": 8,
                "end": 8,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    safe_cb = lambda: safe_val\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": allowlist,
            },
            tree=tree,
        )
        assert sc_safe["is_control_flow_safe"] is True, (mode, use_callable_alias)
        sc_risky = inspect_single_unit_scope(
            {
                "start": 9,
                "end": 9,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    risky_cb = lambda: risky_val\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": allowlist,
            },
            tree=tree,
        )
        assert sc_risky["is_control_flow_safe"] is False, (mode, use_callable_alias)
        assert "severed_closure_cell" in sc_risky["control_flow_hazards"]

    # Negative control: rebinding from a suppressor to proven contextlib.nullcontext restores safety
    recover_src = (
        "import contextlib\n"
        "def recover_pipeline(compute):\n"
        "    cm = SuppressErrors()\n"
        "    with cm:\n"
        "        _ignored = compute()\n"
        "    cm = contextlib.nullcontext()\n"
        "    with cm:\n"
        "        clean_val = compute()\n"
        "    clean_cb = lambda: clean_val\n"
        "    return clean_cb()\n"
    )
    sc_recovered = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 9,
            "kind": "basic_block",
            "source_text": recover_src,
            "source_code": "    clean_cb = lambda: clean_val\n",
            "closure_strictness": "pedantic",
        },
        tree=ast.parse(recover_src),
    )
    assert sc_recovered["is_control_flow_safe"] is True

@pytest.mark.parametrize(
    ("scope_kind", "initial_mgr"),
    [
        ("global", "Transaction"),
        ("global", "contextlib.nullcontext"),
        ("nonlocal", "Transaction"),
        ("nonlocal", "contextlib.nullcontext"),
    ],
)
def test_cross_scope_global_and_nonlocal_helper_rebinding_invalidates_ctx_manager_provenance(
    scope_kind: str,
    initial_mgr: str,
) -> None:
    """Verifies helper calls that rebind global/nonlocal manager aliases invalidate non-suppression provenance."""
    if scope_kind == "global":
        src = (
            "import contextlib\n"
            f"cm = {initial_mgr}\n"
            "def replace_manager():\n"
            "    global cm\n"
            "    cm = SuppressErrors\n"
            "def execute_step(compute):\n"
            "    with cm():\n"
            "        replace_manager()\n"
            "    with cm():\n"
            "        after_swap = compute()\n"
            "    cb = lambda: after_swap\n"
            "    return cb()\n"
        )
        unit_line = 11
    else:
        src = (
            "import contextlib\n"
            "def outer(compute):\n"
            f"    cm = {initial_mgr}\n"
            "    def replace_manager():\n"
            "        nonlocal cm\n"
            "        cm = SuppressErrors\n"
            "    with cm():\n"
            "        replace_manager()\n"
            "    with cm():\n"
            "        after_swap = compute()\n"
            "    cb = lambda: after_swap\n"
            "    return cb()\n"
        )
        unit_line = 11

    tree = ast.parse(src)
    for mode in ("strict", "pedantic"):
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    cb = lambda: after_swap\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": ["Transaction"],
            },
            tree=tree,
        )
        assert sc["is_control_flow_safe"] is False, (scope_kind, initial_mgr, mode)
        assert "severed_closure_cell" in sc["control_flow_hazards"]

@pytest.mark.parametrize(
    ("module_src", "unit_src", "expected_safe"),
    [
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "Safe = nullcontext\n"
            "Alias = Safe\n"
            "Safe = suppress\n"
            "def run(compute):\n"
            "    with Safe(Exception):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="rebound_original_name_is_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "Safe = nullcontext\n"
            "Alias = Safe\n"
            "Safe = suppress\n"
            "def run(compute):\n"
            "    with Alias():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            True,
            id="alias_keeps_object_bound_before_rebinding",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "Risky = suppress\n"
            "Alias = Risky\n"
            "Risky = nullcontext\n"
            "def run(compute):\n"
            "    with Alias(Exception):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="alias_of_suppressor_survives_original_rebinding",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(compute):\n"
            "    manager = nullcontext\n"
            "    def rebind():\n"
            "        nonlocal manager\n"
            "        manager = suppress\n"
            "    alias = manager\n"
            "    rebind()\n"
            "    with manager(Exception):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="nonlocal_rebinding_invalidates_original_name",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(compute, flag):\n"
            "    manager = nullcontext\n"
            "    if flag:\n"
            "        manager = suppress\n"
            "    with manager(Exception):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="conditional_rebinding_to_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(compute, items):\n"
            "    manager = nullcontext\n"
            "    for _ in items:\n"
            "        with manager():\n"
            "            v = compute()\n"
            "        manager = suppress\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="loop_carried_rebinding_to_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(compute):\n"
            "    with nullcontext(), suppress(Exception):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="multi_item_header_with_trailing_suppressor",
        ),
        pytest.param(
            "from contextlib import suppress\n"
            "def run(compute):\n"
            "    with suppress(ValueError) as (first, second):\n"
            "        pass\n"
            "    cb = lambda: first\n"
            "    return cb()\n",
            "    cb = lambda: first",
            False,
            id="raising_as_target_unpacking_under_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext\n"
            "def run(compute):\n"
            "    with nullcontext(compute()) as (first, second):\n"
            "        pass\n"
            "    cb = lambda: first\n"
            "    return cb()\n",
            "    cb = lambda: first",
            True,
            id="as_target_unpacking_under_proven_manager",
        ),
        pytest.param(
            "from contextlib import suppress\n"
            "def run(compute):\n"
            "    try:\n"
            "        with suppress(Exception):\n"
            "            v = compute()\n"
            "    finally:\n"
            "        pass\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            id="suppressor_nested_in_try_finally",
        ),
    ],
)
def test_ctx_manager_alias_rebinding_and_header_shapes_track_actual_object(
    module_src: str, unit_src: str, expected_safe: bool
) -> None:
    """Verifies provenance follows the object actually bound at with-entry across aliases, rebinding, and header shapes."""
    safety = _late_bound_capture_safety_by_mode(module_src, unit_src)
    assert safety == {"strict": expected_safe, "pedantic": expected_safe}

@pytest.mark.parametrize(
    ("case_id", "src", "unit_line_text", "expected_safe_by_mode"),
    [
        pytest.param(
            "multi_item_with_second_suppressor_taints_block",
            (
                "import contextlib\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext() as a, contextlib.suppress(RuntimeError) as b:\n"
                "        v = compute(a, b)\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": False, "pedantic": False},
            id="multi_item_with_second_suppressor",
        ),
        pytest.param(
            "multi_item_with_both_proven_safe",
            (
                "import contextlib\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext() as a, contextlib.nullcontext() as b:\n"
                "        v = compute(a, b)\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": True, "pedantic": True},
            id="multi_item_with_both_proven_safe",
        ),
        pytest.param(
            "unpacking_with_target_under_outer_suppress",
            (
                "import contextlib\n"
                "def run(pair_cm, compute):\n"
                "    with contextlib.suppress(ValueError, TypeError):\n"
                "        with pair_cm() as (a, b):\n"
                "            v = compute(a, b)\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": False, "pedantic": False},
            id="unpacking_with_target_under_outer_suppress",
        ),
        pytest.param(
            "nested_try_finally_rebinding_cm_alias_invalidates_proof",
            (
                "import contextlib\n"
                "from external_pkg import NeutralCM\n"
                "def run(compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    try:\n"
                "        cm = NeutralCM\n"
                "    finally:\n"
                "        pass\n"
                "    with cm():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": True, "pedantic": False},
            id="try_finally_rebinding_cm_alias_neutral",
        ),
        pytest.param(
            "nested_try_finally_rebinding_cm_to_local_suppressor",
            (
                "import contextlib\n"
                "class CustomSup:\n"
                "    def __enter__(self): return self\n"
                "    def __exit__(self, *a): return True\n"
                "def run(compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    try:\n"
                "        cm = CustomSup\n"
                "    finally:\n"
                "        pass\n"
                "    with cm():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": False, "pedantic": False},
            id="try_finally_rebinding_cm_to_local_suppressor",
        ),
        pytest.param(
            "loop_rebinding_cm_alias_across_iterations",
            (
                "import contextlib\n"
                "class CustomSup:\n"
                "    def __enter__(self): return self\n"
                "    def __exit__(self, *a): return True\n"
                "def run(items):\n"
                "    cm = contextlib.nullcontext\n"
                "    for item in items:\n"
                "        with cm():\n"
                "            v = item\n"
                "        cm = CustomSup\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": False, "pedantic": False},
            id="loop_rebinding_cm_alias",
        ),
        pytest.param(
            "outer_suppress_catches_inner_cm_enter_failure",
            (
                "import contextlib\n"
                + _PROVEN_EXIT_CLASS
                + "def run(compute):\n"
                "    with contextlib.suppress(ValueError):\n"
                "        with Mgr():\n"
                "            v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": False, "pedantic": False},
            id="outer_suppress_catches_inner_enter_raise",
        ),
        pytest.param(
            "outer_suppress_does_not_suppress_outer_pre_with_assignment",
            (
                "import contextlib\n"
                + _PROVEN_EXIT_CLASS
                + "def run(compute):\n"
                "    with Mgr():\n"
                "        with contextlib.nullcontext():\n"
                "            v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            {"strict": True, "pedantic": True},
            id="nested_proven_non_suppressing_withs",
        ),
    ],
)
def test_with_provenance_multi_item_unpacking_rebinding_and_outer_suppress_interactions(
    case_id: str,
    src: str,
    unit_line_text: str,
    expected_safe_by_mode: dict[str, bool],
) -> None:
    """Verifies multi-item with, unpacking targets, try/loop CM rebinding, and outer suppress wrapping __enter__/__exit__."""
    safety = _late_bound_capture_safety_by_mode(src, unit_line_text)
    for mode, expected_safe in expected_safe_by_mode.items():
        assert safety[mode] is expected_safe, (case_id, mode, safety)

@pytest.mark.parametrize(
    ("module_src", "expected_strict", "expected_pedantic"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "contextlib.nullcontext = lambda *a: True\n"
                "def run(compute):\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="prebound_nullcontext_instance_invalidated_by_factory_rebind",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "cm2 = cm\n"
                "contextlib.nullcontext = lambda *a: True\n"
                "def run(compute):\n"
                "    with cm2:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="chained_prebound_nullcontext_alias_invalidated_by_factory_rebind",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def rebind():\n"
                "    contextlib.nullcontext = lambda *a: True\n"
                "def run(compute):\n"
                "    cm = contextlib.nullcontext()\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="in_function_prebound_nullcontext_invalidated_before_subroutine",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "cm2 = cm\n"
                "def run(compute):\n"
                "    with cm2:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_prebound_nullcontext_chain_without_rebind",
        ),
        pytest.param(
            (
                "import tempfile\n"
                "from tempfile import TemporaryDirectory\n"
                "td = TemporaryDirectory\n"
                "td2 = td\n"
                "tempfile.TemporaryDirectory = lambda *a: True\n"
                "def run(compute):\n"
                "    with td2():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="general_stdlib_factory_alias_invalidated_by_rebind",
        ),
        pytest.param(
            (
                "from tempfile import TemporaryDirectory\n"
                "td = TemporaryDirectory\n"
                "td2 = td\n"
                "def run(compute):\n"
                "    with td2():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_general_stdlib_factory_alias_without_rebind",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "payload = {'k': 1, 'cm': True}\n"
                "match payload:\n"
                "    case {'k': 1, **cm}:\n"
                "        pass\n"
                "def run(compute):\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            False,
            id="module_level_match_mapping_rest_rebinds_cm",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "payload = {'k': 1, 'other': True}\n"
                "match payload:\n"
                "    case {'k': 1, **other}:\n"
                "        pass\n"
                "def run(compute):\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_module_level_match_mapping_rest_unrelated_target",
        ),
    ],
)
def test_rebound_prebound_instances_factory_aliases_and_mapping_rest(
    module_src: str, expected_strict: bool, expected_pedantic: bool
) -> None:
    """Verifies bound dunder rebinding, prebound stdlib instances, factory aliases, and mapping rest bindings."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expected_strict, "pedantic": expected_pedantic}

def test_try_except_finally_unhandled_exception_preserves_handler_base() -> None:
    """Verifies finally sees unhandled-exception handler_base provenance and post-try preserves it only on jump suppression."""
    from pydoppelgangerhunt.fixer._df_block_flow import _block_terminates
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

    with_finally_return_src = (
        "import contextlib\n"
        "def run(may_raise, step):\n"
        "    cm = contextlib.suppress\n"
        "    try:\n"
        "        may_raise()\n"
        "        cm = contextlib.nullcontext\n"
        "    except ValueError:\n"
        "        cm = contextlib.nullcontext\n"
        "    finally:\n"
        "        with cm(KeyError):\n"
        "            x = step()\n"
        "        return 1\n"
    )
    tree = ast.parse(with_finally_return_src)
    mod_prov = _update_suppress_provenance_from_stmts(
        tree.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    fn = tree.body[1]
    assert isinstance(fn, ast.FunctionDef)
    prov_after = _update_suppress_provenance_from_stmts(
        fn.body, mod_prov, in_branch=False
    )
    assert "cm" in prov_after.suppressors
    assert "cm" not in prov_after.non_suppressors

    # Without return/break/continue, unhandled exceptions re-raise on exit from finally
    without_jump_src = with_finally_return_src.replace("        return 1\n", "        step()\n")
    no_jump_tree = ast.parse(without_jump_src)
    no_jump_prov = _update_suppress_provenance_from_stmts(
        no_jump_tree.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    no_jump_fn = no_jump_tree.body[1]
    assert isinstance(no_jump_fn, ast.FunctionDef)
    no_jump_after = _update_suppress_provenance_from_stmts(
        no_jump_fn.body, no_jump_prov, in_branch=False
    )
    assert "cm" not in no_jump_after.suppressors
    assert "cm" in no_jump_after.non_suppressors

    # Inside finally, with cm(KeyError) returning inside body does not guarantee termination because cm may still be suppress
    finally_return_inside_with = (
        "import contextlib\n"
        "def run(may_raise, step):\n"
        "    cm = contextlib.suppress\n"
        "    try:\n"
        "        may_raise()\n"
        "        cm = contextlib.nullcontext\n"
        "    except ValueError:\n"
        "        cm = contextlib.nullcontext\n"
        "    finally:\n"
        "        with cm(KeyError):\n"
        "            x = step()\n"
        "            return x\n"
    )
    term_tree = ast.parse(finally_return_inside_with)
    term_mod_prov = _update_suppress_provenance_from_stmts(
        term_tree.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    term_fn = term_tree.body[1]
    assert isinstance(term_fn, ast.FunctionDef)
    assert _block_terminates(term_fn.body, suppress_aliases=term_mod_prov) is False

    loop_break_src = (
        "import contextlib\n"
        "def run(items, check, may_raise, compute):\n"
        "    for _ in items:\n"
        "        cm = contextlib.suppress\n"
        "        try:\n"
        "            may_raise()\n"
        "            cm = contextlib.nullcontext\n"
        "        except ValueError:\n"
        "            cm = contextlib.nullcontext\n"
        "        finally:\n"
        "            if check():\n"
        "                break\n"
        "    with cm(KeyError):\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    break_safety = _late_bound_capture_safety_by_mode(loop_break_src, "    cb = lambda: v")
    assert break_safety == {"strict": False, "pedantic": False}

    loop_break_unknown_src = (
        "import contextlib\n"
        "def run(items, check, may_raise, CustomCtx, compute):\n"
        "    for _ in items:\n"
        "        cm = CustomCtx\n"
        "        try:\n"
        "            may_raise()\n"
        "            cm = contextlib.nullcontext\n"
        "        except ValueError:\n"
        "            cm = contextlib.nullcontext\n"
        "        finally:\n"
        "            if check():\n"
        "                break\n"
        "    with cm():\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    unknown_safety = _late_bound_capture_safety_by_mode(
        loop_break_unknown_src, "    cb = lambda: v"
    )
    assert unknown_safety["pedantic"] is False

    # Control: when try body has no raising prefix before rebinding, cm is nullcontext on all paths into finally
    safe_prefix_only_src = (
        "import contextlib\n"
        "def run(compute):\n"
        "    cm = contextlib.suppress\n"
        "    try:\n"
        "        cm = contextlib.nullcontext\n"
        "    except ValueError:\n"
        "        cm = contextlib.nullcontext\n"
        "    finally:\n"
        "        pass\n"
        "    with cm():\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safe_prefix_safety = _late_bound_capture_safety_by_mode(
        safe_prefix_only_src, "    cb = lambda: v"
    )
    assert safe_prefix_safety == {"strict": True, "pedantic": True}

@pytest.mark.parametrize(
    ("module_src", "unit_src", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "def run(may_raise, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return ValueError\n"
                "    try:\n"
                "        may_raise()\n"
                "    except flip():\n"
                "        with cm(RuntimeError):\n"
                "            v = compute()\n"
                "        cb = lambda: v\n"
                "        return cb()\n"
            ),
            "        cb = lambda: v",
            False,
            id="except_header_cross_scope_mutator_invalidates_handler_body",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(may_raise, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return KeyError\n"
                "    try:\n"
                "        may_raise()\n"
                "    except flip():\n"
                "        pass\n"
                "    except ValueError:\n"
                "        def inner():\n"
                "            with cm(RuntimeError):\n"
                "                v = compute()\n"
                "            cb = lambda: v\n"
                "            return cb()\n"
                "        return inner()\n"
            ),
            "            cb = lambda: v",
            False,
            id="prior_except_header_cross_scope_mutator_invalidates_later_handler_seed",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return True\n"
                "    match payload:\n"
                "        case 1 if flip():\n"
                "            with cm(RuntimeError):\n"
                "                v = compute()\n"
                "            cb = lambda: v\n"
                "            return cb()\n"
            ),
            "            cb = lambda: v",
            False,
            id="match_guard_cross_scope_mutator_invalidates_own_case_body",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return False\n"
                "    match payload:\n"
                "        case 1 if flip():\n"
                "            pass\n"
                "        case _:\n"
                "            with cm(RuntimeError):\n"
                "                v = compute()\n"
                "            cb = lambda: v\n"
                "            return cb()\n"
            ),
            "            cb = lambda: v",
            False,
            id="prior_match_guard_cross_scope_mutator_invalidates_later_case_body",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return False\n"
                "    match payload:\n"
                "        case 1 if flip():\n"
                "            pass\n"
                "        case _:\n"
                "            def inner():\n"
                "                with cm(RuntimeError):\n"
                "                    v = compute()\n"
                "                cb = lambda: v\n"
                "                return cb()\n"
                "            return inner()\n"
            ),
            "                cb = lambda: v",
            False,
            id="prior_match_guard_cross_scope_mutator_invalidates_later_case_nested_seed",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def flip():\n"
                "        nonlocal cm\n"
                "        cm = contextlib.suppress\n"
                "        return False\n"
                "    match payload:\n"
                "        case 1 if flip():\n"
                "            cm = contextlib.nullcontext\n"
                "        case _:\n"
                "            pass\n"
                "    with cm(RuntimeError):\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "    cb = lambda: v",
            False,
            id="prior_match_guard_mutator_with_taken_restore_still_invalidates_post_match",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def check():\n"
                "        return False\n"
                "    match payload:\n"
                "        case cm if check():\n"
                "            pass\n"
                "        case _:\n"
                "            def inner():\n"
                "                with cm(RuntimeError):\n"
                "                    v = compute()\n"
                "                cb = lambda: v\n"
                "                return cb()\n"
                "            return inner()\n"
            ),
            "                cb = lambda: v",
            False,
            id="prior_match_pattern_binding_cm_before_failed_guard_invalidates_later_seed",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(payload, may_raise, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    def exc_type():\n"
                "        return ValueError\n"
                "    def ok():\n"
                "        return False\n"
                "    try:\n"
                "        may_raise()\n"
                "    except exc_type():\n"
                "        pass\n"
                "    except TypeError:\n"
                "        match payload:\n"
                "            case 1 if ok():\n"
                "                pass\n"
                "            case _:\n"
                "                def inner():\n"
                "                    with cm():\n"
                "                        v = compute()\n"
                "                    cb = lambda: v\n"
                "                    return cb()\n"
                "                return inner()\n"
            ),
            "                    cb = lambda: v",
            True,
            id="control_benign_except_header_and_match_guard_preserve_proven_cm",
        ),
    ],
)
def test_clause_header_and_prior_match_guard_cross_scope_invalidation(
    module_src: str, unit_src: str, expect_pedantic_safe: bool
) -> None:
    """Verifies except headers and match guards propagate cross-scope mutator invalidation."""
    safety = _late_bound_capture_safety_by_mode(module_src, unit_src)
    assert safety["pedantic"] is expect_pedantic_safe

@pytest.mark.parametrize(
    ("module_src", "target_marker", "expected_safety"),
    [
        pytest.param(
            (
                "from my_suppressors import SuppressAll as cm\n"
                "\n"
                "class Worker:\n"
                "    from contextlib import nullcontext as cm\n"
                "\n"
                "    def run(self, compute):\n"
                "        with cm():\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "        return cb()\n"
            ),
            "        cb = lambda: val",
            {"strict": False, "pedantic": False},
            id="class_body_nullcontext_does_not_mask_module_suppressor",
        ),
        pytest.param(
            (
                "class FileService:\n"
                "    def open(self):\n"
                "        return 1\n"
                "\n"
                "    def read_items(self, path, compute):\n"
                "        with open(path) as fh:\n"
                "            val = compute(fh)\n"
                "        cb = lambda: val\n"
                "        return cb()\n"
            ),
            "        cb = lambda: val",
            {"strict": True, "pedantic": True},
            id="sibling_method_open_does_not_shadow_builtin_open",
        ),
        pytest.param(
            (
                "class FileService:\n"
                "    def open(self, path, compute):\n"
                "        with open(path) as fh:\n"
                "            val = compute(fh)\n"
                "        cb = lambda: val\n"
                "        return cb()\n"
            ),
            "        cb = lambda: val",
            {"strict": True, "pedantic": True},
            id="method_named_open_uses_builtin_open_not_class_entry",
        ),
        pytest.param(
            (
                "def outer(flag):\n"
                "    from my_suppressors import SuppressAll as cm\n"
                "    class Worker:\n"
                "        from contextlib import nullcontext as cm\n"
                "        def run(self, compute):\n"
                "            with cm():\n"
                "                val = compute()\n"
                "            cb = lambda: val\n"
                "            return cb()\n"
                "    return Worker()\n"
            ),
            "            cb = lambda: val",
            {"strict": False, "pedantic": False},
            id="outer_fn_suppressor_not_masked_by_inner_class_binding",
        ),
        pytest.param(
            (
                "from my_suppressors import SuppressAll as cm\n"
                "\n"
                "class Outer:\n"
                "    if True:\n"
                "        from contextlib import nullcontext as cm\n"
                "        def run(self, compute):\n"
                "            with cm():\n"
                "                val = compute()\n"
                "            cb = lambda: val\n"
                "            return cb()\n"
            ),
            "            cb = lambda: val",
            {"strict": False, "pedantic": False},
            id="conditional_class_body_nullcontext_does_not_mask_module_cm",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext as cm\n"
                "from my_suppressors import SuppressAll\n"
                "\n"
                "class Worker:\n"
                "    global cm\n"
                "    cm = SuppressAll()\n"
                "\n"
                "    def run(self, compute):\n"
                "        with cm:\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "        return cb()\n"
            ),
            "        cb = lambda: val",
            {"strict": False, "pedantic": False},
            id="adversarial_class_global_rebind_still_invalidates_module_cm",
        ),
        pytest.param(
            (
                "class FileService:\n"
                "    global open\n"
                "    def open(self):\n"
                "        return 1\n"
                "\n"
                "    def read_items(self, path, compute):\n"
                "        with open(path) as fh:\n"
                "            val = compute(fh)\n"
                "        cb = lambda: val\n"
                "        return cb()\n"
            ),
            "        cb = lambda: val",
            {"strict": False, "pedantic": False},
            id="adversarial_class_global_open_still_shadows_builtin_open",
        ),
        pytest.param(
            (
                "from my_suppressors import SuppressAll as cm\n"
                "\n"
                "class Worker:\n"
                "    from contextlib import nullcontext as cm\n"
                "    with cm():\n"
                "        val = compute()\n"
                "    cb = lambda: val\n"
            ),
            "    cb = lambda: val",
            {"strict": True, "pedantic": True},
            id="control_direct_class_body_block_sees_class_local_nullcontext",
        ),
    ],
)
def test_class_scope_pep227_provenance_isolation(
    module_src: str,
    target_marker: str,
    expected_safety: dict[str, bool],
) -> None:
    """Verifies PEP 227 class-body bindings neither spoof outer context managers nor shadow builtins in sibling methods."""
    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == target_marker
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
                "source_code": target_marker + "\n",
                "closure_strictness": mode,
            },
            tree=tree,
        )
        got[mode] = bool(sc["is_control_flow_safe"])
    assert got == expected_safety

@pytest.mark.parametrize(
    ("cm_body", "use_async", "expected_safe"),
    [
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self, x=patch_cm()):\n"
                "        return x\n"
            ),
            False,
            False,
            id="method_positional_default_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self, *, cb=patch_cm()):\n"
                "        return cb\n"
            ),
            False,
            False,
            id="method_kwonly_default_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a, _=patch_cm()):\n"
                "        return False\n"
            ),
            False,
            False,
            id="exit_method_default_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    async def __aenter__(self):\n"
                "        return self\n"
                "    async def __aexit__(self, *a, _=patch_cm()):\n"
                "        return False\n"
            ),
            True,
            False,
            id="async_aexit_method_default_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self, x: patch_cm()):\n"
                "        return x\n"
            ),
            False,
            False,
            id="method_param_annotation_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self, *args: patch_cm(), **kwargs: int):\n"
                "        return args\n"
            ),
            False,
            False,
            id="method_vararg_annotation_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self) -> patch_cm():\n"
                "        return None\n"
            ),
            False,
            False,
            id="method_return_annotation_call_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(self, x: [patch_cm() for _ in (1,)]):\n"
                "        return x\n"
            ),
            False,
            False,
            id="adversarial_annotation_comprehension_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *a):\n"
                "        return False\n"
                "    def helper(\n"
                "        self,\n"
                "        x: Optional[int] = None,\n"
                "        *args: str,\n"
                "        flag: bool = False,\n"
                "        **kwargs: int,\n"
                "    ) -> bool:\n"
                "        return flag\n"
            ),
            False,
            True,
            id="control_static_defaults_and_call_free_annotations_stay_safe",
        ),
    ],
)
def test_class_body_method_defaults_and_annotations_affect_cm_provenance(
    cm_body: str,
    use_async: bool,
    expected_safe: bool,
) -> None:
    """Verifies eager method defaults and annotations on class context managers invalidate class-body non-suppression proof."""
    def_hdr = "async def run(compute):" if use_async else "def run(compute):"
    with_hdr = "    async with CM():" if use_async else "    with CM():"
    module_src = (
        "def patch_cm():\n"
        "    return 1\n"
        "\n"
        "class CM:\n"
        f"{cm_body}"
        "\n"
        f"{def_hdr}\n"
        f"{with_hdr}\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    target_marker = "    cb = lambda: val"
    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == target_marker
    )
    tree = ast.parse(module_src)
    safety: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": line_num,
                "end": line_num,
                "kind": "basic_block",
                "source_text": module_src,
                "source_code": target_marker + "\n",
                "closure_strictness": mode,
            },
            tree=tree,
        )
        safety[mode] = bool(sc["is_control_flow_safe"])
    assert safety == {"strict": True, "pedantic": expected_safe}

@pytest.mark.parametrize(
    ("try_block_src", "expect_safe", "expect_cm_non_suppressor"),
    [
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        risky()\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
                "        return None\n"
            ),
            True,
            True,
            id="handler_ending_in_return_excluded_from_fallthrough_provenance",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        risky()\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
                "        raise\n"
            ),
            True,
            True,
            id="handler_ending_in_raise_excluded_from_fallthrough_provenance",
        ),
        pytest.param(
            (
                "    try:\n"
                "        cm = suppress(Exception)\n"
                "        return None\n"
                "    except ValueError:\n"
                "        cm = nullcontext()\n"
            ),
            True,
            True,
            id="terminating_try_body_excluded_when_handler_falls_through",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        return None\n"
                "    except ValueError:\n"
                "        pass\n"
                "    else:\n"
                "        cm = suppress(Exception)\n"
            ),
            True,
            True,
            id="unreachable_orelse_ignored_after_terminating_try_body",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        risky()\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
            ),
            False,
            False,
            id="negative_control_fallthrough_handler_invalidates_cm_proof",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        risky()\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
                "        with suppress(RuntimeError):\n"
                "            raise RuntimeError('swallowed')\n"
            ),
            False,
            False,
            id="adversarial_suppressed_raise_does_not_terminate_handler",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        risky()\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
                "        return None\n"
                "    finally:\n"
                "        risky()\n"
            ),
            True,
            True,
            id="handler_ending_in_return_with_pass_through_finally_keeps_proof",
        ),
        pytest.param(
            (
                "    cm = nullcontext()\n"
                "    try:\n"
                "        return None\n"
                "    except ValueError:\n"
                "        cm = suppress(Exception)\n"
                "        raise\n"
            ),
            False,
            False,
            id="control_all_paths_terminating_retains_conservative_fallback",
        ),
    ],
)
def test_update_try_stmt_provenance_skips_terminating_branches(
    try_block_src: str,
    expect_safe: bool,
    expect_cm_non_suppressor: bool,
) -> None:
    """Verifies standalone provenance updates skip terminating try/orelse/handler branches like _analyze_try_stmt_assignment."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

    module_src = (
        "from contextlib import nullcontext, suppress\n"
        "\n"
        "def run(risky, compute):\n"
        f"{try_block_src}"
        "    with cm:\n"
        "        val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    tree = ast.parse(module_src)
    mod_prov = _update_suppress_provenance_from_stmts(
        tree.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    fn_node = tree.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    out_prov = _update_suppress_provenance_from_stmts(
        fn_node.body[:-3], mod_prov, in_branch=False
    )
    assert ("cm" in out_prov.non_suppressor_instances) is expect_cm_non_suppressor, out_prov
    if expect_cm_non_suppressor:
        assert "cm" not in out_prov.suppressors

    target_marker = "    cb = lambda: val"
    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == target_marker
    )
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": line_num,
            "end": line_num,
            "kind": "basic_block",
            "source_text": module_src,
            "source_code": target_marker + "\n",
            "closure_strictness": "pedantic",
        },
        tree=tree,
    )
    assert bool(sc["is_control_flow_safe"]) is expect_safe, sc

@pytest.mark.parametrize(
    ("setup_src", "ctx_line", "expect_safe"),
    [
        pytest.param(
            (
                "    box = {}\n"
                "    cm = contextlib.suppress\n"
                "    box[(cm := contextlib.nullcontext)] = alias = cm\n"
            ),
            "    with alias():\n",
            False,
            id="target_walrus_nullcontext_does_not_poison_rhs_suppress_into_alias",
        ),
        pytest.param(
            (
                "    box = {}\n"
                "    cm = contextlib.suppress\n"
                "    box[(cm := contextlib.nullcontext)] = alias = cm\n"
            ),
            "    with cm():\n",
            True,
            id="target_walrus_nullcontext_still_updates_cm_after_rhs_snapshot",
        ),
        pytest.param(
            (
                "    box = {}\n"
                "    cm = contextlib.nullcontext\n"
                "    box[(cm := contextlib.suppress)] = alias = cm\n"
            ),
            "    with alias():\n",
            True,
            id="control_target_walrus_suppress_does_not_poison_rhs_nullcontext_into_alias",
        ),
        pytest.param(
            (
                "    box = {}\n"
                "    cm = contextlib.nullcontext\n"
                "    box[(cm := contextlib.suppress)] = alias = cm\n"
            ),
            "    with cm():\n",
            False,
            id="control_target_walrus_suppress_marks_cm_suppressing",
        ),
        pytest.param(
            (
                "    box = {}\n"
                "    cm = contextlib.nullcontext\n"
                "    cm = box[(cm := contextlib.suppress)] = contextlib.nullcontext\n"
            ),
            "    with cm():\n",
            False,
            id="target_order_left_to_right_walrus_in_second_target_overrides_first_target_bind",
        ),
        pytest.param(
            (
                "    cm = contextlib.suppress\n"
                "    alias: (cm := contextlib.nullcontext) = cm\n"
            ),
            "    with alias():\n",
            False,
            id="annassign_annotation_walrus_does_not_poison_rhs_suppress",
        ),
    ],
)
def test_assign_and_annassign_target_walrus_does_not_poison_rhs_provenance(
    setup_src: str,
    ctx_line: str,
    expect_safe: bool,
) -> None:
    """Verifies Assign/AnnAssign targets cannot walrus-poison rhs_prov before binding targets."""
    module_src = (
        "import contextlib\n"
        "def run(step):\n"
        + setup_src
        + ctx_line
        + "        v = step()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

@pytest.mark.parametrize(
    ("with_header", "body_line", "expect_safe"),
    [
        pytest.param(
            "        with contextlib.nullcontext(1, 2):\n",
            "            v = step()\n",
            False,
            id="nullcontext_too_many_positional_args_raises_type_error",
        ),
        pytest.param(
            "        with contextlib.nullcontext(1, enter_result=2):\n",
            "            v = step()\n",
            False,
            id="nullcontext_mixed_pos_and_kw_args_raises_type_error",
        ),
        pytest.param(
            "        with contextlib.nullcontext(unexpected=1):\n",
            "            v = step()\n",
            False,
            id="nullcontext_unknown_keyword_raises_type_error",
        ),
        pytest.param(
            "        with contextlib.nullcontext(*items):\n",
            "            v = step()\n",
            False,
            id="nullcontext_starred_args_may_raise_type_error",
        ),
        pytest.param(
            "        with contextlib.nullcontext(**kw):\n",
            "            v = step()\n",
            False,
            id="nullcontext_double_starred_kwargs_may_raise_type_error",
        ),
        pytest.param(
            "        with contextlib.suppress(exc=ValueError):\n",
            "            v = step()\n",
            False,
            id="suppress_keyword_arg_raises_type_error",
        ),
        pytest.param(
            "        with contextlib.nullcontext(enter_result=1):\n",
            "            v = 1\n",
            True,
            id="valid_nullcontext_enter_result_keyword_and_pure_body_is_safe",
        ),
    ],
)
def test_nested_suppress_rejects_invalid_nullcontext_and_suppress_signatures(
    with_header: str,
    body_line: str,
    expect_safe: bool,
) -> None:
    """Verifies _is_non_raising_suppress_or_null_with enforces exact call signatures inside suppress."""
    module_src = (
        "import contextlib\n"
        "def run(step, items, kw):\n"
        "    with contextlib.suppress(ValueError):\n"
        + with_header
        + body_line
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

@pytest.mark.parametrize(
    (
        "hazard_exit_targets",
        "hazard_has_exit_patch",
        "hazard_rebound_targets",
        "expected_non_suppressors",
        "expected_suspicious",
        "expected_has_exit_patch",
    ),
    [
        pytest.param(
            frozenset({"SafeMgr"}),
            True,
            frozenset({(None, None)}),
            frozenset(),
            frozenset({"SafeMgr"}),
            True,
            id="exit_patch_demotes_base_positive_proofs_and_preserves_targets",
        ),
        pytest.param(
            frozenset(),
            False,
            frozenset({("contextlib", "nullcontext")}),
            frozenset({"other_lock"}),
            frozenset(),
            False,
            id="rebound_factory_invalidates_matching_imported_binding_only",
        ),
        pytest.param(
            frozenset(),
            False,
            frozenset(),
            frozenset({"my_null", "other_lock"}),
            frozenset(),
            False,
            id="control_clean_hazard_src_preserves_positive_bindings",
        ),
    ],
)
def test_merge_hazard_provenance_preserves_exit_patch_and_rebound_factories(
    hazard_exit_targets: frozenset[str],
    hazard_has_exit_patch: bool,
    hazard_rebound_targets: frozenset[tuple[str | None, str | None]],
    expected_non_suppressors: frozenset[str],
    expected_suspicious: frozenset[str],
    expected_has_exit_patch: bool,
) -> None:
    """Verifies _merge_hazard_provenance preserves exit_patch_targets, has_exit_patch, and rebound_factory_targets."""
    from pydoppelgangerhunt.fixer._df_block_flow import _merge_hazard_provenance
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

    base = _SuppressProvenance(
        non_suppressors=frozenset({"my_null", "other_lock"}),
        imported_symbols=frozenset(
            {
                ("my_null", "contextlib.nullcontext"),
                ("other_lock", "threading.Lock"),
            }
        ),
    )
    hazard_src = _SuppressProvenance(
        exit_patch_targets=hazard_exit_targets,
        has_exit_patch=hazard_has_exit_patch,
        rebound_factory_targets=hazard_rebound_targets,
    )
    merged = _merge_hazard_provenance(base, hazard_src)
    assert merged.has_exit_patch is expected_has_exit_patch
    assert merged.exit_patch_targets == hazard_exit_targets
    assert merged.rebound_factory_targets == hazard_rebound_targets
    assert merged.non_suppressors == expected_non_suppressors
    assert merged.suspicious_aliases == expected_suspicious

@pytest.mark.parametrize(
    ("loop_src", "var_name", "expect_definite", "expect_conditional"),
    [
        pytest.param(
            "for arr[(x := 1)] in empty_seq:\n    pass\n",
            "x",
            False,
            True,
            id="for_target_walrus_is_conditional_on_zero_iterations",
        ),
        pytest.param(
            "async for arr[(x := 1)] in empty_aiter:\n    pass\n",
            "x",
            False,
            True,
            id="async_for_target_walrus_is_conditional_on_zero_iterations",
        ),
        pytest.param(
            "for _ in (y := [1]):\n    pass\n",
            "y",
            True,
            False,
            id="control_for_iter_walrus_is_definite",
        ),
        pytest.param(
            "async for _ in (y := empty_aiter):\n    pass\n",
            "y",
            True,
            False,
            id="control_async_for_iter_walrus_is_definite",
        ),
    ],
)
def test_for_and_async_for_target_walrus_is_conditional_while_iter_walrus_is_definite(
    loop_src: str,
    var_name: str,
    expect_definite: bool,
    expect_conditional: bool,
) -> None:
    """Verifies For/AsyncFor target walruses are conditional (0-iteration safe) while iter walruses are definite."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    wrapper = f"async def _probe(arr, empty_seq, empty_aiter):\n    " + loop_src.replace(
        "\n", "\n    "
    )
    fn_node = ast.parse(wrapper).body[0]
    assert isinstance(fn_node, ast.AsyncFunctionDef)
    definite, conditional = _analyze_block_assignment(fn_node.body)
    assert (var_name in definite) is expect_definite
    assert (var_name in conditional) is expect_conditional

@pytest.mark.parametrize(
    ("except_body", "unit_marker", "expect_safe"),
    [
        pytest.param(
            "        with cm():\n            v = step()\n        cb = lambda: v\n",
            "        cb = lambda: v",
            False,
            id="handler_observes_intermediate_suppressor_before_restore",
        ),
        pytest.param(
            "        raise\n    with cm():\n        v = step()\n    cb = lambda: v\n",
            "    cb = lambda: v",
            True,
            id="control_post_try_sees_restored_nullcontext_when_handler_raises",
        ),
    ],
)
def test_try_body_single_pass_mid_prov_preserves_intermediate_suppressor_hazards(
    except_body: str,
    unit_marker: str,
    expect_safe: bool,
) -> None:
    """Verifies single-pass try-body provenance retains mid-body suppressor hazards for except handlers while restoring normal completion."""
    module_src = (
        "import contextlib\n"
        "def run(step):\n"
        "    cm = contextlib.nullcontext\n"
        "    try:\n"
        "        step()\n"
        "        cm = contextlib.suppress\n"
        "        step()\n"
        "        cm = contextlib.nullcontext\n"
        "    except Exception:\n"
        + except_body
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, unit_marker)
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

@pytest.mark.parametrize(
    ("loop_lines", "expect_safe"),
    [
        pytest.param(
            (
                "            if i > 0:\n"
                "                with cm(ValueError):\n"
                "                    return compute()\n"
                "                break\n"
                "            cm = contextlib.suppress\n"
            ),
            False,
            id="post_loop_capture_unsafe_when_iteration2_rebinding_enables_break",
        ),
        pytest.param(
            (
                "            cm = contextlib.nullcontext\n"
                "            if i > 0:\n"
                "                with cm():\n"
                "                    return compute()\n"
                "                break\n"
                "            cm = contextlib.suppress\n"
            ),
            True,
            id="control_post_loop_capture_safe_when_cm_restored_before_with",
        ),
    ],
)
def test_loop_carried_ctx_manager_rebinding_end_to_end_scope_safety(
    loop_lines: str,
    expect_safe: bool,
) -> None:
    """Verifies end-to-end scope safety rejects post-loop late-bound capture when loop-carried rebinding makes break reachable."""
    module_src = (
        "import contextlib\n"
        "def run(cond, items, compute):\n"
        "    if cond:\n"
        "        cm = contextlib.nullcontext\n"
        "        for i in items:\n"
        + loop_lines
        + "        else:\n"
        "            return 0\n"
        "    else:\n"
        "        v = 1\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

@pytest.mark.parametrize(
    ("del_stmt_src", "expect_non_raising"),
    [
        pytest.param("del x, x\n", False, id="duplicate_delete_target_pair"),
        pytest.param("del x, y, x\n", False, id="duplicate_delete_target_non_adjacent"),
        pytest.param("del x, y\n", True, id="control_distinct_delete_targets"),
    ],
)
def test_duplicate_delete_target_in_suppress_is_not_non_raising(
    del_stmt_src: str,
    expect_non_raising: bool,
) -> None:
    """Verifies duplicate ast.Delete targets like 'del x, x' are treated as raising inside suppress blocks."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment
    from pydoppelgangerhunt.fixer._df_provenance_walk import _stmt_cannot_raise_in_suppress

    del_stmt = ast.parse(del_stmt_src).body[0]
    assert (
        _stmt_cannot_raise_in_suppress(del_stmt, safe_names=frozenset({"x", "y"}))
        is expect_non_raising
    )
    block_src = (
        "from contextlib import suppress\n"
        "x = 1\n"
        "y = 2\n"
        "with suppress(UnboundLocalError):\n"
        f"    {del_stmt_src}"
        "    v = 99\n"
    )
    definite, conditional = _analyze_block_assignment(ast.parse(block_src).body)
    assert ("v" in definite) is expect_non_raising
    assert ("v" in conditional) is (not expect_non_raising)

@pytest.mark.parametrize(
    ("block_src", "expect_cm_proven", "expect_v_definite"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "try:\n"
                "    raise ValueError()\n"
                "except ValueError as cm:\n"
                "    cm = contextlib.nullcontext\n"
                "with cm():\n"
                "    v = compute()\n"
            ),
            False,
            False,
            id="handler_rebound_as_name_cleared_on_exit_when_try_terminates",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext\n"
                "try:\n"
                "    may_raise()\n"
                "except ValueError as cm:\n"
                "    cm = contextlib.nullcontext\n"
                "with cm():\n"
                "    v = compute()\n"
            ),
            False,
            False,
            id="handler_rebound_as_name_clears_pre_try_proof_on_exit",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext\n"
                "try:\n"
                "    may_raise()\n"
                "except ValueError as cm:\n"
                "    cm = contextlib.nullcontext\n"
                "finally:\n"
                "    with cm():\n"
                "        v = compute()\n"
            ),
            False,
            False,
            id="handler_as_name_cleared_before_finally_on_normal_handler_exit",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext\n"
                "for _ in items:\n"
                "    try:\n"
                "        may_raise()\n"
                "    except ValueError as cm:\n"
                "        cm = contextlib.nullcontext\n"
                "        raise\n"
                "    finally:\n"
                "        with cm():\n"
                "            v = compute()\n"
                "        break\n"
            ),
            False,
            False,
            id="handler_as_name_cleared_before_finally_when_finally_suppresses_via_break",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext\n"
                "try:\n"
                "    may_raise()\n"
                "except ValueError as cm:\n"
                "    cm = contextlib.nullcontext\n"
                "    raise\n"
                "finally:\n"
                "    pass\n"
                "with cm():\n"
                "    v = compute()\n"
            ),
            True,
            True,
            id="control_reraising_handler_does_not_affect_post_try_finally_fallthrough",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext\n"
                "try:\n"
                "    may_raise()\n"
                "except ValueError as err:\n"
                "    err = contextlib.suppress\n"
                "with cm():\n"
                "    v = compute()\n"
            ),
            True,
            True,
            id="control_unrelated_handler_as_name_preserves_cm_proof",
        ),
    ],
)
def test_except_handler_name_cleared_from_provenance_on_handler_exit(
    block_src: str,
    expect_cm_proven: bool,
    expect_v_definite: bool,
) -> None:
    """Verifies ExceptHandler.name is cleared from suppression provenance after handler exit in both provenance and flow walks."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment_impl,
    )
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    stmts = ast.parse(block_src).body
    prov_walk = _update_suppress_provenance_from_stmts(
        stmts,
        _EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
        strict_with_suppression=True,
    )
    assert ("cm" in prov_walk.non_suppressors) is expect_cm_proven

    flow_res = _analyze_block_assignment_impl(
        stmts,
        strict_with_suppression=True,
    )
    assert ("cm" in flow_res.prov.non_suppressors) is expect_cm_proven
    assert ("v" in flow_res.definite) is expect_v_definite

@pytest.mark.parametrize(
    ("src", "expect_terminates"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "with contextlib.suppress(NameError):\n"
                "    with suppress(KeyError):\n"
                "        return 1\n"
            ),
            False,
            id="unimported_bare_suppress_may_raise_name_error",
        ),
        pytest.param(
            (
                "from contextlib import suppress as outer_sup\n"
                "with outer_sup(NameError):\n"
                "    with contextlib.suppress(KeyError):\n"
                "        return 1\n"
            ),
            False,
            id="unimported_contextlib_attr_may_raise_name_error",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "from custom_pkg import suppress\n"
                "with contextlib.suppress(RuntimeError):\n"
                "    with suppress(KeyError):\n"
                "        return 1\n"
            ),
            False,
            id="custom_imported_suppress_is_not_assumed_non_raising",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "with suppress(ValueError):\n"
                "    with suppress(KeyError):\n"
                "        return 1\n"
            ),
            True,
            id="control_imported_contextlib_suppress_proves_termination",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "with contextlib.suppress(ValueError):\n"
                "    with contextlib.suppress(KeyError):\n"
                "        return 1\n"
            ),
            True,
            id="control_imported_contextlib_module_suppress_proves_termination",
        ),
    ],
)
def test_nested_suppress_requires_proven_contextlib_import_for_non_raising_proof(
    src: str,
    expect_terminates: bool,
) -> None:
    """Verifies nested with suppress(...) non-raising proof requires proven contextlib provenance instead of assuming bare or unimported names."""
    from pydoppelgangerhunt.fixer._df_block_flow import _block_terminates

    assert _block_terminates(ast.parse(src).body) is expect_terminates

@pytest.mark.parametrize(
    ("fn_src", "mode_expected_definite"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer():\n"
                "    with suppress(NameError):\n"
                "        x: (w := 1) = 2\n"
                "        y = w\n"
                "    return y\n"
            ),
            False,
            id="annotation_walrus_does_not_mark_var_safe_in_suppress_block",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def outer(risky):\n"
                "    cm = suppress\n"
                "    x: (cm := nullcontext) = 1\n"
                "    with cm(ValueError):\n"
                "        y = risky()\n"
                "    return y\n"
            ),
            False,
            id="annotation_walrus_does_not_promote_suppressor_to_nullcontext",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer():\n"
                "    with suppress(NameError):\n"
                "        x: None = (w := 1)\n"
                "        y = w\n"
                "    return y\n"
            ),
            True,
            id="control_rhs_value_walrus_marks_var_safe_in_suppress_block",
        ),
    ],
)
def test_annassign_annotation_walrus_neither_grants_non_raising_safe_nor_positive_provenance(
    fn_src: str,
    mode_expected_definite: bool,
) -> None:
    """Verifies AnnAssign.annotation walruses neither seed cur_safe nor grant positive non-suppression provenance."""
    lines = fn_src.splitlines()
    start_ln = next(i for i, ln in enumerate(lines, start=1) if ln.startswith("def outer")) + 1
    end_ln = next(i for i, ln in enumerate(lines, start=1) if ln.strip().startswith("return ")) - 1
    body_code = "\n".join(lines[start_ln - 1 : end_ln]) + "\n"
    tree = ast.parse(fn_src)
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": start_ln,
                "end": end_ln,
                "kind": "compound_block",
                "source_text": fn_src,
                "source_code": body_code,
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert ("y" in sc["definite_stores"]) is mode_expected_definite, (mode, sc)

@pytest.mark.parametrize(
    ("module_src", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "def build(flag):\n"
                "    if flag:\n"
                "        import evil as contextlib\n"
                "    class Mgr(contextlib.AbstractContextManager):\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="branch_import_evil_as_contextlib_invalidates_base",
        ),
        pytest.param(
            (
                "def build(flag):\n"
                "    if flag:\n"
                "        from evil import property\n"
                "    class Mgr:\n"
                "        @property\n"
                "        def label(self):\n"
                "            return 1\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="branch_import_evil_property_invalidates_decorator",
        ),
        pytest.param(
            (
                "def build(flag):\n"
                "    if flag:\n"
                "        from evil import object\n"
                "    class Mgr(object):\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="branch_import_evil_object_invalidates_base",
        ),
        pytest.param(
            (
                "def build(flag):\n"
                "    if flag:\n"
                "        from evil import AbstractContextManager\n"
                "    else:\n"
                "        from contextlib import AbstractContextManager\n"
                "    class Mgr(AbstractContextManager):\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="branch_divergent_abstract_context_manager_invalidates_base",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def build():\n"
                "    property = contextlib.nullcontext\n"
                "    class Mgr:\n"
                "        @property\n"
                "        def label(self):\n"
                "            return 1\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="positive_non_suppressor_rebind_of_property_invalidates_decorator",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def build():\n"
                "    object = contextlib.nullcontext\n"
                "    class Mgr(object):\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="positive_non_suppressor_rebind_of_object_invalidates_base",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def build():\n"
                "    contextlib = contextlib.nullcontext\n"
                "    class Mgr(contextlib.AbstractContextManager):\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="positive_non_suppressor_rebind_of_contextlib_invalidates_base",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def build(flag):\n"
                "    if flag:\n"
                "        other = 1\n"
                "    class Mgr(contextlib.AbstractContextManager):\n"
                "        @property\n"
                "        def label(self):\n"
                "            return 1\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_unshadowed_contextlib_base_and_property_stay_proven",
        ),
    ],
)
def test_branch_divergent_and_positive_binding_shadowing_blocks_cm_proofs(
    module_src: str,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies branch-divergent imports and positive non-suppressor bindings shadow builtin decorators, bases, and implicit stdlib modules."""
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe, safety

@pytest.mark.parametrize(
    ("module_src", "expect_strict", "expect_pedantic"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm, other = suppress, 1\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="tuple_unpack_direct_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    [cm, other] = [suppress, 1]\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="list_unpack_direct_suppress",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def f(d):\n"
                "    (a, (cm, b)) = (1, (contextlib.suppress, 2))\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="nested_tuple_unpack_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm1, x = cm2, y = suppress, 1\n"
                "    with cm2(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="chained_tuple_unpack_suppress",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def f(d):\n"
                "    a, b = suppress, nullcontext\n"
                "    a, b = b, a\n"
                "    with b(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="simultaneous_swap_tracks_swapped_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def f(d):\n"
                "    a, b = suppress, nullcontext\n"
                "    a, b = b, a\n"
                "    with a():\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_simultaneous_swap_preserves_swapped_nullcontext",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm, *rest = [suppress, 1, 2]\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="starred_target_prefix_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    *head, cm = [1, 2, suppress]\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="starred_target_suffix_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    head, *rest = [1, suppress]\n"
                "    cm, = rest\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="starred_target_middle_propagates_suspicious_alias",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm, other = [suppress] * 2\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="fallback_repeated_list_carrying_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm, other = (*[suppress], 1)\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="fallback_starred_rhs_carrying_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    cm, other = [suppress for _ in range(2)]\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="fallback_listcomp_carrying_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d, make_pair):\n"
                "    cm, other = make_pair(suppress)\n"
                "    with cm(KeyError):\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="fallback_call_arg_carrying_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(d):\n"
                "    for cm, other in [(suppress, 1)]:\n"
                "        with cm(KeyError):\n"
                "            v = d['k']\n"
                "        cb = lambda: v\n"
                "        return cb()\n"
            ),
            False,
            False,
            id="for_loop_tuple_unpack_carrying_suppress",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def f(d):\n"
                "    cm, other = nullcontext, 1\n"
                "    with cm():\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_tuple_unpack_preserves_nullcontext_proof",
        ),
        pytest.param(
            (
                "def f(d, make_pair):\n"
                "    cm, other = make_pair(suppress_errors=False)\n"
                "    with cm():\n"
                "        v = d['k']\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            False,
            id="control_benign_kwarg_named_suppress_does_not_taint_strict_unpack",
        ),
    ],
)
def test_tuple_and_list_unpacking_preserves_suppressor_and_non_suppressor_provenance(
    module_src: str,
    expect_strict: bool,
    expect_pedantic: bool,
) -> None:
    """Verifies tuple/list, nested, starred, and loop unpacking preserve suppressor and non-suppressor provenance."""
    target_line = next(
        ln for ln in module_src.splitlines() if ln.strip() == "cb = lambda: v"
    )
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(module_src, target_line)
    assert safety["strict"] is expect_strict, safety
    assert safety["pedantic"] is expect_pedantic, safety

@pytest.mark.parametrize(
    ("module_src", "target_fn_name", "expect_terminates"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(cond):\n"
                "    if cond:\n"
                "        ValueError = TypeError\n"
                "    with suppress(UnboundLocalError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            False,
            id="conditionally_bound_exception_name_can_raise_unbound_local",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    ValueError = TypeError\n"
                "    del ValueError\n"
                "    with suppress(UnboundLocalError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            False,
            id="deleted_local_exception_name_can_raise_unbound_local",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    with suppress(UnboundLocalError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
                "    ValueError = TypeError\n"
            ),
            "f",
            False,
            id="forward_local_binding_of_exception_name_can_raise_unbound_local",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    ValueError = TypeError\n"
                "    with suppress(UnboundLocalError):\n"
                "        del ValueError\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            False,
            id="del_inside_outer_suppress_invalidates_nested_exception_name",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer(cond):\n"
                "    if cond:\n"
                "        ValueError = TypeError\n"
                "    def inner():\n"
                "        with suppress(NameError):\n"
                "            with suppress(ValueError):\n"
                "                return 1\n"
            ),
            "inner",
            False,
            id="conditionally_bound_enclosing_freevar_exception_name_can_raise",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "from other_mod import *\n"
                "def f():\n"
                "    with suppress(NameError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            False,
            id="star_import_prevents_unshadowed_builtin_exception_assumption",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    with suppress(TypeError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            True,
            id="control_unshadowed_exception_name_proves_termination",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    ValueError = TypeError\n"
                "    with suppress(UnboundLocalError):\n"
                "        with suppress(ValueError):\n"
                "            return 1\n"
            ),
            "f",
            True,
            id="control_definitely_bound_shadowed_exception_name_proves_termination",
        ),
    ],
)
def test_nested_suppress_non_raising_proof_respects_shadowed_and_unbound_exception_names(
    module_src: str,
    target_fn_name: str,
    expect_terminates: bool,
) -> None:
    """Verifies _stmt_cannot_raise_in_suppress does not assume shadowed, deleted, or unbound exception names cannot raise."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment_impl
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    mod = ast.parse(module_src)
    fn_node = next(
        n
        for n in ast.walk(mod)
        if isinstance(n, ast.FunctionDef) and n.name == target_fn_name
    )
    clear_dataflow_cache()
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree=mod)
    param_names = {a.arg for a in fn_node.args.args}
    term = _analyze_block_assignment_impl(
        fn_node.body,
        definite=set(param_names),
        suppress_aliases=prov,
    ).terminated
    assert term is expect_terminates

@pytest.mark.parametrize(
    ("module_src", "expect_terminates", "expect_x_definite"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(cond):\n"
                "    if cond:\n"
                "        cm = suppress\n"
                "    with suppress(UnboundLocalError):\n"
                "        x = cm\n"
                "        return 1\n"
            ),
            False,
            False,
            id="conditional_suppress_alias_not_assumed_safe_inside_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f(cond):\n"
                "    if cond:\n"
                "        from contextlib import suppress as cm\n"
                "    with suppress(UnboundLocalError):\n"
                "        x = cm\n"
                "        return 1\n"
            ),
            False,
            False,
            id="conditional_suppress_import_not_assumed_safe_inside_suppress",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def f():\n"
                "    cm = suppress\n"
                "    with suppress(UnboundLocalError):\n"
                "        x = cm\n"
                "        return 1\n"
            ),
            True,
            True,
            id="control_unconditional_suppress_alias_remains_safe_inside_suppress",
        ),
    ],
)
def test_conditional_suppressor_alias_is_not_assumed_bound_inside_suppress_body(
    module_src: str,
    expect_terminates: bool,
    expect_x_definite: bool,
) -> None:
    """Verifies may-bound entries in suppressors are not unioned into cur_safe when proving non-raising prefixes."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment_impl
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    mod = ast.parse(module_src)
    fn_node = next(
        n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "f"
    )
    clear_dataflow_cache()
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree=mod)
    param_names = {a.arg for a in fn_node.args.args}
    flow_res = _analyze_block_assignment_impl(
        fn_node.body,
        definite=set(param_names),
        suppress_aliases=prov,
    )
    assert flow_res.terminated is expect_terminates
    assert ("x" in flow_res.definite) is expect_x_definite

@pytest.mark.parametrize(
    ("module_src", "expect_strict", "expect_pedantic"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    cm2 = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm(ValueError):\n"
                "                compute()\n"
                "                return 0\n"
                "            cm2 = contextlib.suppress\n"
                "        cm = contextlib.suppress\n"
                "    with cm2(ValueError):\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="for_loop_carried_rebind_propagates_across_second_iteration",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(flag, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    cm2 = contextlib.nullcontext\n"
                "    while flag():\n"
                "        if flag():\n"
                "            with cm(ValueError):\n"
                "                compute()\n"
                "                return 0\n"
                "            cm2 = contextlib.suppress\n"
                "        cm = contextlib.suppress\n"
                "    with cm2(ValueError):\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="while_loop_carried_rebind_propagates_across_second_iteration",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    cm2 = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm(ValueError):\n"
                "                compute()\n"
                "                return 0\n"
                "            cm2 = contextlib.suppress\n"
                "        with cm2(ValueError):\n"
                "            v = compute()\n"
                "        cb = lambda: v\n"
                "        cm = contextlib.suppress\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="enclosing_loop_carried_rebind_propagates_to_unit_inside_loop",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    cm2 = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm():\n"
                "                compute()\n"
                "                return 0\n"
                "            cm2 = contextlib.suppress\n"
                "    with cm2():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_non_rebound_loop_manager_remains_non_suppressing",
        ),
    ],
)
def test_loop_carried_context_manager_rebind_propagates_across_iterations(
    module_src: str,
    expect_strict: bool,
    expect_pedantic: bool,
) -> None:
    """Verifies loop-body provenance iterates to a fixed point so multi-iteration rebinds are never missed."""
    target_line = next(
        ln for ln in module_src.splitlines() if ln.strip() == "cb = lambda: v"
    )
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(module_src, target_line)
    assert safety["strict"] is expect_strict, safety
    assert safety["pedantic"] is expect_pedantic, safety

@pytest.mark.parametrize(
    ("body_src", "expect_safe"),
    [
        pytest.param(
            (
                "    with suppress(UnboundLocalError):\n"
                "        x: (w := 1) = w\n"
                "        y = 1\n"
            ),
            False,
            id="annotation_walrus_skipped_in_fn_does_not_prove_rhs_w_safe",
        ),
        pytest.param(
            (
                "    w = 0\n"
                "    with suppress(UnboundLocalError):\n"
                "        x: (w := 1) = w\n"
                "        y = 1\n"
            ),
            True,
            id="control_prebound_w_with_annotation_walrus_is_safe",
        ),
        pytest.param(
            (
                "    with suppress(UnboundLocalError):\n"
                "        x: 'int' = 1\n"
                "        y = 1\n"
            ),
            True,
            id="control_plain_non_raising_annassign_is_safe",
        ),
    ],
)
def test_suppress_annassign_annotation_walrus_does_not_mark_rhs_safe(
    body_src: str,
    expect_safe: bool,
) -> None:
    """Verifies AnnAssign.annotation walrus expressions do not advance safe_names for AnnAssign.value inside suppress()."""
    src = (
        "from contextlib import suppress\n"
        "def run():\n"
        + body_src
        + "    cb = lambda: y\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: y")
    assert safety["strict"] is expect_safe, safety
    assert safety["pedantic"] is expect_safe, safety

@pytest.mark.parametrize(
    ("guard_block", "expect_safe"),
    [
        pytest.param(
            "    if cond:\n"
            "        cm = suppress\n"
            "        return 0\n"
            "    with cm():\n"
            "        v = compute()\n",
            True,
            id="terminating_if_body_omitted_from_fallthrough_prov",
        ),
        pytest.param(
            "    if cond:\n"
            "        pass\n"
            "    else:\n"
            "        cm = suppress\n"
            "        raise RuntimeError()\n"
            "    with cm():\n"
            "        v = compute()\n",
            True,
            id="terminating_else_body_omitted_from_fallthrough_prov",
        ),
        pytest.param(
            "    if (cm := suppress) and cond:\n"
            "        return 0\n"
            "    with cm():\n"
            "        v = compute()\n",
            False,
            id="adversarial_if_test_walrus_still_taints_fallthrough",
        ),
        pytest.param(
            "    try:\n"
            "        if cond:\n"
            "            cm = suppress\n"
            "            raise ValueError()\n"
            "    except ValueError:\n"
            "        with cm():\n"
            "            v = compute()\n",
            False,
            id="adversarial_raising_if_arm_still_reaches_except_via_mid_prov",
        ),
        pytest.param(
            "    match cond:\n"
            "        case 1:\n"
            "            cm = suppress\n"
            "            return 0\n"
            "        case _:\n"
            "            pass\n"
            "    with cm():\n"
            "        v = compute()\n",
            True,
            id="terminating_match_case_omitted_with_wildcard_survivor",
        ),
        pytest.param(
            "    match cond:\n"
            "        case 1:\n"
            "            cm = suppress\n"
            "            return 0\n"
            "    with cm():\n"
            "        v = compute()\n",
            True,
            id="terminating_match_case_omitted_with_implicit_fallthrough",
        ),
        pytest.param(
            "    match cond:\n"
            "        case y if (cm := suppress) and False:\n"
            "            return 0\n"
            "        case _:\n"
            "            pass\n"
            "    with cm():\n"
            "        v = compute()\n",
            False,
            id="adversarial_match_guard_walrus_still_taints_subsequent_cases",
        ),
    ],
)
def test_terminating_if_and_match_arms_omitted_from_fallthrough_provenance(
    guard_block: str,
    expect_safe: bool,
) -> None:
    """Verifies terminating if/match arms do not poison fallthrough provenance while preserving test/guard/mid-block hazards."""
    src = (
        "from contextlib import nullcontext, suppress\n"
        "def run(cond, compute):\n"
        "    cm = nullcontext\n"
        + guard_block
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["strict"] is expect_safe, safety
    assert safety["pedantic"] is expect_safe, safety

@pytest.mark.parametrize(
    ("src", "marker", "expect_strict_safe", "expect_pedantic_safe"),
    [
        pytest.param(
            "from contextlib import suppress\n"
            "def outer(step):\n"
            "    cm = suppress\n"
            "    def inner():\n"
            "        nonlocal cm\n"
            "        with cm(ValueError):\n"
            "            v = step()\n"
            "        cb = lambda: v\n"
            "        return cb()\n"
            "    return inner()\n",
            "        cb = lambda: v",
            False,
            False,
            id="nonlocal_declaration_preserves_enclosing_suppressor",
        ),
        pytest.param(
            "from contextlib import suppress\n"
            "cm = suppress\n"
            "def inner(step):\n"
            "    global cm\n"
            "    with cm(ValueError):\n"
            "        v = step()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            "    cb = lambda: v",
            False,
            False,
            id="global_declaration_preserves_module_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext\n"
            "def outer(step):\n"
            "    cm = nullcontext\n"
            "    def inner():\n"
            "        nonlocal cm\n"
            "        with cm():\n"
            "            v = step()\n"
            "        cb = lambda: v\n"
            "        return cb()\n"
            "    return inner()\n",
            "        cb = lambda: v",
            True,
            False,
            id="nonlocal_declaration_invalidates_positive_non_suppressor_in_pedantic",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def outer(step):\n"
            "    cm = suppress\n"
            "    def inner():\n"
            "        nonlocal cm\n"
            "        cm = nullcontext\n"
            "        with cm():\n"
            "            v = step()\n"
            "        cb = lambda: v\n"
            "        return cb()\n"
            "    return inner()\n",
            "        cb = lambda: v",
            True,
            True,
            id="control_nonlocal_explicitly_rebound_to_nullcontext_is_safe",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def outer(step):\n"
            "    cm = suppress\n"
            "    def inner():\n"
            "        cm = nullcontext\n"
            "        with cm():\n"
            "            v = step()\n"
            "        cb = lambda: v\n"
            "        return cb()\n"
            "    return inner()\n",
            "        cb = lambda: v",
            True,
            True,
            id="control_true_local_shadow_overrides_enclosing_suppressor",
        ),
    ],
)
def test_nonlocal_and_global_declarations_preserve_enclosing_suppressor_provenance(
    src: str,
    marker: str,
    expect_strict_safe: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies nonlocal/global declarations invalidate positive proofs without erasing enclosing suppressors."""
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(src, marker)
    assert safety["strict"] is expect_strict_safe, safety
    assert safety["pedantic"] is expect_pedantic_safe, safety

@pytest.mark.parametrize(
    ("first_handler_body", "expect_second_handler_terminates", "expect_post_cm_safe"),
    [
        pytest.param(
            "        cm = suppress\n",
            False,
            False,
            id="trystar_first_handler_suppress_mutation_propagates_to_second_handler",
        ),
        pytest.param(
            "        cm = suppress\n        return 0\n",
            True,
            True,
            id="control_trystar_terminating_first_handler_cannot_reach_second_handler",
        ),
        pytest.param(
            "        pass\n",
            True,
            True,
            id="control_trystar_benign_first_handler_preserves_nullcontext",
        ),
    ],
)
def test_trystar_sequential_handlers_propagate_provenance_mutations(
    first_handler_body: str,
    expect_second_handler_terminates: bool,
    expect_post_cm_safe: bool,
) -> None:
    """Verifies sequential Python 3.11+ except* handlers propagate non-terminating provenance mutations."""
    if sys.version_info < (3, 11):
        pytest.skip("Requires Python 3.11+ for except* (TryStar)")
    from pydoppelgangerhunt.fixer._df_block_flow import (  # pylint: disable=import-outside-toplevel
        _analyze_block_assignment,
    )
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=import-outside-toplevel
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    src = (
        "from contextlib import nullcontext, suppress\n"
        "def run(risky, compute):\n"
        "    cm = nullcontext\n"
        "    try:\n"
        "        risky()\n"
        "    except* ValueError:\n"
        + first_handler_body
        + "    except* TypeError:\n"
        "        with cm(KeyError):\n"
        "            x = compute()\n"
        "            return x\n"
        "        fell_through = 1\n"
        "    return 0\n"
    )
    mod = ast.parse(src)
    fn_node = mod.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    seed_prov = _update_suppress_provenance_from_stmts(
        mod.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    definite, conditional = _analyze_block_assignment(
        fn_node.body[:-1],
        suppress_aliases=seed_prov,
        strict_with_suppression=True,
    )
    # When cm is non-suppressing in the second except* handler, `with cm(...): return x`
    # terminates the handler so `fell_through = 1` is unreachable; when cm was mutated to
    # suppress by a non-terminating earlier except* handler, `fell_through` becomes reachable.
    assert ("fell_through" not in (definite | conditional)) is expect_second_handler_terminates, (
        definite,
        conditional,
    )

    # Also verify _update_suppress_provenance_from_stmts propagates cm=suppress into the
    # second except* handler (suppressing post-with provenance for a binder inside it).
    prov_src = (
        "from contextlib import nullcontext, suppress\n"
        "def run(risky, compute):\n"
        "    cm = nullcontext\n"
        "    try:\n"
        "        risky()\n"
        "    except* ValueError:\n"
        + first_handler_body
        + "    except* TypeError:\n"
        "        with cm(KeyError):\n"
        "            return compute()\n"
        "        tainted_after = suppress\n"
    )
    prov_mod = ast.parse(prov_src)
    prov_fn = prov_mod.body[1]
    assert isinstance(prov_fn, ast.FunctionDef)
    post_prov = _update_suppress_provenance_from_stmts(
        prov_fn.body,
        seed_prov,
        in_branch=False,
        strict_with_suppression=True,
    )
    assert ("tainted_after" not in post_prov.suppressors) is expect_second_handler_terminates, post_prov
    assert (
        "cm" in post_prov.non_suppressors and "cm" not in post_prov.suppressors
    ) is expect_post_cm_safe, post_prov

@pytest.mark.parametrize(
    ("handlers_src", "expect_post_cm_safe"),
    [
        pytest.param(
            "    except* ValueError:\n"
            "        cm = suppress\n"
            "        raise\n",
            True,
            id="single_terminating_trystar_handler_preserves_clean_fallthrough",
        ),
        pytest.param(
            "    except* ValueError:\n"
            "        cm = suppress\n"
            "        raise RuntimeError('v')\n"
            "    except* TypeError:\n"
            "        cm = suppress\n"
            "        raise RuntimeError('t')\n",
            True,
            id="multiple_all_terminating_trystar_handlers_preserve_clean_fallthrough",
        ),
        pytest.param(
            "    except* ValueError:\n"
            "        cm = suppress\n",
            False,
            id="adversarial_non_terminating_trystar_handler_taints_fallthrough",
        ),
        pytest.param(
            "    except* ValueError:\n"
            "        cm = suppress\n"
            "        raise\n"
            "    finally:\n"
            "        final_cm = cm\n"
            "        if flag:\n"
            "            return final_cm\n",
            False,
            id="adversarial_trystar_finally_suppressing_exception_observes_handler_taint",
        ),
    ],
)
def test_trystar_all_terminating_handlers_do_not_leak_suppressor_into_fallthrough(
    handlers_src: str,
    expect_post_cm_safe: bool,
) -> None:
    """Verifies TryStar fallthrough keeps clean try provenance when all except* handlers terminate."""
    if sys.version_info < (3, 11):
        pytest.skip("Requires Python 3.11+ for except* (TryStar)")
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=import-outside-toplevel
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    src = (
        "from contextlib import nullcontext, suppress\n"
        "def run(risky, flag):\n"
        "    cm = nullcontext\n"
        "    try:\n"
        "        risky()\n"
        + handlers_src
    )
    mod = ast.parse(src)
    fn_node = mod.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    seed_prov = _update_suppress_provenance_from_stmts(
        mod.body[:1], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    post_prov = _update_suppress_provenance_from_stmts(
        fn_node.body,
        seed_prov,
        in_branch=False,
        strict_with_suppression=True,
    )
    target_var = "final_cm" if "final_cm" in handlers_src else "cm"
    is_safe = (
        target_var in post_prov.non_suppressors
        and target_var not in post_prov.suppressors
    )
    assert is_safe is expect_post_cm_safe, post_prov

@pytest.mark.parametrize(
    ("cm_init_before_try", "expect_safe"),
    [
        pytest.param(
            False,
            False,
            id="abrupt_entry_suppressor_reaches_finally_taint_observed_by_outer_except",
        ),
        pytest.param(
            True,
            True,
            id="control_non_suppressor_before_try_keeps_finally_taint_unreachable",
        ),
    ],
)
def test_try_finally_abrupt_entry_preserves_mid_prov_for_outer_except(
    cm_init_before_try: bool,
    expect_safe: bool,
) -> None:
    """Verifies finally-body provenance reachable on unhandled exceptions propagates into enclosing except handlers."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _update_prov_and_term_from_stmt,
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=import-outside-toplevel
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    if cm_init_before_try:
        cm_seed = "    cm = nullcontext\n"
        inner_body = "            risky()\n"
    else:
        cm_seed = "    cm = suppress\n"
        inner_body = "            risky()\n            cm = nullcontext\n"
    src = (
        "from contextlib import nullcontext, suppress\n"
        "\n"
        "def outer(risky, step):\n"
        + cm_seed
        + "    alias = nullcontext\n"
        "    try:\n"
        "        try:\n"
        + inner_body
        + "        finally:\n"
        "            with cm(ValueError):\n"
        "                raise ValueError()\n"
        "            alias = suppress\n"
        "    except Exception:\n"
        "        with alias(ValueError):\n"
        "            v = step()\n"
        "        cb = lambda: v\n"
        "        return cb()\n"
    )
    tree = ast.parse(src)
    fn_node = tree.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    outer_try = fn_node.body[2]
    assert isinstance(outer_try, ast.Try)
    inner_try = outer_try.body[0]
    assert isinstance(inner_try, ast.Try)
    pre_inner = _update_suppress_provenance_from_stmts(
        [*tree.body[:1], *fn_node.body[:2]],
        _EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
    )
    inner_res = _update_prov_and_term_from_stmt(
        inner_try, pre_inner, in_branch=False, strict_with_suppression=True
    )
    assert ("alias" in inner_res.mid_prov.suppressors) is (not expect_safe), inner_res

    cb_line = next(
        idx for idx, ln in enumerate(src.splitlines(), start=1) if "cb = lambda: v" in ln
    )
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "name": "outer:cb",
                "kind": "sliding_window",
                "start": cb_line,
                "end": cb_line,
                "source_text": src,
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert sc["is_control_flow_safe"] is expect_safe, (mode, sc)
        assert ("severed_closure_cell" in sc["control_flow_hazards"]) is (
            not expect_safe
        ), (mode, sc)

@pytest.mark.parametrize(
    ("src", "expect_safe"),
    [
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def outer(cm, step):\n"
                "    other = nullcontext\n"
                "    def mutate():\n"
                "        nonlocal other\n"
                "        other = suppress\n"
                "    with cm, other(ValueError):\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="unproven_bare_with_cm_enter_invalidates_subsequent_with_item",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def outer(cm, step):\n"
                "    other = nullcontext\n"
                "    def mutate():\n"
                "        nonlocal other\n"
                "        other = suppress\n"
                "    with cm:\n"
                "        other = nullcontext\n"
                "    with other(ValueError):\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="unproven_bare_with_cm_exit_invalidates_reestablished_manager",
        ),
        pytest.param(
            (
                "import threading\n"
                "from contextlib import nullcontext, suppress\n"
                "def outer(step):\n"
                "    lk = threading.Lock()\n"
                "    other = nullcontext\n"
                "    def mutate():\n"
                "        nonlocal other\n"
                "        other = suppress\n"
                "    with lk, other(ValueError):\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_proven_stdlib_lock_instance_does_not_invalidate_other",
        ),
    ],
)
def test_pre_instantiated_with_cm_triggers_cross_scope_mutator_invalidation(
    src: str,
    expect_safe: bool,
) -> None:
    """Verifies bare pre-instantiated unproven `with cm:` invalidates cross-scope targets on enter and exit."""
    tree = ast.parse(src)
    cb_line = next(
        idx for idx, ln in enumerate(src.splitlines(), start=1) if "cb = lambda: v" in ln
    )
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "name": "outer:cb",
                "kind": "sliding_window",
                "start": cb_line,
                "end": cb_line,
                "source_text": src,
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert sc["is_control_flow_safe"] is expect_safe, (mode, sc)
        assert ("severed_closure_cell" in sc["control_flow_hazards"]) is (
            not expect_safe
        ), (mode, sc)

@pytest.mark.parametrize(
    ("with_src", "extra_allowlist", "expect_wildcard_calls"),
    [
        pytest.param(
            "with open('f') as fh:\n    x = 1\n",
            frozenset(),
            0,
            id="proven_open_scans_header_once",
        ),
        pytest.param(
            "with db.atomic():\n    x = 1\n",
            frozenset({"*.atomic"}),
            1,
            id="wildcard_allowlisted_scans_header_once_and_records_once",
        ),
        pytest.param(
            "with CustomTx():\n    x = 1\n",
            frozenset(),
            0,
            id="unproven_ctx_manager_scans_header_once",
        ),
    ],
)
def test_update_with_stmt_provenance_scans_header_once_without_duplicate_wildcard_hits(
    monkeypatch: pytest.MonkeyPatch,
    with_src: str,
    extra_allowlist: frozenset[str],
    expect_wildcard_calls: int,
) -> None:
    """Verifies _update_with_stmt_provenance traverses the with-header once without double-scanning or duplicate wildcard records."""
    import pydoppelgangerhunt.fixer._df_provenance_walk as prov_mod  # pylint: disable=import-outside-toplevel

    stmt = ast.parse(with_src).body[0]
    assert isinstance(stmt, ast.With)

    real_iter = prov_mod._iter_with_item_provenance_steps
    iter_calls = 0

    def _spy_iter(*args: Any, **kwargs: Any) -> Any:
        nonlocal iter_calls
        iter_calls += 1
        return real_iter(*args, **kwargs)

    real_record_wc = prov_mod.record_wildcard_ctx_manager_match
    wc_calls = 0

    def _spy_wc(pat: str, mgr_name: str) -> None:
        nonlocal wc_calls
        wc_calls += 1
        real_record_wc(pat, mgr_name)

    monkeypatch.setattr(prov_mod, "_iter_with_item_provenance_steps", _spy_iter)
    monkeypatch.setattr(prov_mod, "record_wildcard_ctx_manager_match", _spy_wc)

    ctx = prov_mod._ProvenanceWalkContext(
        in_branch=False,
        strict_with_suppression=True,
        extra_non_suppressing_ctx_managers=extra_allowlist,
        depth=0,
    )
    res = prov_mod._update_with_stmt_provenance(
        stmt, prov_mod._EMPTY_SUPPRESS_PROVENANCE, ctx
    )
    assert res.terminated is False
    assert iter_calls == 1
    assert wc_calls == expect_wildcard_calls

@pytest.mark.parametrize(
    ("try_block_src", "expect_after_try_sup", "expect_after_try_safe", "expect_after_loop_safe"),
    [
        pytest.param(
            "        try:\n"
            "            if 1:\n"
            "                cm = contextlib.suppress\n"
            "                break\n"
            "            cm = contextlib.nullcontext\n"
            "        finally:\n"
            "            if flag:\n"
            "                continue\n",
            True,
            False,
            False,
            id="non_raising_try_break_suppressed_by_finally_continue",
        ),
        pytest.param(
            "        try:\n"
            "            if 1:\n"
            "                cm = contextlib.suppress\n"
            "                return 0\n"
            "            cm = contextlib.nullcontext\n"
            "        finally:\n"
            "            if flag:\n"
            "                continue\n",
            True,
            False,
            False,
            id="non_raising_try_return_suppressed_by_finally_continue",
        ),
        pytest.param(
            "        try:\n"
            "            x = 1\n"
            "        except ValueError:\n"
            "            cm = contextlib.nullcontext\n"
            "        else:\n"
            "            if 1:\n"
            "                cm = contextlib.suppress\n"
            "                break\n"
            "            cm = contextlib.nullcontext\n"
            "        finally:\n"
            "            if flag:\n"
            "                continue\n",
            True,
            False,
            False,
            id="non_raising_else_break_suppressed_by_finally_continue",
        ),
        pytest.param(
            "        try:\n"
            "            if 1:\n"
            "                cm = contextlib.suppress\n"
            "                break\n"
            "            cm = contextlib.nullcontext\n"
            "        finally:\n"
            "            cleanup = 1\n",
            False,
            True,
            False,
            id="control_non_suppressing_finally_keeps_fallthrough_safe_and_loop_exit_unsafe",
        ),
    ],
)
def test_try_else_non_raising_abrupt_exit_suppressed_by_finally_preserves_suppressor(
    try_block_src: str,
    expect_after_try_sup: bool,
    expect_after_try_safe: bool,
    expect_after_loop_safe: bool,
) -> None:
    """Verifies non-raising return/break/continue exits in try/else feed finally abrupt-exit provenance consistently."""
    from pydoppelgangerhunt.fixer._df_block_flow import (  # pylint: disable=import-outside-toplevel
        _analyze_block_assignment,
    )
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=import-outside-toplevel
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    # Check both pre-unit provenance walk and intra-unit block assignment on the try stmt
    fn_src = (
        "import contextlib\n"
        "def run(items, flag, step):\n"
        "    cm = contextlib.nullcontext\n"
        "    for _ in items:\n"
        + try_block_src
        + "        with cm(ValueError):\n"
        + "            v = step()\n"
        + "        cb = lambda: v\n"
        + "        return cb()\n"
    )
    mod = ast.parse(fn_src)
    fn_node = mod.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    loop_node = fn_node.body[1]
    assert isinstance(loop_node, ast.For)
    mod_prov = _update_suppress_provenance_from_stmts(
        [mod.body[0]], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    pre_loop_prov = _update_suppress_provenance_from_stmts(
        [fn_node.body[0]], mod_prov, in_branch=False
    )
    post_try_prov = _update_suppress_provenance_from_stmts(
        [loop_node.body[0]], pre_loop_prov, in_branch=False
    )
    assert ("cm" in post_try_prov.suppressors) is expect_after_try_sup
    assert (
        post_try_prov.suppressors
        & (post_try_prov.non_suppressors | post_try_prov.non_suppressor_instances)
        == frozenset()
    )

    def_intra, _ = _analyze_block_assignment(
        loop_node.body[:2],
        suppress_aliases=pre_loop_prov,
        strict_with_suppression=True,
    )
    assert ("v" in def_intra) is expect_after_try_safe

    # End-to-end scope check after the loop (and without loop back-edge for the non-suppressing return control)
    after_loop_src = (
        "import contextlib\n"
        "def run(items, flag, step):\n"
        "    cm = contextlib.nullcontext\n"
        "    for _ in items:\n"
        + try_block_src
        + "    with cm(ValueError):\n"
        + "        v = step()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety_after_loop = _late_bound_capture_safety_by_mode(
        after_loop_src, "    cb = lambda: v"
    )
    assert safety_after_loop == {
        "strict": expect_after_loop_safe,
        "pedantic": expect_after_loop_safe,
    }
    if not expect_after_try_sup:
        no_loop_control_src = (
            "import contextlib\n"
            "def run(step):\n"
            "    cm = contextlib.nullcontext\n"
            "    try:\n"
            "        if 1:\n"
            "            cm = contextlib.suppress\n"
            "            return 0\n"
            "        cm = contextlib.nullcontext\n"
            "    finally:\n"
            "        cleanup = 1\n"
            "    with cm(ValueError):\n"
            "        v = step()\n"
            "    cb = lambda: v\n"
            "    return cb()\n"
        )
        assert _late_bound_capture_safety_by_mode(
            no_loop_control_src, "    cb = lambda: v"
        ) == {"strict": True, "pedantic": True}

@pytest.mark.parametrize(
    ("comp_src", "expect_alias_suppressor"),
    [
        pytest.param(
            "[a for a in items if (cm := suppress) for b in [(alias := cm)]]\n",
            True,
            id="listcomp_filter_walrus_propagates_to_second_generator_iter",
        ),
        pytest.param(
            "{a: b for a in items if (cm := suppress) for b in [(alias := cm)]}\n",
            True,
            id="dictcomp_filter_walrus_propagates_to_second_generator_iter",
        ),
        pytest.param(
            "(a for a in items if (cm := suppress) for b in [(alias := cm)])\n",
            True,
            id="genexp_filter_walrus_propagates_to_second_generator_iter",
        ),
        pytest.param(
            "[a for a in items if (cm := nullcontext) for b in [(alias := cm)]]\n",
            False,
            id="control_non_suppressor_filter_walrus_not_marked_suppressor",
        ),
    ],
)
def test_multi_generator_comprehension_walrus_evaluation_order(
    comp_src: str,
    expect_alias_suppressor: bool,
) -> None:
    """Verifies comprehension walrus provenance evaluates gen[0].ifs before gen[1].iter."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import (
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance

    init_prov = _SuppressProvenance(
        suppressors=frozenset({"suppress"}),
        non_suppressors=frozenset({"nullcontext"}),
    )
    out_prov = _update_suppress_provenance_from_stmts(
        ast.parse(comp_src).body,
        init_prov,
        in_branch=False,
    )
    assert ("alias" in out_prov.suppressors) is expect_alias_suppressor, out_prov
    assert ("cm" in out_prov.suppressors) is expect_alias_suppressor, out_prov
