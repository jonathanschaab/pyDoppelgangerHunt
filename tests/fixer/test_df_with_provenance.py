"""Tests for _df_with_provenance context-manager suppression, allowlists, and definite-binding termination."""

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
            parsed.body[0].body
            if isinstance(parsed.body[0], ast.AsyncFunctionDef)
            else parsed.body
        )
        got_bound, got_term = _extract_block_definite_bindings_and_term(
            stmts, None, strict_with_suppression=True
        )
        assert got_bound == expected_bound, f"Mismatch for:\n{code_str}"
        assert got_term is expected_term, f"Term mismatch for:\n{code_str}"


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
    expected_rebind_muts = {"a", "b", "d", "e", "h"}
    if sys.version_info >= (3, 10):
        rebinding_fn.body.append(
            ast.parse("match a:\n    case f:\n        pass\n").body[0]
        )
        expected_rebind_muts.add("f")
    if sys.version_info >= (3, 12):
        rebinding_fn.body.append(ast.parse("type g = int\n").body[0])
        expected_rebind_muts.add("g")
    b_muts, b_dels = _extract_closure_nonlocal_mutations_and_deletions_uncached(
        rebinding_fn
    )
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
        assert (
            def_a == expected_def_set
        ), f"_analyze_block_assignment mismatch on:\n{snippet}"
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
    prov_after_both = _update_suppress_provenance_from_stmt(
        both_clear_stmts[0], prov_init
    )
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


def test_multi_item_with_subsequent_targets_and_walrus_after_suppressor() -> None:
    """Verifies that in a multi-item with statement, items after a suppressing or unknown item bind targets and walrus names only conditionally."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
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
    bound_ris, term_ris = _extract_block_definite_bindings_and_term(
        raises_in_sup_stmts, None
    )
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
    bound_nsr, term_nsr = _extract_block_definite_bindings_and_term(
        nested_safe_ret_stmts, None
    )
    assert bound_nsr == {"x"}
    assert term_nsr is True


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
        extra_non_suppressing_ctx_managers=frozenset(
            {"DynamicTransaction", ".RelativeTx"}
        ),
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


def test_pre_try_and_pre_match_bindings_survive_loop_control_termination() -> None:
    """Verifies pre-try and pre-match definite bindings survive when try or match terminates via loop control."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
    )

    try_loop_src = (
        "x = 1\n" "try:\n" "    break\n" "except RuntimeError:\n" "    continue\n"
    )
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
    assert (
        _class_def_is_proven_non_suppressing_in_exit(cls_node, prov)
        is expect_proven_safe
    )
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
    defs_short, conds_short = _analyze_block_assignment(
        ast.parse(src_short_circuit_walrus).body
    )
    assert "b" not in defs_short
    assert "b" in conds_short


def test_module_compound_statement_suppressor_alias_propagation_and_linear_seed() -> (
    None
):
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

    seeded_prov = _seed_scope_suppress_aliases(
        target_fn, mod_tree, u_start=target_fn.lineno
    )
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
    defs, conds = _analyze_block_assignment(stmts, strict_with_suppression=strict_with)
    assert expected_defs <= defs
    assert not (expected_conds & defs)
    assert expected_conds <= conds


def test_terminating_try_block_preserves_try_def_in_conditional_and_skips_unreachable_orelse() -> (
    None
):
    """Verifies terminating try blocks keep try_def in conditional when a handler falls through and ignore unreachable else blocks."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
    )

    try_src = (
        "try:\n"
        "    x = 20\n"
        "    may_raise()\n"
        "    return 0\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    unreachable_else = 99\n"
    )
    defs, conds = _analyze_block_assignment(ast.parse(try_src).body)
    assert "x" not in defs
    assert "x" in conds
    assert "unreachable_else" not in defs
    assert "unreachable_else" not in conds

    fn_src = (
        "def run(may_raise):\n"
        "    try:\n"
        "        x = 20\n"
        "        may_raise()\n"
        "        return 0\n"
        "    except Exception:\n"
        "        pass\n"
        "    return x\n"
    )
    sc = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 7,
            "kind": "compound_block",
            "source_text": fn_src,
            "source_code": (
                "    try:\n"
                "        x = 20\n"
                "        may_raise()\n"
                "        return 0\n"
                "    except Exception:\n"
                "        pass\n"
            ),
        },
        tree=ast.parse(fn_src),
    )
    assert "x" in sc["conditional_outputs"]
    assert "x" not in sc["definite_stores"]


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


@pytest.mark.parametrize(
    "block_src",
    [
        (
            "from contextlib import suppress\n"
            "if cond:\n"
            "    d = 1\n"
            "else:\n"
            "    d = 2\n"
            "with suppress(ZeroDivisionError):\n"
            "    _ = d\n"
            "    return 1\n"
        ),
        (
            "from contextlib import suppress, nullcontext\n"
            "with nullcontext() as d:\n"
            "    pass\n"
            "with suppress(ZeroDivisionError):\n"
            "    _ = d\n"
            "    return 1\n"
        ),
        (
            "from contextlib import suppress\n"
            "try:\n"
            "    d = 1\n"
            "except Exception as exc:\n"
            "    with suppress(ZeroDivisionError):\n"
            "        _ = exc\n"
            "        return 2\n"
            "else:\n"
            "    with suppress(ZeroDivisionError):\n"
            "        _ = d\n"
            "        return 1\n"
        ),
        (
            "from contextlib import suppress\n"
            "_ = (d := 1)\n"
            "with suppress(ZeroDivisionError):\n"
            "    _ = d\n"
            "    return 1\n"
        ),
    ],
)
def test_block_terminates_matches_analyze_block_assignment_across_compound_safe_names(
    block_src: str,
) -> None:
    """Verifies _block_terminates and _analyze_block_assignment_impl agree on safe_names across compound statements."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
        _block_terminates,
        _seed_scope_suppress_aliases,
    )

    mod_tree = ast.parse(block_src)
    prov = _seed_scope_suppress_aliases(mod_tree, mod_tree)
    _, _, impl_term, _ = _analyze_block_assignment_impl(
        mod_tree.body,
        suppress_aliases=prov,
    )
    bt_strict = _block_terminates(
        mod_tree.body,
        suppress_aliases=prov,
        strict_names=True,
    )
    bt_non_strict = _block_terminates(
        mod_tree.body,
        suppress_aliases=prov,
        strict_names=False,
    )
    assert impl_term is True
    assert bt_strict == impl_term
    assert bt_non_strict is True


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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
        _block_terminates,
        _extract_block_definite_bindings_and_term,
        _seed_scope_suppress_aliases,
    )

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
        f"Summary: {_MAX_HEURISTIC_CTX_MANAGERS} unproven context manager(s)"
        in rec.message
        and "5 additional occurrence(s) omitted beyond cap" in rec.message
        for rec in caplog.records
    )


def test_pathlib_path_and_uncalled_factories_not_classified_as_non_suppressing(
    tmp_path: Path,
) -> None:
    """Verifies pathlib.Path and bare uncalled context-manager factories are never treated as non-suppressing."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _STDLIB_NON_SUPPRESSING_MODULES,
        _classify_with_suppression,
        _seed_scope_suppress_aliases,
    )

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
        with_b = next(n for n in run_b.body if isinstance(n, ast.With))
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
        "masked()",
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
