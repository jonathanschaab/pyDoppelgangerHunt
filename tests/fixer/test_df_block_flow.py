"""Tests for unified block definite-assignment, deletion tracking, termination, and differential runtime checks."""

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
    _late_bound_capture_safety_by_mode,
)

def test_with_and_async_with_definite_bindings() -> None:
    """Covers definite-binding analysis inside with and async with bodies."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _find_enclosing_chain_definite_pre_bound,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_block_flow import (
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
        _extract_fn_outer_carrier_targets_uncached,
        _propagate_sibling_closures,
    )
    from pydoppelgangerhunt.fixer._df_scope_ctx import (
        _extract_closure_nonlocal_mutations_and_deletions_uncached,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
        _resolve_unit_col_bounds,
    )
    from pydoppelgangerhunt.fixer._df_block_flow import (
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
    from pydoppelgangerhunt.fixer._df_scope_ctx import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )
    from pydoppelgangerhunt.fixer._df_block_flow import (
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
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment
    from pydoppelgangerhunt.fixer._df_with_rules import _walrus_assignment_in_expr

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

def test_deep_nested_with_early_return_propagation() -> None:
    """Verifies 3-level nested with + early return propagation across known and unknown context managers."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

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
    from pydoppelgangerhunt.fixer._df_block_flow import (
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
        "from contextlib import suppress\n"
        "with suppress(ValueError):\n"
        "    with suppress(KeyError):\n"
        "        x = 10\n"
        "    return x\n"
        "    risky_unreachable()\n"
    )
    nested_safe_ret_stmts = ast.parse(nested_safe_ret_src).body
    assert _block_terminates(nested_safe_ret_stmts) is True
    bound_nsr, term_nsr = _extract_block_definite_bindings_and_term(
        nested_safe_ret_stmts, None
    )
    assert bound_nsr == {"suppress", "x"}
    assert term_nsr is True

def test_pre_try_and_pre_match_bindings_survive_loop_control_termination() -> None:
    """Verifies pre-try and pre-match definite bindings survive when try or match terminates via loop control."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment_impl

    try_loop_src = (
        "x = 1\n" "try:\n" "    break\n" "except RuntimeError:\n" "    continue\n"
    )
    res_try = _analyze_block_assignment_impl(
        ast.parse(try_loop_src).body,
        loop_control_terminates=True,
    )
    assert res_try.terminated is True
    assert "x" in res_try.definite
    assert "x" not in res_try.conditional

    if sys.version_info >= (3, 10):
        match_loop_src = (
            "x = 1\n"
            "match status:\n"
            "    case 200:\n"
            "        break\n"
            "    case _:\n"
            "        continue\n"
        )
        res_match = _analyze_block_assignment_impl(
            ast.parse(match_loop_src).body,
            loop_control_terminates=True,
        )
        assert res_match.terminated is True
        assert "x" in res_match.definite
        assert "x" not in res_match.conditional

def test_terminating_try_block_preserves_try_def_in_conditional_and_skips_unreachable_orelse() -> (
    None
):
    """Verifies terminating try blocks keep try_def in conditional when a handler falls through and ignore unreachable else blocks."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

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
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment_impl,
        _block_terminates,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    mod_tree = ast.parse(block_src)
    prov = _seed_scope_suppress_aliases(mod_tree, mod_tree)
    impl_term = _analyze_block_assignment_impl(
        mod_tree.body,
        suppress_aliases=prov,
    ).terminated
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

def _execute_block_runtime_paths(
    body_src: str,
    pre_init_names: set[str],
    input_envs: list[dict[str, object]],
) -> tuple[int, set[str], set[str]]:
    """Executes body_src inside a wrapper function across input_envs and returns (fallthrough_runs, always_bound, ever_bound)."""
    import warnings

    indented_lines = [
        f"    {line}" if line.strip() else ""
        for line in body_src.splitlines()
    ]
    init_lines = [f"    {name} = 'init'" for name in sorted(pre_init_names)]
    env_keys = sorted({k for env in input_envs for k in env})
    unpack_lines = [f"    {k} = __env.get({k!r})" for k in env_keys]
    fn_src = "\n".join(
        [
            "def _probe(__env):",
            *unpack_lines,
            *init_lines,
            *indented_lines,
            "    return ('__FALLTHROUGH__', set(locals().keys()))",
        ]
    )
    ns: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        code_obj = compile(fn_src, "<diff_oracle>", "exec")
    exec(code_obj, ns)  # noqa: S102  # pylint: disable=exec-used
    probe_fn = ns["_probe"]
    assert callable(probe_fn)

    excluded = {"__env", *env_keys}
    fallthrough_sets: list[set[str]] = []
    for env in input_envs:
        try:
            outcome = probe_fn(env)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if (
            isinstance(outcome, tuple)
            and len(outcome) == 2
            and outcome[0] == "__FALLTHROUGH__"
            and isinstance(outcome[1], set)
        ):
            fallthrough_sets.append(outcome[1] - excluded)

    if not fallthrough_sets:
        return 0, set(), set()
    always_bound = set.intersection(*fallthrough_sets)
    ever_bound = set.union(*fallthrough_sets)
    return len(fallthrough_sets), always_bound, ever_bound

@pytest.mark.parametrize(
    ("shape_id", "pre_init", "body_src", "envs", "expected_static_def", "expect_term"),
    [
        (
            "branch_return_vs_fallthrough",
            set(),
            (
                "if flag:\n"
                "    early = 1\n"
                "    return early\n"
                "else:\n"
                "    kept = 2\n"
                "after = kept + 1\n"
            ),
            [{"flag": True}, {"flag": False}],
            {"kept", "after"},
            False,
        ),
        (
            "raise_caught_by_handler",
            set(),
            (
                "try:\n"
                "    before_raise = 1\n"
                "    if fail:\n"
                "        raise ValueError('boom')\n"
                "    survivor = 10\n"
                "except ValueError:\n"
                "    survivor = 20\n"
                "    only_err = 99\n"
            ),
            [{"fail": False}, {"fail": True}],
            {"survivor"},
            False,
        ),
        (
            "handler_reraises_to_outer",
            {"pre"},
            (
                "try:\n"
                "    if mode == 'val':\n"
                "        del pre\n"
                "        raise ValueError('v')\n"
                "    if mode == 'type':\n"
                "        raise TypeError('t')\n"
                "    res = 1\n"
                "except ValueError:\n"
                "    dead_on_fallthrough = 1\n"
                "    raise RuntimeError('escalate')\n"
                "except TypeError:\n"
                "    res = 2\n"
            ),
            [{"mode": "ok"}, {"mode": "val"}, {"mode": "type"}],
            {"res"},
            False,
        ),
        (
            "terminating_finally_block",
            set(),
            (
                "try:\n"
                "    body_var = 1\n"
                "finally:\n"
                "    fin_var = 2\n"
                "    return fin_var\n"
            ),
            [{}],
            {"body_var", "fin_var"},
            True,
        ),
        (
            "conditional_del_followed_by_reassignment",
            {"x", "victim"},
            (
                "if flag:\n"
                "    del x\n"
                "    del victim\n"
                "x = 42\n"
            ),
            [{"flag": False}, {"flag": True}],
            {"x"},
            False,
        ),
        (
            "loop_break_and_continue_inside_try_finally",
            set(),
            (
                "seen = 0\n"
                "for item in items:\n"
                "    try:\n"
                "        if item < 0:\n"
                "            neg = item\n"
                "            break\n"
                "        if item == 0:\n"
                "            zero = 0\n"
                "            continue\n"
                "        pos = item\n"
                "    finally:\n"
                "        fin_loop = 1\n"
                "else:\n"
                "    no_break = 1\n"
            ),
            [{"items": []}, {"items": [1, 2]}, {"items": [0, -1]}],
            {"seen"},
            False,
        ),
        (
            "finally_continue_overrides_try_break",
            set(),
            (
                "for item in items:\n"
                "    try:\n"
                "        if item < 0:\n"
                "            break\n"
                "    finally:\n"
                "        if swallow_break:\n"
                "            continue\n"
                "    post_try = item\ndone = 1\n"
            ),
            [
                {"items": [], "swallow_break": False},
                {"items": [-1], "swallow_break": False},
                {"items": [-1], "swallow_break": True},
                {"items": [2], "swallow_break": False},
            ],
            {"done"},
            False,
        ),
        (
            "multiple_handlers_distinct_bindings_and_err_cleanup",
            {"err"},
            (
                "try:\n"
                "    if kind == 1:\n"
                "        raise ValueError('v')\n"
                "    if kind == 2:\n"
                "        raise KeyError('k')\n"
                "    shared = 'ok'\n"
                "except ValueError as err:\n"
                "    shared = 'val'\n"
                "    only_v = 1\n"
                "except KeyError:\n"
                "    shared = 'key'\n"
                "    only_k = 2\n"
            ),
            [{"kind": 0}, {"kind": 1}, {"kind": 2}],
            {"shared"},
            False,
        ),
    ],
)
def test_differential_block_assignment_against_python_runtime_across_control_flow_shapes(
    shape_id: str,
    pre_init: set[str],
    body_src: str,
    envs: list[dict[str, object]],
    expected_static_def: set[str],
    expect_term: bool,
) -> None:
    """Compares static definite/conditional/deletion/termination results against actual Python execution."""
    import warnings
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment_impl,
        _extract_block_definite_bindings_and_term,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        stmts = ast.parse(body_src).body
    may_del: set[str] = set()
    live_in_del: set[str] = set()
    res = _analyze_block_assignment_impl(
        stmts,
        definite=set(pre_init),
        may_deleted=may_del,
        live_in_deleted=live_in_del,
    )
    static_def, static_cond, static_term = res.definite, res.conditional, res.terminated
    def_helper, term_helper = _extract_block_definite_bindings_and_term(
        stmts, None, initial_bound=frozenset(pre_init)
    )
    assert static_def.isdisjoint(static_cond), shape_id
    assert static_term is expect_term, shape_id
    assert term_helper is expect_term, shape_id
    assert def_helper == static_def, shape_id
    assert static_def == expected_static_def, (shape_id, static_def, expected_static_def)

    runs, always_bound, ever_bound = _execute_block_runtime_paths(
        body_src, pre_init, envs
    )
    if expect_term:
        assert runs == 0, shape_id
    else:
        assert runs > 0, shape_id
        assert static_def <= always_bound, (shape_id, static_def, always_bound)
        assert (ever_bound - static_def) <= static_cond, (
            shape_id,
            ever_bound,
            static_def,
            static_cond,
        )
        assert (pre_init - always_bound) <= may_del, (
            shape_id,
            pre_init - always_bound,
            may_del,
        )

@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* requires Python 3.11+")
def test_differential_except_star_exception_group_subset_execution() -> None:
    """Compares static except* multi-handler subset semantics against Python 3.11+ ExceptionGroup execution."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment_impl,
        _extract_block_definite_bindings_and_term,
    )

    body_src = (
        "try:\n"
        "    if exc is not None:\n"
        "        raise exc\n"
        "    all_paths = 0\n"
        "    clean_only = 1\n"
        "except* ValueError as eg_v:\n"
        "    all_paths = 1\n"
        "    saw_v = 1\n"
        "    del survivor\n"
        "except* TypeError as eg_t:\n"
        "    all_paths = 2\n"
        "    saw_t = 1\n"
        "finally:\n"
        "    fin_always = 9\n"
    )
    eg_cls = getattr(builtins_mod := __import__("builtins"), "ExceptionGroup")
    envs: list[dict[str, object]] = [
        {"exc": None},
        {"exc": ValueError("v")},
        {"exc": TypeError("t")},
        {"exc": eg_cls("both", [ValueError("v"), TypeError("t")])},
        {"exc": eg_cls("partial_unhandled", [ValueError("v"), KeyError("k")])},
    ]
    pre_init = {"survivor", "keep"}
    stmts = ast.parse(body_src).body
    may_del: set[str] = set()
    res = _analyze_block_assignment_impl(
        stmts,
        definite=set(pre_init),
        may_deleted=may_del,
    )
    static_def, static_cond, static_term = res.definite, res.conditional, res.terminated
    def_helper, term_helper = _extract_block_definite_bindings_and_term(
        stmts, None, initial_bound=frozenset(pre_init)
    )
    runs, always_bound, ever_bound = _execute_block_runtime_paths(
        body_src, pre_init, envs
    )
    assert runs == 4
    assert static_term is False
    assert term_helper is False
    assert always_bound == {"keep", "all_paths", "fin_always"}
    assert static_def == {"keep", "fin_always"}
    assert static_def <= always_bound
    assert def_helper == static_def
    assert "all_paths" in static_cond
    assert (ever_bound - static_def) <= static_cond
    assert "survivor" in may_del
    assert {"eg_v", "eg_t"}.isdisjoint(always_bound)
    assert {"eg_v", "eg_t"}.isdisjoint(static_def)
    assert builtins_mod is not None

def test_generated_try_except_else_finally_combinatorial_differential_oracle() -> None:
    """Exhaustively compares _analyze_block_assignment_impl against runtime execution across generated try/except/else/finally blocks."""
    import warnings
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment_impl

    try_variants = [
        "    if mode == 'ret':\n        return 0\n    if mode == 'val':\n        raise ValueError('v')\n    x = 1\n",
        "    x = 1\n    if mode == 'val':\n        del pre\n        raise ValueError('v')\n",
    ]
    handler_variants = [
        "except ValueError:\n    x = 2\n",
        "except ValueError:\n    h_only = 2\n",
        "except ValueError:\n    del pre\n    raise RuntimeError('esc')\n",
    ]
    else_variants = [
        "",
        "else:\n    e_var = 3\n    x = 3\n",
    ]
    finally_variants = [
        "",
        "finally:\n    fin = 4\n",
        "finally:\n    del x\n    fin = 4\n",
        "finally:\n    fin = 4\n    return fin\n",
    ]
    envs: list[dict[str, object]] = [
        {"mode": "ok"},
        {"mode": "val"},
        {"mode": "ret"},
    ]
    pre_init = {"pre"}

    checked = 0
    for t_body in try_variants:
        for h_clause in handler_variants:
            for e_clause in else_variants:
                for f_clause in finally_variants:
                    if not h_clause and not f_clause:
                        continue
                    parts = ["x = 0\n", "try:\n", t_body, h_clause]
                    if e_clause:
                        parts.append(e_clause)
                    if f_clause:
                        parts.append(f_clause)
                    prog = "".join(parts)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        stmts = ast.parse(prog).body
                    may_del: set[str] = set()
                    res = _analyze_block_assignment_impl(
                        stmts,
                        definite=set(pre_init),
                        may_deleted=may_del,
                    )
                    s_def, s_cond, s_term = res.definite, res.conditional, res.terminated
                    runs, always_bound, ever_bound = _execute_block_runtime_paths(
                        prog, pre_init, envs
                    )
                    assert s_def.isdisjoint(s_cond), prog
                    if runs == 0:
                        assert s_term is True, prog
                    else:
                        assert s_term is False, prog
                        assert s_def <= always_bound, (prog, s_def, always_bound)
                        assert (ever_bound - s_def) <= s_cond, (
                            prog,
                            ever_bound,
                            s_def,
                            s_cond,
                        )
                        assert (pre_init - always_bound) <= may_del, (
                            prog,
                            always_bound,
                            may_del,
                        )
                    checked += 1
    assert checked == 48

@pytest.mark.parametrize(
    ("min_py", "header_stmt", "expect_safe"),
    [
        pytest.param(
            (3, 9),
            "    if mutator():\n        pass\n",
            False,
            id="if_test_cross_scope_mutator_invalidates_proven_cm",
        ),
        pytest.param(
            (3, 9),
            "    while mutator():\n        pass\n",
            False,
            id="while_test_cross_scope_mutator_invalidates_proven_cm",
        ),
        pytest.param(
            (3, 9),
            "    for cm in (contextlib.suppress,):\n        pass\n",
            False,
            id="for_target_clears_cm_and_invalidates_nullcontext",
        ),
        pytest.param(
            (3, 10),
            "    match mutator():\n        case _:\n            pass\n",
            False,
            id="match_subject_cross_scope_mutator_invalidates_proven_cm",
        ),
        pytest.param(
            (3, 9),
            "    if (flag := 1):\n        pass\n",
            True,
            id="call_free_if_test_walrus_preserves_proven_nullcontext",
        ),
    ],
)
def test_block_assignment_headers_track_cross_scope_mutators_and_for_targets(
    min_py: tuple[int, int],
    header_stmt: str,
    expect_safe: bool,
) -> None:
    """Verifies If/While/For/Match headers apply cross-scope invalidation and for-target clearing."""
    if sys.version_info < min_py:
        pytest.skip(f"Requires Python {min_py[0]}.{min_py[1]}+")
    module_src = (
        "import contextlib\n"
        "def run(step):\n"
        "    cm = contextlib.nullcontext\n"
        "    def mutator():\n"
        "        nonlocal cm\n"
        "        cm = contextlib.suppress\n"
        "        return False\n"
        + header_stmt
        + "    with cm():\n"
        + "        v = step()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

def test_annassign_without_value_applies_annotation_walrus_in_block_assignment() -> None:
    """Verifies bare AnnAssign records annotation walrus bindings as conditional rather than definite."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    ann_stmt = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.NamedExpr(
            target=ast.Name(id="y", ctx=ast.Store()),
            value=ast.Constant(value=1),
        ),
        value=None,
        simple=1,
    )
    ast.fix_missing_locations(ann_stmt)
    definite, conditional = _analyze_block_assignment([ann_stmt])
    assert "y" not in definite
    assert "y" in conditional
    assert "x" not in definite
    assert "x" not in conditional

@pytest.mark.parametrize(
    ("shape", "depth", "max_calls"),
    [
        pytest.param("nested_try_prov", 30, 120, id="nested_try_provenance_linear"),
        pytest.param("nested_try_terminating_handler", 30, 180, id="nested_try_terminating_handler_linear"),
        pytest.param("nested_loop_finally_no_break", 30, 180, id="nested_loop_finally_no_break_linear"),
        pytest.param("nested_loop_finally_with_break", 25, 2500, id="nested_loop_finally_with_break_polynomial"),
    ],
)
def test_deeply_nested_try_and_loop_finally_avoid_exponential_blowup(
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
    depth: int,
    max_calls: int,
) -> None:
    """Verifies deeply nested try and loop try/finally blocks run in polynomial call counts with exact semantics."""
    import pydoppelgangerhunt.fixer._df_block_flow as block_mod
    import pydoppelgangerhunt.fixer._df_provenance_walk as walk_mod
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

    prov_calls = 0
    flow_calls = 0
    real_prov_stmt = walk_mod._update_prov_and_term_from_stmt
    real_flow_impl = block_mod._analyze_block_assignment_impl

    def _spy_prov_stmt(*args: Any, **kwargs: Any) -> Any:
        nonlocal prov_calls
        prov_calls += 1
        return real_prov_stmt(*args, **kwargs)

    def _spy_flow_impl(*args: Any, **kwargs: Any) -> Any:
        nonlocal flow_calls
        flow_calls += 1
        return real_flow_impl(*args, **kwargs)

    monkeypatch.setattr(walk_mod, "_update_prov_and_term_from_stmt", _spy_prov_stmt)
    monkeypatch.setattr(block_mod, "_analyze_block_assignment_impl", _spy_flow_impl)

    if shape == "nested_try_prov":
        lines = ["import contextlib\n", "cm = contextlib.nullcontext\n"]
        for idx in range(depth):
            lines.append("    " * idx + "try:\n")
            lines.append("    " * (idx + 1) + "step()\n")
        lines.append("    " * depth + "cm = contextlib.suppress\n")
        for idx in range(depth - 1, -1, -1):
            lines.append("    " * idx + "except Exception:\n")
            lines.append("    " * (idx + 1) + "pass\n")
        stmts = ast.parse("".join(lines)).body
        out_prov = walk_mod._update_suppress_provenance_from_stmts(
            stmts, _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
        )
        assert "cm" in out_prov.suppressors
        assert "cm" not in out_prov.non_suppressors
        assert prov_calls <= max_calls
        assert flow_calls == 0
    elif shape == "nested_try_terminating_handler":
        lines = ["import contextlib\n", "cm = contextlib.nullcontext\n"]
        for idx in range(depth):
            lines.append("    " * idx + "try:\n")
            lines.append("    " * (idx + 1) + "step()\n")
        for idx in range(depth - 1, -1, -1):
            lines.append("    " * idx + "except Exception:\n")
            lines.append("    " * (idx + 1) + "cm = contextlib.suppress\n")
            lines.append("    " * (idx + 1) + "raise\n")
        stmts = ast.parse("".join(lines)).body
        out_prov = walk_mod._update_suppress_provenance_from_stmts(
            stmts, _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
        )
        assert "cm" in out_prov.non_suppressors
        assert "cm" not in out_prov.suppressors
        assert prov_calls <= max_calls
        assert flow_calls == 0
    elif shape == "nested_loop_finally_no_break":
        lines = ["for _ in items:\n"]
        for idx in range(depth):
            lines.append("    " * (idx + 1) + "try:\n")
            lines.append("    " * (idx + 2) + "x = step()\n")
            lines.append("    " * (idx + 1) + "finally:\n")
        lines.append("    " * (depth + 1) + "done = 1\n")
        stmts = ast.parse("".join(lines)).body
        definite, conditional = block_mod._analyze_block_assignment(stmts)
        assert "done" in conditional and "done" not in definite
        assert flow_calls <= max_calls
    else:
        lines = ["for _ in items:\n"]
        for idx in range(depth):
            lines.append("    " * (idx + 1) + "try:\n")
            lines.append("    " * (idx + 2) + "if flag:\n")
            lines.append("    " * (idx + 3) + "break\n")
            lines.append("    " * (idx + 1) + "finally:\n")
        lines.append("    " * (depth + 1) + "done = 1\n")
        lines.append("    " * (depth + 1) + "if flag:\n")
        lines.append("    " * (depth + 2) + "break\n")
        stmts = ast.parse("".join(lines)).body
        definite, conditional = block_mod._analyze_block_assignment(stmts)
        assert "done" in conditional and "done" not in definite
        assert flow_calls <= max_calls

@pytest.mark.parametrize(
    ("loop_body_src", "expect_definite"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "if cond:\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm(ValueError):\n"
                "                return compute()\n"
                "            break\n"
                "        cm = contextlib.suppress\n"
                "    else:\n"
                "        return 0\n"
                "else:\n"
                "    v = 1\n"
            ),
            False,
            id="for_loop_rebinding_after_with_enables_iteration2_break_fallthrough",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "if cond:\n"
                "    cm = contextlib.nullcontext\n"
                "    while flag:\n"
                "        if ready:\n"
                "            with cm(ValueError):\n"
                "                return compute()\n"
                "            break\n"
                "        cm = contextlib.suppress\n"
                "    else:\n"
                "        return 0\n"
                "else:\n"
                "    v = 1\n"
            ),
            False,
            id="while_loop_rebinding_after_with_enables_iteration2_break_fallthrough",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "if cond:\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        cm = contextlib.nullcontext\n"
                "        if i > 0:\n"
                "            with cm():\n"
                "                return compute()\n"
                "            break\n"
                "        cm = contextlib.suppress\n"
                "    else:\n"
                "        return 0\n"
                "else:\n"
                "    v = 1\n"
            ),
            True,
            id="control_unconditional_in_loop_restore_before_with_keeps_termination",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "if cond:\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm():\n"
                "                return compute()\n"
                "            break\n"
                "    else:\n"
                "        return 0\n"
                "else:\n"
                "    v = 1\n"
            ),
            True,
            id="control_unmodified_cm_in_loop_keeps_termination",
        ),
    ],
)
def test_loop_carried_ctx_manager_rebinding_invalidates_block_definite_binding(
    loop_body_src: str,
    expect_definite: bool,
) -> None:
    """Verifies loop-body context-manager rebinding propagates across loop back-edges in _analyze_block_assignment."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    stmts = ast.parse(loop_body_src).body
    for strict_mode in (False, True):
        definite, conditional = _analyze_block_assignment(
            stmts, strict_with_suppression=strict_mode
        )
        assert ("v" in definite) is expect_definite
        assert ("v" in conditional) is (not expect_definite)

@pytest.mark.parametrize(
    ("stmt_src", "active_nonlocals", "fn_globals", "expect_deleted"),
    [
        pytest.param("del x\n", set(), frozenset(), False, id="class_local_delete_ignored"),
        pytest.param(
            "del x\n", None, frozenset({"x"}), False, id="global_delete_ignored"
        ),
        pytest.param(
            "try:\n    op()\nexcept Exception as x:\n    pass\n",
            set(),
            frozenset(),
            False,
            id="class_local_except_as_ignored",
        ),
        pytest.param(
            "del x\n", {"x"}, frozenset(), True, id="control_nonlocal_delete_tracked"
        ),
        pytest.param(
            "del x\n", None, frozenset(), True, id="control_function_local_delete_tracked"
        ),
    ],
)
@pytest.mark.parametrize("hit_depth_limit", [False, True], ids=["normal", "depth_bailout"])
def test_delete_and_except_handler_respect_active_nonlocals_and_fn_globals(
    stmt_src: str,
    active_nonlocals: set[str] | None,
    fn_globals: frozenset[str],
    expect_deleted: bool,
    hit_depth_limit: bool,
) -> None:
    """Verifies ast.Delete and depth-limit deletion flow filter names via active_nonlocals and fn_globals."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _analyze_block_assignment_impl,
    )
    from pydoppelgangerhunt.fixer._df_visitors import _MAX_AST_RECURSION_DEPTH

    stmts = ast.parse(stmt_src).body
    may_del: set[str] = set()
    must_del: set[str] = set()
    live_in_del: set[str] = set()
    depth = _MAX_AST_RECURSION_DEPTH + 1 if hit_depth_limit else 0
    definite = _analyze_block_assignment_impl(
        stmts,
        definite={"x"},
        may_deleted=may_del,
        must_deleted=must_del,
        live_in_deleted=live_in_del,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        _depth=depth,
    ).definite
    assert ("x" not in definite) is expect_deleted
    assert ("x" in may_del) is expect_deleted

@pytest.mark.parametrize(
    ("try_src", "var_name", "expect_live_in_deleted"),
    [
        pytest.param(
            "try:\n    risky()\n    x = 1\nfinally:\n    del x\n",
            "x",
            True,
            id="try_risky_before_assign_finally_del_marks_live_in_deleted",
        ),
        pytest.param(
            "try:\n    risky()\n    x = 1\nexcept ValueError:\n    x = 2\nfinally:\n    del x\n",
            "x",
            True,
            id="try_with_narrow_except_finally_del_marks_live_in_deleted",
        ),
        pytest.param(
            "try:\n    x = 1\nexcept RuntimeError:\n    y = 3\nelse:\n    risky()\n    y = 2\nfinally:\n    del y\n",
            "y",
            True,
            id="try_else_risky_before_assign_finally_del_marks_live_in_deleted",
        ),
        pytest.param(
            "try:\n    if True:\n        return\n    x = 1\nfinally:\n    del x\n",
            "x",
            True,
            id="try_early_return_before_assign_finally_del_marks_live_in_deleted",
        ),
        pytest.param(
            "try:\n    x = 1\n    risky()\nfinally:\n    del x\n",
            "x",
            False,
            id="control_non_raising_prefix_binds_var_before_risky",
        ),
        pytest.param(
            "try:\n    x = 1\n    if True:\n        return\nfinally:\n    del x\n",
            "x",
            False,
            id="control_non_raising_prefix_binds_var_before_early_return",
        ),
        pytest.param(
            "try:\n    risky()\n    x = 1\nfinally:\n    x = 2\n    del x\n",
            "x",
            False,
            id="control_finally_rebinds_var_before_delete",
        ),
        pytest.param(
            "try:\n    x = 1\nfinally:\n    del x\n",
            "x",
            False,
            id="control_non_raising_try_body_binds_var_unconditionally",
        ),
    ],
)
def test_try_finally_delete_records_live_in_deleted_on_abrupt_exception_entry(
    try_src: str,
    var_name: str,
    expect_live_in_deleted: bool,
) -> None:
    """Verifies try/finally deletion records live_in_deleted on abrupt exits before try-body binding."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    live_in_del: set[str] = set()
    _analyze_block_assignment(ast.parse(try_src).body, live_in_deleted=live_in_del)
    assert (var_name in live_in_del) is expect_live_in_deleted

    indented = "\n".join(f"    {ln}" for ln in try_src.splitlines())
    fn_src = f"def outer(risky):\n{indented}\n"
    fn_lines = fn_src.splitlines()
    start_line = next(i + 1 for i, ln in enumerate(fn_lines) if ln.strip() == "try:")
    end_line = len(fn_lines)
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": start_line,
            "end": end_line,
            "kind": "compound_block",
            "source_text": fn_src,
            "source_code": indented + "\n",
        },
        tree=ast.parse(fn_src),
    )
    assert (var_name in sc["inputs"]) is expect_live_in_deleted, sc
    assert (
        "deleted_output" in sc["control_flow_hazards"]
    ) is expect_live_in_deleted, sc

@pytest.mark.parametrize(
    ("fn_src", "expect_strict_definite", "expect_pedantic_definite"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer(items, risky):\n"
                "    cm = suppress\n"
                "    for cm in items:\n"
                "        pass\n"
                "    with cm(ValueError):\n"
                "        x = risky()\n"
                "    return x\n"
            ),
            False,
            False,
            id="for_zero_trip_preserves_outer_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "async def outer(items, risky):\n"
                "    cm = suppress\n"
                "    async for cm in items:\n"
                "        pass\n"
                "    with cm(ValueError):\n"
                "        x = risky()\n"
                "    return x\n"
            ),
            False,
            False,
            id="async_for_zero_trip_preserves_outer_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer(items, risky):\n"
                "    cm = suppress\n"
                "    for cm in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm(ValueError):\n"
                "            x = risky()\n"
                "    return x\n"
            ),
            False,
            False,
            id="for_else_zero_trip_preserves_outer_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext, suppress\n"
                "def outer(items, risky):\n"
                "    cm = suppress\n"
                "    for cm in items:\n"
                "        cm = nullcontext\n"
                "    with cm(ValueError):\n"
                "        x = risky()\n"
                "    return x\n"
            ),
            False,
            False,
            id="for_zero_trip_preserves_suppressor_despite_body_rebind_to_nullcontext",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, risky):\n"
                "    cm = nullcontext\n"
                "    for cm in items:\n"
                "        pass\n"
                "    with cm():\n"
                "        x = risky()\n"
                "    return x\n"
            ),
            True,
            False,
            id="for_rebind_clears_nullcontext_proof_in_pedantic",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, risky):\n"
                "    cm = nullcontext\n"
                "    for other in items:\n"
                "        pass\n"
                "    with cm():\n"
                "        x = risky()\n"
                "    return x\n"
            ),
            True,
            True,
            id="control_unrelated_for_target_preserves_nullcontext_proof",
        ),
    ],
)
def test_for_loop_zero_trip_preserves_outer_suppressor_in_block_assignment(
    fn_src: str,
    expect_strict_definite: bool,
    expect_pedantic_definite: bool,
) -> None:
    """Verifies For/AsyncFor keeps post-iter provenance as the zero-trip baseline so outer suppressors are not erased."""
    lines = fn_src.splitlines()
    start_ln = next(
        idx for idx, ln in enumerate(lines, start=1) if ln.strip().startswith("cm = ")
    )
    end_ln = next(
        idx for idx, ln in enumerate(lines, start=1) if ln.strip().startswith("return ")
    ) - 1
    body_code = "\n".join(lines[start_ln - 1 : end_ln]) + "\n"
    tree = ast.parse(fn_src)
    for mode, expect_def in (
        ("strict", expect_strict_definite),
        ("pedantic", expect_pedantic_definite),
    ):
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
        assert ("x" in sc["definite_stores"]) is expect_def, (mode, sc)

@pytest.mark.parametrize(
    ("depth",),
    [
        pytest.param(3, id="shallow_nested_try_finally_in_loop"),
        pytest.param(22, id="deep_nested_try_finally_in_loop_avoids_exponential_blowup"),
    ],
)
def test_deeply_nested_try_finally_in_loop_with_break_stays_linear(depth: int) -> None:
    """Verifies deeply nested try/finally blocks with break inside loops do not trigger O(3^N) blowup."""
    import time
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment

    lines = ["for _ in items:"]
    indent = "    "
    for level in range(depth):
        lines.append(f"{indent}try:")
        lines.append(f"{indent}    if cond:")
        lines.append(f"{indent}        v_{level} = 1")
        lines.append(f"{indent}        break")
        lines.append(f"{indent}finally:")
        indent += "    "
    lines.append(f"{indent}if cond:")
    lines.append(f"{indent}    break")
    stmts = ast.parse("\n".join(lines) + "\n").body

    t0 = time.perf_counter()
    definite, conditional = _analyze_block_assignment(stmts)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"Nested try/finally loop took {elapsed:.2f}s at depth={depth}"
    assert "v_0" not in definite
    assert "v_0" in conditional
    assert f"v_{depth - 1}" in conditional

@pytest.mark.parametrize(
    ("body_src", "active_nonlocals", "fn_globals", "expect_x_still_definite"),
    [
        pytest.param(
            "with suppress(ValueError):\n    risky()\n    del x\n",
            set(),
            frozenset(),
            True,
            id="class_body_with_risky_del_ignored_without_nonlocal",
        ),
        pytest.param(
            "with suppress(ValueError):\n    risky()\n    del x\n",
            None,
            frozenset({"x"}),
            True,
            id="fn_global_with_risky_del_ignored",
        ),
        pytest.param(
            "with suppress(ValueError):\n    risky()\n    del x\n",
            {"x"},
            frozenset(),
            False,
            id="control_class_body_with_nonlocal_del_clears_definite",
        ),
        pytest.param(
            "try:\n    risky()\n    del x\nexcept ValueError:\n    pass\n",
            set(),
            frozenset(),
            True,
            id="class_body_try_del_ignored_without_nonlocal",
        ),
        pytest.param(
            "try:\n    risky()\n    del x\nexcept ValueError:\n    pass\n",
            {"x"},
            frozenset(),
            False,
            id="control_class_body_try_nonlocal_del_clears_definite",
        ),
        pytest.param(
            (
                "for _ in items:\n"
                "    try:\n"
                "        pass\n"
                "    except ValueError:\n"
                "        pass\n"
                "    else:\n"
                "        del x\n"
                "    finally:\n"
                "        break\n"
            ),
            set(),
            frozenset(),
            True,
            id="class_body_try_else_finally_abrupt_del_ignored_without_nonlocal",
        ),
        pytest.param(
            (
                "for _ in items:\n"
                "    try:\n"
                "        pass\n"
                "    except ValueError:\n"
                "        pass\n"
                "    else:\n"
                "        del x\n"
                "    finally:\n"
                "        break\n"
            ),
            {"x"},
            frozenset(),
            False,
            id="control_class_body_try_else_finally_nonlocal_del_clears_definite",
        ),
    ],
)
def test_scope_filtered_deletions_in_with_try_and_try_else_finally(
    body_src: str,
    active_nonlocals: set[str] | None,
    fn_globals: frozenset[str],
    expect_x_still_definite: bool,
) -> None:
    """Verifies class-local and global deletions inside with/try/try-else-finally do not clear enclosing definite locals."""
    from pydoppelgangerhunt.fixer._df_block_flow import (  # pylint: disable=import-outside-toplevel
        _extract_block_definite_bindings_and_term,
    )

    stmts = ast.parse(body_src).body
    def_out, _ = _extract_block_definite_bindings_and_term(
        stmts,
        active_nonlocals=active_nonlocals,
        initial_bound=frozenset({"x"}),
        fn_globals=fn_globals,
        suppress_aliases=("suppress",),
    )
    assert ("x" in def_out) is expect_x_still_definite, def_out

@pytest.mark.parametrize(
    ("body_src", "expect_v_definite", "expect_cm_in_sup", "expect_cm_in_nonsup"),
    [
        pytest.param(
            (
                "try:\n"
                "    cm = contextlib.nullcontext\n"
                "except ValueError:\n"
                "    cm = contextlib.suppress\n"
                "    raise\n"
            ),
            True,
            False,
            True,
            id="terminating_raise_handler_does_not_leak_suppressor",
        ),
        pytest.param(
            (
                "try:\n"
                "    cm = contextlib.nullcontext\n"
                "except ValueError:\n"
                "    cm = contextlib.suppress\n"
                "    return 0\n"
            ),
            True,
            False,
            True,
            id="terminating_return_handler_does_not_leak_suppressor",
        ),
        pytest.param(
            (
                "try:\n"
                "    cm = contextlib.nullcontext\n"
                "except ValueError:\n"
                "    cm = contextlib.suppress\n"
                "    raise\n"
                "finally:\n"
                "    log_done()\n"
            ),
            True,
            False,
            True,
            id="terminating_handler_with_non_terminating_finally_preserves_non_suppressor",
        ),
        pytest.param(
            (
                "try:\n"
                "    cm = contextlib.nullcontext\n"
                "except ValueError:\n"
                "    cm = contextlib.suppress\n"
            ),
            False,
            True,
            False,
            id="negative_surviving_handler_marks_cm_suppressor_and_clears_non_suppressor",
        ),
    ],
)
def test_try_except_terminating_handler_preserves_provenance_disjointness_and_definiteness(
    body_src: str,
    expect_v_definite: bool,
    expect_cm_in_sup: bool,
    expect_cm_in_nonsup: bool,
) -> None:
    """Verifies terminating except handlers neither leak suppressors nor break _SuppressProvenance disjointness."""
    from pydoppelgangerhunt.fixer._df_block_flow import (  # pylint: disable=import-outside-toplevel
        _analyze_block_assignment,
    )
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _update_suppress_provenance_from_stmts,
    )
    from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=import-outside-toplevel
        _EMPTY_SUPPRESS_PROVENANCE,
    )

    fn_src = (
        "import contextlib\n"
        "def run(step, log_done):\n"
        + "".join(f"    {ln}\n" for ln in body_src.splitlines())
        + "    with cm():\n"
        + "        v = step()\n"
        + "    return v\n"
    )
    mod = ast.parse(fn_src)
    fn_node = mod.body[1]
    assert isinstance(fn_node, ast.FunctionDef)
    init_prov = _update_suppress_provenance_from_stmts(
        [mod.body[0]], _EMPTY_SUPPRESS_PROVENANCE, in_branch=False
    )
    post_try_prov = _update_suppress_provenance_from_stmts(
        [fn_node.body[0]], init_prov, in_branch=False
    )
    assert (
        post_try_prov.suppressors
        & (post_try_prov.non_suppressors | post_try_prov.non_suppressor_instances)
        == frozenset()
    )
    assert ("cm" in post_try_prov.suppressors) is expect_cm_in_sup
    assert ("cm" in post_try_prov.non_suppressors) is expect_cm_in_nonsup

    def_block, _ = _analyze_block_assignment(
        fn_node.body,
        suppress_aliases=init_prov,
        strict_with_suppression=True,
    )
    def_after_try, _ = _analyze_block_assignment(
        fn_node.body[1:],
        suppress_aliases=post_try_prov,
        strict_with_suppression=True,
    )
    assert ("v" in def_block) is expect_v_definite
    assert ("v" in def_after_try) is expect_v_definite

def test_try_except_terminating_handler_retains_cross_scope_side_effect_for_outer_catch() -> None:
    """Verifies cross-scope mutator calls executed before raise inside a terminating handler remain tracked."""
    module_src = (
        "import contextlib\n"
        "cm = contextlib.nullcontext\n"
        "def corrupt():\n"
        "    global cm\n"
        "    cm = contextlib.suppress\n"
        "def run(step):\n"
        "    try:\n"
        "        try:\n"
        "            step()\n"
        "        except ValueError:\n"
        "            corrupt()\n"
        "            raise\n"
        "    except ValueError:\n"
        "        pass\n"
        "    with cm():\n"
        "        v = step()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": False, "pedantic": False}

@pytest.mark.parametrize(
    ("strict_with", "expect_definite", "expect_heuristics"),
    [
        pytest.param(False, True, {"cache.lock"}, id="strict_mode_single_pass"),
        pytest.param(True, False, set(), id="pedantic_mode_single_pass"),
    ],
)
def test_with_stmt_assignment_evaluates_header_items_in_single_pass(
    monkeypatch: pytest.MonkeyPatch,
    strict_with: bool,
    expect_definite: bool,
    expect_heuristics: set[str],
) -> None:
    """Verifies _analyze_with_stmt_assignment evaluates _iter_with_item_provenance_steps once per with statement."""
    from pydoppelgangerhunt.config import (
        collect_heuristic_ctx_managers,
        collect_wildcard_ctx_managers,
    )
    import pydoppelgangerhunt.fixer._df_block_flow as block_mod
    import pydoppelgangerhunt.fixer._df_provenance_walk as walk_mod

    orig_iter = walk_mod._iter_with_item_provenance_steps
    orig_wc = walk_mod.record_wildcard_ctx_manager_match
    iter_calls = 0
    wc_calls = 0

    def _spy_iter(*args: Any, **kwargs: Any) -> Any:
        nonlocal iter_calls
        iter_calls += 1
        return orig_iter(*args, **kwargs)

    def _spy_wc(*args: Any, **kwargs: Any) -> Any:
        nonlocal wc_calls
        wc_calls += 1
        return orig_wc(*args, **kwargs)

    monkeypatch.setattr(block_mod, "_iter_with_item_provenance_steps", _spy_iter)
    monkeypatch.setattr(walk_mod, "_iter_with_item_provenance_steps", _spy_iter)
    monkeypatch.setattr(walk_mod, "record_wildcard_ctx_manager_match", _spy_wc)

    tree = ast.parse(
        "with db.atomic(), cache.lock():\n"
        "    v = compute()\n"
    )
    with collect_heuristic_ctx_managers(isolate=True) as heur, collect_wildcard_ctx_managers(
        isolate=True
    ) as wc:
        definite, _cond = block_mod._analyze_block_assignment(
            tree.body,
            extra_non_suppressing_ctx_managers=frozenset({"*.atomic"}),
            strict_with_suppression=strict_with,
        )
    assert iter_calls == 1
    assert wc_calls == 1
    assert ("v" in definite) is expect_definite
    assert heur == expect_heuristics
    assert wc == {("*.atomic", "db.atomic")}

@pytest.mark.parametrize(
    (
        "with_header",
        "strict_with_suppression",
        "expect_nc_proven",
        "expect_ret_terminated",
    ),
    [
        pytest.param(
            "with Transaction():\n",
            False,
            True,
            True,
            id="strict_neutral_with_preserves_positive_binding_and_termination",
        ),
        pytest.param(
            "with Transaction():\n",
            True,
            False,
            False,
            id="pedantic_neutral_with_drops_positive_binding_and_termination",
        ),
        pytest.param(
            "with suppress_errors():\n",
            False,
            False,
            False,
            id="strict_suspicious_with_drops_positive_binding_and_termination",
        ),
    ],
)
def test_neutral_with_provenance_aligns_with_strict_and_pedantic_assignment_flow(
    with_header: str,
    strict_with_suppression: bool,
    expect_nc_proven: bool,
    expect_ret_terminated: bool,
) -> None:
    """Verifies _update_with_stmt_provenance agrees with _analyze_with_stmt_assignment on neutral vs suspicious managers."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment_impl
    from pydoppelgangerhunt.fixer._df_provenance_walk import _update_prov_and_term_from_stmts
    from pydoppelgangerhunt.fixer._df_suppress_prov import _EMPTY_SUPPRESS_PROVENANCE

    bind_stmts = ast.parse(
        with_header + "    from contextlib import nullcontext as nc\n"
    ).body
    prov_walk = _update_prov_and_term_from_stmts(
        bind_stmts,
        _EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
        strict_with_suppression=strict_with_suppression,
    )
    flow_res = _analyze_block_assignment_impl(
        bind_stmts,
        strict_with_suppression=strict_with_suppression,
    )
    assert ("nc" in prov_walk.prov.non_suppressors) is expect_nc_proven
    assert ("nc" in flow_res.prov.non_suppressors) is expect_nc_proven

    ret_stmts = ast.parse(with_header + "    work()\n    return 1\n").body
    ret_prov = _update_prov_and_term_from_stmts(
        ret_stmts,
        _EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
        strict_with_suppression=strict_with_suppression,
    )
    ret_flow = _analyze_block_assignment_impl(
        ret_stmts,
        strict_with_suppression=strict_with_suppression,
    )
    assert ret_prov.terminated is expect_ret_terminated
    assert ret_flow.terminated is expect_ret_terminated

def test_nested_try_block_assignment_single_pass_no_duplicate_walks_or_wildcards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _analyze_try_stmt_assignment runs in a single linear pass without duplicate walks or duplicate wildcard records."""
    import pydoppelgangerhunt.fixer._df_block_flow as block_mod
    import pydoppelgangerhunt.fixer._df_provenance_walk as walk_mod
    from pydoppelgangerhunt.config import collect_wildcard_ctx_managers

    def _build_nested_try(depth: int) -> str:
        lines: list[str] = []
        for idx in range(depth):
            ind = "    " * idx
            lines.append(f"{ind}try:")
            lines.append(f"{ind}    with db.atomic():")
            lines.append(f"{ind}        v_{idx} = step({idx})")
        leaf_ind = "    " * depth
        lines.append(f"{leaf_ind}leaf = done()")
        for idx in range(depth - 1, -1, -1):
            ind = "    " * idx
            lines.append(f"{ind}except ValueError:")
            lines.append(f"{ind}    v_{idx} = -1")
            lines.append(f"{ind}else:")
            lines.append(f"{ind}    with db.atomic():")
            lines.append(f"{ind}        e_{idx} = ok({idx})")
            lines.append(f"{ind}finally:")
            lines.append(f"{ind}    f_{idx} = 1")
        return "\n".join(lines) + "\n"

    depth = 12
    stmts = ast.parse(_build_nested_try(depth)).body

    split_prov_calls = 0
    walk_child_calls = 0
    with_step_calls = 0
    wildcard_record_calls = 0

    orig_split_prov = walk_mod._compute_try_body_split_provenance
    orig_walk_child = walk_mod._walk_child_stmts_prov
    orig_iter_steps = walk_mod._iter_with_item_provenance_steps
    orig_record_wc = walk_mod.record_wildcard_ctx_manager_match

    def _spy_split_prov(*args: Any, **kwargs: Any) -> Any:
        nonlocal split_prov_calls
        split_prov_calls += 1
        return orig_split_prov(*args, **kwargs)

    def _spy_walk_child(*args: Any, **kwargs: Any) -> Any:
        nonlocal walk_child_calls
        walk_child_calls += 1
        return orig_walk_child(*args, **kwargs)

    def _spy_iter_steps(*args: Any, **kwargs: Any) -> Any:
        nonlocal with_step_calls
        with_step_calls += 1
        return orig_iter_steps(*args, **kwargs)

    def _spy_record_wc(pattern: str, mgr_name: str) -> None:
        nonlocal wildcard_record_calls
        wildcard_record_calls += 1
        orig_record_wc(pattern, mgr_name)

    monkeypatch.setattr(walk_mod, "_compute_try_body_split_provenance", _spy_split_prov)
    monkeypatch.setattr(walk_mod, "_walk_child_stmts_prov", _spy_walk_child)
    monkeypatch.setattr(block_mod, "_iter_with_item_provenance_steps", _spy_iter_steps)
    monkeypatch.setattr(walk_mod, "_iter_with_item_provenance_steps", _spy_iter_steps)
    monkeypatch.setattr(walk_mod, "record_wildcard_ctx_manager_match", _spy_record_wc)

    with collect_wildcard_ctx_managers(isolate=True) as recorded:
        res = block_mod._analyze_block_assignment_impl(
            stmts,
            extra_non_suppressing_ctx_managers=frozenset({"*.atomic"}),
            strict_with_suppression=True,
        )

    assert split_prov_calls == 0
    assert walk_child_calls == 0
    assert with_step_calls == 2 * depth
    assert wildcard_record_calls == 2 * depth
    assert recorded == {("*.atomic", "db.atomic")}
    assert "f_0" in res.definite
    assert "v_0" in res.definite
    assert "e_0" in res.conditional
    assert res.terminated is False

    # Deep 20-level try nesting preserves definite/conditional results linearly
    deep_stmts = ast.parse(_build_nested_try(20)).body
    split_prov_calls = 0
    with_step_calls = 0
    deep_res = block_mod._analyze_block_assignment_impl(
        deep_stmts,
        extra_non_suppressing_ctx_managers=frozenset({"*.atomic"}),
        strict_with_suppression=True,
    )
    assert split_prov_calls == 0
    assert with_step_calls == 40
    assert {"f_0", "v_0"} <= deep_res.definite
    assert "e_0" in deep_res.conditional
