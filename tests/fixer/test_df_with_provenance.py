"""Tests for _df_with_provenance context-manager suppression, allowlists, and definite-binding termination."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
from pathlib import Path
import sys
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    generate_refactoring_patch,
    inspect_single_unit_scope,
)


@pytest.mark.parametrize(
    ("case_id", "src", "expect_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "shadowed_eval_param_assigned_to_flag_and_unrelated_call",
            (
                "class Mgr:\n"
                "    def __enter__(self): return self\n"
                "    def __exit__(self, *a): return False\n"
                "\n"
                "def configure(eval: bool = False):\n"
                "    flag = eval\n"
                "    return flag\n"
                "\n"
                "def run():\n"
                "    return flag()\n"
            ),
            False,
            frozenset(),
            id="shadowed_eval_param_assigned_to_flag_and_unrelated_call",
        ),
        pytest.param(
            "shadowed_eval_and_exec_params_called",
            (
                "def run_hook(eval):\n"
                "    return eval(1)\n"
                "\n"
                "def run_local():\n"
                "    exec = lambda cmd: cmd\n"
                "    return exec('ok')\n"
            ),
            False,
            frozenset(),
            id="shadowed_eval_and_exec_params_called",
        ),
        pytest.param(
            "shadowed_builtins_importlib_sys_and_attr_helpers",
            (
                "def configure(builtins=None, importlib=None, sys=None, vars=None, globals=None, getattr=None, setattr=None, delattr=None, obj=None, fn=None):\n"
                "    builtins.setattr(obj, '__exit__', fn)\n"
                "    builtins.eval('1')\n"
                "    importlib.import_module('builtins')\n"
                "    vars(builtins)\n"
                "    globals()['__builtins__']\n"
                "    sys.modules['builtins']\n"
                "    getattr(builtins, 'eval')\n"
                "    setattr(obj, '__exit__', fn)\n"
                "    delattr(obj, '__exit__')\n"
            ),
            False,
            frozenset(),
            id="shadowed_builtins_importlib_sys_and_attr_helpers",
        ),
        pytest.param(
            "shadowed_type_and_object_dunder_refs_and_comprehension_shadow",
            (
                "def clean(type, object):\n"
                "    type.__setattr__('label', 1)\n"
                "    object.__delattr__('label')\n"
                "    vals = [eval(i) for eval in (lambda x: x,) for i in (1, 2)]\n"
                "    fn = lambda setattr: setattr(object(), '__exit__', type)\n"
                "    return vals, fn\n"
            ),
            False,
            frozenset(),
            id="shadowed_type_and_object_dunder_refs_and_comprehension_shadow",
        ),
        pytest.param(
            "pure_setattr_safe_attr_only",
            (
                "class Mgr:\n"
                "    pass\n"
                "\n"
                "def setup():\n"
                "    sa = setattr\n"
                "    sa(Mgr, 'safe_flag', True)\n"
            ),
            False,
            frozenset(),
            id="pure_setattr_safe_attr_only",
        ),
        pytest.param(
            "class_body_assign_setattr_exposes_class_attr",
            (
                "class C:\n"
                "    sa = setattr\n"
                "    sa(Obj, 'safe_label', 1)\n"
            ),
            True,
            frozenset(),
            id="class_body_assign_setattr_exposes_class_attr",
        ),
        pytest.param(
            "positive_module_alias_eval_call",
            (
                "flag = eval\n"
                "\n"
                "def run():\n"
                "    return flag('1 + 1')\n"
            ),
            True,
            frozenset(),
            id="positive_module_alias_eval_call",
        ),
        pytest.param(
            "positive_nested_from_builtins_import_eval_inside_shadowed_outer",
            (
                "def outer():\n"
                "    eval = False\n"
                "    def inner():\n"
                "        from builtins import eval\n"
                "        return eval('1')\n"
                "    return eval, inner\n"
            ),
            True,
            frozenset(),
            id="positive_nested_from_builtins_import_eval_inside_shadowed_outer",
        ),
        pytest.param(
            "positive_walrus_import_builtins_eval_call",
            (
                "def trigger():\n"
                "    b = (mod := __import__('builtins'))\n"
                "    return b.eval('1')\n"
            ),
            True,
            frozenset(),
            id="positive_walrus_import_builtins_eval_call",
        ),
        pytest.param(
            "positive_nested_global_eval_inside_shadowed_outer",
            (
                "def outer(eval: bool = False):\n"
                "    def inner():\n"
                "        global eval\n"
                "        return eval('1')\n"
                "    return eval, inner\n"
            ),
            True,
            frozenset(),
            id="positive_nested_global_eval_inside_shadowed_outer",
        ),
        pytest.param(
            "positive_class_body_setattr_alias_targets_manager",
            (
                "class Mgr:\n"
                "    pass\n"
                "\n"
                "class Patcher:\n"
                "    sa = setattr\n"
                "    sa(Mgr, '__exit__', lambda *a: True)\n"
            ),
            True,
            frozenset({"Mgr"}),
            id="positive_class_body_setattr_alias_targets_manager",
        ),
    ],
)
def test_exit_protocol_lexical_slot_shadowing_and_aliases(
    case_id: str,
    src: str,
    expect_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies lexical-slot isolation prevents shadowed parameter pollution while preserving real alias detection."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    tree = ast.parse(src)
    scan = scan_exit_protocol_patches(tree)
    assert scan.has_patch is expect_has_patch, case_id
    assert expected_patched_names <= scan.patched_names, case_id


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

    if sys.version_info >= (3, 10):
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
    assert "late_sup" not in prov_mid.suppressors

    prov_after_sup = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=7)
    assert "nullcontext" not in prov_after_sup.non_suppressors
    assert "late_sup" in prov_after_sup.suppressors

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


def test_comprehension_walrus_bindings_and_provenance_updates() -> None:
    """Verifies comprehension walrus expressions bind conditionally and update context-manager provenance."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _analyze_block_assignment,
        _classify_with_suppression,
        _seed_scope_suppress_aliases,
        _update_suppress_provenance_from_expr_walrus,
    )

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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
        _extract_block_definite_bindings_and_term,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        stmts = ast.parse(body_src).body
    may_del: set[str] = set()
    live_in_del: set[str] = set()
    static_def, static_cond, static_term, _ = _analyze_block_assignment_impl(
        stmts,
        definite=set(pre_init),
        may_deleted=may_del,
        live_in_deleted=live_in_del,
    )
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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
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
    static_def, static_cond, static_term, _ = _analyze_block_assignment_impl(
        stmts,
        definite=set(pre_init),
        may_deleted=may_del,
    )
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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
    )

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
                    s_def, s_cond, s_term, _ = _analyze_block_assignment_impl(
                        stmts,
                        definite=set(pre_init),
                        may_deleted=may_del,
                    )
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


def _late_bound_capture_safety_by_mode(src: str, unit_src: str) -> dict[str, bool]:
    """Returns {mode: is_control_flow_safe} for a single-line lambda unit capturing a with-bound name."""
    unit_line = next(
        idx for idx, ln in enumerate(src.splitlines(), start=1) if ln == unit_src
    )
    tree = ast.parse(src)
    results: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": src,
                "source_code": unit_src + "\n",
                "closure_strictness": mode,
            },
            tree=tree,
        )
        results[mode] = bool(sc["is_control_flow_safe"])
    return results


_PROVEN_EXIT_CLASS = (
    "class Mgr:\n"
    "    def __enter__(self):\n"
    "        return self\n"
    "    def __exit__(self, *exc_info):\n"
    "        return False\n"
)
_WITH_MGR_RUN = (
    "def run(compute):\n"
    "    with Mgr():\n"
    "        v = compute()\n"
    "    cb = lambda: v\n"
    "    return cb()\n"
)
_WITH_NULLCONTEXT_RUN = (
    "def run(compute):\n"
    "    with contextlib.nullcontext():\n"
    "        v = compute()\n"
    "    cb = lambda: v\n"
    "    return cb()\n"
)


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
    ("patch_stmt", "expect_has_patch"),
    [
        pytest.param(
            'import builtins\nbuiltins.setattr(Mgr, "__exit__", swallow)\n',
            True,
            id="builtins_setattr_exit",
        ),
        pytest.param(
            's = setattr\ns(Mgr, "__exit__", swallow)\n',
            True,
            id="aliased_setattr",
        ),
        pytest.param(
            'import functools\nfunctools.partial(setattr, Mgr, "__exit__")(swallow)\n',
            True,
            id="functools_partial_setattr",
        ),
        pytest.param(
            'import builtins\ngetattr(builtins, "setattr")(Mgr, "__exit__", swallow)\n',
            True,
            id="getattr_builtins_setattr",
        ),
        pytest.param(
            'import builtins\nbuiltins.setattr(Mgr, "label", "safe")\n',
            False,
            id="builtins_setattr_unrelated_literal_attr",
        ),
    ],
)
def test_exit_protocol_scanner_catches_indirect_setattr_bypasses(
    patch_stmt: str, expect_has_patch: bool
) -> None:
    """Verifies indirect setattr forms (builtins.setattr, aliases, partial, getattr) are caught without false positives."""
    import ast

    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    src = (
        "class Mgr:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "def swallow(*a): return True\n"
        f"{patch_stmt}"
    )
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


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
    ("patch_stmt", "expect_has_patch", "expect_patched_mgr"),
    [
        pytest.param(
            'from builtins import exec as run_code\nrun_code("Mgr.__exit__ = swallow")\n',
            True,
            False,
            id="imported_exec_alias",
        ),
        pytest.param(
            'from builtins import eval as eval_code\neval_code("Mgr.__exit__")\n',
            True,
            False,
            id="imported_eval_alias",
        ),
        pytest.param(
            'from builtins import setattr as patch_attr\npatch_attr(Mgr, "__exit__", swallow)\n',
            True,
            True,
            id="imported_setattr_alias",
        ),
        pytest.param(
            'from builtins import delattr as drop_attr\ndrop_attr(Mgr, "__exit__")\n',
            True,
            True,
            id="imported_delattr_alias",
        ),
        pytest.param(
            'from builtins import setattr as pa\nrebound = pa\nrebound(Mgr, "__exit__", swallow)\n',
            True,
            True,
            id="imported_setattr_chained_assignment_alias",
        ),
        pytest.param(
            'from builtins import setattr as patch_attr\npatch_attr(Mgr, "label", "safe")\n',
            False,
            False,
            id="imported_setattr_alias_unrelated_literal_attr",
        ),
    ],
)
def test_exit_protocol_scanner_resolves_imported_builtin_aliases(
    patch_stmt: str, expect_has_patch: bool, expect_patched_mgr: bool
) -> None:
    """Verifies imported aliases of exec/eval/setattr/delattr and chained assignments are resolved."""
    import ast

    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    src = (
        "class Mgr:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "def swallow(*a): return True\n"
        f"{patch_stmt}"
    )
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch
    assert ("Mgr" in scan.patched_names) is expect_patched_mgr


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

    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

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


@pytest.mark.parametrize(
    ("src", "expect_has_patch"),
    [
        ("class C:\n    def m(self, k, v):\n        setattr(self, k, v)\n", True),
        ("class C:\n    def m(self, v):\n        setattr(self, 'safe_field', v)\n", False),
        ("import builtins\nclass C:\n    def m(self, k, v):\n        builtins.setattr(self, k, v)\n", True),
        ("class C:\n    def m(self, v):\n        setattr(self, '__class__', v)\n", True),
        ("class C:\n    @classmethod\n    def m(cls, k, v):\n        setattr(cls, k, v)\n", True),
        ("__import__('builtins').setattr(M, '__exit__', f)\n", True),
        ("imp = __import__\nimp('builtins').setattr(M, '__exit__', f)\n", True),
        ("import importlib\nimportlib.import_module('builtins').setattr(M, '__exit__', f)\n", True),
        ("import builtins\nvars(builtins)['setattr'](M, '__exit__', f)\n", True),
        ("import builtins\nbuiltins.__dict__['setattr'](M, '__exit__', f)\n", True),
        ("__builtins__['setattr'](M, '__exit__', f)\n", True),
        ("import builtins\nname = 'setattr'\ngetattr(builtins, name)(M, '__exit__', f)\n", True),
        ("import sys\ns = sys\ns.modules['builtins'].setattr(M, '__exit__', f)\n", True),
    ],
)
def test_exit_protocol_scanner_self_setattr_and_dynamic_builtins_lookup(
    src: str, expect_has_patch: bool
) -> None:
    """Verifies dynamic setattr on self/builtins and dynamic builtins namespace lookups invalidate proofs."""
    import ast

    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


@pytest.mark.parametrize(
    ("include_self_setattr", "patch_stmt", "expected_pedantic_safe", "expected_strict_safe"),
    [
        pytest.param(False, "    pass", True, True, id="clean_control"),
        pytest.param(
            True,
            "    cm.change('__class__', Suppressing)",
            False,
            False,
            id="builtins_setattr_self_dynamic_attr_swap",
        ),
        pytest.param(
            False,
            "    name = 'setattr'\n    getattr(builtins, name)(Mgr, '__exit__', lambda *a: True)",
            False,
            None,
            id="dynamic_getattr_builtins_setattr",
        ),
        pytest.param(
            False,
            "    builtins.__dict__['setattr'](Mgr, '__exit__', lambda *a: True)",
            False,
            None,
            id="builtins_dunder_dict_setattr",
        ),
        pytest.param(
            False,
            "    __builtins__['setattr'](Mgr, '__exit__', lambda *a: True)",
            False,
            None,
            id="dunder_builtins_subscript_setattr",
        ),
    ],
)
def test_exit_protocol_dynamic_lookup_and_self_setattr_withdraws_provenance_proof(
    include_self_setattr: bool,
    patch_stmt: str,
    expected_pedantic_safe: bool,
    expected_strict_safe: bool | None,
) -> None:
    """Verifies builtins.setattr(self, name, cls) and dynamic builtins lookups withdraw class non-suppression proofs."""
    change_method = (
        "    def change(self, name, cls):\n"
        "        builtins.setattr(self, name, cls)\n"
        if include_self_setattr
        else ""
    )
    src = (
        "import builtins\n"
        "class Suppressing:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return True\n"
        "class Mgr:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        f"{change_method}"
        "def run(compute):\n"
        "    cm = Mgr()\n"
        f"{patch_stmt}\n"
        "    with cm:\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["pedantic"] is expected_pedantic_safe
    if expected_strict_safe is not None:
        assert safety["strict"] is expected_strict_safe


@pytest.mark.parametrize(
    ("src", "expect_has_patch"),
    [
        pytest.param(
            "import builtins\nbuiltins.vars(builtins)['setattr'](M, '__exit__', f)\n",
            True,
            id="builtins_dot_vars",
        ),
        pytest.param(
            "import builtins\nget_vars = vars\nget_vars(builtins)['setattr'](M, '__exit__', f)\n",
            True,
            id="aliased_vars",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module(name='builtins').setattr(M, '__exit__', f)\n",
            True,
            id="importlib_kwarg_name",
        ),
        pytest.param(
            "b = globals()['__builtins__']\nb['setattr'](M, '__exit__', f)\n",
            True,
            id="globals_subscript_builtins",
        ),
        pytest.param(
            "b = globals().get('__builtins__')\nb.setattr(M, '__exit__', f)\n",
            True,
            id="globals_get_builtins",
        ),
        pytest.param(
            "__import__(name='builtins').setattr(M, '__exit__', f)\n",
            True,
            id="dunder_import_kwarg_name",
        ),
        pytest.param(
            "import builtins\ngetattr(builtins, 'vars')(builtins)['setattr'](M, '__exit__', f)\n",
            True,
            id="getattr_builtins_vars",
        ),
        pytest.param(
            "import importlib\nimportlib.import_module(name='math').sin(1)\n",
            False,
            id="negative_importlib_kwarg_math",
        ),
        pytest.param(
            "g = globals()\ng['helper'](1)\n",
            False,
            id="negative_globals_regular_key",
        ),
        pytest.param(
            "v = vars(obj)\nv['x'] = 1\n",
            False,
            id="negative_vars_regular_obj",
        ),
    ],
)
def test_exit_protocol_scanner_extended_vars_importlib_and_globals(
    src: str, expect_has_patch: bool
) -> None:
    """Verifies extended vars/importlib/globals builtins lookups are recognized uniformly."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


@pytest.mark.parametrize(
    ("patch_stmt", "expected_pedantic_safe"),
    [
        pytest.param("    pass", True, id="clean_control"),
        pytest.param(
            "    builtins.vars(builtins)['setattr'](Mgr, '__exit__', lambda *a: True)",
            False,
            id="builtins_vars_setattr",
        ),
        pytest.param(
            "    get_vars = vars\n    get_vars(builtins)['setattr'](Mgr, '__exit__', lambda *a: True)",
            False,
            id="vars_alias_setattr",
        ),
        pytest.param(
            "    importlib.import_module(name='builtins').setattr(Mgr, '__exit__', lambda *a: True)",
            False,
            id="import_module_kwarg_builtins_setattr",
        ),
        pytest.param(
            "    globals()['__builtins__']['setattr'](Mgr, '__exit__', lambda *a: True)",
            False,
            id="globals_builtins_setattr",
        ),
    ],
)
def test_exit_protocol_extended_builtins_lookup_withdraws_provenance_proof(
    patch_stmt: str, expected_pedantic_safe: bool
) -> None:
    """Verifies extended dynamic builtins lookups withdraw class non-suppression proofs."""
    src = (
        "import builtins\n"
        "import importlib\n"
        "class Mgr:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "def run(compute):\n"
        "    cm = Mgr()\n"
        f"{patch_stmt}\n"
        "    with cm:\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["pedantic"] is expected_pedantic_safe


def test_seed_scope_suppress_aliases_stops_before_u_start_and_post_unit() -> None:
    """Verifies pre-unit provenance seeding stops before u_start in linear/if blocks while remaining sound across loops."""
    from pydoppelgangerhunt.fixer._df_with_provenance import _seed_scope_suppress_aliases

    # 1. Loop body can re-enter after post-unit rebind, so loop body invalidates cm conservatively
    loop_src = (
        "import contextlib\n"
        "def run_loop(compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    for _ in range(1):\n"
        "        with cm():\n"
        "            val = compute()\n"
        "        cm = custom_unknown\n"
        "    return val\n"
    )
    loop_mod = ast.parse(loop_src)
    loop_fn = loop_mod.body[1]
    assert isinstance(loop_fn, ast.FunctionDef)
    prov_loop = _seed_scope_suppress_aliases(loop_fn, loop_mod, u_start=5)
    assert "cm" not in prov_loop.non_suppressors

    # 2. Single-pass `if` branch: post-unit `cm = custom_unknown` does not pre-invalidate u_start=5
    if_src = (
        "import contextlib\n"
        "def run_if(cond, compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    if cond:\n"
        "        with cm() as cm:\n"
        "            val = compute()\n"
        "        cm = custom_unknown\n"
        "    return val\n"
    )
    if_mod = ast.parse(if_src)
    if_fn = if_mod.body[1]
    assert isinstance(if_fn, ast.FunctionDef)
    prov_if_before = _seed_scope_suppress_aliases(if_fn, if_mod, u_start=5)
    assert "cm" in prov_if_before.non_suppressors
    prov_if_after = _seed_scope_suppress_aliases(if_fn, if_mod, u_start=8)
    assert "cm" not in prov_if_after.non_suppressors

    # 3. Direct sequential statements: `with cm() as cm:` and post-unit rebind must not pre-invalidate u_start=4
    direct_src = (
        "import contextlib\n"
        "def run_direct(compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    with cm() as cm:\n"
        "        val = compute()\n"
        "    cm = custom_unknown\n"
        "    return val\n"
    )
    direct_mod = ast.parse(direct_src)
    direct_fn = direct_mod.body[1]
    assert isinstance(direct_fn, ast.FunctionDef)
    prov_direct = _seed_scope_suppress_aliases(direct_fn, direct_mod, u_start=4)
    assert "cm" in prov_direct.non_suppressors
    prov_direct_late = _seed_scope_suppress_aliases(direct_fn, direct_mod, u_start=7)
    assert "cm" not in prov_direct_late.non_suppressors


@pytest.mark.parametrize(
    ("src", "expect_has_patch"),
    [
        pytest.param(
            "def f(eval: bool = False):\n    return 1 if eval else 0\n",
            False,
            id="param_shadows_eval",
        ),
        pytest.param(
            "def f():\n    setattr = object()\n    return setattr\n",
            False,
            id="local_shadows_setattr",
        ),
        pytest.param(
            "def f(flags):\n    return [eval for eval in flags]\n",
            False,
            id="comp_target_shadows_eval",
        ),
        pytest.param(
            "def outer():\n    eval = False\n    class Inner:\n        def m(self):\n            return eval\n",
            False,
            id="enclosing_fn_shadows_eval_inside_class_method",
        ),
        pytest.param(
            "s = setattr\ns(obj, 'normal_label', 1)\n",
            False,
            id="benign_setattr_alias_safe_attr",
        ),
        pytest.param(
            "import builtins\ns = builtins.setattr\ns(obj, 'normal_label', 1)\n",
            False,
            id="benign_builtins_setattr_alias_safe_attr",
        ),
        pytest.param(
            "s1 = setattr\ns2 = s1\ns2(obj, 'normal_label', 1)\n",
            False,
            id="chained_benign_setattr_aliases",
        ),
        pytest.param(
            "(s := setattr)(obj, 'normal_label', 1)\n",
            False,
            id="benign_walrus_setattr_call",
        ),
        pytest.param(
            "s = setattr\ndef unrelated(s):\n    return s\ns(obj, 'normal_label', 1)\n",
            False,
            id="unrelated_param_with_same_alias_name",
        ),
        pytest.param(
            "def f(x=eval):\n    return x\n",
            True,
            id="negative_default_eval_in_enclosing_scope",
        ),
        pytest.param(
            "class Config:\n    eval = False\n    def m(self):\n        return eval\n",
            True,
            id="negative_pep227_class_attr_does_not_shadow_method_eval",
        ),
        pytest.param(
            "def outer():\n    eval = False\n    def inner():\n        global eval\n        return eval\n",
            True,
            id="negative_global_eval_bypasses_outer_shadow",
        ),
        pytest.param(
            "cb(setattr)\n",
            True,
            id="negative_uncalled_setattr_arg_escapes",
        ),
        pytest.param(
            "s = setattr\ncb(s)\n",
            True,
            id="negative_alias_escapes_in_arg",
        ),
        pytest.param(
            "s1 = setattr\ns2 = s1\ncb(s2)\n",
            True,
            id="negative_chained_alias_escapes",
        ),
        pytest.param(
            "class Box:\n    s = setattr\n",
            True,
            id="negative_class_body_alias_exposes_attr",
        ),
        pytest.param(
            "f((s := setattr))\n",
            True,
            id="negative_walrus_alias_escapes_as_arg",
        ),
        pytest.param(
            "def f(cond):\n    s = setattr\n    if cond:\n        s = other_fn\n    s(obj, 'normal_label', 1)\n",
            True,
            id="negative_local_alias_rebound",
        ),
        pytest.param(
            "s = eval\n",
            True,
            id="negative_exec_eval_alias_not_exempt",
        ),
    ],
)
def test_exit_protocol_shadowed_names_and_benign_setattr_aliases(
    src: str, expect_has_patch: bool
) -> None:
    """Verifies shadowed local non-builtins and call-only safe setattr aliases avoid false patch alarms while preserving escape detection."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


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
    ("patch_stmt", "expected_names", "expected_strict_safe"),
    [
        pytest.param(
            "getattr(builtins, 'setattr')(Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="getattr_builtins_setattr",
        ),
        pytest.param(
            "vars(builtins)['setattr'](Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="vars_builtins_subscript_setattr",
        ),
        pytest.param(
            "vars(builtins).get('setattr')(Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="vars_builtins_get_setattr",
        ),
        pytest.param(
            "__builtins__['setattr'](Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="dunder_builtins_subscript_setattr",
        ),
        pytest.param(
            "importlib.import_module('builtins').setattr(Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="importlib_builtins_attr_setattr",
        ),
        pytest.param(
            "(s := setattr)(Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="walrus_setattr_call",
        ),
        pytest.param(
            "s = getattr(builtins, 'setattr')\ns(Mgr, '__exit__', lambda *a: True)",
            {"Mgr"},
            False,
            id="alias_from_getattr_builtins_setattr",
        ),
        pytest.param(
            "getattr(builtins, 'setattr')(Mgr, 'normal_label', 1)",
            set(),
            True,
            id="control_getattr_builtins_safe_attr",
        ),
        pytest.param(
            "vars(builtins)['setattr'](Other, '__exit__', lambda *a: True)",
            {"Other"},
            True,
            id="control_unrelated_target_keeps_mgr_unpatched_in_strict",
        ),
    ],
)
def test_exit_protocol_indirect_and_walrus_setattr_record_patched_names(
    patch_stmt: str, expected_names: set[str], expected_strict_safe: bool
) -> None:
    """Verifies indirect and walrus setattr calls targeting exit protocols populate patched_names for strict mode."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    patch_indented = "\n".join(f"    {line}" for line in patch_stmt.splitlines())
    src = (
        "import builtins\n"
        "import importlib\n"
        "class Other:\n"
        "    pass\n"
        "class Mgr:\n"
        "    def __enter__(self): return self\n"
        "    def __exit__(self, *a): return False\n"
        "def run(compute):\n"
        "    cm = Mgr()\n"
        f"{patch_indented}\n"
        "    with cm:\n"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert expected_names.issubset(scan.patched_names)
    if not expected_names:
        assert "Mgr" not in scan.patched_names
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["strict"] is expected_strict_safe


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
    ("case_id", "patch_block"),
    [
        pytest.param(
            "builtin_setattr_alias_setter",
            "setter = setattr\nsetter(Mgr, '__exit__', lambda *a: True)",
            id="builtin_setattr_alias",
        ),
        pytest.param(
            "builtins_module_alias_setattr",
            "b = builtins\nb.setattr(Mgr, '__exit__', lambda *a: True)",
            id="builtins_module_alias",
        ),
        pytest.param(
            "global_setattr_call",
            "global setattr\nsetattr(Mgr, '__exit__', lambda *a: True)",
            id="global_setattr_call",
        ),
        pytest.param(
            "dynamic_attribute_name_setattr",
            "attr_name = '__exit__'\nsetattr(Mgr, attr_name, lambda *a: True)",
            id="dynamic_attr_name_setattr",
        ),
        pytest.param(
            "rebind_and_restore_exit_protocol",
            "orig = Mgr.__exit__\nMgr.__exit__ = lambda *a: True\nMgr.__exit__ = orig",
            id="rebind_and_restore_exit_protocol",
        ),
        pytest.param(
            "imported_builtins_alias_setattr",
            "from builtins import setattr as patch_attr\npatch_attr(Mgr, '__exit__', lambda *a: True)",
            id="imported_builtins_alias_setattr",
        ),
    ],
)
def test_exit_protocol_aliases_globals_dynamic_names_and_restoration(
    case_id: str,
    patch_block: str,
) -> None:
    """Verifies runtime exit-protocol invalidation catches aliases, globals, dynamic attr names, and restore pairs."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    indented = "\n".join(f"    {ln}" for ln in patch_block.splitlines())
    src = (
        "import builtins\n"
        + _PROVEN_EXIT_CLASS
        + "def patch_mgr():\n"
        + f"{indented}\n"
        + _WITH_MGR_RUN
    )
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is True, case_id
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["pedantic"] is False, (case_id, safety)
    assert safety["strict"] is False, (case_id, safety)


@pytest.mark.parametrize(
    ("case_id", "src", "expect_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "pep227_module_eval_alias_method_self_name",
            (
                "flag = eval\n"
                "class Runner:\n"
                "    def flag(self, cmd: str):\n"
                "        return flag(cmd)\n"
            ),
            True,
            frozenset(),
            id="pep227_module_eval_alias_method_self_name",
        ),
        pytest.param(
            "pep227_enclosing_fn_eval_alias_method_self_name",
            (
                "def outer():\n"
                "    flag = eval\n"
                "    class Runner:\n"
                "        def flag(self, cmd: str):\n"
                "            return flag(cmd)\n"
                "    return Runner\n"
            ),
            True,
            frozenset(),
            id="pep227_enclosing_fn_eval_alias_method_self_name",
        ),
        pytest.param(
            "pep227_method_named_eval_calls_builtin_eval",
            (
                "class Runner:\n"
                "    def eval(self, cmd: str):\n"
                "        return eval(cmd)\n"
            ),
            True,
            frozenset(),
            id="pep227_method_named_eval_calls_builtin_eval",
        ),
        pytest.param(
            "pep227_method_named_sa_calls_outer_setattr",
            (
                "sa = setattr\n"
                "class Runner:\n"
                "    def sa(self, mgr):\n"
                "        sa(mgr, '__exit__', lambda *a: True)\n"
            ),
            True,
            frozenset({"mgr"}),
            id="pep227_method_named_sa_calls_outer_setattr",
        ),
        pytest.param(
            "control_nested_recursive_fn_shadows_module_setattr_alias",
            (
                "sa = setattr\n"
                "def outer():\n"
                "    def sa(n):\n"
                "        return sa(n - 1) if n > 0 else 0\n"
                "    return sa\n"
                "sa(obj, 'normal_label', 1)\n"
            ),
            False,
            frozenset(),
            id="control_nested_recursive_fn_shadows_module_setattr_alias",
        ),
        pytest.param(
            "control_nested_recursive_fn_named_eval_shadows_builtin",
            (
                "def outer():\n"
                "    def eval(n: int) -> int:\n"
                "        return eval(n - 1) if n > 0 else 0\n"
                "    return eval(2)\n"
            ),
            False,
            frozenset(),
            id="control_nested_recursive_fn_named_eval_shadows_builtin",
        ),
        pytest.param(
            "control_unrelated_method_self_name",
            (
                "class Runner:\n"
                "    def flag(self, cmd: str):\n"
                "        return cmd.strip()\n"
            ),
            False,
            frozenset(),
            id="control_unrelated_method_self_name",
        ),
        pytest.param(
            "class_body_walrus_setattr_exposes_class_attr",
            (
                "class Box:\n"
                "    _ = (s := setattr)(obj, 'normal_label', 1)\n"
            ),
            True,
            frozenset(),
            id="class_body_walrus_setattr_exposes_class_attr",
        ),
        pytest.param(
            "nested_fn_class_body_walrus_setattr_exposes_class_attr",
            (
                "def outer(obj):\n"
                "    class Box:\n"
                "        _ = (s := setattr)(obj, 'normal_label', 1)\n"
                "    return Box\n"
            ),
            True,
            frozenset(),
            id="nested_fn_class_body_walrus_setattr_exposes_class_attr",
        ),
        pytest.param(
            "class_body_walrus_setattr_alias_then_patches_exit",
            (
                "class Mgr:\n"
                "    pass\n"
                "class Box:\n"
                "    _ = (s := setattr)(Mgr, 'normal_label', 1)\n"
                "    _p = s(Mgr, '__exit__', lambda *a: True)\n"
            ),
            True,
            frozenset({"Mgr"}),
            id="class_body_walrus_setattr_alias_then_patches_exit",
        ),
        pytest.param(
            "control_function_local_walrus_setattr_safe_attr",
            (
                "def helper(obj):\n"
                "    return (s := setattr)(obj, 'normal_label', 1)\n"
            ),
            False,
            frozenset(),
            id="control_function_local_walrus_setattr_safe_attr",
        ),
    ],
)
def test_exit_protocol_pep227_method_self_name_and_class_walrus(
    case_id: str,
    src: str,
    expect_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies PEP 227 method self-name scoping and class-body walrus attribute disqualification."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch, case_id
    assert expected_patched_names <= scan.patched_names, case_id


def test_exit_protocol_pep227_and_class_walrus_withdraw_proven_cm_proofs() -> None:
    """Verifies PEP 227 outer eval/setattr calls and class-body walrus setattr leaks withdraw Mgr proofs."""
    src_pep227_eval = (
        _PROVEN_EXIT_CLASS
        + "flag = eval\n"
        + "class Runner:\n"
        + "    def flag(self, cmd: str):\n"
        + "        return flag(cmd)\n"
        + _WITH_MGR_RUN
    )
    safety_pep227_eval = _late_bound_capture_safety_by_mode(src_pep227_eval, "    cb = lambda: v")
    assert safety_pep227_eval["pedantic"] is False
    assert safety_pep227_eval["strict"] is True

    src_pep227_targeted = (
        _PROVEN_EXIT_CLASS
        + "sa = setattr\n"
        + "class Runner:\n"
        + "    def sa(self):\n"
        + "        sa(Mgr, '__exit__', lambda *a: True)\n"
        + _WITH_MGR_RUN
    )
    safety_pep227_targeted = _late_bound_capture_safety_by_mode(
        src_pep227_targeted, "    cb = lambda: v"
    )
    assert safety_pep227_targeted["pedantic"] is False
    assert safety_pep227_targeted["strict"] is False

    src_walrus_exposed = (
        _PROVEN_EXIT_CLASS
        + "class Box:\n"
        + "    _ = (s := setattr)(Mgr, 'normal_label', 1)\n"
        + _WITH_MGR_RUN
    )
    safety_walrus_exposed = _late_bound_capture_safety_by_mode(
        src_walrus_exposed, "    cb = lambda: v"
    )
    assert safety_walrus_exposed["pedantic"] is False
    assert safety_walrus_exposed["strict"] is True

    src_walrus_targeted = (
        _PROVEN_EXIT_CLASS
        + "class Box:\n"
        + "    _ = (s := setattr)(Mgr, 'normal_label', 1)\n"
        + "    _p = s(Mgr, '__exit__', lambda *a: True)\n"
        + _WITH_MGR_RUN
    )
    safety_walrus_targeted = _late_bound_capture_safety_by_mode(
        src_walrus_targeted, "    cb = lambda: v"
    )
    assert safety_walrus_targeted["pedantic"] is False
    assert safety_walrus_targeted["strict"] is False

    src_safe_control = (
        _PROVEN_EXIT_CLASS
        + "class Other:\n"
        + "    pass\n"
        + "sa = setattr\n"
        + "def outer():\n"
        + "    def sa(n):\n"
        + "        return sa(n - 1) if n > 0 else 0\n"
        + "    return sa\n"
        + "sa(Other, 'normal_label', 1)\n"
        + _WITH_MGR_RUN
    )
    safety_safe = _late_bound_capture_safety_by_mode(src_safe_control, "    cb = lambda: v")
    assert safety_safe["pedantic"] is True
    assert safety_safe["strict"] is True


@pytest.mark.parametrize(
    ("case_id", "src", "expect_has_patch"),
    [
        pytest.param(
            "module_unused_setattr_alias_escapes",
            "s = setattr\n",
            True,
            id="module_unused_setattr_alias_escapes",
        ),
        pytest.param(
            "local_unused_delattr_alias_escapes",
            "def setup():\n    d = delattr\n    return 0\n",
            True,
            id="local_unused_delattr_alias_escapes",
        ),
        pytest.param(
            "chained_unused_setattr_aliases_escape",
            "s1 = setattr\ns2 = s1\n",
            True,
            id="chained_unused_setattr_aliases_escape",
        ),
        pytest.param(
            "walrus_called_then_alias_passed_to_callback_escapes",
            "def f(cb, obj):\n    _ = (s := setattr)(obj, 'normal_label', 1)\n    cb(s)\n",
            True,
            id="walrus_called_then_alias_passed_to_callback_escapes",
        ),
        pytest.param(
            "safe_called_only_setattr_alias",
            "s = setattr\ns(obj, 'normal_label', 1)\n",
            False,
            id="safe_called_only_setattr_alias",
        ),
        pytest.param(
            "safe_direct_walrus_setattr_call",
            "def f(obj):\n    return (s := setattr)(obj, 'normal_label', 1)\n",
            False,
            id="safe_direct_walrus_setattr_call",
        ),
    ],
)
def test_exit_protocol_unreferenced_setattr_aliases_escape(
    case_id: str,
    src: str,
    expect_has_patch: bool,
) -> None:
    """Verifies unused setattr/delattr aliases are treated as escaping while direct call-only aliases stay exempt."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch, case_id

    if case_id in (
        "module_unused_setattr_alias_escapes",
        "safe_called_only_setattr_alias",
        "safe_direct_walrus_setattr_call",
    ):
        safety = _late_bound_capture_safety_by_mode(
            _PROVEN_EXIT_CLASS + "obj = object()\n" + src + _WITH_MGR_RUN,
            "    cb = lambda: v",
        )
        assert safety["pedantic"] is (not expect_has_patch), case_id
        assert safety["strict"] is True, case_id


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
            "import contextlib as cl\n"
            "m = cl\n"
            "m.nullcontext = lambda *a: True\n"
            "def run(compute):\n"
            "    with cl.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="chained_module_variable_alias_rebinding",
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
            "import unittest\n"
            "m = unittest.mock\n"
            "m.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    import unittest.mock\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            False,
            id="assigned_unittest_mock_patch_rebinding",
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
            "import contextlib as cl\n"
            "def f(cl):\n"
            "    cl.nullcontext = lambda *a: True\n"
            "def run(compute):\n"
            "    with cl.nullcontext():\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="control_shadowed_local_param_cl_nullcontext",
        ),
        pytest.param(
            "import unittest.mock\n"
            "def f(unittest):\n"
            "    unittest.mock.patch = lambda *a: True\n"
            "def run(compute):\n"
            "    with unittest.mock.patch('pkg.fn'):\n"
            "        v = compute()\n"
            "    cb = lambda: v\n"
            "    return cb()\n",
            True,
            id="control_shadowed_local_param_unittest_mock_patch",
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
    ("module_src", "expected_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "fn = cm.__setattr__\nfn('__class__', Suppressor)\n",
            True,
            frozenset({"cm"}),
            id="bound_instance_setattr_exit_protocol_call",
        ),
        pytest.param(
            "fn = Mgr.__delattr__\nfn('__exit__')\n",
            True,
            frozenset({"Mgr"}),
            id="bound_class_delattr_exit_protocol_call",
        ),
        pytest.param(
            "def f(type):\n    return type.__setattr__\n",
            True,
            frozenset(),
            id="escaping_param_setattr_reference",
        ),
        pytest.param(
            "def f(type):\n    return type.__name__\n",
            False,
            frozenset(),
            id="control_escaping_param_non_mutation_dunder",
        ),
        pytest.param(
            "cm.__setattr__('label', 1)\nMgr.__delattr__('label')\n",
            False,
            frozenset(),
            id="control_direct_safe_attr_dunder_calls",
        ),
        pytest.param(
            (
                "def f(code):\n"
                "    global eval\n"
                "    _ = [(eval := eval) for _ in (1,)]\n"
                "    eval(code)\n"
            ),
            True,
            frozenset(),
            id="global_eval_comprehension_walrus_keeps_builtin_binding",
        ),
        pytest.param(
            (
                "def outer(code):\n"
                "    fn = lambda x: x\n"
                "    def set_fn():\n"
                "        nonlocal fn\n"
                "        _ = [(fn := eval) for _ in (1,)]\n"
                "    set_fn()\n"
                "    fn(code)\n"
            ),
            True,
            frozenset(),
            id="nonlocal_eval_alias_comprehension_walrus",
        ),
        pytest.param(
            (
                "def outer(code):\n"
                "    fn = lambda x: x\n"
                "    _ = [(fn := fn) for _ in (1,)]\n"
                "    return fn(code)\n"
            ),
            False,
            frozenset(),
            id="control_local_comprehension_walrus_stays_clean",
        ),
    ],
)
def test_bound_dunder_aliases_and_comprehension_walrus_exit_protocol_scan(
    module_src: str,
    expected_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies bound dunder aliases and global/nonlocal comprehension walrus targets during exit-protocol scans."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expected_has_patch
    assert scan.patched_names == expected_patched_names


@pytest.mark.parametrize(
    ("module_src", "expected_strict", "expected_pedantic"),
    [
        pytest.param(
            (
                "class Mgr:\n"
                "    def __enter__(self):\n"
                "        return self\n"
                "    def __exit__(self, *exc):\n"
                "        return False\n"
                "class Suppressor:\n"
                "    def __exit__(self, *exc):\n"
                "        return True\n"
                "def evil(cm):\n"
                "    fn = cm.__setattr__\n"
                "    fn('__class__', Suppressor)\n"
                "def run(compute):\n"
                "    cm = Mgr()\n"
                "    evil(cm)\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            False,
            id="bound_cm_setattr_class_rebind_withdraws_local_mgr_proof",
        ),
        pytest.param(
            (
                "def run(compute):\n"
                "    class Mgr:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *exc):\n"
                "            return False\n"
                "    fn = Mgr.__setattr__\n"
                "    fn('__exit__', lambda self, *a: True)\n"
                "    with Mgr():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="direct_class_bound_setattr_rebind_invalidates_both_modes",
        ),
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


@pytest.mark.parametrize(
    ("src", "expect_has_patch"),
    [
        pytest.param(
            (
                "class Patcher:\n"
                "    def setattr(self, target, handler):\n"
                "        setattr(target, '__exit__', handler)\n"
            ),
            True,
            id="pep227_method_named_setattr_calls_builtin_setattr",
        ),
        pytest.param(
            (
                "def outer():\n"
                "    class Patcher:\n"
                "        def eval(self, code):\n"
                "            return eval(code)\n"
            ),
            True,
            id="pep227_nested_method_named_eval_calls_builtin_eval",
        ),
        pytest.param(
            (
                "def outer(flag):\n"
                "    def eval(x):\n"
                "        return x if not flag else eval(False)\n"
                "    return eval(1)\n"
            ),
            False,
            id="control_nested_function_named_eval_shadows_in_enclosing_fn",
        ),
        pytest.param(
            (
                "s = setattr\n"
                "def run(target, handler):\n"
                "    class C:\n"
                "        x = (s := 1)\n"
                "    return s(target, '__exit__', handler)\n"
            ),
            True,
            id="class_body_walrus_does_not_shadow_enclosing_fn_alias",
        ),
        pytest.param(
            (
                "def run():\n"
                "    s = lambda *a: None\n"
                "    class C:\n"
                "        x = (s := 1)\n"
                "    return s(1, '__exit__', 2)\n"
            ),
            False,
            id="control_class_body_walrus_leaves_enclosing_local_shadow_intact",
        ),
    ],
)
def test_pep227_method_self_name_and_class_walrus_exit_protocol_scan(
    src: str, expect_has_patch: bool
) -> None:
    """Verifies PEP 227 method self-names do not shadow builtins and class-body walruses do not shadow enclosing locals."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


def test_try_except_finally_unhandled_exception_preserves_handler_base() -> None:
    """Verifies finally sees unhandled-exception handler_base provenance and post-try preserves it only on jump suppression."""
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _EMPTY_SUPPRESS_PROVENANCE,
        _block_terminates,
        _update_suppress_provenance_from_stmts,
    )

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
    ("extra_src", "expect_has_patch", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "def eval(model):\n"
                "    return model(1)\n"
                "def caller(m):\n"
                "    return eval(m)\n"
            ),
            False,
            True,
            id="module_level_def_eval_shadows_builtin",
        ),
        pytest.param(
            (
                "eval = False\n"
                "def caller():\n"
                "    return 1 if eval else 0\n"
            ),
            False,
            True,
            id="module_level_var_eval_shadows_builtin",
        ),
        pytest.param(
            (
                "setattr = lambda *a: None\n"
                "def configure(obj):\n"
                "    return setattr(obj, '__exit__', True)\n"
            ),
            False,
            True,
            id="module_level_var_setattr_shadows_builtin",
        ),
        pytest.param(
            (
                "class Config:\n"
                "    eval = False\n"
                "    enabled = bool(eval)\n"
            ),
            False,
            True,
            id="class_body_eval_shadows_direct_class_load",
        ),
        pytest.param(
            (
                "class PatchCfg:\n"
                "    setattr = lambda *a: None\n"
                "    applied = setattr(object(), '__exit__', True)\n"
            ),
            False,
            True,
            id="class_body_setattr_shadows_direct_class_call",
        ),
        pytest.param(
            (
                "class Config:\n"
                "    eval = False\n"
                "    def method(self, code):\n"
                "        return eval(code)\n"
            ),
            True,
            False,
            id="negative_pep227_method_bypasses_class_eval_shadow",
        ),
        pytest.param(
            (
                "my_eval = eval\n"
                "def caller(code):\n"
                "    return my_eval(code)\n"
            ),
            True,
            False,
            id="negative_unshadowed_module_eval_alias",
        ),
        pytest.param(
            (
                "def caller(target, fn):\n"
                "    global setattr\n"
                "    setattr(target, '__exit__', fn)\n"
            ),
            True,
            False,
            id="negative_global_setattr_without_module_binding",
        ),
        pytest.param(
            (
                "def installer():\n"
                "    global custom_alias\n"
                "    custom_alias = setattr\n"
                "def caller(target, fn):\n"
                "    custom_alias(target, '__exit__', fn)\n"
            ),
            True,
            False,
            id="negative_global_setattr_alias_created_inside_function",
        ),
    ],
)
def test_module_and_class_body_shadowed_builtins_exit_protocol_scan(
    extra_src: str, expect_has_patch: bool, expect_pedantic_safe: bool
) -> None:
    """Verifies module-level and direct class-body bindings shadow builtins while keeping PEP 227/global escapes sound."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = _PROVEN_EXIT_CLASS + extra_src + _WITH_MGR_RUN
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


@pytest.mark.parametrize(
    ("mut_stmt", "ctx_open_line", "expect_pedantic_safe"),
    [
        pytest.param(
            "    setattr(self.conn, 'connect', fn)\n",
            "    with sqlite3.connect(path):\n",
            True,
            id="compound_receiver_setattr_connect_does_not_invalidate_sqlite3",
        ),
        pytest.param(
            "    setattr(get_door(), 'open', False)\n",
            "    with open(path):\n",
            True,
            id="call_receiver_setattr_open_does_not_invalidate_builtin_open",
        ),
        pytest.param(
            "    self.conn.__setattr__('open', False)\n",
            "    with open(path):\n",
            True,
            id="compound_receiver_dunder_setattr_open_does_not_invalidate_open",
        ),
        pytest.param(
            "    setattr(importlib.import_module('sqlite3'), 'connect', fn)\n",
            "    with sqlite3.connect(path):\n",
            False,
            id="negative_importlib_setattr_connect_invalidates_sqlite3",
        ),
        pytest.param(
            "    setattr(sys.modules['builtins'], 'open', fn)\n",
            "    with open(path):\n",
            False,
            id="negative_sys_modules_builtins_setattr_open_invalidates_open",
        ),
    ],
)
def test_compound_receiver_setattr_stdlib_factory_attr_scoping(
    mut_stmt: str, ctx_open_line: str, expect_pedantic_safe: bool
) -> None:
    """Verifies setattr/dunder on compound non-module receivers does not wildcard-invalidate stdlib factories."""
    module_src = (
        "import importlib\n"
        "import sqlite3\n"
        "import sys\n"
        "def mutate(self, fn, get_door):\n"
        f"{mut_stmt}"
        "def run(path, compute):\n"
        f"{ctx_open_line}"
        "        v = compute()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


@pytest.mark.parametrize(
    ("extra_src", "expect_has_patch", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "def outer(eval=False):\n"
                "    def inner():\n"
                "        global eval\n"
                "        return [eval('1 + 1') for _ in (1,)]\n"
                "    return inner()\n"
            ),
            True,
            False,
            id="ancestor_global_eval_escapes_outer_param_in_comprehension",
        ),
        pytest.param(
            (
                "def outer(setattr=False):\n"
                "    def inner():\n"
                "        global setattr\n"
                "        fn = lambda obj, cb: setattr(obj, '__exit__', cb)\n"
                "        return fn(object(), None)\n"
                "    return inner()\n"
            ),
            True,
            False,
            id="ancestor_global_setattr_escapes_outer_param_in_lambda",
        ),
        pytest.param(
            (
                "def outer():\n"
                "    eval = lambda x: x\n"
                "    def mid():\n"
                "        global eval\n"
                "        def leaf(code):\n"
                "            return eval(code)\n"
                "        return leaf('1')\n"
                "    return mid()\n"
            ),
            True,
            False,
            id="ancestor_global_eval_escapes_outer_local_in_nested_def",
        ),
        pytest.param(
            (
                "custom_eval = eval\n"
                "def outer(custom_eval=False):\n"
                "    def mid():\n"
                "        global custom_eval\n"
                "        return [custom_eval('1 + 1') for _ in (1,)]\n"
                "    return mid()\n"
            ),
            True,
            False,
            id="ancestor_global_module_alias_escapes_outer_param_in_comprehension",
        ),
        pytest.param(
            (
                "def outer(eval=False):\n"
                "    def inner():\n"
                "        return [eval('1 + 1') for _ in (1,)]\n"
                "    return inner()\n"
            ),
            False,
            True,
            id="negative_comprehension_without_ancestor_global_stays_shadowed",
        ),
        pytest.param(
            (
                "def outer(eval=False):\n"
                "    def mid():\n"
                "        global eval\n"
                "        def leaf(eval=lambda x: x):\n"
                "            return eval('1')\n"
                "        return leaf()\n"
                "    return mid()\n"
            ),
            False,
            True,
            id="negative_inner_param_re_shadows_ancestor_global_decl",
        ),
        pytest.param(
            (
                "eval = lambda x: x\n"
                "def outer(eval=False):\n"
                "    def mid():\n"
                "        global eval\n"
                "        return [eval('1') for _ in (1,)]\n"
                "    return mid()\n"
            ),
            False,
            True,
            id="negative_module_shadow_prevents_ancestor_global_builtin_match",
        ),
    ],
)
def test_ancestor_global_decl_escapes_outer_shadow_exit_protocol_scan(
    extra_src: str, expect_has_patch: bool, expect_pedantic_safe: bool
) -> None:
    """Verifies ancestor global declarations bypass enclosing lexical shadows in nested scopes."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = _PROVEN_EXIT_CLASS + extra_src + _WITH_MGR_RUN
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


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
            "s = setattr\n",
            "open(path)",
            {"strict": True, "pedantic": True},
            True,
            id="uncalled_setattr_keeps_builtin_open_trusted",
        ),
        pytest.param(
            "def trigger(fn):\n    patch_m = Mgr.__setattr__\n    patch_m('__exit__', fn)\n",
            "open(path)",
            {"strict": True, "pedantic": True},
            True,
            id="bound_mgr_dunder_setattr_keeps_open_trusted",
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
            "def patch(fn):\n    builtins.__dict__['open'] = fn\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_builtins_dict_open_rebind_invalidates_open",
        ),
        pytest.param(
            "def patch(fn):\n    __builtins__['open'] = fn\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_dunder_builtins_open_rebind_invalidates_open",
        ),
        pytest.param(
            "def patch(fn):\n    vars(builtins)['open'] = fn\n",
            "open(path)",
            {"strict": False, "pedantic": False},
            True,
            id="negative_vars_builtins_open_rebind_invalidates_open",
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
    ("stmt", "expect_has_patch", "expect_rebound", "expect_pedantic_safe"),
    [
        pytest.param(
            "    getattr(builtins, 'setattr')(Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_getattr_builtins_setattr_safe_attr",
        ),
        pytest.param(
            "    builtins.__dict__['setattr'](Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_builtins_dict_subscript_setattr_safe_attr",
        ),
        pytest.param(
            "    vars(builtins)['setattr'](Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_vars_builtins_subscript_setattr_safe_attr",
        ),
        pytest.param(
            "    vars(builtins).get('setattr')(Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_vars_builtins_get_setattr_safe_attr",
        ),
        pytest.param(
            "    importlib.import_module('builtins').setattr(Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_importlib_builtins_setattr_safe_attr",
        ),
        pytest.param(
            "    globals()['__builtins__']['setattr'](Mgr, 'normal_label', 1)\n",
            False,
            False,
            True,
            id="immediate_globals_dunder_builtins_setattr_safe_attr",
        ),
        pytest.param(
            "    getattr(builtins, 'setattr')(Mgr, '__exit__', lambda *a: True)\n",
            True,
            False,
            False,
            id="negative_immediate_getattr_setattr_exit_protocol_patch",
        ),
        pytest.param(
            "    fn = getattr(builtins, 'setattr')\n    fn(Mgr, 'normal_label', 1)\n",
            True,
            True,
            False,
            id="negative_uncalled_getattr_builtins_setattr_still_flags",
        ),
        pytest.param(
            "    (fn := getattr(builtins, 'setattr'))(Mgr, 'normal_label', 1)\n",
            True,
            True,
            False,
            id="negative_walrus_getattr_builtins_setattr_still_flags",
        ),
        pytest.param(
            "    (ns := vars(builtins))['setattr'](Mgr, 'normal_label', 1)\n",
            True,
            True,
            False,
            id="negative_walrus_vars_builtins_escape_still_flags",
        ),
        pytest.param(
            "    getattr(builtins, 'eval')('1 + 1')\n",
            True,
            True,
            False,
            id="negative_immediate_getattr_builtins_eval_still_flags",
        ),
        pytest.param(
            "    getattr(eval('builtins'), 'setattr')(Mgr, 'normal_label', 1)\n",
            True,
            True,
            False,
            id="negative_nested_eval_inside_getattr_receiver_still_flags",
        ),
    ],
)
def test_immediate_safe_indirect_setattr_callees_keep_module_proofs_intact(
    stmt: str,
    expect_has_patch: bool,
    expect_rebound: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies immediate getattr/namespace setattr callees with safe attrs leave module proofs intact."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = (
        "import builtins\n"
        "import importlib\n"
        + _PROVEN_EXIT_CLASS
        + "def configure():\n"
        + stmt
        + _WITH_MGR_RUN
    )
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch
    assert bool(scan.rebound_attrs) is expect_rebound
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


@pytest.mark.parametrize(
    (
        "extra_src",
        "expect_has_patch",
        "expect_patched_names",
        "expect_strict_safe",
        "expect_pedantic_safe",
    ),
    [
        pytest.param(
            (
                "def top(b=False):\n"
                "    def outer():\n"
                "        global b\n"
                "        import builtins as b\n"
                "        def inner(code):\n"
                "            return b.eval(code)\n"
                "        return inner('1')\n"
                "    return outer()\n"
            ),
            True,
            set(),
            True,
            False,
            id="global_import_builtins_nested_eval_beats_outer_shadow",
        ),
        pytest.param(
            (
                "def outer(swallow):\n"
                "    global b\n"
                "    _ = (b := __import__('builtins'))\n"
                "    def inner(obj):\n"
                "        b.setattr(obj, '__exit__', swallow)\n"
                "    return inner(Mgr)\n"
            ),
            True,
            {"obj"},
            True,
            False,
            id="global_walrus_dunder_import_builtins_nested_setattr",
        ),
        pytest.param(
            (
                "def grand():\n"
                "    b = False\n"
                "    def outer():\n"
                "        nonlocal b\n"
                "        import builtins as b\n"
                "        def inner(obj, swallow):\n"
                "            b.setattr(obj, '__exit__', swallow)\n"
                "        return inner(Mgr, lambda *a: True)\n"
                "    return outer()\n"
            ),
            True,
            {"obj"},
            True,
            False,
            id="nonlocal_import_builtins_nested_setattr",
        ),
        pytest.param(
            (
                "import sys\n"
                "def patch(swallow):\n"
                "    sys.modules.get('builtins').setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            {"Mgr"},
            False,
            False,
            id="sys_modules_get_builtins_setattr_exit_records_mgr",
        ),
        pytest.param(
            (
                "import sys\n"
                "def configure():\n"
                "    sys.modules.get('builtins').setattr(Mgr, 'normal_label', 1)\n"
            ),
            False,
            set(),
            True,
            True,
            id="control_sys_modules_get_builtins_safe_attr",
        ),
        pytest.param(
            (
                "def patch(swallow):\n"
                "    b = globals().get('__builtins__')\n"
                "    b.setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            {"Mgr"},
            False,
            False,
            id="globals_get_dunder_builtins_alias_setattr_records_mgr",
        ),
        pytest.param(
            (
                "def patch(swallow):\n"
                "    b = vars().get('__builtins__')\n"
                "    b.setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            {"Mgr"},
            False,
            False,
            id="vars_get_dunder_builtins_alias_setattr_records_mgr",
        ),
        pytest.param(
            "def patch(vals):\n    setattr(Mgr, '__exit__', *vals)\n",
            True,
            {"Mgr"},
            False,
            False,
            id="setattr_starred_value_arg_records_target_mgr",
        ),
        pytest.param(
            "def patch(vals):\n    Mgr.__setattr__('__exit__', *vals)\n",
            True,
            {"Mgr"},
            False,
            False,
            id="dunder_setattr_starred_value_arg_records_bound_mgr",
        ),
        pytest.param(
            "def patch(args):\n    type.__setattr__(Mgr, *args)\n",
            True,
            {"Mgr"},
            False,
            False,
            id="type_dunder_setattr_starred_args_records_target_mgr",
        ),
        pytest.param(
            "def patch(fn):\n    setattr(Mgr, attr='__exit__', value=fn)\n",
            True,
            {"Mgr"},
            False,
            False,
            id="setattr_keyword_args_records_target_mgr",
        ),
        pytest.param(
            "def patch(vals):\n    setattr(Other, '__exit__', *vals)\n",
            True,
            {"Other"},
            True,
            False,
            id="control_setattr_starred_on_other_leaves_mgr_unpatched_in_strict",
        ),
    ],
)
def test_global_import_walrus_dot_get_and_starred_setattr_exit_protocol(
    extra_src: str,
    expect_has_patch: bool,
    expect_patched_names: set[str],
    expect_strict_safe: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies global/nonlocal imports/walrus, .get() namespace lookups, and starred setattr targets."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = "class Other:\n    pass\n" + _PROVEN_EXIT_CLASS + extra_src + _WITH_MGR_RUN
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch
    assert expect_patched_names.issubset(scan.patched_names)
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_strict_safe, "pedantic": expect_pedantic_safe}


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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _EMPTY_SUPPRESS_PROVENANCE,
        _update_suppress_provenance_from_stmts,
    )

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
    ("extra_src", "expect_has_patch", "expect_patched", "expect_rebound", "expect_pedantic_safe"),
    [
        pytest.param(
            "b, _ = builtins, 1\nb.setattr(Mgr, '__exit__', swallow)\n",
            True,
            {"Mgr"},
            set(),
            False,
            id="unpack_builtins_setattr",
        ),
        pytest.param(
            "[t, _] = [type, 1]\nt.__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            {"Mgr"},
            set(),
            False,
            id="unpack_list_type_dunder_setattr",
        ),
        pytest.param(
            "(v, g) = (vars, globals)\nv(builtins)['setattr'](Mgr, '__exit__', swallow)\n",
            True,
            {"Mgr"},
            set(),
            False,
            id="unpack_vars_globals_subscript_setattr",
        ),
        pytest.param(
            "b, *rest = builtins, 1, 2\nb.setattr(Mgr, '__exit__', swallow)\n",
            True,
            {"Mgr"},
            set(),
            False,
            id="starred_unpack_builtins_setattr",
        ),
        pytest.param(
            "pair = (builtins, type)\nb, t = pair\nt.__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            {"Mgr"},
            set(),
            False,
            id="aliased_tuple_unpack_type_dunder_setattr",
        ),
        pytest.param(
            "c, _ = contextlib, 1\nc.nullcontext = swallow\n",
            False,
            set(),
            {("contextlib", "nullcontext")},
            True,
            id="unpack_stdlib_module_alias_rebinds_factory",
        ),
        pytest.param(
            "def benign(m, attr, fn):\n    return (m, attr, fn)\na, _ = benign, 1\na(Mgr, '__exit__', swallow)\n",
            False,
            set(),
            set(),
            True,
            id="negative_unrelated_unpack_call_stays_safe",
        ),
        pytest.param(
            "x, y = Mgr, swallow\nx.label = 1\n",
            False,
            set(),
            set(),
            True,
            id="negative_unpack_unrelated_attribute_write_stays_safe",
        ),
    ],
)
def test_unpacking_assignments_track_builtin_module_and_type_aliases(
    extra_src: str,
    expect_has_patch: bool,
    expect_patched: set[str],
    expect_rebound: set[tuple[str, str]],
    expect_pedantic_safe: bool,
) -> None:
    """Verifies tuple/list unpacking and container aliases propagate builtin, module, and type slots."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = (
        "import builtins\n"
        "import contextlib\n"
        + _PROVEN_EXIT_CLASS
        + "def swallow(*a):\n    return True\n"
        + extra_src
        + "def run(compute):\n"
        + "    with Mgr():\n"
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch, scan
    assert expect_patched.issubset(scan.patched_names), scan
    for pair in expect_rebound:
        assert pair in scan.rebound_attrs, scan
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


@pytest.mark.parametrize(
    (
        "patch_src",
        "ctx_expr",
        "expect_has_patch",
        "expect_patched",
        "expect_rebound_pair",
        "expect_pedantic_safe",
    ),
    [
        pytest.param(
            "contextlib.__dict__['nullcontext'] = swallow\n",
            "contextlib.nullcontext()",
            False,
            set(),
            ("contextlib", "nullcontext"),
            False,
            id="contextlib_dunder_dict_subscript_rebind",
        ),
        pytest.param(
            "del vars(contextlib)['nullcontext']\n",
            "contextlib.nullcontext()",
            False,
            set(),
            ("contextlib", "nullcontext"),
            False,
            id="vars_contextlib_subscript_delete",
        ),
        pytest.param(
            "vars(contextlib).update({'nullcontext': swallow})\n",
            "contextlib.nullcontext()",
            False,
            set(),
            ("contextlib", "nullcontext"),
            False,
            id="vars_contextlib_update_rebind",
        ),
        pytest.param(
            "d = contextlib.__dict__\nd.setdefault('nullcontext', swallow)\n",
            "contextlib.nullcontext()",
            False,
            set(),
            ("contextlib", "nullcontext"),
            False,
            id="aliased_contextlib_dict_setdefault",
        ),
        pytest.param(
            "contextlib.__dict__.pop('nullcontext', None)\n",
            "contextlib.nullcontext()",
            False,
            set(),
            ("contextlib", "nullcontext"),
            False,
            id="contextlib_dict_pop",
        ),
        pytest.param(
            "vars(unittest.mock)['patch'] = swallow\n",
            "unittest.mock.patch('x')",
            False,
            set(),
            ("unittest.mock", "patch"),
            False,
            id="vars_dotted_stdlib_subscript_rebind",
        ),
        pytest.param(
            "globals()['open'] = swallow\n",
            "open('f')",
            False,
            set(),
            ("builtins", "open"),
            False,
            id="globals_subscript_rebinds_open",
        ),
        pytest.param(
            "globals().update({'open': swallow})\n",
            "open('f')",
            False,
            set(),
            ("builtins", "open"),
            False,
            id="globals_update_rebinds_open",
        ),
        pytest.param(
            "Mgr.__dict__['__exit__'] = swallow\n",
            "Mgr()",
            True,
            {"Mgr"},
            None,
            False,
            id="mgr_dunder_dict_exit_rebind",
        ),
        pytest.param(
            "vars(Mgr).update(__exit__=swallow)\n",
            "Mgr()",
            True,
            {"Mgr"},
            None,
            False,
            id="vars_mgr_update_kwarg_exit_rebind",
        ),
        pytest.param(
            "contextlib.__dict__.get('nullcontext')\n",
            "contextlib.nullcontext()",
            False,
            set(),
            None,
            True,
            id="control_readonly_contextlib_dict_get",
        ),
        pytest.param(
            "v = vars(Other)\nv['x'] = 1\nv.update({'label': 2})\n",
            "contextlib.nullcontext()",
            False,
            set(),
            None,
            True,
            id="control_non_factory_vars_dict_mutation_safe",
        ),
        pytest.param(
            "local_d = {}\nlocal_d['nullcontext'] = 1\nlocal_d.update({'__exit__': 1})\n",
            "contextlib.nullcontext()",
            False,
            set(),
            None,
            True,
            id="control_ordinary_dict_mutation_ignored",
        ),
        pytest.param(
            "class Helper:\n    def sync(self, data):\n        self.__dict__.update(data)\n",
            "Mgr()",
            False,
            set(),
            None,
            True,
            id="control_instance_dict_update_dynamic_safe",
        ),
    ],
)
def test_namespace_dict_and_vars_mutations_withdraw_factory_and_exit_proofs(
    patch_src: str,
    ctx_expr: str,
    expect_has_patch: bool,
    expect_patched: set[str],
    expect_rebound_pair: tuple[str, str] | None,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies __dict__/vars(mod)/globals() mutations invalidate targeted proofs without breaking controls."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    module_src = (
        "import contextlib\n"
        "import unittest.mock\n"
        "class Other:\n    pass\n"
        + _PROVEN_EXIT_CLASS
        + "def swallow(*a):\n    return True\n"
        + patch_src
        + "def run(compute):\n"
        + f"    with {ctx_expr}:\n"
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(module_src))
    assert scan.has_patch is expect_has_patch, scan
    assert expect_patched.issubset(scan.patched_names), scan
    if expect_rebound_pair is not None:
        assert expect_rebound_pair in scan.rebound_attrs, scan
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety["pedantic"] is expect_pedantic_safe


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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _merge_hazard_provenance,
    )

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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

    wrapper = f"async def _probe(arr, empty_seq, empty_aiter):\n    " + loop_src.replace(
        "\n", "\n    "
    )
    fn_node = ast.parse(wrapper).body[0]
    assert isinstance(fn_node, ast.AsyncFunctionDef)
    definite, conditional = _analyze_block_assignment(fn_node.body)
    assert (var_name in definite) is expect_definite
    assert (var_name in conditional) is expect_conditional


@pytest.mark.parametrize(
    ("stmt_src", "expect_has_patch"),
    [
        pytest.param("getattr(*args)\n", True, id="starred_first_arg"),
        pytest.param("getattr(*pair, None)\n", True, id="starred_first_arg_with_default"),
        pytest.param("getattr(obj, *rest)\n", True, id="starred_second_arg"),
        pytest.param(
            "import builtins\ngetattr(builtins, *rest)\n",
            True,
            id="builtins_with_starred_second_arg",
        ),
        pytest.param(
            "getattr(obj, (k := '__setattr__'))\n",
            True,
            id="walrus_wrapped_mutation_attr_name",
        ),
        pytest.param(
            "getattr(obj, 'label', *defaults)\n",
            False,
            id="control_literal_benign_attr_with_starred_default",
        ),
        pytest.param(
            "import builtins\ngetattr(builtins, 'len', *defaults)\n",
            False,
            id="control_builtins_benign_literal_with_starred_default",
        ),
    ],
)
def test_exit_protocol_scanner_starred_getattr_args(
    stmt_src: str,
    expect_has_patch: bool,
) -> None:
    """Verifies getattr(*args) and getattr(obj, *rest) invalidate proofs while safe starred defaults do not."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    scan = scan_exit_protocol_patches(ast.parse(stmt_src))
    assert scan.has_patch is expect_has_patch


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
    import pydoppelgangerhunt.fixer._df_with_provenance as prov_mod

    prov_calls = 0
    flow_calls = 0
    real_prov_stmt = prov_mod._update_prov_and_term_from_stmt
    real_flow_impl = prov_mod._analyze_block_assignment_impl

    def _spy_prov_stmt(*args: Any, **kwargs: Any) -> Any:
        nonlocal prov_calls
        prov_calls += 1
        return real_prov_stmt(*args, **kwargs)

    def _spy_flow_impl(*args: Any, **kwargs: Any) -> Any:
        nonlocal flow_calls
        flow_calls += 1
        return real_flow_impl(*args, **kwargs)

    monkeypatch.setattr(prov_mod, "_update_prov_and_term_from_stmt", _spy_prov_stmt)
    monkeypatch.setattr(prov_mod, "_analyze_block_assignment_impl", _spy_flow_impl)

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
        out_prov = prov_mod._update_suppress_provenance_from_stmts(
            stmts, prov_mod._EMPTY_SUPPRESS_PROVENANCE, in_branch=False
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
        out_prov = prov_mod._update_suppress_provenance_from_stmts(
            stmts, prov_mod._EMPTY_SUPPRESS_PROVENANCE, in_branch=False
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
        definite, conditional = prov_mod._analyze_block_assignment(stmts)
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
        definite, conditional = prov_mod._analyze_block_assignment(stmts)
        assert "done" in conditional and "done" not in definite
        assert flow_calls <= max_calls


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
    ("module_src", "expect_safe"),
    [
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_from_import_nullcontext",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_stdlib_module_alias",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_instance_alias",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
                "from evil import *\n"
            ),
            False,
            id="post_def_star_import_invalidates_prior_nullcontext",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "if flag:\n"
                "    from evil import *\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="conditional_star_import_invalidates_prior_nullcontext",
        ),
        pytest.param(
            (
                "from evil import *\n"
                "from contextlib import nullcontext\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_explicit_import_after_star_import_remains_proven",
        ),
        pytest.param(
            (
                "from evil import *\n"
                "def run(compute):\n"
                "    from contextlib import nullcontext\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_function_local_import_after_module_star_import_remains_proven",
        ),
    ],
)
def test_wildcard_star_import_invalidates_prior_non_suppression_proofs(
    module_src: str,
    expect_safe: bool,
) -> None:
    """Verifies wildcard imports withdraw prior positive non-suppression proofs in strict and pedantic modes while preserving later explicit imports."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}


@pytest.mark.parametrize(
    ("src", "expect_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "type(Mgr).__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="type_call_dunder_setattr_records_target",
        ),
        pytest.param(
            "import builtins\nbuiltins.type(Mgr).__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="builtins_type_call_dunder_setattr_records_target",
        ),
        pytest.param(
            "T = type\nT(Mgr).__setattr__(Mgr, '__aexit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="aliased_type_call_dunder_setattr_records_target",
        ),
        pytest.param(
            "getattr(type(Mgr), '__setattr__')(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="getattr_type_call_dunder_setattr_records_target",
        ),
        pytest.param(
            "patch = type(Mgr).__setattr__\npatch(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="stored_type_call_dunder_setattr_alias_records_target",
        ),
        pytest.param(
            "type(Mgr).__setattr__(Mgr, 'benign_label', 1)\n",
            False,
            frozenset(),
            id="control_type_call_dunder_setattr_non_exit_attr_ignored",
        ),
        pytest.param(
            "object().__setattr__('__class__', Suppressor)\n",
            True,
            frozenset(),
            id="control_bound_object_instance_dunder_setattr_class_detected",
        ),
        pytest.param(
            "O = object\nO().__setattr__('__class__', Suppressor)\n",
            True,
            frozenset(),
            id="control_aliased_object_instance_dunder_setattr_class_detected",
        ),
        pytest.param(
            "O = object\nO().__setattr__('benign_label', 1)\n",
            False,
            frozenset(),
            id="control_aliased_object_instance_benign_attr_ignored",
        ),
    ],
)
def test_type_call_dunder_setattr_records_patched_target_and_filters_benign_attrs(
    src: str,
    expect_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies type(Mgr).__setattr__(Mgr, ...) uses unbound argument indexing to record patched targets and ignore non-exit attributes."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch
    assert scan.patched_names == expected_patched_names


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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

    stmts = ast.parse(loop_body_src).body
    for strict_mode in (False, True):
        definite, conditional = _analyze_block_assignment(
            stmts, strict_with_suppression=strict_mode
        )
        assert ("v" in definite) is expect_definite
        assert ("v" in conditional) is (not expect_definite)


@pytest.mark.parametrize(
    ("module_src", "expect_pre_bound"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm(ValueError):\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            False,
            id="for_enclosing_loop_rebinding_after_unit_clears_pre_bound",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(conds, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    while conds:\n"
                "        if conds.pop():\n"
                "            with cm(ValueError):\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            False,
            id="while_enclosing_loop_rebinding_after_unit_clears_pre_bound",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        cm = contextlib.nullcontext\n"
                "        if i > 0:\n"
                "            with cm():\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            True,
            id="control_for_enclosing_loop_restores_cm_before_with",
        ),
    ],
)
def test_enclosing_loop_carried_ctx_manager_rebinding_invalidates_pre_unit_with(
    module_src: str,
    expect_pre_bound: bool,
) -> None:
    """Verifies cell traversal propagates loop-body context-manager rebinding into pre-unit with statements."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells

    tree = ast.parse(module_src)
    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    unit_line = next(
        idx for idx, ln in enumerate(module_src.splitlines(), start=1) if "res = v + 1" in ln
    )
    for strict_mode in (False, True):
        clear_dataflow_cache()
        info = _analyze_enclosing_fn_cells(
            fn_node,
            (unit_line, unit_line, None, None),
            mod_tree=tree,
            strict_with_suppression=strict_mode,
        )
        assert ("v" in info.pre_bound) is expect_pre_bound


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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
        _stmt_cannot_raise_in_suppress,
    )

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
    from pydoppelgangerhunt.fixer import _df_with_provenance as prov_mod

    stmts = ast.parse(stmt_src).body
    may_del: set[str] = set()
    must_del: set[str] = set()
    live_in_del: set[str] = set()
    depth = prov_mod._MAX_AST_RECURSION_DEPTH + 1 if hit_depth_limit else 0
    definite, _conditional, _term, _prov = prov_mod._analyze_block_assignment_impl(
        stmts,
        definite={"x"},
        may_deleted=may_del,
        must_deleted=must_del,
        live_in_deleted=live_in_del,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        _depth=depth,
    )
    assert ("x" not in definite) is expect_deleted
    assert ("x" in may_del) is expect_deleted


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
    from pydoppelgangerhunt.fixer import _df_with_provenance as prov_mod

    stmts = ast.parse(block_src).body
    prov_walk = prov_mod._update_suppress_provenance_from_stmts(
        stmts,
        prov_mod._EMPTY_SUPPRESS_PROVENANCE,
        in_branch=False,
        strict_with_suppression=True,
    )
    assert ("cm" in prov_walk.non_suppressors) is expect_cm_proven

    definite, _cond, _term, flow_prov = prov_mod._analyze_block_assignment_impl(
        stmts,
        strict_with_suppression=True,
    )
    assert ("cm" in flow_prov.non_suppressors) is expect_cm_proven
    assert ("v" in definite) is expect_v_definite


@pytest.mark.parametrize(
    ("src", "expect_has_patch", "expected_patched_names"),
    [
        pytest.param(
            "import builtins\nbuiltins.type.__setattr__(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="builtins_type_dunder_setattr_records_target",
        ),
        pytest.param(
            "import builtins\nbuiltins.object.__setattr__(Mgr, '__aexit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="builtins_object_dunder_setattr_records_target",
        ),
        pytest.param(
            "import builtins\ngetattr(builtins.type, '__setattr__')(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="getattr_builtins_type_dunder_setattr_records_target",
        ),
        pytest.param(
            "import builtins\npatch = builtins.type.__setattr__\npatch(Mgr, '__exit__', swallow)\n",
            True,
            frozenset({"Mgr"}),
            id="stored_builtins_type_dunder_setattr_alias_records_target",
        ),
        pytest.param(
            "import builtins\nbuiltins.type.__setattr__(Mgr, 'benign_label', 1)\n",
            False,
            frozenset(),
            id="control_builtins_type_dunder_setattr_benign_attr_ignored",
        ),
        pytest.param(
            "import builtins\nbuiltins.object.__setattr__(Mgr, 'benign_label', 1)\n",
            False,
            frozenset(),
            id="control_builtins_object_dunder_setattr_benign_attr_ignored",
        ),
        pytest.param(
            "import builtins\npatch = builtins.type.__setattr__\npatch(Mgr, 'benign_label', 1)\n",
            False,
            frozenset(),
            id="control_stored_builtins_type_dunder_setattr_benign_attr_ignored",
        ),
        pytest.param(
            "import builtins\nbuiltins.object().__setattr__('__class__', Suppressor)\n",
            True,
            frozenset(),
            id="control_builtins_object_instance_class_mutation_detected",
        ),
        pytest.param(
            "import builtins\nbuiltins.object().__setattr__('benign_label', 1)\n",
            False,
            frozenset(),
            id="control_builtins_object_instance_benign_attr_ignored",
        ),
    ],
)
def test_builtins_type_and_object_dunder_setattr_records_target_and_filters_benign(
    src: str,
    expect_has_patch: bool,
    expected_patched_names: frozenset[str],
) -> None:
    """Verifies builtins.type.__setattr__ and builtins.object.__setattr__ resolve unbound targets without benign false positives."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch
    assert scan.patched_names == expected_patched_names


@pytest.mark.parametrize(
    ("src", "expect_has_patch", "min_python"),
    [
        pytest.param(
            (
                "s = setattr\n"
                "def mutate():\n"
                "    global s\n"
                "    try:\n"
                "        op()\n"
                "    except Exception as s:\n"
                "        pass\n"
                "s(Mgr, 'label', 1)\n"
            ),
            True,
            (3, 9),
            id="global_except_as_disqualifies_pure_setattr_alias",
        ),
        pytest.param(
            (
                "def outer():\n"
                "    s = setattr\n"
                "    def mutate():\n"
                "        nonlocal s\n"
                "        try:\n"
                "            op()\n"
                "        except Exception as s:\n"
                "            pass\n"
                "    s(Mgr, 'label', 1)\n"
            ),
            True,
            (3, 9),
            id="nonlocal_except_as_disqualifies_pure_setattr_alias",
        ),
        pytest.param(
            (
                "s = setattr\n"
                "def mutate(v):\n"
                "    global s\n"
                "    match v:\n"
                "        case s:\n"
                "            pass\n"
                "s(Mgr, 'label', 1)\n"
            ),
            True,
            (3, 10),
            id="global_match_as_disqualifies_pure_setattr_alias",
        ),
        pytest.param(
            (
                "def outer():\n"
                "    s = setattr\n"
                "    def unrelated():\n"
                "        try:\n"
                "            op()\n"
                "        except Exception as s:\n"
                "            pass\n"
                "    s(Mgr, 'label', 1)\n"
            ),
            False,
            (3, 9),
            id="control_local_except_as_does_not_disqualify_enclosing_pure_setattr_alias",
        ),
    ],
)
def test_exit_protocol_except_and_match_bindings_resolve_global_nonlocal_slots(
    src: str,
    expect_has_patch: bool,
    min_python: tuple[int, int],
) -> None:
    """Verifies ExceptHandler and match-pattern writes honor global/nonlocal lexical resolution in exit-protocol scanning."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    if sys.version_info < min_python:
        pytest.skip(f"requires Python {min_python[0]}.{min_python[1]}+")
    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(src))
    assert scan.has_patch is expect_has_patch


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
    from pydoppelgangerhunt.fixer._df_with_provenance import _block_terminates

    assert _block_terminates(ast.parse(src).body) is expect_terminates


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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

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
    ("fn_src", "expect_strict_safe", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer(items, compute):\n"
                "    cm = suppress\n"
                "    for cm in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm(ValueError):\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="enclosing_for_else_zero_trip_preserves_outer_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, compute):\n"
                "    cm = nullcontext\n"
                "    for cm in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm():\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            True,
            False,
            id="enclosing_for_else_clears_nullcontext_proof_in_pedantic",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, compute):\n"
                "    cm = nullcontext\n"
                "    for other in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm():\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_enclosing_for_else_unrelated_target_keeps_nullcontext_proof",
        ),
    ],
)
def test_enclosing_for_else_zero_trip_preserves_outer_suppressor_in_cell_traversal(
    fn_src: str,
    expect_strict_safe: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies enclosing for/else traversal preserves outer suppressor provenance on the zero-trip path into else."""
    lines = fn_src.splitlines()
    unit_line = next(
        idx for idx, ln in enumerate(lines, start=1) if ln.strip() == "cb = lambda: val"
    )
    tree = ast.parse(fn_src)
    for mode, expect_safe in (
        ("strict", expect_strict_safe),
        ("pedantic", expect_pedantic_safe),
    ):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": fn_src,
                "source_code": lines[unit_line - 1] + "\n",
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert bool(sc["is_control_flow_safe"]) is expect_safe, (mode, sc)


@pytest.mark.parametrize(
    ("mod_src", "expect_has_patch", "expect_patched_name", "expect_rebound"),
    [
        pytest.param(
            (
                "from sys import modules\n"
                "def patch(Mgr, swallow):\n"
                "    modules['builtins'].setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            "Mgr",
            None,
            id="from_sys_import_modules_builtins_setattr_exit_patch",
        ),
        pytest.param(
            (
                "from sys import modules as mods\n"
                "def patch(Mgr, swallow):\n"
                "    mods.get('builtins').setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            "Mgr",
            None,
            id="from_sys_import_modules_alias_get_builtins_setattr",
        ),
        pytest.param(
            (
                "import sys\n"
                "def patch(Mgr, swallow):\n"
                "    m = sys.modules\n"
                "    m['builtins'].setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            "Mgr",
            None,
            id="local_alias_sys_modules_builtins_setattr_exit_patch",
        ),
        pytest.param(
            (
                "import sys\n"
                "def patch(Mgr, swallow):\n"
                "    m = getattr(sys, 'modules')\n"
                "    m['builtins'].setattr(Mgr, '__exit__', swallow)\n"
            ),
            True,
            "Mgr",
            None,
            id="getattr_sys_modules_alias_builtins_setattr",
        ),
        pytest.param(
            (
                "import sys\n"
                "def patch(swallow):\n"
                "    m = sys.modules\n"
                "    m2 = m\n"
                "    m2['contextlib'].nullcontext = swallow\n"
            ),
            False,
            None,
            ("contextlib", "nullcontext"),
            id="transitive_alias_sys_modules_stdlib_factory_rebind",
        ),
        pytest.param(
            (
                "from sys import modules\n"
                "def patch(swallow):\n"
                "    modules['contextlib'].nullcontext = swallow\n"
            ),
            False,
            None,
            ("contextlib", "nullcontext"),
            id="from_sys_import_modules_stdlib_factory_rebind",
        ),
        pytest.param(
            (
                "from sys import modules\n"
                "def benign(Mgr):\n"
                "    modules['builtins'].setattr(Mgr, 'label', 1)\n"
            ),
            False,
            None,
            None,
            id="control_from_sys_import_modules_safe_attr_exempt",
        ),
        pytest.param(
            (
                "from custom_pkg import modules\n"
                "def benign(Mgr, fn):\n"
                "    modules['builtins'].setattr(Mgr, '__exit__', fn)\n"
            ),
            False,
            None,
            None,
            id="control_non_sys_modules_import_not_flagged",
        ),
    ],
)
def test_exit_protocol_scanner_tracks_sys_modules_imports_and_aliases(
    mod_src: str,
    expect_has_patch: bool,
    expect_patched_name: str | None,
    expect_rebound: tuple[str, str] | None,
) -> None:
    """Verifies exit-protocol and stdlib-factory scanning tracks `from sys import modules` and local `sys.modules` aliases."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches

    clear_dataflow_cache()
    scan = scan_exit_protocol_patches(ast.parse(mod_src))
    assert scan.has_patch is expect_has_patch, scan
    if expect_patched_name is not None:
        assert expect_patched_name in scan.patched_names, scan
    if expect_rebound is not None:
        assert expect_rebound in scan.rebound_attrs, scan
    elif not expect_has_patch:
        assert not scan.rebound_attrs, scan


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
    from pydoppelgangerhunt.fixer._df_with_provenance import _analyze_block_assignment

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
            "vars(Mgr)['__bases__'] = (Suppressor,)\n",
            True,
            False,
            id="vars_dict_bases_mutation",
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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
        _seed_scope_suppress_aliases,
    )

    mod = ast.parse(module_src)
    fn_node = next(
        n
        for n in ast.walk(mod)
        if isinstance(n, ast.FunctionDef) and n.name == target_fn_name
    )
    clear_dataflow_cache()
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree=mod)
    param_names = {a.arg for a in fn_node.args.args}
    _, _, term, _ = _analyze_block_assignment_impl(
        fn_node.body,
        definite=set(param_names),
        suppress_aliases=prov,
    )
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
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment_impl,
        _seed_scope_suppress_aliases,
    )

    mod = ast.parse(module_src)
    fn_node = next(
        n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "f"
    )
    clear_dataflow_cache()
    prov = _seed_scope_suppress_aliases(fn_node, mod_tree=mod)
    param_names = {a.arg for a in fn_node.args.args}
    definite, _, term, _ = _analyze_block_assignment_impl(
        fn_node.body,
        definite=set(param_names),
        suppress_aliases=prov,
    )
    assert term is expect_terminates
    assert ("x" in definite) is expect_x_definite




