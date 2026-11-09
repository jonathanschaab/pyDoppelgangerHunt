"""Differential execution tests and context-manager/lexical closure semantics tests."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Any

from pydoppelgangerhunt.fixer import (
    analyze_unit_variable_scope,
    clear_dataflow_cache,
    inspect_single_unit_scope,
    resolve_clone_pair_outputs,
    synthesize_shared_helper_code,
)
from tests.fixer._closure_test_helpers import (
    _run_differential_refactoring_case,
)


def test_pep227_pep3104_class_body_vs_method_lexical_declarations_and_differential(
    tmp_path: Path,
) -> None:
    """Covers PEP 227 / PEP 3104 class-body vs method nonlocals/globals and differential execution."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
    )

    # Case 1: method nonlocal x binding to outer's x across class C
    case1_src = (
        "def outer():\n"
        "    x = 1\n"
        "    class C:\n"
        "        def method(self):\n"
        "            nonlocal x\n"
        "            x += 1\n"
        "    return C\n"
    )
    case1_tree = ast.parse(case1_src)
    case1_chain = _find_enclosing_function_chain(case1_tree, 3, 6)
    case1_cls = case1_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert (
        _has_invalid_lexical_scope_declarations(
            [case1_cls], enclosing_chain=case1_chain, mod_tree=case1_tree
        )
        is False
    )

    # Case 2: class C defines x = 2, method declares nonlocal x (binds to outer's x, valid PEP 3104)
    case2_src = (
        "def outer():\n"
        "    x = 1\n"
        "    class C:\n"
        "        x = 2\n"
        "        def method(self):\n"
        "            nonlocal x\n"
        "            x += 1\n"
        "    return C\n"
    )
    case2_tree = ast.parse(case2_src)
    case2_chain = _find_enclosing_function_chain(case2_tree, 3, 7)
    case2_cls = case2_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert (
        _has_invalid_lexical_scope_declarations(
            [case2_cls], enclosing_chain=case2_chain, mod_tree=case2_tree
        )
        is False
    )

    # Case 3: class C declares global x, method declares nonlocal x (class global does not propagate into method)
    case3_src = (
        "def outer():\n"
        "    x = 1\n"
        "    class C:\n"
        "        global x\n"
        "        def method(self):\n"
        "            nonlocal x\n"
        "            x += 1\n"
        "        def method_global(self):\n"
        "            global x\n"
        "            x = 99\n"
        "    return C\n"
    )
    case3_tree = ast.parse(case3_src)
    case3_chain = _find_enclosing_function_chain(case3_tree, 3, 10)
    case3_cls = case3_tree.body[0].body[1]  # type: ignore[attr-defined]
    assert (
        _has_invalid_lexical_scope_declarations(
            [case3_cls], enclosing_chain=case3_chain, mod_tree=case3_tree
        )
        is False
    )
    muts_c3, _ = _extract_closure_nonlocal_mutations_and_deletions(case3_cls)
    assert "x" in muts_c3

    # Carrier target propagation when class body declares `global items` but method mutates outer cell `items`
    carrier_cls_src = (
        "def outer():\n"
        "    items = []\n"
        "    val = 1\n"
        "    class C:\n"
        "        global items\n"
        "        def mutate(self):\n"
        "            items.append(lambda: val)\n"
        "    return C\n"
    )
    carrier_cls_tree = ast.parse(carrier_cls_src)
    c_node = carrier_cls_tree.body[0].body[2]  # type: ignore[attr-defined]
    assert "items" in _extract_fn_outer_carrier_targets(c_node)

    # Differential execution tests for class-body nonlocal and method nonlocal
    diff_class_nonlocal_src = (
        "def entry(step: int) -> tuple[int, int]:\n"
        "    value = 10\n"
        "    class Worker:\n"
        "        nonlocal value\n"
        "        value += step\n"
        "        def bump(self) -> object:\n"
        "            nonlocal value\n"
        "            value += step * 2\n"
        "            return lambda: value\n"
        "    if step > 0:\n"
        "        w = Worker()\n"
        "        getter = w.bump()\n"
        "        res = getter()\n"
        "    else:\n"
        "        res = value\n"
        "    return res, value\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "class_body_and_method_nonlocal_hazard",
        diff_class_nonlocal_src,
        u1_bounds=(3, 9),
        expect_patch=False,
        call_args_list=[(3,), (5,)],
    )


def test_differential_global_and_nonlocal_declarations_before_inside_after() -> None:
    """Differential test for global/nonlocal declarations before, inside, and after the extracted unit."""
    from pydoppelgangerhunt.fixer.patch import _build_unit_delegation_call

    def _run_differential_case(
        src: str,
        u1_lines: tuple[int, int],
        u2_lines: tuple[int, int],
        eval_expr: str,
        *,
        expect_permitted: bool,
        expected_hazard_reason: str | None = None,
    ) -> None:
        # Every differential test input must be valid, compilable Python
        compile(src, "<orig>", "exec")
        u1 = {
            "name": "u1",
            "file": "sample.py",
            "kind": "compound_block",
            "start": u1_lines[0],
            "end": u1_lines[1],
            "source_text": src,
        }
        u2 = {
            "name": "u2",
            "file": "sample.py",
            "kind": "compound_block",
            "start": u2_lines[0],
            "end": u2_lines[1],
            "source_text": src,
        }
        s1 = inspect_single_unit_scope(u1)
        s2 = inspect_single_unit_scope(u2)
        if expected_hazard_reason is not None:
            assert expected_hazard_reason in s1["closure_hazard_reasons"]
            assert expected_hazard_reason in s2["closure_hazard_reasons"]
        pair_outs = resolve_clone_pair_outputs(u1, u2, scope1=s1, scope2=s2)
        if not expect_permitted:
            assert pair_outs is None
            return
        assert pair_outs is not None
        outs1, outs2 = pair_outs
        pair_scope = analyze_unit_variable_scope(u1, u2, scope1=s1, scope2=s2)
        helper_code = synthesize_shared_helper_code(
            u1,
            u2,
            helper_name="_extracted_helper",
            scope=pair_scope,
            scope1=s1,
            scope2=s2,
            outputs=outs1,
            outputs2=outs2,
        )
        assert "def _extracted_helper(" in helper_code
        orig_lines = src.splitlines(keepends=True)
        inputs = list(pair_scope["inputs"])
        rep1 = _build_unit_delegation_call(
            u1,
            src,
            orig_lines,
            effective_binding="none",
            helper_name="_extracted_helper",
            inputs=inputs,
            outputs=outs1,
            scope=pair_scope,
            target_inputs=list(s1["inputs"]),
            target_outputs=outs1,
        )
        rep2 = _build_unit_delegation_call(
            u2,
            src,
            orig_lines,
            effective_binding="none",
            helper_name="_extracted_helper",
            inputs=inputs,
            outputs=outs1,
            scope=pair_scope,
            target_inputs=list(s2["inputs"]),
            target_outputs=outs2,
        )
        refactored_lines = (
            [helper_code + "\n"]
            + orig_lines[: u1_lines[0] - 1]
            + [rep1]
            + orig_lines[u1_lines[1] : u2_lines[0] - 1]
            + [rep2]
            + orig_lines[u2_lines[1] :]
        )
        refactored_src = "".join(refactored_lines)
        compile(refactored_src, "<ref>", "exec")
        ns_orig: dict[str, Any] = {}
        ns_ref: dict[str, Any] = {}
        exec(src, ns_orig)
        orig_val = eval(eval_expr, ns_orig)
        exec(refactored_src, ns_ref)
        ref_val = eval(eval_expr, ns_ref)
        assert ref_val == orig_val

    # 1. global declaration BEFORE the extracted unit (permitted when unit does not mutate or call-clobber global;
    #    rejected when unit mutates global or calls a helper while reading global)
    src_global_before_ok = (
        "G = 10\n"
        "def f1(n):\n"
        "    global G\n"
        "    G = G + 1\n"
        "    a = n + G\n"
        "    b = a * 2\n"
        "    return b\n"
        "def f2(n):\n"
        "    global G\n"
        "    G = G + 2\n"
        "    a = n + G\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    _run_differential_case(
        src_global_before_ok,
        (5, 6),
        (11, 12),
        "(f1(3), f2(4), G)",
        expect_permitted=True,
    )

    src_global_before_mutated = (
        "G = 10\n"
        "def f1(n):\n"
        "    global G\n"
        "    G = n + 1\n"
        "    b = G * 2\n"
        "    return b\n"
        "def f2(n):\n"
        "    global G\n"
        "    G = n + 1\n"
        "    b = G * 2\n"
        "    return b\n"
    )
    _run_differential_case(
        src_global_before_mutated,
        (4, 5),
        (10, 11),
        "(f1(3), f2(4), G)",
        expect_permitted=False,
    )

    # Outer global `G` read inside unit alongside a call that mutates `G` -> rejected ("global_escape")
    src_global_before_call_mutates_g = (
        "G = 10\n"
        "def mutate_g():\n"
        "    global G\n"
        "    G = 99\n"
        "    return 1\n"
        "def f1(n):\n"
        "    global G\n"
        "    a = mutate_g() + n\n"
        "    b = G + a\n"
        "    return b\n"
        "def f2(n):\n"
        "    global G\n"
        "    a = mutate_g() + n\n"
        "    b = G + a\n"
        "    return b\n"
    )
    _run_differential_case(
        src_global_before_call_mutates_g,
        (8, 9),
        (13, 14),
        "(f1(3), f2(4), G)",
        expect_permitted=False,
        expected_hazard_reason="global_escape",
    )

    # 2. global declaration INSIDE the extracted unit (permitted when global is not touched outside unit;
    #    rejected when enclosing function also mutates global outside unit)
    src_global_inside_ok = (
        "G = 10\n"
        "def f1(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    b = G * 2\n"
        "    return b\n"
        "def f2(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    b = G * 2\n"
        "    return b\n"
    )
    _run_differential_case(
        src_global_inside_ok,
        (3, 5),
        (8, 10),
        "(f1(3), G, f2(4), G)",
        expect_permitted=True,
    )

    src_global_inside_escaped = (
        "G = 10\n"
        "def f1(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    G = G + 1\n"
        "    return G\n"
        "def f2(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    G = G + 1\n"
        "    return G\n"
    )
    _run_differential_case(
        src_global_inside_escaped,
        (3, 4),
        (8, 9),
        "(f1(3), f2(4), G)",
        expect_permitted=False,
        expected_hazard_reason="global_escape",
    )

    # 3. global declaration AFTER the extracted unit
    src_global_after_ok = (
        "G = 10\n"
        "def f1(n):\n"
        "    a = n + 5\n"
        "    b = a * 2\n"
        "    global G\n"
        "    G = b + 1\n"
        "    return G\n"
        "def f2(n):\n"
        "    a = n + 5\n"
        "    b = a * 2\n"
        "    global G\n"
        "    G = b + 2\n"
        "    return G\n"
    )
    _run_differential_case(
        src_global_after_ok,
        (3, 4),
        (9, 10),
        "(f1(3), G, f2(4), G)",
        expect_permitted=True,
    )

    src_global_declared_and_reassigned_after = (
        "G = 10\n"
        "def f1(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    b = G * 2\n"
        "    G = b + 1\n"
        "    return b\n"
        "def f2(n):\n"
        "    global G\n"
        "    G = n + 5\n"
        "    b = G * 2\n"
        "    G = b + 1\n"
        "    return b\n"
    )
    _run_differential_case(
        src_global_declared_and_reassigned_after,
        (4, 5),
        (10, 11),
        "(f1(3), f2(4), G)",
        expect_permitted=False,
    )

    # 4. nonlocal declaration BEFORE, INSIDE, and AFTER the extracted unit
    src_nonlocal_before_ok = (
        "def run(n):\n"
        "    cell = 5\n"
        "    def f1(x):\n"
        "        nonlocal cell\n"
        "        cell = cell + 1\n"
        "        a = x + cell\n"
        "        b = a * 3\n"
        "        return b\n"
        "    def f2(x):\n"
        "        nonlocal cell\n"
        "        cell = cell + 2\n"
        "        a = x + cell\n"
        "        b = a * 3\n"
        "        return b\n"
        "    return f1(n), f2(n), cell\n"
    )
    _run_differential_case(
        src_nonlocal_before_ok,
        (6, 7),
        (12, 13),
        "run(4)",
        expect_permitted=True,
    )

    src_nonlocal_inside_rejected = (
        "def run(n):\n"
        "    cell = 5\n"
        "    def f1(x):\n"
        "        nonlocal cell\n"
        "        cell = x + 1\n"
        "        return cell\n"
        "    def f2(x):\n"
        "        nonlocal cell\n"
        "        cell = x + 1\n"
        "        return cell\n"
        "    return f1(n), f2(n), cell\n"
    )
    _run_differential_case(
        src_nonlocal_inside_rejected,
        (4, 5),
        (8, 9),
        "run(4)",
        expect_permitted=False,
    )

    src_nonlocal_after_ok = (
        "def run(n):\n"
        "    cell = 5\n"
        "    def f1(x):\n"
        "        a = x + 2\n"
        "        b = a * 3\n"
        "        nonlocal cell\n"
        "        cell = b + 1\n"
        "        return cell\n"
        "    def f2(x):\n"
        "        a = x + 2\n"
        "        b = a * 3\n"
        "        nonlocal cell\n"
        "        cell = b + 2\n"
        "        return cell\n"
        "    return f1(n), f2(n), cell\n"
    )
    _run_differential_case(
        src_nonlocal_after_ok,
        (4, 5),
        (10, 11),
        "run(4)",
        expect_permitted=True,
    )

    src_nonlocal_before_mutated = (
        "def run(n):\n"
        "    cell = 5\n"
        "    def f1(x):\n"
        "        nonlocal cell\n"
        "        cell = x + 2\n"
        "        b = cell * 3\n"
        "        return b\n"
        "    def f2(x):\n"
        "        nonlocal cell\n"
        "        cell = x + 2\n"
        "        b = cell * 3\n"
        "        return b\n"
        "    return f1(n), f2(n), cell\n"
    )
    _run_differential_case(
        src_nonlocal_before_mutated,
        (5, 6),
        (10, 11),
        "run(4)",
        expect_permitted=True,
    )


def test_context_manager_provenance_definite_bindings_and_termination() -> None:
    """Covers context-manager definite bindings, termination, provenance, and branch aliases."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _find_enclosing_chain_definite_pre_bound,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _classify_with_suppression,
        _clear_name_provenance,
        _extract_block_definite_bindings_and_term,
        _is_proven_non_suppressing_call,
        _is_proven_non_suppressing_target_expr,
        _seed_scope_suppress_aliases,
        _split_unknown_with_body,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _analyze_block_assignment,
        _block_terminates,
    )

    # Unknown context managers do not establish definite bindings on implicit raises
    unknown_attr_src = (
        "def example(obj, condition):\n"
        "    with MaybeSuppress():\n"
        "        value = obj.attribute\n"
        "    if condition:\n"
        "        value = 1\n"
        "    result = value + 1\n"
        "    return result\n"
    )
    unknown_attr_tree = ast.parse(unknown_attr_src)
    chain = _find_enclosing_function_chain(unknown_attr_tree, 4, 6)
    pre_bound = _find_enclosing_chain_definite_pre_bound(
        chain, 4, 6, mod_tree=unknown_attr_tree
    )
    assert "obj" in pre_bound
    assert "value" not in pre_bound
    sc_unknown_attr = inspect_single_unit_scope(
        {
            "start": 4,
            "end": 6,
            "kind": "compound_block",
            "source_text": unknown_attr_src,
            "source_code": (
                "if condition:\n"
                "    value = 1\n"
                "result = value + 1\n"
            ),
        },
        tree=unknown_attr_tree,
    )
    assert "severed_closure_cell" in sc_unknown_attr["control_flow_hazards"]

    with_body_stmts = ast.parse("a = 1\nb = obj.attr\nc = 2\n").body
    safe_pre, risky_suf = _split_unknown_with_body(with_body_stmts)
    assert len(safe_pre) == 1
    assert len(risky_suf) == 2

    # Unknown context manager with explicit raise or raising return does not terminate
    term_raise_stmts = ast.parse(
        "with MaybeSuppress():\n    raise RuntimeError()\n"
    ).body
    bound_r, term_r = _extract_block_definite_bindings_and_term(
        term_raise_stmts, None
    )
    assert bound_r == set()
    assert term_r is False
    assert _block_terminates(term_raise_stmts) is False

    term_calc_ret_stmts = ast.parse(
        "with MaybeSuppress():\n    return calculate()\n"
    ).body
    _, term_calc = _extract_block_definite_bindings_and_term(
        term_calc_ret_stmts, None
    )
    assert term_calc is False
    assert _block_terminates(term_calc_ret_stmts) is False

    # Return of a literal inside unknown context manager DOES terminate
    term_const_ret_stmts = ast.parse(
        "with MaybeSuppress():\n    return 1\n"
    ).body
    _, term_const = _extract_block_definite_bindings_and_term(
        term_const_ret_stmts, None
    )
    assert term_const is True
    assert _block_terminates(term_const_ret_stmts) is True

    # Non-suppressing context manager with raising return DOES terminate
    nonsup_ret_stmts = ast.parse(
        "import contextlib\nwith contextlib.nullcontext():\n    return calculate()\n"
    ).body
    _, term_nonsup = _extract_block_definite_bindings_and_term(
        nonsup_ret_stmts, None
    )
    assert term_nonsup is True
    assert _block_terminates(nonsup_ret_stmts) is True

    # Shadowed `open` and arbitrary `obj.open()` / `obj.nullcontext()` are not trusted as non-suppressing
    shadow_open_src = (
        "class SuppressAll:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return True\n"
        "def open():\n"
        "    return SuppressAll()\n"
        "def example():\n"
        "    with open():\n"
        "        raise RuntimeError()\n"
        "    del x\n"
        "    return 1\n"
    )
    shadow_open_tree = ast.parse(shadow_open_src)
    sc_shadow_open = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_text": shadow_open_src,
            "source_code": "with open():\n    raise RuntimeError()\ndel x\n",
        },
        tree=shadow_open_tree,
    )
    assert "x" in sc_shadow_open["inputs"]
    assert "x" in sc_shadow_open["deleted_names"]

    param_shadow_src = (
        "def example(open):\n"
        "    with open():\n"
        "        x = may_fail()\n"
        "    return x\n"
    )
    param_shadow_sc = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 3,
            "kind": "compound_block",
            "source_text": param_shadow_src,
            "source_code": "with open():\n    x = may_fail()\n",
        },
        tree=ast.parse(param_shadow_src),
    )
    assert "x" not in param_shadow_sc["definite_stores"]
    assert "x" in param_shadow_sc["conditional_outputs"]

    with_obj_open = ast.parse("with obj.open():\n    pass\n").body[0]
    with_obj_null = ast.parse("with obj.nullcontext():\n    pass\n").body[0]
    assert isinstance(with_obj_open, ast.With)
    assert isinstance(with_obj_null, ast.With)
    assert _classify_with_suppression(with_obj_open) == "unknown"
    assert _classify_with_suppression(with_obj_null) == "unknown"

    stdlib_proven_stmts = ast.parse(
        "import contextlib as cl\n"
        "from tempfile import TemporaryDirectory as TD\n"
        "nc = cl.nullcontext\n"
        "td = TD\n"
        "with cl.nullcontext(), td(), nc(), open('f'):\n"
        "    val = compute()\n"
    ).body
    bound_proven, _ = _extract_block_definite_bindings_and_term(
        stdlib_proven_stmts, None
    )
    assert "val" in bound_proven

    star_imp_stmts = ast.parse(
        "from custom_io import *\n"
        "with open('f'):\n"
        "    val = compute()\n"
    ).body
    bound_star, _ = _extract_block_definite_bindings_and_term(
        star_imp_stmts, None
    )
    assert "val" not in bound_star

    sup_prov = _SuppressProvenance(
        non_suppressors=frozenset({"nc"}),
        stdlib_modules=frozenset({("cl", "contextlib")}),
    )
    assert _clear_name_provenance("", sup_prov) == sup_prov
    assert "nc" in sup_prov.non_suppressors
    cleared_prov = _clear_name_provenance("nc", sup_prov)
    assert "nc" not in cleared_prov.non_suppressors
    assert (
        _is_proven_non_suppressing_target_expr(ast.Constant(value=1), cleared_prov)
        is False
    )
    assert (
        _is_proven_non_suppressing_call(ast.Name(id="nc", ctx=ast.Load()), cleared_prov)
        is False
    )
    assert (
        _is_proven_non_suppressing_call(
            ast.parse("(1)()").body[0].value,  # type: ignore[attr-defined]
            cleared_prov,
        )
        is False
    )

    seed_src = (
        "import contextlib as cl\n"
        "from contextlib import suppress as sup_mod\n"
        "def outer(open):\n"
        "    from pkg import *\n"
        "    class LocalSup:\n"
        "        def __exit__(self, *a):\n"
        "            return True\n"
        "    if (w_sup := sup_mod) and (w_nc := cl.nullcontext):\n"
        "        pass\n"
        "    ann_var: int = 1\n"
        "    for loop_v in (1,):\n"
        "        del loop_v\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner\n"
    )
    seed_tree = ast.parse(seed_src)
    outer_fn = seed_tree.body[2]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[-2]
    ctx = _get_fn_scope_context(inner_fn, seed_tree)
    seeded = _seed_scope_suppress_aliases(inner_fn, seed_tree, ctx)
    assert "open" in seeded.shadowed_builtins
    assert seeded.has_star_import is True
    assert "LocalSup" in seeded.suppressors
    assert "sup_mod" in seeded.suppressors

    branch_alias_stmts = ast.parse(
        "from contextlib import suppress\n"
        "try:\n"
        "    cm = suppress\n"
        "except Exception as cm_exc:\n"
        "    cm = suppress\n"
        "else:\n"
        "    cm_else = suppress\n"
        "finally:\n"
        "    cm_fin = suppress\n"
        "with cm(ValueError):\n"
        "    x = int('bad')\n"
        "with cm_else(ValueError):\n"
        "    y = int('bad')\n"
        "with cm_fin(ValueError):\n"
        "    z = int('bad')\n"
    ).body
    b_try, _ = _extract_block_definite_bindings_and_term(
        branch_alias_stmts, None
    )
    assert "x" not in b_try
    assert "y" not in b_try
    assert "z" not in b_try
    def_try, cond_try = _analyze_block_assignment(branch_alias_stmts)
    assert "x" not in def_try and "x" in cond_try
    assert "y" not in def_try and "y" in cond_try
    assert "z" not in def_try and "z" in cond_try

    loop_alias_stmts = ast.parse(
        "from contextlib import suppress\n"
        "while cond:\n"
        "    cm_w = suppress\n"
        "    break\n"
        "for item in seq:\n"
        "    cm_f = suppress\n"
        "with cm_w(ValueError):\n"
        "    b = int('bad')\n"
        "with cm_f(ValueError):\n"
        "    c = int('bad')\n"
    ).body
    b_loop, _ = _extract_block_definite_bindings_and_term(
        loop_alias_stmts, None
    )
    assert not ({"b", "c"} & b_loop)
    def_loop, cond_loop = _analyze_block_assignment(loop_alias_stmts)
    assert not ({"b", "c"} & def_loop)
    assert {"b", "c"} <= cond_loop

    if sys.version_info >= (3, 10):
        match_alias_stmts = ast.parse(
            "from contextlib import suppress\n"
            "match flag:\n"
            "    case 1 as m_var:\n"
            "        cm_m = suppress\n"
            "    case _:\n"
            "        cm_m = open\n"
            "with cm_m(ValueError):\n"
            "    a = int('bad')\n"
        ).body
        b_match, _ = _extract_block_definite_bindings_and_term(
            match_alias_stmts, None
        )
        assert "a" not in b_match
        def_match, cond_match = _analyze_block_assignment(match_alias_stmts)
        assert "a" not in def_match
        assert "a" in cond_match


def test_lexical_closure_semantics_lambda_comprehension_and_class_attributes() -> None:
    """Covers enclosing lambda/comprehension cell analysis and class attribute non-shadowing."""
    from pydoppelgangerhunt.fixer._df_closures import (
        _extract_nested_scope_free_reads,
        _extract_scope_local_names,
        _is_span_in_scope_header,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _has_unresolved_span_bounds,
    )

    # Enclosing lambda scope captures and mutates x via walrus
    lambda_src = (
        "callback = lambda x: (\n"
        "    (lambda: x),\n"
        "    (x := 2),\n"
        ")[0]\n"
    )
    lambda_tree = ast.parse(lambda_src)
    chain_lam = _find_enclosing_function_chain(lambda_tree, 2, 2)
    assert any(isinstance(n, ast.Lambda) for _, _, n in chain_lam)
    sc_inner_lam = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "closure",
            "source_text": lambda_src,
            "source_code": "(lambda: x)",
        },
        tree=lambda_tree,
    )
    assert "severed_closure_cell" in sc_inner_lam["control_flow_hazards"]
    assert sc_inner_lam["is_control_flow_safe"] is False

    sc_walrus_in_lam = inspect_single_unit_scope(
        {
            "start": 3,
            "end": 3,
            "kind": "complex_expr",
            "source_text": lambda_src,
            "source_code": "(x := 2)",
        },
        tree=lambda_tree,
    )
    assert "severed_closure_cell" in sc_walrus_in_lam["control_flow_hazards"]

    # Enclosing comprehension scopes (ListComp, SetComp, DictComp, GeneratorExp)
    comp_src = "callbacks = [(lambda: x) for x in range(3)]\n"
    comp_tree = ast.parse(comp_src)
    chain_comp = _find_enclosing_function_chain(
        comp_tree, 1, 1, u_start_col=13, u_end_col=24
    )
    assert any(isinstance(n, ast.ListComp) for _, _, n in chain_comp)
    sc_comp_lam = inspect_single_unit_scope(
        {
            "start": 1,
            "end": 1,
            "kind": "closure",
            "source_text": comp_src,
            "source_code": "(lambda: x)",
        },
        tree=comp_tree,
    )
    assert "severed_closure_cell" in sc_comp_lam["control_flow_hazards"]
    assert sc_comp_lam["is_control_flow_safe"] is False

    dict_comp_src = (
        "d_callbacks = {\n"
        "    k: (lambda: k + w)\n"
        "    for k in range(3)\n"
        "    if (w := k + 1)\n"
        "}\n"
    )
    dict_comp_tree = ast.parse(dict_comp_src)
    sc_dict_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "closure",
            "source_text": dict_comp_src,
            "source_code": "(lambda: k + w)",
        },
        tree=dict_comp_tree,
    )
    assert "severed_closure_cell" in sc_dict_comp["control_flow_hazards"]

    list_comp_node = comp_tree.body[0].value  # type: ignore[attr-defined]
    assert isinstance(list_comp_node, ast.ListComp)
    assert _is_span_in_scope_header(list_comp_node, 1, 1, 29, 37) is True
    assert _is_span_in_scope_header(list_comp_node, 1, 1, 13, 24) is False
    assert (
        _has_unresolved_span_bounds(list_comp_node, 1, 1, None, None) is True
    )
    lam_with_default = ast.parse("f = lambda x=y + 1: x\n").body[0].value.args.defaults[0]  # type: ignore[attr-defined]
    lam_node = ast.parse("f = lambda x=y + 1: x\n").body[0].value  # type: ignore[attr-defined]
    assert (
        _is_span_in_scope_header(
            lam_node,
            1,
            1,
            lam_with_default.col_offset,
            lam_with_default.end_col_offset,
        )
        is True
    )

    # Class attributes do NOT shadow names inside methods (attached and detached)
    cls_shadow_src = (
        "def outer(sum):\n"
        "    class Calculator:\n"
        "        sum = 123\n"
        "        def total(self, values):\n"
        "            return sum(x for x in values)\n"
        "    return Calculator\n"
    )
    cls_shadow_tree = ast.parse(cls_shadow_src)
    outer_fn = cls_shadow_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    calc_cls = outer_fn.body[0]
    assert isinstance(calc_cls, ast.ClassDef)
    total_method = calc_cls.body[1]
    assert isinstance(total_method, ast.FunctionDef)

    method_locals_attached = _extract_scope_local_names(
        total_method, module_tree=cls_shadow_tree
    )
    assert "sum" in method_locals_attached

    cls_only_src = (
        "class Calculator:\n"
        "    sum = 123\n"
        "    def total(self, values):\n"
        "        return sum(x for x in values)\n"
    )
    cls_only_tree = ast.parse(cls_only_src)
    cls_only_node = cls_only_tree.body[0]
    assert isinstance(cls_only_node, ast.ClassDef)
    total_only_method = cls_only_node.body[1]
    assert isinstance(total_only_method, ast.FunctionDef)
    genexp_only = total_only_method.body[0].value.args[0]  # type: ignore[attr-defined]

    assert "sum" not in _extract_scope_local_names(
        total_only_method, module_tree=cls_only_tree
    )
    assert "sum" not in _extract_scope_local_names(
        genexp_only, module_tree=cls_only_tree
    )

    clear_dataflow_cache()
    assert "sum" not in _extract_scope_local_names(
        total_only_method, module_tree=None
    )
    assert "sum" not in _extract_scope_local_names(
        genexp_only, module_tree=None
    )

    outer_cls_method_src = (
        "def outer():\n"
        "    items = [1, 2]\n"
        "    class Holder:\n"
        "        items = None\n"
        "        def get_items(self):\n"
        "            return [x for x in items]\n"
        "    items = [3, 4]\n"
        "    res = Holder().get_items()\n"
        "    return res\n"
    )
    outer_cls_tree = ast.parse(outer_cls_method_src)
    outer_cls_fn = outer_cls_tree.body[0]
    assert "items" in _extract_nested_scope_free_reads(outer_cls_fn.body[1])  # type: ignore[attr-defined]
    sc_rebind_items = inspect_single_unit_scope(
        {
            "start": 7,
            "end": 8,
            "kind": "linear_block",
            "source_text": outer_cls_method_src,
            "source_code": "items = [3, 4]\nres = Holder().get_items()\n",
        },
        tree=outer_cls_tree,
    )
    assert "severed_closure_cell" in sc_rebind_items["control_flow_hazards"]


def test_tracked_cached_dict_ior_and_stale_source_and_whitespace_source_text() -> None:
    """Covers _TrackedCachedDict.__ior__, ordered contiguous source matching, and whitespace source_text."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _evict_tree_scopes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _is_ordered_contiguous_source_match,
        ensure_unit_scope_closure_mode,
    )

    clear_dataflow_cache()
    mod_node = ast.parse("x = 1\ny = 2\n")
    stmt_node = mod_node.body[1]
    cache_key = _ScopeCacheKey(owner=mod_node, kind="test_ior")
    tracked_d = _get_or_compute_scope_cached(
        _scope_cell_info_cache, cache_key, dict
    )
    assert isinstance(tracked_d, _TrackedCachedDict)
    assert (tracked_d | {stmt_node: "preview"}) == {stmt_node: "preview"}
    tracked_d |= {stmt_node: "payload"}
    assert stmt_node in _scope_owner_keys_index
    _evict_tree_scopes(stmt_node)
    assert cache_key not in _scope_cell_info_cache

    real_src = (
        "def f(x):\n"
        "    x = 1\n"
        "    del x\n"
        "    return 0\n"
    )
    real_tree = ast.parse(real_src)
    reordered_unit = {
        "start": 2,
        "end": 3,
        "kind": "linear_block",
        "source_text": real_src,
        "source_code": "del x\nx = 1\n",
    }
    sc_reordered = inspect_single_unit_scope(reordered_unit, tree=real_tree)
    assert "stale_source_code" in sc_reordered["control_flow_hazards"]
    assert sc_reordered["is_control_flow_safe"] is False

    dup_src = (
        "def f(x):\n"
        "    x += 1\n"
        "    y = 2\n"
    )
    dup_unit = {
        "start": 2,
        "end": 3,
        "kind": "linear_block",
        "source_text": dup_src,
        "source_code": "x += 1\nx += 1\n",
    }
    sc_dup = inspect_single_unit_scope(dup_unit, tree=ast.parse(dup_src))
    assert "stale_source_code" in sc_dup["control_flow_hazards"]
    assert sc_dup["is_control_flow_safe"] is False

    noncontig_src = (
        "def f(a):\n"
        "    b = a + 1\n"
        "    c = b * 2\n"
        "    d = c - 1\n"
    )
    noncontig_unit = {
        "start": 2,
        "end": 4,
        "kind": "linear_block",
        "source_text": noncontig_src,
        "source_code": "b = a + 1\nd = c - 1\n",
    }
    sc_noncontig = inspect_single_unit_scope(
        noncontig_unit, tree=ast.parse(noncontig_src)
    )
    assert "stale_source_code" in sc_noncontig["control_flow_hazards"]
    assert sc_noncontig["is_control_flow_safe"] is False

    assert _is_ordered_contiguous_source_match([], ["a = 1"]) is False
    assert (
        _is_ordered_contiguous_source_match(
            ["a = 1", "b = 2"], ["x = 0; a = 1", "b = 2; y = 3"]
        )
        is True
    )
    assert (
        _is_ordered_contiguous_source_match(
            ["a = 1", "b = 2"], ["a = 1", "other = 9"]
        )
        is False
    )

    ws_src = (
        "def f1(a):\n"
        "    r = a + 10\n"
        "    s = r * 2\n"
        "    return s\n"
        "def f2(a):\n"
        "    r = a + 10\n"
        "    s = r * 2\n"
        "    return s\n"
    )
    u1_ws = {
        "name": "f1",
        "start": 2,
        "end": 3,
        "kind": "linear_block",
        "source_text": "   \n\t  ",
        "source_code": "    r = a + 10\n    s = r * 2\n",
    }
    u2_ws = {
        "name": "f2",
        "start": 6,
        "end": 7,
        "kind": "linear_block",
        "source_text": "   \n  ",
        "source_code": "    r = a + 10\n    s = r * 2\n",
    }
    sc_ensured = ensure_unit_scope_closure_mode(
        u1_ws, None, False, source_text=ws_src
    )
    assert sc_ensured["is_control_flow_safe"] is True
    assert "a" in sc_ensured["inputs"]
    helper_ws = synthesize_shared_helper_code(
        u1_ws,
        u2_ws,
        helper_name="_ws_helper",
        source_text1=ws_src,
        source_text2=ws_src,
    )
    assert "def _ws_helper(" in helper_ws


def test_differential_execution_closure_and_context_manager_hazards() -> None:
    """Differential execution suite verifying runtime equivalence or fail-closed rejection across context managers, lambda/comprehension scopes, and class attributes."""
    # 1. Unknown context manager suppressing implicit AttributeError before conditional assignment
    src_unknown_cm = (
        "class MaybeSuppress:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return exc_type is AttributeError\n"
        "def run(obj, cond):\n"
        "    with MaybeSuppress():\n"
        "        value = obj.missing_attr\n"
        "    if cond:\n"
        "        value = 41\n"
        "    res = value + 1\n"
        "    return res\n"
    )
    ns_cm: dict[str, object] = {}
    exec(src_unknown_cm, ns_cm)  # nosec B102
    orig_cm_fn = ns_cm["run"]
    assert callable(orig_cm_fn)
    assert orig_cm_fn(object(), True) == 42
    sc_cm = inspect_single_unit_scope(
        {
            "start": 9,
            "end": 11,
            "kind": "compound_block",
            "source_text": src_unknown_cm,
            "source_code": "if cond:\n    value = 41\nres = value + 1\n",
        },
        tree=ast.parse(src_unknown_cm),
    )
    assert "severed_closure_cell" in sc_cm["control_flow_hazards"]

    # 2. Enclosing lambda with walrus mutation vs inner lambda closure
    src_lambda_cell = (
        "def run():\n"
        "    callback = lambda x: ((lambda: x), (x := x + 10))[0]\n"
        "    fn = callback(5)\n"
        "    return fn()\n"
    )
    ns_lam: dict[str, object] = {}
    exec(src_lambda_cell, ns_lam)  # nosec B102
    assert ns_lam["run"]() == 15  # type: ignore[operator]
    sc_lam = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "closure",
            "source_text": src_lambda_cell,
            "source_code": "(lambda: x)",
        },
        tree=ast.parse(src_lambda_cell),
    )
    assert "severed_closure_cell" in sc_lam["control_flow_hazards"]

    # 3. Enclosing comprehension capturing loop variable in lambda
    src_comp_cell = (
        "def run():\n"
        "    cbs = [(lambda: x) for x in range(3)]\n"
        "    return [cb() for cb in cbs]\n"
    )
    ns_comp: dict[str, object] = {}
    exec(src_comp_cell, ns_comp)  # nosec B102
    assert ns_comp["run"]() == [2, 2, 2]  # type: ignore[operator]
    sc_comp = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "closure",
            "source_text": src_comp_cell,
            "source_code": "(lambda: x)",
        },
        tree=ast.parse(src_comp_cell),
    )
    assert "severed_closure_cell" in sc_comp["control_flow_hazards"]

    # 4. Class attribute with same name as builtin `sum` used in method generator expression
    src_cls_sum = (
        "class Calculator:\n"
        "    sum = 123\n"
        "    def total1(self, values):\n"
        "        scaled = [v * 2 for v in values]\n"
        "        res = sum(x for x in scaled)\n"
        "        return res\n"
        "    def total2(self, values):\n"
        "        scaled = [v * 2 for v in values]\n"
        "        res = sum(x for x in scaled)\n"
        "        return res\n"
    )
    cls_tree = ast.parse(src_cls_sum)
    u_c1 = {
        "name": "total1",
        "start": 4,
        "end": 5,
        "kind": "linear_block",
        "source_text": src_cls_sum,
        "source_code": "        scaled = [v * 2 for v in values]\n        res = sum(x for x in scaled)\n",
    }
    u_c2 = {
        "name": "total2",
        "start": 8,
        "end": 9,
        "kind": "linear_block",
        "source_text": src_cls_sum,
        "source_code": "        scaled = [v * 2 for v in values]\n        res = sum(x for x in scaled)\n",
    }
    comb_cls = analyze_unit_variable_scope(u_c1, u_c2, tree1=cls_tree, tree2=cls_tree)
    assert "sum" not in comb_cls["inputs"]
    helper_cls = synthesize_shared_helper_code(
        u_c1,
        u_c2,
        helper_name="_calc_helper",
        scope=comb_cls,
        outputs=["res"],
        outputs2=["res"],
        source_text1=src_cls_sum,
        source_text2=src_cls_sum,
        tree1=cls_tree,
        tree2=cls_tree,
    )
    assert "def _calc_helper(values" in helper_cls
    ns_cls: dict[str, object] = {}
    exec(src_cls_sum + "\n" + helper_cls, ns_cls)  # nosec B102
    calc_inst = ns_cls["Calculator"]()  # type: ignore[operator]
    helper_fn = ns_cls["_calc_helper"]
    assert callable(helper_fn)
    assert calc_inst.total1([1, 2, 3]) == helper_fn([1, 2, 3]) == 12


def test_seeded_generated_closure_program_differential_fuzzer(tmp_path: Path) -> None:
    """Seeded differential fuzzer generating nested-closure programs and verifying runtime equivalence."""
    import itertools  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel
    import random  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    import pytest  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )

    if shutil.which("git") is None:
        pytest.skip("git executable is required for differential fuzzer patch application")

    all_combinations = [
        (has_pre, mut_acc, call_sib, cap_mode, post_mut, use_with)
        for has_pre, mut_acc, call_sib, cap_mode, post_mut, use_with in itertools.product(
            [False, True],
            [False, True],
            [False, True],
            ["default_arg", "late_cell", "nonlocal_inc", "none"],
            [False, True],
            [False, True],
        )
        if not (call_sib and not has_pre)
    ]
    rng = random.Random(20261106)
    rng.shuffle(all_combinations)
    seed_limit = int(os.environ.get("PDH_FUZZ_SEEDS", str(len(all_combinations))))
    selected_combinations = all_combinations[:seed_limit]
    patched_count = 0
    blocked_count = 0
    git_apply_count = 0

    for idx, (
        has_pre_sibling,
        mutate_acc_in_unit,
        call_sibling_in_unit,
        capture_mode,
        post_mutate_scale,
        use_with_lock,
    ) in enumerate(selected_combinations):
        lines: list[str] = [
            "import contextlib",
            "def entry(items, cond):",
            "    scale = 3",
            "    acc = 1",
            "    cbs = []",
        ]
        if has_pre_sibling:
            lines.extend(
                [
                    "    def read_acc():",
                    "        return acc",
                ]
            )
        u_start = len(lines) + 1
        if use_with_lock:
            lines.append("    with contextlib.nullcontext():")
            indent = "        "
        else:
            indent = "    "
        lines.append(f"{indent}for x in items:")
        loop_indent = indent + "    "
        if mutate_acc_in_unit:
            lines.append(f"{loop_indent}acc = acc + x")
        else:
            lines.append(f"{loop_indent}step = x + scale")
        if call_sibling_in_unit:
            lines.append(f"{loop_indent}cbs.append(read_acc())")
        elif capture_mode == "default_arg":
            lines.append(f"{loop_indent}cbs.append((lambda v, s=scale, i=x: v + s + i)(2))")
        elif capture_mode == "late_cell":
            lines.append(f"{loop_indent}cbs.append(lambda v: v + scale + x)")
        elif capture_mode == "nonlocal_inc":
            lines.extend(
                [
                    f"{loop_indent}def bump():",
                    f"{loop_indent}    nonlocal acc",
                    f"{loop_indent}    acc += 1",
                    f"{loop_indent}bump()",
                ]
            )
        else:
            lines.append(f"{loop_indent}cbs.append(x * scale)")
        u_end = len(lines)
        if post_mutate_scale:
            lines.append("    scale = 10")
        if not call_sibling_in_unit and capture_mode == "late_cell":
            lines.append("    out = [fn(1) for fn in cbs]")
        else:
            lines.append("    out = list(cbs)")
        lines.append("    return out, acc, scale")

        fn_src = "\n".join(lines) + "\n"
        fn_b_src = fn_src.replace("def entry(", "def entry_b(", 1)
        line_offset = len(lines)
        combined_src = f"{fn_src}{fn_b_src}"
        fuzz_tree = ast.parse(combined_src)
        fuzz_fn = fuzz_tree.body[1]
        assert isinstance(fuzz_fn, ast.FunctionDef)
        span_bounds = _SpanBounds(u_start, u_end)

        for mode in ("strict", "pedantic"):
            strict_w = mode == "pedantic"
            info_with_deltas = _analyze_enclosing_fn_cells_uncached(
                fuzz_fn,
                span_bounds,
                mod_tree=fuzz_tree,
                use_stmt_deltas=True,
                strict_with_suppression=strict_w,
            )
            info_without_deltas = _analyze_enclosing_fn_cells_uncached(
                fuzz_fn,
                span_bounds,
                mod_tree=fuzz_tree,
                use_stmt_deltas=False,
                strict_with_suppression=strict_w,
            )
            assert info_with_deltas == info_without_deltas

            if idx >= 24:
                continue

            case_dir = tmp_path / f"fuzz_{idx}_{mode}"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / ".git").mkdir(exist_ok=True)
            mod_file = case_dir / "mod.py"
            mod_file.write_text(combined_src, encoding="utf-8")

            u1 = {
                "file": str(mod_file),
                "start": u_start,
                "end": u_end,
                "name": "entry:block",
                "kind": "compound_block",
            }
            u2 = {
                "file": str(mod_file),
                "start": u_start + line_offset,
                "end": u_end + line_offset,
                "name": "entry_b:block",
                "kind": "compound_block",
            }
            patch = generate_refactoring_patch(
                [(1.0, u1, u2)],
                repo_root=str(case_dir),
                replace_clones=True,
                closure_strictness=mode,
            )
            if patch:
                patched_count += 1
                git_apply_count += 1
                subprocess.run(
                    ["git", "apply", "--whitespace=nowarn"],
                    input=patch,
                    text=True,
                    cwd=str(case_dir),
                    check=True,
                    capture_output=True,
                )
                refactored_src = mod_file.read_text(encoding="utf-8")
                assert "_shared_" in refactored_src
                for test_items, test_cond in [([1, 2, 3], True), ([5], False)]:
                    ns_orig: dict[str, Any] = {}
                    ns_ref: dict[str, Any] = {}
                    exec(compile(combined_src, "<orig>", "exec"), ns_orig)  # pylint: disable=exec-used
                    exec(compile(refactored_src, "<ref>", "exec"), ns_ref)  # pylint: disable=exec-used
                    assert ns_orig["entry"](test_items, test_cond) == ns_ref["entry"](
                        test_items, test_cond
                    )
                    assert ns_orig["entry"](test_items, test_cond) == ns_ref["entry_b"](
                        test_items, test_cond
                    )
            else:
                blocked_count += 1

    assert len(selected_combinations) == min(seed_limit, len(all_combinations))
    assert len(selected_combinations) >= 1
    if len(selected_combinations) >= 8:
        assert patched_count > 0
        assert blocked_count > 0
        assert git_apply_count > 0

    # Negative control: forcing extraction on a blocked late-binding cell + post-unit mutation
    # case produces observable runtime divergence (helper captures scale=3 instead of post-unit scale=10).
    neg_orig_src = (
        "def entry(items):\n"
        "    scale = 3\n"
        "    cbs = []\n"
        "    for x in items:\n"
        "        cbs.append(lambda v: v + scale + x)\n"
        "    scale = 10\n"
        "    return [fn(1) for fn in cbs]\n"
    )
    neg_forced_src = (
        "def _forced_helper(items, cbs, scale):\n"
        "    for x in items:\n"
        "        cbs.append(lambda v: v + scale + x)\n"
        "def entry(items):\n"
        "    scale = 3\n"
        "    cbs = []\n"
        "    _forced_helper(items, cbs, scale)\n"
        "    scale = 10\n"
        "    return [fn(1) for fn in cbs]\n"
    )
    ns_neg_orig: dict[str, Any] = {}
    ns_neg_forced: dict[str, Any] = {}
    exec(compile(neg_orig_src, "<neg_orig>", "exec"), ns_neg_orig)  # pylint: disable=exec-used
    exec(compile(neg_forced_src, "<neg_forced>", "exec"), ns_neg_forced)  # pylint: disable=exec-used
    assert ns_neg_orig["entry"]([1, 2]) != ns_neg_forced["entry"]([1, 2])


def test_duplicate_global_declaration_in_enclosing_function_and_unit_rejected_as_global_escape(
    tmp_path: Path,
) -> None:
    """Verifies duplicate global G declarations in enclosing function and extracted unit are rejected with global_escape."""
    fn_src = (
        "G = 0\n"
        "def entry(x):\n"
        "    global G\n"
        "    if x > 0:\n"
        "        global G\n"
        "        G = x + 1\n"
        "        y = G * 2\n"
        "    else:\n"
        "        y = 0\n"
        "    return G, y\n"
    )
    _run_differential_refactoring_case(
        tmp_path,
        "dup_global_in_unit_and_outer",
        fn_src,
        (4, 9),
        expect_patch=False,
        call_args_list=[(5,)],
        expected_hazard_reason="global_escape",
    )


def test_definite_assignment_engine_parity_across_generated_control_flow() -> None:
    """Verifies parity between _analyze_block_assignment and _extract_block_definite_bindings_and_term across generated control-flow ASTs."""
    import random  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _analyze_block_assignment,
        _block_terminates,
        _extract_block_definite_bindings_and_term,
    )

    rng = random.Random(20261108)
    ctx_choices = [
        "open('f')",
        "contextlib.nullcontext()",
        "contextlib.suppress(ValueError)",
        "UnknownTransaction()",
    ]
    stmt_templates = [
        "a = 1",
        "b = compute(a)",
        "if (w := flag):\n    a = 2\nelse:\n    a = 3",
        "if cond:\n    b = 4\nelse:\n    return 0",
        "try:\n    c = 1\n    d = risky()\nexcept ValueError:\n    d = 2\nelse:\n    e = 3\nfinally:\n    f = 4",
        "try:\n    c = risky()\nexcept Exception:\n    return 1",
        "for item in seq:\n    loop_v = item\nelse:\n    loop_else = 1",
    ]
    for _ in range(80):
        chosen_ctx = rng.choice(ctx_choices)
        chosen_stmts = rng.sample(stmt_templates, k=3)
        with_stmt = (
            f"with {chosen_ctx} as ctx_var:\n"
            f"    w_pre = 10\n"
            f"    w_post = risky_call()\n"
        )
        block_src = "import contextlib\n" + "\n".join([chosen_stmts[0], with_stmt, *chosen_stmts[1:]]) + "\n"
        stmts = ast.parse(block_src).body
        for strict_w in (False, True):
            def_assign, cond_assign = _analyze_block_assignment(
                stmts,
                strict_with_suppression=strict_w,
            )
            ext_bound, ext_term = _extract_block_definite_bindings_and_term(
                stmts,
                None,
                strict_with_suppression=strict_w,
            )
            block_term = _block_terminates(
                stmts,
                strict_with_suppression=strict_w,
            )
            assert ext_term == block_term
            assert def_assign == ext_bound
            assert def_assign.isdisjoint(cond_assign)
