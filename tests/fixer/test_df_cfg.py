"""Tests for Layer-0 statement/CFG lowering and lexical binding extraction in _df_cfg."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from pydoppelgangerhunt.fixer._df_cfg import (
    _collect_scope_direct_bindings,
    _extract_arg_names,
    _extract_import_bound_names,
    _extract_store_names_in_node,
    _extract_target_bound_names,
    _extract_type_alias_or_pattern_name,
    _extract_type_param_names,
    _lower_compound_clauses,
)
from pydoppelgangerhunt.fixer._df_closures import _extract_comprehension_walrus_names
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


def test_df_cfg_remains_layer_zero_dependency_free() -> None:
    """Verifies _df_cfg imports only stdlib modules and _ast_compat at runtime."""
    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "pydoppelgangerhunt"
        / "fixer"
        / "_df_cfg.py"
    )
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    runtime_internal_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("pydoppelgangerhunt"):
                runtime_internal_imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pydoppelgangerhunt"):
                    runtime_internal_imports.append(alias.name)
    assert runtime_internal_imports == ["pydoppelgangerhunt._ast_compat"]


@pytest.mark.parametrize(
    ("source", "expected_locals", "expected_globals", "expected_nonlocals"),
    [
        pytest.param(
            """
def outer():
    import pkg.sub, mod as m
    from other import a, b as c
    from starmod import *
    x = 1
    y: int
    z += 2
    for item in (seq := [1, 2]):
        pass
    with ctx() as (w1, *w2):
        pass
    if (cond := True):
        def inner(p= (hdr := 1)):
            inner_only = 99
        class Cls(base := object):
            cls_only = 88
    try:
        t_val = 1
    except Exception as exc:
        h_val = 2
    global g_var
    g_var = 10
    nonlocal nl_var
    nl_var = 20
""",
            frozenset(
                {
                    "pkg",
                    "m",
                    "a",
                    "c",
                    "x",
                    "y",
                    "z",
                    "item",
                    "seq",
                    "w1",
                    "w2",
                    "cond",
                    "inner",
                    "hdr",
                    "Cls",
                    "base",
                    "t_val",
                    "exc",
                    "h_val",
                }
            ),
            frozenset({"g_var"}),
            frozenset({"nl_var"}),
            id="all_standard_binding_forms_and_walrus_headers",
        ),
        pytest.param(
            """
def outer():
    vals = [w := i for i in range(3)]
    lam = lambda p=(ld := 5): (hidden := p)
""",
            frozenset({"vals", "w", "lam", "ld"}),
            frozenset(),
            frozenset(),
            id="comprehension_walrus_and_lambda_default_walrus_exclude_lambda_body",
        ),
    ],
)
def test_collect_scope_direct_bindings_covers_all_python_binding_forms(
    source: str,
    expected_locals: frozenset[str],
    expected_globals: frozenset[str],
    expected_nonlocals: frozenset[str],
) -> None:
    """Verifies _collect_scope_direct_bindings symmetrically captures direct bindings without leaking nested bodies."""
    mod = ast.parse(source)
    fn_node = mod.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    result = _collect_scope_direct_bindings(
        fn_node.body,
        iter_header_nodes=iter_nested_scope_header_nodes,
        extract_comp_walrus=_extract_comprehension_walrus_names,
    )
    assert result.locals == expected_locals
    assert result.globals == expected_globals
    assert result.nonlocals == expected_nonlocals
    assert "inner_only" not in result.locals
    assert "cls_only" not in result.locals
    assert "hidden" not in result.locals


def test_collect_scope_direct_bindings_honors_option_flags() -> None:
    """Verifies include_imports, include_except_names, include_bare_annassign, and exclude_declared_globals flags."""
    source = """
import os
x: int
y: int = 1
global g
g = 2
try:
    a = 1
except ValueError as err:
    b = 2
"""
    mod = ast.parse(source)
    strict_res = _collect_scope_direct_bindings(
        mod.body,
        include_imports=False,
        include_except_names=False,
        include_bare_annassign=False,
        exclude_declared_globals=False,
    )
    assert strict_res.locals == frozenset({"y", "g", "a", "b"})
    assert "os" not in strict_res.locals
    assert "x" not in strict_res.locals
    assert "err" not in strict_res.locals


def test_extract_arg_import_and_target_primitives() -> None:
    """Verifies arg, import, target-leaf, and store-name extractors handle complex targets cleanly."""
    mod = ast.parse(
        "def f(a, /, b, *args, c=1, **kwargs): pass\n"
        "import x.y.z, p.q as r\n"
        "from pkg import item as alias\n"
        "from starpkg import *\n"
        "(u, [v, *w], obj.attr, arr[idx]) = rhs\n"
    )
    fn_node = mod.body[0]
    imp_node = mod.body[1]
    imp_from = mod.body[2]
    star_imp = mod.body[3]
    assign_node = mod.body[4]
    assert isinstance(fn_node, ast.FunctionDef)
    assert isinstance(imp_node, ast.Import)
    assert isinstance(imp_from, ast.ImportFrom)
    assert isinstance(star_imp, ast.ImportFrom)
    assert isinstance(assign_node, ast.Assign)

    assert _extract_arg_names(fn_node.args) == ("a", "b", "c", "args", "kwargs")
    assert _extract_import_bound_names(imp_node) == ("x", "r")
    assert _extract_import_bound_names(imp_from) == ("alias",)
    assert not _extract_import_bound_names(star_imp)
    assert _extract_target_bound_names(assign_node.targets[0]) == ("u", "v", "w")
    assert _extract_store_names_in_node(assign_node.targets[0]) == {"u", "v", "w"}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 634 match syntax requires Python 3.10+")
def test_collect_scope_direct_bindings_handles_match_patterns() -> None:
    """Verifies match subject walrus, guard walrus, and pattern bindings are collected."""
    source = """
match (subj := data):
    case [head, *tail] if (guard_v := head > 0):
        body_v = tail
    case {"k": val, **rest}:
        pass
    case _:
        pass
"""
    mod = ast.parse(source)
    clauses = _lower_compound_clauses(mod.body[0])
    assert len(clauses) == 3
    assert set(clauses[0].bound_names) == {"head", "tail"}
    assert set(clauses[1].bound_names) == {"val", "rest"}
    assert clauses[2].is_irrefutable is True

    direct = _collect_scope_direct_bindings(mod.body)
    assert direct.locals == frozenset({"subj", "head", "tail", "guard_v", "body_v", "val", "rest"})


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 syntax requires Python 3.12+")
def test_extract_type_alias_and_type_params() -> None:
    """Verifies PEP 695 TypeAlias binding and generic type_params extraction."""
    source = """
type Alias[T] = list[T]
def generic_fn[U, *Vs, **P](x: U) -> U:
    return x
"""
    mod = ast.parse(source)
    assert _extract_type_alias_or_pattern_name(mod.body[0]) == "Alias"
    assert _extract_type_param_names(mod.body[0]) == ("T",)
    assert _extract_type_param_names(mod.body[1]) == ("U", "Vs", "P")
    direct = _collect_scope_direct_bindings(mod.body)
    assert direct.locals == frozenset({"Alias", "generic_fn"})
