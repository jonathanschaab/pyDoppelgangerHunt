"""Tests for scope schema validation, source slicing/origin invariants, and fail-closed source contracts."""

from __future__ import annotations

import ast
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    analyze_unit_variable_scope,
    clear_dataflow_cache,
    generate_refactoring_patch,
    inspect_single_unit_scope,
    resolve_clone_pair_outputs,
    synthesize_shared_helper_code,
)


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

    dup_line_src = (
        "def f(xs):\n" "    a = 1; a = 1\n" "    return [lambda: a for _ in xs]\n"
    )
    dup_line_tree = ast.parse(dup_line_src)
    cand_assign = ast.parse("a = 1\n").body
    assert _resolve_unit_col_bounds(cand_assign, dup_line_tree, 2, 2, None, None) == (
        None,
        None,
    )
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
        "def f(xs):\n"
        "    res = [x + 1 for x in xs] + [x + 1 for x in xs]\n"
        "    return res\n"
    )
    subexpr_tree = ast.parse(subexpr_line_src)
    cand_comp_expr = ast.parse("[x + 1 for x in xs]\n").body
    assert _resolve_unit_col_bounds(cand_comp_expr, subexpr_tree, 2, 2, None, None) == (
        None,
        None,
    )
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
        "def f(xs):\n"
        "    a = 1\n"
        "    b = a + 2\n"
        "    return [lambda: b for _ in xs]\n"
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

    corrupt_col_tree = ast.parse(
        "def f(xs):\n" "    a = 1\n" "    b = a + 2\n" "    return b\n"
    )
    corrupt_col_fn = corrupt_col_tree.body[0]
    assert isinstance(corrupt_col_fn, ast.FunctionDef)
    corrupt_col_fn.body[1].end_col_offset = None
    assert (
        _is_node_inside_span(corrupt_col_fn.body[1], _SpanBounds(3, 3, 4, 13)) is False
    )
    assert _has_unresolved_span_bounds(corrupt_col_fn, 3, 3, 4, 13) is True


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
    assert (
        "unavailable_source_for_closure_mode"
        in unmarked_no_source["closure_hazard_reasons"]
    )
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
    assert sliced_exact_first == []
    sliced_beyond_first = _slice_unit_token_lines(
        {
            "kind": "complex_expr",
            "start_col": first_bytes_len + 10,
            "end_col": last_bytes_len,
        },
        ml_lines,
    )
    assert sliced_beyond_first == []
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
            },
            presliced_ml,
            is_pre_sliced=True,
        )
        == []
    )
    assert (
        _slice_unit_token_lines(
            {
                "kind": "complex_expr",
                "start_col": 8,
                "end_col": last_bytes_len,
            },
            presliced_ml,
            explicit_col_sliced=True,
        )
        == presliced_ml
    )


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
    comb_lenient = analyze_unit_variable_scope(
        u_len, dict(u_len), skip_pre_unit_closures=True
    )
    assert comb_lenient["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" not in comb_lenient["control_flow_hazards"]


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

    mod_one_line = ast.parse(
        "def outer(items): sum = custom_sum; return sum(x for x in items)\n"
    )
    outer_one = mod_one_line.body[0]
    assert isinstance(outer_one, ast.FunctionDef)
    ret_one = outer_one.body[1]
    enc_one_line = _resolve_enclosing_scopes_by_coords(ret_one, mod_one_line)
    assert outer_one in enc_one_line
    locals_one_line = _extract_scope_local_names(ret_one, module_tree=mod_one_line)
    assert "sum" in locals_one_line
    closures_one_line = _collect_scope_closures_uncached(
        outer_one, module_tree=mod_one_line
    )
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
    closures_async = _collect_scope_closures_uncached(
        outer_async, module_tree=mod_async_one_line
    )
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
    detached_cls_unit = ast.parse(
        "r1 = sum(x for x in items)\nr2 = max(y for y in items)\n"
    )
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
    lines_with_state, origin = extract_unit_source_lines_with_state(
        unit_empty_src
    )
    assert lines_with_state == ["def example():\n", "    return 42\n"]
    assert origin.is_pre_sliced is False
    assert origin.is_placeholder is False

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

    # 4. _has_invalid_lexical_scope_declarations suppresses SyntaxWarning/DeprecationWarning from invalid escape sequences
    warn_src = 'def f():\n    s = "\\d+"\n    return 1 is 1\n'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        warn_mod = ast.parse(warn_src)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        is_invalid = _has_invalid_lexical_scope_declarations(
            [warn_mod.body[0]],
            mod_tree=warn_mod,
            source_lines=warn_src.splitlines(),
        )
    assert is_invalid is False
    assert not [
        w
        for w in caught
        if issubclass(w.category, (SyntaxWarning, DeprecationWarning))
    ]


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


def test_placeholder_origin_vs_real_comment_and_source_code_only_unit() -> None:
    """Verifies real comments starting with '# Source for ' are not treated as placeholders and source_code + tree units are not falsely flagged stale."""
    mod_with_comment = (
        "def f(a, b):\n"
        "    # Source for custom step\n"
        "    res = a + b\n"
        "    return res\n"
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


def test_clause_header_source_code_normalization_and_multiline_string_comments() -> (
    None
):
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
        (pos.line, pos.col) == (2, 4) and "x" in reads for pos, reads in closures
    )
    # Subsequent call to _node_earliest_start_pos without explicit source_lines resolves line 2 via parent_map + cached tree_source_lines
    assert _node_earliest_start_pos(inner_fn) == (2, 4)

    cell_info = _analyze_enclosing_fn_cells(
        outer_fn,
        _SpanBounds(7, 7),
        mod_tree=mod_tree,
    )
    assert any(
        "x" in reads for name, reads, _ in cell_info.sibling_closures if name == "inner"
    )


def test_harvest_file_units_passes_strict_unit_schema_and_scope_inspection(
    tmp_path: Any,
) -> None:
    """Verifies every unit dict produced by harvest_file_units passes validate_unit_schema(strict=True) and inspect_single_unit_scope without stale_source_code."""
    import json
    from pydoppelgangerhunt.fixer.source import validate_unit_schema
    from pydoppelgangerhunt.parser import harvest_file_units, harvest_notebook_units

    sample_src = (
        "def identity_deco(fn):\n"
        "    return fn\n\n"
        "def tab_helper(a, b):\n"
        "\ttotal = (a + b) * (a - b)\n"
        "\tscaled = total + a * 2\n"
        "\treturn scaled\n\n"
        "class Widget:\n"
        "    TABLE = [\n"
        "        (1, 'a', True),\n"
        "        (2, 'b', False),\n"
        "        (3, 'c', True),\n"
        "        (4, 'd', False),\n"
        "    ]\n"
        "    @identity_deco\n"
        "    def compute(self, items, scale):\n"
        "        banner = '''header line\n"
        "# hash inside multiline string\n"
        "footer line'''  # trailing comment\n"
        "\x0c        def scale_it(v):\n"
        "            return v * scale + 1\n"
        "        vals = [scale_it(x) for x in items if x > 0]\n"
        "        if len(vals) > 2:\n"
        "            total = sum(vals) + (scale * 2 + 3) * (scale - 1)\n"
        "            avg = total / len(vals) + len(banner)\n"
        "            return avg\n"
        "        else:\n"
        "            fallback = (scale + 10) * (scale + 20) - 5\n"
        "            return fallback\n"
    )
    sample_file = tmp_path / "harvest_all_kinds.py"
    sample_file.write_text(sample_src, encoding="utf-8")

    crlf_file = tmp_path / "harvest_crlf.py"
    crlf_file.write_bytes(sample_src.replace("\n", "\r\n").encode("utf-8"))

    for target_file in (sample_file, crlf_file):
        units = harvest_file_units(
            str(target_file),
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

    nb_file = tmp_path / "sample.ipynb"
    nb_payload = {
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "def nb_compute(items, scale):\n",
                    "    vals = [x * scale for x in items if x > 0]\n",
                    "    total = sum(vals) + scale\n",
                    "    return total\n",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_file.write_text(json.dumps(nb_payload), encoding="utf-8")
    nb_units = harvest_notebook_units(
        str(nb_file),
        str(tmp_path),
        min_lines=2,
        min_tokens=4,
    )
    assert nb_units
    for nu in nb_units:
        assert validate_unit_schema(nu, strict=True) == set()
        sc_nb = inspect_single_unit_scope(nu, repo_root=str(tmp_path))
        assert "stale_source_code" not in sc_nb["control_flow_hazards"]


def test_source_lines_origin_enum_properties_and_hash_contract() -> None:
    """Verifies SourceLinesOrigin is an Enum with transitive equality, consistent hashing, and boolean properties."""
    import enum
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    assert issubclass(SourceLinesOrigin, enum.Enum)
    assert "__bool__" in SourceLinesOrigin.__dict__
    for member in SourceLinesOrigin:
        with pytest.raises(TypeError, match="no implicit truth value"):
            bool(member)
    assert SourceLinesOrigin.PRE_SLICED.is_pre_sliced is True
    assert SourceLinesOrigin.PRE_SLICED.is_placeholder is False
    assert SourceLinesOrigin.PRE_SLICED.has_real_source is True

    assert SourceLinesOrigin.PLACEHOLDER.is_pre_sliced is True
    assert SourceLinesOrigin.PLACEHOLDER.is_placeholder is True
    assert SourceLinesOrigin.PLACEHOLDER.has_real_source is False

    assert SourceLinesOrigin.FULL_BUFFER.is_pre_sliced is False
    assert SourceLinesOrigin.FULL_BUFFER.has_real_source is True

    assert SourceLinesOrigin.DISK.is_pre_sliced is False
    assert SourceLinesOrigin.DISK.has_real_source is True

    assert SourceLinesOrigin.PRE_SLICED != "PRE_SLICED"
    assert SourceLinesOrigin.PRE_SLICED != True  # noqa: E712
    assert len(set(SourceLinesOrigin)) == 4
    mixed_set = {
        SourceLinesOrigin.PRE_SLICED,
        SourceLinesOrigin.PLACEHOLDER,
        True,
        "PRE_SLICED",
        "PLACEHOLDER",
    }
    assert len(mixed_set) == 5


def test_ensure_unit_scope_closure_mode_preserves_terminal_pathological_or_stale_scope() -> (
    None
):
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
    "stale_code",
    [
        "    b = a + 2\n",
        "    a = x + 1\n    b = a + 2\n",
        "    c = b + 3\n    b = a + 2\n    a = x + 1\n",
        "    b = a + 2\n    b = a + 2\n    b = a + 2\n",
    ],
)
def test_stale_source_code_partial_slice_and_tree_mismatch_rejected(
    stale_code: str,
) -> None:
    """Verifies partial, truncated, reordered, or duplicated unit['source_code'] is rejected as stale_source_code."""
    src = (
        "def compute(x):\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    c = b + 3\n"
        "    return c\n"
    )
    mod_tree = ast.parse(src)

    sc_with_text = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": src,
            "source_code": stale_code,
        },
        tree=mod_tree,
    )
    assert "stale_source_code" in sc_with_text["control_flow_hazards"]
    assert sc_with_text["is_control_flow_safe"] is False

    sc_tree_only = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_code": stale_code,
        },
        tree=mod_tree,
    )
    assert "stale_source_code" in sc_tree_only["control_flow_hazards"]
    assert sc_tree_only["is_control_flow_safe"] is False

    sc_single_line_partial = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "kind": "basic_block",
            "source_text": src,
            "source_code": "x + 1",
        },
        tree=mod_tree,
    )
    assert "stale_source_code" in sc_single_line_partial["control_flow_hazards"]
    assert sc_single_line_partial["is_control_flow_safe"] is False

    sc_matching = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 4,
            "kind": "compound_block",
            "source_text": src,
            "source_code": "    a = x + 1\n    b = a + 2\n    c = b + 3\n",
        },
        tree=mod_tree,
    )
    assert "stale_source_code" not in sc_matching["control_flow_hazards"]
    assert sc_matching["is_control_flow_safe"] is True
    assert sc_matching["inputs"] == ["x"]
    assert sc_matching["outputs"] == ["a", "b", "c"]


def test_write_only_conditional_store_strictness_monotonicity_with_returned_closure() -> (
    None
):
    """Verifies write-only conditional store captured by a post-unit referenced closure preserves strictness monotonicity."""
    src = (
        "def work(items, flag):\n"
        "    tmp = 0\n"
        "    cb = lambda: tmp\n"
        "    acc = 1\n"
        "    if flag:\n"
        "        tmp = 2\n"
        "    acc = acc + len(items)\n"
        "    return acc, cb\n"
    )
    mod_tree = ast.parse(src)
    unit_base = {
        "start": 5,
        "end": 7,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    if flag:\n        tmp = 2\n    acc = acc + len(items)\n",
    }
    scopes = {
        mode: inspect_single_unit_scope(
            {**unit_base, "closure_strictness": mode},
            tree=mod_tree,
        )
        for mode in ("lenient", "strict", "pedantic")
    }
    for mode, sc in scopes.items():
        assert "tmp" in sc["inputs"], mode
        assert "severed_closure_cell" not in sc["control_flow_hazards"], mode
        assert sc["is_control_flow_safe"] is True, mode


def test_slice_unit_token_lines_clips_multiline_presliced_line_excerpt_without_duplicating_suffix(
    tmp_path: Any,
) -> None:
    """Verifies multiline line-sliced excerpts clip boundary columns so neighboring '+ tail' suffixes are not duplicated."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    excerpt = [
        "    result = [\n",
        "        x * 2\n",
        "        for x in items\n",
        "    ] + tail\n",
    ]
    unit = {
        "name": "f1:comp",
        "start": 2,
        "end": 5,
        "start_col": 13,
        "end_col": 5,
        "kind": "comprehension",
        "source_lines": excerpt,
    }
    clipped = _slice_unit_token_lines(
        unit,
        [ln.rstrip("\r\n") for ln in excerpt],
        is_pre_sliced=SourceLinesOrigin.PRE_SLICED,
    )
    assert clipped == ["[", "        x * 2", "        for x in items", "    ]"]

    src = (
        "def f1(items, tail):\n"
        "    result = [\n"
        "        x * 2\n"
        "        for x in items\n"
        "    ] + tail\n"
        "    return result\n\n"
        "def f2(items, tail):\n"
        "    result = [\n"
        "        x * 2\n"
        "        for x in items\n"
        "    ] + tail\n"
        "    return result\n"
    )
    mod_file = tmp_path / "comp_tail.py"
    mod_file.write_text(src, encoding="utf-8")
    u1 = {
        "name": "f1:comp",
        "file": str(mod_file),
        "start": 2,
        "end": 5,
        "start_col": 13,
        "end_col": 5,
        "kind": "comprehension",
        "source_lines": list(excerpt),
    }
    u2 = {
        "name": "f2:comp",
        "file": str(mod_file),
        "start": 9,
        "end": 12,
        "start_col": 13,
        "end_col": 5,
        "kind": "comprehension",
        "source_lines": list(excerpt),
    }
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_" in helper
    assert "tail" not in helper

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    refactored = mod_file.read_text(encoding="utf-8")
    ns: dict[str, Any] = {}
    exec(compile(refactored, str(mod_file), "exec"), ns)  # nosec B102
    assert ns["f1"]([1, 2, 3], [99]) == [2, 4, 6, 99]
    assert ns["f2"]([4, 5], [0]) == [8, 10, 0]


def test_out_of_range_unit_columns_fail_closed_in_scope_and_slicing() -> None:
    """Verifies out-of-range start_col/end_col fail closed in _slice_unit_token_lines and _inspect_unit_scope."""
    from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

    src = "def f(items):\n    res = [x * 2 for x in items]\n    return res\n"
    line = "    res = [x * 2 for x in items]\n"
    assert (
        _slice_unit_token_lines(
            {"kind": "comprehension", "start_col": 999, "end_col": 1010},
            [line],
        )
        == []
    )
    assert (
        _slice_unit_token_lines(
            {"kind": "comprehension", "start_col": 10, "end_col": 999},
            [line],
        )
        == []
    )
    sc_oob = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": 999,
            "end_col": 1010,
            "kind": "comprehension",
            "source_text": src,
        },
        tree=ast.parse(src),
    )
    assert "invalid_unit_columns" in sc_oob["control_flow_hazards"]
    assert "stale_source_code" not in sc_oob["control_flow_hazards"]
    assert sc_oob["is_control_flow_safe"] is False


def test_col_sliced_flag_separation_and_unrecognized_strictness_rejection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies source_lines_is_sliced vs source_lines_is_col_sliced and rejection of non-canonical strictness modes."""
    import logging
    from pydoppelgangerhunt.config import (
        clear_closure_warning_cache,
        resolve_closure_strictness_mode,
    )
    from pydoppelgangerhunt.fixer.source import (
        _slice_unit_token_lines,
        validate_unit_schema,
    )

    line_excerpt = ["    result = [v + offset for v in items] + tail_extra\n"]
    unit_line_sliced = {
        "start": 10,
        "end": 10,
        "start_col": 13,
        "end_col": 40,
        "kind": "comprehension",
        "source_lines": line_excerpt,
        "source_lines_is_sliced": True,
    }
    assert validate_unit_schema(unit_line_sliced, strict=True) == set()
    trimmed = _slice_unit_token_lines(unit_line_sliced, line_excerpt)
    assert trimmed == ["[v + offset for v in items]"]
    sc_line_sliced = inspect_single_unit_scope(unit_line_sliced)
    assert "tail_extra" not in sc_line_sliced["inputs"]
    assert set(sc_line_sliced["inputs"]) == {"items", "offset"}

    already_col_sliced = ["[v + offset for v in items]"]
    unit_col_sliced = {
        **unit_line_sliced,
        "source_lines": already_col_sliced,
        "source_lines_is_col_sliced": True,
    }
    assert validate_unit_schema(unit_col_sliced, strict=True) == set()
    untouched = _slice_unit_token_lines(unit_col_sliced, already_col_sliced)
    assert untouched == already_col_sliced
    sc_col_sliced = inspect_single_unit_scope(unit_col_sliced)
    assert set(sc_col_sliced["inputs"]) == {"items", "offset"}

    for deprecated_mode, expected_resolved in (
        ("fail_closed", ("pedantic", False)),
        ("fast", ("lenient", True)),
        ("skip", ("lenient", True)),
    ):
        clear_closure_warning_cache()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            res_mode, res_skip = resolve_closure_strictness_mode(
                closure_strictness=deprecated_mode
            )
        assert (res_mode, res_skip) == expected_resolved
        assert any(
            "Deprecated closure_strictness" in rec.message
            for rec in caplog.records
        )
        with pytest.raises(ValueError, match="closure_strictness"):
            validate_unit_schema(
                {"start": 1, "end": 2, "closure_strictness": deprecated_mode},
                strict=True,
            )

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        bogus_mode, bogus_skip = resolve_closure_strictness_mode(
            closure_strictness="bogus_mode"
        )
    assert (bogus_mode, bogus_skip) == ("strict", False)
    assert any(
        "Unrecognized closure_strictness" in rec.message
        for rec in caplog.records
    )


@pytest.mark.parametrize(
    ("bad_unit", "err_pattern"),
    [
        ({"start": "not_an_int", "end": 2}, "Invalid coordinate value"),
        ({"start": 0, "end": 2}, "Invalid coordinate value"),
        ({"start": 5, "end": 3}, "Inverted line coordinates"),
        ({"start": 1, "end": 2, "start_col": -1}, "Invalid coordinate value"),
        ({"start": 1, "end": 2, "skip_pre_unit_closures": "yes"}, "Expected bool"),
        ({"start": 1, "end": 2, "strict_with_suppression": 1}, "Expected bool"),
        (
            {"start": 1, "end": 2, "closure_strictness": "bogus_mode"},
            "Invalid closure_strictness",
        ),
        (
            {"start": 1, "end": 2, "non_suppressing_ctx_managers": "SafeTx"},
            "Expected sequence or set of str",
        ),
        (
            {"start": 1, "end": 2, "non_suppressing_ctx_managers": [123]},
            "Expected sequence or set of str",
        ),
        ({"start": 1, "end": 2, "token_count": -1}, "Expected non-negative int"),
        ({"start": 1, "end": 2, "token_count": True}, "Expected non-negative int"),
        (
            {"start": 1, "end": 2, "tokens": -5},
            "Expected non-negative int or ordered sequence of str",
        ),
        (
            {"start": 1, "end": 2, "tokens": {"tok_a", "tok_b"}},
            "Expected non-negative int or ordered sequence of str",
        ),
        ({"start": 1, "end": 2, "inputs": {"first", "second"}}, "ordered sequence of str"),
        ({"start": 1, "end": 2, "outputs": frozenset({"res_a"})}, "ordered sequence of str"),
        (
            {"start": 1, "end": 2, "precomputed_outputs": {"res_a", "res_b"}},
            "ordered sequence of str",
        ),
        ({"start": 1, "end": 2, "norm_tokens": {"NAME"}}, "ordered sequence of str"),
        ({"start": 1, "end": 2, "calls": {"len"}}, "ordered sequence of str"),
        ({"start": 1, "end": 2, "source_lines": {"x = 1\n"}}, "ordered sequence of str"),
        ({"start": 1, "end": 2, "outputs": {"res": 1}}, "ordered sequence of str"),
        ({"start": 1, "end": 2, "docstring_end": 0}, "Expected positive int"),
        ({"start": 1, "end": 2, "enclosing_class_start": 0}, "Expected positive int"),
        ({"start": 1, "end": 2, "docstring_end": True}, "Expected positive int"),
        ({"start": 1, "end": 2, "lineno": 3}, "Conflicting coordinate aliases"),
        ({"start": 1, "end": 2, "end_lineno": 4}, "Conflicting coordinate aliases"),
        (
            {"start": 1, "end": 2, "start_col": 0, "col_offset": 4},
            "Conflicting coordinate aliases",
        ),
        (
            {"start": 1, "end": 2, "start_col": 0, "start_col_offset": 2},
            "Conflicting coordinate aliases",
        ),
        (
            {"start": 1, "end": 2, "end_col": 5, "end_col_offset": 6},
            "Conflicting coordinate aliases",
        ),
        ({"start": 1, "end": 2, "scope": "not_a_dict"}, "Expected mapping for 'scope'"),
        (
            {"start": 1, "end": 2, "ast_node": "not_an_ast"},
            "Expected ast.AST for 'ast_node'",
        ),
    ],
)
def test_validate_unit_schema_validates_field_value_types_and_bounds(
    bad_unit: dict[str, Any],
    err_pattern: str,
) -> None:
    """Verifies validate_unit_schema validates field types, coordinate bounds, and enum values."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    with pytest.raises(ValueError, match=err_pattern):
        validate_unit_schema(bad_unit, strict=True)


def test_snippet_only_subroutine_fails_closed_by_default_and_opts_in_explicitly() -> None:
    """Verifies subroutine units lacking enclosing module context fail closed unless allow_snippet_only=True."""
    from pydoppelgangerhunt.fixer import (
        ensure_unit_scope_closure_mode,
        validate_unit_schema,
    )

    snippet_unit = {
        "start": 5,
        "end": 6,
        "kind": "compound_block",
        "source_code": "step = x + 1\nout = step * 2\n",
    }
    sc_default = inspect_single_unit_scope(snippet_unit)
    assert sc_default["is_control_flow_safe"] is False
    assert "severed_closure_cell" in sc_default["control_flow_hazards"]
    assert (
        "snippet_only_missing_enclosing_context"
        in sc_default["closure_hazard_reasons"]
    )

    opt_in_unit = {**snippet_unit, "allow_snippet_only": True}
    assert validate_unit_schema(opt_in_unit, strict=True) == set()
    sc_opt_in = inspect_single_unit_scope(opt_in_unit)
    assert sc_opt_in["is_control_flow_safe"] is True
    assert sc_opt_in["inputs"] == ["x"]
    assert sc_opt_in["outputs"] == ["step", "out"]

    stale_lenient_scope = {
        **sc_opt_in,
        "skip_pre_unit_closures": True,
        "strict_with_suppression": False,
    }
    recomputed_no_backing = ensure_unit_scope_closure_mode(
        snippet_unit,
        stale_lenient_scope,
        False,
    )
    assert recomputed_no_backing["is_control_flow_safe"] is False
    assert (
        "unavailable_source_for_closure_mode"
        in recomputed_no_backing["closure_hazard_reasons"]
    )


@pytest.mark.parametrize(
    "unit_kind",
    ["function", "method", "async_function", "class", "data_table"],
)
def test_placeholder_source_non_subroutine_units_fail_closed(unit_kind: str) -> None:
    """Verifies missing-source placeholder units of non-subroutine kinds fail closed rather than reporting safe."""
    u1 = {
        "file": "missing_non_subroutine_source_1.py",
        "start": 1,
        "end": 6,
        "name": "missing_alpha",
        "kind": unit_kind,
    }
    u2 = {
        "file": "missing_non_subroutine_source_2.py",
        "start": 1,
        "end": 6,
        "name": "missing_beta",
        "kind": unit_kind,
    }
    sc = inspect_single_unit_scope(u1)
    sc2 = inspect_single_unit_scope(u2)
    assert sc["is_control_flow_safe"] is False
    assert "unavailable_source_placeholder" in sc["control_flow_hazards"]
    assert "unavailable_source_placeholder" in sc["closure_hazard_reasons"]

    res = resolve_clone_pair_outputs(u1, u2, sc, sc2, return_details=True)
    assert res.outputs is None
    assert res.skip_reason == "unavailable source placeholder"

    # Also blocked when require_pairing=False (replace_clones=False) and in synthesize_shared_helper_code
    res_preview = resolve_clone_pair_outputs(
        u1, u2, sc, sc2, require_pairing=False, return_details=True
    )
    assert res_preview.outputs is None
    assert res_preview.skip_reason == "unavailable source placeholder"
    assert synthesize_shared_helper_code(u1, u2) == ""


def test_placeholder_non_subroutine_pair_blocked_in_patch_preview(tmp_path: Path) -> None:
    """Verifies non-subroutine pairs with placeholder source are blocked in generate_refactoring_patch(..., replace_clones=False)."""
    real_file = tmp_path / "tiny.py"
    real_file.write_text("x = 1\n", encoding="utf-8")
    # Lines 50..60 exceed file length so extract_unit_source_lines_with_state yields placeholder lines on readable file
    u1 = {"file": str(real_file), "start": 50, "end": 60, "name": "fn_a", "kind": "function"}
    u2 = {"file": str(real_file), "start": 70, "end": 80, "name": "fn_b", "kind": "function"}
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert patch == ""


def test_invalid_unit_columns_propagates_across_pairing_synthesis_patch_and_suggest(
    tmp_path: Path,
) -> None:
    """Verifies out-of-bounds unit columns surface as 'invalid_unit_columns' across all refactoring stages."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    src = (
        "def f1(items):\n"
        "    return [x + 1 for x in items]\n\n"
        "def f2(items):\n"
        "    return [x + 1 for x in items]\n"
    )
    mod_file = tmp_path / "cols.py"
    mod_file.write_text(src, encoding="utf-8")
    u1 = {
        "name": "f1:comp",
        "file": str(mod_file),
        "start": 2,
        "end": 2,
        "start_col": 500,
        "end_col": 600,
        "kind": "comprehension",
    }
    u2 = {
        "name": "f2:comp",
        "file": str(mod_file),
        "start": 5,
        "end": 5,
        "start_col": 500,
        "end_col": 600,
        "kind": "comprehension",
    }
    sc1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    sc2 = inspect_single_unit_scope(u2, repo_root=str(tmp_path))
    assert "invalid_unit_columns" in sc1["control_flow_hazards"]

    res = resolve_clone_pair_outputs(
        u1, u2, sc1, sc2, repo_root=str(tmp_path), return_details=True
    )
    assert res.outputs is None
    assert res.skip_reason == "invalid unit columns"
    assert synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path)) == ""
    assert (
        generate_refactoring_patch(
            [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )
    sug = synthesize_refactoring_suggestion(u1, u2, repo_root=str(tmp_path))
    assert "rejected as unsafe" in sug
    assert "invalid unit columns" in sug


def test_validate_unit_schema_accepts_sets_only_for_membership_fields() -> None:
    """Verifies sets are accepted only for order-insensitive fields while ordered fields require sequences."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    class _CustomStringSet(AbstractSet[str]):
        def __init__(self, items: tuple[str, ...]) -> None:
            self._items = items

        def __contains__(self, x: object) -> bool:
            return x in self._items

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._items)

        def __len__(self) -> int:
            return len(self._items)

    unit_with_sets = {
        "start": 1,
        "end": 3,
        "inputs": ("x", "y"),
        "outputs": ["z"],
        "precomputed_outputs": ("z",),
        "shingles": frozenset({("tok_a", "tok_b", "tok_c")}),
        "non_suppressing_ctx_managers": _CustomStringSet(("SafeTx", "db.atomic")),
    }
    assert validate_unit_schema(unit_with_sets, strict=True) == set()

    with pytest.raises(ValueError, match="ordered sequence of str for unit field 'outputs'"):
        validate_unit_schema(
            {**unit_with_sets, "outputs": _CustomStringSet(("z",))}, strict=True
        )
    with pytest.raises(ValueError, match="non_suppressing_ctx_managers"):
        validate_unit_schema(
            {"start": 1, "end": 2, "non_suppressing_ctx_managers": frozenset({123})},  # type: ignore[arg-type]
            strict=True,
        )


@pytest.mark.parametrize(
    "good_unit",
    [
        pytest.param({"start": 1, "end": 2, "docstring_end": 1}, id="docstring_end_one"),
        pytest.param({"start": 1, "end": 2, "enclosing_class_start": 1}, id="class_start_one"),
        pytest.param({"start": 1, "end": 2, "token_count": 0}, id="zero_token_count"),
        pytest.param({"start": 3, "end": 4, "lineno": 3, "end_lineno": 4}, id="agreeing_lines"),
        pytest.param(
            {"start": 1, "end": 1, "start_col": 2, "col_offset": "2", "end_col": 9, "end_col_offset": 9},
            id="agreeing_columns_across_int_and_str",
        ),
        pytest.param({"start": 1, "end": 2, "tokens": ["NAME", "OP"]}, id="ordered_token_list"),
    ],
)
def test_validate_unit_schema_accepts_positive_lines_and_agreeing_aliases(
    good_unit: dict[str, Any],
) -> None:
    """Verifies 1-based metadata, zero-valued counters, and agreeing coordinate aliases pass strict validation."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    assert validate_unit_schema(good_unit, strict=True) == set()


def test_tree_only_placeholder_unit_fails_closed_and_host_caller_placeholder_contract(
    tmp_path: Path,
) -> None:
    """Verifies tree-only units without source emit unavailable_source_placeholder and placeholder u1 never leaks into helpers."""
    mod_tree = ast.parse("def real_fn(x: int) -> int:\n    return x * 4\n")
    sc_tree_only = inspect_single_unit_scope(
        {"name": "ghost_fn", "start": 1, "end": 2, "kind": "function"},
        tree=mod_tree,
    )
    assert "unavailable_source_placeholder" in sc_tree_only["control_flow_hazards"]
    assert sc_tree_only["is_control_flow_safe"] is False

    pkg = tmp_path / "ph_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    real_file = pkg / "real_mod.py"
    real_file.write_text(
        "def compute_item(x: int) -> int:\n    y = x + 2\n    return y * 3\n",
        encoding="utf-8",
    )
    u_real_fn = {"name": "compute_item", "file": "ph_pkg/real_mod.py", "start": 1, "end": 3, "kind": "function"}
    u_missing_fn = {"name": "compute_item", "file": "ph_pkg/missing_mod.py", "start": 1, "end": 3, "kind": "function"}
    u_real_sub = {"name": "compute_item:blk", "file": "ph_pkg/real_mod.py", "start": 2, "end": 2, "kind": "compound_block"}
    u_missing_sub = {"name": "compute_item:blk", "file": "ph_pkg/missing_mod.py", "start": 2, "end": 2, "kind": "compound_block"}

    # Placeholder on u1 is unconditionally rejected in both preview and rewrite modes
    sc_missing = inspect_single_unit_scope(u_missing_fn, repo_root=str(tmp_path))
    sc_real = inspect_single_unit_scope(u_real_fn, repo_root=str(tmp_path))
    assert (
        resolve_clone_pair_outputs(
            u_missing_fn, u_real_fn, sc_missing, sc_real, repo_root=str(tmp_path), require_pairing=False
        )
        is None
    )
    assert synthesize_shared_helper_code(u_missing_fn, u_real_fn, repo_root=str(tmp_path)) == ""
    assert (
        generate_refactoring_patch(
            [(1.0, u_missing_fn, u_real_fn)], repo_root=str(tmp_path), replace_clones=False
        )
        == ""
    )

    # Placeholder on u2 is rejected for subroutines and for replace_clones=True
    assert synthesize_shared_helper_code(u_real_sub, u_missing_sub, repo_root=str(tmp_path)) == ""
    assert (
        generate_refactoring_patch(
            [(1.0, u_real_fn, u_missing_fn)], repo_root=str(tmp_path), replace_clones=True
        )
        == ""
    )

    # Whole-function preview mode synthesizes helper strictly from u1's real AST without placeholder comments
    helper_preview = synthesize_shared_helper_code(u_real_fn, u_missing_fn, repo_root=str(tmp_path))
    assert "return y * 3" in helper_preview
    assert "# Source for" not in helper_preview


def test_discarded_precomputed_scope_emits_once_per_run_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies legacy/mock precomputed scopes lacking scope_schema_version==2 emit a once-per-run WARNING."""
    import logging
    from pydoppelgangerhunt.config import clear_closure_warning_cache

    src = "def f(x):\n    y = x + 1\n    return y\n"
    u1 = {"name": "f:blk", "start": 2, "end": 2, "kind": "basic_block", "source_text": src}
    u2 = {"name": "f:blk2", "start": 2, "end": 2, "kind": "basic_block", "source_text": src}
    legacy_scope = {"inputs": ["bogus_override"], "outputs": [], "is_control_flow_safe": True}

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        combined_first = analyze_unit_variable_scope(u1, u2, scope1=legacy_scope, scope2=legacy_scope)
        combined_second = analyze_unit_variable_scope(u1, u2, scope1=legacy_scope, scope2=legacy_scope)
    assert combined_first["inputs"] == ["x"]
    assert combined_second["inputs"] == ["x"]
    discard_warnings = [
        r for r in caplog.records if "Discarding caller-supplied precomputed scope" in r.message
    ]
    assert len(discard_warnings) == 1


def test_validate_unit_schema_strict_accepts_post_fixer_mutated_unit_dicts(
    tmp_path: Path,
) -> None:
    """Verifies strict UnitSchema validation succeeds on unit dicts after fixer/reporter mutations."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema
    from pydoppelgangerhunt.parser import harvest_file_units
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    src = (
        "class Processor:\n"
        "    def run_a(self, items: list[int], scale: int) -> list[int]:\n"
        "        filtered = [v * scale for v in items if v > 0]\n"
        "        total = sum(filtered)\n"
        "        return [total]\n\n"
        "    def run_b(self, items: list[int], scale: int) -> list[int]:\n"
        "        filtered = [v * scale for v in items if v > 0]\n"
        "        total = sum(filtered)\n"
        "        return [total]\n"
    )
    mod_file = tmp_path / "proc.py"
    mod_file.write_text(src, encoding="utf-8")
    harvested = harvest_file_units(
        str(mod_file),
        str(tmp_path),
        min_lines=1,
        min_tokens=3,
        comprehensions=True,
        complex_expressions=True,
        harvest_closures=True,
    )
    assert len(harvested) >= 4
    methods = [
        u
        for u in harvested
        if u.get("kind") == "function" and u.get("enclosing_class") == "Processor"
    ]
    assert len(methods) == 2
    u1, u2 = methods[0], methods[1]

    inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    synthesize_refactoring_suggestion(u1, u2, repo_root=str(tmp_path))
    generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)

    for u_item in harvested:
        inspect_single_unit_scope(u_item, repo_root=str(tmp_path))
        assert validate_unit_schema(u_item, strict=True) == set()


def test_source_lines_origin_enum_properties_and_truthiness_guard() -> None:
    """Verifies all four SourceLinesOrigin enum members, explicit properties, and implicit-bool rejection."""
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    expected = {
        SourceLinesOrigin.PRE_SLICED: (True, False, True),
        SourceLinesOrigin.PLACEHOLDER: (True, True, False),
        SourceLinesOrigin.FULL_BUFFER: (False, False, True),
        SourceLinesOrigin.DISK: (False, False, True),
    }
    assert set(SourceLinesOrigin) == set(expected.keys())
    for member, (exp_sliced, exp_placeholder, exp_real) in expected.items():
        assert member.is_pre_sliced is exp_sliced
        assert member.is_placeholder is exp_placeholder
        assert member.has_real_source is exp_real
        with pytest.raises(TypeError, match="no implicit truth value"):
            bool(member)


@pytest.mark.parametrize(
    ("case_id", "unit_dict", "expected_lines", "expected_origin_name"),
    [
        (
            "empty_source_lines_list",
            {"name": "fn", "start": 1, "end": 2, "source_lines": []},
            ["# Source for fn lines 1-2\n"],
            "PLACEHOLDER",
        ),
        (
            "empty_source_text_and_file_source",
            {"name": "fn", "start": 1, "end": 1, "source_text": "", "file_source": ""},
            ["# Source for fn lines 1-1\n"],
            "PLACEHOLDER",
        ),
        (
            "whitespace_only_source_text_and_file_source",
            {
                "name": "fn",
                "start": 1,
                "end": 2,
                "source_text": "   \n\t\r\n",
                "file_source": "  \n",
            },
            ["# Source for fn lines 1-2\n"],
            "PLACEHOLDER",
        ),
        (
            "source_text_precedes_file_source",
            {
                "name": "fn",
                "start": 1,
                "end": 1,
                "source_text": "from_source_text = 1\nsecond = 2\n",
                "file_source": "from_file_source = 99\n",
            },
            ["from_source_text = 1\n"],
            "FULL_BUFFER",
        ),
        (
            "file_source_used_when_source_text_blank",
            {
                "name": "fn",
                "start": 2,
                "end": 2,
                "source_text": "   \n",
                "file_source": "line_one = 1\nline_two = 2\n",
            },
            ["line_two = 2\n"],
            "FULL_BUFFER",
        ),
        (
            "explicit_pre_sliced_lines",
            {
                "name": "fn",
                "start": 10,
                "end": 12,
                "source_lines": ["alpha = 1", "beta = 2\n"],
                "source_lines_is_sliced": True,
            },
            ["alpha = 1\n", "beta = 2\n"],
            "PRE_SLICED",
        ),
        (
            "explicit_unsliced_full_buffer_line_one_excerpt",
            {
                "name": "fn",
                "start": 1,
                "end": 1,
                "source_lines": ["only_line = 42\n"],
                "source_lines_is_sliced": False,
            },
            ["only_line = 42\n"],
            "FULL_BUFFER",
        ),
        (
            "heuristic_pre_sliced_line_one_excerpt",
            {
                "name": "fn",
                "start": 1,
                "end": 1,
                "source_lines": ["only_line = 42\n"],
            },
            ["only_line = 42\n"],
            "PRE_SLICED",
        ),
        (
            "crlf_source_text_full_buffer",
            {
                "name": "fn",
                "start": 2,
                "end": 3,
                "source_text": "a = 1\r\nb = 2\r\nc = 3\r\n",
            },
            ["b = 2\r\n", "c = 3\r\n"],
            "FULL_BUFFER",
        ),
        (
            "lone_cr_source_text_normalized_trailing_lf",
            {
                "name": "fn",
                "start": 1,
                "end": 2,
                "source_text": "a = 1\rb = 2\rc = 3\r",
            },
            ["a = 1\n", "b = 2\n"],
            "FULL_BUFFER",
        ),
        (
            "end_exceeds_available_buffer_lines_clamps_slice",
            {
                "name": "fn",
                "start": 2,
                "end": 25,
                "source_lines": ["l1 = 1\n", "l2 = 2\n", "l3 = 3\n"],
                "source_lines_is_sliced": False,
            },
            ["l2 = 2\n", "l3 = 3\n"],
            "FULL_BUFFER",
        ),
    ],
)
def test_extract_unit_source_lines_with_state_buffer_and_line_ending_matrix(
    case_id: str,
    unit_dict: dict[str, Any],
    expected_lines: list[str],
    expected_origin_name: str,
) -> None:
    """Verifies extract_unit_source_lines_with_state across buffer precedence, flags, CRLF/CR, and bounds."""
    from pydoppelgangerhunt.reporters import (
        extract_unit_source_code,
        extract_unit_source_lines_with_state,
    )
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    lines, origin = extract_unit_source_lines_with_state(unit_dict)
    assert lines == expected_lines, case_id
    assert origin is SourceLinesOrigin[expected_origin_name], case_id
    assert extract_unit_source_code(unit_dict) == expected_lines, case_id


def test_source_lines_origin_disk_notebooks_col_sliced_and_comment_impostor(
    tmp_path: Path,
) -> None:
    """Verifies DISK vs PLACEHOLDER across notebooks, CRLF/CR files, col-sliced flags, and comment-only files."""
    import json
    from pydoppelgangerhunt.reporters import extract_unit_source_lines_with_state
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    # 1. Valid notebook cell vs out-of-range cell vs corrupt JSON notebook
    valid_nb = tmp_path / "valid.ipynb"
    valid_nb.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": ["x = 10\n", "y = x + 5"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    nb_lines, nb_origin = extract_unit_source_lines_with_state(
        {"file": "valid.ipynb#cell_1", "name": "nb_u", "start": 1, "end": 2},
        repo_root=str(tmp_path),
    )
    assert nb_lines == ["x = 10\n", "y = x + 5\n"]
    assert nb_origin is SourceLinesOrigin.DISK

    oob_lines, oob_origin = extract_unit_source_lines_with_state(
        {"file": "valid.ipynb#cell_9", "name": "nb_oob", "start": 1, "end": 2},
        repo_root=str(tmp_path),
    )
    assert oob_lines == []
    assert oob_origin is SourceLinesOrigin.PLACEHOLDER
    oob_scope = inspect_single_unit_scope(
        {"file": "valid.ipynb#cell_9", "name": "nb_oob", "start": 1, "end": 2},
        repo_root=str(tmp_path),
    )
    assert "unavailable_source_placeholder" in oob_scope["control_flow_hazards"]

    corrupt_nb = tmp_path / "corrupt.ipynb"
    corrupt_nb.write_text("{not: valid json", encoding="utf-8")
    bad_cell_lines, bad_cell_origin = extract_unit_source_lines_with_state(
        {"file": "corrupt.ipynb#cell_1", "name": "bad_nb", "start": 1, "end": 2},
        repo_root=str(tmp_path),
    )
    assert bad_cell_lines == []
    assert bad_cell_origin is SourceLinesOrigin.PLACEHOLDER

    bad_raw_lines, bad_raw_origin = extract_unit_source_lines_with_state(
        {"file": "corrupt.ipynb", "name": "bad_nb", "start": 1, "end": 2},
        repo_root=str(tmp_path),
    )
    assert bad_raw_lines == ["# Source for bad_nb lines 1-2\n"]
    assert bad_raw_origin is SourceLinesOrigin.PLACEHOLDER

    # 2. Disk files with CRLF and lone-CR line endings, plus end > available lines
    crlf_py = tmp_path / "crlf_mod.py"
    crlf_py.write_bytes(b"first = 1\r\nsecond = 2\r\n")
    d_crlf, o_crlf = extract_unit_source_lines_with_state(
        {"file": "crlf_mod.py", "start": 1, "end": 10},
        repo_root=str(tmp_path),
    )
    assert d_crlf == ["first = 1\n", "second = 2\n"]
    assert o_crlf is SourceLinesOrigin.DISK

    cr_py = tmp_path / "cr_mod.py"
    cr_py.write_bytes(b"alpha = 1\rbeta = 2\r")
    d_cr, o_cr = extract_unit_source_lines_with_state(
        {"file": "cr_mod.py", "start": 2, "end": 2},
        repo_root=str(tmp_path),
    )
    assert d_cr == ["beta = 2\n"]
    assert o_cr is SourceLinesOrigin.DISK

    # 3. source_lines_is_col_sliced=False trims columns on line-sliced excerpt; True preserves pre-trimmed excerpt
    full_line = "    result = [item * 2 for item in items] + suffix\n"
    s_col = full_line.index("[")
    e_col = full_line.index("]") + 1
    sc_line_only = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": s_col,
            "end_col": e_col,
            "kind": "comprehension",
            "source_lines": [full_line],
            "source_lines_is_sliced": True,
            "source_lines_is_col_sliced": False,
            "allow_snippet_only": True,
        }
    )
    assert sc_line_only["inputs"] == ["items"]
    assert "suffix" not in sc_line_only["inputs"]

    sc_col_sliced = inspect_single_unit_scope(
        {
            "start": 2,
            "end": 2,
            "start_col": s_col,
            "end_col": e_col,
            "kind": "comprehension",
            "source_lines": ["[item * 2 for item in items]\n"],
            "source_lines_is_sliced": True,
            "source_lines_is_col_sliced": True,
            "allow_snippet_only": True,
        }
    )
    assert sc_col_sliced["inputs"] == ["items"]

    # 4. Real file containing literal `# Source for ...` comment vs missing-file placeholder with identical text
    impostor_text = "# Source for comment_unit lines 1-1\n"
    real_comment_file = tmp_path / "real_comment.py"
    real_comment_file.write_text(impostor_text, encoding="utf-8")

    real_lines, real_origin = extract_unit_source_lines_with_state(
        {"file": "real_comment.py", "name": "comment_unit", "start": 1, "end": 1},
        repo_root=str(tmp_path),
    )
    missing_lines, missing_origin = extract_unit_source_lines_with_state(
        {"file": "missing_comment.py", "name": "comment_unit", "start": 1, "end": 1},
        repo_root=str(tmp_path),
    )
    assert real_lines == missing_lines == [impostor_text]
    assert real_origin is SourceLinesOrigin.DISK
    assert real_origin.has_real_source is True
    assert missing_origin is SourceLinesOrigin.PLACEHOLDER
    assert missing_origin.is_placeholder is True

    sc_real_comment = inspect_single_unit_scope(
        {
            "file": "real_comment.py",
            "name": "comment_unit",
            "start": 1,
            "end": 1,
            "kind": "data_table",
        },
        repo_root=str(tmp_path),
    )
    sc_missing_comment = inspect_single_unit_scope(
        {
            "file": "missing_comment.py",
            "name": "comment_unit",
            "start": 1,
            "end": 1,
            "kind": "data_table",
        },
        repo_root=str(tmp_path),
    )
    assert "unavailable_source_placeholder" not in sc_real_comment["control_flow_hazards"]
    assert sc_real_comment["is_control_flow_safe"] is True
    assert "unavailable_source_placeholder" in sc_missing_comment["control_flow_hazards"]
    assert sc_missing_comment["is_control_flow_safe"] is False


@pytest.mark.parametrize(
    ("cell", "expected_lines"),
    [
        pytest.param(
            {"cell_type": "code", "source": [None, 123, False]},
            None,
            id="non_string_fragments",
        ),
        pytest.param(
            {"cell_type": "code", "source": ["x = 1\n", 2]},
            None,
            id="mixed_string_and_int_fragments",
        ),
        pytest.param({"cell_type": "code"}, None, id="missing_source"),
        pytest.param({"cell_type": "code", "source": {"x": 1}}, None, id="mapping_source"),
        pytest.param({"cell_type": "markdown", "source": ["x = 1\n"]}, None, id="markdown_cell"),
        pytest.param({"source": ["x = 1\n"]}, None, id="missing_cell_type"),
        pytest.param(["x = 1\n"], None, id="non_dict_cell"),
        pytest.param({"cell_type": "code", "source": []}, [], id="empty_fragment_list"),
        pytest.param({"cell_type": "code", "source": ""}, [], id="empty_source_string"),
        pytest.param(
            {"cell_type": "code", "source": ["a = 1\nb = 2\n", "c = 3"]},
            ["a = 1\n", "b = 2\n", "c = 3\n"],
            id="embedded_newlines_in_fragment",
        ),
        pytest.param(
            {"cell_type": "code", "source": ["total = ", "1 + 2\n", "out = total\n"]},
            ["total = 1 + 2\n", "out = total\n"],
            id="fragments_without_newline_are_concatenated",
        ),
        pytest.param(
            {"cell_type": "code", "source": ["a = 1\r\n", "b = 2\r\n"]},
            ["a = 1\n", "b = 2\n"],
            id="crlf_fragments_normalized",
        ),
        pytest.param(
            {"cell_type": "code", "source": "a = 1\rb = 2\r"},
            ["a = 1\n", "b = 2\n"],
            id="lone_cr_string_normalized",
        ),
        pytest.param(
            {"cell_type": "code", "source": "val = 1\n\x0cres = val + 2\n"},
            ["val = 1\n", "\x0cres = val + 2\n"],
            id="form_feed_is_not_a_line_break",
        ),
    ],
)
def test_notebook_cell_lines_reject_malformed_source_and_split_physical_lines(
    cell: Any, expected_lines: Any
) -> None:
    """Verifies notebook cell extraction never coerces malformed source and splits joined fragments on physical newlines."""
    from pydoppelgangerhunt.reporters import _extract_notebook_cell_lines

    assert _extract_notebook_cell_lines({"cells": [cell]}, 0) == expected_lines


@pytest.mark.parametrize(
    "bad_cell",
    [
        pytest.param({"cell_type": "code", "source": [None, 123, False]}, id="non_string_fragments"),
        pytest.param({"cell_type": "code"}, id="missing_source"),
        pytest.param({"cell_type": "markdown", "source": ["x = 1\n"]}, id="markdown_cell"),
        pytest.param("not_a_cell", id="non_dict_cell"),
    ],
)
def test_malformed_notebook_cell_reports_placeholder_origin_and_fails_closed(
    tmp_path: Path, bad_cell: Any
) -> None:
    """Verifies malformed notebook cells yield placeholder origin and unavailable_source_placeholder hazards."""
    import json

    from pydoppelgangerhunt.reporters import extract_unit_source_lines_with_state
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    nb_file = tmp_path / "malformed.ipynb"
    nb_file.write_text(json.dumps({"cells": [bad_cell]}), encoding="utf-8")
    unit = {"file": "malformed.ipynb#cell_1", "name": "bad_cell", "start": 1, "end": 3}
    lines, origin = extract_unit_source_lines_with_state(unit, repo_root=str(tmp_path))
    assert lines == []
    assert origin is SourceLinesOrigin.PLACEHOLDER
    scope = inspect_single_unit_scope(unit, repo_root=str(tmp_path))
    assert "unavailable_source_placeholder" in scope["control_flow_hazards"]
    assert scope["is_control_flow_safe"] is False


def test_valid_notebook_cell_with_embedded_newline_fragments_matches_parser_coordinates(
    tmp_path: Path,
) -> None:
    """Verifies reporter line coordinates match parser-harvested units when fragments embed newlines."""
    import json

    from pydoppelgangerhunt.parser import harvest_notebook_units
    from pydoppelgangerhunt.reporters import extract_unit_source_lines_with_state
    from pydoppelgangerhunt.source_lines import SourceLinesOrigin

    nb_file = tmp_path / "joined.ipynb"
    nb_file.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "def scale(items, factor):\n    out = []\n",
                            "    for item in items:\n        out.append(item * factor)\n",
                            "    return out",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    units = harvest_notebook_units(str(nb_file), str(tmp_path), min_lines=1, min_tokens=1)
    fn_unit = next(u for u in units if u.get("kind") == "function")
    lines, origin = extract_unit_source_lines_with_state(fn_unit, repo_root=str(tmp_path))
    assert origin is SourceLinesOrigin.DISK
    assert lines == [
        "def scale(items, factor):\n",
        "    out = []\n",
        "    for item in items:\n",
        "        out.append(item * factor)\n",
        "    return out\n",
    ]


def test_notebook_harvest_and_metrics_skip_malformed_cells_without_crashing(tmp_path: Path) -> None:
    """Verifies parser harvesting and DRY metrics skip malformed cells and non-object notebook roots."""
    import json

    from pydoppelgangerhunt.metrics import compute_repository_dry_stats
    from pydoppelgangerhunt.parser import harvest_notebook_units

    nb_file = tmp_path / "mixed.ipynb"
    nb_file.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": [None, 7]},
                    {"cell_type": "code", "source": {"bogus": True}},
                    {
                        "cell_type": "code",
                        "source": ["def keep(a):\n", "    b = a + 1\n", "    return b\n"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    units = harvest_notebook_units(str(nb_file), str(tmp_path), min_lines=1, min_tokens=1)
    assert units
    assert {str(u["file"]).rsplit("#", 1)[-1] for u in units} == {"cell_3"}

    root_list_nb = tmp_path / "root_list.ipynb"
    root_list_nb.write_text(json.dumps([{"cell_type": "code", "source": "x = 1\n"}]), encoding="utf-8")
    assert harvest_notebook_units(str(root_list_nb), str(tmp_path), min_lines=1, min_tokens=1) == []

    stats = compute_repository_dry_stats(str(tmp_path), [], include_notebooks=True)
    assert stats["sloc"] == 3


def test_generator_yield_structural_rejection_not_classified_as_safety_hazard() -> None:
    """Verifies non-blocking generator_yield in control_flow_hazards is not treated as a safety hazard."""
    from pydoppelgangerhunt.reporters import (
        _collect_suggestion_rejection_reasons,
        synthesize_refactoring_suggestion,
    )

    sc_gen1: dict[str, Any] = {
        "control_flow_hazards": ["generator_yield"],
        "closure_hazard_reasons": [],
        "is_async": False,
        "has_yield": True,
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
    }
    sc_gen2 = dict(sc_gen1)
    u1_stub: dict[str, Any] = {"file": "a.py", "start": 1, "end": 4, "kind": "function"}
    u2_stub: dict[str, Any] = {"file": "a.py", "start": 6, "end": 9, "kind": "function"}

    info = _collect_suggestion_rejection_reasons(
        u1_stub,
        u2_stub,
        sc_gen1,
        sc_gen2,
        pair_skip_reason="output arity mismatch",
        has_async_gen_hazard=False,
    )
    assert info.has_safety_hazard is False
    assert "generator yield" not in info.reasons
    assert "output arity mismatch" in info.reasons

    # Negative control: blocking control-flow hazard still sets has_safety_hazard=True.
    sc_blocking = dict(sc_gen1, control_flow_hazards=["generator_yield", "naked_break"])
    info_blocking = _collect_suggestion_rejection_reasons(
        u1_stub,
        u2_stub,
        sc_blocking,
        sc_gen2,
        pair_skip_reason="",
        has_async_gen_hazard=False,
    )
    assert info_blocking.has_safety_hazard is True
    assert "naked break" in info_blocking.reasons

    gen_src_a = [
        "def gen_a(xs):\n",
        "    a = 1\n",
        "    yield xs\n",
        "    return a\n",
    ]
    gen_src_b = [
        "def gen_b(xs):\n",
        "    a = 1\n",
        "    b = 2\n",
        "    yield xs\n",
        "    return a, b\n",
    ]
    sug = synthesize_refactoring_suggestion(
        {"file": "gen.py", "start": 1, "end": 4, "kind": "function", "source_lines": gen_src_a},
        {"file": "gen.py", "start": 1, "end": 5, "kind": "function", "source_lines": gen_src_b},
    )
    assert "skipped as structurally incompatible" in sug
    assert "rejected as unsafe" not in sug


def test_multiline_end_col_zero_allowed_while_single_line_rejected() -> None:
    """Verifies multiline spans with end_col=0 are valid while single-line end_col=0 is rejected."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols

    multi_unit: dict[str, Any] = {
        "start": 1,
        "end": 2,
        "start_col": 4,
        "end_col": 0,
        "kind": "block",
        "source_lines": ["    x = 1\n", "y = 2\n"],
    }
    assert (
        _has_out_of_bounds_unit_cols(
            multi_unit,
            multi_unit["source_lines"],
            is_pre_sliced=False,
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is False
    )
    sc_multi = inspect_single_unit_scope(multi_unit)
    assert "invalid_unit_columns" not in sc_multi["control_flow_hazards"]

    single_unit: dict[str, Any] = {
        "start": 1,
        "end": 1,
        "start_col": 0,
        "end_col": 0,
        "kind": "complex_expr",
        "source_lines": ["x + 1\n"],
    }
    assert (
        _has_out_of_bounds_unit_cols(
            single_unit,
            single_unit["source_lines"],
            is_pre_sliced=False,
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is True
    )
    sc_single = inspect_single_unit_scope(single_unit)
    assert "invalid_unit_columns" in sc_single["control_flow_hazards"]


@pytest.mark.parametrize(
    ("unit_dict", "expect_valid"),
    [
        ({"start": 5, "end": 5, "start_col": 10, "end_col": 5}, False),
        ({"lineno": 5, "end_lineno": 5, "col_offset": 10, "end_col_offset": 5}, False),
        ({"start": 5, "end": 6, "start_col": 10, "end_col": 5}, True),
        ({"start": 5, "end": 5, "start_col": 5, "end_col": 10}, True),
    ],
)
def test_validate_unit_schema_rejects_inverted_single_line_columns(
    unit_dict: dict[str, Any], expect_valid: bool
) -> None:
    """Verifies validate_unit_schema rejects inverted single-line columns while allowing multiline/valid spans."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    if expect_valid:
        assert validate_unit_schema(unit_dict, strict=True) == set()
    else:
        with pytest.raises(ValueError, match="Inverted column coordinates"):
            validate_unit_schema(unit_dict, strict=True)


def test_weak_parent_map_len_excludes_collected_parents() -> None:
    """Verifies _WeakParentMap.__len__ matches live iterated entries after parent garbage collection."""
    import gc
    from pydoppelgangerhunt.fixer._df_cache import _WeakParentMap

    parent = ast.Module(body=[], type_ignores=[])
    child = ast.Expr(value=ast.Constant(value=1))
    pmap = _WeakParentMap({child: parent})
    assert len(pmap) == 1
    assert len(list(pmap)) == 1
    del parent
    gc.collect()
    assert len(pmap) == 0
    assert len(list(pmap)) == 0


def test_normalize_non_suppressing_ctx_managers_handles_cyclic_input() -> None:
    """Verifies self-referential containers passed to normalize_non_suppressing_ctx_managers terminate cleanly."""
    from pydoppelgangerhunt.config import normalize_non_suppressing_ctx_managers

    cyclic: list[Any] = ["LockA"]
    cyclic.append(cyclic)
    assert normalize_non_suppressing_ctx_managers(cyclic) == ("LockA",)


def test_app_version_matches_pyproject_toml() -> None:
    """Verifies APP_VERSION stays synchronized with pyproject.toml."""
    import tomllib
    from pydoppelgangerhunt.config import APP_VERSION

    repo_root = Path(__file__).resolve().parents[2]
    pyproject_data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert APP_VERSION == pyproject_data["project"]["version"]


def test_full_buffer_out_of_range_slice_returns_placeholder() -> None:
    """Verifies out-of-range FULL_BUFFER line slices fail closed as PLACEHOLDER."""
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    _, origin = extract_unit_source_lines_with_state(
        {
            "file": "missing.py",
            "start": 10,
            "end": 12,
            "source_lines": ["x = 1\n", "y = 2\n"],
        }
    )
    assert origin is SourceLinesOrigin.PLACEHOLDER
