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
    assert sliced_inside is not None
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
    assert sliced_exact_first is None
    sliced_beyond_first = _slice_unit_token_lines(
        {
            "kind": "complex_expr",
            "start_col": first_bytes_len + 10,
            "end_col": last_bytes_len,
        },
        ml_lines,
    )
    assert sliced_beyond_first is None
    utf8_ml = ['    "\u00e9\u754c" + (\n', "        1 + 2\n", "    )\n"]
    after_mb_col = len('    "\u00e9\u754c" + '.encode("utf-8"))
    sliced_mb = _slice_unit_token_lines(
        {"kind": "complex_expr", "start_col": after_mb_col, "end_col": 5},
        utf8_ml,
    )
    assert sliced_mb is not None and sliced_mb[0] == "(\n"
    presliced_ml = ["(\n", "        1 + 2\n", "    )\n"]
    assert (
        _slice_unit_token_lines(
            {
                "kind": "complex_expr",
                "start_col": 8,
                "end_col": last_bytes_len,
            },
            presliced_ml,
        )
        is None
    )
    assert (
        _slice_unit_token_lines(
            {
                "kind": "complex_expr",
                "start_col": 8,
                "end_col": last_bytes_len,
            },
            presliced_ml,
            explicit_col_sliced=False,
        )
        is None
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

    # 1. Column clipping when explicit_col_sliced=False on a 2-line file where unit is lines 1..2
    full_two_lines = ["a = 1\n", "b = 2\n"]
    sliced_full = _slice_unit_token_lines(
        {"start": 1, "end": 2},
        full_two_lines,
        explicit_col_sliced=False,
    )
    assert sliced_full == ["a = 1\n", "b = 2\n"]

    # 2. Line-sliced source_lines (without source_lines_is_col_sliced) still undergoes column slicing
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

    # 3. Pre-sliced multi-line snippet with start_col/end_col is not double-sliced under explicit_col_sliced=True
    pre_sliced_multi = ["x = (\n", "    a + 1\n", ")"]
    res_multi = _slice_unit_token_lines(
        {"start": 10, "end": 12, "start_col": 4, "end_col": 5},
        pre_sliced_multi,
        explicit_col_sliced=True,
    )
    assert res_multi == pre_sliced_multi

    # 4. _has_invalid_lexical_scope_declarations lets SyntaxWarning/DeprecationWarning propagate in library mode
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
    assert any(
        issubclass(w.category, (SyntaxWarning, DeprecationWarning))
        for w in caught
    )


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
    assert "__bool__" not in SourceLinesOrigin.__dict__
    for member in SourceLinesOrigin:
        assert bool(member) is True
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
        is None
    )
    assert (
        _slice_unit_token_lines(
            {"kind": "comprehension", "start_col": 10, "end_col": 999},
            [line],
        )
        is None
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

    for removed_alias in ("fail_closed", "fast", "skip"):
        clear_closure_warning_cache()
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            res_mode, res_skip = resolve_closure_strictness_mode(
                closure_strictness=removed_alias
            )
        assert (res_mode, res_skip) == ("strict", False)
        assert any(
            f"Unrecognized closure_strictness '{removed_alias}'" in rec.message
            for rec in caplog.records
        )
        with pytest.raises(ValueError, match="closure_strictness"):
            validate_unit_schema(
                {"start": 1, "end": 2, "closure_strictness": removed_alias},
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
        assert bool(member) is True


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
    assert bad_raw_lines == ["{not: valid json\n"]
    assert bad_raw_origin is SourceLinesOrigin.DISK

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
    from pydoppelgangerhunt.source_lines import _extract_notebook_cell_lines

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
    import re
    from pydoppelgangerhunt.config import APP_VERSION

    repo_root = Path(__file__).resolve().parents[2]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    project_section = pyproject_text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_section)
    assert match is not None
    assert APP_VERSION == match.group(1)


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


@pytest.mark.parametrize(
    ("call_expr", "lambda_body", "expect_safe"),
    [
        ("self.cb()", "total", False),
        ("Worker.cb(self)", "total", False),
        ("self.cb()", "1", True),
    ],
)
def test_local_class_body_lambda_attr_closure_captured_in_method_unit(
    call_expr: str, lambda_body: str, expect_safe: bool
) -> None:
    """Verifies class-body lambda closures on an enclosing local class are tracked across self.<attr>() and ClassName.<attr>() calls."""
    src = (
        "def outer(items):\n"
        "    total = 0\n"
        "    class Worker:\n"
        f"        cb = lambda self: {lambda_body}\n"
        "        def run(self):\n"
        "            nonlocal total\n"
        "            out = []\n"
        "            for x in items:\n"
        "                total += x\n"
        f"                out.append({call_expr})\n"
        "            return out\n"
        "    return Worker().run(), total\n"
    )
    tree = ast.parse(src)
    sc = inspect_single_unit_scope(
        {
            "start": 8,
            "end": 10,
            "kind": "block",
            "source_text": src,
            "source_lines": src.splitlines(keepends=True),
        },
        tree=tree,
    )
    assert sc["is_control_flow_safe"] is expect_safe
    if not expect_safe:
        assert "severed_closure_cell" in sc["control_flow_hazards"]
        assert sc["closure_hazard_reasons"]


def test_tree_only_except_handler_unit_does_not_trigger_stale_source_code() -> None:
    """Verifies tree-only except clause units match ExceptHandler nodes without false stale_source_code."""
    src = (
        "def f(x):\n"
        "    try:\n"
        "        a = int(x)\n"
        "    except ValueError:\n"
        "        a = 2\n"
        "    except RuntimeError:\n"
        "        a = 3\n"
        "    return a\n"
    )
    tree = ast.parse(src)
    unit: dict[str, Any] = {
        "file": "missing_disk_tree_only.py",
        "start": 4,
        "end": 5,
        "kind": "block",
        "source_code": "    except ValueError:\n        a = 2\n",
    }
    sc = inspect_single_unit_scope(unit, tree=tree)
    assert "stale_source_code" not in sc["control_flow_hazards"]


def test_stale_source_code_classified_as_safety_hazard_on_expression_units() -> None:
    """Verifies stale_source_code on complex_expr units is reported as a blocking safety hazard."""
    from pydoppelgangerhunt.fixer.scope import BLOCKING_CONTROL_FLOW_HAZARDS
    from pydoppelgangerhunt.reporters import (
        _collect_suggestion_rejection_reasons,
        synthesize_refactoring_suggestion,
    )

    assert "stale_source_code" in BLOCKING_CONTROL_FLOW_HAZARDS
    sc_stale: dict[str, Any] = {
        "control_flow_hazards": ["stale_source_code"],
        "closure_hazard_reasons": [],
        "is_async": False,
        "has_yield": False,
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
    }
    info = _collect_suggestion_rejection_reasons(
        {"file": "a.py", "start": 1, "end": 1, "kind": "complex_expr"},
        {"file": "a.py", "start": 2, "end": 2, "kind": "complex_expr"},
        sc_stale,
        sc_stale,
        pair_skip_reason=None,
        has_async_gen_hazard=False,
    )
    assert info.has_safety_hazard is True
    assert "stale source code" in info.reasons

    u_stale: dict[str, Any] = {
        "file": "stale_expr.py",
        "start": 1,
        "end": 1,
        "kind": "complex_expr",
        "source_text": "z = b + 2\nx = c + 3\n",
        "source_code": "y = a + 1",
    }
    sug = synthesize_refactoring_suggestion(u_stale, dict(u_stale, start=2, end=2))
    assert "rejected as unsafe" in sug
    assert "stale source code" in sug


def test_enclosing_lambda_and_comprehension_nonlocal_and_outer_global_hazards(
    tmp_path: Path,
) -> None:
    """Verifies shared enclosing_chain helper extracts outer nonlocal/global hazards through enclosing lambda."""
    from pydoppelgangerhunt.fixer._df_closures import (
        _extract_chain_nonlocals_and_outer_globals,
    )
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    # Unit sits on a comprehension inside outer()'s lambda while outer() binds nonlocal & outer_global
    src = (
        "EXTERN_FLAG = 0\n"
        "def make_ops():\n"
        "    shared = 0\n"
        "    def outer(cond):\n"
        "        nonlocal shared\n"
        "        global EXTERN_FLAG\n"
        "        if cond:\n"
        "            res = (lambda items: [shared + EXTERN_FLAG + x for x in items])([1])\n"
        "        return res\n"
        "    return outer\n"
    )
    tree = ast.parse(src)
    make_ops_fn = tree.body[1]
    assert isinstance(make_ops_fn, ast.FunctionDef)
    outer_fn = make_ops_fn.body[1]
    assert isinstance(outer_fn, ast.FunctionDef)
    if_stmt = outer_fn.body[2]
    assert isinstance(if_stmt, ast.If)
    assign_stmt = if_stmt.body[0]
    assert isinstance(assign_stmt, ast.Assign)
    call_expr = assign_stmt.value
    assert isinstance(call_expr, ast.Call)
    lam_node = call_expr.func
    assert isinstance(lam_node, ast.Lambda)

    chain: list[tuple[int, int, ast.AST]] = [
        (2, 10, make_ops_fn),
        (4, 9, outer_fn),
        (8, 8, lam_node),
    ]
    inner_stmt_fn, enc_nl, enc_og = _extract_chain_nonlocals_and_outer_globals(
        chain, 8, 8
    )
    assert inner_stmt_fn is outer_fn
    assert "shared" in enc_nl
    assert "EXTERN_FLAG" in enc_og

    mod_file = tmp_path / "enclosing_lambda.py"
    mod_file.write_text(src, encoding="utf-8")
    line8 = src.splitlines()[7]
    comp_text = "[shared + EXTERN_FLAG + x for x in items]"
    comp_start_col = line8.index(comp_text)
    comp_end_col = comp_start_col + len(comp_text)
    comp_unit: dict[str, Any] = {
        "file": str(mod_file),
        "start": 8,
        "end": 8,
        "start_col": comp_start_col,
        "end_col": comp_end_col,
        "kind": "comprehension",
        "source_code": comp_text,
    }
    sc = inspect_single_unit_scope(comp_unit, repo_root=str(tmp_path), tree=tree)
    assert "shared" in sc["inputs"]
    assert "EXTERN_FLAG" in sc["inputs"]

    # Negative control: outer function without nonlocal/global reports empty sets
    clean_src = (
        "def make_clean():\n"
        "    def outer():\n"
        "        return lambda items: [x + 1 for x in items]\n"
        "    return outer\n"
    )
    clean_tree = ast.parse(clean_src)
    clean_outer = clean_tree.body[0].body[0]  # type: ignore[attr-defined]
    clean_lam = clean_outer.body[0].value
    clean_stmt_fn, clean_nl, clean_og = _extract_chain_nonlocals_and_outer_globals(
        [(1, 4, clean_tree.body[0]), (2, 3, clean_outer), (3, 3, clean_lam)],
        3,
        3,
    )
    assert clean_stmt_fn is clean_outer
    assert clean_nl == set()
    assert clean_og == set()


def test_validate_unit_schema_and_oob_column_edge_cases() -> None:
    """Verifies omitted-end single-line column inversion check and _has_out_of_bounds_unit_cols edge cases."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    with pytest.raises(ValueError, match="Inverted column coordinates"):
        validate_unit_schema({"file": "a.py", "start": 2, "start_col": 10, "end_col": 4})

    # Control: valid single-line columns with omitted end
    validate_unit_schema({"file": "a.py", "start": 2, "start_col": 2, "end_col": 8})

    # 1. Multi-line starting on empty line with start_col=0 is in bounds; start_col=1 is OOB
    assert (
        _has_out_of_bounds_unit_cols(
            {"file": "a.py", "start": 1, "end": 2, "start_col": 0},
            ["\n", "x = 1\n"],
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is False
    )
    assert (
        _has_out_of_bounds_unit_cols(
            {"file": "a.py", "start": 1, "end": 2, "start_col": 1},
            ["\n", "x = 1\n"],
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is True
    )
    # 2. Single-line zero-width end_col=0 with omitted start_col is OOB
    assert (
        _has_out_of_bounds_unit_cols(
            {"file": "a.py", "start": 1, "end": 1, "end_col": 0},
            ["x = 1\n"],
            is_placeholder_origin=False,
            explicit_col_sliced=False,
        )
        is True
    )
    # 3. explicit_col_sliced=True still flags inverted single-line columns
    assert (
        _has_out_of_bounds_unit_cols(
            {"file": "a.py", "start": 1, "end": 1, "start_col": 5, "end_col": 2},
            ["abc\n"],
            is_placeholder_origin=False,
            explicit_col_sliced=True,
        )
        is True
    )


def test_synthesize_refactoring_suggestion_and_align_pair_handles_none_kind(
    tmp_path: Path,
) -> None:
    """Verifies None/empty unit kinds adopt partner expression kind during suggestion synthesis."""
    from pydoppelgangerhunt.fixer.synthesis import _align_pair_expr_kinds
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    p1: dict[str, Any] = {"kind": None, "file": "a.py", "start": 1, "end": 1}
    p2: dict[str, Any] = {"kind": "comprehension", "file": "b.py", "start": 1, "end": 1}
    assert _align_pair_expr_kinds(p1, p2) is True
    assert p1["kind"] == "comprehension"
    assert p2["kind"] == "comprehension"

    b1: dict[str, Any] = {"kind": "block", "file": "a.py", "start": 1, "end": 1}
    b2: dict[str, Any] = {"kind": "comprehension", "file": "b.py", "start": 1, "end": 1}
    assert _align_pair_expr_kinds(b1, b2) is False
    assert b1["kind"] == "block"
    assert b2["kind"] == "comprehension"

    src_a = "def fa(items):\n    vals = [x + 1 for x in items]\n    return vals\n"
    src_b = "def fb(items):\n    out = [y + 1 for y in items]\n    return out\n"
    fa = tmp_path / "a.py"
    fb = tmp_path / "b.py"
    fa.write_text(src_a, encoding="utf-8")
    fb.write_text(src_b, encoding="utf-8")
    line_a2 = src_a.splitlines()[1]
    expr_a = "[x + 1 for x in items]"
    col_a = line_a2.index(expr_a)
    line_b2 = src_b.splitlines()[1]
    expr_b = "[y + 1 for y in items]"
    col_b = line_b2.index(expr_b)
    u1: dict[str, Any] = {
        "file": str(fa),
        "start": 2,
        "end": 2,
        "start_col": col_a,
        "end_col": col_a + len(expr_a),
        "kind": None,
        "source_code": expr_a,
    }
    u2: dict[str, Any] = {
        "file": str(fb),
        "start": 2,
        "end": 2,
        "start_col": col_b,
        "end_col": col_b + len(expr_b),
        "kind": "comprehension",
        "source_code": expr_b,
    }
    sug = synthesize_refactoring_suggestion(u1, u2, repo_root=str(tmp_path))
    assert "rejected as unsafe" not in sug
    assert "def _shared" in sug


def test_resolve_cached_unit_source_and_tree_populates_cache_for_presliced_unit(
    tmp_path: Path,
) -> None:
    """Verifies pre-sliced unit source_lines loads and caches disk module instead of returning (None, None)."""
    from collections import OrderedDict

    from pydoppelgangerhunt.reporters import resolve_cached_unit_source_and_tree

    mod_file = tmp_path / "sliced.py"
    full_src = "def alpha(x):\n    y = x + 1\n    return y\n"
    mod_file.write_text(full_src, encoding="utf-8")

    # Start=2, end=3 with only the 2 sliced lines (NOT full 3-line buffer)
    sliced_unit: dict[str, Any] = {
        "file": str(mod_file),
        "start": 2,
        "end": 3,
        "source_lines": ["    y = x + 1\n", "    return y\n"],
    }
    cache: OrderedDict[str, tuple[str | None, ast.AST | None]] = OrderedDict()
    text, tree = resolve_cached_unit_source_and_tree(
        sliced_unit, repo_root=str(tmp_path), file_cache=cache
    )
    assert text == full_src
    assert tree is not None
    assert str(mod_file.resolve()) in cache

    # Control: full-file buffer in source_lines keeps in-memory bypass without populating disk cache
    full_buf_unit: dict[str, Any] = {
        "file": str(mod_file),
        "start": 2,
        "end": 3,
        "source_lines": full_src.splitlines(keepends=True),
    }
    empty_cache: OrderedDict[str, tuple[str | None, ast.AST | None]] = OrderedDict()
    t2, tr2 = resolve_cached_unit_source_and_tree(
        full_buf_unit, repo_root=str(tmp_path), file_cache=empty_cache
    )
    assert (t2, tr2) == (None, None)
    assert not empty_cache


@pytest.mark.parametrize(
    ("setup_stmt", "run_stmt", "expect_safe"),
    [
        pytest.param(
            "self.callbacks.append(lambda: total)",
            "out.append(self.callbacks[0]())",
            False,
            id="append_mutation_links_sibling_closure",
        ),
        pytest.param(
            "self.callbacks[0] = lambda: total",
            "out.append(self.callbacks[0]())",
            False,
            id="subscript_store_mutation_links_sibling_closure",
        ),
        pytest.param(
            "self.state.cb = lambda: total",
            "out.append(self.state.cb())",
            False,
            id="nested_attr_store_links_sibling_closure",
        ),
        pytest.param(
            "self.callbacks = [lambda: total]",
            "self.callbacks += [lambda: 1]",
            False,
            id="augassign_on_unit_attr_links_sibling_store",
        ),
        pytest.param(
            "self.callbacks += [lambda: total]",
            "out.append(self.callbacks[0]())",
            False,
            id="augassign_in_sibling_links_unit_load",
        ),
        pytest.param(
            "cbs = self.callbacks\ncbs.append(lambda: total)",
            "out.append(self.callbacks[0]())",
            False,
            id="local_alias_append_links_sibling_closure",
        ),
        pytest.param(
            "cbs = self.callbacks\ncbs2 = cbs\ncbs2[0] = lambda: total",
            "out.append(self.callbacks[0]())",
            False,
            id="transitive_local_alias_subscript_store_links_sibling_closure",
        ),
        pytest.param(
            "cbs = self.callbacks\ncbs += [lambda: total]",
            "out.append(self.callbacks[0]())",
            False,
            id="local_alias_augassign_links_sibling_closure",
        ),
        pytest.param(
            "st = self.state\nst.cb = lambda: total",
            "out.append(self.state.cb())",
            False,
            id="local_alias_attr_store_links_sibling_closure",
        ),
        pytest.param(
            "cbs = self.callbacks\nregister_hook(cbs, lambda: total)",
            "out.append(self.callbacks[0]())",
            False,
            id="local_alias_passed_to_helper_links_sibling_closure",
        ),
        pytest.param(
            "return self.callbacks.copy(), total",
            "out.append(self.callbacks[0]())",
            True,
            id="negative_readonly_method_does_not_link_sibling_closure",
        ),
        pytest.param(
            "cbs = self.callbacks\nreturn cbs.copy(), total",
            "out.append(self.callbacks[0]())",
            True,
            id="negative_local_alias_readonly_method_does_not_link",
        ),
        pytest.param(
            "cbs = self.callbacks\nreturn len(cbs), total",
            "out.append(self.callbacks[0]())",
            True,
            id="negative_local_alias_readonly_builtin_does_not_link",
        ),
        pytest.param(
            "self.other.append(lambda: total)",
            "out.append(self.callbacks[0]())",
            True,
            id="negative_unrelated_attr_mutation_does_not_link",
        ),
    ],
)
def test_local_class_receiver_in_place_container_and_subattr_mutations(
    setup_stmt: str, run_stmt: str, expect_safe: bool
) -> None:
    """Verifies in-place container, subscript, nested-attribute, and augmented receiver mutations link sibling closures."""
    setup_body = "\n".join(f"            {line}" for line in setup_stmt.splitlines())
    run_lines = [f"                {line}" for line in run_stmt.splitlines()]
    src = (
        "def outer(items):\n"
        "    total = 0\n"
        "    class Worker:\n"
        "        def __init__(self):\n"
        "            self.callbacks = []\n"
        "            self.other = []\n"
        "            self.state = type('S', (), {})()\n"
        "        def setup(self):\n"
        f"{setup_body}\n"
        "        def run(self):\n"
        "            nonlocal total\n"
        "            out = []\n"
        "            for x in items:\n"
        "                total += x\n"
        + "\n".join(run_lines)
        + "\n"
        "            return out\n"
        "    return Worker(), total\n"
    )
    lines = src.splitlines()
    start_idx = next(i + 1 for i, ln in enumerate(lines) if "out = []" in ln)
    end_idx = next(i + 1 for i, ln in enumerate(lines) if "return out" in ln) - 1
    tree = ast.parse(src)
    sc = inspect_single_unit_scope(
        {
            "start": start_idx,
            "end": end_idx,
            "kind": "block",
            "source_text": src,
            "source_lines": src.splitlines(keepends=True),
        },
        tree=tree,
    )
    assert sc["is_control_flow_safe"] is expect_safe
    if not expect_safe:
        assert "severed_closure_cell" in sc["control_flow_hazards"]
        assert sc["closure_hazard_reasons"]


@pytest.mark.parametrize(
    ("case_id", "unit_kwargs", "expect_oob"),
    [
        pytest.param(
            "single_line_inverted_with_placeholder_origin",
            {
                "start": 1,
                "end": 1,
                "start_col": 8,
                "end_col": 4,
                "is_placeholder_origin": True,
                "explicit_col_sliced": True,
            },
            True,
            id="single_line_inverted_with_placeholder_origin",
        ),
        pytest.param(
            "single_line_zero_width_with_placeholder_origin",
            {
                "start": 1,
                "end": 1,
                "start_col": 4,
                "end_col": 4,
                "is_placeholder_origin": True,
                "explicit_col_sliced": True,
            },
            True,
            id="single_line_zero_width_with_placeholder_origin",
        ),
        pytest.param(
            "multiline_allows_end_col_less_than_start_col",
            {
                "start": 1,
                "end": 2,
                "start_col": 10,
                "end_col": 5,
                "is_placeholder_origin": True,
                "explicit_col_sliced": True,
            },
            False,
            id="multiline_allows_end_col_less_than_start_col",
        ),
        pytest.param(
            "valid_single_line_nonempty_span",
            {
                "start": 1,
                "end": 1,
                "start_col": 2,
                "end_col": 6,
                "is_placeholder_origin": True,
                "explicit_col_sliced": True,
            },
            False,
            id="valid_single_line_nonempty_span",
        ),
    ],
)
def test_has_out_of_bounds_unit_cols_checks_placeholder_single_line_order(
    case_id: str,
    unit_kwargs: dict[str, Any],
    expect_oob: bool,
) -> None:
    """Verifies single-line inverted and zero-width columns are rejected even on placeholder/pre-sliced units."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols

    unit_dict = dict(unit_kwargs)
    is_placeholder = bool(unit_dict.pop("is_placeholder_origin", False))
    explicit_sliced = bool(unit_dict.pop("explicit_col_sliced", False))
    assert (
        _has_out_of_bounds_unit_cols(
            unit_dict,
            [],
            is_placeholder_origin=is_placeholder,
            explicit_col_sliced=explicit_sliced,
        )
        is expect_oob
    ), case_id


@pytest.mark.parametrize(
    ("start", "end", "start_col", "end_col", "err_pattern"),
    [
        pytest.param(1, 1, 4, 4, "Zero-width column coordinates", id="single_line_zero_width"),
        pytest.param(1, 1, 8, 4, "Inverted column coordinates", id="single_line_inverted"),
        pytest.param(1, 2, 10, 5, None, id="multiline_cross_col_ok"),
        pytest.param(1, 1, 2, 6, None, id="single_line_valid"),
    ],
)
def test_validate_unit_schema_single_line_zero_width_and_inverted_cols(
    start: int,
    end: int,
    start_col: int,
    end_col: int,
    err_pattern: str | None,
) -> None:
    """Verifies validate_unit_schema rejects both zero-width and inverted single-line column spans."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    unit: dict[str, Any] = {
        "start": start,
        "end": end,
        "kind": "expression",
        "start_col": start_col,
        "end_col": end_col,
    }
    if err_pattern is not None:
        with pytest.raises(ValueError, match=err_pattern):
            validate_unit_schema(unit)
    else:
        validate_unit_schema(unit)


def test_notebook_in_memory_validation_across_load_text_and_reporter_cache() -> None:
    """Verifies _load_unit_file_text and resolve_cached_unit_source_and_tree match notebook anchor validation."""
    import json
    from pydoppelgangerhunt.fixer.source import _load_unit_file_text
    from pydoppelgangerhunt.reporters import resolve_cached_unit_source_and_tree

    nb_json = json.dumps(
        {
            "nbformat": 4,
            "cells": [
                {"cell_type": "code", "source": ["def ok():\n", "    return 1\n"]},
                {"cell_type": "markdown", "source": ["# md\n"]},
            ],
        }
    )

    for bad_unit in (
        {"file": "a.ipynb#cell_0", "start": 1, "end": 1, "source_text": "x = 1\n"},
        {"file": "a.ipynb#bad", "start": 1, "end": 1, "file_source": "x = 1\n"},
        {"file": "a.ipynb#cell_99", "start": 1, "end": 1, "file_source": nb_json},
        {"file": "a.ipynb#cell_2", "start": 1, "end": 1, "source_text": nb_json},
        {"file": "a.ipynb#cell_1", "start": 1, "end": 1, "file_source": "{broken"},
    ):
        assert _load_unit_file_text(bad_unit) is None, bad_unit["file"]
        assert resolve_cached_unit_source_and_tree(bad_unit) == (None, None), bad_unit["file"]

    good_unit = {"file": "a.ipynb#cell_1", "start": 1, "end": 2, "file_source": nb_json}
    loaded_src = _load_unit_file_text(good_unit)
    assert loaded_src == "def ok():\n    return 1\n"
    cached_src, cached_tree = resolve_cached_unit_source_and_tree(good_unit)
    assert cached_src == "def ok():\n    return 1\n"
    assert isinstance(cached_tree, ast.Module)


def test_validate_unit_schema_boundary_explicit_vs_high_level_entrypoints(
    tmp_path: Path,
) -> None:
    """Verifies validate_unit_schema runs automatically at high-level entrypoints and must be called explicitly for low-level helpers."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    src = "def f(x):\n    y = x + 1\n    return y\n"
    mod = tmp_path / "mod.py"
    mod.write_text(src, encoding="utf-8")

    bad_unit = {
        "file": str(mod),
        "start": 1,
        "end": 3,
        "kind": "function",
        "inputs": {"x"},
    }
    # Lower-level helpers do not enforce validate_unit_schema automatically
    lines, origin = extract_unit_source_lines_with_state(bad_unit, repo_root=str(tmp_path))
    assert origin is SourceLinesOrigin.DISK
    assert len(lines) == 3
    low_scope = inspect_single_unit_scope(bad_unit, repo_root=str(tmp_path))
    assert isinstance(low_scope, dict)

    # Explicit validation rejects unordered set in positionally consumed 'inputs'
    with pytest.raises(ValueError, match="ordered sequence"):
        validate_unit_schema(bad_unit)

    # High-level refactoring entrypoints enforce validate_unit_schema automatically
    peer_unit = {"file": str(mod), "start": 1, "end": 3, "kind": "function"}
    with pytest.raises(ValueError, match="ordered sequence"):
        synthesize_shared_helper_code(bad_unit, peer_unit, repo_root=str(tmp_path))
    with pytest.raises(ValueError, match="ordered sequence"):
        generate_refactoring_patch([(1.0, bad_unit, peer_unit)], repo_root=str(tmp_path))


@pytest.mark.parametrize(
    "terminal_hazard",
    ["stale_source_code", "pathological_ast", "invalid_unit_columns"],
)
def test_ensure_unit_scope_closure_mode_rejects_contradictory_terminal_hazard(
    terminal_hazard: str,
) -> None:
    """Verifies terminal hazard scopes only short-circuit when is_control_flow_safe is False."""
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    src = "def f(x):\n    y = x + 1\n    return y\n"
    unit = {"start": 2, "end": 3, "kind": "compound_block", "source_text": src}

    # Forged or inconsistent dict claiming safe=True despite terminal hazard must NOT short-circuit
    contradictory: dict[str, Any] = {
        "inputs": ["x"],
        "outputs": ["y"],
        "control_flow_hazards": [terminal_hazard],
        "is_control_flow_safe": True,
    }
    recomputed = ensure_unit_scope_closure_mode(
        unit,
        contradictory,
        False,
        strict_with_suppression=True,
        source_text=src,
    )
    assert recomputed is not contradictory
    assert recomputed["inputs"] == ["x"]

    # Without backing source, recomputation still fails closed with is_control_flow_safe=False
    failed_closed = ensure_unit_scope_closure_mode(
        {"start": 2, "end": 3, "kind": "compound_block"},
        contradictory,
        False,
        strict_with_suppression=True,
    )
    assert failed_closed["is_control_flow_safe"] is False

    # Omitting is_control_flow_safe altogether also avoids unsafe short-circuit
    missing_flag: dict[str, Any] = {
        "inputs": ["x"],
        "outputs": ["y"],
        "control_flow_hazards": [terminal_hazard],
    }
    combined = analyze_unit_variable_scope(
        {"start": 2, "end": 3, "kind": "compound_block"},
        {"start": 2, "end": 3, "kind": "compound_block"},
        scope1=missing_flag,
        scope2=contradictory,
    )
    assert combined["is_control_flow_safe"] is False

    # Positive control: genuine terminal hazard scope (is_control_flow_safe=False) is preserved
    genuine = {**contradictory, "is_control_flow_safe": False}
    preserved = ensure_unit_scope_closure_mode(
        unit,
        genuine,
        False,
        strict_with_suppression=True,
        source_text=src,
    )
    assert preserved is genuine
    assert preserved["is_control_flow_safe"] is False


@pytest.mark.parametrize(
    ("unit", "err_pattern", "expected_oob"),
    [
        pytest.param(
            {"start": 1, "end": 1, "end_col": 0},
            "Zero-width column coordinates",
            True,
            id="explicit_1_1_omitted_start_col_zero_end_col",
        ),
        pytest.param(
            {"end_col": 0},
            "Zero-width column coordinates",
            True,
            id="omitted_lines_omitted_start_col_zero_end_col",
        ),
        pytest.param(
            {"start_col": 10, "end_col": 5},
            "Inverted column coordinates",
            True,
            id="omitted_lines_inverted_cols",
        ),
        pytest.param(
            {"end": 1, "start_col": 10, "end_col": 5},
            "Inverted column coordinates",
            True,
            id="omitted_start_end_1_inverted_cols",
        ),
        pytest.param(
            {"start": 5, "end_col": 0},
            "Zero-width column coordinates",
            True,
            id="omitted_end_omitted_start_col_zero_end_col",
        ),
        pytest.param(
            {"end": 2, "start_col": 10, "end_col": 5},
            None,
            False,
            id="control_omitted_start_end_2_multiline_cross_col_ok",
        ),
        pytest.param(
            {"start": 1, "end": 1, "end_col": 5},
            None,
            False,
            id="control_omitted_start_col_positive_end_col_ok",
        ),
        pytest.param(
            {"end_col": 5},
            None,
            False,
            id="control_omitted_lines_and_start_col_positive_end_col_ok",
        ),
    ],
)
def test_validate_unit_schema_omitted_lines_and_default_start_col_bounds(
    unit: dict[str, Any],
    err_pattern: str | None,
    expected_oob: bool,
) -> None:
    """Verifies validate_unit_schema aligns single-line column checks when start/end or start_col default."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    sample_lines = ["abcdefghijklmno\n", "abcdefghijklmno\n"]
    is_oob = _has_out_of_bounds_unit_cols(
        unit,
        sample_lines[: (unit.get("end") or 1)],
        is_placeholder_origin=False,
        explicit_col_sliced=False,
    )
    assert is_oob is expected_oob
    if err_pattern is not None:
        with pytest.raises(ValueError, match=err_pattern):
            validate_unit_schema(unit)
    else:
        validate_unit_schema(unit)


@pytest.mark.parametrize(
    ("unit", "is_placeholder_origin", "explicit_col_sliced", "expected_oob"),
    [
        pytest.param(
            {"start": 1, "end": 1, "start_col": -5, "end_col": 5},
            False,
            False,
            True,
            id="negative_start_col_minus_5",
        ),
        pytest.param(
            {"start": 1, "end": 1, "start_col": 0, "end_col": -2},
            False,
            False,
            True,
            id="negative_end_col_minus_2",
        ),
        pytest.param(
            {"start": 1, "end": 1, "start_col": -5, "end_col": 5},
            True,
            True,
            True,
            id="negative_start_col_even_on_placeholder_col_sliced",
        ),
        pytest.param(
            {"start": 1, "end": 1, "start_col": 0, "end_col": -2},
            False,
            True,
            True,
            id="negative_end_col_even_on_explicit_col_sliced",
        ),
        pytest.param(
            {"start": 1, "end": 1, "start_col": 0, "end_col": 5},
            False,
            False,
            False,
            id="control_valid_non_negative_cols",
        ),
    ],
)
def test_has_out_of_bounds_unit_cols_rejects_all_negative_columns(
    unit: dict[str, Any],
    is_placeholder_origin: bool,
    explicit_col_sliced: bool,
    expected_oob: bool,
) -> None:
    """Verifies _has_out_of_bounds_unit_cols rejects any negative start_col or end_col (< 0)."""
    from pydoppelgangerhunt.fixer.scope import _has_out_of_bounds_unit_cols

    sample_lines = ["abcdefghijklmno\n"]
    assert (
        _has_out_of_bounds_unit_cols(
            unit,
            sample_lines,
            is_placeholder_origin=is_placeholder_origin,
            explicit_col_sliced=explicit_col_sliced,
        )
        is expected_oob
    )


def test_patch_and_synthesis_preserve_hazard_skip_reasons_before_expr_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies blocking unit hazards run before comprehension/complex_expr pruning in patch and synthesis."""
    import logging
    from pydoppelgangerhunt.fixer import synthesis as synth_mod

    src = (
        "def fn1(items):\n"
        "    vals = [x + 1 for x in items]\n"
        "    return vals\n\n"
        "def fn2(items):\n"
        "    out = [y + 1 for y in items]\n"
        "    return out\n"
    )
    target_file = tmp_path / "expr_hazards.py"
    target_file.write_text(src, encoding="utf-8")

    # 1. Invalid columns on comprehension units report 'invalid unit columns'
    bad_col_u1 = {
        "name": "fn1:comp",
        "file": str(target_file),
        "start": 2,
        "end": 2,
        "start_col": 500,
        "end_col": 600,
        "kind": "comprehension",
    }
    bad_col_u2 = {
        "name": "fn2:comp",
        "file": str(target_file),
        "start": 6,
        "end": 6,
        "start_col": 500,
        "end_col": 600,
        "kind": "comprehension",
    }
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pydoppelgangerhunt.fixer.patch"):
        patch_oob = generate_refactoring_patch(
            [(1.0, bad_col_u1, bad_col_u2)],
            repo_root=str(tmp_path),
        )
    assert patch_oob == ""
    assert "1 invalid unit columns" in caplog.text

    # 2. Stale source_code on complex_expr units reports 'stale source code' instead of expr-pruning fallback
    expr_line = src.splitlines()[1]
    expr_start = expr_line.index("[x + 1")
    expr_end = expr_start + len("[x + 1 for x in items]")
    expr_line2 = src.splitlines()[5]
    expr_start2 = expr_line2.index("[y + 1")
    expr_end2 = expr_start2 + len("[y + 1 for y in items]")
    stale_u1 = {
        "name": "fn1:comp",
        "file": str(target_file),
        "start": 2,
        "end": 2,
        "start_col": expr_start,
        "end_col": expr_end,
        "kind": "complex_expr",
        "source_code": "[x + 999 for x in items]\n",
    }
    stale_u2 = {
        "name": "fn2:comp",
        "file": str(target_file),
        "start": 6,
        "end": 6,
        "start_col": expr_start2,
        "end_col": expr_end2,
        "kind": "complex_expr",
        "source_code": "[y + 1 for y in items]\n",
    }
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pydoppelgangerhunt.fixer.patch"):
        patch_stale = generate_refactoring_patch(
            [(1.0, stale_u1, stale_u2)],
            repo_root=str(tmp_path),
        )
    assert patch_stale == ""
    assert "1 stale source code" in caplog.text

    # 3. synthesize_shared_helper_code short-circuits on blocking unit hazards before _prune_pair_lhs_and_augassign_scopes
    prune_called = 0
    orig_prune = synth_mod._prune_pair_lhs_and_augassign_scopes

    def _spy_prune(*args: Any, **kwargs: Any) -> bool:
        nonlocal prune_called
        prune_called += 1
        return orig_prune(*args, **kwargs)

    monkeypatch.setattr(synth_mod, "_prune_pair_lhs_and_augassign_scopes", _spy_prune)
    u1_pre_lines = {
        **stale_u1,
        "source_lines": [expr_line + "\n"],
    }
    u2_pre_lines = {
        **stale_u2,
        "source_lines": [expr_line2 + "\n"],
    }
    assert (
        synthesize_shared_helper_code(u1_pre_lines, u2_pre_lines, repo_root=str(tmp_path))
        == ""
    )
    assert prune_called == 0


@pytest.mark.parametrize(
    ("unit_overrides", "expected_hazard"),
    [
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 500,
                "end_col": 600,
                "kind": "basic_block",
                "source_code": "    x = (a + 1)\n",
            },
            "invalid_unit_columns",
            id="tree_only_out_of_range_cols_with_source_code",
        ),
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 4,
                "end_col": 600,
                "kind": "comprehension",
                "source_code": "[i for i in items]",
            },
            "invalid_unit_columns",
            id="tree_only_end_col_past_matched_expr",
        ),
        pytest.param(
            {
                "start": 3,
                "end": 3,
                "start_col": 12,
                "end_col": 17,
                "kind": "basic_block",
                "source_code": "a = 1",
            },
            "invalid_unit_columns",
            id="tree_only_cols_pointing_to_second_stmt_on_semicolon_line",
        ),
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 500,
                "end_col": 600,
                "kind": "comprehension",
            },
            "invalid_unit_columns",
            id="tree_only_out_of_range_cols_without_source_code",
        ),
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 4,
                "end_col": 15,
                "kind": "basic_block",
                "source_code": "    z = 999\n",
            },
            "stale_source_code",
            id="control_mismatched_source_code_reports_stale_source_code",
        ),
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 4,
                "end_col": 15,
                "kind": "basic_block",
                "source_code": "    x = (a + 1)\n",
            },
            None,
            id="control_matching_stmt_with_exact_ast_cols_ok",
        ),
        pytest.param(
            {
                "start": 2,
                "end": 2,
                "start_col": 8,
                "end_col": 15,
                "kind": "complex_expr",
                "source_code": "(a + 1)",
            },
            None,
            id="control_parenthesized_expr_cols_including_outer_parens_ok",
        ),
    ],
)
def test_tree_only_placeholder_unit_validates_columns_against_ast(
    unit_overrides: dict[str, Any], expected_hazard: Any
) -> None:
    """Verifies tree-only placeholder units check start_col/end_col against matched AST bounds."""
    src = (
        "def fn(a, items):\n"
        "    x = (a + 1)\n"
        "    a = 1; b = 2\n"
        "    vals = [i for i in items]\n"
        "    return x, b, vals\n"
    )
    mod_tree = ast.parse(src)
    # Adjust line 4 vs line 2 for the comprehension case cleanly by pointing at line 4
    unit = dict(unit_overrides)
    if unit.get("kind") == "comprehension":
        unit["start"] = 4
        unit["end"] = 4
    sc = inspect_single_unit_scope(unit, tree=mod_tree)
    hazards = sc["control_flow_hazards"]
    if expected_hazard is None:
        assert "invalid_unit_columns" not in hazards
        assert "stale_source_code" not in hazards
    else:
        assert expected_hazard in hazards


@pytest.mark.parametrize(
    ("src", "lineno", "start_col", "end_col", "kind", "expect_invalid"),
    [
        pytest.param(
            "def fn(items):\n    [x for x in items]\n",
            2,
            0,
            4,
            "comprehension",
            True,
            id="leading_indentation_whitespace_slice_fails_closed",
        ),
        pytest.param(
            "def fn():\n    x = '€'\n",
            2,
            10,
            11,
            "complex_expr",
            True,
            id="intra_utf8_codepoint_empty_slice_fails_closed",
        ),
        pytest.param(
            "def fn(items):\n    [x for x in items]\n",
            2,
            4,
            22,
            "comprehension",
            False,
            id="control_valid_comprehension_slice_ok",
        ),
        pytest.param(
            "def fn():\n    \n    return 1\n",
            2,
            None,
            None,
            "basic_block",
            False,
            id="control_blank_source_line_without_cols_not_invalid_columns",
        ),
    ],
)
def test_whitespace_and_empty_utf8_column_slices_fail_closed(
    src: str,
    lineno: int,
    start_col: int | None,
    end_col: int | None,
    kind: str,
    expect_invalid: bool,
) -> None:
    """Verifies column slices yielding only whitespace or empty intra-codepoint spans fail closed."""
    clear_dataflow_cache()
    unit: dict[str, Any] = {
        "start": lineno,
        "end": lineno,
        "kind": kind,
        "source_text": src,
    }
    if start_col is not None:
        unit["start_col"] = start_col
    if end_col is not None:
        unit["end_col"] = end_col
    sc = inspect_single_unit_scope(unit)
    if expect_invalid:
        assert sc["is_control_flow_safe"] is False
        assert "invalid_unit_columns" in sc["control_flow_hazards"]
    else:
        assert "invalid_unit_columns" not in sc["control_flow_hazards"]


def test_kindless_colon_subroutine_placeholder_and_closure_mode_recompute(
    tmp_path: Path,
) -> None:
    """Verifies kindless colon-named units use is_subroutine_unit in placeholder and mode-recompute paths."""
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    clear_dataflow_cache()
    kindless_placeholder = {
        "name": "fn:block",
        "file": str(tmp_path / "missing_mod.py"),
        "start": 10,
        "end": 12,
    }
    sc_sub = inspect_single_unit_scope(kindless_placeholder)
    assert sc_sub["is_control_flow_safe"] is False
    assert set(sc_sub["control_flow_hazards"]) >= {
        "severed_closure_cell",
        "unavailable_source_placeholder",
    }

    whole_fn_placeholder = {**kindless_placeholder, "kind": "function"}
    sc_whole = inspect_single_unit_scope(whole_fn_placeholder)
    assert "unavailable_source_placeholder" in sc_whole["control_flow_hazards"]
    assert "severed_closure_cell" not in sc_whole["control_flow_hazards"]

    # Mode-recompute path: kindless colon unit with only snippet source_code cannot
    # recompute closure analysis when closure mode mismatches precomputed scope.
    pre_lenient_scope = {
        "scope_schema_version": 2,
        "inputs": ["x"],
        "outputs": ["y"],
        "skip_pre_unit_closures": True,
        "strict_with_suppression": False,
        "non_suppressing_ctx_managers": [],
        "is_control_flow_safe": True,
        "control_flow_hazards": [],
    }
    snippet_sub_unit = {
        "name": "fn:block",
        "file": str(tmp_path / "missing_snippet.py"),
        "start": 4,
        "end": 5,
        "source_code": "y = x + 1\n",
    }
    recomp_sub = ensure_unit_scope_closure_mode(
        snippet_sub_unit, pre_lenient_scope, skip_pre_unit_closures=False
    )
    assert recomp_sub["is_control_flow_safe"] is False
    assert "unavailable_source_for_closure_mode" in recomp_sub["control_flow_hazards"]

    snippet_whole_unit = {**snippet_sub_unit, "kind": "function", "source_code": "def fn(x):\n    return x + 1\n"}
    recomp_whole = ensure_unit_scope_closure_mode(
        snippet_whole_unit, pre_lenient_scope, skip_pre_unit_closures=False
    )
    assert "unavailable_source_for_closure_mode" not in recomp_whole["control_flow_hazards"]


def test_custom_sequence_and_abstract_set_declared_outputs() -> None:
    """Verifies live_write_only_cond and outputs support custom Sequence[str] and AbstractSet types."""
    from collections import UserList
    from collections.abc import Iterator

    class _OrderedSetView(AbstractSet[str]):
        def __init__(self, items: list[str]) -> None:
            self._items = tuple(items)

        def __contains__(self, item: object) -> bool:
            return item in self._items

        def __iter__(self) -> Iterator[str]:
            return iter(self._items)

        def __len__(self) -> int:
            return len(self._items)

    src = (
        "def run(flag, x):\n"
        "    if flag:\n"
        "        acc = x + 1\n"
        "    return acc\n"
    )
    clear_dataflow_cache()
    sc_seq = inspect_single_unit_scope(
        {
            "name": "run:block",
            "kind": "compound_block",
            "start": 2,
            "end": 3,
            "source_text": src,
            "precomputed_outputs": UserList(["acc"]),
        }
    )
    assert sc_seq["outputs"] == ["acc"]

    sc_single_set = inspect_single_unit_scope(
        {
            "name": "run:block",
            "kind": "compound_block",
            "start": 2,
            "end": 3,
            "source_text": src,
            "precomputed_outputs": _OrderedSetView(["acc"]),
        }
    )
    assert sc_single_set["outputs"] == ["acc"]

    sc_multi_set = inspect_single_unit_scope(
        {
            "name": "run:block",
            "kind": "compound_block",
            "start": 2,
            "end": 3,
            "source_text": src,
            "precomputed_outputs": _OrderedSetView(["acc", "other"]),
        }
    )
    assert sc_multi_set["outputs"] == []


def test_with_unit_source_text_and_suggestion_when_explicit_none_source_text(
    tmp_path: Path,
) -> None:
    """Verifies explicit 'source_text': None or '' still attaches preloaded source_text cleanly."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion
    from pydoppelgangerhunt.source_lines import with_unit_source_text

    full_src = (
        "def alpha(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b + 3\n\n"
        "def beta(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b + 3\n"
    )
    for empty_val in (None, "", "   "):
        enriched = with_unit_source_text({"start": 2, "end": 4, "source_text": empty_val}, full_src)
        assert enriched["source_text"] == full_src

    u1 = {
        "name": "alpha",
        "kind": "function",
        "file": str(tmp_path / "missing_disk.py"),
        "start": 1,
        "end": 4,
        "source_text": None,
        "source_lines": ["def alpha(x):\n", "    a = x + 1\n", "    b = a * 2\n", "    return b + 3\n"],
    }
    u2 = {
        "name": "beta",
        "kind": "function",
        "file": str(tmp_path / "missing_disk.py"),
        "start": 6,
        "end": 9,
        "source_text": "",
        "source_lines": ["def beta(x):\n", "    a = x + 1\n", "    b = a * 2\n", "    return b + 3\n"],
    }
    suggestion = synthesize_refactoring_suggestion(
        u1, u2, source_text1=full_src, source_text2=full_src
    )
    assert "def _shared_" in suggestion
    assert "Placeholder" not in suggestion


@pytest.mark.parametrize(
    ("mod_src", "marker", "span_lines", "unit_src_code", "expect_stale"),
    [
        pytest.param(
            "def f(x):\n    if x > 10:\n        out = 'high'\n    elif x > 0:\n        out = 'pos'\n    else:\n        out = 'neg'\n    return out\n",
            "    elif x > 0:",
            2,
            "elif x > 0:\n    out = 'pos'\n",
            False,
            id="elif_matching_with_else_is_valid",
        ),
        pytest.param(
            "def f(x):\n    if x > 10:\n        out = 'high'\n    elif x > 0:\n        out = 'pos'\n    return out\n",
            "    elif x > 0:",
            2,
            "elif x > 0:\n    out = 'pos'\n",
            False,
            id="elif_matching_without_else_is_valid",
        ),
        pytest.param(
            "def f(x):\n    if x > 10:\n        out = 'high'\n    elif x > 0: out = 'pos'\n    return out\n",
            "    elif x > 0: out = 'pos'",
            1,
            "elif x > 0: out = 'pos'\n",
            False,
            id="elif_one_line_matching_is_valid",
        ),
        pytest.param(
            "def f(x):\n    if x > 10:\n        out = 'high'\n    elif x > 0:\n        out = 'pos'\n    else:\n        out = 'neg'\n    return out\n",
            "    elif x > 0:",
            2,
            "elif x < 0:\n    out = 'pos'\n",
            True,
            id="elif_stale_condition_rejected",
        ),
        pytest.param(
            "def f(x):\n    if x > 0:\n        out = 'pos'\n    else:\n        out = 'neg'\n    return out\n",
            "    if x > 0:",
            2,
            "elif x > 0:\n    out = 'pos'\n",
            True,
            id="adversarial_elif_snippet_cannot_match_if_branch",
        ),
        pytest.param(
            "def f(x):\n    if x > 0:\n        out = 'pos'\n    else:\n        out = 'neg'\n    return out\n",
            "    else:",
            2,
            "else:\n    out = 'neg'\n",
            False,
            id="else_matching_branch_is_valid",
        ),
        pytest.param(
            "def f(x):\n    if x > 10:\n        out = 'high'\n    elif x > 0:\n        out = 'neg'\n    return out\n",
            "    elif x > 0:",
            2,
            "else:\n    out = 'neg'\n",
            True,
            id="adversarial_else_snippet_cannot_match_elif_branch",
        ),
        pytest.param(
            "def f():\n    try:\n        work()\n    except ValueError:\n        out = 1\n    finally:\n        cleanup()\n",
            "    except ValueError:",
            2,
            "except ValueError:\n    out = 1\n",
            False,
            id="except_matching_handler_is_valid",
        ),
        pytest.param(
            "def f():\n    try:\n        work()\n    except ValueError:\n        out = 1\n",
            "    except ValueError:",
            2,
            "except KeyError:\n    out = 1\n",
            True,
            id="except_stale_type_rejected",
        ),
        pytest.param(
            "def f():\n    try:\n        work()\n    finally:\n        cleanup()\n",
            "    finally:",
            2,
            "finally:\n    cleanup()\n",
            False,
            id="finally_matching_block_is_valid",
        ),
        pytest.param(
            "def f(x):\n    if x:\n        work()\n    else:\n        cleanup()\n",
            "    else:",
            2,
            "finally:\n    cleanup()\n",
            True,
            id="adversarial_finally_snippet_cannot_match_else_branch",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case [1, x] if x > 0:\n            out = x\n",
            "        case [1, x] if x > 0:",
            2,
            "case [1, x] if x > 0:\n    out = x\n",
            False,
            id="match_case_matching_pattern_and_guard_is_valid",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case [1, x] if x > 0:\n            out = x\n",
            "        case [1, x] if x > 0:",
            2,
            "case [2, x] if x > 0:\n    out = x\n",
            True,
            id="match_case_stale_pattern_rejected",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case [1, x] if x > 0:\n            out = x\n",
            "        case [1, x] if x > 0:",
            2,
            "case [1, x] if x < 0:\n    out = x\n",
            True,
            id="match_case_stale_guard_rejected",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case [1, x] if x > 0:\n            out = x\n",
            "        case [1, x] if x > 0:",
            2,
            "case [1, x]:\n    out = x\n",
            True,
            id="match_case_dropped_guard_rejected",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case 1: out = 'one'\n",
            "        case 1: out = 'one'",
            1,
            "case 1: out = 'one'\n",
            False,
            id="match_case_one_line_matching_is_valid",
        ),
        pytest.param(
            "def f(item):\n    match item:\n        case 1: out = 'one'\n",
            "        case 1: out = 'one'",
            1,
            "case 2: out = 'one'\n",
            True,
            id="match_case_one_line_stale_pattern_rejected",
        ),
    ],
)
def test_clause_header_source_matching_verifies_header_test_pattern_and_guard(
    mod_src: str,
    marker: str,
    span_lines: int,
    unit_src_code: str,
    expect_stale: bool,
) -> None:
    """Verifies clause-header snippet matching rejects stale elif conditions, except types, else/finally mismatches, and match-case patterns/guards."""
    lines = mod_src.splitlines()
    start_line = next(
        idx for idx, ln in enumerate(lines, start=1) if ln == marker
    )
    end_line = start_line + span_lines - 1
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": start_line,
            "end": end_line,
            "kind": "clause",
            "source_text": mod_src,
            "source_code": unit_src_code,
        },
        tree=ast.parse(mod_src),
    )
    assert ("stale_source_code" in sc["control_flow_hazards"]) is expect_stale, sc


@pytest.mark.parametrize(
    ("file_target", "expected_present", "expected_snippet"),
    [
        pytest.param("analysis.ipynb#cell_1", True, "def outer_one(items, scale):", id="anchored_cell_1_fragments"),
        pytest.param("analysis.ipynb#cell1", True, "def outer_one(items, scale):", id="anchored_cell1_no_underscore"),
        pytest.param("analysis.ipynb#cell_2", True, "def outer_two(values, factor):", id="anchored_cell_2_string_source"),
        pytest.param("analysis.ipynb", False, None, id="neg_unanchored_notebook_fails_closed"),
        pytest.param("report.ipynb#cellular.ipynb", False, None, id="neg_literal_hash_ipynb_fails_closed"),
        pytest.param("analysis.ipynb#cell_0", False, None, id="neg_zero_based_anchor_fails_closed"),
        pytest.param("analysis.ipynb#bad_anchor", False, None, id="neg_malformed_anchor_fails_closed"),
        pytest.param("analysis.ipynb#cell_3", False, None, id="neg_markdown_cell_fails_closed"),
        pytest.param("analysis.ipynb#cell_4", False, None, id="neg_empty_code_cell_fails_closed"),
        pytest.param("analysis.ipynb#cell_99", False, None, id="neg_out_of_range_cell_fails_closed"),
    ],
)
def test_disk_notebook_cell_extraction_and_fail_closed_anchors(
    tmp_path: Path,
    file_target: str,
    expected_present: bool,
    expected_snippet: str | None,
) -> None:
    """Verifies _load_unit_file_text and resolve_cached_unit_source_and_tree extract only code-cell Python source for anchored disk notebooks."""
    import json

    from pydoppelgangerhunt.fixer.source import _load_unit_file_text
    from pydoppelgangerhunt.reporters import resolve_cached_unit_source_and_tree

    nb_file = tmp_path / "analysis.ipynb"
    nb_payload = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "def outer_one(items, scale):\n    total = 0\n",
                    "    for x in items:\n        total += x * scale\n",
                    "    return total\n",
                ],
            },
            {
                "cell_type": "code",
                "source": (
                    "def outer_two(values, factor):\n"
                    "    acc = 0\n"
                    "    for y in values:\n"
                    "        acc += y * factor\n"
                    "    return acc\n"
                ),
            },
            {"cell_type": "markdown", "source": ["# Heading\n"]},
            {"cell_type": "code", "source": ""},
        ],
    }
    nb_file.write_text(json.dumps(nb_payload), encoding="utf-8")

    literal_hash_nb = tmp_path / "report.ipynb#cellular.ipynb"
    literal_hash_nb.write_text(json.dumps(nb_payload), encoding="utf-8")

    unit: dict[str, Any] = {
        "file": file_target,
        "name": "outer:for",
        "start": 3,
        "end": 4,
    }
    clear_dataflow_cache()
    loaded = _load_unit_file_text(unit, repo_root=str(tmp_path))
    cached_src, cached_tree = resolve_cached_unit_source_and_tree(
        unit, repo_root=str(tmp_path)
    )
    if expected_present:
        assert isinstance(loaded, str)
        assert expected_snippet is not None and expected_snippet in loaded
        assert '"nbformat"' not in loaded and '"cells"' not in loaded
        assert cached_src == loaded
        assert isinstance(cached_tree, ast.Module)
    else:
        assert loaded is None
        assert cached_src is None
        assert cached_tree is None


def test_disk_notebook_subroutine_scope_stale_check_synthesis_and_patch_safety(
    tmp_path: Path,
) -> None:
    """Verifies disk .ipynb#cell_N units preserve enclosing AST context for scope, stale check, and helper synthesis while blocking raw-text patches."""
    import json

    from pydoppelgangerhunt.fixer.dataflow import collect_downstream_read_names
    from pydoppelgangerhunt.fixer.source import _load_unit_file_text

    nb_file = tmp_path / "notebook.ipynb"
    nb_payload = {
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "source": [
                    "def outer_a(items, scale):\n",
                    "    total = 0\n",
                    "    for x in items:\n",
                    "        total += x * scale\n",
                    "    return total\n",
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    "def outer_b(items, scale):\n",
                    "    total = 0\n",
                    "    for x in items:\n",
                    "        total += x * scale\n",
                    "    return total\n",
                ],
            },
            {
                "cell_type": "code",
                "source": [
                    "def enclosing_cell():\n",
                    "    count = 0\n",
                    "    def bump():\n",
                    "        nonlocal count\n",
                    "        count += 1\n",
                    "    for _ in range(2):\n",
                    "        count += 1\n",
                    "        bump()\n",
                    "    return count\n",
                ],
            },
        ],
    }
    nb_file.write_text(json.dumps(nb_payload), encoding="utf-8")

    sub_a = {
        "file": "notebook.ipynb#cell_1",
        "name": "outer_a:for",
        "kind": "compound_block",
        "start": 3,
        "end": 4,
    }
    sub_b = {
        "file": "notebook.ipynb#cell_2",
        "name": "outer_b:for",
        "kind": "compound_block",
        "start": 3,
        "end": 4,
    }
    clear_dataflow_cache()
    loaded_cell1 = _load_unit_file_text(sub_a, repo_root=str(tmp_path))
    assert loaded_cell1 is not None
    assert collect_downstream_read_names(loaded_cell1, sub_a) == {"total"}

    sc_a = inspect_single_unit_scope(sub_a, repo_root=str(tmp_path))
    sc_b = inspect_single_unit_scope(sub_b, repo_root=str(tmp_path))
    assert sc_a["is_control_flow_safe"] is True
    assert "severed_closure_cell" not in sc_a["control_flow_hazards"]
    assert sc_a["outputs"] == ["x", "total"]
    assert resolve_clone_pair_outputs(
        sub_a, sub_b, sc_a, sc_b, repo_root=str(tmp_path)
    ) == (["x", "total"], ["x", "total"])

    # Adversarial controls on anchored disk cells:
    # 1. Stale inline source_code must still be caught against the disk notebook cell AST
    sc_stale = inspect_single_unit_scope(
        {**sub_a, "source_code": "for y in other:\n    acc += y\n"},
        repo_root=str(tmp_path),
    )
    assert sc_stale["is_control_flow_safe"] is False
    assert "stale_source_code" in sc_stale["control_flow_hazards"]

    # 2. Enclosing closure cell sharing variable with loop in disk cell must flag real closure hazard
    closure_loop_unit = {
        "file": "notebook.ipynb#cell_3",
        "name": "enclosing_cell:for",
        "kind": "compound_block",
        "start": 6,
        "end": 8,
    }
    sc_closure = inspect_single_unit_scope(closure_loop_unit, repo_root=str(tmp_path))
    assert sc_closure["is_control_flow_safe"] is False
    assert "severed_closure_cell" in sc_closure["control_flow_hazards"]
    assert sc_closure["closure_hazard_reasons"] == ["cell_severing"]

    # Helper suggestion succeeds from disk cells, while patch generation still refuses .ipynb files
    helper = synthesize_shared_helper_code(sub_a, sub_b, repo_root=str(tmp_path))
    assert "for x in items:" in helper
    ast.parse(helper)
    assert generate_refactoring_patch([(1.0, sub_a, sub_b)], repo_root=str(tmp_path)) == ""


@pytest.mark.parametrize(
    ("case_id", "fn1_body", "fn2_body", "u_extra1", "u_extra2", "expected_ret", "expected_tail"),
    [
        pytest.param(
            "comprehension_source_code_without_cols",
            "def f1(items, tail):\n    res = [x * 2 for x in items] + tail\n    return res\n",
            "def f2(items, tail):\n    out = [x * 2 for x in items] + tail\n    return out\n",
            {"kind": "comprehension", "source_code": "[x * 2 for x in items]"},
            {"kind": "comprehension", "source_code": "[x * 2 for x in items]"},
            "return [x * 2 for x in items]",
            "+ tail",
            id="comprehension_source_code_without_cols",
        ),
        pytest.param(
            "complex_expr_attr_call_without_cols",
            "def f1(obj, bar):\n    res = obj.foo(1) + bar(2)\n    return res\n",
            "def f2(obj, bar):\n    out = obj.foo(1) + bar(2)\n    return out\n",
            {"kind": "complex_expr", "source_code": "obj.foo(1)"},
            {"kind": "complex_expr", "source_code": "obj.foo(1)"},
            "return obj.foo(1)",
            "+ bar(2)",
            id="complex_expr_attr_call_without_cols",
        ),
        pytest.param(
            "partial_start_col_only",
            "def f1(items, tail):\n    res = [x * 2 for x in items] + tail\n    return res\n",
            "def f2(items, tail):\n    out = [x * 2 for x in items] + tail\n    return out\n",
            {"kind": "comprehension", "start_col": 10},
            {"kind": "comprehension", "start_col": 10},
            "return [x * 2 for x in items]",
            "+ tail",
            id="partial_start_col_only",
        ),
        pytest.param(
            "partial_col_offset_alias_only",
            "def f1(items, tail):\n    res = [x * 2 for x in items] + tail\n    return res\n",
            "def f2(items, tail):\n    out = [x * 2 for x in items] + tail\n    return out\n",
            {"kind": "comprehension", "col_offset": 10},
            {"kind": "comprehension", "col_offset": 10},
            "return [x * 2 for x in items]",
            "+ tail",
            id="partial_col_offset_alias_only",
        ),
        pytest.param(
            "multiline_comprehension_source_code_without_cols",
            "def f1(items, tail):\n    res = [\n        x * 2\n        for x in items\n    ] + tail\n    return res\n",
            "def f2(items, tail):\n    out = [\n        x * 2\n        for x in items\n    ] + tail\n    return out\n",
            {
                "kind": "comprehension",
                "end": 5,
                "source_code": "[\n    x * 2\n    for x in items\n]",
            },
            {
                "kind": "comprehension",
                "end": 5,
                "source_code": "[\n    x * 2\n    for x in items\n]",
            },
            "for x in items",
            "+ tail",
            id="multiline_comprehension_source_code_without_cols",
        ),
    ],
)
def test_subline_expression_inferred_columns_propagate_across_scope_synthesis_and_patch(
    tmp_path: Path,
    case_id: str,
    fn1_body: str,
    fn2_body: str,
    u_extra1: dict[str, Any],
    u_extra2: dict[str, Any],
    expected_ret: str,
    expected_tail: str,
) -> None:
    """Verifies inferred sub-line expression columns match across scope analysis, helper synthesis, suggestion, and call-site patch replacement."""
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    f1 = tmp_path / "expr_a.py"
    f2 = tmp_path / "expr_b.py"
    f1.write_text(fn1_body, encoding="utf-8")
    f2.write_text(fn2_body, encoding="utf-8")

    u1: dict[str, Any] = {"file": str(f1), "name": "f1", "start": 2, "end": 2, **u_extra1}
    u2: dict[str, Any] = {"file": str(f2), "name": "f2", "start": 2, "end": 2, **u_extra2}

    clear_dataflow_cache()
    sc1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    assert sc1["is_control_flow_safe"] is True, case_id
    assert "tail" not in sc1["inputs"] and "bar" not in sc1["inputs"], case_id

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert expected_ret in helper, case_id
    assert expected_tail not in helper, case_id
    ast.parse(helper)

    suggestion = synthesize_refactoring_suggestion(u1, u2, repo_root=str(tmp_path))
    assert expected_ret in suggestion and expected_tail not in suggestion, case_id

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert "+    res = _shared_f1_f2(" in patch, case_id
    assert "+    out = _shared_f1_f2(" in patch, case_id
    for line in patch.splitlines():
        if line.startswith("+    res = _shared_") or line.startswith("+    out = _shared_"):
            assert expected_tail in line, line


@pytest.mark.parametrize(
    ("line_src", "snippet", "expect_cols", "expect_stale"),
    [
        pytest.param(
            "def f(obj, bar):\n    res = obj.foo(1) + bar(2)\n    return res\n",
            "foo(1)",
            (None, None),
            True,
            id="neg_dot_prefix_before_call_masked_without_attr_boundary",
        ),
        pytest.param(
            "def f(obj, bar):\n    res = obj.foo(1) + bar(2)  # keep note\n    return res\n",
            "foo(1)",
            (None, None),
            True,
            id="neg_dot_prefix_with_inline_comment_still_stale",
        ),
        pytest.param(
            "def f(a, b):\n    res = a + b.attr\n    return res\n",
            "a + b",
            (None, None),
            True,
            id="neg_dot_suffix_after_ident_rejected",
        ),
        pytest.param(
            "def f(obj, bar):\n    res = obj.foo(1) + bar(2)  # keep note\n    return res\n",
            "obj.foo(1)",
            (10, 20),
            False,
            id="pos_full_attr_call_inferred_with_note",
        ),
    ],
)
def test_unit_cols_attr_dot_boundary_rejects_suffix_and_flags_stale_source(
    tmp_path: Path,
    line_src: str,
    snippet: str,
    expect_cols: tuple[int | None, int | None],
    expect_stale: bool,
) -> None:
    """Verifies _unit_cols and stale_source_code checks enforce attribute dot boundaries across raw and comment-normalized paths."""
    from pydoppelgangerhunt.fixer.scope import _unit_cols
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    clear_dataflow_cache()
    unit_line = line_src.splitlines(keepends=True)[1]
    assert (
        _unit_cols({"kind": "complex_expr", "source_code": snippet}, [unit_line])
        == expect_cols
    )
    mod_a = tmp_path / "mod_a.py"
    mod_b = tmp_path / "mod_b.py"
    mod_a.write_text(line_src, encoding="utf-8")
    mod_b.write_text(line_src, encoding="utf-8")
    unit: dict[str, Any] = {
        "file": str(mod_a),
        "name": "f:expr",
        "start": 2,
        "end": 2,
        "kind": "complex_expr",
        "source_code": snippet,
    }
    unit_b = {**unit, "file": str(mod_b)}
    sc = inspect_single_unit_scope(unit, repo_root=str(tmp_path), tree=ast.parse(line_src))
    assert ("stale_source_code" in sc["control_flow_hazards"]) is expect_stale, sc
    assert sc["is_control_flow_safe"] is (not expect_stale), sc

    helper = synthesize_shared_helper_code(unit, unit_b, repo_root=str(tmp_path))
    suggestion = synthesize_refactoring_suggestion(unit, unit_b, repo_root=str(tmp_path))
    patch = generate_refactoring_patch(
        [(1.0, unit, unit_b)], repo_root=str(tmp_path), replace_clones=True
    )
    if expect_stale:
        assert helper == ""
        assert patch == ""
        assert "stale source code" in suggestion
    else:
        assert f"return {snippet}" in helper
        assert f"return {snippet}" in patch
        assert "+ bar(2)" in patch
        assert "stale source code" not in suggestion


@pytest.mark.parametrize(
    ("min_py", "mod_src", "target_fn_name", "expect_safe"),
    [
        pytest.param(
            (3, 9),
            (
                "import contextlib\n"
                "def target(contextlib, step):\n"
                "    with contextlib.nullcontext():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "target",
            False,
            id="param_shadows_contextlib_in_whole_function",
        ),
        pytest.param(
            (3, 9),
            (
                "def target(open, step):\n"
                "    with open('path'):\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "target",
            False,
            id="param_shadows_builtin_open_in_whole_function",
        ),
        pytest.param(
            (3, 9),
            (
                "from contextlib import nullcontext\n"
                "def target(nullcontext, step):\n"
                "    with nullcontext():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "target",
            False,
            id="param_shadows_imported_nullcontext_in_whole_function",
        ),
        pytest.param(
            (3, 9),
            (
                "import contextlib\n"
                "def outer(step):\n"
                "    cm = contextlib.nullcontext\n"
                "    def target(cm):\n"
                "        with cm():\n"
                "            v = step()\n"
                "        cb = lambda: v\n"
                "        return cb()\n"
                "    return target(cm)\n"
            ),
            "target",
            False,
            id="nested_whole_function_param_shadows_enclosing_cm",
        ),
        pytest.param(
            (3, 9),
            (
                "import contextlib\n"
                "def outer(step):\n"
                "    cm = contextlib.nullcontext\n"
                "    def target():\n"
                "        with cm():\n"
                "            v = step()\n"
                "        cb = lambda: v\n"
                "        return cb()\n"
                "    return target()\n"
            ),
            "target",
            True,
            id="unshadowed_nested_whole_function_inherits_enclosing_nullcontext",
        ),
        pytest.param(
            (3, 12),
            (
                "import contextlib\n"
                "def target[contextlib](step):\n"
                "    with contextlib.nullcontext():\n"
                "        v = step()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            "target",
            False,
            id="pep695_type_param_shadows_contextlib_in_whole_function",
        ),
    ],
)
def test_whole_function_unit_resolves_fn_seed_and_clears_params_and_type_params(
    tmp_path: Path,
    min_py: tuple[int, int],
    mod_src: str,
    target_fn_name: str,
    expect_safe: bool,
) -> None:
    """Verifies whole-function units seed suppress provenance on their own node and clear params/type_params."""
    import sys

    if sys.version_info < min_py:
        pytest.skip(f"Requires Python {min_py[0]}.{min_py[1]}+")
    clear_dataflow_cache()
    mod_path = tmp_path / "whole_fn_mod.py"
    mod_path.write_text(mod_src, encoding="utf-8")
    tree = ast.parse(mod_src)
    fn_node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == target_fn_name
    )
    fn_lines = mod_src.splitlines(keepends=True)[fn_node.lineno - 1 : fn_node.end_lineno]
    sc = inspect_single_unit_scope(
        {
            "file": str(mod_path),
            "name": target_fn_name,
            "kind": "function",
            "start": fn_node.lineno,
            "end": fn_node.end_lineno,
            "source_code": "".join(fn_lines),
            "closure_strictness": "pedantic",
        },
        repo_root=str(tmp_path),
        tree=tree,
    )
    assert ("v" in sc["definite_stores"]) is expect_safe, sc


@pytest.mark.parametrize(
    ("body_src", "expect_violation"),
    [
        pytest.param(
            "    a = [x := 1 for _ in (1,)]\n    global x\n",
            True,
            id="listcomp_walrus_before_global",
        ),
        pytest.param(
            "    a = [[x := 1 for _ in (1,)] for _ in (1,)]\n    global x\n",
            True,
            id="nested_listcomp_walrus_before_global",
        ),
        pytest.param(
            "    a = (x := 1 for _ in (1,))\n    nonlocal x\n",
            True,
            id="genexpr_walrus_before_nonlocal",
        ),
        pytest.param(
            "    a = [(lambda y=(x := 1): y) for _ in (1,)]\n    global x\n",
            True,
            id="lambda_default_walrus_inside_comp_before_global",
        ),
        pytest.param(
            "    a = [x for _ in (1,)]\n    global x\n",
            False,
            id="control_comp_plain_read_before_global_is_valid",
        ),
        pytest.param(
            "    a = [(lambda: (x := 1)) for _ in (1,)]\n    global x\n",
            False,
            id="control_lambda_body_walrus_inside_comp_is_lambda_local",
        ),
        pytest.param(
            "    a = [x for x in (1,)]\n    global x\n",
            False,
            id="control_comp_loop_target_does_not_bind_enclosing_scope",
        ),
    ],
)
def test_scope_has_use_before_lexical_declaration_detects_comp_walrus_bindings(
    body_src: str,
    expect_violation: bool,
) -> None:
    """Verifies AST-only use-before-declaration check catches PEP 572 walruses inside comprehensions."""
    from pydoppelgangerhunt.fixer._df_span_index import (
        _scope_has_use_before_lexical_declaration,
    )

    fn_node = ast.parse(f"def outer():\n{body_src}").body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    assert (
        _scope_has_use_before_lexical_declaration(fn_node, frozenset({"x"}))
        is expect_violation
    )


@pytest.mark.parametrize(
    ("src", "expect_invalid"),
    [
        pytest.param(
            "def f():\n    x = 1 + \\\n        2\n    return x\n",
            False,
            id="lf_backslash_line_continuation",
        ),
        pytest.param(
            "def f():\r\n    x = 1 + \\\r\n        2\r\n    return x\r\n",
            False,
            id="crlf_backslash_line_continuation",
        ),
        pytest.param(
            "def f():\n    s = '''line1\nline2'''\n    return s\n",
            False,
            id="multiline_triple_quoted_string_keepends",
        ),
        pytest.param(
            "def f():\n    x = 1 + \\\n        2\n    global x\n",
            True,
            id="control_invalid_global_after_backslash_continuation",
        ),
    ],
)
def test_compute_scope_invalid_lexical_declarations_handles_keepends_lines(
    src: str,
    expect_invalid: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies symtable fast-path strips keepends newlines cleanly so valid line continuations do not fall back to AST walking."""
    from pydoppelgangerhunt.fixer import _df_span_index as span_mod

    tree = ast.parse(src)
    fallback_calls = 0
    orig_walk = span_mod._walk_invalid_lexical_scope_declarations

    def _spy_walk(*args: Any, **kwargs: Any) -> bool:
        nonlocal fallback_calls
        fallback_calls += 1
        return orig_walk(*args, **kwargs)

    monkeypatch.setattr(span_mod, "_walk_invalid_lexical_scope_declarations", _spy_walk)
    result = span_mod._compute_scope_invalid_lexical_declarations(
        tree,
        is_module_rooted=True,
        source_lines=src.splitlines(keepends=True),
    )
    assert result is expect_invalid
    if not expect_invalid:
        assert fallback_calls == 0


@pytest.mark.parametrize(
    ("src", "unit_code", "start_marker", "end_marker", "expect_match"),
    [
        pytest.param(
            (
                "def f(flag, x):\n"
                "    if flag:\n"
                "        y = x + 1\n"
                "    else:\n"
                "        # fallback comment\n"
                "\n"
                "        y = x + 2\n"
                "    return y\n"
            ),
            "    else:\n        # fallback comment\n\n        y = x + 2\n",
            "    else:",
            "        y = x + 2",
            True,
            id="if_else_clause_with_comment_and_blank_line",
        ),
        pytest.param(
            (
                "def f(a, b, x):\n"
                "    if a:\n"
                "        y = x + 1\n"
                "    elif b:\n"
                "        # elif comment\n"
                "        y = x + 2\n"
                "    return y\n"
            ),
            "    elif b:\n        # elif comment\n        y = x + 2\n",
            "    elif b:",
            "        y = x + 2",
            True,
            id="elif_clause_with_comment_before_body",
        ),
        pytest.param(
            (
                "def f(op, x):\n"
                "    try:\n"
                "        y = op(x)\n"
                "    finally:\n"
                "        # cleanup comment\n"
                "\n"
                "        z = x + 1\n"
                "    return y, z\n"
            ),
            "    finally:\n        # cleanup comment\n\n        z = x + 1\n",
            "    finally:",
            "        z = x + 1",
            True,
            id="try_finally_clause_with_comment_and_blank_line",
        ),
        pytest.param(
            (
                "def f(op, x):\n"
                "    try:\n"
                "        y = op(x)\n"
                "    except ValueError:\n"
                "        y = 0\n"
                "    else:\n"
                "        # ok comment\n"
                "        y = y + 1\n"
                "    return y\n"
            ),
            "    else:\n        # ok comment\n        y = y + 1\n",
            "    else:",
            "        y = y + 1",
            True,
            id="try_else_clause_with_comment_before_body",
        ),
        pytest.param(
            (
                "def f(flag, x):\n"
                "    if flag:\n"
                "        y = x + 1\n"
                "    else:\n"
                "        # fallback comment\n"
                "        y = x + 2\n"
                "    return y\n"
            ),
            "    else:\n        # fallback comment\n        y = x + 2\n",
            "        y = x + 1",
            "        y = x + 2",
            False,
            id="control_start_inside_preceding_clause_rejected",
        ),
        pytest.param(
            (
                "def f(flag, x):\n"
                "    if flag:\n"
                "        y = x + 1\n"
                "    else:\n"
                "        # fallback comment\n"
                "        y = x + 2\n"
                "    return y\n"
            ),
            "        y = x + 2\n",
            "        # fallback comment",
            "        y = x + 2",
            False,
            id="control_non_clause_snippet_shifted_onto_comment_rejected",
        ),
    ],
)
def test_tree_only_clause_header_with_leading_comment_or_blank_line_matches_tree(
    src: str,
    unit_code: str,
    start_marker: str,
    end_marker: str,
    expect_match: bool,
) -> None:
    """Verifies tree-only clause verification allows comments/blank lines after clause headers while rejecting misaligned spans."""
    from pydoppelgangerhunt.fixer.scope import _unit_source_code_matches_tree

    lines = src.splitlines()
    u_start = next(i + 1 for i, ln in enumerate(lines) if ln == start_marker)
    u_end = next(i + 1 for i, ln in enumerate(lines) if ln == end_marker)
    tree = ast.parse(src)
    unit = {
        "start": u_start,
        "end": u_end,
        "kind": "compound_block",
        "source_code": unit_code,
    }
    assert _unit_source_code_matches_tree(unit, unit_code, tree) is expect_match
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(unit, tree=tree)
    assert ("stale_source_code" not in sc["control_flow_hazards"]) is expect_match


def test_tree_only_match_case_clause_header_with_leading_comment_matches_tree() -> None:
    """Verifies tree-only Python 3.10+ match-case header matching allows comments between case header and body."""
    import sys
    from pydoppelgangerhunt.fixer.scope import _unit_source_code_matches_tree

    if sys.version_info < (3, 10):
        pytest.skip("match/case requires Python 3.10+")
    src = (
        "def f(val, x):\n"
        "    match val:\n"
        "        case 1:\n"
        "            # case comment\n"
        "\n"
        "            y = x + 1\n"
        "        case _:\n"
        "            y = x + 2\n"
        "    return y\n"
    )
    unit_code = "        case 1:\n            # case comment\n\n            y = x + 1\n"
    lines = src.splitlines()
    u_start = next(i + 1 for i, ln in enumerate(lines) if ln == "        case 1:")
    u_end = next(i + 1 for i, ln in enumerate(lines) if ln == "            y = x + 1")
    tree = ast.parse(src)
    unit = {
        "start": u_start,
        "end": u_end,
        "kind": "compound_block",
        "source_code": unit_code,
    }
    assert _unit_source_code_matches_tree(unit, unit_code, tree) is True
    bad_unit = {**unit, "start": u_start - 1}
    assert _unit_source_code_matches_tree(bad_unit, unit_code, tree) is False


@pytest.mark.parametrize(
    ("has_source", "expected_phrase", "forbidden_phrase"),
    [
        pytest.param(
            False,
            "failing closed because backing source is unavailable",
            "recomputing from source",
            id="missing_source_logs_fail_closed",
        ),
        pytest.param(
            True,
            "recomputing from source",
            "failing closed",
            id="available_source_logs_recompute",
        ),
    ],
)
def test_ensure_unit_scope_closure_mode_logs_fail_closed_vs_recompute_accurately(
    has_source: bool,
    expected_phrase: str,
    forbidden_phrase: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies discarded precomputed scopes log whether they fail closed or recompute from source."""
    import logging
    from pydoppelgangerhunt.config import clear_closure_warning_cache
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    src = "def f(x):\n    y = x + 1\n    return y\n"
    unit: dict[str, Any] = {
        "name": "f:blk",
        "start": 2,
        "end": 2,
        "kind": "basic_block",
    }
    if has_source:
        unit["source_text"] = src
    legacy_scope = {"inputs": ["stale"], "outputs": [], "is_control_flow_safe": True}

    clear_closure_warning_cache()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        resolved = ensure_unit_scope_closure_mode(
            unit,
            legacy_scope,
            skip_pre_unit_closures=False,
        )
    warnings = [
        rec.message
        for rec in caplog.records
        if "Discarding caller-supplied precomputed scope" in rec.message
    ]
    assert len(warnings) == 1
    assert expected_phrase in warnings[0]
    assert forbidden_phrase not in warnings[0]
    if has_source:
        assert resolved["is_control_flow_safe"] is True
        assert resolved["inputs"] == ["x"]
    else:
        assert resolved["is_control_flow_safe"] is False
        assert "unavailable_source_for_closure_mode" in resolved["control_flow_hazards"]


@pytest.mark.parametrize(
    ("missing_u1", "missing_u2", "is_subroutine", "pass_source_text2", "expect_emitted"),
    [
        pytest.param(
            True,
            False,
            False,
            False,
            False,
            id="missing_u1_rejected_even_with_stamped_v2_scope1",
        ),
        pytest.param(
            False,
            True,
            True,
            False,
            False,
            id="missing_u2_subroutine_rejected_even_with_stamped_v2_scope2",
        ),
        pytest.param(
            False,
            True,
            False,
            True,
            False,
            id="missing_u2_with_explicit_source_text2_rejected_even_with_stamped_v2_scope2",
        ),
        pytest.param(
            False,
            True,
            False,
            False,
            True,
            id="control_whole_fn_preview_allows_missing_u2_with_valid_u1",
        ),
    ],
)
def test_synthesize_shared_helper_rejects_placeholder_origin_even_with_stamped_v2_scope(
    tmp_path: Path,
    missing_u1: bool,
    missing_u2: bool,
    is_subroutine: bool,
    pass_source_text2: bool,
    expect_emitted: bool,
) -> None:
    """Verifies synthesize_shared_helper_code rejects placeholder origins even when callers pass stamped v2 scopes."""
    from pydoppelgangerhunt.fixer import synthesize_shared_helper_code
    from pydoppelgangerhunt.fixer.scope import SCOPE_SCHEMA_VERSION

    fn_src = "def compute(a, b):\n    res = a + b\n    return res\n"
    real_file = tmp_path / "real_mod.py"
    real_file.write_text(fn_src, encoding="utf-8")
    missing_file = tmp_path / "does_not_exist.py"

    kind = "basic_block" if is_subroutine else "function"
    start_ln, end_ln = (2, 2) if is_subroutine else (1, 3)
    u1: dict[str, Any] = {
        "file": str(missing_file if missing_u1 else real_file),
        "name": "compute",
        "kind": kind,
        "start": start_ln,
        "end": end_ln,
    }
    u2: dict[str, Any] = {
        "file": str(missing_file if missing_u2 else real_file),
        "name": "compute",
        "kind": kind,
        "start": start_ln,
        "end": end_ln,
    }
    real_unit: dict[str, Any] = {
        "file": str(real_file),
        "name": "compute",
        "kind": kind,
        "start": start_ln,
        "end": end_ln,
    }
    stamped_v2_scope = inspect_single_unit_scope(
        real_unit,
        repo_root=str(tmp_path),
        closure_strictness="strict",
    )
    assert stamped_v2_scope["scope_schema_version"] == SCOPE_SCHEMA_VERSION
    assert stamped_v2_scope["is_control_flow_safe"] is True
    assert "unavailable_source_placeholder" not in stamped_v2_scope["control_flow_hazards"]
    res = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        closure_strictness="strict",
        source_text2="   \n" if pass_source_text2 else None,
        scope1=dict(stamped_v2_scope),
        scope2=dict(stamped_v2_scope),
    )
    if expect_emitted:
        assert "def _shared_compute" in res
        assert "# Source for" not in res
    else:
        assert res == ""


@pytest.mark.parametrize(
    ("outer_prefix", "setup_sig", "setup_body", "expect_safe"),
    [
        pytest.param(
            "    def setattr(obj, k, v):\n        obj.callbacks.append(v)\n",
            "def setup(self):",
            "            setattr(self, 'other', lambda: total)\n",
            False,
            id="enclosing_fn_shadowed_setattr_triggers_bare_escape",
        ),
        pytest.param(
            "",
            "def setup(self):",
            "            setattr = custom_fn\n            setattr(self, 'other', lambda: total)\n",
            False,
            id="method_local_shadowed_setattr_triggers_bare_escape",
        ),
        pytest.param(
            "",
            "def setup(self, setattr):",
            "            setattr(self, 'other', lambda: total)\n",
            False,
            id="method_param_shadowed_setattr_triggers_bare_escape",
        ),
        pytest.param(
            "",
            "def setup(self):",
            "            setattr(self, 'other', lambda: total)\n",
            True,
            id="control_unshadowed_builtin_setattr_on_unrelated_attr_stays_safe",
        ),
    ],
)
def test_local_class_shadowed_setattr_on_receiver_triggers_bare_escape(
    outer_prefix: str,
    setup_sig: str,
    setup_body: str,
    expect_safe: bool,
) -> None:
    """Verifies shadowed setattr/delattr on self does not suppress bare_escape in receiver closure linking."""
    src = (
        "def outer(items, custom_fn):\n"
        "    total = 0\n"
        f"{outer_prefix}"
        "    class Worker:\n"
        "        def __init__(self):\n"
        "            self.callbacks = []\n"
        f"        {setup_sig}\n"
        f"{setup_body}"
        "        def run(self):\n"
        "            nonlocal total\n"
        "            out = []\n"
        "            for x in items:\n"
        "                total += x\n"
        "                out.append(self.callbacks[0]())\n"
        "            return out\n"
        "    return Worker(), total\n"
    )
    lines = src.splitlines()
    start_idx = next(i + 1 for i, ln in enumerate(lines) if "out = []" in ln)
    end_idx = next(i + 1 for i, ln in enumerate(lines) if "return out" in ln) - 1
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": start_idx,
            "end": end_idx,
            "kind": "block",
            "source_text": src,
            "source_lines": src.splitlines(keepends=True),
        },
        tree=ast.parse(src),
    )
    assert sc["is_control_flow_safe"] is expect_safe, sc
    if not expect_safe:
        assert "severed_closure_cell" in sc["control_flow_hazards"], sc


@pytest.mark.parametrize(
    ("src", "unit_fields", "expect_stale"),
    [
        pytest.param(
            "def f(alpha_one, beta_two):\n    x = alpha_one + beta_two\n    return x\n",
            {
                "start": 2,
                "end": 2,
                "start_col": 8,
                "end_col": 28,
                "kind": "complex_expr",
                "source_code": "e + beta_two",
            },
            True,
            id="single_line_complex_expr_mid_identifier_double_slice_rejected",
        ),
        pytest.param(
            "def f(alpha_one, beta_two, gamma_three):\n"
            "    x = (alpha_one + beta_two\n"
            "         + gamma_three)\n"
            "    return x\n",
            {
                "start": 2,
                "end": 3,
                "start_col": 8,
                "end_col": 23,
                "kind": "complex_expr",
                "source_code": "ne + beta_two\n         + gamma_three)",
            },
            True,
            id="multiline_complex_expr_mid_identifier_double_slice_rejected",
        ),
        pytest.param(
            "def f(alpha_one, beta_two):\n    x = alpha_one + beta_two\n    return x\n",
            {
                "start": 2,
                "end": 2,
                "start_col": 8,
                "end_col": 28,
                "kind": "complex_expr",
                "source_code": "alpha_one + beta_two",
            },
            False,
            id="control_matching_complex_expr_source_code_accepted",
        ),
    ],
)
def test_stale_source_check_does_not_double_slice_clipped_expression_units(
    src: str,
    unit_fields: dict[str, Any],
    expect_stale: bool,
) -> None:
    """Verifies stale-source validation does not re-slice already column-clipped expression units with file-relative offsets."""
    clear_dataflow_cache()
    unit: dict[str, Any] = {
        **unit_fields,
        "source_text": src,
        "source_lines": src.splitlines(keepends=True),
    }
    sc = inspect_single_unit_scope(unit, tree=ast.parse(src))
    assert ("stale_source_code" in sc["control_flow_hazards"]) is expect_stale, sc


@pytest.mark.parametrize(
    ("dec_line", "expected_col"),
    [
        pytest.param("\u2003@dec\n", 3, id="three_byte_em_space_before_decorator"),
        pytest.param("\u00a0@dec\n", 2, id="two_byte_nbsp_before_decorator"),
        pytest.param("    @dec\n", 4, id="ascii_four_space_indent_before_decorator"),
    ],
)
def test_node_earliest_start_pos_computes_utf8_byte_col_offset(
    dec_line: str,
    expected_col: int,
) -> None:
    """Verifies _node_earliest_start_pos reports UTF-8 byte offsets for leading whitespace before '@' decorators."""
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos

    tree = ast.parse("@dec\ndef f():\n    pass\n")
    fn_node = tree.body[0]
    source_lines = [dec_line, "def f():\n", "    pass\n"]
    assert _node_earliest_start_pos(fn_node, source_lines=source_lines) == (
        1,
        expected_col,
    )


@pytest.mark.parametrize(
    ("unit_file", "expect_invalid"),
    [
        pytest.param("nb.ipynb#bad_anchor", True, id="malformed_plain_anchor"),
        pytest.param("nb.ipynb#a#b", True, id="malformed_double_hash_anchor"),
        pytest.param("nb.ipynb#bad/anchor", True, id="malformed_slash_in_anchor"),
        pytest.param("NB.IPYNB#bad/frag", True, id="malformed_uppercase_ipynb_anchor"),
        pytest.param("nb.ipynb#cell_1", False, id="control_valid_cell_anchor"),
        pytest.param("my#notebook.ipynb", False, id="control_hash_in_notebook_stem"),
        pytest.param("report.ipynb#cellular.ipynb", False, id="control_literal_ipynb_hash_filename"),
        pytest.param("dir#1/nb.ipynb#cell_1", False, id="control_hash_in_dir_with_valid_anchor"),
        pytest.param("dir.ipynb#1/mod.py", False, id="control_py_module_under_hash_dir"),
    ],
)
def test_malformed_notebook_anchors_fail_closed_across_slashes_hashes_and_suffixes(
    unit_file: str,
    expect_invalid: bool,
) -> None:
    """Verifies malformed .ipynb#<fragment> anchors fail closed even with slashes, extra hashes, or .ipynb suffixes."""
    from pydoppelgangerhunt.source_lines import (
        SourceLinesOrigin,
        extract_unit_source_lines_with_state,
    )

    lines, origin = extract_unit_source_lines_with_state(
        {
            "file": unit_file,
            "start": 1,
            "end": 1,
            "source_text": "x = 1\n",
        }
    )
    if expect_invalid:
        assert lines == []
        assert origin is SourceLinesOrigin.PLACEHOLDER
    else:
        assert lines == ["x = 1\n"]
        assert origin is SourceLinesOrigin.FULL_BUFFER


@pytest.mark.parametrize(
    ("src", "expect_invalid"),
    [
        pytest.param("global x\nx: int = 1\n", False, id="module_global_then_ann_assign"),
        pytest.param("global x\nx: int\n", False, id="module_global_then_bare_ann"),
        pytest.param("x: int = 1\nglobal x\n", True, id="module_ann_assign_then_global"),
        pytest.param("x: int\nglobal x\n", True, id="module_bare_ann_then_global"),
        pytest.param("global x\nx: int = 1\nglobal x\n", True, id="module_global_ann_then_second_global"),
        pytest.param("global x\n(x): int = 1\n", False, id="module_global_then_paren_ann"),
        pytest.param("(x): int = 1\nglobal x\n", True, id="module_paren_ann_val_then_global"),
        pytest.param("(x): int\nglobal x\n", False, id="module_paren_bare_ann_then_global"),
        pytest.param("def f():\n    global x\n    x: int = 1\n", True, id="fn_global_then_ann"),
        pytest.param("def f():\n    x: int = 1\n    global x\n", True, id="fn_ann_then_global"),
        pytest.param("def f():\n    global x\n    (x): int = 1\n", False, id="fn_global_then_paren_ann"),
        pytest.param("def f():\n    (x): int = 1\n    global x\n", True, id="fn_paren_ann_val_then_global"),
        pytest.param("def f():\n    (x): int\n    global x\n", False, id="fn_paren_bare_ann_then_global"),
        pytest.param(
            "def outer():\n    x = 0\n    def f():\n        nonlocal x\n        x: int = 1\n",
            True,
            id="fn_nonlocal_then_ann",
        ),
        pytest.param(
            "def outer():\n    x = 0\n    def f():\n        nonlocal x\n        (x): int = 1\n",
            False,
            id="fn_nonlocal_then_paren_ann",
        ),
        pytest.param("class C:\n    global x\n    x: int = 1\n", True, id="class_global_then_ann"),
        pytest.param("class C:\n    global x\n    (x): int = 1\n", False, id="class_global_then_paren_ann"),
    ],
)
def test_ast_only_invalid_lexical_declarations_match_cpython_symtable_for_annassign(
    src: str,
    expect_invalid: bool,
) -> None:
    """Verifies AST-only AnnAssign global/nonlocal validation matches CPython symtable across module, function, and class scopes."""
    import symtable
    from pydoppelgangerhunt.fixer._df_span_index import (
        _compute_scope_invalid_lexical_declarations,
    )

    symtable_invalid = False
    try:
        symtable.symtable(src, "<annassign_check>", "exec")
    except SyntaxError:
        symtable_invalid = True
    assert symtable_invalid is expect_invalid

    clear_dataflow_cache()
    tree = ast.parse(src)
    assert (
        _compute_scope_invalid_lexical_declarations(
            tree, is_module_rooted=True, source_lines=None
        )
        is expect_invalid
    )







