"""Tests for UnitSchema validation, stale source_code freshness checks, and precomputed scope closure-mode contracts."""

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
        "strict",
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
        "strict",
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
            "closure_strictness": "lenient",
            "scope_schema_version": SCOPE_SCHEMA_VERSION,
        },
        "strict",
    )
    assert none_source_text_res["is_control_flow_safe"] is True
    assert none_source_text_res["closure_strictness"] == "strict"
    assert none_source_text_res["scope_schema_version"] == SCOPE_SCHEMA_VERSION

    # Properly stamped v2 scope with matching mode is preserved even without source
    stamped_strict = {
        **unmarked_precomputed,
        "closure_strictness": "strict",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    preserved = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        stamped_strict,
        "strict",
    )
    assert preserved["is_control_flow_safe"] is True
    assert "severed_closure_cell" not in preserved["control_flow_hazards"]

    marked_lenient = {
        **unmarked_precomputed,
        "closure_strictness": "lenient",
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
    }
    strict_mismatch_res = ensure_unit_scope_closure_mode(
        {"file": "nonexistent_unavailable.py", "start": 1, "end": 2},
        marked_lenient,
        "strict",
    )
    assert strict_mismatch_res["is_control_flow_safe"] is False
    assert "severed_closure_cell" in strict_mismatch_res["control_flow_hazards"]
    assert strict_mismatch_res["closure_strictness"] == "strict"

def test_detached_unit_ast_isolation_and_stale_source_code_fail_closed() -> None:
    """Verifies stale source_code fails closed and detached unit AST nodes never pollute caches."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _SessionBucket,
        analysis_session,
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
    with analysis_session() as sess:
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
        for p_key in sess.parent_maps:
            assert p_key in mod_nodes
        for c_key in sess.get_bucket(_SessionBucket.CELL_INFO):
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
        u_len, dict(u_len), closure_strictness="lenient"
    )
    assert comb_lenient["closure_strictness"] == "lenient"
    assert "severed_closure_cell" not in comb_lenient["control_flow_hazards"]

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
        "strict",
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

def test_validate_unit_schema_strict_accepts_closure_strictness_and_rejects_removed_booleans() -> None:
    """Verifies UnitSchema and validate_unit_schema(strict=True) accept closure_strictness and reject removed boolean flags."""
    from pydoppelgangerhunt.fixer.source import validate_unit_schema

    unit_with_flag = {
        "start": 1,
        "end": 2,
        "kind": "compound_block",
        "closure_strictness": "pedantic",
    }
    assert validate_unit_schema(unit_with_flag, strict=True) == set()

    for removed_key in ("strict_with_suppression", "skip_pre_unit_closures", "unrecognized_extra_field"):
        with pytest.raises(ValueError, match="Unrecognized keys"):
            validate_unit_schema(
                {**unit_with_flag, removed_key: True},
                strict=True,
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
        "closure_strictness": "strict",
        "non_suppressing_ctx_managers": [],
    }
    preserved = ensure_unit_scope_closure_mode(
        {"start": 1, "end": 2},
        stale_scope,
        "pedantic",
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

@pytest.mark.parametrize(
    ("bad_unit", "err_pattern"),
    [
        ({"start": "not_an_int", "end": 2}, "Invalid coordinate value"),
        ({"start": 0, "end": 2}, "Invalid coordinate value"),
        ({"start": 5, "end": 3}, "Inverted line coordinates"),
        ({"start": 1, "end": 2, "start_col": -1}, "Invalid coordinate value"),
        ({"start": 1, "end": 2, "allow_snippet_only": "yes"}, "Expected bool"),
        ({"start": 1, "end": 2, "is_subroutine": 1}, "Expected bool"),
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
        "closure_strictness": "lenient",
    }
    recomputed_no_backing = ensure_unit_scope_closure_mode(
        snippet_unit,
        stale_lenient_scope,
        "strict",
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
    from pydoppelgangerhunt.reporters import synthesize_refactoring_suggestion

    peer_unit = {"file": str(mod), "start": 1, "end": 3, "kind": "function"}
    with pytest.raises(ValueError, match="ordered sequence"):
        synthesize_shared_helper_code(bad_unit, peer_unit, repo_root=str(tmp_path))
    with pytest.raises(ValueError, match="ordered sequence"):
        synthesize_refactoring_suggestion(bad_unit, peer_unit, repo_root=str(tmp_path))
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
        "pedantic",
        source_text=src,
    )
    assert recomputed is not contradictory
    assert recomputed["inputs"] == ["x"]

    # Without backing source, recomputation still fails closed with is_control_flow_safe=False
    failed_closed = ensure_unit_scope_closure_mode(
        {"start": 2, "end": 3, "kind": "compound_block"},
        contradictory,
        "pedantic",
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
        "pedantic",
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
        "closure_strictness": "lenient",
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
        snippet_sub_unit, pre_lenient_scope, "strict"
    )
    assert recomp_sub["is_control_flow_safe"] is False
    assert "unavailable_source_for_closure_mode" in recomp_sub["control_flow_hazards"]

    snippet_whole_unit = {**snippet_sub_unit, "kind": "function", "source_code": "def fn(x):\n    return x + 1\n"}
    recomp_whole = ensure_unit_scope_closure_mode(
        snippet_whole_unit, pre_lenient_scope, "strict"
    )
    assert "unavailable_source_for_closure_mode" not in recomp_whole["control_flow_hazards"]

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
            "strict",
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
    ("module_src", "line_marker", "col_kwargs", "expect_invalid_cols"),
    [
        pytest.param(
            "def f(x):\n    val = x + 1\n    return val\n",
            "val = x + 1",
            {"start_col": 4, "end_col": 600},
            True,
            id="single_line_in_range_start_col_and_out_of_range_end_col_rejected",
        ),
        pytest.param(
            "def f(x):\n    val = x + 1\n    return val\n",
            "val = x + 1",
            {"col_offset": 4, "end_col_offset": 600},
            True,
            id="single_line_alias_keys_out_of_range_end_col_rejected",
        ),
        pytest.param(
            "def f(x):\n    val = x + 1\n    return val\n",
            "val = x + 1",
            {"start_col": 4, "end_col": 15},
            False,
            id="control_single_line_valid_start_and_end_cols_accepted",
        ),
        pytest.param(
            "def f():\n    x = (\n        1\n    ); longer_stmt_name = 123456\n",
            "longer_stmt_name",
            {"start_col": 7, "end_col": 32},
            False,
            id="mixed_multiline_and_single_line_valid_end_col_accepted",
        ),
        pytest.param(
            "def f():\n    x = (\n        1\n    ); longer_stmt_name = 123456\n",
            "longer_stmt_name",
            {"start_col": 7, "end_col": 600},
            True,
            id="mixed_multiline_and_single_line_out_of_range_end_col_rejected",
        ),
    ],
)
def test_placeholder_tree_cols_out_of_bounds_checks_single_line_and_mixed_end_cols(
    module_src: str,
    line_marker: str,
    col_kwargs: dict[str, int],
    expect_invalid_cols: bool,
) -> None:
    """Verifies tree-only placeholder units without source_code validate end_col against single-line and multi-line AST nodes on u_end."""
    mod_tree = ast.parse(module_src)
    target_line = next(
        idx for idx, ln in enumerate(module_src.splitlines(), start=1) if line_marker in ln
    )
    unit: dict[str, Any] = {
        "name": "target",
        "kind": "basic_block",
        "start": target_line,
        "end": target_line,
        **col_kwargs,
    }
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(unit, tree=mod_tree)
    has_invalid = "invalid_unit_columns" in sc["control_flow_hazards"]
    assert has_invalid is expect_invalid_cols, sc

@pytest.mark.parametrize(
    ("backing_line", "unit_kind", "src_code", "extra_cols", "expect_stale"),
    [
        pytest.param(
            "    x = foo(1); y = bar(2)\n",
            "basic_block",
            "foo(1)",
            {},
            True,
            id="basic_block_subexpr_on_semicolon_line_is_stale",
        ),
        pytest.param(
            "    x = foo(1); y = bar(2)\n",
            "compound_block",
            "bar(2)",
            {},
            True,
            id="compound_block_subexpr_on_semicolon_line_is_stale",
        ),
        pytest.param(
            "    x = foo(1); y = bar(2)\n",
            "basic_block",
            "x = foo(1)",
            {},
            False,
            id="control_basic_block_full_first_stmt_on_semicolon_line_valid",
        ),
        pytest.param(
            "    x = foo(1); bar(2)\n",
            "basic_block",
            "bar(2)",
            {},
            False,
            id="control_basic_block_full_second_expr_stmt_on_semicolon_line_valid",
        ),
        pytest.param(
            "    x = foo(1); y = bar(2)\n",
            "complex_expr",
            "foo(1)",
            {},
            False,
            id="control_complex_expr_subexpr_on_semicolon_line_valid",
        ),
        pytest.param(
            "    x = foo(1); y = bar(2)\n",
            "basic_block",
            "foo(1)",
            {"start_col": 8, "end_col": 14},
            False,
            id="control_basic_block_with_explicit_col_bounds_valid",
        ),
    ],
)
def test_semicolon_line_stale_source_code_requires_stmt_delimiters_for_stmt_units(
    backing_line: str,
    unit_kind: str,
    src_code: str,
    extra_cols: dict[str, int],
    expect_stale: bool,
) -> None:
    """Verifies statement units without column bounds reject sub-expression snippets on semicolon lines as stale_source_code."""
    full_src = f"def run(foo, bar):\n{backing_line}    return x\n"
    unit: dict[str, Any] = {
        "name": "run:stmt",
        "kind": unit_kind,
        "start": 2,
        "end": 2,
        "source_text": full_src,
        "source_code": src_code,
        **extra_cols,
    }
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(unit)
    has_stale = "stale_source_code" in sc["control_flow_hazards"]
    assert has_stale is expect_stale, sc

@pytest.mark.parametrize(
    ("case_id", "expect_allowed"),
    [
        pytest.param("inline_source_text_oob", False, id="u2_inline_source_text_oob_rejected"),
        pytest.param("inline_source_lines_oob", False, id="u2_inline_source_lines_oob_rejected"),
        pytest.param("inline_file_source_oob", False, id="u2_inline_file_source_oob_rejected"),
        pytest.param("disk_py_file_oob", False, id="u2_readable_disk_py_oob_rejected"),
        pytest.param("disk_ipynb_oob_cell", False, id="u2_disk_ipynb_oob_cell_rejected"),
        pytest.param("malformed_ipynb_anchor", False, id="u2_malformed_ipynb_anchor_rejected"),
        pytest.param("control_missing_disk_file", True, id="control_genuinely_missing_u2_allowed"),
    ],
)
def test_allows_u2_source_placeholder_rejects_backed_or_invalid_u2_in_preview_mode(
    tmp_path: Path,
    case_id: str,
    expect_allowed: bool,
) -> None:
    """Verifies preview-mode u2 placeholder waiver rejects inline buffers, readable disk files, and invalid notebooks."""
    import json  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        allows_u2_source_placeholder,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        synthesize_shared_helper_code,
    )

    fn_src = "def compute(x):\n    return x * 2 + 1\n"
    f1 = tmp_path / "a.py"
    f1.write_text(fn_src, encoding="utf-8")
    u1: dict[str, Any] = {
        "file": str(f1),
        "name": "compute",
        "kind": "function",
        "start": 1,
        "end": 2,
    }

    if case_id == "inline_source_text_oob":
        u2: dict[str, Any] = {
            "file": str(tmp_path / "missing_b.py"),
            "name": "compute_b",
            "kind": "function",
            "start": 50,
            "end": 52,
            "source_text": fn_src,
        }
    elif case_id == "inline_source_lines_oob":
        u2 = {
            "file": str(tmp_path / "missing_b.py"),
            "name": "compute_b",
            "kind": "function",
            "start": 50,
            "end": 52,
            "source_lines": fn_src.splitlines(keepends=True),
            "source_lines_is_sliced": False,
        }
    elif case_id == "inline_file_source_oob":
        u2 = {
            "file": str(tmp_path / "missing_b.py"),
            "name": "compute_b",
            "kind": "function",
            "start": 50,
            "end": 52,
            "file_source": fn_src,
        }
    elif case_id == "disk_py_file_oob":
        f2 = tmp_path / "b.py"
        f2.write_text(fn_src, encoding="utf-8")
        u2 = {
            "file": str(f2),
            "name": "compute_b",
            "kind": "function",
            "start": 50,
            "end": 52,
        }
    elif case_id == "disk_ipynb_oob_cell":
        nb = tmp_path / "nb.ipynb"
        nb.write_text(
            json.dumps(
                {
                    "nbformat": 4,
                    "cells": [{"cell_type": "code", "source": [fn_src]}],
                }
            ),
            encoding="utf-8",
        )
        u2 = {
            "file": f"{nb}#cell_9",
            "name": "compute_b",
            "kind": "function",
            "start": 1,
            "end": 2,
        }
    elif case_id == "malformed_ipynb_anchor":
        u2 = {
            "file": f"{tmp_path / 'nb.ipynb'}#bad_anchor",
            "name": "compute_b",
            "kind": "function",
            "start": 1,
            "end": 2,
        }
    else:
        u2 = {
            "file": str(tmp_path / "missing_b.py"),
            "name": "compute_b",
            "kind": "function",
            "start": 1,
            "end": 2,
        }

    clear_dataflow_cache()
    waived = allows_u2_source_placeholder(
        is_subroutine=False,
        requires_u2_edit=False,
        source_text2=None,
        u2=u2,
        repo_root=str(tmp_path),
    )
    assert waived is expect_allowed

    s1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    s2 = inspect_single_unit_scope(u2, repo_root=str(tmp_path))
    assert "unavailable_source_placeholder" in s2["control_flow_hazards"]

    pair_res = resolve_clone_pair_outputs(
        u1,
        u2,
        s1,
        s2,
        source_text1=fn_src,
        source_text2=None,
        repo_root=str(tmp_path),
        require_pairing=False,
        return_details=True,
    )
    helper_code = synthesize_shared_helper_code(
        u1,
        u2,
        source_text1=fn_src,
        source_text2=None,
        repo_root=str(tmp_path),
    )
    if expect_allowed:
        assert pair_res.outputs is not None
        assert "def _shared_" in helper_code
    else:
        assert pair_res.outputs is None
        assert helper_code == ""

@pytest.mark.parametrize(
    ("raw_heuristics", "raw_wildcards", "expect_heuristics", "expect_wildcards"),
    [
        pytest.param(None, None, set(), set(), id="none_fields_ignored_safely"),
        pytest.param(
            "LockMgr",
            "*.atomic",
            set(),
            set(),
            id="string_fields_not_iterated_as_characters",
        ),
        pytest.param(123, 456, set(), set(), id="int_fields_ignored_safely"),
        pytest.param(
            ["LockMgr", "", 99],
            [["*.atomic", "db.atomic"], "bad_entry"],
            {"LockMgr"},
            {("*.atomic", "db.atomic")},
            id="mixed_sequences_keep_only_valid_entries",
        ),
    ],
)
def test_ensure_unit_scope_closure_mode_handles_malformed_ctx_manager_fields(
    raw_heuristics: Any,
    raw_wildcards: Any,
    expect_heuristics: set[str],
    expect_wildcards: set[tuple[str, str]],
) -> None:
    """Verifies ensure_unit_scope_closure_mode tolerates None/non-iterable/string heuristic and wildcard fields."""
    from pydoppelgangerhunt.config import (
        collect_heuristic_ctx_managers,
        collect_wildcard_ctx_managers,
    )
    from pydoppelgangerhunt.fixer.scope import (
        SCOPE_SCHEMA_VERSION,
        ensure_unit_scope_closure_mode,
    )

    matching_scope: dict[str, Any] = {
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "closure_strictness": "strict",
        "non_suppressing_ctx_managers": [],
        "heuristic_ctx_managers": raw_heuristics,
        "wildcard_ctx_managers": raw_wildcards,
        "is_control_flow_safe": True,
        "control_flow_hazards": [],
    }
    unit: dict[str, Any] = {"name": "f", "start": 1, "end": 2, "kind": "function"}
    with collect_heuristic_ctx_managers(isolate=True) as h_out, collect_wildcard_ctx_managers(
        isolate=True
    ) as w_out:
        res = ensure_unit_scope_closure_mode(
            unit, matching_scope, "strict", allow_disk_read=False
        )
    assert res is matching_scope
    assert h_out == expect_heuristics
    assert w_out == expect_wildcards

    # Also verify fail-closed fallback path when mode mismatches and source is unavailable
    fallback = ensure_unit_scope_closure_mode(
        unit, matching_scope, "pedantic", allow_disk_read=False
    )
    assert fallback["is_control_flow_safe"] is False
    assert fallback["heuristic_ctx_managers"] == sorted(expect_heuristics)
    assert fallback["wildcard_ctx_managers"] == sorted(expect_wildcards)
