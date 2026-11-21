"""Tests for closure strictness policy defaults, concurrency isolation, CLI exit status, per-unit allowlists, and AST recursion guards."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from pydoppelgangerhunt.cli import main
from pydoppelgangerhunt.config import (
    clear_closure_warning_cache,
    collect_heuristic_ctx_managers,
    heuristic_ctx_manager_session,
    log_heuristic_ctx_manager_summary,
    record_heuristic_non_suppressing_ctx_manager,
)
from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    generate_refactoring_patch,
    inspect_single_unit_scope,
    resolve_clone_pair_outputs,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer._df_cell_hazards import (
    _PedanticPrimitiveEnv,
    _extract_unpack_leaf_names,
    _is_proven_primitive_dunder_expr,
)
from pydoppelgangerhunt.fixer._df_closures import _is_irrefutable_pattern
from pydoppelgangerhunt.fixer._df_pairing import (
    ASYNC_GENERATOR_DELEGATION_HAZARD_REASON,
    DELETED_OUTPUT_HAZARD_REASON,
    ENCLOSING_TRY_OR_WITH_CLEANUP_HAZARD_REASON,
    GENERATOR_CLOSURE_CONFLICT_REASON,
    GENERATOR_OUTPUT_CONFLICT_REASON,
    INVALID_UNIT_BOUNDS_REASON,
)
from pydoppelgangerhunt.fixer._df_with_provenance import (
    _SuppressProvenance,
    _analyze_block_assignment,
    _extract_definite_target_names,
    _extract_deleted_names,
    _is_non_raising_expr,
    _is_non_raising_store_target,
    _update_suppress_provenance_from_stmts,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.fixer.dataflow import SAFETY_PAIR_SKIP_REASONS
from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines
from pydoppelgangerhunt.reporters import (
    generate_html_report,
    synthesize_refactoring_suggestion,
)

_CONDITIONAL_TX_GENERATOR_SRC = (
    "def gen_alpha(compute, scale):\n"
    "    with OpaqueCustomTx():\n"
    "        val = compute(scale)\n"
    "        yield val\n"
    "        extra = val + 1\n"
    "    yield extra * 2\n"
    "\n"
    "def gen_beta(compute, scale):\n"
    "    with OpaqueCustomTx():\n"
    "        val = compute(scale)\n"
    "        yield val\n"
    "        extra = val + 1\n"
    "    yield extra * 2\n"
)


def _conditional_tx_pair(
    tmp_path: Path,
    *,
    per_unit_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mod_p = tmp_path / "cond_tx_gen.py"
    mod_p.write_text(_CONDITIONAL_TX_GENERATOR_SRC, encoding="utf-8")
    extra = dict(per_unit_overrides or {})
    u1: dict[str, Any] = {
        "name": "gen_alpha:with",
        "file": "cond_tx_gen.py",
        "start": 2,
        "end": 5,
        "kind": "compound_block",
        **extra,
    }
    u2: dict[str, Any] = {
        "name": "gen_beta:with",
        "file": "cond_tx_gen.py",
        "start": 9,
        "end": 12,
        "kind": "compound_block",
        **extra,
    }
    return u1, u2


@pytest.mark.parametrize(
    ("patch_kwargs", "unit_overrides", "expect_blocked"),
    [
        pytest.param(
            {"skip_pre_unit_closures": False},
            None,
            True,
            id="caller_skip_false_preserves_pedantic_default",
        ),
        pytest.param(
            {"closure_strictness": "unrecognized_policy_xyz"},
            None,
            True,
            id="caller_unrecognized_mode_preserves_pedantic_default",
        ),
        pytest.param(
            {},
            {"skip_pre_unit_closures": False},
            True,
            id="unit_skip_false_preserves_pedantic_default",
        ),
        pytest.param(
            {},
            {"strict_with_suppression": False},
            False,
            id="unit_strict_with_suppression_false_opts_out_of_pedantic",
        ),
        pytest.param(
            {"closure_strictness": "strict"},
            None,
            False,
            id="caller_explicit_strict_opts_out_of_pedantic",
        ),
        pytest.param(
            {},
            {"closure_strictness": "strict"},
            False,
            id="unit_explicit_strict_opts_out_of_pedantic",
        ),
        pytest.param(
            {"skip_pre_unit_closures": True},
            None,
            False,
            id="caller_skip_true_opts_out_of_pedantic",
        ),
    ],
)
def test_replace_clones_pedantic_policy_ignores_false_and_unrecognized_flags(
    tmp_path: Path,
    patch_kwargs: dict[str, Any],
    unit_overrides: dict[str, Any] | None,
    expect_blocked: bool,
) -> None:
    """Verifies legacy False flags and invalid modes never opt replace_clones=True out of pedantic safety."""
    clear_dataflow_cache()
    clear_closure_warning_cache()
    u1, u2 = _conditional_tx_pair(tmp_path, per_unit_overrides=unit_overrides)
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        **patch_kwargs,
    )
    if expect_blocked:
        assert patch == ""
    else:
        assert "def _shared_" in patch


def test_heuristic_ctx_manager_asyncio_task_isolation() -> None:
    """Verifies interleaved asyncio tasks retain disjoint collect_heuristic_ctx_managers buckets."""

    async def _worker(tag: str, delay_steps: int) -> frozenset[str]:
        with collect_heuristic_ctx_managers() as seen:
            record_heuristic_non_suppressing_ctx_manager(f"{tag}.First")
            for _ in range(delay_steps):
                await asyncio.sleep(0)
            record_heuristic_non_suppressing_ctx_manager(f"{tag}.Second")
            await asyncio.sleep(0)
            return frozenset(seen)

    async def _run_both() -> tuple[frozenset[str], frozenset[str]]:
        res_a, res_b = await asyncio.gather(_worker("Alpha", 2), _worker("Beta", 1))
        return res_a, res_b

    clear_closure_warning_cache()
    bucket_a, bucket_b = asyncio.run(_run_both())
    assert bucket_a == frozenset({"Alpha.First", "Alpha.Second"})
    assert bucket_b == frozenset({"Beta.First", "Beta.Second"})


def test_clear_closure_warning_cache_preserves_other_threads_and_sessions() -> None:
    """Verifies clear_closure_warning_cache clears only current thread/session without dropping concurrent state."""
    clear_closure_warning_cache()
    entered_session = threading.Barrier(2)
    cleared_main = threading.Barrier(2)
    session_flushed: list[str] = []
    thread_default_flushed: list[str] = []

    def _session_worker() -> None:
        with heuristic_ctx_manager_session(warn=False):
            record_heuristic_non_suppressing_ctx_manager("WorkerSessionMgr")
            entered_session.wait(timeout=5)
            cleared_main.wait(timeout=5)
            session_flushed.extend(log_heuristic_ctx_manager_summary())

    t_sess = threading.Thread(target=_session_worker)
    t_sess.start()
    try:
        entered_session.wait(timeout=5)
        record_heuristic_non_suppressing_ctx_manager("MainDefaultMgr")
        clear_closure_warning_cache()
        assert log_heuristic_ctx_manager_summary() == ()
    finally:
        cleared_main.wait(timeout=5)
        t_sess.join(timeout=5)

    assert session_flushed == ["WorkerSessionMgr"]

    entered_default = threading.Barrier(2)
    cleared_default_main = threading.Barrier(2)

    def _default_bucket_worker() -> None:
        record_heuristic_non_suppressing_ctx_manager("WorkerDefaultMgr")
        entered_default.wait(timeout=5)
        cleared_default_main.wait(timeout=5)
        thread_default_flushed.extend(log_heuristic_ctx_manager_summary())

    t_def = threading.Thread(target=_default_bucket_worker)
    t_def.start()
    try:
        entered_default.wait(timeout=5)
        record_heuristic_non_suppressing_ctx_manager("MainSecondDefaultMgr")
        clear_closure_warning_cache()
        assert log_heuristic_ctx_manager_summary() == ()
    finally:
        cleared_default_main.wait(timeout=5)
        t_def.join(timeout=5)

    assert thread_default_flushed == ["WorkerDefaultMgr"]


@pytest.mark.parametrize(
    ("cli_format", "extra_flags"),
    [
        pytest.param("text", ["--suggest"], id="text_format_with_suggest"),
        pytest.param("json", [], id="json_format"),
        pytest.param("sarif", [], id="sarif_format"),
    ],
)
def test_cli_exit_code_4_on_diagnostic_errors_with_clones_across_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_format: str,
    extra_flags: list[str],
) -> None:
    """Verifies internal synthesis/HTML errors cause CLI exit code 4 (clones + diagnostic errors) and stderr warning across output formats."""
    target = tmp_path / "dup.py"
    target.write_text(
        "def alpha(x):\n    a = x + 1\n    b = a * 2\n    return b + 3\n\n"
        "def beta(x):\n    a = x + 1\n    b = a * 2\n    return b + 3\n",
        encoding="utf-8",
    )

    clean_exit = main([str(target), "--min-lines", "3", "--format", cli_format])
    assert clean_exit == 1
    capsys.readouterr()

    def _boom(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr("pydoppelgangerhunt.cli.synthesize_refactoring_suggestion", _boom)
    monkeypatch.setattr("pydoppelgangerhunt.reporters.synthesize_refactoring_suggestion", _boom)
    html_out = tmp_path / f"report_{cli_format}.html"
    err_exit = main(
        [str(target), "--min-lines", "3", "--format", cli_format, "--html", str(html_out), *extra_flags]
    )
    captured = capsys.readouterr()
    assert err_exit == 4
    assert "internal diagnostic error(s)" in captured.err
    if cli_format in ("json", "sarif"):
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)


def test_per_unit_non_suppressing_ctx_managers_honored_end_to_end(tmp_path: Path) -> None:
    """Verifies unit-level non_suppressing_ctx_managers propagates consistently across all analysis and synthesis phases."""
    clear_dataflow_cache()
    clear_closure_warning_cache()

    blocked_u1, blocked_u2 = _conditional_tx_pair(tmp_path)
    sc_blocked = inspect_single_unit_scope(
        blocked_u1, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    sc_blocked2 = inspect_single_unit_scope(
        blocked_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "extra" not in sc_blocked["definite_stores"]
    assert "extra" in sc_blocked["conditional_outputs"]
    assert (
        resolve_clone_pair_outputs(
            blocked_u1,
            blocked_u2,
            sc_blocked,
            sc_blocked2,
            repo_root=str(tmp_path),
            closure_strictness="pedantic",
        )
        is None
    )
    assert (
        synthesize_shared_helper_code(
            blocked_u1, blocked_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
        )
        == ""
    )
    sug_blocked = synthesize_refactoring_suggestion(
        blocked_u1, blocked_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "def _shared_" not in sug_blocked
    assert "generator output conflict" in sug_blocked
    html_blocked = generate_html_report(
        [(1.0, blocked_u1, blocked_u2)],
        str(tmp_path),
        0.8,
        repo_root=str(tmp_path),
        closure_strictness="pedantic",
    )
    assert "def _shared_" not in html_blocked
    assert (
        generate_refactoring_patch(
            [(1.0, blocked_u1, blocked_u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
            closure_strictness="pedantic",
        )
        == ""
    )

    allow_override = {"non_suppressing_ctx_managers": ["OpaqueCustomTx"]}
    allowed_u1, allowed_u2 = _conditional_tx_pair(tmp_path, per_unit_overrides=allow_override)
    sc_allowed = inspect_single_unit_scope(
        allowed_u1, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    sc_allowed2 = inspect_single_unit_scope(
        allowed_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "extra" in sc_allowed["definite_stores"]
    assert sc_allowed["conditional_outputs"] == []
    assert resolve_clone_pair_outputs(
        allowed_u1,
        allowed_u2,
        sc_allowed,
        sc_allowed2,
        repo_root=str(tmp_path),
        closure_strictness="pedantic",
    ) == (["extra"], ["extra"])
    helper = synthesize_shared_helper_code(
        allowed_u1, allowed_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "def _shared_" in helper
    sug = synthesize_refactoring_suggestion(
        allowed_u1, allowed_u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "def _shared_" in sug
    html_allowed = generate_html_report(
        [(1.0, allowed_u1, allowed_u2)],
        str(tmp_path),
        0.8,
        repo_root=str(tmp_path),
        closure_strictness="pedantic",
    )
    assert "def _shared_" in html_allowed
    patch_allowed = generate_refactoring_patch(
        [(1.0, allowed_u1, allowed_u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="pedantic",
    )
    assert "def _shared_" in patch_allowed


def test_safety_pair_skip_reasons_and_suggestion_rejection_enclosing_with_cleanup(
    tmp_path: Path,
) -> None:
    """Verifies SAFETY_PAIR_SKIP_REASONS aligns with pairing hazards and suggestion classifies enclosing cleanup hazards."""
    assert ENCLOSING_TRY_OR_WITH_CLEANUP_HAZARD_REASON == "enclosing try or with cleanup hazard"
    for reason in (
        ENCLOSING_TRY_OR_WITH_CLEANUP_HAZARD_REASON,
        DELETED_OUTPUT_HAZARD_REASON,
        INVALID_UNIT_BOUNDS_REASON,
        ASYNC_GENERATOR_DELEGATION_HAZARD_REASON,
    ):
        assert reason in SAFETY_PAIR_SKIP_REASONS
    assert GENERATOR_OUTPUT_CONFLICT_REASON not in SAFETY_PAIR_SKIP_REASONS
    assert GENERATOR_CLOSURE_CONFLICT_REASON in SAFETY_PAIR_SKIP_REASONS

    src = (
        "def run_a(GuardCtx, items):\n"
        "    with GuardCtx():\n"
        "        acc = 0\n"
        "        for item in items:\n"
        "            acc += item\n"
        "            yield acc\n"
        "    return acc\n\n"
        "def run_b(GuardCtx, items):\n"
        "    with GuardCtx():\n"
        "        acc = 0\n"
        "        for item in items:\n"
        "            acc += item\n"
        "            yield acc\n"
        "    return acc\n"
    )
    mod_p = tmp_path / "enclosing_with_cleanup.py"
    mod_p.write_text(src, encoding="utf-8")
    u1 = {"name": "run_a:loop", "file": "enclosing_with_cleanup.py", "start": 3, "end": 6, "kind": "compound_block"}
    u2 = {"name": "run_b:loop", "file": "enclosing_with_cleanup.py", "start": 10, "end": 13, "kind": "compound_block"}

    sc1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path), closure_strictness="pedantic")
    sc2 = inspect_single_unit_scope(u2, repo_root=str(tmp_path), closure_strictness="pedantic")
    res = resolve_clone_pair_outputs(
        u1, u2, sc1, sc2, repo_root=str(tmp_path), closure_strictness="pedantic", return_details=True
    )
    assert res.outputs is None
    assert res.skip_reason == ENCLOSING_TRY_OR_WITH_CLEANUP_HAZARD_REASON

    sug = synthesize_refactoring_suggestion(
        u1, u2, repo_root=str(tmp_path), closure_strictness="pedantic"
    )
    assert "rejected as unsafe" in sug
    assert ENCLOSING_TRY_OR_WITH_CLEANUP_HAZARD_REASON in sug


def test_slice_unit_token_lines_inverted_single_line_columns_fail_closed() -> None:
    """Verifies single-line inverted column bounds fail closed by returning None in _slice_unit_token_lines."""
    assert (
        _slice_unit_token_lines(
            {"kind": "comprehension", "start_col": 12, "end_col": 4},
            ["    [x for x in y]\n"],
        )
        is None
    )
    assert (
        _slice_unit_token_lines(
            {"kind": "comprehension", "start_col": 4, "end_col": 18},
            ["    [x for x in y]\n"],
        )
        == ["[x for x in y]"]
    )


def test_cli_empty_scan_exits_0(tmp_path: Path) -> None:
    """Verifies CLI exits 0 on a clean empty target with zero clones and zero diagnostic errors."""
    empty_file = tmp_path / "clean_only.py"
    empty_file.write_text("x = 1\n", encoding="utf-8")
    assert main([str(empty_file)]) == 0
    with mock.patch("pydoppelgangerhunt.cli.scan_target", return_value=[]):
        assert main([str(tmp_path)]) == 0


def _build_nested_binop(depth: int, inner: ast.expr | None = None) -> ast.expr:
    node: ast.expr = inner if inner is not None else ast.Constant(value=1)
    for _ in range(depth):
        node = ast.BinOp(left=node, op=ast.Add(), right=ast.Constant(value=1))
    return ast.fix_missing_locations(node)


def _build_nested_ifexp(depth: int, leaf: ast.expr) -> ast.expr:
    node: ast.expr = leaf
    for _ in range(depth):
        node = ast.IfExp(
            test=ast.Constant(value=True),
            body=ast.Constant(value=0),
            orelse=node,
        )
    return ast.fix_missing_locations(node)


def _build_nested_boolop(depth: int, leaf: ast.expr) -> ast.expr:
    node: ast.expr = leaf
    for _ in range(depth):
        node = ast.BoolOp(op=ast.And(), values=[ast.Constant(value=True), node])
    return ast.fix_missing_locations(node)


def _build_nested_tuple_target(depth: int, prefix: str = "x") -> tuple[ast.expr, ast.expr]:
    tgt: ast.expr = ast.Name(id=f"{prefix}_leaf", ctx=ast.Store())
    val: ast.expr = ast.Constant(value=0)
    for idx in range(depth):
        tgt = ast.Tuple(
            elts=[ast.Name(id=f"{prefix}_{idx}", ctx=ast.Store()), tgt],
            ctx=ast.Store(),
        )
        val = ast.Tuple(elts=[ast.Constant(value=idx), val], ctx=ast.Load())
    return ast.fix_missing_locations(tgt), ast.fix_missing_locations(val)


def _build_nested_with_chain(depth: int, leaf_stmts: list[ast.stmt]) -> ast.With:
    body: list[ast.stmt] = leaf_stmts
    for _ in range(depth):
        body = [
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Call(
                            func=ast.Name(id="cm", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        ),
                        optional_vars=None,
                    )
                ],
                body=body,
            )
        ]
    return ast.fix_missing_locations(body[0])  # type: ignore[return-value]


@pytest.mark.parametrize("shape", ["elif_chain", "ternary_chain", "binop_chain"])
def test_end_to_end_pipeline_handles_deep_ast_without_recursion_error(
    tmp_path: Path, shape: str
) -> None:
    """Verifies scope inspection, helper synthesis, and patching handle 420+ deep ASTs without RecursionError."""
    clear_dataflow_cache()
    clear_closure_warning_cache()
    if shape == "elif_chain":
        elifs = "".join("        elif c:\n            pass\n" for _ in range(420))
        body_tpl = (
            "def {fn}(c):\n"
            "    with CustomGuard():\n"
            "        if c:\n"
            "            res = 1\n"
            f"{elifs}"
            "        else:\n"
            "            res = 2\n"
            "        cb = lambda: res\n"
            "        return cb()\n"
        )
        line_count = 4 + 420 * 2 + 4
    elif shape == "ternary_chain":
        chain = "1 if c else " * 420 + "2"
        body_tpl = (
            "def {fn}(c):\n"
            "    res = " + chain + "\n"
            "    other = res + 1\n"
            "    return other\n"
        )
        line_count = 4
    else:
        sum_chain = " + ".join(["x"] * 420)
        body_tpl = (
            "def {fn}(x):\n"
            "    total = " + sum_chain + "\n"
            "    doubled = total * 2\n"
            "    return doubled\n"
        )
        line_count = 4

    mod_src = body_tpl.format(fn="fn_a") + "\n" + body_tpl.format(fn="fn_b")
    mod_p = tmp_path / f"deep_{shape}.py"
    mod_p.write_text(mod_src, encoding="utf-8")
    u1 = {"name": "fn_a", "file": mod_p.name, "start": 1, "end": line_count, "kind": "function"}
    u2 = {
        "name": "fn_b",
        "file": mod_p.name,
        "start": line_count + 2,
        "end": line_count * 2 + 1,
        "kind": "function",
    }

    sc1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    assert isinstance(sc1["definite_stores"], list)
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=False)
    assert isinstance(helper, str)
    assert isinstance(patch, str)


def test_deep_ast_expression_and_target_walkers() -> None:
    """Verifies AST walkers handle 420-deep trees iteratively or fail closed without RecursionError."""
    walrus_leaf = ast.NamedExpr(
        target=ast.Name(id="w_deep", ctx=ast.Store()),
        value=ast.Constant(value=1),
    )

    deep_binop_walrus = _build_nested_binop(420, walrus_leaf)
    d_bin, c_bin = _walrus_assignment_in_expr(deep_binop_walrus, is_conditional=False)
    assert d_bin == {"w_deep"}
    assert c_bin == set()

    deep_ifexp_walrus = _build_nested_ifexp(420, walrus_leaf)
    d_if, c_if = _walrus_assignment_in_expr(deep_ifexp_walrus, is_conditional=False)
    assert d_if == set()
    assert c_if == {"w_deep"}

    deep_bool_walrus = _build_nested_boolop(420, walrus_leaf)
    d_bool, c_bool = _walrus_assignment_in_expr(deep_bool_walrus, is_conditional=False)
    assert d_bool == set()
    assert c_bool == {"w_deep"}

    assert _is_non_raising_expr(_build_nested_ifexp(420, ast.Constant(value=1))) is False
    assert _is_non_raising_expr(_build_nested_boolop(420, ast.Constant(value=True))) is False

    assert _is_proven_primitive_dunder_expr(
        _build_nested_binop(5), _PedanticPrimitiveEnv()
    ) is True
    assert _is_proven_primitive_dunder_expr(
        _build_nested_binop(420), _PedanticPrimitiveEnv()
    ) is False

    deep_tgt, deep_val = _build_nested_tuple_target(420)
    assert _extract_unpack_leaf_names(deep_tgt) == frozenset(
        {"x_leaf", *(f"x_{i}" for i in range(420))}
    )
    assert _extract_definite_target_names(deep_tgt) == {
        "x_leaf",
        *(f"x_{i}" for i in range(420)),
    }
    assert _is_non_raising_store_target(deep_tgt, deep_val) is True

    deep_del: ast.stmt = ast.Delete(targets=[ast.Name(id="gone", ctx=ast.Del())])
    for _ in range(420):
        deep_del = ast.If(test=ast.Constant(value=True), body=[deep_del], orelse=[])
    assert _extract_deleted_names([ast.fix_missing_locations(deep_del)]) == {"gone"}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match/case AST added in Python 3.10")
def test_deep_match_pattern_irrefutability_is_iterative() -> None:
    """Verifies _is_irrefutable_pattern evaluates 420+ deep MatchOr/MatchAs chains without RecursionError."""
    pat_irref: ast.pattern = ast.MatchAs(pattern=None, name="wildcard")
    pat_ref: ast.pattern = ast.MatchValue(value=ast.Constant(value=999))
    for i in range(420):
        pat_irref = ast.MatchOr(patterns=[ast.MatchValue(value=ast.Constant(value=i)), pat_irref])
        pat_ref = ast.MatchOr(patterns=[ast.MatchValue(value=ast.Constant(value=i)), pat_ref])
    assert _is_irrefutable_pattern(ast.fix_missing_locations(pat_irref)) is True
    assert _is_irrefutable_pattern(ast.fix_missing_locations(pat_ref)) is False


def test_deep_nested_with_block_flow_and_provenance_fail_closed() -> None:
    """Verifies 420-deep compound blocks bail out fail-closed for flow bindings and provenance."""
    leaf_stmts: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id="x_deep", ctx=ast.Store())],
            value=ast.Constant(value=1),
        ),
        ast.Assign(
            targets=[ast.Name(id="deep_mgr", ctx=ast.Store())],
            value=ast.Name(id="suppress", ctx=ast.Load()),
        ),
    ]
    deep_with = _build_nested_with_chain(420, leaf_stmts)

    def_out, cond_out = _analyze_block_assignment(
        [deep_with],
        definite={"outer_ok"},
    )
    assert "outer_ok" in def_out
    assert "x_deep" not in def_out
    assert "x_deep" in cond_out

    seed_prov = _SuppressProvenance(
        non_suppressors=frozenset({"nullcontext"}),
        imported_symbols=frozenset({("nullcontext", "contextlib.nullcontext")}),
    )
    out_prov = _update_suppress_provenance_from_stmts([deep_with], seed_prov, in_branch=False)
    assert out_prov.non_suppressors == frozenset()
    assert out_prov.imported_symbols == frozenset()
    assert "deep_mgr" in out_prov.suppressors
    assert "deep_mgr" in out_prov.shadowed_names


@pytest.mark.parametrize("cli_format", ["text", "json", "sarif"])
def test_cli_exit_code_matrix_and_inline_fallback_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_format: str,
) -> None:
    """Verifies CLI exit codes 0, 1, 2, 3, and 4 and ensures inline error fallbacks are distinct from helper code."""
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("def only_one():\n    return 1\n", encoding="utf-8")

    # Exit code 0: clean run with 0 clones
    assert main([str(clean_file), "--min-lines", "3", "--format", cli_format]) == 0
    capsys.readouterr()

    # Exit code 2: invalid config in pyproject.toml exits via parser.error (SystemExit(2))
    bad_cfg = tmp_path / "bad_pyproject.toml"
    bad_cfg.write_text(
        '[tool.pydoppelgangerhunt]\nclosure_strictness = "invalid_legacy_value"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as cfg_exc:
        main([str(clean_file), "--config", str(bad_cfg), "--format", cli_format])
    assert cfg_exc.value.code == 2
    capsys.readouterr()

    # Exit code 3: 0 clones + internal diagnostic error reported via callback
    def _html_err(*_args: Any, **kwargs: Any) -> str:
        cb = kwargs.get("on_diagnostic_error")
        if cb is not None:
            cb()
        return "<html>fallback</html>"

    html_clean = tmp_path / f"clean_{cli_format}.html"
    with monkeypatch.context() as mp:
        mp.setattr("pydoppelgangerhunt.cli.generate_html_report", _html_err)
        exit_clean_err = main(
            [str(clean_file), "--min-lines", "3", "--format", cli_format, "--html", str(html_clean)]
        )
    captured_clean = capsys.readouterr()
    assert exit_clean_err == 3
    assert "internal diagnostic error(s)" in captured_clean.err

    # Exit code 1 & 4 + inline fallback is never mistaken for a synthesized def _shared_ helper
    dup_file = tmp_path / "dup.py"
    dup_file.write_text(
        "def alpha(x):\n    a = x + 1\n    b = a * 2\n    return b + 3\n\n"
        "def beta(x):\n    a = x + 1\n    b = a * 2\n    return b + 3\n",
        encoding="utf-8",
    )
    assert main([str(dup_file), "--min-lines", "3", "--format", cli_format]) == 1
    capsys.readouterr()

    def _boom(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("synthetic synthesis failure")

    html_dup = tmp_path / f"dup_{cli_format}.html"
    with monkeypatch.context() as mp:
        mp.setattr("pydoppelgangerhunt.cli.synthesize_refactoring_suggestion", _boom)
        mp.setattr("pydoppelgangerhunt.reporters.synthesize_refactoring_suggestion", _boom)
        exit_dup_err = main(
            [str(dup_file), "--min-lines", "3", "--format", cli_format, "--html", str(html_dup), "--suggest"]
        )
    captured_dup = capsys.readouterr()
    assert exit_dup_err == 4
    html_text = html_dup.read_text(encoding="utf-8")
    assert "Diagnostic: pair rendering failed" in html_text
    assert "def _shared_" not in html_text
    if cli_format == "text":
        assert "Diagnostic: suggestion skipped due to" in captured_dup.out
        assert "def _shared_" not in captured_dup.out


def test_cli_config_closure_strictness_skip_warning_and_ctx_manager_help(
    tmp_path: Path,
    caplog: Any,
    capsys: Any,
) -> None:
    """Verifies configured closure_strictness conflicts, legacy rejection, and help distinguishing .Name from *.attr."""
    import logging

    src_file = tmp_path / "sample.py"
    src_file.write_text("def clean(x):\n    return x + 1\n", encoding="utf-8")

    cfg_pedantic = tmp_path / "pedantic.toml"
    cfg_pedantic.write_text(
        '[tool.pydoppelgangerhunt]\nclosure_strictness = "pedantic"\n',
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt"):
        exit_rc = main(
            [
                str(src_file),
                "--config",
                str(cfg_pedantic),
                "--skip-pre-unit-closures",
            ]
        )
    assert exit_rc == 0
    assert any(
        "overrides configured closure_strictness='pedantic'" in rec.getMessage()
        for rec in caplog.records
    )

    # Negative control: lenient config (already lenient) does not warn on --skip-pre-unit-closures
    cfg_lenient = tmp_path / "lenient.toml"
    cfg_lenient.write_text(
        '[tool.pydoppelgangerhunt]\nclosure_strictness = "lenient"\n',
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pydoppelgangerhunt"):
        exit_lenient = main(
            [
                str(src_file),
                "--config",
                str(cfg_lenient),
                "--skip-pre-unit-closures",
            ]
        )
    assert exit_lenient == 0
    assert not any(
        "overrides configured closure_strictness=" in rec.getMessage()
        for rec in caplog.records
    )

    # Removed legacy aliases in TOML fail strictly at CLI boundary
    cfg_legacy = tmp_path / "legacy_fail_closed.toml"
    cfg_legacy.write_text(
        '[tool.pydoppelgangerhunt]\nclosure_strictness = "fail_closed"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        main([str(src_file), "--config", str(cfg_legacy), "--skip-pre-unit-closures"])
    assert exc_info.value.code == 2
    assert "Invalid closure_strictness value 'fail_closed'" in capsys.readouterr().err

    # Help string accurately distinguishes .Name relative imports from *.attr suffix wildcards
    from pydoppelgangerhunt.cli import build_arg_parser

    help_text = " ".join(build_arg_parser().format_help().split())
    assert "'.Name' relative imports" in help_text
    assert "'*.attr' suffix wildcards" in help_text
    assert "'*.attr' / '.attr'" not in help_text

    # Behavioral check: '.lock' matches `from . import lock`, NOT `obj.lock()` (which requires '*.lock')
    attr_src = (
        "def run(obj, step):\n"
        "    with obj.lock():\n"
        "        v = step()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    u_attr = {
        "start": 4,
        "end": 4,
        "kind": "statement",
        "source_text": attr_src,
        "closure_strictness": "pedantic",
        "non_suppressing_ctx_managers": [".lock"],
    }
    sc_dot_name = inspect_single_unit_scope(u_attr)
    assert sc_dot_name["is_control_flow_safe"] is False

    sc_star_attr = inspect_single_unit_scope(
        {**u_attr, "non_suppressing_ctx_managers": ["*.lock"]}
    )
    assert sc_star_attr["is_control_flow_safe"] is True

    rel_src = (
        "from . import lock\n"
        "def run(step):\n"
        "    with lock():\n"
        "        v = step()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
    )
    sc_rel = inspect_single_unit_scope(
        {
            "start": 5,
            "end": 5,
            "kind": "statement",
            "source_text": rel_src,
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": [".lock"],
        }
    )
    assert sc_rel["is_control_flow_safe"] is True


def test_unrecognized_caller_strictness_preview_mode_enforces_strict_over_lenient_unit(
    tmp_path: Path,
) -> None:
    """Verifies unrecognized caller closure_strictness in replace_clones=False enforces 'strict' fallback over unit 'lenient'."""
    src_closure = (
        "def g1(items: list[int]):\n"
        "    cb = lambda: extra\n"
        "    for x in items:\n"
        "        yield x\n"
        "        extra = x\n"
        "    return 0\n\n"
        "def g2(items: list[int]):\n"
        "    for y in items:\n"
        "        yield y\n"
        "        extra = y\n"
        "    return 0\n"
    )
    mod_closure = tmp_path / "closure_pair.py"
    mod_closure.write_text(src_closure, encoding="utf-8")
    u1_lenient = {
        "file": str(mod_closure),
        "start": 3,
        "end": 5,
        "name": "g1:block",
        "kind": "compound_block",
        "closure_strictness": "lenient",
    }
    u2_lenient = {
        "file": str(mod_closure),
        "start": 9,
        "end": 11,
        "name": "g2:block",
        "kind": "compound_block",
        "closure_strictness": "lenient",
    }

    clear_dataflow_cache()
    clear_closure_warning_cache()
    # Control: explicit lenient works as expected in preview mode
    preview_lenient = generate_refactoring_patch(
        [(1.0, u1_lenient, u2_lenient)],
        repo_root=str(tmp_path),
        replace_clones=False,
        closure_strictness="lenient",
    )
    assert "yield from" in preview_lenient

    # Unrecognized caller strictness in replace_clones=False must fall back to 'strict'
    # rather than letting u1['closure_strictness'] = 'lenient' weaken it.
    clear_dataflow_cache()
    clear_closure_warning_cache()
    preview_unrecognized = generate_refactoring_patch(
        [(1.0, u1_lenient, u2_lenient)],
        repo_root=str(tmp_path),
        replace_clones=False,
        closure_strictness="unrecognized_policy_xyz",
    )
    assert preview_unrecognized == ""

    # Confirm the fallback in replace_clones=False is 'strict' (allows heuristic unknown ctx mgr preview),
    # unlike replace_clones=True on unset units which preserves 'pedantic' default.
    tx_src = (
        "def alpha(compute, scale):\n"
        "    with UnknownTxGuard():\n"
        "        val = compute(scale)\n"
        "        yield val\n"
        "        extra = val + 1\n"
        "    yield extra * 2\n\n"
        "def beta(compute, scale):\n"
        "    with UnknownTxGuard():\n"
        "        val = compute(scale)\n"
        "        yield val\n"
        "        extra = val + 1\n"
        "    yield extra * 2\n"
    )
    mod_tx = tmp_path / "tx_pair.py"
    mod_tx.write_text(tx_src, encoding="utf-8")
    u_tx1 = {"file": str(mod_tx), "start": 2, "end": 5, "name": "alpha:block", "kind": "compound_block"}
    u_tx2 = {"file": str(mod_tx), "start": 9, "end": 12, "name": "beta:block", "kind": "compound_block"}
    clear_dataflow_cache()
    clear_closure_warning_cache()
    tx_preview_fallback = generate_refactoring_patch(
        [(1.0, u_tx1, u_tx2)],
        repo_root=str(tmp_path),
        replace_clones=False,
        closure_strictness="unrecognized_policy_xyz",
    )
    assert "def _shared_" in tx_preview_fallback
    clear_dataflow_cache()
    clear_closure_warning_cache()
    tx_rewrite_fallback = generate_refactoring_patch(
        [(1.0, u_tx1, u_tx2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        closure_strictness="unrecognized_policy_xyz",
    )
    assert tx_rewrite_fallback == ""

