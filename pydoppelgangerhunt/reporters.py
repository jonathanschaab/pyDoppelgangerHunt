"""Reporting, formatters (SARIF, JSON, Markdown), unified diffing, and ANSI terminal color."""

from __future__ import annotations

import ast
from collections import OrderedDict
import difflib
import html
import json
import logging
import os
from pathlib import Path
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.config import (
    APP_VERSION,
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    format_ctx_manager_assumption_comments,
    heuristic_ctx_manager_session,
    normalize_heuristic_ctx_managers,
    normalize_path_string,
    normalize_wildcard_ctx_manager_matches,
    record_heuristic_non_suppressing_ctx_manager,
    record_wildcard_ctx_manager_match,
)
from pydoppelgangerhunt.source_lines import (
    extract_unit_source_code,
    parse_unit_coord,
    resolve_safe_unit_file_path,
    resolve_unit_line_bounds,
    split_source_lines,
    unit_file_str,
    with_unit_source_text,
)
from pydoppelgangerhunt.source_lines import (  # pylint: disable=unused-import,useless-import-alias
    extract_unit_source_lines_with_state as extract_unit_source_lines_with_state,
)
from pydoppelgangerhunt.unit_boundary import (
    normalize_unit_boundary,
    resolve_unit_source_buffer,
    unit_has_inline_source_buffers,
)
from pydoppelgangerhunt.fixer._df_cache import _with_analysis_session
from pydoppelgangerhunt.fixer._pair_eval import (
    classify_pair_rejection_reasons as _collect_suggestion_rejection_reasons,
    evaluate_clone_pair_scopes,
)

logger = logging.getLogger(__name__)


class _UnsetType:
    """Sentinel distinguishing omitted preloaded source/tree parameters from explicit None."""


_UNSET = _UnsetType()


_unit_has_inline_source_buffers = unit_has_inline_source_buffers


def _extract_unit_lines_for_report(
    unit: Dict[str, Any],
    repo_root: Optional[str],
    source_text: Optional[str],
    *,
    preloaded: bool,
) -> List[str]:
    """Slices report lines for unit, honoring preloaded disk misses without re-probing disk."""
    raw_file = unit_file_str(unit).strip()
    if (
        preloaded
        and source_text is None
        and not _unit_has_inline_source_buffers(unit)
        and not raw_file.lower().endswith(".ipynb")
    ):
        if ".ipynb#" in raw_file.lower():
            return []
        s_d, e_d = _unit_line_bounds(unit)
        n_d = str(unit.get("name") or "unit")
        return [f"# Source for {n_d} lines {s_d}-{e_d}\n"]
    return extract_unit_source_code(with_unit_source_text(unit, source_text), repo_root)


def _safe_unit_coord(unit: Dict[str, Any], key: str) -> Optional[int]:
    """Safely extracts and parses an integer coordinate, returning None on failure."""
    try:
        return parse_unit_coord(unit, key, default=None)
    except (ValueError, TypeError):
        return None


# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"


def supports_color(color_override: Optional[bool] = None) -> bool:
    """Determines whether the terminal environment supports ANSI colors."""
    if color_override is not None:
        return color_override
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def colorize(text: str, color_code: str, enabled: bool) -> str:
    """Wraps text in ANSI color escape codes if color is enabled."""
    if not enabled:
        return text
    return f"{color_code}{text}{COLOR_RESET}"


def _unit_line_bounds(unit: Dict[str, Any]) -> Tuple[int, int]:
    """Extracts parsed (start, end) line coordinates from a unit dictionary."""
    return resolve_unit_line_bounds(unit)


def _safe_unit_line_bounds_fallback(unit: Any) -> Tuple[int, int]:
    """Extracts sanitized (start, end) line bounds without raising on invalid metadata."""
    if not isinstance(unit, dict):
        return 1, 1
    raw_start = _safe_unit_coord(unit, "start")
    raw_end = _safe_unit_coord(unit, "end")
    start_val = max(1, raw_start) if raw_start is not None else 1
    end_val = max(start_val, raw_end) if raw_end is not None else start_val
    return start_val, end_val


def generate_clone_diff(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    color: bool = False,
    *,
    source_text1: Union[Optional[str], _UnsetType] = _UNSET,
    source_text2: Union[Optional[str], _UnsetType] = _UNSET,
) -> str:
    """Produces unified line diff between two cloned code blocks, with optional syntax coloring."""
    lines1 = (
        extract_unit_source_code(u1, repo_root)
        if isinstance(source_text1, _UnsetType)
        else _extract_unit_lines_for_report(u1, repo_root, source_text1, preloaded=True)
    )
    lines2 = (
        extract_unit_source_code(u2, repo_root)
        if isinstance(source_text2, _UnsetType)
        else _extract_unit_lines_for_report(u2, repo_root, source_text2, preloaded=True)
    )
    norm1 = normalize_unit_boundary(u1)
    norm2 = normalize_unit_boundary(u2)
    f1_norm = normalize_path_string(norm1.file_path, strip_anchor=False)
    f2_norm = normalize_path_string(norm2.file_path, strip_anchor=False)
    s1 = norm1.coords.start if norm1.coords.start is not None else 1
    e1 = norm1.coords.end if norm1.coords.end is not None else s1
    s2 = norm2.coords.start if norm2.coords.start is not None else 1
    e2 = norm2.coords.end if norm2.coords.end is not None else s2
    n1 = norm1.name or "unit1"
    n2 = norm2.name or "unit2"
    from_label = f"{f1_norm}:{s1}-{e1} ({n1})"
    to_label = f"{f2_norm}:{s2}-{e2} ({n2})"
    diff = list(difflib.unified_diff(
        lines1, lines2, fromfile=from_label, tofile=to_label, lineterm=""
    ))

    if not color:
        return "\n".join(diff)

    colored_diff: List[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            colored_diff.append(colorize(line, COLOR_BOLD + COLOR_CYAN, True))
        elif line.startswith("@@"):
            colored_diff.append(colorize(line, COLOR_CYAN, True))
        elif line.startswith("-"):
            colored_diff.append(colorize(line, COLOR_RED, True))
        elif line.startswith("+"):
            colored_diff.append(colorize(line, COLOR_GREEN, True))
        else:
            colored_diff.append(line)
    return "\n".join(colored_diff)


_MAX_REPORT_FILE_CACHE_ENTRIES = 256


def _parse_inline_source_text(
    src_text: str,
    parse_fn: Callable[[str], Optional[ast.AST]],
) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Parses non-blank inline source text into (source_text, tree_or_None)."""
    if not src_text.strip():
        return None, None
    try:
        return src_text, parse_fn(src_text)
    except (SyntaxError, ValueError, RecursionError):
        return src_text, None


class _ReportFileCacheKey(NamedTuple):
    """Typed key distinguishing cached missing-file lookups from resolved disk/notebook paths."""

    is_missing: bool
    scope: str
    path: str


def resolve_cached_unit_source_and_tree(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_cache: Optional[
        OrderedDict[Any, Tuple[Optional[str], Optional[ast.AST]]]
    ] = None,
) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Loads and caches (source_text, parsed_ast) for a unit across multi-pair reporting passes."""
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        _get_or_parse_tree,
    )
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _is_sliced_excerpt_for_file,
        _load_unit_file_text,
    )

    buf = resolve_unit_source_buffer(
        unit,
        include_lines=False,
        is_sliced_lines_fn=_is_sliced_excerpt_for_file,
    )
    if buf.is_invalid_notebook:
        return None, None
    if buf.full_text:
        return _parse_inline_source_text(buf.full_text, _get_or_parse_tree)
    if buf.blocked_by_lines:
        return None, None

    raw_file = buf.file_path
    if not raw_file:
        return None, None

    raw_miss_key: Optional[_ReportFileCacheKey] = None
    if file_cache is not None:
        raw_miss_key = _ReportFileCacheKey(
            True,
            repo_root or "",
            normalize_path_string(raw_file, strip_anchor=False),
        )
        if raw_miss_key in file_cache:
            file_cache.move_to_end(raw_miss_key)
            return file_cache[raw_miss_key]

    cache_key: Optional[_ReportFileCacheKey] = None
    resolved_p = resolve_safe_unit_file_path(unit, repo_root=repo_root)
    if resolved_p is None:
        miss_entry: Tuple[Optional[str], Optional[ast.AST]] = (None, None)
        if file_cache is not None and raw_miss_key is not None:
            file_cache[raw_miss_key] = miss_entry
            if len(file_cache) > _MAX_REPORT_FILE_CACHE_ENTRIES:
                file_cache.popitem(last=False)
        return miss_entry
    if file_cache is not None:
        anchor = raw_file.split("#", 1)[1] if "#" in raw_file else ""
        cache_key = _ReportFileCacheKey(False, anchor, str(resolved_p))
        if cache_key in file_cache:
            file_cache.move_to_end(cache_key)
            return file_cache[cache_key]

    disk_unit = {
        k: v
        for k, v in unit.items()
        if k
        not in (
            "source_text",
            "file_source",
            "source_lines",
            "source_lines_is_sliced",
        )
    }
    src_text = _load_unit_file_text(disk_unit, repo_root=repo_root)
    parsed_tree: Optional[ast.AST] = (
        _get_or_parse_tree(src_text)
        if isinstance(src_text, str) and src_text.strip()
        else None
    )
    entry = (src_text, parsed_tree)
    if file_cache is not None and cache_key is not None:
        file_cache[cache_key] = entry
        if len(file_cache) > _MAX_REPORT_FILE_CACHE_ENTRIES:
            file_cache.popitem(last=False)
    return entry


class _PreparedSuggestionUnit(NamedTuple):
    """Resolved unit dictionary, backing module source/AST, and sliced lines for suggestion synthesis."""

    unit: Dict[str, Any]
    source_text: Optional[str]
    tree: Optional[ast.AST]
    lines: List[str]
    is_preloaded_miss: bool = False


def _prepare_effective_suggestion_unit(
    unit: Dict[str, Any],
    *,
    repo_root: Optional[str],
    source_text: Union[Optional[str], _UnsetType],
    tree: Union[Optional[ast.AST], _UnsetType],
) -> _PreparedSuggestionUnit:
    """Copies unit, attaches resolved full-module source_text when not inline, and slices unit lines."""
    was_preloaded = not isinstance(source_text, _UnsetType) or not isinstance(
        tree, _UnsetType
    )
    if was_preloaded:
        eff_src: Optional[str] = None if isinstance(source_text, _UnsetType) else source_text
        eff_tree: Optional[ast.AST] = None if isinstance(tree, _UnsetType) else tree
    else:
        eff_src, eff_tree = resolve_cached_unit_source_and_tree(
            unit, repo_root=repo_root
        )
    is_preloaded_miss = (
        was_preloaded
        and eff_src is None
        and eff_tree is None
        and not _unit_has_inline_source_buffers(unit)
    )
    u_eff = with_unit_source_text(unit, eff_src)
    if u_eff is unit:
        u_eff = dict(unit)
    return _PreparedSuggestionUnit(
        unit=u_eff,
        source_text=eff_src,
        tree=eff_tree,
        lines=_extract_unit_lines_for_report(
            u_eff, repo_root, eff_src, preloaded=was_preloaded
        ),
        is_preloaded_miss=is_preloaded_miss,
    )


class _RejectionScopeContext(NamedTuple):
    """Adjusted scopes and pair skip reason for computing suggestion rejection causes."""

    scope1: Dict[str, Any]
    scope2: Dict[str, Any]
    pair_skip_reason: Optional[str]


def _resolve_rejection_scope_context(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
    lines1: Sequence[str],
    lines2: Sequence[str],
    *,
    is_expr_unit: bool,
    pair_skip_reason: Optional[str],
    prune_lhs_fn: Callable[[Dict[str, Any], Dict[str, Any], Sequence[str]], Any],
    superseded_skip_reasons: Tuple[str, ...],
) -> _RejectionScopeContext:
    """Prunes expression LHS-only scope state and clears superseded output-pairing skip reasons."""
    if not is_expr_unit:
        return _RejectionScopeContext(sc1, sc2, pair_skip_reason)
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        _copy_scope_dict,
    )

    pruned1 = _copy_scope_dict(sc1)
    pruned2 = _copy_scope_dict(sc2)
    prune_lhs_fn(_copy_scope_dict(u1), pruned1, lines1)
    prune_lhs_fn(_copy_scope_dict(u2), pruned2, lines2)
    effective_reason = (
        None
        if (
            pair_skip_reason in superseded_skip_reasons
            and not pruned1.get("outputs")
            and not pruned2.get("outputs")
        )
        else pair_skip_reason
    )
    return _RejectionScopeContext(pruned1, pruned2, effective_reason)


def _format_synthesized_helper_suggestion(
    helper_name: str,
    diff_summary: str,
    helper_code: str,
    *,
    pair_heuristics: Set[str],
    pair_wildcards: Set[Tuple[str, str]],
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
) -> str:
    """Formats an actionable extracted-helper suggestion and records context-manager telemetry."""
    effective_heuristics = normalize_heuristic_ctx_managers(
        pair_heuristics,
        sc1.get("heuristic_ctx_managers"),
        sc2.get("heuristic_ctx_managers"),
    )
    effective_wildcards = normalize_wildcard_ctx_manager_matches(
        pair_wildcards,
        sc1.get("wildcard_ctx_managers"),
        sc2.get("wildcard_ctx_managers"),
    )
    for mgr_name in effective_heuristics:
        record_heuristic_non_suppressing_ctx_manager(mgr_name)
    for pat, mgr_name in effective_wildcards:
        record_wildcard_ctx_manager_match(pat, mgr_name)
    note_comment = format_ctx_manager_assumption_comments(
        effective_heuristics, effective_wildcards
    )
    note_line = (
        "".join(
            f"    {ln.rstrip(chr(13) + chr(10))}\n"
            for ln in split_source_lines(note_comment)
        )
        if note_comment
        else ""
    )
    indented_helper = "\n".join(
        f"    {ln}" if ln.strip() else ""
        for ln in (
            raw_ln.rstrip("\r\n")
            for raw_ln in split_source_lines(helper_code.rstrip("\n"))
        )
    )
    return (
        f"[REFACTOR SUGGESTION] Extract duplicated logic into common helper '{helper_name}':\n"
        f"    # {diff_summary}\n"
        f"{note_line}"
        f"{indented_helper}"
    )


def _format_rejected_suggestion(
    p1: _PreparedSuggestionUnit,
    p2: _PreparedSuggestionUnit,
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
    *,
    n1: str,
    n2: str,
    helper_name: str,
    diff_summary: str,
    repo_root: Optional[str],
    is_expr_unit: bool,
    kinds_aligned: bool,
    pair_outputs: Optional[Tuple[List[str], List[str]]],
    pair_skip_reason: Optional[str],
    has_nonterminal_embedded_return: bool,
    synthesis_failure_reason: Optional[str],
) -> str:
    """Computes rejection reasons and formats a safety-hazard or structural-incompatibility suggestion."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        PairSkipReason,
        allows_u2_source_placeholder,
        has_async_generator_delegation_hazard,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        is_subroutine_unit,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _prune_unreferenced_lhs_scope,
    )

    has_async_gen_hazard = has_async_generator_delegation_hazard(
        sc1,
        sc2,
        has_outputs=bool(pair_outputs and (pair_outputs[0] or pair_outputs[1])),
    )
    rej_ctx = _resolve_rejection_scope_context(
        p1.unit,
        p2.unit,
        sc1,
        sc2,
        p1.lines,
        p2.lines,
        is_expr_unit=is_expr_unit,
        pair_skip_reason=pair_skip_reason,
        prune_lhs_fn=_prune_unreferenced_lhs_scope,
        superseded_skip_reasons=(
            PairSkipReason.OUTPUT_ARITY_MISMATCH.value,
            PairSkipReason.PERMUTED_OUTPUTS.value,
            PairSkipReason.UNPAIRED_OUTPUTS.value,
        ),
    )
    u2_allows_placeholder = False
    u2_has_placeholder = (
        "unavailable_source_placeholder" in rej_ctx.scope2.get("control_flow_hazards", ())
        or "unavailable_source_placeholder" in rej_ctx.scope2.get("closure_hazard_reasons", ())
    )
    u1_has_placeholder = (
        "unavailable_source_placeholder" in rej_ctx.scope1.get("control_flow_hazards", ())
        or "unavailable_source_placeholder" in rej_ctx.scope1.get("closure_hazard_reasons", ())
    )
    if u2_has_placeholder and not u1_has_placeholder:
        is_sub1 = is_subroutine_unit(p1.unit, source_text=p1.source_text, tree=p1.tree)
        is_sub2 = is_subroutine_unit(p2.unit, source_text=p2.source_text, tree=p2.tree)
        u2_allows_placeholder = is_sub1 == is_sub2 and allows_u2_source_placeholder(
            is_subroutine=is_sub1,
            requires_u2_edit=False,
            source_text2=p2.source_text,
            u2=p2.unit,
            repo_root=repo_root,
        )
    rejection = _collect_suggestion_rejection_reasons(
        p1.unit,
        p2.unit,
        rej_ctx.scope1,
        rej_ctx.scope2,
        pair_skip_reason=rej_ctx.pair_skip_reason,
        has_async_gen_hazard=has_async_gen_hazard,
        has_nonterminal_embedded_return=has_nonterminal_embedded_return,
        synthesis_failure_reason=synthesis_failure_reason,
        kinds_aligned=kinds_aligned,
        u2_allows_placeholder=u2_allows_placeholder,
    )
    reason_suffix = f" [{', '.join(rejection.reasons)}]" if rejection.reasons else ""
    if rejection.has_safety_hazard:
        return (
            f"[REFACTOR SUGGESTION] Automatic extraction into '{helper_name}' rejected as unsafe "
            f"(safety hazard detected for {n1} and {n2}{reason_suffix}); manual review required:\n"
            f"    # {diff_summary}\n"
            f"    # WARNING: Automatic helper synthesis was skipped due to a safety hazard{reason_suffix}."
        )
    return (
        f"[REFACTOR SUGGESTION] Automatic extraction into '{helper_name}' skipped as structurally "
        f"incompatible (incompatible pair structure or outputs for {n1} and {n2}{reason_suffix}); "
        f"manual review required:\n"
        f"    # {diff_summary}\n"
        f"    # WARNING: Automatic helper synthesis was skipped due to structural incompatibility{reason_suffix}."
    )


@_with_analysis_session
def synthesize_refactoring_suggestion(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    source_text1: Union[Optional[str], _UnsetType] = _UNSET,
    source_text2: Union[Optional[str], _UnsetType] = _UNSET,
    tree1: Union[Optional[ast.AST], _UnsetType] = _UNSET,
    tree2: Union[Optional[ast.AST], _UnsetType] = _UNSET,
) -> str:
    """Synthesizes actionable refactoring recommendation and shared helper template."""
    from pydoppelgangerhunt.fixer.source import (  # pylint: disable=import-outside-toplevel
        lines_end_with_unconditional_return,
        validate_unit_schema,
    )

    validate_unit_schema(u1)
    validate_unit_schema(u2)
    n1 = str(u1.get("name") or "unit1").rsplit(":", maxsplit=1)[-1]
    n2 = str(u2.get("name") or "unit2").rsplit(":", maxsplit=1)[-1]

    if n1.startswith("test_") and n2.startswith("test_"):
        return (
            f"[REFACTOR SUGGESTION] Combine '{n1}' and '{n2}' into a single parametrized test:\n"
            f"    @pytest.mark.parametrize(\"param1, param2, expected\", [\n"
            f"        (..., ..., ...),  # Test case from {n1}\n"
            f"        (..., ..., ...),  # Test case from {n2}\n"
            f"    ])\n"
            f"    def test_shared_behavior(param1, param2, expected):\n"
            f"        ..."
        )

    p1, p2 = (
        _prepare_effective_suggestion_unit(
            u_item, repo_root=repo_root, source_text=s_item, tree=t_item
        )
        for u_item, s_item, t_item in (
            (u1, source_text1, tree1),
            (u2, source_text2, tree2),
        )
    )

    shared_prefix = ""
    for c1, c2 in zip(n1, n2):
        if c1 == c2:
            shared_prefix += c1
        else:
            break
    clean_prefix = shared_prefix.strip("_")
    helper_name = f"_shared_{clean_prefix}" if clean_prefix else "_shared_helper"

    diff_lines = [
        line for line in difflib.ndiff(p1.lines, p2.lines)
        if line.startswith("- ") or line.startswith("+ ")
    ]
    diff_summary = f"{len(diff_lines)} varying line(s) detected between clone blocks."

    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        SAFETY_PAIR_SKIP_REASONS,
        CloneOutputResolution,
        PairSkipReason,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        resolve_unit_inferred_col_bounds,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _align_pair_expr_kinds,
        synthesize_shared_helper_code,
    )

    kinds_aligned = _align_pair_expr_kinds(p1.unit, p2.unit)
    p1 = p1._replace(
        unit=resolve_unit_inferred_col_bounds(
            p1.unit,
            repo_root=repo_root,
            unit_code_lines=p1.lines if p1.is_preloaded_miss else None,
            is_placeholder_origin=p1.is_preloaded_miss,
        ).unit
    )
    p2 = p2._replace(
        unit=resolve_unit_inferred_col_bounds(
            p2.unit,
            repo_root=repo_root,
            unit_code_lines=p2.lines if p2.is_preloaded_miss else None,
            is_placeholder_origin=p2.is_preloaded_miss,
        ).unit
    )
    is_expr_unit = kinds_aligned and p1.unit.get("kind") in (
        "comprehension",
        "complex_expr",
    )

    with collect_heuristic_ctx_managers(isolate=True) as pair_heuristics:
        with collect_wildcard_ctx_managers(isolate=True) as pair_wildcards:
            scope_eval = evaluate_clone_pair_scopes(
                p1.unit,
                p2.unit,
                closure_strictness=closure_strictness,
                non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                default_strictness="strict",
                source_text1=p1.source_text,
                source_text2=p2.source_text,
                tree1=p1.tree,
                tree2=p2.tree,
                repo_root=repo_root,
                allow_disk_read1=not p1.is_preloaded_miss,
                allow_disk_read2=not p2.is_preloaded_miss,
            )
            effective_pair_strictness = scope_eval.policy.effective_mode
            sc1 = scope_eval.scope1
            sc2 = scope_eval.scope2
            pair_heuristics.update(scope_eval.heuristics)
            pair_wildcards.update(scope_eval.wildcards)
            combined_hazards = (
                *sc1.get("control_flow_hazards", ()),
                *sc2.get("control_flow_hazards", ()),
            )
            has_naked_loop_jump = any(
                h in ("naked_break", "naked_continue") for h in combined_hazards
            )
            has_nonterminal_embedded_return = (
                "embedded_return" in combined_hazards
                and not (
                    lines_end_with_unconditional_return(p1.lines)
                    and lines_end_with_unconditional_return(p2.lines)
                )
            )
            pair_outputs: Optional[Tuple[List[str], List[str]]] = None
            pair_skip_reason: Optional[str] = None
            if kinds_aligned:
                pair_out_raw = resolve_clone_pair_outputs(
                    u1=p1.unit,
                    u2=p2.unit,
                    scope1=sc1,
                    scope2=sc2,
                    source_text1=p1.source_text,
                    source_text2=p2.source_text,
                    tree1=p1.tree,
                    tree2=p2.tree,
                    repo_root=repo_root,
                    closure_strictness=effective_pair_strictness,
                    non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                    return_details=True,
                )
                pair_outputs = (
                    pair_out_raw.outputs
                    if isinstance(pair_out_raw, CloneOutputResolution)
                    else pair_out_raw
                )
                pair_skip_reason = (
                    pair_out_raw.skip_reason
                    if isinstance(pair_out_raw, CloneOutputResolution)
                    else None
                )

            helper_code = ""
            synthesis_failure_reason: Optional[str] = None
            if (
                kinds_aligned
                and not has_naked_loop_jump
                and not has_nonterminal_embedded_return
                and pair_skip_reason not in SAFETY_PAIR_SKIP_REASONS
                and (pair_outputs is not None or is_expr_unit)
            ):
                try:
                    helper_code = synthesize_shared_helper_code(
                        p1.unit,
                        p2.unit,
                        helper_name=helper_name,
                        repo_root=repo_root,
                        source_text1=p1.source_text,
                        source_text2=p2.source_text,
                        tree1=p1.tree,
                        tree2=p2.tree,
                        closure_strictness=effective_pair_strictness,
                        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                        outputs=None if is_expr_unit or pair_outputs is None else pair_outputs[0],
                        outputs2=None if is_expr_unit or pair_outputs is None else pair_outputs[1],
                        scope1=sc1,
                        scope2=sc2,
                    )
                except SyntaxError as exc:
                    logger.debug(
                        "Failed to synthesize refactoring suggestion for (%s, %s): %s",
                        n1,
                        n2,
                        exc,
                        exc_info=True,
                    )
                    synthesis_failure_reason = "unparseable or invalid syntax"
                except (OSError, UnicodeDecodeError) as exc:
                    logger.debug(
                        "Failed to synthesize refactoring suggestion for (%s, %s): %s",
                        n1,
                        n2,
                        exc,
                        exc_info=True,
                    )
                    synthesis_failure_reason = "source file read or decode failure"
                except RecursionError as exc:
                    logger.debug(
                        "Failed to synthesize refactoring suggestion for (%s, %s): %s",
                        n1,
                        n2,
                        exc,
                        exc_info=True,
                    )
                    synthesis_failure_reason = PairSkipReason.PATHOLOGICAL_AST.value
    if helper_code:
        return _format_synthesized_helper_suggestion(
            helper_name,
            diff_summary,
            helper_code,
            pair_heuristics=pair_heuristics,
            pair_wildcards=pair_wildcards,
            sc1=sc1,
            sc2=sc2,
        )
    return _format_rejected_suggestion(
        p1,
        p2,
        sc1,
        sc2,
        n1=n1,
        n2=n2,
        helper_name=helper_name,
        diff_summary=diff_summary,
        repo_root=repo_root,
        is_expr_unit=is_expr_unit,
        kinds_aligned=kinds_aligned,
        pair_outputs=pair_outputs,
        pair_skip_reason=pair_skip_reason,
        has_nonterminal_embedded_return=has_nonterminal_embedded_return,
        synthesis_failure_reason=synthesis_failure_reason,
    )


def format_sarif_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
) -> Dict[str, Any]:
    """Formats detected clones into OASIS SARIF 2.1.0 standard schema for GitHub Code Scanning."""
    results: List[Dict[str, Any]] = []
    for idx, (sim, u1, u2) in enumerate(clones):
        f1_norm = normalize_path_string(unit_file_str(u1), strip_anchor=False)
        f2_norm = normalize_path_string(unit_file_str(u2), strip_anchor=False)
        s1, e1 = _unit_line_bounds(u1)
        s2, e2 = _unit_line_bounds(u2)
        n1 = str(u1.get("name") or "unit1")
        n2 = str(u2.get("name") or "unit2")
        rule_id = "PYDOPPEL001"
        message = (
            f"Code duplication: AST block '{n1}' in {f1_norm}:{s1}-{e1} "
            f"is {sim:.1%} structurally identical to '{n2}' in {f2_norm}:{s2}-{e2} "
            f"(threshold >= {threshold:.0%})."
        )
        r1: Dict[str, Any] = {
            "startLine": s1,
            "endLine": e1,
        }
        sc1 = _safe_unit_coord(u1, "start_col")
        if sc1 is not None:
            r1["startColumn"] = max(1, sc1 + 1)
        ec1 = _safe_unit_coord(u1, "end_col")
        if ec1 is not None:
            r1["endColumn"] = max(1, ec1 + 1)

        r2: Dict[str, Any] = {
            "startLine": s2,
            "endLine": e2,
        }
        sc2 = _safe_unit_coord(u2, "start_col")
        if sc2 is not None:
            r2["startColumn"] = max(1, sc2 + 1)
        ec2 = _safe_unit_coord(u2, "end_col")
        if ec2 is not None:
            r2["endColumn"] = max(1, ec2 + 1)

        result_item: Dict[str, Any] = {
            "ruleId": rule_id,
            "ruleIndex": 0,
            "level": "warning" if sim < 0.95 else "error",
            "message": {"text": message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f1_norm,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": r1,
                    }
                }
            ],
            "relatedLocations": [
                {
                    "id": idx + 1,
                    "message": {"text": f"Clone twin: '{n2}' in {f2_norm}:{s2}-{e2}"},
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": f2_norm,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": r2,
                    },
                }
            ],
            "properties": {
                "similarity": round(sim, 4),
                "cloneKind": u1.get("kind", "function"),
            },
        }
        results.append(result_item)

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "pyDoppelgangerHunt",
                        "version": APP_VERSION,
                        "informationUri": "https://github.com/jonathanschaab/pyDoppelgangerHunt",
                        "rules": [
                            {
                                "id": "PYDOPPEL001",
                                "name": "StructuralCodeClone",
                                "shortDescription": {"text": "Structural code duplication detected across AST units."},
                                "fullDescription": {
                                    "text": (
                                        "A structural AST code clone was detected that violates DRY principles. "
                                        "Refactor the common logic into a shared helper function or base model."
                                    )
                                },
                                "defaultConfiguration": {"level": "warning"},
                            }
                        ],
                    }
                },
                "results": results,
                "properties": {
                    "target": target,
                    "threshold": threshold,
                    "cloneCount": len(clones),
                },
            }
        ],
    }


def _format_unit_json(
    unit: Dict[str, Any], default_name: str, include_tokens: bool = False
) -> Dict[str, Any]:
    """Formats an AST unit into a structured JSON dictionary."""
    s, e = _unit_line_bounds(unit)
    data: Dict[str, Any] = {
        "name": str(unit.get("name") or default_name),
        "file": normalize_path_string(unit_file_str(unit), strip_anchor=False),
        "start": s,
        "end": e,
        "kind": unit.get("kind"),
    }
    if include_tokens:
        data["tokens"] = unit.get("token_count", 0)
    return data


def format_json_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
    families: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates structured JSON representation of clone matches."""
    data: Dict[str, Any] = {
        "tool": "pyDoppelgangerHunt",
        "version": APP_VERSION,
        "target": target,
        "threshold": threshold,
        "clone_count": len(clones),
        "clones": [
            {
                "similarity": round(sim, 4),
                "unit_a": _format_unit_json(u1, "unit1", include_tokens=True),
                "unit_b": _format_unit_json(u2, "unit2", include_tokens=True),
            }
            for sim, u1, u2 in clones
        ],
    }
    if families is not None:
        data["families"] = [
            {
                "family_id": f["family_id"],
                "member_count": f["member_count"],
                "unique_files": f["unique_files"],
                "avg_similarity": round(f["avg_similarity"], 4),
                "max_similarity": round(f["max_similarity"], 4),
                "min_similarity": round(f.get("min_similarity", f["avg_similarity"]), 4),
                "coherence": round(f.get("coherence", 1.0), 4),
                "total_lines": f["total_lines"],
                "medoid": (
                    _format_unit_json(f["medoid"], "medoid")
                    if "medoid" in f and f["medoid"]
                    else None
                ),
                "members": [
                    _format_unit_json(m, "member")
                    for m in f["members"]
                ],
            }
            for f in families
        ]
    if stats is not None:
        data["stats"] = stats
    return data


def format_markdown_summary(stats: Dict[str, Any], target: str) -> str:
    """Formats GitHub Step Summary Markdown report."""
    clean_target = str(target).replace("`", "'")
    dry_score = float(stats.get("dry_score", 100.0))
    grade = str(stats.get("grade", "A+"))
    sloc = int(stats.get("sloc", 0))
    dloc = int(stats.get("dloc", 0))
    dup_pct = float(stats.get("duplication_pct", 0.0))
    clone_pairs = int(stats.get("clone_pairs", 0))
    clone_fams = int(stats.get("clone_families", 0))

    md_lines: List[str] = [
        "## pyDoppelgangerHunt DRY Quality Audit Summary",
        "",
        f"**Audit Target:** `{clean_target}`",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Repository DRY Score** | **{dry_score:.1f}% (Grade: {grade})** |",
        f"| **Total Source Lines (SLOC)** | {sloc:,} |",
        f"| **Duplicated Lines (DLOC)** | {dloc:,} |",
        f"| **Duplication Rate** | {dup_pct:.2f}% |",
        f"| **Total Clone Pairs** | {clone_pairs} |",
        f"| **Clone Families** | {clone_fams} |",
        "",
    ]
    if stats.get("package_sloc"):
        md_lines.append("### Package SLOC Breakdown")
        md_lines.append("| Package / Directory | SLOC |")
        md_lines.append("| :--- | :--- |")
        for pkg, lines in sorted(stats["package_sloc"].items(), key=lambda x: x[1], reverse=True):
            clean_pkg = str(pkg).replace("|", "\\|").replace("`", "'")
            md_lines.append(f"| `{clean_pkg}` | {lines:,} |")
        md_lines.append("")

    return "\n".join(md_lines)


def emit_structured_report(
    data: Dict[str, Any], fmt_label: str, output_path: Optional[str]
) -> None:
    """Emits formatted JSON or SARIF report to stdout or output file."""
    text = json.dumps(data, indent=2)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Report saved in {fmt_label} format to {output_path}")
    else:
        print(text)


def format_github_annotations(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]]
) -> List[str]:
    """Generates GitHub Actions workflow commands to annotate PR line diffs."""
    annotations: List[str] = []
    for sim, u1, u2 in clones:
        f1 = normalize_path_string(unit_file_str(u1), strip_anchor=True)
        s1, e1 = _unit_line_bounds(u1)
        f2 = normalize_path_string(unit_file_str(u2), strip_anchor=True)
        s2, e2 = _unit_line_bounds(u2)
        n1 = str(u1.get("name") or "unit1")
        n2 = str(u2.get("name") or "unit2")
        msg1 = f"Structural clone ({sim:.1%}) matching {f2}:{s2}-{e2} ({n2})"
        col_part1 = ""
        sc1 = _safe_unit_coord(u1, "start_col")
        if sc1 is not None:
            col_part1 += f",col={max(1, sc1 + 1)}"
            ec1 = _safe_unit_coord(u1, "end_col")
            if ec1 is not None:
                col_part1 += f",endColumn={max(1, ec1 + 1)}"
        ann1 = (
            f"::warning file={f1},line={s1},endLine={e1}{col_part1},"
            f"title=pyDoppelgangerHunt Duplicate Code::{msg1}"
        )
        annotations.append(ann1)
        msg2 = f"Structural clone ({sim:.1%}) matching {f1}:{s1}-{e1} ({n1})"
        col_part2 = ""
        sc2 = _safe_unit_coord(u2, "start_col")
        if sc2 is not None:
            col_part2 += f",col={max(1, sc2 + 1)}"
            ec2 = _safe_unit_coord(u2, "end_col")
            if ec2 is not None:
                col_part2 += f",endColumn={max(1, ec2 + 1)}"
        ann2 = (
            f"::warning file={f2},line={s2},endLine={e2}{col_part2},"
            f"title=pyDoppelgangerHunt Duplicate Code::{msg2}"
        )
        annotations.append(ann2)
    return annotations


_HTML_REPORT_STYLESHEET_TEMPLATE = """        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }}
        h1 {{ font-size: 1.8rem; color: var(--primary); margin-bottom: 0.5rem; }}
        .meta {{ color: var(--muted); font-size: 0.95rem; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }}
        .stat-card .val {{ font-size: 1.8rem; font-weight: bold; margin-top: 0.5rem; color: var(--text); }}
        .stat-card .lbl {{ font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
        .gauge-container {{ display: flex; align-items: center; justify-content: center; position: relative; }}
        .gauge-text {{
            position: absolute;
            font-size: 1.4rem;
            font-weight: bold;
            color: {meter_color};
            text-align: center;
        }}
        .search-box {{
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }}
        .clone-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }}
        .card-header {{
            padding: 1rem 1.25rem;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .sim-badge {{
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: bold;
            color: #fff;
        }}
        .card-title {{ font-family: monospace; font-weight: 600; }}
        .card-body {{ padding: 1.25rem; }}
        .loc-row {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }}
        .file-tag {{
            background: rgba(56, 189, 248, 0.1);
            color: var(--primary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9rem;
        }}
        .arrow {{ color: var(--muted); }}
        .sug {{
            background: rgba(245, 158, 11, 0.08);
            border-left: 3px solid var(--warning);
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }}
        details {{ margin-top: 0.5rem; }}
        summary {{ cursor: pointer; color: var(--primary); font-size: 0.9rem; margin-bottom: 0.5rem; }}
        pre.diff {{
            background: #090d16;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #cbd5e1;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        }}
        .family-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}
        .family-card h3 {{ font-size: 1.1rem; color: var(--primary); margin-bottom: 0.5rem; }}
        .family-card ul {{ margin-left: 1.5rem; color: var(--muted); font-size: 0.9rem; }}"""


def _render_html_clone_card(
    idx: int,
    sim: float,
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    effective_repo_root: Optional[str],
    file_cache: OrderedDict[Any, Tuple[Optional[str], Optional[ast.AST]]],
    closure_strictness: Optional[str],
    non_suppressing_ctx_managers: Optional[Sequence[str]],
    on_diagnostic_error: Optional[Callable[[], None]],
) -> str:
    """Renders a single clone-pair card HTML fragment, isolating per-pair diagnostic errors."""
    try:
        src1, tree1 = resolve_cached_unit_source_and_tree(
            u1, repo_root=effective_repo_root, file_cache=file_cache
        )
        src2, tree2 = resolve_cached_unit_source_and_tree(
            u2, repo_root=effective_repo_root, file_cache=file_cache
        )
        diff_txt = generate_clone_diff(
            with_unit_source_text(u1, src1),
            with_unit_source_text(u2, src2),
            repo_root=effective_repo_root,
            source_text1=src1,
            source_text2=src2,
        )
        sug_txt = synthesize_refactoring_suggestion(
            u1,
            u2,
            repo_root=effective_repo_root,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
            source_text1=src1,
            source_text2=src2,
            tree1=tree1,
            tree2=tree2,
        )
        f1_norm = normalize_path_string(unit_file_str(u1), strip_anchor=False)
        f2_norm = normalize_path_string(unit_file_str(u2), strip_anchor=False)
        s1, e1 = _unit_line_bounds(u1)
        s2, e2 = _unit_line_bounds(u2)
        n1 = str(u1.get("name") or "unit1")
        n2 = str(u2.get("name") or "unit2")
    except MemoryError:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if on_diagnostic_error is not None:
            on_diagnostic_error()
        logger.warning(
            "Failed to render HTML card for clone pair #%d", idx, exc_info=True
        )
        n1 = str(u1.get("name") or "unit1") if isinstance(u1, dict) else "unit1"
        n2 = str(u2.get("name") or "unit2") if isinstance(u2, dict) else "unit2"
        f1_norm = (
            normalize_path_string(unit_file_str(u1), strip_anchor=False)
            if isinstance(u1, dict)
            else ""
        )
        f2_norm = (
            normalize_path_string(unit_file_str(u2), strip_anchor=False)
            if isinstance(u2, dict)
            else ""
        )
        s1, e1 = _safe_unit_line_bounds_fallback(u1)
        s2, e2 = _safe_unit_line_bounds_fallback(u2)
        diag = f"Diagnostic: pair rendering failed ({type(exc).__name__}: {exc})"
        diff_txt = f"# {diag}"
        sug_txt = f"[REFACTOR SUGGESTION] {diag}"

    diff_block = (
        f"<pre class='diff'><code>{html.escape(diff_txt)}</code></pre>"
        if diff_txt
        else "<em>No textual diff</em>"
    )
    sug_block = (
        f"<div class='sug'><strong>Refactoring Suggestion:</strong>"
        f"<pre><code>{html.escape(sug_txt)}</code></pre></div>"
        if sug_txt
        else ""
    )
    search_attr = html.escape(f"{f1_norm} {n1} {f2_norm} {n2}", quote=True)
    badge_bg = "#ef4444" if sim >= 0.85 else "#f59e0b"
    return f"""
        <div class="clone-card" data-search="{search_attr}">
            <div class="card-header">
                <span class="sim-badge" style="background: {badge_bg};">
                    {sim:.1%} Match
                </span>
                <span class="card-title">#{idx}: {html.escape(n1)} &harr; {html.escape(n2)}</span>
            </div>
            <div class="card-body">
                <div class="loc-row">
                    <span class="file-tag">&#128196; {html.escape(f1_norm)}:{s1}-{e1}</span>
                    <span class="arrow">&harr;</span>
                    <span class="file-tag">&#128196; {html.escape(f2_norm)}:{s2}-{e2}</span>
                </div>
                {sug_block}
                <details>
                    <summary>View Unified AST Diff</summary>
                    {diff_block}
                </details>
            </div>
        </div>
        """


def _render_html_family_cards(families: Sequence[Dict[str, Any]]) -> str:
    """Renders HTML cards for clustered clone families."""
    families_html: List[str] = []
    for fam in families:
        medoid_name = (
            str(fam["medoid"].get("name") or "")
            if isinstance(fam.get("medoid"), dict)
            else None
        )
        coherence_val = fam.get("coherence")
        coherence_str = (
            f" &bull; {coherence_val:.1%} coherence"
            if coherence_val is not None
            else ""
        )
        members_li_parts: List[str] = []
        for m in fam.get("members", []):
            m_file = html.escape(
                normalize_path_string(unit_file_str(m), strip_anchor=False)
            )
            m_start, m_end = _unit_line_bounds(m)
            raw_m_name = str(m.get("name") or "member")
            medoid_tag = (
                " <strong>[medoid]</strong>"
                if bool(medoid_name and raw_m_name == medoid_name)
                else ""
            )
            members_li_parts.append(
                f"<li><code>{m_file}:{m_start}-{m_end}</code> "
                f"({html.escape(raw_m_name)}){medoid_tag}</li>"
            )
        fam_id = html.escape(str(fam.get("family_id", "")))
        families_html.append(
            f"""
            <div class="family-card">
                <h3>{fam_id} &mdash; {fam['member_count']} Members ({fam['avg_similarity']:.1%} avg sim{coherence_str})</h3>
                <ul>{"".join(members_li_parts)}</ul>
            </div>
            """
        )
    return "".join(families_html)


def generate_html_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
    families: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    on_diagnostic_error: Optional[Callable[[], None]] = None,
) -> str:
    """Generates a standalone, self-contained interactive HTML audit report."""
    effective_repo_root = (
        repo_root
        if repo_root is not None
        else (target if os.path.isdir(target) else (os.path.dirname(target) or None))
    )
    dry_score = stats["dry_score"] if stats else 100.0
    grade = stats["grade"] if stats else "A+"
    sloc = stats["sloc"] if stats else 0
    dloc = stats["dloc"] if stats else 0
    duplication_pct = stats["duplication_pct"] if stats else 0.0

    meter_color = "#22c55e" if dry_score >= 90 else ("#eab308" if dry_score >= 80 else "#ef4444")
    circumference = 282.74  # 2 * pi * 45
    dashoffset = circumference * (1.0 - (dry_score / 100.0))

    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        analysis_session,
    )

    cards_html: List[str] = []
    file_cache: OrderedDict[Any, Tuple[Optional[str], Optional[ast.AST]]] = OrderedDict()
    with heuristic_ctx_manager_session(), analysis_session():
        for idx, (sim, u1, u2) in enumerate(clones, 1):
            cards_html.append(
                _render_html_clone_card(
                    idx,
                    sim,
                    u1,
                    u2,
                    effective_repo_root=effective_repo_root,
                    file_cache=file_cache,
                    closure_strictness=closure_strictness,
                    non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                    on_diagnostic_error=on_diagnostic_error,
                )
            )

    families_block = _render_html_family_cards(families) if families else ""
    cards_block = (
        "".join(cards_html)
        if cards_html
        else (
            "<p style='color: var(--muted);'>No code clones detected with "
            f"similarity &ge; {threshold:.0%}. Codebase is DRY!</p>"
        )
    )
    target_esc = html.escape(str(target))
    stylesheet = _HTML_REPORT_STYLESHEET_TEMPLATE.format(meter_color=meter_color)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pyDoppelgangerHunt Audit Report - {target_esc}</title>
    <style>
{stylesheet}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>pyDoppelgangerHunt Audit Report</h1>
            <div class="meta">Target: <code>{target_esc}</code> | Similarity Threshold: {threshold:.0%} | Generated by pyDoppelgangerHunt v{APP_VERSION}</div>
        </header>

        <div class="grid-stats">
            <div class="stat-card">
                <div class="gauge-container">
                    <svg width="120" height="120" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="45" stroke="#334155" stroke-width="8" fill="none" />
                        <circle cx="50" cy="50" r="45" stroke="{meter_color}" stroke-width="8" fill="none"
                                stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}"
                                stroke-linecap="round" transform="rotate(-90 50 50)" />
                    </svg>
                    <div class="gauge-text">{dry_score:.1f}%<br><span style="font-size: 0.9rem; font-weight: normal;">Grade {grade}</span></div>
                </div>
                <div class="lbl" style="margin-top: 0.5rem;">DRY Score</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Source Lines (SLOC)</div>
                <div class="val">{sloc:,}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Duplicated Lines (DLOC)</div>
                <div class="val">{dloc:,}</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Duplication Rate</div>
                <div class="val">{duplication_pct:.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="lbl">Clone Pairs</div>
                <div class="val">{len(clones)}</div>
            </div>
        </div>

        {"<h2>Clone Families</h2>" if families else ""}
        {families_block}

        <h2 style="margin-top: 2rem; margin-bottom: 1rem;">Detected Duplicate Code Clones</h2>
        <input type="text" id="searchInput" class="search-box" placeholder="Filter clones by path, function, or keyword..." onkeyup="filterCards()">

        <div id="cardsList">
            {cards_block}
        </div>
    </div>

    <script>
        function filterCards() {{
            const filter = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.clone-card');
            cards.forEach(card => {{
                const searchData = card.getAttribute('data-search').toLowerCase();
                if (searchData.includes(filter)) {{
                    card.style.display = '';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>"""
