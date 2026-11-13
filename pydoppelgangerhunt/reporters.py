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
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from pydoppelgangerhunt.canonical_path import parse_notebook_cell_anchor
from pydoppelgangerhunt.config import (
    APP_VERSION,
    heuristic_ctx_manager_session,
    normalize_path_string,
    resolve_pair_closure_flags,
)
from pydoppelgangerhunt.source_lines import (
    _extract_notebook_cell_lines,
    extract_unit_source_code,
    parse_unit_coord,
    resolve_safe_unit_file_path,
    resolve_unit_line_bounds,
    with_unit_source_text,
)
from pydoppelgangerhunt.source_lines import (  # pylint: disable=unused-import
    extract_unit_source_lines_with_state,
)

logger = logging.getLogger(__name__)


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


def _read_notebook_cell_lines(resolved_file: Path, cell_idx: int) -> Optional[List[str]]:
    """Reads and extracts newline-terminated source lines for a 0-indexed Jupyter notebook cell on disk."""
    try:
        raw_nb_text = resolved_file.read_text(encoding="utf-8", errors="replace")
        nb_data = json.loads(raw_nb_text)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return _extract_notebook_cell_lines(nb_data, cell_idx)


def generate_clone_diff(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    color: bool = False,
) -> str:
    """Produces unified line diff between two cloned code blocks, with optional syntax coloring."""
    lines1 = extract_unit_source_code(u1, repo_root)
    lines2 = extract_unit_source_code(u2, repo_root)
    f1_norm = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
    f2_norm = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
    s1, e1 = _unit_line_bounds(u1)
    s2, e2 = _unit_line_bounds(u2)
    n1 = str(u1.get("name") or "unit1")
    n2 = str(u2.get("name") or "unit2")
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


class _SuggestionRejectionInfo(NamedTuple):
    """Classified reasons explaining why automatic helper synthesis was skipped."""

    reasons: Tuple[str, ...]
    has_safety_hazard: bool


_PAIR_SAFETY_HAZARD_REASONS = frozenset(
    {
        "enclosing try/with cleanup hazard",
        "deleted output hazard",
        "generator cleanup hazard",
        "severed closure cell",
        "stale source code",
        "invalid unit columns",
        "pathological ast",
        "unavailable source for closure mode",
        "unavailable source placeholder",
    }
)


def _collect_suggestion_rejection_reasons(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
    *,
    pair_skip_reason: Optional[str],
    has_async_gen_hazard: bool,
    synthesis_failed_with_error: bool = False,
) -> _SuggestionRejectionInfo:
    """Classifies per-unit safety hazards and pair-level structural incompatibilities."""
    safety_reasons: List[str] = []
    structural_reasons: List[str] = []

    def _add_unique(dest: List[str], item: str) -> None:
        if item and item not in dest:
            dest.append(item)

    for sc_cand in (sc1, sc2):
        for h in sc_cand.get("control_flow_hazards", ()):
            _add_unique(safety_reasons, str(h).replace("_", " "))
        for cr in sc_cand.get("closure_hazard_reasons", ()):
            _add_unique(safety_reasons, str(cr).replace("_", " "))

    if pair_skip_reason:
        if pair_skip_reason in _PAIR_SAFETY_HAZARD_REASONS or "hazard" in pair_skip_reason:
            _add_unique(safety_reasons, pair_skip_reason)
        elif pair_skip_reason not in safety_reasons:
            _add_unique(structural_reasons, pair_skip_reason)

    if has_async_gen_hazard:
        _add_unique(safety_reasons, "async generator delegation hazard")
    if sc1.get("nonlocals") or sc2.get("nonlocals"):
        _add_unique(safety_reasons, "nonlocal binding")

    if (
        sc1.get("is_async") is None
        or sc2.get("is_async") is None
    ):
        _add_unique(structural_reasons, "unknown async status")
    elif bool(sc1.get("is_async")) != bool(sc2.get("is_async")):
        _add_unique(structural_reasons, "async mismatch")
    if bool(sc1.get("has_yield")) != bool(sc2.get("has_yield")):
        _add_unique(structural_reasons, "generator mismatch")
    if (u1.get("kind") in ("comprehension", "complex_expr")) != (
        u2.get("kind") in ("comprehension", "complex_expr")
    ):
        _add_unique(structural_reasons, "mismatched expression kinds")
    f1 = normalize_path_string(str(u1.get("file") or ""))
    f2 = normalize_path_string(str(u2.get("file") or ""))
    if f1 and f2 and f1 != f2 and (sc1.get("globals") or sc2.get("globals")):
        _add_unique(structural_reasons, "cross-file global access")
    if not safety_reasons and not structural_reasons:
        if (
            sc1.get("attrs_read") != sc2.get("attrs_read")
            or sc1.get("attrs_written") != sc2.get("attrs_written")
        ):
            _add_unique(structural_reasons, "mismatched attribute access")
        elif (
            sc1.get("has_super")
            or sc2.get("has_super")
            or sc1.get("has_mangled_names")
            or sc2.get("has_mangled_names")
            or sc1.get("has_class_scoped_type_params")
            or sc2.get("has_class_scoped_type_params")
        ):
            _add_unique(structural_reasons, "incompatible class binding")
        elif synthesis_failed_with_error:
            _add_unique(structural_reasons, "unparseable or invalid syntax")

    combined = tuple(dict.fromkeys([*safety_reasons, *structural_reasons]))
    return _SuggestionRejectionInfo(
        reasons=combined,
        has_safety_hazard=bool(safety_reasons),
    )


_MAX_REPORT_FILE_CACHE_ENTRIES = 256


def resolve_cached_unit_source_and_tree(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_cache: Optional[
        OrderedDict[str, Tuple[Optional[str], Optional[ast.AST]]]
    ] = None,
) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Loads and caches (source_text, parsed_ast) for a unit across multi-pair reporting passes."""
    from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=import-outside-toplevel
        _get_or_parse_tree,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        _load_unit_file_text,
    )

    inline_src = unit.get("source_text")
    if not isinstance(inline_src, str) or not inline_src:
        inline_src = unit.get("file_source")
    if isinstance(inline_src, str) and inline_src:
        try:
            return inline_src, _get_or_parse_tree(inline_src)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return inline_src, None

    if "source_lines" in unit and unit.get("source_lines") is not None:
        return None, None

    raw_file = str(unit.get("file") or "")
    if not raw_file:
        return None, None

    cache_key: Optional[str] = None
    resolved_p = resolve_safe_unit_file_path(unit, repo_root=repo_root)
    if file_cache is not None and resolved_p is not None:
        anchor = raw_file.split("#", 1)[1] if "#" in raw_file else ""
        cache_key = f"{resolved_p}#{anchor}" if anchor else str(resolved_p)
        if cache_key in file_cache:
            file_cache.move_to_end(cache_key)
            return file_cache[cache_key]

    src_text: Optional[str]
    if resolved_p is not None and resolved_p.suffix.lower() == ".ipynb" and "#" in raw_file:
        parsed_nb = parse_notebook_cell_anchor(raw_file)
        if parsed_nb is not None:
            cell_lines = _read_notebook_cell_lines(resolved_p, parsed_nb[1])
            src_text = "".join(cell_lines) if cell_lines else None
        else:
            src_text = None
    else:
        src_text = _load_unit_file_text(unit, repo_root=repo_root)
    parsed_tree: Optional[ast.AST] = None
    if isinstance(src_text, str) and src_text:
        try:
            parsed_tree = _get_or_parse_tree(src_text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            parsed_tree = None
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


def _prepare_effective_suggestion_unit(
    unit: Dict[str, Any],
    *,
    repo_root: Optional[str],
    source_text: Optional[str],
    tree: Optional[ast.AST],
) -> _PreparedSuggestionUnit:
    """Copies unit, attaches resolved full-module source_text when not inline, and slices unit lines."""
    u_eff = dict(unit)
    eff_src, eff_tree = (
        (source_text, tree)
        if source_text is not None or tree is not None
        else resolve_cached_unit_source_and_tree(u_eff, repo_root=repo_root)
    )
    if (
        eff_src is not None
        and "source_text" not in u_eff
        and "source_lines" not in u_eff
    ):
        u_eff["source_text"] = eff_src
    return _PreparedSuggestionUnit(
        unit=u_eff,
        source_text=eff_src,
        tree=eff_tree,
        lines=extract_unit_source_code(u_eff, repo_root),
    )


def synthesize_refactoring_suggestion(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    *,
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
) -> str:
    """Synthesizes actionable refactoring recommendation and shared helper template."""
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
        CloneOutputResolution,
        has_async_generator_delegation_hazard,
        resolve_clone_pair_outputs,
    )
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        ensure_unit_scope_closure_mode,
    )
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        synthesize_shared_helper_code,
    )

    k1_expr = p1.unit.get("kind") in ("comprehension", "complex_expr")
    k2_expr = p2.unit.get("kind") in ("comprehension", "complex_expr")
    if k1_expr != k2_expr and not (p1.unit.get("kind") and p2.unit.get("kind")):
        eff_expr_kind = p1.unit.get("kind") if k1_expr else p2.unit.get("kind")
        p1.unit.setdefault("kind", eff_expr_kind)
        p2.unit.setdefault("kind", eff_expr_kind)
    is_expr_unit = p1.unit.get("kind") in ("comprehension", "complex_expr")

    pair_flags = resolve_pair_closure_flags(
        p1.unit,
        p2.unit,
        closure_strictness=closure_strictness,
        skip_pre_unit_closures=skip_pre_unit_closures,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
    )
    pre_sc1 = p1.unit.get("scope") if isinstance(p1.unit.get("scope"), dict) else None
    pre_sc2 = p2.unit.get("scope") if isinstance(p2.unit.get("scope"), dict) else None
    sc1 = ensure_unit_scope_closure_mode(
        p1.unit,
        pre_sc1,
        pair_flags.effective_skip_pre_unit_closures,
        strict_with_suppression=pair_flags.effective_strict_with_suppression,
        non_suppressing_ctx_managers=pair_flags.unit1.non_suppressing_ctx_managers,
        source_text=p1.source_text,
        tree=p1.tree,
        repo_root=repo_root,
    )
    sc2 = ensure_unit_scope_closure_mode(
        p2.unit,
        pre_sc2,
        pair_flags.effective_skip_pre_unit_closures,
        strict_with_suppression=pair_flags.effective_strict_with_suppression,
        non_suppressing_ctx_managers=pair_flags.unit2.non_suppressing_ctx_managers,
        source_text=p2.source_text,
        tree=p2.tree,
        repo_root=repo_root,
    )
    has_naked_loop_jump = any(
        h in ("naked_break", "naked_continue")
        for h in (*sc1.get("control_flow_hazards", ()), *sc2.get("control_flow_hazards", ()))
    )
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
        skip_pre_unit_closures=pair_flags.effective_skip_pre_unit_closures,
        closure_strictness=closure_strictness,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        return_details=True,
    )
    pair_outputs: Optional[Tuple[List[str], List[str]]] = (
        pair_out_raw.outputs
        if isinstance(pair_out_raw, CloneOutputResolution)
        else pair_out_raw
    )
    pair_skip_reason: Optional[str] = (
        pair_out_raw.skip_reason
        if isinstance(pair_out_raw, CloneOutputResolution)
        else None
    )

    helper_code = ""
    synthesis_failed_with_error = False
    if not has_naked_loop_jump and (pair_outputs is not None or is_expr_unit):
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
                closure_strictness=closure_strictness,
                skip_pre_unit_closures=skip_pre_unit_closures,
                non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                outputs=None if is_expr_unit or pair_outputs is None else pair_outputs[0],
                outputs2=None if is_expr_unit or pair_outputs is None else pair_outputs[1],
                scope1=sc1,
                scope2=sc2,
            )
        except (
            SyntaxError,
            OSError,
            UnicodeDecodeError,
            RecursionError,
            MemoryError,
        ) as exc:
            logger.debug(
                "Failed to synthesize refactoring suggestion for (%s, %s): %s",
                n1,
                n2,
                exc,
                exc_info=True,
            )
            helper_code = ""
            synthesis_failed_with_error = True
    if helper_code:
        indented_helper = "\n".join(
            f"    {ln}" if ln.strip() else ""
            for ln in helper_code.rstrip("\n").splitlines()
        )
        return (
            f"[REFACTOR SUGGESTION] Extract duplicated logic into common helper '{helper_name}':\n"
            f"    # {diff_summary}\n"
            f"{indented_helper}"
        )

    has_async_gen_hazard = has_async_generator_delegation_hazard(
        sc1,
        sc2,
        has_outputs=bool(pair_outputs and (pair_outputs[0] or pair_outputs[1])),
    )
    rejection = _collect_suggestion_rejection_reasons(
        p1.unit,
        p2.unit,
        sc1,
        sc2,
        pair_skip_reason=pair_skip_reason,
        has_async_gen_hazard=has_async_gen_hazard,
        synthesis_failed_with_error=synthesis_failed_with_error,
    )
    reason_suffix = (
        f" [{', '.join(rejection.reasons)}]" if rejection.reasons else ""
    )
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


def format_sarif_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
) -> Dict[str, Any]:
    """Formats detected clones into OASIS SARIF 2.1.0 standard schema for GitHub Code Scanning."""
    results: List[Dict[str, Any]] = []
    for idx, (sim, u1, u2) in enumerate(clones):
        f1_norm = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
        f2_norm = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
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
        "file": normalize_path_string(str(unit.get("file") or ""), strip_anchor=False),
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
        f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=True)
        s1, e1 = _unit_line_bounds(u1)
        f2 = normalize_path_string(str(u2.get("file") or ""), strip_anchor=True)
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


def generate_html_report(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    target: str,
    threshold: float,
    families: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
    *,
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: Optional[bool] = None,
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

    cards_html: List[str] = []
    file_cache: OrderedDict[str, Tuple[Optional[str], Optional[ast.AST]]] = OrderedDict()
    with heuristic_ctx_manager_session():
        for idx, (sim, u1, u2) in enumerate(clones, 1):
            try:
                src1, tree1 = resolve_cached_unit_source_and_tree(
                    u1, repo_root=effective_repo_root, file_cache=file_cache
                )
                src2, tree2 = resolve_cached_unit_source_and_tree(
                    u2, repo_root=effective_repo_root, file_cache=file_cache
                )
                u1_diff = with_unit_source_text(u1, src1)
                u2_diff = with_unit_source_text(u2, src2)
                diff_txt = generate_clone_diff(
                    u1_diff, u2_diff, repo_root=effective_repo_root
                )
                sug_txt = synthesize_refactoring_suggestion(
                    u1,
                    u2,
                    repo_root=effective_repo_root,
                    closure_strictness=closure_strictness,
                    skip_pre_unit_closures=skip_pre_unit_closures,
                    non_suppressing_ctx_managers=non_suppressing_ctx_managers,
                    source_text1=src1,
                    source_text2=src2,
                    tree1=tree1,
                    tree2=tree2,
                )
                f1_norm = normalize_path_string(
                    str(u1.get("file") or ""), strip_anchor=False
                )
                f2_norm = normalize_path_string(
                    str(u2.get("file") or ""), strip_anchor=False
                )
                s1, e1 = _unit_line_bounds(u1)
                s2, e2 = _unit_line_bounds(u2)
                n1 = str(u1.get("name") or "unit1")
                n2 = str(u2.get("name") or "unit2")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if on_diagnostic_error is not None:
                    on_diagnostic_error()
                logger.warning(
                    "Failed to render HTML card for clone pair #%d",
                    idx,
                    exc_info=True,
                )
                n1 = (
                    str(u1.get("name") or "unit1")
                    if isinstance(u1, dict)
                    else "unit1"
                )
                n2 = (
                    str(u2.get("name") or "unit2")
                    if isinstance(u2, dict)
                    else "unit2"
                )
                f1_norm = (
                    normalize_path_string(
                        str(u1.get("file") or ""), strip_anchor=False
                    )
                    if isinstance(u1, dict)
                    else ""
                )
                f2_norm = (
                    normalize_path_string(
                        str(u2.get("file") or ""), strip_anchor=False
                    )
                    if isinstance(u2, dict)
                    else ""
                )
                s1, e1, s2, e2 = 1, 1, 1, 1
                diag = f"Diagnostic: pair rendering failed ({type(exc).__name__}: {exc})"
                diff_txt = f"# {diag}"
                sug_txt = f"[REFACTOR SUGGESTION] {diag}"
            diff_block = (
                f"<pre class='diff'><code>{html.escape(diff_txt)}</code></pre>"
                if diff_txt
                else "<em>No textual diff</em>"
            )
            sug_block = (
                f"<div class='sug'><strong>Refactoring Suggestion:</strong><pre><code>{html.escape(sug_txt)}</code></pre></div>"
                if sug_txt
                else ""
            )

            f1_esc = html.escape(f1_norm)
            f2_esc = html.escape(f2_norm)
            n1_esc = html.escape(n1)
            n2_esc = html.escape(n2)
            search_attr = html.escape(f"{f1_norm} {n1} {f2_norm} {n2}", quote=True)

            card = f"""
        <div class="clone-card" data-search="{search_attr}">
            <div class="card-header">
                <span class="sim-badge" style="background: {'#ef4444' if sim >= 0.85 else '#f59e0b'};">
                    {sim:.1%} Match
                </span>
                <span class="card-title">#{idx}: {n1_esc} &harr; {n2_esc}</span>
            </div>
            <div class="card-body">
                <div class="loc-row">
                    <span class="file-tag">&#128196; {f1_esc}:{s1}-{e1}</span>
                    <span class="arrow">&harr;</span>
                    <span class="file-tag">&#128196; {f2_esc}:{s2}-{e2}</span>
                </div>
                {sug_block}
                <details>
                    <summary>View Unified AST Diff</summary>
                    {diff_block}
                </details>
            </div>
        </div>
        """
            cards_html.append(card)

    families_html: List[str] = []
    if families:
        for fam in families:
            medoid_name = (
                str(fam["medoid"].get("name") or "")
                if isinstance(fam.get("medoid"), dict)
                else None
            )
            coherence_val = fam.get("coherence")
            coherence_str = f" &bull; {coherence_val:.1%} coherence" if coherence_val is not None else ""
            members_li_parts: List[str] = []
            for m in fam.get("members", []):
                m_file = html.escape(
                    normalize_path_string(str(m.get("file") or ""), strip_anchor=False)
                )
                m_start, m_end = _unit_line_bounds(m)
                raw_m_name = str(m.get("name") or "member")
                m_name = html.escape(raw_m_name)
                is_medoid = bool(medoid_name and raw_m_name == medoid_name)
                medoid_tag = " <strong>[medoid]</strong>" if is_medoid else ""
                members_li_parts.append(
                    f"<li><code>{m_file}:{m_start}-{m_end}</code> ({m_name}){medoid_tag}</li>"
                )
            members_li = "".join(members_li_parts)
            fam_id = html.escape(str(fam.get("family_id", "")))
            f_card = f"""
            <div class="family-card">
                <h3>{fam_id} &mdash; {fam['member_count']} Members ({fam['avg_similarity']:.1%} avg sim{coherence_str})</h3>
                <ul>{members_li}</ul>
            </div>
            """
            families_html.append(f_card)

    target_esc = html.escape(str(target))
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pyDoppelgangerHunt Audit Report - {target_esc}</title>
    <style>
        :root {{
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
        .family-card ul {{ margin-left: 1.5rem; color: var(--muted); font-size: 0.9rem; }}
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
        {"".join(families_html)}

        <h2 style="margin-top: 2rem; margin-bottom: 1rem;">Detected Duplicate Code Clones</h2>
        <input type="text" id="searchInput" class="search-box" placeholder="Filter clones by path, function, or keyword..." onkeyup="filterCards()">

        <div id="cardsList">
            {"".join(cards_html) if cards_html else "<p style='color: var(--muted);'>No code clones detected with similarity &ge; " + f"{threshold:.0%}. Codebase is DRY!</p>"}
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
    return html_content
