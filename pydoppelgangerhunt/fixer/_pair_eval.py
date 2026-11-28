"""Unified clone-pair scope evaluation, core hazard checking, and rejection classification.

# INVARIANT (SINGLE PAIR-EVALUATION PIPELINE):
# Preview/suggestion synthesis (``reporters.py``), shared helper synthesis
# (``fixer/synthesis.py``), and patch generation (``fixer/patch.py``) MUST
# evaluate pair closure policies, inspect per-unit scopes, check shared core
# pair-level safety/compatibility hazards, and classify structured rejection
# reasons through this module rather than maintaining parallel validation
# pipelines that drift out of sync.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from pydoppelgangerhunt.config import (
    PairClosurePolicy,
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    normalize_heuristic_ctx_managers,
    normalize_path_string,
    normalize_wildcard_ctx_manager_matches,
    resolve_pair_closure_policy,
)
from pydoppelgangerhunt.fixer._df_hazards import (
    has_async_generator_delegation_hazard,
)
from pydoppelgangerhunt.fixer._df_pairing import SAFETY_PAIR_SKIP_REASONS
from pydoppelgangerhunt.fixer.scope import (
    BLOCKING_CONTROL_FLOW_HAZARDS,
    ensure_unit_scope_closure_mode,
)
from pydoppelgangerhunt.fixer.source import _copy_scope_dict
from pydoppelgangerhunt.unit_boundary import normalize_unit_boundary

__all__ = [
    "PairCoreHazardCheck",
    "PairRejectionInfo",
    "PairScopeEvaluation",
    "check_pair_core_hazards",
    "classify_pair_rejection_reasons",
    "evaluate_clone_pair_scopes",
]


class PairScopeEvaluation(NamedTuple):
    """Resolved pair closure policy, validated unit scopes, and isolated telemetry."""

    policy: PairClosurePolicy
    scope1: Dict[str, Any]
    scope2: Dict[str, Any]
    heuristics: Tuple[str, ...]
    wildcards: Tuple[Tuple[str, str], ...]


class PairCoreHazardCheck(NamedTuple):
    """Result of shared pair-level blocking hazard and compatibility checks."""

    skip_reason: Optional[str]
    detail: Optional[str]
    combined_hazards: frozenset[str]


class PairRejectionInfo(NamedTuple):
    """Classified reasons explaining why automatic helper synthesis was skipped."""

    reasons: Tuple[str, ...]
    has_safety_hazard: bool


_TERMINAL_HAZARD_REASONS: Tuple[Tuple[str, str], ...] = (
    ("unavailable_source_for_closure_mode", "unavailable source for closure mode"),
    ("invalid_unit_columns", "invalid unit columns"),
    ("stale_source_code", "stale source code"),
    ("pathological_ast", "pathological ast"),
)


def evaluate_clone_pair_scopes(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    *,
    policy_u1: Optional[Dict[str, Any]] = None,
    policy_u2: Optional[Dict[str, Any]] = None,
    scope1: Optional[Dict[str, Any]] = None,
    scope2: Optional[Dict[str, Any]] = None,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    default_strictness: str = "strict",
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    allow_disk_read1: bool = True,
    allow_disk_read2: bool = True,
    copy_scopes: bool = False,
    stamp_units: bool = False,
    isolate_telemetry: bool = True,
) -> PairScopeEvaluation:
    """Resolves pair policy and inspects both unit scopes under telemetry collectors."""
    pair_policy = resolve_pair_closure_policy(
        policy_u1 if policy_u1 is not None else u1,
        policy_u2 if policy_u2 is not None else u2,
        closure_strictness=closure_strictness,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        default=default_strictness,
    )
    eff_mode = pair_policy.effective_mode
    non_sup1 = pair_policy.unit1.non_suppressing_ctx_managers
    non_sup2 = pair_policy.unit2.non_suppressing_ctx_managers
    if stamp_units:
        u1["closure_strictness"] = eff_mode
        u2["closure_strictness"] = eff_mode
        if non_sup1 or "non_suppressing_ctx_managers" in u1:
            u1["non_suppressing_ctx_managers"] = list(non_sup1)
        if non_sup2 or "non_suppressing_ctx_managers" in u2:
            u2["non_suppressing_ctx_managers"] = list(non_sup2)

    pre_sc1 = (
        scope1
        if scope1 is not None
        else (u1.get("scope") if isinstance(u1.get("scope"), dict) else None)
    )
    pre_sc2 = (
        scope2
        if scope2 is not None
        else (u2.get("scope") if isinstance(u2.get("scope"), dict) else None)
    )
    with collect_heuristic_ctx_managers(isolate=isolate_telemetry) as heuristics:
        with collect_wildcard_ctx_managers(isolate=isolate_telemetry) as wildcards:
            sc1 = ensure_unit_scope_closure_mode(
                u1,
                pre_sc1,
                eff_mode,
                non_suppressing_ctx_managers=non_sup1,
                source_text=source_text1,
                tree=tree1,
                repo_root=repo_root,
                allow_disk_read=allow_disk_read1,
            )
            sc2 = ensure_unit_scope_closure_mode(
                u2,
                pre_sc2,
                eff_mode,
                non_suppressing_ctx_managers=non_sup2,
                source_text=source_text2,
                tree=tree2,
                repo_root=repo_root,
                allow_disk_read=allow_disk_read2,
            )
    if copy_scopes:
        sc1 = _copy_scope_dict(sc1)
        sc2 = _copy_scope_dict(sc2)
    return PairScopeEvaluation(
        policy=pair_policy,
        scope1=sc1,
        scope2=sc2,
        heuristics=tuple(normalize_heuristic_ctx_managers(heuristics)),
        wildcards=tuple(normalize_wildcard_ctx_manager_matches(wildcards)),
    )


def check_pair_core_hazards(
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    *,
    is_same_file: bool,
    allow_u2_placeholder: bool = False,
    check_stage: str = "all",
    pair_scope: Optional[Dict[str, Any]] = None,
) -> PairCoreHazardCheck:
    """Evaluates shared blocking control-flow, async, generator, nonlocal, and global hazards.

    ``check_stage`` accepts ``"terminal"`` (placeholder/schema/AST hazards evaluated
    before LHS expression pruning), ``"compatibility"`` (async/generator/nonlocal/global
    parity checks), ``"post_compat"`` (remaining closure/jump hazards), or ``"all"``.
    """
    hazards1 = set(scope1.get("control_flow_hazards", ()))
    hazards2 = set(scope2.get("control_flow_hazards", ()))
    pair_hazards = set(pair_scope.get("control_flow_hazards", ())) if pair_scope else set()
    combined = frozenset(hazards1 | hazards2 | pair_hazards)

    if check_stage in ("all", "terminal"):
        if "unavailable_source_placeholder" in hazards1 or (
            "unavailable_source_placeholder" in hazards2 and not allow_u2_placeholder
        ):
            return PairCoreHazardCheck("unavailable source placeholder", None, combined)
        for haz_key, reason_label in _TERMINAL_HAZARD_REASONS:
            if haz_key in combined:
                return PairCoreHazardCheck(reason_label, None, combined)

    if check_stage in ("all", "compatibility"):
        if scope1.get("is_async") is None or scope2.get("is_async") is None:
            return PairCoreHazardCheck("indeterminate async state", None, combined)
        if bool(scope1.get("is_async")) != bool(scope2.get("is_async")):
            return PairCoreHazardCheck("mismatched async state", None, combined)
        if bool(scope1.get("has_yield")) != bool(scope2.get("has_yield")):
            return PairCoreHazardCheck("mismatched generator state", None, combined)
        if (
            has_async_generator_delegation_hazard(scope1, scope2, pair_scope)
            if pair_scope is not None
            else has_async_generator_delegation_hazard(scope1, scope2)
        ):
            return PairCoreHazardCheck(
                "async generator delegation hazard", None, combined
            )
        if (
            scope1.get("nonlocals")
            or scope2.get("nonlocals")
            or (pair_scope is not None and pair_scope.get("nonlocals"))
        ):
            return PairCoreHazardCheck("nonlocal variables present", None, combined)
        if not is_same_file and (
            scope1.get("globals")
            or scope2.get("globals")
            or (pair_scope is not None and pair_scope.get("globals"))
        ):
            return PairCoreHazardCheck("cross-file global variables", None, combined)

    if check_stage in ("all", "post_compat"):
        if "type_param_closure_capture" in combined:
            return PairCoreHazardCheck("type_param closure capture", None, combined)
        if "naked_break" in combined or "naked_continue" in combined:
            return PairCoreHazardCheck("naked break or continue", None, combined)
        if "deleted_output" in combined:
            return PairCoreHazardCheck("deleted output hazard", None, combined)
        if "severed_closure_cell" in combined:
            sub_reasons = list(
                dict.fromkeys(
                    list(scope1.get("closure_hazard_reasons", ()))
                    + list(scope2.get("closure_hazard_reasons", ()))
                )
            )
            detail = (
                "/".join(r.replace("_", " ") for r in sub_reasons)
                if sub_reasons and sub_reasons != ["cell_severing"]
                else None
            )
            return PairCoreHazardCheck("severed closure cell", detail, combined)

    return PairCoreHazardCheck(None, None, combined)


def classify_pair_rejection_reasons(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    sc1: Dict[str, Any],
    sc2: Dict[str, Any],
    *,
    pair_skip_reason: Optional[str],
    has_async_gen_hazard: bool,
    has_nonterminal_embedded_return: bool = True,
    synthesis_failure_reason: Optional[str] = None,
    kinds_aligned: Optional[bool] = None,
    u2_allows_placeholder: bool = False,
) -> PairRejectionInfo:
    """Classifies per-unit safety hazards and pair-level structural incompatibilities."""
    safety_reasons: List[str] = []
    structural_reasons: List[str] = []

    def _add_unique(dest: List[str], item: str) -> None:
        if item and item not in dest:
            dest.append(item)

    for sc_idx, sc_cand in enumerate((sc1, sc2)):
        skip_u2_placeholder = sc_idx == 1 and u2_allows_placeholder
        for h in sc_cand.get("control_flow_hazards", ()):
            if h == "embedded_return" and not has_nonterminal_embedded_return:
                continue
            if skip_u2_placeholder and h == "unavailable_source_placeholder":
                continue
            if h in BLOCKING_CONTROL_FLOW_HAZARDS:
                _add_unique(safety_reasons, str(h).replace("_", " "))
        for cr in sc_cand.get("closure_hazard_reasons", ()):
            if skip_u2_placeholder and cr == "unavailable_source_placeholder":
                continue
            _add_unique(safety_reasons, str(cr).replace("_", " "))

    if pair_skip_reason:
        reason_text = str(pair_skip_reason)
        if reason_text in SAFETY_PAIR_SKIP_REASONS:
            _add_unique(safety_reasons, reason_text)
        elif reason_text not in safety_reasons:
            _add_unique(structural_reasons, reason_text)

    if synthesis_failure_reason and synthesis_failure_reason in SAFETY_PAIR_SKIP_REASONS:
        _add_unique(safety_reasons, synthesis_failure_reason)

    if has_async_gen_hazard:
        _add_unique(safety_reasons, "async generator delegation hazard")
    if sc1.get("nonlocals") or sc2.get("nonlocals"):
        _add_unique(safety_reasons, "nonlocal binding")

    if sc1.get("is_async") is None or sc2.get("is_async") is None:
        _add_unique(structural_reasons, "unknown async status")
    elif bool(sc1.get("is_async")) != bool(sc2.get("is_async")):
        _add_unique(structural_reasons, "async mismatch")
    if bool(sc1.get("has_yield")) != bool(sc2.get("has_yield")):
        _add_unique(structural_reasons, "generator mismatch")
    norm1 = normalize_unit_boundary(u1)
    norm2 = normalize_unit_boundary(u2)
    expr_kind_mismatch = (
        not kinds_aligned
        if kinds_aligned is not None
        else (norm1.explicit_kind in ("comprehension", "complex_expr"))
        != (norm2.explicit_kind in ("comprehension", "complex_expr"))
    )
    if expr_kind_mismatch:
        _add_unique(structural_reasons, "mismatched expression kinds")
    f1 = normalize_path_string(norm1.file_path)
    f2 = normalize_path_string(norm2.file_path)
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
        elif synthesis_failure_reason:
            _add_unique(structural_reasons, synthesis_failure_reason)

    combined = tuple(dict.fromkeys([*safety_reasons, *structural_reasons]))
    return PairRejectionInfo(
        reasons=combined,
        has_safety_hazard=bool(safety_reasons),
    )
