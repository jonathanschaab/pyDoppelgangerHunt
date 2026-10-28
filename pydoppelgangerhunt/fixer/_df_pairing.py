"""Clone output variable pairing and generator subroutine output resolution.

Handles ordered 1-to-1 output variable pairing between duplicate code units,
validating against downstream requirements and enforcing definite assignment safety.
"""

from __future__ import annotations

import ast
import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _extract_first_unit_coord,
    _extract_unit_end_col,
    _get_or_parse_tree,
    _get_valid_unit_bounds,
    _resolve_unit_ast_end_col,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _classify_closure_position,
    _collect_scope_closures,
    _normalize_closure_position,
)
from pydoppelgangerhunt.fixer._df_hazards import (
    _enclosing_try_reads_outputs,
    _resolve_downstream_scope_node,
    collect_downstream_read_names,
)
from pydoppelgangerhunt.config import resolve_closure_strictness_mode
from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode
from pydoppelgangerhunt.fixer.source import (
    _load_unit_file_text,
    is_subroutine_unit,
    resolve_unit_declared_outputs,
)

logger = logging.getLogger(__name__)


class CloneOutputResolution(NamedTuple):
    """Result of clone pair output resolution."""

    outputs: Optional[Tuple[List[str], List[str]]]
    skip_reason: Optional[str] = None


def _check_enclosing_try_cleanup_hazard(
    unit: Dict[str, Any],
    needed_outs: Sequence[str],
    source_text: Optional[str],
    tree: Optional[ast.AST],
    *,
    scope_node: Optional[ast.AST] = None,
    bounds: Optional[Tuple[int, int]] = None,
) -> bool:
    """Checks whether any needed outputs are read in enclosing try/except/finally/with blocks."""
    if not needed_outs:
        return False
    if scope_node is None:
        if source_text is None and tree is None:
            return False
        scope_node = _resolve_downstream_scope_node(source_text or "", unit, tree)
    if bounds is None:
        bounds = _get_valid_unit_bounds(unit)
    if scope_node is None or bounds is None:
        return False
    u_s, u_e = bounds
    return _enclosing_try_reads_outputs(scope_node, u_s, u_e, set(needed_outs))


def _has_unordered_multiset_outputs(unit: Dict[str, Any]) -> bool:
    """Returns True if unit outputs are stored in an unordered set with multiple items."""
    for key in ("precomputed_outputs", "outputs"):
        val = unit.get(key)
        if isinstance(val, (set, frozenset)) and len(val) > 1:
            return True
    return False


def _validate_try_cleanup_hazard(
    unit: Dict[str, Any],
    needed_outs: Sequence[str],
    source_text: Optional[str],
    tree: Optional[ast.AST],
    *,
    side_idx: int,
    context: str,
) -> Optional[CloneOutputResolution]:
    """Validates that needed outputs are not read in enclosing try/except/finally/with blocks."""
    if not needed_outs:
        return None
    hazard = "enclosing try or with cleanup hazard"
    missing_inputs = source_text is None and tree is None
    scope_node = (
        None
        if missing_inputs
        else _resolve_downstream_scope_node(source_text or "", unit, tree)
    )
    bounds = _get_valid_unit_bounds(unit)
    if missing_inputs or scope_node is None or bounds is None:
        logger.debug(
            "Rejecting %s: missing source, scope AST, or valid bounds for unit %s",
            context,
            side_idx + 1,
        )
        return CloneOutputResolution(None, hazard)
    if _check_enclosing_try_cleanup_hazard(
        unit, needed_outs, source_text, tree, scope_node=scope_node, bounds=bounds
    ):
        logger.debug(
            "Rejecting %s: needed output(s) read in enclosing try cleanup for unit %s",
            context,
            side_idx + 1,
        )
        return CloneOutputResolution(None, hazard)
    return None


def _pair_clone_outputs(
    u1_outs: Union[Sequence[str], Set[str], frozenset[str]],
    u2_outs: Union[Sequence[str], Set[str], frozenset[str]],
    allow_permuted_outputs: bool = False,
) -> List[Tuple[str, str]]:
    """Establishes an ordered 1-to-1 mapping between clone output variables."""
    if isinstance(u1_outs, (set, frozenset)) or isinstance(u2_outs, (set, frozenset)):
        logger.debug(
            "Rejecting clone output pairing: set-valued outputs lack positional ordering"
        )
        return []
    u1_dedup = list(dict.fromkeys(u1_outs))
    u2_dedup = list(dict.fromkeys(u2_outs))

    if len(u1_dedup) != len(u2_dedup):
        logger.debug(
            "Rejecting clone output pairing: output arities differ (%d vs %d: %s vs %s)",
            len(u1_dedup),
            len(u2_dedup),
            u1_dedup,
            u2_dedup,
        )
        return []
    if u1_dedup == u2_dedup:
        return [(name, name) for name in u1_dedup]

    common = set(u1_dedup) & set(u2_dedup)
    # Design Note (Recall Trade-off & Semantic Soundness):
    # For strictly alpha-equivalent clones, positional first-store pairing is theoretically
    # sound regardless of name permutations. However, in practical refactoring of real codebases,
    # when clones share variable names in differing orders (e.g. [a, b] vs [b, a]), this almost
    # always signals semantic differences or Type-3 drift rather than pure alpha-renaming.
    # Rejecting pairs with conflicting positional orders for common variables trades away a small
    # amount of recall to decisively prevent silent semantic bugs (such as accidental variable
    # swapping at extraction call sites).
    if not allow_permuted_outputs and any(
        u1_dedup.index(name) != u2_dedup.index(name) for name in common
    ):
        logger.debug(
            "Rejecting clone output pairing: common variables have conflicting "
            "positional orderings (%s vs %s)",
            u1_dedup,
            u2_dedup,
        )
        return []
    # True 1-to-1 positional pairing
    return list(zip(u1_dedup, u2_dedup))


def _filter_valid_output_pairs(
    pairs: Sequence[Tuple[str, str]],
    u1_allowed: Optional[Set[str]],
    u2_allowed: Optional[Set[str]],
    *,
    reason: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """Validates output pairs against allowed bindings and returns separated output lists."""
    for side_label, allowed_set, col_idx in (("u1", u1_allowed, 0), ("u2", u2_allowed, 1)):
        effective_allowed = set() if allowed_set is None else allowed_set
        missing = [p[col_idx] for p in pairs if p[col_idx] not in effective_allowed]
        if missing:
            logger.debug(
                "Rejecting generator subroutine outputs: %s output %s: %s",
                side_label,
                reason,
                missing,
            )
            return None
    return [p[0] for p in pairs], [p[1] for p in pairs]


class GeneratorCloneSideData(NamedTuple):
    """Encapsulates per-side output candidates, downstream reads, and definite stores."""

    outputs: Sequence[str]
    downstream: Optional[Set[str]] = None
    definite: Optional[Set[str]] = None


def resolve_generator_subroutine_outputs(
    side1: GeneratorCloneSideData,
    side2: GeneratorCloneSideData,
    allow_permuted_outputs: bool = False,
) -> Optional[Tuple[List[str], List[str]]]:
    """Selects output variables needed downstream by either clone side, preserving
    per-side necessity."""
    outs1 = list(side1.outputs)
    outs2 = list(side2.outputs)
    d1_raw = side1.downstream
    d2_raw = side2.downstream
    u1_def = side1.definite if side1.definite is not None else set()
    u2_def = side2.definite if side2.definite is not None else set()

    d1_needed = d1_raw if d1_raw is not None else set(outs1)
    d2_needed = d2_raw if d2_raw is not None else set(outs2)

    needed1 = [o for o in outs1 if o in d1_needed]
    needed2 = [o for o in outs2 if o in d2_needed]

    if not needed1 and not needed2:
        return [], []

    pairs = _pair_clone_outputs(
        outs1, outs2, allow_permuted_outputs=allow_permuted_outputs
    )
    paired_u1 = {o1 for o1, _ in pairs}
    paired_u2 = {o2 for _, o2 in pairs}

    # Fail closed: every required downstream output must have a paired counterpart
    if any(o1 not in paired_u1 for o1 in needed1) or any(o2 not in paired_u2 for o2 in needed2):
        logger.debug(
            "Rejecting generator subroutine outputs: required downstream outputs "
            "could not be paired (needed1=%s, needed2=%s, pairs=%s)",
            needed1,
            needed2,
            pairs,
        )
        return None

    kept_pairs = [
        (o1, o2)
        for o1, o2 in pairs
        if o1 in d1_needed or o2 in d2_needed
    ]

    # Fail closed on non-definitely assigned outputs to prevent UnboundLocalError
    return _filter_valid_output_pairs(
        kept_pairs,
        u1_def,
        u2_def,
        reason="lacks definite store before exit, risking UnboundLocalError",
    )


def _extract_effective_unit_outputs(
    unit: Dict[str, Any], scope: Dict[str, Any]
) -> List[str]:
    """Extracts non-global, non-local outputs from precomputed unit dict or analyzed scope."""
    raw = resolve_unit_declared_outputs(unit)

    if isinstance(raw, (set, frozenset)):
        if len(raw) > 1:
            logger.debug(
                "Rejecting set-valued outputs (%s): unordered set cannot establish "
                "positional order",
                raw,
            )
            return []
        cands: Sequence[str] = list(raw)
    elif isinstance(raw, (list, tuple)):
        cands = list(raw)
    elif scope.get("outputs_ambiguous"):
        return []
    else:
        cands = scope.get("outputs", [])
    excluded = set(scope.get("globals", [])) | set(scope.get("nonlocals", []))
    return list(dict.fromkeys(v for v in cands if v not in excluded))


@overload
def resolve_clone_generator_subroutine_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    allow_permuted_outputs: bool = False,
    *,
    return_details: Literal[True],
    closure_strictness: Optional[str] = None,
) -> CloneOutputResolution: ...


@overload
def resolve_clone_generator_subroutine_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    allow_permuted_outputs: bool = False,
    return_details: Literal[False] = False,
    closure_strictness: Optional[str] = None,
) -> Optional[Tuple[List[str], List[str]]]: ...


@overload
def resolve_clone_generator_subroutine_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    allow_permuted_outputs: bool = False,
    return_details: bool = False,
    closure_strictness: Optional[str] = None,
) -> Union[Optional[Tuple[List[str], List[str]]], CloneOutputResolution]: ...


def resolve_clone_generator_subroutine_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    allow_permuted_outputs: bool = False,
    return_details: bool = False,
    closure_strictness: Optional[str] = None,
) -> Union[Optional[Tuple[List[str], List[str]]], CloneOutputResolution]:
    """Resolves output variable bindings for clone generator subroutines across two code units.

    Ensures necessity by analyzing downstream reads and enforces definite assignment
    before unit exit to prevent runtime UnboundLocalError at exhaustion.
    """
    for u_chk, side_lbl in ((u1, "u1"), (u2, "u2")):
        if "start" in u_chk or "end" in u_chk:
            if _get_valid_unit_bounds(u_chk) is None:
                logger.debug(
                    "Rejecting generator subroutine pair: invalid unit bounds on %s: %r",
                    side_lbl,
                    u_chk,
                )
                return (
                    CloneOutputResolution(None, "invalid unit bounds")
                    if return_details
                    else None
                )

    _, effective_skip_closures = resolve_closure_strictness_mode(
        closure_strictness=closure_strictness,
        skip_pre_unit_closures=skip_pre_unit_closures,
    )
    scope1 = ensure_unit_scope_closure_mode(
        u1,
        scope1,
        effective_skip_closures,
        source_text=source_text1,
        tree=tree1,
        repo_root=repo_root,
    )
    scope2 = ensure_unit_scope_closure_mode(
        u2,
        scope2,
        effective_skip_closures,
        source_text=source_text2,
        tree=tree2,
        repo_root=repo_root,
    )

    for u_chk, s_chk, side_lbl in ((u1, scope1, "u1"), (u2, scope2, "u2")):
        if _has_unordered_multiset_outputs(u_chk):
            logger.debug(
                "Rejecting generator subroutine pair: set-valued outputs lack positional ordering"
            )
            return (
                CloneOutputResolution(None, "unpaired outputs")
                if return_details
                else None
            )
        if s_chk.get("outputs_ambiguous"):
            logger.debug(
                "Rejecting generator subroutine pair: ambiguous scope outputs "
                "lack positional ordering"
            )
            return (
                CloneOutputResolution(None, "ambiguous outputs")
                if return_details
                else None
            )
        for hazard_key, reason_str in (
            ("deleted_output", "deleted output hazard"),
            ("severed_closure_cell", "severed closure cell"),
        ):
            if hazard_key in s_chk.get("control_flow_hazards", ()):
                logger.debug(
                    "Rejecting generator subroutine pair: %s on %s",
                    reason_str,
                    side_lbl,
                )
                return (
                    CloneOutputResolution(None, reason_str)
                    if return_details
                    else None
                )

    outs1 = _extract_effective_unit_outputs(u1, scope1)
    outs2 = _extract_effective_unit_outputs(u2, scope2)

    u1_cond = set(scope1.get("conditional_outputs", [])) | set(
        scope1.get("deleted_names", [])
    )
    u2_cond = set(scope2.get("conditional_outputs", [])) | set(
        scope2.get("deleted_names", [])
    )
    u1_def = (
        set(scope1.get("definite_stores", []))
        | (set(scope1.get("inputs", [])) - u1_cond)
        if ("definite_stores" in scope1 or "inputs" in scope1)
        else set()
    )
    u2_def = (
        set(scope2.get("definite_stores", []))
        | (set(scope2.get("inputs", [])) - u2_cond)
        if ("definite_stores" in scope2 or "inputs" in scope2)
        else set()
    )

    cands1 = set(outs1)
    cands2 = set(outs2)

    targets = [
        (u1, cands1, source_text1, tree1, source_digest1, scope1),
        (u2, cands2, source_text2, tree2, source_digest2, scope2),
    ]
    target_srcs: List[Optional[str]] = []
    target_trees: List[Optional[ast.AST]] = []
    downstream_reads: List[Optional[Set[str]]] = []
    for u_item, cands, s_text, tr, s_digest, sc in targets:
        src = (
            s_text
            if s_text is not None
            else _load_unit_file_text(u_item, repo_root=repo_root)
        )
        target_srcs.append(src)
        eff_tr = tr if tr is not None else (
            _get_or_parse_tree(src, source_digest=s_digest) if src is not None else None
        )
        target_trees.append(eff_tr)
        if src is not None:
            unit_inps = (
                set(sc["inputs"])
                if "inputs" in sc and sc["inputs"] is not None
                else None
            )
            reads = collect_downstream_read_names(
                src,
                u_item,
                candidates=cands,
                tree=eff_tr,
                skip_pre_unit_closures=effective_skip_closures,
                source_digest=s_digest,
                unit_inputs=unit_inps,
            )
        else:
            reads = None
        downstream_reads.append(reads)

    side1 = GeneratorCloneSideData(outs1, downstream_reads[0], u1_def)
    side2 = GeneratorCloneSideData(outs2, downstream_reads[1], u2_def)
    resolved = resolve_generator_subroutine_outputs(
        side1, side2, allow_permuted_outputs=allow_permuted_outputs
    )
    if resolved is None:
        return (
            CloneOutputResolution(None, "generator output or closure conflict")
            if return_details
            else None
        )

    out1, out2 = resolved
    try:
        for side_idx, (u_item, needed_outs, s_text, eff_tr) in enumerate([
            (u1, out1, target_srcs[0], target_trees[0]),
            (u2, out2, target_srcs[1], target_trees[1]),
        ]):
            if needed_outs:
                hazard_res = _validate_try_cleanup_hazard(
                    u_item, needed_outs, s_text, eff_tr,
                    side_idx=side_idx, context="generator subroutine pair",
                )
                if hazard_res is not None:
                    return hazard_res if return_details else None
                scope_node = _resolve_downstream_scope_node(s_text or "", u_item, eff_tr)
                bounds = _get_valid_unit_bounds(u_item)
                if scope_node is None or bounds is None:
                    return (
                        CloneOutputResolution(None, "generator output or closure conflict")
                        if return_details
                        else None
                    )
                u_s, u_e = bounds
                u_start_col = _extract_first_unit_coord(
                    u_item, ("start_col", "start_col_offset")
                )
                u_end_col = _extract_unit_end_col(u_item)
                if u_end_col is None:
                    u_end_col = _resolve_unit_ast_end_col(scope_node, u_item)
                for c_pos, reads in _collect_scope_closures(scope_node, module_tree=eff_tr):
                    _, is_intra, _ = _classify_closure_position(
                        c_pos, u_s, u_e, u_start_col=u_start_col, u_end_col=u_end_col
                    )
                    if is_intra and bool(reads & set(needed_outs)):
                        c_line, _ = _normalize_closure_position(c_pos)
                        logger.debug(
                            "Rejecting generator subroutine pair: intra-unit closure at line %d "
                            "captures needed output(s) %r for unit %s",
                            c_line,
                            reads & set(needed_outs),
                            side_idx + 1,
                        )
                        return (
                            CloneOutputResolution(None, "generator output or closure conflict")
                            if return_details
                            else None
                        )
    except (RecursionError, MemoryError):
        return (
            CloneOutputResolution(None, "generator output or closure conflict")
            if return_details
            else None
        )

    return CloneOutputResolution(resolved, None) if return_details else resolved


@overload
def resolve_clone_pair_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    closure_strictness: Optional[str] = None,
    require_pairing: Optional[bool] = None,
    allow_permuted_outputs: bool = False,
    *,
    return_details: Literal[True],
) -> CloneOutputResolution: ...


@overload
def resolve_clone_pair_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    closure_strictness: Optional[str] = None,
    require_pairing: Optional[bool] = None,
    allow_permuted_outputs: bool = False,
    return_details: Literal[False] = False,
) -> Optional[Tuple[List[str], List[str]]]: ...


@overload
def resolve_clone_pair_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    closure_strictness: Optional[str] = None,
    require_pairing: Optional[bool] = None,
    allow_permuted_outputs: bool = False,
    return_details: bool = False,
) -> Union[Optional[Tuple[List[str], List[str]]], CloneOutputResolution]: ...


def resolve_clone_pair_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
    closure_strictness: Optional[str] = None,
    require_pairing: Optional[bool] = None,
    allow_permuted_outputs: bool = False,
    return_details: bool = False,
) -> Union[Optional[Tuple[List[str], List[str]]], CloneOutputResolution]:
    """Resolves output variable bindings between two clone units, enforcing
    strict positional pairing and generator subroutine safety.

    Returns:
        (outputs1, outputs2) tuple of paired output names, or None if pairing
        fails or safety invariants are violated (or CloneOutputResolution if return_details=True).
    """
    is_sub1 = is_subroutine_unit(u1, source_text=source_text1, tree=tree1)
    is_sub2 = is_subroutine_unit(u2, source_text=source_text2, tree=tree2)
    if is_sub1 != is_sub2:
        logger.debug(
            "Rejecting clone pair output resolution: mismatched subroutine kinds "
            "(is_sub1=%s, is_sub2=%s)",
            is_sub1,
            is_sub2,
        )
        return (
            CloneOutputResolution(None, "mismatched subroutine kinds")
            if return_details
            else None
        )
    is_sub = is_sub1

    has_precomputed = (
        "outputs" in u1
        and isinstance(u1["outputs"], (list, tuple, set, frozenset))
        and "outputs" in u2
        and isinstance(u2["outputs"], (list, tuple, set, frozenset))
    )
    strict_pairing = (is_sub or has_precomputed) if require_pairing is None else require_pairing

    if any(_has_unordered_multiset_outputs(u) for u in (u1, u2)):
        logger.debug(
            "Rejecting clone pair output resolution: set-valued outputs "
            "lack positional ordering"
        )
        return (
            CloneOutputResolution(None, "unpaired outputs")
            if return_details
            else None
        )

    if scope1.get("outputs_ambiguous") or scope2.get("outputs_ambiguous"):
        logger.debug(
            "Rejecting clone pair output resolution: ambiguous scope outputs "
            "lack positional ordering"
        )
        return (
            CloneOutputResolution(None, "ambiguous outputs")
            if return_details
            else None
        )

    _, effective_skip_closures = resolve_closure_strictness_mode(
        closure_strictness=closure_strictness,
        skip_pre_unit_closures=skip_pre_unit_closures,
    )
    scope1 = ensure_unit_scope_closure_mode(
        u1,
        scope1,
        effective_skip_closures,
        source_text=source_text1,
        tree=tree1,
        repo_root=repo_root,
    )
    scope2 = ensure_unit_scope_closure_mode(
        u2,
        scope2,
        effective_skip_closures,
        source_text=source_text2,
        tree=tree2,
        repo_root=repo_root,
    )

    has_yield = bool(scope1.get("has_yield") or scope2.get("has_yield"))
    if is_sub and has_yield:
        return resolve_clone_generator_subroutine_outputs(
            u1=u1,
            u2=u2,
            scope1=scope1,
            scope2=scope2,
            source_text1=source_text1,
            source_text2=source_text2,
            tree1=tree1,
            tree2=tree2,
            repo_root=repo_root,
            skip_pre_unit_closures=effective_skip_closures,
            source_digest1=source_digest1,
            source_digest2=source_digest2,
            allow_permuted_outputs=allow_permuted_outputs,
            return_details=return_details,
        )

    outs1 = _extract_effective_unit_outputs(u1, scope1)
    outs2 = _extract_effective_unit_outputs(u2, scope2)
    if (
        "severed_closure_cell" in scope1.get("control_flow_hazards", ())
        or "severed_closure_cell" in scope2.get("control_flow_hazards", ())
    ):
        logger.debug(
            "Rejecting clone pair output resolution: severed closure cell"
        )
        return (
            CloneOutputResolution(None, "severed closure cell")
            if return_details
            else None
        )
    if is_sub and (
        "deleted_output" in scope1.get("control_flow_hazards", ())
        or "deleted_output" in scope2.get("control_flow_hazards", ())
        or bool(set(outs1) & set(scope1.get("deleted_names", [])))
        or bool(set(outs2) & set(scope2.get("deleted_names", [])))
    ):
        logger.debug(
            "Rejecting clone pair output resolution: deleted output hazard"
        )
        return (
            CloneOutputResolution(None, "deleted output hazard")
            if return_details
            else None
        )
    if not outs1 and not outs2:
        res: Tuple[List[str], List[str]] = ([], [])
        return CloneOutputResolution(res, None) if return_details else res

    pairs = _pair_clone_outputs(
        outs1, outs2, allow_permuted_outputs=allow_permuted_outputs
    )
    if len(pairs) == len(outs1) == len(outs2):
        if is_sub and outs1:
            for side_idx, (u_item, s_text, tr, s_digest, outs) in enumerate([
                (u1, source_text1, tree1, source_digest1, outs1),
                (u2, source_text2, tree2, source_digest2, outs2),
            ]):
                eff_src = (
                    s_text
                    if s_text is not None
                    else _load_unit_file_text(u_item, repo_root=repo_root)
                )
                eff_tr = tr if tr is not None else (
                    _get_or_parse_tree(eff_src, source_digest=s_digest)
                    if eff_src is not None
                    else None
                )
                hazard_res = _validate_try_cleanup_hazard(
                    u_item, outs, eff_src, eff_tr,
                    side_idx=side_idx, context="clone pair output resolution",
                )
                if hazard_res is not None:
                    return hazard_res if return_details else None
        paired_res = ([p[0] for p in pairs], [p[1] for p in pairs])
        return CloneOutputResolution(paired_res, None) if return_details else paired_res

    if not strict_pairing:
        unpaired_res = (outs1, outs2)
        return CloneOutputResolution(unpaired_res, None) if return_details else unpaired_res

    if len(outs1) != len(outs2):
        skip_reason = "output arity mismatch"
    elif not allow_permuted_outputs and any(
        outs1.index(n) != outs2.index(n) for n in (set(outs1) & set(outs2))
    ):
        skip_reason = "permuted outputs"
    else:
        skip_reason = "unpaired outputs"

    logger.debug(
        "Rejecting clone pair output resolution: outputs could not be paired "
        "(outs1=%s, outs2=%s, pairs=%s, reason=%s)",
        outs1,
        outs2,
        pairs,
        skip_reason,
    )
    return CloneOutputResolution(None, skip_reason) if return_details else None
