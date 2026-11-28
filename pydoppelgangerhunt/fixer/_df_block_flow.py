"""Unified definite-assignment, deletion tracking, and block termination analysis."""

from __future__ import annotations

import ast
import contextlib
from dataclasses import (
    dataclass,
    field,
    replace,
)
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
)
from pydoppelgangerhunt._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
    is_trystar_node,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import _can_bind_in_fn_scope
from pydoppelgangerhunt.fixer._df_cfg import (
    _CompoundClause,
    _apply_with_exit_invalidation,
    _extract_store_names_in_node,
    _extract_type_alias_name,
    _lower_compound_clauses,
    _merge_surviving_branches,
    _run_match_cfg,
    _run_try_handlers_cfg,
    _run_with_cfg,
    _scan_stmts_shallow_exits,
    _summarize_with_header_steps,
)
from pydoppelgangerhunt.fixer._df_prov_exprs import (
    _advance_clause_entry_prov,
    _advance_prov_through_for_header,
    _advance_prov_through_opt_expr,
    _merge_prior_clause_guard_and_binds,
    _merge_prior_compound_clause_prov,
)
from pydoppelgangerhunt.fixer._df_provenance_walk import (
    _ProvenanceWalkContext,
    _iter_with_item_provenance_steps,
    _run_loop_body_prov_fixpoint,
    _split_suppress_with_body,
    _split_with_body_by_mode,
    _update_suppress_provenance_from_stmt,
    _update_suppress_provenance_from_stmts,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _record_import_names,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _EMPTY_SUPPRESS_PROVENANCE,
    _SuppressProvenance,
    _coerce_suppress_provenance,
)
from pydoppelgangerhunt.fixer._df_with_rules import (
    _collect_shallow_ast_flow_summary,
    _conservatively_update_prov_at_depth_limit,
    _extract_definite_target_names,
    _extract_deleted_names,
    _extract_shallow_deleted_names,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


def _demote_deletions_to_conditional(
    deleted_names: Iterable[str],
    definite: Set[str],
    conditional: Set[str],
) -> None:
    """Discards deleted variables from definite and promotes previously definite variables to conditional."""
    for v_del in deleted_names:
        if v_del in definite:
            definite.discard(v_del)
            conditional.add(v_del)


def _block_terminates(
    statements: Sequence[ast.AST],
    *,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    loop_control_terminates: bool = False,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> bool:
    """Returns True if the statement sequence unconditionally terminates."""
    return _analyze_block_assignment_impl(
        statements,
        definite=set(safe_names),
        suppress_aliases=suppress_aliases,
        loop_control_terminates=loop_control_terminates,
        known_safe_names=safe_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth,
    ).terminated


def _filter_scope_names(
    names: Iterable[str],
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
) -> Set[str]:
    """Filters candidate bound names by scope nonlocal/global rules when active."""
    if active_nonlocals is None and not fn_globals:
        return set(names)
    return {n for n in names if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)}


def _extract_block_combined_deletions(stmts: Sequence[ast.AST]) -> Set[str]:
    """Returns combined deep and shallow deleted/unbound variable names for stmts."""
    if not stmts:
        return set()
    return _extract_deleted_names(stmts) | set().union(
        *(_extract_shallow_deleted_names(s) for s in stmts)
    )


class _BranchFlowState(NamedTuple):
    """Snapshot of definite, conditional, may_deleted, and must_deleted sets on a control-flow path."""

    definite: Set[str]
    conditional: Set[str]
    may_del: Set[str]
    must_del: Set[str]


@dataclass
class _LoopJumpSummary:
    """Tracks variable state across break/continue jumps targeting the innermost enclosing loop."""

    has_jump: bool = False
    has_break: bool = False
    may_bound: Set[str] = field(default_factory=set)
    may_deleted: Set[str] = field(default_factory=set)


class _TryFlowContext(NamedTuple):
    """Shared invariant analysis context passed across compound-statement block-flow dispatch."""

    active_nonlocals: Optional[Set[str]]
    fn_globals: AbstractSet[str]
    loop_control_terminates: bool
    loop_jumps: Optional[_LoopJumpSummary]
    ever_bound: Optional[Set[str]]
    known_safe_names: AbstractSet[str]
    strict_names: bool
    extra_non_suppressing_ctx_managers: AbstractSet[str]
    strict_with_suppression: bool
    depth: int


_UNSET_CTX_FIELD: Any = object()


def _run_child_block(
    statements: Sequence[ast.AST],
    *,
    definite: Set[str],
    conditional: Set[str],
    may_deleted: Set[str],
    must_deleted: Optional[Set[str]],
    live_in_deleted: Optional[Set[str]],
    suppress_aliases: _SuppressProvenance,
    ctx: _TryFlowContext,
    known_safe_names: Optional[AbstractSet[str]] = None,
    loop_control_terminates: Optional[bool] = None,
    loop_jumps: Any = _UNSET_CTX_FIELD,
    ever_bound: Any = _UNSET_CTX_FIELD,
) -> _BlockAssignmentResult:
    """Invokes _analyze_block_assignment_impl at ctx.depth + 1 with shared invariant context."""
    return _analyze_block_assignment_impl(
        statements,
        definite=definite,
        conditional=conditional,
        may_deleted=may_deleted,
        must_deleted=must_deleted,
        live_in_deleted=live_in_deleted,
        suppress_aliases=suppress_aliases,
        active_nonlocals=ctx.active_nonlocals,
        fn_globals=ctx.fn_globals,
        loop_control_terminates=(
            ctx.loop_control_terminates
            if loop_control_terminates is None
            else loop_control_terminates
        ),
        loop_jumps=ctx.loop_jumps if loop_jumps is _UNSET_CTX_FIELD else loop_jumps,
        ever_bound=ctx.ever_bound if ever_bound is _UNSET_CTX_FIELD else ever_bound,
        known_safe_names=(
            ctx.known_safe_names if known_safe_names is None else known_safe_names
        ),
        strict_names=ctx.strict_names,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        strict_with_suppression=ctx.strict_with_suppression,
        _depth=ctx.depth + 1,
    )


def _join_branch_flow_states(states: Sequence[_BranchFlowState]) -> _BranchFlowState:
    """Computes the conservative lattice join across one or more control-flow branch states."""
    def_out = set.intersection(*(s.definite for s in states))
    may_bound_out = set.union(*(s.definite | s.conditional for s in states))
    may_del_out = set.union(*(s.may_del for s in states))
    must_del_out = set.intersection(*(s.must_del for s in states))
    return _BranchFlowState(
        definite=def_out,
        conditional=may_bound_out - def_out,
        may_del=may_del_out,
        must_del=must_del_out,
    )


def _overwrite_flow_state(
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    state: _BranchFlowState,
) -> None:
    """Replaces in-place flow tracking sets with the joined _BranchFlowState."""
    definite.clear()
    definite.update(state.definite)
    conditional.clear()
    conditional.update(state.conditional)
    may_del.clear()
    may_del.update(state.may_del)
    must_del.clear()
    must_del.update(state.must_del)


def _merge_hazard_provenance(
    base: _SuppressProvenance,
    hazard_src: _SuppressProvenance,
) -> _SuppressProvenance:
    """Preserves base's reachable bindings while unioning cross-scope and exit-protocol side effects."""
    merged = replace(
        base,
        cross_scope_mutated=base.cross_scope_mutated | hazard_src.cross_scope_mutated,
        cross_scope_suppressors=(
            base.cross_scope_suppressors | hazard_src.cross_scope_suppressors
        ),
        cross_scope_suspicious=(
            base.cross_scope_suspicious | hazard_src.cross_scope_suspicious
        ),
        exit_patch_targets=base.exit_patch_targets | hazard_src.exit_patch_targets,
        has_exit_patch=base.has_exit_patch or hazard_src.has_exit_patch,
        rebound_factory_targets=(
            base.rebound_factory_targets | hazard_src.rebound_factory_targets
        ),
    )
    if (
        hazard_src.has_exit_patch
        or hazard_src.rebound_factory_targets
        or hazard_src.exit_patch_targets
    ):
        return merged.with_exit_protocol_patches(
            hazard_src.exit_patch_targets,
            has_patch=hazard_src.has_exit_patch,
            rebound_attrs=hazard_src.rebound_factory_targets,
        )
    return merged


class _BlockAssignmentResult(NamedTuple):
    """Result of single-pass block definite/conditional assignment, termination, and provenance analysis."""

    definite: Set[str]
    conditional: Set[str]
    terminated: bool
    prov: _SuppressProvenance
    mid_prov: _SuppressProvenance


class _StmtFlowResult(NamedTuple):
    """Termination, post-statement provenance, and mid-statement exception provenance for a compound stmt."""

    terminated: bool
    prov: _SuppressProvenance
    mid_prov: _SuppressProvenance


def _finalize_surviving_branch_flow(
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    candidates: Sequence[Tuple[bool, _BranchFlowState]],
    terminated_fallback: _BranchFlowState,
    out_prov: _SuppressProvenance,
    mid_prov: _SuppressProvenance,
) -> _StmtFlowResult:
    """Commits joined surviving branch flow states or terminated_fallback when all paths terminate."""
    surviving = [state for is_alive, state in candidates if is_alive]
    target_state = (
        _join_branch_flow_states(surviving) if surviving else terminated_fallback
    )
    _overwrite_flow_state(definite, conditional, may_del, must_del, target_state)
    return _StmtFlowResult(not surviving, out_prov, mid_prov)


def _collect_block_shallow_flow_at_depth_limit(
    statements: Sequence[ast.AST],
) -> Tuple[Set[str], Set[str], bool, bool]:
    """Iteratively collects (raw_assigned, raw_deleted, has_break, has_continue) for depth fallback."""
    summary = _collect_shallow_ast_flow_summary(statements)
    return (
        summary.assigned | summary.except_names,
        summary.deleted | summary.except_names,
        summary.has_break,
        summary.has_continue,
    )


def _apply_depth_bailout_block_flow(
    statements: Sequence[ast.AST],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
) -> _BlockAssignmentResult:
    """Conservatively summarizes a statement sequence when recursion reaches _MAX_AST_RECURSION_DEPTH."""
    raw_assigned, raw_deleted, has_break, has_continue = (
        _collect_block_shallow_flow_at_depth_limit(statements)
    )
    assigned = _filter_scope_names(raw_assigned, active_nonlocals, fn_globals)
    deleted = _filter_scope_names(raw_deleted, active_nonlocals, fn_globals)
    before_def = set(definite)
    live_in_del.update(deleted - before_def)
    definite.difference_update(deleted)
    conditional.update(assigned | (before_def & deleted))
    may_del.update(deleted)
    must_del.difference_update(assigned)
    if ever_bound is not None:
        ever_bound.update(assigned)
    if loop_jumps is not None and (has_break or has_continue):
        loop_jumps.has_jump = True
        if has_break:
            loop_jumps.has_break = True
        loop_jumps.may_bound.update(definite | conditional | assigned)
        loop_jumps.may_deleted.update(may_del)
    local_prov = _conservatively_update_prov_at_depth_limit(statements, local_prov)
    return _BlockAssignmentResult(
        definite, conditional - definite, False, local_prov, local_prov
    )


def _analyze_if_stmt_assignment(
    stmt: ast.If,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
) -> _StmtFlowResult:
    """Analyzes an ast.If statement and updates flow sets in place."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    cur_safe = d_in | set(ctx.known_safe_names)

    b_may = set(may_in)
    b_must = set(must_in)
    b_live: Set[str] = set()
    b_res = _run_child_block(
        stmt.body,
        definite=set(d_in),
        conditional=set(c_in),
        may_deleted=b_may,
        must_deleted=b_must,
        live_in_deleted=b_live,
        suppress_aliases=local_prov,
        ctx=ctx,
        known_safe_names=cur_safe,
    )
    b_state = _BranchFlowState(b_res.definite, b_res.conditional, b_may, b_must)

    o_live: Set[str] = set()
    if stmt.orelse:
        o_may = set(may_in)
        o_must = set(must_in)
        o_res = _run_child_block(
            stmt.orelse,
            definite=set(d_in),
            conditional=set(c_in),
            may_deleted=o_may,
            must_deleted=o_must,
            live_in_deleted=o_live,
            suppress_aliases=local_prov,
            ctx=ctx,
            known_safe_names=cur_safe,
        )
        o_term = o_res.terminated
        o_prov = o_res.prov
        o_mid = o_res.mid_prov
        o_state = _BranchFlowState(o_res.definite, o_res.conditional, o_may, o_must)
    else:
        o_term = False
        o_prov = local_prov
        o_mid = local_prov
        o_state = _BranchFlowState(d_in, c_in, may_in, must_in)

    all_prov = b_res.prov.merge_branch_conservative(o_prov)
    surviving_provs: List[_SuppressProvenance] = []
    if not b_res.terminated:
        surviving_provs.append(b_res.prov)
    if not o_term:
        surviving_provs.append(o_prov)
    out_prov = _merge_hazard_provenance(
        _merge_surviving_branches(
            surviving_provs,
            (all_prov,),
            lambda a, b: a.merge_branch_conservative(b),
        ),
        all_prov,
    )
    out_mid = b_res.mid_prov.merge_branch_conservative(o_mid)
    live_in_del.update((b_live | o_live) - d_in)

    return _finalize_surviving_branch_flow(
        definite,
        conditional,
        may_del,
        must_del,
        ((not b_res.terminated, b_state), (not o_term, o_state)),
        _join_branch_flow_states((b_state, o_state)),
        out_prov,
        out_mid,
    )


def _analyze_with_stmt_assignment(
    stmt: Union[ast.With, ast.AsyncWith],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    bind_def_cond: Any,
    apply_expr_effects: Any,
) -> _StmtFlowResult:
    """Analyzes a With/AsyncWith statement and updates flow sets in place."""
    header_prov = local_prov
    hdr = _summarize_with_header_steps(
        list(
            _iter_with_item_provenance_steps(
                stmt,
                local_prov,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
                strict_with_suppression=ctx.strict_with_suppression,
            )
        ),
        local_prov,
    )
    item_steps = hdr.steps
    has_unproven_exit = hdr.has_unproven_exit
    for step in item_steps:
        item = step.item
        apply_expr_effects(item.context_expr, is_cond=step.after_prior_sup)
        if item.optional_vars is not None:
            if step.opt_vars_conditional:
                bind_def_cond((), _extract_store_names_in_node(item.optional_vars))
            else:
                bind_def_cond(_extract_definite_target_names(item.optional_vars))
            apply_expr_effects(item.optional_vars, is_cond=step.opt_vars_conditional)
        local_prov = step.post_item_prov
    w_mid_base = header_prov.merge_branch_conservative(local_prov)
    cur_safe = definite | set(ctx.known_safe_names)
    safe_prefix, risky_suffix = _split_with_body_by_mode(
        stmt,
        header_prov,
        body_suppress_aliases=local_prov,
        item_steps=item_steps,
        safe_names=cur_safe,
        strict_names=ctx.strict_names,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        strict_with_suppression=ctx.strict_with_suppression,
        _depth=ctx.depth + 1,
    )

    def _eval_prefix(
        prefix: Sequence[ast.stmt],
    ) -> Tuple[Tuple[_SuppressProvenance, _SuppressProvenance], bool]:
        pre_res = _run_child_block(
            prefix,
            definite=definite,
            conditional=conditional,
            may_deleted=may_del,
            must_deleted=must_del,
            live_in_deleted=live_in_del,
            suppress_aliases=local_prov,
            ctx=ctx,
            known_safe_names=cur_safe,
        )
        pre_mid = w_mid_base.merge_branch_conservative(
            pre_res.mid_prov
        ).merge_branch_conservative(pre_res.prov)
        return (pre_res.prov, pre_mid), pre_res.terminated

    def _eval_suffix(
        suffix: Sequence[ast.stmt],
        pre_state: Tuple[_SuppressProvenance, _SuppressProvenance],
    ) -> Tuple[_SuppressProvenance, _SuppressProvenance]:
        pre_prov, pre_mid = pre_state
        d_pre = set(definite)
        c_pre = set(conditional)
        may_pre = set(may_del)
        must_pre = set(must_del)
        w_may = set(may_pre)
        w_must = set(must_pre)
        w_live: Set[str] = set()
        w_ever: Set[str] = set()
        suf_res = _run_child_block(
            suffix,
            definite=set(d_pre),
            conditional=set(c_pre),
            may_deleted=w_may,
            must_deleted=w_must,
            live_in_deleted=w_live,
            suppress_aliases=pre_prov,
            ctx=ctx,
            ever_bound=w_ever,
            known_safe_names=d_pre | set(ctx.known_safe_names),
        )
        if ctx.ever_bound is not None:
            ctx.ever_bound.update(w_ever)
        out_prov = pre_prov.merge_branch_conservative(suf_res.prov)
        out_mid = pre_mid.merge_branch_conservative(
            suf_res.mid_prov
        ).merge_branch_conservative(suf_res.prov)
        live_in_del.update(w_live - d_pre)
        all_risky_dels = _filter_scope_names(
            _extract_block_combined_deletions(suffix),
            ctx.active_nonlocals,
            ctx.fn_globals,
        )
        sup_def = d_pre - all_risky_dels
        sup_state = _BranchFlowState(
            definite=sup_def,
            conditional=(d_pre | c_pre | w_ever) - sup_def,
            may_del=may_pre | all_risky_dels,
            must_del=must_pre - w_ever,
        )
        branches: List[_BranchFlowState] = [sup_state]
        if not suf_res.terminated:
            branches.append(
                _BranchFlowState(suf_res.definite, suf_res.conditional, w_may, w_must)
            )
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(branches)
        )
        return out_prov, out_mid

    def _finalize(
        final_state: Tuple[_SuppressProvenance, _SuppressProvenance],
        terminated: bool,
    ) -> _StmtFlowResult:
        out_prov, out_mid = final_state
        return _StmtFlowResult(
            terminated,
            _apply_with_exit_invalidation(out_prov, has_unproven_exit),
            _apply_with_exit_invalidation(out_mid, has_unproven_exit),
        )

    return _run_with_cfg(
        safe_prefix,
        risky_suffix,
        eval_prefix=_eval_prefix,
        eval_suffix=_eval_suffix,
        finalize_exit=_finalize,
    )


class _TryHandlersResult(NamedTuple):
    """Aggregated flow state and provenance across all except/except* handlers."""

    target_branches: List[_BranchFlowState]
    pre_finally_term: bool
    all_branches: List[_BranchFlowState]
    final_prov: _SuppressProvenance
    post_try_prov: _SuppressProvenance
    h_ever_all: Set[str]
    mid_prov: _SuppressProvenance


def _analyze_try_except_handlers(
    handlers: Sequence[ast.AST],
    *,
    is_trystar: bool,
    h_entry: _BranchFlowState,
    succ_state: _BranchFlowState,
    succ_term: bool,
    try_body_prov: _SuppressProvenance,
    handler_base_prov: _SuppressProvenance,
    d_in: Set[str],
    raw_try_del: Set[str],
    live_in_del: Set[str],
    cur_safe: AbstractSet[str],
    inner_jumps: Optional[_LoopJumpSummary],
    ctx: _TryFlowContext,
) -> _TryHandlersResult:
    """Evaluates except / except* handler branches and merges surviving provenance."""
    surviving: List[_BranchFlowState] = []
    all_branches: List[_BranchFlowState] = [succ_state]
    surviving_provs: List[_SuppressProvenance] = []
    if not succ_term:
        surviving.append(succ_state)
        surviving_provs.append(try_body_prov)
    final_prov = handler_base_prov.merge_branch_conservative(try_body_prov)
    h_mid_prov = final_prov
    handler_scan_prov = handler_base_prov

    trystar_acc: Optional[_BranchFlowState] = h_entry if is_trystar else None
    any_trystar_survives = False
    h_ever_all: Set[str] = set()
    cur_h_entry = h_entry

    def _eval_handler(
        h: ast.AST,
    ) -> Tuple[Tuple[_BranchFlowState, _SuppressProvenance], bool]:
        nonlocal cur_h_entry, final_prov, h_mid_prov, handler_scan_prov
        h_body = list(getattr(h, "body", []))
        h_type = getattr(h, "type", None)
        if isinstance(h_type, ast.AST):
            d_ht, c_ht = _walrus_assignment_in_expr(h_type, is_conditional=True)
            ht_filt = _filter_scope_names(
                d_ht | c_ht, ctx.active_nonlocals, ctx.fn_globals
            )
            h_ever_all.update(ht_filt)
            if ctx.ever_bound is not None:
                ctx.ever_bound.update(ht_filt)
            cur_h_entry = _BranchFlowState(
                definite=cur_h_entry.definite,
                conditional=(cur_h_entry.conditional | ht_filt) - cur_h_entry.definite,
                may_del=cur_h_entry.may_del,
                must_del=cur_h_entry.must_del - ht_filt,
            )
        base_h = (
            _join_branch_flow_states((cur_h_entry, trystar_acc))
            if (is_trystar and trystar_acc is not None)
            else cur_h_entry
        )
        h_name = getattr(h, "name", None)
        h_bound: Set[str] = set()
        if isinstance(h_name, str) and _can_bind_in_fn_scope(
            h_name, ctx.active_nonlocals, ctx.fn_globals
        ):
            h_bound.add(h_name)
        h_prov = _advance_clause_entry_prov(
            handler_scan_prov,
            h_type,
            (h_name,) if isinstance(h_name, str) else (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        h_live: Set[str] = set()
        h_ever: Set[str] = set()
        hmay_in = base_h.may_del - h_bound
        hmust_in = base_h.must_del - h_bound
        hd_in = base_h.definite | h_bound
        h_res = _run_child_block(
            h_body,
            definite=hd_in,
            conditional=base_h.conditional - hd_in,
            may_deleted=hmay_in,
            must_deleted=hmust_in,
            live_in_deleted=h_live,
            suppress_aliases=h_prov,
            ctx=ctx,
            loop_jumps=inner_jumps,
            ever_bound=h_ever,
            known_safe_names=(cur_safe - raw_try_del) | h_bound,
        )
        hd_out = h_res.definite
        hc_out = h_res.conditional
        h_out_prov = (
            h_res.prov.clear_name(h_name) if isinstance(h_name, str) else h_res.prov
        )
        h_body_mid = (
            h_res.mid_prov.clear_name(h_name)
            if isinstance(h_name, str)
            else h_res.mid_prov
        )
        final_prov = final_prov.merge_branch_conservative(h_out_prov)
        h_mid_prov = h_mid_prov.merge_branch_conservative(
            h_body_mid
        ).merge_branch_conservative(h_out_prov)
        h_ever_clean = h_ever - h_bound
        h_ever_all.update(h_ever_clean)
        if ctx.ever_bound is not None:
            ctx.ever_bound.update(h_ever_clean)
        live_in_del.update((h_live - h_bound) - d_in)

        hd_out.difference_update(h_bound)
        hc_out.difference_update(h_bound)
        if h_bound:
            hmay_in.update(h_bound)
            hmust_in.update(h_bound)
        h_state = _BranchFlowState(hd_out, hc_out, hmay_in, hmust_in)
        return (h_state, h_out_prov), h_res.terminated

    def _advance_prior(
        h: ast.AST,
        outcome: Tuple[_BranchFlowState, _SuppressProvenance],
        chain_trystar: bool,
    ) -> None:
        nonlocal handler_scan_prov, trystar_acc, any_trystar_survives
        h_state, h_out_prov = outcome
        h_type = getattr(h, "type", None)
        handler_scan_prov = _merge_prior_clause_guard_and_binds(
            handler_scan_prov,
            h_type,
            (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        if chain_trystar:
            any_trystar_survives = True
            trystar_acc = (
                h_state
                if trystar_acc is None
                else _join_branch_flow_states((trystar_acc, h_state))
            )
            handler_scan_prov = handler_scan_prov.merge_branch_conservative(h_out_prov)

    cfg_res = _run_try_handlers_cfg(
        handlers,
        is_trystar=is_trystar,
        eval_handler=_eval_handler,
        advance_prior_handler=_advance_prior,
    )
    for h_state, _ in cfg_res.handler_outcomes:
        all_branches.append(h_state)
    for h_state, h_out_prov in cfg_res.surviving_outcomes:
        surviving_provs.append(h_out_prov)
        if not is_trystar:
            surviving.append(h_state)

    if is_trystar and any_trystar_survives and trystar_acc is not None:
        surviving.append(trystar_acc)

    post_try_prov = (
        _merge_hazard_provenance(
            _merge_surviving_branches(
                surviving_provs,
                (final_prov,),
                lambda a, b: a.merge_branch_conservative(b),
            ),
            final_prov,
        )
        if surviving_provs
        else final_prov
    )
    final_prov = final_prov.merge_branch_conservative(handler_scan_prov)
    h_mid_prov = h_mid_prov.merge_branch_conservative(final_prov)

    return _TryHandlersResult(
        target_branches=surviving if surviving else all_branches,
        pre_finally_term=not surviving,
        all_branches=all_branches,
        final_prov=final_prov,
        post_try_prov=post_try_prov,
        h_ever_all=h_ever_all,
        mid_prov=h_mid_prov,
    )


class _TryBodyFlowResult(NamedTuple):
    """Single-pass try-body assignment, prefix snapshot, and handler-entry provenance."""

    definite: Set[str]
    conditional: Set[str]
    may_del: Set[str]
    must_del: Set[str]
    live_in_del: Set[str]
    ever_bound: Set[str]
    terminated: bool
    try_body_prov: _SuppressProvenance
    handler_base_prov: _SuppressProvenance
    safe_prefix_len: int
    has_risky_suffix: bool
    pre_def: Set[str]


def _analyze_try_body_split_flow(
    t_body: Sequence[ast.stmt],
    *,
    d_in: Set[str],
    c_in: Set[str],
    may_in: Set[str],
    must_in: Set[str],
    local_prov: _SuppressProvenance,
    cur_safe: AbstractSet[str],
    inner_jumps: Optional[_LoopJumpSummary],
    ctx: _TryFlowContext,
) -> _TryBodyFlowResult:
    """Analyzes a try body in a single linear pass across its non-raising prefix and raising suffix."""
    safe_prefix, risky_suffix = _split_suppress_with_body(
        t_body,
        safe_names=cur_safe,
        strict_names=ctx.strict_names,
        suppress_aliases=local_prov,
        _depth=ctx.depth + 1,
    )
    t_def = set(d_in)
    t_cond = set(c_in)
    t_may = set(may_in)
    t_must = set(must_in)
    t_live: Set[str] = set()
    t_ever: Set[str] = set()
    pre_has_exit = bool(safe_prefix and any(_scan_stmts_shallow_exits(safe_prefix)))
    if safe_prefix:
        pre_res = _run_child_block(
            safe_prefix,
            definite=t_def,
            conditional=t_cond,
            may_deleted=t_may,
            must_deleted=t_must,
            live_in_deleted=t_live,
            suppress_aliases=local_prov,
            ctx=ctx,
            loop_jumps=inner_jumps,
            ever_bound=t_ever,
            known_safe_names=cur_safe,
        )
        t_def = pre_res.definite
        t_cond = pre_res.conditional
        prefix_prov = pre_res.prov
        pre_mid_prov = pre_res.mid_prov
        pre_term = pre_res.terminated
    else:
        prefix_prov = local_prov
        pre_mid_prov = local_prov
        pre_term = False
    pre_def = set(t_def)
    if pre_term or not risky_suffix:
        split_mid = pre_mid_prov if pre_has_exit else prefix_prov
        handler_base_prov = prefix_prov.merge_branch_conservative(split_mid)
        return _TryBodyFlowResult(
            definite=t_def,
            conditional=t_cond,
            may_del=t_may,
            must_del=t_must,
            live_in_del=t_live,
            ever_bound=t_ever,
            terminated=pre_term,
            try_body_prov=prefix_prov,
            handler_base_prov=handler_base_prov,
            safe_prefix_len=len(safe_prefix),
            has_risky_suffix=False,
            pre_def=pre_def,
        )
    suf_res = _run_child_block(
        risky_suffix,
        definite=t_def,
        conditional=t_cond,
        may_deleted=t_may,
        must_deleted=t_must,
        live_in_deleted=t_live,
        suppress_aliases=prefix_prov,
        ctx=ctx,
        loop_jumps=inner_jumps,
        ever_bound=t_ever,
        known_safe_names=cur_safe,
    )
    split_mid = (
        pre_mid_prov.merge_branch_conservative(suf_res.mid_prov)
        if pre_has_exit
        else suf_res.mid_prov
    )
    handler_base_prov = prefix_prov.merge_branch_conservative(
        split_mid
    ).merge_branch_conservative(suf_res.prov)
    return _TryBodyFlowResult(
        definite=suf_res.definite,
        conditional=suf_res.conditional,
        may_del=t_may,
        must_del=t_must,
        live_in_del=t_live,
        ever_bound=t_ever,
        terminated=suf_res.terminated,
        try_body_prov=suf_res.prov,
        handler_base_prov=handler_base_prov,
        safe_prefix_len=len(safe_prefix),
        has_risky_suffix=True,
        pre_def=pre_def,
    )


class _TryFinallyAbruptEntry(NamedTuple):
    """Abrupt-exception entry flow state for a try-statement finally block."""

    pre_try_def: Set[str]
    definite: Set[str]
    conditional: Set[str]
    may_del: Set[str]


def _compute_try_finally_abrupt_entry(
    t_body: Sequence[ast.stmt],
    o_stmts: Sequence[ast.stmt],
    f_body: Sequence[ast.stmt],
    *,
    body_flow: _TryBodyFlowResult,
    o_ever: Set[str],
    d_in: Set[str],
    c_in: Set[str],
    may_in: Set[str],
    raw_try_del: Set[str],
    h_res: _TryHandlersResult,
    local_prov: _SuppressProvenance,
    cur_safe: Set[str],
    ctx: _TryFlowContext,
) -> _TryFinallyAbruptEntry:
    """Computes the conservative flow state when entering finally via an abrupt exit."""
    abrupt_cond = d_in | c_in | body_flow.ever_bound | o_ever | h_res.h_ever_all
    abrupt_may = (
        may_in
        | raw_try_del
        | body_flow.may_del
        | _filter_scope_names(
            _extract_block_combined_deletions(o_stmts),
            ctx.active_nonlocals,
            ctx.fn_globals,
        )
        | set().union(*(s.may_del for s in h_res.all_branches))
    )
    safe_prefix_len = body_flow.safe_prefix_len
    guaranteed_prefix_len = 0
    for s_item in t_body[:safe_prefix_len]:
        if any(_scan_stmts_shallow_exits([s_item])):
            break
        guaranteed_prefix_len += 1
    if (
        not body_flow.has_risky_suffix
        and not body_flow.terminated
        and guaranteed_prefix_len == len(t_body)
    ):
        abrupt_def = set(body_flow.definite) - abrupt_may
    elif guaranteed_prefix_len > 0 and guaranteed_prefix_len == safe_prefix_len:
        abrupt_def = set(body_flow.pre_def) - abrupt_may
    elif (
        guaranteed_prefix_len > 0
        and ctx.depth < 8
        and _extract_block_combined_deletions(f_body)
    ):
        with (
            collect_heuristic_ctx_managers(isolate=True)
        ), collect_wildcard_ctx_managers(isolate=True):
            pre_def = _run_child_block(
                t_body[:guaranteed_prefix_len],
                definite=set(d_in),
                conditional=set(),
                may_deleted=set(),
                must_deleted=None,
                live_in_deleted=None,
                suppress_aliases=local_prov,
                ctx=ctx,
                loop_control_terminates=False,
                loop_jumps=None,
                ever_bound=None,
                known_safe_names=cur_safe,
            ).definite
        abrupt_def = pre_def - abrupt_may
    else:
        abrupt_def = set(d_in) - abrupt_may
    return _TryFinallyAbruptEntry(
        pre_try_def=d_in,
        definite=abrupt_def,
        conditional=abrupt_cond,
        may_del=abrupt_may,
    )


def _analyze_try_finally_clause(
    f_body: Sequence[ast.AST],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    final_prov: _SuppressProvenance,
    post_try_prov: _SuppressProvenance,
    *,
    abrupt_entry: _TryFinallyAbruptEntry,
    inner_jumps: Optional[_LoopJumpSummary],
    ctx: _TryFlowContext,
) -> _StmtFlowResult:
    """Analyzes a try-statement finally block and propagates loop jump effects."""
    entry_def = set(definite)
    post_known_safe = definite | set(ctx.known_safe_names)
    has_ret, has_brk, has_cont = _scan_stmts_shallow_exits(f_body)
    has_unescaped_jump = has_brk or has_cont
    may_suppress_exc = has_ret or has_unescaped_jump

    def _run_finally(
        d_set: Set[str],
        c_set: Set[str],
        may_set: Set[str],
        must_set: Optional[Set[str]] = None,
        *,
        live_del: Optional[Set[str]] = None,
        jumps: Optional[_LoopJumpSummary] = None,
        ever: Optional[Set[str]] = None,
        isolate_telemetry: bool = False,
    ) -> _BlockAssignmentResult:
        with contextlib.ExitStack() as telemetry_stack:
            if isolate_telemetry:
                telemetry_stack.enter_context(
                    collect_heuristic_ctx_managers(isolate=True)
                )
                telemetry_stack.enter_context(
                    collect_wildcard_ctx_managers(isolate=True)
                )
            return _run_child_block(
                f_body,
                definite=d_set,
                conditional=c_set,
                may_deleted=may_set,
                must_deleted=must_set,
                live_in_deleted=live_del,
                suppress_aliases=final_prov,
                ctx=ctx,
                loop_jumps=jumps,
                ever_bound=ever,
                known_safe_names=post_known_safe,
            )

    f_ever: Set[str] = set()
    f_res = _run_finally(
        definite,
        conditional,
        may_del,
        must_del,
        live_del=live_in_del,
        ever=f_ever,
    )
    f_term = f_res.terminated
    f_out_prov = f_res.prov
    if ctx.ever_bound is not None:
        ctx.ever_bound.update(f_ever)
    f_raw_dels = _filter_scope_names(
        _extract_block_combined_deletions(f_body),
        ctx.active_nonlocals,
        ctx.fn_globals,
    )
    if f_raw_dels & (entry_def - abrupt_entry.definite):
        if ctx.depth < 8:
            abrupt_live: Set[str] = set()
            _run_finally(
                set(abrupt_entry.definite),
                set(abrupt_entry.conditional),
                set(abrupt_entry.may_del),
                live_del=abrupt_live,
                isolate_telemetry=True,
            )
            live_in_del.update(abrupt_live - abrupt_entry.pre_try_def)
        else:
            live_in_del.update(
                (f_raw_dels - abrupt_entry.definite) - abrupt_entry.pre_try_def
            )
    if ctx.loop_jumps is not None:
        if has_unescaped_jump:
            if ctx.depth < 8:
                f_jumps = _LoopJumpSummary()
                _run_finally(
                    set(),
                    set(abrupt_entry.conditional),
                    set(abrupt_entry.may_del),
                    jumps=f_jumps,
                    isolate_telemetry=True,
                )
                if f_jumps.has_jump:
                    ctx.loop_jumps.has_jump = True
                    if f_jumps.has_break:
                        ctx.loop_jumps.has_break = True
                    ctx.loop_jumps.may_bound.update(f_jumps.may_bound)
                    ctx.loop_jumps.may_deleted.update(f_jumps.may_deleted)
            else:
                ctx.loop_jumps.has_jump = True
                if has_brk:
                    ctx.loop_jumps.has_break = True
                ctx.loop_jumps.may_bound.update(
                    abrupt_entry.conditional | definite | conditional | f_ever
                )
                ctx.loop_jumps.may_deleted.update(
                    abrupt_entry.may_del | may_del | f_raw_dels
                )
        if inner_jumps is not None and inner_jumps.has_jump and not f_term:
            ctx.loop_jumps.has_jump = True
            if inner_jumps.has_break:
                ctx.loop_jumps.has_break = True
            if ctx.depth < 8:
                f_standalone_may: Set[str] = set()
                f_standalone_must: Set[str] = set()
                f_sa_res = _run_finally(
                    set(),
                    set(),
                    f_standalone_may,
                    f_standalone_must,
                    isolate_telemetry=True,
                )
                ctx.loop_jumps.may_bound.update(
                    (inner_jumps.may_bound - f_standalone_must)
                    | f_sa_res.definite
                    | f_sa_res.conditional
                )
                ctx.loop_jumps.may_deleted.update(
                    (inner_jumps.may_deleted - f_sa_res.definite) | f_standalone_may
                )
            else:
                ctx.loop_jumps.may_bound.update(
                    inner_jumps.may_bound | definite | conditional | f_ever
                )
                ctx.loop_jumps.may_deleted.update(
                    inner_jumps.may_deleted | may_del | f_raw_dels
                )
    if post_try_prov == final_prov:
        post_f_prov = f_out_prov
    else:
        with (
            collect_heuristic_ctx_managers(isolate=True)
        ), collect_wildcard_ctx_managers(isolate=True):
            post_f_prov = _update_suppress_provenance_from_stmts(
                f_body,
                post_try_prov,
                in_branch=False,
                strict_with_suppression=ctx.strict_with_suppression,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
                _depth=ctx.depth + 1,
            )
    out_prov = (
        post_f_prov.merge_branch_conservative(f_out_prov)
        if may_suppress_exc
        else _merge_hazard_provenance(post_f_prov, f_out_prov)
    )
    f_mid_prov = final_prov.merge_branch_conservative(
        f_res.mid_prov
    ).merge_branch_conservative(out_prov)
    return _StmtFlowResult(f_term, out_prov, f_mid_prov)


def _analyze_try_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
) -> _StmtFlowResult:
    """Analyzes a Try/TryStar statement in a single linear pass and updates flow sets in place."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    t_body = list(getattr(stmt, "body", []))
    o_stmts = list(getattr(stmt, "orelse", []))
    f_body = list(getattr(stmt, "finalbody", []))
    raw_try_del = _filter_scope_names(
        _extract_block_combined_deletions(t_body),
        ctx.active_nonlocals,
        ctx.fn_globals,
    )
    is_trystar = is_trystar_node(stmt)
    cur_safe = d_in | set(ctx.known_safe_names)
    inner_jumps: Optional[_LoopJumpSummary] = (
        _LoopJumpSummary() if (ctx.loop_jumps is not None and f_body) else ctx.loop_jumps
    )
    body_flow = _analyze_try_body_split_flow(
        t_body,
        d_in=d_in,
        c_in=c_in,
        may_in=may_in,
        must_in=must_in,
        local_prov=local_prov,
        cur_safe=cur_safe,
        inner_jumps=inner_jumps,
        ctx=ctx,
    )
    if ctx.ever_bound is not None:
        ctx.ever_bound.update(body_flow.ever_bound)
    live_in_del.update(body_flow.live_in_del - d_in)

    o_ever: Set[str] = set()
    o_mid_prov: Optional[_SuppressProvenance] = None
    try_body_prov = body_flow.try_body_prov
    succ_term = body_flow.terminated
    if o_stmts and not body_flow.terminated:
        o_may = set(body_flow.may_del)
        o_must = set(body_flow.must_del)
        o_live: Set[str] = set()
        o_res = _run_child_block(
            o_stmts,
            definite=set(body_flow.definite),
            conditional=set(body_flow.conditional),
            may_deleted=o_may,
            must_deleted=o_must,
            live_in_deleted=o_live,
            suppress_aliases=try_body_prov,
            ctx=ctx,
            loop_jumps=inner_jumps,
            ever_bound=o_ever,
            known_safe_names=cur_safe | body_flow.definite,
        )
        try_body_prov = o_res.prov
        o_mid_prov = o_res.mid_prov
        if ctx.ever_bound is not None:
            ctx.ever_bound.update(o_ever)
        live_in_del.update(o_live - body_flow.definite)
        succ_state = _BranchFlowState(o_res.definite, o_res.conditional, o_may, o_must)
        if o_res.terminated:
            succ_term = True
    else:
        succ_state = _BranchFlowState(
            body_flow.definite,
            body_flow.conditional,
            body_flow.may_del,
            body_flow.must_del,
        )

    h_entry_def = d_in - (raw_try_del | body_flow.may_del)
    h_entry = _BranchFlowState(
        definite=h_entry_def,
        conditional=(d_in | c_in | body_flow.ever_bound) - h_entry_def,
        may_del=may_in | raw_try_del | body_flow.may_del,
        must_del=must_in - body_flow.ever_bound,
    )
    h_res = _analyze_try_except_handlers(
        list(getattr(stmt, "handlers", [])),
        is_trystar=is_trystar,
        h_entry=h_entry,
        succ_state=succ_state,
        succ_term=succ_term,
        try_body_prov=try_body_prov,
        handler_base_prov=body_flow.handler_base_prov,
        d_in=d_in,
        raw_try_del=raw_try_del,
        live_in_del=live_in_del,
        cur_safe=cur_safe,
        inner_jumps=inner_jumps,
        ctx=ctx,
    )
    _overwrite_flow_state(
        definite,
        conditional,
        may_del,
        must_del,
        _join_branch_flow_states(h_res.target_branches),
    )

    try_mid_prov = (
        h_res.mid_prov.merge_branch_conservative(o_mid_prov)
        if o_mid_prov is not None
        else h_res.mid_prov
    )
    f_term = False
    if f_body:
        abrupt_entry = _compute_try_finally_abrupt_entry(
            t_body,
            o_stmts,
            f_body,
            body_flow=body_flow,
            o_ever=o_ever,
            d_in=d_in,
            c_in=c_in,
            may_in=may_in,
            raw_try_del=raw_try_del,
            h_res=h_res,
            local_prov=local_prov,
            cur_safe=cur_safe,
            ctx=ctx,
        )
        finally_entry_prov = (
            h_res.final_prov.merge_branch_conservative(o_mid_prov)
            if o_mid_prov is not None
            else h_res.final_prov
        )
        f_res = _analyze_try_finally_clause(
            f_body,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            finally_entry_prov,
            h_res.post_try_prov,
            abrupt_entry=abrupt_entry,
            inner_jumps=inner_jumps,
            ctx=ctx,
        )
        f_term = f_res.terminated
        local_prov = f_res.prov
        try_mid_prov = try_mid_prov.merge_branch_conservative(f_res.mid_prov)
    else:
        local_prov = h_res.post_try_prov

    return _StmtFlowResult(f_term or h_res.pre_finally_term, local_prov, try_mid_prov)


def _analyze_match_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    apply_expr_effects: Any,
) -> _StmtFlowResult:
    """Analyzes a Match statement and updates flow sets in place."""
    subject = getattr(stmt, "subject", None)
    if isinstance(subject, ast.AST):
        local_prov = _advance_prov_through_opt_expr(
            subject,
            local_prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        apply_expr_effects(subject)
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    cur_safe = d_in | set(ctx.known_safe_names)

    prior_guard_cond: Set[str] = set()
    merged_match_prov = local_prov
    match_mid_prov = local_prov

    def _eval_case(
        clause: _CompoundClause,
        scan_prov: _SuppressProvenance,
    ) -> Tuple[Tuple[_BranchFlowState, _SuppressProvenance, Set[str]], bool]:
        nonlocal merged_match_prov, match_mid_prov
        raw_pattern_stores: Set[str] = set(clause.bound_names)
        pattern_stores = _filter_scope_names(
            raw_pattern_stores, ctx.active_nonlocals, ctx.fn_globals
        )
        guard = clause.guard_or_header
        d_g: Set[str] = set()
        c_g: Set[str] = set()
        if isinstance(guard, ast.AST):
            d_g_raw, c_g_raw = _walrus_assignment_in_expr(guard, is_conditional=False)
            d_g = _filter_scope_names(d_g_raw, ctx.active_nonlocals, ctx.fn_globals)
            c_g = _filter_scope_names(c_g_raw, ctx.active_nonlocals, ctx.fn_globals)
        case_init_def = pattern_stores | d_g
        if ctx.ever_bound is not None:
            ctx.ever_bound.update(case_init_def | c_g)
        case_prov = scan_prov
        for p_s in raw_pattern_stores:
            case_prov = case_prov.clear_name(p_s)
        case_prov = _advance_clause_entry_prov(
            case_prov,
            guard,
            (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        c_def_in = d_in | case_init_def
        c_cond_in = (c_in | prior_guard_cond | c_g) - c_def_in
        c_may = may_in - case_init_def
        c_must = must_in - case_init_def - c_g - prior_guard_cond
        c_live: Set[str] = set()
        c_res = _run_child_block(
            clause.body,
            definite=c_def_in,
            conditional=c_cond_in,
            may_deleted=c_may,
            must_deleted=c_must,
            live_in_deleted=c_live,
            suppress_aliases=case_prov,
            ctx=ctx,
            known_safe_names=cur_safe | case_init_def,
        )
        merged_match_prov = merged_match_prov.merge_branch_conservative(c_res.prov)
        match_mid_prov = match_mid_prov.merge_branch_conservative(
            c_res.mid_prov
        ).merge_branch_conservative(c_res.prov)
        live_in_del.update((c_live - case_init_def) - d_in)
        c_state = _BranchFlowState(c_res.definite, c_res.conditional, c_may, c_must)
        guard_cond_delta = (case_init_def | c_g) if guard is not None else set()
        if guard_cond_delta:
            prior_guard_cond.update(guard_cond_delta)
        return (c_state, c_res.prov, raw_pattern_stores), c_res.terminated

    cfg_res = _run_match_cfg(
        _lower_compound_clauses(stmt),
        initial_prior=local_prov,
        eval_case=_eval_case,
        advance_prior=lambda p, cl: _merge_prior_compound_clause_prov(
            p,
            cl,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        ),
    )
    local_prov = cfg_res.final_prior
    all_cases: List[_BranchFlowState] = [s for s, _, _ in cfg_res.case_outcomes]
    surviving: List[_BranchFlowState] = [s for s, _, _ in cfg_res.surviving_outcomes]
    surviving_provs: List[_SuppressProvenance] = [
        p for _, p, _ in cfg_res.surviving_outcomes
    ]
    all_prov = (
        merged_match_prov
        if cfg_res.has_irrefutable
        else local_prov.merge_branch_conservative(merged_match_prov)
    )
    if not cfg_res.has_irrefutable:
        surviving_provs.append(local_prov)
        fallthrough_state = _BranchFlowState(
            definite=d_in,
            conditional=(c_in | prior_guard_cond) - d_in,
            may_del=may_in,
            must_del=must_in - prior_guard_cond,
        )
        surviving.append(fallthrough_state)
        all_cases.append(fallthrough_state)
    out_prov = _merge_hazard_provenance(
        _merge_surviving_branches(
            surviving_provs,
            (all_prov,),
            lambda a, b: a.merge_branch_conservative(b),
        ),
        all_prov,
    )
    match_mid_prov = match_mid_prov.merge_branch_conservative(all_prov)

    if surviving:
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(surviving)
        )
        return _StmtFlowResult(False, out_prov, match_mid_prov)
    if all_cases:
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(all_cases)
        )
        return _StmtFlowResult(True, out_prov, match_mid_prov)
    return _StmtFlowResult(False, out_prov, match_mid_prov)


def _merge_loop_carried_suppress_provenance(
    body_stmts: Sequence[ast.AST],
    entry_prov: _SuppressProvenance,
    *,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Conservatively merges loop-body provenance hazards back into loop-entry provenance via bounded fixed point."""
    if not body_stmts:
        return entry_prov
    loop_ctx = _ProvenanceWalkContext(
        in_branch=True,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        depth=_depth,
    )
    with (
        collect_heuristic_ctx_managers(isolate=True)
    ), collect_wildcard_ctx_managers(isolate=True):
        loop_fixed, _ = _run_loop_body_prov_fixpoint(body_stmts, entry_prov, loop_ctx)
    return loop_fixed


def _analyze_loop_stmt_assignment(
    body_stmts: Sequence[ast.AST],
    orelse_stmts: Sequence[ast.AST],
    target_raw_names: AbstractSet[str],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    body_prov: Optional[_SuppressProvenance] = None,
    ctx: _TryFlowContext,
) -> _StmtFlowResult:
    """Analyzes a While/For/AsyncFor loop with break/continue jump tracking and orelse composition."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    target_filt = _filter_scope_names(
        target_raw_names, ctx.active_nonlocals, ctx.fn_globals
    )
    if ctx.ever_bound is not None and target_filt:
        ctx.ever_bound.update(target_filt)

    body_jumps = _LoopJumpSummary()
    b_def_in = d_in | target_filt
    b_cond_in = c_in - b_def_in
    b_may = may_in - target_filt
    b_must = must_in - target_filt
    b_live: Set[str] = set()
    eff_body_prov = body_prov if body_prov is not None else local_prov
    loop_entry_prov = _merge_loop_carried_suppress_provenance(
        body_stmts,
        eff_body_prov,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        _depth=ctx.depth + 1,
    )
    b_res = _run_child_block(
        body_stmts,
        definite=b_def_in,
        conditional=b_cond_in,
        may_deleted=b_may,
        must_deleted=b_must,
        live_in_deleted=b_live,
        suppress_aliases=loop_entry_prov,
        ctx=ctx,
        loop_control_terminates=True,
        loop_jumps=body_jumps,
        known_safe_names=d_in | set(ctx.known_safe_names) | target_raw_names,
    )
    live_in_del.update((b_live - target_raw_names) - d_in)
    merged_prov = local_prov.merge_branch_conservative(b_res.prov)
    loop_mid_prov = local_prov.merge_branch_conservative(
        b_res.mid_prov
    ).merge_branch_conservative(b_res.prov)

    iter_dels = (b_may - may_in) | body_jumps.may_deleted
    iter_bounds = target_filt | b_res.definite | b_res.conditional | body_jumps.may_bound
    exh_def = d_in - iter_dels
    exh_state = _BranchFlowState(
        definite=exh_def,
        conditional=(d_in | c_in | iter_bounds) - exh_def,
        may_del=may_in | iter_dels,
        must_del=must_in - iter_bounds,
    )
    if not orelse_stmts:
        _overwrite_flow_state(definite, conditional, may_del, must_del, exh_state)
        return _StmtFlowResult(False, merged_prov, loop_mid_prov)

    o_may = set(exh_state.may_del)
    o_must = set(exh_state.must_del)
    o_live: Set[str] = set()
    o_res = _run_child_block(
        orelse_stmts,
        definite=set(exh_state.definite),
        conditional=set(exh_state.conditional),
        may_deleted=o_may,
        must_deleted=o_must,
        live_in_deleted=o_live,
        suppress_aliases=merged_prov,
        ctx=ctx,
        known_safe_names=exh_state.definite | set(ctx.known_safe_names),
    )
    live_in_del.update(o_live - exh_state.definite)
    merged_prov = merged_prov.merge_branch_conservative(o_res.prov)
    loop_mid_prov = loop_mid_prov.merge_branch_conservative(
        o_res.mid_prov
    ).merge_branch_conservative(o_res.prov)
    after_orelse = _BranchFlowState(o_res.definite, o_res.conditional, o_may, o_must)

    return _finalize_surviving_branch_flow(
        definite,
        conditional,
        may_del,
        must_del,
        ((not o_res.terminated, after_orelse), (body_jumps.has_break, exh_state)),
        after_orelse,
        merged_prov,
        loop_mid_prov,
    )


def _analyze_simple_non_terminating_stmt(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    bind_def_cond: Callable[[Iterable[str], Iterable[str]], None],
    apply_expr_effects: Any,
) -> _SuppressProvenance:
    """Updates provenance, definite/conditional bindings, and deletions for non-compound stmts."""
    next_prov = _update_suppress_provenance_from_stmt(
        stmt,
        local_prov,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        _depth=ctx.depth,
    )
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            bind_def_cond(_extract_definite_target_names(target), ())
            apply_expr_effects(target)
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.AnnAssign):
        apply_expr_effects(stmt.annotation, is_cond=True)
        if stmt.value is not None:
            bind_def_cond(_extract_definite_target_names(stmt.target), ())
            apply_expr_effects(stmt.target)
            apply_expr_effects(stmt.value)
        elif not isinstance(stmt.target, ast.Name):
            apply_expr_effects(stmt.target)
    elif isinstance(stmt, ast.AugAssign):
        bind_def_cond(_extract_definite_target_names(stmt.target), ())
        apply_expr_effects(stmt.target)
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.Expr):
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            apply_expr_effects(target)
            for node in ast.walk(target):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Del)
                    and _can_bind_in_fn_scope(
                        node.id, ctx.active_nonlocals, ctx.fn_globals
                    )
                ):
                    if node.id not in definite:
                        live_in_del.add(node.id)
                    definite.discard(node.id)
                    conditional.discard(node.id)
                    may_del.add(node.id)
                    must_del.add(node.id)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for hdr in iter_nested_scope_header_nodes(stmt):
            apply_expr_effects(hdr)
        bind_def_cond((stmt.name,), ())
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        imp_names: Set[str] = set()
        _record_import_names(stmt, imp_names)
        bind_def_cond(imp_names, ())
    elif isinstance(stmt, ast.Assert):
        apply_expr_effects(stmt.test)
        if stmt.msg is not None:
            apply_expr_effects(stmt.msg, is_cond=True)
    else:
        ta_name = _extract_type_alias_name(stmt)
        if ta_name is not None:
            bind_def_cond((ta_name,), ())
    return next_prov


def _analyze_compound_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    bind_def_cond: Callable[[Iterable[str], Iterable[str]], None],
    apply_expr_effects: Any,
) -> Optional[_StmtFlowResult]:
    """Dispatches compound statements (If/While/For/With/Try/Match); returns _StmtFlowResult or None."""
    if isinstance(stmt, ast.If):
        local_prov = _advance_prov_through_opt_expr(
            stmt.test,
            local_prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        apply_expr_effects(stmt.test)
        return _analyze_if_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
        )

    if isinstance(stmt, (ast.While, ast.For, ast.AsyncFor)):
        if isinstance(stmt, ast.While):
            local_prov = _advance_prov_through_opt_expr(
                stmt.test,
                local_prov,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
            apply_expr_effects(stmt.test)
            loop_targets: AbstractSet[str] = frozenset()
            loop_body_prov = local_prov
        else:
            local_prov, loop_body_prov = _advance_prov_through_for_header(
                stmt,
                local_prov,
                allow_positive=True,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
            apply_expr_effects(stmt.iter)
            apply_expr_effects(stmt.target, is_cond=True)
            loop_targets = _extract_definite_target_names(stmt.target)
        return _analyze_loop_stmt_assignment(
            stmt.body,
            stmt.orelse,
            loop_targets,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            body_prov=loop_body_prov,
            ctx=ctx,
        )

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _analyze_with_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
            bind_def_cond=bind_def_cond,
            apply_expr_effects=apply_expr_effects,
        )
    if is_try_or_trystar_node(stmt):
        return _analyze_try_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
        )
    if is_match_node(stmt):  # pragma: no cover (py310+)
        return _analyze_match_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
            apply_expr_effects=apply_expr_effects,
        )

    return None


def _analyze_block_assignment_impl(
    statements: Sequence[ast.AST],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    must_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    active_nonlocals: Optional[Set[str]] = None,
    fn_globals: AbstractSet[str] = frozenset(),
    loop_control_terminates: bool = False,
    loop_jumps: Optional[_LoopJumpSummary] = None,
    ever_bound: Optional[Set[str]] = None,
    known_safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _BlockAssignmentResult:
    """Unified single-pass analyzer for definite/conditional bindings, deletions, termination, and provenance."""
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()
    may_del = may_deleted if may_deleted is not None else set()
    must_del = must_deleted if must_deleted is not None else set()
    live_in_del = live_in_deleted if live_in_deleted is not None else set()
    local_prov = _coerce_suppress_provenance(suppress_aliases)
    mid_prov = local_prov
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _apply_depth_bailout_block_flow(
            statements,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_jumps=loop_jumps,
            ever_bound=ever_bound,
        )

    def _bind_def_cond(d_raw: Iterable[str], c_raw: Iterable[str] = ()) -> None:
        d_filt = _filter_scope_names(d_raw, active_nonlocals, fn_globals)
        c_filt = _filter_scope_names(c_raw, active_nonlocals, fn_globals)
        if d_filt:
            definite.update(d_filt)
            conditional.difference_update(d_filt)
            may_del.difference_update(d_filt)
            must_del.difference_update(d_filt)
            if ever_bound is not None:
                ever_bound.update(d_filt)
        if c_filt:
            conditional.update(c_filt - definite)
            must_del.difference_update(c_filt)
            if ever_bound is not None:
                ever_bound.update(c_filt)

    def _apply_expr_effects(
        expr_node: Optional[ast.AST], *, is_cond: bool = False
    ) -> None:
        if not isinstance(expr_node, ast.AST):
            return
        d_expr, c_expr = _walrus_assignment_in_expr(expr_node, is_conditional=is_cond)
        _bind_def_cond(d_expr, c_expr)

    ctx = _TryFlowContext(
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=known_safe_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        depth=_depth,
    )

    for stmt in statements:
        before_def = set(definite)
        if isinstance(stmt, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt,
                local_prov,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                _depth=_depth,
            )
            mid_prov = mid_prov.merge_branch_conservative(local_prov)
            if isinstance(stmt, ast.Raise):
                _apply_expr_effects(stmt.exc)
                _apply_expr_effects(stmt.cause)
                term_flag = True
            elif isinstance(stmt, ast.Return):
                _apply_expr_effects(stmt.value)
                term_flag = True
            else:
                if loop_jumps is not None:
                    loop_jumps.has_jump = True
                    if isinstance(stmt, ast.Break):
                        loop_jumps.has_break = True
                    loop_jumps.may_bound.update(definite | conditional)
                    loop_jumps.may_deleted.update(may_del)
                term_flag = bool(loop_control_terminates)
            may_del.difference_update(definite - before_def)
            return _BlockAssignmentResult(
                definite, conditional - definite, term_flag, local_prov, mid_prov
            )

        comp_res = _analyze_compound_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
            bind_def_cond=_bind_def_cond,
            apply_expr_effects=_apply_expr_effects,
        )
        if comp_res is not None:
            local_prov = comp_res.prov
            mid_prov = mid_prov.merge_branch_conservative(
                comp_res.mid_prov
            ).merge_branch_conservative(local_prov)
            if comp_res.terminated:
                may_del.difference_update(definite - before_def)
                return _BlockAssignmentResult(
                    definite, conditional - definite, True, local_prov, mid_prov
                )
        else:
            local_prov = _analyze_simple_non_terminating_stmt(
                stmt,
                definite,
                conditional,
                may_del,
                must_del,
                live_in_del,
                local_prov,
                ctx=ctx,
                bind_def_cond=_bind_def_cond,
                apply_expr_effects=_apply_expr_effects,
            )
            mid_prov = mid_prov.merge_branch_conservative(local_prov)

        may_del.difference_update(definite - before_def)

    return _BlockAssignmentResult(
        definite, conditional - definite, False, local_prov, mid_prov
    )


def _analyze_block_assignment(
    statements: Sequence[ast.stmt],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Analyzes a sequence of statements to determine unconditionally and conditionally assigned variables."""
    res = _analyze_block_assignment_impl(
        statements,
        definite=definite,
        conditional=conditional,
        may_deleted=may_deleted,
        live_in_deleted=live_in_deleted,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    return res.definite, res.conditional


def _extract_block_definite_bindings_and_term(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Tuple[Set[str], bool]:
    """Extracts (unconditionally_bound_names, block_terminates) across a statement sequence.

    Passes ``loop_control_terminates=True`` because callers inspect pre-unit or
    intra-loop compound statements (``if``/``try``/``match``) where a ``break`` or
    ``continue`` branch exits the clause without falling through to statements
    immediately following the compound statement.
    """
    res = _analyze_block_assignment_impl(
        stmts,
        definite=set(initial_bound),
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=True,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    return res.definite, res.terminated


def _extract_block_definite_bindings(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Set[str]:
    """Extracts variable names unconditionally bound across a statement sequence."""
    return _extract_block_definite_bindings_and_term(
        stmts,
        active_nonlocals,
        initial_bound,
        fn_globals,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )[0]
