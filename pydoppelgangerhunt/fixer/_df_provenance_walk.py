"""Forward suppression provenance walker and with-body splitting engine."""

from __future__ import annotations

import ast
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.config import record_wildcard_ctx_manager_match
from pydoppelgangerhunt._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
    is_trystar_node,
    is_type_alias_node,
)
from pydoppelgangerhunt.fixer._df_cfg import (
    _CompoundClause,
    _apply_with_exit_invalidation,
    _extract_type_alias_or_pattern_name,
    _finally_may_suppress_active_exception,
    _loop_orelse_terminates,
    _lower_compound_clauses,
    _merge_surviving_branches,
    _run_match_cfg,
    _run_try_handlers_cfg,
    _run_with_cfg,
    _scan_stmts_shallow_exits,
    _summarize_with_header_steps,
    _try_has_abrupt_entry,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _extract_single_node_local_names,
)
from pydoppelgangerhunt.fixer._df_non_raising import (
    _apply_non_raising_expr_walrus_updates,
    _apply_non_raising_stmt_safe_updates,
    _check_and_advance_non_raising_seq,
    _is_non_raising_expr,
    _is_valid_suppress_or_null_call_signature,
    _non_raising_stmt_terminates,
)
from pydoppelgangerhunt.fixer._df_prov_exprs import (
    _advance_clause_entry_prov,
    _advance_prov_through_exprs,
    _advance_prov_through_for_header,
    _advance_prov_through_opt_expr,
    _bind_assign_target_provenance,
    _bind_expr_target_provenance,
    _merge_prior_clause_guard_and_binds,
    _merge_prior_compound_clause_prov,
    _update_import_stmt_provenance,
    _update_suppress_provenance_from_expr_walrus,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _collect_func_params as _extract_arg_names,
    _extract_pattern_bound_name,
    _record_import_names,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _EMPTY_SUPPRESS_PROVENANCE,
    _SAFE_EXCEPTION_TYPE_NAMES,
    _SHADOWABLE_BUILTIN_CTX_NAMES,
    _SUPPRESSING_CTX_MANAGER_NAMES,
    _SuppressProvenance,
    _coerce_suppress_provenance,
    _ident_has_suspicious_suppress_token,
)
from pydoppelgangerhunt.fixer._df_with_rules import (
    _DEFERRED_DEF_SCOPE_TYPES,
    _WithItemProvenanceStep,
    _class_def_is_proven_non_suppressing_in_exit,
    _class_def_may_suppress_in_exit,
    _classify_with_item_suppression,
    _clear_store_target_provenance,
    _conservatively_update_prov_at_depth_limit,
    _exprs_may_invoke_cross_scope_mutator,
    _extract_definite_target_names,
    _extract_shallow_deleted_names,
    _find_ctx_manager_wildcard_match,
    _format_ctx_manager_display_name,
    _has_shallow_unproven_call_or_await,
    _is_direct_suppressor_expr,
    _is_invalid_with_context_expr,
    _is_non_raising_store_target,
    _is_proven_non_suppressing_call,
    _is_proven_non_suppressing_target_expr,
    _is_suppress_alias_target_expr,
    _iter_shallow_scope_body_nodes,
    _looks_like_suppressing_ctx_manager,
    _resolve_tracked_factory_qualname,
    _unpack_fallback_rhs_is_suspicious,
    _warn_heuristic_non_suppressing_ctx_manager,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _FuncDefCtxKind(Enum):
    """Classification of context-manager factory behavior returned or yielded by a function definition."""

    NONE = "none"
    SUSPICIOUS = "suspicious"
    SUPPRESSOR = "suppressor"


# INVARIANT (CONVERGENCE BOUNDARY — FINITE FACTORY PROOF):
# Context-manager factory classification inspects only the direct function body
# (single-hop return/yield expressions and local provenance). Do NOT expand
# into whole-program call-graph chasing or symbolic decorator execution;
# unclassifiable factories must fail closed in ``pedantic`` mode or be
# explicitly listed in ``non_suppressing_ctx_managers``.
def _classify_func_def_ctx_manager_kind(
    fn_stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _FuncDefCtxKind:
    """Classifies whether a function definition returns/yields a suppressor or suspicious manager."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _FuncDefCtxKind.SUPPRESSOR
    local_prov = prov
    for p_name in _extract_arg_names(fn_stmt.args):
        local_prov = local_prov.clear_name(p_name)
    for tp_name in _extract_node_type_param_names(fn_stmt):
        local_prov = local_prov.clear_name(tp_name)
    body_prov = _update_suppress_provenance_from_stmts(
        fn_stmt.body,
        local_prov,
        in_branch=True,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    fn_prov = local_prov.merge_branch_conservative(body_prov)

    found_suspicious = False
    stack: List[Tuple[ast.AST, bool]] = [
        (stmt, False) for stmt in reversed(fn_stmt.body)
    ]
    while stack:
        curr, in_try_or_with = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            for hdr in reversed(tuple(iter_nested_scope_header_nodes(curr))):
                stack.append((hdr, in_try_or_with))
            continue
        if isinstance(curr, (ast.Yield, ast.YieldFrom)) and in_try_or_with:
            return _FuncDefCtxKind.SUPPRESSOR
        if isinstance(curr, ast.Return) and curr.value is not None:
            val_stack: List[ast.AST] = [curr.value]
            while val_stack:
                v_curr = val_stack.pop()
                if _is_suppress_alias_target_expr(v_curr, fn_prov):
                    return _FuncDefCtxKind.SUPPRESSOR
                if isinstance(v_curr, _INNER_SCOPE_TYPES):
                    val_stack.extend(iter_nested_scope_header_nodes(v_curr))
                    continue
                val_stack.extend(ast.iter_child_nodes(v_curr))
            if (
                not found_suspicious
                and not _is_proven_non_suppressing_call(
                    curr.value,
                    fn_prov,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                and not _is_proven_non_suppressing_target_expr(
                    curr.value,
                    fn_prov,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                and _looks_like_suppressing_ctx_manager(curr.value, fn_prov)
            ):
                found_suspicious = True
        next_in_try_or_with = (
            in_try_or_with
            or isinstance(curr, (ast.With, ast.AsyncWith))
            or (
                is_try_or_trystar_node(curr)
                and bool(
                    getattr(curr, "handlers", None) or getattr(curr, "finalbody", None)
                )
            )
        )
        for ch in reversed(tuple(ast.iter_child_nodes(curr))):
            stack.append((ch, next_in_try_or_with))
    return (
        _FuncDefCtxKind.SUSPICIOUS
        if found_suspicious
        else _FuncDefCtxKind.NONE
    )


def _func_def_may_return_suppressor(
    fn_stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    prov: _SuppressProvenance,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True if a function definition may return a suppressor or suppress exceptions around yield."""
    return (
        _classify_func_def_ctx_manager_kind(fn_stmt, prov, _depth=_depth)
        is _FuncDefCtxKind.SUPPRESSOR
    )


class _CrossScopeMutationAcc(NamedTuple):
    """Mutable sets accumulating global/nonlocal target names and provenance tags."""

    mutated: Set[str]
    sup_targets: Set[str]
    susp_targets: Set[str]


def _collect_scope_escaping_into(
    shallow_nodes: Sequence[ast.AST],
    anc_locals: AbstractSet[str],
    prov: _SuppressProvenance,
    acc: _CrossScopeMutationAcc,
) -> Tuple[Set[str], Set[str]]:
    """Collects global/nonlocal escaping mutations from shallow_nodes into acc, returning (globs, nonlocs)."""
    scope_globs: Set[str] = set()
    scope_nonlocs: Set[str] = set()
    for sub in shallow_nodes:
        if isinstance(sub, ast.Global):
            scope_globs.update(sub.names)
        elif isinstance(sub, ast.Nonlocal):
            scope_nonlocs.update(sub.names)
    escaping = scope_globs | (scope_nonlocs - anc_locals)
    if escaping:
        def _record_escaping_value(
            targets: AbstractSet[str], val_expr: ast.AST
        ) -> None:
            if not targets:
                return
            acc.mutated.update(targets)
            if _is_direct_suppressor_expr(val_expr, prov):
                acc.sup_targets.update(targets)
            elif _unpack_fallback_rhs_is_suspicious(val_expr, prov):
                acc.susp_targets.update(targets)

        for sub in shallow_nodes:
            if isinstance(sub, ast.Assign):
                _record_escaping_value(
                    {
                        n
                        for tgt in sub.targets
                        for n in _extract_definite_target_names(tgt)
                        if n in escaping
                    },
                    sub.value,
                )
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                _record_escaping_value(
                    _extract_definite_target_names(sub.target) & escaping,
                    sub.value,
                )
            elif (
                isinstance(sub, ast.NamedExpr)
                and isinstance(sub.target, ast.Name)
                and sub.target.id in escaping
            ):
                _record_escaping_value({sub.target.id}, sub.value)
            elif isinstance(sub, ast.Name) and isinstance(
                sub.ctx, (ast.Store, ast.Del)
            ):
                if sub.id in escaping:
                    acc.mutated.add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if sub.name in escaping:
                    acc.mutated.add(sub.name)
                    fn_kind = _classify_func_def_ctx_manager_kind(sub, prov)
                    if fn_kind is _FuncDefCtxKind.SUPPRESSOR:
                        acc.sup_targets.add(sub.name)
                    elif fn_kind is _FuncDefCtxKind.SUSPICIOUS:
                        acc.susp_targets.add(sub.name)
            elif isinstance(sub, ast.ClassDef):
                if sub.name in escaping:
                    acc.mutated.add(sub.name)
                    if _class_def_may_suppress_in_exit(sub, prov):
                        acc.sup_targets.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                imp_s: Set[str] = set()
                _record_import_names(sub, imp_s)
                acc.mutated.update(imp_s & escaping)
                if isinstance(sub, ast.ImportFrom):
                    for alias in sub.names:
                        if alias.name == "*":
                            continue
                        bound = alias.asname or alias.name
                        if bound not in escaping:
                            continue
                        if alias.name in _SUPPRESSING_CTX_MANAGER_NAMES:
                            acc.sup_targets.add(bound)
                        elif (
                            alias.name != "catch_warnings"
                            and _ident_has_suspicious_suppress_token(alias.name)
                        ):
                            acc.susp_targets.add(bound)
            elif isinstance(sub, ast.ExceptHandler) and isinstance(sub.name, str):
                if sub.name in escaping:
                    acc.mutated.add(sub.name)
            else:
                p_name = _extract_pattern_bound_name(sub)
                if p_name is not None and p_name in escaping:
                    acc.mutated.add(p_name)
    return scope_globs, scope_nonlocs


def _extract_nested_cross_scope_mutations(
    scope_stmt: ast.AST,
    prov: _SuppressProvenance,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (mutated, suppressor_targets, suspicious_targets) for global/nonlocal stores in nested scopes."""
    acc = _CrossScopeMutationAcc(set(), set(), set())
    work: List[Tuple[ast.AST, frozenset[str]]] = [(scope_stmt, frozenset())]
    while work:
        curr_scope, anc_locals = work.pop()
        if not isinstance(
            curr_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        shallow_nodes = tuple(_iter_shallow_scope_body_nodes(curr_scope))
        sc_globs, sc_nonlocs = _collect_scope_escaping_into(
            shallow_nodes, anc_locals, prov, acc
        )
        if isinstance(curr_scope, ast.ClassDef):
            next_anc = anc_locals
        else:
            own_locals = (
                (
                    _extract_arg_names(curr_scope.args)
                    | _extract_single_node_local_names(curr_scope)
                )
                - sc_globs
                - sc_nonlocs
            )
            next_anc = anc_locals | frozenset(own_locals)
        for sub in shallow_nodes:
            if isinstance(sub, _DEFERRED_DEF_SCOPE_TYPES) and not isinstance(
                sub, ast.Lambda
            ):
                work.append((sub, next_anc))
    return acc.mutated, acc.sup_targets, acc.susp_targets


class _ProvWalkResult(NamedTuple):
    """Single-pass provenance, intermediate-hazard provenance, and termination status."""

    prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    terminated: bool


def _update_prov_and_term_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Applies provenance and termination analysis across a statement sequence in one pass."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        out = _conservatively_update_prov_at_depth_limit(stmts, prov)
        return _ProvWalkResult(out, prov.merge_branch_conservative(out), False)
    cur_prov = prov
    mid_prov = prov
    for sub_stmt in stmts:
        step = _update_prov_and_term_from_stmt(
            sub_stmt,
            cur_prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        cur_prov = step.prov
        mid_prov = mid_prov.merge_branch_conservative(step.mid_prov)
        if step.terminated:
            return _ProvWalkResult(cur_prov, mid_prov, True)
    return _ProvWalkResult(cur_prov, mid_prov, False)


def _update_suppress_provenance_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Applies _update_suppress_provenance_from_stmt across a statement sequence."""
    return _update_prov_and_term_from_stmts(
        stmts,
        prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth,
    ).prov


def _update_clause_branch_provenance_with_term(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    body: Iterable[ast.AST],
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Updates provenance and termination status for an except handler or match case branch."""
    entry_prov = _advance_clause_entry_prov(
        prov,
        header_expr,
        bound_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    res = _update_prov_and_term_from_stmts(
        body,
        entry_prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    return _ProvWalkResult(
        res.prov, entry_prov.merge_branch_conservative(res.mid_prov), res.terminated
    )


def _update_assign_or_delete_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> _SuppressProvenance:
    """Updates provenance for Assign, AnnAssign, AugAssign, Delete, and TypeAlias statements."""
    if isinstance(stmt, ast.Assign):
        rhs_prov = _advance_prov_through_exprs(
            (stmt.value,),
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        prov = rhs_prov
        for tgt in stmt.targets:
            prov = _advance_prov_through_exprs(
                (tgt,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            prov = _bind_assign_target_provenance(
                rhs_prov,
                tgt,
                stmt.value,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                target_prov=prov,
            )
        return prov

    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        rhs_prov = (
            _advance_prov_through_exprs(
                (stmt.value,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if isinstance(stmt.value, ast.AST)
            else prov
        )
        ann_expr = getattr(stmt, "annotation", None)
        prov = rhs_prov
        if isinstance(ann_expr, ast.AST):
            prov = _advance_prov_through_exprs(
                (ann_expr,),
                prov,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        if isinstance(stmt.target, ast.AST):
            prov = _advance_prov_through_exprs(
                (stmt.target,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        if stmt.value is not None:
            is_plain_ann = isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name
            )
            for t_name in _extract_definite_target_names(stmt.target):
                prov = (
                    _bind_expr_target_provenance(
                        rhs_prov,
                        t_name,
                        stmt.value,
                        allow_positive=allow_positive and isinstance(stmt, ast.AnnAssign),
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                        target_prov=prov,
                    )
                    if is_plain_ann
                    else prov.clear_name(t_name)
                )
        return prov

    if isinstance(stmt, ast.Delete):
        prov = _advance_prov_through_exprs(
            stmt.targets,
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        for d_name in _extract_shallow_deleted_names(stmt):
            prov = prov.clear_name(d_name)
        return prov

    ta_val = getattr(stmt, "value", None)
    if isinstance(ta_val, ast.AST):
        prov = _update_suppress_provenance_from_expr_walrus(
            (ta_val,),
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    ta_name = _extract_type_alias_or_pattern_name(stmt)
    return prov.clear_name(ta_name) if ta_name is not None else prov


class _ProvenanceWalkContext(NamedTuple):
    """Policy and recursion depth context passed across compound provenance updaters."""

    in_branch: bool = False
    strict_with_suppression: bool = False
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset()
    depth: int = 0


def _walk_child_stmts_prov(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
    *,
    in_branch: Optional[bool] = None,
) -> _ProvWalkResult:
    """Steps into child statement sequences at ctx.depth + 1 using ctx policy flags."""
    return _update_prov_and_term_from_stmts(
        stmts,
        prov,
        in_branch=ctx.in_branch if in_branch is None else in_branch,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        _depth=ctx.depth + 1,
    )


class _TryBodySplitProvenance(NamedTuple):
    """Remaining suffix, provenance states, and termination status across a split try body."""

    risky_suffix: List[ast.stmt]
    prefix_prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    try_prov: _SuppressProvenance
    terminated: bool


def _compute_try_body_split_provenance(
    t_body: Sequence[ast.stmt],
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
) -> _TryBodySplitProvenance:
    """Split a try body at its first potentially raising statement and advance provenance in one pass."""
    safe_prefix, risky_suffix = _split_suppress_with_body(
        t_body,
        safe_names=safe_names,
        strict_names=strict_names,
        suppress_aliases=prov,
        _depth=ctx.depth + 1,
    )
    pre_res = _walk_child_stmts_prov(safe_prefix, prov, ctx)
    pre_has_exit = bool(safe_prefix and any(_scan_stmts_shallow_exits(safe_prefix)))
    if pre_res.terminated or not risky_suffix:
        split_mid = pre_res.mid_prov if pre_has_exit else pre_res.prov
        return _TryBodySplitProvenance(
            [], pre_res.prov, split_mid, pre_res.prov, pre_res.terminated
        )
    suf_res = _walk_child_stmts_prov(risky_suffix, pre_res.prov, ctx)
    split_mid = (
        pre_res.mid_prov.merge_branch_conservative(suf_res.mid_prov)
        if pre_has_exit
        else suf_res.mid_prov
    )
    return _TryBodySplitProvenance(
        risky_suffix,
        pre_res.prov,
        split_mid,
        suf_res.prov,
        suf_res.terminated,
    )


def _apply_immediate_class_body_provenance(
    cls_stmt: ast.ClassDef,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Applies immediate global/nonlocal rebinds and cross-scope call invalidations for ClassDef bodies."""
    imm_acc = _CrossScopeMutationAcc(set(), set(), set())
    has_unproven_call = False
    cls_work: List[ast.ClassDef] = [cls_stmt]
    while cls_work:
        curr_cls = cls_work.pop()
        shallow_nodes = tuple(_iter_shallow_scope_body_nodes(curr_cls))
        _collect_scope_escaping_into(shallow_nodes, frozenset(), prov, imm_acc)
        if prov.cross_scope_mutated and not has_unproven_call:
            has_unproven_call = _has_shallow_unproven_call_or_await(
                shallow_nodes,
                prov,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
        for sub in shallow_nodes:
            if isinstance(sub, ast.ClassDef):
                cls_work.append(sub)
    if imm_acc.mutated:
        prov = prov.invalidate_cross_scope_targets(
            imm_acc.mutated, imm_acc.sup_targets, imm_acc.susp_targets
        )
    if has_unproven_call:
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _apply_class_def_external_effects(
    cls_stmt: ast.ClassDef,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Applies outer header, cross-scope, and global/nonlocal effects of ClassDef without leaking class locals."""
    allow_pos = not ctx.in_branch
    hdr_nodes = tuple(iter_nested_scope_header_nodes(cls_stmt))
    prov = _advance_prov_through_exprs(
        hdr_nodes,
        prov,
        allow_positive=allow_pos,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    cs_mut, cs_sup, cs_susp = _extract_nested_cross_scope_mutations(cls_stmt, prov)
    prov = prov.with_cross_scope_mutations(cs_mut, cs_sup, cs_susp)
    prov = _apply_immediate_class_body_provenance(cls_stmt, prov, ctx)
    if ctx.in_branch and cs_mut:
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _update_scope_def_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Updates provenance from a FunctionDef, AsyncFunctionDef, or ClassDef statement."""
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return prov
    allow_pos = not ctx.in_branch
    if isinstance(stmt, ast.ClassDef):
        prov = _apply_class_def_external_effects(stmt, prov, ctx)
    else:
        hdr_nodes = tuple(iter_nested_scope_header_nodes(stmt))
        prov = _advance_prov_through_exprs(
            hdr_nodes,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        cs_mut, cs_sup, cs_susp = _extract_nested_cross_scope_mutations(stmt, prov)
        prov = prov.with_cross_scope_mutations(cs_mut, cs_sup, cs_susp)
        if ctx.in_branch and cs_mut:
            prov = prov.apply_cross_scope_invalidation()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn_kind = _classify_func_def_ctx_manager_kind(
            stmt,
            prov,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            _depth=ctx.depth + 1,
        )
        if fn_kind is _FuncDefCtxKind.SUPPRESSOR:
            return prov.with_suppressor(stmt.name)
        if fn_kind is _FuncDefCtxKind.SUSPICIOUS:
            return prov.with_suspicious_alias(stmt.name)
    elif _class_def_may_suppress_in_exit(stmt, prov):
        return prov.with_suppressor(stmt.name)
    elif (
        _class_def_is_proven_non_suppressing_in_exit(stmt, prov)
        and stmt.name not in _SUPPRESSING_CTX_MANAGER_NAMES
    ):
        return prov.with_non_suppressor(
            stmt.name,
            allow_positive=allow_pos,
            imported_qualname=stmt.name if allow_pos else None,
        )
    if allow_pos and stmt.name not in _SHADOWABLE_BUILTIN_CTX_NAMES:
        return prov.with_imported_symbol(stmt.name, stmt.name, allow_positive=True)
    return prov.clear_name(stmt.name)


def _update_scope_def_stmt_prov_with_term(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Wraps _update_scope_def_stmt_provenance into a non-terminating _ProvWalkResult."""
    out = _update_scope_def_stmt_provenance(stmt, prov, ctx)
    return _ProvWalkResult(out, out, False)


def _run_loop_body_prov_fixpoint(
    body_stmts: Sequence[ast.AST],
    entry_prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
    *,
    recheck_test: Optional[ast.AST] = None,
) -> Tuple[_SuppressProvenance, _ProvWalkResult]:
    """Iterates loop-body provenance to a bounded fixed point so loop-carried rebinds propagate across back-edges."""
    loop_entry = entry_prov
    max_iters = 8 if ctx.depth < 3 else (2 if ctx.depth < 6 else 1)
    b_res = _ProvWalkResult(loop_entry, loop_entry, False)
    for _ in range(max_iters):
        b_res = _walk_child_stmts_prov(body_stmts, loop_entry, ctx, in_branch=True)
        back_edge = (
            loop_entry.merge_branch_conservative(b_res.mid_prov)
            .merge_branch_conservative(b_res.prov)
        )
        if isinstance(recheck_test, ast.AST):
            retested = _advance_prov_through_opt_expr(
                recheck_test,
                back_edge,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
            back_edge = back_edge.merge_branch_conservative(retested)
        if back_edge == loop_entry:
            break
        loop_entry = back_edge
    return loop_entry, b_res


def _update_loop_or_if_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination status across If, While, For, and AsyncFor statements."""
    if not isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        return _ProvWalkResult(prov, prov, False)
    allow_pos = not ctx.in_branch
    if isinstance(stmt, ast.If):
        prov = _advance_prov_through_opt_expr(
            stmt.test,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        b_res = _walk_child_stmts_prov(stmt.body, prov, ctx, in_branch=ctx.in_branch)
        o_res = (
            _walk_child_stmts_prov(stmt.orelse, prov, ctx, in_branch=ctx.in_branch)
            if stmt.orelse
            else _ProvWalkResult(prov, prov, False)
        )
        if b_res.terminated and not o_res.terminated:
            merged = o_res.prov
        elif o_res.terminated and not b_res.terminated:
            merged = b_res.prov
        else:
            merged = b_res.prov.merge_branch_conservative(o_res.prov)
        out_mid = (
            prov.merge_branch_conservative(b_res.mid_prov)
            .merge_branch_conservative(o_res.mid_prov)
        )
        return _ProvWalkResult(merged, out_mid, b_res.terminated and o_res.terminated)

    if isinstance(stmt, ast.While):
        prov = _advance_prov_through_opt_expr(
            stmt.test,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        loop_fixed, b_res = _run_loop_body_prov_fixpoint(
            stmt.body, prov, ctx, recheck_test=stmt.test
        )
        o_res = (
            _walk_child_stmts_prov(stmt.orelse, loop_fixed, ctx, in_branch=True)
            if stmt.orelse
            else _ProvWalkResult(loop_fixed, loop_fixed, False)
        )
        out_prov = loop_fixed.merge_branch_conservative(o_res.prov)
        out_mid = (
            loop_fixed.merge_branch_conservative(b_res.mid_prov)
            .merge_branch_conservative(o_res.mid_prov)
        )
        out_term = _loop_orelse_terminates(stmt.body, stmt.orelse, o_res.terminated)
        return _ProvWalkResult(out_prov, out_mid, out_term)

    prov, iter_prov = _advance_prov_through_for_header(
        stmt,
        prov,
        allow_positive=allow_pos,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    loop_fixed, b_res = _run_loop_body_prov_fixpoint(stmt.body, iter_prov, ctx)
    o_base = prov.merge_branch_conservative(loop_fixed)
    o_res = (
        _walk_child_stmts_prov(stmt.orelse, o_base, ctx, in_branch=True)
        if stmt.orelse
        else _ProvWalkResult(loop_fixed, b_res.mid_prov, False)
    )
    out_prov = o_base.merge_branch_conservative(o_res.prov)
    out_mid = (
        o_base.merge_branch_conservative(b_res.mid_prov)
        .merge_branch_conservative(o_res.mid_prov)
    )
    out_term = _loop_orelse_terminates(stmt.body, stmt.orelse, o_res.terminated)
    return _ProvWalkResult(out_prov, out_mid, out_term)


def _update_with_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across a With/AsyncWith header and body prefix/suffix."""
    if not isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _ProvWalkResult(prov, prov, False)
    hdr = _summarize_with_header_steps(
        list(
            _iter_with_item_provenance_steps(
                stmt,
                prov,
                allow_positive=not ctx.in_branch,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
                strict_with_suppression=ctx.strict_with_suppression,
            )
        ),
        prov,
    )
    item_prov = hdr.post_items_prov
    safe_prefix, risky_suffix = _split_with_body_by_mode(
        stmt,
        prov,
        body_suppress_aliases=item_prov,
        item_steps=hdr.steps,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        strict_with_suppression=ctx.strict_with_suppression,
        emit_heuristic_warnings=False,
        _depth=ctx.depth + 1,
    )

    def _eval_prefix(
        prefix: Sequence[ast.stmt],
    ) -> Tuple[Tuple[_SuppressProvenance, _SuppressProvenance], bool]:
        pre_res = _walk_child_stmts_prov(prefix, item_prov, ctx)
        pre_mid = item_prov.merge_branch_conservative(pre_res.mid_prov)
        return (pre_res.prov, pre_mid), pre_res.terminated

    def _eval_suffix(
        suffix: Sequence[ast.stmt],
        pre_state: Tuple[_SuppressProvenance, _SuppressProvenance],
    ) -> Tuple[_SuppressProvenance, _SuppressProvenance]:
        pre_prov, pre_mid = pre_state
        # Statements in risky_suffix execute after a potentially raising expression inside a
        # potentially exception-suppressing with-block; if an exception is raised and suppressed,
        # the remainder of risky_suffix is skipped, so bindings in risky_suffix are conditional
        # (in_branch=True) relative to code following the with-block.
        suf_res = _walk_child_stmts_prov(suffix, pre_prov, ctx, in_branch=True)
        return (
            pre_prov.merge_branch_conservative(suf_res.prov),
            pre_mid.merge_branch_conservative(suf_res.mid_prov),
        )

    def _finalize(
        final_state: Tuple[_SuppressProvenance, _SuppressProvenance],
        terminated: bool,
    ) -> _ProvWalkResult:
        out_prov, out_mid = final_state
        return _ProvWalkResult(
            _apply_with_exit_invalidation(out_prov, hdr.has_unproven_exit),
            _apply_with_exit_invalidation(out_mid, hdr.has_unproven_exit),
            terminated,
        )

    return _run_with_cfg(
        safe_prefix,
        risky_suffix,
        eval_prefix=_eval_prefix,
        eval_suffix=_eval_suffix,
        finalize_exit=_finalize,
    )


def _update_try_finally_provenance(
    f_body: Sequence[ast.AST],
    merged_try: _SuppressProvenance,
    handler_base: _SuppressProvenance,
    *,
    has_abrupt_entry: bool,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Evaluates finally-body provenance and termination including unhandled-exception entry."""
    f_entry_prov = (
        merged_try.merge_branch_conservative(handler_base)
        if has_abrupt_entry
        else merged_try
    )
    if not has_abrupt_entry or _finally_may_suppress_active_exception(f_body):
        return _walk_child_stmts_prov(f_body, f_entry_prov, ctx)
    norm_res = _walk_child_stmts_prov(f_body, merged_try, ctx)
    if f_entry_prov == merged_try:
        return norm_res
    abrupt_res = _walk_child_stmts_prov(f_body, f_entry_prov, ctx)
    combined_mid = (
        norm_res.mid_prov.merge_branch_conservative(abrupt_res.mid_prov)
        .merge_branch_conservative(abrupt_res.prov)
    )
    return _ProvWalkResult(norm_res.prov, combined_mid, norm_res.terminated)


class _TryHandlersProvenance(NamedTuple):
    """Merged fallthrough provenance, handler scan state, mid-hazards, and termination flag."""

    post_try_prov: _SuppressProvenance
    handler_scan_prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    all_handlers_terminated: bool


def _update_try_handlers_provenance(
    stmt: ast.AST,
    handler_base: _SuppressProvenance,
    succ_prov: _SuppressProvenance,
    succ_term: bool,
    ctx: _ProvenanceWalkContext,
) -> _TryHandlersProvenance:
    """Evaluates except handlers in one pass, omitting terminating paths from fallthrough provenance."""
    handlers = getattr(stmt, "handlers", ())
    is_trystar = is_trystar_node(stmt)
    final_prov = handler_base.merge_branch_conservative(succ_prov)
    mid_prov = final_prov
    scan_prov = handler_base

    def _eval_handler(h_node: ast.AST) -> Tuple[_SuppressProvenance, bool]:
        nonlocal final_prov, mid_prov
        h_name = getattr(h_node, "name", None)
        h_type = getattr(h_node, "type", None)
        h_entry = _advance_clause_entry_prov(
            scan_prov,
            h_type,
            (h_name,) if isinstance(h_name, str) else (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        h_body = getattr(h_node, "body", ())
        h_res = _walk_child_stmts_prov(h_body, h_entry, ctx)
        h_exit_prov = (
            h_res.prov.clear_name(h_name) if isinstance(h_name, str) else h_res.prov
        )
        final_prov = final_prov.merge_branch_conservative(h_exit_prov)
        mid_prov = mid_prov.merge_branch_conservative(
            h_res.mid_prov
        ).merge_branch_conservative(h_exit_prov)
        return h_exit_prov, h_res.terminated

    def _advance_prior(
        h_node: ast.AST, h_exit_prov: _SuppressProvenance, chain_trystar: bool
    ) -> None:
        nonlocal scan_prov
        h_type = getattr(h_node, "type", None)
        scan_prov = _merge_prior_clause_guard_and_binds(
            scan_prov,
            h_type,
            (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        if chain_trystar:
            scan_prov = scan_prov.merge_branch_conservative(h_exit_prov)

    cfg_res = _run_try_handlers_cfg(
        handlers,
        is_trystar=is_trystar,
        eval_handler=_eval_handler,
        advance_prior_handler=_advance_prior,
    )
    surviving: List[_SuppressProvenance] = [succ_prov] if not succ_term else []
    surviving.extend(cfg_res.surviving_outcomes)
    post_prov = _merge_surviving_branches(
        surviving,
        (final_prov,),
        lambda a, b: a.merge_branch_conservative(b),
    )
    return _TryHandlersProvenance(
        post_prov, scan_prov, mid_prov, cfg_res.all_handlers_terminated
    )


def _update_try_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across Try/TryStar body, handlers, orelse, and finalbody in one pass."""
    t_body = getattr(stmt, "body", ())
    split = _compute_try_body_split_provenance(t_body, prov, ctx)
    try_prov = split.try_prov
    handler_base = (
        split.prefix_prov.merge_branch_conservative(
            split.mid_prov
        ).merge_branch_conservative(try_prov)
        if split.risky_suffix
        else split.prefix_prov
    )
    t_term = split.terminated
    o_stmts = getattr(stmt, "orelse", ())
    if o_stmts and not t_term:
        o_res = _walk_child_stmts_prov(o_stmts, try_prov, ctx)
        succ_prov, succ_mid, succ_term = o_res.prov, o_res.mid_prov, o_res.terminated
    else:
        succ_prov, succ_mid, succ_term = try_prov, split.mid_prov, t_term
    h_res = _update_try_handlers_provenance(
        stmt, handler_base, succ_prov, succ_term, ctx
    )
    pre_finally_term = succ_term and h_res.all_handlers_terminated
    try_mid = (
        prov.merge_branch_conservative(handler_base)
        .merge_branch_conservative(split.mid_prov)
        .merge_branch_conservative(succ_mid)
        .merge_branch_conservative(h_res.mid_prov)
    )
    f_body = getattr(stmt, "finalbody", ())
    if not f_body:
        return _ProvWalkResult(h_res.post_try_prov, try_mid, pre_finally_term)
    o_has_risky = bool(
        o_stmts
        and not t_term
        and _split_suppress_with_body(
            o_stmts, suppress_aliases=try_prov, _depth=ctx.depth + 1
        )[1]
    )
    has_abrupt_entry = _try_has_abrupt_entry(
        t_body,
        o_stmts,
        body_has_risky=bool(split.risky_suffix),
        orelse_has_risky=o_has_risky,
        body_terminated=t_term,
    )
    finally_abrupt_base = (
        h_res.handler_scan_prov.merge_branch_conservative(h_res.mid_prov)
        .merge_branch_conservative(split.mid_prov)
        .merge_branch_conservative(succ_mid)
    )
    f_res = _update_try_finally_provenance(
        f_body,
        h_res.post_try_prov,
        finally_abrupt_base,
        has_abrupt_entry=has_abrupt_entry,
        ctx=ctx,
    )
    return _ProvWalkResult(
        f_res.prov,
        try_mid.merge_branch_conservative(f_res.mid_prov),
        f_res.terminated or pre_finally_term,
    )


def _update_match_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across a Python 3.10+ Match statement."""
    prov = _advance_prov_through_opt_expr(
        getattr(stmt, "subject", None),
        prov,
        allow_positive=not ctx.in_branch,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    clauses = _lower_compound_clauses(stmt)
    if not clauses:
        return _ProvWalkResult(prov, prov, False)
    mid_prov = prov

    def _eval_case(
        clause: _CompoundClause,
        scan_prov: _SuppressProvenance,
    ) -> Tuple[_SuppressProvenance, bool]:
        nonlocal mid_prov
        c_res = _update_clause_branch_provenance_with_term(
            scan_prov,
            clause.guard_or_header,
            clause.bound_names,
            clause.body,
            strict_with_suppression=ctx.strict_with_suppression,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            _depth=ctx.depth + 1,
        )
        mid_prov = mid_prov.merge_branch_conservative(c_res.mid_prov)
        return c_res.prov, c_res.terminated

    cfg_res = _run_match_cfg(
        clauses,
        initial_prior=prov,
        eval_case=_eval_case,
        advance_prior=lambda p, cl: _merge_prior_compound_clause_prov(
            p,
            cl,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        ),
    )
    surviving_provs: List[_SuppressProvenance] = list(cfg_res.surviving_outcomes)
    if not cfg_res.has_irrefutable:
        surviving_provs.append(cfg_res.final_prior)
    merged_match = _merge_surviving_branches(
        surviving_provs,
        cfg_res.case_outcomes,
        lambda a, b: a.merge_branch_conservative(b),
    )
    return _ProvWalkResult(
        merged_match,
        mid_prov,
        cfg_res.has_irrefutable and cfg_res.all_cases_terminated,
    )


_COMPOUND_STMT_PROVENANCE_HANDLERS: Tuple[
    Tuple[
        Callable[[ast.AST], bool],
        Callable[
            [ast.AST, _SuppressProvenance, _ProvenanceWalkContext],
            _ProvWalkResult,
        ],
    ],
    ...,
] = (
    (
        lambda s: isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)),
        _update_scope_def_stmt_prov_with_term,
    ),
    (
        lambda s: isinstance(s, (ast.If, ast.While, ast.For, ast.AsyncFor)),
        _update_loop_or_if_stmt_provenance,
    ),
    (lambda s: isinstance(s, (ast.With, ast.AsyncWith)), _update_with_stmt_provenance),
    (is_try_or_trystar_node, _update_try_stmt_provenance),
    (is_match_node, _update_match_stmt_provenance),
)


def _update_prov_and_term_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Returns updated provenance, intermediate-hazard provenance, and termination for stmt."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        out = _conservatively_update_prov_at_depth_limit((stmt,), prov)
        return _ProvWalkResult(out, prov.merge_branch_conservative(out), False)
    allow_pos = not in_branch
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        out = _update_import_stmt_provenance(stmt, prov, allow_positive=allow_pos)
        return _ProvWalkResult(out, out, False)
    if isinstance(
        stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)
    ) or is_type_alias_node(stmt):
        out = _update_assign_or_delete_stmt_provenance(
            stmt,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        return _ProvWalkResult(out, out, False)
    walk_ctx = _ProvenanceWalkContext(
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        depth=_depth,
    )
    for matcher, handler in _COMPOUND_STMT_PROVENANCE_HANDLERS:
        if matcher(stmt):
            return handler(stmt, prov, walk_ctx)
    if not isinstance(stmt, _INNER_SCOPE_TYPES):
        child_exprs = tuple(
            ch for ch in ast.iter_child_nodes(stmt) if not isinstance(ch, ast.stmt)
        )
        out = _advance_prov_through_exprs(
            child_exprs,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        return _ProvWalkResult(out, out, isinstance(stmt, (ast.Return, ast.Raise)))
    return _ProvWalkResult(prov, prov, False)


def _update_suppress_provenance_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Returns updated _SuppressProvenance from a statement, recursing into compound blocks."""
    return _update_prov_and_term_from_stmt(
        stmt,
        prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth,
    ).prov


def _classify_with_suppression(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> str:
    """Classifies a with/async with statement as 'suppressing', 'non_suppressing', or 'unknown'."""
    if not stmt.items:
        return "non_suppressing"
    prov = _coerce_suppress_provenance(suppress_aliases)
    all_non_suppressing = True
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    ):
        if step.item_mode == "suppressing":
            return "suppressing"
        if step.item_mode != "non_suppressing":
            all_non_suppressing = False
    return "non_suppressing" if all_non_suppressing else "unknown"


def _iter_with_item_provenance_steps(
    stmt: Union[ast.With, ast.AsyncWith],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool = True,
    encloses_span: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Iterator[_WithItemProvenanceStep]:
    """Yields per-item suppression provenance progression across a With/AsyncWith header."""
    item_prov = prov
    after_sup = False
    for item in stmt.items:
        after_prior_sup = after_sup and not encloses_span
        w_prov = _update_suppress_provenance_from_expr_walrus(
            (item.context_expr,),
            item_prov,
            allow_positive=allow_positive and not after_prior_sup,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        post_expr_prov = (
            item_prov.merge_branch_conservative(w_prov) if after_prior_sup else w_prov
        )
        pre_call_subexprs: Tuple[ast.AST, ...] = (
            (
                item.context_expr.func,
                *item.context_expr.args,
                *(kw.value for kw in item.context_expr.keywords),
            )
            if isinstance(item.context_expr, ast.Call)
            else ()
        )
        if pre_call_subexprs and _exprs_may_invoke_cross_scope_mutator(
            pre_call_subexprs,
            post_expr_prov,
            extra_non_suppressing_ctx_managers,
        ):
            post_expr_prov = post_expr_prov.apply_cross_scope_invalidation()
        is_invalid_ctx = _is_invalid_with_context_expr(
            item.context_expr,
            post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        item_mode = _classify_with_item_suppression(
            item.context_expr,
            post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        item_is_proven_non_sup = not is_invalid_ctx and item_mode == "non_suppressing"
        if item_mode == "non_suppressing" and extra_non_suppressing_ctx_managers:
            matched_wc = _find_ctx_manager_wildcard_match(
                item.context_expr,
                post_expr_prov,
                extra_non_suppressing_ctx_managers,
            )
            if matched_wc is not None:
                record_wildcard_ctx_manager_match(
                    matched_wc,
                    _format_ctx_manager_display_name(
                        item.context_expr, post_expr_prov
                    ),
                )
        item_looks_sup = (
            item_mode == "unknown"
            and _looks_like_suppressing_ctx_manager(item.context_expr, post_expr_prov)
        )
        item_sup = (
            is_invalid_ctx
            or item_mode == "suppressing"
            or (
                item_mode != "non_suppressing"
                and (strict_with_suppression or item_looks_sup)
            )
        )
        opt_cond = (
            after_prior_sup
            or is_invalid_ctx
            or (
                item_sup
                and item.optional_vars is not None
                and not _is_non_raising_store_target(item.optional_vars)
            )
        )
        post_item_prov = (
            _clear_store_target_provenance(
                item.optional_vars, post_expr_prov, is_conditional=opt_cond
            )
            if item.optional_vars is not None
            else post_expr_prov
        )
        # Entering any unproven context manager invokes __enter__/__aenter__ even without
        # an explicit ast.Call (e.g. `with cm:`), which may execute a cross-scope mutator.
        if (
            (post_item_prov.cross_scope_mutated and not item_is_proven_non_sup)
            or _exprs_may_invoke_cross_scope_mutator(
                (item.context_expr,),
                post_expr_prov,
                extra_non_suppressing_ctx_managers,
            )
        ):
            post_item_prov = post_item_prov.apply_cross_scope_invalidation()
        yield _WithItemProvenanceStep(
            item=item,
            after_prior_sup=after_prior_sup,
            post_expr_prov=post_expr_prov,
            item_may_suppress=item_sup,
            opt_vars_conditional=opt_cond,
            post_item_prov=post_item_prov,
            item_is_proven_non_suppressing=item_is_proven_non_sup,
            item_mode=item_mode,
            is_invalid_ctx=is_invalid_ctx,
            item_looks_suppressing=item_looks_sup,
        )
        if item_sup:
            after_sup = True
        item_prov = post_item_prov


def _is_non_raising_suppress_or_null_with(
    stmt: ast.AST,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if stmt is a sync With (suppress/nullcontext) or AsyncWith (nullcontext) with proven non-raising entry/exit."""
    if not isinstance(stmt, (ast.With, ast.AsyncWith)) or not stmt.items:
        return False
    is_async = isinstance(stmt, ast.AsyncWith)
    cur_prov = prov
    for item in stmt.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            return False
        cur_prov = _update_suppress_provenance_from_expr_walrus((call,), cur_prov)
        qual = _resolve_tracked_factory_qualname(call.func, cur_prov)
        allowed = (
            ("contextlib.nullcontext",)
            if is_async
            else ("contextlib.suppress", "contextlib.nullcontext")
        )
        if (
            qual is None
            or qual not in allowed
            or not _is_valid_suppress_or_null_call_signature(qual, call)
            or cur_prov.has_exit_patch
            or cur_prov.is_factory_rebound(
                cur_prov.module_aliases_for("contextlib"),
                qual.rsplit(".", 1)[1],
            )
        ):
            return False
        if item.optional_vars is not None:
            cur_prov = _clear_store_target_provenance(item.optional_vars, cur_prov)
    return True


# INVARIANT (CONVERGENCE BOUNDARY — FINITE NON-RAISING STATEMENT PROOF):
# Statement non-raising proofs are intentionally bounded to a small, closed set
# of provably non-raising forms (literal/name/container operations and direct
# ``contextlib.suppress`` / ``contextlib.nullcontext`` wrappers). Do NOT turn
# this into a general exception-type abstract interpreter; any statement outside
# the closed proof set is conservatively assumed to be able to raise.
def _stmt_cannot_raise_in_suppress(
    stmt: ast.AST,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    _depth: int = 0,
) -> bool:
    """Returns True if stmt executes and completes without raising a catchable exception."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(stmt, (ast.Pass, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, ast.Delete):
        seen_del: Set[str] = set()
        for t in stmt.targets:
            if (
                not isinstance(t, ast.Name)
                or t.id not in safe_names
                or t.id in seen_del
            ):
                return False
            seen_del.add(t.id)
        return bool(seen_del)
    if isinstance(stmt, (ast.Return, ast.Expr)):
        return _is_non_raising_expr(
            stmt.value,
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    if isinstance(stmt, ast.Assign):
        return (
            stmt.value is not None
            and _is_non_raising_expr(
                stmt.value,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            and all(_is_non_raising_store_target(t, stmt.value) for t in stmt.targets)
        )
    if isinstance(stmt, ast.AnnAssign):
        if not _is_non_raising_expr(
            stmt.annotation,
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            _depth=_depth + 1,
        ):
            return False
        return stmt.value is None or (
            _is_non_raising_expr(
                stmt.value,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            and _is_non_raising_store_target(stmt.target, stmt.value)
        )
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.decorator_list:
            return False
        return (
            _check_and_advance_non_raising_seq(
                tuple(iter_nested_scope_header_nodes(stmt)),
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            is not None
        )
    if isinstance(stmt, ast.If):
        test_sets = _check_and_advance_non_raising_seq(
            (stmt.test,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
            _depth=_depth + 1,
        )
        if test_sets is None:
            return False
        test_safe, test_bool = test_sets
        test_prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.test,), _coerce_suppress_provenance(suppress_aliases)
        )
        return (
            not _split_suppress_with_body(
                stmt.body,
                safe_names=test_safe,
                safe_bool_names=test_bool,
                strict_names=strict_names,
                suppress_aliases=test_prov,
                _depth=_depth + 1,
            )[1]
            and not _split_suppress_with_body(
                stmt.orelse,
                safe_names=test_safe,
                safe_bool_names=test_bool,
                strict_names=strict_names,
                suppress_aliases=test_prov,
                _depth=_depth + 1,
            )[1]
        )
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        prov = _coerce_suppress_provenance(suppress_aliases)
        if not _is_non_raising_suppress_or_null_with(stmt, prov):
            return False
        inner_safe = set(safe_names)
        inner_bool = set(safe_bool_names)
        inner_prov = prov
        for item in stmt.items:
            if not (
                (item.optional_vars is None or isinstance(item.optional_vars, ast.Name))
                and isinstance(item.context_expr, ast.Call)
            ):
                return False
            call_args = (*item.context_expr.args, *item.context_expr.keywords)
            if any(
                isinstance(a, ast.Starred) or getattr(a, "arg", "k") is None
                for a in call_args
            ):
                return False
            arg_exprs = tuple(
                a.value if isinstance(a, ast.keyword) else a for a in call_args
            )
            unshadowed_exc_names = (
                frozenset()
                if inner_prov.has_star_import
                else (_SAFE_EXCEPTION_TYPE_NAMES - inner_prov.shadowed_names)
            )
            if (
                _check_and_advance_non_raising_seq(
                    arg_exprs,
                    inner_safe | unshadowed_exc_names,
                    inner_bool,
                    strict_names=strict_names,
                    _depth=_depth + 1,
                )
                is None
            ):
                return False
            _apply_non_raising_expr_walrus_updates(
                item.context_expr,
                inner_safe,
                inner_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            inner_prov = _update_suppress_provenance_from_expr_walrus(
                (item.context_expr,), inner_prov
            )
            if isinstance(item.optional_vars, ast.Name):
                inner_safe.add(item.optional_vars.id)
                inner_bool.discard(item.optional_vars.id)
                inner_prov = _clear_store_target_provenance(
                    item.optional_vars, inner_prov
                )
        return not _split_suppress_with_body(
            stmt.body,
            safe_names=inner_safe,
            safe_bool_names=inner_bool,
            strict_names=strict_names,
            suppress_aliases=inner_prov,
            _depth=_depth + 1,
        )[1]
    return False


def _split_suppress_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a potentially suppressing with-block body into (non_raising_prefix, potentially_raising_suffix)."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return [], list(body)
    safe_prefix: List[ast.stmt] = []
    cur_prov = _coerce_suppress_provenance(suppress_aliases)
    cur_safe: Set[str] = (
        set(safe_names)
        | set(cur_prov.non_suppressors)
        | set(cur_prov.non_suppressor_instances)
        | {alias for alias, _ in cur_prov.stdlib_modules}
        | {alias for alias, _ in cur_prov.imported_symbols}
    )
    cur_bool: Set[str] = set(safe_bool_names)
    for idx, b_stmt in enumerate(body):
        if _stmt_cannot_raise_in_suppress(
            b_stmt,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            suppress_aliases=cur_prov,
            _depth=_depth + 1,
        ):
            safe_prefix.append(b_stmt)
            if _non_raising_stmt_terminates(b_stmt, _depth=_depth + 1):
                return safe_prefix, []
            _apply_non_raising_stmt_safe_updates(
                b_stmt,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            cur_prov = _update_suppress_provenance_from_stmt(
                b_stmt, cur_prov, _depth=_depth + 1
            )
        else:
            return safe_prefix, list(body[idx:])
    return safe_prefix, []


def _split_unknown_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits an unknown context manager's body into (guaranteed_non_raising_prefix, potentially_raising_suffix)."""
    return _split_suppress_with_body(
        body,
        safe_names=safe_names,
        strict_names=strict_names,
        suppress_aliases=suppress_aliases,
        _depth=_depth,
    )


def _advance_prov_through_with_items(
    stmt: Union[ast.With, ast.AsyncWith],
    prov: _SuppressProvenance,
    *,
    safe_names_out: Optional[Set[str]] = None,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Advances provenance across with-header items (walrus and as-targets) for body analysis."""
    item_prov = prov
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    ):
        if (
            step.item.optional_vars is not None
            and not step.opt_vars_conditional
            and safe_names_out is not None
        ):
            safe_names_out.update(
                _extract_definite_target_names(step.item.optional_vars)
            )
        item_prov = step.post_item_prov
    return item_prov


def _split_with_body_by_mode(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    body_suppress_aliases: Optional[Union[_SuppressProvenance, Iterable[str]]] = None,
    item_steps: Optional[Sequence[_WithItemProvenanceStep]] = None,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    emit_heuristic_warnings: bool = True,
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a with-block body into (guaranteed_prefix, potentially_suppressed_suffix) across 3 suppression states."""
    steps: Tuple[_WithItemProvenanceStep, ...] = (
        tuple(item_steps)
        if item_steps is not None
        else tuple(
            _iter_with_item_provenance_steps(
                stmt,
                _coerce_suppress_provenance(suppress_aliases),
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
        )
    )
    if emit_heuristic_warnings and not strict_with_suppression:
        for step in steps:
            if (
                step.item_mode == "unknown"
                and not step.is_invalid_ctx
                and not step.item_looks_suppressing
            ):
                _warn_heuristic_non_suppressing_ctx_manager(
                    step.item.context_expr, step.post_expr_prov
                )
    if _depth > _MAX_AST_RECURSION_DEPTH or any(
        step.after_prior_sup or step.opt_vars_conditional for step in steps
    ):
        return [], list(stmt.body)
    if not steps or all(step.item_mode == "non_suppressing" for step in steps):
        w_mode = "non_suppressing"
    elif any(step.item_mode == "suppressing" for step in steps):
        w_mode = "suppressing"
    else:
        w_mode = "unknown"
    body_prov = (
        _coerce_suppress_provenance(body_suppress_aliases)
        if body_suppress_aliases is not None
        else (
            steps[-1].post_item_prov
            if steps
            else _coerce_suppress_provenance(suppress_aliases)
        )
    )
    if w_mode == "suppressing":
        return _split_suppress_with_body(
            stmt.body,
            safe_names=safe_names,
            strict_names=strict_names,
            suppress_aliases=body_prov,
            _depth=_depth,
        )
    if w_mode == "unknown" and (
        strict_with_suppression
        or any(step.item_looks_suppressing for step in steps)
    ):
        return _split_unknown_with_body(
            stmt.body,
            safe_names=safe_names,
            strict_names=strict_names,
            suppress_aliases=body_prov,
            _depth=_depth,
        )
    return list(stmt.body), []
