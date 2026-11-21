"""Binding-state and definite-binding AST traversal for enclosing function cells."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    record_heuristic_non_suppressing_ctx_manager,
    record_wildcard_ctx_manager_match,
)
from pydoppelgangerhunt.fixer._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _MAX_STMT_DELTAS_PER_FN,
    _MAX_STMT_WITH_SUMMARY_CACHE_SIZE,
    _TrackedCachedDict,
    _downstream_cache_lock,
    _get_or_compute_scope_cached,
    _get_scope_ancestor_map,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
    _stmt_with_summary_cache,
    scope_fingerprint_verification_session,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _EnclosingFnCellInfo,
    _FnScopeContext,
    _SpanBounds,
    _block_contains_span,
    _can_bind_in_fn_scope,
    _extract_cached_scope_free_reads,
    _extract_carrier_value_closure_and_load_info,
    _extract_closure_nonlocal_mutations_and_deletions,
    _extract_comprehension_body_walrus_names,
    _extract_comprehension_deferred_closure_info,
    _extract_decorator_carrier_targets,
    _extract_expr_scope_free_reads,
    _extract_fn_outer_carrier_targets,
    _extract_fn_returned_closure_refs,
    _extract_store_names_in_node,
    _get_fn_scope_context,
    _is_closure_factory_scope,
    _is_node_inside_span,
    _is_pos_after_span,
    _is_pos_before_span,
    _iter_carrier_target_entries,
    _iter_class_body_methods,
    _iter_class_body_non_method_stmts,
    _iter_comprehension_body_exprs_and_targets,
    _node_contains_or_encloses_span,
    _propagate_sibling_closures,
    _strictly_encloses_span,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _extract_node_type_param_names,
    _extract_single_node_local_names,
    _node_earliest_start_pos,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _extract_direct_nonlocals,
    _extract_type_alias_or_pattern_name,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _record_import_names,
)
from pydoppelgangerhunt.fixer._df_with_provenance import (
    _SuppressProvenance,
    _advance_prov_through_for_header,
    _advance_prov_through_opt_expr,
    _coerce_suppress_provenance,
    _extract_block_definite_bindings,
    _extract_shallow_deleted_names,
    _iter_with_item_provenance_steps,
    _merge_loop_carried_suppress_provenance,
    _seed_scope_suppress_aliases,
    _update_suppress_provenance_from_stmt,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

__all__ = [
    "_CellBindingState",
    "_CellTraversalBuffers",
    "_StackFrame",
    "_StmtCellDelta",
    "_analyze_enclosing_fn_cells",
    "_analyze_enclosing_fn_cells_uncached",
    "_apply_stmt_cell_delta",
    "_build_enclosing_chain_cell_infos",
    "_find_enclosing_chain_definite_pre_bound",
    "_handle_nested_scope_node",
    "_is_frame_after_span",
    "_is_frame_before_span",
    "_make_empty_traversal_buffers",
    "_maybe_push_pre_span_exhaustive_frame",
    "_push_compound_children",
    "_push_seq",
    "_record_binding_at_pos",
    "_record_carrier_closures_and_aliases",
    "_record_deletion_at_pos",
    "_record_simple_node_bindings_and_reads",
    "_traverse_fn_cell_frames",
    "_try_apply_cached_stmt_delta",
]


@dataclass
class _CellBindingState:
    """Mutable sets tracking cell bindings during enclosing function AST traversal."""

    pre_bound: Set[str]
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]
    pre_unbound: Set[str]
    fn_globals: AbstractSet[str] = frozenset()
    suppress_aliases: Optional[_SuppressProvenance] = None
    strict_with_suppression: bool = False
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset()


class _StackFrame(NamedTuple):
    """Traversal frame for _analyze_enclosing_fn_cells."""

    node: ast.AST
    is_branched: bool
    in_loop: bool
    active_nonlocals: Optional[Set[str]]
    post_pre_bound: Optional[AbstractSet[str]] = None
    force_after_span: bool = False
    save_suppress_box: Optional[List[Optional[_SuppressProvenance]]] = None
    restore_suppress_box: Optional[List[Optional[_SuppressProvenance]]] = None
    advance_suppress_header: Optional[
        Tuple[Optional[ast.AST], AbstractSet[str], AbstractSet[str], bool]
    ] = None


def _merge_branched_suppress_update(
    pre_prov: Optional[_SuppressProvenance],
    next_prov: _SuppressProvenance,
    *,
    is_branched: bool,
) -> Optional[_SuppressProvenance]:
    """Merges a provenance update conservatively when executing along a branched path."""
    if pre_prov is None:
        return None
    return pre_prov.merge_branch_conservative(next_prov) if is_branched else next_prov


def _compute_advanced_suppress_aliases(
    bufs: _CellTraversalBuffers,
    stmt: ast.stmt,
    pre_prov: _SuppressProvenance,
    *,
    is_branched: bool,
) -> _SuppressProvenance:
    """Computes updated suppression provenance after a non-enclosing pre-span statement."""
    next_prov = _update_suppress_provenance_from_stmt(
        stmt,
        pre_prov,
        strict_with_suppression=bufs.state.strict_with_suppression,
        extra_non_suppressing_ctx_managers=bufs.state.extra_non_suppressing_ctx_managers,
    )
    return pre_prov.merge_branch_conservative(next_prov) if is_branched else next_prov


def _advance_suppress_aliases_for_stmt(
    bufs: _CellTraversalBuffers,
    stmt: ast.stmt,
    *,
    is_branched: bool,
) -> None:
    """Advances bufs.state.suppress_aliases across a non-enclosing pre-span statement."""
    if bufs.state.suppress_aliases is None:
        return
    pre_prov = _coerce_suppress_provenance(bufs.state.suppress_aliases)
    bufs.state.suppress_aliases = _compute_advanced_suppress_aliases(
        bufs,
        stmt,
        pre_prov,
        is_branched=is_branched,
    )


def _push_suppress_box_frame(
    stack: List[_StackFrame],
    curr: ast.AST,
    frame: _StackFrame,
    box: List[Optional[_SuppressProvenance]],
    *,
    save: bool = False,
    is_branched: Optional[bool] = None,
) -> None:
    """Pushes a stack checkpoint to snapshot or restore suppress_aliases across sibling branches."""
    eff_br = frame.is_branched if is_branched is None else is_branched
    stack.append(
        _StackFrame(
            curr,
            eff_br,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
            save_suppress_box=box if save else None,
            restore_suppress_box=None if save else box,
        )
    )


def _push_suppress_header_frame(
    stack: List[_StackFrame],
    curr: ast.AST,
    frame: _StackFrame,
    hdr_expr: Optional[ast.AST],
    span: _SpanBounds,
    *,
    pre_clear: AbstractSet[str] = frozenset(),
    post_clear: AbstractSet[str] = frozenset(),
    allow_positive: bool = False,
    is_branched: Optional[bool] = None,
) -> None:
    """Pushes a post-header provenance frame when hdr_expr does not overlap span."""
    if hdr_expr is not None and _node_contains_or_encloses_span(hdr_expr, span):
        return
    eff_br = frame.is_branched if is_branched is None else is_branched
    stack.append(
        _StackFrame(
            curr,
            eff_br,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
            advance_suppress_header=(
                hdr_expr,
                pre_clear,
                post_clear,
                allow_positive,
            ),
        )
    )


def _is_compound_scope_traversal_stmt(node: ast.AST) -> bool:
    """Checks whether _traverse_fn_cell_frames descends into child suites of node."""
    return (
        isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
                ast.ClassDef,
            ),
        )
        or is_try_or_trystar_node(node)
        or is_match_node(node)
    )


def _is_frame_before_span(
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
) -> bool:
    """Checks whether a frame position precedes span when not forced after span."""
    return not frame.force_after_span and _is_pos_before_span(line, col, span)


def _is_frame_after_span(
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
) -> bool:
    """Checks whether a frame position follows span or is forced after span."""
    return frame.force_after_span or _is_pos_after_span(line, col, span)


def _record_binding_at_pos(
    names: Iterable[str],
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
) -> None:
    """Records bound variable names into pre_bound, pre_mutated, post_mutated, and loop_mutated."""
    filtered = [
        n for n in names if _can_bind_in_fn_scope(n, frame.active_nonlocals, state.fn_globals)
    ]
    if not filtered:
        return
    if _is_frame_before_span(line, col, span, frame):
        state.pre_mutated.update(filtered)
        if not frame.is_branched:
            state.pre_bound.update(filtered)
            state.pre_unbound.difference_update(filtered)
    if _is_frame_after_span(line, col, span, frame):
        state.post_mutated.update(filtered)
    if frame.in_loop:
        state.loop_mutated.update(filtered)


def _record_deletion_at_pos(
    name: str,
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    *,
    unconditional_post_del: bool = False,
) -> None:
    """Records a variable deletion or ExceptHandler unbinding relative to span."""
    if not _can_bind_in_fn_scope(name, frame.active_nonlocals, state.fn_globals):
        return
    is_after = _is_frame_after_span(line, col, span, frame)
    if (
        _is_frame_before_span(line, col, span, frame)
        or frame.in_loop
        or (unconditional_post_del and is_after)
    ):
        state.pre_bound.discard(name)
        state.pre_unbound.add(name)
        state.pre_mutated.add(name)
    if is_after:
        state.post_mutated.add(name)
    if frame.in_loop:
        state.loop_mutated.add(name)


def _push_seq(
    stack: List[_StackFrame],
    nodes: Sequence[ast.AST],
    is_branched: bool,
    in_loop: bool,
    active_nonlocals: Optional[Set[str]],
    *,
    force_after_span: bool = False,
) -> None:
    """Pushes a sequence of AST nodes in reverse order onto the traversal stack."""
    for node in reversed(nodes):
        stack.append(
            _StackFrame(
                node,
                is_branched,
                in_loop,
                active_nonlocals,
                force_after_span=force_after_span,
            )
        )


def _maybe_push_pre_span_exhaustive_frame(
    curr: ast.AST,
    encloses: bool,
    has_clauses: bool,
    frame: _StackFrame,
    span: _SpanBounds,
    state: _CellBindingState,
    stack: List[_StackFrame],
    *,
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]] = None,
) -> None:
    """Pushes a post_pre_bound frame when a non-enclosing compound statement precedes span."""
    if (
        not encloses
        and not frame.is_branched
        and has_clauses
        and _is_frame_before_span(
            getattr(curr, "end_lineno", getattr(curr, "lineno", None)),
            getattr(curr, "end_col_offset", getattr(curr, "col_offset", None)),
            span,
            frame,
        )
    ):
        active_sup = pre_stmt_suppress if pre_stmt_suppress is not None else state.suppress_aliases
        exh_bound = _extract_block_definite_bindings(
            (curr,),
            frame.active_nonlocals,
            state.pre_bound,
            state.fn_globals,
            suppress_aliases=active_sup or frozenset(),
            strict_with_suppression=state.strict_with_suppression,
            extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
        )
        if exh_bound:
            stack.append(
                _StackFrame(
                    curr,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                    post_pre_bound=exh_bound,
                )
            )


def _push_with_compound_children(
    curr: Union[ast.With, ast.AsyncWith],
    encloses: bool,
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    stack: List[_StackFrame],
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]],
) -> None:
    """Pushes child frames for a With or AsyncWith statement."""
    is_br, in_lp, act_nl, f_after = (
        frame.is_branched,
        frame.in_loop,
        frame.active_nonlocals,
        frame.force_after_span,
    )
    body_span = encloses and _block_contains_span(curr.body, span)
    _maybe_push_pre_span_exhaustive_frame(
        curr,
        encloses,
        bool(curr.body or curr.items),
        frame,
        span,
        state,
        stack,
        pre_stmt_suppress=pre_stmt_suppress,
    )
    if body_span:
        for item in curr.items:
            if item.optional_vars is not None:
                w_stores = [
                    n
                    for n in _extract_store_names_in_node(item.optional_vars)
                    if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
                ]
                state.pre_mutated.update(w_stores)
                if not is_br:
                    state.pre_bound.update(w_stores)
                    state.pre_unbound.difference_update(w_stores)
    _push_seq(
        stack,
        curr.body,
        is_br if body_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    c_line = getattr(curr, "lineno", None)
    c_col = getattr(curr, "col_offset", None)
    is_before_span = _is_frame_before_span(c_line, c_col, span, frame)
    if not encloses and not is_before_span:
        _push_seq(
            stack,
            curr.items,
            is_br,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        return
    active_sup = _coerce_suppress_provenance(
        pre_stmt_suppress if pre_stmt_suppress is not None else state.suppress_aliases
    )
    steps = list(
        _iter_with_item_provenance_steps(
            curr,
            active_sup,
            allow_positive=not is_br,
            encloses_span=encloses,
            extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
            strict_with_suppression=state.strict_with_suppression,
        )
    )
    if state.suppress_aliases is not None and (
        body_span or (not encloses and is_before_span)
    ):
        if steps:
            last_step = steps[-1]
            post_items_sup = _merge_branched_suppress_update(
                state.suppress_aliases,
                last_step.post_item_prov,
                is_branched=is_br or last_step.after_prior_sup,
            )
        else:
            post_items_sup = state.suppress_aliases
        stack.append(
            _StackFrame(
                curr,
                is_br,
                in_lp,
                act_nl,
                force_after_span=f_after,
                restore_suppress_box=[post_items_sup],
            )
        )
    item_frames: List[_StackFrame] = []
    for step in steps:
        item = step.item
        item_br = is_br or step.after_prior_sup
        if (
            not encloses
            and not item_br
            and step.opt_vars_conditional
            and item.optional_vars is not None
        ):
            item_frames.append(
                _StackFrame(
                    item.context_expr,
                    item_br,
                    in_lp,
                    act_nl,
                    force_after_span=f_after,
                )
            )
            item_frames.append(
                _StackFrame(
                    item.optional_vars,
                    True,
                    in_lp,
                    act_nl,
                    force_after_span=f_after,
                )
            )
        else:
            item_frames.append(
                _StackFrame(
                    item,
                    item_br,
                    in_lp,
                    act_nl,
                    force_after_span=f_after,
                )
            )
    stack.extend(reversed(item_frames))


def _push_try_compound_children(
    curr: ast.AST,
    encloses: bool,
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    stack: List[_StackFrame],
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]],
) -> None:
    """Pushes child frames for a Try or TryStar statement."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    body_l = getattr(curr, "body", [])
    else_l = getattr(curr, "orelse", [])
    final_l = getattr(curr, "finalbody", [])
    handlers_l = getattr(curr, "handlers", [])
    else_span = encloses and _block_contains_span(else_l, span)
    body_span = encloses and (_block_contains_span(body_l, span) or else_span)
    final_span = encloses and _block_contains_span(final_l, span)
    target_h_idx: Optional[int] = None
    if encloses:
        for h_idx, h_node in enumerate(handlers_l):
            if _block_contains_span(
                getattr(h_node, "body", ()), span
            ) or _node_contains_or_encloses_span(getattr(h_node, "type", None), span):
                target_h_idx = h_idx
                break
    _maybe_push_pre_span_exhaustive_frame(
        curr,
        encloses,
        bool(body_l or final_l),
        frame,
        span,
        state,
        stack,
        pre_stmt_suppress=pre_stmt_suppress,
    )
    saved_after_body_box: Optional[List[Optional[_SuppressProvenance]]] = None
    isolate_else_from_handlers = (
        state.suppress_aliases is not None and else_span and bool(handlers_l)
    )
    isolate_handler_from_prior = (
        state.suppress_aliases is not None
        and isinstance(curr, ast.Try)
        and target_h_idx is not None
        and target_h_idx > 0
    )
    if isolate_else_from_handlers or isolate_handler_from_prior:
        saved_after_body_box = [None]
    _push_seq(
        stack,
        final_l,
        is_br if final_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    _push_seq(
        stack,
        else_l,
        is_br if else_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    if isolate_else_from_handlers and saved_after_body_box is not None:
        _push_suppress_box_frame(stack, curr, frame, saved_after_body_box)
    for h_idx in range(len(handlers_l) - 1, -1, -1):
        h_node = handlers_l[h_idx]
        h_span = encloses and _block_contains_span(getattr(h_node, "body", ()), span)
        stack.append(
            _StackFrame(
                h_node,
                is_br if h_span else True,
                in_lp,
                act_nl,
                force_after_span=f_after,
            )
        )
        if (
            isolate_handler_from_prior
            and h_idx == target_h_idx
            and saved_after_body_box is not None
        ):
            _push_suppress_box_frame(stack, curr, frame, saved_after_body_box)
    if saved_after_body_box is not None:
        _push_suppress_box_frame(
            stack, curr, frame, saved_after_body_box, save=True
        )
    _push_seq(
        stack,
        body_l,
        is_br if body_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )


def _push_match_compound_children(
    curr: ast.AST,
    encloses: bool,
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    stack: List[_StackFrame],
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]],
) -> None:
    """Pushes child frames for a Match statement."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    cases_l = getattr(curr, "cases", [])
    span_case_idx: Optional[int] = None
    span_in_case_body = False
    if encloses:
        for idx, case in enumerate(cases_l):
            c_body = getattr(case, "body", ())
            if _block_contains_span(c_body, span):
                span_case_idx = idx
                span_in_case_body = True
                break
            if any(
                _node_contains_or_encloses_span(getattr(case, attr, None), span)
                for attr in ("pattern", "guard")
            ):
                span_case_idx = idx
                break
    _maybe_push_pre_span_exhaustive_frame(
        curr,
        encloses,
        bool(cases_l),
        frame,
        span,
        state,
        stack,
        pre_stmt_suppress=pre_stmt_suppress,
    )
    for idx in range(len(cases_l) - 1, -1, -1):
        case = cases_l[idx]
        case_body = getattr(case, "body", [])
        c_br = is_br if (idx == span_case_idx and span_in_case_body) else True
        isolate_case_body = (
            state.suppress_aliases is not None
            and bool(case_body)
            and (
                (span_case_idx is not None and idx < span_case_idx)
                or (not encloses and idx < len(cases_l) - 1)
            )
        )
        saved_case_box: Optional[List[Optional[_SuppressProvenance]]] = (
            [None] if isolate_case_body else None
        )
        if saved_case_box is not None:
            _push_suppress_box_frame(
                stack, curr, frame, saved_case_box, is_branched=c_br
            )
        _push_seq(
            stack,
            case_body,
            c_br,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        if saved_case_box is not None:
            _push_suppress_box_frame(
                stack, curr, frame, saved_case_box, save=True, is_branched=c_br
            )
        if state.suppress_aliases is not None and (
            not encloses
            or (
                span_case_idx is not None
                and (idx < span_case_idx or (idx == span_case_idx and span_in_case_body))
            )
        ):
            pattern = getattr(case, "pattern", None)
            pat_names = (
                frozenset(
                    p_name
                    for p_node in ast.walk(pattern)
                    for p_name in (_extract_pattern_bound_name(p_node),)
                    if p_name is not None
                )
                if isinstance(pattern, ast.AST)
                else frozenset()
            )
            guard = getattr(case, "guard", None)
            hdr_guard = (
                guard
                if isinstance(guard, ast.AST)
                and not _node_contains_or_encloses_span(guard, span)
                else None
            )
            if pat_names or hdr_guard is not None:
                _push_suppress_header_frame(
                    stack,
                    curr,
                    frame,
                    hdr_guard,
                    span,
                    pre_clear=pat_names,
                    is_branched=c_br,
                )
        _push_seq(
            stack,
            [
                sub
                for attr in ("pattern", "guard")
                if isinstance(sub := getattr(case, attr, None), ast.AST)
            ],
            c_br,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
    if isinstance(subject := getattr(curr, "subject", None), ast.AST):
        if state.suppress_aliases is not None:
            _push_suppress_header_frame(
                stack, curr, frame, subject, span, allow_positive=True
            )
        stack.append(_StackFrame(subject, is_br, in_lp, act_nl, force_after_span=f_after))


def _push_if_compound_children(
    curr: ast.If,
    encloses: bool,
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    stack: List[_StackFrame],
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]],
) -> None:
    """Pushes child frames for an If statement while isolating mutually exclusive arms."""
    _maybe_push_pre_span_exhaustive_frame(
        curr,
        encloses,
        bool(curr.body),
        frame,
        span,
        state,
        stack,
        pre_stmt_suppress=pre_stmt_suppress,
    )
    body_span = encloses and _block_contains_span(curr.body, span)
    else_span = encloses and _block_contains_span(curr.orelse, span)
    isolate_body_from_else = (
        state.suppress_aliases is not None
        and bool(curr.body)
        and (else_span or (not encloses and bool(curr.orelse)))
    )
    saved_if_box: Optional[List[Optional[_SuppressProvenance]]] = (
        [None] if isolate_body_from_else else None
    )
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    _push_seq(
        stack,
        curr.orelse,
        is_br if else_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    if saved_if_box is not None:
        _push_suppress_box_frame(stack, curr, frame, saved_if_box)
    _push_seq(
        stack,
        curr.body,
        is_br if body_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    if saved_if_box is not None:
        _push_suppress_box_frame(stack, curr, frame, saved_if_box, save=True)
    if state.suppress_aliases is not None and (body_span or else_span or not encloses):
        _push_suppress_header_frame(
            stack, curr, frame, curr.test, span, allow_positive=True
        )
    stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl, force_after_span=f_after))


def _push_loop_compound_children(
    curr: Union[ast.For, ast.AsyncFor, ast.While],
    encloses: bool,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
) -> None:
    """Pushes child frames for For/AsyncFor/While loops with loop-carried provenance."""
    is_br, in_lp, act_nl, f_after = (
        frame.is_branched,
        frame.in_loop,
        frame.active_nonlocals,
        frame.force_after_span,
    )
    if isinstance(curr, (ast.For, ast.AsyncFor)):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        iter_span = _node_contains_or_encloses_span(curr.iter, span)
        if body_span:
            state.loop_mutated.update(unit_mutations)
            t_stores = [
                n
                for n in _extract_store_names_in_node(curr.target)
                if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
            ]
            state.pre_mutated.update(t_stores)
            if not is_br:
                state.pre_bound.update(t_stores)
                state.pre_unbound.difference_update(t_stores)
            state.loop_mutated.update(t_stores)
        after_iter = f_after or iter_span
        _push_seq(
            stack,
            curr.orelse,
            is_br if else_span else True,
            in_lp,
            act_nl,
            force_after_span=after_iter,
        )
        _push_seq(
            stack,
            curr.body,
            is_br if body_span else True,
            True if (body_span or iter_span) else in_lp,
            act_nl,
            force_after_span=after_iter,
        )
        if (
            state.suppress_aliases is not None
            and (body_span or else_span or not encloses)
            and not iter_span
        ):
            post_for_iter, post_for_tgt = _advance_prov_through_for_header(
                curr,
                state.suppress_aliases,
                allow_positive=not is_br,
                extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
            )
            eff_for_hdr = (
                post_for_tgt
                if body_span
                else post_for_iter.merge_branch_conservative(post_for_tgt)
            )
            post_for_sup = _merge_branched_suppress_update(
                state.suppress_aliases,
                eff_for_hdr,
                is_branched=is_br,
            )
            if body_span and post_for_sup is not None:
                post_for_sup = _merge_loop_carried_suppress_provenance(
                    curr.body,
                    post_for_sup,
                    strict_with_suppression=state.strict_with_suppression,
                    extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
                )
            stack.append(
                _StackFrame(
                    curr,
                    is_br,
                    in_lp,
                    act_nl,
                    force_after_span=after_iter,
                    restore_suppress_box=[post_for_sup],
                )
            )
        stack.append(
            _StackFrame(
                curr.target,
                is_br if body_span else True,
                True if (body_span or iter_span) else in_lp,
                act_nl,
                force_after_span=after_iter,
            )
        )
        stack.append(_StackFrame(curr.iter, is_br, in_lp, act_nl, force_after_span=f_after))
        return

    body_span = encloses and (
        _block_contains_span(curr.body, span) or _is_node_inside_span(curr.test, span)
    )
    else_span = encloses and _block_contains_span(curr.orelse, span)
    if body_span:
        state.loop_mutated.update(unit_mutations)
    _push_seq(
        stack,
        curr.orelse,
        is_br if else_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    _push_seq(
        stack,
        curr.body,
        is_br if body_span else True,
        True if body_span else in_lp,
        act_nl,
        force_after_span=f_after,
    )
    if state.suppress_aliases is not None and (body_span or else_span or not encloses):
        if body_span and not _node_contains_or_encloses_span(curr.test, span):
            after_test = _advance_prov_through_opt_expr(
                curr.test,
                state.suppress_aliases,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
            )
            merged_while = _merge_branched_suppress_update(
                state.suppress_aliases,
                after_test,
                is_branched=is_br,
            )
            post_while_sup: Optional[_SuppressProvenance] = (
                _merge_loop_carried_suppress_provenance(
                    curr.body,
                    merged_while,
                    strict_with_suppression=state.strict_with_suppression,
                    extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
                )
                if merged_while is not None
                else None
            )
            stack.append(
                _StackFrame(
                    curr,
                    is_br,
                    in_lp,
                    act_nl,
                    force_after_span=f_after,
                    restore_suppress_box=[post_while_sup],
                )
            )
        else:
            _push_suppress_header_frame(
                stack, curr, frame, curr.test, span, allow_positive=False
            )
    stack.append(
        _StackFrame(
            curr.test,
            is_br,
            True if body_span else in_lp,
            act_nl,
            force_after_span=f_after,
        )
    )


def _push_compound_children(
    curr: ast.AST,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
    *,
    unconditional_post_del: bool = False,
    pre_stmt_suppress: Optional[Union[_SuppressProvenance, AbstractSet[str]]] = None,
) -> bool:
    """Pushes child nodes of compound control-flow statements with branch and loop state."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        _push_if_compound_children(
            curr, encloses, span, frame, state, stack, pre_stmt_suppress
        )
        return True

    if isinstance(curr, (ast.For, ast.AsyncFor, ast.While)):
        _push_loop_compound_children(
            curr, encloses, span, frame, unit_mutations, state, stack
        )
        return True

    if isinstance(curr, (ast.With, ast.AsyncWith)):
        _push_with_compound_children(curr, encloses, span, frame, state, stack, pre_stmt_suppress)
        return True

    if is_try_or_trystar_node(curr):
        _push_try_compound_children(curr, encloses, span, frame, state, stack, pre_stmt_suppress)
        return True

    if isinstance(curr, ast.ExceptHandler):
        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)
        h_span = encloses and _block_contains_span(curr.body, span)
        h_br = is_br if h_span else True
        if isinstance(curr.name, str) and _can_bind_in_fn_scope(
            curr.name, act_nl, state.fn_globals
        ):
            if h_span:
                state.pre_mutated.add(curr.name)
                state.post_mutated.add(curr.name)
                if not h_br:
                    state.pre_bound.add(curr.name)
                    state.pre_unbound.discard(curr.name)
                if in_lp:
                    state.loop_mutated.add(curr.name)
            else:
                _record_deletion_at_pos(
                    curr.name,
                    c_line,
                    c_col,
                    span,
                    frame,
                    state,
                    unconditional_post_del=unconditional_post_del,
                )
        _push_seq(stack, curr.body, h_br, in_lp, act_nl, force_after_span=f_after)
        if state.suppress_aliases is not None and (
            h_span or _is_frame_before_span(c_line, c_col, span, frame)
        ):
            h_clear = frozenset((curr.name,)) if isinstance(curr.name, str) else frozenset()
            if (
                curr.type is not None
                and not _node_contains_or_encloses_span(curr.type, span)
            ) or h_clear:
                _push_suppress_header_frame(
                    stack,
                    curr,
                    frame,
                    curr.type,
                    span,
                    post_clear=h_clear,
                    is_branched=h_br,
                )
        if curr.type is not None:
            stack.append(_StackFrame(curr.type, True, in_lp, act_nl, force_after_span=f_after))
        return True

    if is_match_node(curr):
        _push_match_compound_children(curr, encloses, span, frame, state, stack, pre_stmt_suppress)
        return True

    if isinstance(curr, (ast.BoolOp, ast.Compare)):
        uncond_exprs: Sequence[ast.AST] = (
            curr.values[:1] if isinstance(curr, ast.BoolOp) else (curr.left, *curr.comparators[:1])
        )
        short_circuit_exprs: Sequence[ast.AST] = (
            curr.values[1:] if isinstance(curr, ast.BoolOp) else curr.comparators[1:]
        )
        _push_seq(
            stack,
            short_circuit_exprs,
            True,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        _push_seq(
            stack,
            uncond_exprs,
            is_br,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        return True

    if isinstance(curr, ast.IfExp):
        stack.append(_StackFrame(curr.orelse, True, in_lp, act_nl, force_after_span=f_after))
        stack.append(_StackFrame(curr.body, True, in_lp, act_nl, force_after_span=f_after))
        stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl, force_after_span=f_after))
        return True

    return False


def _record_carrier_target_closures(
    carrier_names: Iterable[str],
    reads: AbstractSet[str],
    muts: AbstractSet[str],
    bufs: _CellTraversalBuffers,
) -> None:
    """Records sibling closure entries and mutation alias edges for carrier targets."""
    for oc_name in carrier_names:
        bufs.sibling_closures.append((oc_name, set(reads), set(muts)))
        if bufs.alias_edges is not None and (reads or muts):
            bufs.alias_edges.append((f"*{oc_name}", {oc_name}))


def _record_class_expr_closures(
    cls_name: str,
    exprs: Sequence[ast.AST],
    bufs: _CellTraversalBuffers,
    *,
    is_enclosing_cls: bool,
) -> bool:
    """Extracts closure reads, mutations, aliases, and calls from class expressions."""
    if not exprs:
        return False
    calls: Set[str] = set()
    reads, muts, loads = _extract_carrier_value_closure_and_load_info(
        exprs, bufs.scope_ctx, called_names_out=calls
    )
    recorded = False
    if reads or muts:
        recorded = True
        bufs.sibling_closures.append((cls_name, set(reads), set(muts)))
        if is_enclosing_cls:
            bufs.class_body_sib_indices.add(len(bufs.sibling_closures) - 1)
    if loads and bufs.alias_edges is not None:
        recorded = True
        bufs.alias_edges.append((cls_name, set(loads)))
        if is_enclosing_cls:
            bufs.class_body_alias_edge_indices.add(len(bufs.alias_edges) - 1)
    if calls and bufs.call_edges is not None:
        bufs.call_edges.append((cls_name, set(calls)))
    return recorded


def _record_enclosing_class_closures(
    cls_node: ast.ClassDef,
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
) -> None:
    """Records sibling closures and class-body carrier edges on an enclosing ClassDef."""
    sibling_nl_deletions = bufs.sibling_nl_deletions
    cls_dec_carriers = {
        d_name
        for d_name in _extract_decorator_carrier_targets(cls_node)
        if d_name not in bufs.state.fn_globals
    }
    cls_hdr_exprs: List[ast.AST] = [
        *cls_node.bases,
        *(kw.value for kw in cls_node.keywords if kw.value is not None),
        *cls_node.decorator_list,
    ]
    has_cls_closures = _record_class_expr_closures(
        cls_node.name, cls_hdr_exprs, bufs, is_enclosing_cls=True
    )
    for stmt in _iter_class_body_methods(cls_node):
        if _is_node_inside_span(stmt, span) or _strictly_encloses_span(stmt, span):
            continue
        s_reads = _extract_cached_scope_free_reads(stmt)
        s_muts, s_dels = _extract_closure_nonlocal_mutations_and_deletions(stmt)
        if sibling_nl_deletions is not None and s_dels:
            sibling_nl_deletions.update(s_dels)
        s_carriers = set(
            _extract_fn_outer_carrier_targets(
                stmt, bufs.state.fn_globals, bufs.scope_ctx
            )
        ) | {
            d_name
            for d_name in _extract_decorator_carrier_targets(stmt)
            if d_name not in bufs.state.fn_globals
        }
        if s_reads or s_muts:
            has_cls_closures = True
            bufs.sibling_closures.append((cls_node.name, set(s_reads), set(s_muts)))
            bufs.class_body_sib_indices.add(len(bufs.sibling_closures) - 1)
            _record_carrier_target_closures(s_carriers, s_reads, s_muts, bufs)
    for attr_stmt in _iter_class_body_non_method_stmts(cls_node):
        if _is_node_inside_span(attr_stmt, span) or _strictly_encloses_span(
            attr_stmt, span
        ):
            continue
        for _is_reb, _c_targets, c_values in _iter_carrier_target_entries(attr_stmt):
            if _record_class_expr_closures(
                cls_node.name, c_values, bufs, is_enclosing_cls=True
            ):
                has_cls_closures = True
    if cls_dec_carriers and has_cls_closures and bufs.alias_edges is not None:
        for d_name in cls_dec_carriers:
            bufs.alias_edges.append((f"*{d_name}", {cls_node.name}))


def _record_non_enclosing_def_or_class_closures(
    curr: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    *,
    span_in_hdr: bool,
    estart: Optional[int],
    ecol: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    bufs: _CellTraversalBuffers,
) -> None:
    """Records closure reads, factories, return refs, and outer carriers for non-enclosing scopes."""
    state = bufs.state
    free_r = (
        _extract_nested_scope_free_reads(curr, include_header=False)
        if span_in_hdr
        else _extract_cached_scope_free_reads(curr)
    )
    nl_mut, nl_del = _extract_closure_nonlocal_mutations_and_deletions(curr)
    if bufs.sibling_nl_deletions is not None and nl_del:
        bufs.sibling_nl_deletions.update(nl_del)
    sib_name = (
        curr.name
        if _can_bind_in_fn_scope(curr.name, frame.active_nonlocals, state.fn_globals)
        else None
    )
    outer_carriers = _extract_fn_outer_carrier_targets(
        curr, state.fn_globals, bufs.scope_ctx
    )
    if sib_name is not None:
        if bufs.factory_names is not None and (
            outer_carriers or _is_closure_factory_scope(curr, bufs.scope_ctx)
        ):
            bufs.factory_names.add(sib_name)
        if bufs.fn_return_refs is not None and isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            ret_loads, ret_calls = _extract_fn_returned_closure_refs(
                curr, bufs.scope_ctx
            )
            if ret_loads or ret_calls:
                entry = bufs.fn_return_refs.setdefault(sib_name, (set(), set()))
                entry[0].update(ret_loads)
                entry[1].update(ret_calls)
        elif (
            bufs.fn_return_refs is not None
            and isinstance(curr, ast.ClassDef)
            and free_r
        ):
            entry = bufs.fn_return_refs.setdefault(sib_name, (set(), set()))
            entry[0].update(free_r)
        if isinstance(curr, ast.ClassDef):
            cls_hdr_exprs: List[ast.AST] = [
                *curr.bases,
                *(kw.value for kw in curr.keywords if kw.value is not None),
                *curr.decorator_list,
            ]
            _record_class_expr_closures(
                sib_name, cls_hdr_exprs, bufs, is_enclosing_cls=False
            )
    carrier_targets = set(outer_carriers) | {
        d_name
        for d_name in _extract_decorator_carrier_targets(curr)
        if d_name not in state.fn_globals
    }
    bufs.sibling_closures.append((sib_name, free_r, nl_mut))
    _record_carrier_target_closures(carrier_targets, free_r, nl_mut, bufs)
    if span_in_hdr or _is_frame_after_span(estart, ecol, span, frame):
        state.post_reads.update(free_r)


def _handle_def_or_class_scope_node(
    curr: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    span: _SpanBounds,
    frame: _StackFrame,
    bufs: _CellTraversalBuffers,
    stack: List[_StackFrame],
    *,
    is_direct_span: bool,
) -> None:
    """Processes nested def/async-def/class statements and schedules class bodies/headers."""
    state = bufs.state
    estart, ecol = _node_earliest_start_pos(curr)
    hdr_nodes = list(iter_nested_scope_header_nodes(curr))
    span_in_hdr = any(
        _is_node_inside_span(hdr, span) or _strictly_encloses_span(hdr, span)
        for hdr in hdr_nodes
    )
    enclosing_cls = (
        isinstance(curr, ast.ClassDef)
        and not span_in_hdr
        and _strictly_encloses_span(curr, span)
    )
    if span_in_hdr or (enclosing_cls and is_direct_span):
        if _can_bind_in_fn_scope(curr.name, frame.active_nonlocals, state.fn_globals):
            state.post_mutated.add(curr.name)
            if frame.in_loop:
                state.loop_mutated.add(curr.name)
    else:
        _record_binding_at_pos((curr.name,), estart, ecol, span, frame, state)
    if enclosing_cls and isinstance(curr, ast.ClassDef):
        _record_enclosing_class_closures(curr, span, bufs)
    elif not enclosing_cls:
        _record_non_enclosing_def_or_class_closures(
            curr,
            span_in_hdr=span_in_hdr,
            estart=estart,
            ecol=ecol,
            span=span,
            frame=frame,
            bufs=bufs,
        )
    if isinstance(curr, ast.ClassDef):
        cls_nonlocals = _extract_direct_nonlocals(curr)
        if (
            enclosing_cls
            or cls_nonlocals
            or not _is_frame_after_span(estart, ecol, span, frame)
        ):
            _push_seq(
                stack,
                curr.body,
                frame.is_branched,
                frame.in_loop,
                cls_nonlocals,
                force_after_span=frame.force_after_span,
            )
        if enclosing_cls and state.suppress_aliases is not None:
            for hdr in reversed(hdr_nodes):
                stack.append(
                    _StackFrame(
                        curr,
                        frame.is_branched,
                        frame.in_loop,
                        frame.active_nonlocals,
                        force_after_span=frame.force_after_span,
                        advance_suppress_header=(hdr, frozenset(), frozenset(), True),
                    )
                )
    _push_seq(
        stack,
        hdr_nodes,
        frame.is_branched,
        frame.in_loop,
        frame.active_nonlocals,
        force_after_span=frame.force_after_span,
    )


def _collect_comprehension_carrier_edges(
    curr: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    frame: _StackFrame,
    bufs: _CellTraversalBuffers,
) -> None:
    """Extracts NamedExpr and Call carrier closure edges nested inside comprehension bodies."""
    init_exprs, init_targets = _iter_comprehension_body_exprs_and_targets(curr)
    base_targets = frozenset(init_targets)
    comp_sub_stack: List[Tuple[ast.AST, frozenset[str]]] = [
        (expr, base_targets) for expr in init_exprs
    ]
    comp_carriers: List[Tuple[ast.AST, frozenset[str]]] = []
    while comp_sub_stack:
        sub, active_targets = comp_sub_stack.pop()
        if isinstance(sub, (ast.NamedExpr, ast.Call)):
            comp_carriers.append((sub, active_targets))
        if isinstance(sub, _COMPREHENSION_TYPES):
            sub_exprs, sub_targets = _iter_comprehension_body_exprs_and_targets(sub)
            if sub.generators:
                comp_sub_stack.append((sub.generators[0].iter, active_targets))
            next_targets = active_targets | frozenset(sub_targets)
            comp_sub_stack.extend((expr, next_targets) for expr in sub_exprs)
            continue
        comp_sub_stack.extend(
            (hdr, active_targets)
            for hdr in (
                iter_nested_scope_header_nodes(sub)
                if isinstance(
                    sub,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Lambda,
                    ),
                )
                else ast.iter_child_nodes(sub)
            )
        )
    for c_node, c_excluded in comp_carriers:
        _record_carrier_closures_and_aliases(
            c_node,
            frame,
            bufs.sibling_closures,
            bufs.alias_edges,
            bufs.scope_ctx,
            bufs.call_edges,
            excluded_names=c_excluded,
            fn_globals=bufs.state.fn_globals,
        )


def _handle_nested_scope_node(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    bufs: _CellTraversalBuffers,
    stack: List[_StackFrame],
    *,
    is_direct_span: bool = True,
) -> bool:
    """Processes nested function, class, lambda, or comprehension scopes during cell analysis."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        _handle_def_or_class_scope_node(
            curr, span, frame, bufs, stack, is_direct_span=is_direct_span
        )
        return True

    if isinstance(curr, ast.Lambda):
        free_r = _extract_expr_scope_free_reads(curr, include_header=False)
        bufs.sibling_closures.append((None, free_r, set()))
        if _is_frame_after_span(c_line, c_col, span, frame):
            bufs.state.post_reads.update(free_r)
        _push_seq(
            stack,
            list(iter_nested_scope_header_nodes(curr)),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
        )
        return True

    if isinstance(curr, _COMPREHENSION_TYPES):
        body_w = _extract_comprehension_body_walrus_names(curr)
        if body_w:
            _record_binding_at_pos(
                body_w,
                c_line,
                c_col,
                span,
                frame._replace(is_branched=True),
                bufs.state,
            )
        c_reads, c_muts = _extract_comprehension_deferred_closure_info(
            curr, bufs.scope_ctx
        )
        if c_reads or c_muts:
            bufs.sibling_closures.append((None, c_reads, c_muts))
        if bufs.alias_edges is not None:
            _collect_comprehension_carrier_edges(curr, frame, bufs)
        if _is_frame_after_span(c_line, c_col, span, frame):
            bufs.state.post_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
        _push_seq(
            stack,
            [curr.generators[0].iter] if curr.generators else (),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
        )
        return True

    return False


def _record_carrier_closures_and_aliases(
    curr: ast.AST,
    frame: _StackFrame,
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]],
    alias_edges: List[Tuple[str, Set[str]]],
    scope_ctx: Optional[_FnScopeContext] = None,
    call_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    *,
    excluded_names: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> None:
    """Extracts inline closures and variable-rebinding alias edges from curr."""
    for is_rebound, c_targets, c_values in _iter_carrier_target_entries(curr):
        valid_targets = [
            t
            for t in c_targets
            if t not in excluded_names
            and (
                _can_bind_in_fn_scope(t, frame.active_nonlocals, fn_globals)
                if is_rebound
                else t not in fn_globals
            )
        ]
        if not valid_targets:
            continue
        c_calls: Set[str] = set()
        c_reads, c_muts, c_loads = _extract_carrier_value_closure_and_load_info(
            c_values, scope_ctx, called_names_out=c_calls
        )
        if excluded_names:
            c_reads -= set(excluded_names)
            c_loads -= set(excluded_names)
            c_calls -= set(excluded_names)
        edge_prefix = "" if is_rebound else "*"
        for t_name in valid_targets:
            if c_reads or c_muts:
                sibling_closures.append((t_name, set(c_reads), set(c_muts)))
            edge_tgt = f"{edge_prefix}{t_name}"
            edge_srcs = set(c_loads)
            if not is_rebound and (c_reads or c_muts):
                edge_srcs.add(t_name)
            if edge_srcs:
                alias_edges.append((edge_tgt, edge_srcs))
            if c_calls and call_edges is not None:
                call_edges.append((edge_tgt, set(c_calls)))


def _record_simple_node_bindings_and_reads(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    *,
    unconditional_post_del: bool = False,
) -> None:
    """Records bindings, deletions, and post-span reads for leaf/simple AST nodes."""
    if isinstance(curr, ast.AugAssign) and isinstance(curr.target, ast.Name):
        if _is_frame_after_span(c_line, c_col, span, frame) or (
            curr.value is not None and _node_contains_or_encloses_span(curr.value, span)
        ):
            state.post_reads.add(curr.target.id)

    if isinstance(curr, ast.Name):
        if isinstance(curr.ctx, ast.Store):
            _record_binding_at_pos((curr.id,), c_line, c_col, span, frame, state)
        elif isinstance(curr.ctx, ast.Del):
            _record_deletion_at_pos(
                curr.id,
                c_line,
                c_col,
                span,
                frame,
                state,
                unconditional_post_del=unconditional_post_del,
            )
        elif isinstance(curr.ctx, ast.Load) and _is_frame_after_span(c_line, c_col, span, frame):
            state.post_reads.add(curr.id)
    elif isinstance(curr, (ast.Import, ast.ImportFrom)):
        imp_names: Set[str] = set()
        _record_import_names(curr, imp_names)
        _record_binding_at_pos(imp_names, c_line, c_col, span, frame, state)
    else:
        bound_name = _extract_type_alias_or_pattern_name(curr)
        if bound_name is not None:
            _record_binding_at_pos((bound_name,), c_line, c_col, span, frame, state)


class _StmtCellDelta(NamedTuple):
    """Cached cell-binding and closure delta for a statement disjoint from the unit span."""

    pre_bound: frozenset[str]
    pre_mutated: frozenset[str]
    post_mutated: frozenset[str]
    post_reads: frozenset[str]
    loop_mutated: frozenset[str]
    sibling_closures: Tuple[Tuple[Optional[str], frozenset[str], frozenset[str]], ...]
    concurrent_sib_offsets: Tuple[int, ...]
    alias_edges: Tuple[Tuple[str, frozenset[str]], ...]
    call_edges: Tuple[Tuple[str, frozenset[str]], ...]
    factory_names: frozenset[str]
    fn_return_refs: Tuple[Tuple[str, frozenset[str], frozenset[str]], ...]
    sibling_nl_deletions: frozenset[str]
    heuristic_ctx_managers: Tuple[str, ...] = ()
    wildcard_ctx_managers: Tuple[Tuple[str, str], ...] = ()

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns an empty tuple because _StmtCellDelta retains only string/int tuples."""
        return ()


class _CellTraversalBuffers(NamedTuple):
    """Mutable accumulators populated while traversing function AST frames."""

    state: _CellBindingState
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]]
    concurrent_sib_indices: Set[int]
    alias_edges: List[Tuple[str, Set[str]]]
    call_edges: List[Tuple[str, Set[str]]]
    factory_names: Set[str]
    fn_return_refs: Dict[str, Tuple[Set[str], Set[str]]]
    sibling_nl_deletions: Set[str]
    scope_ctx: _FnScopeContext
    class_body_sib_indices: Set[int]
    class_body_alias_edge_indices: Set[int]
    unit_mutations: AbstractSet[str] = frozenset()


def _make_empty_traversal_buffers(
    fn_globals: AbstractSet[str],
    scope_ctx: _FnScopeContext,
    pre_bound: Optional[Set[str]] = None,
    unit_mutations: AbstractSet[str] = frozenset(),
    initial_suppress_aliases: Optional[Union[_SuppressProvenance, AbstractSet[str]]] = None,
    *,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    seed_pre_mutated: bool = True,
) -> _CellTraversalBuffers:
    """Creates fresh mutable traversal buffers for enclosing function cell analysis."""
    bound_set: Set[str] = set(pre_bound) if pre_bound is not None else set()
    return _CellTraversalBuffers(
        state=_CellBindingState(
            pre_bound=bound_set,
            pre_mutated=set(bound_set) if seed_pre_mutated else set(),
            post_mutated=set(),
            post_reads=set(),
            loop_mutated=set(),
            pre_unbound=set(),
            fn_globals=fn_globals,
            suppress_aliases=_coerce_suppress_provenance(initial_suppress_aliases),
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=frozenset(extra_non_suppressing_ctx_managers),
        ),
        sibling_closures=[],
        concurrent_sib_indices=set(),
        alias_edges=[],
        call_edges=[],
        factory_names=set(),
        fn_return_refs={},
        sibling_nl_deletions=set(),
        scope_ctx=scope_ctx,
        class_body_sib_indices=set(),
        class_body_alias_edge_indices=set(),
        unit_mutations=unit_mutations,
    )


def _apply_stmt_cell_delta(
    bufs: _CellTraversalBuffers,
    delta: _StmtCellDelta,
    *,
    is_branched: bool = False,
    in_loop: bool = False,
) -> None:
    """Applies a cached disjoint statement cell delta to traversal buffers."""
    if not is_branched:
        bufs.state.pre_bound.update(delta.pre_bound)
        bufs.state.pre_unbound.difference_update(delta.pre_bound)
    bufs.state.pre_mutated.update(delta.pre_mutated)
    bufs.state.post_mutated.update(delta.post_mutated)
    bufs.state.post_reads.update(delta.post_reads)
    bufs.state.loop_mutated.update(delta.loop_mutated)
    if in_loop:
        bufs.state.loop_mutated.update(delta.pre_mutated)
        bufs.state.loop_mutated.update(delta.post_mutated)
    base_sib_len = len(bufs.sibling_closures)
    for name, reads, muts in delta.sibling_closures:
        bufs.sibling_closures.append((name, set(reads), set(muts)))
    if in_loop:
        bufs.concurrent_sib_indices.update(range(base_sib_len, len(bufs.sibling_closures)))
    else:
        for rel_idx in delta.concurrent_sib_offsets:
            bufs.concurrent_sib_indices.add(base_sib_len + rel_idx)
    for t_name, srcs in delta.alias_edges:
        bufs.alias_edges.append((t_name, set(srcs)))
    for t_name, calls in delta.call_edges:
        bufs.call_edges.append((t_name, set(calls)))
    bufs.factory_names.update(delta.factory_names)
    for fn_name, ret_loads, ret_calls in delta.fn_return_refs:
        entry = bufs.fn_return_refs.setdefault(fn_name, (set(), set()))
        entry[0].update(ret_loads)
        entry[1].update(ret_calls)
    bufs.sibling_nl_deletions.update(delta.sibling_nl_deletions)


def _get_stmt_with_summary(
    stmt: ast.stmt,
) -> Tuple[bool, frozenset[str]]:
    """Returns cached (has_with, stmt_name_ids) for a statement node."""
    with _downstream_cache_lock:
        if stmt in _stmt_with_summary_cache:
            _stmt_with_summary_cache.move_to_end(stmt)
            return _stmt_with_summary_cache[stmt]
    has_w = False
    name_ids: Set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            has_w = True
        elif isinstance(node, ast.Name):
            name_ids.add(node.id)
    result: Tuple[bool, frozenset[str]] = (
        has_w,
        frozenset(name_ids) if has_w else frozenset(),
    )
    with _downstream_cache_lock:
        if stmt in _stmt_with_summary_cache:
            _stmt_with_summary_cache.move_to_end(stmt)
            return _stmt_with_summary_cache[stmt]
        while (
            len(_stmt_with_summary_cache) >= _MAX_STMT_WITH_SUMMARY_CACHE_SIZE
            and _stmt_with_summary_cache
        ):
            _stmt_with_summary_cache.popitem(last=False)
        _stmt_with_summary_cache[stmt] = result
        return result


def _evict_stmt_delta_if_full(
    stmt_deltas: Dict[Tuple[Any, ...], Optional[_StmtCellDelta]],
    incoming_key: Tuple[Any, ...],
) -> None:
    """Evicts the least-recently-used entry when stmt_deltas reaches _MAX_STMT_DELTAS_PER_FN."""
    if (
        incoming_key in stmt_deltas
        or len(stmt_deltas) < _MAX_STMT_DELTAS_PER_FN
        or not stmt_deltas
    ):
        return
    stmt_deltas.pop(next(iter(stmt_deltas)), None)


def _try_apply_cached_stmt_delta(
    stmt: ast.stmt,
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
    stmt_deltas: Dict[Tuple[Any, ...], Optional[_StmtCellDelta]],
    *,
    is_branched: bool = False,
    in_loop: bool = False,
) -> bool:
    """Computes or applies a cached _StmtCellDelta for a disjoint statement."""
    estart, ecol = _node_earliest_start_pos(stmt)
    c_end = getattr(stmt, "end_lineno", None)
    c_end_col = getattr(stmt, "end_col_offset", None)
    if not (isinstance(estart, int) and estart > 0 and isinstance(c_end, int)):
        return False
    is_before = c_end < span.start or (
        c_end == span.start
        and span.start_col is not None
        and isinstance(c_end_col, int)
        and c_end_col <= span.start_col
    )
    is_after = estart > span.end or (
        estart == span.end
        and span.end_col is not None
        and isinstance(ecol, int)
        and ecol >= span.end_col
    )
    if not (is_before or is_after):
        return False
    pre_prov = _coerce_suppress_provenance(bufs.state.suppress_aliases)
    sup_key = pre_prov.as_cache_key_tuple() if pre_prov else ()
    strict_w = bool(bufs.state.strict_with_suppression)
    extra_non_sup = frozenset(bufs.state.extra_non_suppressing_ctx_managers)
    has_with = False
    stmt_name_ids: frozenset[str] = frozenset()
    if is_before:
        has_with, stmt_name_ids = _get_stmt_with_summary(stmt)
    init_pre_bound: frozenset[str] = (
        frozenset(bufs.state.pre_bound & stmt_name_ids)
        if has_with and bufs.state.pre_bound
        else frozenset()
    )
    key: Tuple[Any, ...] = (
        (stmt, is_before, sup_key, strict_w, init_pre_bound, extra_non_sup)
        if has_with
        else (stmt, is_before)
    )
    with _downstream_cache_lock:
        has_key = key in stmt_deltas
        if has_key:
            delta = stmt_deltas.get(key)
            if isinstance(stmt_deltas, _TrackedCachedDict):
                stmt_deltas.touch_lru(key)
        else:
            delta = None
    if not has_key:
        if _extract_shallow_deleted_names(stmt):
            with _downstream_cache_lock:
                _evict_stmt_delta_if_full(stmt_deltas, key)
                stmt_deltas.setdefault(key, None)
            return False
        dummy_span = _SpanBounds(c_end + 1, c_end + 1) if is_before else _SpanBounds(0, 0)
        sub_bufs = _make_empty_traversal_buffers(
            bufs.state.fn_globals,
            bufs.scope_ctx,
            pre_bound=set(init_pre_bound) if init_pre_bound else None,
            initial_suppress_aliases=pre_prov,
            strict_with_suppression=strict_w,
            extra_non_suppressing_ctx_managers=extra_non_sup,
            seed_pre_mutated=False,
        )
        with collect_heuristic_ctx_managers(isolate=is_branched) as stmt_heuristics, collect_wildcard_ctx_managers(isolate=is_branched) as stmt_wildcards:
            _traverse_fn_cell_frames(
                [_StackFrame(stmt, False, False, None)],
                dummy_span,
                sub_bufs,
            )
        delta_pre_mutated = frozenset(sub_bufs.state.pre_mutated)
        delta_pre_bound = frozenset(
            (sub_bufs.state.pre_bound - init_pre_bound)
            | (sub_bufs.state.pre_bound & delta_pre_mutated)
        )
        computed_delta = _StmtCellDelta(
            pre_bound=delta_pre_bound,
            pre_mutated=delta_pre_mutated,
            post_mutated=frozenset(sub_bufs.state.post_mutated),
            post_reads=frozenset(sub_bufs.state.post_reads),
            loop_mutated=frozenset(sub_bufs.state.loop_mutated),
            sibling_closures=tuple(
                (name, frozenset(reads), frozenset(muts))
                for name, reads, muts in sub_bufs.sibling_closures
            ),
            concurrent_sib_offsets=tuple(sorted(sub_bufs.concurrent_sib_indices)),
            alias_edges=tuple((t_name, frozenset(srcs)) for t_name, srcs in sub_bufs.alias_edges),
            call_edges=tuple((t_name, frozenset(calls)) for t_name, calls in sub_bufs.call_edges),
            factory_names=frozenset(sub_bufs.factory_names),
            fn_return_refs=tuple(
                (fn_name, frozenset(r_loads), frozenset(r_calls))
                for fn_name, (
                    r_loads,
                    r_calls,
                ) in sub_bufs.fn_return_refs.items()
            ),
            sibling_nl_deletions=frozenset(sub_bufs.sibling_nl_deletions),
            heuristic_ctx_managers=tuple(sorted(stmt_heuristics)),
            wildcard_ctx_managers=tuple(sorted(stmt_wildcards)),
        )
        with _downstream_cache_lock:
            _evict_stmt_delta_if_full(stmt_deltas, key)
            delta = stmt_deltas.setdefault(key, computed_delta)
    if delta is None:
        return False
    if has_key and not is_branched:
        if not strict_w and delta.heuristic_ctx_managers:
            for mgr_name in delta.heuristic_ctx_managers:
                record_heuristic_non_suppressing_ctx_manager(mgr_name)
        if delta.wildcard_ctx_managers:
            for wc_pat, mgr_name in delta.wildcard_ctx_managers:
                record_wildcard_ctx_manager_match(wc_pat, mgr_name)
    if is_before and bufs.state.suppress_aliases is not None:
        _advance_suppress_aliases_for_stmt(
            bufs,
            stmt,
            is_branched=is_branched,
        )
    _apply_stmt_cell_delta(bufs, delta, is_branched=is_branched, in_loop=in_loop)
    return True


def _handle_enclosed_fn_frame(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    stack: List[_StackFrame],
) -> Optional[bool]:
    """Handles a stack frame when curr is the target enclosed_fn; returns multi-invoked flag."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
        estart, ecol = _node_earliest_start_pos(curr)
        _record_binding_at_pos((curr.name,), estart, ecol, span, frame, state)
        multi = (
            frame.in_loop
            or frame.active_nonlocals is not None
            or bool(curr.decorator_list)
            or curr.name in _extract_cached_scope_free_reads(curr)
        )
        _push_seq(
            stack,
            list(iter_nested_scope_header_nodes(curr)),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
        )
        return multi
    if isinstance(curr, ast.Lambda):
        _push_seq(
            stack,
            list(iter_nested_scope_header_nodes(curr)),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
        )
        return True
    if isinstance(curr, _COMPREHENSION_TYPES):
        body_w = _extract_comprehension_body_walrus_names(curr)
        if body_w:
            _record_binding_at_pos(
                body_w,
                c_line,
                c_col,
                span,
                frame._replace(is_branched=True, in_loop=True),
                state,
            )
        _push_seq(
            stack,
            [curr.generators[0].iter] if curr.generators else (),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
        )
        return True
    return None


def _push_inside_span_def_headers(
    curr: ast.AST, span: _SpanBounds, frame: _StackFrame, stack: List[_StackFrame]
) -> None:
    """Pushes non-span header expressions of def/class/lambda nodes inside the unit span."""
    if isinstance(
        curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    ):
        pre_hdrs = [
            hdr
            for hdr in iter_nested_scope_header_nodes(curr)
            if not _is_node_inside_span(hdr, span)
        ]
        _push_seq(
            stack, pre_hdrs, frame.is_branched, frame.in_loop, frame.active_nonlocals
        )


def _node_triggers_enclosed_fn_multi_invoke(
    curr: ast.AST,
    enclosed_fn_name: str,
    span: _SpanBounds,
    frame: _StackFrame,
    direct_enclosed_call_funcs: Set[int],
    call_count_box: List[int],
    *,
    is_nested_scope: bool,
) -> bool:
    """Checks whether curr invokes/captures enclosed_fn_name in a multi-invocation context."""
    if is_nested_scope:
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return not (
                isinstance(curr, ast.ClassDef) and _strictly_encloses_span(curr, span)
            ) and enclosed_fn_name in _extract_cached_scope_free_reads(curr)
        if isinstance(curr, (ast.Lambda, *_COMPREHENSION_TYPES)):
            return enclosed_fn_name in _extract_expr_scope_free_reads(
                curr, include_header=False
            )
        return False
    if isinstance(curr, ast.Call):
        if isinstance(curr.func, ast.Name) and curr.func.id == enclosed_fn_name:
            direct_enclosed_call_funcs.add(id(curr.func))
            call_count_box[0] += 1
            return frame.in_loop or call_count_box[0] >= 2
        return False
    return (
        isinstance(curr, ast.Name)
        and isinstance(curr.ctx, ast.Load)
        and curr.id == enclosed_fn_name
        and id(curr) not in direct_enclosed_call_funcs
    )


def _push_assign_or_child_frames(
    curr: ast.AST, span: _SpanBounds, frame: _StackFrame, stack: List[_StackFrame]
) -> None:
    """Schedules assignment RHS/LHS or generic AST child frames onto the traversal stack."""
    if isinstance(curr, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        rhs_val = curr.value
        if rhs_val is not None and _node_contains_or_encloses_span(rhs_val, span):
            lhs_targets: Sequence[ast.AST] = (
                curr.targets if isinstance(curr, ast.Assign) else (curr.target,)
            )
            target_after = not (
                isinstance(curr, ast.AugAssign) and not isinstance(curr.target, ast.Name)
            )
            _push_seq(
                stack,
                lhs_targets,
                frame.is_branched,
                frame.in_loop,
                frame.active_nonlocals,
                force_after_span=frame.force_after_span or target_after,
            )
            rhs_seq: Sequence[ast.AST] = (
                (curr.annotation, rhs_val)
                if isinstance(curr, ast.AnnAssign)
                else (rhs_val,)
            )
            _push_seq(
                stack,
                rhs_seq,
                frame.is_branched,
                frame.in_loop,
                frame.active_nonlocals,
                force_after_span=frame.force_after_span,
            )
            return
    child_nodes: Sequence[ast.AST] = (
        (curr.annotation,)
        if isinstance(curr, ast.AnnAssign)
        and curr.value is None
        and isinstance(curr.target, ast.Name)
        else tuple(ast.iter_child_nodes(curr))
    )
    _push_seq(
        stack,
        child_nodes,
        frame.is_branched,
        frame.in_loop,
        frame.active_nonlocals,
        force_after_span=frame.force_after_span,
    )


def _traverse_fn_cell_frames(
    stack: List[_StackFrame],
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
    enclosed_fn: Optional[ast.AST] = None,
    stmt_deltas: Optional[Dict[Tuple[Any, ...], Optional[_StmtCellDelta]]] = None,
) -> bool:
    """Traverses AST frames to populate cell bindings, closures, and multi-invocation state."""
    enclosed_fn_name: Optional[str] = (
        enclosed_fn.name
        if isinstance(enclosed_fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    call_count_box = [0]
    enclosed_fn_multi_invoked = False
    direct_enclosed_call_funcs: Set[int] = set()

    while stack:
        frame = stack.pop()
        if frame.post_pre_bound is not None:
            bufs.state.pre_bound.update(frame.post_pre_bound)
            bufs.state.pre_unbound.difference_update(frame.post_pre_bound)
            continue
        if frame.save_suppress_box is not None:
            frame.save_suppress_box[0] = bufs.state.suppress_aliases
            continue
        if frame.restore_suppress_box is not None:
            bufs.state.suppress_aliases = frame.restore_suppress_box[0]
            continue
        if frame.advance_suppress_header is not None:
            if bufs.state.suppress_aliases is not None:
                hdr_expr, pre_clear, post_clear, hdr_allow_pos = (
                    frame.advance_suppress_header
                )
                pre_h = _coerce_suppress_provenance(bufs.state.suppress_aliases)
                upd_h = pre_h
                for c_name in pre_clear:
                    upd_h = upd_h.clear_name(c_name)
                if hdr_expr is not None:
                    upd_h = _advance_prov_through_opt_expr(
                        hdr_expr,
                        upd_h,
                        allow_positive=hdr_allow_pos and not frame.is_branched,
                        extra_non_suppressing_ctx_managers=bufs.state.extra_non_suppressing_ctx_managers,
                    )
                for c_name in post_clear:
                    upd_h = upd_h.clear_name(c_name)
                bufs.state.suppress_aliases = _merge_branched_suppress_update(
                    pre_h,
                    upd_h,
                    is_branched=frame.is_branched or not hdr_allow_pos,
                )
            continue
        curr = frame.node
        if (
            stmt_deltas is not None
            and enclosed_fn is None
            and not frame.force_after_span
            and frame.active_nonlocals is None
            and isinstance(curr, ast.stmt)
            and _try_apply_cached_stmt_delta(
                curr,
                span,
                bufs,
                stmt_deltas,
                is_branched=frame.is_branched,
                in_loop=frame.in_loop,
            )
        ):
            continue

        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)
        pre_stmt_suppress = bufs.state.suppress_aliases
        if (
            isinstance(curr, ast.stmt)
            and not _strictly_encloses_span(curr, span)
            and bufs.state.suppress_aliases is not None
            and _is_frame_before_span(c_line, c_col, span, frame)
        ):
            if _is_compound_scope_traversal_stmt(curr):
                post_stmt_sup = _compute_advanced_suppress_aliases(
                    bufs,
                    curr,
                    _coerce_suppress_provenance(pre_stmt_suppress),
                    is_branched=frame.is_branched,
                )
                stack.append(
                    _StackFrame(
                        curr,
                        frame.is_branched,
                        frame.in_loop,
                        frame.active_nonlocals,
                        force_after_span=frame.force_after_span,
                        restore_suppress_box=[post_stmt_sup],
                    )
                )
            else:
                _advance_suppress_aliases_for_stmt(
                    bufs,
                    curr,
                    is_branched=frame.is_branched,
                )

        if curr is enclosed_fn:
            enc_res = _handle_enclosed_fn_frame(
                curr, c_line, c_col, span, frame, bufs.state, stack
            )
            if enc_res is not None:
                if enc_res:
                    enclosed_fn_multi_invoked = True
                continue

        if _is_node_inside_span(curr, span):
            _push_inside_span_def_headers(curr, span, frame, stack)
            continue

        prev_sib_len = len(bufs.sibling_closures)
        eff_line, eff_col = (
            _node_earliest_start_pos(curr)
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else (c_line, c_col)
        )
        is_conc_pos = (
            enclosed_fn is not None
            or frame.in_loop
            or not _is_frame_after_span(eff_line, eff_col, span, frame)
        )
        if _handle_nested_scope_node(
            curr, c_line, c_col, span, frame, bufs, stack,
            is_direct_span=enclosed_fn is None,
        ):
            if is_conc_pos:
                bufs.concurrent_sib_indices.update(
                    range(prev_sib_len, len(bufs.sibling_closures))
                )
            if (
                enclosed_fn_name is not None
                and not enclosed_fn_multi_invoked
                and _node_triggers_enclosed_fn_multi_invoke(
                    curr, enclosed_fn_name, span, frame,
                    direct_enclosed_call_funcs, call_count_box, is_nested_scope=True,
                )
            ):
                enclosed_fn_multi_invoked = True
            continue

        if (
            enclosed_fn_name is not None
            and _node_triggers_enclosed_fn_multi_invoke(
                curr, enclosed_fn_name, span, frame,
                direct_enclosed_call_funcs, call_count_box, is_nested_scope=False,
            )
        ):
            enclosed_fn_multi_invoked = True

        _record_carrier_closures_and_aliases(
            curr, frame, bufs.sibling_closures, bufs.alias_edges,
            bufs.scope_ctx, bufs.call_edges, fn_globals=bufs.state.fn_globals,
        )
        if is_conc_pos:
            bufs.concurrent_sib_indices.update(
                range(prev_sib_len, len(bufs.sibling_closures))
            )

        if _push_compound_children(
            curr, span, frame, bufs.unit_mutations, bufs.state, stack,
            unconditional_post_del=enclosed_fn is not None,
            pre_stmt_suppress=pre_stmt_suppress,
        ):
            continue

        _record_simple_node_bindings_and_reads(
            curr, c_line, c_col, span, frame, bufs.state,
            unconditional_post_del=enclosed_fn is not None,
        )
        _push_assign_or_child_frames(curr, span, frame, stack)
    return enclosed_fn_multi_invoked


def _analyze_enclosing_fn_cells_uncached(
    fn_node: ast.AST,
    span: _SpanBounds,
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
    enclosed_fn: Optional[ast.AST] = None,
    mod_tree: Optional[ast.AST] = None,
    enclosing_scopes: Sequence[ast.AST] = (),
    use_stmt_deltas: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _EnclosingFnCellInfo:
    """Computes variable bindings, post-span mutations, loop mutations, and sibling closures."""
    scope_ctx = _get_fn_scope_context(fn_node, mod_tree=mod_tree, enclosing_scopes=enclosing_scopes)
    fn_globals = _extract_direct_nonlocals(fn_node, is_global=True)
    tp_names = set(_extract_node_type_param_names(fn_node)) - fn_globals
    fn_locals = set(scope_ctx.fn_locals) | tp_names
    pre_bound: Set[str] = set(tp_names)
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args) - fn_globals)

    comp_targets: Set[str] = set()
    comp_walrus: Set[str] = set()
    if isinstance(fn_node, _COMPREHENSION_TYPES):
        for gen in fn_node.generators:
            t_names = _extract_store_names_in_node(gen.target) - fn_globals
            comp_targets.update(t_names)
            t_line, t_col = _node_earliest_start_pos(gen.target)
            if _is_pos_before_span(t_line, t_col, span) or _node_contains_or_encloses_span(
                gen.target, span
            ):
                pre_bound.update(t_names)
        comp_walrus = _extract_comprehension_body_walrus_names(fn_node) - fn_globals
        fn_locals |= comp_targets

    initial_sup = _seed_scope_suppress_aliases(
        fn_node,
        mod_tree,
        scope_ctx,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    bufs = _make_empty_traversal_buffers(
        fn_globals,
        scope_ctx,
        pre_bound=pre_bound,
        unit_mutations=unit_mutations,
        initial_suppress_aliases=initial_sup,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    if isinstance(fn_node, _COMPREHENSION_TYPES):
        bufs.state.loop_mutated.update(comp_targets | comp_walrus | set(unit_mutations))
    delta_aux_nodes = tuple(n for n in scope_ctx.cache_aux_nodes() if n is not fn_node)
    stmt_deltas: Optional[Dict[Tuple[Any, ...], Optional[_StmtCellDelta]]] = (
        cast(
            Dict[Tuple[Any, ...], Optional[_StmtCellDelta]],
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                _make_trusted_scope_cache_key(
                    owner=fn_node,
                    kind="stmt_deltas",
                    aux_ast_nodes=delta_aux_nodes,
                ),
                dict,
            ),
        )
        if enclosed_fn is None
        and use_stmt_deltas
        and isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    if isinstance(fn_node, ast.Lambda):
        stack: List[_StackFrame] = (
            [_StackFrame(fn_node.body, False, False, None)]
            if isinstance(fn_node.body, ast.AST)
            else []
        )
    elif isinstance(fn_node, _COMPREHENSION_TYPES):
        comp_frames: List[Tuple[ast.AST, bool]] = []
        has_ifs = False
        for gen_idx, gen in enumerate(fn_node.generators):
            if gen_idx > 0:
                comp_frames.append((gen.iter, has_ifs))
            comp_frames.append((gen.target, has_ifs))
            for if_expr in gen.ifs:
                comp_frames.append((if_expr, True))
                has_ifs = True
        if isinstance(fn_node, ast.DictComp):
            comp_frames.append((fn_node.key, has_ifs))
            comp_frames.append((fn_node.value, has_ifs))
        else:
            comp_frames.append((fn_node.elt, has_ifs))
        stack = [
            _StackFrame(expr_node, is_br, True, None) for expr_node, is_br in reversed(comp_frames)
        ]
    else:
        body_seq = getattr(fn_node, "body", ())
        stack = [
            _StackFrame(stmt, False, False, None)
            for stmt in reversed(body_seq if isinstance(body_seq, (list, tuple)) else ())
        ]
    with collect_heuristic_ctx_managers() as fn_heuristics, collect_wildcard_ctx_managers() as fn_wildcards:
        enclosed_fn_multi_invoked = _traverse_fn_cell_frames(
            stack, span, bufs, enclosed_fn=enclosed_fn, stmt_deltas=stmt_deltas
        )

    if enclosed_fn is not None and enclosed_fn_multi_invoked:
        enc_muts, enc_dels = _extract_closure_nonlocal_mutations_and_deletions(enclosed_fn)
        bufs.state.loop_mutated.update(unit_mutations)
        bufs.state.loop_mutated.update(enc_muts)
        if enc_dels:
            bufs.state.pre_bound.difference_update(enc_dels)
            bufs.state.pre_unbound.update(enc_dels)

    if bufs.sibling_nl_deletions:
        bufs.state.pre_bound.difference_update(bufs.sibling_nl_deletions)
        bufs.state.pre_unbound.update(bufs.sibling_nl_deletions)

    _resolve_call_carrier_alias_edges(bufs)
    rebound_alias_edges: List[Tuple[str, Set[str]]] = [
        (t_name, srcs) for t_name, srcs in bufs.alias_edges if not t_name.startswith("*")
    ]
    mutation_edges: List[Tuple[str, Set[str]]] = [
        (t_name[1:], srcs) for t_name, srcs in bufs.alias_edges if t_name.startswith("*")
    ]
    propagated_closures = _propagate_sibling_closures(
        bufs.sibling_closures,
        post_reads=bufs.state.post_reads,
        alias_edges=rebound_alias_edges,
        mutation_edges=mutation_edges,
    )
    concurrent_sib_vars: Set[str] = set()
    for s_idx in bufs.concurrent_sib_indices:
        _, s_reads, s_muts = propagated_closures[s_idx]
        concurrent_sib_vars.update(s_reads)
        concurrent_sib_vars.update(s_muts)
    all_sib_nl = (
        set().union(*(nl for _, _, nl in propagated_closures)) if propagated_closures else set()
    )
    return _EnclosingFnCellInfo(
        fn_locals=frozenset(fn_locals),
        pre_bound=frozenset(bufs.state.pre_bound),
        pre_mutated=frozenset(bufs.state.pre_mutated),
        post_mutated=frozenset(bufs.state.post_mutated),
        post_reads=frozenset(bufs.state.post_reads),
        loop_mutated=frozenset(bufs.state.loop_mutated),
        sibling_closures=tuple(
            (name, frozenset(reads), frozenset(muts)) for name, reads, muts in propagated_closures
        ),
        all_sibling_nonlocal_mutations=frozenset(all_sib_nl),
        alias_edges=tuple((t_name, frozenset(srcs)) for t_name, srcs in rebound_alias_edges),
        pre_unbound=frozenset(bufs.state.pre_unbound),
        concurrent_sibling_vars=frozenset(concurrent_sib_vars),
        heuristic_ctx_managers=tuple(sorted(fn_heuristics)),
        wildcard_ctx_managers=tuple(sorted(fn_wildcards)),
        mutation_edges=tuple((t_name, frozenset(srcs)) for t_name, srcs in mutation_edges),
        receiver_external_closures=(
            _collect_enclosing_cls_external_closures(fn_node, enclosed_fn, bufs)
            if enclosed_fn is not None
            else (frozenset(), frozenset())
        ),
    )


def _collect_receiver_carrier_names(
    cls_name: str,
    bufs: _CellTraversalBuffers,
) -> Set[str]:
    """Collects cls_name, aliases, instantiated bindings, and returned aliases in bufs."""
    targets: Set[str] = {cls_name}
    factory_targets: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for t_name, srcs in bufs.alias_edges:
            if (
                t_name
                and not t_name.startswith("*")
                and bool(srcs & targets)
                and t_name not in targets
            ):
                targets.add(t_name)
                changed = True
        for ret_fn, (ret_loads, ret_calls) in bufs.fn_return_refs.items():
            if ret_fn not in factory_targets and (
                bool(ret_calls & targets)
                or bool(ret_loads & targets)
                or bool(ret_calls & factory_targets)
                or bool(ret_loads & factory_targets)
            ):
                factory_targets.add(ret_fn)
                changed = True
        for c_tgt, called in bufs.call_edges:
            if (
                c_tgt
                and not c_tgt.startswith("*")
                and (bool(called & targets) or bool(called & factory_targets))
                and c_tgt not in targets
            ):
                targets.add(c_tgt)
                changed = True
    return targets


def _collect_enclosing_cls_external_closures(
    fn_node: ast.AST,
    enclosed_fn: ast.AST,
    bufs: _CellTraversalBuffers,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Collects reads and nonlocal mutations from external closures attached to enclosed_fn's class or instances."""
    cls_node = _find_enclosing_method_class_node(fn_node, enclosed_fn)
    if cls_node is None:
        return (frozenset(), frozenset())
    ext_sibs = [
        (s_name, set(s_reads), set(s_muts))
        for idx, (s_name, s_reads, s_muts) in enumerate(bufs.sibling_closures)
        if idx not in bufs.class_body_sib_indices
    ]
    if not ext_sibs:
        return (frozenset(), frozenset())
    targets = _collect_receiver_carrier_names(cls_node.name, bufs)
    ext_rebound_edges: List[Tuple[str, Set[str]]] = []
    ext_mut_edges: List[Tuple[str, Set[str]]] = []
    for idx, (t_name, srcs) in enumerate(bufs.alias_edges):
        if idx in bufs.class_body_alias_edge_indices:
            continue
        if t_name.startswith("*"):
            ext_mut_edges.append((t_name[1:], srcs))
        else:
            ext_rebound_edges.append((t_name, srcs))
    mutated_targets = {t_name for t_name, _ in ext_mut_edges}
    if mutated_targets.isdisjoint(targets):
        return (frozenset(), frozenset())
    propagated_ext = _propagate_sibling_closures(
        ext_sibs,
        alias_edges=ext_rebound_edges,
        mutation_edges=ext_mut_edges,
    )
    ext_reads: Set[str] = set()
    ext_muts: Set[str] = set()
    for p_name, p_reads, p_muts in propagated_ext:
        if p_name is not None and p_name in targets and p_name in mutated_targets:
            ext_reads.update(p_reads)
            ext_muts.update(p_muts)
    if not ext_reads and not ext_muts:
        return (frozenset(), frozenset())
    return (frozenset(ext_reads), frozenset(ext_muts))


def _resolve_call_carrier_alias_edges(bufs: _CellTraversalBuffers) -> None:
    """Resolves multi-layer and aliased factory/carrier calls into bufs.alias_edges."""
    if not bufs.call_edges:
        return
    mutated_carrier_targets: Set[str] = {
        t[1:] for t, _ in bufs.alias_edges if t.startswith("*")
    } | {t[1:] for t, _ in bufs.call_edges if t.startswith("*")}
    closure_carrier_names: Set[str] = (
        {name for name, _, _ in bufs.sibling_closures if name is not None}
        | bufs.factory_names
        | {t.lstrip("*") for t, _ in bufs.alias_edges}
        | {t.lstrip("*") for t, _ in bufs.call_edges}
    )
    var_alias_map: Dict[str, Set[str]] = {}
    for t_name, srcs in bufs.alias_edges:
        var_alias_map.setdefault(t_name.lstrip("*"), set()).update(srcs)
    resolved_call_targets: List[Set[str]] = [set() for _ in bufs.call_edges]
    call_changed = True
    while call_changed:
        call_changed = False
        for idx, (t_name, called_set) in enumerate(bufs.call_edges):
            alias_targets = resolved_call_targets[idx]
            visited_fn_refs: Set[str] = set()
            pending_fn_refs = list(called_set)
            while pending_fn_refs:
                fn_ref = pending_fn_refs.pop()
                if fn_ref in visited_fn_refs:
                    continue
                visited_fn_refs.add(fn_ref)
                for aliased_fn in var_alias_map.get(fn_ref, ()):
                    if aliased_fn not in visited_fn_refs:
                        pending_fn_refs.append(aliased_fn)
                if fn_ref in bufs.factory_names or fn_ref in mutated_carrier_targets:
                    if fn_ref not in alias_targets:
                        alias_targets.add(fn_ref)
                        call_changed = True
                ret_loads_set, ret_calls_set = bufs.fn_return_refs.get(fn_ref, (set(), set()))
                for ret_ref in ret_loads_set:
                    if ret_ref in closure_carrier_names and ret_ref not in alias_targets:
                        alias_targets.add(ret_ref)
                        call_changed = True
                for call_ref in ret_calls_set:
                    if call_ref in bufs.factory_names and call_ref not in alias_targets:
                        alias_targets.add(call_ref)
                        call_changed = True
                    if (
                        call_ref in bufs.fn_return_refs
                        or call_ref in bufs.factory_names
                        or call_ref in var_alias_map
                    ) and call_ref not in visited_fn_refs:
                        pending_fn_refs.append(call_ref)
            if alias_targets:
                bare_tgt = t_name[1:] if t_name.startswith("*") else t_name
                closure_carrier_names.add(bare_tgt)
                tgt_aliases = var_alias_map.setdefault(bare_tgt, set())
                if not alias_targets.issubset(tgt_aliases):
                    tgt_aliases.update(alias_targets)
                    call_changed = True
    for idx, (t_name, _) in enumerate(bufs.call_edges):
        if resolved_call_targets[idx]:
            bufs.alias_edges.append((t_name, resolved_call_targets[idx]))


def _analyze_enclosing_fn_cells(
    fn_node: ast.AST,
    span: Union[_SpanBounds, Tuple[int, int, Optional[int], Optional[int]]],
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
    enclosed_fn: Optional[ast.AST] = None,
    mod_tree: Optional[ast.AST] = None,
    enclosing_scopes: Sequence[ast.AST] = (),
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _EnclosingFnCellInfo:
    """Analyzes and caches cell bindings, post-span mutations, loop mutations, and closures."""
    norm_span = span if isinstance(span, _SpanBounds) else _SpanBounds(*span)
    norm_enclosing = tuple(
        n for n in enclosing_scopes if isinstance(n, ast.AST) and n is not fn_node
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *(n for n in (enclosed_fn, mod_tree) if n is not None),
        *norm_enclosing,
    )
    extra_non_sup = frozenset(extra_non_suppressing_ctx_managers)
    params: Tuple[Any, ...] = (
        norm_span,
        frozenset(unit_mutations),
        enclosed_fn is not None,
        bool(strict_with_suppression),
        extra_non_sup,
    )
    cache_key = _make_trusted_scope_cache_key(
        owner=fn_node,
        kind="enclosing_fn_cells",
        params=params,
        aux_ast_nodes=aux_nodes,
    )
    was_computed = [False]

    def _compute_cells() -> _EnclosingFnCellInfo:
        was_computed[0] = True
        return _analyze_enclosing_fn_cells_uncached(
            fn_node,
            norm_span,
            unit_mutations=unit_mutations,
            enclosed_fn=enclosed_fn,
            mod_tree=mod_tree,
            enclosing_scopes=norm_enclosing,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_sup,
        )

    with scope_fingerprint_verification_session():
        info = cast(
            _EnclosingFnCellInfo,
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                cache_key,
                _compute_cells,
            ),
        )
    if not was_computed[0]:
        if not strict_with_suppression and info.heuristic_ctx_managers:
            for mgr_name in info.heuristic_ctx_managers:
                record_heuristic_non_suppressing_ctx_manager(mgr_name)
        if info.wildcard_ctx_managers:
            for wc_pat, mgr_name in info.wildcard_ctx_managers:
                record_wildcard_ctx_manager_match(wc_pat, mgr_name)
    return info


def _find_enclosing_method_class_node(
    fn_node: ast.AST,
    enc_child: ast.AST,
) -> Optional[ast.ClassDef]:
    """Finds the ClassDef inside fn_node that directly defines method enc_child."""
    parent_map = _get_scope_ancestor_map(fn_node)
    curr_p = parent_map.get(enc_child)
    while curr_p is not None and curr_p is not fn_node:
        if isinstance(curr_p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        if isinstance(curr_p, ast.ClassDef):
            return (
                curr_p
                if enc_child in _iter_class_body_methods(curr_p)
                else None
            )
        curr_p = parent_map.get(curr_p)
    return None


_RECEIVER_MUTATING_METHODS: frozenset[str] = frozenset(
    {
        "add",
        "append",
        "appendleft",
        "clear",
        "discard",
        "extend",
        "extendleft",
        "insert",
        "pop",
        "popitem",
        "popleft",
        "remove",
        "reverse",
        "rotate",
        "setdefault",
        "sort",
        "update",
        "__delattr__",
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__isub__",
        "__setattr__",
        "__setitem__",
    }
)
_RECEIVER_READONLY_METHODS: frozenset[str] = frozenset(
    {
        "copy",
        "count",
        "get",
        "index",
        "items",
        "keys",
        "values",
    }
)
_RECEIVER_ESCAPE_ATTRS: frozenset[str] = frozenset({"__class__", "__dict__"})
_RECEIVER_ATTR_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})
_ALIAS_READONLY_CALL_BUILTINS: frozenset[str] = frozenset(
    {
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "frozenset",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)


def _extract_receiver_alias_attrs(
    expr: ast.AST,
    rec_param: str,
    alias_to_attrs: Mapping[str, Set[str]],
) -> Set[str]:
    """Returns receiver attributes referenced directly or via a local alias in expr."""
    curr_expr: ast.AST = expr
    root_attr: Optional[str] = None
    while True:
        if isinstance(curr_expr, ast.NamedExpr):
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Attribute):
            root_attr = curr_expr.attr
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Subscript):
            curr_expr = curr_expr.value
            continue
        if (
            isinstance(curr_expr, ast.Call)
            and isinstance(curr_expr.func, ast.Attribute)
            and curr_expr.func.attr in ("get", "setdefault")
        ):
            curr_expr = curr_expr.func.value
            continue
        break
    if isinstance(curr_expr, ast.Name):
        if curr_expr.id == rec_param and root_attr is not None:
            return {root_attr}
        if curr_expr.id in alias_to_attrs:
            return set(alias_to_attrs[curr_expr.id])
    return set()


def _collect_receiver_local_aliases(
    visited_nodes: Sequence[ast.AST],
    rec_param: str,
) -> Dict[str, Set[str]]:
    """Maps local variable aliases in visited_nodes to the rec_param attributes they reference."""
    pairs: List[Tuple[str, ast.AST]] = []
    for node in visited_nodes:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id != rec_param:
                    pairs.append((tgt.id, node.value))
                elif (
                    isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(tgt.elts) == len(node.value.elts)
                    and not any(isinstance(el, ast.Starred) for el in tgt.elts)
                    and not any(isinstance(el, ast.Starred) for el in node.value.elts)
                ):
                    for t_el, v_el in zip(tgt.elts, node.value.elts):
                        if isinstance(t_el, ast.Name) and t_el.id != rec_param:
                            pairs.append((t_el.id, v_el))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and node.target.id != rec_param
        ):
            pairs.append((node.target.id, node.value))
        elif (
            isinstance(node, ast.NamedExpr)
            and isinstance(node.target, ast.Name)
            and node.target.id != rec_param
        ):
            pairs.append((node.target.id, node.value))
    alias_to_attrs: Dict[str, Set[str]] = {}
    if not pairs:
        return alias_to_attrs
    changed = True
    while changed:
        changed = False
        for tgt_name, rhs in pairs:
            resolved = _extract_receiver_alias_attrs(rhs, rec_param, alias_to_attrs)
            if resolved:
                dest = alias_to_attrs.setdefault(tgt_name, set())
                before = len(dest)
                dest.update(resolved)
                if len(dest) > before:
                    changed = True
    return alias_to_attrs


def _record_aliased_receiver_mutations(
    visited_nodes: Sequence[ast.AST],
    rec_param: str,
    alias_to_attrs: Mapping[str, Set[str]],
    stored_attrs: Set[str],
) -> None:
    """Records rec_param attribute mutations performed through local container aliases."""
    if not alias_to_attrs:
        return
    for curr in visited_nodes:
        if isinstance(curr, ast.AugAssign):
            stored_attrs.update(
                _extract_receiver_alias_attrs(curr.target, rec_param, alias_to_attrs)
            )
        elif isinstance(curr, (ast.Attribute, ast.Subscript)) and isinstance(
            curr.ctx, (ast.Store, ast.Del)
        ):
            stored_attrs.update(
                _extract_receiver_alias_attrs(curr, rec_param, alias_to_attrs)
            )
        elif isinstance(curr, ast.Call):
            is_readonly_call = False
            if isinstance(curr.func, ast.Attribute):
                if curr.func.attr in _RECEIVER_MUTATING_METHODS or (
                    (curr.args or curr.keywords)
                    and curr.func.attr not in _RECEIVER_READONLY_METHODS
                ):
                    stored_attrs.update(
                        _extract_receiver_alias_attrs(
                            curr.func.value, rec_param, alias_to_attrs
                        )
                    )
                elif curr.func.attr in _RECEIVER_READONLY_METHODS:
                    is_readonly_call = True
            elif isinstance(curr.func, ast.Name):
                if curr.func.id in ("setattr", "delattr") and curr.args:
                    stored_attrs.update(
                        _extract_receiver_alias_attrs(
                            curr.args[0], rec_param, alias_to_attrs
                        )
                    )
                elif curr.func.id in _ALIAS_READONLY_CALL_BUILTINS:
                    is_readonly_call = True
            if not is_readonly_call:
                for arg in curr.args:
                    a_expr = arg.value if isinstance(arg, ast.Starred) else arg
                    if isinstance(a_expr, ast.Name) and a_expr.id in alias_to_attrs:
                        stored_attrs.update(alias_to_attrs[a_expr.id])
                for kw in curr.keywords:
                    if isinstance(kw.value, ast.Name) and kw.value.id in alias_to_attrs:
                        stored_attrs.update(alias_to_attrs[kw.value.id])


def _extract_receiver_root_attr(expr: ast.AST, rec_param: str) -> Optional[str]:
    """Returns the attribute name on rec_param when expr is rooted at ``rec_param.<attr>``."""
    curr_expr: ast.AST = expr
    root_attr: Optional[str] = None
    while isinstance(curr_expr, (ast.Attribute, ast.Subscript)):
        if isinstance(curr_expr, ast.Attribute):
            root_attr = curr_expr.attr
        curr_expr = curr_expr.value
    if isinstance(curr_expr, ast.Name) and curr_expr.id == rec_param:
        return root_attr
    return None


def _record_receiver_call_mutation(
    curr: ast.Call,
    rec_param: str,
    stored_attrs: Set[str],
    attr_val_ids: Optional[Set[int]] = None,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> bool:
    """Records receiver attribute mutations from mutating calls; returns True on dynamic bare-receiver mutation."""
    if isinstance(curr.func, ast.Attribute):
        if (
            curr.func.attr in ("__setattr__", "__delattr__")
            and isinstance(curr.func.value, ast.Name)
            and curr.func.value.id == rec_param
        ):
            if (
                not curr.keywords
                and curr.args
                and isinstance(curr.args[0], ast.Constant)
                and isinstance(curr.args[0].value, str)
            ):
                stored_attrs.add(curr.args[0].value)
                return False
            return True
        if curr.func.attr in _RECEIVER_MUTATING_METHODS or (
            (curr.args or curr.keywords)
            and curr.func.attr not in _RECEIVER_READONLY_METHODS
        ):
            call_attr = _extract_receiver_root_attr(curr.func.value, rec_param)
            if call_attr is not None:
                stored_attrs.add(call_attr)
        return False
    if (
        isinstance(curr.func, ast.Name)
        and curr.func.id in _RECEIVER_ATTR_BUILTINS
        and curr.args
    ):
        target_arg = curr.args[0]
        if isinstance(target_arg, ast.Name) and target_arg.id == rec_param:
            if (
                curr.func.id not in shadowed_builtins
                and not curr.keywords
                and len(curr.args) >= 2
                and not any(isinstance(a, ast.Starred) for a in curr.args[:2])
                and isinstance(curr.args[1], ast.Constant)
                and isinstance(curr.args[1].value, str)
            ):
                stored_attrs.add(curr.args[1].value)
                if attr_val_ids is not None:
                    attr_val_ids.add(id(target_arg))
            else:
                return True
        else:
            call_attr = _extract_receiver_root_attr(target_arg, rec_param)
            if call_attr is not None:
                stored_attrs.add(call_attr)
    return False


def _record_receiver_container_mutations(
    curr: ast.AST,
    rec_param: str,
    loaded_attrs: Set[str],
    stored_attrs: Set[str],
    attr_val_ids: Optional[Set[int]] = None,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> bool:
    """Records rec_param root attribute mutations across AugAssign, nested targets, and mutating calls."""
    if isinstance(curr, ast.AugAssign) and isinstance(
        curr.target, (ast.Attribute, ast.Subscript)
    ):
        aug_attr = _extract_receiver_root_attr(curr.target, rec_param)
        if aug_attr is not None:
            loaded_attrs.add(aug_attr)
            stored_attrs.add(aug_attr)
        return False
    if isinstance(curr, (ast.Attribute, ast.Subscript)) and isinstance(
        curr.ctx, (ast.Store, ast.Del)
    ):
        mut_attr = _extract_receiver_root_attr(curr, rec_param)
        if mut_attr is not None:
            stored_attrs.add(mut_attr)
        return False
    if isinstance(curr, ast.Call):
        return _record_receiver_call_mutation(
            curr,
            rec_param,
            stored_attrs,
            attr_val_ids,
            shadowed_builtins=shadowed_builtins,
        )
    return False


def _has_staticmethod_decorator(node: Optional[ast.AST]) -> bool:
    """Returns True if node carries a @staticmethod decorator."""
    return any(
        (isinstance(d, ast.Name) and d.id == "staticmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
        for d in getattr(node, "decorator_list", ())
    )


def _extract_method_receiver_param(
    fn_node: ast.AST,
    *,
    is_static: bool = False,
) -> Tuple[Optional[str], bool]:
    """Returns (receiver_param_name, is_vararg_receiver) for a method or lambda."""
    if is_static:
        return None, False
    fn_args = getattr(fn_node, "args", None)
    if not isinstance(fn_args, ast.arguments):
        return None, False
    pos_args = fn_args.posonlyargs + fn_args.args
    if pos_args:
        return pos_args[0].arg, False
    if fn_args.vararg is not None:
        return fn_args.vararg.arg, True
    return None, False


def _extract_callable_receiver_usage(
    fn_node: ast.AST,
    *,
    is_static: bool = False,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[frozenset[str], frozenset[str], bool]:
    """Extracts receiver usage, treating any access to a vararg receiver as a bare escape."""
    rec_param, is_vararg = _extract_method_receiver_param(fn_node, is_static=is_static)
    if rec_param is None:
        return frozenset(), frozenset(), False
    loaded, stored, bare_escape = _extract_receiver_usage_in_method(
        fn_node, rec_param, shadowed_builtins=shadowed_builtins
    )
    if is_vararg and (loaded or stored or bare_escape):
        return frozenset(), frozenset(), True
    return loaded, stored, bare_escape


def _extract_receiver_usage_in_method(
    method_node: ast.AST,
    rec_param: str,
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[frozenset[str], frozenset[str], bool]:
    """Returns (loaded_attrs, stored_attrs, bare_escape) for rec_param inside method_node."""
    loaded_attrs: Set[str] = set()
    stored_attrs: Set[str] = set()
    bare_escape = False
    attr_val_ids: Set[int] = set()
    visited_nodes: List[ast.AST] = []
    eff_shadowed: Set[str] = set(shadowed_builtins) & _RECEIVER_ATTR_BUILTINS
    m_args = getattr(method_node, "args", None)
    if isinstance(m_args, ast.arguments):
        eff_shadowed.update(_extract_arg_names(m_args) & _RECEIVER_ATTR_BUILTINS)
    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        eff_shadowed.update(
            _extract_single_node_local_names(method_node, include_globals=True)
            & _RECEIVER_ATTR_BUILTINS
        )
    raw_body = getattr(method_node, "body", ())
    if isinstance(raw_body, ast.AST):
        stack: List[ast.AST] = [raw_body]
    elif isinstance(raw_body, (list, tuple)):
        stack = list(raw_body)
    else:
        stack = []
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            sub_args = getattr(curr, "args", None)
            if isinstance(sub_args, ast.arguments):
                sub_arg_names = _extract_arg_names(sub_args)
                if rec_param in sub_arg_names:
                    stack.extend(iter_nested_scope_header_nodes(curr))
                    continue
                eff_shadowed.update(sub_arg_names & _RECEIVER_ATTR_BUILTINS)
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sub_locals = _extract_single_node_local_names(curr, include_globals=True)
                if rec_param in sub_locals:
                    stack.extend(iter_nested_scope_header_nodes(curr))
                    continue
                eff_shadowed.update(sub_locals & _RECEIVER_ATTR_BUILTINS)
        elif isinstance(curr, _COMPREHENSION_TYPES):
            comp_targets = {
                n
                for gen in curr.generators
                for n in _extract_store_names_in_node(gen.target)
            }
            if rec_param in comp_targets:
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
        visited_nodes.append(curr)
        if _record_receiver_container_mutations(
            curr,
            rec_param,
            loaded_attrs,
            stored_attrs,
            attr_val_ids,
            shadowed_builtins=eff_shadowed,
        ):
            bare_escape = True
        if (
            isinstance(curr, ast.Attribute)
            and isinstance(curr.value, ast.Name)
            and curr.value.id == rec_param
        ):
            attr_val_ids.add(id(curr.value))
            if isinstance(curr.ctx, ast.Load):
                loaded_attrs.add(curr.attr)
            else:
                stored_attrs.add(curr.attr)
        elif (
            isinstance(curr, ast.Name)
            and curr.id == rec_param
            and isinstance(curr.ctx, ast.Load)
            and id(curr) not in attr_val_ids
        ):
            bare_escape = True
        stack.extend(reversed(list(ast.iter_child_nodes(curr))))
    if (loaded_attrs & _RECEIVER_ESCAPE_ATTRS) or (
        stored_attrs & _RECEIVER_ESCAPE_ATTRS
    ):
        bare_escape = True
    if loaded_attrs and not bare_escape:
        alias_to_attrs = _collect_receiver_local_aliases(visited_nodes, rec_param)
        _record_aliased_receiver_mutations(
            visited_nodes, rec_param, alias_to_attrs, stored_attrs
        )
    return frozenset(loaded_attrs), frozenset(stored_attrs), bare_escape


class _ClassAttrClosureEntry(NamedTuple):
    attr_names: frozenset[str]
    reads: frozenset[str]
    muts: frozenset[str]
    rec_loaded_attrs: frozenset[str]
    rec_stored_attrs: frozenset[str]
    rec_bare_escape: bool
    referenced_names: frozenset[str] = frozenset()


def _collect_class_attr_closure_entries(
    cls_node: ast.ClassDef,
    info_sibling_closures: Sequence[Tuple[Optional[str], frozenset[str], frozenset[str]]],
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> List[_ClassAttrClosureEntry]:
    """Collects non-method class-attribute closure entries and lambda receiver usages."""
    info_by_name: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for s_name, s_reads, s_muts in info_sibling_closures:
        if s_name is not None:
            r_set, m_set = info_by_name.setdefault(s_name, (set(), set()))
            r_set.update(s_reads)
            m_set.update(s_muts)
    entries: List[_ClassAttrClosureEntry] = []
    for attr_stmt in _iter_class_body_non_method_stmts(cls_node):
        for _is_reb, c_targets, c_values in _iter_carrier_target_entries(attr_stmt):
            c_calls: Set[str] = set()
            c_reads, c_muts, c_loads = _extract_carrier_value_closure_and_load_info(
                c_values, called_names_out=c_calls
            )
            c_refs = c_loads | c_calls
            for ref in c_refs:
                if ref in info_by_name:
                    c_reads.update(info_by_name[ref][0])
                    c_muts.update(info_by_name[ref][1])
            l_load: Set[str] = set()
            l_store: Set[str] = set()
            l_bare = False
            for val in c_values:
                for sub in ast.walk(val):
                    if isinstance(sub, ast.Lambda):
                        ll, ls, lb = _extract_callable_receiver_usage(
                            sub, shadowed_builtins=shadowed_builtins
                        )
                        l_load.update(ll)
                        l_store.update(ls)
                        l_bare = l_bare or lb
            entries.append(
                _ClassAttrClosureEntry(
                    frozenset(c_targets),
                    frozenset(c_reads),
                    frozenset(c_muts),
                    frozenset(l_load),
                    frozenset(l_store),
                    l_bare,
                    frozenset(c_refs),
                )
            )
    return entries


def _has_non_plain_method_decorator(node: ast.AST) -> bool:
    """Returns True if node carries any decorator other than @staticmethod or @classmethod."""
    for dec in getattr(node, "decorator_list", ()):
        if (
            isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod")
        ) or (
            isinstance(dec, ast.Attribute)
            and dec.attr in ("staticmethod", "classmethod")
        ):
            continue
        return True
    return False


def _method_is_reachable_via_receiver(
    method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    m_store: frozenset[str],
    loaded_attrs: AbstractSet[str],
    stored_attrs: AbstractSet[str],
) -> bool:
    """Returns True when receiver attribute reads or writes can invoke or share state with method_node."""
    if loaded_attrs and (
        method_node.name in loaded_attrs
        or method_node.name in ("__getattribute__", "__getattr__")
        or bool(m_store & loaded_attrs)
    ):
        return True
    if not stored_attrs:
        return False
    if method_node.name in ("__setattr__", "__delattr__"):
        return True
    if method_node.name in stored_attrs and _has_non_plain_method_decorator(method_node):
        return True
    return any(
        isinstance(dec, ast.Attribute)
        and dec.attr in ("setter", "deleter")
        and isinstance(dec.value, ast.Name)
        and dec.value.id in stored_attrs
        for dec in method_node.decorator_list
    )


def _collect_method_receiver_sibling_closures(
    fn_node: ast.AST,
    enc_child: ast.AST,
    rec_param: str,
    info_sibling_closures: Sequence[Tuple[Optional[str], frozenset[str], frozenset[str]]],
    receiver_external_closures: Tuple[frozenset[str], frozenset[str]] = (
        frozenset(),
        frozenset(),
    ),
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[Tuple[str, frozenset[str], frozenset[str]], ...]:
    """Computes sibling closure reads and mutations reachable via rec_param on enc_child's class."""
    loaded_attrs, stored_attrs, bare_escape = _extract_receiver_usage_in_method(
        enc_child, rec_param, shadowed_builtins=shadowed_builtins
    )
    if not loaded_attrs and not stored_attrs and not bare_escape:
        return ()
    cls_node = _find_enclosing_method_class_node(fn_node, enc_child)
    if cls_node is None:
        return ()
    reads: Set[str] = set(receiver_external_closures[0])
    muts: Set[str] = set(receiver_external_closures[1])
    if cls_node.bases or cls_node.keywords:
        for s_name, s_reads, s_muts in info_sibling_closures:
            if s_name == cls_node.name:
                reads.update(s_reads)
                muts.update(s_muts)
        if not reads and not muts:
            return ()
        return ((rec_param, frozenset(reads), frozenset(muts)),)
    sib_methods = [m for m in _iter_class_body_methods(cls_node) if m is not enc_child]
    method_usages: Dict[int, Tuple[frozenset[str], frozenset[str], bool]] = {}
    for m in sib_methods:
        is_stat = _has_staticmethod_decorator(m)
        method_usages[id(m)] = _extract_callable_receiver_usage(
            m, is_static=is_stat, shadowed_builtins=shadowed_builtins
        )
    attr_entries = _collect_class_attr_closure_entries(
        cls_node, info_sibling_closures, shadowed_builtins=shadowed_builtins
    )
    if any(m_bare for _ml, _ms, m_bare in method_usages.values()) or any(
        entry.rec_bare_escape for entry in attr_entries
    ):
        bare_escape = True
    reachable_attrs: Set[str] = set(loaded_attrs)
    reachable_stored_attrs: Set[str] = set(stored_attrs)
    reachable_method_ids: Set[int] = set()
    reachable_attr_idxs: Set[int] = set()
    if not bare_escape:
        changed = True
        while changed:
            changed = False
            for m in sib_methods:
                mid = id(m)
                if mid in reachable_method_ids:
                    continue
                m_load, m_store, m_bare = method_usages[mid]
                if _method_is_reachable_via_receiver(
                    m, m_store, reachable_attrs, reachable_stored_attrs
                ):
                    reachable_method_ids.add(mid)
                    changed = True
                    if m_bare:
                        bare_escape = True
                        break
                    reachable_attrs.update(m_load)
                    reachable_stored_attrs.update(m_store)
            if bare_escape:
                break
            for idx, entry in enumerate(attr_entries):
                if idx in reachable_attr_idxs:
                    continue
                if (
                    bool(entry.attr_names & reachable_attrs)
                    or bool(entry.attr_names & reachable_stored_attrs)
                    or bool(entry.rec_stored_attrs & reachable_attrs)
                ):
                    reachable_attr_idxs.add(idx)
                    changed = True
                    if entry.rec_bare_escape:
                        bare_escape = True
                        break
                    reachable_attrs.update(entry.rec_loaded_attrs | entry.referenced_names)
                    reachable_stored_attrs.update(entry.rec_stored_attrs)
            if bare_escape:
                break
    active_methods = (
        sib_methods
        if bare_escape
        else [m for m in sib_methods if id(m) in reachable_method_ids]
    )
    active_attrs = (
        attr_entries
        if bare_escape
        else [attr_entries[i] for i in reachable_attr_idxs]
    )
    for m in active_methods:
        reads.update(_extract_cached_scope_free_reads(m))
        m_muts, m_dels = _extract_closure_nonlocal_mutations_and_deletions(m)
        muts.update(m_muts | m_dels)
    for entry in active_attrs:
        reads.update(entry.reads)
        muts.update(entry.muts)
    if not reads and not muts:
        return ()
    return ((rec_param, frozenset(reads), frozenset(muts)),)


def _attach_receiver_closures_to_info(
    child_info: _EnclosingFnCellInfo,
    rec_param: str,
    extra_sibs: Sequence[Tuple[str, frozenset[str], frozenset[str]]],
) -> _EnclosingFnCellInfo:
    """Attaches receiver sibling closures and post-reads to an inner enclosing cell summary."""
    extra_reads = (
        set().union(*(r for _, r, _ in extra_sibs))
        if rec_param in child_info.post_reads
        else set()
    )
    return child_info._replace(
        post_reads=frozenset(set(child_info.post_reads) | extra_reads),
        sibling_closures=(*child_info.sibling_closures, *extra_sibs),
        all_sibling_nonlocal_mutations=frozenset(
            set(child_info.all_sibling_nonlocal_mutations)
            | set().union(*(m for _, _, m in extra_sibs))
        ),
    )


def _collect_propagated_prefix_closures(
    chain_infos: Sequence[_EnclosingFnCellInfo],
    upto_idx: int,
) -> Tuple[Tuple[Optional[str], frozenset[str], frozenset[str]], ...]:
    """Propagates unshadowed sibling closures and alias/mutation edges across chain_infos[:upto_idx + 1]."""
    sibs: List[Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]] = []
    edges: List[Tuple[str, AbstractSet[str]]] = []
    mut_edges: List[Tuple[str, AbstractSet[str]]] = []
    for j in range(upto_idx + 1):
        sub_info = chain_infos[j]
        inner_shadowed: Set[str] = (
            set().union(
                *(chain_infos[k].fn_locals for k in range(j + 1, upto_idx + 1))
            )
            if j < upto_idx
            else set()
        )
        for s_name, s_reads, s_muts in sub_info.sibling_closures:
            eff_name = (
                None if (s_name is not None and s_name in inner_shadowed) else s_name
            )
            sibs.append((eff_name, s_reads - inner_shadowed, s_muts - inner_shadowed))
        for t_name, srcs in sub_info.alias_edges:
            if t_name not in inner_shadowed:
                edges.append((t_name, srcs - inner_shadowed))
        for t_name, srcs in sub_info.mutation_edges:
            if t_name not in inner_shadowed:
                mut_edges.append((t_name, srcs - inner_shadowed))
    propagated = _propagate_sibling_closures(
        sibs, alias_edges=edges, mutation_edges=mut_edges
    )
    return tuple((n, frozenset(r), frozenset(m)) for n, r, m in propagated)


def _resolve_enclosing_receiver_shadowed_builtins(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST],
    enclosing_scopes: Sequence[ast.AST],
) -> frozenset[str]:
    """Returns setattr/delattr names shadowed in fn_node or its enclosing scopes."""
    enc_scope_ctx = _get_fn_scope_context(
        fn_node, mod_tree=mod_tree, enclosing_scopes=enclosing_scopes
    )
    if enc_scope_ctx.has_star_import:
        return _RECEIVER_ATTR_BUILTINS
    return frozenset(enc_scope_ctx.shadowed_names & _RECEIVER_ATTR_BUILTINS)


def _build_enclosing_chain_cell_infos(
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int],
    u_end_col: Optional[int],
    unit_mutations: AbstractSet[str],
    mod_tree: Optional[ast.AST] = None,
    *,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> List[_EnclosingFnCellInfo]:
    """Builds _EnclosingFnCellInfo summaries for each function in enclosing_chain."""
    rev_infos: List[_EnclosingFnCellInfo] = []
    escaping_muts: Set[str] = set(unit_mutations)
    cross_level_receivers: List[Tuple[int, ast.AST, ast.AST, str]] = []
    for idx in range(len(enclosing_chain) - 1, -1, -1):
        _estart, _nend, fn_node = enclosing_chain[idx]
        if idx == len(enclosing_chain) - 1:
            span = _SpanBounds(u_start, u_end, u_start_col, u_end_col)
            enc_child: Optional[ast.AST] = None
        else:
            next_start, next_end, next_fn = enclosing_chain[idx + 1]
            next_body = getattr(next_fn, "body", None)
            first_stmt: ast.AST = (
                next_body[0]
                if isinstance(next_body, (list, tuple)) and next_body
                else next_body if isinstance(next_body, ast.AST) else next_fn
            )
            f_line, f_col = _node_earliest_start_pos(first_stmt)
            span = _SpanBounds(
                int(f_line or next_start),
                next_end,
                f_col,
                getattr(next_fn, "end_col_offset", None),
            )
            enc_child = next_fn
        info = _analyze_enclosing_fn_cells(
            fn_node,
            span,
            unit_mutations=escaping_muts,
            enclosed_fn=enc_child,
            mod_tree=mod_tree,
            enclosing_scopes=tuple(n for _, _, n in enclosing_chain[:idx]),
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        is_static_enc = _has_staticmethod_decorator(enc_child)
        if enc_child is not None and not is_static_enc and rev_infos:
            rec_param, _ = _extract_method_receiver_param(enc_child)
            if rec_param is not None:
                enc_shadowed = _resolve_enclosing_receiver_shadowed_builtins(
                    fn_node,
                    mod_tree,
                    tuple(n for _, _, n in enclosing_chain[:idx]),
                )
                extra_sibs = _collect_method_receiver_sibling_closures(
                    fn_node,
                    enc_child,
                    rec_param,
                    info.sibling_closures,
                    info.receiver_external_closures,
                    shadowed_builtins=enc_shadowed,
                )
                if extra_sibs:
                    rev_infos[-1] = _attach_receiver_closures_to_info(
                        rev_infos[-1], rec_param, extra_sibs
                    )
                elif idx > 0:
                    cls_node = _find_enclosing_method_class_node(fn_node, enc_child)
                    if cls_node is not None and (cls_node.bases or cls_node.keywords):
                        cross_level_receivers.append(
                            (idx, fn_node, enc_child, rec_param)
                        )
        rev_infos.append(info)
        own_bound_locals = (
            _extract_single_node_local_names(fn_node)
            if isinstance(fn_node, _COMPREHENSION_TYPES)
            else info.fn_locals
        )
        escaping_muts = escaping_muts - own_bound_locals
    rev_infos.reverse()
    for idx, fn_node, enc_child, rec_param in cross_level_receivers:
        chain_sibs = _collect_propagated_prefix_closures(rev_infos, idx)
        enc_shadowed = _resolve_enclosing_receiver_shadowed_builtins(
            fn_node,
            mod_tree,
            tuple(n for _, _, n in enclosing_chain[:idx]),
        )
        extra_sibs = _collect_method_receiver_sibling_closures(
            fn_node,
            enc_child,
            rec_param,
            chain_sibs,
            rev_infos[idx].receiver_external_closures,
            shadowed_builtins=enc_shadowed,
        )
        if extra_sibs:
            rev_infos[idx + 1] = _attach_receiver_closures_to_info(
                rev_infos[idx + 1], rec_param, extra_sibs
            )
    return rev_infos


def _find_enclosing_chain_definite_pre_bound(
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
    mod_tree: Optional[ast.AST] = None,
    *,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Computes variables must-bound (definitely assigned) on all paths prior to the unit.

    Complements ``_find_enclosing_pre_bound_names`` (which computes may-bound names
    across module, class, and function scopes) by performing branch-sensitive definite
    assignment analysis across the enclosing function/comprehension chain.
    """
    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        u_start,
        u_end,
        u_start_col,
        u_end_col,
        frozenset(),
        mod_tree=mod_tree,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    acc_pre_bound: Set[str] = set()
    acc_pre_unbound: Set[str] = set()
    for idx in range(len(chain_infos) - 1, -1, -1):
        info = chain_infos[idx]
        acc_pre_bound.update(info.pre_bound - acc_pre_unbound)
        acc_pre_unbound.update((set(info.pre_unbound) | info.fn_locals) - info.pre_bound)
    return acc_pre_bound
