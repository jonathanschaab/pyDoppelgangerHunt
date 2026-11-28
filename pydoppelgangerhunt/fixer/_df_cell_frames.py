"""Compound-statement traversal frames and binding-state helpers for enclosing function cells."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
)
from pydoppelgangerhunt.fixer._df_block_flow import (
    _extract_block_definite_bindings,
    _merge_loop_carried_suppress_provenance,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import _extract_store_names_in_node
from pydoppelgangerhunt.fixer._df_scope_ctx import (
    _SpanBounds,
    _block_contains_span,
    _can_bind_in_fn_scope,
    _is_node_inside_span,
    _is_pos_after_span,
    _is_pos_before_span,
    _node_contains_or_encloses_span,
    _strictly_encloses_span,
)
from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_pattern_bound_names,
    _is_compound_cfg_stmt,
    _locate_match_span,
    _locate_try_span,
)
from pydoppelgangerhunt.fixer._df_prov_exprs import (
    _advance_prov_through_for_header,
    _advance_prov_through_opt_expr,
)
from pydoppelgangerhunt.fixer._df_provenance_walk import (
    _iter_with_item_provenance_steps,
    _update_prov_and_term_from_stmt,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _SuppressProvenance,
    _coerce_suppress_provenance,
)

if TYPE_CHECKING:
    from pydoppelgangerhunt.fixer._df_cell_traversal import _CellTraversalBuffers

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


_BoundSnapshot = Tuple[Set[str], Set[str]]


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
    save_bound_box: Optional[List[Optional[_BoundSnapshot]]] = None
    restore_bound_box: Optional[List[Optional[_BoundSnapshot]]] = None
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
    walk_res = _update_prov_and_term_from_stmt(
        stmt,
        pre_prov,
        strict_with_suppression=bufs.state.strict_with_suppression,
        extra_non_suppressing_ctx_managers=bufs.state.extra_non_suppressing_ctx_managers,
    )
    if is_branched:
        return pre_prov.merge_branch_conservative(
            walk_res.prov
        ).merge_branch_conservative(walk_res.mid_prov)
    return walk_res.prov


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


def _push_branch_checkpoint_frame(
    stack: List[_StackFrame],
    curr: ast.AST,
    frame: _StackFrame,
    suppress_box: Optional[List[Optional[_SuppressProvenance]]] = None,
    bound_box: Optional[List[Optional[_BoundSnapshot]]] = None,
    *,
    save: bool = False,
    is_branched: Optional[bool] = None,
) -> None:
    """Pushes a checkpoint to snapshot or restore suppress_aliases and/or pre_bound across mutually exclusive arms."""
    if suppress_box is None and bound_box is None:
        return
    eff_br = frame.is_branched if is_branched is None else is_branched
    stack.append(
        _StackFrame(
            curr,
            eff_br,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
            save_suppress_box=suppress_box if save else None,
            restore_suppress_box=None if save else suppress_box,
            save_bound_box=bound_box if save else None,
            restore_bound_box=None if save else bound_box,
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
    return _is_compound_cfg_stmt(node) or isinstance(node, ast.ClassDef)


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
    body_l, else_l, final_l, handlers_l = (
        getattr(curr, "body", []),
        getattr(curr, "orelse", []),
        getattr(curr, "finalbody", []),
        getattr(curr, "handlers", []),
    )
    body_span, else_span, final_span, target_h_idx = _locate_try_span(
        curr,
        encloses,
        lambda blk: _block_contains_span(blk, span),
        lambda node: _node_contains_or_encloses_span(node, span),
    )
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
    is_br, in_lp, act_nl, f_after = (
        frame.is_branched,
        frame.in_loop,
        frame.active_nonlocals,
        frame.force_after_span,
    )
    saved_after_body_box: Optional[List[Optional[_SuppressProvenance]]] = None
    saved_bound_after_body_box: Optional[List[Optional[_BoundSnapshot]]] = None
    isolate_else_sup = (
        state.suppress_aliases is not None and else_span and bool(handlers_l)
    )
    isolate_else_bounds = not in_lp and else_span and bool(handlers_l)
    is_exclusive_later_handler = (
        isinstance(curr, ast.Try) and target_h_idx is not None and target_h_idx > 0
    )
    isolate_handler_sup = (
        state.suppress_aliases is not None and is_exclusive_later_handler
    )
    isolate_handler_bounds = not in_lp and is_exclusive_later_handler
    if isolate_else_sup or isolate_handler_sup:
        saved_after_body_box = [None]
    if isolate_else_bounds or isolate_handler_bounds:
        saved_bound_after_body_box = [None]
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
    _push_branch_checkpoint_frame(
        stack,
        curr,
        frame,
        suppress_box=saved_after_body_box if isolate_else_sup else None,
        bound_box=saved_bound_after_body_box if isolate_else_bounds else None,
    )
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
        if h_idx == target_h_idx and (isolate_handler_sup or isolate_handler_bounds):
            if isolate_handler_sup:
                for prior_idx in range(target_h_idx - 1, -1, -1):
                    prior_type = getattr(handlers_l[prior_idx], "type", None)
                    if isinstance(prior_type, ast.AST):
                        _push_suppress_header_frame(
                            stack,
                            curr,
                            frame,
                            prior_type,
                            span,
                            is_branched=True,
                        )
            _push_branch_checkpoint_frame(
                stack,
                curr,
                frame,
                suppress_box=saved_after_body_box if isolate_handler_sup else None,
                bound_box=saved_bound_after_body_box if isolate_handler_bounds else None,
            )
    _push_branch_checkpoint_frame(
        stack,
        curr,
        frame,
        suppress_box=saved_after_body_box,
        bound_box=saved_bound_after_body_box,
        save=True,
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
    cases_l = getattr(curr, "cases", [])
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
    span_case_idx, span_in_case_body = _locate_match_span(
        curr,
        encloses,
        lambda blk: _block_contains_span(blk, span),
        lambda node: _node_contains_or_encloses_span(node, span),
    )
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    for idx in range(len(cases_l) - 1, -1, -1):
        case = cases_l[idx]
        case_body = getattr(case, "body", [])
        c_br = is_br if (idx == span_case_idx and span_in_case_body) else True
        isolate_case_sup = (
            state.suppress_aliases is not None
            and bool(case_body)
            and (
                (span_case_idx is not None and idx < span_case_idx)
                or (not encloses and idx < len(cases_l) - 1)
            )
        )
        isolate_case_bounds = (
            not in_lp
            and bool(case_body)
            and span_case_idx is not None
            and idx < span_case_idx
        )
        saved_case_box: Optional[List[Optional[_SuppressProvenance]]] = (
            [None] if isolate_case_sup else None
        )
        saved_bound_case_box: Optional[List[Optional[_BoundSnapshot]]] = (
            [None] if isolate_case_bounds else None
        )
        _push_branch_checkpoint_frame(
            stack,
            curr,
            frame,
            suppress_box=saved_case_box,
            bound_box=saved_bound_case_box,
            is_branched=c_br,
        )
        _push_seq(
            stack,
            case_body,
            c_br,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        _push_branch_checkpoint_frame(
            stack,
            curr,
            frame,
            suppress_box=saved_case_box,
            bound_box=saved_bound_case_box,
            save=True,
            is_branched=c_br,
        )
        if state.suppress_aliases is not None and (
            not encloses
            or (
                span_case_idx is not None
                and (idx < span_case_idx or (idx == span_case_idx and span_in_case_body))
            )
        ):
            pattern = getattr(case, "pattern", None)
            pat_names = frozenset(_extract_pattern_bound_names(pattern))
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
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    isolate_body_sup = (
        state.suppress_aliases is not None
        and bool(curr.body)
        and (else_span or (not encloses and bool(curr.orelse)))
    )
    isolate_body_bounds = not in_lp and else_span and bool(curr.body)
    saved_if_box: Optional[List[Optional[_SuppressProvenance]]] = (
        [None] if isolate_body_sup else None
    )
    saved_if_bound_box: Optional[List[Optional[_BoundSnapshot]]] = (
        [None] if isolate_body_bounds else None
    )
    _push_seq(
        stack,
        curr.orelse,
        is_br if else_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    _push_branch_checkpoint_frame(
        stack, curr, frame, suppress_box=saved_if_box, bound_box=saved_if_bound_box
    )
    _push_seq(
        stack,
        curr.body,
        is_br if body_span else True,
        in_lp,
        act_nl,
        force_after_span=f_after,
    )
    _push_branch_checkpoint_frame(
        stack,
        curr,
        frame,
        suppress_box=saved_if_box,
        bound_box=saved_if_bound_box,
        save=True,
    )
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


# INVARIANT (SHARED CFG LOWERING CONTRACT — POSITION-AWARE CELL FRAME STACK):
# This explicit frame-stack dispatcher tracks pre-span, in-span, and post-span
# cell bindings/mutations relative to a target unit span (which cannot be expressed
# as a pure whole-block transfer). Compound statement membership and span location
# MUST stay aligned with `_df_cfg` (`_is_compound_cfg_stmt`, `_locate_try_span`,
# `_locate_match_span`, `_extract_pattern_bound_names`) and `_df_block_flow`
# (`_extract_block_definite_bindings`, `_merge_loop_carried_suppress_provenance`).
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
