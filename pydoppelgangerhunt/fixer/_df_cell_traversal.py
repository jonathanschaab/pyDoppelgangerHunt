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
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _downstream_cache_lock,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
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
    _record_import_names,
)
from pydoppelgangerhunt.fixer._df_with_provenance import (
    _SuppressProvenance,
    _clear_store_target_provenance,
    _coerce_suppress_provenance,
    _extract_block_definite_bindings,
    _extract_shallow_deleted_names,
    _is_non_raising_store_target,
    _seed_scope_suppress_aliases,
    _update_suppress_provenance_from_expr_walrus,
    _update_suppress_provenance_from_stmt,
    _with_item_may_suppress_exceptions,
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
    "_find_enclosing_method_class_names",
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
        n
        for n in names
        if _can_bind_in_fn_scope(n, frame.active_nonlocals, state.fn_globals)
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
    if not _can_bind_in_fn_scope(
        name, frame.active_nonlocals, state.fn_globals
    ):
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
    pre_stmt_suppress: Optional[
        Union[_SuppressProvenance, AbstractSet[str]]
    ] = None,
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
        active_sup = (
            pre_stmt_suppress
            if pre_stmt_suppress is not None
            else state.suppress_aliases
        )
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


def _push_compound_children(
    curr: ast.AST,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
    *,
    unconditional_post_del: bool = False,
    pre_stmt_suppress: Optional[
        Union[_SuppressProvenance, AbstractSet[str]]
    ] = None,
) -> bool:
    """Pushes child nodes of compound control-flow statements with branch and loop state."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    f_after = frame.force_after_span
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
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
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
        stack.append(
            _StackFrame(
                curr.test, is_br, in_lp, act_nl, force_after_span=f_after
            )
        )
        return True

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
        stack.append(
            _StackFrame(
                curr.target,
                is_br if body_span else True,
                True if (body_span or iter_span) else in_lp,
                act_nl,
                force_after_span=after_iter,
            )
        )
        stack.append(
            _StackFrame(
                curr.iter, is_br, in_lp, act_nl, force_after_span=f_after
            )
        )
        return True

    if isinstance(curr, ast.While):
        body_span = encloses and (
            _block_contains_span(curr.body, span)
            or _is_node_inside_span(curr.test, span)
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
        stack.append(
            _StackFrame(
                curr.test,
                is_br,
                True if body_span else in_lp,
                act_nl,
                force_after_span=f_after,
            )
        )
        return True

    if isinstance(curr, (ast.With, ast.AsyncWith)):
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
        active_sup = _coerce_suppress_provenance(
            pre_stmt_suppress
            if pre_stmt_suppress is not None
            else state.suppress_aliases
        )
        item_prov = active_sup
        after_sup = False
        item_frames: List[_StackFrame] = []
        for item in curr.items:
            item_br = is_br or (after_sup and not encloses)
            w_prov = _update_suppress_provenance_from_expr_walrus(
                (item.context_expr,),
                item_prov,
                allow_positive=not item_br,
            )
            item_prov = (
                item_prov.merge_branch_conservative(w_prov)
                if item_br
                else w_prov
            )
            item_may_sup = _with_item_may_suppress_exceptions(
                item.context_expr,
                item_prov,
                strict_with_suppression=state.strict_with_suppression,
                extra_non_suppressing_ctx_managers=state.extra_non_suppressing_ctx_managers,
            )
            if (
                not encloses
                and not item_br
                and item_may_sup
                and item.optional_vars is not None
                and not _is_non_raising_store_target(item.optional_vars)
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
            if item.optional_vars is not None:
                opt_cond = item_br or (
                    item_may_sup
                    and not _is_non_raising_store_target(item.optional_vars)
                )
                item_prov = _clear_store_target_provenance(
                    item.optional_vars, item_prov, is_conditional=opt_cond
                )
            if item_may_sup:
                after_sup = True
        stack.extend(reversed(item_frames))
        return True

    if isinstance(curr, ast.Try) or type(curr).__name__ == "TryStar":
        body_l = getattr(curr, "body", [])
        else_l = getattr(curr, "orelse", [])
        final_l = getattr(curr, "finalbody", [])
        handlers_l = getattr(curr, "handlers", [])
        else_span = encloses and _block_contains_span(else_l, span)
        body_span = encloses and (
            _block_contains_span(body_l, span) or else_span
        )
        final_span = encloses and _block_contains_span(final_l, span)
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
        for h_node in reversed(handlers_l):
            h_span = encloses and _block_contains_span(
                getattr(h_node, "body", ()), span
            )
            stack.append(
                _StackFrame(
                    h_node,
                    is_br if h_span else True,
                    in_lp,
                    act_nl,
                    force_after_span=f_after,
                )
            )
        _push_seq(
            stack,
            body_l,
            is_br if body_span else True,
            in_lp,
            act_nl,
            force_after_span=f_after,
        )
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
        _push_seq(
            stack, curr.body, h_br, in_lp, act_nl, force_after_span=f_after
        )
        if curr.type is not None:
            stack.append(
                _StackFrame(
                    curr.type, True, in_lp, act_nl, force_after_span=f_after
                )
            )
        return True

    if type(curr).__name__ == "Match":
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
        for case in reversed(cases_l):
            case_body = getattr(case, "body", [])
            c_br = (
                is_br
                if (encloses and _block_contains_span(case_body, span))
                else True
            )
            _push_seq(
                stack,
                case_body,
                c_br,
                in_lp,
                act_nl,
                force_after_span=f_after,
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
            stack.append(
                _StackFrame(
                    subject, is_br, in_lp, act_nl, force_after_span=f_after
                )
            )
        return True

    if isinstance(curr, (ast.BoolOp, ast.Compare)):
        uncond_exprs: Sequence[ast.AST] = (
            curr.values[:1]
            if isinstance(curr, ast.BoolOp)
            else (curr.left, *curr.comparators[:1])
        )
        short_circuit_exprs: Sequence[ast.AST] = (
            curr.values[1:]
            if isinstance(curr, ast.BoolOp)
            else curr.comparators[1:]
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
        stack.append(
            _StackFrame(
                curr.orelse, True, in_lp, act_nl, force_after_span=f_after
            )
        )
        stack.append(
            _StackFrame(
                curr.body, True, in_lp, act_nl, force_after_span=f_after
            )
        )
        stack.append(
            _StackFrame(
                curr.test, is_br, in_lp, act_nl, force_after_span=f_after
            )
        )
        return True

    return False


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
    state = bufs.state
    sibling_closures = bufs.sibling_closures
    scope_ctx = bufs.scope_ctx
    factory_names = bufs.factory_names
    fn_return_refs = bufs.fn_return_refs
    alias_edges = bufs.alias_edges
    call_edges = bufs.call_edges
    sibling_nl_deletions = bufs.sibling_nl_deletions
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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
            if _can_bind_in_fn_scope(
                curr.name, frame.active_nonlocals, state.fn_globals
            ):
                state.post_mutated.add(curr.name)
                if frame.in_loop:
                    state.loop_mutated.add(curr.name)
        else:
            _record_binding_at_pos(
                (curr.name,), estart, ecol, span, frame, state
            )
        if enclosing_cls and isinstance(curr, ast.ClassDef):
            for stmt in _iter_class_body_methods(curr):
                if not (
                    _is_node_inside_span(stmt, span)
                    or _strictly_encloses_span(stmt, span)
                ):
                    s_reads = _extract_cached_scope_free_reads(stmt)
                    s_muts, s_dels = (
                        _extract_closure_nonlocal_mutations_and_deletions(stmt)
                    )
                    if sibling_nl_deletions is not None and s_dels:
                        sibling_nl_deletions.update(s_dels)
                    s_carriers = _extract_fn_outer_carrier_targets(
                        stmt, state.fn_globals, scope_ctx
                    )
                    if s_reads or s_muts:
                        sibling_closures.append(
                            (curr.name, set(s_reads), set(s_muts))
                        )
                        for oc_name in s_carriers:
                            sibling_closures.append(
                                (oc_name, set(s_reads), set(s_muts))
                            )
        elif not enclosing_cls:
            free_r = (
                _extract_nested_scope_free_reads(curr, include_header=False)
                if span_in_hdr
                else _extract_cached_scope_free_reads(curr)
            )
            nl_mut, nl_del = _extract_closure_nonlocal_mutations_and_deletions(
                curr
            )
            if sibling_nl_deletions is not None and nl_del:
                sibling_nl_deletions.update(nl_del)
            sib_name = (
                curr.name
                if _can_bind_in_fn_scope(
                    curr.name, frame.active_nonlocals, state.fn_globals
                )
                else None
            )
            outer_carriers = _extract_fn_outer_carrier_targets(
                curr, state.fn_globals, scope_ctx
            )
            if sib_name is not None:
                if factory_names is not None and (
                    outer_carriers
                    or _is_closure_factory_scope(curr, scope_ctx)
                ):
                    factory_names.add(sib_name)
                if fn_return_refs is not None and isinstance(
                    curr, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    ret_loads, ret_calls = _extract_fn_returned_closure_refs(
                        curr, scope_ctx
                    )
                    if ret_loads or ret_calls:
                        entry = fn_return_refs.setdefault(
                            sib_name, (set(), set())
                        )
                        entry[0].update(ret_loads)
                        entry[1].update(ret_calls)
            sibling_closures.append((sib_name, free_r, nl_mut))
            for oc_name in outer_carriers:
                sibling_closures.append((oc_name, set(free_r), set(nl_mut)))
            if span_in_hdr or _is_frame_after_span(estart, ecol, span, frame):
                state.post_reads.update(free_r)
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
        _push_seq(
            stack,
            hdr_nodes,
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
            force_after_span=frame.force_after_span,
        )
        return True

    if isinstance(curr, ast.Lambda):
        free_r = _extract_expr_scope_free_reads(curr, include_header=False)
        sibling_closures.append((None, free_r, set()))
        if _is_frame_after_span(c_line, c_col, span, frame):
            state.post_reads.update(free_r)
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
                state,
            )
        c_reads, c_muts = _extract_comprehension_deferred_closure_info(
            curr, scope_ctx
        )
        if c_reads or c_muts:
            sibling_closures.append((None, c_reads, c_muts))
        if alias_edges is not None:
            init_exprs, init_targets = (
                _iter_comprehension_body_exprs_and_targets(curr)
            )
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
                    sub_exprs, sub_targets = (
                        _iter_comprehension_body_exprs_and_targets(sub)
                    )
                    if sub.generators:
                        comp_sub_stack.append(
                            (sub.generators[0].iter, active_targets)
                        )
                    next_targets = active_targets | frozenset(sub_targets)
                    comp_sub_stack.extend(
                        (expr, next_targets) for expr in sub_exprs
                    )
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
                    sibling_closures,
                    alias_edges,
                    scope_ctx,
                    call_edges,
                    excluded_names=c_excluded,
                    fn_globals=state.fn_globals,
                )
        if _is_frame_after_span(c_line, c_col, span, frame):
            state.post_reads.update(
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
        c_reads, c_muts, c_loads = (
            _extract_carrier_value_closure_and_load_info(
                c_values, scope_ctx, called_names_out=c_calls
            )
        )
        if excluded_names:
            c_reads -= set(excluded_names)
            c_loads -= set(excluded_names)
            c_calls -= set(excluded_names)
        edge_prefix = "" if is_rebound else "*"
        record_edges = is_rebound or isinstance(curr, ast.Call)
        for t_name in valid_targets:
            if c_reads or c_muts:
                sibling_closures.append((t_name, set(c_reads), set(c_muts)))
            if record_edges:
                edge_tgt = f"{edge_prefix}{t_name}"
                if c_loads:
                    alias_edges.append((edge_tgt, set(c_loads)))
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
            curr.value is not None
            and _node_contains_or_encloses_span(curr.value, span)
        ):
            state.post_reads.add(curr.target.id)

    if isinstance(curr, ast.Name):
        if isinstance(curr.ctx, ast.Store):
            _record_binding_at_pos(
                (curr.id,), c_line, c_col, span, frame, state
            )
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
        elif isinstance(curr.ctx, ast.Load) and _is_frame_after_span(
            c_line, c_col, span, frame
        ):
            state.post_reads.add(curr.id)
    elif isinstance(curr, (ast.Import, ast.ImportFrom)):
        imp_names: Set[str] = set()
        _record_import_names(curr, imp_names)
        _record_binding_at_pos(imp_names, c_line, c_col, span, frame, state)
    else:
        bound_name = _extract_type_alias_or_pattern_name(curr)
        if bound_name is not None:
            _record_binding_at_pos(
                (bound_name,), c_line, c_col, span, frame, state
            )


class _StmtCellDelta(NamedTuple):
    """Cached cell-binding and closure delta for a statement disjoint from the unit span."""

    pre_bound: frozenset[str]
    pre_mutated: frozenset[str]
    post_mutated: frozenset[str]
    post_reads: frozenset[str]
    loop_mutated: frozenset[str]
    sibling_closures: Tuple[
        Tuple[Optional[str], frozenset[str], frozenset[str]], ...
    ]
    concurrent_sib_offsets: Tuple[int, ...]
    alias_edges: Tuple[Tuple[str, frozenset[str]], ...]
    call_edges: Tuple[Tuple[str, frozenset[str]], ...]
    factory_names: frozenset[str]
    fn_return_refs: Tuple[Tuple[str, frozenset[str], frozenset[str]], ...]
    sibling_nl_deletions: frozenset[str]

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
    unit_mutations: AbstractSet[str] = frozenset()


def _make_empty_traversal_buffers(
    fn_globals: AbstractSet[str],
    scope_ctx: _FnScopeContext,
    pre_bound: Optional[Set[str]] = None,
    unit_mutations: AbstractSet[str] = frozenset(),
    initial_suppress_aliases: Optional[
        Union[_SuppressProvenance, AbstractSet[str]]
    ] = None,
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
            suppress_aliases=_coerce_suppress_provenance(
                initial_suppress_aliases
            ),
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=frozenset(
                extra_non_suppressing_ctx_managers
            ),
        ),
        sibling_closures=[],
        concurrent_sib_indices=set(),
        alias_edges=[],
        call_edges=[],
        factory_names=set(),
        fn_return_refs={},
        sibling_nl_deletions=set(),
        scope_ctx=scope_ctx,
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
        bufs.concurrent_sib_indices.update(
            range(base_sib_len, len(bufs.sibling_closures))
        )
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
    has_with = is_before and any(
        isinstance(n, (ast.With, ast.AsyncWith)) for n in ast.walk(stmt)
    )
    init_pre_bound: frozenset[str] = (
        frozenset(
            bufs.state.pre_bound
            & {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
        )
        if has_with and bufs.state.pre_bound
        else frozenset()
    )
    key: Tuple[Any, ...] = (
        (stmt, is_before, sup_key, strict_w, init_pre_bound, extra_non_sup)
        if has_with and (sup_key or strict_w or init_pre_bound or extra_non_sup)
        else (stmt, is_before)
    )
    with _downstream_cache_lock:
        has_key = key in stmt_deltas
        delta = stmt_deltas.get(key) if has_key else None
    if not has_key:
        if _extract_shallow_deleted_names(stmt):
            with _downstream_cache_lock:
                stmt_deltas.setdefault(key, None)
            return False
        dummy_span = (
            _SpanBounds(c_end + 1, c_end + 1)
            if is_before
            else _SpanBounds(0, 0)
        )
        sub_bufs = _make_empty_traversal_buffers(
            bufs.state.fn_globals,
            bufs.scope_ctx,
            pre_bound=set(init_pre_bound) if init_pre_bound else None,
            initial_suppress_aliases=pre_prov,
            strict_with_suppression=strict_w,
            extra_non_suppressing_ctx_managers=extra_non_sup,
            seed_pre_mutated=False,
        )
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
            concurrent_sib_offsets=tuple(
                sorted(sub_bufs.concurrent_sib_indices)
            ),
            alias_edges=tuple(
                (t_name, frozenset(srcs))
                for t_name, srcs in sub_bufs.alias_edges
            ),
            call_edges=tuple(
                (t_name, frozenset(calls))
                for t_name, calls in sub_bufs.call_edges
            ),
            factory_names=frozenset(sub_bufs.factory_names),
            fn_return_refs=tuple(
                (fn_name, frozenset(r_loads), frozenset(r_calls))
                for fn_name, (
                    r_loads,
                    r_calls,
                ) in sub_bufs.fn_return_refs.items()
            ),
            sibling_nl_deletions=frozenset(sub_bufs.sibling_nl_deletions),
        )
        with _downstream_cache_lock:
            delta = stmt_deltas.setdefault(key, computed_delta)
    if delta is None:
        return False
    if (
        is_before
        and not is_branched
        and bufs.state.suppress_aliases is not None
    ):
        bufs.state.suppress_aliases = _update_suppress_provenance_from_stmt(
            stmt,
            pre_prov,
            strict_with_suppression=strict_w,
            extra_non_suppressing_ctx_managers=extra_non_sup,
        )
    _apply_stmt_cell_delta(
        bufs, delta, is_branched=is_branched, in_loop=in_loop
    )
    return True


def _traverse_fn_cell_frames(
    stack: List[_StackFrame],
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
    enclosed_fn: Optional[ast.AST] = None,
    stmt_deltas: Optional[
        Dict[Tuple[Any, ...], Optional[_StmtCellDelta]]
    ] = None,
) -> bool:
    """Traverses AST frames to populate cell bindings, closures, and multi-invocation state."""
    enclosed_fn_name: Optional[str] = (
        enclosed_fn.name
        if isinstance(enclosed_fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    enclosed_fn_call_count = 0
    enclosed_fn_multi_invoked = False
    direct_enclosed_call_funcs: Set[int] = set()

    while stack:
        frame = stack.pop()
        if frame.post_pre_bound is not None:
            bufs.state.pre_bound.update(frame.post_pre_bound)
            bufs.state.pre_unbound.difference_update(frame.post_pre_bound)
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
            and not frame.is_branched
            and bufs.state.suppress_aliases is not None
            and _is_frame_before_span(c_line, c_col, span, frame)
        ):
            bufs.state.suppress_aliases = _update_suppress_provenance_from_stmt(
                curr,
                bufs.state.suppress_aliases,
                strict_with_suppression=bufs.state.strict_with_suppression,
                extra_non_suppressing_ctx_managers=bufs.state.extra_non_suppressing_ctx_managers,
            )

        if curr is enclosed_fn:
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
                estart, ecol = _node_earliest_start_pos(curr)
                _record_binding_at_pos(
                    (curr.name,), estart, ecol, span, frame, bufs.state
                )
                if (
                    frame.in_loop
                    or frame.active_nonlocals is not None
                    or bool(curr.decorator_list)
                    or curr.name in _extract_cached_scope_free_reads(curr)
                ):
                    enclosed_fn_multi_invoked = True
                _push_seq(
                    stack,
                    list(iter_nested_scope_header_nodes(curr)),
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
                continue
            if isinstance(curr, ast.Lambda):
                enclosed_fn_multi_invoked = True
                _push_seq(
                    stack,
                    list(iter_nested_scope_header_nodes(curr)),
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
                continue
            if isinstance(curr, _COMPREHENSION_TYPES):
                enclosed_fn_multi_invoked = True
                body_w = _extract_comprehension_body_walrus_names(curr)
                if body_w:
                    _record_binding_at_pos(
                        body_w,
                        c_line,
                        c_col,
                        span,
                        frame._replace(is_branched=True, in_loop=True),
                        bufs.state,
                    )
                _push_seq(
                    stack,
                    [curr.generators[0].iter] if curr.generators else (),
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                    force_after_span=frame.force_after_span,
                )
                continue

        if _is_node_inside_span(curr, span):
            if isinstance(
                curr,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                ),
            ):
                pre_hdrs = [
                    hdr
                    for hdr in iter_nested_scope_header_nodes(curr)
                    if not _is_node_inside_span(hdr, span)
                ]
                _push_seq(
                    stack,
                    pre_hdrs,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
            continue

        prev_sib_len = len(bufs.sibling_closures)
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            eff_line, eff_col = _node_earliest_start_pos(curr)
        else:
            eff_line, eff_col = c_line, c_col
        is_conc_pos = (
            enclosed_fn is not None
            or frame.in_loop
            or not _is_frame_after_span(eff_line, eff_col, span, frame)
        )
        if _handle_nested_scope_node(
            curr,
            c_line,
            c_col,
            span,
            frame,
            bufs,
            stack,
            is_direct_span=enclosed_fn is None,
        ):
            if is_conc_pos:
                bufs.concurrent_sib_indices.update(
                    range(prev_sib_len, len(bufs.sibling_closures))
                )
            if enclosed_fn_name is not None and not enclosed_fn_multi_invoked:
                if isinstance(
                    curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if not (
                        isinstance(curr, ast.ClassDef)
                        and _strictly_encloses_span(curr, span)
                    ) and enclosed_fn_name in _extract_cached_scope_free_reads(
                        curr
                    ):
                        enclosed_fn_multi_invoked = True
                elif isinstance(curr, (ast.Lambda, *_COMPREHENSION_TYPES)):
                    if enclosed_fn_name in _extract_expr_scope_free_reads(
                        curr, include_header=False
                    ):
                        enclosed_fn_multi_invoked = True
            continue

        if enclosed_fn_name is not None:
            if isinstance(curr, ast.Call):
                if (
                    isinstance(curr.func, ast.Name)
                    and curr.func.id == enclosed_fn_name
                ):
                    direct_enclosed_call_funcs.add(id(curr.func))
                    enclosed_fn_call_count += 1
                    if frame.in_loop or enclosed_fn_call_count >= 2:
                        enclosed_fn_multi_invoked = True
            elif (
                isinstance(curr, ast.Name)
                and isinstance(curr.ctx, ast.Load)
                and curr.id == enclosed_fn_name
                and id(curr) not in direct_enclosed_call_funcs
            ):
                enclosed_fn_multi_invoked = True

        _record_carrier_closures_and_aliases(
            curr,
            frame,
            bufs.sibling_closures,
            bufs.alias_edges,
            bufs.scope_ctx,
            bufs.call_edges,
            fn_globals=bufs.state.fn_globals,
        )
        if is_conc_pos:
            bufs.concurrent_sib_indices.update(
                range(prev_sib_len, len(bufs.sibling_closures))
            )

        if _push_compound_children(
            curr,
            span,
            frame,
            bufs.unit_mutations,
            bufs.state,
            stack,
            unconditional_post_del=enclosed_fn is not None,
            pre_stmt_suppress=pre_stmt_suppress,
        ):
            continue

        _record_simple_node_bindings_and_reads(
            curr,
            c_line,
            c_col,
            span,
            frame,
            bufs.state,
            unconditional_post_del=enclosed_fn is not None,
        )

        if isinstance(
            curr, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)
        ):
            rhs_val = curr.value
            if rhs_val is not None and _node_contains_or_encloses_span(
                rhs_val, span
            ):
                lhs_targets: Sequence[ast.AST] = (
                    curr.targets
                    if isinstance(curr, ast.Assign)
                    else (curr.target,)
                )
                target_after = not (
                    isinstance(curr, ast.AugAssign)
                    and not isinstance(curr.target, ast.Name)
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
                continue

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
    scope_ctx = _get_fn_scope_context(
        fn_node, mod_tree=mod_tree, enclosing_scopes=enclosing_scopes
    )
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
            if _is_pos_before_span(
                t_line, t_col, span
            ) or _node_contains_or_encloses_span(gen.target, span):
                pre_bound.update(t_names)
        comp_walrus = (
            _extract_comprehension_body_walrus_names(fn_node) - fn_globals
        )
        fn_locals |= comp_targets | comp_walrus

    initial_sup = _seed_scope_suppress_aliases(fn_node, mod_tree, scope_ctx)
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
        bufs.state.loop_mutated.update(
            comp_targets | comp_walrus | set(unit_mutations)
        )
    delta_aux_nodes = tuple(
        n for n in scope_ctx.cache_aux_nodes() if n is not fn_node
    )
    stmt_deltas: Optional[
        Dict[Tuple[Any, ...], Optional[_StmtCellDelta]]
    ] = (
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
            _StackFrame(expr_node, is_br, True, None)
            for expr_node, is_br in reversed(comp_frames)
        ]
    else:
        body_seq = getattr(fn_node, "body", ())
        stack = [
            _StackFrame(stmt, False, False, None)
            for stmt in reversed(
                body_seq if isinstance(body_seq, (list, tuple)) else ()
            )
        ]
    enclosed_fn_multi_invoked = _traverse_fn_cell_frames(
        stack, span, bufs, enclosed_fn=enclosed_fn, stmt_deltas=stmt_deltas
    )

    if enclosed_fn is not None and enclosed_fn_multi_invoked:
        enc_muts, enc_dels = _extract_closure_nonlocal_mutations_and_deletions(
            enclosed_fn
        )
        bufs.state.loop_mutated.update(unit_mutations)
        bufs.state.loop_mutated.update(enc_muts)
        if enc_dels:
            bufs.state.pre_bound.difference_update(enc_dels)
            bufs.state.pre_unbound.update(enc_dels)

    if bufs.sibling_nl_deletions:
        bufs.state.pre_bound.difference_update(bufs.sibling_nl_deletions)
        bufs.state.pre_unbound.update(bufs.sibling_nl_deletions)

    closure_carrier_names: Set[str] = (
        {name for name, _, _ in bufs.sibling_closures if name is not None}
        | bufs.factory_names
        | {t.lstrip("*") for t, _ in bufs.alias_edges}
        | {t.lstrip("*") for t, _ in bufs.call_edges}
    )
    for t_name, called_set in bufs.call_edges:
        alias_targets: Set[str] = set(called_set & bufs.factory_names)
        visited_fn_refs: Set[str] = set()
        pending_fn_refs = list(called_set)
        while pending_fn_refs:
            fn_ref = pending_fn_refs.pop()
            if fn_ref in visited_fn_refs:
                continue
            visited_fn_refs.add(fn_ref)
            if fn_ref in bufs.factory_names:
                alias_targets.add(fn_ref)
            ret_loads_set, ret_calls_set = bufs.fn_return_refs.get(
                fn_ref, (set(), set())
            )
            for ret_ref in ret_loads_set:
                if ret_ref in closure_carrier_names:
                    alias_targets.add(ret_ref)
            for call_ref in ret_calls_set:
                if call_ref in bufs.factory_names:
                    alias_targets.add(call_ref)
                if (
                    call_ref in bufs.fn_return_refs
                    or call_ref in bufs.factory_names
                ):
                    pending_fn_refs.append(call_ref)
        if alias_targets:
            bufs.alias_edges.append((t_name, alias_targets))

    propagation_edges: List[Tuple[str, Set[str]]] = [
        (t_name[1:] if t_name.startswith("*") else t_name, srcs)
        for t_name, srcs in bufs.alias_edges
    ]
    rebound_alias_edges: List[Tuple[str, Set[str]]] = [
        (t_name, srcs)
        for t_name, srcs in bufs.alias_edges
        if not t_name.startswith("*")
    ]
    propagated_closures = _propagate_sibling_closures(
        bufs.sibling_closures,
        post_reads=bufs.state.post_reads,
        alias_edges=propagation_edges,
    )
    concurrent_sib_vars: Set[str] = set()
    for s_idx in bufs.concurrent_sib_indices:
        _, s_reads, s_muts = propagated_closures[s_idx]
        concurrent_sib_vars.update(s_reads)
        concurrent_sib_vars.update(s_muts)
    all_sib_nl = (
        set().union(*(nl for _, _, nl in propagated_closures))
        if propagated_closures
        else set()
    )
    return _EnclosingFnCellInfo(
        fn_locals=frozenset(fn_locals),
        pre_bound=frozenset(bufs.state.pre_bound),
        pre_mutated=frozenset(bufs.state.pre_mutated),
        post_mutated=frozenset(bufs.state.post_mutated),
        post_reads=frozenset(bufs.state.post_reads),
        loop_mutated=frozenset(bufs.state.loop_mutated),
        sibling_closures=tuple(
            (name, frozenset(reads), frozenset(muts))
            for name, reads, muts in propagated_closures
        ),
        all_sibling_nonlocal_mutations=frozenset(all_sib_nl),
        alias_edges=tuple(
            (t_name, frozenset(srcs)) for t_name, srcs in rebound_alias_edges
        ),
        pre_unbound=frozenset(bufs.state.pre_unbound),
        concurrent_sibling_vars=frozenset(concurrent_sib_vars),
    )


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
        n
        for n in enclosing_scopes
        if isinstance(n, ast.AST) and n is not fn_node
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *(n for n in (enclosed_fn, mod_tree) if n is not None),
        *norm_enclosing,
    )
    extra_non_sup = frozenset(extra_non_suppressing_ctx_managers)
    params: Tuple[Any, ...] = (
        (
            norm_span,
            frozenset(unit_mutations),
            enclosed_fn is not None,
            bool(strict_with_suppression),
            extra_non_sup,
        )
        if extra_non_sup
        else (
            norm_span,
            frozenset(unit_mutations),
            enclosed_fn is not None,
            True,
        )
        if strict_with_suppression
        else (norm_span, frozenset(unit_mutations), enclosed_fn is not None)
    )
    cache_key = _make_trusted_scope_cache_key(
        owner=fn_node,
        kind="enclosing_fn_cells",
        params=params,
        aux_ast_nodes=aux_nodes,
    )
    return cast(
        _EnclosingFnCellInfo,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _analyze_enclosing_fn_cells_uncached(
                fn_node,
                norm_span,
                unit_mutations=unit_mutations,
                enclosed_fn=enclosed_fn,
                mod_tree=mod_tree,
                enclosing_scopes=norm_enclosing,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_sup,
            ),
        ),
    )


def _find_enclosing_method_class_names(
    fn_node: ast.AST,
    enc_child: ast.AST,
) -> frozenset[str]:
    """Finds and caches class names in fn_node that directly define method enc_child."""
    def _compute() -> frozenset[str]:
        parent_map = _get_scope_parent_map(fn_node)
        curr_p = parent_map.get(enc_child)
        cls_names: Set[str] = set()
        while curr_p is not None and curr_p is not fn_node:
            if isinstance(curr_p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
            if isinstance(curr_p, ast.ClassDef):
                if enc_child in _iter_class_body_methods(curr_p):
                    cls_names.add(curr_p.name)
                break
            curr_p = parent_map.get(curr_p)
        return frozenset(cls_names)

    return cast(
        frozenset[str],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(
                owner=fn_node,
                kind="method_cls_names",
                aux_ast_nodes=(enc_child,),
            ),
            _compute,
        ),
    )


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
                else next_body
                if isinstance(next_body, ast.AST)
                else next_fn
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
        is_static_enc = any(
            (isinstance(d, ast.Name) and d.id == "staticmethod")
            or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
            for d in getattr(enc_child, "decorator_list", ())
        )
        if enc_child is not None and not is_static_enc and rev_infos:
            enc_args = getattr(enc_child, "args", None)
            pos_args = (
                (enc_args.posonlyargs + enc_args.args)
                if isinstance(enc_args, ast.arguments)
                else []
            )
            if pos_args:
                rec_param = pos_args[0].arg
                enclosing_cls_names = _find_enclosing_method_class_names(
                    fn_node, enc_child
                )
                if enclosing_cls_names:
                    extra_sibs = tuple(
                        (rec_param, frozenset(s_reads), frozenset(s_muts))
                        for s_name, s_reads, s_muts in info.sibling_closures
                        if s_name in enclosing_cls_names
                    )
                    if extra_sibs:
                        prev_info = rev_infos[-1]
                        rev_infos[-1] = prev_info._replace(
                            sibling_closures=(
                                *prev_info.sibling_closures,
                                *extra_sibs,
                            ),
                            all_sibling_nonlocal_mutations=frozenset(
                                set(prev_info.all_sibling_nonlocal_mutations)
                                | set().union(*(m for _, _, m in extra_sibs))
                            ),
                        )
        rev_infos.append(info)
        own_bound_locals = (
            _extract_single_node_local_names(fn_node)
            if isinstance(fn_node, _COMPREHENSION_TYPES)
            else info.fn_locals
        )
        escaping_muts = escaping_muts - own_bound_locals
    rev_infos.reverse()
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
        acc_pre_unbound.update(
            (set(info.pre_unbound) | info.fn_locals) - info.pre_bound
        )
    return acc_pre_bound
