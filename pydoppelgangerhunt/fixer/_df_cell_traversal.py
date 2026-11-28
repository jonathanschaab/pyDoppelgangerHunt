"""Binding-state and definite-binding AST traversal for enclosing function cells."""

from __future__ import annotations

import ast
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
)

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    record_heuristic_non_suppressing_ctx_manager,
    record_wildcard_ctx_manager_match,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _SessionBucket,
    _get_or_compute_scope_cached,
    _get_or_compute_scope_cached_with_hit,
    _make_trusted_scope_cache_key,
    analysis_session,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _CarrierEdge,
    _extract_carrier_value_closure_and_load_info,
    _extract_fn_outer_carrier_info,
    _extract_fn_returned_closure_refs,
    _extract_store_names_in_node,
    _is_closure_factory_scope,
    _iter_carrier_target_entries,
    _iter_class_body_methods,
    _iter_class_body_non_method_stmts,
    _propagate_sibling_closures,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import (
    _EnclosingFnCellInfo,
    _FnScopeContext,
    _SpanBounds,
    _can_bind_in_fn_scope,
    _extract_cached_scope_free_reads,
    _extract_closure_nonlocal_mutations_and_deletions,
    _extract_comprehension_body_walrus_names,
    _extract_comprehension_deferred_closure_info,
    _extract_decorator_carrier_targets,
    _extract_expr_scope_free_reads,
    _get_fn_scope_context,
    _is_node_inside_span,
    _is_pos_before_span,
    _iter_comprehension_body_exprs_and_targets,
    _node_contains_or_encloses_span,
    _strictly_encloses_span,
)
from pydoppelgangerhunt.fixer._df_cell_frames import (
    _CellBindingState,
    _StackFrame,
    _advance_suppress_aliases_for_stmt,
    _compute_advanced_suppress_aliases,
    _is_compound_scope_traversal_stmt,
    _is_frame_after_span,
    _is_frame_before_span,
    _maybe_push_pre_span_exhaustive_frame,
    _merge_branched_suppress_update,
    _push_compound_children,
    _push_seq,
    _record_binding_at_pos,
    _record_deletion_at_pos,
)
from pydoppelgangerhunt.fixer._df_cell_receivers import (
    _attach_receiver_closures_to_info,
    _collect_method_receiver_sibling_closures,
    _collect_propagated_prefix_closures,
    _extract_method_receiver_param,
    _find_enclosing_method_class_node,
    _has_staticmethod_decorator,
    _resolve_enclosing_receiver_shadowed_builtins,
)
from pydoppelgangerhunt.fixer._df_prov_exprs import _advance_prov_through_opt_expr
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _extract_node_type_param_names,
    _extract_single_node_local_names,
    _node_earliest_start_pos,
)
from pydoppelgangerhunt.fixer._df_span_index import _extract_direct_nonlocals
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _record_import_names,
)
from pydoppelgangerhunt.fixer._df_cfg import _extract_type_alias_or_pattern_name
from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _SuppressProvenance,
    _coerce_suppress_provenance,
)
from pydoppelgangerhunt.fixer._df_with_rules import _extract_shallow_deleted_names
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
            bufs.alias_edges.append(
                _CarrierEdge(oc_name, frozenset((oc_name,)), is_mutation=True)
            )


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
        bufs.alias_edges.append(_CarrierEdge(cls_name, frozenset(loads), is_mutation=False))
        if is_enclosing_cls:
            bufs.class_body_alias_edge_indices.add(len(bufs.alias_edges) - 1)
    if calls and bufs.call_edges is not None:
        bufs.call_edges.append(_CarrierEdge(cls_name, frozenset(calls), is_mutation=False))
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
        s_carrier_info = _extract_fn_outer_carrier_info(
            stmt, bufs.state.fn_globals, bufs.scope_ctx
        )
        s_carriers = set(s_carrier_info.direct_targets) | {
            d_name
            for d_name in _extract_decorator_carrier_targets(stmt)
            if d_name not in bufs.state.fn_globals
        }
        if bufs.alias_edges is not None and s_carrier_info.alias_edges:
            bufs.alias_edges.extend(s_carrier_info.alias_edges)
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
        for _is_reb, _c_targets, c_values in _iter_carrier_target_entries(
            attr_stmt, scope_ctx=bufs.scope_ctx
        ):
            if _record_class_expr_closures(
                cls_node.name, c_values, bufs, is_enclosing_cls=True
            ):
                has_cls_closures = True
    if cls_dec_carriers and has_cls_closures and bufs.alias_edges is not None:
        for d_name in cls_dec_carriers:
            bufs.alias_edges.append(
                _CarrierEdge(d_name, frozenset((cls_node.name,)), is_mutation=True)
            )


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
    carrier_info = _extract_fn_outer_carrier_info(
        curr, state.fn_globals, bufs.scope_ctx
    )
    outer_carriers = carrier_info.direct_targets
    if bufs.alias_edges is not None and carrier_info.alias_edges:
        bufs.alias_edges.extend(carrier_info.alias_edges)
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
    alias_edges: List[_CarrierEdge],
    scope_ctx: Optional[_FnScopeContext] = None,
    call_edges: Optional[List[_CarrierEdge]] = None,
    *,
    excluded_names: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> None:
    """Extracts inline closures and variable-rebinding alias edges from curr."""
    for is_rebound, c_targets, c_values in _iter_carrier_target_entries(
        curr, scope_ctx=scope_ctx
    ):
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
        is_mut = not is_rebound
        for t_name in valid_targets:
            if c_reads or c_muts:
                sibling_closures.append((t_name, set(c_reads), set(c_muts)))
            edge_srcs = set(c_loads)
            if is_mut and (c_reads or c_muts):
                edge_srcs.add(t_name)
            if edge_srcs:
                alias_edges.append(
                    _CarrierEdge(t_name, frozenset(edge_srcs), is_mutation=is_mut)
                )
            if c_calls and call_edges is not None:
                call_edges.append(
                    _CarrierEdge(t_name, frozenset(c_calls), is_mutation=is_mut)
                )


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
    alias_edges: Tuple[_CarrierEdge, ...]
    call_edges: Tuple[_CarrierEdge, ...]
    factory_names: frozenset[str]
    fn_return_refs: Tuple[Tuple[str, frozenset[str], frozenset[str]], ...]
    sibling_nl_deletions: frozenset[str]
    heuristic_ctx_managers: Tuple[str, ...] = ()
    wildcard_ctx_managers: Tuple[Tuple[str, str], ...] = ()


class _CellTraversalBuffers(NamedTuple):
    """Mutable accumulators populated while traversing function AST frames."""

    state: _CellBindingState
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]]
    concurrent_sib_indices: Set[int]
    alias_edges: List[_CarrierEdge]
    call_edges: List[_CarrierEdge]
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
    bufs.alias_edges.extend(delta.alias_edges)
    bufs.call_edges.extend(delta.call_edges)
    bufs.factory_names.update(delta.factory_names)
    for fn_name, ret_loads, ret_calls in delta.fn_return_refs:
        entry = bufs.fn_return_refs.setdefault(fn_name, (set(), set()))
        entry[0].update(ret_loads)
        entry[1].update(ret_calls)
    bufs.sibling_nl_deletions.update(delta.sibling_nl_deletions)


def _get_stmt_with_summary(
    stmt: ast.stmt,
) -> Tuple[bool, frozenset[str]]:
    """Returns session-cached (has_with, stmt_name_ids) for a statement node."""
    def _compute() -> Tuple[bool, frozenset[str]]:
        has_w = False
        name_ids: Set[str] = set()
        for node in ast.walk(stmt):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                has_w = True
            elif isinstance(node, ast.Name):
                name_ids.add(node.id)
        return (
            has_w,
            frozenset(name_ids) if has_w else frozenset(),
        )

    return _get_or_compute_scope_cached(
        _SessionBucket.STMT_WITH_SUMMARY,
        stmt,
        _compute,
    )


def _try_apply_cached_stmt_delta(
    stmt: ast.stmt,
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
    stmt_delta_owner: Optional[ast.AST] = None,
    *,
    stmt_delta_aux_nodes: Tuple[ast.AST, ...] = (),
    is_branched: bool = False,
    in_loop: bool = False,
) -> bool:
    """Computes or applies a session-cached _StmtCellDelta for a disjoint statement."""
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
    cache_owner = stmt_delta_owner if isinstance(stmt_delta_owner, ast.AST) else stmt
    delta_params: Tuple[Any, ...] = (
        (stmt, is_before, sup_key, strict_w, init_pre_bound, extra_non_sup)
        if has_with
        else (stmt, is_before)
    )
    cache_key = _make_trusted_scope_cache_key(
        owner=cache_owner,
        kind="stmt_delta",
        params=delta_params,
        aux_ast_nodes=stmt_delta_aux_nodes,
    )

    def _compute_delta() -> Optional[_StmtCellDelta]:
        if _extract_shallow_deleted_names(stmt):
            return None
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
        return _StmtCellDelta(
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
            alias_edges=tuple(sub_bufs.alias_edges),
            call_edges=tuple(sub_bufs.call_edges),
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

    delta, was_hit = _get_or_compute_scope_cached_with_hit(
        _SessionBucket.STMT_DELTAS,
        cache_key,
        _compute_delta,
    )
    if delta is None:
        return False
    if was_hit and not is_branched:
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
    stmt_delta_owner: Optional[ast.AST] = None,
    stmt_delta_aux_nodes: Tuple[ast.AST, ...] = (),
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
        if frame.save_suppress_box is not None or frame.save_bound_box is not None:
            if frame.save_suppress_box is not None:
                frame.save_suppress_box[0] = bufs.state.suppress_aliases
            if frame.save_bound_box is not None:
                frame.save_bound_box[0] = (
                    set(bufs.state.pre_bound),
                    set(bufs.state.pre_unbound),
                )
            continue
        if frame.restore_suppress_box is not None or frame.restore_bound_box is not None:
            if frame.restore_suppress_box is not None:
                bufs.state.suppress_aliases = frame.restore_suppress_box[0]
            if frame.restore_bound_box is not None:
                saved_bounds = frame.restore_bound_box[0]
                if saved_bounds is not None:
                    bufs.state.pre_bound = set(saved_bounds[0])
                    bufs.state.pre_unbound = set(saved_bounds[1])
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
            stmt_delta_owner is not None
            and enclosed_fn is None
            and not frame.force_after_span
            and frame.active_nonlocals is None
            and isinstance(curr, ast.stmt)
            and _try_apply_cached_stmt_delta(
                curr,
                span,
                bufs,
                stmt_delta_owner,
                stmt_delta_aux_nodes=stmt_delta_aux_nodes,
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
    stmt_delta_owner: Optional[ast.AST] = (
        fn_node
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
            stack,
            span,
            bufs,
            enclosed_fn=enclosed_fn,
            stmt_delta_owner=stmt_delta_owner,
            stmt_delta_aux_nodes=delta_aux_nodes,
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
        (edge.target, set(edge.sources))
        for edge in bufs.alias_edges
        if not edge.is_mutation
    ]
    mutation_edges: List[Tuple[str, Set[str]]] = [
        (edge.target, set(edge.sources))
        for edge in bufs.alias_edges
        if edge.is_mutation
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
        for edge in bufs.alias_edges:
            if (
                edge.target
                and not edge.is_mutation
                and bool(edge.sources & targets)
                and edge.target not in targets
            ):
                targets.add(edge.target)
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
        for c_edge in bufs.call_edges:
            if (
                c_edge.target
                and not c_edge.is_mutation
                and (
                    bool(c_edge.sources & targets)
                    or bool(c_edge.sources & factory_targets)
                )
                and c_edge.target not in targets
            ):
                targets.add(c_edge.target)
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
    for idx, edge in enumerate(bufs.alias_edges):
        if idx in bufs.class_body_alias_edge_indices:
            continue
        if edge.is_mutation:
            ext_mut_edges.append((edge.target, set(edge.sources)))
        else:
            ext_rebound_edges.append((edge.target, set(edge.sources)))
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
        edge.target for edge in bufs.alias_edges if edge.is_mutation
    } | {edge.target for edge in bufs.call_edges if edge.is_mutation}
    closure_carrier_names: Set[str] = (
        {name for name, _, _ in bufs.sibling_closures if name is not None}
        | bufs.factory_names
        | {edge.target for edge in bufs.alias_edges}
        | {edge.target for edge in bufs.call_edges}
    )
    var_alias_map: Dict[str, Set[str]] = {}
    for edge in bufs.alias_edges:
        var_alias_map.setdefault(edge.target, set()).update(edge.sources)
    resolved_call_targets: List[Set[str]] = [set() for _ in bufs.call_edges]
    call_changed = True
    while call_changed:
        call_changed = False
        for idx, c_edge in enumerate(bufs.call_edges):
            alias_targets = resolved_call_targets[idx]
            visited_fn_refs: Set[str] = set()
            pending_fn_refs = list(c_edge.sources)
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
                closure_carrier_names.add(c_edge.target)
                tgt_aliases = var_alias_map.setdefault(c_edge.target, set())
                if not alias_targets.issubset(tgt_aliases):
                    tgt_aliases.update(alias_targets)
                    call_changed = True
    for idx, c_edge in enumerate(bufs.call_edges):
        if resolved_call_targets[idx]:
            bufs.alias_edges.append(
                _CarrierEdge(
                    c_edge.target,
                    frozenset(resolved_call_targets[idx]),
                    is_mutation=c_edge.is_mutation,
                )
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

    with analysis_session():
        info = _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            cache_key,
            _compute_cells,
        )
    if not was_computed[0]:
        if not strict_with_suppression and info.heuristic_ctx_managers:
            for mgr_name in info.heuristic_ctx_managers:
                record_heuristic_non_suppressing_ctx_manager(mgr_name)
        if info.wildcard_ctx_managers:
            for wc_pat, mgr_name in info.wildcard_ctx_managers:
                record_wildcard_ctx_manager_match(wc_pat, mgr_name)
    return info


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
