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
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
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
    _extract_direct_nonlocals,
    _extract_expr_scope_free_reads,
    _extract_fn_outer_carrier_targets,
    _extract_fn_returned_closure_refs,
    _extract_store_names_in_node,
    _extract_type_alias_or_pattern_name,
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
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _is_irrefutable_case,
    _node_earliest_start_pos,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _record_import_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _CellBindingState(NamedTuple):
    """Mutable sets tracking cell bindings during enclosing function AST traversal."""

    pre_bound: Set[str]
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]
    pre_unbound: Set[str]
    fn_globals: AbstractSet[str] = frozenset()


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


def _extract_definite_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names unconditionally bound by an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: Set[str] = set()
        for elt in target.elts:
            names.update(_extract_definite_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_definite_target_names(target.value)
    return set()


def _extract_shallow_deleted_names(node: ast.AST) -> Set[str]:
    """Extracts variable names deleted or unbound by ExceptHandler in node outside inner scopes."""
    deleted: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Del):
            deleted.add(curr.id)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            deleted.add(curr.name)
        stack.extend(ast.iter_child_nodes(curr))
    return deleted


def _extract_unconditional_walrus_names(node: ast.AST) -> Set[str]:
    """Extracts NamedExpr target names unconditionally evaluated in an AST node."""
    names: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            if curr.generators:
                stack.append(curr.generators[0].iter)
            continue
        if isinstance(curr, ast.BoolOp):
            if curr.values:
                stack.append(curr.values[0])
            continue
        if isinstance(curr, ast.Compare):
            if curr.comparators:
                stack.append(curr.comparators[0])
            stack.append(curr.left)
            continue
        if isinstance(curr, ast.IfExp):
            stack.append(curr.test)
            names.update(
                _extract_unconditional_walrus_names(curr.body)
                & _extract_unconditional_walrus_names(curr.orelse)
            )
            continue
        if isinstance(curr, ast.NamedExpr) and isinstance(curr.target, ast.Name):
            names.add(curr.target.id)
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _eval_try_definite_bindings(
    stmt: ast.AST,
    active_nonlocals: Optional[Set[str]],
    bound: Set[str],
    fn_globals: AbstractSet[str],
) -> Tuple[Set[str], bool]:
    """Computes definite bindings and termination across a Try/TryStar statement."""
    body_l = getattr(stmt, "body", ())
    orelse_l = getattr(stmt, "orelse", ())
    handlers_l = getattr(stmt, "handlers", ())
    final_l = getattr(stmt, "finalbody", ())
    if not handlers_l:
        for sub_seq in (body_l, orelse_l, final_l):
            bound, t_term = _extract_block_definite_bindings_and_term(
                sub_seq, active_nonlocals, bound, fn_globals
            )
            if t_term:
                if sub_seq is not final_l and final_l:
                    f_bound, _ = _extract_block_definite_bindings_and_term(
                        final_l, active_nonlocals, bound, fn_globals
                    )
                    return f_bound, True
                return bound, True
        return bound, False

    if type(stmt).__name__ == "TryStar":
        # PEP 654 exception groups: multiple except* clauses can execute sequentially
        # for different sub-exceptions in a single ExceptionGroup, so conservatively
        # clear shallow deletions across the TryStar subtree and evaluate finalbody.
        bound.difference_update(_extract_shallow_deleted_names(stmt))
        if final_l:
            return _extract_block_definite_bindings_and_term(
                final_l, active_nonlocals, bound, fn_globals
            )
        return bound, False

    pre_try_surviving = set(bound)
    for b_stmt in body_l:
        pre_try_surviving.difference_update(
            _extract_shallow_deleted_names(b_stmt)
        )
    for h_node in handlers_l:
        h_name = getattr(h_node, "name", None)
        if isinstance(h_name, str):
            pre_try_surviving.discard(h_name)

    normal_bound, normal_term = _extract_block_definite_bindings_and_term(
        body_l, active_nonlocals, bound, fn_globals
    )
    if not normal_term and orelse_l:
        normal_bound, normal_term = _extract_block_definite_bindings_and_term(
            orelse_l, active_nonlocals, normal_bound, fn_globals
        )

    fallthrough_sets: List[Set[str]] = []
    if not normal_term:
        fallthrough_sets.append(normal_bound)

    for h_node in handlers_l:
        h_body = getattr(h_node, "body", ())
        h_init = set(pre_try_surviving)
        h_name = getattr(h_node, "name", None)
        if isinstance(h_name, str) and _can_bind_in_fn_scope(
            h_name, active_nonlocals, fn_globals
        ):
            h_init.add(h_name)
        h_bound, h_term = _extract_block_definite_bindings_and_term(
            h_body, active_nonlocals, h_init, fn_globals
        )
        if isinstance(h_name, str):
            h_bound.discard(h_name)
        if not h_term:
            fallthrough_sets.append(h_bound)

    if not fallthrough_sets:
        if final_l:
            f_bound, _ = _extract_block_definite_bindings_and_term(
                final_l, active_nonlocals, pre_try_surviving, fn_globals
            )
            return f_bound, True
        return set(), True

    out_bound = set.intersection(*fallthrough_sets)
    if final_l:
        out_bound, f_term = _extract_block_definite_bindings_and_term(
            final_l, active_nonlocals, out_bound, fn_globals
        )
        if f_term:
            return out_bound, True
    return out_bound, False


def _apply_expr_definite_effects(
    target_bound: Set[str],
    expr_node: Any,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
) -> None:
    """Applies shallow deletions and unconditional walrus bindings from an AST node."""
    if not isinstance(expr_node, ast.AST):
        return
    target_bound.difference_update(_extract_shallow_deleted_names(expr_node))
    target_bound.update(
        n
        for n in _extract_unconditional_walrus_names(expr_node)
        if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
    )


def _eval_match_definite_bindings(
    stmt: ast.AST,
    active_nonlocals: Optional[Set[str]],
    bound: Set[str],
    fn_globals: AbstractSet[str],
) -> Tuple[Set[str], bool]:
    """Computes definite bindings and termination across a Match statement."""
    _apply_expr_definite_effects(
        bound, getattr(stmt, "subject", None), active_nonlocals, fn_globals
    )
    cases = getattr(stmt, "cases", ())
    if not cases:
        return bound, False
    fallthrough_sets: List[Set[str]] = []
    has_irrefutable = False
    for case in cases:
        case_init = set(bound)
        pattern = getattr(case, "pattern", None)
        if isinstance(pattern, ast.AST):
            for p_node in ast.walk(pattern):
                p_name = _extract_pattern_bound_name(p_node)
                if p_name is not None and _can_bind_in_fn_scope(
                    p_name, active_nonlocals, fn_globals
                ):
                    case_init.add(p_name)
        _apply_expr_definite_effects(
            case_init, getattr(case, "guard", None), active_nonlocals, fn_globals
        )
        c_bound, c_term = _extract_block_definite_bindings_and_term(
            getattr(case, "body", ()),
            active_nonlocals,
            case_init,
            fn_globals,
        )
        if not c_term:
            fallthrough_sets.append(c_bound)
        if _is_irrefutable_case(case):
            has_irrefutable = True
            break
    if not has_irrefutable:
        fallthrough_sets.append(bound)
        return set.intersection(*fallthrough_sets), False
    if not fallthrough_sets:
        return set(), True
    return set.intersection(*fallthrough_sets), False


def _extract_block_definite_bindings_and_term(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> Tuple[Set[str], bool]:
    """Extracts (unconditionally_bound_names, block_terminates) across a statement sequence."""
    bound: Set[str] = set(initial_bound)
    for stmt in stmts:
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            for hdr in iter_nested_scope_header_nodes(stmt):
                _apply_expr_definite_effects(
                    bound, hdr, active_nonlocals, fn_globals
                )
            if _can_bind_in_fn_scope(stmt.name, active_nonlocals, fn_globals):
                bound.add(stmt.name)
            continue
        if isinstance(stmt, ast.If):
            _apply_expr_definite_effects(
                bound, stmt.test, active_nonlocals, fn_globals
            )
            b_bound, b_term = _extract_block_definite_bindings_and_term(
                stmt.body, active_nonlocals, bound, fn_globals
            )
            if stmt.orelse:
                o_bound, o_term = _extract_block_definite_bindings_and_term(
                    stmt.orelse, active_nonlocals, bound, fn_globals
                )
                if b_term and o_term:
                    return b_bound & o_bound, True
                if b_term:
                    bound = o_bound
                elif o_term:
                    bound = b_bound
                else:
                    bound = b_bound & o_bound
            elif not b_term:
                bound = bound & b_bound
            continue
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                _apply_expr_definite_effects(
                    bound, item.context_expr, active_nonlocals, fn_globals
                )
                if item.optional_vars is not None:
                    bound.update(
                        n
                        for n in _extract_definite_target_names(
                            item.optional_vars
                        )
                        if _can_bind_in_fn_scope(
                            n, active_nonlocals, fn_globals
                        )
                    )
            for b_stmt in stmt.body:
                bound.difference_update(_extract_shallow_deleted_names(b_stmt))
            continue
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            bound.difference_update(_extract_shallow_deleted_names(stmt))
            bound.update(
                n
                for n in _extract_unconditional_walrus_names(stmt.iter)
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
            continue
        if isinstance(stmt, ast.While):
            bound.difference_update(_extract_shallow_deleted_names(stmt))
            bound.update(
                n
                for n in _extract_unconditional_walrus_names(stmt.test)
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
            continue
        if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            bound, t_term = _eval_try_definite_bindings(
                stmt, active_nonlocals, bound, fn_globals
            )
            if t_term:
                return bound, True
            continue
        if type(stmt).__name__ == "Match":
            bound, m_term = _eval_match_definite_bindings(
                stmt, active_nonlocals, bound, fn_globals
            )
            if m_term:
                return bound, True
            continue
        _apply_expr_definite_effects(bound, stmt, active_nonlocals, fn_globals)
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return bound, True
        if isinstance(stmt, ast.Assign) and stmt.value is not None:
            for tgt in stmt.targets:
                bound.update(
                    n
                    for n in _extract_definite_target_names(tgt)
                    if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
                )
        elif (
            isinstance(stmt, (ast.AnnAssign, ast.AugAssign))
            and stmt.value is not None
        ):
            bound.update(
                n
                for n in _extract_definite_target_names(stmt.target)
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imp_names: Set[str] = set()
            _record_import_names(stmt, imp_names)
            bound.update(
                n
                for n in imp_names
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
        elif type(stmt).__name__ == "TypeAlias":
            ta_name = _extract_type_alias_or_pattern_name(stmt)
            if ta_name is not None and _can_bind_in_fn_scope(
                ta_name, active_nonlocals, fn_globals
            ):
                bound.add(ta_name)
    return bound, False


def _extract_block_definite_bindings(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Extracts variable names unconditionally bound across a statement sequence."""
    return _extract_block_definite_bindings_and_term(
        stmts, active_nonlocals, initial_bound, fn_globals
    )[0]


def _maybe_push_pre_span_exhaustive_frame(
    curr: ast.AST,
    encloses: bool,
    has_clauses: bool,
    frame: _StackFrame,
    span: _SpanBounds,
    state: _CellBindingState,
    stack: List[_StackFrame],
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
        exh_bound = _extract_block_definite_bindings(
            (curr,), frame.active_nonlocals, state.pre_bound, state.fn_globals
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
            curr, encloses, bool(curr.body), frame, span, state, stack
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
        _push_seq(
            stack, curr.items, is_br, in_lp, act_nl, force_after_span=f_after
        )
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
            curr, encloses, bool(cases_l), frame, span, state, stack
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
    state: _CellBindingState,
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]],
    stack: List[_StackFrame],
    scope_ctx: Optional[_FnScopeContext] = None,
    factory_names: Optional[Set[str]] = None,
    fn_return_refs: Optional[Dict[str, Tuple[Set[str], Set[str]]]] = None,
    alias_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    call_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    sibling_nl_deletions: Optional[Set[str]] = None,
    is_direct_span: bool = True,
) -> bool:
    """Processes nested function, class, lambda, or comprehension scopes during cell analysis."""
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
                    if s_reads or s_muts:
                        sibling_closures.append(
                            (curr.name, set(s_reads), set(s_muts))
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
        if curr.generators:
            stack.append(
                _StackFrame(
                    curr.generators[0].iter,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                    force_after_span=frame.force_after_span,
                )
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
) -> _CellTraversalBuffers:
    """Creates fresh mutable traversal buffers for enclosing function cell analysis."""
    bound_set: Set[str] = set(pre_bound) if pre_bound is not None else set()
    return _CellTraversalBuffers(
        state=_CellBindingState(
            pre_bound=bound_set,
            pre_mutated=set(bound_set),
            post_mutated=set(),
            post_reads=set(),
            loop_mutated=set(),
            pre_unbound=set(),
            fn_globals=fn_globals,
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
) -> None:
    """Applies a cached disjoint statement cell delta to traversal buffers."""
    bufs.state.pre_bound.update(delta.pre_bound)
    bufs.state.pre_unbound.difference_update(delta.pre_bound)
    bufs.state.pre_mutated.update(delta.pre_mutated)
    bufs.state.post_mutated.update(delta.post_mutated)
    bufs.state.post_reads.update(delta.post_reads)
    bufs.state.loop_mutated.update(delta.loop_mutated)
    base_sib_len = len(bufs.sibling_closures)
    for name, reads, muts in delta.sibling_closures:
        bufs.sibling_closures.append((name, set(reads), set(muts)))
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
    stmt_deltas: Dict[Tuple[ast.AST, bool], Optional[_StmtCellDelta]],
) -> bool:
    """Computes or applies a cached _StmtCellDelta for a disjoint top-level statement."""
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
    key = (stmt, is_before)
    if key not in stmt_deltas:
        if _extract_shallow_deleted_names(stmt):
            stmt_deltas[key] = None
            return False
        dummy_span = (
            _SpanBounds(c_end + 1, c_end + 1)
            if is_before
            else _SpanBounds(0, 0)
        )
        sub_bufs = _make_empty_traversal_buffers(
            bufs.state.fn_globals, bufs.scope_ctx
        )
        _traverse_fn_cell_frames(
            [_StackFrame(stmt, False, False, None)],
            dummy_span,
            sub_bufs,
        )
        stmt_deltas[key] = _StmtCellDelta(
            pre_bound=frozenset(sub_bufs.state.pre_bound),
            pre_mutated=frozenset(sub_bufs.state.pre_mutated),
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
    delta = stmt_deltas[key]
    if delta is None:
        return False
    _apply_stmt_cell_delta(bufs, delta)
    return True


def _traverse_fn_cell_frames(
    stack: List[_StackFrame],
    span: _SpanBounds,
    bufs: _CellTraversalBuffers,
    enclosed_fn: Optional[ast.AST] = None,
    stmt_deltas: Optional[
        Dict[Tuple[ast.AST, bool], Optional[_StmtCellDelta]]
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
            and not frame.is_branched
            and not frame.in_loop
            and not frame.force_after_span
            and frame.active_nonlocals is None
            and isinstance(curr, ast.stmt)
            and _try_apply_cached_stmt_delta(curr, span, bufs, stmt_deltas)
        ):
            continue

        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)

        if curr is enclosed_fn and isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
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

        if _is_node_inside_span(curr, span):
            if isinstance(
                curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
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
            bufs.state,
            bufs.sibling_closures,
            stack,
            bufs.scope_ctx,
            bufs.factory_names,
            bufs.fn_return_refs,
            bufs.alias_edges,
            bufs.call_edges,
            bufs.sibling_nl_deletions,
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
) -> _EnclosingFnCellInfo:
    """Computes variable bindings, post-span mutations, loop mutations, and sibling closures."""
    scope_ctx = _get_fn_scope_context(fn_node, mod_tree=mod_tree)
    fn_globals = _extract_direct_nonlocals(fn_node, is_global=True)
    tp_names = set(_extract_node_type_param_names(fn_node)) - fn_globals
    fn_locals = set(scope_ctx.fn_locals) | tp_names
    pre_bound: Set[str] = set(tp_names)
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args) - fn_globals)

    bufs = _make_empty_traversal_buffers(
        fn_globals, scope_ctx, pre_bound=pre_bound, unit_mutations=unit_mutations
    )
    stmt_deltas: Optional[
        Dict[Tuple[ast.AST, bool], Optional[_StmtCellDelta]]
    ] = (
        cast(
            Dict[Tuple[ast.AST, bool], Optional[_StmtCellDelta]],
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                (fn_node, "stmt_deltas", mod_tree),
                dict,
            ),
        )
        if enclosed_fn is None
        else None
    )
    stack: List[_StackFrame] = [
        _StackFrame(stmt, False, False, None)
        for stmt in reversed(getattr(fn_node, "body", []))
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
) -> _EnclosingFnCellInfo:
    """Analyzes and caches cell bindings, post-span mutations, loop mutations, and closures."""
    norm_span = span if isinstance(span, _SpanBounds) else _SpanBounds(*span)
    cache_key: Any = (
        fn_node,
        norm_span,
        frozenset(unit_mutations),
        enclosed_fn,
        mod_tree,
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
            ),
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
            next_body = getattr(next_fn, "body", [])
            first_stmt = next_body[0] if next_body else next_fn
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
                enclosing_cls_names = {
                    sub.name
                    for sub in ast.walk(fn_node)
                    if isinstance(sub, ast.ClassDef)
                    and enc_child in _iter_class_body_methods(sub)
                }
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
        escaping_muts = escaping_muts - info.fn_locals
    rev_infos.reverse()
    return rev_infos


def _find_enclosing_chain_definite_pre_bound(
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
    mod_tree: Optional[ast.AST] = None,
) -> Set[str]:
    """Computes variables definitely bound prior to the unit across enclosing functions."""
    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        u_start,
        u_end,
        u_start_col,
        u_end_col,
        frozenset(),
        mod_tree=mod_tree,
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
