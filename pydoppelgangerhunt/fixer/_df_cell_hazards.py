"""Enclosing function cell binding analysis and severed closure cell hazard checks."""

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
    _get_scope_parent_map,
    _scope_cell_info_cache,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _collect_scope_closures_uncached,
    _extract_comprehension_walrus_names,
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    _has_wildcard_star_import,
    _is_eagerly_consumed_genexp,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _is_ast_node_within_unit_bounds,
    _record_import_names,
)
from pydoppelgangerhunt.fixer.source import (
    _get_tree_scope_intervals,
    _node_scope_bounds,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _FnScopeContext(NamedTuple):
    """Cached per-function lexical scope metadata for eager generator expression checks."""

    fn_locals: Set[str]
    parent_map: Dict[ast.AST, ast.AST]
    shadowed_names: Set[str]
    has_star_import: bool


def _get_fn_scope_context_uncached(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    *,
    cache_parent_map: bool = True,
) -> _FnScopeContext:
    """Computes lexical scope metadata for a function or ephemeral unit module without caching."""
    return _FnScopeContext(
        fn_locals=_extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        ),
        parent_map=_get_scope_parent_map(scope_node, use_cache=cache_parent_map),
        shadowed_names=_extract_scope_local_names(
            scope_node, module_tree=mod_tree
        ),
        has_star_import=_has_wildcard_star_import(
            scope_node, module_tree=mod_tree
        ),
    )


def _get_fn_scope_context(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
) -> _FnScopeContext:
    """Computes and caches lexical scope metadata for an AST scope node."""
    cache_key: Any = (
        (scope_node, "fn_scope_ctx")
        if mod_tree is None
        else (scope_node, "fn_scope_ctx", mod_tree)
    )
    return cast(
        _FnScopeContext,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _get_fn_scope_context_uncached(
                scope_node, mod_tree=mod_tree
            ),
        ),
    )


def _is_lazy_genexp(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node is a GeneratorExp that is not eagerly consumed."""
    if not isinstance(node, ast.GeneratorExp):
        return False
    if scope_ctx is None:
        return True
    return not _is_eagerly_consumed_genexp(
        node,
        scope_ctx.parent_map,
        scope_ctx.shadowed_names,
        has_star_import=scope_ctx.has_star_import,
    )


def _node_earliest_start_lineno(node: ast.AST) -> int:
    """Computes the earliest starting line of an AST node including any decorators."""
    return _node_scope_bounds(node)[0]


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing FunctionDef/AsyncFunctionDef nodes ordered outermost to innermost."""
    chain: List[Tuple[int, int, int, ast.AST]] = []
    for estart, n_end, _span, node in _get_tree_scope_intervals(mod_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if estart <= 0 or estart > u_start or u_end > n_end:
            continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
        if (
            u_start in (estart, int(getattr(node, "lineno", estart) or estart))
            and u_end == n_end
            and (u_start_col is None or n_col == u_start_col)
        ):
            continue
        chain.append((estart, n_end, n_col, node))
    chain.sort(key=lambda item: (item[0], -item[1], item[2]))
    return [(estart, n_end, node) for estart, n_end, _col, node in chain]


def _extract_direct_nonlocals(scope_node: ast.AST) -> Set[str]:
    """Extracts variable names declared nonlocal directly in scope_node."""
    names: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Nonlocal):
            names.update(curr.names)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _extract_type_alias_or_pattern_name(node: ast.AST) -> Optional[str]:
    """Extracts the bound identifier from a TypeAlias or match pattern AST node."""
    cls_name = type(node).__name__
    if cls_name == "TypeAlias":
        target_name = getattr(node, "name", None)
        return target_name.id if isinstance(target_name, ast.Name) else None
    if cls_name in ("MatchAs", "MatchStar", "MatchMapping"):
        return _extract_pattern_bound_name(node)
    return None


def _extract_closure_nonlocal_mutations_uncached(scope_node: ast.AST) -> Set[str]:
    """Computes outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set()
    is_cls = isinstance(scope_node, ast.ClassDef)
    own_locals = (
        set()
        if is_cls
        else _extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        )
    )
    direct_nonlocals: Set[str] = set()
    direct_mutated: Set[str] = set()
    nested_mutated: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            direct_mutated.add(curr.name)
            nested_mutated.update(
                _extract_closure_nonlocal_mutations(curr) - own_locals
            )
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            direct_mutated.update(_extract_comprehension_walrus_names(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Nonlocal):
            direct_nonlocals.update(curr.names)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, (ast.Store, ast.Del)):
            direct_mutated.add(curr.id)
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            _record_import_names(curr, direct_mutated)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            direct_mutated.add(curr.name)
        else:
            bound_name = _extract_type_alias_or_pattern_name(curr)
            if bound_name is not None:
                direct_mutated.add(bound_name)
        stack.extend(ast.iter_child_nodes(curr))
    return (direct_nonlocals & direct_mutated) | nested_mutated


def _extract_closure_nonlocal_mutations(scope_node: ast.AST) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set()
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (scope_node, "nl_muts"),
            lambda: _extract_closure_nonlocal_mutations_uncached(scope_node),
        )
    )


def _extract_cached_scope_free_reads(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> Set[str]:
    """Extracts and caches free variable reads escaping a nested function or class."""
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (node, "free_reads"),
            lambda: _extract_nested_scope_free_reads(node),
        )
    )


def _extract_expr_scope_free_reads(
    expr: Union[
        ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp
    ],
    *,
    include_header: bool = False,
) -> Set[str]:
    """Extracts free variable reads from a Lambda or comprehension without synthetic AST nodes."""
    return _extract_nested_scope_free_reads(expr, include_header=include_header)


def _extract_lazy_genexp_walrus_names(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts walrus target names inside lazy GeneratorExp subtrees of node."""
    if _is_lazy_genexp(node, scope_ctx):
        return _extract_comprehension_walrus_names(node)
    names: Set[str] = set()
    stack: List[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                names.update(_extract_comprehension_walrus_names(curr))
                continue
            stack.extend(ast.iter_child_nodes(curr))
            continue
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _extract_comprehension_deferred_closure_info(
    comp_node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (deferred_free_reads, lazy_walrus_mutations) from a comprehension node."""
    is_lazy = _is_lazy_genexp(comp_node, scope_ctx)
    nested_lazy_nodes: List[Union[ast.Lambda, ast.GeneratorExp]] = []
    sub_stack: List[ast.AST] = (
        [comp_node.generators[0].iter]
        if (is_lazy and comp_node.generators)
        else list(ast.iter_child_nodes(comp_node))
    )
    while sub_stack:
        sub = sub_stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sub_stack.extend(iter_nested_scope_header_nodes(sub))
            continue
        if isinstance(sub, ast.Lambda):
            nested_lazy_nodes.append(sub)
            sub_stack.extend(iter_nested_scope_header_nodes(sub))
            continue
        if isinstance(sub, ast.GeneratorExp):
            if _is_lazy_genexp(sub, scope_ctx):
                nested_lazy_nodes.append(sub)
                if sub.generators:
                    sub_stack.append(sub.generators[0].iter)
                continue
        sub_stack.extend(ast.iter_child_nodes(sub))

    if not is_lazy and not nested_lazy_nodes:
        return set(), set()

    deferred_reads: Set[str] = set()
    if is_lazy:
        deferred_reads.update(
            _extract_expr_scope_free_reads(comp_node, include_header=False)
        )
    if nested_lazy_nodes:
        comp_all_free = _extract_expr_scope_free_reads(
            comp_node, include_header=True
        )
        for lazy_sub in nested_lazy_nodes:
            deferred_reads.update(
                _extract_expr_scope_free_reads(lazy_sub, include_header=False)
                & comp_all_free
            )

    lazy_muts = _extract_lazy_genexp_walrus_names(comp_node, scope_ctx)
    return deferred_reads, lazy_muts


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    mod_tree: Optional[ast.AST] = None,
) -> List[Tuple[str, Set[str], bool]]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    unit_mod = ast.Module(body=list(candidate_stmts), type_ignores=[])
    scope_ctx = _get_fn_scope_context_uncached(
        unit_mod, mod_tree=mod_tree, cache_parent_map=False
    )
    mutators: List[Tuple[str, Set[str], bool]] = []
    stack: List[ast.AST] = list(candidate_stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = _extract_closure_nonlocal_mutations(curr)
            if mutated:
                is_escaped_def = isinstance(curr, ast.ClassDef) or bool(
                    getattr(curr, "decorator_list", None)
                )
                mutators.append((curr.name, mutated, is_escaped_def))
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                w_names = _extract_comprehension_walrus_names(curr)
                if w_names:
                    mutators.append(("", w_names, True))
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
            stack.extend(ast.iter_child_nodes(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return mutators


class _SpanBounds(NamedTuple):
    """Line and column span boundaries for enclosing function cell analysis."""

    start: int
    end: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None


class _EnclosingFnCellInfo(NamedTuple):
    """Cell binding and mutation summary for an enclosing function relative to a unit span."""

    fn_locals: Set[str]
    pre_bound: Set[str]
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]]
    all_sibling_nonlocal_mutations: Set[str]
    alias_edges: Sequence[Tuple[str, Set[str]]] = ()


def _is_pos_before_span(
    line: Optional[int], col: Optional[int], span: _SpanBounds
) -> bool:
    """Checks whether a 1-based line and column precede the span start."""
    return bool(
        line is not None
        and line > 0
        and (
            line < span.start
            or (
                line == span.start
                and span.start_col is not None
                and col is not None
                and int(col) < int(span.start_col)
            )
        )
    )


def _is_pos_after_span(
    line: Optional[int], col: Optional[int], span: _SpanBounds
) -> bool:
    """Checks whether a 1-based line and column follow the span end."""
    return bool(
        line is not None
        and line > 0
        and (
            line > span.end
            or (
                line == span.end
                and span.end_col is not None
                and col is not None
                and int(col) >= int(span.end_col)
            )
        )
    )


def _is_node_inside_span(node: ast.AST, span: _SpanBounds) -> bool:
    """Checks whether an AST node's start and end positions lie within span."""
    if not _is_ast_node_within_unit_bounds(
        node,
        span.start,
        span.end,
        u_end_col=span.end_col,
        u_start_col=span.start_col,
    ):
        return False
    c_end = getattr(node, "end_lineno", None)
    if isinstance(c_end, int):
        if c_end > span.end:
            return False
        if c_end == span.end and span.end_col is not None:
            end_col = getattr(node, "end_col_offset", None)
            if isinstance(end_col, int) and end_col > span.end_col:
                return False
    return True


def _strictly_encloses_span(node: ast.AST, span: _SpanBounds) -> bool:
    """Checks whether an AST compound node strictly encloses span."""
    c_line = getattr(node, "lineno", None)
    c_end = getattr(node, "end_lineno", c_line)
    if not (
        isinstance(c_line, int)
        and isinstance(c_end, int)
        and c_line <= span.start
        and span.end <= c_end
    ):
        return False
    if (c_line, c_end) != (span.start, span.end):
        return True
    c_col = getattr(node, "col_offset", None)
    if (
        span.start_col is not None
        and isinstance(c_col, int)
        and c_col < span.start_col
    ):
        return True
    c_end_col = getattr(node, "end_col_offset", None)
    if (
        span.end_col is not None
        and isinstance(c_end_col, int)
        and c_end_col > span.end_col
    ):
        return True
    return False


def _block_contains_span(stmts: Sequence[ast.AST], span: _SpanBounds) -> bool:
    """Checks whether any statement in stmts lies inside or strictly encloses span."""
    return any(
        _is_node_inside_span(s, span) or _strictly_encloses_span(s, span)
        for s in stmts
    )


def _can_bind_in_fn_scope(name: str, active_nonlocals: Optional[Set[str]]) -> bool:
    """Checks whether a binding in the current AST frame targets the enclosing function scope."""
    return active_nonlocals is None or name in active_nonlocals


def _extract_root_target_names(target: ast.AST) -> Set[str]:
    """Extracts root variable names written or mutated by an assignment target."""
    names: Set[str] = set()
    stack: List[ast.AST] = [target]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.Name):
            names.add(curr.id)
        elif isinstance(curr, (ast.Tuple, ast.List)):
            stack.extend(curr.elts)
        elif isinstance(curr, ast.Starred):
            stack.append(curr.value)
        elif isinstance(curr, (ast.Attribute, ast.Subscript)):
            stack.append(curr.value)
    return names


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions."""
    inline_reads: Set[str] = set()
    inline_muts: Set[str] = set()
    loaded_names: Set[str] = set()
    stack: List[ast.AST] = list(nodes)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(curr, ast.Lambda):
            inline_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            c_reads, c_muts = _extract_comprehension_deferred_closure_info(
                curr, scope_ctx
            )
            inline_reads.update(c_reads)
            inline_muts.update(c_muts)
            continue
        if isinstance(curr, ast.Call):
            if isinstance(curr.func, ast.Attribute):
                if not isinstance(curr.func.value, ast.Name):
                    stack.append(curr.func.value)
            elif not isinstance(curr.func, ast.Name):
                stack.append(curr.func)
            stack.extend(curr.args)
            stack.extend(
                kw.value for kw in curr.keywords if kw.value is not None
            )
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded_names.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return inline_reads, inline_muts, loaded_names


def _extract_closure_carrier_targets_and_values(
    node: ast.AST,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (target_root_names, value_nodes) from assignments or mutating method calls."""
    if isinstance(node, ast.Assign) and node.value is not None:
        targets: Set[str] = set()
        for tgt in node.targets:
            targets.update(_extract_root_target_names(tgt))
        return targets, (node.value,)
    if (
        isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        and node.value is not None
    ):
        return _extract_root_target_names(node.target), (node.value,)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        val_nodes: Sequence[ast.AST] = [
            *node.args,
            *(kw.value for kw in node.keywords if kw.value is not None),
        ]
        if val_nodes:
            return _extract_root_target_names(node.func.value), val_nodes
    return None


class _CellBindingState(NamedTuple):
    """Mutable sets tracking cell bindings during enclosing function AST traversal."""

    pre_bound: Set[str]
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]


class _StackFrame(NamedTuple):
    """Traversal frame for _analyze_enclosing_fn_cells."""

    node: ast.AST
    is_branched: bool
    in_loop: bool
    active_nonlocals: Optional[Set[str]]


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
        n for n in names if _can_bind_in_fn_scope(n, frame.active_nonlocals)
    ]
    if not filtered:
        return
    if _is_pos_before_span(line, col, span):
        state.pre_mutated.update(filtered)
        if not frame.is_branched:
            state.pre_bound.update(filtered)
    if _is_pos_after_span(line, col, span):
        state.post_mutated.update(filtered)
    if frame.in_loop:
        state.loop_mutated.update(filtered)


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


def _find_non_call_loaded_names(stmts: Sequence[ast.stmt]) -> Set[str]:
    """Finds variable names loaded outside direct Call.func positions at unit top level."""
    loaded: Set[str] = set()
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Call):
            if not isinstance(curr.func, ast.Name):
                stack.append(curr.func)
            stack.extend(curr.args)
            stack.extend(curr.keywords)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return loaded


def _propagate_sibling_closures(
    closures: Sequence[Tuple[Optional[str], Set[str], Set[str]]],
    post_reads: Optional[Set[str]] = None,
    alias_edges: Sequence[Tuple[str, Set[str]]] = (),
) -> List[Tuple[Optional[str], Set[str], Set[str]]]:
    """Propagates free reads and nonlocal mutations transitively across named sibling closures."""
    result: List[Tuple[Optional[str], Set[str], Set[str]]] = [
        (name, set(reads), set(muts)) for name, reads, muts in closures
    ]
    by_name: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for name, reads, muts in result:
        if name is not None:
            if name not in by_name:
                by_name[name] = (set(reads), set(muts))
            else:
                by_name[name][0].update(reads)
                by_name[name][1].update(muts)
    changed = True
    while changed:
        changed = False
        for name, reads, muts in result:
            for ref in list(reads):
                target = by_name.get(ref)
                if target is None:
                    continue
                t_reads, t_muts = target
                if not t_reads.issubset(reads):
                    reads.update(t_reads)
                    changed = True
                if not t_muts.issubset(muts):
                    muts.update(t_muts)
                    changed = True
            if name is not None:
                by_name[name][0].update(reads)
                by_name[name][1].update(muts)
        for tgt_name, src_names in alias_edges:
            for src in src_names:
                target = by_name.get(src)
                if target is None:
                    continue
                t_reads, t_muts = target
                if tgt_name not in by_name:
                    by_name[tgt_name] = (set(t_reads), set(t_muts))
                    result.append((tgt_name, set(t_reads), set(t_muts)))
                    changed = True
                else:
                    if not t_reads.issubset(by_name[tgt_name][0]):
                        by_name[tgt_name][0].update(t_reads)
                        changed = True
                    if not t_muts.issubset(by_name[tgt_name][1]):
                        by_name[tgt_name][1].update(t_muts)
                        changed = True
        for name, reads, muts in result:
            if name is not None and name in by_name:
                reads.update(by_name[name][0])
                muts.update(by_name[name][1])
    if post_reads is not None:
        for ref in list(post_reads):
            target = by_name.get(ref)
            if target is not None:
                post_reads.update(target[0])
    return result


def _push_seq(
    stack: List[_StackFrame],
    nodes: Sequence[ast.AST],
    is_branched: bool,
    in_loop: bool,
    active_nonlocals: Optional[Set[str]],
) -> None:
    """Pushes a sequence of AST nodes in reverse order onto the traversal stack."""
    for node in reversed(nodes):
        stack.append(_StackFrame(node, is_branched, in_loop, active_nonlocals))


def _push_compound_children(
    curr: ast.AST,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
) -> bool:
    """Pushes child nodes of compound control-flow statements with branch and loop state."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(stack, curr.body, is_br if body_span else True, in_lp, act_nl)
        stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, (ast.For, ast.AsyncFor)):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        if body_span:
            state.loop_mutated.update(unit_mutations)
            t_stores = [
                n
                for n in _extract_store_names_in_node(curr.target)
                if _can_bind_in_fn_scope(n, act_nl)
            ]
            state.pre_mutated.update(t_stores)
            if not is_br:
                state.pre_bound.update(t_stores)
            state.loop_mutated.update(t_stores)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(
            stack,
            curr.body,
            is_br if body_span else True,
            True if body_span else in_lp,
            act_nl,
        )
        stack.append(
            _StackFrame(
                curr.target,
                is_br if body_span else True,
                True if body_span else in_lp,
                act_nl,
            )
        )
        stack.append(_StackFrame(curr.iter, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.While):
        body_span = encloses and (
            _block_contains_span(curr.body, span)
            or _is_node_inside_span(curr.test, span)
        )
        else_span = encloses and _block_contains_span(curr.orelse, span)
        if body_span:
            state.loop_mutated.update(unit_mutations)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(
            stack,
            curr.body,
            is_br if body_span else True,
            True if body_span else in_lp,
            act_nl,
        )
        stack.append(
            _StackFrame(curr.test, is_br, True if body_span else in_lp, act_nl)
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
                        if _can_bind_in_fn_scope(n, act_nl)
                    ]
                    state.pre_mutated.update(w_stores)
                    if not is_br:
                        state.pre_bound.update(w_stores)
        _push_seq(stack, curr.body, is_br if body_span else True, in_lp, act_nl)
        _push_seq(stack, curr.items, is_br if body_span else True, in_lp, act_nl)
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
        _push_seq(
            stack,
            final_l,
            is_br if (not encloses or final_span) else True,
            in_lp,
            act_nl,
        )
        _push_seq(stack, else_l, is_br if else_span else True, in_lp, act_nl)
        _push_seq(stack, handlers_l, is_br if encloses else True, in_lp, act_nl)
        _push_seq(stack, body_l, is_br if body_span else True, in_lp, act_nl)
        return True

    if isinstance(curr, ast.ExceptHandler):
        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)
        h_br = is_br if encloses else True
        if isinstance(curr.name, str) and _can_bind_in_fn_scope(curr.name, act_nl):
            if encloses:
                state.pre_mutated.add(curr.name)
                state.post_mutated.add(curr.name)
                if not h_br:
                    state.pre_bound.add(curr.name)
            elif _is_pos_before_span(c_line, c_col, span):
                state.pre_bound.discard(curr.name)
                state.pre_mutated.add(curr.name)
            if _is_pos_after_span(c_line, c_col, span):
                state.post_mutated.add(curr.name)
            if in_lp:
                state.loop_mutated.add(curr.name)
        _push_seq(stack, curr.body, h_br, in_lp, act_nl)
        if curr.type is not None:
            stack.append(_StackFrame(curr.type, True, in_lp, act_nl))
        return True

    if type(curr).__name__ == "Match":
        for case in reversed(getattr(curr, "cases", [])):
            case_body = getattr(case, "body", [])
            c_br = (
                is_br
                if (encloses and _block_contains_span(case_body, span))
                else True
            )
            _push_seq(stack, case_body, c_br, in_lp, act_nl)
            guard = getattr(case, "guard", None)
            if isinstance(guard, ast.AST):
                stack.append(_StackFrame(guard, c_br, in_lp, act_nl))
            pattern = getattr(case, "pattern", None)
            if isinstance(pattern, ast.AST):
                stack.append(_StackFrame(pattern, c_br, in_lp, act_nl))
        subject = getattr(curr, "subject", None)
        if isinstance(subject, ast.AST):
            stack.append(_StackFrame(subject, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.BoolOp) and curr.values:
        _push_seq(stack, curr.values[1:], True, in_lp, act_nl)
        stack.append(_StackFrame(curr.values[0], is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.IfExp):
        stack.append(_StackFrame(curr.orelse, True, in_lp, act_nl))
        stack.append(_StackFrame(curr.body, True, in_lp, act_nl))
        stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl))
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
) -> bool:
    """Processes nested function, class, lambda, or comprehension scopes during cell analysis."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        estart = _node_earliest_start_lineno(curr)
        _record_binding_at_pos(
            (curr.name,), estart, c_col, span, frame, state
        )
        free_r = _extract_cached_scope_free_reads(curr)
        nl_mut = _extract_closure_nonlocal_mutations(curr)
        sib_name = (
            curr.name
            if _can_bind_in_fn_scope(curr.name, frame.active_nonlocals)
            else None
        )
        sibling_closures.append((sib_name, free_r, nl_mut))
        if _is_pos_after_span(estart, c_col, span):
            state.post_reads.update(free_r)
        if isinstance(curr, ast.ClassDef):
            _push_seq(
                stack,
                curr.body,
                frame.is_branched,
                frame.in_loop,
                _extract_direct_nonlocals(curr),
            )
        _push_seq(
            stack,
            list(iter_nested_scope_header_nodes(curr)),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
        )
        return True

    if isinstance(curr, ast.Lambda):
        free_r = _extract_expr_scope_free_reads(curr, include_header=False)
        sibling_closures.append((None, free_r, set()))
        if _is_pos_after_span(c_line, c_col, span):
            state.post_reads.update(free_r)
        _push_seq(
            stack,
            list(iter_nested_scope_header_nodes(curr)),
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
        )
        return True

    if isinstance(curr, _COMPREHENSION_TYPES):
        w_names = _extract_comprehension_walrus_names(curr)
        comp_frame = (
            frame._replace(is_branched=True)
            if _is_lazy_genexp(curr, scope_ctx)
            else frame
        )
        _record_binding_at_pos(w_names, c_line, c_col, span, comp_frame, state)
        c_reads, c_muts = _extract_comprehension_deferred_closure_info(
            curr, scope_ctx
        )
        if c_reads or c_muts:
            sibling_closures.append((None, c_reads, c_muts))
        if _is_pos_after_span(c_line, c_col, span):
            state.post_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=True)
            )
        return True

    return False


def _record_carrier_closures_and_aliases(
    curr: ast.AST,
    frame: _StackFrame,
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]],
    alias_edges: List[Tuple[str, Set[str]]],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> None:
    """Extracts inline closures and alias edges from assignment or mutating call nodes."""
    carrier = _extract_closure_carrier_targets_and_values(curr)
    if carrier is None:
        return
    c_targets, c_values = carrier
    valid_targets = [
        t
        for t in c_targets
        if _can_bind_in_fn_scope(t, frame.active_nonlocals)
    ]
    if not valid_targets:
        return
    c_reads, c_muts, c_loads = _extract_expr_closure_and_load_info(
        c_values, scope_ctx
    )
    for t_name in valid_targets:
        if c_reads or c_muts:
            sibling_closures.append((t_name, set(c_reads), set(c_muts)))
        if c_loads:
            alias_edges.append((t_name, set(c_loads)))


def _record_simple_node_bindings_and_reads(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
) -> None:
    """Records bindings, deletions, and post-span reads for leaf/simple AST nodes."""
    if isinstance(curr, ast.AugAssign) and isinstance(curr.target, ast.Name):
        if _is_pos_after_span(c_line, c_col, span):
            state.post_reads.add(curr.target.id)

    if isinstance(curr, ast.Name):
        if isinstance(curr.ctx, ast.Store):
            _record_binding_at_pos(
                (curr.id,), c_line, c_col, span, frame, state
            )
        elif isinstance(curr.ctx, ast.Del) and _can_bind_in_fn_scope(
            curr.id, frame.active_nonlocals
        ):
            if _is_pos_before_span(c_line, c_col, span):
                state.pre_bound.discard(curr.id)
                state.pre_mutated.add(curr.id)
            if _is_pos_after_span(c_line, c_col, span):
                state.post_mutated.add(curr.id)
            if frame.in_loop:
                state.loop_mutated.add(curr.id)
        elif isinstance(curr.ctx, ast.Load) and _is_pos_after_span(
            c_line, c_col, span
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
    fn_locals = set(scope_ctx.fn_locals)
    pre_bound: Set[str] = set(_extract_node_type_param_names(fn_node))
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args))

    state = _CellBindingState(
        pre_bound=pre_bound,
        pre_mutated=set(pre_bound),
        post_mutated=set(),
        post_reads=set(),
        loop_mutated=set(),
    )
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]] = []
    alias_edges: List[Tuple[str, Set[str]]] = []

    stack: List[_StackFrame] = [
        _StackFrame(stmt, False, False, None)
        for stmt in reversed(getattr(fn_node, "body", []))
    ]
    while stack:
        frame = stack.pop()
        curr = frame.node
        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)

        if curr is enclosed_fn and isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            estart = _node_earliest_start_lineno(curr)
            _record_binding_at_pos(
                (curr.name,), estart, c_col, span, frame, state
            )
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
                    if _is_pos_before_span(
                        getattr(hdr, "lineno", None),
                        getattr(hdr, "col_offset", None),
                        span,
                    )
                ]
                _push_seq(
                    stack,
                    pre_hdrs,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
            continue

        if _handle_nested_scope_node(
            curr,
            c_line,
            c_col,
            span,
            frame,
            state,
            sibling_closures,
            stack,
            scope_ctx,
        ):
            continue

        _record_carrier_closures_and_aliases(
            curr, frame, sibling_closures, alias_edges, scope_ctx
        )

        if _push_compound_children(
            curr, span, frame, unit_mutations, state, stack
        ):
            continue

        _record_simple_node_bindings_and_reads(
            curr, c_line, c_col, span, frame, state
        )

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
        )

    sibling_closures = _propagate_sibling_closures(
        sibling_closures,
        post_reads=state.post_reads,
        alias_edges=alias_edges,
    )
    all_sib_nl = (
        set().union(*(nl for _, _, nl in sibling_closures))
        if sibling_closures
        else set()
    )
    return _EnclosingFnCellInfo(
        fn_locals=fn_locals,
        pre_bound=state.pre_bound,
        pre_mutated=state.pre_mutated,
        post_mutated=state.post_mutated,
        post_reads=state.post_reads,
        loop_mutated=state.loop_mutated,
        sibling_closures=sibling_closures,
        all_sibling_nonlocal_mutations=all_sib_nl,
        alias_edges=alias_edges,
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
    chain_infos: List[_EnclosingFnCellInfo] = []
    for idx, (_estart, _nend, fn_node) in enumerate(enclosing_chain):
        if idx == len(enclosing_chain) - 1:
            span = _SpanBounds(u_start, u_end, u_start_col, u_end_col)
            u_mut: AbstractSet[str] = unit_mutations
            enc_child: Optional[ast.AST] = None
        else:
            next_start, next_end, next_fn = enclosing_chain[idx + 1]
            next_body = getattr(next_fn, "body", [])
            first_stmt = next_body[0] if next_body else next_fn
            span = _SpanBounds(
                int(getattr(first_stmt, "lineno", next_start) or next_start),
                next_end,
                getattr(first_stmt, "col_offset", None),
                getattr(next_fn, "end_col_offset", None),
            )
            u_mut = frozenset()
            enc_child = next_fn
        chain_infos.append(
            _analyze_enclosing_fn_cells(
                fn_node,
                span,
                unit_mutations=u_mut,
                enclosed_fn=enc_child,
                mod_tree=mod_tree,
            )
        )
    return chain_infos


def _check_captured_cell_mutations(
    captured_candidates: AbstractSet[str],
    eff_input_set: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
) -> bool:
    """Checks 1 & 2: captured cell variable is unbound before unit or mutated outside unit."""
    for var in captured_candidates:
        acc_ext_muts: Set[str] = set()
        acc_pre_bound: Set[str] = set()
        for idx in range(len(chain_infos) - 1, -1, -1):
            info = chain_infos[idx]
            acc_ext_muts.update(
                info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            acc_pre_bound.update(info.pre_bound)
            if var not in info.fn_locals:
                continue
            if (
                var in eff_input_set and var not in acc_pre_bound
            ) or var in acc_ext_muts:
                return True
            break
    return False


def _check_intra_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    intra_nl_mutators: Sequence[Tuple[str, Set[str], bool]],
    unit_top_locals: AbstractSet[str],
    nonlocal_set: AbstractSet[str],
    intra_closure_reads: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    is_subroutine: bool = True,
) -> bool:
    """Check 3: intra-unit closure mutates an outer cell or escaping top-level variable."""
    if not intra_nl_mutators:
        return False
    all_chain_locals: Set[str] = (
        set().union(*(info.fn_locals for info in chain_infos))
        if chain_infos
        else set()
    )
    top_locals_set = set(unit_top_locals)
    for _mut_name, mutated_vars, _escaped_def in intra_nl_mutators:
        escaping_nl_muts = (mutated_vars - top_locals_set) | (
            mutated_vars & set(nonlocal_set)
        )
        if bool(escaping_nl_muts & all_chain_locals):
            return True

    if not is_subroutine:
        return False

    non_call_loads = _find_non_call_loaded_names(candidate_stmts)
    outer_touched: Set[str] = set()
    all_post_reads: Set[str] = set()
    all_sib_reads: Set[str] = set()
    for info in chain_infos:
        outer_touched.update(
            info.pre_mutated
            | info.post_mutated
            | info.loop_mutated
            | info.all_sibling_nonlocal_mutations
        )
        all_post_reads.update(info.post_reads)
        for _, sib_r, _ in info.sibling_closures:
            all_sib_reads.update(sib_r)
    for mut_fn_name, mutated_vars, is_escaped_def in intra_nl_mutators:
        mut_fn_escapes = (
            is_escaped_def
            or mut_fn_name in all_post_reads
            or mut_fn_name in all_sib_reads
            or mut_fn_name in intra_closure_reads
            or mut_fn_name in non_call_loads
        )
        for var in mutated_vars & top_locals_set:
            if var in outer_touched or (
                mut_fn_escapes
                and (var in all_post_reads or var in all_sib_reads)
            ):
                return True
    return False


def _check_sibling_closure_sharing(
    eff_input_set: AbstractSet[str],
    sub_mutated: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
) -> bool:
    """Check 4: sibling closure referenced by unit shares a mutated cell with the unit."""
    if not chain_infos:
        return False
    all_chain_locals: Set[str] = set().union(
        *(info.fn_locals for info in chain_infos)
    )
    all_sib_closures = _propagate_sibling_closures(
        [sib for info in chain_infos for sib in info.sibling_closures],
        alias_edges=[
            edge for info in chain_infos for edge in info.alias_edges
        ],
    )
    sub_touched = set(eff_input_set) | set(sub_mutated)
    for sib_name, sib_reads, sib_nl_muts in all_sib_closures:
        if sib_name is None or sib_name not in eff_input_set:
            continue
        if bool(sib_reads & all_chain_locals & set(sub_mutated)):
            return True
        if bool(sib_nl_muts & all_chain_locals & sub_touched):
            return True
    return False


class _UnitClosureHazardContext(NamedTuple):
    """Input context for checking whether extracting a unit severs closure cell bindings."""

    candidate_stmts: Sequence[ast.stmt]
    mod_tree: Optional[ast.AST]
    u_start: int
    u_end: int
    u_start_col: Optional[int]
    u_end_col: Optional[int]
    is_subroutine: bool
    unit_kind: Optional[str]
    inputs: Sequence[str]
    outputs: Sequence[str]
    stores: Sequence[str]
    deleted_names: AbstractSet[str]
    unit_top_locals: AbstractSet[str]
    free_vars: Sequence[str] = ()
    nonlocals: Sequence[str] = ()
    has_yield: bool = False
    is_async: bool = False


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings."""
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return False
    if not ctx.is_subroutine and ctx.unit_kind != "closure":
        return False

    enclosing_chain = _find_enclosing_function_chain(
        ctx.mod_tree, ctx.u_start, ctx.u_end, ctx.u_start_col
    )
    if not enclosing_chain:
        return False

    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    nonlocal_set = set(ctx.nonlocals)
    unit_mutated_names = store_set | set(ctx.deleted_names)
    if not ctx.is_subroutine and bool(nonlocal_set & unit_mutated_names):
        return True

    enc_nonlocals, enc_outer_globals = _extract_scope_nonlocals_and_outer_globals(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_end_col=ctx.u_end_col,
        u_start_col=ctx.u_start_col,
    )
    enc_escaping = enc_nonlocals | enc_outer_globals
    if ctx.is_subroutine and bool(unit_mutated_names & enc_outer_globals):
        return True
    intra_nl_mutators = _collect_unit_nonlocal_mutators(
        ctx.candidate_stmts, mod_tree=ctx.mod_tree
    )
    if ctx.is_subroutine and (ctx.has_yield or ctx.is_async):
        if bool(unit_mutated_names & (enc_escaping | nonlocal_set)) or bool(
            intra_nl_mutators
        ):
            return True

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    unit_closures = _collect_scope_closures_uncached(
        unit_mod,
        module_tree=ctx.mod_tree,
        genexp_include_header=False,
        cache_parent_map=False,
    )
    intra_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures))
        if unit_closures
        else set()
    )
    intra_closure_reads.update(nonlocal_set)
    if ctx.is_subroutine:
        intra_closure_reads.update(enc_escaping & unit_mutated_names)
        if ctx.has_yield or ctx.is_async:
            intra_closure_reads.update(ctx.inputs)
    else:
        intra_closure_reads.update(ctx.free_vars)

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    unit_mutations = unit_mutated_names if ctx.is_subroutine else set()

    if not intra_closure_reads and not intra_nl_mutators and not ctx.is_subroutine:
        return False

    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        ctx.u_start_col,
        ctx.u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
    )

    if _check_intra_unit_nonlocal_mutators(
        ctx.candidate_stmts,
        intra_nl_mutators,
        ctx.unit_top_locals,
        nonlocal_set,
        intra_closure_reads,
        chain_infos,
        is_subroutine=ctx.is_subroutine,
    ):
        return True

    captured_candidates = intra_closure_reads & (
        eff_input_set | ((store_set | output_set) if ctx.is_subroutine else set())
    )
    if _check_captured_cell_mutations(
        captured_candidates, eff_input_set, chain_infos
    ):
        return True

    sub_mutated = (
        unit_mutated_names
        if ctx.is_subroutine
        else (nonlocal_set & unit_mutated_names)
    )
    if _check_sibling_closure_sharing(
        eff_input_set, sub_mutated, chain_infos
    ):
        return True

    return False
