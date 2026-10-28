"""Enclosing function scope context, column bounds, and closure carrier analysis."""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    Any,
    Dict,
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
    _extract_comprehension_walrus_names,
    _extract_scope_local_names,
    _extract_strict_fn_local_names,
    _has_wildcard_star_import,
    _is_eagerly_consumed_genexp,
    _is_span_in_scope_header,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _is_ast_node_within_unit_bounds,
    _record_import_names,
)
from pydoppelgangerhunt.fixer.source import _get_tree_scope_intervals
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _FnScopeContext(NamedTuple):
    """Cached per-function lexical scope metadata for eager generator expression checks."""

    fn_locals: frozenset[str]
    parent_map: Dict[ast.AST, ast.AST]
    shadowed_names: frozenset[str]
    has_star_import: bool


def _get_fn_scope_context_uncached(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    *,
    cache_parent_map: bool = True,
) -> _FnScopeContext:
    """Computes lexical scope metadata for a function or ephemeral unit module without caching."""
    return _FnScopeContext(
        fn_locals=frozenset(_extract_strict_fn_local_names(scope_node)),
        parent_map=_get_scope_parent_map(scope_node, use_cache=cache_parent_map),
        shadowed_names=frozenset(
            _extract_scope_local_names(
                scope_node,
                module_tree=mod_tree,
                cache_scope_node=cache_parent_map,
            )
        ),
        has_star_import=_has_wildcard_star_import(
            scope_node,
            module_tree=mod_tree,
            cache_scope_node=cache_parent_map,
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
        set(scope_ctx.shadowed_names),
        has_star_import=scope_ctx.has_star_import,
    )


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing FunctionDef/AsyncFunctionDef nodes ordered outermost to innermost."""
    chain: List[Tuple[int, int, int, ast.AST]] = []
    for estart, n_end, _span, node in _get_tree_scope_intervals(mod_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        n_lineno = int(getattr(node, "lineno", estart) or estart)
        if n_lineno <= 0 or u_start < n_lineno or u_end > n_end:
            continue
        if _is_span_in_scope_header(
            node,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        ):
            continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
        n_end_col = getattr(node, "end_col_offset", None)
        if (
            u_end == n_end
            and u_end_col is not None
            and isinstance(n_end_col, int)
            and u_end_col > n_end_col
        ):
            continue
        if u_start in (estart, n_lineno):
            _, e_col = _node_earliest_start_pos(node)
            bound_col = (
                e_col
                if (u_start == estart and e_col is not None)
                else n_col
            )
            if u_start_col is not None and bound_col > u_start_col:
                continue
            if u_end == n_end and (
                u_start_col is None
                or (
                    bound_col == u_start_col
                    and (
                        u_end_col is None
                        or not isinstance(n_end_col, int)
                        or u_end_col == n_end_col
                    )
                )
            ):
                continue
        chain.append((estart, n_end, n_col, node))
    chain.sort(key=lambda item: (item[0], -item[1], item[2]))
    return [(estart, n_end, node) for estart, n_end, _col, node in chain]


def _stmt_dump_set(stmts: Sequence[ast.AST], idx: int) -> Set[str]:
    """Builds the set of AST dumps for a boundary statement (and inner Expr value)."""
    if not stmts:
        return set()
    target = stmts[idx]
    dumps = {ast.dump(target)}
    if isinstance(target, ast.Expr):
        dumps.add(ast.dump(target.value))
    return dumps


def _pick_inferred_col(
    explicit_col: Optional[int],
    matches: Sequence[Tuple[int, int]],
    span_stmts: Sequence[Tuple[int, int]],
    stmt_idx: int,
    opposite_col: Optional[int] = None,
) -> Optional[int]:
    """Selects explicit_col, unique best dump match column, or unique statement column."""
    if explicit_col is not None:
        return explicit_col
    if matches:
        best_rank = min(m[0] for m in matches)
        best_cols = {m[1] for m in matches if m[0] == best_rank}
        if len(best_cols) == 1:
            return next(iter(best_cols))
        return None
    if opposite_col is not None:
        opp_idx = 1 - stmt_idx
        matched_stmts = [
            s[stmt_idx] for s in span_stmts if s[opp_idx] == opposite_col
        ]
        if len(matched_stmts) == 1:
            return matched_stmts[0]
    if len(span_stmts) == 1:
        return span_stmts[0][stmt_idx]
    return None


def _resolve_unit_col_bounds(
    candidate_stmts: Sequence[ast.AST],
    mod_tree: Optional[ast.AST],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int],
    u_end_col: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """Infers missing (u_start_col, u_end_col) from candidate_stmts and mod_tree."""
    if (
        (u_start_col is not None and u_end_col is not None)
        or mod_tree is None
        or not candidate_stmts
    ):
        return u_start_col, u_end_col
    first_dumps = (
        _stmt_dump_set(candidate_stmts, 0) if u_start_col is None else set()
    )
    last_dumps = (
        _stmt_dump_set(candidate_stmts, -1) if u_end_col is None else set()
    )
    start_matches: List[Tuple[int, int]] = []
    end_matches: List[Tuple[int, int]] = []
    span_stmts: List[Tuple[int, int]] = []
    single_stmt = len(candidate_stmts) == 1
    walk_stack: List[ast.AST] = [mod_tree]
    while walk_stack:
        node = walk_stack.pop()
        n_start, n_col = _node_earliest_start_pos(node)
        n_end = getattr(node, "end_lineno", n_start)
        n_end_col = getattr(node, "end_col_offset", None)
        if isinstance(n_start, int) and isinstance(n_end, int):
            if n_end < u_start or n_start > u_end:
                continue
            walk_stack.extend(reversed(list(ast.iter_child_nodes(node))))
            if (
                isinstance(node, ast.stmt)
                and n_start == u_start
                and n_end == u_end
                and isinstance(n_col, int)
                and isinstance(n_end_col, int)
            ):
                span_stmts.append((n_col, n_end_col))
            if not (u_start <= n_start and n_end <= u_end):
                continue
            if (
                u_start_col is not None
                and n_start == u_start
                and isinstance(n_col, int)
                and n_col < u_start_col
            ):
                continue
            if (
                u_end_col is not None
                and n_end == u_end
                and isinstance(n_end_col, int)
                and n_end_col > u_end_col
            ):
                continue
            need_start = (
                bool(first_dumps)
                and n_start == u_start
                and isinstance(n_col, int)
            )
            need_end = (
                bool(last_dumps)
                and n_end == u_end
                and isinstance(n_end_col, int)
            )
            if need_start or need_end:
                dumped = ast.dump(node)
                base_rank = 1 if isinstance(node, ast.Expr) else 0
                for (
                    need_flag,
                    col_val,
                    dumps,
                    opp_col,
                    opp_n_ln,
                    opp_u_ln,
                    opp_n_col,
                    matches_out,
                ) in (
                    (
                        need_start,
                        n_col,
                        first_dumps,
                        u_end_col,
                        n_end,
                        u_end,
                        n_end_col,
                        start_matches,
                    ),
                    (
                        need_end,
                        n_end_col,
                        last_dumps,
                        u_start_col,
                        n_start,
                        u_start,
                        n_col,
                        end_matches,
                    ),
                ):
                    if (
                        need_flag
                        and isinstance(col_val, int)
                        and dumped in dumps
                    ):
                        rank = (
                            base_rank - 2
                            if (
                                single_stmt
                                and opp_col is not None
                                and opp_n_ln == opp_u_ln
                                and opp_n_col == opp_col
                            )
                            else base_rank
                        )
                        matches_out.append((rank, col_val))
        else:
            walk_stack.extend(reversed(list(ast.iter_child_nodes(node))))

    return (
        _pick_inferred_col(
            u_start_col, start_matches, span_stmts, 0, opposite_col=u_end_col
        ),
        _pick_inferred_col(
            u_end_col, end_matches, span_stmts, 1, opposite_col=u_start_col
        ),
    )


def _extract_direct_nonlocals(
    scope_node: ast.AST,
    *,
    is_global: bool = False,
) -> Set[str]:
    """Extracts variable names declared nonlocal (or global) directly in scope_node."""
    names: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, (ast.Global, ast.Nonlocal)):
            if is_global == isinstance(curr, ast.Global):
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


def _extract_closure_nonlocal_mutations_and_deletions_uncached(
    scope_node: ast.AST,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (mutated_or_deleted, deleted) outer-scope variables via nonlocal in scope_node."""
    own_locals = (
        set()
        if isinstance(scope_node, ast.ClassDef)
        else _extract_strict_fn_local_names(scope_node)
    )
    direct_nonlocals: Set[str] = set()
    direct_mutated: Set[str] = set()
    direct_deleted: Set[str] = set()
    nested_mutated: Set[str] = set()
    nested_deleted: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            direct_mutated.add(curr.name)
            sub_mut, sub_del = _extract_closure_nonlocal_mutations_and_deletions(
                curr
            )
            nested_mutated.update(sub_mut - own_locals)
            nested_deleted.update(sub_del - own_locals)
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
            if isinstance(curr.ctx, ast.Del):
                direct_deleted.add(curr.id)
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            _record_import_names(curr, direct_mutated)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            direct_mutated.add(curr.name)
            direct_deleted.add(curr.name)
        else:
            bound_name = _extract_type_alias_or_pattern_name(curr)
            if bound_name is not None:
                direct_mutated.add(bound_name)
        stack.extend(ast.iter_child_nodes(curr))
    return (
        frozenset((direct_nonlocals & direct_mutated) | nested_mutated),
        frozenset((direct_nonlocals & direct_deleted) | nested_deleted),
    )


def _extract_closure_nonlocal_mutations_and_deletions(
    scope_node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Extracts and caches (mutated_or_deleted, deleted) nonlocal variables in scope_node."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set(), set()
    muts, dels = cast(
        Tuple[frozenset[str], frozenset[str]],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (scope_node, "nl_muts_dels"),
            lambda: _extract_closure_nonlocal_mutations_and_deletions_uncached(
                scope_node
            ),
        ),
    )
    return set(muts), set(dels)


def _extract_closure_nonlocal_mutations(scope_node: ast.AST) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    return _extract_closure_nonlocal_mutations_and_deletions(scope_node)[0]


def _extract_cached_scope_free_reads(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> Set[str]:
    """Extracts and caches free variable reads escaping a nested function or class."""
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (node, "free_reads"),
            lambda: frozenset(_extract_nested_scope_free_reads(node)),
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


def _extract_comprehension_body_walrus_names(comp_node: ast.AST) -> Set[str]:
    """Extracts walrus target names inside a comprehension excluding generators[0].iter."""
    return _extract_comprehension_walrus_names(
        comp_node, include_first_iter=False
    )


def _extract_lazy_genexp_walrus_names(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts walrus target names inside lazy GeneratorExp subtrees of node."""
    names: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                names.update(_extract_comprehension_body_walrus_names(curr))
                if curr.generators:
                    stack.append(curr.generators[0].iter)
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


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


def _iter_comprehension_body_exprs_and_targets(
    comp_node: Union[
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp
    ],
) -> Tuple[List[ast.AST], Set[str]]:
    """Extracts (non-header child expressions, comprehension target names)."""
    comp_targets = {
        n
        for gen in comp_node.generators
        for n in _extract_store_names_in_node(gen.target)
    }
    comp_exprs: List[ast.AST] = [
        ch
        for ch in ast.iter_child_nodes(comp_node)
        if not isinstance(ch, ast.comprehension)
    ]
    for idx, gen in enumerate(comp_node.generators):
        if idx > 0:
            comp_exprs.append(gen.iter)
        comp_exprs.extend(gen.ifs)
    return comp_exprs, comp_targets


def _extract_comprehension_deferred_closure_info(
    comp_node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (deferred_free_reads, lazy_walrus_mutations) from a comprehension node."""
    is_lazy = _is_lazy_genexp(comp_node, scope_ctx)
    first_iter = comp_node.generators[0].iter if comp_node.generators else None
    init_exprs, init_targets = _iter_comprehension_body_exprs_and_targets(
        comp_node
    )
    base_targets = frozenset(init_targets)
    nested_lazy_nodes: List[
        Tuple[Union[ast.Lambda, ast.GeneratorExp], frozenset[str]]
    ] = []
    if is_lazy:
        sub_stack: List[Tuple[ast.AST, frozenset[str]]] = (
            [(first_iter, frozenset())] if first_iter is not None else []
        )
    else:
        sub_stack = [(expr, base_targets) for expr in init_exprs]
        if first_iter is not None:
            sub_stack.append((first_iter, frozenset()))
    while sub_stack:
        sub, shadowed_targets = sub_stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sub_stack.extend(
                (hdr, shadowed_targets)
                for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, ast.Lambda):
            nested_lazy_nodes.append((sub, shadowed_targets))
            sub_stack.extend(
                (hdr, shadowed_targets)
                for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, _COMPREHENSION_TYPES):
            if isinstance(sub, ast.GeneratorExp) and _is_lazy_genexp(
                sub, scope_ctx
            ):
                nested_lazy_nodes.append((sub, shadowed_targets))
                if sub.generators:
                    sub_stack.append((sub.generators[0].iter, shadowed_targets))
                continue
            sub_exprs, sub_targets = _iter_comprehension_body_exprs_and_targets(
                sub
            )
            if sub.generators:
                sub_stack.append((sub.generators[0].iter, shadowed_targets))
            inner_shadowed = shadowed_targets | frozenset(sub_targets)
            sub_stack.extend((expr, inner_shadowed) for expr in sub_exprs)
            continue
        sub_stack.extend(
            (ch, shadowed_targets) for ch in ast.iter_child_nodes(sub)
        )

    if not is_lazy and not nested_lazy_nodes:
        return set(), set()

    deferred_reads: Set[str] = set()
    if is_lazy:
        deferred_reads.update(
            _extract_expr_scope_free_reads(comp_node, include_header=False)
        )
    if nested_lazy_nodes:
        for lazy_sub, shadowed_targets in nested_lazy_nodes:
            sub_reads = (
                _extract_expr_scope_free_reads(lazy_sub, include_header=False)
                - shadowed_targets
            )
            deferred_reads.update(sub_reads)

    lazy_muts = _extract_lazy_genexp_walrus_names(comp_node, scope_ctx)
    return deferred_reads, lazy_muts


class _IntraUnitMutator(NamedTuple):
    """Intra-unit closure or lazy generator expression mutating outer variables."""

    name: str
    mutated_vars: Set[str]
    is_escaped_def: bool
    node: Optional[ast.AST] = None


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    mod_tree: Optional[ast.AST] = None,
) -> List[_IntraUnitMutator]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    unit_mod = ast.Module(body=list(candidate_stmts), type_ignores=[])
    scope_ctx = _get_fn_scope_context_uncached(
        unit_mod, mod_tree=mod_tree, cache_parent_map=False
    )
    mutators: List[_IntraUnitMutator] = []
    stack: List[Tuple[ast.AST, frozenset[str]]] = [
        (stmt, frozenset()) for stmt in candidate_stmts
    ]
    while stack:
        curr, shadowed = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = _extract_closure_nonlocal_mutations(curr) - shadowed
            if mutated:
                is_escaped_def = (
                    isinstance(curr, ast.ClassDef)
                    or bool(getattr(curr, "decorator_list", None))
                    or _is_closure_factory_scope(curr, scope_ctx)
                )
                mutators.append(
                    _IntraUnitMutator(
                        curr.name, mutated, is_escaped_def, node=curr
                    )
                )
            own_locals = (
                frozenset()
                if isinstance(curr, ast.ClassDef)
                else frozenset(_extract_strict_fn_local_names(curr))
            )
            body_shadowed = (
                shadowed
                | own_locals
                | frozenset(_extract_direct_nonlocals(curr, is_global=True))
            )
            stack.extend(
                (ch, body_shadowed) for ch in getattr(curr, "body", [])
            )
            stack.extend(
                (hdr, shadowed) for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                w_names = (
                    _extract_comprehension_body_walrus_names(curr) - shadowed
                )
                if w_names:
                    mutators.append(
                        _IntraUnitMutator("", w_names, True, node=curr)
                    )
                if curr.generators:
                    stack.append((curr.generators[0].iter, shadowed))
                continue
            stack.extend((ch, shadowed) for ch in ast.iter_child_nodes(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(
                (hdr, shadowed) for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        stack.extend((ch, shadowed) for ch in ast.iter_child_nodes(curr))
    return mutators


class _SpanBounds(NamedTuple):
    """Line and column span boundaries for enclosing function cell analysis."""

    start: int
    end: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None


class _EnclosingFnCellInfo(NamedTuple):
    """Cell binding and mutation summary for an enclosing function relative to a unit span."""

    fn_locals: frozenset[str]
    pre_bound: frozenset[str]
    pre_mutated: frozenset[str]
    post_mutated: frozenset[str]
    post_reads: frozenset[str]
    loop_mutated: frozenset[str]
    sibling_closures: Tuple[
        Tuple[Optional[str], frozenset[str], frozenset[str]], ...
    ]
    all_sibling_nonlocal_mutations: frozenset[str]
    alias_edges: Tuple[Tuple[str, frozenset[str]], ...] = ()
    pre_unbound: frozenset[str] = frozenset()
    concurrent_sibling_vars: frozenset[str] = frozenset()


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
    return _node_strictly_encloses_bounds(
        node,
        span.start,
        span.end,
        u_start_col=span.start_col,
        u_end_col=span.end_col,
    )


def _node_contains_or_encloses_span(
    node: Optional[ast.AST], span: _SpanBounds
) -> bool:
    """Checks whether node lies inside or strictly encloses span."""
    return node is not None and (
        _is_node_inside_span(node, span) or _strictly_encloses_span(node, span)
    )


def _block_contains_span(stmts: Sequence[ast.AST], span: _SpanBounds) -> bool:
    """Checks whether any statement in stmts lies inside or strictly encloses span."""
    return any(_node_contains_or_encloses_span(s, span) for s in stmts)


def _can_bind_in_fn_scope(
    name: str,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str] = frozenset(),
) -> bool:
    """Checks whether a binding in the current AST frame targets the enclosing function scope."""
    if active_nonlocals is not None:
        return name in active_nonlocals
    return name not in fn_globals


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


_NON_CLOSURE_RETURN_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "divmod",
        "float",
        "format",
        "hash",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "round",
        "str",
        "sum",
    }
)


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions."""
    inline_reads: Set[str] = set()
    inline_muts: Set[str] = set()
    loaded_names: Set[str] = set()
    stack: List[ast.AST] = list(nodes)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
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
            is_lazy = _is_lazy_genexp(curr, scope_ctx)
            if curr.generators:
                stack.append(curr.generators[0].iter)
            if not is_lazy:
                comp_exprs, comp_targets = (
                    _iter_comprehension_body_exprs_and_targets(curr)
                )
                sub_calls: Set[str] = set()
                _, _, sub_loads = _extract_expr_closure_and_load_info(
                    comp_exprs, scope_ctx, called_names_out=sub_calls
                )
                loaded_names.update(sub_loads - comp_targets)
                if called_names_out is not None:
                    called_names_out.update(sub_calls - comp_targets)
            continue
        if isinstance(curr, ast.Call):
            is_non_closure_builtin = (
                isinstance(curr.func, ast.Name)
                and curr.func.id in _NON_CLOSURE_RETURN_BUILTINS
                and (
                    scope_ctx is None
                    or curr.func.id not in scope_ctx.shadowed_names
                )
            )
            if is_non_closure_builtin:
                _, sub_muts, _ = _extract_expr_closure_and_load_info(
                    [
                        *curr.args,
                        *(
                            kw.value
                            for kw in curr.keywords
                            if kw.value is not None
                        ),
                    ],
                    scope_ctx,
                    called_names_out=called_names_out,
                )
                inline_muts.update(sub_muts)
                continue
            if isinstance(curr.func, ast.Attribute):
                stack.append(curr.func.value)
            elif isinstance(curr.func, ast.Name):
                if called_names_out is not None:
                    called_names_out.add(curr.func.id)
            else:
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


def _extract_shallow_loads_and_calls(
    stmts: Sequence[ast.AST],
) -> Tuple[Set[str], Set[str]]:
    """Extracts (non_call_loaded_names, direct_called_names) stopping at nested def/class bodies."""
    loaded: Set[str] = set()
    called: Set[str] = set()
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            if isinstance(curr, ast.Lambda):
                loaded.update(
                    _extract_expr_scope_free_reads(curr, include_header=False)
                )
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Call):
            if isinstance(curr.func, ast.Name):
                called.add(curr.func.id)
            else:
                stack.append(curr.func)
            stack.extend(curr.args)
            stack.extend(curr.keywords)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return loaded, called


def _extract_shallow_defs(
    stmts: Sequence[ast.AST],
) -> Dict[str, List[ast.AST]]:
    """Extracts direct FunctionDef, AsyncFunctionDef, and ClassDef nodes in stmts."""
    defs: Dict[str, List[ast.AST]] = {}
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            defs.setdefault(curr.name, []).append(curr)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return defs


def _collect_escaped_intra_unit_defs(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Tuple[Set[str], Set[int]]:
    """Computes (escaped_names, escaped_def_node_ids) with lexical-scope definition tracking."""
    root_defs = _extract_shallow_defs(candidate_stmts)
    root_env: Dict[str, Tuple[ast.AST, ...]] = {
        name: tuple(nodes) for name, nodes in root_defs.items()
    }
    all_def_nodes: List[ast.AST] = []
    defs_by_name: Dict[str, List[ast.AST]] = {}
    direct_children: Dict[int, List[ast.AST]] = {}
    body_env: Dict[int, Dict[str, Tuple[ast.AST, ...]]] = {}
    shadowed_map: Dict[int, frozenset[str]] = {}

    scope_stack: List[
        Tuple[ast.AST, Dict[str, Tuple[ast.AST, ...]], frozenset[str]]
    ] = [
        (d_node, root_env, frozenset())
        for nodes in root_defs.values()
        for d_node in nodes
    ]
    while scope_stack:
        d_node, parent_env, shadowed_from_root = scope_stack.pop()
        nid = id(d_node)
        all_def_nodes.append(d_node)
        d_name = getattr(d_node, "name", "")
        if d_name:
            defs_by_name.setdefault(d_name, []).append(d_node)
        c_defs = _extract_shallow_defs(getattr(d_node, "body", ()))
        c_list = [n for nodes in c_defs.values() for n in nodes]
        direct_children[nid] = c_list
        if isinstance(d_node, ast.ClassDef):
            body_env[nid] = {
                **parent_env,
                **{k: tuple(v) for k, v in c_defs.items()},
            }
            shadowed_map[nid] = shadowed_from_root | frozenset(c_defs.keys())
            for c_node in c_list:
                scope_stack.append((c_node, parent_env, shadowed_from_root))
        else:
            own_locals = _extract_strict_fn_local_names(d_node)
            own_globals = _extract_direct_nonlocals(d_node, is_global=True)
            own_nonlocals = _extract_direct_nonlocals(d_node, is_global=False)
            fn_shadowed = shadowed_from_root | frozenset(
                (own_locals | own_globals) - own_nonlocals
            )
            cur_env: Dict[str, Tuple[ast.AST, ...]] = {
                k: v
                for k, v in parent_env.items()
                if k not in own_locals and k not in own_globals
            }
            for c_name, c_nodes in c_defs.items():
                if c_name in own_nonlocals:
                    cur_env[c_name] = parent_env.get(c_name, ()) + tuple(
                        c_nodes
                    )
                    if c_name not in shadowed_from_root:
                        root_env[c_name] = root_env.get(c_name, ()) + tuple(
                            c_nodes
                        )
                else:
                    cur_env[c_name] = tuple(c_nodes)
            body_env[nid] = cur_env
            shadowed_map[nid] = fn_shadowed
            for c_node in c_list:
                scope_stack.append((c_node, cur_env, fn_shadowed))

    top_loads, top_calls = _extract_shallow_loads_and_calls(candidate_stmts)
    escaped_top_names: Set[str] = set(initial_escaped) | top_loads
    executed_top_names: Set[str] = set(top_calls) | escaped_top_names
    escaped_def_ids: Set[int] = set()
    executed_def_ids: Set[int] = set()

    for name in escaped_top_names:
        for d_node in root_env.get(name, ()):
            escaped_def_ids.add(id(d_node))
            executed_def_ids.add(id(d_node))
    for name in executed_top_names:
        for d_node in root_env.get(name, ()):
            executed_def_ids.add(id(d_node))
    for nodes in root_defs.values():
        for d_node in nodes:
            if isinstance(d_node, ast.ClassDef) or getattr(
                d_node, "decorator_list", None
            ):
                executed_def_ids.add(id(d_node))

    body_info: Dict[int, Tuple[Set[str], Set[str]]] = {}
    changed = True
    while changed:
        changed = False
        for d_node in all_def_nodes:
            nid = id(d_node)
            is_esc = nid in escaped_def_ids
            is_exec = is_esc or (nid in executed_def_ids)
            if not is_exec:
                continue
            for c_node in direct_children.get(nid, ()):
                cid = id(c_node)
                if isinstance(d_node, ast.ClassDef) and is_esc:
                    if cid not in escaped_def_ids:
                        escaped_def_ids.add(cid)
                        executed_def_ids.add(cid)
                        changed = True
                elif isinstance(c_node, ast.ClassDef) or getattr(
                    c_node, "decorator_list", None
                ):
                    if cid not in executed_def_ids:
                        executed_def_ids.add(cid)
                        changed = True
            if nid not in body_info:
                body_info[nid] = _extract_shallow_loads_and_calls(
                    getattr(d_node, "body", ())
                )
            b_loads, b_calls = body_info[nid]
            env = body_env[nid]
            shadowed_here = shadowed_map[nid]
            for ref_name in b_loads | b_calls:
                if (
                    (is_esc or ref_name in b_loads)
                    and ref_name not in shadowed_here
                    and ref_name not in escaped_top_names
                ):
                    escaped_top_names.add(ref_name)
                    changed = True
                for target_node in env.get(ref_name, ()) + (
                    root_env.get(ref_name, ())
                    if ref_name not in shadowed_here
                    else ()
                ):
                    tid = id(target_node)
                    if (
                        is_esc or ref_name in b_loads
                    ) and tid not in escaped_def_ids:
                        escaped_def_ids.add(tid)
                        changed = True
                    if tid not in executed_def_ids:
                        executed_def_ids.add(tid)
                        changed = True

    unambiguous_escaped_def_names = {
        name
        for name, nodes in defs_by_name.items()
        if name and all(id(n) in escaped_def_ids for n in nodes)
    }
    return escaped_top_names | unambiguous_escaped_def_names, escaped_def_ids


def _collect_escaped_intra_unit_names(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Set[str]:
    """Computes function/variable names that escape candidate_stmts via loads or escaped closures."""
    return _collect_escaped_intra_unit_defs(candidate_stmts, initial_escaped)[0]


def _propagate_sibling_closures(
    closures: Sequence[Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]],
    post_reads: Optional[Set[str]] = None,
    alias_edges: Sequence[Tuple[str, AbstractSet[str]]] = (),
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


def _make_nested_fn_scope_context(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    own_locals: frozenset[str],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> _FnScopeContext:
    """Builds a _FnScopeContext for analyzing expressions inside a nested function."""
    return _FnScopeContext(
        fn_locals=(
            (scope_ctx.fn_locals if scope_ctx is not None else frozenset())
            | own_locals
        ),
        shadowed_names=(
            (scope_ctx.shadowed_names if scope_ctx is not None else frozenset())
            | own_locals
        ),
        parent_map=(
            scope_ctx.parent_map
            if scope_ctx is not None
            else _get_scope_parent_map(fn_node)
        ),
        has_star_import=(
            scope_ctx.has_star_import if scope_ctx is not None else False
        ),
    )


def _propagate_local_closure_syms(
    local_closure_syms: Set[str],
    local_edges: Sequence[Tuple[str, Set[str]]],
) -> Set[str]:
    """Propagates local closure symbol reachability across local assignment/call edges."""
    if local_closure_syms and local_edges:
        changed = True
        while changed:
            changed = False
            for t_name, refs in local_edges:
                if t_name not in local_closure_syms and (
                    refs & local_closure_syms
                ):
                    local_closure_syms.add(t_name)
                    changed = True
    return local_closure_syms


def _is_closure_factory_scope_uncached(
    scope_node: ast.FunctionDef,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Computes whether a FunctionDef can return a closure or generator without caching."""
    own_locals = frozenset(_extract_strict_fn_local_names(scope_node))
    local_ctx = _make_nested_fn_scope_context(scope_node, own_locals, scope_ctx)
    has_nested_closure = False
    local_closure_syms: Set[str] = set()
    local_edges: List[Tuple[str, Set[str]]] = []
    ret_exprs: List[ast.AST] = []
    stack: List[ast.AST] = list(scope_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            has_nested_closure = True
            local_closure_syms.add(curr.name)
            continue
        if isinstance(curr, ast.Lambda):
            has_nested_closure = True
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, local_ctx):
                has_nested_closure = True
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
        carrier = _extract_closure_carrier_targets_and_values(curr)
        if carrier is not None:
            c_targets, c_values = carrier
            local_tgts = c_targets & own_locals
            if local_tgts:
                c_calls: Set[str] = set()
                c_reads, c_muts, c_loads = (
                    _extract_expr_closure_and_load_info(
                        c_values, local_ctx, called_names_out=c_calls
                    )
                )
                if c_reads or c_muts:
                    local_closure_syms.update(local_tgts)
                local_refs = (c_loads | c_calls) & own_locals
                if local_refs:
                    for t_name in local_tgts:
                        local_edges.append((t_name, local_refs))
        if isinstance(curr, ast.Return) and curr.value is not None:
            ret_exprs.append(curr.value)
        stack.extend(ast.iter_child_nodes(curr))
    if not has_nested_closure or not ret_exprs:
        return False
    _propagate_local_closure_syms(local_closure_syms, local_edges)
    ret_calls: Set[str] = set()
    r_reads, r_muts, r_loads = _extract_expr_closure_and_load_info(
        ret_exprs, local_ctx, called_names_out=ret_calls
    )
    return bool(
        r_reads
        or r_muts
        or ((r_loads | ret_calls) & local_closure_syms)
    )


def _is_closure_factory_scope(
    scope_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Checks whether a nested function or class can return a closure or generator/coroutine."""
    if isinstance(scope_node, (ast.ClassDef, ast.AsyncFunctionDef)):
        return True
    if not isinstance(scope_node, ast.FunctionDef):
        return False
    cache_key: Any = (
        scope_node,
        "is_closure_factory",
        scope_ctx.shadowed_names if scope_ctx is not None else None,
        scope_ctx.has_star_import if scope_ctx is not None else False,
    )
    return bool(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _is_closure_factory_scope_uncached(scope_node, scope_ctx),
        )
    )


def _extract_fn_returned_closure_refs_uncached(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (returned_loaded_names, returned_called_names) for fn_node without caching."""
    own_locals = _extract_strict_fn_local_names(fn_node)
    ret_exprs: List[ast.AST] = []
    stack: List[ast.AST] = list(fn_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Return) and curr.value is not None:
            ret_exprs.append(curr.value)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    if not ret_exprs:
        return frozenset(), frozenset()
    ret_calls: Set[str] = set()
    _, _, ret_loads = _extract_expr_closure_and_load_info(
        ret_exprs, scope_ctx, called_names_out=ret_calls
    )
    return frozenset(ret_loads - own_locals), frozenset(ret_calls - own_locals)


def _extract_fn_returned_closure_refs(
    fn_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts non-local (returned_loaded_names, returned_called_names) for a nested function."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set(), set()
    cache_key: Any = (
        fn_node,
        "ret_closure_refs",
        scope_ctx.shadowed_names if scope_ctx is not None else None,
        scope_ctx.has_star_import if scope_ctx is not None else False,
    )
    cached_loads, cached_calls = cast(
        Tuple[frozenset[str], frozenset[str]],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _extract_fn_returned_closure_refs_uncached(
                fn_node, scope_ctx
            ),
        ),
    )
    return set(cached_loads), set(cached_calls)


def _extract_fn_outer_carrier_targets_uncached(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
) -> frozenset[str]:
    """Computes outer-scope container/variable names mutated by carrier operations in fn_node."""
    has_closure = any(
        isinstance(
            sub,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.GeneratorExp,
            ),
        )
        and sub is not fn_node
        for sub in ast.walk(fn_node)
    )
    if not has_closure:
        return frozenset()
    own_locals = _extract_strict_fn_local_names(fn_node)
    local_ctx = _make_nested_fn_scope_context(
        fn_node, frozenset(own_locals), scope_ctx
    )
    direct_nl = _extract_direct_nonlocals(fn_node)
    fn_g = set(fn_globals) | _extract_direct_nonlocals(fn_node, is_global=True)
    targets: Set[str] = set()
    local_closure_syms: Set[str] = set()
    local_edges: List[Tuple[str, Set[str]]] = []
    outer_candidates: List[Tuple[str, Set[str]]] = []
    stack: List[Tuple[ast.AST, Set[str], Set[str], Set[str]]] = [
        (stmt, own_locals, direct_nl, fn_g) for stmt in fn_node.body
    ]
    while stack:
        sub, cur_locals, cur_nl, cur_g = stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_closure_syms.add(sub.name)
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append((hdr, cur_locals, cur_nl, cur_g))
            sub_direct_nl = _extract_direct_nonlocals(sub)
            sub_g = cur_g | _extract_direct_nonlocals(sub, is_global=True)
            sub_own = (
                set()
                if isinstance(sub, ast.ClassDef)
                else _extract_strict_fn_local_names(sub)
            )
            sub_nl = (
                (sub_direct_nl & cur_nl)
                | (sub_direct_nl - cur_locals)
                | (cur_nl - (sub_own - sub_direct_nl) - sub_g)
            )
            next_locals = (cur_locals | sub_own) - sub_g
            for ch in sub.body:
                stack.append((ch, next_locals, sub_nl, sub_g))
            continue
        if isinstance(sub, ast.Lambda):
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append((hdr, cur_locals, cur_nl, cur_g))
            lam_args = _extract_arg_names(sub.args)
            lam_locals = cur_locals | lam_args
            stack.append(
                (sub.body, lam_locals, cur_nl - lam_args, cur_g - lam_args)
            )
            continue
        if isinstance(sub, _COMPREHENSION_TYPES):
            comp_exprs, comp_targets = _iter_comprehension_body_exprs_and_targets(
                sub
            )
            if sub.generators:
                stack.append((sub.generators[0].iter, cur_locals, cur_nl, cur_g))
            comp_locals = cur_locals | comp_targets
            comp_nl = cur_nl - comp_targets
            comp_g = cur_g - comp_targets
            for expr in comp_exprs:
                stack.append((expr, comp_locals, comp_nl, comp_g))
            continue
        carrier = _extract_closure_carrier_targets_and_values(sub)
        if carrier is not None:
            c_targets, c_values = carrier
            c_calls: Set[str] = set()
            c_reads, c_muts, c_loads = _extract_expr_closure_and_load_info(
                c_values, local_ctx, called_names_out=c_calls
            )
            has_inline = bool(c_reads or c_muts)
            val_refs = c_loads | c_calls
            store_names = (
                set()
                if isinstance(sub, ast.Call)
                else _extract_store_names_in_node(sub)
            )
            for t_name in c_targets:
                if t_name in cur_g:
                    continue
                is_outer_target = (
                    (t_name in cur_nl)
                    if t_name in store_names
                    else (t_name not in cur_locals or t_name in cur_nl)
                )
                if is_outer_target:
                    if has_inline:
                        targets.add(t_name)
                    elif val_refs:
                        outer_candidates.append((t_name, val_refs))
                else:
                    if has_inline:
                        local_closure_syms.add(t_name)
                    elif val_refs & cur_locals:
                        local_edges.append((t_name, val_refs & cur_locals))
        for child_node in ast.iter_child_nodes(sub):
            stack.append((child_node, cur_locals, cur_nl, cur_g))
    if outer_candidates:
        _propagate_local_closure_syms(local_closure_syms, local_edges)
        for t_name, val_refs in outer_candidates:
            if val_refs & local_closure_syms:
                targets.add(t_name)
    return frozenset(targets)


def _extract_fn_outer_carrier_targets(
    fn_node: ast.AST,
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts outer-scope container/variable names mutated by carrier operations in fn_node."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (
                fn_node,
                "outer_carriers",
                frozenset(fn_globals),
                scope_ctx.shadowed_names if scope_ctx is not None else None,
                scope_ctx.has_star_import if scope_ctx is not None else False,
            ),
            lambda: _extract_fn_outer_carrier_targets_uncached(
                fn_node, fn_globals, scope_ctx
            ),
        )
    )


def _iter_class_body_methods(
    cls_node: ast.ClassDef,
) -> List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    """Collects method definitions in cls_node.body without descending into nested scopes."""
    methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []
    stack: List[ast.AST] = list(reversed(cls_node.body))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(curr)
            continue
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(curr))))
    return methods
