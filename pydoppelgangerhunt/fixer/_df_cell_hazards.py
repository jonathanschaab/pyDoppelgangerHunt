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
from pydoppelgangerhunt.fixer.source import (
    _get_tree_scope_intervals,
)
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
        fn_locals=frozenset(
            _extract_single_node_local_names(
                scope_node,
                include_globals=False,
                include_nonlocals=False,
                include_self_name=False,
                include_bare_annotations=True,
            )
        ),
        parent_map=_get_scope_parent_map(scope_node, use_cache=cache_parent_map),
        shadowed_names=frozenset(
            _extract_scope_local_names(scope_node, module_tree=mod_tree)
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
        set(scope_ctx.shadowed_names),
        has_star_import=scope_ctx.has_star_import,
    )


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
        n_lineno = int(getattr(node, "lineno", estart) or estart)
        if n_lineno <= 0 or u_start < n_lineno or u_end > n_end:
            continue
        body = getattr(node, "body", None)
        if body:
            body_start, body_col = _node_earliest_start_pos(body[0])
            if body_start is not None and (
                u_end < body_start
                or (
                    u_start < body_start
                    and (
                        u_start <= n_lineno
                        or any(
                            (getattr(h, "lineno", 0) or 0)
                            <= u_start
                            <= (getattr(h, "end_lineno", 0) or 0)
                            for h in (
                                *iter_nested_scope_header_nodes(node),
                                *getattr(
                                    getattr(node, "args", None),
                                    "posonlyargs",
                                    (),
                                ),
                                *getattr(
                                    getattr(node, "args", None), "args", ()
                                ),
                                *getattr(
                                    getattr(node, "args", None),
                                    "kwonlyargs",
                                    (),
                                ),
                            )
                        )
                    )
                )
                or (
                    u_start == body_start
                    and u_start_col is not None
                    and body_col is not None
                    and u_start_col < body_col
                )
            ):
                continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
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
                u_start_col is None or bound_col == u_start_col
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
    nested_lazy_nodes: List[Tuple[Union[ast.Lambda, ast.GeneratorExp], bool]] = []
    sub_stack: List[Tuple[ast.AST, bool]] = (
        [(first_iter, True)]
        if (is_lazy and first_iter is not None)
        else [
            (ch, ch is first_iter)
            for ch in ast.iter_child_nodes(comp_node)
        ]
    )
    while sub_stack:
        sub, in_hdr = sub_stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sub_stack.extend(
                (hdr, in_hdr) for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, ast.Lambda):
            nested_lazy_nodes.append((sub, in_hdr))
            sub_stack.extend(
                (hdr, in_hdr) for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, ast.GeneratorExp):
            if _is_lazy_genexp(sub, scope_ctx):
                nested_lazy_nodes.append((sub, in_hdr))
                if sub.generators:
                    sub_stack.append((sub.generators[0].iter, in_hdr))
                continue
        sub_stack.extend(
            (
                ch,
                in_hdr
                or (
                    isinstance(sub, ast.comprehension)
                    and bool(comp_node.generators)
                    and sub is comp_node.generators[0]
                    and ch is first_iter
                ),
            )
            for ch in ast.iter_child_nodes(sub)
        )

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
        _, comp_targets = _iter_comprehension_body_exprs_and_targets(comp_node)
        for lazy_sub, in_hdr in nested_lazy_nodes:
            sub_reads = _extract_expr_scope_free_reads(
                lazy_sub, include_header=False
            )
            if not in_hdr:
                sub_reads -= comp_targets
            deferred_reads.update(sub_reads & comp_all_free)

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
                is_escaped_def = (
                    isinstance(curr, ast.ClassDef)
                    or bool(getattr(curr, "decorator_list", None))
                    or _is_closure_factory_scope(curr, scope_ctx)
                )
                mutators.append((curr.name, mutated, is_escaped_def))
            stack.extend(getattr(curr, "body", []))
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                w_names = _extract_comprehension_body_walrus_names(curr)
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


def _block_contains_span(stmts: Sequence[ast.AST], span: _SpanBounds) -> bool:
    """Checks whether any statement in stmts lies inside or strictly encloses span."""
    return any(
        _is_node_inside_span(s, span) or _strictly_encloses_span(s, span)
        for s in stmts
    )


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


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


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
        if isinstance(curr, ast.Compare):
            stack.append(curr.left)
            stack.extend(curr.comparators)
            continue
        if isinstance(curr, ast.UnaryOp):
            stack.append(curr.operand)
            continue
        if isinstance(curr, ast.JoinedStr):
            for val in curr.values:
                if isinstance(val, ast.FormattedValue):
                    stack.append(val.value)
                    if val.format_spec is not None:
                        stack.append(val.format_spec)
            continue
        if isinstance(curr, ast.Lambda):
            inline_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
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
        if isinstance(curr, ast.IfExp):
            stack.append(curr.test)
            stack.append(curr.orelse)
            stack.append(curr.body)
            continue
        if isinstance(curr, ast.Subscript):
            stack.append(curr.slice)
            stack.append(curr.value)
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
    if _is_pos_before_span(line, col, span):
        state.pre_mutated.update(filtered)
        if not frame.is_branched:
            state.pre_bound.update(filtered)
            state.pre_unbound.difference_update(filtered)
    if _is_pos_after_span(line, col, span):
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
    if (
        _is_pos_before_span(line, col, span)
        or frame.in_loop
        or (unconditional_post_del and _is_pos_after_span(line, col, span))
    ):
        state.pre_bound.discard(name)
        state.pre_unbound.add(name)
        state.pre_mutated.add(name)
    if _is_pos_after_span(line, col, span):
        state.post_mutated.add(name)
    if frame.in_loop:
        state.loop_mutated.add(name)


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
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
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


def _collect_escaped_intra_unit_names(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Set[str]:
    """Computes function/variable names that escape candidate_stmts via loads or escaped closures."""
    top_loads, top_calls = _extract_shallow_loads_and_calls(candidate_stmts)
    escaped: Set[str] = set(initial_escaped) | top_loads
    executed: Set[str] = set(top_calls) | escaped
    nested_defs: Dict[str, List[ast.AST]] = {}
    walk_stack: List[ast.AST] = list(candidate_stmts)
    while walk_stack:
        node = walk_stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            nested_defs.setdefault(node.name, []).append(node)
            if isinstance(node, ast.ClassDef) or getattr(
                node, "decorator_list", None
            ):
                executed.add(node.name)
        walk_stack.extend(ast.iter_child_nodes(node))

    body_info: Dict[int, Tuple[Set[str], Set[str]]] = {}
    changed = True
    while changed:
        changed = False
        for name, nodes in nested_defs.items():
            is_esc = name in escaped
            is_exec = is_esc or name in executed
            if not is_exec:
                continue
            for d_node in nodes:
                nid = id(d_node)
                if nid not in body_info:
                    body_info[nid] = _extract_shallow_loads_and_calls(
                        getattr(d_node, "body", ())
                    )
                b_loads, b_calls = body_info[nid]
                new_esc = (b_loads | b_calls) if is_esc else b_loads
                if not new_esc.issubset(escaped):
                    escaped.update(new_esc)
                    executed.update(new_esc)
                    changed = True
                if not b_calls.issubset(executed):
                    executed.update(b_calls)
                    changed = True
    return escaped


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


def _block_has_abnormal_exit(stmts: Sequence[ast.AST]) -> bool:
    """Checks whether a statement block contains Return, Break, or Continue outside inner scopes."""
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, (ast.Return, ast.Break, ast.Continue)):
            return True
        stack.extend(ast.iter_child_nodes(curr))
    return False


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
    if not handlers_l and not _block_has_abnormal_exit(final_l):
        for sub_seq in (body_l, orelse_l, final_l):
            bound, t_term = _extract_block_definite_bindings_and_term(
                sub_seq, active_nonlocals, bound, fn_globals
            )
            if t_term:
                return bound, True
        return bound, False

    if type(stmt).__name__ == "TryStar":
        bound.difference_update(_extract_shallow_deleted_names(stmt))
        if final_l and not _block_has_abnormal_exit(final_l):
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
        h_bound, h_term = _extract_block_definite_bindings_and_term(
            h_body, active_nonlocals, pre_try_surviving, fn_globals
        )
        h_name = getattr(h_node, "name", None)
        if isinstance(h_name, str):
            h_bound.discard(h_name)
        if not h_term:
            fallthrough_sets.append(h_bound)

    if not fallthrough_sets:
        if final_l and not _block_has_abnormal_exit(final_l):
            f_bound, _ = _extract_block_definite_bindings_and_term(
                final_l, active_nonlocals, pre_try_surviving, fn_globals
            )
            return f_bound, True
        return set(), True

    out_bound = set.intersection(*fallthrough_sets)
    if final_l:
        if _block_has_abnormal_exit(final_l):
            for f_stmt in final_l:
                out_bound.difference_update(
                    _extract_shallow_deleted_names(f_stmt)
                )
        else:
            out_bound, f_term = _extract_block_definite_bindings_and_term(
                final_l, active_nonlocals, out_bound, fn_globals
            )
            if f_term:
                return out_bound, True
    return out_bound, False


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
                bound.difference_update(_extract_shallow_deleted_names(hdr))
            if _can_bind_in_fn_scope(stmt.name, active_nonlocals, fn_globals):
                bound.add(stmt.name)
            continue
        if isinstance(stmt, ast.If):
            bound.difference_update(_extract_shallow_deleted_names(stmt.test))
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
                bound.difference_update(
                    _extract_shallow_deleted_names(item.context_expr)
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
        if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            bound, t_term = _eval_try_definite_bindings(
                stmt, active_nonlocals, bound, fn_globals
            )
            if t_term:
                return bound, True
            continue
        bound.difference_update(_extract_shallow_deleted_names(stmt))
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
        and _is_pos_before_span(
            getattr(curr, "end_lineno", getattr(curr, "lineno", None)),
            getattr(curr, "end_col_offset", getattr(curr, "col_offset", None)),
            span,
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
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        _maybe_push_pre_span_exhaustive_frame(
            curr, encloses, bool(curr.body), frame, span, state, stack
        )
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
                if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
            ]
            state.pre_mutated.update(t_stores)
            if not is_br:
                state.pre_bound.update(t_stores)
                state.pre_unbound.difference_update(t_stores)
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
                        if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
                    ]
                    state.pre_mutated.update(w_stores)
                    if not is_br:
                        state.pre_bound.update(w_stores)
                        state.pre_unbound.difference_update(w_stores)
        _push_seq(stack, curr.body, is_br if body_span else True, in_lp, act_nl)
        _push_seq(stack, curr.items, is_br, in_lp, act_nl)
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
        no_handler_fallthrough = (
            not encloses
            and not handlers_l
            and not _block_has_abnormal_exit(final_l)
        )
        _maybe_push_pre_span_exhaustive_frame(
            curr, encloses, bool(handlers_l), frame, span, state, stack
        )
        _push_seq(
            stack,
            final_l,
            is_br if (not encloses or final_span) else True,
            in_lp,
            act_nl,
        )
        _push_seq(
            stack,
            else_l,
            is_br if (else_span or no_handler_fallthrough) else True,
            in_lp,
            act_nl,
        )
        for h_node in reversed(handlers_l):
            h_span = encloses and _block_contains_span(
                getattr(h_node, "body", ()), span
            )
            stack.append(
                _StackFrame(
                    h_node, is_br if h_span else True, in_lp, act_nl
                )
            )
        _push_seq(
            stack,
            body_l,
            is_br if (body_span or no_handler_fallthrough) else True,
            in_lp,
            act_nl,
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


def _is_closure_factory_scope_uncached(
    scope_node: ast.FunctionDef,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Computes whether a FunctionDef can return a closure or generator without caching."""
    own_locals = frozenset(
        _extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        )
    )
    parent_map = (
        scope_ctx.parent_map
        if scope_ctx is not None
        else _get_scope_parent_map(scope_node)
    )
    shadowed = (
        (scope_ctx.shadowed_names if scope_ctx is not None else frozenset())
        | own_locals
    )
    all_fn_locals = (
        (scope_ctx.fn_locals if scope_ctx is not None else frozenset())
        | own_locals
    )
    local_ctx = _FnScopeContext(
        fn_locals=all_fn_locals,
        shadowed_names=shadowed,
        parent_map=parent_map,
        has_star_import=(
            scope_ctx.has_star_import if scope_ctx is not None else False
        ),
    )
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
    own_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
        include_bare_annotations=True,
    )
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
    own_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
        include_bare_annotations=True,
    )
    direct_nl = _extract_direct_nonlocals(fn_node)
    fn_g = set(fn_globals) | _extract_direct_nonlocals(fn_node, is_global=True)
    targets: Set[str] = set()
    stack: List[Tuple[ast.AST, Set[str], Set[str], Set[str]]] = [
        (stmt, own_locals, direct_nl, fn_g) for stmt in fn_node.body
    ]
    while stack:
        sub, cur_locals, cur_nl, cur_g = stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append((hdr, cur_locals, cur_nl, cur_g))
            sub_direct_nl = _extract_direct_nonlocals(sub)
            sub_g = cur_g | _extract_direct_nonlocals(sub, is_global=True)
            sub_own = (
                set()
                if isinstance(sub, ast.ClassDef)
                else _extract_single_node_local_names(
                    sub,
                    include_globals=False,
                    include_nonlocals=False,
                    include_self_name=False,
                    include_bare_annotations=True,
                )
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
            c_targets, _ = carrier
            store_names = (
                set()
                if isinstance(sub, ast.Call)
                else _extract_store_names_in_node(sub)
            )
            for t_name in c_targets:
                if t_name in cur_g:
                    continue
                if t_name in store_names:
                    if t_name in cur_nl:
                        targets.add(t_name)
                elif t_name not in cur_locals or t_name in cur_nl:
                    targets.add(t_name)
        for child_node in ast.iter_child_nodes(sub):
            stack.append((child_node, cur_locals, cur_nl, cur_g))
    return frozenset(targets)


def _extract_fn_outer_carrier_targets(
    fn_node: ast.AST,
    fn_globals: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Extracts outer-scope container/variable names mutated by carrier operations in fn_node."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (fn_node, "outer_carriers", frozenset(fn_globals)),
            lambda: _extract_fn_outer_carrier_targets_uncached(
                fn_node, fn_globals
            ),
        )
    )


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
            for stmt in curr.body:
                if isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not (
                    _is_node_inside_span(stmt, span)
                    or _strictly_encloses_span(stmt, span)
                ):
                    s_reads = _extract_cached_scope_free_reads(stmt)
                    s_muts, _ = (
                        _extract_closure_nonlocal_mutations_and_deletions(stmt)
                    )
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
                curr, state.fn_globals
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
            if span_in_hdr or _is_pos_after_span(estart, ecol, span):
                state.post_reads.update(free_r)
        if isinstance(curr, ast.ClassDef):
            cls_nonlocals = _extract_direct_nonlocals(curr)
            if (
                enclosing_cls
                or cls_nonlocals
                or not _is_pos_after_span(estart, ecol, span)
            ):
                _push_seq(
                    stack,
                    curr.body,
                    frame.is_branched,
                    frame.in_loop,
                    cls_nonlocals,
                )
        _push_seq(
            stack,
            hdr_nodes,
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
            comp_sub_stack, comp_targets = (
                _iter_comprehension_body_exprs_and_targets(curr)
            )
            comp_carriers: List[ast.AST] = []
            while comp_sub_stack:
                sub = comp_sub_stack.pop()
                if isinstance(sub, (ast.NamedExpr, ast.Call)):
                    comp_carriers.append(sub)
                comp_sub_stack.extend(
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
            for c_node in comp_carriers:
                _record_carrier_closures_and_aliases(
                    c_node,
                    frame,
                    sibling_closures,
                    alias_edges,
                    scope_ctx,
                    call_edges,
                    excluded_names=comp_targets,
                    fn_globals=state.fn_globals,
                )
        if _is_pos_after_span(c_line, c_col, span):
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
    """Extracts inline closures and alias edges from assignment or mutating call nodes."""
    carrier = _extract_closure_carrier_targets_and_values(curr)
    if carrier is None:
        return
    c_targets, c_values = carrier
    store_names = (
        set()
        if isinstance(curr, ast.Call)
        else _extract_store_names_in_node(curr)
    )
    valid_targets = [
        t
        for t in c_targets
        if t not in excluded_names
        and (
            _can_bind_in_fn_scope(t, frame.active_nonlocals, fn_globals)
            if t in store_names
            else t not in fn_globals
        )
    ]
    if not valid_targets:
        return
    c_calls: Set[str] = set()
    c_reads, c_muts, c_loads = _extract_expr_closure_and_load_info(
        c_values, scope_ctx, called_names_out=c_calls
    )
    if excluded_names:
        c_reads -= set(excluded_names)
        c_loads -= set(excluded_names)
        c_calls -= set(excluded_names)
    for t_name in valid_targets:
        if c_reads or c_muts:
            sibling_closures.append((t_name, set(c_reads), set(c_muts)))
        if c_loads:
            alias_edges.append((t_name, set(c_loads)))
        if c_calls and call_edges is not None:
            call_edges.append((t_name, set(c_calls)))


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
        if _is_pos_after_span(c_line, c_col, span):
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
    estart, _ = _node_earliest_start_pos(stmt)
    c_end = getattr(stmt, "end_lineno", None)
    if not (isinstance(estart, int) and estart > 0 and isinstance(c_end, int)):
        return False
    is_before = c_end < span.start
    is_after = estart > span.end
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
                or bool(getattr(curr, "decorator_list", None))
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
            or not _is_pos_after_span(eff_line, eff_col, span)
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
        | {t for t, _ in bufs.alias_edges}
        | {t for t, _ in bufs.call_edges}
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

    propagated_closures = _propagate_sibling_closures(
        bufs.sibling_closures,
        post_reads=bufs.state.post_reads,
        alias_edges=bufs.alias_edges,
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
            (t_name, frozenset(srcs)) for t_name, srcs in bufs.alias_edges
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
                    if isinstance(sub, ast.ClassDef) and enc_child in sub.body
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


def _check_captured_cell_mutations(
    captured_candidates: AbstractSet[str],
    eff_input_set: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    unit_closure_reads: Optional[AbstractSet[str]] = None,
) -> bool:
    """Checks 1 & 2: captured cell variable is unbound before unit or mutated outside unit."""
    actual_closure_reads = (
        captured_candidates
        if unit_closure_reads is None
        else unit_closure_reads
    )
    for var in captured_candidates:
        acc_ext_muts: Set[str] = set()
        acc_pre_bound: Set[str] = set()
        acc_pre_unbound: Set[str] = set()
        for idx in range(len(chain_infos) - 1, -1, -1):
            info = chain_infos[idx]
            acc_ext_muts.update(
                info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            acc_pre_bound.update(info.pre_bound - acc_pre_unbound)
            acc_pre_unbound.update(set(info.pre_unbound) - info.pre_bound)
            if var not in info.fn_locals:
                if var in actual_closure_reads:
                    acc_ext_muts.update(info.pre_mutated)
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
    escaped_unit_names = _collect_escaped_intra_unit_names(
        candidate_stmts, all_post_reads | all_sib_reads
    )
    for mut_fn_name, mutated_vars, is_escaped_def in intra_nl_mutators:
        mut_fn_escapes = is_escaped_def or mut_fn_name in escaped_unit_names
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
    *,
    is_suspended: bool = False,
    unreturned_mutations: bool = False,
    skip_pre_unit_closures: bool = False,
) -> bool:
    """Check 4: sibling closure referenced by unit (or concurrent with suspended unit) shares a mutated cell."""
    if not chain_infos:
        return False
    all_chain_locals: Set[str] = set().union(
        *(info.fn_locals for info in chain_infos)
    )
    if is_suspended and bool(
        set(sub_mutated)
        & chain_infos[-1].fn_locals
        & set(chain_infos[-1].concurrent_sibling_vars)
    ):
        return True
    if (
        unreturned_mutations
        and skip_pre_unit_closures
        and any(
            bool(set(sub_mutated) & all_chain_locals & set(info.post_reads))
            for info in chain_infos
        )
    ):
        return True
    unshadowed_sibs: List[
        Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]
    ] = []
    unshadowed_edges: List[Tuple[str, AbstractSet[str]]] = []
    for idx, info in enumerate(chain_infos):
        inner_shadowed: Set[str] = (
            set().union(
                *(
                    chain_infos[j].fn_locals
                    for j in range(idx + 1, len(chain_infos))
                )
            )
            if idx + 1 < len(chain_infos)
            else set()
        )
        for s_name, s_reads, s_muts in info.sibling_closures:
            if s_name is not None and s_name in inner_shadowed:
                continue
            unshadowed_sibs.append(
                (s_name, s_reads - inner_shadowed, s_muts - inner_shadowed)
            )
        for t_name, src_names in info.alias_edges:
            if t_name in inner_shadowed:
                continue
            unshadowed_edges.append((t_name, src_names - inner_shadowed))
    all_sib_closures = _propagate_sibling_closures(
        unshadowed_sibs,
        alias_edges=unshadowed_edges,
    )
    sub_touched = set(eff_input_set) | set(sub_mutated)
    for sib_name, sib_reads, sib_nl_muts in all_sib_closures:
        is_ref_sib = sib_name is not None and sib_name in eff_input_set
        if not is_ref_sib and (
            not unreturned_mutations or skip_pre_unit_closures
        ):
            continue
        if bool(sib_reads & all_chain_locals & set(sub_mutated)):
            return True
        if bool(
            sib_nl_muts
            & all_chain_locals
            & (sub_touched if is_ref_sib else set(sub_mutated))
        ):
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
    skip_pre_unit_closures: bool = False
    definite_stores: Optional[AbstractSet[str]] = None
    cols_resolved: bool = False


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings.

    Note:
        When ``ctx.skip_pre_unit_closures`` is True (lenient mode), only the speculative
        ``is_suspended`` check for unreferenced concurrent pre-unit closures in Check 4 is
        relaxed; Checks 1/2 (unbound or externally mutated captured cells) and Check 4 for
        sibling closures explicitly referenced via ``eff_input_set`` remain enforced.
    """
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return False
    if not ctx.is_subroutine and ctx.unit_kind not in (
        "closure",
        "comprehension",
        "complex_expr",
    ):
        return False

    u_start_col, u_end_col = (
        (ctx.u_start_col, ctx.u_end_col)
        if ctx.cols_resolved
        else _resolve_unit_col_bounds(
            ctx.candidate_stmts,
            ctx.mod_tree,
            ctx.u_start,
            ctx.u_end,
            ctx.u_start_col,
            ctx.u_end_col,
        )
    )
    enclosing_chain = _find_enclosing_function_chain(
        ctx.mod_tree, ctx.u_start, ctx.u_end, u_start_col
    )
    if not enclosing_chain:
        return False

    is_expr_unit = ctx.unit_kind in ("comprehension", "complex_expr")
    expr_unit_stores_escape = ctx.is_subroutine or is_expr_unit
    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    nonlocal_set = set(ctx.nonlocals)
    unit_mutated_names = store_set | set(ctx.deleted_names)
    if not expr_unit_stores_escape and bool(nonlocal_set & unit_mutated_names):
        return True

    enc_nonlocals, enc_outer_globals = _extract_scope_nonlocals_and_outer_globals(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_start_col=u_start_col,
        u_end_col=u_end_col,
    )
    enc_escaping = enc_nonlocals | enc_outer_globals
    if expr_unit_stores_escape and bool(
        unit_mutated_names
        & (enc_outer_globals if ctx.is_subroutine else enc_escaping)
    ):
        return True
    intra_nl_mutators = _collect_unit_nonlocal_mutators(
        ctx.candidate_stmts, mod_tree=ctx.mod_tree
    )
    is_suspended = ctx.is_subroutine and (ctx.has_yield or ctx.is_async)
    if is_suspended:
        if bool(unit_mutated_names & (enc_escaping | nonlocal_set)) or bool(
            intra_nl_mutators
        ):
            return True

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    if ctx.is_subroutine and _extract_direct_nonlocals(unit_mod):
        return True
    unit_closures = _collect_scope_closures_uncached(
        unit_mod,
        module_tree=ctx.mod_tree,
        genexp_include_header=False,
        cache_parent_map=False,
    )
    unit_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures))
        if unit_closures
        else set()
    )
    if ctx.is_subroutine:
        if is_suspended:
            unit_closure_reads.update(ctx.inputs)
    elif ctx.unit_kind == "closure":
        unit_closure_reads.update(ctx.free_vars)

    intra_closure_reads: Set[str] = (
        set(unit_closure_reads)
        if ctx.is_subroutine
        else (set(unit_closure_reads) | nonlocal_set)
    )

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    if expr_unit_stores_escape and bool(
        unit_closure_reads
        & (
            (unit_mutated_names & (enc_escaping | nonlocal_set))
            | (enc_outer_globals & eff_input_set)
        )
    ):
        return True
    unit_mutations = unit_mutated_names if expr_unit_stores_escape else set()

    if (
        not intra_closure_reads
        and not intra_nl_mutators
        and not ctx.is_subroutine
        and not unit_mutations
    ):
        return False

    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        u_start_col,
        u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
    )
    if (
        ctx.definite_stores is not None
        and ctx.is_subroutine
        and not ctx.has_yield
        and chain_infos
    ):
        unseeded_cond_stores = (
            store_set - set(ctx.definite_stores)
        ) - set(ctx.inputs)
        if unseeded_cond_stores:
            imm_info = chain_infos[-1]
            imm_sib_reads: Set[str] = (
                set().union(*(s_r for _, s_r, _ in imm_info.sibling_closures))
                if imm_info.sibling_closures
                else set()
            )
            eff_sib_reads = (
                imm_sib_reads & imm_info.post_reads
                if ctx.skip_pre_unit_closures
                else imm_sib_reads
            )
            cond_pre_post = (
                imm_info.pre_mutated - imm_info.pre_bound
            ) & imm_info.post_reads
            if bool(
                unseeded_cond_stores
                & imm_info.fn_locals
                & (eff_sib_reads | cond_pre_post)
            ):
                return True

    captured_candidates = intra_closure_reads & (
        eff_input_set
        | ((store_set | output_set) if expr_unit_stores_escape else set())
    )
    sub_mutated = (
        unit_mutated_names
        if expr_unit_stores_escape
        else (nonlocal_set & unit_mutated_names)
    )
    return (
        _check_intra_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            intra_nl_mutators,
            ctx.unit_top_locals,
            nonlocal_set,
            chain_infos,
            is_subroutine=ctx.is_subroutine,
        )
        or _check_captured_cell_mutations(
            captured_candidates,
            eff_input_set,
            chain_infos,
            unit_closure_reads=unit_closure_reads,
        )
        or _check_sibling_closure_sharing(
            eff_input_set,
            sub_mutated,
            chain_infos,
            is_suspended=(is_suspended or is_expr_unit)
            and not ctx.skip_pre_unit_closures,
            unreturned_mutations=is_expr_unit,
            skip_pre_unit_closures=ctx.skip_pre_unit_closures,
        )
    )
