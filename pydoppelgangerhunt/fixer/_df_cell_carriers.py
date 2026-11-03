"""Enclosing function scope context, column bounds, and closure carrier analysis."""

from __future__ import annotations

import ast
from collections import ChainMap
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
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

from pydoppelgangerhunt.fixer._df_cache import (
    _ScopeCacheKey,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _make_trusted_scope_cache_key,
    _params_contain_ast,
    _scope_cell_info_cache,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _BUILTIN_NAMES,
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_comprehension_walrus_names,
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _extract_single_node_local_names,
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
    """Cached per-function lexical scope metadata for eager generator expression checks.

    Note:
        ``parent_map`` is a read-only ``MappingProxyType`` view over a dictionary
        that is also shared across nested scope contexts; the underlying dictionary
        must never be mutated after construction.
    """

    fn_locals: frozenset[str]
    parent_map: Mapping[ast.AST, ast.AST]
    shadowed_names: frozenset[str]
    has_star_import: bool
    scope_owner: Optional[ast.AST] = None
    mod_tree: Optional[ast.AST] = None
    enclosing_scopes: Tuple[ast.AST, ...] = ()

    def cache_fingerprint(self) -> Tuple[Any, ...]:
        """Returns the non-AST semantic fingerprint of this scope context."""
        return (
            self.fn_locals,
            self.shadowed_names,
            self.has_star_import,
        )

    def cache_aux_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns all AST owner and enclosing scope nodes referenced by this context."""
        aux_nodes: List[ast.AST] = []
        if isinstance(self.scope_owner, ast.AST):
            aux_nodes.append(self.scope_owner)
        if isinstance(self.mod_tree, ast.AST):
            aux_nodes.append(self.mod_tree)
        for enc in self.enclosing_scopes:
            if isinstance(enc, ast.AST):
                aux_nodes.append(enc)
        return tuple(aux_nodes)


def _make_scope_ctx_cache_key(
    owner: ast.AST,
    kind: str,
    scope_ctx: Optional[_FnScopeContext],
    extra_params: Tuple[Any, ...] = (),
) -> _ScopeCacheKey:
    """Builds a _ScopeCacheKey incorporating full _FnScopeContext identity and AST owners."""
    if extra_params and _params_contain_ast(extra_params):
        raise ValueError(
            "_ScopeCacheKey.params must not contain ast.AST nodes; use owner or aux_ast_nodes"
        )
    if scope_ctx is None:
        return _make_trusted_scope_cache_key(
            owner, kind, params=(*extra_params, None)
        )
    return _make_trusted_scope_cache_key(
        owner,
        kind,
        params=(*extra_params, *scope_ctx.cache_fingerprint()),
        aux_ast_nodes=scope_ctx.cache_aux_nodes(),
    )


def _get_fn_scope_context_uncached(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    *,
    cache_parent_map: bool = True,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> _FnScopeContext:
    """Computes lexical scope metadata for a function or ephemeral unit module without caching."""
    raw_parent_map = _get_scope_parent_map(
        scope_node, use_cache=cache_parent_map
    )
    return _FnScopeContext(
        fn_locals=frozenset(_extract_strict_fn_local_names(scope_node)),
        parent_map=MappingProxyType(raw_parent_map),
        shadowed_names=frozenset(
            _extract_scope_local_names(
                scope_node,
                module_tree=mod_tree,
                cache_scope_node=cache_parent_map,
                enclosing_scopes=enclosing_scopes,
            )
        ),
        has_star_import=_has_wildcard_star_import(
            scope_node,
            module_tree=mod_tree,
            cache_scope_node=cache_parent_map,
            enclosing_scopes=enclosing_scopes,
        ),
        scope_owner=scope_node,
        mod_tree=mod_tree,
        enclosing_scopes=tuple(enclosing_scopes),
    )


def _get_fn_scope_context(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
) -> _FnScopeContext:
    """Computes and caches lexical scope metadata for an AST scope node."""
    cache_key = _ScopeCacheKey(
        scope_node,
        "fn_scope_ctx",
        aux_ast_nodes=() if mod_tree is None else (mod_tree,),
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
    """Returns True if node is a GeneratorExp that is not eagerly consumed.

    Note:
        When ``scope_ctx`` is None on a detached ``ast.GeneratorExp`` (so no parent
        call context is available), this function deliberately defaults to ``True``
        (fail-closed conservative treatment as a lazy generator). Higher-level helpers
        such as ``_extract_comprehension_deferred_closure_info`` and
        ``_extract_lazy_genexp_walrus_names`` automatically synthesize a subtree
        ``_FnScopeContext`` when ``scope_ctx`` is omitted so nested eager consumers
        (e.g., ``sum(x for x in row)``) are still classified accurately.
    """
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


class _IndexedSpanNode(NamedTuple):
    """Pre-indexed AST node entry for fast line/column span resolution."""

    node: ast.AST
    n_start: int
    n_end: int
    n_col: Optional[int]
    n_end_col: Optional[int]
    req_u_start_le: int
    req_u_end_ge: int


class _SpanLineIndex:
    """Cached per-tree line index and lazy AST dump cache for O(1) boundary lookups."""

    __slots__ = (
        "by_start_line",
        "by_end_line",
        "stmts_by_start_line",
        "stmts_by_end_line",
        "missing_end_lineno_starts",
        "missing_col_spans",
        "_dump_cache",
    )

    def __init__(self, root_node: ast.AST) -> None:
        self.by_start_line: Dict[int, List[_IndexedSpanNode]] = {}
        self.by_end_line: Dict[int, List[_IndexedSpanNode]] = {}
        self.stmts_by_start_line: Dict[int, List[ast.stmt]] = {}
        self.stmts_by_end_line: Dict[int, List[Tuple[ast.stmt, int, bool]]] = {}
        missing_end_starts: List[int] = []
        missing_cols: List[Tuple[int, int]] = []
        self._dump_cache: Dict[int, str] = {}

        stack: List[Tuple[ast.AST, int, int]] = [(root_node, 0, 10**9)]
        while stack:
            node, anc_max_start, anc_min_end = stack.pop()
            n_start, n_col = _node_earliest_start_pos(node)
            raw_end = getattr(node, "end_lineno", None)
            n_end_col = getattr(node, "end_col_offset", None)
            is_stmt_or_expr = isinstance(node, (ast.stmt, ast.expr))
            if is_stmt_or_expr and isinstance(n_start, int):
                if not isinstance(raw_end, int):
                    missing_end_starts.append(n_start)
                elif not isinstance(n_col, int) or not isinstance(
                    n_end_col, int
                ):
                    missing_cols.append((n_start, raw_end))

            n_end = raw_end if isinstance(raw_end, int) else n_start
            if isinstance(n_start, int) and isinstance(n_end, int):
                req_start_le = (
                    n_start if n_start < anc_min_end else anc_min_end
                )
                req_end_ge = (
                    n_end if n_end > anc_max_start else anc_max_start
                )
                entry = _IndexedSpanNode(
                    node=node,
                    n_start=n_start,
                    n_end=n_end,
                    n_col=n_col if isinstance(n_col, int) else None,
                    n_end_col=n_end_col if isinstance(n_end_col, int) else None,
                    req_u_start_le=req_start_le,
                    req_u_end_ge=req_end_ge,
                )
                self.by_start_line.setdefault(n_start, []).append(entry)
                if n_end != n_start:
                    self.by_end_line.setdefault(n_end, []).append(entry)
                if (
                    isinstance(node, ast.stmt)
                    and isinstance(raw_end, int)
                    and isinstance(n_col, int)
                    and isinstance(n_end_col, int)
                ):
                    self.stmts_by_start_line.setdefault(n_start, []).append(
                        node
                    )
                    has_compound_child = n_start < raw_end and any(
                        isinstance(ch, ast.stmt)
                        or type(ch).__name__ in ("ExceptHandler", "match_case")
                        for ch in ast.iter_child_nodes(node)
                    )
                    self.stmts_by_end_line.setdefault(raw_end, []).append(
                        (node, n_start, has_compound_child)
                    )
                next_max_start = (
                    n_start if n_start > anc_max_start else anc_max_start
                )
                next_min_end = n_end if n_end < anc_min_end else anc_min_end
            else:
                next_max_start = anc_max_start
                next_min_end = anc_min_end

            children = list(ast.iter_child_nodes(node))
            for ch in reversed(children):
                stack.append((ch, next_max_start, next_min_end))

        self.missing_end_lineno_starts: Tuple[int, ...] = tuple(
            missing_end_starts
        )
        self.missing_col_spans: Tuple[Tuple[int, int], ...] = tuple(
            missing_cols
        )

    def get_dump(self, node: ast.AST) -> str:
        """Returns the cached ast.dump(node) string."""
        nid = id(node)
        cached = self._dump_cache.get(nid)
        if cached is None:
            cached = ast.dump(node)
            self._dump_cache[nid] = cached
        return cached

    def iter_boundary_entries(
        self, u_start: int, u_end: int
    ) -> Sequence[_IndexedSpanNode]:
        """Returns indexed AST entries starting at u_start or ending at u_end."""
        start_entries = self.by_start_line.get(u_start, ())
        if u_start == u_end:
            extra_end = self.by_end_line.get(u_end, ())
            if not extra_end:
                return start_entries
            return [*start_entries, *extra_end]
        end_entries = self.by_end_line.get(u_end, ())
        end_same_line = self.by_start_line.get(u_end, ())
        if not end_entries and not end_same_line:
            return start_entries
        seen_ids = {id(e.node) for e in start_entries}
        combined = list(start_entries)
        for entry in (*end_entries, *end_same_line):
            if entry.n_end == u_end and id(entry.node) not in seen_ids:
                seen_ids.add(id(entry.node))
                combined.append(entry)
        return combined


def _get_span_line_index(root_node: ast.AST) -> _SpanLineIndex:
    """Returns or builds the cached _SpanLineIndex for root_node."""
    return cast(
        _SpanLineIndex,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(root_node, "span_line_index"),
            lambda: _SpanLineIndex(root_node),
        ),
    )


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
    single_stmt = len(candidate_stmts) == 1
    first_dumps = (
        _stmt_dump_set(candidate_stmts, 0) if u_start_col is None else set()
    )
    if u_end_col is not None:
        last_dumps: Set[str] = set()
    elif single_stmt and u_start_col is None:
        last_dumps = first_dumps
    else:
        last_dumps = _stmt_dump_set(candidate_stmts, -1)

    line_index = _get_span_line_index(mod_tree)
    start_matches: List[Tuple[int, int]] = []
    end_matches: List[Tuple[int, int]] = []
    span_stmts: List[Tuple[int, int]] = []
    for entry in line_index.iter_boundary_entries(u_start, u_end):
        if not (
            u_start <= entry.req_u_start_le and entry.req_u_end_ge <= u_end
        ):
            continue
        node = entry.node
        n_start = entry.n_start
        n_end = entry.n_end
        n_col = entry.n_col
        n_end_col = entry.n_end_col
        if (
            isinstance(node, ast.stmt)
            and n_start == u_start
            and n_end == u_end
            and isinstance(n_col, int)
            and isinstance(n_end_col, int)
            and not (
                single_stmt
                and isinstance(candidate_stmts[0], ast.Expr)
                and not isinstance(node, ast.Expr)
            )
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
            dumped = line_index.get_dump(node)
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

    return (
        _pick_inferred_col(
            u_start_col, start_matches, span_stmts, 0, opposite_col=u_end_col
        ),
        _pick_inferred_col(
            u_end_col, end_matches, span_stmts, 1, opposite_col=u_start_col
        ),
    )


def _has_unresolved_span_bounds(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int],
    u_end_col: Optional[int],
    candidate_stmts: Sequence[ast.AST] = (),
) -> bool:
    """Returns True if boundary columns on multi-statement/sub-expression lines are unresolved or AST end coords are missing."""
    line_index = _get_span_line_index(scope_node)
    if any(ln <= u_end for ln in line_index.missing_end_lineno_starts):
        return True
    if any(
        not (n_end < u_start or n_start > u_end)
        for n_start, n_end in line_index.missing_col_spans
    ):
        return True
    for entry in line_index.iter_boundary_entries(u_start, u_end):
        node = entry.node
        if not isinstance(node, (ast.stmt, ast.expr)):
            continue
        live_end = getattr(node, "end_lineno", None)
        live_end_col = getattr(node, "end_col_offset", None)
        if not isinstance(live_end, int) or not isinstance(live_end_col, int):
            return True
    if u_start_col is not None and u_end_col is not None:
        return False

    start_line_stmts: List[ast.stmt] = [
        stmt
        for stmt in line_index.stmts_by_start_line.get(u_start, ())
        if isinstance(getattr(stmt, "end_lineno", None), int)
        and getattr(stmt, "end_lineno", 0) >= u_start
        and not _node_strictly_encloses_bounds(
            stmt, u_start, u_end, u_start_col=u_start_col, u_end_col=u_end_col
        )
    ]
    end_line_stmts: List[ast.stmt] = [
        stmt
        for stmt, n_start, has_compound_child in line_index.stmts_by_end_line.get(
            u_end, ()
        )
        if n_start <= u_end
        and not has_compound_child
        and not _node_strictly_encloses_bounds(
            stmt, u_start, u_end, u_start_col=u_start_col, u_end_col=u_end_col
        )
    ]
    is_expr_cand = len(candidate_stmts) == 1 and isinstance(
        candidate_stmts[0], ast.Expr
    )
    cand_dump = ast.dump(candidate_stmts[0]) if is_expr_cand else ""
    return any(
        col_bound is None
        and (
            len(line_stmts) > 1
            or (
                is_expr_cand
                and any(line_index.get_dump(s) != cand_dump for s in line_stmts)
            )
        )
        for col_bound, line_stmts in (
            (u_start_col, start_line_stmts),
            (u_end_col, end_line_stmts),
        )
    )


def _walk_invalid_lexical_scope_declarations(
    initial_stack: List[Tuple[ast.AST, frozenset[str], frozenset[str], bool]],
) -> bool:
    """Walks initial_stack checking for PEP 3104 nonlocal/global declaration violations."""
    stack = initial_stack
    while stack:
        curr, cur_bound, cur_blocked, cur_rooted = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for hdr in iter_nested_scope_header_nodes(curr):
                stack.append((hdr, cur_bound, cur_blocked, cur_rooted))
            d_nl = _extract_direct_nonlocals(curr, is_global=False)
            d_g = _extract_direct_nonlocals(curr, is_global=True)
            if d_nl & d_g:
                return True
            if d_nl & cur_blocked:
                return True
            if cur_rooted and (d_nl - cur_bound):
                return True
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = _extract_arg_names(
                    curr.args
                ) | _extract_node_type_param_names(curr)
                if params & (d_nl | d_g):
                    return True
                own_locals = frozenset(_extract_strict_fn_local_names(curr))
                next_bound = (cur_bound - d_g) | own_locals
                next_blocked = (cur_blocked - own_locals) | d_g
            else:
                next_bound = cur_bound
                next_blocked = cur_blocked
            for ch in curr.body:
                stack.append((ch, next_bound, next_blocked, cur_rooted))
            continue
        for child in ast.iter_child_nodes(curr):
            stack.append((child, cur_bound, cur_blocked, cur_rooted))
    return False


def _scope_has_invalid_lexical_declarations(
    root_scope: ast.AST,
    is_module_rooted: bool,
) -> bool:
    """Checks (and caches per AST root scope) whether root_scope has invalid lexical declarations."""
    return bool(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(
                root_scope, "invalid_lex_decls", params=(is_module_rooted,)
            ),
            lambda: _walk_invalid_lexical_scope_declarations(
                [(root_scope, frozenset(), frozenset(), is_module_rooted)]
            ),
        )
    )


def _has_invalid_lexical_scope_declarations(
    candidate_stmts: Sequence[ast.AST],
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]] = (),
    mod_tree: Optional[ast.AST] = None,
) -> bool:
    """Returns True if any scope in enclosing_chain or candidate_stmts violates PEP 3104 nonlocal/global rules."""
    is_module_rooted = isinstance(mod_tree, ast.Module)
    if enclosing_chain and _scope_has_invalid_lexical_declarations(
        enclosing_chain[0][2], is_module_rooted
    ):
        return True
    bound_in_fns: frozenset[str] = frozenset()
    blocked_globals: frozenset[str] = frozenset()
    for _, _, enc_fn in enclosing_chain:
        d_nl = _extract_direct_nonlocals(enc_fn, is_global=False)
        d_g = _extract_direct_nonlocals(enc_fn, is_global=True)
        if d_nl & d_g:
            return True
        if isinstance(enc_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = _extract_arg_names(enc_fn.args) | _extract_node_type_param_names(
                enc_fn
            )
            if params & (d_nl | d_g):
                return True
        if d_nl & blocked_globals:
            return True
        if is_module_rooted and (d_nl - bound_in_fns):
            return True
        own_locals = frozenset(_extract_strict_fn_local_names(enc_fn))
        bound_in_fns = (bound_in_fns - d_g) | own_locals
        blocked_globals = (blocked_globals - own_locals) | d_g

    unit_mod = ast.Module(
        body=[s for s in candidate_stmts if isinstance(s, ast.stmt)],
        type_ignores=[],
    )
    top_nl = _extract_direct_nonlocals(unit_mod, is_global=False)
    top_g = _extract_direct_nonlocals(unit_mod, is_global=True)
    if top_nl & top_g:
        return True
    if enclosing_chain and (top_nl & blocked_globals):
        return True
    if is_module_rooted and (top_nl - bound_in_fns):
        return True

    return _walk_invalid_lexical_scope_declarations(
        [
            (stmt, bound_in_fns, blocked_globals, is_module_rooted)
            for stmt in candidate_stmts
        ]
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


def _scan_scope_nonlocal_mutations_and_deletions(
    scope_node: ast.AST,
    *,
    include_nested: bool = True,
    use_cache: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Scans scope_node for (mutated_or_deleted, deleted) nonlocal variables."""
    own_locals = (
        set()
        if (not include_nested or isinstance(scope_node, ast.ClassDef))
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
            if include_nested:
                sub_mut, sub_del = _extract_closure_nonlocal_mutations_and_deletions(
                    curr, use_cache=use_cache
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
        (direct_nonlocals & direct_mutated) | nested_mutated,
        (direct_nonlocals & direct_deleted) | nested_deleted,
    )


def _extract_direct_scope_nonlocal_mutations_and_deletions(
    scope_node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (mutated_or_deleted, deleted) nonlocal variables directly in scope_node.body."""
    return _scan_scope_nonlocal_mutations_and_deletions(
        scope_node, include_nested=False, use_cache=False
    )


def _extract_closure_nonlocal_mutations_and_deletions_uncached(
    scope_node: ast.AST,
    *,
    use_cache: bool = True,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (mutated_or_deleted, deleted) outer-scope variables via nonlocal in scope_node.

    Per PEP 227, class body scopes are not visible to nested methods and do not create
    an intermediate lexical barrier for nonlocal declarations in nested functions; thus
    own_locals is empty when scope_node is an ast.ClassDef.
    """
    muts, dels = _scan_scope_nonlocal_mutations_and_deletions(
        scope_node, include_nested=True, use_cache=use_cache
    )
    return frozenset(muts), frozenset(dels)


def _resolve_cached_str_set_pair(
    node: ast.AST,
    allowed_types: Tuple[ type[ast.AST], ...],
    cache_key: _ScopeCacheKey,
    compute_fn: Callable[[bool], Tuple[frozenset[str], frozenset[str]]],
    *,
    use_cache: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Evaluates (or fetches from _scope_cell_info_cache) a pair of frozenset[str] and returns mutable sets."""
    if not isinstance(node, allowed_types):
        return set(), set()
    first, second = (
        cast(
            Tuple[frozenset[str], frozenset[str]],
            _get_or_compute_scope_cached(
                _scope_cell_info_cache, cache_key, lambda: compute_fn(True)
            ),
        )
        if use_cache
        else compute_fn(False)
    )
    return set(first), set(second)


def _extract_closure_nonlocal_mutations_and_deletions(
    scope_node: ast.AST, *, use_cache: bool = True
) -> Tuple[Set[str], Set[str]]:
    """Extracts and caches (mutated_or_deleted, deleted) nonlocal variables in scope_node."""
    return _resolve_cached_str_set_pair(
        scope_node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        _ScopeCacheKey(scope_node, "nl_muts_dels"),
        lambda c: _extract_closure_nonlocal_mutations_and_deletions_uncached(scope_node, use_cache=c),
        use_cache=use_cache,
    )


def _extract_closure_nonlocal_mutations(
    scope_node: ast.AST,
    *,
    use_cache: bool = True,
) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    return _extract_closure_nonlocal_mutations_and_deletions(
        scope_node, use_cache=use_cache
    )[0]


def _extract_cached_scope_free_reads(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    *,
    use_cache: bool = True,
) -> Set[str]:
    """Extracts and caches free variable reads escaping a nested function or class."""
    raw = (
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _ScopeCacheKey(node, "free_reads"),
            lambda: frozenset(_extract_nested_scope_free_reads(node)),
        )
        if use_cache
        else _extract_nested_scope_free_reads(node)
    )
    return set(raw)


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


def _ensure_subtree_scope_ctx(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext],
) -> Optional[_FnScopeContext]:
    """Synthesizes a subtree _FnScopeContext when scope_ctx is None and node is not a bare GeneratorExp."""
    if scope_ctx is not None or isinstance(node, ast.GeneratorExp):
        return scope_ctx
    wrap: ast.AST = (
        node
        if isinstance(node, ast.Module)
        else ast.Module(
            body=[node if isinstance(node, ast.stmt) else ast.Expr(value=cast(ast.expr, node))],
            type_ignores=[],
        )
    )
    return _get_fn_scope_context_uncached(wrap, cache_parent_map=False)


def _extract_lazy_genexp_walrus_names(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts walrus target names inside lazy GeneratorExp subtrees of node."""
    eff_ctx = _ensure_subtree_scope_ctx(node, scope_ctx)
    names: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, eff_ctx):
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
    eff_ctx = _ensure_subtree_scope_ctx(comp_node, scope_ctx)
    is_lazy = _is_lazy_genexp(comp_node, eff_ctx)
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
                sub, eff_ctx
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

    lazy_muts = _extract_lazy_genexp_walrus_names(comp_node, eff_ctx)
    return deferred_reads, lazy_muts


class _IntraUnitMutator(NamedTuple):
    """Intra-unit closure or lazy generator expression mutating outer variables."""

    name: str
    mutated_vars: Set[str]
    is_escaped_def: bool
    node: Optional[ast.AST] = None


def _is_self_escaping_def_node(
    d_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    *,
    use_cache: bool = True,
) -> bool:
    """Checks whether a function or class definition escapes or mutates on definition."""
    if d_node.decorator_list:
        return True
    if isinstance(d_node, ast.ClassDef):
        if d_node.bases or d_node.keywords:
            return True
        if _extract_direct_scope_nonlocal_mutations_and_deletions(d_node)[0]:
            return True
        if _extract_direct_nonlocals(
            d_node, is_global=True
        ) & _extract_closure_nonlocal_mutations(d_node, use_cache=use_cache):
            return True
    return False


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    mod_tree: Optional[ast.AST] = None,
    *,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> List[_IntraUnitMutator]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    unit_mod = ast.Module(
        body=[s for s in candidate_stmts if isinstance(s, ast.stmt)],
        type_ignores=[],
    )
    use_node_cache = bool(
        mod_tree is not None
        and candidate_stmts
        and candidate_stmts[0] in _get_scope_parent_map(mod_tree)
    )
    fn_enclosing = [
        s for s in enclosing_scopes if not isinstance(s, ast.ClassDef)
    ]
    base_fn_ctx = _get_fn_scope_context_uncached(
        unit_mod,
        mod_tree=mod_tree,
        cache_parent_map=False,
        enclosing_scopes=fn_enclosing,
    )
    top_ctx = (
        _get_fn_scope_context_uncached(
            unit_mod,
            mod_tree=mod_tree,
            cache_parent_map=False,
            enclosing_scopes=enclosing_scopes,
        )
        if len(fn_enclosing) != len(enclosing_scopes)
        else base_fn_ctx
    )
    _esc_names, escaped_def_ids, _exec_ids = (
        _collect_escaped_and_executed_intra_unit_defs(
            candidate_stmts, frozenset(), use_cache=use_node_cache
        )
    )
    mutators: List[_IntraUnitMutator] = []
    stack: List[
        Tuple[ast.AST, frozenset[str], _FnScopeContext, _FnScopeContext, bool]
    ] = [
        (stmt, frozenset(), base_fn_ctx, top_ctx, False)
        for stmt in candidate_stmts
    ]
    while stack:
        curr, shadowed, fn_ctx, active_ctx, in_unescaped_cls = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = (
                _extract_closure_nonlocal_mutations(
                    curr, use_cache=use_node_cache
                )
                - shadowed
            )
            is_self_esc = _is_self_escaping_def_node(
                curr, use_cache=use_node_cache
            )
            if isinstance(curr, ast.ClassDef):
                is_escaped_def = (id(curr) in escaped_def_ids) or is_self_esc
            else:
                is_escaped_def = (
                    (id(curr) in escaped_def_ids)
                    or is_self_esc
                    or (
                        not in_unescaped_cls
                        and _is_closure_factory_scope(
                            curr, active_ctx, use_cache=use_node_cache
                        )
                    )
                )
            if mutated:
                mutators.append(
                    _IntraUnitMutator(
                        curr.name, mutated, is_escaped_def, node=curr
                    )
                )
            if isinstance(curr, ast.ClassDef):
                # PEP 227: class body bindings and class-level global declarations do not
                # create a lexical barrier for nonlocal declarations in nested methods.
                body_shadowed = shadowed
                body_fn_ctx = fn_ctx
                body_active_ctx = _make_nested_fn_scope_context(
                    curr,
                    frozenset(_extract_single_node_local_names(curr)),
                    fn_ctx,
                    use_cache=use_node_cache,
                )
                body_unesc_cls = not is_escaped_def
            else:
                fn_own = frozenset(_extract_strict_fn_local_names(curr))
                body_shadowed = shadowed | fn_own
                body_fn_ctx = _make_nested_fn_scope_context(
                    curr, fn_own, fn_ctx, use_cache=use_node_cache
                )
                body_active_ctx = body_fn_ctx
                body_unesc_cls = False
            stack.extend(
                (ch, body_shadowed, body_fn_ctx, body_active_ctx, body_unesc_cls)
                for ch in curr.body
            )
            stack.extend(
                (hdr, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, active_ctx):
                w_names = (
                    _extract_comprehension_body_walrus_names(curr) - shadowed
                )
                if w_names:
                    mutators.append(
                        _IntraUnitMutator("", w_names, True, node=curr)
                    )
                if curr.generators:
                    stack.append(
                        (
                            curr.generators[0].iter,
                            shadowed,
                            fn_ctx,
                            active_ctx,
                            in_unescaped_cls,
                        )
                    )
                continue
            stack.extend(
                (ch, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for ch in ast.iter_child_nodes(curr)
            )
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(
                (hdr, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        stack.extend(
            (ch, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
            for ch in ast.iter_child_nodes(curr)
        )
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
    """Checks whether an AST node's start and end positions lie within span.

    Fails closed (returns False) if node lacks end_lineno, or lacks col_offset /
    end_col_offset when span specifies column bounds on the corresponding boundary line.
    """
    estart, ecol = _node_earliest_start_pos(node)
    if not isinstance(estart, int) or estart < span.start or estart > span.end:
        return False
    if estart == span.start and span.start_col is not None:
        if not isinstance(ecol, int) or ecol < span.start_col:
            return False
    if not _is_ast_node_within_unit_bounds(
        node,
        span.start,
        span.end,
        u_end_col=span.end_col,
        u_start_col=span.start_col,
    ):
        return False
    c_end = getattr(node, "end_lineno", None)
    if not isinstance(c_end, int) or c_end > span.end:
        return False
    if c_end == span.end and span.end_col is not None:
        end_col = getattr(node, "end_col_offset", None)
        if not isinstance(end_col, int) or end_col > span.end_col:
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


def _collect_target_names(target: ast.AST, *, mode: str) -> Set[str]:
    """Collects target variable names for 'rebound', 'receiver', or 'mutated' mode."""
    names: Set[str] = set()
    stack: List[Tuple[ast.AST, bool]] = [(target, mode == "receiver")]
    while stack:
        curr, in_receiver = stack.pop()
        if isinstance(curr, ast.Name):
            if in_receiver or mode == "rebound":
                names.add(curr.id)
        elif isinstance(curr, (ast.Tuple, ast.List)):
            stack.extend((elt, in_receiver) for elt in curr.elts)
        elif isinstance(curr, ast.Starred):
            stack.append((curr.value, in_receiver))
        elif isinstance(curr, (ast.Attribute, ast.Subscript)) and mode != "rebound":
            stack.append((curr.value, True))
    return names


def _extract_root_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names actually bound or rebound by an assignment target."""
    return _collect_target_names(target, mode="rebound")


def _extract_receiver_root_names(node: ast.AST) -> Set[str]:
    """Extracts root variable names from an attribute or subscript receiver chain."""
    return _collect_target_names(node, mode="receiver")


def _extract_mutated_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names whose referenced objects are mutated by an assignment target."""
    return _collect_target_names(target, mode="mutated")


# Built-in functions whose return types are strictly enforced by the CPython runtime
# to be primitive scalars, booleans, strings, byte sequences, ranges, file streams, or None
# (e.g., TypeError is raised in C if __int__, __index__, __len__, __bool__, __str__,
# __repr__, __format__, __float__, __complex__, or __hash__ returns a non-primitive).
# Built-ins that delegate to unconstrained user dunders (abs -> __abs__, round -> __round__,
# divmod -> __divmod__, pow -> __pow__, sum -> __add__) are intentionally excluded so custom
# objects whose dunders return closures are never misclassified as non-closure scalars.
_NON_CLOSURE_RETURN_BUILTINS: frozenset[str] = frozenset(
    {
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
        "print",
        "range",
        "repr",
        "str",
    }
)


def _is_non_closure_builtin_call(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node is a Call to an unshadowed scalar/non-closure built-in.

    When ``scope_ctx`` is provided, both ``scope_ctx.has_star_import`` and
    ``scope_ctx.shadowed_names`` must confirm the built-in is not shadowed.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _NON_CLOSURE_RETURN_BUILTINS
        and (
            scope_ctx is None
            or (
                not scope_ctx.has_star_import
                and "*" not in scope_ctx.shadowed_names
                and node.func.id not in scope_ctx.shadowed_names
            )
        )
    )


def _is_non_closure_scalar_expr(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node is guaranteed to evaluate to a non-closure scalar value.

    Note:
        Only literals (``ast.Constant``, ``ast.JoinedStr``), boolean negation
        (``not x``), calls to unshadowed C-enforced ``_NON_CLOSURE_RETURN_BUILTINS``,
        and unary/binary/comparison operations whose operands are all recursively
        non-closure scalars are classified as scalar. Binary, unary, or comparison
        operations involving arbitrary variables or custom objects are NOT assumed
        to be scalar because custom dunders (``__add__``, ``__abs__``, ``__eq__``,
        etc.) can return closures. Inline mutations inside scalar sub-expressions
        are still extracted by ``_extract_carrier_value_closure_and_load_info``.
    """
    if isinstance(node, (ast.Constant, ast.JoinedStr)):
        return True
    if _is_non_closure_builtin_call(node, scope_ctx):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.Not) or _is_non_closure_scalar_expr(
            node.operand, scope_ctx
        )
    if isinstance(node, ast.BinOp):
        return _is_non_closure_scalar_expr(
            node.left, scope_ctx
        ) and _is_non_closure_scalar_expr(node.right, scope_ctx)
    if isinstance(node, ast.Compare):
        return _is_non_closure_scalar_expr(node.left, scope_ctx) and all(
            _is_non_closure_scalar_expr(comp, scope_ctx)
            for comp in node.comparators
        )
    return False


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions.

    Note:
        Direct ``ast.Name`` call targets (``curr.func`` when ``isinstance(curr.func, ast.Name)``
        and ``curr`` is not an unshadowed scalar built-in call) are routed to
        ``called_names_out`` (when provided) rather than ``loaded_names`` so callers can
        distinguish direct function/factory invocations from closure value references.
    """
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
            if _is_non_closure_builtin_call(curr, scope_ctx):
                sub_reads, sub_muts, sub_loads = (
                    _extract_expr_closure_and_load_info(
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
                )
                inline_reads.update(sub_reads)
                inline_muts.update(sub_muts)
                loaded_names.update(sub_loads)
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


def _extract_carrier_value_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts closure reads, mutations, and loads for carrier value expressions."""
    carrier_nodes: List[ast.AST] = []
    scalar_nodes: List[ast.AST] = []
    for node in nodes:
        if _is_non_closure_scalar_expr(node, scope_ctx):
            scalar_nodes.append(node)
        else:
            carrier_nodes.append(node)
    c_reads, c_muts, c_loads = (
        _extract_expr_closure_and_load_info(
            carrier_nodes, scope_ctx, called_names_out=called_names_out
        )
        if carrier_nodes
        else (set(), set(), set())
    )
    if scalar_nodes:
        _, s_muts, _ = _extract_expr_closure_and_load_info(
            scalar_nodes,
            scope_ctx,
            called_names_out=called_names_out,
        )
        c_muts.update(s_muts)
    return c_reads, c_muts, c_loads


def _extract_assignment_targets_and_value(
    node: ast.AST,
    mode: str,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (target_names, (value,)) from assignment nodes using the specified target mode."""
    if isinstance(node, ast.Assign) and node.value is not None:
        targets: Set[str] = set()
        for tgt in node.targets:
            targets.update(_collect_target_names(tgt, mode=mode))
        return (targets, (node.value,)) if targets else None
    if (
        isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        and node.value is not None
    ):
        targets = _collect_target_names(node.target, mode=mode)
        return (targets, (node.value,)) if targets else None
    return None


def _extract_closure_carrier_targets_and_values(
    node: ast.AST,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (rebound_target_names, value_nodes) from variable-binding assignments."""
    return _extract_assignment_targets_and_value(node, mode="rebound")


def _extract_closure_carrier_mutation_targets_and_values(
    node: ast.AST,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (mutated_object_root_names, value_nodes) from attribute/subscript/method mutations."""
    assign_res = _extract_assignment_targets_and_value(node, mode="mutated")
    if assign_res is not None:
        return assign_res
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        val_nodes: Sequence[ast.AST] = [
            *node.args,
            *(kw.value for kw in node.keywords if kw.value is not None),
        ]
        if val_nodes:
            targets = _extract_receiver_root_names(node.func.value)
            return (targets, val_nodes) if targets else None
    return None


def _iter_carrier_target_entries(
    node: ast.AST,
) -> List[Tuple[bool, Set[str], Sequence[ast.AST]]]:
    """Returns (is_rebound, target_names, value_nodes) entries for a carrier AST node."""
    entries: List[Tuple[bool, Set[str], Sequence[ast.AST]]] = []
    rebound = _extract_closure_carrier_targets_and_values(node)
    if rebound is not None:
        entries.append((True, rebound[0], rebound[1]))
    mutated = _extract_closure_carrier_mutation_targets_and_values(node)
    if mutated is not None:
        entries.append((False, mutated[0], mutated[1]))
    return entries


def _extract_shallow_loads_and_calls(
    stmts: Sequence[ast.AST],
    *,
    deletions_out: Optional[Set[str]] = None,
    pep227_loads_out: Optional[Set[str]] = None,
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
                lam_reads = _extract_expr_scope_free_reads(
                    curr, include_header=False
                )
                loaded.update(lam_reads)
                if pep227_loads_out is not None:
                    pep227_loads_out.update(lam_reads)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if pep227_loads_out is not None and isinstance(
            curr, _COMPREHENSION_TYPES
        ):
            pep227_loads_out.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
        if deletions_out is not None and isinstance(curr, ast.Delete):
            for d_tgt in curr.targets:
                if isinstance(d_tgt, ast.Name):
                    deletions_out.add(d_tgt.id)
                elif isinstance(d_tgt, (ast.Tuple, ast.List)):
                    deletions_out.update(
                        elt.id
                        for elt in ast.walk(d_tgt)
                        if isinstance(elt, ast.Name)
                    )
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


_NamedScopeNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


def _extract_shallow_defs(
    stmts: Sequence[ast.AST],
) -> Dict[str, List[_NamedScopeNode]]:
    """Extracts direct FunctionDef, AsyncFunctionDef, and ClassDef nodes in stmts."""
    defs: Dict[str, List[_NamedScopeNode]] = {}
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


def _extract_class_body_unconditional_def_names(
    cls_node: ast.ClassDef,
) -> frozenset[str]:
    """Returns class-level def/class names unconditionally bound for all class-body uses."""
    currently_bound: Set[str] = set()
    used_when_unbound: Set[str] = set()
    for stmt in cls_node.body:
        shallow_dels: Set[str] = set()
        pep227_loads: Set[str] = set()
        s_loads, s_calls = _extract_shallow_loads_and_calls(
            [stmt],
            deletions_out=shallow_dels,
            pep227_loads_out=pep227_loads,
        )
        for ref in s_loads | s_calls:
            if ref not in currently_bound:
                used_when_unbound.add(ref)
        used_when_unbound.update(pep227_loads)
        if shallow_dels:
            currently_bound.difference_update(shallow_dels)
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            currently_bound.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    currently_bound.add(tgt.id)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and isinstance(stmt.target, ast.Name)
        ):
            currently_bound.add(stmt.target.id)
    return frozenset(currently_bound - used_when_unbound)


def _propagate_top_cell_defs(
    escaped_top_names: Set[str],
    executed_top_names: Set[str],
    scope_cell_defs: Mapping[Tuple[int, str], Sequence[_NamedScopeNode]],
    escaped_def_ids: Set[int],
    executed_def_ids: Set[int],
) -> bool:
    """Propagates escaped_top_names and executed_top_names into root scope_cell_defs."""
    changed = False
    for name in executed_top_names | escaped_top_names:
        mark_esc = name in escaped_top_names
        for d_node in scope_cell_defs.get((0, name), ()):
            did = id(d_node)
            if did not in executed_def_ids:
                executed_def_ids.add(did)
                changed = True
            if (
                mark_esc or isinstance(d_node, ast.ClassDef)
            ) and did not in escaped_def_ids:
                escaped_def_ids.add(did)
                changed = True
    return changed


def _activate_fn_global_and_nonlocal_defs(
    nid: int,
    global_defs_by_fn: Mapping[int, Sequence[Tuple[str, _NamedScopeNode]]],
    nonlocal_defs_by_fn: Mapping[
        int, Sequence[Tuple[int, str, _NamedScopeNode]]
    ],
    scope_cell_defs: Dict[Tuple[int, str], List[_NamedScopeNode]],
    escaped_top_names: Set[str],
    escaped_def_ids: Set[int],
    executed_def_ids: Set[int],
) -> bool:
    """Activates global and nonlocal nested definitions inside an executed function nid."""
    changed = False
    for g_name, g_node in global_defs_by_fn.get(nid, ()):
        gid = id(g_node)
        cell_list = scope_cell_defs.setdefault((0, g_name), [])
        if g_node not in cell_list:
            cell_list.append(g_node)
            changed = True
        if gid not in escaped_def_ids:
            escaped_def_ids.add(gid)
            changed = True
        if gid not in executed_def_ids:
            executed_def_ids.add(gid)
            changed = True
        if g_name not in escaped_top_names:
            escaped_top_names.add(g_name)
            changed = True
    for target_origin, nl_name, nl_node in nonlocal_defs_by_fn.get(nid, ()):
        nl_id = id(nl_node)
        eff_origin = 0 if target_origin < 0 else target_origin
        cell_list = scope_cell_defs.setdefault((eff_origin, nl_name), [])
        if nl_node not in cell_list:
            cell_list.append(nl_node)
            changed = True
        if target_origin < 0 or target_origin in escaped_def_ids:
            if nl_id not in escaped_def_ids:
                escaped_def_ids.add(nl_id)
                changed = True
            if nl_id not in executed_def_ids:
                executed_def_ids.add(nl_id)
                changed = True
            if target_origin < 0 and nl_name not in escaped_top_names:
                escaped_top_names.add(nl_name)
                changed = True
    return changed


def _collect_escaped_and_executed_intra_unit_defs(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
    *,
    use_cache: bool = False,
) -> Tuple[Set[str], Set[int], Set[int]]:
    """Computes (escaped_names, escaped_def_node_ids, executed_def_node_ids)."""
    root_defs = _extract_shallow_defs(candidate_stmts)
    scope_cell_defs: Dict[Tuple[int, str], List[_NamedScopeNode]] = {
        (0, name): list(nodes) for name, nodes in root_defs.items()
    }
    unit_mod = ast.Module(body=list(candidate_stmts), type_ignores=[])
    top_bound_names = (
        _extract_single_node_local_names(unit_mod)
        - _extract_direct_nonlocals(unit_mod, is_global=False)
        - _extract_direct_nonlocals(unit_mod, is_global=True)
    ) | set(root_defs.keys())
    root_origins: Dict[str, int] = {name: 0 for name in top_bound_names}
    all_def_nodes: List[_NamedScopeNode] = []
    defs_by_name: Dict[str, List[_NamedScopeNode]] = {}
    direct_children: Dict[int, List[_NamedScopeNode]] = {}
    body_origins: Dict[int, Dict[str, int]] = {}
    class_local_defs: Dict[int, Dict[str, Tuple[_NamedScopeNode, ...]]] = {}
    class_uncond_defs: Dict[int, frozenset[str]] = {}
    global_defs_by_fn: Dict[int, List[Tuple[str, _NamedScopeNode]]] = {}
    nonlocal_defs_by_fn: Dict[int, List[Tuple[int, str, _NamedScopeNode]]] = {}
    shadowed_map: Dict[int, frozenset[str]] = {}

    scope_stack: List[
        Tuple[
            _NamedScopeNode,
            Dict[str, int],
            frozenset[str],
        ]
    ] = [
        (d_node, root_origins, frozenset())
        for nodes in root_defs.values()
        for d_node in nodes
    ]
    while scope_stack:
        d_node, parent_origins, shadowed_from_root = scope_stack.pop()
        nid = id(d_node)
        all_def_nodes.append(d_node)
        d_name = d_node.name
        if d_name:
            defs_by_name.setdefault(d_name, []).append(d_node)
        c_defs = _extract_shallow_defs(d_node.body)
        c_list = [n for nodes in c_defs.values() for n in nodes]
        direct_children[nid] = c_list
        if isinstance(d_node, ast.ClassDef):
            body_origins[nid] = dict(parent_origins)
            class_local_defs[nid] = {
                k: tuple(v) for k, v in c_defs.items()
            }
            uncond_cls = _extract_class_body_unconditional_def_names(d_node)
            class_uncond_defs[nid] = uncond_cls
            shadowed_map[nid] = shadowed_from_root | uncond_cls
            for c_node in c_list:
                scope_stack.append(
                    (c_node, parent_origins, shadowed_from_root)
                )
        else:
            own_locals = _extract_strict_fn_local_names(d_node)
            own_globals = _extract_direct_nonlocals(d_node, is_global=True)
            own_nonlocals = _extract_direct_nonlocals(d_node, is_global=False)
            fn_shadowed = shadowed_from_root | frozenset(
                (own_locals | own_globals) - own_nonlocals
            )
            cur_origins: Dict[str, int] = {
                k: v
                for k, v in parent_origins.items()
                if k not in own_locals and k not in own_globals
            }
            for loc_name in (own_locals - set(c_defs.keys())) - own_nonlocals - own_globals:
                cur_origins[loc_name] = nid
            for g_name in own_globals:
                cur_origins[g_name] = 0
            for c_name, c_nodes in c_defs.items():
                if c_name in own_globals:
                    cur_origins[c_name] = nid
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
                    for c_node in c_nodes:
                        global_defs_by_fn.setdefault(nid, []).append(
                            (c_name, c_node)
                        )
                elif c_name in own_nonlocals:
                    target_origin = parent_origins.get(c_name, -1)
                    cur_origins[c_name] = (
                        0 if target_origin < 0 else target_origin
                    )
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
                    for c_node in c_nodes:
                        nonlocal_defs_by_fn.setdefault(nid, []).append(
                            (target_origin, c_name, c_node)
                        )
                else:
                    cur_origins[c_name] = nid
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
            body_origins[nid] = cur_origins
            shadowed_map[nid] = fn_shadowed
            for c_node in c_list:
                scope_stack.append((c_node, cur_origins, fn_shadowed))

    top_loads, top_calls = _extract_shallow_loads_and_calls(candidate_stmts)
    escaped_top_names: Set[str] = set(initial_escaped) | top_loads
    executed_top_names: Set[str] = set(top_calls) | escaped_top_names
    escaped_def_ids: Set[int] = set()
    executed_def_ids: Set[int] = set()

    _propagate_top_cell_defs(
        escaped_top_names,
        executed_top_names,
        scope_cell_defs,
        escaped_def_ids,
        executed_def_ids,
    )
    for nodes in root_defs.values():
        for d_node in nodes:
            if _is_self_escaping_def_node(d_node, use_cache=use_cache):
                escaped_def_ids.add(id(d_node))
                executed_def_ids.add(id(d_node))
            elif isinstance(d_node, ast.ClassDef):
                executed_def_ids.add(id(d_node))

    body_info: Dict[int, Tuple[Set[str], Set[str]]] = {}
    changed = True
    while changed:
        changed = _propagate_top_cell_defs(
            escaped_top_names,
            executed_top_names,
            scope_cell_defs,
            escaped_def_ids,
            executed_def_ids,
        )
        for d_node in all_def_nodes:
            nid = id(d_node)
            is_esc = nid in escaped_def_ids
            is_exec = is_esc or (nid in executed_def_ids)
            if not is_exec:
                continue
            changed |= _activate_fn_global_and_nonlocal_defs(
                nid,
                global_defs_by_fn,
                nonlocal_defs_by_fn,
                scope_cell_defs,
                escaped_top_names,
                escaped_def_ids,
                executed_def_ids,
            )
            for c_node in direct_children.get(nid, ()):
                cid = id(c_node)
                if (
                    isinstance(d_node, ast.ClassDef) and is_esc
                ) or _is_self_escaping_def_node(c_node, use_cache=use_cache):
                    if cid not in escaped_def_ids:
                        escaped_def_ids.add(cid)
                        executed_def_ids.add(cid)
                        changed = True
                elif isinstance(c_node, ast.ClassDef):
                    if cid not in executed_def_ids:
                        executed_def_ids.add(cid)
                        changed = True
            if nid not in body_info:
                body_info[nid] = _extract_shallow_loads_and_calls(d_node.body)
            b_loads, b_calls = body_info[nid]
            origins_here = body_origins[nid]
            cls_defs_here = class_local_defs.get(nid, {})
            uncond_cls_here = class_uncond_defs.get(nid, frozenset())
            shadowed_here = shadowed_map[nid]
            for ref_name in b_loads | b_calls:
                if (
                    (is_esc or ref_name in b_loads)
                    and ref_name not in shadowed_here
                    and ref_name not in escaped_top_names
                ):
                    escaped_top_names.add(ref_name)
                    changed = True
                if ref_name in cls_defs_here and ref_name in uncond_cls_here:
                    target_nodes: Sequence[_NamedScopeNode] = cls_defs_here[
                        ref_name
                    ]
                else:
                    origin_scope = origins_here.get(
                        ref_name,
                        0 if ref_name not in shadowed_here else -1,
                    )
                    target_nodes = [
                        *cls_defs_here.get(ref_name, ()),
                        *scope_cell_defs.get((origin_scope, ref_name), ()),
                        *scope_cell_defs.get((nid, ref_name), ()),
                    ]
                for target_node in target_nodes:
                    tid = id(target_node)
                    if (
                        is_esc
                        or ref_name in b_loads
                        or isinstance(target_node, ast.ClassDef)
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
    return (
        escaped_top_names | unambiguous_escaped_def_names,
        escaped_def_ids,
        executed_def_ids,
    )


def _collect_escaped_intra_unit_defs(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Tuple[Set[str], Set[int]]:
    """Computes (escaped_names, escaped_def_node_ids) with lexical-scope definition tracking."""
    esc_names, esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(
        candidate_stmts, initial_escaped
    )
    return esc_names, esc_ids


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
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    own_locals: frozenset[str],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> _FnScopeContext:
    """Builds a _FnScopeContext for analyzing expressions inside a nested function."""
    if scope_ctx is None:
        parent_map: Mapping[ast.AST, ast.AST] = MappingProxyType(
            _get_scope_parent_map(fn_node, use_cache=use_cache)
        )
    elif fn_node.body and fn_node.body[0] not in scope_ctx.parent_map:
        parent_map = MappingProxyType(
            ChainMap(
                _get_scope_parent_map(fn_node, use_cache=use_cache),
                cast(Dict[ast.AST, ast.AST], scope_ctx.parent_map),
            )
        )
    else:
        parent_map = scope_ctx.parent_map
    enc_scopes: Tuple[ast.AST, ...] = (
        (*scope_ctx.enclosing_scopes, scope_ctx.scope_owner)
        if scope_ctx is not None and isinstance(scope_ctx.scope_owner, ast.AST)
        else (scope_ctx.enclosing_scopes if scope_ctx is not None else ())
    )
    return _FnScopeContext(
        fn_locals=(
            (scope_ctx.fn_locals if scope_ctx is not None else frozenset())
            | own_locals
        ),
        shadowed_names=(
            (scope_ctx.shadowed_names if scope_ctx is not None else frozenset())
            | own_locals
        ),
        parent_map=parent_map,
        has_star_import=(
            scope_ctx.has_star_import if scope_ctx is not None else False
        ),
        scope_owner=fn_node,
        mod_tree=scope_ctx.mod_tree if scope_ctx is not None else None,
        enclosing_scopes=enc_scopes,
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
    scope_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> bool:
    """Computes whether a nested function or class can return a closure or carrier."""
    if isinstance(scope_node, (ast.ClassDef, ast.AsyncFunctionDef)):
        unshadowed_builtins = _BUILTIN_NAMES - (
            set(scope_ctx.shadowed_names) if scope_ctx is not None else set()
        )
        has_free_or_nl = bool(
            _extract_cached_scope_free_reads(scope_node, use_cache=use_cache)
            - unshadowed_builtins
        ) or bool(
            _extract_closure_nonlocal_mutations_and_deletions(
                scope_node, use_cache=use_cache
            )[0]
        )
        if isinstance(scope_node, ast.ClassDef):
            has_deferred_member = any(
                isinstance(
                    sub,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Lambda,
                        ast.GeneratorExp,
                    ),
                )
                and sub is not scope_node
                for sub in ast.walk(scope_node)
            )
            if not has_deferred_member:
                return False
            if has_free_or_nl:
                return True
            return any(
                isinstance(sub, (ast.Lambda, ast.GeneratorExp))
                or (
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _is_closure_factory_scope_uncached(
                        sub, scope_ctx, use_cache=use_cache
                    )
                )
                for sub in ast.walk(scope_node)
                if sub is not scope_node
            )
        if has_free_or_nl or any(
            _extract_fn_returned_closure_refs(
                scope_node, scope_ctx, use_cache=use_cache
            )
        ):
            return True
    own_locals = frozenset(_extract_strict_fn_local_names(scope_node))
    local_ctx = _make_nested_fn_scope_context(
        scope_node, own_locals, scope_ctx, use_cache=use_cache
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
        for _is_reb, c_targets, c_values in _iter_carrier_target_entries(curr):
            local_tgts = c_targets & own_locals
            if local_tgts:
                c_calls: Set[str] = set()
                c_reads, c_muts, c_loads = (
                    _extract_carrier_value_closure_and_load_info(
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
    r_reads, r_muts, r_loads = _extract_carrier_value_closure_and_load_info(
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
    *,
    use_cache: bool = True,
) -> bool:
    """Checks whether a nested function or class can return a closure or generator/coroutine."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return False
    if not use_cache:
        return _is_closure_factory_scope_uncached(
            scope_node, scope_ctx, use_cache=False
        )
    cache_key = _make_scope_ctx_cache_key(
        scope_node, "is_closure_factory", scope_ctx
    )
    return bool(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _is_closure_factory_scope_uncached(
                scope_node, scope_ctx, use_cache=True
            ),
        )
    )


def _extract_fn_returned_closure_refs_uncached(
    fn_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (returned_loaded_names, returned_called_names) for fn_node without caching."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset(), frozenset()
    own_locals = _extract_strict_fn_local_names(fn_node)
    local_ctx = _make_nested_fn_scope_context(
        fn_node, frozenset(own_locals), scope_ctx, use_cache=use_cache
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
    _, _, ret_loads = _extract_carrier_value_closure_and_load_info(
        ret_exprs, local_ctx, called_names_out=ret_calls
    )
    return frozenset(ret_loads - own_locals), frozenset(ret_calls - own_locals)


def _extract_fn_returned_closure_refs(
    fn_node: ast.AST, scope_ctx: Optional[_FnScopeContext] = None, *, use_cache: bool = True
) -> Tuple[Set[str], Set[str]]:
    """Extracts non-local (returned_loaded_names, returned_called_names) for a nested function."""
    return _resolve_cached_str_set_pair(
        fn_node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
        _make_scope_ctx_cache_key(fn_node, "ret_closure_refs", scope_ctx),
        lambda c: _extract_fn_returned_closure_refs_uncached(fn_node, scope_ctx, use_cache=c),
        use_cache=use_cache,
    )


def _extract_fn_outer_carrier_targets_uncached(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> frozenset[str]:
    """Computes outer-scope container/variable names mutated by carrier operations in fn_node.

    Per PEP 227, ``global`` declarations inside a ``ClassDef`` body only govern name
    resolution in that immediate class body and do not propagate into nested methods,
    lambdas, comprehensions, or nested classes.
    """
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
    is_root_cls = isinstance(fn_node, ast.ClassDef)
    own_locals = (
        set()
        if is_root_cls
        else _extract_strict_fn_local_names(fn_node)
    )
    local_ctx = _make_nested_fn_scope_context(
        fn_node, frozenset(own_locals), scope_ctx, use_cache=use_cache
    )
    direct_nl = _extract_direct_nonlocals(fn_node)
    direct_g = _extract_direct_nonlocals(fn_node, is_global=True)
    fn_g = set(fn_globals) if is_root_cls else (set(fn_globals) | direct_g)
    cls_g = direct_g if is_root_cls else set()
    targets: Set[str] = set()
    local_closure_syms: Set[str] = set()
    local_edges: List[Tuple[str, Set[str]]] = []
    outer_candidates: List[Tuple[str, Set[str]]] = []
    stack: List[Tuple[ast.AST, Set[str], Set[str], Set[str], Set[str]]] = [
        (stmt, own_locals, direct_nl, fn_g, cls_g) for stmt in fn_node.body
    ]
    while stack:
        sub, cur_locals, cur_nl, cur_fn_g, cur_cls_g = stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_closure_syms.add(sub.name)
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append((hdr, cur_locals, cur_nl, cur_fn_g, cur_cls_g))
            sub_direct_nl = _extract_direct_nonlocals(sub)
            sub_direct_g = _extract_direct_nonlocals(sub, is_global=True)
            if isinstance(sub, ast.ClassDef):
                sub_fn_g = cur_fn_g - sub_direct_nl
                sub_cls_g = sub_direct_g
                sub_own: Set[str] = set()
            else:
                sub_own = _extract_strict_fn_local_names(sub)
                sub_fn_g = (
                    (cur_fn_g - sub_own) - sub_direct_nl
                ) | sub_direct_g
                sub_cls_g = set()
            sub_nl = (
                (
                    (sub_direct_nl & cur_nl) | (sub_direct_nl - cur_locals)
                )
                - sub_fn_g
                - sub_cls_g
            )
            next_locals = (cur_locals | sub_own) - sub_fn_g - sub_cls_g
            for ch in sub.body:
                stack.append((ch, next_locals, sub_nl, sub_fn_g, sub_cls_g))
            continue
        if isinstance(sub, ast.Lambda):
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append((hdr, cur_locals, cur_nl, cur_fn_g, cur_cls_g))
            lam_bound = _extract_strict_fn_local_names(sub)
            lam_locals = cur_locals | lam_bound
            stack.append(
                (
                    sub.body,
                    lam_locals,
                    cur_nl - lam_bound,
                    cur_fn_g - lam_bound,
                    set(),
                )
            )
            continue
        if isinstance(sub, _COMPREHENSION_TYPES):
            comp_exprs, comp_targets = _iter_comprehension_body_exprs_and_targets(
                sub
            )
            comp_locals = cur_locals | comp_targets
            comp_nl = cur_nl - comp_targets
            comp_fn_g = cur_fn_g - comp_targets
            stack.extend(
                [
                    *(
                        [(sub.generators[0].iter, cur_locals, cur_nl, cur_fn_g, cur_cls_g)]
                        if sub.generators
                        else []
                    ),
                    *((expr, comp_locals, comp_nl, comp_fn_g, set()) for expr in comp_exprs),
                ]
            )
            continue
        for is_rebound, c_targets, c_values in _iter_carrier_target_entries(sub):
            c_calls: Set[str] = set()
            c_reads, c_muts, c_loads = (
                _extract_carrier_value_closure_and_load_info(
                    c_values, local_ctx, called_names_out=c_calls
                )
            )
            has_inline = bool(c_reads or c_muts)
            val_refs = c_loads | c_calls
            for t_name in c_targets:
                if t_name in cur_fn_g or t_name in cur_cls_g:
                    continue
                is_outer_target = (
                    (t_name in cur_nl)
                    if is_rebound
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
            stack.append((child_node, cur_locals, cur_nl, cur_fn_g, cur_cls_g))
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
    *,
    use_cache: bool = True,
) -> Set[str]:
    """Extracts outer-scope container/variable names mutated by carrier operations in fn_node."""
    if not isinstance(
        fn_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set()
    if not use_cache:
        return set(
            _extract_fn_outer_carrier_targets_uncached(
                fn_node, fn_globals, scope_ctx, use_cache=False
            )
        )
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_scope_ctx_cache_key(
                fn_node,
                "outer_carriers",
                scope_ctx,
                extra_params=(frozenset(fn_globals),),
            ),
            lambda: _extract_fn_outer_carrier_targets_uncached(
                fn_node, fn_globals, scope_ctx, use_cache=True
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
