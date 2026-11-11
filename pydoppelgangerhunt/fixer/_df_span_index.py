"""AST span indexing, column-bound inference, and PEP 3104 lexical scope validation."""

from __future__ import annotations

import ast
import symtable
from typing import (
    AbstractSet,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from pydoppelgangerhunt.fixer._ast_compat import (
    is_match_as_node,
    is_match_case_node,
    is_match_mapping_node,
    is_match_star_node,
    is_type_alias_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _get_cached_tree_source_lines,
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _register_tree_source_lines,
    _scope_cell_info_cache,
    _suppress_syntax_warnings,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _extract_strict_fn_local_names,
    _get_tree_expr_scope_intervals,
    _get_tree_scope_intervals,
    _is_span_in_scope_header,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_pattern_bound_name,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _TreeEnclosingScopeIndex:
    """Segment-tree interval index over function, lambda, and comprehension scopes in a tree."""

    __slots__ = ("entries", "_size", "_min_start", "_max_end")

    def __init__(
        self,
        mod_tree: ast.AST,
        source_lines: Optional[Sequence[str]] = None,
    ) -> None:
        raw_entries: List[Tuple[int, int, bool, ast.AST]] = []
        for estart, n_end, _span, node in _get_tree_scope_intervals(
            mod_tree, source_lines=source_lines
        ):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raw_entries.append((estart, n_end, False, node))
        for n_start, n_end, _n_col, node in _get_tree_expr_scope_intervals(mod_tree):
            raw_entries.append((n_start, n_end, True, node))
        raw_entries.sort(key=lambda item: (item[0], -item[1]))
        self.entries: Tuple[Tuple[int, int, bool, ast.AST], ...] = tuple(raw_entries)
        n = len(self.entries)
        if n == 0:
            self._size = 0
            self._min_start: Tuple[int, ...] = ()
            self._max_end: Tuple[int, ...] = ()
            return
        size = 1 << (n - 1).bit_length()
        inf_start = 1 << 30
        min_start = [inf_start] * (2 * size)
        max_end = [-1] * (2 * size)
        for idx, (s_ln, e_ln, _is_expr, _node) in enumerate(self.entries):
            leaf = size + idx
            min_start[leaf] = s_ln
            max_end[leaf] = e_ln
        for p in range(size - 1, 0, -1):
            left = 2 * p
            right = left + 1
            min_start[p] = min_start[left]
            max_end[p] = (
                max_end[left] if max_end[left] >= max_end[right] else max_end[right]
            )
        self._size = size
        self._min_start = tuple(min_start)
        self._max_end = tuple(max_end)

    def query_enclosing(
        self, u_start: int, u_end: int
    ) -> List[Tuple[int, int, bool, ast.AST]]:
        """Returns all indexed scope intervals with start <= u_start and end >= u_end in O((1 + k) log S)."""
        size = self._size
        if size == 0:
            return []
        min_start = self._min_start
        max_end = self._max_end
        if min_start[1] > u_start or max_end[1] < u_end:
            return []
        entries = self.entries
        n = len(entries)
        matches: List[Tuple[int, int, bool, ast.AST]] = []
        stack: List[int] = [1]
        while stack:
            p = stack.pop()
            if p >= size:
                idx = p - size
                if idx < n:
                    matches.append(entries[idx])
                continue
            right = 2 * p + 1
            if min_start[right] <= u_start and max_end[right] >= u_end:
                stack.append(right)
            left = 2 * p
            if min_start[left] <= u_start and max_end[left] >= u_end:
                stack.append(left)
        return matches

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns empty tuple; root ownership is tracked on the owning _ScopeCacheKey."""
        return ()


def _get_tree_enclosing_scope_index(
    mod_tree: ast.AST,
    source_lines: Optional[Sequence[str]] = None,
) -> _TreeEnclosingScopeIndex:
    """Returns or builds cached _TreeEnclosingScopeIndex for mod_tree."""
    if source_lines is not None:
        _register_tree_source_lines(mod_tree, source_lines)
    eff_lines = (
        source_lines
        if source_lines is not None
        else _get_cached_tree_source_lines(mod_tree)
    )
    if _is_ephemeral_unit_ast(mod_tree):
        return _TreeEnclosingScopeIndex(mod_tree, source_lines=eff_lines)
    return cast(
        _TreeEnclosingScopeIndex,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(
                mod_tree,
                "tree_enclosing_index",
                params=(bool(eff_lines),),
            ),
            lambda: _TreeEnclosingScopeIndex(mod_tree, source_lines=eff_lines),
        ),
    )


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
    *,
    source_lines: Optional[Sequence[str]] = None,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing function, lambda, and comprehension scopes ordered outermost to innermost."""
    index = _get_tree_enclosing_scope_index(mod_tree, source_lines=source_lines)
    chain: List[Tuple[int, int, int, int, ast.AST]] = []
    for estart, n_end, is_expr_scope, node in index.query_enclosing(u_start, u_end):
        if not is_expr_scope:
            n_lineno = int(getattr(node, "lineno", estart) or estart)
            if (
                n_lineno <= 0
                or u_start < n_lineno
                or u_end > n_end
                or _is_span_in_scope_header(
                    node,
                    u_start,
                    u_end,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                )
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
                _, e_col = _node_earliest_start_pos(node, source_lines=source_lines)
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
            chain.append(
                (
                    estart,
                    n_end,
                    n_col,
                    -(n_end_col if isinstance(n_end_col, int) else 0),
                    node,
                )
            )
            continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
        if (
            estart <= 0
            or u_start < estart
            or u_end > n_end
            or not _node_strictly_encloses_bounds(
                node,
                u_start,
                u_end,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
            or _is_span_in_scope_header(
                node,
                u_start,
                u_end,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
        ):
            continue
        expr_end_col = getattr(node, "end_col_offset", None)
        chain.append(
            (
                estart,
                n_end,
                n_col,
                -(expr_end_col if isinstance(expr_end_col, int) else 0),
                node,
            )
        )
    chain.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
    return [
        (estart, n_end, node) for estart, n_end, _col, _ecol, node in chain
    ]


def _stmt_dump_set(stmts: Sequence[ast.AST], idx: int) -> Set[str]:
    """Builds the set of AST dumps for a boundary statement (and inner Expr value)."""
    if not stmts:
        return set()
    target = stmts[idx]
    dumps = {ast.dump(target)}
    if isinstance(target, ast.Expr):
        dumps.add(ast.dump(target.value))
    return dumps


class _SpanLineIndex:
    """Cached per-tree compact line index and lazy AST dump cache for O(1) boundary lookups."""

    __slots__ = (
        "by_start_line",
        "by_end_line",
        "stmts_by_start_line",
        "stmts_by_end_line",
        "one_line_scope_lines",
        "missing_end_lineno_starts",
        "missing_col_spans",
        "closure_like_nodes",
        "_decorated_starts",
        "_dump_cache",
    )

    def __init__(
        self,
        root_node: ast.AST,
        source_lines: Optional[Sequence[str]] = None,
    ) -> None:
        by_start: Dict[int, List[ast.AST]] = {}
        by_end: Dict[int, List[ast.AST]] = {}
        self.stmts_by_start_line: Dict[int, List[ast.stmt]] = {}
        self.stmts_by_end_line: Dict[int, List[Tuple[ast.stmt, int, bool]]] = {}
        one_line_scopes: Set[int] = set()
        missing_end_starts: List[int] = []
        missing_cols: List[Tuple[int, int]] = []
        closure_nodes: List[ast.AST] = []
        decorated_starts: Dict[int, Tuple[int, Optional[int]]] = {}
        self._dump_cache: Dict[int, str] = {}

        stack: List[ast.AST] = [root_node]
        while stack:
            node = stack.pop()
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Lambda,
                    *_COMPREHENSION_TYPES,
                ),
            ):
                closure_nodes.append(node)
            is_stmt_or_expr = isinstance(node, (ast.stmt, ast.expr))
            if is_stmt_or_expr:
                n_start, n_col = _node_earliest_start_pos(
                    node, source_lines=source_lines
                )
                raw_lineno = getattr(node, "lineno", None)
                raw_col = getattr(node, "col_offset", None)
                if isinstance(n_start, int) and (
                    n_start != raw_lineno or n_col != raw_col
                ):
                    decorated_starts[id(node)] = (
                        n_start,
                        n_col if isinstance(n_col, int) else None,
                    )
                raw_end = getattr(node, "end_lineno", None)
                n_end_col = getattr(node, "end_col_offset", None)
                if isinstance(n_start, int):
                    if not isinstance(raw_end, int):
                        missing_end_starts.append(n_start)
                    elif not isinstance(n_col, int) or not isinstance(
                        n_end_col, int
                    ):
                        missing_cols.append((n_start, raw_end))

                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if (
                        isinstance(raw_lineno, int)
                        and isinstance(raw_end, int)
                        and raw_lineno == raw_end
                    ):
                        one_line_scopes.add(raw_lineno)

                n_end = raw_end if isinstance(raw_end, int) else n_start
                if isinstance(n_start, int) and isinstance(n_end, int):
                    by_start.setdefault(n_start, []).append(node)
                    if n_end != n_start:
                        by_end.setdefault(n_end, []).append(node)
                    if (
                        isinstance(node, ast.stmt)
                        and isinstance(raw_end, int)
                        and isinstance(n_col, int)
                        and isinstance(n_end_col, int)
                    ):
                        self.stmts_by_start_line.setdefault(n_start, []).append(
                            node
                        )
                        eff_header_line = (
                            raw_lineno
                            if isinstance(raw_lineno, int)
                            else n_start
                        )
                        if (
                            eff_header_line != n_start
                            and eff_header_line == raw_end
                        ):
                            self.stmts_by_start_line.setdefault(
                                eff_header_line, []
                            ).append(node)
                        has_compound_child = eff_header_line < raw_end and any(
                            isinstance(ch, (ast.stmt, ast.ExceptHandler))
                            or is_match_case_node(ch)
                            for ch in ast.iter_child_nodes(node)
                        )
                        self.stmts_by_end_line.setdefault(raw_end, []).append(
                            (node, n_start, has_compound_child)
                        )

            children = list(ast.iter_child_nodes(node))
            for ch in reversed(children):
                stack.append(ch)

        self.by_start_line: Dict[int, Tuple[ast.AST, ...]] = {
            ln: tuple(nodes) for ln, nodes in by_start.items()
        }
        self.by_end_line: Dict[int, Tuple[ast.AST, ...]] = {
            ln: tuple(nodes) for ln, nodes in by_end.items()
        }
        self._decorated_starts = decorated_starts
        self.one_line_scope_lines: frozenset[int] = frozenset(one_line_scopes)
        self.missing_end_lineno_starts: Tuple[int, ...] = tuple(
            missing_end_starts
        )
        self.missing_col_spans: Tuple[Tuple[int, int], ...] = tuple(
            missing_cols
        )
        self.closure_like_nodes: Tuple[ast.AST, ...] = tuple(closure_nodes)

    def node_coords(
        self, node: ast.AST
    ) -> Tuple[int, int, Optional[int], Optional[int]]:
        """Returns (n_start, n_end, n_col, n_end_col) for an indexed AST node."""
        dec = self._decorated_starts.get(id(node))
        if dec is not None:
            n_start, n_col = dec
        else:
            raw_ln = getattr(node, "lineno", 0)
            n_start = raw_ln if isinstance(raw_ln, int) else 0
            raw_col = getattr(node, "col_offset", None)
            n_col = raw_col if isinstance(raw_col, int) else None
        raw_end = getattr(node, "end_lineno", None)
        n_end = raw_end if isinstance(raw_end, int) else n_start
        raw_end_col = getattr(node, "end_col_offset", None)
        n_end_col = raw_end_col if isinstance(raw_end_col, int) else None
        return n_start, n_end, n_col, n_end_col

    def get_dump(self, node: ast.AST) -> str:
        """Returns the cached ast.dump(node) string."""
        nid = id(node)
        cached = self._dump_cache.get(nid)
        if cached is None:
            cached = ast.dump(node)
            self._dump_cache[nid] = cached
        return cached

    def iter_boundary_nodes(
        self, u_start: int, u_end: int
    ) -> Sequence[ast.AST]:
        """Returns indexed stmt/expr AST nodes starting at u_start or ending at u_end."""
        start_nodes = self.by_start_line.get(u_start, ())
        if u_start == u_end:
            extra_end = self.by_end_line.get(u_end, ())
            if not extra_end:
                return start_nodes
            return (*start_nodes, *extra_end)
        end_nodes = self.by_end_line.get(u_end, ())
        end_same_line = self.by_start_line.get(u_end, ())
        if not end_nodes and not end_same_line:
            return start_nodes
        seen_ids = {id(n) for n in start_nodes}
        combined = list(start_nodes)
        for node in (*end_nodes, *end_same_line):
            _s, n_end, _c, _ec = self.node_coords(node)
            if n_end == u_end and id(node) not in seen_ids:
                seen_ids.add(id(node))
                combined.append(node)
        return combined

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns empty tuple; root ownership is indexed on the owning _ScopeCacheKey(root_node, 'span_line_index')."""
        return ()


def _get_span_line_index(
    root_node: ast.AST,
    source_lines: Optional[Sequence[str]] = None,
) -> _SpanLineIndex:
    """Returns or builds the cached _SpanLineIndex for root_node."""
    if source_lines is not None:
        _register_tree_source_lines(root_node, source_lines)
    eff_source_lines = (
        source_lines
        if source_lines is not None
        else _get_cached_tree_source_lines(root_node)
    )
    return cast(
        _SpanLineIndex,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(
                root_node,
                "span_line_index",
                params=(bool(eff_source_lines),) if eff_source_lines is not None else (),
            ),
            lambda: (
                _SpanLineIndex(root_node)
                if eff_source_lines is None
                else _SpanLineIndex(root_node, source_lines=eff_source_lines)
            ),
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
    for node in line_index.iter_boundary_nodes(u_start, u_end):
        n_start, n_end, n_col, n_end_col = line_index.node_coords(node)
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
    for node in line_index.iter_boundary_nodes(u_start, u_end):
        live_end = getattr(node, "end_lineno", None)
        live_end_col = getattr(node, "end_col_offset", None)
        if not isinstance(live_end, int) or not isinstance(live_end_col, int):
            return True
    if u_start_col is not None and u_end_col is not None:
        return False
    if (
        u_start == u_end
        and u_start in line_index.one_line_scope_lines
        and (u_start_col is None or u_end_col is None)
    ):
        return True
    if isinstance(scope_node, (ast.Lambda, *_COMPREHENSION_TYPES)) and (
        (
            u_start == getattr(scope_node, "lineno", None)
            and u_start_col is None
        )
        or (
            u_end == getattr(scope_node, "end_lineno", None)
            and u_end_col is None
        )
    ):
        return True

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


def _extract_direct_nonlocals(
    scope_node: ast.AST,
    *,
    is_global: bool = False,
) -> Set[str]:
    """Extracts variable names declared nonlocal (or global) directly in scope_node."""
    names: Set[str] = set()
    raw_body = getattr(scope_node, "body", None)
    if not isinstance(raw_body, (list, tuple)):
        return names
    stack: List[ast.AST] = list(raw_body)
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
    if is_type_alias_node(node):
        target_name = getattr(node, "name", None)
        return target_name.id if isinstance(target_name, ast.Name) else None
    if (
        is_match_as_node(node)
        or is_match_star_node(node)
        or is_match_mapping_node(node)
    ):
        return _extract_pattern_bound_name(node)
    return None


def _scope_has_use_before_lexical_declaration(
    scope_node: ast.AST,
    target_decls: AbstractSet[str],
) -> bool:
    """Returns True if any name in target_decls is used, assigned, or annotated prior to its global/nonlocal declaration in scope_node."""
    raw_body = getattr(scope_node, "body", None)
    if not target_decls or not isinstance(raw_body, (list, tuple)):
        return False
    seen_in_scope: Set[str] = set()
    walk_stack: List[ast.AST] = list(reversed(raw_body))
    while walk_stack:
        node = walk_stack.pop()
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            if any(name in seen_in_scope for name in node.names):
                return True
            continue
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in target_decls
        ):
            return True
        if isinstance(node, _INNER_SCOPE_TYPES):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in target_decls
            ):
                seen_in_scope.add(node.name)
            has_type_params = bool(getattr(node, "type_params", ()))
            for hdr in reversed(
                iter_nested_scope_header_nodes(
                    node, include_annotations=not has_type_params
                )
            ):
                walk_stack.append(hdr)
            if isinstance(node, _COMPREHENSION_TYPES) and node.generators:
                walk_stack.append(node.generators[0].iter)
            continue
        if isinstance(node, ast.Name):
            if node.id in target_decls:
                seen_in_scope.add(node.id)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound in target_decls:
                    seen_in_scope.add(bound)
            continue
        if isinstance(node, ast.ExceptHandler):
            if isinstance(node.name, str) and node.name in target_decls:
                seen_in_scope.add(node.name)
        else:
            pat_name = _extract_type_alias_or_pattern_name(node)
            if pat_name is not None and pat_name in target_decls:
                seen_in_scope.add(pat_name)
        for ch in reversed(tuple(ast.iter_child_nodes(node))):
            walk_stack.append(ch)
    return False


def _walk_invalid_lexical_scope_declarations(
    initial_stack: List[Tuple[ast.AST, frozenset[str], frozenset[str], bool]],
) -> bool:
    """Walks initial_stack checking for PEP 3104 nonlocal/global declaration violations."""
    stack = initial_stack
    while stack:
        curr, cur_bound, cur_blocked, cur_rooted = stack.pop()
        if isinstance(curr, ast.Module):
            d_nl = _extract_direct_nonlocals(curr, is_global=False)
            d_g = _extract_direct_nonlocals(curr, is_global=True)
            if d_nl & d_g:
                return True
            if cur_rooted and d_nl:
                return True
            if (d_nl or d_g) and _scope_has_use_before_lexical_declaration(
                curr, d_nl | d_g
            ):
                return True
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
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
            if (d_nl or d_g) and _scope_has_use_before_lexical_declaration(
                curr, d_nl | d_g
            ):
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


def _compute_scope_invalid_lexical_declarations(
    root_scope: ast.AST,
    is_module_rooted: bool,
    source_lines: Optional[Sequence[str]] = None,
) -> bool:
    """Checks whether root_scope has invalid lexical declarations using symtable fast-path for Module."""
    eff_lines = (
        source_lines
        if isinstance(source_lines, (list, tuple)) and source_lines
        else _get_cached_tree_source_lines(root_scope)
    )
    if (
        is_module_rooted
        and isinstance(root_scope, ast.Module)
        and isinstance(eff_lines, (list, tuple))
        and eff_lines
    ):
        try:
            with _suppress_syntax_warnings():
                symtable.symtable("\n".join(eff_lines), "<symtable>", "exec")
            return False
        except (SyntaxError, TypeError, ValueError, RecursionError, MemoryError):
            pass
    return _walk_invalid_lexical_scope_declarations(
        [(root_scope, frozenset(), frozenset(), is_module_rooted)]
    )


def _scope_has_invalid_lexical_declarations(
    root_scope: ast.AST,
    is_module_rooted: bool,
    source_lines: Optional[Sequence[str]] = None,
) -> bool:
    """Checks (and caches per AST root scope) whether root_scope has invalid lexical declarations."""
    if isinstance(source_lines, (list, tuple)) and source_lines:
        _register_tree_source_lines(root_scope, source_lines)
    return bool(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(
                root_scope,
                "invalid_lex_decls",
                params=(is_module_rooted,),
            ),
            lambda: _compute_scope_invalid_lexical_declarations(
                root_scope, is_module_rooted, source_lines=source_lines
            ),
        )
    )


def _has_invalid_lexical_scope_declarations(
    candidate_stmts: Sequence[ast.AST],
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]] = (),
    mod_tree: Optional[ast.AST] = None,
    *,
    source_lines: Optional[Sequence[str]] = None,
) -> bool:
    """Returns True if any scope in enclosing_chain or candidate_stmts violates PEP 3104 nonlocal/global rules."""
    is_module_rooted = isinstance(mod_tree, ast.Module)
    mod_has_invalid = (
        _scope_has_invalid_lexical_declarations(
            mod_tree, True, source_lines=source_lines
        )
        if isinstance(mod_tree, ast.Module)
        else True
    )
    if (
        enclosing_chain
        and mod_has_invalid
        and _scope_has_invalid_lexical_declarations(
            enclosing_chain[0][2], is_module_rooted
        )
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
            params = _extract_arg_names(
                enc_fn.args
            ) | _extract_node_type_param_names(enc_fn)
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
