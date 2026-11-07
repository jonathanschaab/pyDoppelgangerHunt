"""AST span indexing, column-bound inference, and PEP 3104 lexical scope validation."""

from __future__ import annotations

import ast
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _extract_strict_fn_local_names,
    _is_span_in_scope_header,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_pattern_bound_name,
)
from pydoppelgangerhunt.fixer.source import _get_tree_scope_intervals
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


def _get_tree_expr_scope_intervals(
    tree: ast.AST,
) -> Tuple[Tuple[int, int, int, ast.AST], ...]:
    """Returns cached (start, end, col_offset, node) tuples for Lambda and comprehension scopes."""

    def _compute() -> Tuple[Tuple[int, int, int, ast.AST], ...]:
        out: List[Tuple[int, int, int, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Lambda, *_COMPREHENSION_TYPES)):
                n_start = int(getattr(node, "lineno", 0) or 0)
                n_end = int(getattr(node, "end_lineno", n_start) or n_start)
                n_col = int(getattr(node, "col_offset", 0) or 0)
                if n_start > 0:
                    out.append((n_start, n_end, n_col, node))
        return tuple(out)

    return cast(
        Tuple[Tuple[int, int, int, ast.AST], ...],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(tree, "expr_scope_intervals"),
            _compute,
        ),
    )


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing function, lambda, and comprehension scopes ordered outermost to innermost."""
    chain: List[Tuple[int, int, int, int, ast.AST]] = []
    src_lines: Optional[Sequence[str]] = getattr(
        mod_tree, "_source_lines", None
    )
    for estart, n_end, _span, node in _get_tree_scope_intervals(mod_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
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
            _, e_col = _node_earliest_start_pos(node, source_lines=src_lines)
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
    for n_start, n_end, n_col, node in _get_tree_expr_scope_intervals(mod_tree):
        if (
            n_start <= 0
            or u_start < n_start
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
                n_start,
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


class _IndexedSpanNode(NamedTuple):
    """Pre-indexed AST node entry for fast line/column span resolution."""

    node: ast.AST
    n_start: int
    n_end: int
    n_col: Optional[int]
    n_end_col: Optional[int]

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns the AST node retained by this span index entry."""
        return (self.node,) if isinstance(self.node, ast.AST) else ()


class _SpanLineIndex:
    """Cached per-tree line index and lazy AST dump cache for O(1) boundary lookups."""

    __slots__ = (
        "by_start_line",
        "by_end_line",
        "stmts_by_start_line",
        "stmts_by_end_line",
        "one_line_scope_lines",
        "missing_end_lineno_starts",
        "missing_col_spans",
        "closure_like_nodes",
        "_dump_cache",
    )

    def __init__(self, root_node: ast.AST) -> None:
        self.by_start_line: Dict[int, List[_IndexedSpanNode]] = {}
        self.by_end_line: Dict[int, List[_IndexedSpanNode]] = {}
        self.stmts_by_start_line: Dict[int, List[ast.stmt]] = {}
        self.stmts_by_end_line: Dict[int, List[Tuple[ast.stmt, int, bool]]] = {}
        one_line_scopes: Set[int] = set()
        missing_end_starts: List[int] = []
        missing_cols: List[Tuple[int, int]] = []
        closure_nodes: List[ast.AST] = []
        self._dump_cache: Dict[int, str] = {}
        src_lines: Optional[Sequence[str]] = getattr(
            root_node, "_source_lines", None
        )

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
            n_start, n_col = _node_earliest_start_pos(
                node, source_lines=src_lines
            )
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

            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                def_lineno = getattr(node, "lineno", None)
                if (
                    isinstance(def_lineno, int)
                    and isinstance(raw_end, int)
                    and def_lineno == raw_end
                ):
                    one_line_scopes.add(def_lineno)

            n_end = raw_end if isinstance(raw_end, int) else n_start
            if isinstance(n_start, int) and isinstance(n_end, int):
                entry = _IndexedSpanNode(
                    node=node,
                    n_start=n_start,
                    n_end=n_end,
                    n_col=n_col if isinstance(n_col, int) else None,
                    n_end_col=n_end_col if isinstance(n_end_col, int) else None,
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
                    stmt_lineno = getattr(node, "lineno", n_start)
                    eff_header_line = (
                        stmt_lineno
                        if isinstance(stmt_lineno, int)
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
                        isinstance(ch, ast.stmt)
                        or type(ch).__name__ in ("ExceptHandler", "match_case")
                        for ch in ast.iter_child_nodes(node)
                    )
                    self.stmts_by_end_line.setdefault(raw_end, []).append(
                        (node, n_start, has_compound_child)
                    )

            children = list(ast.iter_child_nodes(node))
            for ch in reversed(children):
                stack.append(ch)

        self.one_line_scope_lines: frozenset[int] = frozenset(one_line_scopes)
        self.missing_end_lineno_starts: Tuple[int, ...] = tuple(
            missing_end_starts
        )
        self.missing_col_spans: Tuple[Tuple[int, int], ...] = tuple(
            missing_cols
        )
        self.closure_like_nodes: Tuple[ast.AST, ...] = tuple(closure_nodes)

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

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns all AST nodes retained across the span-line index maps."""
        nodes: List[ast.AST] = []
        seen_ids: Set[int] = set()
        for entry_map in (self.by_start_line, self.by_end_line):
            for entries in entry_map.values():
                for entry in entries:
                    if (
                        isinstance(entry.node, ast.AST)
                        and id(entry.node) not in seen_ids
                    ):
                        seen_ids.add(id(entry.node))
                        nodes.append(entry.node)
        for stmts in self.stmts_by_start_line.values():
            for stmt in stmts:
                if isinstance(stmt, ast.AST) and id(stmt) not in seen_ids:
                    seen_ids.add(id(stmt))
                    nodes.append(stmt)
        for end_tuples in self.stmts_by_end_line.values():
            for stmt, _, _ in end_tuples:
                if isinstance(stmt, ast.AST) and id(stmt) not in seen_ids:
                    seen_ids.add(id(stmt))
                    nodes.append(stmt)
        for c_node in self.closure_like_nodes:
            if isinstance(c_node, ast.AST) and id(c_node) not in seen_ids:
                seen_ids.add(id(c_node))
                nodes.append(c_node)
        return tuple(nodes)


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
    cls_name = type(node).__name__
    if cls_name == "TypeAlias":
        target_name = getattr(node, "name", None)
        return target_name.id if isinstance(target_name, ast.Name) else None
    if cls_name in ("MatchAs", "MatchStar", "MatchMapping"):
        return _extract_pattern_bound_name(node)
    return None


def _walk_invalid_lexical_scope_declarations(
    initial_stack: List[Tuple[ast.AST, frozenset[str], frozenset[str], bool]],
) -> bool:
    """Walks initial_stack checking for PEP 3104 nonlocal/global declaration violations."""
    stack = initial_stack
    while stack:
        curr, cur_bound, cur_blocked, cur_rooted = stack.pop()
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
) -> bool:
    """Checks whether root_scope has invalid lexical declarations using compile() fast-path for Module."""
    if is_module_rooted and isinstance(root_scope, ast.Module):
        try:
            compile(root_scope, "<ast>", "exec")
            return False
        except (SyntaxError, TypeError, ValueError):
            pass
    return _walk_invalid_lexical_scope_declarations(
        [(root_scope, frozenset(), frozenset(), is_module_rooted)]
    )


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
            lambda: _compute_scope_invalid_lexical_declarations(
                root_scope, is_module_rooted
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
    if isinstance(
        mod_tree, ast.Module
    ) and _scope_has_invalid_lexical_declarations(mod_tree, True):
        return True
    if (
        enclosing_chain
        and not isinstance(mod_tree, ast.Module)
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
