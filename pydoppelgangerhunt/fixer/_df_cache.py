"""Dataflow cache structures, LRU memoization, and AST coordinate caching."""

from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import logging
import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from pydoppelgangerhunt.config import clear_closure_warning_cache
from pydoppelgangerhunt.source_lines import parse_unit_coord

logger = logging.getLogger(__name__)

_CacheValT = TypeVar("_CacheValT")

_downstream_cache_lock: threading.RLock = threading.RLock()
_MAX_DOWNSTREAM_CACHE_SIZE: int = 1024
_MAX_SCOPE_CACHE_SIZE: int = 1024
_MAX_PARSED_TREE_CACHE_SIZE: int = 128
_downstream_reads_cache: OrderedDict[Tuple[Any, ...], frozenset[str]] = OrderedDict()
_parsed_tree_cache: OrderedDict[Tuple[str, int], Optional[ast.AST]] = OrderedDict()

_scope_parent_maps: OrderedDict[ast.AST, Dict[ast.AST, ast.AST]] = OrderedDict()
_scope_stmts_by_end: OrderedDict[ast.AST, Dict[int, List[ast.stmt]]] = OrderedDict()
_scope_stmts_by_start: OrderedDict[ast.AST, Dict[int, List[ast.stmt]]] = OrderedDict()
_scope_closures_cache: OrderedDict[Any, List[Tuple[Any, Set[str]]]] = OrderedDict()
_scope_loops_cache: OrderedDict[ast.AST, List[Tuple[int, int, int]]] = OrderedDict()
_scope_try_with_cache: OrderedDict[ast.AST, List[ast.AST]] = OrderedDict()
_warned_lenient_closure: List[bool] = [False]


_SCOPE_ROOT_TYPES: Tuple[type, ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _evict_tree_scopes(tree: Optional[ast.AST]) -> None:
    """Purges scope-cache entries belonging to an evicted AST tree."""
    if tree is None or not isinstance(tree, ast.AST):
        return
    scope_nodes = {
        node
        for node in ast.walk(tree)
        if isinstance(node, _SCOPE_ROOT_TYPES)
    }
    scope_nodes.add(tree)
    with _downstream_cache_lock:
        for node in scope_nodes:
            _scope_parent_maps.pop(node, None)
            _scope_stmts_by_end.pop(node, None)
            _scope_stmts_by_start.pop(node, None)
            _scope_closures_cache.pop(node, None)
            _scope_loops_cache.pop(node, None)
            _scope_try_with_cache.pop(node, None)
        closure_keys_to_remove = [
            k
            for k in _scope_closures_cache
            if isinstance(k, tuple)
            and len(k) == 2
            and (k[0] in scope_nodes or k[1] in scope_nodes)
        ]
        for k in closure_keys_to_remove:
            _scope_closures_cache.pop(k, None)


def _store_lru_cache_entry(
    cache: OrderedDict[Any, _CacheValT],
    key: Any,
    value: _CacheValT,
    max_size: int,
) -> _CacheValT:
    """Stores key-value pair in an OrderedDict LRU cache with eviction outside lock."""
    evicted_trees: List[ast.AST] = []
    with _downstream_cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        while len(cache) >= max_size and cache:
            _, evicted_val = cache.popitem(last=False)
            if cache is _parsed_tree_cache and isinstance(evicted_val, ast.AST):
                evicted_trees.append(evicted_val)
        cache[key] = value
        cache.move_to_end(key)
    for evicted_tree in evicted_trees:
        _evict_tree_scopes(evicted_tree)
    return value


def _get_or_compute_scope_cached(
    cache: OrderedDict[Any, _CacheValT],
    scope_node: Any,
    compute_fn: Callable[[], _CacheValT],
) -> _CacheValT:
    """Retrieves cached scope metadata or computes, caches, and returns it thread-safely."""
    try:
        with _downstream_cache_lock:
            if scope_node in cache:
                cache.move_to_end(scope_node)
                return cache[scope_node]
    except TypeError:
        return compute_fn()

    # Concurrency Design Note: compute_fn() executes outside _downstream_cache_lock to
    # eliminate lock contention under multi-threading; transient duplicate computation during
    # concurrent misses is harmless because compute_fn is deterministic and pure.
    computed = compute_fn()

    try:
        return _store_lru_cache_entry(cache, scope_node, computed, _MAX_SCOPE_CACHE_SIZE)
    except TypeError:
        return computed


def _get_or_parse_tree(
    source_text: str,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Retrieves cached AST tree or parses source_text into an AST tree."""
    if not source_text.strip():
        return None
    content_hash = (
        str(source_digest)
        if source_digest is not None
        else hashlib.sha256(source_text.encode("utf-8", errors="replace")).hexdigest()
    )
    key = (content_hash, len(source_text))
    with _downstream_cache_lock:
        if key in _parsed_tree_cache:
            cached_tree = _parsed_tree_cache[key]
            _parsed_tree_cache.move_to_end(key)
            return cached_tree

    parsed: Optional[ast.AST] = None
    try:
        parsed = ast.parse(source_text)
        setattr(parsed, "_immutable_cached", True)
    except (SyntaxError, ValueError, UnicodeDecodeError, RecursionError, MemoryError):
        parsed = None

    stored_tree = _store_lru_cache_entry(
        _parsed_tree_cache, key, parsed, _MAX_PARSED_TREE_CACHE_SIZE
    )
    return stored_tree


def _parse_source_tree(
    source_text: str,
    tree: Optional[ast.AST] = None,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Parses Python source code into an AST tree if not already provided."""
    if tree is not None:
        return tree
    return _get_or_parse_tree(source_text, source_digest=source_digest)


def _get_valid_unit_bounds(unit: Any) -> Optional[Tuple[int, int]]:
    """Extracts positive start and end line coordinates from a unit dict."""
    if not isinstance(unit, dict):
        return None
    try:
        u_start = parse_unit_coord(unit, "start", default=0)
        u_end = parse_unit_coord(unit, "end", default=u_start)
    except (ValueError, TypeError, AttributeError):
        return None
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None
    return u_start, u_end


def _extract_first_unit_coord(
    unit: Dict[str, Any], keys: Sequence[str]
) -> Optional[int]:
    """Returns the first successfully parsed coordinate among candidate keys."""
    if not isinstance(unit, dict):
        return None
    for key in keys:
        if key in unit and unit.get(key) is not None:
            try:
                val = parse_unit_coord(unit, key, default=None)
                if val is not None:
                    return val
            except (ValueError, TypeError, AttributeError):
                pass
    return None


def _extract_unit_end_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts end column offset from unit dictionary if present."""
    return _extract_first_unit_coord(unit, ("end_col", "end_col_offset"))


def _get_scope_parent_map(scope_node: ast.AST) -> Dict[ast.AST, ast.AST]:
    """Retrieves or builds cached parent map for AST nodes within the scope."""
    def _compute() -> Dict[ast.AST, ast.AST]:
        parent_map: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(scope_node):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent
        return parent_map

    return _get_or_compute_scope_cached(_scope_parent_maps, scope_node, _compute)


def _index_scope_stmts(
    scope_node: ast.AST, *, use_end_lineno: bool
) -> Dict[int, List[ast.stmt]]:
    """Indexes statement nodes within a scope by start or end line number."""
    m: Dict[int, List[ast.stmt]] = {}
    for node in ast.walk(scope_node):
        if node is scope_node or not isinstance(node, ast.stmt):
            continue
        lineno = getattr(node, "lineno", 0) or 0
        line_key = (
            (getattr(node, "end_lineno", None) or lineno)
            if use_end_lineno
            else lineno
        )
        if line_key > 0:
            m.setdefault(line_key, []).append(node)
    return m


def _get_scope_stmts_by_end_lineno(
    scope_node: ast.AST,
) -> Dict[int, List[ast.stmt]]:
    """Retrieves or builds cached map of statement end line numbers to statement nodes."""
    return _get_or_compute_scope_cached(
        _scope_stmts_by_end,
        scope_node,
        lambda: _index_scope_stmts(scope_node, use_end_lineno=True),
    )


def _get_scope_stmts_by_start_lineno(
    scope_node: ast.AST,
) -> Dict[int, List[ast.stmt]]:
    """Retrieves or builds cached map of statement start line numbers to statement nodes."""
    return _get_or_compute_scope_cached(
        _scope_stmts_by_start,
        scope_node,
        lambda: _index_scope_stmts(scope_node, use_end_lineno=False),
    )


def _resolve_unit_ast_end_col(
    scope_node: ast.AST, unit: Dict[str, Any]
) -> Optional[int]:
    """Resolves end column offset from statement boundaries within the enclosing scope."""
    bounds = _get_valid_unit_bounds(unit)
    if bounds is None:
        return None
    u_start, u_end = bounds

    parent_map = _get_scope_parent_map(scope_node)
    stmts_by_end = _get_scope_stmts_by_end_lineno(scope_node)
    matching_stmts = list(stmts_by_end.get(u_end, []))

    inner_stmts: List[ast.stmt] = []
    if matching_stmts:
        matching_set = set(matching_stmts)
        non_leaf_stmts: Set[ast.stmt] = set()
        for s in matching_stmts:
            curr = parent_map.get(s)
            while curr is not None:
                if isinstance(curr, ast.stmt) and curr in matching_set:
                    if curr in non_leaf_stmts:
                        break
                    non_leaf_stmts.add(curr)
                curr = parent_map.get(curr)

        inner_stmts = [s for s in matching_stmts if s not in non_leaf_stmts]
    if not inner_stmts:
        logger.debug(
            "No statement ending at line %d found in scope; falling back to whole-line",
            u_end,
        )
        return None

    inner_stmts.sort(key=lambda s: getattr(s, "col_offset", 0))
    start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
    if start_col is not None and (
        u_start < u_end or unit.get("kind") in ("comprehension", "complex_expr")
    ):
        return None
    if len(inner_stmts) == 1:
        target_stmt = inner_stmts[0]
        if getattr(target_stmt, "lineno", 0) < u_start:
            return None
    elif u_start == u_end and start_col is not None:
        if any(
            (0 if getattr(s, "lineno", 0) < u_start else getattr(s, "col_offset", 0))
            < start_col
            < (getattr(s, "end_col_offset", 0) or 0)
            for s in inner_stmts
        ):
            return None
        matched = [
            s
            for s in inner_stmts
            if getattr(s, "lineno", 0) == u_start
            and getattr(s, "col_offset", 0) >= start_col
        ]
        if len(matched) == 1:
            target_stmt = matched[0]
        else:
            return None
    else:
        return None

    if start_col is not None and (
        getattr(target_stmt, "lineno", 0) != u_start
        or start_col != getattr(target_stmt, "col_offset", 0)
        or isinstance(target_stmt, ast.Expr)
    ):
        return None

    end_col = getattr(target_stmt, "end_col_offset", None)
    if end_col is None:
        logger.debug(
            "Statement ending at line %d has no end_col_offset; falling back to whole-line",
            u_end,
        )
        return None
    if (u_start < u_end or start_col is None) and any(
        s is not target_stmt and getattr(s, "col_offset", 0) >= int(end_col)
        for s in _get_scope_stmts_by_start_lineno(scope_node).get(u_end, ())
    ):
        return None
    return int(end_col)


def clear_dataflow_cache() -> None:
    """Clears all dataflow AST, parent-map, and downstream read memoization caches.

    Memory Lifecycle Recommendation for Long-Running Processes:
        To support concurrent patch generation and prevent cache thrashing across
        multi-pair sessions, `generate_refactoring_patch` leaves memoization intact
        across runs. Callers in batch CLI runners, IDE language server sidecars,
        daemon file watchers, or interactive test sessions should invoke
        `clear_dataflow_cache()` explicitly at batch boundaries or periodically
        between scan runs. This explicitly purges cached module ASTs, parent node
        maps, and downstream liveness tables, preventing unbounded memory retention
        across long sessions.
    """
    with _downstream_cache_lock:
        _downstream_reads_cache.clear()
        _parsed_tree_cache.clear()
        _scope_parent_maps.clear()
        _scope_stmts_by_end.clear()
        _scope_stmts_by_start.clear()
        _scope_closures_cache.clear()
        _scope_loops_cache.clear()
        _scope_try_with_cache.clear()
        _warned_lenient_closure[0] = False
    clear_closure_warning_cache()


_clear_downstream_reads_cache = clear_dataflow_cache


class _DownstreamCacheKey(NamedTuple):
    """Immutable cache key for downstream AST read analysis."""

    content_digest: str
    start: int
    end: int
    start_col: Optional[int]
    end_col: Optional[int]
    candidates: Optional[frozenset[str]]
    skip_pre_unit_closures: bool
    unit_inputs: Optional[frozenset[str]] = None


def _coerce_string_frozenset(raw_items: Any) -> Optional[frozenset[str]]:
    """Safely coerces strings and ints from arbitrary raw inputs into a frozenset."""
    if raw_items is None:
        return None
    iterable = (
        [raw_items]
        if isinstance(raw_items, (str, bytes, int))
        else raw_items
    )
    try:
        return frozenset(
            str(x) for x in iterable if isinstance(x, (str, int))
        )
    except (TypeError, ValueError):
        return None


def _build_downstream_cache_key(
    source_text: str,
    unit: Dict[str, Any],
    u_start: int,
    u_end: int,
    candidates: Optional[Set[str]],
    skip_pre_unit_closures: bool,
    source_digest: Optional[str] = None,
    resolved_end_col: Optional[int] = None,
    unit_inputs: Optional[Set[str]] = None,
) -> _DownstreamCacheKey:
    """Forms a persistent cache key for downstream AST read analysis.

    Args:
        source_text: Full module source code string.
        unit: AST unit dictionary.
        u_start: 1-indexed unit start line.
        u_end: 1-indexed unit end line.
        candidates: Candidate output variable names to inspect.
        skip_pre_unit_closures: Whether pre-unit closure read capture is bypassed.
        source_digest: Optional hex SHA-256 digest of source_text.
        resolved_end_col: Pre-resolved end column offset if available.
        unit_inputs: Precomputed live unit inputs.
    """
    if source_digest is not None:
        content_digest = str(source_digest)
    else:
        content_digest = hashlib.sha256(
            source_text.encode("utf-8", errors="replace")
        ).hexdigest()

    start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
    end_col = (
        resolved_end_col
        if resolved_end_col is not None
        else _extract_unit_end_col(unit)
    )
    cands_key = _coerce_string_frozenset(candidates)

    unit_inputs_val = unit_inputs
    if unit_inputs_val is None and "inputs" in unit and unit["inputs"] is not None:
        unit_inputs_val = unit["inputs"]
    unit_inputs_key = _coerce_string_frozenset(unit_inputs_val)

    return _DownstreamCacheKey(
        content_digest=content_digest,
        start=u_start,
        end=u_end,
        start_col=start_col,
        end_col=end_col,
        candidates=cands_key,
        skip_pre_unit_closures=bool(skip_pre_unit_closures),
        unit_inputs=unit_inputs_key,
    )
