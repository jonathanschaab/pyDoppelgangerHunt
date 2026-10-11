"""Dataflow cache structures, LRU memoization, and AST coordinate caching."""

from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import logging
import sys
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
_scope_closures_cache: OrderedDict[Any, List[Tuple[Any, Set[str]]]] = OrderedDict()
_scope_loops_cache: OrderedDict[ast.AST, List[Tuple[int, int]]] = OrderedDict()
_scope_try_with_cache: OrderedDict[ast.AST, List[ast.AST]] = OrderedDict()
_warned_lenient_closure: List[bool] = [False]


def _evict_tree_scopes(tree: Optional[ast.AST]) -> None:
    """Purges scope-cache entries belonging to an evicted AST tree."""
    if tree is None:
        return
    tree_nodes = set(ast.walk(tree))
    with _downstream_cache_lock:
        for node in tree_nodes:
            _scope_parent_maps.pop(node, None)
            _scope_stmts_by_end.pop(node, None)
            _scope_closures_cache.pop(node, None)
            _scope_loops_cache.pop(node, None)
            _scope_try_with_cache.pop(node, None)
        closure_keys_to_remove = [
            k
            for k in _scope_closures_cache
            if isinstance(k, tuple)
            and len(k) == 2
            and (k[0] in tree_nodes or k[1] in tree_nodes)
        ]
        for k in closure_keys_to_remove:
            _scope_closures_cache.pop(k, None)


def _store_lru_cache_entry(
    cache: OrderedDict[Any, Any],
    key: Any,
    value: Any,
    max_size: int,
) -> None:
    """Stores key-value pair in an OrderedDict LRU cache with eviction outside lock."""
    evicted_tree: Optional[ast.AST] = None
    with _downstream_cache_lock:
        if key not in cache and len(cache) >= max_size:
            _, evicted_val = cache.popitem(last=False)
            if cache is _parsed_tree_cache and isinstance(evicted_val, ast.AST):
                evicted_tree = evicted_val
        cache[key] = value
        cache.move_to_end(key)
    if evicted_tree is not None:
        _evict_tree_scopes(evicted_tree)


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

    computed = compute_fn()

    try:
        _store_lru_cache_entry(cache, scope_node, computed, _MAX_SCOPE_CACHE_SIZE)
    except TypeError:
        pass

    return computed


def _get_or_parse_tree(
    source_text: str,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Retrieves cached AST tree or parses source_text into an AST tree."""
    if not source_text.strip():
        return None
    content_hash = (
        source_digest
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
    except (SyntaxError, ValueError, UnicodeDecodeError):
        parsed = None

    _store_lru_cache_entry(_parsed_tree_cache, key, parsed, _MAX_PARSED_TREE_CACHE_SIZE)
    return parsed


def _parse_source_tree(
    source_text: str,
    tree: Optional[ast.AST] = None,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Parses Python source code into an AST tree if not already provided."""
    if tree is not None:
        return tree
    return _get_or_parse_tree(source_text, source_digest=source_digest)


def _get_valid_unit_bounds(unit: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Extracts positive start and end line coordinates from a unit dict."""
    try:
        u_start = parse_unit_coord(unit, "start", default=0)
        u_end = parse_unit_coord(unit, "end", default=u_start)
    except (ValueError, TypeError):
        return None
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None
    return u_start, u_end


def _extract_first_unit_coord(
    unit: Dict[str, Any], keys: Sequence[str]
) -> Optional[int]:
    """Returns the first successfully parsed coordinate among candidate keys."""
    for key in keys:
        if key in unit and unit.get(key) is not None:
            try:
                val = parse_unit_coord(unit, key, default=None)
                if val is not None:
                    return val
            except (ValueError, TypeError):
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


def _get_scope_stmts_by_end_lineno(
    scope_node: ast.AST,
) -> Dict[int, List[ast.stmt]]:
    """Retrieves or builds cached map of statement end line numbers to statement nodes."""
    def _compute() -> Dict[int, List[ast.stmt]]:
        m: Dict[int, List[ast.stmt]] = {}
        for node in ast.walk(scope_node):
            if node is scope_node:
                continue
            if isinstance(node, ast.stmt):
                e = getattr(node, "end_lineno", None) or getattr(node, "lineno", 0) or 0
                if e > 0:
                    m.setdefault(e, []).append(node)
        return m

    return _get_or_compute_scope_cached(_scope_stmts_by_end, scope_node, _compute)


def _resolve_unit_ast_end_col(
    scope_node: ast.AST, unit: Dict[str, Any]
) -> Optional[int]:
    """Resolves end column offset from statement boundaries within the enclosing scope."""
    try:
        u_start = parse_unit_coord(unit, "start", default=0)
        u_end = parse_unit_coord(unit, "end", default=u_start)
    except (ValueError, TypeError):
        return None
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None

    parent_map = _get_scope_parent_map(scope_node)
    matching_stmts = list(_get_scope_stmts_by_end_lineno(scope_node).get(u_end, []))

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
    if len(inner_stmts) == 1:
        target_stmt = inner_stmts[0]
    elif u_start == u_end:
        start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
        if start_col is not None:
            matched = [s for s in inner_stmts if getattr(s, "col_offset", 0) >= start_col]
            if len(matched) == 1:
                target_stmt = matched[0]
            else:
                return None
        else:
            return None
    else:
        return None

    end_col = getattr(target_stmt, "end_col_offset", None)
    if end_col is None:
        logger.debug(
            "Statement ending at line %d has no end_col_offset; falling back to whole-line",
            u_end,
        )
        return None
    return int(end_col)


def clear_dataflow_cache() -> None:
    """Clears all dataflow AST, parent-map, and downstream read memoization caches.

    Memory Lifecycle Recommendation for Long-Running Processes:
        In batch CLI workflows (`generate_refactoring_patch`), this cleanup is invoked
        automatically in a `finally` block at the conclusion of patch generation.
        However, long-running processes—such as IDE language server sidecars, daemon
        file watchers, or interactive test sessions—that call dataflow inspection functions
        (`collect_downstream_read_names`, `resolve_clone_pair_outputs`, etc.) directly should
        invoke `clear_dataflow_cache()` periodically between analysis cycles.
        This explicitly purges cached module ASTs, parent node maps, and downstream
        liveness tables, preventing unbounded memory retention across long sessions.
    """
    with _downstream_cache_lock:
        _downstream_reads_cache.clear()
        _parsed_tree_cache.clear()
        _scope_parent_maps.clear()
        _scope_stmts_by_end.clear()
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


def _build_downstream_cache_key(
    source_text: str,
    unit: Dict[str, Any],
    u_start: int,
    u_end: int,
    candidates: Optional[Set[str]],
    skip_pre_unit_closures: bool,
    source_digest: Optional[str] = None,
    resolved_end_col: Optional[int] = None,
    tree: Optional[ast.AST] = None,
    unit_inputs: Optional[Set[str]] = None,
) -> _DownstreamCacheKey:
    """Forms a persistent cache key for downstream AST read analysis.

    Note:
        The `tree` parameter is retained for backwards compatibility. AST nodes
        are intentionally excluded from cache keys to prevent memory pinning
        of parsed module trees and cross-thread lifecycle retention.

    Args:
        source_text: Full module source code string.
        unit: AST unit dictionary.
        u_start: 1-indexed unit start line.
        u_end: 1-indexed unit end line.
        candidates: Candidate output variable names to inspect.
        skip_pre_unit_closures: Whether pre-unit closure read capture is bypassed.
        source_digest: Optional hex SHA-256 digest of source_text.
        resolved_end_col: Pre-resolved end column offset if available.
        tree: Optional pre-parsed module AST (unused; retained for compatibility).
        unit_inputs: Precomputed live unit inputs.
    """
    del tree
    if source_digest is not None:
        content_digest = str(source_digest)
    else:
        sha_module = getattr(
            sys.modules.get("pydoppelgangerhunt.fixer.dataflow"),
            "hashlib",
            hashlib,
        )
        content_digest = sha_module.sha256(
            source_text.encode("utf-8", errors="replace")
        ).hexdigest()

    start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
    end_col = (
        resolved_end_col
        if resolved_end_col is not None
        else _extract_unit_end_col(unit)
    )
    cands_key = frozenset(candidates) if candidates is not None else None
    unit_inputs_val = unit_inputs
    if unit_inputs_val is None and "inputs" in unit and unit["inputs"] is not None:
        unit_inputs_val = set(unit["inputs"])
    unit_inputs_key = frozenset(unit_inputs_val) if unit_inputs_val is not None else None
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
