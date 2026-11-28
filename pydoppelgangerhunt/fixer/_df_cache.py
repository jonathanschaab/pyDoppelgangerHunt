"""Run-scoped AnalysisSession and immutable source-hash caches for dataflow analysis.

# INVARIANT (NO GLOBAL MUTABLE AST CACHES — RUN-SCOPED SESSION):
# Process-global caches in this module (``_parsed_tree_cache`` and
# ``_downstream_reads_cache``) are keyed STRICTLY by immutable SHA-256 digests
# of ``str`` source text and scalar parameters—never by caller-owned mutable
# ``ast.AST`` objects.
#
# All AST-keyed derived structures (parent maps, statement line indexes,
# closure/loop/try-with/cell-info/exit-protocol/with-summary memo tables, and
# registered tree source lines) live strictly inside a run-scoped
# :class:`AnalysisSession` bound to a ``contextvars.ContextVar``.
#
# - Batch entrypoints (``cli.main``, ``generate_refactoring_patch``,
#   ``generate_html_report``, ``synthesize_refactoring_suggestion``,
#   ``synthesize_shared_helper_code``, ``inspect_single_unit_scope``,
#   ``analyze_unit_variable_scope``, ``collect_downstream_read_names``) enter
#   ``with analysis_session():`` so repeated lookups within a run share O(1)
#   cached maps and release all AST references immediately when the run ends.
# - Standalone calls outside an explicit session get a fresh transient session
#   that is discarded on return, so mutating a caller-owned ``ast.AST`` between
#   calls can never observe stale cached state.
#
# Do NOT re-introduce process-global AST-keyed dictionaries, sampled structural
# fingerprinting (``_compute_scope_fingerprint``), global mutation/invalidation
# epochs, ``_WeakParentMap``, ``_TrackedScopeCache``, or ``evict_tree_scopes``.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
import contextlib
import contextvars
from enum import Enum
import functools
import hashlib
import threading
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
)
import weakref

from pydoppelgangerhunt.config import (
    bind_contextvar,
    clear_closure_warning_cache,
)
from pydoppelgangerhunt._ast_compat import safe_parse_ast
from pydoppelgangerhunt.source_lines import (
    clear_notebook_cache,
    is_non_string_sequence,
    parse_unit_coord,
    split_source_lines,
)
from pydoppelgangerhunt.unit_boundary import (
    END_COL_KEYS,
    START_COL_KEYS,
)

__all__ = [
    "AnalysisSession",
    "analysis_session",
    "clear_dataflow_cache",
]

_CacheValT = TypeVar("_CacheValT")
_FuncT = TypeVar("_FuncT", bound=Callable[..., Any])

_MAX_DOWNSTREAM_CACHE_SIZE: int = 1024
_MAX_SCOPE_CACHE_SIZE: int = 2048
_MAX_PARSED_TREE_CACHE_SIZE: int = 128

_NESTED_UNIT_KINDS: frozenset[str] = frozenset(
    {
        "compound_block",
        "sliding_window",
        "clause_branch",
        "comprehension",
        "complex_expr",
        "data_table",
    }
)

_SINGLETON_AST_TYPES: Tuple[Type[ast.AST], ...] = (
    ast.expr_context,
    ast.boolop,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
)


class _SessionBucket(str, Enum):
    """Named memoization buckets scoped strictly to an active AnalysisSession."""

    CELL_INFO = "cell_info"
    CLOSURES = "closures"
    LOOPS = "loops"
    TRY_WITH = "try_with"
    STMTS_BY_START = "stmts_by_start"
    STMTS_BY_END = "stmts_by_end"
    STMT_WITH_SUMMARY = "stmt_with_summary"
    STMT_DELTAS = "stmt_deltas"
    DOWNSTREAM_READS = "downstream_reads"


class _ScopeCacheKey(NamedTuple):
    """Composite cache key for session-scoped AST scope analyses."""

    owner: ast.AST
    kind: str
    params: Tuple[Any, ...] = ()
    aux_ast_nodes: Tuple[ast.AST, ...] = ()


def _make_trusted_scope_cache_key(
    owner: ast.AST,
    kind: str,
    *,
    params: Tuple[Any, ...] = (),
    aux_ast_nodes: Tuple[ast.AST, ...] = (),
) -> _ScopeCacheKey:
    """Constructs a _ScopeCacheKey for internal session memoization."""
    return _ScopeCacheKey(
        owner=owner,
        kind=kind,
        params=params,
        aux_ast_nodes=aux_ast_nodes,
    )


class AnalysisSession:
    """Run-scoped container for AST parent maps, line indexes, and scope caches.

    Every AST reference stored inside an ``AnalysisSession`` is owned strictly
    for the lifetime of that session and garbage-collected on exit. Thread-safe
    when a session context is shared across worker threads via
    ``contextvars.copy_context().run``.
    """

    def __init__(self, *, max_bucket_entries: int = _MAX_SCOPE_CACHE_SIZE) -> None:
        self._lock = threading.RLock()
        self._max_bucket_entries = max(16, int(max_bucket_entries))
        self.parent_maps: OrderedDict[ast.AST, Dict[ast.AST, ast.AST]] = OrderedDict()
        self.node_to_roots: Dict[ast.AST, List[ast.AST]] = {}
        self.buckets: Dict[_SessionBucket, OrderedDict[Any, Any]] = {}
        self.tree_source_lines: OrderedDict[ast.AST, Tuple[str, ...]] = OrderedDict()
        self.ephemeral_ast_ids: Set[int] = set()

    def clear(self) -> None:
        """Clears all session-local AST maps and memoization buckets."""
        with self._lock:
            self.parent_maps.clear()
            self.node_to_roots.clear()
            self.buckets.clear()
            self.tree_source_lines.clear()
            self.ephemeral_ast_ids.clear()

    def get_bucket(self, bucket: _SessionBucket) -> OrderedDict[Any, Any]:
        """Returns the session-local LRU OrderedDict for bucket."""
        with self._lock:
            existing = self.buckets.get(bucket)
            if existing is None:
                existing = OrderedDict()
                self.buckets[bucket] = existing
            return existing

    def get_or_compute_with_hit(
        self,
        bucket: _SessionBucket,
        key: Any,
        compute_fn: Callable[[], _CacheValT],
    ) -> Tuple[_CacheValT, bool]:
        """Looks up or computes a session-local value, returning ``(value, was_cache_hit)``."""
        try:
            hash(key)
        except TypeError:
            return compute_fn(), False
        with self._lock:
            store = self.get_bucket(bucket)
            if key in store:
                store.move_to_end(key)
                return cast(_CacheValT, store[key]), True
        computed = compute_fn()
        with self._lock:
            store = self.get_bucket(bucket)
            if key in store:
                store.move_to_end(key)
                return cast(_CacheValT, store[key]), True
            while len(store) >= self._max_bucket_entries and store:
                store.popitem(last=False)
            store[key] = computed
            return computed, False

    def get_or_compute(
        self,
        bucket: _SessionBucket,
        key: Any,
        compute_fn: Callable[[], _CacheValT],
    ) -> _CacheValT:
        """Looks up or computes a session-local memoized value under bucket."""
        return self.get_or_compute_with_hit(bucket, key, compute_fn)[0]

    def _evict_oldest_parent_map_unlocked(self) -> None:
        """Evicts the least-recently-used parent map and prunes its node_to_roots and source-line links."""
        evicted_root, evicted_map = self.parent_maps.popitem(last=False)
        self.tree_source_lines.pop(evicted_root, None)
        for child in evicted_map:
            roots = self.node_to_roots.get(child)
            if not roots:
                continue
            remaining = [r for r in roots if r is not evicted_root]
            if remaining:
                self.node_to_roots[child] = remaining
            else:
                self.node_to_roots.pop(child, None)

    def get_or_build_parent_map(
        self, scope_node: ast.AST
    ) -> Dict[ast.AST, ast.AST]:
        """Retrieves or builds a child-to-parent AST map within this session."""
        try:
            hash(scope_node)
        except TypeError:
            return _build_parent_map_unlocked(scope_node)
        with self._lock:
            direct = self.parent_maps.get(scope_node)
            if direct is not None:
                self.parent_maps.move_to_end(scope_node)
                return direct
            roots = self.node_to_roots.get(scope_node)
            if roots:
                for root in roots:
                    pmap = self.parent_maps.get(root)
                    if pmap is not None:
                        self.parent_maps.move_to_end(root)
                        return pmap
        built = _build_parent_map_unlocked(scope_node)
        with self._lock:
            existing = self.parent_maps.get(scope_node)
            if existing is not None:
                self.parent_maps.move_to_end(scope_node)
                return existing
            while len(self.parent_maps) >= self._max_bucket_entries and self.parent_maps:
                self._evict_oldest_parent_map_unlocked()
            self.parent_maps[scope_node] = built
            for child in built:
                self.node_to_roots.setdefault(child, []).append(scope_node)
            return built

    def register_tree_source_lines(
        self, tree: ast.AST, norm_lines: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        """Associates stripped source lines with tree in a bounded LRU map for this session."""
        with self._lock:
            if tree in self.tree_source_lines:
                self.tree_source_lines.move_to_end(tree)
            else:
                while (
                    len(self.tree_source_lines) >= self._max_bucket_entries
                    and self.tree_source_lines
                ):
                    self.tree_source_lines.popitem(last=False)
            self.tree_source_lines[tree] = norm_lines
        return norm_lines

    def get_tree_source_lines(
        self, node: ast.AST
    ) -> Optional[Tuple[str, ...]]:
        """Retrieves stripped source lines for node or an indexed enclosing root in this session."""
        with self._lock:
            direct = self.tree_source_lines.get(node)
            if direct is not None:
                self.tree_source_lines.move_to_end(node)
                if node in self.parent_maps:
                    self.parent_maps.move_to_end(node)
                return direct
            for root in self.node_to_roots.get(node, ()):
                found = self.tree_source_lines.get(root)
                if found is not None:
                    self.tree_source_lines.move_to_end(root)
                    if root in self.parent_maps:
                        self.parent_maps.move_to_end(root)
                    return found
            if not isinstance(node, ast.Module) and self.tree_source_lines:
                for root, lines in reversed(list(self.tree_source_lines.items())):
                    pmap = self.parent_maps.get(root)
                    if pmap is not None and node in pmap:
                        self.tree_source_lines.move_to_end(root)
                        self.parent_maps.move_to_end(root)
                        return lines
        return None


_active_analysis_session: contextvars.ContextVar[Optional[AnalysisSession]] = (
    contextvars.ContextVar("active_analysis_session", default=None)
)
_active_ephemeral_ast_ids: contextvars.ContextVar[frozenset[int]] = (
    contextvars.ContextVar("active_ephemeral_ast_ids", default=frozenset())
)


def get_active_analysis_session() -> Optional[AnalysisSession]:
    """Returns the currently active AnalysisSession, or None if outside a session."""
    return _active_analysis_session.get()


@contextlib.contextmanager
def analysis_session(
    session: Optional[AnalysisSession] = None,
) -> Iterator[AnalysisSession]:
    """Context manager establishing or reusing a run-scoped AnalysisSession.

    When called without an explicit *session* while a session is already active
    in the current context, yields the enclosing session without resetting it.
    When an explicit *session* is supplied, binds that session for the block.
    """
    existing = _active_analysis_session.get()
    if session is None and existing is not None:
        yield existing
        return
    bound = session if session is not None else AnalysisSession()
    with bind_contextvar(_active_analysis_session, bound):
        yield bound


def _with_analysis_session(fn: _FuncT) -> _FuncT:
    """Decorator ensuring fn runs inside an AnalysisSession (reusing any outer session)."""

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with analysis_session():
            return fn(*args, **kwargs)

    return cast(_FuncT, _wrapped)


# Process-global immutable source-hash caches (keyed ONLY by SHA-256 digests of `str`
# source text and immutable scalar fields — NEVER by caller-owned mutable AST nodes).
_downstream_cache_lock: threading.RLock = threading.RLock()
_parsed_tree_cache: OrderedDict[
    Tuple[str, int], Tuple[Optional[ast.AST], Tuple[str, ...]]
] = OrderedDict()
_parsed_tree_source_lines: weakref.WeakKeyDictionary[
    ast.AST, Tuple[str, ...]
] = weakref.WeakKeyDictionary()
_downstream_reads_cache: OrderedDict[Tuple[Any, ...], frozenset[str]] = OrderedDict()


def _mark_ephemeral_unit_ast(node: ast.AST) -> ast.AST:
    """Marks an ephemeral synthetic AST wrapper so session caches skip it."""
    sess = _active_analysis_session.get()
    if sess is not None:
        with sess._lock:  # pylint: disable=protected-access
            sess.ephemeral_ast_ids.add(id(node))
    return node


@contextlib.contextmanager
def _ephemeral_unit_ast_context(node: ast.AST) -> Iterator[ast.AST]:
    """Marks node as an ephemeral synthetic unit AST for the duration of the block."""
    prev = _active_ephemeral_ast_ids.get()
    with bind_contextvar(_active_ephemeral_ast_ids, prev | {id(node)}):
        yield node


def _is_ephemeral_unit_ast(node: Optional[ast.AST]) -> bool:
    """Returns True when node is marked as an ephemeral synthetic unit AST wrapper."""
    if node is None or not isinstance(node, ast.AST):
        return False
    node_id = id(node)
    if node_id in _active_ephemeral_ast_ids.get():
        return True
    sess = _active_analysis_session.get()
    if sess is not None:
        with sess._lock:  # pylint: disable=protected-access
            return node_id in sess.ephemeral_ast_ids
    return False


def _store_lru_cache_entry(
    cache: "OrderedDict[Any, _CacheValT]",
    key: Any,
    value: _CacheValT,
    max_size: int,
) -> _CacheValT:
    """Stores key->value in a bounded LRU OrderedDict using double-checked locking."""
    with _downstream_cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        while len(cache) >= max_size and cache:
            cache.popitem(last=False)
        cache[key] = value
        return value


def _key_has_ephemeral_ast(key: Any) -> bool:
    """Returns True when key is or references an ephemeral synthetic unit AST."""
    if isinstance(key, ast.AST):
        return _is_ephemeral_unit_ast(key)
    if isinstance(key, _ScopeCacheKey):
        if _is_ephemeral_unit_ast(key.owner):
            return True
        return any(_is_ephemeral_unit_ast(n) for n in key.aux_ast_nodes)
    return False


def _get_or_compute_scope_cached_with_hit(
    bucket: _SessionBucket,
    key: Any,
    compute_fn: Callable[[], _CacheValT],
) -> Tuple[_CacheValT, bool]:
    """Computes and memoizes a scope value in the active AnalysisSession, returning (value, was_hit)."""
    sess = _active_analysis_session.get()
    if sess is None or _key_has_ephemeral_ast(key):
        return compute_fn(), False
    return sess.get_or_compute_with_hit(bucket, key, compute_fn)


def _get_or_compute_scope_cached(
    bucket: _SessionBucket,
    key: Any,
    compute_fn: Callable[[], _CacheValT],
) -> _CacheValT:
    """Computes and memoizes a scope value strictly within the active AnalysisSession.

    Outside an active ``AnalysisSession`` (or for ephemeral synthetic unit ASTs),
    computes statelessly without storing AST nodes in process-global state.
    """
    return _get_or_compute_scope_cached_with_hit(bucket, key, compute_fn)[0]


def _build_parent_map_unlocked(owner: ast.AST) -> Dict[ast.AST, ast.AST]:
    """Walks owner to construct a child-to-parent AST map excluding CPython singleton nodes."""
    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(owner):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, _SINGLETON_AST_TYPES):
                parent_map[child] = parent
    return parent_map


def _get_scope_parent_map(
    scope_node: ast.AST, *, use_cache: bool = True
) -> Mapping[ast.AST, ast.AST]:
    """Retrieves or builds a child-to-parent AST map within the active AnalysisSession."""
    sess = _active_analysis_session.get()
    if not use_cache or sess is None or _is_ephemeral_unit_ast(scope_node):
        return _build_parent_map_unlocked(scope_node)
    return sess.get_or_build_parent_map(scope_node)


def _get_scope_ancestor_map(tree: ast.AST) -> Mapping[ast.AST, ast.AST]:
    """Returns the child-to-parent map for tree within the active AnalysisSession."""
    return _get_scope_parent_map(tree, use_cache=True)


def _tree_contains_scope_node(tree: ast.AST, node: ast.AST) -> bool:
    """Returns True if node is tree or belongs to tree."""
    if _is_ephemeral_unit_ast(node):
        return False
    if node is tree:
        return True
    return node in _get_scope_parent_map(tree, use_cache=True)


def _split_source_lines_stripped(source_text: str) -> Tuple[str, ...]:
    """Splits source_text on LF, CRLF, and CR boundaries and strips line endings."""
    normalized = (
        source_text.replace("\r\n", "\n").replace("\r", "\n")
        if "\r" in source_text
        else source_text
    )
    return tuple(ln.rstrip("\r\n") for ln in split_source_lines(normalized))


def _register_tree_source_lines(
    tree: Optional[ast.AST],
    source_lines: Optional[Sequence[str]] = None,
    *,
    source_text: Optional[str] = None,
    source_digest: Optional[str] = None,  # pylint: disable=unused-argument
) -> Optional[Tuple[str, ...]]:
    """Associates stripped source_lines with tree inside the active AnalysisSession."""
    if source_lines is not None and not is_non_string_sequence(source_lines):
        raise TypeError(
            "source_lines must be a non-string Sequence of line strings; "
            "pass source_text=... for raw source text"
        )
    if (
        tree is None
        or not isinstance(tree, ast.AST)
        or _is_ephemeral_unit_ast(tree)
        or (not source_lines and not source_text)
    ):
        return None
    norm_lines = (
        tuple(ln.rstrip("\r\n") for ln in source_lines)
        if source_lines is not None
        else _split_source_lines_stripped(source_text or "")
    )
    if not norm_lines:
        return None
    sess = _active_analysis_session.get()
    if sess is not None:
        sess.register_tree_source_lines(tree, norm_lines)
        sess.get_or_build_parent_map(tree)
    return norm_lines


def _get_cached_tree_source_lines(
    node: Optional[ast.AST],
) -> Optional[Tuple[str, ...]]:
    """Retrieves stripped source_lines for node from the active AnalysisSession or parsed-tree table."""
    if node is None or not isinstance(node, ast.AST):
        return None
    sess = _active_analysis_session.get()
    if sess is not None:
        found = sess.get_tree_source_lines(node)
        if found is not None:
            return found
    with _downstream_cache_lock:
        try:
            return _parsed_tree_source_lines.get(node)
        except TypeError:
            return None


def _get_or_parse_tree(
    source_text: str,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Retrieves an immutable source-hash cached AST tree or parses source_text."""
    if not source_text.strip():
        return None
    content_hash = (
        str(source_digest)
        if source_digest is not None
        else hashlib.sha256(
            source_text.encode("utf-8", errors="replace")
        ).hexdigest()
    )
    key = (content_hash, len(source_text))
    with _downstream_cache_lock:
        if key in _parsed_tree_cache:
            _parsed_tree_cache.move_to_end(key)
            cached_tree, cached_lines = _parsed_tree_cache[key]
            if cached_tree is not None and cached_lines:
                sess = _active_analysis_session.get()
                if sess is not None:
                    sess.register_tree_source_lines(cached_tree, cached_lines)
            return cached_tree

    parsed = safe_parse_ast(source_text)
    norm_lines = _split_source_lines_stripped(source_text) if parsed is not None else ()
    stored_tree, stored_lines = _store_lru_cache_entry(
        _parsed_tree_cache,
        key,
        (parsed, norm_lines),
        _MAX_PARSED_TREE_CACHE_SIZE,
    )
    if stored_tree is not None and stored_lines:
        with _downstream_cache_lock:
            try:
                _parsed_tree_source_lines[stored_tree] = stored_lines
            except TypeError:
                pass
        sess = _active_analysis_session.get()
        if sess is not None:
            sess.register_tree_source_lines(stored_tree, stored_lines)
    return stored_tree


def _parse_source_tree(
    source_text: str,
    tree: Optional[ast.AST] = None,
    source_digest: Optional[str] = None,
) -> Optional[ast.AST]:
    """Parses Python source code into an AST tree if not already provided."""
    if tree is not None:
        if source_text and source_text.strip():
            _register_tree_source_lines(
                tree, source_text=source_text, source_digest=source_digest
            )
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


def _extract_unit_start_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts start column offset from unit dictionary if present."""
    return _extract_first_unit_coord(unit, START_COL_KEYS)


def _extract_unit_end_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts end column offset from unit dictionary if present."""
    return _extract_first_unit_coord(unit, END_COL_KEYS)


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
    """Retrieves or builds session-cached map of statement end line numbers to statements."""
    return _get_or_compute_scope_cached(
        _SessionBucket.STMTS_BY_END,
        scope_node,
        lambda: _index_scope_stmts(scope_node, use_end_lineno=True),
    )


def _get_scope_stmts_by_start_lineno(
    scope_node: ast.AST,
) -> Dict[int, List[ast.stmt]]:
    """Retrieves or builds session-cached map of statement start line numbers to statements."""
    return _get_or_compute_scope_cached(
        _SessionBucket.STMTS_BY_START,
        scope_node,
        lambda: _index_scope_stmts(scope_node, use_end_lineno=False),
    )


def clear_dataflow_cache() -> None:
    """Clears the immutable source-hash caches and any active AnalysisSession.

    Long-running processes (such as IDE language servers or daemons) may call
    ``clear_dataflow_cache()`` at batch boundaries to purge memoized parsed ASTs
    and downstream liveness entries. Per-run AST maps live strictly inside
    :class:`AnalysisSession` and are released automatically when each session exits.
    """
    with _downstream_cache_lock:
        _downstream_reads_cache.clear()
        _parsed_tree_cache.clear()
        _parsed_tree_source_lines.clear()
    active_sess = _active_analysis_session.get()
    if active_sess is not None:
        active_sess.clear()
    clear_notebook_cache()
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
    is_lenient: bool
    unit_inputs: Optional[frozenset[str]] = None
    is_subroutine: bool = False
    unit_source_hash: Optional[str] = None


def _coerce_string_frozenset(raw_items: Any) -> Optional[frozenset[str]]:
    """Safely coerces strings and ints from arbitrary raw inputs into a frozenset."""
    if raw_items is None:
        return None
    iterable = [raw_items] if isinstance(raw_items, (str, bytes, int)) else raw_items
    try:
        return frozenset(str(x) for x in iterable if isinstance(x, (str, int)))
    except (TypeError, ValueError):
        return None


def _build_downstream_cache_key(
    source_text: str,
    unit: Dict[str, Any],
    u_start: int,
    u_end: int,
    candidates: Optional[Set[str]],
    is_lenient: bool,
    source_digest: Optional[str] = None,
    resolved_end_col: Optional[int] = None,
    unit_inputs: Optional[Set[str]] = None,
) -> _DownstreamCacheKey:
    """Forms an immutable cache key for downstream AST read analysis."""
    if source_digest is not None:
        content_digest = str(source_digest)
    else:
        content_digest = hashlib.sha256(
            source_text.encode("utf-8", errors="replace")
        ).hexdigest()

    start_col = _extract_unit_start_col(unit)
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
    is_sub_key = bool(unit.get("is_subroutine")) or unit.get("kind") in _NESTED_UNIT_KINDS
    raw_src_code = unit.get("source_code")
    unit_source_hash: Optional[str] = None
    if (
        (start_col is None or end_col is None)
        and isinstance(raw_src_code, str)
        and raw_src_code.strip()
    ):
        unit_source_hash = hashlib.sha256(
            raw_src_code.strip().encode("utf-8", errors="replace")
        ).hexdigest()

    return _DownstreamCacheKey(
        content_digest=content_digest,
        start=u_start,
        end=u_end,
        start_col=start_col,
        end_col=end_col,
        candidates=cands_key,
        is_lenient=bool(is_lenient),
        unit_inputs=unit_inputs_key,
        is_subroutine=is_sub_key,
        unit_source_hash=unit_source_hash,
    )
