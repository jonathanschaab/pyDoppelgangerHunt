"""Dataflow cache structures, LRU memoization, and AST coordinate caching."""

from __future__ import annotations

import ast
from collections import OrderedDict, deque
from collections.abc import Mapping as _AbcMapping
import contextlib
import hashlib
import logging
import math
import sys
import threading
from types import MappingProxyType
import warnings
import weakref
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
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

from pydoppelgangerhunt.config import clear_closure_warning_cache
from pydoppelgangerhunt.source_lines import (
    parse_unit_coord,
    split_source_lines,
)

logger = logging.getLogger(__name__)

_CacheValT = TypeVar("_CacheValT")

_downstream_cache_lock: threading.RLock = threading.RLock()
_syntax_warning_lock: threading.RLock = threading.RLock()
_RLOCK_TYPE: type = type(_downstream_cache_lock)
_syntax_warning_active_count: List[int] = [0]
_syntax_warning_tls: threading.local = threading.local()


class _MissingCachedAstNodesError(TypeError):
    """Raised when a custom cached value type does not implement cached_ast_nodes()."""


def _category_unconditionally_ignored(target_cat: Type[Warning]) -> bool:
    """Returns True when active warnings.filters unconditionally ignores target_cat ahead of non-ignore rules."""
    for entry in warnings.filters:
        if len(entry) < 5:
            continue
        action, msg, cat, mod, ln = entry[:5]
        if not isinstance(cat, type) or not issubclass(target_cat, cat):
            continue
        if action in ("error", "always", "default", "module", "once"):
            return False
        if action == "ignore" and msg is None and mod is None and ln == 0:
            return True
    return False


def _filters_already_ignore_syntax_warnings() -> bool:
    """Returns True when active warnings.filters unconditionally ignores SyntaxWarning (and <3.12 DeprecationWarning)."""
    if not _category_unconditionally_ignored(SyntaxWarning):
        return False
    if sys.version_info < (3, 12) and not _category_unconditionally_ignored(
        DeprecationWarning
    ):
        return False
    return True


@contextlib.contextmanager
def _catch_and_ignore_syntax_warnings() -> Iterator[None]:
    """Enters catch_warnings() configured to ignore SyntaxWarning and invalid-escape DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*invalid escape sequence.*",
            category=DeprecationWarning,
        )
        yield


@contextlib.contextmanager
def _suppress_syntax_warnings() -> Iterator[None]:
    """Suppresses SyntaxWarning and invalid-escape DeprecationWarning during parsing or compilation.

    When Python 3.14+ context-aware warnings are enabled (``sys.flags.context_aware_warnings``,
    via ``-X context_aware_warnings=1`` or ``PYTHON_CONTEXT_AWARE_WARNINGS=1``),
    ``warnings.catch_warnings()`` is backed by a ``ContextVar`` and runs completely
    lock-free without mutating process-global ``warnings.filters`` (recommended for
    multi-threaded IDE and language-server hosts). When ``warnings.filters`` already
    has an unconditional top-level ignore rule for ``SyntaxWarning`` (and
    ``DeprecationWarning`` on Python < 3.12), filter mutation
    and locking are also bypassed. Otherwise, ``_syntax_warning_lock`` serializes the
    brief ``ast.parse`` / ``symtable.symtable`` / ``compile`` call across cooperating
    threads.
    """
    if getattr(_syntax_warning_tls, "depth", 0) == 0:
        if bool(getattr(sys.flags, "context_aware_warnings", 0)):
            with _catch_and_ignore_syntax_warnings():
                yield
            return
        if _filters_already_ignore_syntax_warnings():
            yield
            return
    with _syntax_warning_lock:
        prev_depth = int(getattr(_syntax_warning_tls, "depth", 0))
        _syntax_warning_tls.depth = prev_depth + 1
        try:
            if _syntax_warning_active_count[0] > 0:
                _syntax_warning_active_count[0] += 1
                try:
                    yield
                finally:
                    _syntax_warning_active_count[0] -= 1
                return
            _syntax_warning_active_count[0] = 1
            try:
                with _catch_and_ignore_syntax_warnings():
                    yield
            finally:
                _syntax_warning_active_count[0] = 0
        finally:
            _syntax_warning_tls.depth = prev_depth


_MAX_DOWNSTREAM_CACHE_SIZE: int = 1024
_MAX_SCOPE_CACHE_SIZE: int = 2048
_MAX_PARENT_MAP_CACHE_ENTRIES: int = 256
_MAX_PARENT_MAP_TOTAL_NODES: int = 262144
_MAX_OVERSIZED_PARENT_MAP_ROOTS: int = 16
_MAX_EPHEMERAL_FALLBACK_ASTS: int = 256
_MAX_PARSED_TREE_CACHE_SIZE: int = 128
_MAX_STMT_DELTAS_PER_FN: int = 512
_MAX_STMT_WITH_SUMMARY_CACHE_SIZE: int = 512
_SCOPE_CACHE_KIND_CAPS: Dict[str, int] = {
    "span_line_index": 128,
    "tree_enclosing_index": 128,
    "fn_scope_ctx": 512,
    "stmt_deltas": 512,
    "enclosing_fn_cells": 512,
    "suppress_seed": 256,
    "tree_sha256_digest": 256,
    "tree_scope_intervals": 256,
    "tree_expr_scope_intervals": 256,
    "tree_source_lines": 256,
}


class _WeakParentMap(_AbcMapping[ast.AST, ast.AST]):
    """Read-only child->parent mapping holding weak references to both child keys and parent values."""

    __slots__ = ("_refs",)

    def __init__(self, parent_map: Mapping[ast.AST, ast.AST]) -> None:
        refs: weakref.WeakKeyDictionary[
            ast.AST, weakref.ReferenceType[ast.AST]
        ] = weakref.WeakKeyDictionary()
        for child, parent in parent_map.items():
            try:
                refs[child] = weakref.ref(parent)
            except TypeError:
                continue
        self._refs = refs

    def __getitem__(self, key: ast.AST) -> ast.AST:
        try:
            ref = self._refs[key]
        except TypeError as exc:
            raise KeyError(key) from exc
        parent = ref()
        if parent is None:
            raise KeyError(key)
        return parent

    def __iter__(self) -> Iterator[ast.AST]:
        for child, ref in list(self._refs.items()):
            if ref() is not None:
                yield child

    def __len__(self) -> int:
        return len(self._refs)


class _OversizedParentEntry(NamedTuple):
    """Compact metadata retained for AST roots exceeding _MAX_PARENT_MAP_TOTAL_NODES."""

    fingerprint: Tuple[Any, ...]
    scope_ancestors: Dict[ast.AST, ast.AST]
    scope_children: Dict[ast.AST, Tuple[ast.AST, ...]]
    attached_nodes: weakref.WeakSet[ast.AST]
    has_non_weakref: bool
    weak_parent_map: Mapping[ast.AST, ast.AST]


_parent_map_total_nodes: List[int] = [0]
_oversized_scope_parent_maps: OrderedDict[ast.AST, _OversizedParentEntry] = (
    OrderedDict()
)
_scope_children_maps: Dict[ast.AST, Dict[ast.AST, Tuple[ast.AST, ...]]] = {}
_downstream_reads_cache: OrderedDict[Tuple[Any, ...], frozenset[str]] = OrderedDict()
_parsed_tree_cache: OrderedDict[Tuple[str, int], Optional[ast.AST]] = OrderedDict()
_cached_scope_fingerprints: OrderedDict[ast.AST, Tuple[Any, ...]] = OrderedDict()
_debug_ast_immutability: List[bool] = [False]

_scope_loops_cache: OrderedDict[ast.AST, List[Tuple[int, int, int]]] = OrderedDict()
_scope_try_with_cache: OrderedDict[ast.AST, List[ast.AST]] = OrderedDict()
_stmt_with_summary_cache: OrderedDict[ast.stmt, Tuple[bool, frozenset[str]]] = (
    OrderedDict()
)
_tree_raw_source_cache: weakref.WeakKeyDictionary[
    ast.AST, Tuple[int, int, str]
] = weakref.WeakKeyDictionary()
_ephemeral_unit_asts: weakref.WeakSet[ast.AST] = weakref.WeakSet()
_ephemeral_unit_ast_fallback: OrderedDict[int, ast.AST] = OrderedDict()
_active_ephemeral_unit_asts: Dict[int, Tuple[ast.AST, int]] = {}
_warned_lenient_closure: List[bool] = [False]


def _mark_ephemeral_unit_ast(node: ast.AST) -> ast.AST:
    """Marks an ephemeral synthetic AST wrapper so node-keyed caches skip it without setting AST attributes."""
    with _downstream_cache_lock:
        try:
            _ephemeral_unit_asts.add(node)
        except TypeError:
            node_id = id(node)
            _ephemeral_unit_ast_fallback[node_id] = node
            _ephemeral_unit_ast_fallback.move_to_end(node_id)
            while len(_ephemeral_unit_ast_fallback) > _MAX_EPHEMERAL_FALLBACK_ASTS:
                _ephemeral_unit_ast_fallback.popitem(last=False)
    return node


@contextlib.contextmanager
def _ephemeral_unit_ast_context(node: ast.AST) -> Iterator[ast.AST]:
    """Pins a non-weakrefable ephemeral synthetic AST in an active refcounted table for the duration of a call."""
    pinned_id: Optional[int] = None
    with _downstream_cache_lock:
        try:
            _ephemeral_unit_asts.add(node)
        except TypeError:
            pinned_id = id(node)
            prev = _active_ephemeral_unit_asts.get(pinned_id)
            prev_cnt = prev[1] if prev is not None and prev[0] is node else 0
            _active_ephemeral_unit_asts[pinned_id] = (node, prev_cnt + 1)
    try:
        yield node
    finally:
        if pinned_id is not None:
            with _downstream_cache_lock:
                cur = _active_ephemeral_unit_asts.get(pinned_id)
                if cur is not None and cur[0] is node:
                    if cur[1] <= 1:
                        _active_ephemeral_unit_asts.pop(pinned_id, None)
                    else:
                        _active_ephemeral_unit_asts[pinned_id] = (node, cur[1] - 1)


def _is_ephemeral_unit_ast(node: Optional[ast.AST]) -> bool:
    """Returns True when node was registered as an ephemeral synthetic unit AST wrapper."""
    if node is None or not isinstance(node, ast.AST):
        return False
    with _downstream_cache_lock:
        try:
            if node in _ephemeral_unit_asts:
                return True
        except TypeError:
            pass
        node_id = id(node)
        active_entry = _active_ephemeral_unit_asts.get(node_id)
        if active_entry is not None and active_entry[0] is node:
            return True
        return _ephemeral_unit_ast_fallback.get(node_id) is node


_SCOPE_ROOT_TYPES: Tuple[Type[ast.AST], ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)
_INDEXED_PARENT_MAP_TYPES: Tuple[Type[ast.AST], ...] = (
    *_SCOPE_ROOT_TYPES,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_SINGLETON_AST_TYPES: Tuple[Type[ast.AST], ...] = (
    ast.expr_context,
    ast.boolop,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
)


_PERMITTED_PARAM_SCALAR_TYPES: Tuple[type, ...] = (
    str,
    int,
    float,
    bool,
    bytes,
    type(None),
)


def _validate_scope_cache_params(params: Any) -> None:
    """Enforces that _ScopeCacheKey.params recursively contains only AST-free immutable values."""
    if not isinstance(params, tuple):
        raise TypeError(
            f"_ScopeCacheKey.params must be a tuple; got {type(params).__name__}"
        )
    stack: List[Any] = list(params)
    seen: Set[int] = set()
    invalid_mutable_type: Optional[str] = None
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.AST):
            raise ValueError(
                "_ScopeCacheKey.params must not contain ast.AST nodes; use owner or aux_ast_nodes"
            )
        if isinstance(curr, _PERMITTED_PARAM_SCALAR_TYPES):
            continue
        if isinstance(curr, (tuple, list, set, frozenset, dict)):
            if (
                not isinstance(curr, (tuple, frozenset))
                and invalid_mutable_type is None
            ):
                invalid_mutable_type = type(curr).__name__
            curr_id = id(curr)
            if curr_id in seen:
                continue
            seen.add(curr_id)
            children: Sequence[Any] = (
                tuple(part for kv in curr.items() for part in kv)
                if isinstance(curr, dict)
                else tuple(curr)
            )
            for elem in children:
                if isinstance(elem, ast.AST):
                    raise ValueError(
                        "_ScopeCacheKey.params must not contain ast.AST nodes; "
                        "use owner or aux_ast_nodes"
                    )
                if not isinstance(elem, _PERMITTED_PARAM_SCALAR_TYPES):
                    stack.append(elem)
            continue
        if invalid_mutable_type is None:
            invalid_mutable_type = type(curr).__name__
    if invalid_mutable_type is not None:
        raise TypeError(
            "_ScopeCacheKey.params only permits immutable scalars, tuples, and frozensets; "
            f"got {invalid_mutable_type}"
        )


def _canonical_param_type_key(obj: Any) -> Any:
    """Builds a canonical structural and exact-type key for params so 1, True, and 1.0 never collide at any depth."""
    if isinstance(obj, tuple):
        return (tuple, tuple(_canonical_param_type_key(x) for x in obj))
    if isinstance(obj, frozenset):
        return (
            frozenset,
            frozenset(_canonical_param_type_key(x) for x in obj),
        )
    if isinstance(obj, float):
        if math.isnan(obj):
            return (float, "nan")
        if obj == 0.0:
            return (float, 0.0, math.copysign(1.0, obj) < 0.0)
        return (float, obj)
    return (type(obj), obj)


def _params_exact_types_match(a: Any, b: Any) -> bool:
    """Checks that equal parameter structures also match exact scalar types at all nesting depths."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(
            _params_exact_types_match(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, frozenset):
        if len(a) != len(b):
            return False
        return bool(_canonical_param_type_key(a) == _canonical_param_type_key(b))
    if isinstance(a, float):
        return bool(_canonical_param_type_key(a) == _canonical_param_type_key(b))
    return bool(a == b)


def _params_type_hash(obj: Any) -> int:
    """Computes a structural type hash for params so 1, True, and 1.0 do not hash-collide at any depth."""
    return hash(_canonical_param_type_key(obj))


def _validate_scope_cache_key_header(
    owner: Any,
    kind: Any,
    params: Any,
    aux_ast_nodes: Any,
) -> None:
    """Validates owner, kind, params, and aux_ast_nodes for _ScopeCacheKey construction."""
    if not isinstance(owner, ast.AST):
        raise TypeError("_ScopeCacheKey.owner must be an ast.AST node")
    if not isinstance(kind, str):
        raise TypeError("_ScopeCacheKey.kind must be a string")
    if not kind:
        raise ValueError("_ScopeCacheKey.kind must be a non-empty string")
    if params:
        _validate_scope_cache_params(params)
    elif not isinstance(params, tuple):
        raise TypeError(
            f"_ScopeCacheKey.params must be a tuple; got {type(params).__name__}"
        )
    if not isinstance(aux_ast_nodes, tuple) or not all(
        isinstance(n, ast.AST) for n in aux_ast_nodes
    ):
        raise TypeError("_ScopeCacheKey.aux_ast_nodes must be a tuple of ast.AST nodes")


class _ScopeCacheKey:
    """Uniform slotted cache key with explicit AST owner(s) for O(1) owner extraction and indexed tree eviction.

    Cache Key & Value Policy:
        Production entries in ``_scope_closures_cache`` and ``_scope_cell_info_cache`` use
        ``_ScopeCacheKey(owner, kind, params, aux_ast_nodes)``, enabling O(1) owner
        extraction via explicit AST owner fields and reverse-indexed key lookup in
        ``_evict_tree_scopes`` via ``_scope_owner_keys_index`` without scanning every
        entry in ``_scope_closures_cache`` and ``_scope_cell_info_cache``.
        ``params`` must contain only AST-free immutable scalars, tuples, and frozensets;
        any additional AST dependencies must be placed in ``aux_ast_nodes``.
        Cached values must be immutable (frozenset, tuple, NamedTuple, MappingProxyType,
        bool), wrapped in ``_TrackedCachedDict`` (protected by ``_downstream_cache_lock``
        with incremental node registration and removal sync), or returned via defensive
        copies to callers. All custom cached value types must implement ``cached_ast_nodes()``
        (enforced eagerly upon cache insertion via ``_extract_value_ast_nodes``).

    AST Identity & Lifetime Invariant:
        Cache keys and ``_scope_parent_maps`` hold strong references to live ``ast.AST``
        objects until LRU eviction or an explicit ``clear_dataflow_cache()`` /
        ``evict_tree_scopes(tree)`` call. Synthesized ephemeral unit modules must always
        be registered via ``_mark_ephemeral_unit_ast(unit_mod)`` and bypass node-keyed
        caches so detached AST nodes never enter ``_scope_owner_keys_index``.
        ``_compute_scope_fingerprint`` is a bounded ``O(min(|T|, 512) + |body|)``
        diagnostic check in production mode covering all direct ``body``/``orelse``/
        ``finalbody`` statement intervals plus up to 512 BFS nodes (not a full recursive
        proof of deep interior expression immutability); any code path that mutates a
        cached AST in place (including deep interior expressions beyond the 512-node
        prefix) must call ``evict_tree_scopes(tree)`` to invalidate cached entries and
        must not mutate an AST concurrently while another thread is analyzing that same
        AST without synchronization.
    """

    __slots__ = ("owner", "kind", "params", "aux_ast_nodes", "cached_hash")

    owner: ast.AST
    kind: str
    params: Tuple[Any, ...]
    aux_ast_nodes: Tuple[ast.AST, ...]
    cached_hash: Optional[int]

    def __init__(
        self,
        owner: ast.AST,
        kind: str,
        params: Tuple[Any, ...] = (),
        aux_ast_nodes: Tuple[ast.AST, ...] = (),
    ) -> None:
        _validate_scope_cache_key_header(owner, kind, params, aux_ast_nodes)
        self.owner = owner
        self.kind = kind
        self.params = params
        self.aux_ast_nodes = aux_ast_nodes
        self.cached_hash = None

    def __iter__(self) -> Iterator[Any]:
        yield self.owner
        yield self.kind
        yield self.params
        yield self.aux_ast_nodes

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Any:
        return (self.owner, self.kind, self.params, self.aux_ast_nodes)[idx]

    def __repr__(self) -> str:
        return (
            f"_ScopeCacheKey(owner={self.owner!r}, kind={self.kind!r}, "
            f"params={self.params!r}, aux_ast_nodes={self.aux_ast_nodes!r})"
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ScopeCacheKey):
            return False
        if (
            self.owner != other.owner
            or self.kind != other.kind
            or self.aux_ast_nodes != other.aux_ast_nodes
        ):
            return False
        if not self.params and not other.params:
            return True
        return _params_exact_types_match(self.params, other.params)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        cached = self.cached_hash
        if cached is not None:
            return cached
        if not self.params:
            res = hash((self.owner, self.kind, (), self.aux_ast_nodes))
        else:
            res = hash(
                (
                    self.owner,
                    self.kind,
                    _params_type_hash(self.params),
                    self.aux_ast_nodes,
                )
            )
        self.cached_hash = res
        return res


def _make_trusted_scope_cache_key(
    owner: ast.AST,
    kind: str,
    params: Tuple[Any, ...] = (),
    aux_ast_nodes: Tuple[ast.AST, ...] = (),
) -> _ScopeCacheKey:
    """Fast-path constructor for internal callers whose arguments are already trusted."""
    obj = object.__new__(_ScopeCacheKey)
    obj.owner = owner
    obj.kind = kind
    obj.params = params
    obj.aux_ast_nodes = aux_ast_nodes
    obj.cached_hash = None
    return obj


def _extract_key_ast_nodes(key: Any) -> Tuple[ast.AST, ...]:
    """Extracts all ast.AST nodes referenced by a cache key (O(1) for _ScopeCacheKey and ast.AST)."""
    if isinstance(key, _ScopeCacheKey):
        return (key.owner, *key.aux_ast_nodes) if key.aux_ast_nodes else (key.owner,)
    if isinstance(key, ast.AST):
        return (key,)
    if not isinstance(key, (tuple, frozenset)):
        return ()
    found: List[ast.AST] = []
    seen: Set[int] = {id(key)}
    stack: List[Any] = [key]
    while stack:
        curr = stack.pop()
        if isinstance(curr, _ScopeCacheKey):
            found.append(curr.owner)
            found.extend(curr.aux_ast_nodes)
            continue
        for elem in curr:
            if isinstance(elem, ast.AST):
                found.append(elem)
            elif isinstance(elem, (tuple, frozenset)):
                elem_id = id(elem)
                if elem_id not in seen:
                    seen.add(elem_id)
                    stack.append(elem)
    return tuple(found)


def _extract_value_ast_nodes(value: Any) -> Tuple[ast.AST, ...]:
    """Extracts ast.AST nodes held inside a cached value via mandatory cached_ast_nodes() or container traversal."""
    if value is None or isinstance(
        value, (bool, int, str, float, bytes, *_SINGLETON_AST_TYPES)
    ):
        return ()
    if isinstance(value, ast.AST):
        return (value,)
    if type(value) in (frozenset, set) and all(
        isinstance(x, (str, int, bool, float)) for x in value
    ):
        return ()
    found: List[ast.AST] = []
    seen: Set[int] = {id(value)}
    stack: List[Any] = [value]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.AST):
            if not isinstance(curr, _SINGLETON_AST_TYPES):
                found.append(curr)
            continue
        cached_nodes_fn = getattr(curr, "cached_ast_nodes", None)
        if callable(cached_nodes_fn):
            for n in cached_nodes_fn():
                if isinstance(n, ast.AST) and not isinstance(n, _SINGLETON_AST_TYPES):
                    found.append(n)
            continue
        curr_type = type(curr)
        children: Sequence[Any] = ()
        if curr_type in (tuple, list, set, frozenset):
            children = tuple(curr)
        elif curr_type in (dict, _TrackedCachedDict, MappingProxyType):
            children = tuple(part for kv in curr.items() for part in kv)
        else:
            raise _MissingCachedAstNodesError(
                f"Cached value of custom type {curr_type.__name__} must implement cached_ast_nodes()"
            )
        for elem in children:
            if isinstance(elem, ast.AST):
                if not isinstance(elem, _SINGLETON_AST_TYPES):
                    found.append(elem)
            elif not isinstance(
                elem,
                (
                    bool,
                    int,
                    str,
                    float,
                    bytes,
                    type(None),
                    *_SINGLETON_AST_TYPES,
                ),
            ):
                elem_id = id(elem)
                if elem_id not in seen:
                    seen.add(elem_id)
                    stack.append(elem)
    return tuple(found)


_scope_owner_keys_index: Dict[ast.AST, Set[Any]] = {}
_scope_key_nodes_index: Dict[Any, Set[ast.AST]] = {}
_node_to_parent_map_roots: Dict[ast.AST, Set[ast.AST]] = {}
_parent_map_root_nodes: Dict[ast.AST, Set[ast.AST]] = {}


def _index_entry_ast_nodes(
    entry: Any,
    nodes: Sequence[ast.AST],
    forward_index: Dict[Any, Set[ast.AST]],
    reverse_index: Dict[ast.AST, Set[Any]],
) -> None:
    """Registers AST nodes for entry in forward_index and reverse_index."""
    entry_nodes = forward_index.setdefault(entry, set())
    for node in nodes:
        if isinstance(node, ast.AST) and not isinstance(node, _SINGLETON_AST_TYPES):
            entry_nodes.add(node)
            reverse_index.setdefault(node, set()).add(entry)


def _prune_reverse_index_buckets(
    entry: Any,
    forward_index: Dict[Any, Set[ast.AST]],
    reverse_index: Dict[ast.AST, Set[Any]],
) -> None:
    """Removes entry from forward_index and prunes empty reverse_index buckets."""
    for node in forward_index.pop(entry, ()):
        bucket = reverse_index.get(node)
        if bucket is not None:
            bucket.discard(entry)
            if not bucket:
                reverse_index.pop(node, None)


def _build_oversized_scope_parent_map(
    root_node: ast.AST, parent_map: Mapping[ast.AST, ast.AST]
) -> Dict[ast.AST, ast.AST]:
    """Compresses an oversized parent_map into a compact scope-only ancestor map."""
    scope_map: Dict[ast.AST, ast.AST] = {}
    for node in parent_map:
        if isinstance(node, _INDEXED_PARENT_MAP_TYPES) and node is not root_node:
            curr: Optional[ast.AST] = parent_map.get(node)
            while (
                curr is not None
                and curr is not root_node
                and not isinstance(curr, _INDEXED_PARENT_MAP_TYPES)
            ):
                curr = parent_map.get(curr)
            scope_map[node] = curr if curr is not None else root_node
    return scope_map


def _build_children_adjacency(
    parent_map: Mapping[ast.AST, ast.AST],
) -> Dict[ast.AST, Tuple[ast.AST, ...]]:
    """Inverts a child->parent map into a compact parent->children tuple adjacency map."""
    adj_lists: Dict[ast.AST, List[ast.AST]] = {}
    for child, parent in parent_map.items():
        adj_lists.setdefault(parent, []).append(child)
    return {parent: tuple(children) for parent, children in adj_lists.items()}


def _build_attached_node_weakset(
    nodes: Iterable[ast.AST],
) -> Tuple[weakref.WeakSet[ast.AST], bool]:
    """Builds a WeakSet of AST nodes, reporting whether any node lacked weakref support."""
    ws: weakref.WeakSet[ast.AST] = weakref.WeakSet()
    has_non_weakref = False
    for n in nodes:
        try:
            ws.add(n)
        except TypeError:
            has_non_weakref = True
    return ws, has_non_weakref


def _oversized_entry_contains_node(
    root_node: ast.AST, entry: _OversizedParentEntry, node: ast.AST
) -> bool:
    """Checks whether node belongs to oversized root_node without id() reuse false positives."""
    if node is root_node:
        return True
    if isinstance(node, _INDEXED_PARENT_MAP_TYPES):
        return node in entry.scope_ancestors
    try:
        if node in entry.attached_nodes:
            return True
    except TypeError:
        if entry.has_non_weakref:
            return any(n is node for n in ast.walk(root_node))
    return False


def _is_oversized_scope_root(node: Optional[ast.AST]) -> bool:
    """Returns True if node is currently tracked as an oversized parent-map root."""
    if node is None or not isinstance(node, ast.AST):
        return False
    with _downstream_cache_lock:
        return node in _oversized_scope_parent_maps


def _store_parent_map_locked(
    root_node: ast.AST,
    parent_map: Dict[ast.AST, ast.AST],
    precomputed_fp: Optional[Tuple[Any, ...]] = None,
) -> Dict[ast.AST, ast.AST]:
    """Stores parent_map in _scope_parent_maps with LRU cap, total-node budget, and scope reverse indexing."""
    if root_node in _scope_parent_maps:
        if next(reversed(_scope_parent_maps)) != root_node:
            _scope_parent_maps.move_to_end(root_node)
        return _scope_parent_maps[root_node]
    new_map_size = len(parent_map)
    if new_map_size > _MAX_PARENT_MAP_TOTAL_NODES:
        scope_anc = _build_oversized_scope_parent_map(root_node, parent_map)
        attached_ws, has_non_weakref = _build_attached_node_weakset(parent_map)
        weak_pmap: Mapping[ast.AST, ast.AST] = (
            parent_map if has_non_weakref else _WeakParentMap(parent_map)
        )
        _oversized_scope_parent_maps[root_node] = _OversizedParentEntry(
            fingerprint=(
                precomputed_fp
                if precomputed_fp is not None
                else _compute_scope_fingerprint(root_node)
            ),
            scope_ancestors=scope_anc,
            scope_children=_build_children_adjacency(scope_anc),
            attached_nodes=attached_ws,
            has_non_weakref=has_non_weakref,
            weak_parent_map=weak_pmap,
        )
        _oversized_scope_parent_maps.move_to_end(root_node)
        while len(_oversized_scope_parent_maps) > _MAX_OVERSIZED_PARENT_MAP_ROOTS:
            _oversized_scope_parent_maps.popitem(last=False)
        return parent_map
    _oversized_scope_parent_maps.pop(root_node, None)
    while _scope_parent_maps and (
        len(_scope_parent_maps) >= _MAX_PARENT_MAP_CACHE_ENTRIES
        or _parent_map_total_nodes[0] + new_map_size > _MAX_PARENT_MAP_TOTAL_NODES
    ):
        evicted_root, evicted_map = _scope_parent_maps.popitem(last=False)
        _scope_children_maps.pop(evicted_root, None)
        _parent_map_total_nodes[0] = max(
            0, _parent_map_total_nodes[0] - len(evicted_map)
        )
        _prune_reverse_index_buckets(
            evicted_root, _parent_map_root_nodes, _node_to_parent_map_roots
        )
    _scope_parent_maps[root_node] = parent_map
    _scope_parent_maps.move_to_end(root_node)
    _scope_children_maps[root_node] = _build_children_adjacency(parent_map)
    _parent_map_total_nodes[0] += new_map_size
    indexed_nodes: Set[ast.AST] = {
        n
        for n in (*parent_map.keys(), *parent_map.values())
        if isinstance(n, _INDEXED_PARENT_MAP_TYPES)
    }
    if isinstance(root_node, _INDEXED_PARENT_MAP_TYPES):
        indexed_nodes.add(root_node)
    _index_entry_ast_nodes(
        root_node,
        tuple(indexed_nodes),
        _parent_map_root_nodes,
        _node_to_parent_map_roots,
    )
    return parent_map


def _pop_parent_map_locked(root_node: ast.AST) -> None:
    """Removes root_node from _scope_parent_maps and prunes its reverse node-to-root index entries."""
    removed_map = _scope_parent_maps.pop(root_node, None)
    _scope_children_maps.pop(root_node, None)
    if removed_map is not None:
        _parent_map_total_nodes[0] = max(
            0, _parent_map_total_nodes[0] - len(removed_map)
        )
    _oversized_scope_parent_maps.pop(root_node, None)
    _prune_reverse_index_buckets(
        root_node, _parent_map_root_nodes, _node_to_parent_map_roots
    )


def _needs_scope_owner_parent_map_locked(owner: ast.AST) -> bool:
    """Returns True if owner is a cacheable scope root that lacks a parent map under lock."""
    return (
        isinstance(owner, _SCOPE_ROOT_TYPES)
        and not _is_ephemeral_unit_ast(owner)
        and owner not in _scope_parent_maps
        and owner not in _node_to_parent_map_roots
        and owner not in _oversized_scope_parent_maps
    )


def _verify_oversized_fingerprint_locked(
    tree: ast.AST,
    entry: _OversizedParentEntry,
    curr_fp: Tuple[Any, ...],
) -> bool:
    """Verifies precomputed curr_fp matches entry.fingerprint under lock, evicting or raising if mutated."""
    if curr_fp == entry.fingerprint:
        return True
    if _debug_ast_immutability[0]:
        raise RuntimeError(
            "Cached AST node was mutated in-place without calling "
            "evict_tree_scopes(tree)"
        )
    _evict_tree_scopes(tree)
    return False


def _lookup_valid_oversized_entry(
    root_node: ast.AST,
) -> Optional[_OversizedParentEntry]:
    """Computes root_node's fingerprint outside the lock and returns its valid oversized entry."""
    curr_fp = _compute_scope_fingerprint(root_node)
    with _downstream_cache_lock:
        entry = _oversized_scope_parent_maps.get(root_node)
        if entry is None or not _verify_oversized_fingerprint_locked(
            root_node, entry, curr_fp
        ):
            return None
        if next(reversed(_oversized_scope_parent_maps)) != root_node:
            _oversized_scope_parent_maps.move_to_end(root_node)
        return entry


def _tree_contains_scope_node(tree: ast.AST, node: ast.AST) -> bool:
    """Returns True if node is tree or belongs to tree without rebuilding oversized parent maps."""
    if _is_ephemeral_unit_ast(node):
        return False
    if node is tree:
        return True
    if _is_ephemeral_unit_ast(tree):
        return node in _build_parent_map_unlocked(tree)
    oversized_present = False
    with _downstream_cache_lock:
        if tree in _scope_parent_maps:
            if next(reversed(_scope_parent_maps)) != tree:
                _scope_parent_maps.move_to_end(tree)
            return node in _scope_parent_maps[tree]
        oversized_present = tree in _oversized_scope_parent_maps
    if oversized_present:
        entry = _lookup_valid_oversized_entry(tree)
        if entry is not None:
            return _oversized_entry_contains_node(tree, entry, node)
    built_map = _build_parent_map_unlocked(tree)
    built_fp = (
        _compute_scope_fingerprint(tree)
        if len(built_map) > _MAX_PARENT_MAP_TOTAL_NODES
        else None
    )
    with _downstream_cache_lock:
        stored_map = _store_parent_map_locked(tree, built_map, precomputed_fp=built_fp)
    return node in stored_map


def _get_scope_ancestor_map(tree: ast.AST) -> Mapping[ast.AST, ast.AST]:
    """Returns cached full parent map or compact scope-only ancestor map for oversized trees."""
    if _is_ephemeral_unit_ast(tree):
        return _build_parent_map_unlocked(tree)
    oversized_present = False
    with _downstream_cache_lock:
        if tree in _scope_parent_maps:
            _scope_parent_maps.move_to_end(tree)
            return _scope_parent_maps[tree]
        oversized_present = tree in _oversized_scope_parent_maps
    if oversized_present:
        entry = _lookup_valid_oversized_entry(tree)
        if entry is not None:
            return entry.scope_ancestors
    built_map = _build_parent_map_unlocked(tree)
    built_fp = (
        _compute_scope_fingerprint(tree)
        if len(built_map) > _MAX_PARENT_MAP_TOTAL_NODES
        else None
    )
    with _downstream_cache_lock:
        _store_parent_map_locked(tree, built_map, precomputed_fp=built_fp)
        entry = _oversized_scope_parent_maps.get(tree)
        if entry is not None:
            return entry.scope_ancestors
        return _scope_parent_maps.get(tree, built_map)


def _build_parent_map_unlocked(owner: ast.AST) -> Dict[ast.AST, ast.AST]:
    """Walks owner to construct a child-to-parent AST map without holding _downstream_cache_lock."""
    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(owner):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, _SINGLETON_AST_TYPES):
                parent_map[child] = parent
    return parent_map


def _ensure_scope_owner_parent_map_locked(owner: ast.AST) -> None:
    """Ensures a scope root has a parent map recorded in O(1) if already indexed, else builds and caches it."""
    if not _needs_scope_owner_parent_map_locked(owner):
        return
    _store_parent_map_locked(owner, _build_parent_map_unlocked(owner))


def _ensure_scope_owner_parent_map(owner: ast.AST) -> None:
    """Ensures a scope root has a cached parent map, walking the AST outside _downstream_cache_lock."""
    with _downstream_cache_lock:
        if not _needs_scope_owner_parent_map_locked(owner):
            return
    built_map = _build_parent_map_unlocked(owner)
    built_fp = (
        _compute_scope_fingerprint(owner)
        if len(built_map) > _MAX_PARENT_MAP_TOTAL_NODES
        else None
    )
    with _downstream_cache_lock:
        if _needs_scope_owner_parent_map_locked(owner):
            _store_parent_map_locked(owner, built_map, precomputed_fp=built_fp)


def _register_key_ast_nodes(key: Any, nodes: Sequence[ast.AST]) -> None:
    """Incrementally registers AST nodes for key in _scope_owner_keys_index in O(|nodes|)."""
    if nodes:
        _index_entry_ast_nodes(
            key, nodes, _scope_key_nodes_index, _scope_owner_keys_index
        )


def _resync_key_ast_nodes_locked(
    key: Any, extra_caches: Iterable[Mapping[Any, Any]] = ()
) -> None:
    """Rebuilds reverse AST node index entries for key across all active scope caches."""
    _prune_reverse_index_buckets(key, _scope_key_nodes_index, _scope_owner_keys_index)
    active_caches: List[Mapping[Any, Any]] = [
        _scope_closures_cache,
        _scope_cell_info_cache,
    ]
    for ec in extra_caches:
        if all(ec is not c for c in active_caches):
            active_caches.append(ec)
    present_in_any = False
    combined_nodes: List[ast.AST] = []
    for c in active_caches:
        if key in c:
            present_in_any = True
            val = dict.__getitem__(c, key) if isinstance(c, dict) else c[key]
            combined_nodes.extend(_extract_value_ast_nodes(val))
    if present_in_any:
        key_nodes = _extract_key_ast_nodes(key)
        _register_key_ast_nodes(key, (*key_nodes, *combined_nodes))


def _unregister_key_if_absent(key: Any) -> None:
    """Removes or resynchronizes key in _scope_owner_keys_index across active scope caches."""
    if (
        isinstance(key, _ScopeCacheKey)
        and key.kind == "tree_source_lines"
        and key not in _scope_cell_info_cache
    ):
        try:
            _tree_raw_source_cache.pop(key.owner, None)
        except TypeError:
            pass
    if key not in _scope_closures_cache and key not in _scope_cell_info_cache:
        _prune_reverse_index_buckets(
            key, _scope_key_nodes_index, _scope_owner_keys_index
        )
    else:
        _resync_key_ast_nodes_locked(key)


class _TrackedCachedDict(Dict[Any, Any]):
    """Dictionary value stored in _TrackedScopeCache that registers new AST nodes in O(1) on insert."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_owners: Dict[Tuple[int, Any], Tuple[Any, Any]] = {}
        self._node_refcounts: Dict[ast.AST, int] = {}
        for k, v in self.items():
            for n in set(_extract_value_ast_nodes(k)) | set(
                _extract_value_ast_nodes(v)
            ):
                self._node_refcounts[n] = self._node_refcounts.get(n, 0) + 1

    def _ensure_bound_owners(self) -> Dict[Tuple[int, Any], Tuple[Any, Any]]:
        return cast(
            Dict[Tuple[int, Any], Tuple[Any, Any]],
            self.__dict__.setdefault("_bound_owners", {}),
        )

    def _ensure_node_refcounts(self) -> Dict[ast.AST, int]:
        return cast(
            Dict[ast.AST, int],
            self.__dict__.setdefault("_node_refcounts", {}),
        )

    def copy(self) -> _TrackedCachedDict:
        """Returns a shallow copy with an independent empty _bound_owners table."""
        with _downstream_cache_lock:
            return _TrackedCachedDict(self)

    def __copy__(self) -> _TrackedCachedDict:
        return self.copy()

    def bind_owner(self, cache: Any, cache_key: Any) -> None:
        """Registers an owning (_TrackedScopeCache, key) pair for incremental node registration."""
        self._ensure_bound_owners()[(id(cache), cache_key)] = (
            cache,
            cache_key,
        )

    def unbind_owner(self, cache: Any, cache_key: Any) -> None:
        """Unregisters an owning (_TrackedScopeCache, key) pair."""
        self._ensure_bound_owners().pop((id(cache), cache_key), None)

    def _register_inserted_pair(self, k: Any, v: Any) -> None:
        entry_nodes = set(_extract_value_ast_nodes(k)) | set(
            _extract_value_ast_nodes(v)
        )
        if not entry_nodes:
            return
        refcounts = self._ensure_node_refcounts()
        new_nodes = [n for n in entry_nodes if refcounts.get(n, 0) == 0]
        for n in entry_nodes:
            refcounts[n] = refcounts.get(n, 0) + 1
        owners = getattr(self, "_bound_owners", None)
        if not owners or not new_nodes:
            return
        for cache, ck in list(owners.values()):
            if ck in cache:
                _register_key_ast_nodes(ck, new_nodes)

    def _unregister_removed_pair(self, k: Any, prev_v: Any) -> None:
        entry_nodes = set(_extract_value_ast_nodes(k)) | set(
            _extract_value_ast_nodes(prev_v)
        )
        if not entry_nodes:
            return
        refcounts = self._ensure_node_refcounts()
        zeroed_nodes: List[ast.AST] = []
        for n in entry_nodes:
            cnt = refcounts.get(n, 0)
            if cnt <= 1:
                refcounts.pop(n, None)
                zeroed_nodes.append(n)
            else:
                refcounts[n] = cnt - 1
        owners = getattr(self, "_bound_owners", None)
        if not owners or not zeroed_nodes:
            return
        if len(owners) == 1:
            cache, ck = next(iter(owners.values()))
            if ck in cache and (
                ck not in _scope_closures_cache or ck not in _scope_cell_info_cache
            ):
                ck_own_nodes = set(_extract_key_ast_nodes(ck))
                key_bucket = _scope_key_nodes_index.get(ck)
                for n in zeroed_nodes:
                    if n not in ck_own_nodes:
                        if key_bucket is not None:
                            key_bucket.discard(n)
                        owner_bucket = _scope_owner_keys_index.get(n)
                        if owner_bucket is not None:
                            owner_bucket.discard(ck)
                            if not owner_bucket:
                                _scope_owner_keys_index.pop(n, None)
                return
        self._sync_owners_ast_nodes()

    def _sync_owners_ast_nodes(self) -> None:
        owners = getattr(self, "_bound_owners", None)
        if not owners:
            return
        owner_caches = [cache for cache, _ in owners.values()]
        seen_keys: Set[Any] = set()
        for cache, ck in list(owners.values()):
            if ck in cache and ck not in seen_keys:
                seen_keys.add(ck)
                _resync_key_ast_nodes_locked(ck, extra_caches=owner_caches)

    def touch_lru(self, k: Any) -> Any:
        """Promotes an existing key to the MRU end in O(1) without re-indexing AST nodes."""
        with _downstream_cache_lock:
            if self and next(reversed(self)) == k:
                return dict.__getitem__(self, k)
            val = super().pop(k)
            super().__setitem__(k, val)
            return val

    def __setitem__(self, k: Any, v: Any) -> None:
        with _downstream_cache_lock:
            overwriting = k in self
            prev_v = dict.__getitem__(self, k) if overwriting else None
            super().__setitem__(k, v)
            if overwriting:
                self._unregister_removed_pair(k, prev_v)
            self._register_inserted_pair(k, v)

    def __delitem__(self, k: Any) -> None:
        with _downstream_cache_lock:
            prev_v = dict.__getitem__(self, k)
            super().__delitem__(k)
            self._unregister_removed_pair(k, prev_v)

    def setdefault(self, k: Any, default: Any = None) -> Any:
        with _downstream_cache_lock:
            if k in self:
                return self[k]
            super().__setitem__(k, default)
            self._register_inserted_pair(k, default)
            return default

    def pop(self, k: Any, *args: Any) -> Any:
        with _downstream_cache_lock:
            had_key = k in self
            res = super().pop(k, *args)
            if had_key:
                self._unregister_removed_pair(k, res)
            return res

    def popitem(self) -> Tuple[Any, Any]:
        with _downstream_cache_lock:
            item = super().popitem()
            self._unregister_removed_pair(item[0], item[1])
            return item

    def update(self, *args: Any, **kwargs: Any) -> None:
        with _downstream_cache_lock:
            temp = dict(*args, **kwargs)
            for k, v in temp.items():
                self[k] = v

    def __or__(self, other: Any) -> Dict[Any, Any]:
        return dict.__or__(self, other)

    def __ior__(self, other: Any) -> _TrackedCachedDict:
        with _downstream_cache_lock:
            temp = dict(other)
            for k, v in temp.items():
                self[k] = v
            return self

    def clear(self) -> None:
        with _downstream_cache_lock:
            super().clear()
            self._ensure_node_refcounts().clear()
            self._sync_owners_ast_nodes()


class _TrackedScopeCache(OrderedDict[Any, _CacheValT]):
    """OrderedDict cache that incrementally registers AST nodes on insert and prunes on eviction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._kind_keys: Dict[str, OrderedDict[Any, None]] = {}
        super().__init__(*args, **kwargs)

    def ensure_kind_keys(self) -> Dict[str, OrderedDict[Any, None]]:
        """Returns the per-kind LRU key ring dictionary for this cache."""
        return cast(
            Dict[str, OrderedDict[Any, None]],
            self.__dict__.setdefault("_kind_keys", {}),
        )

    def set_entry_locked(
        self,
        key: Any,
        value: _CacheValT,
        *,
        key_nodes: Optional[Tuple[ast.AST, ...]] = None,
        ensure_parent_maps: bool = True,
    ) -> Tuple[_CacheValT, bool]:
        """Inserts or updates key under _downstream_cache_lock; returns (stored_or_raw_value, was_stored)."""
        resolved_key_nodes = (
            key_nodes if key_nodes is not None else _extract_key_ast_nodes(key)
        )
        try:
            val_nodes = _extract_value_ast_nodes(value)
        except _MissingCachedAstNodesError:
            if _debug_ast_immutability[0]:
                raise
            logger.debug(
                "Bypassing scope cache storage for unregistered value type %s "
                "(missing cached_ast_nodes() implementation)",
                type(value).__qualname__,
            )
            return value, False
        if isinstance(value, dict) and not isinstance(value, _TrackedCachedDict):
            value = cast(
                _CacheValT, _TrackedCachedDict(cast(Dict[Any, Any], value))
            )
        was_present = key in self
        if was_present:
            prev_val = dict.__getitem__(self, key)
            if isinstance(prev_val, _TrackedCachedDict) and prev_val is not value:
                prev_val.unbind_owner(self, key)
        if isinstance(value, _TrackedCachedDict):
            value.bind_owner(self, key)
        super().__setitem__(key, value)
        super().move_to_end(key, last=True)
        if isinstance(key, _ScopeCacheKey):
            kind_ring = self.ensure_kind_keys().setdefault(key.kind, OrderedDict())
            kind_ring[key] = None
            kind_ring.move_to_end(key)
        if was_present:
            _resync_key_ast_nodes_locked(key, extra_caches=(self,))
        else:
            _register_key_ast_nodes(key, (*resolved_key_nodes, *val_nodes))
        if ensure_parent_maps:
            for n in resolved_key_nodes:
                _ensure_scope_owner_parent_map_locked(n)
        return value, True

    def __setitem__(self, key: Any, value: _CacheValT) -> None:
        key_nodes = _extract_key_ast_nodes(key)
        for n in key_nodes:
            _ensure_scope_owner_parent_map(n)
        with _downstream_cache_lock:
            self.set_entry_locked(
                key, value, key_nodes=key_nodes, ensure_parent_maps=True
            )

    def move_to_end(self, key: Any, last: bool = True) -> None:
        with _downstream_cache_lock:
            if not last or not self or next(reversed(self)) != key:
                super().move_to_end(key, last=last)
            if isinstance(key, _ScopeCacheKey):
                kind_ring = self.ensure_kind_keys().get(key.kind)
                if kind_ring is not None and key in kind_ring:
                    if not last or next(reversed(kind_ring)) != key:
                        kind_ring.move_to_end(key, last=last)

    def __delitem__(self, key: Any) -> None:
        with _downstream_cache_lock:
            if key in self:
                prev_val = dict.__getitem__(self, key)
                if isinstance(prev_val, _TrackedCachedDict):
                    prev_val.unbind_owner(self, key)
            if isinstance(key, _ScopeCacheKey):
                kind_ring = self.ensure_kind_keys().get(key.kind)
                if kind_ring is not None:
                    kind_ring.pop(key, None)
            super().__delitem__(key)
            _unregister_key_if_absent(key)

    def pop(self, key: Any, *args: Any) -> Any:
        with _downstream_cache_lock:
            in_self = key in self
            if in_self:
                prev_val = dict.__getitem__(self, key)
                if isinstance(prev_val, _TrackedCachedDict):
                    prev_val.unbind_owner(self, key)
                if isinstance(key, _ScopeCacheKey):
                    kind_ring = self.ensure_kind_keys().get(key.kind)
                    if kind_ring is not None:
                        kind_ring.pop(key, None)
            res = super().pop(key, *args)
            if in_self:
                _unregister_key_if_absent(key)
            return res

    def popitem(self, last: bool = True) -> Tuple[Any, _CacheValT]:
        with _downstream_cache_lock:
            k, v = super().popitem(last=last)
            if isinstance(v, _TrackedCachedDict):
                v.unbind_owner(self, k)
            if isinstance(k, _ScopeCacheKey):
                kind_ring = self.ensure_kind_keys().get(k.kind)
                if kind_ring is not None:
                    kind_ring.pop(k, None)
            _unregister_key_if_absent(k)
            return k, v

    def clear(self) -> None:
        with _downstream_cache_lock:
            for k in list(self.keys()):
                prev_val = dict.__getitem__(self, k)
                if isinstance(prev_val, _TrackedCachedDict):
                    prev_val.unbind_owner(self, k)
            self.ensure_kind_keys().clear()
            super().clear()
            if self is _scope_cell_info_cache:
                _tree_raw_source_cache.clear()
            other = (
                _scope_cell_info_cache
                if self is _scope_closures_cache
                else _scope_closures_cache
            )
            if not other:
                _scope_owner_keys_index.clear()
                _scope_key_nodes_index.clear()
            else:
                for k in list(_scope_key_nodes_index.keys()):
                    _unregister_key_if_absent(k)


_scope_closures_cache: _TrackedScopeCache[Any] = _TrackedScopeCache()
_scope_cell_info_cache: _TrackedScopeCache[Any] = _TrackedScopeCache()

_scope_parent_maps: OrderedDict[ast.AST, Dict[ast.AST, ast.AST]] = OrderedDict()
_scope_stmts_by_end: OrderedDict[ast.AST, Dict[int, List[ast.stmt]]] = OrderedDict()
_scope_stmts_by_start: OrderedDict[ast.AST, Dict[int, List[ast.stmt]]] = OrderedDict()


def _key_references_scope_nodes(
    key: Any,
    tree_nodes: Set[ast.AST],
    safe_containers: Optional[Set[int]] = None,
) -> bool:
    """Returns True if a cache key references any AST node in tree_nodes."""
    if isinstance(key, _ScopeCacheKey):
        return key.owner in tree_nodes or any(
            n in tree_nodes for n in key.aux_ast_nodes
        )
    if isinstance(key, ast.AST):
        return key in tree_nodes
    if not isinstance(key, (tuple, frozenset)):
        return False
    if safe_containers is not None and id(key) in safe_containers:
        return False
    ast_nodes = _extract_key_ast_nodes(key)
    if any(n in tree_nodes for n in ast_nodes):
        return True
    if safe_containers is not None and not ast_nodes:
        safe_containers.add(id(key))
    return False


_MAX_PROD_FINGERPRINT_NODES: int = 512
_MAX_DEBUG_FINGERPRINT_NODES: int = 65536


def _node_fingerprint_scalar(curr: ast.AST) -> Any:
    """Extracts hashable scalar AST attributes that can be mutated in place without replacing node identity."""
    if isinstance(curr, ast.Name):
        return (curr.id, type(curr.ctx))
    if isinstance(curr, ast.Constant):
        val = curr.value
        try:
            hash(val)
            return (type(val), val)
        except TypeError:
            return (type(val), repr(val))
    if isinstance(curr, ast.Attribute):
        return (curr.attr, type(curr.ctx))
    if isinstance(curr, ast.arg):
        return curr.arg
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return curr.name
    if isinstance(curr, (ast.Nonlocal, ast.Global)):
        return tuple(curr.names)
    if isinstance(curr, ast.alias):
        return (curr.name, curr.asname)
    if isinstance(curr, ast.ImportFrom):
        return (curr.module, curr.level)
    if isinstance(curr, (ast.BinOp, ast.UnaryOp, ast.AugAssign, ast.BoolOp)):
        return type(curr.op)
    if isinstance(curr, ast.Compare):
        return tuple(type(op) for op in curr.ops)
    return None


def _compute_scope_fingerprint(node: ast.AST) -> Tuple[Any, ...]:
    """Computes a structural AST fingerprint of a scope root outside lock.

    Note:
        Runs in ``O(min(|T|, _MAX_PROD_FINGERPRINT_NODES) + |body|)`` time in production
        mode: walks up to ``_MAX_PROD_FINGERPRINT_NODES`` (512) AST nodes in BFS order and
        hashes every direct statement interval in ``body``/``orelse``/``finalbody`` without
        striding, and walks up to ``_MAX_DEBUG_FINGERPRINT_NODES`` (65,536) nodes when
        ``_debug_ast_immutability[0]`` is True. Deep interior expression mutations beyond
        the 512-node BFS prefix in large modules are outside the production diagnostic
        bound, so callers that mutate a cached AST in place must still call
        ``evict_tree_scopes(tree)`` explicitly.
    """
    max_nodes = (
        _MAX_DEBUG_FINGERPRINT_NODES
        if _debug_ast_immutability[0]
        else _MAX_PROD_FINGERPRINT_NODES
    )
    queue: deque[ast.AST] = deque([node])
    visited = 0
    acc = 0x9E3779B97F4A7C15
    mask = 0xFFFFFFFFFFFFFFFF
    while queue and visited < max_nodes:
        curr = queue.popleft()
        visited += 1
        children = [
            ch
            for ch in ast.iter_child_nodes(curr)
            if not isinstance(ch, _SINGLETON_AST_TYPES)
        ]
        if visited + len(queue) < max_nodes:
            rem = max_nodes - (visited + len(queue))
            queue.extend(children[:rem])
        item_hash = hash(
            (
                type(curr),
                id(curr),
                getattr(curr, "lineno", None),
                getattr(curr, "end_lineno", None),
                getattr(curr, "col_offset", None),
                getattr(curr, "end_col_offset", None),
                len(children),
                _node_fingerprint_scalar(curr),
            )
        )
        acc = ((acc ^ (item_hash & mask)) * 0xBF58476D1CE4E5B9) & mask

    stmt_acc = 0x9E3779B97F4A7C15
    total_stmts = 0
    suite_ids: List[int] = []
    for attr_name in ("body", "orelse", "finalbody"):
        suite = getattr(node, attr_name, None)
        if isinstance(suite, list):
            suite_ids.append(len(suite))
            total_stmts += len(suite)
            for s in suite:
                s_hash = hash(
                    (
                        id(s),
                        type(s),
                        getattr(s, "lineno", None),
                        getattr(s, "end_lineno", None),
                        getattr(s, "col_offset", None),
                        getattr(s, "end_col_offset", None),
                    )
                )
                stmt_acc = ((stmt_acc ^ (s_hash & mask)) * 0x94D049BB133111EB) & mask
        elif suite is not None:
            suite_ids.append(id(suite))
    body_sig: Any = (total_stmts, tuple(suite_ids), stmt_acc)
    return (
        type(node),
        getattr(node, "name", None),
        getattr(node, "lineno", None),
        getattr(node, "end_lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_col_offset", None),
        visited,
        acc,
        body_sig,
    )


def _collect_historical_descendants(
    seeds: Iterable[ast.AST],
    children_map: Mapping[ast.AST, Tuple[ast.AST, ...]],
    tree_nodes: Set[ast.AST],
    scope_nodes: Set[ast.AST],
) -> None:
    """Adds historical descendants from children_map (including already-detached subtrees) in O(|T_old|)."""
    stack: List[ast.AST] = [s for s in seeds if s in children_map]
    seen_stack: Set[ast.AST] = set(stack)
    while stack:
        curr = stack.pop()
        for child in children_map.get(curr, ()):
            if child not in tree_nodes:
                tree_nodes.add(child)
                if isinstance(child, _SCOPE_ROOT_TYPES):
                    scope_nodes.add(child)
            if child in children_map and child not in seen_stack:
                seen_stack.add(child)
                stack.append(child)


def _evict_tree_scopes(tree: Optional[ast.AST]) -> None:
    """Purges scope-cache entries belonging to an evicted AST tree or detached subtree.

    Note:
        For indexed scope roots (``_INDEXED_PARENT_MAP_TYPES``), runs in ``O(|T| + k)`` time
        under ``_downstream_cache_lock`` via ``_node_to_parent_map_roots`` and
        ``_scope_owner_keys_index`` without scanning ``_scope_closures_cache`` or
        ``_scope_cell_info_cache``. For non-indexed statement/expression subtrees, also
        checks up to ``_MAX_PARENT_MAP_CACHE_ENTRIES`` (256) entries in
        ``_scope_parent_maps`` and ``_MAX_OVERSIZED_PARENT_MAP_ROOTS`` (16) entries in
        ``_oversized_scope_parent_maps`` to locate enclosing roots without flushing
        unrelated sibling scopes.
    """
    if tree is None or not isinstance(tree, ast.AST):
        return
    with _downstream_cache_lock:
        tree_nodes: Set[ast.AST] = {
            n for n in ast.walk(tree) if not isinstance(n, _SINGLETON_AST_TYPES)
        }
        initial_tree_nodes: Set[ast.AST] = set(tree_nodes)
        scope_nodes: Set[ast.AST] = {
            node for node in tree_nodes if isinstance(node, _SCOPE_ROOT_TYPES)
        }
        scope_nodes.add(tree)
        worklist: List[ast.AST] = list(scope_nodes)
        while worklist:
            sc_root = worklist.pop()
            indexed_children = _parent_map_root_nodes.get(sc_root)
            if indexed_children:
                for child_node in indexed_children:
                    if child_node not in tree_nodes:
                        tree_nodes.add(child_node)
                        initial_tree_nodes.add(child_node)
                        if (
                            isinstance(child_node, _SCOPE_ROOT_TYPES)
                            and child_node not in scope_nodes
                        ):
                            scope_nodes.add(child_node)
                            worklist.append(child_node)
        candidate_roots: Set[ast.AST] = set()
        if tree in _scope_parent_maps:
            candidate_roots.add(tree)
        for n in scope_nodes:
            roots = _node_to_parent_map_roots.get(n)
            if roots:
                candidate_roots.update(roots)
        if not isinstance(tree, _INDEXED_PARENT_MAP_TYPES):
            for r_node, pmap in _scope_parent_maps.items():
                if r_node not in candidate_roots and (r_node is tree or tree in pmap):
                    candidate_roots.add(r_node)
        for o_root in list(_oversized_scope_parent_maps.keys()):
            o_entry = _oversized_scope_parent_maps.get(o_root)
            if o_entry is None:
                continue
            o_map = o_entry.scope_ancestors
            if (
                o_root in tree_nodes
                or _oversized_entry_contains_node(o_root, o_entry, tree)
                or any(sn in o_map for sn in scope_nodes)
            ):
                _collect_historical_descendants(
                    initial_tree_nodes,
                    o_entry.scope_children,
                    tree_nodes,
                    scope_nodes,
                )
                _oversized_scope_parent_maps.pop(o_root, None)
                scope_nodes.add(o_root)
                candidate_roots.add(o_root)
        keys_to_remove: Set[Any] = set()
        for root_node in candidate_roots:
            keys_to_remove.update(_scope_owner_keys_index.get(root_node, ()))
            children_map = _scope_children_maps.get(root_node)
            if children_map is not None:
                _collect_historical_descendants(
                    initial_tree_nodes,
                    children_map,
                    tree_nodes,
                    scope_nodes,
                )
            pmap_opt = _scope_parent_maps.get(root_node)
            if pmap_opt is None:
                continue
            matched_child: Optional[ast.AST] = None
            if tree in pmap_opt:
                matched_child = tree
            elif root_node in tree_nodes:
                matched_child = root_node
            else:
                for n in initial_tree_nodes:
                    if n in pmap_opt:
                        matched_child = n
                        break
            if matched_child is None:
                continue
            scope_nodes.add(root_node)
            has_inner_scope_anc = isinstance(
                matched_child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                ),
            )
            curr_anc: Optional[ast.AST] = pmap_opt.get(matched_child)
            while curr_anc is not None:
                if curr_anc is not root_node or not isinstance(root_node, ast.Module):
                    tree_nodes.add(curr_anc)
                if isinstance(curr_anc, _SCOPE_ROOT_TYPES):
                    scope_nodes.add(curr_anc)
                    if curr_anc is not root_node:
                        has_inner_scope_anc = True
                curr_anc = pmap_opt.get(curr_anc)
            if not has_inner_scope_anc:
                tree_nodes.add(root_node)
        for node in tree_nodes:
            node_keys = _scope_owner_keys_index.get(node)
            if node_keys:
                keys_to_remove.update(node_keys)
        for k in keys_to_remove:
            for knode in _extract_key_ast_nodes(k):
                if isinstance(knode, _SCOPE_ROOT_TYPES) and knode in tree_nodes:
                    scope_nodes.add(knode)
            _scope_closures_cache.pop(k, None)
            _scope_cell_info_cache.pop(k, None)
            _unregister_key_if_absent(k)
        for node in tree_nodes:
            _scope_owner_keys_index.pop(node, None)
            _cached_scope_fingerprints.pop(node, None)
            if isinstance(node, ast.stmt):
                _stmt_with_summary_cache.pop(node, None)
        for node in scope_nodes:
            _pop_parent_map_locked(node)
            _cached_scope_fingerprints.pop(node, None)
            try:
                _tree_raw_source_cache.pop(node, None)
            except TypeError:
                pass
            _scope_stmts_by_end.pop(node, None)
            _scope_stmts_by_start.pop(node, None)
            _scope_loops_cache.pop(node, None)
            _scope_try_with_cache.pop(node, None)


evict_tree_scopes = _evict_tree_scopes


def _ensure_cache_key_parent_maps(
    cache: OrderedDict[Any, Any],
    key: Any,
    ensure_fn: Callable[[ast.AST], Any],
) -> None:
    """Ensures parent maps are registered for AST scope roots referenced by ``cache`` and ``key``."""
    if (cache is _scope_stmts_by_end or cache is _scope_stmts_by_start) and isinstance(
        key, ast.AST
    ):
        ensure_fn(key)
    elif (
        cache is _scope_closures_cache or cache is _scope_cell_info_cache
    ) and isinstance(key, _ScopeCacheKey):
        for aux_node in key.aux_ast_nodes:
            if isinstance(aux_node, _SCOPE_ROOT_TYPES):
                ensure_fn(aux_node)
        if isinstance(key.owner, _SCOPE_ROOT_TYPES):
            ensure_fn(key.owner)


def _store_lru_cache_entry(
    cache: OrderedDict[Any, _CacheValT],
    key: Any,
    value: _CacheValT,
    max_size: int,
) -> _CacheValT:
    """Stores key-value pair in an OrderedDict LRU cache with per-kind caps and eviction outside lock.

    Design Notes:
        - ``_scope_parent_maps`` is routed through ``_store_parent_map_locked`` so its
            reverse node-to-root indexes (``_parent_map_root_nodes`` and
            ``_node_to_parent_map_roots``) remain synchronized on insertion and eviction.
            Any new cache requiring reverse AST node indexing should either use
            ``_TrackedScopeCache`` or register a dedicated reverse-index handler here.
        - For ``_TrackedScopeCache``, per-kind ring caps (``_SCOPE_CACHE_KIND_CAPS``) evict
            the oldest key of that specific kind before global ``_MAX_SCOPE_CACHE_SIZE``
            eviction runs. Under mixed-kind workloads where one kind ring saturates early,
            dropping the oldest key of that kind before global LRU eviction can produce
            slightly non-strict global LRU ordering while preventing any single high-volume
            kind from starving other cache kinds.
        - ``_downstream_cache_lock`` is a reentrant ``RLock`` so nested cache lookups while
            holding the lock are safe; callers that release the lock across ``compute_fn()``
            and re-acquire here tolerate intermediate insertions or evictions cleanly.
    """
    _ensure_cache_key_parent_maps(cache, key, _ensure_scope_owner_parent_map)
    evicted_trees: List[ast.AST] = []
    with _downstream_cache_lock:
        if (
            cache is _scope_parent_maps
            and isinstance(key, ast.AST)
            and isinstance(value, dict)
        ):
            return cast(
                _CacheValT,
                _store_parent_map_locked(
                    key, value, precomputed_fp=_cached_scope_fingerprints.get(key)
                ),
            )
        _ensure_cache_key_parent_maps(cache, key, _ensure_scope_owner_parent_map_locked)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        if isinstance(cache, _TrackedScopeCache):
            if isinstance(key, _ScopeCacheKey):
                kind_cap = _SCOPE_CACHE_KIND_CAPS.get(key.kind)
                if kind_cap is not None and max_size > kind_cap:
                    kind_ring = cache.ensure_kind_keys().get(key.kind)
                    while kind_ring and len(kind_ring) >= kind_cap:
                        oldest_k, _ = kind_ring.popitem(last=False)
                        cache.pop(oldest_k, None)
            while len(cache) >= max_size and cache:
                cache.popitem(last=False)
            stored_value, was_stored = cache.set_entry_locked(
                key, value, ensure_parent_maps=False
            )
            return cast(_CacheValT, cache[key] if was_stored else stored_value)
        while len(cache) >= max_size and cache:
            _, evicted_val = cache.popitem(last=False)
            if cache is _parsed_tree_cache and isinstance(evicted_val, ast.AST):
                evicted_trees.append(evicted_val)
        cache[key] = value
        cache.move_to_end(key)
        stored_value = cache[key]
    for evicted_tree in evicted_trees:
        _evict_tree_scopes(evicted_tree)
    return stored_value


def _move_to_end_if_needed(cache: OrderedDict[Any, Any], key: Any) -> None:
    """Moves key to MRU tail unless it is already the most recently used entry."""
    if isinstance(cache, _TrackedScopeCache):
        cache.move_to_end(key)
    elif cache and next(reversed(cache)) != key:
        cache.move_to_end(key)


def _get_or_compute_scope_cached(
    cache: OrderedDict[Any, _CacheValT],
    scope_node: Any,
    compute_fn: Callable[[], _CacheValT],
) -> _CacheValT:
    """Retrieves cached scope metadata or computes, caches, and returns it thread-safely."""
    try:
        hash(scope_node)
    except TypeError:
        return compute_fn()
    owner_ast: Optional[ast.AST] = (
        scope_node.owner
        if isinstance(scope_node, _ScopeCacheKey)
        else (scope_node if isinstance(scope_node, ast.AST) else None)
    )
    curr_fp: Optional[Tuple[Any, ...]] = (
        _compute_scope_fingerprint(owner_ast)
        if isinstance(owner_ast, ast.AST)
        else None
    )
    with _downstream_cache_lock:
        if isinstance(owner_ast, ast.AST) and curr_fp is not None:
            prev_fp = _cached_scope_fingerprints.get(owner_ast)
            if prev_fp is not None and prev_fp != curr_fp:
                if _debug_ast_immutability[0]:
                    raise RuntimeError(
                        "Cached AST node was mutated in-place without calling "
                        "evict_tree_scopes(tree)"
                    )
                _evict_tree_scopes(owner_ast)
                prev_fp = None
            if prev_fp is None:
                _cached_scope_fingerprints[owner_ast] = curr_fp
                _move_to_end_if_needed(_cached_scope_fingerprints, owner_ast)
                while len(_cached_scope_fingerprints) > _MAX_SCOPE_CACHE_SIZE:
                    _cached_scope_fingerprints.popitem(last=False)
        if scope_node in cache:
            _move_to_end_if_needed(cache, scope_node)
            return cache[scope_node]

    # Concurrency Design Note: compute_fn() executes outside _downstream_cache_lock to
    # eliminate lock contention under multi-threading; transient duplicate computation during
    # concurrent misses is harmless because compute_fn is deterministic and pure.
    computed = compute_fn()
    return _store_lru_cache_entry(cache, scope_node, computed, _MAX_SCOPE_CACHE_SIZE)


def _split_source_lines_stripped(source_text: str) -> Tuple[str, ...]:
    """Splits source_text on LF, CRLF, and CR boundaries (matching CPython's tokenizer) and strips line endings."""
    normalized = (
        source_text.replace("\r\n", "\n").replace("\r", "\n")
        if "\r" in source_text
        else source_text
    )
    return tuple(ln.rstrip("\r\n") for ln in split_source_lines(normalized))


def _safe_get_raw_source_digest(
    tree: ast.AST,
) -> Optional[Tuple[int, int, str]]:
    """Returns cached (text_hash, text_len, digest_str) for tree without failing on non-weakrefable ASTs."""
    try:
        return _tree_raw_source_cache.get(tree)
    except TypeError:
        return None


def _safe_set_raw_source_digest(
    tree: ast.AST, entry: Tuple[int, int, str]
) -> None:
    """Stores (text_hash, text_len, digest_str) in WeakKeyDictionary without pinning tree or raw text."""
    try:
        _tree_raw_source_cache[tree] = entry
    except TypeError:
        pass


def _resolve_raw_text_digest(
    source_text: str, source_digest: Optional[str]
) -> str:
    """Returns caller-provided source_digest or computes hex SHA-256 of source_text."""
    if source_digest is not None:
        return str(source_digest)
    return hashlib.sha256(
        source_text.encode("utf-8", errors="replace")
    ).hexdigest()


def _register_tree_source_lines(
    tree: Optional[ast.AST],
    source_lines: Optional[Sequence[str]] = None,
    *,
    source_text: Optional[str] = None,
    source_digest: Optional[str] = None,
) -> Optional[Tuple[str, ...]]:
    """Associates stripped source_lines with tree in _scope_cell_info_cache and indexes tree's parent map."""
    if (
        tree is None
        or not isinstance(tree, ast.AST)
        or _is_ephemeral_unit_ast(tree)
        or (not source_lines and not source_text)
    ):
        return None
    ck = _make_trusted_scope_cache_key(owner=tree, kind="tree_source_lines")
    text_hash: Optional[int] = hash(source_text) if source_text is not None else None
    text_len: int = len(source_text) if source_text is not None else 0
    text_digest: Optional[str] = (
        str(source_digest) if source_digest is not None else None
    )
    with _downstream_cache_lock:
        if ck in _scope_cell_info_cache:
            existing_lines = cast(Tuple[str, ...], _scope_cell_info_cache[ck])
            if source_lines is not None and existing_lines is source_lines:
                _scope_cell_info_cache.move_to_end(ck)
                return existing_lines
            if source_lines is None and source_text is not None and text_hash is not None:
                cached_raw = _safe_get_raw_source_digest(tree)
                if (
                    cached_raw is not None
                    and cached_raw[0] == text_hash
                    and cached_raw[1] == text_len
                ):
                    if text_digest is None:
                        text_digest = _resolve_raw_text_digest(
                            source_text, source_digest
                        )
                    if cached_raw[2] == text_digest:
                        _scope_cell_info_cache.move_to_end(ck)
                        return existing_lines
    norm_lines = (
        tuple(ln.rstrip("\r\n") for ln in source_lines)
        if source_lines is not None
        else _split_source_lines_stripped(source_text or "")
    )
    if not norm_lines:
        return None
    if source_text is not None and text_hash is not None and text_digest is None:
        text_digest = _resolve_raw_text_digest(source_text, source_digest)
    raw_digest_entry: Optional[Tuple[int, int, str]] = (
        (text_hash, text_len, text_digest)
        if source_text is not None
        and text_hash is not None
        and text_digest is not None
        else None
    )
    matched_existing: Optional[Tuple[str, ...]] = None
    with _downstream_cache_lock:
        if ck in _scope_cell_info_cache:
            existing_lines = cast(Tuple[str, ...], _scope_cell_info_cache[ck])
            if existing_lines == norm_lines:
                _scope_cell_info_cache.move_to_end(ck)
                matched_existing = existing_lines
            else:
                _evict_tree_scopes(tree)
        if raw_digest_entry is not None:
            _safe_set_raw_source_digest(tree, raw_digest_entry)
    if matched_existing is not None:
        return matched_existing
    _ensure_scope_owner_parent_map(tree)
    return cast(
        Tuple[str, ...],
        _store_lru_cache_entry(
            _scope_cell_info_cache, ck, norm_lines, _MAX_SCOPE_CACHE_SIZE
        ),
    )


def _get_cached_tree_source_lines(node: Optional[ast.AST]) -> Optional[Tuple[str, ...]]:
    """Retrieves cached source_lines for node or any of its indexed ancestor scope roots."""
    if node is None or not isinstance(node, ast.AST):
        return None
    with _downstream_cache_lock:
        direct_ck = _make_trusted_scope_cache_key(owner=node, kind="tree_source_lines")
        if direct_ck in _scope_cell_info_cache:
            _scope_cell_info_cache.move_to_end(direct_ck)
            return cast(Tuple[str, ...], _scope_cell_info_cache[direct_ck])
        if isinstance(node, ast.Module):
            return None
        for root in _node_to_parent_map_roots.get(node, ()):
            root_ck = _make_trusted_scope_cache_key(
                owner=root, kind="tree_source_lines"
            )
            if root_ck in _scope_cell_info_cache:
                _scope_cell_info_cache.move_to_end(root_ck)
                return cast(Tuple[str, ...], _scope_cell_info_cache[root_ck])
        src_ring = _scope_cell_info_cache.ensure_kind_keys().get("tree_source_lines")
        if not src_ring:
            return None
        matched_ck: Optional[_ScopeCacheKey] = None
        for root_ck in reversed(src_ring):
            if (
                not isinstance(root_ck, _ScopeCacheKey)
                or root_ck not in _scope_cell_info_cache
            ):
                continue
            root_owner = root_ck.owner
            pmap = _scope_parent_maps.get(root_owner)
            if pmap is not None and node in pmap:
                matched_ck = root_ck
                break
            o_entry = _oversized_scope_parent_maps.get(root_owner)
            if o_entry is not None and _oversized_entry_contains_node(
                root_owner, o_entry, node
            ):
                matched_ck = root_ck
                break
        if matched_ck is not None:
            _scope_cell_info_cache.move_to_end(matched_ck)
            return cast(Tuple[str, ...], _scope_cell_info_cache[matched_ck])
    return None


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
            if cached_tree is not None:
                _register_tree_source_lines(
                    cached_tree, source_text=source_text, source_digest=content_hash
                )
            return cached_tree

    parsed: Optional[ast.AST] = None
    try:
        with _suppress_syntax_warnings():
            parsed = ast.parse(source_text)
    except (
        SyntaxError,
        ValueError,
        UnicodeDecodeError,
        RecursionError,
        MemoryError,
    ):
        parsed = None

    stored_tree = _store_lru_cache_entry(
        _parsed_tree_cache, key, parsed, _MAX_PARSED_TREE_CACHE_SIZE
    )
    if stored_tree is not None:
        _register_tree_source_lines(
            stored_tree, source_text=source_text, source_digest=content_hash
        )
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


def _extract_unit_end_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts end column offset from unit dictionary if present."""
    return _extract_first_unit_coord(unit, ("end_col", "end_col_offset"))


def _get_scope_parent_map(
    scope_node: ast.AST, *, use_cache: bool = True
) -> Mapping[ast.AST, ast.AST]:
    """Retrieves or builds cached parent map for AST nodes within the scope.

    Note:
        The returned mapping is stored in ``_scope_parent_maps`` (or as a weak-referenced
        ``_WeakParentMap`` in ``_oversized_scope_parent_maps`` for oversized roots) and
        wrapped in ``MappingProxyType`` by ``_FnScopeContext``. Callers must treat the
        returned mapping as strictly read-only and never mutate it after construction.
    """

    if not use_cache or _is_ephemeral_unit_ast(scope_node):
        return _build_parent_map_unlocked(scope_node)
    chosen_root: Optional[ast.AST] = None
    oversized_root: Optional[ast.AST] = None
    with _downstream_cache_lock:
        if scope_node not in _scope_parent_maps:
            if scope_node in _oversized_scope_parent_maps:
                oversized_root = scope_node
            else:
                roots = _node_to_parent_map_roots.get(scope_node)
                if roots:
                    for r in roots:
                        if r in _scope_parent_maps:
                            if isinstance(r, ast.Module):
                                chosen_root = r
                                break
                            if chosen_root is None:
                                chosen_root = r
                if chosen_root is None and _oversized_scope_parent_maps:
                    for o_cand, o_ent in reversed(
                        list(_oversized_scope_parent_maps.items())
                    ):
                        if _oversized_entry_contains_node(o_cand, o_ent, scope_node):
                            oversized_root = o_cand
                            break
    if oversized_root is not None:
        o_entry = _lookup_valid_oversized_entry(oversized_root)
        if o_entry is not None:
            return o_entry.weak_parent_map
    if chosen_root is not None:
        curr_fp = _compute_scope_fingerprint(chosen_root)
        with _downstream_cache_lock:
            if chosen_root in _scope_parent_maps:
                prev_fp = _cached_scope_fingerprints.get(chosen_root)
                if prev_fp is not None and prev_fp != curr_fp:
                    if _debug_ast_immutability[0]:
                        raise RuntimeError(
                            "Cached AST node was mutated in-place without calling "
                            "evict_tree_scopes(tree)"
                        )
                    _evict_tree_scopes(chosen_root)
                else:
                    _scope_parent_maps.move_to_end(chosen_root)
                    return _scope_parent_maps[chosen_root]
    built = _get_or_compute_scope_cached(
        _scope_parent_maps, scope_node, lambda: _build_parent_map_unlocked(scope_node)
    )
    with _downstream_cache_lock:
        o_stored = _oversized_scope_parent_maps.get(scope_node)
        if o_stored is not None:
            return o_stored.weak_parent_map
    return built


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
            (getattr(node, "end_lineno", None) or lineno) if use_end_lineno else lineno
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
    if start_col is not None and unit.get("kind") in (
        "comprehension",
        "complex_expr",
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
        isinstance(target_stmt, ast.Expr)
        or not (
            (
                getattr(target_stmt, "lineno", 0) == u_start
                and start_col == getattr(target_stmt, "col_offset", 0)
            )
            or (
                u_start < u_end
                and any(
                    not isinstance(s, ast.Expr)
                    and getattr(s, "lineno", 0) == u_start
                    and getattr(s, "col_offset", 0) == start_col
                    for s in _get_scope_stmts_by_start_lineno(scope_node).get(
                        u_start, []
                    )
                )
            )
        )
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
        _cached_scope_fingerprints.clear()
        _scope_parent_maps.clear()
        _scope_children_maps.clear()
        _oversized_scope_parent_maps.clear()
        _parent_map_total_nodes[0] = 0
        _node_to_parent_map_roots.clear()
        _parent_map_root_nodes.clear()
        _scope_stmts_by_end.clear()
        _scope_stmts_by_start.clear()
        _scope_closures_cache.clear()
        _scope_cell_info_cache.clear()
        _scope_owner_keys_index.clear()
        _scope_key_nodes_index.clear()
        _scope_loops_cache.clear()
        _scope_try_with_cache.clear()
        _stmt_with_summary_cache.clear()
        _tree_raw_source_cache.clear()
        _ephemeral_unit_asts.clear()
        _ephemeral_unit_ast_fallback.clear()
        _active_ephemeral_unit_asts.clear()
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
    is_sub_key = bool(unit.get("is_subroutine")) or unit.get("kind") in (
        "compound_block",
        "sliding_window",
        "clause_branch",
        "comprehension",
        "complex_expr",
        "data_table",
    )
    raw_src_code = unit.get("source_code")
    unit_source_hash: Optional[str] = None
    if (
        (start_col is None or end_col is None)
        and isinstance(raw_src_code, str)
        and raw_src_code.strip()
    ):
        unit_source_hash = hashlib.sha256(
            raw_src_code.strip().encode("utf-8", errors="replace")
        ).hexdigest()[:16]

    return _DownstreamCacheKey(
        content_digest=content_digest,
        start=u_start,
        end=u_end,
        start_col=start_col,
        end_col=end_col,
        candidates=cands_key,
        skip_pre_unit_closures=bool(skip_pre_unit_closures),
        unit_inputs=unit_inputs_key,
        is_subroutine=is_sub_key,
        unit_source_hash=unit_source_hash,
    )
