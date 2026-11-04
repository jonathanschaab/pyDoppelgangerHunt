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
    Iterator,
    List,
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
_scope_loops_cache: OrderedDict[ast.AST, List[Tuple[int, int, int]]] = OrderedDict()
_scope_try_with_cache: OrderedDict[ast.AST, List[ast.AST]] = OrderedDict()
_warned_lenient_closure: List[bool] = [False]


_SCOPE_ROOT_TYPES: Tuple[Type[ast.AST], ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
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
_INSPECTED_CONTAINER_TYPES: Tuple[type, ...] = (
    tuple,
    frozenset,
    list,
    set,
    dict,
)


def _params_contain_ast(params: Tuple[Any, ...]) -> bool:
    """Checks recursively whether any element or nested container in params is an ast.AST node."""
    stack: List[Any] = list(params)
    seen: Set[int] = set()
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.AST):
            return True
        if isinstance(curr, _INSPECTED_CONTAINER_TYPES):
            curr_id = id(curr)
            if curr_id in seen:
                continue
            seen.add(curr_id)
        if isinstance(curr, dict):
            for k, v in curr.items():
                if isinstance(k, ast.AST) or isinstance(v, ast.AST):
                    return True
                if isinstance(k, _INSPECTED_CONTAINER_TYPES):
                    stack.append(k)
                if isinstance(v, _INSPECTED_CONTAINER_TYPES):
                    stack.append(v)
        elif isinstance(curr, (tuple, frozenset, list, set)):
            for elem in curr:
                if isinstance(elem, ast.AST):
                    return True
                if isinstance(elem, _INSPECTED_CONTAINER_TYPES):
                    stack.append(elem)
    return False


def _validate_scope_cache_params(params: Any) -> None:
    """Enforces that _ScopeCacheKey.params recursively contains only AST-free immutable values."""
    if not isinstance(params, tuple):
        raise TypeError(
            f"_ScopeCacheKey.params must be a tuple; got {type(params).__name__}"
        )
    stack: List[Any] = list(params)
    seen: Set[int] = set()
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.AST):
            raise ValueError(
                "_ScopeCacheKey.params must not contain ast.AST nodes; use owner or aux_ast_nodes"
            )
        if isinstance(curr, _PERMITTED_PARAM_SCALAR_TYPES):
            continue
        if isinstance(curr, (tuple, frozenset)):
            curr_id = id(curr)
            if curr_id in seen:
                continue
            seen.add(curr_id)
            for elem in curr:
                if isinstance(elem, ast.AST):
                    raise ValueError(
                        "_ScopeCacheKey.params must not contain ast.AST nodes; use owner or aux_ast_nodes"
                    )
                if not isinstance(elem, _PERMITTED_PARAM_SCALAR_TYPES):
                    stack.append(elem)
            continue
        if isinstance(curr, (list, set, dict)) and _params_contain_ast((curr,)):
            raise ValueError(
                "_ScopeCacheKey.params must not contain ast.AST nodes; use owner or aux_ast_nodes"
            )
        raise TypeError(
            "_ScopeCacheKey.params only permits immutable scalars, tuples, and frozensets; "
            f"got {type(curr).__name__}"
        )


class _BaseScopeCacheKey(NamedTuple):
    """Base NamedTuple layout for _ScopeCacheKey."""

    owner: ast.AST
    kind: str
    params: Tuple[Any, ...] = ()
    aux_ast_nodes: Tuple[ast.AST, ...] = ()


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
        raise TypeError(
            "_ScopeCacheKey.aux_ast_nodes must be a tuple of ast.AST nodes"
        )


class _ScopeCacheKey(_BaseScopeCacheKey):
    """Uniform cache key with explicit AST owner(s) for O(1) owner extraction and indexed tree eviction."""

    __slots__ = ()

    def __new__(
        cls, owner: ast.AST, kind: str, params: Tuple[Any, ...] = (), aux_ast_nodes: Tuple[ast.AST, ...] = ()
    ) -> _ScopeCacheKey:
        _validate_scope_cache_key_header(owner, kind, params, aux_ast_nodes)
        return super().__new__(cls, owner, kind, params, aux_ast_nodes)


def _make_trusted_scope_cache_key(
    owner: ast.AST, kind: str, params: Tuple[Any, ...] = (), aux_ast_nodes: Tuple[ast.AST, ...] = ()
) -> _ScopeCacheKey:
    """Internal constructor for _ScopeCacheKey that validates header and parameter immutability."""
    _validate_scope_cache_key_header(owner, kind, params, aux_ast_nodes)
    return _BaseScopeCacheKey.__new__(_ScopeCacheKey, owner, kind, params, aux_ast_nodes)


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
    """Extracts ast.AST nodes held inside a cached value via cached_ast_nodes() or container traversal."""
    if value is None or isinstance(
        value, (bool, int, str, float, bytes, *_SINGLETON_AST_TYPES)
    ):
        return ()
    if isinstance(value, ast.AST):
        return (value,)
    if isinstance(value, (frozenset, set)) and all(
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
                if isinstance(n, ast.AST) and not isinstance(
                    n, _SINGLETON_AST_TYPES
                ):
                    found.append(n)
            continue
        node_attr = getattr(curr, "node", None)
        if isinstance(node_attr, ast.AST) and not isinstance(
            node_attr, _SINGLETON_AST_TYPES
        ):
            found.append(node_attr)
        pmap_attr = getattr(curr, "parent_map", None)
        if pmap_attr is not None and hasattr(pmap_attr, "items"):
            for pk, pv in pmap_attr.items():
                if isinstance(pk, ast.AST) and not isinstance(
                    pk, _SINGLETON_AST_TYPES
                ):
                    found.append(pk)
                if isinstance(pv, ast.AST) and not isinstance(
                    pv, _SINGLETON_AST_TYPES
                ):
                    found.append(pv)
        has_span_maps = False
        for map_attr_name in (
            "by_start",
            "by_start_line",
            "by_end_line",
            "stmts_by_start_line",
            "stmts_by_end_line",
        ):
            map_attr = getattr(curr, map_attr_name, None)
            if isinstance(map_attr, dict):
                has_span_maps = True
                for entries in map_attr.values():
                    for entry in entries:
                        enode = (
                            entry[0]
                            if isinstance(entry, tuple) and entry
                            else getattr(entry, "node", entry)
                        )
                        if isinstance(enode, ast.AST) and not isinstance(
                            enode, _SINGLETON_AST_TYPES
                        ):
                            found.append(enode)
        if has_span_maps:
            continue
        children: Sequence[Any] = ()
        if isinstance(curr, (tuple, list, set, frozenset)):
            children = tuple(curr)
        elif hasattr(curr, "items") and callable(curr.items):
            children = tuple(part for kv in curr.items() for part in kv)
        for elem in children:
            if isinstance(elem, ast.AST):
                if not isinstance(elem, _SINGLETON_AST_TYPES):
                    found.append(elem)
            elif not isinstance(
                elem, (bool, int, str, float, bytes, type(None))
            ):
                elem_id = id(elem)
                if elem_id not in seen:
                    seen.add(elem_id)
                    stack.append(elem)
    return tuple(found)


class _TrackedCachedDict(Dict[Any, Any]):
    """Dictionary value stored in _TrackedScopeCache that synchronizes _scope_owner_keys_index on mutation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_owners: Dict[Tuple[int, Any], Tuple[Any, Any]] = {}

    def bind_owner(self, cache: Any, cache_key: Any) -> None:
        """Registers an owning (_TrackedScopeCache, key) pair for index synchronization."""
        self._bound_owners[(id(cache), cache_key)] = (cache, cache_key)

    def unbind_owner(self, cache: Any, cache_key: Any) -> None:
        """Unregisters an owning (_TrackedScopeCache, key) pair."""
        self._bound_owners.pop((id(cache), cache_key), None)

    def _run_mutating(self, fn: Callable[[], Any]) -> Any:
        with _downstream_cache_lock:
            if not self._bound_owners:
                return fn()
            snapshots = [
                (c, ck, c.active_nodes_for_key(ck))
                for c, ck in list(self._bound_owners.values())
                if ck in c
            ]
            res = fn()
            for c, ck, old_nodes in snapshots:
                c.sync_key_index(ck, old_nodes)
            return res

    def __setitem__(self, k: Any, v: Any) -> None:
        self._run_mutating(lambda: dict.__setitem__(self, k, v))

    def __delitem__(self, k: Any) -> None:
        self._run_mutating(lambda: dict.__delitem__(self, k))

    def setdefault(self, k: Any, default: Any = None) -> Any:
        if k in self:
            return self[k]
        return self._run_mutating(lambda: dict.setdefault(self, k, default))

    def pop(self, k: Any, *args: Any) -> Any:
        return self._run_mutating(lambda: dict.pop(self, k, *args))

    def popitem(self) -> Tuple[Any, Any]:
        return cast(
            Tuple[Any, Any],
            self._run_mutating(lambda: dict.popitem(self)),
        )

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._run_mutating(lambda: dict.update(self, *args, **kwargs))

    def __or__(self, other: Any) -> Dict[Any, Any]:
        return dict.__or__(self, other)

    def __ior__(self, other: Any) -> _TrackedCachedDict:
        self._run_mutating(lambda: dict.__ior__(self, other))
        return self

    def clear(self) -> None:
        self._run_mutating(lambda: dict.clear(self))


class _OwnerKeysIndex(Dict[ast.AST, Set[Any]]):
    """Reverse index from AST nodes to active cache keys, syncing any untracked plain dicts on access."""

    def __init__(self) -> None:
        super().__init__()
        self._untracked_keys: Dict[Any, Set[ast.AST]] = {}

    def track_plain_dict_key(self, key: Any, nodes: Set[ast.AST]) -> None:
        """Records a cache key whose stored value is a plain mutable dict."""
        self._untracked_keys[key] = set(nodes)

    def untrack_plain_dict_key(self, key: Any) -> None:
        """Removes a cache key from plain-dict tracking."""
        self._untracked_keys.pop(key, None)

    def sync_pending(self) -> None:
        """Synchronizes AST node references for any plain dicts mutated in-place."""
        if not self._untracked_keys:
            return
        for key, prev_nodes in list(self._untracked_keys.items()):
            in_any = False
            for cache in (_scope_closures_cache, _scope_cell_info_cache):
                if key in cache:
                    in_any = True
                    cache.sync_key_index(key, prev_nodes)
                    self._untracked_keys[key] = cache.active_nodes_for_key(key)
                    break
            if not in_any:
                self._untracked_keys.pop(key, None)

    def __getitem__(self, k: ast.AST) -> Set[Any]:
        self.sync_pending()
        return super().__getitem__(k)

    def get(self, k: ast.AST, default: Any = None) -> Any:
        self.sync_pending()
        return super().get(k, default)

    def __contains__(self, o: object) -> bool:
        self.sync_pending()
        return super().__contains__(o)

    def __iter__(self) -> Iterator[ast.AST]:
        self.sync_pending()
        return super().__iter__()

    def __len__(self) -> int:
        self.sync_pending()
        return super().__len__()

    def keys(self) -> Any:
        self.sync_pending()
        return super().keys()

    def values(self) -> Any:
        self.sync_pending()
        return super().values()

    def items(self) -> Any:
        self.sync_pending()
        return super().items()

    def clear(self) -> None:
        self._untracked_keys.clear()
        super().clear()


_scope_owner_keys_index: _OwnerKeysIndex = _OwnerKeysIndex()


class _TrackedScopeCache(OrderedDict[Any, _CacheValT]):
    """OrderedDict cache that maintains _scope_owner_keys_index for keys and cached values."""

    def active_nodes_for_key(self, key: Any) -> Set[ast.AST]:
        """Returns all AST nodes referenced by key and its currently cached values."""
        nodes: Set[ast.AST] = set()
        in_any = False
        for cache in (_scope_closures_cache, _scope_cell_info_cache):
            if key in cache:
                in_any = True
                nodes.update(_extract_value_ast_nodes(dict.__getitem__(cache, key)))
        if in_any:
            nodes.update(_extract_key_ast_nodes(key))
        return nodes

    def sync_key_index(self, key: Any, old_nodes: Set[ast.AST]) -> None:
        """Synchronizes _scope_owner_keys_index for key given its previous AST node set."""
        new_nodes = self.active_nodes_for_key(key)
        for node in old_nodes - new_nodes:
            kset = dict.get(_scope_owner_keys_index, node)
            if kset is not None:
                kset.discard(key)
                if not kset:
                    dict.pop(_scope_owner_keys_index, node, None)
        for node in new_nodes - old_nodes:
            dict.setdefault(_scope_owner_keys_index, node, set()).add(key)
        has_plain_dict = any(
            key in c
            and isinstance(dict.__getitem__(c, key), dict)
            and not isinstance(dict.__getitem__(c, key), _TrackedCachedDict)
            for c in (_scope_closures_cache, _scope_cell_info_cache)
        )
        if has_plain_dict:
            _scope_owner_keys_index.track_plain_dict_key(key, new_nodes)
        else:
            _scope_owner_keys_index.untrack_plain_dict_key(key)

    def __setitem__(self, key: Any, value: _CacheValT) -> None:
        old_nodes = self.active_nodes_for_key(key) if key in self else set()
        if key in self:
            prev_val = dict.__getitem__(self, key)
            if isinstance(prev_val, _TrackedCachedDict) and prev_val is not value:
                prev_val.unbind_owner(self, key)
        if isinstance(value, _TrackedCachedDict):
            value.bind_owner(self, key)
        super().__setitem__(key, value)
        self.sync_key_index(key, old_nodes)

    def __delitem__(self, key: Any) -> None:
        old_nodes = self.active_nodes_for_key(key)
        if key in self:
            prev_val = dict.__getitem__(self, key)
            if isinstance(prev_val, _TrackedCachedDict):
                prev_val.unbind_owner(self, key)
        super().__delitem__(key)
        self.sync_key_index(key, old_nodes)

    def pop(self, key: Any, *args: Any) -> Any:
        old_nodes = self.active_nodes_for_key(key) if key in self else set()
        if key in self:
            prev_val = dict.__getitem__(self, key)
            if isinstance(prev_val, _TrackedCachedDict):
                prev_val.unbind_owner(self, key)
        res = super().pop(key, *args)
        if old_nodes:
            self.sync_key_index(key, old_nodes)
        return res

    def popitem(self, last: bool = True) -> Tuple[Any, _CacheValT]:
        if not self:
            return super().popitem(last=last)
        target_key = next(reversed(self) if last else iter(self))
        old_nodes = self.active_nodes_for_key(target_key)
        k, v = super().popitem(last=last)
        if isinstance(v, _TrackedCachedDict):
            v.unbind_owner(self, k)
        self.sync_key_index(k, old_nodes)
        return k, v

    def clear(self) -> None:
        old_map = {k: self.active_nodes_for_key(k) for k in self}
        for k in list(self.keys()):
            prev_val = dict.__getitem__(self, k)
            if isinstance(prev_val, _TrackedCachedDict):
                prev_val.unbind_owner(self, k)
        super().clear()
        for k, old_nodes in old_map.items():
            self.sync_key_index(k, old_nodes)


_scope_closures_cache: _TrackedScopeCache[List[Tuple[Any, Set[str]]]] = (
    _TrackedScopeCache()
)
_scope_cell_info_cache: _TrackedScopeCache[Any] = _TrackedScopeCache()


def _key_references_scope_nodes(
    key: Any,
    tree_nodes: Set[ast.AST],
    safe_containers: Optional[Set[int]] = None,
) -> bool:
    """Returns True if a cache key references any AST node in tree_nodes.

    Cache Key & Value Policy:
    Production entries in _scope_closures_cache and _scope_cell_info_cache use
    _ScopeCacheKey(owner, kind, params, aux_ast_nodes), enabling O(1) owner
    extraction via explicit AST owner fields and indexed tree eviction via
    _scope_owner_keys_index without scanning unrelated cache entries.
    For arbitrary tuple, NamedTuple, or frozenset keys (including heterogeneous
    frozensets), every element is inspected recursively. Cached values must be
    immutable (frozenset, tuple, NamedTuple, MappingProxyType, bool), protected
    by _downstream_cache_lock on inner mutation (e.g., stmt_deltas), or returned
    via defensive copies to callers.
    """
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


def _evict_tree_scopes(tree: Optional[ast.AST]) -> None:
    """Purges scope-cache entries belonging to an evicted AST tree or detached subtree.

    Note:
        Because _TrackedScopeCache indexes AST nodes referenced by both cache keys
        and cached values (including _FnScopeContext.parent_map, _SpanLineIndex,
        closures, and statement deltas), and _evict_tree_scopes also checks
        _scope_parent_maps for enclosing scopes referencing nodes in tree_nodes,
        evicting either a whole AST tree or an independently detached AST subtree
        invalidates all cached contexts and entries holding references to it.
    """
    if tree is None or not isinstance(tree, ast.AST):
        return
    tree_nodes: Set[ast.AST] = {
        n for n in ast.walk(tree) if not isinstance(n, _SINGLETON_AST_TYPES)
    }
    scope_nodes: Set[ast.AST] = {
        node for node in tree_nodes if isinstance(node, _SCOPE_ROOT_TYPES)
    }
    scope_nodes.add(tree)
    with _downstream_cache_lock:
        _scope_owner_keys_index.sync_pending()
        for root_node, pmap in list(_scope_parent_maps.items()):
            if root_node not in scope_nodes and (
                tree in pmap or any(n in pmap for n in tree_nodes)
            ):
                scope_nodes.add(root_node)
                tree_nodes.add(root_node)
                curr_anc: Optional[ast.AST] = pmap.get(tree)
                while curr_anc is not None and curr_anc not in tree_nodes:
                    tree_nodes.add(curr_anc)
                    if isinstance(curr_anc, _SCOPE_ROOT_TYPES):
                        scope_nodes.add(curr_anc)
                    curr_anc = pmap.get(curr_anc)
        for stmt_cache in (_scope_stmts_by_start, _scope_stmts_by_end):
            for root_node, line_map in list(stmt_cache.items()):
                if root_node not in scope_nodes and any(
                    s in tree_nodes
                    for stmts in line_map.values()
                    for s in stmts
                ):
                    scope_nodes.add(root_node)
                    tree_nodes.add(root_node)
        keys_to_remove: Set[Any] = set()
        for node in tree_nodes:
            node_keys = dict.get(_scope_owner_keys_index, node)
            if node_keys:
                keys_to_remove.update(node_keys)
        for k in keys_to_remove:
            for knode in _extract_key_ast_nodes(k):
                if isinstance(knode, _SCOPE_ROOT_TYPES):
                    scope_nodes.add(knode)
            _scope_closures_cache.pop(k, None)
            _scope_cell_info_cache.pop(k, None)
        for node in tree_nodes:
            _scope_owner_keys_index.pop(node, None)
        for node in scope_nodes:
            _scope_parent_maps.pop(node, None)
            _scope_stmts_by_end.pop(node, None)
            _scope_stmts_by_start.pop(node, None)
            _scope_loops_cache.pop(node, None)
            _scope_try_with_cache.pop(node, None)


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
    effective_fn: Callable[[], _CacheValT] = (
        cast(Callable[[], _CacheValT], _TrackedCachedDict)
        if compute_fn is dict and isinstance(cache, _TrackedScopeCache)
        else compute_fn
    )
    computed = effective_fn()

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


def _get_scope_parent_map(
    scope_node: ast.AST, *, use_cache: bool = True
) -> Dict[ast.AST, ast.AST]:
    """Retrieves or builds cached parent map for AST nodes within the scope.

    Note:
        The returned dictionary is stored in ``_scope_parent_maps`` and wrapped in
        ``MappingProxyType`` by ``_FnScopeContext`` (and shared across nested scope
        contexts). Callers must treat the underlying dictionary as strictly
        read-only and never mutate it after construction.
    """
    def _compute() -> Dict[ast.AST, ast.AST]:
        parent_map: Dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(scope_node):
            for child in ast.iter_child_nodes(parent):
                if not isinstance(child, _SINGLETON_AST_TYPES):
                    parent_map[child] = parent
        return parent_map

    if not use_cache:
        return _compute()
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
        _scope_parent_maps.clear()
        _scope_stmts_by_end.clear()
        _scope_stmts_by_start.clear()
        _scope_closures_cache.clear()
        _scope_cell_info_cache.clear()
        _scope_owner_keys_index.clear()
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
    is_subroutine: bool = False


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
    is_sub_key = bool(unit.get("is_subroutine")) or unit.get("kind") in (
        "compound_block",
        "sliding_window",
        "clause_branch",
        "comprehension",
        "complex_expr",
        "data_table",
    )

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
    )
