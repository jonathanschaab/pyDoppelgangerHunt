"""Dataflow cache structures, LRU memoization, and AST coordinate caching."""

from __future__ import annotations

import ast
from collections import OrderedDict
import contextlib
import hashlib
import logging
import threading
from types import MappingProxyType
import warnings
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
from pydoppelgangerhunt.source_lines import parse_unit_coord, split_source_lines

logger = logging.getLogger(__name__)

_CacheValT = TypeVar("_CacheValT")

_downstream_cache_lock: threading.RLock = threading.RLock()
_syntax_warning_lock: threading.RLock = threading.RLock()
_RLOCK_TYPE: type = type(_downstream_cache_lock)


@contextlib.contextmanager
def _suppress_syntax_warnings() -> Iterator[None]:
    """Suppresses SyntaxWarning during AST/symtable parsing, serializing callers of this helper via _syntax_warning_lock."""
    with _syntax_warning_lock, warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        yield


_MAX_DOWNSTREAM_CACHE_SIZE: int = 1024
_MAX_SCOPE_CACHE_SIZE: int = 4096
_MAX_PARSED_TREE_CACHE_SIZE: int = 128
_SCOPE_CACHE_KIND_CAPS: Dict[str, int] = {
    "span_line_index": 128,
    "fn_scope_ctx": 512,
    "stmt_deltas": 512,
    "enclosing_fn_cells": 512,
    "suppress_seed": 256,
}
_downstream_reads_cache: OrderedDict[Tuple[Any, ...], frozenset[str]] = OrderedDict()
_parsed_tree_cache: OrderedDict[Tuple[str, int], Optional[ast.AST]] = OrderedDict()

_scope_loops_cache: OrderedDict[ast.AST, List[Tuple[int, int, int]]] = OrderedDict()
_scope_try_with_cache: OrderedDict[ast.AST, List[ast.AST]] = OrderedDict()
_stmt_with_summary_cache: OrderedDict[ast.stmt, Tuple[bool, frozenset[str]]] = OrderedDict()
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
            if not isinstance(curr, (tuple, frozenset)) and invalid_mutable_type is None:
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


def _params_contain_ast(params: Tuple[Any, ...]) -> bool:
    """Checks recursively whether any element or nested container in params is an ast.AST node."""
    try:
        _validate_scope_cache_params(params)
        return False
    except ValueError:
        return True
    except TypeError:
        return False


def _params_exact_types_match(a: Any, b: Any) -> bool:
    """Checks that equal parameter structures also match exact scalar types (distinguishing 1 from True)."""
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
        if 1 not in a and 0 not in a and 1 not in b and 0 not in b:
            return True
        return {(type(x), x) for x in a} == {(type(y), y) for y in b}
    return True


def _params_type_hash(obj: Any) -> int:
    """Computes a structural type hash for params so 1 and True do not hash-collide."""
    if isinstance(obj, tuple):
        return hash((tuple, tuple(_params_type_hash(x) for x in obj)))
    if isinstance(obj, frozenset):
        if 1 not in obj and 0 not in obj:
            return hash((frozenset, obj))
        return hash((frozenset, frozenset((type(x), x) for x in obj)))
    if isinstance(obj, (bool, int, float)):
        return hash((type(obj), obj))
    return hash(type(obj))


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
    """Uniform cache key with explicit AST owner(s) for O(1) owner extraction and indexed tree eviction.

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
        ``_evict_tree_scopes(tree)`` call. Synthesized ephemeral unit modules must always
        set ``_is_ephemeral_unit_mod = True`` and bypass node-keyed caches so detached AST
        nodes never enter ``_scope_owner_keys_index``. Any future code path that mutates a
        cached AST in place must call ``_evict_tree_scopes(tree)`` prior to mutation so
        reverse owner and parent-map indexes never become stale.
    """

    def __new__(
        cls,
        owner: ast.AST,
        kind: str,
        params: Tuple[Any, ...] = (),
        aux_ast_nodes: Tuple[ast.AST, ...] = (),
    ) -> _ScopeCacheKey:
        _validate_scope_cache_key_header(owner, kind, params, aux_ast_nodes)
        return super().__new__(cls, owner, kind, params, aux_ast_nodes)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ScopeCacheKey):
            return False
        if not tuple.__eq__(self, other):
            return False
        if not self.params:
            return True
        return _params_exact_types_match(self.params, other.params)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        cached = self.__dict__.get("_cached_hash")
        if cached is not None:
            return int(cached)
        base_h = tuple.__hash__(self)
        res = (
            base_h
            if not self.params
            else hash((base_h, _params_type_hash(self.params)))
        )
        self.__dict__["_cached_hash"] = res
        return res


def _make_trusted_scope_cache_key(
    owner: ast.AST,
    kind: str,
    params: Tuple[Any, ...] = (),
    aux_ast_nodes: Tuple[ast.AST, ...] = (),
) -> _ScopeCacheKey:
    """Fast-path constructor for internal callers whose arguments are already trusted."""
    return _BaseScopeCacheKey.__new__(
        _ScopeCacheKey, owner, kind, params, aux_ast_nodes
    )


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
                if isinstance(n, ast.AST) and not isinstance(
                    n, _SINGLETON_AST_TYPES
                ):
                    found.append(n)
            continue
        curr_type = type(curr)
        children: Sequence[Any] = ()
        if curr_type in (tuple, list, set, frozenset):
            children = tuple(curr)
        elif curr_type in (dict, _TrackedCachedDict, MappingProxyType):
            children = tuple(part for kv in curr.items() for part in kv)
        else:
            raise TypeError(
                f"Cached value of custom type {curr_type.__name__} must implement cached_ast_nodes()"
            )
        for elem in children:
            if isinstance(elem, ast.AST):
                if not isinstance(elem, _SINGLETON_AST_TYPES):
                    found.append(elem)
            elif not isinstance(
                elem, (bool, int, str, float, bytes, type(None), *_SINGLETON_AST_TYPES)
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
        if isinstance(node, ast.AST) and not isinstance(
            node, _SINGLETON_AST_TYPES
        ):
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


def _store_parent_map_locked(
    root_node: ast.AST, parent_map: Dict[ast.AST, ast.AST]
) -> Dict[ast.AST, ast.AST]:
    """Stores parent_map in _scope_parent_maps with LRU cap and reverse node-to-root indexing."""
    if root_node in _scope_parent_maps:
        _scope_parent_maps.move_to_end(root_node)
        return _scope_parent_maps[root_node]
    while len(_scope_parent_maps) >= _MAX_SCOPE_CACHE_SIZE and _scope_parent_maps:
        evicted_root, _ = _scope_parent_maps.popitem(last=False)
        _prune_reverse_index_buckets(
            evicted_root, _parent_map_root_nodes, _node_to_parent_map_roots
        )
    _scope_parent_maps[root_node] = parent_map
    _scope_parent_maps.move_to_end(root_node)
    scope_boundary_nodes = [root_node]
    for top_stmt in getattr(root_node, "body", ()):
        if isinstance(top_stmt, ast.AST):
            scope_boundary_nodes.append(top_stmt)
    for child_node in parent_map:
        if isinstance(child_node, _SCOPE_ROOT_TYPES):
            scope_boundary_nodes.append(child_node)
    _index_entry_ast_nodes(
        root_node,
        scope_boundary_nodes,
        _parent_map_root_nodes,
        _node_to_parent_map_roots,
    )
    return parent_map


def _pop_parent_map_locked(root_node: ast.AST) -> None:
    """Removes root_node from _scope_parent_maps and prunes its reverse node-to-root index entries."""
    if root_node in _scope_parent_maps:
        _scope_parent_maps.pop(root_node, None)
    _prune_reverse_index_buckets(
        root_node, _parent_map_root_nodes, _node_to_parent_map_roots
    )


def _ensure_scope_owner_parent_map_locked(owner: ast.AST) -> None:
    """Ensures a scope root has a parent map recorded in O(1) if already indexed, else builds and caches it."""
    if (
        not isinstance(owner, _SCOPE_ROOT_TYPES)
        or owner in _scope_parent_maps
        or owner in _node_to_parent_map_roots
    ):
        return
    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(owner):
        for child in ast.iter_child_nodes(parent):
            if not isinstance(child, _SINGLETON_AST_TYPES):
                parent_map[child] = parent
    _store_parent_map_locked(owner, parent_map)


def _register_key_ast_nodes(key: Any, nodes: Sequence[ast.AST]) -> None:
    """Incrementally registers AST nodes for key in _scope_owner_keys_index in O(|nodes|)."""
    if nodes:
        _index_entry_ast_nodes(
            key, nodes, _scope_key_nodes_index, _scope_owner_keys_index
        )


def _unregister_key_if_absent(key: Any) -> None:
    """Removes key from _scope_owner_keys_index when no longer held in any scope cache."""
    if key not in _scope_closures_cache and key not in _scope_cell_info_cache:
        _prune_reverse_index_buckets(
            key, _scope_key_nodes_index, _scope_owner_keys_index
        )


class _TrackedCachedDict(Dict[Any, Any]):
    """Dictionary value stored in _TrackedScopeCache that registers new AST nodes in O(1) on insert."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bound_owners: Dict[Tuple[int, Any], Tuple[Any, Any]] = {}

    def _ensure_bound_owners(self) -> Dict[Tuple[int, Any], Tuple[Any, Any]]:
        return cast(
            Dict[Tuple[int, Any], Tuple[Any, Any]],
            self.__dict__.setdefault("_bound_owners", {}),
        )

    def copy(self) -> _TrackedCachedDict:
        """Returns a shallow copy with an independent empty _bound_owners table."""
        with _downstream_cache_lock:
            return _TrackedCachedDict(self)

    def __copy__(self) -> _TrackedCachedDict:
        return self.copy()

    def bind_owner(self, cache: Any, cache_key: Any) -> None:
        """Registers an owning (_TrackedScopeCache, key) pair for incremental node registration."""
        self._ensure_bound_owners()[(id(cache), cache_key)] = (cache, cache_key)

    def unbind_owner(self, cache: Any, cache_key: Any) -> None:
        """Unregisters an owning (_TrackedScopeCache, key) pair."""
        self._ensure_bound_owners().pop((id(cache), cache_key), None)

    def _register_inserted_pair(self, k: Any, v: Any) -> None:
        owners = getattr(self, "_bound_owners", None)
        if not owners:
            return
        k_nodes = _extract_value_ast_nodes(k)
        v_nodes = _extract_value_ast_nodes(v)
        if not k_nodes and not v_nodes:
            return
        combined = (*k_nodes, *v_nodes)
        for cache, ck in list(owners.values()):
            if ck in cache:
                _register_key_ast_nodes(ck, combined)

    def _sync_owners_ast_nodes(self) -> None:
        owners = getattr(self, "_bound_owners", None)
        if not owners:
            return
        dict_nodes = _extract_value_ast_nodes(dict(self))
        for cache, ck in list(owners.values()):
            if ck in cache:
                _prune_reverse_index_buckets(
                    ck, _scope_key_nodes_index, _scope_owner_keys_index
                )
                key_nodes = _extract_key_ast_nodes(ck)
                _register_key_ast_nodes(ck, (*key_nodes, *dict_nodes))

    def __setitem__(self, k: Any, v: Any) -> None:
        with _downstream_cache_lock:
            overwriting = k in self
            super().__setitem__(k, v)
            if overwriting:
                self._sync_owners_ast_nodes()
            else:
                self._register_inserted_pair(k, v)

    def __delitem__(self, k: Any) -> None:
        with _downstream_cache_lock:
            super().__delitem__(k)
            self._sync_owners_ast_nodes()

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
                self._sync_owners_ast_nodes()
            return res

    def popitem(self) -> Tuple[Any, Any]:
        with _downstream_cache_lock:
            item = super().popitem()
            self._sync_owners_ast_nodes()
            return item

    def update(self, *args: Any, **kwargs: Any) -> None:
        with _downstream_cache_lock:
            temp = dict(*args, **kwargs)
            had_overwrite = any(k in self for k in temp)
            super().update(temp)
            if had_overwrite:
                self._sync_owners_ast_nodes()
            else:
                for k, v in temp.items():
                    self._register_inserted_pair(k, v)

    def __or__(self, other: Any) -> Dict[Any, Any]:
        return dict.__or__(self, other)

    def __ior__(self, other: Any) -> _TrackedCachedDict:
        with _downstream_cache_lock:
            temp = dict(other)
            had_overwrite = any(k in self for k in temp)
            super().update(temp)
            if had_overwrite:
                self._sync_owners_ast_nodes()
            else:
                for k, v in temp.items():
                    self._register_inserted_pair(k, v)
            return self

    def clear(self) -> None:
        with _downstream_cache_lock:
            super().clear()
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

    def __setitem__(self, key: Any, value: _CacheValT) -> None:
        val_nodes = _extract_value_ast_nodes(value)
        key_nodes = _extract_key_ast_nodes(key)
        if isinstance(value, dict) and not isinstance(value, _TrackedCachedDict):
            value = cast(_CacheValT, _TrackedCachedDict(cast(Dict[Any, Any], value)))
        if key in self:
            prev_val = dict.__getitem__(self, key)
            if isinstance(prev_val, _TrackedCachedDict) and prev_val is not value:
                prev_val.unbind_owner(self, key)
        if isinstance(value, _TrackedCachedDict):
            value.bind_owner(self, key)
        super().__setitem__(key, value)
        if isinstance(key, _ScopeCacheKey):
            kind_ring = self.ensure_kind_keys().setdefault(key.kind, OrderedDict())
            kind_ring[key] = None
            kind_ring.move_to_end(key)
            if not val_nodes and isinstance(key.owner, _SCOPE_ROOT_TYPES):
                _ensure_scope_owner_parent_map_locked(key.owner)
        _register_key_ast_nodes(key, (*key_nodes, *val_nodes))

    def move_to_end(self, key: Any, last: bool = True) -> None:
        super().move_to_end(key, last=last)
        if isinstance(key, _ScopeCacheKey):
            kind_ring = self.ensure_kind_keys().get(key.kind)
            if kind_ring is not None and key in kind_ring:
                kind_ring.move_to_end(key, last=last)

    def __delitem__(self, key: Any) -> None:
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
        for k in list(self.keys()):
            prev_val = dict.__getitem__(self, k)
            if isinstance(prev_val, _TrackedCachedDict):
                prev_val.unbind_owner(self, k)
        self.ensure_kind_keys().clear()
        super().clear()
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


def _evict_tree_scopes(tree: Optional[ast.AST]) -> None:
    """Purges scope-cache entries belonging to an evicted AST tree or detached subtree.

    Note:
        Walks ``tree`` and inspects ``_node_to_parent_map_roots``, ``_scope_parent_maps``,
        and statement-line caches (``_scope_stmts_by_start`` / ``_scope_stmts_by_end``) to
        identify enclosing ancestor scopes, then uses ``_scope_owner_keys_index`` to evict
        matching keys from ``_scope_closures_cache`` and ``_scope_cell_info_cache`` without
        iterating over all cached scope keys or flushing unrelated sibling scopes.
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
        candidate_roots: Set[ast.AST] = set()
        if tree in _scope_parent_maps:
            candidate_roots.add(tree)
        for n in scope_nodes:
            roots = _node_to_parent_map_roots.get(n)
            if roots:
                candidate_roots.update(roots)
        for r_node, r_pmap in _scope_parent_maps.items():
            if tree in r_pmap:
                candidate_roots.add(r_node)
        keys_to_remove: Set[Any] = set()
        for root_node in candidate_roots:
            for rk in _scope_owner_keys_index.get(root_node, ()):
                if rk is root_node or (
                    isinstance(rk, _ScopeCacheKey) and rk.owner is root_node
                ):
                    keys_to_remove.add(rk)
            pmap = _scope_parent_maps.get(root_node)
            if pmap is None:
                continue
            matched_child: Optional[ast.AST] = None
            if tree in pmap:
                matched_child = tree
            elif root_node in tree_nodes:
                matched_child = root_node
            else:
                for n in tree_nodes:
                    if n in pmap:
                        matched_child = n
                        break
            if matched_child is None:
                continue
            scope_nodes.add(root_node)
            has_inner_scope_anc = isinstance(
                matched_child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            )
            curr_anc: Optional[ast.AST] = pmap.get(matched_child)
            while curr_anc is not None:
                if curr_anc is not root_node or not isinstance(
                    root_node, ast.Module
                ):
                    tree_nodes.add(curr_anc)
                if isinstance(curr_anc, _SCOPE_ROOT_TYPES):
                    scope_nodes.add(curr_anc)
                    if curr_anc is not root_node:
                        has_inner_scope_anc = True
                curr_anc = pmap.get(curr_anc)
            if not has_inner_scope_anc:
                tree_nodes.add(root_node)
        for stmt_cache in (_scope_stmts_by_start, _scope_stmts_by_end):
            for s_owner, line_map in list(stmt_cache.items()):
                if s_owner in scope_nodes:
                    continue
                t_lineno = getattr(tree, "lineno", 0) or 0
                if t_lineno > 0 and any(
                    s in tree_nodes for s in line_map.get(t_lineno, ())
                ):
                    scope_nodes.add(s_owner)
                    tree_nodes.add(s_owner)
        for node in tree_nodes:
            node_keys = _scope_owner_keys_index.get(node)
            if node_keys:
                keys_to_remove.update(node_keys)
        for k in keys_to_remove:
            for knode in _extract_key_ast_nodes(k):
                if isinstance(knode, _SCOPE_ROOT_TYPES):
                    scope_nodes.add(knode)
            _scope_closures_cache.pop(k, None)
            _scope_cell_info_cache.pop(k, None)
            _unregister_key_if_absent(k)
        for node in tree_nodes:
            _scope_owner_keys_index.pop(node, None)
            if isinstance(node, ast.stmt):
                _stmt_with_summary_cache.pop(node, None)
        for node in scope_nodes:
            _pop_parent_map_locked(node)
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
    evicted_trees: List[ast.AST] = []
    with _downstream_cache_lock:
        if (
            cache is _scope_parent_maps
            and isinstance(key, ast.AST)
            and isinstance(value, dict)
        ):
            return cast(_CacheValT, _store_parent_map_locked(key, value))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        if isinstance(cache, _TrackedScopeCache) and isinstance(
            key, _ScopeCacheKey
        ):
            kind_cap = _SCOPE_CACHE_KIND_CAPS.get(key.kind)
            if kind_cap is not None and max_size > kind_cap:
                kind_ring = cache.ensure_kind_keys().get(key.kind)
                while kind_ring and len(kind_ring) >= kind_cap:
                    oldest_k, _ = kind_ring.popitem(last=False)
                    cache.pop(oldest_k, None)
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
        return _store_lru_cache_entry(
            cache, scope_node, computed, _MAX_SCOPE_CACHE_SIZE
        )
    except TypeError as exc:
        if "cached_ast_nodes()" in str(exc):
            raise
        logger.debug(
            "Skipping scope cache insertion due to unhashable key: %s",
            exc,
        )
        return computed


def _split_source_lines_stripped(source_text: str) -> Tuple[str, ...]:
    """Splits source_text on LF, CRLF, and CR boundaries (matching CPython's tokenizer) and strips line endings."""
    normalized = (
        source_text.replace("\r\n", "\n").replace("\r", "\n")
        if "\r" in source_text
        else source_text
    )
    return tuple(ln.rstrip("\r\n") for ln in split_source_lines(normalized))


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
        with _suppress_syntax_warnings():
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
    return _get_or_compute_scope_cached(
        _scope_parent_maps, scope_node, _compute
    )


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
    raw_src_code = unit.get("source_code")
    unit_source_hash: Optional[str] = None
    if (start_col is None or end_col is None) and isinstance(raw_src_code, str) and raw_src_code.strip():
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
