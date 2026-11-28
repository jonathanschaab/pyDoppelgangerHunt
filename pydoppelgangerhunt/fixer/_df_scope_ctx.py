"""Enclosing function scope context, span coordinate bounds, and deferred closure extraction."""

from __future__ import annotations

import ast
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Callable,
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
    _SessionBucket,
    _ephemeral_unit_ast_context,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _mark_ephemeral_unit_ast,
    _tree_contains_scope_node,
    analysis_session,
)
from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_non_name_bound_names,
    _extract_store_names_in_node,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _extract_comprehension_walrus_names,
    _extract_scope_local_names,
    _extract_strict_fn_local_names,
    _has_wildcard_star_import,
    _is_eagerly_consumed_genexp,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _extract_nested_scope_free_reads,
    _is_ast_node_within_unit_bounds,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

def _is_node_attached_to_tree(node: ast.AST, mod_tree: ast.AST) -> bool:
    """Returns True if node is mod_tree or belongs to mod_tree without rebuilding oversized maps."""
    return _tree_contains_scope_node(mod_tree, node)


def _compute_scope_ctx_aux_nodes(
    scope_owner: Optional[ast.AST],
    mod_tree: Optional[ast.AST],
    enclosing_scopes: Tuple[ast.AST, ...],
    has_parent_map: bool,
) -> Tuple[ast.AST, ...]:
    """Computes attached AST owner, module root, and enclosing scope nodes for _FnScopeContext."""
    has_mod_tree = isinstance(mod_tree, ast.AST)
    aux_nodes: List[ast.AST] = []
    if isinstance(scope_owner, ast.AST) and not _is_ephemeral_unit_ast(scope_owner):
        if has_mod_tree and mod_tree is not None:
            if _tree_contains_scope_node(mod_tree, scope_owner):
                aux_nodes.append(scope_owner)
        elif not isinstance(scope_owner, ast.Module) or has_parent_map:
            aux_nodes.append(scope_owner)
    if has_mod_tree and mod_tree is not None and not _is_ephemeral_unit_ast(mod_tree):
        aux_nodes.append(mod_tree)
    for enc in enclosing_scopes:
        if isinstance(enc, ast.AST) and not _is_ephemeral_unit_ast(enc):
            if (
                not has_mod_tree
                or mod_tree is None
                or _tree_contains_scope_node(mod_tree, enc)
            ):
                aux_nodes.append(enc)
    return tuple(aux_nodes)


class _FnScopeContext:
    """Cached per-function lexical scope metadata for eager generator expression checks.

    Note:
        ``parent_map`` is a read-only ``MappingProxyType`` view. When a cached
        scope exceeds ``_MAX_PARENT_MAP_TOTAL_NODES``, ``_stored_parent_map`` is
        left ``None`` so ``_SessionBucket.CELL_INFO`` never pins an oversized AST
        parent dictionary in memory; ``parent_map`` resolves transiently on demand.
    """

    __slots__ = (
        "fn_locals",
        "_stored_parent_map",
        "has_nonempty_parent_map",
        "shadowed_names",
        "has_star_import",
        "scope_owner",
        "mod_tree",
        "enclosing_scopes",
        "precomputed_aux_nodes",
    )

    fn_locals: frozenset[str]
    _stored_parent_map: Optional[Mapping[ast.AST, ast.AST]]
    has_nonempty_parent_map: bool
    shadowed_names: frozenset[str]
    has_star_import: bool
    scope_owner: Optional[ast.AST]
    mod_tree: Optional[ast.AST]
    enclosing_scopes: Tuple[ast.AST, ...]
    precomputed_aux_nodes: Optional[Tuple[ast.AST, ...]]

    def __init__(
        self,
        fn_locals: frozenset[str],
        parent_map: Optional[Mapping[ast.AST, ast.AST]],
        shadowed_names: frozenset[str],
        has_star_import: bool,
        scope_owner: Optional[ast.AST] = None,
        mod_tree: Optional[ast.AST] = None,
        enclosing_scopes: Tuple[ast.AST, ...] = (),
        precomputed_aux_nodes: Optional[Tuple[ast.AST, ...]] = None,
        *,
        has_nonempty_parent_map: Optional[bool] = None,
    ) -> None:
        object.__setattr__(self, "fn_locals", fn_locals)
        object.__setattr__(self, "_stored_parent_map", parent_map)
        object.__setattr__(
            self,
            "has_nonempty_parent_map",
            bool(parent_map)
            if has_nonempty_parent_map is None
            else bool(has_nonempty_parent_map),
        )
        object.__setattr__(self, "shadowed_names", shadowed_names)
        object.__setattr__(self, "has_star_import", has_star_import)
        object.__setattr__(self, "scope_owner", scope_owner)
        object.__setattr__(self, "mod_tree", mod_tree)
        object.__setattr__(self, "enclosing_scopes", enclosing_scopes)
        object.__setattr__(
            self, "precomputed_aux_nodes", precomputed_aux_nodes
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("_FnScopeContext is immutable")

    @property
    def parent_map(self) -> Mapping[ast.AST, ast.AST]:
        """Returns the read-only AST child-to-parent mapping, resolving oversized scopes lazily."""
        if self._stored_parent_map is not None:
            return self._stored_parent_map
        target = (
            self.scope_owner
            if isinstance(self.scope_owner, ast.AST)
            else self.mod_tree
        )
        if isinstance(target, ast.AST):
            return MappingProxyType(_get_scope_parent_map(target, use_cache=True))
        return MappingProxyType({})

    def cache_fingerprint(self) -> Tuple[Any, ...]:
        """Returns the non-AST semantic fingerprint of this scope context."""
        return (
            self.fn_locals,
            self.shadowed_names,
            self.has_star_import,
            self.has_nonempty_parent_map,
        )

    def cache_aux_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns attached AST owner and enclosing scope nodes referenced by this context in O(1)."""
        if self.precomputed_aux_nodes is not None:
            return self.precomputed_aux_nodes
        return _compute_scope_ctx_aux_nodes(
            self.scope_owner,
            self.mod_tree,
            self.enclosing_scopes,
            self.has_nonempty_parent_map,
        )


def _scope_ctx_allows_node_cache(
    owner: ast.AST,
    scope_ctx: Optional[_FnScopeContext],
) -> bool:
    """Returns False when owner or scope_ctx.scope_owner is detached from the target AST."""
    if scope_ctx is None:
        return True
    if _is_ephemeral_unit_ast(owner):
        return False
    stored_pmap = scope_ctx._stored_parent_map  # pylint: disable=protected-access
    if isinstance(scope_ctx.mod_tree, ast.AST):
        if isinstance(scope_ctx.scope_owner, ast.AST):
            if not _is_node_attached_to_tree(scope_ctx.scope_owner, scope_ctx.mod_tree):
                return False
            if owner is scope_ctx.scope_owner or (
                stored_pmap is not None and owner in stored_pmap
            ):
                return True
        return _is_node_attached_to_tree(owner, scope_ctx.mod_tree)
    if isinstance(scope_ctx.scope_owner, ast.AST):
        if _is_ephemeral_unit_ast(scope_ctx.scope_owner):
            return False
        if scope_ctx.scope_owner is owner:
            return True
        if stored_pmap is None:
            return _is_node_attached_to_tree(owner, scope_ctx.scope_owner)
        if (
            scope_ctx.scope_owner not in stored_pmap
            and owner not in stored_pmap
        ):
            return False
    return True


def _make_scope_ctx_cache_key(
    owner: ast.AST,
    kind: str,
    scope_ctx: Optional[_FnScopeContext],
    extra_params: Tuple[Any, ...] = (),
) -> _ScopeCacheKey:
    """Builds a _ScopeCacheKey incorporating full _FnScopeContext identity and AST owners."""
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
    scope_owner: Optional[ast.AST] = None,
    parent_map_override: Optional[Mapping[ast.AST, ast.AST]] = None,
) -> _FnScopeContext:
    """Computes lexical scope metadata for a function or ephemeral unit module without caching."""
    has_nonempty_pmap: bool
    if parent_map_override is not None:
        eff_parent_map: Optional[Mapping[ast.AST, ast.AST]] = (
            parent_map_override
            if isinstance(parent_map_override, MappingProxyType)
            else MappingProxyType(dict(parent_map_override))
        )
        has_nonempty_pmap = bool(eff_parent_map)
    else:
        if (
            cache_parent_map
            and isinstance(mod_tree, ast.AST)
            and not _is_ephemeral_unit_ast(mod_tree)
            and scope_node is not mod_tree
        ):
            _get_scope_parent_map(mod_tree, use_cache=True)
        raw_parent_map = _get_scope_parent_map(
            scope_node, use_cache=cache_parent_map
        )
        has_nonempty_pmap = bool(raw_parent_map)
        eff_parent_map = MappingProxyType(raw_parent_map)
    if scope_owner is None and (
        cache_parent_map or not isinstance(scope_node, ast.Module)
    ):
        eff_owner: Optional[ast.AST] = scope_node
    else:
        eff_owner = scope_owner
    if (
        isinstance(eff_owner, ast.AST)
        and isinstance(mod_tree, ast.AST)
        and not _is_node_attached_to_tree(eff_owner, mod_tree)
    ):
        eff_owner = None
    enc_tuple = tuple(enclosing_scopes)
    return _FnScopeContext(
        fn_locals=frozenset(_extract_strict_fn_local_names(scope_node)),
        parent_map=eff_parent_map,
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
        scope_owner=eff_owner,
        mod_tree=mod_tree,
        enclosing_scopes=enc_tuple,
        precomputed_aux_nodes=_compute_scope_ctx_aux_nodes(
            eff_owner, mod_tree, enc_tuple, has_nonempty_pmap
        ),
        has_nonempty_parent_map=has_nonempty_pmap,
    )


def _get_fn_scope_context(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    *,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> _FnScopeContext:
    """Computes and caches lexical scope metadata for an AST scope node."""
    norm_enclosing = tuple(
        n
        for n in enclosing_scopes
        if isinstance(n, ast.AST) and n is not scope_node
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *((mod_tree,) if mod_tree is not None else ()),
        *norm_enclosing,
    )
    cache_key = _make_trusted_scope_cache_key(
        scope_node,
        "fn_scope_ctx",
        aux_ast_nodes=aux_nodes,
    )
    with analysis_session():
        return _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            cache_key,
            lambda: _get_fn_scope_context_uncached(
                scope_node,
                mod_tree=mod_tree,
                enclosing_scopes=norm_enclosing,
                scope_owner=scope_node,
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


def _scan_scope_nonlocal_mutations_and_deletions(
    scope_node: ast.AST,
    *,
    include_nested: bool = True,
    use_cache: bool = True,
    _depth: int = 0,
) -> Tuple[Set[str], Set[str]]:
    """Scans scope_node for (mutated_or_deleted, deleted) nonlocal variables."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        nonlocal_names: Set[str] = set()
        for sub in ast.walk(scope_node):
            if isinstance(sub, ast.Nonlocal):
                nonlocal_names.update(sub.names)
        return nonlocal_names, set(nonlocal_names)
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
    raw_body = getattr(scope_node, "body", None)
    stack: List[ast.AST] = (
        list(raw_body)
        if isinstance(raw_body, (list, tuple))
        else [raw_body]
        if isinstance(raw_body, ast.AST)
        else []
    )
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            direct_mutated.add(curr.name)
            if include_nested:
                sub_mut, sub_del = _extract_closure_nonlocal_mutations_and_deletions(
                    curr, use_cache=use_cache, _depth=_depth + 1
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
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            direct_mutated.add(curr.name)
            direct_deleted.add(curr.name)
        else:
            direct_mutated.update(_extract_non_name_bound_names(curr))
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
    _depth: int = 0,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (mutated_or_deleted, deleted) outer-scope variables via nonlocal in scope_node.

    Per PEP 227, class body scopes are not visible to nested methods and do not create
    an intermediate lexical barrier for nonlocal declarations in nested functions; thus
    own_locals is empty when scope_node is an ast.ClassDef.
    """
    muts, dels = _scan_scope_nonlocal_mutations_and_deletions(
        scope_node, include_nested=True, use_cache=use_cache, _depth=_depth
    )
    return frozenset(muts), frozenset(dels)


def _resolve_cached_str_set_pair(
    node: ast.AST,
    allowed_types: Tuple[type[ast.AST], ...],
    cache_key: _ScopeCacheKey,
    compute_fn: Callable[[bool], Tuple[frozenset[str], frozenset[str]]],
    *,
    use_cache: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Evaluates (or fetches from _SessionBucket.CELL_INFO) a pair of frozenset[str] and returns mutable sets."""
    if not isinstance(node, allowed_types):
        return set(), set()
    first, second = (
        _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO, cache_key, lambda: compute_fn(True)
        )
        if use_cache
        else compute_fn(False)
    )
    return set(first), set(second)


def _extract_closure_nonlocal_mutations_and_deletions(
    scope_node: ast.AST, *, use_cache: bool = True, _depth: int = 0
) -> Tuple[Set[str], Set[str]]:
    """Extracts and caches (mutated_or_deleted, deleted) nonlocal variables in scope_node."""
    return _resolve_cached_str_set_pair(
        scope_node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        _make_trusted_scope_cache_key(scope_node, "nl_muts_dels"),
        lambda c: _extract_closure_nonlocal_mutations_and_deletions_uncached(
            scope_node, use_cache=c, _depth=_depth
        ),
        use_cache=use_cache and _depth == 0,
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
            _SessionBucket.CELL_INFO,
            _make_trusted_scope_cache_key(node, "free_reads"),
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
    if isinstance(node, ast.Module):
        return _get_fn_scope_context_uncached(node, cache_parent_map=False)
    wrap = ast.Module(
        body=[node if isinstance(node, ast.stmt) else ast.Expr(value=cast(ast.expr, node))],
        type_ignores=[],
    )
    _mark_ephemeral_unit_ast(wrap)
    with _ephemeral_unit_ast_context(wrap):
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
    heuristic_ctx_managers: Tuple[str, ...] = ()
    wildcard_ctx_managers: Tuple[Tuple[str, str], ...] = ()
    mutation_edges: Tuple[Tuple[str, frozenset[str]], ...] = ()
    receiver_external_closures: Tuple[frozenset[str], frozenset[str]] = (
        frozenset(),
        frozenset(),
    )


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


def _extract_decorator_carrier_targets(node: ast.AST) -> Set[str]:
    """Extracts receiver root names and call-argument carriers from decorators on node."""
    targets: Set[str] = set()
    for dec in getattr(node, "decorator_list", ()):
        expr: ast.AST = dec
        while isinstance(expr, ast.Call):
            for arg in expr.args:
                targets.update(_extract_receiver_root_names(arg))
            for kw in expr.keywords:
                targets.update(_extract_receiver_root_names(kw.value))
            expr = expr.func
        targets.update(_extract_receiver_root_names(expr))
    return targets


def _extract_mutated_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names whose referenced objects are mutated by an assignment target."""
    return _collect_target_names(target, mode="mutated")
