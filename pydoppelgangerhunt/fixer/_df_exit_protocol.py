"""Module-wide detection of runtime context-manager exit-protocol patching.

A static proof that a context manager cannot suppress exceptions inspects the
``__exit__`` / ``__aexit__`` definitions visible at class-definition time. Python
resolves those special methods on ``type(manager)`` when the ``with`` statement
executes, so any later assignment such as ``Manager.__exit__ = swallow`` or
``setattr(Manager, "__exit__", swallow)`` (in any scope, including helper
functions invoked indirectly) invalidates that proof. This module scans a tree
for every such construct so provenance analysis can withdraw positive
non-suppression proofs instead of trusting a stale class body.
"""

from __future__ import annotations

import ast
from typing import List, NamedTuple, Optional, Set, Tuple, cast

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
)

__all__ = [
    "ExitProtocolPatchScan",
    "scan_exit_protocol_patches",
]

_EXIT_PROTOCOL_ATTRS: frozenset[str] = frozenset({"__exit__", "__aexit__", "__class__"})
_ATTR_MUTATION_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})
_ATTR_MUTATION_DUNDERS: frozenset[str] = frozenset({"__setattr__", "__delattr__"})
_DYNAMIC_CODE_BUILTINS: frozenset[str] = frozenset({"exec", "eval"})


class ExitProtocolPatchScan(NamedTuple):
    """Result of scanning a tree for runtime exit-protocol patching.

    Attributes:
        patched_names: Bare names whose ``__exit__`` / ``__aexit__`` / ``__class__``
            attribute is assigned, deleted, or targeted by ``setattr`` / ``delattr``.
        has_patch: True when any exit-protocol patch (targeted at a bare name or not)
            or dynamic code execution (``exec`` / ``eval``) occurs anywhere in the tree.
    """

    patched_names: frozenset[str]
    has_patch: bool

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()


_NO_PATCH_SCAN = ExitProtocolPatchScan(frozenset(), False)


def _attr_mutation_call_target(node: ast.Call) -> Optional[ast.AST]:
    """Returns the object expression whose attribute a setattr-style call may mutate, else None.

    Returns the receiver for ``setattr(obj, name, ...)`` / ``delattr(obj, name)``,
    ``builtins.setattr(obj, name, ...)``, and ``type.__setattr__(obj, name, ...)``
    when ``name`` is a non-literal or an exit-protocol attribute literal. Calls that
    provably target an unrelated literal attribute name return None. Calls with
    unresolvable receivers return the call itself so callers treat the patch as
    untargeted.
    """
    func = node.func
    if isinstance(func, ast.Name) and func.id in _ATTR_MUTATION_BUILTINS:
        target_idx = 0
    elif isinstance(func, ast.Attribute) and func.attr in _ATTR_MUTATION_BUILTINS:
        target_idx = 0
    elif isinstance(func, ast.Attribute) and func.attr in _ATTR_MUTATION_DUNDERS:
        target_idx = (
            0
            if isinstance(func.value, ast.Name) and func.value.id in ("type", "object")
            else -1
        )
    else:
        return None
    name_idx = target_idx + 1
    if any(isinstance(arg, ast.Starred) for arg in node.args) or node.keywords:
        return node
    if len(node.args) <= name_idx:
        return node
    name_arg = node.args[name_idx]
    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        if name_arg.value not in _EXIT_PROTOCOL_ATTRS:
            return None
    if target_idx < 0:
        return cast(ast.Attribute, func).value
    return node.args[target_idx]


_REF_TRIGGER_NAMES: frozenset[str] = _ATTR_MUTATION_BUILTINS | _DYNAMIC_CODE_BUILTINS
_REF_TRIGGER_ATTRS: frozenset[str] = (
    _ATTR_MUTATION_BUILTINS | _ATTR_MUTATION_DUNDERS | _DYNAMIC_CODE_BUILTINS
)


def _is_getattr_of_mutation_builtin(node: ast.Call) -> bool:
    """Returns True when node is getattr(..., '<attr_mutation_or_exec_builtin>')."""
    func = node.func
    is_getattr = (isinstance(func, ast.Name) and func.id == "getattr") or (
        isinstance(func, ast.Attribute) and func.attr == "getattr"
    )
    if not is_getattr or len(node.args) < 2:
        return False
    attr_arg = node.args[1]
    return (
        isinstance(attr_arg, ast.Constant)
        and isinstance(attr_arg.value, str)
        and attr_arg.value in _REF_TRIGGER_ATTRS
    )


def _scan_exit_protocol_patches_uncached(tree: ast.AST) -> ExitProtocolPatchScan:
    """Walks every scope of tree collecting exit-protocol patch targets."""
    patched: Set[str] = set()
    has_patch = False
    call_func_ids: Set[int] = set()
    stack: List[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call):
            call_func_ids.add(id(node.func))
        stack.extend(ast.iter_child_nodes(node))
        target: Optional[ast.AST] = None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr in _EXIT_PROTOCOL_ATTRS
        ):
            target = node.value
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                (isinstance(func, ast.Name) and func.id in _DYNAMIC_CODE_BUILTINS)
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr in _DYNAMIC_CODE_BUILTINS
                )
                or _is_getattr_of_mutation_builtin(node)
            ):
                has_patch = True
                continue
            target = _attr_mutation_call_target(node)
        elif id(node) not in call_func_ids:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in _REF_TRIGGER_NAMES
            ) or (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in _REF_TRIGGER_ATTRS
            ):
                has_patch = True
                continue
        if target is None:
            continue
        has_patch = True
        if isinstance(target, ast.Name):
            patched.add(target.id)
    if not has_patch:
        return _NO_PATCH_SCAN
    return ExitProtocolPatchScan(frozenset(patched), True)


def scan_exit_protocol_patches(tree: Optional[ast.AST]) -> ExitProtocolPatchScan:
    """Returns the cached exit-protocol patch scan for tree (empty scan when tree is None)."""
    if tree is None:
        return _NO_PATCH_SCAN
    if _is_ephemeral_unit_ast(tree):
        return _scan_exit_protocol_patches_uncached(tree)
    return cast(
        ExitProtocolPatchScan,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(tree, "exit_protocol_patches"),
            lambda: _scan_exit_protocol_patches_uncached(tree),
        ),
    )
