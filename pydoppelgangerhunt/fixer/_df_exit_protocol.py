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
from typing import AbstractSet, List, NamedTuple, Optional, Sequence, Set, Tuple, cast

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
)

__all__ = [
    "ExitProtocolPatchScan",
    "is_factory_rebound_target",
    "scan_exit_protocol_patches",
]

_EXIT_PROTOCOL_ATTRS: frozenset[str] = frozenset({"__exit__", "__aexit__", "__class__"})
_STDLIB_FACTORY_ATTRS: frozenset[str] = frozenset(
    {
        "open",
        "StringIO",
        "BytesIO",
        "TextIOWrapper",
        "BufferedReader",
        "BufferedWriter",
        "BufferedRandom",
        "BufferedRWPair",
        "FileIO",
        "nullcontext",
        "suppress",
        "closing",
        "aclosing",
        "redirect_stdout",
        "redirect_stderr",
        "chdir",
        "TemporaryDirectory",
        "NamedTemporaryFile",
        "SpooledTemporaryFile",
        "TemporaryFile",
        "catch_warnings",
        "Lock",
        "RLock",
        "Condition",
        "Semaphore",
        "BoundedSemaphore",
        "localcontext",
        "patch",
        "scandir",
        "mmap",
        "socket",
        "create_connection",
        "connect",
        "ZipFile",
        "TarFile",
        "GzipFile",
        "BZ2File",
        "LZMAFile",
    }
)
_ATTR_MUTATION_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})
_ATTR_MUTATION_DUNDERS: frozenset[str] = frozenset({"__setattr__", "__delattr__"})
_DYNAMIC_CODE_BUILTINS: frozenset[str] = frozenset({"exec", "eval"})


class ExitProtocolPatchScan(NamedTuple):
    """Result of scanning a tree for runtime exit-protocol and stdlib-factory patching.

    Attributes:
        patched_names: Bare names whose ``__exit__`` / ``__aexit__`` / ``__class__``
            attribute is assigned, deleted, or targeted by ``setattr`` / ``delattr``.
        has_patch: True when any exit-protocol patch (targeted at a bare name or not)
            or dynamic code execution (``exec`` / ``eval``) occurs anywhere in the tree.
        rebound_attrs: ``(target_name_or_None, attr_name_or_None)`` pairs for stdlib
            context-manager factory attributes rebound or deleted anywhere in the tree.
    """

    patched_names: frozenset[str]
    has_patch: bool
    rebound_attrs: frozenset[Tuple[Optional[str], Optional[str]]] = frozenset()

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()

    def is_factory_rebound(
        self, target_names: AbstractSet[str], attr_name: str
    ) -> bool:
        """Returns True if any matching module/alias attribute or untargeted dynamic patch was seen."""
        return is_factory_rebound_target(self.rebound_attrs, target_names, attr_name)


def is_factory_rebound_target(
    rebound_targets: AbstractSet[Tuple[Optional[str], Optional[str]]],
    target_names: AbstractSet[str],
    attr_name: str,
) -> bool:
    """Returns True if any (target, attr) pair in rebound_targets matches target_names and attr_name."""
    for t_name, a_name in rebound_targets:
        if (t_name is None or t_name in target_names) and (
            a_name is None or a_name == attr_name
        ):
            return True
    return False


_NO_PATCH_SCAN = ExitProtocolPatchScan(frozenset(), False, frozenset())


class _BuiltinAliasTable(NamedTuple):
    exec_eval: Set[str]
    setattr_delattr: Set[str]
    getattr_fns: Set[str]
    builtins_mods: Set[str]
    import_fns: Set[str]
    importlib_mods: Set[str]
    sys_mods: Set[str]
    has_wildcard: bool


def _propagate_alias_step(
    assign_pairs: Sequence[Tuple[str, ast.AST]],
    table: _BuiltinAliasTable,
) -> bool:
    """Propagates one fixed-point iteration across assignment pairs; returns True if changed."""
    changed = False
    attr_targets: Tuple[Tuple[AbstractSet[str], str, Set[str]], ...] = (
        (table.builtins_mods, "__import__", table.import_fns),
        (table.builtins_mods, "getattr", table.getattr_fns),
        (table.importlib_mods, "import_module", table.import_fns),
    )
    name_sets: Tuple[Set[str], ...] = (
        table.exec_eval,
        table.setattr_delattr,
        table.getattr_fns,
        table.builtins_mods,
        table.import_fns,
        table.importlib_mods,
        table.sys_mods,
    )
    for tgt_name, val in assign_pairs:
        if isinstance(val, ast.Name):
            for alias_set in name_sets:
                if val.id in alias_set and tgt_name not in alias_set:
                    alias_set.add(tgt_name)
                    changed = True
        elif isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
            if val.value.id in table.builtins_mods:
                if val.attr in _DYNAMIC_CODE_BUILTINS and tgt_name not in table.exec_eval:
                    table.exec_eval.add(tgt_name)
                    changed = True
                elif (
                    val.attr in _ATTR_MUTATION_BUILTINS
                    and tgt_name not in table.setattr_delattr
                ):
                    table.setattr_delattr.add(tgt_name)
                    changed = True
            for mod_set, attr_name, dest_set in attr_targets:
                if (
                    val.value.id in mod_set
                    and val.attr == attr_name
                    and tgt_name not in dest_set
                ):
                    dest_set.add(tgt_name)
                    changed = True
    return changed


def _collect_builtin_aliases(tree: ast.AST) -> _BuiltinAliasTable:
    """Collects imported and assigned aliases for builtins, dynamic import helpers, and sys."""
    table = _BuiltinAliasTable(
        exec_eval=set(_DYNAMIC_CODE_BUILTINS),
        setattr_delattr=set(_ATTR_MUTATION_BUILTINS),
        getattr_fns={"getattr"},
        builtins_mods={"builtins", "__builtins__"},
        import_fns={"__import__"},
        importlib_mods={"importlib"},
        sys_mods={"sys"},
        has_wildcard=False,
    )
    has_wildcard = False
    assign_pairs: List[Tuple[str, ast.AST]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name in ("builtins", "__builtins__"):
                    table.builtins_mods.add(bound)
                elif alias.name == "importlib":
                    table.importlib_mods.add(bound)
                elif alias.name == "sys":
                    table.sys_mods.add(bound)
        elif isinstance(node, ast.ImportFrom) and node.module in (
            "builtins",
            "__builtins__",
            "importlib",
        ):
            for alias in node.names:
                if alias.name == "*":
                    has_wildcard = True
                    continue
                bound = alias.asname or alias.name
                if node.module == "importlib":
                    if alias.name == "import_module":
                        table.import_fns.add(bound)
                elif alias.name in _DYNAMIC_CODE_BUILTINS:
                    table.exec_eval.add(bound)
                elif alias.name in _ATTR_MUTATION_BUILTINS:
                    table.setattr_delattr.add(bound)
                elif alias.name == "getattr":
                    table.getattr_fns.add(bound)
                elif alias.name == "__import__":
                    table.import_fns.add(bound)
        elif isinstance(node, ast.Assign) and node.value is not None:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    assign_pairs.append((tgt.id, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                assign_pairs.append((node.target.id, node.value))

    while _propagate_alias_step(assign_pairs, table):
        pass

    return table._replace(has_wildcard=has_wildcard)


def _is_builtins_attr(
    node: ast.AST, builtins_mod_aliases: AbstractSet[str], allowed_attrs: AbstractSet[str]
) -> bool:
    """Returns True when node is ``<builtins_or_alias>.<attr>`` for an attr in allowed_attrs."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in builtins_mod_aliases
        and node.attr in allowed_attrs
    )


def _attr_mutation_call_info(
    node: ast.Call,
    setattr_delattr_aliases: AbstractSet[str],
    builtins_mod_aliases: AbstractSet[str],
) -> Optional[Tuple[ast.AST, Optional[str]]]:
    """Returns ``(target_expr, literal_attr_name_or_None)`` for a setattr/delattr call, else None.

    Handles direct calls, ``builtins.setattr``, imported/assigned aliases, and
    ``type.__setattr__`` / ``object.__setattr__``. Returns None when the call
    provably targets a literal attribute outside ``_EXIT_PROTOCOL_ATTRS | _STDLIB_FACTORY_ATTRS``.
    """
    func = node.func
    if isinstance(func, ast.Name) and func.id in setattr_delattr_aliases:
        target_idx = 0
    elif _is_builtins_attr(func, builtins_mod_aliases, _ATTR_MUTATION_BUILTINS):
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
        return node, None
    if len(node.args) <= name_idx:
        return node, None
    name_arg = node.args[name_idx]
    literal_attr: Optional[str] = None
    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        literal_attr = name_arg.value
        if (
            literal_attr not in _EXIT_PROTOCOL_ATTRS
            and literal_attr not in _STDLIB_FACTORY_ATTRS
        ):
            return None
    target_expr = (
        cast(ast.Attribute, func).value if target_idx < 0 else node.args[target_idx]
    )
    return target_expr, literal_attr


_REF_TRIGGER_ATTRS: frozenset[str] = (
    _ATTR_MUTATION_BUILTINS | _ATTR_MUTATION_DUNDERS | _DYNAMIC_CODE_BUILTINS
)


def _is_getattr_of_mutation_builtin(
    node: ast.Call,
    aliases: _BuiltinAliasTable,
) -> bool:
    """Returns True when node is getattr(..., '<mutation>') or getattr(<builtins>, dynamic_attr)."""
    func = node.func
    is_getattr = (isinstance(func, ast.Name) and func.id in aliases.getattr_fns) or _is_builtins_attr(
        func, aliases.builtins_mods, frozenset({"getattr"})
    )
    if not is_getattr or len(node.args) < 2:
        return False
    attr_arg = node.args[1]
    if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
        return attr_arg.value in _REF_TRIGGER_ATTRS
    return isinstance(node.args[0], ast.Name) and node.args[0].id in aliases.builtins_mods


def _is_dynamic_builtins_lookup(
    node: ast.AST,
    aliases: _BuiltinAliasTable,
) -> bool:
    """Returns True for __import__('builtins'), vars(builtins), builtins.__dict__, sys.modules['builtins'], or __builtins__[...] mutation lookups."""
    if isinstance(node, ast.Call):
        func = node.func
        is_import_call = (isinstance(func, ast.Name) and func.id in aliases.import_fns) or (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and (
                (func.value.id in aliases.importlib_mods and func.attr == "import_module")
                or (func.value.id in aliases.builtins_mods and func.attr == "__import__")
            )
        )
        if (
            is_import_call
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "builtins"
        ):
            return True
        if (
            isinstance(func, ast.Name)
            and func.id == "vars"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in aliases.builtins_mods
        ):
            return True
    elif isinstance(node, ast.Subscript):
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in aliases.sys_mods
            and node.value.attr == "modules"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "builtins"
        ):
            return True
        if isinstance(node.value, ast.Name) and node.value.id in aliases.builtins_mods:
            return not isinstance(node.slice, ast.Constant) or (
                node.slice.value in _REF_TRIGGER_ATTRS
            )
    elif (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases.builtins_mods
        and node.attr == "__dict__"
    ):
        return True
    return False


def _scan_exit_protocol_patches_uncached(tree: ast.AST) -> ExitProtocolPatchScan:
    """Walks every scope of tree collecting exit-protocol and stdlib-factory patch targets."""
    aliases = _collect_builtin_aliases(tree)
    ref_trigger_names = aliases.exec_eval | aliases.setattr_delattr
    patched: Set[str] = set()
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]] = set()
    has_patch = aliases.has_wildcard
    if aliases.has_wildcard:
        rebound_attrs.add((None, None))
    call_func_ids: Set[int] = set()
    stack: List[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call):
            call_func_ids.add(id(node.func))
        stack.extend(ast.iter_child_nodes(node))
        if _is_dynamic_builtins_lookup(node, aliases):
            has_patch = True
            rebound_attrs.add((None, None))
            continue
        if isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if node.attr in _EXIT_PROTOCOL_ATTRS:
                has_patch = True
                if isinstance(node.value, ast.Name):
                    patched.add(node.value.id)
            elif node.attr in _STDLIB_FACTORY_ATTRS and isinstance(node.value, ast.Name):
                rebound_attrs.add((node.value.id, node.attr))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                (isinstance(func, ast.Name) and func.id in aliases.exec_eval)
                or _is_builtins_attr(func, aliases.builtins_mods, _DYNAMIC_CODE_BUILTINS)
                or _is_getattr_of_mutation_builtin(node, aliases)
            ):
                has_patch = True
                rebound_attrs.add((None, None))
                continue
            mut_info = _attr_mutation_call_info(
                node, aliases.setattr_delattr, aliases.builtins_mods
            )
            if mut_info is None:
                continue
            target_expr, literal_attr = mut_info
            if literal_attr is not None and literal_attr in _STDLIB_FACTORY_ATTRS:
                rebound_attrs.add(
                    (
                        target_expr.id if isinstance(target_expr, ast.Name) else None,
                        literal_attr,
                    )
                )
            else:
                has_patch = True
                if isinstance(target_expr, ast.Name):
                    patched.add(target_expr.id)
                    if literal_attr is None:
                        rebound_attrs.add((target_expr.id, None))
                else:
                    rebound_attrs.add((None, None))
        elif id(node) not in call_func_ids:
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in ref_trigger_names
            ) or (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and (
                    _is_builtins_attr(node, aliases.builtins_mods, _REF_TRIGGER_ATTRS)
                    or (
                        isinstance(node.value, ast.Name)
                        and node.value.id in ("type", "object")
                        and node.attr in _ATTR_MUTATION_DUNDERS
                    )
                )
            ):
                has_patch = True
                rebound_attrs.add((None, None))
    if not has_patch and not rebound_attrs:
        return _NO_PATCH_SCAN
    return ExitProtocolPatchScan(
        frozenset(patched), has_patch, frozenset(rebound_attrs)
    )


def scan_exit_protocol_patches(tree: Optional[ast.AST]) -> ExitProtocolPatchScan:
    """Returns the cached exit-protocol patch scan for tree (empty scan when tree is None)."""
    if tree is None:
        return _NO_PATCH_SCAN
    if not isinstance(tree, ast.Module) or _is_ephemeral_unit_ast(tree):
        return _scan_exit_protocol_patches_uncached(tree)
    return cast(
        ExitProtocolPatchScan,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(tree, "exit_protocol_patches"),
            lambda: _scan_exit_protocol_patches_uncached(tree),
        ),
    )
