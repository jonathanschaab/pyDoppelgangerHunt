"""Finite syntactic guard for direct runtime context-manager exit-protocol patching.

# INVARIANT (STATIC TRUST BOUNDARY — CONVERGENCE):
# Single-file static AST analysis cannot decide arbitrary Python reflection,
# dynamic builtin aliasing (``f = setattr``), ``sys.modules`` / ``importlib`` /
# ``vars()`` / ``globals()`` / ``dict`` namespace mutation, ``type.__setattr__``
# indirection, metaclass hooks, or out-of-module monkey-patching. Attempting to
# chase indirect reflection via alias-tracking abstract interpretation does not
# converge in Python.
#
# Therefore this scanner is intentionally bounded to a single-pass O(|T|)
# syntactic guard covering ONLY:
#   1. Direct attribute assignment, augmented/annotated assignment, unpacking,
#      or deletion of ``__exit__``, ``__aexit__``, ``__class__``, or ``__bases__``.
#   2. Direct calls to ``exec(...)`` or ``eval(...)`` (bare or on ``builtins`` /
#      ``__builtins__``).
#   3. Direct calls to ``setattr(...)`` or ``delattr(...)`` (bare or on
#      ``builtins`` / ``__builtins__``) targeting exit-protocol attributes,
#      non-literal attribute names, or direct imported/stdlib module attributes.
#   4. Direct attribute assignment or deletion on imported modules or tracked
#      stdlib non-suppressing modules (e.g. ``contextlib.nullcontext = ...``,
#      ``builtins.open = ...``, ``mylib.Guard = ...``).
#
# Do NOT re-introduce alias graphs, fixed-point propagation, ``sys.modules`` /
# ``importlib`` / ``vars`` / ``globals`` / ``dict`` tracking, or dynamic-name
# reflection heuristics here. Callers that require fail-closed guarantees
# against untrusted context managers must use ``closure_strictness = "pedantic"``.
"""

from __future__ import annotations

import ast
from types import MappingProxyType
from typing import (
    AbstractSet,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _SessionBucket,
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
)
from pydoppelgangerhunt.fixer._df_visitors import _iter_target_leaves

__all__ = [
    "ExitProtocolPatchScan",
    "is_factory_rebound_target",
    "scan_exit_protocol_patches",
]

_EXIT_PROTOCOL_ATTRS: frozenset[str] = frozenset(
    {"__exit__", "__aexit__", "__class__", "__bases__"}
)
_STDLIB_NON_SUPPRESSING_MODULES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "builtins": frozenset({"open"}),
        "__builtins__": frozenset({"open"}),
        "io": frozenset(
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
            }
        ),
        "contextlib": frozenset(
            {
                "nullcontext",
                "closing",
                "aclosing",
                "redirect_stdout",
                "redirect_stderr",
                "chdir",
            }
        ),
        "tempfile": frozenset(
            {
                "TemporaryDirectory",
                "NamedTemporaryFile",
                "SpooledTemporaryFile",
                "TemporaryFile",
            }
        ),
        "warnings": frozenset({"catch_warnings"}),
        "threading": frozenset(
            {
                "Lock",
                "RLock",
                "Condition",
                "Semaphore",
                "BoundedSemaphore",
            }
        ),
        "asyncio": frozenset(
            {
                "Lock",
                "Condition",
                "Semaphore",
                "BoundedSemaphore",
            }
        ),
        "decimal": frozenset({"localcontext"}),
        "unittest.mock": frozenset({"patch"}),
        "mock": frozenset({"patch"}),
        "os": frozenset({"scandir"}),
        "mmap": frozenset({"mmap"}),
        "socket": frozenset({"socket", "create_connection"}),
        "sqlite3": frozenset({"connect"}),
        "zipfile": frozenset({"ZipFile"}),
        "tarfile": frozenset({"TarFile", "open"}),
        "gzip": frozenset({"open", "GzipFile"}),
        "bz2": frozenset({"open", "BZ2File"}),
        "lzma": frozenset({"open", "LZMAFile"}),
    }
)
_STDLIB_PACKAGE_MODS: frozenset[str] = frozenset(
    mod.rsplit(".", depth)[0]
    for mod in _STDLIB_NON_SUPPRESSING_MODULES
    if "." in mod
    for depth in range(1, mod.count(".") + 1)
)
_STDLIB_TRACKED_MOD_NAMES: frozenset[str] = (
    frozenset(_STDLIB_NON_SUPPRESSING_MODULES) | _STDLIB_PACKAGE_MODS
)
_BUILTIN_MODULE_NAMES: frozenset[str] = frozenset({"builtins", "__builtins__"})
_ATTR_MUTATION_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})
_ATTR_MUTATION_DUNDERS: frozenset[str] = frozenset({"__setattr__", "__delattr__"})
_UNBOUND_ATTR_DUNDER_ROOTS: frozenset[str] = frozenset({"type", "object"})
_DYNAMIC_CODE_BUILTINS: frozenset[str] = frozenset({"exec", "eval"})


class ExitProtocolPatchScan(NamedTuple):
    """Result of scanning a tree for direct runtime exit-protocol and factory patching.

    Attributes:
        patched_names: Bare names whose ``__exit__`` / ``__aexit__`` / ``__class__`` / ``__bases__``
            attribute is directly assigned, deleted, or targeted by ``setattr`` / ``delattr``.
        has_patch: True when any direct exit-protocol patch or direct ``exec`` / ``eval``
            call occurs anywhere in the tree.
        rebound_attrs: ``(target_name_or_None, attr_name_or_None)`` pairs for imported or
            stdlib context-manager factory attributes directly rebound or deleted.
    """

    patched_names: frozenset[str]
    has_patch: bool
    rebound_attrs: frozenset[Tuple[Optional[str], Optional[str]]] = frozenset()

    def is_factory_rebound(
        self,
        target_names: AbstractSet[str],
        attr_name: str,
        *,
        include_untargeted: bool = True,
    ) -> bool:
        """Returns True if any matching module/alias attribute or untargeted patch was seen."""
        return is_factory_rebound_target(
            self.rebound_attrs,
            target_names,
            attr_name,
            include_untargeted=include_untargeted,
        )


def is_factory_rebound_target(
    rebound_targets: AbstractSet[Tuple[Optional[str], Optional[str]]],
    target_names: AbstractSet[str],
    attr_name: str,
    *,
    include_untargeted: bool = True,
) -> bool:
    """Returns True if any (target, attr) pair in rebound_targets matches target_names and attr_name."""
    for t_name, a_name in rebound_targets:
        if t_name is None and a_name is None and not include_untargeted:
            continue
        if (t_name is None or t_name in target_names) and (
            a_name is None or a_name == attr_name
        ):
            return True
    return False


_NO_PATCH_SCAN = ExitProtocolPatchScan(frozenset(), False, frozenset())


def _iter_tree_iterative(root: ast.AST) -> Iterator[ast.AST]:
    """Iterates every AST node under root without recursion depth limits."""
    stack: List[ast.AST] = [root]
    while stack:
        node = stack.pop()
        yield node
        children = list(ast.iter_child_nodes(node))
        if children:
            stack.extend(reversed(children))


def _collect_imported_module_aliases(tree: ast.AST) -> Dict[str, Set[str]]:
    """Collects syntactic import aliases mapping local names to qualified module/symbol paths."""
    aliases: Dict[str, Set[str]] = {}
    for node in _iter_tree_iterative(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases.setdefault(alias.asname, set()).add(alias.name)
                else:
                    top_pkg = alias.name.split(".", 1)[0]
                    aliases.setdefault(top_pkg, set()).add(top_pkg)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                qual = f"{prefix}.{alias.name}" if node.module else f"{prefix}{alias.name}"
                aliases.setdefault(bound, set()).add(qual)
    return aliases


def _unwind_attr_target(
    expr: ast.AST,
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Unwinds a dotted Name.a.b expression into (root_name, attr_chain), or None."""
    parts: List[str] = []
    cur = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.reverse()
    return cur.id, tuple(parts)


def _record_module_attr_rebind(
    receiver: ast.AST,
    attr_name: Optional[str],
    import_aliases: Mapping[str, AbstractSet[str]],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> None:
    """Records a direct attribute rebind on an imported or tracked stdlib module receiver."""
    unwound = _unwind_attr_target(receiver)
    if unwound is None:
        return
    root_name, chain = unwound
    quals = import_aliases.get(root_name)
    is_tracked_builtin = root_name in _BUILTIN_MODULE_NAMES
    is_tracked_stdlib = root_name in _STDLIB_TRACKED_MOD_NAMES
    if quals is None and not is_tracked_stdlib:
        return
    resolved_roots: Set[str] = set(quals) if quals else set()
    if is_tracked_stdlib:
        resolved_roots.add(root_name)
    if is_tracked_builtin and not chain:
        rebound_attrs.add(("builtins", attr_name))
        rebound_attrs.add(("__builtins__", attr_name))
    for base_qual in resolved_roots:
        full_mod = ".".join((base_qual, *chain)) if chain else base_qual
        rebound_attrs.add((full_mod, attr_name))
    if not chain:
        rebound_attrs.add((root_name, attr_name))


def _direct_builtin_call_name(
    func: ast.AST,
    allowed_names: AbstractSet[str],
    import_aliases: Mapping[str, AbstractSet[str]],
) -> Optional[str]:
    """Returns the matched builtin name for direct bare or builtins.<name> call targets."""
    if isinstance(func, ast.Name) and func.id in allowed_names:
        return func.id
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.attr in allowed_names
    ):
        root = func.value.id
        if root in _BUILTIN_MODULE_NAMES or bool(
            import_aliases.get(root, frozenset()) & _BUILTIN_MODULE_NAMES
        ):
            return func.attr
    return None


def _extract_call_arg(node: ast.Call, pos: int, kw_name: str) -> Optional[ast.AST]:
    """Returns the expression passed at positional index pos or keyword kw_name."""
    if len(node.args) > pos and not any(
        isinstance(arg, ast.Starred) for arg in node.args[: pos + 1]
    ):
        return node.args[pos]
    for kw in node.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _handle_attr_target(
    attr_node: ast.Attribute,
    import_aliases: Mapping[str, AbstractSet[str]],
    patched: Set[str],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Inspects one assigned/deleted ast.Attribute target; returns True if exit-protocol patched."""
    _record_module_attr_rebind(
        attr_node.value, attr_node.attr, import_aliases, rebound_attrs
    )
    if attr_node.attr in _EXIT_PROTOCOL_ATTRS:
        if isinstance(attr_node.value, ast.Name):
            patched.add(attr_node.value.id)
        return True
    return False


def _record_setattr_effects(
    target_expr: Optional[ast.AST],
    attr_expr: Optional[ast.AST],
    has_starred_or_kwargs: bool,
    import_aliases: Mapping[str, AbstractSet[str]],
    patched: Set[str],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Applies syntactic effects of a direct setattr/delattr or __setattr__/__delattr__ call."""
    if isinstance(attr_expr, ast.Constant) and isinstance(attr_expr.value, str):
        attr_lit = attr_expr.value
        if target_expr is not None:
            _record_module_attr_rebind(
                target_expr, attr_lit, import_aliases, rebound_attrs
            )
        if attr_lit in _EXIT_PROTOCOL_ATTRS:
            if isinstance(target_expr, ast.Name):
                patched.add(target_expr.id)
            return True
        return False
    # Non-literal, starred, or missing attribute name: withdraw positive proofs.
    if isinstance(target_expr, ast.Name):
        patched.add(target_expr.id)
    if target_expr is not None:
        _record_module_attr_rebind(target_expr, None, import_aliases, rebound_attrs)
    if target_expr is None or has_starred_or_kwargs:
        rebound_attrs.add((None, None))
    return True


def _handle_setattr_delattr_call(
    call: ast.Call,
    import_aliases: Mapping[str, AbstractSet[str]],
    patched: Set[str],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Inspects a direct setattr/delattr call; returns True if it may patch the exit protocol."""
    has_starred_or_kwargs = any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    )
    return _record_setattr_effects(
        _extract_call_arg(call, 0, "object"),
        _extract_call_arg(call, 1, "name"),
        has_starred_or_kwargs,
        import_aliases,
        patched,
        rebound_attrs,
    )


def _handle_dunder_setattr_call(
    call: ast.Call,
    func: ast.Attribute,
    import_aliases: Mapping[str, AbstractSet[str]],
    patched: Set[str],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Inspects direct ``obj.__setattr__`` / ``type.__setattr__`` / ``object.__setattr__`` calls."""
    has_starred_or_kwargs = any(isinstance(a, ast.Starred) for a in call.args) or any(
        kw.arg is None for kw in call.keywords
    )
    is_unbound_builtin_type = (
        isinstance(func.value, ast.Name)
        and func.value.id in _UNBOUND_ATTR_DUNDER_ROOTS
    )
    if is_unbound_builtin_type:
        target_expr = _extract_call_arg(call, 0, "object")
        attr_expr = _extract_call_arg(call, 1, "name")
    else:
        target_expr = func.value
        attr_expr = _extract_call_arg(call, 0, "name")
    return _record_setattr_effects(
        target_expr,
        attr_expr,
        has_starred_or_kwargs,
        import_aliases,
        patched,
        rebound_attrs,
    )


def _scan_exit_protocol_patches_uncached(tree: ast.AST) -> ExitProtocolPatchScan:
    """Performs a single-pass syntactic scan of tree for direct exit-protocol and factory patches."""
    import_aliases = _collect_imported_module_aliases(tree)
    patched: Set[str] = set()
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]] = set()
    has_patch = False

    for node in _iter_tree_iterative(tree):
        targets: Tuple[ast.AST, ...] = ()
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = tuple(node.targets)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = (node.target,)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = tuple(
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            )
        elif isinstance(node, ast.Call):
            if _direct_builtin_call_name(
                node.func, _DYNAMIC_CODE_BUILTINS, import_aliases
            ):
                has_patch = True
                rebound_attrs.add((None, None))
            elif _direct_builtin_call_name(
                node.func, _ATTR_MUTATION_BUILTINS, import_aliases
            ):
                if _handle_setattr_delattr_call(
                    node, import_aliases, patched, rebound_attrs
                ):
                    has_patch = True
            elif (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _ATTR_MUTATION_DUNDERS
            ):
                if _handle_dunder_setattr_call(
                    node, node.func, import_aliases, patched, rebound_attrs
                ):
                    has_patch = True
            continue
        else:
            continue

        for tgt in targets:
            for leaf in _iter_target_leaves(tgt):
                if isinstance(leaf, ast.Attribute) and _handle_attr_target(
                    leaf, import_aliases, patched, rebound_attrs
                ):
                    has_patch = True

    if not has_patch and not rebound_attrs and not patched:
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
    return _get_or_compute_scope_cached(
        _SessionBucket.CELL_INFO,
        _make_trusted_scope_cache_key(tree, "exit_protocol_patches"),
        lambda: _scan_exit_protocol_patches_uncached(tree),
    )
