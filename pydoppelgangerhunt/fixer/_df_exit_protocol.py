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
from types import MappingProxyType
from typing import (
    AbstractSet,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
)
from pydoppelgangerhunt.fixer._df_closures import _extract_single_node_local_names
from pydoppelgangerhunt.fixer._df_span_index import _extract_type_alias_or_pattern_name
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

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
_STDLIB_FACTORY_ATTRS: frozenset[str] = frozenset().union(
    {"suppress"}, *_STDLIB_NON_SUPPRESSING_MODULES.values()
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
_ATTR_MUTATION_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})
_ATTR_MUTATION_DUNDERS: frozenset[str] = frozenset({"__setattr__", "__delattr__"})
_DYNAMIC_CODE_BUILTINS: frozenset[str] = frozenset({"exec", "eval"})
_DICT_MUTATION_METHODS: frozenset[str] = frozenset(
    {"update", "pop", "setdefault", "popitem", "clear"}
)


class ExitProtocolPatchScan(NamedTuple):
    """Result of scanning a tree for runtime exit-protocol and stdlib-factory patching.

    Attributes:
        patched_names: Bare names whose ``__exit__`` / ``__aexit__`` / ``__class__`` / ``__bases__``
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
        self,
        target_names: AbstractSet[str],
        attr_name: str,
        *,
        include_untargeted: bool = True,
    ) -> bool:
        """Returns True if any matching module/alias attribute or untargeted dynamic patch was seen."""
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


_Slot = Tuple[int, str]


class _SlotAliasTable(NamedTuple):
    graph: _CollectedScopeGraph
    exec_eval: Set[_Slot]
    setattr_delattr: Set[_Slot]
    getattr_fns: Set[_Slot]
    vars_fns: Set[_Slot]
    globals_fns: Set[_Slot]
    namespace_dicts: Set[_Slot]
    builtins_mods: Set[_Slot]
    import_fns: Set[_Slot]
    import_module_fns: Set[_Slot]
    importlib_mods: Set[_Slot]
    sys_mods: Set[_Slot]
    sys_modules_dicts: Set[_Slot]
    type_object_refs: Set[_Slot]
    stdlib_mods: Dict[_Slot, Set[str]]
    bound_dunders: Dict[_Slot, Set[Tuple[Optional[str], Optional[str]]]]
    written_slots: AbstractSet[_Slot]
    non_stdlib_import_slots: AbstractSet[_Slot]


def _unwrap_named_expr(node: ast.AST) -> ast.AST:
    """Unwraps outer walrus (``:=``) expressions to expose the underlying value AST node."""
    while isinstance(node, ast.NamedExpr):
        node = node.value
    return node


def _extract_call_arg(node: ast.Call, pos: int, kw_name: str) -> Optional[ast.AST]:
    """Returns the AST expression passed at positional index pos or keyword kw_name."""
    if len(node.args) > pos and not any(
        isinstance(arg, ast.Starred) for arg in node.args[: pos + 1]
    ):
        return node.args[pos]
    for kw in node.keywords:
        if kw.arg == kw_name:
            return kw.value
    return None


def _resolve_expr_name_slots(node: ast.Name, table: _SlotAliasTable) -> Tuple[_Slot, ...]:
    """Returns candidate defining slots visible when evaluating ast.Name node."""
    eval_sid = table.graph.node_sids.get(id(node), table.graph.root_sid)
    lex_slot = _resolve_lexical_slot(
        table.graph.scopes, table.graph.root_sid, eval_sid, node.id
    )
    sc = table.graph.scopes[eval_sid]
    if (
        sc.kind == "class"
        and node.id not in sc.globals_decl
        and node.id not in sc.nonlocals_decl
    ):
        class_slot = (eval_sid, node.id)
        if class_slot != lex_slot:
            return (lex_slot, class_slot)
    return (lex_slot,)


def _is_name_in_slots(
    node: ast.AST,
    slot_set: AbstractSet[_Slot],
    table: _SlotAliasTable,
) -> bool:
    """Returns True when node is an ast.Name whose resolved slot is in slot_set."""
    if not isinstance(node, ast.Name):
        return False
    return any(slot in slot_set for slot in _resolve_expr_name_slots(node, table))


def _visible_builtins_mod_names(node: ast.AST, table: _SlotAliasTable) -> Set[str]:
    """Returns builtins-module identifier names visible from node's lexical scope chain."""
    anc_sids: Set[int] = {_BUILTIN_SID, table.graph.root_sid}
    curr: Optional[int] = table.graph.node_sids.get(id(node), table.graph.root_sid)
    while curr is not None:
        anc_sids.add(curr)
        curr = table.graph.scopes[curr].parent_lexical_sid
    return {name for sid, name in table.builtins_mods if sid in anc_sids}


def _import_call_kind(node: ast.Call, table: _SlotAliasTable) -> Optional[str]:
    """Returns 'import_module' or 'dunder_import' if node invokes importlib.import_module or __import__."""
    func = _unwrap_named_expr(node.func)
    if _is_name_in_slots(func, table.import_module_fns, table):
        return "import_module"
    if _is_name_in_slots(func, table.import_fns, table):
        return "dunder_import"
    if isinstance(func, ast.Attribute):
        if (
            _is_name_in_slots(func.value, table.importlib_mods, table)
            and func.attr == "import_module"
        ):
            return "import_module"
        if (
            _is_name_in_slots(func.value, table.builtins_mods, table)
            and func.attr == "__import__"
        ):
            return "dunder_import"
    return None


def _is_sys_modules_expr(node: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True when node is sys.modules (via sys, imported modules, or a local alias)."""
    unwrapped = _unwrap_named_expr(node)
    if _is_name_in_slots(unwrapped, table.sys_modules_dicts, table):
        return True
    if (
        isinstance(unwrapped, ast.Attribute)
        and unwrapped.attr == "modules"
        and _is_name_in_slots(unwrapped.value, table.sys_mods, table)
    ):
        return True
    getattr_call = _is_getattr_call(unwrapped, table)
    return (
        getattr_call is not None
        and len(getattr_call.args) >= 2
        and not getattr_call.keywords
        and not any(isinstance(arg, ast.Starred) for arg in getattr_call.args)
        and _is_name_in_slots(getattr_call.args[0], table.sys_mods, table)
        and isinstance(getattr_call.args[1], ast.Constant)
        and getattr_call.args[1].value == "modules"
    )


def _name_stdlib_mods(node: ast.Name, table: _SlotAliasTable) -> Set[str]:
    """Returns canonical tracked stdlib module names bound to ast.Name node."""
    eval_sid = table.graph.node_sids.get(id(node), table.graph.root_sid)
    sc = table.graph.scopes[eval_sid]
    lex_slot = _resolve_lexical_slot(
        table.graph.scopes, table.graph.root_sid, eval_sid, node.id
    )
    mods: Set[str] = set(table.stdlib_mods.get(lex_slot, ()))
    has_class_override = (
        sc.kind == "class"
        and node.id not in sc.globals_decl
        and node.id not in sc.nonlocals_decl
    )
    class_slot = (eval_sid, node.id)
    if has_class_override:
        mods.update(table.stdlib_mods.get(class_slot, ()))
    if (
        not mods
        and lex_slot[0] in (_BUILTIN_SID, table.graph.root_sid)
        and lex_slot[1] in _STDLIB_TRACKED_MOD_NAMES
        and lex_slot not in table.written_slots
        and (not has_class_override or class_slot not in table.written_slots)
    ):
        mods.add(lex_slot[1])
    return mods


def _call_stdlib_mods(call: ast.Call, table: _SlotAliasTable) -> Set[str]:
    """Returns canonical stdlib module names resolved from import_module/__import__/sys.modules.get."""
    imp_kind = _import_call_kind(call, table)
    if imp_kind is not None:
        mod_arg = _extract_call_arg(call, 0, "name")
        if isinstance(mod_arg, ast.Constant) and isinstance(mod_arg.value, str):
            target_mod = mod_arg.value
            if imp_kind == "dunder_import":
                fromlist = _extract_call_arg(call, 3, "fromlist")
                has_fromlist = isinstance(
                    fromlist, (ast.List, ast.Tuple, ast.Set)
                ) and bool(fromlist.elts)
                if not has_fromlist:
                    target_mod = target_mod.split(".", 1)[0]
            if target_mod in _STDLIB_TRACKED_MOD_NAMES:
                return {target_mod}
    elif (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and _is_sys_modules_expr(call.func.value, table)
    ):
        key_arg = _extract_call_arg(call, 0, "key")
        if (
            isinstance(key_arg, ast.Constant)
            and isinstance(key_arg.value, str)
            and key_arg.value in _STDLIB_TRACKED_MOD_NAMES
        ):
            return {key_arg.value}
    return set()


def _resolve_expr_stdlib_mods(node: ast.AST, table: _SlotAliasTable) -> Set[str]:
    """Iteratively resolves an expression or dotted module chain to tracked stdlib module names."""
    parts: List[str] = []
    curr = _unwrap_named_expr(node)
    while True:
        if isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = _unwrap_named_expr(curr.value)
            continue
        getattr_call = _is_getattr_call(curr, table)
        if (
            getattr_call is not None
            and len(getattr_call.args) >= 2
            and not getattr_call.keywords
            and not any(isinstance(arg, ast.Starred) for arg in getattr_call.args)
            and isinstance(getattr_call.args[1], ast.Constant)
            and isinstance(getattr_call.args[1].value, str)
        ):
            parts.append(getattr_call.args[1].value)
            curr = _unwrap_named_expr(getattr_call.args[0])
            continue
        break
    if isinstance(curr, ast.Name):
        mods = _name_stdlib_mods(curr, table)
    elif isinstance(curr, ast.Call):
        mods = _call_stdlib_mods(curr, table)
    elif (
        isinstance(curr, ast.Subscript)
        and _is_sys_modules_expr(curr.value, table)
        and isinstance(curr.slice, ast.Constant)
        and isinstance(curr.slice.value, str)
        and curr.slice.value in _STDLIB_TRACKED_MOD_NAMES
    ):
        mods = {curr.slice.value}
    else:
        return set()
    for attr in reversed(parts):
        if not mods:
            return set()
        mods = {
            f"{m}.{attr}"
            for m in mods
            if f"{m}.{attr}" in _STDLIB_TRACKED_MOD_NAMES
        }
    return mods


def _is_namespace_dict_call(node: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True if node calls globals() (or alias) or 0-arg vars() (or alias)."""
    if not isinstance(node, ast.Call):
        return False
    func = _unwrap_named_expr(node.func)
    is_glob = _is_name_in_slots(func, table.globals_fns, table) or _is_builtins_attr(
        func, table, frozenset({"globals"})
    )
    if is_glob:
        return True
    is_zero_arg_vars = not node.args and not node.keywords and (
        _is_name_in_slots(func, table.vars_fns, table)
        or _is_builtins_attr(func, table, frozenset({"vars"}))
    )
    return is_zero_arg_vars


def _is_builtins_namespace_expr(node: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True when node dynamically evaluates to the builtins module or its __dict__."""
    expr = _unwrap_named_expr(node)
    if isinstance(expr, ast.Call):
        func = _unwrap_named_expr(expr.func)
        if _import_call_kind(expr, table) is not None:
            mod_arg = _extract_call_arg(expr, 0, "name")
            if isinstance(mod_arg, ast.Constant) and mod_arg.value in (
                "builtins",
                "__builtins__",
            ):
                return True
        is_vars_call = _is_name_in_slots(
            func, table.vars_fns, table
        ) or _is_builtins_attr(func, table, frozenset({"vars"}))
        if (
            is_vars_call
            and expr.args
            and _is_name_in_slots(expr.args[0], table.builtins_mods, table)
        ):
            return True
    elif isinstance(expr, ast.Attribute):
        return (
            _is_name_in_slots(expr.value, table.builtins_mods, table)
            and expr.attr == "__dict__"
        )
    extracted = _extract_container_and_key(expr)
    if extracted is not None:
        container, key_expr = extracted
        if isinstance(key_expr, ast.Constant) and (
            (
                _is_sys_modules_expr(container, table)
                and key_expr.value in ("builtins", "__builtins__")
            )
            or (
                _is_namespace_dict_expr(container, table)
                and key_expr.value
                in ("__builtins__", *_visible_builtins_mod_names(expr, table))
            )
        ):
            return True
    return False


def _is_namespace_dict_expr(node: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True for globals() / 0-arg vars() calls or variables aliasing them."""
    return _is_name_in_slots(
        node, table.namespace_dicts, table
    ) or _is_namespace_dict_call(node, table)


class _AttrMutationCallee(NamedTuple):
    target_arg_idx: int
    bound_target: Optional[ast.AST]


class _ScopeRecord(NamedTuple):
    sid: int
    kind: str
    parent_lexical_sid: Optional[int]
    parent_non_comp_sid: int
    local_names: Set[str]
    globals_decl: Set[str]
    nonlocals_decl: Set[str]


class _AliasCandidate(NamedTuple):
    target_owner_sid: int
    target_name: str
    rhs_node: ast.AST
    rhs_eval_sid: int
    is_direct_call: bool = False


class _BuiltinBindRecord(NamedTuple):
    owner_sid: int
    name: str
    alias_kind: str
    is_pure_setattr_import: bool


class _CollectedScopeGraph(NamedTuple):
    root_sid: int
    scopes: Dict[int, _ScopeRecord]
    node_sids: Dict[int, int]
    nodes: List[ast.AST]
    call_func_ids: Set[int]
    wildcard_sids: Set[int]
    alias_candidates: List[_AliasCandidate]
    non_alias_writes: List[Tuple[int, str, bool]]
    builtin_binds: List[_BuiltinBindRecord]
    stdlib_binds: List[Tuple[int, str, str]]
    other_import_binds: List[Tuple[int, str]]
    load_records: List[Tuple[ast.Name, int]]
    unpack_candidates: List[Tuple[int, ast.AST, ast.AST]]


def _is_getattr_call(node: ast.AST, table: _SlotAliasTable) -> Optional[ast.Call]:
    """Returns node when it calls getattr (or an alias), else None."""
    if not isinstance(node, ast.Call):
        return None
    func = _unwrap_named_expr(node.func)
    if _is_name_in_slots(func, table.getattr_fns, table) or _is_builtins_attr(
        func, table, frozenset({"getattr"})
    ):
        return node
    return None


def _extract_container_and_key(node: ast.AST) -> Optional[Tuple[ast.AST, ast.AST]]:
    """Extracts (container_expr, key_expr) from subscript or .get(...) lookups."""
    if isinstance(node, ast.Subscript):
        return node.value, node.slice
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ):
        key_arg = _extract_call_arg(node, 0, "key")
        if key_arg is not None:
            return node.func.value, key_arg
    return None


def _is_container_key_mutation_ref(node: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True for builtins/namespace[...] or .get(...) indexing setattr or delattr."""
    extracted = _extract_container_and_key(node)
    if extracted is None:
        return False
    container, key_expr = extracted
    if not (
        isinstance(key_expr, ast.Constant)
        and isinstance(key_expr.value, str)
        and key_expr.value in _ATTR_MUTATION_BUILTINS
    ):
        return False
    return (
        _is_name_in_slots(container, table.builtins_mods, table)
        or _is_namespace_dict_expr(container, table)
        or _is_builtins_namespace_expr(container, table)
    )


def _is_type_call_or_ref(expr: ast.AST, table: _SlotAliasTable) -> bool:
    """Returns True for type/object refs or type(...) calls whose __setattr__/__delattr__ is unbound."""
    unwrapped = _unwrap_named_expr(expr)
    if _is_name_in_slots(
        unwrapped, table.type_object_refs, table
    ) or _is_builtins_attr(unwrapped, table, frozenset({"type", "object"})):
        return True
    if isinstance(unwrapped, ast.Call) and unwrapped.args:
        callee = _unwrap_named_expr(unwrapped.func)
        if _is_builtins_attr(callee, table, frozenset({"type"})):
            return True
        if _is_name_in_slots(callee, table.type_object_refs, table):
            if isinstance(callee, ast.Name) and all(
                slot == (_BUILTIN_SID, "object")
                for slot in _resolve_expr_name_slots(callee, table)
            ):
                return False
            return True
    return False


def _resolve_getattr_mutation_callee(
    getattr_call: ast.Call,
    table: _SlotAliasTable,
    *,
    allow_bound_dunder: bool,
) -> Optional[_AttrMutationCallee]:
    """Resolves ``getattr(target, 'setattr'|'delattr'|'__setattr__'|'__delattr__')``."""
    if (
        any(isinstance(a, ast.Starred) for a in getattr_call.args)
        or getattr_call.keywords
        or len(getattr_call.args) < 2
    ):
        return None
    attr_arg = getattr_call.args[1]
    if not (isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str)):
        return None
    attr_name = attr_arg.value
    if attr_name in _ATTR_MUTATION_BUILTINS:
        return _AttrMutationCallee(0, None)
    if attr_name in _ATTR_MUTATION_DUNDERS:
        recv = getattr_call.args[0]
        if _is_type_call_or_ref(recv, table):
            return _AttrMutationCallee(0, None)
        if allow_bound_dunder:
            return _AttrMutationCallee(-1, recv)
    return None


def _resolve_attr_mutation_callee(
    func: ast.AST,
    table: _SlotAliasTable,
    *,
    allow_bound_dunder: bool = True,
) -> Optional[_AttrMutationCallee]:
    """Resolves a direct, aliased, subscript, getattr, or dunder setattr/delattr callee."""
    expr = _unwrap_named_expr(func)
    result: Optional[_AttrMutationCallee] = None
    if (
        _is_name_in_slots(expr, table.setattr_delattr, table)
        or _is_builtins_attr(expr, table, _ATTR_MUTATION_BUILTINS)
        or (
            isinstance(expr, ast.Attribute)
            and expr.attr in _ATTR_MUTATION_BUILTINS
            and _is_builtins_namespace_expr(expr.value, table)
        )
        or _is_container_key_mutation_ref(expr, table)
    ):
        result = _AttrMutationCallee(0, None)
    else:
        getattr_call = _is_getattr_call(expr, table)
        if getattr_call is not None:
            result = _resolve_getattr_mutation_callee(
                getattr_call, table, allow_bound_dunder=allow_bound_dunder
            )
        elif isinstance(expr, ast.Attribute) and expr.attr in _ATTR_MUTATION_DUNDERS:
            if _is_type_call_or_ref(expr.value, table):
                result = _AttrMutationCallee(0, None)
            elif allow_bound_dunder:
                result = _AttrMutationCallee(-1, expr.value)
    return result


def _propagate_attr_alias(
    tgt_slot: _Slot,
    val: ast.Attribute,
    table: _SlotAliasTable,
) -> bool:
    """Propagates attribute-load alias candidates (e.g. builtins.eval, importlib.import_module)."""
    attr_rules: Tuple[Tuple[Set[_Slot], AbstractSet[str], Set[_Slot]], ...] = (
        (table.builtins_mods, _DYNAMIC_CODE_BUILTINS, table.exec_eval),
        (table.builtins_mods, _ATTR_MUTATION_BUILTINS, table.setattr_delattr),
        (table.builtins_mods, frozenset({"__import__"}), table.import_fns),
        (table.builtins_mods, frozenset({"getattr"}), table.getattr_fns),
        (table.builtins_mods, frozenset({"vars"}), table.vars_fns),
        (table.builtins_mods, frozenset({"globals"}), table.globals_fns),
        (table.builtins_mods, frozenset({"type", "object"}), table.type_object_refs),
        (table.importlib_mods, frozenset({"import_module"}), table.import_module_fns),
    )
    changed = False
    for mod_set, allowed_attrs, dest_set in attr_rules:
        if (
            _is_name_in_slots(val.value, mod_set, table)
            and val.attr in allowed_attrs
            and tgt_slot not in dest_set
        ):
            dest_set.add(tgt_slot)
            changed = True
    return changed


def _propagate_slot_alias_step(table: _SlotAliasTable) -> bool:
    """Propagates one fixed-point iteration across lexical alias candidates; returns True if changed."""
    changed = False
    slot_sets: Tuple[Set[_Slot], ...] = (
        table.exec_eval,
        table.setattr_delattr,
        table.getattr_fns,
        table.vars_fns,
        table.globals_fns,
        table.namespace_dicts,
        table.builtins_mods,
        table.import_fns,
        table.import_module_fns,
        table.importlib_mods,
        table.sys_mods,
        table.sys_modules_dicts,
        table.type_object_refs,
    )
    scopes = table.graph.scopes
    root_sid = table.graph.root_sid
    for cand in table.graph.alias_candidates:
        tgt_slot = _resolve_target_slot(
            scopes, root_sid, cand.target_owner_sid, cand.target_name
        )
        val = _unwrap_named_expr(cand.rhs_node)
        if isinstance(val, ast.Name):
            for slot_set in slot_sets:
                if _is_name_in_slots(val, slot_set, table) and tgt_slot not in slot_set:
                    slot_set.add(tgt_slot)
                    changed = True
            for src_slot in _resolve_expr_name_slots(val, table):
                if src_slot in table.bound_dunders:
                    dest_recvs = table.bound_dunders.setdefault(tgt_slot, set())
                    before_recvs = len(dest_recvs)
                    dest_recvs.update(table.bound_dunders[src_slot])
                    if len(dest_recvs) > before_recvs:
                        changed = True
        elif isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
            if _propagate_attr_alias(tgt_slot, val, table):
                changed = True
        if _is_namespace_dict_expr(val, table) and tgt_slot not in table.namespace_dicts:
            table.namespace_dicts.add(tgt_slot)
            changed = True
        if _is_builtins_namespace_expr(val, table) and tgt_slot not in table.builtins_mods:
            table.builtins_mods.add(tgt_slot)
            changed = True
        if _is_sys_modules_expr(val, table) and tgt_slot not in table.sys_modules_dicts:
            table.sys_modules_dicts.add(tgt_slot)
            changed = True
        mut_callee = _resolve_attr_mutation_callee(val, table, allow_bound_dunder=True)
        if mut_callee is not None:
            if mut_callee.target_arg_idx == 0:
                if tgt_slot not in table.setattr_delattr:
                    table.setattr_delattr.add(tgt_slot)
                    changed = True
            else:
                bt = mut_callee.bound_target
                recv_name: Optional[str] = (
                    bt.id if isinstance(bt, ast.Name) else None
                )
                rebound_targets: Tuple[Optional[str], ...]
                if bt is None:
                    rebound_targets = (None,)
                else:
                    bt_mods = _resolve_expr_stdlib_mods(bt, table)
                    if bt_mods:
                        rebound_targets = tuple(sorted(bt_mods))
                    elif isinstance(bt, ast.Name):
                        is_ext_imp = any(
                            slot in table.non_stdlib_import_slots
                            for slot in _resolve_expr_name_slots(bt, table)
                        )
                        rebound_targets = (bt.id,) if is_ext_imp else ("",)
                    else:
                        rebound_targets = (None,)
                dest_recvs = table.bound_dunders.setdefault(tgt_slot, set())
                for r_mod in rebound_targets:
                    pair = (recv_name, r_mod)
                    if pair not in dest_recvs:
                        dest_recvs.add(pair)
                        changed = True
        rhs_stdlib_mods = _resolve_expr_stdlib_mods(val, table)
        if rhs_stdlib_mods:
            dest_mods = table.stdlib_mods.setdefault(tgt_slot, set())
            before_count = len(dest_mods)
            dest_mods.update(rhs_stdlib_mods)
            if len(dest_mods) > before_count:
                changed = True
    return changed


def _expand_unpack_candidates(graph: _CollectedScopeGraph) -> None:
    """Expands Tuple/List unpacking over literal container aliases into alias_candidates."""
    if not graph.unpack_candidates:
        return
    root = graph.root_sid
    seq_slots: Dict[_Slot, ast.AST] = {}
    for cand in graph.alias_candidates:
        unwrapped = _unwrap_named_expr(cand.rhs_node)
        if isinstance(unwrapped, (ast.Tuple, ast.List)) and not any(
            isinstance(el, ast.Starred) for el in unwrapped.elts
        ):
            tgt_slot = _resolve_target_slot(
                graph.scopes, root, cand.target_owner_sid, cand.target_name
            )
            seq_slots[tgt_slot] = unwrapped
    if not seq_slots:
        return
    for eval_sid, tgt, rhs in graph.unpack_candidates:
        unwrapped_rhs = _unwrap_named_expr(rhs)
        if not isinstance(unwrapped_rhs, ast.Name):
            continue
        rhs_slot = _resolve_lexical_slot(graph.scopes, root, eval_sid, unwrapped_rhs.id)
        seq_expr = seq_slots.get(rhs_slot)
        if seq_expr is None:
            continue
        extra_pairs: List[Tuple[str, ast.AST]] = []
        _extract_unpack_alias_pairs(tgt, seq_expr, extra_pairs, [])
        for name, val_expr in extra_pairs:
            graph.alias_candidates.append(_AliasCandidate(eval_sid, name, val_expr, eval_sid))


def _collect_builtin_slot_aliases(tree: ast.AST) -> _SlotAliasTable:
    """Builds the scope graph and computes fixed-point lexical slot aliases for builtins."""
    graph = _collect_scope_graph(tree)
    _expand_unpack_candidates(graph)
    root = graph.root_sid
    written_slots: Set[_Slot] = {
        (
            _resolve_target_slot(graph.scopes, root, sid, name)
            if resolve_lex
            else (sid, name)
        )
        for sid, name, resolve_lex in graph.non_alias_writes
    }
    for cand in graph.alias_candidates:
        written_slots.add(
            _resolve_target_slot(graph.scopes, root, cand.target_owner_sid, cand.target_name)
        )
    non_stdlib_import_slots: Set[_Slot] = {
        _resolve_target_slot(graph.scopes, root, sid, name)
        for sid, name in graph.other_import_binds
    }
    table = _SlotAliasTable(
        graph=graph,
        exec_eval={(_BUILTIN_SID, "exec"), (_BUILTIN_SID, "eval")},
        setattr_delattr={(_BUILTIN_SID, "setattr"), (_BUILTIN_SID, "delattr")},
        getattr_fns={(_BUILTIN_SID, "getattr")},
        vars_fns={(_BUILTIN_SID, "vars")},
        globals_fns={(_BUILTIN_SID, "globals")},
        namespace_dicts=set(),
        builtins_mods={(_BUILTIN_SID, "builtins"), (_BUILTIN_SID, "__builtins__")},
        import_fns={(_BUILTIN_SID, "__import__")},
        import_module_fns=set(),
        importlib_mods={(_BUILTIN_SID, "importlib")},
        sys_mods={(_BUILTIN_SID, "sys")},
        sys_modules_dicts=set(),
        type_object_refs={(_BUILTIN_SID, "type"), (_BUILTIN_SID, "object")},
        stdlib_mods={
            (_BUILTIN_SID, "builtins"): {"builtins"},
            (_BUILTIN_SID, "__builtins__"): {"__builtins__"},
        },
        bound_dunders={},
        written_slots=written_slots,
        non_stdlib_import_slots=non_stdlib_import_slots,
    )
    kind_map: Dict[str, Set[_Slot]] = {
        "exec_eval": table.exec_eval,
        "setattr_delattr": table.setattr_delattr,
        "getattr_fns": table.getattr_fns,
        "vars_fns": table.vars_fns,
        "globals_fns": table.globals_fns,
        "builtins_mods": table.builtins_mods,
        "import_fns": table.import_fns,
        "import_module_fns": table.import_module_fns,
        "importlib_mods": table.importlib_mods,
        "sys_mods": table.sys_mods,
        "sys_modules_dicts": table.sys_modules_dicts,
        "type_object_refs": table.type_object_refs,
    }
    for b in graph.builtin_binds:
        b_slot = _resolve_target_slot(graph.scopes, root, b.owner_sid, b.name)
        kind_map[b.alias_kind].add(b_slot)
    for sid, name, mod_name in graph.stdlib_binds:
        s_slot = _resolve_target_slot(graph.scopes, root, sid, name)
        table.stdlib_mods.setdefault(s_slot, set()).add(mod_name)
    while _propagate_slot_alias_step(table):
        pass
    return table


def _is_builtins_attr(
    node: ast.AST, table: _SlotAliasTable, allowed_attrs: AbstractSet[str]
) -> bool:
    """Returns True when node is ``<builtins_or_alias>.<attr>`` in allowed_attrs."""
    return (
        isinstance(node, ast.Attribute)
        and _is_name_in_slots(node.value, table.builtins_mods, table)
        and node.attr in allowed_attrs
    )


def _attr_mutation_call_info(
    node: ast.Call,
    table: _SlotAliasTable,
) -> Optional[Tuple[ast.AST, Optional[str]]]:
    """Returns ``(target_expr, literal_attr_name_or_None)`` for a setattr/delattr call, else None.

    Handles direct calls, ``builtins.setattr``, imported/assigned/walrus aliases,
    indirect ``getattr(...)`` / ``builtins[...]`` / ``vars(builtins)[...]`` lookups,
    and ``type.__setattr__`` / ``object.__setattr__``. Returns None when the call
    provably targets a literal attribute outside ``_EXIT_PROTOCOL_ATTRS | _STDLIB_FACTORY_ATTRS``.
    """
    callee = _resolve_attr_mutation_callee(node.func, table)
    if callee is None:
        return None
    target_idx = callee.target_arg_idx
    name_idx = target_idx + 1
    target_expr = (
        callee.bound_target
        if callee.bound_target is not None
        else (
            node.args[target_idx]
            if 0 <= target_idx < len(node.args)
            and not any(
                isinstance(arg, ast.Starred) for arg in node.args[: target_idx + 1]
            )
            else node
        )
    )
    if any(isinstance(arg, ast.Starred) for arg in node.args) or node.keywords:
        return target_expr, None
    if len(node.args) <= name_idx:
        return target_expr, None
    name_arg = node.args[name_idx]
    literal_attr: Optional[str] = None
    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        literal_attr = name_arg.value
        if (
            literal_attr not in _EXIT_PROTOCOL_ATTRS
            and literal_attr not in _STDLIB_FACTORY_ATTRS
        ):
            return None
    return target_expr, literal_attr


_REF_TRIGGER_ATTRS: frozenset[str] = (
    _ATTR_MUTATION_BUILTINS | _ATTR_MUTATION_DUNDERS | _DYNAMIC_CODE_BUILTINS
)


_BUILTIN_NAMESPACE_ATTRS: frozenset[str] = frozenset({"vars", "globals", "__import__"})


def _is_getattr_of_mutation_builtin(
    node: ast.Call,
    table: _SlotAliasTable,
) -> bool:
    """Returns True when node is getattr(..., '<mutation>'), getattr(<builtins>, dynamic_attr), or starred getattr."""
    getattr_call = _is_getattr_call(node, table)
    if getattr_call is None or not getattr_call.args:
        return False
    if isinstance(getattr_call.args[0], ast.Starred):
        return True
    if len(getattr_call.args) < 2:
        return False
    if isinstance(getattr_call.args[1], ast.Starred):
        return True
    target_arg = _unwrap_named_expr(getattr_call.args[0])
    target_is_builtins = _is_name_in_slots(
        target_arg, table.builtins_mods, table
    ) or _is_builtins_namespace_expr(target_arg, table)
    attr_arg = _unwrap_named_expr(getattr_call.args[1])
    if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
        return attr_arg.value in _REF_TRIGGER_ATTRS or (
            target_is_builtins and attr_arg.value in _BUILTIN_NAMESPACE_ATTRS
        )
    return target_is_builtins


def _is_dynamic_builtins_lookup(
    node: ast.AST,
    table: _SlotAliasTable,
) -> bool:
    """Returns True for dynamic builtins, vars(builtins), globals()['__builtins__'], sys.modules['builtins'], or __builtins__[...] mutation lookups."""
    if _is_builtins_namespace_expr(node, table):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and _is_namespace_dict_expr(func.value, table)
        ):
            key_arg = _extract_call_arg(node, 0, "key")
            return key_arg is not None and (
                not isinstance(key_arg, ast.Constant)
                or key_arg.value in _REF_TRIGGER_ATTRS
                or key_arg.value in _BUILTIN_NAMESPACE_ATTRS
                or key_arg.value in _visible_builtins_mod_names(node, table)
            )
    elif isinstance(node, ast.Subscript) and (
        _is_name_in_slots(node.value, table.builtins_mods, table)
        or _is_namespace_dict_expr(node.value, table)
    ):
        is_direct_builtins_slice = (
            _is_name_in_slots(node.value, table.builtins_mods, table)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "open"
        )
        return (
            not isinstance(node.slice, ast.Constant)
            or node.slice.value in _REF_TRIGGER_ATTRS
            or node.slice.value in _BUILTIN_NAMESPACE_ATTRS
            or is_direct_builtins_slice
        )
    return False


_COMPREHENSION_TYPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_SCOPE_CONTAINER_TYPES = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    *_COMPREHENSION_TYPES,
)
_BUILTIN_SID: int = -2
_PURE_DIRECT_ROOT: _Slot = (-3, "")
_FROM_BUILTIN_KIND: Mapping[Tuple[str, str], Tuple[str, bool]] = {
    ("importlib", "import_module"): ("import_module_fns", False),
    ("sys", "modules"): ("sys_modules_dicts", False),
    ("builtins", "exec"): ("exec_eval", False),
    ("builtins", "eval"): ("exec_eval", False),
    ("__builtins__", "exec"): ("exec_eval", False),
    ("__builtins__", "eval"): ("exec_eval", False),
    ("builtins", "setattr"): ("setattr_delattr", True),
    ("builtins", "delattr"): ("setattr_delattr", True),
    ("__builtins__", "setattr"): ("setattr_delattr", True),
    ("__builtins__", "delattr"): ("setattr_delattr", True),
    ("builtins", "getattr"): ("getattr_fns", False),
    ("__builtins__", "getattr"): ("getattr_fns", False),
    ("builtins", "vars"): ("vars_fns", False),
    ("__builtins__", "vars"): ("vars_fns", False),
    ("builtins", "globals"): ("globals_fns", False),
    ("__builtins__", "globals"): ("globals_fns", False),
    ("builtins", "__import__"): ("import_fns", False),
    ("__builtins__", "__import__"): ("import_fns", False),
    ("builtins", "type"): ("type_object_refs", False),
    ("builtins", "object"): ("type_object_refs", False),
    ("__builtins__", "type"): ("type_object_refs", False),
    ("__builtins__", "object"): ("type_object_refs", False),
}
_IMPORT_MOD_KIND: Mapping[str, str] = {
    "builtins": "builtins_mods",
    "__builtins__": "builtins_mods",
    "importlib": "importlib_mods",
    "sys": "sys_mods",
}


def _extract_param_names(args: ast.arguments) -> List[str]:
    """Extracts parameter identifiers from an ast.arguments node."""
    params = [
        arg.arg
        for arg in (
            *getattr(args, "posonlyargs", ()),
            *args.args,
            *args.kwonlyargs,
        )
    ]
    if args.vararg is not None:
        params.append(args.vararg.arg)
    if args.kwarg is not None:
        params.append(args.kwarg.arg)
    return params


def _extract_bound_names_in_target(target: ast.AST) -> List[str]:
    """Extracts stored or deleted bare identifier names in an assignment/deletion target."""
    return [
        sub.id
        for sub in ast.walk(target)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del))
    ]


def _split_scope_children(node: ast.AST) -> Tuple[List[ast.AST], List[ast.AST]]:
    """Splits a scope node's children into (enclosing_scope_nodes, inner_scope_nodes)."""
    if isinstance(node, ast.Module):
        return [], list(node.body)
    if isinstance(node, _COMPREHENSION_TYPES):
        gens = node.generators
        enc: List[ast.AST] = [gens[0].iter] if gens else []
        own: List[ast.AST] = [
            ch for ch in ast.iter_child_nodes(node) if not isinstance(ch, ast.comprehension)
        ]
        for idx, gen in enumerate(gens):
            if idx > 0:
                own.append(gen.iter)
            own.append(gen.target)
            own.extend(gen.ifs)
        return enc, own
    enc = iter_nested_scope_header_nodes(node, include_annotations=True)
    enc.extend(getattr(node, "type_params", ()))
    raw_body = getattr(node, "body", ())
    if isinstance(raw_body, ast.AST):
        return enc, [raw_body]
    return enc, list(raw_body)


def _resolve_global_or_builtin_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    name: str,
    *,
    allow_builtin_fallback: bool,
) -> _Slot:
    """Resolves a module-global lookup or store to root_sid or _BUILTIN_SID."""
    if allow_builtin_fallback and name not in scopes[root_sid].local_names:
        return _BUILTIN_SID, name
    return root_sid, name


def _resolve_decl_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    sc: _ScopeRecord,
    name: str,
    *,
    allow_builtin_fallback: bool,
) -> Optional[_Slot]:
    """Resolves explicit global or nonlocal declarations for name in sc, else None."""
    if name in sc.globals_decl:
        return _resolve_global_or_builtin_slot(
            scopes, root_sid, name, allow_builtin_fallback=allow_builtin_fallback
        )
    if name in sc.nonlocals_decl:
        curr = sc.parent_lexical_sid
        while curr is not None and curr != root_sid:
            curr_sc = scopes[curr]
            if name in curr_sc.globals_decl:
                return _resolve_global_or_builtin_slot(
                    scopes, root_sid, name, allow_builtin_fallback=allow_builtin_fallback
                )
            if name in curr_sc.local_names:
                return curr, name
            curr = curr_sc.parent_lexical_sid
        return _resolve_global_or_builtin_slot(
            scopes, root_sid, name, allow_builtin_fallback=allow_builtin_fallback
        )
    return None


def _resolve_lexical_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    scope_sid: int,
    name: str,
) -> _Slot:
    """Resolves a name load in scope_sid to its defining (scope_id, name) lexical slot."""
    sc = scopes[scope_sid]
    decl_slot = _resolve_decl_slot(
        scopes, root_sid, sc, name, allow_builtin_fallback=True
    )
    if decl_slot is not None:
        return decl_slot
    if name in sc.local_names:
        return scope_sid, name
    curr = sc.parent_lexical_sid
    while curr is not None:
        curr_sc = scopes[curr]
        if name in curr_sc.globals_decl:
            return _resolve_global_or_builtin_slot(
                scopes, root_sid, name, allow_builtin_fallback=True
            )
        if name in curr_sc.local_names:
            return curr, name
        curr = curr_sc.parent_lexical_sid
    return _BUILTIN_SID, name


def _resolve_target_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    scope_sid: int,
    name: str,
) -> _Slot:
    """Resolves a binding target in scope_sid to its owning (scope_id, name) slot."""
    sc = scopes[scope_sid]
    decl_slot = _resolve_decl_slot(
        scopes, root_sid, sc, name, allow_builtin_fallback=False
    )
    if decl_slot is not None:
        return decl_slot
    return scope_sid, name


def _make_scope_record(
    node: ast.AST,
    sid: int,
    parent_lex_sid: Optional[int],
    parent_non_comp_sid: int,
) -> _ScopeRecord:
    """Creates a _ScopeRecord for node with precomputed module/class/function/comprehension locals."""
    if isinstance(node, ast.Module):
        kind = "module"
    elif isinstance(node, ast.ClassDef):
        kind = "class"
    elif isinstance(node, _COMPREHENSION_TYPES):
        kind = "comp"
    else:
        kind = "function"
    if kind == "module":
        local_names = (
            _extract_single_node_local_names(
                node,
                include_globals=True,
                include_nonlocals=True,
                include_self_name=False,
                include_bare_annotations=False,
            )
            if isinstance(node, ast.Module)
            else set()
        )
    else:
        local_names = _extract_single_node_local_names(
            node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=kind in ("function", "comp"),
        )
    return _ScopeRecord(
        sid=sid,
        kind=kind,
        parent_lexical_sid=parent_lex_sid,
        parent_non_comp_sid=parent_non_comp_sid,
        local_names=local_names,
        globals_decl=set(),
        nonlocals_decl=set(),
    )


def _mark_function_local_name(sc: _ScopeRecord, name: str) -> None:
    """Adds name to function local_names unless explicitly declared global or nonlocal."""
    if (
        sc.kind == "function"
        and name not in sc.globals_decl
        and name not in sc.nonlocals_decl
    ):
        sc.local_names.add(name)


def _record_import_in_scope(
    curr: Union[ast.Import, ast.ImportFrom],
    curr_sid: int,
    graph: _CollectedScopeGraph,
) -> None:
    """Records imported bindings as builtin/stdlib binds or non-alias writes and marks function locals."""
    sc = graph.scopes[curr_sid]
    if isinstance(curr, ast.Import):
        for alias in curr.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            _mark_function_local_name(sc, bound)
            kind = _IMPORT_MOD_KIND.get(alias.name)
            if kind is not None:
                graph.builtin_binds.append(_BuiltinBindRecord(curr_sid, bound, kind, False))
            else:
                graph.non_alias_writes.append((curr_sid, bound, True))
            target_mod = (
                alias.name if alias.asname is not None else alias.name.split(".", 1)[0]
            )
            if target_mod in _STDLIB_TRACKED_MOD_NAMES:
                graph.stdlib_binds.append((curr_sid, bound, target_mod))
            elif kind is None:
                graph.other_import_binds.append((curr_sid, bound))
        return
    mod = curr.module if getattr(curr, "level", 0) == 0 else None
    for alias in curr.names:
        if alias.name == "*":
            if mod in ("builtins", "__builtins__", "importlib"):
                graph.wildcard_sids.add(curr_sid)
            continue
        bound = alias.asname or alias.name
        _mark_function_local_name(sc, bound)
        entry = _FROM_BUILTIN_KIND.get((mod, alias.name)) if mod is not None else None
        if entry is not None:
            graph.builtin_binds.append(
                _BuiltinBindRecord(curr_sid, bound, entry[0], entry[1])
            )
        else:
            graph.non_alias_writes.append((curr_sid, bound, True))
            qual_mod = f"{mod}.{alias.name}" if mod else None
            if qual_mod is not None and qual_mod in _STDLIB_TRACKED_MOD_NAMES:
                graph.stdlib_binds.append((curr_sid, bound, qual_mod))
            else:
                graph.other_import_binds.append((curr_sid, bound))


def _extract_unpack_alias_pairs(
    tgt: ast.AST,
    rhs: ast.AST,
    alias_pairs: List[Tuple[str, ast.AST]],
    non_alias_names: List[str],
) -> None:
    """Pairs matching Tuple/List unpacking targets elementwise with RHS expressions."""
    if isinstance(tgt, ast.Name):
        alias_pairs.append((tgt.id, rhs))
        return
    unwrapped = _unwrap_named_expr(rhs)
    if (
        isinstance(tgt, (ast.Tuple, ast.List))
        and isinstance(unwrapped, (ast.Tuple, ast.List))
        and not any(isinstance(el, ast.Starred) for el in unwrapped.elts)
    ):
        star_indices = [
            idx for idx, el in enumerate(tgt.elts) if isinstance(el, ast.Starred)
        ]
        if not star_indices and len(tgt.elts) == len(unwrapped.elts):
            for t_el, r_el in zip(tgt.elts, unwrapped.elts):
                _extract_unpack_alias_pairs(t_el, r_el, alias_pairs, non_alias_names)
            return
        if len(star_indices) == 1 and len(unwrapped.elts) >= len(tgt.elts) - 1:
            star_idx = star_indices[0]
            for idx in range(star_idx):
                _extract_unpack_alias_pairs(
                    tgt.elts[idx], unwrapped.elts[idx], alias_pairs, non_alias_names
                )
            non_alias_names.extend(_extract_bound_names_in_target(tgt.elts[star_idx]))
            trailing = len(tgt.elts) - 1 - star_idx
            for offset in range(trailing):
                t_el = tgt.elts[star_idx + 1 + offset]
                r_el = unwrapped.elts[len(unwrapped.elts) - trailing + offset]
                _extract_unpack_alias_pairs(t_el, r_el, alias_pairs, non_alias_names)
            return
    non_alias_names.extend(_extract_bound_names_in_target(tgt))


def _record_assign_candidate(
    curr: ast.AST,
    curr_sid: int,
    graph: _CollectedScopeGraph,
) -> bool:
    """Records Assign, AnnAssign, and NamedExpr bindings; returns True if curr matched."""
    sc_kind = graph.scopes[curr_sid].kind
    if isinstance(curr, ast.Assign):
        if (
            curr.value is not None
            and curr.targets
            and sc_kind in ("module", "function", "class")
        ):
            alias_pairs: List[Tuple[str, ast.AST]] = []
            non_alias_names: List[str] = []
            for tgt in curr.targets:
                if isinstance(tgt, (ast.Tuple, ast.List)) and not isinstance(
                    _unwrap_named_expr(curr.value), (ast.Tuple, ast.List)
                ):
                    graph.unpack_candidates.append((curr_sid, tgt, curr.value))
                _extract_unpack_alias_pairs(
                    tgt, curr.value, alias_pairs, non_alias_names
                )
            for name, val_expr in alias_pairs:
                graph.alias_candidates.append(
                    _AliasCandidate(curr_sid, name, val_expr, curr_sid)
                )
            for name in non_alias_names:
                graph.non_alias_writes.append((curr_sid, name, True))
        else:
            for tgt in curr.targets:
                for name in _extract_bound_names_in_target(tgt):
                    graph.non_alias_writes.append((curr_sid, name, True))
        return True
    if isinstance(curr, ast.AnnAssign):
        if (
            curr.value is not None
            and isinstance(curr.target, ast.Name)
            and sc_kind in ("module", "function", "class")
        ):
            graph.alias_candidates.append(
                _AliasCandidate(curr_sid, curr.target.id, curr.value, curr_sid)
            )
        else:
            for name in _extract_bound_names_in_target(curr.target):
                graph.non_alias_writes.append((curr_sid, name, True))
        return True
    if isinstance(curr, ast.NamedExpr):
        owner_sid = (
            curr_sid
            if sc_kind in ("module", "function", "class")
            else graph.scopes[curr_sid].parent_non_comp_sid
        )
        owner_sc = graph.scopes[owner_sid]
        _mark_function_local_name(owner_sc, curr.target.id)
        is_direct_call = id(curr) in graph.call_func_ids
        graph.alias_candidates.append(
            _AliasCandidate(
                owner_sid,
                curr.target.id,
                curr.value,
                curr_sid,
                is_direct_call=is_direct_call,
            )
        )
        if not is_direct_call:
            graph.non_alias_writes.append((owner_sid, curr.target.id, True))
        return True
    return False


def _record_non_assign_node(
    curr: ast.AST,
    curr_sid: int,
    graph: _CollectedScopeGraph,
) -> None:
    """Records loads, scope declarations, imports, and non-assignment bindings."""
    if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
        graph.load_records.append((curr, curr_sid))
    elif isinstance(curr, ast.Global):
        graph.scopes[curr_sid].globals_decl.update(curr.names)
    elif isinstance(curr, ast.Nonlocal):
        graph.scopes[curr_sid].nonlocals_decl.update(curr.names)
    elif isinstance(curr, (ast.Import, ast.ImportFrom)):
        _record_import_in_scope(curr, curr_sid, graph)
    elif isinstance(curr, (ast.AugAssign, ast.For, ast.AsyncFor)):
        for name in _extract_bound_names_in_target(curr.target):
            graph.non_alias_writes.append((curr_sid, name, True))
    elif isinstance(curr, ast.Delete):
        for tgt in curr.targets:
            for name in _extract_bound_names_in_target(tgt):
                graph.non_alias_writes.append((curr_sid, name, True))
    elif isinstance(curr, (ast.With, ast.AsyncWith)):
        for item in curr.items:
            if item.optional_vars is not None:
                for name in _extract_bound_names_in_target(item.optional_vars):
                    graph.non_alias_writes.append((curr_sid, name, True))
    elif isinstance(curr, ast.ExceptHandler):
        if isinstance(curr.name, str):
            graph.non_alias_writes.append((curr_sid, curr.name, True))
    else:
        bound = _extract_type_alias_or_pattern_name(curr)
        if bound is not None and bound != "_":
            graph.non_alias_writes.append((curr_sid, bound, True))


def _collect_scope_graph(tree: ast.AST) -> _CollectedScopeGraph:
    """Iteratively collects lexical scopes, bindings, load nodes, and call targets across tree."""
    is_mod = isinstance(tree, ast.Module)
    root_sid = id(tree) if is_mod else -1
    root_sc = (
        _make_scope_record(tree, root_sid, None, root_sid)
        if is_mod
        else _ScopeRecord(root_sid, "module", None, root_sid, set(), set(), set())
    )
    graph = _CollectedScopeGraph(
        root_sid=root_sid,
        scopes={root_sid: root_sc},
        node_sids={},
        nodes=[],
        call_func_ids=set(),
        wildcard_sids=set(),
        alias_candidates=[],
        non_alias_writes=[],
        builtin_binds=[],
        stdlib_binds=[],
        other_import_binds=[],
        load_records=[],
        unpack_candidates=[],
    )
    stack: List[Tuple[ast.AST, int]] = (
        [(ch, root_sid) for ch in reversed(tree.body)]
        if isinstance(tree, ast.Module)
        else [(tree, root_sid)]
    )
    while stack:
        curr, curr_sid = stack.pop()
        graph.node_sids[id(curr)] = curr_sid
        graph.nodes.append(curr)
        if isinstance(curr, _SCOPE_CONTAINER_TYPES) and not isinstance(curr, ast.Module):
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                graph.non_alias_writes.append((curr_sid, curr.name, True))
            cur_sc = graph.scopes[curr_sid]
            lex_parent = cur_sc.sid if cur_sc.kind != "class" else cur_sc.parent_lexical_sid
            non_comp_parent = (
                cur_sc.sid
                if cur_sc.kind in ("module", "function", "class")
                else cur_sc.parent_non_comp_sid
            )
            child_sid = id(curr)
            child_sc = _make_scope_record(curr, child_sid, lex_parent, non_comp_parent)
            graph.scopes[child_sid] = child_sc
            if child_sc.kind == "function":
                for p_name in _extract_param_names(cast(ast.FunctionDef, curr).args):
                    graph.non_alias_writes.append((child_sid, p_name, False))
            elif child_sc.kind == "comp":
                for gen in cast(ast.ListComp, curr).generators:
                    for name in _extract_bound_names_in_target(gen.target):
                        graph.non_alias_writes.append((child_sid, name, False))
            enc, own = _split_scope_children(curr)
            stack.extend((ch, child_sid) for ch in reversed(own))
            stack.extend((ch, curr_sid) for ch in reversed(enc))
            continue
        if isinstance(curr, ast.Call):
            graph.call_func_ids.add(id(curr.func))
        if isinstance(curr, (ast.Import, ast.ImportFrom)):
            _record_import_in_scope(curr, curr_sid, graph)
            continue
        if not _record_assign_candidate(curr, curr_sid, graph):
            _record_non_assign_node(curr, curr_sid, graph)
        stack.extend((ch, curr_sid) for ch in reversed(list(ast.iter_child_nodes(curr))))
    seeded_builtins = (
        _DYNAMIC_CODE_BUILTINS
        | _ATTR_MUTATION_BUILTINS
        | frozenset(
            {
                "getattr",
                "vars",
                "globals",
                "__import__",
                "builtins",
                "__builtins__",
                "importlib",
                "sys",
                "type",
                "object",
            }
        )
        | _STDLIB_TRACKED_MOD_NAMES
    )
    for sc in graph.scopes.values():
        if sc.globals_decl or sc.nonlocals_decl:
            sc.local_names.difference_update(sc.globals_decl)
            sc.local_names.difference_update(sc.nonlocals_decl)
    for cand in graph.alias_candidates:
        if (
            cand.target_name not in seeded_builtins
            and cand.target_name in graph.scopes[cand.target_owner_sid].globals_decl
        ):
            root_sc.local_names.add(cand.target_name)
    for w_sid, w_name, resolve_lex in graph.non_alias_writes:
        if (
            resolve_lex
            and w_name not in seeded_builtins
            and w_name in graph.scopes[w_sid].globals_decl
        ):
            root_sc.local_names.add(w_name)
    for b in graph.builtin_binds:
        if b.owner_sid == root_sid or b.name in graph.scopes[b.owner_sid].globals_decl:
            root_sc.local_names.add(b.name)
    return graph


def _resolve_pure_setattr_rhs_dep(
    cand: _AliasCandidate,
    table: _SlotAliasTable,
) -> Optional[_Slot]:
    """Returns _PURE_DIRECT_ROOT or source variable slot for a pure setattr/delattr RHS."""
    graph = table.graph
    rhs = _unwrap_named_expr(cand.rhs_node)
    if isinstance(rhs, ast.Name):
        return _resolve_lexical_slot(
            graph.scopes, graph.root_sid, cand.rhs_eval_sid, rhs.id
        )
    if isinstance(rhs, ast.Attribute) and (
        any(
            rhs.attr in attrs and _is_name_in_slots(rhs.value, slots, table)
            for attrs, slots in (
                (_ATTR_MUTATION_BUILTINS, table.builtins_mods),
                (_ATTR_MUTATION_DUNDERS, table.type_object_refs),
            )
        )
        or (
            rhs.attr in _ATTR_MUTATION_DUNDERS
            and _is_builtins_attr(rhs.value, table, frozenset({"type", "object"}))
        )
    ):
        return _PURE_DIRECT_ROOT
    return None


def _is_pure_setattr_dep(
    dep: Optional[_Slot],
    pure_imported_slots: AbstractSet[_Slot],
    pure_slots: AbstractSet[_Slot],
) -> bool:
    """Returns True when dep resolves cleanly to builtin/imported/pure setattr or delattr."""
    if dep is None:
        return False
    return (
        dep == _PURE_DIRECT_ROOT
        or (dep[0] == _BUILTIN_SID and dep[1] in _ATTR_MUTATION_BUILTINS)
        or dep in pure_imported_slots
        or dep in pure_slots
    )


def _prune_escaped_pure_setattr_slots(
    table: _SlotAliasTable,
    pure_imported_slots: AbstractSet[_Slot],
    direct_call_slots: AbstractSet[_Slot],
    cands_by_slot: Mapping[_Slot, List[Tuple[int, Optional[_Slot]]]],
    all_tgts_per_rhs: Mapping[int, Set[_Slot]],
    pure_slots: Set[_Slot],
) -> None:
    """Iteratively removes slots whose loads escape uncalled or whose dependencies break."""
    graph = table.graph
    loads_by_slot: Dict[_Slot, List[int]] = {}
    for load_node, eval_sid in graph.load_records:
        l_slot = _resolve_lexical_slot(
            graph.scopes, graph.root_sid, eval_sid, load_node.id
        )
        loads_by_slot.setdefault(l_slot, []).append(id(load_node))

    changed = True
    while changed:
        changed = False
        for slot in list(pure_slots):
            slot_loads = loads_by_slot.get(slot)
            if not slot_loads and slot not in direct_call_slots:
                has_escape = True
            else:
                has_escape = any(
                    lid not in graph.call_func_ids
                    and not (
                        lid in all_tgts_per_rhs
                        and all_tgts_per_rhs[lid].issubset(pure_slots)
                    )
                    for lid in (slot_loads or ())
                )
            has_broken_dep = any(
                not _is_pure_setattr_dep(
                    dep, pure_imported_slots, pure_slots
                )
                for _, dep in cands_by_slot[slot]
            )
            if has_escape or has_broken_dep:
                pure_slots.remove(slot)
                changed = True


def _collect_pure_setattr_rhs_ids(table: _SlotAliasTable) -> Set[int]:
    """Returns RHS AST node ids of assignments forming call-only setattr/delattr aliases."""
    graph = table.graph
    disqualified: Set[_Slot] = {
        (
            _resolve_target_slot(graph.scopes, graph.root_sid, sid, name)
            if resolve_lex
            else (sid, name)
        )
        for sid, name, resolve_lex in graph.non_alias_writes
    }
    pure_imported_slots: Set[_Slot] = set()
    for b in graph.builtin_binds:
        b_slot = _resolve_target_slot(graph.scopes, graph.root_sid, b.owner_sid, b.name)
        if b.is_pure_setattr_import:
            pure_imported_slots.add(b_slot)
        else:
            disqualified.add(b_slot)
    pure_imported_slots -= disqualified

    direct_call_slots: Set[_Slot] = set()
    cands_by_slot: Dict[_Slot, List[Tuple[int, Optional[_Slot]]]] = {}
    all_tgts_per_rhs: Dict[int, Set[_Slot]] = {}
    for cand in graph.alias_candidates:
        tgt_slot = _resolve_target_slot(
            graph.scopes, graph.root_sid, cand.target_owner_sid, cand.target_name
        )
        dep = _resolve_pure_setattr_rhs_dep(cand, table)
        if (
            dep is None
            or graph.scopes[tgt_slot[0]].kind not in ("module", "function")
            or tgt_slot[1] in _DYNAMIC_CODE_BUILTINS
        ):
            disqualified.add(tgt_slot)
        if cand.is_direct_call:
            direct_call_slots.add(tgt_slot)
        cands_by_slot.setdefault(tgt_slot, []).append((id(cand.rhs_node), dep))
        all_tgts_per_rhs.setdefault(id(cand.rhs_node), set()).add(tgt_slot)

    pure_slots: Set[_Slot] = set()
    changed = True
    while changed:
        changed = False
        for slot, pairs in cands_by_slot.items():
            if slot in disqualified or slot in pure_slots:
                continue
            if all(
                _is_pure_setattr_dep(dep, pure_imported_slots, pure_slots)
                for _, dep in pairs
            ):
                pure_slots.add(slot)
                changed = True

    _prune_escaped_pure_setattr_slots(
        table,
        pure_imported_slots,
        direct_call_slots,
        cands_by_slot,
        all_tgts_per_rhs,
        pure_slots,
    )
    return {
        rhs_id
        for rhs_id, tgts in all_tgts_per_rhs.items()
        if tgts and tgts.issubset(pure_slots)
    }


def _is_uncalled_patch_ref(
    node: ast.AST,
    table: _SlotAliasTable,
    ref_trigger_slots: AbstractSet[_Slot],
) -> bool:
    """Returns True when node is an uncalled load of dynamic code or attr mutation builtins/dunders."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return _is_name_in_slots(node, ref_trigger_slots, table)
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return (
            _is_builtins_attr(node, table, _REF_TRIGGER_ATTRS)
            or node.attr in _ATTR_MUTATION_DUNDERS
        )
    return False


def _record_stdlib_factory_rebind(
    target_expr: ast.AST,
    attr_name: Optional[str],
    table: _SlotAliasTable,
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
    *,
    allow_dynamic_fallback: bool,
) -> None:
    """Records canonical stdlib module or imported-target factory rebinding pairs."""
    mods = _resolve_expr_stdlib_mods(target_expr, table)
    if mods:
        for mod in mods:
            rebound_attrs.add((mod, attr_name))
        return
    if isinstance(target_expr, ast.Name):
        if any(
            slot in table.non_stdlib_import_slots
            for slot in _resolve_expr_name_slots(target_expr, table)
        ):
            rebound_attrs.add((target_expr.id, attr_name))
        return
    if allow_dynamic_fallback:
        rebound_attrs.add((None, attr_name))


def _mark_exempt_namespace_chain(
    expr: ast.AST, table: _SlotAliasTable, exempt_ids: Set[int]
) -> None:
    """Marks pure builtins-namespace container nodes used strictly within an attr-mutation callee."""
    if expr is not _unwrap_named_expr(expr):
        return
    if _is_builtins_namespace_expr(expr, table):
        exempt_ids.add(id(expr))
        if isinstance(expr, ast.Subscript) and expr.value is _unwrap_named_expr(
            expr.value
        ):
            exempt_ids.add(id(expr.value))


def _collect_exempt_attr_mutation_callee_ids(
    graph: _CollectedScopeGraph,
    table: _SlotAliasTable,
) -> Set[int]:
    """Returns node ids of immediate setattr/delattr callee lookups handled by _attr_mutation_call_info."""
    exempt_ids: Set[int] = set()
    for node in graph.nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.NamedExpr):
            continue
        if _resolve_attr_mutation_callee(func, table) is None:
            continue
        exempt_ids.add(id(func))
        extracted = _extract_container_and_key(func)
        if extracted is not None:
            _mark_exempt_namespace_chain(extracted[0], table, exempt_ids)
        elif isinstance(func, ast.Attribute) and _is_builtins_namespace_expr(
            func.value, table
        ):
            _mark_exempt_namespace_chain(func.value, table, exempt_ids)
        else:
            getattr_call = _is_getattr_call(func, table)
            if getattr_call is not None and len(getattr_call.args) >= 2:
                _mark_exempt_namespace_chain(getattr_call.args[0], table, exempt_ids)
    return exempt_ids


def _record_dynamic_builtins_rebind(
    node: ast.AST,
    table: _SlotAliasTable,
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> None:
    """Records stdlib and explicit builtins factory rebind markers for dynamic builtins lookups."""
    rebound_attrs.add((None, None))
    targets_builtins_ns = _is_builtins_namespace_expr(node, table)
    if not targets_builtins_ns and isinstance(node, ast.Subscript):
        if _is_name_in_slots(node.value, table.builtins_mods, table) and (
            not isinstance(node.slice, ast.Constant) or node.slice.value == "open"
        ):
            targets_builtins_ns = True
    if targets_builtins_ns:
        rebound_attrs.add(("builtins", None))
        rebound_attrs.add(("__builtins__", None))


def _extract_vars_target_arg(node: ast.AST, table: _SlotAliasTable) -> Optional[ast.AST]:
    """Returns target_expr when node calls 1-arg vars(target_expr), else None."""
    if not isinstance(node, ast.Call):
        return None
    func = _unwrap_named_expr(node.func)
    is_vars = _is_name_in_slots(func, table.vars_fns, table) or _is_builtins_attr(
        func, table, frozenset({"vars"})
    )
    if (
        is_vars
        and len(node.args) == 1
        and not isinstance(node.args[0], ast.Starred)
        and not node.keywords
    ):
        return node.args[0]
    return None


def _resolve_direct_attr_dict_owner(
    expr: ast.AST, table: _SlotAliasTable
) -> Optional[ast.AST]:
    """Returns target_expr if expr is directly `<target>.__dict__` or `vars(<target>)`."""
    unwrapped = _unwrap_named_expr(expr)
    if isinstance(unwrapped, ast.Attribute) and unwrapped.attr == "__dict__":
        return unwrapped.value
    return _extract_vars_target_arg(unwrapped, table)


def _collect_attr_dict_aliases(table: _SlotAliasTable) -> Dict[_Slot, ast.AST]:
    """Maps variable slots aliasing `<target>.__dict__` or `vars(<target>)` to `<target>`."""
    aliases: Dict[_Slot, ast.AST] = {}
    graph = table.graph
    for cand in graph.alias_candidates:
        owner = _resolve_direct_attr_dict_owner(cand.rhs_node, table)
        if owner is None:
            continue
        tgt_slot = _resolve_target_slot(
            graph.scopes, graph.root_sid, cand.target_owner_sid, cand.target_name
        )
        aliases[tgt_slot] = owner
    return aliases


def _resolve_attr_dict_target(
    expr: ast.AST,
    table: _SlotAliasTable,
    dict_aliases: Mapping[_Slot, ast.AST],
) -> Optional[ast.AST]:
    """Returns owner target AST for direct or variable-aliased `__dict__` / `vars(x)` containers."""
    direct = _resolve_direct_attr_dict_owner(expr, table)
    if direct is not None:
        return direct
    unwrapped = _unwrap_named_expr(expr)
    if isinstance(unwrapped, ast.Name):
        for slot in _resolve_expr_name_slots(unwrapped, table):
            if slot in dict_aliases:
                return dict_aliases[slot]
    return None


def _extract_dict_literal_keys(dict_node: ast.Dict) -> List[Optional[str]]:
    """Extracts string keys from a dict literal, using None for dynamic or unpacked keys."""
    out: List[Optional[str]] = []
    for k in dict_node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            out.append(k.value)
        else:
            out.append(None)
    return out


def _extract_dict_mutation_keys(
    node: ast.Call,
) -> Optional[Tuple[ast.AST, Tuple[Optional[str], ...]]]:
    """Returns (receiver_container, mutated_keys_or_None) for dict mutation method calls."""
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _DICT_MUTATION_METHODS
    ):
        return None
    method = node.func.attr
    container = node.func.value
    if method in ("popitem", "clear"):
        return container, (None,)
    if method in ("pop", "setdefault"):
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return container, (node.args[0].value,)
        return container, (None,)
    keys: List[Optional[str]] = []
    for arg in node.args:
        if isinstance(arg, ast.Dict):
            keys.extend(_extract_dict_literal_keys(arg))
        else:
            keys.append(None)
    for kw in node.keywords:
        if kw.arg is not None:
            keys.append(kw.arg)
        elif isinstance(kw.value, ast.Dict):
            keys.extend(_extract_dict_literal_keys(kw.value))
        else:
            keys.append(None)
    return (container, tuple(keys)) if keys else None


def _record_namespace_dict_mutation(
    container: ast.AST,
    keys: Tuple[Optional[str], ...],
    table: _SlotAliasTable,
    dict_aliases: Mapping[_Slot, ast.AST],
    patched: Set[str],
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]],
) -> bool:
    """Records exit-protocol or stdlib-factory mutations via namespace dicts; returns True if has_patch."""
    has_patch_hit = False
    if _is_namespace_dict_expr(container, table):
        vis_builtins = _visible_builtins_mod_names(container, table)
        for k in keys:
            if k is None:
                has_patch_hit = True
                rebound_attrs.add((None, None))
            elif (
                k in _EXIT_PROTOCOL_ATTRS
                or k in _REF_TRIGGER_ATTRS
                or k in _BUILTIN_NAMESPACE_ATTRS
                or k in vis_builtins
            ):
                has_patch_hit = True
                rebound_attrs.add((None, None))
            elif k == "open":
                rebound_attrs.add(("builtins", "open"))
                rebound_attrs.add(("__builtins__", "open"))
            elif k in _STDLIB_FACTORY_ATTRS:
                rebound_attrs.add((None, k))
        return has_patch_hit
    target_expr = _resolve_attr_dict_target(container, table, dict_aliases)
    if target_expr is None:
        return False
    for k in keys:
        if k is not None and k in _EXIT_PROTOCOL_ATTRS:
            has_patch_hit = True
            if isinstance(target_expr, ast.Name):
                patched.add(target_expr.id)
        elif k is None or k in _STDLIB_FACTORY_ATTRS:
            _record_stdlib_factory_rebind(
                target_expr,
                k,
                table,
                rebound_attrs,
                allow_dynamic_fallback=False,
            )
    return has_patch_hit


def _scan_exit_protocol_patches_uncached(tree: ast.AST) -> ExitProtocolPatchScan:
    """Walks every scope of tree collecting exit-protocol and stdlib-factory patch targets."""
    table = _collect_builtin_slot_aliases(tree)
    graph = table.graph
    dict_aliases = _collect_attr_dict_aliases(table)
    ref_trigger_slots = table.exec_eval | table.setattr_delattr
    patched: Set[str] = set()
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]] = set()
    has_patch = bool(graph.wildcard_sids)
    if has_patch:
        rebound_attrs.add((None, None))
    pure_setattr_rhs_ids = _collect_pure_setattr_rhs_ids(table)
    exempt_lookup_ids = _collect_exempt_attr_mutation_callee_ids(graph, table)
    for node in graph.nodes:
        if id(node) not in exempt_lookup_ids and _is_dynamic_builtins_lookup(
            node, table
        ):
            has_patch = True
            _record_dynamic_builtins_rebind(node, table, rebound_attrs)
            continue
        if isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if node.attr in _EXIT_PROTOCOL_ATTRS:
                has_patch = True
                if isinstance(node.value, ast.Name):
                    patched.add(node.value.id)
            elif node.attr in _STDLIB_FACTORY_ATTRS:
                _record_stdlib_factory_rebind(
                    node.value,
                    node.attr,
                    table,
                    rebound_attrs,
                    allow_dynamic_fallback=False,
                )
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            sub_key: Tuple[Optional[str], ...] = (
                (node.slice.value,)
                if isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                else (None,)
            )
            if _record_namespace_dict_mutation(
                node.value, sub_key, table, dict_aliases, patched, rebound_attrs
            ):
                has_patch = True
        elif isinstance(node, ast.Call):
            func = _unwrap_named_expr(node.func)
            if (
                _is_name_in_slots(func, table.exec_eval, table)
                or _is_builtins_attr(func, table, _DYNAMIC_CODE_BUILTINS)
                or (
                    id(node) not in exempt_lookup_ids
                    and _is_getattr_of_mutation_builtin(node, table)
                )
            ):
                has_patch = True
                rebound_attrs.add((None, None))
                continue
            if isinstance(func, ast.Name):
                bound_hit = False
                for f_slot in _resolve_expr_name_slots(func, table):
                    if f_slot in table.bound_dunders:
                        bound_hit = True
                        for r_name, r_mod in table.bound_dunders[f_slot]:
                            if r_name is not None:
                                patched.add(r_name)
                            if r_mod != "":
                                rebound_attrs.add((r_mod, None))
                if bound_hit:
                    has_patch = True
                    continue
            mut_info = _attr_mutation_call_info(node, table)
            if mut_info is None:
                dict_mut = _extract_dict_mutation_keys(node)
                if dict_mut is not None and _record_namespace_dict_mutation(
                    dict_mut[0],
                    dict_mut[1],
                    table,
                    dict_aliases,
                    patched,
                    rebound_attrs,
                ):
                    has_patch = True
                continue
            target_expr, literal_attr = mut_info
            if literal_attr is not None and literal_attr in _STDLIB_FACTORY_ATTRS:
                _record_stdlib_factory_rebind(
                    target_expr,
                    literal_attr,
                    table,
                    rebound_attrs,
                    allow_dynamic_fallback=False,
                )
            else:
                has_patch = True
                if isinstance(target_expr, ast.Name):
                    patched.add(target_expr.id)
                if literal_attr is None:
                    _record_stdlib_factory_rebind(
                        target_expr,
                        None,
                        table,
                        rebound_attrs,
                        allow_dynamic_fallback=not isinstance(target_expr, ast.Name),
                    )
        elif (
            id(node) not in graph.call_func_ids
            and id(node) not in pure_setattr_rhs_ids
            and _is_uncalled_patch_ref(node, table, ref_trigger_slots)
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
