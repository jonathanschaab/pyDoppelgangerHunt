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
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

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
    importlib_mods: Set[_Slot]
    sys_mods: Set[_Slot]
    type_object_refs: Set[_Slot]


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


def _is_name_in_slots(
    node: ast.AST,
    slot_set: AbstractSet[_Slot],
    table: _SlotAliasTable,
) -> bool:
    """Returns True when node is an ast.Name whose resolved slot is in slot_set."""
    if not isinstance(node, ast.Name):
        return False
    eval_sid = table.graph.node_sids.get(id(node), table.graph.root_sid)
    lex_slot = _resolve_lexical_slot(
        table.graph.scopes, table.graph.root_sid, eval_sid, node.id
    )
    if lex_slot in slot_set:
        return True
    sc = table.graph.scopes[eval_sid]
    return (
        sc.kind == "class"
        and node.id not in sc.globals_decl
        and node.id not in sc.nonlocals_decl
        and (eval_sid, node.id) in slot_set
    )


def _visible_builtins_mod_names(node: ast.AST, table: _SlotAliasTable) -> Set[str]:
    """Returns builtins-module identifier names visible from node's lexical scope chain."""
    anc_sids: Set[int] = {table.graph.root_sid}
    curr: Optional[int] = table.graph.node_sids.get(id(node), table.graph.root_sid)
    while curr is not None:
        anc_sids.add(curr)
        curr = table.graph.scopes[curr].parent_lexical_sid
    return {name for sid, name in table.builtins_mods if sid in anc_sids}


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
        is_import_call = _is_name_in_slots(func, table.import_fns, table) or (
            isinstance(func, ast.Attribute)
            and (
                (
                    _is_name_in_slots(func.value, table.importlib_mods, table)
                    and func.attr == "import_module"
                )
                or (
                    _is_name_in_slots(func.value, table.builtins_mods, table)
                    and func.attr == "__import__"
                )
            )
        )
        if is_import_call:
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
    elif isinstance(expr, ast.Subscript):
        if (
            isinstance(expr.value, ast.Attribute)
            and _is_name_in_slots(expr.value.value, table.sys_mods, table)
            and expr.value.attr == "modules"
            and isinstance(expr.slice, ast.Constant)
            and expr.slice.value in ("builtins", "__builtins__")
        ):
            return True
        if _is_namespace_dict_expr(expr.value, table):
            if isinstance(expr.slice, ast.Constant) and expr.slice.value in (
                "__builtins__",
                *_visible_builtins_mod_names(expr, table),
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
    load_records: List[Tuple[ast.Name, int]]


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
        if _is_name_in_slots(recv, table.type_object_refs, table):
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
            if _is_name_in_slots(expr.value, table.type_object_refs, table):
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
        (table.importlib_mods, frozenset({"import_module"}), table.import_fns),
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
        table.importlib_mods,
        table.sys_mods,
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
        elif isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
            if _propagate_attr_alias(tgt_slot, val, table):
                changed = True
        if _is_namespace_dict_expr(val, table) and tgt_slot not in table.namespace_dicts:
            table.namespace_dicts.add(tgt_slot)
            changed = True
        if _is_builtins_namespace_expr(val, table) and tgt_slot not in table.builtins_mods:
            table.builtins_mods.add(tgt_slot)
            changed = True
        if (
            _resolve_attr_mutation_callee(val, table, allow_bound_dunder=False) is not None
            and tgt_slot not in table.setattr_delattr
        ):
            table.setattr_delattr.add(tgt_slot)
            changed = True
    return changed


def _collect_builtin_slot_aliases(tree: ast.AST) -> _SlotAliasTable:
    """Builds the scope graph and computes fixed-point lexical slot aliases for builtins."""
    graph = _collect_scope_graph(tree)
    root = graph.root_sid
    table = _SlotAliasTable(
        graph=graph,
        exec_eval={(root, "exec"), (root, "eval")},
        setattr_delattr={(root, "setattr"), (root, "delattr")},
        getattr_fns={(root, "getattr")},
        vars_fns={(root, "vars")},
        globals_fns={(root, "globals")},
        namespace_dicts=set(),
        builtins_mods={(root, "builtins"), (root, "__builtins__")},
        import_fns={(root, "__import__")},
        importlib_mods={(root, "importlib")},
        sys_mods={(root, "sys")},
        type_object_refs={(root, "type"), (root, "object")},
    )
    kind_map: Dict[str, Set[_Slot]] = {
        "exec_eval": table.exec_eval,
        "setattr_delattr": table.setattr_delattr,
        "getattr_fns": table.getattr_fns,
        "vars_fns": table.vars_fns,
        "globals_fns": table.globals_fns,
        "builtins_mods": table.builtins_mods,
        "import_fns": table.import_fns,
        "importlib_mods": table.importlib_mods,
        "sys_mods": table.sys_mods,
        "type_object_refs": table.type_object_refs,
    }
    for b in graph.builtin_binds:
        b_slot = _resolve_target_slot(graph.scopes, root, b.owner_sid, b.name)
        kind_map[b.alias_kind].add(b_slot)
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
        callee.bound_target if callee.bound_target is not None else node.args[target_idx]
    )
    return target_expr, literal_attr


_REF_TRIGGER_ATTRS: frozenset[str] = (
    _ATTR_MUTATION_BUILTINS | _ATTR_MUTATION_DUNDERS | _DYNAMIC_CODE_BUILTINS
)


_BUILTIN_NAMESPACE_ATTRS: frozenset[str] = frozenset({"vars", "globals", "__import__"})


def _is_getattr_of_mutation_builtin(
    node: ast.Call,
    table: _SlotAliasTable,
) -> bool:
    """Returns True when node is getattr(..., '<mutation>') or getattr(<builtins>, dynamic_attr)."""
    getattr_call = _is_getattr_call(node, table)
    if getattr_call is None or len(getattr_call.args) < 2:
        return False
    target_is_builtins = _is_name_in_slots(
        getattr_call.args[0], table.builtins_mods, table
    ) or _is_builtins_namespace_expr(getattr_call.args[0], table)
    attr_arg = getattr_call.args[1]
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
        return (
            not isinstance(node.slice, ast.Constant)
            or node.slice.value in _REF_TRIGGER_ATTRS
            or node.slice.value in _BUILTIN_NAMESPACE_ATTRS
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
_PURE_DIRECT_ROOT: _Slot = (-2, "")
_FROM_BUILTIN_KIND: Mapping[Tuple[str, str], Tuple[str, bool]] = {
    ("importlib", "import_module"): ("import_fns", False),
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


def _resolve_decl_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    sc: _ScopeRecord,
    name: str,
) -> Optional[_Slot]:
    """Resolves explicit global or nonlocal declarations for name in sc, else None."""
    if name in sc.globals_decl:
        return root_sid, name
    if name in sc.nonlocals_decl:
        curr = sc.parent_lexical_sid
        while curr is not None and curr != root_sid:
            if name in scopes[curr].local_names:
                return curr, name
            curr = scopes[curr].parent_lexical_sid
        return root_sid, name
    return None


def _resolve_lexical_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    scope_sid: int,
    name: str,
) -> _Slot:
    """Resolves a name load in scope_sid to its defining (scope_id, name) lexical slot."""
    sc = scopes[scope_sid]
    decl_slot = _resolve_decl_slot(scopes, root_sid, sc, name)
    if decl_slot is not None:
        return decl_slot
    if sc.kind != "class" and name in sc.local_names:
        return scope_sid, name
    curr = sc.parent_lexical_sid
    while curr is not None:
        if name in scopes[curr].local_names:
            return curr, name
        curr = scopes[curr].parent_lexical_sid
    return root_sid, name


def _resolve_target_slot(
    scopes: Mapping[int, _ScopeRecord],
    root_sid: int,
    scope_sid: int,
    name: str,
) -> _Slot:
    """Resolves a binding target in scope_sid to its owning (scope_id, name) slot."""
    sc = scopes[scope_sid]
    decl_slot = _resolve_decl_slot(scopes, root_sid, sc, name)
    if decl_slot is not None:
        return decl_slot
    return scope_sid, name


def _make_scope_record(
    node: ast.AST,
    sid: int,
    parent_lex_sid: Optional[int],
    parent_non_comp_sid: int,
) -> _ScopeRecord:
    """Creates a _ScopeRecord for node with precomputed function/comprehension locals."""
    if isinstance(node, ast.Module):
        kind = "module"
    elif isinstance(node, ast.ClassDef):
        kind = "class"
    elif isinstance(node, _COMPREHENSION_TYPES):
        kind = "comp"
    else:
        kind = "function"
    local_names = (
        _extract_single_node_local_names(
            node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=True,
            include_bare_annotations=True,
        )
        if kind in ("function", "comp")
        else set()
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


def _record_import_in_scope(
    curr: Union[ast.Import, ast.ImportFrom],
    curr_sid: int,
    graph: _CollectedScopeGraph,
) -> None:
    """Records imported bindings as builtin binds or non-alias writes and marks function locals."""
    sc = graph.scopes[curr_sid]
    if isinstance(curr, ast.Import):
        for alias in curr.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if sc.kind == "function":
                sc.local_names.add(bound)
            kind = _IMPORT_MOD_KIND.get(alias.name)
            if kind is not None:
                graph.builtin_binds.append(_BuiltinBindRecord(curr_sid, bound, kind, False))
            else:
                graph.non_alias_writes.append((curr_sid, bound, True))
        return
    mod = curr.module if getattr(curr, "level", 0) == 0 else None
    for alias in curr.names:
        if alias.name == "*":
            if mod in ("builtins", "__builtins__", "importlib"):
                graph.wildcard_sids.add(curr_sid)
            continue
        bound = alias.asname or alias.name
        if sc.kind == "function":
            sc.local_names.add(bound)
        entry = _FROM_BUILTIN_KIND.get((mod, alias.name)) if mod is not None else None
        if entry is not None:
            graph.builtin_binds.append(
                _BuiltinBindRecord(curr_sid, bound, entry[0], entry[1])
            )
        else:
            graph.non_alias_writes.append((curr_sid, bound, True))


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
            and all(isinstance(t, ast.Name) for t in curr.targets)
            and sc_kind in ("module", "function", "class")
        ):
            for tgt in curr.targets:
                graph.alias_candidates.append(
                    _AliasCandidate(curr_sid, cast(ast.Name, tgt).id, curr.value, curr_sid)
                )
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
            if sc_kind in ("module", "function")
            else graph.scopes[curr_sid].parent_non_comp_sid
        )
        if graph.scopes[owner_sid].kind == "function":
            graph.scopes[owner_sid].local_names.add(curr.target.id)
        graph.alias_candidates.append(
            _AliasCandidate(owner_sid, curr.target.id, curr.value, curr_sid)
        )
        if id(curr) not in graph.call_func_ids:
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
            graph.non_alias_writes.append((curr_sid, curr.name, False))
    else:
        bound = getattr(curr, "name", None) or getattr(curr, "rest", None)
        if isinstance(bound, str) and bound != "_" and not isinstance(curr, ast.Attribute):
            graph.non_alias_writes.append((curr_sid, bound, False))


def _collect_scope_graph(tree: ast.AST) -> _CollectedScopeGraph:
    """Iteratively collects lexical scopes, bindings, load nodes, and call targets across tree."""
    is_mod = isinstance(tree, ast.Module)
    root_sid = id(tree) if is_mod else -1
    root_sc = _ScopeRecord(root_sid, "module", None, root_sid, set(), set(), set())
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
        load_records=[],
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
                cur_sc.sid if cur_sc.kind in ("module", "function") else cur_sc.parent_non_comp_sid
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
        if not _record_assign_candidate(curr, curr_sid, graph):
            _record_non_assign_node(curr, curr_sid, graph)
        stack.extend((ch, curr_sid) for ch in reversed(list(ast.iter_child_nodes(curr))))
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
    if isinstance(rhs, ast.Attribute) and any(
        rhs.attr in attrs and _is_name_in_slots(rhs.value, slots, table)
        for attrs, slots in (
            (_ATTR_MUTATION_BUILTINS, table.builtins_mods),
            (_ATTR_MUTATION_DUNDERS, table.type_object_refs),
        )
    ):
        return _PURE_DIRECT_ROOT
    return None


def _is_pure_setattr_dep(
    dep: Optional[_Slot],
    root_sid: int,
    pure_imported_slots: AbstractSet[_Slot],
    pure_slots: AbstractSet[_Slot],
) -> bool:
    """Returns True when dep resolves cleanly to builtin/imported/pure setattr or delattr."""
    if dep is None:
        return False
    return (
        dep == _PURE_DIRECT_ROOT
        or (dep[0] == root_sid and dep[1] in _ATTR_MUTATION_BUILTINS)
        or dep in pure_imported_slots
        or dep in pure_slots
    )


def _prune_escaped_pure_setattr_slots(
    table: _SlotAliasTable,
    pure_imported_slots: AbstractSet[_Slot],
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
            has_escape = any(
                lid not in graph.call_func_ids
                and not (
                    lid in all_tgts_per_rhs and all_tgts_per_rhs[lid].issubset(pure_slots)
                )
                for lid in loads_by_slot.get(slot, ())
            )
            has_broken_dep = any(
                not _is_pure_setattr_dep(
                    dep, graph.root_sid, pure_imported_slots, pure_slots
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
                _is_pure_setattr_dep(dep, graph.root_sid, pure_imported_slots, pure_slots)
                for _, dep in pairs
            ):
                pure_slots.add(slot)
                changed = True

    _prune_escaped_pure_setattr_slots(
        table, pure_imported_slots, cands_by_slot, all_tgts_per_rhs, pure_slots
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
    """Returns True when node is an uncalled load of dynamic code or attr mutation builtins."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return _is_name_in_slots(node, ref_trigger_slots, table)
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return _is_builtins_attr(node, table, _REF_TRIGGER_ATTRS) or (
            node.attr in _ATTR_MUTATION_DUNDERS
            and _is_name_in_slots(node.value, table.type_object_refs, table)
        )
    return False


def _scan_exit_protocol_patches_uncached(tree: ast.AST) -> ExitProtocolPatchScan:
    """Walks every scope of tree collecting exit-protocol and stdlib-factory patch targets."""
    table = _collect_builtin_slot_aliases(tree)
    graph = table.graph
    ref_trigger_slots = table.exec_eval | table.setattr_delattr
    patched: Set[str] = set()
    rebound_attrs: Set[Tuple[Optional[str], Optional[str]]] = set()
    has_patch = bool(graph.wildcard_sids)
    if has_patch:
        rebound_attrs.add((None, None))
    pure_setattr_rhs_ids = _collect_pure_setattr_rhs_ids(table)
    for node in graph.nodes:
        if _is_dynamic_builtins_lookup(node, table):
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
            elif node.attr in _STDLIB_FACTORY_ATTRS and isinstance(
                node.value, ast.Name
            ):
                rebound_attrs.add((node.value.id, node.attr))
        elif isinstance(node, ast.Call):
            func = _unwrap_named_expr(node.func)
            if (
                _is_name_in_slots(func, table.exec_eval, table)
                or _is_builtins_attr(func, table, _DYNAMIC_CODE_BUILTINS)
                or _is_getattr_of_mutation_builtin(node, table)
            ):
                has_patch = True
                rebound_attrs.add((None, None))
                continue
            mut_info = _attr_mutation_call_info(node, table)
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
