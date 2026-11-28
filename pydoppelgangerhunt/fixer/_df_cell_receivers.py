"""Receiver-attribute mutation tracking and enclosing class-method sibling closure analysis."""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cache import _get_scope_ancestor_map
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _extract_carrier_value_closure_and_load_info,
    _extract_store_names_in_node,
    _iter_carrier_target_entries,
    _iter_class_body_methods,
    _iter_class_body_non_method_stmts,
    _propagate_sibling_closures,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import (
    _EnclosingFnCellInfo,
    _extract_cached_scope_free_reads,
    _extract_closure_nonlocal_mutations_and_deletions,
    _get_fn_scope_context,
)
from pydoppelgangerhunt.fixer._df_cfg import _extract_type_param_names
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _extract_single_node_local_names,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

def _find_enclosing_method_class_node(
    fn_node: ast.AST,
    enc_child: ast.AST,
) -> Optional[ast.ClassDef]:
    """Finds the ClassDef inside fn_node that directly defines method enc_child."""
    parent_map = _get_scope_ancestor_map(fn_node)
    curr_p = parent_map.get(enc_child)
    while curr_p is not None and curr_p is not fn_node:
        if isinstance(curr_p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        if isinstance(curr_p, ast.ClassDef):
            return (
                curr_p
                if enc_child in _iter_class_body_methods(curr_p)
                else None
            )
        curr_p = parent_map.get(curr_p)
    return None


# INVARIANT (CONVERGENCE BOUNDARY — CLOSED RECEIVER OVER-APPROXIMATION):
# Receiver-attribute mutation tracking uses closed, frozen method/builtin name
# tables (``_RECEIVER_MUTATING_METHODS``, ``_RECEIVER_READONLY_METHODS``,
# ``_ALIAS_READONLY_CALL_BUILTINS``) combined with conservative over-approximation
# (unrecognized calls with arguments or bare receiver escapes fail closed).
# Do NOT incrementally add third-party container method names or chase arbitrary
# inter-procedural points-to chains; callers needing stricter guarantees must use
# ``closure_strictness = "pedantic"``.
_RECEIVER_MUTATING_METHODS: frozenset[str] = frozenset(
    {
        "add",
        "append",
        "appendleft",
        "clear",
        "discard",
        "extend",
        "extendleft",
        "insert",
        "pop",
        "popitem",
        "popleft",
        "remove",
        "reverse",
        "rotate",
        "setdefault",
        "sort",
        "update",
        "__delattr__",
        "__delitem__",
        "__iadd__",
        "__imul__",
        "__isub__",
        "__setattr__",
        "__setitem__",
    }
)


_RECEIVER_READONLY_METHODS: frozenset[str] = frozenset(
    {
        "copy",
        "count",
        "get",
        "index",
        "items",
        "keys",
        "values",
    }
)


_RECEIVER_ESCAPE_ATTRS: frozenset[str] = frozenset({"__class__", "__dict__"})


_RECEIVER_ATTR_BUILTINS: frozenset[str] = frozenset({"setattr", "delattr"})


_ALIAS_READONLY_CALL_BUILTINS: frozenset[str] = frozenset(
    {
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "frozenset",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }
)


def _extract_receiver_alias_attrs(
    expr: ast.AST,
    rec_param: str,
    alias_to_attrs: Mapping[str, Set[str]],
) -> Set[str]:
    """Returns receiver attributes referenced directly or via a local alias in expr."""
    curr_expr: ast.AST = expr
    root_attr: Optional[str] = None
    while True:
        if isinstance(curr_expr, ast.NamedExpr):
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Attribute):
            root_attr = curr_expr.attr
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Subscript):
            curr_expr = curr_expr.value
            continue
        if (
            isinstance(curr_expr, ast.Call)
            and isinstance(curr_expr.func, ast.Attribute)
            and curr_expr.func.attr in ("get", "setdefault")
        ):
            curr_expr = curr_expr.func.value
            continue
        break
    if isinstance(curr_expr, ast.Name):
        if curr_expr.id == rec_param and root_attr is not None:
            return {root_attr}
        if curr_expr.id in alias_to_attrs:
            return set(alias_to_attrs[curr_expr.id])
    return set()


def _collect_receiver_local_aliases(
    visited_nodes: Sequence[Tuple[ast.AST, frozenset[str]]],
    rec_param: str,
) -> Dict[str, Set[str]]:
    """Maps local variable aliases in visited_nodes to the rec_param attributes they reference."""
    pairs: List[Tuple[str, ast.AST]] = []
    for node, _ in visited_nodes:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id != rec_param:
                    pairs.append((tgt.id, node.value))
                elif (
                    isinstance(tgt, (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(tgt.elts) == len(node.value.elts)
                    and not any(isinstance(el, ast.Starred) for el in tgt.elts)
                    and not any(isinstance(el, ast.Starred) for el in node.value.elts)
                ):
                    for t_el, v_el in zip(tgt.elts, node.value.elts):
                        if isinstance(t_el, ast.Name) and t_el.id != rec_param:
                            pairs.append((t_el.id, v_el))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and node.target.id != rec_param
        ):
            pairs.append((node.target.id, node.value))
        elif (
            isinstance(node, ast.NamedExpr)
            and isinstance(node.target, ast.Name)
            and node.target.id != rec_param
        ):
            pairs.append((node.target.id, node.value))
    alias_to_attrs: Dict[str, Set[str]] = {}
    if not pairs:
        return alias_to_attrs
    changed = True
    while changed:
        changed = False
        for tgt_name, rhs in pairs:
            resolved = _extract_receiver_alias_attrs(rhs, rec_param, alias_to_attrs)
            if resolved:
                dest = alias_to_attrs.setdefault(tgt_name, set())
                before = len(dest)
                dest.update(resolved)
                if len(dest) > before:
                    changed = True
    return alias_to_attrs


def _record_aliased_receiver_mutations(
    visited_nodes: Sequence[Tuple[ast.AST, frozenset[str]]],
    rec_param: str,
    alias_to_attrs: Mapping[str, Set[str]],
    stored_attrs: Set[str],
) -> None:
    """Records rec_param attribute mutations performed through local container aliases."""
    if not alias_to_attrs:
        return
    for curr, cur_shadowed in visited_nodes:
        if isinstance(curr, ast.AugAssign):
            stored_attrs.update(
                _extract_receiver_alias_attrs(curr.target, rec_param, alias_to_attrs)
            )
        elif isinstance(curr, (ast.Attribute, ast.Subscript)) and isinstance(
            curr.ctx, (ast.Store, ast.Del)
        ):
            stored_attrs.update(
                _extract_receiver_alias_attrs(curr, rec_param, alias_to_attrs)
            )
        elif isinstance(curr, ast.Call):
            is_readonly_call = False
            if isinstance(curr.func, ast.Attribute):
                if curr.func.attr in _RECEIVER_MUTATING_METHODS or (
                    (curr.args or curr.keywords)
                    and curr.func.attr not in _RECEIVER_READONLY_METHODS
                ):
                    stored_attrs.update(
                        _extract_receiver_alias_attrs(
                            curr.func.value, rec_param, alias_to_attrs
                        )
                    )
                elif curr.func.attr in _RECEIVER_READONLY_METHODS:
                    is_readonly_call = True
            elif isinstance(curr.func, ast.Name):
                if curr.func.id in ("setattr", "delattr") and curr.args:
                    stored_attrs.update(
                        _extract_receiver_alias_attrs(
                            curr.args[0], rec_param, alias_to_attrs
                        )
                    )
                elif (
                    curr.func.id in _ALIAS_READONLY_CALL_BUILTINS
                    and curr.func.id not in cur_shadowed
                ):
                    is_readonly_call = True
            if not is_readonly_call:
                for arg in curr.args:
                    a_expr = arg.value if isinstance(arg, ast.Starred) else arg
                    if isinstance(a_expr, ast.Name) and a_expr.id in alias_to_attrs:
                        stored_attrs.update(alias_to_attrs[a_expr.id])
                for kw in curr.keywords:
                    if isinstance(kw.value, ast.Name) and kw.value.id in alias_to_attrs:
                        stored_attrs.update(alias_to_attrs[kw.value.id])


def _extract_receiver_root_attr(expr: ast.AST, rec_param: str) -> Optional[str]:
    """Returns the attribute name on rec_param when expr is rooted at ``rec_param.<attr>``."""
    curr_expr: ast.AST = expr
    root_attr: Optional[str] = None
    while True:
        if isinstance(curr_expr, ast.NamedExpr):
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Attribute):
            root_attr = curr_expr.attr
            curr_expr = curr_expr.value
            continue
        if isinstance(curr_expr, ast.Subscript):
            curr_expr = curr_expr.value
            continue
        break
    if isinstance(curr_expr, ast.Name) and curr_expr.id == rec_param:
        return root_attr
    return None


def _record_receiver_call_mutation(
    curr: ast.Call,
    rec_param: str,
    stored_attrs: Set[str],
    attr_val_ids: Optional[Set[int]] = None,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> bool:
    """Records receiver attribute mutations from mutating calls; returns True on dynamic bare-receiver mutation."""
    is_readonly_call = False
    if isinstance(curr.func, ast.Attribute):
        if (
            curr.func.attr in ("__setattr__", "__delattr__")
            and isinstance(curr.func.value, ast.Name)
            and curr.func.value.id == rec_param
        ):
            if (
                not curr.keywords
                and curr.args
                and isinstance(curr.args[0], ast.Constant)
                and isinstance(curr.args[0].value, str)
            ):
                stored_attrs.add(curr.args[0].value)
                return False
            return True
        if curr.func.attr in _RECEIVER_MUTATING_METHODS or (
            (curr.args or curr.keywords)
            and curr.func.attr not in _RECEIVER_READONLY_METHODS
        ):
            call_attr = _extract_receiver_root_attr(curr.func.value, rec_param)
            if call_attr is not None:
                stored_attrs.add(call_attr)
        elif curr.func.attr in _RECEIVER_READONLY_METHODS:
            is_readonly_call = True
    elif (
        isinstance(curr.func, ast.Name)
        and curr.func.id in _RECEIVER_ATTR_BUILTINS
        and curr.args
    ):
        target_arg = curr.args[0]
        if isinstance(target_arg, ast.Name) and target_arg.id == rec_param:
            if (
                curr.func.id not in shadowed_builtins
                and not curr.keywords
                and len(curr.args) >= 2
                and not any(isinstance(a, ast.Starred) for a in curr.args[:2])
                and isinstance(curr.args[1], ast.Constant)
                and isinstance(curr.args[1].value, str)
            ):
                stored_attrs.add(curr.args[1].value)
                if attr_val_ids is not None:
                    attr_val_ids.add(id(target_arg))
                return False
            return True
        call_attr = _extract_receiver_root_attr(target_arg, rec_param)
        if call_attr is not None:
            stored_attrs.add(call_attr)
        return False
    elif (
        isinstance(curr.func, ast.Name)
        and curr.func.id in _ALIAS_READONLY_CALL_BUILTINS
        and curr.func.id not in shadowed_builtins
    ):
        is_readonly_call = True
    if not is_readonly_call:
        for arg in curr.args:
            a_expr = arg.value if isinstance(arg, ast.Starred) else arg
            arg_attr = _extract_receiver_root_attr(a_expr, rec_param)
            if arg_attr is not None:
                stored_attrs.add(arg_attr)
        for kw in curr.keywords:
            kw_attr = _extract_receiver_root_attr(kw.value, rec_param)
            if kw_attr is not None:
                stored_attrs.add(kw_attr)
    return False


def _record_receiver_container_mutations(
    curr: ast.AST,
    rec_param: str,
    loaded_attrs: Set[str],
    stored_attrs: Set[str],
    attr_val_ids: Optional[Set[int]] = None,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> bool:
    """Records rec_param root attribute mutations across AugAssign, nested targets, and mutating calls."""
    if isinstance(curr, ast.AugAssign) and isinstance(
        curr.target, (ast.Attribute, ast.Subscript)
    ):
        aug_attr = _extract_receiver_root_attr(curr.target, rec_param)
        if aug_attr is not None:
            loaded_attrs.add(aug_attr)
            stored_attrs.add(aug_attr)
        return False
    if isinstance(curr, (ast.Attribute, ast.Subscript)) and isinstance(
        curr.ctx, (ast.Store, ast.Del)
    ):
        mut_attr = _extract_receiver_root_attr(curr, rec_param)
        if mut_attr is not None:
            stored_attrs.add(mut_attr)
        return False
    if isinstance(curr, ast.Call):
        return _record_receiver_call_mutation(
            curr,
            rec_param,
            stored_attrs,
            attr_val_ids,
            shadowed_builtins=shadowed_builtins,
        )
    return False


def _has_staticmethod_decorator(node: Optional[ast.AST]) -> bool:
    """Returns True if node carries a @staticmethod decorator."""
    return any(
        (isinstance(d, ast.Name) and d.id == "staticmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
        for d in getattr(node, "decorator_list", ())
    )


def _extract_method_receiver_param(
    fn_node: ast.AST,
    *,
    is_static: bool = False,
) -> Tuple[Optional[str], bool]:
    """Returns (receiver_param_name, is_vararg_receiver) for a method or lambda."""
    if is_static:
        return None, False
    fn_args = getattr(fn_node, "args", None)
    if not isinstance(fn_args, ast.arguments):
        return None, False
    pos_args = fn_args.posonlyargs + fn_args.args
    if pos_args:
        return pos_args[0].arg, False
    if fn_args.vararg is not None:
        return fn_args.vararg.arg, True
    return None, False


def _extract_callable_receiver_usage(
    fn_node: ast.AST,
    *,
    is_static: bool = False,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[frozenset[str], frozenset[str], bool]:
    """Extracts receiver usage, treating any access to a vararg receiver as a bare escape."""
    rec_param, is_vararg = _extract_method_receiver_param(fn_node, is_static=is_static)
    if rec_param is None:
        return frozenset(), frozenset(), False
    loaded, stored, bare_escape = _extract_receiver_usage_in_method(
        fn_node, rec_param, shadowed_builtins=shadowed_builtins
    )
    if is_vararg and (loaded or stored or bare_escape):
        return frozenset(), frozenset(), True
    return loaded, stored, bare_escape


_RECEIVER_TRACKED_BUILTINS: frozenset[str] = (
    _RECEIVER_ATTR_BUILTINS | _ALIAS_READONLY_CALL_BUILTINS
)


def _push_nested_receiver_scope(
    curr: ast.AST,
    rec_param: str,
    cur_shadowed: frozenset[str],
    stack: List[Tuple[ast.AST, frozenset[str]]],
    visited_nodes: List[Tuple[ast.AST, frozenset[str]]],
) -> bool:
    """Handles nested function/lambda/comprehension scoping for receiver usage; returns True if handled."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        extra_shadowed: Set[str] = set()
        sub_args = getattr(curr, "args", None)
        if isinstance(sub_args, ast.arguments):
            sub_arg_names = _extract_arg_names(sub_args)
            if rec_param in sub_arg_names:
                stack.extend((h, cur_shadowed) for h in iter_nested_scope_header_nodes(curr))
                return True
            extra_shadowed.update(sub_arg_names & _RECEIVER_TRACKED_BUILTINS)
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sub_locals = _extract_single_node_local_names(curr, include_globals=True)
            sub_tp_names = set(_extract_type_param_names(curr))
            if rec_param in sub_locals or rec_param in sub_tp_names:
                stack.extend((h, cur_shadowed) for h in iter_nested_scope_header_nodes(curr))
                return True
            extra_shadowed.update((sub_locals | sub_tp_names) & _RECEIVER_TRACKED_BUILTINS)
        inner_shadowed = (
            cur_shadowed | extra_shadowed if extra_shadowed else cur_shadowed
        )
        visited_nodes.append((curr, cur_shadowed))
        body_nodes = (
            [curr.body] if isinstance(curr.body, ast.AST) else list(curr.body)
        )
        for stmt in reversed(body_nodes):
            stack.append((stmt, inner_shadowed))
        for hdr in reversed(
            [
                *iter_nested_scope_header_nodes(curr),
                *getattr(curr, "type_params", ()),
            ]
        ):
            stack.append((hdr, cur_shadowed))
        return True
    if isinstance(curr, _COMPREHENSION_TYPES):
        comp_targets = {
            n
            for gen in curr.generators
            for n in _extract_store_names_in_node(gen.target)
        }
        if rec_param in comp_targets:
            if curr.generators:
                stack.append((curr.generators[0].iter, cur_shadowed))
            return True
        comp_shadowed = comp_targets & _RECEIVER_TRACKED_BUILTINS
        if comp_shadowed:
            inner_shadowed = cur_shadowed | comp_shadowed
            visited_nodes.append((curr, cur_shadowed))
            first_gen = curr.generators[0] if curr.generators else None
            for child in reversed(list(ast.iter_child_nodes(curr))):
                if child is first_gen and first_gen is not None:
                    for sub in reversed(list(ast.iter_child_nodes(first_gen))):
                        stack.append(
                            (
                                sub,
                                cur_shadowed
                                if sub is first_gen.iter
                                else inner_shadowed,
                            )
                        )
                else:
                    stack.append((child, inner_shadowed))
            return True
    return False


def _extract_receiver_usage_in_method(
    method_node: ast.AST,
    rec_param: str,
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[frozenset[str], frozenset[str], bool]:
    """Returns (loaded_attrs, stored_attrs, bare_escape) for rec_param inside method_node."""
    loaded_attrs: Set[str] = set()
    stored_attrs: Set[str] = set()
    bare_escape = False
    attr_val_ids: Set[int] = set()
    visited_nodes: List[Tuple[ast.AST, frozenset[str]]] = []
    eff_shadowed: Set[str] = set(shadowed_builtins) & _RECEIVER_TRACKED_BUILTINS
    m_args = getattr(method_node, "args", None)
    if isinstance(m_args, ast.arguments):
        eff_shadowed.update(_extract_arg_names(m_args) & _RECEIVER_TRACKED_BUILTINS)
    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        eff_shadowed.update(
            _extract_single_node_local_names(method_node, include_globals=True)
            & _RECEIVER_TRACKED_BUILTINS
        )
    root_shadowed = frozenset(eff_shadowed)
    raw_body = getattr(method_node, "body", ())
    if isinstance(raw_body, ast.AST):
        stack: List[Tuple[ast.AST, frozenset[str]]] = [(raw_body, root_shadowed)]
    elif isinstance(raw_body, (list, tuple)):
        stack = [(stmt, root_shadowed) for stmt in raw_body]
    else:
        stack = []
    while stack:
        curr, cur_shadowed = stack.pop()
        if _push_nested_receiver_scope(
            curr, rec_param, cur_shadowed, stack, visited_nodes
        ):
            continue
        visited_nodes.append((curr, cur_shadowed))
        if _record_receiver_container_mutations(
            curr,
            rec_param,
            loaded_attrs,
            stored_attrs,
            attr_val_ids,
            shadowed_builtins=cur_shadowed,
        ):
            bare_escape = True
        if (
            isinstance(curr, ast.Attribute)
            and isinstance(curr.value, ast.Name)
            and curr.value.id == rec_param
        ):
            attr_val_ids.add(id(curr.value))
            if isinstance(curr.ctx, ast.Load):
                loaded_attrs.add(curr.attr)
            else:
                stored_attrs.add(curr.attr)
        elif (
            isinstance(curr, ast.Name)
            and curr.id == rec_param
            and isinstance(curr.ctx, ast.Load)
            and id(curr) not in attr_val_ids
        ):
            bare_escape = True
        for child in reversed(list(ast.iter_child_nodes(curr))):
            stack.append((child, cur_shadowed))
    if (loaded_attrs & _RECEIVER_ESCAPE_ATTRS) or (
        stored_attrs & _RECEIVER_ESCAPE_ATTRS
    ):
        bare_escape = True
    if loaded_attrs and not bare_escape:
        alias_to_attrs = _collect_receiver_local_aliases(visited_nodes, rec_param)
        _record_aliased_receiver_mutations(
            visited_nodes,
            rec_param,
            alias_to_attrs,
            stored_attrs,
        )
    return frozenset(loaded_attrs), frozenset(stored_attrs), bare_escape


class _ClassAttrClosureEntry(NamedTuple):
    attr_names: frozenset[str]
    reads: frozenset[str]
    muts: frozenset[str]
    rec_loaded_attrs: frozenset[str]
    rec_stored_attrs: frozenset[str]
    rec_bare_escape: bool
    referenced_names: frozenset[str] = frozenset()


def _collect_class_attr_closure_entries(
    cls_node: ast.ClassDef,
    info_sibling_closures: Sequence[Tuple[Optional[str], frozenset[str], frozenset[str]]],
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> List[_ClassAttrClosureEntry]:
    """Collects non-method class-attribute closure entries and lambda receiver usages."""
    info_by_name: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for s_name, s_reads, s_muts in info_sibling_closures:
        if s_name is not None:
            r_set, m_set = info_by_name.setdefault(s_name, (set(), set()))
            r_set.update(s_reads)
            m_set.update(s_muts)
    entries: List[_ClassAttrClosureEntry] = []
    for attr_stmt in _iter_class_body_non_method_stmts(cls_node):
        for _is_reb, c_targets, c_values in _iter_carrier_target_entries(attr_stmt):
            c_calls: Set[str] = set()
            c_reads, c_muts, c_loads = _extract_carrier_value_closure_and_load_info(
                c_values, called_names_out=c_calls
            )
            c_refs = c_loads | c_calls
            for ref in c_refs:
                if ref in info_by_name:
                    c_reads.update(info_by_name[ref][0])
                    c_muts.update(info_by_name[ref][1])
            l_load: Set[str] = set()
            l_store: Set[str] = set()
            l_bare = False
            for val in c_values:
                for sub in ast.walk(val):
                    if isinstance(sub, ast.Lambda):
                        ll, ls, lb = _extract_callable_receiver_usage(
                            sub, shadowed_builtins=shadowed_builtins
                        )
                        l_load.update(ll)
                        l_store.update(ls)
                        l_bare = l_bare or lb
            entries.append(
                _ClassAttrClosureEntry(
                    frozenset(c_targets),
                    frozenset(c_reads),
                    frozenset(c_muts),
                    frozenset(l_load),
                    frozenset(l_store),
                    l_bare,
                    frozenset(c_refs),
                )
            )
    return entries


def _has_non_plain_method_decorator(node: ast.AST) -> bool:
    """Returns True if node carries any decorator other than @staticmethod or @classmethod."""
    for dec in getattr(node, "decorator_list", ()):
        if (
            isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod")
        ) or (
            isinstance(dec, ast.Attribute)
            and dec.attr in ("staticmethod", "classmethod")
        ):
            continue
        return True
    return False


def _method_is_reachable_via_receiver(
    method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    m_store: frozenset[str],
    loaded_attrs: AbstractSet[str],
    stored_attrs: AbstractSet[str],
) -> bool:
    """Returns True when receiver attribute reads or writes can invoke or share state with method_node."""
    if loaded_attrs and (
        method_node.name in loaded_attrs
        or method_node.name in ("__getattribute__", "__getattr__")
        or bool(m_store & loaded_attrs)
    ):
        return True
    if not stored_attrs:
        return False
    if method_node.name in ("__setattr__", "__delattr__"):
        return True
    if method_node.name in stored_attrs and _has_non_plain_method_decorator(method_node):
        return True
    return any(
        isinstance(dec, ast.Attribute)
        and dec.attr in ("setter", "deleter")
        and isinstance(dec.value, ast.Name)
        and dec.value.id in stored_attrs
        for dec in method_node.decorator_list
    )


def _collect_method_receiver_sibling_closures(
    fn_node: ast.AST,
    enc_child: ast.AST,
    rec_param: str,
    info_sibling_closures: Sequence[Tuple[Optional[str], frozenset[str], frozenset[str]]],
    receiver_external_closures: Tuple[frozenset[str], frozenset[str]] = (
        frozenset(),
        frozenset(),
    ),
    *,
    shadowed_builtins: AbstractSet[str] = frozenset(),
) -> Tuple[Tuple[str, frozenset[str], frozenset[str]], ...]:
    """Computes sibling closure reads and mutations reachable via rec_param on enc_child's class."""
    loaded_attrs, stored_attrs, bare_escape = _extract_receiver_usage_in_method(
        enc_child, rec_param, shadowed_builtins=shadowed_builtins
    )
    if not loaded_attrs and not stored_attrs and not bare_escape:
        return ()
    cls_node = _find_enclosing_method_class_node(fn_node, enc_child)
    if cls_node is None:
        return ()
    reads: Set[str] = set(receiver_external_closures[0])
    muts: Set[str] = set(receiver_external_closures[1])
    if cls_node.bases or cls_node.keywords:
        for s_name, s_reads, s_muts in info_sibling_closures:
            if s_name == cls_node.name:
                reads.update(s_reads)
                muts.update(s_muts)
        if not reads and not muts:
            return ()
        return ((rec_param, frozenset(reads), frozenset(muts)),)
    sib_methods = [m for m in _iter_class_body_methods(cls_node) if m is not enc_child]
    method_usages: Dict[int, Tuple[frozenset[str], frozenset[str], bool]] = {}
    for m in sib_methods:
        is_stat = _has_staticmethod_decorator(m)
        method_usages[id(m)] = _extract_callable_receiver_usage(
            m, is_static=is_stat, shadowed_builtins=shadowed_builtins
        )
    attr_entries = _collect_class_attr_closure_entries(
        cls_node, info_sibling_closures, shadowed_builtins=shadowed_builtins
    )
    if any(m_bare for _ml, _ms, m_bare in method_usages.values()) or any(
        entry.rec_bare_escape for entry in attr_entries
    ):
        bare_escape = True
    reachable_attrs: Set[str] = set(loaded_attrs)
    reachable_stored_attrs: Set[str] = set(stored_attrs)
    reachable_method_ids: Set[int] = set()
    reachable_attr_idxs: Set[int] = set()
    if not bare_escape:
        changed = True
        while changed:
            changed = False
            for m in sib_methods:
                mid = id(m)
                if mid in reachable_method_ids:
                    continue
                m_load, m_store, m_bare = method_usages[mid]
                if _method_is_reachable_via_receiver(
                    m, m_store, reachable_attrs, reachable_stored_attrs
                ):
                    reachable_method_ids.add(mid)
                    changed = True
                    if m_bare:
                        bare_escape = True
                        break
                    reachable_attrs.update(m_load)
                    reachable_stored_attrs.update(m_store)
            if bare_escape:
                break
            for idx, entry in enumerate(attr_entries):
                if idx in reachable_attr_idxs:
                    continue
                if (
                    bool(entry.attr_names & reachable_attrs)
                    or bool(entry.attr_names & reachable_stored_attrs)
                    or bool(entry.rec_stored_attrs & reachable_attrs)
                ):
                    reachable_attr_idxs.add(idx)
                    changed = True
                    if entry.rec_bare_escape:
                        bare_escape = True
                        break
                    reachable_attrs.update(entry.rec_loaded_attrs | entry.referenced_names)
                    reachable_stored_attrs.update(entry.rec_stored_attrs)
            if bare_escape:
                break
    active_methods = (
        sib_methods
        if bare_escape
        else [m for m in sib_methods if id(m) in reachable_method_ids]
    )
    active_attrs = (
        attr_entries
        if bare_escape
        else [attr_entries[i] for i in reachable_attr_idxs]
    )
    for m in active_methods:
        reads.update(_extract_cached_scope_free_reads(m))
        m_muts, m_dels = _extract_closure_nonlocal_mutations_and_deletions(m)
        muts.update(m_muts | m_dels)
    for entry in active_attrs:
        reads.update(entry.reads)
        muts.update(entry.muts)
    if not reads and not muts:
        return ()
    return ((rec_param, frozenset(reads), frozenset(muts)),)


def _attach_receiver_closures_to_info(
    child_info: _EnclosingFnCellInfo,
    rec_param: str,
    extra_sibs: Sequence[Tuple[str, frozenset[str], frozenset[str]]],
) -> _EnclosingFnCellInfo:
    """Attaches receiver sibling closures and post-reads to an inner enclosing cell summary."""
    extra_reads = (
        set().union(*(r for _, r, _ in extra_sibs))
        if rec_param in child_info.post_reads
        else set()
    )
    return child_info._replace(
        post_reads=frozenset(set(child_info.post_reads) | extra_reads),
        sibling_closures=(*child_info.sibling_closures, *extra_sibs),
        all_sibling_nonlocal_mutations=frozenset(
            set(child_info.all_sibling_nonlocal_mutations)
            | set().union(*(m for _, _, m in extra_sibs))
        ),
    )


def _collect_propagated_prefix_closures(
    chain_infos: Sequence[_EnclosingFnCellInfo],
    upto_idx: int,
) -> Tuple[Tuple[Optional[str], frozenset[str], frozenset[str]], ...]:
    """Propagates unshadowed sibling closures and alias/mutation edges across chain_infos[:upto_idx + 1]."""
    sibs: List[Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]] = []
    edges: List[Tuple[str, AbstractSet[str]]] = []
    mut_edges: List[Tuple[str, AbstractSet[str]]] = []
    for j in range(upto_idx + 1):
        sub_info = chain_infos[j]
        inner_shadowed: Set[str] = (
            set().union(
                *(chain_infos[k].fn_locals for k in range(j + 1, upto_idx + 1))
            )
            if j < upto_idx
            else set()
        )
        for s_name, s_reads, s_muts in sub_info.sibling_closures:
            eff_name = (
                None if (s_name is not None and s_name in inner_shadowed) else s_name
            )
            sibs.append((eff_name, s_reads - inner_shadowed, s_muts - inner_shadowed))
        for t_name, srcs in sub_info.alias_edges:
            if t_name not in inner_shadowed:
                edges.append((t_name, srcs - inner_shadowed))
        for t_name, srcs in sub_info.mutation_edges:
            if t_name not in inner_shadowed:
                mut_edges.append((t_name, srcs - inner_shadowed))
    propagated = _propagate_sibling_closures(
        sibs, alias_edges=edges, mutation_edges=mut_edges
    )
    return tuple((n, frozenset(r), frozenset(m)) for n, r, m in propagated)


def _resolve_enclosing_receiver_shadowed_builtins(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST],
    enclosing_scopes: Sequence[ast.AST],
) -> frozenset[str]:
    """Returns receiver-attr and read-only builtin names shadowed in fn_node or its enclosing scopes."""
    enc_scope_ctx = _get_fn_scope_context(
        fn_node, mod_tree=mod_tree, enclosing_scopes=enclosing_scopes
    )
    if enc_scope_ctx.has_star_import:
        return _RECEIVER_TRACKED_BUILTINS
    return frozenset(enc_scope_ctx.shadowed_names & _RECEIVER_TRACKED_BUILTINS)
