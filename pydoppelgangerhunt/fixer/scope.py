"""Lexical variable scope analysis, AST visitor, and control flow hazard tracking."""

from __future__ import annotations

import ast
import builtins
import logging
import re
import textwrap
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cache import _get_or_parse_tree
from pydoppelgangerhunt.fixer._df_cell_hazards import (
    _UnitClosureHazardContext,
    _find_enclosing_chain_definite_pre_bound,
    _find_enclosing_function_chain,
    _has_severed_closure_cell_hazard,
    _resolve_unit_col_bounds,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _extract_comprehension_walrus_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    iter_nested_scope_header_nodes,
)
from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names
from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor, _UnitLiveInVisitor
from pydoppelgangerhunt.reporters import extract_unit_source_code
from pydoppelgangerhunt.fixer.source import (
    _SUBROUTINE_KINDS,
    _WHOLE_UNIT_KINDS,
    _get_tree_scope_intervals,
    _load_unit_file_text,
    _slice_unit_token_lines,
    _slice_utf8_line,
    dedent_source_text,
    find_enclosing_function_is_async,
    is_subroutine_unit,
    parse_unit_coord,
    resolve_unit_declared_outputs,
    split_source_lines,
)

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


_SYNTHETIC_WRAPPER_NAME: str = "__pdh_wrapper__"


def _is_mangled_name(name: str) -> bool:
    """Checks if an identifier is subject to Python private name mangling (__foo, not __foo__)."""
    return name.startswith("__") and not name.endswith("__") and len(name) > 2


def _unfold_receiver_attribute(
    node: ast.AST,
    receiver_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Unfolds chained attribute access on receiver parameters (e.g. self.config.timeout -> 'self.config.timeout')."""
    parts: List[str] = []
    curr: ast.AST = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    valid_receivers = tuple(receiver_names) if receiver_names is not None else ("self", "cls")
    if isinstance(curr, ast.Name) and curr.id in valid_receivers:
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


def _normalize_receiver_attr_name(
    attr: str,
    receiver_param: Optional[str] = None,
) -> str:
    """Normalizes receiver attribute prefix (e.g. 'self.x', 'this.x', 'cls.x') to canonical '<rec>.x'."""
    pfxs = ["self.", "cls."]
    if receiver_param:
        pfxs.append(f"{receiver_param}.")
    for pfx in pfxs:
        if attr.startswith(pfx):
            return "<rec>." + attr[len(pfx):]
    return attr


def _normalize_receiver_attrs(
    attrs: Sequence[str],
    receiver_param: Optional[str] = None,
) -> Set[str]:
    """Normalizes a collection of receiver attribute strings to canonical '<rec>.*' form."""
    return {_normalize_receiver_attr_name(a, receiver_param) for a in attrs}

def _extract_arg_names(args: ast.arguments) -> Set[str]:
    """Extracts all parameter names from an ast.arguments node."""
    arg_set = {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
    if args.vararg:
        arg_set.add(args.vararg.arg)
    if args.kwarg:
        arg_set.add(args.kwarg.arg)
    return arg_set


def _extract_type_param_names(node: ast.AST) -> Set[str]:
    """Extracts PEP 695 type parameter names from an AST node."""
    return {
        tp.name
        for tp in getattr(node, "type_params", [])
        if isinstance(getattr(tp, "name", None), str)
    }


def _block_contains_yield(node: ast.AST) -> bool:
    """Returns True if the node contains a Yield or YieldFrom at the current scope level."""
    stack = [node]
    while stack:
        curr = stack.pop()
        for child in ast.iter_child_nodes(curr):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                stack.append(child)
    return False


class _ScopeVisitor(ast.NodeVisitor):

    """Inspects AST loads, stores, parameters, returns, nonlocals, globals, and attrs."""

    def __init__(
        self,
        loop_offset: int = 0,
        source_lines: Optional[Sequence[str]] = None,
        is_subroutine: bool = False,
        receiver_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.loads: List[str] = []
        self.stores: List[str] = []
        self.read_before_write: List[str] = []
        self.params: List[str] = []
        self.param_details: List[Dict[str, Any]] = []
        self.returns: List[str] = []
        self.return_type: Optional[str] = None
        self.nonlocals: List[str] = []
        self.globals: List[str] = []
        self.attrs_read: List[str] = []
        self.attrs_written: List[str] = []
        self.except_vars: Set[str] = set()
        self.deleted_names: Set[str] = set()
        self.type_aliases: Set[str] = set()
        self.type_params: List[str] = []
        self.type_param_free_vars: List[str] = []
        self._scope_stack: List[Set[str]] = []
        self._type_param_scopes: List[Set[str]] = []
        self._comprehension_depth: int = 0
        self.class_depth: int = 0
        self.loop_offset: int = loop_offset
        self.loop_depth: int = 0
        self.has_return: bool = False
        self.has_return_value: bool = False
        self.has_yield: bool = False
        self.has_yield_assignment: bool = False
        self.has_try_or_with: bool = False
        self._ancestors: List[ast.AST] = []
        self.has_super: bool = False
        self.has_mangled_names: bool = False
        self.naked_breaks: int = 0
        self.naked_continues: int = 0
        self.local_imports: List[str] = []
        self.imported_names: Dict[str, str] = {}
        self.yield_expr_names: List[Tuple[str, str]] = []
        self.is_async: bool = False
        self.is_subroutine: bool = is_subroutine
        self._has_processed_top_func: bool = False
        self.source_lines: Optional[List[str]] = list(source_lines) if source_lines is not None else None
        self.receiver_names: Tuple[str, ...] = (
            tuple(dict.fromkeys(list(receiver_names) + ["self", "cls"]))
            if receiver_names is not None
            else ("self", "cls")
        )
        self.unit_top_locals: Set[str] = set()
        self._unit_top_stored_locals: Set[str] = set()

    def _init_unit_top_locals(self, target_node: ast.AST) -> None:
        self.unit_top_locals = _extract_single_node_local_names(
            target_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        )
        self._unit_top_stored_locals = _extract_single_node_local_names(
            target_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=False,
        )

    def visit_Module(self, node: ast.Module) -> None:
        if self.is_subroutine and not self.unit_top_locals:
            if (
                len(node.body) == 1
                and isinstance(node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body[0].name == _SYNTHETIC_WRAPPER_NAME
            ):
                wrapper_body = node.body[0].body
                if (
                    self.loop_offset > 0
                    and len(wrapper_body) == 1
                    and isinstance(wrapper_body[0], (ast.For, ast.AsyncFor, ast.While))
                ):
                    wrapper_body = wrapper_body[0].body
                target_mod = ast.Module(body=wrapper_body, type_ignores=[])
            else:
                target_mod = node
            self._init_unit_top_locals(target_mod)
        self.generic_visit(node)

    def visit(self, node: ast.AST) -> Any:
        self._ancestors.append(node)
        try:
            return super().visit(node)
        finally:
            self._ancestors.pop()

    def _record_arg(
        self,
        arg_node: Optional[ast.arg],
        default_node: Optional[ast.expr],
        kind: str,
    ) -> None:
        if arg_node is None or arg_node.arg in BUILTIN_NAMES:
            return
        arg_name = arg_node.arg
        if arg_name not in self.params:
            self.params.append(arg_name)

        type_val = None
        if arg_node.annotation is not None:
            try:
                type_val = ast.unparse(arg_node.annotation)
            except Exception as exc:
                logger.debug("Failed to unparse argument annotation %r: %s", arg_node.annotation, exc)
                type_val = None

        default_val = None
        if default_node is not None:
            try:
                default_val = ast.unparse(default_node)
            except Exception as exc:
                logger.debug("Failed to unparse default value %r: %s", default_node, exc)
                default_val = None

        prefix = "*" if kind == "vararg" else ("**" if kind == "kwarg" else "")
        self.param_details.append({
            "name": f"{prefix}{arg_name}",
            "type": type_val,
            "default": default_val,
            "kind": kind,
        })

    def _process_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if node.returns is not None:
            try:
                self.return_type = ast.unparse(node.returns)
            except Exception as exc:
                logger.debug("Failed to unparse return annotation %r: %s", node.returns, exc)
                self.return_type = None

        pos_and_plain = node.args.posonlyargs + node.args.args
        num_pos = len(pos_and_plain)
        num_defaults = len(node.args.defaults)
        first_default_idx = num_pos - num_defaults

        for i, a in enumerate(pos_and_plain):
            def_node = node.args.defaults[i - first_default_idx] if i >= first_default_idx else None
            self._record_arg(a, def_node, "pos")

        self._record_arg(node.args.vararg, None, "vararg")

        for k, kw_def in zip(node.args.kwonlyargs, node.args.kw_defaults):
            self._record_arg(k, kw_def, "kwonly")

        self._record_arg(node.args.kwarg, None, "kwarg")

    _extract_arg_names = staticmethod(_extract_arg_names)


    def _record_store_name(self, name: str, comp_depth: int = 0) -> None:
        if self._comprehension_depth > 0 and comp_depth == 0:
            self._scope_stack[-1].add(name)
            return
        eff_len = len(self._scope_stack) - comp_depth
        bound_in_intermediate_fn = eff_len > 2 and any(
            name in s
            for s in self._scope_stack[1 : -1 - comp_depth]
            if ":class_scope:" not in s
        )
        is_active_nonlocal = (
            name in self.nonlocals
            if eff_len <= 1
            else (
                f"nonlocal:{name}" in self._scope_stack[-1 - comp_depth]
                and not bound_in_intermediate_fn
            )
        )
        is_active_global = (
            name in self.globals
            if eff_len <= 1
            else f"global:{name}" in self._scope_stack[-1 - comp_depth]
        )
        if eff_len <= 1 or is_active_nonlocal:
            self.deleted_names.discard(name)
        if is_active_nonlocal:
            if (
                name not in BUILTIN_NAMES
                and name not in self.stores
                and name not in self._unit_top_stored_locals
            ):
                self.stores.append(name)
        elif is_active_global:
            pass
        elif eff_len > 1:
            target_frame = self._scope_stack[-1 - comp_depth]
            if (
                f"nonlocal:{name}" not in target_frame
                and not (comp_depth > 0 and ":class_scope:" in target_frame)
            ):
                target_frame.add(name)
        elif name not in BUILTIN_NAMES and name not in self.stores:
            self.stores.append(name)

    def _process_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if (
            self.is_subroutine
            and len(self._scope_stack) == 0
            and node.name == _SYNTHETIC_WRAPPER_NAME
        ):
            body = node.body
            if (
                self.loop_offset > 0
                and len(body) == 1
                and isinstance(body[0], (ast.For, ast.AsyncFor, ast.While))
            ):
                body = body[0].body
                self.loop_offset = 0
            if not self.unit_top_locals:
                self._init_unit_top_locals(ast.Module(body=body, type_ignores=[]))
            if isinstance(node, ast.AsyncFunctionDef):
                self.is_async = True
            for stmt in body:
                self.visit(stmt)
            return

        tp_names = _extract_type_param_names(node)
        is_top = (
            len(self._scope_stack) == 0
            and not self.is_subroutine
            and not self._has_processed_top_func
        )
        created_outer = False
        if is_top:
            self._has_processed_top_func = True
            for tp in getattr(node, "type_params", []):
                try:
                    self.type_params.append(ast.unparse(tp))
                except Exception as exc:
                    logger.debug("Failed to unparse type_param %r: %s", tp, exc)
                tp_visitor = _OuterExprVisitor()
                tp_visitor.visit(tp)
                for name in tp_visitor.ordered_names:
                    if (
                        name not in tp_names
                        and name not in BUILTIN_NAMES
                        and name not in self.type_param_free_vars
                    ):
                        self.type_param_free_vars.append(name)
            self._process_args(node)
        else:
            for dec in node.decorator_list:
                self.visit(dec)
            for default in node.args.defaults:
                self.visit(default)
            for kw_default in node.args.kw_defaults:
                if kw_default is not None:
                    self.visit(kw_default)

        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            if is_top:
                self._init_unit_top_locals(node)
                self._scope_stack.append(
                    set(self.params) | tp_names | self.unit_top_locals
                )
            else:
                for tp in getattr(node, "type_params", []):
                    self.visit(tp)
                for arg in (
                    list(getattr(node.args, "posonlyargs", []))
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                    + ([node.args.vararg] if node.args.vararg else [])
                    + ([node.args.kwarg] if node.args.kwarg else [])
                ):
                    if arg.annotation is not None:
                        self.visit(arg.annotation)
                if node.returns is not None:
                    self.visit(node.returns)
                self._record_store_name(node.name)
                if not self._scope_stack:
                    self._scope_stack.append(set(self.unit_top_locals))
                    created_outer = True
                nested_locals = _extract_single_node_local_names(
                    node,
                    include_globals=False,
                    include_nonlocals=False,
                    include_self_name=False,
                )
                self._scope_stack.append(
                    self._extract_arg_names(node.args) | tp_names | nested_locals
                )
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self._scope_stack.pop()
                if created_outer:
                    self._scope_stack.pop()
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set(self.unit_top_locals))
            created_outer = True
        self._scope_stack.append(self._extract_arg_names(node.args))
        saved_comp_depth = self._comprehension_depth
        self._comprehension_depth = 0
        try:
            self.visit(node.body)
        finally:
            self._comprehension_depth = saved_comp_depth
            self._scope_stack.pop()
            if created_outer:
                self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if len(self._scope_stack) == 0 and not self.is_subroutine:
            self.is_async = True
        self._process_func(node)

    def visit_Await(self, node: ast.Await) -> None:
        if (len(self._scope_stack) - self._comprehension_depth) <= 1:
            self.is_async = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        tp_names = _extract_type_param_names(node)
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            for tp in getattr(node, "type_params", []):
                self.visit(tp)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword)
            self._record_store_name(node.name)
            created_outer = False
            if not self._scope_stack:
                self._scope_stack.append(set(self.unit_top_locals))
                created_outer = True
            self._scope_stack.append({":class_scope:", *tp_names})
            self.class_depth += 1
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self.class_depth -= 1
                self._scope_stack.pop()
                if created_outer:
                    self._scope_stack.pop()
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_TypeAlias(self, node: ast.AST) -> None:  # pragma: no cover (py312+)
        name_node = getattr(node, "name", None)
        alias_name = (
            name_node.id
            if isinstance(name_node, ast.Name)
            else (name_node if isinstance(name_node, str) else None)
        )
        if isinstance(alias_name, str):
            is_nested = (
                len(self._scope_stack) > 0
                if self.is_subroutine
                else len(self._scope_stack) > 1
            )
            if not is_nested:
                self.type_aliases.add(alias_name)
            self._record_store_name(alias_name)
        tp_names = _extract_type_param_names(node)
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            for tp in getattr(node, "type_params", []):
                self.visit(tp)
            val_node = getattr(node, "value", None)
            if isinstance(val_node, ast.AST):
                self.visit(val_node)
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.class_depth == 0:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "super"
                and "super" not in self.stores
                and "super" not in self.params
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "super"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
            ):
                self.has_super = True
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        name = getattr(node, "name", None)
        if name and isinstance(name, str) and name != "_":
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        name = getattr(node, "name", None)
        if name and isinstance(name, str) and name != "_":
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        rest = getattr(node, "rest", None)
        if rest and isinstance(rest, str) and rest != "_":
            self._record_store_name(rest)
        self.generic_visit(node)

    def _deferred_scope_depth(self) -> int:
        return (
            sum(1 for s in self._scope_stack if ":class_scope:" not in s)
            - self._comprehension_depth
        )

    def _record_load_name(self, name: str) -> None:
        is_inner = any(
            name in s
            for s in self._scope_stack[1:]
            if ":class_scope:" not in s or s is self._scope_stack[-1]
        ) or any(name in s for s in self._type_param_scopes)
        if not is_inner and name not in BUILTIN_NAMES:
            if name not in self.loads:
                self.loads.append(name)
            if (
                self._deferred_scope_depth() <= 1
                and name not in self.params
                and name not in self.stores
                and name not in self.read_before_write
            ):
                self.read_before_write.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.id):
            self.has_mangled_names = True
        if isinstance(node.ctx, ast.Load):
            self._record_load_name(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._record_store_name(node.id)
        elif isinstance(node.ctx, ast.Del):
            if (
                self._deferred_scope_depth() <= 1
                and self.class_depth == 0
            ):
                self._record_load_name(node.id)
            if len(self._scope_stack) <= 1 or (
                f"nonlocal:{node.id}" in self._scope_stack[-1]
                and not any(
                    node.id in s
                    for s in self._scope_stack[1:-1]
                    if ":class_scope:" not in s
                )
            ):
                self.deleted_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.attr):
            self.has_mangled_names = True
        attr_name = _unfold_receiver_attribute(node, receiver_names=self.receiver_names)
        if attr_name is not None:
            receiver_id = attr_name.split(".", 1)[0]
            is_inner = any(
                receiver_id in s
                for s in self._scope_stack[1:]
                if ":class_scope:" not in s or s is self._scope_stack[-1]
            )
            if not is_inner and self.class_depth == 0:
                if isinstance(node.ctx, ast.Load):
                    if attr_name not in self.attrs_read:
                        self.attrs_read.append(attr_name)
                elif isinstance(node.ctx, (ast.Store, ast.Del)):
                    if attr_name not in self.attrs_written:
                        self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name and isinstance(node.name, str):
            self.except_vars.add(node.name)
            self._record_store_name(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # AugAssign both loads and stores the target
        if isinstance(node.target, ast.Name):
            self._record_load_name(node.target.id)
            self._record_store_name(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            attr_name = _unfold_receiver_attribute(node.target, receiver_names=self.receiver_names)
            if attr_name is not None:
                receiver_id = attr_name.split(".", 1)[0]
                is_inner = any(
                    receiver_id in s
                    for s in self._scope_stack[1:]
                    if ":class_scope:" not in s or s is self._scope_stack[-1]
                )
                if not is_inner and self.class_depth == 0:
                    if attr_name not in self.attrs_read:
                        self.attrs_read.append(attr_name)
                    if attr_name not in self.attrs_written:
                        self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self._record_store_name(node.target.id)
            else:
                self.visit(node.target)
        elif not isinstance(node.target, ast.Name):
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._record_store_name(
                node.target.id, comp_depth=self._comprehension_depth
            )

    def _visit_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp],
    ) -> None:
        if node.generators:
            self.visit(node.generators[0].iter)
        self._comprehension_depth += 1
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set(self.unit_top_locals))
            created_outer = True
        inner_vars: Set[str] = set()
        self._scope_stack.append(inner_vars)
        if (
            any(getattr(g, "is_async", False) for g in node.generators)
            and (len(self._scope_stack) - self._comprehension_depth) <= 1
        ):
            self.is_async = True
        try:
            for gen in node.generators:
                if gen is not node.generators[0]:
                    self.visit(gen.iter)
                for sub in (gen.target, *gen.ifs):
                    self.visit(sub)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._scope_stack.pop()
            if created_outer:
                self._scope_stack.pop()
            self._comprehension_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        eff_len = len(self._scope_stack) - self._comprehension_depth
        enclosing_frames = self._scope_stack[: -1 - self._comprehension_depth]
        for name in node.names:
            is_unit_bound = (
                any(
                    name in s and ":class_scope:" not in s
                    for s in enclosing_frames
                )
                or name in self.stores
                or name in self.params
                or name in self.unit_top_locals
            )
            if (eff_len <= 1 or not is_unit_bound) and name not in self.nonlocals:
                self.nonlocals.append(name)
            if eff_len > 1:
                self._scope_stack[-1 - self._comprehension_depth].add(
                    f"nonlocal:{name}"
                )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        eff_len = len(self._scope_stack) - self._comprehension_depth
        for name in node.names:
            if eff_len <= 1:
                if name not in self.globals:
                    self.globals.append(name)
            else:
                self._scope_stack[-1 - self._comprehension_depth].add(
                    f"global:{name}"
                )
        self.generic_visit(node)

    def _visit_loop(self, node: Union[ast.For, ast.AsyncFor, ast.While]) -> None:
        self.loop_depth += 1
        try:
            if isinstance(node, (ast.For, ast.AsyncFor)):
                self.visit(node.iter)
                self.visit(node.target)
            else:
                self.visit(node.test)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)
        finally:
            self.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
        self._visit_loop(node)

    def visit_With(self, node: ast.With) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
            if _block_contains_yield(node):
                self.has_try_or_with = True
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.AST) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _record_loop_control(self, is_break: bool) -> None:
        if self.loop_depth <= self.loop_offset:
            if is_break:
                self.naked_breaks += 1
            else:
                self.naked_continues += 1

    def visit_Break(self, node: ast.Break) -> None:
        self._record_loop_control(is_break=True)
        self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        self._record_loop_control(is_break=False)
        self.generic_visit(node)

    def _record_yield_expr(self, kind: str, node: Union[ast.Yield, ast.YieldFrom]) -> None:
        if len(self._scope_stack) <= 1:
            self.has_yield = True
            if node.value is not None:
                if isinstance(node.value, ast.Name):
                    self.yield_expr_names.append((kind, node.value.id))
                elif isinstance(node.value, ast.Constant) and node.value.value is not None:
                    self.yield_expr_names.append((kind, f":literal:{type(node.value.value).__name__}"))
                elif (
                    kind == "yield_from"
                    and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
                    and node.value.elts
                ):
                    first_elt = node.value.elts[0]
                    if isinstance(first_elt, ast.Constant) and first_elt.value is not None:
                        t_name = type(first_elt.value).__name__
                        if all(
                            isinstance(e, ast.Constant)
                            and e.value is not None
                            and type(e.value).__name__ == t_name
                            for e in node.value.elts
                        ):
                            self.yield_expr_names.append((kind, f":literal:{t_name}"))

    def _inspect_yield_assignment(self) -> None:
        if len(self._scope_stack) <= 1 and len(self._ancestors) >= 2:
            immediate_parent = self._ancestors[-2]
            if not isinstance(immediate_parent, ast.Expr):
                self.has_yield_assignment = True

    def visit_Yield(self, node: ast.Yield) -> None:
        self._record_yield_expr("yield", node)
        if len(self._scope_stack) <= 1:
            self._inspect_yield_assignment()
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._record_yield_expr("yield_from", node)
        self.generic_visit(node)


    def visit_Return(self, node: ast.Return) -> None:
        if len(self._scope_stack) <= 1:
            self.has_return = True
            if node.value is not None:
                self.has_return_value = True
                if isinstance(node.value, ast.Name):
                    if node.value.id not in self.returns:
                        self.returns.append(node.value.id)
                elif isinstance(node.value, ast.Tuple):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Name) and elt.id not in self.returns:
                            self.returns.append(elt.id)
        self.generic_visit(node)

    def _record_import_node(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        stmt: Optional[str] = None
        if self.source_lines is not None:
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            if lineno is not None and end_lineno is not None:
                if 1 <= lineno <= len(self.source_lines) and 1 <= end_lineno <= len(self.source_lines):
                    raw_slice = self.source_lines[lineno - 1 : end_lineno]
                    raw_str = "".join(raw_slice).strip()
                    if raw_str:
                        stmt = raw_str

        if stmt is None:
            try:
                stmt = ast.unparse(node)
            except Exception as exc:
                logger.debug("Failed to unparse import node %r: %s", node, exc)
                return

        is_nested = (
            len(self._scope_stack) > 0
            if self.is_subroutine
            else len(self._scope_stack) > 1
        )
        if not is_nested and stmt not in self.local_imports:
            self.local_imports.append(stmt)
        for alias in node.names:
            if isinstance(node, ast.Import):
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            else:
                bound_name = alias.asname or alias.name
            if bound_name != "*":
                if not is_nested:
                    self.imported_names[bound_name] = stmt
                self._record_store_name(bound_name)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_node(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_node(node)
        self.generic_visit(node)


def _walrus_in_sequence(
    items: Sequence[Optional[ast.AST]],
    is_conditional: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Evaluates walrus assignments sequentially across expressions with short-circuiting tail conditions."""
    definite: Set[str] = set()
    conditional: Set[str] = set()
    for idx, item in enumerate(items):
        if item is not None:
            d, c = _walrus_assignment_in_expr(item, is_conditional if idx == 0 else True)
            definite.update(d)
            conditional.update(c)
    return definite, conditional


def _walrus_assignment_in_expr(
    expr: Optional[ast.AST],
    is_conditional: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Analyzes walrus assignments in an expression to distinguish definite from conditional stores."""
    definite: Set[str] = set()
    conditional: Set[str] = set()
    if expr is None:
        return definite, conditional

    if isinstance(expr, ast.NamedExpr) and isinstance(expr.target, ast.Name):
        if is_conditional:
            conditional.add(expr.target.id)
        else:
            definite.add(expr.target.id)

    if isinstance(expr, ast.BoolOp):
        d_seq, c_seq = _walrus_in_sequence(expr.values, is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        return definite, conditional - definite

    if isinstance(expr, ast.IfExp):
        d_t, c_t = _walrus_assignment_in_expr(expr.test, is_conditional)
        d_b, c_b = _walrus_assignment_in_expr(expr.body, True)
        d_o, c_o = _walrus_assignment_in_expr(expr.orelse, True)
        definite.update(d_t)
        conditional.update(c_t | d_b | c_b | d_o | c_o)
        return definite, conditional - definite

    if isinstance(expr, ast.Compare):
        d_l, c_l = _walrus_assignment_in_expr(expr.left, is_conditional)
        definite.update(d_l)
        conditional.update(c_l)
        d_seq, c_seq = _walrus_in_sequence(expr.comparators, is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        return definite, conditional - definite

    if isinstance(expr, ast.Lambda):
        d_l, c_l = _walrus_assignment_in_expr(expr.body, True)
        conditional.update(d_l | c_l)
        return definite, conditional - definite

    if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        d_seq, c_seq = _walrus_in_sequence([g.iter for g in expr.generators], is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        for child in ast.iter_child_nodes(expr):
            if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) and child is expr.elt:
                d_c, c_c = _walrus_assignment_in_expr(child, True)
                conditional.update(d_c | c_c)
            elif isinstance(expr, ast.DictComp) and child in (expr.key, expr.value):
                d_c, c_c = _walrus_assignment_in_expr(child, True)
                conditional.update(d_c | c_c)
            elif isinstance(child, ast.comprehension):
                for if_expr in child.ifs:
                    d_c, c_c = _walrus_assignment_in_expr(if_expr, True)
                    conditional.update(d_c | c_c)
        return definite, conditional - definite

    for child in ast.iter_child_nodes(expr):
        target_child = child.value if isinstance(child, ast.keyword) else child
        if isinstance(target_child, ast.AST):
            d_ch, c_ch = _walrus_assignment_in_expr(target_child, is_conditional)
            definite.update(d_ch)
            conditional.update(c_ch)

    return definite, conditional - definite


def _is_irrefutable_pattern(pattern: Optional[ast.AST]) -> bool:  # pragma: no cover (py310+)
    """Recursively determines if a pattern matching AST node unconditionally matches any subject."""
    if pattern is None:
        return True
    if type(pattern).__name__ == "MatchAs":
        sub_pat = getattr(pattern, "pattern", None)
        return sub_pat is None or _is_irrefutable_pattern(sub_pat)
    if type(pattern).__name__ == "MatchOr":
        patterns = getattr(pattern, "patterns", [])
        return any(_is_irrefutable_pattern(p) for p in patterns)
    return False


def _is_irrefutable_case(case: ast.AST) -> bool:  # pragma: no cover (py310+)
    """Returns True if the match case has no guard and an irrefutable pattern."""
    guard = getattr(case, "guard", None)
    pattern = getattr(case, "pattern", None)
    if guard is not None or pattern is None:
        return False
    return _is_irrefutable_pattern(pattern)


def _block_terminates(statements: Sequence[ast.stmt]) -> bool:
    """Returns True if the statement sequence unconditionally terminates via return or raise."""
    for stmt in statements:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if (
                stmt.orelse
                and _block_terminates(stmt.body)
                and _block_terminates(stmt.orelse)
            ):
                return True
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            if _block_terminates(stmt.body):
                return True
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            cases = getattr(stmt, "cases", [])
            has_irrefutable = any(_is_irrefutable_case(c) for c in cases)
            if has_irrefutable and cases and all(_block_terminates(c.body) for c in cases):
                return True
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            f_body = getattr(stmt, "finalbody", [])
            if f_body and _block_terminates(f_body):
                return True
            handlers = getattr(stmt, "handlers", [])
            o_stmts = getattr(stmt, "orelse", [])
            body_terminates = _block_terminates(getattr(stmt, "body", [])) or (
                bool(o_stmts) and _block_terminates(o_stmts)
            )
            if (
                handlers
                and body_terminates
                and all(_block_terminates(getattr(h, "body", [])) for h in handlers)
            ):
                return True
    return False


def _extract_deleted_names(statements: Sequence[ast.AST]) -> Set[str]:
    """Finds all variable names deleted within a statement sequence without recursing into nested functions."""
    deleted: Set[str] = set()
    for stmt in statements:
        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                        deleted.add(node.id)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        elif isinstance(stmt, ast.ExceptHandler):
            deleted.update(_extract_deleted_names(stmt.body))
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            for case in getattr(stmt, "cases", []):
                deleted.update(_extract_deleted_names(getattr(case, "body", [])))
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, (ast.stmt, ast.ExceptHandler)):
                    deleted.update(_extract_deleted_names([child]))
    return deleted


def _demote_deletions_to_conditional(
    deleted_names: Iterable[str],
    definite: Set[str],
    conditional: Set[str],
) -> None:
    """Discards deleted variables from definite and promotes previously definite variables to conditional."""
    for v_del in deleted_names:
        if v_del in definite:
            definite.discard(v_del)
            conditional.add(v_del)


def _analyze_block_assignment(
    statements: Sequence[ast.stmt],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str]]:
    """Analyzes a sequence of statements to determine unconditionally and conditionally assigned variables.

    Returns:
        A tuple of (definitely_assigned, conditionally_assigned) variable name sets.
    """
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()
    may_del = may_deleted if may_deleted is not None else set()
    live_in_del = live_in_deleted if live_in_deleted is not None else set()

    for stmt in statements:
        before_def = set(definite)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        definite.add(node.id)
                        may_del.discard(node.id)
                d_t, c_t = _walrus_assignment_in_expr(target)
                definite.update(d_t)
                may_del.difference_update(d_t)
                conditional.update(c_t)
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            may_del.difference_update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                if isinstance(stmt.target, ast.Name):
                    definite.add(stmt.target.id)
                    may_del.discard(stmt.target.id)
                d_val, c_val = _walrus_assignment_in_expr(stmt.value)
                definite.update(d_val)
                may_del.difference_update(d_val)
                conditional.update(c_val)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                definite.add(stmt.target.id)
                may_del.discard(stmt.target.id)
            d_t, c_t = _walrus_assignment_in_expr(stmt.target)
            definite.update(d_t)
            may_del.difference_update(d_t)
            conditional.update(c_t)
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            may_del.difference_update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.Expr):
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            may_del.difference_update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                d_del, c_del = _walrus_assignment_in_expr(target)
                definite.update(d_del)
                may_del.difference_update(d_del)
                conditional.update(c_del)
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                        if node.id not in definite:
                            live_in_del.add(node.id)
                        definite.discard(node.id)
                        conditional.discard(node.id)
                        may_del.add(node.id)
        elif isinstance(stmt, ast.Raise):
            if stmt.exc is not None:
                d_exc, c_exc = _walrus_assignment_in_expr(stmt.exc)
                definite.update(d_exc)
                may_del.difference_update(d_exc)
                conditional.update(c_exc)
            if stmt.cause is not None:
                d_cause, c_cause = _walrus_assignment_in_expr(stmt.cause)
                definite.update(d_cause)
                may_del.difference_update(d_cause)
                conditional.update(c_cause)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                d_val, c_val = _walrus_assignment_in_expr(stmt.value)
                definite.update(d_val)
                may_del.difference_update(d_val)
                conditional.update(c_val)
        elif isinstance(stmt, ast.If):
            d_test, c_test = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_test)
            may_del.difference_update(d_test)
            conditional.update(c_test)
            b_may: Set[str] = set()
            b_live: Set[str] = set()
            o_may: Set[str] = set()
            o_live: Set[str] = set()
            b_def, b_cond = _analyze_block_assignment(
                stmt.body, may_deleted=b_may, live_in_deleted=b_live
            )
            o_def, o_cond = (
                _analyze_block_assignment(
                    stmt.orelse, may_deleted=o_may, live_in_deleted=o_live
                )
                if stmt.orelse
                else (set(), set())
            )
            live_in_del.update((b_live | o_live) - definite)
            b_term = _block_terminates(stmt.body)
            o_term = _block_terminates(stmt.orelse) if stmt.orelse else False
            if stmt.orelse:
                if o_term and not b_term:
                    definite.update(b_def)
                    conditional.update((b_def | b_cond) - definite)
                    may_del.difference_update(b_def)
                    may_del.update(b_may)
                elif b_term and not o_term:
                    definite.update(o_def)
                    conditional.update((o_def | o_cond) - definite)
                    may_del.difference_update(o_def)
                    may_del.update(o_may)
                elif not b_term and not o_term:
                    both = b_def & o_def
                    definite.update(both)
                    conditional.update((b_def | b_cond | o_def | o_cond) - definite)
                    may_del.difference_update(both)
                    may_del.update(b_may | o_may)
                eff_b_del = _extract_deleted_names(stmt.body) - b_def
                eff_o_del = _extract_deleted_names(stmt.orelse) - o_def
                for v_del in eff_b_del & eff_o_del:
                    definite.discard(v_del)
                    conditional.discard(v_del)
                _demote_deletions_to_conditional(eff_b_del ^ eff_o_del, definite, conditional)
            else:
                if not b_term:
                    conditional.update((b_def | b_cond) - definite)
                    may_del.update(b_may)
                    eff_b_del = _extract_deleted_names(stmt.body) - b_def
                    _demote_deletions_to_conditional(eff_b_del, definite, conditional)
        elif isinstance(stmt, ast.While):
            d_test, c_test = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_test)
            may_del.difference_update(d_test)
            conditional.update(c_test)
            sub_stmts = list(stmt.body)
            if hasattr(stmt, "orelse") and stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            sub_may: Set[str] = set()
            sub_live: Set[str] = set()
            sub_def, sub_cond = _analyze_block_assignment(
                sub_stmts, may_deleted=sub_may, live_in_deleted=sub_live
            )
            live_in_del.update(sub_live - definite)
            may_del.update(sub_may)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(_extract_deleted_names(sub_stmts), definite, conditional)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            d_iter, c_iter = _walrus_assignment_in_expr(stmt.iter)
            definite.update(d_iter)
            may_del.difference_update(d_iter)
            conditional.update(c_iter)
            t_names: Set[str] = set()
            for p_node in ast.walk(stmt.target):
                if isinstance(p_node, ast.Name) and isinstance(p_node.ctx, ast.Store):
                    conditional.add(p_node.id)
                    t_names.add(p_node.id)
            sub_stmts = list(stmt.body)
            if hasattr(stmt, "orelse") and stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            for_may: Set[str] = set()
            for_live: Set[str] = set()
            sub_def, sub_cond = _analyze_block_assignment(
                sub_stmts, may_deleted=for_may, live_in_deleted=for_live
            )
            live_in_del.update((for_live - t_names) - definite)
            may_del.update(for_may)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(_extract_deleted_names(sub_stmts), definite, conditional)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                d_ctx, c_ctx = _walrus_assignment_in_expr(item.context_expr)
                definite.update(d_ctx)
                may_del.difference_update(d_ctx)
                conditional.update(c_ctx)
                if item.optional_vars is not None:
                    for p_node in ast.walk(item.optional_vars):
                        if isinstance(p_node, ast.Name) and isinstance(p_node.ctx, ast.Store):
                            definite.add(p_node.id)
                            may_del.discard(p_node.id)
            _analyze_block_assignment(
                stmt.body,
                definite,
                conditional,
                may_deleted=may_del,
                live_in_deleted=live_in_del,
            )
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            t_body = list(getattr(stmt, "body", []))
            t_may: Set[str] = set()
            t_live: Set[str] = set()
            t_def, t_cond = _analyze_block_assignment(
                t_body, may_deleted=t_may, live_in_deleted=t_live
            )
            live_in_del.update(t_live - definite)
            t_term = _block_terminates(t_body)
            o_stmts = list(getattr(stmt, "orelse", []))
            o_try_may: Set[str] = set()
            if o_stmts:
                o_try_live: Set[str] = set()
                o_def, o_cond = _analyze_block_assignment(
                    o_stmts, may_deleted=o_try_may, live_in_deleted=o_try_live
                )
                if not t_term:
                    live_in_del.update(o_try_live - (definite | t_def))
                o_term = _block_terminates(o_stmts)
                if not o_term:
                    try_def = t_def | o_def
                    try_cond = (t_cond | o_cond) - try_def
                    succ_may = (t_may - o_def) | o_try_may
                else:
                    try_def = set()
                    try_cond = set()
                    succ_may = set()
                    t_term = True
            else:
                try_def = t_def
                try_cond = t_cond
                succ_may = t_may

            handlers = list(getattr(stmt, "handlers", []))
            h_defs: List[Set[str]] = []
            h_dels: List[Tuple[Set[str], bool]] = []
            h_mays: List[Set[str]] = []
            all_handlers_terminate = bool(
                handlers and all(_block_terminates(getattr(h, "body", [])) for h in handlers)
            )

            for h in handlers:
                h_body = list(getattr(h, "body", []))
                if getattr(h, "type", None) is not None:
                    d_ht, c_ht = _walrus_assignment_in_expr(h.type, is_conditional=True)
                    conditional.update(d_ht | c_ht)
                h_may: Set[str] = set()
                h_live: Set[str] = set()
                h_d, h_c = _analyze_block_assignment(
                    h_body, may_deleted=h_may, live_in_deleted=h_live
                )
                h_bound = {h.name} if isinstance(getattr(h, "name", None), str) else set()
                h_may.difference_update(h_bound)
                live_in_del.update((h_live - h_bound) - definite)
                h_term = _block_terminates(h_body)
                eff_h_del = _extract_deleted_names(h_body) - h_d
                h_dels.append((eff_h_del, h_term))
                if not h_term:
                    h_defs.append(h_d)
                    h_mays.append(h_may)
                conditional.update((h_d | h_c) - definite)

            if not t_term:
                if all_handlers_terminate:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
                    may_del.difference_update(try_def)
                    may_del.update(succ_may)
                elif handlers and h_defs:
                    common = try_def.intersection(*h_defs)
                    definite.update(common)
                    conditional.update((try_def | try_cond) - definite)
                    may_del.difference_update(common)
                    may_del.update(succ_may)
                    for hm in h_mays:
                        may_del.update(hm)
                elif not handlers:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
                    may_del.difference_update(try_def)
                    may_del.update(succ_may)
            elif h_defs:
                common = set.intersection(*h_defs)
                definite.update(common)
                conditional.update(try_cond - definite)
                may_del.difference_update(common)
                for hm in h_mays:
                    may_del.update(hm)

            eff_try_del = _extract_deleted_names(t_body) - t_def
            eff_o_del = _extract_deleted_names(o_stmts) - o_def if o_stmts else set()
            succ_del = eff_try_del | eff_o_del

            surviving_dels: List[Set[str]] = []
            all_try_dels = succ_del.copy()
            if not t_term:
                surviving_dels.append(succ_del)

            for eff_h_del, h_term in h_dels:
                all_try_dels.update(eff_h_del)
                if not h_term:
                    surviving_dels.append(eff_h_del)

            if surviving_dels:
                uncond_dels = set.intersection(*surviving_dels)
                for v_del in uncond_dels:
                    definite.discard(v_del)
                    conditional.discard(v_del)
                _demote_deletions_to_conditional(all_try_dels - uncond_dels, definite, conditional)
            else:
                _demote_deletions_to_conditional(all_try_dels, definite, conditional)

            f_body = list(getattr(stmt, "finalbody", []))
            if f_body:
                _analyze_block_assignment(
                    f_body,
                    definite,
                    conditional,
                    may_deleted=may_del,
                    live_in_deleted=live_in_del,
                )
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            d_sub, c_sub = _walrus_assignment_in_expr(getattr(stmt, "subject", None))
            definite.update(d_sub)
            may_del.difference_update(d_sub)
            conditional.update(c_sub)
            case_defs: List[Set[str]] = []
            non_term_case_defs: List[Set[str]] = []
            case_conds: Set[str] = set()
            case_dels: Set[str] = set()
            non_term_case_dels: List[Set[str]] = []
            has_irrefutable_default = False
            for case in getattr(stmt, "cases", []):
                pattern_stores: Set[str] = set()
                pattern = getattr(case, "pattern", None)
                if pattern is not None:
                    for p_node in ast.walk(pattern):
                        if type(p_node).__name__ in ("MatchAs", "MatchStar"):
                            p_name = getattr(p_node, "name", None)
                            if p_name and isinstance(p_name, str) and p_name != "_":
                                pattern_stores.add(p_name)
                        elif type(p_node).__name__ == "MatchMapping":
                            p_rest = getattr(p_node, "rest", None)
                            if p_rest and isinstance(p_rest, str) and p_rest != "_":
                                pattern_stores.add(p_rest)
                guard = getattr(case, "guard", None)
                d_g: Set[str] = set()
                c_g: Set[str] = set()
                if guard is not None:
                    d_g, c_g = _walrus_assignment_in_expr(guard, is_conditional=False)
                case_init_def = pattern_stores | d_g
                c_may: Set[str] = set()
                c_live: Set[str] = set()
                c_def, c_cond = _analyze_block_assignment(
                    case.body,
                    definite=case_init_def.copy(),
                    conditional=c_g.copy() if c_g else None,
                    may_deleted=c_may,
                    live_in_deleted=c_live,
                )
                live_in_del.update((c_live - case_init_def) - definite)
                case_defs.append(c_def)
                case_conds.update(c_def | c_cond)
                c_del = _extract_deleted_names(case.body) - c_def
                case_dels.update(c_del)
                if not _block_terminates(case.body):
                    non_term_case_defs.append(c_def)
                    non_term_case_dels.append(c_del)
                    may_del.update(c_may - pattern_stores)
                if _is_irrefutable_case(case):
                    has_irrefutable_default = True
            if case_defs and has_irrefutable_default:
                if non_term_case_defs:
                    common = set.intersection(*non_term_case_defs)
                    definite.update(common)
                    may_del.difference_update(common)
                conditional.update(case_conds - definite)
            else:
                conditional.update(case_conds - definite)
            all_del = (
                {v for v in case_dels if all(v in d for d in non_term_case_dels)}
                if (has_irrefutable_default and non_term_case_dels)
                else set()
            )
            for v_del in all_del:
                definite.discard(v_del)
                conditional.discard(v_del)
            _demote_deletions_to_conditional(case_dels - all_del, definite, conditional)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definite.add(stmt.name)
            may_del.discard(stmt.name)
        elif type(stmt).__name__ == "TypeAlias":  # pragma: no cover (py312+)
            name_node = getattr(stmt, "name", None)
            if isinstance(name_node, ast.Name):
                definite.add(name_node.id)
                may_del.discard(name_node.id)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                definite.add(bound_name)
                may_del.discard(bound_name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    bound_name = alias.asname or alias.name
                    definite.add(bound_name)
                    may_del.discard(bound_name)
        elif isinstance(stmt, ast.Assert):
            d_ast, c_ast = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_ast)
            may_del.difference_update(d_ast)
            conditional.update(c_ast)
            if stmt.msg is not None:
                d_m, c_m = _walrus_assignment_in_expr(stmt.msg, is_conditional=True)
                conditional.update(d_m | c_m)

        may_del.difference_update(definite - before_def)
        if _block_terminates([stmt]):
            break

    return definite, conditional - definite


def _determine_binding_kind(
    has_instance_binding: bool,
    has_class_binding: bool,
) -> Optional[str]:
    """Determines the scope binding category from instance and class binding flags."""
    if has_instance_binding and has_class_binding:
        return "mixed"
    if has_instance_binding:
        return "instance"
    if has_class_binding:
        return "class"
    return None


def _normalize_receiver_order(
    inputs: List[str],
    has_instance_binding: bool,
    has_class_binding: bool,
    primary_receiver: Optional[str] = None,
) -> None:
    """Ensures self or cls is positioned as the primary receiver parameter in inputs."""
    def _bring_to_front(name: str) -> None:
        if name in inputs:
            inputs.remove(name)
            inputs.insert(0, name)

    if has_instance_binding and has_class_binding:
        first = primary_receiver or "self"
        second = "cls" if first == "self" else "self"
        _bring_to_front(second)
        _bring_to_front(first)
    elif has_instance_binding or has_class_binding:
        default_rec = "self" if has_instance_binding else "cls"
        target = primary_receiver if (primary_receiver and primary_receiver in inputs) else default_rec
        _bring_to_front(target)



def _rank_param_kind(
    var_name: str,
    param_map: Dict[str, str],
    receiver_names: Optional[Sequence[str]] = None,
) -> int:
    """Returns canonical parameter sorting rank: receiver=0, pos=1, vararg=2, kwonly=3, kwarg=4."""
    clean = var_name.lstrip("*")
    valid_receivers = (
        tuple(receiver_names)
        if receiver_names is not None
        else ("self", "cls", "this", "klass")
    )
    if clean in valid_receivers:
        return 0
    kind = param_map.get(clean, "pos")
    if var_name.startswith("**") or kind == "kwarg":
        return 4
    if kind == "kwonly":
        return 3
    if var_name.startswith("*") or kind == "vararg":
        return 2
    return 1


def _find_enclosing_pre_bound_names(
    mod_tree: ast.AST,
    s_line: int,
    e_line: int,
    u_start_col: Optional[int] = None,
) -> Tuple[Set[str], bool]:
    """Finds names bound before s_line in scopes enclosing [s_line, e_line] and loop containment."""
    bound_before: Set[str] = set()
    in_loop = False

    def _is_before_unit(c_line: Optional[int], c_col: Optional[int]) -> bool:
        if not isinstance(c_line, int):
            return False
        if c_line < s_line:
            return True
        return (
            c_line == s_line
            and u_start_col is not None
            and isinstance(c_col, int)
            and c_col < u_start_col
        )

    intervals = [
        (earliest_start, n_end, span, node)
        for earliest_start, n_end, span, node in _get_tree_scope_intervals(mod_tree)
        if earliest_start <= s_line and e_line <= n_end
    ]
    intervals.sort(key=lambda item: (item[0], -item[1], -item[2]))
    innermost_scope = intervals[-1][3] if intervals else mod_tree
    active_scopes: List[ast.AST] = []
    if isinstance(mod_tree, ast.Module):
        active_scopes.append(mod_tree)
    for _estart, _nend, _span, s_node in intervals:
        if isinstance(s_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            active_scopes.append(s_node)
        elif isinstance(s_node, ast.ClassDef) and s_node is innermost_scope:
            active_scopes.append(s_node)

    for scope_node in active_scopes:
        is_class_scope = isinstance(scope_node, ast.ClassDef)
        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (
                scope_node.args.posonlyargs
                + scope_node.args.args
                + scope_node.args.kwonlyargs
                + ([scope_node.args.vararg] if scope_node.args.vararg else [])
                + ([scope_node.args.kwarg] if scope_node.args.kwarg else [])
            ):
                bound_before.add(arg.arg)
        stack: List[Tuple[ast.AST, bool]] = [
            (stmt, False) for stmt in reversed(getattr(scope_node, "body", []))
        ]
        while stack:
            curr, is_branched = stack.pop()
            c_start = getattr(curr, "lineno", None)
            c_col = getattr(curr, "col_offset", None)
            c_end = getattr(curr, "end_lineno", c_start)
            c_end_col = getattr(curr, "end_col_offset", c_col)
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if _is_before_unit(c_start, c_col) and (
                    curr in active_scopes
                    or _is_before_unit(c_end, c_end_col)
                ):
                    bound_before.add(curr.name)
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(curr, ast.Lambda):
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(
                curr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if _is_before_unit(c_start, c_col) and not is_class_scope:
                    bound_before.update(_extract_comprehension_walrus_names(curr))
                continue
            if type(curr).__name__ == "TypeAlias":
                if _is_before_unit(c_start, c_col):
                    alias_name = getattr(curr, "name", None)
                    if isinstance(alias_name, ast.Name):
                        bound_before.add(alias_name.id)
                continue
            if (
                isinstance(curr, ast.AnnAssign)
                and curr.value is None
                and isinstance(curr.target, ast.Name)
            ):
                stack.append((curr.annotation, is_branched))
                continue
            if isinstance(curr, (ast.For, ast.AsyncFor, ast.While)):
                if (
                    isinstance(c_start, int)
                    and isinstance(c_end, int)
                    and c_start <= s_line
                    and e_line <= c_end
                    and (
                        c_start < s_line
                        or c_end > e_line
                        or (
                            u_start_col is not None
                            and isinstance(c_col, int)
                            and c_col < u_start_col
                        )
                    )
                ):
                    in_loop = True
            if _is_before_unit(c_start, c_col):
                if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Store):
                    bound_before.add(curr.id)
                elif (
                    isinstance(curr, ast.Name)
                    and isinstance(curr.ctx, ast.Del)
                    and not is_branched
                ):
                    bound_before.discard(curr.id)
                elif isinstance(curr, (ast.Import, ast.ImportFrom)):
                    for alias in curr.names:
                        b_name = alias.asname or alias.name.split(".", 1)[0]
                        if b_name != "*":
                            bound_before.add(b_name)
                elif (
                    isinstance(curr, ast.ExceptHandler)
                    and isinstance(curr.name, str)
                    and isinstance(c_start, int)
                    and isinstance(c_end, int)
                    and c_start <= s_line
                    and e_line <= c_end
                ):
                    bound_before.add(curr.name)
                elif type(curr).__name__ in ("MatchAs", "MatchStar", "MatchMapping"):
                    m_name = getattr(curr, "name", None) or getattr(curr, "rest", None)
                    if isinstance(m_name, str) and m_name != "_":
                        bound_before.add(m_name)
            child_branched = is_branched or (
                isinstance(
                    curr,
                    (
                        ast.If,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.Try,
                        ast.ExceptHandler,
                    ),
                )
                or type(curr).__name__ in ("TryStar", "Match")
            )
            stack.extend(
                (ch, child_branched)
                for ch in reversed(list(ast.iter_child_nodes(curr)))
            )
    return bound_before, in_loop


def _infer_missing_col_from_line_ast(
    line0: str,
    s_col: Optional[int],
    e_col: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """Infers a missing start_col or end_col from single-line AST nodes."""
    stripped = line0.lstrip()
    if not stripped:
        return s_col, e_col
    pad = len(line0[: len(line0) - len(stripped)].encode("utf-8"))
    try:
        mod = ast.parse(stripped)
    except SyntaxError:
        return s_col, e_col
    for node in ast.walk(mod):
        nc = getattr(node, "col_offset", None)
        nec = getattr(node, "end_col_offset", None)
        if not isinstance(nc, int) or not isinstance(nec, int):
            continue
        abs_s = pad + nc
        abs_e = pad + nec
        if s_col is not None and e_col is None and abs_s == s_col < abs_e:
            return s_col, abs_e
        if e_col is not None and s_col is None and abs_s < e_col == abs_e:
            return abs_s, e_col
    return s_col, e_col


def _unit_cols(
    unit: Dict[str, Any],
    unit_code_lines: Optional[Sequence[str]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Extracts (start_col, end_col) coordinates from a unit dictionary."""
    s_col = parse_unit_coord(unit, "start_col", default=None)
    if s_col is None:
        s_col = parse_unit_coord(unit, "col_offset", default=None)
    e_col = parse_unit_coord(unit, "end_col", default=None)
    if e_col is None:
        e_col = parse_unit_coord(unit, "end_col_offset", default=None)
    if (
        (s_col is None or e_col is None)
        and unit_code_lines
        and len(unit_code_lines) == 1
        and isinstance(unit.get("source_code"), str)
    ):
        snippet = unit["source_code"].strip()
        if snippet and "\n" not in snippet:
            line0 = unit_code_lines[0]
            if (
                snippet != line0.strip()
                or s_col is not None
                or e_col is not None
            ):
                start_id = snippet[0].isalnum() or snippet[0] == "_"
                end_id = snippet[-1].isalnum() or snippet[-1] == "_"
                search_idx = 0
                best_pair: Optional[Tuple[int, int]] = None
                while True:
                    pos = line0.find(snippet, search_idx)
                    if pos < 0:
                        break
                    end_pos = pos + len(snippet)
                    left_ok = not start_id or pos == 0 or not (
                        line0[pos - 1].isalnum() or line0[pos - 1] == "_"
                    )
                    right_ok = not end_id or end_pos >= len(line0) or not (
                        line0[end_pos].isalnum() or line0[end_pos] == "_"
                    )
                    if left_ok and right_ok:
                        cand_s = len(line0[:pos].encode("utf-8"))
                        cand_e = len(line0[:end_pos].encode("utf-8"))
                        if (s_col is None or cand_s == s_col) and (
                            e_col is None or cand_e == e_col
                        ):
                            best_pair = (cand_s, cand_e)
                            break
                        if (
                            best_pair is None
                            and (s_col is None or cand_e > s_col)
                            and (e_col is None or cand_s < e_col)
                        ):
                            best_pair = (cand_s, cand_e)
                    search_idx = pos + 1
                if best_pair is not None:
                    if s_col is None:
                        s_col = best_pair[0]
                    if e_col is None:
                        e_col = best_pair[1]
    if (
        (s_col is None) != (e_col is None)
        and unit_code_lines
        and len(unit_code_lines) == 1
    ):
        s_col, e_col = _infer_missing_col_from_line_ast(
            unit_code_lines[0], s_col, e_col
        )
    return s_col, e_col


def _inspect_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_tree: Optional[ast.AST] = None,
) -> Dict[str, Any]:
    cached_full_source: List[Optional[str]] = [None]
    full_source_loaded: List[bool] = [False]

    def _get_full_source() -> Optional[str]:
        if not full_source_loaded[0]:
            cached_full_source[0] = _load_unit_file_text(unit, repo_root=repo_root)
            full_source_loaded[0] = True
        return cached_full_source[0]

    unit_code_lines = extract_unit_source_code(unit, repo_root=repo_root)
    inferred_s_col, inferred_e_col = _unit_cols(unit, unit_code_lines)
    slice_unit = (
        {**unit, "start_col": inferred_s_col, "end_col": inferred_e_col}
        if inferred_s_col is not None or inferred_e_col is not None
        else unit
    )
    raw_lines = _slice_unit_token_lines(slice_unit, unit_code_lines)
    if (
        len(raw_lines) == 1
        and ";" in raw_lines[0]
        and unit.get("kind") not in ("comprehension", "complex_expr")
        and inferred_s_col is not None
        and inferred_e_col is not None
        and 0 <= inferred_s_col < inferred_e_col
    ):
        sliced_stmt = _slice_utf8_line(
            raw_lines[0], inferred_s_col, inferred_e_col
        )
        if sliced_stmt.strip():
            raw_lines = [sliced_stmt]
    dedented = dedent_source_text("".join(raw_lines))
    empty_res: Dict[str, Any] = {
        "inputs": [],
        "outputs": [],
        "stores": [],
        "free_vars": [],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
        "control_flow_hazards": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_yield_assignment": False,
        "has_try_or_with": False,
        "has_return": False,
        "has_return_value": False,
        "has_super": False,
        "has_mangled_names": False,
        "local_imports": [],
        "yield_expr_names": [],
        "is_async": False,
        "conditional_outputs": [],
        "definite_stores": [],
        "has_instance_binding": False,
        "has_class_binding": False,
        "binding_kind": None,
        "has_receiver_access": False,
        "instance_attrs": [],
        "class_attrs": [],
        "outputs_ambiguous": False,
        "type_params": [],
        "type_param_free_vars": [],
        "has_class_scoped_type_params": False,
        "deleted_names": [],
        "_skip_pre_unit_closures": bool(unit.get("skip_pre_unit_closures")),
    }
    if not dedented.strip():
        return empty_res

    tree: Optional[ast.AST] = None
    loop_offset = 0
    candidate_stmts: List[ast.stmt] = []

    parse_candidates = [
        (dedented, 0),
        (f"if 1:\n{dedented}", 0),
        (f"def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"async def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        (
            f"def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
        (
            f"async def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
    ]

    had_resource_error = False
    for cand_text, offset in parse_candidates:
        try:
            tree = ast.parse(cand_text)
            loop_offset = offset
            if cand_text == dedented:
                if isinstance(tree, ast.Module):
                    if len(tree.body) == 1 and isinstance(
                        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = tree.body[0].body
                    else:
                        candidate_stmts = tree.body
            elif cand_text.startswith("if 1:\n"):
                if (
                    isinstance(tree, ast.Module)
                    and len(tree.body) == 1
                    and isinstance(tree.body[0], ast.If)
                ):
                    inner_body = tree.body[0].body
                    tree = ast.Module(body=inner_body, type_ignores=[])
                    if len(inner_body) == 1 and isinstance(
                        inner_body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = inner_body[0].body
                    else:
                        candidate_stmts = inner_body
            elif offset == 1:
                wrapper_fn = tree.body[0]  # type: ignore[attr-defined]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for_loop = wrapper_fn.body[0]
                    if isinstance(for_loop, (ast.For, ast.AsyncFor, ast.While)):
                        candidate_stmts = for_loop.body
            else:
                wrapper_fn = tree.body[0]  # type: ignore[attr-defined]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    candidate_stmts = wrapper_fn.body
            break
        except (RecursionError, MemoryError):
            had_resource_error = True
            break
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue

    if tree is None:
        if had_resource_error:
            return {
                **empty_res,
                "control_flow_hazards": ["pathological_ast"],
                "is_control_flow_safe": False,
            }
        return empty_res

    try:
        if "is_subroutine" in unit and unit.get("is_subroutine") is not None:
            is_subroutine = bool(unit["is_subroutine"])
        elif str(unit.get("kind") or "") in _SUBROUTINE_KINDS:
            is_subroutine = True
        elif str(unit.get("kind") or "") in _WHOLE_UNIT_KINDS:
            is_subroutine = False
        else:
            is_subroutine = is_subroutine_unit(
                unit, source_text=_get_full_source(), tree=file_tree
            )

        rec_param = unit.get("receiver_param")
        rec_kind = unit.get("receiver_kind")
        is_class_rec = rec_kind == "class"

        inst_receivers: Set[str] = {"self", "this"}
        class_receivers: Set[str] = {"cls", "klass"}
        if rec_param:
            if is_class_rec:
                class_receivers.add(rec_param)
            else:
                inst_receivers.add(rec_param)

        all_receivers = inst_receivers | class_receivers

        cand_lines = split_source_lines(cand_text)
        visitor = _ScopeVisitor(
            loop_offset=loop_offset,
            source_lines=cand_lines,
            is_subroutine=is_subroutine,
            receiver_names=tuple(all_receivers),
        )
        visitor.visit(tree)

        hazards: List[str] = []
        if visitor.naked_breaks > 0:
            hazards.append("naked_break")
        if visitor.naked_continues > 0:
            hazards.append("naked_continue")
        if visitor.has_yield:
            hazards.append("generator_yield")
        if is_subroutine and visitor.has_return:
            hazards.append("embedded_return")

        has_class_scoped_type_params = False
        if visitor.type_param_free_vars and not is_subroutine:
            s_line = parse_unit_coord(unit, "start", default=1)
            e_line = parse_unit_coord(unit, "end", default=s_line)
            u_col, _ = _unit_cols(unit)
            top_fn = (
                tree.body[0]
                if isinstance(tree, ast.Module)
                and len(tree.body) == 1
                and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                else None
            )
            top_fn_name = getattr(top_fn, "name", None) if top_fn is not None else None
            top_fn_count = (
                sum(
                    1
                    for n in ast.walk(top_fn)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                if top_fn is not None
                else None
            )
            parsed_tree = file_tree
            if parsed_tree is None:
                full_src = _get_full_source()
                if full_src:
                    parsed_tree = _get_or_parse_tree(full_src)
            enclosing_fn_locals: Set[str] = set()
            enclosing_cls_locals: Set[str] = set()
            has_enclosing_fn = False
            has_enclosing_cls = False
            prev_enclosed_cls_name: Optional[str] = None
            if parsed_tree is not None:
                matched_self = False
                for earliest_start, n_end, _span, scope_node in _get_tree_scope_intervals(
                    parsed_tree
                ):
                    if earliest_start > s_line or e_line > n_end:
                        continue
                    if isinstance(scope_node, ast.ClassDef):
                        if not has_enclosing_cls:
                            has_enclosing_cls = True
                            prev_enclosed_cls_name = getattr(scope_node, "name", None)
                            enclosing_cls_locals.update(
                                _extract_single_node_local_names(
                                    scope_node, include_globals=False
                                )
                            )
                        else:
                            if isinstance(prev_enclosed_cls_name, str):
                                enclosing_cls_locals.add(prev_enclosed_cls_name)
                            prev_enclosed_cls_name = getattr(scope_node, "name", None)
                            enclosing_cls_locals.update(
                                _extract_type_param_names(scope_node)
                            )
                    elif isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if (
                            not matched_self
                            and s_line
                            in (
                                earliest_start,
                                getattr(scope_node, "lineno", earliest_start),
                            )
                            and (
                                e_line == n_end
                                or top_fn_name is not None
                                or u_col is not None
                            )
                            and (
                                u_col is None
                                or getattr(scope_node, "col_offset", 0) == u_col
                            )
                            and (
                                top_fn_name is None
                                or getattr(scope_node, "name", None) == top_fn_name
                            )
                            and (
                                top_fn_count is None
                                or sum(
                                    1
                                    for n in ast.walk(scope_node)
                                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                )
                                == top_fn_count
                            )
                        ):
                            matched_self = True
                            continue
                        if (
                            earliest_start < s_line
                            or n_end > e_line
                            or (u_col is None and matched_self)
                            or (
                                u_col is not None
                                and getattr(scope_node, "col_offset", 0) < u_col
                            )
                        ):
                            has_enclosing_fn = True
                            enclosing_fn_locals.update(
                                _extract_single_node_local_names(
                                    scope_node, include_globals=False
                                )
                            )
            if has_enclosing_fn and any(
                fv in enclosing_fn_locals for fv in visitor.type_param_free_vars
            ):
                hazards.append("type_param_closure_capture")
            elif (
                not has_enclosing_fn
                and not has_enclosing_cls
                and unit.get("kind") == "closure"
            ):
                hazards.append("type_param_closure_capture")
            if has_enclosing_cls and any(
                fv in enclosing_cls_locals for fv in visitor.type_param_free_vars
            ):
                has_class_scoped_type_params = True
                if (
                    has_enclosing_fn
                    or unit.get("kind") == "closure"
                ) and "type_param_closure_capture" not in hazards:
                    hazards.append("type_param_closure_capture")
            elif not has_enclosing_cls and (
                unit.get("enclosing_class")
                or unit.get("kind") in ("method", "async_method")
            ):
                has_class_scoped_type_params = True

        # Definite, conditional, and branch-aware deletion analysis across candidate statements
        may_deleted: Set[str] = set()
        live_in_deleted: Set[str] = set()
        def_assigned, cond_assigned = _analyze_block_assignment(
            candidate_stmts,
            may_deleted=may_deleted,
            live_in_deleted=live_in_deleted,
        )
        for del_name in sorted(live_in_deleted):
            if (
                del_name not in visitor.params
                and del_name not in visitor.globals
                and del_name not in BUILTIN_NAMES
            ):
                if del_name not in visitor.loads:
                    visitor.loads.append(del_name)
                if del_name not in visitor.read_before_write:
                    visitor.read_before_write.append(del_name)

        if is_subroutine:
            live_in_visitor = _UnitLiveInVisitor(1, 10**9, include_annotations=False)
            for stmt in candidate_stmts:
                live_in_visitor.visit(stmt)
            excluded_cond_vars = (
                set(visitor.params) | set(visitor.globals) | BUILTIN_NAMES
            )
            cond_live_ins = [
                v
                for v in sorted(live_in_visitor.live_ins)
                if v in visitor.loads
                and v not in excluded_cond_vars
                and v not in visitor.read_before_write
            ]
            cond_escaping_cands = [
                v
                for v in sorted(
                    (cond_assigned - def_assigned)
                    | set(cond_live_ins)
                    | (live_in_visitor.live_ins & set(visitor.stores))
                )
                if v not in excluded_cond_vars
            ]
            if cond_live_ins or cond_escaping_cands:
                pre_bound: Set[str] = set()
                rejected_escaping_vars: Set[str] = set()
                u_start = parse_unit_coord(unit, "start", default=0)
                u_end = parse_unit_coord(unit, "end", default=u_start)
                u_start_col, u_end_col = inferred_s_col, inferred_e_col
                mod_tree = file_tree
                if mod_tree is None and u_start > 0:
                    full_src = _get_full_source()
                    if full_src:
                        mod_tree = _get_or_parse_tree(full_src)
                if mod_tree is not None and u_start > 0:
                    u_start_col, u_end_col = _resolve_unit_col_bounds(
                        candidate_stmts,
                        mod_tree,
                        u_start,
                        u_end,
                        u_start_col,
                        u_end_col,
                    )
                    pre_bound, _ = _find_enclosing_pre_bound_names(
                        mod_tree, u_start, u_end, u_start_col=u_start_col
                    )
                    if cond_escaping_cands:
                        enc_chain = _find_enclosing_function_chain(
                            mod_tree, u_start, u_end, u_start_col
                        )
                        if enc_chain:
                            enc_nonlocals, enc_outer_globals = (
                                _extract_scope_nonlocals_and_outer_globals(
                                    enc_chain[-1][2],
                                    u_start,
                                    u_end,
                                    u_end_col=u_end_col,
                                    u_start_col=u_start_col,
                                )
                            )
                            enc_escaping = enc_nonlocals | enc_outer_globals
                            def_pre_bound = (
                                _find_enclosing_chain_definite_pre_bound(
                                    enc_chain,
                                    u_start,
                                    u_end,
                                    u_start_col=u_start_col,
                                    u_end_col=u_end_col,
                                    mod_tree=mod_tree,
                                )
                            )
                            for v in cond_escaping_cands:
                                if v in enc_escaping:
                                    if (
                                        v in enc_nonlocals
                                        and v in pre_bound
                                        and v in def_pre_bound
                                    ):
                                        if v not in visitor.loads:
                                            visitor.loads.append(v)
                                        if v not in visitor.read_before_write:
                                            visitor.read_before_write.append(v)
                                    else:
                                        rejected_escaping_vars.add(v)
                                        if (
                                            v in visitor.read_before_write
                                            and v not in def_pre_bound
                                        ):
                                            visitor.read_before_write.remove(v)
                                        if "severed_closure_cell" not in hazards:
                                            hazards.append("severed_closure_cell")
                                elif (
                                    not visitor.has_yield
                                    and v in pre_bound
                                    and v in def_pre_bound
                                ):
                                    if v not in visitor.loads:
                                        visitor.loads.append(v)
                                    if v not in visitor.read_before_write:
                                        visitor.read_before_write.append(v)
                has_enclosing_context = mod_tree is not None and (
                    u_start > 1 or bool(_get_tree_scope_intervals(mod_tree))
                )
                for v in cond_live_ins:
                    if (
                        v not in rejected_escaping_vars
                        and (
                            v in pre_bound
                            or (not has_enclosing_context and v in def_assigned)
                        )
                        and v not in visitor.read_before_write
                    ):
                        visitor.read_before_write.append(v)

        free_vars = [
            name for name in visitor.loads
            if name not in visitor.params
            and (name not in visitor.stores or name in visitor.read_before_write)
            and name not in visitor.globals
            and name not in BUILTIN_NAMES
        ]

        inputs: List[str] = list(visitor.params)
        for fv in free_vars:
            if fv not in inputs:
                inputs.append(fv)
        for nl in visitor.nonlocals:
            if nl not in inputs:
                inputs.append(nl)

        param_map = {p["name"].lstrip("*"): p.get("kind", "pos") for p in visitor.param_details}
        inputs.sort(key=lambda v: _rank_param_kind(v, param_map, receiver_names=tuple(all_receivers)))

        has_instance_binding = any(
            r in visitor.loads
            or r in visitor.params
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            for r in inst_receivers
            if r in ("self", "this")
            or (rec_param and r == rec_param and rec_kind == "instance")
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
        )
        has_class_binding = any(
            r in visitor.loads
            or r in visitor.params
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            for r in class_receivers
            if r in ("cls", "klass")
            or (rec_param and r == rec_param and is_class_rec)
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
        )
        binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

        primary_rec = rec_param or ("cls" if is_class_rec else None)
        _normalize_receiver_order(inputs, has_instance_binding, has_class_binding, primary_receiver=primary_rec)

        # Unit outputs: explicit returns if present; for subroutines without explicit
        # returns, all local variable stores act as unit outputs to preserve caller mutations.
        pre_outs = resolve_unit_declared_outputs(unit)

        outputs_ambiguous = False
        alias_cands = visitor.type_aliases & (set(visitor.stores) | set(visitor.nonlocals))
        downstream_aliases: Set[str] = set()
        if is_subroutine and alias_cands:
            full_src = _get_full_source()
            if full_src:
                ds_reads = collect_downstream_read_names(
                    full_src,
                    unit,
                    candidates=alias_cands,
                    tree=file_tree,
                    unit_inputs=set(inputs),
                )
                downstream_aliases = (
                    set(alias_cands) if ds_reads is None else ds_reads
                )
            else:
                downstream_aliases = set(alias_cands)
        excluded_local_defs = (visitor.type_aliases - downstream_aliases) | set(
            visitor.imported_names
        )
        all_deleted = _extract_deleted_names(candidate_stmts) | visitor.deleted_names
        uncond_deleted = all_deleted - (def_assigned | cond_assigned)

        if is_subroutine and isinstance(pre_outs, (list, tuple)):
            outputs = list(dict.fromkeys(pre_outs))
        elif is_subroutine and isinstance(pre_outs, (set, frozenset)):
            if len(pre_outs) > 1:
                logger.debug(
                    "Rejecting set-valued outputs (%s): unordered set cannot establish "
                    "positional order",
                    pre_outs,
                )
                outputs = []
                outputs_ambiguous = True
            else:
                outputs = list(pre_outs)
        elif visitor.returns:
            outputs = list(visitor.returns)
        elif is_subroutine:
            seen: Set[str] = set()
            outputs = []
            for v in visitor.stores:
                if (
                    v not in visitor.globals
                    and v not in BUILTIN_NAMES
                    and v not in visitor.except_vars
                    and v not in uncond_deleted
                    and v not in excluded_local_defs
                    and v not in seen
                ):
                    seen.add(v)
                    outputs.append(v)
        else:
            outputs = []

        for nl in visitor.nonlocals:
            if (
                nl in visitor.stores
                and nl not in outputs
                and nl not in visitor.except_vars
                and nl not in uncond_deleted
                and nl not in excluded_local_defs
            ):
                outputs.append(nl)

        # Detect conditional variable escapes (outputs assigned conditionally without prior
        # unconditional assignment in this block, or potentially deleted along a path).
        conditional_outputs = [
            v for v in outputs
            if (v not in inputs or v in may_deleted) and v not in def_assigned
        ]

        if (
            is_subroutine
            and bool((set(outputs) | set(inputs)) & may_deleted)
            and "deleted_output" not in hazards
        ):
            hazards.append("deleted_output")

        if is_subroutine or unit.get("kind") in (
            "closure",
            "comprehension",
            "complex_expr",
        ):
            c_u_start = parse_unit_coord(unit, "start", default=0)
            c_u_end = parse_unit_coord(unit, "end", default=c_u_start)
            if c_u_start > 0:
                c_mod_tree = file_tree
                if c_mod_tree is None:
                    full_src = _get_full_source()
                    if full_src:
                        c_mod_tree = _get_or_parse_tree(full_src)
                if c_mod_tree is not None:
                    c_start_col, c_end_col = inferred_s_col, inferred_e_col
                    if (
                        "severed_closure_cell" not in hazards
                        and _has_severed_closure_cell_hazard(
                            _UnitClosureHazardContext(
                                candidate_stmts=candidate_stmts,
                                mod_tree=c_mod_tree,
                                u_start=c_u_start,
                                u_end=c_u_end,
                                u_start_col=c_start_col,
                                u_end_col=c_end_col,
                                is_subroutine=is_subroutine,
                                unit_kind=str(unit.get("kind") or ""),
                                inputs=inputs,
                                outputs=outputs,
                                stores=visitor.stores,
                                deleted_names=all_deleted,
                                unit_top_locals=visitor.unit_top_locals,
                                free_vars=free_vars,
                                nonlocals=visitor.nonlocals,
                                has_yield=visitor.has_yield,
                                is_async=visitor.is_async,
                                skip_pre_unit_closures=bool(
                                    unit.get("skip_pre_unit_closures")
                                ),
                                definite_stores=def_assigned,
                            )
                        )
                    ):
                        hazards.append("severed_closure_cell")

        is_control_flow_safe = not any(
            h
            in (
                "naked_break",
                "naked_continue",
                "embedded_return",
                "type_param_closure_capture",
                "pathological_ast",
                "deleted_output",
                "severed_closure_cell",
            )
            for h in hazards
        )

        has_receiver_access = bool(
            any(
                r in visitor.loads
                or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
                for r in all_receivers
                if r in ("self", "cls", "this", "klass")
                or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            )
        )

        is_sub_or_inner = is_subroutine or unit.get("kind") in (
            "comprehension",
            "complex_expr",
            "class",
            "data_table",
        )
        unit_is_async = unit.get("is_async")
        if unit_is_async is None:
            if is_sub_or_inner and not visitor.has_yield:
                unit_is_async = visitor.is_async
            else:
                s_line = parse_unit_coord(unit, "start", default=1)
                e_line = parse_unit_coord(unit, "end", default=s_line)
                full_src = _get_full_source()
                if full_src:
                    unit_is_async = find_enclosing_function_is_async(
                        full_src, s_line, e_line, tree=file_tree
                    )
                else:
                    unit_is_async = None

        effective_is_async: Optional[bool]
        if visitor.is_async or unit.get("kind") in ("async_function", "async_method"):
            effective_is_async = True
        elif "is_async" in unit and unit.get("is_async") is not None:
            effective_is_async = bool(unit.get("is_async"))
        elif visitor.has_yield:
            effective_is_async = bool(unit_is_async) if unit_is_async is not None else None
        elif not is_sub_or_inner and unit.get("kind") not in (
            "function",
            "method",
            "async_function",
            "async_method",
        ):
            effective_is_async = bool(unit_is_async)
        else:
            effective_is_async = False

        return {
            "inputs": inputs,
            "outputs": outputs,
            "stores": visitor.stores,
            "free_vars": free_vars,
            "nonlocals": visitor.nonlocals,
            "globals": visitor.globals,
            "attrs_read": visitor.attrs_read,
            "attrs_written": visitor.attrs_written,
            "param_details": visitor.param_details,
            "return_type": visitor.return_type,
            "control_flow_hazards": hazards,
            "is_control_flow_safe": is_control_flow_safe,
            "has_yield": visitor.has_yield,
            "has_yield_assignment": visitor.has_yield_assignment,
            "has_try_or_with": visitor.has_try_or_with,
            "has_return": visitor.has_return,
            "has_return_value": visitor.has_return_value,
            "has_super": visitor.has_super,
            "has_mangled_names": visitor.has_mangled_names,
            "local_imports": visitor.local_imports,
            "yield_expr_names": visitor.yield_expr_names,
            "is_async": effective_is_async,
            "conditional_outputs": conditional_outputs,
            "definite_stores": sorted(def_assigned),
            "has_instance_binding": has_instance_binding,
            "has_class_binding": has_class_binding,
            "binding_kind": binding_kind,
            "has_receiver_access": has_receiver_access,
            "instance_attrs": [
                a for a in visitor.attrs_read + visitor.attrs_written
                if any(a.startswith(f"{r}.") for r in inst_receivers)
            ],
            "class_attrs": [
                a for a in visitor.attrs_read + visitor.attrs_written
                if any(a.startswith(f"{r}.") for r in class_receivers)
            ],
            "outputs_ambiguous": outputs_ambiguous,
            "type_params": list(visitor.type_params),
            "type_param_free_vars": list(visitor.type_param_free_vars),
            "has_class_scoped_type_params": has_class_scoped_type_params,
            "deleted_names": sorted(may_deleted),
            "_skip_pre_unit_closures": bool(unit.get("skip_pre_unit_closures")),
        }
    except (RecursionError, MemoryError):
        return {
            **empty_res,
            "control_flow_hazards": ["pathological_ast"],
            "is_control_flow_safe": False,
        }



def inspect_single_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    tree: Optional[ast.AST] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping for an individual code unit."""
    return _inspect_unit_scope(unit, repo_root=repo_root, file_tree=tree)


def _align_symmetric_var_list(list1: List[str], list2: List[str]) -> List[str]:
    """Aligns shared variables between two units, preserving side 1 order if equal length."""
    common = [var for var in list1 if var in list2]
    if len(list1) == len(list2):
        return common if len(common) == len(list1) else list1
    return common if common else list1


def _has_type_param_default(tp_str: str) -> bool:
    """Checks whether a PEP 695/696 type parameter string has a top-level default."""
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", '""', tp_str)
    depth = 0
    for ch in cleaned:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return True
    return False


def _merge_type_params(tp1: Sequence[str], tp2: Sequence[str]) -> List[str]:
    """Merges PEP 695 type parameters across two units without duplicate parameter names."""
    if not tp2:
        return list(tp1)
    if not tp1:
        return list(tp2)
    seen: Dict[str, str] = {}
    order: List[str] = []
    for raw_tp in list(tp1) + list(tp2):
        head = raw_tp.split(":", 1)[0].split("=", 1)[0].strip()
        ident = head.lstrip("*").strip()
        if not ident:
            continue
        if ident not in seen:
            seen[ident] = raw_tp
            order.append(ident)
        elif seen[ident] != raw_tp:
            prev_head = seen[ident].split(":", 1)[0].split("=", 1)[0].strip()
            seen[ident] = head if prev_head == head else ident
    merged = [seen[ident] for ident in order]
    return [tp for tp in merged if not _has_type_param_default(tp)] + [
        tp for tp in merged if _has_type_param_default(tp)
    ]


def _scope_matches_closure_mode(
    scope: Optional[Dict[str, Any]], expected_skip: bool
) -> bool:
    """Checks whether a precomputed scope dictionary matches expected_skip."""
    if scope is None:
        return False
    raw_flag = scope.get("_skip_pre_unit_closures", False)
    return isinstance(raw_flag, bool) and raw_flag is bool(expected_skip)


def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    scope1: Optional[Dict[str, Any]] = None,
    scope2: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping to determine inputs, outputs, closures, and attributes."""
    expected_skip1 = bool(u1.get("skip_pre_unit_closures", False))
    info1 = (
        scope1
        if (
            scope1 is not None
            and _scope_matches_closure_mode(scope1, expected_skip1)
        )
        else inspect_single_unit_scope(u1, repo_root=repo_root, tree=tree1)
    )
    if u2 is not None:
        expected_skip2 = bool(u2.get("skip_pre_unit_closures", False))
        info2 = (
            scope2
            if (
                scope2 is not None
                and _scope_matches_closure_mode(scope2, expected_skip2)
            )
            else inspect_single_unit_scope(u2, repo_root=repo_root, tree=tree2)
        )
        inputs = _align_symmetric_var_list(info1["inputs"], info2["inputs"])
        outputs = _align_symmetric_var_list(info1["outputs"], info2["outputs"])
        free_vars = list(dict.fromkeys(info1["free_vars"] + info2["free_vars"]))
        nonlocals = list(dict.fromkeys(info1["nonlocals"] + info2["nonlocals"]))
        globals_ = list(dict.fromkeys(info1["globals"] + info2["globals"]))
        attrs_read = list(dict.fromkeys(info1["attrs_read"] + info2["attrs_read"]))
        attrs_written = list(dict.fromkeys(info1["attrs_written"] + info2["attrs_written"]))
        if info1["return_type"] == info2["return_type"]:
            return_type = info1["return_type"]
        elif info1["return_type"] is None:
            return_type = info2["return_type"]
        elif info2["return_type"] is None:
            return_type = info1["return_type"]
        else:
            return_type = None
        hazards = list(dict.fromkeys(info1["control_flow_hazards"] + info2["control_flow_hazards"]))
        is_safe = info1["is_control_flow_safe"] and info2["is_control_flow_safe"]
        has_yield = info1["has_yield"] or info2["has_yield"]
        has_yield_assignment = bool(
            info1.get("has_yield_assignment", False)
            or info2.get("has_yield_assignment", False)
        )
        has_try_or_with = bool(
            info1.get("has_try_or_with", False)
            or info2.get("has_try_or_with", False)
        )
        has_return = info1["has_return"] or info2["has_return"]
        has_return_value = bool(info1.get("has_return_value") or info2.get("has_return_value"))
        has_super = bool(info1.get("has_super", False) or info2.get("has_super", False))
        has_mangled = bool(
            info1.get("has_mangled_names", False) or info2.get("has_mangled_names", False)
        )
        local_imports = list(dict.fromkeys(info1["local_imports"] + info2["local_imports"]))
        local_imports_by_unit = [info1["local_imports"], info2["local_imports"]]
        yield_expr_names = info1["yield_expr_names"] + info2["yield_expr_names"]
        is_async = info1.get("is_async", False) or info2.get("is_async", False)
        conditional_outputs = list(dict.fromkeys(
            info1.get("conditional_outputs", []) + info2.get("conditional_outputs", [])
        ))
        definite_stores = sorted(
            set(info1.get("definite_stores", [])) & set(info2.get("definite_stores", []))
        )
        outputs_ambiguous = bool(
            info1.get("outputs_ambiguous", False) or info2.get("outputs_ambiguous", False)
        )
        type_params = _merge_type_params(
            info1.get("type_params", []), info2.get("type_params", [])
        )
        deleted_names = list(
            dict.fromkeys(
                info1.get("deleted_names", []) + info2.get("deleted_names", [])
            )
        )
    else:
        inputs = info1["inputs"]
        outputs = info1["outputs"]
        free_vars = info1["free_vars"]
        nonlocals = info1["nonlocals"]
        globals_ = info1["globals"]
        attrs_read = info1["attrs_read"]
        attrs_written = info1["attrs_written"]
        return_type = info1["return_type"]
        hazards = info1["control_flow_hazards"]
        is_safe = info1["is_control_flow_safe"]
        has_yield = info1["has_yield"]
        has_yield_assignment = bool(info1.get("has_yield_assignment", False))
        has_try_or_with = bool(info1.get("has_try_or_with", False))
        has_return = info1["has_return"]
        has_return_value = bool(info1.get("has_return_value"))
        has_super = bool(info1.get("has_super", False))
        has_mangled = bool(info1.get("has_mangled_names", False))
        local_imports = info1["local_imports"]
        local_imports_by_unit = [info1["local_imports"]]
        yield_expr_names = info1["yield_expr_names"]
        is_async = info1.get("is_async", False)
        conditional_outputs = info1.get("conditional_outputs", [])
        definite_stores = info1.get("definite_stores", [])
        outputs_ambiguous = bool(info1.get("outputs_ambiguous", False))
        type_params = list(info1.get("type_params", []))
        deleted_names = list(info1.get("deleted_names", []))

    has_instance_binding = info1.get("has_instance_binding", False) or (
        info2.get("has_instance_binding", False) if u2 is not None else False
    )
    has_class_binding = info1.get("has_class_binding", False) or (
        info2.get("has_class_binding", False) if u2 is not None else False
    )
    has_receiver_access = info1.get("has_receiver_access", False) or (
        info2.get("has_receiver_access", False) if u2 is not None else False
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

    primary_rec = (
        u1.get("receiver_param")
        or (u2.get("receiver_param") if u2 else None)
        or (
            "cls"
            if (
                u1.get("receiver_kind") == "class"
                or (u2 and u2.get("receiver_kind") == "class")
            )
            else None
        )
    )
    _normalize_receiver_order(
        inputs,
        has_instance_binding,
        has_class_binding,
        primary_receiver=primary_rec,
    )

    locals_ = [var for var in info1["stores"] if var not in inputs]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "locals": locals_,
        "free_vars": free_vars,
        "nonlocals": nonlocals,
        "globals": globals_,
        "attrs_read": attrs_read,
        "attrs_written": attrs_written,
        "param_details": info1["param_details"],
        "return_type": return_type,
        "control_flow_hazards": hazards,
        "is_control_flow_safe": is_safe,
        "has_yield": has_yield,
        "has_yield_assignment": has_yield_assignment,
        "has_try_or_with": has_try_or_with,
        "has_return": has_return,
        "has_return_value": has_return_value,
        "has_super": has_super,
        "has_mangled_names": has_mangled,
        "local_imports": local_imports,
        "local_imports_by_unit": local_imports_by_unit,
        "yield_expr_names": yield_expr_names,
        "is_async": is_async,
        "conditional_outputs": conditional_outputs,
        "definite_stores": definite_stores,
        "has_instance_binding": has_instance_binding,
        "has_class_binding": has_class_binding,
        "binding_kind": binding_kind,
        "has_receiver_access": has_receiver_access,
        "instance_attrs": list(
            dict.fromkeys(
                info1.get("instance_attrs", [])
                + (info2.get("instance_attrs", []) if u2 is not None else [])
            )
        ),
        "class_attrs": list(
            dict.fromkeys(
                info1.get("class_attrs", [])
                + (info2.get("class_attrs", []) if u2 is not None else [])
            )
        ),
        "outputs_ambiguous": outputs_ambiguous,
        "type_params": type_params,
        "type_param_free_vars": list(
            dict.fromkeys(
                info1.get("type_param_free_vars", [])
                + (info2.get("type_param_free_vars", []) if u2 is not None else [])
            )
        ),
        "has_class_scoped_type_params": bool(
            info1.get("has_class_scoped_type_params", False)
            or (
                info2.get("has_class_scoped_type_params", False)
                if u2 is not None
                else False
            )
        ),
        "deleted_names": deleted_names,
        "_skip_pre_unit_closures": (
            bool(info1.get("_skip_pre_unit_closures", False))
            if (
                u2 is None
                or _scope_matches_closure_mode(
                    info2, bool(info1.get("_skip_pre_unit_closures", False))
                )
            )
            else None
        ),
    }
