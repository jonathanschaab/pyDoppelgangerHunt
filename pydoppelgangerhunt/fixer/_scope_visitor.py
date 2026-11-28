"""AST lexical scope visitor and receiver/parameter helper primitives."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_arg_names as _extract_arg_names_tuple,
    _extract_import_bound_names,
    _extract_type_param_names as _extract_type_param_names_tuple,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _extract_single_node_local_names,
    _extract_strict_fn_local_names,
)
from pydoppelgangerhunt.fixer._df_visitors import _OuterExprVisitor

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))

_SYNTHETIC_WRAPPER_NAME: str = "__pdh_wrapper__"


class YieldExprEntry(NamedTuple):
    """Typed record for a yielded variable reference or inferred literal element type."""

    kind: Literal["yield", "yield_from", "literal"]
    target: str


@dataclass
class _ScopeFrame:
    """Typed lexical scope frame tracking locals, nonlocal/global directives, and class bodies."""

    names: Set[str] = field(default_factory=set)
    nonlocals: Set[str] = field(default_factory=set)
    globals: Set[str] = field(default_factory=set)
    is_class: bool = False


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
    return set(_extract_arg_names_tuple(args))


def _extract_type_param_names(node: ast.AST) -> Set[str]:
    """Extracts PEP 695 type parameter names from an AST node."""
    return set(_extract_type_param_names_tuple(node))


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
        self.type_param_names: Set[str] = set()
        self.type_param_free_vars: List[str] = []
        self._scope_stack: List[_ScopeFrame] = []
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
        self.yield_expr_names: List[YieldExprEntry] = []
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
        self.unit_top_locals = _extract_strict_fn_local_names(target_node)
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
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                logger.debug("Failed to unparse argument annotation %r: %s", arg_node.annotation, exc)
                type_val = None

        default_val = None
        if default_node is not None:
            try:
                default_val = ast.unparse(default_node)
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
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
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
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
            self._scope_stack[-1].names.add(name)
            return
        eff_len = len(self._scope_stack) - comp_depth
        bound_in_intermediate_fn = eff_len > 2 and any(
            name in frame.names
            for frame in self._scope_stack[1 : -1 - comp_depth]
            if not frame.is_class
        )
        is_active_nonlocal = (
            name in self.nonlocals
            if eff_len <= 1
            else (
                name in self._scope_stack[-1 - comp_depth].nonlocals
                and not bound_in_intermediate_fn
            )
        )
        is_active_global = (
            name in self.globals
            if eff_len <= 1
            else name in self._scope_stack[-1 - comp_depth].globals
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
                name not in target_frame.nonlocals
                and not (comp_depth > 0 and target_frame.is_class)
            ):
                target_frame.names.add(name)
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
            self.type_param_names.update(tp_names)
            for tp in getattr(node, "type_params", []):
                try:
                    self.type_params.append(ast.unparse(tp))
                except (ValueError, TypeError, AttributeError, RecursionError) as exc:
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
                    _ScopeFrame(names=set(self.params) | tp_names | self.unit_top_locals)
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
                    self._scope_stack.append(_ScopeFrame(names=set(self.unit_top_locals)))
                    created_outer = True
                nested_locals = _extract_single_node_local_names(
                    node,
                    include_globals=False,
                    include_nonlocals=False,
                    include_self_name=False,
                )
                self._scope_stack.append(
                    _ScopeFrame(
                        names=self._extract_arg_names(node.args) | tp_names | nested_locals
                    )
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
            self._scope_stack.append(_ScopeFrame(names=set(self.unit_top_locals)))
            created_outer = True
        lambda_locals = _extract_single_node_local_names(
            node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
        )
        self._scope_stack.append(
            _ScopeFrame(names=self._extract_arg_names(node.args) | lambda_locals)
        )
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
                self._scope_stack.append(_ScopeFrame(names=set(self.unit_top_locals)))
                created_outer = True
            self._scope_stack.append(_ScopeFrame(names=set(tp_names), is_class=True))
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
            sum(1 for frame in self._scope_stack if not frame.is_class)
            - self._comprehension_depth
        )

    def _is_inner_bound(self, name: str) -> bool:
        return any(
            name in frame.names
            for frame in self._scope_stack[1:]
            if not frame.is_class or frame is self._scope_stack[-1]
        )

    def _record_load_name(self, name: str) -> None:
        is_inner = self._is_inner_bound(name) or any(
            name in s for s in self._type_param_scopes
        )
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
                node.id in self._scope_stack[-1].nonlocals
                and not any(
                    node.id in frame.names
                    for frame in self._scope_stack[1:-1]
                    if not frame.is_class
                )
            ):
                self.deleted_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.attr):
            self.has_mangled_names = True
        attr_name = _unfold_receiver_attribute(node, receiver_names=self.receiver_names)
        if attr_name is not None:
            receiver_id = attr_name.split(".", 1)[0]
            if not self._is_inner_bound(receiver_id) and self.class_depth == 0:
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
                if not self._is_inner_bound(receiver_id) and self.class_depth == 0:
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
            self._scope_stack.append(_ScopeFrame(names=set(self.unit_top_locals)))
            created_outer = True
        self._scope_stack.append(_ScopeFrame())
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
                    name in frame.names and not frame.is_class
                    for frame in enclosing_frames
                )
                or name in self.stores
                or name in self.params
                or name in self.unit_top_locals
            )
            if (eff_len <= 1 or not is_unit_bound) and name not in self.nonlocals:
                self.nonlocals.append(name)
            if eff_len > 1:
                self._scope_stack[-1 - self._comprehension_depth].nonlocals.add(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        eff_len = len(self._scope_stack) - self._comprehension_depth
        for name in node.names:
            if eff_len <= 1:
                if name not in self.globals:
                    self.globals.append(name)
            else:
                self._scope_stack[-1 - self._comprehension_depth].globals.add(name)
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

    def _record_yield_expr(
        self,
        kind: Literal["yield", "yield_from"],
        node: Union[ast.Yield, ast.YieldFrom],
    ) -> None:
        if len(self._scope_stack) <= 1:
            self.has_yield = True
            if node.value is not None:
                if isinstance(node.value, ast.Name):
                    self.yield_expr_names.append(YieldExprEntry(kind, node.value.id))
                elif isinstance(node.value, ast.Constant) and node.value.value is not None:
                    self.yield_expr_names.append(
                        YieldExprEntry("literal", type(node.value.value).__name__)
                    )
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
                            self.yield_expr_names.append(
                                YieldExprEntry("literal", t_name)
                            )

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
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                logger.debug("Failed to unparse import node %r: %s", node, exc)
                return

        is_nested = (
            len(self._scope_stack) > 0
            if self.is_subroutine
            else len(self._scope_stack) > 1
        )
        if not is_nested and stmt not in self.local_imports:
            self.local_imports.append(stmt)
        for bound_name in _extract_import_bound_names(node):
            if not is_nested:
                self.imported_names[bound_name] = stmt
            self._record_store_name(bound_name)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_node(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_node(node)
        self.generic_visit(node)
