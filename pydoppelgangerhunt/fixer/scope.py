"""Lexical variable scope analysis, AST visitor, and control flow hazard tracking."""

from __future__ import annotations

import ast
import builtins
import logging
import sys
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.reporters import extract_unit_source_code
from pydoppelgangerhunt.fixer.source import _slice_unit_token_lines

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


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


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, returns, nonlocals, globals, and attributes."""

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
        self._scope_stack: List[Set[str]] = []
        self._comprehension_depth: int = 0
        self.class_depth: int = 0
        self.loop_offset: int = loop_offset
        self.loop_depth: int = 0
        self.has_return: bool = False
        self.has_yield: bool = False
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

    @staticmethod
    def _extract_arg_names(args: ast.arguments) -> Set[str]:
        arg_set = {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
        if args.vararg:
            arg_set.add(args.vararg.arg)
        if args.kwarg:
            arg_set.add(args.kwarg.arg)
        return arg_set

    def _record_store_name(self, name: str) -> None:
        if len(self._scope_stack) <= 1 or name in self.nonlocals:
            self.deleted_names.discard(name)
        if name in self.nonlocals:
            if name not in BUILTIN_NAMES and name not in self.stores:
                self.stores.append(name)
        elif name in self.globals:
            pass
        elif len(self._scope_stack) > 1:
            self._scope_stack[-1].add(name)
        elif name not in BUILTIN_NAMES and name not in self.stores:
            self.stores.append(name)

    def _process_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        is_top = (
            len(self._scope_stack) == 0
            and not self.is_subroutine
            and not self._has_processed_top_func
        )
        created_outer = False
        if is_top:
            self._has_processed_top_func = True
            self._process_args(node)
            self._scope_stack.append(set(self.params))
        else:
            for dec in node.decorator_list:
                self.visit(dec)
            for default in node.args.defaults:
                self.visit(default)
            for kw_default in node.args.kw_defaults:
                if kw_default is not None:
                    self.visit(kw_default)
            self._record_store_name(node.name)
            if not self._scope_stack:
                self._scope_stack.append(set())
                created_outer = True
            self._scope_stack.append(self._extract_arg_names(node.args))

        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()
        if created_outer:
            self._scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set())
            created_outer = True
        self._scope_stack.append(self._extract_arg_names(node.args))
        self.visit(node.body)
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
        if len(self._scope_stack) <= 1:
            self.is_async = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_store_name(node.name)
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set())
            created_outer = True
        self._scope_stack.append(set())
        self.class_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.class_depth -= 1
        self._scope_stack.pop()
        if created_outer:
            self._scope_stack.pop()

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
        if name and isinstance(name, str):
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        name = getattr(node, "name", None)
        if name and isinstance(name, str):
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        rest = getattr(node, "rest", None)
        if rest and isinstance(rest, str):
            self._record_store_name(rest)
        self.generic_visit(node)

    def _record_load_name(self, name: str) -> None:
        is_inner = any(name in s for s in self._scope_stack[1:])
        if not is_inner and name not in BUILTIN_NAMES:
            if name not in self.loads:
                self.loads.append(name)
            if (
                len(self._scope_stack) <= 1
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
            if len(self._scope_stack) <= 1 or node.id in self.nonlocals:
                self.deleted_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.attr):
            self.has_mangled_names = True
        attr_name = _unfold_receiver_attribute(node, receiver_names=self.receiver_names)
        if attr_name is not None:
            receiver_id = attr_name.split(".", 1)[0]
            is_inner = any(receiver_id in s for s in self._scope_stack[1:])
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
                is_inner = any(receiver_id in s for s in self._scope_stack[1:])
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

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            target_id = node.target.id
            if len(self._scope_stack) <= 1 + self._comprehension_depth or target_id in self.nonlocals:
                self.deleted_names.discard(target_id)
            if self._comprehension_depth > 0:
                target_idx = -1 - self._comprehension_depth
                stack_len = len(self._scope_stack)
                if stack_len >= abs(target_idx):
                    self._scope_stack[target_idx].add(target_id)
                if stack_len <= abs(target_idx):
                    if (
                        target_id not in BUILTIN_NAMES
                        and target_id not in self.globals
                        and target_id not in self.stores
                    ):
                        self.stores.append(target_id)
            else:
                self._record_store_name(target_id)

    def _visit_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp],
    ) -> None:
        self._comprehension_depth += 1
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set())
            created_outer = True
        inner_vars: Set[str] = set()
        self._scope_stack.append(inner_vars)
        if any(getattr(g, "is_async", False) for g in node.generators):
            if len(self._scope_stack) <= 2:
                self.is_async = True
        for gen in node.generators:
            self.visit(gen.iter)
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    inner_vars.add(n.id)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
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
        for name in node.names:
            if name not in self.nonlocals:
                self.nonlocals.append(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name not in self.globals:
                self.globals.append(name)
        self.generic_visit(node)

    def _visit_loop(self, node: Union[ast.For, ast.AsyncFor, ast.While]) -> None:
        self.loop_depth += 1
        if isinstance(node, (ast.For, ast.AsyncFor)):
            self.visit(node.iter)
            self.visit(node.target)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)
        else:
            self.generic_visit(node)
        self.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
        self._visit_loop(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
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

    def visit_Yield(self, node: ast.Yield) -> None:
        self._record_yield_expr("yield", node)
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._record_yield_expr("yield_from", node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if len(self._scope_stack) <= 1:
            self.has_return = True
            if node.value is not None:
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
    match_as = getattr(ast, "MatchAs", ())
    if isinstance(pattern, match_as):
        sub_pat = getattr(pattern, "pattern", None)
        return sub_pat is None or _is_irrefutable_pattern(sub_pat)
    match_or = getattr(ast, "MatchOr", ())
    if isinstance(pattern, match_or):
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
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):  # pragma: no cover (py310+)
            cases = getattr(stmt, "cases", [])
            has_irrefutable = any(_is_irrefutable_case(c) for c in cases)
            if has_irrefutable and cases and all(_block_terminates(c.body) for c in cases):
                return True
        elif isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
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
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):  # pragma: no cover (py310+)
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
) -> Tuple[Set[str], Set[str]]:
    """Analyzes a sequence of statements to determine unconditionally and conditionally assigned variables.

    Returns:
        A tuple of (definitely_assigned, conditionally_assigned) variable name sets.
    """
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()

    for stmt in statements:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        definite.add(node.id)
                d_t, c_t = _walrus_assignment_in_expr(target)
                definite.update(d_t)
                conditional.update(c_t)
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                if isinstance(stmt.target, ast.Name):
                    definite.add(stmt.target.id)
                d_val, c_val = _walrus_assignment_in_expr(stmt.value)
                definite.update(d_val)
                conditional.update(c_val)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                definite.add(stmt.target.id)
            d_t, c_t = _walrus_assignment_in_expr(stmt.target)
            definite.update(d_t)
            conditional.update(c_t)
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.Expr):
            d_val, c_val = _walrus_assignment_in_expr(stmt.value)
            definite.update(d_val)
            conditional.update(c_val)
        elif isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                d_del, c_del = _walrus_assignment_in_expr(target)
                definite.update(d_del)
                conditional.update(c_del)
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                        definite.discard(node.id)
                        conditional.discard(node.id)
        elif isinstance(stmt, ast.Raise):
            if stmt.exc is not None:
                d_exc, c_exc = _walrus_assignment_in_expr(stmt.exc)
                definite.update(d_exc)
                conditional.update(c_exc)
            if stmt.cause is not None:
                d_cause, c_cause = _walrus_assignment_in_expr(stmt.cause)
                definite.update(d_cause)
                conditional.update(c_cause)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                d_val, c_val = _walrus_assignment_in_expr(stmt.value)
                definite.update(d_val)
                conditional.update(c_val)
        elif isinstance(stmt, ast.If):
            d_test, c_test = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_test)
            conditional.update(c_test)
            b_def, b_cond = _analyze_block_assignment(stmt.body)
            o_def, o_cond = _analyze_block_assignment(stmt.orelse) if stmt.orelse else (set(), set())
            b_term = _block_terminates(stmt.body)
            o_term = _block_terminates(stmt.orelse) if stmt.orelse else False
            if stmt.orelse:
                if o_term and not b_term:
                    definite.update(b_def)
                    conditional.update((b_def | b_cond) - definite)
                elif b_term and not o_term:
                    definite.update(o_def)
                    conditional.update((o_def | o_cond) - definite)
                elif not b_term and not o_term:
                    both = b_def & o_def
                    definite.update(both)
                    conditional.update((b_def | b_cond | o_def | o_cond) - definite)
                eff_b_del = _extract_deleted_names(stmt.body) - b_def
                eff_o_del = _extract_deleted_names(stmt.orelse) - o_def
                for v_del in eff_b_del & eff_o_del:
                    definite.discard(v_del)
                    conditional.discard(v_del)
                _demote_deletions_to_conditional(eff_b_del ^ eff_o_del, definite, conditional)
            else:
                if not b_term:
                    conditional.update((b_def | b_cond) - definite)
                    eff_b_del = _extract_deleted_names(stmt.body) - b_def
                    _demote_deletions_to_conditional(eff_b_del, definite, conditional)
        elif isinstance(stmt, ast.While):
            d_test, c_test = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_test)
            conditional.update(c_test)
            sub_stmts = list(stmt.body)
            if hasattr(stmt, "orelse") and stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            sub_def, sub_cond = _analyze_block_assignment(sub_stmts)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(_extract_deleted_names(sub_stmts), definite, conditional)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            d_iter, c_iter = _walrus_assignment_in_expr(stmt.iter)
            definite.update(d_iter)
            conditional.update(c_iter)
            for p_node in ast.walk(stmt.target):
                if isinstance(p_node, ast.Name):
                    conditional.add(p_node.id)
            sub_stmts = list(stmt.body)
            if hasattr(stmt, "orelse") and stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            sub_def, sub_cond = _analyze_block_assignment(sub_stmts)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(_extract_deleted_names(sub_stmts), definite, conditional)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                d_ctx, c_ctx = _walrus_assignment_in_expr(item.context_expr)
                definite.update(d_ctx)
                conditional.update(c_ctx)
                if item.optional_vars is not None:
                    for p_node in ast.walk(item.optional_vars):
                        if isinstance(p_node, ast.Name) and isinstance(p_node.ctx, ast.Store):
                            definite.add(p_node.id)
            _analyze_block_assignment(stmt.body, definite, conditional)
        elif isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            t_body = list(getattr(stmt, "body", []))
            t_def, t_cond = _analyze_block_assignment(t_body)
            t_term = _block_terminates(t_body)
            o_stmts = list(getattr(stmt, "orelse", []))
            if o_stmts:
                o_def, o_cond = _analyze_block_assignment(o_stmts)
                o_term = _block_terminates(o_stmts)
                if not o_term:
                    try_def = t_def | o_def
                    try_cond = (t_cond | o_cond) - try_def
                else:
                    try_def = set()
                    try_cond = set()
                    t_term = True
            else:
                try_def = t_def
                try_cond = t_cond

            handlers = list(getattr(stmt, "handlers", []))
            h_defs: List[Set[str]] = []
            h_dels: List[Tuple[Set[str], bool]] = []
            all_handlers_terminate = bool(
                handlers and all(_block_terminates(getattr(h, "body", [])) for h in handlers)
            )

            for h in handlers:
                h_body = list(getattr(h, "body", []))
                if getattr(h, "type", None) is not None:
                    d_ht, c_ht = _walrus_assignment_in_expr(h.type, is_conditional=True)
                    conditional.update(d_ht | c_ht)
                h_d, h_c = _analyze_block_assignment(h_body)
                h_term = _block_terminates(h_body)
                eff_h_del = _extract_deleted_names(h_body) - h_d
                h_dels.append((eff_h_del, h_term))
                if not h_term:
                    h_defs.append(h_d)
                conditional.update((h_d | h_c) - definite)

            if not t_term:
                if all_handlers_terminate:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
                elif handlers and h_defs:
                    common = try_def.intersection(*h_defs)
                    definite.update(common)
                    conditional.update((try_def | try_cond) - definite)
                elif not handlers:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
            elif h_defs:
                common = set.intersection(*h_defs)
                definite.update(common)
                conditional.update(try_cond - definite)

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
                _analyze_block_assignment(f_body, definite, conditional)
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):  # pragma: no cover (py310+)
            d_sub, c_sub = _walrus_assignment_in_expr(getattr(stmt, "subject", None))
            definite.update(d_sub)
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
                        p_name = getattr(p_node, "name", None)
                        if p_name and isinstance(p_name, str):
                            pattern_stores.add(p_name)
                        p_rest = getattr(p_node, "rest", None)
                        if p_rest and isinstance(p_rest, str):
                            pattern_stores.add(p_rest)
                guard = getattr(case, "guard", None)
                d_g: Set[str] = set()
                c_g: Set[str] = set()
                if guard is not None:
                    d_g, c_g = _walrus_assignment_in_expr(guard, is_conditional=False)
                case_init_def = pattern_stores | d_g
                c_def, c_cond = _analyze_block_assignment(
                    case.body,
                    definite=case_init_def.copy(),
                    conditional=c_g.copy() if c_g else None,
                )
                case_defs.append(c_def)
                case_conds.update(c_def | c_cond)
                c_del = _extract_deleted_names(case.body) - c_def
                case_dels.update(c_del)
                if not _block_terminates(case.body):
                    non_term_case_defs.append(c_def)
                    non_term_case_dels.append(c_del)
                if _is_irrefutable_case(case):
                    has_irrefutable_default = True
            if case_defs and has_irrefutable_default:
                if non_term_case_defs:
                    common = set.intersection(*non_term_case_defs)
                    definite.update(common)
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
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                definite.add(bound_name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    bound_name = alias.asname or alias.name
                    definite.add(bound_name)
        elif isinstance(stmt, ast.Assert):
            d_ast, c_ast = _walrus_assignment_in_expr(stmt.test)
            definite.update(d_ast)
            conditional.update(c_ast)
            if stmt.msg is not None:
                d_m, c_m = _walrus_assignment_in_expr(stmt.msg, is_conditional=True)
                conditional.update(d_m | c_m)

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

def _inspect_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts lexical and AST scope metadata for a single unit."""
    raw_lines = _slice_unit_token_lines(unit, extract_unit_source_code(unit, repo_root=repo_root))
    dedented = textwrap.dedent("".join(raw_lines))
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
        "has_return": False,
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
    }
    if not dedented.strip():
        return empty_res

    tree: Optional[ast.AST] = None
    loop_offset = 0
    candidate_stmts: List[ast.stmt] = []

    parse_candidates = [
        (dedented, 0),
        (f"async def _wrapper():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"def _wrapper():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"async def _wrapper():\n    for _ in (0,):\n{textwrap.indent(dedented, '        ')}", 1),
        (f"def _wrapper():\n    for _ in (0,):\n{textwrap.indent(dedented, '        ')}", 1),
    ]

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
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue

    if tree is None:
        return empty_res

    unit_name = str(unit.get("name") or "")
    unit_kind = str(unit.get("kind") or "")
    is_subroutine = unit_kind in ("compound_block", "sliding_window", "clause_branch") or (
        unit_kind not in ("function", "closure", "method", "comprehension", "complex_expr") and ":" in unit_name
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

    cand_lines = cand_text.splitlines(keepends=True)
    is_sub = is_subroutine or (
        cand_text == dedented and isinstance(tree, ast.Module) and len(tree.body) > 1
    )
    visitor = _ScopeVisitor(
        loop_offset=loop_offset,
        source_lines=cand_lines,
        is_subroutine=is_sub,
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

    is_control_flow_safe = not any(
        h in ("naked_break", "naked_continue", "embedded_return") for h in hazards
    )

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

    # Definite and conditional assignment analysis across candidate statements
    def_assigned, _ = _analyze_block_assignment(candidate_stmts)

    # Unit outputs: explicit returns if present; for subroutines without explicit
    # returns, all local variable stores act as unit outputs to preserve caller mutations.
    if visitor.returns:
        outputs = list(visitor.returns)
    elif is_subroutine:
        outputs = [
            v for v in visitor.stores
            if v not in visitor.globals
            and v not in BUILTIN_NAMES
            and v not in visitor.except_vars
            and v not in visitor.deleted_names
            and v not in visitor.imported_names
        ]
    else:
        outputs = []

    for nl in visitor.nonlocals:
        if (
            nl in visitor.stores
            and nl not in outputs
            and nl not in visitor.except_vars
            and nl not in visitor.deleted_names
            and nl not in visitor.imported_names
        ):
            outputs.append(nl)

    # Detect conditional variable escapes (outputs assigned conditionally without prior
    # unconditional assignment in this block, and not provided as input arguments).
    conditional_outputs = [
        v for v in outputs
        if v not in inputs and v not in def_assigned
    ]

    has_receiver_access = bool(
        any(
            r in visitor.loads
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            for r in all_receivers
            if r in ("self", "cls", "this", "klass")
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
        )
    )

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
        "has_return": visitor.has_return,
        "has_super": visitor.has_super,
        "has_mangled_names": visitor.has_mangled_names,
        "local_imports": visitor.local_imports,
        "yield_expr_names": visitor.yield_expr_names,
        "is_async": visitor.is_async,
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
    }



def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping to determine inputs, outputs, closures, and attributes."""
    info1 = _inspect_unit_scope(u1, repo_root=repo_root)
    if u2 is not None:
        info2 = _inspect_unit_scope(u2, repo_root=repo_root)
        common_inputs = [var for var in info1["inputs"] if var in info2["inputs"]]
        if len(info1["inputs"]) == len(info2["inputs"]):
            inputs = common_inputs if len(common_inputs) == len(info1["inputs"]) else info1["inputs"]
        else:
            inputs = common_inputs if common_inputs else info1["inputs"]
        common_outputs = [var for var in info1["outputs"] if var in info2["outputs"]]
        if len(info1["outputs"]) == len(info2["outputs"]):
            outputs = common_outputs if len(common_outputs) == len(info1["outputs"]) else info1["outputs"]
        else:
            outputs = common_outputs if common_outputs else info1["outputs"]
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
        has_return = info1["has_return"] or info2["has_return"]
        has_super = bool(info1.get("has_super", False) or info2.get("has_super", False))
        has_mangled = bool(info1.get("has_mangled_names", False) or info2.get("has_mangled_names", False))
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
        has_return = info1["has_return"]
        has_super = bool(info1.get("has_super", False))
        has_mangled = bool(info1.get("has_mangled_names", False))
        local_imports = info1["local_imports"]
        local_imports_by_unit = [info1["local_imports"]]
        yield_expr_names = info1["yield_expr_names"]
        is_async = info1.get("is_async", False)
        conditional_outputs = info1.get("conditional_outputs", [])
        definite_stores = info1.get("definite_stores", [])

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

    primary_rec = u1.get("receiver_param") or (u2.get("receiver_param") if u2 else None) or (
        "cls" if (u1.get("receiver_kind") == "class" or (u2 and u2.get("receiver_kind") == "class")) else None
    )
    _normalize_receiver_order(inputs, has_instance_binding, has_class_binding, primary_receiver=primary_rec)

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
        "has_return": has_return,
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
        "instance_attrs": list(dict.fromkeys(
            info1.get("instance_attrs", []) + (info2.get("instance_attrs", []) if u2 is not None else [])
        )),
        "class_attrs": list(dict.fromkeys(
            info1.get("class_attrs", []) + (info2.get("class_attrs", []) if u2 is not None else [])
        )),
    }



def dispatch_analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatches analyze_unit_variable_scope, honoring active mock patches on pydoppelgangerhunt.fixer."""
    pkg = sys.modules.get("pydoppelgangerhunt.fixer")
    target = getattr(pkg, "analyze_unit_variable_scope", analyze_unit_variable_scope)
    return target(u1, u2=u2, repo_root=repo_root)
