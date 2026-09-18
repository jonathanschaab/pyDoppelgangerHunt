"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import ast
import builtins
import difflib
import io
import logging
import os
from pathlib import Path
import re
import textwrap
import tokenize
import typing
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import normalize_path_string, paths_match_boundary
from pydoppelgangerhunt.parser import is_decorator_named
from pydoppelgangerhunt.reporters import extract_unit_source_code

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _is_mangled_name(name: str) -> bool:
    """Checks if an identifier is subject to Python private name mangling (__foo, not __foo__)."""
    return name.startswith("__") and not name.endswith("__") and len(name) > 2


def _unfold_receiver_attribute(node: ast.AST) -> Optional[str]:
    """Unfolds chained attribute access on 'self' or 'cls' (e.g. self.config.timeout -> 'self.config.timeout')."""
    parts: List[str] = []
    curr: ast.AST = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name) and curr.id in ("self", "cls"):
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, returns, nonlocals, globals, and attributes."""

    def __init__(
        self,
        loop_offset: int = 0,
        source_lines: Optional[Sequence[str]] = None,
        is_subroutine: bool = False,
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
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse argument annotation %r: %s", arg_node.annotation, exc)
                type_val = None

        default_val = None
        if default_node is not None:
            try:
                default_val = ast.unparse(default_node)
            except Exception as exc:  # pylint: disable=broad-exception-caught
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
            except Exception as exc:  # pylint: disable=broad-exception-caught
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
        attr_name = _unfold_receiver_attribute(node)
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
            attr_name = _unfold_receiver_attribute(node.target)
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
            if node.value is not None and isinstance(node.value, ast.Name):
                self.yield_expr_names.append((kind, node.value.id))

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
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse import node %r: %s", node, exc)
                return

        if stmt not in self.local_imports:
            self.local_imports.append(stmt)
        for alias in node.names:
            if isinstance(node, ast.Import):
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            else:
                bound_name = alias.asname or alias.name
            if bound_name != "*":
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


def _is_irrefutable_case(case: ast.AST) -> bool:  # pragma: no cover (py310+)
    """Returns True if the match case has no guard and an irrefutable wildcard or as-pattern."""
    guard = getattr(case, "guard", None)
    pattern = getattr(case, "pattern", None)
    if guard is not None or pattern is None:
        return False
    pat_type = type(pattern).__name__
    return pat_type == "MatchWildcard" or (
        pat_type == "MatchAs" and getattr(pattern, "pattern", None) is None
    )


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
                if guard is not None:
                    d_g, c_g = _walrus_assignment_in_expr(guard, is_conditional=True)
                    case_conds.update(d_g | c_g)
                c_def, c_cond = _analyze_block_assignment(case.body)
                c_def.update(pattern_stores)
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
) -> None:
    """Ensures self or cls is positioned as the primary receiver parameter in inputs."""
    if has_instance_binding and has_class_binding:
        if "cls" in inputs:
            inputs.remove("cls")
            inputs.insert(0, "cls")
        if "self" in inputs:
            inputs.remove("self")
            inputs.insert(0, "self")
    elif has_instance_binding:
        if "self" in inputs:
            inputs.remove("self")
            inputs.insert(0, "self")
    elif has_class_binding:
        if "cls" in inputs:
            inputs.remove("cls")
            inputs.insert(0, "cls")


def _rank_param_kind(var_name: str, param_map: Dict[str, str]) -> int:
    """Returns canonical parameter sorting rank: self/cls=0, pos=1, vararg=2, kwonly=3, kwarg=4."""
    clean = var_name.lstrip("*")
    if clean in ("self", "cls"):
        return 0
    kind = param_map.get(clean, "pos")
    if var_name.startswith("**") or kind == "kwarg":
        return 4
    if kind == "kwonly":
        return 3
    if var_name.startswith("*") or kind == "vararg":
        return 2
    return 1


def _format_call_arguments(
    inputs: List[str],
    param_details: List[Dict[str, Any]],
    receiver_to_omit: Optional[str] = None,
    target_inputs: Optional[List[str]] = None,
) -> str:
    """Formats argument strings for helper call sites, preserving keyword-only, vararg, and kwarg syntax."""
    param_map: Dict[str, str] = {
        p["name"].lstrip("*"): p.get("kind", "pos") for p in param_details
    }
    sorted_inputs = sorted(inputs, key=lambda v: _rank_param_kind(v, param_map))
    if target_inputs is not None and len(target_inputs) == len(inputs):
        inp_to_target = dict(zip(inputs, target_inputs))
        sorted_targets = [inp_to_target.get(k, k) for k in sorted_inputs]
    else:
        sorted_targets = sorted_inputs

    formatted_args: List[str] = []
    for h_var, t_var in zip(sorted_inputs, sorted_targets):
        clean_h = h_var.lstrip("*")
        clean_t = t_var.lstrip("*")
        if receiver_to_omit and (
            clean_t == receiver_to_omit
            or (receiver_to_omit == "receivers" and clean_t in ("self", "cls"))
        ):
            continue
        kind = param_map.get(clean_h, "")
        if kind == "kwonly":
            formatted_args.append(f"{clean_h}={clean_t}")
        elif kind == "vararg" or (h_var.startswith("*") and not h_var.startswith("**")):
            formatted_args.append(f"*{clean_t}")
        elif kind == "kwarg" or h_var.startswith("**"):
            formatted_args.append(f"**{clean_t}")
        else:
            formatted_args.append(clean_t)
    return ", ".join(formatted_args)


def _extract_unit_body_lines(unit: Dict[str, Any], raw_lines: List[str]) -> List[str]:
    """Extracts executable body lines for a unit, stripping function headers and docstrings for whole functions."""
    if unit.get("kind") not in ("function", "closure"):
        return raw_lines

    code_block = "\n".join(raw_lines)
    dedented = textwrap.dedent(code_block)
    try:
        tree = ast.parse(dedented)
        if len(tree.body) == 1 and isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            fn_node = tree.body[0]
            body_nodes = list(fn_node.body)
            if (
                body_nodes
                and isinstance(body_nodes[0], ast.Expr)
                and isinstance(body_nodes[0].value, ast.Constant)
                and isinstance(body_nodes[0].value.value, str)
            ):
                body_nodes = body_nodes[1:]
            if body_nodes:
                d_lines = dedented.splitlines()
                first_body = body_nodes[0]
                start_l = first_body.lineno
                end_l = getattr(body_nodes[-1], "end_lineno", len(d_lines))
                extracted = d_lines[start_l - 1 : end_l]
                if extracted:
                    if start_l == fn_node.lineno:
                        b_col = getattr(first_body, "col_offset", 0)
                        extracted[0] = extracted[0][b_col:]
                    return textwrap.dedent("\n".join(extracted)).splitlines()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return raw_lines


def _detect_indent_step(indent_str: str) -> str:
    """Detects indentation step (tab, 2 spaces, or 4 spaces) from an indentation prefix."""
    if "\t" in indent_str:
        return "\t"
    if indent_str:
        num_spaces = len(indent_str)
        if num_spaces > 0 and num_spaces % 4 == 0:
            return "    "
        if num_spaces > 0 and num_spaces % 2 == 0:
            return "  "
    return "    "


def _find_header_cookie_boundary(lines: List[str]) -> int:
    """Finds the line index after shebang and PEP 263 source encoding declarations.

    PEP 263 allows encoding cookies on line 1 or line 2 following an initial comment.
    """
    boundary = 0
    if lines and lines[0].startswith("#!"):
        boundary = max(boundary, 1)

    for idx in range(min(2, len(lines))):
        line = lines[idx].strip()
        if line.startswith("#") and ("coding:" in line or "coding=" in line):
            boundary = max(boundary, idx + 1)

    return boundary


def _find_module_helper_insertion_index(lines: List[str]) -> int:
    """Finds the line index after module docstring, future imports, and all module imports."""
    min_insert_idx = _find_header_cookie_boundary(lines)

    try:
        tree = ast.parse("".join(lines))
    except SyntaxError:
        return min_insert_idx

    last_import_line = 0
    docstring_line = 0

    for idx, node in enumerate(tree.body):
        if (
            idx == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docstring_line = getattr(node, "end_lineno", node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            end_l = getattr(node, "end_lineno", node.lineno)
            last_import_line = max(last_import_line, end_l)
        else:
            break

    target_line = max(last_import_line, docstring_line, min_insert_idx)
    return min(target_line, len(lines))


def _slice_unit_token_lines(unit: Dict[str, Any], lines: List[str]) -> List[str]:
    """Slices source lines to the exact start_col and end_col offsets of the unit."""
    if not lines or unit.get("kind") not in ("comprehension", "complex_expr"):
        return lines
    s_col = unit.get("start_col", 0) or 0
    e_col = unit.get("end_col")
    res = list(lines)
    if len(res) == 1:
        res[0] = res[0][s_col:e_col]
    else:
        res[0] = res[0][s_col:]
        if e_col is not None:
            res[-1] = res[-1][:e_col]
    return res


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
        except SyntaxError:
            continue

    if tree is None:
        return empty_res

    unit_name = str(unit.get("name") or "")
    unit_kind = str(unit.get("kind") or "")
    is_subroutine = unit_kind in ("compound_block", "sliding_window", "clause_branch") or (
        unit_kind not in ("function", "closure", "comprehension", "complex_expr") and ":" in unit_name
    )

    cand_lines = cand_text.splitlines(keepends=True)
    is_sub = is_subroutine or (
        cand_text == dedented and isinstance(tree, ast.Module) and len(tree.body) > 1
    )
    visitor = _ScopeVisitor(
        loop_offset=loop_offset,
        source_lines=cand_lines,
        is_subroutine=is_sub,
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
    inputs.sort(key=lambda v: _rank_param_kind(v, param_map))

    has_instance_binding = (
        "self" in visitor.loads
        or "self" in visitor.params
        or any(a.startswith("self.") for a in visitor.attrs_read + visitor.attrs_written)
    )
    has_class_binding = (
        "cls" in visitor.loads
        or "cls" in visitor.params
        or any(a.startswith("cls.") for a in visitor.attrs_read + visitor.attrs_written)
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

    _normalize_receiver_order(inputs, has_instance_binding, has_class_binding)

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
        "self" in visitor.loads
        or "cls" in visitor.loads
        or any(a.startswith("self.") or a.startswith("cls.") for a in visitor.attrs_read + visitor.attrs_written)
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
        "instance_attrs": [a for a in visitor.attrs_read + visitor.attrs_written if a.startswith("self.")],
        "class_attrs": [a for a in visitor.attrs_read + visitor.attrs_written if a.startswith("cls.")],
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

    _normalize_receiver_order(inputs, has_instance_binding, has_class_binding)

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


def _merge_types(t1: Optional[str], t2: Optional[str], strategy: str) -> str:
    """Merges two type annotations according to the specified merge strategy."""
    if t1 and t2:
        if t1 == t2:
            return t1
        if strategy == "union":
            return f"Union[{t1}, {t2}]"
        return "Any"
    return t1 or t2 or "Any"


TYPING_SYMBOLS: Set[str] = set(typing.__all__)


def _extract_required_typing_imports(signature_or_func_text: str) -> List[str]:
    """Identifies typing symbols referenced in type annotations via AST inspection."""
    text = signature_or_func_text.strip()
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        pass

    if tree is None and "->" in text:
        parts = text.split("->", 1)
        params_part = parts[0].strip()
        if params_part.startswith("(") and params_part.endswith(")"):
            params_part = params_part[1:-1].strip()
        ret_part = parts[1].strip()
        try:
            tree = ast.parse(f"def _sig_wrapper({params_part}) -> {ret_part}: pass")
        except SyntaxError:
            pass

    if tree is None:
        try:
            tree = ast.parse(f"def _sig_wrapper({text}): pass")
        except SyntaxError:
            pass

    if tree is None:
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        return sorted(s for s in TYPING_SYMBOLS if s in tokens)

    has_defs = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign))
        for n in ast.walk(tree)
    )
    needed: Set[str] = set()
    for node in ast.walk(tree):
        annotations: List[ast.AST] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            for a in all_args:
                if a.annotation is not None:
                    annotations.append(a.annotation)
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None:
                annotations.append(node.annotation)
        elif not has_defs and isinstance(node, ast.Expr):
            annotations.append(node.value)

        for ann in annotations:
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id in TYPING_SYMBOLS:
                    needed.add(sub.id)
    return sorted(needed)


def _get_module_imported_names(source: str) -> Set[str]:
    """Extracts top-level imported module and symbol names from source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imported: Set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
    return imported


def _insert_imports_into_module(
    orig_lines: List[str],
    import_lines: List[str],
) -> List[str]:
    """Inserts import statements after module docstring and future imports, deduplicating existing statements."""
    if not import_lines:
        return orig_lines

    existing_stripped = {ln.strip() for ln in orig_lines}
    seen: Set[str] = set()
    deduped_imports: List[str] = []
    for imp in import_lines:
        s = imp.strip()
        if s not in existing_stripped and s not in seen:
            seen.add(s)
            deduped_imports.append(imp)
    if not deduped_imports:
        return orig_lines

    insert_idx = _find_header_cookie_boundary(orig_lines)

    parsed_with_ast = False
    try:
        tree = ast.parse("".join(orig_lines))
        doc_end = 0
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            doc_end = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
        future_end = max(
            (
                getattr(node, "end_lineno", node.lineno)
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            ),
            default=0,
        )
        insert_idx = max(insert_idx, doc_end, future_end)
        while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
            insert_idx += 1
        parsed_with_ast = True
    except SyntaxError:
        pass

    if not parsed_with_ast:
        in_docstring = False
        doc_quote = ""
        doc_prefixes = ('"""', "'''", 'r"""', "r'''", 'R"""', "R'''", 'u"""', "u'''", 'U"""', "U'''")
        for idx in range(insert_idx, len(orig_lines)):
            line = orig_lines[idx].strip()
            if not in_docstring:
                matching_pfx = next((p for p in doc_prefixes if line.startswith(p)), None)
                if matching_pfx:
                    doc_quote = matching_pfx[-3:]
                    if line.endswith(doc_quote) and len(line) > len(matching_pfx):
                        insert_idx = idx + 1
                        break
                    in_docstring = True
                elif line.startswith("#") or not line:
                    continue
                else:
                    break
            else:
                if line.endswith(doc_quote):
                    insert_idx = idx + 1
                    break

        last_future_idx = -1
        for idx in range(insert_idx, len(orig_lines)):
            line = orig_lines[idx].strip()
            if line.startswith("from __future__"):
                last_future_idx = idx
            elif line.startswith("#") or not line:
                continue
            else:
                break

        if last_future_idx != -1:
            insert_idx = last_future_idx + 1
            while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
                insert_idx += 1

    formatted = [imp.rstrip("\r\n") + "\n" for imp in deduped_imports]
    return orig_lines[:insert_idx] + formatted + ["\n"] + orig_lines[insert_idx:]


def slice_source_by_token_range(
    source_text: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: Optional[int] = None,
) -> str:
    """Extracts a precise slice of source code spanning 1-indexed lines and 0-indexed columns.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number.
        start_col: 0-indexed start column offset.
        end_line: 1-indexed end line number.
        end_col: 0-indexed end column offset (exclusive). If None, includes through the end of end_line.

    Returns:
        The exact string slice of source code between the specified coordinates.
    """
    if not source_text:
        return ""
    lines = source_text.splitlines(keepends=True)
    if not lines or start_line < 1 or start_line > len(lines) or start_line > end_line:
        return ""

    effective_end_line = min(len(lines), end_line)
    if start_line == effective_end_line:
        line_str = lines[start_line - 1]
        start_c = max(0, min(len(line_str), start_col))
        end_c = len(line_str) if end_col is None else max(start_c, min(len(line_str), end_col))
        return line_str[start_c:end_c]

    first_line = lines[start_line - 1]
    start_c = max(0, min(len(first_line), start_col))
    part_first = first_line[start_c:]
    middle_parts = lines[start_line : effective_end_line - 1]
    last_line = lines[effective_end_line - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), end_col))
    part_last = last_line[:end_c]
    return part_first + "".join(middle_parts) + part_last


def extract_unit_comments_and_pragmas(
    source_text: str,
    start_line: int,
    end_line: int,
    start_col: int = 0,
    end_col: Optional[int] = None,
    include_leading: bool = False,
) -> List[Dict[str, Any]]:
    """Extracts comments and pragmas within or attached to a unit's coordinate bounds.

    Uses Python's standard library `tokenize` module to harvest exact token positions
    for comments, identifying `# type: ignore`, `# noqa`, and `# pylint:` annotations.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number of the unit.
        end_line: 1-indexed end line number of the unit.
        start_col: 0-indexed start column offset.
        end_col: 0-indexed end column offset.
        include_leading: If True, also harvests consecutive comment lines immediately
            preceding start_line.

    Returns:
        A list of dictionaries representing discovered comments:
        [{
            "line": int,
            "col": int,
            "end_col": int,
            "text": str,
            "is_pragma": bool,
            "pragma_kind": Optional[str],
        }]
    """
    if not source_text:
        return []

    lines = source_text.splitlines(keepends=True)
    min_check_line = start_line
    if include_leading and start_line > 1:
        curr = start_line - 1
        while curr >= 1:
            stripped = lines[curr - 1].strip()
            if stripped.startswith("#"):
                min_check_line = curr
                curr -= 1
            else:
                break

    results: List[Dict[str, Any]] = []
    try:
        token_gen = tokenize.generate_tokens(io.StringIO(source_text).readline)
        for tok in token_gen:
            if tok.type == tokenize.COMMENT:
                tok_line, tok_col = tok.start
                _, tok_end_col = tok.end

                if min_check_line <= tok_line <= end_line:
                    if tok_line == start_line and tok_line == min_check_line and tok_col < start_col:
                        continue
                    comment_text = tok.string
                    lower = comment_text.lower()
                    is_pragma = False
                    pragma_kind = None
                    if "type: ignore" in lower:
                        is_pragma = True
                        pragma_kind = "type_ignore"
                    elif "noqa" in lower:
                        is_pragma = True
                        pragma_kind = "noqa"
                    elif "pylint:" in lower:
                        is_pragma = True
                        pragma_kind = "pylint"

                    results.append({
                        "line": tok_line,
                        "col": tok_col,
                        "end_col": tok_end_col,
                        "text": comment_text,
                        "is_pragma": is_pragma,
                        "pragma_kind": pragma_kind,
                    })
    except (tokenize.TokenError, IndentationError) as exc:
        logger.debug("Failed to tokenize source for comments: %s", exc)

    return results


def _find_innermost_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
) -> Optional[Tuple[ast.AST, int, int]]:
    """Locates the innermost AST node of matching types enclosing the given unit."""
    if not source_text.strip():
        return None

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    u_start = int(unit.get("start") or 0)
    u_end = int(unit.get("end") or u_start)
    if u_start <= 0:
        return None

    candidates: List[Tuple[int, ast.AST, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, node_types):
            n_start = getattr(node, "lineno", 0)
            n_end = getattr(node, "end_lineno", n_start)
            decorators = getattr(node, "decorator_list", [])
            dec_start = (
                min(getattr(d, "lineno", n_start) for d in decorators)
                if decorators
                else n_start
            )
            earliest_start = min(dec_start, n_start)
            if earliest_start <= u_start <= u_end <= n_end:
                candidates.append((n_end - earliest_start, node, earliest_start, n_end))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[2]))
    _, matched_node, n_start, n_end = candidates[0]
    return matched_node, n_start, n_end


def _inspect_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
) -> Optional[Dict[str, Any]]:
    """Locates and extracts metadata for the innermost enclosing AST node of matching types."""
    res = _find_innermost_enclosing_node(source_text, unit, node_types)
    if not res:
        return None
    matched_node, start, end = res
    decorators = list(getattr(matched_node, "decorator_list", []))
    return {
        "node": matched_node,
        "name": getattr(matched_node, "name", ""),
        "start": start,
        "def_start": getattr(matched_node, "lineno", start),
        "end": end,
        "decorators": decorators,
    }


def _extract_child_indentation(
    lines: Sequence[str],
    line_numbers: Iterable[int],
    parent_indent: str,
) -> Optional[str]:
    """Finds the indentation of the first line from line_numbers that is more indented than parent_indent."""
    for line_num in line_numbers:
        if 1 <= line_num <= len(lines):
            line_str: str = str(lines[line_num - 1])
            ind: str = line_str[: len(line_str) - len(line_str.lstrip())]
            if ind.startswith(parent_indent) and len(ind) > len(parent_indent):
                return str(ind)
    return None


def find_enclosing_class(
    source_text: str,
    unit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locates the innermost enclosing ClassDef for an AST code unit.

    Args:
        source_text: The complete original Python source code.
        unit: AST unit dictionary with 1-indexed 'start' and 'end' line bounds.

    Returns:
        A dictionary with class metadata ('name', 'start', 'end', 'indent', 'method_indent')
        if the unit is enclosed within a ClassDef; None otherwise.
    """
    meta = _inspect_enclosing_node(source_text, unit, (ast.ClassDef,))
    if not meta:
        return None

    lines = source_text.splitlines(keepends=True)
    c_start = meta["start"]
    cls_line = lines[c_start - 1] if 1 <= c_start <= len(lines) else ""
    indent = cls_line[: len(cls_line) - len(cls_line.lstrip())]
    meta["indent"] = indent

    node = meta.get("node")
    suite_indent = None
    def_start = meta.get("def_start", c_start)
    direct_methods: Set[Tuple[int, int]] = set()
    if isinstance(node, ast.ClassDef) and node.body:
        cand_lines: List[int] = []
        for stmt in node.body:
            stmt_lineno = int(getattr(stmt, "lineno", 0))
            earliest_line = stmt_lineno
            decorators = getattr(stmt, "decorator_list", [])
            if decorators:
                dec_lines = [int(getattr(d, "lineno", stmt_lineno)) for d in decorators]
                if dec_lines:
                    dec_start = min(dec_lines)
                    earliest_line = min(dec_start, earliest_line) if earliest_line > 0 else dec_start
            if earliest_line > def_start:
                cand_lines.append(earliest_line)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_l = int(getattr(stmt, "end_lineno", stmt_lineno))
                direct_methods.add((earliest_line, end_l))
                direct_methods.add((stmt_lineno, end_l))
        suite_indent = _extract_child_indentation(lines, cand_lines, indent)

    meta["methods"] = direct_methods
    meta["method_indent"] = suite_indent if suite_indent is not None else (indent + "    ")
    meta.pop("node", None)
    meta.pop("decorators", None)
    meta.pop("def_start", None)
    return meta


def find_enclosing_function(
    source_text: str,
    unit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locates the innermost enclosing FunctionDef/AsyncFunctionDef for an AST code unit."""
    meta = _inspect_enclosing_node(
        source_text, unit, (ast.FunctionDef, ast.AsyncFunctionDef)
    )
    if not meta:
        return None

    decs = meta.pop("decorators", [])
    meta.pop("node", None)
    meta["is_static"] = any(is_decorator_named(d, "staticmethod") for d in decs)
    meta["is_class_method"] = any(is_decorator_named(d, "classmethod") for d in decs)
    return meta


def _normalize_file_path(
    f_str: str,
    repo_root: Optional[str] = None,
) -> str:
    """Normalizes a file path string to a canonical resolved POSIX path."""
    norm = normalize_path_string(f_str)
    if not norm:
        return ""
    root = Path(repo_root or os.getcwd())
    try:
        p = Path(norm)
        p_full = p if p.is_file() or p.is_absolute() else (root / p)
        return str(p_full.resolve()).replace("\\", "/")
    except OSError:
        return norm


def _is_same_file_path(
    f1_str: str,
    f2_str: str,
    repo_root: Optional[str] = None,
) -> bool:
    """Checks whether two file path strings refer to the identical file on disk."""
    if not f1_str or not f2_str:
        return False
    norm1 = normalize_path_string(f1_str)
    norm2 = normalize_path_string(f2_str)
    if paths_match_boundary(norm1, norm2):
        return True
    root = Path(repo_root or os.getcwd())
    try:
        p1 = Path(norm1)
        p2 = Path(norm2)
        p1_full = p1 if p1.is_file() or p1.is_absolute() else (root / p1)
        p2_full = p2 if p2.is_file() or p2.is_absolute() else (root / p2)
        if p1_full.resolve() == p2_full.resolve():
            return True
    except OSError:
        return False
    return False


def _is_method_of_class(
    fn_meta: Optional[Dict[str, Any]],
    cls_meta: Optional[Dict[str, Any]],
) -> bool:
    """Returns True if the function is a direct method defined inside the enclosing class."""
    if not fn_meta or not cls_meta:
        return False
    methods = cls_meta.get("methods")
    if methods is not None:
        fn_span = (fn_meta["start"], fn_meta["end"])
        fn_def_span = (fn_meta.get("def_start", fn_meta["start"]), fn_meta["end"])
        return fn_span in methods or fn_def_span in methods
    return bool(
        cls_meta["start"] < fn_meta["start"]
        and fn_meta["end"] <= cls_meta["end"]
    )


def _has_receiver_reference(
    unit: Dict[str, Any],
    scope: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
) -> bool:
    """Checks if a unit actually references an instance or class receiver in its executable body."""
    target_scope = (
        scope if scope is not None else analyze_unit_variable_scope(unit, repo_root=repo_root)
    )
    return bool(
        target_scope.get("has_receiver_access")
        or target_scope.get("instance_attrs")
        or target_scope.get("class_attrs")
    )


def _prune_unshared_receivers(
    inputs: List[str],
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> List[str]:
    """Removes 'self' and 'cls' from inputs when only one unit receives them and neither references them."""
    res = list(inputs)
    for rec in ("self", "cls"):
        if (rec in scope1.get("inputs", [])) != (rec in scope2.get("inputs", [])):
            if not (
                _has_receiver_reference(u1, scope1, repo_root=repo_root)
                or _has_receiver_reference(u2, scope2, repo_root=repo_root)
            ):
                res = [v for v in res if v != rec]
    return res


def _get_enclosing_receiver_kind(fn_info: Optional[Dict[str, Any]]) -> str:
    """Returns the receiver kind for an enclosing function: static, class, instance, or none."""
    if not fn_info:
        return "none"
    if fn_info.get("is_static"):
        return "static"
    if fn_info.get("is_class_method"):
        return "class"
    return "instance"


def _populate_unit_receiver_metadata(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> None:
    """Populates receiver_kind, enclosing_class, and is_static on a unit from enclosing AST nodes if absent."""
    if (
        "receiver_kind" in unit
        and "enclosing_class" in unit
        and "enclosing_class_start" in unit
    ):
        return
    f_raw = normalize_path_string(str(unit.get("file") or ""), strip_anchor=True)
    if not f_raw:
        return
    p = Path(f_raw)
    f_path = p if p.is_file() or p.is_absolute() else (Path(repo_root or os.getcwd()) / p)
    if not f_path.is_file():
        return
    try:
        source = f_path.read_text(encoding="utf-8")
        fn_meta = find_enclosing_function(source, unit)
        cls_meta = find_enclosing_class(source, unit)
        if cls_meta:
            if "enclosing_class" not in unit:
                unit["enclosing_class"] = cls_meta["name"]
            if "enclosing_class_start" not in unit:
                unit["enclosing_class_start"] = cls_meta["start"]
        if "receiver_kind" not in unit:
            if _is_method_of_class(fn_meta, cls_meta):
                rec_k = _get_enclosing_receiver_kind(fn_meta)
                unit["receiver_kind"] = rec_k
                if rec_k == "static":
                    unit["is_static"] = True
            else:
                unit["receiver_kind"] = None
    except (OSError, UnicodeDecodeError):
        pass


def _base_unit_name(u: Dict[str, Any]) -> str:
    """Extracts base function or method name from unit dictionary for helper synthesis."""
    name = str(u.get("name") or "")
    if u.get("kind") == "closure" and ":" in name:
        base = name.rsplit(":", maxsplit=1)[-1].strip("_")
    else:
        base = name.split(":", maxsplit=1)[0].strip("_")
    if "." in base:
        base = base.rsplit(".", maxsplit=1)[-1].strip("_")
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_")
    return base or "helper"



def _resolve_effective_binding(
    method_binding: str,
    is_same_class: bool,
    is_static: bool = False,
    receiver_kinds_differ: bool = False,
    is_in_method: bool = True,
    receiver_kind: Optional[str] = None,
) -> str:
    """Resolve the effective helper binding mode ('method' vs 'module')."""
    if method_binding == "module":
        return "module"

    # Method binding is only viable if both units share the same class,
    # have compatible receiver kinds, and reside within methods (not class body).
    if not is_same_class or receiver_kinds_differ or not is_in_method:
        return "module"

    if method_binding == "method":
        return "method"

    # auto mode
    if is_static or receiver_kind in ("none", None):
        return "module"
    return "method"


def _format_helper_parameters(
    inputs: Sequence[str],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    *,
    effective_binding: str,
    is_static_clone: bool,
    is_class_receiver: bool,
    type_merge_strategy: str,
    inputs2: Optional[Sequence[str]] = None,
) -> List[str]:
    """Formats helper function parameters with type annotations and default values."""
    params: List[str] = []
    seen_kwonly = False

    # Extract parameter descriptors
    descriptors: List[Dict[str, Any]] = []
    for idx, var in enumerate(inputs):
        m1 = meta1.get(var, {})
        m2 = meta2.get(var, {})
        if not m2 and inputs2 and idx < len(inputs2):
            m2 = meta2.get(inputs2[idx], {})

        t1 = m1.get("type")
        t2 = m2.get("type")
        resolved_type = _merge_types(t1, t2, type_merge_strategy)

        d1 = m1.get("default")
        d2 = m2.get("default")
        resolved_default = d1 if (d1 and d2 and d1 == d2) else None

        kind = m1.get("kind") or m2.get("kind") or "pos"
        descriptors.append({
            "var": var,
            "type": resolved_type,
            "default": resolved_default,
            "kind": kind,
        })

    # Validate and ensure canonical argument kind ordering (self/cls -> pos -> vararg -> kwonly -> kwarg)
    def _kind_rank(desc: Dict[str, Any]) -> Tuple[int, int]:
        var = desc["var"]
        kind = desc["kind"]
        if var in ("self", "cls"):
            return (0, 0)
        rank_map = {"pos": 1, "vararg": 2, "kwonly": 3, "kwarg": 4}
        return (rank_map.get(kind, 1), 1)

    descriptors.sort(key=_kind_rank)

    # In method binding mode, ensure receiver parameter exists (unless static)
    if (
        effective_binding == "method"
        and not is_static_clone
        and not any(desc["var"] in ("self", "cls") for desc in descriptors)
    ):
        rec_var = "cls" if is_class_receiver else "self"
        descriptors.insert(0, {
            "var": rec_var,
            "type": "Any",
            "default": None,
            "kind": "pos",
        })

    # Validate positional argument default order (non-default cannot follow default)
    has_pos_default = False
    invalid_pos_defaults = False
    for desc in descriptors:
        if desc["kind"] == "pos":
            if desc["default"] is not None:
                has_pos_default = True
            elif has_pos_default:
                invalid_pos_defaults = True
                break
    if invalid_pos_defaults:
        for desc in descriptors:
            if desc["kind"] == "pos":
                desc["default"] = None

    # Render formatted parameters
    for desc in descriptors:
        var = desc["var"]
        resolved_type = desc["type"]
        resolved_default = desc["default"]
        kind = desc["kind"]
        if kind in ("vararg", "kwarg"):
            resolved_default = None
        prefix = "*" if kind == "vararg" else ("**" if kind == "kwarg" else "")
        var_name = f"{prefix}{var}"

        if (
            kind == "kwonly"
            and not seen_kwonly
            and not any(p.startswith("*") and not p.startswith("**") for p in params)
        ):
            params.append("*")
            seen_kwonly = True

        if var in ("self", "cls") and effective_binding == "method":
            params.append(var)
        elif resolved_default is not None:
            params.append(f"{var_name}: {resolved_type} = {resolved_default}")
        else:
            params.append(f"{var_name}: {resolved_type}")

    return params


def _infer_helper_return_type(
    resolved_ret: str,
    helper_outputs: List[str],
    conditional_outs: Set[str],
    scope: Dict[str, Any],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    is_async: bool = False,
    unit_kind: Optional[str] = None,
    outputs2: Optional[List[str]] = None,
) -> str:
    """Infers the return type annotation for a synthesized shared helper function."""
    if scope.get("has_yield"):
        if is_async:
            return "AsyncIterator[Any]"
        inferred_yield_type = None
        for kind, name in scope.get("yield_expr_names", []):
            m_t = meta1.get(name, {}).get("type") or meta2.get(name, {}).get("type")
            if not m_t:
                continue
            if kind == "yield":
                inferred_yield_type = m_t
                break
            if kind == "yield_from":
                for prefix in ("Iterator[", "Iterable[", "List[", "Sequence[", "Set["):
                    if m_t.startswith(prefix) and m_t.endswith("]"):
                        inferred_yield_type = m_t[len(prefix) : -1].strip()
                        break
                if inferred_yield_type:
                    break
        if resolved_ret != "Any":
            return resolved_ret
        if inferred_yield_type:
            return f"Iterator[{inferred_yield_type}]"
        return "Iterator[Any]"

    if len(helper_outputs) >= 2:
        out_types: List[str] = []
        for idx, out_var in enumerate(helper_outputs):
            t1 = meta1.get(out_var, {}).get("type")
            m2 = meta2.get(out_var, {})
            if not m2 and outputs2 and idx < len(outputs2):
                m2 = meta2.get(outputs2[idx], {})
            t2 = m2.get("type")
            t_merged = _merge_types(t1, t2, type_merge_strategy)
            if out_var in conditional_outs:
                if not t_merged.startswith("Optional[") and "None" not in t_merged:
                    t_merged = f"Optional[{t_merged}]"
            out_types.append(t_merged)
        return f"Tuple[{', '.join(out_types)}]"

    if len(helper_outputs) == 1:
        out_var = helper_outputs[0]
        t1 = meta1.get(out_var, {}).get("type")
        m2 = meta2.get(out_var, {})
        if not m2 and outputs2 and len(outputs2) >= 1:
            m2 = meta2.get(outputs2[0], {})
        t2 = m2.get("type")
        out_t = _merge_types(t1, t2, type_merge_strategy)
        if out_var in conditional_outs:
            if not out_t.startswith("Optional[") and "None" not in out_t:
                out_t = f"Optional[{out_t}]"
        return_type = resolved_ret if resolved_ret != "Any" else out_t
        if out_var in conditional_outs:
            if not return_type.startswith("Optional[") and "None" not in return_type and return_type != "None":
                return_type = f"Optional[{return_type}]"
        return return_type

    if not helper_outputs and not scope.get("has_return") and resolved_ret == "Any":
        return "Any" if unit_kind in ("comprehension", "complex_expr") else "None"

    return resolved_ret


def _format_helper_docstring(
    call_target: str,
    helper_outputs: List[str],
    scope: Dict[str, Any],
    *,
    is_async: bool = False,
    doc_indent: str = "    ",
    sub_indent: str = "        ",
    await_prefix: str = "",
) -> str:
    """Formats docstring with call-site example and control flow hazard warnings."""
    docstring_lines = [f"{doc_indent}\"\"\"Auto-extracted shared helper for duplicate logic."]
    is_async_gen = bool(scope.get("has_yield") and is_async)
    is_sync_gen = bool(scope.get("has_yield") and not is_async)
    if is_async_gen:
        call_site = f"async for _item in {call_target}(...):\n{sub_indent}yield _item"
    elif is_sync_gen:
        if helper_outputs:
            assign = ", ".join(helper_outputs) if len(helper_outputs) >= 2 else helper_outputs[0]
            call_site = f"{assign} = (yield from {call_target}(...))"
        else:
            call_site = f"yield from {call_target}(...)"
    elif helper_outputs:
        assign = ", ".join(helper_outputs) if len(helper_outputs) >= 2 else helper_outputs[0]
        call_site = f"{assign} = {await_prefix}{call_target}(...)"
    else:
        call_site = f"{await_prefix}{call_target}(...)"
    docstring_lines.append("")
    docstring_lines.append(f"{doc_indent}Call site:")
    docstring_lines.append(f"{sub_indent}{call_site}")

    hazards = scope.get("control_flow_hazards", [])
    if hazards:
        hazard_str = ", ".join(hazards)
        docstring_lines.append("")
        docstring_lines.append(
            f"{doc_indent}WARNING: Non-local control flow hazard detected ({hazard_str}). "
            "Direct extraction alters caller control flow semantics."
        )
    docstring_lines.append(f"{doc_indent}\"\"\"")
    return "\n".join(docstring_lines)


def synthesize_shared_helper_code(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    include_imports: bool = False,
    preserve_pragmas: bool = True,
    method_binding: str = "auto",
    indent: str = "",
    is_static: bool = False,
    receiver_kind: Optional[str] = None,
    step: Optional[str] = None,
    repo_root: Optional[str] = None,
    helper_name: Optional[str] = None,
) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units.

    Args:
        u1: First clone unit dictionary.
        u2: Second clone unit dictionary.
        type_merge_strategy: Strategy for merging conflicting types.
            Options: "fallback_any" (default, falls back to Any)
            or "union" (suggests typing.Union[T1, T2]).
        include_imports: Whether to prepend required typing and local imports.
        preserve_pragmas: Whether to preserve comments and pragmas (# type: ignore / # noqa)
            from the original clone units in the extracted helper function.
        method_binding: Scope binding strategy ("auto", "method", "module").
            "auto" emits a private class method if both clones reside in the same class,
            or a module-level helper with explicit receiver injection otherwise.
        indent: Base indentation prefix for helper definition lines (defaults to 4 spaces for methods).
        is_static: Whether the cloned methods are static methods.
        receiver_kind: Receiver kind of enclosing method ('class', 'instance', 'static').
        step: Indentation step per indentation level (defaults to "\t" for tabs, 4 spaces otherwise).
        repo_root: Optional root directory of the repository for relative path resolution.
        helper_name: Optional custom helper name override.
    """
    lines1 = _slice_unit_token_lines(
        u1,
        _extract_unit_body_lines(u1, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u1, repo_root=repo_root)]),
    )
    lines2 = _slice_unit_token_lines(
        u2,
        _extract_unit_body_lines(u2, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u2, repo_root=repo_root)]),
    )

    base_name1 = _base_unit_name(u1)
    base_name2 = _base_unit_name(u2)
    if not helper_name:
        helper_name = (
            f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
        )

    # Find common lines using difflib matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])

    if not common_lines or len(common_lines) < len(lines1):
        common_lines = lines1

    if preserve_pragmas:
        raw1 = "\n".join(lines1)
        raw2 = "\n".join(lines2)
        pragmas1 = extract_unit_comments_and_pragmas(raw1, 1, len(lines1))
        pragmas2 = extract_unit_comments_and_pragmas(raw2, 1, len(lines2))
        for p in pragmas1 + pragmas2:
            if p["is_pragma"]:
                p_text = p["text"].strip()
                if not any(p_text in ln for ln in common_lines):
                    for idx, ln in enumerate(common_lines):
                        if ln.strip() and not ln.strip().startswith("#"):
                            common_lines[idx] = ln.rstrip() + f"  {p_text}"
                            break

    _populate_unit_receiver_metadata(u1, repo_root=repo_root)
    _populate_unit_receiver_metadata(u2, repo_root=repo_root)

    # Variable scope analysis for concrete parameter signatures
    scope1 = analyze_unit_variable_scope(u1, repo_root=repo_root)
    scope2 = analyze_unit_variable_scope(u2, repo_root=repo_root)
    scope = analyze_unit_variable_scope(u1, u2, repo_root=repo_root)

    enc1 = u1.get("enclosing_class")
    enc2 = u2.get("enclosing_class")
    f1 = normalize_path_string(str(u1.get("file") or ""))
    f2 = normalize_path_string(str(u2.get("file") or ""))
    enc1_start = u1.get("enclosing_class_start")
    enc2_start = u2.get("enclosing_class_start")
    is_same_class = bool(
        enc1
        and enc2
        and enc1 == enc2
        and _is_same_file_path(f1, f2, repo_root=repo_root)
        and (enc1_start is None or enc2_start is None or enc1_start == enc2_start)
    )

    inputs = list(scope["inputs"])

    def _unit_receiver_kind(u: Dict[str, Any], u_scope: Dict[str, Any]) -> str:
        if u.get("receiver_kind"):
            return str(u["receiver_kind"])
        if "receiver_kind" in u and u.get("receiver_kind") is None:
            return "none"
        if u.get("kind") in ("closure", "class", "comprehension", "data_table", "complex_expr"):
            return "none"
        if u.get("is_static"):
            return "static"
        bk = u_scope.get("binding_kind")
        if bk in ("static", "class", "instance"):
            return str(bk)
        if receiver_kind:
            return receiver_kind
        if is_static:
            return "static"
        if u.get("enclosing_class"):
            return "instance"
        return "none"

    k1 = _unit_receiver_kind(u1, scope1)
    k2 = _unit_receiver_kind(u2, scope2)
    receiver_kinds_differ = bool(k1 != k2)
    if receiver_kinds_differ and (
        _has_receiver_reference(u1, scope1, repo_root=repo_root)
        or _has_receiver_reference(u2, scope2, repo_root=repo_root)
    ):
        return ""

    if bool(scope1.get("is_async")) != bool(scope2.get("is_async")):
        return ""
    if bool(scope1.get("has_yield")) != bool(scope2.get("has_yield")):
        return ""
    if scope1.get("nonlocals") or scope2.get("nonlocals") or scope.get("nonlocals"):
        return ""
    if not is_same_class and not _is_same_file_path(f1, f2, repo_root=repo_root) and (
        scope1.get("globals") or scope2.get("globals") or scope.get("globals")
    ):
        return ""
    if (
        set(scope1.get("attrs_read", [])) != set(scope2.get("attrs_read", []))
        or set(scope1.get("attrs_written", [])) != set(scope2.get("attrs_written", []))
    ):
        return ""

    is_static_clone = bool(
        is_static
        or receiver_kind == "static"
        or (k1 == "static" and k2 == "static")
    )
    is_class_receiver = bool(
        receiver_kind == "class"
        or (k1 == "class" and k2 == "class")
    )

    is_in_method = bool(
        u1.get("kind") not in ("comprehension", "complex_expr", "closure", "class")
        and u2.get("kind") not in ("comprehension", "complex_expr", "closure", "class")
        and k1 != "none"
        and k2 != "none"
    )

    effective_binding = _resolve_effective_binding(
        method_binding,
        is_same_class,
        is_static=is_static_clone,
        receiver_kinds_differ=receiver_kinds_differ,
        is_in_method=is_in_method,
        receiver_kind=k1,
    )

    has_super = bool(scope1.get("has_super") or scope2.get("has_super"))
    if has_super and (effective_binding == "module" or is_static_clone):
        return ""

    has_mangled = bool(scope1.get("has_mangled_names") or scope2.get("has_mangled_names"))
    if has_mangled and (effective_binding != "method" or not is_same_class or is_static_clone):
        return ""

    if effective_binding == "module":
        inputs = _prune_unshared_receivers(inputs, u1, u2, scope1, scope2, repo_root=repo_root)

    if effective_binding == "method" and not indent:
        indent = "    "

    inputs2 = list(scope2.get("inputs", []))
    if effective_binding == "module":
        inputs2 = _prune_unshared_receivers(inputs2, u2, u1, scope2, scope1, repo_root=repo_root)

    meta1 = {p["name"].lstrip("*"): p for p in scope1.get("param_details", [])}
    meta2 = {p["name"].lstrip("*"): p for p in scope2.get("param_details", [])}

    params = _format_helper_parameters(
        inputs,
        meta1,
        meta2,
        effective_binding=effective_binding,
        is_static_clone=is_static_clone,
        is_class_receiver=is_class_receiver,
        type_merge_strategy=type_merge_strategy,
        inputs2=inputs2 if len(inputs2) == len(inputs) else None,
    )

    r1 = scope1.get("return_type")
    r2 = scope2.get("return_type")
    resolved_ret = _merge_types(r1, r2, type_merge_strategy)

    outputs = list(scope.get("outputs", []))
    conditional_outs = set(scope.get("conditional_outputs", []))
    is_async = scope.get("is_async", False)
    func_keyword = "async def" if is_async else "def"
    await_prefix = "await " if is_async else ""

    # Variables altered via global or nonlocal statements must not be captured in the helper's return tuple
    helper_outputs = [
        v for v in outputs
        if v not in scope.get("globals", [])
        and v not in scope.get("nonlocals", [])
    ]

    u2_outs = [
        v for v in scope2.get("outputs", [])
        if v not in scope2.get("globals", [])
        and v not in scope2.get("nonlocals", [])
    ]

    return_type = _infer_helper_return_type(
        resolved_ret,
        helper_outputs,
        conditional_outs,
        scope,
        meta1,
        meta2,
        type_merge_strategy=type_merge_strategy,
        is_async=is_async,
        unit_kind=u1.get("kind"),
        outputs2=u2_outs if len(u2_outs) == len(helper_outputs) else None,
    )

    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    # Dedent common_lines first so relative block indentation is normalized
    common_lines = textwrap.dedent("\n".join(common_lines)).splitlines()

    # Prepend scope modifiers and conditional variable initializations to preserve runtime safety
    prefix_stmts: List[str] = []
    if scope.get("globals"):
        g_vars = sorted(set(scope["globals"]))
        if not any(ln.strip().startswith("global ") for ln in common_lines):
            prefix_stmts.append(f"global {', '.join(g_vars)}")
    for out_var in helper_outputs:
        if out_var in conditional_outs and out_var not in inputs:
            prefix_stmts.append(f"{out_var} = None")

    if prefix_stmts:
        common_lines = prefix_stmts + common_lines

    has_trailing_return = any(
        ln.strip() == "return"
        or (
            ln.strip().startswith("return")
            and len(ln.strip()) > 6
            and ln.strip()[6] in (" ", "\t", "(", "#")
        )
        for ln in common_lines[-3:]
    )
    if u1.get("kind") in ("comprehension", "complex_expr"):
        if len(common_lines) == 1:
            common_lines = [f"return {common_lines[0].strip()}"]
        else:
            eff_step = step or _detect_indent_step(indent)
            expr_inner = [f"{eff_step}{ln}" for ln in common_lines]
            common_lines = ["return ("] + expr_inner + [")"]
        helper_outputs = []
    elif not has_trailing_return and helper_outputs and not scope.get("has_yield"):
        if len(helper_outputs) >= 2:
            common_lines = common_lines + [f"return {', '.join(helper_outputs)}"]
        else:
            common_lines = common_lines + [f"return {helper_outputs[0]}"]

    if step is None:
        step = _detect_indent_step(indent)

    body_indent = indent + step
    doc_indent = indent + step
    sub_indent = indent + step + step
    if effective_binding == "method":
        if is_class_receiver:
            call_target = f"cls.{helper_name}"
        elif is_static_clone:
            call_target = f"__class__.{helper_name}"
        else:
            call_target = f"self.{helper_name}"
    else:
        call_target = helper_name

    docstring_str = _format_helper_docstring(
        call_target,
        helper_outputs,
        scope,
        is_async=is_async,
        doc_indent=doc_indent,
        sub_indent=sub_indent,
        await_prefix=await_prefix,
    )

    indented_body = "\n".join(f"{body_indent}{ln}" if ln.strip() else "" for ln in common_lines)
    if effective_binding == "method":
        if is_class_receiver:
            dec_prefix = f"{indent}@classmethod\n"
        elif is_static_clone:
            dec_prefix = f"{indent}@staticmethod\n"
        else:
            dec_prefix = ""
    else:
        dec_prefix = ""
    helper_def = (
        f"{dec_prefix}"
        f"{indent}{func_keyword} {helper_name}({params_str}) -> {return_type}:\n"
        f"{docstring_str}\n"
        f"{indented_body}\n"
    )

    if include_imports:
        needed_typing = _extract_required_typing_imports(helper_def)
        import_header_parts: List[str] = []
        if needed_typing:
            import_header_parts.append(f"from typing import {', '.join(sorted(needed_typing))}")
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in import_header_parts:
                import_header_parts.append(loc_imp)
        if import_header_parts:
            return "\n".join(import_header_parts) + "\n\n\n" + helper_def

    return helper_def


def replace_unit_in_source(
    source_text: str,
    unit: Dict[str, Any],
    replacement_text: str,
    preserve_boundary_pragmas: bool = True,
) -> str:
    """Replaces an AST code unit in source text with replacement text, preserving comments and formatting.

    Supports sub-line and token-range precision slicing when 'start_col' and 'end_col'
    are present in the unit dictionary. Preserves surrounding indentation, preceding code
    on the start line, trailing pragmas/comments on the end line, and attached pragmas.

    Args:
        source_text: The complete original Python source code.
        unit: The AST unit mapping containing 1-indexed 'start' and 'end' line numbers,
            and optional 0-indexed 'start_col' and 'end_col' column offsets.
        replacement_text: The text to substitute in place of the unit lines.
        preserve_boundary_pragmas: If True, retains boundary pragmas (# type: ignore / # noqa)
            attached to the original code unit and appends them to the replacement statement.

    Returns:
        The modified source code with comments, type ignores, and whitespace outside the unit preserved.
    """
    lines = source_text.splitlines(keepends=True)
    start = max(1, int(unit.get("start") or 1))
    end = min(len(lines), int(unit.get("end") or len(lines)))
    if start > len(lines) or start > end:
        return source_text

    start_col = unit.get("start_col")
    end_col = unit.get("end_col")

    rep = replacement_text

    # Extract boundary pragmas if requested
    attached_pragmas: List[str] = []
    if preserve_boundary_pragmas:
        boundary_comments = extract_unit_comments_and_pragmas(
            source_text, start_line=start, end_line=end, start_col=start_col or 0, end_col=end_col
        )
        for c in boundary_comments:
            if c["is_pragma"] and c["line"] in (start, end):
                p_text = c["text"].strip()
                if p_text not in rep and p_text not in attached_pragmas:
                    attached_pragmas.append(p_text)

    first_line = lines[start - 1]
    start_c = max(0, min(len(first_line), int(start_col or 0)))
    last_line = lines[end - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), int(end_col)))
    if start == end:
        end_c = max(start_c, end_c)

    prefix_is_whitespace = not first_line[:start_c].strip()
    suffix_stripped = last_line[end_c:].strip()
    suffix_is_boundary_only = not suffix_stripped or suffix_stripped.startswith("#")
    is_expr_kind = unit.get("kind") in ("comprehension", "complex_expr")
    is_column_bounded = (start_col is not None or end_col is not None) and (
        is_expr_kind or not (prefix_is_whitespace and suffix_is_boundary_only)
    )

    if is_column_bounded:
        prefix_line = first_line[:start_c]
        suffix_line = last_line[end_c:]

        prefix_all = "".join(lines[: start - 1]) + prefix_line
        suffix_all = suffix_line + "".join(lines[end:])

        final_rep = rep
        if attached_pragmas and not any(p in suffix_line for p in attached_pragmas):
            pragma_suffix = "  " + "  ".join(attached_pragmas)
            if not suffix_stripped or suffix_stripped.startswith("#"):
                if final_rep.endswith("\n"):
                    final_rep = final_rep[:-1] + pragma_suffix + "\n"
                else:
                    final_rep += pragma_suffix
            else:
                s_lines = suffix_all.splitlines(keepends=True)
                if s_lines:
                    first_s = s_lines[0]
                    if first_s.endswith("\n"):
                        s_lines[0] = first_s[:-1].rstrip() + pragma_suffix + "\n"
                    else:
                        s_lines[0] = first_s.rstrip() + pragma_suffix
                    suffix_all = "".join(s_lines)

        return prefix_all + final_rep + suffix_all

    # Whole-line replacement
    if attached_pragmas and rep:
        pragma_suffix = "  " + "  ".join(attached_pragmas)
        rep_lines = rep.splitlines(keepends=True)
        if rep_lines:
            last_rep = rep_lines[-1]
            if last_rep.endswith("\n"):
                rep_lines[-1] = last_rep[:-1] + pragma_suffix + "\n"
            else:
                rep_lines[-1] = last_rep + pragma_suffix
            rep = "".join(rep_lines)

    if rep and not rep.endswith("\n"):
        rep += "\n"

    prefix_lines = lines[: start - 1]
    suffix_lines = lines[end:]
    return "".join(prefix_lines) + rep + "".join(suffix_lines)


def check_units_overlap(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> bool:
    """Determines whether two AST code units in the same file share overlapping line ranges.

    Args:
        u1: First AST unit dictionary with 'file', 'start', and 'end'.
        u2: Second AST unit dictionary with 'file', 'start', and 'end'.
        repo_root: Optional repository root path for resolving relative file paths.

    Returns:
        True if both units reside in the same normalized file path and their [start, end]
        intervals overlap; False otherwise.
    """
    f1 = normalize_path_string(str(u1.get("file") or ""))
    f2 = normalize_path_string(str(u2.get("file") or ""))
    if not f1 or not f2 or not _is_same_file_path(f1, f2, repo_root=repo_root):
        return False
    start1 = int(u1.get("start") or 1)
    end1 = int(u1.get("end") or start1)
    start2 = int(u2.get("start") or 1)
    end2 = int(u2.get("end") or start2)

    if end1 < start2 or end2 < start1:
        return False

    s_col1 = u1.get("start_col")
    e_col1 = u1.get("end_col")
    s_col2 = u2.get("start_col")
    e_col2 = u2.get("end_col")

    if start1 == end1 == start2 == end2:
        if s_col1 is not None and e_col1 is not None and s_col2 is not None and e_col2 is not None:
            sc1, ec1 = int(s_col1), int(e_col1)
            sc2, ec2 = int(s_col2), int(e_col2)
            if sc1 <= ec1 and sc2 <= ec2:
                return max(sc1, sc2) < min(ec1, ec2)
            return False

    if start1 < start2 and end1 == start2:
        if e_col1 is not None and s_col2 is not None:
            return int(s_col2) < int(e_col1)

    if start2 < start1 and end2 == start1:
        if e_col2 is not None and s_col1 is not None:
            return int(s_col1) < int(e_col2)

    return max(start1, start2) <= min(end1, end2)


def filter_overlapping_clone_units(
    units: Sequence[Dict[str, Any]],
    repo_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filters a sequence of candidate code units to retain a maximal non-overlapping subset.

    When candidates share overlapping lines/tokens within the same module, candidates
    spanning a larger line range are prioritized, with ties broken by earlier start line.

    Args:
        units: Sequence of AST unit dictionaries proposed for refactoring.
        repo_root: Optional repository root path for resolving relative file paths.

    Returns:
        List of non-overlapping units safe for concurrent refactoring within the same pass.
    """
    if not units:
        return []

    file_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for u in units:
        f_raw = str(u.get("file") or "")
        matched = False
        for rep_f, group in file_groups:
            if (not f_raw and not rep_f) or (
                f_raw and rep_f and _is_same_file_path(f_raw, rep_f, repo_root=repo_root)
            ):
                group.append(u)
                matched = True
                break
        if not matched:
            file_groups.append((f_raw, [u]))

    retained: List[Dict[str, Any]] = []
    for _, file_units in file_groups:
        sorted_candidates = sorted(
            file_units,
            key=lambda u: (
                -(int(u.get("end") or int(u.get("start") or 0)) - int(u.get("start") or 0)),
                int(u.get("start") or 0),
                int(u.get("start_col") or 0),
                str(u.get("name") or ""),
            ),
        )
        file_retained: List[Dict[str, Any]] = []
        for cand in sorted_candidates:
            if not any(check_units_overlap(cand, prev, repo_root=repo_root) for prev in file_retained):
                file_retained.append(cand)
        retained.extend(file_retained)

    return retained


def refactor_module_units(
    source_text: str,
    replacements: Sequence[Tuple[Dict[str, Any], str]],
) -> str:
    """Applies multiple non-overlapping unit replacements in reverse source order.

    Applying substitutions bottom-to-top (descending by unit 'start' line) guarantees
    that line count changes downstream never invalidate the 1-indexed source line
    coordinates of earlier units in the same file.

    Args:
        source_text: The complete original Python source code.
        replacements: Sequence of (unit, replacement_text) tuples.

    Returns:
        The refactored module source text.

    Raises:
        ValueError: If any pair of units in replacements shares overlapping line ranges.
    """
    if not replacements:
        return source_text

    rep_list = list(replacements)
    for i, (u1, _) in enumerate(rep_list):
        for u2, _ in rep_list[i + 1:]:
            if check_units_overlap(u1, u2):
                n1 = str(u1.get("name") or "unit")
                s1 = int(u1.get("start") or 1)
                e1 = int(u1.get("end") or s1)
                n2 = str(u2.get("name") or "unit")
                s2 = int(u2.get("start") or 1)
                e2 = int(u2.get("end") or s2)
                f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
                raise ValueError(
                    f"Overlapping unit collision detected between "
                    f"'{n1}' ({s1}-{e1}) and "
                    f"'{n2}' ({s2}-{e2}) in {f1}."
                )

    sorted_replacements = sorted(
        rep_list,
        key=lambda item: (
            int(item[0].get("start") or 0),
            int(item[0].get("start_col") or 0),
        ),
        reverse=True,
    )

    current_text = source_text
    for unit, replacement in sorted_replacements:
        current_text = replace_unit_in_source(current_text, unit, replacement)

    return current_text


def _scan_sig_line(line: str, initial_paren_depth: int = 0) -> Tuple[int, int]:
    """Scans a line for the function header terminating colon and computes resulting paren depth."""
    paren_depth = initial_paren_depth
    in_quote: Optional[str] = None
    for i, ch in enumerate(line):
        if in_quote:
            if ch == in_quote:
                num_bs = 0
                k = i - 1
                while k >= 0 and line[k] == "\\":
                    num_bs += 1
                    k -= 1
                if num_bs % 2 == 0:
                    in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch == "#":
            break
        if ch in "([{":
            paren_depth += 1
        elif ch in ")]}":
            paren_depth = max(0, paren_depth - 1)
        elif ch == ":" and paren_depth == 0:
            return i, paren_depth
    return -1, paren_depth


def _find_sig_colon(line: str) -> int:
    """Finds the colon terminating a function definition header on a single line."""
    colon_idx, _ = _scan_sig_line(line, 0)
    if colon_idx != -1:
        return colon_idx
    return line.rfind(":")



def _build_whole_method_delegation(
    source_text: str,
    unit: Dict[str, Any],
    call_prefix: str,
    helper_name: str,
    args_str: str,
    await_prefix: str = "",
    has_return: bool = True,
    is_async: bool = False,
    has_yield: bool = False,
) -> str:
    """Builds a delegated method replacement body preserving method signature and docstring."""
    lines = source_text.splitlines(keepends=True)
    u_start = int(unit.get("start") or 1)
    u_end = int(unit.get("end") or max(u_start, len(lines)))

    lead = lines[u_start - 1] if 1 <= u_start <= len(lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]
    default_step = _detect_indent_step(indent)
    body_indent = indent + default_step

    sig_end_line: Optional[int] = None
    docstring_end_line: Optional[int] = None
    header = ""
    raw_u_name = str(unit.get("name") or "")
    base_u_name = raw_u_name.rsplit(":", maxsplit=1)[-1] if raw_u_name else ""

    try:
        tree = ast.parse(source_text)
        cand_nodes: List[Tuple[int, Union[ast.FunctionDef, ast.AsyncFunctionDef]]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_start = getattr(node, "lineno", 0)
                n_end = getattr(node, "end_lineno", n_start)
                decs = getattr(node, "decorator_list", [])
                dec_start = min(getattr(d, "lineno", n_start) for d in decs) if decs else n_start
                earliest_start = min(dec_start, n_start)
                if u_start in (n_start, earliest_start):
                    cand_nodes.append((0, node))
                elif node.name in (raw_u_name, base_u_name) and earliest_start <= u_start <= n_end:
                    cand_nodes.append((n_end - earliest_start + 1, node))
        if cand_nodes:
            cand_nodes.sort(key=lambda item: (item[0], -getattr(item[1], "lineno", 0)))
            matched_node = cand_nodes[0][1]
            if matched_node.body:
                b_cand_lines = [
                    int(getattr(b_stmt, "lineno", 0))
                    for b_stmt in matched_node.body
                    if getattr(b_stmt, "lineno", 0) > 0
                ]
                found_indent = _extract_child_indentation(lines, b_cand_lines, indent)
                if found_indent is not None:
                    body_indent = found_indent
                first_body = matched_node.body[0]
                fn_def_line = getattr(matched_node, "lineno", u_start)
                if first_body.lineno == fn_def_line:
                    b_col = getattr(first_body, "col_offset", len(lines[fn_def_line - 1]))
                    same_line = lines[fn_def_line - 1][:b_col].rstrip()
                    if not same_line.endswith(":"):
                        colon_pos = _find_sig_colon(lines[fn_def_line - 1][:b_col])
                        if colon_pos != -1:
                            same_line = lines[fn_def_line - 1][: colon_pos + 1]
                    header = "".join(lines[u_start - 1 : fn_def_line - 1]) + same_line + "\n"
                else:
                    sig_end_line = first_body.lineno - 1
                    if (
                        isinstance(first_body, ast.Expr)
                        and isinstance(first_body.value, ast.Constant)
                        and isinstance(first_body.value.value, str)
                    ):
                        docstring_end_line = getattr(first_body, "end_lineno", first_body.lineno)
    except SyntaxError:
        pass

    if not header:
        if docstring_end_line is not None and docstring_end_line >= u_start:
            header = "".join(lines[u_start - 1 : docstring_end_line])
        elif sig_end_line is not None and sig_end_line >= u_start:
            header = "".join(lines[u_start - 1 : sig_end_line])
        else:
            hdr_lines = []
            curr_depth = 0
            for l_num in range(u_start, min(u_end + 1, len(lines) + 1)):
                ln = lines[l_num - 1]
                colon_pos, curr_depth = _scan_sig_line(ln, initial_paren_depth=curr_depth)
                if colon_pos != -1:
                    hdr_lines.append(ln[: colon_pos + 1] + "\n")
                    break
                hdr_lines.append(ln)
                if ln.rstrip().endswith(":") and curr_depth == 0:
                    break
            header = "".join(hdr_lines)

    if not header.endswith("\n"):
        header += "\n"

    body_step = (
        body_indent[len(indent):]
        if body_indent.startswith(indent) and len(body_indent) > len(indent)
        else ("\t" if "\t" in body_indent else "    ")
    )

    effective_async = is_async or bool(await_prefix.strip())
    if effective_async and has_yield:
        delegation_stmt = (
            f"{body_indent}async for _item in {call_prefix}{helper_name}({args_str}):\n"
            f"{body_indent}{body_step}yield _item\n"
        )
    elif has_yield:
        if has_return:
            delegation_stmt = (
                f"{body_indent}return (yield from {call_prefix}{helper_name}({args_str}))\n"
            )
        else:
            delegation_stmt = f"{body_indent}yield from {call_prefix}{helper_name}({args_str})\n"
    elif effective_async:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}await {call_prefix}{helper_name}({args_str})\n"
    else:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}{call_prefix}{helper_name}({args_str})\n"
    return header + delegation_stmt


def _has_unconditional_terminal_return(
    unit: Dict[str, Any], orig_lines: Sequence[str]
) -> bool:
    """Checks whether an AST code unit terminates with an unconditional return at its base indentation."""
    u_start = max(1, int(unit.get("start") or 1))
    u_end = min(len(orig_lines), int(unit.get("end") or u_start))
    if u_start > len(orig_lines) or u_start > u_end:
        return False
    cand_lines = orig_lines[u_start - 1 : u_end]
    non_empty = [ln for ln in cand_lines if ln.strip() and not ln.strip().startswith("#")]
    if not non_empty:
        return False
    base_ind = len(non_empty[0]) - len(non_empty[0].lstrip())
    last_ln = non_empty[-1]
    last_ind = len(last_ln) - len(last_ln.lstrip())
    if last_ind != base_ind:
        return False
    s = last_ln.strip()
    return s == "return" or s.startswith("return ") or s.startswith("return(") or s.startswith("return\t")


def _build_unit_delegation_call(
    target_unit: Dict[str, Any],
    orig_text: str,
    orig_lines: List[str],
    *,
    effective_binding: str,
    helper_name: str,
    inputs: List[str],
    outputs: List[str],
    scope: Dict[str, Any],
    target_inputs: Optional[List[str]] = None,
    target_outputs: Optional[List[str]] = None,
    await_prefix: str = "",
    step: Optional[str] = None,
) -> str:
    """Constructs replacement delegation call statement for a clone unit in refactoring patches."""
    u_start = int(target_unit.get("start") or 1)
    lead = orig_lines[u_start - 1] if 1 <= u_start <= len(orig_lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]

    t_fn = find_enclosing_function(orig_text, target_unit)
    t_enc = find_enclosing_class(orig_text, target_unit)
    if not _is_method_of_class(t_fn, t_enc):
        t_fn = None
    t_kind = _get_enclosing_receiver_kind(t_fn) if t_fn else "none"

    if effective_binding == "method":
        if t_kind == "static":
            unit_call_prefix = "__class__."
            unit_receiver_omit: Optional[str] = None
        elif t_kind == "class":
            unit_call_prefix = "cls."
            unit_receiver_omit = "cls"
        else:
            unit_call_prefix = "self."
            unit_receiver_omit = "self"
    else:
        unit_call_prefix = ""
        unit_receiver_omit = (
            "receivers"
            if t_kind in ("static", "none")
            else ("self" if t_kind == "class" else "cls")
        )

    effective_targets = target_inputs if target_inputs is not None else inputs
    effective_outputs = target_outputs if target_outputs is not None else outputs

    unit_args_str = _format_call_arguments(
        inputs,
        scope.get("param_details", []),
        receiver_to_omit=unit_receiver_omit,
        target_inputs=effective_targets,
    )

    is_whole_method = target_unit.get("kind") in ("function", "closure", "method")
    if is_whole_method:
        is_init = bool(
            target_unit.get("name") == "__init__"
            or str(target_unit.get("name", "")).endswith((".__init__", ":__init__"))
        )
        has_return = False if is_init else bool(effective_outputs or scope.get("has_return", True))
        return _build_whole_method_delegation(
            orig_text,
            target_unit,
            call_prefix=unit_call_prefix,
            helper_name=helper_name,
            args_str=unit_args_str,
            await_prefix=await_prefix,
            has_return=has_return,
            is_async=bool(scope.get("is_async")),
            has_yield=bool(scope.get("has_yield")),
        )

    if target_unit.get("kind") in ("comprehension", "complex_expr"):
        call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
        return f"{await_prefix}{call_expr}"

    call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
    rep_step = step or ("\t" if "\t" in indent else "    ")
    if scope.get("has_yield") and scope.get("is_async"):
        return (
            f"{indent}async for _item in {call_expr}:\n"
            f"{indent}{rep_step}yield _item\n"
        )

    is_sync_gen = bool(scope.get("has_yield"))
    if _has_unconditional_terminal_return(target_unit, orig_lines):
        prefix = "yield from " if is_sync_gen else await_prefix
        return f"{indent}return {prefix}{call_expr}\n"

    if effective_outputs:
        assign_target = ", ".join(effective_outputs) if len(effective_outputs) >= 2 else effective_outputs[0]
        rhs = (
            f"(yield from {call_expr})"
            if is_sync_gen
            else f"{await_prefix}{call_expr}"
        )
        return f"{indent}{assign_target} = {rhs}\n"

    prefix = "yield from " if is_sync_gen else await_prefix
    return f"{indent}{prefix}{call_expr}\n"


def _derive_module_import_path(file_path: Union[Path, str], repo_root: Union[Path, str]) -> str:
    """Derives the importable Python module dot-path for a file relative to repository root.

    Handles flat layouts (foo.py -> foo), package layouts (pkg/mod.py -> pkg.mod),
    PEP 517/518 src layouts (src/pkg/mod.py -> pkg.mod), and __init__.py files (pkg/__init__.py -> pkg).
    """
    p_file = Path(file_path)
    p_root = Path(repo_root)
    try:
        rel = p_file.resolve().relative_to(p_root.resolve())
    except ValueError:
        rel = p_file
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return ""
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_imports_target(source_text: str, target_module: str) -> bool:
    """Checks whether a module's source imports a specific target module."""
    if not target_module:
        return False
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return target_module in source_text
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == target_module or alias.name.startswith(f"{target_module}."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = getattr(node, "module", None) or ""
            if mod == target_module or mod.startswith(f"{target_module}."):
                return True
            for alias in node.names:
                full = f"{mod}.{alias.name}" if mod else alias.name
                if full == target_module or full.startswith(f"{target_module}."):
                    return True
    return False


class _FilePatchPlan:
    """Internal accumulator of proposed transformations for a single file."""

    def __init__(self, file_path: Path, orig_text: str, rel_path: str) -> None:
        self.path = file_path
        self.orig_text = orig_text
        self.orig_lines = orig_text.splitlines(keepends=True)
        self.rel_path = rel_path
        self.replacements: List[Tuple[Dict[str, Any], str]] = []
        self.method_helpers: List[Tuple[int, str]] = []
        self.module_helpers: List[str] = []
        self.missing_imports: List[str] = []
        self.comments: List[str] = []
        self.used_helper_names: Set[str] = set()
        self.claimed_units: List[Dict[str, Any]] = []


def _adjust_line_for_replacements(
    target_line: int,
    reps: Sequence[Tuple[Dict[str, Any], str]],
    orig_text: str,
) -> int:
    """Adjusts a target source line number to account for upstream line expansions or contractions."""
    orig_lines = orig_text.splitlines(keepends=True)
    offset = 0
    for u, rep in reps:
        u_start = int(u.get("start") or 1)
        u_end = int(u.get("end") or u_start)
        if u_end < target_line:
            orig_count = u_end - u_start + 1
            unit_orig_text = "".join(orig_lines[u_start - 1 : u_end])
            replaced_text = replace_unit_in_source(
                unit_orig_text, {**u, "start": 1, "end": orig_count}, rep
            )
            new_count = len(replaced_text.splitlines(keepends=True))
            offset += (new_count - orig_count)
    return max(1, target_line + offset)


def _render_file_patch_plan(
    plan: _FilePatchPlan,
    replace_clones: bool,
    repo_root: Optional[str] = None,
) -> str:
    """Renders a single cumulative unified diff for all modifications in a file plan."""
    retained_cands = filter_overlapping_clone_units(
        [u for u, _ in plan.replacements], repo_root=repo_root
    )
    filtered_reps: List[Tuple[Dict[str, Any], str]] = []
    for u, rep in plan.replacements:
        if any(
            u is r
            or (
                u.get("file") == r.get("file")
                and u.get("start") == r.get("start")
                and u.get("end") == r.get("end")
            )
            for r in retained_cands
        ):
            if not any(
                check_units_overlap(u, prev_u, repo_root=repo_root)
                for prev_u, _ in filtered_reps
            ):
                filtered_reps.append((u, rep))

    if replace_clones and filtered_reps:
        current_text = refactor_module_units(plan.orig_text, filtered_reps)
    else:
        current_text = plan.orig_text

    if plan.method_helpers:
        adjusted_methods = [
            (
                _adjust_line_for_replacements(ins_line, filtered_reps, plan.orig_text)
                if replace_clones and filtered_reps
                else ins_line,
                h_code,
            )
            for ins_line, h_code in plan.method_helpers
        ]
        sorted_methods = sorted(adjusted_methods, key=lambda m: m[0], reverse=True)
        for ins_line, h_code in sorted_methods:
            c_lines = current_text.splitlines(keepends=True)
            idx = max(0, ins_line - 1)
            h_lines = [ln + "\n" for ln in h_code.splitlines()] + ["\n"]
            current_text = "".join(c_lines[:idx] + h_lines + c_lines[idx:])

    c_lines = current_text.splitlines(keepends=True)
    deduped_imports = list(dict.fromkeys(plan.missing_imports))
    lines_with_imports = _insert_imports_into_module(c_lines, deduped_imports)

    if plan.module_helpers:
        ins_idx = _find_module_helper_insertion_index(lines_with_imports)
        all_h_lines: List[str] = []
        for h_code in plan.module_helpers:
            all_h_lines.extend(["\n"] + [ln + "\n" for ln in h_code.splitlines()] + ["\n"])
        modified_lines = lines_with_imports[:ins_idx] + all_h_lines + lines_with_imports[ins_idx:]
    else:
        modified_lines = lines_with_imports

    diff = difflib.unified_diff(
        plan.orig_lines,
        modified_lines,
        fromfile=f"a/{plan.rel_path}",
        tofile=f"b/{plan.rel_path}",
        n=3,
    )
    diff_str = "".join(diff)
    if not diff_str:
        return ""

    deduped_comments = list(dict.fromkeys(plan.comments))
    return "".join(deduped_comments) + diff_str


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
    method_binding: str = "auto",
) -> str:
    """Generates a git-apply compatible unified diff patch proposing shared helper extractions."""
    if not clones:
        return ""

    root = Path(repo_root or os.getcwd())
    file_plans: Dict[Path, _FilePatchPlan] = {}

    def _get_plan(file_p: Path, rel_f: str, text: str) -> _FilePatchPlan:
        if file_p not in file_plans:
            file_plans[file_p] = _FilePatchPlan(file_p, text, rel_f)
        return file_plans[file_p]

    for sim, u1, u2 in clones:
        f1_raw = normalize_path_string(str(u1.get("file") or ""), strip_anchor=True)
        if not f1_raw:
            continue
        p = Path(f1_raw)
        f1_path = p if p.is_file() or p.is_absolute() else (root / p)
        if not f1_path.is_file():
            continue

        try:
            rel_f1 = str(f1_path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            rel_f1 = str(f1_path.name)

        f1_plan = file_plans.get(f1_path)
        if f1_plan is None:
            try:
                orig_text = f1_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            f1_plan = _get_plan(f1_path, rel_f1, orig_text)
        else:
            orig_text = f1_plan.orig_text

        orig_lines = f1_plan.orig_lines

        f2_raw = normalize_path_string(str(u2.get("file") or ""), strip_anchor=True)
        is_same_file = _is_same_file_path(f1_raw, f2_raw, repo_root=str(root))

        enc1 = find_enclosing_class(orig_text, u1)
        fn1 = find_enclosing_function(orig_text, u1)

        enc2 = None
        fn2 = None
        f2_plan: Optional[_FilePatchPlan] = None
        if is_same_file:
            enc2 = find_enclosing_class(orig_text, u2)
            fn2 = find_enclosing_function(orig_text, u2)
        elif f2_raw:
            p2 = Path(f2_raw)
            f2_path = p2 if p2.is_file() or p2.is_absolute() else (root / p2)
            if f2_path.is_file():
                try:
                    rel_f2 = str(f2_path.resolve().relative_to(root.resolve())).replace("\\", "/")
                except ValueError:
                    rel_f2 = str(f2_path.name)
                f2_plan = file_plans.get(f2_path)
                if f2_plan is None:
                    try:
                        f2_text = f2_path.read_text(encoding="utf-8")
                        f2_plan = _get_plan(f2_path, rel_f2, f2_text)
                    except (OSError, UnicodeDecodeError):
                        f2_plan = None
                if f2_plan is not None:
                    enc2 = find_enclosing_class(f2_plan.orig_text, u2)
                    fn2 = find_enclosing_function(f2_plan.orig_text, u2)

        if replace_clones:
            u1_claimed = any(
                check_units_overlap(u1, prev_u, repo_root=str(root))
                for prev_u in f1_plan.claimed_units
            )
            if is_same_file:
                u2_claimed = any(
                    check_units_overlap(u2, prev_u, repo_root=str(root))
                    for prev_u in f1_plan.claimed_units
                )
            else:
                u2_claimed = (
                    any(
                        check_units_overlap(u2, prev_u, repo_root=str(root))
                        for prev_u in f2_plan.claimed_units
                    )
                    if f2_plan is not None
                    else False
                )
            if u1_claimed or u2_claimed:
                continue

        is_same_class = bool(
            is_same_file
            and enc1
            and enc2
            and enc1["name"] == enc2["name"]
            and enc1["start"] == enc2["start"]
        )

        if not _is_method_of_class(fn1, enc1):
            fn1 = None
        if not _is_method_of_class(fn2, enc2):
            fn2 = None

        non_method_kinds = ("closure", "class", "comprehension", "data_table", "complex_expr")
        fn1_kind = _get_enclosing_receiver_kind(fn1) if fn1 else (u1.get("receiver_kind") or ("instance" if enc1 and u1.get("kind") not in non_method_kinds else "none"))
        fn2_kind = _get_enclosing_receiver_kind(fn2) if fn2 else (u2.get("receiver_kind") or ("instance" if enc2 and u2.get("kind") not in non_method_kinds else "none"))
        receiver_kinds_differ = bool(fn1_kind != fn2_kind)
        is_in_method = bool(
            fn1
            and fn2
            and u1.get("kind") not in ("comprehension", "complex_expr")
            and u2.get("kind") not in ("comprehension", "complex_expr")
        )
        if fn1 and fn2:
            is_static = bool(fn1.get("is_static") and fn2.get("is_static"))
        else:
            is_static = bool((fn1 and fn1.get("is_static")) or (fn2 and fn2.get("is_static")))

        s1 = analyze_unit_variable_scope(u1, repo_root=str(root))
        s2 = analyze_unit_variable_scope(u2, repo_root=str(root))
        if bool(s1.get("is_async")) != bool(s2.get("is_async")):
            continue
        if bool(s1.get("has_yield")) != bool(s2.get("has_yield")):
            continue
        if s1.get("nonlocals") or s2.get("nonlocals"):
            continue
        if not is_same_file and (s1.get("globals") or s2.get("globals")):
            continue
        if (
            set(s1.get("attrs_read", [])) != set(s2.get("attrs_read", []))
            or set(s1.get("attrs_written", [])) != set(s2.get("attrs_written", []))
        ):
            continue

        hazards1 = set(s1.get("control_flow_hazards", []))
        hazards2 = set(s2.get("control_flow_hazards", []))
        if any(h in ("naked_break", "naked_continue") for h in hazards1 | hazards2):
            continue
        if "embedded_return" in hazards1 | hazards2:
            lines1 = orig_lines
            lines2 = orig_lines if is_same_file else (f2_plan.orig_lines if f2_plan else [])
            if not (
                _has_unconditional_terminal_return(u1, lines1)
                and _has_unconditional_terminal_return(u2, lines2)
            ):
                continue

        if receiver_kinds_differ:
            if _has_receiver_reference(u1, s1, repo_root=str(root)) or _has_receiver_reference(u2, s2, repo_root=str(root)):
                continue

        effective_binding = _resolve_effective_binding(
            method_binding,
            is_same_class,
            is_static=is_static,
            receiver_kinds_differ=receiver_kinds_differ,
            is_in_method=is_in_method,
            receiver_kind=fn1_kind,
        )

        has_super = bool(s1.get("has_super") or s2.get("has_super"))
        if has_super and (effective_binding == "module" or is_static):
            continue

        has_mangled = bool(s1.get("has_mangled_names") or s2.get("has_mangled_names"))
        if has_mangled and (effective_binding != "method" or not is_same_class or is_static):
            continue

        helper_indent = (
            enc1["method_indent"]
            if (effective_binding == "method" and enc1)
            else ("    " if effective_binding == "method" else "")
        )
        step = (
            enc1["method_indent"][len(enc1["indent"]):]
            if (
                enc1
                and enc1.get("indent") is not None
                and enc1["method_indent"].startswith(enc1["indent"])
                and len(enc1["method_indent"]) > len(enc1["indent"])
            )
            else None
        )
        if step is None:
            u1_s = int(u1.get("start") or 1)
            u1_lead = orig_lines[u1_s - 1] if 1 <= u1_s <= len(orig_lines) else ""
            u1_ind = u1_lead[: len(u1_lead) - len(u1_lead.lstrip())]
            step = _detect_indent_step(u1_ind)

        scope = analyze_unit_variable_scope(u1, u2, repo_root=str(root))
        inputs = list(scope.get("inputs", []))
        if effective_binding == "module":
            inputs = _prune_unshared_receivers(inputs, u1, u2, s1, s2, repo_root=str(root))
        outputs = [
            v for v in scope.get("outputs", [])
            if v not in scope.get("globals", [])
            and v not in scope.get("nonlocals", [])
        ]
        u1_outs = [
            v for v in s1.get("outputs", [])
            if v not in s1.get("globals", [])
            and v not in s1.get("nonlocals", [])
        ]
        u2_outs = [
            v for v in s2.get("outputs", [])
            if v not in s2.get("globals", [])
            and v not in s2.get("nonlocals", [])
        ]
        t_inputs1 = list(s1.get("inputs", []))
        t_inputs2 = list(s2.get("inputs", []))
        if effective_binding == "module":
            t_inputs1 = _prune_unshared_receivers(
                t_inputs1, u1, u2, s1, s2, repo_root=str(root)
            )
            t_inputs2 = _prune_unshared_receivers(
                t_inputs2, u2, u1, s2, s1, repo_root=str(root)
            )

        if replace_clones and (
            len(t_inputs1) != len(inputs)
            or len(t_inputs2) != len(inputs)
            or len(u1_outs) != len(outputs)
            or len(u2_outs) != len(outputs)
        ):
            continue

        base_name1 = _base_unit_name(u1)
        base_name2 = _base_unit_name(u2)
        base_helper = (
            f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
        )
        helper_name = base_helper
        h_idx = 2
        while (
            helper_name in f1_plan.used_helper_names
            or bool(re.search(rf"\b{re.escape(helper_name)}\b", orig_text))
            or (
                f2_plan is not None
                and (
                    helper_name in f2_plan.used_helper_names
                    or bool(re.search(rf"\b{re.escape(helper_name)}\b", f2_plan.orig_text))
                )
            )
        ):
            helper_name = f"{base_helper}_{h_idx}"
            h_idx += 1
        f1_plan.used_helper_names.add(helper_name)
        if f2_plan is not None:
            f2_plan.used_helper_names.add(helper_name)

        helper_code = synthesize_shared_helper_code(
            u1,
            u2,
            type_merge_strategy=type_merge_strategy,
            method_binding=effective_binding,
            indent=helper_indent,
            is_static=is_static,
            receiver_kind=fn1_kind if effective_binding == "method" else None,
            step=step,
            repo_root=str(root),
            helper_name=helper_name,
        )
        if not helper_code:
            continue

        needed_typing = _extract_required_typing_imports(helper_code)
        existing_imports = _get_module_imported_names(orig_text)
        missing_typing = [s for s in needed_typing if s not in existing_imports]

        missing_import_lines: List[str] = []
        if missing_typing:
            missing_import_lines.append(f"from typing import {', '.join(sorted(missing_typing))}")
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in orig_text and loc_imp not in missing_import_lines:
                missing_import_lines.append(loc_imp)

        await_prefix = "await " if scope.get("is_async") else ""

        f1_disp = normalize_path_string(str(u1.get("file") or "file1"), strip_anchor=False)
        f2_disp = normalize_path_string(str(u2.get("file") or "file2"), strip_anchor=False)
        pair_comment = f"# Clone Pair ({sim:.1%}): {f1_disp} <===> {f2_disp}\n"
        f1_plan.comments.append(pair_comment)

        candidate_units = [u1]
        if is_same_file and not check_units_overlap(u1, u2, repo_root=str(root)):
            candidate_units.append(u2)

        earliest_unit = min(candidate_units, key=lambda u: int(u.get("start") or 1))
        enc_fn_earliest = (
            find_enclosing_function(orig_text, earliest_unit)
            if effective_binding == "method"
            else None
        )
        insert_line = (
            enc_fn_earliest["start"] if enc_fn_earliest else int(earliest_unit.get("start") or 1)
        )

        rep_stmt1 = _build_unit_delegation_call(
            u1,
            orig_text,
            orig_lines,
            effective_binding=effective_binding,
            helper_name=helper_name,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            target_inputs=t_inputs1,
            target_outputs=u1_outs if len(u1_outs) == len(outputs) else outputs,
            await_prefix=await_prefix,
            step=step,
        )
        f1_plan.replacements.append((u1, rep_stmt1))
        f1_plan.claimed_units.append(u1)

        if is_same_file and len(candidate_units) > 1:
            rep_stmt2 = _build_unit_delegation_call(
                u2,
                orig_text,
                orig_lines,
                effective_binding=effective_binding,
                helper_name=helper_name,
                inputs=inputs,
                outputs=outputs,
                scope=scope,
                target_inputs=t_inputs2,
                target_outputs=u2_outs if len(u2_outs) == len(outputs) else outputs,
                await_prefix=await_prefix,
                step=step,
            )
            f1_plan.replacements.append((u2, rep_stmt2))
            f1_plan.claimed_units.append(u2)
        elif not is_same_file and f2_plan is not None:
            mod1 = _derive_module_import_path(f1_path, root)
            mod2 = _derive_module_import_path(f2_plan.path, root)
            is_circular = bool(mod2 and _module_imports_target(orig_text, mod2))
            if is_circular or not mod1:
                f1_plan.comments.append(
                    f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                    f"Circular import or unresolvable module path; import manually into {f2_disp}.\n"
                )
            elif replace_clones:
                f2_plan.comments.append(pair_comment)
                f2_plan.missing_imports.append(f"from {mod1} import {helper_name}")
                u2_s = int(u2.get("start") or 1)
                u2_lead = (
                    f2_plan.orig_lines[u2_s - 1]
                    if 1 <= u2_s <= len(f2_plan.orig_lines)
                    else ""
                )
                u2_ind = u2_lead[: len(u2_lead) - len(u2_lead.lstrip())]
                step2 = _detect_indent_step(u2_ind)

                rep_stmt2 = _build_unit_delegation_call(
                    u2,
                    f2_plan.orig_text,
                    f2_plan.orig_lines,
                    effective_binding="module",
                    helper_name=helper_name,
                    inputs=inputs,
                    outputs=outputs,
                    scope=scope,
                    target_inputs=t_inputs2,
                    target_outputs=u2_outs if len(u2_outs) == len(outputs) else outputs,
                    await_prefix=await_prefix,
                    step=step2,
                )
                f2_plan.replacements.append((u2, rep_stmt2))
                f2_plan.claimed_units.append(u2)
            else:
                f1_plan.comments.append(
                    f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                    f"Complete refactoring by importing the helper into {f2_disp}.\n"
                )
        elif not is_same_file:
            f1_plan.comments.append(
                f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                f"Complete refactoring by importing the helper into {f2_disp}.\n"
            )

        if effective_binding == "method":
            f1_plan.method_helpers.append((insert_line, helper_code))
            f1_plan.missing_imports.extend(missing_import_lines)
        else:
            f1_plan.module_helpers.append(helper_code)
            f1_plan.missing_imports.extend(missing_import_lines)

    patch_chunks: List[str] = []
    for plan in file_plans.values():
        chunk = _render_file_patch_plan(
            plan, replace_clones=replace_clones, repo_root=str(root)
        )
        if chunk:
            patch_chunks.append(chunk)

    return "\n".join(patch_chunks)
