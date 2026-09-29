"""Enclosing class and method boundary inspection, receiver kind resolution, and binding."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import normalize_path_string, paths_match_boundary
from pydoppelgangerhunt.parser import is_decorator_named
from pydoppelgangerhunt.fixer.scope import (
    dispatch_analyze_unit_variable_scope as analyze_unit_variable_scope,
)
from pydoppelgangerhunt.fixer.source import parse_unit_coord, split_source_lines


def _parse_source_tree(
    source_text: str,
    tree: Optional[ast.AST] = None,
) -> Optional[ast.AST]:
    """Parses Python source code into an AST tree if not already provided."""
    if tree is not None:
        return tree
    if not source_text.strip():
        return None
    try:
        return ast.parse(source_text)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None


def _get_valid_unit_bounds(unit: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Extracts positive start and end line coordinates from a unit dict."""
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    if u_start <= 0 or u_end <= 0:
        return None
    return u_start, u_end


def _extract_unit_end_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts end column offset from unit dictionary if present."""
    for col_key in ("end_col_offset", "end_col"):
        col_val = unit.get(col_key)
        if col_val is not None:
            try:
                return int(col_val)
            except (ValueError, TypeError):
                pass
    return None


def _resolve_unit_ast_end_col(
    scope_node: ast.AST, unit: Dict[str, Any]
) -> Optional[int]:
    """Resolves end column offset from statement boundaries within the enclosing scope.

    Limitations & Heuristics:
    This AST-based statement boundary derivation is a heuristic fallback when an explicit
    'end_col' or 'end_col_offset' is absent from the unit dict. It assumes unit boundaries
    align with top-level or leaf statement boundaries. Specifically:
    - Units ending mid-statement (such as sub-expressions or chained method calls) cannot be
      accurately resolved and may extend past the unit's actual token span.
    - Nested or compound statements sharing the same end line may cause ambiguities.
    - When u_start == u_end, the first statement on that line with col_offset >= start_col
      is selected.
    - When u_start < u_end, the last leaf statement ending on u_end is selected.
    Callers should supply an explicit token-level end_col whenever available. If this function
    returns None, downstream callers safely fall back to whole-line granularity (treating only
    nodes with lineno > u_end as downstream).
    """
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None

    cand_stmts: List[ast.stmt] = [
        node for node in ast.walk(scope_node) if isinstance(node, ast.stmt)
    ]
    inner_stmts = [
        s
        for s in cand_stmts
        if getattr(s, "end_lineno", getattr(s, "lineno", 0)) == u_end
        and not any(
            other is not s
            and isinstance(other, ast.stmt)
            and getattr(other, "end_lineno", getattr(other, "lineno", 0)) == u_end
            for other in ast.walk(s)
        )
    ]
    if not inner_stmts:
        return None

    inner_stmts.sort(key=lambda s: getattr(s, "col_offset", 0))
    if u_start == u_end:
        start_col_raw = unit.get("start_col")
        if start_col_raw is None:
            start_col_raw = unit.get("start_col_offset")
        try:
            start_col = int(start_col_raw) if start_col_raw is not None else None
        except (ValueError, TypeError):
            start_col = None

        if start_col is not None:
            matched = [s for s in inner_stmts if getattr(s, "col_offset", 0) >= start_col]
            target_stmt = matched[0] if matched else inner_stmts[0]
        else:
            target_stmt = inner_stmts[0]
    else:
        target_stmt = inner_stmts[-1]

    end_col = getattr(target_stmt, "end_col_offset", None)
    return int(end_col) if end_col is not None else None


def _find_innermost_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
    tree: Optional[ast.AST] = None,
) -> Optional[Tuple[ast.AST, int, int]]:
    """Locates the innermost AST node of matching types enclosing the given unit."""
    bounds = _get_valid_unit_bounds(unit)
    if bounds is None:
        return None
    u_start, u_end = bounds

    parsed_tree = _parse_source_tree(source_text, tree=tree)
    if parsed_tree is None:
        return None

    candidates: List[Tuple[int, ast.AST, int, int]] = []
    for node in ast.walk(parsed_tree):
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

    lines = split_source_lines(source_text)
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
    node = meta.pop("node", None)
    meta["is_async"] = isinstance(node, ast.AsyncFunctionDef)
    meta["is_static"] = any(is_decorator_named(d, "staticmethod") for d in decs)
    meta["is_class_method"] = any(is_decorator_named(d, "classmethod") for d in decs)
    receiver_param = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        all_pos = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
        if all_pos:
            receiver_param = all_pos[0].arg
    meta["receiver_param"] = receiver_param
    return meta


class _BaseScopeVisitor(ast.NodeVisitor):
    """Shared helpers for scope visitors."""

    def __init__(self) -> None:
        self._comp_targets: List[Set[str]] = []
        self._in_comp_body: int = 0
        self.globals: Set[str] = set()
        self.nonlocals: Set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def _is_in_comp(self, name: str) -> bool:
        return any(name in targets for targets in self._comp_targets)

    def _visit_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    ) -> None:
        if not node.generators:
            return
        # First generator's iter evaluates in enclosing scope (class scope if in class body)
        self.visit(node.generators[0].iter)
        comp_set: Set[str] = set()
        self._comp_targets.append(comp_set)
        self._in_comp_body += 1
        try:
            for idx, gen in enumerate(node.generators):
                if idx > 0:
                    self.visit(gen.iter)
                comp_set.update(
                    n.id for n in ast.walk(gen.target) if isinstance(n, ast.Name)
                )
                for if_expr in gen.ifs:
                    self.visit(if_expr)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._in_comp_body -= 1
            self._comp_targets.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _visit_named_expr(
        self,
        node: ast.NamedExpr,
        store_set: Set[str],
        check_comp: bool = False,
    ) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            if not check_comp or not self._is_in_comp(node.target.id):
                store_set.add(node.target.id)
        else:
            self.visit(node.target)

    @staticmethod
    def _record_import_names(
        node: Union[ast.Import, ast.ImportFrom], store_set: Set[str]
    ) -> None:
        for alias in node.names:
            bound = (
                alias.asname
                or (alias.name if isinstance(node, ast.ImportFrom) else alias.name.split(".", 1)[0])
            )
            if bound != "*":
                store_set.add(bound)

    def _visit_nested_named_scope(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        store_set: Set[str],
        reads_set: Set[str],
    ) -> None:
        store_set.add(node.name)
        reads_set.update(_extract_nested_scope_free_reads(node))

    def _visit_nested_lambda(
        self,
        node: ast.Lambda,
        reads_set: Set[str],
    ) -> None:
        reads_set.update(_extract_nested_scope_free_reads(node))

    @staticmethod
    def _extract_pattern_bound_name(node: ast.AST) -> Optional[str]:
        """Extracts identifier bound by MatchAs, MatchStar, or MatchMapping pattern nodes."""
        name = getattr(node, "name", None) or getattr(node, "rest", None)
        if isinstance(name, str) and name and name != "_":
            return name
        return None

    def _record_pattern_binding(self, node: ast.AST, store_set: Set[str]) -> None:
        bound = self._extract_pattern_bound_name(node)
        if bound is not None and not self._is_in_comp(bound):
            store_set.add(bound)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.AST) -> None:
        self.visit_MatchAs(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:
        self.visit_MatchAs(node)

    def visit_MatchAs(self, node: ast.AST) -> None:
        self.generic_visit(node)


class _FuncScopeVisitor(_BaseScopeVisitor):
    """Tracks local bindings and free variable loads escaping a nested function or lambda scope."""

    def __init__(self) -> None:
        super().__init__()
        self.params: Set[str] = set()
        self.local_stores: Set[str] = set()
        self.direct_loads: Set[str] = set()
        self.nested_free_reads: Set[str] = set()

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        self._visit_nested_named_scope(node, self.local_stores, self.nested_free_reads)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_nested_named_scope(node, self.local_stores, self.nested_free_reads)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_nested_lambda(node, self.nested_free_reads)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._visit_named_expr(node, self.local_stores, check_comp=False)

    def visit_MatchAs(self, node: ast.AST) -> None:
        self._record_pattern_binding(node, self.local_stores)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name and not self._is_in_comp(node.name):
            self.local_stores.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            if not self._is_in_comp(node.id):
                self.local_stores.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            if not self._is_in_comp(node.id):
                self.direct_loads.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_names(node, self.local_stores)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_names(node, self.local_stores)


class _ClassScopeVisitor(_BaseScopeVisitor):
    """Tracks class attributes and free variable reads escaping an executed class body."""

    def __init__(self) -> None:
        super().__init__()
        self.class_stores: Set[str] = set()
        self.free_reads: Set[str] = set()

    def _record_class_load(self, name: str) -> None:
        if self._is_in_comp(name):
            return
        if self._in_comp_body > 0:
            # Comprehensions in class bodies cannot resolve class attributes (PEP 227)
            if name not in self.globals:
                self.free_reads.add(name)
        elif name not in (self.class_stores | self.globals):
            self.free_reads.add(name)

    def _record_class_store(self, name: str) -> None:
        if (
            not self._is_in_comp(name)
            and self._in_comp_body == 0
            and name not in (self.globals | self.nonlocals)
        ):
            self.class_stores.add(name)

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        self._visit_nested_named_scope(node, self.class_stores, self.free_reads)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_nested_named_scope(node, self.class_stores, self.free_reads)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_nested_lambda(node, self.free_reads)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._record_class_store(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._record_class_load(node.target.id)
            self.visit(node.value)
            self._record_class_store(node.target.id)
        else:
            self.visit(node.target)
            self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._visit_named_expr(node, self.class_stores, check_comp=True)

    def visit_MatchAs(self, node: ast.AST) -> None:
        self._record_pattern_binding(node, self.class_stores)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record_class_load(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._record_class_store(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_names(node, self.class_stores)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_names(node, self.class_stores)


def _collect_func_args(args: ast.arguments) -> List[ast.arg]:
    """Collects all arg objects declared in a function or lambda signature."""
    all_args = (
        list(getattr(args, "posonlyargs", []))
        + list(args.args)
        + list(args.kwonlyargs)
    )
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)
    return all_args


def _collect_func_params(args: ast.arguments) -> Set[str]:
    """Collects parameter names declared in a function or lambda signature."""
    return {a.arg for a in _collect_func_args(args)}


class _OuterExprVisitor(_BaseScopeVisitor):
    """Collects loaded names from outer-scope expressions (defaults, decorators, annotations, bases)."""

    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self._is_in_comp(node.id):
            self.names.add(node.id)

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
    ) -> None:
        dummy_stores: Set[str] = set()
        self._visit_nested_named_scope(node, dummy_stores, self.names)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_nested_lambda(node, self.names)


def _extract_nested_scope_free_reads(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef],
) -> Set[str]:
    """Extracts free variable references escaping a nested function or executed class body."""
    type_param_names: Set[str] = set()
    for tp in getattr(node, "type_params", []):
        tp_name = getattr(tp, "name", None)
        if isinstance(tp_name, str):
            type_param_names.add(tp_name)

    outer_visitor = _OuterExprVisitor()
    for tp in getattr(node, "type_params", []):
        outer_visitor.visit(tp)

    if isinstance(node, ast.ClassDef):
        for b in node.bases:
            outer_visitor.visit(b)
        for kw in node.keywords:
            outer_visitor.visit(kw.value)
        for dec in node.decorator_list:
            outer_visitor.visit(dec)
        class_visitor = _ClassScopeVisitor()
        for tp_name in type_param_names:
            class_visitor.class_stores.add(tp_name)
        for stmt in node.body:
            class_visitor.visit(stmt)
        return (outer_visitor.names - type_param_names) | (
            class_visitor.free_reads - class_visitor.globals - type_param_names
        )

    defaults = node.args.defaults + [kw for kw in node.args.kw_defaults if kw is not None]
    for d in defaults:
        outer_visitor.visit(d)
    for arg in _collect_func_args(node.args):
        if arg.annotation is not None:
            outer_visitor.visit(arg.annotation)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in node.decorator_list:
            outer_visitor.visit(dec)
        if node.returns is not None:
            outer_visitor.visit(node.returns)

    func_visitor = _FuncScopeVisitor()
    for tp_name in type_param_names:
        func_visitor.local_stores.add(tp_name)
    func_visitor.params = _collect_func_params(node.args)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for stmt in node.body:
            func_visitor.visit(stmt)
    elif isinstance(node, ast.Lambda):
        func_visitor.visit(node.body)

    escaped = (
        (func_visitor.direct_loads | func_visitor.nested_free_reads)
        - (func_visitor.params | func_visitor.local_stores)
    ) | func_visitor.nonlocals
    escaped -= func_visitor.globals
    return (outer_visitor.names - type_param_names) | escaped


def _extract_assigned_names(target: ast.AST) -> List[str]:
    """Extracts bare variable names bound by an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_extract_assigned_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_assigned_names(target.value)
    return []


class _DownstreamReadVisitor(_BaseScopeVisitor):
    """Walks AST statements in a lexical scope tracking direct and nested free reads.

    Note: Control-flow kill tracking is a practical must/may approximation,
    not a full data-flow engine.
    """

    def __init__(
        self,
        u_end: int,
        u_end_col: Optional[int],
        candidates: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.candidates = candidates
        self.loaded: Set[str] = set()
        self.killed: Set[str] = set()

    def _is_node_after_unit(self, node: ast.AST) -> bool:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if lineno > self.u_end:
            return True
        if lineno == self.u_end:
            return (
                self.u_end_col is not None
                and getattr(node, "col_offset", 0) >= int(self.u_end_col)
            )
        return False

    def _record_killed_targets(self, targets: Iterable[ast.AST]) -> None:
        for t in targets:
            if self._is_node_after_unit(t):
                for name in _extract_assigned_names(t):
                    if not self._is_in_comp(name):
                        self.killed.add(name)

    def _record_downstream_read(self, name: str) -> None:
        if self._is_in_comp(name) or name in self.killed:
            return
        if self.candidates is None or name in self.candidates:
            self.loaded.add(name)

    def visit_FunctionDef(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef],
    ) -> None:
        if self._is_node_after_unit(node):
            free_reads = _extract_nested_scope_free_reads(node)
            for name in free_reads:
                self._record_downstream_read(name)
            name_attr = getattr(node, "name", None)
            if name_attr and not self._is_in_comp(name_attr):
                self.killed.add(name_attr)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._is_node_after_unit(node):
            free_reads = _extract_nested_scope_free_reads(node)
            for name in free_reads:
                self._record_downstream_read(name)
            if not self._is_in_comp(node.name):
                self.killed.add(node.name)
        else:
            n_start = getattr(node, "lineno", 0)
            n_end = getattr(node, "end_lineno", None) or n_start
            if n_start <= self.u_end <= n_end:
                for stmt in node.body:
                    self.visit(stmt)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_FunctionDef(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and self._is_node_after_unit(node):
            self._record_downstream_read(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and self._is_node_after_unit(node.target):
            self._record_downstream_read(node.target.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)
        self._record_killed_targets(node.targets)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)
            self._record_killed_targets([node.target])
        else:
            self.visit(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            self.visit(t)
        self._record_killed_targets(node.targets)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)
        # Named expressions (:=) may be conditionally evaluated (e.g. in 'and', 'or',
        # 'if', or comprehensions). Conservatively do not treat them as definite kills.

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        killed_before = set(self.killed)
        self.killed = set(killed_before)
        for stmt in node.body:
            self.visit(stmt)
        killed_body = set(self.killed)

        if node.orelse:
            self.killed = set(killed_before)
            for stmt in node.orelse:
                self.visit(stmt)
            self.killed = killed_body & self.killed
        else:
            self.killed = killed_before

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        killed_before = set(self.killed)
        self._record_killed_targets([node.target])
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        self.killed = killed_before

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_While(self, node: ast.While) -> None:
        killed_before = set(self.killed)
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        self.killed = killed_before

    def visit_Try(self, node: ast.Try) -> None:
        killed_before = set(self.killed)
        for stmt in node.body:
            self.visit(stmt)
        killed_try_body = set(self.killed)

        handler_kills: List[Set[str]] = []
        for handler in node.handlers:
            self.killed = set(killed_before)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name and self._is_node_after_unit(handler):
                self.killed.add(handler.name)
            for stmt in handler.body:
                self.visit(stmt)
            if handler.name:
                self.killed.discard(handler.name)
            handler_kills.append(set(self.killed))

        self.killed = killed_try_body
        for stmt in node.orelse:
            self.visit(stmt)
        killed_try_else = set(self.killed)

        if node.handlers:
            surviving = killed_try_else
            for hk in handler_kills:
                surviving = surviving & hk
        else:
            surviving = killed_try_else

        self.killed = surviving
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_TryStar(self, node: ast.AST) -> None:
        self.visit_Try(node)  # type: ignore[arg-type]

    def visit_With(self, node: ast.With) -> None:
        killed_before = set(self.killed)
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._record_killed_targets([item.optional_vars])
        for stmt in node.body:
            self.visit(stmt)
        self.killed = killed_before

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)  # type: ignore[arg-type]

    def visit_Match(self, node: ast.AST) -> None:
        self.visit(getattr(node, "subject"))
        killed_before = set(self.killed)
        for case in getattr(node, "cases", []):
            self.killed = set(killed_before)
            self.visit(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)
        self.killed = killed_before

    def visit_MatchAs(self, node: ast.AST) -> None:
        bound = self._extract_pattern_bound_name(node)
        if bound is not None and self._is_node_after_unit(node):
            if not self._is_in_comp(bound):
                self.killed.add(bound)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_node_after_unit(node):
            self._record_import_names(node, self.killed)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_node_after_unit(node):
            self._record_import_names(node, self.killed)


def collect_downstream_read_names(
    source_text: str,
    unit: Dict[str, Any],
    candidates: Optional[Set[str]] = None,
    *,
    tree: Optional[ast.AST] = None,
) -> Optional[Set[str]]:
    """Identifies variable names loaded downstream of a unit within its lexical execution scope.

    Algorithmic Complexity & Scope Capping:
    Traversal is capped to the body of the innermost enclosing function (`scope_node.body`),
    limiting visitor complexity to $O(S)$ where $S$ is the statement count within the enclosing
    lexical scope, rather than scanning the entire module. Passing a pre-parsed AST via `tree`
    avoids redundant full-module parsing across multiple units.
    """
    u_end = parse_unit_coord(unit, "end", default=parse_unit_coord(unit, "start", default=0))
    if u_end <= 0:
        return None

    enc = _find_innermost_enclosing_node(
        source_text, unit, (ast.FunctionDef, ast.AsyncFunctionDef), tree=tree
    )
    if enc is not None:
        scope_node: ast.AST = enc[0]
    else:
        scope_tree = _parse_source_tree(source_text, tree=tree)
        if scope_tree is None:
            return None
        scope_node = scope_tree

    u_end_col = _extract_unit_end_col(unit)
    if u_end_col is None:
        u_end_col = _resolve_unit_ast_end_col(scope_node, unit)

    visitor = _DownstreamReadVisitor(
        u_end, u_end_col, candidates=candidates
    )
    for stmt in getattr(scope_node, "body", []):
        visitor.visit(stmt)

    return visitor.loaded


def _pair_clone_outputs(
    u1_outs: Sequence[str],
    u2_outs: Sequence[str],
) -> List[Tuple[str, str]]:
    """Establishes an ordered 1-to-1 mapping between clone output variables.

    Structural Semantic Invariant:
    For isomorphic Type-1 (identical) and Type-2 (renamed variable) clones of equal output arity
    (`len(u1_outs) == len(u2_outs)`), AST statements follow identical top-to-bottom execution sequence.
    Because unit outputs are harvested in lexical AST store order (first write per variable), the
    positional sequence of outputs reflects the corresponding semantic role of each output slot.
    Positional 1-to-1 alignment (`zip(u1_dedup, u2_dedup)`) strictly preserves AST semantic roles
    across renamed variables, preventing inverted variable assignments when common names collide.

    When candidate output lengths differ (`len(u1_outs) != len(u2_outs)`), positional alignment across
    distinct variable names no longer holds. In this unequal-length scenario, only identical common
    names `(o, o)` present in both units can be safely paired without guessing correspondence, and any
    unique names are excluded from pairing.
    """
    u1_dedup = list(dict.fromkeys(u1_outs))
    u2_dedup = list(dict.fromkeys(u2_outs))
    if len(u1_dedup) == len(u2_dedup):
        if set(u1_dedup) == set(u2_dedup):
            return [(o, o) for o in u1_dedup]
        return list(zip(u1_dedup, u2_dedup))
    common = [o for o in u1_dedup if o in u2_dedup]
    return [(o, o) for o in common]


class GeneratorCloneSideData(NamedTuple):
    """Encapsulates per-side output candidates, downstream reads, and definite stores for generator refactoring."""

    outputs: Sequence[str]
    downstream: Optional[Set[str]] = None
    definite: Optional[Set[str]] = None


def resolve_generator_subroutine_outputs(
    u1_outs: Union[GeneratorCloneSideData, Sequence[str]],
    u2_outs: Union[GeneratorCloneSideData, Sequence[str]] = (),
    *,
    downstream1: Optional[Set[str]] = None,
    downstream2: Optional[Set[str]] = None,
    u1_definite: Optional[Set[str]] = None,
    u2_definite: Optional[Set[str]] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Selects output variables needed downstream by either clone side, preserving per-side necessity.

    Fail-Closed Policy:
    1. If output arities differ, every required downstream output must have a verified counterpart
       on the other side. If any needed output lacks a counterpart, the candidate is rejected (returns None).
    2. Safe Fallback: When downstream usage is unknown (None), fallback only considers outputs that
       are definitely assigned (in u1_definite / u2_definite if supplied), preventing unassigned loop
       variables from being conservatively retained.

    Returns:
        (outputs, target_outs2) if all needed downstream outputs have valid counterparts and can be
        safely returned; None if counterpart mismatches prevent safe refactoring.
    """
    if isinstance(u1_outs, GeneratorCloneSideData):
        s1 = u1_outs
    else:
        s1 = GeneratorCloneSideData(u1_outs, downstream1, u1_definite)

    if isinstance(u2_outs, GeneratorCloneSideData):
        s2 = u2_outs
    else:
        s2 = GeneratorCloneSideData(u2_outs, downstream2, u2_definite)

    outs1 = list(s1.outputs)
    outs2 = list(s2.outputs)
    d1_raw = s1.downstream
    d2_raw = s2.downstream
    u1_def = s1.definite
    u2_def = s2.definite

    d1_needed = (
        d1_raw
        if d1_raw is not None
        else (set(outs1) & u1_def if u1_def is not None else set(outs1))
    )
    d2_needed = (
        d2_raw
        if d2_raw is not None
        else (set(outs2) & u2_def if u2_def is not None else set(outs2))
    )

    needed1 = [o for o in outs1 if o in d1_needed]
    needed2 = [o for o in outs2 if o in d2_needed]

    if not needed1 and not needed2:
        return [], []

    pairs = _pair_clone_outputs(outs1, outs2)
    paired_u1 = {o1 for o1, _ in pairs}
    paired_u2 = {o2 for _, o2 in pairs}

    # Fail closed: every required downstream output must have a paired counterpart
    if any(o1 not in paired_u1 for o1 in needed1) or any(o2 not in paired_u2 for o2 in needed2):
        return None

    kept_pairs = [
        (o1, o2)
        for o1, o2 in pairs
        if o1 in d1_needed or o2 in d2_needed
    ]

    return [o1 for o1, _ in kept_pairs], [o2 for _, o2 in kept_pairs]


def is_async_generator_with_return_value(
    *scopes: Optional[Dict[str, Any]],
    has_outputs: bool = False,
) -> bool:
    """Checks whether scopes represent an async generator attempting to return values or outputs.

    Under PEP 525, asynchronous generators cannot return values via StopAsyncIteration.
    Any attempt to return explicit values or propagate downstream outputs from an async
    generator subroutine is illegal and must be rejected.
    """
    has_yield = any(bool(s.get("has_yield")) for s in scopes if s is not None)
    is_async = any(bool(s.get("is_async")) for s in scopes if s is not None)
    has_ret = has_outputs or any(
        bool(s.get("has_return_value")) for s in scopes if s is not None
    )
    return has_yield and is_async and has_ret


def _load_unit_file_text(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Reads source text from disk for a unit when not provided in-memory."""
    f_raw = normalize_path_string(str(unit.get("file") or ""), strip_anchor=True)
    if not f_raw:
        return None
    p = Path(f_raw)
    f_path = p if p.is_file() or p.is_absolute() else (Path(repo_root or os.getcwd()) / p)
    if not f_path.is_file():
        return None
    try:
        return f_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _extract_effective_unit_outputs(
    unit: Dict[str, Any], scope: Dict[str, Any]
) -> List[str]:
    """Extracts non-global, non-local outputs from precomputed unit dict or analyzed scope."""
    raw = unit.get("outputs")
    cands = raw if isinstance(raw, (list, tuple, set)) else scope.get("outputs", [])
    excluded = set(scope.get("globals", [])) | set(scope.get("nonlocals", []))
    return [v for v in cands if v not in excluded]


def resolve_clone_generator_subroutine_outputs(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    source_text1: Optional[str] = None,
    source_text2: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Resolves output variable bindings for clone generator subroutines across two code units.

    Prefers precomputed 'outputs' on u1 and u2 if already present. Otherwise, computes
    downstream read names for both clone sides, combines definite stores and inputs, and
    delegates to resolve_generator_subroutine_outputs.

    Returns:
        A tuple of (outputs_side1, outputs_side2) if compatible; None if output
        counterpart mismatches prevent safe refactoring.
    """
    if (
        "outputs" in u1
        and isinstance(u1["outputs"], (list, tuple, set))
        and "outputs" in u2
        and isinstance(u2["outputs"], (list, tuple, set))
    ):
        return (
            _extract_effective_unit_outputs(u1, scope1),
            _extract_effective_unit_outputs(u2, scope2),
        )

    u1_raw = _extract_effective_unit_outputs(u1, scope1)
    u2_raw = _extract_effective_unit_outputs(u2, scope2)

    targets = [
        (u1, set(u1_raw), source_text1, tree1),
        (u2, set(u2_raw), source_text2, tree2),
    ]
    downstreams: List[Optional[Set[str]]] = []
    for unit_target, cand_set, in_mem_text, parsed_ast in targets:
        text_to_use = (
            in_mem_text
            if in_mem_text is not None
            else (unit_target.get("source_text") or unit_target.get("file_source"))
        )
        if text_to_use is None:
            text_to_use = _load_unit_file_text(unit_target, repo_root=repo_root)

        if text_to_use is not None:
            d_reads = collect_downstream_read_names(
                str(text_to_use), unit_target, candidates=cand_set, tree=parsed_ast
            )
        else:
            d_reads = None
        downstreams.append(d_reads)

    downstream1, downstream2 = downstreams[0], downstreams[1]
    u1_def = set(scope1.get("definite_stores", [])) | set(scope1.get("inputs", []))
    u2_def = set(scope2.get("definite_stores", [])) | set(scope2.get("inputs", []))
    side1 = GeneratorCloneSideData(u1_raw, downstream1, u1_def)
    side2 = GeneratorCloneSideData(u2_raw, downstream2, u2_def)
    return resolve_generator_subroutine_outputs(side1, side2)


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
    root = Path(repo_root or os.getcwd())
    try:
        p1 = Path(norm1)
        p2 = Path(norm2)
        p1_full = p1 if p1.is_file() or p1.is_absolute() else (root / p1)
        p2_full = p2 if p2.is_file() or p2.is_absolute() else (root / p2)
        if p1_full.is_symlink() or p2_full.is_symlink():
            return False
        if p1_full.is_file() and p2_full.is_file():
            return p1_full.resolve() == p2_full.resolve()
        if paths_match_boundary(norm1, norm2):
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
    """Removes receiver parameters from inputs when only one unit receives them and neither references them."""
    res = list(inputs)
    rec_candidates: Set[str] = {"self", "cls"}
    for u in (u1, u2):
        rec_p = u.get("receiver_param")
        if rec_p:
            rec_candidates.add(rec_p)
    for rec in sorted(rec_candidates):
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
        and "receiver_param" in unit
        and "is_async" in unit
    ):
        return
    source = _load_unit_file_text(unit, repo_root=repo_root)
    if source is None:
        return
    fn_meta = find_enclosing_function(source, unit)
    cls_meta = find_enclosing_class(source, unit)
    if cls_meta:
        if "enclosing_class" not in unit:
            unit["enclosing_class"] = cls_meta["name"]
        if "enclosing_class_start" not in unit:
            unit["enclosing_class_start"] = cls_meta["start"]
    is_method = _is_method_of_class(fn_meta, cls_meta)
    if "receiver_kind" not in unit:
        if is_method:
            rec_k = _get_enclosing_receiver_kind(fn_meta)
            unit["receiver_kind"] = rec_k
            if rec_k == "static":
                unit["is_static"] = True
        else:
            unit["receiver_kind"] = None
    if "receiver_param" not in unit:
        if is_method and unit.get("receiver_kind") in ("instance", "class"):
            unit["receiver_param"] = fn_meta.get("receiver_param") if fn_meta else None
        else:
            unit["receiver_param"] = None
    if "is_async" not in unit:
        unit["is_async"] = bool(fn_meta.get("is_async")) if fn_meta else False


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
