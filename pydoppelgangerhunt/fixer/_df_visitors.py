"""AST scope and downstream read visitors for lexical dataflow analysis."""

from __future__ import annotations

import ast
import builtins
from enum import Enum
import logging
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

_BUILTIN_NAMES: Set[str] = set(dir(builtins))
logger = logging.getLogger(__name__)


def _record_import_names(
    node: Union[ast.Import, ast.ImportFrom], store_set: Set[str]
) -> None:
    """Extracts bound symbol names from Import or ImportFrom statements, ignoring '*'.

    Note:
        Wildcard star imports (e.g., ``from numpy import *``) cannot statically reveal their
        bound symbols without executing or resolving the module namespace. Dropping ``'*'``
        avoids polluting symbol sets with invalid identifiers. In strict closure mode,
        scopes containing wildcard imports are treated as having dynamic scope reflection,
        failing closed rather than optimistically assuming built-ins are untouched.
    """
    for alias in node.names:
        bound = (
            alias.asname
            or (
                alias.name
                if isinstance(node, ast.ImportFrom)
                else alias.name.split(".", 1)[0]
            )
        )
        if bound != "*":
            store_set.add(bound)
        else:
            mod = getattr(node, "module", None) or "<unknown>"
            logger.debug(
                "Wildcard import detected ('from %s import *'); built-in symbol "
                "shadowing analysis remains optimistic for this scope",
                mod,
            )


def _extract_scope_stmts(scope_node: ast.AST) -> List[ast.AST]:
    """Defensively extracts statements or expression nodes from a scope's body attribute."""
    raw_body = getattr(scope_node, "body", [])
    if isinstance(raw_body, list):
        return raw_body
    if isinstance(raw_body, ast.AST):
        return [raw_body]
    return []


def _extract_pattern_bound_name(node: ast.AST) -> Optional[str]:
    """Extracts identifier bound by MatchAs, MatchStar, or MatchMapping pattern nodes."""
    name = getattr(node, "name", None) or getattr(node, "rest", None)
    if isinstance(name, str) and name and name != "_":
        return name
    return None


class _BaseScopeVisitor(ast.NodeVisitor):
    """Shared helpers for scope visitors."""

    _record_import_names = staticmethod(_record_import_names)
    _extract_pattern_bound_name = staticmethod(_extract_pattern_bound_name)

    def __init__(self) -> None:
        self._comp_targets: List[Set[str]] = []
        self._in_comp_body: int = 0
        self.globals: Set[str] = set()
        self.nonlocals: Set[str] = set()
        self.stored: Set[str] = set()
        self.load_nodes: List[ast.Name] = []

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def _is_in_comp(self, name: str) -> bool:
        return any(name in targets for targets in self._comp_targets)

    def _traverse_isolated_cases(
        self,
        node: ast.AST,
        save_state: Callable[[], Set[str]],
        restore_state: Callable[[Set[str]], None],
    ) -> None:
        self.visit(getattr(node, "subject"))
        base_state = save_state()
        for case in getattr(node, "cases", []):
            restore_state(set(base_state))
            self.visit(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)
        restore_state(base_state)

    def _visit_comp_target(self, target: ast.AST, comp_set: Set[str]) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_comp_target(elt, comp_set)
        elif isinstance(target, ast.Starred):
            self._visit_comp_target(target.value, comp_set)
        else:
            if isinstance(target, ast.Name):
                comp_set.add(target.id)
            self.visit(target)

    def _visit_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    ) -> None:
        if not node.generators:
            return
        # First generator's iter evaluates in enclosing scope (class scope if in class body)
        self.visit(node.generators[0].iter)
        stored_before = set(self.stored)
        comp_set: Set[str] = set()
        self._comp_targets.append(comp_set)
        self._in_comp_body += 1
        try:
            for idx, gen in enumerate(node.generators):
                if idx > 0:
                    self.visit(gen.iter)
                self._visit_comp_target(gen.target, comp_set)
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
            self.stored = stored_before

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def _can_store_in_class_scope(self, name: str) -> bool:
        return (
            not self._is_in_comp(name)
            and self._in_comp_body == 0
            and name not in (self.globals | self.nonlocals)
        )

    def _visit_named_expr(
        self,
        node: ast.NamedExpr,
        store_set: Set[str],
        check_comp: bool = False,
    ) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            if not check_comp or self._can_store_in_class_scope(node.target.id):
                store_set.add(node.target.id)
        else:
            self.visit(node.target)


    def _visit_nested_named_scope(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        store_set: Set[str],
        reads_set: Set[str],
    ) -> None:
        store_set.add(node.name)
        free_reads, free_nodes = _extract_nested_scope_free_reads_with_nodes(node)
        reads_set.update(r for r in free_reads if not self._is_in_comp(r))
        self.load_nodes.extend(n for n in free_nodes if not self._is_in_comp(n.id))

    def _visit_nested_lambda(
        self,
        node: ast.Lambda,
        reads_set: Set[str],
    ) -> None:
        free_reads, free_nodes = _extract_nested_scope_free_reads_with_nodes(node)
        reads_set.update(r for r in free_reads if not self._is_in_comp(r))
        self.load_nodes.extend(n for n in free_nodes if not self._is_in_comp(n.id))

    def _record_pattern_binding(self, node: ast.AST, store_set: Set[str]) -> None:
        self.generic_visit(node)
        bound = self._extract_pattern_bound_name(node)
        if bound is not None and not self._is_in_comp(bound):
            store_set.add(bound)

    def visit_MatchStar(self, node: ast.AST) -> None:
        self.visit_MatchAs(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:
        self.visit_MatchAs(node)

    def visit_MatchAs(self, node: ast.AST) -> None:
        self.generic_visit(node)

    def _visit_type_alias_node(
        self,
        node: ast.AST,
        store_set: Set[str],
        on_free_read: Callable[[str], None],
    ) -> None:
        name_node = getattr(node, "name", None)
        if isinstance(name_node, ast.Name):
            store_set.add(name_node.id)
        free_reads, free_nodes = _extract_type_alias_free_reads_with_nodes(node)
        for free in free_reads:
            on_free_read(free)
        self.load_nodes.extend(n for n in free_nodes if not self._is_in_comp(n.id))


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
                self.load_nodes.append(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_names(node, self.local_stores)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_names(node, self.local_stores)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        self._visit_type_alias_node(
            node, self.local_stores, self.nested_free_reads.add
        )


class _ClassScopeVisitor(_BaseScopeVisitor):
    """Tracks class attributes and free variable reads escaping an executed class body."""

    def __init__(self) -> None:
        super().__init__()
        self.class_stores: Set[str] = set()
        self.free_reads: Set[str] = set()

    def _record_class_load(self, name: str, node: Optional[ast.Name] = None) -> None:
        if self._is_in_comp(name):
            return
        if self._in_comp_body > 0:
            # Comprehensions in class bodies cannot resolve class attributes (PEP 227)
            if name not in self.globals:
                self.free_reads.add(name)
                if node is not None:
                    self.load_nodes.append(node)
        elif name not in (self.class_stores | self.globals):
            self.free_reads.add(name)
            if node is not None:
                self.load_nodes.append(node)

    def _record_class_store(self, name: str) -> None:
        if self._can_store_in_class_scope(name):
            self.class_stores.add(name)

    def _visit_class_nested_scope(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda],
    ) -> None:
        hdr_reads, hdr_nodes = _extract_nested_scope_free_reads_with_nodes(
            node, include_header=True, include_body=False
        )
        seen_hdr: Set[str] = set()
        for n in hdr_nodes:
            self._record_class_load(n.id, n)
            seen_hdr.add(n.id)
        for r in hdr_reads - seen_hdr:
            self._record_class_load(r)
        if not isinstance(node, ast.Lambda):
            self._record_class_store(node.name)
        body_reads, body_nodes = _extract_nested_scope_free_reads_with_nodes(
            node, include_header=False, include_body=True
        )
        self.free_reads.update(
            r for r in body_reads if not self._is_in_comp(r) and r not in self.globals
        )
        self.load_nodes.extend(
            n for n in body_nodes if not self._is_in_comp(n.id) and n.id not in self.globals
        )

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        self._visit_class_nested_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_class_nested_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_class_nested_scope(node)

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
            self._record_class_load(node.target.id, node.target)
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
        elif not isinstance(node.target, ast.Name):
            self.visit(node.target)

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._visit_named_expr(node, self.class_stores, check_comp=True)

    def visit_MatchAs(self, node: ast.AST) -> None:
        self._record_pattern_binding(node, self.class_stores)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record_class_load(node.id, node)
        elif isinstance(node.ctx, ast.Store):
            self._record_class_store(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_names(node, self.class_stores)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_names(node, self.class_stores)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        self._visit_type_alias_node(
            node, self.class_stores, self._record_class_load
        )


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
    """Collects loaded names from outer-scope expressions (defaults, decorators,
    annotations, bases)."""

    def __init__(self) -> None:
        super().__init__()
        self.names: Set[str] = set()
        self.ordered_names: List[str] = []
        self.stored: Set[str] = set()

    def _record_name(self, name: str) -> None:
        if name not in self.names:
            self.ordered_names.append(name)
        self.names.add(name)

    def _sync_added_names(self, before: Set[str], before_nodes_len: int) -> None:
        self.names -= self.stored - before
        added = self.names - before
        if not added:
            return
        load_nodes = [
            child
            for child in self.load_nodes[before_nodes_len:]
            if child.id in added and not self._is_in_comp(child.id)
        ]
        load_nodes.sort(
            key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
        )
        for child in load_nodes:
            if child.id not in self.ordered_names:
                self.ordered_names.append(child.id)
        for name in sorted(added):
            if name not in self.ordered_names:
                self.ordered_names.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and not self._is_in_comp(node.id)
            and node.id not in self.stored
        ):
            self._record_name(node.id)
            self.load_nodes.append(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.stored.add(node.target.id)

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
    ) -> None:
        before = set(self.names)
        before_nodes_len = len(self.load_nodes)
        dummy_stores: Set[str] = set()
        self._visit_nested_named_scope(node, dummy_stores, self.names)
        self._sync_added_names(before, before_nodes_len)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        before = set(self.names)
        before_nodes_len = len(self.load_nodes)
        self._visit_nested_lambda(node, self.names)
        self._sync_added_names(before, before_nodes_len)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        before = set(self.names)
        before_nodes_len = len(self.load_nodes)
        dummy_stores: Set[str] = set()
        self._visit_type_alias_node(
            node,
            dummy_stores,
            lambda free: self.names.add(free) if not self._is_in_comp(free) else None,
        )
        self._sync_added_names(before, before_nodes_len)


def _extract_type_alias_free_reads_with_nodes(
    node: ast.AST,
) -> Tuple[Set[str], List[ast.Name]]:
    """Extracts free variable references and their AST load nodes from a TypeAlias."""
    tp_names = {
        getattr(tp, "name", None)
        for tp in getattr(node, "type_params", [])
        if isinstance(getattr(tp, "name", None), str)
    }
    alias_name_node = getattr(node, "name", None)
    own_name = {alias_name_node.id} if isinstance(alias_name_node, ast.Name) else set()
    sub_visitor = _OuterExprVisitor()
    for tp in getattr(node, "type_params", []):
        sub_visitor.visit(tp)
    if hasattr(node, "value"):
        sub_visitor.visit(getattr(node, "value"))
    allowed = sub_visitor.names - tp_names - own_name
    return allowed, [n for n in sub_visitor.load_nodes if n.id in allowed]


def _extract_type_alias_free_reads(node: ast.AST) -> Set[str]:
    """Extracts free variable references escaping a TypeAlias definition."""
    return _extract_type_alias_free_reads_with_nodes(node)[0]


def _extract_nested_scope_free_reads_with_nodes(
    node: Union[
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.GeneratorExp
    ],
    candidates: Optional[Set[str]] = None,
    *,
    include_header: bool = True,
    include_body: bool = True,
) -> Tuple[Set[str], List[ast.Name]]:
    """Extracts free variable references and their AST load nodes escaping a nested scope."""
    if isinstance(node, ast.GeneratorExp):
        gen_visitor = _OuterExprVisitor()
        gen_visitor.visit(node)
        res = (
            gen_visitor.names & candidates
            if candidates is not None
            else gen_visitor.names
        )
        return res, [n for n in gen_visitor.load_nodes if n.id in res]

    type_param_names: Set[str] = set()
    for tp in getattr(node, "type_params", []):
        tp_name = getattr(tp, "name", None)
        if isinstance(tp_name, str):
            type_param_names.add(tp_name)

    unscoped_visitor = _OuterExprVisitor()
    outer_visitor = _OuterExprVisitor()
    if include_header:
        for tp in getattr(node, "type_params", []):
            outer_visitor.visit(tp)

    body_free: Set[str] = set()
    body_nodes: List[ast.Name] = []
    if isinstance(node, ast.ClassDef):
        if include_header:
            for b in node.bases:
                outer_visitor.visit(b)
            for kw in node.keywords:
                outer_visitor.visit(kw.value)
            for dec in node.decorator_list:
                unscoped_visitor.visit(dec)
        if include_body:
            class_visitor = _ClassScopeVisitor()
            for tp_name in type_param_names:
                class_visitor.class_stores.add(tp_name)
            for stmt in node.body:
                class_visitor.visit(stmt)
            body_free = (
                (class_visitor.free_reads | class_visitor.nonlocals)
                - class_visitor.globals
                - type_param_names
            )
            body_nodes = [
                n for n in class_visitor.load_nodes if n.id in body_free
            ]
    else:
        if include_header:
            defaults = node.args.defaults + [
                kw for kw in node.args.kw_defaults if kw is not None
            ]
            for d in defaults:
                unscoped_visitor.visit(d)
            for arg in _collect_func_args(node.args):
                if arg.annotation is not None:
                    outer_visitor.visit(arg.annotation)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    unscoped_visitor.visit(dec)
                if node.returns is not None:
                    outer_visitor.visit(node.returns)
        if include_body:
            func_visitor = _FuncScopeVisitor()
            for tp_name in type_param_names:
                func_visitor.local_stores.add(tp_name)
            func_visitor.params = _collect_func_params(node.args)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for stmt in node.body:
                    func_visitor.visit(stmt)
            elif isinstance(node, ast.Lambda):
                func_visitor.visit(node.body)
            body_escaped = (
                (func_visitor.direct_loads | func_visitor.nested_free_reads)
                - (
                    func_visitor.params
                    | func_visitor.local_stores
                    | func_visitor.globals
                )
            )
            body_free = (
                body_escaped | func_visitor.nonlocals
            ) - func_visitor.globals
            body_nodes = [
                n for n in func_visitor.load_nodes if n.id in body_escaped
            ]

    all_free = (
        (
            unscoped_visitor.names
            | (outer_visitor.names - type_param_names)
            | body_free
        )
        if include_header
        else body_free
    )
    res_free = all_free & candidates if candidates is not None else all_free
    all_nodes = (
        (
            unscoped_visitor.load_nodes
            + [n for n in outer_visitor.load_nodes if n.id not in type_param_names]
        )
        if include_header
        else []
    ) + body_nodes
    return res_free, [n for n in all_nodes if n.id in res_free]


def _extract_nested_scope_free_reads(
    node: Union[
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.GeneratorExp
    ],
    candidates: Optional[Set[str]] = None,
    *,
    include_header: bool = True,
    include_body: bool = True,
) -> Set[str]:
    """Extracts free variable references escaping a nested function, genexp, or class body."""
    return _extract_nested_scope_free_reads_with_nodes(
        node,
        candidates=candidates,
        include_header=include_header,
        include_body=include_body,
    )[0]


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


class _VisitorPassMode(str, Enum):
    """Execution pass mode for downstream AST read analysis."""

    AFTER_UNIT = "after_unit"
    LOOP_CARRIED = "loop_carried"


def _is_ast_node_within_unit_bounds(
    node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int],
    u_start_col: Optional[int] = None,
) -> bool:
    """Determines whether an AST node is positioned inside unit line and column bounds.

    When u_end_col is None, falls back to whole-line containment on u_end.
    """
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return False
    if lineno < u_start or lineno > u_end:
        return False
    if lineno == u_start and u_start_col is not None:
        col = getattr(node, "col_offset", None)
        if col is not None and int(col) < int(u_start_col):
            return False
    if lineno == u_end and u_end_col is not None:
        col = getattr(node, "col_offset", None)
        if col is None:
            return True
        return bool(int(col) < int(u_end_col))
    return True


class _DownstreamReadVisitor(_BaseScopeVisitor):
    """Walks AST statements in a lexical scope tracking direct, nested, and loop-carried reads."""

    def __init__(
        self,
        u_start: int,
        u_end: int,
        u_end_col: Optional[int],
        candidates: Optional[Set[str]] = None,
        enclosing_loops: Optional[List[Tuple[int, int]]] = None,
        pass_mode: Union[_VisitorPassMode, str] = _VisitorPassMode.AFTER_UNIT,
        is_module_scope: bool = False,
        u_start_col: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.u_start = u_start
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.u_start_col = u_start_col
        self.candidates = candidates
        self.enclosing_loops = enclosing_loops or []
        self.pass_mode = (
            _VisitorPassMode(pass_mode)
            if isinstance(pass_mode, str)
            else pass_mode
        )
        self.loaded: Set[str] = set()
        self.killed: Set[str] = set()
        self._nonclass_killed: Optional[Set[str]] = None
        self.outer_bound: Set[str] = set()
        self.unit_entry_killed: Optional[Set[str]] = None
        self.has_dynamic_read: bool = False
        self.is_module_scope = is_module_scope

    _DYNAMIC_READ_FUNCS = frozenset({"locals", "vars", "eval", "exec", "dir", "globals"})

    def visit(self, node: ast.AST) -> Any:
        if self.unit_entry_killed is None and self._is_node_inside_unit(node):
            self.unit_entry_killed = set(self.killed)
        return super().visit(node)

    def _is_node_after_unit(self, node: ast.AST) -> bool:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if lineno > self.u_end:
            return True
        if lineno == self.u_end:
            if self.u_end_col is not None:
                col = getattr(node, "col_offset", None)
                if col is None:
                    return True
                return bool(int(col) >= int(self.u_end_col))
            # When u_end_col cannot be resolved, fail-closed by treating
            # same-line nodes on u_end as downstream.
            return True
        return False

    def _is_node_inside_unit(self, node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node, self.u_start, self.u_end, self.u_end_col, self.u_start_col
        )

    def _is_loop_carried(self, node: ast.AST) -> bool:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if self._is_node_inside_unit(node) or self._is_node_after_unit(node):
            return False
        return any(l_start <= lineno <= l_end for l_start, l_end in self.enclosing_loops)

    def _loop_encloses_unit(self, node: ast.AST) -> bool:
        n_start = getattr(node, "lineno", 0) or 0
        n_end = getattr(node, "end_lineno", None) or n_start
        if not (n_start <= self.u_start and self.u_end <= n_end):
            return False
        if n_start < self.u_start or n_end > self.u_end:
            return True
        if self.u_start_col is not None:
            col = getattr(node, "col_offset", None)
            if col is not None and int(col) < int(self.u_start_col):
                return True
        return False

    def _should_inspect_read(self, node: ast.AST) -> bool:
        if self.pass_mode is _VisitorPassMode.LOOP_CARRIED:
            return self._is_loop_carried(node)
        return self._is_node_after_unit(node)

    def _record_killed_targets(self, targets: Iterable[ast.AST]) -> None:
        for t in targets:
            extracted = _extract_assigned_names(t)
            if self._nonclass_killed is None:
                self.outer_bound.update(extracted)
            if self._should_inspect_read(t):
                lineno = getattr(t, "lineno", None)
                if lineno == self.u_end and self.u_end_col is None:
                    continue
                for name in extracted:
                    if not self._is_in_comp(name):
                        self.killed.add(name)

    def _record_downstream_read(
        self, name: str, *, ignore_class_kills: bool = False
    ) -> None:
        bypass_class_kills = ignore_class_kills or (
            self._in_comp_body > 0 and name in self.outer_bound
        )
        active_killed = (
            self._nonclass_killed
            if bypass_class_kills and self._nonclass_killed is not None
            else self.killed
        )
        if self._is_in_comp(name) or name in active_killed:
            return
        if self.candidates is None or name in self.candidates:
            self.loaded.add(name)

    def visit_FunctionDef(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef],
    ) -> None:
        if self._should_inspect_read(node):
            if self._nonclass_killed is not None:
                for name in _extract_nested_scope_free_reads(
                    node,
                    candidates=self.candidates,
                    include_header=True,
                    include_body=False,
                ):
                    self._record_downstream_read(name)
                for name in _extract_nested_scope_free_reads(
                    node,
                    candidates=self.candidates,
                    include_header=False,
                    include_body=True,
                ):
                    self._record_downstream_read(name, ignore_class_kills=True)
            else:
                free_reads = _extract_nested_scope_free_reads(
                    node, candidates=self.candidates
                )
                for name in free_reads:
                    self._record_downstream_read(name)
            name_attr = getattr(node, "name", None)
            if isinstance(name_attr, str) and self._nonclass_killed is None:
                self.outer_bound.add(name_attr)
            lineno = getattr(node, "lineno", None)
            if (
                name_attr
                and not self._is_in_comp(name_attr)
                and not (lineno == self.u_end and self.u_end_col is None)
            ):
                self.killed.add(name_attr)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._nonclass_killed is None:
            self.outer_bound.add(node.name)
        n_start = getattr(node, "lineno", 0)
        n_end = getattr(node, "end_lineno", None) or n_start
        if n_start <= self.u_end <= n_end and not self._is_node_after_unit(node):
            for dec in node.decorator_list:
                self.visit(dec)
            for tp in getattr(node, "type_params", []):
                self.visit(tp)
            for base in node.bases:
                self.visit(base)
            for kw in node.keywords:
                self.visit(kw)
            prev_nonclass = self._nonclass_killed
            killed_before_class = set(self.killed)
            if self._nonclass_killed is None:
                self._nonclass_killed = set(killed_before_class)
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self._nonclass_killed = prev_nonclass
                self.killed = killed_before_class
            return
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_FunctionDef(node)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        name_node = getattr(node, "name", None)
        if isinstance(name_node, ast.Name) and self._nonclass_killed is None:
            self.outer_bound.add(name_node.id)
        if self._should_inspect_read(node):
            for name in _extract_type_alias_free_reads(node):
                self._record_downstream_read(name)
            if (
                isinstance(name_node, ast.Name)
                and not self._is_in_comp(name_node.id)
                and not (getattr(node, "lineno", None) == self.u_end and self.u_end_col is None)
            ):
                self.killed.add(name_node.id)

    def visit_Name(self, node: ast.Name) -> None:
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and self._should_inspect_read(node):
            self._record_downstream_read(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            if self._nonclass_killed is None:
                self.outer_bound.add(node.target.id)
            if self._should_inspect_read(node.target):
                self._record_downstream_read(node.target.id)
                if not self._is_in_comp(node.target.id):
                    lineno = getattr(node.target, "lineno", None)
                    if not (lineno == self.u_end and self.u_end_col is None):
                        self.killed.add(node.target.id)
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

    def _record_delete_reads(self, target: ast.AST) -> None:
        """Records variables targeted for deletion as downstream reads if downstream of
        unit boundary."""
        for subnode in ast.walk(target):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Del):
                if self._should_inspect_read(subnode):
                    self._record_downstream_read(subnode.id)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            self._record_delete_reads(t)
            self.visit(t)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if isinstance(node.target, ast.Name) and self._nonclass_killed is None:
            self.outer_bound.add(node.target.id)
        self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        killed_before = set(self.killed)
        for stmt in node.body:
            self.visit(stmt)
        killed_then = set(self.killed)

        self.killed = set(killed_before)
        for stmt in node.orelse:
            self.visit(stmt)
        killed_else = set(self.killed)

        if node.orelse:
            self.killed = killed_then & killed_else
        else:
            self.killed = killed_before

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        if self.pass_mode is _VisitorPassMode.LOOP_CARRIED and self._loop_encloses_unit(node):
            self.killed = set()
        self.visit(node.iter)
        killed_before = set(self.killed)
        self.visit(node.target)
        self._record_killed_targets([node.target])
        for stmt in node.body:
            self.visit(stmt)
        self.killed = set(killed_before)
        for stmt in node.orelse:
            self.visit(stmt)
        self.killed = killed_before

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_While(self, node: ast.While) -> None:
        if self.pass_mode is _VisitorPassMode.LOOP_CARRIED and self._loop_encloses_unit(node):
            self.killed = set()
        killed_before = set(self.killed)
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        self.killed = set(killed_before)
        for stmt in node.orelse:
            self.visit(stmt)
        self.killed = killed_before

    def visit_Try(self, node: Union[ast.Try, ast.AST]) -> None:
        killed_before = set(self.killed)
        for stmt in getattr(node, "body", []):
            self.visit(stmt)
        killed_try_body = set(self.killed)

        handler_kills: List[Set[str]] = []
        for handler in getattr(node, "handlers", []):
            self.killed = set(killed_before)
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name and self._should_inspect_read(handler):
                self.killed.add(handler.name)
            for stmt in handler.body:
                self.visit(stmt)
            # PEP 3110: Exception variables bound via 'except ... as e:' are implicitly
            # cleared (equivalent to 'del e') at the end of the except block in Python 3.
            if handler.name:
                self.killed.discard(handler.name)
            handler_kills.append(set(self.killed))

        self.killed = killed_try_body
        for stmt in getattr(node, "orelse", []):
            self.visit(stmt)
        killed_try_else = set(self.killed)

        handlers = getattr(node, "handlers", [])
        if handlers:
            surviving = killed_try_else
            for hk in handler_kills:
                surviving = surviving & hk
        else:
            surviving = killed_try_else

        # Seed finalbody with killed_before because exceptions in try body
        # could cause finalbody to run without executing try body assignments
        self.killed = set(killed_before)
        for stmt in getattr(node, "finalbody", []):
            self.visit(stmt)

        self.killed = surviving | set(self.killed)

    def visit_TryStar(self, node: ast.AST) -> None:
        self.visit_Try(node)

    def visit_With(self, node: Union[ast.With, ast.AsyncWith]) -> None:
        killed_before = set(self.killed)
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._record_killed_targets([item.optional_vars])
        for stmt in node.body:
            self.visit(stmt)
        # Deliberately fail-closed: context managers may exit early or swallow exceptions
        self.killed = killed_before

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Match(self, node: ast.AST) -> None:
        self._traverse_isolated_cases(
            node,
            lambda: self.killed,
            lambda s: setattr(self, "killed", s),
        )

    def visit_MatchAs(self, node: ast.AST) -> None:
        self.generic_visit(node)
        bound = self._extract_pattern_bound_name(node)
        if bound is not None and self._should_inspect_read(node):
            if not self._is_in_comp(bound) and not (
                getattr(node, "lineno", None) == self.u_end and self.u_end_col is None
            ):
                self.killed.add(bound)

    def visit_Import(self, node: ast.Import) -> None:
        if self._should_inspect_read(node) and not (
            getattr(node, "lineno", None) == self.u_end and self.u_end_col is None
        ):
            self._record_import_names(node, self.killed)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._should_inspect_read(node):
            if not (getattr(node, "lineno", None) == self.u_end and self.u_end_col is None):
                self._record_import_names(node, self.killed)
            if any(alias.name == "*" for alias in node.names):
                self.has_dynamic_read = True
                if self.candidates is not None:
                    for name in self.candidates:
                        self._record_downstream_read(name)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if self._should_inspect_read(node):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("builtins", "__builtins__")
            ):
                func_name = node.func.attr
            if func_name in self._DYNAMIC_READ_FUNCS:
                if func_name == "globals" and not self.is_module_scope:
                    return
                self.has_dynamic_read = True
                if self.candidates is not None:
                    for name in self.candidates:
                        self._record_downstream_read(name)


class _StrictPostLineVisitor(_DownstreamReadVisitor):
    """Downstream read visitor strictly checking nodes starting after a given line and column.

    Boundary Invariant:
        When ``u_end_col`` is None, same-line statements on line ``u_end`` return False
        (not after unit), whereas ``_DownstreamReadVisitor`` returns True (fail-closed).
        This distinction is deliberate: ``_StrictPostLineVisitor`` evaluates statements
        strictly occurring downstream of compound constructs (such as ``with`` or ``try``
        blocks), whereas ``_DownstreamReadVisitor`` enforces fail-closed capture on
        ambiguous unit boundaries.
    """

    def _is_node_after_unit(self, node: ast.AST) -> bool:
        """Determines if an AST node strictly occurs downstream of the unit boundary.

        When ``u_end_col`` is None, same-line statements on line ``u_end`` return False
        as they are not strictly downstream of the compound construct.
        """
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if lineno > self.u_end:
            return True
        if lineno == self.u_end and self.u_end_col is not None:
            col = getattr(node, "col_offset", None)
            return col is not None and col >= int(self.u_end_col)
        return False

    def _is_node_inside_unit(self, node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node, self.u_start, self.u_end, self.u_end_col, self.u_start_col
        )


def _block_terminates(statements: Sequence[ast.stmt]) -> bool:
    """Checks whether a sequence of statements unconditionally returns or raises."""
    for stmt in statements:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and _block_terminates(stmt.body) and _block_terminates(stmt.orelse):
                return True
    return False


class _UnitLiveInVisitor(_BaseScopeVisitor):
    """Extracts live-in variable names (loaded before being stored) within a unit's bounds."""

    def __init__(
        self,
        u_start: int,
        u_end: int,
        u_end_col: Optional[int] = None,
        candidates: Optional[Set[str]] = None,
        u_start_col: Optional[int] = None,
        scope_node: Optional[ast.AST] = None,
        *,
        include_annotations: bool = True,
    ) -> None:
        super().__init__()
        self.u_start = u_start
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.u_start_col = u_start_col
        self.candidates = candidates
        self.scope_node = scope_node
        self.include_annotations = include_annotations
        self.live_ins: Set[str] = set()
        self.stored: Set[str] = set()

    def _is_inside(self, node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node, self.u_start, self.u_end, self.u_end_col, self.u_start_col
        )

    def _record_live_in(self, name: str) -> None:
        """Records a variable name as a live-in if not stored, in comp, or builtin."""
        if (
            name not in self.stored
            and not self._is_in_comp(name)
            and (self.candidates is None or name in self.candidates)
            and name not in _BUILTIN_NAMES
        ):
            self.live_ins.add(name)

    def _record_nested_free_reads(
        self,
        node: Union[
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            ast.ClassDef,
        ],
        *,
        include_header: bool = True,
        include_body: bool = True,
    ) -> None:
        for free in _extract_nested_scope_free_reads(
            node,
            candidates=self.candidates,
            include_header=include_header,
            include_body=include_body,
        ):
            self._record_live_in(free)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if self._is_inside(node):
            self._record_nested_free_reads(node)
        else:
            self.generic_visit(node)

    def _visit_named_scope_def(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    ) -> None:
        if self.scope_node is not None and node is self.scope_node:
            self.generic_visit(node)
            return
        if self._is_inside(node):
            self._record_nested_free_reads(node, include_body=False)
            self.stored.add(node.name)
            if self.include_annotations or isinstance(node, ast.ClassDef):
                self._record_nested_free_reads(node, include_header=False)
        else:
            self.generic_visit(node)

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        self._visit_named_scope_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_named_scope_def(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_named_scope_def(node)

    def _visit_import_node(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        if self._is_inside(node):
            self._record_import_names(node, self.stored)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self._visit_import_node(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._visit_import_node(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self._is_inside(node):
            if isinstance(node.ctx, ast.Load):
                self._record_live_in(node.id)
            elif isinstance(node.ctx, ast.Store):
                if not self._is_in_comp(node.id):
                    self.stored.add(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._is_inside(node):
            if isinstance(node.target, ast.Name):
                self._record_live_in(node.target.id)
            else:
                self.visit(node.target)
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self.stored.add(node.target.id)
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.include_annotations:
            self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            self.visit(node.target)
        elif not isinstance(node.target, ast.Name):
            self.visit(node.target)

    def _visit_delete_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_delete_target(elt)
        elif isinstance(target, ast.Name) and isinstance(target.ctx, ast.Del):
            if self._is_inside(target):
                self._record_live_in(target.id)
                self.stored.discard(target.id)
        else:
            self.visit(target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._visit_delete_target(target)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        if not self._is_inside(node):
            self.generic_visit(node)
            return
        self._visit_type_alias_node(
            node, self.stored, self._record_live_in
        )


    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not node.values:
            return
        self.visit(node.values[0])
        stored_after_first = set(self.stored)
        for val in node.values[1:]:
            self.visit(val)
        self.stored = stored_after_first

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        stored_before = set(self.stored)
        self.visit(node.body)
        stored_body = set(self.stored)
        self.stored = set(stored_before)
        self.visit(node.orelse)
        stored_else = set(self.stored)
        self.stored = stored_body & stored_else

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if self._is_inside(node.target) and isinstance(node.target, ast.Name):
            self.stored.add(node.target.id)

    def _collect_branch_stored(
        self, stmts: Sequence[ast.stmt], initial: Set[str]
    ) -> Set[str]:
        self.stored = set(initial)
        for s in stmts:
            self.visit(s)
        return set(self.stored)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        stored_before = set(self.stored)
        stored_then = self._collect_branch_stored(node.body, stored_before)
        stored_else = self._collect_branch_stored(node.orelse, stored_before)

        if node.orelse:
            then_term = _block_terminates(node.body)
            else_term = _block_terminates(node.orelse)
            if then_term and not else_term:
                self.stored = stored_else
            elif else_term and not then_term:
                self.stored = stored_then
            elif then_term and else_term:
                self.stored = stored_before
            else:
                self.stored = stored_then & stored_else
        else:
            self.stored = stored_before

    def _visit_loop_body(
        self,
        test_or_iter: ast.AST,
        body: Sequence[ast.stmt],
        orelse: Sequence[ast.stmt],
        target: Optional[ast.AST] = None,
    ) -> None:
        self.visit(test_or_iter)
        stored_before = set(self.stored)
        if target is not None:
            self.visit(target)
        for s in body:
            self.visit(s)
        self.stored = set(stored_before)
        for s in orelse:
            self.visit(s)
        self.stored = set(stored_before)

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        self._visit_loop_body(node.iter, node.body, node.orelse, target=node.target)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop_body(node.test, node.body, node.orelse)

    def visit_Try(self, node: Union[ast.Try, ast.AST]) -> None:
        stored_before = set(self.stored)
        for s in getattr(node, "body", []):
            self.visit(s)
        stored_after_body = set(self.stored)
        for h in getattr(node, "handlers", []):
            self.stored = set(stored_before)
            h_type = getattr(h, "type", None)
            if h_type is not None:
                self.visit(h_type)
            name_val = getattr(h, "name", None)
            if isinstance(name_val, str) and self._is_inside(h):
                self.stored.add(name_val)
            for s in getattr(h, "body", []):
                self.visit(s)
        self.stored = set(stored_after_body)
        for s in getattr(node, "orelse", []):
            self.visit(s)
        self.stored = set(stored_before)
        for s in getattr(node, "finalbody", []):
            self.visit(s)
        self.stored = set(stored_before)

    def visit_TryStar(self, node: ast.AST) -> None:
        self.visit_Try(node)

    def visit_With(self, node: Union[ast.With, ast.AsyncWith]) -> None:
        stored_before = set(self.stored)
        for item in getattr(node, "items", []):
            self.visit(item.context_expr)
            if getattr(item, "optional_vars", None) is not None:
                self.visit(item.optional_vars)
        for s in getattr(node, "body", []):
            self.visit(s)
        self.stored = set(stored_before)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Match(self, node: ast.AST) -> None:
        self._traverse_isolated_cases(
            node,
            lambda: self.stored,
            lambda s: setattr(self, "stored", s),
        )

    def visit_MatchAs(self, node: ast.AST) -> None:
        if self._is_inside(node):
            self._record_pattern_binding(node, self.stored)
        else:
            self.generic_visit(node)


def _extract_unit_live_in_names(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int] = None,
    candidates: Optional[Set[str]] = None,
    u_start_col: Optional[int] = None,
) -> Set[str]:
    """Extracts variables read within unit bounds before any store, within the scope."""
    visitor = _UnitLiveInVisitor(
        u_start,
        u_end,
        u_end_col=u_end_col,
        candidates=candidates,
        u_start_col=u_start_col,
        scope_node=scope_node,
    )
    stmts = _extract_scope_stmts(scope_node)
    if stmts:
        for stmt in stmts:
            visitor.visit(stmt)
    else:
        visitor.visit(scope_node)
    return visitor.live_ins
