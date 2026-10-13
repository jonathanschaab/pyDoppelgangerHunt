"""AST scope and downstream read visitors for lexical dataflow analysis."""

from __future__ import annotations

import ast
import builtins
from enum import Enum
import logging
from typing import (
    Any,
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


class _BaseScopeVisitor(ast.NodeVisitor):
    """Shared helpers for scope visitors."""

    _record_import_names = staticmethod(_record_import_names)

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


    def _visit_nested_named_scope(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
        store_set: Set[str],
        reads_set: Set[str],
    ) -> None:
        store_set.add(node.name)
        for tp in getattr(node, "type_params", []):
            tp_name = getattr(tp, "name", None)
            if isinstance(tp_name, str):
                store_set.add(tp_name)
        reads_set.update(_extract_nested_scope_free_reads(node))

    def _visit_nested_lambda(
        self,
        node: ast.Lambda,
        reads_set: Set[str],
    ) -> None:
        reads_set.update(_extract_nested_scope_free_reads(node))

    def _visit_type_alias_stmt(self, node: ast.AST, store_set: Set[str]) -> None:
        name_node = getattr(node, "name", None)
        if isinstance(name_node, ast.Name):
            store_set.add(name_node.id)
        for tp in getattr(node, "type_params", []):
            tp_name = getattr(tp, "name", None)
            if isinstance(tp_name, str):
                store_set.add(tp_name)
        if hasattr(node, "value"):
            self.visit(getattr(node, "value"))

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

    def visit_TypeAlias(self, node: ast.AST) -> None:
        self._visit_type_alias_stmt(node, self.local_stores)


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

    def visit_TypeAlias(self, node: ast.AST) -> None:
        self._visit_type_alias_stmt(node, self.class_stores)


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

    def visit_TypeAlias(self, node: ast.AST) -> None:
        tp_names = {
            getattr(tp, "name", None)
            for tp in getattr(node, "type_params", [])
            if isinstance(getattr(tp, "name", None), str)
        }
        sub_visitor = _OuterExprVisitor()
        if hasattr(node, "value"):
            sub_visitor.visit(getattr(node, "value"))
        self.names.update(sub_visitor.names - tp_names)


def _extract_nested_scope_free_reads(
    node: Union[
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.GeneratorExp
    ],
    candidates: Optional[Set[str]] = None,
) -> Set[str]:
    """Extracts free variable references escaping a nested function, genexp, or class body."""
    if isinstance(node, ast.GeneratorExp):
        gen_visitor = _OuterExprVisitor()
        gen_visitor.visit(node)
        if candidates is not None:
            return gen_visitor.names & candidates
        return gen_visitor.names

    type_param_names: Set[str] = set()
    for tp in getattr(node, "type_params", []):
        tp_name = getattr(tp, "name", None)
        if isinstance(tp_name, str):
            type_param_names.add(tp_name)

    inner_type_params: Set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for sub in ast.walk(node):
            if sub is not node and hasattr(sub, "type_params"):
                for tp in getattr(sub, "type_params", []):
                    tp_name = getattr(tp, "name", None)
                    if isinstance(tp_name, str):
                        inner_type_params.add(tp_name)

    all_type_params = type_param_names | inner_type_params

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
        for tp_name in all_type_params:
            class_visitor.class_stores.add(tp_name)
        for stmt in node.body:
            class_visitor.visit(stmt)
        class_free = (
            (class_visitor.free_reads | class_visitor.nonlocals)
            - class_visitor.globals
            - all_type_params
        )
        class_reads = (outer_visitor.names - all_type_params) | class_free
        if candidates is not None:
            return class_reads & candidates
        return class_reads

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
    for tp_name in all_type_params:
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
    all_free = (outer_visitor.names - all_type_params) | (escaped - inner_type_params)
    if candidates is not None:
        return all_free & candidates
    return all_free


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


def _stmt_assigns_to_any_name(stmt: ast.AST, names: Set[str]) -> bool:
    """Checks whether an AST statement binds any variable in the given names set."""
    targets: List[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets.extend(stmt.targets)
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets.append(stmt.target)
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        targets.append(stmt.target)
    for t in targets:
        if any(name in names for name in _extract_assigned_names(t)):
            return True
    return False


class _VisitorPassMode(str, Enum):
    """Execution pass mode for downstream AST read analysis."""

    AFTER_UNIT = "after_unit"
    LOOP_CARRIED = "loop_carried"


def _is_ast_node_within_unit_bounds(
    node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int],
) -> bool:
    """Determines whether an AST node is positioned inside unit line and column bounds.

    When u_end_col is None, falls back to whole-line containment on u_end.
    """
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return False
    if lineno < u_start or lineno > u_end:
        return False
    if lineno == u_end and u_end_col is not None:
        col = getattr(node, "col_offset", 0)
        return col < int(u_end_col)
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
    ) -> None:
        super().__init__()
        self.u_start = u_start
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.candidates = candidates
        self.enclosing_loops = enclosing_loops or []
        self.pass_mode = (
            _VisitorPassMode(pass_mode)
            if isinstance(pass_mode, str)
            else pass_mode
        )
        self.loaded: Set[str] = set()
        self.killed: Set[str] = set()
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
                return getattr(node, "col_offset", 0) >= int(self.u_end_col)
            # When u_end_col cannot be resolved, fail-closed by treating
            # same-line nodes on u_end as downstream.
            return True
        return False

    def _is_node_inside_unit(self, node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node, self.u_start, self.u_end, self.u_end_col
        )

    def _is_loop_carried(self, node: ast.AST) -> bool:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if self._is_node_inside_unit(node) or self._is_node_after_unit(node):
            return False
        return any(l_start <= lineno <= l_end for l_start, l_end in self.enclosing_loops)

    def _should_inspect_read(self, node: ast.AST) -> bool:
        if self.pass_mode is _VisitorPassMode.LOOP_CARRIED:
            return self._is_loop_carried(node)
        return self._is_node_after_unit(node)

    def _record_killed_targets(self, targets: Iterable[ast.AST]) -> None:
        for t in targets:
            if self._should_inspect_read(t):
                lineno = getattr(t, "lineno", None)
                if lineno == self.u_end and self.u_end_col is None:
                    continue
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
        if self._should_inspect_read(node):
            free_reads = _extract_nested_scope_free_reads(node, candidates=self.candidates)
            for name in free_reads:
                self._record_downstream_read(name)
            name_attr = getattr(node, "name", None)
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
        if self._should_inspect_read(node):
            free_reads = _extract_nested_scope_free_reads(node, candidates=self.candidates)
            for name in free_reads:
                self._record_downstream_read(name)
            lineno = getattr(node, "lineno", None)
            if not self._is_in_comp(node.name) and not (
                lineno == self.u_end and self.u_end_col is None
            ):
                self.killed.add(node.name)
        else:
            n_start = getattr(node, "lineno", 0)
            n_end = getattr(node, "end_lineno", None) or n_start
            if n_start <= self.u_end <= n_end:
                for stmt in node.body:
                    self.visit(stmt)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_FunctionDef(node)

    def visit_TypeAlias(self, node: ast.AST) -> None:
        if self._should_inspect_read(node):
            tp_names = {
                getattr(tp, "name", None)
                for tp in getattr(node, "type_params", [])
                if isinstance(getattr(tp, "name", None), str)
            }
            if hasattr(node, "value"):
                sub_visitor = _OuterExprVisitor()
                sub_visitor.visit(getattr(node, "value"))
                for name in sub_visitor.names - tp_names:
                    self._record_downstream_read(name)
            name_node = getattr(node, "name", None)
            if isinstance(name_node, ast.Name) and not self._is_in_comp(name_node.id):
                self.killed.add(name_node.id)

    def visit_Name(self, node: ast.Name) -> None:
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and self._should_inspect_read(node):
            self._record_downstream_read(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and self._should_inspect_read(node.target):
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
        self.visit(node.iter)
        killed_before = set(self.killed)
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
                self._record_killed_targets([item.optional_vars])
        for stmt in node.body:
            self.visit(stmt)
        # Deliberately fail-closed: context managers may exit early or swallow exceptions
        self.killed = killed_before

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

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
        if bound is not None and self._should_inspect_read(node):
            if not self._is_in_comp(bound):
                self.killed.add(bound)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._should_inspect_read(node):
            self._record_import_names(node, self.killed)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._should_inspect_read(node):
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
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if lineno < self.u_end:
            return True
        if lineno == self.u_end:
            if self.u_end_col is not None:
                col = getattr(node, "col_offset", None)
                return col is not None and col < int(self.u_end_col)
            return True
        return False


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
    ) -> None:
        super().__init__()
        self.u_start = u_start
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.candidates = candidates
        self.live_ins: Set[str] = set()
        self.stored: Set[str] = set()

    def _is_inside(self, node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node, self.u_start, self.u_end, self.u_end_col
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
    ) -> None:
        for free in _extract_nested_scope_free_reads(
            node, candidates=self.candidates
        ):
            self._record_live_in(free)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if self._is_inside(node):
            self._record_nested_free_reads(node)
        else:
            self.generic_visit(node)

    def visit_FunctionDef(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        if self._is_inside(node):
            self.stored.add(node.name)
            self._record_nested_free_reads(node)
        else:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._is_inside(node):
            self.stored.add(node.name)
            self._record_nested_free_reads(node)
        else:
            self.generic_visit(node)

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
        if self._is_inside(node):
            self.visit(node.value)
            for t in node.targets:
                self.visit(t)
        else:
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._is_inside(node):
            self.generic_visit(node)
            return
        if node.value is None:
            return
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.stored.add(node.target.id)
        else:
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if self._is_inside(node):
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self.stored.add(node.target.id)
        else:
            self.generic_visit(node)

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

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        if self._is_inside(node):
            self.visit(node.iter)
            self.visit(node.target)
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)
        else:
            self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)


def _extract_unit_live_in_names(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int] = None,
    candidates: Optional[Set[str]] = None,
) -> Set[str]:
    """Extracts variables read within unit bounds before any store, within the scope."""
    visitor = _UnitLiveInVisitor(
        u_start, u_end, u_end_col=u_end_col, candidates=candidates
    )
    visitor.visit(scope_node)
    return visitor.live_ins
