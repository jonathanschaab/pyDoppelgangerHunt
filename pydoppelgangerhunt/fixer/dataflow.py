"""AST dataflow visitors, downstream read analysis, and output pairing."""

from __future__ import annotations

import ast
import builtins
from collections import OrderedDict
from enum import Enum
import hashlib
import logging
import threading
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.fixer.source import (
    _find_innermost_enclosing_node,
    _resolve_safe_unit_file_path,
    parse_unit_coord,
)

logger = logging.getLogger(__name__)

_BUILTIN_NAMES: Set[str] = set(dir(builtins))

__all__ = [
    "GeneratorCloneSideData",
    "collect_downstream_read_names",
    "is_async_generator_with_return_value",
    "is_subroutine_unit",
    "resolve_clone_generator_subroutine_outputs",
    "resolve_closure_strictness_mode",
    "resolve_generator_subroutine_outputs",
]


def _load_unit_file_text(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Reads full source text from memory or disk for downstream AST read analysis.

    For in-memory units with 'source_lines', note that when 'source_lines_is_sliced' is
    unspecified (None), len(unit["source_lines"]) must strictly exceed end line e_d,
    or reach EOF with s_d > 1, to prevent mistaking sliced excerpts as complete files.
    When a unit reaches EOF starting at line 1 (e_d == len(source_lines) and s_d == 1),
    callers providing in-memory source_lines without a disk backing file should
    explicitly specify source_lines_is_sliced=False.
    """
    is_sliced = unit.get("source_lines_is_sliced")
    if is_sliced is True:
        return None

    source_text = unit.get("source_text")
    if source_text is None:
        source_text = unit.get("file_source")
    if source_text is not None:
        return str(source_text)

    s_d = max(1, parse_unit_coord(unit, "start", default=1))
    e_d = max(s_d, parse_unit_coord(unit, "end", default=s_d))

    if "source_lines" in unit and isinstance(unit["source_lines"], (list, tuple)):
        is_full_file = (
            is_sliced is False
            or len(unit["source_lines"]) > e_d
            or (len(unit["source_lines"]) == e_d and s_d > 1)
        )
        if is_full_file:
            return "".join(
                ln if ln.endswith("\n") else ln + "\n"
                for ln in unit["source_lines"]
            )

    # Disk fallback: strictly only .py files within repo_root or CWD are accepted.
    # Notebooks (.ipynb) and out-of-root files fail closed and return None.
    resolved_file = _resolve_safe_unit_file_path(
        unit, repo_root=repo_root, allowed_suffixes=(".py",)
    )
    if resolved_file is not None:
        try:
            with open(resolved_file, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            pass

    return None


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
    try:
        u_start = parse_unit_coord(unit, "start", default=0)
        u_end = parse_unit_coord(unit, "end", default=u_start)
    except (ValueError, TypeError):
        return None
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None
    return u_start, u_end


def _extract_first_unit_coord(
    unit: Dict[str, Any], keys: Sequence[str]
) -> Optional[int]:
    """Returns the first successfully parsed coordinate among candidate keys."""
    for key in keys:
        if key in unit and unit.get(key) is not None:
            try:
                val = parse_unit_coord(unit, key, default=None)
                if val is not None:
                    return val
            except (ValueError, TypeError):
                pass
    return None


def is_subroutine_unit(unit: Dict[str, Any]) -> bool:
    """Checks whether an AST code unit is a subroutine block rather than a whole function.

    Classification Rules:
    1. Known subroutine kinds ('compound_block', 'sliding_window', 'clause_branch') return True.
    2. Whole-callable or expression kinds ('function', 'closure', 'method', 'comprehension',
       'complex_expr') return False.
    3. Name heuristic fallback: when 'kind' is unspecified or unrecognized, units whose 'name'
       contains ':' (e.g. 'fn:for#1' or 'process:if') are classified as subroutine blocks.
    """
    unit_kind = str(unit.get("kind") or "")
    if unit_kind in ("compound_block", "sliding_window", "clause_branch"):
        return True
    if unit_kind in ("function", "closure", "method", "comprehension", "complex_expr"):
        return False
    return ":" in str(unit.get("name") or "")


def _extract_unit_end_col(unit: Dict[str, Any]) -> Optional[int]:
    """Extracts end column offset from unit dictionary if present."""
    return _extract_first_unit_coord(unit, ("end_col_offset", "end_col"))


def _resolve_unit_ast_end_col(
    scope_node: ast.AST, unit: Dict[str, Any]
) -> Optional[int]:
    """Resolves end column offset from statement boundaries within the enclosing scope."""
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None

    parent_map: Dict[ast.AST, ast.AST] = {}
    matching_stmts: List[ast.stmt] = []
    for parent in ast.walk(scope_node):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
        if (
            isinstance(parent, ast.stmt)
            and getattr(parent, "end_lineno", getattr(parent, "lineno", 0)) == u_end
        ):
            matching_stmts.append(parent)

    inner_stmts: List[ast.stmt] = []
    if matching_stmts:
        matching_set = set(matching_stmts)
        non_leaf_stmts: Set[ast.stmt] = set()
        for s in matching_stmts:
            curr = parent_map.get(s)
            while curr is not None:
                if isinstance(curr, ast.stmt) and curr in matching_set:
                    if curr in non_leaf_stmts:
                        break
                    non_leaf_stmts.add(curr)
                curr = parent_map.get(curr)

        inner_stmts = [s for s in matching_stmts if s not in non_leaf_stmts]
    if not inner_stmts:
        logger.debug(
            "No statement ending at line %d found in scope; falling back to whole-line",
            u_end,
        )
        return None

    inner_stmts.sort(key=lambda s: getattr(s, "col_offset", 0))
    if u_start == u_end:
        start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
        if start_col is not None:
            matched = [s for s in inner_stmts if getattr(s, "col_offset", 0) >= start_col]
            target_stmt = matched[0] if matched else inner_stmts[0]
        elif is_subroutine_unit(unit):
            # For single-line compound blocks or sliding windows without explicit start_col,
            # encompass the full body of the compound statement through its trailing child.
            target_stmt = inner_stmts[-1]
        else:
            # When start_col is omitted for a single-line standalone unit, default to the first
            # statement on the line (e.g. 'yield x; print(total)'). Note that if a unit
            # represents a subsequent statement on a semicolon-separated line, supplying
            # 'start_col' during harvesting ensures exact boundary matching.
            target_stmt = inner_stmts[0]
    else:
        # For multi-line units (u_start != u_end), default to the last statement ending
        # on line u_end (inner_stmts[-1]). In the rare event of semicolon-separated statements
        # on the closing line of a block where the unit terminates earlier, providing
        # 'end_col' during harvesting ensures exact sub-line boundary resolution.
        target_stmt = inner_stmts[-1]

    end_col = getattr(target_stmt, "end_col_offset", None)
    if end_col is None:
        logger.debug(
            "Statement ending at line %d has no end_col_offset; falling back to whole-line",
            u_end,
        )
        return None
    return int(end_col)


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


def _extract_nested_scope_free_reads(
    node: Union[
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.GeneratorExp
    ],
) -> Set[str]:
    """Extracts free variable references escaping a nested function, genexp, or class body."""
    if isinstance(node, ast.GeneratorExp):
        gen_visitor = _OuterExprVisitor()
        gen_visitor.visit(node)
        return gen_visitor.names - _BUILTIN_NAMES

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
        class_free = (
            (class_visitor.free_reads | class_visitor.nonlocals)
            - class_visitor.globals
            - type_param_names
        )
        return (outer_visitor.names - type_param_names) | class_free

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


class _VisitorPassMode(str, Enum):
    """Execution pass mode for downstream AST read analysis."""

    AFTER_UNIT = "after_unit"
    LOOP_CARRIED = "loop_carried"


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

    def _is_node_inside_unit(self, node: ast.AST) -> bool:
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return False
        if self.u_start < lineno < self.u_end:
            return True
        if lineno == self.u_start == self.u_end:
            col = getattr(node, "col_offset", 0)
            return self.u_end_col is None or col < int(self.u_end_col)
        if lineno == self.u_start:
            return True
        if lineno == self.u_end:
            col = getattr(node, "col_offset", 0)
            return self.u_end_col is None or col < int(self.u_end_col)
        return False

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
            free_reads = _extract_nested_scope_free_reads(node)
            for name in free_reads:
                self._record_downstream_read(name)
            name_attr = getattr(node, "name", None)
            if name_attr and not self._is_in_comp(name_attr):
                self.killed.add(name_attr)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self._should_inspect_read(node):
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
        if isinstance(node.ctx, ast.Load) and self._should_inspect_read(node):
            self._record_downstream_read(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and self._should_inspect_read(node.target):
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
            # Discarding handler.name leaves the variable in whatever killed state it held
            # prior to the handler. This is conservatively safe: if a prior unit defined 'e',
            # subsequent reads of 'e' after the try/except continue to be treated as requiring
            # the unit's output rather than pruning it.
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


def _collect_pre_unit_closures(
    scope_node: ast.AST,
    u_start: int,
) -> Set[str]:
    """Discovers free variables captured by functions, class methods, and lambdas defined prior
    to the unit in the same lexical scope.

    Scope Boundary:
    This inspects closures defined before the unit in the same lexical scope to ensure fail-closed
    safety for escaping closures (e.g. callbacks registered prior to unit execution). Closures
    defined downstream of the unit are inspected by the normal downstream AST traversal; closures
    defined in sibling scopes outside the current enclosing scope (e.g. sibling methods in a
    class) are not inspected.

    Line-Granularity Limitation:
    Statements and subnodes are filtered by line number (strictly less than `u_start`). A closure
    defined on the exact start line of the unit (such as `cb = lambda: total; for x in ...` on a
    single line) begins at `lineno == u_start` and is therefore not treated as pre-unit.

    Over-Capture Trade-off:
    `ast.walk` over compound statements prior to `u_start` also traverses lambdas inside nested
    functions. Free variables of such lambdas that reference intermediate nested locals are
    unioned into captured reads, potentially causing benign over-rejection. This fail-closed
    behavior is intentional.
    """
    captured_reads: Set[str] = set()

    def _extract_header_nodes(s: ast.stmt) -> List[ast.AST]:
        if isinstance(s, ast.If):
            return [s.test]
        if isinstance(s, (ast.For, ast.AsyncFor)):
            return [s.target, s.iter]
        if isinstance(s, ast.While):
            return [s.test]
        if isinstance(s, (ast.With, ast.AsyncWith)):
            headers: List[ast.AST] = []
            for item in s.items:
                headers.append(item.context_expr)
                if item.optional_vars is not None:
                    headers.append(item.optional_vars)
            return headers
        if isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            handlers = getattr(s, "handlers", [])
            return [h.type for h in handlers if getattr(h, "type", None) is not None]
        if type(s).__name__ == "Match":
            headers = [getattr(s, "subject")]
            for case in getattr(s, "cases", []):
                headers.append(getattr(case, "pattern"))
                if getattr(case, "guard", None) is not None:
                    headers.append(getattr(case, "guard"))
            return headers
        return [s]

    def _walk_stmts(stmts: Iterable[ast.stmt]) -> None:
        for stmt in stmts:
            stmt_start = getattr(stmt, "lineno", 0)
            decorators = getattr(stmt, "decorator_list", [])
            if decorators:
                dec_start = min(getattr(d, "lineno", stmt_start) for d in decorators)
                stmt_start = min(stmt_start, dec_start)

            if stmt_start >= u_start:
                continue

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                captured_reads.update(_extract_nested_scope_free_reads(stmt))
                continue

            if isinstance(stmt, ast.ClassDef):
                for item in ast.walk(stmt):
                    if item is not stmt and isinstance(
                        item,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.Lambda,
                            ast.GeneratorExp,
                        ),
                    ):
                        item_start = getattr(item, "lineno", stmt_start)
                        if item_start < u_start:
                            captured_reads.update(_extract_nested_scope_free_reads(item))
                continue

            for hnode in _extract_header_nodes(stmt):
                for subnode in ast.walk(hnode):
                    if isinstance(subnode, (ast.Lambda, ast.GeneratorExp)):
                        sub_lineno = getattr(subnode, "lineno", stmt_start)
                        if sub_lineno < u_start:
                            captured_reads.update(_extract_nested_scope_free_reads(subnode))

            for attr in ("body", "orelse", "finalbody"):
                sub_stmts = getattr(stmt, attr, None)
                if isinstance(sub_stmts, list):
                    _walk_stmts(sub_stmts)

            for handler in getattr(stmt, "handlers", []):
                _walk_stmts(getattr(handler, "body", []))

            for case in getattr(stmt, "cases", []):
                _walk_stmts(getattr(case, "body", []))

    _walk_stmts(getattr(scope_node, "body", []))
    return captured_reads


_MAX_DOWNSTREAM_CACHE_SIZE: int = 1024
_downstream_reads_cache: OrderedDict[Tuple[Any, ...], frozenset[str]] = OrderedDict()
_downstream_cache_lock: threading.Lock = threading.Lock()


def _clear_downstream_reads_cache() -> None:
    """Clears the downstream read memoization cache."""
    with _downstream_cache_lock:
        _downstream_reads_cache.clear()


def _build_downstream_cache_key(
    source_text: str,
    unit: Dict[str, Any],
    u_start: int,
    u_end: int,
    candidates: Optional[Set[str]],
    skip_pre_unit_closures: bool,
    source_digest: Optional[str] = None,
) -> Tuple[Any, ...]:
    """Forms a persistent cache key for downstream AST read analysis."""
    file_path_str = str(unit.get("file") or "")
    mtime = unit.get("mtime") or unit.get("timestamp")
    digest = source_digest or unit.get("source_digest") or unit.get("content_digest")
    if digest is not None:
        content_digest = str(digest)[:16]
    else:
        content_digest = hashlib.sha256(
            source_text.encode("utf-8", errors="replace")
        ).hexdigest()[:16]
    start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
    end_col = _extract_unit_end_col(unit)
    cands_key = frozenset(candidates) if candidates is not None else None
    unit_kind = unit.get("kind")
    unit_name = unit.get("name")
    return (
        content_digest,
        mtime,
        file_path_str,
        u_start,
        u_end,
        start_col,
        end_col,
        unit_kind,
        unit_name,
        cands_key,
        bool(skip_pre_unit_closures),
    )


def _resolve_downstream_scope_node(
    source_text: str,
    unit: Dict[str, Any],
    tree: Optional[ast.AST],
) -> Optional[ast.AST]:
    """Resolves the innermost enclosing function AST node or falls back to module scope."""
    enc = _find_innermost_enclosing_node(
        source_text, unit, (ast.FunctionDef, ast.AsyncFunctionDef), tree=tree
    )
    if enc is not None:
        return enc[0]
    return _parse_source_tree(source_text, tree=tree)


def _find_enclosing_loops(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
) -> List[Tuple[int, int]]:
    """Detects loops enclosing the unit boundaries for loop-carried dependence analysis."""
    loops: List[Tuple[int, int]] = []
    for node in ast.walk(scope_node):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            l_start = getattr(node, "lineno", 0)
            l_end = getattr(node, "end_lineno", None) or l_start
            if l_start <= u_start and u_end <= l_end:
                if (l_start, l_end) != (u_start, u_end):
                    loops.append((l_start, l_end))
    return loops


def _stmt_definitely_terminates(stmt: ast.stmt) -> bool:
    """Checks whether an AST statement unconditionally exits execution."""
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, ast.If):
        return (
            bool(stmt.orelse)
            and _block_definitely_terminates(stmt.body)
            and _block_definitely_terminates(stmt.orelse)
        )
    if isinstance(stmt, ast.Try):
        if stmt.finalbody and _block_definitely_terminates(stmt.finalbody):
            return True
        if stmt.handlers and _block_definitely_terminates(stmt.body):
            return all(_block_definitely_terminates(h.body) for h in stmt.handlers)
    if type(stmt).__name__ == "TryStar":
        finalbody = getattr(stmt, "finalbody", None)
        if finalbody and _block_definitely_terminates(finalbody):
            return True
        handlers = getattr(stmt, "handlers", [])
        body = getattr(stmt, "body", [])
        if handlers and _block_definitely_terminates(body):
            return all(_block_definitely_terminates(h.body) for h in handlers)
    return False


def _block_definitely_terminates(stmts: Sequence[ast.stmt]) -> bool:
    """Checks whether any statement in a sequence unconditionally terminates execution."""
    return any(_stmt_definitely_terminates(s) for s in stmts)


def _enclosing_try_reads_outputs(
    scope_node: Optional[ast.AST],
    u_start: int,
    u_end: int,
    outputs: Set[str],
) -> bool:
    """Returns True if unit is inside an enclosing try whose handlers/finally read outputs,
    or whose handlers can fall through and outputs are read downstream of the try."""
    if scope_node is None or not outputs:
        return False
    for node in ast.walk(scope_node):
        if isinstance(node, ast.Try) or type(node).__name__ == "TryStar":
            body_nodes = getattr(node, "body", [])
            if not body_nodes:
                continue
            b_start = getattr(body_nodes[0], "lineno", 0)
            b_end_raw = getattr(body_nodes[-1], "end_lineno", None)
            b_end: int = (
                b_end_raw
                if isinstance(b_end_raw, int)
                else int(getattr(body_nodes[-1], "lineno", b_start) or b_start)
            )
            if b_start <= u_start and u_end <= b_end:
                cleanup_nodes: List[ast.AST] = list(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    cleanup_nodes.extend(getattr(handler, "body", []))
                    if getattr(handler, "type", None):
                        cleanup_nodes.append(handler.type)
                for c_node in cleanup_nodes:
                    for sub in ast.walk(c_node):
                        if (
                            isinstance(sub, ast.Name)
                            and isinstance(sub.ctx, ast.Load)
                            and sub.id in outputs
                        ):
                            return True

                # If any handler can fall through (e.g. swallows GeneratorExit or exceptions
                # without re-raising or returning), check if any needed output is read
                # downstream of the enclosing try block.
                handlers = getattr(node, "handlers", [])
                if handlers and any(
                    not _block_definitely_terminates(h.body) for h in handlers
                ):
                    try_end_raw = getattr(node, "end_lineno", None)
                    t_end = try_end_raw if isinstance(try_end_raw, int) else b_end
                    post_try_visitor = _DownstreamReadVisitor(
                        t_end,
                        t_end,
                        None,
                        candidates=outputs,
                        pass_mode=_VisitorPassMode.AFTER_UNIT,
                    )
                    for stmt in getattr(scope_node, "body", []):
                        post_try_visitor.visit(stmt)
                    if post_try_visitor.loaded & outputs:
                        return True
    return False


def _merge_pre_unit_closure_reads(
    loaded: Set[str],
    pre_unit_captured_reads: Set[str],
    candidates: Optional[Set[str]],
) -> None:
    """Merges escaping pre-unit closure free reads into live downstream names fail-closed."""
    if candidates is not None:
        effective_captured = pre_unit_captured_reads & candidates
        if effective_captured:
            logger.debug(
                "Captured pre-unit closure candidate reads (fail-closed callback protection): %s",
                sorted(effective_captured),
            )
        loaded.update(effective_captured)
    else:
        if pre_unit_captured_reads:
            logger.debug(
                "Captured pre-unit closure reads (fail-closed callback protection): %s",
                sorted(pre_unit_captured_reads),
            )
        loaded.update(pre_unit_captured_reads)


def collect_downstream_read_names(
    source_text: str,
    unit: Dict[str, Any],
    candidates: Optional[Set[str]] = None,
    *,
    tree: Optional[ast.AST] = None,
    skip_pre_unit_closures: bool = False,
    source_digest: Optional[str] = None,
) -> Optional[Set[str]]:
    """Identifies variable names loaded downstream of a unit within its lexical execution scope.

    Module-Level Fallback:
    When a unit is defined at top-level module scope (having no enclosing FunctionDef),
    `scope_node` defaults to the module AST. All top-level function and class definitions
    prior to the unit in the module contribute their free global reads into the live candidate
    set. If an output candidate matches one of these captured names without a definite store
    beforehand, the generator subroutine will fail the definiteness check and fail closed
    (skipped during refactoring patch synthesis with a diagnostic DEBUG log).
    """
    # Invariant: downstream live names originate from three distinct sources:
    # (1) post-unit statements, (2) enclosing loop-carried paths, (3) escaping pre-unit closures.
    bounds = _get_valid_unit_bounds(unit)
    if bounds is None:
        return None
    u_start, u_end = bounds

    # Fast-path early out: if candidate set is explicitly empty, no outputs can match
    if candidates is not None and not candidates:
        return set()

    cache_key = _build_downstream_cache_key(
        source_text,
        unit,
        u_start,
        u_end,
        candidates,
        skip_pre_unit_closures,
        source_digest=source_digest,
    )
    with _downstream_cache_lock:
        if cache_key in _downstream_reads_cache:
            cached = _downstream_reads_cache[cache_key]
            _downstream_reads_cache.move_to_end(cache_key)
            return set(cached)

    scope_node = _resolve_downstream_scope_node(source_text, unit, tree)
    if scope_node is None:
        return None

    u_end_col = _extract_unit_end_col(unit)
    if u_end_col is None:
        u_end_col = _resolve_unit_ast_end_col(scope_node, unit)

    enclosing_loops = _find_enclosing_loops(scope_node, u_start, u_end)

    if skip_pre_unit_closures:
        pre_unit_captured_reads: Set[str] = set()
    else:
        pre_unit_captured_reads = _collect_pre_unit_closures(scope_node, u_start)

    visitor = _DownstreamReadVisitor(
        u_start,
        u_end,
        u_end_col,
        candidates=candidates,
        enclosing_loops=enclosing_loops,
        pass_mode=_VisitorPassMode.AFTER_UNIT,
    )
    for stmt in getattr(scope_node, "body", []):
        visitor.visit(stmt)

    loaded = set(visitor.loaded)
    if enclosing_loops:
        loop_visitor = _DownstreamReadVisitor(
            u_start,
            u_end,
            u_end_col,
            candidates=candidates,
            enclosing_loops=enclosing_loops,
            pass_mode=_VisitorPassMode.LOOP_CARRIED,
        )
        for stmt in getattr(scope_node, "body", []):
            loop_visitor.visit(stmt)
        loaded.update(loop_visitor.loaded)

    _merge_pre_unit_closure_reads(loaded, pre_unit_captured_reads, candidates)

    with _downstream_cache_lock:
        if (
            cache_key not in _downstream_reads_cache
            and len(_downstream_reads_cache) >= _MAX_DOWNSTREAM_CACHE_SIZE
        ):
            _downstream_reads_cache.popitem(last=False)
        _downstream_reads_cache[cache_key] = frozenset(loaded)
        _downstream_reads_cache.move_to_end(cache_key)

    return loaded


def _pair_clone_outputs(
    u1_outs: Sequence[str],
    u2_outs: Sequence[str],
) -> List[Tuple[str, str]]:
    """Establishes an ordered 1-to-1 mapping between clone output variables."""
    u1_dedup = list(dict.fromkeys(u1_outs))
    u2_dedup = list(dict.fromkeys(u2_outs))
    if len(u1_dedup) != len(u2_dedup):
        logger.debug(
            "Rejecting clone output pairing: output arities differ (%d vs %d: %s vs %s)",
            len(u1_dedup),
            len(u2_dedup),
            u1_dedup,
            u2_dedup,
        )
        return []
    common = set(u1_dedup) & set(u2_dedup)
    # Fail closed if any common variable occupies different positional indices
    if any(u1_dedup.index(name) != u2_dedup.index(name) for name in common):
        logger.debug(
            "Rejecting clone output pairing: common variables have conflicting "
            "positional orderings (%s vs %s)",
            u1_dedup,
            u2_dedup,
        )
        return []
    # True 1-to-1 positional pairing
    return list(zip(u1_dedup, u2_dedup))


def _filter_valid_output_pairs(
    pairs: Sequence[Tuple[str, str]],
    u1_allowed: Optional[Set[str]],
    u2_allowed: Optional[Set[str]],
    *,
    reason: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """Validates output pairs against allowed bindings and returns separated output lists."""
    for side_label, allowed_set, col_idx in (("u1", u1_allowed, 0), ("u2", u2_allowed, 1)):
        if allowed_set is not None:
            missing = [p[col_idx] for p in pairs if p[col_idx] not in allowed_set]
            if missing:
                logger.debug(
                    "Rejecting generator subroutine outputs: %s output %s: %s",
                    side_label,
                    reason,
                    missing,
                )
                return None
    return [p[0] for p in pairs], [p[1] for p in pairs]


class GeneratorCloneSideData(NamedTuple):
    """Encapsulates per-side output candidates, downstream reads, and definite stores."""

    outputs: Sequence[str]
    downstream: Optional[Set[str]] = None
    definite: Optional[Set[str]] = None


def resolve_generator_subroutine_outputs(
    side1: GeneratorCloneSideData,
    side2: GeneratorCloneSideData,
) -> Optional[Tuple[List[str], List[str]]]:
    """Selects output variables needed downstream by either clone side, preserving
    per-side necessity."""
    outs1 = list(side1.outputs)
    outs2 = list(side2.outputs)
    d1_raw = side1.downstream
    d2_raw = side2.downstream
    u1_def = side1.definite
    u2_def = side2.definite

    d1_needed = d1_raw if d1_raw is not None else set(outs1)
    d2_needed = d2_raw if d2_raw is not None else set(outs2)

    needed1 = [o for o in outs1 if o in d1_needed]
    needed2 = [o for o in outs2 if o in d2_needed]

    if not needed1 and not needed2:
        return [], []

    pairs = _pair_clone_outputs(outs1, outs2)
    paired_u1 = {o1 for o1, _ in pairs}
    paired_u2 = {o2 for _, o2 in pairs}

    # Fail closed: every required downstream output must have a paired counterpart
    if any(o1 not in paired_u1 for o1 in needed1) or any(o2 not in paired_u2 for o2 in needed2):
        logger.debug(
            "Rejecting generator subroutine outputs: required downstream outputs "
            "could not be paired (needed1=%s, needed2=%s, pairs=%s)",
            needed1,
            needed2,
            pairs,
        )
        return None

    kept_pairs = [
        (o1, o2)
        for o1, o2 in pairs
        if o1 in d1_needed or o2 in d2_needed
    ]

    # Fail closed on non-definitely assigned outputs to prevent UnboundLocalError
    return _filter_valid_output_pairs(
        kept_pairs,
        u1_def,
        u2_def,
        reason="lacks definite store before exit, risking UnboundLocalError",
    )


def is_async_generator_with_return_value(
    *scopes: Optional[Dict[str, Any]],
    has_outputs: bool = False,
) -> bool:
    """Checks whether scopes represent an async generator attempting to return values or outputs."""
    has_yield = any(bool(s.get("has_yield")) for s in scopes if s is not None)
    is_async = any(bool(s.get("is_async")) for s in scopes if s is not None)
    has_ret = has_outputs or any(
        bool(s.get("has_return_value")) for s in scopes if s is not None
    )
    return has_yield and is_async and has_ret


def _extract_effective_unit_outputs(
    unit: Dict[str, Any], scope: Dict[str, Any]
) -> List[str]:
    """Extracts non-global, non-local outputs from precomputed unit dict or analyzed scope."""
    raw = unit.get("outputs")
    if isinstance(raw, set):
        cands: Sequence[str] = sorted(raw)
    elif isinstance(raw, (list, tuple)):
        cands = list(raw)
    else:
        cands = scope.get("outputs", [])
    excluded = set(scope.get("globals", [])) | set(scope.get("nonlocals", []))
    return [v for v in cands if v not in excluded]


_warned_closure_strictness_values: Set[str] = set()


def resolve_closure_strictness_mode(
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: bool = False,
) -> Tuple[str, bool]:
    """Resolves canonical closure strictness mode ('strict' or 'lenient') and boolean skip flag.

    Precedence:
    Explicit closure_strictness takes precedence over the boolean flag. Permissive aliases
    ('lenient', 'fast', 'skip') map to ('lenient', True), while conservative aliases
    ('strict', 'fail_closed') map to ('strict', False). If unspecified or unrecognized,
    falls back to skip_pre_unit_closures.
    """
    if closure_strictness is not None:
        c_mode = str(closure_strictness).strip().lower()
        if c_mode in ("lenient", "fast", "skip"):
            return "lenient", True
        if c_mode in ("strict", "fail_closed"):
            return "strict", False
        fallback = "lenient" if skip_pre_unit_closures else "strict"
        raw_key = str(closure_strictness)
        if raw_key not in _warned_closure_strictness_values:
            _warned_closure_strictness_values.add(raw_key)
            logger.warning(
                "Unrecognized closure_strictness '%s'; falling back to %s mode",
                closure_strictness,
                fallback,
            )
    is_lenient = bool(skip_pre_unit_closures)
    return ("lenient" if is_lenient else "strict"), is_lenient


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
    skip_pre_unit_closures: bool = False,
    source_digest1: Optional[str] = None,
    source_digest2: Optional[str] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Resolves output variable bindings for clone generator subroutines across two code units.

    Ensures necessity by analyzing downstream reads and enforces definite assignment
    before unit exit to prevent runtime UnboundLocalError at exhaustion.
    """
    is_precomputed = (
        isinstance(u1.get("outputs"), (list, tuple, set))
        and isinstance(u2.get("outputs"), (list, tuple, set))
    )
    if is_precomputed:
        outs1 = list(dict.fromkeys(_extract_effective_unit_outputs(u1, scope1)))
        outs2 = list(dict.fromkeys(_extract_effective_unit_outputs(u2, scope2)))
        if len(outs1) != len(outs2):
            return None
    else:
        outs1 = _extract_effective_unit_outputs(u1, scope1)
        outs2 = _extract_effective_unit_outputs(u2, scope2)

    u1_def: Optional[Set[str]] = None
    if "definite_stores" in scope1 or "inputs" in scope1:
        u1_def = set(scope1.get("definite_stores", [])) | set(scope1.get("inputs", []))
    u2_def: Optional[Set[str]] = None
    if "definite_stores" in scope2 or "inputs" in scope2:
        u2_def = set(scope2.get("definite_stores", [])) | set(scope2.get("inputs", []))

    cands1 = set(outs1)
    cands2 = set(outs2)

    targets = [
        (u1, cands1, source_text1, tree1, source_digest1),
        (u2, cands2, source_text2, tree2, source_digest2),
    ]
    target_srcs: List[Optional[str]] = []
    downstream_reads: List[Optional[Set[str]]] = []
    for u_item, cands, s_text, tr, s_digest in targets:
        src = (
            s_text
            if s_text is not None
            else _load_unit_file_text(u_item, repo_root=repo_root)
        )
        target_srcs.append(src)
        if src is not None:
            reads = collect_downstream_read_names(
                src,
                u_item,
                candidates=cands,
                tree=tr,
                skip_pre_unit_closures=skip_pre_unit_closures,
                source_digest=s_digest,
            )
        else:
            reads = None
        downstream_reads.append(reads)

    side1 = GeneratorCloneSideData(outs1, downstream_reads[0], u1_def)
    side2 = GeneratorCloneSideData(outs2, downstream_reads[1], u2_def)
    resolved = resolve_generator_subroutine_outputs(side1, side2)
    if resolved is None:
        return None

    out1, out2 = resolved
    for side_idx, (u_item, needed_outs, s_text, tr) in enumerate([
        (u1, out1, target_srcs[0], tree1),
        (u2, out2, target_srcs[1], tree2),
    ]):
        if needed_outs and s_text is not None:
            scope_node = _resolve_downstream_scope_node(s_text, u_item, tr)
            u_s = max(1, parse_unit_coord(u_item, "start", default=1))
            u_e = max(u_s, parse_unit_coord(u_item, "end", default=u_s))
            if _enclosing_try_reads_outputs(scope_node, u_s, u_e, set(needed_outs)):
                logger.debug(
                    "Rejecting generator subroutine pair: needed output(s) read in "
                    "enclosing try cleanup for unit %s",
                    side_idx + 1,
                )
                return None

    return resolved
