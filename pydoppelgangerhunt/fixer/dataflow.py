"""AST dataflow visitors, downstream read analysis, and output pairing."""

from __future__ import annotations

import ast
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import normalize_path_string
from pydoppelgangerhunt.fixer.source import parse_unit_coord

logger = logging.getLogger(__name__)


def _load_unit_file_text(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Reads source text from disk or unit dictionary for a unit."""
    source_text = unit.get("source_text")
    if source_text is None:
        source_text = unit.get("file_source")
    if source_text is not None:
        return str(source_text)
    f_raw = normalize_path_string(str(unit.get("file") or ""), strip_anchor=True)
    if f_raw:
        p = Path(f_raw)
        if p.is_absolute():
            f_path = p
        elif repo_root:
            cand = Path(repo_root) / p
            f_path = cand if cand.is_file() else p
        else:
            f_path = p
        if f_path.is_file():
            try:
                return f_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass
    if "source_lines" in unit and isinstance(unit["source_lines"], (list, tuple)):
        return "".join(unit["source_lines"])
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
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
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
        else:
            # When start_col is omitted for a single-line unit, default to the first
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


class _DownstreamReadVisitor(_BaseScopeVisitor):
    """Walks AST statements in a lexical scope tracking direct, nested, and loop-carried reads."""

    def __init__(
        self,
        u_start: int,
        u_end: int,
        u_end_col: Optional[int],
        candidates: Optional[Set[str]] = None,
        enclosing_loops: Optional[List[Tuple[int, int]]] = None,
        pass_mode: str = "after_unit",
    ) -> None:
        super().__init__()
        self.u_start = u_start
        self.u_end = u_end
        self.u_end_col = u_end_col
        self.candidates = candidates
        self.enclosing_loops = enclosing_loops or []
        self.pass_mode = pass_mode
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
        if self.pass_mode == "loop_carried":
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
        """Records variables targeted for deletion as downstream reads if within unit boundary."""
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
                        item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                    ):
                        item_start = getattr(item, "lineno", stmt_start)
                        if item_start < u_start:
                            captured_reads.update(_extract_nested_scope_free_reads(item))
                continue

            for subnode in ast.walk(stmt):
                if isinstance(subnode, ast.Lambda):
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


def _clear_downstream_reads_cache() -> None:
    """Clears the downstream read memoization cache."""
    _downstream_reads_cache.clear()


def collect_downstream_read_names(
    source_text: str,
    unit: Dict[str, Any],
    candidates: Optional[Set[str]] = None,
    *,
    tree: Optional[ast.AST] = None,
    skip_pre_unit_closures: bool = False,
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
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    if u_start <= 0 or u_end <= 0 or u_start > u_end:
        return None

    # Fast-path early out: if candidate set is explicitly empty, no outputs can match
    if candidates is not None and not candidates:
        return set()

    # Form memoization cache key to accelerate repeated candidate pair inspections
    tree_key = id(tree) if tree is not None else (len(source_text), hash(source_text))
    start_col = _extract_first_unit_coord(unit, ("start_col", "start_col_offset"))
    end_col = _extract_unit_end_col(unit)
    cands_key = frozenset(candidates) if candidates is not None else None
    cache_key = (
        tree_key,
        unit.get("file", ""),
        u_start,
        u_end,
        start_col,
        end_col,
        cands_key,
        bool(skip_pre_unit_closures),
    )
    if cache_key in _downstream_reads_cache:
        cached = _downstream_reads_cache[cache_key]
        _downstream_reads_cache.move_to_end(cache_key)
        return set(cached)

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

    # Detect loops enclosing the unit for loop-carried dependence analysis
    enclosing_loops: List[Tuple[int, int]] = []
    for node in ast.walk(scope_node):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            l_start = getattr(node, "lineno", 0)
            l_end = getattr(node, "end_lineno", l_start)
            if l_start <= u_start and u_end <= l_end:
                if (l_start, l_end) != (u_start, u_end):
                    enclosing_loops.append((l_start, l_end))

    # Free variables captured by pre-unit closures (skippable in fast mode when precomputed)
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
        pass_mode="after_unit",
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
            pass_mode="loop_carried",
        )
        for stmt in getattr(scope_node, "body", []):
            loop_visitor.visit(stmt)
        loaded.update(loop_visitor.loaded)

    # Any pre-unit closure, lambda, or class method capturing an output candidate could escape
    # or be invoked indirectly (e.g. registered into a callback table prior to unit execution).
    # Treat captured candidates as live to ensure fail-closed safety.
    if candidates is not None:
        loaded.update(pre_unit_captured_reads & candidates)
    else:
        loaded.update(pre_unit_captured_reads)

    if len(_downstream_reads_cache) >= _MAX_DOWNSTREAM_CACHE_SIZE:
        _downstream_reads_cache.popitem(last=False)
    _downstream_reads_cache[cache_key] = frozenset(loaded)

    return loaded


def _pair_clone_outputs(
    u1_outs: Sequence[str],
    u2_outs: Sequence[str],
) -> List[Tuple[str, str]]:
    """Establishes an ordered 1-to-1 mapping between clone output variables.

    Pairing Algorithm:
    1. Input deduplication: preserves first-occurrence order via dict.fromkeys.
    2. Arity equivalence: if the deduplicated counts match (len(u1_dedup) == len(u2_dedup)),
       it pairs identical names first ('identity-first'), then maps remaining names by position.
    3. Arity collapse: if deduplicated counts differ (e.g., duplicate output variable names
       collapsing one side to fewer unique names), it falls back to identical names only.
       Downstream callers (such as `resolve_generator_subroutine_outputs`) enforce strict
       arity retention and fail closed if any required output is left unpaired.
    """
    u1_dedup = list(dict.fromkeys(u1_outs))
    u2_dedup = list(dict.fromkeys(u2_outs))
    if len(u1_dedup) == len(u2_dedup):
        common = set(u1_dedup) & set(u2_dedup)
        out_map = {o: o for o in common}
        rem_u1 = [o for o in u1_dedup if o not in common]
        rem_u2 = [o for o in u2_dedup if o not in common]
        for o1, o2 in zip(rem_u1, rem_u2):
            out_map[o1] = o2
        return [(o1, out_map[o1]) for o1 in u1_dedup]
    common_names = [o for o in u1_dedup if o in u2_dedup]
    return [(o, o) for o in common_names]


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
    for side_label, def_set, col_idx in (("u1", u1_def, 0), ("u2", u2_def, 1)):
        if def_set is not None:
            missing = [p[col_idx] for p in kept_pairs if p[col_idx] not in def_set]
            if missing:
                logger.debug(
                    "Rejecting generator subroutine outputs: %s output lacks definite store "
                    "before exit, risking UnboundLocalError: %s",
                    side_label,
                    missing,
                )
                return None

    return [o1 for o1, _ in kept_pairs], [o2 for _, o2 in kept_pairs]


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
    skip_pre_unit_closures: bool = False,
) -> Optional[Tuple[List[str], List[str]]]:
    """Resolves output variable bindings for clone generator subroutines across two code units.

    Note: When both u1 and u2 supply precomputed 'outputs', this fast path skips downstream read
    and definiteness checks, assuming caller resolution has already validated output necessity.
    """
    if (
        isinstance(u1.get("outputs"), (list, tuple, set))
        and isinstance(u2.get("outputs"), (list, tuple, set))
    ):
        outs1 = _extract_effective_unit_outputs(u1, scope1)
        outs2 = _extract_effective_unit_outputs(u2, scope2)
        if len(outs1) == len(outs2):
            pairs = _pair_clone_outputs(outs1, outs2)
            if len(pairs) == len(outs1):
                return [p[0] for p in pairs], [p[1] for p in pairs]
        return None

    u1_raw = _extract_effective_unit_outputs(u1, scope1)
    u2_raw = _extract_effective_unit_outputs(u2, scope2)

    targets = [
        (u1, set(u1_raw), source_text1, tree1),
        (u2, set(u2_raw), source_text2, tree2),
    ]
    downstream_reads: List[Optional[Set[str]]] = []
    for u, cands, s_text, tr in targets:
        src = s_text if s_text is not None else _load_unit_file_text(u, repo_root=repo_root)
        if src is not None:
            reads = collect_downstream_read_names(
                src,
                u,
                candidates=cands,
                tree=tr,
                skip_pre_unit_closures=skip_pre_unit_closures,
            )
        else:
            reads = None
        downstream_reads.append(reads)

    u1_def = set(scope1.get("definite_stores", [])) | set(scope1.get("inputs", []))
    u2_def = set(scope2.get("definite_stores", [])) | set(scope2.get("inputs", []))

    side1 = GeneratorCloneSideData(u1_raw, downstream_reads[0], u1_def)
    side2 = GeneratorCloneSideData(u2_raw, downstream_reads[1], u2_def)

    return resolve_generator_subroutine_outputs(side1, side2)
