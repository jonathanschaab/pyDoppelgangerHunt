"""AST parsing, canonicalization, token normalization, and unit harvesting."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

BUILTIN_NAMES: Set[str] = set(dir(builtins))


def is_boilerplate_node(node: ast.AST) -> bool:
    """Checks if an AST statement node is logging, print, or assertion boilerplate."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id in ("print", "log"):
            return True
        if isinstance(call.func, ast.Attribute):
            if call.func.attr in ("debug", "info", "warning", "error", "critical", "log", "_log"):
                return True
            if isinstance(call.func.value, ast.Name) and call.func.value.id in ("logger", "logging"):
                return True
    return False


class _CommutativeCanonicalizer(ast.NodeTransformer):
    """Canonicalizes commutative and symmetric AST nodes to ensure permutation invariance (NiCad-style)."""

    def _sort_key(self, node: ast.AST) -> str:
        """Generates a stable string key for sorting commutative AST operands."""
        try:
            return ast.dump(node)
        except Exception:  # pylint: disable=broad-exception-caught
            return type(node).__name__

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        """Canonicalizes boolean operations (and, or) by sorting operands."""
        self.generic_visit(node)
        node.values = sorted(node.values, key=self._sort_key)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """Canonicalizes commutative binary operations (+, *, &, |, ^) by sorting operands."""
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)):
            k_left = self._sort_key(node.left)
            k_right = self._sort_key(node.right)
            if k_left > k_right:
                node.left, node.right = node.right, node.left
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        """Canonicalizes symmetric comparisons (==, !=) by sorting operands."""
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            k_left = self._sort_key(node.left)
            k_right = self._sort_key(node.comparators[0])
            if k_left > k_right:
                node.left, node.comparators[0] = node.comparators[0], node.left
        return node


class _IdiomCanonicalizer(ast.NodeTransformer):
    """Canonicalizes Python idioms (loops to comprehensions, search loops to any/all) (PyChase-style)."""

    def _build_listcomp_assign(
        self, node: ast.For, call: ast.Call, ifs: Sequence[ast.expr]
    ) -> Optional[ast.Assign]:
        """Constructs an accumulator list comprehension assignment node."""
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "append"
            and isinstance(call.func.value, ast.Name)
            and len(call.args) == 1
        ):
            return ast.Assign(
                targets=[ast.Name(id=call.func.value.id, ctx=ast.Store())],
                value=ast.ListComp(
                    elt=call.args[0],
                    generators=[
                        ast.comprehension(
                            target=node.target,
                            iter=node.iter,
                            ifs=list(ifs),
                            is_async=0,
                        )
                    ],
                ),
            )
        return None

    @staticmethod
    def _get_single_stmt(node: ast.For) -> Optional[ast.stmt]:
        if len(node.body) == 1 and not node.orelse:
            return node.body[0]
        return None

    @staticmethod
    def _get_single_if_child(stmt: ast.stmt) -> Optional[Tuple[ast.expr, ast.stmt]]:
        if isinstance(stmt, ast.If) and len(stmt.body) == 1 and not stmt.orelse:
            return stmt.test, stmt.body[0]
        return None

    def _transform_accumulator_loop(self, node: ast.For) -> Optional[ast.Assign]:
        """Transforms a single-statement accumulator for-loop into an equivalent ListComp assignment."""
        stmt = self._get_single_stmt(node)
        if stmt is None:
            return None

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return self._build_listcomp_assign(node, stmt.value, [])

        if_info = self._get_single_if_child(stmt)
        if if_info is not None:
            test, inner = if_info
            if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                return self._build_listcomp_assign(node, inner.value, [test])
        return None

    def _build_reduction_return(
        self, func_name: str, test_expr: ast.expr, node: ast.For
    ) -> ast.Return:
        """Constructs a functional reduction (any/all) generator return statement."""
        return ast.Return(
            value=ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[
                    ast.GeneratorExp(
                        elt=test_expr,
                        generators=[
                            ast.comprehension(
                                target=node.target,
                                iter=node.iter,
                                ifs=[],
                                is_async=0,
                            )
                        ],
                    )
                ],
                keywords=[],
            )
        )

    def _transform_reduction_loop(self, node: ast.For) -> Optional[ast.Return]:
        """Transforms search loops into equivalent functional any() or all() generator calls."""
        stmt = self._get_single_stmt(node)
        if stmt is None:
            return None
        if_info = self._get_single_if_child(stmt)
        if if_info is None:
            return None
        test, inner = if_info
        if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Constant):
            ret_val = inner.value.value
            if ret_val is True:
                return self._build_reduction_return("any", test, node)
            if ret_val is False and isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                return self._build_reduction_return("all", test.operand, node)
        return None

    def visit_For(self, node: ast.For) -> ast.AST:
        """Visits and canonicalizes accumulator and reduction for-loops."""
        self.generic_visit(node)
        acc_assign = self._transform_accumulator_loop(node)
        if acc_assign is not None:
            return ast.fix_missing_locations(ast.copy_location(acc_assign, node))
        red_ret = self._transform_reduction_loop(node)
        if red_ret is not None:
            return ast.fix_missing_locations(ast.copy_location(red_ret, node))
        return node


def _walk_ast_nodes(
    root: ast.AST,
    strip_annotations: bool = False,
    filter_boilerplate: bool = False,
    abstract_expressions: bool = False,
) -> List[ast.AST]:
    """Traverses AST nodes in BFS order, optionally omitting annotations, boilerplate, or abstracting expressions."""
    nodes: List[ast.AST] = []
    todo: List[ast.AST] = [root]
    while todo:
        node = todo.pop(0)
        if filter_boilerplate and is_boilerplate_node(node):
            continue
        nodes.append(node)
        for name, value in ast.iter_fields(node):
            if strip_annotations and name in ("annotation", "returns", "type_comment"):
                continue
            if abstract_expressions and name == "test" and isinstance(node, (ast.If, ast.While, ast.IfExp)):
                if isinstance(value, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                    nodes.append(ast.Name(id="__ABSTRACT_COND__", ctx=ast.Load()))
                    continue
            if abstract_expressions and name == "value" and isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                if isinstance(value, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp)):
                    nodes.append(ast.Name(id="__ABSTRACT_EXPR__", ctx=ast.Load()))
                    continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        todo.append(item)
            elif isinstance(value, ast.AST):
                todo.append(value)
    return nodes


def _resolve_consistent_token(
    name: str,
    arg_map: Dict[str, int],
    lvar_map: Optional[Dict[str, int]] = None,
) -> str:
    """Resolves identifier token preserving builtins, parameter bindings, and local assignments."""
    if name in BUILTIN_NAMES:
        return f"BUILTIN_{name}"
    if name in arg_map:
        return f"ARG_{arg_map[name]}"
    if lvar_map is not None:
        return f"LVAR_{lvar_map.setdefault(name, len(lvar_map))}"
    return f"ARG_{arg_map.setdefault(name, len(arg_map))}"


def _resolve_named_token(
    name: str,
    kind: str,
    consistent_renaming: bool,
    blind_indexing: bool,
    primary_map: Dict[str, int],
    secondary_map: Optional[Dict[str, int]] = None,
) -> str:
    """Formats an AST identifier token according to renaming and indexing configurations."""
    if consistent_renaming:
        return _resolve_consistent_token(name, primary_map, secondary_map)
    if blind_indexing:
        target = secondary_map if secondary_map is not None else primary_map
        return f"{kind}_{target.setdefault(name, len(target))}"
    return kind


def _get_docstring_node(root_node: ast.AST) -> Optional[ast.AST]:
    """Retrieves the docstring expression node from a statement container."""
    if hasattr(root_node, "body") and root_node.body:
        first_stmt = root_node.body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str):
            return first_stmt
    return None


def get_ast_tokens(
    node_or_nodes: Any,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
) -> List[str]:
    """Normalizes an AST node or list of statements into a structural token sequence."""
    nodes = node_or_nodes if isinstance(node_or_nodes, (list, tuple)) else [node_or_nodes]
    tokens: List[str] = []
    var_map: Dict[str, int] = {}
    attr_map: Dict[str, int] = {}
    arg_map: Dict[str, int] = {}
    lvar_map: Dict[str, int] = {}

    for root_node in nodes:
        doc_nodes: Set[int] = set()
        if strip_docstrings or blind_literals:
            sub_roots = [root_node] if isinstance(root_node, ast.AST) else [x for x in root_node if isinstance(x, ast.AST)]
            for r in sub_roots:
                for container in ast.walk(r):
                    if isinstance(container, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                        dn = _get_docstring_node(container)
                        if dn is not None:
                            doc_nodes.add(id(dn))
                            if hasattr(dn, "value"):
                                doc_nodes.add(id(dn.value))

        if consistent_renaming and isinstance(root_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in (
                root_node.args.posonlyargs
                + root_node.args.args
                + root_node.args.kwonlyargs
            ):
                if a.arg not in BUILTIN_NAMES:
                    arg_map.setdefault(a.arg, len(arg_map))
            if root_node.args.vararg and root_node.args.vararg.arg not in BUILTIN_NAMES:
                arg_map.setdefault(root_node.args.vararg.arg, len(arg_map))
            if root_node.args.kwarg and root_node.args.kwarg.arg not in BUILTIN_NAMES:
                arg_map.setdefault(root_node.args.kwarg.arg, len(arg_map))

        for child in _walk_ast_nodes(
            root_node,
            strip_annotations=strip_annotations,
            filter_boilerplate=filter_boilerplate,
            abstract_expressions=abstract_expressions,
        ):
            if id(child) in doc_nodes:
                continue
            node_type = type(child).__name__
            if isinstance(child, ast.Constant):
                if blind_literals:
                    tokens.append("LITERAL")
                else:
                    tokens.append(f"CONST_{type(child.value).__name__}")
            elif isinstance(child, ast.Name):
                if child.id == "__ABSTRACT_EXPR__":
                    tokens.append("EXPR")
                elif child.id == "__ABSTRACT_COND__":
                    tokens.append("COND")
                else:
                    tokens.append(
                        _resolve_named_token(
                            child.id,
                            "VAR",
                            consistent_renaming,
                            blind_indexing,
                            arg_map,
                            lvar_map if consistent_renaming else var_map,
                        )
                    )
            elif isinstance(child, ast.Attribute):
                if consistent_renaming or blind_indexing:
                    tokens.append(f"ATTR_{attr_map.setdefault(child.attr, len(attr_map))}")
                else:
                    tokens.append("ATTR")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tokens.append("FUNC")
            elif isinstance(child, ast.ClassDef):
                tokens.append("CLASS")
            elif isinstance(child, ast.arg):
                tokens.append(
                    _resolve_named_token(
                        child.arg,
                        "ARG",
                        consistent_renaming,
                        blind_indexing,
                        arg_map,
                    )
                )
            elif strip_annotations and isinstance(child, ast.AnnAssign):
                tokens.append("Assign")
            else:
                tokens.append(node_type)

    return tokens


def get_ast_shingles(
    node_or_nodes: Any,
    k: int = 3,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
) -> Tuple[Set[Tuple[str, ...]], int]:
    """Normalizes an AST node or list of statements into structural tokens and produces overlapping k-shingles."""
    tokens = get_ast_tokens(
        node_or_nodes,
        blind_indexing=blind_indexing,
        strip_annotations=strip_annotations,
        blind_literals=blind_literals,
        filter_boilerplate=filter_boilerplate,
        consistent_renaming=consistent_renaming,
        abstract_expressions=abstract_expressions,
        strip_docstrings=strip_docstrings,
    )
    if len(tokens) < k:
        return {tuple(tokens)} if tokens else set(), len(tokens)

    shingles: Set[Tuple[str, ...]] = set()
    for i in range(len(tokens) - k + 1):
        shingles.add(tuple(tokens[i:i + k]))

    return shingles, len(tokens)


def get_ast_characteristic_vector(
    node_or_nodes: Any,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
) -> Dict[str, int]:
    """Computes a Deckard-style multiset frequency vector of AST tokens for permutation-invariant comparison."""
    tokens = get_ast_tokens(
        node_or_nodes,
        blind_indexing=blind_indexing,
        strip_annotations=strip_annotations,
        blind_literals=blind_literals,
        filter_boilerplate=filter_boilerplate,
        consistent_renaming=consistent_renaming,
        abstract_expressions=abstract_expressions,
        strip_docstrings=strip_docstrings,
    )
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return counts


class _CallSequenceExtractor(ast.NodeVisitor):
    """Extracts the ordered sequence of call target names from an AST node."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Visits function/method call nodes and extracts their function name or attribute."""
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


def extract_call_sequence(node_or_nodes: Any) -> List[str]:
    """Extracts ordered list of function and method calls from AST node(s)."""
    extractor = _CallSequenceExtractor()
    targets = node_or_nodes if isinstance(node_or_nodes, list) else [node_or_nodes]
    for item in targets:
        if isinstance(item, ast.AST):
            extractor.visit(item)
    return extractor.calls


def _compute_expression_complexity(node: ast.AST) -> int:
    """Computes structural operator complexity of an AST expression."""
    count = 0
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Call,
                ast.Subscript,
                ast.IfExp,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
            ),
        ):
            count += 1
    return count


def _harvest_complex_expressions(
    root: ast.AST, min_complexity: int = 4
) -> List[Tuple[ast.AST, int, int]]:
    """Harvests outermost expression subtrees with operator complexity >= min_complexity."""
    results: List[Tuple[ast.AST, int, int]] = []
    expr_classes = (
        ast.BinOp,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.IfExp,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
    )

    def _walk_outer(node: ast.AST) -> None:
        if isinstance(node, expr_classes):
            if _compute_expression_complexity(node) >= min_complexity:
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                results.append((node, start, end))
                return
        for child in ast.iter_child_nodes(node):
            _walk_outer(child)

    _walk_outer(root)
    return results


class _ScopeHierarchyVisitor(ast.NodeVisitor):
    """Tracks lexical parent scopes for nested functions, closures, and enclosing classes."""

    def __init__(
        self,
        closure_parents: Optional[Dict[int, str]] = None,
        enclosing_classes: Optional[Dict[int, str]] = None,
        enclosing_class_starts: Optional[Dict[int, int]] = None,
        func_owners: Optional[Dict[int, str]] = None,
        harvest_closures: bool = True,
    ) -> None:
        self.closure_parents: Dict[int, str] = closure_parents if closure_parents is not None else {}
        self.enclosing_classes: Dict[int, str] = enclosing_classes if enclosing_classes is not None else {}
        self.enclosing_class_starts: Dict[int, int] = (
            enclosing_class_starts if enclosing_class_starts is not None else {}
        )
        self.func_owners: Dict[int, str] = func_owners if func_owners is not None else {}
        self.harvest_closures: bool = harvest_closures
        self.func_stack: List[str] = []
        self.class_stack: List[Tuple[str, int]] = []

    def _record_class_enclosure(self, node: ast.AST) -> None:
        if self.class_stack:
            self.enclosing_classes[id(node)] = self.class_stack[-1][0]
            self.enclosing_class_starts[id(node)] = self.class_stack[-1][1]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for tp in getattr(node, "type_params", ()):
            self.visit(tp)

        self._record_class_enclosure(node)
        c_start = int(getattr(node, "lineno", 0))
        self.class_stack.append((node.name, c_start))
        saved_func_stack = self.func_stack
        self.func_stack = []
        for stmt in node.body:
            self.visit(stmt)
        self.func_stack = saved_func_stack
        self.class_stack.pop()

    def _scope_function(self, fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        for dec in fn.decorator_list:
            self.visit(dec)
        for default in fn.args.defaults:
            self.visit(default)
        for kw_default in fn.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        if fn.returns is not None:
            self.visit(fn.returns)
        for tp in getattr(fn, "type_params", ()):
            self.visit(tp)

        self._record_class_enclosure(fn)
        if self.func_stack:
            self.closure_parents[id(fn)] = ":".join(self.func_stack)
        self.func_stack.append(fn.name)
        for stmt in fn.body:
            self.visit(stmt)
        self.func_stack.pop()

    def visit_FunctionDef(self, fn: ast.FunctionDef) -> None:
        self._scope_function(fn)

    def visit_AsyncFunctionDef(self, fn: ast.AsyncFunctionDef) -> None:
        self._scope_function(fn)

    def _scope_comprehension(self, comp: ast.AST) -> None:
        self._record_class_enclosure(comp)
        if self.func_stack:
            if self.harvest_closures or len(self.func_stack) == 1:
                self.func_owners[id(comp)] = ":".join(self.func_stack)
            else:
                self.func_owners[id(comp)] = self.func_stack[-1]
        self.generic_visit(comp)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._scope_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._scope_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._scope_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._scope_comprehension(node)


_ClosureScoper = _ScopeHierarchyVisitor


class _DataTableVisitor(ast.NodeVisitor):
    """Harvests top-level module data tables (dict, list, set, tuple assignments)."""

    def __init__(self) -> None:
        self.tables: List[Tuple[str, ast.AST, int, int]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                self.tables.append((name, node.value, start, end))


def check_inline_suppression(file_lines: List[str], start: int, end: int) -> bool:
    """Checks whether the AST unit's source range contains an inline suppression pragma."""
    check_start = max(1, start - 1)
    check_end = min(len(file_lines), end)
    for line_num in range(check_start, check_end + 1):
        line = file_lines[line_num - 1]
        if "#" in line:
            comment = line.split("#", 1)[1].lower()
            if "pydoppelgangerhunt: ignore" in comment or "pydoppelgangerhunt:ignore" in comment:
                return True
            if "noqa: clone" in comment or "noqa:clone" in comment or "noqa: duplicate" in comment:
                return True
    return False


def is_decorator_named(d: ast.AST, name: str) -> bool:
    """Checks if an AST decorator node matches a given name, supporting call expressions."""
    target = d.func if isinstance(d, ast.Call) else d
    if isinstance(target, ast.Name):
        return bool(target.id == name)
    if isinstance(target, ast.Attribute):
        return bool(target.attr == name)
    return False


def compute_cyclomatic_complexity(target: Any) -> int:
    """Calculates McCabe Cyclomatic Complexity for an AST node or statement sequence."""
    if not target:
        return 1
    nodes: Sequence[ast.AST]
    if isinstance(target, ast.AST):
        nodes = [target]
    elif isinstance(target, (list, tuple)):
        nodes = [n for n in target if isinstance(n, ast.AST)]
    else:
        return 1

    complexity = 1
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.IfExp,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.With,
                    ast.AsyncWith,
                ),
            ):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += max(0, len(node.values) - 1)
            elif isinstance(node, ast.comprehension):
                complexity += len(node.ifs)
    return complexity


def _record_unit(
    units: List[Dict[str, Any]],
    name: str,
    rel_file: str,
    start: int,
    end: int,
    ast_target: Any,
    min_lines: int,
    min_tokens: int,
    kind: str,
    file_lines: Optional[List[str]] = None,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
    start_col: Optional[int] = None,
    end_col: Optional[int] = None,
    enclosing_class: Optional[str] = None,
    enclosing_class_start: Optional[int] = None,
    receiver_kind: Optional[str] = None,
    is_static: bool = False,
) -> None:
    """Records an AST unit if it satisfies thresholds and is not suppressed by inline comments."""
    if file_lines and check_inline_suppression(file_lines, start, end):
        return

    if start_col is None:
        if isinstance(ast_target, ast.AST):
            start_col = getattr(ast_target, "col_offset", 0)
        elif isinstance(ast_target, (list, tuple)) and ast_target:
            start_col = getattr(ast_target[0], "col_offset", 0)
        else:
            start_col = 0
    if end_col is None:
        if isinstance(ast_target, ast.AST):
            end_col = getattr(ast_target, "end_col_offset", None)
        elif isinstance(ast_target, (list, tuple)) and ast_target:
            end_col = getattr(ast_target[-1], "end_col_offset", None)
        else:
            end_col = None

    lines = end - start + 1
    if kind == "complex_expr":
        effective_min_lines = 1
        effective_min_tokens = max(8, min_tokens // 2)
    elif kind == "data_table":
        effective_min_lines = 1
        effective_min_tokens = max(10, min_tokens // 2)
    elif kind == "clause_branch":
        effective_min_lines = max(3, min_lines // 2)
        effective_min_tokens = max(10, min_tokens // 2)
    elif kind == "class":
        effective_min_lines = max(3, min_lines // 2)
        effective_min_tokens = min_tokens
    elif kind == "closure":
        effective_min_lines = max(4, min_lines // 2)
        effective_min_tokens = max(10, min_tokens // 2)
    elif kind == "comprehension":
        effective_min_lines = 1
        effective_min_tokens = max(8, min_tokens // 2)
    else:
        effective_min_lines = min_lines
        effective_min_tokens = min_tokens

    if lines >= effective_min_lines:
        tokens = get_ast_tokens(
            ast_target,
            blind_indexing=blind_indexing,
            strip_annotations=strip_annotations,
            blind_literals=blind_literals,
            filter_boilerplate=filter_boilerplate,
            consistent_renaming=consistent_renaming,
            abstract_expressions=abstract_expressions,
            strip_docstrings=strip_docstrings,
        )
        token_count = len(tokens)
        if token_count >= effective_min_tokens:
            shingles, _ = get_ast_shingles(
                ast_target,
                blind_indexing=blind_indexing,
                strip_annotations=strip_annotations,
                blind_literals=blind_literals,
                filter_boilerplate=filter_boilerplate,
                consistent_renaming=consistent_renaming,
                abstract_expressions=abstract_expressions,
                strip_docstrings=strip_docstrings,
            )
            char_vector = get_ast_characteristic_vector(
                ast_target,
                blind_indexing=blind_indexing,
                strip_annotations=strip_annotations,
                blind_literals=blind_literals,
                filter_boilerplate=filter_boilerplate,
                consistent_renaming=consistent_renaming,
                abstract_expressions=abstract_expressions,
                strip_docstrings=strip_docstrings,
            )
            # 16 hex chars = 64 bits of entropy (2^64 keyspace). Collision probability for N=100k
            # units is P ≈ N^2 / (2 * 2^64) ≈ 2.7e-10 (< 1 in 3.7B), rendering collisions negligible.
            structural_hash = hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]
            units.append({
                "name": name,
                "file": rel_file,
                "start": start,
                "end": end,
                "start_col": start_col,
                "end_col": end_col,
                "lines": lines,
                "tokens": tokens,
                "shingles": shingles,
                "token_count": token_count,
                "vector": char_vector,
                "kind": kind,
                "calls": extract_call_sequence(ast_target),
                "complexity": compute_cyclomatic_complexity(ast_target),
                "structural_hash": structural_hash,
                "enclosing_class": enclosing_class,
                "enclosing_class_start": enclosing_class_start,
                "receiver_kind": receiver_kind,
                "is_static": is_static,
            })


def _record_clause_branch(
    units: List[Dict[str, Any]],
    func_name: str,
    clause_label: str,
    rel_file: str,
    body: List[ast.stmt],
    default_line: int,
    min_lines: int,
    min_tokens: int,
    file_lines: Optional[List[str]] = None,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
    enclosing_class: Optional[str] = None,
    enclosing_class_start: Optional[int] = None,
    receiver_kind: Optional[str] = None,
    is_static: bool = False,
) -> None:
    """Records an if-branch or except-handler clause if it contains at least 3 statements."""
    if len(body) < 3:
        return
    c_start = getattr(body[0], "lineno", default_line)
    c_end = getattr(body[-1], "end_lineno", c_start)
    _record_unit(
        units,
        f"{func_name}:{clause_label}",
        rel_file,
        c_start,
        c_end,
        body,
        min_lines,
        min_tokens,
        "clause_branch",
        file_lines=file_lines,
        blind_indexing=blind_indexing,
        strip_annotations=strip_annotations,
        blind_literals=blind_literals,
        filter_boilerplate=filter_boilerplate,
        consistent_renaming=consistent_renaming,
        abstract_expressions=abstract_expressions,
        strip_docstrings=strip_docstrings,
        enclosing_class=enclosing_class,
        enclosing_class_start=enclosing_class_start,
        receiver_kind=receiver_kind,
        is_static=is_static,
    )


def _record_node_unit(
    units: List[Dict[str, Any]],
    name: str,
    rel_file: str,
    node: ast.AST,
    min_lines: int,
    min_tokens: int,
    kind: str,
    file_lines: Optional[List[str]] = None,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
    enclosing_class: Optional[str] = None,
    enclosing_class_start: Optional[int] = None,
    receiver_kind: Optional[str] = None,
    is_static: bool = False,
) -> None:
    """Records an AST node unit by extracting its start and end line bounds."""
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    _record_unit(
        units,
        name,
        rel_file,
        start,
        end,
        node,
        min_lines,
        min_tokens,
        kind,
        file_lines=file_lines,
        blind_indexing=blind_indexing,
        strip_annotations=strip_annotations,
        blind_literals=blind_literals,
        filter_boilerplate=filter_boilerplate,
        consistent_renaming=consistent_renaming,
        abstract_expressions=abstract_expressions,
        strip_docstrings=strip_docstrings,
        enclosing_class=enclosing_class,
        enclosing_class_start=enclosing_class_start,
        receiver_kind=receiver_kind,
        is_static=is_static,
    )


def harvest_notebook_units(
    file_path: str,
    repo_root: str,
    min_lines: int = 8,
    min_tokens: int = 15,
    blind_indexing: bool = False,
    strip_annotations: bool = True,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    commutative: bool = False,
    idioms: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
) -> List[Dict[str, Any]]:
    """Extracts code cells from Jupyter Notebook (.ipynb) files and harvests AST units."""
    units: List[Dict[str, Any]] = []
    p = Path(file_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return units

    cells = data.get("cells", [])
    if not isinstance(cells, list):
        return units

    try:
        rel_file = str(p.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        rel_file = str(p).replace("\\", "/")

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source_raw = cell.get("source", [])
        cell_code = "".join(source_raw) if isinstance(source_raw, list) else str(source_raw)
        if not cell_code.strip():
            continue

        cell_label = f"{rel_file}#cell_{idx + 1}"
        try:
            tree = ast.parse(cell_code, filename=cell_label)
        except SyntaxError:
            continue

        file_lines = cell_code.splitlines()
        if idioms:
            tree = _IdiomCanonicalizer().visit(tree)
            ast.fix_missing_locations(tree)
        if commutative:
            tree = _CommutativeCanonicalizer().visit(tree)
            ast.fix_missing_locations(tree)

        _record_node_unit(
            units,
            f"cell_{idx + 1}",
            cell_label,
            tree,
            min_lines,
            min_tokens,
            "notebook_cell",
            file_lines=file_lines,
            blind_indexing=blind_indexing,
            strip_annotations=strip_annotations,
            blind_literals=blind_literals,
            filter_boilerplate=filter_boilerplate,
            consistent_renaming=consistent_renaming,
            abstract_expressions=abstract_expressions,
            strip_docstrings=strip_docstrings,
        )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_start = getattr(node, "lineno", 0)
                fn_end = getattr(node, "end_lineno", fn_start)
                if file_lines and check_inline_suppression(file_lines, fn_start, fn_end):
                    continue
                _record_node_unit(
                    units,
                    node.name,
                    cell_label,
                    node,
                    min_lines,
                    min_tokens,
                    "function",
                    file_lines=file_lines,
                    blind_indexing=blind_indexing,
                    strip_annotations=strip_annotations,
                    blind_literals=blind_literals,
                    filter_boilerplate=filter_boilerplate,
                    consistent_renaming=consistent_renaming,
                    abstract_expressions=abstract_expressions,
                    strip_docstrings=strip_docstrings,
                )

    return units


def _iter_local_nodes(root: ast.AST) -> Iterator[ast.AST]:
    """Iterates through descendant AST nodes without crossing nested function or class definitions."""
    todo = deque(ast.iter_child_nodes(root))
    while todo:
        current = todo.popleft()
        yield current
        if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(current))


def harvest_file_units(
    file_path: str,
    repo_root: str,
    min_lines: int = 8,
    min_tokens: int = 15,
    functions_only: bool = False,
    sliding_window: bool = False,
    window_size: int = 5,
    blind_indexing: bool = False,
    complex_expressions: bool = False,
    min_expr_complexity: int = 4,
    clause_level: bool = False,
    data_tables: bool = False,
    strip_annotations: bool = True,
    class_level: bool = False,
    blind_literals: bool = False,
    filter_boilerplate: bool = False,
    consistent_renaming: bool = False,
    harvest_closures: bool = False,
    commutative: bool = False,
    comprehensions: bool = False,
    idioms: bool = False,
    abstract_expressions: bool = False,
    strip_docstrings: bool = True,
) -> List[Dict[str, Any]]:
    """Harvests AST code units from a single Python source file. Picklable for multi-core worker pools."""
    if file_path.endswith(".ipynb"):
        return harvest_notebook_units(
            file_path,
            repo_root,
            min_lines=min_lines,
            min_tokens=min_tokens,
            blind_indexing=blind_indexing,
            strip_annotations=strip_annotations,
            blind_literals=blind_literals,
            filter_boilerplate=filter_boilerplate,
            consistent_renaming=consistent_renaming,
            commutative=commutative,
            idioms=idioms,
            abstract_expressions=abstract_expressions,
            strip_docstrings=strip_docstrings,
        )

    units: List[Dict[str, Any]] = []
    p = Path(file_path)
    try:
        source = p.read_text(encoding="utf-8")
        file_lines = source.splitlines()
        tree = ast.parse(source, filename=str(p))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return units

    if idioms:
        tree = _IdiomCanonicalizer().visit(tree)
        ast.fix_missing_locations(tree)
    if commutative:
        tree = _CommutativeCanonicalizer().visit(tree)
        ast.fix_missing_locations(tree)

    try:
        rel_file = str(p.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        rel_file = str(p).replace("\\", "/")

    closure_parents: Dict[int, str] = {}
    enclosing_classes: Dict[int, str] = {}
    enclosing_class_starts: Dict[int, int] = {}
    func_owners: Dict[int, str] = {}
    _ScopeHierarchyVisitor(
        closure_parents,
        enclosing_classes,
        enclosing_class_starts,
        func_owners=func_owners,
        harvest_closures=harvest_closures,
    ).visit(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_start = getattr(node, "lineno", 0)
            fn_end = getattr(node, "end_lineno", fn_start)
            if file_lines and check_inline_suppression(file_lines, fn_start, fn_end):
                continue

            enc_class = enclosing_classes.get(id(node))
            enc_class_start = enclosing_class_starts.get(id(node))
            decs = getattr(node, "decorator_list", [])
            fn_is_static = any(is_decorator_named(d, "staticmethod") for d in decs)
            fn_is_class_method = any(is_decorator_named(d, "classmethod") for d in decs)
            is_nested = id(node) in closure_parents
            is_closure = harvest_closures and is_nested
            if is_nested:
                fn_receiver_kind: Optional[str] = None
                fn_is_static = False
                fn_is_class_method = False
            elif fn_is_static:
                fn_receiver_kind = "static"
            elif fn_is_class_method:
                fn_receiver_kind = "class"
            elif enc_class:
                fn_receiver_kind = "instance"
            else:
                fn_receiver_kind = None
            if is_closure:
                unit_kind = "closure"
                unit_name = f"{closure_parents[id(node)]}:{node.name}"
            else:
                unit_kind = "function"
                unit_name = node.name

            # 1. Whole function / closure unit: omit thin 1-statement delegate wrappers
            doc_offset = (
                1
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                else 0
            )
            if len(node.body) - doc_offset > 1 or is_closure:
                _record_node_unit(
                    units,
                    unit_name,
                    rel_file,
                    node,
                    min_lines,
                    min_tokens,
                    unit_kind,
                    file_lines=file_lines,
                    blind_indexing=blind_indexing,
                    strip_annotations=strip_annotations,
                    blind_literals=blind_literals,
                    filter_boilerplate=filter_boilerplate,
                    consistent_renaming=consistent_renaming,
                    abstract_expressions=abstract_expressions,
                    strip_docstrings=strip_docstrings,
                    enclosing_class=enc_class,
                    enclosing_class_start=enc_class_start,
                    receiver_kind=fn_receiver_kind,
                    is_static=fn_is_static,
                )

            if not functions_only:
                for item in _iter_local_nodes(node):
                    if item is not node and isinstance(
                        item,
                        (
                            ast.If,
                            ast.For,
                            ast.AsyncFor,
                            ast.While,
                            ast.Try,
                            ast.With,
                            ast.AsyncWith,
                        ),
                    ):
                        item_start = getattr(item, "lineno", 0)
                        item_end = getattr(item, "end_lineno", item_start)
                        block_lines = item_end - item_start + 1
                        if block_lines < (node.end_lineno - node.lineno + 1):  # type: ignore[operator]
                            _record_unit(
                                units,
                                f"{unit_name}:{type(item).__name__}",
                                rel_file,
                                item_start,
                                item_end,
                                item,
                                min_lines,
                                min_tokens,
                                "compound_block",
                                file_lines=file_lines,
                                blind_indexing=blind_indexing,
                                strip_annotations=strip_annotations,
                                blind_literals=blind_literals,
                                filter_boilerplate=filter_boilerplate,
                                consistent_renaming=consistent_renaming,
                                abstract_expressions=abstract_expressions,
                                strip_docstrings=strip_docstrings,
                                enclosing_class=enc_class,
                                enclosing_class_start=enc_class_start,
                                receiver_kind=fn_receiver_kind,
                                is_static=fn_is_static,
                            )

            if sliding_window and hasattr(node, "body"):
                body_stmts = [s for s in node.body if isinstance(s, ast.stmt)]
                if len(body_stmts) >= window_size:
                    for w_idx in range(len(body_stmts) - window_size + 1):
                        window_slice = body_stmts[w_idx : w_idx + window_size]
                        w_start = getattr(window_slice[0], "lineno", 0)
                        w_end = getattr(window_slice[-1], "end_lineno", w_start)
                        _record_unit(
                            units,
                            f"{unit_name}:stmts_{w_idx+1}-{w_idx+window_size}",
                            rel_file,
                            w_start,
                            w_end,
                            window_slice,
                            min_lines=min_lines,
                            min_tokens=min_tokens,
                            kind="sliding_window",
                            file_lines=file_lines,
                            blind_indexing=blind_indexing,
                            strip_annotations=strip_annotations,
                            blind_literals=blind_literals,
                            filter_boilerplate=filter_boilerplate,
                            consistent_renaming=consistent_renaming,
                            abstract_expressions=abstract_expressions,
                            strip_docstrings=strip_docstrings,
                            enclosing_class=enc_class,
                            enclosing_class_start=enc_class_start,
                            receiver_kind=fn_receiver_kind,
                            is_static=fn_is_static,
                        )

            if clause_level and hasattr(node, "body"):
                for stmt in _iter_local_nodes(node):
                    if isinstance(stmt, ast.If):
                        s_line = getattr(stmt, "lineno", 0)
                        _record_clause_branch(
                            units,
                            unit_name,
                            "if_branch",
                            rel_file,
                            stmt.body,
                            s_line,
                            min_lines,
                            min_tokens,
                            file_lines=file_lines,
                            blind_indexing=blind_indexing,
                            strip_annotations=strip_annotations,
                            blind_literals=blind_literals,
                            filter_boilerplate=filter_boilerplate,
                            consistent_renaming=consistent_renaming,
                            abstract_expressions=abstract_expressions,
                            strip_docstrings=strip_docstrings,
                            enclosing_class=enc_class,
                            enclosing_class_start=enc_class_start,
                            receiver_kind=fn_receiver_kind,
                            is_static=fn_is_static,
                        )
                        if stmt.orelse:
                            _record_clause_branch(
                                units,
                                unit_name,
                                "else_branch",
                                rel_file,
                                stmt.orelse,
                                s_line,
                                min_lines,
                                min_tokens,
                                file_lines=file_lines,
                                blind_indexing=blind_indexing,
                                strip_annotations=strip_annotations,
                                blind_literals=blind_literals,
                                filter_boilerplate=filter_boilerplate,
                                consistent_renaming=consistent_renaming,
                                abstract_expressions=abstract_expressions,
                                strip_docstrings=strip_docstrings,
                                enclosing_class=enc_class,
                                enclosing_class_start=enc_class_start,
                                receiver_kind=fn_receiver_kind,
                                is_static=fn_is_static,
                            )
                    elif isinstance(stmt, ast.Try):
                        t_line = getattr(stmt, "lineno", 0)
                        for h_idx, handler in enumerate(stmt.handlers):
                            h_name = (
                                handler.type.id
                                if isinstance(handler.type, ast.Name)
                                else f"handler_{h_idx}"
                            )
                            _record_clause_branch(
                                units,
                                unit_name,
                                f"except_{h_name}",
                                rel_file,
                                handler.body,
                                t_line,
                                min_lines,
                                min_tokens,
                                file_lines=file_lines,
                                blind_indexing=blind_indexing,
                                strip_annotations=strip_annotations,
                                blind_literals=blind_literals,
                                filter_boilerplate=filter_boilerplate,
                                consistent_renaming=consistent_renaming,
                                abstract_expressions=abstract_expressions,
                                strip_docstrings=strip_docstrings,
                                enclosing_class=enc_class,
                                enclosing_class_start=enc_class_start,
                                receiver_kind=fn_receiver_kind,
                                is_static=fn_is_static,
                            )

        elif class_level and isinstance(node, ast.ClassDef):
            _record_node_unit(
                units,
                f"class:{node.name}",
                rel_file,
                node,
                min_lines,
                min_tokens,
                "class",
                file_lines=file_lines,
                blind_indexing=blind_indexing,
                strip_annotations=strip_annotations,
                blind_literals=blind_literals,
                filter_boilerplate=filter_boilerplate,
                consistent_renaming=consistent_renaming,
                abstract_expressions=abstract_expressions,
                strip_docstrings=strip_docstrings,
                enclosing_class=enclosing_classes.get(id(node)),
                enclosing_class_start=enclosing_class_starts.get(id(node)),
            )

    if complex_expressions:  # pydoppelgangerhunt: ignore
        for expr_node, expr_start, expr_end in _harvest_complex_expressions(
            tree, min_complexity=min_expr_complexity
        ):
            _record_unit(
                units,
                f"expr:{expr_start}",
                rel_file,
                expr_start,
                expr_end,
                expr_node,
                min_lines=1,
                min_tokens=max(8, min_tokens // 2),
                kind="complex_expr",
                file_lines=file_lines,
                blind_indexing=blind_indexing,
                strip_annotations=strip_annotations,
                blind_literals=blind_literals,
                filter_boilerplate=filter_boilerplate,
                consistent_renaming=consistent_renaming,
                abstract_expressions=abstract_expressions,
                strip_docstrings=strip_docstrings,
            )

    if data_tables:  # pydoppelgangerhunt: ignore
        dt_visitor = _DataTableVisitor()
        dt_visitor.visit(tree)
        for tbl_name, tbl_node, tbl_start, tbl_end in dt_visitor.tables:
            _record_unit(
                units,
                f"data_table:{tbl_name}",
                rel_file,
                tbl_start,
                tbl_end,
                tbl_node,
                min_lines=1,
                min_tokens=max(10, min_tokens // 2),
                kind="data_table",
                file_lines=file_lines,
                blind_indexing=blind_indexing,
                strip_annotations=strip_annotations,
                blind_literals=blind_literals,
                filter_boilerplate=filter_boilerplate,
                consistent_renaming=consistent_renaming,
                abstract_expressions=abstract_expressions,
                strip_docstrings=strip_docstrings,
            )

    if comprehensions:
        for comp in ast.walk(tree):
            if isinstance(
                comp,
                (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp),
            ):
                comp_name = type(comp).__name__.lower()
                enc_cls = enclosing_classes.get(id(comp))
                owner = func_owners.get(id(comp)) or enc_cls or "module"
                start_l = getattr(comp, "lineno", 0)
                _record_node_unit(
                    units,
                    f"{owner}:{comp_name}_L{start_l}",
                    rel_file,
                    comp,
                    min_lines=1,
                    min_tokens=max(8, min_tokens // 2),
                    kind="comprehension",
                    file_lines=file_lines,
                    blind_indexing=blind_indexing,
                    strip_annotations=strip_annotations,
                    blind_literals=blind_literals,
                    filter_boilerplate=filter_boilerplate,
                    consistent_renaming=consistent_renaming,
                    abstract_expressions=abstract_expressions,
                    strip_docstrings=strip_docstrings,
                    enclosing_class=enc_cls,
                    enclosing_class_start=enclosing_class_starts.get(id(comp)),
                )

    return units
