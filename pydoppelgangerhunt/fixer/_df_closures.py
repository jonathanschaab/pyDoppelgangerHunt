"""Lexical scope closure discovery, position classification, and capture analysis."""

from __future__ import annotations

import ast
import builtins
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _scope_closures_cache,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _BaseScopeVisitor,
    _collect_func_args,
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _is_ast_node_within_unit_bounds,
    _record_import_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _extract_node_type_param_names(node: ast.AST) -> Set[str]:
    """Extracts PEP 695 type parameter names declared directly on an AST node."""
    return {
        tp.name
        for tp in getattr(node, "type_params", [])
        if isinstance(getattr(tp, "name", None), str)
    }


class _ClosurePos(NamedTuple):
    """Line and column position of a closure for boundary matching."""

    line: int
    col: int = 0


_EAGER_CONSUMER_NAMES: frozenset[str] = frozenset(
    {"sum", "min", "max", "any", "all", "list", "set", "tuple", "dict", "sorted", "frozenset"}
)


_COMPREHENSION_TYPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

_INNER_SCOPE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _extract_comprehension_walrus_names(comp_node: ast.AST) -> Set[str]:
    """Extracts walrus target names that bind to the enclosing scope per PEP 572."""
    walrus_names: Set[str] = set()
    node_stack = list(ast.iter_child_nodes(comp_node))
    while node_stack:
        curr = node_stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            node_stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.NamedExpr) and isinstance(curr.target, ast.Name):
            walrus_names.add(curr.target.id)
        node_stack.extend(ast.iter_child_nodes(curr))
    return walrus_names


def _extract_single_node_local_names(
    scope_node: ast.AST,
    include_globals: bool = True,
    *,
    include_nonlocals: bool = True,
    include_self_name: bool = True,
    include_bare_annotations: Optional[bool] = None,
) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node."""
    names: Set[str] = set()
    global_names: Set[str] = set()
    node_name = getattr(scope_node, "name", None)
    if (
        include_self_name
        and isinstance(node_name, str)
        and not isinstance(scope_node, ast.ClassDef)
    ):
        names.add(node_name)

    names.update(_extract_node_type_param_names(scope_node))

    node_args = getattr(scope_node, "args", None)
    if isinstance(node_args, ast.arguments):
        names.update(_extract_arg_names(node_args))

    body = getattr(scope_node, "body", [])
    stack: List[ast.AST] = (
        list(body) if isinstance(body, list)
        else [body] if isinstance(body, ast.AST)
        else []
    )
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(curr.name)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            if not isinstance(scope_node, ast.ClassDef):
                names.update(_extract_comprehension_walrus_names(curr))
            continue
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.AnnAssign):
            stack.append(curr.annotation)
            if curr.value is not None:
                stack.append(curr.value)
                stack.append(curr.target)
            elif (
                include_bare_annotations
                if include_bare_annotations is not None
                else not isinstance(scope_node, (ast.ClassDef, ast.Module))
            ) and isinstance(curr.target, ast.Name):
                names.add(curr.target.id)
            elif not isinstance(curr.target, ast.Name):
                stack.append(curr.target)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Store):
            names.add(curr.id)
        elif isinstance(curr, ast.Import):
            for alias in curr.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if (
                    bound == "builtins"
                    and alias.name == "builtins"
                    and alias.asname in (None, "builtins")
                ):
                    continue
                if bound != "*":
                    names.add(bound)
            continue
        elif isinstance(curr, ast.ImportFrom):
            if curr.module in ("builtins", "__builtins__"):
                for alias in curr.names:
                    if alias.asname and alias.asname != alias.name:
                        names.add(alias.asname)
                continue
            _record_import_names(curr, names)
            continue
        elif isinstance(curr, ast.Global):
            if include_globals:
                names.update(curr.names)
            else:
                global_names.update(curr.names)
            continue
        elif isinstance(curr, ast.Nonlocal):
            if include_nonlocals:
                names.update(curr.names)
            else:
                global_names.update(curr.names)
            continue
        elif type(curr).__name__ == "TypeAlias":
            alias_name = getattr(curr, "name", None)
            if isinstance(alias_name, ast.Name):
                names.add(alias_name.id)
            continue
        else:
            bound_sym = getattr(curr, "name", None) or getattr(curr, "rest", None)
            if isinstance(bound_sym, str) and not isinstance(curr, ast.Attribute):
                names.add(bound_sym)

        stack.extend(ast.iter_child_nodes(curr))

    return names - global_names


def _extract_scope_local_names(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    include_classes: bool = True,
    include_self_name: bool = True,
) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node
    and all enclosing scopes up to module level."""
    names = (
        _extract_single_node_local_names(scope_node)
        if include_self_name
        else _extract_single_node_local_names(scope_node, include_self_name=False)
    )
    allowed_types = (
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        if include_classes
        else (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
    )
    if module_tree is not None and module_tree is not scope_node:
        parent_map = _get_scope_parent_map(module_tree)
        curr: Optional[ast.AST] = parent_map.get(scope_node)
        visited_module_tree = False
        while curr is not None:
            if curr is module_tree:
                visited_module_tree = True
            if isinstance(curr, allowed_types):
                names.update(_extract_single_node_local_names(curr))
            curr = parent_map.get(curr)
        if not visited_module_tree and isinstance(module_tree, allowed_types):
            names.update(_extract_single_node_local_names(module_tree))
    return names


def _is_confirmed_str_receiver(
    node: ast.AST,
    shadowed_names: Optional[Set[str]] = None,
) -> bool:
    """Verifies that an AST node is definitively a string receiver for .join calls."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Name) and node.id == "str":
        if shadowed_names is not None and "str" in shadowed_names:
            return False
        return True
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("builtins", "__builtins__")
        and node.attr == "str"
    ):
        if shadowed_names is not None and node.value.id in shadowed_names:
            return False
        return True
    return False


def _node_has_star_import(root: ast.AST) -> bool:
    """Checks whether an AST subtree contains a non-builtin wildcard star import."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module not in ("builtins", "__builtins__")
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(root)
    )


def _has_wildcard_star_import(
    scope_node: ast.AST, module_tree: Optional[ast.AST] = None
  ) -> bool:
    """Checks whether scope_node or any enclosing scope contains a wildcard star import."""
    if module_tree is not None:
        if _node_has_star_import(module_tree):
            return True
        if scope_node is not module_tree and scope_node not in _get_scope_parent_map(
            module_tree
        ):
            return _node_has_star_import(scope_node)
        return False
    return _node_has_star_import(scope_node)


def _is_eagerly_consumed_genexp(
    genexp: ast.GeneratorExp,
    parent_map: Dict[ast.AST, ast.AST],
    shadowed_names: Optional[Set[str]] = None,
    has_star_import: bool = False,
) -> bool:
    """Returns True if the generator expression is directly consumed by an eager built-in."""
    if has_star_import or (shadowed_names is not None and "*" in shadowed_names):
        return False
    parent = parent_map.get(genexp)
    if isinstance(parent, ast.Call):
        if any(isinstance(a, ast.Starred) for a in parent.args):
            return False
        func = parent.func
        if isinstance(func, ast.Name) and func.id in _EAGER_CONSUMER_NAMES:
            if shadowed_names is not None and func.id in shadowed_names:
                return False
            if func.id == "sum":
                return 1 <= len(parent.args) <= 2 and parent.args[0] is genexp
            return len(parent.args) == 1 and parent.args[0] is genexp
        if isinstance(func, ast.Attribute) and func.attr == "join":
            if not _is_confirmed_str_receiver(func.value, shadowed_names=shadowed_names):
                return False
            if isinstance(func.value, (ast.Constant, ast.JoinedStr)):
                return len(parent.args) == 1 and parent.args[0] is genexp
            return len(parent.args) == 2 and parent.args[1] is genexp
    return False


def _collect_scope_closures_uncached(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
) -> List[Tuple[_ClosurePos, Set[str]]]:
    closures: List[Tuple[_ClosurePos, Set[str]]] = []
    parent_map = _get_scope_parent_map(scope_node)
    shadowed_names = _extract_scope_local_names(scope_node, module_tree=module_tree)
    has_star_import = _has_wildcard_star_import(scope_node, module_tree=module_tree)

    def _resolve_stmt_start_pos(s: ast.AST) -> Tuple[int, int]:
        s_line = int(getattr(s, "lineno", 0) or 0)
        s_col = int(getattr(s, "col_offset", 0) or 0)
        decs = getattr(s, "decorator_list", [])
        if decs:
            dec_pos = []
            for d in decs:
                d_l = getattr(d, "lineno", None)
                d_c = getattr(d, "col_offset", None)
                dec_pos.append(
                    (
                        int(d_l) if d_l is not None else s_line,
                        int(d_c) if d_c is not None else s_col,
                    )
                )
            if dec_pos:
                min_l, min_c = min(dec_pos, key=lambda p: (p[0], p[1]))
                if min_l < s_line or (min_l == s_line and min_c < s_col):
                    s_line, s_col = min_l, min_c
        return s_line, s_col

    class _ExprClosureCollector(_BaseScopeVisitor):
        def __init__(
            self,
            default_line: int,
            default_col: int,
            bound_names: AbstractSet[str] = frozenset(),
        ) -> None:
            super().__init__()
            self.default_line = default_line
            self.default_col = default_col
            self.bound_names = bound_names

        def _record_closure_node(
            self,
            subnode: Union[ast.Lambda, ast.GeneratorExp],
            *,
            include_header: bool = True,
        ) -> None:
            sub_l = getattr(subnode, "lineno", None)
            sub_c = getattr(subnode, "col_offset", None)
            sub_line = int(sub_l) if sub_l is not None else self.default_line
            sub_col = int(sub_c) if sub_c is not None else self.default_col
            closures.append(
                (
                    _ClosurePos(sub_line, sub_col),
                    {
                        r
                        for r in _extract_nested_scope_free_reads(
                            subnode, include_header=include_header
                        )
                        if not self._is_in_comp(r) and r not in self.bound_names
                    },
                )
            )

        def visit_FunctionDef(self, node: ast.AST) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AST) -> None:
            return

        def visit_ClassDef(self, node: ast.AST) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            for default in node.args.defaults:
                self.visit(default)
            for kw_default in node.args.kw_defaults:
                if kw_default is not None:
                    self.visit(kw_default)
            self._record_closure_node(node, include_header=False)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            if _is_eagerly_consumed_genexp(
                node, parent_map, shadowed_names, has_star_import=has_star_import
            ):
                self._visit_comprehension(node)
            else:
                if node.generators:
                    self.visit(node.generators[0].iter)
                self._record_closure_node(node)

    def _record_expr_closures(
        nodes: Iterable[ast.AST],
        default_line: int,
        default_col: int,
        bound_names: AbstractSet[str] = frozenset(),
    ) -> None:
        collector = _ExprClosureCollector(default_line, default_col, bound_names)
        for node in nodes:
            collector.visit(node)

    def _extract_header_nodes(s: ast.AST) -> List[ast.AST]:
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
            return [getattr(s, "subject")]
        return [s]

    def _walk_stmts(
        stmts: Iterable[ast.AST],
        bound_names: AbstractSet[str] = frozenset(),
    ) -> None:
        for stmt in stmts:
            stmt_start, stmt_col = _resolve_stmt_start_pos(stmt)

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                before_len = len(closures)
                _record_expr_closures(
                    iter_nested_scope_header_nodes(stmt, include_annotations=False),
                    stmt_start,
                    stmt_col,
                    bound_names,
                )
                inner_bound = bound_names | _extract_node_type_param_names(stmt)
                fn_scoped_exprs = (
                    [
                        arg.annotation
                        for arg in _collect_func_args(stmt.args)
                        if arg.annotation is not None
                    ]
                    + ([stmt.returns] if stmt.returns is not None else [])
                    + list(getattr(stmt, "type_params", []))
                )
                _record_expr_closures(
                    fn_scoped_exprs, stmt_start, stmt_col, inner_bound
                )
                body_reads = (
                    _extract_nested_scope_free_reads(stmt, include_header=False)
                    - bound_names
                )
                if body_reads or len(closures) == before_len:
                    closures.append(
                        (_ClosurePos(stmt_start, stmt_col), body_reads)
                    )
                continue

            if isinstance(stmt, ast.ClassDef):
                _record_expr_closures(
                    stmt.decorator_list, stmt_start, stmt_col, bound_names
                )
                inner_bound = bound_names | _extract_node_type_param_names(stmt)
                class_scoped_exprs = (
                    list(stmt.bases)
                    + [kw.value for kw in stmt.keywords]
                    + list(getattr(stmt, "type_params", []))
                )
                _record_expr_closures(
                    class_scoped_exprs, stmt_start, stmt_col, inner_bound
                )
                _walk_stmts(stmt.body, inner_bound)
                continue

            header_bound = bound_names | _extract_node_type_param_names(stmt)
            _record_expr_closures(
                _extract_header_nodes(stmt), stmt_start, stmt_col, header_bound
            )

            for attr in ("body", "orelse", "finalbody"):
                sub_stmts = getattr(stmt, attr, None)
                if isinstance(sub_stmts, list):
                    _walk_stmts(sub_stmts, bound_names)

            # Intentional precision trade-off: treat ExceptHandler.name and match_case
            # pattern bindings as locally bound inside their respective blocks so closures
            # referencing them are not misreported as capturing prior outer variables.
            for handler in getattr(stmt, "handlers", []):
                h_name = getattr(handler, "name", None)
                h_bound = (
                    bound_names | {h_name}
                    if isinstance(h_name, str)
                    else bound_names
                )
                _walk_stmts(getattr(handler, "body", []), h_bound)

            for case in getattr(stmt, "cases", []):
                pattern = getattr(case, "pattern", None)
                case_bound = bound_names
                if isinstance(pattern, ast.AST):
                    _record_expr_closures([pattern], stmt_start, stmt_col, bound_names)
                    pattern_names = {
                        name
                        for sub in ast.walk(pattern)
                        for name in (_extract_pattern_bound_name(sub),)
                        if name is not None
                    }
                    if pattern_names:
                        case_bound = bound_names | pattern_names
                guard = getattr(case, "guard", None)
                if isinstance(guard, ast.AST):
                    _record_expr_closures([guard], stmt_start, stmt_col, case_bound)
                _walk_stmts(getattr(case, "body", []), case_bound)

    raw_scope_body = getattr(scope_node, "body", [])
    scope_stmts = (
        raw_scope_body if isinstance(raw_scope_body, list)
        else [raw_scope_body] if isinstance(raw_scope_body, ast.AST)
        else []
    )
    root_tp_names = frozenset(_extract_node_type_param_names(scope_node))
    _walk_stmts(scope_stmts, root_tp_names)
    return closures


def _collect_scope_closures(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
) -> List[Tuple[_ClosurePos, Set[str]]]:
    """Discovers all closures defined in a scope, recorded with their starting line."""
    cache_key: Any = scope_node if module_tree is None else (scope_node, module_tree)
    return _get_or_compute_scope_cached(
        _scope_closures_cache,
        cache_key,
        lambda: _collect_scope_closures_uncached(scope_node, module_tree=module_tree),
    )


def _normalize_closure_position(c_pos: Any) -> Tuple[int, int]:
    """Normalizes closure position into (line, col) integer coordinates."""
    if isinstance(c_pos, _ClosurePos):
        return c_pos.line, c_pos.col
    if isinstance(c_pos, (tuple, list)) and len(c_pos) >= 2:
        return int(c_pos[0]), int(c_pos[1])
    line = int(getattr(c_pos, "line", c_pos))
    col = int(getattr(c_pos, "col", 0))
    return line, col


def _classify_closure_position(
    c_pos: Any,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[bool, bool, bool]:
    """Classifies closure position as (is_pre, is_intra, is_post) relative to unit bounds.

    Same-line closures at boundary lines with unknown column offsets are conservatively
    treated as touching that boundary (pre and intra at u_start, intra and post at u_end)
    to fail closed across dataflow analysis.
    """
    c_line, c_col = _normalize_closure_position(c_pos)
    if c_line < u_start:
        return True, False, False
    if c_line > u_end:
        return False, False, True

    is_pre = (c_line == u_start and (u_start_col is None or c_col < u_start_col))
    is_post = (c_line == u_end and (u_end_col is None or c_col >= u_end_col))

    not_intra = (
        (c_line == u_start and u_start_col is not None and c_col < u_start_col)
        or (c_line == u_end and u_end_col is not None and c_col >= u_end_col)
    )
    is_intra = not not_intra
    return is_pre, is_intra, is_post


def _collect_pre_unit_closures(
    scope_node: ast.AST,
    u_start: int,
    candidates: Optional[Set[str]] = None,
    u_start_col: Optional[int] = None,
    module_tree: Optional[ast.AST] = None,
) -> Set[str]:
    """Discovers free variables captured by functions, class methods, and lambdas defined prior
    to the unit in the same lexical scope."""
    captured_reads: Set[str] = set()
    for c_pos, reads in _collect_scope_closures(scope_node, module_tree=module_tree):
        is_pre, _, _ = _classify_closure_position(
            c_pos, u_start, u_start, u_start_col=u_start_col
        )
        if is_pre:
            if candidates is not None:
                captured_reads.update(reads & candidates)
            else:
                captured_reads.update(reads - _BUILTIN_NAMES)
    return captured_reads


def _extract_scope_escaping_bindings(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int] = None,
    u_start_col: Optional[int] = None,
) -> Set[str]:
    """Extracts nonlocal and outer global names declared in a function scope outside unit bounds."""
    if not isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    nonlocal_names: Set[str] = set()
    outer_globals: Set[str] = set()
    intra_globals: Set[str] = set()
    body = getattr(scope_node, "body", [])
    stack: List[ast.AST] = (
        list(body)
        if isinstance(body, list)
        else [body]
        if isinstance(body, ast.AST)
        else []
    )
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Nonlocal):
            if not _is_ast_node_within_unit_bounds(
                curr, u_start, u_end, u_end_col, u_start_col
            ):
                nonlocal_names.update(curr.names)
            continue
        if isinstance(curr, ast.Global):
            if _is_ast_node_within_unit_bounds(
                curr, u_start, u_end, u_end_col, u_start_col
            ):
                intra_globals.update(curr.names)
            else:
                outer_globals.update(curr.names)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return nonlocal_names | (outer_globals - intra_globals)


def _node_earliest_start_lineno(node: ast.AST) -> int:
    """Computes the earliest starting line of an AST node including any decorators."""
    s_line = int(getattr(node, "lineno", 0) or 0)
    for dec in getattr(node, "decorator_list", []):
        d_line = getattr(dec, "lineno", None)
        if isinstance(d_line, int) and 0 < d_line < s_line:
            s_line = d_line
    return s_line


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int],
    is_subroutine: bool,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing FunctionDef/AsyncFunctionDef nodes ordered outermost to innermost."""
    chain: List[Tuple[int, int, int, ast.AST]] = []
    for node in ast.walk(mod_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        estart = _node_earliest_start_lineno(node)
        n_end = int(getattr(node, "end_lineno", None) or estart)
        if estart <= 0 or estart > u_start or u_end > n_end:
            continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
        if (
            not is_subroutine
            and u_start in (estart, int(getattr(node, "lineno", estart) or estart))
            and u_end == n_end
            and (u_start_col is None or n_col == u_start_col)
        ):
            continue
        chain.append((estart, n_end, n_col, node))
    chain.sort(key=lambda item: (item[0], -item[1], item[2]))
    return [(estart, n_end, node) for estart, n_end, _col, node in chain]


def _extract_closure_nonlocal_mutations(fn_node: ast.AST) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside fn_node."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    own_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
        include_bare_annotations=True,
    )
    direct_nonlocals: Set[str] = set()
    direct_mutated: Set[str] = set()
    nested_mutated: Set[str] = set()
    stack: List[ast.AST] = list(getattr(fn_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_mutated.update(
                _extract_closure_nonlocal_mutations(curr) - own_locals
            )
            continue
        if isinstance(curr, ast.ClassDef):
            for sub in curr.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested_mutated.update(
                        _extract_closure_nonlocal_mutations(sub) - own_locals
                    )
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            direct_mutated.update(_extract_comprehension_walrus_names(curr))
            continue
        if isinstance(curr, ast.Lambda):
            continue
        if isinstance(curr, ast.Nonlocal):
            direct_nonlocals.update(curr.names)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, (ast.Store, ast.Del)):
            direct_mutated.add(curr.id)
        stack.extend(ast.iter_child_nodes(curr))
    return (direct_nonlocals & direct_mutated) | nested_mutated


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    unit_top_locals: AbstractSet[str],
) -> List[Tuple[str, Set[str]]]:
    """Collects intra-unit nested functions that mutate unit_top_locals via nonlocal."""
    if not unit_top_locals:
        return []
    mutators: List[Tuple[str, Set[str]]] = []
    stack: List[ast.AST] = list(candidate_stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mutated = _extract_closure_nonlocal_mutations(curr) & set(unit_top_locals)
            if mutated:
                mutators.append((curr.name, mutated))
            continue
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return mutators


class _SpanBounds(NamedTuple):
    """Line and column span boundaries for enclosing function cell analysis."""

    start: int
    end: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None


class _EnclosingFnCellInfo(NamedTuple):
    """Cell binding and mutation summary for an enclosing function relative to a unit span."""

    fn_locals: Set[str]
    pre_bound: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]]
    all_sibling_nonlocal_mutations: Set[str]


def _is_pos_before_span(line: Optional[int], col: Optional[int], span: _SpanBounds) -> bool:
    """Checks whether a 1-based line and column precede the span start."""
    return bool(
        line is not None
        and line > 0
        and (
            line < span.start
            or (
                line == span.start
                and span.start_col is not None
                and col is not None
                and int(col) < int(span.start_col)
            )
        )
    )


def _is_pos_after_span(line: Optional[int], col: Optional[int], span: _SpanBounds) -> bool:
    """Checks whether a 1-based line and column follow the span end."""
    return bool(
        line is not None
        and line > 0
        and (
            line > span.end
            or (
                line == span.end
                and span.end_col is not None
                and col is not None
                and int(col) >= int(span.end_col)
            )
        )
    )


def _record_binding_at_pos(
    names: Iterable[str],
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    in_loop: bool,
    is_inside: bool,
    pre_bound: Set[str],
    post_mutated: Set[str],
    loop_mutated: Set[str],
) -> None:
    """Records bound variable names into pre_bound, post_mutated, and loop_mutated sets."""
    if _is_pos_before_span(line, col, span):
        pre_bound.update(names)
    if _is_pos_after_span(line, col, span):
        post_mutated.update(names)
    if in_loop and not is_inside:
        loop_mutated.update(names)


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


def _analyze_enclosing_fn_cells(
    fn_node: ast.AST,
    span: _SpanBounds,
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
) -> _EnclosingFnCellInfo:
    """Analyzes variable bindings, post-span mutations, loop mutations, and sibling closures."""
    fn_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=True,
        include_bare_annotations=True,
    )
    pre_bound: Set[str] = set(_extract_node_type_param_names(fn_node))
    fn_name = getattr(fn_node, "name", None)
    if isinstance(fn_name, str):
        pre_bound.add(fn_name)
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args))

    post_mutated: Set[str] = set()
    post_reads: Set[str] = set()
    loop_mutated: Set[str] = set()
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]] = []

    stack: List[Tuple[ast.AST, bool, bool]] = [
        (stmt, False, False) for stmt in reversed(getattr(fn_node, "body", []))
    ]
    while stack:
        curr, is_branched, in_loop = stack.pop()
        c_line = getattr(curr, "lineno", None)
        c_end = getattr(curr, "end_lineno", c_line)
        c_col = getattr(curr, "col_offset", None)
        is_inside = _is_ast_node_within_unit_bounds(
            curr, span.start, span.end, span.end_col, span.start_col
        )

        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            estart = _node_earliest_start_lineno(curr)
            _record_binding_at_pos(
                (curr.name,), estart, c_col, span, in_loop, is_inside,
                pre_bound, post_mutated, loop_mutated,
            )
            if not is_inside:
                free_r = _extract_nested_scope_free_reads(curr)
                nl_mut = _extract_closure_nonlocal_mutations(curr)
                sibling_closures.append((curr.name, free_r, nl_mut))
                if _is_pos_after_span(estart, c_col, span):
                    post_reads.update(free_r)
            stack.extend(
                (hdr, is_branched, in_loop)
                for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
            )
            continue

        if isinstance(curr, ast.ClassDef):
            estart = _node_earliest_start_lineno(curr)
            _record_binding_at_pos(
                (curr.name,), estart, c_col, span, in_loop, is_inside,
                pre_bound, post_mutated, loop_mutated,
            )
            if not is_inside:
                free_r = _extract_nested_scope_free_reads(curr)
                if _is_pos_after_span(estart, c_col, span):
                    post_reads.update(free_r)
                for sub in curr.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sibling_closures.append(
                            (
                                None,
                                _extract_nested_scope_free_reads(sub),
                                _extract_closure_nonlocal_mutations(sub),
                            )
                        )
            stack.extend(
                (hdr, is_branched, in_loop)
                for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
            )
            if isinstance(c_line, int) and isinstance(c_end, int) and estart <= span.start and span.end <= c_end:
                stack.extend((s, is_branched, in_loop) for s in reversed(curr.body))
            continue

        if isinstance(curr, ast.Lambda):
            if not is_inside:
                free_r = _extract_nested_scope_free_reads(curr)
                sibling_closures.append((None, free_r, set()))
                if _is_pos_after_span(c_line, c_col, span):
                    post_reads.update(free_r)
            stack.extend(
                (hdr, is_branched, in_loop)
                for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
            )
            continue

        if isinstance(curr, _COMPREHENSION_TYPES):
            w_names = _extract_comprehension_walrus_names(curr)
            _record_binding_at_pos(
                w_names, c_line, c_col, span, in_loop, is_inside,
                pre_bound, post_mutated, loop_mutated,
            )
            if not is_inside:
                free_r = _extract_nested_scope_free_reads(ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=curr))
                if isinstance(curr, ast.GeneratorExp):
                    sibling_closures.append((None, free_r, set()))
                if _is_pos_after_span(c_line, c_col, span):
                    post_reads.update(free_r)
            continue

        curr_in_loop = in_loop
        if isinstance(curr, (ast.For, ast.AsyncFor, ast.While)):
            if (
                isinstance(c_line, int)
                and isinstance(c_end, int)
                and c_line <= span.start
                and span.end <= c_end
                and (
                    (c_line, c_end) != (span.start, span.end)
                    or (span.start_col is not None and (c_col or 0) < span.start_col)
                )
            ):
                curr_in_loop = True
                loop_mutated.update(unit_mutations)
                if isinstance(curr, (ast.For, ast.AsyncFor)):
                    t_stores = _extract_store_names_in_node(curr.target)
                    pre_bound.update(t_stores)
                    loop_mutated.update(t_stores)

        if (
            isinstance(curr, (ast.With, ast.AsyncWith))
            and isinstance(c_line, int)
            and isinstance(c_end, int)
            and c_line <= span.start
            and span.end <= c_end
        ):
            for item in curr.items:
                if item.optional_vars is not None:
                    pre_bound.update(_extract_store_names_in_node(item.optional_vars))

        if isinstance(curr, ast.Name):
            if isinstance(curr.ctx, ast.Store):
                _record_binding_at_pos(
                    (curr.id,), c_line, c_col, span, curr_in_loop, is_inside,
                    pre_bound, post_mutated, loop_mutated,
                )
            elif isinstance(curr.ctx, ast.Del):
                if _is_pos_before_span(c_line, c_col, span) and not is_branched:
                    pre_bound.discard(curr.id)
                if _is_pos_after_span(c_line, c_col, span):
                    post_mutated.add(curr.id)
                if curr_in_loop and not is_inside:
                    loop_mutated.add(curr.id)
            elif isinstance(curr.ctx, ast.Load) and _is_pos_after_span(c_line, c_col, span):
                post_reads.add(curr.id)
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            imp_names: Set[str] = set()
            _record_import_names(curr, imp_names)
            _record_binding_at_pos(
                imp_names, c_line, c_col, span, curr_in_loop, is_inside,
                pre_bound, post_mutated, loop_mutated,
            )
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            if isinstance(c_line, int) and isinstance(c_end, int) and c_line <= span.start and span.end <= c_end:
                pre_bound.add(curr.name)
            if _is_pos_after_span(c_line, c_col, span):
                post_mutated.add(curr.name)
            if curr_in_loop and not is_inside:
                loop_mutated.add(curr.name)
        elif type(curr).__name__ == "TypeAlias":
            alias_name = getattr(curr, "name", None)
            if isinstance(alias_name, ast.Name):
                _record_binding_at_pos(
                    (alias_name.id,), c_line, c_col, span, curr_in_loop, is_inside,
                    pre_bound, post_mutated, loop_mutated,
                )
        elif type(curr).__name__ in ("MatchAs", "MatchStar", "MatchMapping"):
            m_name = _extract_pattern_bound_name(curr)
            if m_name is not None:
                _record_binding_at_pos(
                    (m_name,), c_line, c_col, span, curr_in_loop, is_inside,
                    pre_bound, post_mutated, loop_mutated,
                )

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
        child_nodes = (
            [curr.annotation]
            if isinstance(curr, ast.AnnAssign)
            and curr.value is None
            and isinstance(curr.target, ast.Name)
            else list(ast.iter_child_nodes(curr))
        )
        stack.extend(
            (ch, child_branched, curr_in_loop)
            for ch in reversed(child_nodes)
        )

    all_sib_nl = (
        set().union(*(nl for _, _, nl in sibling_closures))
        if sibling_closures
        else set()
    )
    return _EnclosingFnCellInfo(
        fn_locals=fn_locals,
        pre_bound=pre_bound,
        post_mutated=post_mutated,
        post_reads=post_reads,
        loop_mutated=loop_mutated,
        sibling_closures=sibling_closures,
        all_sibling_nonlocal_mutations=all_sib_nl,
    )


class _UnitClosureHazardContext(NamedTuple):
    """Input context for checking whether extracting a unit severs closure cell bindings."""

    candidate_stmts: Sequence[ast.stmt]
    mod_tree: Optional[ast.AST]
    u_start: int
    u_end: int
    u_start_col: Optional[int]
    u_end_col: Optional[int]
    is_subroutine: bool
    unit_kind: Optional[str]
    inputs: Sequence[str]
    outputs: Sequence[str]
    stores: Sequence[str]
    deleted_names: AbstractSet[str]
    unit_top_locals: AbstractSet[str]


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings."""
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return False
    if not ctx.is_subroutine and ctx.unit_kind != "closure":
        return False

    enclosing_chain = _find_enclosing_function_chain(
        ctx.mod_tree, ctx.u_start, ctx.u_end, ctx.u_start_col, ctx.is_subroutine
    )
    if not enclosing_chain:
        return False

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    unit_closures = _collect_scope_closures_uncached(unit_mod)
    intra_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures))
        if unit_closures
        else set()
    )
    intra_nl_mutators = (
        _collect_unit_nonlocal_mutators(ctx.candidate_stmts, ctx.unit_top_locals)
        if ctx.is_subroutine
        else []
    )
    input_set = set(ctx.inputs)
    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    unit_mutations = (store_set | set(ctx.deleted_names)) if ctx.is_subroutine else set()

    if not intra_closure_reads and not intra_nl_mutators and not ctx.is_subroutine:
        return False

    chain_infos: List[_EnclosingFnCellInfo] = []
    for idx, (_estart, _nend, fn_node) in enumerate(enclosing_chain):
        if idx == len(enclosing_chain) - 1:
            span = _SpanBounds(ctx.u_start, ctx.u_end, ctx.u_start_col, ctx.u_end_col)
            u_mut = unit_mutations
        else:
            next_start, next_end, next_fn = enclosing_chain[idx + 1]
            span = _SpanBounds(
                next_start,
                next_end,
                getattr(next_fn, "col_offset", None),
                getattr(next_fn, "end_col_offset", None),
            )
            u_mut = set()
        chain_infos.append(
            _analyze_enclosing_fn_cells(fn_node, span, unit_mutations=u_mut)
        )

    # 1 & 2: Intra-unit closure captures an enclosing cell variable that is unbound before
    # the unit, mutated/deleted after the unit, mutated in an enclosing loop, or mutated
    # via nonlocal in a sibling closure.
    captured_candidates = intra_closure_reads & (
        input_set | ((store_set | output_set) if ctx.is_subroutine else set())
    )
    for var in captured_candidates:
        for idx in range(len(chain_infos) - 1, -1, -1):
            info = chain_infos[idx]
            if var not in info.fn_locals:
                continue
            ext_muts = (
                info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            if (var in input_set and var not in info.pre_bound) or var in ext_muts:
                return True
            break

    # 3: Subroutine intra-unit closure mutates a subroutine top-level variable via nonlocal
    # that is also bound/mutated/read in the enclosing function outside the subroutine.
    if intra_nl_mutators:
        inner_info = chain_infos[-1]
        outer_touched = (
            inner_info.pre_bound | inner_info.post_mutated | inner_info.loop_mutated
        )
        for mut_fn_name, mutated_vars in intra_nl_mutators:
            for var in mutated_vars:
                if var in outer_touched or (
                    var in inner_info.post_reads and mut_fn_name in inner_info.post_reads
                ):
                    return True

    # 4: Sibling closure in an enclosing function is passed into the subroutine as an input
    # and shares a mutated cell variable with the subroutine.
    if ctx.is_subroutine:
        sub_mutated = store_set | set(ctx.deleted_names)
        sub_touched = input_set | sub_mutated
        for info in chain_infos:
            for sib_name, sib_reads, sib_nl_muts in info.sibling_closures:
                if sib_name is None or sib_name not in input_set:
                    continue
                if bool(sib_reads & info.fn_locals & sub_mutated):
                    return True
                if bool(sib_nl_muts & info.fn_locals & sub_touched):
                    return True

    return False
