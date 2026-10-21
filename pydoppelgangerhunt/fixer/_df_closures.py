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


def _extract_scope_nonlocals_and_outer_globals(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int] = None,
    u_start_col: Optional[int] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (nonlocal_names, outer_only_globals) declared in a function scope."""
    if not isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set(), set()
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
    return nonlocal_names, outer_globals - intra_globals


def _extract_scope_escaping_bindings(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_end_col: Optional[int] = None,
    u_start_col: Optional[int] = None,
) -> Set[str]:
    """Extracts nonlocal and outer global names declared in a function scope."""
    nonlocal_names, outer_globals = _extract_scope_nonlocals_and_outer_globals(
        scope_node, u_start, u_end, u_end_col=u_end_col, u_start_col=u_start_col
    )
    return nonlocal_names | outer_globals


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
    is_subroutine: bool = False,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing FunctionDef/AsyncFunctionDef nodes ordered outermost to innermost."""
    _ = is_subroutine
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
            u_start in (estart, int(getattr(node, "lineno", estart) or estart))
            and u_end == n_end
            and (u_start_col is None or n_col == u_start_col)
        ):
            continue
        chain.append((estart, n_end, n_col, node))
    chain.sort(key=lambda item: (item[0], -item[1], item[2]))
    return [(estart, n_end, node) for estart, n_end, _col, node in chain]


def _extract_direct_nonlocals(scope_node: ast.AST) -> Set[str]:
    """Extracts variable names declared nonlocal directly in scope_node."""
    names: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Nonlocal):
            names.update(curr.names)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _extract_closure_nonlocal_mutations(scope_node: ast.AST) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set()
    is_cls = isinstance(scope_node, ast.ClassDef)
    own_locals = (
        set()
        if is_cls
        else _extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        )
    )
    direct_nonlocals: Set[str] = set()
    direct_mutated: Set[str] = set()
    nested_mutated: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            direct_mutated.add(curr.name)
            nested_mutated.update(
                _extract_closure_nonlocal_mutations(curr) - own_locals
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
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            _record_import_names(curr, direct_mutated)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            direct_mutated.add(curr.name)
        elif type(curr).__name__ == "TypeAlias" and isinstance(
            getattr(curr, "name", None), ast.Name
        ):
            direct_mutated.add(curr.name.id)  # type: ignore[attr-defined]
        elif type(curr).__name__ in ("MatchAs", "MatchStar", "MatchMapping"):
            m_name = _extract_pattern_bound_name(curr)
            if m_name is not None:
                direct_mutated.add(m_name)
        stack.extend(ast.iter_child_nodes(curr))
    return (direct_nonlocals & direct_mutated) | nested_mutated


def _extract_expr_scope_free_reads(expr: ast.expr) -> Set[str]:
    """Extracts free variable reads from a Lambda or comprehension expression."""
    if isinstance(expr, ast.Lambda):
        return _extract_nested_scope_free_reads(expr)
    return _extract_nested_scope_free_reads(
        ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=expr,
        )
    )


def _extract_lazy_genexp_walrus_names(node: ast.AST) -> Set[str]:
    """Extracts walrus target names inside lazy GeneratorExp subtrees of node."""
    if isinstance(node, ast.GeneratorExp):
        return _extract_comprehension_walrus_names(node)
    names: Set[str] = set()
    stack: List[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.GeneratorExp):
            names.update(_extract_comprehension_walrus_names(curr))
            continue
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
) -> List[Tuple[str, Set[str], bool]]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    mutators: List[Tuple[str, Set[str], bool]] = []
    stack: List[ast.AST] = list(candidate_stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = _extract_closure_nonlocal_mutations(curr)
            if mutated:
                is_escaped_def = isinstance(curr, ast.ClassDef) or bool(
                    getattr(curr, "decorator_list", None)
                )
                mutators.append((curr.name, mutated, is_escaped_def))
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.GeneratorExp):
            w_names = _extract_comprehension_walrus_names(curr)
            if w_names:
                mutators.append(("", w_names, True))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
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
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]]
    all_sibling_nonlocal_mutations: Set[str]
    alias_edges: Sequence[Tuple[str, Set[str]]] = ()


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


def _is_node_inside_span(node: ast.AST, span: _SpanBounds) -> bool:
    """Checks whether an AST node's start and end positions lie within span."""
    if not _is_ast_node_within_unit_bounds(
        node, span.start, span.end, span.end_col, span.start_col
    ):
        return False
    c_end = getattr(node, "end_lineno", None)
    if isinstance(c_end, int):
        if c_end > span.end:
            return False
        if c_end == span.end and span.end_col is not None:
            end_col = getattr(node, "end_col_offset", None)
            if isinstance(end_col, int) and end_col > span.end_col:
                return False
    return True


def _strictly_encloses_span(node: ast.AST, span: _SpanBounds) -> bool:
    """Checks whether an AST compound node strictly encloses span."""
    c_line = getattr(node, "lineno", None)
    c_end = getattr(node, "end_lineno", c_line)
    if not (
        isinstance(c_line, int)
        and isinstance(c_end, int)
        and c_line <= span.start
        and span.end <= c_end
    ):
        return False
    if (c_line, c_end) != (span.start, span.end):
        return True
    c_col = getattr(node, "col_offset", None)
    if (
        span.start_col is not None
        and isinstance(c_col, int)
        and c_col < span.start_col
    ):
        return True
    c_end_col = getattr(node, "end_col_offset", None)
    if (
        span.end_col is not None
        and isinstance(c_end_col, int)
        and c_end_col > span.end_col
    ):
        return True
    return False


def _block_contains_span(stmts: Sequence[ast.AST], span: _SpanBounds) -> bool:
    """Checks whether any statement in stmts lies inside or strictly encloses span."""
    return any(
        _is_node_inside_span(s, span) or _strictly_encloses_span(s, span)
        for s in stmts
    )


def _can_bind_in_fn_scope(name: str, active_nonlocals: Optional[Set[str]]) -> bool:
    """Checks whether a binding in the current AST frame targets the enclosing function scope."""
    return active_nonlocals is None or name in active_nonlocals


def _extract_root_target_names(target: ast.AST) -> Set[str]:
    """Extracts root variable names written or mutated by an assignment target."""
    names: Set[str] = set()
    stack: List[ast.AST] = [target]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.Name):
            names.add(curr.id)
        elif isinstance(curr, (ast.Tuple, ast.List)):
            stack.extend(curr.elts)
        elif isinstance(curr, ast.Starred):
            stack.append(curr.value)
        elif isinstance(curr, (ast.Attribute, ast.Subscript)):
            stack.append(curr.value)
    return names


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions."""
    inline_reads: Set[str] = set()
    inline_muts: Set[str] = set()
    loaded_names: Set[str] = set()
    stack: List[ast.AST] = list(nodes)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(curr, ast.Lambda):
            inline_reads.update(_extract_expr_scope_free_reads(curr))
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            if isinstance(curr, ast.GeneratorExp) or any(
                isinstance(sub, (ast.Lambda, ast.GeneratorExp))
                for sub in ast.walk(curr)
            ):
                inline_reads.update(_extract_expr_scope_free_reads(curr))
                inline_muts.update(_extract_lazy_genexp_walrus_names(curr))
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded_names.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return inline_reads, inline_muts, loaded_names


def _extract_closure_carrier_targets_and_values(
    node: ast.AST,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (target_root_names, value_nodes) from assignments or mutating method calls."""
    if isinstance(node, ast.Assign) and node.value is not None:
        targets: Set[str] = set()
        for tgt in node.targets:
            targets.update(_extract_root_target_names(tgt))
        return targets, (node.value,)
    if (
        isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        and node.value is not None
    ):
        return _extract_root_target_names(node.target), (node.value,)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        val_nodes: Sequence[ast.AST] = [
            *node.args,
            *(kw.value for kw in node.keywords if kw.value is not None),
        ]
        if val_nodes:
            return _extract_root_target_names(node.func.value), val_nodes
    return None


class _CellBindingState(NamedTuple):
    """Mutable sets tracking cell bindings during enclosing function AST traversal."""

    pre_bound: Set[str]
    pre_mutated: Set[str]
    post_mutated: Set[str]
    post_reads: Set[str]
    loop_mutated: Set[str]


class _StackFrame(NamedTuple):
    """Traversal frame for _analyze_enclosing_fn_cells."""

    node: ast.AST
    is_branched: bool
    in_loop: bool
    active_nonlocals: Optional[Set[str]]


def _record_binding_at_pos(
    names: Iterable[str],
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
) -> None:
    """Records bound variable names into pre_bound, pre_mutated, post_mutated, and loop_mutated."""
    filtered = [
        n for n in names if _can_bind_in_fn_scope(n, frame.active_nonlocals)
    ]
    if not filtered:
        return
    if _is_pos_before_span(line, col, span):
        state.pre_mutated.update(filtered)
        if not frame.is_branched:
            state.pre_bound.update(filtered)
    if _is_pos_after_span(line, col, span):
        state.post_mutated.update(filtered)
    if frame.in_loop:
        state.loop_mutated.update(filtered)


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


def _find_non_call_loaded_names(stmts: Sequence[ast.stmt]) -> Set[str]:
    """Finds variable names loaded outside direct Call.func positions at unit top level."""
    loaded: Set[str] = set()
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Call):
            if not isinstance(curr.func, ast.Name):
                stack.append(curr.func)
            stack.extend(curr.args)
            stack.extend(curr.keywords)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return loaded


def _propagate_sibling_closures(
    closures: Sequence[Tuple[Optional[str], Set[str], Set[str]]],
    post_reads: Optional[Set[str]] = None,
    alias_edges: Sequence[Tuple[str, Set[str]]] = (),
) -> List[Tuple[Optional[str], Set[str], Set[str]]]:
    """Propagates free reads and nonlocal mutations transitively across named sibling closures."""
    result: List[Tuple[Optional[str], Set[str], Set[str]]] = [
        (name, set(reads), set(muts)) for name, reads, muts in closures
    ]
    by_name: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for name, reads, muts in result:
        if name is not None:
            if name not in by_name:
                by_name[name] = (set(reads), set(muts))
            else:
                by_name[name][0].update(reads)
                by_name[name][1].update(muts)
    changed = True
    while changed:
        changed = False
        for name, reads, muts in result:
            for ref in list(reads):
                target = by_name.get(ref)
                if target is None:
                    continue
                t_reads, t_muts = target
                if not t_reads.issubset(reads):
                    reads.update(t_reads)
                    changed = True
                if not t_muts.issubset(muts):
                    muts.update(t_muts)
                    changed = True
            if name is not None:
                by_name[name][0].update(reads)
                by_name[name][1].update(muts)
        for tgt_name, src_names in alias_edges:
            for src in src_names:
                target = by_name.get(src)
                if target is None:
                    continue
                t_reads, t_muts = target
                if tgt_name not in by_name:
                    by_name[tgt_name] = (set(t_reads), set(t_muts))
                    result.append((tgt_name, set(t_reads), set(t_muts)))
                    changed = True
                else:
                    if not t_reads.issubset(by_name[tgt_name][0]):
                        by_name[tgt_name][0].update(t_reads)
                        changed = True
                    if not t_muts.issubset(by_name[tgt_name][1]):
                        by_name[tgt_name][1].update(t_muts)
                        changed = True
        for name, reads, muts in result:
            if name is not None and name in by_name:
                reads.update(by_name[name][0])
                muts.update(by_name[name][1])
    if post_reads is not None:
        for ref in list(post_reads):
            target = by_name.get(ref)
            if target is not None:
                post_reads.update(target[0])
    return result


def _push_seq(
    stack: List[_StackFrame],
    nodes: Sequence[ast.AST],
    is_branched: bool,
    in_loop: bool,
    active_nonlocals: Optional[Set[str]],
) -> None:
    """Pushes a sequence of AST nodes in reverse order onto the traversal stack."""
    for node in reversed(nodes):
        stack.append(_StackFrame(node, is_branched, in_loop, active_nonlocals))


def _push_compound_children(
    curr: ast.AST,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
) -> bool:
    """Pushes child nodes of compound control-flow statements with branch and loop state."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(stack, curr.body, is_br if body_span else True, in_lp, act_nl)
        stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, (ast.For, ast.AsyncFor)):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        if body_span:
            state.loop_mutated.update(unit_mutations)
            t_stores = [
                n
                for n in _extract_store_names_in_node(curr.target)
                if _can_bind_in_fn_scope(n, act_nl)
            ]
            state.pre_mutated.update(t_stores)
            if not is_br:
                state.pre_bound.update(t_stores)
            state.loop_mutated.update(t_stores)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(
            stack,
            curr.body,
            is_br if body_span else True,
            True if body_span else in_lp,
            act_nl,
        )
        stack.append(
            _StackFrame(
                curr.target,
                is_br if body_span else True,
                True if body_span else in_lp,
                act_nl,
            )
        )
        stack.append(_StackFrame(curr.iter, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.While):
        body_span = encloses and (
            _block_contains_span(curr.body, span)
            or _is_node_inside_span(curr.test, span)
        )
        else_span = encloses and _block_contains_span(curr.orelse, span)
        if body_span:
            state.loop_mutated.update(unit_mutations)
        _push_seq(stack, curr.orelse, is_br if else_span else True, in_lp, act_nl)
        _push_seq(
            stack,
            curr.body,
            is_br if body_span else True,
            True if body_span else in_lp,
            act_nl,
        )
        stack.append(
            _StackFrame(curr.test, is_br, True if body_span else in_lp, act_nl)
        )
        return True

    if isinstance(curr, (ast.With, ast.AsyncWith)):
        if encloses:
            for item in curr.items:
                if item.optional_vars is not None:
                    w_stores = [
                        n
                        for n in _extract_store_names_in_node(item.optional_vars)
                        if _can_bind_in_fn_scope(n, act_nl)
                    ]
                    state.pre_mutated.update(w_stores)
                    if not is_br:
                        state.pre_bound.update(w_stores)
        _push_seq(stack, curr.body, is_br, in_lp, act_nl)
        _push_seq(stack, curr.items, is_br, in_lp, act_nl)
        return True

    if isinstance(curr, ast.Try) or type(curr).__name__ == "TryStar":
        body_l = getattr(curr, "body", [])
        else_l = getattr(curr, "orelse", [])
        final_l = getattr(curr, "finalbody", [])
        handlers_l = getattr(curr, "handlers", [])
        else_span = encloses and _block_contains_span(else_l, span)
        body_span = encloses and (
            _block_contains_span(body_l, span) or else_span
        )
        final_span = encloses and _block_contains_span(final_l, span)
        _push_seq(
            stack,
            final_l,
            is_br if (not encloses or final_span) else True,
            in_lp,
            act_nl,
        )
        _push_seq(stack, else_l, is_br if else_span else True, in_lp, act_nl)
        _push_seq(stack, handlers_l, is_br if encloses else True, in_lp, act_nl)
        _push_seq(stack, body_l, is_br if body_span else True, in_lp, act_nl)
        return True

    if isinstance(curr, ast.ExceptHandler):
        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)
        h_br = is_br if encloses else True
        if isinstance(curr.name, str) and _can_bind_in_fn_scope(curr.name, act_nl):
            if encloses:
                state.pre_mutated.add(curr.name)
                state.post_mutated.add(curr.name)
                if not h_br:
                    state.pre_bound.add(curr.name)
            elif _is_pos_before_span(c_line, c_col, span):
                state.pre_bound.discard(curr.name)
                state.pre_mutated.add(curr.name)
            if _is_pos_after_span(c_line, c_col, span):
                state.post_mutated.add(curr.name)
            if in_lp:
                state.loop_mutated.add(curr.name)
        _push_seq(stack, curr.body, h_br, in_lp, act_nl)
        if curr.type is not None:
            stack.append(_StackFrame(curr.type, True, in_lp, act_nl))
        return True

    if type(curr).__name__ == "Match":
        for case in reversed(getattr(curr, "cases", [])):
            case_body = getattr(case, "body", [])
            c_br = is_br if (encloses and _block_contains_span(case_body, span)) else True
            _push_seq(stack, case_body, c_br, in_lp, act_nl)
            guard = getattr(case, "guard", None)
            if isinstance(guard, ast.AST):
                stack.append(_StackFrame(guard, c_br, in_lp, act_nl))
            pattern = getattr(case, "pattern", None)
            if isinstance(pattern, ast.AST):
                stack.append(_StackFrame(pattern, c_br, in_lp, act_nl))
        subject = getattr(curr, "subject", None)
        if isinstance(subject, ast.AST):
            stack.append(_StackFrame(subject, is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.BoolOp) and curr.values:
        _push_seq(stack, curr.values[1:], True, in_lp, act_nl)
        stack.append(_StackFrame(curr.values[0], is_br, in_lp, act_nl))
        return True

    if isinstance(curr, ast.IfExp):
        stack.append(_StackFrame(curr.orelse, True, in_lp, act_nl))
        stack.append(_StackFrame(curr.body, True, in_lp, act_nl))
        stack.append(_StackFrame(curr.test, is_br, in_lp, act_nl))
        return True

    return False


def _analyze_enclosing_fn_cells(
    fn_node: ast.AST,
    span: _SpanBounds,
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
    enclosed_fn: Optional[ast.AST] = None,
) -> _EnclosingFnCellInfo:
    """Analyzes variable bindings, post-span mutations, loop mutations, and sibling closures."""
    fn_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
        include_bare_annotations=True,
    )
    pre_bound: Set[str] = set(_extract_node_type_param_names(fn_node))
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args))

    state = _CellBindingState(
        pre_bound=pre_bound,
        pre_mutated=set(pre_bound),
        post_mutated=set(),
        post_reads=set(),
        loop_mutated=set(),
    )
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]] = []
    alias_edges: List[Tuple[str, Set[str]]] = []

    stack: List[_StackFrame] = [
        _StackFrame(stmt, False, False, None)
        for stmt in reversed(getattr(fn_node, "body", []))
    ]
    while stack:
        frame = stack.pop()
        curr = frame.node
        c_line = getattr(curr, "lineno", None)
        c_col = getattr(curr, "col_offset", None)

        if curr is enclosed_fn and isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            estart = _node_earliest_start_lineno(curr)
            _record_binding_at_pos(
                (curr.name,), estart, c_col, span, frame, state
            )
            _push_seq(
                stack,
                list(iter_nested_scope_header_nodes(curr)),
                frame.is_branched,
                frame.in_loop,
                frame.active_nonlocals,
            )
            continue

        if _is_node_inside_span(curr, span):
            if isinstance(
                curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                pre_hdrs = [
                    hdr
                    for hdr in iter_nested_scope_header_nodes(curr)
                    if _is_pos_before_span(
                        getattr(hdr, "lineno", None),
                        getattr(hdr, "col_offset", None),
                        span,
                    )
                ]
                _push_seq(
                    stack,
                    pre_hdrs,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
            continue

        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            estart = _node_earliest_start_lineno(curr)
            _record_binding_at_pos(
                (curr.name,), estart, c_col, span, frame, state
            )
            free_r = _extract_nested_scope_free_reads(curr)
            nl_mut = _extract_closure_nonlocal_mutations(curr)
            sib_name = (
                curr.name
                if _can_bind_in_fn_scope(curr.name, frame.active_nonlocals)
                else None
            )
            sibling_closures.append((sib_name, free_r, nl_mut))
            if _is_pos_after_span(estart, c_col, span):
                state.post_reads.update(free_r)
            if isinstance(curr, ast.ClassDef):
                _push_seq(
                    stack,
                    curr.body,
                    frame.is_branched,
                    frame.in_loop,
                    _extract_direct_nonlocals(curr),
                )
            _push_seq(
                stack,
                list(iter_nested_scope_header_nodes(curr)),
                frame.is_branched,
                frame.in_loop,
                frame.active_nonlocals,
            )
            continue

        if isinstance(curr, ast.Lambda):
            free_r = _extract_expr_scope_free_reads(curr)
            sibling_closures.append((None, free_r, set()))
            if _is_pos_after_span(c_line, c_col, span):
                state.post_reads.update(free_r)
            _push_seq(
                stack,
                list(iter_nested_scope_header_nodes(curr)),
                frame.is_branched,
                frame.in_loop,
                frame.active_nonlocals,
            )
            continue

        if isinstance(curr, _COMPREHENSION_TYPES):
            w_names = _extract_comprehension_walrus_names(curr)
            comp_frame = (
                frame._replace(is_branched=True)
                if isinstance(curr, ast.GeneratorExp)
                else frame
            )
            _record_binding_at_pos(w_names, c_line, c_col, span, comp_frame, state)
            free_r = _extract_expr_scope_free_reads(curr)
            if isinstance(curr, ast.GeneratorExp) or any(
                isinstance(sub, (ast.Lambda, ast.GeneratorExp))
                for sub in ast.walk(curr)
            ):
                sibling_closures.append(
                    (None, free_r, _extract_lazy_genexp_walrus_names(curr))
                )
            if _is_pos_after_span(c_line, c_col, span):
                state.post_reads.update(free_r)
            continue

        carrier = _extract_closure_carrier_targets_and_values(curr)
        if carrier is not None:
            c_targets, c_values = carrier
            valid_targets = [
                t
                for t in c_targets
                if _can_bind_in_fn_scope(t, frame.active_nonlocals)
            ]
            if valid_targets:
                c_reads, c_muts, c_loads = _extract_expr_closure_and_load_info(
                    c_values
                )
                for t_name in valid_targets:
                    if c_reads or c_muts:
                        sibling_closures.append(
                            (t_name, set(c_reads), set(c_muts))
                        )
                    if c_loads:
                        alias_edges.append((t_name, set(c_loads)))

        if _push_compound_children(
            curr, span, frame, unit_mutations, state, stack
        ):
            continue

        if isinstance(curr, ast.AugAssign) and isinstance(curr.target, ast.Name):
            if _is_pos_after_span(c_line, c_col, span):
                state.post_reads.add(curr.target.id)

        if isinstance(curr, ast.Name):
            if isinstance(curr.ctx, ast.Store):
                _record_binding_at_pos(
                    (curr.id,), c_line, c_col, span, frame, state
                )
            elif isinstance(curr.ctx, ast.Del) and _can_bind_in_fn_scope(
                curr.id, frame.active_nonlocals
            ):
                if _is_pos_before_span(c_line, c_col, span):
                    state.pre_bound.discard(curr.id)
                    state.pre_mutated.add(curr.id)
                if _is_pos_after_span(c_line, c_col, span):
                    state.post_mutated.add(curr.id)
                if frame.in_loop:
                    state.loop_mutated.add(curr.id)
            elif isinstance(curr.ctx, ast.Load) and _is_pos_after_span(
                c_line, c_col, span
            ):
                state.post_reads.add(curr.id)
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            imp_names: Set[str] = set()
            _record_import_names(curr, imp_names)
            _record_binding_at_pos(
                imp_names, c_line, c_col, span, frame, state
            )
        elif type(curr).__name__ == "TypeAlias":
            alias_name = getattr(curr, "name", None)
            if isinstance(alias_name, ast.Name):
                _record_binding_at_pos(
                    (alias_name.id,), c_line, c_col, span, frame, state
                )
        elif type(curr).__name__ in ("MatchAs", "MatchStar", "MatchMapping"):
            m_name = _extract_pattern_bound_name(curr)
            if m_name is not None:
                _record_binding_at_pos(
                    (m_name,), c_line, c_col, span, frame, state
                )

        child_nodes: Sequence[ast.AST] = (
            (curr.annotation,)
            if isinstance(curr, ast.AnnAssign)
            and curr.value is None
            and isinstance(curr.target, ast.Name)
            else tuple(ast.iter_child_nodes(curr))
        )
        _push_seq(
            stack,
            child_nodes,
            frame.is_branched,
            frame.in_loop,
            frame.active_nonlocals,
        )

    sibling_closures = _propagate_sibling_closures(
        sibling_closures,
        post_reads=state.post_reads,
        alias_edges=alias_edges,
    )
    all_sib_nl = (
        set().union(*(nl for _, _, nl in sibling_closures))
        if sibling_closures
        else set()
    )
    return _EnclosingFnCellInfo(
        fn_locals=fn_locals,
        pre_bound=state.pre_bound,
        pre_mutated=state.pre_mutated,
        post_mutated=state.post_mutated,
        post_reads=state.post_reads,
        loop_mutated=state.loop_mutated,
        sibling_closures=sibling_closures,
        all_sibling_nonlocal_mutations=all_sib_nl,
        alias_edges=alias_edges,
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
    free_vars: Sequence[str] = ()
    nonlocals: Sequence[str] = ()
    has_yield: bool = False
    is_async: bool = False


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

    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    nonlocal_set = set(ctx.nonlocals)
    unit_mutated_names = store_set | set(ctx.deleted_names)
    if not ctx.is_subroutine and bool(nonlocal_set & unit_mutated_names):
        return True

    enc_nonlocals, enc_outer_globals = _extract_scope_nonlocals_and_outer_globals(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_end_col=ctx.u_end_col,
        u_start_col=ctx.u_start_col,
    )
    enc_escaping = enc_nonlocals | enc_outer_globals
    if ctx.is_subroutine and bool(unit_mutated_names & enc_outer_globals):
        return True
    intra_nl_mutators = _collect_unit_nonlocal_mutators(ctx.candidate_stmts)
    if ctx.is_subroutine and (ctx.has_yield or ctx.is_async):
        if bool(unit_mutated_names & (enc_escaping | nonlocal_set)) or bool(
            intra_nl_mutators
        ):
            return True

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    unit_closures = _collect_scope_closures_uncached(unit_mod)
    intra_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures))
        if unit_closures
        else set()
    )
    intra_closure_reads.update(nonlocal_set)
    if ctx.is_subroutine:
        intra_closure_reads.update(enc_escaping & unit_mutated_names)
        if ctx.has_yield or ctx.is_async:
            intra_closure_reads.update(ctx.inputs)
    else:
        intra_closure_reads.update(ctx.free_vars)

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    unit_mutations = unit_mutated_names if ctx.is_subroutine else set()

    if not intra_closure_reads and not intra_nl_mutators and not ctx.is_subroutine:
        return False

    chain_infos: List[_EnclosingFnCellInfo] = []
    for idx, (_estart, _nend, fn_node) in enumerate(enclosing_chain):
        if idx == len(enclosing_chain) - 1:
            span = _SpanBounds(ctx.u_start, ctx.u_end, ctx.u_start_col, ctx.u_end_col)
            u_mut = unit_mutations
            enc_child: Optional[ast.AST] = None
        else:
            next_start, next_end, next_fn = enclosing_chain[idx + 1]
            next_body = getattr(next_fn, "body", [])
            first_stmt = next_body[0] if next_body else next_fn
            span = _SpanBounds(
                int(getattr(first_stmt, "lineno", next_start) or next_start),
                next_end,
                getattr(first_stmt, "col_offset", None),
                getattr(next_fn, "end_col_offset", None),
            )
            u_mut = set()
            enc_child = next_fn
        chain_infos.append(
            _analyze_enclosing_fn_cells(
                fn_node, span, unit_mutations=u_mut, enclosed_fn=enc_child
            )
        )

    all_chain_locals: Set[str] = (
        set().union(*(info.fn_locals for info in chain_infos))
        if chain_infos
        else set()
    )
    for _mut_name, mutated_vars, _escaped_def in intra_nl_mutators:
        escaping_nl_muts = (
            mutated_vars - set(ctx.unit_top_locals)
        ) | (mutated_vars & nonlocal_set)
        if bool(escaping_nl_muts & all_chain_locals):
            return True

    # 1 & 2: Intra-unit closure (or closure unit) captures an enclosing cell variable that is
    # unbound before the unit, mutated/deleted after the unit, mutated in an enclosing loop,
    # or mutated via nonlocal in a sibling closure across the enclosing chain.
    captured_candidates = intra_closure_reads & (
        eff_input_set | ((store_set | output_set) if ctx.is_subroutine else set())
    )
    for var in captured_candidates:
        acc_ext_muts: Set[str] = set()
        acc_pre_bound: Set[str] = set()
        for idx in range(len(chain_infos) - 1, -1, -1):
            info = chain_infos[idx]
            acc_ext_muts.update(
                info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            acc_pre_bound.update(info.pre_bound)
            if var not in info.fn_locals:
                continue
            if (
                var in eff_input_set and var not in acc_pre_bound
            ) or var in acc_ext_muts:
                return True
            break

    # 3: Subroutine intra-unit closure mutates a subroutine top-level variable via nonlocal
    # that is also bound/mutated/read in the enclosing function outside the subroutine.
    if ctx.is_subroutine and intra_nl_mutators:
        non_call_loads = _find_non_call_loaded_names(ctx.candidate_stmts)
        outer_touched: Set[str] = set()
        all_post_reads: Set[str] = set()
        all_sib_reads: Set[str] = set()
        for info in chain_infos:
            outer_touched.update(
                info.pre_mutated
                | info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            all_post_reads.update(info.post_reads)
            for _, sib_r, _ in info.sibling_closures:
                all_sib_reads.update(sib_r)
        for mut_fn_name, mutated_vars, is_escaped_def in intra_nl_mutators:
            mut_fn_escapes = (
                is_escaped_def
                or mut_fn_name in all_post_reads
                or mut_fn_name in all_sib_reads
                or mut_fn_name in intra_closure_reads
                or mut_fn_name in non_call_loads
            )
            for var in mutated_vars & set(ctx.unit_top_locals):
                if var in outer_touched or (
                    mut_fn_escapes
                    and (var in all_post_reads or var in all_sib_reads)
                ):
                    return True

    # 4: Sibling closure in an enclosing function is passed into or referenced by the unit
    # and shares a mutated cell variable with the unit.
    all_sib_closures = _propagate_sibling_closures(
        [
            sib
            for info in chain_infos
            for sib in info.sibling_closures
        ],
        alias_edges=[
            edge
            for info in chain_infos
            for edge in info.alias_edges
        ],
    )
    sub_mutated = (
        unit_mutated_names
        if ctx.is_subroutine
        else (nonlocal_set & unit_mutated_names)
    )
    sub_touched = eff_input_set | sub_mutated
    for sib_name, sib_reads, sib_nl_muts in all_sib_closures:
        if sib_name is None or sib_name not in eff_input_set:
            continue
        if bool(sib_reads & all_chain_locals & sub_mutated):
            return True
        if bool(sib_nl_muts & all_chain_locals & sub_touched):
            return True

    return False
