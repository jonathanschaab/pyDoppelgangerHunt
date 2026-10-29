"""Lexical scope closure discovery, position classification, and capture analysis."""

from __future__ import annotations

import ast
import builtins
from typing import (
    AbstractSet,
    Any,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _scope_cell_info_cache,
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


def _node_earliest_start_pos(
    node: ast.AST,
) -> Tuple[Optional[int], Optional[int]]:
    """Computes the earliest (lineno, col_offset) of an AST node including decorators."""
    n_line = getattr(node, "lineno", None)
    n_col = getattr(node, "col_offset", None)
    best_line: Optional[int] = n_line if isinstance(n_line, int) else None
    best_col: Optional[int] = n_col if isinstance(n_col, int) else None
    for dec in getattr(node, "decorator_list", ()):
        d_line = getattr(dec, "lineno", None)
        d_col = getattr(dec, "col_offset", None)
        if isinstance(d_line, int):
            if isinstance(d_col, int):
                raw_at_col = max(0, d_col - 1)
                at_col: Optional[int] = (
                    n_col
                    if isinstance(n_col, int) and n_col <= raw_at_col
                    else raw_at_col
                )
            else:
                at_col = best_col
            if best_line is None or d_line < best_line:
                best_line = d_line
                best_col = at_col
            elif (
                d_line == best_line
                and at_col is not None
                and (best_col is None or at_col < best_col)
            ):
                best_col = at_col
    return best_line, best_col


def _node_strictly_encloses_bounds(
    node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> bool:
    """Checks whether an AST node strictly encloses the 2D span [u_start, u_end]."""
    c_line = getattr(node, "lineno", None)
    c_end = getattr(node, "end_lineno", c_line)
    if not (
        isinstance(c_line, int)
        and isinstance(c_end, int)
        and c_line <= u_start
        and u_end <= c_end
    ):
        return False

    c_col = getattr(node, "col_offset", None)
    c_end_col = getattr(node, "end_col_offset", None)

    if c_line == u_start and u_start_col is not None:
        if not isinstance(c_col, int) or c_col > u_start_col:
            return False
    if c_line == u_end and u_end_col is not None:
        if not isinstance(c_col, int) or c_col >= u_end_col:
            return False

    if c_end == u_end and u_end_col is not None:
        if not isinstance(c_end_col, int) or c_end_col < u_end_col:
            return False
    if c_end == u_start and u_start_col is not None:
        if not isinstance(c_end_col, int) or c_end_col <= u_start_col:
            return False

    is_strictly_larger_start = (c_line < u_start) or (
        c_line == u_start
        and u_start_col is not None
        and isinstance(c_col, int)
        and c_col < u_start_col
    )
    is_strictly_larger_end = (c_end > u_end) or (
        c_end == u_end
        and u_end_col is not None
        and isinstance(c_end_col, int)
        and c_end_col > u_end_col
    )
    return is_strictly_larger_start or is_strictly_larger_end


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


def _extract_comprehension_walrus_names(
    comp_node: ast.AST,
    *,
    include_first_iter: bool = True,
) -> Set[str]:
    """Extracts walrus target names that bind to the enclosing scope per PEP 572."""
    walrus_names: Set[str] = set()
    first_iter = (
        comp_node.generators[0].iter
        if (
            not include_first_iter
            and isinstance(comp_node, _COMPREHENSION_TYPES)
            and comp_node.generators
        )
        else None
    )
    node_stack = list(ast.iter_child_nodes(comp_node))
    while node_stack:
        curr = node_stack.pop()
        if curr is first_iter:
            continue
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


def _extract_strict_fn_local_names(scope_node: ast.AST) -> Set[str]:
    """Extracts strict function-local bindings excluding globals, nonlocals, and self-name."""
    return _extract_single_node_local_names(
        scope_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
        include_bare_annotations=True,
    )


def _get_cached_single_node_local_names(scope_node: ast.AST) -> frozenset[str]:
    """Returns cached default single-node local names for persistent AST scope nodes."""
    return cast(
        frozenset[str],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (scope_node, "single_node_locals"),
            lambda: frozenset(_extract_single_node_local_names(scope_node)),
        ),
    )


def _extract_scope_local_names(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    include_classes: bool = True,
    include_self_name: bool = True,
    cache_scope_node: bool = True,
) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node
    and all enclosing scopes up to module level."""
    if include_self_name and cache_scope_node:
        names = set(_get_cached_single_node_local_names(scope_node))
    else:
        names = (
            _extract_single_node_local_names(scope_node)
            if include_self_name
            else _extract_single_node_local_names(
                scope_node, include_self_name=False
            )
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
                names.update(_get_cached_single_node_local_names(curr))
            curr = parent_map.get(curr)
        if not visited_module_tree and isinstance(module_tree, allowed_types):
            names.update(_get_cached_single_node_local_names(module_tree))
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


def _node_has_star_import_uncached(root: ast.AST) -> bool:
    """Checks whether an AST subtree contains a non-builtin wildcard star import."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module not in ("builtins", "__builtins__")
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(root)
    )


def _node_has_star_import(root: ast.AST, *, use_cache: bool = True) -> bool:
    """Checks (with optional scope-cache memoization) whether root has a star import."""
    if not use_cache:
        return _node_has_star_import_uncached(root)
    return bool(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (root, "has_star_import"),
            lambda: _node_has_star_import_uncached(root),
        )
    )


def _has_wildcard_star_import(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    cache_scope_node: bool = True,
) -> bool:
    """Checks whether scope_node or any enclosing scope contains a wildcard star import."""
    if module_tree is not None:
        if _node_has_star_import(module_tree, use_cache=True):
            return True
        if scope_node is not module_tree and scope_node not in _get_scope_parent_map(
            module_tree
        ):
            return _node_has_star_import(scope_node, use_cache=cache_scope_node)
        return False
    return _node_has_star_import(scope_node, use_cache=cache_scope_node)


def _is_eagerly_consumed_genexp(
    genexp: ast.GeneratorExp,
    parent_map: Mapping[ast.AST, ast.AST],
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
    *,
    genexp_include_header: bool = True,
    cache_parent_map: bool = True,
) -> List[Tuple[_ClosurePos, Set[str]]]:
    closures: List[Tuple[_ClosurePos, Set[str]]] = []
    use_parent_cache = cache_parent_map and not (
        isinstance(scope_node, ast.Module)
        and module_tree is not None
        and scope_node is not module_tree
    )
    parent_map = _get_scope_parent_map(scope_node, use_cache=use_parent_cache)
    shadowed_names = _extract_scope_local_names(
        scope_node,
        module_tree=module_tree,
        cache_scope_node=use_parent_cache,
    )
    has_star_import = _has_wildcard_star_import(
        scope_node,
        module_tree=module_tree,
        cache_scope_node=use_parent_cache,
    )

    def _resolve_stmt_start_pos(s: ast.AST) -> Tuple[int, int]:
        estart, ecol = _node_earliest_start_pos(s)
        return int(estart or 0), int(ecol or 0)

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
                self._record_closure_node(node, include_header=genexp_include_header)

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


def _is_irrefutable_pattern(
    pattern: Optional[ast.AST],
) -> bool:  # pragma: no cover (py310+)
    """Recursively determines if a pattern matching AST node unconditionally matches any subject."""
    if pattern is None:
        return False
    if type(pattern).__name__ == "MatchAs":
        sub_pat = getattr(pattern, "pattern", None)
        return sub_pat is None or _is_irrefutable_pattern(sub_pat)
    if type(pattern).__name__ == "MatchOr":
        patterns = getattr(pattern, "patterns", [])
        return any(_is_irrefutable_pattern(p) for p in patterns)
    return False


def _is_irrefutable_case(case: ast.AST) -> bool:  # pragma: no cover (py310+)
    """Returns True if the match case has no guard and an irrefutable pattern."""
    if getattr(case, "guard", None) is not None:
        return False
    return _is_irrefutable_pattern(getattr(case, "pattern", None))


def _is_span_in_scope_header(
    node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> bool:
    """Checks whether [u_start, u_end] precedes node.body[0] or lies inside node's header expressions."""
    if any(
        _is_ast_node_within_unit_bounds(
            hdr,
            u_start,
            u_end,
            u_end_col=u_end_col,
            u_start_col=u_start_col,
        )
        or _node_strictly_encloses_bounds(
            hdr,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
        for hdr in iter_nested_scope_header_nodes(node)
    ):
        return True
    body = getattr(node, "body", None)
    if body:
        b_start, b_col = _node_earliest_start_pos(body[0])
        n_lineno = int(getattr(node, "lineno", 0) or 0)
        if b_start is not None and (
            u_end < b_start
            or (
                u_start < b_start
                and (
                    u_start <= n_lineno
                    or any(
                        (getattr(h, "lineno", 0) or 0)
                        <= u_start
                        <= (getattr(h, "end_lineno", 0) or 0)
                        for h in (
                            *iter_nested_scope_header_nodes(node),
                            *getattr(
                                getattr(node, "args", None),
                                "posonlyargs",
                                (),
                            ),
                            *getattr(
                                getattr(node, "args", None), "args", ()
                            ),
                            *getattr(
                                getattr(node, "args", None),
                                "kwonlyargs",
                                (),
                            ),
                        )
                    )
                )
            )
            or (
                u_start == b_start
                and u_start_col is not None
                and isinstance(b_col, int)
                and u_start_col < b_col
            )
        ):
            return True
    return False


def _resolve_immediate_scope_in_fn(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> ast.AST:
    """Finds the innermost ClassDef in scope_node strictly enclosing the unit, or scope_node."""
    if u_start <= 0:
        return scope_node
    best: ast.AST = scope_node
    best_span: Optional[Tuple[int, int, int]] = None
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.ClassDef):
            c_start = int(getattr(curr, "lineno", 0) or 0)
            c_end = int(getattr(curr, "end_lineno", c_start) or c_start)
            c_col = int(getattr(curr, "col_offset", 0) or 0)
            in_hdr = _is_span_in_scope_header(
                curr,
                u_start,
                u_end,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
            strictly_inside = not in_hdr and _node_strictly_encloses_bounds(
                curr,
                u_start,
                u_end,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
            if strictly_inside:
                cand_span = (c_start, -c_end, c_col)
                if best_span is None or cand_span > best_span:
                    best = curr
                    best_span = cand_span
                stack.extend(curr.body)
            continue
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return best


def _extract_scope_nonlocals_and_outer_globals(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (nonlocal_names, outer_only_globals) declared in the unit's immediate scope."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set(), set()
    target_scope = _resolve_immediate_scope_in_fn(
        scope_node,
        u_start,
        u_end,
        u_start_col=u_start_col,
        u_end_col=u_end_col,
    )
    nonlocal_names: Set[str] = set()
    outer_globals: Set[str] = set()
    intra_globals: Set[str] = set()
    body = getattr(target_scope, "body", [])
    stack: List[ast.AST] = list(body) if isinstance(body, list) else []
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Nonlocal):
            nonlocal_names.update(curr.names)
            continue
        if isinstance(curr, ast.Global):
            if _is_ast_node_within_unit_bounds(
                curr,
                u_start,
                u_end,
                u_end_col=u_end_col,
                u_start_col=u_start_col,
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
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Set[str]:
    """Extracts nonlocal and outer global names declared in a function scope."""
    nonlocal_names, outer_globals = _extract_scope_nonlocals_and_outer_globals(
        scope_node,
        u_start,
        u_end,
        u_start_col=u_start_col,
        u_end_col=u_end_col,
    )
    return nonlocal_names | outer_globals
