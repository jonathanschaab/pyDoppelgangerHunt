"""Lexical scope closure discovery, position classification, and capture analysis."""

from __future__ import annotations

import ast
import builtins
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _scope_closures_cache,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _extract_nested_scope_free_reads,
    _record_import_names,
)
from pydoppelgangerhunt.fixer.scope import _extract_arg_names

_BUILTIN_NAMES: Set[str] = set(dir(builtins))


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
            continue
        if isinstance(curr, ast.NamedExpr) and isinstance(curr.target, ast.Name):
            walrus_names.add(curr.target.id)
        node_stack.extend(ast.iter_child_nodes(curr))
    return walrus_names


def _extract_single_node_local_names(scope_node: ast.AST) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node."""
    names: Set[str] = set()
    node_name = getattr(scope_node, "name", None)
    if isinstance(node_name, str):
        names.add(node_name)

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
        elif isinstance(curr, (ast.Global, ast.Nonlocal)):
            names.update(curr.names)
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

        for child in ast.iter_child_nodes(curr):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
            elif isinstance(child, _COMPREHENSION_TYPES):
                names.update(_extract_comprehension_walrus_names(child))
            elif not isinstance(child, _INNER_SCOPE_TYPES):
                stack.append(child)

    return names


def _extract_scope_local_names(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node
    and all enclosing scopes up to module level."""
    names = _extract_single_node_local_names(scope_node)
    if module_tree is not None and module_tree is not scope_node:
        parent_map = _get_scope_parent_map(module_tree)
        curr: Optional[ast.AST] = parent_map.get(scope_node)
        visited_module_tree = False
        while curr is not None:
            if curr is module_tree:
                visited_module_tree = True
            if isinstance(
                curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                names.update(_extract_single_node_local_names(curr))
            curr = parent_map.get(curr)
        if not visited_module_tree and isinstance(
            module_tree, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
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
        func = parent.func
        if isinstance(func, ast.Name) and func.id in _EAGER_CONSUMER_NAMES:
            if shadowed_names is not None and func.id in shadowed_names:
                return False
            if func.id in ("min", "max", "dict"):
                return len(parent.args) == 1 and parent.args[0] is genexp
            return len(parent.args) >= 1 and parent.args[0] is genexp
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

    def _record_expr_closures(
        nodes: Iterable[ast.AST], default_line: int, default_col: int
    ) -> None:
        for node in nodes:
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.GeneratorExp) and _is_eagerly_consumed_genexp(
                    subnode, parent_map, shadowed_names, has_star_import=has_star_import
                ):
                    continue
                if isinstance(subnode, (ast.Lambda, ast.GeneratorExp)):
                    sub_l = getattr(subnode, "lineno", None)
                    sub_c = getattr(subnode, "col_offset", None)
                    sub_line = int(sub_l) if sub_l is not None else default_line
                    sub_col = int(sub_c) if sub_c is not None else default_col
                    closures.append(
                        (
                            _ClosurePos(sub_line, sub_col),
                            _extract_nested_scope_free_reads(subnode),
                        )
                    )

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
            headers = [getattr(s, "subject")]
            for case in getattr(s, "cases", []):
                headers.append(getattr(case, "pattern"))
                if getattr(case, "guard", None) is not None:
                    headers.append(getattr(case, "guard"))
            return headers
        return [s]

    def _walk_stmts(stmts: Iterable[ast.AST]) -> None:
        for stmt in stmts:
            stmt_start, stmt_col = _resolve_stmt_start_pos(stmt)

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                closures.append(
                    (
                        _ClosurePos(stmt_start, stmt_col),
                        _extract_nested_scope_free_reads(stmt),
                    )
                )
                continue

            if isinstance(stmt, ast.ClassDef):
                class_exprs = (
                    list(stmt.decorator_list)
                    + list(stmt.bases)
                    + [kw.value for kw in stmt.keywords]
                    + list(getattr(stmt, "type_params", []))
                )
                _record_expr_closures(class_exprs, stmt_start, stmt_col)
                for item in stmt.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        i_start, i_col = _resolve_stmt_start_pos(item)
                        closures.append(
                            (
                                _ClosurePos(i_start, i_col),
                                _extract_nested_scope_free_reads(item),
                            )
                        )
                    elif isinstance(item, ast.ClassDef):
                        _walk_stmts([item])
                    elif isinstance(item, ast.stmt):
                        i_start, i_col = _resolve_stmt_start_pos(item)
                        _record_expr_closures(
                            _extract_header_nodes(item), i_start, i_col
                        )
                        for attr in ("body", "orelse", "finalbody"):
                            sub_body = getattr(item, attr, None)
                            if isinstance(sub_body, list):
                                _walk_stmts(sub_body)
                        for handler in getattr(item, "handlers", []):
                            _walk_stmts(getattr(handler, "body", []))
                        for case in getattr(item, "cases", []):
                            _walk_stmts(getattr(case, "body", []))
                continue

            _record_expr_closures(_extract_header_nodes(stmt), stmt_start, stmt_col)

            for attr in ("body", "orelse", "finalbody"):
                sub_stmts = getattr(stmt, attr, None)
                if isinstance(sub_stmts, list):
                    _walk_stmts(sub_stmts)

            for handler in getattr(stmt, "handlers", []):
                _walk_stmts(getattr(handler, "body", []))

            for case in getattr(stmt, "cases", []):
                _walk_stmts(getattr(case, "body", []))

    raw_scope_body = getattr(scope_node, "body", [])
    scope_stmts = (
        raw_scope_body if isinstance(raw_scope_body, list)
        else [raw_scope_body] if isinstance(raw_scope_body, ast.AST)
        else []
    )
    _walk_stmts(scope_stmts)
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
