"""Lexical scope closure discovery, position classification, and capture analysis."""

from __future__ import annotations

import ast
import builtins
from typing import (
    AbstractSet,
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.fixer._ast_compat import (
    is_match_as_node,
    is_match_node,
    is_match_or_node,
    is_try_or_trystar_node,
    is_type_alias_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _get_cached_tree_source_lines,
    _get_or_compute_scope_cached,
    _get_scope_ancestor_map,
    _get_scope_parent_map,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _register_tree_source_lines,
    _scope_cell_info_cache,
    _scope_closures_cache,
    _tree_contains_scope_node,
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
    source_lines: Optional[Sequence[str]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Computes the earliest (lineno, col_offset) of an AST node including decorators.

    Note:
        Python's AST records decorator positions on the decorator expression node
        rather than the ``@`` token, while Python's grammar places ``@`` at the same
        indentation column as ``def``/``class`` (``node.col_offset``). When
        ``source_lines`` is provided (or cached on the enclosing tree via
        ``_register_tree_source_lines`` / ``_get_or_parse_tree``), this helper scans
        upward from each decorator expression to locate the ``@`` line; for raw
        unregistered ``ast.parse()`` trees without ``source_lines``, it falls back to
        ``dec.lineno`` and ``node.col_offset``.
    """
    n_line = getattr(node, "lineno", None)
    n_col = getattr(node, "col_offset", None)
    best_line: Optional[int] = n_line if isinstance(n_line, int) else None
    best_col: Optional[int] = n_col if isinstance(n_col, int) else None
    decorators = getattr(node, "decorator_list", ())
    if decorators and source_lines is None:
        source_lines = _get_cached_tree_source_lines(node)
    for dec in decorators:
        d_line = getattr(dec, "lineno", None)
        if isinstance(d_line, int):
            eff_d_line = d_line
            at_col: Optional[int] = best_col
            if source_lines is not None and 1 <= d_line <= len(source_lines):
                scan_ln = d_line
                while scan_ln >= 1:
                    line_txt = source_lines[scan_ln - 1]
                    stripped = line_txt.lstrip()
                    if stripped.startswith("@"):
                        eff_d_line = scan_ln
                        at_col = len(line_txt) - len(stripped)
                        break
                    if (
                        scan_ln < d_line
                        and stripped
                        and not stripped.startswith(("#", "("))
                        and not line_txt.rstrip().endswith("\\")
                    ):
                        break
                    scan_ln -= 1
            if best_line is None or eff_d_line < best_line:
                best_line = eff_d_line
                best_col = at_col
            elif (
                eff_d_line == best_line
                and at_col is not None
                and (best_col is None or at_col < best_col)
            ):
                best_col = at_col
    return best_line, best_col


def _node_scope_bounds(
    node: ast.AST, source_lines: Optional[Sequence[str]] = None
) -> Tuple[int, int]:
    """Computes (earliest_start, n_end) line bounds for a scope AST node."""
    n_start = getattr(node, "lineno", 0) or 0
    n_end = getattr(node, "end_lineno", None) or n_start
    estart, _ = _node_earliest_start_pos(node, source_lines=source_lines)
    return (estart if isinstance(estart, int) and estart > 0 else n_start), n_end


def _get_tree_scope_intervals(
    tree: ast.AST,
    source_lines: Optional[Sequence[str]] = None,
) -> Sequence[Tuple[int, int, int, ast.AST]]:
    """Builds and caches sorted intervals (earliest_start, n_end, span, node) for scope nodes in tree.

    Note:
        AST trees passed via ``tree=`` are assumed immutable while cached. Callers that
        mutate a cached AST in place must call ``evict_tree_scopes(tree)`` (or
        ``clear_dataflow_cache()``) after mutation so cached scope intervals and parent
        maps are invalidated.
    """
    if source_lines is not None:
        _register_tree_source_lines(tree, source_lines)
    else:
        source_lines = _get_cached_tree_source_lines(tree)

    def _compute() -> Tuple[Tuple[int, int, int, ast.AST], ...]:
        intervals: List[Tuple[int, int, int, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                earliest_start, n_end = _node_scope_bounds(node, source_lines)
                intervals.append((earliest_start, n_end, n_end - earliest_start, node))
        intervals.sort(
            key=lambda item: (item[2], -item[0], -getattr(item[3], "col_offset", 0))
        )
        return tuple(intervals)

    if _is_ephemeral_unit_ast(tree):
        return _compute()
    cache_key = _make_trusted_scope_cache_key(
        owner=tree,
        kind="tree_scope_intervals",
        params=(bool(source_lines),),
    )
    return cast(
        Tuple[Tuple[int, int, int, ast.AST], ...],
        _get_or_compute_scope_cached(_scope_cell_info_cache, cache_key, _compute),
    )


def _get_tree_expr_scope_intervals(
    tree: ast.AST,
) -> Tuple[Tuple[int, int, int, ast.AST], ...]:
    """Returns cached (start, end, col_offset, node) tuples for Lambda and comprehension scopes."""

    def _compute() -> Tuple[Tuple[int, int, int, ast.AST], ...]:
        out: List[Tuple[int, int, int, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Lambda, *_COMPREHENSION_TYPES)):
                n_start = int(getattr(node, "lineno", 0) or 0)
                n_end = int(getattr(node, "end_lineno", n_start) or n_start)
                n_col = int(getattr(node, "col_offset", 0) or 0)
                if n_start > 0:
                    out.append((n_start, n_end, n_col, node))
        return tuple(out)

    if _is_ephemeral_unit_ast(tree):
        return _compute()
    return cast(
        Tuple[Tuple[int, int, int, ast.AST], ...],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _make_trusted_scope_cache_key(tree, "tree_expr_scope_intervals"),
            _compute,
        ),
    )


def _node_strictly_encloses_bounds(
    node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> bool:
    """Checks whether an AST node strictly encloses the 2D span [u_start, u_end].

    Returns False when line or required same-line column bounds are unavailable,
    so callers treating non-enclosing nodes as inside/overlapping the unit fail closed.
    """
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

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()


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

    if isinstance(scope_node, _COMPREHENSION_TYPES):
        for gen in scope_node.generators:
            for t_node in ast.walk(gen.target):
                if isinstance(t_node, ast.Name) and isinstance(
                    t_node.ctx, ast.Store
                ):
                    names.add(t_node.id)
        return names

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
        elif is_type_alias_node(curr):
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
            _make_trusted_scope_cache_key(scope_node, "single_node_locals"),
            lambda: frozenset(_extract_single_node_local_names(scope_node)),
        ),
    )


def _is_function_like_scope_node(node: ast.AST) -> bool:
    """Returns True if node is a function, lambda, or comprehension scope (or a single-item Module wrapper)."""
    target = (
        node.body[0]
        if isinstance(node, ast.Module) and len(node.body) == 1
        else node
    )
    if isinstance(target, ast.Expr):
        target = target.value
    return isinstance(
        target,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            *_COMPREHENSION_TYPES,
        ),
    )


def _collect_ancestors_from_parent_map(
    start_node: ast.AST,
    parent_map: Mapping[ast.AST, ast.AST],
    module_tree: ast.AST,
    *,
    include_classes: bool = False,
) -> List[ast.AST]:
    """Walks parent_map from start_node, keeping immediate ClassDef and enclosing functions/module."""
    ancestors: List[ast.AST] = []
    curr: Optional[ast.AST] = parent_map.get(start_node)
    first_scope = not _is_function_like_scope_node(start_node)
    visited_mod = False
    while curr is not None:
        if curr is module_tree:
            visited_mod = True
        if isinstance(
            curr,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                *_COMPREHENSION_TYPES,
                ast.Module,
            ),
        ):
            ancestors.append(curr)
            first_scope = False
        elif isinstance(curr, ast.ClassDef):
            if include_classes or first_scope:
                ancestors.append(curr)
            first_scope = False
        curr = parent_map.get(curr)
    if not visited_mod and (
        isinstance(
            module_tree,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                *_COMPREHENSION_TYPES,
                ast.Module,
            ),
        )
        or (
            isinstance(module_tree, ast.ClassDef)
            and (include_classes or first_scope)
        )
    ):
        ancestors.append(module_tree)
    return ancestors


def _resolve_enclosing_scopes_by_coords(
    scope_node: ast.AST,
    module_tree: ast.AST,
    *,
    include_classes: bool = False,
) -> List[ast.AST]:
    """Resolves enclosing lexical scope AST nodes in module_tree by parent_map or unambiguous coordinates."""
    parent_map = _get_scope_parent_map(module_tree)
    if scope_node in parent_map:
        return _collect_ancestors_from_parent_map(
            scope_node, parent_map, module_tree, include_classes=include_classes
        )
    is_fn_like = _is_function_like_scope_node(scope_node)
    body = getattr(scope_node, "body", None)
    if isinstance(body, (list, tuple)):
        stmts = [s for s in body if isinstance(s, ast.AST)]
    elif isinstance(scope_node, ast.AST) and not isinstance(
        scope_node, ast.Module
    ):
        stmts = [scope_node]
    else:
        stmts = []
    if not stmts:
        return [module_tree] if isinstance(module_tree, ast.Module) else []
    first_stmt = stmts[0]
    if first_stmt in parent_map:
        return _collect_ancestors_from_parent_map(
            first_stmt, parent_map, module_tree, include_classes=include_classes
        )
    s_line = getattr(first_stmt, "lineno", None)
    e_line = getattr(stmts[-1], "end_lineno", getattr(stmts[-1], "lineno", None))
    s_col = getattr(first_stmt, "col_offset", None)
    e_col = getattr(stmts[-1], "end_col_offset", None)
    has_valid_lines = (
        isinstance(s_line, int)
        and isinstance(e_line, int)
        and 1 <= s_line <= e_line
    )
    all_sub_scopes: List[ast.AST] = [
        node for _, _, _, node in _get_tree_scope_intervals(module_tree)
    ] + [
        node for _, _, _, node in _get_tree_expr_scope_intervals(module_tree)
    ]
    is_ambiguous_rel_coords = (
        not has_valid_lines
        or (
            s_line == 1
            and s_col in (None, 0)
            and any(
                (getattr(n, "lineno", 0) or 0) > 1
                or (getattr(n, "end_lineno", 0) or 0) > 1
                for n in all_sub_scopes
            )
        )
    )
    if has_valid_lines and not is_ambiguous_rel_coords:
        assert isinstance(s_line, int) and isinstance(e_line, int)
        eff_s_col = None if (s_line == 1 and s_col == 0) else s_col
        eff_e_col = None if (s_line == 1 and s_col == 0) else e_col
        matched_fns: List[ast.AST] = []
        for node in all_sub_scopes:
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Lambda,
                    *_COMPREHENSION_TYPES,
                ),
            ):
                n_start = getattr(node, "lineno", 0) or 0
                n_end = getattr(node, "end_lineno", n_start) or n_start
                if n_start <= s_line and e_line <= n_end and not _is_span_in_scope_header(
                    node, s_line, e_line, u_start_col=eff_s_col, u_end_col=eff_e_col
                ):
                    matched_fns.append(node)
        if matched_fns:
            matched_fns.sort(
                key=lambda n: (
                    int(getattr(n, "lineno", 0) or 0),
                    -int(getattr(n, "end_lineno", 0) or 0),
                    int(getattr(n, "col_offset", 0) or 0),
                )
            )
            imm = _resolve_immediate_scope_in_fn(
                matched_fns[-1],
                s_line,
                e_line,
                u_start_col=eff_s_col,
                u_end_col=eff_e_col,
            )
            res = list(matched_fns)
            if isinstance(imm, ast.ClassDef) and (
                include_classes or not is_fn_like
            ):
                res.append(imm)
            if isinstance(module_tree, ast.Module):
                res.append(module_tree)
            return res
        imm_mod = _resolve_immediate_scope_in_fn(
            module_tree, s_line, e_line, u_start_col=eff_s_col, u_end_col=eff_e_col
        )
        if isinstance(imm_mod, ast.ClassDef) and (
            include_classes or not is_fn_like
        ):
            res_cls: List[ast.AST] = [imm_mod]
            if isinstance(module_tree, ast.Module):
                res_cls.append(module_tree)
            return res_cls
    return [module_tree] if isinstance(module_tree, ast.Module) else []


def _extract_scope_local_names(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    include_classes: bool = False,
    include_self_name: bool = True,
    cache_scope_node: bool = True,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> Set[str]:
    """Extracts parameters and locally assigned/imported/defined symbols in scope_node
    and all enclosing scopes up to module level.

    By default ``include_classes=False`` enforces PEP 227 lexical scoping rules:
    enclosing ``ClassDef`` namespaces do not form lexical closures for nested
    functions, lambdas, or comprehensions (only when ``scope_node`` itself is a
    non-function block directly inside a ``ClassDef`` are immediate class attributes
    visible). Pass ``include_classes=True`` to include all ancestor ``ClassDef``
    bindings unconditionally.
    """
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
    for enc in enclosing_scopes:
        names.update(_get_cached_single_node_local_names(enc))
    allowed_types = (
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            *_COMPREHENSION_TYPES,
            ast.ClassDef,
            ast.Module,
        )
        if include_classes
        else (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Lambda,
            *_COMPREHENSION_TYPES,
            ast.Module,
        )
    )
    if module_tree is not None and module_tree is not scope_node:
        parent_map = _get_scope_ancestor_map(module_tree)
        curr: Optional[ast.AST] = parent_map.get(scope_node)
        if curr is None and not enclosing_scopes:
            for enc in _resolve_enclosing_scopes_by_coords(
                scope_node, module_tree, include_classes=include_classes
            ):
                names.update(_get_cached_single_node_local_names(enc))
            return names
        visited_module_tree = False
        first_anc = not _is_function_like_scope_node(scope_node)
        while curr is not None:
            if curr is module_tree:
                visited_module_tree = True
            if isinstance(curr, allowed_types) or (
                first_anc and isinstance(curr, ast.ClassDef)
            ):
                names.update(_get_cached_single_node_local_names(curr))
            if isinstance(
                curr,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.Lambda,
                    *_COMPREHENSION_TYPES,
                    ast.ClassDef,
                    ast.Module,
                ),
            ):
                first_anc = False
            curr = parent_map.get(curr)
        if not visited_module_tree and (
            isinstance(module_tree, allowed_types)
            or (first_anc and isinstance(module_tree, ast.ClassDef))
        ):
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
            _make_trusted_scope_cache_key(root, "has_star_import"),
            lambda: _node_has_star_import_uncached(root),
        )
    )


def _has_wildcard_star_import(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    cache_scope_node: bool = True,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> bool:
    """Checks whether scope_node or any enclosing scope contains a wildcard star import."""
    if any(_node_has_star_import(enc, use_cache=True) for enc in enclosing_scopes):
        return True
    if module_tree is not None:
        if _node_has_star_import(module_tree, use_cache=True):
            return True
        if scope_node is not module_tree and not _tree_contains_scope_node(
            module_tree, scope_node
        ):
            if _node_has_star_import(scope_node, use_cache=cache_scope_node):
                return True
            if not enclosing_scopes:
                return any(
                    _node_has_star_import(enc, use_cache=True)
                    for enc in _resolve_enclosing_scopes_by_coords(
                        scope_node, module_tree
                    )
                )
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


class _FrozenClosureList(List[Tuple[Any, frozenset[str]]]):
    """Immutable list subclass for cached scope closure records."""

    __slots__ = ()

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes referenced by this cached closure list (none)."""
        return ()

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("_FrozenClosureList is immutable")

    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable
    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


_EMPTY_FROZEN_CLOSURE_LIST: _FrozenClosureList = _FrozenClosureList()


def _collect_scope_closures_uncached(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    genexp_include_header: bool = True,
    cache_parent_map: bool = True,
    enclosing_scopes: Sequence[ast.AST] = (),
    source_lines: Optional[Sequence[str]] = None,
) -> _FrozenClosureList:
    """Walks ``scope_node`` to collect nested closures and their free outer reads.

    Note:
        ``genexp_include_header`` defaults to ``True`` so a single cached scan serves both
        post-unit downstream-read analysis (where a post-unit generator expression's
        outermost iterable evaluates after the unit) and pre-unit closure scanning. For
        pre-unit generator expressions, including the outermost iterable is a conservative
        over-approximation because Python evaluates the first ``for`` iterable eagerly at
        generator construction time rather than capturing it in a deferred closure cell;
        dedicated intra-unit cell analysis passes ``include_header=False`` directly.
    """
    raw_scope_body = getattr(scope_node, "body", None)
    skip_root = isinstance(raw_scope_body, list)
    if not any(
        (not skip_root or n is not scope_node)
        and isinstance(
            n,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.GeneratorExp,
            ),
        )
        for n in ast.walk(scope_node)
    ):
        return _EMPTY_FROZEN_CLOSURE_LIST
    closures: List[Tuple[_ClosurePos, frozenset[str]]] = []
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
        enclosing_scopes=enclosing_scopes,
    )
    has_star_import = _has_wildcard_star_import(
        scope_node,
        module_tree=module_tree,
        cache_scope_node=use_parent_cache,
        enclosing_scopes=enclosing_scopes,
    )

    if source_lines is not None:
        if isinstance(module_tree, ast.AST):
            _register_tree_source_lines(module_tree, source_lines)
        _register_tree_source_lines(scope_node, source_lines)
    else:
        source_lines = _get_cached_tree_source_lines(
            module_tree if isinstance(module_tree, ast.AST) else scope_node
        )

    def _resolve_stmt_start_pos(s: ast.AST) -> Tuple[int, int]:
        estart, ecol = _node_earliest_start_pos(s, source_lines=source_lines)
        return int(estart or 0), int(ecol or 0)

    class _ExprClosureCollector(_BaseScopeVisitor):
        def __init__(
            self,
            default_line: int,
            default_col: int,
            bound_names: AbstractSet[str] = frozenset(),
            active_shadowed: Optional[Set[str]] = None,
        ) -> None:
            super().__init__()
            self.default_line = default_line
            self.default_col = default_col
            self.bound_names = bound_names
            self.active_shadowed = (
                shadowed_names if active_shadowed is None else active_shadowed
            )

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
                    frozenset(
                        r
                        for r in _extract_nested_scope_free_reads(
                            subnode, include_header=include_header
                        )
                        if not self._is_in_comp(r) and r not in self.bound_names
                    ),
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
                node,
                parent_map,
                self.active_shadowed,
                has_star_import=has_star_import,
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
        active_shadowed: Optional[Set[str]] = None,
    ) -> None:
        collector = _ExprClosureCollector(
            default_line, default_col, bound_names, active_shadowed
        )
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
        if is_try_or_trystar_node(s):
            handlers = getattr(s, "handlers", [])
            return [h.type for h in handlers if getattr(h, "type", None) is not None]
        if is_match_node(s):
            return [getattr(s, "subject")]
        return [s]

    def _walk_stmts(
        stmts: Iterable[ast.AST],
        bound_names: AbstractSet[str] = frozenset(),
        cls_shadowed: AbstractSet[str] = frozenset(),
    ) -> None:
        eff_shadowed = (
            shadowed_names | set(cls_shadowed)
            if cls_shadowed
            else shadowed_names
        )
        for stmt in stmts:
            stmt_start, stmt_col = _resolve_stmt_start_pos(stmt)

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                before_len = len(closures)
                _record_expr_closures(
                    iter_nested_scope_header_nodes(stmt, include_annotations=False),
                    stmt_start,
                    stmt_col,
                    bound_names,
                    eff_shadowed,
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
                    fn_scoped_exprs,
                    stmt_start,
                    stmt_col,
                    inner_bound,
                    eff_shadowed,
                )
                body_reads = frozenset(
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
                    stmt.decorator_list,
                    stmt_start,
                    stmt_col,
                    bound_names,
                    eff_shadowed,
                )
                inner_bound = bound_names | _extract_node_type_param_names(stmt)
                class_scoped_exprs = (
                    list(stmt.bases)
                    + [kw.value for kw in stmt.keywords]
                    + list(getattr(stmt, "type_params", []))
                )
                _record_expr_closures(
                    class_scoped_exprs,
                    stmt_start,
                    stmt_col,
                    inner_bound,
                    eff_shadowed,
                )
                inner_cls_shadowed = frozenset(
                    _extract_single_node_local_names(stmt)
                )
                _walk_stmts(stmt.body, inner_bound, inner_cls_shadowed)
                continue

            header_bound = bound_names | _extract_node_type_param_names(stmt)
            _record_expr_closures(
                _extract_header_nodes(stmt),
                stmt_start,
                stmt_col,
                header_bound,
                eff_shadowed,
            )

            for attr in ("body", "orelse", "finalbody"):
                sub_stmts = getattr(stmt, attr, None)
                if isinstance(sub_stmts, list):
                    _walk_stmts(sub_stmts, bound_names, cls_shadowed)

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
                _walk_stmts(getattr(handler, "body", []), h_bound, cls_shadowed)

            for case in getattr(stmt, "cases", []):
                pattern = getattr(case, "pattern", None)
                case_bound = bound_names
                if isinstance(pattern, ast.AST):
                    _record_expr_closures(
                        [pattern], stmt_start, stmt_col, bound_names, eff_shadowed
                    )
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
                    _record_expr_closures(
                        [guard], stmt_start, stmt_col, case_bound, eff_shadowed
                    )
                _walk_stmts(getattr(case, "body", []), case_bound, cls_shadowed)

    scope_stmts = (
        raw_scope_body if isinstance(raw_scope_body, list)
        else [raw_scope_body] if isinstance(raw_scope_body, ast.AST)
        else []
    )
    root_tp_names = frozenset(_extract_node_type_param_names(scope_node))
    _walk_stmts(scope_stmts, root_tp_names)
    return _FrozenClosureList(closures) if closures else _EMPTY_FROZEN_CLOSURE_LIST


def _collect_scope_closures(
    scope_node: ast.AST,
    module_tree: Optional[ast.AST] = None,
    *,
    enclosing_scopes: Sequence[ast.AST] = (),
    source_lines: Optional[Sequence[str]] = None,
) -> _FrozenClosureList:
    """Discovers all closures defined in a scope, recorded with their starting line."""
    if source_lines is not None:
        if isinstance(module_tree, ast.AST):
            _register_tree_source_lines(module_tree, source_lines)
        _register_tree_source_lines(scope_node, source_lines)
    eff_source_lines = (
        source_lines
        if source_lines is not None
        else _get_cached_tree_source_lines(
            module_tree if isinstance(module_tree, ast.AST) else scope_node
        )
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *(
            (module_tree,)
            if isinstance(module_tree, ast.AST)
            else ()
        ),
        *(s for s in enclosing_scopes if isinstance(s, ast.AST)),
    )
    cache_key: Any = (
        scope_node
        if not aux_nodes and eff_source_lines is None
        else _make_trusted_scope_cache_key(
            scope_node,
            "scope_closures",
            params=(bool(eff_source_lines),) if eff_source_lines is not None else (),
            aux_ast_nodes=aux_nodes,
        )
    )
    return cast(
        _FrozenClosureList,
        _get_or_compute_scope_cached(
            _scope_closures_cache,
            cache_key,
            lambda: _collect_scope_closures_uncached(
                scope_node,
                module_tree=module_tree,
                enclosing_scopes=enclosing_scopes,
                source_lines=eff_source_lines,
            ),
        ),
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
    *,
    enclosing_scopes: Sequence[ast.AST] = (),
    source_lines: Optional[Sequence[str]] = None,
) -> Set[str]:
    """Discovers free variables captured by functions, class methods, and lambdas defined prior
    to the unit in the same lexical scope."""
    captured_reads: Set[str] = set()
    for c_pos, reads in _collect_scope_closures(
        scope_node,
        module_tree=module_tree,
        enclosing_scopes=enclosing_scopes,
        source_lines=source_lines,
    ):
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
    """Recursively determines if a pattern matching AST node unconditionally matches any subject.

    Note:
        Passing ``None`` directly returns ``False`` (no pattern node exists). Inside an
        ``ast.MatchAs`` node, ``pattern.pattern is None`` represents a wildcard ``case _:``
        or bare capture ``case x:`` and is handled directly before recursing.
    """
    if pattern is None:
        return False
    if is_match_as_node(pattern):
        sub_pat = getattr(pattern, "pattern", None)
        return sub_pat is None or _is_irrefutable_pattern(sub_pat)
    if is_match_or_node(pattern):
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
    if isinstance(node, _COMPREHENSION_TYPES):
        if not node.generators:
            return False
        iter0 = node.generators[0].iter
        return _is_ast_node_within_unit_bounds(
            iter0,
            u_start,
            u_end,
            u_end_col=u_end_col,
            u_start_col=u_start_col,
        ) or _node_strictly_encloses_bounds(
            iter0,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
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
    first_body: Optional[ast.AST] = (
        body[0]
        if isinstance(body, (list, tuple)) and body
        else body
        if isinstance(body, ast.AST)
        else None
    )
    if first_body is not None:
        b_start, b_col = _node_earliest_start_pos(first_body)
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
    raw_body = getattr(scope_node, "body", None)
    if not isinstance(raw_body, (list, tuple)):
        return scope_node
    stack: List[ast.AST] = list(raw_body)
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


def _extract_scope_nonlocals_and_globals_sets(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[ast.AST, Set[str], Set[str], Set[str]]:
    """Extracts (target_scope, nonlocal_names, outer_globals, intra_globals) in immediate scope."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return scope_node, set(), set(), set()
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
    return target_scope, nonlocal_names, outer_globals, intra_globals


def _extract_scope_nonlocals_and_outer_globals(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (nonlocal_names, outer_globals) declared in the unit's immediate scope."""
    _, nonlocal_names, outer_globals, _ = (
        _extract_scope_nonlocals_and_globals_sets(
            scope_node,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
    )
    return nonlocal_names, outer_globals


def _has_escaped_intra_unit_global_declarations(
    scope_node: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
    *,
    is_subroutine: bool = True,
) -> bool:
    """Checks if an in-unit global declaration affects symbol bindings outside the unit."""
    target_scope, _, outer_globals, intra_globals = (
        _extract_scope_nonlocals_and_globals_sets(
            scope_node,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
    )
    if not intra_globals:
        return False
    if not is_subroutine:
        return True
    body = getattr(target_scope, "body", [])
    shared_globals = intra_globals & outer_globals
    if shared_globals and isinstance(body, list):
        u_stack: List[ast.AST] = list(body)
        while u_stack:
            u_curr = u_stack.pop()
            if isinstance(u_curr, _INNER_SCOPE_TYPES):
                continue
            if _is_ast_node_within_unit_bounds(
                u_curr, u_start, u_end, u_end_col=u_end_col, u_start_col=u_start_col
            ) and isinstance(u_curr, ast.Name) and isinstance(u_curr.ctx, (ast.Store, ast.Del)):
                if u_curr.id in shared_globals:
                    return True
            u_stack.extend(ast.iter_child_nodes(u_curr))
    intra_globals -= outer_globals
    if not intra_globals:
        return False
    stack: List[ast.AST] = list(body) if isinstance(body, list) else []
    while stack:
        curr = stack.pop()
        if isinstance(
            curr, (ast.Global, ast.Nonlocal)
        ) or _is_ast_node_within_unit_bounds(
            curr, u_start, u_end, u_end_col=u_end_col, u_start_col=u_start_col
        ):
            continue
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if curr.name in intra_globals:
                return True
            if _extract_nested_scope_free_reads(curr) & intra_globals:
                return True
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, (ast.Lambda, *_COMPREHENSION_TYPES)):
            if not _node_strictly_encloses_bounds(
                curr,
                u_start,
                u_end,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            ):
                if (
                    _extract_nested_scope_free_reads(curr, include_header=True)
                    & intra_globals
                ):
                    return True
                if (
                    isinstance(curr, _COMPREHENSION_TYPES)
                    and not isinstance(target_scope, ast.ClassDef)
                    and (
                        _extract_comprehension_walrus_names(curr)
                        & intra_globals
                    )
                ):
                    return True
                continue
        if isinstance(curr, ast.Name) and curr.id in intra_globals:
            return True
        if isinstance(curr, (ast.Import, ast.ImportFrom)):
            imp_names: Set[str] = set()
            _record_import_names(curr, imp_names)
            if imp_names & intra_globals:
                return True
        elif (
            isinstance(curr, ast.ExceptHandler)
            and isinstance(curr.name, str)
            and curr.name in intra_globals
        ):
            return True
        elif is_type_alias_node(curr):
            alias_name = getattr(curr, "name", None)
            if (
                isinstance(alias_name, ast.Name)
                and alias_name.id in intra_globals
            ):
                return True
        else:
            pat_name = _extract_pattern_bound_name(curr)
            if pat_name is not None and pat_name in intra_globals:
                return True
        stack.extend(ast.iter_child_nodes(curr))
    return False


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


def _iter_shallow_scope_nodes(
    nodes: Sequence[ast.AST],
    *,
    deferred_scope_types: Tuple[type, ...] = _INNER_SCOPE_TYPES,
) -> Iterator[ast.AST]:
    """Yields AST nodes in nodes, visiting headers of deferred_scope_types without descending into their bodies."""
    stack: List[ast.AST] = list(reversed(nodes))
    while stack:
        curr = stack.pop()
        yield curr
        if isinstance(curr, deferred_scope_types):
            for hdr in reversed(tuple(iter_nested_scope_header_nodes(curr))):
                stack.append(hdr)
            continue
        for ch in reversed(tuple(ast.iter_child_nodes(curr))):
            stack.append(ch)
