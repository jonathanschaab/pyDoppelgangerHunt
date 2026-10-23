"""Enclosing function cell binding analysis and severed closure cell hazard checks."""

from __future__ import annotations

import ast
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
    cast,
)

from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _scope_cell_info_cache,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _collect_scope_closures_uncached,
    _extract_comprehension_walrus_names,
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    _has_wildcard_star_import,
    _is_eagerly_consumed_genexp,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_nested_scope_free_reads,
    _extract_pattern_bound_name,
    _is_ast_node_within_unit_bounds,
    _record_import_names,
)
from pydoppelgangerhunt.fixer.source import (
    _get_tree_scope_intervals,
    _node_scope_bounds,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


class _FnScopeContext(NamedTuple):
    """Cached per-function lexical scope metadata for eager generator expression checks."""

    fn_locals: Set[str]
    parent_map: Dict[ast.AST, ast.AST]
    shadowed_names: Set[str]
    has_star_import: bool


def _get_fn_scope_context_uncached(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    *,
    cache_parent_map: bool = True,
) -> _FnScopeContext:
    """Computes lexical scope metadata for a function or ephemeral unit module without caching."""
    return _FnScopeContext(
        fn_locals=_extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=True,
        ),
        parent_map=_get_scope_parent_map(scope_node, use_cache=cache_parent_map),
        shadowed_names=_extract_scope_local_names(
            scope_node, module_tree=mod_tree
        ),
        has_star_import=_has_wildcard_star_import(
            scope_node, module_tree=mod_tree
        ),
    )


def _get_fn_scope_context(
    scope_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
) -> _FnScopeContext:
    """Computes and caches lexical scope metadata for an AST scope node."""
    cache_key: Any = (
        (scope_node, "fn_scope_ctx")
        if mod_tree is None
        else (scope_node, "fn_scope_ctx", mod_tree)
    )
    return cast(
        _FnScopeContext,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _get_fn_scope_context_uncached(
                scope_node, mod_tree=mod_tree
            ),
        ),
    )


def _is_lazy_genexp(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node is a GeneratorExp that is not eagerly consumed."""
    if not isinstance(node, ast.GeneratorExp):
        return False
    if scope_ctx is None:
        return True
    return not _is_eagerly_consumed_genexp(
        node,
        scope_ctx.parent_map,
        scope_ctx.shadowed_names,
        has_star_import=scope_ctx.has_star_import,
    )


def _node_earliest_start_lineno(node: ast.AST) -> int:
    """Computes the earliest starting line of an AST node including any decorators."""
    return _node_scope_bounds(node)[0]


def _find_enclosing_function_chain(
    mod_tree: ast.AST,
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
) -> List[Tuple[int, int, ast.AST]]:
    """Finds enclosing FunctionDef/AsyncFunctionDef nodes ordered outermost to innermost."""
    chain: List[Tuple[int, int, int, ast.AST]] = []
    for estart, n_end, _span, node in _get_tree_scope_intervals(mod_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if estart <= 0 or estart > u_start or u_end > n_end:
            continue
        n_col = int(getattr(node, "col_offset", 0) or 0)
        n_lineno = int(getattr(node, "lineno", estart) or estart)
        if u_start in (estart, n_lineno):
            if u_start_col is not None and n_col > u_start_col:
                continue
            if u_end == n_end and (u_start_col is None or n_col == u_start_col):
                continue
        chain.append((estart, n_end, n_col, node))
    chain.sort(key=lambda item: (item[0], -item[1], item[2]))
    return [(estart, n_end, node) for estart, n_end, _col, node in chain]


def _extract_direct_nonlocals(
    scope_node: ast.AST,
    *,
    is_global: bool = False,
) -> Set[str]:
    """Extracts variable names declared nonlocal (or global) directly in scope_node."""
    target_cls = ast.Global if is_global else ast.Nonlocal
    names: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, target_cls):
            names.update(curr.names)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _extract_type_alias_or_pattern_name(node: ast.AST) -> Optional[str]:
    """Extracts the bound identifier from a TypeAlias or match pattern AST node."""
    cls_name = type(node).__name__
    if cls_name == "TypeAlias":
        target_name = getattr(node, "name", None)
        return target_name.id if isinstance(target_name, ast.Name) else None
    if cls_name in ("MatchAs", "MatchStar", "MatchMapping"):
        return _extract_pattern_bound_name(node)
    return None


def _extract_closure_nonlocal_mutations_and_deletions_uncached(
    scope_node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Computes (mutated_or_deleted, deleted) outer-scope variables via nonlocal in scope_node."""
    own_locals = (
        set()
        if isinstance(scope_node, ast.ClassDef)
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
    direct_deleted: Set[str] = set()
    nested_mutated: Set[str] = set()
    nested_deleted: Set[str] = set()
    stack: List[ast.AST] = list(getattr(scope_node, "body", []))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            direct_mutated.add(curr.name)
            sub_mut, sub_del = _extract_closure_nonlocal_mutations_and_deletions(
                curr
            )
            nested_mutated.update(sub_mut - own_locals)
            nested_deleted.update(sub_del - own_locals)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            direct_mutated.update(_extract_comprehension_walrus_names(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Nonlocal):
            direct_nonlocals.update(curr.names)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, (ast.Store, ast.Del)):
            direct_mutated.add(curr.id)
            if isinstance(curr.ctx, ast.Del):
                direct_deleted.add(curr.id)
        elif isinstance(curr, (ast.Import, ast.ImportFrom)):
            _record_import_names(curr, direct_mutated)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            direct_mutated.add(curr.name)
            direct_deleted.add(curr.name)
        else:
            bound_name = _extract_type_alias_or_pattern_name(curr)
            if bound_name is not None:
                direct_mutated.add(bound_name)
        stack.extend(ast.iter_child_nodes(curr))
    return (
        (direct_nonlocals & direct_mutated) | nested_mutated,
        (direct_nonlocals & direct_deleted) | nested_deleted,
    )


def _extract_closure_nonlocal_mutations_and_deletions(
    scope_node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Extracts and caches (mutated_or_deleted, deleted) nonlocal variables in scope_node."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return set(), set()
    muts, dels = cast(
        Tuple[Set[str], Set[str]],
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (scope_node, "nl_muts_dels"),
            lambda: _extract_closure_nonlocal_mutations_and_deletions_uncached(
                scope_node
            ),
        ),
    )
    return set(muts), set(dels)


def _extract_closure_nonlocal_mutations(scope_node: ast.AST) -> Set[str]:
    """Extracts outer-scope variables mutated or deleted via nonlocal inside scope_node."""
    return _extract_closure_nonlocal_mutations_and_deletions(scope_node)[0]


def _extract_cached_scope_free_reads(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
) -> Set[str]:
    """Extracts and caches free variable reads escaping a nested function or class."""
    return set(
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            (node, "free_reads"),
            lambda: _extract_nested_scope_free_reads(node),
        )
    )


def _extract_expr_scope_free_reads(
    expr: Union[
        ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp
    ],
    *,
    include_header: bool = False,
) -> Set[str]:
    """Extracts free variable reads from a Lambda or comprehension without synthetic AST nodes."""
    return _extract_nested_scope_free_reads(expr, include_header=include_header)


def _extract_comprehension_body_walrus_names(comp_node: ast.AST) -> Set[str]:
    """Extracts walrus target names inside a comprehension excluding generators[0].iter."""
    return _extract_comprehension_walrus_names(
        comp_node, include_first_iter=False
    )


def _extract_lazy_genexp_walrus_names(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts walrus target names inside lazy GeneratorExp subtrees of node."""
    names: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                names.update(_extract_comprehension_body_walrus_names(curr))
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
            stack.extend(ast.iter_child_nodes(curr))
            continue
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return names


def _iter_comprehension_body_exprs_and_targets(
    comp_node: Union[
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp
    ],
) -> Tuple[List[ast.AST], Set[str]]:
    """Extracts (non-header child expressions, comprehension target names)."""
    comp_targets = {
        n
        for gen in comp_node.generators
        for n in _extract_store_names_in_node(gen.target)
    }
    comp_exprs: List[ast.AST] = [
        ch
        for ch in ast.iter_child_nodes(comp_node)
        if not isinstance(ch, ast.comprehension)
    ]
    for idx, gen in enumerate(comp_node.generators):
        if idx > 0:
            comp_exprs.append(gen.iter)
        comp_exprs.extend(gen.ifs)
    return comp_exprs, comp_targets


def _extract_comprehension_deferred_closure_info(
    comp_node: Union[ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp],
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (deferred_free_reads, lazy_walrus_mutations) from a comprehension node."""
    is_lazy = _is_lazy_genexp(comp_node, scope_ctx)
    first_iter = comp_node.generators[0].iter if comp_node.generators else None
    nested_lazy_nodes: List[Tuple[Union[ast.Lambda, ast.GeneratorExp], bool]] = []
    sub_stack: List[Tuple[ast.AST, bool]] = (
        [(first_iter, True)]
        if (is_lazy and first_iter is not None)
        else [
            (ch, ch is first_iter)
            for ch in ast.iter_child_nodes(comp_node)
        ]
    )
    while sub_stack:
        sub, in_hdr = sub_stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sub_stack.extend(
                (hdr, in_hdr) for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, ast.Lambda):
            nested_lazy_nodes.append((sub, in_hdr))
            sub_stack.extend(
                (hdr, in_hdr) for hdr in iter_nested_scope_header_nodes(sub)
            )
            continue
        if isinstance(sub, ast.GeneratorExp):
            if _is_lazy_genexp(sub, scope_ctx):
                nested_lazy_nodes.append((sub, in_hdr))
                if sub.generators:
                    sub_stack.append((sub.generators[0].iter, in_hdr))
                continue
        sub_stack.extend(
            (
                ch,
                in_hdr
                or (
                    isinstance(sub, ast.comprehension)
                    and bool(comp_node.generators)
                    and sub is comp_node.generators[0]
                    and ch is first_iter
                ),
            )
            for ch in ast.iter_child_nodes(sub)
        )

    if not is_lazy and not nested_lazy_nodes:
        return set(), set()

    deferred_reads: Set[str] = set()
    if is_lazy:
        deferred_reads.update(
            _extract_expr_scope_free_reads(comp_node, include_header=False)
        )
    if nested_lazy_nodes:
        comp_all_free = _extract_expr_scope_free_reads(
            comp_node, include_header=True
        )
        _, comp_targets = _iter_comprehension_body_exprs_and_targets(comp_node)
        for lazy_sub, in_hdr in nested_lazy_nodes:
            sub_reads = _extract_expr_scope_free_reads(
                lazy_sub, include_header=False
            )
            if not in_hdr:
                sub_reads -= comp_targets
            deferred_reads.update(sub_reads & comp_all_free)

    lazy_muts = _extract_lazy_genexp_walrus_names(comp_node, scope_ctx)
    return deferred_reads, lazy_muts


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    mod_tree: Optional[ast.AST] = None,
) -> List[Tuple[str, Set[str], bool]]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    unit_mod = ast.Module(body=list(candidate_stmts), type_ignores=[])
    scope_ctx = _get_fn_scope_context_uncached(
        unit_mod, mod_tree=mod_tree, cache_parent_map=False
    )
    mutators: List[Tuple[str, Set[str], bool]] = []
    stack: List[ast.AST] = list(candidate_stmts)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = _extract_closure_nonlocal_mutations(curr)
            if mutated:
                is_escaped_def = (
                    isinstance(curr, ast.ClassDef)
                    or bool(getattr(curr, "decorator_list", None))
                    or _is_closure_factory_scope(curr, scope_ctx)
                )
                mutators.append((curr.name, mutated, is_escaped_def))
            stack.extend(getattr(curr, "body", []))
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, scope_ctx):
                w_names = _extract_comprehension_body_walrus_names(curr)
                if w_names:
                    mutators.append(("", w_names, True))
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
            stack.extend(ast.iter_child_nodes(curr))
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
    pre_unbound: AbstractSet[str] = frozenset()
    concurrent_sibling_vars: AbstractSet[str] = frozenset()


def _is_pos_before_span(
    line: Optional[int], col: Optional[int], span: _SpanBounds
) -> bool:
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


def _is_pos_after_span(
    line: Optional[int], col: Optional[int], span: _SpanBounds
) -> bool:
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
        node,
        span.start,
        span.end,
        u_end_col=span.end_col,
        u_start_col=span.start_col,
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


def _can_bind_in_fn_scope(
    name: str,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str] = frozenset(),
) -> bool:
    """Checks whether a binding in the current AST frame targets the enclosing function scope."""
    if active_nonlocals is not None:
        return name in active_nonlocals
    return name not in fn_globals


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


def _extract_store_names_in_node(node: ast.AST) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST target subtree."""
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


_NON_CLOSURE_RETURN_BUILTINS: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "divmod",
        "enumerate",
        "float",
        "format",
        "hash",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "round",
        "str",
        "sum",
    }
)


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions."""
    inline_reads: Set[str] = set()
    inline_muts: Set[str] = set()
    loaded_names: Set[str] = set()
    stack: List[ast.AST] = list(nodes)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Compare,
                ast.UnaryOp,
                ast.JoinedStr,
            ),
        ):
            continue
        if isinstance(curr, ast.Lambda):
            inline_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
            continue
        if isinstance(curr, _COMPREHENSION_TYPES):
            c_reads, c_muts = _extract_comprehension_deferred_closure_info(
                curr, scope_ctx
            )
            inline_reads.update(c_reads)
            inline_muts.update(c_muts)
            is_lazy = _is_lazy_genexp(curr, scope_ctx)
            if curr.generators:
                stack.append(curr.generators[0].iter)
            if not is_lazy:
                comp_exprs, comp_targets = (
                    _iter_comprehension_body_exprs_and_targets(curr)
                )
                sub_calls: Set[str] = set()
                _, _, sub_loads = _extract_expr_closure_and_load_info(
                    comp_exprs, scope_ctx, called_names_out=sub_calls
                )
                loaded_names.update(sub_loads - comp_targets)
                if called_names_out is not None:
                    called_names_out.update(sub_calls - comp_targets)
            continue
        if isinstance(curr, ast.IfExp):
            stack.append(curr.orelse)
            stack.append(curr.body)
            continue
        if isinstance(curr, ast.Subscript):
            stack.append(curr.value)
            continue
        if isinstance(curr, ast.Call):
            is_non_closure_builtin = (
                isinstance(curr.func, ast.Name)
                and curr.func.id in _NON_CLOSURE_RETURN_BUILTINS
                and (
                    scope_ctx is None
                    or curr.func.id not in scope_ctx.shadowed_names
                )
            )
            if not is_non_closure_builtin:
                if isinstance(curr.func, ast.Attribute):
                    stack.append(curr.func.value)
                elif isinstance(curr.func, ast.Name):
                    if called_names_out is not None:
                        called_names_out.add(curr.func.id)
                else:
                    stack.append(curr.func)
                stack.extend(curr.args)
                stack.extend(
                    kw.value for kw in curr.keywords if kw.value is not None
                )
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
    pre_unbound: Set[str]
    fn_globals: AbstractSet[str] = frozenset()


class _StackFrame(NamedTuple):
    """Traversal frame for _analyze_enclosing_fn_cells."""

    node: ast.AST
    is_branched: bool
    in_loop: bool
    active_nonlocals: Optional[Set[str]]
    post_pre_bound: Optional[AbstractSet[str]] = None


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
        n
        for n in names
        if _can_bind_in_fn_scope(n, frame.active_nonlocals, state.fn_globals)
    ]
    if not filtered:
        return
    if _is_pos_before_span(line, col, span):
        state.pre_mutated.update(filtered)
        if not frame.is_branched:
            state.pre_bound.update(filtered)
            state.pre_unbound.difference_update(filtered)
    if _is_pos_after_span(line, col, span):
        state.post_mutated.update(filtered)
    if frame.in_loop:
        state.loop_mutated.update(filtered)


def _record_deletion_at_pos(
    name: str,
    line: Optional[int],
    col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    *,
    unconditional_post_del: bool = False,
) -> None:
    """Records a variable deletion or ExceptHandler unbinding relative to span."""
    if not _can_bind_in_fn_scope(
        name, frame.active_nonlocals, state.fn_globals
    ):
        return
    if _is_pos_before_span(line, col, span) or (
        unconditional_post_del and _is_pos_after_span(line, col, span)
    ):
        state.pre_bound.discard(name)
        state.pre_unbound.add(name)
        state.pre_mutated.add(name)
    if _is_pos_after_span(line, col, span):
        state.post_mutated.add(name)
    if frame.in_loop:
        state.loop_mutated.add(name)


def _find_non_call_loaded_names(stmts: Sequence[ast.stmt]) -> Set[str]:
    """Finds variable names loaded outside direct Call.func positions in stmts."""
    loaded: Set[str] = set()
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
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


def _extract_definite_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names unconditionally bound by an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: Set[str] = set()
        for elt in target.elts:
            names.update(_extract_definite_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_definite_target_names(target.value)
    return set()


def _extract_shallow_deleted_names(node: ast.AST) -> Set[str]:
    """Extracts variable names deleted or unbound by ExceptHandler in node outside inner scopes."""
    deleted: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Del):
            deleted.add(curr.id)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            deleted.add(curr.name)
        stack.extend(ast.iter_child_nodes(curr))
    return deleted


def _extract_block_definite_bindings(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Extracts variable names unconditionally bound across a statement sequence."""
    bound: Set[str] = set(initial_bound)
    for stmt in stmts:
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            for hdr in iter_nested_scope_header_nodes(stmt):
                bound.difference_update(_extract_shallow_deleted_names(hdr))
            if _can_bind_in_fn_scope(stmt.name, active_nonlocals, fn_globals):
                bound.add(stmt.name)
            continue
        if isinstance(stmt, ast.If):
            bound.difference_update(_extract_shallow_deleted_names(stmt.test))
            if stmt.body and stmt.orelse:
                bound = _extract_block_definite_bindings(
                    stmt.body, active_nonlocals, bound, fn_globals
                ) & _extract_block_definite_bindings(
                    stmt.orelse, active_nonlocals, bound, fn_globals
                )
            else:
                bound.difference_update(_extract_shallow_deleted_names(stmt))
            continue
        bound.difference_update(_extract_shallow_deleted_names(stmt))
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
        if isinstance(stmt, ast.Assign) and stmt.value is not None:
            for tgt in stmt.targets:
                bound.update(
                    n
                    for n in _extract_definite_target_names(tgt)
                    if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
                )
        elif (
            isinstance(stmt, (ast.AnnAssign, ast.AugAssign))
            and stmt.value is not None
        ):
            bound.update(
                n
                for n in _extract_definite_target_names(stmt.target)
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imp_names: Set[str] = set()
            _record_import_names(stmt, imp_names)
            bound.update(
                n
                for n in imp_names
                if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
            )
    return bound


def _extract_if_exhaustive_bindings(
    if_node: ast.If,
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Extracts variable names unconditionally bound across both branches of an If statement."""
    if not if_node.body or not if_node.orelse:
        return set()
    post_test_bound = set(initial_bound) - _extract_shallow_deleted_names(
        if_node.test
    )
    return _extract_block_definite_bindings(
        if_node.body, active_nonlocals, post_test_bound, fn_globals
    ) & _extract_block_definite_bindings(
        if_node.orelse, active_nonlocals, post_test_bound, fn_globals
    )


def _push_compound_children(
    curr: ast.AST,
    span: _SpanBounds,
    frame: _StackFrame,
    unit_mutations: AbstractSet[str],
    state: _CellBindingState,
    stack: List[_StackFrame],
    *,
    unconditional_post_del: bool = False,
) -> bool:
    """Pushes child nodes of compound control-flow statements with branch and loop state."""
    is_br = frame.is_branched
    in_lp = frame.in_loop
    act_nl = frame.active_nonlocals
    encloses = _strictly_encloses_span(curr, span)

    if isinstance(curr, ast.If):
        body_span = encloses and _block_contains_span(curr.body, span)
        else_span = encloses and _block_contains_span(curr.orelse, span)
        if (
            not encloses
            and not is_br
            and curr.body
            and curr.orelse
            and _is_pos_before_span(
                getattr(curr, "end_lineno", getattr(curr, "lineno", None)),
                getattr(curr, "end_col_offset", getattr(curr, "col_offset", None)),
                span,
            )
        ):
            exh_bound = _extract_if_exhaustive_bindings(
                curr, act_nl, state.pre_bound, state.fn_globals
            )
            if exh_bound:
                stack.append(
                    _StackFrame(
                        curr, is_br, in_lp, act_nl, post_pre_bound=exh_bound
                    )
                )
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
                if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
            ]
            state.pre_mutated.update(t_stores)
            if not is_br:
                state.pre_bound.update(t_stores)
                state.pre_unbound.difference_update(t_stores)
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
        body_span = encloses and _block_contains_span(curr.body, span)
        if body_span:
            for item in curr.items:
                if item.optional_vars is not None:
                    w_stores = [
                        n
                        for n in _extract_store_names_in_node(item.optional_vars)
                        if _can_bind_in_fn_scope(n, act_nl, state.fn_globals)
                    ]
                    state.pre_mutated.update(w_stores)
                    if not is_br:
                        state.pre_bound.update(w_stores)
                        state.pre_unbound.difference_update(w_stores)
        _push_seq(stack, curr.body, is_br if body_span else True, in_lp, act_nl)
        _push_seq(stack, curr.items, is_br if body_span else True, in_lp, act_nl)
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
        if isinstance(curr.name, str) and _can_bind_in_fn_scope(
            curr.name, act_nl, state.fn_globals
        ):
            if encloses:
                state.pre_mutated.add(curr.name)
                state.post_mutated.add(curr.name)
                if not h_br:
                    state.pre_bound.add(curr.name)
                    state.pre_unbound.discard(curr.name)
                if in_lp:
                    state.loop_mutated.add(curr.name)
            else:
                _record_deletion_at_pos(
                    curr.name,
                    c_line,
                    c_col,
                    span,
                    frame,
                    state,
                    unconditional_post_del=unconditional_post_del,
                )
        _push_seq(stack, curr.body, h_br, in_lp, act_nl)
        if curr.type is not None:
            stack.append(_StackFrame(curr.type, True, in_lp, act_nl))
        return True

    if type(curr).__name__ == "Match":
        for case in reversed(getattr(curr, "cases", [])):
            case_body = getattr(case, "body", [])
            c_br = (
                is_br
                if (encloses and _block_contains_span(case_body, span))
                else True
            )
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


def _is_closure_factory_scope(
    scope_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Checks whether a nested function or class can return a closure or generator/coroutine."""
    if isinstance(scope_node, (ast.ClassDef, ast.AsyncFunctionDef)):
        return True
    if not isinstance(scope_node, ast.FunctionDef):
        return False
    own_locals = _extract_single_node_local_names(
        scope_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
    )
    parent_map = (
        scope_ctx.parent_map
        if scope_ctx is not None
        else _get_scope_parent_map(scope_node)
    )
    shadowed = (
        (scope_ctx.shadowed_names if scope_ctx is not None else set())
        | own_locals
    )
    local_ctx = _FnScopeContext(
        fn_locals=own_locals,
        shadowed_names=shadowed,
        parent_map=parent_map,
        has_star_import=(
            scope_ctx.has_star_import if scope_ctx is not None else False
        ),
    )
    has_nested_closure = False
    ret_exprs: List[ast.AST] = []
    stack: List[ast.AST] = list(scope_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(
            curr,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            has_nested_closure = True
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, local_ctx):
                has_nested_closure = True
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
        if isinstance(curr, ast.Return) and curr.value is not None:
            ret_exprs.append(curr.value)
        stack.extend(ast.iter_child_nodes(curr))
    if not has_nested_closure or not ret_exprs:
        return False
    ret_calls: Set[str] = set()
    r_reads, r_muts, r_loads = _extract_expr_closure_and_load_info(
        ret_exprs, local_ctx, called_names_out=ret_calls
    )
    return bool(r_reads or r_muts or r_loads or ret_calls)


def _extract_fn_returned_closure_refs(
    fn_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> Set[str]:
    """Extracts non-local variable/callable names returned by a nested function."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    own_locals = _extract_single_node_local_names(
        fn_node,
        include_globals=False,
        include_nonlocals=False,
        include_self_name=False,
    )
    ret_exprs: List[ast.AST] = []
    stack: List[ast.AST] = list(fn_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.Return) and curr.value is not None:
            ret_exprs.append(curr.value)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    if not ret_exprs:
        return set()
    ret_calls: Set[str] = set()
    _, _, ret_loads = _extract_expr_closure_and_load_info(
        ret_exprs, scope_ctx, called_names_out=ret_calls
    )
    return (ret_loads | ret_calls) - own_locals


def _handle_nested_scope_node(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]],
    stack: List[_StackFrame],
    scope_ctx: Optional[_FnScopeContext] = None,
    factory_names: Optional[Set[str]] = None,
    fn_return_refs: Optional[Dict[str, Set[str]]] = None,
    alias_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    call_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    sibling_nl_deletions: Optional[Set[str]] = None,
) -> bool:
    """Processes nested function, class, lambda, or comprehension scopes during cell analysis."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        estart = _node_earliest_start_lineno(curr)
        _record_binding_at_pos(
            (curr.name,), estart, c_col, span, frame, state
        )
        enclosing_cls = isinstance(
            curr, ast.ClassDef
        ) and _strictly_encloses_span(curr, span)
        if not enclosing_cls:
            free_r = _extract_cached_scope_free_reads(curr)
            nl_mut, nl_del = _extract_closure_nonlocal_mutations_and_deletions(
                curr
            )
            if sibling_nl_deletions is not None and nl_del:
                sibling_nl_deletions.update(nl_del)
            sib_name = (
                curr.name
                if _can_bind_in_fn_scope(
                    curr.name, frame.active_nonlocals, state.fn_globals
                )
                else None
            )
            if sib_name is not None:
                if factory_names is not None and _is_closure_factory_scope(
                    curr, scope_ctx
                ):
                    factory_names.add(sib_name)
                if fn_return_refs is not None and isinstance(
                    curr, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    ret_refs = _extract_fn_returned_closure_refs(curr, scope_ctx)
                    if ret_refs:
                        fn_return_refs.setdefault(sib_name, set()).update(
                            ret_refs
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
        return True

    if isinstance(curr, ast.Lambda):
        free_r = _extract_expr_scope_free_reads(curr, include_header=False)
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
        return True

    if isinstance(curr, _COMPREHENSION_TYPES):
        body_w = _extract_comprehension_body_walrus_names(curr)
        if body_w:
            _record_binding_at_pos(
                body_w,
                c_line,
                c_col,
                span,
                frame._replace(is_branched=True),
                state,
            )
        c_reads, c_muts = _extract_comprehension_deferred_closure_info(
            curr, scope_ctx
        )
        if c_reads or c_muts:
            sibling_closures.append((None, c_reads, c_muts))
        if alias_edges is not None:
            comp_sub_stack, comp_targets = (
                _iter_comprehension_body_exprs_and_targets(curr)
            )
            comp_carriers: List[ast.AST] = []
            while comp_sub_stack:
                sub = comp_sub_stack.pop()
                if isinstance(sub, (ast.NamedExpr, ast.Call)):
                    comp_carriers.append(sub)
                comp_sub_stack.extend(
                    iter_nested_scope_header_nodes(sub)
                    if isinstance(
                        sub,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Lambda,
                        ),
                    )
                    else ast.iter_child_nodes(sub)
                )
            for c_node in comp_carriers:
                _record_carrier_closures_and_aliases(
                    c_node,
                    frame,
                    sibling_closures,
                    alias_edges,
                    scope_ctx,
                    call_edges,
                    excluded_names=comp_targets,
                )
        if _is_pos_after_span(c_line, c_col, span):
            state.post_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
        if curr.generators:
            stack.append(
                _StackFrame(
                    curr.generators[0].iter,
                    frame.is_branched,
                    frame.in_loop,
                    frame.active_nonlocals,
                )
            )
        return True

    return False


def _record_carrier_closures_and_aliases(
    curr: ast.AST,
    frame: _StackFrame,
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]],
    alias_edges: List[Tuple[str, Set[str]]],
    scope_ctx: Optional[_FnScopeContext] = None,
    call_edges: Optional[List[Tuple[str, Set[str]]]] = None,
    *,
    excluded_names: AbstractSet[str] = frozenset(),
) -> None:
    """Extracts inline closures and alias edges from assignment or mutating call nodes."""
    carrier = _extract_closure_carrier_targets_and_values(curr)
    if carrier is None:
        return
    c_targets, c_values = carrier
    valid_targets = [
        t
        for t in c_targets
        if t not in excluded_names
        and _can_bind_in_fn_scope(t, frame.active_nonlocals)
    ]
    if not valid_targets:
        return
    c_calls: Set[str] = set()
    c_reads, c_muts, c_loads = _extract_expr_closure_and_load_info(
        c_values, scope_ctx, called_names_out=c_calls
    )
    if excluded_names:
        c_reads -= set(excluded_names)
        c_loads -= set(excluded_names)
        c_calls -= set(excluded_names)
    for t_name in valid_targets:
        if c_reads or c_muts:
            sibling_closures.append((t_name, set(c_reads), set(c_muts)))
        if c_loads:
            alias_edges.append((t_name, set(c_loads)))
        if c_calls and call_edges is not None:
            call_edges.append((t_name, set(c_calls)))


def _record_simple_node_bindings_and_reads(
    curr: ast.AST,
    c_line: Optional[int],
    c_col: Optional[int],
    span: _SpanBounds,
    frame: _StackFrame,
    state: _CellBindingState,
    *,
    unconditional_post_del: bool = False,
) -> None:
    """Records bindings, deletions, and post-span reads for leaf/simple AST nodes."""
    if isinstance(curr, ast.AugAssign) and isinstance(curr.target, ast.Name):
        if _is_pos_after_span(c_line, c_col, span):
            state.post_reads.add(curr.target.id)

    if isinstance(curr, ast.Name):
        if isinstance(curr.ctx, ast.Store):
            _record_binding_at_pos(
                (curr.id,), c_line, c_col, span, frame, state
            )
        elif isinstance(curr.ctx, ast.Del):
            _record_deletion_at_pos(
                curr.id,
                c_line,
                c_col,
                span,
                frame,
                state,
                unconditional_post_del=unconditional_post_del,
            )
        elif isinstance(curr.ctx, ast.Load) and _is_pos_after_span(
            c_line, c_col, span
        ):
            state.post_reads.add(curr.id)
    elif isinstance(curr, (ast.Import, ast.ImportFrom)):
        imp_names: Set[str] = set()
        _record_import_names(curr, imp_names)
        _record_binding_at_pos(imp_names, c_line, c_col, span, frame, state)
    else:
        bound_name = _extract_type_alias_or_pattern_name(curr)
        if bound_name is not None:
            _record_binding_at_pos(
                (bound_name,), c_line, c_col, span, frame, state
            )


def _analyze_enclosing_fn_cells_uncached(
    fn_node: ast.AST,
    span: _SpanBounds,
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
    enclosed_fn: Optional[ast.AST] = None,
    mod_tree: Optional[ast.AST] = None,
) -> _EnclosingFnCellInfo:
    """Computes variable bindings, post-span mutations, loop mutations, and sibling closures."""
    scope_ctx = _get_fn_scope_context(fn_node, mod_tree=mod_tree)
    fn_locals = set(scope_ctx.fn_locals)
    fn_globals = _extract_direct_nonlocals(fn_node, is_global=True)
    pre_bound: Set[str] = (
        set(_extract_node_type_param_names(fn_node)) - fn_globals
    )
    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        pre_bound.update(_extract_arg_names(fn_args) - fn_globals)

    state = _CellBindingState(
        pre_bound=pre_bound,
        pre_mutated=set(pre_bound),
        post_mutated=set(),
        post_reads=set(),
        loop_mutated=set(),
        pre_unbound=set(),
        fn_globals=fn_globals,
    )
    sibling_closures: List[Tuple[Optional[str], Set[str], Set[str]]] = []
    concurrent_sib_indices: Set[int] = set()
    alias_edges: List[Tuple[str, Set[str]]] = []
    call_edges: List[Tuple[str, Set[str]]] = []
    factory_names: Set[str] = set()
    fn_return_refs: Dict[str, Set[str]] = {}
    sibling_nl_deletions: Set[str] = set()
    enclosed_fn_name: Optional[str] = (
        enclosed_fn.name
        if isinstance(enclosed_fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    enclosed_fn_call_count = 0
    enclosed_fn_multi_invoked = False
    direct_enclosed_call_funcs: Set[int] = set()

    stack: List[_StackFrame] = [
        _StackFrame(stmt, False, False, None)
        for stmt in reversed(getattr(fn_node, "body", []))
    ]
    while stack:
        frame = stack.pop()
        if frame.post_pre_bound is not None:
            state.pre_bound.update(frame.post_pre_bound)
            state.pre_unbound.difference_update(frame.post_pre_bound)
            continue
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
            if (
                frame.in_loop
                or frame.active_nonlocals is not None
                or bool(getattr(curr, "decorator_list", None))
            ):
                enclosed_fn_multi_invoked = True
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

        prev_sib_len = len(sibling_closures)
        eff_line = (
            _node_earliest_start_lineno(curr)
            if isinstance(
                curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            else c_line
        )
        is_conc_pos = (
            enclosed_fn is not None
            or frame.in_loop
            or not _is_pos_after_span(eff_line, c_col, span)
        )
        if _handle_nested_scope_node(
            curr,
            c_line,
            c_col,
            span,
            frame,
            state,
            sibling_closures,
            stack,
            scope_ctx,
            factory_names,
            fn_return_refs,
            alias_edges,
            call_edges,
            sibling_nl_deletions,
        ):
            if is_conc_pos:
                concurrent_sib_indices.update(
                    range(prev_sib_len, len(sibling_closures))
                )
            if enclosed_fn_name is not None and not enclosed_fn_multi_invoked:
                if isinstance(
                    curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if not (
                        isinstance(curr, ast.ClassDef)
                        and _strictly_encloses_span(curr, span)
                    ) and enclosed_fn_name in _extract_cached_scope_free_reads(
                        curr
                    ):
                        enclosed_fn_multi_invoked = True
                elif isinstance(curr, (ast.Lambda, *_COMPREHENSION_TYPES)):
                    if enclosed_fn_name in _extract_expr_scope_free_reads(
                        curr, include_header=False
                    ):
                        enclosed_fn_multi_invoked = True
            continue

        if enclosed_fn_name is not None:
            if isinstance(curr, ast.Call):
                if (
                    isinstance(curr.func, ast.Name)
                    and curr.func.id == enclosed_fn_name
                ):
                    direct_enclosed_call_funcs.add(id(curr.func))
                    enclosed_fn_call_count += 1
                    if frame.in_loop or enclosed_fn_call_count >= 2:
                        enclosed_fn_multi_invoked = True
            elif (
                isinstance(curr, ast.Name)
                and isinstance(curr.ctx, ast.Load)
                and curr.id == enclosed_fn_name
                and id(curr) not in direct_enclosed_call_funcs
            ):
                enclosed_fn_multi_invoked = True

        _record_carrier_closures_and_aliases(
            curr,
            frame,
            sibling_closures,
            alias_edges,
            scope_ctx,
            call_edges,
        )
        if is_conc_pos:
            concurrent_sib_indices.update(
                range(prev_sib_len, len(sibling_closures))
            )

        if _push_compound_children(
            curr,
            span,
            frame,
            unit_mutations,
            state,
            stack,
            unconditional_post_del=enclosed_fn is not None,
        ):
            continue

        _record_simple_node_bindings_and_reads(
            curr,
            c_line,
            c_col,
            span,
            frame,
            state,
            unconditional_post_del=enclosed_fn is not None,
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

    if enclosed_fn is not None and enclosed_fn_multi_invoked:
        state.loop_mutated.update(unit_mutations)
        state.loop_mutated.update(_extract_closure_nonlocal_mutations(enclosed_fn))

    if sibling_nl_deletions:
        state.pre_bound.difference_update(sibling_nl_deletions)
        state.pre_unbound.update(sibling_nl_deletions)

    for t_name, called_set in call_edges:
        alias_targets: Set[str] = set(called_set & factory_names)
        visited_fn_refs: Set[str] = set()
        pending_fn_refs = list(called_set)
        while pending_fn_refs:
            fn_ref = pending_fn_refs.pop()
            if fn_ref in visited_fn_refs:
                continue
            visited_fn_refs.add(fn_ref)
            if fn_ref in factory_names:
                alias_targets.add(fn_ref)
            for ret_ref in fn_return_refs.get(fn_ref, ()):
                alias_targets.add(ret_ref)
                if ret_ref in fn_return_refs or ret_ref in factory_names:
                    pending_fn_refs.append(ret_ref)
        if alias_targets:
            alias_edges.append((t_name, alias_targets))

    sibling_closures = _propagate_sibling_closures(
        sibling_closures,
        post_reads=state.post_reads,
        alias_edges=alias_edges,
    )
    concurrent_sib_vars: Set[str] = set()
    for s_idx in concurrent_sib_indices:
        _, s_reads, s_muts = sibling_closures[s_idx]
        concurrent_sib_vars.update(s_reads)
        concurrent_sib_vars.update(s_muts)
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
        pre_unbound=state.pre_unbound,
        concurrent_sibling_vars=concurrent_sib_vars,
    )


def _analyze_enclosing_fn_cells(
    fn_node: ast.AST,
    span: Union[_SpanBounds, Tuple[int, int, Optional[int], Optional[int]]],
    *,
    unit_mutations: AbstractSet[str] = frozenset(),
    enclosed_fn: Optional[ast.AST] = None,
    mod_tree: Optional[ast.AST] = None,
) -> _EnclosingFnCellInfo:
    """Analyzes and caches cell bindings, post-span mutations, loop mutations, and closures."""
    norm_span = span if isinstance(span, _SpanBounds) else _SpanBounds(*span)
    cache_key: Any = (
        fn_node,
        norm_span,
        frozenset(unit_mutations),
        enclosed_fn,
        mod_tree,
    )
    return cast(
        _EnclosingFnCellInfo,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _analyze_enclosing_fn_cells_uncached(
                fn_node,
                norm_span,
                unit_mutations=unit_mutations,
                enclosed_fn=enclosed_fn,
                mod_tree=mod_tree,
            ),
        ),
    )


def _build_enclosing_chain_cell_infos(
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int],
    u_end_col: Optional[int],
    unit_mutations: AbstractSet[str],
    mod_tree: Optional[ast.AST] = None,
) -> List[_EnclosingFnCellInfo]:
    """Builds _EnclosingFnCellInfo summaries for each function in enclosing_chain."""
    rev_infos: List[_EnclosingFnCellInfo] = []
    escaping_muts: Set[str] = set(unit_mutations)
    for idx in range(len(enclosing_chain) - 1, -1, -1):
        _estart, _nend, fn_node = enclosing_chain[idx]
        if idx == len(enclosing_chain) - 1:
            span = _SpanBounds(u_start, u_end, u_start_col, u_end_col)
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
            enc_child = next_fn
        info = _analyze_enclosing_fn_cells(
            fn_node,
            span,
            unit_mutations=escaping_muts,
            enclosed_fn=enc_child,
            mod_tree=mod_tree,
        )
        rev_infos.append(info)
        escaping_muts = escaping_muts - info.fn_locals
    rev_infos.reverse()
    return rev_infos


def _find_enclosing_chain_definite_pre_bound(
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
    mod_tree: Optional[ast.AST] = None,
) -> Set[str]:
    """Computes variables definitely bound prior to the unit across enclosing functions."""
    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        u_start,
        u_end,
        u_start_col,
        u_end_col,
        frozenset(),
        mod_tree=mod_tree,
    )
    acc_pre_bound: Set[str] = set()
    acc_pre_unbound: Set[str] = set()
    for idx in range(len(chain_infos) - 1, -1, -1):
        info = chain_infos[idx]
        acc_pre_bound.update(info.pre_bound - acc_pre_unbound)
        acc_pre_unbound.update(
            (set(info.pre_unbound) | info.fn_locals) - info.pre_bound
        )
    return acc_pre_bound


def _check_captured_cell_mutations(
    captured_candidates: AbstractSet[str],
    eff_input_set: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    unit_closure_reads: Optional[AbstractSet[str]] = None,
) -> bool:
    """Checks 1 & 2: captured cell variable is unbound before unit or mutated outside unit."""
    actual_closure_reads = (
        captured_candidates
        if unit_closure_reads is None
        else unit_closure_reads
    )
    for var in captured_candidates:
        acc_ext_muts: Set[str] = set()
        acc_pre_bound: Set[str] = set()
        acc_pre_unbound: Set[str] = set()
        for idx in range(len(chain_infos) - 1, -1, -1):
            info = chain_infos[idx]
            acc_ext_muts.update(
                info.post_mutated
                | info.loop_mutated
                | info.all_sibling_nonlocal_mutations
            )
            acc_pre_bound.update(info.pre_bound - acc_pre_unbound)
            acc_pre_unbound.update(set(info.pre_unbound) - info.pre_bound)
            if var not in info.fn_locals:
                if var in actual_closure_reads:
                    acc_ext_muts.update(info.pre_mutated)
                continue
            if (
                var in eff_input_set and var not in acc_pre_bound
            ) or var in acc_ext_muts:
                return True
            break
    return False


def _check_intra_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    intra_nl_mutators: Sequence[Tuple[str, Set[str], bool]],
    unit_top_locals: AbstractSet[str],
    nonlocal_set: AbstractSet[str],
    intra_closure_reads: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    is_subroutine: bool = True,
) -> bool:
    """Check 3: intra-unit closure mutates an outer cell or escaping top-level variable."""
    if not intra_nl_mutators:
        return False
    all_chain_locals: Set[str] = (
        set().union(*(info.fn_locals for info in chain_infos))
        if chain_infos
        else set()
    )
    top_locals_set = set(unit_top_locals)
    for _mut_name, mutated_vars, _escaped_def in intra_nl_mutators:
        escaping_nl_muts = (mutated_vars - top_locals_set) | (
            mutated_vars & set(nonlocal_set)
        )
        if bool(escaping_nl_muts & all_chain_locals):
            return True

    if not is_subroutine:
        return False

    non_call_loads = _find_non_call_loaded_names(candidate_stmts)
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
        for var in mutated_vars & top_locals_set:
            if var in outer_touched or (
                mut_fn_escapes
                and (var in all_post_reads or var in all_sib_reads)
            ):
                return True
    return False


def _check_sibling_closure_sharing(
    eff_input_set: AbstractSet[str],
    sub_mutated: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    is_suspended: bool = False,
) -> bool:
    """Check 4: sibling closure referenced by unit shares a mutated cell with the unit."""
    if not chain_infos:
        return False
    all_chain_locals: Set[str] = set().union(
        *(info.fn_locals for info in chain_infos)
    )
    if is_suspended and bool(
        set(sub_mutated)
        & chain_infos[-1].fn_locals
        & set(chain_infos[-1].concurrent_sibling_vars)
    ):
        return True
    unshadowed_sibs: List[Tuple[Optional[str], Set[str], Set[str]]] = []
    unshadowed_edges: List[Tuple[str, Set[str]]] = []
    for idx, info in enumerate(chain_infos):
        inner_shadowed: Set[str] = (
            set().union(
                *(
                    chain_infos[j].fn_locals
                    for j in range(idx + 1, len(chain_infos))
                )
            )
            if idx + 1 < len(chain_infos)
            else set()
        )
        for s_name, s_reads, s_muts in info.sibling_closures:
            if s_name is not None and s_name in inner_shadowed:
                continue
            unshadowed_sibs.append(
                (s_name, s_reads - inner_shadowed, s_muts - inner_shadowed)
            )
        for t_name, src_names in info.alias_edges:
            if t_name in inner_shadowed:
                continue
            unshadowed_edges.append((t_name, src_names - inner_shadowed))
    all_sib_closures = _propagate_sibling_closures(
        unshadowed_sibs,
        alias_edges=unshadowed_edges,
    )
    sub_touched = set(eff_input_set) | set(sub_mutated)
    for sib_name, sib_reads, sib_nl_muts in all_sib_closures:
        if sib_name is None or sib_name not in eff_input_set:
            continue
        if bool(sib_reads & all_chain_locals & set(sub_mutated)):
            return True
        if bool(sib_nl_muts & all_chain_locals & sub_touched):
            return True
    return False


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
    skip_pre_unit_closures: bool = False


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings."""
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return False
    if not ctx.is_subroutine and ctx.unit_kind != "closure":
        return False

    enclosing_chain = _find_enclosing_function_chain(
        ctx.mod_tree, ctx.u_start, ctx.u_end, ctx.u_start_col
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
    intra_nl_mutators = _collect_unit_nonlocal_mutators(
        ctx.candidate_stmts, mod_tree=ctx.mod_tree
    )
    is_suspended = ctx.is_subroutine and (ctx.has_yield or ctx.is_async)
    if is_suspended:
        if bool(unit_mutated_names & (enc_escaping | nonlocal_set)) or bool(
            intra_nl_mutators
        ):
            return True

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    unit_closures = _collect_scope_closures_uncached(
        unit_mod,
        module_tree=ctx.mod_tree,
        genexp_include_header=False,
        cache_parent_map=False,
    )
    unit_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures))
        if unit_closures
        else set()
    )
    if ctx.is_subroutine:
        if is_suspended:
            unit_closure_reads.update(ctx.inputs)
    else:
        unit_closure_reads.update(ctx.free_vars)

    intra_closure_reads: Set[str] = (
        set(unit_closure_reads)
        if ctx.is_subroutine
        else (set(unit_closure_reads) | nonlocal_set)
    )

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    if ctx.is_subroutine and bool(
        unit_closure_reads
        & (
            (unit_mutated_names & (enc_escaping | nonlocal_set))
            | (enc_outer_globals & eff_input_set)
        )
    ):
        return True
    unit_mutations = unit_mutated_names if ctx.is_subroutine else set()

    if not intra_closure_reads and not intra_nl_mutators and not ctx.is_subroutine:
        return False

    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        ctx.u_start_col,
        ctx.u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
    )

    captured_candidates = intra_closure_reads & (
        eff_input_set | ((store_set | output_set) if ctx.is_subroutine else set())
    )
    sub_mutated = (
        unit_mutated_names
        if ctx.is_subroutine
        else (nonlocal_set & unit_mutated_names)
    )
    return (
        _check_intra_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            intra_nl_mutators,
            ctx.unit_top_locals,
            nonlocal_set,
            intra_closure_reads,
            chain_infos,
            is_subroutine=ctx.is_subroutine,
        )
        or _check_captured_cell_mutations(
            captured_candidates,
            eff_input_set,
            chain_infos,
            unit_closure_reads=unit_closure_reads,
        )
        or _check_sibling_closure_sharing(
            eff_input_set,
            sub_mutated,
            chain_infos,
            is_suspended=is_suspended and not ctx.skip_pre_unit_closures,
        )
    )
