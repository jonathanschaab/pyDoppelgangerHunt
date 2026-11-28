"""Unified intra-procedural statement and CFG lowering shared across all dataflow walkers.

# INVARIANT (SINGLE STATEMENT/CFG LOWERING — LAYER 0):
# This module sits at Layer 0 of the fixer import DAG (depending only on ``ast``
# and ``_ast_compat`` at runtime) so EVERY intra-procedural statement/CFG consumer:
#   1. Forward suppression provenance walking (``_df_provenance_walk``)
#   2. Enclosing-statement seed advancement (``_df_module_seed``)
#   3. Block definite-assignment and termination analysis (``_df_block_flow``)
#   4. Enclosing-function closure-cell traversal (``_df_cell_traversal``)
#   5. Downstream-read and live-in AST visitors (``_df_visitors``)
#   6. Closure discovery and escaping-binding walkers (``_df_closures``)
#   7. Control-flow hazard and import-graph walkers (``_df_hazards``, ``depgraph``)
# MUST obtain compound-statement header expressions, lowered clause structure,
# match pattern bindings/exhaustiveness, shallow exit scans, unconditional-
# termination checks, loop ``orelse`` termination rules, ``try``/``finally``
# abrupt-entry and sibling-clause reachability rules, ``with`` header summaries,
# span locators, and branch-join semantics through this module rather than
# duplicating parallel ``if isinstance(stmt, ...)`` control-flow lowering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pydoppelgangerhunt._ast_compat import (
    is_match_as_node,
    is_match_mapping_node,
    is_match_node,
    is_match_or_node,
    is_match_star_node,
    is_try_or_trystar_node,
    is_trystar_node,
    is_type_alias_node,
)

if TYPE_CHECKING:
    from pydoppelgangerhunt.fixer._df_suppress_prov import _SuppressProvenance
    from pydoppelgangerhunt.fixer._df_with_rules import _WithItemProvenanceStep

_T = TypeVar("_T")
# Bounded so even deepest compound-CFG call chains (~6 Python frames per nesting level)
# stay well below CPython's default recursion limit (1000) under test runners.
_MAX_AST_RECURSION_DEPTH: int = 120


class _CompoundClause(NamedTuple):
    """Canonical lowered clause of a Python compound statement."""

    kind: str
    body: Sequence[ast.AST]
    bound_names: Tuple[str, ...] = ()
    guard_or_header: Optional[ast.AST] = None
    clear_bound_on_exit: bool = False
    is_irrefutable: bool = False
    pattern: Optional[ast.AST] = None
    clause_node: Optional[ast.AST] = None


class _ShallowExitSummary(NamedTuple):
    """Shallow abrupt-exit flags outside nested callable/class scopes."""

    has_return: bool
    has_break: bool
    has_continue: bool


class _WithHeaderSummary(NamedTuple):
    """Aggregated suppression and provenance summary across a With/AsyncWith header."""

    post_items_prov: _SuppressProvenance
    header_risky: bool
    after_sup: bool
    has_unproven_exit: bool
    steps: Tuple[_WithItemProvenanceStep, ...]


class _TrySpanLocation(NamedTuple):
    """Locates which clause of a Try/TryStar statement encloses a target span."""

    body_span: bool
    else_span: bool
    final_span: bool
    target_handler_idx: Optional[int]


class _MatchSpanLocation(NamedTuple):
    """Locates which case of a Match statement encloses a target span."""

    case_idx: Optional[int]
    in_case_body: bool


def _extract_pattern_bound_name(node: ast.AST) -> Optional[str]:
    """Extracts identifier bound by MatchAs, MatchStar, or MatchMapping pattern nodes."""
    name = getattr(node, "name", None) or getattr(node, "rest", None)
    if isinstance(name, str) and name and name != "_":
        return name
    return None


def _extract_pattern_bound_names(pattern: Optional[ast.AST]) -> Tuple[str, ...]:
    """Extracts all variable names bound by a PEP 634 match pattern in walk order."""
    if not isinstance(pattern, ast.AST):
        return ()
    return tuple(
        p_name
        for p_node in ast.walk(pattern)
        for p_name in (_extract_pattern_bound_name(p_node),)
        if p_name is not None
    )


def _extract_import_bound_names(
    stmt: Union[ast.Import, ast.ImportFrom],
) -> Tuple[str, ...]:
    """Extracts top-level identifiers lexically bound by an Import or ImportFrom node."""
    names: List[str] = []
    for alias in stmt.names:
        bound = alias.asname or alias.name.split(".", 1)[0]
        if bound and bound != "*":
            names.append(bound)
    return tuple(names)


def _extract_type_alias_name(node: ast.AST) -> Optional[str]:
    """Extracts the bound identifier name from a PEP 695 TypeAlias statement."""
    if not is_type_alias_node(node):
        return None
    target_name = getattr(node, "name", None)
    if isinstance(target_name, ast.Name):
        return target_name.id
    if isinstance(target_name, str) and target_name:
        return target_name
    return None


def _extract_type_alias_or_pattern_name(node: ast.AST) -> Optional[str]:
    """Extracts the bound identifier from a TypeAlias or match pattern AST node."""
    alias_name = _extract_type_alias_name(node)
    if alias_name is not None:
        return alias_name
    if (
        is_match_as_node(node)
        or is_match_star_node(node)
        or is_match_mapping_node(node)
    ):
        return _extract_pattern_bound_name(node)
    return None


def _extract_non_name_bound_names(node: ast.AST) -> Tuple[str, ...]:
    """Extracts lexical bindings from Import/ImportFrom, TypeAlias, or match pattern nodes."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _extract_import_bound_names(node)
    bound_name = _extract_type_alias_or_pattern_name(node)
    return (bound_name,) if bound_name is not None else ()


def _extract_arg_names(args: ast.arguments) -> Tuple[str, ...]:
    """Extracts all parameter names declared in an ast.arguments signature."""
    names = [a.arg for a in args.posonlyargs]
    names.extend(a.arg for a in args.args)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return tuple(names)


def _extract_type_param_names(node: ast.AST) -> Tuple[str, ...]:
    """Extracts PEP 695 type parameter names declared directly on an AST node."""
    return tuple(
        tp.name
        for tp in getattr(node, "type_params", ())
        if isinstance(getattr(tp, "name", None), str)
    )


def _iter_target_leaves(target: ast.AST) -> Iterator[ast.AST]:
    """Yields leaf assignment target nodes (Name, Attribute, Subscript) left-to-right."""
    stack: List[ast.AST] = [target]
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Tuple, ast.List)):
            stack.extend(reversed(curr.elts))
        elif isinstance(curr, ast.Starred):
            stack.append(curr.value)
        else:
            yield curr


def _extract_target_bound_names(target: Optional[ast.AST]) -> Tuple[str, ...]:
    """Extracts bare variable names bound by an assignment, loop, or with target."""
    if not isinstance(target, ast.AST):
        return ()
    return tuple(
        leaf.id
        for leaf in _iter_target_leaves(target)
        if isinstance(leaf, ast.Name)
    )


def _extract_store_names_in_node(node: Optional[ast.AST]) -> Set[str]:
    """Extracts all ast.Name store identifiers inside an AST subtree."""
    if not isinstance(node, ast.AST):
        return set()
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    }


class _ScopeDirectBindings(NamedTuple):
    """Direct lexical bindings and global/nonlocal declarations in one scope body."""

    locals: FrozenSet[str]
    globals: FrozenSet[str]
    nonlocals: FrozenSet[str]


_NESTED_DEF_SCOPE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)
_COMP_EXPR_TYPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _collect_shallow_named_expr_targets(
    root: ast.AST,
    out: Set[str],
    *,
    iter_header_nodes: Optional[Callable[[ast.AST], Sequence[ast.AST]]] = None,
    extract_comp_walrus: Optional[Callable[[ast.AST], Iterable[str]]] = None,
) -> None:
    """Collects NamedExpr targets binding in the current lexical scope under root."""
    stack: List[ast.AST] = [root]
    while stack:
        child = stack.pop()
        if isinstance(child, _NESTED_DEF_SCOPE_TYPES):
            if iter_header_nodes is not None:
                stack.extend(iter_header_nodes(child))
            continue
        if is_type_alias_node(child):
            continue
        if isinstance(child, _COMP_EXPR_TYPES):
            if extract_comp_walrus is not None:
                out.update(extract_comp_walrus(child))
                continue
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            out.add(child.target.id)
        stack.extend(ast.iter_child_nodes(child))


def _collect_scope_direct_bindings(
    stmts: Sequence[ast.AST],
    *,
    iter_header_nodes: Optional[Callable[[ast.AST], Sequence[ast.AST]]] = None,
    extract_comp_walrus: Optional[Callable[[ast.AST], Iterable[str]]] = None,
    include_imports: bool = True,
    include_except_names: bool = True,
    include_bare_annassign: bool = True,
    exclude_declared_globals: bool = True,
) -> _ScopeDirectBindings:
    """Collects names lexically bound directly in the immediate scope of stmts.

    Uses ``_lower_compound_clauses`` so every Python binding form (def/class,
    import, assign/annassign/augassign, for/with, except, match patterns,
    PEP 695 type alias, global/nonlocal, and walrus expressions) is extracted
    symmetrically from a single Layer-0 implementation.
    """
    local_names: Set[str] = set()
    global_names: Set[str] = set()
    nonlocal_names: Set[str] = set()

    def _collect_walrus_from(roots: Iterable[ast.AST]) -> None:
        for expr_root in roots:
            _collect_shallow_named_expr_targets(
                expr_root,
                local_names,
                iter_header_nodes=iter_header_nodes,
                extract_comp_walrus=extract_comp_walrus,
            )

    worklist: List[Sequence[ast.AST]] = [stmts]
    while worklist:
        seq = worklist.pop()
        for stmt in seq:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local_names.add(stmt.name)
                if iter_header_nodes is not None:
                    _collect_walrus_from(iter_header_nodes(stmt))
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                if include_imports:
                    local_names.update(_extract_import_bound_names(stmt))
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    local_names.update(_extract_target_bound_names(target))
            elif isinstance(stmt, ast.AugAssign):
                local_names.update(_extract_target_bound_names(stmt.target))
            elif isinstance(stmt, ast.AnnAssign):
                if stmt.value is not None or include_bare_annassign:
                    local_names.update(_extract_target_bound_names(stmt.target))
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                local_names.update(_extract_target_bound_names(stmt.target))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    local_names.update(_extract_target_bound_names(item.optional_vars))
            elif isinstance(stmt, ast.Global):
                global_names.update(stmt.names)
            elif isinstance(stmt, ast.Nonlocal):
                nonlocal_names.update(stmt.names)
            else:
                alias_name = _extract_type_alias_name(stmt)
                if alias_name is not None:
                    local_names.add(alias_name)

            clauses = _lower_compound_clauses(stmt)
            if clauses:
                for clause in clauses:
                    if clause.kind == "handler" and not include_except_names:
                        pass
                    elif clause.bound_names:
                        local_names.update(clause.bound_names)
                    if clause.guard_or_header is not None:
                        _collect_walrus_from((clause.guard_or_header,))
                    if clause.body:
                        worklist.append(clause.body)
                _collect_walrus_from(_extract_compound_header_nodes(stmt))
            else:
                _collect_walrus_from((stmt,))

    rebound_elsewhere = (
        (global_names | nonlocal_names) if exclude_declared_globals else nonlocal_names
    )
    return _ScopeDirectBindings(
        locals=frozenset(local_names - rebound_elsewhere),
        globals=frozenset(global_names),
        nonlocals=frozenset(nonlocal_names),
    )


def _is_irrefutable_pattern(
    pattern: Optional[ast.AST],
) -> bool:  # pragma: no cover (py310+)
    """Determines whether a pattern matching AST node unconditionally matches any subject.

    Note:
        Passing ``None`` directly returns ``False`` (no pattern node exists). Inside an
        ``ast.MatchAs`` node, ``pattern.pattern is None`` represents a wildcard ``case _:``
        or bare capture ``case x:`` and matches unconditionally.
    """
    if pattern is None:
        return False
    stack: List[Optional[ast.AST]] = [pattern]
    while stack:
        curr = stack.pop()
        if curr is None:
            continue
        if is_match_as_node(curr):
            sub_pat = getattr(curr, "pattern", None)
            if sub_pat is None:
                return True
            stack.append(sub_pat)
        elif is_match_or_node(curr):
            stack.extend(getattr(curr, "patterns", []))
    return False


def _is_irrefutable_case(case: ast.AST) -> bool:  # pragma: no cover (py310+)
    """Returns True if the match case has no guard and an irrefutable pattern."""
    if getattr(case, "guard", None) is not None:
        return False
    return _is_irrefutable_pattern(getattr(case, "pattern", None))


# INVARIANT (SHARED CFG LOWERING CONTRACT):
# Every Python compound control-flow statement handled by dataflow walkers
# (`_df_provenance_walk`, `_df_block_flow`, `_df_cell_frames`, `_df_module_seed`,
# and `scope._find_enclosing_pre_bound_names`) MUST be registered here in
# `_is_compound_cfg_stmt`, `_is_branching_cfg_node`, `_extract_compound_header_nodes`,
# and `_lower_compound_clauses` first so syntax additions stay symmetric across engines.
def _is_compound_cfg_stmt(node: ast.AST) -> bool:
    """Returns True when *node* is a Python compound control-flow statement."""
    return (
        isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.With,
                ast.AsyncWith,
            ),
        )
        or is_try_or_trystar_node(node)
        or is_match_node(node)
    )


def _is_branching_cfg_node(node: ast.AST) -> bool:
    """Returns True when *node* introduces conditional/exceptional execution paths."""
    return (
        isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.ExceptHandler,
            ),
        )
        or is_trystar_node(node)
        or is_match_node(node)
    )


def _extract_compound_header_nodes(stmt: ast.AST) -> List[ast.AST]:
    """Returns header expressions evaluated by a compound statement, or ``[stmt]`` for leaf statements."""
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.target, stmt.iter]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        headers: List[ast.AST] = []
        for item in stmt.items:
            headers.append(item.context_expr)
            if item.optional_vars is not None:
                headers.append(item.optional_vars)
        return headers
    if is_try_or_trystar_node(stmt):
        handlers = getattr(stmt, "handlers", ())
        return [h.type for h in handlers if getattr(h, "type", None) is not None]
    if is_match_node(stmt):
        return [getattr(stmt, "subject")]
    return [stmt]


def _lower_compound_clauses(stmt: ast.AST) -> Tuple[_CompoundClause, ...]:
    """Lowers any Python compound statement into its canonical ordered sequence of clauses."""
    clauses: List[_CompoundClause] = []
    if isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        clauses.append(_CompoundClause(kind="body", body=stmt.body))
        if stmt.orelse:
            clauses.append(_CompoundClause(kind="orelse", body=stmt.orelse))
    elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.ClassDef)):
        clauses.append(_CompoundClause(kind="body", body=stmt.body))
    elif is_try_or_trystar_node(stmt):
        clauses.append(_CompoundClause(kind="body", body=getattr(stmt, "body", ())))
        for h_node in getattr(stmt, "handlers", ()):
            h_name = getattr(h_node, "name", None)
            clauses.append(
                _CompoundClause(
                    kind="handler",
                    body=getattr(h_node, "body", ()),
                    bound_names=(h_name,) if isinstance(h_name, str) else (),
                    guard_or_header=getattr(h_node, "type", None),
                    clear_bound_on_exit=True,
                    clause_node=h_node,
                )
            )
        o_stmts = getattr(stmt, "orelse", ())
        if o_stmts:
            clauses.append(_CompoundClause(kind="orelse", body=o_stmts))
        f_stmts = getattr(stmt, "finalbody", ())
        if f_stmts:
            clauses.append(_CompoundClause(kind="finalbody", body=f_stmts))
    elif is_match_node(stmt):
        for case in getattr(stmt, "cases", ()):
            pattern = getattr(case, "pattern", None)
            clauses.append(
                _CompoundClause(
                    kind="case",
                    body=getattr(case, "body", ()),
                    bound_names=_extract_pattern_bound_names(pattern),
                    guard_or_header=getattr(case, "guard", None),
                    clear_bound_on_exit=False,
                    is_irrefutable=_is_irrefutable_case(case),
                    pattern=pattern if isinstance(pattern, ast.AST) else None,
                    clause_node=case,
                )
            )
    return tuple(clauses)


def _block_unconditionally_terminates(
    statements: Sequence[ast.AST],
    *,
    include_try: bool = False,
    _depth: int = 0,
) -> bool:
    """Checks whether a sequence of statements unconditionally returns or raises."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    for stmt in statements:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if (
                stmt.orelse
                and _block_unconditionally_terminates(
                    stmt.body, include_try=include_try, _depth=_depth + 1
                )
                and _block_unconditionally_terminates(
                    stmt.orelse, include_try=include_try, _depth=_depth + 1
                )
            ):
                return True
        elif include_try and is_try_or_trystar_node(stmt):
            finalbody = getattr(stmt, "finalbody", None)
            if finalbody and _block_unconditionally_terminates(
                finalbody, include_try=True, _depth=_depth + 1
            ):
                return True
            handlers = getattr(stmt, "handlers", ())
            body = getattr(stmt, "body", ())
            if handlers and _block_unconditionally_terminates(
                body, include_try=True, _depth=_depth + 1
            ):
                if all(
                    _block_unconditionally_terminates(
                        h.body, include_try=True, _depth=_depth + 1
                    )
                    for h in handlers
                ):
                    return True
    return False


def _scan_stmts_shallow_exits(stmts: Sequence[ast.AST]) -> _ShallowExitSummary:
    """Returns (has_return, has_unescaped_break, has_unescaped_continue) outside nested scopes."""
    has_ret = False
    has_brk = False
    has_cont = False
    stack: List[Tuple[ast.AST, int]] = [(s, 0) for s in stmts]
    while stack:
        curr, loop_depth = stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(curr, ast.Return):
            has_ret = True
        elif loop_depth == 0:
            if isinstance(curr, ast.Break):
                has_brk = True
            elif isinstance(curr, ast.Continue):
                has_cont = True
        if has_ret and has_brk and has_cont:
            break
        next_depth = (
            loop_depth + 1
            if isinstance(curr, (ast.For, ast.AsyncFor, ast.While))
            else loop_depth
        )
        for child in ast.iter_child_nodes(curr):
            stack.append((child, next_depth))
    return _ShallowExitSummary(has_ret, has_brk, has_cont)


def _finally_may_suppress_active_exception(f_body: Sequence[ast.AST]) -> bool:
    """Returns True if f_body contains return/break/continue targeting an enclosing scope."""
    exits = _scan_stmts_shallow_exits(f_body)
    return exits.has_return or exits.has_break or exits.has_continue


def _loop_orelse_terminates(
    body: Sequence[ast.AST],
    orelse: Sequence[ast.AST],
    orelse_terminated: bool,
) -> bool:
    """Returns True when a loop's orelse guarantees loop termination (no unescaped break in body)."""
    return (
        bool(orelse)
        and orelse_terminated
        and not _scan_stmts_shallow_exits(body).has_break
    )


def _try_has_abrupt_entry(
    t_body: Sequence[ast.AST],
    o_stmts: Sequence[ast.AST],
    *,
    body_has_risky: bool,
    orelse_has_risky: bool,
    body_terminated: bool,
) -> bool:
    """Returns True if finalbody of a Try/TryStar can be entered via exception or abrupt jump."""
    return bool(
        body_has_risky
        or orelse_has_risky
        or any(_scan_stmts_shallow_exits(t_body))
        or (o_stmts and not body_terminated and any(_scan_stmts_shallow_exits(o_stmts)))
    )


def _is_try_sibling_reachable_after(
    stmt: ast.AST,
    enclosing_clause_idx: int,
    sib_idx: int,
    sib_body: Sequence[ast.AST],
    enclosing_body: Sequence[ast.AST],
    has_target_node: bool,
) -> bool:
    """Returns True if sibling clause sib_idx of Try/TryStar can run after enclosing_clause_idx."""
    if not has_target_node:
        return False
    f_stmts = getattr(stmt, "finalbody", ())
    if f_stmts and enclosing_body is f_stmts:
        return False
    if enclosing_clause_idx == 0:
        return sib_idx > 0
    if f_stmts and sib_body is f_stmts:
        return True
    o_stmts = getattr(stmt, "orelse", ())
    if o_stmts and (enclosing_body is o_stmts or sib_body is o_stmts):
        return False
    return is_trystar_node(stmt) and sib_idx > enclosing_clause_idx


def _summarize_with_header_steps(
    steps: Sequence[_WithItemProvenanceStep],
    base_prov: _SuppressProvenance,
) -> _WithHeaderSummary:
    """Aggregates header risk, suppression, exit-invalidation, and post-item provenance."""
    step_tuple = tuple(steps)
    item_prov = base_prov
    header_risky = False
    after_sup = False
    has_unproven_exit = False
    for step in step_tuple:
        if step.after_prior_sup or step.opt_vars_conditional:
            header_risky = True
        if step.item_may_suppress:
            after_sup = True
        if not step.item_is_proven_non_suppressing:
            has_unproven_exit = True
        item_prov = step.post_item_prov
    return _WithHeaderSummary(
        post_items_prov=item_prov,
        header_risky=header_risky,
        after_sup=after_sup,
        has_unproven_exit=has_unproven_exit,
        steps=step_tuple,
    )


def _apply_with_exit_invalidation(
    prov: _SuppressProvenance,
    has_unproven_exit: bool,
) -> _SuppressProvenance:
    """Invalidates cross-scope mutated provenance after exiting an unproven context manager."""
    if has_unproven_exit and prov.cross_scope_mutated:
        return prov.apply_cross_scope_invalidation()
    return prov


def _locate_enclosing_clause_idx(
    clauses: Sequence[_CompoundClause],
    contains_fn: Callable[[Sequence[ast.AST]], bool],
) -> Optional[int]:
    """Returns the index of the first compound clause whose body satisfies contains_fn."""
    for idx, clause in enumerate(clauses):
        if contains_fn(clause.body):
            return idx
    return None


def _locate_try_span(
    stmt: ast.AST,
    encloses: bool,
    block_contains: Callable[[Sequence[ast.AST]], bool],
    node_contains: Callable[[Optional[ast.AST]], bool],
) -> _TrySpanLocation:
    """Identifies which clause (body, orelse, finalbody, or handler index) encloses a target span."""
    if not encloses:
        return _TrySpanLocation(False, False, False, None)
    body_l = getattr(stmt, "body", ())
    else_l = getattr(stmt, "orelse", ())
    final_l = getattr(stmt, "finalbody", ())
    handlers_l = getattr(stmt, "handlers", ())
    else_span = block_contains(else_l)
    body_span = block_contains(body_l) or else_span
    final_span = block_contains(final_l)
    target_h_idx: Optional[int] = None
    for h_idx, h_node in enumerate(handlers_l):
        if block_contains(getattr(h_node, "body", ())) or node_contains(
            getattr(h_node, "type", None)
        ):
            target_h_idx = h_idx
            break
    return _TrySpanLocation(body_span, else_span, final_span, target_h_idx)


def _locate_match_span(
    stmt: ast.AST,
    encloses: bool,
    block_contains: Callable[[Sequence[ast.AST]], bool],
    node_contains: Callable[[Optional[ast.AST]], bool],
) -> _MatchSpanLocation:
    """Identifies which case of a Match statement encloses a target span."""
    if not encloses:
        return _MatchSpanLocation(None, False)
    for idx, case in enumerate(getattr(stmt, "cases", ())):
        if block_contains(getattr(case, "body", ())):
            return _MatchSpanLocation(idx, True)
        if node_contains(getattr(case, "pattern", None)) or node_contains(
            getattr(case, "guard", None)
        ):
            return _MatchSpanLocation(idx, False)
    return _MatchSpanLocation(None, False)


def _merge_surviving_branches(
    surviving: Sequence[_T],
    fallback: Sequence[_T],
    merge_fn: Callable[[_T, _T], _T],
) -> _T:
    """Folds surviving branch states with merge_fn, falling back to fallback when all terminate."""
    pool = surviving if surviving else fallback
    acc = pool[0]
    for item in pool[1:]:
        acc = merge_fn(acc, item)
    return acc


_U = TypeVar("_U")


# INVARIANT (SINGLE CFG TRANSFER ENGINE — PROVENANCE & BLOCK FLOW):
# Both standalone suppression provenance walking (``_df_provenance_walk``) and
# definite-assignment block flow analysis (``_df_block_flow``) share these
# Layer-0 CFG transfer combinators for multi-item ``With`` prefix/suffix splits,
# PEP 634 ``Match`` sequential case/guard/irrefutable termination, and
# ``Try``/``TryStar`` sequential handler evaluation. Keeping the control-flow
# skeleton here prevents the two passes from drifting on evaluation order,
# irrefutable-case short-circuiting, or exception-skip edges.


def _run_with_cfg(
    safe_prefix: Sequence[ast.stmt],
    risky_suffix: Sequence[ast.stmt],
    *,
    eval_prefix: Callable[[Sequence[ast.stmt]], Tuple[_T, bool]],
    eval_suffix: Callable[[Sequence[ast.stmt], _T], _T],
    finalize_exit: Callable[[_T, bool], _U],
) -> _U:
    """Runs the canonical 2-phase With/AsyncWith CFG transfer (safe prefix -> optional risky suffix)."""
    pre_state, pre_terminated = eval_prefix(safe_prefix)
    if pre_terminated or not risky_suffix:
        return finalize_exit(pre_state, pre_terminated)
    post_state = eval_suffix(risky_suffix, pre_state)
    return finalize_exit(post_state, False)


@dataclass(frozen=True)
class _MatchCfgResult(Generic[_T, _U]):
    """Aggregated case outcomes, irrefutability summary, and post-match prior state."""

    has_irrefutable: bool
    all_cases_terminated: bool
    case_outcomes: Tuple[_T, ...]
    surviving_outcomes: Tuple[_T, ...]
    final_prior: _U


def _run_match_cfg(
    clauses: Sequence[_CompoundClause],
    *,
    initial_prior: _U,
    eval_case: Callable[[_CompoundClause, _U], Tuple[_T, bool]],
    advance_prior: Callable[[_U, _CompoundClause], _U],
) -> _MatchCfgResult[_T, _U]:
    """Runs canonical PEP 634 Match case sequencing with irrefutable short-circuiting."""
    case_outcomes: List[_T] = []
    surviving_outcomes: List[_T] = []
    has_irrefutable = False
    all_cases_terminated = True
    prior_state = initial_prior
    for clause in clauses:
        outcome, terminated = eval_case(clause, prior_state)
        case_outcomes.append(outcome)
        if not terminated:
            all_cases_terminated = False
            surviving_outcomes.append(outcome)
        if clause.is_irrefutable:
            has_irrefutable = True
            break
        prior_state = advance_prior(prior_state, clause)
    return _MatchCfgResult(
        has_irrefutable=has_irrefutable,
        all_cases_terminated=all_cases_terminated,
        case_outcomes=tuple(case_outcomes),
        surviving_outcomes=tuple(surviving_outcomes),
        final_prior=prior_state,
    )


@dataclass(frozen=True)
class _TryHandlersCfgResult(Generic[_T]):
    """Aggregated handler outcomes and termination flag from _run_try_handlers_cfg."""

    all_handlers_terminated: bool
    handler_outcomes: Tuple[_T, ...]
    surviving_outcomes: Tuple[_T, ...]


def _run_try_handlers_cfg(
    handlers: Sequence[ast.AST],
    *,
    is_trystar: bool,
    eval_handler: Callable[[ast.AST], Tuple[_T, bool]],
    advance_prior_handler: Callable[[ast.AST, _T, bool], None],
) -> _TryHandlersCfgResult[_T]:
    """Runs canonical Try/TryStar handler sequencing including except* sequential chaining."""
    handler_outcomes: List[_T] = []
    surviving_outcomes: List[_T] = []
    all_handlers_terminated = True
    for h_node in handlers:
        outcome, terminated = eval_handler(h_node)
        handler_outcomes.append(outcome)
        if not terminated:
            all_handlers_terminated = False
            surviving_outcomes.append(outcome)
        advance_prior_handler(h_node, outcome, is_trystar and not terminated)
    return _TryHandlersCfgResult(
        all_handlers_terminated=all_handlers_terminated,
        handler_outcomes=tuple(handler_outcomes),
        surviving_outcomes=tuple(surviving_outcomes),
    )
