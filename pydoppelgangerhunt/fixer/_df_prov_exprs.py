"""Expression, walrus, assignment-target, for-header, and import suppression provenance updates."""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.fixer._df_cfg import _CompoundClause
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
)
from pydoppelgangerhunt.fixer._df_exit_protocol import _STDLIB_NON_SUPPRESSING_MODULES
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _SHADOWABLE_BUILTIN_CTX_NAMES,
    _SUPPRESSING_CTX_MANAGER_NAMES,
    _SuppressProvenance,
    _ident_has_suspicious_suppress_token,
)
from pydoppelgangerhunt.fixer._df_with_rules import (
    _exprs_may_invoke_cross_scope_mutator,
    _extract_definite_target_names,
    _is_builtin_or_io_open_target,
    _is_direct_suppressor_expr,
    _is_pathlib_path_expr,
    _is_proven_non_suppressing_call,
    _is_proven_non_suppressing_target_expr,
    _is_suppress_alias_target_expr,
    _looks_like_suppressing_ctx_manager,
    _match_sequence_unpack_pairs,
    _resolve_tracked_factory_qualname,
    _unpack_fallback_rhs_is_suspicious,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

def _update_suppress_provenance_from_expr_walrus(
    expr_roots: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance from NamedExpr (walrus) bindings inside expression trees in left-to-right order."""
    w_stack: List[Tuple[ast.AST, bool, bool]] = [
        (root, allow_positive, False) for root in reversed(tuple(expr_roots))
    ]
    while w_stack:
        w_curr, cur_allow_pos, is_post_bind = w_stack.pop()
        if is_post_bind:
            if isinstance(w_curr, ast.NamedExpr) and isinstance(
                w_curr.target, ast.Name
            ):
                bound_prov = _bind_expr_target_provenance(
                    prov,
                    w_curr.target.id,
                    w_curr.value,
                    allow_positive=cur_allow_pos,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                prov = (
                    bound_prov
                    if cur_allow_pos
                    else prov.merge_branch_conservative(bound_prov)
                )
            continue
        if isinstance(w_curr, _COMPREHENSION_TYPES):
            if isinstance(w_curr, ast.DictComp):
                w_stack.append((w_curr.value, False, False))
                w_stack.append((w_curr.key, False, False))
            else:
                w_stack.append((w_curr.elt, False, False))
            if w_curr.generators:
                for idx in range(len(w_curr.generators) - 1, -1, -1):
                    gen = w_curr.generators[idx]
                    for if_e in reversed(gen.ifs):
                        w_stack.append((if_e, False, False))
                    w_stack.append((gen.iter, cur_allow_pos and idx == 0, False))
            continue
        if isinstance(w_curr, _INNER_SCOPE_TYPES):
            for hdr in reversed(tuple(iter_nested_scope_header_nodes(w_curr))):
                w_stack.append((hdr, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.NamedExpr):
            if isinstance(w_curr.target, ast.Name):
                w_stack.append((w_curr, cur_allow_pos, True))
            w_stack.append((w_curr.value, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.BoolOp):
            for idx in range(len(w_curr.values) - 1, -1, -1):
                w_stack.append((w_curr.values[idx], cur_allow_pos and idx == 0, False))
            continue
        if isinstance(w_curr, ast.Compare):
            for idx in range(len(w_curr.comparators) - 1, -1, -1):
                w_stack.append(
                    (
                        w_curr.comparators[idx],
                        cur_allow_pos and idx == 0,
                        False,
                    )
                )
            w_stack.append((w_curr.left, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.IfExp):
            w_stack.append((w_curr.orelse, False, False))
            w_stack.append((w_curr.body, False, False))
            w_stack.append((w_curr.test, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.Dict):
            for k_node, v_node in reversed(list(zip(w_curr.keys, w_curr.values))):
                if v_node is not None:
                    w_stack.append((v_node, cur_allow_pos, False))
                if k_node is not None:
                    w_stack.append((k_node, cur_allow_pos, False))
            continue
        for ch in reversed(tuple(ast.iter_child_nodes(w_curr))):
            if not isinstance(ch, ast.stmt):
                w_stack.append((ch, cur_allow_pos, False))
    return prov


def _bind_expr_target_provenance(
    prov: _SuppressProvenance,
    name: str,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    target_prov: Optional[_SuppressProvenance] = None,
) -> _SuppressProvenance:
    """Updates target_prov (defaulting to prov) when name is directly bound to value_expr evaluated in prov."""
    out_prov = prov if target_prov is None else target_prov
    while isinstance(value_expr, ast.NamedExpr):
        value_expr = value_expr.value
    target_src_expr = (
        value_expr.func if isinstance(value_expr, ast.Call) else value_expr
    )
    target_qual = (
        _resolve_tracked_factory_qualname(target_src_expr, prov)
        if allow_positive
        else None
    )
    if _is_direct_suppressor_expr(value_expr, prov):
        return out_prov.with_suppressor(
            name, allow_positive=allow_positive, imported_qualname=target_qual
        )
    if value_expr is not None:
        if _is_proven_non_suppressing_call(
            value_expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ) or (
            isinstance(value_expr, ast.Name)
            and value_expr.id in prov.non_suppressor_instances
            and value_expr.id not in prov.suppressors
        ):
            return out_prov.with_non_suppressor_instance(
                name, allow_positive=allow_positive, imported_qualname=target_qual
            )
        if _is_proven_non_suppressing_target_expr(
            value_expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            open_qual = (
                "builtins.open"
                if (
                    allow_positive
                    and target_qual is None
                    and _is_builtin_or_io_open_target(value_expr, prov)
                )
                else target_qual
            )
            return out_prov.with_non_suppressor(
                name, allow_positive=allow_positive, imported_qualname=open_qual
            )
        if allow_positive and _is_pathlib_path_expr(value_expr, prov):
            return out_prov.with_imported_symbol(
                name, "pathlib.Path", allow_positive=True
            )
        if _looks_like_suppressing_ctx_manager(value_expr, prov):
            return out_prov.with_suspicious_alias(name)
    return out_prov.clear_name(name)


def _bind_assign_target_provenance(
    rhs_prov: _SuppressProvenance,
    target: ast.AST,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    target_prov: _SuppressProvenance,
) -> _SuppressProvenance:
    """Binds assignment target provenance against value_expr, supporting nested and starred unpacking."""
    prov = target_prov
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(target, value_expr)]
    while stack:
        curr_target, curr_value = stack.pop()
        while isinstance(curr_value, ast.NamedExpr):
            curr_value = curr_value.value
        if isinstance(curr_target, ast.Name):
            prov = _bind_expr_target_provenance(
                rhs_prov,
                curr_target.id,
                curr_value,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                target_prov=prov,
            )
            continue
        matched_pairs = _match_sequence_unpack_pairs(
            curr_target, curr_value, allow_starred_target=True
        )
        if matched_pairs is not None:
            stack.extend(reversed(matched_pairs))
            continue
        is_susp_rhs = _unpack_fallback_rhs_is_suspicious(curr_value, rhs_prov)
        for t_name in sorted(_extract_definite_target_names(curr_target)):
            prov = (
                prov.with_suspicious_alias(t_name)
                if is_susp_rhs
                else prov.clear_name(t_name)
            )
    return prov


def _advance_clause_entry_prov(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Advances provenance through a clause header expression and clears clause-bound names."""
    prov = _advance_prov_through_opt_expr(
        header_expr,
        prov,
        allow_positive=False,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    for b_name in bound_names:
        prov = prov.clear_name(b_name)
    return prov


def _merge_prior_clause_guard_and_binds(
    prov: _SuppressProvenance,
    prior_hdr: Any,
    prior_bound: Iterable[str],
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Conservatively merges effects of evaluating a prior failed clause's bindings and guard/type."""
    branch = prov
    for b_name in prior_bound:
        branch = branch.clear_name(b_name)
    branch = _advance_clause_entry_prov(
        branch,
        prior_hdr,
        (),
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    return prov if branch is prov else prov.merge_branch_conservative(branch)


def _merge_prior_compound_clause_prov(
    prov: _SuppressProvenance,
    clause: _CompoundClause,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Conservatively advances provenance past a failed _CompoundClause header and pattern binds."""
    return _merge_prior_clause_guard_and_binds(
        prov,
        clause.guard_or_header,
        clause.bound_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _advance_prov_through_exprs(
    exprs: Sequence[ast.AST],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance across shallow expression roots (walrus + cross-scope call check)."""
    if not exprs:
        return prov
    updated = _update_suppress_provenance_from_expr_walrus(
        exprs,
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    if _exprs_may_invoke_cross_scope_mutator(
        exprs, updated, extra_non_suppressing_ctx_managers
    ):
        return updated.apply_cross_scope_invalidation()
    return updated


def _advance_prov_through_opt_expr(
    expr: Any,
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance across an optional header expression (walrus + cross-scope call check)."""
    if not isinstance(expr, ast.AST):
        return prov
    return _advance_prov_through_exprs(
        (expr,),
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _advance_prov_through_for_header(
    stmt: Union[ast.For, ast.AsyncFor],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[_SuppressProvenance, _SuppressProvenance]:
    """Returns (post_iter_prov, post_target_prov) after evaluating a For/AsyncFor header."""
    post_iter = _advance_prov_through_opt_expr(
        stmt.iter,
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    post_target = _update_suppress_provenance_from_expr_walrus(
        (stmt.target,),
        post_iter,
        allow_positive=False,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    iter_is_proven_sup = _is_suppress_alias_target_expr(stmt.iter, post_iter)
    iter_is_suspicious = (
        not iter_is_proven_sup
        and _unpack_fallback_rhs_is_suspicious(stmt.iter, post_iter)
    )
    for t_name in sorted(_extract_definite_target_names(stmt.target)):
        if iter_is_proven_sup and isinstance(stmt.target, ast.Name):
            post_target = post_target.with_suppressor(t_name, allow_positive=True)
        elif iter_is_proven_sup or iter_is_suspicious:
            post_target = post_target.with_suspicious_alias(t_name)
        else:
            post_target = post_target.clear_name(t_name)
    return post_iter, post_target


def _update_import_stmt_provenance(
    stmt: Union[ast.Import, ast.ImportFrom],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
) -> _SuppressProvenance:
    """Updates provenance for ast.Import and ast.ImportFrom statements."""
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if bound == "*":
                prov = prov.with_star_import()
                continue
            target_qual = (
                alias.name if alias.asname is not None else alias.name.split(".", 1)[0]
            )
            if alias.name in _STDLIB_NON_SUPPRESSING_MODULES and (
                alias.asname is not None or "." not in alias.name
            ):
                prov = prov.with_stdlib_module(
                    bound, alias.name, allow_positive=allow_positive
                )
            elif allow_positive and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES:
                prov = prov.with_imported_symbol(
                    bound, target_qual, allow_positive=allow_positive
                )
            else:
                prov = prov.clear_name(bound)
        return prov

    level = int(getattr(stmt, "level", 0) or 0)
    mod_name = stmt.module or ""
    is_abs_mod = level == 0 and bool(mod_name)
    rel_prefix = "." * level if level > 0 else ""
    allowed_nonsup = (
        _STDLIB_NON_SUPPRESSING_MODULES.get(mod_name, frozenset())
        if is_abs_mod
        else frozenset()
    )
    for alias in stmt.names:
        if alias.name == "*":
            prov = prov.with_star_import()
            continue
        bound = alias.asname or alias.name
        if is_abs_mod:
            imp_qual: Optional[str] = f"{mod_name}.{alias.name}"
        elif level > 0:
            imp_qual = (
                f"{rel_prefix}{mod_name}.{alias.name}"
                if mod_name
                else f"{rel_prefix}{alias.name}"
            )
        else:
            imp_qual = None
        if alias.name in allowed_nonsup:
            prov = prov.with_non_suppressor(
                bound,
                allow_positive=allow_positive,
                imported_qualname=imp_qual,
            )
        elif alias.name in _SUPPRESSING_CTX_MANAGER_NAMES:
            prov = prov.with_suppressor(
                bound,
                allow_positive=allow_positive,
                imported_qualname=imp_qual,
            )
        elif is_abs_mod and imp_qual in _STDLIB_NON_SUPPRESSING_MODULES:
            prov = prov.with_stdlib_module(
                bound,
                cast(str, imp_qual),
                allow_positive=allow_positive,
            )
        elif (
            alias.name != "catch_warnings"
            and _ident_has_suspicious_suppress_token(alias.name)
        ):
            prov = prov.with_suspicious_alias(
                bound,
                allow_positive=allow_positive,
                imported_qualname=(
                    imp_qual
                    if (
                        imp_qual is not None
                        and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES
                    )
                    else None
                ),
            )
        elif (
            imp_qual is not None
            and allow_positive
            and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES
        ):
            prov = prov.with_imported_symbol(
                bound,
                imp_qual,
                allow_positive=allow_positive,
            )
        else:
            prov = prov.clear_name(bound)
    return prov
