"""Pure non-raising expression and safe-update proof predicates for suppressing with-blocks.

# INVARIANT (CONVERGENCE BOUNDARY — FINITE NON-RAISING PROOF):
# Proving arbitrary untyped Python expressions cannot raise is undecidable.
# This module MUST remain restricted to a closed syntactic subset of strictly
# non-raising Python forms (literals, proven-bound ``Name`` loads, unstarred
# tuple/list literals, ``is``/``is not`` comparisons, ``not``/literal unary ops,
# ``Lambda`` with non-raising defaults, ``BoolOp``/``IfExp``/``NamedExpr`` over
# that same closed subset). Do NOT expand this prover into general type/value
# inference, dunder-dispatch analysis, or cross-call purity proofs; statements
# outside this finite subset conservatively fail closed as potentially raising.
"""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_visitors import _MAX_AST_RECURSION_DEPTH
from pydoppelgangerhunt.fixer._df_with_rules import (
    _extract_definite_target_names,
    _match_sequence_unpack_pairs,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


def _is_non_raising_bool_expr(
    expr: Optional[ast.AST],
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    strict_names: bool = True,
    _depth: int = 0,
) -> bool:
    """Returns True if evaluating bool(expr) cannot raise an exception."""
    return _is_non_raising_expr(
        expr,
        safe_names,
        safe_bool_names,
        require_bool_safe=True,
        strict_names=strict_names,
        _depth=_depth,
    )


def _apply_non_raising_if_branch_updates(
    node: Union[ast.If, ast.IfExp],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    is_conditional: bool = False,
    _depth: int = 0,
) -> None:
    """Applies walrus/statement safe updates across test, body, and orelse of If/IfExp."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    _apply_non_raising_expr_walrus_updates(
        node.test,
        cur_safe,
        cur_bool,
        strict_names=strict_names,
        is_conditional=is_conditional,
        _depth=_depth + 1,
    )
    then_safe, then_bool = set(cur_safe), set(cur_bool)
    else_safe, else_bool = set(cur_safe), set(cur_bool)
    if isinstance(node, ast.IfExp):
        _apply_non_raising_expr_walrus_updates(
            node.body,
            then_safe,
            then_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
        _apply_non_raising_expr_walrus_updates(
            node.orelse,
            else_safe,
            else_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
    else:
        _apply_non_raising_seq_safe_updates(
            node.body,
            then_safe,
            then_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
        _apply_non_raising_seq_safe_updates(
            node.orelse,
            else_safe,
            else_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    cur_safe.clear()
    cur_safe.update(then_safe & else_safe)
    cur_bool.clear()
    cur_bool.update(then_bool & else_bool)


def _apply_non_raising_expr_walrus_updates(
    expr: Optional[ast.AST],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    is_conditional: bool = False,
    _depth: int = 0,
) -> None:
    """Applies left-to-right cur_safe and cur_bool updates from NamedExpr nodes in a non-raising expr."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    if expr is None or isinstance(expr, ast.Constant):
        return
    if isinstance(expr, ast.IfExp):
        _apply_non_raising_if_branch_updates(
            expr,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
        return
    if isinstance(expr, ast.NamedExpr):
        _apply_non_raising_expr_walrus_updates(
            expr.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
        if isinstance(expr.target, ast.Name):
            t_name = expr.target.id
            val_bool = _is_non_raising_bool_expr(
                expr.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            if not is_conditional:
                cur_safe.add(t_name)
                if val_bool:
                    cur_bool.add(t_name)
                else:
                    cur_bool.discard(t_name)
            elif not val_bool:
                cur_bool.discard(t_name)
        return
    sub_exprs: Sequence[Tuple[ast.AST, bool]] = ()
    if isinstance(expr, ast.Lambda):
        sub_exprs = tuple(
            (d, is_conditional)
            for d in (
                *expr.args.defaults,
                *(d for d in expr.args.kw_defaults if d is not None),
            )
        )
    elif isinstance(expr, (ast.Tuple, ast.List)):
        sub_exprs = tuple((elt, is_conditional) for elt in expr.elts)
    elif isinstance(expr, ast.UnaryOp):
        sub_exprs = ((expr.operand, is_conditional),)
    elif isinstance(expr, ast.BoolOp):
        sub_exprs = tuple(
            (val, is_conditional or idx > 0) for idx, val in enumerate(expr.values)
        )
    elif isinstance(expr, ast.Compare):
        sub_exprs = (
            (expr.left, is_conditional),
            *(
                (comp, is_conditional or idx > 0)
                for idx, comp in enumerate(expr.comparators)
            ),
        )
    elif isinstance(expr, ast.Call):
        sub_exprs = (
            (expr.func, is_conditional),
            *((arg, is_conditional) for arg in expr.args),
            *((kw.value, is_conditional) for kw in expr.keywords),
        )
    for sub_node, sub_cond in sub_exprs:
        _apply_non_raising_expr_walrus_updates(
            sub_node,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=sub_cond,
            _depth=_depth + 1,
        )


def _check_and_advance_non_raising_seq(
    nodes: Sequence[ast.AST],
    safe_names: AbstractSet[str],
    safe_bool_names: AbstractSet[str],
    *,
    strict_names: bool,
    require_bool_safe: bool = False,
    conditional_from_idx: Optional[int] = None,
    _depth: int = 0,
) -> Optional[Tuple[Set[str], Set[str]]]:
    """Verifies each node in nodes is non-raising left-to-right and returns updated (safe, bool) sets."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return None
    cur_safe = set(safe_names)
    cur_bool = set(safe_bool_names)
    for idx, node in enumerate(nodes):
        ok = (
            _is_non_raising_bool_expr(
                node,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            if require_bool_safe
            else _is_non_raising_expr(
                node,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
        )
        if not ok:
            return None
        _apply_non_raising_expr_walrus_updates(
            node,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=(
                conditional_from_idx is not None and idx >= conditional_from_idx
            ),
            _depth=_depth + 1,
        )
    return cur_safe, cur_bool


def _is_non_raising_expr(
    expr: Optional[ast.AST],
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    require_bool_safe: bool = False,
    strict_names: bool = True,
    _depth: int = 0,
) -> bool:
    """Returns True if evaluating expr cannot raise an exception caught by a with block."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    while isinstance(expr, ast.NamedExpr):
        if not isinstance(expr.target, ast.Name):
            return False
        expr = expr.value
    if expr is None or isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        if require_bool_safe:
            return expr.id in safe_bool_names
        return (not strict_names) or (expr.id in safe_names)
    if isinstance(expr, ast.Lambda):
        defaults = (
            *expr.args.defaults,
            *(d for d in expr.args.kw_defaults if d is not None),
        )
        return (
            _check_and_advance_non_raising_seq(
                defaults,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            is not None
        )
    if isinstance(expr, (ast.Tuple, ast.List)):
        if any(isinstance(e, ast.Starred) for e in expr.elts):
            return False
        return (
            _check_and_advance_non_raising_seq(
                expr.elts,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            is not None
        )
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.Not):
            return _is_non_raising_bool_expr(
                expr.operand,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
        if isinstance(expr.op, ast.Invert):
            return isinstance(expr.operand, ast.Constant) and isinstance(
                expr.operand.value, (int, bool)
            )
        return (
            isinstance(expr.op, (ast.UAdd, ast.USub))
            and isinstance(expr.operand, ast.Constant)
            and isinstance(expr.operand.value, (int, float, bool))
        )
    if isinstance(expr, ast.BoolOp):
        if not expr.values:
            return False
        advanced = _check_and_advance_non_raising_seq(
            expr.values[:-1],
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
            conditional_from_idx=1,
            _depth=_depth + 1,
        )
        return advanced is not None and _is_non_raising_expr(
            expr.values[-1],
            advanced[0],
            advanced[1],
            require_bool_safe=require_bool_safe,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    if isinstance(expr, ast.IfExp):
        advanced = _check_and_advance_non_raising_seq(
            (expr.test,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
            _depth=_depth + 1,
        )
        return advanced is not None and all(
            _is_non_raising_expr(
                b,
                advanced[0],
                advanced[1],
                require_bool_safe=require_bool_safe,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            for b in (expr.body, expr.orelse)
        )
    if isinstance(expr, ast.Compare):
        if not all(isinstance(op, (ast.Is, ast.IsNot)) for op in expr.ops):
            return False
        return (
            _check_and_advance_non_raising_seq(
                (expr.left, *expr.comparators),
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                conditional_from_idx=2,
                _depth=_depth + 1,
            )
            is not None
        )
    return False


def _is_valid_suppress_or_null_call_signature(
    qual: str,
    call: ast.Call,
) -> bool:
    """Validates positional/keyword arity and names for contextlib.suppress/nullcontext calls."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        return False
    if qual == "contextlib.suppress":
        return not call.keywords
    if qual == "contextlib.nullcontext":
        return len(call.args) + len(call.keywords) <= 1 and all(
            kw.arg == "enter_result" for kw in call.keywords
        )
    return False


def _non_raising_stmt_terminates(stmt: ast.AST, *, _depth: int = 0) -> bool:
    """Returns True if a proven non-raising statement unconditionally exits the block."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.body
        )
    if isinstance(stmt, ast.If) and stmt.orelse:
        return any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.body
        ) and any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.orelse
        )
    return False


def _collect_target_unpack_bool_flags(
    target: ast.AST,
    value: Optional[ast.AST],
    rhs_safe: AbstractSet[str],
    rhs_bool: AbstractSet[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> List[Tuple[str, bool]]:
    """Collects left-to-right (target_name, is_bool_safe) pairs evaluated against RHS state."""
    pairs: List[Tuple[str, bool]] = []
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(target, value)]
    while stack:
        curr_target, curr_value = stack.pop()
        while isinstance(curr_value, ast.NamedExpr):
            curr_value = curr_value.value
        if isinstance(curr_target, ast.Name):
            pairs.append(
                (
                    curr_target.id,
                    _is_non_raising_bool_expr(
                        curr_value,
                        rhs_safe,
                        rhs_bool,
                        strict_names=strict_names,
                        _depth=_depth + 1,
                    ),
                )
            )
        else:
            matched_pairs = _match_sequence_unpack_pairs(
                curr_target, curr_value, allow_starred_target=False
            )
            if matched_pairs is not None:
                stack.extend(reversed(matched_pairs))
            else:
                for t_name in _extract_definite_target_names(curr_target):
                    pairs.append((t_name, False))
    return pairs


def _apply_target_unpack_safe_updates(
    target: ast.AST,
    value: Optional[ast.AST],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> None:
    """Updates cur_safe and cur_bool for an assignment target matched against value."""
    for t_name, is_bool_safe in _collect_target_unpack_bool_flags(
        target,
        value,
        frozenset(cur_safe),
        frozenset(cur_bool),
        strict_names=strict_names,
        _depth=_depth,
    ):
        cur_safe.add(t_name)
        if is_bool_safe:
            cur_bool.add(t_name)
        else:
            cur_bool.discard(t_name)


def _apply_non_raising_stmt_safe_updates(
    b_stmt: ast.stmt,
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> None:
    """Updates cur_safe and cur_bool after a statement proven non-raising by _stmt_cannot_raise_in_suppress."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    if isinstance(b_stmt, ast.Delete):
        for tgt in b_stmt.targets:
            if isinstance(tgt, ast.Name):
                cur_safe.discard(tgt.id)
                cur_bool.discard(tgt.id)
    elif isinstance(b_stmt, (ast.Return, ast.Expr)):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    elif isinstance(b_stmt, ast.Assign):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
        for tgt in b_stmt.targets:
            _apply_target_unpack_safe_updates(
                tgt,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
    elif isinstance(b_stmt, ast.AnnAssign):
        ann_def, ann_cond = _walrus_assignment_in_expr(
            b_stmt.annotation, is_conditional=True, _depth=_depth + 1
        )
        ann_walrus_names = ann_def | ann_cond
        cur_bool.difference_update(ann_walrus_names)
        if b_stmt.value is not None:
            _apply_non_raising_expr_walrus_updates(
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            _apply_target_unpack_safe_updates(
                b_stmt.target,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
        cur_bool.difference_update(ann_walrus_names)
    elif isinstance(b_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for h in iter_nested_scope_header_nodes(b_stmt):
            _apply_non_raising_expr_walrus_updates(
                h,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
        cur_safe.add(b_stmt.name)
        cur_bool.add(b_stmt.name)
    elif isinstance(b_stmt, (ast.With, ast.AsyncWith)):
        for item in b_stmt.items:
            _apply_non_raising_expr_walrus_updates(
                item.context_expr,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            if isinstance(item.optional_vars, ast.Name):
                cur_safe.add(item.optional_vars.id)
                cur_bool.discard(item.optional_vars.id)
        _apply_non_raising_seq_safe_updates(
            b_stmt.body,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    elif isinstance(b_stmt, ast.If):
        _apply_non_raising_if_branch_updates(
            b_stmt,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )


def _apply_non_raising_seq_safe_updates(
    stmts: Sequence[ast.stmt],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> None:
    """Propagates safe bindings sequentially across a non-raising statement list until termination."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    for sub_s in stmts:
        if _non_raising_stmt_terminates(sub_s, _depth=_depth + 1):
            break
        _apply_non_raising_stmt_safe_updates(
            sub_s,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
