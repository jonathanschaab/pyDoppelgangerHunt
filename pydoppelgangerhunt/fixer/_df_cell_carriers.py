"""Enclosing function scope context, column bounds, and closure carrier analysis."""

from __future__ import annotations

import ast
from collections import ChainMap
from types import MappingProxyType
from typing import (
    AbstractSet,
    Dict,
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

from pydoppelgangerhunt.fixer._df_cache import (
    _SessionBucket,
    _ephemeral_unit_ast_context,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _is_ephemeral_unit_ast,
    _mark_ephemeral_unit_ast,
)
from pydoppelgangerhunt.fixer._df_cfg import (  # pylint: disable=unused-import,useless-import-alias
    _extract_store_names_in_node as _extract_store_names_in_node,
    _extract_type_alias_or_pattern_name,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _BUILTIN_NAMES,
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_single_node_local_names,
    _extract_strict_fn_local_names,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import (
    _EnclosingFnCellInfo,
    _FnScopeContext,
    _SpanBounds,
    _block_contains_span,
    _can_bind_in_fn_scope,
    _collect_target_names,
    _compute_scope_ctx_aux_nodes,
    _ensure_subtree_scope_ctx,
    _extract_cached_scope_free_reads,
    _extract_closure_nonlocal_mutations,
    _extract_closure_nonlocal_mutations_and_deletions,
    _extract_closure_nonlocal_mutations_and_deletions_uncached,
    _extract_comprehension_body_walrus_names,
    _extract_comprehension_deferred_closure_info,
    _extract_decorator_carrier_targets,
    _extract_direct_scope_nonlocal_mutations_and_deletions,
    _extract_expr_scope_free_reads,
    _extract_lazy_genexp_walrus_names,
    _extract_mutated_target_names,
    _extract_receiver_root_names,
    _extract_root_target_names,
    _get_fn_scope_context,
    _get_fn_scope_context_uncached,
    _is_lazy_genexp,
    _is_node_attached_to_tree,
    _is_node_inside_span,
    _is_pos_after_span,
    _is_pos_before_span,
    _iter_comprehension_body_exprs_and_targets,
    _make_scope_ctx_cache_key,
    _node_contains_or_encloses_span,
    _resolve_cached_str_set_pair,
    _scope_ctx_allows_node_cache,
    _strictly_encloses_span,
)
from pydoppelgangerhunt.fixer._df_span_index import _extract_direct_nonlocals
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _collect_reachable_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

__all__ = [
    "_CarrierEdge",
    "_EnclosingFnCellInfo",
    "_FnOuterCarrierInfo",
    "_FnScopeContext",
    "_IntraUnitMutator",
    "_NON_CLOSURE_RETURN_BUILTINS",
    "_SpanBounds",
    "_block_contains_span",
    "_can_bind_in_fn_scope",
    "_collect_escaped_and_executed_intra_unit_defs",
    "_collect_escaped_intra_unit_defs",
    "_collect_escaped_intra_unit_names",
    "_collect_unit_nonlocal_mutators",
    "_ensure_subtree_scope_ctx",
    "_extract_cached_scope_free_reads",
    "_extract_carrier_value_closure_and_load_info",
    "_extract_closure_carrier_mutation_targets_and_values",
    "_extract_closure_carrier_targets_and_values",
    "_extract_closure_nonlocal_mutations",
    "_extract_closure_nonlocal_mutations_and_deletions",
    "_extract_closure_nonlocal_mutations_and_deletions_uncached",
    "_extract_comprehension_body_walrus_names",
    "_extract_comprehension_deferred_closure_info",
    "_extract_direct_scope_nonlocal_mutations_and_deletions",
    "_extract_expr_closure_and_load_info",
    "_extract_expr_scope_free_reads",
    "_extract_fn_outer_carrier_info",
    "_extract_fn_outer_carrier_info_uncached",
    "_extract_fn_outer_carrier_targets",
    "_extract_fn_outer_carrier_targets_uncached",
    "_extract_fn_returned_closure_refs",
    "_extract_fn_returned_closure_refs_uncached",
    "_extract_lazy_genexp_walrus_names",
    "_extract_mutated_target_names",
    "_extract_receiver_root_names",
    "_extract_root_target_names",
    "_extract_shallow_defs",
    "_extract_shallow_loads_and_calls",
    "_extract_store_names_in_node",
    "_get_fn_scope_context",
    "_get_fn_scope_context_uncached",
    "_is_closure_factory_scope",
    "_is_closure_factory_scope_uncached",
    "_is_lazy_genexp",
    "_is_node_inside_span",
    "_is_non_closure_builtin_call",
    "_is_non_closure_scalar_expr",
    "_is_pos_after_span",
    "_is_pos_before_span",
    "_is_self_escaping_def_node",
    "_iter_carrier_target_entries",
    "_iter_class_body_methods",
    "_iter_comprehension_body_exprs_and_targets",
    "_make_nested_fn_scope_context",
    "_make_scope_ctx_cache_key",
    "_node_contains_or_encloses_span",
    "_propagate_sibling_closures",
    "_strictly_encloses_span",
]


_DEFAULT_BUILTINS_ALIASES: frozenset[str] = frozenset({"builtins", "__builtins__"})


class _IntraUnitMutator(NamedTuple):
    """Intra-unit closure or lazy generator expression mutating outer variables."""

    name: str
    mutated_vars: Set[str]
    is_escaped_def: bool
    node: Optional[ast.AST] = None


def _is_self_escaping_def_node(
    d_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    *,
    use_cache: bool = True,
) -> bool:
    """Checks whether a function or class definition escapes or mutates on definition."""
    if d_node.decorator_list:
        return True
    if isinstance(d_node, ast.ClassDef):
        if d_node.bases or d_node.keywords:
            return True
        if _extract_direct_scope_nonlocal_mutations_and_deletions(d_node)[0]:
            return True
        if _extract_direct_nonlocals(
            d_node, is_global=True
        ) & _extract_closure_nonlocal_mutations(d_node, use_cache=use_cache):
            return True
    return False


def _collect_unit_nonlocal_mutators(
    candidate_stmts: Sequence[ast.stmt],
    mod_tree: Optional[ast.AST] = None,
    *,
    enclosing_scopes: Sequence[ast.AST] = (),
) -> List[_IntraUnitMutator]:
    """Collects intra-unit nested functions/classes and lazy genexps that mutate outer cells."""
    unit_mod = ast.Module(
        body=[s for s in candidate_stmts if isinstance(s, ast.stmt)],
        type_ignores=[],
    )
    _mark_ephemeral_unit_ast(unit_mod)
    if mod_tree is not None and not _is_ephemeral_unit_ast(mod_tree):
        _get_scope_parent_map(mod_tree, use_cache=True)
    use_node_cache = bool(
        mod_tree is not None
        and candidate_stmts
        and _is_node_attached_to_tree(candidate_stmts[0], mod_tree)
    )
    mod_parent_map = (
        _get_scope_parent_map(mod_tree)
        if use_node_cache and mod_tree is not None
        else None
    )
    fn_enclosing = [
        s for s in enclosing_scopes if not isinstance(s, ast.ClassDef)
    ]
    with _ephemeral_unit_ast_context(unit_mod):
        base_fn_ctx = _get_fn_scope_context_uncached(
            unit_mod,
            mod_tree=mod_tree,
            cache_parent_map=False,
            enclosing_scopes=fn_enclosing,
            scope_owner=None,
            parent_map_override=mod_parent_map,
        )
        top_ctx = (
            _get_fn_scope_context_uncached(
                unit_mod,
                mod_tree=mod_tree,
                cache_parent_map=False,
                enclosing_scopes=enclosing_scopes,
                scope_owner=None,
                parent_map_override=mod_parent_map,
            )
            if len(fn_enclosing) != len(enclosing_scopes)
            else base_fn_ctx
        )
    _esc_names, escaped_def_ids, _exec_ids = (
        _collect_escaped_and_executed_intra_unit_defs(
            candidate_stmts, frozenset(), use_cache=use_node_cache
        )
    )
    mutators: List[_IntraUnitMutator] = []
    stack: List[
        Tuple[ast.AST, frozenset[str], _FnScopeContext, _FnScopeContext, bool]
    ] = [
        (stmt, frozenset(), base_fn_ctx, top_ctx, False)
        for stmt in candidate_stmts
    ]
    while stack:
        curr, shadowed, fn_ctx, active_ctx, in_unescaped_cls = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mutated = (
                _extract_closure_nonlocal_mutations(
                    curr, use_cache=use_node_cache
                )
                - shadowed
            )
            is_self_esc = _is_self_escaping_def_node(
                curr, use_cache=use_node_cache
            )
            if isinstance(curr, ast.ClassDef):
                is_escaped_def = (id(curr) in escaped_def_ids) or is_self_esc
            else:
                is_escaped_def = (
                    (id(curr) in escaped_def_ids)
                    or is_self_esc
                    or (
                        not in_unescaped_cls
                        and _is_closure_factory_scope(
                            curr, active_ctx, use_cache=use_node_cache
                        )
                    )
                )
            if mutated:
                mutators.append(
                    _IntraUnitMutator(
                        curr.name, mutated, is_escaped_def, node=curr
                    )
                )
            if isinstance(curr, ast.ClassDef):
                # PEP 227: class body bindings and class-level global declarations do not
                # create a lexical barrier for nonlocal declarations in nested methods.
                body_shadowed = shadowed
                body_fn_ctx = fn_ctx
                body_active_ctx = _make_nested_fn_scope_context(
                    curr,
                    frozenset(_extract_single_node_local_names(curr)),
                    fn_ctx,
                    use_cache=use_node_cache,
                )
                body_unesc_cls = not is_escaped_def
            else:
                fn_own = frozenset(_extract_strict_fn_local_names(curr))
                body_shadowed = shadowed | fn_own
                body_fn_ctx = _make_nested_fn_scope_context(
                    curr, fn_own, fn_ctx, use_cache=use_node_cache
                )
                body_active_ctx = body_fn_ctx
                body_unesc_cls = False
            stack.extend(
                (ch, body_shadowed, body_fn_ctx, body_active_ctx, body_unesc_cls)
                for ch in curr.body
            )
            stack.extend(
                (hdr, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, active_ctx):
                w_names = (
                    _extract_comprehension_body_walrus_names(curr) - shadowed
                )
                if w_names:
                    mutators.append(
                        _IntraUnitMutator("", w_names, True, node=curr)
                    )
                if curr.generators:
                    stack.append(
                        (
                            curr.generators[0].iter,
                            shadowed,
                            fn_ctx,
                            active_ctx,
                            in_unescaped_cls,
                        )
                    )
                continue
            stack.extend(
                (ch, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for ch in ast.iter_child_nodes(curr)
            )
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(
                (hdr, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
                for hdr in iter_nested_scope_header_nodes(curr)
            )
            continue
        stack.extend(
            (ch, shadowed, fn_ctx, active_ctx, in_unescaped_cls)
            for ch in ast.iter_child_nodes(curr)
        )
    return mutators


# INVARIANT (CONVERGENCE BOUNDARY — CLOSED SCALAR BUILTIN TABLE):
# Built-in functions used in _is_non_closure_scalar_expr (for closure carrier tracking, not
# side-effect purity) that return primitive scalars, booleans, strings, byte sequences,
# ranges, file streams, or None under standard Python conventions. CPython enforces that
# conversion dunders (__int__, __index__, __len__, __bool__, __str__, __repr__, __format__,
# __float__, __complex__, __hash__) return instances of the corresponding base type rather
# than arbitrary function/lambda closures. Note that constructor-style built-ins such as
# int(x), str(x), float(x), complex(x), bytes(x), and bytearray(x) can still return an
# instance of a user subclass of that base type which could theoretically define __call__;
# treating them as non-closure scalar values is a closed, frozen static boundary — do NOT
# expand this table incrementally for third-party APIs or attempt inter-procedural
# points-to resolution across dynamic dispatch. Built-ins that delegate to unconstrained
# user dunders (abs -> __abs__, round -> __round__, divmod -> __divmod__, pow -> __pow__,
# sum -> __add__) remain excluded.
_NON_CLOSURE_RETURN_BUILTINS: frozenset[str] = frozenset(
    {
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
        "print",
        "range",
        "repr",
        "str",
    }
)


def _is_non_closure_builtin_call(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node is a Call to an unshadowed scalar/non-closure built-in.

    Requires ``scope_ctx`` to be provided so ``scope_ctx.has_star_import`` and
    ``scope_ctx.shadowed_names`` can verify the built-in is not shadowed. When
    ``scope_ctx`` is None, returns ``False`` (conservative/fail-closed).
    """
    return (
        scope_ctx is not None
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _NON_CLOSURE_RETURN_BUILTINS
        and not scope_ctx.has_star_import
        and "*" not in scope_ctx.shadowed_names
        and node.func.id not in scope_ctx.shadowed_names
    )


def _is_non_closure_scalar_expr(
    node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> bool:
    """Returns True if node evaluates to a scalar/primitive non-closure value.

    Note:
        Only literals (``ast.Constant``, ``ast.JoinedStr``), boolean negation
        (``not x``), calls to unshadowed ``_NON_CLOSURE_RETURN_BUILTINS`` (excluding
        callable subclasses of ``int``/``str``/``float``/``bytes``),
        and unary/binary/comparison operations whose operands are all
        non-closure scalars are classified as scalar. Binary, unary, or comparison
        operations involving arbitrary variables or custom objects are NOT assumed
        to be scalar because custom dunders (``__add__``, ``__abs__``, ``__eq__``,
        etc.) can return closures. Inline mutations inside scalar sub-expressions
        are still extracted by ``_extract_carrier_value_closure_and_load_info``.
    """
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Constant, ast.JoinedStr)):
            continue
        if _is_non_closure_builtin_call(curr, scope_ctx):
            continue
        if isinstance(curr, ast.UnaryOp):
            if not isinstance(curr.op, ast.Not):
                stack.append(curr.operand)
            continue
        if isinstance(curr, ast.BinOp):
            stack.append(curr.left)
            stack.append(curr.right)
            continue
        if isinstance(curr, ast.Compare):
            stack.append(curr.left)
            stack.extend(curr.comparators)
            continue
        return False
    return True


def _extract_expr_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
    _depth: int = 0,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (inline_closure_reads, inline_closure_muts, loaded_names) from expressions.

    Note:
        Direct ``ast.Name`` call targets (``curr.func`` when ``isinstance(curr.func, ast.Name)``
        and ``curr`` is not an unshadowed scalar built-in call) are routed to
        ``called_names_out`` when provided; when ``called_names_out`` is None, direct
        callee names are included in ``loaded_names`` so they are never silently dropped.
        Attribute calls (``factory.make()``, ``self.factory.make()``,
        ``get_factory().make()``, ``Factory.make(factory)``) traverse the receiver
        expression, so the receiver's root names land in ``loaded_names`` (or its direct
        callees in ``called_names_out``) and the call result is aliased to the receiver
        object. This over-approximates every method of a closure-carrying instance or
        class: a method can only close over enclosing function cells when it is lexically
        defined inside a local class or factory that is itself recorded as a closure
        carrier, so resolving the individual method name cannot add escaping cells
        (closures handed to external receivers are already recorded as escaping when
        passed as call arguments).
    """
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
            ),
        ):
            continue
        if isinstance(curr, ast.Lambda):
            inline_reads.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
            stack.extend(iter_nested_scope_header_nodes(curr))
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
                if _depth >= _MAX_AST_RECURSION_DEPTH:
                    stack.extend(comp_exprs)
                else:
                    sub_calls: Set[str] = set()
                    _, _, sub_loads = _extract_expr_closure_and_load_info(
                        comp_exprs,
                        scope_ctx,
                        called_names_out=sub_calls,
                        _depth=_depth + 1,
                    )
                    loaded_names.update(sub_loads - comp_targets)
                    if called_names_out is not None:
                        called_names_out.update(sub_calls - comp_targets)
                    else:
                        loaded_names.update(sub_calls - comp_targets)
            continue
        if isinstance(curr, ast.Call):
            if _is_non_closure_builtin_call(curr, scope_ctx):
                stack.extend(curr.args)
                stack.extend(
                    kw.value for kw in curr.keywords if kw.value is not None
                )
                continue
            if isinstance(curr.func, ast.Attribute):
                stack.append(curr.func.value)
            elif isinstance(curr.func, ast.Name):
                if called_names_out is not None:
                    called_names_out.add(curr.func.id)
                else:
                    loaded_names.add(curr.func.id)
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


def _extract_carrier_value_closure_and_load_info(
    nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    called_names_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts closure reads, mutations, and loads for carrier value expressions."""
    carrier_nodes: List[ast.AST] = []
    scalar_nodes: List[ast.AST] = []
    for node in nodes:
        if _is_non_closure_scalar_expr(node, scope_ctx):
            scalar_nodes.append(node)
        else:
            carrier_nodes.append(node)
    c_reads, c_muts, c_loads = (
        _extract_expr_closure_and_load_info(
            carrier_nodes, scope_ctx, called_names_out=called_names_out
        )
        if carrier_nodes
        else (set(), set(), set())
    )
    if scalar_nodes:
        _, s_muts, _ = _extract_expr_closure_and_load_info(
            scalar_nodes,
            scope_ctx,
            called_names_out=called_names_out,
        )
        c_muts.update(s_muts)
    return c_reads, c_muts, c_loads


def _extract_assignment_targets_and_value(
    node: ast.AST,
    mode: str,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (target_names, (value,)) from assignment nodes using the specified target mode."""
    if isinstance(node, ast.Assign) and node.value is not None:
        targets: Set[str] = set()
        for tgt in node.targets:
            targets.update(_collect_target_names(tgt, mode=mode))
        return (targets, (node.value,)) if targets else None
    if (
        isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        and node.value is not None
    ):
        targets = _collect_target_names(node.target, mode=mode)
        return (targets, (node.value,)) if targets else None
    if isinstance(node, (ast.For, ast.AsyncFor)):
        targets = _collect_target_names(node.target, mode=mode)
        return (targets, (node.iter,)) if targets else None
    return None


def _extract_closure_carrier_targets_and_values(
    node: ast.AST,
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (rebound_target_names, value_nodes) from variable-binding assignments."""
    return _extract_assignment_targets_and_value(node, mode="rebound")


# INVARIANT (STATIC TRUST BOUNDARY — CONVERGENCE):
# Direct carrier-mutation detection recognizes syntactic ``setattr(...)`` and
# ``builtins.setattr(...)`` / ``__builtins__.setattr(...)`` calls in O(1) per
# call node. Do NOT re-introduce multi-scope alias chasing for obfuscated
# reflection (``import builtins as b``, ``sa = setattr``, ``getattr(b, 'setattr')``,
# ``vars(builtins)['setattr']``, ``__import__('builtins')``); indirect runtime
# aliasing is unbounded in dynamic Python and mirrors the static trust boundary
# documented in ``_df_exit_protocol.py``.
def _is_setattr_call(node: ast.Call) -> bool:
    """Returns True when node is a direct setattr(...) or builtins.setattr(...) call."""
    has_target_and_value = len(node.args) >= 3 or (
        len(node.args) == 2 and isinstance(node.args[1], ast.Starred)
    )
    if not has_target_and_value:
        return False
    func: ast.AST = node.func
    while isinstance(func, ast.NamedExpr):
        func = func.value
    if isinstance(func, ast.Name):
        return func.id == "setattr"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "setattr"
        and isinstance(func.value, ast.Name)
        and func.value.id in _DEFAULT_BUILTINS_ALIASES
    )


def _extract_closure_carrier_mutation_targets_and_values(
    node: ast.AST,
    *,
    scope_ctx: Optional[_FnScopeContext] = None,  # pylint: disable=unused-argument
) -> Optional[Tuple[Set[str], Sequence[ast.AST]]]:
    """Extracts (mutated_object_root_names, value_nodes) from attribute/subscript/method/setattr mutations."""
    assign_res = _extract_assignment_targets_and_value(node, mode="mutated")
    if assign_res is not None:
        return assign_res
    if (
        isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return ({node.target.id}, (node.value,))
    if isinstance(node, ast.Call):
        if _is_setattr_call(node):
            arg_vals = node.args[2:] if len(node.args) >= 3 else node.args[1:]
            receiver_expr: ast.AST = node.args[0]
        elif isinstance(node.func, ast.Attribute):
            arg_vals = node.args
            receiver_expr = node.func.value
        else:
            return None
        val_nodes: Sequence[ast.AST] = [
            *arg_vals,
            *(kw.value for kw in node.keywords if kw.value is not None),
        ]
        if val_nodes:
            targets = _extract_receiver_root_names(receiver_expr)
            return (targets, val_nodes) if targets else None
    return None


def _iter_carrier_target_entries(
    node: ast.AST,
    *,
    scope_ctx: Optional[_FnScopeContext] = None,
) -> List[Tuple[bool, Set[str], Sequence[ast.AST]]]:
    """Returns (is_rebound, target_names, value_nodes) entries for a carrier AST node."""
    entries: List[Tuple[bool, Set[str], Sequence[ast.AST]]] = []
    rebound = _extract_closure_carrier_targets_and_values(node)
    if rebound is not None:
        entries.append((True, rebound[0], rebound[1]))
    mutated = _extract_closure_carrier_mutation_targets_and_values(
        node, scope_ctx=scope_ctx
    )
    if mutated is not None:
        entries.append((False, mutated[0], mutated[1]))
    return entries


def _extract_shallow_loads_and_calls(
    stmts: Sequence[ast.AST],
    *,
    deletions_out: Optional[Set[str]] = None,
    pep227_loads_out: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str]]:
    """Extracts (non_call_loaded_names, direct_called_names) stopping at nested def/class bodies."""
    loaded: Set[str] = set()
    called: Set[str] = set()
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            if isinstance(curr, ast.Lambda):
                lam_reads = _extract_expr_scope_free_reads(
                    curr, include_header=False
                )
                loaded.update(lam_reads)
                if pep227_loads_out is not None:
                    pep227_loads_out.update(lam_reads)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if pep227_loads_out is not None and isinstance(
            curr, _COMPREHENSION_TYPES
        ):
            pep227_loads_out.update(
                _extract_expr_scope_free_reads(curr, include_header=False)
            )
        if deletions_out is not None and isinstance(curr, ast.Delete):
            for d_tgt in curr.targets:
                if isinstance(d_tgt, ast.Name):
                    deletions_out.add(d_tgt.id)
                elif isinstance(d_tgt, (ast.Tuple, ast.List)):
                    deletions_out.update(
                        elt.id
                        for elt in ast.walk(d_tgt)
                        if isinstance(elt, ast.Name)
                    )
        if isinstance(curr, ast.Call):
            if isinstance(curr.func, ast.Name):
                called.add(curr.func.id)
            else:
                stack.append(curr.func)
            stack.extend(curr.args)
            stack.extend(curr.keywords)
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Load):
            loaded.add(curr.id)
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return loaded, called


_NamedScopeNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


def _extract_shallow_defs(
    stmts: Sequence[ast.AST],
) -> Dict[str, List[_NamedScopeNode]]:
    """Extracts direct FunctionDef, AsyncFunctionDef, and ClassDef nodes in stmts."""
    defs: Dict[str, List[_NamedScopeNode]] = {}
    stack: List[ast.AST] = list(stmts)
    while stack:
        curr = stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            defs.setdefault(curr.name, []).append(curr)
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Lambda):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return defs


def _extract_class_body_unconditional_def_names(
    cls_node: ast.ClassDef,
) -> frozenset[str]:
    """Returns class-level def/class names unconditionally bound for all class-body uses."""
    currently_bound: Set[str] = set()
    used_when_unbound: Set[str] = set()
    for stmt in cls_node.body:
        shallow_dels: Set[str] = set()
        pep227_loads: Set[str] = set()
        s_loads, s_calls = _extract_shallow_loads_and_calls(
            [stmt],
            deletions_out=shallow_dels,
            pep227_loads_out=pep227_loads,
        )
        if shallow_dels:
            currently_bound.difference_update(shallow_dels)
        for ref in s_loads | s_calls:
            if ref not in currently_bound:
                used_when_unbound.add(ref)
        used_when_unbound.update(pep227_loads)
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            currently_bound.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    currently_bound.add(tgt.id)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and stmt.value is not None
            and isinstance(stmt.target, ast.Name)
        ):
            currently_bound.add(stmt.target.id)
    return frozenset(currently_bound - used_when_unbound)


def _propagate_top_cell_defs(
    escaped_top_names: Set[str],
    executed_top_names: Set[str],
    scope_cell_defs: Mapping[Tuple[int, str], Sequence[_NamedScopeNode]],
    escaped_def_ids: Set[int],
    executed_def_ids: Set[int],
) -> bool:
    """Propagates escaped_top_names and executed_top_names into root scope_cell_defs."""
    changed = False
    for name in executed_top_names | escaped_top_names:
        mark_esc = name in escaped_top_names
        for d_node in scope_cell_defs.get((0, name), ()):
            did = id(d_node)
            if did not in executed_def_ids:
                executed_def_ids.add(did)
                changed = True
            if (
                mark_esc or isinstance(d_node, ast.ClassDef)
            ) and did not in escaped_def_ids:
                escaped_def_ids.add(did)
                changed = True
    return changed


def _activate_fn_global_and_nonlocal_defs(
    nid: int,
    global_defs_by_fn: Mapping[int, Sequence[Tuple[str, _NamedScopeNode]]],
    nonlocal_defs_by_fn: Mapping[
        int, Sequence[Tuple[int, str, _NamedScopeNode]]
    ],
    scope_cell_defs: Dict[Tuple[int, str], List[_NamedScopeNode]],
    escaped_top_names: Set[str],
    escaped_def_ids: Set[int],
    executed_def_ids: Set[int],
) -> bool:
    """Activates global and nonlocal nested definitions inside an executed function nid."""
    changed = False
    for g_name, g_node in global_defs_by_fn.get(nid, ()):
        gid = id(g_node)
        cell_list = scope_cell_defs.setdefault((0, g_name), [])
        if g_node not in cell_list:
            cell_list.append(g_node)
            changed = True
        if gid not in escaped_def_ids:
            escaped_def_ids.add(gid)
            changed = True
        if gid not in executed_def_ids:
            executed_def_ids.add(gid)
            changed = True
        if g_name not in escaped_top_names:
            escaped_top_names.add(g_name)
            changed = True
    for target_origin, nl_name, nl_node in nonlocal_defs_by_fn.get(nid, ()):
        nl_id = id(nl_node)
        eff_origin = 0 if target_origin < 0 else target_origin
        cell_list = scope_cell_defs.setdefault((eff_origin, nl_name), [])
        if nl_node not in cell_list:
            cell_list.append(nl_node)
            changed = True
        if target_origin < 0 or target_origin in escaped_def_ids:
            if nl_id not in escaped_def_ids:
                escaped_def_ids.add(nl_id)
                changed = True
            if nl_id not in executed_def_ids:
                executed_def_ids.add(nl_id)
                changed = True
            if target_origin < 0 and nl_name not in escaped_top_names:
                escaped_top_names.add(nl_name)
                changed = True
    return changed


def _collect_escaped_and_executed_intra_unit_defs(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
    *,
    use_cache: bool = False,
) -> Tuple[Set[str], Set[int], Set[int]]:
    """Computes (escaped_names, escaped_def_node_ids, executed_def_node_ids)."""
    root_defs = _extract_shallow_defs(candidate_stmts)
    scope_cell_defs: Dict[Tuple[int, str], List[_NamedScopeNode]] = {
        (0, name): list(nodes) for name, nodes in root_defs.items()
    }
    unit_mod = ast.Module(body=list(candidate_stmts), type_ignores=[])
    _mark_ephemeral_unit_ast(unit_mod)
    with _ephemeral_unit_ast_context(unit_mod):
        top_bound_names = (
            _extract_single_node_local_names(unit_mod)
            - _extract_direct_nonlocals(unit_mod, is_global=False)
            - _extract_direct_nonlocals(unit_mod, is_global=True)
        ) | set(root_defs.keys())
    root_origins: Dict[str, int] = {name: 0 for name in top_bound_names}
    all_def_nodes: List[_NamedScopeNode] = []
    defs_by_name: Dict[str, List[_NamedScopeNode]] = {}
    direct_children: Dict[int, List[_NamedScopeNode]] = {}
    body_origins: Dict[int, Dict[str, int]] = {}
    class_local_defs: Dict[int, Dict[str, Tuple[_NamedScopeNode, ...]]] = {}
    class_uncond_defs: Dict[int, frozenset[str]] = {}
    global_defs_by_fn: Dict[int, List[Tuple[str, _NamedScopeNode]]] = {}
    nonlocal_defs_by_fn: Dict[int, List[Tuple[int, str, _NamedScopeNode]]] = {}
    shadowed_map: Dict[int, frozenset[str]] = {}

    scope_stack: List[
        Tuple[
            _NamedScopeNode,
            Dict[str, int],
            frozenset[str],
        ]
    ] = [
        (d_node, root_origins, frozenset())
        for nodes in root_defs.values()
        for d_node in nodes
    ]
    while scope_stack:
        d_node, parent_origins, shadowed_from_root = scope_stack.pop()
        nid = id(d_node)
        all_def_nodes.append(d_node)
        d_name = d_node.name
        if d_name:
            defs_by_name.setdefault(d_name, []).append(d_node)
        c_defs = _extract_shallow_defs(d_node.body)
        c_list = [n for nodes in c_defs.values() for n in nodes]
        direct_children[nid] = c_list
        if isinstance(d_node, ast.ClassDef):
            body_origins[nid] = dict(parent_origins)
            class_local_defs[nid] = {
                k: tuple(v) for k, v in c_defs.items()
            }
            uncond_cls = _extract_class_body_unconditional_def_names(d_node)
            class_uncond_defs[nid] = uncond_cls
            shadowed_map[nid] = shadowed_from_root | uncond_cls
            for c_node in c_list:
                scope_stack.append(
                    (c_node, parent_origins, shadowed_from_root)
                )
        else:
            own_locals = _extract_strict_fn_local_names(d_node)
            own_globals = _extract_direct_nonlocals(d_node, is_global=True)
            own_nonlocals = _extract_direct_nonlocals(d_node, is_global=False)
            fn_shadowed = shadowed_from_root | frozenset(
                (own_locals | own_globals) - own_nonlocals
            )
            cur_origins: Dict[str, int] = {
                k: v
                for k, v in parent_origins.items()
                if k not in own_locals and k not in own_globals
            }
            for loc_name in (own_locals - set(c_defs.keys())) - own_nonlocals - own_globals:
                cur_origins[loc_name] = nid
            for g_name in own_globals:
                cur_origins[g_name] = 0
            for c_name, c_nodes in c_defs.items():
                if c_name in own_globals:
                    cur_origins[c_name] = nid
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
                    for c_node in c_nodes:
                        global_defs_by_fn.setdefault(nid, []).append(
                            (c_name, c_node)
                        )
                elif c_name in own_nonlocals:
                    target_origin = parent_origins.get(c_name, -1)
                    cur_origins[c_name] = (
                        0 if target_origin < 0 else target_origin
                    )
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
                    for c_node in c_nodes:
                        nonlocal_defs_by_fn.setdefault(nid, []).append(
                            (target_origin, c_name, c_node)
                        )
                else:
                    cur_origins[c_name] = nid
                    scope_cell_defs.setdefault((nid, c_name), []).extend(
                        c_nodes
                    )
            body_origins[nid] = cur_origins
            shadowed_map[nid] = fn_shadowed
            for c_node in c_list:
                scope_stack.append((c_node, cur_origins, fn_shadowed))

    top_loads, top_calls = _extract_shallow_loads_and_calls(candidate_stmts)
    escaped_top_names: Set[str] = set(initial_escaped) | top_loads
    executed_top_names: Set[str] = set(top_calls) | escaped_top_names
    escaped_def_ids: Set[int] = set()
    executed_def_ids: Set[int] = set()

    _propagate_top_cell_defs(
        escaped_top_names,
        executed_top_names,
        scope_cell_defs,
        escaped_def_ids,
        executed_def_ids,
    )
    for nodes in root_defs.values():
        for d_node in nodes:
            if _is_self_escaping_def_node(d_node, use_cache=use_cache):
                escaped_def_ids.add(id(d_node))
                executed_def_ids.add(id(d_node))
            elif isinstance(d_node, ast.ClassDef):
                executed_def_ids.add(id(d_node))

    body_info: Dict[int, Tuple[Set[str], Set[str]]] = {}
    changed = True
    while changed:
        changed = _propagate_top_cell_defs(
            escaped_top_names,
            executed_top_names,
            scope_cell_defs,
            escaped_def_ids,
            executed_def_ids,
        )
        for d_node in all_def_nodes:
            nid = id(d_node)
            is_esc = nid in escaped_def_ids
            is_exec = is_esc or (nid in executed_def_ids)
            if not is_exec:
                continue
            changed |= _activate_fn_global_and_nonlocal_defs(
                nid,
                global_defs_by_fn,
                nonlocal_defs_by_fn,
                scope_cell_defs,
                escaped_top_names,
                escaped_def_ids,
                executed_def_ids,
            )
            for c_node in direct_children.get(nid, ()):
                cid = id(c_node)
                if (
                    isinstance(d_node, ast.ClassDef) and is_esc
                ) or _is_self_escaping_def_node(c_node, use_cache=use_cache):
                    if cid not in escaped_def_ids:
                        escaped_def_ids.add(cid)
                        executed_def_ids.add(cid)
                        changed = True
                elif isinstance(c_node, ast.ClassDef):
                    if cid not in executed_def_ids:
                        executed_def_ids.add(cid)
                        changed = True
            if nid not in body_info:
                body_info[nid] = _extract_shallow_loads_and_calls(d_node.body)
            b_loads, b_calls = body_info[nid]
            origins_here = body_origins[nid]
            cls_defs_here = class_local_defs.get(nid, {})
            uncond_cls_here = class_uncond_defs.get(nid, frozenset())
            shadowed_here = shadowed_map[nid]
            for ref_name in b_loads | b_calls:
                if (
                    (is_esc or ref_name in b_loads)
                    and ref_name not in shadowed_here
                    and ref_name not in escaped_top_names
                ):
                    escaped_top_names.add(ref_name)
                    changed = True
                if ref_name in cls_defs_here and ref_name in uncond_cls_here:
                    target_nodes: Sequence[_NamedScopeNode] = cls_defs_here[
                        ref_name
                    ]
                else:
                    origin_scope = origins_here.get(
                        ref_name,
                        0 if ref_name not in shadowed_here else -1,
                    )
                    target_nodes = [
                        *cls_defs_here.get(ref_name, ()),
                        *scope_cell_defs.get((origin_scope, ref_name), ()),
                        *scope_cell_defs.get((nid, ref_name), ()),
                    ]
                for target_node in target_nodes:
                    tid = id(target_node)
                    if (
                        is_esc
                        or ref_name in b_loads
                        or isinstance(target_node, ast.ClassDef)
                    ) and tid not in escaped_def_ids:
                        escaped_def_ids.add(tid)
                        changed = True
                    if tid not in executed_def_ids:
                        executed_def_ids.add(tid)
                        changed = True

    unambiguous_escaped_def_names = {
        name
        for name, nodes in defs_by_name.items()
        if name and all(id(n) in escaped_def_ids for n in nodes)
    }
    return (
        escaped_top_names | unambiguous_escaped_def_names,
        escaped_def_ids,
        executed_def_ids,
    )


def _collect_escaped_intra_unit_defs(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Tuple[Set[str], Set[int]]:
    """Computes (escaped_names, escaped_def_node_ids) with lexical-scope definition tracking."""
    esc_names, esc_ids, _ = _collect_escaped_and_executed_intra_unit_defs(
        candidate_stmts, initial_escaped
    )
    return esc_names, esc_ids


def _collect_escaped_intra_unit_names(
    candidate_stmts: Sequence[ast.stmt],
    initial_escaped: AbstractSet[str],
) -> Set[str]:
    """Computes function/variable names that escape candidate_stmts via loads or escaped closures."""
    return _collect_escaped_intra_unit_defs(candidate_stmts, initial_escaped)[0]


def _propagate_sibling_closures(
    closures: Sequence[Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]],
    post_reads: Optional[Set[str]] = None,
    alias_edges: Sequence[Tuple[str, AbstractSet[str]]] = (),
    mutation_edges: Sequence[Tuple[str, AbstractSet[str]]] = (),
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
    effective_edges: List[Tuple[str, AbstractSet[str]]] = list(alias_edges)
    if mutation_edges:
        rev_rebind: Dict[str, Set[str]] = {}
        for tgt_name, src_names in alias_edges:
            rev_rebind.setdefault(tgt_name, set()).update(src_names)
        for mut_tgt, mut_srcs in mutation_edges:
            for origin_name in _collect_reachable_names(mut_tgt, rev_rebind):
                effective_edges.append((origin_name, mut_srcs))
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
        for tgt_name, src_names in effective_edges:
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


def _make_nested_fn_scope_context(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    own_locals: frozenset[str],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> _FnScopeContext:
    """Builds a _FnScopeContext for analyzing expressions inside a nested function."""
    prev_stored = (
        scope_ctx._stored_parent_map  # pylint: disable=protected-access
        if scope_ctx is not None
        else None
    )
    parent_map: Optional[Mapping[ast.AST, ast.AST]]
    has_nonempty_pmap: bool
    if prev_stored is not None and (
        not fn_node.body or fn_node.body[0] in prev_stored
    ):
        parent_map = prev_stored
        has_nonempty_pmap = bool(prev_stored)
    else:
        raw_fn_map = _get_scope_parent_map(fn_node, use_cache=use_cache)
        has_nonempty_pmap = bool(raw_fn_map)
        if prev_stored is not None:
            parent_map = MappingProxyType(
                ChainMap(
                    cast(Dict[ast.AST, ast.AST], raw_fn_map),
                    cast(Dict[ast.AST, ast.AST], prev_stored),
                )
            )
            has_nonempty_pmap = bool(parent_map)
        else:
            parent_map = MappingProxyType(raw_fn_map)
    include_prev_owner = (
        scope_ctx is not None
        and isinstance(scope_ctx.scope_owner, ast.AST)
        and (
            not isinstance(scope_ctx.mod_tree, ast.AST)
            or _is_node_attached_to_tree(
                scope_ctx.scope_owner, scope_ctx.mod_tree
            )
        )
    )
    enc_scopes: Tuple[ast.AST, ...] = (
        (*scope_ctx.enclosing_scopes, cast(ast.AST, scope_ctx.scope_owner))
        if scope_ctx is not None and include_prev_owner
        else (scope_ctx.enclosing_scopes if scope_ctx is not None else ())
    )
    eff_mod_tree = scope_ctx.mod_tree if scope_ctx is not None else None
    return _FnScopeContext(
        fn_locals=(
            (scope_ctx.fn_locals if scope_ctx is not None else frozenset())
            | own_locals
        ),
        shadowed_names=(
            (scope_ctx.shadowed_names if scope_ctx is not None else frozenset())
            | own_locals
        ),
        parent_map=parent_map,
        has_star_import=(
            scope_ctx.has_star_import if scope_ctx is not None else False
        ),
        scope_owner=fn_node,
        mod_tree=eff_mod_tree,
        enclosing_scopes=enc_scopes,
        precomputed_aux_nodes=_compute_scope_ctx_aux_nodes(
            fn_node, eff_mod_tree, enc_scopes, has_nonempty_pmap
        ),
        has_nonempty_parent_map=has_nonempty_pmap,
    )


def _propagate_local_closure_syms(
    local_closure_syms: Set[str],
    local_edges: Sequence[Tuple[str, Set[str]]],
) -> Set[str]:
    """Propagates local closure symbol reachability across local assignment/call edges."""
    if local_closure_syms and local_edges:
        changed = True
        while changed:
            changed = False
            for t_name, refs in local_edges:
                if t_name not in local_closure_syms and (
                    refs & local_closure_syms
                ):
                    local_closure_syms.add(t_name)
                    changed = True
    return local_closure_syms


# INVARIANT (CONVERGENCE BOUNDARY — BOUNDED CLOSURE CARRIER PROPAGATION):
# Closure-factory and outer-carrier analysis below is bounded strictly to
# intra-procedural lexical scopes and single-level nested factory return/carrier
# summaries (guarded by ``_MAX_AST_RECURSION_DEPTH``). Do NOT expand this into
# inter-procedural points-to graphs, heap shape analysis, or unbounded container
# alias chasing; ambiguous or dynamic carrier escapes must fail closed.
def _is_closure_factory_scope_uncached(
    scope_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
    _depth: int = 0,
) -> bool:
    """Computes whether a nested function or class can return a closure or carrier."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return True
    if isinstance(scope_node, (ast.ClassDef, ast.AsyncFunctionDef)):
        unshadowed_builtins = _BUILTIN_NAMES - (
            set(scope_ctx.shadowed_names) if scope_ctx is not None else set()
        )
        has_free_or_nl = bool(
            _extract_cached_scope_free_reads(scope_node, use_cache=use_cache)
            - unshadowed_builtins
        ) or bool(
            _extract_closure_nonlocal_mutations_and_deletions(
                scope_node, use_cache=use_cache
            )[0]
        )
        if isinstance(scope_node, ast.ClassDef):
            has_deferred_member = any(
                isinstance(
                    sub,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Lambda,
                        ast.GeneratorExp,
                    ),
                )
                and sub is not scope_node
                for sub in ast.walk(scope_node)
            )
            if not has_deferred_member:
                return False
            if has_free_or_nl:
                return True
            return any(
                isinstance(sub, (ast.Lambda, ast.GeneratorExp))
                or (
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _is_closure_factory_scope_uncached(
                        sub, scope_ctx, use_cache=use_cache, _depth=_depth + 1
                    )
                )
                for sub in ast.walk(scope_node)
                if sub is not scope_node
            )
        if has_free_or_nl or any(
            _extract_fn_returned_closure_refs(
                scope_node, scope_ctx, use_cache=use_cache
            )
        ):
            return True
    own_locals = frozenset(_extract_strict_fn_local_names(scope_node))
    local_ctx = _make_nested_fn_scope_context(
        scope_node, own_locals, scope_ctx, use_cache=use_cache
    )
    has_nested_closure = False
    local_closure_syms: Set[str] = set()
    local_edges: List[Tuple[str, Set[str]]] = []
    ret_exprs: List[ast.AST] = []
    stack: List[ast.AST] = list(scope_node.body)
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            has_nested_closure = True
            local_closure_syms.add(curr.name)
            continue
        if isinstance(curr, ast.Lambda):
            has_nested_closure = True
            continue
        if isinstance(curr, ast.GeneratorExp):
            if _is_lazy_genexp(curr, local_ctx):
                has_nested_closure = True
                if curr.generators:
                    stack.append(curr.generators[0].iter)
                continue
        for _is_reb, c_targets, c_values in _iter_carrier_target_entries(
            curr, scope_ctx=local_ctx
        ):
            local_tgts = c_targets & own_locals
            if local_tgts:
                c_calls: Set[str] = set()
                c_reads, c_muts, c_loads = (
                    _extract_carrier_value_closure_and_load_info(
                        c_values, local_ctx, called_names_out=c_calls
                    )
                )
                if c_reads or c_muts:
                    local_closure_syms.update(local_tgts)
                local_refs = (c_loads | c_calls) & own_locals
                if local_refs:
                    for t_name in local_tgts:
                        local_edges.append((t_name, local_refs))
        if isinstance(curr, ast.Return) and curr.value is not None:
            ret_exprs.append(curr.value)
        stack.extend(ast.iter_child_nodes(curr))
    if not has_nested_closure or not ret_exprs:
        return False
    _propagate_local_closure_syms(local_closure_syms, local_edges)
    ret_calls: Set[str] = set()
    r_reads, r_muts, r_loads = _extract_carrier_value_closure_and_load_info(
        ret_exprs, local_ctx, called_names_out=ret_calls
    )
    return bool(
        r_reads
        or r_muts
        or ((r_loads | ret_calls) & local_closure_syms)
    )


def _is_closure_factory_scope(
    scope_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> bool:
    """Checks whether a nested function or class can return a closure or generator/coroutine."""
    if not isinstance(
        scope_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return False
    eff_cache = use_cache and _scope_ctx_allows_node_cache(
        scope_node, scope_ctx
    )
    if not eff_cache:
        return _is_closure_factory_scope_uncached(
            scope_node, scope_ctx, use_cache=False
        )
    cache_key = _make_scope_ctx_cache_key(
        scope_node, "is_closure_factory", scope_ctx
    )
    return bool(
        _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            cache_key,
            lambda: _is_closure_factory_scope_uncached(
                scope_node, scope_ctx, use_cache=True
            ),
        )
    )


def _extract_fn_returned_closure_refs_uncached(
    fn_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (returned_loaded_names, returned_called_names) for fn_node without caching."""
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset(), frozenset()
    own_locals = _extract_strict_fn_local_names(fn_node)
    local_ctx = _make_nested_fn_scope_context(
        fn_node, frozenset(own_locals), scope_ctx, use_cache=use_cache
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
        return frozenset(), frozenset()
    ret_calls: Set[str] = set()
    _, _, ret_loads = _extract_carrier_value_closure_and_load_info(
        ret_exprs, local_ctx, called_names_out=ret_calls
    )
    return frozenset(ret_loads - own_locals), frozenset(ret_calls - own_locals)


def _extract_fn_returned_closure_refs(
    fn_node: ast.AST,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Extracts non-local (returned_loaded_names, returned_called_names) for a nested function."""
    eff_cache = use_cache and _scope_ctx_allows_node_cache(fn_node, scope_ctx)
    return _resolve_cached_str_set_pair(
        fn_node,
        (ast.FunctionDef, ast.AsyncFunctionDef),
        _make_scope_ctx_cache_key(fn_node, "ret_closure_refs", scope_ctx),
        lambda c: _extract_fn_returned_closure_refs_uncached(
            fn_node, scope_ctx, use_cache=c
        ),
        use_cache=eff_cache,
    )


def _get_nested_shadowed_scope_context(
    base_ctx: Optional[_FnScopeContext],
    extra_shadows: frozenset[str],
    cache: Dict[frozenset[str], _FnScopeContext],
) -> Optional[_FnScopeContext]:
    """Returns a _FnScopeContext incorporating nested local/class-body shadowed names."""
    if base_ctx is None or not extra_shadows:
        return base_ctx
    merged_shadows = base_ctx.shadowed_names | extra_shadows
    if (
        merged_shadows == base_ctx.shadowed_names
        and extra_shadows <= base_ctx.fn_locals
    ):
        return base_ctx
    cached = cache.get(extra_shadows)
    if cached is not None:
        return cached
    derived = _FnScopeContext(
        fn_locals=base_ctx.fn_locals | extra_shadows,
        parent_map=base_ctx._stored_parent_map,  # pylint: disable=protected-access
        shadowed_names=merged_shadows,
        has_star_import=base_ctx.has_star_import,
        scope_owner=base_ctx.scope_owner,
        mod_tree=base_ctx.mod_tree,
        enclosing_scopes=base_ctx.enclosing_scopes,
        precomputed_aux_nodes=base_ctx.precomputed_aux_nodes,
        has_nonempty_parent_map=base_ctx.has_nonempty_parent_map,
    )
    cache[extra_shadows] = derived
    return derived


class _CarrierEdge(NamedTuple):
    """Typed carrier edge distinguishing variable rebinding from in-place container mutation."""

    target: str
    sources: frozenset[str]
    is_mutation: bool = False


class _FnOuterCarrierInfo(NamedTuple):
    """Outer carrier targets directly receiving nested closures and alias edges from outer free vars."""

    direct_targets: frozenset[str]
    alias_edges: Tuple[_CarrierEdge, ...] = ()


_EMPTY_FN_OUTER_CARRIER_INFO = _FnOuterCarrierInfo(frozenset(), ())


def _finalize_fn_outer_carrier_info(
    targets: Set[str],
    local_closure_syms: Set[str],
    local_edges: Sequence[Tuple[str, Set[str]]],
    local_outer_refs: Dict[str, Set[str]],
    outer_candidates: Sequence[Tuple[str, bool, Set[str], Set[str]]],
) -> _FnOuterCarrierInfo:
    """Propagates local closure symbols and outer free-var refs into final outer carrier info."""
    if not outer_candidates:
        return (
            _FnOuterCarrierInfo(frozenset(targets), ())
            if targets
            else _EMPTY_FN_OUTER_CARRIER_INFO
        )
    if local_closure_syms and local_edges:
        _propagate_local_closure_syms(local_closure_syms, local_edges)
    if local_outer_refs and local_edges:
        changed = True
        while changed:
            changed = False
            for tgt, srcs in local_edges:
                dest: Optional[Set[str]] = None
                for src in srcs:
                    src_refs = local_outer_refs.get(src)
                    if not src_refs:
                        continue
                    if dest is None:
                        dest = local_outer_refs.setdefault(tgt, set())
                    before = len(dest)
                    dest.update(src_refs)
                    if len(dest) > before:
                        changed = True
    alias_edge_map: Dict[Tuple[str, bool], Set[str]] = {}
    for t_name, is_rebound, local_refs, direct_outer_refs in outer_candidates:
        if local_refs & local_closure_syms:
            targets.add(t_name)
        eff_outer = set(direct_outer_refs)
        for lr in local_refs:
            lr_refs = local_outer_refs.get(lr)
            if lr_refs:
                eff_outer.update(lr_refs)
        eff_outer.discard(t_name)
        if eff_outer:
            alias_edge_map.setdefault((t_name, not is_rebound), set()).update(eff_outer)
    alias_edges = tuple(
        _CarrierEdge(tgt, frozenset(srcs), is_mutation=is_mut)
        for (tgt, is_mut), srcs in sorted(alias_edge_map.items())
    )
    if not targets and not alias_edges:
        return _EMPTY_FN_OUTER_CARRIER_INFO
    return _FnOuterCarrierInfo(frozenset(targets), alias_edges)


def _extract_fn_outer_carrier_info_uncached(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> _FnOuterCarrierInfo:
    """Computes outer-scope carrier targets and outer-ref alias edges mutated inside fn_node.

    Per PEP 227, ``global`` declarations inside a ``ClassDef`` body only govern name
    resolution in that immediate class body and do not propagate into nested methods,
    lambdas, comprehensions, or nested classes.
    """
    is_root_cls = isinstance(fn_node, ast.ClassDef)
    own_locals = (
        set()
        if is_root_cls
        else _extract_strict_fn_local_names(fn_node)
    )
    local_ctx = _make_nested_fn_scope_context(
        fn_node, frozenset(own_locals), scope_ctx, use_cache=use_cache
    )
    direct_nl = _extract_direct_nonlocals(fn_node)
    direct_g = _extract_direct_nonlocals(fn_node, is_global=True)
    unshadowed_builtins: AbstractSet[str] = (
        _BUILTIN_NAMES - set(scope_ctx.shadowed_names)
        if scope_ctx is not None and not scope_ctx.has_star_import
        else frozenset()
    )
    own_locals_f = frozenset(own_locals)
    derived_ctx_cache: Dict[frozenset[str], _FnScopeContext] = {}
    empty_shadows: frozenset[str] = frozenset()
    root_cls_shadows: frozenset[str] = (
        frozenset(_extract_strict_fn_local_names(fn_node))
        if is_root_cls
        else empty_shadows
    )
    fn_g = set(fn_globals) if is_root_cls else (set(fn_globals) | direct_g)
    cls_g = direct_g if is_root_cls else set()
    targets: Set[str] = set()
    local_closure_syms: Set[str] = set()
    local_edges: List[Tuple[str, Set[str]]] = []
    local_outer_refs: Dict[str, Set[str]] = {}
    outer_candidates: List[Tuple[str, bool, Set[str], Set[str]]] = []
    stack: List[
        Tuple[ast.AST, Set[str], Set[str], Set[str], Set[str], frozenset[str]]
    ] = [
        (stmt, own_locals, direct_nl, fn_g, cls_g, root_cls_shadows)
        for stmt in fn_node.body
    ]
    while stack:
        sub, cur_locals, cur_nl, cur_fn_g, cur_cls_g, cur_cls_shadows = stack.pop()
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_closure_syms.add(sub.name)
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append(
                    (hdr, cur_locals, cur_nl, cur_fn_g, cur_cls_g, cur_cls_shadows)
                )
            dec_targets = _extract_decorator_carrier_targets(sub)
            if dec_targets:
                active_builtins = unshadowed_builtins - cur_locals - cur_cls_shadows
                sub_reads = (
                    _extract_cached_scope_free_reads(sub, use_cache=use_cache)
                    - active_builtins
                )
                sub_muts, _ = _extract_closure_nonlocal_mutations_and_deletions(
                    sub, use_cache=use_cache
                )
                if sub_reads or sub_muts:
                    for d_name in dec_targets:
                        if d_name in cur_fn_g or d_name in cur_cls_g:
                            continue
                        if d_name not in cur_locals or d_name in cur_nl:
                            targets.add(d_name)
                        else:
                            local_edges.append((d_name, {sub.name}))
            sub_direct_nl = _extract_direct_nonlocals(sub)
            sub_direct_g = _extract_direct_nonlocals(sub, is_global=True)
            if isinstance(sub, ast.ClassDef):
                sub_fn_g = cur_fn_g - sub_direct_nl
                sub_cls_g = sub_direct_g
                sub_own: Set[str] = set()
                next_cls_shadows = frozenset(_extract_strict_fn_local_names(sub))
            else:
                sub_own = _extract_strict_fn_local_names(sub)
                sub_fn_g = (
                    (cur_fn_g - sub_own) - sub_direct_nl
                ) | sub_direct_g
                sub_cls_g = set()
                next_cls_shadows = empty_shadows
            sub_nl = (
                (
                    (sub_direct_nl & cur_nl) | (sub_direct_nl - cur_locals)
                )
                - sub_fn_g
                - sub_cls_g
            )
            next_locals = (cur_locals | sub_own) - sub_fn_g - sub_cls_g
            for ch in sub.body:
                stack.append(
                    (ch, next_locals, sub_nl, sub_fn_g, sub_cls_g, next_cls_shadows)
                )
            continue
        if isinstance(sub, ast.Lambda):
            for hdr in iter_nested_scope_header_nodes(sub):
                stack.append(
                    (hdr, cur_locals, cur_nl, cur_fn_g, cur_cls_g, cur_cls_shadows)
                )
            lam_bound = _extract_strict_fn_local_names(sub)
            lam_locals = cur_locals | lam_bound
            stack.append(
                (
                    sub.body,
                    lam_locals,
                    cur_nl - lam_bound,
                    cur_fn_g - lam_bound,
                    set(),
                    empty_shadows,
                )
            )
            continue
        if isinstance(sub, _COMPREHENSION_TYPES):
            comp_exprs, comp_targets = _iter_comprehension_body_exprs_and_targets(
                sub
            )
            comp_locals = cur_locals | comp_targets
            comp_nl = cur_nl - comp_targets
            comp_fn_g = cur_fn_g - comp_targets
            stack.extend(
                [
                    *(
                        [
                            (
                                sub.generators[0].iter,
                                cur_locals,
                                cur_nl,
                                cur_fn_g,
                                cur_cls_g,
                                cur_cls_shadows,
                            )
                        ]
                        if sub.generators
                        else []
                    ),
                    *(
                        (expr, comp_locals, comp_nl, comp_fn_g, set(), empty_shadows)
                        for expr in comp_exprs
                    ),
                ]
            )
            continue
        extra_shadows = (frozenset(cur_locals) - own_locals_f) | cur_cls_shadows
        active_ctx = _get_nested_shadowed_scope_context(
            local_ctx, extra_shadows, derived_ctx_cache
        )
        strict_locals = cur_locals - cur_nl
        active_non_closure_builtins = (
            unshadowed_builtins - strict_locals - cur_cls_shadows
        )
        for is_rebound, c_targets, c_values in _iter_carrier_target_entries(
            sub, scope_ctx=active_ctx
        ):
            c_calls: Set[str] = set()
            c_reads, c_muts, c_loads = (
                _extract_carrier_value_closure_and_load_info(
                    c_values, active_ctx, called_names_out=c_calls
                )
            )
            has_inline = bool(c_reads or c_muts)
            val_refs = c_loads | c_calls
            local_val_refs = val_refs & strict_locals
            outer_val_refs = (
                val_refs - strict_locals - cur_fn_g - cur_cls_g
            ) - active_non_closure_builtins
            for t_name in c_targets:
                if t_name in cur_fn_g or t_name in cur_cls_g:
                    continue
                is_outer_target = (
                    (t_name in cur_nl)
                    if is_rebound
                    else (t_name not in cur_locals or t_name in cur_nl)
                )
                if is_outer_target:
                    if has_inline:
                        targets.add(t_name)
                    if local_val_refs or outer_val_refs:
                        outer_candidates.append(
                            (t_name, is_rebound, local_val_refs, outer_val_refs)
                        )
                else:
                    if has_inline:
                        local_closure_syms.add(t_name)
                    if local_val_refs:
                        local_edges.append((t_name, local_val_refs))
                    if outer_val_refs:
                        local_outer_refs.setdefault(t_name, set()).update(
                            outer_val_refs
                        )
        for child_node in ast.iter_child_nodes(sub):
            stack.append(
                (child_node, cur_locals, cur_nl, cur_fn_g, cur_cls_g, cur_cls_shadows)
            )
    return _finalize_fn_outer_carrier_info(
        targets, local_closure_syms, local_edges, local_outer_refs, outer_candidates
    )


def _extract_fn_outer_carrier_info(
    fn_node: ast.AST,
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> _FnOuterCarrierInfo:
    """Returns cached direct outer carrier targets and outer-ref alias edges mutated in fn_node."""
    if not isinstance(
        fn_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return _EMPTY_FN_OUTER_CARRIER_INFO
    eff_cache = use_cache and _scope_ctx_allows_node_cache(fn_node, scope_ctx)
    if not eff_cache:
        return _extract_fn_outer_carrier_info_uncached(
            fn_node, fn_globals, scope_ctx, use_cache=False
        )
    return _get_or_compute_scope_cached(
        _SessionBucket.CELL_INFO,
        _make_scope_ctx_cache_key(
            fn_node,
            "outer_carriers",
            scope_ctx,
            extra_params=(frozenset(fn_globals),),
        ),
        lambda: _extract_fn_outer_carrier_info_uncached(
            fn_node, fn_globals, scope_ctx, use_cache=True
        ),
    )


def _extract_fn_outer_carrier_targets_uncached(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef],
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> frozenset[str]:
    """Computes outer-scope container/variable names directly mutated by local closures in fn_node."""
    return _extract_fn_outer_carrier_info_uncached(
        fn_node, fn_globals, scope_ctx, use_cache=use_cache
    ).direct_targets


def _extract_fn_outer_carrier_targets(
    fn_node: ast.AST,
    fn_globals: AbstractSet[str] = frozenset(),
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    use_cache: bool = True,
) -> Set[str]:
    """Extracts outer-scope container/variable names directly mutated by local closures in fn_node."""
    return set(
        _extract_fn_outer_carrier_info(
            fn_node, fn_globals, scope_ctx, use_cache=use_cache
        ).direct_targets
    )


def _partition_class_body_shallow_stmts(
    cls_node: ast.ClassDef,
) -> Tuple[List[Union[ast.FunctionDef, ast.AsyncFunctionDef]], List[ast.stmt]]:
    """Partitions cls_node.body shallow statements into (methods, non_method_stmts)."""
    methods: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = []
    non_methods: List[ast.stmt] = []
    stack: List[ast.AST] = list(reversed(cls_node.body))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(curr)
            continue
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.stmt):
            non_methods.append(curr)
        stack.extend(reversed(list(ast.iter_child_nodes(curr))))
    return methods, non_methods


def _iter_class_body_methods(
    cls_node: ast.ClassDef,
) -> List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
    """Collects method definitions in cls_node.body without descending into nested scopes."""
    return _partition_class_body_shallow_stmts(cls_node)[0]


def _iter_class_body_non_method_stmts(cls_node: ast.ClassDef) -> List[ast.stmt]:
    """Collects non-method shallow statements in cls_node.body (e.g. class-attribute assignments)."""
    return _partition_class_body_shallow_stmts(cls_node)[1]
