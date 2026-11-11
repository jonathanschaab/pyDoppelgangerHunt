"""Severed closure cell hazard checks and enclosing cell analysis facade.

Design Note — Closure-Cell Hazard Architecture & Invariants
===========================================================
Extracting a code unit (subroutine block, nested closure, comprehension, or
complex expression) into a standalone module-level helper replaces lexical cell
sharing with value-based parameter passing and return-tuple unpacking. That
transformation is unsound whenever late-bound cell semantics, shared `nonlocal`
mutations, or suspended execution observe a different binding state than pass-
by-value/return-by-value.

The analysis is partitioned across seven modules:
1. `_df_cache`: LRU caches, parent-map and source-line caches, and composite
   `_ScopeCacheKey` indexing.
2. `_df_closures`: Lexical scope local/nonlocal/global extraction, eager
   generator-expression classification, and closure free-variable collection.
3. `_df_span_index`: 2D span/column resolution (`_SpanLineIndex`,
   `_resolve_unit_col_bounds`, `_has_unresolved_span_bounds`) and syntax-level
   lexical declaration validation (`_has_invalid_lexical_scope_declarations`).
4. `_df_with_provenance`: Typed context-manager suppression provenance
   (`_SuppressProvenance`) and the unified definite/conditional binding,
   deletion, and block-termination engine (`_analyze_block_assignment`,
   `_extract_block_definite_bindings_and_term`, `_block_terminates`).
5. `_df_cell_carriers`: Lexical scope contexts (`_FnScopeContext`), deferred
   comprehension/generator-expression capture analysis, closure carrier/factory
   extraction, and lexical-scope definition escape tracking.
6. `_df_cell_traversal`: Flow-sensitive AST frame traversal across enclosing
   function chains, producing per-function `_EnclosingFnCellInfo` summaries.
7. `_df_cell_hazards` (this module): Numbered hazard checks 1–4 and the
   top-level `_has_severed_closure_cell_hazard` entry point.

The Four Numbered Hazard Checks and Their Invariants
----------------------------------------------------
- **Check 1 — Unbound Captured Cell at Extraction Point**
  (`_check_captured_cell_mutations`):
  *Invariant*: Every enclosing-function local variable captured by an intra-unit
  closure (or read by a suspended generator/coroutine unit) that is passed as a
  helper parameter (`var in eff_input_set`) MUST be definitely bound on all
  control-flow paths prior to the unit (`var in acc_pre_bound`) and not shadowed
  by an uninitialized inner scope (`acc_pre_unbound`). Otherwise, evaluating the
  argument at the extracted helper call site raises `UnboundLocalError`, whereas
  original late-bound cell resolution would have succeeded after a later
  binding.

- **Check 2 — Externally Mutated Captured Cell**
  (`_check_captured_cell_mutations`):
  *Invariant*: Any enclosing-function variable captured by an intra-unit closure
  (or suspended unit) MUST NOT be mutated or deleted outside the unit in the
  defining function scope (`post_mutated`, `loop_mutated`, or
  `all_sibling_nonlocal_mutations`), nor mutated prior to the unit in an outer
  chain scope when an intermediate scope also touches it. Freezing the variable
  into a by-value helper argument or helper-local binding would sever late-bound
  cell updates.

- **Check 3 — Intra-Unit Nonlocal/Walrus Mutator Escapes**
  (`_check_intra_unit_nonlocal_mutators`):
  *Invariant*: No nested function, class, or lazy generator expression defined
  inside the unit may mutate an enclosing-chain variable via `nonlocal` or PEP
  572 walrus assignment, nor may an escaping intra-unit mutator (tracked per
  lexical AST definition identity via `_collect_escaped_intra_unit_defs`) mutate
  a unit-top-level variable that is shared with outer sibling closures or
  read/mutated outside the unit.

- **Check 4 — Sibling Closure Cell Sharing & Concurrent Suspension**
  (`_check_sibling_closure_sharing`):
  *Invariant*: No sibling closure in an enclosing function (resolved
  transitively across direct definitions, container/attribute carriers, factory
  calls, and alias edges) that is referenced by the unit (`sib_name in
  eff_input_set`) or active alongside unreturned expression mutations may read a
  cell mutated by the unit or mutate a cell touched by the unit. Additionally,
  in strict closure mode (`not skip_pre_unit_closures`), a suspended (`yield` /
  `async`) or expression unit MUST NOT mutate a cell captured by any concurrent
  sibling closure (`concurrent_sibling_vars`).

Soundness Boundaries
--------------------
- **Anonymous Closure Registration via Opaque Calls**: Sibling closures stored
  in local variables, containers, attributes, or returned by in-scope factory
  functions are tracked transitively. An anonymous closure passed directly to an
  opaque external call (e.g., `register(lambda: x)`) without binding a local
  carrier name is captured as an anonymous sibling closure (`s_name is None`),
  which is enforced for suspended/expression units (`concurrent_sibling_vars`)
  and unreturned expression mutations in strict mode, while synchronous
  subroutine calls to opaque external functions (`fire()`) are not assumed to
  invoke previously registered anonymous callbacks unless a carrier name is
  referenced in `eff_input_set`.
- **Dynamic Introspection**: Dynamic frame/namespace introspection (`locals()`,
  `eval()`, `exec()`, `sys._getframe()`) and zero-argument `super()` / implicit
  `__class__` cell capture are handled by separate scope/AST hazard gates
  (`has_super`, dynamic execution checks) rather than `PyCellObject` dataflow.

Hazard Naming Note (`"severed_closure_cell"`)
---------------------------------------------
Outer `global` and `nonlocal` binding escapes (e.g., an intra-unit `global x`
that re-routes assignments in the enclosing function, a subroutine mutating or
observing a mutated outer `global` declared in the enclosing function, or a
conditionally pre-bound live-in variable in `scope.py`), invalid lexical scope
declarations, and unresolved span bounds are reported under the canonical
`"severed_closure_cell"` hazard key (and `"severed closure cell"` skip reason)
for backward compatibility, while granular sub-reasons (`"cell_severing"`,
`"global_escape"`, `"invalid_lexical_declaration"`, `"unresolved_span_bounds"`,
`"conditional_live_in"`) are recorded in `scope["closure_hazard_reasons"]`.
"""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._ast_compat import is_match_node
from pydoppelgangerhunt.fixer._df_cache import (
    _ephemeral_unit_ast_context,
    _mark_ephemeral_unit_ast,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _EnclosingFnCellInfo,
    _IntraUnitMutator,
    _collect_escaped_and_executed_intra_unit_defs,
    _collect_unit_nonlocal_mutators,
    _extract_root_target_names,
    _propagate_sibling_closures,
)
from pydoppelgangerhunt.fixer._df_cell_traversal import (
    _build_enclosing_chain_cell_infos,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _collect_scope_closures_uncached,
    _extract_comprehension_walrus_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    _has_escaped_intra_unit_global_declarations,
    _has_wildcard_star_import,
    _iter_shallow_scope_nodes,
    _resolve_immediate_scope_in_fn,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _extract_direct_nonlocals,
    _find_enclosing_function_chain,
    _get_span_line_index,
    _has_invalid_lexical_scope_declarations,
    _has_unresolved_span_bounds,
    _resolve_unit_col_bounds,
)
from pydoppelgangerhunt.fixer._df_with_provenance import (
    _SAFE_EXCEPTION_TYPE_NAMES,
    _is_non_raising_store_target,
)
from pydoppelgangerhunt.parser import _node_is_effectively_async

__all__ = [
    "_HEURISTIC_GLOBAL_SAFE_BUILTINS",
    "_PROVEN_PURE_GLOBAL_SAFE_BUILTINS",
    "_PURE_GLOBAL_SAFE_BUILTINS",
    "_UnitClosureHazardContext",
    "_check_captured_cell_mutations",
    "_check_intra_unit_nonlocal_mutators",
    "_check_sibling_closure_sharing",
    "_evaluate_severed_closure_cell_hazard",
    "_has_severed_closure_cell_hazard",
    "_unit_has_global_mutating_call",
]


def _check_captured_cell_mutations(
    captured_candidates: AbstractSet[str],
    eff_input_set: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    unit_closure_reads: Optional[AbstractSet[str]] = None,
) -> bool:
    """Checks 1 & 2: captured cell variable is unbound before unit or mutated outside unit."""
    actual_closure_reads = (
        captured_candidates if unit_closure_reads is None else unit_closure_reads
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
    intra_nl_mutators: Sequence[Union[_IntraUnitMutator, Tuple[str, Set[str], bool]]],
    unit_top_locals: AbstractSet[str],
    nonlocal_set: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    is_subroutine: bool = True,
    is_suspended: bool = False,
    eff_input_set: AbstractSet[str] = frozenset(),
    enc_escaping: AbstractSet[str] = frozenset(),
) -> bool:
    """Check 3: intra-unit closure mutates an outer cell or escaping top-level variable."""
    if not intra_nl_mutators:
        return False
    all_chain_locals: Set[str] = (
        set().union(*(info.fn_locals for info in chain_infos)) if chain_infos else set()
    )
    top_locals_set = set(unit_top_locals)
    outer_escape_targets = all_chain_locals | set(enc_escaping) | set(nonlocal_set)
    for mut_entry in intra_nl_mutators:
        mutated_vars = mut_entry[1]
        escaping_nl_muts = (mutated_vars - top_locals_set) | (
            mutated_vars & (set(nonlocal_set) | set(enc_escaping))
        )
        if bool(escaping_nl_muts & outer_escape_targets):
            return True

    if not is_subroutine:
        return False

    outer_touched: Set[str] = set()
    all_post_reads: Set[str] = set()
    all_sib_reads: Set[str] = set()
    outer_sib_cells: Set[str] = set()
    for info in chain_infos:
        outer_touched.update(
            info.pre_mutated
            | info.post_mutated
            | info.loop_mutated
            | info.all_sibling_nonlocal_mutations
        )
        all_post_reads.update(info.post_reads)
        outer_sib_cells.update(info.all_sibling_nonlocal_mutations)
        for _, sib_r, _ in info.sibling_closures:
            all_sib_reads.update(sib_r)
            outer_sib_cells.update(sib_r)
    escaped_unit_names, escaped_def_ids, executed_def_ids = (
        _collect_escaped_and_executed_intra_unit_defs(
            candidate_stmts, all_post_reads | all_sib_reads
        )
    )
    for mut_entry in intra_nl_mutators:
        mut_fn_name, mutated_vars, is_escaped_def = (
            mut_entry[0],
            mut_entry[1],
            mut_entry[2],
        )
        mut_node = getattr(mut_entry, "node", None)
        mut_fn_escapes = is_escaped_def or (
            id(mut_node) in escaped_def_ids
            if isinstance(mut_node, ast.AST)
            else mut_fn_name in escaped_unit_names
        )
        mut_fn_executed = mut_fn_escapes or (
            id(mut_node) in executed_def_ids and not isinstance(mut_node, ast.ClassDef)
            if isinstance(mut_node, ast.AST)
            else True
        )
        for var in mutated_vars & top_locals_set:
            if (mut_fn_executed and var in outer_sib_cells) or (
                (mut_fn_escapes or is_suspended)
                and (
                    var in outer_touched
                    or var in all_post_reads
                    or var in eff_input_set
                )
            ):
                return True
    return False


class _UnshadowedChainClosureInfo(NamedTuple):
    """Unshadowed enclosing locals, concurrent sibling vars, post-reads, and closures."""

    chain_locals: Set[str]
    concurrent_vars: Set[str]
    post_reads: Set[str]
    sib_closures: List[Tuple[Optional[str], Set[str], Set[str]]]


def _collect_unshadowed_chain_closure_info(
    chain_infos: Sequence[_EnclosingFnCellInfo],
) -> _UnshadowedChainClosureInfo:
    """Aggregates unshadowed locals, concurrent sibling vars, post-reads, and closures across chain_infos."""
    chain_locals: Set[str] = set()
    concurrent_vars: Set[str] = set()
    post_reads: Set[str] = set()
    unshadowed_sibs: List[Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]] = []
    unshadowed_edges: List[Tuple[str, AbstractSet[str]]] = []
    for idx, info in enumerate(chain_infos):
        inner_shadowed: Set[str] = (
            set().union(
                *(chain_infos[j].fn_locals for j in range(idx + 1, len(chain_infos)))
            )
            if idx + 1 < len(chain_infos)
            else set()
        )
        unshadowed_fn_locals = set(info.fn_locals) - inner_shadowed
        chain_locals.update(unshadowed_fn_locals)
        concurrent_vars.update(
            unshadowed_fn_locals & (set(info.concurrent_sibling_vars) - inner_shadowed)
        )
        post_reads.update(
            unshadowed_fn_locals & (set(info.post_reads) - inner_shadowed)
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
    sib_closures = _propagate_sibling_closures(
        unshadowed_sibs,
        alias_edges=unshadowed_edges,
    )
    return _UnshadowedChainClosureInfo(
        chain_locals=chain_locals,
        concurrent_vars=concurrent_vars,
        post_reads=post_reads,
        sib_closures=sib_closures,
    )


def _check_sibling_closure_sharing(
    eff_input_set: AbstractSet[str],
    sub_mutated: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    helper_rebound_vars: AbstractSet[str] = frozenset(),
    is_suspended: bool = False,
    unreturned_mutations: bool = False,
    skip_pre_unit_closures: bool = False,
) -> bool:
    """Check 4: sibling closure referenced by unit (or concurrent with suspended unit) shares a mutated cell."""
    if not chain_infos:
        return False
    unshadowed = _collect_unshadowed_chain_closure_info(chain_infos)
    mutated_set = set(sub_mutated) | set(helper_rebound_vars)
    if is_suspended and bool(mutated_set & unshadowed.concurrent_vars):
        return True
    if unreturned_mutations and bool(mutated_set & unshadowed.post_reads):
        return True
    sub_touched = set(eff_input_set) | mutated_set
    for sib_name, sib_reads, sib_nl_muts in unshadowed.sib_closures:
        is_ref_sib = sib_name is not None and sib_name in eff_input_set
        if not is_ref_sib and (not unreturned_mutations or skip_pre_unit_closures):
            continue
        if bool(sib_reads & unshadowed.chain_locals & mutated_set):
            return True
        if bool(
            sib_nl_muts
            & unshadowed.chain_locals
            & (sub_touched if is_ref_sib else mutated_set)
        ):
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
    definite_stores: Optional[AbstractSet[str]] = None
    cols_resolved: bool = False
    enclosing_chain: Optional[Sequence[Tuple[int, int, ast.AST]]] = None
    lex_validated: bool = False
    strict_with_suppression: bool = False
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset()


class _ClosureHazardResult(NamedTuple):
    """Outcome of closure cell hazard evaluation with structured diagnostic reasons."""

    has_hazard: bool
    reasons: Tuple[str, ...] = ()


# Built-in functions that operate strictly on C-level object headers or primitive values
# without invoking user-defined Python dunders (`__len__`, `__bool__`, `__int__`,
# `__index__`, `__instancecheck__`, `__subclasscheck__`):
# - `id(x)`: returns the C pointer address of `x` without calling any Python method.
# - `callable(x)`: inspects `Py_TYPE(x)->tp_call` in C without calling any Python method.
# - `ord(c)`: inspects 1-char `str`/`bytes`/`bytearray` buffers in C without dunders.
_PROVEN_PURE_GLOBAL_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "callable",
        "id",
        "ord",
    }
)

# Built-ins that are side-effect-free for standard types in `strict` mode under the
# heuristic assumption that user-defined dunders (`__len__`, `__bool__`, `__int__`,
# `__index__`, `__instancecheck__`, `__subclasscheck__`) do not mutate module-level
# global variables (note: `chr(i)` invokes `__index__` via `PyNumber_AsSsize_t`).
# Rejected in `pedantic` mode (`strict_with_suppression=True`) when a unit reads an
# enclosing `global` variable.
_HEURISTIC_GLOBAL_SAFE_BUILTINS: frozenset[str] = frozenset(
    {
        "bin",
        "bool",
        "chr",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "oct",
        "range",
    }
)

_PURE_GLOBAL_SAFE_BUILTINS: frozenset[str] = (
    _PROVEN_PURE_GLOBAL_SAFE_BUILTINS | _HEURISTIC_GLOBAL_SAFE_BUILTINS
)


def _iter_immediate_unit_exec_nodes(
    stmts: Sequence[ast.AST],
) -> Iterator[ast.AST]:
    """Yields AST nodes executed immediately by stmts, visiting nested function/lambda headers without descending into deferred bodies."""
    return _iter_shallow_scope_nodes(
        stmts,
        deferred_scope_types=(ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
    )


_PRIMITIVE_ACCUMULATOR_CONST_TYPES = (int, float, complex, str, bytes, bool)
_PRIMITIVE_ACCUMULATOR_AUG_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
)
_STATIC_NON_DESCRIPTOR_VALUE_TYPES = (
    ast.Constant,
    ast.JoinedStr,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
)


def _node_contains_named_expr(node: ast.AST) -> bool:
    """Returns True if node or any immediate sub-expression (excluding deferred scopes) contains a walrus operator."""
    for sub in _iter_immediate_unit_exec_nodes((node,)):
        if isinstance(sub, ast.NamedExpr):
            return True
    return False


def _is_safe_non_descriptor_class_body_stmt(stmt: ast.stmt) -> bool:
    """Returns True when a class-body statement cannot bind a custom descriptor with __set_name__."""
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        return not _node_contains_named_expr(stmt.value)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return not stmt.decorator_list and not _node_contains_named_expr(stmt)
    if isinstance(stmt, ast.ClassDef):
        return not stmt.decorator_list and not stmt.bases and not stmt.keywords
    if isinstance(stmt, ast.Assign):
        return (
            all(isinstance(t, ast.Name) for t in stmt.targets)
            and isinstance(stmt.value, _STATIC_NON_DESCRIPTOR_VALUE_TYPES)
            and not _node_contains_named_expr(stmt.value)
        )
    if isinstance(stmt, ast.AnnAssign):
        return (
            isinstance(stmt.target, ast.Name)
            and not _node_contains_named_expr(stmt.annotation)
            and (
                stmt.value is None
                or (
                    isinstance(stmt.value, _STATIC_NON_DESCRIPTOR_VALUE_TYPES)
                    and not _node_contains_named_expr(stmt.value)
                )
            )
        )
    return False


_IMMEDIATE_GLOBAL_HAZARD_NODES = (
    ast.With,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.AsyncFor,
    ast.Delete,
)


def _extract_immediate_bool_test_exprs(node: ast.AST) -> Tuple[ast.AST, ...]:
    """Returns sub-expressions of node that Python implicitly evaluates for truthiness."""
    if isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp)):
        return (node.test,)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return (node.operand,)
    if isinstance(node, ast.BoolOp):
        return tuple(node.values)
    if isinstance(node, ast.comprehension):
        return tuple(node.ifs)
    if type(node).__name__ == "match_case":
        guard = getattr(node, "guard", None)
        if isinstance(guard, ast.AST):
            return (guard,)
    return ()


class _PedanticPrimitiveEnv(NamedTuple):
    """Statically proven scalar-primitive and primitive-tuple bindings for pedantic dunder checks."""

    primitives: frozenset[str] = frozenset()
    prim_tuples: frozenset[str] = frozenset()
    effective_shadowed: frozenset[str] = frozenset()
    all_builtins_shadowed: bool = False


def _check_proven_sync_iter(
    target: ast.AST,
    iter_expr: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    require_primitive_elts: bool = False,
) -> bool:
    """Returns True when iterating iter_expr into target avoids user dunders (and optionally binds only scalar primitives)."""
    while isinstance(iter_expr, ast.NamedExpr):
        iter_expr = iter_expr.value
    if (
        isinstance(iter_expr, ast.Constant)
        and isinstance(iter_expr.value, (str, bytes))
    ) or (isinstance(iter_expr, ast.Name) and iter_expr.id in env.prim_tuples):
        return isinstance(target, ast.Name)
    if isinstance(iter_expr, (ast.Tuple, ast.List)) and not any(
        isinstance(e, ast.Starred) for e in iter_expr.elts
    ):
        leaf_pred: Optional[Callable[[ast.AST], bool]] = (
            (lambda leaf: _is_proven_primitive_dunder_expr(leaf, env))
            if require_primitive_elts
            else None
        )
        return all(
            _is_non_raising_store_target(target, elt, leaf_predicate=leaf_pred)
            for elt in iter_expr.elts
        )
    return False


def _is_proven_primitive_subscript_expr(
    expr: ast.Subscript,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when a Load subscript avoids user dunders and yields a scalar primitive."""
    if not isinstance(expr.ctx, ast.Load):
        return False
    val: ast.AST = expr.value
    while isinstance(val, ast.NamedExpr):
        val = val.value
    slc = expr.slice
    if isinstance(val, ast.Constant) and isinstance(val.value, (str, bytes)):
        if isinstance(slc, ast.Slice):
            return all(
                bound is None or _is_proven_primitive_dunder_expr(bound, env)
                for bound in (slc.lower, slc.upper, slc.step)
            )
        return _is_proven_primitive_dunder_expr(slc, env)
    if isinstance(slc, ast.Slice) or not _is_proven_primitive_dunder_expr(slc, env):
        return False
    if isinstance(val, ast.Name) and val.id in env.prim_tuples:
        return True
    if isinstance(val, (ast.Tuple, ast.List)) and not any(
        isinstance(elt, ast.Starred) for elt in val.elts
    ):
        return all(_is_proven_primitive_dunder_expr(elt, env) for elt in val.elts)
    return False


def _is_proven_primitive_dunder_expr(
    expr: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when expr evaluates to a built-in scalar primitive without invoking user dunders."""
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        return expr.id in env.primitives
    if isinstance(expr, ast.NamedExpr):
        return _is_proven_primitive_dunder_expr(expr.value, env)
    if isinstance(expr, ast.Subscript):
        return _is_proven_primitive_subscript_expr(expr, env)
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.Not):
            return _is_proven_primitive_bool_expr(expr.operand, env)
        return _is_proven_primitive_dunder_expr(expr.operand, env)
    if isinstance(expr, ast.BinOp) and not isinstance(expr.op, ast.MatMult):
        return _is_proven_primitive_dunder_expr(
            expr.left, env
        ) and _is_proven_primitive_dunder_expr(expr.right, env)
    if isinstance(expr, ast.BoolOp):
        return all(_is_proven_primitive_dunder_expr(val, env) for val in expr.values)
    if isinstance(expr, ast.IfExp):
        return (
            _is_proven_primitive_bool_expr(expr.test, env)
            and _is_proven_primitive_dunder_expr(expr.body, env)
            and _is_proven_primitive_dunder_expr(expr.orelse, env)
        )
    if isinstance(expr, ast.Compare):
        return _is_proven_safe_compare(expr, env)
    if isinstance(expr, ast.JoinedStr):
        return all(
            isinstance(part, ast.Constant)
            or (
                isinstance(part, ast.FormattedValue)
                and _is_proven_safe_formatted_value(part, env)
            )
            for part in expr.values
        )
    return (
        isinstance(expr, ast.Call)
        and not env.all_builtins_shadowed
        and isinstance(expr.func, ast.Name)
        and expr.func.id in _PROVEN_PURE_GLOBAL_SAFE_BUILTINS
        and expr.func.id not in env.effective_shadowed
        and len(expr.args) == 1
        and not expr.keywords
        and not isinstance(expr.args[0], ast.Starred)
    )


def _is_proven_safe_formatted_value(
    node: ast.FormattedValue,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when an f-string FormattedValue formats only a proven scalar primitive."""
    return _is_proven_primitive_dunder_expr(node.value, env) and (
        node.format_spec is None
        or _is_proven_primitive_dunder_expr(node.format_spec, env)
    )


def _is_proven_primitive_container_for_in(
    rhs: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when rhs of 'in'/'not in' is a built-in container of proven primitives."""
    while isinstance(rhs, ast.NamedExpr):
        rhs = rhs.value
    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (str, bytes)):
        return True
    if isinstance(rhs, ast.Name) and rhs.id in env.prim_tuples:
        return True
    if isinstance(rhs, (ast.Tuple, ast.List, ast.Set)):
        return not any(isinstance(elt, ast.Starred) for elt in rhs.elts) and all(
            _is_proven_primitive_dunder_expr(elt, env) for elt in rhs.elts
        )
    if isinstance(rhs, ast.Dict):
        return all(
            key is not None and _is_proven_primitive_dunder_expr(key, env)
            for key in rhs.keys
        )
    return False


def _is_proven_safe_compare(
    node: ast.Compare,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when every step of a Compare expression cannot invoke user comparison/containment dunders."""
    operands = (node.left, *node.comparators)
    for lhs, op, rhs in zip(operands, node.ops, node.comparators):
        if isinstance(op, (ast.Is, ast.IsNot)):
            continue
        if isinstance(op, (ast.In, ast.NotIn)):
            if not (
                _is_proven_primitive_dunder_expr(lhs, env)
                and _is_proven_primitive_container_for_in(rhs, env)
            ):
                return False
        elif not (
            _is_proven_primitive_dunder_expr(lhs, env)
            and _is_proven_primitive_dunder_expr(rhs, env)
        ):
            return False
    return True


def _is_proven_safe_subscript_load(
    node: ast.Subscript,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when a Load subscript indexes a literal or proven sequence/string with primitive bounds."""
    if not isinstance(node.ctx, ast.Load):
        return False
    val: ast.AST = node.value
    while isinstance(val, ast.NamedExpr):
        val = val.value
    val_ok = (
        (isinstance(val, ast.Constant) and isinstance(val.value, (str, bytes)))
        or (isinstance(val, ast.Name) and val.id in env.prim_tuples)
        or (
            isinstance(val, (ast.Tuple, ast.List))
            and not any(isinstance(elt, ast.Starred) for elt in val.elts)
        )
    )
    if not val_ok:
        return False
    slc = node.slice
    if isinstance(slc, ast.Slice):
        return all(
            bound is None or _is_proven_primitive_dunder_expr(bound, env)
            for bound in (slc.lower, slc.upper, slc.step)
        )
    return _is_proven_primitive_dunder_expr(slc, env)


def _is_proven_primitive_bool_expr(
    expr: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when evaluating expr in a boolean context cannot invoke user-defined __bool__ or __len__."""
    if isinstance(expr, _STATIC_NON_DESCRIPTOR_VALUE_TYPES):
        return True
    if isinstance(expr, ast.Name) and expr.id in env.prim_tuples:
        return True
    sub_bool_exprs: Tuple[ast.AST, ...] = ()
    if isinstance(expr, ast.NamedExpr):
        sub_bool_exprs = (expr.value,)
    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        sub_bool_exprs = (expr.operand,)
    elif isinstance(expr, ast.BoolOp):
        sub_bool_exprs = tuple(expr.values)
    elif isinstance(expr, ast.IfExp):
        sub_bool_exprs = (expr.test, expr.body, expr.orelse)
    if sub_bool_exprs:
        return all(
            _is_proven_primitive_bool_expr(sub_e, env) for sub_e in sub_bool_exprs
        )
    return _is_proven_primitive_dunder_expr(expr, env)


def _node_invokes_pedantic_dunder_protocol(
    node: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when an immediate AST node may invoke user-defined dunder or import hooks in pedantic mode."""
    if any(
        not _is_proven_primitive_bool_expr(b_expr, env)
        for b_expr in _extract_immediate_bool_test_exprs(node)
    ):
        return True
    if isinstance(node, (ast.Attribute, ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.BinOp):
        return not _is_proven_primitive_dunder_expr(node, env)
    if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.Not):
        return not _is_proven_primitive_dunder_expr(node.operand, env)
    if isinstance(node, ast.Compare):
        return not _is_proven_safe_compare(node, env)
    if isinstance(node, ast.Subscript):
        return not _is_proven_safe_subscript_load(node, env)
    if isinstance(node, (ast.For, ast.comprehension)):
        return not _check_proven_sync_iter(node.target, node.iter, env)
    if isinstance(node, ast.FormattedValue):
        return not _is_proven_safe_formatted_value(node, env)
    if isinstance(node, ast.Set):
        return not all(_is_proven_primitive_dunder_expr(elt, env) for elt in node.elts)
    if isinstance(node, ast.SetComp):
        return not _is_proven_primitive_dunder_expr(node.elt, env)
    if isinstance(node, ast.Dict):
        return not all(
            (key is not None and _is_proven_primitive_dunder_expr(key, env))
            or (key is None and isinstance(val, ast.Dict))
            for key, val in zip(node.keys, node.values)
        )
    if isinstance(node, ast.DictComp):
        return not _is_proven_primitive_dunder_expr(node.key, env)
    if isinstance(node, ast.Assign):
        return any(
            isinstance(tgt, (ast.Tuple, ast.List))
            and not _is_non_raising_store_target(tgt, node.value)
            for tgt in node.targets
        )
    if isinstance(node, ast.Starred) and isinstance(node.ctx, ast.Load):
        return not isinstance(node.value, (ast.Tuple, ast.List))
    if isinstance(node, ast.Call) and (
        any(
            isinstance(arg, ast.Starred)
            and not isinstance(arg.value, (ast.Tuple, ast.List))
            for arg in node.args
        )
        or any(
            kw.arg is None and not isinstance(kw.value, ast.Dict)
            for kw in node.keywords
        )
    ):
        return True
    if is_match_node(node):  # pragma: no cover (py310+)
        subject = getattr(node, "subject", None)
        return not (
            isinstance(subject, ast.AST)
            and _is_proven_primitive_dunder_expr(subject, env)
            and not any(
                type(sub).__name__ == "MatchClass"
                for case in getattr(node, "cases", ())
                for sub in ast.walk(getattr(case, "pattern", node))
            )
        )
    if isinstance(node, ast.Raise) and node.exc is not None:
        return not (
            not env.all_builtins_shadowed
            and isinstance(node.exc, ast.Name)
            and node.exc.id in _SAFE_EXCEPTION_TYPE_NAMES
            and node.exc.id not in env.effective_shadowed
            and node.cause is None
        )
    return False


def _extract_non_assign_bound_names(node: ast.AST) -> Set[str]:
    """Returns variable names bound or re-routed by non-assignment/non-for AST nodes."""
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    if isinstance(node, ast.Delete):
        out: Set[str] = set()
        for tgt in node.targets:
            out.update(_extract_root_target_names(tgt))
        return out
    if isinstance(node, (ast.With, ast.AsyncWith)):
        out = set()
        for item in node.items:
            if item.optional_vars is not None:
                out.update(_extract_root_target_names(item.optional_vars))
        return out
    if isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {
            alias.asname or alias.name.split(".", 1)[0]
            for alias in node.names
            if alias.name != "*"
        }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    cls_name = type(node).__name__
    if cls_name in ("MatchAs", "MatchStar"):
        m_name = getattr(node, "name", None)
        return {m_name} if isinstance(m_name, str) else set()
    if cls_name == "MatchMapping":
        r_name = getattr(node, "rest", None)
        return {r_name} if isinstance(r_name, str) else set()
    return set()


def _extract_unpack_leaf_names(target: ast.AST) -> Optional[Set[str]]:
    """Returns leaf Name ids when target is a Name or unstarred nested Tuple/List of Names."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)) and not any(
        isinstance(e, ast.Starred) for e in target.elts
    ):
        names: Set[str] = set()
        for elt in target.elts:
            sub = _extract_unpack_leaf_names(elt)
            if sub is None:
                return None
            names.update(sub)
        return names
    return None


def _is_proven_primitive_target_value(
    target: ast.AST,
    value: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when assigning value to target binds every leaf Name to a proven scalar primitive."""
    return _is_non_raising_store_target(
        target,
        value,
        leaf_predicate=lambda leaf: _is_proven_primitive_dunder_expr(leaf, env),
    )


def _record_immediate_expr_names(nodes: Sequence[ast.AST], seen: Set[str]) -> None:
    """Adds all immediate ast.Name identifiers under nodes into seen."""
    for sub in _iter_immediate_unit_exec_nodes(nodes):
        if isinstance(sub, ast.Name):
            seen.add(sub.id)


def _seed_pedantic_candidates(
    candidate_stmts: Sequence[ast.stmt],
    immediate_nodes: Sequence[ast.AST],
    class_body_node_ids: AbstractSet[int],
) -> Tuple[Set[str], Dict[str, ast.Tuple]]:
    """Seeds candidate scalar primitives and primitive-tuple bindings unconditionally initialized before first read."""
    seen_before_init: Set[str] = set()
    cand_prims: Set[str] = set()
    cand_tuples: Dict[str, ast.Tuple] = {}
    for stmt in candidate_stmts:
        if isinstance(stmt, ast.Assign):
            _record_immediate_expr_names((stmt.value,), seen_before_init)
            rhs: ast.AST = stmt.value
            while isinstance(rhs, ast.NamedExpr):
                rhs = rhs.value
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                t_name = stmt.targets[0].id
                if t_name not in seen_before_init:
                    if isinstance(rhs, ast.Tuple) and not any(
                        isinstance(e, ast.Starred) for e in rhs.elts
                    ):
                        cand_tuples[t_name] = rhs
                    else:
                        cand_prims.add(t_name)
            else:
                for tgt in stmt.targets:
                    leaf_names = _extract_unpack_leaf_names(tgt)
                    if leaf_names is not None:
                        for t_name in leaf_names:
                            if t_name not in seen_before_init:
                                cand_prims.add(t_name)
                    else:
                        _record_immediate_expr_names((tgt,), seen_before_init)
            for tgt in stmt.targets:
                seen_before_init.update(_extract_root_target_names(tgt))
        elif isinstance(stmt, ast.AnnAssign):
            _record_immediate_expr_names((stmt.annotation,), seen_before_init)
            if stmt.value is not None:
                _record_immediate_expr_names((stmt.value,), seen_before_init)
                rhs = stmt.value
                while isinstance(rhs, ast.NamedExpr):
                    rhs = rhs.value
                if (
                    isinstance(stmt.target, ast.Name)
                    and stmt.target.id not in seen_before_init
                ):
                    if isinstance(rhs, ast.Tuple) and not any(
                        isinstance(e, ast.Starred) for e in rhs.elts
                    ):
                        cand_tuples[stmt.target.id] = rhs
                    else:
                        cand_prims.add(stmt.target.id)
                else:
                    _record_immediate_expr_names((stmt.target,), seen_before_init)
                seen_before_init.update(_extract_root_target_names(stmt.target))
            elif not isinstance(stmt.target, ast.Name):
                _record_immediate_expr_names((stmt.target,), seen_before_init)
        elif isinstance(stmt, ast.For):
            _record_immediate_expr_names((stmt.iter,), seen_before_init)
            it: ast.AST = stmt.iter
            while isinstance(it, ast.NamedExpr):
                it = it.value
            is_nonempty = (
                (
                    isinstance(it, ast.Constant)
                    and isinstance(it.value, (str, bytes))
                    and len(it.value) > 0
                )
                or (
                    isinstance(it, (ast.Tuple, ast.List))
                    and len(it.elts) > 0
                    and not any(isinstance(e, ast.Starred) for e in it.elts)
                )
                or (
                    isinstance(it, ast.Name)
                    and it.id in cand_tuples
                    and len(cand_tuples[it.id].elts) > 0
                )
            )
            leaf_names = _extract_unpack_leaf_names(stmt.target)
            if (
                is_nonempty
                and leaf_names is not None
                and not (leaf_names & seen_before_init)
            ):
                cand_prims.update(leaf_names)
            _record_immediate_expr_names(
                (stmt.target, *stmt.body, *stmt.orelse), seen_before_init
            )
        else:
            _record_immediate_expr_names((stmt,), seen_before_init)

    name_occ_counts: Dict[str, int] = {}
    binding_counts: Dict[str, int] = {}
    for node in immediate_nodes:
        if isinstance(node, ast.Name):
            name_occ_counts[node.id] = name_occ_counts.get(node.id, 0) + 1
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                binding_counts[node.id] = binding_counts.get(node.id, 0) + 1
        for b_name in _extract_non_assign_bound_names(node):
            binding_counts[b_name] = binding_counts.get(b_name, 0) + 1

    for node in immediate_nodes:
        if id(node) in class_body_node_ids:
            continue
        if isinstance(node, ast.For):
            leaf_names = _extract_unpack_leaf_names(node.target)
            if leaf_names:
                scope_counts: Dict[str, int] = {}
                for sub in _iter_immediate_unit_exec_nodes((node.target, *node.body)):
                    if isinstance(sub, ast.Name):
                        scope_counts[sub.id] = scope_counts.get(sub.id, 0) + 1
                for t_name in leaf_names:
                    if scope_counts.get(t_name, 0) == name_occ_counts.get(t_name, 0):
                        cand_prims.add(t_name)
        elif isinstance(node, _COMPREHENSION_TYPES) and node.generators:
            comp_counts: Dict[str, int] = {}
            for sub in _iter_immediate_unit_exec_nodes((node,)):
                if isinstance(sub, ast.Name):
                    comp_counts[sub.id] = comp_counts.get(sub.id, 0) + 1
            prior_gen_names: Set[str] = set()
            for gen in node.generators:
                _record_immediate_expr_names((gen.iter,), prior_gen_names)
                leaf_names = (
                    _extract_unpack_leaf_names(gen.target)
                    if not gen.is_async
                    else None
                )
                if leaf_names:
                    for t_name in leaf_names:
                        if t_name not in prior_gen_names and comp_counts.get(
                            t_name, 0
                        ) == name_occ_counts.get(t_name, 0):
                            cand_prims.add(t_name)
                _record_immediate_expr_names((gen.target, *gen.ifs), prior_gen_names)

    cand_tuples = {
        k: v for k, v in cand_tuples.items() if binding_counts.get(k, 0) == 1
    }
    cand_prims -= set(cand_tuples)
    return cand_prims, cand_tuples


def _collect_pedantic_primitive_env(
    candidate_stmts: Sequence[ast.stmt],
    immediate_nodes: Sequence[ast.AST],
    class_body_node_ids: AbstractSet[int],
    *,
    effective_shadowed: AbstractSet[str],
    all_builtins_shadowed: bool,
) -> _PedanticPrimitiveEnv:
    """Computes the greatest fixed point of local scalar-primitive and primitive-tuple bindings."""
    eff_shadowed_fs = frozenset(effective_shadowed)
    empty_local_env = _PedanticPrimitiveEnv(
        effective_shadowed=eff_shadowed_fs,
        all_builtins_shadowed=all_builtins_shadowed,
    )
    cur_prims, cand_tuples = _seed_pedantic_candidates(
        candidate_stmts, immediate_nodes, class_body_node_ids
    )
    cur_tuples: Set[str] = set(cand_tuples)
    while cur_prims or cur_tuples:
        env = _PedanticPrimitiveEnv(
            primitives=frozenset(cur_prims),
            prim_tuples=frozenset(cur_tuples),
            effective_shadowed=eff_shadowed_fs,
            all_builtins_shadowed=all_builtins_shadowed,
        )
        next_tuples = {
            t_name
            for t_name in cur_tuples
            if all(
                _is_proven_primitive_dunder_expr(elt, env)
                for elt in cand_tuples[t_name].elts
            )
        }
        disq_prims: Set[str] = set()
        for node in immediate_nodes:
            disq_prims.update(_extract_non_assign_bound_names(node))
            node_env = empty_local_env if id(node) in class_body_node_ids else env
            if isinstance(node, (ast.AugAssign, ast.AsyncFor)):
                disq_prims.update(_extract_root_target_names(node.target))
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if not _is_proven_primitive_target_value(tgt, node.value, node_env):
                        disq_prims.update(_extract_root_target_names(tgt))
            elif isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    if not _is_proven_primitive_target_value(
                        node.target, node.value, node_env
                    ):
                        disq_prims.update(_extract_root_target_names(node.target))
                elif not isinstance(node.target, ast.Name):
                    disq_prims.update(_extract_root_target_names(node.target))
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                if not _is_proven_primitive_dunder_expr(node.value, node_env):
                    disq_prims.add(node.target.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                if getattr(node, "is_async", 0) or not _check_proven_sync_iter(
                    node.target, node.iter, node_env, require_primitive_elts=True
                ):
                    disq_prims.update(_extract_root_target_names(node.target))
        next_prims = cur_prims - disq_prims
        if next_prims == cur_prims and next_tuples == cur_tuples:
            return _PedanticPrimitiveEnv(
                primitives=frozenset(next_prims),
                prim_tuples=frozenset(next_tuples),
                effective_shadowed=eff_shadowed_fs,
                all_builtins_shadowed=all_builtins_shadowed,
            )
        cur_prims, cur_tuples = next_prims, next_tuples
    return empty_local_env


def _collect_local_primitive_accumulators(
    candidate_stmts: Sequence[ast.stmt],
    immediate_nodes: Sequence[ast.AST],
) -> Set[str]:
    """Identifies local variables unconditionally initialized to a primitive constant before any read or augassign."""
    seen_before_init: Set[str] = set()
    initialized_primitives: Set[str] = set()
    for stmt in candidate_stmts:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, _PRIMITIVE_ACCUMULATOR_CONST_TYPES)
        ):
            t_name = stmt.targets[0].id
            if t_name not in seen_before_init:
                initialized_primitives.add(t_name)
                continue
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, _PRIMITIVE_ACCUMULATOR_CONST_TYPES)
        ):
            t_name = stmt.target.id
            if t_name not in seen_before_init:
                initialized_primitives.add(t_name)
                continue
        _record_immediate_expr_names((stmt,), seen_before_init)
    if not initialized_primitives:
        return set()
    disqualified: Set[str] = set()
    for node in immediate_nodes:
        disqualified.update(_extract_non_assign_bound_names(node))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if not isinstance(node.op, _PRIMITIVE_ACCUMULATOR_AUG_OPS):
                disqualified.add(node.target.id)
        elif isinstance(node, ast.Assign):
            is_prim = isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, _PRIMITIVE_ACCUMULATOR_CONST_TYPES
            )
            if (
                not is_prim
                or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
            ):
                for t in node.targets:
                    disqualified.update(_extract_root_target_names(t))
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                is_prim = isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, _PRIMITIVE_ACCUMULATOR_CONST_TYPES
                )
                if not is_prim or not isinstance(node.target, ast.Name):
                    disqualified.update(_extract_root_target_names(node.target))
            elif not isinstance(node.target, ast.Name):
                disqualified.update(_extract_root_target_names(node.target))
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            if not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, _PRIMITIVE_ACCUMULATOR_CONST_TYPES)
            ):
                disqualified.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            disqualified.update(_extract_root_target_names(node.target))
    return initialized_primitives - disqualified


def _collect_immediate_class_body_node_ids(
    immediate_nodes: Sequence[ast.AST],
) -> frozenset[int]:
    """Returns object ids of immediate AST nodes nested inside any ClassDef body."""
    ids: Set[int] = set()
    for node in immediate_nodes:
        if isinstance(node, ast.ClassDef):
            for sub in _iter_immediate_unit_exec_nodes(node.body):
                ids.add(id(sub))
    return frozenset(ids)


def _unit_has_global_mutating_call(
    candidate_stmts: Sequence[ast.stmt],
    shadowed_names: AbstractSet[str],
    *,
    has_star_import: bool = False,
    allow_heuristic_builtins: bool = True,
) -> bool:
    """Returns True when candidate_stmts contain an immediately executed call or mutation protocol hook other than an allowed unshadowed builtin.

    When an extracted helper reads a variable declared ``global`` in an enclosing
    function, the helper receives that global's value as a snapshot parameter at
    call entry. Any impure or user-defined call, setter/augmented-assignment
    protocol hook (``__setattr__``, ``__setitem__``, ``__iadd__``, etc.),
    class-body descriptor hook (``__set_name__`` invoked by ``type.__new__``),
    context-manager protocol hook (``with`` / ``async with``), coroutine/generator
    suspension point (``await``, ``yield``, ``yield from``, ``async for``, or async
    comprehensions), or deletion (``del``) executed during the unit could mutate the
    module-level global mid-execution, causing subsequent reads inside the helper to
    observe the stale parameter rather than the updated global. Deferred bodies of
    nested functions and lambdas are not traversed here (their headers are traversed,
    and any direct calls or closure escapes are handled separately).

    In ``strict`` mode (``allow_heuristic_builtins=True``), both
    ``_PROVEN_PURE_GLOBAL_SAFE_BUILTINS`` and ``_HEURISTIC_GLOBAL_SAFE_BUILTINS``
    are permitted when unshadowed, along with implicit boolean truth-value tests,
    read-only non-call AST expressions (binary/comparison operators, attribute and
    subscript reads, f-strings, and synchronous ``for`` iteration), and
    arithmetic/bitwise augmented assignments (``_PRIMITIVE_ACCUMULATOR_AUG_OPS``,
    excluding ``@=``) on local variables unconditionally initialized inside the unit
    to a primitive constant (e.g., ``total = 0; total += x``).

    In ``pedantic`` mode (``allow_heuristic_builtins=False``), only
    ``_PROVEN_PURE_GLOBAL_SAFE_BUILTINS`` (``callable``, ``id``, ``ord``) are
    permitted when unshadowed, all augmented assignments are rejected, and all
    implicit dunder-invoking AST operations on unproven operands — including binary
    and unary operators (``__add__``/``__radd__``, ``__neg__``, etc.), non-identity
    comparisons and membership tests (``__eq__``, ``__lt__``, ``__contains__``),
    attribute reads (``@property``, descriptors, ``__getattribute__``/``__getattr__``),
    subscript reads (``__getitem__``, ``__index__``), synchronous ``for`` and
    comprehension iteration (``__iter__``/``__next__``), sequence/iterable/mapping
    unpacking, f-string formatting (``__format__``/``__str__``/``__repr__``),
    ``set``/``dict`` hashing (``__hash__``), ``match`` class/dunder patterns,
    ``import`` statements, non-builtin ``raise`` instantiation, and implicit boolean
    truthiness (``__bool__``/``__len__``) — are rejected unless their operands are
    statically proven built-in primitives or literal containers.
    """
    allowed_builtins = (
        _PURE_GLOBAL_SAFE_BUILTINS
        if allow_heuristic_builtins
        else _PROVEN_PURE_GLOBAL_SAFE_BUILTINS
    )
    unit_stmt_mod = ast.Module(
        body=[s for s in candidate_stmts if isinstance(s, ast.stmt)],
        type_ignores=[],
    )
    effective_shadowed: Set[str] = set(
        shadowed_names
    ) | _extract_single_node_local_names(unit_stmt_mod, include_globals=True)
    immediate_nodes = tuple(_iter_immediate_unit_exec_nodes(candidate_stmts))
    local_accumulators: Set[str] = (
        _collect_local_primitive_accumulators(candidate_stmts, immediate_nodes)
        if allow_heuristic_builtins
        else set()
    )
    for node in immediate_nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            effective_shadowed.add(node.id)
        else:
            effective_shadowed.update(_extract_non_assign_bound_names(node))
        if isinstance(node, ast.ClassDef):
            effective_shadowed.update(
                _extract_single_node_local_names(node, include_globals=True)
            )
        elif isinstance(node, _COMPREHENSION_TYPES):
            effective_shadowed.update(
                _extract_single_node_local_names(node, include_globals=True)
            )
    all_builtins_shadowed = has_star_import or ("*" in effective_shadowed)
    class_body_node_ids = (
        _collect_immediate_class_body_node_ids(immediate_nodes)
        if not allow_heuristic_builtins
        else frozenset()
    )
    pedantic_env = (
        _collect_pedantic_primitive_env(
            candidate_stmts,
            immediate_nodes,
            class_body_node_ids,
            effective_shadowed=effective_shadowed,
            all_builtins_shadowed=all_builtins_shadowed,
        )
        if not allow_heuristic_builtins
        else _PedanticPrimitiveEnv()
    )
    empty_class_env = _PedanticPrimitiveEnv(
        effective_shadowed=pedantic_env.effective_shadowed,
        all_builtins_shadowed=all_builtins_shadowed,
    )
    for node in immediate_nodes:
        if isinstance(node, _IMMEDIATE_GLOBAL_HAZARD_NODES):
            return True
        if isinstance(node, ast.comprehension) and bool(node.is_async):
            return True
        if isinstance(node, ast.AugAssign):
            if not (
                allow_heuristic_builtins
                and isinstance(node.target, ast.Name)
                and isinstance(node.op, _PRIMITIVE_ACCUMULATOR_AUG_OPS)
                and node.target.id in local_accumulators
            ):
                return True
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            return True
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.decorator_list
        ):
            return True
        if isinstance(node, ast.ClassDef):
            if node.bases or node.keywords:
                return True
            if not all(_is_safe_non_descriptor_class_body_stmt(s) for s in node.body):
                return True
        if not allow_heuristic_builtins and _node_invokes_pedantic_dunder_protocol(
            node,
            empty_class_env if id(node) in class_body_node_ids else pedantic_env,
        ):
            return True
        if isinstance(node, ast.Call):
            if all_builtins_shadowed or not (
                isinstance(node.func, ast.Name)
                and node.func.id in allowed_builtins
                and node.func.id not in effective_shadowed
            ):
                return True
    return False


def _append_hazard_reason(
    reasons: List[str],
    reason: str,
) -> None:
    """Appends a granular closure hazard sub-reason if not already present."""
    if reason not in reasons:
        reasons.append(reason)


def _append_global_or_cell_hazard(
    reasons: List[str],
    names: AbstractSet[str],
    outer_globals: AbstractSet[str],
) -> None:
    """Appends 'global_escape' when names overlap outer_globals, otherwise 'cell_severing'."""
    _append_hazard_reason(
        reasons,
        "global_escape" if bool(names & outer_globals) else "cell_severing",
    )


def _evaluate_severed_closure_cell_hazard(
    ctx: _UnitClosureHazardContext,
) -> _ClosureHazardResult:
    """Evaluates whether extracting a unit to module scope would sever closure cell bindings.

    Note:
        When ``ctx.skip_pre_unit_closures`` is True (lenient mode), speculative checks for
        unreferenced sibling closures are relaxed in three places:
        1. Unseeded conditional stores only check ``imm_sib_reads & imm_info.post_reads``
           rather than all ``imm_sib_reads``.
        2. The ``is_suspended`` check against ``concurrent_sibling_vars`` in Check 4 is
           disabled.
        3. Unreferenced sibling closures (``not is_ref_sib``) are skipped in Check 4 even
           when ``unreturned_mutations`` is True.
        Checks 1, 2, 3, post-unit reads of ``unreturned_mutations``, and Check 4 for
        sibling closures explicitly referenced via ``eff_input_set`` remain enforced in
        both modes, guaranteeing monotonicity (lenient hazard implies strict hazard).
    """
    reasons: List[str] = []
    if not ctx.is_subroutine and ctx.unit_kind not in (
        "closure",
        "comprehension",
        "complex_expr",
    ):
        return _ClosureHazardResult(False, ())
    if ctx.mod_tree is None or ctx.u_start <= 0:
        _append_hazard_reason(reasons, "unresolved_span_bounds")
        return _ClosureHazardResult(True, tuple(reasons))

    u_start_col, u_end_col = (
        (ctx.u_start_col, ctx.u_end_col)
        if ctx.cols_resolved
        else _resolve_unit_col_bounds(
            ctx.candidate_stmts,
            ctx.mod_tree,
            ctx.u_start,
            ctx.u_end,
            ctx.u_start_col,
            ctx.u_end_col,
        )
    )
    enclosing_chain = (
        list(ctx.enclosing_chain)
        if ctx.enclosing_chain is not None
        else _find_enclosing_function_chain(
            ctx.mod_tree, ctx.u_start, ctx.u_end, u_start_col, u_end_col
        )
    )
    if not ctx.lex_validated and _has_invalid_lexical_scope_declarations(
        ctx.candidate_stmts,
        enclosing_chain=enclosing_chain,
        mod_tree=ctx.mod_tree,
    ):
        _append_hazard_reason(reasons, "invalid_lexical_declaration")
        return _ClosureHazardResult(True, tuple(reasons))
    if not enclosing_chain:
        span_idx = _get_span_line_index(ctx.mod_tree)
        overlap = span_idx.has_closure_overlap(ctx.u_start, ctx.u_end)
        has_unresolved = overlap.has_missing_end or (
            overlap.has_overlapping_closure
            and (u_start_col is None or u_end_col is None)
            and _has_unresolved_span_bounds(
                ctx.mod_tree,
                ctx.u_start,
                ctx.u_end,
                u_start_col,
                u_end_col,
                ctx.candidate_stmts,
            )
        )
        if has_unresolved:
            _append_hazard_reason(reasons, "unresolved_span_bounds")
            return _ClosureHazardResult(True, tuple(reasons))
        return _ClosureHazardResult(False, ())
    if _has_unresolved_span_bounds(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_start_col,
        u_end_col,
        ctx.candidate_stmts,
    ):
        _append_hazard_reason(reasons, "unresolved_span_bounds")
        return _ClosureHazardResult(True, tuple(reasons))

    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    _mark_ephemeral_unit_ast(unit_mod)
    is_expr_unit = (
        ctx.unit_kind in ("comprehension", "complex_expr") and not ctx.is_subroutine
    )
    unit_stores_affect_enclosing = ctx.is_subroutine or is_expr_unit
    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    nonlocal_set = set(ctx.nonlocals)
    unit_mutated_names = store_set | set(ctx.deleted_names)
    explicit_nonlocal_mutations: Set[str] = nonlocal_set & unit_mutated_names
    helper_rebound_vars: Set[str] = (
        (unit_mutated_names - explicit_nonlocal_mutations)
        if ctx.is_subroutine
        else set()
    )
    with _ephemeral_unit_ast_context(unit_mod):
        immediate_cell_mutations: Set[str] = (
            (_extract_comprehension_walrus_names(unit_mod) | explicit_nonlocal_mutations)
            if is_expr_unit
            else explicit_nonlocal_mutations
        )
        escaping_unit_mutations: Set[str] = helper_rebound_vars | immediate_cell_mutations
        if not unit_stores_affect_enclosing and bool(explicit_nonlocal_mutations):
            _append_hazard_reason(reasons, "cell_severing")
            return _ClosureHazardResult(True, tuple(reasons))

        innermost_stmt_fn: ast.AST = next(
            (
                fn_n
                for _, _, fn_n in reversed(enclosing_chain)
                if isinstance(fn_n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            enclosing_chain[-1][2],
        )
        if _has_escaped_intra_unit_global_declarations(
            innermost_stmt_fn,
            ctx.u_start,
            ctx.u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
            is_subroutine=ctx.is_subroutine,
        ):
            _append_hazard_reason(reasons, "global_escape")
            return _ClosureHazardResult(True, tuple(reasons))
        enc_nonlocals, enc_outer_globals = _extract_scope_nonlocals_and_outer_globals(
            innermost_stmt_fn,
            ctx.u_start,
            ctx.u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
        fn_scopes_in_chain = [
            fn_n
            for _, _, fn_n in enclosing_chain
            if isinstance(fn_n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(fn_scopes_in_chain) > 1:
            for idx_fn, outer_fn in enumerate(fn_scopes_in_chain[:-1]):
                _, outer_og = _extract_scope_nonlocals_and_outer_globals(
                    outer_fn,
                    ctx.u_start,
                    ctx.u_end,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                )
                if outer_og:
                    inner_locals: Set[str] = set()
                    for inner_fn in fn_scopes_in_chain[idx_fn + 1 :]:
                        inner_locals.update(
                            _extract_single_node_local_names(
                                inner_fn, include_globals=False
                            )
                        )
                    enc_outer_globals.update((outer_og - inner_locals) - enc_nonlocals)
        enc_escaping = enc_nonlocals | enc_outer_globals
        if unit_stores_affect_enclosing and bool(
            escaping_unit_mutations
            & (enc_outer_globals if ctx.is_subroutine else enc_escaping)
        ):
            _append_global_or_cell_hazard(
                reasons, escaping_unit_mutations, enc_outer_globals
            )
            return _ClosureHazardResult(True, tuple(reasons))
        imm_scope = _resolve_immediate_scope_in_fn(
            innermost_stmt_fn,
            ctx.u_start,
            ctx.u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
        unit_enclosing_scopes: List[ast.AST] = [
            enc_fn for _, _, enc_fn in enclosing_chain
        ] + ([imm_scope] if isinstance(imm_scope, ast.ClassDef) else [])
        intra_nl_mutators = _collect_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            mod_tree=ctx.mod_tree,
            enclosing_scopes=unit_enclosing_scopes,
        )
        is_suspended = ctx.is_subroutine and (
            ctx.has_yield
            or (
                ctx.is_async
                and _node_is_effectively_async(
                    (
                        ctx.candidate_stmts[0].body
                        if len(ctx.candidate_stmts) == 1
                        and isinstance(
                            ctx.candidate_stmts[0],
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        )
                        else ctx.candidate_stmts
                    ),
                    True,
                )
            )
        )
        if is_suspended and bool(unit_mutated_names & (enc_escaping | nonlocal_set)):
            _append_global_or_cell_hazard(reasons, unit_mutated_names, enc_outer_globals)
            return _ClosureHazardResult(True, tuple(reasons))

        if ctx.is_subroutine and _extract_direct_nonlocals(unit_mod):
            _append_hazard_reason(reasons, "cell_severing")
            return _ClosureHazardResult(True, tuple(reasons))
        unit_closures = _collect_scope_closures_uncached(
            unit_mod,
            module_tree=ctx.mod_tree,
            genexp_include_header=False,
            cache_parent_map=False,
            enclosing_scopes=unit_enclosing_scopes,
        )
    unit_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures)) if unit_closures else set()
    )
    if ctx.is_subroutine:
        if is_suspended:
            unit_closure_reads.update(ctx.inputs)
    elif ctx.unit_kind == "closure":
        unit_closure_reads.update(ctx.free_vars)

    intra_closure_reads: Set[str] = (
        set(unit_closure_reads)
        if ctx.is_subroutine
        else (set(unit_closure_reads) | nonlocal_set)
    )

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    if unit_stores_affect_enclosing and bool(eff_input_set & enc_outer_globals):
        shadowed_builtin_names: Set[str] = (
            set(eff_input_set)
            | store_set
            | set(ctx.deleted_names)
            | set(ctx.unit_top_locals)
            | set(enc_escaping)
        )
        if ctx.mod_tree is not None:
            shadowed_builtin_names.update(
                _extract_single_node_local_names(ctx.mod_tree)
            )
        for enc_scope_node in unit_enclosing_scopes:
            shadowed_builtin_names.update(
                _extract_single_node_local_names(enc_scope_node)
            )
        has_star = _has_wildcard_star_import(
            unit_mod,
            module_tree=ctx.mod_tree,
            cache_scope_node=False,
            enclosing_scopes=unit_enclosing_scopes,
        )
        if _unit_has_global_mutating_call(
            ctx.candidate_stmts,
            shadowed_builtin_names,
            has_star_import=has_star,
            allow_heuristic_builtins=not ctx.strict_with_suppression,
        ):
            _append_hazard_reason(reasons, "global_escape")
            return _ClosureHazardResult(True, tuple(reasons))
    if unit_stores_affect_enclosing and bool(
        unit_closure_reads
        & (
            (unit_mutated_names & (enc_escaping | nonlocal_set))
            | (enc_outer_globals & eff_input_set)
        )
    ):
        _append_global_or_cell_hazard(reasons, unit_closure_reads, enc_outer_globals)
        return _ClosureHazardResult(True, tuple(reasons))
    unit_mutations = (
        (escaping_unit_mutations | (unit_mutated_names & unit_closure_reads))
        if unit_stores_affect_enclosing
        else set()
    )

    if (
        not intra_closure_reads
        and not intra_nl_mutators
        and not ctx.is_subroutine
        and not unit_mutations
    ):
        return _ClosureHazardResult(False, ())

    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        u_start_col,
        u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    if (
        ctx.definite_stores is not None
        and ctx.is_subroutine
        and not ctx.has_yield
        and chain_infos
    ):
        unseeded_cond_stores = (store_set - set(ctx.definite_stores)) - set(ctx.inputs)
        if unseeded_cond_stores:
            imm_info = chain_infos[-1]
            imm_sib_reads: Set[str] = (
                set().union(*(s_r for _, s_r, _ in imm_info.sibling_closures))
                if imm_info.sibling_closures
                else set()
            )
            eff_sib_reads = (
                imm_sib_reads & imm_info.post_reads
                if ctx.skip_pre_unit_closures
                else imm_sib_reads
            )
            cond_pre_post = (imm_info.pre_mutated - imm_info.pre_bound) & (
                imm_info.post_reads | unit_closure_reads
            )
            cond_exported_closure_capture = (
                output_set & set(imm_info.post_reads) & unit_closure_reads
            )
            unexported_pre_post = (
                imm_info.pre_mutated & imm_info.post_reads
            ) - output_set
            if bool(
                unseeded_cond_stores
                & imm_info.fn_locals
                & (
                    eff_sib_reads
                    | cond_pre_post
                    | cond_exported_closure_capture
                    | unexported_pre_post
                )
            ):
                _append_hazard_reason(reasons, "cell_severing")
                return _ClosureHazardResult(True, tuple(reasons))

    captured_candidates = intra_closure_reads & (
        eff_input_set
        | ((store_set | output_set) if unit_stores_affect_enclosing else set())
    )
    sub_mutated = helper_rebound_vars | immediate_cell_mutations
    has_cell_hazard = (
        _check_intra_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            intra_nl_mutators,
            ctx.unit_top_locals,
            nonlocal_set,
            chain_infos,
            is_subroutine=ctx.is_subroutine,
            is_suspended=is_suspended,
            eff_input_set=eff_input_set,
            enc_escaping=enc_escaping,
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
            helper_rebound_vars=helper_rebound_vars,
            is_suspended=(is_suspended or is_expr_unit)
            and not ctx.skip_pre_unit_closures,
            unreturned_mutations=is_expr_unit,
            skip_pre_unit_closures=ctx.skip_pre_unit_closures,
        )
    )
    if has_cell_hazard:
        _append_hazard_reason(reasons, "cell_severing")
        return _ClosureHazardResult(True, tuple(reasons))
    return _ClosureHazardResult(False, ())


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings."""
    return _evaluate_severed_closure_cell_hazard(ctx).has_hazard
