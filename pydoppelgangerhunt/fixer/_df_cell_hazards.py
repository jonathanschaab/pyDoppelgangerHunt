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
4. `_df_with_rules`, `_df_provenance_walk`, and `_df_block_flow`: Typed
   context-manager suppression provenance (`_SuppressProvenance`), forward
   provenance walking, and the unified definite/conditional binding, deletion,
   and block-termination engine (`_analyze_block_assignment`,
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

from pydoppelgangerhunt._ast_compat import (
    is_match_case_node,
    is_match_class_node,
    is_match_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _ephemeral_unit_ast_context,
    _mark_ephemeral_unit_ast,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _IntraUnitMutator,
    _collect_escaped_and_executed_intra_unit_defs,
    _collect_unit_nonlocal_mutators,
    _propagate_sibling_closures,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import (
    _EnclosingFnCellInfo,
    _extract_root_target_names,
)
from pydoppelgangerhunt.fixer._df_cell_traversal import (
    _build_enclosing_chain_cell_infos,
)
from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_import_bound_names,
    _extract_type_alias_or_pattern_name,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _collect_scope_closures_uncached,
    _extract_chain_nonlocals_and_outer_globals,
    _extract_comprehension_walrus_names,
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
from pydoppelgangerhunt.fixer._df_visitors import _MAX_AST_RECURSION_DEPTH
from pydoppelgangerhunt.fixer._df_suppress_prov import _SAFE_EXCEPTION_TYPE_NAMES
from pydoppelgangerhunt.fixer._df_with_rules import _is_non_raising_store_target
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
    unshadowed_mut_edges: List[Tuple[str, AbstractSet[str]]] = []
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
            eff_s_name = (
                None if (s_name is not None and s_name in inner_shadowed) else s_name
            )
            unshadowed_sibs.append(
                (eff_s_name, s_reads - inner_shadowed, s_muts - inner_shadowed)
            )
        for t_name, src_names in info.alias_edges:
            if t_name in inner_shadowed:
                continue
            unshadowed_edges.append((t_name, src_names - inner_shadowed))
        for t_name, src_names in info.mutation_edges:
            if t_name in inner_shadowed:
                continue
            unshadowed_mut_edges.append((t_name, src_names - inner_shadowed))
    sib_closures = _propagate_sibling_closures(
        unshadowed_sibs,
        alias_edges=unshadowed_edges,
        mutation_edges=unshadowed_mut_edges,
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
    is_lenient: bool = False,
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
        if not is_ref_sib and (not unreturned_mutations or is_lenient):
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
    closure_strictness: str = "strict"
    definite_stores: Optional[AbstractSet[str]] = None
    cols_resolved: bool = False
    enclosing_chain: Optional[Sequence[Tuple[int, int, ast.AST]]] = None
    lex_validated: bool = False
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset()

    @property
    def is_lenient(self) -> bool:
        """Returns True when lenient closure strictness is active."""
        return self.closure_strictness == "lenient"

    @property
    def is_pedantic(self) -> bool:
        """Returns True when pedantic closure strictness is active."""
        return self.closure_strictness == "pedantic"


class _ClosureHazardResult(NamedTuple):
    """Outcome of closure cell hazard evaluation with structured diagnostic reasons."""

    has_hazard: bool
    reasons: Tuple[str, ...] = ()


class _ResolvedHazardAnchor(NamedTuple):
    """Validated unit column bounds and enclosing function chain for cell hazard analysis."""

    u_start_col: Optional[int]
    u_end_col: Optional[int]
    enclosing_chain: List[Tuple[int, int, ast.AST]]


class _UnitMutationSets(NamedTuple):
    """Partitioned store, output, nonlocal, and escaping mutation sets for a candidate unit."""

    is_expr_unit: bool
    unit_stores_affect_enclosing: bool
    store_set: Set[str]
    output_set: Set[str]
    nonlocal_set: Set[str]
    unit_mutated_names: Set[str]
    explicit_nonlocal_mutations: Set[str]
    helper_rebound_vars: Set[str]
    escaping_unit_mutations: Set[str]


class _EphemeralClosureFacts(NamedTuple):
    """Intermediate facts computed inside the ephemeral unit AST context."""

    enc_outer_globals: AbstractSet[str]
    enc_escaping: AbstractSet[str]
    unit_enclosing_scopes: List[ast.AST]
    intra_nl_mutators: Sequence[Union[_IntraUnitMutator, Tuple[str, Set[str], bool]]]
    is_suspended: bool
    base_closure_reads: Set[str]


class _UnitClosureHazardFacts(NamedTuple):
    """Aggregated unit mutation, closure read, and escaping context facts."""

    mut_sets: _UnitMutationSets
    enc_outer_globals: AbstractSet[str]
    enc_escaping: AbstractSet[str]
    unit_enclosing_scopes: List[ast.AST]
    intra_nl_mutators: Sequence[Union[_IntraUnitMutator, Tuple[str, Set[str], bool]]]
    is_suspended: bool
    unit_closure_reads: Set[str]
    intra_closure_reads: Set[str]
    eff_input_set: Set[str]


def _single_hazard_result(reason: str) -> _ClosureHazardResult:
    """Build a positive hazard result with a single normalized reason."""
    reasons: List[str] = []
    _append_hazard_reason(reasons, reason)
    return _ClosureHazardResult(True, tuple(reasons))


def _global_or_cell_hazard_result(
    names: AbstractSet[str],
    enc_outer_globals: AbstractSet[str],
) -> _ClosureHazardResult:
    """Build a positive hazard result classifying global escapes vs cell severing."""
    reasons: List[str] = []
    _append_global_or_cell_hazard(reasons, names, enc_outer_globals)
    return _ClosureHazardResult(True, tuple(reasons))


def _unit_is_suspended_async_or_generator(ctx: _UnitClosureHazardContext) -> bool:
    """Return True when extracting a subroutine suspends execution via yield or await."""
    if not ctx.is_subroutine:
        return False
    if ctx.has_yield:
        return True
    if not ctx.is_async:
        return False
    target: Union[Sequence[ast.stmt], ast.AST] = (
        ctx.candidate_stmts[0].body
        if len(ctx.candidate_stmts) == 1
        and isinstance(ctx.candidate_stmts[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        else ctx.candidate_stmts
    )
    return _node_is_effectively_async(target, True)


def _resolve_unit_hazard_anchor(
    ctx: _UnitClosureHazardContext,
) -> Union[_ClosureHazardResult, _ResolvedHazardAnchor]:
    """Resolve unit column bounds, enclosing chain, and span/lexical validity."""
    if not ctx.is_subroutine and ctx.unit_kind not in (
        "closure",
        "comprehension",
        "complex_expr",
    ):
        return _ClosureHazardResult(False, ())
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return _single_hazard_result("unresolved_span_bounds")
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
        return _single_hazard_result("invalid_lexical_declaration")
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
        return (
            _single_hazard_result("unresolved_span_bounds")
            if has_unresolved
            else _ClosureHazardResult(False, ())
        )
    if _has_unresolved_span_bounds(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_start_col,
        u_end_col,
        ctx.candidate_stmts,
    ):
        return _single_hazard_result("unresolved_span_bounds")
    return _ResolvedHazardAnchor(u_start_col, u_end_col, enclosing_chain)


def _extract_unit_mutation_sets(
    ctx: _UnitClosureHazardContext,
    unit_mod: ast.Module,
) -> _UnitMutationSets:
    """Partition unit stores, outputs, nonlocals, and escaping mutation names."""
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
    immediate_cell_mutations: Set[str] = (
        (_extract_comprehension_walrus_names(unit_mod) | explicit_nonlocal_mutations)
        if is_expr_unit
        else explicit_nonlocal_mutations
    )
    escaping_unit_mutations = helper_rebound_vars | immediate_cell_mutations
    return _UnitMutationSets(
        is_expr_unit=is_expr_unit,
        unit_stores_affect_enclosing=unit_stores_affect_enclosing,
        store_set=store_set,
        output_set=output_set,
        nonlocal_set=nonlocal_set,
        unit_mutated_names=unit_mutated_names,
        explicit_nonlocal_mutations=explicit_nonlocal_mutations,
        helper_rebound_vars=helper_rebound_vars,
        escaping_unit_mutations=escaping_unit_mutations,
    )


def _collect_ephemeral_closure_facts(
    ctx: _UnitClosureHazardContext,
    anchor: _ResolvedHazardAnchor,
    unit_mod: ast.Module,
    mut: _UnitMutationSets,
) -> Union[_ClosureHazardResult, _EphemeralClosureFacts]:
    """Collect enclosing escape/closure facts inside `_ephemeral_unit_ast_context`."""
    innermost_stmt_fn, enc_nonlocals, enc_outer_globals = (
        _extract_chain_nonlocals_and_outer_globals(
            anchor.enclosing_chain,
            ctx.u_start,
            ctx.u_end,
            u_start_col=anchor.u_start_col,
            u_end_col=anchor.u_end_col,
        )
    )
    if _has_escaped_intra_unit_global_declarations(
        innermost_stmt_fn,
        ctx.u_start,
        ctx.u_end,
        u_start_col=anchor.u_start_col,
        u_end_col=anchor.u_end_col,
        is_subroutine=ctx.is_subroutine,
    ):
        return _single_hazard_result("global_escape")
    enc_escaping = enc_nonlocals | enc_outer_globals
    if mut.unit_stores_affect_enclosing and bool(
        mut.escaping_unit_mutations
        & (enc_outer_globals if ctx.is_subroutine else enc_escaping)
    ):
        return _global_or_cell_hazard_result(
            mut.escaping_unit_mutations, enc_outer_globals
        )
    imm_scope = _resolve_immediate_scope_in_fn(
        innermost_stmt_fn,
        ctx.u_start,
        ctx.u_end,
        u_start_col=anchor.u_start_col,
        u_end_col=anchor.u_end_col,
    )
    unit_enclosing_scopes: List[ast.AST] = [
        enc_fn for _, _, enc_fn in anchor.enclosing_chain
    ] + ([imm_scope] if isinstance(imm_scope, ast.ClassDef) else [])
    intra_nl_mutators = _collect_unit_nonlocal_mutators(
        ctx.candidate_stmts,
        mod_tree=ctx.mod_tree,
        enclosing_scopes=unit_enclosing_scopes,
    )
    is_suspended = _unit_is_suspended_async_or_generator(ctx)
    if is_suspended and bool(
        mut.unit_mutated_names & (enc_escaping | mut.nonlocal_set)
    ):
        return _global_or_cell_hazard_result(
            mut.unit_mutated_names, enc_outer_globals
        )
    if ctx.is_subroutine and _extract_direct_nonlocals(unit_mod):
        return _single_hazard_result("cell_severing")
    unit_closures = _collect_scope_closures_uncached(
        unit_mod,
        module_tree=ctx.mod_tree,
        genexp_include_header=False,
        cache_parent_map=False,
        enclosing_scopes=unit_enclosing_scopes,
    )
    base_closure_reads: Set[str] = (
        set().union(*(reads for _, reads in unit_closures)) if unit_closures else set()
    )
    return _EphemeralClosureFacts(
        enc_outer_globals=enc_outer_globals,
        enc_escaping=enc_escaping,
        unit_enclosing_scopes=unit_enclosing_scopes,
        intra_nl_mutators=intra_nl_mutators,
        is_suspended=is_suspended,
        base_closure_reads=base_closure_reads,
    )


def _check_global_and_closure_read_escapes(
    ctx: _UnitClosureHazardContext,
    unit_mod: ast.Module,
    facts: _UnitClosureHazardFacts,
) -> Optional[_ClosureHazardResult]:
    """Check global mutating calls and closure reads over escaping/mutated outer names."""
    mut = facts.mut_sets
    eff_input_set = facts.eff_input_set
    if mut.unit_stores_affect_enclosing and bool(
        eff_input_set & facts.enc_outer_globals
    ):
        shadowed: Set[str] = (
            set(eff_input_set)
            | mut.unit_mutated_names
            | set(ctx.unit_top_locals)
            | set(facts.enc_escaping)
        )
        if ctx.mod_tree is not None:
            shadowed.update(_extract_single_node_local_names(ctx.mod_tree))
        for enc_scope_node in facts.unit_enclosing_scopes:
            shadowed.update(_extract_single_node_local_names(enc_scope_node))
        has_star = _has_wildcard_star_import(
            unit_mod,
            module_tree=ctx.mod_tree,
            cache_scope_node=False,
            enclosing_scopes=facts.unit_enclosing_scopes,
        )
        if _unit_has_global_mutating_call(
            ctx.candidate_stmts,
            shadowed,
            has_star_import=has_star,
            allow_heuristic_builtins=not ctx.is_pedantic,
        ):
            return _single_hazard_result("global_escape")
    if mut.unit_stores_affect_enclosing and bool(
        facts.unit_closure_reads
        & (
            (mut.unit_mutated_names & (facts.enc_escaping | mut.nonlocal_set))
            | (facts.enc_outer_globals & eff_input_set)
        )
    ):
        return _global_or_cell_hazard_result(
            facts.unit_closure_reads, facts.enc_outer_globals
        )
    return None


def _collect_unit_closure_hazard_facts(
    ctx: _UnitClosureHazardContext,
    anchor: _ResolvedHazardAnchor,
    unit_mod: ast.Module,
) -> Union[_ClosureHazardResult, _UnitClosureHazardFacts]:
    """Compute unit closure hazard facts and evaluate immediate outer-escape checks."""
    with _ephemeral_unit_ast_context(unit_mod):
        mut = _extract_unit_mutation_sets(ctx, unit_mod)
        if not mut.unit_stores_affect_enclosing and bool(
            mut.explicit_nonlocal_mutations
        ):
            return _single_hazard_result("cell_severing")
        eph_or_result = _collect_ephemeral_closure_facts(ctx, anchor, unit_mod, mut)
        if isinstance(eph_or_result, _ClosureHazardResult):
            return eph_or_result
    unit_closure_reads = set(eph_or_result.base_closure_reads)
    if ctx.is_subroutine:
        if eph_or_result.is_suspended:
            unit_closure_reads.update(ctx.inputs)
    elif ctx.unit_kind == "closure":
        unit_closure_reads.update(ctx.free_vars)
    intra_closure_reads: Set[str] = (
        set(unit_closure_reads)
        if ctx.is_subroutine
        else (set(unit_closure_reads) | mut.nonlocal_set)
    )
    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | mut.nonlocal_set
    facts = _UnitClosureHazardFacts(
        mut_sets=mut,
        enc_outer_globals=eph_or_result.enc_outer_globals,
        enc_escaping=eph_or_result.enc_escaping,
        unit_enclosing_scopes=eph_or_result.unit_enclosing_scopes,
        intra_nl_mutators=eph_or_result.intra_nl_mutators,
        is_suspended=eph_or_result.is_suspended,
        unit_closure_reads=unit_closure_reads,
        intra_closure_reads=intra_closure_reads,
        eff_input_set=eff_input_set,
    )
    escape_result = _check_global_and_closure_read_escapes(ctx, unit_mod, facts)
    if escape_result is not None:
        return escape_result
    return facts


def _collect_returned_expr_containers(ret_node: ast.Return) -> Set[int]:
    """Returns ids of ret_node.value and any non-starred literal containers wrapping it."""
    allowed_ids: Set[int] = set()
    if ret_node.value is None:
        return allowed_ids
    stack: List[ast.AST] = [ret_node.value]
    while stack:
        curr = stack.pop()
        allowed_ids.add(id(curr))
        if isinstance(curr, (ast.Tuple, ast.List, ast.Set)):
            stack.extend(elt for elt in curr.elts if not isinstance(elt, ast.Starred))
        elif isinstance(curr, ast.Dict):
            for key, val in zip(curr.keys, curr.values):
                if key is not None:
                    stack.append(key)
                    stack.append(val)
    return allowed_ids


def _add_deferred_genexp_subnodes(
    gen: ast.GeneratorExp,
    deferred_ids: Set[int],
) -> None:
    """Marks GeneratorExp body and deferred clauses while keeping generators[0].iter eager."""
    deferred_roots: List[ast.AST] = [gen.elt]
    if gen.generators:
        first_comp = gen.generators[0]
        deferred_roots.append(first_comp.target)
        deferred_roots.extend(first_comp.ifs)
        deferred_roots.extend(gen.generators[1:])
    for root in deferred_roots:
        for sub in ast.walk(root):
            deferred_ids.add(id(sub))


def _collect_unconsumed_genexp_deferred_node_ids(
    candidate_stmts: Sequence[ast.stmt],
) -> Set[int]:
    """Collects node ids inside GeneratorExps provably unconsumed in candidate_stmts."""
    parent_map: Dict[int, ast.AST] = {}
    returned_slot_ids: Set[int] = set()
    genexps: List[ast.GeneratorExp] = []
    stack: List[ast.AST] = list(reversed(candidate_stmts))
    while stack:
        curr = stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        if isinstance(curr, ast.Return):
            returned_slot_ids.update(_collect_returned_expr_containers(curr))
        elif isinstance(curr, ast.GeneratorExp):
            genexps.append(curr)
        children = list(ast.iter_child_nodes(curr))
        for child in children:
            parent_map[id(child)] = curr
        stack.extend(reversed(children))
    if not genexps:
        return set()
    name_nodes: Dict[str, List[ast.Name]] = {}
    blocked_names: Set[str] = set()
    for stmt in candidate_stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                name_nodes.setdefault(node.id, []).append(node)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                blocked_names.update(node.names)
            else:
                blocked_names.update(_extract_non_assign_bound_names(node))
    deferred_ids: Set[int] = set()
    for gen in genexps:
        if id(gen) in returned_slot_ids:
            _add_deferred_genexp_subnodes(gen, deferred_ids)
            continue
        parent = parent_map.get(id(gen))
        target: Optional[ast.AST] = None
        if (
            isinstance(parent, ast.Assign)
            and len(parent.targets) == 1
            and parent.value is gen
        ):
            target = parent.targets[0]
        elif (
            isinstance(parent, ast.AnnAssign)
            and parent.simple
            and parent.value is gen
        ):
            target = parent.target
        if not isinstance(target, ast.Name) or target.id in blocked_names:
            continue
        occ = name_nodes.get(target.id, ())
        if len(occ) >= 2 and all(
            id(n) == id(target)
            or (isinstance(n.ctx, ast.Load) and id(n) in returned_slot_ids)
            for n in occ
        ):
            _add_deferred_genexp_subnodes(gen, deferred_ids)
    return deferred_ids


def _has_unseeded_conditional_store_hazard(
    ctx: _UnitClosureHazardContext,
    mut: _UnitMutationSets,
    unit_closure_reads: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
) -> bool:
    """Return True if a conditionally assigned unit store reaches enclosing readers/closures."""
    if (
        ctx.definite_stores is None
        or not ctx.is_subroutine
        or ctx.has_yield
        or not chain_infos
    ):
        return False
    unseeded_cond_stores = (mut.store_set - set(ctx.definite_stores)) - set(ctx.inputs)
    if not unseeded_cond_stores:
        return False
    imm_info = chain_infos[-1]
    imm_sib_reads: Set[str] = (
        set().union(*(s_r for _, s_r, _ in imm_info.sibling_closures))
        if imm_info.sibling_closures
        else set()
    )
    eff_sib_reads = (
        imm_sib_reads & imm_info.post_reads
        if ctx.is_lenient
        else imm_sib_reads
    )
    cond_pre_post = (imm_info.pre_mutated - imm_info.pre_bound) & (
        imm_info.post_reads | unit_closure_reads
    )
    cond_exported_closure_capture = (
        mut.output_set & set(imm_info.post_reads) & unit_closure_reads
    )
    unexported_pre_post = (
        imm_info.pre_mutated & imm_info.post_reads
    ) - mut.output_set
    return bool(
        unseeded_cond_stores
        & imm_info.fn_locals
        & (
            eff_sib_reads
            | cond_pre_post
            | cond_exported_closure_capture
            | unexported_pre_post
        )
    )


# INVARIANT (CONVERGENCE BOUNDARY — CLOSED HEURISTIC TABLE):
# `_PROVEN_PURE_GLOBAL_SAFE_BUILTINS` and `_HEURISTIC_GLOBAL_SAFE_BUILTINS` are
# frozen, closed tables. Do NOT expand these tables incrementally for arbitrary
# user-defined callables or additional stdlib functions; callers requiring
# stricter purity proofs inside global-reading units must use
# `closure_strictness = "pedantic"`.
#
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
    ast.Import,
    ast.ImportFrom,
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
    if is_match_case_node(node):
        guard = getattr(node, "guard", None)
        if isinstance(guard, ast.AST):
            return (guard,)
    return ()


class _PedanticPrimitiveEnv(NamedTuple):
    """Shadowing context for closed single-pass syntactic pedantic dunder checks."""

    effective_shadowed: frozenset[str] = frozenset()
    all_builtins_shadowed: bool = False


def _check_proven_sync_iter(
    target: ast.AST,
    iter_expr: ast.AST,
) -> bool:
    """Returns True when iterating a literal string/bytes or unstarred tuple/list avoids user dunders."""
    while isinstance(iter_expr, ast.NamedExpr):
        iter_expr = iter_expr.value
    if isinstance(iter_expr, ast.Constant) and isinstance(iter_expr.value, (str, bytes)):
        return isinstance(target, ast.Name)
    if isinstance(iter_expr, (ast.Tuple, ast.List)) and not any(
        isinstance(e, ast.Starred) for e in iter_expr.elts
    ):
        return all(_is_non_raising_store_target(target, elt) for elt in iter_expr.elts)
    return False


def _has_proven_subscript_slice_or_index(
    slc: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    allow_slice: bool,
    _depth: int,
) -> bool:
    """Return True when subscript index or slice bounds evaluate strictly to syntactic primitives."""
    if isinstance(slc, ast.Slice):
        return allow_slice and all(
            bound is None
            or _is_proven_primitive_dunder_expr(bound, env, _depth=_depth + 1)
            for bound in (slc.lower, slc.upper, slc.step)
        )
    return _is_proven_primitive_dunder_expr(slc, env, _depth=_depth + 1)


def _check_subscript_load_safety(
    node: ast.Subscript,
    env: _PedanticPrimitiveEnv,
    *,
    require_primitive_result: bool,
    _depth: int = 0,
) -> bool:
    """Validate Load subscript literal container and bounds, optionally requiring primitive elements."""
    if _depth > _MAX_AST_RECURSION_DEPTH or not isinstance(node.ctx, ast.Load):
        return False
    val: ast.AST = node.value
    while isinstance(val, ast.NamedExpr):
        val = val.value
    is_str_or_bytes = isinstance(val, ast.Constant) and isinstance(
        val.value, (str, bytes)
    )
    if not _has_proven_subscript_slice_or_index(
        node.slice,
        env,
        allow_slice=is_str_or_bytes or not require_primitive_result,
        _depth=_depth,
    ):
        return False
    if is_str_or_bytes:
        return True
    if isinstance(val, (ast.Tuple, ast.List)) and not any(
        isinstance(elt, ast.Starred) for elt in val.elts
    ):
        return not require_primitive_result or all(
            _is_proven_primitive_dunder_expr(elt, env, _depth=_depth + 1)
            for elt in val.elts
        )
    return False


def _is_proven_primitive_subscript_expr(
    expr: ast.Subscript,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when a Load subscript avoids user dunders and yields a literal scalar primitive."""
    return _check_subscript_load_safety(
        expr, env, require_primitive_result=True, _depth=_depth
    )


# INVARIANT (CONVERGENCE BOUNDARY — SYNTACTIC PEDANTIC DUNDER CHECK):
# Proving arbitrary untyped Python expressions cannot invoke user-defined
# dunder hooks (``__add__``, ``__bool__``, ``__iter__``, ``__hash__``, etc.)
# is undecidable. This pedantic dunder check is a CLOSED single-pass syntactic
# whitelist covering ONLY literals (``ast.Constant``), unshadowed
# ``_PROVEN_PURE_GLOBAL_SAFE_BUILTINS`` (``callable``, ``id``, ``ord``),
# identity comparisons (``is`` / ``is not``), literal containers, and pure
# non-dunder compositions over them. Do NOT re-add fixed-point local-variable
# or tuple type inference; untyped variable operands MUST fail closed in
# ``pedantic`` mode (callers wanting permissive local-operand rules use
# ``closure_strictness = "strict"``).
def _is_proven_primitive_dunder_expr(
    expr: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when expr syntactically evaluates to a built-in scalar primitive without user dunders."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.NamedExpr):
        return _is_proven_primitive_dunder_expr(expr.value, env, _depth=_depth + 1)
    if isinstance(expr, ast.Subscript):
        return _is_proven_primitive_subscript_expr(expr, env, _depth=_depth + 1)
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.Not):
            return _is_proven_primitive_bool_expr(expr.operand, env, _depth=_depth + 1)
        return _is_proven_primitive_dunder_expr(expr.operand, env, _depth=_depth + 1)
    if isinstance(expr, ast.BinOp) and not isinstance(expr.op, ast.MatMult):
        return _is_proven_primitive_dunder_expr(
            expr.left, env, _depth=_depth + 1
        ) and _is_proven_primitive_dunder_expr(expr.right, env, _depth=_depth + 1)
    if isinstance(expr, ast.BoolOp):
        return all(
            _is_proven_primitive_dunder_expr(val, env, _depth=_depth + 1)
            for val in expr.values
        )
    if isinstance(expr, ast.IfExp):
        return (
            _is_proven_primitive_bool_expr(expr.test, env, _depth=_depth + 1)
            and _is_proven_primitive_dunder_expr(expr.body, env, _depth=_depth + 1)
            and _is_proven_primitive_dunder_expr(expr.orelse, env, _depth=_depth + 1)
        )
    if isinstance(expr, ast.Compare):
        return _is_proven_safe_compare(expr, env, _depth=_depth + 1)
    if isinstance(expr, ast.JoinedStr):
        return all(
            isinstance(part, ast.Constant)
            or (
                isinstance(part, ast.FormattedValue)
                and _is_proven_safe_formatted_value(part, env, _depth=_depth + 1)
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
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when an f-string FormattedValue formats only a syntactically proven primitive."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    return _is_proven_primitive_dunder_expr(node.value, env, _depth=_depth + 1) and (
        node.format_spec is None
        or _is_proven_primitive_dunder_expr(node.format_spec, env, _depth=_depth + 1)
    )


def _is_proven_primitive_container_for_in(
    rhs: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when rhs of 'in'/'not in' is a literal built-in container of proven primitives."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    while isinstance(rhs, ast.NamedExpr):
        rhs = rhs.value
    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (str, bytes)):
        return True
    if isinstance(rhs, (ast.Tuple, ast.List, ast.Set)):
        return not any(isinstance(elt, ast.Starred) for elt in rhs.elts) and all(
            _is_proven_primitive_dunder_expr(elt, env, _depth=_depth + 1)
            for elt in rhs.elts
        )
    if isinstance(rhs, ast.Dict):
        return all(
            key is not None
            and _is_proven_primitive_dunder_expr(key, env, _depth=_depth + 1)
            for key in rhs.keys
        )
    return False


def _is_proven_safe_compare(
    node: ast.Compare,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when every step of a Compare expression cannot invoke user comparison/containment dunders."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    operands = (node.left, *node.comparators)
    for lhs, op, rhs in zip(operands, node.ops, node.comparators):
        if isinstance(op, (ast.Is, ast.IsNot)):
            continue
        if isinstance(op, (ast.In, ast.NotIn)):
            if not (
                _is_proven_primitive_dunder_expr(lhs, env, _depth=_depth + 1)
                and _is_proven_primitive_container_for_in(rhs, env, _depth=_depth + 1)
            ):
                return False
        elif not (
            _is_proven_primitive_dunder_expr(lhs, env, _depth=_depth + 1)
            and _is_proven_primitive_dunder_expr(rhs, env, _depth=_depth + 1)
        ):
            return False
    return True


def _is_proven_safe_subscript_load(
    node: ast.Subscript,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when a Load subscript indexes a literal sequence/string with primitive bounds."""
    return _check_subscript_load_safety(
        node, env, require_primitive_result=False, _depth=_depth
    )


def _is_proven_primitive_bool_expr(
    expr: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when evaluating expr in a boolean context cannot invoke user-defined __bool__ or __len__."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(expr, _STATIC_NON_DESCRIPTOR_VALUE_TYPES):
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
            _is_proven_primitive_bool_expr(sub_e, env, _depth=_depth + 1)
            for sub_e in sub_bool_exprs
        )
    return _is_proven_primitive_dunder_expr(expr, env, _depth=_depth + 1)


def _is_safe_pedantic_except_type(
    exc_type: ast.AST,
    env: _PedanticPrimitiveEnv,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True when an ExceptHandler.type only references unshadowed builtin exception types."""
    if env.all_builtins_shadowed or _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(exc_type, ast.Name):
        return (
            exc_type.id in _SAFE_EXCEPTION_TYPE_NAMES
            and exc_type.id not in env.effective_shadowed
        )
    if isinstance(exc_type, ast.Tuple):
        return all(
            not isinstance(elt, ast.Starred)
            and _is_safe_pedantic_except_type(elt, env, _depth=_depth + 1)
            for elt in exc_type.elts
        )
    return False


def _node_invokes_pedantic_dunder_protocol(
    node: ast.AST,
    env: _PedanticPrimitiveEnv,
) -> bool:
    """Returns True when an immediate AST node may invoke user-defined dunder hooks in pedantic mode."""
    if any(
        not _is_proven_primitive_bool_expr(b_expr, env)
        for b_expr in _extract_immediate_bool_test_exprs(node)
    ):
        return True
    if isinstance(node, ast.Attribute):
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
        return not _check_proven_sync_iter(node.target, node.iter)
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
                is_match_class_node(sub)
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
    if isinstance(node, ast.ExceptHandler) and node.type is not None:
        return not _is_safe_pedantic_except_type(node.type, env)
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
        return set(_extract_import_bound_names(node))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    pattern_or_alias_name = _extract_type_alias_or_pattern_name(node)
    if pattern_or_alias_name is not None:
        return {pattern_or_alias_name}
    return set()


def _record_immediate_expr_names(nodes: Sequence[ast.AST], seen: Set[str]) -> None:
    """Adds all immediate ast.Name identifiers under nodes into seen."""
    for sub in _iter_immediate_unit_exec_nodes(nodes):
        if isinstance(sub, ast.Name):
            seen.add(sub.id)


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
            _record_immediate_expr_names((stmt.annotation,), seen_before_init)
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
    context-manager protocol hook (``with`` / ``async with``), import statement
    (``import`` / ``from ... import``), coroutine/generator
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
    implicit dunder-invoking AST operations — including binary/unary operators,
    non-identity comparisons and membership tests, attribute/subscript reads,
    synchronous ``for`` and comprehension iteration, unpacking, f-string
    formatting, ``set``/``dict`` hashing, ``match`` class/dunder patterns,
    non-builtin ``raise``/``except`` handlers, and implicit boolean truthiness —
    are rejected unless their operands are syntactically literal constants or
    literal containers (no untyped local-variable type inference).
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
    deferred_genexp_ids = _collect_unconsumed_genexp_deferred_node_ids(candidate_stmts)
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
    pedantic_env = _PedanticPrimitiveEnv(
        effective_shadowed=frozenset(effective_shadowed),
        all_builtins_shadowed=all_builtins_shadowed,
    )
    for node in immediate_nodes:
        if id(node) in deferred_genexp_ids:
            continue
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
            node, pedantic_env
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
        When ``ctx.is_lenient`` is True (lenient mode), speculative checks for
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
    anchor_or_result = _resolve_unit_hazard_anchor(ctx)
    if isinstance(anchor_or_result, _ClosureHazardResult):
        return anchor_or_result
    unit_mod = ast.Module(body=list(ctx.candidate_stmts), type_ignores=[])
    _mark_ephemeral_unit_ast(unit_mod)
    facts_or_result = _collect_unit_closure_hazard_facts(
        ctx, anchor_or_result, unit_mod
    )
    if isinstance(facts_or_result, _ClosureHazardResult):
        return facts_or_result
    mut = facts_or_result.mut_sets
    unit_mutations = (
        (
            mut.escaping_unit_mutations
            | (mut.unit_mutated_names & facts_or_result.unit_closure_reads)
        )
        if mut.unit_stores_affect_enclosing
        else set()
    )
    if (
        not facts_or_result.intra_closure_reads
        and not facts_or_result.intra_nl_mutators
        and not ctx.is_subroutine
        and not unit_mutations
    ):
        return _ClosureHazardResult(False, ())

    chain_infos = _build_enclosing_chain_cell_infos(
        anchor_or_result.enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        anchor_or_result.u_start_col,
        anchor_or_result.u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
        strict_with_suppression=ctx.is_pedantic,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    if _has_unseeded_conditional_store_hazard(
        ctx, mut, facts_or_result.unit_closure_reads, chain_infos
    ):
        return _single_hazard_result("cell_severing")

    captured_candidates = facts_or_result.intra_closure_reads & (
        facts_or_result.eff_input_set
        | (
            (mut.store_set | mut.output_set)
            if mut.unit_stores_affect_enclosing
            else set()
        )
    )
    has_cell_hazard = (
        _check_intra_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            facts_or_result.intra_nl_mutators,
            ctx.unit_top_locals,
            mut.nonlocal_set,
            chain_infos,
            is_subroutine=ctx.is_subroutine,
            is_suspended=facts_or_result.is_suspended,
            eff_input_set=facts_or_result.eff_input_set,
            enc_escaping=facts_or_result.enc_escaping,
        )
        or _check_captured_cell_mutations(
            captured_candidates,
            facts_or_result.eff_input_set,
            chain_infos,
            unit_closure_reads=facts_or_result.unit_closure_reads,
        )
        or _check_sibling_closure_sharing(
            facts_or_result.eff_input_set,
            mut.escaping_unit_mutations,
            chain_infos,
            helper_rebound_vars=mut.helper_rebound_vars,
            is_suspended=(facts_or_result.is_suspended or mut.is_expr_unit)
            and not ctx.is_lenient,
            unreturned_mutations=mut.is_expr_unit,
            is_lenient=ctx.is_lenient,
        )
    )
    return (
        _single_hazard_result("cell_severing")
        if has_cell_hazard
        else _ClosureHazardResult(False, ())
    )


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings."""
    return _evaluate_severed_closure_cell_hazard(ctx).has_hazard
