"""Severed closure cell hazard checks and enclosing cell analysis facade.

Design Note — Closure-Cell Hazard Architecture & Invariants
===========================================================
Extracting a code unit (subroutine block, nested closure, comprehension, or
complex expression) into a standalone module-level helper replaces lexical cell
sharing with value-based parameter passing and return-tuple unpacking. That
transformation is unsound whenever late-bound cell semantics, shared `nonlocal`
mutations, or suspended execution observe a different binding state than pass-
by-value/return-by-value.

The analysis is partitioned across three modules:
1. `_df_cell_carriers`: Lexical scope contexts, 2D span/column resolution,
   deferred comprehension/generator-expression capture analysis, closure
   carrier/factory extraction, and lexical-scope definition escape tracking.
2. `_df_cell_traversal`: Flow-sensitive AST frame traversal and definite-
   binding analysis across enclosing function chains, producing per-function
   `_EnclosingFnCellInfo` summaries.
3. `_df_cell_hazards` (this module): Numbered hazard checks 1–4 and the
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

Hazard Naming Note (`"severed_closure_cell"`)
---------------------------------------------
Outer `global` and `nonlocal` binding escapes (e.g., an intra-unit `global x`
that re-routes assignments in the enclosing function, a subroutine mutating an
outer `global` declared in the enclosing function, or a conditionally pre-bound
live-in variable in `scope.py`) are reported under the same canonical
`"severed_closure_cell"` hazard key (and `"severed closure cell"` skip reason).
Although `global` bindings are module-level rather than `PyCellObject` slots,
extracting the unit into a separate helper severs the shared lexical binding
contract with the enclosing function scope in the exact same way, and retaining
a single hazard key preserves backward compatibility for callers and reports.
"""

from __future__ import annotations

import ast
from typing import (
    AbstractSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _NON_CLOSURE_RETURN_BUILTINS,
    _EnclosingFnCellInfo,
    _FnScopeContext,
    _IntraUnitMutator,
    _SpanBounds,
    _block_contains_span,
    _can_bind_in_fn_scope,
    _collect_escaped_intra_unit_defs,
    _collect_escaped_intra_unit_names,
    _collect_unit_nonlocal_mutators,
    _extract_cached_scope_free_reads,
    _extract_carrier_value_closure_and_load_info,
    _extract_closure_carrier_mutation_targets_and_values,
    _extract_closure_carrier_targets_and_values,
    _extract_closure_nonlocal_mutations,
    _extract_closure_nonlocal_mutations_and_deletions,
    _extract_closure_nonlocal_mutations_and_deletions_uncached,
    _extract_comprehension_body_walrus_names,
    _extract_comprehension_deferred_closure_info,
    _extract_direct_nonlocals,
    _extract_expr_closure_and_load_info,
    _extract_expr_scope_free_reads,
    _extract_fn_outer_carrier_targets,
    _extract_fn_outer_carrier_targets_uncached,
    _extract_fn_returned_closure_refs,
    _extract_fn_returned_closure_refs_uncached,
    _extract_lazy_genexp_walrus_names,
    _extract_mutated_target_names,
    _extract_receiver_root_names,
    _extract_root_target_names,
    _extract_shallow_defs,
    _extract_shallow_loads_and_calls,
    _extract_store_names_in_node,
    _extract_type_alias_or_pattern_name,
    _find_enclosing_function_chain,
    _get_fn_scope_context,
    _get_fn_scope_context_uncached,
    _is_closure_factory_scope,
    _is_closure_factory_scope_uncached,
    _is_lazy_genexp,
    _is_node_inside_span,
    _is_pos_after_span,
    _is_pos_before_span,
    _iter_class_body_methods,
    _iter_comprehension_body_exprs_and_targets,
    _make_nested_fn_scope_context,
    _node_contains_or_encloses_span,
    _pick_inferred_col,
    _propagate_local_closure_syms,
    _propagate_sibling_closures,
    _resolve_unit_col_bounds,
    _stmt_dump_set,
    _strictly_encloses_span,
)
from pydoppelgangerhunt.fixer._df_cell_traversal import (
    _CellBindingState,
    _CellTraversalBuffers,
    _StackFrame,
    _StmtCellDelta,
    _analyze_enclosing_fn_cells,
    _analyze_enclosing_fn_cells_uncached,
    _apply_expr_definite_effects,
    _apply_stmt_cell_delta,
    _build_enclosing_chain_cell_infos,
    _eval_match_definite_bindings,
    _eval_try_definite_bindings,
    _extract_block_definite_bindings,
    _extract_block_definite_bindings_and_term,
    _extract_definite_target_names,
    _extract_shallow_deleted_names,
    _extract_unconditional_walrus_names,
    _find_enclosing_chain_definite_pre_bound,
    _handle_nested_scope_node,
    _is_frame_after_span,
    _is_frame_before_span,
    _make_empty_traversal_buffers,
    _maybe_push_pre_span_exhaustive_frame,
    _push_compound_children,
    _push_seq,
    _record_binding_at_pos,
    _record_carrier_closures_and_aliases,
    _record_deletion_at_pos,
    _record_simple_node_bindings_and_reads,
    _traverse_fn_cell_frames,
    _try_apply_cached_stmt_delta,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _collect_scope_closures_uncached,
    _extract_scope_nonlocals_and_outer_globals,
    _node_earliest_start_pos,
)

__all__ = [
    "_CellBindingState",
    "_CellTraversalBuffers",
    "_EnclosingFnCellInfo",
    "_FnScopeContext",
    "_IntraUnitMutator",
    "_NON_CLOSURE_RETURN_BUILTINS",
    "_SpanBounds",
    "_StackFrame",
    "_StmtCellDelta",
    "_UnitClosureHazardContext",
    "_analyze_enclosing_fn_cells",
    "_analyze_enclosing_fn_cells_uncached",
    "_apply_expr_definite_effects",
    "_apply_stmt_cell_delta",
    "_block_contains_span",
    "_build_enclosing_chain_cell_infos",
    "_can_bind_in_fn_scope",
    "_check_captured_cell_mutations",
    "_check_intra_unit_nonlocal_mutators",
    "_check_sibling_closure_sharing",
    "_collect_escaped_intra_unit_defs",
    "_collect_escaped_intra_unit_names",
    "_collect_unit_nonlocal_mutators",
    "_eval_match_definite_bindings",
    "_eval_try_definite_bindings",
    "_extract_block_definite_bindings",
    "_extract_block_definite_bindings_and_term",
    "_extract_cached_scope_free_reads",
    "_extract_carrier_value_closure_and_load_info",
    "_extract_closure_carrier_mutation_targets_and_values",
    "_extract_closure_carrier_targets_and_values",
    "_extract_closure_nonlocal_mutations",
    "_extract_closure_nonlocal_mutations_and_deletions",
    "_extract_closure_nonlocal_mutations_and_deletions_uncached",
    "_extract_comprehension_body_walrus_names",
    "_extract_comprehension_deferred_closure_info",
    "_extract_definite_target_names",
    "_extract_direct_nonlocals",
    "_extract_expr_closure_and_load_info",
    "_extract_expr_scope_free_reads",
    "_extract_fn_outer_carrier_targets",
    "_extract_fn_outer_carrier_targets_uncached",
    "_extract_fn_returned_closure_refs",
    "_extract_fn_returned_closure_refs_uncached",
    "_extract_lazy_genexp_walrus_names",
    "_extract_mutated_target_names",
    "_extract_receiver_root_names",
    "_extract_root_target_names",
    "_extract_shallow_deleted_names",
    "_extract_shallow_defs",
    "_extract_shallow_loads_and_calls",
    "_extract_store_names_in_node",
    "_extract_type_alias_or_pattern_name",
    "_extract_unconditional_walrus_names",
    "_find_enclosing_chain_definite_pre_bound",
    "_find_enclosing_function_chain",
    "_get_fn_scope_context",
    "_get_fn_scope_context_uncached",
    "_handle_nested_scope_node",
    "_has_severed_closure_cell_hazard",
    "_is_closure_factory_scope",
    "_is_closure_factory_scope_uncached",
    "_is_frame_after_span",
    "_is_frame_before_span",
    "_is_lazy_genexp",
    "_is_node_inside_span",
    "_is_pos_after_span",
    "_is_pos_before_span",
    "_iter_class_body_methods",
    "_iter_comprehension_body_exprs_and_targets",
    "_make_empty_traversal_buffers",
    "_make_nested_fn_scope_context",
    "_maybe_push_pre_span_exhaustive_frame",
    "_node_contains_or_encloses_span",
    "_node_earliest_start_pos",
    "_pick_inferred_col",
    "_propagate_local_closure_syms",
    "_propagate_sibling_closures",
    "_push_compound_children",
    "_push_seq",
    "_record_binding_at_pos",
    "_record_carrier_closures_and_aliases",
    "_record_deletion_at_pos",
    "_record_simple_node_bindings_and_reads",
    "_resolve_unit_col_bounds",
    "_stmt_dump_set",
    "_strictly_encloses_span",
    "_traverse_fn_cell_frames",
    "_try_apply_cached_stmt_delta",
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
    intra_nl_mutators: Sequence[
        Union[_IntraUnitMutator, Tuple[str, Set[str], bool]]
    ],
    unit_top_locals: AbstractSet[str],
    nonlocal_set: AbstractSet[str],
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
    for mut_entry in intra_nl_mutators:
        mutated_vars = mut_entry[1]
        escaping_nl_muts = (mutated_vars - top_locals_set) | (
            mutated_vars & set(nonlocal_set)
        )
        if bool(escaping_nl_muts & all_chain_locals):
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
    escaped_unit_names, escaped_def_ids = _collect_escaped_intra_unit_defs(
        candidate_stmts, all_post_reads | all_sib_reads
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
        for var in mutated_vars & top_locals_set:
            if var in outer_sib_cells or (
                mut_fn_escapes
                and (var in outer_touched or var in all_post_reads)
            ):
                return True
    return False


def _check_sibling_closure_sharing(
    eff_input_set: AbstractSet[str],
    sub_mutated: AbstractSet[str],
    chain_infos: Sequence[_EnclosingFnCellInfo],
    *,
    is_suspended: bool = False,
    unreturned_mutations: bool = False,
    skip_pre_unit_closures: bool = False,
) -> bool:
    """Check 4: sibling closure referenced by unit (or concurrent with suspended unit) shares a mutated cell."""
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
    if (
        unreturned_mutations
        and skip_pre_unit_closures
        and any(
            bool(set(sub_mutated) & all_chain_locals & set(info.post_reads))
            for info in chain_infos
        )
    ):
        return True
    unshadowed_sibs: List[
        Tuple[Optional[str], AbstractSet[str], AbstractSet[str]]
    ] = []
    unshadowed_edges: List[Tuple[str, AbstractSet[str]]] = []
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
        is_ref_sib = sib_name is not None and sib_name in eff_input_set
        if not is_ref_sib and (
            not unreturned_mutations or skip_pre_unit_closures
        ):
            continue
        if bool(sib_reads & all_chain_locals & set(sub_mutated)):
            return True
        if bool(
            sib_nl_muts
            & all_chain_locals
            & (sub_touched if is_ref_sib else set(sub_mutated))
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


def _has_severed_closure_cell_hazard(ctx: _UnitClosureHazardContext) -> bool:
    """Detects whether extracting a unit to module scope would sever closure cell bindings.

    Note:
        When ``ctx.skip_pre_unit_closures`` is True (lenient mode), only the speculative
        ``is_suspended`` check for unreferenced concurrent pre-unit closures in Check 4 is
        relaxed; Checks 1/2 (unbound or externally mutated captured cells) and Check 4 for
        sibling closures explicitly referenced via ``eff_input_set`` remain enforced.
    """
    if ctx.mod_tree is None or ctx.u_start <= 0:
        return False
    if not ctx.is_subroutine and ctx.unit_kind not in (
        "closure",
        "comprehension",
        "complex_expr",
    ):
        return False

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
    enclosing_chain = _find_enclosing_function_chain(
        ctx.mod_tree, ctx.u_start, ctx.u_end, u_start_col, u_end_col
    )
    if not enclosing_chain:
        return False

    is_expr_unit = ctx.unit_kind in ("comprehension", "complex_expr")
    expr_unit_stores_escape = ctx.is_subroutine or is_expr_unit
    store_set = set(ctx.stores)
    output_set = set(ctx.outputs)
    nonlocal_set = set(ctx.nonlocals)
    unit_mutated_names = store_set | set(ctx.deleted_names)
    if not expr_unit_stores_escape and bool(nonlocal_set & unit_mutated_names):
        return True

    enc_nonlocals, enc_outer_globals = _extract_scope_nonlocals_and_outer_globals(
        enclosing_chain[-1][2],
        ctx.u_start,
        ctx.u_end,
        u_start_col=u_start_col,
        u_end_col=u_end_col,
    )
    enc_escaping = enc_nonlocals | enc_outer_globals
    if expr_unit_stores_escape and bool(
        unit_mutated_names
        & (enc_outer_globals if ctx.is_subroutine else enc_escaping)
    ):
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
    if ctx.is_subroutine and _extract_direct_nonlocals(unit_mod):
        return True
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
    elif ctx.unit_kind == "closure":
        unit_closure_reads.update(ctx.free_vars)

    intra_closure_reads: Set[str] = (
        set(unit_closure_reads)
        if ctx.is_subroutine
        else (set(unit_closure_reads) | nonlocal_set)
    )

    eff_input_set = set(ctx.inputs) | set(ctx.free_vars) | nonlocal_set
    if expr_unit_stores_escape and bool(
        unit_closure_reads
        & (
            (unit_mutated_names & (enc_escaping | nonlocal_set))
            | (enc_outer_globals & eff_input_set)
        )
    ):
        return True
    unit_mutations = unit_mutated_names if expr_unit_stores_escape else set()

    if (
        not intra_closure_reads
        and not intra_nl_mutators
        and not ctx.is_subroutine
        and not unit_mutations
    ):
        return False

    chain_infos = _build_enclosing_chain_cell_infos(
        enclosing_chain,
        ctx.u_start,
        ctx.u_end,
        u_start_col,
        u_end_col,
        unit_mutations,
        mod_tree=ctx.mod_tree,
    )
    if (
        ctx.definite_stores is not None
        and ctx.is_subroutine
        and not ctx.has_yield
        and chain_infos
    ):
        unseeded_cond_stores = (
            store_set - set(ctx.definite_stores)
        ) - set(ctx.inputs)
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
            cond_pre_post = (
                imm_info.pre_mutated - imm_info.pre_bound
            ) & (imm_info.post_reads | unit_closure_reads)
            cond_exported_closure_capture = (
                output_set & set(imm_info.post_reads) & unit_closure_reads
            )
            if bool(
                unseeded_cond_stores
                & imm_info.fn_locals
                & (
                    eff_sib_reads
                    | cond_pre_post
                    | cond_exported_closure_capture
                )
            ):
                return True

    captured_candidates = intra_closure_reads & (
        eff_input_set
        | ((store_set | output_set) if expr_unit_stores_escape else set())
    )
    sub_mutated = (
        unit_mutated_names
        if expr_unit_stores_escape
        else (nonlocal_set & unit_mutated_names)
    )
    return (
        _check_intra_unit_nonlocal_mutators(
            ctx.candidate_stmts,
            intra_nl_mutators,
            ctx.unit_top_locals,
            nonlocal_set,
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
            is_suspended=(is_suspended or is_expr_unit)
            and not ctx.skip_pre_unit_closures,
            unreturned_mutations=is_expr_unit,
            skip_pre_unit_closures=ctx.skip_pre_unit_closures,
        )
    )
