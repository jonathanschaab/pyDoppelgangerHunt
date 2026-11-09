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
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _EnclosingFnCellInfo,
    _IntraUnitMutator,
    _collect_escaped_and_executed_intra_unit_defs,
    _collect_unit_nonlocal_mutators,
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
from pydoppelgangerhunt.parser import _node_is_effectively_async

__all__ = [
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
    is_suspended: bool = False,
    eff_input_set: AbstractSet[str] = frozenset(),
    enc_escaping: AbstractSet[str] = frozenset(),
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
    outer_escape_targets = (
        all_chain_locals | set(enc_escaping) | set(nonlocal_set)
    )
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
            id(mut_node) in executed_def_ids
            and not isinstance(mut_node, ast.ClassDef)
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
        unshadowed_fn_locals = set(info.fn_locals) - inner_shadowed
        chain_locals.update(unshadowed_fn_locals)
        concurrent_vars.update(
            unshadowed_fn_locals
            & (set(info.concurrent_sibling_vars) - inner_shadowed)
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
        if not is_ref_sib and (
            not unreturned_mutations or skip_pre_unit_closures
        ):
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


def _unit_has_global_mutating_call(
    candidate_stmts: Sequence[ast.stmt],
    shadowed_names: AbstractSet[str],
    *,
    has_star_import: bool = False,
    allow_heuristic_builtins: bool = True,
) -> bool:
    """Returns True when candidate_stmts contain a call or protocol hook other than an allowed unshadowed builtin.

    When an extracted helper reads a variable declared ``global`` in an enclosing
    function, the helper receives that global's value as a snapshot parameter at
    call entry. Any impure or user-defined call, context-manager protocol hook
    (``with`` / ``async with``), coroutine/generator suspension point (``await``,
    ``yield``, ``yield from``, ``async for``), or deletion (``del``) inside the
    helper could mutate the module-level global mid-execution, causing subsequent
    reads inside the helper to observe the stale parameter rather than the updated
    global.

    In ``strict`` mode (``allow_heuristic_builtins=True``), both
    ``_PROVEN_PURE_GLOBAL_SAFE_BUILTINS`` and ``_HEURISTIC_GLOBAL_SAFE_BUILTINS``
    are permitted when unshadowed. In ``pedantic`` mode
    (``allow_heuristic_builtins=False``), only ``_PROVEN_PURE_GLOBAL_SAFE_BUILTINS``
    are permitted because dunder-invoking built-ins (such as ``len()``, ``bool()``,
    ``int()``, ``isinstance()``) can execute arbitrary user code in ``__len__``,
    ``__bool__``, ``__int__``, ``__index__``, or ``__instancecheck__``. Non-call
    AST operations (operators, subscripts, attribute/``@property`` reads, and
    synchronous ``for`` iteration) are not treated as global-mutating even in
    ``pedantic`` mode.
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
    effective_shadowed: Set[str] = set(shadowed_names) | _extract_single_node_local_names(
        unit_stmt_mod, include_globals=True
    )
    for st in candidate_stmts:
        for node in ast.walk(st):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                effective_shadowed.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
                effective_shadowed.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                effective_shadowed.update(node.names)
            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.Lambda,
                    *_COMPREHENSION_TYPES,
                ),
            ):
                effective_shadowed.update(
                    _extract_single_node_local_names(node, include_globals=True)
                )
    all_builtins_shadowed = has_star_import or ("*" in effective_shadowed)
    for st in candidate_stmts:
        for node in ast.walk(st):
            if isinstance(
                node,
                (
                    ast.With,
                    ast.AsyncWith,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.AsyncFor,
                    ast.Delete,
                ),
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
        has_missing_end = False
        has_overlapping_closure = False
        for node in span_idx.closure_like_nodes:
            n_start = getattr(node, "lineno", None)
            if not isinstance(n_start, int) or n_start > ctx.u_end:
                continue
            n_end = getattr(node, "end_lineno", None)
            if not isinstance(n_end, int):
                has_missing_end = True
                break
            if n_end >= ctx.u_start:
                has_overlapping_closure = True
        has_unresolved = has_missing_end or (
            has_overlapping_closure
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
    setattr(unit_mod, "_is_ephemeral_unit_mod", True)
    is_expr_unit = (
        ctx.unit_kind in ("comprehension", "complex_expr")
        and not ctx.is_subroutine
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
        (
            _extract_comprehension_walrus_names(unit_mod)
            | explicit_nonlocal_mutations
        )
        if is_expr_unit
        else explicit_nonlocal_mutations
    )
    escaping_unit_mutations: Set[str] = (
        helper_rebound_vars | immediate_cell_mutations
    )
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
                ctx.candidate_stmts[0].body
                if len(ctx.candidate_stmts) == 1
                and isinstance(
                    ctx.candidate_stmts[0],
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                )
                else ctx.candidate_stmts,
                True,
            )
        )
    )
    if is_suspended and bool(unit_mutated_names & (enc_escaping | nonlocal_set)):
        _append_global_or_cell_hazard(
            reasons, unit_mutated_names, enc_outer_globals
        )
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
    if unit_stores_affect_enclosing and bool(
        eff_input_set & enc_outer_globals
    ):
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
        _append_global_or_cell_hazard(
            reasons, unit_closure_reads, enc_outer_globals
        )
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
            unexported_pre_post = (
                (imm_info.pre_mutated & imm_info.post_reads) - output_set
            )
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
