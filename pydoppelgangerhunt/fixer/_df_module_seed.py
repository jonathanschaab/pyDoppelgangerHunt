"""Module-level suppression seed builder, lazy statement seed map, and scope alias seeding."""

from __future__ import annotations

import ast
import math
import threading
from dataclasses import (
    dataclass,
    field,
    replace,
)
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydoppelgangerhunt._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
    is_trystar_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _SessionBucket,
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _tree_contains_scope_node,
    analysis_session,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import _FnScopeContext
from pydoppelgangerhunt.fixer._df_cfg import (
    _CompoundClause,
    _is_try_sibling_reachable_after,
    _locate_enclosing_clause_idx,
    _lower_compound_clauses,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _extract_single_node_local_names,
    _node_has_star_import,
)
from pydoppelgangerhunt.fixer._df_exit_protocol import (
    ExitProtocolPatchScan,
    _NO_PATCH_SCAN,
    scan_exit_protocol_patches,
)
from pydoppelgangerhunt.fixer._df_prov_exprs import (
    _advance_clause_entry_prov,
    _advance_prov_through_for_header,
    _advance_prov_through_opt_expr,
    _merge_prior_clause_guard_and_binds,
)
from pydoppelgangerhunt.fixer._df_provenance_walk import (
    _ProvenanceWalkContext,
    _advance_prov_through_with_items,
    _apply_class_def_external_effects,
    _update_prov_and_term_from_stmt,
    _update_prov_and_term_from_stmts,
    _update_suppress_provenance_from_stmt,
)
from pydoppelgangerhunt.fixer._df_span_index import _find_enclosing_function_chain
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _collect_func_params as _extract_arg_names,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _EMPTY_SUPPRESS_PROVENANCE,
    _SAFE_EXCEPTION_TYPE_NAMES,
    _SHADOWABLE_BUILTIN_CTX_NAMES,
    _SuppressProvenance,
    _TRACKED_SHADOW_NAMES,
)
from pydoppelgangerhunt.fixer._df_with_rules import (
    _conservatively_update_prov_at_depth_limit,
    _scope_has_direct_star_import,
)


def _extract_stmt_referenced_and_bound_names_and_with(
    stmt: ast.AST,
) -> Tuple[Set[str], bool]:
    """Extracts identifier strings referenced or bound in stmt and whether stmt contains With/AsyncWith."""
    names: Set[str] = set()
    has_with = False
    for node in ast.walk(stmt):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            has_with = True
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            if node.asname:
                names.add(node.asname)
            if node.name:
                names.add(node.name.split(".", 1)[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        else:
            for attr_name in ("name", "rest"):
                bound_val = getattr(node, attr_name, None)
                if isinstance(bound_val, str) and bound_val:
                    names.add(bound_val)
    return names, has_with


class _ModuleStmtDelta(NamedTuple):
    """Per-statement forward and branch provenance delta for linear-time module seeding."""

    stmt_id: int
    fwd_add_sup: frozenset[str]
    fwd_rem_sup: frozenset[str]
    fwd_add_susp: frozenset[str]
    fwd_rem_susp: frozenset[str]
    fwd_add_nonsup: frozenset[str]
    fwd_rem_nonsup: frozenset[str]
    fwd_add_nonsup_inst: frozenset[str]
    fwd_rem_nonsup_inst: frozenset[str]
    fwd_add_mods: frozenset[Tuple[str, str]]
    fwd_rem_mods: frozenset[Tuple[str, str]]
    fwd_add_imps: frozenset[Tuple[str, str]]
    fwd_rem_imps: frozenset[Tuple[str, str]]
    fwd_add_shad_b: frozenset[str]
    fwd_rem_shad_b: frozenset[str]
    fwd_add_shad_n: frozenset[str]
    fwd_rem_shad_n: frozenset[str]
    fwd_star: bool
    fwd_cs_mut: frozenset[str]
    fwd_cs_sup: frozenset[str]
    fwd_cs_susp: frozenset[str]
    branch_new_sup: frozenset[str]
    branch_new_susp: frozenset[str]
    branch_shad_b: frozenset[str]
    branch_shad_n: frozenset[str]
    branch_star: bool


def _with_patch_scan_state(
    prov: _SuppressProvenance, patch_scan: ExitProtocolPatchScan
) -> _SuppressProvenance:
    """Attaches module-wide exit-protocol patch state without mutating existing bindings."""
    if (
        not patch_scan.has_patch
        and not patch_scan.rebound_attrs
        and not patch_scan.patched_names
    ):
        return prov
    return replace(
        prov,
        exit_patch_targets=prov.exit_patch_targets | patch_scan.patched_names,
        has_exit_patch=prov.has_exit_patch or patch_scan.has_patch,
        rebound_factory_targets=(
            prov.rebound_factory_targets | patch_scan.rebound_attrs
        ),
    )


def _apply_exit_protocol_patch_scan(
    prov: _SuppressProvenance, patch_scan: ExitProtocolPatchScan
) -> _SuppressProvenance:
    """Applies exit-protocol and factory-rebind invalidations from patch_scan onto prov."""
    return prov.with_exit_protocol_patches(
        patch_scan.patched_names,
        has_patch=patch_scan.has_patch,
        rebound_attrs=patch_scan.rebound_attrs,
    )


def _apply_forward_delta_to_sets(
    d: _ModuleStmtDelta,
    p_sup: Set[str],
    p_susp: Set[str],
    p_nonsup: Set[str],
    p_nonsup_inst: Set[str],
    p_mods: Set[Tuple[str, str]],
    p_imps: Set[Tuple[str, str]],
    p_shad_b: Set[str],
    p_shad_n: Set[str],
    p_cs_mut: Set[str],
    p_cs_sup: Set[str],
    p_cs_susp: Set[str],
    p_star: bool,
) -> bool:
    """Applies one forward _ModuleStmtDelta in place and returns updated p_star."""
    empty_str: frozenset[str] = frozenset()
    for target_set, rem_items, add_items in (
        (p_sup, d.fwd_rem_sup, d.fwd_add_sup),
        (p_susp, d.fwd_rem_susp, d.fwd_add_susp),
        (p_nonsup, d.fwd_rem_nonsup, d.fwd_add_nonsup),
        (p_nonsup_inst, d.fwd_rem_nonsup_inst, d.fwd_add_nonsup_inst),
        (p_shad_b, d.fwd_rem_shad_b, d.fwd_add_shad_b),
        (p_shad_n, d.fwd_rem_shad_n, d.fwd_add_shad_n),
        (p_cs_mut, empty_str, d.fwd_cs_mut),
        (p_cs_sup, empty_str, d.fwd_cs_sup),
        (p_cs_susp, empty_str, d.fwd_cs_susp),
    ):
        if rem_items:
            target_set.difference_update(rem_items)
        if add_items:
            target_set.update(add_items)
    for pair_set, rem_pairs, add_pairs in (
        (p_mods, d.fwd_rem_mods, d.fwd_add_mods),
        (p_imps, d.fwd_rem_imps, d.fwd_add_imps),
    ):
        if rem_pairs:
            pair_set.difference_update(rem_pairs)
        if add_pairs:
            pair_set.update(add_pairs)
    return p_star or d.fwd_star


def _apply_suffix_delta_to_sets(
    d: _ModuleStmtDelta,
    s_sup: Set[str],
    s_susp: Set[str],
    s_shad_b: Set[str],
    s_shad_n: Set[str],
    s_star: bool,
) -> bool:
    """Applies one suffix _ModuleStmtDelta in place and returns updated s_star."""
    for target_set, add_items in (
        (s_sup, d.branch_new_sup),
        (s_susp, d.branch_new_susp),
        (s_shad_b, d.branch_shad_b),
        (s_shad_n, d.branch_shad_n),
    ):
        if add_items:
            target_set.update(add_items)
    return s_star or d.branch_star


_SuffixCheckpoint = Tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    bool,
]


class _LazyModuleStmtSeedMap(Mapping[int, _SuppressProvenance]):
    """Checkpointed on-demand mapping from top-level statement id() to prefix, suffix, or combined provenance."""

    __slots__ = (
        "_deltas",
        "_idx_by_id",
        "_kind",
        "_patch_scan",
        "_cache",
        "_stride",
        "_prefix_checkpoints",
        "_suffix_checkpoints",
        "_lock",
    )

    def __init__(
        self,
        deltas: Tuple[_ModuleStmtDelta, ...],
        idx_by_id: Mapping[int, int],
        kind: str,
        patch_scan: ExitProtocolPatchScan = _NO_PATCH_SCAN,
    ) -> None:
        self._deltas = deltas
        self._idx_by_id = idx_by_id
        self._kind = kind
        self._patch_scan = patch_scan
        self._cache: dict[int, _SuppressProvenance] = {}
        self._stride: int = max(64, math.isqrt(len(deltas))) if deltas else 1
        self._prefix_checkpoints: Optional[Tuple[_SuppressProvenance, ...]] = None
        self._suffix_checkpoints: Optional[Tuple[_SuffixCheckpoint, ...]] = None
        self._lock: threading.RLock = threading.RLock()

    def __len__(self) -> int:
        return len(self._idx_by_id)

    def __iter__(self) -> Iterator[int]:
        return iter(self._idx_by_id)

    def __contains__(self, key: object) -> bool:
        return key in self._idx_by_id

    def _snapshot_prefix_prov(
        self,
        p_sup: AbstractSet[str],
        p_nonsup: AbstractSet[str],
        p_mods: AbstractSet[Tuple[str, str]],
        p_shad_b: AbstractSet[str],
        p_star: bool,
        p_imps: AbstractSet[Tuple[str, str]],
        p_shad_n: AbstractSet[str],
        p_nonsup_inst: AbstractSet[str],
        p_susp: AbstractSet[str],
        p_cs_mut: AbstractSet[str],
        p_cs_sup: AbstractSet[str],
        p_cs_susp: AbstractSet[str],
    ) -> _SuppressProvenance:
        return _SuppressProvenance(
            frozenset(p_sup),
            frozenset(p_nonsup),
            frozenset(p_mods),
            frozenset(p_shad_b),
            p_star,
            frozenset(p_imps),
            frozenset(p_shad_n),
            frozenset(p_nonsup_inst),
            frozenset(p_susp),
            frozenset(p_cs_mut),
            frozenset(p_cs_sup),
            frozenset(p_cs_susp),
            self._patch_scan.patched_names,
            self._patch_scan.has_patch,
            self._patch_scan.rebound_attrs,
        )

    def _ensure_checkpoints(self) -> None:
        with self._lock:
            deltas = self._deltas
            n = len(deltas)
            stride = self._stride
            num_blocks = (n + stride - 1) // stride if n else 0
            if self._kind in ("prefix", "combined") and self._prefix_checkpoints is None:
                p_sup: Set[str] = set()
                p_susp: Set[str] = set()
                p_nonsup: Set[str] = set()
                p_nonsup_inst: Set[str] = set()
                p_mods: Set[Tuple[str, str]] = set()
                p_imps: Set[Tuple[str, str]] = set()
                p_shad_b: Set[str] = set()
                p_shad_n: Set[str] = set()
                p_cs_mut: Set[str] = set()
                p_cs_sup: Set[str] = set()
                p_cs_susp: Set[str] = set()
                p_star = False
                pref_cps: List[_SuppressProvenance] = []
                for idx in range(n):
                    if idx % stride == 0:
                        pref_cps.append(
                            self._snapshot_prefix_prov(
                                p_sup,
                                p_nonsup,
                                p_mods,
                                p_shad_b,
                                p_star,
                                p_imps,
                                p_shad_n,
                                p_nonsup_inst,
                                p_susp,
                                p_cs_mut,
                                p_cs_sup,
                                p_cs_susp,
                            )
                        )
                    p_star = _apply_forward_delta_to_sets(
                        deltas[idx],
                        p_sup,
                        p_susp,
                        p_nonsup,
                        p_nonsup_inst,
                        p_mods,
                        p_imps,
                        p_shad_b,
                        p_shad_n,
                        p_cs_mut,
                        p_cs_sup,
                        p_cs_susp,
                        p_star,
                    )
                self._prefix_checkpoints = tuple(pref_cps)

            if self._kind in ("suffix", "combined") and self._suffix_checkpoints is None:
                s_sup: Set[str] = set()
                s_susp: Set[str] = set()
                s_shad_b: Set[str] = set()
                s_shad_n: Set[str] = set()
                s_star = False
                empty_cp: _SuffixCheckpoint = (
                    frozenset(),
                    frozenset(),
                    frozenset(),
                    frozenset(),
                    False,
                )
                suff_cps: List[_SuffixCheckpoint] = [empty_cp] * num_blocks
                for idx in range(n - 1, -1, -1):
                    if idx == n - 1 or (idx + 1) % stride == 0:
                        suff_cps[idx // stride] = (
                            frozenset(s_sup),
                            frozenset(s_susp),
                            frozenset(s_shad_b),
                            frozenset(s_shad_n),
                            s_star,
                        )
                    s_star = _apply_suffix_delta_to_sets(
                        deltas[idx], s_sup, s_susp, s_shad_b, s_shad_n, s_star
                    )
                self._suffix_checkpoints = tuple(suff_cps)

    def _materialize_prefix(self, target_idx: int) -> _SuppressProvenance:
        if self._prefix_checkpoints is None:
            raise RuntimeError("prefix provenance checkpoints not initialized")
        block = target_idx // self._stride
        base_idx = block * self._stride
        cp = self._prefix_checkpoints[block]
        if base_idx == target_idx:
            return cp
        p_sup = set(cp.suppressors)
        p_susp = set(cp.suspicious_aliases)
        p_nonsup = set(cp.non_suppressors)
        p_nonsup_inst = set(cp.non_suppressor_instances)
        p_mods = set(cp.stdlib_modules)
        p_imps = set(cp.imported_symbols)
        p_shad_b = set(cp.shadowed_builtins)
        p_shad_n = set(cp.shadowed_names)
        p_cs_mut = set(cp.cross_scope_mutated)
        p_cs_sup = set(cp.cross_scope_suppressors)
        p_cs_susp = set(cp.cross_scope_suspicious)
        p_star = cp.has_star_import
        for idx in range(base_idx, target_idx):
            p_star = _apply_forward_delta_to_sets(
                self._deltas[idx],
                p_sup,
                p_susp,
                p_nonsup,
                p_nonsup_inst,
                p_mods,
                p_imps,
                p_shad_b,
                p_shad_n,
                p_cs_mut,
                p_cs_sup,
                p_cs_susp,
                p_star,
            )
        return self._snapshot_prefix_prov(
            p_sup,
            p_nonsup,
            p_mods,
            p_shad_b,
            p_star,
            p_imps,
            p_shad_n,
            p_nonsup_inst,
            p_susp,
            p_cs_mut,
            p_cs_sup,
            p_cs_susp,
        )

    def _materialize_suffix(self, target_idx: int) -> _SuffixCheckpoint:
        if self._suffix_checkpoints is None:
            raise RuntimeError("suffix provenance checkpoints not initialized")
        n = len(self._deltas)
        block = target_idx // self._stride
        end_idx = min((block + 1) * self._stride - 1, n - 1)
        cp = self._suffix_checkpoints[block]
        if end_idx == target_idx:
            return cp
        s_sup = set(cp[0])
        s_susp = set(cp[1])
        s_shad_b = set(cp[2])
        s_shad_n = set(cp[3])
        s_star = cp[4]
        for idx in range(end_idx, target_idx, -1):
            s_star = _apply_suffix_delta_to_sets(
                self._deltas[idx], s_sup, s_susp, s_shad_b, s_shad_n, s_star
            )
        return (
            frozenset(s_sup),
            frozenset(s_susp),
            frozenset(s_shad_b),
            frozenset(s_shad_n),
            s_star,
        )

    def __getitem__(self, stmt_id: int) -> _SuppressProvenance:
        target_idx = self._idx_by_id[stmt_id]
        with self._lock:
            cached = self._cache.get(stmt_id)
            if cached is not None:
                return cached
            self._ensure_checkpoints()
            if self._kind == "prefix":
                res = _apply_exit_protocol_patch_scan(
                    self._materialize_prefix(target_idx), self._patch_scan
                )
                self._cache[stmt_id] = res
                return res

            s_sup, s_susp, s_shad_b, s_shad_n, s_star = self._materialize_suffix(
                target_idx
            )
            if self._kind == "suffix":
                res = _SuppressProvenance(
                    suppressors=s_sup,
                    shadowed_builtins=s_shad_b,
                    has_star_import=s_star,
                    shadowed_names=s_shad_n,
                    suspicious_aliases=s_susp,
                )
                self._cache[stmt_id] = res
                return res

            pref = self._materialize_prefix(target_idx)
            res = _apply_exit_protocol_patch_scan(
                pref.apply_branch_delta(
                    s_shad_n,
                    s_shad_b,
                    s_sup,
                    s_star,
                    new_suspicious=s_susp,
                ),
                self._patch_scan,
            )
            self._cache[stmt_id] = res
            return res


_EMPTY_STMT_SEED_MAP: Mapping[int, _SuppressProvenance] = MappingProxyType({})


@dataclass(frozen=True)
class _ModuleSuppressSeed:
    """Cached per-module top-level suppression provenance summary."""

    full_forward_prov: _SuppressProvenance
    by_top_stmt_id: Mapping[int, _SuppressProvenance]
    has_any_positive_or_suppressor: bool
    prefix_by_top_stmt_id: Mapping[int, _SuppressProvenance] = field(
        default_factory=lambda: _EMPTY_STMT_SEED_MAP
    )
    suffix_delta_by_top_stmt_id: Mapping[int, _SuppressProvenance] = field(
        default_factory=lambda: _EMPTY_STMT_SEED_MAP
    )
    module_cross_scope_mutated: frozenset[str] = frozenset()
    module_cross_scope_suppressors: frozenset[str] = frozenset()
    module_cross_scope_suspicious: frozenset[str] = frozenset()

def _apply_set_delta(
    target: Set[str],
    before: frozenset[str],
    after: frozenset[str],
) -> Tuple[frozenset[str], frozenset[str]]:
    """Computes (added, removed) between before/after and applies them to target in place."""
    added = after - before
    removed = before - after
    if removed:
        target.difference_update(removed)
    if added:
        target.update(added)
    return added, removed


def _apply_map_delta(
    target: dict[str, str],
    before: frozenset[Tuple[str, str]],
    after: frozenset[Tuple[str, str]],
    *,
    mirror_all: Optional[dict[str, str]] = None,
) -> Tuple[frozenset[Tuple[str, str]], frozenset[Tuple[str, str]]]:
    """Computes (added, removed) pair sets and applies them to target dict (and optional mirror_all)."""
    added = after - before
    removed = before - after
    for k, _ in removed:
        target.pop(k, None)
    for k, v in added:
        target[k] = v
        if mirror_all is not None:
            mirror_all[k] = v
    return added, removed


def _collect_initial_relevant_seed_names(
    patch_scan: ExitProtocolPatchScan,
    stmt_infos: Sequence[Tuple[AbstractSet[str], bool]],
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> Set[str]:
    """Collects symbol names referenced by patch scans, allowlists, and statements containing with blocks."""
    relevant: Set[str] = (
        set(_TRACKED_SHADOW_NAMES)
        | set(_SAFE_EXCEPTION_TYPE_NAMES)
        | set(patch_scan.patched_names)
    )
    for pat in extra_non_suppressing_ctx_managers:
        for part in pat.split("."):
            if part and part != "*":
                relevant.add(part)
    for r_tgt, r_attr in patch_scan.rebound_attrs:
        if r_tgt:
            relevant.add(r_tgt)
        if r_attr:
            relevant.add(r_attr)
    for s_names, s_has_with in stmt_infos:
        if s_has_with:
            relevant.update(s_names)
    return relevant


@dataclass
class _ModuleSeedAccumulator:
    """Mutable O(N) accumulator for top-level module suppression provenance and statement deltas."""

    patch_scan: ExitProtocolPatchScan
    extra_non_sup: AbstractSet[str]
    strict_with: bool
    cur_sup: Set[str] = field(default_factory=set)
    cur_susp: Set[str] = field(default_factory=set)
    cur_all_sup: Set[str] = field(default_factory=set)
    cur_all_susp: Set[str] = field(default_factory=set)
    cur_all_imps: dict[str, str] = field(default_factory=dict)
    cur_nonsup: Set[str] = field(default_factory=set)
    cur_nonsup_inst: Set[str] = field(default_factory=set)
    cur_mods: dict[str, str] = field(default_factory=dict)
    cur_imps: dict[str, str] = field(default_factory=dict)
    cur_shad_b: Set[str] = field(default_factory=set)
    cur_shad_n: Set[str] = field(default_factory=set)
    cur_cs_mut: Set[str] = field(default_factory=set)
    cur_cs_sup: Set[str] = field(default_factory=set)
    cur_cs_susp: Set[str] = field(default_factory=set)
    all_cs_mut: Set[str] = field(default_factory=set)
    all_cs_sup: Set[str] = field(default_factory=set)
    all_cs_susp: Set[str] = field(default_factory=set)
    cur_star: bool = False
    has_pos_or_sup: bool = False

    def _compute_branch_delta(
        self,
        m_stmt: ast.stmt,
        stmt_names: AbstractSet[str],
    ) -> Tuple[_SuppressProvenance, frozenset[str], frozenset[str]]:
        """Computes branch-mode provenance delta and newly introduced suppressor/suspicious sets."""
        local_all_sup = frozenset(n for n in stmt_names if n in self.cur_all_sup)
        local_all_susp = frozenset(n for n in stmt_names if n in self.cur_all_susp)
        local_all_imps = frozenset(
            (n, self.cur_all_imps[n]) for n in stmt_names if n in self.cur_all_imps
        )
        seed_in = _with_patch_scan_state(
            _SuppressProvenance(
                suppressors=local_all_sup,
                suspicious_aliases=local_all_susp,
                imported_symbols=local_all_imps,
            )
            if (local_all_sup or local_all_susp or local_all_imps)
            else _EMPTY_SUPPRESS_PROVENANCE,
            self.patch_scan,
        )
        b_delta = _update_suppress_provenance_from_stmt(
            m_stmt,
            seed_in,
            in_branch=True,
            strict_with_suppression=self.strict_with,
            extra_non_suppressing_ctx_managers=self.extra_non_sup,
        )
        rebound_existing = (local_all_sup | local_all_susp) & b_delta.shadowed_names
        if rebound_existing:
            b_without_rebound = _update_suppress_provenance_from_stmt(
                m_stmt,
                _with_patch_scan_state(
                    _SuppressProvenance(
                        suppressors=local_all_sup - rebound_existing,
                        suspicious_aliases=local_all_susp - rebound_existing,
                        imported_symbols=frozenset(
                            pair
                            for pair in local_all_imps
                            if pair[0] not in rebound_existing
                        ),
                    ),
                    self.patch_scan,
                ),
                in_branch=True,
                strict_with_suppression=self.strict_with,
                extra_non_suppressing_ctx_managers=self.extra_non_sup,
            )
            new_sup = (b_delta.suppressors - local_all_sup) | (
                b_without_rebound.suppressors & rebound_existing
            )
            new_susp = (b_delta.suspicious_aliases - local_all_susp) | (
                b_without_rebound.suspicious_aliases & rebound_existing
            )
        else:
            new_sup = b_delta.suppressors - local_all_sup
            new_susp = b_delta.suspicious_aliases - local_all_susp
        if new_sup:
            self.cur_all_sup.update(new_sup)
        if new_susp:
            self.cur_all_susp.update(new_susp)
        if b_delta.cross_scope_mutated:
            self.all_cs_mut.update(b_delta.cross_scope_mutated)
            self.all_cs_sup.update(b_delta.cross_scope_suppressors)
            self.all_cs_susp.update(b_delta.cross_scope_suspicious)
        return b_delta, new_sup, new_susp

    def step_stmt(
        self,
        m_stmt: ast.stmt,
        stmt_names: AbstractSet[str],
    ) -> _ModuleStmtDelta:
        """Advances forward and branch provenance state across one top-level module statement."""
        b_delta, new_sup, new_susp = self._compute_branch_delta(m_stmt, stmt_names)
        eff_names: AbstractSet[str]
        if b_delta.has_star_import:
            eff_names = (
                stmt_names
                | self.cur_cs_mut
                | self.cur_nonsup
                | self.cur_nonsup_inst
                | self.cur_mods.keys()
                | self.cur_imps.keys()
                | self.cur_sup
            )
        else:
            eff_names = stmt_names | self.cur_cs_mut if self.cur_cs_mut else stmt_names

        local_sup = frozenset(n for n in eff_names if n in self.cur_sup)
        local_susp = frozenset(n for n in eff_names if n in self.cur_susp)
        local_nonsup = frozenset(n for n in eff_names if n in self.cur_nonsup)
        local_nonsup_inst = frozenset(n for n in eff_names if n in self.cur_nonsup_inst)
        local_mods = frozenset((n, self.cur_mods[n]) for n in eff_names if n in self.cur_mods)
        local_imps = frozenset((n, self.cur_imps[n]) for n in eff_names if n in self.cur_imps)
        local_shad_b = frozenset(self.cur_shad_b)
        local_shad_n = frozenset(
            n for n in (eff_names | _TRACKED_SHADOW_NAMES) if n in self.cur_shad_n
        )
        local_cur = _SuppressProvenance(
            suppressors=local_sup,
            non_suppressors=local_nonsup,
            stdlib_modules=local_mods,
            shadowed_builtins=local_shad_b,
            has_star_import=self.cur_star,
            imported_symbols=local_imps,
            shadowed_names=local_shad_n,
            non_suppressor_instances=local_nonsup_inst,
            suspicious_aliases=local_susp,
            cross_scope_mutated=frozenset(self.cur_cs_mut),
            cross_scope_suppressors=frozenset(self.cur_cs_sup),
            cross_scope_suspicious=frozenset(self.cur_cs_susp),
            exit_patch_targets=self.patch_scan.patched_names,
            has_exit_patch=self.patch_scan.has_patch,
            rebound_factory_targets=self.patch_scan.rebound_attrs,
        )
        out_cur = _update_suppress_provenance_from_stmt(
            m_stmt,
            local_cur,
            in_branch=False,
            strict_with_suppression=self.strict_with,
            extra_non_suppressing_ctx_managers=self.extra_non_sup,
        )

        fwd_add_sup, fwd_rem_sup = _apply_set_delta(self.cur_sup, local_sup, out_cur.suppressors)
        fwd_add_susp, fwd_rem_susp = _apply_set_delta(
            self.cur_susp, local_susp, out_cur.suspicious_aliases
        )
        fwd_add_nonsup, fwd_rem_nonsup = _apply_set_delta(
            self.cur_nonsup, local_nonsup, out_cur.non_suppressors
        )
        fwd_add_nonsup_inst, fwd_rem_nonsup_inst = _apply_set_delta(
            self.cur_nonsup_inst, local_nonsup_inst, out_cur.non_suppressor_instances
        )
        fwd_add_mods, fwd_rem_mods = _apply_map_delta(
            self.cur_mods, local_mods, out_cur.stdlib_modules
        )
        fwd_add_imps, fwd_rem_imps = _apply_map_delta(
            self.cur_imps, local_imps, out_cur.imported_symbols, mirror_all=self.cur_all_imps
        )
        fwd_add_shad_b, fwd_rem_shad_b = _apply_set_delta(
            self.cur_shad_b, local_shad_b, out_cur.shadowed_builtins
        )
        fwd_add_shad_n, fwd_rem_shad_n = _apply_set_delta(
            self.cur_shad_n, local_shad_n, out_cur.shadowed_names
        )

        fwd_cs_mut = out_cur.cross_scope_mutated - self.cur_cs_mut
        fwd_cs_sup = out_cur.cross_scope_suppressors - self.cur_cs_sup
        fwd_cs_susp = out_cur.cross_scope_suspicious - self.cur_cs_susp
        if fwd_cs_mut:
            self.cur_cs_mut.update(fwd_cs_mut)
            self.all_cs_mut.update(fwd_cs_mut)
        if fwd_cs_sup:
            self.cur_cs_sup.update(fwd_cs_sup)
            self.all_cs_sup.update(fwd_cs_sup)
        if fwd_cs_susp:
            self.cur_cs_susp.update(fwd_cs_susp)
            self.all_cs_susp.update(fwd_cs_susp)
        if out_cur.has_star_import:
            self.cur_star = True
        if (
            self.cur_sup
            or self.cur_susp
            or self.cur_nonsup
            or self.cur_nonsup_inst
            or self.cur_mods
            or self.cur_imps
            or self.cur_all_sup
            or self.cur_all_susp
            or self.cur_cs_mut
            or self.all_cs_mut
        ):
            self.has_pos_or_sup = True

        return _ModuleStmtDelta(
            stmt_id=id(m_stmt),
            fwd_add_sup=fwd_add_sup,
            fwd_rem_sup=fwd_rem_sup,
            fwd_add_susp=fwd_add_susp,
            fwd_rem_susp=fwd_rem_susp,
            fwd_add_nonsup=fwd_add_nonsup,
            fwd_rem_nonsup=fwd_rem_nonsup,
            fwd_add_nonsup_inst=fwd_add_nonsup_inst,
            fwd_rem_nonsup_inst=fwd_rem_nonsup_inst,
            fwd_add_mods=fwd_add_mods,
            fwd_rem_mods=fwd_rem_mods,
            fwd_add_imps=fwd_add_imps,
            fwd_rem_imps=fwd_rem_imps,
            fwd_add_shad_b=fwd_add_shad_b,
            fwd_rem_shad_b=fwd_rem_shad_b,
            fwd_add_shad_n=fwd_add_shad_n,
            fwd_rem_shad_n=fwd_rem_shad_n,
            fwd_star=out_cur.has_star_import,
            fwd_cs_mut=fwd_cs_mut,
            fwd_cs_sup=fwd_cs_sup,
            fwd_cs_susp=fwd_cs_susp,
            branch_new_sup=new_sup,
            branch_new_susp=new_susp,
            branch_shad_b=b_delta.shadowed_builtins,
            branch_shad_n=b_delta.shadowed_names,
            branch_star=b_delta.has_star_import,
        )

    def build_full_forward_prov(self) -> _SuppressProvenance:
        """Materializes the final full-module forward suppression provenance."""
        return _apply_exit_protocol_patch_scan(
            _SuppressProvenance(
                suppressors=frozenset(self.cur_sup),
                non_suppressors=frozenset(self.cur_nonsup),
                stdlib_modules=frozenset(self.cur_mods.items()),
                shadowed_builtins=frozenset(self.cur_shad_b),
                has_star_import=self.cur_star,
                imported_symbols=frozenset(self.cur_imps.items()),
                shadowed_names=frozenset(self.cur_shad_n),
                non_suppressor_instances=frozenset(self.cur_nonsup_inst),
                suspicious_aliases=frozenset(self.cur_susp),
                cross_scope_mutated=frozenset(self.all_cs_mut),
                cross_scope_suppressors=frozenset(self.all_cs_sup),
                cross_scope_suspicious=frozenset(self.all_cs_susp),
            ),
            self.patch_scan,
        )


def _build_module_suppress_seed(
    eff_mod: ast.Module,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _ModuleSuppressSeed:
    """Builds a reusable module-level suppression provenance summary in O(N) time."""
    patch_scan = scan_exit_protocol_patches(eff_mod)
    body = eff_mod.body
    stmt_infos = [
        _extract_stmt_referenced_and_bound_names_and_with(m_stmt) for m_stmt in body
    ]
    relevant_seed_names = _collect_initial_relevant_seed_names(
        patch_scan, stmt_infos, extra_non_suppressing_ctx_managers
    )

    acc = _ModuleSeedAccumulator(
        patch_scan=patch_scan,
        extra_non_sup=extra_non_suppressing_ctx_managers,
        strict_with=strict_with_suppression,
    )
    deltas: List[_ModuleStmtDelta] = []
    idx_by_id: dict[int, int] = {}
    for idx, (m_stmt, (stmt_names, _)) in enumerate(zip(body, stmt_infos)):
        idx_by_id[id(m_stmt)] = idx
        deltas.append(acc.step_stmt(m_stmt, stmt_names))

    relevant_seed_names.update(
        acc.cur_sup
        | acc.cur_susp
        | acc.cur_nonsup
        | acc.cur_nonsup_inst
        | acc.cur_mods.keys()
        | acc.cur_all_sup
        | acc.cur_all_susp
        | acc.all_cs_mut
        | acc.all_cs_sup
        | acc.all_cs_susp
    )
    pruned_deltas = tuple(
        d._replace(
            fwd_add_imps=frozenset(
                pair for pair in d.fwd_add_imps if pair[0] in relevant_seed_names
            ),
            fwd_rem_imps=frozenset(
                pair for pair in d.fwd_rem_imps if pair[0] in relevant_seed_names
            ),
            fwd_add_shad_n=d.fwd_add_shad_n & relevant_seed_names,
            fwd_rem_shad_n=d.fwd_rem_shad_n & relevant_seed_names,
            branch_shad_n=d.branch_shad_n & relevant_seed_names,
        )
        for d in deltas
    )

    full_forward_prov = acc.build_full_forward_prov()
    if not acc.has_pos_or_sup:
        return _ModuleSuppressSeed(
            full_forward_prov=full_forward_prov,
            by_top_stmt_id=MappingProxyType({}),
            has_any_positive_or_suppressor=False,
        )
    idx_map = MappingProxyType(idx_by_id)
    return _ModuleSuppressSeed(
        full_forward_prov=full_forward_prov,
        by_top_stmt_id=_LazyModuleStmtSeedMap(
            pruned_deltas, idx_map, "combined", patch_scan
        ),
        has_any_positive_or_suppressor=True,
        prefix_by_top_stmt_id=_LazyModuleStmtSeedMap(
            pruned_deltas, idx_map, "prefix", patch_scan
        ),
        suffix_delta_by_top_stmt_id=_LazyModuleStmtSeedMap(
            pruned_deltas, idx_map, "suffix", patch_scan
        ),
        module_cross_scope_mutated=frozenset(acc.all_cs_mut),
        module_cross_scope_suppressors=frozenset(acc.all_cs_sup),
        module_cross_scope_suspicious=frozenset(acc.all_cs_susp),
    )


def _get_module_suppress_seed(
    eff_mod: ast.Module,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _ModuleSuppressSeed:
    """Returns the cached _ModuleSuppressSeed for eff_mod."""
    extra_f = frozenset(extra_non_suppressing_ctx_managers)
    strict_w = bool(strict_with_suppression)
    if _is_ephemeral_unit_ast(eff_mod):
        return _build_module_suppress_seed(
            eff_mod,
            extra_non_suppressing_ctx_managers=extra_f,
            strict_with_suppression=strict_w,
        )
    seed_params: Tuple[Any, ...] = (
        (strict_w, extra_f) if (strict_w or extra_f) else ()
    )
    cache_key = _make_trusted_scope_cache_key(
        eff_mod, "module_suppress_seed", params=seed_params
    )
    return _get_or_compute_scope_cached(
        _SessionBucket.CELL_INFO,
        cache_key,
        lambda: _build_module_suppress_seed(
            eff_mod,
            extra_non_suppressing_ctx_managers=extra_f,
            strict_with_suppression=strict_w,
        ),
    )


def _merge_clause_branch_effects(
    prov: _SuppressProvenance,
    sib_body: Sequence[ast.AST],
    sib_bound: Tuple[str, ...],
    sib_hdr: Optional[ast.AST],
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    clear_bound_on_exit: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Conservatively merges a prior or sibling clause header, bindings, and body into prov."""
    sib_prov = _advance_prov_through_opt_expr(
        sib_hdr,
        prov,
        allow_positive=False,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    for b_name in sib_bound:
        sib_prov = sib_prov.clear_name(b_name)
    sib_res = _update_prov_and_term_from_stmts(
        sib_body,
        sib_prov,
        in_branch=True,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    sib_prov = sib_res.prov.merge_branch_conservative(sib_res.mid_prov)
    if clear_bound_on_exit:
        for b_name in sib_bound:
            sib_prov = sib_prov.clear_name(b_name)
    return prov.merge_branch_conservative(sib_prov)


def _merge_enclosing_prior_clause_effects(
    stmt: ast.AST,
    clauses: Sequence[_CompoundClause],
    enclosing_clause_idx: int,
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Merges try-body and prior clause header/binding effects before enclosing_clause_idx."""
    if enclosing_clause_idx <= 0:
        return prov
    enclosing_clause = clauses[enclosing_clause_idx]
    if is_try_or_trystar_node(stmt):
        t_body = getattr(stmt, "body", ())
        is_orelse = enclosing_clause.kind == "orelse"
        is_final = enclosing_clause.kind == "finalbody"
        for tb_stmt in t_body:
            tb_res = _update_prov_and_term_from_stmt(
                tb_stmt,
                prov,
                in_branch=not is_orelse,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                _depth=_depth + 1,
            )
            prov = (
                tb_res.prov
                if is_orelse
                else prov.merge_branch_conservative(
                    tb_res.prov
                ).merge_branch_conservative(tb_res.mid_prov)
            )
        if is_orelse:
            return prov
        merge_prior_bodies = is_final or is_trystar_node(stmt)
        for prior in clauses[1:enclosing_clause_idx]:
            if merge_prior_bodies:
                prov = _merge_clause_branch_effects(
                    prov,
                    prior.body,
                    prior.bound_names,
                    prior.guard_or_header,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    strict_with_suppression=strict_with_suppression,
                    clear_bound_on_exit=True,
                    _depth=_depth,
                )
            else:
                prov = _merge_prior_clause_guard_and_binds(
                    prov,
                    prior.guard_or_header,
                    (),
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
        return prov
    if is_match_node(stmt):
        for prior in clauses[:enclosing_clause_idx]:
            prov = _merge_prior_clause_guard_and_binds(
                prov,
                prior.guard_or_header,
                prior.bound_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
    return prov


# INVARIANT (SHARED CFG LOWERING CONTRACT — ENCLOSING-ANCESTOR SEEDER):
# This walker seeds suppression provenance strictly along the ancestor chain
# enclosing `target_node` / `target_lineno` using `_df_cfg._lower_compound_clauses`,
# `_df_cfg._locate_enclosing_clause_idx`, and `_df_cfg._is_try_sibling_reachable_after`.
# Any new Python compound statement must be lowered in `_df_cfg._lower_compound_clauses`
# first rather than introducing bespoke clause-unpacking branches here.
def _advance_enclosing_compound_stmt_seed(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    target_node: Optional[ast.AST],
    target_lineno: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Advances provenance into a compound statement enclosing target_node or target_lineno."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _conservatively_update_prov_at_depth_limit((stmt,), prov)
    if isinstance(stmt, ast.ClassDef) and target_node is not None:
        # Per PEP 227, ClassDef bodies do not form a lexical scope for nested
        # methods/classes/callables; only header, global/nonlocal, and cross-scope
        # mutations propagate across the class boundary.
        cls_ctx = _ProvenanceWalkContext(
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            depth=_depth + 1,
        )
        return _apply_class_def_external_effects(stmt, prov, cls_ctx).clear_name(
            stmt.name
        )
    is_loop = isinstance(stmt, (ast.While, ast.For, ast.AsyncFor))
    if isinstance(stmt, (ast.If, ast.While)):
        prov = _advance_prov_through_opt_expr(
            stmt.test,
            prov,
            allow_positive=not is_loop,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        _, prov = _advance_prov_through_for_header(
            stmt,
            prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        prov = _advance_prov_through_with_items(
            stmt,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
    elif is_match_node(stmt):
        prov = _advance_prov_through_opt_expr(
            getattr(stmt, "subject", None),
            prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )

    clauses = _lower_compound_clauses(stmt)
    enclosing_clause_idx = _locate_enclosing_clause_idx(
        clauses,
        lambda body: any(
            (target_node is not None and s is target_node)
            or (
                target_lineno > 0
                and int(getattr(s, "lineno", 0) or 0)
                <= target_lineno
                <= int(getattr(s, "end_lineno", getattr(s, "lineno", 0)) or 0)
            )
            for s in body
        ),
    )
    if enclosing_clause_idx is None:
        stmt_res = _update_prov_and_term_from_stmt(
            stmt,
            prov,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        return prov.merge_branch_conservative(
            stmt_res.prov
        ).merge_branch_conservative(stmt_res.mid_prov)

    target_clause = clauses[enclosing_clause_idx]
    prov = _merge_enclosing_prior_clause_effects(
        stmt,
        clauses,
        enclosing_clause_idx,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth,
    )
    prov = _advance_clause_entry_prov(
        prov,
        target_clause.guard_or_header,
        target_clause.bound_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )

    prov = _advance_prov_across_stmts_until_target(
        target_clause.body,
        prov,
        target_node,
        target_lineno,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )

    for c_idx, sib_clause in enumerate(clauses):
        if c_idx == enclosing_clause_idx and not is_loop:
            continue
        if isinstance(stmt, ast.If) or is_match_node(stmt):
            continue
        if is_try_or_trystar_node(stmt) and not _is_try_sibling_reachable_after(
            stmt,
            enclosing_clause_idx,
            c_idx,
            sib_clause.body,
            target_clause.body,
            target_node is not None,
        ):
            continue
        prov = _merge_clause_branch_effects(
            prov,
            sib_clause.body,
            sib_clause.bound_names,
            sib_clause.guard_or_header,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
            clear_bound_on_exit=is_try_or_trystar_node(stmt),
            _depth=_depth,
        )
    return prov


def _advance_prov_across_stmts_until_target(
    stmts: Sequence[ast.AST],
    prov: _SuppressProvenance,
    target_node: Optional[ast.AST],
    target_lineno: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Advances provenance forward to (target_node, target_lineno) and conservatively merges trailing statements."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _conservatively_update_prov_at_depth_limit(stmts, prov)
    reached = False
    for sub_stmt in stmts:
        if target_node is not None and sub_stmt is target_node:
            reached = True
            continue
        s_start = int(getattr(sub_stmt, "lineno", 0) or 0)
        s_end = int(getattr(sub_stmt, "end_lineno", s_start) or s_start)
        if not reached and target_lineno > 0 and s_start <= target_lineno <= s_end:
            if target_node is None and s_start >= target_lineno:
                return prov
            prov = _advance_enclosing_compound_stmt_seed(
                sub_stmt,
                prov,
                target_node,
                target_lineno,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
                _depth=_depth + 1,
            )
            if target_node is None:
                return prov
            reached = True
            continue
        if not reached and 0 < target_lineno <= s_start:
            if target_node is None:
                return prov
            reached = True
        next_sub = _update_suppress_provenance_from_stmt(
            sub_stmt,
            prov,
            in_branch=reached,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        prov = prov.merge_branch_conservative(next_sub) if reached else next_sub
    return prov


def _resolve_top_level_ancestor_seed(
    fn_node: ast.AST,
    eff_mod: ast.Module,
    enclosing_nodes: Sequence[ast.AST],
    mod_seed: _ModuleSuppressSeed,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Resolves module-level suppression seed using the top-level ancestor of fn_node."""
    top_target = enclosing_nodes[0] if enclosing_nodes else fn_node
    res: Optional[_SuppressProvenance] = None
    if id(top_target) in mod_seed.by_top_stmt_id:
        res = mod_seed.by_top_stmt_id[id(top_target)]
    else:
        target_lineno = int(getattr(top_target, "lineno", 0) or 0)
        if target_lineno > 0:
            for m_stmt in eff_mod.body:
                m_start = int(getattr(m_stmt, "lineno", 0) or 0)
                m_end = int(getattr(m_stmt, "end_lineno", m_start) or m_start)
                stmt_id = id(m_stmt)
                if m_start <= target_lineno <= m_end and stmt_id in mod_seed.by_top_stmt_id:
                    if top_target is m_stmt:
                        res = mod_seed.by_top_stmt_id[stmt_id]
                        break
                    pre_prov = mod_seed.prefix_by_top_stmt_id.get(
                        stmt_id, mod_seed.by_top_stmt_id[stmt_id]
                    )
                    inner_prov = _advance_enclosing_compound_stmt_seed(
                        m_stmt,
                        pre_prov,
                        top_target,
                        target_lineno,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                        strict_with_suppression=strict_with_suppression,
                    )
                    suf_delta = mod_seed.suffix_delta_by_top_stmt_id.get(stmt_id)
                    if suf_delta is not None:
                        inner_prov = inner_prov.apply_branch_delta(
                            suf_delta.shadowed_names,
                            suf_delta.shadowed_builtins,
                            suf_delta.suppressors,
                            suf_delta.has_star_import,
                            new_suspicious=suf_delta.suspicious_aliases,
                        )
                    res = inner_prov
                    break
    if res is None:
        res = mod_seed.full_forward_prov
    return res.with_cross_scope_mutations(
        mod_seed.module_cross_scope_mutated,
        mod_seed.module_cross_scope_suppressors,
        mod_seed.module_cross_scope_suspicious,
    )


def _clear_callable_scope_locals_and_invalidate(
    prov: _SuppressProvenance,
    scope_node: ast.AST,
    *,
    explicit_locals: Optional[AbstractSet[str]] = None,
) -> _SuppressProvenance:
    """Clears callable parameters/locals on entry and applies outer cross-scope invalidation."""
    s_args = getattr(scope_node, "args", None)
    if isinstance(s_args, ast.arguments):
        for p_name in _extract_arg_names(s_args):
            prov = prov.clear_name(p_name, clear_cross_scope=True)
    for tp_name in _extract_node_type_param_names(scope_node):
        prov = prov.clear_name(tp_name, clear_cross_scope=True)
    true_locals = (
        set(explicit_locals)
        if explicit_locals is not None
        else _extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
        )
    )
    for loc_name in true_locals:
        prov = prov.clear_name(loc_name, clear_cross_scope=True)
    decl_names = (
        _extract_scope_local_names(scope_node, include_self_name=False) - true_locals
    )
    if decl_names:
        prov = prov.invalidate_cross_scope_targets(decl_names)
    if not isinstance(scope_node, ast.ClassDef):
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _seed_scope_suppress_aliases_uncached(
    fn_node: ast.AST,
    eff_mod: Optional[ast.Module],
    enclosing_nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext],
    u_start: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Uncached implementation of _seed_scope_suppress_aliases."""
    patch_scan = scan_exit_protocol_patches(eff_mod if eff_mod is not None else fn_node)
    prov = _apply_exit_protocol_patch_scan(_EMPTY_SUPPRESS_PROVENANCE, patch_scan)
    if eff_mod is not None:
        mod_seed = _get_module_suppress_seed(
            eff_mod,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        if not mod_seed.has_any_positive_or_suppressor:
            prov = mod_seed.full_forward_prov
        elif fn_node is not eff_mod:
            prov = _resolve_top_level_ancestor_seed(
                fn_node,
                eff_mod,
                enclosing_nodes,
                mod_seed,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
        elif u_start <= 0:
            prov = mod_seed.full_forward_prov
        else:
            prov = _advance_prov_across_stmts_until_target(
                eff_mod.body,
                prov,
                None,
                u_start,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
        prov = _apply_exit_protocol_patch_scan(prov, patch_scan)

    for idx_enc, enc in enumerate(enclosing_nodes):
        if isinstance(enc, ast.ClassDef):
            # Per PEP 227, enclosing class attributes/methods do not bind lexical
            # names inside nested methods or inner classes.
            cls_ctx = _ProvenanceWalkContext(
                in_branch=True,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            prov = _apply_class_def_external_effects(enc, prov, cls_ctx).clear_name(
                enc.name
            )
            continue
        prov = _clear_callable_scope_locals_and_invalidate(prov, enc)
        enc_body = getattr(enc, "body", None)
        if isinstance(enc_body, (list, tuple)):
            next_target = (
                enclosing_nodes[idx_enc + 1]
                if idx_enc + 1 < len(enclosing_nodes)
                else fn_node
            )
            next_target_lineno = int(getattr(next_target, "lineno", 0) or 0)
            prov = _advance_prov_across_stmts_until_target(
                enc_body,
                prov,
                next_target,
                next_target_lineno,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )

    if fn_node is not eff_mod:
        prov = _clear_callable_scope_locals_and_invalidate(
            prov,
            fn_node,
            explicit_locals=(scope_ctx.fn_locals if scope_ctx is not None else None),
        )
        fn_body = getattr(fn_node, "body", None)
        if isinstance(fn_body, (list, tuple)) and u_start > 0:
            prov = _advance_prov_across_stmts_until_target(
                fn_body,
                prov,
                None,
                u_start,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
    else:
        fn_args = getattr(fn_node, "args", None)
        if isinstance(fn_args, ast.arguments):
            for p_name in _extract_arg_names(fn_args):
                prov = prov.clear_name(p_name)
        for tp_name in _extract_node_type_param_names(fn_node):
            prov = prov.clear_name(tp_name)

    # Soundness check: any binding to 'open', 'builtins', or '__builtins__' anywhere
    # in the module, enclosing scopes, or current scope marks it shadowed unless
    # already proven as an unshadowed non-suppressor or stdlib module in prov.
    scope_chain: Tuple[ast.AST, ...] = (
        *((eff_mod,) if eff_mod is not None else ()),
        *enclosing_nodes,
        *((fn_node,) if fn_node is not eff_mod else ()),
    )
    bound_in_chain: Set[str] = set()
    has_star = prov.has_star_import
    if scope_ctx is not None:
        bound_in_chain.update(scope_ctx.shadowed_names | scope_ctx.fn_locals)
        if scope_ctx.has_star_import:
            has_star = True
    if not has_star and eff_mod is not None and _node_has_star_import(eff_mod):
        has_star = True
    for sc_node in scope_chain:
        if isinstance(sc_node, ast.ClassDef) and sc_node is not fn_node:
            continue
        bound_in_chain.update(
            _extract_scope_local_names(
                sc_node,
                module_tree=eff_mod if sc_node is fn_node else None,
                enclosing_scopes=enclosing_nodes if sc_node is fn_node else (),
                include_self_name=(sc_node is fn_node),
            )
        )
        if (
            not has_star
            and sc_node is not eff_mod
            and _scope_has_direct_star_import(sc_node)
        ):
            has_star = True
    if has_star and not prov.has_star_import:
        prov = prov.with_star_import()
    proven_stdlib_aliases = {alias for alias, _ in prov.stdlib_modules}
    fn_own_locals = (
        _extract_single_node_local_names(fn_node, include_self_name=False)
        if (fn_node is not eff_mod and u_start <= 0)
        else set()
    )
    shadowed_b = {
        b_name
        for b_name in (bound_in_chain & _SHADOWABLE_BUILTIN_CTX_NAMES)
        if b_name in fn_own_locals
        or (b_name not in prov.non_suppressors and b_name not in proven_stdlib_aliases)
    }
    if shadowed_b:
        prov = prov.with_shadowed_builtins(shadowed_b)
    return _apply_exit_protocol_patch_scan(prov, patch_scan)


def _seed_scope_suppress_aliases(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    u_start: int = 0,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Seeds initial context-manager suppression and shadowing provenance for fn_node."""
    eff_mod = (
        mod_tree
        if isinstance(mod_tree, ast.Module)
        else (fn_node if isinstance(fn_node, ast.Module) else None)
    )
    enclosing_nodes: Tuple[ast.AST, ...] = (
        tuple(scope_ctx.enclosing_scopes)
        if scope_ctx is not None and scope_ctx.enclosing_scopes
        else (
            tuple(
                enc_n
                for _, _, enc_n in _find_enclosing_function_chain(
                    eff_mod,
                    int(getattr(fn_node, "lineno", 0) or 0),
                    int(getattr(fn_node, "end_lineno", 0) or 0),
                    getattr(fn_node, "col_offset", None),
                    getattr(fn_node, "end_col_offset", None),
                )
                if enc_n is not fn_node
            )
            if eff_mod is not None and fn_node is not eff_mod
            else ()
        )
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *((eff_mod,) if eff_mod is not None and eff_mod is not fn_node else ()),
        *enclosing_nodes,
    )
    extra_f = frozenset(extra_non_suppressing_ctx_managers)
    strict_w = bool(strict_with_suppression)
    can_cache = not _is_ephemeral_unit_ast(fn_node) and all(
        not _is_ephemeral_unit_ast(n) for n in aux_nodes
    )
    with analysis_session():
        if can_cache and eff_mod is not None and fn_node is not eff_mod:
            if not _tree_contains_scope_node(eff_mod, fn_node) or any(
                n is not eff_mod and not _tree_contains_scope_node(eff_mod, n)
                for n in aux_nodes
            ):
                can_cache = False
        if not can_cache:
            return _seed_scope_suppress_aliases_uncached(
                fn_node,
                eff_mod,
                enclosing_nodes,
                scope_ctx,
                int(u_start),
                extra_non_suppressing_ctx_managers=extra_f,
                strict_with_suppression=strict_w,
            )
        seed_params: Tuple[Any, ...] = (
            (int(u_start), strict_w, extra_f)
            if (strict_w or extra_f)
            else (int(u_start),)
        )
        cache_key = _make_trusted_scope_cache_key(
            fn_node,
            "suppress_seed",
            params=seed_params,
            aux_ast_nodes=aux_nodes,
        )
        return _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            cache_key,
            lambda: _seed_scope_suppress_aliases_uncached(
                fn_node,
                eff_mod,
                enclosing_nodes,
                scope_ctx,
                int(u_start),
                extra_non_suppressing_ctx_managers=extra_f,
                strict_with_suppression=strict_w,
            ),
        )
