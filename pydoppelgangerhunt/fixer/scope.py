"""Lexical variable scope analysis, AST visitor, and control flow hazard tracking."""

from __future__ import annotations

import ast
import contextlib
import logging
import re
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    combine_closure_strictness,
    normalize_heuristic_ctx_managers,
    normalize_non_suppressing_ctx_managers,
    normalize_wildcard_ctx_manager_matches,
    record_heuristic_non_suppressing_ctx_manager,
    record_wildcard_ctx_manager_match,
    resolve_closure_strictness_mode,
    resolve_pair_closure_policy,
    resolve_unit_closure_policy,
    should_warn_closure_event,
    validate_closure_strictness,
)
from pydoppelgangerhunt.fixer._df_block_flow import (  # pylint: disable=unused-import
    _analyze_block_assignment,
    _block_terminates,
)
from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_arg_names as _extract_arg_names_tuple,
    _extract_non_name_bound_names,
    _extract_type_alias_name,
    _is_branching_cfg_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _ephemeral_unit_ast_context,
    _get_or_parse_tree,
    _register_tree_source_lines,
    _split_source_lines_stripped,
    analysis_session,
)
from pydoppelgangerhunt.fixer._df_scope_ctx import _get_fn_scope_context
from pydoppelgangerhunt.fixer._df_cell_hazards import (
    _UnitClosureHazardContext,
    _evaluate_severed_closure_cell_hazard,
)
from pydoppelgangerhunt.fixer._df_cell_traversal import (
    _find_enclosing_chain_definite_pre_bound,
)
from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=unused-import
    _extract_chain_nonlocals_and_outer_globals,
    _extract_comprehension_walrus_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    _extract_strict_fn_local_names,
    _get_tree_scope_intervals,
    _is_irrefutable_case,
    _is_irrefutable_pattern,
    _is_span_in_scope_header,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
    _resolve_immediate_scope_in_fn,
    iter_nested_scope_header_nodes,
)
from pydoppelgangerhunt.fixer._df_exit_protocol import scan_exit_protocol_patches
from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names
from pydoppelgangerhunt.fixer._df_module_seed import (
    _apply_exit_protocol_patch_scan,
    _seed_scope_suppress_aliases,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _find_enclosing_function_chain,
    _has_invalid_lexical_scope_declarations,
    _resolve_unit_col_bounds,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _UnitLiveInVisitor,
    _is_ast_node_within_unit_bounds,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (  # pylint: disable=unused-import
    _EMPTY_SUPPRESS_PROVENANCE,
    _SuppressProvenance,
)
from pydoppelgangerhunt.fixer._df_with_rules import (  # pylint: disable=unused-import
    _extract_deleted_names,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.fixer._scope_source import (  # pylint: disable=unused-import
    SCOPE_SCHEMA_VERSION,
    _get_tree_sha256_digest,
    _has_out_of_bounds_unit_cols,
    _infer_missing_col_from_line_ast,
    _is_ordered_contiguous_source_match,
    _is_subroutine_or_nested_expr_unit,
    _make_empty_scope_result,
    _matched_ast_cols_compatible,
    _parse_unit_scope_ast,
    _placeholder_tree_cols_out_of_bounds,
    _prepare_unit_scope_source,
    _strip_comments_from_lines,
    _strip_inline_comment_line,
    _unit_cols,
    resolve_unit_inferred_col_bounds,
)
from pydoppelgangerhunt.fixer._scope_visitor import (  # pylint: disable=unused-import
    BUILTIN_NAMES,
    _SYNTHETIC_WRAPPER_NAME,
    _ScopeVisitor,
    _block_contains_yield,
    _extract_arg_names,
    _extract_type_param_names,
    _is_mangled_name,
    _normalize_receiver_attr_name,
    _normalize_receiver_attrs,
    _unfold_receiver_attribute,
)
from pydoppelgangerhunt.fixer.source import (
    _SUBROUTINE_KINDS,
    _WHOLE_UNIT_KINDS,
    _is_sliced_excerpt_for_file,
    _load_unit_file_text,
    find_enclosing_function_is_async,
    is_subroutine_unit,
    parse_unit_coord,
    resolve_unit_declared_outputs,
    split_source_lines,
)
from pydoppelgangerhunt.source_lines import is_non_string_sequence
from pydoppelgangerhunt.unit_boundary import (
    normalize_unit_boundary,
    resolve_unit_source_buffer,
)

__all__ = [
    "BUILTIN_NAMES",
    "SCOPE_SCHEMA_VERSION",
    "analyze_unit_variable_scope",
    "ensure_unit_scope_closure_mode",
    "inspect_single_unit_scope",
    "is_subroutine_unit",
    "scope_matches_closure_mode",
]

logger = logging.getLogger(__name__)

BLOCKING_CONTROL_FLOW_HAZARDS: FrozenSet[str] = frozenset(
    {
        "naked_break",
        "naked_continue",
        "embedded_return",
        "type_param_closure_capture",
        "pathological_ast",
        "deleted_output",
        "severed_closure_cell",
        "stale_source_code",
        "unavailable_source_for_closure_mode",
        "unavailable_source_placeholder",
        "invalid_unit_columns",
    }
)


@contextlib.contextmanager
def _capture_unit_ctx_managers(
    heur_out: Set[str],
    wc_out: Set[Tuple[str, str]],
) -> Iterator[None]:
    """Collects heuristic and wildcard context-manager matches into caller accumulator sets."""
    with collect_heuristic_ctx_managers() as heur, collect_wildcard_ctx_managers() as wc:
        try:
            yield
        finally:
            heur_out.update(heur)
            wc_out.update(wc)


def _determine_binding_kind(
    has_instance_binding: bool,
    has_class_binding: bool,
) -> Optional[str]:
    """Determines the scope binding category from instance and class binding flags."""
    if has_instance_binding and has_class_binding:
        return "mixed"
    if has_instance_binding:
        return "instance"
    if has_class_binding:
        return "class"
    return None


def _normalize_receiver_order(
    inputs: List[str],
    has_instance_binding: bool,
    has_class_binding: bool,
    primary_receiver: Optional[str] = None,
) -> None:
    """Ensures self or cls is positioned as the primary receiver parameter in inputs."""
    def _bring_to_front(name: str) -> None:
        if name in inputs:
            inputs.remove(name)
            inputs.insert(0, name)

    if has_instance_binding and has_class_binding:
        first = primary_receiver or "self"
        second = "cls" if first == "self" else "self"
        _bring_to_front(second)
        _bring_to_front(first)
    elif has_instance_binding or has_class_binding:
        default_rec = "self" if has_instance_binding else "cls"
        target = primary_receiver if (primary_receiver and primary_receiver in inputs) else default_rec
        _bring_to_front(target)



def _rank_param_kind(
    var_name: str,
    param_map: Dict[str, str],
    receiver_names: Optional[Sequence[str]] = None,
) -> int:
    """Returns canonical parameter sorting rank: receiver=0, pos=1, vararg=2, kwonly=3, kwarg=4."""
    clean = var_name.lstrip("*")
    valid_receivers = (
        tuple(receiver_names)
        if receiver_names is not None
        else ("self", "cls", "this", "klass")
    )
    if clean in valid_receivers:
        return 0
    kind = param_map.get(clean, "pos")
    if var_name.startswith("**") or kind == "kwarg":
        return 4
    if kind == "kwonly":
        return 3
    if var_name.startswith("*") or kind == "vararg":
        return 2
    return 1


# INVARIANT (SHARED CFG LOWERING CONTRACT — ENCLOSING MAY-BOUND SCAN):
# This walker computes *may-bound* names before the unit across module, class, and
# function scopes to contrast against *must-bound* definite assignment from
# `_df_cell_traversal._find_enclosing_chain_definite_pre_bound`. Branching and
# non-Name binding classification MUST route through `_df_cfg._is_branching_cfg_node`
# and `_df_cfg._extract_non_name_bound_names` rather than duplicating syntax checks.
def _find_enclosing_pre_bound_names(
    mod_tree: ast.AST,
    s_line: int,
    e_line: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[Set[str], bool]:
    """Finds names may-bound before s_line in enclosing module/class/function scopes.

    Unlike ``_find_enclosing_chain_definite_pre_bound`` (which computes must-bound
    definite assignment across every control-flow path in enclosing functions), this
    function collects all names that have a lexical binding on at least one path prior
    to the unit across module, class, and function scopes, and reports whether the unit
    is enclosed inside a loop. Comparing the may-bound set against the must-bound set
    identifies variables that are only conditionally initialized before the unit.
    """
    bound_before: Set[str] = set()
    in_loop = False

    def _is_before_unit(c_line: Optional[int], c_col: Optional[int]) -> bool:
        if not isinstance(c_line, int):
            return False
        if c_line < s_line:
            return True
        return (
            c_line == s_line
            and u_start_col is not None
            and isinstance(c_col, int)
            and c_col < u_start_col
        )

    def _node_contains_or_encloses_unit(node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node,
            s_line,
            e_line,
            u_end_col=u_end_col,
            u_start_col=u_start_col,
        ) or _node_strictly_encloses_bounds(
            node,
            s_line,
            e_line,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )

    intervals = [
        (earliest_start, n_end, span, node)
        for earliest_start, n_end, span, node in _get_tree_scope_intervals(mod_tree)
        if earliest_start <= s_line
        and e_line <= n_end
        and (
            _node_strictly_encloses_bounds(
                node,
                s_line,
                e_line,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
            or (
                u_start_col is None
                and u_end_col is None
                and (
                    earliest_start < s_line
                    or n_end > e_line
                    or earliest_start == n_end
                )
            )
        )
        and not _is_span_in_scope_header(
            node,
            s_line,
            e_line,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
    ]
    intervals.sort(key=lambda item: (item[0], -item[1], -item[2]))
    innermost_scope = intervals[-1][3] if intervals else mod_tree
    active_scopes: List[ast.AST] = []
    if isinstance(mod_tree, ast.Module):
        active_scopes.append(mod_tree)
    for _estart, _nend, _span, s_node in intervals:
        if isinstance(s_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            active_scopes.append(s_node)
        elif isinstance(s_node, ast.ClassDef) and s_node is innermost_scope:
            active_scopes.append(s_node)

    for scope_node in active_scopes:
        is_class_scope = isinstance(scope_node, ast.ClassDef)
        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound_before.difference_update(
                _extract_strict_fn_local_names(scope_node)
            )
            bound_before.update(_extract_arg_names_tuple(scope_node.args))
        stack: List[Tuple[ast.AST, bool]] = [
            (stmt, False) for stmt in reversed(getattr(scope_node, "body", []))
        ]
        while stack:
            curr, is_branched = stack.pop()
            c_start = getattr(curr, "lineno", None)
            c_col = getattr(curr, "col_offset", None)
            c_end = getattr(curr, "end_lineno", c_start)
            c_end_col = getattr(curr, "end_col_offset", c_col)
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                estart, ecol = _node_earliest_start_pos(curr)
                if _is_before_unit(estart, ecol) and (
                    (
                        curr in active_scopes
                        and not isinstance(curr, ast.ClassDef)
                    )
                    or _is_before_unit(c_end, c_end_col)
                ):
                    bound_before.add(curr.name)
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(curr, ast.Lambda):
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(
                curr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if _is_before_unit(c_start, c_col) and not is_class_scope:
                    bound_before.update(_extract_comprehension_walrus_names(curr))
                continue
            alias_name = _extract_type_alias_name(curr)
            if alias_name is not None:
                if _is_before_unit(c_start, c_col):
                    bound_before.add(alias_name)
                continue
            if (
                isinstance(curr, ast.AnnAssign)
                and curr.value is None
                and isinstance(curr.target, ast.Name)
            ):
                stack.append((curr.annotation, is_branched))
                continue
            if isinstance(
                curr,
                (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr),
            ):
                rhs_val = curr.value
                if rhs_val is not None and _node_contains_or_encloses_unit(
                    rhs_val
                ):
                    stack.append((rhs_val, is_branched))
                    if isinstance(curr, ast.AnnAssign):
                        stack.append((curr.annotation, is_branched))
                    elif isinstance(curr, ast.AugAssign) and not isinstance(
                        curr.target, ast.Name
                    ):
                        stack.append((curr.target, is_branched))
                    continue
            if isinstance(curr, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(
                    curr, (ast.For, ast.AsyncFor)
                ) and _node_contains_or_encloses_unit(curr.iter):
                    stack.append((curr.iter, is_branched))
                    continue
                if _node_strictly_encloses_bounds(
                    curr,
                    s_line,
                    e_line,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                ):
                    in_loop = True
            if _is_before_unit(c_start, c_col):
                if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Store):
                    bound_before.add(curr.id)
                elif (
                    isinstance(curr, ast.Name)
                    and isinstance(curr.ctx, ast.Del)
                    and not is_branched
                ):
                    bound_before.discard(curr.id)
                elif (
                    isinstance(curr, ast.ExceptHandler)
                    and isinstance(curr.name, str)
                    and _node_strictly_encloses_bounds(
                        curr,
                        s_line,
                        e_line,
                        u_start_col=u_start_col,
                        u_end_col=u_end_col,
                    )
                ):
                    bound_before.add(curr.name)
                else:
                    bound_before.update(_extract_non_name_bound_names(curr))
            child_branched = is_branched or _is_branching_cfg_node(curr)
            stack.extend(
                (ch, child_branched)
                for ch in reversed(list(ast.iter_child_nodes(curr)))
            )
    return bound_before, in_loop


def _inspect_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_tree: Optional[ast.AST] = None,
    *,
    allow_disk_read: bool = True,
) -> Dict[str, Any]:
    with analysis_session():
        return _inspect_unit_scope_impl(
            unit,
            repo_root=repo_root,
            file_tree=file_tree,
            allow_disk_read=allow_disk_read,
        )


def _check_type_param_closure_capture(
    unit: Dict[str, Any],
    tree: ast.AST,
    visitor: _ScopeVisitor,
    *,
    inferred_s_col: Optional[int],
    file_tree: Optional[ast.AST],
    get_full_source: Callable[[], Optional[str]],
    hazards: List[str],
) -> bool:
    """Checks PEP 695 type-parameter closure capture hazards and returns has_class_scoped_type_params."""
    s_line = parse_unit_coord(unit, "start", default=1)
    e_line = parse_unit_coord(unit, "end", default=s_line)
    u_col = inferred_s_col
    top_fn = (
        tree.body[0]
        if isinstance(tree, ast.Module)
        and len(tree.body) == 1
        and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    top_fn_name = getattr(top_fn, "name", None) if top_fn is not None else None
    top_fn_count = (
        sum(
            1
            for n in ast.walk(top_fn)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if top_fn is not None
        else None
    )
    parsed_tree = file_tree
    if parsed_tree is None:
        full_src = get_full_source()
        if full_src:
            parsed_tree = _get_or_parse_tree(full_src)
    enclosing_fn_locals: Set[str] = set()
    enclosing_cls_locals: Set[str] = set()
    has_enclosing_fn = False
    has_enclosing_cls = False
    prev_enclosed_cls_name: Optional[str] = None
    if parsed_tree is not None:
        full_src_for_lines = get_full_source()
        mod_src_lines: Optional[Tuple[str, ...]] = (
            _register_tree_source_lines(parsed_tree, source_text=full_src_for_lines)
            if full_src_for_lines
            else None
        )
        matched_self = False
        for earliest_start, n_end, _span, scope_node in _get_tree_scope_intervals(
            parsed_tree, source_lines=mod_src_lines
        ):
            if earliest_start > s_line or e_line > n_end:
                continue
            if isinstance(scope_node, ast.ClassDef):
                if not has_enclosing_cls:
                    has_enclosing_cls = True
                    prev_enclosed_cls_name = getattr(scope_node, "name", None)
                    enclosing_cls_locals.update(
                        _extract_single_node_local_names(
                            scope_node, include_globals=False
                        )
                    )
                else:
                    if isinstance(prev_enclosed_cls_name, str):
                        enclosing_cls_locals.add(prev_enclosed_cls_name)
                    prev_enclosed_cls_name = getattr(scope_node, "name", None)
                    enclosing_cls_locals.update(_extract_type_param_names(scope_node))
            elif isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    not matched_self
                    and s_line
                    in (
                        earliest_start,
                        getattr(scope_node, "lineno", earliest_start),
                    )
                    and (
                        e_line == n_end
                        or top_fn_name is not None
                        or u_col is not None
                    )
                    and (
                        u_col is None
                        or getattr(scope_node, "col_offset", 0) == u_col
                    )
                    and (
                        top_fn_name is None
                        or getattr(scope_node, "name", None) == top_fn_name
                    )
                    and (
                        top_fn_count is None
                        or sum(
                            1
                            for n in ast.walk(scope_node)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        )
                        == top_fn_count
                    )
                ):
                    matched_self = True
                    continue
                if (
                    earliest_start < s_line
                    or n_end > e_line
                    or (u_col is None and matched_self)
                    or (
                        u_col is not None
                        and getattr(scope_node, "col_offset", 0) < u_col
                    )
                ):
                    has_enclosing_fn = True
                    enclosing_fn_locals.update(
                        _extract_single_node_local_names(
                            scope_node, include_globals=False
                        )
                    )
    has_class_scoped_type_params = False
    if has_enclosing_fn and any(
        fv in enclosing_fn_locals for fv in visitor.type_param_free_vars
    ):
        hazards.append("type_param_closure_capture")
    elif (
        not has_enclosing_fn
        and not has_enclosing_cls
        and unit.get("kind") == "closure"
    ):
        hazards.append("type_param_closure_capture")
    if has_enclosing_cls and any(
        fv in enclosing_cls_locals for fv in visitor.type_param_free_vars
    ):
        has_class_scoped_type_params = True
        if (
            has_enclosing_fn or unit.get("kind") == "closure"
        ) and "type_param_closure_capture" not in hazards:
            hazards.append("type_param_closure_capture")
    elif not has_enclosing_cls and (
        unit.get("enclosing_class")
        or unit.get("kind") in ("method", "async_method")
    ):
        has_class_scoped_type_params = True
    return has_class_scoped_type_params


def _seed_unit_with_suppress_provenance(
    tree: ast.AST,
    candidate_stmts: Sequence[ast.stmt],
    visitor: _ScopeVisitor,
    *,
    mod_tree: Optional[ast.AST],
    full_mod_lines: Optional[Tuple[str, ...]],
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    is_subroutine: bool,
    u_start: int,
    u_end: int,
    resolved_s_col: Optional[int],
    extra_non_sup: FrozenSet[str],
    strict_with_flag: bool,
) -> _SuppressProvenance:
    """Seeds context-manager suppression provenance for units containing With/AsyncWith blocks."""
    initial_sup_prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE
    has_any_with = any(
        isinstance(n, (ast.With, ast.AsyncWith))
        for st in candidate_stmts
        for n in ast.walk(st)
    )
    if mod_tree is not None and has_any_with:
        unit_fn_node: Optional[ast.AST] = None
        if not is_subroutine:
            enclosing_set = {f for _, _, f in enclosing_chain}
            best_span: Optional[int] = None
            for earliest_start, n_end, span_sz, sc_node in _get_tree_scope_intervals(
                mod_tree, source_lines=full_mod_lines
            ):
                if not (
                    sc_node not in enclosing_set
                    and isinstance(sc_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and earliest_start <= u_start <= getattr(sc_node, "lineno", u_start)
                    and n_end == u_end
                ):
                    continue
                if (
                    resolved_s_col is not None
                    and not getattr(sc_node, "decorator_list", None)
                    and getattr(sc_node, "lineno", None) == u_start
                    and getattr(sc_node, "col_offset", resolved_s_col) != resolved_s_col
                ):
                    continue
                if best_span is None or span_sz < best_span:
                    unit_fn_node = sc_node
                    best_span = span_sz
        if unit_fn_node is not None:
            sup_fn = unit_fn_node
            sup_enclosing_scopes = [f for _, _, f in enclosing_chain]
        else:
            sup_fn = enclosing_chain[-1][2] if enclosing_chain else mod_tree
            sup_enclosing_scopes = [f for _, _, f in enclosing_chain[:-1]]
        sup_scope_ctx = (
            _get_fn_scope_context(
                sup_fn,
                mod_tree=mod_tree,
                enclosing_scopes=sup_enclosing_scopes,
            )
            if isinstance(sup_fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )
        initial_sup_prov = _seed_scope_suppress_aliases(
            sup_fn,
            mod_tree=mod_tree,
            u_start=u_start,
            scope_ctx=sup_scope_ctx,
            extra_non_suppressing_ctx_managers=extra_non_sup,
            strict_with_suppression=strict_with_flag,
        )
    elif has_any_with:
        is_synthetic_wrapper = (
            isinstance(tree, ast.Module)
            and len(tree.body) == 1
            and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            and tree.body[0].name == _SYNTHETIC_WRAPPER_NAME
        )
        scan_tree: ast.AST = (
            tree
            if isinstance(tree, ast.Module) and not is_synthetic_wrapper
            else ast.Module(body=list(candidate_stmts), type_ignores=[])
        )
        with _ephemeral_unit_ast_context(scan_tree):
            initial_sup_prov = _apply_exit_protocol_patch_scan(
                initial_sup_prov,
                scan_exit_protocol_patches(scan_tree),
            )
    for p_name in set(visitor.params) | visitor.type_param_names:
        initial_sup_prov = initial_sup_prov.clear_name(p_name)
    return initial_sup_prov


def _analyze_subroutine_conditional_live_ins(
    unit: Dict[str, Any],
    candidate_stmts: Sequence[ast.stmt],
    visitor: _ScopeVisitor,
    *,
    is_subroutine: bool,
    def_assigned: Set[str],
    cond_assigned: Set[str],
    mod_tree: Optional[ast.AST],
    full_mod_lines: Optional[Tuple[str, ...]],
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    resolved_s_col: Optional[int],
    resolved_e_col: Optional[int],
    strict_with_flag: bool,
    extra_non_sup: FrozenSet[str],
    get_full_source: Callable[[], Optional[str]],
    hazards: List[str],
    closure_hazard_reasons: List[str],
    unit_heuristic_mgrs: Set[str],
    unit_wildcard_mgrs: Set[Tuple[str, str]],
) -> None:
    """Evaluates subroutine conditional live-ins and write-only conditional candidates against enclosing pre-bounds."""
    if not is_subroutine:
        return
    live_in_visitor = _UnitLiveInVisitor(1, 10**9, include_annotations=False)
    for stmt in candidate_stmts:
        live_in_visitor.visit(stmt)
    excluded_cond_vars = set(visitor.params) | set(visitor.globals) | BUILTIN_NAMES
    cond_live_ins = [
        v
        for v in sorted(live_in_visitor.live_ins)
        if v in visitor.loads
        and v not in excluded_cond_vars
        and v not in visitor.read_before_write
    ]
    cond_escaping_cands = [
        v
        for v in sorted(
            (cond_assigned - def_assigned)
            | set(cond_live_ins)
            | (live_in_visitor.live_ins & set(visitor.stores))
        )
        if v not in excluded_cond_vars
    ]
    if not (cond_live_ins or cond_escaping_cands):
        return
    pre_bound: Set[str] = set()
    rejected_escaping_vars: Set[str] = set()
    if mod_tree is not None and full_mod_lines is not None:
        _register_tree_source_lines(mod_tree, full_mod_lines)
    if mod_tree is not None and u_start > 0:
        u_start_col, u_end_col = resolved_s_col, resolved_e_col
        pre_bound, _ = _find_enclosing_pre_bound_names(
            mod_tree,
            u_start,
            u_end,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
        if enclosing_chain:
            innermost_stmt_fn, enc_nonlocals, enc_outer_globals = (
                _extract_chain_nonlocals_and_outer_globals(
                    enclosing_chain,
                    u_start,
                    u_end,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                )
            )
            in_enclosing_class_body = isinstance(
                _resolve_immediate_scope_in_fn(
                    innermost_stmt_fn,
                    u_start,
                    u_end,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                ),
                ast.ClassDef,
            )
            enc_escaping = enc_nonlocals | enc_outer_globals
            with _capture_unit_ctx_managers(unit_heuristic_mgrs, unit_wildcard_mgrs):
                def_pre_bound = _find_enclosing_chain_definite_pre_bound(
                    enclosing_chain,
                    u_start,
                    u_end,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                    mod_tree=mod_tree,
                    strict_with_suppression=strict_with_flag,
                    extra_non_suppressing_ctx_managers=extra_non_sup,
                )
            write_only_cond_cands = {
                v
                for v in cond_escaping_cands
                if v not in enc_escaping
                and v not in visitor.loads
                and v not in live_in_visitor.live_ins
                and v in pre_bound
                and v in def_pre_bound
            }
            live_write_only_cond: Set[str] = set()
            if write_only_cond_cands:
                pre_outs_decl = resolve_unit_declared_outputs(unit)
                if is_non_string_sequence(pre_outs_decl) or isinstance(
                    pre_outs_decl, AbstractSet
                ):
                    live_write_only_cond = write_only_cond_cands & set(pre_outs_decl)
                else:
                    full_src_text = get_full_source() or ""
                    ds_unit: Dict[str, Any] = {
                        **unit,
                        "start": u_start,
                        "end": u_end,
                    }
                    if u_start_col is not None:
                        ds_unit["start_col"] = u_start_col
                    if u_end_col is not None:
                        ds_unit["end_col"] = u_end_col
                    # Always include pre-unit closures when checking whether a
                    # write-only conditional variable is live downstream: pruning
                    # a required output in lenient mode would break callers.
                    ds_live = collect_downstream_read_names(
                        full_src_text,
                        ds_unit,
                        candidates=write_only_cond_cands,
                        tree=mod_tree,
                        closure_strictness="strict",
                        source_digest=(
                            None
                            if full_src_text
                            else _get_tree_sha256_digest(mod_tree)
                        ),
                    )
                    live_write_only_cond = (
                        set(write_only_cond_cands)
                        if ds_live is None
                        else (write_only_cond_cands & ds_live)
                    )
            for v in cond_escaping_cands:
                if v in enc_escaping:
                    if v in enc_nonlocals and v in pre_bound and v in def_pre_bound:
                        if v not in visitor.loads:
                            visitor.loads.append(v)
                        if v not in visitor.read_before_write:
                            visitor.read_before_write.append(v)
                    else:
                        rejected_escaping_vars.add(v)
                        if v in visitor.read_before_write and v not in def_pre_bound:
                            visitor.read_before_write.remove(v)
                        if "severed_closure_cell" not in hazards:
                            hazards.append("severed_closure_cell")
                        sub_r = (
                            "global_escape"
                            if v in enc_outer_globals
                            else "cell_severing"
                        )
                        if sub_r not in closure_hazard_reasons:
                            closure_hazard_reasons.append(sub_r)
                elif (
                    not in_enclosing_class_body
                    and not visitor.has_yield
                    and v in pre_bound
                    and v in def_pre_bound
                    and (
                        v in visitor.loads
                        or v in live_in_visitor.live_ins
                        or v in live_write_only_cond
                    )
                ):
                    if v not in visitor.loads:
                        visitor.loads.append(v)
                    if v not in visitor.read_before_write:
                        visitor.read_before_write.append(v)
                elif in_enclosing_class_body or v not in def_pre_bound:
                    rejected_escaping_vars.add(v)
            for v in cond_live_ins:
                if v not in def_pre_bound or (
                    in_enclosing_class_body and v not in enc_escaping
                ):
                    rejected_escaping_vars.add(v)
                    if v in pre_bound:
                        if "severed_closure_cell" not in hazards:
                            hazards.append("severed_closure_cell")
                        if "conditional_live_in" not in closure_hazard_reasons:
                            closure_hazard_reasons.append("conditional_live_in")
    has_enclosing_context = mod_tree is not None and (
        u_start > 1 or bool(_get_tree_scope_intervals(mod_tree))
    )
    for v in cond_live_ins:
        if (
            v not in rejected_escaping_vars
            and (
                v in pre_bound
                or (not has_enclosing_context and v in def_assigned)
            )
            and v not in visitor.read_before_write
        ):
            visitor.read_before_write.append(v)


def _resolve_unit_outputs(
    unit: Dict[str, Any],
    visitor: _ScopeVisitor,
    *,
    is_subroutine: bool,
    inputs: Sequence[str],
    def_assigned: Set[str],
    cond_assigned: Set[str],
    all_deleted: Set[str],
    file_tree: Optional[ast.AST],
    resolved_strictness: str,
    get_full_source: Callable[[], Optional[str]],
) -> Tuple[List[str], bool]:
    """Computes ordered unit outputs and ambiguity flag from declared outputs, returns, or stores."""
    pre_outs = resolve_unit_declared_outputs(unit)
    outputs_ambiguous = False
    alias_cands = visitor.type_aliases & (set(visitor.stores) | set(visitor.nonlocals))
    downstream_aliases: Set[str] = set()
    if is_subroutine and alias_cands:
        full_src = get_full_source()
        if full_src:
            ds_reads = collect_downstream_read_names(
                full_src,
                unit,
                candidates=alias_cands,
                tree=file_tree,
                closure_strictness=resolved_strictness,
                unit_inputs=set(inputs),
            )
            downstream_aliases = set(alias_cands) if ds_reads is None else ds_reads
        else:
            downstream_aliases = set(alias_cands)
    excluded_local_defs = (visitor.type_aliases - downstream_aliases) | set(
        visitor.imported_names
    )
    uncond_deleted = all_deleted - (def_assigned | cond_assigned)

    if is_subroutine and is_non_string_sequence(pre_outs):
        outputs = list(dict.fromkeys(pre_outs))
    elif is_subroutine and isinstance(pre_outs, AbstractSet):
        if len(pre_outs) > 1:
            logger.debug(
                "Rejecting set-valued outputs (%s): unordered set cannot establish "
                "positional order",
                pre_outs,
            )
            outputs = []
            outputs_ambiguous = True
        else:
            outputs = list(pre_outs)
    elif visitor.returns:
        outputs = list(visitor.returns)
    elif is_subroutine:
        seen: Set[str] = set()
        outputs = []
        for v in visitor.stores:
            if (
                v not in visitor.globals
                and v not in BUILTIN_NAMES
                and v not in visitor.except_vars
                and v not in uncond_deleted
                and v not in excluded_local_defs
                and v not in seen
            ):
                seen.add(v)
                outputs.append(v)
    else:
        outputs = []

    for nl in visitor.nonlocals:
        if (
            nl in visitor.stores
            and nl not in outputs
            and nl not in visitor.except_vars
            and nl not in uncond_deleted
            and nl not in excluded_local_defs
        ):
            outputs.append(nl)
    return outputs, outputs_ambiguous


def _evaluate_unit_closure_and_lexical_hazards(
    unit: Dict[str, Any],
    tree: ast.AST,
    candidate_stmts: Sequence[ast.stmt],
    visitor: _ScopeVisitor,
    *,
    is_subroutine: bool,
    inputs: Sequence[str],
    outputs: Sequence[str],
    free_vars: Sequence[str],
    all_deleted: Set[str],
    def_assigned: Set[str],
    mod_tree: Optional[ast.AST],
    full_mod_lines: Optional[Tuple[str, ...]],
    enclosing_chain: Sequence[Tuple[int, int, ast.AST]],
    u_start: int,
    u_end: int,
    resolved_s_col: Optional[int],
    resolved_e_col: Optional[int],
    cols_resolved: bool,
    resolved_strictness: str,
    extra_non_sup: FrozenSet[str],
    get_full_source: Callable[[], Optional[str]],
    hazards: List[str],
    closure_hazard_reasons: List[str],
    unit_heuristic_mgrs: Set[str],
    unit_wildcard_mgrs: Set[Tuple[str, str]],
) -> None:
    """Validates lexical declarations and severed closure-cell hazards for a unit."""
    lex_stmts: Sequence[ast.stmt] = (
        tree.body
        if isinstance(tree, ast.Module)
        and len(tree.body) == 1
        and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        and getattr(tree.body[0], "name", None) != _SYNTHETIC_WRAPPER_NAME
        else candidate_stmts
    )
    if (
        "severed_closure_cell" not in hazards
        and _has_invalid_lexical_scope_declarations(
            lex_stmts,
            enclosing_chain=enclosing_chain,
            mod_tree=mod_tree,
            source_lines=full_mod_lines,
        )
    ):
        hazards.append("severed_closure_cell")
        if "invalid_lexical_declaration" not in closure_hazard_reasons:
            closure_hazard_reasons.append("invalid_lexical_declaration")

    if not (
        is_subroutine
        or unit.get("kind") in ("closure", "comprehension", "complex_expr")
    ):
        return
    if u_start > 0 and mod_tree is not None:
        if "severed_closure_cell" not in hazards:
            with collect_heuristic_ctx_managers() as cell_heur_mgrs, collect_wildcard_ctx_managers() as cell_wc_mgrs:
                hazard_res = _evaluate_severed_closure_cell_hazard(
                    _UnitClosureHazardContext(
                        candidate_stmts=candidate_stmts,
                        mod_tree=mod_tree,
                        u_start=u_start,
                        u_end=u_end,
                        u_start_col=resolved_s_col,
                        u_end_col=resolved_e_col,
                        is_subroutine=is_subroutine,
                        unit_kind=str(unit.get("kind") or ""),
                        inputs=inputs,
                        outputs=outputs,
                        stores=visitor.stores,
                        deleted_names=all_deleted,
                        unit_top_locals=visitor.unit_top_locals,
                        free_vars=free_vars,
                        nonlocals=visitor.nonlocals,
                        has_yield=visitor.has_yield,
                        is_async=visitor.is_async,
                        closure_strictness=resolved_strictness,
                        extra_non_suppressing_ctx_managers=extra_non_sup,
                        definite_stores=def_assigned,
                        cols_resolved=cols_resolved,
                        enclosing_chain=enclosing_chain,
                        lex_validated=True,
                    )
                )
            unit_heuristic_mgrs.update(cell_heur_mgrs)
            unit_wildcard_mgrs.update(cell_wc_mgrs)
            if hazard_res.has_hazard:
                hazards.append("severed_closure_cell")
                for r in hazard_res.reasons:
                    if r not in closure_hazard_reasons:
                        closure_hazard_reasons.append(r)
    elif "severed_closure_cell" not in hazards:
        if bool(get_full_source()) or (
            unit.get("kind") == "closure" and bool(free_vars)
        ):
            hazards.append("severed_closure_cell")
            if "unresolved_span_bounds" not in closure_hazard_reasons:
                closure_hazard_reasons.append("unresolved_span_bounds")
        elif not bool(unit.get("allow_snippet_only")):
            # Fail closed when enclosing module context is absent unless the
            # caller explicitly opted into isolated snippet-only inspection.
            hazards.append("severed_closure_cell")
            if "snippet_only_missing_enclosing_context" not in closure_hazard_reasons:
                closure_hazard_reasons.append(
                    "snippet_only_missing_enclosing_context"
                )


def _resolve_unit_effective_async(
    unit: Dict[str, Any],
    visitor: _ScopeVisitor,
    *,
    is_subroutine: bool,
    file_tree: Optional[ast.AST],
    get_full_source: Callable[[], Optional[str]],
) -> Optional[bool]:
    """Resolves whether a unit executes inside an async context."""
    is_sub_or_inner = is_subroutine or unit.get("kind") in (
        "comprehension",
        "complex_expr",
        "class",
        "data_table",
    )
    unit_is_async = unit.get("is_async")
    if unit_is_async is None:
        if is_sub_or_inner and not visitor.has_yield:
            unit_is_async = visitor.is_async
        else:
            s_line = parse_unit_coord(unit, "start", default=1)
            e_line = parse_unit_coord(unit, "end", default=s_line)
            full_src = get_full_source()
            if full_src:
                unit_is_async = find_enclosing_function_is_async(
                    full_src, s_line, e_line, tree=file_tree
                )
            else:
                unit_is_async = None
    if visitor.is_async or unit.get("kind") in ("async_function", "async_method"):
        return True
    if "is_async" in unit and unit.get("is_async") is not None:
        return bool(unit.get("is_async"))
    if visitor.has_yield:
        return bool(unit_is_async) if unit_is_async is not None else None
    if not is_sub_or_inner and unit.get("kind") not in (
        "function",
        "method",
        "async_function",
        "async_method",
    ):
        return bool(unit_is_async)
    return False


class _UnitEnclosingContext(NamedTuple):
    """Resolved module AST, source lines, column bounds, and enclosing function chain for a unit."""

    u_start: int
    u_end: int
    mod_tree: Optional[ast.AST]
    full_mod_lines: Optional[Tuple[str, ...]]
    resolved_s_col: Optional[int]
    resolved_e_col: Optional[int]
    cols_resolved: bool
    enclosing_chain: List[Tuple[int, int, ast.AST]]


def _resolve_unit_enclosing_context(
    unit: Dict[str, Any],
    candidate_stmts: Sequence[ast.stmt],
    *,
    inferred_s_col: Optional[int],
    inferred_e_col: Optional[int],
    file_tree: Optional[ast.AST],
    get_full_source: Callable[[], Optional[str]],
) -> _UnitEnclosingContext:
    """Resolves the enclosing module tree, line table, column bounds, and enclosing function chain."""
    u_start = parse_unit_coord(unit, "start", default=0)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    mod_tree: Optional[ast.AST] = file_tree
    full_mod_src = get_full_source()
    if mod_tree is None and u_start > 0 and full_mod_src:
        mod_tree = _get_or_parse_tree(full_mod_src)
    resolved_s_col, resolved_e_col = inferred_s_col, inferred_e_col
    cols_resolved = False
    enclosing_chain: List[Tuple[int, int, ast.AST]] = []
    if mod_tree is not None and full_mod_src:
        full_mod_lines: Optional[Tuple[str, ...]] = _register_tree_source_lines(
            mod_tree, source_text=full_mod_src
        )
    else:
        full_mod_lines = (
            _split_source_lines_stripped(full_mod_src) if full_mod_src else None
        )
    if mod_tree is not None and u_start > 0:
        resolved_s_col, resolved_e_col = _resolve_unit_col_bounds(
            candidate_stmts,
            mod_tree,
            u_start,
            u_end,
            resolved_s_col,
            resolved_e_col,
        )
        cols_resolved = True
        enclosing_chain = _find_enclosing_function_chain(
            mod_tree,
            u_start,
            u_end,
            resolved_s_col,
            resolved_e_col,
            source_lines=full_mod_lines,
        )
    return _UnitEnclosingContext(
        u_start=u_start,
        u_end=u_end,
        mod_tree=mod_tree,
        full_mod_lines=full_mod_lines,
        resolved_s_col=resolved_s_col,
        resolved_e_col=resolved_e_col,
        cols_resolved=cols_resolved,
        enclosing_chain=enclosing_chain,
    )


class _UnitInputsAndReceivers(NamedTuple):
    """Ordered helper inputs, free variables, and receiver-binding metadata for a unit."""

    free_vars: List[str]
    inputs: List[str]
    all_attrs: List[str]
    has_instance_binding: bool
    has_class_binding: bool
    binding_kind: Optional[str]
    has_receiver_access: bool


def _compute_unit_inputs_and_receivers(
    visitor: _ScopeVisitor,
    *,
    rec_param: Optional[str],
    rec_kind: Optional[str],
    inst_receivers: Set[str],
    class_receivers: Set[str],
    all_receivers: Set[str],
) -> _UnitInputsAndReceivers:
    """Computes ordered helper inputs, free variables, and instance/class receiver bindings."""
    is_class_rec = rec_kind == "class"
    free_vars = [
        name
        for name in visitor.loads
        if name not in visitor.params
        and (name not in visitor.stores or name in visitor.read_before_write)
        and name not in visitor.globals
        and name not in BUILTIN_NAMES
    ]
    inputs: List[str] = list(visitor.params)
    for fv in free_vars:
        if fv not in inputs:
            inputs.append(fv)
    for nl in visitor.nonlocals:
        if nl not in inputs:
            inputs.append(nl)
    param_map = {
        p["name"].lstrip("*"): p.get("kind", "pos") for p in visitor.param_details
    }
    inputs.sort(
        key=lambda v: _rank_param_kind(
            v, param_map, receiver_names=tuple(all_receivers)
        )
    )
    all_attrs = visitor.attrs_read + visitor.attrs_written
    has_instance_binding = any(
        r in visitor.loads
        or r in visitor.params
        or any(a.startswith(f"{r}.") for a in all_attrs)
        for r in inst_receivers
        if r in ("self", "this")
        or (rec_param and r == rec_param and rec_kind == "instance")
        or any(a.startswith(f"{r}.") for a in all_attrs)
    )
    has_class_binding = any(
        r in visitor.loads
        or r in visitor.params
        or any(a.startswith(f"{r}.") for a in all_attrs)
        for r in class_receivers
        if r in ("cls", "klass")
        or (rec_param and r == rec_param and is_class_rec)
        or any(a.startswith(f"{r}.") for a in all_attrs)
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)
    primary_rec = rec_param or ("cls" if is_class_rec else None)
    _normalize_receiver_order(
        inputs,
        has_instance_binding,
        has_class_binding,
        primary_receiver=primary_rec,
    )
    has_receiver_access = bool(
        any(
            r in visitor.loads or any(a.startswith(f"{r}.") for a in all_attrs)
            for r in all_receivers
            if r in ("self", "cls", "this", "klass")
            or any(a.startswith(f"{r}.") for a in all_attrs)
        )
    )
    return _UnitInputsAndReceivers(
        free_vars=free_vars,
        inputs=inputs,
        all_attrs=all_attrs,
        has_instance_binding=has_instance_binding,
        has_class_binding=has_class_binding,
        binding_kind=binding_kind,
        has_receiver_access=has_receiver_access,
    )


def _build_unit_scope_result_dict(
    visitor: _ScopeVisitor,
    io_info: _UnitInputsAndReceivers,
    *,
    outputs: List[str],
    conditional_outputs: List[str],
    def_assigned: Set[str],
    may_deleted: Set[str],
    hazards: List[str],
    closure_hazard_reasons: List[str],
    inst_receivers: Set[str],
    class_receivers: Set[str],
    outputs_ambiguous: bool,
    has_class_scoped_type_params: bool,
    effective_is_async: Optional[bool],
    resolved_strictness: str,
    extra_non_sup: AbstractSet[str],
    unit_heuristic_mgrs: Set[str],
    unit_wildcard_mgrs: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """Assembles the canonical schema-v2 unit scope analysis result dictionary."""
    return {
        "inputs": io_info.inputs,
        "outputs": outputs,
        "stores": visitor.stores,
        "free_vars": io_info.free_vars,
        "nonlocals": visitor.nonlocals,
        "globals": visitor.globals,
        "attrs_read": visitor.attrs_read,
        "attrs_written": visitor.attrs_written,
        "param_details": visitor.param_details,
        "return_type": visitor.return_type,
        "control_flow_hazards": hazards,
        "closure_hazard_reasons": closure_hazard_reasons,
        "is_control_flow_safe": not any(
            h in BLOCKING_CONTROL_FLOW_HAZARDS for h in hazards
        ),
        "has_yield": visitor.has_yield,
        "has_yield_assignment": visitor.has_yield_assignment,
        "has_try_or_with": visitor.has_try_or_with,
        "has_return": visitor.has_return,
        "has_return_value": visitor.has_return_value,
        "has_super": visitor.has_super,
        "has_mangled_names": visitor.has_mangled_names,
        "local_imports": visitor.local_imports,
        "yield_expr_names": visitor.yield_expr_names,
        "is_async": effective_is_async,
        "conditional_outputs": conditional_outputs,
        "definite_stores": sorted(def_assigned),
        "has_instance_binding": io_info.has_instance_binding,
        "has_class_binding": io_info.has_class_binding,
        "binding_kind": io_info.binding_kind,
        "has_receiver_access": io_info.has_receiver_access,
        "instance_attrs": [
            a
            for a in io_info.all_attrs
            if any(a.startswith(f"{r}.") for r in inst_receivers)
        ],
        "class_attrs": [
            a
            for a in io_info.all_attrs
            if any(a.startswith(f"{r}.") for r in class_receivers)
        ],
        "outputs_ambiguous": outputs_ambiguous,
        "type_params": list(visitor.type_params),
        "type_param_free_vars": list(visitor.type_param_free_vars),
        "has_class_scoped_type_params": has_class_scoped_type_params,
        "deleted_names": sorted(may_deleted),
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "closure_strictness": resolved_strictness,
        "non_suppressing_ctx_managers": sorted(extra_non_sup),
        "heuristic_ctx_managers": sorted(unit_heuristic_mgrs),
        "wildcard_ctx_managers": sorted(unit_wildcard_mgrs),
    }


class _UnitVisitorSetup(NamedTuple):
    """Initial AST visitor pass, receiver sets, and baseline control-flow hazards for a unit."""

    visitor: _ScopeVisitor
    is_subroutine: bool
    rec_param: Optional[str]
    rec_kind: Optional[str]
    inst_receivers: Set[str]
    class_receivers: Set[str]
    all_receivers: Set[str]
    hazards: List[str]
    has_class_scoped_type_params: bool


def _run_unit_visitor_and_initial_hazards(
    unit: Dict[str, Any],
    prep: Any,
    parsed: Any,
    *,
    file_tree: Optional[ast.AST],
    get_full_source: Callable[[], Optional[str]],
) -> _UnitVisitorSetup:
    """Runs _ScopeVisitor on the parsed unit AST and collects baseline control-flow hazards."""
    if prep.normalized.is_subroutine_flag is not None:
        is_subroutine = prep.normalized.is_subroutine_flag
    elif prep.normalized.explicit_kind in _SUBROUTINE_KINDS:
        is_subroutine = True
    elif prep.normalized.explicit_kind in _WHOLE_UNIT_KINDS:
        is_subroutine = False
    else:
        is_subroutine = is_subroutine_unit(
            unit, source_text=get_full_source(), tree=file_tree
        )

    rec_param = unit.get("receiver_param")
    rec_kind = unit.get("receiver_kind")
    inst_receivers: Set[str] = {"self", "this"}
    class_receivers: Set[str] = {"cls", "klass"}
    if rec_param:
        (class_receivers if rec_kind == "class" else inst_receivers).add(rec_param)
    all_receivers = inst_receivers | class_receivers

    visitor = _ScopeVisitor(
        loop_offset=parsed.loop_offset,
        source_lines=split_source_lines(parsed.cand_text),
        is_subroutine=is_subroutine,
        receiver_names=tuple(all_receivers),
    )
    visitor.visit(parsed.tree)

    hazards: List[str] = []
    if visitor.naked_breaks > 0:
        hazards.append("naked_break")
    if visitor.naked_continues > 0:
        hazards.append("naked_continue")
    if visitor.has_yield:
        hazards.append("generator_yield")
    if is_subroutine and visitor.has_return:
        hazards.append("embedded_return")

    has_class_scoped_type_params = (
        _check_type_param_closure_capture(
            unit,
            parsed.tree,
            visitor,
            inferred_s_col=prep.inferred_s_col,
            file_tree=file_tree,
            get_full_source=get_full_source,
            hazards=hazards,
        )
        if visitor.type_param_free_vars and not is_subroutine
        else False
    )
    return _UnitVisitorSetup(
        visitor=visitor,
        is_subroutine=is_subroutine,
        rec_param=rec_param,
        rec_kind=rec_kind,
        inst_receivers=inst_receivers,
        class_receivers=class_receivers,
        all_receivers=all_receivers,
        hazards=hazards,
        has_class_scoped_type_params=has_class_scoped_type_params,
    )


def _analyze_unit_block_flow_and_deletions(
    tree: ast.AST,
    candidate_stmts: Sequence[ast.stmt],
    visitor: _ScopeVisitor,
    enc: _UnitEnclosingContext,
    prep: Any,
    *,
    is_subroutine: bool,
    unit_heuristic_mgrs: Set[str],
    unit_wildcard_mgrs: Set[Tuple[str, str]],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Runs block definite-assignment/deletion analysis and records live-in deletions on visitor."""
    may_deleted: Set[str] = set()
    live_in_deleted: Set[str] = set()
    initial_sup_prov = _seed_unit_with_suppress_provenance(
        tree,
        candidate_stmts,
        visitor,
        mod_tree=enc.mod_tree,
        full_mod_lines=enc.full_mod_lines,
        enclosing_chain=enc.enclosing_chain,
        is_subroutine=is_subroutine,
        u_start=enc.u_start,
        u_end=enc.u_end,
        resolved_s_col=enc.resolved_s_col,
        extra_non_sup=prep.extra_non_sup,
        strict_with_flag=prep.strict_with_flag,
    )
    with _capture_unit_ctx_managers(unit_heuristic_mgrs, unit_wildcard_mgrs):
        def_assigned, cond_assigned = _analyze_block_assignment(
            candidate_stmts,
            may_deleted=may_deleted,
            live_in_deleted=live_in_deleted,
            suppress_aliases=initial_sup_prov,
            extra_non_suppressing_ctx_managers=prep.extra_non_sup,
            strict_with_suppression=prep.strict_with_flag,
        )
    for del_name in sorted(live_in_deleted):
        if (
            del_name not in visitor.params
            and del_name not in visitor.globals
            and del_name not in BUILTIN_NAMES
        ):
            if del_name not in visitor.loads:
                visitor.loads.append(del_name)
            if del_name not in visitor.read_before_write:
                visitor.read_before_write.append(del_name)
    return def_assigned, cond_assigned, may_deleted


def _inspect_unit_scope_impl(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_tree: Optional[ast.AST] = None,
    *,
    allow_disk_read: bool = True,
) -> Dict[str, Any]:
    """Coordinates source preparation, AST parsing, block-flow, and closure-hazard analysis for one unit."""
    cached_full_source: List[Optional[str]] = [None]
    full_source_loaded: List[bool] = [False]

    def _get_full_source() -> Optional[str]:
        if not full_source_loaded[0]:
            if allow_disk_read:
                cached_full_source[0] = _load_unit_file_text(unit, repo_root=repo_root)
            else:
                buf = resolve_unit_source_buffer(
                    unit,
                    include_lines=False,
                    is_sliced_lines_fn=_is_sliced_excerpt_for_file,
                )
                cached_full_source[0] = None if buf.is_invalid_notebook else buf.full_text
            full_source_loaded[0] = True
        return cached_full_source[0]

    prep = _prepare_unit_scope_source(
        unit,
        repo_root=repo_root,
        file_tree=file_tree,
        allow_disk_read=allow_disk_read,
        get_full_source=_get_full_source,
    )
    if prep.early_result is not None:
        return prep.early_result

    parsed = _parse_unit_scope_ast(prep.dedented)
    if parsed.tree is None:
        if parsed.had_resource_error:
            return {
                **prep.empty_res,
                "control_flow_hazards": ["pathological_ast"],
                "is_control_flow_safe": False,
            }
        return prep.empty_res

    tree = parsed.tree
    candidate_stmts = parsed.candidate_stmts
    try:
        setup = _run_unit_visitor_and_initial_hazards(
            unit, prep, parsed, file_tree=file_tree, get_full_source=_get_full_source
        )
        visitor = setup.visitor
        is_subroutine = setup.is_subroutine
        hazards = setup.hazards
        closure_hazard_reasons: List[str] = []

        enc = _resolve_unit_enclosing_context(
            unit,
            candidate_stmts,
            inferred_s_col=prep.inferred_s_col,
            inferred_e_col=prep.inferred_e_col,
            file_tree=file_tree,
            get_full_source=_get_full_source,
        )
        unit_heuristic_mgrs: Set[str] = set()
        unit_wildcard_mgrs: Set[Tuple[str, str]] = set()
        def_assigned, cond_assigned, may_deleted = (
            _analyze_unit_block_flow_and_deletions(
                tree,
                candidate_stmts,
                visitor,
                enc,
                prep,
                is_subroutine=is_subroutine,
                unit_heuristic_mgrs=unit_heuristic_mgrs,
                unit_wildcard_mgrs=unit_wildcard_mgrs,
            )
        )

        _analyze_subroutine_conditional_live_ins(
            unit,
            candidate_stmts,
            visitor,
            is_subroutine=is_subroutine,
            def_assigned=def_assigned,
            cond_assigned=cond_assigned,
            mod_tree=enc.mod_tree,
            full_mod_lines=enc.full_mod_lines,
            enclosing_chain=enc.enclosing_chain,
            u_start=enc.u_start,
            u_end=enc.u_end,
            resolved_s_col=enc.resolved_s_col,
            resolved_e_col=enc.resolved_e_col,
            strict_with_flag=prep.strict_with_flag,
            extra_non_sup=prep.extra_non_sup,
            get_full_source=_get_full_source,
            hazards=hazards,
            closure_hazard_reasons=closure_hazard_reasons,
            unit_heuristic_mgrs=unit_heuristic_mgrs,
            unit_wildcard_mgrs=unit_wildcard_mgrs,
        )

        io_info = _compute_unit_inputs_and_receivers(
            visitor,
            rec_param=setup.rec_param,
            rec_kind=setup.rec_kind,
            inst_receivers=setup.inst_receivers,
            class_receivers=setup.class_receivers,
            all_receivers=setup.all_receivers,
        )

        all_deleted = _extract_deleted_names(candidate_stmts) | visitor.deleted_names
        outputs, outputs_ambiguous = _resolve_unit_outputs(
            unit,
            visitor,
            is_subroutine=is_subroutine,
            inputs=io_info.inputs,
            def_assigned=def_assigned,
            cond_assigned=cond_assigned,
            all_deleted=all_deleted,
            file_tree=file_tree,
            resolved_strictness=prep.resolved_strictness,
            get_full_source=_get_full_source,
        )
        conditional_outputs = [
            v
            for v in outputs
            if (v not in io_info.inputs or v in may_deleted) and v not in def_assigned
        ]
        if (
            is_subroutine
            and bool((set(outputs) | set(io_info.inputs)) & may_deleted)
            and "deleted_output" not in hazards
        ):
            hazards.append("deleted_output")

        _evaluate_unit_closure_and_lexical_hazards(
            unit,
            tree,
            candidate_stmts,
            visitor,
            is_subroutine=is_subroutine,
            inputs=io_info.inputs,
            outputs=outputs,
            free_vars=io_info.free_vars,
            all_deleted=all_deleted,
            def_assigned=def_assigned,
            mod_tree=enc.mod_tree,
            full_mod_lines=enc.full_mod_lines,
            enclosing_chain=enc.enclosing_chain,
            u_start=enc.u_start,
            u_end=enc.u_end,
            resolved_s_col=enc.resolved_s_col,
            resolved_e_col=enc.resolved_e_col,
            cols_resolved=enc.cols_resolved,
            resolved_strictness=prep.resolved_strictness,
            extra_non_sup=prep.extra_non_sup,
            get_full_source=_get_full_source,
            hazards=hazards,
            closure_hazard_reasons=closure_hazard_reasons,
            unit_heuristic_mgrs=unit_heuristic_mgrs,
            unit_wildcard_mgrs=unit_wildcard_mgrs,
        )

        effective_is_async = _resolve_unit_effective_async(
            unit,
            visitor,
            is_subroutine=is_subroutine,
            file_tree=file_tree,
            get_full_source=_get_full_source,
        )
        return _build_unit_scope_result_dict(
            visitor,
            io_info,
            outputs=outputs,
            conditional_outputs=conditional_outputs,
            def_assigned=def_assigned,
            may_deleted=may_deleted,
            hazards=hazards,
            closure_hazard_reasons=closure_hazard_reasons,
            inst_receivers=setup.inst_receivers,
            class_receivers=setup.class_receivers,
            outputs_ambiguous=outputs_ambiguous,
            has_class_scoped_type_params=setup.has_class_scoped_type_params,
            effective_is_async=effective_is_async,
            resolved_strictness=prep.resolved_strictness,
            extra_non_sup=prep.extra_non_sup,
            unit_heuristic_mgrs=unit_heuristic_mgrs,
            unit_wildcard_mgrs=unit_wildcard_mgrs,
        )
    except RecursionError:
        return {
            **prep.empty_res,
            "control_flow_hazards": ["pathological_ast"],
            "is_control_flow_safe": False,
        }


def inspect_single_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    allow_disk_read: bool = True,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping for an individual code unit."""
    if closure_strictness is not None or non_suppressing_ctx_managers is not None:
        normalized = normalize_unit_boundary(
            unit,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        )
        unit = dict(unit)
        if closure_strictness is not None:
            unit["closure_strictness"] = normalized.policy.mode
        if non_suppressing_ctx_managers is not None:
            unit["non_suppressing_ctx_managers"] = list(
                normalized.policy.non_suppressing_ctx_managers
            )
    return _inspect_unit_scope(
        unit,
        repo_root=repo_root,
        file_tree=tree,
        allow_disk_read=allow_disk_read,
    )


def _align_symmetric_var_list(list1: List[str], list2: List[str]) -> List[str]:
    """Aligns shared variables between two units, preserving side 1 order if equal length."""
    common = [var for var in list1 if var in list2]
    if len(list1) == len(list2):
        return common if len(common) == len(list1) else list1
    return common if common else list1


def _has_type_param_default(tp_str: str) -> bool:
    """Checks whether a PEP 695/696 type parameter string has a top-level default."""
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", '""', tp_str)
    depth = 0
    for ch in cleaned:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return True
    return False


def _merge_type_params(tp1: Sequence[str], tp2: Sequence[str]) -> List[str]:
    """Merges PEP 695 type parameters across two units without duplicate parameter names."""
    if not tp2:
        return list(tp1)
    if not tp1:
        return list(tp2)
    seen: Dict[str, str] = {}
    order: List[str] = []
    for raw_tp in list(tp1) + list(tp2):
        head = raw_tp.split(":", 1)[0].split("=", 1)[0].strip()
        ident = head.lstrip("*").strip()
        if not ident:
            continue
        if ident not in seen:
            seen[ident] = raw_tp
            order.append(ident)
        elif seen[ident] != raw_tp:
            prev_head = seen[ident].split(":", 1)[0].split("=", 1)[0].strip()
            seen[ident] = head if prev_head == head else ident
    merged = [seen[ident] for ident in order]
    return [tp for tp in merged if not _has_type_param_default(tp)] + [
        tp for tp in merged if _has_type_param_default(tp)
    ]


def scope_matches_closure_mode(
    scope: Optional[Dict[str, Any]],
    expected_strictness: str,
    *,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> bool:
    """Checks whether a precomputed scope dictionary matches the current schema and mode.

    Requires ``scope_schema_version == SCOPE_SCHEMA_VERSION`` and an explicit canonical
    ``closure_strictness`` string matching ``expected_strictness`` so unmarked or
    mismatched serialized scopes are never trusted without recomputation.
    """
    if scope is None:
        return False
    if scope.get("scope_schema_version") != SCOPE_SCHEMA_VERSION:
        return False
    raw_mode = scope.get("closure_strictness")
    if not isinstance(raw_mode, str):
        return False
    expected_mode = validate_closure_strictness(expected_strictness)
    if raw_mode != expected_mode:
        return False
    scope_non_sup = frozenset(
        normalize_non_suppressing_ctx_managers(
            scope.get("non_suppressing_ctx_managers")
        )
    )
    target_non_sup = frozenset(
        normalize_non_suppressing_ctx_managers(non_suppressing_ctx_managers)
    )
    if scope_non_sup != target_non_sup:
        return False
    return True


def ensure_unit_scope_closure_mode(
    unit: Dict[str, Any],
    scope: Optional[Dict[str, Any]],
    closure_strictness: str,
    *,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    source_text: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
    allow_disk_read: bool = True,
) -> Dict[str, Any]:
    """Returns scope if its schema version and closure mode match, else recomputes it.

    Caller-supplied scopes lacking ``scope_schema_version == SCOPE_SCHEMA_VERSION`` (or
    whose ``closure_strictness`` / ``non_suppressing_ctx_managers`` values differ from
    the requested configuration) are discarded — losing any manual caller overrides in
    those dicts — and recomputed from the backing module source/AST, or fail closed when
    backing source is unavailable.

    Note:
        Scopes that record terminal ``"pathological_ast"``, ``"stale_source_code"``, or
        ``"invalid_unit_columns"`` in ``control_flow_hazards`` alongside
        ``is_control_flow_safe=False`` are returned directly without re-parsing or
        mutating the terminal hazard dict, as they are unconditionally rejected
        downstream regardless of closure mode. Caller-supplied scopes stamped with
        matching ``scope_schema_version`` and ``closure_strictness`` are also
        returned directly without re-running source validation (such as
        ``stale_source_code``, ``invalid_unit_columns``, or
        ``unavailable_source_placeholder`` checks). When a caller-supplied ``scope`` (or
        ``unit["scope"]``) is discarded due to a missing schema stamp or mode mismatch,
        a once-per-run ``WARNING`` is emitted.
    """
    eff_mode = resolve_closure_strictness_mode(closure_strictness)
    eff_non_sup = normalize_non_suppressing_ctx_managers(
        non_suppressing_ctx_managers,
        unit.get("non_suppressing_ctx_managers"),
    )
    eff_scope: Optional[Dict[str, Any]] = (
        scope
        if scope is not None
        else (unit.get("scope") if isinstance(unit.get("scope"), dict) else None)
    )
    if eff_scope is not None and (
        scope_matches_closure_mode(
            eff_scope,
            eff_mode,
            non_suppressing_ctx_managers=eff_non_sup,
        )
        or (
            eff_scope.get("is_control_flow_safe") is False
            and any(
                h in eff_scope.get("control_flow_hazards", ())
                for h in (
                    "pathological_ast",
                    "stale_source_code",
                    "invalid_unit_columns",
                )
            )
        )
    ):
        if eff_mode != "pedantic":
            for mgr_name in normalize_heuristic_ctx_managers(
                eff_scope.get("heuristic_ctx_managers")
            ):
                record_heuristic_non_suppressing_ctx_manager(mgr_name)
        for pat, mgr_name in normalize_wildcard_ctx_manager_matches(
            eff_scope.get("wildcard_ctx_managers")
        ):
            record_wildcard_ctx_manager_match(pat, mgr_name)
        return eff_scope
    u_copy = {
        **unit,
        "closure_strictness": eff_mode,
    }
    u_copy.pop("scope", None)
    if eff_non_sup or "non_suppressing_ctx_managers" in u_copy:
        u_copy["non_suppressing_ctx_managers"] = list(eff_non_sup)
    if source_text is not None and not str(u_copy.get("source_text") or "").strip():
        u_copy["source_text"] = source_text
    if eff_scope is not None:
        norm_copy = normalize_unit_boundary(u_copy)
        buf_copy = resolve_unit_source_buffer(
            u_copy,
            include_lines=False,
            is_sliced_lines_fn=_is_sliced_excerpt_for_file,
        )
        src_lines_val = u_copy.get("source_lines")
        needs_enclosing_module = _is_subroutine_or_nested_expr_unit(
            u_copy, source_text=source_text, tree=tree
        )
        has_full_module_source = (
            tree is not None
            or (
                not buf_copy.is_invalid_notebook
                and isinstance(buf_copy.full_text, str)
                and bool(buf_copy.full_text.strip())
            )
            or (allow_disk_read and bool(_load_unit_file_text(u_copy, repo_root=repo_root)))
        )
        has_snippet_source = bool(norm_copy.source_code) or (
            is_non_string_sequence(src_lines_val)
            and any(
                isinstance(ln, str) and bool(ln.strip())
                for ln in src_lines_val
            )
        )
        has_unit_source = has_full_module_source or (
            has_snippet_source
            and (not needs_enclosing_module or norm_copy.allow_snippet_only)
        )
        if should_warn_closure_event("precomputed_scope_discarded"):
            discard_reason = (
                f"missing scope_schema_version={SCOPE_SCHEMA_VERSION}"
                if eff_scope.get("scope_schema_version") != SCOPE_SCHEMA_VERSION
                else "closure mode or allowlist mismatch"
            )
            action_msg = (
                "recomputing from source"
                if has_unit_source
                else "failing closed because backing source is unavailable"
            )
            logger.warning(
                "Discarding caller-supplied precomputed scope (%s) and %s "
                "(further occurrences suppressed for this run)",
                discard_reason,
                action_msg,
            )
        if not has_unit_source:
            logger.debug(
                "Precomputed scope for unit %r does not match active closure mode "
                "and backing source is unavailable; failing closed",
                unit.get("name"),
            )
            hazards = list(eff_scope.get("control_flow_hazards", ()))
            if "unavailable_source_for_closure_mode" not in hazards:
                hazards.append("unavailable_source_for_closure_mode")
            if "severed_closure_cell" not in hazards:
                hazards.append("severed_closure_cell")
            reasons = list(eff_scope.get("closure_hazard_reasons", ()))
            if "unavailable_source_for_closure_mode" not in reasons:
                reasons.append("unavailable_source_for_closure_mode")
            return {
                **_make_empty_scope_result(eff_mode, eff_non_sup),
                **eff_scope,
                "control_flow_hazards": hazards,
                "closure_hazard_reasons": reasons,
                "is_control_flow_safe": False,
                "scope_schema_version": SCOPE_SCHEMA_VERSION,
                "closure_strictness": eff_mode,
                "non_suppressing_ctx_managers": list(eff_non_sup),
                "heuristic_ctx_managers": normalize_heuristic_ctx_managers(
                    eff_scope.get("heuristic_ctx_managers")
                ),
                "wildcard_ctx_managers": normalize_wildcard_ctx_manager_matches(
                    eff_scope.get("wildcard_ctx_managers")
                ),
            }
        logger.debug(
            "Recomputing precomputed scope for unit %r to match active closure mode",
            unit.get("name"),
        )
    return inspect_single_unit_scope(
        u_copy,
        repo_root=repo_root,
        tree=tree,
        allow_disk_read=allow_disk_read,
    )


def _combine_unit_scope_dicts(
    info1: Dict[str, Any],
    info2: Optional[Dict[str, Any]],
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combines one or two validated unit scope dictionaries into a unified scope dict."""
    def _merge_list(key: str) -> List[Any]:
        first = list(info1.get(key, []))
        return (
            list(dict.fromkeys(first + list(info2.get(key, []))))
            if info2 is not None
            else first
        )

    def _or_bool(key: str) -> bool:
        return bool(info1.get(key, False) or (info2.get(key, False) if info2 is not None else False))

    if info2 is not None:
        inputs = _align_symmetric_var_list(info1["inputs"], info2["inputs"])
        outputs = _align_symmetric_var_list(info1["outputs"], info2["outputs"])
        rt1, rt2 = info1["return_type"], info2["return_type"]
        return_type = rt1 if (rt1 == rt2 or rt2 is None) else (rt2 if rt1 is None else None)
        is_safe = info1["is_control_flow_safe"] and info2["is_control_flow_safe"]
        local_imports_by_unit = [info1["local_imports"], info2["local_imports"]]
        yield_expr_names = info1["yield_expr_names"] + info2["yield_expr_names"]
        definite_stores = sorted(
            set(info1.get("definite_stores", [])) & set(info2.get("definite_stores", []))
        )
        type_params = _merge_type_params(
            info1.get("type_params", []), info2.get("type_params", [])
        )
        combined_strictness = combine_closure_strictness(
            str(info1.get("closure_strictness") or "strict"),
            str(info2.get("closure_strictness") or "strict"),
        )
        non_sup_out = sorted(
            set(info1.get("non_suppressing_ctx_managers", ()))
            & set(info2.get("non_suppressing_ctx_managers", ()))
        )
    else:
        inputs = info1["inputs"]
        outputs = info1["outputs"]
        return_type = info1["return_type"]
        is_safe = info1["is_control_flow_safe"]
        local_imports_by_unit = [info1["local_imports"]]
        yield_expr_names = info1["yield_expr_names"]
        definite_stores = info1.get("definite_stores", [])
        type_params = list(info1.get("type_params", []))
        combined_strictness = str(info1.get("closure_strictness") or "strict")
        non_sup_out = sorted(set(info1.get("non_suppressing_ctx_managers", ())))

    has_instance_binding = _or_bool("has_instance_binding")
    has_class_binding = _or_bool("has_class_binding")
    primary_rec = (
        u1.get("receiver_param")
        or (u2.get("receiver_param") if u2 else None)
        or (
            "cls"
            if (
                u1.get("receiver_kind") == "class"
                or (u2 and u2.get("receiver_kind") == "class")
            )
            else None
        )
    )
    _normalize_receiver_order(
        inputs,
        has_instance_binding,
        has_class_binding,
        primary_receiver=primary_rec,
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "locals": [var for var in info1["stores"] if var not in inputs],
        "free_vars": _merge_list("free_vars"),
        "nonlocals": _merge_list("nonlocals"),
        "globals": _merge_list("globals"),
        "attrs_read": _merge_list("attrs_read"),
        "attrs_written": _merge_list("attrs_written"),
        "param_details": info1["param_details"],
        "return_type": return_type,
        "control_flow_hazards": _merge_list("control_flow_hazards"),
        "is_control_flow_safe": is_safe,
        "has_yield": _or_bool("has_yield"),
        "has_yield_assignment": _or_bool("has_yield_assignment"),
        "has_try_or_with": _or_bool("has_try_or_with"),
        "has_return": _or_bool("has_return"),
        "has_return_value": _or_bool("has_return_value"),
        "has_super": _or_bool("has_super"),
        "has_mangled_names": _or_bool("has_mangled_names"),
        "local_imports": _merge_list("local_imports"),
        "local_imports_by_unit": local_imports_by_unit,
        "yield_expr_names": yield_expr_names,
        "is_async": _or_bool("is_async"),
        "conditional_outputs": _merge_list("conditional_outputs"),
        "definite_stores": definite_stores,
        "has_instance_binding": has_instance_binding,
        "has_class_binding": has_class_binding,
        "binding_kind": _determine_binding_kind(has_instance_binding, has_class_binding),
        "has_receiver_access": _or_bool("has_receiver_access"),
        "instance_attrs": _merge_list("instance_attrs"),
        "class_attrs": _merge_list("class_attrs"),
        "outputs_ambiguous": _or_bool("outputs_ambiguous"),
        "type_params": type_params,
        "type_param_free_vars": _merge_list("type_param_free_vars"),
        "has_class_scoped_type_params": _or_bool("has_class_scoped_type_params"),
        "deleted_names": _merge_list("deleted_names"),
        "closure_hazard_reasons": _merge_list("closure_hazard_reasons"),
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "closure_strictness": combined_strictness,
        "non_suppressing_ctx_managers": non_sup_out,
        "heuristic_ctx_managers": normalize_heuristic_ctx_managers(
            info1.get("heuristic_ctx_managers"),
            info2.get("heuristic_ctx_managers") if info2 is not None else (),
        ),
        "wildcard_ctx_managers": normalize_wildcard_ctx_manager_matches(
            info1.get("wildcard_ctx_managers"),
            info2.get("wildcard_ctx_managers") if info2 is not None else (),
        ),
    }


def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    scope1: Optional[Dict[str, Any]] = None,
    scope2: Optional[Dict[str, Any]] = None,
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping to determine inputs, outputs, closures, and attributes.

    Note:
        Precomputed ``scope1`` / ``scope2`` dictionaries (or ``unit["scope"]``) are validated
        via ``ensure_unit_scope_closure_mode``: any caller-supplied scope missing
        ``scope_schema_version == SCOPE_SCHEMA_VERSION`` or computed under a different
        closure strictness / allowlist configuration is discarded (dropping manual caller
        overrides) and recomputed from backing module source, or fails closed if backing
        source is unavailable.
    """
    info2: Optional[Dict[str, Any]] = None
    if u2 is not None:
        pair_policy = resolve_pair_closure_policy(
            u1,
            u2,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        )
        eff_mode = pair_policy.effective_mode
        with analysis_session():
            info1 = ensure_unit_scope_closure_mode(
                u1,
                scope1,
                eff_mode,
                non_suppressing_ctx_managers=pair_policy.unit1.non_suppressing_ctx_managers,
                repo_root=repo_root,
                tree=tree1,
            )
            info2 = ensure_unit_scope_closure_mode(
                u2,
                scope2,
                eff_mode,
                non_suppressing_ctx_managers=pair_policy.unit2.non_suppressing_ctx_managers,
                repo_root=repo_root,
                tree=tree2,
            )
    else:
        policy1 = resolve_unit_closure_policy(
            u1,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        )
        info1 = ensure_unit_scope_closure_mode(
            u1,
            scope1,
            policy1.mode,
            non_suppressing_ctx_managers=policy1.non_suppressing_ctx_managers,
            repo_root=repo_root,
            tree=tree1,
        )
    return _combine_unit_scope_dicts(info1, info2, u1, u2)
