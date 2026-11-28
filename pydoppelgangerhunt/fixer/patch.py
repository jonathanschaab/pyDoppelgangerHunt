"""Delegation call synthesis, unit overlap filtering, and multi-file patch generation."""

from __future__ import annotations

import ast
from collections import defaultdict
import contextlib
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
from typing import (
    Any,
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

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    collect_wildcard_ctx_managers,
    format_ctx_manager_assumption_comments,
    heuristic_ctx_manager_session,
    is_recognized_closure_strictness,
    normalize_heuristic_ctx_managers,
    normalize_non_suppressing_ctx_managers,
    normalize_path_string,
    normalize_wildcard_ctx_manager_matches,
    record_heuristic_non_suppressing_ctx_manager,
    record_wildcard_ctx_manager_match,
)
from pydoppelgangerhunt._ast_compat import is_cpython_parser_stack_overflow
from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_parse_tree,
    analysis_session,
)
from pydoppelgangerhunt.fixer._pair_eval import (
    check_pair_core_hazards,
    evaluate_clone_pair_scopes,
)
from pydoppelgangerhunt.fixer._patch_delegation import (  # pylint: disable=unused-import
    UnitCollisionError,
    _DelegationCallError,
    _FilePatchPlan,
    _PlanSnapshot,
    _apply_line_deltas,
    _attach_helper_to_plan,
    _build_unit_delegation_call,
    _build_whole_method_delegation,
    _compute_replacement_line_deltas,
    _delegate_unit_in_plan,
    _derive_unit_indent_step,
    _finalize_host_unit_and_helper,
    _has_unconditional_terminal_return,
    _intervals_overlap,
    _module_imports_target,
    _outputs_compatible,
    _render_file_patch_plan,
    _wire_cross_module_host_delegation,
    check_units_overlap,
    filter_overlapping_clone_units,
    intervals_overlap,
    refactor_module_units,
    validate_module_unit_replacements,
)
from pydoppelgangerhunt.fixer._patch_imports import (  # pylint: disable=unused-import
    _canonicalize_helper_relative_imports,
    _collect_host_missing_imports,
    _extract_helper_symbols,
    _extract_module_defined_names,
    _extract_module_import_statements,
    _has_future_annotations,
    _helper_requires_future_annotations,
    _unpack_source_item,
)
from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _get_enclosing_receiver_kind,
    _has_receiver_reference,
    _is_method_of_class,
    _is_pair_in_method,
    _is_same_file_path,
    _prune_unshared_receivers,
    _resolve_effective_binding,
    find_enclosing_class,
    find_enclosing_function,
)
from pydoppelgangerhunt.fixer.dataflow import (
    allows_u2_source_placeholder,
    has_async_generator_delegation_hazard,
    resolve_clone_pair_outputs,
)
from pydoppelgangerhunt.fixer.depgraph import (
    ModuleDependencyGraph,
    _find_enclosing_package_root,
    _find_project_filesystem_root,
    _is_safe_repo_python_path,
    _parse_source_imports,
    build_module_graph,
    derive_module_import_path as _derive_module_import_path,
    find_nearest_common_package,
    resolve_shared_module_file,
)
from pydoppelgangerhunt.fixer.scope import (
    _normalize_receiver_attrs,
    analyze_unit_variable_scope,
    resolve_unit_inferred_col_bounds,
)
from pydoppelgangerhunt.fixer.source import (  # pylint: disable=unused-import
    _extract_unit_token_lines,
    compute_line_offsets,
    extract_unit_comments_and_pragmas,
    is_subroutine_unit,
    is_valid_unit_coordinates,
    parse_unit_coord,
    split_source_lines,
    validate_unit_schema,
)
from pydoppelgangerhunt.fixer.synthesis import (
    _align_pair_expr_kinds,
    _compile_source_safely,
    _prune_pair_lhs_and_augassign_scopes,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.source_lines import (
    resolve_unit_line_bounds,
    unit_file_str,
)
from pydoppelgangerhunt.unit_boundary import normalize_unit_boundary

logger = logging.getLogger(__name__)


def _safely_prepare_helper_and_imports(
    host_plan: _FilePatchPlan,
    helper_code: str,
    scope: Dict[str, Any],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    f1_plan: _FilePatchPlan,
    f2_plan: Optional[_FilePatchPlan],
    host_mod: Optional[str] = None,
    is_host_pkg: bool = False,
) -> Tuple[str, Optional[List[str]]]:
    """Canonicalizes helper relative imports and collects host imports, handling conflicts."""
    try:
        source_mods = [
            (_unpack_source_item(s)[1], _unpack_source_item(s)[2])
            for s in source_texts
        ]
        canon_code = _canonicalize_helper_relative_imports(helper_code, source_mods)
        imports = _collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=canon_code,
            scope=scope,
            source_texts=source_texts,
            host_mod=host_mod,
            is_host_pkg=is_host_pkg,
        )
        return canon_code, imports
    except ValueError as err:
        prefix = "Cross-module clone pair" if f2_plan is not None else "Clone pair"
        conflict_msg = (
            f"# Note: {prefix}; {err}; skipping extraction.\n"
        )
        f1_plan.comments.append(conflict_msg)
        if f2_plan is not None:
            f2_plan.comments.append(conflict_msg)
        return helper_code, None


def _safely_collect_host_missing_imports(
    host_plan: _FilePatchPlan,
    helper_code: str,
    scope: Dict[str, Any],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    f1_plan: _FilePatchPlan,
    f2_plan: Optional[_FilePatchPlan],
    host_mod: Optional[str] = None,
    is_host_pkg: bool = False,
) -> Optional[List[str]]:
    """Legacy helper: collects host imports or records skip comments on plans if a conflict is detected."""
    _, maybe_imports = _safely_prepare_helper_and_imports(
        host_plan=host_plan,
        helper_code=helper_code,
        scope=scope,
        source_texts=source_texts,
        f1_plan=f1_plan,
        f2_plan=f2_plan,
        host_mod=host_mod,
        is_host_pkg=is_host_pkg,
    )
    return maybe_imports


def _safely_resolve_shared_module_file(
    f1_path: Path,
    f2_path: Path,
    root: Path,
    shared_module_name: str,
    f1_plan: _FilePatchPlan,
    f2_plan: Optional[_FilePatchPlan] = None,
) -> Optional[Path]:
    """Resolves the shared module file path, appending skip advisory comments on containment error."""
    try:
        resolved = resolve_shared_module_file(
            f1_path, f2_path, root, shared_module_name=shared_module_name
        )
        if resolved.is_symlink():
            raise ValueError(
                f"shared module path {resolved.name} is an existing symlink"
            )
        if resolved.is_dir():
            raise ValueError(
                f"shared module path {resolved.name} is an existing directory"
            )
        return resolved
    except (ValueError, OSError, RuntimeError) as exc:
        err_msg = f"# Note: Cross-module clone pair; {exc}; skipping extraction.\n"
        f1_plan.comments.append(err_msg)
        if f2_plan is not None:
            f2_plan.comments.append(err_msg)
        return None


def _register_plan_dependencies_in_graph(
    dg: ModuleDependencyGraph,
    mod_name: str,
    file_path: Path,
    import_stmts: Sequence[str],
) -> None:
    """Registers a module and its synthesized import statements into the dependency graph."""
    if not mod_name:
        return
    dg.add_module(mod_name, file_path)
    if not import_stmts:
        return
    known_modules = set(dg.mod_to_file.keys())
    raw_imports = _parse_source_imports(
        "\n".join(import_stmts),
        mod_name,
        is_package=(file_path.name == "__init__.py"),
    )
    for imp in raw_imports:
        dg.add_import_dependency(mod_name, imp, known_modules=known_modules)


def _format_patch_relative_path(
    target_path: Path,
    primary_root: Path,
    fallback_root: Optional[Path] = None,
) -> str:
    """Formats target_path relative to primary_root, falling back to fallback_root or filename."""
    try:
        resolved_target = (
            target_path.resolve()
            if target_path.is_absolute()
            else (primary_root / target_path).resolve()
        )
    except (OSError, RuntimeError, ValueError):
        resolved_target = target_path
    for candidate in (primary_root, fallback_root):
        if candidate is not None:
            try:
                cand_res = candidate.resolve()
                return str(resolved_target.relative_to(cand_res)).replace("\\", "/")
            except (ValueError, OSError, RuntimeError):
                pass
    norm = normalize_path_string(str(target_path), strip_anchor=True)
    return norm if norm else str(target_path.name)


def _is_same_file_or_resolved(p1: Path, p2: Path) -> bool:
    """Checks whether two paths refer to the same file or resolved target."""
    if p1 == p2:
        return True
    try:
        return p1.resolve() == p2.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _validate_clone_file_path(
    raw_path: str, candidate_roots: Sequence[Path]
) -> Tuple[Optional[Path], bool]:
    """Validates raw_path across candidate roots, returning (resolved_path, is_rejected).

    Returns:
        (resolved_path, False) if path is safe, within root, non-symlinked, and exists on disk.
        (None, False) if path is safe, within root, non-symlinked, but does not exist on disk.
        (None, True) if path violates boundaries (outside root, symlink, non-Python, or unresolvable).
    """
    if not raw_path or not candidate_roots:
        return None, True
    try:
        p = Path(raw_path)
        roots_to_check = candidate_roots if p.is_absolute() else (candidate_roots[0],)
        found_nonexistent_safe = False
        for root_dir in roots_to_check:
            resolved, is_rejected = _is_safe_repo_python_path(raw_path, root_dir)
            if not is_rejected:
                if resolved is not None:
                    return resolved, False
                found_nonexistent_safe = True
        if found_nonexistent_safe:
            return None, False
        return None, True
    except (OSError, RuntimeError, ValueError):
        return None, True


def _resolve_safe_clone_file_path(
    raw_path: str, candidate_roots: Sequence[Path]
) -> Optional[Path]:
    """Resolves raw_path across candidate roots, rejecting symlinks and resolution errors."""
    resolved, is_rejected = _validate_clone_file_path(raw_path, candidate_roots)
    if is_rejected or resolved is None:
        return None
    return resolved


def _validate_helper_compilation(helper_code: str) -> bool:
    """Validates that synthesized helper code compiles without syntax or indentation errors.

    Attempts direct or compound-wrapped compilation first so multiline string literals
    containing triple newlines are never split, falling back to splitting top-level
    import headers from indented method definitions when combined strings are passed.

    Note:
        Intentionally preserved for standalone verification, post-synthesis validation
        assertions, and test harness compatibility.
    """
    try:
        compile(helper_code, "<patch_helper>", "exec")
        return True
    except IndentationError:
        wrapped_code = "if 1:\n" + helper_code
        try:
            compile(wrapped_code, "<patch_helper>", "exec")
            return True
        except IndentationError:
            pass
        except (SyntaxError, ValueError, RecursionError):
            return False
        except MemoryError as exc:
            if not is_cpython_parser_stack_overflow(exc, source=wrapped_code):
                raise
            return False
    except (SyntaxError, ValueError, RecursionError):
        return False
    except MemoryError as exc:
        if not is_cpython_parser_stack_overflow(exc, source=helper_code):
            raise
        return False

    sep = "\r\n\r\n\r\n" if "\r\n\r\n\r\n" in helper_code else "\n\n\n"
    if sep in helper_code:
        header, body = helper_code.split(sep, 1)
        if header.strip() and not _compile_source_safely(header, "<patch_imports>"):
            return False
        if body.strip():
            body_src = (
                "if 1:\n" + body
                if body.lstrip("\r\n").startswith((" ", "\t"))
                else body
            )
            if not _compile_source_safely(body_src, "<patch_helper>"):
                return False
        return True
    return False


_NO_REFACTOR_RE = re.compile(r"#\s*pragma\s*:\s*no[-_]refactor\b", re.IGNORECASE)


def _unit_has_no_refactor_pragma(
    unit: Dict[str, Any],
    file_lines: Optional[Sequence[str]],
    orig_text: Optional[str] = None,
    tree: Optional[ast.AST] = None,
) -> bool:
    """Checks whether the unit contains a '# pragma: no-refactor' directive."""
    if not file_lines and not orig_text:
        return False
    full_text = orig_text if orig_text is not None else "".join(file_lines or [])
    if not _NO_REFACTOR_RE.search(full_text):
        return False
    if not is_valid_unit_coordinates(unit):
        return False
    u_start, u_end = resolve_unit_line_bounds(unit)
    eff_lines = (
        file_lines if file_lines is not None else split_source_lines(full_text)
    )
    if u_start > len(eff_lines):
        return False
    start_idx = min(len(eff_lines), max(0, u_start - 1))
    effective_tree = tree if tree is not None else _get_or_parse_tree(full_text)
    if effective_tree is not None:
        for node in ast.walk(effective_tree):
            decs = getattr(node, "decorator_list", None)
            if decs:
                dec_start = min((getattr(d, "lineno", None) or u_start) for d in decs)
                node_lineno = getattr(node, "lineno", None) or 0
                if dec_start <= u_start <= node_lineno:
                    start_idx = min(start_idx, max(0, dec_start - 1))
    while start_idx > 0 and eff_lines[start_idx - 1].strip().startswith(("#", "@")):
        start_idx -= 1
    if not any(_NO_REFACTOR_RE.search(line) for line in eff_lines[start_idx:u_end]):
        return False
    try:
        comments = extract_unit_comments_and_pragmas(
            full_text,
            min(u_start, start_idx + 1),
            u_end,
            include_leading=False,
        )
        return any(bool(_NO_REFACTOR_RE.search(c["text"])) for c in comments)
    except Exception as exc:
        logger.debug(
            "Structured pragma extraction failed for unit %s-%s: %s; falling back to line scan",
            u_start,
            u_end,
            exc,
        )
        return True



def _determine_unit_receiver_kind(
    fn: Optional[Dict[str, Any]],
    unit: Dict[str, Any],
    enc: Optional[Dict[str, Any]],
) -> str:
    """Determines the receiver kind ('instance', 'class', 'static', 'none') for a unit."""
    non_method_kinds = (
        "closure",
        "class",
        "comprehension",
        "data_table",
        "complex_expr",
    )
    if fn:
        return _get_enclosing_receiver_kind(fn)
    if unit.get("receiver_kind"):
        return str(unit["receiver_kind"])
    if enc and unit.get("kind") not in non_method_kinds:
        return "instance"
    return "none"


def _record_skip_comment(
    f1_plan: _FilePatchPlan,
    f2_plan: Optional[_FilePatchPlan],
    skip_reasons: Dict[str, int],
    reason_key: str,
    comment: str,
) -> None:
    """Records a rejection reason counter and attaches the diagnostic comment to file plans."""
    skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
    f1_plan.comments.append(comment)
    if f2_plan is not None and f2_plan is not f1_plan:
        f2_plan.comments.append(comment)


def _unit_start(u: Dict[str, Any]) -> int:
    """Returns the clamped 1-based start line of a unit dictionary."""
    return max(1, parse_unit_coord(u, "start", default=1))


@contextlib.contextmanager
def _isolated_pair_ctx_manager_collectors() -> Iterator[
    Tuple[Set[str], Set[Tuple[str, str]]]
]:
    """Collects heuristic and wildcard context-manager hits for one pair without polluting the session."""
    with collect_heuristic_ctx_managers(isolate=True) as h_bucket:
        with collect_wildcard_ctx_managers(isolate=True) as wc_bucket:
            yield h_bucket, wc_bucket


def _record_committed_pair_ctx_managers(
    heuristics: Sequence[str],
    wildcards: Sequence[Tuple[str, str]],
) -> None:
    """Records heuristic and wildcard context-manager assumptions for a committed patch pair."""
    for mgr_name in heuristics:
        record_heuristic_non_suppressing_ctx_manager(mgr_name)
    for wc_pat, mgr_name in wildcards:
        record_wildcard_ctx_manager_match(wc_pat, mgr_name)


def _slice_loaded_unit_lines(
    unit: Dict[str, Any], full_lines: Sequence[str]
) -> Tuple[List[str], bool]:
    """Slices already-loaded full-file lines to unit line bounds, returning (lines, is_placeholder)."""
    u_start, u_end = resolve_unit_line_bounds(unit)
    start = max(1, u_start)
    end = min(len(full_lines), u_end)
    if start - 1 < end:
        return list(full_lines[start - 1 : end]), False
    return [], True


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
    method_binding: str = "auto",
    cross_file_strategy: str = "auto",
    shared_module_name: str = "_common.py",
    depgraph: Optional[ModuleDependencyGraph] = None,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    *,
    on_diagnostic_error: Optional[Callable[[], None]] = None,
) -> str:
    """Generates a git-apply compatible unified diff patch proposing shared helper extractions.

    Per-unit ``UnitSchema`` fields (including ``unit["closure_strictness"]``) are validated
    upfront via :func:`validate_unit_schema` and raise ``ValueError`` if invalid. Passing an
    unrecognized string for the ``closure_strictness`` keyword argument logs a deduplicated
    warning and falls back to the ``"pedantic"`` patch default.
    """
    with heuristic_ctx_manager_session(), analysis_session():
        return _generate_refactoring_patch_impl(
            clones=clones,
            repo_root=repo_root,
            type_merge_strategy=type_merge_strategy,
            replace_clones=replace_clones,
            method_binding=method_binding,
            cross_file_strategy=cross_file_strategy,
            shared_module_name=shared_module_name,
            depgraph=depgraph,
            closure_strictness=closure_strictness,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
            on_diagnostic_error=on_diagnostic_error,
        )


@dataclass
class _PatchBatchState:
    """Shared batch-level configuration, file plan registry, and dependency graph holder."""

    fs_root: Path
    patch_root: Path
    import_root: Path
    candidate_roots: Tuple[Path, ...]
    cross_file_strategy: str
    type_merge_strategy: str
    method_binding: str
    replace_clones: bool
    shared_module_name: str
    eff_caller_strictness: Optional[str]
    eff_non_sup: Tuple[str, ...]
    graph_holder: List[Optional[ModuleDependencyGraph]]
    file_plans: Dict[Path, _FilePatchPlan] = field(default_factory=dict)
    skip_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def get_depgraph(self) -> ModuleDependencyGraph:
        """Lazily builds and returns the mutable module dependency graph for this batch."""
        g = self.graph_holder[0]
        if g is None:
            g = build_module_graph(self.patch_root, import_root=self.import_root)
            self.graph_holder[0] = g
        return g

    def get_plan(
        self, file_p: Path, rel_f: str, text: str, is_new_file: bool = False
    ) -> _FilePatchPlan:
        """Returns the existing _FilePatchPlan for file_p or creates and registers a new one."""
        if file_p not in self.file_plans:
            self.file_plans[file_p] = _FilePatchPlan(
                file_p, rel_f, text, is_new_file=is_new_file
            )
        return self.file_plans[file_p]


@dataclass
class _LoadedPairFiles:
    """Validated file paths, plans, parsed ASTs, and effective unit dicts for one clone pair."""

    u1: Dict[str, Any]
    u2: Dict[str, Any]
    u1_eff: Dict[str, Any]
    u2_eff: Dict[str, Any]
    f1_path: Path
    f1_plan: _FilePatchPlan
    f2_plan: Optional[_FilePatchPlan]
    is_same_file: bool
    orig_text: str
    orig_lines: List[str]
    f2_text: Optional[str]
    u1_lines: List[str]
    u2_lines: Optional[List[str]]
    tree1: Optional[ast.AST]
    tree2: Optional[ast.AST]
    is_sub1: bool


@dataclass
class _ValidatedPairScopes:
    """Validated scope analysis, receiver binding, and paired inputs/outputs for one clone pair."""

    s1: Dict[str, Any]
    s2: Dict[str, Any]
    scope: Dict[str, Any]
    enc1: Optional[Dict[str, Any]]
    fn1_kind: str
    is_static: bool
    effective_binding: str
    helper_indent: str
    step: str
    effective_pair_strictness: str
    inputs: List[str]
    outputs: List[str]
    target_outs2: List[str]
    t_inputs1: List[str]
    t_inputs2: List[str]
    scope_heuristics: Set[str]
    scope_wildcards: Set[Tuple[str, str]]
    out_heuristics: Set[str]
    out_wildcards: Set[Tuple[str, str]]
    lhs_pruned: bool = False


@dataclass
class _PreparedPairHelper:
    """Synthesized helper code, naming, comments, and routing decision for one clone pair."""

    loaded: _LoadedPairFiles
    validated: _ValidatedPairScopes
    helper_name: str
    helper_code: str
    await_prefix: str
    f1_disp: str
    f2_disp: str
    pair_comment: str
    pair_heuristics: List[str]
    pair_wildcards: List[Tuple[str, str]]
    candidate_units: List[Dict[str, Any]]
    insert_line: int
    cross_file_action: str
    shared_p: Optional[Path]


def _load_and_validate_pair_files(
    batch: _PatchBatchState,
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    initial_plan_keys: Set[Path],
    plans_snapshot: Dict[Path, _PlanSnapshot],
) -> Optional[_LoadedPairFiles]:
    """Resolves file paths, loads plans, checks pragmas/overlaps, and aligns subroutine kinds."""
    norm1 = normalize_unit_boundary(u1)
    norm2 = normalize_unit_boundary(u2)
    f1_raw = normalize_path_string(norm1.file_path, strip_anchor=True)
    if not f1_raw:
        batch.skip_reasons["empty file path"] += 1
        return None
    f1_path, is_f1_rejected = _validate_clone_file_path(f1_raw, batch.candidate_roots)
    if is_f1_rejected or f1_path is None:
        batch.skip_reasons["invalid or rejected file path"] += 1
        return None

    if f1_path in initial_plan_keys and f1_path not in plans_snapshot:
        plans_snapshot[f1_path] = batch.file_plans[f1_path].snapshot()

    rel_f1 = _format_patch_relative_path(f1_path, batch.patch_root, batch.fs_root)
    f1_plan = batch.file_plans.get(f1_path)
    if f1_plan is None:
        try:
            orig_text = f1_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            batch.skip_reasons["unreadable file"] += 1
            return None
        f1_plan = batch.get_plan(f1_path, rel_f1, orig_text)
    else:
        orig_text = f1_plan.orig_text
    orig_lines = f1_plan.orig_lines

    f2_raw = normalize_path_string(norm2.file_path, strip_anchor=True)
    f2_path: Optional[Path] = None
    if f2_raw:
        f2_path, is_f2_rejected = _validate_clone_file_path(f2_raw, batch.candidate_roots)
        if is_f2_rejected:
            batch.skip_reasons["invalid or rejected file path"] += 1
            return None
        if (
            f2_path is not None
            and f2_path in initial_plan_keys
            and f2_path not in plans_snapshot
        ):
            plans_snapshot[f2_path] = batch.file_plans[f2_path].snapshot()

    is_same_file = False
    if f2_path is not None:
        is_same_file = f1_path == f2_path or f1_path.resolve() == f2_path.resolve()
    elif not f2_raw:
        is_same_file = True
    elif _is_same_file_path(f1_raw, f2_raw, repo_root=str(batch.fs_root)):
        is_same_file = True

    f2_plan: Optional[_FilePatchPlan] = None
    if not is_same_file:
        if not f2_raw:
            batch.skip_reasons["missing second file"] += 1
            return None
        if f2_path is not None:
            rel_f2 = _format_patch_relative_path(f2_path, batch.patch_root, batch.fs_root)
            f2_plan = batch.file_plans.get(f2_path)
            if f2_plan is None:
                try:
                    loaded_text = f2_path.read_text(encoding="utf-8")
                    f2_plan = batch.get_plan(f2_path, rel_f2, loaded_text)
                except (OSError, UnicodeDecodeError):
                    f2_plan = None
            if f2_plan is None and batch.replace_clones:
                batch.skip_reasons["unreadable file"] += 1
                return None
        elif batch.replace_clones:
            batch.skip_reasons["unreadable file"] += 1
            return None

    u1_lines = orig_lines
    u2_lines = (
        orig_lines
        if is_same_file
        else (f2_plan.orig_lines if f2_plan is not None else None)
    )
    u1_eff = dict(u1, source_lines=u1_lines, source_lines_is_sliced=False)
    u2_eff = (
        dict(u2, source_lines=u2_lines, source_lines_is_sliced=False)
        if u2_lines is not None
        else dict(u2)
    )
    f2_text: Optional[str] = (
        f2_plan.orig_text
        if f2_plan is not None
        else (f1_plan.orig_text if is_same_file else None)
    )
    tree1 = f1_plan.parsed_tree
    tree2 = (
        f2_plan.parsed_tree
        if f2_plan is not None
        else (tree1 if is_same_file else None)
    )
    if _unit_has_no_refactor_pragma(
        u1, u1_lines, orig_text=orig_text, tree=tree1
    ) or _unit_has_no_refactor_pragma(
        u2, u2_lines, orig_text=f2_text, tree=tree2
    ):
        f1_disp = normalize_path_string(
            unit_file_str(u1, default="file1"), strip_anchor=False
        )
        f2_disp = normalize_path_string(
            unit_file_str(u2, default="file2"), strip_anchor=False
        )
        _record_skip_comment(
            f1_plan,
            f2_plan,
            batch.skip_reasons,
            "pragma inline suppression",
            f"# Note: Clone pair ({f1_disp} vs {f2_disp}) skipped: "
            f"inline '# pragma: no-refactor' present\n",
        )
        return None

    if not _align_pair_expr_kinds(u1_eff, u2_eff):
        batch.skip_reasons["mismatched subroutine kinds"] += 1
        return None
    root_str = str(batch.fs_root)
    u1_col_lines, u1_col_ph = _slice_loaded_unit_lines(u1_eff, u1_lines)
    u1_eff = resolve_unit_inferred_col_bounds(
        u1_eff,
        repo_root=root_str,
        unit_code_lines=u1_col_lines,
        is_placeholder_origin=u1_col_ph,
        explicit_col_sliced=False,
    ).unit
    if u2_lines is not None:
        u2_col_lines, u2_col_ph = _slice_loaded_unit_lines(u2_eff, u2_lines)
        u2_eff = resolve_unit_inferred_col_bounds(
            u2_eff,
            repo_root=root_str,
            unit_code_lines=u2_col_lines,
            is_placeholder_origin=u2_col_ph,
            explicit_col_sliced=False,
        ).unit
    elif not (
        u2_eff.get("source_lines")
        or u2_eff.get("source_text")
        or u2_eff.get("file_source")
    ):
        u2_eff = resolve_unit_inferred_col_bounds(
            u2_eff,
            repo_root=root_str,
            unit_code_lines=(),
            is_placeholder_origin=True,
        ).unit
    else:
        u2_eff = resolve_unit_inferred_col_bounds(u2_eff, repo_root=root_str).unit

    if batch.replace_clones:
        u1_claimed = any(
            check_units_overlap(u1_eff, prev_u, repo_root=root_str)
            for prev_u in f1_plan.claimed_units
        )
        target_plan_for_u2 = f1_plan if is_same_file else f2_plan
        u2_claimed = (
            any(
                check_units_overlap(u2_eff, prev_u, repo_root=root_str)
                for prev_u in target_plan_for_u2.claimed_units
            )
            if target_plan_for_u2 is not None
            else False
        )
        if u1_claimed or u2_claimed:
            batch.skip_reasons["overlapping claimed unit"] += 1
            return None

    is_sub1 = is_subroutine_unit(u1_eff, source_text=orig_text, tree=tree1)
    is_sub2 = is_subroutine_unit(u2_eff, source_text=f2_text, tree=tree2)
    u1_eff["is_subroutine"] = is_sub1
    u2_eff["is_subroutine"] = is_sub2
    if is_sub1 != is_sub2:
        batch.skip_reasons["mismatched subroutine kinds"] += 1
        logger.debug(
            "Skipping clone pair (%s, %s): mismatched subroutine kinds "
            "(is_sub1=%s, is_sub2=%s)",
            u1.get("name"),
            u2.get("name"),
            is_sub1,
            is_sub2,
        )
        return None
    return _LoadedPairFiles(
        u1=u1,
        u2=u2,
        u1_eff=u1_eff,
        u2_eff=u2_eff,
        f1_path=f1_path,
        f1_plan=f1_plan,
        f2_plan=f2_plan,
        is_same_file=is_same_file,
        orig_text=orig_text,
        orig_lines=orig_lines,
        f2_text=f2_text,
        u1_lines=u1_lines,
        u2_lines=u2_lines,
        tree1=tree1,
        tree2=tree2,
        is_sub1=is_sub1,
    )


def _check_pair_scope_hazards(
    batch: _PatchBatchState,
    loaded: _LoadedPairFiles,
    s1: Dict[str, Any],
    s2: Dict[str, Any],
    *,
    fn1_kind: str,
    fn2_kind: str,
    receiver_kinds_differ: bool,
) -> bool:
    """Checks scope hazards, async/generator/receiver parity, and returns True if pair is safe."""
    root_str = str(batch.fs_root)
    allow_u2_ph = allows_u2_source_placeholder(
        is_subroutine=loaded.is_sub1,
        requires_u2_edit=batch.replace_clones,
        source_text2=loaded.f2_text,
        u2=loaded.u2_eff,
        repo_root=root_str,
    )
    term_check = check_pair_core_hazards(
        s1,
        s2,
        is_same_file=loaded.is_same_file,
        allow_u2_placeholder=allow_u2_ph,
        check_stage="terminal",
    )
    if term_check.skip_reason is not None:
        batch.skip_reasons[term_check.skip_reason] += 1
        return False

    if loaded.u1_eff.get("kind") in (
        "comprehension",
        "complex_expr",
    ) or loaded.u2_eff.get("kind") in ("comprehension", "complex_expr"):
        stmt_lines1 = _extract_unit_token_lines(loaded.u1_eff, repo_root=root_str)
        stmt_lines2 = _extract_unit_token_lines(loaded.u2_eff, repo_root=root_str)
        if (
            stmt_lines1 is None
            or stmt_lines2 is None
            or not _prune_pair_lhs_and_augassign_scopes(
                loaded.u1_eff, loaded.u2_eff, s1, s2, stmt_lines1, stmt_lines2
            )
        ):
            batch.skip_reasons["incompatible inputs or outputs"] += 1
            return False

    compat_check = check_pair_core_hazards(
        s1,
        s2,
        is_same_file=loaded.is_same_file,
        allow_u2_placeholder=allow_u2_ph,
        check_stage="compatibility",
    )
    if compat_check.skip_reason is not None:
        batch.skip_reasons[compat_check.skip_reason] += 1
        if compat_check.skip_reason == "indeterminate async state":
            logger.debug(
                "Skipping clone pair (%s, %s): indeterminate async state",
                loaded.u1.get("name"),
                loaded.u2.get("name"),
            )
        return False

    rec1 = loaded.u1_eff.get("receiver_param") or (
        "cls" if fn1_kind == "class" else "self"
    )
    rec2 = loaded.u2_eff.get("receiver_param") or (
        "cls" if fn2_kind == "class" else "self"
    )
    if (
        _normalize_receiver_attrs(s1.get("attrs_read", []), rec1)
        != _normalize_receiver_attrs(s2.get("attrs_read", []), rec2)
        or _normalize_receiver_attrs(s1.get("attrs_written", []), rec1)
        != _normalize_receiver_attrs(s2.get("attrs_written", []), rec2)
    ):
        batch.skip_reasons["incompatible receiver attribute accesses"] += 1
        return False

    post_check = check_pair_core_hazards(
        s1,
        s2,
        is_same_file=loaded.is_same_file,
        allow_u2_placeholder=allow_u2_ph,
        check_stage="post_compat",
    )
    if post_check.skip_reason is not None:
        batch.skip_reasons[post_check.skip_reason] += 1
        if post_check.detail:
            logger.debug(
                "Skipping clone pair (%s, %s): %s (%s)",
                loaded.u1.get("name"),
                loaded.u2.get("name"),
                post_check.skip_reason,
                post_check.detail,
            )
        return False
    if "embedded_return" in post_check.combined_hazards:
        if loaded.u1_lines is None or loaded.u2_lines is None:
            batch.skip_reasons["missing source lines"] += 1
            return False
        if not (
            _has_unconditional_terminal_return(loaded.u1, loaded.u1_lines)
            and _has_unconditional_terminal_return(loaded.u2, loaded.u2_lines)
        ):
            batch.skip_reasons["embedded return hazard"] += 1
            return False
    if receiver_kinds_differ and (
        _has_receiver_reference(loaded.u1_eff, s1, repo_root=root_str)
        or _has_receiver_reference(loaded.u2_eff, s2, repo_root=root_str)
    ):
        batch.skip_reasons["unbound receiver reference"] += 1
        return False
    return True


class _PairEnclosingContext(NamedTuple):
    """Resolved enclosing class/function metadata and receiver kinds for a clone pair."""

    enc1: Optional[Dict[str, Any]]
    orig_fn1: Optional[Dict[str, Any]]
    orig_fn2: Optional[Dict[str, Any]]
    is_same_class: bool
    fn1_kind: str
    fn2_kind: str
    receiver_kinds_differ: bool
    has_method_ctx: bool
    is_static: bool


def _resolve_pair_enclosing_context(
    loaded: _LoadedPairFiles,
) -> _PairEnclosingContext:
    """Resolves enclosing class/method context and stamps inferred receiver metadata on u1_eff/u2_eff."""
    enc1 = find_enclosing_class(loaded.orig_text, loaded.u1_eff, tree=loaded.tree1)
    fn1 = find_enclosing_function(loaded.orig_text, loaded.u1_eff, tree=loaded.tree1)
    enc2 = (
        find_enclosing_class(loaded.f2_text, loaded.u2_eff, tree=loaded.tree2)
        if loaded.f2_text is not None
        else None
    )
    fn2 = (
        find_enclosing_function(loaded.f2_text, loaded.u2_eff, tree=loaded.tree2)
        if loaded.f2_text is not None
        else None
    )
    is_same_class = bool(
        loaded.is_same_file
        and enc1
        and enc2
        and enc1["name"] == enc2["name"]
        and enc1["start"] == enc2["start"]
    )
    orig_fn1, orig_fn2 = fn1, fn2
    if not _is_method_of_class(fn1, enc1):
        fn1 = None
    if not _is_method_of_class(fn2, enc2):
        fn2 = None

    fn1_kind = _determine_unit_receiver_kind(fn1, loaded.u1_eff, enc1)
    fn2_kind = _determine_unit_receiver_kind(fn2, loaded.u2_eff, enc2)
    receiver_kinds_differ = bool(fn1_kind != fn2_kind)
    has_method_ctx = bool(
        fn1
        and fn2
        and (fn1.get("receiver_param") or fn1.get("is_static"))
        and (fn2.get("receiver_param") or fn2.get("is_static"))
    )
    for fn_meta, fn_k, u_eff in (
        (fn1, fn1_kind, loaded.u1_eff),
        (fn2, fn2_kind, loaded.u2_eff),
    ):
        if fn_meta and u_eff.get("receiver_param") is None:
            u_eff["receiver_param"] = fn_meta.get("receiver_param")
        if fn_k != "none" and u_eff.get("receiver_kind") is None:
            u_eff["receiver_kind"] = fn_k
            if fn_k == "static":
                u_eff["is_static"] = True
    if fn1 and fn2:
        is_static = bool(fn1.get("is_static") and fn2.get("is_static"))
    else:
        is_static = bool(
            (fn1 and fn1.get("is_static")) or (fn2 and fn2.get("is_static"))
        )
    return _PairEnclosingContext(
        enc1=enc1,
        orig_fn1=orig_fn1,
        orig_fn2=orig_fn2,
        is_same_class=is_same_class,
        fn1_kind=fn1_kind,
        fn2_kind=fn2_kind,
        receiver_kinds_differ=receiver_kinds_differ,
        has_method_ctx=has_method_ctx,
        is_static=is_static,
    )


def _inspect_and_validate_pair_scopes(
    batch: _PatchBatchState,
    loaded: _LoadedPairFiles,
) -> Optional[_ValidatedPairScopes]:
    """Resolves enclosing scopes, closure policies, hazards, bindings, and paired I/O."""
    root_str = str(batch.fs_root)
    enc_ctx = _resolve_pair_enclosing_context(loaded)
    enc1 = enc_ctx.enc1
    is_same_class = enc_ctx.is_same_class
    fn1_kind = enc_ctx.fn1_kind
    fn2_kind = enc_ctx.fn2_kind
    receiver_kinds_differ = enc_ctx.receiver_kinds_differ
    has_method_ctx = enc_ctx.has_method_ctx
    is_static = enc_ctx.is_static
    orig_fn1 = enc_ctx.orig_fn1
    orig_fn2 = enc_ctx.orig_fn2

    # Apply monotone strict-wins pair resolution (pedantic > strict > lenient)
    # across both units through the shared pair-evaluation pipeline.
    scope_eval = evaluate_clone_pair_scopes(
        loaded.u1_eff,
        loaded.u2_eff,
        policy_u1=loaded.u1,
        policy_u2=loaded.u2,
        closure_strictness=batch.eff_caller_strictness,
        non_suppressing_ctx_managers=batch.eff_non_sup or None,
        default_strictness="pedantic",
        source_text1=loaded.orig_text,
        source_text2=loaded.f2_text,
        tree1=loaded.tree1,
        tree2=loaded.tree2,
        repo_root=root_str,
        copy_scopes=True,
        stamp_units=True,
    )
    effective_pair_strictness = scope_eval.policy.effective_mode
    s1 = scope_eval.scope1
    s2 = scope_eval.scope2
    scope_heuristics = set(scope_eval.heuristics)
    scope_wildcards = set(scope_eval.wildcards)

    if orig_fn1 and orig_fn1.get("is_async") and s1.get("has_yield"):
        s1["is_async"] = True
    if orig_fn2 and orig_fn2.get("is_async") and s2.get("has_yield"):
        s2["is_async"] = True
    if not _check_pair_scope_hazards(
        batch,
        loaded,
        s1,
        s2,
        fn1_kind=fn1_kind,
        fn2_kind=fn2_kind,
        receiver_kinds_differ=receiver_kinds_differ,
    ):
        return None

    has_super = bool(s1.get("has_super") or s2.get("has_super"))
    has_mangled = bool(s1.get("has_mangled_names") or s2.get("has_mangled_names"))
    is_in_method = _is_pair_in_method(
        loaded.u1_eff,
        loaded.u2_eff,
        method_binding=batch.method_binding,
        has_super=has_super,
        has_mangled=has_mangled,
        has_method_context=has_method_ctx,
    )
    effective_binding = _resolve_effective_binding(
        batch.method_binding,
        is_same_class,
        is_static=is_static,
        receiver_kinds_differ=receiver_kinds_differ,
        is_in_method=is_in_method,
        receiver_kind=fn1_kind,
    )
    if has_super and (effective_binding == "module" or is_static):
        batch.skip_reasons["super() in non-method helper"] += 1
        return None
    if has_mangled and (
        effective_binding != "method" or not is_same_class or is_static
    ):
        batch.skip_reasons["mangled names in non-method helper"] += 1
        return None
    if (
        s1.get("has_class_scoped_type_params")
        or s2.get("has_class_scoped_type_params")
    ) and (effective_binding != "method" or not is_same_class):
        batch.skip_reasons["type_param closure capture"] += 1
        return None

    helper_indent = (
        enc1["method_indent"]
        if (effective_binding == "method" and enc1)
        else ("    " if effective_binding == "method" else "")
    )
    step_hint = (
        enc1["method_indent"][len(enc1["indent"]) :]
        if (
            enc1
            and enc1.get("indent") is not None
            and enc1["method_indent"].startswith(enc1["indent"])
            and len(enc1["method_indent"]) > len(enc1["indent"])
        )
        else None
    )
    step = _derive_unit_indent_step(loaded.u1, loaded.orig_lines, step_hint)

    digest1 = loaded.f1_plan.content_digest
    digest2 = (
        loaded.f2_plan.content_digest
        if loaded.f2_plan is not None
        else (digest1 if loaded.is_same_file else None)
    )
    with _isolated_pair_ctx_manager_collectors() as (out_heuristics, out_wildcards):
        scope = analyze_unit_variable_scope(
            loaded.u1_eff,
            loaded.u2_eff,
            repo_root=root_str,
            tree1=loaded.tree1,
            tree2=loaded.tree2,
            scope1=s1,
            scope2=s2,
        )
        inputs = list(scope.get("inputs", []))
        if effective_binding == "module":
            inputs = _prune_unshared_receivers(
                inputs, loaded.u1_eff, loaded.u2_eff, s1, s2, repo_root=root_str
            )
        resolved_outs = resolve_clone_pair_outputs(
            u1=loaded.u1_eff,
            u2=loaded.u2_eff,
            scope1=s1,
            scope2=s2,
            source_text1=loaded.f1_plan.orig_text,
            source_text2=loaded.f2_text,
            tree1=loaded.tree1,
            tree2=loaded.tree2,
            repo_root=root_str,
            source_digest1=digest1,
            source_digest2=digest2,
            closure_strictness=effective_pair_strictness,
            non_suppressing_ctx_managers=batch.eff_non_sup or None,
            require_pairing=batch.replace_clones,
            return_details=True,
        )
    if resolved_outs.outputs is None:
        reason = resolved_outs.skip_reason or "unpaired outputs"
        batch.skip_reasons[reason] += 1
        logger.debug(
            "Skipping clone pair (%s, %s): failed to resolve clone pair outputs (%s)",
            loaded.u1.get("name"),
            loaded.u2.get("name"),
            reason,
        )
        return None
    outputs, target_outs2 = resolved_outs.outputs
    if has_async_generator_delegation_hazard(
        s1, s2, has_outputs=bool(outputs or target_outs2)
    ):
        batch.skip_reasons["async generator delegation hazard"] += 1
        logger.debug(
            "Skipping clone pair (%s, %s): async generator cannot return "
            "values or outputs",
            loaded.u1.get("name"),
            loaded.u2.get("name"),
        )
        return None
    loaded.u1_eff["outputs"] = outputs
    loaded.u2_eff["outputs"] = target_outs2
    t_inputs1 = list(s1.get("inputs", []))
    t_inputs2 = list(s2.get("inputs", []))
    if effective_binding == "module":
        t_inputs1 = _prune_unshared_receivers(
            t_inputs1, loaded.u1_eff, loaded.u2_eff, s1, s2, repo_root=root_str
        )
        t_inputs2 = _prune_unshared_receivers(
            t_inputs2, loaded.u2_eff, loaded.u1_eff, s2, s1, repo_root=root_str
        )
    if batch.replace_clones and (
        len(t_inputs1) != len(inputs)
        or len(t_inputs2) != len(inputs)
        or not _outputs_compatible(
            outputs,
            target_outs2,
            is_subroutine=loaded.is_sub1,
            has_yield=bool(s1.get("has_yield") or s2.get("has_yield")),
        )
    ):
        batch.skip_reasons["incompatible inputs or outputs"] += 1
        logger.debug(
            "Skipping clone pair (%s, %s): incompatible inputs or outputs",
            loaded.u1.get("name"),
            loaded.u2.get("name"),
        )
        return None
    return _ValidatedPairScopes(
        s1=s1,
        s2=s2,
        scope=scope,
        enc1=enc1,
        fn1_kind=fn1_kind,
        is_static=is_static,
        effective_binding=effective_binding,
        helper_indent=helper_indent,
        step=step,
        effective_pair_strictness=effective_pair_strictness,
        inputs=inputs,
        outputs=outputs,
        target_outs2=target_outs2,
        t_inputs1=t_inputs1,
        t_inputs2=t_inputs2,
        scope_heuristics=scope_heuristics,
        scope_wildcards=scope_wildcards,
        out_heuristics=out_heuristics,
        out_wildcards=out_wildcards,
        lhs_pruned=(
            loaded.u1_eff.get("kind") in ("comprehension", "complex_expr")
            or loaded.u2_eff.get("kind") in ("comprehension", "complex_expr")
        ),
    )


def _prepare_pair_helper_and_target(
    batch: _PatchBatchState,
    sim: float,
    loaded: _LoadedPairFiles,
    validated: _ValidatedPairScopes,
    initial_plan_keys: Set[Path],
    plans_snapshot: Dict[Path, _PlanSnapshot],
) -> Optional[_PreparedPairHelper]:
    """Selects cross-file routing, generates collision-free helper name & code, and builds pair comments."""
    base_name1 = _base_unit_name(loaded.u1)
    base_name2 = _base_unit_name(loaded.u2)
    base_helper = (
        f"_shared_{base_name1}"
        if base_name1 == base_name2
        else f"_shared_{base_name1}_{base_name2}"
    )
    shared_p: Optional[Path] = None
    target_host_plan: Optional[_FilePatchPlan] = None
    target_host_text: Optional[str] = None
    cross_file_action = batch.cross_file_strategy
    if not loaded.is_same_file and cross_file_action in ("skip", "none"):
        batch.skip_reasons["cross-file refactoring skipped"] += 1
        return None
    if (
        cross_file_action == "auto"
        and not loaded.is_same_file
        and loaded.f2_plan is not None
    ):
        common_dir = find_nearest_common_package(
            loaded.f1_path, loaded.f2_plan.path, batch.patch_root
        )
        try:
            res_common = common_dir.resolve()
            is_top_level = res_common in (
                batch.patch_root.resolve(),
                (batch.patch_root / "src").resolve(),
                batch.import_root.resolve(),
            )
        except (OSError, RuntimeError, ValueError):
            is_top_level = False
        cross_file_action = "host_module" if is_top_level else "shared_module"

    if (
        not loaded.is_same_file
        and loaded.f2_plan is not None
        and cross_file_action in ("shared_module", "shared")
    ):
        maybe_shared_p = _safely_resolve_shared_module_file(
            loaded.f1_path,
            loaded.f2_plan.path,
            batch.patch_root,
            batch.shared_module_name,
            loaded.f1_plan,
            loaded.f2_plan,
        )
        if maybe_shared_p is None:
            batch.skip_reasons["shared module path unresolvable"] += 1
            return None
        shared_p = maybe_shared_p
        if shared_p in initial_plan_keys and shared_p not in plans_snapshot:
            plans_snapshot[shared_p] = batch.file_plans[shared_p].snapshot()
        if _is_same_file_or_resolved(shared_p, loaded.f1_path):
            target_host_plan = loaded.f1_plan
        elif _is_same_file_or_resolved(shared_p, loaded.f2_plan.path):
            target_host_plan = loaded.f2_plan
        else:
            target_host_plan = batch.file_plans.get(shared_p)
            try:
                if (
                    target_host_plan is None
                    and shared_p.is_file()
                    and not shared_p.is_symlink()
                ):
                    try:
                        shared_p.resolve().relative_to(batch.patch_root.resolve())
                        target_host_text = shared_p.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError, ValueError, RuntimeError):
                        pass
            except (OSError, RuntimeError, ValueError):
                pass

    helper_name = base_helper
    h_idx = 2
    while (
        helper_name in loaded.f1_plan.used_helper_names
        or bool(re.search(rf"\b{re.escape(helper_name)}\b", loaded.orig_text))
        or (
            loaded.f2_plan is not None
            and (
                helper_name in loaded.f2_plan.used_helper_names
                or bool(
                    re.search(rf"\b{re.escape(helper_name)}\b", loaded.f2_plan.orig_text)
                )
            )
        )
        or (
            target_host_plan is not None
            and (
                helper_name in target_host_plan.used_helper_names
                or bool(
                    re.search(
                        rf"\b{re.escape(helper_name)}\b", target_host_plan.orig_text
                    )
                )
            )
        )
        or (
            target_host_text is not None
            and bool(re.search(rf"\b{re.escape(helper_name)}\b", target_host_text))
        )
    ):
        helper_name = f"{base_helper}_{h_idx}"
        h_idx += 1

    with _isolated_pair_ctx_manager_collectors() as (synth_heuristics, synth_wildcards):
        helper_code = synthesize_shared_helper_code(
            loaded.u1_eff,
            loaded.u2_eff,
            type_merge_strategy=batch.type_merge_strategy,
            method_binding=validated.effective_binding,
            indent=validated.helper_indent,
            is_static=validated.is_static,
            receiver_kind=(
                validated.fn1_kind if validated.effective_binding == "method" else None
            ),
            step=validated.step,
            repo_root=str(batch.fs_root),
            helper_name=helper_name,
            source_text1=loaded.f1_plan.orig_text,
            source_text2=loaded.f2_text,
            tree1=loaded.tree1,
            tree2=loaded.tree2,
            closure_strictness=validated.effective_pair_strictness,
            non_suppressing_ctx_managers=batch.eff_non_sup or None,
            outputs=validated.outputs,
            outputs2=validated.target_outs2,
            scope1=validated.s1,
            scope2=validated.s2,
            scope=validated.scope,
            scopes_lhs_pruned=validated.lhs_pruned,
        )
    if not helper_code:
        batch.skip_reasons["shared helper synthesis failed"] += 1
        return None

    await_prefix = "await " if validated.scope.get("is_async") else ""
    f1_disp = normalize_path_string(
        unit_file_str(loaded.u1, default="file1"), strip_anchor=False
    )
    f2_disp = normalize_path_string(
        unit_file_str(loaded.u2, default="file2"), strip_anchor=False
    )
    pair_comment = f"# Clone Pair ({sim:.1%}): {f1_disp} <===> {f2_disp}\n"
    pair_heuristics = normalize_heuristic_ctx_managers(
        validated.scope_heuristics,
        validated.out_heuristics,
        synth_heuristics,
        validated.s1.get("heuristic_ctx_managers"),
        validated.s2.get("heuristic_ctx_managers"),
        validated.scope.get("heuristic_ctx_managers"),
    )
    pair_wildcards = normalize_wildcard_ctx_manager_matches(
        validated.scope_wildcards,
        validated.out_wildcards,
        synth_wildcards,
        validated.s1.get("wildcard_ctx_managers"),
        validated.s2.get("wildcard_ctx_managers"),
        validated.scope.get("wildcard_ctx_managers"),
    )
    assumption_notes = format_ctx_manager_assumption_comments(
        pair_heuristics,
        pair_wildcards,
    )
    if assumption_notes:
        pair_comment += assumption_notes + "\n"

    candidate_units = [loaded.u1_eff]
    if loaded.is_same_file and not check_units_overlap(
        loaded.u1_eff, loaded.u2_eff, repo_root=str(batch.fs_root)
    ):
        candidate_units.append(loaded.u2_eff)
    earliest_unit = min(candidate_units, key=_unit_start)
    enc_fn_earliest = (
        find_enclosing_function(
            loaded.orig_text, earliest_unit, tree=loaded.f1_plan.parsed_tree
        )
        if validated.effective_binding == "method"
        else None
    )
    insert_line = (
        enc_fn_earliest["start"] if enc_fn_earliest else _unit_start(earliest_unit)
    )
    return _PreparedPairHelper(
        loaded=loaded,
        validated=validated,
        helper_name=helper_name,
        helper_code=helper_code,
        await_prefix=await_prefix,
        f1_disp=f1_disp,
        f2_disp=f2_disp,
        pair_comment=pair_comment,
        pair_heuristics=pair_heuristics,
        pair_wildcards=pair_wildcards,
        candidate_units=candidate_units,
        insert_line=insert_line,
        cross_file_action=cross_file_action,
        shared_p=shared_p,
    )


def _prepare_host_module_imports(
    batch: _PatchBatchState,
    prep: _PreparedPairHelper,
    host_plan: _FilePatchPlan,
    f2_plan: Optional[_FilePatchPlan],
) -> Optional[Tuple[str, List[str], str, bool, str]]:
    """Derives module paths and safely prepares host helper imports, returning None on skip."""
    loaded = prep.loaded
    mod1 = _derive_module_import_path(loaded.f1_path, batch.import_root)
    is_pkg1 = loaded.f1_path.name == "__init__.py"
    source_texts: List[Tuple[str, str, bool]] = [(loaded.orig_text, mod1, is_pkg1)]
    mod2 = ""
    if f2_plan is not None:
        mod2 = _derive_module_import_path(f2_plan.path, batch.import_root)
        source_texts.append(
            (f2_plan.orig_text, mod2, f2_plan.path.name == "__init__.py")
        )
    mod_host = (
        mod1
        if host_plan is loaded.f1_plan
        else _derive_module_import_path(host_plan.path, batch.import_root)
    )
    is_host_pkg = host_plan.path.name == "__init__.py"
    helper_code, maybe_imports = _safely_prepare_helper_and_imports(
        host_plan=host_plan,
        helper_code=prep.helper_code,
        scope=prep.validated.scope,
        source_texts=source_texts,
        f1_plan=loaded.f1_plan,
        f2_plan=f2_plan,
        host_mod=mod_host,
        is_host_pkg=is_host_pkg,
    )
    if maybe_imports is None:
        batch.skip_reasons["circular import or unresolvable import"] += 1
        return None
    return helper_code, maybe_imports, mod_host, is_host_pkg, mod2


def _apply_same_file_pair_plan(
    batch: _PatchBatchState,
    prep: _PreparedPairHelper,
) -> bool:
    """Applies helper insertion and optional call delegation for a same-file clone pair."""
    loaded = prep.loaded
    val = prep.validated
    prepared_imports = _prepare_host_module_imports(batch, prep, loaded.f1_plan, None)
    if prepared_imports is None:
        return False
    helper_code, host_imports, _, _, _ = prepared_imports
    loaded.f1_plan.used_helper_names.add(prep.helper_name)
    loaded.f1_plan.comments.append(prep.pair_comment)
    if batch.replace_clones:
        _delegate_unit_in_plan(
            loaded.u1_eff,
            loaded.f1_plan,
            helper_name=prep.helper_name,
            inputs=val.inputs,
            outputs=val.outputs,
            scope=val.scope,
            target_inputs=val.t_inputs1,
            target_outputs=val.outputs,
            await_prefix=prep.await_prefix,
            binding=val.effective_binding,
            step=val.step,
        )
        if len(prep.candidate_units) > 1:
            _delegate_unit_in_plan(
                loaded.u2_eff,
                loaded.f1_plan,
                helper_name=prep.helper_name,
                inputs=val.inputs,
                outputs=val.target_outs2,
                scope=val.scope,
                target_inputs=val.t_inputs2,
                target_outputs=val.target_outs2,
                await_prefix=prep.await_prefix,
                binding=val.effective_binding,
                step=val.step,
            )
    _attach_helper_to_plan(
        loaded.f1_plan,
        binding=val.effective_binding,
        insert_line=prep.insert_line,
        helper_code=helper_code,
        missing_imports=host_imports,
    )
    return True


def _resolve_shared_host_plan_and_callers(
    batch: _PatchBatchState,
    prep: _PreparedPairHelper,
    f2_plan: _FilePatchPlan,
) -> Optional[
    Tuple[
        _FilePatchPlan,
        List[Tuple[_FilePatchPlan, Dict[str, Any], List[str], List[str]]],
    ]
]:
    """Resolves or creates the shared-module host plan and caller list, failing closed on unsafe targets."""
    loaded = prep.loaded
    val = prep.validated
    shared_p = prep.shared_p
    if shared_p is None:
        batch.skip_reasons["shared module path unresolvable"] += 1
        return None
    rel_shared = _format_patch_relative_path(shared_p, batch.patch_root, batch.fs_root)
    if _is_same_file_or_resolved(shared_p, loaded.f1_path):
        return loaded.f1_plan, [(f2_plan, loaded.u2_eff, val.t_inputs2, val.target_outs2)]
    if _is_same_file_or_resolved(shared_p, f2_plan.path):
        return f2_plan, [(loaded.f1_plan, loaded.u1_eff, val.t_inputs1, val.outputs)]

    shared_plan = batch.file_plans.get(shared_p)
    if shared_plan is None:
        is_bad_target = False
        kind_desc = ""
        try:
            if shared_p.is_symlink() or shared_p.is_dir():
                is_bad_target = True
                kind_desc = (
                    "an existing symlink"
                    if shared_p.is_symlink()
                    else "an existing directory"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            is_bad_target = True
            kind_desc = f"unresolvable ({exc})"
        if is_bad_target:
            _record_skip_comment(
                loaded.f1_plan,
                f2_plan,
                batch.skip_reasons,
                "shared module target is symlink or directory",
                f"# Note: Cross-module clone pair; shared module path "
                f"{rel_shared} is {kind_desc}; skipping extraction.\n",
            )
            return None
        try:
            is_existing_file = shared_p.is_file()
        except (OSError, RuntimeError, ValueError):
            is_existing_file = False
        if is_existing_file:
            try:
                shared_p.resolve().relative_to(batch.patch_root.resolve())
                shared_text = shared_p.read_text(encoding="utf-8")
                shared_plan = batch.get_plan(
                    shared_p, rel_shared, shared_text, is_new_file=False
                )
            except (OSError, UnicodeDecodeError, ValueError, RuntimeError):
                _record_skip_comment(
                    loaded.f1_plan,
                    f2_plan,
                    batch.skip_reasons,
                    "shared module read error",
                    f"# Note: Cross-module clone pair; shared module file "
                    f"{rel_shared} exists but could not be read; "
                    f"skipping extraction.\n",
                )
                return None
        else:
            shared_plan = batch.get_plan(shared_p, rel_shared, "", is_new_file=True)
    return shared_plan, [
        (loaded.f1_plan, loaded.u1_eff, val.t_inputs1, val.outputs),
        (f2_plan, loaded.u2_eff, val.t_inputs2, val.target_outs2),
    ]


def _apply_shared_module_pair_plan(
    batch: _PatchBatchState,
    prep: _PreparedPairHelper,
    f2_plan: _FilePatchPlan,
) -> bool:
    """Applies cross-file shared-module extraction with circular-dependency verification."""
    loaded = prep.loaded
    val = prep.validated
    resolved_host = _resolve_shared_host_plan_and_callers(batch, prep, f2_plan)
    if resolved_host is None:
        return False
    host_plan, callers = resolved_host

    prepared_imports = _prepare_host_module_imports(batch, prep, host_plan, f2_plan)
    if prepared_imports is None:
        return False
    helper_code, host_imports, mod_host, is_host_pkg, _ = prepared_imports

    host_disp = normalize_path_string(str(host_plan.rel_path), strip_anchor=False)
    dg = batch.get_depgraph()
    if mod_host and host_imports:
        host_cycle = dg.check_cycle_if_imports_added(
            mod_host,
            host_imports,
            file_path=host_plan.path,
            is_package=is_host_pkg,
        )
        if host_cycle:
            cycle_desc = (
                f"Circular import detected (cycle: {' -> '.join(host_cycle)})"
            )
            _record_skip_comment(
                loaded.f1_plan,
                f2_plan,
                batch.skip_reasons,
                "circular import detected",
                f"# Note: Cross-module clone pair; "
                f"helper extraction to {host_disp} "
                f"rejected due to circular dependency ({cycle_desc}).\n",
            )
            return False

    tentative_dg = dg.copy()
    if mod_host:
        _register_plan_dependencies_in_graph(
            tentative_dg, mod_host, host_plan.path, host_imports
        )

    caller_evaluations: List[
        Tuple[
            _FilePatchPlan,
            Dict[str, Any],
            List[str],
            List[str],
            str,
            Optional[List[str]],
        ]
    ] = []
    for c_plan, c_unit, c_tin, c_tout in callers:
        c_mod = _derive_module_import_path(c_plan.path, batch.import_root)
        c_cycle = (
            tentative_dg.check_cycle_if_added(c_mod, mod_host)
            if (c_mod and mod_host)
            else None
        )
        caller_evaluations.append((c_plan, c_unit, c_tin, c_tout, c_mod, c_cycle))

    if host_plan in (loaded.f1_plan, f2_plan) and caller_evaluations:
        _, _, _, _, c_mod, c_cycle = caller_evaluations[0]
        if c_cycle or not mod_host or not c_mod:
            c_desc = f" (cycle: {' -> '.join(c_cycle)})" if c_cycle else ""
            _record_skip_comment(
                loaded.f1_plan,
                f2_plan,
                batch.skip_reasons,
                "circular import detected",
                f"# Note: Cross-module clone pair; "
                f"helper extraction to {host_disp} "
                f"rejected. Circular import or unresolvable module path{c_desc}.\n",
            )
            return False

    if host_plan not in (loaded.f1_plan, f2_plan) and all(
        bool(cycle or not mod_host or not m_call)
        for _, _, _, _, m_call, cycle in caller_evaluations
    ):
        c_descs = [
            f"{m_call}: {' -> '.join(cycle)}"
            for _, _, _, _, m_call, cycle in caller_evaluations
            if cycle
        ]
        c_txt = f" ({'; '.join(c_descs)})" if c_descs else ""
        _record_skip_comment(
            loaded.f1_plan,
            f2_plan,
            batch.skip_reasons,
            "circular import detected",
            f"# Note: Cross-module clone pair; helper extraction to {host_disp} "
            f"rejected due to circular dependency{c_txt}.\n",
        )
        return False

    host_plan.module_helpers.append(helper_code)
    host_plan.missing_imports.extend(host_imports)
    host_plan.used_helper_names.add(prep.helper_name)
    loaded.f1_plan.used_helper_names.add(prep.helper_name)
    f2_plan.used_helper_names.add(prep.helper_name)
    host_plan.comments.append(prep.pair_comment)

    if mod_host:
        _register_plan_dependencies_in_graph(
            dg, mod_host, host_plan.path, host_imports
        )

    if batch.replace_clones:
        if host_plan is loaded.f1_plan:
            _delegate_unit_in_plan(
                loaded.u1_eff,
                loaded.f1_plan,
                helper_name=prep.helper_name,
                inputs=val.inputs,
                outputs=val.outputs,
                scope=val.scope,
                target_inputs=val.t_inputs1,
                target_outputs=val.outputs,
                await_prefix=prep.await_prefix,
                binding=val.effective_binding,
                step=val.step,
            )
        elif host_plan is f2_plan:
            _delegate_unit_in_plan(
                loaded.u2_eff,
                f2_plan,
                helper_name=prep.helper_name,
                inputs=val.inputs,
                outputs=val.target_outs2,
                scope=val.scope,
                target_inputs=val.t_inputs2,
                target_outputs=val.target_outs2,
                await_prefix=prep.await_prefix,
                binding="module",
            )

    for c_plan, c_unit, c_tin, c_tout, mod_caller, cycle in caller_evaluations:
        c_plan.comments.append(prep.pair_comment)
        c_disp = normalize_path_string(str(c_plan.rel_path), strip_anchor=False)
        if cycle or not mod_host or not mod_caller:
            cycle_desc = (
                f"Circular import detected (cycle: {' -> '.join(cycle)})"
                if cycle
                else "Unresolvable module import path"
            )
            cycle_msg = (
                f"# Note: Cross-module clone pair; helper extracted to {host_disp}. "
                f"{cycle_desc}; import manually into {c_disp}.\n"
            )
            c_plan.comments.append(cycle_msg)
            host_plan.comments.append(cycle_msg)
        else:
            c_plan.missing_imports.append(f"from {mod_host} import {prep.helper_name}")
            dg.add_dependency(mod_caller, mod_host)
            if batch.replace_clones:
                _delegate_unit_in_plan(
                    c_unit,
                    c_plan,
                    helper_name=prep.helper_name,
                    inputs=val.inputs,
                    outputs=c_tout if c_tout is not None else val.outputs,
                    scope=val.scope,
                    target_inputs=c_tin,
                    target_outputs=c_tout,
                    await_prefix=prep.await_prefix,
                    binding="module",
                )
            else:
                c_plan.comments.append(
                    f"# Note: Cross-module clone pair; helper extracted to {host_disp}. "
                    f"Complete refactoring by replacing the clone with a call in {c_disp}.\n"
                )
    return True


def _apply_host_module_pair_plan(
    batch: _PatchBatchState,
    prep: _PreparedPairHelper,
    f2_plan: Optional[_FilePatchPlan],
) -> bool:
    """Applies cross-file host-module extraction (both live f2_plan and preview-only missing f2)."""
    loaded = prep.loaded
    val = prep.validated
    if f2_plan is None and batch.replace_clones:
        batch.skip_reasons["unreadable file"] += 1
        return False
    prepared_imports = _prepare_host_module_imports(batch, prep, loaded.f1_plan, f2_plan)
    if prepared_imports is None:
        return False
    helper_code, host_imports, mod1, is_pkg1, mod2 = prepared_imports
    if f2_plan is None:
        loaded.f1_plan.used_helper_names.add(prep.helper_name)
        loaded.f1_plan.comments.append(prep.pair_comment)
        loaded.f1_plan.comments.append(
            f"# Note: Cross-module clone pair; helper generated in {prep.f1_disp}. "
            f"Complete refactoring by importing the helper into {prep.f2_disp}.\n"
        )
        _finalize_host_unit_and_helper(
            plan=loaded.f1_plan,
            unit=loaded.u1_eff,
            helper_name=prep.helper_name,
            inputs=val.inputs,
            outputs=val.outputs,
            scope=val.scope,
            target_inputs=val.t_inputs1,
            binding=val.effective_binding,
            step=val.step,
            insert_line=prep.insert_line,
            helper_code=helper_code,
            missing_imports=host_imports,
            await_prefix=prep.await_prefix,
            replace_clones=batch.replace_clones,
        )
        return True

    dg = batch.get_depgraph()

    if mod1 and host_imports:
        host_cycle = dg.check_cycle_if_imports_added(
            mod1,
            host_imports,
            file_path=loaded.f1_plan.path,
            is_package=is_pkg1,
        )
        if host_cycle:
            batch.skip_reasons["circular import detected"] += 1
            cycle_desc = f" (cycle: {' -> '.join(host_cycle)})"
            cycle_msg = (
                f"# Note: Cross-module clone pair; helper extraction to {prep.f1_disp} "
                f"rejected due to circular dependency{cycle_desc}.\n"
            )
            for plan in (loaded.f1_plan, f2_plan):
                plan.comments.append(cycle_msg)
            return False

    tentative_dg = dg.copy()
    if mod1:
        _register_plan_dependencies_in_graph(
            tentative_dg, mod1, loaded.f1_plan.path, host_imports
        )
    cycle = (
        tentative_dg.check_cycle_if_added(mod2, mod1) if (mod1 and mod2) else None
    )
    direct_import = bool(
        mod2
        and _module_imports_target(
            loaded.orig_text, mod2, current_mod=mod1, is_package=is_pkg1
        )
    )
    if bool(cycle or direct_import) or not mod1 or not mod2:
        batch.skip_reasons["circular import detected"] += 1
        cycle_desc = ""
        if cycle:
            cycle_desc = f" (cycle: {' -> '.join(cycle)})"
        elif direct_import:
            cycle_desc = f" (cycle: {mod2} -> {mod1} -> {mod2})"
        cycle_msg = (
            f"# Note: Cross-module clone pair; helper extraction to {prep.f1_disp} rejected. "
            f"Circular import or unresolvable module path{cycle_desc}.\n"
        )
        for plan in (loaded.f1_plan, f2_plan):
            plan.comments.append(cycle_msg)
        return False

    _register_plan_dependencies_in_graph(dg, mod1, loaded.f1_plan.path, host_imports)
    loaded.f1_plan.used_helper_names.add(prep.helper_name)
    f2_plan.used_helper_names.add(prep.helper_name)
    loaded.f1_plan.comments.append(prep.pair_comment)

    _wire_cross_module_host_delegation(
        f1_plan=loaded.f1_plan,
        f2_plan=f2_plan,
        u2=loaded.u2_eff,
        mod1=mod1,
        helper_name=prep.helper_name,
        pair_comment=prep.pair_comment,
        f1_disp=prep.f1_disp,
        f2_disp=prep.f2_disp,
        replace_clones=batch.replace_clones,
        inputs=val.inputs,
        outputs=val.outputs,
        scope=val.scope,
        t_inputs2=val.t_inputs2,
        target_outs2=val.target_outs2,
        await_prefix=prep.await_prefix,
    )
    if batch.replace_clones and mod1 and mod2:
        dg.add_dependency(mod2, mod1)
    _finalize_host_unit_and_helper(
        plan=loaded.f1_plan,
        unit=loaded.u1_eff,
        helper_name=prep.helper_name,
        inputs=val.inputs,
        outputs=val.outputs,
        scope=val.scope,
        target_inputs=val.t_inputs1,
        binding=val.effective_binding,
        step=val.step,
        insert_line=prep.insert_line,
        helper_code=helper_code,
        missing_imports=host_imports,
        await_prefix=prep.await_prefix,
        replace_clones=batch.replace_clones,
    )
    return True


def _generate_refactoring_patch_impl(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
    method_binding: str = "auto",
    cross_file_strategy: str = "auto",
    shared_module_name: str = "_common.py",
    depgraph: Optional[ModuleDependencyGraph] = None,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    *,
    on_diagnostic_error: Optional[Callable[[], None]] = None,
) -> str:
    """Core implementation of generate_refactoring_patch inside active heuristic & analysis sessions."""
    if not clones:
        return ""
    seen_unit_ids: Set[int] = set()
    for _, u1_item, u2_item in clones:
        for unit_item in (u1_item, u2_item):
            unit_obj_id = id(unit_item)
            if unit_obj_id not in seen_unit_ids:
                seen_unit_ids.add(unit_obj_id)
                validate_unit_schema(unit_item)

    fs_root = Path(repo_root or os.getcwd()).resolve()
    if fs_root.is_file():
        fs_root = fs_root.parent
    patch_root = _find_project_filesystem_root(fs_root)
    import_root = _find_enclosing_package_root(fs_root)
    norm_cross_strategy = (cross_file_strategy or "auto").strip().lower()
    if norm_cross_strategy not in (
        "auto",
        "host_module",
        "host",
        "shared_module",
        "shared",
        "skip",
    ):
        norm_cross_strategy = "auto"
    norm_type_strategy = (type_merge_strategy or "fallback_any").strip().lower()
    if norm_type_strategy not in ("fallback_any", "union", "intersect", "strict"):
        norm_type_strategy = "fallback_any"
    norm_binding = (method_binding or "auto").strip().lower()
    if norm_binding not in ("auto", "method", "module"):
        norm_binding = "auto"

    is_unrecognized_caller_strictness = (
        closure_strictness is not None
        and not is_recognized_closure_strictness(closure_strictness)
    )
    eff_caller_strictness: Optional[str] = (
        str(closure_strictness).strip().lower()
        if closure_strictness is not None and not is_unrecognized_caller_strictness
        else closure_strictness
    )
    eff_non_sup = normalize_non_suppressing_ctx_managers(non_suppressing_ctx_managers)
    if eff_caller_strictness == "lenient":
        logger.debug("Generating patches with caller lenient closure strictness")

    batch = _PatchBatchState(
        fs_root=fs_root,
        patch_root=patch_root,
        import_root=import_root,
        candidate_roots=(
            (fs_root,) if repo_root is not None else (fs_root, patch_root, import_root)
        ),
        cross_file_strategy=norm_cross_strategy,
        type_merge_strategy=norm_type_strategy,
        method_binding=norm_binding,
        replace_clones=replace_clones,
        shared_module_name=shared_module_name,
        eff_caller_strictness=eff_caller_strictness,
        eff_non_sup=eff_non_sup,
        graph_holder=[depgraph.copy() if depgraph is not None else None],
    )

    for sim, u1, u2 in clones:
        if (
            not is_valid_unit_coordinates(u1, strict=True)
            or not is_valid_unit_coordinates(u2, strict=True)
        ):
            batch.skip_reasons["malformed coordinates"] += 1
            logger.debug(
                "Skipping clone pair with malformed unit coordinates: (%s, %s)",
                u1,
                u2,
            )
            continue

        # Snapshot only plans touched in this pair so rollback stays O(1) per pair.
        initial_plan_keys = set(batch.file_plans.keys())
        plans_snapshot: Dict[Path, _PlanSnapshot] = {}
        graph_snapshot = (
            batch.graph_holder[0].copy() if batch.graph_holder[0] is not None else None
        )
        try:
            loaded = _load_and_validate_pair_files(
                batch, u1, u2, initial_plan_keys, plans_snapshot
            )
            if loaded is None:
                continue
            validated = _inspect_and_validate_pair_scopes(batch, loaded)
            if validated is None:
                continue
            prep = _prepare_pair_helper_and_target(
                batch, sim, loaded, validated, initial_plan_keys, plans_snapshot
            )
            if prep is None:
                continue
            if loaded.is_same_file:
                committed = _apply_same_file_pair_plan(batch, prep)
            elif loaded.f2_plan is not None and prep.cross_file_action in (
                "skip",
                "none",
            ):
                batch.skip_reasons["cross-file refactoring skipped"] += 1
                committed = False
            elif loaded.f2_plan is not None and prep.cross_file_action in (
                "shared_module",
                "shared",
            ):
                committed = _apply_shared_module_pair_plan(batch, prep, loaded.f2_plan)
            else:
                committed = _apply_host_module_pair_plan(batch, prep, loaded.f2_plan)
            if committed:
                _record_committed_pair_ctx_managers(
                    prep.pair_heuristics, prep.pair_wildcards
                )
        except (AttributeError, NameError, RecursionError, ImportError, MemoryError):
            raise
        except Exception as exc:
            batch.skip_reasons["processing exception"] += 1
            if isinstance(exc, (UnitCollisionError, _DelegationCallError, SyntaxError)):
                logger.debug(
                    "Skipping clone pair due to collision, delegation, or syntax error: %s",
                    exc,
                )
            else:
                if on_diagnostic_error is not None:
                    on_diagnostic_error()
                logger.warning(
                    "Unexpected failure processing clone pair; rolling back changes for this pair: %s",
                    exc,
                    exc_info=True,
                )
            for p, snap in plans_snapshot.items():
                if p in batch.file_plans:
                    batch.file_plans[p].restore(snap)
            for p in list(batch.file_plans.keys()):
                if p not in initial_plan_keys:
                    del batch.file_plans[p]
            batch.graph_holder[0] = graph_snapshot

    total_skipped = sum(batch.skip_reasons.values())
    if total_skipped > 0:
        sorted_reasons = sorted(
            batch.skip_reasons.items(), key=lambda kv: (-kv[1], kv[0])
        )
        reason_summary = ", ".join(f"{cnt} {reason}" for reason, cnt in sorted_reasons)
        logger.info("%d clone pair(s) skipped: %s", total_skipped, reason_summary)

    patch_chunks: List[str] = []
    seen_comments: Set[str] = set()
    for plan in batch.file_plans.values():
        chunk = _render_file_patch_plan(
            plan, replace_clones=replace_clones, repo_root=str(patch_root)
        )
        if chunk:
            if (
                not plan.replacements
                and not plan.module_helpers
                and not plan.method_helpers
            ):
                if chunk in seen_comments:
                    continue
                seen_comments.add(chunk)
            patch_chunks.append(chunk)

    return "\n".join(patch_chunks)
