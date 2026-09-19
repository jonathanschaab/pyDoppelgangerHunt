"""Delegation call synthesis, unit overlap filtering, and multi-file patch generation."""

from __future__ import annotations

import ast
import builtins
import difflib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import normalize_path_string
from pydoppelgangerhunt.fixer.binding import (
    _base_unit_name,
    _extract_child_indentation,
    _get_enclosing_receiver_kind,
    _has_receiver_reference,
    _is_method_of_class,
    _is_same_file_path,
    _prune_unshared_receivers,
    _resolve_effective_binding,
    find_enclosing_class,
    find_enclosing_function,
)
from pydoppelgangerhunt.fixer.depgraph import (
    ModuleDependencyGraph,
    _collect_top_level_import_nodes,
    _find_enclosing_package_root,
    _find_project_filesystem_root,
    _is_safe_repo_python_file,
    _parse_source_imports,
    _resolve_relative_import_path,
    build_module_graph,
    derive_module_import_path as _derive_module_import_path,
    find_nearest_common_package,
    resolve_shared_module_file,
)
from pydoppelgangerhunt.fixer.scope import (
    _normalize_receiver_attrs,
    dispatch_analyze_unit_variable_scope as analyze_unit_variable_scope,
)
from pydoppelgangerhunt.fixer.source import (
    _detect_indent_step,
    _find_module_helper_insertion_index,
    _find_sig_colon,
    _get_module_imported_names,
    _insert_imports_into_module,
    _is_docstring_node,
    _scan_sig_line,
    replace_unit_in_source,
)
from pydoppelgangerhunt.fixer.synthesis import (
    _extract_required_typing_imports,
    _format_call_arguments,
    synthesize_shared_helper_code,
)

logger = logging.getLogger(__name__)
_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def check_units_overlap(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> bool:
    """Determines whether two AST code units in the same file share overlapping line ranges.

    Args:
        u1: First AST unit dictionary with 'file', 'start', and 'end'.
        u2: Second AST unit dictionary with 'file', 'start', and 'end'.
        repo_root: Optional repository root path for resolving relative file paths.

    Returns:
        True if both units reside in the same normalized file path and their [start, end]
        intervals overlap; False otherwise.
    """
    f1 = normalize_path_string(str(u1.get("file") or ""))
    f2 = normalize_path_string(str(u2.get("file") or ""))
    if not f1 or not f2 or not _is_same_file_path(f1, f2, repo_root=repo_root):
        return False
    start1 = int(u1.get("start") or 1)
    end1 = int(u1.get("end") or start1)
    start2 = int(u2.get("start") or 1)
    end2 = int(u2.get("end") or start2)

    if end1 < start2 or end2 < start1:
        return False

    s_col1 = u1.get("start_col")
    e_col1 = u1.get("end_col")
    s_col2 = u2.get("start_col")
    e_col2 = u2.get("end_col")

    if start1 == end1 == start2 == end2:
        if s_col1 is not None and e_col1 is not None and s_col2 is not None and e_col2 is not None:
            sc1, ec1 = int(s_col1), int(e_col1)
            sc2, ec2 = int(s_col2), int(e_col2)
            if sc1 <= ec1 and sc2 <= ec2:
                return max(sc1, sc2) < min(ec1, ec2)
            return False

    if start1 < start2 and end1 == start2:
        if e_col1 is not None and s_col2 is not None:
            return int(s_col2) < int(e_col1)

    if start2 < start1 and end2 == start1:
        if e_col2 is not None and s_col1 is not None:
            return int(s_col1) < int(e_col2)

    return max(start1, start2) <= min(end1, end2)


def filter_overlapping_clone_units(
    units: Sequence[Dict[str, Any]],
    repo_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filters a sequence of candidate code units to retain a maximal non-overlapping subset.

    When candidates share overlapping lines/tokens within the same module, candidates
    spanning a larger line range are prioritized, with ties broken by earlier start line.

    Args:
        units: Sequence of AST unit dictionaries proposed for refactoring.
        repo_root: Optional repository root path for resolving relative file paths.

    Returns:
        List of non-overlapping units safe for concurrent refactoring within the same pass.
    """
    if not units:
        return []

    file_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for u in units:
        f_raw = str(u.get("file") or "")
        matched = False
        for rep_f, group in file_groups:
            if (not f_raw and not rep_f) or (
                f_raw and rep_f and _is_same_file_path(f_raw, rep_f, repo_root=repo_root)
            ):
                group.append(u)
                matched = True
                break
        if not matched:
            file_groups.append((f_raw, [u]))

    retained: List[Dict[str, Any]] = []
    for _, file_units in file_groups:
        sorted_candidates = sorted(
            file_units,
            key=lambda u: (
                -(int(u.get("end") or int(u.get("start") or 0)) - int(u.get("start") or 0)),
                int(u.get("start") or 0),
                int(u.get("start_col") or 0),
                str(u.get("name") or ""),
            ),
        )
        file_retained: List[Dict[str, Any]] = []
        for cand in sorted_candidates:
            if not any(check_units_overlap(cand, prev, repo_root=repo_root) for prev in file_retained):
                file_retained.append(cand)
        retained.extend(file_retained)

    return retained


def refactor_module_units(
    source_text: str,
    replacements: Sequence[Tuple[Dict[str, Any], str]],
) -> str:
    """Applies multiple non-overlapping unit replacements in reverse source order.

    Applying substitutions bottom-to-top (descending by unit 'start' line) guarantees
    that line count changes downstream never invalidate the 1-indexed source line
    coordinates of earlier units in the same file.

    Args:
        source_text: The complete original Python source code.
        replacements: Sequence of (unit, replacement_text) tuples.

    Returns:
        The refactored module source text.

    Raises:
        ValueError: If any pair of units in replacements shares overlapping line ranges.
    """
    if not replacements:
        return source_text

    rep_list = list(replacements)
    for i, (u1, _) in enumerate(rep_list):
        for u2, _ in rep_list[i + 1:]:
            if check_units_overlap(u1, u2):
                n1 = str(u1.get("name") or "unit")
                s1 = int(u1.get("start") or 1)
                e1 = int(u1.get("end") or s1)
                n2 = str(u2.get("name") or "unit")
                s2 = int(u2.get("start") or 1)
                e2 = int(u2.get("end") or s2)
                f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
                raise ValueError(
                    f"Overlapping unit collision detected between "
                    f"'{n1}' ({s1}-{e1}) and "
                    f"'{n2}' ({s2}-{e2}) in {f1}."
                )

    sorted_replacements = sorted(
        rep_list,
        key=lambda item: (
            int(item[0].get("start") or 0),
            int(item[0].get("start_col") or 0),
        ),
        reverse=True,
    )

    current_text = source_text
    for unit, replacement in sorted_replacements:
        current_text = replace_unit_in_source(current_text, unit, replacement)

    return current_text

def _build_whole_method_delegation(
    source_text: str,
    unit: Dict[str, Any],
    call_prefix: str,
    helper_name: str,
    args_str: str,
    await_prefix: str = "",
    has_return: bool = True,
    is_async: bool = False,
    has_yield: bool = False,
) -> str:
    """Builds a delegated method replacement body preserving method signature and docstring."""
    lines = source_text.splitlines(keepends=True)
    u_start = int(unit.get("start") or 1)
    u_end = int(unit.get("end") or max(u_start, len(lines)))

    lead = lines[u_start - 1] if 1 <= u_start <= len(lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]
    default_step = _detect_indent_step(indent)
    body_indent = indent + default_step

    sig_end_line: Optional[int] = None
    docstring_end_line: Optional[int] = None
    header = ""
    raw_u_name = str(unit.get("name") or "")
    base_u_name = raw_u_name.rsplit(":", maxsplit=1)[-1] if raw_u_name else ""

    try:
        tree = ast.parse(source_text)
        cand_nodes: List[Tuple[int, Union[ast.FunctionDef, ast.AsyncFunctionDef]]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_start = getattr(node, "lineno", 0)
                n_end = getattr(node, "end_lineno", n_start)
                decs = getattr(node, "decorator_list", [])
                dec_start = min(getattr(d, "lineno", n_start) for d in decs) if decs else n_start
                earliest_start = min(dec_start, n_start)
                if u_start in (n_start, earliest_start):
                    cand_nodes.append((0, node))
                elif node.name in (raw_u_name, base_u_name) and earliest_start <= u_start <= n_end:
                    cand_nodes.append((n_end - earliest_start + 1, node))
        if cand_nodes:
            cand_nodes.sort(key=lambda item: (item[0], -getattr(item[1], "lineno", 0)))
            matched_node = cand_nodes[0][1]
            if matched_node.body:
                b_cand_lines = [
                    int(getattr(b_stmt, "lineno", 0))
                    for b_stmt in matched_node.body
                    if getattr(b_stmt, "lineno", 0) > 0
                ]
                found_indent = _extract_child_indentation(lines, b_cand_lines, indent)
                if found_indent is not None:
                    body_indent = found_indent
                first_body = matched_node.body[0]
                fn_def_line = getattr(matched_node, "lineno", u_start)
                if first_body.lineno == fn_def_line:
                    b_col = getattr(first_body, "col_offset", len(lines[fn_def_line - 1]))
                    same_line = lines[fn_def_line - 1][:b_col].rstrip()
                    if not same_line.endswith(":"):
                        colon_pos = _find_sig_colon(lines[fn_def_line - 1][:b_col])
                        if colon_pos != -1:
                            same_line = lines[fn_def_line - 1][: colon_pos + 1]
                    header = "".join(lines[u_start - 1 : fn_def_line - 1]) + same_line + "\n"
                else:
                    sig_end_line = first_body.lineno - 1
                    if _is_docstring_node(first_body):
                        docstring_end_line = getattr(first_body, "end_lineno", first_body.lineno)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        pass

    if not header:
        if docstring_end_line is not None and docstring_end_line >= u_start:
            header = "".join(lines[u_start - 1 : docstring_end_line])
        elif sig_end_line is not None and sig_end_line >= u_start:
            header = "".join(lines[u_start - 1 : sig_end_line])
        else:
            hdr_lines = []
            curr_depth = 0
            for l_num in range(u_start, min(u_end + 1, len(lines) + 1)):
                ln = lines[l_num - 1]
                colon_pos, curr_depth = _scan_sig_line(ln, initial_paren_depth=curr_depth)
                if colon_pos != -1:
                    hdr_lines.append(ln[: colon_pos + 1] + "\n")
                    break
                hdr_lines.append(ln)
                if ln.rstrip().endswith(":") and curr_depth == 0:
                    break
            header = "".join(hdr_lines)

    if not header.endswith("\n"):
        header += "\n"

    body_step = (
        body_indent[len(indent):]
        if body_indent.startswith(indent) and len(body_indent) > len(indent)
        else ("\t" if "\t" in body_indent else "    ")
    )

    effective_async = is_async or bool(await_prefix.strip())
    if effective_async and has_yield:
        delegation_stmt = (
            f"{body_indent}async for _item in {call_prefix}{helper_name}({args_str}):\n"
            f"{body_indent}{body_step}yield _item\n"
        )
    elif has_yield:
        if has_return:
            delegation_stmt = (
                f"{body_indent}return (yield from {call_prefix}{helper_name}({args_str}))\n"
            )
        else:
            delegation_stmt = f"{body_indent}yield from {call_prefix}{helper_name}({args_str})\n"
    elif effective_async:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}await {call_prefix}{helper_name}({args_str})\n"
    else:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}{call_prefix}{helper_name}({args_str})\n"
    return header + delegation_stmt


def _has_unconditional_terminal_return(
    unit: Dict[str, Any], orig_lines: Sequence[str]
) -> bool:
    """Checks whether an AST code unit terminates with an unconditional return at its base indentation."""
    u_start = max(1, int(unit.get("start") or 1))
    u_end = min(len(orig_lines), int(unit.get("end") or u_start))
    if u_start > len(orig_lines) or u_start > u_end:
        return False
    cand_lines = orig_lines[u_start - 1 : u_end]
    non_empty = [ln for ln in cand_lines if ln.strip() and not ln.strip().startswith("#")]
    if not non_empty:
        return False
    base_ind = len(non_empty[0]) - len(non_empty[0].lstrip())
    last_ln = non_empty[-1]
    last_ind = len(last_ln) - len(last_ln.lstrip())
    if last_ind != base_ind:
        return False
    s = last_ln.strip()
    return s == "return" or s.startswith("return ") or s.startswith("return(") or s.startswith("return\t")


def _build_unit_delegation_call(
    target_unit: Dict[str, Any],
    orig_text: str,
    orig_lines: List[str],
    *,
    effective_binding: str,
    helper_name: str,
    inputs: List[str],
    outputs: List[str],
    scope: Dict[str, Any],
    target_inputs: Optional[List[str]] = None,
    target_outputs: Optional[List[str]] = None,
    await_prefix: str = "",
    step: Optional[str] = None,
) -> str:
    """Constructs replacement delegation call statement for a clone unit in refactoring patches."""
    u_start = int(target_unit.get("start") or 1)
    lead = orig_lines[u_start - 1] if 1 <= u_start <= len(orig_lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]

    t_fn = find_enclosing_function(orig_text, target_unit)
    t_enc = find_enclosing_class(orig_text, target_unit)
    if not _is_method_of_class(t_fn, t_enc):
        t_fn = None
    t_kind = _get_enclosing_receiver_kind(t_fn) if t_fn else "none"

    rec_param = t_fn.get("receiver_param") if t_fn else None
    rec_name = rec_param or ("cls" if t_kind == "class" else "self")

    if effective_binding == "method":
        if t_kind == "static":
            unit_call_prefix = "__class__."
            unit_receiver_omit: Optional[str] = None
        elif t_kind == "class":
            unit_call_prefix = f"{rec_name}."
            unit_receiver_omit = rec_name
        else:
            unit_call_prefix = f"{rec_name}."
            unit_receiver_omit = rec_name
    else:
        unit_call_prefix = ""
        unit_receiver_omit = (
            "receivers"
            if t_kind in ("static", "none")
            else ("self" if t_kind == "class" else "cls")
        )

    effective_targets = target_inputs if target_inputs is not None else inputs
    effective_outputs = target_outputs if target_outputs is not None else outputs

    unit_args_str = _format_call_arguments(
        inputs,
        scope.get("param_details", []),
        receiver_to_omit=unit_receiver_omit,
        target_inputs=effective_targets,
    )

    is_whole_method = target_unit.get("kind") in ("function", "closure", "method")
    if is_whole_method:
        is_init = bool(
            target_unit.get("name") == "__init__"
            or str(target_unit.get("name", "")).endswith((".__init__", ":__init__"))
        )
        has_return = False if is_init else bool(effective_outputs or scope.get("has_return", True))
        return _build_whole_method_delegation(
            orig_text,
            target_unit,
            call_prefix=unit_call_prefix,
            helper_name=helper_name,
            args_str=unit_args_str,
            await_prefix=await_prefix,
            has_return=has_return,
            is_async=bool(scope.get("is_async")),
            has_yield=bool(scope.get("has_yield")),
        )

    if target_unit.get("kind") in ("comprehension", "complex_expr"):
        call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
        return f"{await_prefix}{call_expr}"

    call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
    rep_step = step or ("\t" if "\t" in indent else "    ")
    if scope.get("has_yield") and scope.get("is_async"):
        return (
            f"{indent}async for _item in {call_expr}:\n"
            f"{indent}{rep_step}yield _item\n"
        )

    is_sync_gen = bool(scope.get("has_yield"))
    if _has_unconditional_terminal_return(target_unit, orig_lines):
        if is_sync_gen:
            return f"{indent}return (yield from {call_expr})\n"
        return f"{indent}return {await_prefix}{call_expr}\n"

    if effective_outputs:
        assign_target = ", ".join(effective_outputs) if len(effective_outputs) >= 2 else effective_outputs[0]
        rhs = (
            f"(yield from {call_expr})"
            if is_sync_gen
            else f"{await_prefix}{call_expr}"
        )
        return f"{indent}{assign_target} = {rhs}\n"

    prefix = "yield from " if is_sync_gen else await_prefix
    return f"{indent}{prefix}{call_expr}\n"


def _module_imports_target(
    source_text: str,
    target_module: str,
    current_mod: str = "",
    is_package: bool = False,
) -> bool:
    """Checks whether a module's source unconditionally imports a specific target module."""
    if not target_module:
        return False
    raw_imports = _parse_source_imports(
        source_text, current_mod, is_package=is_package
    )
    for imported_mod in raw_imports:
        if imported_mod == target_module or imported_mod.startswith(
            f"{target_module}."
        ):
            return True
    return False


class _FilePatchPlan:
    """Internal accumulator of proposed transformations for a single file."""

    def __init__(
        self,
        file_path: Path,
        orig_text: str,
        rel_path: str,
        is_new_file: bool = False,
    ) -> None:
        self.path = file_path
        self.orig_text = orig_text
        self.orig_lines = [] if is_new_file else orig_text.splitlines(keepends=True)
        self.rel_path = rel_path
        self.is_new_file = is_new_file
        self.replacements: List[Tuple[Dict[str, Any], str]] = []
        self.method_helpers: List[Tuple[int, str]] = []
        self.module_helpers: List[str] = []
        self.missing_imports: List[str] = []
        self.comments: List[str] = []
        self.used_helper_names: Set[str] = set()
        self.claimed_units: List[Dict[str, Any]] = []


def _adjust_line_for_replacements(
    target_line: int,
    reps: Sequence[Tuple[Dict[str, Any], str]],
    orig_text: str,
) -> int:
    """Adjusts a target source line number to account for upstream line expansions or contractions."""
    orig_lines = orig_text.splitlines(keepends=True)
    offset = 0
    for u, rep in reps:
        u_start = int(u.get("start") or 1)
        u_end = int(u.get("end") or u_start)
        if u_end < target_line:
            orig_count = u_end - u_start + 1
            unit_orig_text = "".join(orig_lines[u_start - 1 : u_end])
            replaced_text = replace_unit_in_source(
                unit_orig_text, {**u, "start": 1, "end": orig_count}, rep
            )
            new_count = len(replaced_text.splitlines(keepends=True))
            offset += (new_count - orig_count)
    return max(1, target_line + offset)


def _delegate_unit_in_plan(
    unit: Dict[str, Any],
    plan: _FilePatchPlan,
    helper_name: str,
    inputs: List[str],
    outputs: List[str],
    scope: Dict[str, Any],
    target_inputs: Optional[List[str]],
    target_outputs: Optional[List[str]],
    await_prefix: str,
    binding: str = "module",
    step: Optional[str] = None,
) -> None:
    """Builds and records a delegation call replacement for a unit inside a file plan."""
    if step is None:
        u_s = int(unit.get("start") or 1)
        u_lead = plan.orig_lines[u_s - 1] if 1 <= u_s <= len(plan.orig_lines) else ""
        u_ind = u_lead[: len(u_lead) - len(u_lead.lstrip())]
        step = _detect_indent_step(u_ind)

    rep_stmt = _build_unit_delegation_call(
        unit,
        plan.orig_text,
        plan.orig_lines,
        effective_binding=binding,
        helper_name=helper_name,
        inputs=inputs,
        outputs=outputs,
        scope=scope,
        target_inputs=target_inputs,
        target_outputs=target_outputs,
        await_prefix=await_prefix,
        step=step,
    )
    plan.replacements.append((unit, rep_stmt))
    plan.claimed_units.append(unit)


def _wire_cross_module_host_delegation(
    f1_plan: _FilePatchPlan,
    f2_plan: _FilePatchPlan,
    u2: Dict[str, Any],
    mod1: str,
    helper_name: str,
    pair_comment: str,
    f1_disp: str,
    f2_disp: str,
    replace_clones: bool,
    inputs: List[str],
    outputs: List[str],
    scope: Dict[str, Any],
    t_inputs2: Optional[List[str]],
    target_outs2: Optional[List[str]],
    await_prefix: str,
) -> None:
    """Wires cross-module import and delegation from target module to host module."""
    if replace_clones:
        f2_plan.comments.append(pair_comment)
        f2_plan.missing_imports.append(f"from {mod1} import {helper_name}")
        _delegate_unit_in_plan(
            u2,
            f2_plan,
            helper_name=helper_name,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            target_inputs=t_inputs2,
            target_outputs=target_outs2,
            await_prefix=await_prefix,
            binding="module",
        )
    else:
        f1_plan.comments.append(
            f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
            f"Complete refactoring by importing the helper into {f2_disp}.\n"
        )


def _attach_helper_to_plan(
    plan: _FilePatchPlan,
    binding: str,
    insert_line: int,
    helper_code: str,
    missing_imports: Sequence[str],
) -> None:
    """Attaches a synthesized helper and its missing imports to the designated file plan."""
    if binding == "method":
        plan.method_helpers.append((insert_line, helper_code))
    else:
        plan.module_helpers.append(helper_code)
    plan.missing_imports.extend(missing_imports)


def _finalize_host_unit_and_helper(
    plan: _FilePatchPlan,
    unit: Dict[str, Any],
    helper_name: str,
    inputs: List[str],
    outputs: List[str],
    scope: Dict[str, Any],
    target_inputs: Optional[List[str]],
    binding: str,
    step: str,
    insert_line: int,
    helper_code: str,
    missing_imports: Sequence[str],
    await_prefix: str,
    replace_clones: bool,
) -> None:
    """Delegates host clone unit if replace_clones is enabled and attaches synthesized helper."""
    if replace_clones:
        _delegate_unit_in_plan(
            unit,
            plan,
            helper_name=helper_name,
            inputs=inputs,
            outputs=outputs,
            scope=scope,
            target_inputs=target_inputs,
            target_outputs=outputs,
            await_prefix=await_prefix,
            binding=binding,
            step=step,
        )
    _attach_helper_to_plan(
        plan,
        binding=binding,
        insert_line=insert_line,
        helper_code=helper_code,
        missing_imports=missing_imports,
    )


def _render_file_patch_plan(
    plan: _FilePatchPlan,
    replace_clones: bool,
    repo_root: Optional[str] = None,
) -> str:
    """Renders a single cumulative unified diff for all modifications in a file plan."""
    if plan.is_new_file:
        deduped_imports = list(dict.fromkeys(plan.missing_imports))
        future_imports = [
            imp for imp in deduped_imports if imp.strip().startswith("from __future__")
        ]
        other_imports = [
            imp for imp in deduped_imports if not imp.strip().startswith("from __future__")
        ]
        ordered_imports = future_imports + other_imports
        formatted_imports = [imp.rstrip("\r\n") + "\n" for imp in ordered_imports]
        new_h_lines: List[str] = []
        for h_code in plan.module_helpers:
            new_h_lines.extend(["\n"] + [ln + "\n" for ln in h_code.splitlines()] + ["\n"])
        new_lines: List[str] = []
        if formatted_imports:
            new_lines.extend(formatted_imports + ["\n"])
        new_lines.extend(new_h_lines)
        while new_lines and not new_lines[0].strip():
            new_lines.pop(0)
        diff = difflib.unified_diff(
            [],
            new_lines,
            fromfile="/dev/null",
            tofile=f"b/{plan.rel_path}",
            n=3,
        )
        diff_str = "".join(diff)
        if not diff_str:
            return ""
        git_header = f"diff --git a/{plan.rel_path} b/{plan.rel_path}\nnew file mode 100644\n"
        deduped_comments = list(dict.fromkeys(plan.comments))
        return "".join(deduped_comments) + git_header + diff_str
    retained_cands = filter_overlapping_clone_units(
        [u for u, _ in plan.replacements], repo_root=repo_root
    )
    filtered_reps: List[Tuple[Dict[str, Any], str]] = []
    for u, rep in plan.replacements:
        if any(
            u is r
            or (
                u.get("file") == r.get("file")
                and u.get("start") == r.get("start")
                and u.get("end") == r.get("end")
            )
            for r in retained_cands
        ):
            if not any(
                check_units_overlap(u, prev_u, repo_root=repo_root)
                for prev_u, _ in filtered_reps
            ):
                filtered_reps.append((u, rep))

    if replace_clones and filtered_reps:
        current_text = refactor_module_units(plan.orig_text, filtered_reps)
    else:
        current_text = plan.orig_text

    if plan.method_helpers:
        adjusted_methods = [
            (
                _adjust_line_for_replacements(ins_line, filtered_reps, plan.orig_text)
                if replace_clones and filtered_reps
                else ins_line,
                h_code,
            )
            for ins_line, h_code in plan.method_helpers
        ]
        sorted_methods = sorted(adjusted_methods, key=lambda m: m[0], reverse=True)
        for ins_line, h_code in sorted_methods:
            c_lines = current_text.splitlines(keepends=True)
            idx = max(0, ins_line - 1)
            h_lines = [ln + "\n" for ln in h_code.splitlines()] + ["\n"]
            current_text = "".join(c_lines[:idx] + h_lines + c_lines[idx:])

    c_lines = current_text.splitlines(keepends=True)
    deduped_imports = list(dict.fromkeys(plan.missing_imports))
    lines_with_imports = _insert_imports_into_module(c_lines, deduped_imports)

    if plan.module_helpers:
        ins_idx = _find_module_helper_insertion_index(lines_with_imports)
        all_h_lines: List[str] = []
        for h_code in plan.module_helpers:
            all_h_lines.extend(["\n"] + [ln + "\n" for ln in h_code.splitlines()] + ["\n"])
        modified_lines = lines_with_imports[:ins_idx] + all_h_lines + lines_with_imports[ins_idx:]
    else:
        modified_lines = lines_with_imports

    diff = difflib.unified_diff(
        plan.orig_lines,
        modified_lines,
        fromfile=f"a/{plan.rel_path}",
        tofile=f"b/{plan.rel_path}",
        n=3,
    )
    diff_str = "".join(diff)
    if not diff_str:
        advisory_comments = [
            c for c in plan.comments if not c.startswith("# Clone Pair (")
        ]
        if advisory_comments:
            deduped_comments = list(dict.fromkeys(advisory_comments))
            return "".join(deduped_comments)
        return ""

    deduped_comments = list(dict.fromkeys(plan.comments))
    return "".join(deduped_comments) + diff_str


def _is_or_contains_bitor(node: ast.AST) -> bool:
    """Checks if an AST expression node contains a bitwise OR (|) operator."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return True
    return False


def _has_future_annotations(source_text: str) -> bool:
    """Checks whether source_text enables postponed evaluation of annotations via __future__."""
    if not source_text or "__future__" not in source_text:
        return False
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    return True
    return False


def _helper_requires_future_annotations(helper_code: str) -> bool:
    """Detects whether helper annotations use PEP 604 union syntax (|)."""
    if not helper_code or "|" not in helper_code:
        return False
    try:
        tree = ast.parse(helper_code)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = [
                *getattr(node.args, "posonlyargs", []),
                *node.args.args,
                *getattr(node.args, "kwonlyargs", []),
                *(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None),
            ]
            for arg in all_args:
                if arg.annotation and _is_or_contains_bitor(arg.annotation):
                    return True
            if node.returns and _is_or_contains_bitor(node.returns):
                return True
    return False


class _ModuleImportsResult(Tuple[Dict[str, str], List[str]]):
    """Encapsulates extracted module imports with binding classification.

    Inherits from Tuple[Dict[str, str], List[str]] so callers unpacking
    (stmts, wildcards) maintain 100% backwards compatibility.
    """

    stmts: Dict[str, str]
    wildcards: List[str]
    unconditional_names: Set[str]
    guarded_names: Set[str]
    rebound_conflicts: Set[str]

    def __new__(
        cls,
        stmts: Dict[str, str],
        wildcards: List[str],
        unconditional_names: Optional[Set[str]] = None,
        guarded_names: Optional[Set[str]] = None,
        rebound_conflicts: Optional[Set[str]] = None,
    ) -> "_ModuleImportsResult":
        instance = super().__new__(cls, (stmts, wildcards))
        instance.stmts = stmts
        instance.wildcards = wildcards
        instance.unconditional_names = (
            set() if unconditional_names is None else unconditional_names
        )
        instance.guarded_names = (
            set() if guarded_names is None else guarded_names
        )
        instance.rebound_conflicts = (
            set() if rebound_conflicts is None else rebound_conflicts
        )
        return instance


def _parse_import_node_statements(
    node: Union[ast.Import, ast.ImportFrom],
    source_mod: Optional[str] = None,
    is_package: bool = False,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Extracts (bound_name, import_stmt) pairs and wildcard strings from an AST import node."""
    name_stmts: List[Tuple[str, str]] = []
    wildcards: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if name != "*":
                if alias.asname:
                    stmt = f"import {alias.name} as {alias.asname}"
                else:
                    stmt = f"import {alias.name}"
                name_stmts.append((name, stmt))
    elif isinstance(node, ast.ImportFrom):
        level = getattr(node, "level", 0) or 0
        target_module = ""
        if level == 0 and node.module:
            target_module = node.module
        elif level > 0:
            if source_mod:
                target_module = _resolve_relative_import_path(
                    source_mod, level, node.module, is_package=is_package
                )
            else:
                dots = "." * level
                target_module = f"{dots}{node.module}" if node.module else dots

        if target_module:
            for alias in node.names:
                if alias.name == "*":
                    wildcards.append(f"from {target_module} import *")
                else:
                    name = alias.asname or alias.name
                    if alias.asname:
                        stmt = (
                            f"from {target_module} import {alias.name} as {alias.asname}"
                        )
                    else:
                        stmt = f"from {target_module} import {alias.name}"
                    name_stmts.append((name, stmt))
    return name_stmts, wildcards


def _extract_module_import_statements(
    source_text: str,
    source_mod: Optional[str] = None,
    is_package: bool = False,
) -> _ModuleImportsResult:
    """Maps imported symbol/module names to clean import statements from source text.

    Tracks final top-level bindings for rebound symbols, distinguishes unconditional
    top-level imports from runtime-guarded imports, and detects conflicting rebindings.
    """
    stmts: Dict[str, str] = {}
    wildcards: List[str] = []
    unconditional_names: Set[str] = set()
    guarded_names: Set[str] = set()
    rebound_conflicts: Set[str] = set()

    if not source_text.strip():
        return _ModuleImportsResult(stmts, wildcards)
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return _ModuleImportsResult(stmts, wildcards)

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            name_stmts, wilds = _parse_import_node_statements(
                stmt, source_mod=source_mod, is_package=is_package
            )
            wildcards.extend(wilds)
            for name, stmt_str in name_stmts:
                if name in unconditional_names and stmts.get(name) != stmt_str:
                    rebound_conflicts.add(name)
                stmts[name] = stmt_str
                unconditional_names.add(name)
        else:
            guarded_nodes = _collect_top_level_import_nodes(
                [stmt], include_classes=False
            )
            for g_node in guarded_nodes:
                name_stmts, wilds = _parse_import_node_statements(
                    g_node, source_mod=source_mod, is_package=is_package
                )
                wildcards.extend(wilds)
                for name, stmt_str in name_stmts:
                    guarded_names.add(name)
                    if name in unconditional_names and stmts.get(name) != stmt_str:
                        rebound_conflicts.add(name)
                    elif name not in stmts:
                        stmts[name] = stmt_str

    return _ModuleImportsResult(
        stmts=stmts,
        wildcards=wildcards,
        unconditional_names=unconditional_names,
        guarded_names=guarded_names,
        rebound_conflicts=rebound_conflicts,
    )


def _canonicalize_helper_relative_imports(
    helper_code: str,
    source_mods: Sequence[Union[str, Tuple[Optional[str], bool], None]] = (),
) -> str:
    """Translates relative imports inside a helper body to canonical absolute imports."""
    if not helper_code:
        return helper_code
    valid_mods: List[Tuple[str, bool]] = []
    for item in source_mods:
        if not item:
            continue
        if isinstance(item, tuple):
            if item[0]:
                valid_mods.append((item[0], bool(item[1])))
        else:
            valid_mods.append((item, False))
    if not valid_mods:
        return helper_code
    try:
        tree = ast.parse(helper_code)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return helper_code

    import_nodes: List[ast.ImportFrom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (getattr(node, "level", 0) or 0) > 0:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                import_nodes.append(node)

    if not import_nodes:
        return helper_code

    import_nodes.sort(key=lambda n: n.lineno, reverse=True)
    lines = helper_code.splitlines()

    for node in import_nodes:
        resolved_targets: Set[str] = set()
        for s_mod, s_ispkg in valid_mods:
            resolved = _resolve_relative_import_path(
                s_mod, node.level, node.module, is_package=s_ispkg
            )
            if resolved:
                resolved_targets.add(resolved)
        if len(resolved_targets) > 1:
            raise ValueError(
                "conflicting relative import in helper across clone sources"
            )
        if not resolved_targets:
            continue
        resolved = next(iter(resolved_targets))

        if node.lineno is None:
            continue
        start_idx = node.lineno - 1
        end_idx = node.end_lineno if node.end_lineno is not None else node.lineno
        if start_idx < 0 or start_idx >= len(lines) or end_idx < start_idx:
            continue

        first_line = lines[start_idx]
        indent = first_line[: len(first_line) - len(first_line.lstrip())]

        trailing_comment = ""
        last_line = lines[end_idx - 1] if 0 <= end_idx - 1 < len(lines) else first_line
        hash_pos = last_line.find("#")
        if hash_pos != -1:
            trailing_comment = f"  {last_line[hash_pos:].strip()}"

        names_str = ", ".join(
            f"{a.name} as {a.asname}" if a.asname else a.name
            for a in node.names
        )
        new_stmt = f"{indent}from {resolved} import {names_str}{trailing_comment}"
        lines[start_idx:end_idx] = [new_stmt]

    result = "\n".join(lines)
    if helper_code.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _unpack_source_item(
    item: Union[str, Tuple[Any, ...], List[Any]]
) -> Tuple[str, Optional[str], bool]:
    if isinstance(item, (tuple, list)):
        text = str(item[0]) if len(item) > 0 else ""
        mod = str(item[1]) if len(item) > 1 and item[1] is not None else None
        ispkg = bool(item[2]) if len(item) > 2 else False
        return text, mod, ispkg
    return str(item), None, False


def _extract_module_defined_names(source_text: str) -> Set[str]:
    """Extracts top-level defined names (classes, functions, variables) from source text."""
    if not source_text.strip():
        return set()
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return set()
    names: Set[str] = set()

    def _collect_top_defs(stmts: Sequence[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            names.add(sub.id)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                for sub in ast.walk(node.target):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                for sub in ast.walk(node.target):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
                _collect_top_defs(node.body)
                _collect_top_defs(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars:
                        for sub in ast.walk(item.optional_vars):
                            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                                names.add(sub.id)
                _collect_top_defs(node.body)
            elif isinstance(node, ast.If):
                _collect_top_defs(node.body)
                _collect_top_defs(node.orelse)
            elif isinstance(node, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(node, getattr(ast, "TryStar"))
            ):
                _collect_top_defs(getattr(node, "body", []))
                for handler in getattr(node, "handlers", []):
                    _collect_top_defs(getattr(handler, "body", []))
                _collect_top_defs(getattr(node, "orelse", []))
                _collect_top_defs(getattr(node, "finalbody", []))
            elif hasattr(ast, "Match") and isinstance(node, getattr(ast, "Match")):
                for case in getattr(node, "cases", []):
                    for sub in ast.walk(case.pattern):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            names.add(sub.id)
                        elif (
                            hasattr(ast, "MatchAs")
                            and isinstance(sub, getattr(ast, "MatchAs"))
                            and getattr(sub, "name", None)
                        ):
                            names.add(sub.name)
                        elif (
                            hasattr(ast, "MatchStar")
                            and isinstance(sub, getattr(ast, "MatchStar"))
                            and getattr(sub, "name", None)
                        ):
                            names.add(sub.name)
                    _collect_top_defs(case.body)

    def _collect_top_named_exprs(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if type(child).__name__ == "NamedExpr":
                tgt = getattr(child, "target", None)
                if isinstance(tgt, ast.Name) and isinstance(tgt.ctx, ast.Store):
                    names.add(tgt.id)
            _collect_top_named_exprs(child)

    _collect_top_defs(tree.body)
    _collect_top_named_exprs(tree)
    return names


def _extract_helper_symbols(
    helper_tree: ast.AST,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (free_names, defined_names, helper_local_imported_names) with lexical scope awareness.

    Parameters and local stores from nested functions/classes/comprehensions do not bleed
    into the outer helper scope, preventing false shadowing of module-level dependencies.
    """
    has_top_level_fn = isinstance(
        helper_tree, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) or (
        isinstance(helper_tree, ast.Module)
        and any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            for stmt in getattr(helper_tree, "body", [])
        )
    )
    max_helper_scope_depth = 2 if has_top_level_fn else 1
    free_names: Set[str] = set()
    defined_names: Set[str] = set()
    local_imported_names: Set[str] = set()
    scope_stack: List[Set[str]] = [set()]

    def _collect_direct_bindings(stmts: Sequence[ast.stmt]) -> Set[str]:
        bound: Set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    bound.add(alias.asname or alias.name.split(".", 1)[0])
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    bound.update(
                        n.id
                        for n in ast.walk(t)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    )
            elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
                target = stmt.target
                bound.update(
                    n.id
                    for n in ast.walk(target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                bound.update(
                    n.id
                    for n in ast.walk(stmt.target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )
                bound.update(_collect_direct_bindings(stmt.body))
                bound.update(_collect_direct_bindings(stmt.orelse))
            elif isinstance(stmt, (ast.While, ast.If)):
                bound.update(_collect_direct_bindings(stmt.body))
                bound.update(_collect_direct_bindings(stmt.orelse))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars:
                        bound.update(
                            n.id
                            for n in ast.walk(item.optional_vars)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                        )
                bound.update(_collect_direct_bindings(stmt.body))
            elif isinstance(stmt, ast.Try):
                bound.update(_collect_direct_bindings(stmt.body))
                for handler in stmt.handlers:
                    if isinstance(handler.name, str):
                        bound.add(handler.name)
                    bound.update(_collect_direct_bindings(handler.body))
                bound.update(_collect_direct_bindings(stmt.orelse))
                bound.update(_collect_direct_bindings(stmt.finalbody))
            elif type(stmt).__name__ == "Match":
                for case in getattr(stmt, "cases", []):
                    pattern = getattr(case, "pattern", None)
                    if pattern is not None:
                        for n in ast.walk(pattern):
                            m_name = getattr(n, "name", None) or getattr(n, "rest", None)
                            if isinstance(m_name, str):
                                bound.add(m_name)
                    bound.update(_collect_direct_bindings(case.body))

            for n in ast.walk(stmt):
                if (
                    isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and n is not stmt
                ):
                    continue
                if type(n).__name__ == "NamedExpr":
                    target_node = getattr(n, "target", None)
                    if isinstance(target_node, ast.Name):
                        bound.add(target_node.id)
        return bound

    class _ScopeVisitor(ast.NodeVisitor):
        def _handle_function_def(
            self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
        ) -> None:
            scope_stack[-1].add(node.name)
            if len(scope_stack) == 1:
                defined_names.add(node.name)
            for d in node.decorator_list:
                self.visit(d)
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if a.annotation:
                    self.visit(a.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                self.visit(node.args.kwarg.annotation)
            for d in node.args.defaults + [df for df in node.args.kw_defaults if df]:
                self.visit(d)
            if node.returns:
                self.visit(node.returns)

            fn_scope: Set[str] = set()
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                fn_scope.add(a.arg)
            if node.args.vararg:
                fn_scope.add(node.args.vararg.arg)
            if node.args.kwarg:
                fn_scope.add(node.args.kwarg.arg)
            fn_scope.update(_collect_direct_bindings(node.body))
            self._enter_block_scope(fn_scope, node.body)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle_function_def(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle_function_def(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            scope_stack[-1].add(node.name)
            if len(scope_stack) == 1:
                defined_names.add(node.name)
            for d in node.decorator_list:
                self.visit(d)
            for base_expr in node.bases:
                self.visit(base_expr)
            for k in node.keywords:
                self.visit(k)

            cls_scope = _collect_direct_bindings(node.body)
            self._enter_block_scope(cls_scope, node.body)

        def _enter_block_scope(
            self, child_scope: Set[str], body: Sequence[ast.stmt]
        ) -> None:
            if len(scope_stack) == 1:
                defined_names.update(child_scope)
            scope_stack.append(child_scope)
            for body_stmt in body:
                self.visit(body_stmt)
            scope_stack.pop()

        def _handle_comprehension(
            self, elts: Sequence[ast.expr], generators: Sequence[ast.comprehension]
        ) -> None:
            comp_scope: Set[str] = set()
            for gen in generators:
                self.visit(gen.iter)
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        comp_scope.add(t.id)
                for if_clause in gen.ifs:
                    scope_stack.append(comp_scope)
                    self.visit(if_clause)
                    scope_stack.pop()
            scope_stack.append(comp_scope)
            for e in elts:
                self.visit(e)
            scope_stack.pop()

        def visit_ListComp(self, node: ast.ListComp) -> None:
            self._handle_comprehension([node.elt], node.generators)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            self._handle_comprehension([node.elt], node.generators)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            self._handle_comprehension([node.elt], node.generators)

        def visit_DictComp(self, node: ast.DictComp) -> None:
            self._handle_comprehension([node.key, node.value], node.generators)

        def visit_Import(self, node: ast.Import) -> None:
            if len(scope_stack) <= max_helper_scope_depth:
                for alias in node.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    local_imported_names.add(name)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if len(scope_stack) <= max_helper_scope_depth:
                for alias in node.names:
                    name = alias.asname or alias.name
                    local_imported_names.add(name)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                if not any(node.id in s for s in reversed(scope_stack)):
                    free_names.add(node.id)

    _ScopeVisitor().visit(helper_tree)
    return free_names, defined_names, local_imported_names


def _parse_local_import_statements(
    loc_imports: Sequence[str],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    source_idx: Optional[int] = None,
) -> Dict[str, str]:
    stmts: Dict[str, str] = {}

    def _add_target(
        s_item: Any,
        lvl: int,
        mod: Optional[str],
        targets: Set[str],
    ) -> None:
        _, sm, s_pkg = _unpack_source_item(s_item)
        if sm:
            tm = _resolve_relative_import_path(sm, lvl, mod, is_package=s_pkg)
            if tm:
                targets.add(tm)

    for loc_imp in loc_imports:
        s_loc = loc_imp.strip()
        if not s_loc:
            continue
        try:
            loc_tree = ast.parse(s_loc)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue

        for node in loc_tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    stmt = (
                        f"import {alias.name} as {alias.asname}"
                        if alias.asname
                        else f"import {alias.name}"
                    )
                    if name in stmts and stmts[name] != stmt:
                        raise ValueError(
                            f"conflicting local import symbol '{name}' across clone sources"
                        )
                    stmts[name] = stmt
            elif isinstance(node, ast.ImportFrom):
                level = getattr(node, "level", 0) or 0
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    if level == 0 and node.module:
                        stmt = (
                            f"from {node.module} import {alias.name} as {alias.asname}"
                            if alias.asname
                            else f"from {node.module} import {alias.name}"
                        )
                    elif level > 0:
                        resolved_targets: Set[str] = set()
                        if source_idx is not None and 0 <= source_idx < len(source_texts):
                            _add_target(
                                source_texts[source_idx],
                                level,
                                node.module,
                                resolved_targets,
                            )
                        if not resolved_targets and source_texts:
                            matching_items = [
                                item for item in source_texts
                                if bool(_unpack_source_item(item)[0] and s_loc in (_unpack_source_item(item)[0] or ""))
                            ]
                            for item in matching_items or source_texts:
                                _add_target(
                                    item,
                                    level,
                                    node.module,
                                    resolved_targets,
                                )

                        if len(resolved_targets) > 1:
                            raise ValueError(
                                f"conflicting local import symbol '{name}' across clone sources"
                            )
                        if not resolved_targets:
                            continue
                        target_mod = next(iter(resolved_targets))
                        stmt = (
                            f"from {target_mod} import {alias.name} as {alias.asname}"
                            if alias.asname
                            else f"from {target_mod} import {alias.name}"
                        )
                    else:
                        continue

                    if name in stmts and stmts[name] != stmt:
                        raise ValueError(
                            f"conflicting local import symbol '{name}' across clone sources"
                        )
                    stmts[name] = stmt
    return stmts


def _collect_host_missing_imports(
    host_plan: _FilePatchPlan,
    helper_code: str,
    scope: Dict[str, Any],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]] = (),
    host_mod: Optional[str] = None,
    is_host_pkg: bool = False,
) -> List[str]:
    """Computes all required imports for a host plan receiving extracted helper_code.

    Considers typing symbols, scope local imports, and module-level source dependencies
    against the destination host plan's actual content and existing missing_imports.
    """
    host_content = (host_plan.orig_text or "") + "\n" + "\n".join(host_plan.missing_imports)
    host_imported = _get_module_imported_names(host_content, include_conditional=False)

    missing: List[str] = []

    # 0. Future annotations for newly synthesized modules or PEP 604 annotations
    raw_sources = [
        _unpack_source_item(s)[0]
        for s in source_texts
        if _unpack_source_item(s)[0]
    ]
    needs_future = any(_has_future_annotations(s) for s in raw_sources) or (
        _helper_requires_future_annotations(helper_code)
    )

    if (
        needs_future
        and not _has_future_annotations(host_plan.orig_text or "")
        and "from __future__ import annotations" not in host_plan.missing_imports
    ):
        missing.append("from __future__ import annotations")

    # 1. Typing annotations
    needed_typing = _extract_required_typing_imports(helper_code)
    missing_typing = [s for s in needed_typing if s not in host_imported]
    if missing_typing:
        missing.append(f"from typing import {', '.join(sorted(missing_typing))}")

    # 2. Local imports discovered during scope analysis
    raw_units_local = scope.get("local_imports_by_unit")
    unit_local_maps: List[Dict[str, str]] = []
    if raw_units_local is not None:
        for u_idx, raw_list in enumerate(raw_units_local):
            s_idx = u_idx if u_idx < len(source_texts) else 0
            u_map = _parse_local_import_statements(raw_list, source_texts, source_idx=s_idx)
            unit_local_maps.append(u_map)
    else:
        flat_loc = scope.get("local_imports", [])
        if len(source_texts) <= 1:
            u_map = _parse_local_import_statements(flat_loc, source_texts, source_idx=0)
            unit_local_maps = [u_map]
        else:
            for s_idx, item in enumerate(source_texts):
                s_text = _unpack_source_item(item)[0] or ""
                attributed = [
                    stmt_str for stmt_str in flat_loc
                    if stmt_str.strip() in s_text
                ]
                if not attributed and flat_loc:
                    attributed = list(flat_loc)
                u_map = _parse_local_import_statements(attributed, source_texts, source_idx=s_idx)
                unit_local_maps.append(u_map)

    local_import_stmts: Dict[str, str] = {}
    for u_map in unit_local_maps:
        for name, stmt in u_map.items():
            if name in local_import_stmts and local_import_stmts[name] != stmt:
                raise ValueError(
                    f"conflicting local import symbol '{name}' across clone sources"
                )
            local_import_stmts[name] = stmt

    # 3. Source dependencies referenced in helper_code
    try:
        helper_tree = ast.parse(helper_code)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        helper_tree = None

    if helper_tree is not None and source_texts:
        free_names, defined_in_helper, helper_local_imported_names = _extract_helper_symbols(
            helper_tree
        )

        is_same_file_host = (
            not host_plan.is_new_file
            and len(source_texts) == 1
            and _unpack_source_item(source_texts[0])[0] == host_plan.orig_text
        )

        source_info: List[Tuple[_ModuleImportsResult, List[str], Set[str]]] = []
        for item in source_texts:
            src_text, src_mod, src_ispkg = _unpack_source_item(item)
            if not src_text:
                continue
            s_res = _extract_module_import_statements(
                src_text, source_mod=src_mod, is_package=src_ispkg
            )
            s_defs = _extract_module_defined_names(src_text)
            source_info.append((s_res, s_res.wildcards, s_defs))

        source_bound_names: Set[str] = (
            set(local_import_stmts.keys()) | helper_local_imported_names
        )
        for s_res, _, s_defs in source_info:
            source_bound_names.update(s_res.stmts.keys())
            source_bound_names.update(s_defs)

        effective_builtins = _BUILTIN_NAMES - source_bound_names
        effective_typing = set(needed_typing) - source_bound_names
        ignored_names = effective_builtins | defined_in_helper | effective_typing
        symbol_to_stmt: Dict[str, str] = dict(local_import_stmts)
        source_hoisted_symbols: Set[str] = set()
        host_defs: Optional[Set[str]] = None

        symbols_to_check = set(free_names) | (helper_local_imported_names & source_bound_names)
        for loaded_name in sorted(symbols_to_check):
            if loaded_name in ignored_names:
                continue

            num_units = max(len(unit_local_maps), len(source_texts))
            unit_bindings: List[Tuple[str, Optional[str], Optional[str], int, int]] = []
            for u_idx in range(num_units):
                u_loc = unit_local_maps[u_idx] if u_idx < len(unit_local_maps) else {}
                s_idx = u_idx if u_idx < len(source_texts) else 0
                s_res, s_wild, s_defs = source_info[s_idx]
                _, s_mod, _ = _unpack_source_item(source_texts[s_idx])

                if loaded_name in u_loc:
                    b_type = "import"
                    b_val = u_loc[loaded_name]
                elif loaded_name in s_res.stmts:
                    if loaded_name in s_defs:
                        raise ValueError(
                            f"conflicting imported symbol '{loaded_name}' rebound by local definition within source module"
                        )
                    if loaded_name in s_res.rebound_conflicts:
                        raise ValueError(
                            f"conflicting imported symbol '{loaded_name}' rebound within source module"
                        )
                    if not is_same_file_host and (
                        loaded_name in s_res.guarded_names
                        and loaded_name not in s_res.unconditional_names
                    ):
                        raise ValueError(
                            f"symbol '{loaded_name}' relies on runtime-guarded import in source module"
                        )
                    b_type = "import"
                    b_val = s_res.stmts[loaded_name]
                elif loaded_name in s_defs:
                    b_type = "def"
                    b_val = s_mod or f"source_{s_idx}"
                elif bool(s_wild):
                    b_type = "wildcard"
                    b_val = "*"
                elif loaded_name in _BUILTIN_NAMES:
                    b_type = "builtin"
                    b_val = loaded_name
                else:
                    b_type = "unresolved"
                    b_val = None

                unit_bindings.append((b_type, b_val, s_mod, s_idx, u_idx))

            # 1. Wildcard check: reject wildcard-dependent cross-file extraction,
            # or mixed wildcard/explicit bindings within the same file.
            has_wildcard = any(b[0] == "wildcard" for b in unit_bindings)
            has_explicit = any(b[0] in ("import", "def") for b in unit_bindings)
            if has_wildcard and (not is_same_file_host or has_explicit):
                raise ValueError(
                    f"symbol '{loaded_name}' potentially relies on wildcard import"
                )

            # 2. Builtin shadowed by local import or def while another unit uses builtin
            has_builtin = any(b[0] == "builtin" for b in unit_bindings)
            has_shadowed = any(b[0] in ("import", "def") for b in unit_bindings)
            if has_builtin and has_shadowed:
                raise ValueError(
                    f"conflicting imported symbol '{loaded_name}' across clone sources"
                )

            # 3. If targeting a new shared module, any symbol without an agreed import cannot be resolved
            has_import = any(b[0] == "import" for b in unit_bindings)
            if host_plan.is_new_file and not has_import:
                raise ValueError(
                    f"unresolved symbol '{loaded_name}' in shared module"
                )

            # 4. Conflicting local definitions across different clone modules
            def_mods = {b[1] for b in unit_bindings if b[0] == "def"}
            if len(def_mods) > 1:
                raise ValueError(
                    f"conflicting local definition '{loaded_name}' across clone sources"
                )

            # 5. Check imports across units
            has_unresolved = any(b[0] == "unresolved" for b in unit_bindings)
            if has_import and (has_builtin or has_unresolved):
                raise ValueError(
                    f"conflicting imported symbol '{loaded_name}' across clone sources"
                )

            agreed_stmt: Optional[str] = None
            for b_type, b_val, _, _, _ in unit_bindings:
                if b_type == "import":
                    imp_stmt = b_val
                    if imp_stmt is None:
                        continue
                    if def_mods:
                        def_mod = next(iter(def_mods))
                        is_from_def = bool(def_mod) and imp_stmt.startswith(f"from {def_mod} import ")
                        if not is_from_def:
                            raise ValueError(
                                f"conflicting imported symbol '{loaded_name}' across clone sources"
                            )
                    if agreed_stmt is None:
                        agreed_stmt = imp_stmt
                    elif agreed_stmt != imp_stmt:
                        raise ValueError(
                            f"conflicting imported symbol '{loaded_name}' across clone sources"
                        )

            if agreed_stmt is not None:
                symbol_to_stmt[loaded_name] = agreed_stmt
                if (
                    loaded_name not in helper_local_imported_names
                    and not (
                        is_same_file_host
                        and any(
                            loaded_name in s_info[0].guarded_names
                            and loaded_name not in s_info[0].unconditional_names
                            for s_info in source_info
                        )
                    )
                ):
                    source_hoisted_symbols.add(loaded_name)
            else:
                # No agreed import exists: verify symbol resolution in host/caller modules
                if not is_same_file_host:
                    if host_plan.is_new_file:
                        raise ValueError(
                            f"unresolved symbol '{loaded_name}' in shared module"
                        )
                    if host_defs is None:
                        host_defs = _extract_module_defined_names(host_plan.orig_text or "")
                    if loaded_name not in host_imported and loaded_name not in host_defs:
                        raise ValueError(
                            f"unresolved symbol '{loaded_name}' in host module"
                        )
                    for item, (s_res, _, s_defs) in zip(source_texts, source_info):
                        src_text, s_mod, _ = _unpack_source_item(item)
                        if (
                            (s_mod is not None and host_mod is not None and s_mod == host_mod)
                            or (
                                (s_mod is None or host_mod is None)
                                and src_text == host_plan.orig_text
                            )
                        ):
                            continue
                        if loaded_name in s_defs:
                            raise ValueError(
                                f"conflicting local definition '{loaded_name}' across clone sources"
                            )
                        if loaded_name not in s_res.stmts and not any(
                            loaded_name in u for u in unit_local_maps
                        ):
                            raise ValueError(
                                f"unresolved symbol '{loaded_name}' in caller module"
                            )
                else:
                    if host_defs is None:
                        host_defs = _extract_module_defined_names(host_plan.orig_text or "")
                    if loaded_name not in host_imported and loaded_name not in host_defs:
                        raise ValueError(
                            f"unresolved symbol '{loaded_name}' in host module"
                        )

        # Treat bindings imported from host_mod as satisfied by host's local definition
        if not host_plan.is_new_file:
            if host_defs is None:
                host_defs = _extract_module_defined_names(host_plan.orig_text or "")
            effective_host_mod = host_mod
            if effective_host_mod is None:
                for item in source_texts:
                    src_text, s_mod, _ = _unpack_source_item(item)
                    if src_text == host_plan.orig_text and s_mod:
                        effective_host_mod = s_mod
                        break
            self_imported = [
                sname
                for sname, stmt in symbol_to_stmt.items()
                if sname in host_defs
                and (
                    (
                        bool(effective_host_mod)
                        and (
                            stmt.startswith(f"from {effective_host_mod} import ")
                            or stmt == f"import {effective_host_mod}"
                            or stmt.startswith(f"import {effective_host_mod} as ")
                        )
                    )
                    or not effective_host_mod
                )
            ]
            for sname in self_imported:
                symbol_to_stmt.pop(sname, None)

        # Cross-check symbol_to_stmt against host module's existing and queued imports
        host_existing_res = _extract_module_import_statements(
            host_plan.orig_text or "", source_mod=host_mod, is_package=is_host_pkg
        )
        host_existing_stmts = host_existing_res.stmts
        host_queued_res = _extract_module_import_statements(
            "\n".join(host_plan.missing_imports), source_mod=host_mod, is_package=is_host_pkg
        )
        host_queued_stmts = host_queued_res.stmts
        all_host_stmts: Dict[str, str] = {**host_existing_stmts, **host_queued_stmts}

        for hname, hstmt in all_host_stmts.items():
            if hname in symbol_to_stmt:
                if host_defs is not None and hname in host_defs:
                    raise ValueError(
                        f"conflicting imported symbol '{hname}' rebound by local definition within host module"
                    )
                if hname in host_existing_res.rebound_conflicts:
                    raise ValueError(
                        f"conflicting imported symbol '{hname}' rebound within host module"
                    )
                if symbol_to_stmt.pop(hname) != hstmt:
                    raise ValueError(
                        f"conflicting imported symbol '{hname}' across clone sources"
                    )

        existing_and_queued = (
            set(all_host_stmts.values()) | set(host_plan.missing_imports)
        )
        unmatched_host = set(symbol_to_stmt.keys()) & host_imported
        for sname in unmatched_host:
            cand_stmt = symbol_to_stmt.pop(sname)
            if cand_stmt not in existing_and_queued:
                raise ValueError(
                    f"conflicting imported symbol '{sname}' with host module"
                )

        for sname, stmt in symbol_to_stmt.items():
            if sname in source_hoisted_symbols:
                if stmt not in missing and stmt not in existing_and_queued:
                    missing.append(stmt)

    return missing


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
            target_path.resolve() if target_path.is_absolute() else target_path
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
    return str(target_path.name)


def _is_same_file_or_resolved(p1: Path, p2: Path) -> bool:
    """Checks whether two paths refer to the same file or resolved target."""
    if p1 == p2:
        return True
    try:
        return p1.resolve() == p2.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _resolve_safe_clone_file_path(
    raw_path: str, candidate_roots: Sequence[Path]
) -> Optional[Path]:
    """Resolves raw_path across candidate roots, rejecting symlinks and resolution errors.

    Relative paths are resolved strictly against the primary scan root (candidate_roots[0]).
    Absolute paths are validated against candidate_roots.
    """
    if not raw_path or not candidate_roots:
        return None
    try:
        p = Path(raw_path)
        roots_to_check = candidate_roots if p.is_absolute() else (candidate_roots[0],)
        for root_dir in roots_to_check:
            safe = _is_safe_repo_python_file(raw_path, root_dir)
            if safe is not None:
                return safe
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
    method_binding: str = "auto",
    cross_file_strategy: str = "auto",
    shared_module_name: str = "_common.py",
    depgraph: Optional[ModuleDependencyGraph] = None,
) -> str:
    """Generates a git-apply compatible unified diff patch proposing shared helper extractions."""
    if not clones:
        return ""

    fs_root = Path(repo_root or os.getcwd()).resolve()
    if fs_root.is_file():
        fs_root = fs_root.parent
    patch_root = _find_project_filesystem_root(fs_root)
    root = fs_root
    import_root = _find_enclosing_package_root(fs_root)
    cross_file_strategy = (cross_file_strategy or "auto").strip().lower()
    if cross_file_strategy not in (
        "auto",
        "host_module",
        "host",
        "shared_module",
        "shared",
        "skip",
    ):
        cross_file_strategy = "auto"
    type_merge_strategy = (type_merge_strategy or "fallback_any").strip().lower()
    if type_merge_strategy not in ("fallback_any", "union", "intersect", "strict"):
        type_merge_strategy = "fallback_any"
    method_binding = (method_binding or "auto").strip().lower()
    if method_binding not in ("auto", "method", "module"):
        method_binding = "auto"
    graph_holder: List[Optional[ModuleDependencyGraph]] = [depgraph]

    def _get_depgraph() -> ModuleDependencyGraph:
        g = graph_holder[0]
        if g is None:
            g = build_module_graph(import_root)
            graph_holder[0] = g
        return g

    file_plans: Dict[Path, _FilePatchPlan] = {}

    def _get_plan(
        file_p: Path, rel_f: str, text: str, is_new_file: bool = False
    ) -> _FilePatchPlan:
        if file_p not in file_plans:
            file_plans[file_p] = _FilePatchPlan(
                file_p, text, rel_f, is_new_file=is_new_file
            )
        return file_plans[file_p]

    for sim, u1, u2 in clones:
        f1_raw = normalize_path_string(str(u1.get("file") or ""), strip_anchor=True)
        if not f1_raw:
            continue
        f1_path = _resolve_safe_clone_file_path(
            f1_raw, (fs_root, patch_root, import_root)
        )
        if f1_path is None:
            continue
        try:
            if not f1_path.is_file() or f1_path.is_symlink():
                continue
        except (OSError, RuntimeError, ValueError):
            continue

        rel_f1 = _format_patch_relative_path(f1_path, patch_root, fs_root)

        f1_plan = file_plans.get(f1_path)
        if f1_plan is None:
            try:
                orig_text = f1_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            f1_plan = _get_plan(f1_path, rel_f1, orig_text)
        else:
            orig_text = f1_plan.orig_text

        orig_lines = f1_plan.orig_lines

        f2_raw = normalize_path_string(str(u2.get("file") or ""), strip_anchor=True)
        is_same_file = _is_same_file_path(f1_raw, f2_raw, repo_root=str(root))

        enc1 = find_enclosing_class(orig_text, u1)
        fn1 = find_enclosing_function(orig_text, u1)

        enc2 = None
        fn2 = None
        f2_plan: Optional[_FilePatchPlan] = None
        if is_same_file:
            enc2 = find_enclosing_class(orig_text, u2)
            fn2 = find_enclosing_function(orig_text, u2)
        elif f2_raw:
            f2_path = _resolve_safe_clone_file_path(
                f2_raw, (fs_root, patch_root, import_root)
            )
            try:
                is_f2_safe = bool(
                    f2_path is not None
                    and f2_path.is_file()
                    and not f2_path.is_symlink()
                )
            except (OSError, RuntimeError, ValueError):
                is_f2_safe = False
            if is_f2_safe and f2_path is not None:
                rel_f2 = _format_patch_relative_path(f2_path, patch_root, fs_root)
                f2_plan = file_plans.get(f2_path)
                if f2_plan is None:
                    try:
                        f2_text = f2_path.read_text(encoding="utf-8")
                        f2_plan = _get_plan(f2_path, rel_f2, f2_text)
                    except (OSError, UnicodeDecodeError):
                        f2_plan = None
                if f2_plan is not None:
                    enc2 = find_enclosing_class(f2_plan.orig_text, u2)
                    fn2 = find_enclosing_function(f2_plan.orig_text, u2)

        if replace_clones:
            u1_claimed = any(
                check_units_overlap(u1, prev_u, repo_root=str(root))
                for prev_u in f1_plan.claimed_units
            )
            if is_same_file:
                u2_claimed = any(
                    check_units_overlap(u2, prev_u, repo_root=str(root))
                    for prev_u in f1_plan.claimed_units
                )
            else:
                u2_claimed = (
                    any(
                        check_units_overlap(u2, prev_u, repo_root=str(root))
                        for prev_u in f2_plan.claimed_units
                    )
                    if f2_plan is not None
                    else False
                )
            if u1_claimed or u2_claimed:
                continue

        is_same_class = bool(
            is_same_file
            and enc1
            and enc2
            and enc1["name"] == enc2["name"]
            and enc1["start"] == enc2["start"]
        )

        if not _is_method_of_class(fn1, enc1):
            fn1 = None
        if not _is_method_of_class(fn2, enc2):
            fn2 = None

        non_method_kinds = ("closure", "class", "comprehension", "data_table", "complex_expr")
        fn1_kind = _get_enclosing_receiver_kind(fn1) if fn1 else (u1.get("receiver_kind") or ("instance" if enc1 and u1.get("kind") not in non_method_kinds else "none"))
        fn2_kind = _get_enclosing_receiver_kind(fn2) if fn2 else (u2.get("receiver_kind") or ("instance" if enc2 and u2.get("kind") not in non_method_kinds else "none"))
        receiver_kinds_differ = bool(fn1_kind != fn2_kind)
        is_in_method = bool(
            fn1
            and fn2
            and u1.get("kind") not in ("comprehension", "complex_expr")
            and u2.get("kind") not in ("comprehension", "complex_expr")
            and (fn1.get("receiver_param") or fn1.get("is_static"))
            and (fn2.get("receiver_param") or fn2.get("is_static"))
        )
        if fn1 and "receiver_param" not in u1:
            u1["receiver_param"] = fn1.get("receiver_param")
        if fn2 and "receiver_param" not in u2:
            u2["receiver_param"] = fn2.get("receiver_param")
        if fn1 and fn2:
            is_static = bool(fn1.get("is_static") and fn2.get("is_static"))
        else:
            is_static = bool((fn1 and fn1.get("is_static")) or (fn2 and fn2.get("is_static")))

        s1 = analyze_unit_variable_scope(u1, repo_root=str(root))
        s2 = analyze_unit_variable_scope(u2, repo_root=str(root))
        if bool(s1.get("is_async")) != bool(s2.get("is_async")):
            continue
        if bool(s1.get("has_yield")) != bool(s2.get("has_yield")):
            continue
        if s1.get("nonlocals") or s2.get("nonlocals"):
            continue
        if not is_same_file and (s1.get("globals") or s2.get("globals")):
            continue
        rec1 = u1.get("receiver_param") or ("cls" if fn1_kind == "class" else "self")
        rec2 = u2.get("receiver_param") or ("cls" if fn2_kind == "class" else "self")
        if (
            _normalize_receiver_attrs(s1.get("attrs_read", []), rec1)
            != _normalize_receiver_attrs(s2.get("attrs_read", []), rec2)
            or _normalize_receiver_attrs(s1.get("attrs_written", []), rec1)
            != _normalize_receiver_attrs(s2.get("attrs_written", []), rec2)
        ):
            continue

        hazards1 = set(s1.get("control_flow_hazards", []))
        hazards2 = set(s2.get("control_flow_hazards", []))
        if any(h in ("naked_break", "naked_continue") for h in hazards1 | hazards2):
            continue
        if "embedded_return" in hazards1 | hazards2:
            lines1 = orig_lines
            lines2 = orig_lines if is_same_file else (f2_plan.orig_lines if f2_plan else [])
            if not (
                _has_unconditional_terminal_return(u1, lines1)
                and _has_unconditional_terminal_return(u2, lines2)
            ):
                continue

        if receiver_kinds_differ:
            if _has_receiver_reference(u1, s1, repo_root=str(root)) or _has_receiver_reference(u2, s2, repo_root=str(root)):
                continue

        effective_binding = _resolve_effective_binding(
            method_binding,
            is_same_class,
            is_static=is_static,
            receiver_kinds_differ=receiver_kinds_differ,
            is_in_method=is_in_method,
            receiver_kind=fn1_kind,
        )

        has_super = bool(s1.get("has_super") or s2.get("has_super"))
        if has_super and (effective_binding == "module" or is_static):
            continue

        has_mangled = bool(s1.get("has_mangled_names") or s2.get("has_mangled_names"))
        if has_mangled and (effective_binding != "method" or not is_same_class or is_static):
            continue

        helper_indent = (
            enc1["method_indent"]
            if (effective_binding == "method" and enc1)
            else ("    " if effective_binding == "method" else "")
        )
        step = (
            enc1["method_indent"][len(enc1["indent"]):]
            if (
                enc1
                and enc1.get("indent") is not None
                and enc1["method_indent"].startswith(enc1["indent"])
                and len(enc1["method_indent"]) > len(enc1["indent"])
            )
            else None
        )
        if step is None:
            u1_s = int(u1.get("start") or 1)
            u1_lead = orig_lines[u1_s - 1] if 1 <= u1_s <= len(orig_lines) else ""
            u1_ind = u1_lead[: len(u1_lead) - len(u1_lead.lstrip())]
            step = _detect_indent_step(u1_ind)

        scope = analyze_unit_variable_scope(u1, u2, repo_root=str(root))
        inputs = list(scope.get("inputs", []))
        if effective_binding == "module":
            inputs = _prune_unshared_receivers(inputs, u1, u2, s1, s2, repo_root=str(root))
        outputs = [
            v for v in scope.get("outputs", [])
            if v not in scope.get("globals", [])
            and v not in scope.get("nonlocals", [])
        ]
        u1_outs = [
            v for v in s1.get("outputs", [])
            if v not in s1.get("globals", [])
            and v not in s1.get("nonlocals", [])
        ]
        u2_outs = [
            v for v in s2.get("outputs", [])
            if v not in s2.get("globals", [])
            and v not in s2.get("nonlocals", [])
        ]
        if set(u2_outs) == set(outputs):
            target_outs2 = outputs
        elif len(u2_outs) == len(outputs):
            out_map = {o: o for o in set(outputs) & set(u2_outs)}
            rem_o = [o for o in outputs if o not in out_map]
            rem_u2 = [o for o in u2_outs if o not in out_map]
            for o1, o2 in zip(rem_o, rem_u2):
                out_map[o1] = o2
            target_outs2 = [out_map.get(o, o) for o in outputs]
        else:
            target_outs2 = outputs
        t_inputs1 = list(s1.get("inputs", []))
        t_inputs2 = list(s2.get("inputs", []))
        if effective_binding == "module":
            t_inputs1 = _prune_unshared_receivers(
                t_inputs1, u1, u2, s1, s2, repo_root=str(root)
            )
            t_inputs2 = _prune_unshared_receivers(
                t_inputs2, u2, u1, s2, s1, repo_root=str(root)
            )

        if replace_clones and (
            len(t_inputs1) != len(inputs)
            or len(t_inputs2) != len(inputs)
            or len(u1_outs) != len(outputs)
            or len(u2_outs) != len(outputs)
        ):
            continue

        base_name1 = _base_unit_name(u1)
        base_name2 = _base_unit_name(u2)
        base_helper = (
            f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
        )
        shared_p: Optional[Path] = None
        target_host_plan: Optional[_FilePatchPlan] = None
        target_host_text: Optional[str] = None
        cross_file_action = cross_file_strategy
        if not is_same_file and cross_file_action in ("skip", "none"):
            continue
        if cross_file_action == "auto" and not is_same_file and f2_plan is not None:
            common_dir = find_nearest_common_package(f1_path, f2_plan.path, patch_root)
            try:
                res_common = common_dir.resolve()
                res_patch_root = patch_root.resolve()
                res_src = (patch_root / "src").resolve()
                res_import_root = import_root.resolve()
                is_top_level = res_common in (res_patch_root, res_src, res_import_root)
            except (OSError, RuntimeError, ValueError):
                is_top_level = False
            if is_top_level:
                cross_file_action = "host_module"
            else:
                cross_file_action = "shared_module"

        if (
            not is_same_file
            and f2_plan is not None
            and cross_file_action in ("shared_module", "shared")
        ):
            maybe_shared_p = _safely_resolve_shared_module_file(
                f1_path, f2_plan.path, patch_root, shared_module_name, f1_plan, f2_plan
            )
            if maybe_shared_p is None:
                continue
            shared_p = maybe_shared_p

            if _is_same_file_or_resolved(shared_p, f1_path):
                target_host_plan = f1_plan
            elif _is_same_file_or_resolved(shared_p, f2_plan.path):
                target_host_plan = f2_plan
            else:
                target_host_plan = file_plans.get(shared_p)
                try:
                    if target_host_plan is None and shared_p.is_file() and not shared_p.is_symlink():
                        try:
                            shared_p.resolve().relative_to(patch_root.resolve())
                            target_host_text = shared_p.read_text(encoding="utf-8")
                        except (OSError, UnicodeDecodeError, ValueError, RuntimeError):
                            pass
                except (OSError, RuntimeError, ValueError):
                    pass

        helper_name = base_helper
        h_idx = 2
        while (
            helper_name in f1_plan.used_helper_names
            or bool(re.search(rf"\b{re.escape(helper_name)}\b", orig_text))
            or (
                f2_plan is not None
                and (
                    helper_name in f2_plan.used_helper_names
                    or bool(re.search(rf"\b{re.escape(helper_name)}\b", f2_plan.orig_text))
                )
            )
            or (
                target_host_plan is not None
                and (
                    helper_name in target_host_plan.used_helper_names
                    or bool(re.search(rf"\b{re.escape(helper_name)}\b", target_host_plan.orig_text))
                )
            )
            or (
                target_host_text is not None
                and bool(re.search(rf"\b{re.escape(helper_name)}\b", target_host_text))
            )
        ):
            helper_name = f"{base_helper}_{h_idx}"
            h_idx += 1
        f1_plan.used_helper_names.add(helper_name)
        if f2_plan is not None:
            f2_plan.used_helper_names.add(helper_name)
        if target_host_plan is not None:
            target_host_plan.used_helper_names.add(helper_name)

        helper_code = synthesize_shared_helper_code(
            u1,
            u2,
            type_merge_strategy=type_merge_strategy,
            method_binding=effective_binding,
            indent=helper_indent,
            is_static=is_static,
            receiver_kind=fn1_kind if effective_binding == "method" else None,
            step=step,
            repo_root=str(root),
            helper_name=helper_name,
        )
        if not helper_code:
            continue

        await_prefix = "await " if scope.get("is_async") else ""

        f1_disp = normalize_path_string(str(u1.get("file") or "file1"), strip_anchor=False)
        f2_disp = normalize_path_string(str(u2.get("file") or "file2"), strip_anchor=False)
        pair_comment = f"# Clone Pair ({sim:.1%}): {f1_disp} <===> {f2_disp}\n"
        f1_plan.comments.append(pair_comment)

        candidate_units = [u1]
        if is_same_file and not check_units_overlap(u1, u2, repo_root=str(root)):
            candidate_units.append(u2)

        earliest_unit = min(candidate_units, key=lambda u: int(u.get("start") or 1))
        enc_fn_earliest = (
            find_enclosing_function(orig_text, earliest_unit)
            if effective_binding == "method"
            else None
        )
        insert_line = (
            enc_fn_earliest["start"] if enc_fn_earliest else int(earliest_unit.get("start") or 1)
        )

        if is_same_file:
            mod1 = _derive_module_import_path(f1_path, import_root)
            is_pkg1 = f1_path.name == "__init__.py"
            helper_code, maybe_imports = _safely_prepare_helper_and_imports(
                host_plan=f1_plan,
                helper_code=helper_code,
                scope=scope,
                source_texts=[(orig_text, mod1, is_pkg1)],
                f1_plan=f1_plan,
                f2_plan=f2_plan,
                host_mod=mod1,
                is_host_pkg=is_pkg1,
            )
            if maybe_imports is None:
                continue
            host_imports = maybe_imports

            if replace_clones:
                _delegate_unit_in_plan(
                    u1,
                    f1_plan,
                    helper_name=helper_name,
                    inputs=inputs,
                    outputs=outputs,
                    scope=scope,
                    target_inputs=t_inputs1,
                    target_outputs=outputs,
                    await_prefix=await_prefix,
                    binding=effective_binding,
                    step=step,
                )
                if len(candidate_units) > 1:
                    _delegate_unit_in_plan(
                        u2,
                        f1_plan,
                        helper_name=helper_name,
                        inputs=inputs,
                        outputs=outputs,
                        scope=scope,
                        target_inputs=t_inputs2,
                        target_outputs=target_outs2,
                        await_prefix=await_prefix,
                        binding=effective_binding,
                        step=step,
                    )
            _attach_helper_to_plan(
                f1_plan,
                binding=effective_binding,
                insert_line=insert_line,
                helper_code=helper_code,
                missing_imports=host_imports,
            )
        elif f2_plan is not None:
            if cross_file_action in ("skip", "none"):
                continue
            if cross_file_action in ("shared_module", "shared"):
                if shared_p is None:
                    maybe_shared_p = _safely_resolve_shared_module_file(
                        f1_path, f2_plan.path, patch_root, shared_module_name, f1_plan, f2_plan
                    )
                    if maybe_shared_p is None:
                        continue
                    shared_p = maybe_shared_p

                rel_shared = _format_patch_relative_path(shared_p, patch_root, fs_root)

                if _is_same_file_or_resolved(shared_p, f1_path):
                    host_plan = f1_plan
                    callers = [(f2_plan, u2, t_inputs2, target_outs2)]
                elif _is_same_file_or_resolved(shared_p, f2_plan.path):
                    host_plan = f2_plan
                    callers = [(f1_plan, u1, t_inputs1, outputs)]
                else:
                    shared_plan = file_plans.get(shared_p)
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
                            skip_msg = (
                                f"# Note: Cross-module clone pair; shared module path {rel_shared} "
                                f"is {kind_desc}; skipping extraction.\n"
                            )
                            f1_plan.comments.append(skip_msg)
                            f2_plan.comments.append(skip_msg)
                            continue

                        try:
                            is_existing_file = shared_p.is_file()
                        except (OSError, RuntimeError, ValueError):
                            is_existing_file = False

                        if is_existing_file:
                            try:
                                shared_p.resolve().relative_to(patch_root.resolve())
                                shared_text = shared_p.read_text(encoding="utf-8")
                                shared_plan = _get_plan(
                                    shared_p, rel_shared, shared_text, is_new_file=False
                                )
                            except (OSError, UnicodeDecodeError, ValueError, RuntimeError):
                                read_err_msg = (
                                    f"# Note: Cross-module clone pair; shared module file {rel_shared} "
                                    f"exists but could not be read; skipping extraction.\n"
                                )
                                f1_plan.comments.append(read_err_msg)
                                f2_plan.comments.append(read_err_msg)
                                continue
                        else:
                            shared_plan = _get_plan(
                                shared_p, rel_shared, "", is_new_file=True
                            )
                    host_plan = shared_plan
                    callers = [
                        (f1_plan, u1, t_inputs1, outputs),
                        (f2_plan, u2, t_inputs2, target_outs2),
                    ]

                mod1 = _derive_module_import_path(f1_path, import_root)
                mod2 = _derive_module_import_path(f2_plan.path, import_root)
                is_pkg1 = f1_path.name == "__init__.py"
                is_pkg2 = f2_plan.path.name == "__init__.py"
                mod_host = _derive_module_import_path(host_plan.path, import_root)
                is_host_pkg = host_plan.path.name == "__init__.py"
                helper_code, maybe_imports = _safely_prepare_helper_and_imports(
                    host_plan=host_plan,
                    helper_code=helper_code,
                    scope=scope,
                    source_texts=[(orig_text, mod1, is_pkg1), (f2_plan.orig_text, mod2, is_pkg2)],
                    f1_plan=f1_plan,
                    f2_plan=f2_plan,
                    host_mod=mod_host,
                    is_host_pkg=is_host_pkg,
                )
                if maybe_imports is None:
                    continue
                host_imports = maybe_imports

                host_disp = normalize_path_string(str(host_plan.rel_path), strip_anchor=False)
                dg = _get_depgraph()
                if mod_host and host_imports:
                    host_cycle = dg.check_cycle_if_imports_added(
                        mod_host,
                        host_imports,
                        file_path=host_plan.path,
                        is_package=is_host_pkg,
                    )
                    if host_cycle:
                        cycle_desc = f"Circular import detected (cycle: {' -> '.join(host_cycle)})"
                        cycle_msg = (
                            f"# Note: Cross-module clone pair; helper extraction to {host_disp} "
                            f"rejected due to circular dependency ({cycle_desc}).\n"
                        )
                        f1_plan.comments.append(cycle_msg)
                        f2_plan.comments.append(cycle_msg)
                        continue

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
                    c_mod = _derive_module_import_path(c_plan.path, import_root)
                    c_cycle = (
                        tentative_dg.check_cycle_if_added(c_mod, mod_host)
                        if (c_mod and mod_host)
                        else None
                    )
                    caller_evaluations.append(
                        (c_plan, c_unit, c_tin, c_tout, c_mod, c_cycle)
                    )

                if host_plan in (f1_plan, f2_plan) and caller_evaluations:
                    c_plan, _, _, _, c_mod, c_cycle = caller_evaluations[0]
                    if c_cycle or not mod_host or not c_mod:
                        c_disp = normalize_path_string(str(c_plan.rel_path), strip_anchor=False)
                        c_desc = f" (cycle: {' -> '.join(c_cycle)})" if c_cycle else ""
                        c_msg = (
                            f"# Note: Cross-module clone pair; helper generated in {host_disp}. "
                            f"Circular import or unresolvable module path{c_desc}; import manually into {c_disp}.\n"
                        )
                        f1_plan.comments.append(c_msg)
                        f2_plan.comments.append(c_msg)
                        continue

                if host_plan not in (f1_plan, f2_plan) and all(
                    bool(cycle or not mod_host or not m_call)
                    for _, _, _, _, m_call, cycle in caller_evaluations
                ):
                    c_descs = [
                        f"{m_call}: {' -> '.join(cycle)}"
                        for _, _, _, _, m_call, cycle in caller_evaluations
                        if cycle
                    ]
                    c_txt = f" ({'; '.join(c_descs)})" if c_descs else ""
                    skip_msg = (
                        f"# Note: Cross-module clone pair; helper extraction to {host_disp} "
                        f"rejected due to circular dependency{c_txt}.\n"
                    )
                    f1_plan.comments.append(skip_msg)
                    f2_plan.comments.append(skip_msg)
                    continue

                host_plan.module_helpers.append(helper_code)
                host_plan.missing_imports.extend(host_imports)
                host_plan.used_helper_names.add(helper_name)
                host_plan.comments.append(pair_comment)

                if mod_host:
                    _register_plan_dependencies_in_graph(
                        dg, mod_host, host_plan.path, host_imports
                    )

                if replace_clones:
                    if host_plan is f1_plan:
                        _delegate_unit_in_plan(
                            u1,
                            f1_plan,
                            helper_name=helper_name,
                            inputs=inputs,
                            outputs=outputs,
                            scope=scope,
                            target_inputs=t_inputs1,
                            target_outputs=outputs,
                            await_prefix=await_prefix,
                            binding=effective_binding,
                            step=step,
                        )
                    elif host_plan is f2_plan:
                        _delegate_unit_in_plan(
                            u2,
                            f2_plan,
                            helper_name=helper_name,
                            inputs=inputs,
                            outputs=outputs,
                            scope=scope,
                            target_inputs=t_inputs2,
                            target_outputs=target_outs2,
                            await_prefix=await_prefix,
                            binding="module",
                        )

                for c_plan, c_unit, c_tin, c_tout, mod_caller, cycle in caller_evaluations:
                    c_plan.comments.append(pair_comment)
                    c_disp = normalize_path_string(str(c_plan.rel_path), strip_anchor=False)
                    if cycle or not mod_host or not mod_caller:
                        cycle_desc = ""
                        if cycle:
                            cycle_desc = f"Circular import detected (cycle: {' -> '.join(cycle)})"
                        else:
                            cycle_desc = "Unresolvable module import path"
                        cycle_msg = (
                            f"# Note: Cross-module clone pair; helper extracted to {host_disp}. "
                            f"{cycle_desc}; import manually into {c_disp}.\n"
                        )
                        c_plan.comments.append(cycle_msg)
                        host_plan.comments.append(cycle_msg)
                    else:
                        c_plan.missing_imports.append(f"from {mod_host} import {helper_name}")
                        if mod_caller and mod_host:
                            dg.add_dependency(mod_caller, mod_host)
                        if replace_clones:
                            _delegate_unit_in_plan(
                                c_unit,
                                c_plan,
                                helper_name=helper_name,
                                inputs=inputs,
                                outputs=outputs,
                                scope=scope,
                                target_inputs=c_tin,
                                target_outputs=c_tout,
                                await_prefix=await_prefix,
                                binding="module",
                            )
                        else:
                            c_plan.comments.append(
                                f"# Note: Cross-module clone pair; helper extracted to {host_disp}. "
                                f"Complete refactoring by replacing the clone with a call in {c_disp}.\n"
                            )
            else:
                mod1 = _derive_module_import_path(f1_path, import_root)
                mod2 = _derive_module_import_path(f2_plan.path, import_root)
                is_pkg1 = f1_path.name == "__init__.py"
                is_pkg2 = f2_plan.path.name == "__init__.py"
                dg = _get_depgraph()

                helper_code, maybe_imports = _safely_prepare_helper_and_imports(
                    host_plan=f1_plan,
                    helper_code=helper_code,
                    scope=scope,
                    source_texts=[(orig_text, mod1, is_pkg1), (f2_plan.orig_text, mod2, is_pkg2)],
                    f1_plan=f1_plan,
                    f2_plan=f2_plan,
                    host_mod=mod1,
                    is_host_pkg=is_pkg1,
                )
                if maybe_imports is None:
                    continue
                host_imports = maybe_imports

                if mod1 and host_imports:
                    host_cycle = dg.check_cycle_if_imports_added(
                        mod1,
                        host_imports,
                        file_path=f1_plan.path,
                        is_package=is_pkg1,
                    )
                    if host_cycle:
                        cycle_desc = f" (cycle: {' -> '.join(host_cycle)})"
                        cycle_msg = (
                            f"# Note: Cross-module clone pair; helper extraction to {f1_disp} "
                            f"rejected due to circular dependency{cycle_desc}; import manually into {f2_disp}.\n"
                        )
                        for plan in (f1_plan, f2_plan):
                            plan.comments.append(cycle_msg)
                        continue

                tentative_dg = dg.copy()
                if mod1:
                    _register_plan_dependencies_in_graph(
                        tentative_dg, mod1, f1_plan.path, host_imports
                    )

                cycle = (
                    tentative_dg.check_cycle_if_added(mod2, mod1)
                    if (mod1 and mod2)
                    else None
                )
                direct_import = bool(
                    mod2
                    and _module_imports_target(
                        orig_text, mod2, current_mod=mod1, is_package=is_pkg1
                    )
                )
                is_circular = bool(cycle or direct_import)

                if is_circular or not mod1 or not mod2:
                    cycle_desc = ""
                    if cycle:
                        cycle_desc = f" (cycle: {' -> '.join(cycle)})"
                    elif direct_import:
                        cycle_desc = f" (cycle: {mod2} -> {mod1} -> {mod2})"
                    cycle_msg = (
                        f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                        f"Circular import or unresolvable module path{cycle_desc}; import manually into {f2_disp}.\n"
                    )
                    for plan in (f1_plan, f2_plan):
                        plan.comments.append(cycle_msg)
                    continue

                if mod1:
                    _register_plan_dependencies_in_graph(
                        dg, mod1, f1_plan.path, host_imports
                    )

                _wire_cross_module_host_delegation(
                    f1_plan=f1_plan,
                    f2_plan=f2_plan,
                    u2=u2,
                    mod1=mod1,
                    helper_name=helper_name,
                    pair_comment=pair_comment,
                    f1_disp=f1_disp,
                    f2_disp=f2_disp,
                    replace_clones=replace_clones,
                    inputs=inputs,
                    outputs=outputs,
                    scope=scope,
                    t_inputs2=t_inputs2,
                    target_outs2=target_outs2,
                    await_prefix=await_prefix,
                )
                if replace_clones and mod1 and mod2:
                    dg.add_dependency(mod2, mod1)

                _finalize_host_unit_and_helper(
                    plan=f1_plan,
                    unit=u1,
                    helper_name=helper_name,
                    inputs=inputs,
                    outputs=outputs,
                    scope=scope,
                    target_inputs=t_inputs1,
                    binding=effective_binding,
                    step=step,
                    insert_line=insert_line,
                    helper_code=helper_code,
                    missing_imports=host_imports,
                    await_prefix=await_prefix,
                    replace_clones=replace_clones,
                )
        else:
            mod1 = _derive_module_import_path(f1_path, import_root)
            is_pkg1 = f1_path.name == "__init__.py"
            helper_code, maybe_imports = _safely_prepare_helper_and_imports(
                host_plan=f1_plan,
                helper_code=helper_code,
                scope=scope,
                source_texts=[(orig_text, mod1, is_pkg1)],
                f1_plan=f1_plan,
                f2_plan=f2_plan,
                host_mod=mod1,
                is_host_pkg=is_pkg1,
            )
            if maybe_imports is None:
                continue
            host_imports = maybe_imports

            f1_plan.comments.append(
                f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                f"Complete refactoring by importing the helper into {f2_disp}.\n"
            )
            _finalize_host_unit_and_helper(
                plan=f1_plan,
                unit=u1,
                helper_name=helper_name,
                inputs=inputs,
                outputs=outputs,
                scope=scope,
                target_inputs=t_inputs1,
                binding=effective_binding,
                step=step,
                insert_line=insert_line,
                helper_code=helper_code,
                missing_imports=host_imports,
                await_prefix=await_prefix,
                replace_clones=replace_clones,
            )

    patch_chunks: List[str] = []
    for plan in file_plans.values():
        chunk = _render_file_patch_plan(
            plan, replace_clones=replace_clones, repo_root=str(patch_root)
        )
        if chunk:
            patch_chunks.append(chunk)

    return "\n".join(patch_chunks)
