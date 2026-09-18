"""Delegation call synthesis, unit overlap filtering, and multi-file patch generation."""

from __future__ import annotations

import ast
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
    build_module_graph,
    derive_module_import_path as _derive_module_import_path,
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
    except SyntaxError:
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





def _module_imports_target(source_text: str, target_module: str) -> bool:
    """Checks whether a module's source imports a specific target module."""
    if not target_module:
        return False
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return target_module in source_text
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == target_module or alias.name.startswith(f"{target_module}."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = getattr(node, "module", None) or ""
            if mod == target_module or mod.startswith(f"{target_module}."):
                return True
            for alias in node.names:
                full = f"{mod}.{alias.name}" if mod else alias.name
                if full == target_module or full.startswith(f"{target_module}."):
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
        formatted_imports = [imp.rstrip("\r\n") + "\n" for imp in deduped_imports]
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
        return ""

    deduped_comments = list(dict.fromkeys(plan.comments))
    return "".join(deduped_comments) + diff_str


def _extract_module_import_statements(source_text: str) -> Dict[str, str]:
    """Maps imported symbol/module names to clean import statements from source text."""
    stmts: Dict[str, str] = {}
    if not source_text.strip():
        return stmts
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError):
        return stmts

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                if name != "*":
                    if alias.asname:
                        stmts[name] = f"import {alias.name} as {alias.asname}"
                    else:
                        stmts[name] = f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            level = getattr(node, "level", 0) or 0
            if level == 0 and node.module:
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name != "*":
                        if alias.asname:
                            stmts[name] = (
                                f"from {node.module} import {alias.name} as {alias.asname}"
                            )
                        else:
                            stmts[name] = f"from {node.module} import {alias.name}"
    return stmts


def _collect_host_missing_imports(
    host_plan: _FilePatchPlan,
    helper_code: str,
    scope: Dict[str, Any],
    source_texts: Sequence[str] = (),
) -> List[str]:
    """Computes all required imports for a host plan receiving extracted helper_code.

    Considers typing symbols, scope local imports, and module-level source dependencies
    against the destination host plan's actual content and existing missing_imports.
    """
    host_content = (host_plan.orig_text or "") + "\n" + "\n".join(host_plan.missing_imports)
    host_imported = _get_module_imported_names(host_content)

    missing: List[str] = []

    # 1. Typing annotations
    needed_typing = _extract_required_typing_imports(helper_code)
    missing_typing = [s for s in needed_typing if s not in host_imported]
    if missing_typing:
        missing.append(f"from typing import {', '.join(sorted(missing_typing))}")

    # 2. Local imports discovered during scope analysis
    for loc_imp in scope.get("local_imports", []):
        s_loc = loc_imp.strip()
        if (
            s_loc
            and s_loc not in host_content
            and s_loc not in missing
            and s_loc not in host_plan.missing_imports
        ):
            missing.append(loc_imp)

    # 3. Source dependencies referenced in helper_code
    try:
        helper_tree = ast.parse(helper_code)
    except (SyntaxError, UnicodeDecodeError):
        helper_tree = None

    if helper_tree is not None and source_texts:
        defined_in_helper: Set[str] = set()
        loaded_in_helper: Set[str] = set()
        for node in ast.walk(helper_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_in_helper.add(node.name)
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    defined_in_helper.add(arg.arg)
                if node.args.vararg:
                    defined_in_helper.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined_in_helper.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defined_in_helper.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    loaded_in_helper.add(node.id)

        for src_text in source_texts:
            if not src_text:
                continue
            src_imports = _extract_module_import_statements(src_text)
            for loaded_name in sorted(loaded_in_helper):
                if (
                    loaded_name not in defined_in_helper
                    and loaded_name not in host_imported
                    and loaded_name not in needed_typing
                    and loaded_name in src_imports
                ):
                    stmt = src_imports[loaded_name]
                    if (
                        not stmt.startswith("from typing ")
                        and stmt not in missing
                        and stmt not in host_content
                        and stmt not in host_plan.missing_imports
                    ):
                        missing.append(stmt)

    return missing


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

    root = Path(repo_root or os.getcwd())
    graph_holder: List[Optional[ModuleDependencyGraph]] = [depgraph]

    def _get_depgraph() -> ModuleDependencyGraph:
        g = graph_holder[0]
        if g is None:
            g = build_module_graph(root)
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
        p = Path(f1_raw)
        f1_path = p if p.is_file() or p.is_absolute() else (root / p)
        if not f1_path.is_file():
            continue

        try:
            rel_f1 = str(f1_path.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            rel_f1 = str(f1_path.name)

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
            p2 = Path(f2_raw)
            f2_path = p2 if p2.is_file() or p2.is_absolute() else (root / p2)
            if f2_path.is_file():
                try:
                    rel_f2 = str(f2_path.resolve().relative_to(root.resolve())).replace("\\", "/")
                except ValueError:
                    rel_f2 = str(f2_path.name)
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
        target_host_plan: Optional[_FilePatchPlan] = None
        target_host_text: Optional[str] = None
        if (
            not is_same_file
            and f2_plan is not None
            and cross_file_strategy in ("auto", "shared_module", "shared")
        ):
            shared_p = resolve_shared_module_file(
                f1_path, f2_plan.path, root, shared_module_name=shared_module_name
            )
            if shared_p.resolve() == f1_path.resolve():
                target_host_plan = f1_plan
            elif shared_p.resolve() == f2_plan.path.resolve():
                target_host_plan = f2_plan
            else:
                target_host_plan = file_plans.get(shared_p)
                if target_host_plan is None and shared_p.is_file():
                    try:
                        target_host_text = shared_p.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
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
            host_imports = _collect_host_missing_imports(
                host_plan=f1_plan,
                helper_code=helper_code,
                scope=scope,
                source_texts=[orig_text],
            )
            _attach_helper_to_plan(
                f1_plan,
                binding=effective_binding,
                insert_line=insert_line,
                helper_code=helper_code,
                missing_imports=host_imports,
            )
        elif f2_plan is not None:
            if cross_file_strategy in ("auto", "shared_module", "shared"):
                shared_p = resolve_shared_module_file(
                    f1_path, f2_plan.path, root, shared_module_name=shared_module_name
                )
                try:
                    rel_shared = str(
                        shared_p.resolve().relative_to(root.resolve())
                    ).replace("\\", "/")
                except ValueError:
                    rel_shared = str(shared_p.name)

                if shared_p.resolve() == f1_path.resolve():
                    host_plan = f1_plan
                    callers = [(f2_plan, u2, t_inputs2, target_outs2)]
                elif shared_p.resolve() == f2_plan.path.resolve():
                    host_plan = f2_plan
                    callers = [(f1_plan, u1, t_inputs1, outputs)]
                else:
                    shared_plan = file_plans.get(shared_p)
                    if shared_plan is None:
                        if shared_p.is_file():
                            try:
                                shared_text = shared_p.read_text(encoding="utf-8")
                                shared_plan = _get_plan(
                                    shared_p, rel_shared, shared_text, is_new_file=False
                                )
                            except (OSError, UnicodeDecodeError):
                                shared_plan = _get_plan(
                                    shared_p, rel_shared, "", is_new_file=True
                                )
                        else:
                            shared_plan = _get_plan(
                                shared_p, rel_shared, "", is_new_file=True
                            )
                    host_plan = shared_plan
                    callers = [
                        (f1_plan, u1, t_inputs1, outputs),
                        (f2_plan, u2, t_inputs2, target_outs2),
                    ]

                host_plan.module_helpers.append(helper_code)
                host_imports = _collect_host_missing_imports(
                    host_plan=host_plan,
                    helper_code=helper_code,
                    scope=scope,
                    source_texts=[orig_text, f2_plan.orig_text],
                )
                host_plan.missing_imports.extend(host_imports)
                host_plan.used_helper_names.add(helper_name)
                host_plan.comments.append(pair_comment)

                mod_host = _derive_module_import_path(host_plan.path, root)
                host_disp = normalize_path_string(str(host_plan.rel_path), strip_anchor=False)

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

                for c_plan, c_unit, c_tin, c_tout in callers:
                    c_plan.comments.append(pair_comment)
                    mod_caller = _derive_module_import_path(c_plan.path, root)
                    dg = _get_depgraph()
                    cycle = (
                        dg.check_cycle_if_added(mod_caller, mod_host)
                        if (mod_caller and mod_host)
                        else None
                    )
                    c_disp = normalize_path_string(str(c_plan.rel_path), strip_anchor=False)
                    if cycle:
                        cycle_msg = (
                            f"# Note: Cross-module clone pair; helper extracted to {host_disp}. "
                            f"Circular import detected (cycle: {' -> '.join(cycle)}); import manually into {c_disp}.\n"
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
                                f"Complete refactoring by importing the helper into {c_disp}.\n"
                            )
            else:
                mod1 = _derive_module_import_path(f1_path, root)
                mod2 = _derive_module_import_path(f2_plan.path, root)
                dg = _get_depgraph()
                cycle = (
                    dg.check_cycle_if_added(mod2, mod1)
                    if (mod1 and mod2)
                    else None
                )
                direct_import = bool(mod2 and _module_imports_target(orig_text, mod2))
                is_circular = bool(cycle or direct_import)

                if is_circular or not mod1:
                    cycle_desc = ""
                    if cycle:
                        cycle_desc = f" (cycle: {' -> '.join(cycle)})"
                    elif direct_import:
                        cycle_desc = f" (cycle: {mod2} -> {mod1} -> {mod2})"
                    f1_plan.comments.append(
                        f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                        f"Circular import or unresolvable module path{cycle_desc}; import manually into {f2_disp}.\n"
                    )
                else:
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
                    if mod1 and mod2:
                        dg.add_dependency(mod2, mod1)

                host_imports = _collect_host_missing_imports(
                    host_plan=f1_plan,
                    helper_code=helper_code,
                    scope=scope,
                    source_texts=[orig_text, f2_plan.orig_text],
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
        else:
            f1_plan.comments.append(
                f"# Note: Cross-module clone pair; helper generated in {f1_disp}. "
                f"Complete refactoring by importing the helper into {f2_disp}.\n"
            )
            host_imports = _collect_host_missing_imports(
                host_plan=f1_plan,
                helper_code=helper_code,
                scope=scope,
                source_texts=[orig_text],
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
            plan, replace_clones=replace_clones, repo_root=str(root)
        )
        if chunk:
            patch_chunks.append(chunk)

    return "\n".join(patch_chunks)
