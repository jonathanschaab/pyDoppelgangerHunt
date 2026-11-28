"""Unit collision filtering, delegation call synthesis, and per-file patch plans."""

from __future__ import annotations

import ast
import difflib
import hashlib
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from pydoppelgangerhunt.config import normalize_path_string
from pydoppelgangerhunt._ast_compat import safe_parse_ast
from pydoppelgangerhunt.fixer._df_cache import _get_or_parse_tree
from pydoppelgangerhunt.fixer.binding import (
    _extract_child_indentation,
    _get_enclosing_receiver_kind,
    _is_method_of_class,
    _is_same_file_path,
    find_enclosing_class,
    find_enclosing_function,
)
from pydoppelgangerhunt.fixer.depgraph import _parse_source_imports
from pydoppelgangerhunt.fixer.source import (
    ReplacementItem,
    UnitDict,
    _detect_indent_step,
    _find_module_helper_insertion_index,
    _find_sig_colon,
    _insert_imports_into_module,
    _is_docstring_node,
    _scan_sig_line,
    _validate_and_parse_unit_bound,
    col_offset_to_char_offset,
    compute_line_offsets,
    count_physical_newlines,
    detect_line_ending,
    extract_unit_comments_and_pragmas,
    is_valid_unit_coordinates,
    lines_end_with_unconditional_return,
    parse_unit_coord,
    resolve_unit_replacement,
    split_source_lines,
)
from pydoppelgangerhunt.fixer.synthesis import (
    _extract_unreferenced_lhs_loads,
    _format_call_arguments,
    _is_subline_unit,
    _node_references_name,
    _order_inputs_by_rhs_loads,
    _parse_dedented_stmts,
)
from pydoppelgangerhunt.source_lines import (
    normalize_unit_coordinates,
    resolve_unit_line_bounds,
    unit_file_str,
)
from pydoppelgangerhunt.unit_boundary import END_COL_KEYS, START_COL_KEYS

logger = logging.getLogger(__name__)


class UnitCollisionError(ValueError):
    """Raised when two or more refactoring replacement units collide or overlap.

    Subclasses :class:`ValueError` so callers expecting standard Python coordinate
    or mapping exceptions can catch it transparently, while allowing specialized
    handlers to differentiate physical and semantic AST coordinate collisions.
    """


class _DelegationCallError(ValueError):
    """Raised when a delegation call cannot be constructed for a candidate clone unit."""


def _check_same_line_overlap(
    sc1: Optional[int],
    ec1: Optional[int],
    sc2: Optional[int],
    ec2: Optional[int],
) -> bool:
    """Case 1: Half-open column interval intersection on the same single line."""
    if sc1 is not None and ec1 is not None and sc2 is not None and ec2 is not None:
        if sc1 <= ec1 and sc2 <= ec2:
            return max(sc1, sc2) < min(ec1, ec2)
        return False
    return True


def _check_sequential_touch(
    prior_ec: Optional[int],
    next_sc: Optional[int],
    next_is_single: bool,
    next_ec: Optional[int],
) -> bool:
    """Cases 2 & 3: Sequential touch where one unit ends on the line the other begins."""
    if prior_ec is not None and next_sc is not None:
        if next_is_single and next_ec is not None and next_sc >= next_ec:
            return False
        return next_sc < prior_ec
    return True


def _check_boundary_sharing(
    multi_sc: Optional[int],
    single_sc: Optional[int],
    single_ec: Optional[int],
) -> bool:
    """Case 4: Multi-line and single-line units sharing a start boundary line."""
    if multi_sc is None or single_sc is None or single_ec is None:
        return True
    if single_sc >= single_ec:
        return False
    return multi_sc < single_ec


def check_units_overlap(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> bool:
    """Determines whether two AST code units in the same file share overlapping line ranges.

    Acts as Tier 1 of the dual-tier overlap defense. When column offsets ('start_col',
    'end_col') are omitted from either unit, this check conservatively assumes the unit
    spans whole lines and treats any shared line as an overlap conflict. When units share
    a single boundary line (same-line, touching boundary, or a boundary line of a multi-line
    unit), it checks sub-line column intervals for half-open intersection. Single-line units
    or boundary intervals with inverted or zero-width column bounds (start_col >= end_col) represent
    empty ranges and evaluate to False. Any shared interior lines between multi-line units unconditionally conflict.

    Args:
        u1: First AST unit dictionary with 'file', 'start', and 'end'.
        u2: Second AST unit dictionary with 'file', 'start', and 'end'.
        repo_root: Optional repository root path for resolving relative file paths.

    Returns:
        True if both units reside in the same normalized file path and their [start, end]
        intervals overlap; False otherwise.
    """
    if not isinstance(u1, dict) or not isinstance(u2, dict):
        raise TypeError(
            f"Units must be dictionaries, got {type(u1).__name__} and {type(u2).__name__}"
        )
    f1 = normalize_path_string(unit_file_str(u1))
    f2 = normalize_path_string(unit_file_str(u2))
    if not f1 or not f2 or not _is_same_file_path(f1, f2, repo_root=repo_root):
        return False

    def _parse_lines(u: Dict[str, Any], label: str, file_path: str) -> Tuple[int, int]:
        try:
            s = max(1, parse_unit_coord(u, "start", default=1))
            e = max(1, parse_unit_coord(u, "end", default=s))
            return s, e
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Malformed unit: invalid line boundary in {label} ({file_path}): {err}"
            ) from err

    start1, end1 = _parse_lines(u1, "u1", f1)
    start2, end2 = _parse_lines(u2, "u2", f2)

    # Inverted line ranges (start > end) represent empty 0-width ranges, which cannot overlap.
    if start1 > end1 or start2 > end2 or end1 < start2 or end2 < start1:
        return False

    def _parse_col(val: Any, col_name: str, file_path: str) -> Optional[int]:
        if val is None:
            return None
        try:
            parsed = parse_unit_coord({col_name: val}, col_name, default=None)
            if parsed is None:
                return None
            return max(0, parsed)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Malformed unit: invalid column offset '{col_name}'={val!r} in {file_path}"
            ) from err

    def _first_non_none(u: Dict[str, Any], *keys: str) -> Any:
        for k in keys:
            if u.get(k) is not None:
                return u.get(k)
        return None

    sc1 = _parse_col(_first_non_none(u1, *START_COL_KEYS), "start_col", f1)
    ec1 = _parse_col(_first_non_none(u1, *END_COL_KEYS), "end_col", f1)
    sc2 = _parse_col(_first_non_none(u2, *START_COL_KEYS), "start_col", f2)
    ec2 = _parse_col(_first_non_none(u2, *END_COL_KEYS), "end_col", f2)

    # -------------------------------------------------------------------------
    # Case -> Expected Behaviour Mapping:
    # -------------------------------------------------------------------------
    # Case 1: Same line (start1 == end1 == start2 == end2)
    #   -> Overlap if half-open column intervals intersect: max(sc1, sc2) < min(ec1, ec2).
    # Case 2: Sequential touch (start1 < start2 and end1 == start2)
    #   -> u1 ends on the line u2 begins: overlap if sc2 < ec1.
    # Case 3: Sequential touch (start2 < start1 and end2 == start1)
    #   -> u2 ends on the line u1 begins: overlap if sc1 < ec2.
    # Case 4: Start boundary line sharing between a multi-line unit and a single-line unit:
    #   -> Shared start line: overlap if single unit ends after multi begins (multi_sc < single_ec).
    #   -> Multi-line vs. multi-line sharing a boundary line: conservatively overlap (True).
    # Case 5: Multi-line interior overlap (units span multiple shared lines)
    #   -> Units overlap across interior lines: unconditionally conflict (True).
    # -------------------------------------------------------------------------

    # Case 1: Both units reside on the exact same single line
    if start1 == end1 == start2 == end2:
        return _check_same_line_overlap(sc1, ec1, sc2, ec2)

    # Case 2: Sequential boundary touch (u1 ends where u2 begins)
    if start1 < start2 and start2 == end1:
        return _check_sequential_touch(ec1, sc2, start2 == end2, ec2)

    # Case 3: Sequential boundary touch (u2 ends where u1 begins)
    if start2 < start1 and start1 == end2:
        return _check_sequential_touch(ec2, sc1, start1 == end1, ec1)

    # Case 4: Units sharing start line (start1 == start2), or multi-line units sharing end line (end1 == end2).
    # Note: Sequential touches and single-line units sharing the end-line of a multi-line unit
    # (e.g. u1 lines 1..3, u2 line 3..3) have end1 == start2 and are fully handled by Case 2/3 above.
    if start1 == start2:
        if start1 < end1 and start2 == end2:
            return _check_boundary_sharing(sc1, sc2, ec2)
        if start2 < end2 and start1 == end1:
            return _check_boundary_sharing(sc2, sc1, ec1)
        return True

    if end1 == end2:
        # Multi-line vs. multi-line sharing an end line (single-line sharing end line is handled in Case 2/3)
        return True

    # Case 5: Multi-line interior overlap
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
        f_raw = unit_file_str(u)
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
        def _sort_key(u: Dict[str, Any]) -> Tuple[int, int, int, str]:
            s = parse_unit_coord(u, "start", default=0)
            e = parse_unit_coord(u, "end", default=s)
            sc = parse_unit_coord(u, "start_col", default=0)
            return (-(e - s), s, sc, str(u.get("name") or ""))

        sorted_candidates = sorted(file_units, key=_sort_key)
        file_retained: List[Dict[str, Any]] = []
        for cand in sorted_candidates:
            if not any(check_units_overlap(cand, prev, repo_root=repo_root) for prev in file_retained):
                file_retained.append(cand)
        retained.extend(file_retained)

    return retained


def intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Checks whether two half-open byte intervals [s1, e1) and [s2, e2) overlap, including zero-width boundaries."""
    if s1 == e1 and s2 == e2:
        return s1 == s2
    if s1 == e1:
        return s2 < s1 < e2
    if s2 == e2:
        return s1 < s2 < e1
    return max(s1, s2) < min(e1, e2)


_intervals_overlap = intervals_overlap


def _format_unit_desc(u: UnitDict) -> Tuple[str, int, int, str]:
    """Extracts (name, start_line, end_line, file_path) from an AST unit dictionary."""
    if not isinstance(u, dict):
        raise TypeError(f"Unit must be a dictionary, got {type(u).__name__}")
    n = str(u.get("name") or "unit")
    s, e = resolve_unit_line_bounds(u)
    f = normalize_path_string(unit_file_str(u), strip_anchor=False) or "<module>"
    return n, s, e, f


def _raise_unit_collision_error(
    u1: UnitDict,
    u2: UnitDict,
    context: str = "",
) -> None:
    """Formats and raises a UnitCollisionError between two conflicting units."""
    n1, s1, e1, f1 = _format_unit_desc(u1)
    n2, s2, e2, _ = _format_unit_desc(u2)
    ctx_prefix = f" in {context} " if context else " "
    raise UnitCollisionError(
        f"Overlapping unit collision detected{ctx_prefix}"
        f"between '{n1}' ({s1}-{e1}) and '{n2}' ({s2}-{e2}) in {f1}."
    )


def _assert_no_interval_collisions(
    items: Sequence[ReplacementItem],
    context: str = "",
) -> None:
    """Verifies that a sorted sequence of ReplacementItems does not contain overlapping byte intervals."""
    max_end_item: Optional[ReplacementItem] = None
    for item in items:
        if (
            max_end_item is not None
            and _intervals_overlap(
                max_end_item.start_byte, max_end_item.end_byte, item.start_byte, item.end_byte
            )
        ):
            _raise_unit_collision_error(max_end_item.unit, item.unit, context=context)
        if max_end_item is None or item.end_byte >= max_end_item.end_byte:
            max_end_item = item


def refactor_module_units(
    source_text: str,
    replacements: Sequence[Tuple[UnitDict, str]],
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    tier1: bool = True,
    tier2: bool = True,
    strict: bool = True,
) -> str:
    """Applies multiple non-overlapping unit replacements in reverse source order.

    Architectural Overview:
        Refactoring uses a dual-tier collision defense and reverse-order buffer application:
        1. Tier 1 (Semantic AST Coordinate Check): Fast-path check using :func:`check_units_overlap`
           to verify that units do not share conflicting lines or column ranges before resolving text.
        2. Pragma & Boundary Span Resolution: :func:`resolve_unit_replacement` maps each unit to
           physical character and UTF-8 byte spans while preserving attached boundary pragmas
           (e.g., `# type: ignore`, `# noqa`, `# pragma: no cover`) and leading decorators.
        3. Tier 2 (Physical Byte-Span Sweep): Verifies via :func:`intervals_overlap` that the physical
           byte intervals `[start_byte, end_byte)` of expanded replacement slices do not collide. This
           catches collisions introduced by pragma expansions or indentation adjustments.
        4. Reverse-Order Execution: Replacements are sorted descending by `(start_byte, end_byte)`.
           Applying edits bottom-to-top ensures downstream text length changes never shift upstream
           character or byte coordinates.

    Performance & Algorithmic Bounds:
        Tier-1 and Tier-2 validation combined with single-pass descending buffer slicing executes
        in sub-millisecond to low-millisecond time (<5ms for hundreds of replacements on 10,000+ line
        modules), avoiding quadratic string re-tokenizations and redundant buffer dry-runs.

    Args:
        source_text: The complete original Python source code.
        replacements: Sequence of (unit, replacement_text) tuples.
        lines: Optional precomputed line strings of source_text.
        line_char_offsets: Optional precomputed character offsets of line starts.
        line_byte_offsets: Optional precomputed UTF-8 byte offsets of line starts.
        dry_run: If True, executes collision validation without slicing or modifying the source buffer.
        tier1: If True, executes Tier 1 semantic line/column overlap validation.
        tier2: If True, executes Tier 2 physical UTF-8 byte interval sweep validation.
        strict: If True, validates unit coordinates with strict positive integer rules.

    Returns:
        The refactored module source text (or unmodified source_text if dry_run=True).

    Raises:
        UnitCollisionError: If any pair of units collides under active validation tiers.
    """
    if not replacements:
        return source_text

    # Tier 1 Semantic Overlap Check:
    # Conservatively reject candidate pairs that share lines when column bounds are
    # omitted or incomplete (whole-line / statement replacements inherently conflict
    # with any other edit on the same line), or when column intervals overlap.
    rep_list: List[Tuple[Dict[str, Any], str]] = []
    for u_raw, rep in replacements:
        if not isinstance(u_raw, dict):
            raise TypeError(f"Unit must be a dictionary, got {type(u_raw).__name__}")
        u = normalize_unit_coordinates(u_raw)
        _validate_and_parse_unit_bound(u, "start", strict=strict)
        _validate_and_parse_unit_bound(u, "end", strict=strict)
        if not is_valid_unit_coordinates(u, strict=strict):
            raise ValueError(
                f"Malformed unit: invalid coordinates for refactoring: {u!r}"
            )
        if not unit_file_str(u):
            u_norm = dict(u)
            u_norm["file"] = "<module>"
            rep_list.append((u_norm, rep))
        else:
            rep_list.append((u, rep))

    if tier1:
        for i, (u1, _) in enumerate(rep_list):
            for u2, _ in rep_list[i + 1:]:
                if check_units_overlap(u1, u2):
                    _raise_unit_collision_error(u1, u2)

    computed_entries: List[ReplacementItem] = []
    if lines is None:
        lines = split_source_lines(source_text)
    if line_char_offsets is None or line_byte_offsets is None:
        line_char_offsets, line_byte_offsets = compute_line_offsets(lines)

    for i, (unit, rep) in enumerate(rep_list):
        item = resolve_unit_replacement(
            source_text,
            unit,
            rep,
            lines=lines,
            line_char_offsets=line_char_offsets,
            line_byte_offsets=line_byte_offsets,
            order_index=i,
        )
        computed_entries.append(item)

    # Tier 2 Physical Byte-Span Overlap Check:
    # After computing exact UTF-8 byte slices in the underlying source buffer,
    # verify via an O(N log N) sweep that no two physical byte intervals [s_b, e_b) collide.
    sorted_by_start = sorted(
        computed_entries,
        key=lambda item: (item.start_byte, item.end_byte, item.order_index),
    )
    if tier2:
        _assert_no_interval_collisions(sorted_by_start)

    if dry_run:
        return source_text

    # Note: reverse order ensures downstream replacements do not alter upstream character offsets
    sorted_replacements = list(reversed(sorted_by_start))

    current_text = source_text
    for item in sorted_replacements:
        s_c = item.start_char
        e_c = item.end_char
        current_text = current_text[:s_c] + item.final_rep + current_text[e_c:]

    return current_text


def validate_module_unit_replacements(
    source_text: str,
    replacements: Sequence[Tuple[UnitDict, str]],
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
    tier1: bool = True,
    tier2: bool = True,
    strict: bool = True,
) -> None:
    """Validates that candidate unit replacements do not contain interval collisions.

    Runs Tier 1 (AST line/column range overlap) and/or Tier 2 (physical UTF-8 byte span
    sweep) validation without modifying or allocating new source string buffers.

    Args:
        source_text: The complete original Python source code.
        replacements: Sequence of (unit, replacement_text) tuples.
        lines: Optional precomputed line strings of source_text.
        line_char_offsets: Optional precomputed character offsets of line starts.
        line_byte_offsets: Optional precomputed UTF-8 byte offsets of line starts.
        tier1: If True, executes Tier 1 semantic line/column overlap validation.
        tier2: If True, executes Tier 2 physical UTF-8 byte interval sweep validation.
        strict: If True, enforces strict coordinate typing and positive bounds.

    Raises:
        UnitCollisionError: If any pair of units collides under active validation tiers.
    """
    refactor_module_units(
        source_text,
        replacements,
        lines=lines,
        line_char_offsets=line_char_offsets,
        line_byte_offsets=line_byte_offsets,
        dry_run=True,
        tier1=tier1,
        tier2=tier2,
        strict=strict,
    )


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
    lines = split_source_lines(source_text)
    u_start, u_end = resolve_unit_line_bounds(unit)

    lead = lines[u_start - 1] if 1 <= u_start <= len(lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]
    default_step = _detect_indent_step(indent)
    body_indent = indent + default_step

    sig_end_line: Optional[int] = None
    docstring_end_line: Optional[int] = None
    header = ""
    raw_u_name = str(unit.get("name") or "")
    base_u_name = raw_u_name.rsplit(":", maxsplit=1)[-1] if raw_u_name else ""

    tree = safe_parse_ast(source_text)
    if tree is not None:
        cand_nodes: List[Tuple[int, Union[ast.FunctionDef, ast.AsyncFunctionDef]]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_start = getattr(node, "lineno", 0) or 0
                n_end = getattr(node, "end_lineno", None) or n_start
                decs = getattr(node, "decorator_list", [])
                dec_start = (
                    min((getattr(d, "lineno", None) or n_start) for d in decs)
                    if decs
                    else n_start
                )
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
                    def_line_text = lines[fn_def_line - 1]
                    b_col_raw = getattr(first_body, "col_offset", None)
                    b_char = (
                        col_offset_to_char_offset(def_line_text, b_col_raw)
                        if b_col_raw is not None
                        else len(def_line_text)
                    )
                    same_line = def_line_text[:b_char].rstrip()
                    if not same_line.endswith(":"):
                        colon_pos = _find_sig_colon(def_line_text[:b_char])
                        if colon_pos != -1:
                            same_line = def_line_text[: colon_pos + 1]
                    header = "".join(lines[u_start - 1 : fn_def_line - 1]) + same_line + "\n"
                else:
                    sig_end_line = first_body.lineno - 1
                    if _is_docstring_node(first_body):
                        docstring_end_line = (
                            getattr(first_body, "end_lineno", None)
                            or getattr(first_body, "lineno", 0)
                        )

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
    u_start = max(1, parse_unit_coord(unit, "start", default=1))
    u_end = min(len(orig_lines), parse_unit_coord(unit, "end", default=u_start))
    if u_start > len(orig_lines) or u_start > u_end:
        return False
    return lines_end_with_unconditional_return(orig_lines[u_start - 1 : u_end])


_AUGASSIGN_OPS: Dict[type, str] = {
    ast.Add: "+=",
    ast.Sub: "-=",
    ast.Mult: "*=",
    ast.MatMult: "@=",
    ast.Div: "/=",
    ast.Mod: "%=",
    ast.Pow: "**=",
    ast.LShift: "<<=",
    ast.RShift: ">>=",
    ast.BitOr: "|=",
    ast.BitXor: "^=",
    ast.BitAnd: "&=",
    ast.FloorDiv: "//=",
}


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
    preserve_boundary_pragmas: bool = True,
) -> str:
    """Constructs replacement delegation call statement for a clone unit in refactoring patches."""
    u_start = max(1, parse_unit_coord(target_unit, "start", default=1))
    u_end_coord = max(u_start, parse_unit_coord(target_unit, "end", default=u_start))
    lead = orig_lines[u_start - 1] if 1 <= u_start <= len(orig_lines) else ""
    if not lead.strip() or lead.lstrip().startswith("#"):
        for cand_ln in orig_lines[u_start - 1 : min(len(orig_lines), u_end_coord)]:
            cand_s = cand_ln.lstrip()
            if cand_s and not cand_s.startswith("#"):
                lead = cand_ln
                break
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

    is_expr_unit = target_unit.get("kind") in ("comprehension", "complex_expr")
    unit_args_str = _format_call_arguments(
        inputs,
        scope.get("param_details", []),
        receiver_to_omit=unit_receiver_omit,
        target_inputs=effective_targets,
        positional_target_map=is_expr_unit,
    )

    is_whole_method = target_unit.get("kind") in (
        "function",
        "closure",
        "method",
        "async_function",
        "async_method",
    )
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

    if is_expr_unit:
        call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
        if not _is_subline_unit(target_unit):
            u_end = max(u_start, parse_unit_coord(target_unit, "end", default=u_start))
            raw_block = "".join(orig_lines[u_start - 1 : u_end])
            assign_prefix: Optional[str] = None
            try:
                parsed_body = _parse_dedented_stmts(raw_block)
                if len(parsed_body) != 1:
                    return ""
                parsed_node = parsed_body[0]
                unref_lhs: Set[str] = set()
                if isinstance(parsed_node, ast.Assign) and parsed_node.targets:
                    targets_str = " = ".join(ast.unparse(t) for t in parsed_node.targets)
                    assign_prefix = f"{targets_str} = "
                    unref_lhs = _extract_unreferenced_lhs_loads(parsed_node)
                elif isinstance(parsed_node, ast.AnnAssign) and parsed_node.value is not None:
                    target_str = ast.unparse(parsed_node.target)
                    anno_str = ast.unparse(parsed_node.annotation)
                    assign_prefix = f"{target_str}: {anno_str} = "
                    unref_lhs = _extract_unreferenced_lhs_loads(parsed_node)
                elif isinstance(parsed_node, ast.AugAssign):
                    target_str = ast.unparse(parsed_node.target)
                    op_symbol = _AUGASSIGN_OPS.get(type(parsed_node.op), "+=")
                    assign_prefix = f"{target_str} {op_symbol} "
                    unref_lhs = _extract_unreferenced_lhs_loads(parsed_node)
                    if isinstance(parsed_node.target, ast.Name):
                        t_name = parsed_node.target.id
                        if not _node_references_name(parsed_node.value, t_name):
                            unref_lhs.add(t_name)
                elif isinstance(parsed_node, ast.Return) and parsed_node.value is not None:
                    assign_prefix = "return "
                elif isinstance(parsed_node, ast.Expr):
                    assign_prefix = ""
                rhs_val = getattr(parsed_node, "value", None)
                if isinstance(
                    parsed_node,
                    (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr),
                ) and isinstance(rhs_val, ast.AST):
                    unit_receiver = rec_param or target_unit.get("receiver_param")
                    rec_names: Tuple[str, ...] = (
                        (str(unit_receiver), "self", "cls", "this", "klass")
                        if unit_receiver
                        else ("self", "cls", "this", "klass")
                    )
                    pruned_eff = (
                        [
                            x
                            for x in effective_targets
                            if x not in unref_lhs
                            or (
                                x == unit_receiver_omit
                                and (x in inputs or any(r in inputs for r in rec_names))
                            )
                        ]
                        if unref_lhs and len(effective_targets) > len(inputs)
                        else list(effective_targets)
                    )
                    if not any(
                        x in unref_lhs and x != unit_receiver_omit for x in pruned_eff
                    ):
                        pruned_eff = _order_inputs_by_rhs_loads(
                            pruned_eff, rhs_val, receiver_names=rec_names
                        )
                    if len(pruned_eff) == len(inputs):
                        pruned_args = _format_call_arguments(
                            inputs,
                            scope.get("param_details", []),
                            receiver_to_omit=unit_receiver_omit,
                            target_inputs=pruned_eff,
                            positional_target_map=True,
                        )
                        call_expr = f"{unit_call_prefix}{helper_name}({pruned_args})"
            except (SyntaxError, IndexError, ValueError):
                pass

            if assign_prefix is None:
                return ""
            rep_call = f"{indent}{assign_prefix}{await_prefix}{call_expr}\n"
            if not preserve_boundary_pragmas:
                return rep_call
            attached_pragmas = [
                c["text"].strip()
                for c in extract_unit_comments_and_pragmas(
                    orig_text,
                    start_line=u_start,
                    end_line=u_end,
                    include_leading=False,
                )
                if c["is_pragma"]
            ]
            dedup_pragmas = list(
                dict.fromkeys(p for p in attached_pragmas if p not in rep_call)
            )
            if dedup_pragmas:
                return rep_call.rstrip("\r\n") + "  " + "  ".join(dedup_pragmas) + "\n"
            return rep_call
        return f"{await_prefix}{call_expr}"

    call_expr = f"{unit_call_prefix}{helper_name}({unit_args_str})"
    rep_step = step or ("\t" if "\t" in indent else "    ")
    if bool(scope.get("has_yield")) and bool(scope.get("is_async")):
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


def _outputs_compatible(
    helper_outs: Sequence[str],
    target_outs: Sequence[str],
    *,
    is_subroutine: bool,
    has_yield: bool,
) -> bool:
    """Checks whether candidate clone outputs match synthesized helper outputs.

    Note: When is_subroutine and has_yield are True, resolve_clone_generator_subroutine_outputs
    strictly guarantees equal-length paired outputs, so output lengths are always compatible.
    """
    if is_subroutine and has_yield:
        return True
    return len(helper_outs) == len(target_outs)


class _PlanSnapshot(TypedDict):
    """Encapsulates a snapshot of mutable _FilePatchPlan state for transactional rollback."""

    replacements: List[Tuple[UnitDict, str]]
    replacement_items: List[ReplacementItem]
    method_helpers: List[Tuple[int, str]]
    module_helpers: List[str]
    missing_imports: List[str]
    comments: List[str]
    used_helper_names: Set[str]
    claimed_units: List[UnitDict]


class _FilePatchPlan:
    """Internal accumulator of proposed transformations for a single file."""

    def __init__(
        self,
        file_path: Path,
        rel_path: str,
        orig_text: str,
        is_new_file: bool = False,
    ) -> None:
        self.path = file_path
        self.rel_path = rel_path
        self.orig_text = orig_text
        self.orig_lines = [] if is_new_file else split_source_lines(orig_text)
        self.is_new_file = is_new_file
        if is_new_file:
            self.line_char_offsets: Sequence[int] = [0]
            self.line_byte_offsets: Sequence[int] = [0]
        else:
            self.line_char_offsets, self.line_byte_offsets = compute_line_offsets(self.orig_lines)
        self.replacements: List[Tuple[UnitDict, str]] = []
        self.replacement_items: List[ReplacementItem] = []
        self.method_helpers: List[Tuple[int, str]] = []
        self.module_helpers: List[str] = []
        self.missing_imports: List[str] = []
        self.comments: List[str] = []
        self.used_helper_names: Set[str] = set()
        self.claimed_units: List[UnitDict] = []
        self._cached_ast: Optional[ast.AST] = None
        self._ast_attempted: bool = False
        self._content_digest: Optional[str] = None

    @property
    def content_digest(self) -> str:
        """SHA-256 digest of orig_text, cached for reuse in downstream analysis."""
        if self._content_digest is None:
            self._content_digest = hashlib.sha256(
                self.orig_text.encode("utf-8", errors="replace")
            ).hexdigest()
        return self._content_digest

    @property
    def parsed_tree(self) -> Optional[ast.AST]:
        """Lazily parsed AST of orig_text, cached for reuse across operations."""
        if not self._ast_attempted:
            self._ast_attempted = True
            self._cached_ast = _get_or_parse_tree(
                self.orig_text, source_digest=self.content_digest
            )
        return self._cached_ast

    def snapshot(self) -> _PlanSnapshot:
        """Creates a snapshot of mutable plan state for transactional rollback."""
        return {
            "replacements": list(self.replacements),
            "replacement_items": list(self.replacement_items),
            "method_helpers": list(self.method_helpers),
            "module_helpers": list(self.module_helpers),
            "missing_imports": list(self.missing_imports),
            "comments": list(self.comments),
            "used_helper_names": set(self.used_helper_names),
            "claimed_units": list(self.claimed_units),
        }

    def restore(self, snap: _PlanSnapshot) -> None:
        """Restores mutable plan state from a snapshot."""
        self.replacements = list(snap["replacements"])
        self.replacement_items = list(snap.get("replacement_items", []))
        self.method_helpers = list(snap["method_helpers"])
        self.module_helpers = list(snap["module_helpers"])
        self.missing_imports = list(snap["missing_imports"])
        self.comments = list(snap["comments"])
        self.used_helper_names = set(snap["used_helper_names"])
        self.claimed_units = list(snap["claimed_units"])


def _compute_replacement_line_deltas(
    reps: Sequence[Tuple[Dict[str, Any], str]],
    orig_text: str,
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """Computes (end_line, line_delta) tuples for replacements in source text.

    For each replacement, calculates net change in line count (newlines added - newlines removed).
    Because candidate replacements are non-overlapping, their line deltas are strictly additive.
    Each entry is an (end_line, delta) tuple where end_line serves as the boundary cutoff
    for downstream target line adjustments in _adjust_line_for_replacements (end_line < target_line).

    Raises:
        TypeError: If any replacement unit is not a dictionary.
        ValueError: If any unit contains invalid line coordinates or if candidate replacements overlap.
    """
    if lines is None:
        lines = split_source_lines(orig_text)
    if line_char_offsets is None or line_byte_offsets is None:
        line_char_offsets, line_byte_offsets = compute_line_offsets(lines)
    computed: List[Tuple[ReplacementItem, int, int]] = []
    for i, (u, rep) in enumerate(reps):
        if not isinstance(u, dict):
            raise TypeError(f"Unit must be a dictionary, got {type(u).__name__}")
        if not is_valid_unit_coordinates(u, strict=False):
            raise ValueError(
                f"Malformed unit: invalid line boundary in {unit_file_str(u)}: {u}"
            )
        try:
            fallback = max(1, parse_unit_coord(u, "start", default=1))
            end_l = max(fallback, parse_unit_coord(u, "end", default=fallback))
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Malformed unit: invalid line boundary in {unit_file_str(u)}: {err}"
            ) from err
        item = resolve_unit_replacement(
            orig_text,
            u,
            rep,
            lines=lines,
            line_char_offsets=line_char_offsets,
            line_byte_offsets=line_byte_offsets,
            order_index=i,
        )
        orig_nl = count_physical_newlines(orig_text[item.start_char : item.end_char])
        rep_nl = count_physical_newlines(item.final_rep)
        computed.append((item, end_l, rep_nl - orig_nl))

    sorted_items = sorted(
        computed,
        key=lambda x: (x[0].start_byte, x[0].end_byte, x[0].order_index),
    )
    _assert_no_interval_collisions(
        [item for item, _, _ in sorted_items],
        context="line delta computation",
    )

    deltas = [(end_l, d) for _, end_l, d in computed]
    deltas.sort(key=lambda item: item[0])
    return deltas


def _apply_line_deltas(target_line: int, deltas: Sequence[Tuple[int, int]]) -> int:
    """Adjusts a target line number given precomputed (end_line, delta) tuples."""
    try:
        t_line = int(target_line)
    except (ValueError, TypeError):
        t_line = 1
    if not deltas or t_line <= 1:
        return max(1, t_line)
    line_delta = sum(d for end_line, d in deltas if end_line < t_line)
    return max(1, t_line + line_delta)


def _adjust_line_for_replacements(
    target_line: int,
    reps: Sequence[Tuple[Dict[str, Any], str]],
    orig_text: str,
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
) -> int:
    """Adjusts a target source line number to account for upstream line expansions or contractions.

    Evaluates cumulative line shift from replacements that conclude strictly prior to
    target_line (end_line < target_line). If a unit's end_line coincides exactly with
    target_line (end_line == target_line), its line delta is NOT included, which preserves
    insertion coordinates immediately preceding target_line (such as prepending helper
    definitions before the first candidate function or class).

    Args:
        target_line: 1-indexed target line number to adjust.
        reps: Sequence of (unit, replacement_text) tuples.
        orig_text: Original unmodified source code.
        lines: Optional pre-split lines of orig_text.
        line_char_offsets: Optional precomputed line start character offsets.
        line_byte_offsets: Optional precomputed line start UTF-8 byte offsets.

    Returns:
        Adjusted 1-indexed line number in the refactored source text.
    """
    try:
        t_line = int(target_line)
    except (ValueError, TypeError):
        t_line = 1
    if t_line <= 1 or not reps:
        return max(1, t_line)
    if lines is None:
        lines = split_source_lines(orig_text)
    deltas = _compute_replacement_line_deltas(
        reps,
        orig_text,
        lines=lines,
        line_char_offsets=line_char_offsets,
        line_byte_offsets=line_byte_offsets,
    )
    return _apply_line_deltas(t_line, deltas)


def _derive_unit_indent_step(
    unit: Dict[str, Any],
    lines: Sequence[str],
    step: Optional[str] = None,
) -> str:
    """Derives indentation step for a unit from its starting line indentation if not provided."""
    if step is not None:
        return step
    u_s = max(1, parse_unit_coord(unit, "start", default=1))
    u_lead = lines[u_s - 1] if 1 <= u_s <= len(lines) else ""
    u_ind = u_lead[: len(u_lead) - len(u_lead.lstrip())]
    return _detect_indent_step(u_ind)


def _delegate_unit_in_plan(
    unit: UnitDict,
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
    """Builds and records a delegation call replacement for a unit inside a file plan.

    Incremental Dual-Tier Validation Architecture & Transactional Safety:
        1. Tier 1 (Semantic Check, O(K)): Fails fast if candidate `unit` overlaps
           with any existing planned replacement's line/column coordinates.
        2. Pragma & Span Resolution: `new_item` is resolved against the unmodified
           source buffer with boundary pragma and indentation preservation.
        3. Tier 2 (Physical Check, O(K)): `intervals_overlap` verifies that `new_item`'s
           UTF-8 byte interval does not intersect any existing replacement item.

    Performance & Rollback:
        Incremental O(K) validation avoids full O(K log K) buffer dry-runs,
        scaling gracefully to hundreds of replacements per module. Plans are snapshotted
        before each candidate pair; any UnitCollisionError rolls back without corruption.
    """
    step = _derive_unit_indent_step(unit, plan.orig_lines, step)

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
        preserve_boundary_pragmas=False,
    )
    if not rep_stmt:
        raise _DelegationCallError(
            f"Failed to generate delegation call for unit {unit.get('name', 'unnamed')}"
        )
    # Tier 1 Semantic Overlap Check against existing plan replacements: O(K)
    for existing_u, _ in plan.replacements:
        if check_units_overlap(existing_u, unit):
            _raise_unit_collision_error(existing_u, unit)

    # Synchronize replacement_items if plan.replacements was populated externally (e.g. fixtures)
    while len(plan.replacement_items) < len(plan.replacements):
        idx = len(plan.replacement_items)
        u_prev, rep_prev = plan.replacements[idx]
        item_prev = resolve_unit_replacement(
            plan.orig_text,
            u_prev,
            rep_prev,
            lines=plan.orig_lines,
            line_char_offsets=plan.line_char_offsets,
            line_byte_offsets=plan.line_byte_offsets,
            order_index=idx,
        )
        plan.replacement_items.append(item_prev)

    # Tier 2 Physical Byte-Span Overlap Check against existing replacement items: O(K)
    new_item = resolve_unit_replacement(
        plan.orig_text,
        unit,
        rep_stmt,
        lines=plan.orig_lines,
        line_char_offsets=plan.line_char_offsets,
        line_byte_offsets=plan.line_byte_offsets,
        order_index=len(plan.replacement_items),
    )
    for existing_item in plan.replacement_items:
        if intervals_overlap(
            existing_item.start_byte,
            existing_item.end_byte,
            new_item.start_byte,
            new_item.end_byte,
        ):
            _raise_unit_collision_error(existing_item.unit, unit)

    plan.replacements.append((unit, rep_stmt))
    plan.replacement_items.append(new_item)
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
            outputs=target_outs2 if target_outs2 is not None else outputs,
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
            new_h_lines.extend(
                ["\n"]
                + [ln.rstrip("\r\n") + "\n" for ln in split_source_lines(h_code)]
                + ["\n"]
            )
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
    coord_valid_reps: List[Tuple[Dict[str, Any], str]] = []
    for u, rep in plan.replacements:
        if not is_valid_unit_coordinates(u, strict=True):
            logger.warning(
                "Skipping replacement with invalid coordinates in %s: %r",
                plan.rel_path,
                u,
            )
            continue
        coord_valid_reps.append((u, rep))

    retained_cands = filter_overlapping_clone_units(
        [u for u, _ in coord_valid_reps], repo_root=repo_root
    )

    def _unit_span(unit: Dict[str, Any]) -> Tuple[int, int]:
        try:
            s = parse_unit_coord(unit, "start", default=1, strict=True)
            return s, parse_unit_coord(unit, "end", default=s, strict=True)
        except (ValueError, TypeError, AttributeError):
            try:
                s = parse_unit_coord(unit, "start", default=1, strict=False)
                return s, parse_unit_coord(unit, "end", default=s, strict=False)
            except (ValueError, TypeError, AttributeError):
                return 1, 1

    def _file_key(unit: Dict[str, Any]) -> str:
        raw = unit_file_str(unit)
        if not raw:
            return ""
        return normalize_path_string(raw, strip_anchor=False)

    def _parse_col_coord(unit: Dict[str, Any], *keys: str) -> Optional[int]:
        for k in keys:
            if k in unit and unit[k] is not None:
                try:
                    val = parse_unit_coord(unit, k, default=None, strict=False)
                    if val is not None:
                        return val
                except (ValueError, TypeError, AttributeError):
                    pass
        return None

    def _unit_exact_key(
        unit: Dict[str, Any],
    ) -> Tuple[str, int, int, Optional[int], Optional[int], str, str]:
        s, e = _unit_span(unit)
        sc = _parse_col_coord(unit, *START_COL_KEYS)
        ec = _parse_col_coord(unit, *END_COL_KEYS)
        return (
            _file_key(unit),
            s,
            e,
            sc,
            ec,
            str(unit.get("name") or ""),
            str(unit.get("kind") or ""),
        )

    retained_ids: Set[int] = {id(r) for r in retained_cands}
    retained_exact_keys = {_unit_exact_key(r) for r in retained_cands}

    filtered_reps: List[Tuple[Dict[str, Any], str]] = []
    for u, rep in coord_valid_reps:
        if id(u) in retained_ids or _unit_exact_key(u) in retained_exact_keys:
            if not any(
                check_units_overlap(u, prev_u, repo_root=repo_root)
                for prev_u, _ in filtered_reps
            ):
                filtered_reps.append((u, rep))

    if replace_clones and filtered_reps:
        try:
            current_text = refactor_module_units(
                plan.orig_text,
                filtered_reps,
                lines=plan.orig_lines,
                line_char_offsets=plan.line_char_offsets,
                line_byte_offsets=plan.line_byte_offsets,
            )
        except UnitCollisionError as exc:
            logger.warning(
                "Collision detected during patch rendering for %s (%s); filtering conflicting replacements",
                plan.rel_path,
                exc,
            )
            valid_reps: List[Tuple[Dict[str, Any], str]] = []
            candidate_text = plan.orig_text
            for item in filtered_reps:
                try:
                    candidate_text = refactor_module_units(
                        plan.orig_text,
                        valid_reps + [item],
                        lines=plan.orig_lines,
                        line_char_offsets=plan.line_char_offsets,
                        line_byte_offsets=plan.line_byte_offsets,
                    )
                    valid_reps.append(item)
                except UnitCollisionError:
                    logger.debug("Omitted conflicting replacement in %s", plan.rel_path)
            filtered_reps = valid_reps
            current_text = candidate_text
    else:
        current_text = plan.orig_text

    nl = detect_line_ending(plan.orig_text)
    if plan.method_helpers:
        deltas = (
            _compute_replacement_line_deltas(
                filtered_reps,
                plan.orig_text,
                lines=plan.orig_lines,
                line_char_offsets=plan.line_char_offsets,
                line_byte_offsets=plan.line_byte_offsets,
            )
            if replace_clones and filtered_reps
            else []
        )
        adjusted_methods = [
            (
                _apply_line_deltas(ins_line, deltas),
                h_code,
            )
            for ins_line, h_code in plan.method_helpers
        ]
        sorted_methods = sorted(adjusted_methods, key=lambda m: m[0], reverse=True)
        for ins_line, h_code in sorted_methods:
            c_lines = split_source_lines(current_text)
            idx = max(0, ins_line - 1)
            h_lines = [ln.rstrip("\r\n") + nl for ln in split_source_lines(h_code)] + [nl]
            current_text = "".join(c_lines[:idx] + h_lines + c_lines[idx:])

    c_lines = split_source_lines(current_text)
    deduped_imports = list(dict.fromkeys(plan.missing_imports))
    lines_with_imports = _insert_imports_into_module(c_lines, deduped_imports)

    if plan.module_helpers:
        ins_idx = _find_module_helper_insertion_index(lines_with_imports)
        all_h_lines: List[str] = []
        for h_code in plan.module_helpers:
            all_h_lines.extend(
                [nl]
                + [ln.rstrip("\r\n") + nl for ln in split_source_lines(h_code)]
                + [nl]
            )
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
