"""Source code inspection, token slicing, docstring bounds, and AST manipulation utilities."""

from __future__ import annotations

import ast
import io
import logging
import textwrap
import tokenize
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.source_lines import (
    count_physical_newlines,
    detect_line_ending,
    split_source_lines,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReplacementItem",
    "UnitDict",
    "UnitSpan",
    "col_offset_to_char_offset",
    "compute_line_offsets",
    "compute_unit_byte_offsets",
    "compute_unit_char_offsets",
    "compute_unit_replacement_span",
    "compute_unit_spans",
    "count_physical_newlines",
    "detect_line_ending",
    "extract_unit_comments_and_pragmas",
    "find_enclosing_function_is_async",
    "is_valid_unit_coordinates",
    "parse_unit_coord",
    "replace_unit_in_source",
    "resolve_unit_replacement",
    "slice_source_by_token_range",
    "split_source_lines",
]

UnitDict = Dict[str, Any]


def parse_unit_coord(unit: UnitDict, key: str, default: int = 1) -> int:
    """Extracts and parses an integer coordinate from a unit dictionary."""
    val = unit.get(key)
    return int(val if val is not None else default)


_parse_unit_coord = parse_unit_coord


def is_valid_unit_coordinates(u: Any) -> bool:
    """Verifies that an AST unit dictionary has valid integer coordinates."""
    if not isinstance(u, dict):
        return False
    try:
        s_val = u.get("start")
        if s_val is not None:
            int(s_val)
        e_val = u.get("end")
        if e_val is not None:
            int(e_val)
        sc_val = u.get("start_col")
        if sc_val is not None:
            int(sc_val)
        ec_val = u.get("end_col")
        if ec_val is not None:
            int(ec_val)
        return True
    except (ValueError, TypeError):
        return False


_is_valid_unit_coordinates = is_valid_unit_coordinates



def _is_docstring_node(node: Optional[ast.AST]) -> bool:
    """Returns True if the AST node is a string literal docstring expression."""
    return bool(
        node is not None
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _extract_docstring_end_line(tree: ast.AST) -> int:
    """Extracts the end line number of a module- or function-level docstring if present."""
    body = getattr(tree, "body", [])
    if body and _is_docstring_node(body[0]):
        return int(getattr(body[0], "end_lineno", getattr(body[0], "lineno", 0)))
    return 0


_extract_module_docstring_end_line = _extract_docstring_end_line


def _extract_unit_body_lines(unit: Dict[str, Any], raw_lines: List[str]) -> List[str]:
    """Extracts executable body lines for a unit, stripping function headers and docstrings for whole functions."""
    if unit.get("kind") not in ("function", "closure", "method"):
        return raw_lines

    code_block = "\n".join(raw_lines)
    dedented = textwrap.dedent(code_block)
    try:
        tree = ast.parse(dedented)
        if len(tree.body) == 1 and isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            fn_node = tree.body[0]
            body_nodes = list(fn_node.body)
            if body_nodes and _is_docstring_node(body_nodes[0]):
                body_nodes = body_nodes[1:]
            if body_nodes:
                d_lines = [ln.rstrip("\r\n") for ln in split_source_lines(dedented)]
                first_body = body_nodes[0]
                start_l = first_body.lineno
                end_l = getattr(body_nodes[-1], "end_lineno", len(d_lines))
                extracted = d_lines[start_l - 1 : end_l]
                if extracted:
                    if start_l == fn_node.lineno:
                        b_col = getattr(first_body, "col_offset", 0)
                        extracted[0] = extracted[0][b_col:]
                    return [
                        ln.rstrip("\r\n")
                        for ln in split_source_lines(textwrap.dedent("\n".join(extracted)))
                    ]
    except Exception:
        pass
    return raw_lines


def _detect_indent_step(indent_str: str) -> str:
    """Detects indentation step (tab, 2 spaces, or 4 spaces) from an indentation prefix."""
    if "\t" in indent_str:
        return "\t"
    if indent_str:
        num_spaces = len(indent_str)
        if num_spaces > 0 and num_spaces % 4 == 0:
            return "    "
        if num_spaces > 0 and num_spaces % 2 == 0:
            return "  "
    return "    "


def _find_header_cookie_boundary(lines: List[str]) -> int:
    """Finds the line index after shebang and PEP 263 source encoding declarations.

    PEP 263 allows encoding cookies on line 1 or line 2 following an initial comment.
    """
    boundary = 0
    if lines and lines[0].startswith("#!"):
        boundary = max(boundary, 1)

    for idx in range(min(2, len(lines))):
        line = lines[idx].strip()
        if line.startswith("#") and ("coding:" in line or "coding=" in line):
            boundary = max(boundary, idx + 1)

    return boundary


def _find_module_helper_insertion_index(lines: List[str]) -> int:
    """Finds the line index after module docstring, future imports, and all module imports."""
    min_insert_idx = _find_header_cookie_boundary(lines)

    try:
        tree = ast.parse("".join(lines))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return min_insert_idx

    last_import_line = 0
    docstring_line = _extract_module_docstring_end_line(tree)

    first_def_line: Optional[int] = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = getattr(node, "lineno", None)
            break

    for node in tree.body:
        if first_def_line is not None and getattr(node, "lineno", 0) >= first_def_line:
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end_l = getattr(node, "end_lineno", node.lineno)
            last_import_line = max(last_import_line, end_l)
        elif isinstance(
            node,
            (ast.Try, getattr(ast, "TryStar", ast.Try), ast.If, ast.With, ast.AsyncWith),
        ):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    end_l = getattr(node, "end_lineno", node.lineno)
                    last_import_line = max(last_import_line, end_l)
                    break

    target_line = max(last_import_line, docstring_line, min_insert_idx)
    if first_def_line is not None:
        target_line = min(target_line, first_def_line - 1)
    return min(max(target_line, min_insert_idx), len(lines))


def _slice_unit_token_lines(unit: Dict[str, Any], lines: List[str]) -> List[str]:
    """Slices source lines to the exact start_col and end_col offsets of the unit."""
    if not lines or unit.get("kind") not in ("comprehension", "complex_expr"):
        return lines
    s_col = _parse_unit_coord(unit, "start_col", default=0)
    e_col = unit.get("end_col")
    res = list(lines)
    if len(res) == 1:
        res[0] = res[0][s_col:e_col] if (e_col is None or e_col > s_col) else res[0][s_col:]
    else:
        res[0] = res[0][s_col:]
        if e_col is not None:
            res[-1] = res[-1][:e_col]
    return res


def _get_module_imported_names(
    source: str, include_conditional: bool = True
) -> Set[str]:
    """Extracts top-level imported module and symbol names from source code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return set()
    imported: Set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
                if not alias.asname and "." in alias.name:
                    imported.add(alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
        elif include_conditional and isinstance(
            stmt,
            (ast.If, ast.Try, getattr(ast, "TryStar", ast.Try), ast.With, ast.AsyncWith),
        ):
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        imported.add(
                            alias.asname or alias.name.split(".", maxsplit=1)[0]
                        )
                        if not alias.asname and "." in alias.name:
                            imported.add(alias.name)
                elif isinstance(sub, ast.ImportFrom):
                    for alias in sub.names:
                        imported.add(alias.asname or alias.name)
    return imported


def _insert_imports_into_module(
    orig_lines: List[str],
    import_lines: List[str],
) -> List[str]:
    """Inserts import statements after module docstring and future imports, deduplicating existing statements."""
    if not import_lines:
        return orig_lines

    existing_stripped = {
        ln.strip() for ln in orig_lines if ln and not ln[0].isspace()
    }
    seen: Set[str] = set()
    deduped_imports: List[str] = []
    for imp in import_lines:
        s = imp.strip()
        if s not in existing_stripped and s not in seen:
            seen.add(s)
            deduped_imports.append(imp)
    future_imps = [
        imp for imp in deduped_imports if imp.strip().startswith("from __future__")
    ]
    other_imps = [
        imp for imp in deduped_imports if not imp.strip().startswith("from __future__")
    ]
    deduped_imports = future_imps + other_imps
    if not deduped_imports:
        return orig_lines

    insert_idx = _find_header_cookie_boundary(orig_lines)

    parsed_with_ast = False
    try:
        tree = ast.parse("".join(orig_lines))
        doc_end = _extract_module_docstring_end_line(tree)
        future_end = max(
            (
                getattr(node, "end_lineno", node.lineno)
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            ),
            default=0,
        )
        insert_idx = max(insert_idx, doc_end, future_end)
        while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
            insert_idx += 1
        parsed_with_ast = True
    except (SyntaxError, ValueError, UnicodeDecodeError):
        pass

    if not parsed_with_ast:
        in_docstring = False
        doc_quote = ""
        doc_prefixes = ('"""', "'''", 'r"""', "r'''", 'R"""', "R'''", 'u"""', "u'''", 'U"""', "U'''")
        for idx in range(insert_idx, len(orig_lines)):
            line = orig_lines[idx].strip()
            if not in_docstring:
                matching_pfx = next((p for p in doc_prefixes if line.startswith(p)), None)
                if matching_pfx:
                    doc_quote = matching_pfx[-3:]
                    if line.endswith(doc_quote) and len(line) > len(matching_pfx):
                        insert_idx = idx + 1
                        break
                    in_docstring = True
                elif line.startswith("#") or not line:
                    continue
                else:
                    break
            else:
                if line.endswith(doc_quote):
                    insert_idx = idx + 1
                    break

        last_future_idx = -1
        for idx in range(insert_idx, len(orig_lines)):
            line = orig_lines[idx].strip()
            if line.startswith("from __future__"):
                last_future_idx = idx
            elif line.startswith("#") or not line:
                continue
            else:
                break

        if last_future_idx != -1:
            insert_idx = last_future_idx + 1
            while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
                insert_idx += 1

    nl = detect_line_ending(orig_lines)
    formatted = [imp.rstrip("\r\n") + nl for imp in deduped_imports]
    return orig_lines[:insert_idx] + formatted + [nl] + orig_lines[insert_idx:]


def slice_source_by_token_range(
    source_text: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: Optional[int] = None,
) -> str:
    """Extracts a precise slice of source code spanning 1-indexed lines and 0-indexed columns.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number.
        start_col: 0-indexed start column offset.
        end_line: 1-indexed end line number.
        end_col: 0-indexed end column offset (exclusive). If None, includes through the end of end_line.

    Returns:
        The exact string slice of source code between the specified coordinates.
    """
    if not source_text:
        return ""
    lines = split_source_lines(source_text)
    if not lines or start_line < 1 or start_line > len(lines) or start_line > end_line:
        return ""

    effective_end_line = min(len(lines), end_line)
    if start_line == effective_end_line:
        line_str = lines[start_line - 1]
        start_c = max(0, min(len(line_str), start_col))
        end_c = len(line_str) if end_col is None else max(start_c, min(len(line_str), end_col))
        return line_str[start_c:end_c]

    first_line = lines[start_line - 1]
    start_c = max(0, min(len(first_line), start_col))
    part_first = first_line[start_c:]
    middle_parts = lines[start_line : effective_end_line - 1]
    last_line = lines[effective_end_line - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), end_col))
    part_last = last_line[:end_c]
    return part_first + "".join(middle_parts) + part_last


def extract_unit_comments_and_pragmas(
    source_text: str,
    start_line: int,
    end_line: int,
    start_col: int = 0,
    end_col: Optional[int] = None,
    include_leading: bool = False,
) -> List[Dict[str, Any]]:
    """Extracts comments and pragmas within or attached to a unit's coordinate bounds.

    Uses Python's standard library `tokenize` module to harvest exact token positions
    for comments, identifying `# type: ignore`, `# noqa`, and `# pylint:` annotations.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number of the unit.
        end_line: 1-indexed end line number of the unit.
        start_col: 0-indexed start column offset.
        end_col: 0-indexed end column offset.
        include_leading: If True, also harvests consecutive comment lines immediately
            preceding start_line.

    Returns:
        A list of dictionaries representing discovered comments:
        [{
            "line": int,
            "col": int,
            "end_col": int,
            "text": str,
            "is_pragma": bool,
            "pragma_kind": Optional[str],
        }]
    """
    if not source_text:
        return []

    lines = split_source_lines(source_text)
    min_check_line = start_line
    if include_leading and start_line > 1:
        curr = start_line - 1
        while curr >= 1:
            stripped = lines[curr - 1].strip()
            if stripped.startswith("#"):
                min_check_line = curr
                curr -= 1
            else:
                break

    results: List[Dict[str, Any]] = []
    try:
        token_gen = tokenize.generate_tokens(io.StringIO(source_text).readline)
        for tok in token_gen:
            if tok.type == tokenize.COMMENT:
                tok_line, tok_col = tok.start
                _, tok_end_col = tok.end

                if min_check_line <= tok_line <= end_line:
                    if tok_line == start_line and tok_line == min_check_line and tok_col < start_col:
                        continue
                    comment_text = tok.string
                    lower = comment_text.lower()
                    is_pragma = False
                    pragma_kind = None
                    if "type: ignore" in lower:
                        is_pragma = True
                        pragma_kind = "type_ignore"
                    elif "noqa" in lower:
                        is_pragma = True
                        pragma_kind = "noqa"
                    elif "pylint:" in lower:
                        is_pragma = True
                        pragma_kind = "pylint"

                    results.append({
                        "line": tok_line,
                        "col": tok_col,
                        "end_col": tok_end_col,
                        "text": comment_text,
                        "is_pragma": is_pragma,
                        "pragma_kind": pragma_kind,
                    })
    except (tokenize.TokenError, IndentationError) as exc:
        logger.debug("Failed to tokenize source for comments: %s", exc)

    return results


class UnitSpan(NamedTuple):
    """Encapsulates exact character and byte offsets for an AST unit.

    Attributes:
        start_char: 0-indexed character offset of the unit start in source text.
        end_char: 0-indexed character offset of the unit end in source text.
        start_byte: 0-indexed UTF-8 byte offset of the unit start in source text.
            Retains original AST UTF-8 byte coordinates (clamped to line byte length).
        end_byte: 0-indexed UTF-8 byte offset of the unit end in source text.
            Retains original AST UTF-8 byte coordinates (clamped to line byte length).
        is_column_bounded: True if the unit is delimited by sub-line column bounds.
        start_line: 1-indexed start line number.
        end_line: 1-indexed end line number.
        start_col_char: 0-indexed character offset within start_line (translated and
            clamped from AST UTF-8 byte offset via col_offset_to_char_offset).
        end_col_char: 0-indexed character offset within end_line (translated and
            clamped from AST UTF-8 byte offset via col_offset_to_char_offset).
    """

    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    is_column_bounded: bool
    start_line: int = 1
    end_line: int = 1
    start_col_char: int = 0
    end_col_char: int = 0


class ReplacementItem(NamedTuple):
    """Encapsulates a fully resolved unit replacement with synchronized spans.

    Attributes:
        unit: AST unit dictionary with 1-indexed 'start' and 'end' lines.
        start_char: 0-indexed character offset of the replacement start.
        end_char: 0-indexed character offset of the replacement end.
        start_byte: 0-indexed UTF-8 byte offset of the replacement start.
        end_byte: 0-indexed UTF-8 byte offset of the replacement end.
        final_rep: Fully formatted replacement text with preserved pragmas/comments.
        consumes_line_suffix: True if trailing line suffix was consumed by pragma preservation.
        order_index: Original replacement sequence index used as a tie-breaker to
            guarantee deterministic, stable descending sorting among non-overlapping
            replacements that share identical starting offsets (e.g. 0-width insertions).
    """

    unit: UnitDict
    start_char: int
    end_char: int
    start_byte: int
    end_byte: int
    final_rep: str
    consumes_line_suffix: bool = False
    order_index: int = 0


def col_offset_to_char_offset(line: str, col_offset: Optional[Union[int, str]] = None) -> int:
    """Translates a 0-indexed column offset into a character index within line.

    Python's AST emits col_offset and end_col_offset as UTF-8 byte offsets.
    This helper translates that byte offset into the character index into line.
    If col_offset lands in the middle of a multi-byte UTF-8 sequence, it safely
    clamps to the nearest valid character boundary preceding the malformed offset.

    Raises:
        ValueError: If col_offset is provided but fails to parse as an integer.
    """
    if col_offset is None:
        return 0
    try:
        c_off = int(col_offset)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed column offset: {col_offset!r}") from err
    if c_off <= 0:
        return 0
    if line.isascii():
        return min(len(line), c_off)
    line_bytes = line.encode("utf-8", errors="surrogatepass")
    if c_off >= len(line_bytes):
        return len(line)
    try:
        prefix = line_bytes[:c_off].decode("utf-8", errors="surrogatepass")
        return len(prefix)
    except UnicodeDecodeError:
        # Clamps to the nearest valid character boundary preceding the malformed byte offset
        # (RFC 3629: UTF-8 sequences are at most 4 bytes, so decrementing by 1..3 bytes reaches the boundary).
        for back in range(1, 4):
            if c_off - back <= 0:
                return 0
            try:
                return len(line_bytes[: c_off - back].decode("utf-8", errors="surrogatepass"))
            except UnicodeDecodeError:
                continue
        return len(line_bytes[:c_off].decode("utf-8", errors="ignore"))


def compute_line_offsets(lines: Sequence[str]) -> Tuple[List[int], List[int]]:
    """Computes cumulative character and UTF-8 byte offsets for line starts."""
    char_offsets = [0]
    byte_offsets = [0]
    for ln in lines:
        ln_len = len(ln)
        char_offsets.append(char_offsets[-1] + ln_len)
        byte_len = ln_len if ln.isascii() else len(ln.encode("utf-8", errors="surrogatepass"))
        byte_offsets.append(byte_offsets[-1] + byte_len)
    return char_offsets, byte_offsets


_compute_line_offsets = compute_line_offsets


def compute_unit_spans(
    source_text: str,
    unit: Dict[str, Any],
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
) -> UnitSpan:
    """Computes exact character and UTF-8 byte spans (UnitSpan) for an AST unit.

    Args:
        source_text: Complete original source code string.
        unit: AST unit dictionary with 1-indexed 'start' and 'end' lines,
            and optional 0-indexed 'start_col' and 'end_col'.
        lines: Optional pre-split lines of source_text.
        line_char_offsets: Optional precomputed line start character offsets.
        line_byte_offsets: Optional precomputed line start UTF-8 byte offsets.

    Returns:
        UnitSpan containing start_char, end_char, start_byte, end_byte, is_column_bounded,
        start_line, end_line, start_col_char, and end_col_char.

    Raises:
        TypeError: If unit is not a dictionary.
        ValueError: If line or column coordinates fail to parse as integers.
    """
    if not isinstance(unit, dict):
        raise TypeError(f"Unit must be a dictionary, got {type(unit).__name__}")

    if lines is None:
        lines = split_source_lines(source_text)
    if not lines:
        return UnitSpan(0, 0, 0, 0, False, 1, 1, 0, 0)

    try:
        start = max(1, _parse_unit_coord(unit, "start", default=1))
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed unit: invalid 'start' line: {unit.get('start')!r}") from err

    try:
        end = max(1, min(len(lines), _parse_unit_coord(unit, "end", default=len(lines))))
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed unit: invalid 'end' line: {unit.get('end')!r}") from err

    if line_char_offsets is None or line_byte_offsets is None:
        line_char_offsets, line_byte_offsets = _compute_line_offsets(lines)

    # Note on inverted ranges: compute_unit_spans treats start > end as an empty 0-width EOF
    # span to ensure no source code is modified during replacement. Harmonized with check_units_overlap,
    # which treats inverted line ranges as empty disjoint sets that never conflict.
    # Note on byte vs. character offsets: start_col_char and end_col_char are translated and
    # clamped character offsets derived from AST byte columns via col_offset_to_char_offset.
    # In contrast, start_byte and end_byte retain the original AST UTF-8 byte coordinates
    # (clamped to line byte length) rather than being recalculated from character offsets.
    if start > len(lines) or start > end:
        sz_c = line_char_offsets[-1]
        sz_b = line_byte_offsets[-1]
        return UnitSpan(sz_c, sz_c, sz_b, sz_b, False, start, end, 0, 0)

    first_line = lines[start - 1]
    last_line = lines[end - 1]
    raw_sc = unit.get("start_col")
    raw_ec = unit.get("end_col")

    try:
        start_col = int(raw_sc) if raw_sc is not None else None
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed unit: invalid 'start_col' offset: {raw_sc!r}") from err

    try:
        end_col = int(raw_ec) if raw_ec is not None else None
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed unit: invalid 'end_col' offset: {raw_ec!r}") from err

    line_code_len = len(last_line.rstrip("\r\n"))
    start_c = col_offset_to_char_offset(first_line, start_col) if start_col is not None else 0
    end_c = col_offset_to_char_offset(last_line, end_col) if end_col is not None else line_code_len
    if start == end:
        end_c = max(start_c, end_c)

    prefix_is_whitespace = not first_line[:start_c].strip()
    suffix_code = last_line[end_c:line_code_len].strip()
    suffix_is_boundary_only = not suffix_code or suffix_code.startswith("#")
    is_expr_kind = unit.get("kind") in ("comprehension", "complex_expr")
    is_column_bounded = bool(
        (start_col is not None or end_col is not None)
        and (is_expr_kind or not (prefix_is_whitespace and suffix_is_boundary_only))
    )

    if is_column_bounded:
        start_char = line_char_offsets[start - 1] + start_c
        end_char = line_char_offsets[end - 1] + end_c

        first_b_len = (
            len(first_line)
            if first_line.isascii()
            else len(first_line.encode("utf-8", errors="surrogatepass"))
        )
        if start == end:
            last_b_len = first_b_len
        elif last_line.isascii():
            last_b_len = len(last_line)
        else:
            last_b_len = len(last_line.encode("utf-8", errors="surrogatepass"))

        nl_len = 2 if last_line.endswith("\r\n") else (1 if last_line.endswith(("\r", "\n")) else 0)
        last_b_code_len = max(0, last_b_len - nl_len)

        start_b = start_col if start_col is not None else 0
        end_b = end_col if end_col is not None else last_b_code_len

        start_b = max(0, min(first_b_len, start_b))
        end_b = max(0, min(last_b_len, end_b))
        if start == end:
            end_b = max(start_b, end_b)
        start_byte = line_byte_offsets[start - 1] + start_b
        end_byte = line_byte_offsets[end - 1] + end_b
    else:
        start_char = line_char_offsets[start - 1]
        end_char = line_char_offsets[end] if end < len(line_char_offsets) else line_char_offsets[-1]
        start_byte = line_byte_offsets[start - 1]
        end_byte = line_byte_offsets[end] if end < len(line_byte_offsets) else line_byte_offsets[-1]

    return UnitSpan(
        start_char=start_char,
        end_char=end_char,
        start_byte=start_byte,
        end_byte=end_byte,
        is_column_bounded=is_column_bounded,
        start_line=start,
        end_line=end,
        start_col_char=start_c,
        end_col_char=end_c,
    )


def _compute_unit_spans(
    source_text: str,
    unit: Dict[str, Any],
) -> Tuple[Tuple[int, int], Tuple[int, int], bool]:
    """Computes ((start_char, end_char), (start_byte, end_byte), is_column_bounded).

    .. deprecated:: 1.4.0
        Use :func:`compute_unit_spans` instead, which returns a structured :class:`UnitSpan`.
    """
    warnings.warn(
        "_compute_unit_spans is deprecated and will be removed in a future release; "
        "use compute_unit_spans instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    span = compute_unit_spans(source_text, unit)
    return (span.start_char, span.end_char), (span.start_byte, span.end_byte), span.is_column_bounded


def compute_unit_byte_offsets(source_text: str, unit: Dict[str, Any]) -> Tuple[int, int]:
    """Computes exact 0-indexed UTF-8 byte offsets (start_byte, end_byte) for an AST unit."""
    span = compute_unit_spans(source_text, unit)
    return span.start_byte, span.end_byte


def compute_unit_char_offsets(source_text: str, unit: Dict[str, Any]) -> Tuple[int, int]:
    """Computes exact 0-indexed character offsets (start_char, end_char) for an AST unit."""
    span = compute_unit_spans(source_text, unit)
    return span.start_char, span.end_char


def resolve_unit_replacement(
    source_text: str,
    unit: Dict[str, Any],
    replacement_text: str,
    preserve_boundary_pragmas: bool = True,
    unit_span: Optional[UnitSpan] = None,
    lines: Optional[Sequence[str]] = None,
    line_char_offsets: Optional[Sequence[int]] = None,
    line_byte_offsets: Optional[Sequence[int]] = None,
    order_index: int = 0,
) -> ReplacementItem:
    """Resolves character and byte slice boundaries and formatted replacement text for unit refactoring.

    Explicitly tracks whether trailing line content is consumed (`consumes_line_suffix`) when
    preserving attached boundary pragmas on column-bounded units, ensuring that `start_char`,
    `end_char`, `start_byte`, and `end_byte` are synchronously updated at the point of origin.

    Args:
        source_text: The complete original Python source code.
        unit: AST unit dictionary with 1-indexed 'start' and 'end' lines.
        replacement_text: The proposed replacement code string.
        preserve_boundary_pragmas: Whether to attach boundary comments/pragmas.
        unit_span: Optional precomputed UnitSpan.
        lines: Optional pre-split lines of source_text.
        line_char_offsets: Optional precomputed line start character offsets.
        line_byte_offsets: Optional precomputed line start UTF-8 byte offsets.
        order_index: Original replacement sequence index for stable tie-breaking.

    Returns:
        ReplacementItem containing synchronized offsets, final replacement, and flags.

    Raises:
        TypeError: If unit is not a dictionary.
        ValueError: If line or column coordinates fail to parse as integers.
    """
    if not isinstance(unit, dict):
        raise TypeError(f"Unit must be a dictionary, got {type(unit).__name__}")

    if lines is None:
        lines = split_source_lines(source_text)
    if not lines:
        return ReplacementItem(
            unit=unit,
            start_char=0,
            end_char=0,
            start_byte=0,
            end_byte=0,
            final_rep="",
            consumes_line_suffix=False,
            order_index=order_index,
        )

    if line_char_offsets is None or line_byte_offsets is None:
        line_char_offsets, line_byte_offsets = _compute_line_offsets(lines)

    if unit_span is None:
        unit_span = compute_unit_spans(
            source_text,
            unit,
            lines=lines,
            line_char_offsets=line_char_offsets,
            line_byte_offsets=line_byte_offsets,
        )

    start_char = unit_span.start_char
    end_char = unit_span.end_char
    start_byte = unit_span.start_byte
    end_byte = unit_span.end_byte
    is_column_bounded = unit_span.is_column_bounded
    start = unit_span.start_line
    end = unit_span.end_line
    if (
        (start_char >= len(source_text) and end_char >= len(source_text))
        or start > len(lines)
        or start > end
    ):
        return ReplacementItem(
            unit=unit,
            start_char=start_char,
            end_char=end_char,
            start_byte=start_byte,
            end_byte=end_byte,
            final_rep="",  # Ensure zero modifications to buffer on empty/invalid slices
            consumes_line_suffix=False,
            order_index=order_index,
        )

    attached_pragmas: List[str] = []
    if preserve_boundary_pragmas:
        boundary_comments = extract_unit_comments_and_pragmas(
            source_text,
            start_line=start,
            end_line=end,
            start_col=unit_span.start_col_char,
            end_col=unit_span.end_col_char,
        )
        for c in boundary_comments:
            if c["is_pragma"] and c["line"] in (start, end):
                p_text = c["text"].strip()
                if p_text not in replacement_text and p_text not in attached_pragmas:
                    attached_pragmas.append(p_text)

    last_line = lines[end - 1]
    end_c = unit_span.end_col_char
    suffix_line = last_line[end_c:]
    suffix_stripped = suffix_line.strip()

    final_rep = replacement_text
    consumes_line_suffix = False

    if is_column_bounded:
        missing_pragmas = [p for p in attached_pragmas if p not in suffix_line]
        pragma_suffix = ("  " + "  ".join(missing_pragmas)) if missing_pragmas else ""
        nl = detect_line_ending(suffix_line, final_rep, source_text)
        if not suffix_stripped and (missing_pragmas or final_rep.endswith(("\r", "\n"))):
            # Clean line end: consume line suffix cleanly to prevent duplicate \n
            consumes_line_suffix = True
            has_nl = suffix_line.endswith(("\r", "\n")) or final_rep.endswith(("\r", "\n"))
            final_rep = final_rep.rstrip("\r\n") + pragma_suffix + (nl if has_nl else "")
            end_char = (
                line_char_offsets[end]
                if end < len(line_char_offsets)
                else line_char_offsets[-1]
            )
            end_byte = (
                line_byte_offsets[end]
                if end < len(line_byte_offsets)
                else line_byte_offsets[-1]
            )
        elif missing_pragmas:
            # Non-comment code or existing trailing comments follow on the same line:
            # consume line suffix to end of line, preserving suffix_line content
            # and appending pragma to the line end.
            consumes_line_suffix = True
            has_suffix_nl = suffix_line.endswith(("\r", "\n"))
            final_suffix = (
                suffix_line.rstrip("\r\n") + pragma_suffix + (nl if has_suffix_nl else "")
            )
            final_rep = final_rep.rstrip("\r\n") + final_suffix
            end_char = (
                line_char_offsets[end]
                if end < len(line_char_offsets)
                else line_char_offsets[-1]
            )
            end_byte = (
                line_byte_offsets[end]
                if end < len(line_byte_offsets)
                else line_byte_offsets[-1]
            )

        return ReplacementItem(
            unit=unit,
            start_char=start_char,
            end_char=end_char,
            start_byte=start_byte,
            end_byte=end_byte,
            final_rep=final_rep,
            consumes_line_suffix=consumes_line_suffix,
            order_index=order_index,
        )

    # Whole-line replacement
    if attached_pragmas and final_rep:
        pragma_suffix = "  " + "  ".join(attached_pragmas)
        rep_lines = split_source_lines(final_rep)
        if rep_lines:
            last_rep = rep_lines[-1]
            nl = detect_line_ending(last_rep, source_text)
            has_nl = last_rep.endswith(("\r", "\n"))
            rep_lines[-1] = last_rep.rstrip("\r\n") + pragma_suffix + (nl if has_nl else "")
            final_rep = "".join(rep_lines)

    if final_rep and not final_rep.endswith(("\r", "\n")):
        nl = detect_line_ending(source_text)
        final_rep += nl

    return ReplacementItem(
        unit=unit,
        start_char=start_char,
        end_char=end_char,
        start_byte=start_byte,
        end_byte=end_byte,
        final_rep=final_rep,
        consumes_line_suffix=False,
        order_index=order_index,
    )


def compute_unit_replacement_span(
    source_text: str,
    unit: Dict[str, Any],
    replacement_text: str,
    preserve_boundary_pragmas: bool = True,
    unit_span: Optional[UnitSpan] = None,
    lines: Optional[Sequence[str]] = None,
) -> Tuple[int, int, str]:
    """Computes exact character slice offsets and formatted replacement text for unit refactoring.

    Returns:
        Tuple of (start_char_offset, end_char_offset, final_replacement_text).
    """
    item = resolve_unit_replacement(
        source_text,
        unit,
        replacement_text,
        preserve_boundary_pragmas=preserve_boundary_pragmas,
        unit_span=unit_span,
        lines=lines,
    )
    return item.start_char, item.end_char, item.final_rep



def replace_unit_in_source(
    source_text: str,
    unit: Dict[str, Any],
    replacement_text: str,
    preserve_boundary_pragmas: bool = True,
) -> str:
    """Replaces an AST code unit in source text with replacement text, preserving comments and formatting.

    Supports sub-line and token-range precision slicing when 'start_col' and 'end_col'
    are present in the unit dictionary. Preserves surrounding indentation, preceding code
    on the start line, trailing pragmas/comments on the end line, and attached pragmas.

    Args:
        source_text: The complete original Python source code.
        unit: The AST unit mapping containing 1-indexed 'start' and 'end' line numbers,
            and optional 0-indexed 'start_col' and 'end_col' column offsets.
        replacement_text: The text to substitute in place of the unit lines.
        preserve_boundary_pragmas: If True, retains boundary pragmas (# type: ignore / # noqa)
            attached to the original code unit and appends them to the replacement statement.

    Returns:
        The modified source code with comments, type ignores, and whitespace outside the unit preserved.
    """
    start_char, end_char, final_rep = compute_unit_replacement_span(
        source_text,
        unit,
        replacement_text,
        preserve_boundary_pragmas=preserve_boundary_pragmas,
    )
    return source_text[:start_char] + final_rep + source_text[end_char:]


def _scan_sig_line(line: str, initial_paren_depth: int = 0) -> Tuple[int, int]:
    """Scans a line for the function header terminating colon and computes resulting paren depth."""
    paren_depth = initial_paren_depth
    in_quote: Optional[str] = None
    for i, ch in enumerate(line):
        if in_quote:
            if ch == in_quote:
                num_bs = 0
                k = i - 1
                while k >= 0 and line[k] == "\\":
                    num_bs += 1
                    k -= 1
                if num_bs % 2 == 0:
                    in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            continue
        if ch == "#":
            break
        if ch in "([{":
            paren_depth += 1
        elif ch in ")]}":
            paren_depth = max(0, paren_depth - 1)
        elif ch == ":" and paren_depth == 0:
            return i, paren_depth
    return -1, paren_depth


def _find_sig_colon(line: str) -> int:
    """Finds the colon terminating a function definition header on a single line."""
    colon_idx, _ = _scan_sig_line(line, 0)
    if colon_idx != -1:
        return colon_idx
    return line.rfind(":")


def find_enclosing_function_is_async(
    source_text: str,
    start_line: int,
    end_line: int,
) -> bool:
    """Checks whether the given line range is enclosed within an AsyncFunctionDef."""
    if start_line <= 0 or end_line < start_line:
        return False
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return False
    candidates: List[Tuple[int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_start = getattr(node, "lineno", 0)
            n_end = getattr(node, "end_lineno", n_start)
            if n_start <= start_line and end_line <= n_end:
                candidates.append((
                    n_end - n_start,
                    isinstance(node, ast.AsyncFunctionDef),
                ))
    if not candidates:
        return False
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
