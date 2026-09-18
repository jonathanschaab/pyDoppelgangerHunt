"""Source code inspection, token slicing, docstring bounds, and AST manipulation utilities."""

from __future__ import annotations

import ast
import io
import logging
import textwrap
import tokenize
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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
                d_lines = dedented.splitlines()
                first_body = body_nodes[0]
                start_l = first_body.lineno
                end_l = getattr(body_nodes[-1], "end_lineno", len(d_lines))
                extracted = d_lines[start_l - 1 : end_l]
                if extracted:
                    if start_l == fn_node.lineno:
                        b_col = getattr(first_body, "col_offset", 0)
                        extracted[0] = extracted[0][b_col:]
                    return textwrap.dedent("\n".join(extracted)).splitlines()
    except Exception:  # pylint: disable=broad-exception-caught
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
    except SyntaxError:
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
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try), ast.If)):
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
    s_col = unit.get("start_col", 0) or 0
    e_col = unit.get("end_col")
    res = list(lines)
    if len(res) == 1:
        res[0] = res[0][s_col:e_col] if (e_col is None or e_col > s_col) else res[0][s_col:]
    else:
        res[0] = res[0][s_col:]
        if e_col is not None:
            res[-1] = res[-1][:e_col]
    return res


def _get_module_imported_names(source: str) -> Set[str]:
    """Extracts top-level imported module and symbol names from source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imported: Set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(stmt, (ast.If, ast.Try, getattr(ast, "TryStar", ast.Try))):
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
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

    existing_stripped = {ln.strip() for ln in orig_lines}
    seen: Set[str] = set()
    deduped_imports: List[str] = []
    for imp in import_lines:
        s = imp.strip()
        if s not in existing_stripped and s not in seen:
            seen.add(s)
            deduped_imports.append(imp)
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
    except SyntaxError:
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

    formatted = [imp.rstrip("\r\n") + "\n" for imp in deduped_imports]
    return orig_lines[:insert_idx] + formatted + ["\n"] + orig_lines[insert_idx:]


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
    lines = source_text.splitlines(keepends=True)
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

    lines = source_text.splitlines(keepends=True)
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
    lines = source_text.splitlines(keepends=True)
    start = max(1, int(unit.get("start") or 1))
    end = min(len(lines), int(unit.get("end") or len(lines)))
    if start > len(lines) or start > end:
        return source_text

    start_col = unit.get("start_col")
    end_col = unit.get("end_col")

    rep = replacement_text

    # Extract boundary pragmas if requested
    attached_pragmas: List[str] = []
    if preserve_boundary_pragmas:
        boundary_comments = extract_unit_comments_and_pragmas(
            source_text, start_line=start, end_line=end, start_col=start_col or 0, end_col=end_col
        )
        for c in boundary_comments:
            if c["is_pragma"] and c["line"] in (start, end):
                p_text = c["text"].strip()
                if p_text not in rep and p_text not in attached_pragmas:
                    attached_pragmas.append(p_text)

    first_line = lines[start - 1]
    start_c = max(0, min(len(first_line), int(start_col or 0)))
    last_line = lines[end - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), int(end_col)))
    if start == end:
        end_c = max(start_c, end_c)

    prefix_is_whitespace = not first_line[:start_c].strip()
    suffix_stripped = last_line[end_c:].strip()
    suffix_is_boundary_only = not suffix_stripped or suffix_stripped.startswith("#")
    is_expr_kind = unit.get("kind") in ("comprehension", "complex_expr")
    is_column_bounded = (start_col is not None or end_col is not None) and (
        is_expr_kind or not (prefix_is_whitespace and suffix_is_boundary_only)
    )

    if is_column_bounded:
        prefix_line = first_line[:start_c]
        suffix_line = last_line[end_c:]

        prefix_all = "".join(lines[: start - 1]) + prefix_line
        suffix_all = suffix_line + "".join(lines[end:])

        final_rep = rep
        if attached_pragmas and not any(p in suffix_line for p in attached_pragmas):
            pragma_suffix = "  " + "  ".join(attached_pragmas)
            if not suffix_stripped or suffix_stripped.startswith("#"):
                if final_rep.endswith("\n"):
                    final_rep = final_rep[:-1] + pragma_suffix + "\n"
                else:
                    final_rep += pragma_suffix
            else:
                s_lines = suffix_all.splitlines(keepends=True)
                if s_lines:
                    first_s = s_lines[0]
                    if first_s.endswith("\n"):
                        s_lines[0] = first_s[:-1].rstrip() + pragma_suffix + "\n"
                    else:
                        s_lines[0] = first_s.rstrip() + pragma_suffix
                    suffix_all = "".join(s_lines)

        return prefix_all + final_rep + suffix_all

    # Whole-line replacement
    if attached_pragmas and rep:
        pragma_suffix = "  " + "  ".join(attached_pragmas)
        rep_lines = rep.splitlines(keepends=True)
        if rep_lines:
            last_rep = rep_lines[-1]
            if last_rep.endswith("\n"):
                rep_lines[-1] = last_rep[:-1] + pragma_suffix + "\n"
            else:
                rep_lines[-1] = last_rep + pragma_suffix
            rep = "".join(rep_lines)

    if rep and not rep.endswith("\n"):
        rep += "\n"

    prefix_lines = lines[: start - 1]
    suffix_lines = lines[end:]
    return "".join(prefix_lines) + rep + "".join(suffix_lines)


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
