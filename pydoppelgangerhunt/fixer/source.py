"""Source code inspection, token slicing, docstring bounds, and AST manipulation utilities."""

from __future__ import annotations

import ast
import io
import logging
import textwrap
import tokenize
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from pydoppelgangerhunt._ast_compat import (
    is_cpython_parser_stack_overflow,
    is_match_node,
    is_trystar_node,
    safe_parse_ast,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _NESTED_UNIT_KINDS,
    _get_cached_tree_source_lines,
    _get_or_parse_tree,
    _get_valid_unit_bounds,
    _register_tree_source_lines,
)
from pydoppelgangerhunt.fixer._df_cfg import _extract_import_bound_names
from pydoppelgangerhunt.fixer._df_closures import (
    _get_tree_scope_intervals,
    _node_scope_bounds,
)
from pydoppelgangerhunt.fixer._unit_schema import (  # pylint: disable=useless-import-alias
    UnitDict as UnitDict,
    UnitSchema as UnitSchema,
    _parse_unit_coord,
    is_valid_unit_coordinates as is_valid_unit_coordinates,
    validate_unit_schema as validate_unit_schema,
)
from pydoppelgangerhunt.source_lines import (
    count_physical_newlines,
    detect_line_ending,
    is_non_string_sequence,
    is_sliced_unit_source_lines,
    normalize_unit_coordinates,
    parse_unit_coord,
    read_notebook_cell_lines_cached,
    resolve_safe_unit_file_path,
    resolve_unit_line_bounds,
    split_source_lines,
)
from pydoppelgangerhunt.unit_boundary import (
    END_COL_KEYS as _END_COL_KEYS,
    END_LINE_KEYS as _END_LINE_KEYS,
    START_COL_KEYS as _START_COL_KEYS,
    START_LINE_KEYS as _START_LINE_KEYS,
    resolve_unit_source_buffer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ReplacementItem",
    "UnitDict",
    "UnitSchema",
    "UnitSpan",
    "col_offset_to_char_offset",
    "compute_line_offsets",
    "compute_unit_byte_offsets",
    "compute_unit_char_offsets",
    "compute_unit_replacement_span",
    "compute_unit_spans",
    "count_physical_newlines",
    "dedent_source_text",
    "detect_line_ending",
    "extract_unit_comments_and_pragmas",
    "find_enclosing_function_is_async",
    "is_sliced_unit_source_lines",
    "is_subroutine_unit",
    "is_valid_unit_coordinates",
    "lines_end_with_unconditional_return",
    "parse_unit_coord",
    "replace_unit_in_source",
    "resolve_safe_unit_file_path",
    "resolve_unit_declared_outputs",
    "resolve_unit_replacement",
    "slice_source_by_token_range",
    "split_source_lines",
    "validate_unit_schema",
]


_is_valid_unit_coordinates = is_valid_unit_coordinates

def _is_sliced_excerpt_for_file(unit: Dict[str, Any], lines: Sequence[str]) -> bool:
    """Returns True if lines represents a sliced excerpt rather than full file text.

    Note:
        When a unit spans lines 1 through N of an N-line file, len(lines) matches the
        unit line span (e_d - s_d + 1 == N). Callers and upstream parsers supplying
        complete-file line buffers should explicitly set 'source_lines_is_sliced': False
        on the unit dictionary to prevent ambiguous buffers from being treated as
        pre-sliced excerpts.
    """
    if unit.get("source_lines_is_sliced") is True:
        return True
    if not lines:
        return unit.get("source_lines_is_sliced") is not False
    if not is_valid_unit_coordinates(unit):
        return True
    _, u_end = resolve_unit_line_bounds(unit)
    if len(lines) < u_end:
        return True
    if unit.get("source_lines_is_sliced") is False:
        return False
    if is_sliced_unit_source_lines(unit, lines):
        return True
    return False


_SUBROUTINE_KINDS: frozenset[str] = frozenset(
    {"compound_block", "sliding_window", "clause_branch"}
)

_WHOLE_UNIT_KINDS: frozenset[str] = frozenset(
    {
        "function",
        "method",
        "async_function",
        "async_method",
        "closure",
        "comprehension",
        "complex_expr",
        "class",
        "data_table",
    }
)


def is_subroutine_unit(
    unit: Any,
    source_text: Optional[str] = None,
    tree: Optional[ast.AST] = None,
) -> bool:
    """Checks whether an AST code unit is a subroutine block rather than a whole function.

    Classification Rules:
    1. Explicit 'is_subroutine' boolean in unit takes precedence if present.
    2. Known subroutine kinds ('compound_block', 'sliding_window', 'clause_branch') return True.
    3. Whole callable, class, and expression kinds ('function', 'method', 'async_function',
       'async_method', 'closure', 'comprehension', 'complex_expr', 'class', 'data_table')
       return False.
    4. Structural inspection: If source_text is available, checks whether the unit's
       line bounds strictly reside inside an enclosing function (True) or cover it (False).
       Kindless units at top-level module scope return True.
    5. All other kinds or unspecified kinds without source return False.
    """
    if not isinstance(unit, dict):
        return False
    if "is_subroutine" in unit and unit.get("is_subroutine") is not None:
        return bool(unit["is_subroutine"])
    kind = str(unit.get("kind") or "")
    if kind in _SUBROUTINE_KINDS:
        return True
    if kind in _WHOLE_UNIT_KINDS:
        return False

    src = source_text or unit.get("source_text") or unit.get("file_source")
    if not src and is_non_string_sequence(unit.get("source_lines")):
        u_lines = unit["source_lines"]
        if not _is_sliced_excerpt_for_file(unit, u_lines):
            src = "".join(u_lines)
    if is_non_string_sequence(src):
        src = "".join(src)
    if (src and isinstance(src, str)) or tree is not None:
        try:
            u_start = _parse_unit_coord(unit, "start", default=0)
            u_end = _parse_unit_coord(unit, "end", default=u_start)
            if 0 < u_start <= u_end:
                res = _find_innermost_enclosing_node(
                    src if isinstance(src, str) else "",
                    unit,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                    tree=tree,
                )
                if res is not None:
                    matched_node, f_start, f_end = res
                    fn_def_start = getattr(matched_node, "lineno", f_start)
                    if u_start in (fn_def_start, f_start) and u_end == f_end:
                        return False
                    return True
                if not kind:
                    return True
        except (ValueError, TypeError):
            pass

    if not kind and ":" in str(unit.get("name") or ""):
        return True

    return False


def resolve_unit_declared_outputs(unit: Dict[str, Any]) -> Any:
    """Extracts declared unit outputs, preferring precomputed_outputs over outputs."""
    if "precomputed_outputs" in unit and unit["precomputed_outputs"] is not None:
        return unit["precomputed_outputs"]
    if "outputs" in unit and unit["outputs"] is not None:
        return unit["outputs"]
    return None



def _load_unit_file_text(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> Optional[str]:
    """Reads full source text from memory or disk for AST inspection and read analysis.

    Buffer Resolution Priority:
    1. In-memory line sequence (``source_lines``) when non-empty and not recognized as a
       pre-sliced excerpt, matching ``extract_unit_source_lines_with_state``. When
       virtual/in-memory units span lines 1 through N of an N-line file, callers must
       explicitly set ``source_lines_is_sliced: False`` (or provide ``source_text``) to
       prevent length ambiguity heuristics from bypassing the buffer.
    2. Direct in-memory full source text (``source_text`` or ``file_source``).
    3. Empty non-sliced ``source_lines`` buffers (ahead of disk lookup).
    4. Disk-based file resolution within ``repo_root``.
    """
    buf = resolve_unit_source_buffer(
        unit,
        include_lines=False,
        is_sliced_lines_fn=_is_sliced_excerpt_for_file,
    )
    if buf.is_invalid_notebook:
        return None
    if buf.full_text is not None:
        return buf.full_text
    if buf.blocked_by_lines:
        return None

    # Disk resolution: accept .py files and anchored .ipynb#cell_N code cells within
    # repo_root or CWD; unanchored/malformed/out-of-range/non-code notebook targets fail
    # closed to None instead of exposing raw nbformat JSON.
    resolved_file = resolve_safe_unit_file_path(
        unit, repo_root=repo_root, allowed_suffixes=(".py", ".ipynb")
    )
    if resolved_file is not None:
        if resolved_file.suffix.lower() == ".ipynb":
            if buf.notebook_anchor is None:
                return None
            cell_lines = read_notebook_cell_lines_cached(
                resolved_file, buf.notebook_anchor[1]
            )
            return "".join(cell_lines) if cell_lines else None
        try:
            with open(resolved_file, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            pass

    return None


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
        return int(getattr(body[0], "end_lineno", None) or getattr(body[0], "lineno", 0) or 0)
    return 0


_extract_module_docstring_end_line = _extract_docstring_end_line


_FSTRING_START_TOK = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END_TOK = getattr(tokenize, "FSTRING_END", None)
_MULTILINE_STR_TOKEN_TYPES = tuple(
    t
    for t in (
        tokenize.STRING,
        getattr(tokenize, "FSTRING_MIDDLE", None),
    )
    if t is not None
)


def _find_multiline_string_continuation_lines(
    text: str, _retry_unindented: bool = True
) -> Set[int]:
    """Returns 1-based line numbers that start inside a multiline string or f-string literal."""
    cont_lines: Set[int] = set()
    if ("\n" not in text and "\r" not in text) or not (
        "'''" in text
        or '"""' in text
        or "\\\n" in text
        or "\\\r" in text
        or "{" in text
    ):
        return cont_lines
    fstring_start_lines: List[Tuple[int, bool]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text, newline=None).readline):
            if _FSTRING_START_TOK is not None and tok.type == _FSTRING_START_TOK:
                fstring_start_lines.append(
                    (tok.start[0], tok.string.endswith(('"""', "'''")))
                )
            elif _FSTRING_END_TOK is not None and tok.type == _FSTRING_END_TOK:
                if fstring_start_lines:
                    start_line, is_triple = fstring_start_lines.pop()
                    if is_triple and start_line < tok.end[0]:
                        cont_lines.update(range(start_line + 1, tok.end[0] + 1))
            elif tok.type in _MULTILINE_STR_TOKEN_TYPES and tok.start[0] < tok.end[0]:
                cont_lines.update(range(tok.start[0] + 1, tok.end[0] + 1))
    except IndentationError:
        if _retry_unindented:
            stripped_text = "".join(
                ln if idx in cont_lines else ln.lstrip(" \t")
                for idx, ln in enumerate(split_source_lines(text), start=1)
            )
            cont_lines.update(
                _find_multiline_string_continuation_lines(
                    stripped_text, _retry_unindented=False
                )
            )
    except (tokenize.TokenError, SyntaxError) as exc:
        triple_fstr_start = next(
            (ln for ln, is_triple in fstring_start_lines if is_triple), None
        )
        err_start = (
            triple_fstr_start
            if triple_fstr_start is not None
            else (
                exc.args[1][0]
                if (
                    isinstance(exc, tokenize.TokenError)
                    and len(exc.args) >= 2
                    and "multi-line string" in str(exc.args[0]).lower()
                    and isinstance(exc.args[1], tuple)
                    and len(exc.args[1]) >= 1
                    and isinstance(exc.args[1][0], int)
                )
                else None
            )
        )
        if err_start is not None:
            total_lines = len(split_source_lines(text))
            if err_start < total_lines:
                cont_lines.update(range(err_start + 1, total_lines + 1))
    return cont_lines


def _dedent_preserving_multiline_strings(text: str) -> str:
    """Dedents source text while leaving multiline string continuation lines untouched."""
    cont_lines = _find_multiline_string_continuation_lines(text)
    if not cont_lines:
        return textwrap.dedent(text)
    raw_lines = split_source_lines(text)
    masked_lines = [
        "\n" if idx in cont_lines else ln.rstrip("\r\n") + "\n"
        for idx, ln in enumerate(raw_lines, start=1)
    ]
    dedented_masked = split_source_lines(textwrap.dedent("".join(masked_lines)))
    return "".join(
        raw_lines[idx - 1]
        if idx in cont_lines
        else ded_ln.rstrip("\r\n")
        + raw_lines[idx - 1][len(raw_lines[idx - 1].rstrip("\r\n")) :]
        for idx, ded_ln in enumerate(dedented_masked, start=1)
    )


def dedent_source_text(text: str) -> str:
    """Dedents source text while preserving leading form feed characters and multiline strings."""
    lines = split_source_lines(text)
    if not any(ln.startswith("\x0c") for ln in lines):
        return _dedent_preserving_multiline_strings(text)
    ff_mask = [ln.startswith("\x0c") for ln in lines]
    cleaned = "".join(ln[1:] if m else ln for ln, m in zip(lines, ff_mask))
    dedented_lines = split_source_lines(_dedent_preserving_multiline_strings(cleaned))
    return "".join("\x0c" + ln if m else ln for ln, m in zip(dedented_lines, ff_mask))


def _extract_unit_body_lines(unit: Dict[str, Any], raw_lines: List[str]) -> List[str]:
    """Extracts executable body lines for a unit, stripping function headers and docstrings for whole functions."""
    if unit.get("kind") not in (
        "function",
        "closure",
        "method",
        "async_function",
        "async_method",
    ):
        return raw_lines

    code_block = "\n".join(raw_lines)
    dedented = dedent_source_text(code_block)
    parse_src = dedented
    try:
        wrapped = False
        try:
            tree = ast.parse(dedented)
        except IndentationError:
            parse_src = f"if 1:\n{dedented}"
            tree = ast.parse(parse_src)
            wrapped = True
        top_body = (
            tree.body[0].body
            if (wrapped and len(tree.body) == 1 and isinstance(tree.body[0], ast.If))
            else tree.body
        )
        line_offset = 1 if wrapped else 0
        if len(top_body) == 1 and isinstance(
            top_body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            fn_node = top_body[0]
            body_nodes = list(fn_node.body)
            if body_nodes and _is_docstring_node(body_nodes[0]):
                body_nodes = body_nodes[1:]
            if body_nodes:
                d_lines = [ln.rstrip("\r\n") for ln in split_source_lines(dedented)]
                first_body = body_nodes[0]
                start_l = _node_scope_bounds(first_body, d_lines)[0] - line_offset
                end_l = (
                    getattr(body_nodes[-1], "end_lineno", len(d_lines) + line_offset)
                    - line_offset
                )
                extracted = d_lines[start_l - 1 : end_l]
                if extracted:
                    if start_l == fn_node.lineno - line_offset:
                        b_col = getattr(first_body, "col_offset", 0)
                        extracted[0] = _slice_utf8_line(extracted[0], b_col)
                    return [
                        ln.rstrip("\r\n")
                        for ln in split_source_lines(dedent_source_text("\n".join(extracted)))
                    ]
    except MemoryError as exc:
        if not is_cpython_parser_stack_overflow(exc, source=parse_src):
            raise
    except (
        SyntaxError,
        ValueError,
        TypeError,
        UnicodeDecodeError,
        RecursionError,
        IndexError,
    ):
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


def _is_top_level_import_container(node: ast.AST) -> bool:
    """Returns True for top-level compound blocks that may guard module-level imports."""
    return (
        isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith))
        or is_trystar_node(node)
        or is_match_node(node)
    )


def _iter_module_scope_imports(
    root: ast.AST,
) -> Iterator[Union[ast.Import, ast.ImportFrom]]:
    """Yields module-scope Import/ImportFrom nodes inside root without descending into nested scopes."""
    stack: List[ast.AST] = list(reversed(list(ast.iter_child_nodes(root))))
    while stack:
        curr = stack.pop()
        if isinstance(curr, (ast.Import, ast.ImportFrom)):
            yield curr
        elif isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        else:
            stack.extend(reversed(list(ast.iter_child_nodes(curr))))


def _find_module_helper_insertion_index(lines: List[str]) -> int:
    """Finds the line index after module docstring, future imports, and all module imports."""
    min_insert_idx = _find_header_cookie_boundary(lines)

    tree = safe_parse_ast("".join(lines))
    if tree is None:
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
            end_l = int(getattr(node, "end_lineno", None) or getattr(node, "lineno", 0) or 0)
            last_import_line = max(last_import_line, end_l)
        elif _is_top_level_import_container(node):
            for _ in _iter_module_scope_imports(node):
                end_l = int(
                    getattr(node, "end_lineno", None) or getattr(node, "lineno", 0) or 0
                )
                last_import_line = max(last_import_line, end_l)
                break

    target_line = max(last_import_line, docstring_line, min_insert_idx)
    if first_def_line is not None:
        target_line = min(target_line, first_def_line - 1)
    return min(max(target_line, min_insert_idx), len(lines))


def _extract_slice_col_offset(
    unit: Dict[str, Any], keys: Tuple[str, ...], default: Optional[int]
) -> Optional[int]:
    """Safely extracts a non-negative column coordinate from key aliases, returning -1 on error."""
    for key in keys:
        if unit.get(key) is not None:
            try:
                val = _parse_unit_coord(unit, key, default=None, strict=True)
                return -1 if (val is None or val < 0) else val
            except (ValueError, TypeError):
                return -1
    return default


def _slice_utf8_line(
    line: str, s_col: int, e_col: Optional[int] = None
) -> str:
    """Slices a single line by UTF-8 byte column offsets.

    When e_col < s_col (inverted column bounds), intentionally falls back to
    line[s_char:] rather than collapsing the line to empty.
    """
    s_char = col_offset_to_char_offset(line, s_col)
    if e_col is not None and e_col >= s_col:
        return line[s_char : col_offset_to_char_offset(line, e_col)]
    return line[s_char:]


def _is_valid_eval_expr(text: str) -> bool:
    """Returns True if text parses as a standalone Python expression."""
    stripped = text.strip()
    if not stripped:
        return False
    return safe_parse_ast(stripped, mode="eval") is not None


def _slice_unit_token_lines(
    unit: Dict[str, Any],
    lines: List[str],
    *,
    explicit_col_sliced: Optional[bool] = None,
) -> Optional[List[str]]:
    """Slices source lines to the exact start_col and end_col offsets of the unit.

    Line-sliced excerpts still undergo first/last-line column clipping unless
    ``source_lines_is_col_sliced=True`` (or ``explicit_col_sliced=True``) explicitly
    declares that column offsets were already stripped. Out-of-bounds or invalid
    ``start_col`` / ``end_col`` coordinates fail closed by returning ``None`` so
    callers can distinguish column-slicing failures from legitimately empty inputs.
    """
    if not lines or unit.get("kind") not in ("comprehension", "complex_expr"):
        return lines
    raw_s_col = _extract_slice_col_offset(unit, _START_COL_KEYS, default=None)
    e_col = _extract_slice_col_offset(unit, _END_COL_KEYS, default=None)
    if (raw_s_col is not None and raw_s_col < 0) or (e_col is not None and e_col < 0):
        return None
    raw_col_flag = unit.get("source_lines_is_col_sliced")
    eff_col_sliced = (
        explicit_col_sliced
        if explicit_col_sliced is not None
        else (raw_col_flag if isinstance(raw_col_flag, bool) else None)
    )
    if eff_col_sliced is True:
        return list(lines)
    s_col = 0 if raw_s_col is None else raw_s_col
    res = list(lines)
    first_byte_len = len(res[0].rstrip("\r\n").encode("utf-8"))
    if len(res) == 1:
        if (
            s_col > first_byte_len
            or (raw_s_col is not None and s_col >= first_byte_len)
            or (e_col is not None and (e_col > first_byte_len or e_col <= s_col))
        ):
            return None
        res[0] = _slice_utf8_line(res[0], s_col, e_col)
        return res
    last_byte_len = len(res[-1].rstrip("\r\n").encode("utf-8"))
    if (
        s_col > first_byte_len
        or 0 < first_byte_len == s_col
        or (e_col is not None and e_col > last_byte_len)
    ):
        return None
    clipped = list(res)
    if 0 < s_col < first_byte_len:
        clipped[0] = _slice_utf8_line(clipped[0], s_col, None)
    if e_col is not None:
        clipped[-1] = _slice_utf8_line(clipped[-1], 0, e_col)
    return clipped


class _UnitTokenLinesResult(NamedTuple):
    """Column-sliced unit body lines paired with placeholder-origin status."""

    lines: Optional[List[str]]
    is_placeholder: bool


def _extract_unit_token_lines_with_state(
    unit: Dict[str, Any], *, repo_root: Optional[str] = None
) -> _UnitTokenLinesResult:
    """Extracts body and column-sliced lines with placeholder status, falling back to ``source_code``."""
    buf = resolve_unit_source_buffer(unit, repo_root=repo_root)
    raw_lines = buf.lines
    origin = buf.origin
    if buf.is_invalid_notebook or (origin.is_placeholder and not raw_lines):
        return _UnitTokenLinesResult(None, True)
    used_source_code_fallback = False
    if origin.is_placeholder:
        src_code = unit.get("source_code")
        if isinstance(src_code, str) and src_code.strip():
            raw_lines = split_source_lines(src_code)
            used_source_code_fallback = True
    sliced = _slice_unit_token_lines(
        unit,
        _extract_unit_body_lines(
            unit, [ln.rstrip("\r\n") for ln in raw_lines]
        ),
        explicit_col_sliced=True if used_source_code_fallback else None,
    )
    return _UnitTokenLinesResult(
        sliced,
        origin.is_placeholder and not used_source_code_fallback,
    )


def _extract_unit_token_lines(
    unit: Dict[str, Any], *, repo_root: Optional[str] = None
) -> Optional[List[str]]:
    """Extracts body and column-sliced lines for a unit, falling back to ``source_code`` on non-empty placeholder origins."""
    return _extract_unit_token_lines_with_state(unit, repo_root=repo_root).lines


def _record_module_import_names(
    stmt: Union[ast.Import, ast.ImportFrom],
    imported: Set[str],
) -> None:
    """Records bound names (plus full dotted names for unaliased Import) into imported."""
    imported.update(_extract_import_bound_names(stmt))
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            if not alias.asname and "." in alias.name:
                imported.add(alias.name)
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            if alias.name == "*" and not alias.asname:
                imported.add("*")


def _get_module_imported_names(
    source: str, include_conditional: bool = True
) -> Set[str]:
    """Extracts top-level imported module and symbol names from source code."""
    tree = safe_parse_ast(source)
    if tree is None:
        return set()
    imported: Set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _record_module_import_names(stmt, imported)
        elif include_conditional and _is_top_level_import_container(stmt):
            for sub in _iter_module_scope_imports(stmt):
                _record_module_import_names(sub, imported)
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
    tree = safe_parse_ast("".join(orig_lines))
    if tree is not None:
        doc_end = _extract_module_docstring_end_line(tree)
        future_end = max(
            (
                int(
                    getattr(node, "end_lineno", None)
                    or getattr(node, "lineno", 0)
                    or 0
                )
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            ),
            default=0,
        )
        insert_idx = max(insert_idx, doc_end, future_end)
        while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
            insert_idx += 1
        parsed_with_ast = True

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
        token_gen = tokenize.generate_tokens(
            io.StringIO(source_text, newline=None).readline
        )
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


def _validate_and_parse_unit_bound(
    u: Dict[str, Any], coord_name: str, strict: bool = False
) -> int:
    """Extracts and validates a required unit coordinate, raising ValueError on failure."""
    try:
        val = parse_unit_coord(u, coord_name, default=None, strict=strict)
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"Malformed unit: invalid {coord_name!r} line: {u.get(coord_name)!r}"
        ) from err
    if val is None:
        raise ValueError(
            f"Malformed unit: missing required {coord_name!r} line coordinate"
        )
    return val


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

    raw_start = _validate_and_parse_unit_bound(unit, "start", strict=False)
    start = max(1, raw_start)

    raw_end = _validate_and_parse_unit_bound(unit, "end", strict=False)
    end = max(1, min(len(lines), raw_end))

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
    raw_sc = next((unit.get(k) for k in _START_COL_KEYS if unit.get(k) is not None), None)
    raw_ec = next((unit.get(k) for k in _END_COL_KEYS if unit.get(k) is not None), None)

    try:
        start_col = _parse_unit_coord(unit, "start_col", default=None)
    except (ValueError, TypeError) as err:
        raise ValueError(f"Malformed unit: invalid 'start_col' offset: {raw_sc!r}") from err

    try:
        end_col = _parse_unit_coord(unit, "end_col", default=None)
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
        is_expr_rep = is_column_bounded or unit.get("kind") in (
            "comprehension",
            "complex_expr",
        )
        for c in boundary_comments:
            if c["is_pragma"] and (
                c["line"] in (start, end)
                or (is_expr_rep and start <= c["line"] <= end)
            ):
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
    strict: bool = True,
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
        strict: If True, validates unit coordinates with strict positive integer rules.

    Returns:
        The modified source code with comments, type ignores, and whitespace outside the unit preserved.
    """
    if not isinstance(unit, dict):
        raise TypeError(f"Unit must be a dictionary, got {type(unit).__name__}")
    unit = normalize_unit_coordinates(unit)
    _validate_and_parse_unit_bound(unit, "start", strict=strict)
    _validate_and_parse_unit_bound(unit, "end", strict=strict)

    if not is_valid_unit_coordinates(unit, strict=strict):
        raise ValueError(
            f"Malformed unit: invalid coordinates for replacement: {unit!r}"
        )
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


def _find_innermost_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
    tree: Optional[ast.AST] = None,
) -> Optional[Tuple[ast.AST, int, int]]:
    """Locates the innermost AST node of matching types enclosing the given unit."""
    bounds = _get_valid_unit_bounds(unit)
    if bounds is None:
        return None
    u_start, u_end = bounds

    if is_non_string_sequence(source_text):
        source_text = "".join(source_text)
    elif not isinstance(source_text, str):
        return None

    if tree is None:
        if not source_text.strip():
            return None
        tree = _get_or_parse_tree(source_text)
        if tree is None:
            return None

    src_lines: Optional[Tuple[str, ...]] = _get_cached_tree_source_lines(tree)
    if src_lines is None and source_text:
        src_lines = _register_tree_source_lines(tree, source_text=source_text)
        if src_lines is None:
            src_lines = tuple(
                ln.rstrip("\r\n") for ln in split_source_lines(source_text)
            )

    candidates: List[Tuple[int, ast.AST, int, int]] = []
    intervals = _get_tree_scope_intervals(tree, source_lines=src_lines)
    for earliest_start, n_end, span, node in intervals:
        if isinstance(node, node_types):
            if earliest_start <= u_start <= u_end <= n_end:
                candidates.append((span, node, earliest_start, n_end))

    if not candidates and not all(
        issubclass(nt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for nt in node_types
        if isinstance(nt, type)
    ):
        for node in ast.walk(tree):
            if isinstance(node, node_types):
                earliest_start, n_end = _node_scope_bounds(node, src_lines)
                if earliest_start <= u_start <= u_end <= n_end:
                    candidates.append((n_end - earliest_start, node, earliest_start, n_end))

    is_sub_kind = (
        bool(unit.get("is_subroutine")) or unit.get("kind") in _NESTED_UNIT_KINDS
    )
    if is_sub_kind and candidates:
        u_start_col = _extract_slice_col_offset(unit, _START_COL_KEYS, default=None)
        has_outer_fn = any(
            isinstance(c_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not (
                u_start in (c_earliest, getattr(c_node, "lineno", c_earliest))
                and u_end == c_end
            )
            for _, c_node, c_earliest, c_end in candidates
        )
        filtered_cands = []
        for item in candidates:
            _, c_node, c_earliest, c_end = item
            c_lineno = getattr(c_node, "lineno", c_earliest) or c_earliest
            if u_start in (c_earliest, c_lineno) and u_end == c_end:
                if c_lineno < c_end or isinstance(c_node, ast.ClassDef):
                    continue
                c_col = getattr(c_node, "col_offset", 0) or 0
                if has_outer_fn and (
                    u_start_col is None or u_start_col < 0 or u_start_col <= c_col
                ):
                    continue
            filtered_cands.append(item)
        candidates = filtered_cands

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (item[0], -item[2], -getattr(item[1], "col_offset", 0))
    )
    _, matched_node, n_start, n_end = candidates[0]
    return matched_node, n_start, n_end


def find_enclosing_function_is_async(
    source_text: str,
    start_line: int,
    end_line: int,
    tree: Optional[ast.AST] = None,
) -> bool:
    """Checks whether the given line range is enclosed within an AsyncFunctionDef."""
    res = _find_innermost_enclosing_node(
        source_text,
        {"start": start_line, "end": end_line},
        (ast.FunctionDef, ast.AsyncFunctionDef),
        tree=tree,
    )
    if res is None:
        return False
    return isinstance(res[0], ast.AsyncFunctionDef)


def lines_end_with_unconditional_return(cand_lines: Sequence[str]) -> bool:
    """Returns True when dedented candidate lines end with an unconditional top-level return statement."""
    if not cand_lines:
        return False
    joined = "".join(
        ln if ln.endswith(("\n", "\r")) else f"{ln}\n" for ln in cand_lines
    )
    dedented = dedent_source_text(joined)
    if not dedented.strip():
        return False
    for wrapper in (None, "async def _w():\n"):
        src = (
            dedented
            if wrapper is None
            else wrapper + textwrap.indent(dedented, "    ")
        )
        mod = safe_parse_ast(src)
        if mod is None:
            continue
        body = (
            mod.body[0].body
            if wrapper is not None
            and len(mod.body) == 1
            and isinstance(mod.body[0], ast.AsyncFunctionDef)
            else mod.body
        )
        return bool(body) and isinstance(body[-1], ast.Return)
    return False


_ScopeValT = TypeVar("_ScopeValT")


def _copy_scope_value(val: _ScopeValT) -> _ScopeValT:
    """Recursively copies plain container values in a unit scope dictionary, preserving NamedTuples."""
    if isinstance(val, dict):
        return cast(_ScopeValT, {k: _copy_scope_value(v) for k, v in val.items()})
    if isinstance(val, list):
        return cast(_ScopeValT, [_copy_scope_value(item) for item in val])
    if isinstance(val, tuple):
        copied_items = [_copy_scope_value(item) for item in val]
        make_fn = getattr(type(val), "_make", None)
        if hasattr(val, "_fields") and callable(make_fn):
            return cast(_ScopeValT, make_fn(copied_items))
        return cast(_ScopeValT, tuple(copied_items))
    if isinstance(val, set):
        return cast(_ScopeValT, {_copy_scope_value(item) for item in val})
    return val


def _copy_scope_dict(scope: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a recursive copy of a unit scope dictionary."""
    return {k: _copy_scope_value(v) for k, v in scope.items()}
