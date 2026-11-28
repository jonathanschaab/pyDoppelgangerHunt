"""Unit coordinate inference, source freshness validation, and snippet AST parsing."""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
import io
from keyword import iskeyword
import re
import textwrap
import tokenize
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt._ast_compat import (
    is_cpython_parser_stack_overflow,
    is_match_case_node,
    is_match_node,
    is_try_or_trystar_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _ScopeCacheKey,
    _SessionBucket,
    _get_cached_tree_source_lines,
    _get_or_compute_scope_cached,
    _get_scope_parent_map,
    _is_ephemeral_unit_ast,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _node_earliest_start_pos,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _SpanLineIndex,
    _get_span_line_index,
)
from pydoppelgangerhunt.fixer._scope_visitor import _SYNTHETIC_WRAPPER_NAME
from pydoppelgangerhunt.fixer.source import (
    _END_COL_KEYS,
    _START_COL_KEYS,
    _extract_slice_col_offset,
    _is_sliced_excerpt_for_file,
    _is_valid_eval_expr,
    _slice_unit_token_lines,
    _slice_utf8_line,
    dedent_source_text,
    is_subroutine_unit,
    parse_unit_coord,
    split_source_lines,
)
from pydoppelgangerhunt.source_lines import safe_parse_ast
from pydoppelgangerhunt.unit_boundary import (
    END_LINE_KEYS,
    START_LINE_KEYS,
    NormalizedUnit,
    normalize_unit_boundary,
    resolve_unit_source_buffer,
)

SCOPE_SCHEMA_VERSION: int = 2


def _make_empty_scope_result(
    closure_strictness: str,
    non_suppressing_ctx_managers: Sequence[str],
) -> Dict[str, Any]:
    """Returns a complete baseline scope analysis dictionary populated with empty defaults."""
    return {
        "inputs": [],
        "outputs": [],
        "stores": [],
        "free_vars": [],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
        "control_flow_hazards": [],
        "closure_hazard_reasons": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_yield_assignment": False,
        "has_try_or_with": False,
        "has_return": False,
        "has_return_value": False,
        "has_super": False,
        "has_mangled_names": False,
        "local_imports": [],
        "yield_expr_names": [],
        "is_async": False,
        "conditional_outputs": [],
        "definite_stores": [],
        "has_instance_binding": False,
        "has_class_binding": False,
        "binding_kind": None,
        "has_receiver_access": False,
        "instance_attrs": [],
        "class_attrs": [],
        "outputs_ambiguous": False,
        "type_params": [],
        "type_param_free_vars": [],
        "has_class_scoped_type_params": False,
        "deleted_names": [],
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "closure_strictness": closure_strictness,
        "non_suppressing_ctx_managers": list(non_suppressing_ctx_managers),
        "heuristic_ctx_managers": [],
        "wildcard_ctx_managers": [],
    }


# INVARIANT (BOUNDED COLUMN INFERENCE — STAGE 1 PRE-TREE LINE MATCH):
# Only fires when exactly one of (start_col, end_col) is provided and the unit's
# physical line(s) contain a unique AST node matching the supplied bound. Ambiguous
# matches return (s_col, e_col) unchanged so downstream `_df_span_index._resolve_unit_col_bounds`
# / `_has_unresolved_span_bounds` can resolve against `mod_tree` or fail closed.
def _infer_missing_col_from_line_ast(
    lines: Union[str, Sequence[str]],
    s_col: Optional[int],
    e_col: Optional[int],
    unit_kind: Optional[str] = None,
    *,
    allow_expr_fallback: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Infers a missing start_col or end_col from single-line or multiline AST nodes."""
    line_seq: Sequence[str] = (lines,) if isinstance(lines, str) else lines
    if not line_seq:
        return s_col, e_col
    n_lines = len(line_seq)
    if n_lines == 1:
        line0 = line_seq[0]
        stripped = line0.lstrip()
        if not stripped:
            return s_col, e_col
        pad_first = len(line0[: len(line0) - len(stripped)].encode("utf-8"))
        pad_last = pad_first
        parse_src = stripped
    else:
        raw_block = "".join(
            ln if ln.endswith("\n") else ln + "\n" for ln in line_seq
        )
        dedented = dedent_source_text(raw_block)
        d_lines = split_source_lines(dedented)
        if len(d_lines) != n_lines:
            return s_col, e_col
        pad_first = len(line_seq[0].rstrip("\r\n").encode("utf-8")) - len(
            d_lines[0].rstrip("\r\n").encode("utf-8")
        )
        pad_last = len(line_seq[-1].rstrip("\r\n").encode("utf-8")) - len(
            d_lines[-1].rstrip("\r\n").encode("utf-8")
        )
        parse_src = dedented
    mod = safe_parse_ast(parse_src)
    if mod is None:
        return s_col, e_col
    if unit_kind == "comprehension":
        candidate_groups: Tuple[Tuple[type, ...], ...] = (_COMPREHENSION_TYPES,)
    elif unit_kind == "complex_expr":
        candidate_groups = ((ast.expr,),)
    elif allow_expr_fallback:
        candidate_groups = ((ast.stmt,), (ast.expr,))
    else:
        candidate_groups = ((ast.stmt,),)
    for target_types in candidate_groups:
        matched_spans: Set[Tuple[int, int]] = set()
        for node in ast.walk(mod):
            if not isinstance(node, target_types) or isinstance(
                node, (ast.Name, ast.Constant)
            ):
                continue
            if (
                getattr(node, "lineno", None) != 1
                or getattr(node, "end_lineno", None) != n_lines
            ):
                continue
            nc = getattr(node, "col_offset", None)
            nec = getattr(node, "end_col_offset", None)
            if not isinstance(nc, int) or not isinstance(nec, int):
                continue
            abs_s = pad_first + nc
            abs_e = pad_last + nec
            if (n_lines > 1 or abs_s < abs_e) and (
                (s_col is not None and abs_s == s_col)
                or (e_col is not None and abs_e == e_col)
            ):
                matched_spans.add((abs_s, abs_e))
        if len(matched_spans) == 1:
            return next(iter(matched_spans))
        if len(matched_spans) > 1:
            return s_col, e_col
    return s_col, e_col


def _unit_cols(
    unit: Dict[str, Any],
    unit_code_lines: Optional[Sequence[str]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Extracts (start_col, end_col) coordinates from a unit dictionary."""
    s_col = _extract_slice_col_offset(unit, _START_COL_KEYS, default=None)
    e_col = _extract_slice_col_offset(unit, _END_COL_KEYS, default=None)
    u_kind_str = str(unit.get("kind") or "")
    is_inline_expr_kind = u_kind_str in (
        "comprehension",
        "complex_expr",
        "closure",
    )
    if (
        (s_col is None or e_col is None)
        and unit_code_lines
        and isinstance(unit.get("source_code"), str)
    ):
        snippet = unit["source_code"].strip()
        if len(unit_code_lines) == 1 and snippet and "\n" not in snippet:
            line0 = unit_code_lines[0]
            if (
                s_col is not None
                or e_col is not None
                or (
                    snippet != line0.strip()
                    and (";" in line0 or is_inline_expr_kind)
                )
            ):
                start_id = snippet[0].isalnum() or snippet[0] == "_"
                end_id = snippet[-1].isalnum() or snippet[-1] == "_"
                require_stmt_delims = (
                    not is_inline_expr_kind and s_col is None and e_col is None
                )
                search_idx = 0
                exact_matches: List[Tuple[int, int]] = []
                while True:
                    pos = line0.find(snippet, search_idx)
                    if pos < 0:
                        break
                    end_pos = pos + len(snippet)
                    left_ok = not start_id or pos == 0 or not _is_word_or_attr_char(
                        line0[pos - 1]
                    )
                    right_ok = (
                        not end_id
                        or end_pos >= len(line0)
                        or not _is_word_or_attr_char(line0[end_pos])
                    )
                    if left_ok and right_ok:
                        if require_stmt_delims:
                            prefix = line0[:pos].rstrip()
                            suffix = line0[end_pos:].lstrip()
                            if (prefix and not prefix.endswith((";", ":"))) or (
                                suffix and not suffix.startswith((";", "#"))
                            ):
                                search_idx = pos + 1
                                continue
                        cand_s = len(line0[:pos].encode("utf-8"))
                        cand_e = len(line0[:end_pos].encode("utf-8"))
                        if (s_col is None or cand_s == s_col) and (
                            e_col is None or cand_e == e_col
                        ):
                            exact_matches.append((cand_s, cand_e))
                    search_idx = pos + 1
                if len(exact_matches) == 1:
                    best_pair = exact_matches[0]
                    if s_col is None:
                        s_col = best_pair[0]
                    if e_col is None:
                        e_col = best_pair[1]
        elif len(unit_code_lines) > 1 and snippet and "\n" in snippet:
            snip_lines = [
                ln.rstrip("\r\n") for ln in split_source_lines(snippet)
            ]
            if (
                len(snip_lines) == len(unit_code_lines)
                and snip_lines[0]
                and snip_lines[-1]
                and all(
                    snip_lines[i].strip() == unit_code_lines[i].strip()
                    for i in range(1, len(snip_lines) - 1)
                )
            ):
                had_explicit_col = s_col is not None or e_col is not None
                line_first = unit_code_lines[0].rstrip("\r\n")
                line_last = unit_code_lines[-1].rstrip("\r\n")
                if (
                    had_explicit_col
                    or ";" in line_first
                    or ";" in line_last
                    or is_inline_expr_kind
                ):
                    ml_cand_s: Optional[int] = s_col
                    ml_cand_e: Optional[int] = e_col
                    snip_first = snip_lines[0]
                    if ml_cand_s is None:
                        start_id = (
                            snip_first[0].isalnum() or snip_first[0] == "_"
                        )
                        s_matches: List[int] = []
                        idx = 0
                        while True:
                            pos = line_first.find(snip_first, idx)
                            if pos < 0:
                                break
                            tail = line_first[
                                pos + len(snip_first) :
                            ].strip()
                            left_ok = (
                                not start_id
                                or pos == 0
                                or not _is_word_or_attr_char(line_first[pos - 1])
                            )
                            if left_ok and (not tail or tail.startswith("#")):
                                s_matches.append(
                                    len(line_first[:pos].encode("utf-8"))
                                )
                            idx = pos + 1
                        if len(s_matches) == 1:
                            ml_cand_s = s_matches[0]
                    snip_last = snip_lines[-1].lstrip()
                    if ml_cand_e is None and snip_last:
                        end_id = (
                            snip_last[-1].isalnum() or snip_last[-1] == "_"
                        )
                        l_stripped = line_last.lstrip()
                        if l_stripped.startswith(snip_last):
                            end_pos = (
                                len(line_last)
                                - len(l_stripped)
                                + len(snip_last)
                            )
                            right_ok = (
                                not end_id
                                or end_pos >= len(line_last)
                                or not _is_word_or_attr_char(line_last[end_pos])
                            )
                            if right_ok:
                                ml_cand_e = len(
                                    line_last[:end_pos].encode("utf-8")
                                )
                    if had_explicit_col or (
                        ml_cand_s is not None and ml_cand_e is not None
                    ):
                        s_col, e_col = ml_cand_s, ml_cand_e
    if (s_col is None) != (e_col is None) and unit_code_lines:
        s_col, e_col = _infer_missing_col_from_line_ast(
            unit_code_lines,
            s_col,
            e_col,
            unit_kind=u_kind_str,
            allow_expr_fallback=is_inline_expr_kind,
        )
    return s_col, e_col


class _ResolvedUnitCols(NamedTuple):
    """Unit dictionary with inferred sub-line columns attached alongside raw column bounds."""

    unit: Dict[str, Any]
    start_col: Optional[int]
    end_col: Optional[int]


def resolve_unit_inferred_col_bounds(
    unit: Dict[str, Any],
    *,
    repo_root: Optional[str] = None,
    unit_code_lines: Optional[Sequence[str]] = None,
    is_placeholder_origin: Optional[bool] = None,
    explicit_col_sliced: Optional[bool] = None,
    expr_only: bool = True,
) -> _ResolvedUnitCols:
    """Resolves missing sub-line column offsets onto a shallow copy of expression units."""
    raw_s = _extract_slice_col_offset(unit, _START_COL_KEYS, default=None)
    raw_e = _extract_slice_col_offset(unit, _END_COL_KEYS, default=None)
    if (raw_s is not None and raw_s < 0) or (raw_e is not None and raw_e < 0):
        return _ResolvedUnitCols(unit, raw_s, raw_e)
    if expr_only and unit.get("kind") not in ("comprehension", "complex_expr"):
        return _ResolvedUnitCols(unit, raw_s, raw_e)

    if unit_code_lines is None or is_placeholder_origin is None:
        buf = resolve_unit_source_buffer(unit, repo_root=repo_root)
        eff_lines: Sequence[str] = (
            buf.lines if unit_code_lines is None else unit_code_lines
        )
        eff_placeholder = (
            buf.origin.is_placeholder
            if is_placeholder_origin is None
            else is_placeholder_origin
        )
    else:
        eff_lines = unit_code_lines
        eff_placeholder = is_placeholder_origin

    exp_col_sliced = (
        bool(unit.get("source_lines_is_col_sliced"))
        if explicit_col_sliced is None
        else explicit_col_sliced
    )
    inferred_s, inferred_e = _unit_cols(
        unit,
        None if (eff_placeholder or exp_col_sliced) else eff_lines,
    )
    if inferred_s is not None or inferred_e is not None:
        updated = dict(unit)
        if inferred_s is not None:
            updated["start_col"] = inferred_s
        if inferred_e is not None:
            updated["end_col"] = inferred_e
        return _ResolvedUnitCols(updated, inferred_s, inferred_e)
    return _ResolvedUnitCols(unit, inferred_s, inferred_e)


def _is_word_or_attr_char(ch: str) -> bool:
    """Returns True if ch is an identifier character or attribute dot."""
    return ch.isalnum() or ch in ("_", ".")


def _has_token_boundary_subslice(
    sub: str,
    line: str,
    *,
    anchor_start: bool = False,
    anchor_end: bool = False,
    require_stmt_delimiters: bool = False,
) -> bool:
    """Checks whether sub occurs in line at valid lexical token boundaries."""
    if not sub:
        return False
    if sub == line:
        return True
    search_start = 0
    while True:
        pos = line.find(sub, search_start)
        if pos == -1:
            return False
        search_start = pos + 1
        end_pos = pos + len(sub)
        if anchor_start and pos != 0:
            continue
        if anchor_end and end_pos != len(line):
            continue
        if pos > 0 and (sub[0].isalnum() or sub[0] == "_"):
            if _is_word_or_attr_char(line[pos - 1]):
                continue
        if end_pos < len(line) and (sub[-1].isalnum() or sub[-1] == "_"):
            if _is_word_or_attr_char(line[end_pos]):
                continue
        if require_stmt_delimiters:
            prefix = line[:pos].rstrip()
            suffix = line[end_pos:].lstrip()
            if prefix and not prefix.endswith((";", ":")):
                continue
            if suffix and not suffix.startswith((";", "#")):
                continue
        return True


def _is_ordered_contiguous_source_match(
    src_code_lines: Sequence[str],
    raw_lines: Sequence[str],
    *,
    allow_subwindow: bool = True,
    require_stmt_delimiters: Optional[bool] = None,
) -> bool:
    """Checks whether src_code_lines matches raw_lines (with token-boundary boundary lines when allow_subwindow is True)."""
    raw_nonempty = [ln.strip() for ln in raw_lines if ln.strip()]
    k = len(src_code_lines)
    if k == 0 or k != len(raw_nonempty):
        return False
    if not allow_subwindow:
        return list(src_code_lines) == raw_nonempty
    if k == 1:
        target = src_code_lines[0]
        eff_req_stmt_delims = (
            (not _is_valid_eval_expr(target))
            if require_stmt_delimiters is None
            else require_stmt_delimiters
        )
        return _has_token_boundary_subslice(
            target,
            raw_nonempty[0],
            require_stmt_delimiters=eff_req_stmt_delims,
        )
    if not _has_token_boundary_subslice(
        src_code_lines[0],
        raw_nonempty[0],
        anchor_end=True,
        require_stmt_delimiters=bool(require_stmt_delimiters),
    ):
        return False
    if not _has_token_boundary_subslice(
        src_code_lines[-1],
        raw_nonempty[-1],
        anchor_start=True,
        require_stmt_delimiters=bool(require_stmt_delimiters),
    ):
        return False
    return all(
        raw_nonempty[i] == src_code_lines[i] for i in range(1, k - 1)
    )


_CLAUSE_HEADER_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("elif", r"^elif(?:\s|\(|\[|:)"),
    ("else", r"^else\s*:"),
    ("except", r"^except(?:\*|\s|\(|:)"),
    ("finally", r"^finally\s*:"),
    ("case", r"^case(?:\s|\(|\[)"),
)


def _leading_clause_header_kind(text: str) -> Optional[str]:
    """Returns canonical clause header keyword when *text* starts with elif/else/except/finally/case."""
    stripped = dedent_source_text(text).lstrip(" \t\r\n")
    for kind, pattern in _CLAUSE_HEADER_PATTERNS:
        if re.match(pattern, stripped):
            return kind
    return None


def _normalize_clause_header_snippet(dedented: str) -> str:
    """Normalizes standalone elif/else/except/finally/case clause headers into parseable statements."""
    stripped = dedented.lstrip(" \t")
    header_kind = _leading_clause_header_kind(stripped)
    if header_kind == "elif":
        return "if" + stripped[4:]
    if header_kind == "else":
        return "if 1" + stripped[4:]
    if header_kind in ("except", "finally"):
        return f"try:\n    pass\n{stripped}"
    if header_kind == "case":
        return f"match 0:\n{textwrap.indent(stripped, '    ')}"
    return dedented


def _parse_normalized_mod_stmts(normalized_text: str) -> Optional[List[ast.stmt]]:
    """Parses normalized source text into top-level or async-wrapped AST statements."""
    for cand in (
        normalized_text,
        f"async def _norm_wrap():\n{textwrap.indent(normalized_text, '    ')}",
    ):
        parsed = safe_parse_ast(cand)
        if parsed is None:
            continue
        stmts = (
            parsed.body
            if cand == normalized_text
            else getattr(parsed.body[0], "body", [])
        )
        if stmts:
            return list(stmts)
    return None


def _parse_normalized_stmt_dumps(text: str) -> Optional[List[str]]:
    """Parses a code snippet into position-independent AST statement dumps."""
    raw_dedented = dedent_source_text(text).strip()
    if not raw_dedented:
        return None
    dedented = _normalize_clause_header_snippet(raw_dedented)
    stmts = _parse_normalized_mod_stmts(dedented)
    return (
        [ast.dump(s, include_attributes=False) for s in stmts]
        if stmts
        else None
    )


def _scan_line_unquoted_comment(
    line: str,
    in_quote: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Strips an unquoted trailing '#' comment while tracking triple-quote state across lines.

    Note:
        Used as a lightweight line-by-line fallback when ``tokenize.generate_tokens``
        cannot tokenize an incomplete fragment; does not parse Python 3.12 nested
        f-string quotes or raw-string trailing backslashes.
    """
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(in_quote, i):
                i += len(in_quote)
                in_quote = None
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            triple = ch * 3
            if line.startswith(triple, i):
                in_quote = triple
                i += 3
            else:
                in_quote = ch
                i += 1
            continue
        if ch == "#":
            return line[:i].strip(), None
        i += 1
    if in_quote in ("'", '"') and not line.rstrip("\r\n").endswith("\\"):
        in_quote = None
    return line.strip(), in_quote


def _strip_unquoted_comment(line: str) -> str:
    """Strips a trailing '#' comment outside of single, double, or triple-quoted strings."""
    stripped, _ = _scan_line_unquoted_comment(line, None)
    return stripped


def _join_normalized_tokens(toks: Sequence[Tuple[int, str]]) -> str:
    """Joins normalized line tokens while preserving dot-attached attribute access."""
    if not toks:
        return ""
    parts: List[str] = [toks[0][1]]
    for idx in range(1, len(toks)):
        prev_type, prev_str = toks[idx - 1]
        cur_type, cur_str = toks[idx]
        attach_before_dot = cur_str == "." and (
            (prev_type == tokenize.NAME and not iskeyword(prev_str))
            or prev_str in (")", "]", "}")
        )
        attach_after_dot = (
            prev_str == "."
            and cur_type == tokenize.NAME
            and not iskeyword(cur_str)
        )
        if not (attach_before_dot or attach_after_dot):
            parts.append(" ")
        parts.append(cur_str)
    return "".join(parts)


def _strip_inline_comment_line(line: str) -> str:
    """Strips trailing Python comments and normalizes intra-line token spacing."""
    toks: List[Tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(
            io.StringIO(line.lstrip(" \t")).readline
        ):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            ):
                continue
            toks.append((tok.type, tok.string))
        return _join_normalized_tokens(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        if toks and "multi-line statement" in str(exc).lower():
            return _join_normalized_tokens(toks)
        return _strip_unquoted_comment(line)


def _strip_comments_from_lines(lines: Sequence[str]) -> List[str]:
    """Strips comments across a sequence of lines while preserving '#' inside multiline strings."""
    result: List[str] = []
    in_quote: Optional[str] = None
    for ln in lines:
        prev_quote = in_quote
        stripped_ln, next_quote = _scan_line_unquoted_comment(ln, prev_quote)
        in_quote = next_quote
        if prev_quote is None and next_quote is None:
            norm = _strip_inline_comment_line(ln)
        else:
            norm = stripped_ln
        if norm:
            result.append(norm)
    return result


def _is_normalized_source_equivalent(
    src_code: str,
    raw_lines: Sequence[str],
    *,
    allow_subwindow: bool = False,
    require_stmt_delimiters: Optional[bool] = None,
) -> bool:
    """Checks whether normalized or comment-stripped src_code matches raw_lines."""
    raw_text = "".join(raw_lines)
    src_dumps = _parse_normalized_stmt_dumps(src_code)
    raw_dumps = _parse_normalized_stmt_dumps(raw_text)
    src_clause_kind = _leading_clause_header_kind(src_code)
    same_clause_header = src_clause_kind == _leading_clause_header_kind(raw_text)
    has_boundary_semicolon = bool(
        raw_lines and (";" in raw_lines[0] or ";" in raw_lines[-1])
    )
    if src_dumps and raw_dumps:
        src_dump_seq = list(src_dumps)
        raw_dump_seq = list(raw_dumps)
        if not (allow_subwindow and has_boundary_semicolon):
            if same_clause_header and src_dump_seq == raw_dump_seq:
                return True
        elif (
            src_clause_kind is None
            and len(src_dump_seq) <= len(raw_dump_seq)
        ):
            k = len(src_dump_seq)
            for i in range(len(raw_dump_seq) - k + 1):
                if raw_dump_seq[i : i + k] == src_dump_seq:
                    return True
    norm_src_lines = _strip_comments_from_lines(split_source_lines(src_code))
    norm_raw_lines = _strip_comments_from_lines(raw_lines)
    if norm_src_lines and norm_raw_lines:
        return _is_ordered_contiguous_source_match(
            norm_src_lines,
            norm_raw_lines,
            allow_subwindow=allow_subwindow,
            require_stmt_delimiters=require_stmt_delimiters,
        )
    return False


_AST_HASH_MARKER = object()


def _update_hasher_from_ast(hasher: Any, root: ast.AST) -> None:
    """Streams AST structure, fields, and location attributes into hasher without large dump allocations."""
    stack: List[Any] = [root]
    while stack:
        item = stack.pop()
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _AST_HASH_MARKER:
            hasher.update(item[1])
        elif isinstance(item, ast.AST):
            hasher.update(b"N:")
            hasher.update(type(item).__qualname__.encode("utf-8", errors="replace"))
            for attr_name in getattr(item, "_attributes", ()):
                val = getattr(item, attr_name, None)
                if val is not None:
                    hasher.update(b"@")
                    hasher.update(attr_name.encode("utf-8"))
                    hasher.update(b"=")
                    hasher.update(str(val).encode("utf-8", errors="replace"))
            hasher.update(b"(")
            fields = item._fields
            if fields:
                stack.append((_AST_HASH_MARKER, b")"))
                for f_idx in range(len(fields) - 1, -1, -1):
                    fname = fields[f_idx]
                    stack.append(getattr(item, fname, None))
                    stack.append((_AST_HASH_MARKER, f"F:{fname}=".encode("utf-8")))
            else:
                hasher.update(b")")
        elif isinstance(item, list):
            hasher.update(b"[")
            stack.append((_AST_HASH_MARKER, b"]"))
            for elem in reversed(item):
                stack.append(elem)
                stack.append((_AST_HASH_MARKER, b","))
        elif isinstance(item, bytes):
            hasher.update(b"B:")
            hasher.update(str(len(item)).encode("ascii"))
            hasher.update(b":")
            hasher.update(item)
            hasher.update(b";")
        else:
            rep = repr(item).encode("utf-8", errors="replace")
            hasher.update(b"V:")
            hasher.update(type(item).__qualname__.encode("utf-8", errors="replace"))
            hasher.update(b":")
            hasher.update(str(len(rep)).encode("ascii"))
            hasher.update(b":")
            hasher.update(rep)
            hasher.update(b";")


def _get_tree_sha256_digest(mod_tree: ast.AST) -> str:
    """Returns a cached SHA-256 hex digest of mod_tree for downstream cache keys when source_text is absent."""
    def _compute() -> str:
        cached_lines = _get_cached_tree_source_lines(mod_tree)
        hasher = hashlib.sha256()
        if cached_lines is not None:
            for line in cached_lines:
                hasher.update(line.encode("utf-8", errors="replace"))
                hasher.update(b"\n")
            return hasher.hexdigest()
        _update_hasher_from_ast(hasher, mod_tree)
        return hasher.hexdigest()

    if _is_ephemeral_unit_ast(mod_tree):
        return _compute()
    cache_key = _ScopeCacheKey(mod_tree, "tree_sha256_digest")
    return _get_or_compute_scope_cached(_SessionBucket.CELL_INFO, cache_key, _compute)


class _TreeSourceMatch(Enum):
    """Outcome of matching placeholder unit source_code and column bounds against an AST."""

    MATCH = "match"
    STALE = "stale"
    INVALID_COLS = "invalid_cols"


def _extra_outer_paren_slack(dedented_code: str) -> int:
    """Counts outer enclosing '(' ... ')' pairs stripped by ast.parse(..., mode='eval')."""
    text = dedented_code.strip()
    slack = 0
    while len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if not inner or safe_parse_ast(inner, mode="eval") is None:
            break
        slack += 1
        text = inner
    return slack


def _has_preceding_ast_node_on_line(
    u_start: int,
    first_c: int,
    s_col: int,
    span_idx: _SpanLineIndex,
) -> bool:
    """Returns True when another AST node on u_start occupies columns before first_c overlapping [s_col, first_c)."""
    for node in span_idx.by_start_line.get(u_start, ()):
        n_s, _, n_c, _ = span_idx.node_coords(node)
        if n_s == u_start and n_c is not None and n_c < first_c:
            return True
    for node in span_idx.by_end_line.get(u_start, ()):
        _, n_e, _, n_ec = span_idx.node_coords(node)
        if n_e == u_start and n_ec is not None and s_col < n_ec <= first_c:
            return True
    return False


def _matched_ast_cols_compatible(
    first_node: ast.AST,
    last_node: ast.AST,
    u_start: int,
    u_end: int,
    s_col: Optional[int],
    e_col: Optional[int],
    span_idx: _SpanLineIndex,
    *,
    paren_slack: int = 0,
    is_expr: bool = False,
) -> bool:
    """Verifies explicit unit columns fit within the matched backing AST node bounds."""
    if s_col is not None:
        first_s, _, first_c, _ = span_idx.node_coords(first_node)
        if first_s == u_start and first_c is not None:
            if s_col > first_c:
                return False
            if is_expr:
                if s_col < first_c - paren_slack:
                    return False
            else:
                decs = getattr(first_node, "decorator_list", None)
                at_slack = (
                    1
                    if isinstance(decs, list)
                    and decs
                    and getattr(decs[0], "lineno", None) == u_start
                    and getattr(first_node, "lineno", 0) > u_start
                    else 0
                )
                min_stmt_col = max(0, first_c - at_slack)
                if s_col < min_stmt_col and (
                    s_col > 0
                    or _has_preceding_ast_node_on_line(
                        u_start, first_c, s_col, span_idx
                    )
                ):
                    return False
    if e_col is not None:
        _, last_e, _, last_ec = span_idx.node_coords(last_node)
        if (
            last_e == u_end
            and last_ec is not None
            and (e_col < last_ec or e_col > last_ec + paren_slack)
        ):
            return False
    return True


def _placeholder_tree_cols_out_of_bounds(
    unit: Dict[str, Any], file_tree: ast.AST
) -> bool:
    """Checks explicit start_col/end_col against tree AST node bounds on u_start/u_end when source_code is absent."""
    s_col = parse_unit_coord(unit, "start_col", default=None)
    e_col = parse_unit_coord(unit, "end_col", default=None)
    if s_col is None and e_col is None:
        return False
    u_start = parse_unit_coord(unit, "start", default=1)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    span_idx = _get_span_line_index(file_tree)
    end_line_nodes = [
        n
        for n in (
            *span_idx.by_end_line.get(u_end, ()),
            *span_idx.by_start_line.get(u_end, ()),
        )
        if span_idx.node_coords(n)[1] == u_end
    ]
    for col_val, line_nodes, is_start_col in (
        (s_col, span_idx.by_start_line.get(u_start, ()), True),
        (e_col, end_line_nodes, False),
    ):
        if col_val is None or not line_nodes:
            continue
        bound_cols = [
            coord
            for n in line_nodes
            for _, _, n_c, n_ec in (span_idx.node_coords(n),)
            for coord in (n_c if is_start_col else n_ec,)
            if coord is not None
        ]
        if bound_cols and col_val > max(bound_cols):
            return True
    return False


class _ClauseDumpCandidate(NamedTuple):
    """Expected AST dump sequence and enclosing clause constraints for source matching."""

    dumps: List[str]
    header_kind: Optional[str] = None
    expected_dump: Optional[str] = None
    expected_guard_dump: Optional[str] = None


def _is_elif_ast_node(node: ast.AST, parent_map: Mapping[ast.AST, ast.AST]) -> bool:
    """Returns True when *node* is an ast.If representing an `elif` branch of an enclosing `if`/`elif`."""
    if not isinstance(node, ast.If):
        return False
    parent = parent_map.get(node)
    if not isinstance(parent, ast.If):
        return False
    return node in parent.orelse and node.col_offset == parent.col_offset


def _build_clause_dump_candidates(
    raw_dedented: str,
    norm_dedented: str,
    src_dumps: List[str],
) -> List[_ClauseDumpCandidate]:
    """Builds candidate AST dump sequences with clause-header checks for normalized snippets."""
    stmts = _parse_normalized_mod_stmts(norm_dedented)
    if not stmts or len(stmts) != 1:
        return [_ClauseDumpCandidate(dumps=src_dumps)]
    top_s = stmts[0]
    header_kind = _leading_clause_header_kind(raw_dedented)
    candidates: List[_ClauseDumpCandidate] = []
    if header_kind == "elif" and isinstance(top_s, ast.If):
        candidates.append(_ClauseDumpCandidate(dumps=src_dumps, header_kind="elif_node"))
        if top_s.body and not top_s.orelse:
            candidates.append(
                _ClauseDumpCandidate(
                    dumps=[ast.dump(n, include_attributes=False) for n in top_s.body],
                    header_kind="elif_body",
                    expected_dump=ast.dump(top_s.test, include_attributes=False),
                )
            )
    elif header_kind == "else" and isinstance(top_s, ast.If):
        if top_s.body and not top_s.orelse:
            candidates.append(
                _ClauseDumpCandidate(
                    dumps=[ast.dump(n, include_attributes=False) for n in top_s.body],
                    header_kind="else",
                )
            )
    elif header_kind in ("finally", "except") and is_try_or_trystar_node(top_s):
        target_attr = "finalbody" if header_kind == "finally" else "handlers"
        other_attr = "handlers" if header_kind == "finally" else "finalbody"
        target_seq = getattr(top_s, target_attr, ())
        if (
            target_seq
            and not getattr(top_s, other_attr, ())
            and not getattr(top_s, "orelse", ())
        ):
            candidates.append(
                _ClauseDumpCandidate(
                    dumps=[ast.dump(n, include_attributes=False) for n in target_seq],
                    header_kind=header_kind,
                )
            )
    elif header_kind == "case" and is_match_node(top_s):
        cases: Sequence[Any] = getattr(top_s, "cases", ())
        if len(cases) == 1:
            c_node = cases[0]
            c_body = getattr(c_node, "body", ())
            if c_body:
                c_pat = getattr(c_node, "pattern", None)
                c_guard = getattr(c_node, "guard", None)
                candidates.append(
                    _ClauseDumpCandidate(
                        dumps=[ast.dump(n, include_attributes=False) for n in c_body],
                        header_kind="case",
                        expected_dump=(
                            ast.dump(c_pat, include_attributes=False)
                            if c_pat is not None
                            else None
                        ),
                        expected_guard_dump=(
                            ast.dump(c_guard, include_attributes=False)
                            if c_guard is not None
                            else None
                        ),
                    )
                )
    return candidates or [_ClauseDumpCandidate(dumps=src_dumps)]


def _clause_parent_matches_spec(
    node: ast.AST,
    attr_name: str,
    in_span: Sequence[ast.AST],
    cand: _ClauseDumpCandidate,
    parent_map: Mapping[ast.AST, ast.AST],
) -> bool:
    """Verifies enclosing AST container and clause-header predicates satisfy *cand*."""
    kind = cand.header_kind
    if kind is None:
        return True
    matched = False
    if kind == "else":
        matched = attr_name == "orelse"
    elif kind == "except":
        matched = attr_name == "handlers" and is_try_or_trystar_node(node)
    elif kind == "finally":
        matched = attr_name == "finalbody" and is_try_or_trystar_node(node)
    elif kind == "elif_node":
        matched = (
            isinstance(node, ast.If)
            and attr_name == "orelse"
            and len(in_span) == 1
            and _is_elif_ast_node(in_span[0], parent_map)
        )
    elif kind == "elif_body":
        matched = (
            isinstance(node, ast.If)
            and attr_name == "body"
            and ast.dump(node.test, include_attributes=False) == cand.expected_dump
            and _is_elif_ast_node(node, parent_map)
        )
    elif kind == "case" and is_match_case_node(node) and attr_name == "body":
        pat = getattr(node, "pattern", None)
        guard = getattr(node, "guard", None)
        pat_dump = (
            ast.dump(pat, include_attributes=False) if pat is not None else None
        )
        guard_dump = (
            ast.dump(guard, include_attributes=False) if guard is not None else None
        )
        matched = (
            pat_dump == cand.expected_dump
            and guard_dump == cand.expected_guard_dump
        )
    return matched


def _clause_header_line_in_bounds(
    node: ast.AST,
    attr_name: str,
    u_start: int,
    first_lineno: Optional[int],
) -> bool:
    """Returns True when u_start falls on a clause header preceding first_lineno (allowing comments/blank lines)."""
    if first_lineno is None or not u_start < first_lineno:
        return False
    if attr_name == "body":
        node_start = getattr(node, "lineno", None)
        if not isinstance(node_start, int):
            pat = getattr(node, "pattern", None)
            node_start = getattr(pat, "lineno", None)
        return isinstance(node_start, int) and node_start <= u_start
    prev_seq: Any = None
    if attr_name == "orelse":
        prev_seq = getattr(node, "handlers", None) or getattr(node, "body", None)
    elif attr_name == "finalbody":
        prev_seq = (
            getattr(node, "orelse", None)
            or getattr(node, "handlers", None)
            or getattr(node, "body", None)
        )
    if not isinstance(prev_seq, list) or not prev_seq:
        return first_lineno == u_start + 1
    prev_last = prev_seq[-1]
    prev_end = getattr(prev_last, "end_lineno", getattr(prev_last, "lineno", None))
    return isinstance(prev_end, int) and prev_end < u_start


def _match_unit_source_code_with_tree(
    unit: Dict[str, Any],
    src_code: str,
    file_tree: ast.AST,
) -> _TreeSourceMatch:
    """Checks whether unit['source_code'] matches AST nodes in file_tree and validates explicit columns."""
    u_start = parse_unit_coord(unit, "start", default=1)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    s_col = parse_unit_coord(unit, "start_col", default=None)
    e_col = parse_unit_coord(unit, "end_col", default=None)
    has_explicit_cols = s_col is not None or e_col is not None
    u_kind_str = str(unit.get("kind") or "")
    is_inline_expr_kind = u_kind_str in (
        "comprehension",
        "complex_expr",
        "closure",
    )
    span_idx = _get_span_line_index(file_tree)
    end_nodes: List[Tuple[ast.AST, int]] = []
    for cand_node in (
        *span_idx.by_start_line.get(u_end, ()),
        *span_idx.by_end_line.get(u_end, ()),
    ):
        cand_start, cand_end, _c, _ec = span_idx.node_coords(cand_node)
        if cand_end == u_end:
            end_nodes.append((cand_node, cand_start))
    matched_structurally_with_bad_cols = False
    src_dumps = _parse_normalized_stmt_dumps(src_code)
    if src_dumps:
        raw_dedented = dedent_source_text(src_code).strip()
        norm_dedented = _normalize_clause_header_snippet(raw_dedented)
        candidate_specs = (
            _build_clause_dump_candidates(raw_dedented, norm_dedented, src_dumps)
            if norm_dedented != raw_dedented
            else [_ClauseDumpCandidate(dumps=src_dumps)]
        )
        parent_map = _get_scope_parent_map(file_tree)
        candidate_parents: List[ast.AST] = []
        seen_parent_ids: Set[int] = set()
        for node, entry_start in end_nodes:
            if isinstance(node, (ast.stmt, ast.ExceptHandler)):
                n_s = getattr(node, "lineno", entry_start)
                if u_start in (n_s, entry_start):
                    single_dump = [span_idx.get_dump(node)]
                    if any(
                        single_dump == cand.dumps
                        and (
                            cand.header_kind is None
                            or (
                                cand.header_kind == "elif_node"
                                and _is_elif_ast_node(node, parent_map)
                            )
                            or (
                                cand.header_kind == "except"
                                and isinstance(node, ast.ExceptHandler)
                            )
                        )
                        for cand in candidate_specs
                    ):
                        if _matched_ast_cols_compatible(
                            node, node, u_start, u_end, s_col, e_col, span_idx
                        ):
                            return _TreeSourceMatch.MATCH
                        matched_structurally_with_bad_cols = True
                parent_node = parent_map.get(node)
                if parent_node is not None and id(parent_node) not in seen_parent_ids:
                    seen_parent_ids.add(id(parent_node))
                    candidate_parents.append(parent_node)
        for node in candidate_parents:
            for attr_name in ("body", "orelse", "finalbody", "handlers"):
                seq = getattr(node, attr_name, None)
                if not isinstance(seq, list) or not seq:
                    continue
                in_span: List[ast.AST] = []
                for s in seq:
                    s_estart, _ = _node_earliest_start_pos(s)
                    s_lineno = getattr(s, "lineno", s_estart)
                    s_end = getattr(s, "end_lineno", s_lineno)
                    if (
                        isinstance(s_lineno, int)
                        and isinstance(s_end, int)
                        and s_lineno >= u_start
                        and s_end <= u_end
                    ):
                        in_span.append(s)
                if not in_span:
                    continue
                first_estart, _ = _node_earliest_start_pos(in_span[0])
                first_lineno = getattr(in_span[0], "lineno", first_estart)
                last_end = getattr(
                    in_span[-1],
                    "end_lineno",
                    getattr(in_span[-1], "lineno", None),
                )
                exact_start = u_start in (first_lineno, first_estart)
                clause_header_start = (
                    not exact_start
                    and in_span[0] is seq[0]
                    and attr_name in ("body", "orelse", "finalbody")
                    and _clause_header_line_in_bounds(
                        node, attr_name, u_start, first_lineno
                    )
                )
                if not ((exact_start or clause_header_start) and last_end == u_end):
                    continue
                span_dumps = [span_idx.get_dump(s) for s in in_span]
                for cand in candidate_specs:
                    if clause_header_start and cand.header_kind is None:
                        continue
                    target_dumps = cand.dumps
                    if span_dumps == target_dumps and _clause_parent_matches_spec(
                        node, attr_name, in_span, cand, parent_map
                    ):
                        if _matched_ast_cols_compatible(
                            in_span[0], in_span[-1], u_start, u_end, s_col, e_col, span_idx
                        ):
                            return _TreeSourceMatch.MATCH
                        matched_structurally_with_bad_cols = True
                    elif (
                        (
                            cand.header_kind is None
                            or _clause_parent_matches_spec(
                                node, attr_name, in_span, cand, parent_map
                            )
                        )
                        and (has_explicit_cols or (u_start == u_end and len(span_dumps) > 1))
                        and len(target_dumps) <= len(span_dumps)
                    ):
                        k = len(target_dumps)
                        for idx in range(len(span_dumps) - k + 1):
                            if span_dumps[idx : idx + k] == target_dumps and (
                                cand.header_kind != "elif_node"
                                or _clause_parent_matches_spec(
                                    node,
                                    attr_name,
                                    in_span[idx : idx + k],
                                    cand,
                                    parent_map,
                                )
                            ):
                                if _matched_ast_cols_compatible(
                                    in_span[idx],
                                    in_span[idx + k - 1],
                                    u_start,
                                    u_end,
                                    s_col,
                                    e_col,
                                    span_idx,
                                ):
                                    return _TreeSourceMatch.MATCH
                                matched_structurally_with_bad_cols = True
    dedented = dedent_source_text(src_code).strip()
    if dedented and (is_inline_expr_kind or has_explicit_cols or not src_dumps):
        parsed_expr = safe_parse_ast(dedented, mode="eval")
        if parsed_expr is not None:
            try:
                expr_node = parsed_expr.body
                expr_dump = ast.dump(expr_node, include_attributes=False)
                paren_slack = _extra_outer_paren_slack(dedented)
                for node, entry_start in end_nodes:
                    if (
                        isinstance(node, ast.expr)
                        and entry_start == u_start
                        and span_idx.get_dump(node) == expr_dump
                    ):
                        if _matched_ast_cols_compatible(
                            node,
                            node,
                            u_start,
                            u_end,
                            s_col,
                            e_col,
                            span_idx,
                            paren_slack=paren_slack,
                            is_expr=True,
                        ):
                            return _TreeSourceMatch.MATCH
                        matched_structurally_with_bad_cols = True
            except RecursionError:
                pass
    return (
        _TreeSourceMatch.INVALID_COLS
        if matched_structurally_with_bad_cols
        else _TreeSourceMatch.STALE
    )


def _unit_source_code_matches_tree(
    unit: Dict[str, Any],
    src_code: str,
    file_tree: ast.AST,
) -> bool:
    """Checks whether unit['source_code'] matches ordered AST nodes and column bounds in file_tree."""
    return (
        _match_unit_source_code_with_tree(unit, src_code, file_tree)
        is _TreeSourceMatch.MATCH
    )


def _is_single_line_col_span(
    unit: Dict[str, Any],
    unit_code_lines: Sequence[str],
    is_placeholder_origin: bool,
) -> bool:
    """Returns True if unit spans a single physical line."""
    if not is_placeholder_origin and unit_code_lines:
        return len(unit_code_lines) == 1
    s_raw = next((unit.get(k) for k in START_LINE_KEYS if unit.get(k) is not None), None)
    e_raw = next((unit.get(k) for k in END_LINE_KEYS if unit.get(k) is not None), None)
    if s_raw is not None:
        try:
            s_val = parse_unit_coord({"start": s_raw}, "start", default=None)
            e_val = (
                parse_unit_coord({"end": e_raw}, "end", default=None)
                if e_raw is not None
                else s_val
            )
            if s_val is not None and e_val is not None:
                return s_val == e_val
        except (ValueError, TypeError, AttributeError):
            pass
    src_code = unit.get("source_code")
    if isinstance(src_code, str) and src_code.strip():
        return len(split_source_lines(src_code.strip())) == 1
    return True


_LEGACY_SUBROUTINE_OR_EXPR_KINDS: FrozenSet[str] = frozenset(
    {
        "subroutine",
        "basic_block",
        "linear_block",
        "stmt_seq",
        "statement",
        "closure",
        "comprehension",
        "complex_expr",
    }
)


def _is_subroutine_or_nested_expr_unit(
    unit: Dict[str, Any],
    source_text: Optional[str] = None,
    tree: Optional[ast.AST] = None,
) -> bool:
    """Returns True if unit is a subroutine or nested expression requiring closure context."""
    return (
        is_subroutine_unit(unit, source_text=source_text, tree=tree)
        or str(unit.get("kind") or "") in _LEGACY_SUBROUTINE_OR_EXPR_KINDS
    )


def _has_out_of_bounds_unit_cols(
    unit: Dict[str, Any],
    unit_code_lines: Sequence[str],
    *,
    is_placeholder_origin: bool,
    explicit_col_sliced: bool,
) -> bool:
    """Returns True if explicit column coordinates on unit are invalid or out of bounds for unit_code_lines."""
    raw_s_col = _extract_slice_col_offset(unit, _START_COL_KEYS, default=None)
    raw_e_col = _extract_slice_col_offset(unit, _END_COL_KEYS, default=None)
    if (raw_s_col is not None and raw_s_col < 0) or (
        raw_e_col is not None and raw_e_col < 0
    ):
        return True
    if raw_s_col is None and raw_e_col is None:
        return False
    is_single_line = _is_single_line_col_span(
        unit, unit_code_lines, is_placeholder_origin
    )
    eff_s_col = 0 if raw_s_col is None else raw_s_col
    order_ok = not is_single_line or raw_e_col is None or eff_s_col < raw_e_col
    if not order_ok:
        return True
    if is_placeholder_origin or not unit_code_lines or explicit_col_sliced:
        return False
    first_b = len(unit_code_lines[0].rstrip("\r\n").encode("utf-8"))
    last_b = len(unit_code_lines[-1].rstrip("\r\n").encode("utf-8"))
    s_in_bounds = raw_s_col is None or (
        0 <= raw_s_col < first_b
        if (is_single_line or first_b > 0)
        else raw_s_col == 0
    )
    e_in_bounds = raw_e_col is None or (0 <= raw_e_col <= last_b)
    return not (s_in_bounds and e_in_bounds)


class _PreparedUnitScopeSource(NamedTuple):
    """Prepared unit source text, inferred columns, closure policy, or early-exit scope result."""

    early_result: Optional[Dict[str, Any]]
    dedented: str
    inferred_s_col: Optional[int]
    inferred_e_col: Optional[int]
    resolved_strictness: str
    strict_with_flag: bool
    extra_non_sup: FrozenSet[str]
    empty_res: Dict[str, Any]
    normalized: NormalizedUnit


class _ParsedUnitScopeAst(NamedTuple):
    """Parsed unit AST, synthetic loop wrapper depth, statement list, and matched candidate text."""

    tree: Optional[ast.AST]
    loop_offset: int
    candidate_stmts: List[ast.stmt]
    cand_text: str
    had_resource_error: bool


def _refine_semicolon_raw_lines(
    unit: Dict[str, Any],
    raw_lines: List[str],
    *,
    is_placeholder_origin: bool,
    explicit_col_sliced: bool,
    src_code_str: str,
    inferred_s_col: Optional[int],
    inferred_e_col: Optional[int],
    raw_lines_col_sliced: bool,
) -> Tuple[List[str], bool]:
    """Slices single-line or multi-line semicolon/lambda spans to their inferred column bounds."""
    if not (
        not is_placeholder_origin
        and not explicit_col_sliced
        and raw_lines
        and unit.get("kind") not in ("comprehension", "complex_expr")
        and (not src_code_str or "".join(raw_lines).strip() != src_code_str)
    ):
        return raw_lines, raw_lines_col_sliced
    if (
        len(raw_lines) == 1
        and (";" in raw_lines[0] or unit.get("kind") == "closure")
        and inferred_s_col is not None
        and inferred_e_col is not None
        and 0 <= inferred_s_col < inferred_e_col
        <= len(raw_lines[0].rstrip("\r\n").encode("utf-8"))
    ):
        sliced_stmt = _slice_utf8_line(raw_lines[0], inferred_s_col, inferred_e_col)
        if sliced_stmt.strip() and (
            ";" in raw_lines[0]
            or sliced_stmt.lstrip().startswith(("lambda", "(lambda"))
        ):
            return [sliced_stmt], True
        return raw_lines, raw_lines_col_sliced
    if len(raw_lines) > 1 and (";" in raw_lines[0] or ";" in raw_lines[-1]):
        refined = list(raw_lines)
        if (
            inferred_s_col is not None
            and 0 < inferred_s_col < len(refined[0].rstrip("\r\n").encode("utf-8"))
            and ";" in refined[0]
        ):
            lead_ws = refined[0][: len(refined[0]) - len(refined[0].lstrip(" \t"))]
            refined[0] = lead_ws + _slice_utf8_line(
                refined[0], inferred_s_col, None
            ).lstrip(" \t")
            raw_lines_col_sliced = True
        if (
            inferred_e_col is not None
            and 0 < inferred_e_col <= len(refined[-1].rstrip("\r\n").encode("utf-8"))
            and ";" in refined[-1]
        ):
            refined[-1] = _slice_utf8_line(refined[-1], 0, inferred_e_col)
            raw_lines_col_sliced = True
        return refined, raw_lines_col_sliced
    return raw_lines, raw_lines_col_sliced


# INVARIANT (SINGLE BOUNDARY NORMALIZATION & CLOSED FRESHNESS MATCHING):
# All incoming UnitSchema payloads are normalized once via ``normalize_unit_boundary``
# and ``resolve_unit_source_buffer`` inside ``_prepare_unit_scope_source`` before
# freshness checking or wrapper parsing. Do NOT re-parse raw coordinate aliases,
# inline-buffer precedence, or notebook anchors downstream of this boundary.
# Freshness verification between caller-supplied ``source_code`` and backing
# module source/AST is strictly bounded to normalized text and AST-dump matching;
# do NOT expand it into semantic equivalence provers—diverging or ambiguous
# snippets MUST fail closed as ``"stale_source_code"``.
def _validate_unit_source_code_freshness(
    unit: Dict[str, Any],
    raw_lines: List[str],
    *,
    file_tree: Optional[ast.AST],
    get_full_source: Callable[[], Optional[str]],
    col_bounds_invalid: bool,
    is_invalid_nb_or_empty_placeholder: bool,
    is_placeholder_origin: bool,
    src_code_str: str,
    raw_lines_col_sliced: bool,
    inferred_s_col: Optional[int],
    inferred_e_col: Optional[int],
) -> Tuple[List[str], bool, bool]:
    """Checks embedded unit['source_code'] against backing module source/AST to detect stale snippets."""
    stale_source_code_mismatch = False
    if col_bounds_invalid or is_invalid_nb_or_empty_placeholder:
        return raw_lines, col_bounds_invalid, stale_source_code_mismatch
    if is_placeholder_origin and file_tree is not None:
        if src_code_str:
            tree_match = _match_unit_source_code_with_tree(
                unit, str(unit["source_code"]), file_tree
            )
            if tree_match is _TreeSourceMatch.INVALID_COLS:
                col_bounds_invalid = True
            elif tree_match is _TreeSourceMatch.STALE:
                stale_source_code_mismatch = True
        elif _placeholder_tree_cols_out_of_bounds(unit, file_tree):
            col_bounds_invalid = True
        return raw_lines, col_bounds_invalid, stale_source_code_mismatch
    if not (
        src_code_str
        and dedent_source_text("".join(raw_lines)).strip()
        != dedent_source_text(str(unit["source_code"])).strip()
    ):
        return raw_lines, col_bounds_invalid, stale_source_code_mismatch

    src_code_lines = [
        ln.strip()
        for ln in split_source_lines(str(unit["source_code"]))
        if ln.strip()
    ]
    has_boundary_semicolon = bool(
        raw_lines and (";" in raw_lines[0] or ";" in raw_lines[-1])
    )
    is_expr_or_closure_kind = str(unit.get("kind") or "") in (
        "comprehension",
        "complex_expr",
        "closure",
    )
    allow_subwindow = (
        inferred_s_col is not None
        or inferred_e_col is not None
        or has_boundary_semicolon
        or is_expr_or_closure_kind
    )
    require_stmt_delims: Optional[bool] = (
        True
        if not is_expr_or_closure_kind
        and inferred_s_col is None
        and inferred_e_col is None
        else None
    )
    candidate_line_windows: List[Sequence[str]] = [raw_lines]
    if (
        not raw_lines_col_sliced
        and (inferred_s_col is not None or inferred_e_col is not None)
        and raw_lines
    ):
        col_sliced = list(raw_lines)
        first_b_len = len(col_sliced[0].rstrip("\r\n").encode("utf-8"))
        if len(col_sliced) == 1:
            s_c = (
                inferred_s_col
                if inferred_s_col is not None and 0 <= inferred_s_col <= first_b_len
                else 0
            )
            e_c = (
                inferred_e_col
                if inferred_e_col is not None and s_c <= inferred_e_col <= first_b_len
                else None
            )
            col_sliced[0] = _slice_utf8_line(col_sliced[0], s_c, e_c)
        else:
            last_b_len = len(col_sliced[-1].rstrip("\r\n").encode("utf-8"))
            if inferred_s_col is not None and 0 < inferred_s_col < first_b_len:
                col_sliced[0] = _slice_utf8_line(col_sliced[0], inferred_s_col, None)
            if inferred_e_col is not None and 0 < inferred_e_col <= last_b_len:
                col_sliced[-1] = _slice_utf8_line(col_sliced[-1], 0, inferred_e_col)
        candidate_line_windows.append(col_sliced)
    matches_backing_source = any(
        dedent_source_text("".join(win)).strip()
        == dedent_source_text(str(unit["source_code"])).strip()
        or _is_ordered_contiguous_source_match(
            src_code_lines,
            win,
            allow_subwindow=allow_subwindow,
            require_stmt_delimiters=require_stmt_delims,
        )
        or _is_normalized_source_equivalent(
            str(unit["source_code"]),
            win,
            allow_subwindow=allow_subwindow,
            require_stmt_delimiters=require_stmt_delims,
        )
        for win in candidate_line_windows
    )
    if (
        (file_tree is not None or get_full_source() is not None)
        and src_code_lines
        and not matches_backing_source
    ):
        stale_source_code_mismatch = True
    elif not raw_lines or allow_subwindow:
        raw_lines = split_source_lines(str(unit["source_code"]))
    return raw_lines, col_bounds_invalid, stale_source_code_mismatch


def _prepare_unit_scope_source(
    unit: Dict[str, Any],
    *,
    repo_root: Optional[str],
    file_tree: Optional[ast.AST],
    allow_disk_read: bool,
    get_full_source: Callable[[], Optional[str]],
) -> _PreparedUnitScopeSource:
    """Loads, slices, and validates unit source text and closure policy once at the boundary."""
    normalized = normalize_unit_boundary(unit)
    src_buf = resolve_unit_source_buffer(
        unit,
        repo_root=repo_root,
        allow_disk_read=allow_disk_read,
        is_sliced_lines_fn=_is_sliced_excerpt_for_file,
    )
    unit_code_lines = src_buf.lines
    is_placeholder_origin = src_buf.origin.is_placeholder
    # Invalid notebook anchors/buffers and empty placeholders must fail closed without
    # falling back to unit["source_code"], matching _extract_unit_token_lines_with_state.
    is_invalid_nb_or_empty_placeholder = src_buf.is_invalid_notebook or (
        is_placeholder_origin and not unit_code_lines
    )
    explicit_col_sliced = bool(normalized.source_lines_is_col_sliced)
    col_bounds_invalid = _has_out_of_bounds_unit_cols(
        unit,
        unit_code_lines,
        is_placeholder_origin=is_placeholder_origin,
        explicit_col_sliced=explicit_col_sliced,
    )
    src_code_str = normalized.source_code
    is_placeholder_source = is_invalid_nb_or_empty_placeholder or (
        not src_code_str and (is_placeholder_origin or not unit_code_lines)
    )
    resolved_cols = resolve_unit_inferred_col_bounds(
        unit,
        repo_root=repo_root,
        unit_code_lines=unit_code_lines,
        is_placeholder_origin=is_placeholder_origin,
        explicit_col_sliced=explicit_col_sliced,
        expr_only=False,
    )
    inferred_s_col, inferred_e_col = resolved_cols.start_col, resolved_cols.end_col
    raw_lines_col_sliced = False
    sliced_raw: Optional[List[str]]
    if col_bounds_invalid or is_invalid_nb_or_empty_placeholder:
        sliced_raw = []
    elif is_placeholder_origin and src_code_str:
        sliced_raw = split_source_lines(str(unit["source_code"]))
    else:
        sliced_raw = _slice_unit_token_lines(
            resolved_cols.unit,
            unit_code_lines,
            explicit_col_sliced=explicit_col_sliced,
        )
        if sliced_raw is not None and (
            explicit_col_sliced
            or normalized.kind in ("comprehension", "complex_expr")
        ):
            raw_lines_col_sliced = True
    if sliced_raw is None:
        col_bounds_invalid = True
        raw_lines: List[str] = []
    else:
        raw_lines = sliced_raw
    has_nonempty_unit_lines = bool(
        unit_code_lines and any(ln.strip() for ln in unit_code_lines)
    )
    has_explicit_col_bounds = inferred_s_col is not None or inferred_e_col is not None
    if not is_placeholder_origin and unit_code_lines and (
        not raw_lines
        or (
            has_explicit_col_bounds
            and has_nonempty_unit_lines
            and not "".join(raw_lines).strip()
        )
    ):
        col_bounds_invalid = True
    raw_lines, raw_lines_col_sliced = _refine_semicolon_raw_lines(
        unit,
        raw_lines,
        is_placeholder_origin=is_placeholder_origin,
        explicit_col_sliced=explicit_col_sliced,
        src_code_str=src_code_str,
        inferred_s_col=inferred_s_col,
        inferred_e_col=inferred_e_col,
        raw_lines_col_sliced=raw_lines_col_sliced,
    )
    raw_lines, col_bounds_invalid, stale_source_code_mismatch = (
        _validate_unit_source_code_freshness(
            unit,
            raw_lines,
            file_tree=file_tree,
            get_full_source=get_full_source,
            col_bounds_invalid=col_bounds_invalid,
            is_invalid_nb_or_empty_placeholder=is_invalid_nb_or_empty_placeholder,
            is_placeholder_origin=is_placeholder_origin,
            src_code_str=src_code_str,
            raw_lines_col_sliced=raw_lines_col_sliced,
            inferred_s_col=inferred_s_col,
            inferred_e_col=inferred_e_col,
        )
    )
    dedented = dedent_source_text("".join(raw_lines))
    unit_policy = normalized.policy
    resolved_strictness = unit_policy.mode
    strict_with_flag = unit_policy.is_pedantic
    extra_non_sup = frozenset(unit_policy.non_suppressing_ctx_managers)
    empty_res: Dict[str, Any] = _make_empty_scope_result(
        resolved_strictness,
        sorted(extra_non_sup),
    )
    if (
        not col_bounds_invalid
        and not is_placeholder_origin
        and has_explicit_col_bounds
        and has_nonempty_unit_lines
        and not dedented.strip()
    ):
        col_bounds_invalid = True

    early_result: Optional[Dict[str, Any]] = None
    if col_bounds_invalid:
        early_result = {
            **empty_res,
            "control_flow_hazards": ["invalid_unit_columns"],
            "is_control_flow_safe": False,
        }
    elif stale_source_code_mismatch:
        early_result = {
            **empty_res,
            "control_flow_hazards": ["stale_source_code"],
            "is_control_flow_safe": False,
        }
    elif is_placeholder_source:
        placeholder_hazards = (
            ["severed_closure_cell", "unavailable_source_placeholder"]
            if _is_subroutine_or_nested_expr_unit(unit, tree=file_tree)
            else ["unavailable_source_placeholder"]
        )
        early_result = {
            **empty_res,
            "control_flow_hazards": placeholder_hazards,
            "closure_hazard_reasons": ["unavailable_source_placeholder"],
            "is_control_flow_safe": False,
        }
    elif not dedented.strip():
        early_result = empty_res
    return _PreparedUnitScopeSource(
        early_result=early_result,
        dedented=dedented,
        inferred_s_col=inferred_s_col,
        inferred_e_col=inferred_e_col,
        resolved_strictness=resolved_strictness,
        strict_with_flag=strict_with_flag,
        extra_non_sup=extra_non_sup,
        empty_res=empty_res,
        normalized=normalized,
    )


def _parse_unit_scope_ast(dedented: str) -> _ParsedUnitScopeAst:
    """Parses dedented unit text through direct, clause-normalized, function, and loop wrappers."""
    tree: Optional[ast.AST] = None
    loop_offset = 0
    candidate_stmts: List[ast.stmt] = []
    clause_dedented = _normalize_clause_header_snippet(dedented)
    has_clause_alt = clause_dedented != dedented

    parse_candidates: List[Tuple[str, int]] = [
        (dedented, 0),
        *([(clause_dedented, 0)] if has_clause_alt else []),
        (f"if 1:\n{dedented}", 0),
        (f"def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"async def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        *(
            [
                (
                    f"def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(clause_dedented, '    ')}",
                    0,
                ),
                (
                    f"async def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(clause_dedented, '    ')}",
                    0,
                ),
            ]
            if has_clause_alt
            else []
        ),
        (
            f"def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
        (
            f"async def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
        *(
            [
                (
                    f"def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
                    f"{textwrap.indent(clause_dedented, '        ')}",
                    1,
                ),
                (
                    f"async def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
                    f"{textwrap.indent(clause_dedented, '        ')}",
                    1,
                ),
            ]
            if has_clause_alt
            else []
        ),
    ]

    had_resource_error = False
    cand_text = dedented
    for cand_text, offset in parse_candidates:
        try:
            tree = ast.parse(cand_text)
            loop_offset = offset
            if cand_text in (dedented, clause_dedented):
                if isinstance(tree, ast.Module):
                    if len(tree.body) == 1 and isinstance(
                        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = tree.body[0].body
                    else:
                        candidate_stmts = tree.body
            elif cand_text.startswith("if 1:\n"):
                if (
                    isinstance(tree, ast.Module)
                    and len(tree.body) == 1
                    and isinstance(tree.body[0], ast.If)
                ):
                    inner_body = tree.body[0].body
                    tree = ast.Module(body=inner_body, type_ignores=[])
                    if len(inner_body) == 1 and isinstance(
                        inner_body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = inner_body[0].body
                    else:
                        candidate_stmts = inner_body
            elif offset == 1 and isinstance(tree, ast.Module):
                wrapper_fn = tree.body[0]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for_loop = wrapper_fn.body[0]
                    if isinstance(for_loop, (ast.For, ast.AsyncFor, ast.While)):
                        candidate_stmts = for_loop.body
            elif isinstance(tree, ast.Module):
                wrapper_fn = tree.body[0]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    candidate_stmts = wrapper_fn.body
            break
        except RecursionError:
            had_resource_error = True
            break
        except MemoryError as exc:
            if not is_cpython_parser_stack_overflow(exc, source=cand_text):
                raise
            had_resource_error = True
            break
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue
    return _ParsedUnitScopeAst(
        tree=tree,
        loop_offset=loop_offset,
        candidate_stmts=candidate_stmts,
        cand_text=cand_text,
        had_resource_error=had_resource_error,
    )
