"""Physical source line splitting, physical newline counting, and line-ending detection utilities."""

from __future__ import annotations

import ast
from collections import OrderedDict
from collections.abc import Sequence as _AbcSequence
import enum
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import threading
import tokenize
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from pydoppelgangerhunt.canonical_path import parse_notebook_cell_anchor
from pydoppelgangerhunt.config import normalize_path_string

if TYPE_CHECKING:
    from typing import TypeGuard

logger = logging.getLogger(__name__)

_PHYSICAL_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")

# INVARIANT (CONVERGENCE BOUNDARY — CPython < 3.12 PEG parser stack overflow shim):
# Do NOT add per-operator regex tables or ad-hoc grammar counters here. CPython 3.12+
# attaches an explicit parser-stack-overflow message to MemoryError. On Python < 3.12,
# all PEG recursion overflows share three finite, grammar-agnostic structural bounds:
# bracket nesting depth, indentation nesting depth, or unbroken operator/keyword-op
# density within a single comma/colon/semicolon/newline-delimited expression frame.
# Delete this shim completely once Python < 3.12 support is removed.
_EXPR_OP_KEYWORDS: frozenset[str] = frozenset(
    {"if", "else", "and", "or", "not", "in", "is", "lambda"}
)
_MAX_PARSER_BRACKET_DEPTH: int = 200
_MAX_PARSER_INDENT_DEPTH: int = 100
_MAX_PARSER_INDENT_COLS: int = 400
_MAX_PARSER_FRAME_OPS: int = 200

__all__ = [
    "SourceLinesOrigin",
    "clear_notebook_cache",
    "coerce_unit_file_str",
    "count_physical_newlines",
    "detect_line_ending",
    "extract_notebook_code_cell_text",
    "extract_unit_source_code",
    "extract_unit_source_lines_with_state",
    "is_cpython_parser_stack_overflow",
    "is_non_string_sequence",
    "is_sliced_unit_source_lines",
    "normalize_unit_coordinates",
    "parse_unit_coord",
    "read_notebook_cell_lines_cached",
    "resolve_safe_unit_file_path",
    "resolve_unit_line_bounds",
    "safe_parse_ast",
    "split_source_lines",
    "unit_file_str",
    "with_unit_source_text",
]


def coerce_unit_file_str(raw_file: Any, *, default: str = "") -> str:
    """Coerces a unit file path (str or str-backed os.PathLike) to str via os.fspath()."""
    if raw_file is None:
        return default
    if isinstance(raw_file, str):
        return raw_file if raw_file else default
    if isinstance(raw_file, os.PathLike):
        try:
            fsp = os.fspath(raw_file)
        except TypeError:
            return default
        if isinstance(fsp, str):
            return fsp if fsp else default
        return default
    return str(raw_file) if raw_file else default


def unit_file_str(unit: Mapping[str, Any], *, default: str = "") -> str:
    """Returns unit['file'] as a str, honoring os.fspath() for custom os.PathLike[str] values."""
    return coerce_unit_file_str(unit.get("file"), default=default)


def is_non_string_sequence(val: Any) -> TypeGuard[Sequence[Any]]:
    """Returns True if val is a Sequence other than str, bytes, or bytearray."""
    return isinstance(val, _AbcSequence) and not isinstance(
        val, (str, bytes, bytearray)
    )


def _source_has_pathological_parser_nesting(source: Union[str, bytes]) -> bool:
    """Returns True when *source* exceeds structural nesting or expression-frame operator bounds.

    Uses Python's own lexical tokenizer rather than operator-specific regexes so all
    expression chains (arithmetic, boolean, attribute, ternary, unary, subscript, call)
    converge under one bounded expression-frame counter that resets on statement and
    collection delimiters (``NEWLINE``, ``,``, ``:``, ``;``).
    """
    text = source.decode("latin-1") if isinstance(source, bytes) else source
    if len(text) < 200:
        return False
    bracket_depth = 0
    indent_depth = 0
    expr_frames: List[int] = [0]
    reader = io.StringIO(text).readline
    try:
        for tok in tokenize.generate_tokens(reader):
            tok_type = tok.type
            tok_str = tok.string
            if tok_type == tokenize.INDENT:
                indent_depth += 1
                if (
                    indent_depth >= _MAX_PARSER_INDENT_DEPTH
                    or len(tok_str.expandtabs(8)) >= _MAX_PARSER_INDENT_COLS
                ):
                    return True
            elif tok_type == tokenize.DEDENT:
                indent_depth = max(0, indent_depth - 1)
            elif tok_type == tokenize.NEWLINE:
                expr_frames[-1] = 0
            elif tok_type == tokenize.OP:
                if tok_str in "([{":
                    expr_frames[-1] += 1
                    if expr_frames[-1] >= _MAX_PARSER_FRAME_OPS:
                        return True
                    bracket_depth += 1
                    if bracket_depth >= _MAX_PARSER_BRACKET_DEPTH:
                        return True
                    expr_frames.append(0)
                elif tok_str in ")]}":
                    bracket_depth = max(0, bracket_depth - 1)
                    if len(expr_frames) > 1:
                        expr_frames.pop()
                elif tok_str in (",", ":", ";"):
                    expr_frames[-1] = 0
                else:
                    expr_frames[-1] += 1
                    if expr_frames[-1] >= _MAX_PARSER_FRAME_OPS:
                        return True
            elif tok_type == tokenize.NAME and tok_str in _EXPR_OP_KEYWORDS:
                expr_frames[-1] += 1
                if expr_frames[-1] >= _MAX_PARSER_FRAME_OPS:
                    return True
    except IndentationError as exc:
        return "too many levels of indentation" in str(exc).lower()
    except tokenize.TokenError:
        return False
    return False


def is_cpython_parser_stack_overflow(
    exc: MemoryError,
    source: Optional[Union[str, bytes]] = None,
) -> bool:
    """Returns True when *exc* is CPython's parser nesting-limit MemoryError rather than heap OOM.

    CPython 3.12+ attaches an explicit parser-stack-overflow message to the exception,
    whereas CPython 3.9-3.11 raises a bare ``MemoryError()`` with empty ``args`` from the
    PEG parser on deeply nested inputs. On Python < 3.12, a bare ``MemoryError()`` is
    classified as a parser stack overflow only when *source* exhibits pathological nesting
    or operator repetition so genuine heap-OOM errors continue to propagate.
    """
    msg = str(exc).lower()
    if (
        "parser stack overflow" in msg
        or "too complex to parse" in msg
        or "too deeply nested" in msg
    ):
        return True
    if (
        sys.version_info < (3, 12)
        and not msg
        and not exc.args
        and source is not None
    ):
        return _source_has_pathological_parser_nesting(source)
    return False


@overload
def safe_parse_ast(
    source: Union[str, bytes],
    filename: str = ...,
    mode: Literal["exec"] = ...,
) -> Optional[ast.Module]:
    ...


@overload
def safe_parse_ast(
    source: Union[str, bytes],
    filename: str = ...,
    mode: Literal["eval"] = ...,
) -> Optional[ast.Expression]:
    ...


@overload
def safe_parse_ast(
    source: Union[str, bytes],
    filename: str = ...,
    mode: str = ...,
) -> Optional[ast.AST]:
    ...


def safe_parse_ast(
    source: Union[str, bytes],
    filename: str = "<unknown>",
    mode: str = "exec",
) -> Optional[ast.AST]:
    """Parses Python source, returning None on syntax/recursion/parser-nesting errors while propagating heap OOM."""
    try:
        return ast.parse(source, filename=filename, mode=mode)
    except (SyntaxError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return None
    except MemoryError as exc:
        if is_cpython_parser_stack_overflow(exc, source=source):
            return None
        raise


class SourceLinesOrigin(enum.Enum):
    """Origin enum for unit source lines returned by extract_unit_source_lines_with_state.

    Query explicit properties (``is_pre_sliced``, ``is_placeholder``, ``has_real_source``)
    to distinguish pre-sliced excerpts, synthetic placeholders, and full-source reads.
    """

    PRE_SLICED = enum.auto()
    PLACEHOLDER = enum.auto()
    FULL_BUFFER = enum.auto()
    DISK = enum.auto()

    @property
    def is_pre_sliced(self) -> bool:
        """True if lines are already pre-sliced (or a placeholder)."""
        return self in (SourceLinesOrigin.PRE_SLICED, SourceLinesOrigin.PLACEHOLDER)

    @property
    def is_placeholder(self) -> bool:
        """True if lines are a synthetic missing-source placeholder."""
        return self is SourceLinesOrigin.PLACEHOLDER

    @property
    def has_real_source(self) -> bool:
        """True if lines come from actual source text rather than a synthetic placeholder."""
        return self is not SourceLinesOrigin.PLACEHOLDER

_COORD_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "start_col": ("start_col", "start_col_offset", "col_offset"),
    "start_col_offset": ("start_col_offset", "start_col", "col_offset"),
    "col_offset": ("col_offset", "start_col", "start_col_offset"),
    "end_col": ("end_col", "end_col_offset"),
    "end_col_offset": ("end_col_offset", "end_col"),
}


def normalize_unit_coordinates(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Populates canonical 'start_col' and 'end_col' keys from offset aliases if missing."""
    if not isinstance(unit, dict):
        return unit
    res = dict(unit)
    if res.get("start_col") is None:
        if res.get("start_col_offset") is not None:
            res["start_col"] = res["start_col_offset"]
        elif res.get("col_offset") is not None:
            res["start_col"] = res["col_offset"]
    if res.get("end_col") is None and res.get("end_col_offset") is not None:
        res["end_col"] = res["end_col_offset"]
    return res


@overload
def parse_unit_coord(unit: Dict[str, Any], key: str, *, strict: bool = ...) -> int:
    ...


@overload
def parse_unit_coord(
    unit: Dict[str, Any], key: str, default: int, *, strict: bool = ...
) -> int:
    ...


@overload
def parse_unit_coord(
    unit: Dict[str, Any], key: str, default: None, *, strict: bool = ...
) -> Optional[int]:
    ...


@overload
def parse_unit_coord(
    unit: Dict[str, Any], key: str, default: Optional[int], *, strict: bool = ...
) -> Optional[int]:
    ...


def parse_unit_coord(
    unit: Dict[str, Any],
    key: str,
    default: Optional[int] = 1,
    *,
    strict: bool = False,
) -> Optional[int]:
    """Extracts and parses an integer coordinate from a unit dictionary.

    Safely handles integers, numeric strings, and colon-delimited coordinate strings
    (e.g., '12:0' or '8:0'). For colon-formatted coordinates emitted by external linters
    or diagnostics (where the suffix denotes a sub-column or character index), the primary
    leading coordinate prefix before the colon is parsed as the integer value.
    In strict mode (strict=True), colon-formatted coordinates, empty/whitespace strings,
    signed prefix strings, negative integers, and floating point values are strictly
    rejected with ValueError.
    If the coordinate value is missing or empty (in non-strict mode), falls back to default.
    Raises ValueError for non-numeric strings or unconvertible/overflow values.
    """
    val = unit.get(key)
    if val is None and key in _COORD_KEY_ALIASES:
        for alias in _COORD_KEY_ALIASES[key]:
            val = unit.get(alias)
            if val is not None:
                break
    if val is None:
        return default
    if strict:
        if isinstance(val, str):
            if ":" in val:
                raise ValueError(
                    f"Invalid coordinate {val!r}: colon format not allowed in strict mode"
                )
            stripped = val.strip()
            if not stripped:
                raise ValueError(
                    f"Invalid coordinate {val!r}: blank coordinate not allowed in strict mode"
                )
            if stripped.startswith(("+", "-")):
                raise ValueError(
                    f"Invalid coordinate {val!r}: signed prefix not allowed in strict mode"
                )
            try:
                return int(val)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert to integer"
                ) from exc
        if isinstance(val, bool):
            raise ValueError(
                f"Invalid coordinate {val!r}: boolean coordinate not allowed in strict mode"
            )
        if isinstance(val, float):
            raise ValueError(
                f"Invalid coordinate {val!r}: float coordinate not allowed in strict mode"
            )
        if isinstance(val, int):
            if val < 0:
                raise ValueError(
                    f"Invalid coordinate {val!r}: negative integer not allowed in strict mode"
                )
            return val
        res = _coerce_int_coord(val)
        if res < 0:
            raise ValueError(
                f"Invalid coordinate {val!r}: negative integer not allowed in strict mode"
            )
        return res

    if isinstance(val, str):
        val = val.split(":", 1)[0].strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError as exc:
            try:
                f_val = float(val)
            except (ValueError, OverflowError) as f_exc:
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert to integer"
                ) from f_exc
            if math.isnan(f_val) or math.isinf(f_val):
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert non-finite float"
                ) from exc
            return int(f_val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid coordinate {val!r}: cannot convert non-finite float")
        return int(val)
    return _coerce_int_coord(val)


def _coerce_int_coord(val: Any) -> int:
    """Coerces coordinate value to int or raises ValueError on invalid/overflow inputs."""
    try:
        return int(val)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid coordinate {val!r}: cannot convert to integer") from exc


# INVARIANT (EXPLICIT SLICE FLAG — NO CONTENT/INDENTATION SNIFFING):
# Explicit `unit["source_lines_is_sliced"]` is always authoritative. When omitted
# by legacy callers, only exact line-count equality (`len(lines) == end - start + 1`)
# is checked; do NOT add indentation, syntax, or AST-sniffing heuristics here.
def is_sliced_unit_source_lines(
    unit: Dict[str, Any],
    source_lines: Optional[Sequence[str]] = None,
) -> bool:
    """Determines whether a unit's source_lines collection represents a pre-sliced excerpt."""
    if not isinstance(unit, dict):
        return False
    is_sliced = unit.get("source_lines_is_sliced")
    if is_sliced is not None:
        return bool(is_sliced)
    lines = source_lines if source_lines is not None else unit.get("source_lines")
    if not lines or not is_non_string_sequence(lines):
        return False
    try:
        raw_s = parse_unit_coord(unit, "start", default=None)
        if raw_s is None or raw_s < 1:
            return False
        raw_e = parse_unit_coord(unit, "end", default=raw_s)
    except (ValueError, TypeError, AttributeError):
        return False
    if raw_e is None or raw_e < raw_s:
        return False
    expected_len = raw_e - raw_s + 1
    return len(lines) == expected_len


def resolve_unit_line_bounds(
    unit: Dict[str, Any], *, strict: bool = False
) -> Tuple[int, int]:
    """Extracts (start, end) line coordinates from a unit dictionary.

    When strict=True, raises ValueError if unit is not a dictionary, if start or end
    is missing, non-positive, or malformed, or if end < start.
    When strict=False, clamps start to >= 1 and end to >= start, logging a DEBUG
    diagnostic whenever clamping occurs.
    """
    if not isinstance(unit, dict):
        if strict:
            raise ValueError("Invalid unit: expected dictionary")
        return 1, 1
    if strict:
        if "start" not in unit or unit.get("start") is None:
            raise ValueError("Missing unit start coordinate")
        if "end" not in unit or unit.get("end") is None:
            raise ValueError("Missing unit end coordinate")
        raw_s = parse_unit_coord(unit, "start", default=None, strict=True)
        if raw_s is None or raw_s < 1:
            raise ValueError(f"Invalid unit start coordinate: {unit.get('start')!r}")
        raw_e = parse_unit_coord(unit, "end", default=None, strict=True)
        if raw_e is None or raw_e < raw_s:
            raise ValueError(
                f"Invalid unit line bounds: start={raw_s!r}, end={unit.get('end')!r}"
            )
        return raw_s, raw_e
    try:
        raw_s = parse_unit_coord(unit, "start", default=1)
    except (ValueError, TypeError, AttributeError):
        raw_s = 1
    try:
        raw_e = parse_unit_coord(unit, "end", default=raw_s)
    except (ValueError, TypeError, AttributeError):
        raw_e = raw_s
    s = max(1, raw_s)
    e = max(s, raw_e)
    if s != raw_s or e != raw_e:
        logger.debug(
            "Clamped invalid unit coordinates (start=%r -> %d, end=%r -> %d) for unit %s",
            raw_s,
            s,
            raw_e,
            e,
            unit.get("name") or unit.get("file"),
        )
    return s, e


def split_source_lines(source_text: str) -> List[str]:
    """Splits source code into physical lines with line terminators preserved.

    Unlike str.splitlines(keepends=True), this strictly splits on Python's physical
    newline grammar (\\r\\n, \\r, \\n) and never splits on intra-line form feeds (\\f / \\x0c)
    or vertical tabs (\\v), keeping line coordinates synchronized with Python ASTs.
    Uses regex finditer to avoid intermediate match list allocations.

    Args:
        source_text: The complete original Python source code.

    Returns:
        A list of physical line strings with line terminators preserved.
    """
    if not source_text:
        return []
    return [m.group(0) for m in _PHYSICAL_LINE_RE.finditer(source_text) if m.group(0)]


def extract_notebook_code_cell_text(cell: Any) -> Optional[str]:
    """Returns the joined source text of a Jupyter notebook code cell, or None if malformed.

    Follows nbformat semantics: ``source`` is either a string or a list of string
    fragments concatenated verbatim (fragments may embed or omit newlines, so they
    are joined rather than treated as one physical line each). Returns None for
    non-dict cells, non-``"code"`` cell types, a missing ``source`` key, or a
    ``source`` that is neither a string nor a list consisting solely of strings, so
    malformed notebook data is never coerced into Python source.
    """
    if not isinstance(cell, dict) or cell.get("cell_type") != "code":
        return None
    src = cell.get("source")
    if isinstance(src, str):
        return src
    if isinstance(src, list) and all(isinstance(fragment, str) for fragment in src):
        return "".join(src)
    return None


def count_physical_newlines(text: str) -> int:
    """Counts the number of physical line terminators (\\r\\n, \\r, or \\n) in text.

    Args:
        text: Input string to scan for line terminators.

    Returns:
        Total count of physical line endings.
    """
    if not text:
        return 0
    if "\r" not in text:
        return text.count("\n")
    # Total newlines = (LF including CRLF) + (CR including CRLF) - (CRLF counted twice)
    return text.count("\n") + text.count("\r") - text.count("\r\n")


def _iter_sources(
    sources: Sequence[Union[Optional[str], Iterable[Any]]]
) -> Iterator[str]:
    """Flattens mixed string arguments and string iterables into a flat stream of non-empty strings."""
    for item in sources:
        if not item:
            continue
        if isinstance(item, str):
            yield item
        elif isinstance(item, Iterable):
            for sub in item:
                if isinstance(sub, str) and sub:
                    yield sub


def detect_line_ending(*sources: Union[Optional[str], Iterable[Optional[str]]]) -> str:
    """Detects the predominant physical newline terminator (\\r\\n, \\r, or \\n) across strings or line sequences.

    Performs a majority vote across all line endings present in the provided sources.
    Accepts arbitrary string arguments, iterables/sequences of strings, or a mix of both.
    In the event of an exact tie among non-zero counts, the tie-breaking priority order
    is LF (\\n), CRLF (\\r\\n), then lone CR (\\r). If no physical newlines are present,
    defaults to '\\n'.

    Args:
        *sources: One or more text strings, or collections/iterables of text strings to probe.

    Returns:
        '\\n' if LF is predominant, '\\r\\n' if CRLF is predominant, otherwise '\\r'.
    """
    crlf_count = 0
    cr_count = 0
    lf_count = 0

    for s in _iter_sources(sources):
        c = s.count("\r\n")
        crlf_count += c
        cr_count += s.count("\r") - c
        lf_count += s.count("\n") - c

    if crlf_count == 0 and cr_count == 0 and lf_count == 0:
        return "\n"

    # Strict majority vote
    if lf_count > crlf_count and lf_count > cr_count:
        return "\n"
    if crlf_count > lf_count and crlf_count > cr_count:
        return "\r\n"
    if cr_count > lf_count and cr_count > crlf_count:
        return "\r"

    # Deterministic tie-breaking priority among non-zero counts: LF -> CRLF -> lone CR
    max_count = max(lf_count, crlf_count, cr_count)
    if lf_count == max_count:
        return "\n"
    if crlf_count == max_count:
        return "\r\n"
    return "\r"


def resolve_safe_unit_file_path(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    allowed_suffixes: Sequence[str] = (".py", ".ipynb"),
) -> Optional[Path]:
    """Safely resolves and validates a unit's file path on disk within repo_root or CWD."""
    raw_file = unit_file_str(unit)
    f_raw = normalize_path_string(raw_file, strip_anchor=True)
    if not f_raw:
        return None

    file_path = Path(f_raw)
    norm_suffixes = tuple(s.lower() for s in allowed_suffixes)
    if file_path.suffix.lower() not in norm_suffixes:
        return None

    if repo_root:
        effective_root = Path(repo_root).resolve()
        if effective_root.is_file():
            effective_root = effective_root.parent
        target_root = effective_root
        if not file_path.is_absolute():
            file_path = effective_root / file_path
    else:
        target_root = Path.cwd().resolve()
        if not file_path.is_absolute():
            file_path = target_root / file_path

    try:
        # Check that neither the file itself nor any unresolved parent directory below
        # target_root is a symlink.
        if file_path.is_symlink():
            return None
        for parent in file_path.parents:
            if parent == target_root:
                break
            if parent.is_symlink():
                resolved_parent_dir = parent.parent.resolve()
                if (
                    resolved_parent_dir == target_root
                    or target_root in resolved_parent_dir.parents
                ):
                    return None
                resolved_self = parent.resolve()
                if (
                    resolved_self == target_root
                    or resolved_self in target_root.parents
                ):
                    continue
                return None

        resolved_file = file_path.resolve()
        if not resolved_file.is_file():
            return None
        resolved_file.relative_to(target_root)
        return resolved_file
    except (OSError, RuntimeError, ValueError):
        return None


_resolve_safe_unit_file_path = resolve_safe_unit_file_path


def _ensure_trailing_lf(line: str) -> str:
    """Ensures line ends with '\\n', converting lone trailing '\\r' to '\\n' while preserving '\\r\\n'."""
    if line.endswith("\n"):
        return line
    if line.endswith("\r"):
        return line[:-1] + "\n"
    return line + "\n"


_MAX_NOTEBOOK_CACHE_ENTRIES = 64
_NOTEBOOK_CACHE_LOCK = threading.Lock()
_NOTEBOOK_CELLS_CACHE: OrderedDict[
    Tuple[str, int, int], Tuple[Optional[Tuple[str, ...]], ...]
] = OrderedDict()


def clear_notebook_cache() -> None:
    """Clears the thread-safe bounded cache of parsed notebook code-cell lines."""
    with _NOTEBOOK_CACHE_LOCK:
        _NOTEBOOK_CELLS_CACHE.clear()


def _extract_cell_text_lines_tuple(cell: Any) -> Optional[Tuple[str, ...]]:
    """Extracts LF-terminated physical source lines for a single notebook cell object."""
    cell_text = extract_notebook_code_cell_text(cell)
    if cell_text is None:
        return None
    return tuple(ln.rstrip("\r\n") + "\n" for ln in split_source_lines(cell_text))


def _extract_notebook_cell_lines(nb_data: Any, cell_idx: int) -> Optional[List[str]]:
    """Extracts LF-terminated physical source lines for a 0-indexed notebook code cell, or None if invalid.

    Source fragments are joined per nbformat semantics before splitting on physical
    newlines (``\\r\\n``, ``\\r``, ``\\n``), and every line terminator is normalized to
    ``\\n`` to match universal-newline disk reads. Malformed cells (non-code cells,
    missing or non-string source fragments) return None so callers fall back to a
    placeholder rather than treating coerced values as real source.
    """
    if not isinstance(nb_data, dict):
        return None
    cells = nb_data.get("cells")
    if not isinstance(cells, list) or cell_idx < 0 or cell_idx >= len(cells):
        return None
    cell_lines = _extract_cell_text_lines_tuple(cells[cell_idx])
    return list(cell_lines) if cell_lines is not None else None


def read_notebook_cell_lines_cached(
    resolved_file: Path, cell_idx: int
) -> Optional[List[str]]:
    """Reads and extracts newline-terminated source lines for a 0-indexed Jupyter cell.

    Caches only extracted code-cell text tuples per ``(resolved_path, mtime_ns, size)``
    in a bounded thread-safe LRU cache to avoid repeatedly JSON-parsing notebooks while
    dropping heavy notebook output payloads. Fails closed to ``None`` on malformed JSON
    or deeply nested ``RecursionError`` inputs without swallowing ``MemoryError``.
    """
    if cell_idx < 0:
        return None
    try:
        st = resolved_file.stat()
    except OSError:
        return None
    cache_key = (str(resolved_file), st.st_mtime_ns, st.st_size)
    with _NOTEBOOK_CACHE_LOCK:
        cached = _NOTEBOOK_CELLS_CACHE.get(cache_key)
        if cached is not None:
            _NOTEBOOK_CELLS_CACHE.move_to_end(cache_key)
            if cell_idx >= len(cached) or cached[cell_idx] is None:
                return None
            cell_lines = cached[cell_idx]
            return list(cell_lines) if cell_lines is not None else None

    try:
        raw_nb_text = resolved_file.read_text(encoding="utf-8", errors="replace")
        nb_data = json.loads(raw_nb_text)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        RecursionError,
    ):
        return None
    if not isinstance(nb_data, dict):
        return None
    cells = nb_data.get("cells")
    if not isinstance(cells, list):
        return None
    parsed_cells = tuple(_extract_cell_text_lines_tuple(cell) for cell in cells)
    with _NOTEBOOK_CACHE_LOCK:
        _NOTEBOOK_CELLS_CACHE[cache_key] = parsed_cells
        _NOTEBOOK_CELLS_CACHE.move_to_end(cache_key)
        while len(_NOTEBOOK_CELLS_CACHE) > _MAX_NOTEBOOK_CACHE_ENTRIES:
            _NOTEBOOK_CELLS_CACHE.popitem(last=False)

    if cell_idx >= len(parsed_cells) or parsed_cells[cell_idx] is None:
        return None
    target_lines = parsed_cells[cell_idx]
    return list(target_lines) if target_lines is not None else None


class _InMemoryNotebookResolution(NamedTuple):
    """Validation and extracted-cell state for anchored in-memory notebook units."""

    is_invalid: bool
    cell_lines: Optional[List[str]]
    source_key: Optional[str]


def _has_malformed_notebook_anchor(unit_file: str) -> bool:
    """Returns True when unit_file has an `.ipynb#<fragment>` anchor suffix that is not a valid 1-based cell anchor."""
    raw = str(unit_file or "").rstrip("\r\n")
    if "#" not in raw:
        return False
    cleaned = raw.replace("\x00", "").replace("\\", "/")
    cleaned_lower = cleaned.lower()
    if cleaned_lower.endswith(".ipynb"):
        return False
    if cleaned_lower.endswith(".py"):
        basename = cleaned_lower.rsplit("/", 1)[-1]
        return ".ipynb#" in basename and parse_notebook_cell_anchor(raw) is None
    return ".ipynb#" in cleaned_lower and parse_notebook_cell_anchor(raw) is None


# INVARIANT (EXPLICIT BUFFER FORMAT — NO STRING/AST SNIFFING):
# Format ambiguity between plain-Python cell source and raw nbformat JSON is
# resolved at the API boundary via ``unit["buffer_format"]`` (``"python"`` or
# ``"nbformat"``). Do NOT re-introduce heuristic substring sniffing
# (e.g., ``'"cells"' in raw_text``) or AST-parse fallback guessing on corrupt
# ``source_text`` / ``source_lines`` strings.
def _inspect_nbformat_buffer(
    raw_text: Any,
    cell_idx: int,
    *,
    strict_json_object: bool,
    force_nbformat: bool = False,
) -> Tuple[bool, Optional[List[str]]]:
    """Checks whether raw_text is an nbformat JSON payload and extracts cell_idx lines.

    Returns ``(is_nbformat_payload, extracted_lines)``. When ``force_nbformat`` is
    True (explicit ``buffer_format="nbformat"``), any non-JSON or malformed payload
    fails closed. When ``strict_json_object`` is True (for raw ``file_source`` backing
    an ``.ipynb`` path), any ``{...}`` object payload that fails JSON decoding or lacks
    valid nbformat cell structure fails closed.
    """
    if not isinstance(raw_text, str):
        return (True, None) if force_nbformat else (False, None)
    if not force_nbformat and not raw_text.lstrip().startswith("{"):
        return False, None
    try:
        nb_data = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
        return (True, None) if (force_nbformat or strict_json_object) else (False, None)
    if isinstance(nb_data, dict) and ("cells" in nb_data or "nbformat" in nb_data):
        return True, _extract_notebook_cell_lines(nb_data, cell_idx)
    if force_nbformat or strict_json_object:
        return True, None
    return False, None


def _nb_resolution_from_buffer(
    raw_text: Any,
    cell_idx: int,
    source_key: str,
    *,
    strict_json_object: bool,
    force_nbformat: bool = False,
) -> Optional[_InMemoryNotebookResolution]:
    """Returns an _InMemoryNotebookResolution if raw_text is an nbformat payload, else None."""
    is_nb, nb_lines = _inspect_nbformat_buffer(
        raw_text,
        cell_idx,
        strict_json_object=strict_json_object,
        force_nbformat=force_nbformat,
    )
    if not is_nb:
        return None
    if nb_lines is None:
        return _InMemoryNotebookResolution(
            is_invalid=True, cell_lines=None, source_key=None
        )
    return _InMemoryNotebookResolution(
        is_invalid=False, cell_lines=nb_lines, source_key=source_key
    )


def resolve_in_memory_notebook_unit(
    unit: Mapping[str, Any],
    *,
    is_sliced_lines_fn: Optional[Callable[[Dict[str, Any], Sequence[str]], bool]] = None,
) -> _InMemoryNotebookResolution:
    """Validates notebook anchor syntax and active in-memory nbformat JSON buffers on unit.

    Malformed ``.ipynb#<fragment>`` anchors fail closed even when in-memory source fields
    are populated. Valid ``#cell_N`` anchors honor explicit ``unit["buffer_format"]``
    (``"python"`` or ``"nbformat"``) and otherwise resolve strictly in
    ``source_lines > source_text > file_source`` order.
    """
    raw_fmt = unit.get("buffer_format")
    norm_fmt: Optional[str] = None
    if raw_fmt is not None:
        if not isinstance(raw_fmt, str) or raw_fmt.strip().lower() not in (
            "python",
            "nbformat",
        ):
            return _InMemoryNotebookResolution(
                is_invalid=True, cell_lines=None, source_key=None
            )
        norm_fmt = raw_fmt.strip().lower()

    u_file = unit_file_str(unit)
    if _has_malformed_notebook_anchor(u_file):
        return _InMemoryNotebookResolution(
            is_invalid=True, cell_lines=None, source_key=None
        )
    parsed_nb = parse_notebook_cell_anchor(u_file)
    if parsed_nb is None or norm_fmt == "python":
        return _InMemoryNotebookResolution(
            is_invalid=False, cell_lines=None, source_key=None
        )
    cell_idx = parsed_nb[1]
    force_nb = norm_fmt == "nbformat"

    raw_src_lines = unit.get("source_lines")
    if (
        raw_src_lines is not None
        and is_non_string_sequence(raw_src_lines)
        and raw_src_lines
        and all(isinstance(ln, str) for ln in raw_src_lines)
    ):
        sl_res = _nb_resolution_from_buffer(
            "".join(raw_src_lines),
            cell_idx,
            "source_lines",
            strict_json_object=False,
            force_nbformat=force_nb,
        )
        if sl_res is not None:
            return sl_res
        slice_check = is_sliced_lines_fn or is_sliced_unit_source_lines
        unit_dict = unit if isinstance(unit, dict) else dict(unit)
        if not slice_check(unit_dict, raw_src_lines):
            return _InMemoryNotebookResolution(
                is_invalid=False, cell_lines=None, source_key=None
            )

    raw_src_text = unit.get("source_text")
    if isinstance(raw_src_text, str) and raw_src_text.strip():
        st_res = _nb_resolution_from_buffer(
            raw_src_text,
            cell_idx,
            "source_text",
            strict_json_object=False,
            force_nbformat=force_nb,
        )
        if st_res is not None:
            return st_res
        return _InMemoryNotebookResolution(
            is_invalid=False, cell_lines=None, source_key=None
        )

    fs_res = _nb_resolution_from_buffer(
        unit.get("file_source"),
        cell_idx,
        "file_source",
        strict_json_object=True,
        force_nbformat=force_nb and unit.get("file_source") is not None,
    )
    if fs_res is not None:
        return fs_res
    return _InMemoryNotebookResolution(
        is_invalid=False, cell_lines=None, source_key=None
    )


class _InMemoryFullTextResolution(NamedTuple):
    """Result of resolving a unit's full in-memory module or notebook-cell text."""

    is_invalid: bool
    text: Optional[str]
    blocked_by_lines: bool


def _resolve_in_memory_text_fallback(
    unit: Mapping[str, Any],
    nb_state: _InMemoryNotebookResolution,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Resolves (raw_text, nb_cell_lines) from source_text ahead of file_source."""
    for field_key in ("source_text", "file_source"):
        raw_text = unit.get(field_key)
        if isinstance(raw_text, str) and raw_text.strip():
            if nb_state.cell_lines is not None and nb_state.source_key == field_key:
                return raw_text, list(nb_state.cell_lines)
            return raw_text, None
    return None, None


def resolve_in_memory_unit_full_text(
    unit: Dict[str, Any],
    *,
    is_sliced_lines_fn: Optional[Callable[[Dict[str, Any], Sequence[str]], bool]] = None,
) -> _InMemoryFullTextResolution:
    """Resolves full in-memory module or notebook-cell text under source_lines > source_text > file_source."""
    nb_state = resolve_in_memory_notebook_unit(
        unit, is_sliced_lines_fn=is_sliced_lines_fn
    )
    if nb_state.is_invalid:
        return _InMemoryFullTextResolution(
            is_invalid=True, text=None, blocked_by_lines=False
        )

    slice_check = is_sliced_lines_fn or is_sliced_unit_source_lines
    unit_lines = unit.get("source_lines")
    non_sliced_lines_text: Optional[str] = None
    has_blocking_lines = False
    if unit_lines is not None and is_non_string_sequence(unit_lines):
        if nb_state.cell_lines is not None and nb_state.source_key == "source_lines":
            non_sliced_lines_text = "".join(nb_state.cell_lines)
        elif not slice_check(unit, unit_lines):
            if all(isinstance(ln, str) for ln in unit_lines):
                non_sliced_lines_text = "".join(
                    _ensure_trailing_lf(ln) for ln in unit_lines
                )
            if unit_lines:
                has_blocking_lines = True

    if non_sliced_lines_text:
        return _InMemoryFullTextResolution(
            is_invalid=False, text=non_sliced_lines_text, blocked_by_lines=False
        )

    raw_text, nb_text_lines = _resolve_in_memory_text_fallback(unit, nb_state)
    if raw_text is not None:
        return _InMemoryFullTextResolution(
            is_invalid=False,
            text="".join(nb_text_lines) if nb_text_lines is not None else raw_text,
            blocked_by_lines=False,
        )

    if non_sliced_lines_text is not None:
        return _InMemoryFullTextResolution(
            is_invalid=False, text=non_sliced_lines_text, blocked_by_lines=True
        )
    return _InMemoryFullTextResolution(
        is_invalid=False, text=None, blocked_by_lines=has_blocking_lines
    )


def extract_unit_source_lines_with_state(
    unit: Dict[str, Any], repo_root: Optional[str] = None
) -> Tuple[List[str], SourceLinesOrigin]:
    """Reads raw source code lines for a unit and returns ``(lines, origin)``.

    Precedence of source fields:
    1. ``source_lines`` (pre-sliced excerpt or full module/cell buffer).
    2. ``source_text`` (full module or notebook cell text).
    3. ``file_source`` (full file buffer or nbformat JSON notebook).
    4. ``file`` on disk (resolved safely within ``repo_root`` or CWD).

    Notebook anchor handling:
    Malformed ``.ipynb#<fragment>`` anchors fail closed to ``SourceLinesOrigin.PLACEHOLDER``
    (``[]``) even when in-memory buffers are provided. When an in-memory buffer contains
    nbformat JSON, ``#cell_N`` is validated against the parsed notebook cells (failing
    closed on out-of-range, non-code, or corrupt JSON) and the code cell's Python lines
    are sliced. Plain-Python in-memory buffers (``source_lines``, or non-JSON
    ``source_text`` / ``file_source``) paired with a syntactically valid ``#cell_N``
    anchor are treated as authoritative unsaved cell buffers and override disk
    ``.ipynb`` anchor lookup.

    ``origin`` is a :class:`SourceLinesOrigin` indicating whether the returned lines
    came from a pre-sliced ``source_lines`` excerpt (``PRE_SLICED``), a full-file
    in-memory buffer (``FULL_BUFFER``), disk (``DISK``), or missing/unreadable source
    (``PLACEHOLDER``). Under ``PLACEHOLDER``, unavailable notebook cell anchors return
    ``[]`` while other missing-source paths return a single synthetic comment line;
    callers should query ``origin.is_pre_sliced``, ``origin.is_placeholder``, or
    ``origin.has_real_source`` explicitly.
    """
    s_d, e_d = resolve_unit_line_bounds(unit)
    n_d = str(unit.get("name") or "unit")
    placeholder = [f"# Source for {n_d} lines {s_d}-{e_d}\n"]

    nb_state = resolve_in_memory_notebook_unit(unit)
    if nb_state.is_invalid:
        return [], SourceLinesOrigin.PLACEHOLDER

    source_lines = unit.get("source_lines")
    from_source_text = False
    if nb_state.source_key == "source_lines":
        if not nb_state.cell_lines:
            return [], SourceLinesOrigin.PLACEHOLDER
        source_lines = nb_state.cell_lines
        from_source_text = True
    elif source_lines is not None and not is_non_string_sequence(source_lines):
        logger.warning(
            "Rejecting non-sequence source_lines entry for unit %r",
            n_d,
        )
        return placeholder, SourceLinesOrigin.PLACEHOLDER

    raw_src_text, nb_text_lines = _resolve_in_memory_text_fallback(unit, nb_state)
    from_nb_buffer = nb_state.source_key == "source_lines"
    if not source_lines and raw_src_text is not None:
        if nb_text_lines is not None:
            if not nb_text_lines:
                return [], SourceLinesOrigin.PLACEHOLDER
            source_lines = nb_text_lines
            from_nb_buffer = True
        else:
            source_lines = split_source_lines(raw_src_text)
        from_source_text = True

    if source_lines is not None and is_non_string_sequence(source_lines):
        if not source_lines:
            return placeholder, SourceLinesOrigin.PLACEHOLDER
        if not all(isinstance(ln, str) for ln in source_lines):
            logger.warning(
                "Rejecting non-string source_lines entry for unit %r",
                n_d,
            )
            return placeholder, SourceLinesOrigin.PLACEHOLDER
        if not from_source_text and is_sliced_unit_source_lines(unit, source_lines):
            return (
                [_ensure_trailing_lf(ln) for ln in source_lines],
                SourceLinesOrigin.PRE_SLICED,
            )
        start = max(1, s_d)
        end = min(len(source_lines), e_d)
        sliced_buf = [source_lines[idx] for idx in range(start - 1, end)] if start - 1 < end else []
        if not sliced_buf:
            return ([] if from_nb_buffer else placeholder), SourceLinesOrigin.PLACEHOLDER
        return (
            [_ensure_trailing_lf(ln) for ln in sliced_buf],
            SourceLinesOrigin.FULL_BUFFER,
        )

    resolved_file = resolve_safe_unit_file_path(unit, repo_root=repo_root)
    if resolved_file is None:
        return placeholder, SourceLinesOrigin.PLACEHOLDER

    try:
        is_cell_anchor = False
        cell_idx = -1
        u_file = unit_file_str(unit)
        if resolved_file.suffix.lower() == ".ipynb" and "#" in u_file:
            parsed_nb = parse_notebook_cell_anchor(u_file)
            if parsed_nb is not None:
                cell_idx = parsed_nb[1]
                is_cell_anchor = True
            elif _has_malformed_notebook_anchor(u_file):
                return [], SourceLinesOrigin.PLACEHOLDER

        if resolved_file.suffix.lower() == ".ipynb" and is_cell_anchor:
            cell_lines = read_notebook_cell_lines_cached(resolved_file, cell_idx)
            if cell_lines is None:
                return [], SourceLinesOrigin.PLACEHOLDER
            all_lines = cell_lines
        else:
            with open(resolved_file, "r", encoding="utf-8", errors="replace") as fh:
                all_lines = fh.readlines()
        start = max(1, s_d)
        end = min(len(all_lines), e_d)
        sliced_disk = all_lines[start - 1 : end]
        if not sliced_disk:
            return ([] if is_cell_anchor else placeholder), SourceLinesOrigin.PLACEHOLDER
        return (
            [_ensure_trailing_lf(ln) for ln in sliced_disk],
            SourceLinesOrigin.DISK,
        )
    except OSError:
        return (
            [f"# Unable to read {unit_file_str(unit)}\n"],
            SourceLinesOrigin.PLACEHOLDER,
        )


def extract_unit_source_code(unit: Dict[str, Any], repo_root: Optional[str] = None) -> List[str]:
    """Reads raw source code lines for a given unit from provided lines or disk."""
    return extract_unit_source_lines_with_state(unit, repo_root=repo_root)[0]


def with_unit_source_text(
    unit: Dict[str, Any], source_text: Optional[str]
) -> Dict[str, Any]:
    """Returns unit with ``source_text`` attached when neither non-whitespace ``source_text`` nor ``source_lines`` is set."""
    cur_text = unit.get("source_text")
    cur_lines = unit.get("source_lines")
    if (
        isinstance(source_text, str)
        and source_text.strip()
        and (not isinstance(cur_text, str) or not cur_text.strip())
        and not cur_lines
    ):
        return {**unit, "source_text": source_text}
    return unit
