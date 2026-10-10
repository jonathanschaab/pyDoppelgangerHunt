"""Physical source line splitting, physical newline counting, and line-ending detection utilities."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, overload

from pydoppelgangerhunt.canonical_path import normalize_lexical_posix as normalize_path_string

logger = logging.getLogger(__name__)

_PHYSICAL_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")

__all__ = [
    "count_physical_newlines",
    "detect_line_ending",
    "is_sliced_unit_source_lines",
    "parse_unit_coord",
    "resolve_safe_unit_file_path",
    "resolve_unit_line_bounds",
    "split_source_lines",
]


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
    and floating point values are strictly rejected with ValueError.
    If the coordinate value is missing or empty (in non-strict mode), falls back to default.
    Raises ValueError for non-numeric strings or unconvertible/overflow values.
    """
    val = unit.get(key)
    if val is None:
        return default
    if strict:
        if isinstance(val, str):
            if ":" in val:
                raise ValueError(
                    f"Invalid coordinate {val!r}: colon format not allowed in strict mode"
                )
            if not val.strip():
                raise ValueError(
                    f"Invalid coordinate {val!r}: blank coordinate not allowed in strict mode"
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
            return val
        return _coerce_int_coord(val)

    if isinstance(val, str):
        val = val.split(":", 1)[0].strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            try:
                f_val = float(val)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert to integer"
                ) from exc
            if not f_val.is_integer():
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert non-integral float"
                ) from None
            return int(f_val)
    if isinstance(val, float):
        if not val.is_integer():
            raise ValueError(f"Invalid coordinate {val!r}: cannot convert non-integral float")
        return int(val)
    return _coerce_int_coord(val)


def _coerce_int_coord(val: Any) -> int:
    """Coerces coordinate value to int or raises ValueError on invalid/overflow inputs."""
    try:
        return int(val)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid coordinate {val!r}: cannot convert to integer") from exc


def is_sliced_unit_source_lines(
    unit: Dict[str, Any],
    source_lines: Optional[Sequence[str]] = None,
) -> bool:
    """Determines whether a unit's source_lines collection represents a pre-sliced excerpt."""
    is_sliced = unit.get("source_lines_is_sliced")
    if is_sliced is not None:
        return bool(is_sliced)
    lines = source_lines if source_lines is not None else unit.get("source_lines")
    if not lines or not isinstance(lines, (list, tuple)):
        return False
    s_d, e_d = resolve_unit_line_bounds(unit)
    expected_len = max(0, e_d - s_d + 1)
    if len(lines) == expected_len:
        return True
    return False


def resolve_unit_line_bounds(unit: Dict[str, Any]) -> Tuple[int, int]:
    """Extracts clamped, positive (start, end) line coordinates from a unit dictionary.

    Logs a DEBUG diagnostic whenever start is clamped to >= 1 or end is clamped to >= start.
    """
    raw_s = parse_unit_coord(unit, "start", default=1)
    raw_e = parse_unit_coord(unit, "end", default=raw_s)
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
    raw_file = str(unit.get("file") or "")
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
        repo_root_path = Path(repo_root) if repo_root else target_root
        for parent in file_path.parents:
            if parent in (target_root, repo_root_path):
                break
            if parent.is_symlink():
                if target_root in parent.parents or repo_root_path in parent.parents:
                    return None
                if parent.resolve() == target_root:
                    break
                return None

        resolved_file = file_path.resolve()
        if not resolved_file.is_file():
            return None
        resolved_file.relative_to(target_root)
        return resolved_file
    except (OSError, RuntimeError, ValueError):
        return None


_resolve_safe_unit_file_path = resolve_safe_unit_file_path
