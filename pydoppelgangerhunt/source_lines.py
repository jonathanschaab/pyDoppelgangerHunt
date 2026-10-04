"""Physical source line splitting, physical newline counting, and line-ending detection utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union, overload

_PHYSICAL_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")

__all__ = [
    "_PHYSICAL_LINE_RE",
    "count_physical_newlines",
    "detect_line_ending",
    "parse_unit_coord",
    "split_source_lines",
]


@overload
def parse_unit_coord(unit: Dict[str, Any], key: str) -> int:
    ...


@overload
def parse_unit_coord(unit: Dict[str, Any], key: str, default: int) -> int:
    ...


@overload
def parse_unit_coord(unit: Dict[str, Any], key: str, default: None) -> Optional[int]:
    ...


@overload
def parse_unit_coord(
    unit: Dict[str, Any], key: str, default: Optional[int]
) -> Optional[int]:
    ...


def parse_unit_coord(
    unit: Dict[str, Any], key: str, default: Optional[int] = 1
) -> Optional[int]:
    """Extracts and parses an integer coordinate from a unit dictionary.

    Safely handles integers, numeric strings, and colon-delimited coordinate strings
    (e.g., '12:0' or '8:0'). For colon-formatted coordinates emitted by external linters
    or diagnostics (where the suffix denotes a sub-column or character index), the primary
    leading coordinate prefix before the colon is parsed as the integer value.
    If the coordinate value is missing or empty, falls back to default.
    Raises ValueError for non-numeric strings or unconvertible/overflow values.
    """
    val = unit.get(key)
    if val is None:
        return default
    if isinstance(val, str):
        val = val.split(":", 1)[0].strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            try:
                return int(float(val))
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Invalid coordinate {val!r}: cannot convert to integer"
                ) from exc
    try:
        return int(val)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid coordinate {val!r}: cannot convert to integer"
        ) from exc


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
