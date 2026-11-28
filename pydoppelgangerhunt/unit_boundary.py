"""Single-boundary normalization for untrusted UnitSchema payloads.

# INVARIANT (PARSE ONCE AT BOUNDARY):
# Public entrypoints normalize untrusted ``UnitSchema`` mappings once into immutable
# typed objects (``NormalizedCoordinates``, ``NormalizedSourceBuffer``, ``NormalizedUnit``)
# via :func:`normalize_unit_boundary` and :func:`resolve_unit_source_buffer`.
# Downstream internal helpers must consume normalized coordinates (covering all line and
# column alias groups via ``START_LINE_KEYS``, ``END_LINE_KEYS``, ``START_COL_KEYS``,
# ``END_COL_KEYS``), ``os.fspath()``-resolved file paths, canonical buffer-origin state,
# and single-knob ``ResolvedClosurePolicy`` values instead of re-implementing alias
# precedence, inline-buffer sniffing, or PathLike coercion ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from pydoppelgangerhunt.canonical_path import parse_notebook_cell_anchor
from pydoppelgangerhunt.config import (
    ResolvedClosurePolicy,
    resolve_unit_closure_policy,
)
from pydoppelgangerhunt.source_lines import (
    SourceLinesOrigin,
    extract_unit_source_lines_with_state,
    is_non_string_sequence,
    parse_unit_coord,
    resolve_in_memory_notebook_unit,
    resolve_in_memory_unit_full_text,
    resolve_unit_line_bounds,
    unit_file_str,
)

__all__ = [
    "END_COL_KEYS",
    "END_LINE_KEYS",
    "NormalizedCoordinates",
    "NormalizedSourceBuffer",
    "NormalizedUnit",
    "START_COL_KEYS",
    "START_LINE_KEYS",
    "normalize_unit_boundary",
    "resolve_unit_source_buffer",
    "unit_has_inline_source_buffers",
]

START_LINE_KEYS: Tuple[str, ...] = ("start", "lineno")
END_LINE_KEYS: Tuple[str, ...] = ("end", "end_lineno")
START_COL_KEYS: Tuple[str, ...] = ("start_col", "start_col_offset", "col_offset")
END_COL_KEYS: Tuple[str, ...] = ("end_col", "end_col_offset")


class NormalizedCoordinates(NamedTuple):
    """Canonical 1-based line bounds and 0-based UTF-8 byte column bounds for a unit."""

    start: Optional[int]
    end: Optional[int]
    start_col: Optional[int]
    end_col: Optional[int]


class NormalizedSourceBuffer(NamedTuple):
    """Resolved source lines, provenance origin, optional full module text, and normalized path."""

    lines: List[str]
    origin: SourceLinesOrigin
    full_text: Optional[str]
    is_invalid_notebook: bool
    blocked_by_lines: bool
    file_path: str
    notebook_anchor: Optional[Tuple[str, int]]


@dataclass(frozen=True)
class NormalizedUnit:
    """Immutable boundary representation of an incoming UnitSchema dictionary."""

    file_path: str
    notebook_anchor: Optional[Tuple[str, int]]
    name: str
    kind: str
    coords: NormalizedCoordinates
    policy: ResolvedClosurePolicy
    source_lines_is_sliced: Optional[bool]
    source_lines_is_col_sliced: Optional[bool]
    allow_snippet_only: bool
    raw: Mapping[str, Any]
    source_code: str = ""
    is_subroutine_flag: Optional[bool] = None
    has_inline_buffers: bool = False
    has_malformed_coords: bool = False
    explicit_kind: str = ""
    buffer_format: Optional[str] = None


def unit_has_inline_source_buffers(
    unit: Mapping[str, Any],
    *,
    ignore_empty_source_lines: bool = False,
) -> bool:
    """Returns True when unit carries inline source_lines, source_text, or file_source."""
    raw_src_lines = unit.get("source_lines")
    has_src_lines = (
        raw_src_lines is not None
        and (
            not ignore_empty_source_lines
            or not is_non_string_sequence(raw_src_lines)
            or len(raw_src_lines) > 0
        )
    )
    return (
        has_src_lines
        or (isinstance(unit.get("source_text"), str) and bool(unit["source_text"].strip()))
        or (isinstance(unit.get("file_source"), str) and bool(unit["file_source"].strip()))
    )


def _safe_parse_optional_coord(
    unit_dict: Dict[str, Any], keys: Tuple[str, ...]
) -> Tuple[Optional[int], bool]:
    """Parses the first present coordinate alias, returning ``(value, had_error)``."""
    for candidate_key in keys:
        if candidate_key not in unit_dict or unit_dict.get(candidate_key) is None:
            continue
        try:
            val = parse_unit_coord(unit_dict, candidate_key, default=None)
        except (ValueError, TypeError):
            return None, True
        if val is not None:
            return val, False
    return None, False


def _parse_normalized_coords(
    unit_dict: Dict[str, Any],
) -> Tuple[NormalizedCoordinates, bool]:
    """Parses canonical start/end line and column coordinate aliases into NormalizedCoordinates."""
    parsed = [
        _safe_parse_optional_coord(unit_dict, keys)
        for keys in (START_LINE_KEYS, END_LINE_KEYS, START_COL_KEYS, END_COL_KEYS)
    ]
    coords = NormalizedCoordinates(
        start=parsed[0][0],
        end=parsed[1][0],
        start_col=parsed[2][0],
        end_col=parsed[3][0],
    )
    return coords, any(err for _, err in parsed)


def normalize_unit_boundary(
    unit: Mapping[str, Any],
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    default_strictness: str = "strict",
) -> NormalizedUnit:
    """Normalizes a UnitSchema mapping once at an API boundary into an immutable NormalizedUnit."""
    unit_dict: Dict[str, Any] = unit if isinstance(unit, dict) else dict(unit)
    file_path = unit_file_str(unit_dict)
    notebook_anchor = parse_notebook_cell_anchor(file_path) if file_path else None
    coords, has_malformed = _parse_normalized_coords(unit_dict)
    policy = resolve_unit_closure_policy(
        unit_dict,
        closure_strictness=closure_strictness,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        default=default_strictness,
    )
    explicit_kind = str(unit_dict.get("kind") or unit_dict.get("type") or "")
    raw_kind = explicit_kind or "function"
    raw_name = unit_dict.get("name") or ""
    raw_sliced = unit_dict.get("source_lines_is_sliced")
    raw_col_sliced = unit_dict.get("source_lines_is_col_sliced")
    raw_src_code = unit_dict.get("source_code")
    raw_sub = unit_dict.get("is_subroutine")
    raw_fmt = unit_dict.get("buffer_format")
    norm_fmt = raw_fmt.strip().lower() if isinstance(raw_fmt, str) else None
    return NormalizedUnit(
        file_path=file_path,
        notebook_anchor=notebook_anchor,
        name=str(raw_name),
        kind=raw_kind,
        coords=coords,
        policy=policy,
        source_lines_is_sliced=raw_sliced if isinstance(raw_sliced, bool) else None,
        source_lines_is_col_sliced=(
            raw_col_sliced if isinstance(raw_col_sliced, bool) else None
        ),
        allow_snippet_only=bool(unit_dict.get("allow_snippet_only", False)),
        raw=unit,
        source_code=raw_src_code.strip() if isinstance(raw_src_code, str) else "",
        is_subroutine_flag=bool(raw_sub) if raw_sub is not None else None,
        has_inline_buffers=unit_has_inline_source_buffers(unit_dict),
        has_malformed_coords=has_malformed,
        explicit_kind=explicit_kind,
        buffer_format=norm_fmt,
    )


def resolve_unit_source_buffer(
    unit: Mapping[str, Any],
    *,
    repo_root: Optional[str] = None,
    allow_disk_read: bool = True,
    include_lines: bool = True,
    is_sliced_lines_fn: Optional[Callable[[Dict[str, Any], Sequence[str]], bool]] = None,
) -> NormalizedSourceBuffer:
    """Resolves a unit's source lines, origin, and in-memory full text using canonical precedence."""
    unit_dict: Dict[str, Any] = unit if isinstance(unit, dict) else dict(unit)
    file_path = unit_file_str(unit_dict)
    stripped_file = file_path.strip()
    notebook_anchor = parse_notebook_cell_anchor(stripped_file) if stripped_file else None
    full_res = resolve_in_memory_unit_full_text(
        unit_dict,
        is_sliced_lines_fn=is_sliced_lines_fn,
    )
    if not include_lines:
        return NormalizedSourceBuffer(
            lines=[],
            origin=SourceLinesOrigin.PLACEHOLDER,
            full_text=full_res.text,
            is_invalid_notebook=full_res.is_invalid,
            blocked_by_lines=full_res.blocked_by_lines,
            file_path=file_path,
            notebook_anchor=notebook_anchor,
        )
    nb_state = resolve_in_memory_notebook_unit(unit_dict)
    has_inline = unit_has_inline_source_buffers(unit_dict)
    if not allow_disk_read and not nb_state.is_invalid and not has_inline:
        s_d, e_d = resolve_unit_line_bounds(unit_dict)
        n_d = str(unit_dict.get("name") or "unit")
        is_nb_file = stripped_file.lower().split("#", 1)[0].endswith(".ipynb")
        lines = [] if is_nb_file else [f"# Source for {n_d} lines {s_d}-{e_d}\n"]
        origin = SourceLinesOrigin.PLACEHOLDER
    else:
        lines, origin = extract_unit_source_lines_with_state(
            unit_dict,
            repo_root=repo_root,
        )
        if (
            origin is SourceLinesOrigin.DISK
            and stripped_file.lower().endswith(".ipynb")
            and notebook_anchor is None
        ):
            lines = []
            origin = SourceLinesOrigin.PLACEHOLDER
    return NormalizedSourceBuffer(
        lines=lines,
        origin=origin,
        full_text=full_res.text,
        is_invalid_notebook=full_res.is_invalid or nb_state.is_invalid,
        blocked_by_lines=full_res.blocked_by_lines,
        file_path=file_path,
        notebook_anchor=notebook_anchor,
    )
