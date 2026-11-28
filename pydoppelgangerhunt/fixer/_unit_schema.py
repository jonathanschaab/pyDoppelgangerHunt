"""UnitSchema definition, runtime field validation, and coordinate checks."""

from __future__ import annotations

import ast
import logging
import os
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from pydoppelgangerhunt.config import normalize_non_suppressing_ctx_managers
from pydoppelgangerhunt.source_lines import (
    is_non_string_sequence,
    parse_unit_coord,
    resolve_safe_unit_file_path,
)
from pydoppelgangerhunt.unit_boundary import (
    END_COL_KEYS as _END_COL_KEYS,
    END_LINE_KEYS as _END_LINE_KEYS,
    START_COL_KEYS as _START_COL_KEYS,
    START_LINE_KEYS as _START_LINE_KEYS,
)

logger = logging.getLogger(__name__)

class UnitSchema(TypedDict, total=False):
    """Documented schema for AST unit dictionaries across parser, matcher, and fixer.

    Source Resolution Precedence:
        When extracting unit source lines (`extract_unit_source_lines_with_state`),
        fields are consulted in the following order:
        1. ``source_lines`` (sliced directly when ``source_lines_is_sliced`` is True or
           matches the unit line span; otherwise indexed by ``start``..``end``).
        2. ``source_text`` (full module text split into physical lines).
        3. ``file_source`` (fallback full module text split into physical lines).
        4. ``file`` on disk (resolved relative to ``repo_root``).
        5. ``source_code`` (used by ``inspect_single_unit_scope`` when no full-module
           source is available, or validated against the backing module/tree for
           staleness when both are present).

    Notebook Anchors & Authoritative Unsaved Cell Buffers:
        Malformed ``.ipynb#<fragment>`` anchors fail closed even when in-memory source
        fields are populated. Valid ``#cell_N`` anchors validate against any in-memory
        nbformat JSON in ``file_source``/``source_text``/``source_lines`` (extracting the
        target code cell's Python lines when no plain-Python cell buffer overrides them).
        Plain-Python in-memory buffers (``source_lines``, or non-JSON ``source_text`` /
        ``file_source``) alongside a valid ``#cell_N`` anchor act as authoritative unsaved
        cell buffers overriding disk ``.ipynb`` anchor lookup.

    Schema Validation:
        ``validate_unit_schema()`` is invoked automatically only at the high-level
        refactoring entrypoints (``generate_refactoring_patch``,
        ``synthesize_shared_helper_code``, and ``synthesize_refactoring_suggestion``);
        lower-level helpers such as ``inspect_single_unit_scope``,
        ``extract_unit_source_lines_with_state``, and ``analyze_unit_variable_scope``
        accept lightweight unit dicts directly without running
        ``validate_unit_schema()`` unless the caller invokes it explicitly.

    Attributes:
        file: Relative or absolute source file path (may include ``#cell_N`` for notebooks).
        name: Qualified or simple name of the unit (e.g., ``"ClassName.method"`` or ``"fn:block"``).
        kind: Structural unit kind (e.g., ``"function"``, ``"method"``, ``"closure"``,
            ``"compound_block"``, ``"sliding_window"``, ``"clause_branch"``,
            ``"comprehension"``, ``"complex_expr"``, ``"class"``, ``"data_table"``).
        type: Legacy alias for ``kind`` in report payloads.
        start: 1-based inclusive start line number (or ``"line:col"`` string).
        end: 1-based inclusive end line number (or ``"line:col"`` string).
        lineno: Alias for ``start`` on AST-derived unit payloads.
        end_lineno: Alias for ``end`` on AST-derived unit payloads.
        start_col: 0-based UTF-8 byte start column offset on ``start`` line.
        end_col: 0-based UTF-8 byte end column offset on ``end`` line.
        col_offset: Alias for ``start_col``.
        start_col_offset: Alias for ``start_col``.
        end_col_offset: Alias for ``end_col``.
        docstring_end: Optional 1-based end line of the unit's leading docstring.
        is_subroutine: Explicit boolean override indicating whether the unit is an
            extracted subroutine block rather than a whole callable/class.
        source_code: Pre-sliced source text of the unit itself.
        source_text: Complete in-memory source text of the containing module.
        file_source: Fallback complete in-memory source text of the containing module.
        source_lines: Sequence of source lines (either pre-sliced excerpt or full module).
        source_lines_is_sliced: Explicit boolean disambiguating whether ``source_lines``
            is already sliced to the unit's line boundaries (``True``) or represents
            the full module buffer (``False``).
        source_lines_is_col_sliced: Explicit boolean disambiguating whether ``source_lines``
            is already trimmed to ``start_col``..``end_col`` (``True``) or still requires
            column slicing (``False``).
        buffer_format: Optional explicit format for in-memory notebook buffers
            (``"python"`` for extracted plain-Python cell source or ``"nbformat"`` for
            raw nbformat JSON), removing format-guessing ambiguity on ``.ipynb#cell_N`` units.
        inputs: Ordered list of live-in variable names required by the unit.
        outputs: Ordered list of live-out variable names exported by the unit.
        precomputed_outputs: Explicit caller-supplied output variable list taking
            precedence over ``outputs``.
        scope: Precomputed scope analysis dictionary from ``inspect_single_unit_scope``.
        receiver_param: Receiver parameter name (e.g., ``"self"`` or ``"cls"``).
        receiver_kind: Receiver binding kind (``"instance"``, ``"class"``, or ``"module"``).
        self_param: Instance receiver parameter name (legacy alias for ``receiver_param``
            on instance methods).
        cls_param: Class receiver parameter name (legacy alias for ``receiver_param``
            on classmethods).
        enclosing_class: Name of the enclosing class when the unit is a method or nested block.
        enclosing_class_start: 1-based start line of the enclosing class definition.
        enclosing_is_async: Whether the immediately enclosing function is ``async def``.
        enclosing_is_classmethod: Whether the immediately enclosing method is decorated
            with ``@classmethod``.
        enclosing_is_staticmethod: Whether the immediately enclosing method is decorated
            with ``@staticmethod``.
        enclosing_function: Name of the immediately enclosing function.
        is_static: Whether the unit is decorated with ``@staticmethod``.
        is_class_method: Whether the unit is decorated with ``@classmethod``.
        is_async: Whether the unit itself is an async function or contains top-level async constructs.
        has_yield: Whether the unit yields values (generator function or generator subroutine).
        has_await: Whether the unit contains ``await`` expressions.
        closure_strictness: Per-unit closure strictness mode override
            (``"strict"``, ``"pedantic"``, or ``"lenient"``).
        non_suppressing_ctx_managers: Optional sequence or set of context manager
            names/paths configured to be treated as non-suppressing in ``with``
            statements. Unlike ``pyproject.toml`` /
            :func:`normalize_non_suppressing_ctx_managers` (which also accept a
            comma-separated string), ``UnitSchema`` intentionally rejects bare ``str``
            values on unit dicts to avoid silently iterating characters.
        allow_snippet_only: Explicit boolean opt-in permitting subroutine, closure,
            comprehension, or complex-expression scope analysis when only isolated
            snippet ``source_code`` / ``source_lines`` is supplied without a backing
            module ``tree``, ``source_text``, or ``file``. Defaults to ``False`` (fail-closed).
        token_count: Number of normalized AST tokens in the unit.
        tokens: Legacy alias for ``token_count``.
        norm_tokens: Normalized AST token sequence emitted by the parser.
        lines: Total line span count emitted by the parser.
        sloc: Source lines of code count.
        complexity: McCabe cyclomatic complexity of the unit.
        shingles: Normalized AST k-shingle collection emitted by the parser (a set of
            token tuples; order-insensitive, so sequences and sets are both accepted).
        vector: Characteristic token/shingle Counter or vector emitted by the parser.
        calls: Sequence of called function/method names extracted by the parser.
        structural_hash: Exact structural AST digest emitted by the parser.
        ast_node: Optional in-memory AST node reference.
    """

    file: Optional[Union[str, os.PathLike[str]]]
    name: Optional[str]
    kind: Optional[str]
    type: Optional[str]
    start: Union[int, str, None]
    end: Union[int, str, None]
    lineno: Union[int, str, None]
    end_lineno: Union[int, str, None]
    start_col: Union[int, str, None]
    end_col: Union[int, str, None]
    col_offset: Union[int, str, None]
    start_col_offset: Union[int, str, None]
    end_col_offset: Union[int, str, None]
    docstring_end: Optional[int]
    is_subroutine: Optional[bool]
    source_code: Optional[str]
    source_text: Optional[str]
    file_source: Optional[str]
    source_lines: Optional[Sequence[str]]
    source_lines_is_sliced: Optional[bool]
    source_lines_is_col_sliced: Optional[bool]
    inputs: Sequence[str]
    outputs: Sequence[str]
    precomputed_outputs: Optional[Sequence[str]]
    scope: Dict[str, Any]
    receiver_param: Optional[str]
    receiver_kind: Optional[str]
    self_param: Optional[str]
    cls_param: Optional[str]
    enclosing_class: Optional[str]
    enclosing_class_start: Optional[int]
    enclosing_is_async: Optional[bool]
    enclosing_is_classmethod: Optional[bool]
    enclosing_is_staticmethod: Optional[bool]
    enclosing_function: Optional[str]
    is_static: Optional[bool]
    is_class_method: Optional[bool]
    is_async: Optional[bool]
    has_yield: Optional[bool]
    has_await: Optional[bool]
    closure_strictness: Optional[str]
    non_suppressing_ctx_managers: Optional[Union[Sequence[str], AbstractSet[str]]]
    allow_snippet_only: Optional[bool]
    buffer_format: Optional[str]
    token_count: Optional[int]
    tokens: Optional[Union[int, Sequence[str]]]
    norm_tokens: Optional[Sequence[str]]
    lines: Optional[int]
    sloc: Optional[int]
    complexity: Optional[int]
    shingles: Optional[Union[Sequence[Any], AbstractSet[Any]]]
    vector: Optional[Any]
    calls: Optional[Sequence[str]]
    structural_hash: Optional[str]
    ast_node: Optional[ast.AST]


UnitDict = Dict[str, Any]


_UNIT_SCHEMA_KEYS: frozenset[str] = frozenset(UnitSchema.__annotations__.keys())


_UNIT_BOOL_KEYS: frozenset[str] = frozenset(
    {
        "is_subroutine",
        "source_lines_is_sliced",
        "source_lines_is_col_sliced",
        "enclosing_is_async",
        "enclosing_is_classmethod",
        "enclosing_is_staticmethod",
        "is_static",
        "is_class_method",
        "is_async",
        "has_yield",
        "has_await",
        "allow_snippet_only",
    }
)


_UNIT_STR_KEYS: frozenset[str] = frozenset(
    {
        "file",
        "name",
        "kind",
        "type",
        "source_code",
        "source_text",
        "file_source",
        "receiver_param",
        "receiver_kind",
        "self_param",
        "cls_param",
        "enclosing_class",
        "enclosing_function",
        "closure_strictness",
        "buffer_format",
        "structural_hash",
    }
)


_UNIT_POSITIVE_INT_KEYS: frozenset[str] = frozenset(
    {
        "docstring_end",
        "enclosing_class_start",
    }
)


_UNIT_NONNEG_INT_KEYS: frozenset[str] = frozenset(
    {
        "token_count",
        "lines",
        "sloc",
        "complexity",
    }
)


_UNIT_LINE_COORD_KEYS: frozenset[str] = frozenset(
    {"start", "end", "lineno", "end_lineno"}
)


_UNIT_COL_COORD_KEYS: frozenset[str] = frozenset(
    {"start_col", "end_col", "col_offset", "start_col_offset", "end_col_offset"}
)


# Positionally consumed fields (helper parameter order, output pairing, token streams)
# require ordered sequences; unordered sets would make helper signatures nondeterministic.
_UNIT_ORDERED_STR_SEQ_KEYS: frozenset[str] = frozenset(
    {
        "source_lines",
        "inputs",
        "outputs",
        "precomputed_outputs",
        "norm_tokens",
        "calls",
    }
)


# Membership-only fields accept either ordered sequences or unordered sets.
_UNIT_STR_COLLECTION_KEYS: frozenset[str] = frozenset({"non_suppressing_ctx_managers"})


# Mutually aliased coordinate keys that must agree when more than one is supplied.
_UNIT_COORD_ALIAS_GROUPS: Tuple[Tuple[str, ...], ...] = (
    _START_LINE_KEYS,
    _END_LINE_KEYS,
    _START_COL_KEYS,
    _END_COL_KEYS,
)


_VALID_CLOSURE_STRICTNESS_VALUES: frozenset[str] = frozenset(
    {"strict", "pedantic", "lenient"}
)


_VALID_BUFFER_FORMAT_VALUES: frozenset[str] = frozenset({"python", "nbformat"})


_UNIT_ENUM_STR_VALIDATORS: Dict[str, frozenset[str]] = {
    "closure_strictness": _VALID_CLOSURE_STRICTNESS_VALUES,
    "buffer_format": _VALID_BUFFER_FORMAT_VALUES,
}


def _is_valid_ordered_str_sequence(val: Any) -> bool:
    """Returns True if val is an ordered non-string sequence of strings (sets are rejected)."""
    if not is_non_string_sequence(val):
        return False
    return all(isinstance(item, str) for item in val)


def _is_valid_str_collection(val: Any) -> bool:
    """Returns True if val is a non-string sequence or set of strings."""
    if isinstance(val, (str, bytes, bytearray, Mapping)) or not isinstance(
        val, (Sequence, AbstractSet)
    ):
        return False
    return all(isinstance(item, str) for item in val)


# (field keys, validator, error description) for string-collection UnitSchema fields.
_UNIT_STR_COLLECTION_RULES: Tuple[
    Tuple[frozenset[str], Callable[[Any], bool], str], ...
] = (
    (_UNIT_ORDERED_STR_SEQ_KEYS, _is_valid_ordered_str_sequence, "ordered sequence"),
    (_UNIT_STR_COLLECTION_KEYS, _is_valid_str_collection, "sequence or set"),
)


def _validate_int_field(key: str, val: Any, *, minimum: int) -> None:
    """Raises ValueError unless val is a non-bool int >= minimum."""
    if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
        qualifier = "positive" if minimum > 0 else "non-negative"
        raise ValueError(
            f"Expected {qualifier} int for unit field {key!r}, got {val!r}"
        )


def _validate_known_unit_field_value(key: str, val: Any) -> None:
    """Raises ValueError if a non-None value for a known UnitSchema key has an invalid type or domain."""
    if key in _UNIT_BOOL_KEYS:
        if not isinstance(val, bool):
            raise ValueError(f"Expected bool for unit field {key!r}, got {val!r}")
        return
    if key in _UNIT_STR_KEYS:
        if key == "file" and isinstance(val, os.PathLike):
            fspath = os.fspath(val)
            if not isinstance(fspath, str):
                raise ValueError(
                    f"Expected str or str PathLike for unit field {key!r}, got {val!r}"
                )
            return
        if not isinstance(val, str):
            raise ValueError(f"Expected str for unit field {key!r}, got {val!r}")
        allowed_values = _UNIT_ENUM_STR_VALIDATORS.get(key)
        if allowed_values is not None and val.strip().lower() not in allowed_values:
            raise ValueError(
                f"Invalid {key} value {val!r}; expected one of {sorted(allowed_values)}"
            )
        return
    if key in _UNIT_POSITIVE_INT_KEYS:
        _validate_int_field(key, val, minimum=1)
        return
    if key in _UNIT_NONNEG_INT_KEYS:
        _validate_int_field(key, val, minimum=0)
        return
    if key in _UNIT_LINE_COORD_KEYS or key in _UNIT_COL_COORD_KEYS:
        min_coord = 1 if key in _UNIT_LINE_COORD_KEYS else 0
        if isinstance(val, bool) or not isinstance(val, (int, str)):
            raise ValueError(
                f"Expected coordinate int or str for unit field {key!r}, got {val!r}"
            )
        try:
            parsed = parse_unit_coord({key: val}, key, default=None, strict=True)
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinate value for unit field {key!r}: {val!r}"
            ) from exc
        if parsed is None or parsed < min_coord:
            raise ValueError(
                f"Invalid coordinate value for unit field {key!r}: {val!r}"
            )
        return
    for keys, is_valid, description in _UNIT_STR_COLLECTION_RULES:
        if key in keys:
            if not is_valid(val):
                raise ValueError(
                    f"Expected {description} of str for unit field {key!r}, got {val!r}"
                )
            if key == "non_suppressing_ctx_managers":
                normalize_non_suppressing_ctx_managers(val)
            return
    if key == "tokens":
        is_nonneg_int = not isinstance(val, bool) and isinstance(val, int) and val >= 0
        if not (is_nonneg_int or _is_valid_ordered_str_sequence(val)):
            raise ValueError(
                f"Expected non-negative int or ordered sequence of str for 'tokens', got {val!r}"
            )
        return
    if key == "shingles":
        if isinstance(val, (str, bytes, bytearray, Mapping)) or not isinstance(
            val, (Sequence, AbstractSet)
        ):
            raise ValueError(f"Expected sequence or set for 'shingles', got {val!r}")
        return
    if key == "scope" and not isinstance(val, Mapping):
        raise ValueError(f"Expected mapping for 'scope', got {val!r}")
    if key == "ast_node" and not isinstance(val, ast.AST):
        raise ValueError(f"Expected ast.AST for 'ast_node', got {val!r}")


def _validate_coordinate_alias_agreement(unit: Mapping[str, Any]) -> None:
    """Raises ValueError when aliased coordinate keys (e.g. ``start`` / ``lineno``) disagree."""
    for group in _UNIT_COORD_ALIAS_GROUPS:
        parsed = {
            key: parse_unit_coord({key: unit[key]}, key, default=None, strict=True)
            for key in group
            if unit.get(key) is not None
        }
        if len(set(parsed.values())) > 1:
            raise ValueError(
                f"Conflicting coordinate aliases in unit dictionary: {parsed}"
            )


def _parse_first_unit_coord(
    unit: Mapping[str, Any], keys: Tuple[str, ...]
) -> Optional[int]:
    for key in keys:
        if unit.get(key) is not None:
            return parse_unit_coord({key: unit[key]}, key, default=None, strict=True)
    return None


def validate_unit_schema(
    unit: Mapping[str, Any], *, strict: bool = False
) -> Set[str]:
    """Validates a unit dictionary against documented UnitSchema keys and field types.

    Raises ``TypeError`` if ``unit`` is not a mapping, or ``ValueError`` if any
    known field has an invalid value type/range (or when ``strict=True`` and
    unrecognized keys are present). Returns the set of unrecognized keys not
    declared on ``UnitSchema``.

    Important:
        ``validate_unit_schema()`` is called automatically ONLY at the high-level
        refactoring entrypoints (``generate_refactoring_patch``,
        ``synthesize_shared_helper_code``, and ``synthesize_refactoring_suggestion``).
        Lower-level inspection and extraction helpers (``inspect_single_unit_scope``,
        ``extract_unit_source_lines_with_state``, ``analyze_unit_variable_scope``,
        etc.) do NOT invoke ``validate_unit_schema()`` automatically; call it
        explicitly before passing untrusted or external unit payloads into those
        lower-level helpers.

    Note:
        Positionally consumed fields (``source_lines``, ``inputs``, ``outputs``,
        ``precomputed_outputs``, ``norm_tokens``, ``calls``, and list-valued
        ``tokens``) must be ordered sequences; unordered sets are rejected because
        they make helper parameter order and output pairing nondeterministic.
        ``non_suppressing_ctx_managers`` accepts sequences or sets of strings on unit
        dicts and intentionally rejects bare ``str`` (whereas ``pyproject.toml`` and
        :func:`normalize_non_suppressing_ctx_managers` accept comma-separated strings).
        ``docstring_end`` and ``enclosing_class_start`` are 1-based line numbers and
        reject zero. Aliased coordinates (``start`` / ``lineno``, ``end`` /
        ``end_lineno``, ``start_col`` / ``col_offset`` / ``start_col_offset``, and
        ``end_col`` / ``end_col_offset``) must agree when more than one is supplied.
    """
    if not isinstance(unit, Mapping):
        raise TypeError(f"Expected a unit mapping, got {type(unit).__name__}")
    unknown = {str(k) for k in unit.keys() if k not in _UNIT_SCHEMA_KEYS}
    if strict and unknown:
        raise ValueError(
            f"Unrecognized keys in unit dictionary: {sorted(unknown)}"
        )
    for k, v in unit.items():
        if k in _UNIT_SCHEMA_KEYS and v is not None:
            _validate_known_unit_field_value(str(k), v)
    _validate_coordinate_alias_agreement(unit)
    start_raw = unit.get("start") if unit.get("start") is not None else unit.get("lineno")
    end_raw = unit.get("end") if unit.get("end") is not None else unit.get("end_lineno")
    s_val = (
        parse_unit_coord({"start": start_raw}, "start", default=None, strict=True)
        if start_raw is not None
        else 1
    )
    e_val = (
        parse_unit_coord({"end": end_raw}, "end", default=None, strict=True)
        if end_raw is not None
        else s_val
    )
    if s_val is not None and e_val is not None:
        if e_val < s_val:
            raise ValueError(
                f"Inverted line coordinates in unit dictionary: start={s_val} > end={e_val}"
            )
        if s_val == e_val:
            raw_s_col = _parse_first_unit_coord(unit, _START_COL_KEYS)
            e_col = _parse_first_unit_coord(unit, _END_COL_KEYS)
            s_col = 0 if raw_s_col is None else raw_s_col
            if e_col is not None and e_col <= s_col:
                if e_col < s_col:
                    raise ValueError(
                        f"Inverted column coordinates on line {s_val}: "
                        f"start_col={s_col} > end_col={e_col}"
                    )
                raise ValueError(
                    f"Zero-width column coordinates on line {s_val}: "
                    f"start_col={s_col} == end_col={e_col}"
                )
    return unknown


_parse_unit_coord = parse_unit_coord


_resolve_safe_unit_file_path = resolve_safe_unit_file_path


def is_valid_unit_coordinates(u: Any, strict: bool = False) -> bool:
    """Verifies that an AST unit dictionary has valid integer coordinates.

    Note (Behavior Change): When strict=False, a positive start coordinate
    (start > 0) is required, and inverted coordinates (end < start) are
    rejected. In strict mode (for code refactoring and patching), both start
    and end are required and must parse strictly to positive integers (rejecting
    floats, colons, and blank strings) with end >= start. End, start_col (or
    start_col_offset), and end_col (or end_col_offset) must also parse strictly
    if present, with end_col >= start_col if on the same line.
    """
    if not isinstance(u, dict):
        return False
    if "start" not in u or u.get("start") is None:
        return False
    if strict and ("end" not in u or u.get("end") is None):
        return False
    try:
        start_val = parse_unit_coord(u, "start", default=None, strict=strict)
        if start_val is None or start_val <= 0:
            return False

        end_val: Optional[int] = None
        if "end" in u and u.get("end") is not None:
            end_val = parse_unit_coord(u, "end", default=None, strict=strict)
            if end_val is None or end_val <= 0 or end_val < start_val:
                return False
        elif strict:
            return False

        def _parse_and_validate_cols(keys: Tuple[str, ...]) -> Optional[int]:
            first_val: Optional[int] = None
            for key in keys:
                if u.get(key) is not None:
                    c_val = parse_unit_coord(u, key, default=None, strict=strict)
                    if c_val is None or c_val < 0:
                        raise ValueError
                    if first_val is None:
                        first_val = c_val
            return first_val

        start_col = _parse_and_validate_cols(_START_COL_KEYS)
        end_col = _parse_and_validate_cols(_END_COL_KEYS)

        effective_end = end_val if end_val is not None else start_val
        if (
            effective_end is not None
            and start_col is not None
            and end_col is not None
            and start_val == effective_end
            and end_col < start_col
        ):
            return False

        return True
    except (ValueError, TypeError, KeyError):
        return False
