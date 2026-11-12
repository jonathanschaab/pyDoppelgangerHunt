"""Configuration loading and initialization for pyDoppelgangerHunt."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Set as AbstractSet
import contextlib
import contextvars
from dataclasses import dataclass, field
import itertools
import logging
import os
from pathlib import Path
import sys
import threading
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.canonical_path import (
    normalize_lexical_posix,
    parse_notebook_cell_anchor,
)

logger = logging.getLogger(__name__)

_warned_closure_lock: threading.Lock = threading.Lock()
_MAX_WARNED_CLOSURE_EVENTS: int = 4096
_MAX_HEURISTIC_CTX_MANAGERS: int = 16384
_MAX_DEFAULT_THREAD_BUCKETS: int = 256
_warned_closure_strictness_values: OrderedDict[str, None] = OrderedDict()
_next_heuristic_session_id = itertools.count(1)
_heuristic_collector_tls: threading.local = threading.local()


@dataclass
class _HeuristicSessionBucket:
    """Bounded accumulator for unproven context managers recorded in a session or thread."""

    names: Set[str] = field(default_factory=set)
    overflow_count: int = 0
    defer_depth: int = 0
    warn_on_summary: bool = False

    def add(self, mgr_name: str) -> None:
        """Records mgr_name up to _MAX_HEURISTIC_CTX_MANAGERS, counting overflow occurrences beyond cap."""
        if mgr_name in self.names:
            return
        if len(self.names) < _MAX_HEURISTIC_CTX_MANAGERS:
            self.names.add(mgr_name)
        else:
            self.overflow_count += 1


_active_heuristic_sessions: Dict[int, _HeuristicSessionBucket] = {}
_default_thread_buckets: OrderedDict[int, _HeuristicSessionBucket] = OrderedDict()
_heuristic_session_stack: contextvars.ContextVar[Tuple[int, ...]] = (
    contextvars.ContextVar("_heuristic_session_stack", default=())
)


def _get_or_create_thread_bucket_locked() -> _HeuristicSessionBucket:
    """Returns the active session bucket for the current context or a bounded per-thread fallback bucket."""
    sess_stack = _heuristic_session_stack.get()
    if sess_stack:
        sid = sess_stack[-1]
        active = _active_heuristic_sessions.get(sid)
        if active is not None:
            return active
    tid = threading.get_ident()
    bucket = _default_thread_buckets.get(tid)
    if bucket is not None:
        _default_thread_buckets.move_to_end(tid)
        return bucket
    bucket = _HeuristicSessionBucket()
    _default_thread_buckets[tid] = bucket
    while len(_default_thread_buckets) > _MAX_DEFAULT_THREAD_BUCKETS:
        _default_thread_buckets.popitem(last=False)
    return bucket


_CONFIG_KEY_NAMES: Tuple[str, ...] = (
    "threshold",
    "min_lines",
    "exemptions",
    "exclude",
    "closure_strictness",
    "skip_pre_unit_closures",
    "strict_with_suppression",
    "non_suppressing_ctx_managers",
    "top",
)


class UnitClosureFlags(NamedTuple):
    """Resolved (skip_pre_unit_closures, strict_with_suppression) flags for a single unit."""

    skip_pre_unit_closures: bool
    strict_with_suppression: bool


class UnitClosureEndpointConfig(NamedTuple):
    """Resolved closure and context-manager allowlist flags for one endpoint of a clone pair."""

    skip_pre_unit_closures: bool
    strict_with_suppression: bool
    non_suppressing_ctx_managers: Tuple[str, ...] = ()


class PairClosureFlags(NamedTuple):
    """Resolved per-unit and pair-level closure and context-manager configuration."""

    unit1: UnitClosureEndpointConfig
    unit2: UnitClosureEndpointConfig
    effective_skip_pre_unit_closures: bool

    @property
    def effective_strict_with_suppression(self) -> bool:
        """Returns True if either endpoint requires pedantic strict_with_suppression."""
        return bool(
            self.unit1.strict_with_suppression or self.unit2.strict_with_suppression
        )


def should_warn_closure_event(event_key: str) -> bool:
    """Returns True the first time *event_key* is seen since cache reset."""
    with _warned_closure_lock:
        if event_key in _warned_closure_strictness_values:
            _warned_closure_strictness_values.move_to_end(event_key)
            return False
        while len(_warned_closure_strictness_values) >= _MAX_WARNED_CLOSURE_EVENTS:
            _warned_closure_strictness_values.popitem(last=False)
        _warned_closure_strictness_values[event_key] = None
        return True


def _warn_closure_event_once(event_key: str, msg: str, *args: Any) -> None:
    """Logs a warning message the first time *event_key* is seen since cache reset."""
    if should_warn_closure_event(event_key):
        logger.warning(msg, *args)


@contextlib.contextmanager
def collect_heuristic_ctx_managers() -> Iterator[Set[str]]:
    """Collects heuristic context-manager names recorded on the current thread within this block."""
    stack: Optional[List[Set[str]]] = getattr(_heuristic_collector_tls, "stack", None)
    if stack is None:
        stack = []
        _heuristic_collector_tls.stack = stack
    bucket: Set[str] = set()
    stack.append(bucket)
    try:
        yield bucket
    finally:
        stack.pop()


def record_heuristic_non_suppressing_ctx_manager(mgr_name: str) -> None:
    """Records an unproven context manager accepted by the strict-mode name heuristic."""
    stack: Optional[List[Set[str]]] = getattr(_heuristic_collector_tls, "stack", None)
    if stack:
        for bucket in stack:
            bucket.add(mgr_name)
    with _warned_closure_lock:
        _get_or_create_thread_bucket_locked().add(mgr_name)


def peek_heuristic_non_suppressing_ctx_managers() -> Tuple[str, ...]:
    """Returns sorted unproven context managers recorded in the active session or current thread without clearing."""
    with _warned_closure_lock:
        bucket = _get_or_create_thread_bucket_locked()
        if not bucket.names:
            return ()
        return tuple(sorted(bucket.names))


def format_heuristic_ctx_manager_comment(names: Sequence[str]) -> str:
    """Formats a summary comment line for unproven context managers accepted via name heuristic."""
    if not names:
        return ""
    preview = ", ".join(repr(n) for n in names[:10])
    if len(names) > 10:
        preview += f", ... ({len(names)} total)"
    return (
        f"# Note: {len(names)} unproven context manager(s) ({preview}) were treated as "
        f"non-suppressing via name heuristic in strict mode; use --closure-strictness "
        f"pedantic to fail closed or list known-safe managers in --non-suppressing-ctx-managers."
    )


@contextlib.contextmanager
def defer_heuristic_ctx_manager_summary() -> Iterator[None]:
    """Defers mid-run log_heuristic_ctx_manager_summary() draining until an outer boundary."""
    with _warned_closure_lock:
        bucket = _get_or_create_thread_bucket_locked()
        bucket.defer_depth += 1
    try:
        yield
    finally:
        with _warned_closure_lock:
            bucket.defer_depth = max(0, bucket.defer_depth - 1)


@contextlib.contextmanager
def heuristic_ctx_manager_session(*, warn: bool = False) -> Iterator[None]:
    """Scopes a batch CLI or patch-generation run in a context-isolated session keyed by session ID.

    Propagates across ``asyncio`` tasks automatically via ``contextvars`` and across worker
    threads when tasks are submitted with ``contextvars.copy_context().run``; unassociated
    threads record into their own isolated per-thread fallback bucket without cross-session leakage.
    """
    prev_stack = _heuristic_session_stack.get()
    with _warned_closure_lock:
        existing_sid = prev_stack[-1] if prev_stack else None
        existing_bucket = (
            _active_heuristic_sessions.get(existing_sid)
            if existing_sid is not None
            else None
        )
        if existing_bucket is None or existing_sid is None:
            is_outermost = True
            sid = next(_next_heuristic_session_id)
            bucket = _HeuristicSessionBucket(defer_depth=1, warn_on_summary=warn)
            _active_heuristic_sessions[sid] = bucket
        else:
            is_outermost = False
            sid = existing_sid
            bucket = existing_bucket
            bucket.defer_depth += 1
            if warn:
                bucket.warn_on_summary = True
    token = _heuristic_session_stack.set((*prev_stack, sid))
    try:
        yield
    finally:
        emit_warn = False
        should_flush = False
        with _warned_closure_lock:
            bucket.defer_depth = max(0, bucket.defer_depth - 1)
            if is_outermost:
                should_flush = True
                emit_warn = bucket.warn_on_summary
        try:
            if should_flush:
                log_heuristic_ctx_manager_summary(force=True, warn=emit_warn)
        finally:
            _heuristic_session_stack.reset(token)
            if is_outermost:
                with _warned_closure_lock:
                    _active_heuristic_sessions.pop(sid, None)


def log_heuristic_ctx_manager_summary(
    *, force: bool = False, warn: Optional[bool] = None
) -> Tuple[str, ...]:
    """Logs a consolidated summary of unproven context managers recorded in the active session or thread.

    Logs at ``INFO`` by default so read-only scans and previews remain quiet in CI,
    and escalates to ``WARNING`` when ``warn=True`` (e.g. ``--replace-clones``).
    """
    with _warned_closure_lock:
        bucket = _get_or_create_thread_bucket_locked()
        if not bucket.names:
            return ()
        if bucket.defer_depth > 0 and not force:
            return tuple(sorted(bucket.names))
        names = tuple(sorted(bucket.names))
        overflow = bucket.overflow_count
        use_warn = bucket.warn_on_summary if warn is None else bool(warn)
        bucket.names.clear()
        bucket.overflow_count = 0
    distinct_count = len(names)
    preview = ", ".join(repr(n) for n in names[:10])
    if distinct_count > 10:
        preview += f", ... ({distinct_count} retained)"
    if overflow > 0:
        preview += f"; {overflow} additional occurrence(s) omitted beyond cap"
    log_fn = logger.warning if use_warn else logger.info
    log_fn(
        "Summary: %d unproven context manager(s) (%s) were treated as non-suppressing via "
        "name heuristic in strict mode; use --closure-strictness pedantic to fail closed or "
        "list known-safe managers in --non-suppressing-ctx-managers.",
        distinct_count,
        preview,
    )
    return names


def _maybe_warn_skip_conflict(
    is_lenient: bool,
    skip_requested: Optional[bool],
    event_prefix: str,
    message: str,
    mode_label: str,
) -> None:
    """Logs a deduplicated warning when skip_pre_unit_closures=True conflicts with a non-lenient strictness mode."""
    if not is_lenient and bool(skip_requested):
        _warn_closure_event_once(f"{event_prefix}:{mode_label}", message, mode_label)


def resolve_closure_strictness_mode(
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: Optional[bool] = False,
    strict_with_suppression: Optional[bool] = None,
) -> Tuple[str, bool]:
    """Resolves canonical closure strictness mode ('strict', 'pedantic', or 'lenient') and boolean skip flag.

    Precision vs Soundness Trade-off:
    'strict' (default for reporting and stub patches) is a heuristic safety mode for context
    managers and global-reading operations: it enforces all escaping closure and callback hazard
    checks while using a name-and-provenance heuristic for unproven unknown context managers
    (treating them as non-suppressing unless their terminal manager/callee name matches a
    suppression idiom in _SUSPICIOUS_SUPPRESS_SUBSTRINGS or is proven suppressing via
    local/function/class __exit__ provenance), and in global-reading units rejects
    attribute/subscript mutations, deletions, non-static class-body attributes (__set_name__
    descriptors), and external or @= augmented assignments while permitting dunder-invoking
    inspection/conversion built-ins, implicit operator/truthiness/iteration protocols, and local
    primitive-initialized arithmetic/bitwise accumulator augmented assignments.
    'pedantic' (default for replace_clones=True / --replace-clones when unset) is a stricter
    static conservative mode: it treats all unproven unknown context managers as potentially
    exception-suppressing unless explicitly listed in non_suppressing_ctx_managers, and in
    global-reading units rejects dunder-invoking built-in calls, all augmented assignments, and
    unproven implicit-dunder AST operations (binary/unary operators, non-identity comparisons and
    in/not in membership checks, attribute/@property and subscript reads, for/comprehension
    iteration, sequence/mapping unpacking, f-string formatting, set/dict key hashing, match
    class/value/mapping/sequence patterns on unproven subjects, import, non-builtin raise, and
    boolean truth-value evaluations) unless operands are statically proven built-in primitives or
    immutable literals (subject to Python's dynamic runtime boundaries such as frame
    introspection, metaclasses, and out-of-module monkeypatching).
    'lenient' relaxes speculative unreferenced-sibling closure checks while still enforcing
    direct cell-capture, intra-unit nonlocal mutator, and referenced-sibling closure checks.

    Unrecognized ``closure_strictness`` strings log a deduplicated warning and fall back to the
    supplied boolean flags: ``'lenient'`` when ``skip_pre_unit_closures=True``, ``'pedantic'``
    when ``strict_with_suppression=True``, and ``'strict'`` otherwise.
    """
    if closure_strictness is not None:
        c_mode = str(closure_strictness).strip().lower()
        if c_mode in ("strict", "pedantic", "lenient"):
            is_lenient_mode = c_mode == "lenient"
            _maybe_warn_skip_conflict(
                is_lenient_mode,
                skip_pre_unit_closures,
                "conflict_skip",
                "Conflicting configuration: closure_strictness '%s' takes precedence over "
                "skip_pre_unit_closures=True",
                closure_strictness,
            )
            expected_strict_with = c_mode == "pedantic"
            if (
                strict_with_suppression is not None
                and bool(strict_with_suppression) != expected_strict_with
                and (not is_lenient_mode or bool(strict_with_suppression))
            ):
                _warn_closure_event_once(
                    f"conflict_strict_with:{closure_strictness}:{bool(strict_with_suppression)}",
                    "Conflicting configuration: closure_strictness '%s' takes precedence over "
                    "strict_with_suppression=%s",
                    closure_strictness,
                    bool(strict_with_suppression),
                )
            return c_mode, is_lenient_mode
        fallback = (
            "lenient"
            if skip_pre_unit_closures
            else ("pedantic" if strict_with_suppression else "strict")
        )
        _warn_closure_event_once(
            f"unrecognized_strictness:{closure_strictness}",
            "Unrecognized closure_strictness '%s'; falling back to %s mode",
            closure_strictness,
            fallback,
        )
    is_lenient = bool(skip_pre_unit_closures)
    if is_lenient and strict_with_suppression:
        _warn_closure_event_once(
            "conflict_strict_with_and_skip:True:True",
            "Conflicting configuration: strict_with_suppression=True and "
            "skip_pre_unit_closures=True cannot both be active; "
            "skip_pre_unit_closures=True takes precedence (using 'lenient' mode)",
        )
        return "lenient", True
    if strict_with_suppression:
        return "pedantic", False
    return ("lenient" if is_lenient else "strict"), is_lenient


def resolve_strict_with_suppression(
    closure_strictness: Optional[str] = None,
    strict_with_suppression: Optional[bool] = None,
) -> bool:
    """Resolves whether unproven unknown context managers are treated as potentially suppressing.

    When both ``closure_strictness`` and ``strict_with_suppression`` are provided,
    ``closure_strictness`` takes precedence and logs a warning on conflict, matching
    ``resolve_closure_strictness_mode``.
    """
    resolved_mode, _ = resolve_closure_strictness_mode(
        closure_strictness=closure_strictness,
        strict_with_suppression=strict_with_suppression,
    )
    return resolved_mode == "pedantic"


def resolve_unit_closure_flags(
    unit: Mapping[str, Any],
    *,
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: Optional[bool] = None,
    strict_with_suppression: Optional[bool] = None,
) -> UnitClosureFlags:
    """Resolves (skip_pre_unit_closures, strict_with_suppression) for a unit with optional overrides.

    Precedence Order:
        1. Explicit ``closure_strictness`` argument from the caller.
        2. When ``closure_strictness`` is ``None`` and the unit itself explicitly sets a
           non-lenient closure policy (``unit["closure_strictness"]``,
           ``unit["strict_with_suppression"] is True``, or
           ``unit["skip_pre_unit_closures"] is False``), caller boolean flags
           (``skip_pre_unit_closures``, ``strict_with_suppression``) combine monotonically:
           they may strengthen the unit's policy (``lenient -> strict -> pedantic``) but
           never weaken an explicit stricter unit policy.
        3. Otherwise, caller boolean flags apply over unit dictionary defaults, falling
           back to ``"strict"`` mode: ``UnitClosureFlags(False, False)``.
    """
    if closure_strictness is not None:
        eff_skip_in = (
            skip_pre_unit_closures
            if skip_pre_unit_closures is not None
            else unit.get("skip_pre_unit_closures", False)
        )
        eff_sw_in = (
            strict_with_suppression
            if strict_with_suppression is not None
            else unit.get("strict_with_suppression")
        )
        resolved_mode, eff_skip = resolve_closure_strictness_mode(
            closure_strictness=closure_strictness,
            skip_pre_unit_closures=bool(eff_skip_in),
            strict_with_suppression=eff_sw_in,
        )
        return UnitClosureFlags(eff_skip, resolved_mode == "pedantic")

    has_explicit_unit_mode = (
        unit.get("closure_strictness") is not None
        or unit.get("strict_with_suppression") is True
        or unit.get("skip_pre_unit_closures") is False
    )
    unit_mode, unit_skip = resolve_closure_strictness_mode(
        closure_strictness=unit.get("closure_strictness"),
        skip_pre_unit_closures=bool(unit.get("skip_pre_unit_closures", False)),
        strict_with_suppression=unit.get("strict_with_suppression"),
    )
    unit_strict_with = unit_mode == "pedantic"
    if skip_pre_unit_closures is None and strict_with_suppression is None:
        return UnitClosureFlags(unit_skip, unit_strict_with)

    eff_skip_in = (
        skip_pre_unit_closures
        if skip_pre_unit_closures is not None
        else (False if strict_with_suppression else unit_skip)
    )
    eff_sw_in = (
        strict_with_suppression
        if strict_with_suppression is not None
        else (False if skip_pre_unit_closures else unit_strict_with)
    )
    resolved_mode, caller_skip = resolve_closure_strictness_mode(
        closure_strictness=None,
        skip_pre_unit_closures=bool(eff_skip_in),
        strict_with_suppression=eff_sw_in,
    )
    caller_strict_with = resolved_mode == "pedantic"
    if has_explicit_unit_mode:
        _maybe_warn_skip_conflict(
            unit_skip,
            caller_skip,
            "conflict_unit_strictness_vs_caller_skip",
            "Conflicting configuration: unit closure strictness '%s' cannot be "
            "weakened by skip_pre_unit_closures=True; pass closure_strictness='lenient' "
            "to override explicitly",
            unit_mode,
        )
        eff_skip = bool(unit_skip and caller_skip)
        eff_strict_with = bool(
            not eff_skip and (unit_strict_with or caller_strict_with)
        )
        return UnitClosureFlags(eff_skip, eff_strict_with)
    return UnitClosureFlags(caller_skip, caller_strict_with)


def resolve_pair_closure_flags(
    u1: Mapping[str, Any],
    u2: Mapping[str, Any],
    *,
    closure_strictness: Optional[str] = None,
    skip_pre_unit_closures: Optional[bool] = None,
    strict_with_suppression: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> PairClosureFlags:
    """Resolves per-unit and pair-level closure and context-manager flags for two clone units."""
    skip1, strict_with1 = resolve_unit_closure_flags(
        u1,
        closure_strictness=closure_strictness,
        skip_pre_unit_closures=skip_pre_unit_closures,
        strict_with_suppression=strict_with_suppression,
    )
    skip2, strict_with2 = resolve_unit_closure_flags(
        u2,
        closure_strictness=closure_strictness,
        skip_pre_unit_closures=skip_pre_unit_closures,
        strict_with_suppression=strict_with_suppression,
    )
    eff_non_sup1 = normalize_non_suppressing_ctx_managers(
        non_suppressing_ctx_managers,
        u1.get("non_suppressing_ctx_managers"),
    )
    eff_non_sup2 = normalize_non_suppressing_ctx_managers(
        non_suppressing_ctx_managers,
        u2.get("non_suppressing_ctx_managers"),
    )
    return PairClosureFlags(
        unit1=UnitClosureEndpointConfig(skip1, strict_with1, eff_non_sup1),
        unit2=UnitClosureEndpointConfig(skip2, strict_with2, eff_non_sup2),
        effective_skip_pre_unit_closures=bool(skip1 and skip2),
    )


def normalize_non_suppressing_ctx_managers(*sources: Any) -> Tuple[str, ...]:
    """Normalizes one or more context-manager name sources into a sorted tuple of names."""
    collected: Set[str] = set()
    stack: List[Any] = list(sources)
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, str):
            for part in item.split(","):
                cleaned = part.strip()
                if cleaned:
                    collected.add(cleaned)
        elif not isinstance(item, (bytes, bytearray, Mapping)) and isinstance(
            item, (Sequence, AbstractSet)
        ):
            stack.extend(item)
        else:
            invalid_key = f"invalid_non_sup_type:{type(item).__name__}"
            if should_warn_closure_event(invalid_key):
                logger.warning(
                    "Ignoring unsupported non_suppressing_ctx_managers value of type %s; "
                    "expected string or sequence of strings",
                    type(item).__name__,
                )
    return tuple(sorted(collected))


def clear_closure_warning_cache() -> None:
    """Clears warned closure strictness events and resets heuristic context-manager accumulators."""
    with _warned_closure_lock:
        _warned_closure_strictness_values.clear()
        _default_thread_buckets.clear()
        for bucket in _active_heuristic_sessions.values():
            bucket.names.clear()
            bucket.overflow_count = 0


DEFAULT_EXCLUDES: List[str] = [
    "checks/encapsulated",
    "checks\\encapsulated",
    "tests",
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    ".gemini",
    "scratch",
]

DEFAULT_TOOL_TOML_CONTENT = """[tool.pydoppelgangerhunt]
threshold = 0.90
min_lines = 8
min_tokens = 15
exclude = [
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "build",
    "dist",
]
call_sequences = true
idioms = true
# closure_strictness = "strict"  # Options: "strict", "pedantic", "lenient"
# non_suppressing_ctx_managers = ["mylib.SafeTransaction", "*.lock"]
"""


def normalize_path_string(path_str: Optional[str], strip_anchor: bool = True) -> str:
    """Normalizes a file path string by optionally stripping anchors, converting backslashes, and stripping leading './'."""
    return normalize_lexical_posix(path_str, strip_anchor=strip_anchor)


def canonical_path_key(path_str: Optional[str], strip_anchor: bool = False) -> str:
    """Returns canonical normalized path key, folding case on Windows for equivalence."""
    norm = normalize_lexical_posix(path_str, strip_anchor=strip_anchor)
    if os.name == "nt" or sys.platform == "win32":
        return norm.lower()
    return norm


def paths_match_boundary(
    p1: Optional[str],
    p2: Optional[str],
    strip_anchor: bool = True,
) -> bool:
    """Checks whether two normalized paths refer to the same file respecting directory boundaries."""
    n1 = normalize_lexical_posix(p1, strip_anchor=strip_anchor)
    n2 = normalize_lexical_posix(p2, strip_anchor=strip_anchor)
    if not n1 or not n2:
        return False
    if os.name == "nt" or sys.platform == "win32":
        n1 = n1.lower()
        n2 = n2.lower()
    if n1 == n2:
        return True
    return n1.endswith("/" + n2) or n2.endswith("/" + n1)


def find_matching_path_value(
    target_path: Optional[str],
    path_map: Dict[str, Any],
) -> Any:
    """Finds a value in a path-keyed mapping where keys match target_path respecting directory boundaries."""
    if not target_path or not path_map:
        return None
    direct = path_map.get(target_path)
    if direct is not None:
        return direct
    # 1. Match exact normalized path (preserving anchors)
    norm_exact = normalize_path_string(target_path, strip_anchor=False)
    if norm_exact in path_map:
        return path_map[norm_exact]
    key_exact = canonical_path_key(norm_exact, strip_anchor=False)
    for k, val in path_map.items():
        k_clean = normalize_path_string(k, strip_anchor=False)
        if norm_exact == k_clean or key_exact == canonical_path_key(k_clean, strip_anchor=False):
            return val
    for k, val in path_map.items():
        if paths_match_boundary(norm_exact, k, strip_anchor=False):
            return val

    # 2. If target_path contains a notebook cell anchor, fallback to stripped anchor matching
    target_raw = str(target_path)
    if parse_notebook_cell_anchor(target_raw) is not None:
        target_stripped = normalize_path_string(target_raw, strip_anchor=True)
        if target_stripped:
            if target_stripped in path_map:
                return path_map[target_stripped]
            key_stripped = canonical_path_key(target_stripped, strip_anchor=False)
            for k, val in path_map.items():
                k_clean = normalize_path_string(k, strip_anchor=False)
                if target_stripped == k_clean or key_stripped == canonical_path_key(k_clean, strip_anchor=False):
                    return val
            for k, val in path_map.items():
                if paths_match_boundary(target_stripped, k, strip_anchor=False):
                    return val
    return None


def find_python_files(
    target: Union[str, Path],
    excludes: Optional[List[str]] = None,
    audit_tests: bool = False,
    include_notebooks: bool = False,
) -> List[Path]:
    """Discovers all target Python source files while filtering excluded paths."""
    exclude_patterns = list(excludes if excludes is not None else DEFAULT_EXCLUDES)
    target_path = Path(target)
    if not target_path.exists():
        return []
    valid_suffixes = (".py", ".ipynb") if include_notebooks else (".py",)
    if target_path.is_file():
        return [target_path] if target_path.suffix.lower() in valid_suffixes else []

    target_clean = str(target_path).replace("\\", "/").strip("./")
    active_excludes = [
        ex
        for ex in exclude_patterns
        if ex.strip("./\\") and ex.replace("\\", "/").strip("./") not in target_clean
    ]
    if audit_tests:
        active_excludes = [ex for ex in active_excludes if "test" not in ex]

    def _is_excluded(full_p: str, rel_p: str) -> bool:
        for ex in active_excludes:
            clean_ex = ex.replace("\\", "/").strip("/")
            if clean_ex and (clean_ex in full_p or clean_ex in rel_p):
                return True
        return False

    found_files: List[Path] = []
    target_str = str(target_path)
    for root, dirs, files in os.walk(target_str):
        dirs.sort()
        rel_root = os.path.relpath(root, target_str).replace("\\", "/")
        norm_root = root.replace("\\", "/")
        if _is_excluded(norm_root, rel_root):
            continue
        for file in sorted(files):
            file_lower = file.lower()
            if not any(file_lower.endswith(sfx) for sfx in valid_suffixes):
                continue
            path = os.path.join(root, file)
            rel_file = os.path.relpath(path, target_str).replace("\\", "/")
            norm_file = path.replace("\\", "/")
            if _is_excluded(norm_file, rel_file):
                continue
            found_files.append(Path(path))

    return found_files


def _strip_toml_inline_comment(line: str) -> str:
    """Strips trailing TOML comments starting with # outside quotes."""
    in_quote: Optional[str] = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote is not None:
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            return line[:idx].strip()
    return line.strip()


def _count_bracket_depth(text: str, initial_depth: int = 0) -> int:
    """Computes net bracket nesting depth for [ and ] ignoring brackets inside quotes."""
    depth = initial_depth
    in_quote: Optional[str] = None
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote is not None:
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif in_quote is None:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
    return depth


def _parse_toml_array_value(val_str: str) -> List[Any]:
    """Parses a TOML array string representation into a list of Python typed values."""
    elems: List[Any] = []
    inner = val_str.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    raw_tokens: List[str] = []
    curr: List[str] = []
    in_quote: Optional[str] = None
    escaped = False
    bracket_depth = 0
    for ch in inner:
        if escaped:
            curr.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_quote is not None:
            curr.append(ch)
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            curr.append(ch)
        elif in_quote is None and ch == "[":
            bracket_depth += 1
            curr.append(ch)
        elif in_quote is None and ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            curr.append(ch)
        elif ch == "," and in_quote is None and bracket_depth == 0:
            raw_tokens.append("".join(curr).strip())
            curr = []
        else:
            curr.append(ch)
    if curr:
        raw_tokens.append("".join(curr).strip())

    for e_str in raw_tokens:
        if not e_str:
            continue
        if e_str.startswith("[") and e_str.endswith("]"):
            elems.append(_parse_toml_array_value(e_str))
        elif (e_str.startswith('"') and e_str.endswith('"')) or (
            e_str.startswith("'") and e_str.endswith("'")
        ):
            parsed_str = e_str[1:-1]
            if '\\"' in parsed_str or "\\'" in parsed_str:
                parsed_str = parsed_str.replace('\\"', '"').replace("\\'", "'")
            elems.append(parsed_str)
        elif e_str.lower() == "true":
            elems.append(True)
        elif e_str.lower() == "false":
            elems.append(False)
        else:
            try:
                elems.append(float(e_str) if "." in e_str else int(e_str))
            except ValueError:
                elems.append(e_str)
    return elems


def load_toml_section(
    target_file: Union[str, Path], section_name: str
) -> Dict[str, Any]:
    """Loads a specific tool configuration section from a TOML file with fallbacks."""
    target_path = Path(target_file)
    if not target_path.is_file():
        return {}

    # Prefer standard library tomllib (Python 3.11+) or third-party tomli if installed
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore
        with open(target_path, "rb") as fh:
            data = tomllib.load(fh)
        tool_sec = data.get("tool", {}) if isinstance(data, dict) else {}
        if isinstance(tool_sec, dict) and isinstance(tool_sec.get(section_name), dict):
            return dict(tool_sec[section_name])
        if isinstance(data, dict) and any(k in data for k in _CONFIG_KEY_NAMES):
            return dict(data)
    except (ImportError, OSError, ValueError, TypeError):
        pass

    # Built-in zero-dependency line parser fallback
    try:
        config: Dict[str, Any] = {}
        in_multiline_key: Optional[str] = None
        multiline_tokens: List[str] = []
        multiline_depth = 0
        target_section = f"[tool.{section_name}]"
        with open(target_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        is_headerless = True
        hdr_depth = 0
        for line in lines:
            s_ln = _strip_toml_inline_comment(line)
            if hdr_depth == 0 and s_ln.startswith("[") and s_ln.endswith("]"):
                is_headerless = False
                break
            hdr_depth = _count_bracket_depth(s_ln, hdr_depth)
        in_section = is_headerless

        for line in lines:
            stripped = _strip_toml_inline_comment(line)
            if not stripped:
                continue
            if in_multiline_key is not None:
                multiline_tokens.append(stripped)
                multiline_depth = _count_bracket_depth(stripped, multiline_depth)
                if multiline_depth == 0:
                    config[in_multiline_key] = _parse_toml_array_value(
                        " ".join(multiline_tokens)
                    )
                    in_multiline_key = None
                    multiline_tokens = []
                continue
            if multiline_depth > 0:
                multiline_depth = _count_bracket_depth(stripped, multiline_depth)
                continue
            if stripped == target_section:
                in_section = True
                continue
            if stripped.startswith("[") and stripped.endswith("]") and in_section:
                in_section = False
                continue
            if "=" in stripped:
                if not in_section:
                    _, v_raw = stripped.split("=", 1)
                    v_str = v_raw.strip()
                    if v_str.startswith("["):
                        multiline_depth = _count_bracket_depth(v_str, 0)
                    continue
                k, v = stripped.split("=", 1)
                k = k.strip().strip("\"'")
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (
                    v.startswith("'") and v.endswith("'")
                ):
                    config[k] = v[1:-1]
                elif v.startswith("["):
                    v_depth = _count_bracket_depth(v, 0)
                    if v_depth == 0:
                        config[k] = _parse_toml_array_value(v)
                    else:
                        in_multiline_key = k
                        multiline_tokens = [v]
                        multiline_depth = v_depth
                elif v.lower() == "true":
                    config[k] = True
                elif v.lower() == "false":
                    config[k] = False
                else:
                    try:
                        config[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
        if is_headerless:
            if any(k in config for k in _CONFIG_KEY_NAMES):
                return config
            return {}
        return config
    except OSError:
        return {}


def load_tool_config(
    config_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads tool configuration from pyproject.toml, .pydoppelgangerhunt.toml, or specified config path."""
    root = Path(repo_root) if repo_root else Path.cwd()
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.extend(
            [
                root / "pyproject.toml",
                root / ".pydoppelgangerhunt.toml",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            cfg = load_toml_section(candidate, "pydoppelgangerhunt")
            if cfg:
                return cfg
    return {}


def init_tool_configuration(target_dir: str = ".") -> str:
    """Initializes configuration for pyDoppelgangerHunt in pyproject.toml or standalone file."""
    pyproject_path = Path(target_dir) / "pyproject.toml"
    if pyproject_path.exists():
        try:
            content: Optional[str] = pyproject_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = None
        if content is not None:
            if "[tool.pydoppelgangerhunt]" in content:
                return f"{pyproject_path} (already configured)"
            with open(pyproject_path, "a", encoding="utf-8") as fh:
                fh.write("\n" + DEFAULT_TOOL_TOML_CONTENT)
            return str(pyproject_path)

    standalone_path = Path(target_dir) / ".pydoppelgangerhunt.toml"
    standalone_path.write_text(DEFAULT_TOOL_TOML_CONTENT, encoding="utf-8")
    return str(standalone_path)
