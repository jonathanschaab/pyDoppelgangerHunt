"""Configuration loading and initialization for pyDoppelgangerHunt."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Set as AbstractSet
import contextlib
import contextvars
from dataclasses import dataclass, field
import itertools
import keyword
import logging
import os
from pathlib import Path
import sys
import threading
import weakref
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
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
_warned_closure_strictness_values: OrderedDict[str, None] = OrderedDict()
_next_heuristic_session_id = itertools.count(1)
_heuristic_collector_stack: contextvars.ContextVar[
    Tuple[Tuple[Set[str], bool], ...]
] = contextvars.ContextVar("_heuristic_collector_stack", default=())
_wildcard_collector_stack: contextvars.ContextVar[
    Tuple[Tuple[Set[Tuple[str, str]], bool], ...]
] = contextvars.ContextVar("_wildcard_collector_stack", default=())


@dataclass
class _HeuristicSessionBucket:
    """Bounded accumulator for unproven heuristic and wildcard context managers recorded in a session or thread."""

    names: Set[str] = field(default_factory=set)
    overflow_count: int = 0
    wildcard_matches: Set[Tuple[str, str]] = field(default_factory=set)
    wildcard_overflow_count: int = 0
    defer_depth: int = 0
    warn_on_summary: bool = True
    closing: bool = False

    def add(self, mgr_name: str) -> None:
        """Records mgr_name up to _MAX_HEURISTIC_CTX_MANAGERS, counting overflow occurrences beyond cap."""
        if mgr_name in self.names:
            return
        if len(self.names) < _MAX_HEURISTIC_CTX_MANAGERS:
            self.names.add(mgr_name)
        else:
            self.overflow_count += 1

    def add_wildcard(self, pattern: str, mgr_name: str) -> None:
        """Records a (pattern, mgr_name) wildcard allowlist match up to _MAX_HEURISTIC_CTX_MANAGERS."""
        entry = (pattern, mgr_name)
        if entry in self.wildcard_matches:
            return
        if len(self.wildcard_matches) < _MAX_HEURISTIC_CTX_MANAGERS:
            self.wildcard_matches.add(entry)
        else:
            self.wildcard_overflow_count += 1


APP_VERSION = "2.0.0"

_active_heuristic_sessions: Dict[int, _HeuristicSessionBucket] = {}
_default_thread_buckets: weakref.WeakKeyDictionary[
    threading.Thread, _HeuristicSessionBucket
] = weakref.WeakKeyDictionary()
_heuristic_session_stack: contextvars.ContextVar[Tuple[int, ...]] = (
    contextvars.ContextVar("_heuristic_session_stack", default=())
)


def _find_active_session_bucket_locked(
    sess_stack: Tuple[int, ...],
    *,
    skip_closing: bool = False,
) -> Optional[_HeuristicSessionBucket]:
    """Returns the innermost active session bucket along *sess_stack*, if any."""
    for sid in reversed(sess_stack):
        active = _active_heuristic_sessions.get(sid)
        if active is not None and (not skip_closing or not active.closing):
            return active
    return None


def _get_thread_bucket_locked() -> Optional[_HeuristicSessionBucket]:
    """Returns the active session bucket or existing per-Thread fallback bucket without allocating."""
    sess_stack = _heuristic_session_stack.get()
    if sess_stack:
        active = _find_active_session_bucket_locked(sess_stack, skip_closing=True)
        if active is not None:
            return active
    return _default_thread_buckets.get(threading.current_thread())


def _get_or_create_thread_bucket_locked() -> _HeuristicSessionBucket:
    """Returns the active session bucket for the current context or a per-Thread fallback bucket."""
    bucket = _get_thread_bucket_locked()
    if bucket is not None:
        return bucket
    bucket = _HeuristicSessionBucket()
    _default_thread_buckets[threading.current_thread()] = bucket
    return bucket


def normalize_heuristic_ctx_managers(*sources: Any) -> List[str]:
    """Normalizes sequence/set heuristic context-manager names into sorted unique non-empty strings."""
    merged: Set[str] = set()
    for src in sources:
        if not (
            isinstance(src, AbstractSet)
            or (isinstance(src, Sequence) and not isinstance(src, (str, bytes, bytearray)))
        ):
            continue
        for item in src:
            if isinstance(item, str) and item:
                merged.add(item)
    return sorted(merged)


def normalize_wildcard_ctx_manager_matches(*sources: Any) -> List[Tuple[str, str]]:
    """Normalizes sequence/set or JSON-deserialized wildcard match pairs into sorted unique 2-tuples."""
    merged: Set[Tuple[str, str]] = set()
    for src in sources:
        if not (
            isinstance(src, AbstractSet)
            or (isinstance(src, Sequence) and not isinstance(src, (str, bytes, bytearray)))
        ):
            continue
        for item in src:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                merged.add((str(item[0]), str(item[1])))
    return sorted(merged)


_CONFIG_KEY_NAMES: Tuple[str, ...] = (
    "threshold",
    "min_lines",
    "exemptions",
    "exclude",
    "closure_strictness",
    "non_suppressing_ctx_managers",
    "top",
)


# INVARIANT (SINGLE KNOB): ``closure_strictness`` ("lenient" | "strict" | "pedantic")
# is the sole configuration knob controlling closure-cell and context-manager strictness
# across CLI flags, pyproject.toml, UnitSchema, and Python APIs. Do NOT introduce
# overlapping boolean flags (such as ``skip_pre_unit_closures`` or
# ``strict_with_suppression``): overlapping knobs create cross-flag precedence drift,
# conflicting override warnings, and contradictory defaults across entrypoints.
_CANONICAL_CLOSURE_MODES: frozenset[str] = frozenset({"strict", "pedantic", "lenient"})
_STRICTNESS_RANK: Mapping[str, int] = {
    "lenient": 0,
    "strict": 1,
    "pedantic": 2,
}


class ResolvedClosurePolicy(NamedTuple):
    """Canonical resolved closure strictness and context-manager allowlist for one unit."""

    mode: str
    non_suppressing_ctx_managers: Tuple[str, ...] = ()

    @property
    def is_lenient(self) -> bool:
        """Returns True when speculative unreferenced-sibling closure checks are relaxed."""
        return self.mode == "lenient"

    @property
    def is_strict(self) -> bool:
        """Returns True when heuristic context-manager strict mode is active."""
        return self.mode == "strict"

    @property
    def is_pedantic(self) -> bool:
        """Returns True when all unproven context managers fail closed."""
        return self.mode == "pedantic"


class PairClosurePolicy(NamedTuple):
    """Canonical resolved per-unit and combined closure policy for a clone pair."""

    unit1: ResolvedClosurePolicy
    unit2: ResolvedClosurePolicy
    effective_mode: str

    @property
    def is_lenient(self) -> bool:
        """Returns True only when both units resolve to lenient mode."""
        return self.effective_mode == "lenient"

    @property
    def is_strict(self) -> bool:
        """Returns True when the combined pair policy is strict mode."""
        return self.effective_mode == "strict"

    @property
    def is_pedantic(self) -> bool:
        """Returns True when either unit requires pedantic mode."""
        return self.effective_mode == "pedantic"


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


_T = TypeVar("_T")


@contextlib.contextmanager
def bind_contextvar(
    var: contextvars.ContextVar[_T],
    value: _T,
) -> Iterator[_T]:
    """Binds *var* to *value* for the block and restores its prior value safely across threads."""
    prev = var.get()
    token = var.set(value)
    try:
        yield value
    finally:
        try:
            var.reset(token)
        except ValueError:
            var.set(prev)


@contextlib.contextmanager
def _collect_ctxvar_bucket(
    stack_var: contextvars.ContextVar[Tuple[Tuple[Set[_T], bool], ...]],
    *,
    isolate: bool = False,
) -> Iterator[Set[_T]]:
    """Pushes a temporary set onto *stack_var* for the duration of the context block."""
    bucket: Set[_T] = set()
    prev_stack = stack_var.get()
    with bind_contextvar(stack_var, (*prev_stack, (bucket, isolate))):
        yield bucket


def collect_heuristic_ctx_managers(*, isolate: bool = False) -> ContextManager[Set[str]]:
    """Collects heuristic context-manager names recorded within this contextvars scope."""
    return _collect_ctxvar_bucket(_heuristic_collector_stack, isolate=isolate)


def collect_wildcard_ctx_managers(
    *, isolate: bool = False
) -> ContextManager[Set[Tuple[str, str]]]:
    """Collects (pattern, manager_name) pairs matched via wildcard allowlist entries in this scope."""
    return _collect_ctxvar_bucket(_wildcard_collector_stack, isolate=isolate)


def _record_ctxvar_and_thread_bucket(
    stack_var: contextvars.ContextVar[Tuple[Tuple[Set[_T], bool], ...]],
    entry: _T,
    record_fn: Callable[[_HeuristicSessionBucket], None],
) -> None:
    """Appends *entry* to active *stack_var* collectors and applies *record_fn* to the active thread bucket."""
    collector_stack = stack_var.get()
    with _warned_closure_lock:
        isolated = False
        for bucket, is_isolated in reversed(collector_stack):
            if entry not in bucket and len(bucket) < _MAX_HEURISTIC_CTX_MANAGERS:
                bucket.add(entry)
            if is_isolated:
                isolated = True
                break
        if not isolated:
            record_fn(_get_or_create_thread_bucket_locked())


def record_heuristic_non_suppressing_ctx_manager(mgr_name: str) -> None:
    """Records an unproven context manager accepted by the strict-mode name heuristic."""
    _record_ctxvar_and_thread_bucket(
        _heuristic_collector_stack,
        mgr_name,
        lambda bucket: bucket.add(mgr_name),
    )


def record_wildcard_ctx_manager_match(pattern: str, mgr_name: str) -> None:
    """Records a context manager accepted via a wildcard allowlist pattern (e.g. '*.atomic')."""
    _record_ctxvar_and_thread_bucket(
        _wildcard_collector_stack,
        (pattern, mgr_name),
        lambda bucket: bucket.add_wildcard(pattern, mgr_name),
    )


def peek_heuristic_non_suppressing_ctx_managers() -> Tuple[str, ...]:
    """Returns sorted unproven context managers recorded in the active session or current thread without clearing."""
    with _warned_closure_lock:
        bucket = _get_thread_bucket_locked()
        if bucket is None or not bucket.names:
            return ()
        return tuple(sorted(bucket.names))


def peek_wildcard_ctx_manager_matches() -> Tuple[Tuple[str, str], ...]:
    """Returns sorted (pattern, manager_name) wildcard matches recorded in the active session or thread without clearing."""
    with _warned_closure_lock:
        bucket = _get_thread_bucket_locked()
        if bucket is None or not bucket.wildcard_matches:
            return ()
        return tuple(sorted(bucket.wildcard_matches))


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


def format_wildcard_ctx_manager_comment(
    matches: Sequence[Tuple[str, str]],
) -> str:
    """Formats a summary comment line for context managers accepted via wildcard allowlist patterns."""
    if not matches:
        return ""
    ordered = sorted(set(matches))
    formatted = [f"{mgr!r} via {pat!r}" for pat, mgr in ordered[:10]]
    preview = ", ".join(formatted)
    if len(ordered) > 10:
        preview += f", ... ({len(ordered)} total)"
    patterns = sorted({pat for pat, _ in ordered})
    patterns_preview = ", ".join(repr(p) for p in patterns[:10])
    if len(patterns) > 10:
        patterns_preview += f", ... ({len(patterns)} total)"
    return (
        f"# Note: {len(ordered)} context manager(s) matched wildcard allowlist pattern(s) "
        f"[{patterns_preview}]: {preview}; use exact qualified names in "
        f"--non-suppressing-ctx-managers when stronger provenance precision is desired."
    )


def format_ctx_manager_assumption_comments(
    heuristic_names: Sequence[str],
    wildcard_matches: Sequence[Tuple[str, str]] = (),
) -> str:
    """Formats combined # Note: comments for strict-mode heuristic and wildcard allowlist assumptions."""
    lines = [
        line
        for line in (
            format_heuristic_ctx_manager_comment(heuristic_names),
            format_wildcard_ctx_manager_comment(wildcard_matches),
        )
        if line
    ]
    return "\n".join(lines)


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


class _DrainedHeuristicSummary(NamedTuple):
    """Snapshot drained atomically from a heuristic/wildcard session bucket for logging."""

    names: Tuple[str, ...]
    overflow: int
    wildcard_matches: Tuple[Tuple[str, str], ...]
    wildcard_overflow: int
    use_warn: bool


def _drain_bucket_locked(
    bucket: _HeuristicSessionBucket,
    *,
    force: bool,
    warn: Optional[bool],
) -> Optional[_DrainedHeuristicSummary]:
    """Drains bucket under _warned_closure_lock when non-empty and un-deferred (or forced)."""
    if not bucket.names and not bucket.wildcard_matches:
        return None
    if bucket.defer_depth > 0 and not force:
        return None
    summary = _DrainedHeuristicSummary(
        names=tuple(sorted(bucket.names)),
        overflow=bucket.overflow_count,
        wildcard_matches=tuple(sorted(bucket.wildcard_matches)),
        wildcard_overflow=bucket.wildcard_overflow_count,
        use_warn=bucket.warn_on_summary if warn is None else bool(warn),
    )
    bucket.names.clear()
    bucket.overflow_count = 0
    bucket.wildcard_matches.clear()
    bucket.wildcard_overflow_count = 0
    return summary


def _emit_drained_heuristic_summary(summary: _DrainedHeuristicSummary) -> None:
    """Logs consolidated heuristic and wildcard context-manager summaries outside locks."""
    log_fn = logger.warning if summary.use_warn else logger.info
    if summary.names:
        distinct_count = len(summary.names)
        preview = ", ".join(repr(n) for n in summary.names[:10])
        if distinct_count > 10:
            preview += f", ... ({distinct_count} retained)"
        if summary.overflow > 0:
            preview += f"; {summary.overflow} additional occurrence(s) omitted beyond cap"
        log_fn(
            "Summary: %d unproven context manager(s) (%s) were treated as non-suppressing via "
            "name heuristic in strict mode; use --closure-strictness pedantic to fail closed or "
            "list known-safe managers in --non-suppressing-ctx-managers.",
            distinct_count,
            preview,
        )
    if summary.wildcard_matches:
        wc_count = len(summary.wildcard_matches)
        wc_patterns = sorted({pat for pat, _ in summary.wildcard_matches})
        pat_preview = ", ".join(repr(p) for p in wc_patterns[:10])
        if len(wc_patterns) > 10:
            pat_preview += f", ... ({len(wc_patterns)} patterns)"
        match_preview = ", ".join(
            f"{mgr!r} via {pat!r}" for pat, mgr in summary.wildcard_matches[:10]
        )
        if wc_count > 10:
            match_preview += f", ... ({wc_count} retained)"
        if summary.wildcard_overflow > 0:
            match_preview += (
                f"; {summary.wildcard_overflow} additional occurrence(s) omitted beyond cap"
            )
        log_fn(
            "Summary: %d context manager(s) accepted via wildcard allowlist pattern(s) [%s]: %s; "
            "consider replacing wildcard entries in --non-suppressing-ctx-managers with "
            "explicit qualified names.",
            wc_count,
            pat_preview,
            match_preview,
        )


@contextlib.contextmanager
def heuristic_ctx_manager_session(*, warn: Optional[bool] = None) -> Iterator[None]:
    """Scopes a batch CLI or patch-generation run in a context-isolated session keyed by session ID.

    When ``warn`` is ``None``, outermost sessions default to ``True`` (consolidated
    ``WARNING``) while nested sessions inherit the enclosing session's warning setting so
    inner wrappers preserve an outer ``heuristic_ctx_manager_session(warn=False)`` context.
    Passing an explicit ``warn=True`` on a nested session that records findings merges
    those findings into the enclosing session and escalates the enclosing session's
    consolidated summary to ``WARNING`` so explicit warning requests are never silently
    downgraded.

    Propagates across ``asyncio`` tasks automatically via ``contextvars`` and across worker
    threads when tasks are submitted with ``contextvars.copy_context().run``; unassociated
    threads record into their own isolated per-thread fallback bucket without cross-session leakage.
    """
    prev_stack = _heuristic_session_stack.get()
    with _warned_closure_lock:
        sid = next(_next_heuristic_session_id)
        prev_parent = _find_active_session_bucket_locked(
            prev_stack, skip_closing=True
        )
        is_outermost = prev_parent is None
        effective_warn = (
            (True if warn is None else bool(warn))
            if prev_parent is None
            else (prev_parent.warn_on_summary if warn is None else bool(warn))
        )
        bucket = _HeuristicSessionBucket(defer_depth=1, warn_on_summary=effective_warn)
        _active_heuristic_sessions[sid] = bucket
    token = _heuristic_session_stack.set((*prev_stack, sid))
    try:
        yield
    finally:
        drained: Optional[_DrainedHeuristicSummary] = None
        with _warned_closure_lock:
            bucket.defer_depth = max(0, bucket.defer_depth - 1)
            parent_bucket = (
                None
                if is_outermost
                else _find_active_session_bucket_locked(
                    prev_stack, skip_closing=True
                )
            )
            bucket.closing = True
            _active_heuristic_sessions.pop(sid, None)
            if parent_bucket is None:
                drained = _drain_bucket_locked(
                    bucket, force=True, warn=bucket.warn_on_summary
                )
            elif (
                bucket.names
                or bucket.overflow_count > 0
                or bucket.wildcard_matches
                or bucket.wildcard_overflow_count > 0
            ):
                for mgr_name in bucket.names:
                    parent_bucket.add(mgr_name)
                parent_bucket.overflow_count += bucket.overflow_count
                for pat, mgr_name in bucket.wildcard_matches:
                    parent_bucket.add_wildcard(pat, mgr_name)
                parent_bucket.wildcard_overflow_count += (
                    bucket.wildcard_overflow_count
                )
                if bucket.warn_on_summary:
                    parent_bucket.warn_on_summary = True
        try:
            if drained is not None:
                _emit_drained_heuristic_summary(drained)
        finally:
            try:
                _heuristic_session_stack.reset(token)
            except ValueError:
                _heuristic_session_stack.set(prev_stack)


def log_heuristic_ctx_manager_summary(
    *, force: bool = False, warn: Optional[bool] = None
) -> Tuple[str, ...]:
    """Logs consolidated summaries for strict-mode heuristic and wildcard allowlist context managers.

    Logs a single consolidated ``WARNING`` by default so assumptions remain visible without
    per-item warning noise, or ``INFO`` when ``warn=False`` is explicitly active.

    When called inside an active ``heuristic_ctx_manager_session()`` (or
    ``defer_heuristic_ctx_manager_summary()``) with ``force=False``, flushing is deferred to
    session exit: this function returns the currently buffered unproven heuristic manager names
    without draining or logging. Wildcard allowlist matches are logged when drained and are
    omitted from the returned tuple (use ``peek_wildcard_ctx_manager_matches()`` to inspect
    buffered wildcard matches).
    """
    with _warned_closure_lock:
        bucket = _get_thread_bucket_locked()
        if bucket is None or (not bucket.names and not bucket.wildcard_matches):
            return ()
        if bucket.defer_depth > 0 and not force:
            return tuple(sorted(bucket.names))
        drained = _drain_bucket_locked(bucket, force=force, warn=warn)
    if drained is None:
        return ()
    _emit_drained_heuristic_summary(drained)
    return drained.names


def validate_closure_strictness(value: Any) -> str:
    """Validates a closure_strictness setting, returning canonical lowercase mode or raising ValueError."""
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid closure_strictness value {value!r}; expected one of "
            f"{sorted(_CANONICAL_CLOSURE_MODES)}"
        )
    c_mode = value.strip().lower()
    if c_mode not in _CANONICAL_CLOSURE_MODES:
        raise ValueError(
            f"Invalid closure_strictness value {value!r}; expected one of "
            f"{sorted(_CANONICAL_CLOSURE_MODES)}"
        )
    return c_mode


def is_recognized_closure_strictness(value: Any) -> bool:
    """Returns True when value is a canonical closure_strictness mode name."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _CANONICAL_CLOSURE_MODES


def _warn_unrecognized_strictness(
    closure_strictness: Any,
    fallback_mode: str,
) -> None:
    """Logs a deduplicated warning when an unrecognized closure_strictness value falls back."""
    _warn_closure_event_once(
        f"unrecognized_strictness:{closure_strictness}",
        "Unrecognized closure_strictness '%s'; falling back to %s mode",
        closure_strictness,
        fallback_mode,
    )


def combine_closure_strictness(mode1: str, mode2: str) -> str:
    """Returns the stricter of two canonical closure strictness modes (lenient < strict < pedantic)."""
    c1 = validate_closure_strictness(mode1)
    c2 = validate_closure_strictness(mode2)
    return c1 if _STRICTNESS_RANK[c1] >= _STRICTNESS_RANK[c2] else c2


def resolve_closure_strictness_mode(
    closure_strictness: Optional[str] = None,
    *,
    default: str = "strict",
) -> str:
    """Resolves canonical closure strictness mode ('lenient', 'strict', or 'pedantic').

    Precision vs Soundness Trade-off:
    'strict' (default for reporting) is a pragmatic safety mode: it enforces all escaping
    closure and callback hazard checks while using a closed name-and-provenance heuristic
    for unproven unknown context managers and permitting common dunder-invoking built-ins
    in global-reading units.
    'pedantic' (default for --patch / generate_refactoring_patch when unset) is a
    conservative static mode: it treats all unproven unknown context managers as
    potentially exception-suppressing unless explicitly listed in
    non_suppressing_ctx_managers, and rejects dunder-invoking built-ins and unproven
    implicit-dunder AST operations in global-reading units.
    'lenient' relaxes speculative unreferenced-sibling closure checks while still
    enforcing direct cell-capture, intra-unit nonlocal mutator, and referenced-sibling
    closure checks.

    Unrecognized ``closure_strictness`` strings log a deduplicated warning and fall back
    to ``default`` (validated as a canonical mode).
    """
    fallback = validate_closure_strictness(default)
    if closure_strictness is None:
        return fallback
    if isinstance(closure_strictness, str):
        c_mode = closure_strictness.strip().lower()
        if c_mode in _CANONICAL_CLOSURE_MODES:
            return c_mode
    _warn_unrecognized_strictness(closure_strictness, fallback)
    return fallback


def resolve_unit_closure_policy(
    unit: Mapping[str, Any],
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    default: str = "strict",
) -> ResolvedClosurePolicy:
    """Resolves the canonical closure policy for a single unit.

    Precedence Order:
        1. Explicit recognized ``closure_strictness`` argument from the caller.
        2. Recognized ``unit["closure_strictness"]`` field on the unit dict.
        3. ``default`` (``"strict"`` for analysis/reporting; ``"pedantic"`` for patching).

    Unrecognized caller or unit ``closure_strictness`` strings log a deduplicated
    warning and fall back down the precedence order.
    """
    fallback_default = validate_closure_strictness(default)
    raw_unit_mode = unit.get("closure_strictness")
    unit_mode: Optional[str] = None
    if raw_unit_mode is not None:
        if is_recognized_closure_strictness(raw_unit_mode):
            unit_mode = str(raw_unit_mode).strip().lower()
        else:
            _warn_unrecognized_strictness(raw_unit_mode, fallback_default)

    if closure_strictness is not None:
        if is_recognized_closure_strictness(closure_strictness):
            resolved_mode = str(closure_strictness).strip().lower()
        else:
            # Fail closed on unrecognized caller strictness: never weaken below
            # either the unit's explicit strictness or the entrypoint default.
            resolved_mode = (
                combine_closure_strictness(unit_mode, fallback_default)
                if unit_mode is not None
                else fallback_default
            )
            _warn_unrecognized_strictness(closure_strictness, resolved_mode)
    else:
        resolved_mode = unit_mode if unit_mode is not None else fallback_default

    eff_non_sup = normalize_non_suppressing_ctx_managers(
        non_suppressing_ctx_managers,
        unit.get("non_suppressing_ctx_managers"),
    )
    return ResolvedClosurePolicy(
        mode=resolved_mode,
        non_suppressing_ctx_managers=eff_non_sup,
    )


def resolve_pair_closure_policy(
    u1: Mapping[str, Any],
    u2: Mapping[str, Any],
    *,
    closure_strictness: Optional[str] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    default: str = "strict",
) -> PairClosurePolicy:
    """Resolves canonical per-unit and combined closure policy for two clone units."""
    fallback_default = validate_closure_strictness(default)
    caller_unrecognized = (
        closure_strictness is not None
        and not is_recognized_closure_strictness(closure_strictness)
    )
    effective_caller = None if caller_unrecognized else closure_strictness
    p1 = resolve_unit_closure_policy(
        u1,
        closure_strictness=effective_caller,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        default=fallback_default,
    )
    p2 = resolve_unit_closure_policy(
        u2,
        closure_strictness=effective_caller,
        non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        default=fallback_default,
    )
    if caller_unrecognized:
        p1 = ResolvedClosurePolicy(
            mode=combine_closure_strictness(p1.mode, fallback_default),
            non_suppressing_ctx_managers=p1.non_suppressing_ctx_managers,
        )
        p2 = ResolvedClosurePolicy(
            mode=combine_closure_strictness(p2.mode, fallback_default),
            non_suppressing_ctx_managers=p2.non_suppressing_ctx_managers,
        )
    effective_mode = combine_closure_strictness(p1.mode, p2.mode)
    if caller_unrecognized:
        _warn_unrecognized_strictness(closure_strictness, effective_mode)
    return PairClosurePolicy(
        unit1=p1,
        unit2=p2,
        effective_mode=effective_mode,
    )


def _validate_ctx_manager_pattern(pattern: str) -> str:
    """Validates one non_suppressing_ctx_managers entry, rejecting degenerate wildcards and invalid identifiers."""
    if pattern == "." * len(pattern) or pattern.endswith("."):
        raise ValueError(
            f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: trailing or empty dot segment"
        )
    prefix_dots = len(pattern) - len(pattern.lstrip("."))
    rest = pattern[prefix_dots:]
    parts = rest.split(".")
    if any(not seg for seg in parts):
        raise ValueError(
            f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: empty dotted segment"
        )
    if "*" in pattern:
        if any("*" in seg and seg != "*" for seg in parts):
            raise ValueError(
                f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: '*' must occupy a full dotted segment"
            )
        if parts.count("*") != 1:
            raise ValueError(
                f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: multiple '*' wildcards are unsupported"
            )
        star_idx = parts.index("*")
        if star_idx != len(parts) - 2 or (prefix_dots > 0 and star_idx == 0):
            raise ValueError(
                f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: only '*.attr' or "
                f"'<module>.*.attr' wildcards are supported"
            )
    if not all(
        seg == "*" or (seg.isidentifier() and not keyword.iskeyword(seg))
        for seg in parts
    ):
        raise ValueError(
            f"Invalid non_suppressing_ctx_managers pattern {pattern!r}: "
            f"segments must be valid identifiers (not Python keywords)"
        )
    return pattern


def normalize_non_suppressing_ctx_managers(*sources: Any) -> Tuple[str, ...]:
    """Normalizes context-manager allowlist sources into a sorted tuple, rejecting invalid wildcard patterns."""
    collected: Set[str] = set()
    seen_container_ids: Set[int] = set()
    stack: List[Any] = list(sources)
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, str):
            for part in item.split(","):
                cleaned = part.strip()
                if cleaned:
                    collected.add(_validate_ctx_manager_pattern(cleaned))
        elif not isinstance(item, (bytes, bytearray, Mapping)) and isinstance(
            item, (Sequence, AbstractSet)
        ):
            obj_id = id(item)
            if obj_id in seen_container_ids:
                continue
            seen_container_ids.add(obj_id)
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
    """Clears warned closure strictness events and stale unassociated thread buckets without wiping active sessions."""
    sess_stack = _heuristic_session_stack.get()
    curr_thread = threading.current_thread()
    discarded_threads: List[Tuple[str, Tuple[str, ...], Tuple[Tuple[str, str], ...]]] = []
    with _warned_closure_lock:
        _warned_closure_strictness_values.clear()
        if (
            not sess_stack
            or _find_active_session_bucket_locked(sess_stack, skip_closing=True) is None
        ):
            _default_thread_buckets.pop(curr_thread, None)
        for thr in list(_default_thread_buckets.keys()):
            if not thr.is_alive():
                bucket = _default_thread_buckets.pop(thr, None)
                if bucket is not None and (bucket.names or bucket.wildcard_matches):
                    discarded_threads.append(
                        (
                            thr.name,
                            tuple(sorted(bucket.names)),
                            tuple(sorted(bucket.wildcard_matches)),
                        )
                    )
    for thr_name, heur_names, wc_matches in discarded_threads:
        logger.debug(
            "Cleared unassociated thread '%s' context-manager bucket (%d heuristic, %d wildcard)",
            thr_name,
            len(heur_names),
            len(wc_matches),
        )


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
# Prefer explicit qualified names or class names; use wildcards ("<module>.*.attr" or "*.attr") only as a last resort:
# non_suppressing_ctx_managers = ["mylib.Transaction", "Transaction", "*.atomic"]
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


# INVARIANT (TOML FALLBACK CONVERGENCE BOUNDARY):
# When stdlib ``tomllib`` (Python 3.11+) or ``tomli`` is importable, it is the
# sole authoritative TOML parser: ``_load_raw_toml_section`` MUST return directly
# (including ``{}`` on absent sections or ``TOMLDecodeError``) and MUST NOT fall
# through to this line scanner. These helpers exist solely as a bounded
# zero-dependency fallback for Python < 3.11 without ``tomli`` (supporting flat
# keys, ``[tool.<section>]`` headers, basic/literal strings, booleans, numbers,
# and single/multiline arrays); per TOML 1.0, backslashes escape only inside
# double-quoted basic strings, never inside single-quoted literal strings.
def _strip_toml_inline_comment(line: str) -> str:
    """Strips trailing TOML comments starting with # outside quotes."""
    in_quote: Optional[str] = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_quote == '"':
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
        if ch == "\\" and in_quote == '"':
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
        if ch == "\\" and in_quote == '"':
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
        elif e_str.startswith('"') and e_str.endswith('"') and len(e_str) >= 2:
            parsed_str = e_str[1:-1]
            if '\\"' in parsed_str:
                parsed_str = parsed_str.replace('\\"', '"')
            elems.append(parsed_str)
        elif e_str.startswith("'") and e_str.endswith("'") and len(e_str) >= 2:
            elems.append(e_str[1:-1])
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


def _validate_loaded_tool_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates closure policy fields in loaded TOML config, raising ValueError on invalid values."""
    if not config:
        return config
    if "closure_strictness" in config and config["closure_strictness"] is not None:
        config["closure_strictness"] = validate_closure_strictness(
            config["closure_strictness"]
        )
    if (
        "non_suppressing_ctx_managers" in config
        and config["non_suppressing_ctx_managers"] is not None
    ):
        raw_mgrs = config["non_suppressing_ctx_managers"]
        is_str = isinstance(raw_mgrs, str)
        is_str_collection = (
            not isinstance(raw_mgrs, (bytes, bytearray, Mapping))
            and isinstance(raw_mgrs, (Sequence, AbstractSet))
            and all(isinstance(item, str) for item in raw_mgrs)
        )
        if not (is_str or is_str_collection):
            raise ValueError(
                f"Invalid non_suppressing_ctx_managers value {raw_mgrs!r}; "
                f"expected string or sequence of strings"
            )
        normalize_non_suppressing_ctx_managers(raw_mgrs)
    return config


def _load_raw_toml_section(target_path: Path, section_name: str) -> Dict[str, Any]:
    """Parses a specific tool configuration section from TOML using tomllib/tomli or line fallback."""
    if not target_path.is_file():
        return {}

    # Prefer standard library tomllib (Python 3.11+) or third-party tomli if installed.
    # Once imported, the stdlib/tomli parser is authoritative: never fall through to the
    # line scanner when the target section is absent or the file has TOML syntax errors.
    tomllib_mod: Any = None
    try:
        if sys.version_info >= (3, 11):
            import tomllib as tomllib_mod
        else:
            import tomli as tomllib_mod  # type: ignore
    except ImportError:
        tomllib_mod = None

    if tomllib_mod is not None:
        try:
            with open(target_path, "rb") as fh:
                data = tomllib_mod.load(fh)
        except (OSError, ValueError, TypeError):
            return {}
        tool_sec = data.get("tool", {}) if isinstance(data, dict) else {}
        if isinstance(tool_sec, dict) and isinstance(tool_sec.get(section_name), dict):
            return dict(tool_sec[section_name])
        if isinstance(data, dict) and any(k in data for k in _CONFIG_KEY_NAMES):
            return dict(data)
        return {}

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


def load_toml_section(
    target_file: Union[str, Path], section_name: str
) -> Dict[str, Any]:
    """Loads and validates a specific tool configuration section from a TOML file."""
    raw_cfg = _load_raw_toml_section(Path(target_file), section_name)
    return _validate_loaded_tool_config(raw_cfg)


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
