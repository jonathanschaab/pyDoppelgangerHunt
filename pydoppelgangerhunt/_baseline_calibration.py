"""Corpus shingle calibration, serialization, and reduction metrics for clone baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.canonical_path import normalize_lexical_posix

ShingleKeyWireFormat = str


MAX_CALIBRATION_UNITS: int = 1_000_000_000


_CASE_FOLD_PLATFORM: bool = os.name == "nt" or sys.platform == "win32"


def _safe_total_units(raw_units: Any) -> int:
    """Clamps total_units to [0, MAX_CALIBRATION_UNITS], returning 0 for malformed/overflowing values."""
    if raw_units is None or isinstance(raw_units, bool):
        return 0
    try:
        val = int(raw_units)
        if 0 <= val <= MAX_CALIBRATION_UNITS:
            return val
        if val > MAX_CALIBRATION_UNITS:
            return MAX_CALIBRATION_UNITS
    except (ValueError, TypeError, OverflowError):
        pass
    return 0


def _safe_int(raw_val: Any, min_val: int = 1) -> Optional[int]:
    """Validates and parses an integer >= min_val, or returns None."""
    if raw_val is None or isinstance(raw_val, bool):
        return None
    try:
        val = int(raw_val)
        if val >= min_val:
            return val
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _safe_min_corpus(raw_min_corpus: Any, filter_stop_shingles: bool = False) -> int:
    """Sanitizes min_corpus_size to a non-negative int, or default based on filter_stop_shingles."""
    val = _safe_int(raw_min_corpus, min_val=0)
    if val is not None:
        return val
    return 4 if filter_stop_shingles else 30


def _safe_bool(val: Any) -> bool:
    """Safely coerces arbitrary input to bool, handling common string encodings and rejecting malformed types."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val != 0)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


HARVEST_BOOLEAN_MODES: Tuple[Tuple[str, bool], ...] = (
    ("blind_literals", False),
    ("strip_annotations", True),
    ("strip_docstrings", True),
    ("idioms", False),
    ("commutative", False),
    ("filter_boilerplate", False),
    ("consistent_renaming", False),
    ("abstract_expressions", False),
    ("blind_indexing", False),
    ("functions_only", False),
    ("class_level", False),
    ("sliding_window", False),
    ("clause_level", False),
    ("data_tables", False),
    ("harvest_closures", False),
    ("comprehensions", False),
    ("complex_expressions", False),
)


def _extract_calibration_settings(source: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts and normalizes harvesting flags and integer bounds from a source configuration dict."""
    settings: Dict[str, Any] = {}
    if not isinstance(source, dict):
        return settings
    for flag in (
        "bag_of_tokens",
        "call_sequences",
        "filter_stop_shingles",
        "audit_tests",
        "include_notebooks",
    ):
        settings[flag] = _safe_bool(source.get(flag, False))
    for flag, default_val in HARVEST_BOOLEAN_MODES:
        settings[flag] = _safe_bool(source.get(flag, default_val))
    for int_flag, default_int in (
        ("window_size", 5),
        ("min_expr_complexity", 4),
        ("min_lines", 8),
        ("min_tokens", 15),
    ):
        raw_val = source.get(int_flag, default_int)
        val = _safe_int(raw_val, min_val=1)
        settings[int_flag] = val if val is not None else default_int
    raw_mcs = source.get("min_corpus_size")
    if raw_mcs is None:
        raw_mcs = source.get("min_corpus_units")
    settings["min_corpus_size"] = _safe_int(raw_mcs, min_val=0)

    raw_mf = source.get("min_frequency")
    if raw_mf is None:
        raw_mf = source.get("min_calibration_frequency")
    if raw_mf is not None:
        mf_val = _safe_int(raw_mf, min_val=1)
        if mf_val is not None and mf_val > 1:
            settings["min_frequency"] = mf_val


    raw_excludes = source.get("excludes")
    if raw_excludes is None:
        raw_excludes = source.get("exclude")
    if raw_excludes and isinstance(raw_excludes, (list, tuple, set)):
        try:
            cleaned_excludes = sorted({
                str(x).replace("\\", "/").rstrip("\r\n").strip()
                for x in raw_excludes
                if str(x).strip()
            })
            settings["excludes"] = cleaned_excludes
        except (TypeError, ValueError):
            settings["excludes"] = []
    else:
        settings["excludes"] = []

    raw_scope = source.get("scope") or source.get("target_repo_relative")
    if raw_scope:
        norm_sc = normalize_lexical_posix(str(raw_scope)).strip("/")
        settings["scope"] = norm_sc if norm_sc else None
    else:
        settings["scope"] = None
    return settings


def _attach_calibration_flags(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    """Attaches feature mode flags and bounds from source dictionary to target calibration dict."""
    if isinstance(target, dict) and isinstance(source, dict):
        target.update(_extract_calibration_settings(source))


def _serialize_shingle_key(sh: Any) -> ShingleKeyWireFormat:
    """Serializes a shingle key into a type-tagged string representation for JSON storage.

    Wire Format Specification:
    --------------------------
    JSON dictionary keys in `corpus_calibration["shingle_frequencies"]` are serialized
    using an unambiguous 2-character type-tag prefix to prevent lossy type coercion
    (e.g., distinguishing integer AST codes from token strings, or tuple sequences from
    scalar strings):

      - `t:<json_array>`: Composite shingles (n-grams, token sequences, call traces)
        represented as JSON-encoded lists. Deserialization applies `_deep_tuple` with
        cycle detection (`seen` set) and recursion depth guard (`max_depth=20`) to
        reconstruct immutable, hashable, nested tuples matching in-memory shingle sets.
      - `s:<string>`: String tokens or shingles. If a string starts with any reserved
        tag prefix (`t:`, `s:`, `i:`, `f:`, `b:`) or `[`, it is escaped with `s:` to
        prevent collision with typed tags or legacy composite shingles. Unprefixed
        strings without reserved prefixes remain untouched.
      - `i:<int>`: Decimal integer values (e.g. numeric token IDs or statement types).
      - `f:<float>`: Finite floating-point values formatted via `str()`.
      - `b:<0|1>`: Booleans encoded as `b:1` (True) or `b:0` (False).
      - Legacy fallback: Untagged JSON arrays `[...]` are supported transparently
        during deserialization for backwards compatibility with earlier v1.x baselines.

    Args:
        sh: The raw shingle key (tuple, list, str, int, float, bool).

    Returns:
        A type-tagged ShingleKeyWireFormat string safe for JSON dictionary keys.
    """
    if isinstance(sh, (tuple, list)):
        try:
            return "t:" + json.dumps(list(sh))
        except (TypeError, ValueError):
            return "s:" + str(sh)
    if isinstance(sh, bool):
        return "b:" + ("1" if sh else "0")
    if isinstance(sh, int):
        return "i:" + str(sh)
    if isinstance(sh, float):
        return "f:" + str(sh)
    if isinstance(sh, str):
        if sh.startswith(("t:", "s:", "i:", "f:", "b:", "[")):
            return "s:" + sh
        return sh
    return str(sh)


def _deep_tuple(
    val: Any,
    depth: int = 0,
    max_depth: int = 20,
    seen: Optional[Set[int]] = None,
) -> Any:
    """Recursively converts nested lists and tuples into immutable, hashable tuples."""
    if depth > max_depth:
        raise ValueError(f"Exceeded maximum nesting depth of {max_depth}")
    if isinstance(val, (list, tuple)):
        if seen is None:
            seen = set()
        val_id = id(val)
        if val_id in seen:
            raise ValueError("Cyclic container detected")
        seen.add(val_id)
        try:
            return tuple(_deep_tuple(x, depth + 1, max_depth, seen) for x in val)
        except RecursionError as err:
            raise ValueError("Exceeded maximum recursion depth") from err
        finally:
            seen.remove(val_id)
    return val


def _deserialize_shingle_key(key: ShingleKeyWireFormat) -> Any:
    """Deserializes a JSON-compatible string key back into a shingle tuple or scalar.

    Decodes keys adhering to the wire format specification documented in
    `_serialize_shingle_key`, recovering immutable tuples for `t:<json_array>` and
    legacy `[...]` encodings, or type-coerced scalars for `s:`, `i:`, `f:`, and `b:` tags.
    """
    if not isinstance(key, str):
        return key

    if key.startswith("s:"):
        return key[2:]

    if key.startswith("i:"):
        try:
            return int(key[2:])
        except (ValueError, TypeError, OverflowError):
            return key[2:]

    if key.startswith("f:"):
        try:
            val = float(key[2:])
            if math.isfinite(val):
                return val
        except (ValueError, TypeError, OverflowError):
            pass
        return key[2:]

    if key.startswith("b:"):
        return key[2:] in ("1", "True", "true")

    # Tagged tuple "t:[...]" or legacy untagged JSON tuple "[...]"
    raw_json: Optional[str] = None
    if key.startswith("t:"):
        raw_json = key[2:]
    elif key.startswith("[") and key.endswith("]"):
        raw_json = key

    if raw_json is not None:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, (list, tuple)):
                return _deep_tuple(parsed)
        except (json.JSONDecodeError, ValueError, TypeError, RecursionError):
            pass

    return key


def _sanitize_shingle_frequency_dict(
    raw_freqs: Any,
    key_transform: Callable[[Any], Any],
    max_units: Optional[int] = None,
) -> Dict[Any, int]:
    """Sanitizes raw shingle frequencies by filtering positive integers and transforming keys."""
    cleaned: Dict[Any, int] = {}
    if not isinstance(raw_freqs, dict):
        return cleaned
    for k, v in raw_freqs.items():
        try:
            if isinstance(v, bool):
                continue
            freq_val = int(v)
            if freq_val > 0:
                if max_units is not None and max_units > 0:
                    freq_val = min(max_units, freq_val)
                cleaned[key_transform(k)] = freq_val
        except (ValueError, TypeError, OverflowError, RecursionError):
            continue
    return cleaned


def _safe_index_frequency(raw_freq: Any) -> Optional[float]:
    """Validates and clamps an index frequency to a finite float in (0.0, 1.0], or None."""
    if raw_freq is None or isinstance(raw_freq, bool):
        return None
    try:
        val = float(raw_freq)
        if math.isfinite(val) and 0.0 < val <= 1.0:
            return val
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _safe_str(val: Any) -> Optional[str]:
    """Validates that a value is a non-empty string."""
    if isinstance(val, str):
        cleaned = val.strip()
        return cleaned if cleaned else None
    return None


def _safe_hex_hash(val: Any, allowed_lengths: Tuple[int, ...] = (64,)) -> Optional[str]:
    """Validates that a value is a valid lowercase hexadecimal string of specific length(s)."""
    s = _safe_str(val)
    if s and len(s) in allowed_lengths and all(c in "0123456789abcdefABCDEF" for c in s):
        return s.lower()
    return None


def compute_calibration_config_hash(source: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash for a set of calibration or scan configuration settings."""
    settings = _extract_calibration_settings(source)
    if "max_index_frequency" in source:
        raw_mif = source.get("max_index_frequency")
        if raw_mif is None:
            settings["max_index_frequency"] = None
        else:
            parsed_mif = _safe_index_frequency(raw_mif)
            settings["max_index_frequency"] = parsed_mif if parsed_mif is not None else 0.25
    else:
        settings["max_index_frequency"] = 0.25
    canonical_json = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


CALIBRATION_REDUCTION_METRIC_KEYS: Tuple[str, ...] = (
    "raw_shingle_count",
    "unpruned_shingle_count",
    "retained_shingle_count",
    "pruned_shingle_count",
    "shingle_reduction_ratio",
    "shingle_compression_ratio",
    "raw_bytes_estimate",
    "retained_bytes_estimate",
    "bytes_saved_estimate",
    "bytes_reduction_ratio",
    "bytes_compression_ratio",
)


def _compute_calibration_reduction_metrics(
    raw_shingle_count: int,
    retained_shingle_count: int,
    raw_bytes_est: int,
    retained_bytes_est: int,
) -> Dict[str, Any]:
    """Computes compression and pruning metrics for corpus calibration."""
    pruned_count = max(0, raw_shingle_count - retained_shingle_count)
    shingle_reduction_ratio = (
        round(pruned_count / raw_shingle_count, 4) if raw_shingle_count > 0 else 0.0
    )
    shingle_compression_ratio = (
        round(raw_shingle_count / retained_shingle_count, 2)
        if retained_shingle_count > 0
        else 1.0
    )
    bytes_reduction_ratio = (
        round((raw_bytes_est - retained_bytes_est) / raw_bytes_est, 4)
        if raw_bytes_est > 0
        else 0.0
    )
    bytes_compression_ratio = (
        round(raw_bytes_est / retained_bytes_est, 2)
        if retained_bytes_est > 0
        else 1.0
    )
    bytes_saved_est = max(0, raw_bytes_est - retained_bytes_est)
    return {
        "raw_shingle_count": raw_shingle_count,
        "unpruned_shingle_count": raw_shingle_count,
        "retained_shingle_count": retained_shingle_count,
        "pruned_shingle_count": pruned_count,
        "shingle_reduction_ratio": shingle_reduction_ratio,
        "shingle_compression_ratio": shingle_compression_ratio,
        "raw_bytes_estimate": raw_bytes_est,
        "retained_bytes_estimate": retained_bytes_est,
        "bytes_saved_estimate": bytes_saved_est,
        "bytes_reduction_ratio": bytes_reduction_ratio,
        "bytes_compression_ratio": bytes_compression_ratio,
    }


def _extract_finite_numeric_metrics(
    source: Optional[Dict[str, Any]],
) -> Dict[str, Union[int, float]]:
    """Extracts finite numeric reduction metrics from a calibration dictionary."""
    metrics: Dict[str, Union[int, float]] = {}
    if not isinstance(source, dict):
        return metrics
    for k in CALIBRATION_REDUCTION_METRIC_KEYS:
        if k in source:
            v = source[k]
            if isinstance(v, (int, float)) and math.isfinite(v):
                metrics[k] = v
    return metrics


def _attach_reduction_metrics(
    target: Dict[str, Any],
    source: Optional[Dict[str, Any]],
) -> None:
    """Attaches sanitized finite reduction metrics and reduction_stats sub-dict to target calibration."""
    if not isinstance(source, dict):
        return
    extracted = _extract_finite_numeric_metrics(source)
    if extracted:
        target.update(extracted)
    raw_sub = source.get("reduction_stats")
    if isinstance(raw_sub, dict):
        sub_metrics = _extract_finite_numeric_metrics(raw_sub)
        if sub_metrics:
            target["reduction_stats"] = sub_metrics


def compute_corpus_calibration(
    units: Sequence[Dict[str, Any]],
    max_index_frequency: Optional[float] = 0.25,
    min_corpus_size: Optional[int] = None,
    filter_stop_shingles: bool = False,
    stop_shingles: Optional[Set[Any]] = None,
    *,
    min_frequency: int = 1,
    bag_of_tokens: bool = False,
    call_sequences: bool = False,
    excludes: Optional[Sequence[str]] = None,
    scope: Optional[str] = None,
    target_repo_relative: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Computes global shingle document frequencies and calibrated stop-shingles for a repository corpus.

    Args:
        units: Sequence of harvested AST code units.
        max_index_frequency: Maximum corpus frequency threshold for stop shingles.
        min_corpus_size: Minimum units required to activate stop-shingle filtering.
        filter_stop_shingles: Whether stop-shingle filtering is active.
        stop_shingles: Accepted for signature parity; ignored to record unskewed empirical stats.
        min_frequency: Minimum document frequency threshold to retain in calibration.
            Defaults to 1 (all shingles retained). In large monorepos (>100,000 units),
            setting min_frequency >= 2 prunes singleton shingles, keeping baseline JSON compact.
        bag_of_tokens: Whether bag-of-tokens indexing is enabled.
        call_sequences: Whether call sequence indexing is enabled.
        excludes: Ignored directory patterns.
        scope: Target scope relative to repository worktree root.
        target_repo_relative: Worktree-relative path of target directory.

    Note:
        The stop_shingles argument is accepted for signature parity with scanner
        pipelines, but is deliberately not incorporated into corpus calibration so
        calibrations record unskewed empirical document frequency statistics.
    """
    _ = stop_shingles
    if min_frequency == 1 and "min_calibration_frequency" in kwargs:
        mf_cand = _safe_int(kwargs.get("min_calibration_frequency"), min_val=1)
        if mf_cand is not None:
            min_frequency = mf_cand
    total_units = len(units)
    shingle_frequencies: Dict[Any, int] = {}
    for u in units:
        if call_sequences:
            keys = set(u.get("calls") or ())
        elif bag_of_tokens:
            keys = set(u.get("vector") or ())
        elif "shingles" in u and u["shingles"]:
            keys = set(u["shingles"])
        elif "vector" in u and u["vector"]:
            keys = set(u["vector"].keys() if hasattr(u["vector"], "keys") else u["vector"])
        elif "calls" in u and u["calls"]:
            keys = set(u["calls"])
        else:
            keys = set()
        for sh in keys:
            shingle_frequencies[sh] = shingle_frequencies.get(sh, 0) + 1

    effective_min_corpus = _safe_min_corpus(min_corpus_size, filter_stop_shingles)
    global_stop_shingles: Set[Any] = set()

    valid_max_freq = _safe_index_frequency(max_index_frequency)

    if valid_max_freq is not None and total_units >= effective_min_corpus:
        try:
            posting_float = total_units * valid_max_freq
            if math.isfinite(posting_float):
                max_posting_len = max(2, int(math.ceil(posting_float)))
                for sh, count in shingle_frequencies.items():
                    if count > max_posting_len:
                        global_stop_shingles.add(sh)
        except (ValueError, TypeError, OverflowError):
            pass

    raw_shingle_count = len(shingle_frequencies)
    raw_bytes_est = sum(
        len(str(k)) + len(str(c)) + 8 for k, c in shingle_frequencies.items()
    )
    if min_frequency > 1:
        shingle_frequencies = {
            sh: count for sh, count in shingle_frequencies.items() if count >= min_frequency
        }
    retained_shingle_count = len(shingle_frequencies)
    retained_bytes_est = (
        raw_bytes_est
        if min_frequency <= 1
        else sum(len(str(k)) + len(str(c)) + 8 for k, c in shingle_frequencies.items())
    )

    reduction_metrics = _compute_calibration_reduction_metrics(
        raw_shingle_count, retained_shingle_count, raw_bytes_est, retained_bytes_est
    )

    calib: Dict[str, Any] = {
        "total_units": total_units,
        "max_index_frequency": valid_max_freq,
        "global_stop_shingles": global_stop_shingles,
        "shingle_frequencies": shingle_frequencies,
        "min_frequency": min_frequency,
        "reduction_stats": dict(reduction_metrics),
    }
    calib.update(reduction_metrics)
    flags_dict: Dict[str, Any] = {
        "bag_of_tokens": bag_of_tokens,
        "call_sequences": call_sequences,
        "filter_stop_shingles": filter_stop_shingles,
        "min_corpus_size": min_corpus_size,
        "min_frequency": min_frequency,
        "excludes": excludes,
        "scope": scope,
        "target_repo_relative": target_repo_relative,
    }
    flags_dict.update(kwargs)
    _attach_calibration_flags(calib, flags_dict)
    calib["config_hash"] = compute_calibration_config_hash(calib)
    return calib
