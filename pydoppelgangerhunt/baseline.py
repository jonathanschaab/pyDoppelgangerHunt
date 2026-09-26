"""Grandfathered clone baseline recording, loading, and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.canonical_path import (
    CanonicalPathResolver,
    lexical_relative_to,
    normalize_lexical_posix,
)
from pydoppelgangerhunt.config import (
    canonical_path_key,
    find_matching_path_value,
    normalize_path_string,
    paths_match_boundary,
)
from . import git_diff

logger = logging.getLogger(__name__)


# ==============================================================================
# Semantic Type Aliases for Fingerprints, Hashes, and Calibration Shingle Keys
# ==============================================================================
UnitStructuralHash = str
PureStructuralFingerprint = str
NamespacedStructuralFingerprint = str
ClonePairStructuralFingerprint = str
ClonePairFingerprint = str
ShingleKeyWireFormat = str


MAX_CALIBRATION_UNITS: int = 1_000_000_000


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

    if min_frequency > 1:
        shingle_frequencies = {
            sh: count for sh, count in shingle_frequencies.items() if count >= min_frequency
        }

    calib: Dict[str, Any] = {
        "total_units": total_units,
        "max_index_frequency": valid_max_freq,
        "global_stop_shingles": global_stop_shingles,
        "shingle_frequencies": shingle_frequencies,
        "min_frequency": min_frequency,
    }
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



def compute_unit_structural_hash(unit: Dict[str, Any]) -> UnitStructuralHash:
    """Computes deterministic structural content hash for an AST unit."""
    if "structural_hash" in unit and unit["structural_hash"]:
        return str(unit["structural_hash"])
    tokens = unit.get("tokens", [])
    # Truncating SHA-256 to 16 hex characters provides 64 bits of entropy (keyspace: 2^64 ≈ 1.84e19).
    # Per the Birthday Paradox, the collision probability P for N items is approximately P ≈ N^2 / (2 * 2^64).
    # For a large repository with N = 100,000 units, P ≈ 2.7e-10 (< 1 in 3.7 billion).
    # Even for N = 1,000,000 units, P ≈ 2.7e-8 (< 1 in 37 million). A 50% collision threshold requires ~5.06 billion units.
    if tokens:
        return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]
    vec = unit.get("vector", {})
    sorted_vec = json.dumps(vec, sort_keys=True)
    return hashlib.sha256(sorted_vec.encode("utf-8")).hexdigest()[:16]


def _format_paired_endpoints(ep1: str, ep2: str) -> str:
    """Formats two endpoints into an order-invariant clone pair representation."""
    ordered = sorted([ep1, ep2])
    return f"{ordered[0]} <===> {ordered[1]}"


def extract_unit_namespace(file_path: str) -> str:
    """Extracts canonical module/package directory namespace from a file path."""
    norm_path = canonical_path_key(file_path, strip_anchor=False)
    if "/" not in norm_path:
        return "."
    parent = norm_path.rsplit("/", 1)[0]
    return parent if parent else "."


def _paired_hashed_fingerprint(
    u1: Dict[str, Any], u2: Dict[str, Any], prefix1: str, prefix2: str
) -> str:
    h1 = compute_unit_structural_hash(u1)
    h2 = compute_unit_structural_hash(u2)
    return _format_paired_endpoints(f"{prefix1}#{h1}", f"{prefix2}#{h2}")


def namespaced_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> NamespacedStructuralFingerprint:
    """Computes order-invariant structural content fingerprint bound to module/package namespaces."""
    ns1 = extract_unit_namespace(str(u1.get("file") or ""))
    ns2 = extract_unit_namespace(str(u2.get("file") or ""))
    return _paired_hashed_fingerprint(u1, u2, ns1, ns2)


def pure_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> PureStructuralFingerprint:
    """Computes path-independent, order-invariant structural content fingerprint for a clone pair."""
    return _format_paired_endpoints(compute_unit_structural_hash(u1), compute_unit_structural_hash(u2))


def clone_pair_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> ClonePairStructuralFingerprint:
    """Computes order-invariant structural content fingerprint for a clone pair."""
    f1 = canonical_path_key(str(u1.get("file") or ""), strip_anchor=False)
    f2 = canonical_path_key(str(u2.get("file") or ""), strip_anchor=False)
    return _paired_hashed_fingerprint(u1, u2, f1, f2)


def clone_pair_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> ClonePairFingerprint:
    """Computes stable, order-invariant fingerprint for a clone pair."""
    f1 = canonical_path_key(str(u1.get("file") or ""), strip_anchor=False)
    f2 = canonical_path_key(str(u2.get("file") or ""), strip_anchor=False)
    n1 = str(u1.get("name") or "unit1")
    n2 = str(u2.get("name") or "unit2")
    return _format_paired_endpoints(f"{f1}:{n1}", f"{f2}:{n2}")


def record_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_path: str,
    target: str,
    threshold: float,
    *,
    corpus_calibration: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    clone_basis: Optional[str] = None,
) -> str:
    """Records detected clones into a JSON baseline file for grandfathering."""
    target_repo_rel = None
    res = _build_baseline_path_resolver(root=repo_root, target=target)
    if res is not None:
        target_repo_rel = res.target_in_repo

    active_clone_basis = _detect_clone_path_basis(
        clones,
        target_repo_rel,
        explicit_basis=clone_basis,
        repo_root=repo_root,
        target=target,
    )

    fingerprints: List[Dict[str, Any]] = []
    for item in clones:
        sim = 1.0
        u1: Dict[str, Any] = {}
        u2: Dict[str, Any] = {}
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            sim = float(item[0])
            u1 = item[1] if isinstance(item[1], dict) else {}
            u2 = item[2] if isinstance(item[2], dict) else {}
        elif isinstance(item, dict):
            sim = float(item.get("similarity", 1.0))
            u1 = item.get("u1") or item.get("unit_a") or item
            u2 = item.get("u2") or item.get("unit_b") or {}

        fa_raw = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
        fb_raw = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)

        fa_target = fa_raw
        fb_target = fb_raw
        if res is not None:
            resolved_a = res.resolve(fa_raw, basis=active_clone_basis)
            if resolved_a.target_relative:
                fa_target = resolved_a.target_relative
            resolved_b = res.resolve(fb_raw, basis=active_clone_basis)
            if resolved_b.target_relative:
                fb_target = resolved_b.target_relative
        elif target_repo_rel and active_clone_basis == "repo_relative":
            rel_a = lexical_relative_to(fa_raw, target_repo_rel)
            if rel_a:
                fa_target = rel_a
            rel_b = lexical_relative_to(fb_raw, target_repo_rel)
            if rel_b:
                fb_target = rel_b
        elif target is not None:
            t_norm = normalize_lexical_posix(str(target))
            rel_a = lexical_relative_to(fa_raw, t_norm)
            if rel_a:
                fa_target = rel_a
            rel_b = lexical_relative_to(fb_raw, t_norm)
            if rel_b:
                fb_target = rel_b

        u1_rec = dict(u1, file=fa_target)
        u2_rec = dict(u2, file=fb_target)

        fingerprints.append({
            "fingerprint": clone_pair_fingerprint(u1_rec, u2_rec),
            "structural_fingerprint": clone_pair_structural_fingerprint(u1_rec, u2_rec),
            "namespaced_structural_fingerprint": namespaced_structural_fingerprint(u1_rec, u2_rec),
            "pure_structural_fingerprint": pure_structural_fingerprint(u1_rec, u2_rec),
            "similarity": round(sim, 4),
            "file_a": fa_target,
            "name_a": str(u1.get("name") or "unit1"),
            "hash_a": compute_unit_structural_hash(u1),
            "namespace_a": extract_unit_namespace(fa_target),
            "file_b": fb_target,
            "name_b": str(u2.get("name") or "unit2"),
            "hash_b": compute_unit_structural_hash(u2),
            "namespace_b": extract_unit_namespace(fb_target),
        })

    data: Dict[str, Any] = {
        "version": "1.5.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "path_basis": "target_relative",
        "threshold": threshold,
        "clone_count": len(fingerprints),
        "fingerprints": fingerprints,
    }
    target_p = Path(baseline_path)
    if target_repo_rel:
        data["target_repo_relative"] = target_repo_rel
    probe_target = repo_root or target
    probe_dir: Optional[Union[str, Path]] = None
    try:
        t_p = Path(probe_target) if probe_target is not None else None
        probe_dir = (
            t_p.parent
            if (t_p and (t_p.is_file() or (not t_p.is_dir() and t_p.suffix.lower() in (".py", ".ipynb"))))
            else t_p
        )
    except (ValueError, OSError, RuntimeError):
        probe_dir = probe_target
    head_commit = git_diff.get_git_head_commit(repo_root=probe_dir)
    if head_commit:
        data["recorded_commit"] = head_commit
    if isinstance(corpus_calibration, dict):
        total_units_val = _safe_total_units(corpus_calibration.get("total_units"))

        parsed_freq = _safe_index_frequency(corpus_calibration.get("max_index_frequency"))
        safe_max_freq: Optional[float] = parsed_freq

        safe_stops: List[Any] = []
        raw_stops = corpus_calibration.get("global_stop_shingles", [])
        if isinstance(raw_stops, (list, set, tuple)):
            for sh in raw_stops:
                try:
                    elem = list(sh) if isinstance(sh, (tuple, list)) else sh
                    try:
                        json.dumps(elem)
                        safe_stops.append(elem)
                    except (TypeError, ValueError):
                        safe_stops.append(str(elem))
                except (TypeError, ValueError, RecursionError):
                    continue
        safe_stops.sort(key=_serialize_shingle_key)

        safe_shingle_freqs = _sanitize_shingle_frequency_dict(
            corpus_calibration.get("shingle_frequencies", {}),
            _serialize_shingle_key,
            max_units=total_units_val,
        )

        calib_entry: Dict[str, Any] = {
            "total_units": total_units_val,
            "max_index_frequency": safe_max_freq,
            "global_stop_shingles": safe_stops,
            "shingle_frequencies": dict(sorted(safe_shingle_freqs.items(), key=lambda item: item[0])),
        }
        _attach_calibration_flags(calib_entry, corpus_calibration)
        if target_repo_rel:
            calib_entry["target_repo_relative"] = target_repo_rel
            calib_entry["scope"] = target_repo_rel
        elif not calib_entry.get("scope"):
            raw_sc = corpus_calibration.get("scope") or corpus_calibration.get("target_repo_relative")
            if raw_sc:
                norm_sc = normalize_lexical_posix(str(raw_sc)).strip("/")
                calib_entry["scope"] = norm_sc if norm_sc else None
                calib_entry["target_repo_relative"] = norm_sc if norm_sc else None
        if target:
            calib_entry["target"] = target
        calib_config_hash = compute_calibration_config_hash(calib_entry)
        calib_entry["config_hash"] = calib_config_hash
        if head_commit:
            calib_entry["recorded_commit"] = head_commit
        data["corpus_calibration"] = calib_entry
        data["config_hash"] = calib_config_hash
    target_p.parent.mkdir(parents=True, exist_ok=True)
    target_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(target_p)


class BaselineFingerprints(set):  # type: ignore[type-arg]
    """Set of baseline fingerprints with structured record metadata for granular disambiguation."""

    def __init__(
        self,
        fps: Optional[Set[str]] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        corpus_calibration: Optional[Dict[str, Any]] = None,
        path_basis: Optional[str] = None,
        target_repo_relative: Optional[str] = None,
        config_hash: Optional[str] = None,
        recorded_commit: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        super().__init__(fps or set())
        self.records: List[Dict[str, Any]] = records or []
        self.corpus_calibration: Optional[Dict[str, Any]] = corpus_calibration
        self.path_basis: Optional[str] = path_basis
        self.target_repo_relative: Optional[str] = target_repo_relative
        self.config_hash: Optional[str] = config_hash
        self.recorded_commit: Optional[str] = recorded_commit
        self.target: Optional[str] = target


def _parse_legacy_fingerprint_record(raw_fp: str) -> Dict[str, Any]:
    """Parses a legacy string fingerprint into a baseline record dictionary."""
    rec: Dict[str, Any] = {"fingerprint": raw_fp}
    parts = raw_fp.split(" <===> ")
    if len(parts) == 2:
        for idx, suffix in enumerate(["a", "b"]):
            if ":" in parts[idx]:
                f_val, n_val = parts[idx].rsplit(":", 1)
                rec[f"file_{suffix}"] = f_val
                rec[f"name_{suffix}"] = n_val
    return rec


def load_baseline(baseline_path: str) -> BaselineFingerprints:
    """Loads grandfathered clone fingerprints and structural hashes from a JSON baseline file."""
    path = Path(baseline_path)
    if not path.exists():
        return BaselineFingerprints()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return BaselineFingerprints()
        fingerprints = data.get("fingerprints", [])
        fps: Set[str] = set()
        records: List[Dict[str, Any]] = []
        for raw_item in fingerprints:
            if isinstance(raw_item, str):
                item_str = str(raw_item).strip()
                if item_str:
                    fps.add(item_str)
                    records.append(_parse_legacy_fingerprint_record(item_str))
                continue
            if not isinstance(raw_item, dict):
                continue
            item = raw_item
            rec = dict(item)
            if "fingerprint" in item:
                fp_str = str(item["fingerprint"])
                if fp_str:
                    fps.add(fp_str)
                parts = fp_str.split(" <===> ")
                if len(parts) == 2:
                    for idx, suffix in enumerate(["a", "b"]):
                        if ":" in parts[idx]:
                            f_val, n_val = parts[idx].rsplit(":", 1)
                            rec.setdefault(f"file_{suffix}", f_val)
                            rec.setdefault(f"name_{suffix}", n_val)

            if "structural_fingerprint" in item and item["structural_fingerprint"]:
                fps.add(item["structural_fingerprint"])
            elif "hash_a" in item and "hash_b" in item and item["hash_a"] and item["hash_b"]:
                f_a = canonical_path_key(str(rec.get("file_a") or ""), strip_anchor=False)
                f_b = canonical_path_key(str(rec.get("file_b") or ""), strip_anchor=False)
                sfp = _format_paired_endpoints(f"{f_a}#{item['hash_a']}", f"{f_b}#{item['hash_b']}")
                rec["structural_fingerprint"] = sfp
                fps.add(sfp)

            if "namespaced_structural_fingerprint" in item and item["namespaced_structural_fingerprint"]:
                fps.add(item["namespaced_structural_fingerprint"])
            elif "hash_a" in item and "hash_b" in item and item["hash_a"] and item["hash_b"]:
                ns_a = item.get("namespace_a") or extract_unit_namespace(str(rec.get("file_a") or ""))
                ns_b = item.get("namespace_b") or extract_unit_namespace(str(rec.get("file_b") or ""))
                rec["namespace_a"] = ns_a
                rec["namespace_b"] = ns_b
                ns_sfp = _format_paired_endpoints(f"{ns_a}#{item['hash_a']}", f"{ns_b}#{item['hash_b']}")
                rec["namespaced_structural_fingerprint"] = ns_sfp
                fps.add(ns_sfp)

            if "pure_structural_fingerprint" in item and item["pure_structural_fingerprint"]:
                pure_sfp = str(item["pure_structural_fingerprint"])
                fps.add(pure_sfp)
                if ("hash_a" not in rec or "hash_b" not in rec) and " <===> " in pure_sfp:
                    h_parts = pure_sfp.split(" <===> ")
                    if len(h_parts) == 2:
                        rec.setdefault("hash_a", h_parts[0])
                        rec.setdefault("hash_b", h_parts[1])
            elif "hash_a" in item and "hash_b" in item and item["hash_a"] and item["hash_b"]:
                pure_sfp = _format_paired_endpoints(str(item["hash_a"]), str(item["hash_b"]))
                rec["pure_structural_fingerprint"] = pure_sfp
                fps.add(pure_sfp)

            records.append(rec)

        raw_calib = data.get("corpus_calibration")
        corpus_calibration: Optional[Dict[str, Any]] = None
        calib_cfg_hash: Optional[str] = None
        calib_commit: Optional[str] = None
        if isinstance(raw_calib, dict):
            raw_stops = raw_calib.get("global_stop_shingles", [])
            decoded_stops: Set[Any] = set()
            if isinstance(raw_stops, (list, set, tuple)):
                for sh in raw_stops:
                    try:
                        decoded_stops.add(_deep_tuple(sh))
                    except (TypeError, ValueError, RecursionError):
                        continue
            calib_total_units = _safe_total_units(raw_calib.get("total_units"))
            raw_freqs = raw_calib.get("shingle_frequencies")
            decoded_freqs: Optional[Dict[Any, int]] = None
            if isinstance(raw_freqs, dict):
                decoded_freqs = _sanitize_shingle_frequency_dict(
                    raw_freqs,
                    _deserialize_shingle_key,
                    max_units=calib_total_units,
                )
            raw_max_freq = raw_calib.get("max_index_frequency")
            max_idx_freq: Optional[float]
            if raw_max_freq is None:
                max_idx_freq = None
            else:
                parsed_max = _safe_index_frequency(raw_max_freq)
                max_idx_freq = parsed_max if parsed_max is not None else 0.25

            raw_hash = raw_calib.get("config_hash")
            calib_cfg_hash = _safe_hex_hash(raw_hash, (64,))
            if not calib_cfg_hash:
                calib_cfg_hash = compute_calibration_config_hash(raw_calib)

            raw_commit = raw_calib.get("recorded_commit")
            calib_commit = _safe_hex_hash(raw_commit, (40, 64))

            corpus_calibration = {
                "total_units": calib_total_units,
                "max_index_frequency": max_idx_freq,
                "global_stop_shingles": decoded_stops,
                "shingle_frequencies": decoded_freqs,
                "config_hash": calib_cfg_hash,
            }
            if calib_commit:
                corpus_calibration["recorded_commit"] = calib_commit
            _attach_calibration_flags(corpus_calibration, raw_calib)

        top_cfg_hash = calib_cfg_hash or _safe_hex_hash(data.get("config_hash"), (64,))
        top_commit = calib_commit or _safe_hex_hash(data.get("recorded_commit"), (40, 64))

        raw_path_basis = data.get("path_basis")
        target_repo_rel = data.get("target_repo_relative")
        top_target = _safe_str(data.get("target"))

        if raw_path_basis:
            path_basis = (
                "repo_relative"
                if raw_path_basis in ("repo", "repo_relative", "worktree_relative")
                else "target_relative"
            )
        else:
            base_offset_inferred = None
            if target_repo_rel:
                base_offset_inferred = target_repo_rel
            elif top_target:
                base_offset_inferred = normalize_lexical_posix(top_target).strip("./").rstrip("/")
            path_basis = _detect_clone_path_basis(
                records or [],
                base_offset_inferred,
                repo_root=path.parent,
                target=top_target,
            )

        if corpus_calibration is not None and isinstance(corpus_calibration, dict):
            if target_repo_rel and not corpus_calibration.get("target_repo_relative"):
                corpus_calibration["target_repo_relative"] = target_repo_rel
            if target_repo_rel and not corpus_calibration.get("scope"):
                corpus_calibration["scope"] = target_repo_rel
            elif not corpus_calibration.get("scope") and corpus_calibration.get("target_repo_relative"):
                corpus_calibration["scope"] = corpus_calibration.get("target_repo_relative")
            if "target" not in corpus_calibration and top_target:
                corpus_calibration["target"] = top_target
            recomputed_hash = compute_calibration_config_hash(corpus_calibration)
            if not calib_cfg_hash or calib_cfg_hash != recomputed_hash:
                calib_cfg_hash = recomputed_hash
                corpus_calibration["config_hash"] = recomputed_hash
                top_cfg_hash = recomputed_hash
        return BaselineFingerprints(
            fps,
            records=records,
            corpus_calibration=corpus_calibration,
            path_basis=path_basis,
            target_repo_relative=target_repo_rel,
            config_hash=top_cfg_hash,
            recorded_commit=top_commit,
            target=top_target,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError, RecursionError) as err:
        logger.debug("Failed to load baseline at %s: %s", baseline_path, err, exc_info=True)
        return BaselineFingerprints()



def _is_absolute_path_str(path_str: str) -> bool:
    """Returns True if the path string represents an absolute POSIX or Windows path."""
    if not path_str:
        return False
    return (
        (
            len(path_str) >= 3
            and path_str[1] == ":"
            and path_str[0].isalpha()
            and path_str[2] in ("/\\")
        )
        or path_str.startswith("/")
        or path_str.startswith("\\")
    )


def _canonicalize_endpoint_path(
    path: str,
    offset: Optional[str],
    path_basis: Optional[str] = "target_relative",
) -> str:
    """Canonicalizes an endpoint path relative to a repository root given an optional target offset."""
    norm = normalize_lexical_posix(path, strip_anchor=False)
    if not norm:
        return ""
    if path_basis in ("repo_relative", "worktree_relative", "repo"):
        return norm
    if _is_absolute_path_str(norm):
        return norm
    if offset:
        off_norm = normalize_lexical_posix(offset, strip_anchor=False).strip("/")
        if off_norm:
            return f"{off_norm}/{norm}"
    return norm


def _compute_path_offset(
    sub_path: Optional[Union[str, Path]], root_path: Optional[Union[str, Path]]
) -> Optional[str]:
    """Computes normalized relative offset between a sub path and root path if distinct."""
    if sub_path is None or root_path is None:
        return None
    try:
        sub_p = Path(sub_path)
        root_p = Path(root_path)
        if not sub_p.is_absolute():
            sub_cand = root_p / sub_p
            try:
                if sub_cand.exists() or root_p.is_absolute():
                    sub_p = sub_cand
            except (ValueError, OSError, RuntimeError):
                pass
        try:
            sub_res = sub_p.resolve()
            root_res = root_p.resolve()
        except (ValueError, OSError, RuntimeError):
            sub_res = sub_p
            root_res = root_p

        try:
            if sub_res.is_file() or (not sub_res.is_dir() and sub_p.suffix.lower() in (".py", ".ipynb")):
                sub_res = sub_res.parent
                sub_p = sub_p.parent
            if root_res.is_file() or (not root_res.is_dir() and root_p.suffix.lower() in (".py", ".ipynb")):
                root_res = root_res.parent
                root_p = root_p.parent
        except (ValueError, OSError, RuntimeError):
            pass

        sub_norm = normalize_lexical_posix(str(sub_res))
        root_norm = normalize_lexical_posix(str(root_res))
        if sub_norm and root_norm and sub_norm != root_norm:
            rel = lexical_relative_to(sub_norm, root_norm)
            if rel is not None:
                return rel.strip("/")

        raw_sub = normalize_lexical_posix(str(sub_p))
        raw_root = normalize_lexical_posix(str(root_p))
        if raw_sub and raw_root and raw_sub != raw_root:
            rel = lexical_relative_to(raw_sub, raw_root)
            if rel is not None:
                return rel.strip("/")
    except (ValueError, TypeError):
        pass
    return None


def _derive_target_offsets(
    base_target: Optional[Union[str, Path]],
    base_target_rel: Optional[str],
    repo_root: Optional[Union[str, Path]],
    target: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Derives normalized baseline and active scan offsets relative to repo root."""
    effective_repo = repo_root
    git_root = None
    try:
        cand = target or repo_root or base_target
        if cand:
            c_p = Path(cand)
            cand_dir: Union[str, Path] = (
                c_p.parent
                if (c_p.is_file() or (not c_p.is_dir() and c_p.suffix.lower() in (".py", ".ipynb")))
                else c_p
            )
            git_root = git_diff.get_git_repo_root(repo_root=cand_dir)
            if git_root:
                effective_repo = git_root
    except (ValueError, OSError, RuntimeError, TypeError):
        pass

    if not git_root and base_target is not None and target is not None:
        try:
            t_p = Path(target)
            t_p_parent = t_p.parent if (t_p.is_file() or (not t_p.is_dir() and t_p.suffix.lower() in (".py", ".ipynb"))) else t_p
        except (ValueError, OSError, RuntimeError):
            t_p_parent = Path(target)
        try:
            b_p = Path(base_target)
            b_p_parent = b_p.parent if (b_p.is_file() or (not b_p.is_dir() and b_p.suffix.lower() in (".py", ".ipynb"))) else b_p
        except (ValueError, OSError, RuntimeError):
            b_p_parent = Path(base_target)

        t_norm = normalize_lexical_posix(str(t_p_parent))
        b_norm = normalize_lexical_posix(str(b_p_parent))
        if t_norm and b_norm and t_norm != b_norm:
            if lexical_relative_to(t_norm, b_norm) is not None:
                effective_repo = base_target
            elif lexical_relative_to(b_norm, t_norm) is not None:
                effective_repo = target

    base_offset = (
        normalize_lexical_posix(str(base_target_rel)).strip("/")
        if base_target_rel
        else _compute_path_offset(base_target, effective_repo)
    )
    scan_offset = _compute_path_offset(target, effective_repo)
    return base_offset, scan_offset


def _parse_structural_fingerprint(sfp: str) -> Tuple[str, str, str, str]:
    """Parses a paired structural fingerprint string into (file_a, hash_a, file_b, hash_b)."""
    parts = sfp.split(" <===> ")
    if len(parts) == 2 and "#" in parts[0] and "#" in parts[1]:
        f_a, h_a = parts[0].rsplit("#", 1)
        f_b, h_b = parts[1].rsplit("#", 1)
        return f_a, h_a, f_b, h_b
    return "", "", "", ""


def _extract_record_endpoint_data(
    item: Dict[str, Any],
) -> Tuple[str, str, str, str, str, str]:
    """Extracts (file_a, name_a, hash_a, file_b, name_b, hash_b) from a clone or baseline dictionary."""
    fa = str(item.get("file_a") or "")
    fb = str(item.get("file_b") or "")
    na = str(item.get("name_a") or "")
    nb = str(item.get("name_b") or "")
    ha = str(item.get("hash_a") or item.get("structural_hash_a") or item.get("structural_hash") or "")
    hb = str(item.get("hash_b") or item.get("structural_hash_b") or item.get("structural_hash") or "")

    raw_fp = item.get("fingerprint") or item.get("fp")
    if not fa and not fb and raw_fp:
        parsed = _parse_legacy_fingerprint_record(str(raw_fp))
        fa = str(parsed.get("file_a") or "")
        fb = str(parsed.get("file_b") or "")
        na = str(parsed.get("name_a") or "")
        nb = str(parsed.get("name_b") or "")

    raw_sfp = item.get("structural_fingerprint") or item.get("sfp")
    if (not ha or not hb) and raw_sfp:
        p_fa, p_ha, p_fb, p_hb = _parse_structural_fingerprint(str(raw_sfp))
        if not fa and not fb:
            fa, fb = p_fa, p_fb
        if fa != fb and fa == p_fb and fb == p_fa:
            parsed_ha, parsed_hb = p_hb, p_ha
        else:
            parsed_ha, parsed_hb = p_ha, p_hb
        if not ha:
            ha = parsed_ha
        if not hb:
            hb = parsed_hb

    return fa, na, ha, fb, nb, hb


def _get_rec_repo_data(
    rec: Dict[str, Any],
    base_offset: Optional[str],
    path_basis: Optional[str] = "target_relative",
) -> Tuple[str, str, str, str, str, List[str]]:
    """Derives canonical repository-relative path and fingerprint representation for a baseline record."""
    r_fa, r_na, r_ha, r_fb, r_nb, r_hb = _extract_record_endpoint_data(rec)
    rec_basis = rec.get("path_basis") or path_basis or "target_relative"
    r_repo_fa = _canonicalize_endpoint_path(r_fa, base_offset, path_basis=rec_basis)
    r_repo_fb = _canonicalize_endpoint_path(r_fb, base_offset, path_basis=rec_basis)

    r_repo_fp = _format_paired_endpoints(f"{r_repo_fa}:{r_na}", f"{r_repo_fb}:{r_nb}")
    r_repo_sfp = _format_paired_endpoints(f"{r_repo_fa}#{r_ha}", f"{r_repo_fb}#{r_hb}")
    r_repo_ns_a = extract_unit_namespace(r_repo_fa)
    r_repo_ns_b = extract_unit_namespace(r_repo_fb)
    r_repo_ns_sfp = _format_paired_endpoints(f"{r_repo_ns_a}#{r_ha}", f"{r_repo_ns_b}#{r_hb}")
    r_repo_namespaces = sorted([r_repo_ns_a, r_repo_ns_b])
    return r_repo_fa, r_repo_fb, r_repo_fp, r_repo_sfp, r_repo_ns_sfp, r_repo_namespaces


def _record_matches_names(rec: Dict[str, Any], target_names: List[str]) -> bool:
    """Checks if a baseline record's paired symbol names match target clone names."""
    rec_names = sorted([str(rec.get("name_a") or ""), str(rec.get("name_b") or "")])
    return rec_names == target_names


def _matches_boundary_and_structural_hashes(
    r_fa: str,
    r_fb: str,
    r_ha: str,
    r_hb: str,
    c_fa: str,
    c_fb: str,
    c_ha: str,
    c_hb: str,
    resolver: Optional[CanonicalPathResolver] = None,
) -> bool:
    """Checks if two clone endpoints match directory boundary paths and identical structural hashes."""
    if not (r_ha and r_hb and c_ha and c_hb):
        return False
    if resolver is not None:
        rk_r_fa = resolver.repo_key(r_fa, basis="repo")
        rk_r_fb = resolver.repo_key(r_fb, basis="repo")
        rk_c_fa = resolver.repo_key(c_fa, basis="repo")
        rk_c_fb = resolver.repo_key(c_fb, basis="repo")
        if rk_r_fa and rk_r_fb and rk_c_fa and rk_c_fb:
            if r_ha == c_ha and r_hb == c_hb:
                if rk_r_fa == rk_c_fa and rk_r_fb == rk_c_fb:
                    return True
            if r_ha == c_hb and r_hb == c_ha:
                if rk_r_fa == rk_c_fb and rk_r_fb == rk_c_fa:
                    return True
            return False

    if r_ha == c_ha and r_hb == c_hb:
        if paths_match_boundary(r_fa, c_fa) and paths_match_boundary(r_fb, c_fb):
            return True
    if r_ha == c_hb and r_hb == c_ha:
        if paths_match_boundary(r_fa, c_fb) and paths_match_boundary(r_fb, c_fa):
            return True
    return False


def _build_baseline_path_resolver(
    root: Optional[Union[str, Path]] = None,
    baseline_target: Optional[Union[str, Path]] = None,
    target: Optional[Union[str, Path]] = None,
) -> Optional[CanonicalPathResolver]:
    """Constructs a CanonicalPathResolver against enclosing Git worktree root or baseline target if available."""
    anchor = root or target or baseline_target
    if anchor is None:
        return None
    try:
        a_p = Path(anchor)
        anchor_dir: Union[str, Path] = (
            a_p.parent
            if (a_p.is_file() or (not a_p.is_dir() and a_p.suffix.lower() in (".py", ".ipynb")))
            else a_p
        )
    except (ValueError, OSError, RuntimeError):
        anchor_dir = anchor
    try:
        git_root = git_diff.get_git_repo_root(repo_root=anchor_dir)
        effective_repo: Union[str, Path] = root or git_root or anchor
        effective_target: Union[str, Path] = target or baseline_target or root or anchor
        if baseline_target is not None and target is not None:
            t_norm = normalize_lexical_posix(str(target))
            b_norm = normalize_lexical_posix(str(baseline_target))
            if t_norm and b_norm and t_norm != b_norm:
                if lexical_relative_to(t_norm, b_norm) is not None:
                    return CanonicalPathResolver(target_root=target, repo_root=baseline_target)
                if lexical_relative_to(b_norm, t_norm) is not None:
                    return CanonicalPathResolver(target_root=baseline_target, repo_root=target)
        return CanonicalPathResolver(target_root=effective_target, repo_root=effective_repo)
    except (ValueError, OSError, RuntimeError):
        return None


def _detect_clone_path_basis(
    clones: Sequence[Any],
    scan_offset: Optional[str],
    explicit_basis: Optional[str] = None,
    repo_root: Optional[Union[str, Path]] = None,
    target: Optional[Union[str, Path]] = None,
) -> str:
    """Detects whether active clone paths are repository-relative or target-relative."""
    if explicit_basis:
        return (
            "repo_relative"
            if explicit_basis in ("repo", "repo_relative", "worktree_relative")
            else "target_relative"
        )
    if not scan_offset:
        return "target_relative"
    off = normalize_lexical_posix(scan_offset).strip("/")
    if not off:
        return "target_relative"

    target_p: Optional[Path] = None
    repo_p: Optional[Path] = None
    try:
        if target is not None:
            tp = Path(target)
            if not tp.is_absolute() and repo_root is not None:
                tp = Path(repo_root) / tp
            tp = tp.resolve()
            target_p = (
                tp.parent
                if (tp.is_file() or (not tp.is_dir() and tp.suffix.lower() in (".py", ".ipynb")))
                else tp
            )
        if repo_root is not None:
            rp = Path(repo_root).resolve()
            repo_p = (
                rp.parent
                if (rp.is_file() or (not rp.is_dir() and rp.suffix.lower() in (".py", ".ipynb")))
                else rp
            )
    except (ValueError, OSError, RuntimeError):
        target_p = None
        repo_p = None

    candidate_files: List[str] = []
    for item in clones[:50]:
        u1, u2 = None, None
        if isinstance(item, str):
            parsed = _parse_legacy_fingerprint_record(item)
            raw_fa = parsed.get("file_a")
            raw_fb = parsed.get("file_b")
            if raw_fa:
                candidate_files.append(normalize_lexical_posix(str(raw_fa)))
            if raw_fb:
                candidate_files.append(normalize_lexical_posix(str(raw_fb)))
            continue
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            u1, u2 = item[1], item[2]
        elif isinstance(item, dict):
            fa, _, _, fb, _, _ = _extract_record_endpoint_data(item)
            if fa:
                candidate_files.append(normalize_lexical_posix(fa))
            if fb:
                candidate_files.append(normalize_lexical_posix(fb))
            u1 = item.get("u1") or item.get("unit_a") or (item if not fa and not fb else None)
            u2 = item.get("u2") or item.get("unit_b")
        if isinstance(u1, dict):
            f1 = normalize_lexical_posix(str(u1.get("file") or ""))
            if f1:
                candidate_files.append(f1)
        if isinstance(u2, dict):
            f2 = normalize_lexical_posix(str(u2.get("file") or ""))
            if f2:
                candidate_files.append(f2)

    candidate_files = [f for f in candidate_files if f]
    if not candidate_files:
        return "target_relative"

    prefix = f"{off}/"

    # Any candidate that does not start with the scan offset prefix proves target-relative
    if any(not (f == off or f.startswith(prefix)) for f in candidate_files):
        return "target_relative"

    # Filesystem disambiguation when target and repo_root directories are provided
    if target_p is not None and repo_p is not None and target_p != repo_p:
        found_target_only = False
        found_repo_only = False
        has_ambiguous_probe = False
        for f in candidate_files:
            f_clean = normalize_lexical_posix(f, strip_anchor=True)
            try:
                exists_target = (target_p / f_clean).is_file()
            except (ValueError, OSError):
                exists_target = False
            try:
                exists_repo = (repo_p / f_clean).is_file()
            except (ValueError, OSError):
                exists_repo = False

            if exists_target and not exists_repo:
                found_target_only = True
                break
            if exists_repo and not exists_target:
                found_repo_only = True
                break
            if exists_target and exists_repo:
                has_ambiguous_probe = True

        if found_target_only:
            return "target_relative"
        if found_repo_only:
            return "repo_relative"
        if has_ambiguous_probe:
            return "target_relative"

    if all(f == off or f.startswith(prefix) for f in candidate_files):
        return "repo_relative"

    return "target_relative"


def _match_exact_and_namespaced_passes(
    unconsumed: List[Dict[str, Any]],
    *,
    c_repo_fp: str,
    c_repo_sfp: str,
    c_sfp: str,
    c_ns_sfp: str,
    c_names: List[str],
    c_ha: str,
    c_hb: str,
    base_offset: Optional[str] = None,
    path_basis: Optional[str] = "target_relative",
) -> Optional[Dict[str, Any]]:
    """Evaluates Pass 1 (symbol-path), Pass 2 (path-structural), and Pass 3 (namespaced structural)."""
    # Pass 1: exact symbol-path fingerprint (file:name <===> file:name) prioritizing matching structural hash
    cand_pass1: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        _, _, r_repo_fp, r_repo_sfp, _, _ = _get_rec_repo_data(
            rec, base_offset, path_basis=path_basis
        )
        if r_repo_fp == c_repo_fp:
            if c_ha and c_hb and r_repo_sfp == c_repo_sfp:
                return rec
            if cand_pass1 is None:
                cand_pass1 = rec
    if cand_pass1 is not None:
        return cand_pass1

    # Pass 2: exact path-structural fingerprint (file#hash <===> file#hash) resilient to function renames
    for rec in unconsumed:
        r_sfp = rec.get("structural_fingerprint")
        if base_offset or not r_sfp:
            _, _, _, r_repo_sfp, _, _ = _get_rec_repo_data(
                rec, base_offset, path_basis=path_basis
            )
            r_sfp = r_repo_sfp
        if r_sfp == c_sfp:
            return rec

    # Pass 3: namespaced structural fingerprint (namespace#hash <===> namespace#hash) resilient to file renames within package
    cand_pass3: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        r_ns_sfp = rec.get("namespaced_structural_fingerprint")
        if base_offset or not r_ns_sfp:
            _, _, _, _, r_repo_ns_sfp, _ = _get_rec_repo_data(
                rec, base_offset, path_basis=path_basis
            )
            r_ns_sfp = r_repo_ns_sfp
        if r_ns_sfp == c_ns_sfp:
            if _record_matches_names(rec, c_names):
                return rec
            if cand_pass3 is None:
                cand_pass3 = rec
    if cand_pass3 is not None:
        return cand_pass3

    return None


def _match_structural_and_boundary_passes(
    unconsumed: List[Dict[str, Any]],
    *,
    c_pure_sfp: str,
    c_namespaces: List[str],
    c_names: List[str],
    c_repo_fa: str,
    c_repo_fb: str,
    c_ha: str,
    c_hb: str,
    resolver: Optional[CanonicalPathResolver] = None,
    base_offset: Optional[str] = None,
    path_basis: Optional[str] = "target_relative",
) -> Optional[Dict[str, Any]]:
    """Evaluates Pass 4 (pure structural with namespace), Pass 5 (cross-namespace moved), and Pass 6 (boundary)."""
    # Pass 4: pure structural fingerprint (hash <===> hash) strictly requiring matching module namespaces,
    # preventing identical boilerplate functions across different modules from colliding
    cand_pass4: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        if rec.get("pure_structural_fingerprint") == c_pure_sfp:
            if base_offset:
                _, _, _, _, _, r_namespaces = _get_rec_repo_data(
                    rec, base_offset, path_basis=path_basis
                )
            else:
                r_namespaces = sorted(
                    [x for x in (rec.get("namespace_a"), rec.get("namespace_b")) if x is not None]
                )
                if not r_namespaces:
                    _, _, _, _, _, r_namespaces = _get_rec_repo_data(
                        rec, base_offset, path_basis=path_basis
                    )
            if r_namespaces == c_namespaces:
                if _record_matches_names(rec, c_names):
                    return rec
                if cand_pass4 is None:
                    cand_pass4 = rec
    if cand_pass4 is not None:
        return cand_pass4

    # Pass 5: pure structural fallback for cross-namespace moved files; requires distinct structural hashes
    # (h_a != h_b) and matching symbols, preventing a grandfathered entry from being hijacked by unrelated cross-namespace code
    for rec in unconsumed:
        if rec.get("pure_structural_fingerprint") == c_pure_sfp:
            h_a = str(rec.get("hash_a", ""))
            h_b = str(rec.get("hash_b", ""))
            if not h_a and not h_b and rec.get("structural_fingerprint"):
                _, h_a, _, h_b = _parse_structural_fingerprint(str(rec["structural_fingerprint"]))
            if not h_a and not h_b and rec.get("pure_structural_fingerprint"):
                parts = str(rec["pure_structural_fingerprint"]).split(" <===> ")
                if len(parts) == 2:
                    h_a, h_b = parts[0], parts[1]
            if h_a and h_b and h_a != h_b:
                if not rec.get("name_a") or _record_matches_names(rec, c_names):
                    return rec

    # Pass 6: cross-root boundary-aware matching (e.g. baseline recorded at repo root vs scan targeting subdirectory)
    for rec in unconsumed:
        r_repo_fa, r_repo_fb, _, _, _, _ = _get_rec_repo_data(
            rec, base_offset, path_basis=path_basis
        )
        r_ha = str(rec.get("hash_a") or "")
        r_hb = str(rec.get("hash_b") or "")
        if not r_ha and not r_hb and rec.get("structural_fingerprint"):
            _, r_ha, _, r_hb = _parse_structural_fingerprint(str(rec["structural_fingerprint"]))
        if _matches_boundary_and_structural_hashes(
            r_repo_fa, r_repo_fb, r_ha, r_hb, c_repo_fa, c_repo_fb, c_ha, c_hb, resolver=resolver
        ):
            if not rec.get("name_a") or _record_matches_names(rec, c_names):
                return rec

    return None


def _resolve_clone_endpoint_repo_path(
    file_path: str,
    scan_offset: Optional[str],
    active_basis: str,
    resolver: Optional[CanonicalPathResolver] = None,
) -> str:
    """Resolves an endpoint file path to its canonical repository-relative path."""
    if resolver is not None and _is_absolute_path_str(file_path):
        res = resolver.resolve(file_path, basis=active_basis)
        return res.repo_relative or file_path
    return _canonicalize_endpoint_path(file_path, scan_offset, path_basis=active_basis)


@dataclass
class _CloneMatchCandidateContext:
    """Canonical representations and query fingerprints for baseline candidate matching."""

    c_repo_fa: str
    c_repo_fb: str
    c_names: List[str]
    c_ha: str
    c_hb: str
    c_repo_fp: str
    c_repo_sfp: str
    c_sfp: str
    c_ns_sfp: str
    c_namespaces: List[str]
    c_pure_sfp: str


def _prepare_clone_candidate_context(
    c_keys: Dict[str, Any],
    *,
    resolver: Optional[CanonicalPathResolver],
    base_offset: Optional[str],
    scan_offset: Optional[str],
    clone_basis: Optional[str],
) -> Optional[_CloneMatchCandidateContext]:
    """Derives and validates canonical candidate representations for matching against baseline records."""
    c_fa, c_na, c_ha, c_fb, c_nb, c_hb = _extract_record_endpoint_data(c_keys)
    active_basis = (
        "repo_relative"
        if clone_basis in ("repo", "repo_relative", "worktree_relative")
        else "target_relative"
    )
    c_repo_fa = _resolve_clone_endpoint_repo_path(c_fa, scan_offset, active_basis, resolver)
    c_repo_fb = _resolve_clone_endpoint_repo_path(c_fb, scan_offset, active_basis, resolver)

    if base_offset:
        rel_a = lexical_relative_to(c_repo_fa, base_offset)
        rel_b = lexical_relative_to(c_repo_fb, base_offset)
        if rel_a is None or rel_b is None:
            return None

    c_names = c_keys.get("names") or sorted([c_na, c_nb])
    c_repo_fp = _format_paired_endpoints(f"{c_repo_fa}:{c_na}", f"{c_repo_fb}:{c_nb}")
    c_repo_sfp = _format_paired_endpoints(f"{c_repo_fa}#{c_ha}", f"{c_repo_fb}#{c_hb}")
    c_repo_ns_a = extract_unit_namespace(c_repo_fa)
    c_repo_ns_b = extract_unit_namespace(c_repo_fb)
    c_repo_ns_sfp = _format_paired_endpoints(f"{c_repo_ns_a}#{c_ha}", f"{c_repo_ns_b}#{c_hb}")
    c_repo_namespaces = sorted([c_repo_ns_a, c_repo_ns_b])
    c_pure_sfp = c_keys.get("pure_sfp") or _format_paired_endpoints(c_ha, c_hb)
    c_sfp = c_repo_sfp if scan_offset else (c_keys.get("sfp") or c_repo_sfp)
    c_ns_sfp = c_repo_ns_sfp if scan_offset else (c_keys.get("ns_sfp") or c_repo_ns_sfp)
    c_namespaces = c_repo_namespaces if scan_offset else (c_keys.get("namespaces") or c_repo_namespaces)

    return _CloneMatchCandidateContext(
        c_repo_fa=c_repo_fa,
        c_repo_fb=c_repo_fb,
        c_names=c_names,
        c_ha=c_ha,
        c_hb=c_hb,
        c_repo_fp=c_repo_fp,
        c_repo_sfp=c_repo_sfp,
        c_sfp=c_sfp,
        c_ns_sfp=c_ns_sfp,
        c_namespaces=c_namespaces,
        c_pure_sfp=c_pure_sfp,
    )


def _match_clone_record(
    c_keys: Dict[str, Any],
    unconsumed: List[Dict[str, Any]],
    resolver: Optional[CanonicalPathResolver] = None,
    base_offset: Optional[str] = None,
    scan_offset: Optional[str] = None,
    path_basis: Optional[str] = "target_relative",
    clone_basis: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Finds matching unconsumed baseline record prioritizing exact and namespaced fingerprints."""
    ctx = _prepare_clone_candidate_context(
        c_keys,
        resolver=resolver,
        base_offset=base_offset,
        scan_offset=scan_offset,
        clone_basis=clone_basis,
    )
    if ctx is None:
        return None

    matched = _match_exact_and_namespaced_passes(
        unconsumed,
        c_repo_fp=ctx.c_repo_fp,
        c_repo_sfp=ctx.c_repo_sfp,
        c_sfp=ctx.c_sfp,
        c_ns_sfp=ctx.c_ns_sfp,
        c_names=ctx.c_names,
        c_ha=ctx.c_ha,
        c_hb=ctx.c_hb,
        base_offset=base_offset,
        path_basis=path_basis,
    )
    if matched is not None:
        return matched

    return _match_structural_and_boundary_passes(
        unconsumed,
        c_pure_sfp=ctx.c_pure_sfp,
        c_namespaces=ctx.c_namespaces,
        c_names=ctx.c_names,
        c_repo_fa=ctx.c_repo_fa,
        c_repo_fb=ctx.c_repo_fb,
        c_ha=ctx.c_ha,
        c_hb=ctx.c_hb,
        resolver=resolver,
        base_offset=base_offset,
        path_basis=path_basis,
    )



def _filter_clones_by_baseline_records(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    records: Sequence[Dict[str, Any]],
    *,
    resolver: Optional[CanonicalPathResolver],
    base_offset: Optional[str],
    scan_offset: Optional[str],
    base_basis: str,
    active_clone_basis: str,
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], int]:
    """Filters active clones against structured baseline fingerprint records."""
    unconsumed = list(records)
    new_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    suppressed_count = 0
    for sim, u1, u2 in clones:
        c_keys = {
            "fp": clone_pair_fingerprint(u1, u2),
            "sfp": clone_pair_structural_fingerprint(u1, u2),
            "ns_sfp": namespaced_structural_fingerprint(u1, u2),
            "pure_sfp": pure_structural_fingerprint(u1, u2),
            "namespaces": sorted([
                extract_unit_namespace(str(u1.get("file") or "")),
                extract_unit_namespace(str(u2.get("file") or "")),
            ]),
            "names": sorted([str(u1.get("name") or ""), str(u2.get("name") or "")]),
            "file_a": str(u1.get("file") or ""),
            "name_a": str(u1.get("name") or ""),
            "file_b": str(u2.get("file") or ""),
            "name_b": str(u2.get("name") or ""),
            "hash_a": str(u1.get("structural_hash") or compute_unit_structural_hash(u1)),
            "hash_b": str(u2.get("structural_hash") or compute_unit_structural_hash(u2)),
        }
        matched_rec = _match_clone_record(
            c_keys,
            unconsumed,
            resolver=resolver,
            base_offset=base_offset,
            scan_offset=scan_offset,
            path_basis=base_basis,
            clone_basis=active_clone_basis,
        )
        if matched_rec is not None:
            suppressed_count += 1
            unconsumed.remove(matched_rec)
        else:
            new_clones.append((sim, u1, u2))
    return new_clones, suppressed_count


def _filter_clones_by_plain_fingerprint_set(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_fingerprints: Union[Set[str], Sequence[str]],
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], int]:
    """Filters active clones against a plain set of grandfathered fingerprint strings."""
    new_clones_plain: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    suppressed_count_plain = 0
    avail_fps = set(baseline_fingerprints)
    for sim, u1, u2 in clones:
        fp = clone_pair_fingerprint(u1, u2)
        sfp = clone_pair_structural_fingerprint(u1, u2)
        ns_sfp = namespaced_structural_fingerprint(u1, u2)
        pure_sfp = pure_structural_fingerprint(u1, u2)
        matched_key: Optional[str] = None
        if fp in avail_fps:
            matched_key = fp
        elif sfp in avail_fps:
            matched_key = sfp
        elif ns_sfp in avail_fps:
            matched_key = ns_sfp
        elif pure_sfp in avail_fps:
            matched_key = pure_sfp

        if matched_key is not None:
            suppressed_count_plain += 1
            avail_fps.remove(matched_key)
        else:
            new_clones_plain.append((sim, u1, u2))
    return new_clones_plain, suppressed_count_plain


@dataclass
class _BaselineCoordinateContext:
    """Encapsulates resolved path offsets, coordinate basis, and canonical path resolver."""

    resolver: Optional[CanonicalPathResolver]
    base_offset: Optional[str]
    scan_offset: Optional[str]
    base_basis: str
    base_target: Optional[str]


def _resolve_baseline_coordinate_context(
    source: Any,
    repo_root: Optional[str],
    target: Optional[str],
) -> _BaselineCoordinateContext:
    """Derives canonical path resolver, base/scan offsets, and coordinate basis from baseline metadata."""
    if isinstance(source, dict):
        base_target = _safe_str(source.get("target"))
        target_repo_rel = source.get("target_repo_relative")
        raw_base_basis = source.get("path_basis")
        sample_records = source.get("fingerprints") or []
    else:
        raw_target = getattr(source, "target", None)
        base_target = _safe_str(raw_target) if raw_target is not None else None
        target_repo_rel = getattr(source, "target_repo_relative", None)
        raw_base_basis = getattr(source, "path_basis", None)
        sample_records = getattr(source, "records", None) or []

    resolver = _build_baseline_path_resolver(repo_root, baseline_target=base_target, target=target)
    base_offset, scan_offset = _derive_target_offsets(
        base_target,
        target_repo_rel,
        repo_root,
        target=target,
    )
    if not raw_base_basis and sample_records:
        inferred_offset = (
            base_offset
            or (normalize_lexical_posix(base_target).strip("./").rstrip("/") if base_target else None)
        )
        raw_base_basis = _detect_clone_path_basis(
            sample_records,
            inferred_offset,
            repo_root=repo_root,
            target=base_target or target,
        )
    base_basis = (
        "repo_relative"
        if raw_base_basis in ("repo", "repo_relative", "worktree_relative")
        else "target_relative"
    )
    return _BaselineCoordinateContext(
        resolver=resolver,
        base_offset=base_offset,
        scan_offset=scan_offset,
        base_basis=base_basis,
        base_target=base_target,
    )


def filter_clones_by_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_fingerprints: Union[Set[str], BaselineFingerprints],
    repo_root: Optional[str] = None,
    target: Optional[str] = None,
    clone_basis: Optional[str] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], int]:
    """Filters out grandfathered clones, returning newly introduced clones and count of suppressed clones."""
    if not baseline_fingerprints:
        return clones, 0

    records = getattr(baseline_fingerprints, "records", None)
    if records:
        ctx = _resolve_baseline_coordinate_context(
            baseline_fingerprints, repo_root=repo_root, target=target
        )
        active_clone_basis = _detect_clone_path_basis(
            clones,
            ctx.scan_offset,
            explicit_basis=clone_basis,
            repo_root=repo_root,
            target=target,
        )
        return _filter_clones_by_baseline_records(
            clones,
            records,
            resolver=ctx.resolver,
            base_offset=ctx.base_offset,
            scan_offset=ctx.scan_offset,
            base_basis=ctx.base_basis,
            active_clone_basis=active_clone_basis,
        )

    return _filter_clones_by_plain_fingerprint_set(clones, baseline_fingerprints)


class PruneResult(tuple):  # type: ignore[type-arg]
    """Result of prune_baseline with backward-compatible 2-tuple unpacking."""

    def __new__(
        cls,
        pruned_count: int,
        retained_count: int,
        skipped_dirty_count: int = 0,
    ) -> PruneResult:
        return super().__new__(cls, (pruned_count, retained_count))

    def __init__(
        self,
        pruned_count: int,
        retained_count: int,
        skipped_dirty_count: int = 0,
    ) -> None:
        super().__init__()
        self.pruned_count = pruned_count
        self.retained_count = retained_count
        self.skipped_dirty_count = skipped_dirty_count


def prune_baseline(
    baseline_path: str,
    active_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    unstaged_modified_ranges: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    repo_root: Optional[str] = None,
    target: Optional[str] = None,
    clone_basis: Optional[str] = None,
) -> PruneResult:
    """Prunes dead or refactored clone fingerprints from an existing baseline file.

    Returns:
        A PruneResult tuple of (pruned_count, retained_count).
    """
    path = Path(baseline_path)
    if not path.exists():
        return PruneResult(0, 0, 0)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return PruneResult(0, 0, 0)

    if not isinstance(data, dict) or "fingerprints" not in data:
        return PruneResult(0, 0, 0)

    ctx = _resolve_baseline_coordinate_context(data, repo_root=repo_root, target=target)
    if unstaged_modified_ranges is None:
        unstaged_modified_ranges = _resolve_unstaged_modified_ranges(repo_root)

    active_clone_basis = _detect_clone_path_basis(
        active_clones,
        ctx.scan_offset,
        explicit_basis=clone_basis,
        repo_root=repo_root,
        target=target,
    )
    indices = _build_pruning_active_clone_indices(
        active_clones, ctx.base_offset, ctx.scan_offset, active_clone_basis
    )

    retained: List[Dict[str, Any]] = []
    pruned_count = 0
    skipped_dirty_count = 0

    for raw_item in data.get("fingerprints", []):
        item, is_pruned, is_skipped_dirty = _evaluate_single_prune_record(
            raw_item,
            indices=indices,
            ctx=ctx,
            active_clone_basis=active_clone_basis,
            unstaged_modified_ranges=unstaged_modified_ranges,
        )
        if item is not None:
            retained.append(item)
            if is_skipped_dirty:
                skipped_dirty_count += 1
        elif is_pruned:
            pruned_count += 1

    _update_pruned_calibration_metadata(
        data, ctx.base_offset, ctx.base_target, repo_root, ctx.base_basis
    )
    data["clone_count"] = len(retained)
    data["fingerprints"] = retained
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return PruneResult(pruned_count, len(retained), skipped_dirty_count)


@dataclass
class _ActivePruningIndices:
    """Indexed collections of active clone representations for baseline pruning."""

    active_target_fps: Set[str]
    active_target_sfps: Set[str]
    active_repo_fps: Set[str]
    active_repo_sfps: Set[str]
    active_repo_ns_sfps: Set[str]
    active_pure_sfps: Set[str]
    repo_sfp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]
    repo_fp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]
    repo_ns_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]
    pure_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]
    scoped_clone_metadata: List[
        Tuple[Dict[str, Any], Dict[str, Any], str, str, str, str, List[str]]
    ]


def _build_pruning_active_clone_indices(
    active_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    base_offset: Optional[str],
    scan_offset: Optional[str],
    active_clone_basis: str,
) -> _ActivePruningIndices:
    """Builds active clone index sets and lookup tables for baseline pruning."""
    active_target_fps: Set[str] = set()
    active_target_sfps: Set[str] = set()
    active_repo_fps: Set[str] = set()
    active_repo_sfps: Set[str] = set()
    active_repo_ns_sfps: Set[str] = set()
    active_pure_sfps: Set[str] = set()
    repo_sfp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    repo_fp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    repo_ns_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    pure_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    scoped_clone_metadata: List[
        Tuple[Dict[str, Any], Dict[str, Any], str, str, str, str, List[str]]
    ] = []

    for _sim, u1, u2 in active_clones:
        u1_repo = _canonicalize_endpoint_path(
            str(u1.get("file") or ""), scan_offset, path_basis=active_clone_basis
        )
        u2_repo = _canonicalize_endpoint_path(
            str(u2.get("file") or ""), scan_offset, path_basis=active_clone_basis
        )
        if base_offset:
            rel_1 = lexical_relative_to(u1_repo, base_offset)
            rel_2 = lexical_relative_to(u2_repo, base_offset)
            if rel_1 is None or rel_2 is None:
                continue

        active_target_fps.add(clone_pair_fingerprint(u1, u2))
        active_target_sfps.add(clone_pair_structural_fingerprint(u1, u2))

        u1_name = str(u1.get("name") or "")
        u2_name = str(u2.get("name") or "")
        u1_hash = str(u1.get("structural_hash") or compute_unit_structural_hash(u1))
        u2_hash = str(u2.get("structural_hash") or compute_unit_structural_hash(u2))
        scoped_clone_metadata.append(
            (u1, u2, u1_repo, u2_repo, u1_hash, u2_hash, sorted([u1_name, u2_name]))
        )
        u1_ns = extract_unit_namespace(u1_repo)
        u2_ns = extract_unit_namespace(u2_repo)

        repo_fp = _format_paired_endpoints(f"{u1_repo}:{u1_name}", f"{u2_repo}:{u2_name}")
        repo_sfp = _format_paired_endpoints(f"{u1_repo}#{u1_hash}", f"{u2_repo}#{u2_hash}")
        repo_ns_sfp = _format_paired_endpoints(f"{u1_ns}#{u1_hash}", f"{u2_ns}#{u2_hash}")
        pure_sfp = _format_paired_endpoints(u1_hash, u2_hash)

        active_repo_fps.add(repo_fp)
        active_repo_sfps.add(repo_sfp)
        active_repo_ns_sfps.add(repo_ns_sfp)
        active_pure_sfps.add(pure_sfp)
        repo_sfp_to_clone[repo_sfp] = (u1, u2)
        repo_fp_to_clone[repo_fp] = (u1, u2)
        repo_ns_sfp_to_clones.setdefault(repo_ns_sfp, []).append((u1, u2))
        pure_sfp_to_clones.setdefault(pure_sfp, []).append((u1, u2))

    return _ActivePruningIndices(
        active_target_fps=active_target_fps,
        active_target_sfps=active_target_sfps,
        active_repo_fps=active_repo_fps,
        active_repo_sfps=active_repo_sfps,
        active_repo_ns_sfps=active_repo_ns_sfps,
        active_pure_sfps=active_pure_sfps,
        repo_sfp_to_clone=repo_sfp_to_clone,
        repo_fp_to_clone=repo_fp_to_clone,
        repo_ns_sfp_to_clones=repo_ns_sfp_to_clones,
        pure_sfp_to_clones=pure_sfp_to_clones,
        scoped_clone_metadata=scoped_clone_metadata,
    )


def _match_record_against_active_indices(
    item: Dict[str, Any],
    indices: _ActivePruningIndices,
    r_repo_fa: str,
    r_repo_fb: str,
    r_repo_fp: Optional[str],
    r_repo_sfp: Optional[str],
    r_repo_ns_sfp: Optional[str],
    r_repo_namespaces: List[str],
    h_a: str,
    h_b: str,
    item_pure_sfp: Optional[str],
    *,
    resolver: Optional[CanonicalPathResolver],
    scan_offset: Optional[str],
    active_clone_basis: str,
) -> Tuple[bool, Optional[Tuple[Dict[str, Any], Dict[str, Any]]], bool]:
    """Matches a baseline record against active clone indices across fingerprint tiers."""
    if r_repo_sfp and r_repo_sfp in indices.active_repo_sfps:
        return True, indices.repo_sfp_to_clone.get(r_repo_sfp), False
    if r_repo_fp and r_repo_fp in indices.active_repo_fps:
        return True, indices.repo_fp_to_clone.get(r_repo_fp), False
    if r_repo_ns_sfp and r_repo_ns_sfp in indices.active_repo_ns_sfps:
        item_names = sorted([str(item.get("name_a") or ""), str(item.get("name_b") or "")])
        for u1, u2 in indices.repo_ns_sfp_to_clones.get(r_repo_ns_sfp, []):
            u_names = sorted([str(u1.get("name") or ""), str(u2.get("name") or "")])
            if not item.get("name_a") or u_names == item_names:
                return True, (u1, u2), False
        if indices.repo_ns_sfp_to_clones.get(r_repo_ns_sfp):
            return True, indices.repo_ns_sfp_to_clones[r_repo_ns_sfp][0], False

    if item_pure_sfp and item_pure_sfp in indices.active_pure_sfps:
        item_ns = r_repo_namespaces
        item_names = sorted([
            str(item.get("name_a") or ""),
            str(item.get("name_b") or ""),
        ])
        for u1, u2 in indices.pure_sfp_to_clones.get(item_pure_sfp, []):
            u1_repo = _canonicalize_endpoint_path(
                str(u1.get("file") or ""), scan_offset, path_basis=active_clone_basis
            )
            u2_repo = _canonicalize_endpoint_path(
                str(u2.get("file") or ""), scan_offset, path_basis=active_clone_basis
            )
            u_ns = sorted([
                extract_unit_namespace(u1_repo),
                extract_unit_namespace(u2_repo),
            ])
            u_names = sorted([str(u1.get("name") or ""), str(u2.get("name") or "")])
            if u_ns == item_ns or (h_a and h_b and h_a != h_b and u_names == item_names):
                return True, (u1, u2), False

    for u1, u2, u1_f, u2_f, u1_h, u2_h, c_names in indices.scoped_clone_metadata:
        if _matches_boundary_and_structural_hashes(
            r_repo_fa, r_repo_fb, h_a, h_b, u1_f, u2_f, u1_h, u2_h, resolver=resolver
        ):
            if not item.get("name_a") or _record_matches_names(item, c_names):
                return True, (u1, u2), True

    return False, None, False


def _rewrite_pruned_record(
    item: Dict[str, Any],
    matched_clone: Tuple[Dict[str, Any], Dict[str, Any]],
    base_basis: str,
    active_clone_basis: str,
    scan_offset: Optional[str],
) -> None:
    """Updates baseline record metadata in-place to match active clone endpoint state."""
    u1, u2 = matched_clone
    rw_fa = str(u1.get("file") or "")
    rw_fb = str(u2.get("file") or "")
    if base_basis == "target_relative" and active_clone_basis == "repo_relative" and scan_offset:
        rel_fa = lexical_relative_to(rw_fa, scan_offset)
        if rel_fa:
            rw_fa = rel_fa
        rel_fb = lexical_relative_to(rw_fb, scan_offset)
        if rel_fb:
            rw_fb = rel_fb
    elif base_basis in ("repo_relative", "worktree_relative") and active_clone_basis == "target_relative" and scan_offset:
        off_prefix = normalize_lexical_posix(scan_offset).strip("/")
        rw_fa = f"{off_prefix}/{rw_fa}"
        rw_fb = f"{off_prefix}/{rw_fb}"
    f_a_norm = normalize_path_string(rw_fa, strip_anchor=False)
    f_b_norm = normalize_path_string(rw_fb, strip_anchor=False)
    u1_rewritten = dict(u1, file=f_a_norm)
    u2_rewritten = dict(u2, file=f_b_norm)
    item["file_a"] = f_a_norm
    item["file_b"] = f_b_norm
    item["name_a"] = str(u1.get("name") or "unit1")
    item["name_b"] = str(u2.get("name") or "unit2")
    item["namespace_a"] = extract_unit_namespace(f_a_norm)
    item["namespace_b"] = extract_unit_namespace(f_b_norm)
    item["fingerprint"] = clone_pair_fingerprint(u1_rewritten, u2_rewritten)
    item["structural_fingerprint"] = clone_pair_structural_fingerprint(u1_rewritten, u2_rewritten)
    item["namespaced_structural_fingerprint"] = namespaced_structural_fingerprint(u1_rewritten, u2_rewritten)
    item["pure_structural_fingerprint"] = pure_structural_fingerprint(u1, u2)
    item["hash_a"] = compute_unit_structural_hash(u1)
    item["hash_b"] = compute_unit_structural_hash(u2)


def _is_pruned_record_dirty(
    f_a: str,
    f_b: str,
    r_repo_fa: str,
    r_repo_fb: str,
    unstaged_modified_ranges: Optional[Dict[str, List[Tuple[int, int]]]],
) -> bool:
    """Checks whether any endpoint of a pruned record touches unstaged modified files."""
    if not unstaged_modified_ranges:
        return False
    return (
        find_matching_path_value(f_a, unstaged_modified_ranges) is not None
        or find_matching_path_value(f_b, unstaged_modified_ranges) is not None
        or find_matching_path_value(r_repo_fa, unstaged_modified_ranges) is not None
        or find_matching_path_value(r_repo_fb, unstaged_modified_ranges) is not None
    )


def _resolve_unstaged_modified_ranges(
    repo_root: Optional[str],
) -> Dict[str, List[Tuple[int, int]]]:
    """Queries git for unstaged modified line ranges, falling back cleanly if git fails."""
    try:
        return git_diff.get_git_modified_line_ranges(since_ref=None, repo_root=repo_root)
    except Exception as err:
        logger.debug(
            "Failed to query unstaged git modified line ranges during baseline pruning: %s",
            err,
        )
        return {}


def _evaluate_single_prune_record(
    raw_item: Any,
    *,
    indices: _ActivePruningIndices,
    ctx: _BaselineCoordinateContext,
    active_clone_basis: str,
    unstaged_modified_ranges: Optional[Dict[str, List[Tuple[int, int]]]],
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Evaluates a single baseline fingerprint record for retention, pruning, or dirty skipping.

    Returns:
        A tuple of (retained_item, is_pruned, is_skipped_dirty).
        retained_item is None when the record is discarded (pruned) or invalid.
    """
    if isinstance(raw_item, str):
        item = _parse_legacy_fingerprint_record(raw_item)
    elif isinstance(raw_item, dict):
        item = dict(raw_item)
    else:
        return None, False, False

    f_a = str(item.get("file_a") or "")
    f_b = str(item.get("file_b") or "")
    if not f_a and not f_b:
        f_a, _, _, f_b, _, _ = _extract_record_endpoint_data(item)

    h_a = str(item.get("hash_a", ""))
    h_b = str(item.get("hash_b", ""))
    if not h_a and not h_b and item.get("structural_fingerprint"):
        _, h_a, _, h_b = _parse_structural_fingerprint(str(item["structural_fingerprint"]))

    r_repo_fa, r_repo_fb, r_repo_fp, r_repo_sfp, r_repo_ns_sfp, r_repo_namespaces = (
        _get_rec_repo_data(item, ctx.base_offset, path_basis=ctx.base_basis)
    )
    if not h_a and not h_b and r_repo_sfp:
        _, h_a, _, h_b = _parse_structural_fingerprint(str(r_repo_sfp))

    item_pure_sfp = item.get("pure_structural_fingerprint")
    if not item_pure_sfp and h_a and h_b:
        item_pure_sfp = _format_paired_endpoints(h_a, h_b)
        item["pure_structural_fingerprint"] = item_pure_sfp

    is_active, matched_clone, is_boundary_matched = _match_record_against_active_indices(
        item,
        indices,
        r_repo_fa,
        r_repo_fb,
        r_repo_fp,
        r_repo_sfp,
        r_repo_ns_sfp,
        r_repo_namespaces,
        h_a,
        h_b,
        item_pure_sfp,
        resolver=ctx.resolver,
        scan_offset=ctx.scan_offset,
        active_clone_basis=active_clone_basis,
    )

    if is_active:
        can_rewrite = (
            ctx.base_offset == ctx.scan_offset
            and not is_boundary_matched
            and matched_clone is not None
            and (
                item.get("fingerprint") not in indices.active_target_fps
                or item.get("structural_fingerprint") not in indices.active_target_sfps
            )
        )
        if can_rewrite and matched_clone is not None:
            _rewrite_pruned_record(
                item, matched_clone, ctx.base_basis, active_clone_basis, ctx.scan_offset
            )
        return item, False, False

    if _is_pruned_record_dirty(f_a, f_b, r_repo_fa, r_repo_fb, unstaged_modified_ranges):
        return item, False, True

    return None, True, False


def _update_pruned_calibration_metadata(
    data: Dict[str, Any],
    base_offset: Optional[str],
    base_target: Optional[str],
    repo_root: Optional[str],
    base_basis: str,
) -> None:
    """Normalizes baseline version, path basis, and calibration scope after pruning."""
    data["version"] = "1.5.0"
    if "path_basis" not in data or not data["path_basis"]:
        data["path_basis"] = base_basis
    if "target_repo_relative" not in data:
        if base_offset:
            data["target_repo_relative"] = base_offset
        elif base_target is not None:
            base_res = _build_baseline_path_resolver(root=repo_root, baseline_target=base_target)
            if base_res is not None and base_res.target_in_repo:
                data["target_repo_relative"] = base_res.target_in_repo
    if isinstance(data.get("corpus_calibration"), dict):
        calib_entry = data["corpus_calibration"]
        target_rel = data.get("target_repo_relative")
        if target_rel:
            if not calib_entry.get("target_repo_relative"):
                calib_entry["target_repo_relative"] = target_rel
            if not calib_entry.get("scope"):
                calib_entry["scope"] = target_rel
        recomputed_calib_hash = compute_calibration_config_hash(calib_entry)
        calib_entry["config_hash"] = recomputed_calib_hash
        data["config_hash"] = recomputed_calib_hash
