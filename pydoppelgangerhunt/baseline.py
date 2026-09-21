"""Grandfathered clone baseline recording, loading, and filtering."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.canonical_path import CanonicalPathResolver
from pydoppelgangerhunt.config import (
    canonical_path_key,
    find_matching_path_value,
    normalize_path_string,
    paths_match_boundary,
)
from pydoppelgangerhunt import git_diff

logger = logging.getLogger(__name__)


MAX_CALIBRATION_UNITS: int = 1_000_000_000


def _safe_total_units(raw_units: Any) -> int:
    """Clamps total_units to [0, MAX_CALIBRATION_UNITS], returning 0 for malformed/overflowing values."""
    if raw_units is None:
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


def _safe_min_corpus(raw_min_corpus: Any, filter_stop_shingles: bool = False) -> int:
    """Sanitizes min_corpus_size to a non-negative int, or default based on filter_stop_shingles."""
    if raw_min_corpus is not None:
        try:
            val = int(raw_min_corpus)
            if val >= 0:
                return val
        except (ValueError, TypeError, OverflowError):
            pass
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


def _attach_calibration_flags(
    target: Dict[str, Any],
    source: Dict[str, Any],
) -> None:
    """Attaches feature mode flags and bounds from source dictionary to target calibration dict."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        return
    for flag in ("bag_of_tokens", "call_sequences", "filter_stop_shingles"):
        target[flag] = _safe_bool(source.get(flag, False))
    for flag, default_val in HARVEST_BOOLEAN_MODES:
        target[flag] = _safe_bool(source.get(flag, default_val))
    for int_flag, default_int in (
        ("window_size", 5),
        ("min_expr_complexity", 4),
        ("min_lines", 8),
        ("min_tokens", 15),
    ):
        raw_val = source.get(int_flag, default_int)
        try:
            val = int(raw_val)
            target[int_flag] = val if val > 0 else default_int
        except (ValueError, TypeError, OverflowError):
            target[int_flag] = default_int
    if "min_corpus_size" in source:
        raw_mcs = source.get("min_corpus_size")
        if raw_mcs is None:
            target["min_corpus_size"] = None
        else:
            try:
                val = int(raw_mcs)
                target["min_corpus_size"] = val if val >= 0 else None
            except (ValueError, TypeError, OverflowError):
                target["min_corpus_size"] = None


def _serialize_shingle_key(sh: Any) -> str:
    """Serializes a shingle key into a type-tagged string representation for JSON."""
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


def _deserialize_shingle_key(key: str) -> Any:
    """Deserializes a JSON-compatible string key back into a shingle tuple or scalar."""
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
    if raw_freq is None:
        return None
    try:
        val = float(raw_freq)
        if math.isfinite(val) and 0.0 < val <= 1.0:
            return val
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def compute_corpus_calibration(
    units: Sequence[Dict[str, Any]],
    max_index_frequency: Optional[float] = 0.25,
    min_corpus_size: Optional[int] = None,
    filter_stop_shingles: bool = False,
    stop_shingles: Optional[Set[Any]] = None,
    *,
    bag_of_tokens: bool = False,
    call_sequences: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Computes global shingle document frequencies and calibrated stop-shingles for a repository corpus."""
    total_units = len(units)
    shingle_frequencies: Dict[Any, int] = {}
    for u in units:
        if call_sequences:
            keys = set(u.get("calls", []))
        elif bag_of_tokens:
            keys = set(u.get("vector", {}).keys())
        elif "shingles" in u and u["shingles"]:
            keys = set(u["shingles"])
        elif "vector" in u and u["vector"]:
            keys = set(u["vector"].keys())
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

    calib = {
        "total_units": total_units,
        "max_index_frequency": valid_max_freq,
        "global_stop_shingles": global_stop_shingles,
        "shingle_frequencies": shingle_frequencies,
    }
    flags_dict: Dict[str, Any] = {
        "bag_of_tokens": bag_of_tokens,
        "call_sequences": call_sequences,
        "filter_stop_shingles": filter_stop_shingles,
        "min_corpus_size": min_corpus_size,
    }
    flags_dict.update(kwargs)
    _attach_calibration_flags(calib, flags_dict)
    return calib



def compute_unit_structural_hash(unit: Dict[str, Any]) -> str:
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


def namespaced_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes order-invariant structural content fingerprint bound to module/package namespaces."""
    ns1 = extract_unit_namespace(str(u1.get("file") or ""))
    ns2 = extract_unit_namespace(str(u2.get("file") or ""))
    return _paired_hashed_fingerprint(u1, u2, ns1, ns2)


def pure_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes path-independent, order-invariant structural content fingerprint for a clone pair."""
    return _format_paired_endpoints(compute_unit_structural_hash(u1), compute_unit_structural_hash(u2))


def clone_pair_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes order-invariant structural content fingerprint for a clone pair."""
    f1 = canonical_path_key(str(u1.get("file") or ""), strip_anchor=False)
    f2 = canonical_path_key(str(u2.get("file") or ""), strip_anchor=False)
    return _paired_hashed_fingerprint(u1, u2, f1, f2)


def clone_pair_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
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
) -> str:
    """Records detected clones into a JSON baseline file for grandfathering."""
    target_repo_rel = None
    try:
        git_root = git_diff.get_git_repo_root(target)
        if git_root:
            res = CanonicalPathResolver(target_root=target, repo_root=git_root)
            target_repo_rel = res.target_in_repo
    except (ValueError, OSError, RuntimeError):
        pass

    data: Dict[str, Any] = {
        "version": "1.5.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "path_basis": "target_relative",
        "threshold": threshold,
        "clone_count": len(clones),
        "fingerprints": [
            {
                "fingerprint": clone_pair_fingerprint(u1, u2),
                "structural_fingerprint": clone_pair_structural_fingerprint(u1, u2),
                "namespaced_structural_fingerprint": namespaced_structural_fingerprint(u1, u2),
                "pure_structural_fingerprint": pure_structural_fingerprint(u1, u2),
                "similarity": round(sim, 4),
                "file_a": normalize_path_string(str(u1.get("file") or ""), strip_anchor=False),
                "name_a": str(u1.get("name") or "unit1"),
                "hash_a": compute_unit_structural_hash(u1),
                "namespace_a": extract_unit_namespace(str(u1.get("file") or "")),
                "file_b": normalize_path_string(str(u2.get("file") or ""), strip_anchor=False),
                "name_b": str(u2.get("name") or "unit2"),
                "hash_b": compute_unit_structural_hash(u2),
                "namespace_b": extract_unit_namespace(str(u2.get("file") or "")),
            }
            for sim, u1, u2 in clones
        ],
    }
    if target_repo_rel:
        data["target_repo_relative"] = target_repo_rel
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

        calib_entry = {
            "total_units": total_units_val,
            "max_index_frequency": safe_max_freq,
            "global_stop_shingles": safe_stops,
            "shingle_frequencies": dict(sorted(safe_shingle_freqs.items(), key=lambda item: item[0])),
        }
        _attach_calibration_flags(calib_entry, corpus_calibration)
        data["corpus_calibration"] = calib_entry
    target_p = Path(baseline_path)
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
    ) -> None:
        super().__init__(fps or set())
        self.records: List[Dict[str, Any]] = records or []
        self.corpus_calibration: Optional[Dict[str, Any]] = corpus_calibration
        self.path_basis: Optional[str] = path_basis
        self.target_repo_relative: Optional[str] = target_repo_relative


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
            decoded_freqs = _sanitize_shingle_frequency_dict(
                raw_calib.get("shingle_frequencies", {}),
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

            corpus_calibration = {
                "total_units": calib_total_units,
                "max_index_frequency": max_idx_freq,
                "global_stop_shingles": decoded_stops,
                "shingle_frequencies": decoded_freqs,
            }
            _attach_calibration_flags(corpus_calibration, raw_calib)

        path_basis = data.get("path_basis")
        target_repo_rel = data.get("target_repo_relative")
        return BaselineFingerprints(
            fps,
            records=records,
            corpus_calibration=corpus_calibration,
            path_basis=path_basis,
            target_repo_relative=target_repo_rel,
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError, RecursionError):
        return BaselineFingerprints()



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
        if r_ha == c_ha and r_hb == c_hb:
            return resolver.equivalent(r_fa, c_fa) and resolver.equivalent(r_fb, c_fb)
        if r_ha == c_hb and r_hb == c_ha:
            return resolver.equivalent(r_fa, c_fb) and resolver.equivalent(r_fb, c_fa)
        return False

    if r_ha == c_ha and r_hb == c_hb:
        return paths_match_boundary(r_fa, c_fa) and paths_match_boundary(r_fb, c_fb)
    if r_ha == c_hb and r_hb == c_ha:
        return paths_match_boundary(r_fa, c_fb) and paths_match_boundary(r_fb, c_fa)
    return False


def _match_clone_record(
    c_keys: Dict[str, Any],
    unconsumed: List[Dict[str, Any]],
    resolver: Optional[CanonicalPathResolver] = None,
) -> Optional[Dict[str, Any]]:
    """Finds matching unconsumed baseline record prioritizing exact and namespaced fingerprints."""
    c_fp = c_keys["fp"]
    c_sfp = c_keys["sfp"]
    c_ns_sfp = c_keys["ns_sfp"]
    c_pure_sfp = c_keys["pure_sfp"]
    c_namespaces = c_keys["namespaces"]
    c_names = c_keys["names"]

    # Pass 1: exact symbol-path fingerprint (file:name <===> file:name) prioritizing matching structural hash
    cand_pass1: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        if rec.get("fingerprint") == c_fp:
            if rec.get("structural_fingerprint") == c_sfp:
                return rec
            if cand_pass1 is None:
                cand_pass1 = rec
    if cand_pass1 is not None:
        return cand_pass1

    # Pass 2: exact path-structural fingerprint (file#hash <===> file#hash) resilient to function renames
    for rec in unconsumed:
        if rec.get("structural_fingerprint") == c_sfp:
            return rec

    # Pass 3: namespaced structural fingerprint (namespace#hash <===> namespace#hash) resilient to file renames within package
    cand_pass3: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        if rec.get("namespaced_structural_fingerprint") == c_ns_sfp:
            if _record_matches_names(rec, c_names):
                return rec
            if cand_pass3 is None:
                cand_pass3 = rec
    if cand_pass3 is not None:
        return cand_pass3

    # Pass 4: pure structural fingerprint (hash <===> hash) strictly requiring matching module namespaces,
    # preventing identical boilerplate functions across different modules from colliding
    cand_pass4: Optional[Dict[str, Any]] = None
    for rec in unconsumed:
        if rec.get("pure_structural_fingerprint") == c_pure_sfp:
            rec_ns = sorted([
                rec.get("namespace_a") or extract_unit_namespace(str(rec.get("file_a") or "")),
                rec.get("namespace_b") or extract_unit_namespace(str(rec.get("file_b") or "")),
            ])
            if rec_ns == c_namespaces:
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
            if h_a and h_b and h_a != h_b:
                if not rec.get("name_a") or _record_matches_names(rec, c_names):
                    return rec

    # Pass 6: cross-root boundary-aware matching (e.g. baseline recorded at repo root vs scan targeting subdirectory)
    for rec in unconsumed:
        r_fa = str(rec.get("file_a") or "")
        r_fb = str(rec.get("file_b") or "")
        r_ha = str(rec.get("hash_a") or "")
        r_hb = str(rec.get("hash_b") or "")
        c_fa = str(c_keys.get("file_a") or "")
        c_fb = str(c_keys.get("file_b") or "")
        c_ha = str(c_keys.get("hash_a") or "")
        c_hb = str(c_keys.get("hash_b") or "")
        if _matches_boundary_and_structural_hashes(
            r_fa, r_fb, r_ha, r_hb, c_fa, c_fb, c_ha, c_hb, resolver=resolver
        ):
            if not rec.get("name_a") or _record_matches_names(rec, c_names):
                return rec

    return None


def filter_clones_by_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_fingerprints: Set[str],
    repo_root: Optional[str] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], int]:
    """Filters out grandfathered clones, returning newly introduced clones and count of suppressed clones."""
    if not baseline_fingerprints:
        return clones, 0

    resolver: Optional[CanonicalPathResolver] = None
    if repo_root is not None:
        try:
            resolver = CanonicalPathResolver(target_root=repo_root, repo_root=repo_root)
        except (ValueError, OSError, RuntimeError):
            resolver = None

    records = getattr(baseline_fingerprints, "records", None)
    if records:
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
                "file_b": str(u2.get("file") or ""),
                "hash_a": str(u1.get("structural_hash") or compute_unit_structural_hash(u1)),
                "hash_b": str(u2.get("structural_hash") or compute_unit_structural_hash(u2)),
            }
            matched_rec = _match_clone_record(c_keys, unconsumed, resolver=resolver)
            if matched_rec is not None:
                suppressed_count += 1
                unconsumed.remove(matched_rec)
            else:
                new_clones.append((sim, u1, u2))
        return new_clones, suppressed_count

    # Consumable set-based matching if a plain set was provided
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

    resolver: Optional[CanonicalPathResolver] = None
    if repo_root is not None:
        try:
            resolver = CanonicalPathResolver(target_root=repo_root, repo_root=repo_root)
        except (ValueError, OSError, RuntimeError):
            resolver = None

    if unstaged_modified_ranges is None:
        try:
            try:
                unstaged_modified_ranges = git_diff.get_git_modified_line_ranges(
                    since_ref=None, repo_root=repo_root
                )
            except TypeError:
                unstaged_modified_ranges = git_diff.get_git_modified_line_ranges(since_ref=None)
        except Exception as err:
            # Pragmatic fallback when git is unavailable, outside a repo, or query fails
            logger.debug(
                "Failed to query unstaged git modified line ranges during baseline pruning: %s",
                err,
            )
            unstaged_modified_ranges = {}

    active_fps: Set[str] = set()
    active_sfps: Set[str] = set()
    active_ns_sfps: Set[str] = set()
    active_pure_sfps: Set[str] = set()
    sfp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    ns_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    pure_sfp_to_clones: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}

    for _sim, u1, u2 in active_clones:
        fp = clone_pair_fingerprint(u1, u2)
        sfp = clone_pair_structural_fingerprint(u1, u2)
        ns_sfp = namespaced_structural_fingerprint(u1, u2)
        pure_sfp = pure_structural_fingerprint(u1, u2)
        active_fps.add(fp)
        active_sfps.add(sfp)
        active_ns_sfps.add(ns_sfp)
        active_pure_sfps.add(pure_sfp)
        sfp_to_clone[sfp] = (u1, u2)
        ns_sfp_to_clones.setdefault(ns_sfp, []).append((u1, u2))
        pure_sfp_to_clones.setdefault(pure_sfp, []).append((u1, u2))

    retained: List[Dict[str, Any]] = []
    pruned_count = 0
    skipped_dirty_count = 0

    for raw_item in data.get("fingerprints", []):
        if isinstance(raw_item, str):
            item = _parse_legacy_fingerprint_record(raw_item)
        elif isinstance(raw_item, dict):
            item = dict(raw_item)
        else:
            continue
        item_fp = item.get("fingerprint")
        item_sfp = item.get("structural_fingerprint")
        item_ns_sfp = item.get("namespaced_structural_fingerprint")
        item_pure_sfp = item.get("pure_structural_fingerprint")
        h_a = str(item.get("hash_a", ""))
        h_b = str(item.get("hash_b", ""))

        if not item_sfp and h_a and h_b:
            f_a = canonical_path_key(str(item.get("file_a") or ""), strip_anchor=False)
            f_b = canonical_path_key(str(item.get("file_b") or ""), strip_anchor=False)
            item_sfp = _format_paired_endpoints(f"{f_a}#{h_a}", f"{f_b}#{h_b}")
            item["structural_fingerprint"] = item_sfp

        if not item_ns_sfp and h_a and h_b:
            ns_a = item.get("namespace_a") or extract_unit_namespace(str(item.get("file_a") or ""))
            ns_b = item.get("namespace_b") or extract_unit_namespace(str(item.get("file_b") or ""))
            item_ns_sfp = _format_paired_endpoints(f"{ns_a}#{h_a}", f"{ns_b}#{h_b}")
            item["namespaced_structural_fingerprint"] = item_ns_sfp
            item["namespace_a"] = ns_a
            item["namespace_b"] = ns_b

        if not item_pure_sfp and h_a and h_b:
            item_pure_sfp = _format_paired_endpoints(h_a, h_b)
            item["pure_structural_fingerprint"] = item_pure_sfp

        is_active = bool(
            (item_fp and item_fp in active_fps)
            or (item_sfp and item_sfp in active_sfps)
            or (item_ns_sfp and item_ns_sfp in active_ns_sfps)
        )

        matched_clone: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        is_boundary_matched = False
        if not is_active and item_pure_sfp and item_pure_sfp in active_pure_sfps:
            item_ns = sorted([
                item.get("namespace_a") or extract_unit_namespace(str(item.get("file_a") or "")),
                item.get("namespace_b") or extract_unit_namespace(str(item.get("file_b") or "")),
            ])
            item_names = sorted([
                str(item.get("name_a") or ""),
                str(item.get("name_b") or ""),
            ])
            for u1, u2 in pure_sfp_to_clones.get(item_pure_sfp, []):
                u_ns = sorted([
                    extract_unit_namespace(str(u1.get("file") or "")),
                    extract_unit_namespace(str(u2.get("file") or "")),
                ])
                u_names = sorted([str(u1.get("name") or ""), str(u2.get("name") or "")])
                if u_ns == item_ns or (h_a != h_b and u_names == item_names):
                    is_active = True
                    matched_clone = (u1, u2)
                    break

        if not is_active:
            r_fa = str(item.get("file_a") or "")
            r_fb = str(item.get("file_b") or "")
            for _sim, u1, u2 in active_clones:
                u1_f = str(u1.get("file") or "")
                u2_f = str(u2.get("file") or "")
                u1_h = compute_unit_structural_hash(u1)
                u2_h = compute_unit_structural_hash(u2)
                if _matches_boundary_and_structural_hashes(
                    r_fa, r_fb, h_a, h_b, u1_f, u2_f, u1_h, u2_h, resolver=resolver
                ):
                    c_names = sorted([str(u1.get("name") or ""), str(u2.get("name") or "")])
                    if not item.get("name_a") or _record_matches_names(item, c_names):
                        is_active = True
                        is_boundary_matched = True
                        matched_clone = (u1, u2)
                        break

        if is_active:
            if not matched_clone and item_sfp and item_sfp in sfp_to_clone:
                matched_clone = sfp_to_clone[item_sfp]
            if not matched_clone and item_ns_sfp and item_ns_sfp in ns_sfp_to_clones:
                item_names = sorted([
                    str(item.get("name_a") or ""),
                    str(item.get("name_b") or ""),
                ])
                for u1, u2 in ns_sfp_to_clones[item_ns_sfp]:
                    u_names = sorted([str(u1.get("name") or ""), str(u2.get("name") or "")])
                    if u_names == item_names:
                        matched_clone = (u1, u2)
                        break
                if not matched_clone:
                    matched_clone = ns_sfp_to_clones[item_ns_sfp][0]

            if (
                matched_clone
                and not is_boundary_matched
                and (item_fp not in active_fps or item_sfp not in active_sfps)
            ):
                u1, u2 = matched_clone
                item["file_a"] = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
                item["file_b"] = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
                item["name_a"] = str(u1.get("name") or "unit1")
                item["name_b"] = str(u2.get("name") or "unit2")
                item["namespace_a"] = extract_unit_namespace(str(u1.get("file") or ""))
                item["namespace_b"] = extract_unit_namespace(str(u2.get("file") or ""))
                item["fingerprint"] = clone_pair_fingerprint(u1, u2)
                item["structural_fingerprint"] = clone_pair_structural_fingerprint(u1, u2)
                item["namespaced_structural_fingerprint"] = namespaced_structural_fingerprint(u1, u2)
                item["pure_structural_fingerprint"] = pure_structural_fingerprint(u1, u2)
                item["hash_a"] = compute_unit_structural_hash(u1)
                item["hash_b"] = compute_unit_structural_hash(u2)
            retained.append(item)
        else:
            f_a = str(item.get("file_a") or "")
            f_b = str(item.get("file_b") or "")
            is_dirty = bool(
                unstaged_modified_ranges
                and (
                    find_matching_path_value(f_a, unstaged_modified_ranges) is not None
                    or find_matching_path_value(f_b, unstaged_modified_ranges) is not None
                )
            )
            if is_dirty:
                skipped_dirty_count += 1
                retained.append(item)
            else:
                pruned_count += 1

    data["version"] = "1.5.0"
    data["clone_count"] = len(retained)
    data["fingerprints"] = retained
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return PruneResult(pruned_count, len(retained), skipped_dirty_count)
