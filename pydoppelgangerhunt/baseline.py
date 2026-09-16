"""Grandfathered clone baseline recording, loading, and filtering."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


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


def pure_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes path-independent, order-invariant structural content fingerprint for a clone pair."""
    h1 = compute_unit_structural_hash(u1)
    h2 = compute_unit_structural_hash(u2)
    return _format_paired_endpoints(h1, h2)


def clone_pair_structural_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes order-invariant structural content fingerprint for a clone pair."""
    f1 = u1["file"].replace("\\", "/")
    f2 = u2["file"].replace("\\", "/")
    h1 = compute_unit_structural_hash(u1)
    h2 = compute_unit_structural_hash(u2)
    return _format_paired_endpoints(f"{f1}#{h1}", f"{f2}#{h2}")


def clone_pair_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes stable, order-invariant fingerprint for a clone pair."""
    f1 = u1["file"].replace("\\", "/")
    f2 = u2["file"].replace("\\", "/")
    return _format_paired_endpoints(f"{f1}:{u1['name']}", f"{f2}:{u2['name']}")


def record_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_path: str,
    target: str,
    threshold: float,
) -> str:
    """Records detected clones into a JSON baseline file for grandfathering."""
    data: Dict[str, Any] = {
        "version": "1.2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "threshold": threshold,
        "clone_count": len(clones),
        "fingerprints": [
            {
                "fingerprint": clone_pair_fingerprint(u1, u2),
                "structural_fingerprint": clone_pair_structural_fingerprint(u1, u2),
                "pure_structural_fingerprint": pure_structural_fingerprint(u1, u2),
                "similarity": round(sim, 4),
                "file_a": u1["file"].replace("\\", "/"),
                "name_a": u1["name"],
                "hash_a": compute_unit_structural_hash(u1),
                "file_b": u2["file"].replace("\\", "/"),
                "name_b": u2["name"],
                "hash_b": compute_unit_structural_hash(u2),
            }
            for sim, u1, u2 in clones
        ],
    }
    target_p = Path(baseline_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)
    target_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(target_p)


def load_baseline(baseline_path: str) -> Set[str]:
    """Loads grandfathered clone fingerprints and structural hashes from a JSON baseline file."""
    path = Path(baseline_path)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fingerprints = data.get("fingerprints", [])
        fps: Set[str] = set()
        for item in fingerprints:
            if "fingerprint" in item and item["fingerprint"]:
                fps.add(item["fingerprint"])
            if "structural_fingerprint" in item and item["structural_fingerprint"]:
                fps.add(item["structural_fingerprint"])
            if "pure_structural_fingerprint" in item and item["pure_structural_fingerprint"]:
                fps.add(item["pure_structural_fingerprint"])
            elif "hash_a" in item and "hash_b" in item and item["hash_a"] and item["hash_b"]:
                fps.add(_format_paired_endpoints(str(item["hash_a"]), str(item["hash_b"])))
        return fps
    except (json.JSONDecodeError, OSError):
        return set()


def filter_clones_by_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_fingerprints: Set[str],
) -> Tuple[List[Tuple[float, Dict[str, Any], Dict[str, Any]]], int]:
    """Filters out grandfathered clones, returning newly introduced clones and count of suppressed clones."""
    if not baseline_fingerprints:
        return clones, 0

    new_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    suppressed_count = 0
    for sim, u1, u2 in clones:
        fp = clone_pair_fingerprint(u1, u2)
        sfp = clone_pair_structural_fingerprint(u1, u2)
        pure_sfp = pure_structural_fingerprint(u1, u2)
        if (
            fp in baseline_fingerprints
            or sfp in baseline_fingerprints
            or pure_sfp in baseline_fingerprints
        ):
            suppressed_count += 1
        else:
            new_clones.append((sim, u1, u2))
    return new_clones, suppressed_count


def prune_baseline(
    baseline_path: str,
    active_clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
) -> Tuple[int, int]:
    """Prunes dead or refactored clone fingerprints from an existing baseline file.

    Returns:
        A tuple of (pruned_count, retained_count).
    """
    path = Path(baseline_path)
    if not path.exists():
        return 0, 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, 0

    if not isinstance(data, dict) or "fingerprints" not in data:
        return 0, 0

    active_fps: Set[str] = set()
    active_sfps: Set[str] = set()
    active_pure_sfps: Set[str] = set()
    pure_sfp_to_clone: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    for _sim, u1, u2 in active_clones:
        fp = clone_pair_fingerprint(u1, u2)
        sfp = clone_pair_structural_fingerprint(u1, u2)
        pure_sfp = pure_structural_fingerprint(u1, u2)
        active_fps.add(fp)
        active_sfps.add(sfp)
        active_pure_sfps.add(pure_sfp)
        pure_sfp_to_clone[pure_sfp] = (u1, u2)

    retained: List[Dict[str, Any]] = []
    pruned_count = 0

    for item in data.get("fingerprints", []):
        item_fp = item.get("fingerprint")
        item_sfp = item.get("structural_fingerprint")
        item_pure_sfp = item.get("pure_structural_fingerprint")
        if not item_pure_sfp and "hash_a" in item and "hash_b" in item:
            item_pure_sfp = _format_paired_endpoints(str(item["hash_a"]), str(item["hash_b"]))

        is_active = bool(
            (item_fp and item_fp in active_fps)
            or (item_sfp and item_sfp in active_sfps)
            or (item_pure_sfp and item_pure_sfp in active_pure_sfps)
        )

        if is_active:
            if item_pure_sfp:
                item["pure_structural_fingerprint"] = item_pure_sfp
            # If the entry matched solely via pure_sfp (e.g. file was moved or renamed),
            # synchronize the file paths and path-bound fingerprints to the new location.
            if (
                item_fp not in active_fps
                and item_sfp not in active_sfps
                and item_pure_sfp in pure_sfp_to_clone
            ):
                u1, u2 = pure_sfp_to_clone[item_pure_sfp]
                item["file_a"] = u1["file"].replace("\\", "/")
                item["file_b"] = u2["file"].replace("\\", "/")
                item["name_a"] = u1["name"]
                item["name_b"] = u2["name"]
                item["fingerprint"] = clone_pair_fingerprint(u1, u2)
                item["structural_fingerprint"] = clone_pair_structural_fingerprint(u1, u2)
            retained.append(item)
        else:
            pruned_count += 1

    data["version"] = "1.2.0"
    data["clone_count"] = len(retained)
    data["fingerprints"] = retained
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return pruned_count, len(retained)
