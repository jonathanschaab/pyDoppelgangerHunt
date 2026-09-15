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
    if tokens:
        return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]
    vec = unit.get("vector", {})
    sorted_vec = json.dumps(vec, sort_keys=True)
    return hashlib.sha256(sorted_vec.encode("utf-8")).hexdigest()[:16]


def _format_paired_endpoints(ep1: str, ep2: str) -> str:
    """Formats two endpoints into an order-invariant clone pair representation."""
    ordered = sorted([ep1, ep2])
    return f"{ordered[0]} <===> {ordered[1]}"


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
        "version": "1.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "threshold": threshold,
        "clone_count": len(clones),
        "fingerprints": [
            {
                "fingerprint": clone_pair_fingerprint(u1, u2),
                "structural_fingerprint": clone_pair_structural_fingerprint(u1, u2),
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
            if "fingerprint" in item:
                fps.add(item["fingerprint"])
            if "structural_fingerprint" in item:
                fps.add(item["structural_fingerprint"])
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
        if fp in baseline_fingerprints or sfp in baseline_fingerprints:
            suppressed_count += 1
        else:
            new_clones.append((sim, u1, u2))
    return new_clones, suppressed_count
