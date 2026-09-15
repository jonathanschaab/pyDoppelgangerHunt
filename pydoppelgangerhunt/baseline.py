"""Grandfathered clone baseline recording, loading, and filtering."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def clone_pair_fingerprint(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Computes stable, order-invariant fingerprint for a clone pair."""
    f1 = u1["file"].replace("\\", "/")
    f2 = u2["file"].replace("\\", "/")
    n1 = u1["name"]
    n2 = u2["name"]
    ep1 = f"{f1}:{n1}"
    ep2 = f"{f2}:{n2}"
    ordered = sorted([ep1, ep2])
    return f"{ordered[0]} <===> {ordered[1]}"


def record_baseline(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    baseline_path: str,
    target: str,
    threshold: float,
) -> str:
    """Records detected clones into a JSON baseline file for grandfathering."""
    data: Dict[str, Any] = {
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "threshold": threshold,
        "clone_count": len(clones),
        "fingerprints": [
            {
                "fingerprint": clone_pair_fingerprint(u1, u2),
                "similarity": round(sim, 4),
                "file_a": u1["file"].replace("\\", "/"),
                "name_a": u1["name"],
                "file_b": u2["file"].replace("\\", "/"),
                "name_b": u2["name"],
            }
            for sim, u1, u2 in clones
        ],
    }
    target_p = Path(baseline_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)
    target_p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(target_p)


def load_baseline(baseline_path: str) -> Set[str]:
    """Loads grandfathered clone fingerprints from a JSON baseline file."""
    path = Path(baseline_path)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fingerprints = data.get("fingerprints", [])
        return {item["fingerprint"] for item in fingerprints if "fingerprint" in item}
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
        if fp in baseline_fingerprints:
            suppressed_count += 1
        else:
            new_clones.append((sim, u1, u2))
    return new_clones, suppressed_count
