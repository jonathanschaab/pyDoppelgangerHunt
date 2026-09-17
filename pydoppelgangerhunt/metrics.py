"""Codebase metrics, Source Lines of Code (SLOC), and DRY scorecard computation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from pydoppelgangerhunt.clustering import cluster_clone_families
from pydoppelgangerhunt.config import find_python_files


def compute_repository_dry_stats(
    target_dir: str,
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    excludes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates repository-wide DRY metrics: SLOC, DLOC, Duplication %, and DRY Grade."""
    total_sloc = 0
    package_sloc: Dict[str, int] = {}
    file_list = find_python_files(target_dir, excludes=excludes)

    for p in file_list:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                count = sum(1 for line in fh if line.strip() and not line.strip().startswith("#"))
            total_sloc += count
            top_pkg = p.parts[0] if len(p.parts) > 1 else str(p)
            package_sloc[top_pkg] = package_sloc.get(top_pkg, 0) + count
        except OSError:
            continue

    duplicated_lines_by_file: Dict[str, Set[int]] = {}
    for _sim, u1, u2 in clones:
        for u in (u1, u2):
            f_norm = str(u.get("file") or "").replace("\\", "/")
            s = int(u.get("start") or 1)
            e = int(u.get("end") or s)
            duplicated_lines_by_file.setdefault(f_norm, set()).update(
                range(s, e + 1)
            )

    total_dloc = sum(len(line_set) for line_set in duplicated_lines_by_file.values())
    dup_pct = (total_dloc / total_sloc * 100.0) if total_sloc > 0 else 0.0
    dry_score = max(0.0, 100.0 - dup_pct)

    if dry_score >= 98.0:
        grade = "A+"
    elif dry_score >= 95.0:
        grade = "A"
    elif dry_score >= 90.0:
        grade = "B"
    elif dry_score >= 80.0:
        grade = "C"
    else:
        grade = "F"

    families = cluster_clone_families(clones)

    return {
        "sloc": total_sloc,
        "dloc": total_dloc,
        "duplication_pct": dup_pct,
        "dry_score": dry_score,
        "grade": grade,
        "clone_pairs": len(clones),
        "clone_families": len(families),
        "package_sloc": package_sloc,
    }
