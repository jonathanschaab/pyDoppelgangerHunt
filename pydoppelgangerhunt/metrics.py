"""Codebase metrics, Source Lines of Code (SLOC), and DRY scorecard computation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from pydoppelgangerhunt.clustering import cluster_clone_families
from pydoppelgangerhunt.config import canonical_path_key, find_python_files


def compute_repository_dry_stats(
    target_dir: str,
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    excludes: Optional[List[str]] = None,
    include_notebooks: bool = False,
) -> Dict[str, Any]:
    """Calculates repository-wide DRY metrics: SLOC, DLOC, Duplication %, and DRY Grade."""
    total_sloc = 0
    package_sloc: Dict[str, int] = {}
    file_list = find_python_files(
        target_dir, excludes=excludes, include_notebooks=include_notebooks
    )

    for p in file_list:
        try:
            if p.suffix == ".ipynb":
                try:
                    nb_data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                    count = 0
                    cells = nb_data.get("cells", [])
                    if isinstance(cells, list):
                        for cell in cells:
                            if isinstance(cell, dict) and cell.get("cell_type") == "code":
                                src = cell.get("source", [])
                                cell_code = "".join(src) if isinstance(src, list) else str(src)
                                count += sum(
                                    1
                                    for line in cell_code.splitlines()
                                    if line.strip() and not line.strip().startswith("#")
                                )
                except (json.JSONDecodeError, OSError):
                    count = 0
            else:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    count = sum(1 for line in fh if line.strip() and not line.strip().startswith("#"))
            total_sloc += count
            try:
                rel = p.relative_to(target_dir)
                top_pkg = rel.parts[0] if len(rel.parts) > 1 else rel.name
            except ValueError:
                top_pkg = p.parts[0] if len(p.parts) > 1 else str(p)
            package_sloc[top_pkg] = package_sloc.get(top_pkg, 0) + count
        except OSError:
            continue

    duplicated_lines_by_file: Dict[str, Set[int]] = {}
    for _sim, u1, u2 in clones:
        for u in (u1, u2):
            f_norm = canonical_path_key(str(u.get("file") or ""), strip_anchor=False)
            s = int(u.get("start") or 1)
            e = int(u.get("end") or s)
            duplicated_lines_by_file.setdefault(f_norm, set()).update(
                range(s, e + 1)
            )

    total_dloc = sum(len(line_set) for line_set in duplicated_lines_by_file.values())
    dup_pct = min(100.0, (total_dloc / total_sloc * 100.0)) if total_sloc > 0 else 0.0
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
