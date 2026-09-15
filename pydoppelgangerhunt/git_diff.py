"""Git incremental diff line parsing and PR clone gating."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _run_git_command(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
    """Executes a git command safely and returns standard output, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
    except (OSError, ValueError):
        pass
    return None


def parse_git_diff_hunks(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parses unified diff output into mapping of file paths to changed line ranges."""
    modified_ranges: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip().replace("\\", "/")
        elif line.startswith("@@ ") and current_file:
            parts = line.split(" ")
            plus_parts = [p for p in parts if p.startswith("+")]
            if plus_parts:
                hunk_spec = plus_parts[0][1:]
                if "," in hunk_spec:
                    start_str, count_str = hunk_spec.split(",", 1)
                    start = int(start_str)
                    count = int(count_str)
                else:
                    start = int(hunk_spec)
                    count = 1
                if count > 0:
                    modified_ranges.setdefault(current_file, []).append((start, start + count - 1))

    return modified_ranges


def get_git_modified_line_ranges(
    since_ref: Optional[str] = None, repo_root: Optional[str] = None
) -> Dict[str, List[Tuple[int, int]]]:
    """Extracts modified line ranges for files using git diff --unified=0."""
    args = ["diff", "--unified=0"]
    if since_ref:
        args.append(since_ref)

    diff_output = _run_git_command(args, cwd=repo_root)
    if not diff_output:
        return {}

    return parse_git_diff_hunks(diff_output)


def is_unit_in_modified_ranges(
    unit: Dict[str, Any], modified_ranges: Dict[str, List[Tuple[int, int]]]
) -> bool:
    """Checks if AST unit overlaps with any git-modified line ranges."""
    norm_file = unit["file"].replace("\\", "/")
    target_ranges = modified_ranges.get(norm_file)
    if not target_ranges:
        for f, ranges in modified_ranges.items():
            if norm_file.endswith(f) or f.endswith(norm_file):
                target_ranges = ranges
                break
    if not target_ranges:
        return False

    u_start = unit["start"]
    u_end = unit["end"]
    for m_start, m_end in target_ranges:
        if max(u_start, m_start) <= min(u_end, m_end):
            return True
    return False


def filter_clones_by_git_diff(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    modified_ranges: Dict[str, List[Tuple[int, int]]],
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Filters clones to only retain pairs where at least one unit intersects git-modified lines."""
    if not modified_ranges:
        return []
    filtered: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for sim, u1, u2 in clones:
        if is_unit_in_modified_ranges(u1, modified_ranges) or is_unit_in_modified_ranges(u2, modified_ranges):
            filtered.append((sim, u1, u2))
    return filtered


def get_git_blame_info(
    file_path: str,
    start_line: int,
    end_line: int,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts git commit and author metadata for a line range using git blame --porcelain."""
    norm_file = file_path.split("#")[0]  # strip notebook cell suffixes if present
    start_l = max(1, start_line)
    end_l = max(start_l, end_line)
    args = ["blame", "-L", f"{start_l},{end_l}", "--porcelain", norm_file]

    blame_text = _run_git_command(args, cwd=repo_root)
    if not blame_text:
        return {"author": "Unknown", "commit": "unknown", "timestamp": 0, "summary": ""}

    latest_time = 0
    latest_author = "Unknown"
    latest_commit = "unknown"
    latest_summary = ""

    current_commit = ""
    for line in blame_text.splitlines():
        parts = line.split(" ", 1)
        if len(parts[0]) == 40 and all(c in "0123456789abcdefABCDEF" for c in parts[0]):
            current_commit = parts[0][:8]
        elif line.startswith("author "):
            author = line[7:].strip()
        elif line.startswith("author-time "):
            try:
                t_val = int(line[12:].strip())
                if t_val > latest_time:
                    latest_time = t_val
                    latest_author = author
                    latest_commit = current_commit
            except ValueError:
                pass
        elif line.startswith("summary ") and current_commit == latest_commit:
            latest_summary = line[8:].strip()

    return {
        "author": latest_author,
        "commit": latest_commit,
        "timestamp": latest_time,
        "summary": latest_summary,
    }


def check_temporal_divergence(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    max_divergence_days: int = 90,
) -> Optional[Dict[str, Any]]:
    """Detects whether two clone instances exhibit temporal divergence (asymmetric commit ages)."""
    b1 = get_git_blame_info(u1["file"], u1["start"], u1["end"], repo_root=repo_root)
    b2 = get_git_blame_info(u2["file"], u2["start"], u2["end"], repo_root=repo_root)

    t1 = b1.get("timestamp", 0)
    t2 = b2.get("timestamp", 0)
    if t1 <= 0 or t2 <= 0:
        return None

    delta_sec = abs(t1 - t2)
    delta_days = delta_sec / 86400.0
    if delta_days >= max_divergence_days:
        return {
            "divergence_days": round(delta_days, 1),
            "u1_blame": b1,
            "u2_blame": b2,
            "newer": u1 if t1 >= t2 else u2,
            "older": u2 if t1 >= t2 else u1,
        }
    return None
