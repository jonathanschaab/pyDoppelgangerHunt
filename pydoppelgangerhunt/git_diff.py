"""Git incremental diff line parsing and PR clone gating."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.config import find_matching_path_value, normalize_path_string

MAJOR_POLICY_THRESHOLD: float = 0.50
NEW_POLICY_THRESHOLD: float = 0.80

_GIT_C_ESCAPES: Dict[int, int] = {
    ord(b"a"): 0x07,
    ord(b"b"): 0x08,
    ord(b"t"): 0x09,
    ord(b"n"): 0x0A,
    ord(b"v"): 0x0B,
    ord(b"f"): 0x0C,
    ord(b"r"): 0x0D,
    ord(b'"'): 0x22,
    ord(b"\\"): 0x5C,
}


def _run_git_command(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
    """Executes a git command safely and returns standard output, or None on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
    except (OSError, ValueError):
        pass
    return None


def _decode_git_cstyle_path(raw_path: str) -> str:
    """Decodes C-style octal and escape sequences in git quotepath strings while preserving literal UTF-8."""
    trimmed = raw_path.rstrip("\r\n")
    if not trimmed or not trimmed.strip():
        return ""
    if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2:
        inner = trimmed[1:-1]
        try:
            raw_bytes = bytearray()
            inner_bytes = inner.encode("utf-8", errors="replace")
            i = 0
            n = len(inner_bytes)
            while i < n:
                b = inner_bytes[i]
                if b == 0x5C and i + 1 < n:
                    nxt = inner_bytes[i + 1]
                    if 0x30 <= nxt <= 0x37:
                        octal_val = nxt - 0x30
                        i += 2
                        for _ in range(2):
                            if i < n and 0x30 <= inner_bytes[i] <= 0x37:
                                octal_val = (octal_val << 3) + (inner_bytes[i] - 0x30)
                                i += 1
                            else:
                                break
                        raw_bytes.append(octal_val & 0xFF)
                        continue
                    if nxt in _GIT_C_ESCAPES:
                        raw_bytes.append(_GIT_C_ESCAPES[nxt])
                        i += 2
                        continue
                    raw_bytes.append(nxt)
                    i += 2
                    continue
                raw_bytes.append(b)
                i += 1
            return raw_bytes.decode("utf-8", errors="replace")
        except (UnicodeError, ValueError):
            return inner
    return trimmed


def parse_git_diff_hunks(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parses unified diff output into mapping of file paths to changed line ranges."""
    modified_ranges: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            current_file = None
        elif line.startswith("+++ "):
            rest = line[4:].strip()
            if rest.startswith('"'):
                closing_idx = rest.rfind('"')
                if 0 < closing_idx < len(rest) - 1:
                    rest = rest[: closing_idx + 1]
            elif "\t" in rest:
                rest = rest.split("\t", 1)[0].strip()
            rest = _decode_git_cstyle_path(rest)
            if rest in ("/dev/null", ""):
                current_file = None
            else:
                if len(rest) > 2 and rest[1] == "/" and rest[0] in "biwc":
                    rest = rest[2:]
                current_file = normalize_path_string(rest, strip_anchor=False)
        elif line.startswith("@@ ") and current_file:
            parts = line.split(" ")
            plus_parts = [p for p in parts if p.startswith("+")]
            if plus_parts:
                hunk_spec = plus_parts[0][1:]
                try:
                    if "," in hunk_spec:
                        start_str, count_str = hunk_spec.split(",", 1)
                        start = int(start_str)
                        count = int(count_str)
                    else:
                        start = int(hunk_spec)
                        count = 1
                    if count > 0:
                        modified_ranges.setdefault(current_file, []).append((start, start + count - 1))
                except ValueError:
                    pass

    return modified_ranges


def _run_git_diff(
    diff_flags: Sequence[str],
    since_ref: Optional[str] = None,
    cwd: Optional[str] = None,
) -> Optional[str]:
    """Builds and executes a safe git diff command isolating revisions from pathspecs."""
    if since_ref and since_ref.strip().startswith("-"):
        return None
    args = ["diff", *diff_flags]
    if since_ref:
        args.extend([since_ref, "--"])
    else:
        args.append("--")
    return _run_git_command(args, cwd=cwd)


def get_git_modified_line_ranges(
    since_ref: Optional[str] = None,
    repo_root: Optional[str] = None,
    cwd: Optional[str] = None,
) -> Dict[str, List[Tuple[int, int]]]:
    """Extracts modified line ranges for files using git diff --unified=0."""
    diff_output = _run_git_diff(
        ["--unified=0", "--src-prefix=a/", "--dst-prefix=b/"],
        since_ref=since_ref,
        cwd=repo_root or cwd,
    )
    if not diff_output:
        return {}

    return parse_git_diff_hunks(diff_output)


def get_git_modified_files(
    since_ref: Optional[str] = None,
    repo_root: Optional[str] = None,
    cwd: Optional[str] = None,
) -> List[str]:
    """Extracts normalized file paths of modified files from git diff."""
    diff_output = _run_git_diff(
        ["--name-only"],
        since_ref=since_ref,
        cwd=repo_root or cwd,
    )
    if not diff_output:
        return []
    modified_files: List[str] = []
    for line in diff_output.splitlines():
        trimmed = _decode_git_cstyle_path(line)
        if trimmed:
            norm = normalize_path_string(trimmed, strip_anchor=False)
            if norm and norm not in modified_files:
                modified_files.append(norm)
    return modified_files


def get_git_repo_root(
    since_ref: Optional[str] = None,
    repo_root: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolves the top-level root directory of the current Git worktree, or None if not in a repository."""
    _ = since_ref
    target_cwd = str(repo_root or cwd) if (repo_root or cwd) else None
    raw = _run_git_command(["rev-parse", "--show-toplevel"], cwd=target_cwd)
    if not raw:
        return None
    top = raw.rstrip("\r\n")
    if top and top.strip():
        return normalize_path_string(top, strip_anchor=False)
    return None


def get_git_head_commit(
    repo_root: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolves the current HEAD commit hash of the Git repository, or None if not available."""
    target_cwd = str(repo_root or cwd) if (repo_root or cwd) else None
    raw = _run_git_command(["rev-parse", "HEAD"], cwd=target_cwd)
    if not raw:
        return None
    commit = raw.strip()
    if len(commit) in (40, 64) and all(c in "0123456789abcdefABCDEF" for c in commit):
        return commit.lower()
    return None




def compute_unit_diff_overlap(
    unit: Dict[str, Any],
    modified_ranges: Dict[str, List[Tuple[int, int]]],
) -> Tuple[int, float]:
    """Computes the number of modified lines and the fractional overlap ratio for an AST unit.

    Returns:
        A tuple of (overlapping_modified_line_count, overlap_ratio) where ratio is in [0.0, 1.0].

    Example:
        >>> from pydoppelgangerhunt.git_diff import (
        ...     compute_unit_diff_overlap,
        ...     parse_git_diff_hunks,
        ... )
        >>> diff_text = (
        ...     "--- a/service.py\\n"
        ...     "+++ b/service.py\\n"
        ...     "@@ -10,0 +10,5 @@\\n"
        ... )
        >>> modified_ranges = parse_git_diff_hunks(diff_text)
        >>> unit = {"file": "service.py", "start": 8, "end": 15}
        >>> overlap_count, overlap_ratio = compute_unit_diff_overlap(unit, modified_ranges)
        >>> overlap_count
        5
        >>> overlap_ratio
        0.625
    """
    target_ranges = find_matching_path_value(str(unit.get("file") or ""), modified_ranges)
    if not target_ranges:
        return 0, 0.0

    u_start = int(unit.get("start") or 1)
    u_end = int(unit.get("end") or u_start)
    total_unit_lines = max(1, u_end - u_start + 1)

    overlapping_lines: Set[int] = set()
    for m_start, m_end in target_ranges:
        overlap_start = max(u_start, m_start)
        overlap_end = min(u_end, m_end)
        if overlap_start <= overlap_end:
            overlapping_lines.update(range(overlap_start, overlap_end + 1))

    overlap_count = len(overlapping_lines)
    overlap_ratio = overlap_count / total_unit_lines
    return overlap_count, round(overlap_ratio, 4)


def is_unit_in_modified_ranges(
    unit: Dict[str, Any],
    modified_ranges: Dict[str, List[Tuple[int, int]]],
    min_overlap_ratio: float = 0.0,
    policy: str = "any",
) -> bool:
    """Checks if AST unit overlaps with git-modified line ranges according to the partial hunk policy.

    Policies:
        - "any": Overlaps if at least 1 line is modified (and overlap_ratio >= min_overlap_ratio).
        - "major": Overlaps if at least 50% of the unit lines are modified (or min_overlap_ratio if > 0).
        - "new": Overlaps if at least 80% of the unit lines are modified (or min_overlap_ratio if > 0).
    """
    overlap_count, ratio = compute_unit_diff_overlap(unit, modified_ranges)
    if overlap_count == 0:
        return False

    effective_ratio_floor = min_overlap_ratio
    if effective_ratio_floor <= 0.0:
        if policy == "major":
            effective_ratio_floor = MAJOR_POLICY_THRESHOLD
        elif policy == "new":
            effective_ratio_floor = NEW_POLICY_THRESHOLD

    return ratio >= effective_ratio_floor


def filter_clones_by_git_diff(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    modified_ranges: Dict[str, List[Tuple[int, int]]],
    policy: str = "any",
    min_overlap_ratio: float = 0.0,
    both_units: bool = False,
) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Filters clones based on git-modified line ranges and partial hunk policy."""
    if not modified_ranges:
        return []
    filtered: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for sim, u1, u2 in clones:
        u1_mod = is_unit_in_modified_ranges(
            u1, modified_ranges, min_overlap_ratio=min_overlap_ratio, policy=policy
        )
        u2_mod = is_unit_in_modified_ranges(
            u2, modified_ranges, min_overlap_ratio=min_overlap_ratio, policy=policy
        )
        if both_units:
            if u1_mod and u2_mod:
                filtered.append((sim, u1, u2))
        else:
            if u1_mod or u2_mod:
                filtered.append((sim, u1, u2))
    return filtered


def get_git_blame_info(
    file_path: str,
    start_line: int,
    end_line: int,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Extracts git commit and author metadata for a line range using git blame --porcelain."""
    norm_file = normalize_path_string(file_path, strip_anchor=True)
    if not norm_file:
        return {"author": "Unknown", "commit": "unknown", "timestamp": 0, "summary": ""}
    start_l = max(1, start_line)
    end_l = max(start_l, end_line)
    args = ["blame", "-L", f"{start_l},{end_l}", "--porcelain", "--", norm_file]

    blame_text = _run_git_command(args, cwd=repo_root)
    if not blame_text:
        return {"author": "Unknown", "commit": "unknown", "timestamp": 0, "summary": ""}

    latest_time = 0
    latest_commit = "unknown"
    commit_authors: Dict[str, str] = {}
    commit_summaries: Dict[str, str] = {}
    commit_times: Dict[str, int] = {}
    current_commit = ""

    for line in blame_text.splitlines():
        parts = line.split(" ", 1)
        if len(parts[0]) in (40, 64) and all(c in "0123456789abcdefABCDEF" for c in parts[0]):
            current_commit = parts[0][:8]
            if latest_commit == "unknown":
                latest_commit = current_commit
        elif line.startswith("author "):
            commit_authors[current_commit] = line[7:].strip()
        elif line.startswith("author-time "):
            try:
                t_val = int(line[12:].strip())
                commit_times[current_commit] = t_val
                if t_val > latest_time:
                    latest_time = t_val
                    latest_commit = current_commit
            except ValueError:
                pass
        elif line.startswith("summary "):
            commit_summaries[current_commit] = line[8:].strip()

    return {
        "author": commit_authors.get(latest_commit, "Unknown"),
        "commit": latest_commit,
        "timestamp": latest_time,
        "summary": commit_summaries.get(latest_commit, ""),
    }


def check_temporal_divergence(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    repo_root: Optional[str] = None,
    max_divergence_days: int = 90,
) -> Optional[Dict[str, Any]]:
    """Detects whether two clone instances exhibit temporal divergence (asymmetric commit ages)."""
    f1 = normalize_path_string(str(u1.get("file") or ""), strip_anchor=False)
    f2 = normalize_path_string(str(u2.get("file") or ""), strip_anchor=False)
    if not f1 or not f2:
        return None
    s1 = int(u1.get("start") or 1)
    e1 = int(u1.get("end") or s1)
    s2 = int(u2.get("start") or 1)
    e2 = int(u2.get("end") or s2)
    b1 = get_git_blame_info(f1, s1, e1, repo_root=repo_root)
    b2 = get_git_blame_info(f2, s2, e2, repo_root=repo_root)

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
