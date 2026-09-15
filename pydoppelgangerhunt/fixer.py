"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydoppelgangerhunt.reporters import extract_unit_source_code


def synthesize_shared_helper_code(u1: Dict[str, Any], u2: Dict[str, Any]) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units."""
    lines1 = [ln.rstrip("\r\n") for ln in extract_unit_source_code(u1)]
    lines2 = [ln.rstrip("\r\n") for ln in extract_unit_source_code(u2)]

    base_name1 = u1["name"].split(":")[0].lstrip("_")
    base_name2 = u2["name"].split(":")[0].lstrip("_")
    helper_name = f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"

    # Find common lines using difflib matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])

    if not common_lines:
        common_lines = lines1

    # Indent body
    indented_body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in common_lines)
    return f"def {helper_name}(*args: Any, **kwargs: Any) -> Any:\n    \"\"\"Auto-extracted shared helper for duplicate logic.\"\"\"\n{indented_body}\n"


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
) -> str:
    """Generates a git-apply compatible unified diff patch proposing shared helper extractions."""
    if not clones:
        return ""

    root = Path(repo_root or os.getcwd())
    patch_chunks: List[str] = []

    for sim, u1, u2 in clones:
        f1_raw = u1["file"].split("#")[0]
        p = Path(f1_raw)
        f1_path = p if p.is_absolute() else (root / f1_raw)
        if not f1_path.is_file():
            continue

        try:
            orig_text = f1_path.read_text(encoding="utf-8")
        except OSError:
            continue

        orig_lines = orig_text.splitlines(keepends=True)
        helper_code = synthesize_shared_helper_code(u1, u2)

        # Prepend helper to file as candidate patch
        modified_lines = [helper_code + "\n\n"] + orig_lines

        try:
            rel_f1 = str(f1_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            rel_f1 = str(f1_path.name)
        diff = difflib.unified_diff(
            orig_lines,
            modified_lines,
            fromfile=f"a/{rel_f1}",
            tofile=f"b/{rel_f1}",
            n=3,
        )
        diff_str = "".join(diff)
        if diff_str:
            patch_chunks.append(f"# Clone Pair ({sim:.1%}): {u1['file']} <===> {u2['file']}\n" + diff_str)

    return "\n".join(patch_chunks)
