#!/usr/bin/env python3
"""Verify Python 3.9 syntax compatibility across package and tests (plus mypy 3.10 floor)."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


def main() -> int:
    """Parse all Python sources under Python 3.9 grammar and check mypy floor on >=3.11."""
    repo_root = Path(__file__).resolve().parent.parent
    errors = []
    for root_name in ("pydoppelgangerhunt", "tests"):
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for py_path in sorted(root.rglob("*.py")):
            text = py_path.read_text(encoding="utf-8")
            try:
                ast.parse(text, filename=str(py_path), feature_version=(3, 9))
            except SyntaxError as exc:
                errors.append(f"{py_path.relative_to(repo_root)}:{exc.lineno}: {exc.msg}")
    if errors:
        for err in errors:
            print(f"PY39_SYNTAX_ERROR: {err}", file=sys.stderr)
        return 1
    if sys.version_info >= (3, 11):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--python-version",
                "3.10",
                "pydoppelgangerhunt",
                "tests",
            ],
            cwd=str(repo_root),
            check=False,
        )
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
