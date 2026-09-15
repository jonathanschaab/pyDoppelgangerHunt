"""Configuration loading and initialization for pyDoppelgangerHunt."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import os

DEFAULT_EXCLUDES: List[str] = [
    "checks/encapsulated",
    "checks\\encapsulated",
    "tests",
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    ".gemini",
    "scratch",
]

DEFAULT_TOOL_TOML_CONTENT = """[tool.pydoppelgangerhunt]
threshold = 0.90
min_lines = 8
min_tokens = 15
exclude = [
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    "build",
    "dist",
]
call_sequences = true
idioms = true
"""


def find_python_files(
    target: Union[str, Path],
    excludes: Optional[List[str]] = None,
    audit_tests: bool = False,
    include_notebooks: bool = False,
) -> List[Path]:
    """Discovers all target Python source files while filtering excluded paths."""
    exclude_patterns = list(excludes if excludes is not None else DEFAULT_EXCLUDES)
    target_path = Path(target)
    if not target_path.exists():
        return []
    valid_suffixes = (".py", ".ipynb") if include_notebooks else (".py",)
    if target_path.is_file():
        return [target_path] if target_path.suffix in valid_suffixes else []

    target_clean = str(target_path).replace("\\", "/").strip("./")
    active_excludes = [
        ex for ex in exclude_patterns
        if ex.replace("\\", "/").strip("./") not in target_clean
    ]
    if audit_tests:
        active_excludes = [ex for ex in active_excludes if "test" not in ex]

    def _is_excluded(full_p: str, rel_p: str) -> bool:
        return any(
            ex.replace("\\", "/").strip("/") in full_p
            or ex.replace("\\", "/").strip("/") in rel_p
            for ex in active_excludes
        )

    found_files: List[Path] = []
    target_str = str(target_path)
    for root, _, files in os.walk(target_str):
        rel_root = os.path.relpath(root, target_str).replace("\\", "/")
        norm_root = root.replace("\\", "/")
        if _is_excluded(norm_root, rel_root):
            continue
        for file in files:
            if not any(file.endswith(sfx) for sfx in valid_suffixes):
                continue
            path = os.path.join(root, file)
            rel_file = os.path.relpath(path, target_str).replace("\\", "/")
            norm_file = path.replace("\\", "/")
            if _is_excluded(norm_file, rel_file):
                continue
            found_files.append(Path(path))

    return found_files


def load_toml_section(target_file: Union[str, Path], section_name: str) -> Dict[str, Any]:
    """Loads a specific tool configuration section from a TOML file with fallbacks."""
    target_path = Path(target_file)
    if not target_path.exists():
        return {}

    # Standard library / tomli parser
    try:
        if sys.version_info >= (3, 11):
            import tomllib  # pylint: disable=import-outside-toplevel
        else:
            import tomli as tomllib  # pylint: disable=import-outside-toplevel
        with open(target_path, "rb") as fh:
            data = tomllib.load(fh)
        tool_sec = data.get("tool", {})
        if section_name in tool_sec and isinstance(tool_sec[section_name], dict):
            return dict(tool_sec[section_name])
        if any(k in data for k in ("threshold", "min_lines", "exemptions", "exclude")):
            return dict(data)
    except (ImportError, OSError):
        pass

    # Built-in zero-dependency line parser fallback
    try:
        config: Dict[str, Any] = {}
        in_section = False
        target_section = f"[tool.{section_name}]"
        with open(target_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        if not any(line.strip().startswith("[") for line in lines):
            in_section = True

        for line in lines:
            stripped = line.strip()
            if stripped == target_section:
                in_section = True
                continue
            if stripped.startswith("[") and in_section:
                in_section = False
                continue
            if in_section and "=" in stripped:
                k, v = stripped.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith('"') and v.endswith('"'):
                    config[k] = v[1:-1]
                elif v.lower() == "true":
                    config[k] = True
                elif v.lower() == "false":
                    config[k] = False
                else:
                    try:
                        config[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
        return config
    except OSError:
        return {}


def load_tool_config(
    config_path: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads tool configuration from pyproject.toml, .pydoppelgangerhunt.toml, or specified config path."""
    root = Path(repo_root) if repo_root else Path.cwd()
    candidates: List[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    else:
        candidates.extend([
            root / "pyproject.toml",
            root / ".pydoppelgangerhunt.toml",
        ])

    for candidate in candidates:
        if candidate.is_file():
            cfg = load_toml_section(candidate, "pydoppelgangerhunt")
            if cfg:
                return cfg
    return {}


def init_tool_configuration(target_dir: str = ".") -> str:
    """Initializes configuration for pyDoppelgangerHunt in pyproject.toml or standalone file."""
    pyproject_path = Path(target_dir) / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text(encoding="utf-8")
        if "[tool.pydoppelgangerhunt]" in content:
            return f"{pyproject_path} (already configured)"
        with open(pyproject_path, "a", encoding="utf-8") as fh:
            fh.write("\n" + DEFAULT_TOOL_TOML_CONTENT)
        return str(pyproject_path)

    standalone_path = Path(target_dir) / ".pydoppelgangerhunt.toml"
    standalone_path.write_text(DEFAULT_TOOL_TOML_CONTENT, encoding="utf-8")
    return str(standalone_path)
