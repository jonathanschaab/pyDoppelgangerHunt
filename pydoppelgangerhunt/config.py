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


def normalize_path_string(path_str: Optional[str], strip_anchor: bool = True) -> str:
    """Normalizes a file path string by optionally stripping anchors, converting backslashes, and stripping leading './'."""
    if not path_str:
        return ""
    raw = str(path_str)
    if strip_anchor and "#" in raw:
        raw = raw.split("#", maxsplit=1)[0]
    norm = raw.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm



def paths_match_boundary(p1: Optional[str], p2: Optional[str]) -> bool:
    """Checks whether two normalized paths refer to the same file respecting directory boundaries."""
    n1 = normalize_path_string(p1)
    n2 = normalize_path_string(p2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1.endswith("/" + n2) or n2.endswith("/" + n1):
        return True
    if os.name == "nt" or sys.platform == "win32":
        n1_lower = n1.lower()
        n2_lower = n2.lower()
        if n1_lower == n2_lower:
            return True
        if n1_lower.endswith("/" + n2_lower) or n2_lower.endswith("/" + n1_lower):
            return True
    return False


def find_matching_path_value(
    target_path: Optional[str],
    path_map: Dict[str, Any],
) -> Any:
    """Finds a value in a path-keyed mapping where keys match target_path respecting directory boundaries."""
    target_norm = normalize_path_string(target_path)
    if not target_norm or not path_map:
        return None
    direct = path_map.get(target_norm)
    if direct is not None:
        return direct
    for k, val in path_map.items():
        if paths_match_boundary(target_norm, k):
            return val
    return None


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
        if ex.strip("./\\") and ex.replace("\\", "/").strip("./") not in target_clean
    ]
    if audit_tests:
        active_excludes = [ex for ex in active_excludes if "test" not in ex]

    def _is_excluded(full_p: str, rel_p: str) -> bool:
        for ex in active_excludes:
            clean_ex = ex.replace("\\", "/").strip("/")
            if clean_ex and (clean_ex in full_p or clean_ex in rel_p):
                return True
        return False

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
            import tomli as tomllib  # type: ignore[import-not-found,unused-ignore] # pylint: disable=import-outside-toplevel
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
        try:
            content = pyproject_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            content = ""
        if "[tool.pydoppelgangerhunt]" in content:
            return f"{pyproject_path} (already configured)"
        with open(pyproject_path, "a", encoding="utf-8") as fh:
            fh.write("\n" + DEFAULT_TOOL_TOML_CONTENT)
        return str(pyproject_path)

    standalone_path = Path(target_dir) / ".pydoppelgangerhunt.toml"
    standalone_path.write_text(DEFAULT_TOOL_TOML_CONTENT, encoding="utf-8")
    return str(standalone_path)
