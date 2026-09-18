"""Module-level import dependency graph analysis, cycle detection, and package path resolution."""

from __future__ import annotations

import ast
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

EXCLUDED_GRAPH_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    ".gemini",
    "scratch",
}


def derive_module_import_path(
    file_path: Union[Path, str], repo_root: Union[Path, str]
) -> str:
    """Derives the importable Python module dot-path for a file relative to repository root.

    Handles flat layouts (foo.py -> foo), package layouts (pkg/mod.py -> pkg.mod),
    PEP 517/518 src layouts (src/pkg/mod.py -> pkg.mod), and __init__.py files
    (pkg/__init__.py -> pkg).

    Args:
        file_path: Absolute or relative path to the Python source file.
        repo_root: Root directory of the repository or project.

    Returns:
        Dot-separated module import path, or an empty string if unresolvable.
    """
    p_file = Path(file_path)
    p_root = Path(repo_root)
    try:
        rel = p_file.resolve().relative_to(p_root.resolve())
    except ValueError:
        rel = Path(p_file.name)
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return ""
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parent_dir_of_path(path: Path) -> Path:
    """Returns the enclosing directory for a file path or directory."""
    return path.parent if path.is_file() or path.suffix == ".py" else path


def find_nearest_common_package(
    file1: Union[Path, str], file2: Union[Path, str], repo_root: Union[Path, str]
) -> Path:
    """Finds the lowest common directory containing both files within the repository.

    If the files reside in subpackages of a common package (or src/pkg), that package
    directory is returned. If they only share the repo root, repo_root is returned.

    Args:
        file1: Path to the first file.
        file2: Path to the second file.
        repo_root: Root directory of the repository.

    Returns:
        Path to the nearest common directory.
    """
    p_root = Path(repo_root).resolve()
    dir1 = _parent_dir_of_path(Path(file1).resolve())
    dir2 = _parent_dir_of_path(Path(file2).resolve())

    try:
        rel1 = dir1.relative_to(p_root)
        rel2 = dir2.relative_to(p_root)
    except ValueError:
        return p_root

    common_parts: List[str] = []
    for part1, part2 in zip(rel1.parts, rel2.parts):
        if part1 == part2:
            common_parts.append(part1)
        else:
            break

    if common_parts:
        return p_root.joinpath(*common_parts)
    return p_root


def resolve_shared_module_file(
    file1: Union[Path, str],
    file2: Union[Path, str],
    repo_root: Union[Path, str],
    shared_module_name: str = "_common.py",
) -> Path:
    """Resolves the target file path for a synthesized shared utility module.

    Args:
        file1: Path to first duplicate file.
        file2: Path to second duplicate file.
        repo_root: Project repository root.
        shared_module_name: Desired file name for the shared helper module (default: '_common.py').

    Returns:
        Path to the shared module file within the nearest common directory.
    """
    common_dir = find_nearest_common_package(file1, file2, repo_root)
    clean_name = (
        shared_module_name
        if shared_module_name.endswith(".py")
        else f"{shared_module_name}.py"
    )
    return common_dir / clean_name


def derive_shared_module_import(
    source_file: Union[Path, str],
    shared_file: Union[Path, str],
    repo_root: Union[Path, str],
    prefer_relative: bool = False,
) -> str:
    """Derives the module import string for importing from shared_file into source_file.

    Args:
        source_file: Calling source file that will contain the import statement.
        shared_file: Shared utility file containing the extracted helper.
        repo_root: Repository root path.
        prefer_relative: Whether to generate relative dot imports where feasible.

    Returns:
        Module path suitable for 'from <module> import <helper>'.
    """
    p_root = Path(repo_root).resolve()
    p_src = Path(source_file).resolve()
    p_shared = Path(shared_file).resolve()

    if not prefer_relative:
        return derive_module_import_path(p_shared, p_root)

    src_dir = _parent_dir_of_path(p_src)
    shared_dir = _parent_dir_of_path(p_shared)
    shared_stem = p_shared.stem if p_shared.stem != "__init__" else ""

    try:
        rel_dir = os.path.relpath(str(shared_dir), str(src_dir))
    except ValueError:
        return derive_module_import_path(p_shared, p_root)

    if rel_dir == ".":
        return f".{shared_stem}" if shared_stem else "."

    rel_parts = Path(rel_dir).parts
    up_count = sum(1 for part in rel_parts if part == "..")
    down_parts = [part for part in rel_parts if part != ".."]

    dots = "." * (up_count + 1)
    if down_parts:
        mod_prefix = ".".join(down_parts)
        return f"{dots}{mod_prefix}.{shared_stem}" if shared_stem else f"{dots}{mod_prefix}"
    return f"{dots}{shared_stem}" if shared_stem else dots


def _resolve_relative_import_path(
    current_mod: str, level: int, module_name: Optional[str]
) -> str:
    """Resolves a relative import (level > 0) to a canonical module dot-path."""
    parts = current_mod.split(".") if current_mod else []
    # If current_mod is empty, relative resolution is impossible
    if not parts:
        return module_name or ""
    # In Python, from . import x inside pkg.mod means x is inside pkg (up 1 level).
    # from .. import x means up 2 levels.
    # So we strip 'level' elements from the end of parts if parts represent a module.
    # Note: caller passes current_pkg or current_mod.
    if level > len(parts):
        return module_name or ""
    base_parts = parts[: len(parts) - level]
    suffix = f".{module_name}" if module_name else ""
    return ".".join(base_parts) + suffix if base_parts else (module_name or "")


def _parse_source_imports(
    source_text: str, current_mod: str, is_package: bool = False
) -> Set[str]:
    """Extracts imported module dot-paths from Python source text via AST traversal."""
    imports: Set[str] = set()
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level = getattr(node, "level", 0) or 0
            raw_mod = getattr(node, "module", None)
            if level > 0:
                mod_parts = current_mod.split(".") if current_mod else []
                pkg_parts = mod_parts if is_package else mod_parts[:-1]
                target_base = (
                    pkg_parts[: len(pkg_parts) - (level - 1)]
                    if len(pkg_parts) >= (level - 1)
                    else []
                )
                base_str = ".".join(target_base)
                resolved = (
                    f"{base_str}.{raw_mod}"
                    if (base_str and raw_mod)
                    else (base_str or raw_mod or "")
                )
                if resolved:
                    imports.add(resolved)
                for alias in node.names:
                    full_cand = f"{resolved}.{alias.name}" if resolved else alias.name
                    imports.add(full_cand)
            elif raw_mod:
                imports.add(raw_mod)
                for alias in node.names:
                    imports.add(f"{raw_mod}.{alias.name}")
    return imports


class ModuleDependencyGraph:
    """Project-wide module import dependency graph for reachability and cycle detection."""

    def __init__(self, repo_root: Union[Path, str]) -> None:
        """Initializes an empty module dependency graph scoped to a repository root."""
        self.repo_root = Path(repo_root).resolve()
        self.adjacency: Dict[str, Set[str]] = {}
        self.mod_to_file: Dict[str, Path] = {}
        self.file_to_mod: Dict[str, str] = {}

    def add_module(self, mod_name: str, file_path: Path) -> None:
        """Registers a module and its backing file path in the graph."""
        if not mod_name:
            return
        if mod_name not in self.adjacency:
            self.adjacency[mod_name] = set()
        resolved = file_path.resolve()
        self.mod_to_file[mod_name] = resolved
        self.file_to_mod[str(resolved).replace("\\", "/").lower()] = mod_name

    def add_dependency(self, from_mod: str, to_mod: str) -> None:
        """Adds a directed import edge from from_mod to to_mod."""
        if not from_mod or not to_mod or from_mod == to_mod:
            return
        if from_mod not in self.adjacency:
            self.adjacency[from_mod] = set()
        self.adjacency[from_mod].add(to_mod)

    def get_dependencies(self, mod_name: str) -> Set[str]:
        """Returns direct dependency module names for a given module."""
        return self.adjacency.get(mod_name, set()).copy()

    def has_transitive_path(self, from_mod: str, to_mod: str) -> bool:
        """Determines reachability from from_mod to to_mod via BFS search in O(V + E) time."""
        if not from_mod or not to_mod:
            return False
        if from_mod == to_mod:
            return True
        visited: Set[str] = {from_mod}
        queue: deque[str] = deque([from_mod])

        while queue:
            curr = queue.popleft()
            for neighbor in self.adjacency.get(curr, ()):
                # If neighbor is exact target or a prefix of target module
                if neighbor == to_mod:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def find_cycle_path(self, from_mod: str, to_mod: str) -> Optional[List[str]]:
        """Finds a directed path from from_mod to to_mod if one exists, returning module names."""
        if not from_mod or not to_mod:
            return None
        if from_mod == to_mod:
            return [from_mod]

        parent: Dict[str, str] = {}
        visited: Set[str] = {from_mod}
        queue: deque[str] = deque([from_mod])
        found = False

        while queue:
            curr = queue.popleft()
            if curr == to_mod:
                found = True
                break
            for neighbor in self.adjacency.get(curr, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        if not found and to_mod not in parent:
            return None

        # Reconstruct path
        path: List[str] = [to_mod]
        curr_node = to_mod
        while curr_node in parent:
            curr_node = parent[curr_node]
            path.append(curr_node)
            if curr_node == from_mod:
                break
        path.reverse()
        return path

    def check_cycle_if_added(
        self, from_mod: str, to_mod: str
    ) -> Optional[List[str]]:
        """Checks whether introducing an edge from from_mod -> to_mod would close a cycle.

        Returns:
            The cycle path (e.g. ['mod_a', 'mod_b', 'mod_a']) if a cycle would form;
            None if the edge is safe and acyclic.
        """
        if from_mod == to_mod:
            return [from_mod, to_mod]
        existing_path = self.find_cycle_path(from_mod=to_mod, to_mod=from_mod)
        if existing_path is not None:
            return [from_mod] + existing_path
        return None

    @classmethod
    def build_from_repository(
        cls,
        repo_root: Union[Path, str],
        file_paths: Optional[Sequence[Path]] = None,
    ) -> ModuleDependencyGraph:
        """Builds a populated dependency graph across the repository's Python source files."""
        root = Path(repo_root).resolve()
        graph = cls(root)

        if file_paths is not None:
            python_files = [p.resolve() for p in file_paths if p.suffix == ".py"]
        else:
            python_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDED_GRAPH_DIRS]
                for filename in filenames:
                    if filename.endswith(".py"):
                        python_files.append(Path(dirpath) / filename)

        # First pass: map all modules
        for p_file in python_files:
            mod_name = derive_module_import_path(p_file, root)
            if mod_name:
                graph.add_module(mod_name, p_file)

        known_modules = set(graph.mod_to_file.keys())

        # Second pass: parse imports and wire edges
        for p_file in python_files:
            mod_name = derive_module_import_path(p_file, root)
            if not mod_name:
                continue
            try:
                content = p_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            raw_imports = _parse_source_imports(
                content, mod_name, is_package=(p_file.name == "__init__.py")
            )
            for imp in raw_imports:
                if imp in known_modules:
                    graph.add_dependency(mod_name, imp)
                else:
                    # Check prefix matching against known packages
                    parts = imp.split(".")
                    for i in range(len(parts), 0, -1):
                        prefix = ".".join(parts[:i])
                        if prefix in known_modules:
                            graph.add_dependency(mod_name, prefix)
                            break
        return graph


def build_module_graph(
    repo_root: Union[Path, str],
    file_paths: Optional[Sequence[Path]] = None,
) -> ModuleDependencyGraph:
    """Convenience helper to construct a ModuleDependencyGraph for a repository."""
    return ModuleDependencyGraph.build_from_repository(repo_root, file_paths=file_paths)
