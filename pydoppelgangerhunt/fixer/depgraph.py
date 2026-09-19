"""Module-level import dependency graph analysis, cycle detection, and package path resolution."""

from __future__ import annotations

import ast
import keyword
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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


def _resolve_repo_relative_path(
    file_path: Union[Path, str], p_root: Path
) -> Path:
    """Normalizes path separators and resolves relative paths against the repository root."""
    norm = str(file_path).replace("\\", "/")
    p = Path(norm)
    if not p.is_absolute():
        cand = (p_root / p).resolve()
        if cand.is_file() or cand.is_dir():
            return cand
        pkg_root = _find_enclosing_package_root(p_root)
        if pkg_root != p_root:
            cand_pkg = (pkg_root / p).resolve()
            if cand_pkg.is_file() or cand_pkg.is_dir():
                return cand_pkg
        if p.is_file() or p.is_dir():
            cand_cwd = p.resolve()
            try:
                cand_cwd.relative_to(p_root)
                return cand_cwd
            except ValueError:
                pass
        p = cand
    return p.resolve()


def _find_enclosing_package_root(path: Path) -> Path:
    """Finds the enclosing non-package directory (import root) for a module or directory.

    Walks ancestor directories to find the outermost ancestor containing an '__init__.py'.
    Returns that outermost package's parent directory. If no ancestor contains '__init__.py',
    checks if the path or any parent is named 'src', or contains a 'src' directory,
    falling back to the directory itself.
    """
    curr = path.resolve()
    if curr.is_file():
        curr = curr.parent

    # 1. Find highest ancestor that is still a Python package (contains __init__.py)
    highest_pkg: Optional[Path] = None
    node: Optional[Path] = curr
    while node is not None and node.parent != node:
        if (node / "__init__.py").is_file():
            highest_pkg = node
        node = node.parent

    if highest_pkg is not None:
        return highest_pkg.parent

    # 2. If no ancestor contains __init__.py, check if curr is inside a 'src' container
    for parent in (curr, *curr.parents):
        if parent.name == "src" and parent.parent != parent:
            return parent

    # 3. If curr contains a 'src' directory, the import root is 'src'
    if (curr / "src").is_dir():
        return curr / "src"

    return curr


def _find_project_filesystem_root(path: Path) -> Path:
    """Finds the project or repository filesystem root enclosing the given path.

    Walks ancestor directories checking for VCS metadata (.git, .hg, .svn)
    or build configurations (pyproject.toml, setup.py, setup.cfg).
    If nested inside a 'src' layout, returns the directory enclosing 'src'.
    Falls back to enclosing package root or the path itself.
    """
    curr = path.resolve()
    if curr.is_file():
        curr = curr.parent

    # 1. Search upwards for VCS metadata or build configurations
    scan_dir = curr
    while True:
        if (
            (scan_dir / ".git").exists()
            or (scan_dir / ".hg").is_dir()
            or (scan_dir / ".svn").is_dir()
            or (scan_dir / "pyproject.toml").is_file()
            or (scan_dir / "setup.py").is_file()
            or (scan_dir / "setup.cfg").is_file()
            or (scan_dir / "tox.ini").is_file()
        ):
            return scan_dir
        parent = scan_dir.parent
        if parent == scan_dir:
            break
        scan_dir = parent

    # 2. Check for standard 'src' container layout
    enclosing_pkg = _find_enclosing_package_root(curr)
    if enclosing_pkg.name == "src" and enclosing_pkg.parent != enclosing_pkg:
        return enclosing_pkg.parent

    # 3. Fallback to enclosing package root (or curr if no package context)
    return enclosing_pkg


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
    p_root = Path(repo_root).resolve()
    p_file = _resolve_repo_relative_path(file_path, p_root)
    try:
        rel = p_file.relative_to(p_root)
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
    dir1 = _parent_dir_of_path(_resolve_repo_relative_path(file1, p_root))
    dir2 = _parent_dir_of_path(_resolve_repo_relative_path(file2, p_root))

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
    # Sanitize shared_module_name to a safe, valid Python module filename
    raw_name = Path(str(shared_module_name)).name.strip()
    stem = raw_name[:-3] if raw_name.endswith(".py") else raw_name
    if not stem or not stem.isidentifier() or keyword.iskeyword(stem):
        clean_name = "_common.py"
    else:
        clean_name = f"{stem}.py"
    resolved_common = common_dir.resolve()
    target = common_dir / clean_name
    if target.is_symlink():
        raise ValueError(
            f"Shared module target path is an existing symlink: {target}"
        )
    try:
        target.resolve().relative_to(resolved_common)
        return target
    except ValueError:
        pass

    fallback = common_dir / "_common.py"
    if fallback.is_symlink():
        raise ValueError(
            f"Shared module fallback path is an existing symlink: {fallback}"
        )
    try:
        fallback.resolve().relative_to(resolved_common)
        return fallback
    except ValueError as exc:
        raise ValueError(
            f"Shared module path resolves outside common package directory {common_dir}"
        ) from exc


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
    effective_root = _find_enclosing_package_root(p_root)
    p_src = _resolve_repo_relative_path(source_file, p_root)
    p_shared = _resolve_repo_relative_path(shared_file, p_root)

    try:
        p_src.relative_to(effective_root)
        p_shared.relative_to(effective_root)
    except ValueError:
        return derive_module_import_path(p_shared, effective_root)

    if not prefer_relative:
        return derive_module_import_path(p_shared, effective_root)

    src_dir = _parent_dir_of_path(p_src)
    shared_dir = _parent_dir_of_path(p_shared)
    shared_stem = p_shared.stem if p_shared.stem != "__init__" else ""

    # Top-level source modules (at repo root or src/ root) have no enclosing package context;
    # relative imports with leading dots raise ImportError, so fall back to absolute module paths.
    if src_dir in (effective_root, effective_root / "src"):
        return derive_module_import_path(p_shared, effective_root)

    # When shared file is at repo root (or src/ root), any relative import from a subpackage
    # would climb above top-level package and raise ImportError; fall back to absolute path.
    if shared_dir in (effective_root, effective_root / "src"):
        return derive_module_import_path(p_shared, effective_root)

    try:
        rel_dir = os.path.relpath(str(shared_dir), str(src_dir))
    except ValueError:
        return derive_module_import_path(p_shared, effective_root)

    if rel_dir == ".":
        return f".{shared_stem}" if shared_stem else "."

    rel_parts = Path(rel_dir).parts
    up_count = sum(1 for part in rel_parts if part == "..")
    down_parts = [part for part in rel_parts if part != ".."]

    # Check if up_count climbs above the top-level package enclosing src_dir
    src_base = effective_root / "src"
    try:
        pkg_depth = len(src_dir.relative_to(src_base).parts)
    except ValueError:
        try:
            pkg_depth = len(src_dir.relative_to(effective_root).parts)
        except ValueError:
            pkg_depth = 0
    if up_count >= pkg_depth:
        return derive_module_import_path(p_shared, effective_root)

    dots = "." * (up_count + 1)
    if down_parts:
        mod_prefix = ".".join(down_parts)
        return f"{dots}{mod_prefix}.{shared_stem}" if shared_stem else f"{dots}{mod_prefix}"
    return f"{dots}{shared_stem}" if shared_stem else dots


def _resolve_relative_import_path(
    current_mod: str, level: int, module_name: Optional[str], is_package: bool = False
) -> str:
    """Resolves a relative import (level > 0) to a canonical module dot-path."""
    parts = current_mod.split(".") if current_mod else []
    if not parts:
        return module_name or ""
    pkg_parts = parts if is_package else parts[:-1]
    if (level - 1) > len(pkg_parts):
        return module_name or ""
    target_base = (
        pkg_parts[: len(pkg_parts) - (level - 1)]
        if len(pkg_parts) >= (level - 1)
        else []
    )
    base_str = ".".join(target_base)
    if base_str and module_name:
        return f"{base_str}.{module_name}"
    return base_str or module_name or ""


def _is_type_checking_guard(test_node: ast.expr) -> bool:
    """Detects whether an AST expression represents a static type-checking guard."""
    if isinstance(test_node, ast.Name) and test_node.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test_node, ast.Attribute)
        and test_node.attr == "TYPE_CHECKING"
        and isinstance(test_node.value, ast.Name)
        and test_node.value.id in ("typing", "typing_extensions")
    ):
        return True
    if isinstance(test_node, ast.Constant) and test_node.value in (False, 0):
        return True
    cmp_info = _extract_guarded_compare_target(test_node)
    if cmp_info is not None:
        target, expected = cmp_info
        return expected and _is_type_checking_guard(target)
    return False


def _extract_guarded_compare_target(
    test_node: ast.expr,
) -> Optional[Tuple[ast.expr, bool]]:
    """Extracts comparison target and resolved boolean truth value for single-comparison guards."""
    if (
        isinstance(test_node, ast.Compare)
        and len(test_node.ops) == 1
        and len(test_node.comparators) == 1
        and isinstance(test_node.comparators[0], ast.Constant)
        and isinstance(test_node.comparators[0].value, bool)
    ):
        op = test_node.ops[0]
        val = bool(test_node.comparators[0].value)
        if isinstance(op, (ast.Is, ast.Eq)):
            return test_node.left, val
        if isinstance(op, (ast.IsNot, ast.NotEq)):
            return test_node.left, not val
    return None


def _is_inverted_type_checking_guard(test_node: ast.expr) -> bool:
    """Detects whether an AST expression inverts a static type-checking guard."""
    if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
        return _is_type_checking_guard(test_node.operand)
    cmp_info = _extract_guarded_compare_target(test_node)
    if cmp_info is not None:
        target, expected = cmp_info
        return (not expected) and _is_type_checking_guard(target)
    return False


def _collect_top_level_import_nodes(
    stmts: Sequence[ast.stmt],
) -> List[Union[ast.Import, ast.ImportFrom]]:
    """Recursively collects module-level runtime import nodes, skipping type-checking and functions."""
    result: List[Union[ast.Import, ast.ImportFrom]] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            result.append(stmt)
        elif isinstance(stmt, ast.If):
            if _is_type_checking_guard(stmt.test):
                result.extend(_collect_top_level_import_nodes(stmt.orelse))
            elif _is_inverted_type_checking_guard(stmt.test):
                result.extend(_collect_top_level_import_nodes(stmt.body))
            else:
                result.extend(_collect_top_level_import_nodes(stmt.body))
                result.extend(_collect_top_level_import_nodes(stmt.orelse))
        elif isinstance(stmt, ast.Try):
            result.extend(_collect_top_level_import_nodes(stmt.body))
            for handler in stmt.handlers:
                result.extend(_collect_top_level_import_nodes(handler.body))
            result.extend(_collect_top_level_import_nodes(stmt.orelse))
            result.extend(_collect_top_level_import_nodes(stmt.finalbody))
        elif hasattr(ast, "TryStar") and isinstance(stmt, getattr(ast, "TryStar")):
            result.extend(_collect_top_level_import_nodes(stmt.body))
            for handler in stmt.handlers:
                result.extend(_collect_top_level_import_nodes(handler.body))
            result.extend(_collect_top_level_import_nodes(stmt.orelse))
            result.extend(_collect_top_level_import_nodes(stmt.finalbody))
        elif isinstance(stmt, ast.ClassDef):
            result.extend(_collect_top_level_import_nodes(stmt.body))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            result.extend(_collect_top_level_import_nodes(stmt.body))
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            result.extend(_collect_top_level_import_nodes(stmt.body))
            result.extend(_collect_top_level_import_nodes(stmt.orelse))
        elif isinstance(stmt, ast.ClassDef):
            result.extend(_collect_top_level_import_nodes(stmt.body))
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):
            for case in getattr(stmt, "cases", []):
                result.extend(_collect_top_level_import_nodes(case.body))
    return result


def _parse_source_imports(
    source_text: str, current_mod: str, is_package: bool = False
) -> Set[str]:
    """Extracts module-level imported module dot-paths executed at runtime."""
    imports: Set[str] = set()
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return imports

    for node in _collect_top_level_import_nodes(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level = getattr(node, "level", 0) or 0
            raw_mod = getattr(node, "module", None)
            if level > 0:
                resolved = _resolve_relative_import_path(
                    current_mod, level, raw_mod, is_package=is_package
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

    def __init__(
        self,
        repo_root: Union[Path, str],
        *,
        adjacency: Optional[Dict[str, Set[str]]] = None,
        mod_to_file: Optional[Dict[str, Path]] = None,
        file_to_mod: Optional[Dict[str, str]] = None,
        pending_descendants: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Initializes a module dependency graph scoped to a repository root."""
        self.repo_root = Path(repo_root).resolve()
        self.adjacency: Dict[str, Set[str]] = (
            {k: v.copy() for k, v in adjacency.items()} if adjacency else {}
        )
        self.mod_to_file: Dict[str, Path] = mod_to_file.copy() if mod_to_file else {}
        self.file_to_mod: Dict[str, str] = file_to_mod.copy() if file_to_mod else {}
        self._sorted_adjacency: Dict[str, List[str]] = {}
        self._pending_descendants: Dict[str, Set[str]] = (
            {k: v.copy() for k, v in pending_descendants.items()}
            if pending_descendants
            else {}
        )

    def add_module(self, mod_name: str, file_path: Path) -> None:
        """Registers a module and its backing file path in the graph."""
        if not mod_name:
            return
        if mod_name not in self.adjacency:
            self.adjacency[mod_name] = set()
        resolved = file_path.resolve()
        self.mod_to_file[mod_name] = resolved
        self.file_to_mod[str(resolved).replace("\\", "/").lower()] = mod_name

        # In Python, importing a submodule first executes its ancestor package initializers
        parts = mod_name.split(".")
        for i in range(1, len(parts)):
            parent_pkg = ".".join(parts[:i])
            if parent_pkg in self.mod_to_file:
                self.add_dependency(mod_name, parent_pkg)
            else:
                self._pending_descendants.setdefault(parent_pkg, set()).add(mod_name)

        if mod_name in self._pending_descendants:
            for desc_mod in self._pending_descendants.pop(mod_name):
                self.add_dependency(desc_mod, mod_name)

    def add_dependency(self, from_mod: str, to_mod: str) -> None:
        """Adds a directed import edge from from_mod to to_mod."""
        if not from_mod or not to_mod or from_mod == to_mod:
            return
        if from_mod not in self.adjacency:
            self.adjacency[from_mod] = set()
        if to_mod not in self.adjacency[from_mod]:
            self.adjacency[from_mod].add(to_mod)
            self._sorted_adjacency.pop(from_mod, None)

    def _get_sorted_neighbors(self, mod_name: str) -> List[str]:
        """Returns deterministic sorted neighbors of mod_name, using a cached list."""
        cached = self._sorted_adjacency.get(mod_name)
        if cached is None:
            neighbors = self.adjacency.get(mod_name)
            cached = sorted(neighbors) if neighbors else []
            self._sorted_adjacency[mod_name] = cached
        return cached

    def copy(self) -> "ModuleDependencyGraph":
        """Returns an isolated shallow copy of the dependency graph with copied adjacency collections."""
        return ModuleDependencyGraph(
            self.repo_root,
            adjacency=self.adjacency,
            mod_to_file=self.mod_to_file,
            file_to_mod=self.file_to_mod,
            pending_descendants=self._pending_descendants,
        )

    def resolve_import_target(
        self,
        raw_import: str,
        known_modules: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Resolves a raw import string to the matching known internal module or package name."""
        if not raw_import:
            return None
        if known_modules is None:
            known_modules = set(self.mod_to_file.keys())
        if raw_import in known_modules:
            return raw_import
        parts = raw_import.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in known_modules:
                return prefix
        return None

    def add_import_dependency(
        self,
        from_mod: str,
        raw_import: str,
        known_modules: Optional[Set[str]] = None,
    ) -> None:
        """Adds a dependency edge from from_mod to raw_import, matching package prefixes."""
        if not from_mod or not raw_import:
            return
        target = self.resolve_import_target(raw_import, known_modules=known_modules)
        if target:
            self.add_dependency(from_mod, target)

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
            for neighbor in self._get_sorted_neighbors(curr):
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
            for neighbor in self._get_sorted_neighbors(curr):
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

        # Submodule import executes ancestor package initializers:
        # e.g., importing `pkg.worker` first executes `pkg/__init__.py`.
        # If `pkg` already depends on `from_mod`, caller -> pkg.worker -> pkg -> caller forms a cycle.
        parts = to_mod.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            prefix_path = self.find_cycle_path(from_mod=prefix, to_mod=from_mod)
            if prefix_path is not None:
                return [from_mod, to_mod] + prefix_path
        return None

    def check_cycle_if_imports_added(
        self,
        from_mod: str,
        import_stmts: Sequence[str],
        file_path: Optional[Path] = None,
        is_package: bool = False,
    ) -> Optional[List[str]]:
        """Checks whether introducing a set of imports to from_mod would create any cycle in the graph."""
        if not from_mod or not import_stmts:
            return None

        tentative = self.copy()
        if file_path is not None:
            tentative.add_module(from_mod, file_path)
        elif from_mod not in tentative.adjacency:
            tentative.adjacency[from_mod] = set()

        raw_imports = _parse_source_imports(
            "\n".join(import_stmts), from_mod, is_package=is_package
        )
        known = set(tentative.mod_to_file.keys())

        for raw_imp in sorted(raw_imports):
            target = tentative.resolve_import_target(raw_imp, known_modules=known)
            if not target:
                continue
            if target == from_mod:
                return [from_mod, target]
            cycle = tentative.check_cycle_if_added(from_mod, target)
            if cycle is not None:
                return cycle
            tentative.add_dependency(from_mod, target)
        return None

    @classmethod
    def build_from_repository(
        cls,
        repo_root: Union[Path, str],
        file_paths: Optional[Sequence[Path]] = None,
    ) -> ModuleDependencyGraph:
        """Builds a populated dependency graph across the repository's Python source files."""
        root = Path(repo_root).resolve()
        effective_root = _find_enclosing_package_root(root)
        graph = cls(effective_root)

        if file_paths is not None:
            python_files = [
                _resolve_repo_relative_path(p, root)
                for p in file_paths
                if p.suffix == ".py"
            ]
        else:
            python_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDED_GRAPH_DIRS]
                for filename in filenames:
                    if filename.endswith(".py"):
                        python_files.append(Path(dirpath) / filename)

        # First pass: map all modules
        for p_file in python_files:
            mod_name = derive_module_import_path(p_file, effective_root)
            if mod_name:
                graph.add_module(mod_name, p_file)

        known_modules = set(graph.mod_to_file.keys())

        # Second pass: parse imports and wire edges
        for p_file in python_files:
            mod_name = derive_module_import_path(p_file, effective_root)
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
                graph.add_import_dependency(mod_name, imp, known_modules=known_modules)
        return graph


def build_module_graph(
    repo_root: Union[Path, str],
    file_paths: Optional[Sequence[Path]] = None,
) -> ModuleDependencyGraph:
    """Convenience helper to construct a ModuleDependencyGraph for a repository."""
    return ModuleDependencyGraph.build_from_repository(repo_root, file_paths=file_paths)
