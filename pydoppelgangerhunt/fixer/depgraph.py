"""Module-level import dependency graph analysis, cycle detection, and package path resolution."""

from __future__ import annotations

import ast
import keyword
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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


def _is_within_root(path: Path, root: Path) -> bool:
    """Checks whether a resolved path is contained within the given root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError, RuntimeError):
        return False


def _resolve_root_path(repo_root: Union[Path, str]) -> Path:
    """Safely resolves a repository or root path, falling back on filesystem error."""
    try:
        return Path(repo_root).resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(repo_root)


def _rejected_outside_root_path(root: Path) -> Path:
    """Returns a non-colliding sentinel path for rejected candidates outside the root."""
    resolved = _resolve_root_path(root)
    return resolved / ".git" / ".pydoppelgangerhunt-invalid-path" / "__outside_root__"


def _has_symlink_component(path: Path, root: Optional[Path] = None) -> bool:
    """Checks whether a path or any ancestor component below root is a symbolic link."""
    try:
        if path.is_symlink():
            return True
        if root is not None:
            resolved_root = _resolve_root_path(root)
            check_path = path if path.is_absolute() else (resolved_root / path)
            if check_path.is_symlink():
                return True
            for parent in check_path.parents:
                if parent != resolved_root and resolved_root in parent.parents:
                    if parent.is_symlink():
                        return True
        return False
    except (OSError, RuntimeError, ValueError):
        return True


def _is_safe_repo_python_path(
    path: Union[Path, str], root: Path
) -> Tuple[Optional[Path], bool]:
    """Validates whether a path is safely contained within root and free of symlinks.

    Returns:
        (resolved_path, is_rejected):
        - (resolved_path, False) if safe, non-symlinked, and exists as a regular file.
        - (None, False) if safe and non-symlinked within root, but does not exist on disk.
        - (None, True) if the path violates security boundaries (outside root, symlink,
          non-Python, or unresolvable).
    """
    try:
        p = Path(path)
        if p.suffix != ".py":
            return None, True
        if _has_symlink_component(p, root):
            return None, True
        resolved_root = root.resolve()
        if not p.is_absolute() and _has_symlink_component(resolved_root / p, resolved_root):
            return None, True
        cand = _resolve_repo_relative_path(p, resolved_root)
        sentinel = _rejected_outside_root_path(resolved_root)
        if cand == sentinel or _has_symlink_component(cand, resolved_root):
            return None, True
        resolved = cand.resolve()
        if (
            _has_symlink_component(resolved, resolved_root)
            or not _is_within_root(resolved, resolved_root)
            or resolved == sentinel
        ):
            return None, True
        if resolved.is_file():
            return resolved, False
        if not resolved.exists():
            return None, False
        return None, True
    except (OSError, RuntimeError, ValueError):
        return None, True


def _is_safe_repo_python_file(path: Union[Path, str], root: Path) -> Optional[Path]:
    """Validates that a path is a regular, non-symlinked Python file strictly within root.

    Returns the canonical resolved Path if safe, or None if the path is a symlink,
    resolves outside the root boundary, or is invalid.
    """
    resolved, is_rejected = _is_safe_repo_python_path(path, root)
    return resolved if not is_rejected else None


def _resolve_repo_relative_path(
    file_path: Union[Path, str], p_root: Path
) -> Path:
    """Normalizes path separators and resolves relative paths against the repository root."""
    try:
        norm = str(file_path).replace("\\", "/")
        p = Path(norm)
        resolved_root = p_root.resolve()
        sentinel = _rejected_outside_root_path(resolved_root)
        if _has_symlink_component(p, resolved_root):
            return sentinel
        pkg_root = _find_enclosing_package_root(resolved_root)
        if not p.is_absolute():
            unresolved = resolved_root / p
            if _has_symlink_component(unresolved, resolved_root):
                return sentinel
            cand = unresolved.resolve()
            if _has_symlink_component(cand, resolved_root):
                return sentinel
            if _is_within_root(cand, resolved_root) and (cand.is_file() or cand.is_dir()):
                return cand
            if pkg_root != resolved_root:
                unresolved_pkg = pkg_root / p
                if _has_symlink_component(unresolved_pkg, resolved_root):
                    return sentinel
                cand_pkg = unresolved_pkg.resolve()
                if _has_symlink_component(cand_pkg, resolved_root):
                    return sentinel
                if _is_within_root(cand_pkg, resolved_root) and (
                    cand_pkg.is_file() or cand_pkg.is_dir()
                ):
                    return cand_pkg
            if p.is_file() or p.is_dir():
                if _has_symlink_component(p, resolved_root):
                    return sentinel
                cand_cwd = p.resolve()
                if _has_symlink_component(cand_cwd, resolved_root):
                    return sentinel
                if _is_within_root(cand_cwd, resolved_root):
                    return cand_cwd
            if not _is_within_root(cand, resolved_root):
                return sentinel
            return cand
        resolved_path = p.resolve()
        if _has_symlink_component(resolved_path, resolved_root):
            return sentinel
        if _is_within_root(resolved_path, resolved_root):
            return resolved_path
        return sentinel
    except (OSError, RuntimeError, ValueError):
        return _rejected_outside_root_path(p_root)


def _safe_parent_dir(path: Path) -> Path:
    """Safely returns the parent directory if path is a file, or path itself."""
    curr = _resolve_root_path(path)
    try:
        return curr.parent if curr.is_file() else curr
    except (OSError, RuntimeError, ValueError):
        return curr


def _find_enclosing_package_root(path: Path) -> Path:
    """Finds the enclosing non-package directory (import root) for a module or directory.

    Walks ancestor directories to find the outermost ancestor containing an '__init__.py'.
    Returns that outermost package's parent directory. If no ancestor contains '__init__.py',
    checks if the path or any parent is named 'src', or contains a 'src' directory,
    falling back to the directory itself.
    """
    curr = _safe_parent_dir(path)

    # 1. Find highest ancestor that is still a Python package (contains __init__.py)
    highest_pkg: Optional[Path] = None
    node: Optional[Path] = curr
    while node is not None and node.parent != node:
        try:
            if (node / "__init__.py").is_file():
                highest_pkg = node
        except (OSError, RuntimeError, ValueError):
            break
        node = node.parent

    if highest_pkg is not None:
        return highest_pkg.parent

    # 2. If no ancestor contains __init__.py, check if curr is inside a 'src' container
    for parent in (curr, *curr.parents):
        if parent.name == "src" and parent.parent != parent:
            return parent

    # 3. If curr contains a 'src' directory, the import root is 'src'
    try:
        if (curr / "src").is_dir():
            return curr / "src"
    except (OSError, RuntimeError, ValueError):
        pass

    return curr


def _find_project_filesystem_root(path: Path) -> Path:
    """Finds the project or repository filesystem root enclosing the given path.

    Walks ancestor directories checking for VCS metadata (.git, .hg, .svn)
    or build configurations (pyproject.toml, setup.py, setup.cfg).
    If nested inside a 'src' layout, returns the directory enclosing 'src'.
    Falls back to enclosing package root or the path itself.
    """
    curr = _safe_parent_dir(path)

    # 1. Search upwards for VCS metadata or build configurations
    scan_dir = curr
    while True:
        try:
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
        except (OSError, RuntimeError, ValueError):
            break
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
    p_root = _resolve_root_path(repo_root)
    p_file = _resolve_repo_relative_path(file_path, p_root)
    if p_file == _rejected_outside_root_path(p_root):
        rel = Path(Path(str(file_path)).name)
    else:
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
    if path.suffix == ".py":
        return path.parent
    return _safe_parent_dir(path)


def _resolve_candidate_pair(
    p_root: Path, file1: Union[Path, str], file2: Union[Path, str]
) -> Tuple[Path, Path, bool]:
    """Resolves a pair of candidate paths against root, returning paths and rejection status."""
    sentinel = _rejected_outside_root_path(p_root)
    p1 = _resolve_repo_relative_path(file1, p_root)
    p2 = _resolve_repo_relative_path(file2, p_root)
    return p1, p2, sentinel in (p1, p2)


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
    p_root = _resolve_root_path(repo_root)
    p_file1, p_file2, has_rejected = _resolve_candidate_pair(p_root, file1, file2)
    if has_rejected:
        return p_root

    dirs = (_parent_dir_of_path(p_file1), _parent_dir_of_path(p_file2))
    try:
        rel_dirs = [d.relative_to(p_root) for d in dirs]
    except ValueError:
        return p_root

    common_parts: List[str] = []
    for part1, part2 in zip(rel_dirs[0].parts, rel_dirs[1].parts):
        if part1 != part2:
            break
        common_parts.append(part1)

    return p_root.joinpath(*common_parts) if common_parts else p_root


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
    try:
        resolved_common = common_dir.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Common package directory resolution failed for {common_dir}: {exc}"
        ) from exc
    target = common_dir / clean_name
    try:
        if target.is_symlink():
            raise ValueError(
                f"Shared module target path is an existing symlink: {target}"
            )
        if target.is_dir():
            if clean_name == "_common.py":
                raise ValueError(
                    f"shared module path {clean_name} is an existing directory"
                )
        else:
            try:
                target.resolve().relative_to(resolved_common)
                return target
            except (ValueError, OSError, RuntimeError):
                pass
    except (OSError, RuntimeError) as exc:
        if clean_name == "_common.py":
            raise ValueError(
                f"Shared module target path error: {exc}"
            ) from exc

    fallback = common_dir / "_common.py"
    try:
        if fallback.is_symlink():
            raise ValueError(
                f"Shared module fallback path is an existing symlink: {fallback}"
            )
        if fallback.is_dir():
            raise ValueError(
                "shared module path _common.py is an existing directory"
            )
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Shared module fallback path error: {exc}"
        ) from exc
    try:
        fallback.resolve().relative_to(resolved_common)
        return fallback
    except (ValueError, OSError, RuntimeError) as exc:
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
    p_root = _resolve_root_path(repo_root)
    effective_root = _find_enclosing_package_root(p_root)
    p_src, p_shared, has_rejected = _resolve_candidate_pair(
        p_root, source_file, shared_file
    )

    if has_rejected or not (
        _is_within_root(p_src, effective_root)
        and _is_within_root(p_shared, effective_root)
    ):
        return derive_module_import_path(shared_file, effective_root)

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
    if (level - 1) >= len(pkg_parts):
        return module_name or ""
    target_base = pkg_parts[: len(pkg_parts) - (level - 1)]
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
    ):
        left = test_node.left
        right = test_node.comparators[0]
        op = test_node.ops[0]
        for const_side, target_side in ((right, left), (left, right)):
            if isinstance(const_side, ast.Constant) and isinstance(const_side.value, bool):
                val = bool(const_side.value)
                if isinstance(op, (ast.Is, ast.Eq)):
                    return target_side, val
                if isinstance(op, (ast.IsNot, ast.NotEq)):
                    return target_side, not val
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


def _collect_try_block_imports(
    node: Any,
    include_classes: bool = True,
) -> List[Union[ast.Import, ast.ImportFrom]]:
    """Helper to collect runtime import nodes from try/try* blocks."""
    res: List[Union[ast.Import, ast.ImportFrom]] = []
    res.extend(
        _collect_top_level_import_nodes(
            getattr(node, "body", []), include_classes=include_classes
        )
    )
    for handler in getattr(node, "handlers", []):
        res.extend(
            _collect_top_level_import_nodes(
                getattr(handler, "body", []), include_classes=include_classes
            )
        )
    res.extend(
        _collect_top_level_import_nodes(
            getattr(node, "orelse", []), include_classes=include_classes
        )
    )
    res.extend(
        _collect_top_level_import_nodes(
            getattr(node, "finalbody", []), include_classes=include_classes
        )
    )
    return res


def _collect_top_level_import_nodes(
    stmts: Sequence[ast.stmt],
    include_classes: bool = True,
) -> List[Union[ast.Import, ast.ImportFrom]]:
    """Recursively collects module-level runtime import nodes, skipping type-checking and functions."""
    result: List[Union[ast.Import, ast.ImportFrom]] = []
    for stmt in stmts:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            result.append(stmt)
        elif isinstance(stmt, ast.If):
            if _is_type_checking_guard(stmt.test):
                result.extend(
                    _collect_top_level_import_nodes(
                        stmt.orelse, include_classes=include_classes
                    )
                )
            elif _is_inverted_type_checking_guard(stmt.test):
                result.extend(
                    _collect_top_level_import_nodes(
                        stmt.body, include_classes=include_classes
                    )
                )
            else:
                result.extend(
                    _collect_top_level_import_nodes(
                        stmt.body, include_classes=include_classes
                    )
                )
                result.extend(
                    _collect_top_level_import_nodes(
                        stmt.orelse, include_classes=include_classes
                    )
                )
        elif isinstance(stmt, ast.Try):
            result.extend(
                _collect_try_block_imports(
                    stmt, include_classes=include_classes
                )
            )
        elif hasattr(ast, "TryStar") and isinstance(stmt, getattr(ast, "TryStar")):
            result.extend(
                _collect_try_block_imports(
                    stmt, include_classes=include_classes
                )
            )
        elif isinstance(stmt, ast.ClassDef):
            if include_classes:
                result.extend(
                    _collect_top_level_import_nodes(
                        stmt.body, include_classes=include_classes
                    )
                )
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            result.extend(
                _collect_top_level_import_nodes(
                    stmt.body, include_classes=include_classes
                )
            )
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            result.extend(
                _collect_top_level_import_nodes(
                    stmt.body, include_classes=include_classes
                )
            )
            result.extend(
                _collect_top_level_import_nodes(
                    stmt.orelse, include_classes=include_classes
                )
            )
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):
            for case in getattr(stmt, "cases", []):
                result.extend(
                    _collect_top_level_import_nodes(
                        case.body, include_classes=include_classes
                    )
                )
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
                        if alias.name != "*":
                            imports.add(f"{resolved}.{alias.name}")
            elif raw_mod:
                imports.add(raw_mod)
                for alias in node.names:
                    if alias.name != "*":
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
        ancestor_edges: Optional[Set[Tuple[str, str]]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes a module dependency graph scoped to a repository root."""
        self.repo_root = _resolve_root_path(repo_root)
        self.adjacency: Dict[str, Set[str]] = (
            {k: v.copy() for k, v in adjacency.items()} if adjacency else {}
        )
        self.mod_to_file: Dict[str, Path] = mod_to_file.copy() if mod_to_file else {}
        self.file_to_mod: Dict[str, str] = file_to_mod.copy() if file_to_mod else {}
        self.aliases: Dict[str, str] = aliases.copy() if aliases else {}
        self._sorted_adjacency: Dict[str, List[str]] = {}
        self._pending_descendants: Dict[str, Set[str]] = (
            {k: v.copy() for k, v in pending_descendants.items()}
            if pending_descendants
            else {}
        )
        self._ancestor_edges: Set[Tuple[str, str]] = (
            ancestor_edges.copy() if ancestor_edges else set()
        )

    def canonicalize_module_name(self, mod_name: str) -> str:
        """Returns the canonical module name resolving any registered module aliases."""
        return self.aliases.get(mod_name, mod_name)

    def add_alias(self, alias_name: str, canonical_name: str) -> None:
        """Registers an alias name mapping to an existing canonical module name."""
        if not alias_name or not canonical_name or alias_name == canonical_name:
            return
        canon = self.canonicalize_module_name(canonical_name)
        self.aliases[alias_name] = canon
        if canon in self.mod_to_file:
            self.mod_to_file[alias_name] = self.mod_to_file[canon]

    def add_module(self, mod_name: str, file_path: Path) -> None:
        """Registers a module and its backing file path in the graph."""
        if not mod_name:
            return
        canon = self.canonicalize_module_name(mod_name)
        if canon not in self.adjacency:
            self.adjacency[canon] = set()
        try:
            resolved = file_path.resolve()
        except (OSError, RuntimeError, ValueError):
            resolved = file_path
        self.mod_to_file[canon] = resolved
        self.mod_to_file[mod_name] = resolved
        self.file_to_mod[str(resolved).replace("\\", "/").lower()] = canon

        # In Python, importing a submodule first executes its ancestor package initializers
        parts = canon.split(".")
        for i in range(1, len(parts)):
            parent_pkg = ".".join(parts[:i])
            parent_canon = self.canonicalize_module_name(parent_pkg)
            if parent_canon in self.mod_to_file:
                self.add_dependency(canon, parent_canon)
                self._ancestor_edges.add((canon, parent_canon))
            else:
                self._pending_descendants.setdefault(parent_canon, set()).add(canon)

        if canon in self._pending_descendants:
            for desc_mod in self._pending_descendants.pop(canon):
                self.add_dependency(desc_mod, canon)
                self._ancestor_edges.add((desc_mod, canon))

    def add_dependency(self, from_mod: str, to_mod: str) -> None:
        """Adds a directed import edge from from_mod to to_mod."""
        from_canon = self.canonicalize_module_name(from_mod)
        to_canon = self.canonicalize_module_name(to_mod)
        if not from_canon or not to_canon or from_canon == to_canon:
            return
        if from_canon not in self.adjacency:
            self.adjacency[from_canon] = set()
        if to_canon not in self.adjacency[from_canon]:
            self.adjacency[from_canon].add(to_canon)
            self._sorted_adjacency.pop(from_canon, None)
        self._ancestor_edges.discard((from_canon, to_canon))

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
            ancestor_edges=self._ancestor_edges,
            aliases=self.aliases,
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
            return self.canonicalize_module_name(raw_import)
        parts = raw_import.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in known_modules:
                return self.canonicalize_module_name(prefix)
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
            self._ancestor_edges.discard((from_mod, target))

    def get_dependencies(self, mod_name: str) -> Set[str]:
        """Returns direct dependency module names for a given module."""
        canon = self.canonicalize_module_name(mod_name)
        return self.adjacency.get(canon, set()).copy()

    def has_transitive_path(
        self,
        from_mod: str,
        to_mod: str,
        ignored_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> bool:
        """Determines reachability from from_mod to to_mod via BFS search in O(V + E) time."""
        from_canon = self.canonicalize_module_name(from_mod)
        to_canon = self.canonicalize_module_name(to_mod)
        if not from_canon or not to_canon:
            return False
        if from_canon == to_canon:
            return True
        visited: Set[str] = {from_canon}
        queue: deque[str] = deque([from_canon])

        while queue:
            curr = queue.popleft()
            for neighbor in self._get_sorted_neighbors(curr):
                if ignored_edges and (curr, neighbor) in ignored_edges:
                    continue
                # If neighbor is the target module
                if neighbor == to_canon:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def find_cycle_path(
        self,
        from_mod: str,
        to_mod: str,
        ignored_edges: Optional[Set[Tuple[str, str]]] = None,
    ) -> Optional[List[str]]:
        """Finds a directed path from from_mod to to_mod if one exists, returning module names."""
        from_canon = self.canonicalize_module_name(from_mod)
        to_canon = self.canonicalize_module_name(to_mod)
        if not from_canon or not to_canon:
            return None
        if from_canon == to_canon:
            return [from_canon]

        parent: Dict[str, str] = {}
        visited: Set[str] = {from_canon}
        queue: deque[str] = deque([from_canon])
        found = False

        while queue:
            curr = queue.popleft()
            if curr == to_canon:
                found = True
                break
            for neighbor in self._get_sorted_neighbors(curr):
                if ignored_edges and (curr, neighbor) in ignored_edges:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        if not found and to_canon not in parent:
            return None

        # Reconstruct path
        path: List[str] = [to_canon]
        curr_node = to_canon
        while curr_node in parent:
            curr_node = parent[curr_node]
            path.append(curr_node)
            if curr_node == from_canon:
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
        from_canon = self.canonicalize_module_name(from_mod)
        to_canon = self.canonicalize_module_name(to_mod)
        if from_canon == to_canon:
            return [from_mod, to_mod]

        ignored: Optional[Set[Tuple[str, str]]] = None
        if to_canon.startswith(f"{from_canon}."):
            # from_canon is an ancestor package of to_canon and is already executing.
            # Implicit ancestor initialization edges back to from_canon (or ancestors of from_canon)
            # do not trigger circular imports unless explicitly imported in code.
            ignored = {
                (u, v)
                for (u, v) in self._ancestor_edges
                if v == from_canon or from_canon.startswith(f"{v}.")
            }

        existing_path = self.find_cycle_path(
            from_mod=to_canon, to_mod=from_canon, ignored_edges=ignored
        )
        if existing_path is not None:
            return [from_mod] + existing_path

        # Submodule import executes ancestor package initializers:
        # e.g., importing `pkg.worker` first executes `pkg/__init__.py`.
        # If `pkg` already depends on `from_canon`, caller -> pkg.worker -> pkg -> caller forms a cycle.
        target_names = [to_canon]
        if to_mod != to_canon:
            target_names.append(to_mod)
        for target in target_names:
            parts = target.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                prefix_canon = self.canonicalize_module_name(prefix)
                if prefix_canon != from_canon and not from_canon.startswith(f"{prefix_canon}."):
                    prefix_path = self.find_cycle_path(
                        from_mod=prefix_canon, to_mod=from_canon, ignored_edges=ignored
                    )
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

        from_canon = self.canonicalize_module_name(from_mod)
        tentative = self.copy()
        if file_path is not None:
            tentative.add_module(from_canon, file_path)
        elif from_canon not in tentative.adjacency:
            tentative.adjacency[from_canon] = set()

        raw_imports = _parse_source_imports(
            "\n".join(import_stmts), from_canon, is_package=is_package
        )
        known = set(tentative.mod_to_file.keys())

        for raw_imp in sorted(raw_imports):
            target = tentative.resolve_import_target(raw_imp, known_modules=known)
            if not target:
                continue
            target_canon = tentative.canonicalize_module_name(target)
            if target_canon == from_canon:
                return [from_mod, target]
            cycle = tentative.check_cycle_if_added(from_canon, target_canon)
            if cycle is not None:
                return cycle
            tentative.add_dependency(from_canon, target_canon)
        return None

    @classmethod
    def build_from_repository(
        cls,
        repo_root: Union[Path, str],
        file_paths: Optional[Sequence[Path]] = None,
        import_root: Optional[Union[Path, str]] = None,
    ) -> ModuleDependencyGraph:
        """Builds a populated dependency graph across the repository's Python source files.

        Args:
            repo_root: Project scan boundary where Python source files are discovered.
            file_paths: Optional explicit sequence of source file paths to include.
            import_root: Optional module import namespace root for deriving module dot-paths.
                Defaults to enclosing package root of repo_root if not specified.

        Returns:
            Populated ModuleDependencyGraph containing discovered modules and dependency edges.
        """
        root = _resolve_root_path(repo_root)
        effective_root = (
            _resolve_root_path(import_root)
            if import_root is not None
            else _find_enclosing_package_root(root)
        )
        graph = cls(effective_root)

        python_files: List[Path] = []
        seen_files: Set[Path] = set()

        if file_paths is not None:
            for p in file_paths:
                safe_file = _is_safe_repo_python_file(p, root)
                if safe_file is not None and safe_file not in seen_files:
                    seen_files.add(safe_file)
                    python_files.append(safe_file)
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dir_p = Path(dirpath)
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in EXCLUDED_GRAPH_DIRS
                    and not (dir_p / d).is_symlink()
                ]
                for filename in filenames:
                    file_p = dir_p / filename
                    safe_file = _is_safe_repo_python_file(file_p, root)
                    if safe_file is not None and safe_file not in seen_files:
                        seen_files.add(safe_file)
                        python_files.append(safe_file)

        # First pass: map all modules and register aliases
        file_to_names: List[Tuple[Path, str, str]] = []
        for p_file in python_files:
            if _is_within_root(p_file, effective_root):
                mod_name = derive_module_import_path(p_file, effective_root)
                alias_name = (
                    derive_module_import_path(p_file, root)
                    if effective_root != root and _is_within_root(p_file, root)
                    else ""
                )
            else:
                mod_name = derive_module_import_path(p_file, root)
                alias_name = ""

            if mod_name:
                graph.add_module(mod_name, p_file)
                if alias_name and alias_name != mod_name:
                    graph.add_alias(alias_name, mod_name)
                file_to_names.append((p_file, mod_name, alias_name))

        known_modules = set(graph.mod_to_file.keys())

        # Second pass: parse imports and wire edges
        for p_file, mod_name, alias_name in file_to_names:
            try:
                content = p_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            raw_imports = _parse_source_imports(
                content, mod_name, is_package=(p_file.name == "__init__.py")
            )
            if alias_name and alias_name != mod_name:
                raw_imports.update(
                    _parse_source_imports(
                        content, alias_name, is_package=(p_file.name == "__init__.py")
                    )
                )
            for imp in raw_imports:
                graph.add_import_dependency(mod_name, imp, known_modules=known_modules)
        return graph


def build_module_graph(
    repo_root: Union[Path, str],
    file_paths: Optional[Sequence[Path]] = None,
    import_root: Optional[Union[Path, str]] = None,
) -> ModuleDependencyGraph:
    """Convenience helper to construct a ModuleDependencyGraph for a repository.

    Args:
        repo_root: Project scan boundary where Python source files are discovered.
        file_paths: Optional explicit sequence of source file paths to include.
        import_root: Optional module import namespace root for deriving module dot-paths.

    Returns:
        Populated ModuleDependencyGraph containing discovered modules and dependency edges.
    """
    return ModuleDependencyGraph.build_from_repository(
        repo_root, file_paths=file_paths, import_root=import_root
    )
