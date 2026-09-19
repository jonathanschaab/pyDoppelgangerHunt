"""Enclosing class and method boundary inspection, receiver kind resolution, and binding."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydoppelgangerhunt.config import normalize_path_string, paths_match_boundary
from pydoppelgangerhunt.parser import is_decorator_named
from pydoppelgangerhunt.fixer.scope import (
    dispatch_analyze_unit_variable_scope as analyze_unit_variable_scope,
)

def _find_innermost_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
) -> Optional[Tuple[ast.AST, int, int]]:
    """Locates the innermost AST node of matching types enclosing the given unit."""
    if not source_text.strip():
        return None

    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None

    u_start = int(unit.get("start") or 0)
    u_end = int(unit.get("end") or u_start)
    if u_start <= 0:
        return None

    candidates: List[Tuple[int, ast.AST, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, node_types):
            n_start = getattr(node, "lineno", 0)
            n_end = getattr(node, "end_lineno", n_start)
            decorators = getattr(node, "decorator_list", [])
            dec_start = (
                min(getattr(d, "lineno", n_start) for d in decorators)
                if decorators
                else n_start
            )
            earliest_start = min(dec_start, n_start)
            if earliest_start <= u_start <= u_end <= n_end:
                candidates.append((n_end - earliest_start, node, earliest_start, n_end))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -item[2]))
    _, matched_node, n_start, n_end = candidates[0]
    return matched_node, n_start, n_end


def _inspect_enclosing_node(
    source_text: str,
    unit: Dict[str, Any],
    node_types: Tuple[type, ...],
) -> Optional[Dict[str, Any]]:
    """Locates and extracts metadata for the innermost enclosing AST node of matching types."""
    res = _find_innermost_enclosing_node(source_text, unit, node_types)
    if not res:
        return None
    matched_node, start, end = res
    decorators = list(getattr(matched_node, "decorator_list", []))
    return {
        "node": matched_node,
        "name": getattr(matched_node, "name", ""),
        "start": start,
        "def_start": getattr(matched_node, "lineno", start),
        "end": end,
        "decorators": decorators,
    }


def _extract_child_indentation(
    lines: Sequence[str],
    line_numbers: Iterable[int],
    parent_indent: str,
) -> Optional[str]:
    """Finds the indentation of the first line from line_numbers that is more indented than parent_indent."""
    for line_num in line_numbers:
        if 1 <= line_num <= len(lines):
            line_str: str = str(lines[line_num - 1])
            ind: str = line_str[: len(line_str) - len(line_str.lstrip())]
            if ind.startswith(parent_indent) and len(ind) > len(parent_indent):
                return str(ind)
    return None


def find_enclosing_class(
    source_text: str,
    unit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locates the innermost enclosing ClassDef for an AST code unit.

    Args:
        source_text: The complete original Python source code.
        unit: AST unit dictionary with 1-indexed 'start' and 'end' line bounds.

    Returns:
        A dictionary with class metadata ('name', 'start', 'end', 'indent', 'method_indent')
        if the unit is enclosed within a ClassDef; None otherwise.
    """
    meta = _inspect_enclosing_node(source_text, unit, (ast.ClassDef,))
    if not meta:
        return None

    lines = source_text.splitlines(keepends=True)
    c_start = meta["start"]
    cls_line = lines[c_start - 1] if 1 <= c_start <= len(lines) else ""
    indent = cls_line[: len(cls_line) - len(cls_line.lstrip())]
    meta["indent"] = indent

    node = meta.get("node")
    suite_indent = None
    def_start = meta.get("def_start", c_start)
    direct_methods: Set[Tuple[int, int]] = set()
    if isinstance(node, ast.ClassDef) and node.body:
        cand_lines: List[int] = []
        for stmt in node.body:
            stmt_lineno = int(getattr(stmt, "lineno", 0))
            earliest_line = stmt_lineno
            decorators = getattr(stmt, "decorator_list", [])
            if decorators:
                dec_lines = [int(getattr(d, "lineno", stmt_lineno)) for d in decorators]
                if dec_lines:
                    dec_start = min(dec_lines)
                    earliest_line = min(dec_start, earliest_line) if earliest_line > 0 else dec_start
            if earliest_line > def_start:
                cand_lines.append(earliest_line)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_l = int(getattr(stmt, "end_lineno", stmt_lineno))
                direct_methods.add((earliest_line, end_l))
                direct_methods.add((stmt_lineno, end_l))
        suite_indent = _extract_child_indentation(lines, cand_lines, indent)

    meta["methods"] = direct_methods
    meta["method_indent"] = suite_indent if suite_indent is not None else (indent + "    ")
    meta.pop("node", None)
    meta.pop("decorators", None)
    meta.pop("def_start", None)
    return meta


def find_enclosing_function(
    source_text: str,
    unit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locates the innermost enclosing FunctionDef/AsyncFunctionDef for an AST code unit."""
    meta = _inspect_enclosing_node(
        source_text, unit, (ast.FunctionDef, ast.AsyncFunctionDef)
    )
    if not meta:
        return None

    decs = meta.pop("decorators", [])
    node = meta.pop("node", None)
    meta["is_static"] = any(is_decorator_named(d, "staticmethod") for d in decs)
    meta["is_class_method"] = any(is_decorator_named(d, "classmethod") for d in decs)
    receiver_param = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        all_pos = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
        if all_pos:
            receiver_param = all_pos[0].arg
    meta["receiver_param"] = receiver_param
    return meta


def _normalize_file_path(
    f_str: str,
    repo_root: Optional[str] = None,
) -> str:
    """Normalizes a file path string to a canonical resolved POSIX path."""
    norm = normalize_path_string(f_str)
    if not norm:
        return ""
    root = Path(repo_root or os.getcwd())
    try:
        p = Path(norm)
        p_full = p if p.is_file() or p.is_absolute() else (root / p)
        return str(p_full.resolve()).replace("\\", "/")
    except OSError:
        return norm


def _is_same_file_path(
    f1_str: str,
    f2_str: str,
    repo_root: Optional[str] = None,
) -> bool:
    """Checks whether two file path strings refer to the identical file on disk."""
    if not f1_str or not f2_str:
        return False
    norm1 = normalize_path_string(f1_str)
    norm2 = normalize_path_string(f2_str)
    root = Path(repo_root or os.getcwd())
    try:
        p1 = Path(norm1)
        p2 = Path(norm2)
        p1_full = p1 if p1.is_file() or p1.is_absolute() else (root / p1)
        p2_full = p2 if p2.is_file() or p2.is_absolute() else (root / p2)
        if p1_full.is_symlink() or p2_full.is_symlink():
            return False
        if paths_match_boundary(norm1, norm2):
            return True
        if (
            p1_full.is_file()
            and p2_full.is_file()
            and p1_full.resolve() == p2_full.resolve()
        ):
            return True
    except OSError:
        return False
    return False


def _is_method_of_class(
    fn_meta: Optional[Dict[str, Any]],
    cls_meta: Optional[Dict[str, Any]],
) -> bool:
    """Returns True if the function is a direct method defined inside the enclosing class."""
    if not fn_meta or not cls_meta:
        return False
    methods = cls_meta.get("methods")
    if methods is not None:
        fn_span = (fn_meta["start"], fn_meta["end"])
        fn_def_span = (fn_meta.get("def_start", fn_meta["start"]), fn_meta["end"])
        return fn_span in methods or fn_def_span in methods
    return bool(
        cls_meta["start"] < fn_meta["start"]
        and fn_meta["end"] <= cls_meta["end"]
    )


def _has_receiver_reference(
    unit: Dict[str, Any],
    scope: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
) -> bool:
    """Checks if a unit actually references an instance or class receiver in its executable body."""
    target_scope = (
        scope if scope is not None else analyze_unit_variable_scope(unit, repo_root=repo_root)
    )
    return bool(
        target_scope.get("has_receiver_access")
        or target_scope.get("instance_attrs")
        or target_scope.get("class_attrs")
    )


def _prune_unshared_receivers(
    inputs: List[str],
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    scope1: Dict[str, Any],
    scope2: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> List[str]:
    """Removes receiver parameters from inputs when only one unit receives them and neither references them."""
    res = list(inputs)
    rec_candidates: Set[str] = {"self", "cls"}
    for u in (u1, u2):
        rec_p = u.get("receiver_param")
        if rec_p:
            rec_candidates.add(rec_p)
    for rec in sorted(rec_candidates):
        if (rec in scope1.get("inputs", [])) != (rec in scope2.get("inputs", [])):
            if not (
                _has_receiver_reference(u1, scope1, repo_root=repo_root)
                or _has_receiver_reference(u2, scope2, repo_root=repo_root)
            ):
                res = [v for v in res if v != rec]
    return res


def _get_enclosing_receiver_kind(fn_info: Optional[Dict[str, Any]]) -> str:
    """Returns the receiver kind for an enclosing function: static, class, instance, or none."""
    if not fn_info:
        return "none"
    if fn_info.get("is_static"):
        return "static"
    if fn_info.get("is_class_method"):
        return "class"
    return "instance"


def _populate_unit_receiver_metadata(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
) -> None:
    """Populates receiver_kind, enclosing_class, and is_static on a unit from enclosing AST nodes if absent."""
    if (
        "receiver_kind" in unit
        and "enclosing_class" in unit
        and "enclosing_class_start" in unit
        and "receiver_param" in unit
    ):
        return
    f_raw = normalize_path_string(str(unit.get("file") or ""), strip_anchor=True)
    if not f_raw:
        return
    p = Path(f_raw)
    f_path = p if p.is_file() or p.is_absolute() else (Path(repo_root or os.getcwd()) / p)
    if not f_path.is_file():
        return
    try:
        source = f_path.read_text(encoding="utf-8")
        fn_meta = find_enclosing_function(source, unit)
        cls_meta = find_enclosing_class(source, unit)
        if cls_meta:
            if "enclosing_class" not in unit:
                unit["enclosing_class"] = cls_meta["name"]
            if "enclosing_class_start" not in unit:
                unit["enclosing_class_start"] = cls_meta["start"]
        is_method = _is_method_of_class(fn_meta, cls_meta)
        if "receiver_kind" not in unit:
            if is_method:
                rec_k = _get_enclosing_receiver_kind(fn_meta)
                unit["receiver_kind"] = rec_k
                if rec_k == "static":
                    unit["is_static"] = True
            else:
                unit["receiver_kind"] = None
        if "receiver_param" not in unit:
            if is_method and unit.get("receiver_kind") in ("instance", "class"):
                unit["receiver_param"] = fn_meta.get("receiver_param") if fn_meta else None
            else:
                unit["receiver_param"] = None
    except (OSError, UnicodeDecodeError):
        pass


def _base_unit_name(u: Dict[str, Any]) -> str:
    """Extracts base function or method name from unit dictionary for helper synthesis."""
    name = str(u.get("name") or "")
    if u.get("kind") == "closure" and ":" in name:
        base = name.rsplit(":", maxsplit=1)[-1].strip("_")
    else:
        base = name.split(":", maxsplit=1)[0].strip("_")
    if "." in base:
        base = base.rsplit(".", maxsplit=1)[-1].strip("_")
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_")
    return base or "helper"



def _resolve_effective_binding(
    method_binding: str,
    is_same_class: bool,
    is_static: bool = False,
    receiver_kinds_differ: bool = False,
    is_in_method: bool = True,
    receiver_kind: Optional[str] = None,
) -> str:
    """Resolve the effective helper binding mode ('method' vs 'module')."""
    if method_binding == "module":
        return "module"

    # Method binding is only viable if both units share the same class,
    # have compatible receiver kinds, and reside within methods (not class body).
    if not is_same_class or receiver_kinds_differ or not is_in_method:
        return "module"

    if method_binding == "method":
        return "method"

    # auto mode
    if is_static or receiver_kind in ("none", None):
        return "module"
    return "method"
