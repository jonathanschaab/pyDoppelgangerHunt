"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import ast
import builtins
import difflib
import logging
import os
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydoppelgangerhunt.reporters import extract_unit_source_code

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, returns, nonlocals, globals, and attributes."""

    def __init__(self) -> None:
        self.loads: List[str] = []
        self.stores: List[str] = []
        self.params: List[str] = []
        self.param_details: List[Dict[str, Any]] = []
        self.returns: List[str] = []
        self.return_type: Optional[str] = None
        self.nonlocals: List[str] = []
        self.globals: List[str] = []
        self.attrs_read: List[str] = []
        self.attrs_written: List[str] = []
        self._scope_stack: List[Set[str]] = []

    def _record_arg(
        self,
        arg_node: Optional[ast.arg],
        default_node: Optional[ast.expr],
        kind: str,
    ) -> None:
        if arg_node is None or arg_node.arg in BUILTIN_NAMES:
            return
        arg_name = arg_node.arg
        if arg_name not in self.params:
            self.params.append(arg_name)

        type_val = None
        if arg_node.annotation is not None:
            try:
                type_val = ast.unparse(arg_node.annotation)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse argument annotation %r: %s", arg_node.annotation, exc)
                type_val = None

        default_val = None
        if default_node is not None:
            try:
                default_val = ast.unparse(default_node)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse default value %r: %s", default_node, exc)
                default_val = None

        prefix = "*" if kind == "vararg" else ("**" if kind == "kwarg" else "")
        self.param_details.append({
            "name": f"{prefix}{arg_name}",
            "type": type_val,
            "default": default_val,
            "kind": kind,
        })

    def _process_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if node.returns is not None:
            try:
                self.return_type = ast.unparse(node.returns)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse return annotation %r: %s", node.returns, exc)
                self.return_type = None

        pos_and_plain = node.args.posonlyargs + node.args.args
        num_pos = len(pos_and_plain)
        num_defaults = len(node.args.defaults)
        first_default_idx = num_pos - num_defaults

        for i, a in enumerate(pos_and_plain):
            def_node = node.args.defaults[i - first_default_idx] if i >= first_default_idx else None
            self._record_arg(a, def_node, "pos")

        self._record_arg(node.args.vararg, None, "vararg")

        for k, kw_def in zip(node.args.kwonlyargs, node.args.kw_defaults):
            self._record_arg(k, kw_def, "kwonly")

        self._record_arg(node.args.kwarg, None, "kwarg")

    def _process_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        is_top = len(self._scope_stack) == 0
        if is_top:
            self._process_args(node)
            self._scope_stack.append(set(self.params))
        else:
            if node.name not in self.stores:
                self.stores.append(node.name)
            inner_args = {
                a.arg for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
            }
            if node.args.vararg:
                inner_args.add(node.args.vararg.arg)
            if node.args.kwarg:
                inner_args.add(node.args.kwarg.arg)
            self._scope_stack.append(inner_args)

        for stmt in node.body:
            self.visit(stmt)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_func(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name not in self.stores:
            self.stores.append(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            is_inner = any(node.id in s for s in self._scope_stack[1:])
            if not is_inner and node.id not in BUILTIN_NAMES and node.id not in self.loads:
                self.loads.append(node.id)
        elif isinstance(node.ctx, ast.Store):
            if len(self._scope_stack) > 1:
                self._scope_stack[-1].add(node.id)
            elif node.id not in BUILTIN_NAMES and node.id not in self.stores:
                self.stores.append(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            attr_name = f"{node.value.id}.{node.attr}"
            if isinstance(node.ctx, ast.Load):
                if attr_name not in self.attrs_read:
                    self.attrs_read.append(attr_name)
            elif isinstance(node.ctx, ast.Store):
                if attr_name not in self.attrs_written:
                    self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # AugAssign both loads and stores the target
        if isinstance(node.target, ast.Name):
            is_inner = any(node.target.id in s for s in self._scope_stack[1:])
            if not is_inner and node.target.id not in BUILTIN_NAMES and node.target.id not in self.loads:
                self.loads.append(node.target.id)
            if len(self._scope_stack) > 1:
                self._scope_stack[-1].add(node.target.id)
            elif node.target.id not in BUILTIN_NAMES and node.target.id not in self.stores:
                self.stores.append(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            if isinstance(node.target.value, ast.Name) and node.target.value.id in ("self", "cls"):
                attr_name = f"{node.target.value.id}.{node.target.attr}"
                if attr_name not in self.attrs_read:
                    self.attrs_read.append(attr_name)
                if attr_name not in self.attrs_written:
                    self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            if name not in self.nonlocals:
                self.nonlocals.append(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name not in self.globals:
                self.globals.append(name)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            if isinstance(node.value, ast.Name) and node.value.id not in self.returns:
                self.returns.append(node.value.id)
        self.generic_visit(node)


def _inspect_unit_scope(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts scope analysis metadata for a single AST unit by parsing its source."""
    raw_lines = extract_unit_source_code(unit)
    dedented = textwrap.dedent("".join(raw_lines))
    empty_res: Dict[str, Any] = {
        "inputs": [],
        "outputs": [],
        "stores": [],
        "free_vars": [],
        "nonlocals": [],
        "globals": [],
        "attrs_read": [],
        "attrs_written": [],
        "param_details": [],
        "return_type": None,
    }
    if not dedented.strip():
        return empty_res

    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        try:
            tree = ast.parse(f"def _wrapper():\n{textwrap.indent(dedented, '    ')}")
        except SyntaxError:
            tree = None

    if tree is None:
        return empty_res

    visitor = _ScopeVisitor()
    visitor.visit(tree)

    free_vars = [
        name for name in visitor.loads
        if name not in visitor.params
        and name not in visitor.stores
        and name not in visitor.globals
        and name not in BUILTIN_NAMES
    ]

    inputs: List[str] = list(visitor.params)
    for fv in free_vars:
        if fv not in inputs:
            inputs.append(fv)
    for nl in visitor.nonlocals:
        if nl not in inputs:
            inputs.append(nl)

    # Nonlocal variables are marked as outputs only when stored (mutated) within this unit.
    # Read-only nonlocals act as outer closure inputs; if they are modified by an outer scope
    # rather than within this function, that mutation is external to this unit's write scope.
    outputs = list(visitor.returns)
    for nl in visitor.nonlocals:
        if nl in visitor.stores and nl not in outputs:
            outputs.append(nl)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "stores": visitor.stores,
        "free_vars": free_vars,
        "nonlocals": visitor.nonlocals,
        "globals": visitor.globals,
        "attrs_read": visitor.attrs_read,
        "attrs_written": visitor.attrs_written,
        "param_details": visitor.param_details,
        "return_type": visitor.return_type,
    }


def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping to determine inputs, outputs, closures, and attributes."""
    info1 = _inspect_unit_scope(u1)
    if u2 is not None:
        info2 = _inspect_unit_scope(u2)
        common_inputs = [var for var in info1["inputs"] if var in info2["inputs"]]
        inputs = common_inputs if common_inputs else info1["inputs"]
        outputs = list(dict.fromkeys(info1["outputs"] + info2["outputs"]))
        free_vars = list(dict.fromkeys(info1["free_vars"] + info2["free_vars"]))
        nonlocals = list(dict.fromkeys(info1["nonlocals"] + info2["nonlocals"]))
        globals_ = list(dict.fromkeys(info1["globals"] + info2["globals"]))
        attrs_read = list(dict.fromkeys(info1["attrs_read"] + info2["attrs_read"]))
        attrs_written = list(dict.fromkeys(info1["attrs_written"] + info2["attrs_written"]))
        if info1["return_type"] == info2["return_type"]:
            return_type = info1["return_type"]
        elif info1["return_type"] is None:
            return_type = info2["return_type"]
        elif info2["return_type"] is None:
            return_type = info1["return_type"]
        else:
            return_type = None
    else:
        inputs = info1["inputs"]
        outputs = info1["outputs"]
        free_vars = info1["free_vars"]
        nonlocals = info1["nonlocals"]
        globals_ = info1["globals"]
        attrs_read = info1["attrs_read"]
        attrs_written = info1["attrs_written"]
        return_type = info1["return_type"]

    locals_ = [var for var in info1["stores"] if var not in inputs]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "locals": locals_,
        "free_vars": free_vars,
        "nonlocals": nonlocals,
        "globals": globals_,
        "attrs_read": attrs_read,
        "attrs_written": attrs_written,
        "param_details": info1["param_details"],
        "return_type": return_type,
    }


def _merge_types(t1: Optional[str], t2: Optional[str], strategy: str) -> str:
    """Merges two type annotations according to the specified merge strategy."""
    if t1 and t2:
        if t1 == t2:
            return t1
        if strategy == "union":
            return f"Union[{t1}, {t2}]"
        return "Any"
    return t1 or t2 or "Any"


def synthesize_shared_helper_code(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units.

    Args:
        u1: First clone unit dictionary.
        u2: Second clone unit dictionary.
        type_merge_strategy: Strategy for merging conflicting types.
            Options: "fallback_any" (default, falls back to Any)
            or "union" (suggests typing.Union[T1, T2]).
    """
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

    # Variable scope analysis for concrete parameter signatures
    scope1 = analyze_unit_variable_scope(u1)
    scope2 = analyze_unit_variable_scope(u2)
    scope = analyze_unit_variable_scope(u1, u2)

    meta1 = {p["name"].lstrip("*"): p for p in scope1.get("param_details", [])}
    meta2 = {p["name"].lstrip("*"): p for p in scope2.get("param_details", [])}

    inputs = list(scope["inputs"])

    params: List[str] = []
    seen_kwonly = False

    # Extract parameter descriptors
    descriptors: List[Dict[str, Any]] = []
    for var in inputs:
        m1 = meta1.get(var, {})
        m2 = meta2.get(var, {})

        t1 = m1.get("type")
        t2 = m2.get("type")
        resolved_type = _merge_types(t1, t2, type_merge_strategy)

        d1 = m1.get("default")
        d2 = m2.get("default")
        resolved_default = d1 if (d1 and d2 and d1 == d2) else None

        kind = m1.get("kind") or m2.get("kind") or "pos"
        descriptors.append({
            "var": var,
            "type": resolved_type,
            "default": resolved_default,
            "kind": kind,
        })

    # Validate and ensure canonical argument kind ordering (self/cls -> pos -> vararg -> kwonly -> kwarg)
    def _kind_rank(desc: Dict[str, Any]) -> Tuple[int, int]:
        var = desc["var"]
        kind = desc["kind"]
        if var in ("self", "cls"):
            return (0, 0)
        rank_map = {"pos": 1, "vararg": 2, "kwonly": 3, "kwarg": 4}
        return (rank_map.get(kind, 1), 1)

    descriptors.sort(key=_kind_rank)

    # Validate positional argument default order (non-default cannot follow default)
    has_pos_default = False
    invalid_pos_defaults = False
    for desc in descriptors:
        if desc["kind"] == "pos":
            if desc["default"] is not None:
                has_pos_default = True
            elif has_pos_default:
                invalid_pos_defaults = True
                break
    if invalid_pos_defaults:
        for desc in descriptors:
            if desc["kind"] == "pos":
                desc["default"] = None

    # Render formatted parameters
    for desc in descriptors:
        var = desc["var"]
        resolved_type = desc["type"]
        resolved_default = desc["default"]
        kind = desc["kind"]
        prefix = "*" if kind == "vararg" else ("**" if kind == "kwarg" else "")
        var_name = f"{prefix}{var}"

        if kind == "kwonly" and not seen_kwonly and not any("*" in p for p in params):
            params.append("*")
            seen_kwonly = True

        if resolved_default is not None:
            params.append(f"{var_name}: {resolved_type} = {resolved_default}")
        else:
            params.append(f"{var_name}: {resolved_type}")

    r1 = scope1.get("return_type")
    r2 = scope2.get("return_type")
    return_type = _merge_types(r1, r2, type_merge_strategy)

    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    # Indent body
    indented_body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in common_lines)
    return (
        f"def {helper_name}({params_str}) -> {return_type}:\n"
        f"    \"\"\"Auto-extracted shared helper for duplicate logic.\"\"\"\n"
        f"{indented_body}\n"
    )


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
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
        helper_code = synthesize_shared_helper_code(
            u1, u2, type_merge_strategy=type_merge_strategy
        )

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
