"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import ast
import builtins
import difflib
import os
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydoppelgangerhunt.reporters import extract_unit_source_code

BUILTIN_NAMES: Set[str] = set(dir(builtins))


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, and returns for scope analysis."""

    def __init__(self) -> None:
        self.loads: List[str] = []
        self.stores: List[str] = []
        self.params: List[str] = []
        self.returns: List[str] = []

    def _process_args(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        for a in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
        ):
            if a.arg not in BUILTIN_NAMES and a.arg not in self.params:
                self.params.append(a.arg)
        if node.args.vararg and node.args.vararg.arg not in BUILTIN_NAMES:
            self.params.append(node.args.vararg.arg)
        if node.args.kwarg and node.args.kwarg.arg not in BUILTIN_NAMES:
            self.params.append(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_args(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_args(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in BUILTIN_NAMES and node.id not in self.loads:
                self.loads.append(node.id)
        elif isinstance(node.ctx, ast.Store):
            if node.id not in BUILTIN_NAMES and node.id not in self.stores:
                self.stores.append(node.id)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            if isinstance(node.value, ast.Name) and node.value.id not in self.returns:
                self.returns.append(node.value.id)
        self.generic_visit(node)


def _inspect_unit_scope(unit: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """Extracts (inputs, outputs, stores) for a single AST unit by parsing its source."""
    raw_lines = extract_unit_source_code(unit)
    dedented = textwrap.dedent("".join(raw_lines))
    if not dedented.strip():
        return [], [], []

    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        try:
            tree = ast.parse(f"def _wrapper():\n{textwrap.indent(dedented, '    ')}")
        except SyntaxError:
            tree = None

    if tree is None:
        return [], [], []

    visitor = _ScopeVisitor()
    visitor.visit(tree)

    if visitor.params:
        inputs = list(visitor.params)
    else:
        inputs = [name for name in visitor.loads if name not in visitor.stores]

    outputs = list(visitor.returns)
    return inputs, outputs, visitor.stores


def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:
    """Analyzes AST variable scoping to determine input parameters, outputs, and local variables."""
    inputs1, outputs1, stores1 = _inspect_unit_scope(u1)
    if u2 is not None:
        inputs2, outputs2, _ = _inspect_unit_scope(u2)
        common_inputs = [var for var in inputs1 if var in inputs2]
        inputs = common_inputs if common_inputs else inputs1
        outputs = list(dict.fromkeys(outputs1 + outputs2))
    else:
        inputs = inputs1
        outputs = outputs1

    locals_ = [var for var in stores1 if var not in inputs]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "locals": locals_,
    }


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

    # Variable scope analysis for concrete parameter signatures
    scope = analyze_unit_variable_scope(u1, u2)
    if scope["inputs"]:
        params_str = ", ".join(f"{arg}: Any" for arg in scope["inputs"])
    else:
        params_str = "*args: Any, **kwargs: Any"

    # Indent body
    indented_body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in common_lines)
    return f"def {helper_name}({params_str}) -> Any:\n    \"\"\"Auto-extracted shared helper for duplicate logic.\"\"\"\n{indented_body}\n"


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
