"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import ast
import builtins
import difflib
import logging
import os
from pathlib import Path
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.reporters import extract_unit_source_code

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, returns, nonlocals, globals, and attributes."""

    def __init__(self, loop_offset: int = 0) -> None:
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
        self.loop_offset: int = loop_offset
        self.loop_depth: int = 0
        self.has_return: bool = False
        self.has_yield: bool = False
        self.naked_breaks: int = 0
        self.naked_continues: int = 0
        self.local_imports: List[str] = []
        self.imported_names: Dict[str, str] = {}
        self.yield_expr_names: List[Tuple[str, str]] = []
        self.is_async: bool = False

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
        self.is_async = True
        self._process_func(node)

    def visit_Await(self, node: ast.Await) -> None:
        self.is_async = True
        self.generic_visit(node)

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

    def _visit_loop(self, node: Union[ast.For, ast.AsyncFor, ast.While]) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.is_async = True
        self._visit_loop(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.is_async = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def _record_loop_control(self, is_break: bool) -> None:
        if self.loop_depth <= self.loop_offset:
            if is_break:
                self.naked_breaks += 1
            else:
                self.naked_continues += 1

    def visit_Break(self, node: ast.Break) -> None:
        self._record_loop_control(is_break=True)
        self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        self._record_loop_control(is_break=False)
        self.generic_visit(node)

    def _record_yield_expr(self, kind: str, node: Union[ast.Yield, ast.YieldFrom]) -> None:
        self.has_yield = True
        if node.value is not None and isinstance(node.value, ast.Name):
            self.yield_expr_names.append((kind, node.value.id))

    def visit_Yield(self, node: ast.Yield) -> None:
        self._record_yield_expr("yield", node)
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._record_yield_expr("yield_from", node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.has_return = True
        if node.value is not None:
            if isinstance(node.value, ast.Name):
                if node.value.id not in self.returns:
                    self.returns.append(node.value.id)
            elif isinstance(node.value, ast.Tuple):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Name) and elt.id not in self.returns:
                        self.returns.append(elt.id)
        self.generic_visit(node)

    def _record_import_node(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        try:
            stmt = ast.unparse(node)
            if stmt not in self.local_imports:
                self.local_imports.append(stmt)
            for alias in node.names:
                name = alias.asname or alias.name
                self.imported_names[name] = stmt
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("Failed to unparse import node %r: %s", node, exc)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_node(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_node(node)
        self.generic_visit(node)


def _analyze_block_assignment(statements: Sequence[ast.stmt]) -> Tuple[Set[str], Set[str]]:
    """Analyzes a sequence of statements to determine unconditionally and conditionally assigned variables.

    Returns:
        A tuple of (definitely_assigned, conditionally_assigned) variable name sets.
    """
    definite: Set[str] = set()
    conditional: Set[str] = set()

    for stmt in statements:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        definite.add(node.id)
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                definite.add(stmt.target.id)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                definite.add(stmt.target.id)
        elif isinstance(stmt, ast.If):
            b_def, b_cond = _analyze_block_assignment(stmt.body)
            o_def, o_cond = _analyze_block_assignment(stmt.orelse) if stmt.orelse else (set(), set())
            if stmt.orelse:
                both = b_def & o_def
                definite.update(both)
                conditional.update((b_def | b_cond | o_def | o_cond) - definite)
            else:
                conditional.update((b_def | b_cond) - definite)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
            sub_stmts: List[ast.stmt] = list(stmt.body)
            if hasattr(stmt, "orelse") and stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            if hasattr(stmt, "handlers") and stmt.handlers:
                for h in stmt.handlers:
                    sub_stmts.extend(h.body)
            if hasattr(stmt, "finalbody") and stmt.finalbody:
                sub_stmts.extend(stmt.finalbody)
            sub_def, sub_cond = _analyze_block_assignment(sub_stmts)
            conditional.update((sub_def | sub_cond) - definite)
        elif hasattr(ast, "Match") and isinstance(stmt, getattr(ast, "Match")):
            case_defs: List[Set[str]] = []
            case_conds: Set[str] = set()
            for case in getattr(stmt, "cases", []):
                c_def, c_cond = _analyze_block_assignment(case.body)
                case_defs.append(c_def)
                case_conds.update(c_def | c_cond)
            if case_defs:
                common = set.intersection(*case_defs)
                definite.update(common)
                conditional.update(case_conds - definite)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definite.add(stmt.name)

    return definite, conditional


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
        "control_flow_hazards": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_return": False,
        "local_imports": [],
        "yield_expr_names": [],
        "is_async": False,
        "conditional_outputs": [],
        "definite_stores": [],
    }
    if not dedented.strip():
        return empty_res

    tree: Optional[ast.AST] = None
    loop_offset = 0
    candidate_stmts: List[ast.stmt] = []

    parse_candidates = [
        (dedented, 0),
        (f"async def _wrapper():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"def _wrapper():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"async def _wrapper():\n    for _ in (0,):\n{textwrap.indent(dedented, '        ')}", 1),
        (f"def _wrapper():\n    for _ in (0,):\n{textwrap.indent(dedented, '        ')}", 1),
    ]

    for cand_text, offset in parse_candidates:
        try:
            tree = ast.parse(cand_text)
            loop_offset = offset
            if cand_text == dedented:
                if isinstance(tree, ast.Module):
                    if len(tree.body) == 1 and isinstance(
                        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = tree.body[0].body
                    else:
                        candidate_stmts = tree.body
            elif offset == 1:
                wrapper_fn = tree.body[0]  # type: ignore[attr-defined]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for_loop = wrapper_fn.body[0]
                    if isinstance(for_loop, (ast.For, ast.AsyncFor, ast.While)):
                        candidate_stmts = for_loop.body
            else:
                wrapper_fn = tree.body[0]  # type: ignore[attr-defined]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    candidate_stmts = wrapper_fn.body
            break
        except SyntaxError:
            continue

    if tree is None:
        return empty_res

    visitor = _ScopeVisitor(loop_offset=loop_offset)
    visitor.visit(tree)

    unit_name = unit.get("name", "")
    unit_kind = unit.get("kind", "")
    is_subroutine = unit_kind in ("compound_block", "sliding_window", "clause_branch") or (
        ":" in unit_name and not unit_name.startswith("closure:")
    )

    hazards: List[str] = []
    if visitor.naked_breaks > 0:
        hazards.append("naked_break")
    if visitor.naked_continues > 0:
        hazards.append("naked_continue")
    if visitor.has_yield:
        hazards.append("generator_yield")
    if is_subroutine and visitor.has_return:
        hazards.append("embedded_return")

    is_control_flow_safe = not any(
        h in ("naked_break", "naked_continue", "embedded_return") for h in hazards
    )

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

    # Definite and conditional assignment analysis across candidate statements
    def_assigned, _ = _analyze_block_assignment(candidate_stmts)

    # Unit outputs: explicit returns if present; for subroutines without explicit
    # returns, all local variable stores act as unit outputs to preserve caller mutations.
    if visitor.returns:
        outputs = list(visitor.returns)
    elif is_subroutine:
        outputs = [
            v for v in visitor.stores
            if v not in visitor.globals
            and v not in BUILTIN_NAMES
        ]
    else:
        outputs = []

    for nl in visitor.nonlocals:
        if nl in visitor.stores and nl not in outputs:
            outputs.append(nl)

    # Detect conditional variable escapes (outputs assigned conditionally without prior
    # unconditional assignment in this block, and not provided as input arguments).
    conditional_outputs = [
        v for v in outputs
        if v not in inputs and v not in def_assigned
    ]

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
        "control_flow_hazards": hazards,
        "is_control_flow_safe": is_control_flow_safe,
        "has_yield": visitor.has_yield,
        "has_return": visitor.has_return,
        "local_imports": visitor.local_imports,
        "yield_expr_names": visitor.yield_expr_names,
        "is_async": visitor.is_async,
        "conditional_outputs": conditional_outputs,
        "definite_stores": sorted(def_assigned),
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
        common_outputs = [var for var in info1["outputs"] if var in info2["outputs"]]
        outputs = common_outputs if common_outputs else list(dict.fromkeys(info1["outputs"] + info2["outputs"]))
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
        hazards = list(dict.fromkeys(info1["control_flow_hazards"] + info2["control_flow_hazards"]))
        is_safe = info1["is_control_flow_safe"] and info2["is_control_flow_safe"]
        has_yield = info1["has_yield"] or info2["has_yield"]
        has_return = info1["has_return"] or info2["has_return"]
        local_imports = list(dict.fromkeys(info1["local_imports"] + info2["local_imports"]))
        yield_expr_names = info1["yield_expr_names"] + info2["yield_expr_names"]
        is_async = info1.get("is_async", False) or info2.get("is_async", False)
        conditional_outputs = list(dict.fromkeys(
            info1.get("conditional_outputs", []) + info2.get("conditional_outputs", [])
        ))
        definite_stores = sorted(
            set(info1.get("definite_stores", [])) & set(info2.get("definite_stores", []))
        )
    else:
        inputs = info1["inputs"]
        outputs = info1["outputs"]
        free_vars = info1["free_vars"]
        nonlocals = info1["nonlocals"]
        globals_ = info1["globals"]
        attrs_read = info1["attrs_read"]
        attrs_written = info1["attrs_written"]
        return_type = info1["return_type"]
        hazards = info1["control_flow_hazards"]
        is_safe = info1["is_control_flow_safe"]
        has_yield = info1["has_yield"]
        has_return = info1["has_return"]
        local_imports = info1["local_imports"]
        yield_expr_names = info1["yield_expr_names"]
        is_async = info1.get("is_async", False)
        conditional_outputs = info1.get("conditional_outputs", [])
        definite_stores = info1.get("definite_stores", [])

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
        "control_flow_hazards": hazards,
        "is_control_flow_safe": is_safe,
        "has_yield": has_yield,
        "has_return": has_return,
        "local_imports": local_imports,
        "yield_expr_names": yield_expr_names,
        "is_async": is_async,
        "conditional_outputs": conditional_outputs,
        "definite_stores": definite_stores,
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


TYPING_SYMBOLS: Set[str] = {
    "Any", "AsyncGenerator", "AsyncIterator", "Callable", "Dict", "Generator", "Iterable",
    "Iterator", "List", "Optional", "Sequence", "Set", "Tuple", "Union",
}


def _extract_required_typing_imports(signature_or_func_text: str) -> List[str]:
    """Identifies typing symbols referenced in type annotations via AST inspection."""
    text = signature_or_func_text.strip()
    tree: Optional[ast.AST] = None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        pass

    if tree is None and "->" in text:
        parts = text.split("->", 1)
        params_part = parts[0].strip()
        if params_part.startswith("(") and params_part.endswith(")"):
            params_part = params_part[1:-1].strip()
        ret_part = parts[1].strip()
        try:
            tree = ast.parse(f"def _sig_wrapper({params_part}) -> {ret_part}: pass")
        except SyntaxError:
            pass

    if tree is None:
        try:
            tree = ast.parse(f"def _sig_wrapper({text}): pass")
        except SyntaxError:
            pass

    if tree is None:
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
        return sorted(s for s in TYPING_SYMBOLS if s in tokens)

    has_defs = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign))
        for n in ast.walk(tree)
    )
    needed: Set[str] = set()
    for node in ast.walk(tree):
        annotations: List[ast.AST] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            for a in all_args:
                if a.annotation is not None:
                    annotations.append(a.annotation)
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None:
                annotations.append(node.annotation)
        elif not has_defs and isinstance(node, ast.Expr):
            annotations.append(node.value)

        for ann in annotations:
            for sub in ast.walk(ann):
                if isinstance(sub, ast.Name) and sub.id in TYPING_SYMBOLS:
                    needed.add(sub.id)
    return sorted(needed)


def _get_module_imported_names(source: str) -> Set[str]:
    """Extracts top-level imported module and symbol names from source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imported: Set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                imported.add(alias.asname or alias.name)
    return imported


def _insert_imports_into_module(
    orig_lines: List[str],
    import_lines: List[str],
) -> List[str]:
    """Inserts import statements after module docstring and future imports, deduplicating existing statements."""
    if not import_lines:
        return orig_lines

    existing_stripped = {ln.strip() for ln in orig_lines}
    deduped_imports = [imp for imp in import_lines if imp.strip() not in existing_stripped]
    if not deduped_imports:
        return orig_lines

    insert_idx = 0
    if insert_idx < len(orig_lines) and orig_lines[insert_idx].startswith("#!"):
        insert_idx += 1

    in_docstring = False
    doc_quote = ""
    for idx in range(insert_idx, len(orig_lines)):
        line = orig_lines[idx].strip()
        if not in_docstring:
            if line.startswith(('"""', "'''")):
                doc_quote = line[:3]
                if line.endswith(doc_quote) and len(line) > 3:
                    insert_idx = idx + 1
                    break
                in_docstring = True
            elif line.startswith("#") or not line:
                continue
            else:
                break
        else:
            if line.endswith(doc_quote):
                insert_idx = idx + 1
                break

    last_future_idx = -1
    for idx in range(insert_idx, len(orig_lines)):
        line = orig_lines[idx].strip()
        if line.startswith("from __future__"):
            last_future_idx = idx
        elif line.startswith("#") or not line:
            continue
        else:
            break

    if last_future_idx != -1:
        insert_idx = last_future_idx + 1
        while insert_idx < len(orig_lines) and not orig_lines[insert_idx].strip():
            insert_idx += 1

    formatted = [imp.rstrip("\r\n") + "\n" for imp in deduped_imports]
    return orig_lines[:insert_idx] + formatted + ["\n"] + orig_lines[insert_idx:]


def synthesize_shared_helper_code(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    include_imports: bool = False,
) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units.

    Args:
        u1: First clone unit dictionary.
        u2: Second clone unit dictionary.
        type_merge_strategy: Strategy for merging conflicting types.
            Options: "fallback_any" (default, falls back to Any)
            or "union" (suggests typing.Union[T1, T2]).
        include_imports: Whether to prepend required typing and local imports.
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
    resolved_ret = _merge_types(r1, r2, type_merge_strategy)

    outputs = list(scope.get("outputs", []))
    conditional_outs = set(scope.get("conditional_outputs", []))
    is_async = scope.get("is_async", False)
    func_keyword = "async def" if is_async else "def"
    await_prefix = "await " if is_async else ""

    # Variables altered via global or nonlocal statements must not be captured in the helper's return tuple
    helper_outputs = [
        v for v in outputs
        if v not in scope.get("globals", [])
        and v not in scope.get("nonlocals", [])
    ]

    if scope.get("has_yield"):
        if is_async:
            return_type = "AsyncIterator[Any]"
        else:
            inferred_yield_type = None
            for kind, name in scope.get("yield_expr_names", []):
                m_t = meta1.get(name, {}).get("type") or meta2.get(name, {}).get("type")
                if not m_t:
                    continue
                if kind == "yield":
                    inferred_yield_type = m_t
                    break
                if kind == "yield_from":
                    for prefix in ("Iterator[", "Iterable[", "List[", "Sequence[", "Set["):
                        if m_t.startswith(prefix) and m_t.endswith("]"):
                            inferred_yield_type = m_t[len(prefix) : -1].strip()
                            break
                    if inferred_yield_type:
                        break
            if resolved_ret != "Any":
                return_type = resolved_ret
            elif inferred_yield_type:
                return_type = f"Iterator[{inferred_yield_type}]"
            else:
                return_type = "Iterator[Any]"
    elif len(helper_outputs) >= 2:
        out_types = []
        for out_var in helper_outputs:
            t1 = meta1.get(out_var, {}).get("type")
            t2 = meta2.get(out_var, {}).get("type")
            t_merged = _merge_types(t1, t2, type_merge_strategy)
            if out_var in conditional_outs:
                if not t_merged.startswith("Optional[") and "None" not in t_merged:
                    t_merged = f"Optional[{t_merged}]"
            out_types.append(t_merged)
        return_type = f"Tuple[{', '.join(out_types)}]"
    elif len(helper_outputs) == 1:
        out_var = helper_outputs[0]
        t1 = meta1.get(out_var, {}).get("type")
        t2 = meta2.get(out_var, {}).get("type")
        out_t = _merge_types(t1, t2, type_merge_strategy)
        if out_var in conditional_outs:
            if not out_t.startswith("Optional[") and "None" not in out_t:
                out_t = f"Optional[{out_t}]"
        return_type = resolved_ret if resolved_ret != "Any" else out_t
        if out_var in conditional_outs:
            if not return_type.startswith("Optional[") and "None" not in return_type and return_type != "None":
                return_type = f"Optional[{return_type}]"
    elif not helper_outputs and not scope.get("has_return") and resolved_ret == "Any":
        return_type = "None"
    else:
        return_type = resolved_ret

    params_str = ", ".join(params) if params else "*args: Any, **kwargs: Any"

    # Dedent common_lines first so relative block indentation is normalized
    common_lines = textwrap.dedent("\n".join(common_lines)).splitlines()

    # Prepend scope modifiers and conditional variable initializations to preserve runtime safety
    prefix_stmts: List[str] = []
    if scope.get("globals"):
        g_vars = sorted(set(scope["globals"]))
        if not any(ln.strip().startswith("global ") for ln in common_lines):
            prefix_stmts.append(f"global {', '.join(g_vars)}")
    if scope.get("nonlocals"):
        nl_vars = sorted(set(scope["nonlocals"]))
        if not any(ln.strip().startswith("nonlocal ") for ln in common_lines):
            prefix_stmts.append(f"nonlocal {', '.join(nl_vars)}")
    for out_var in scope.get("conditional_outputs", []):
        prefix_stmts.append(f"{out_var} = None")

    if prefix_stmts:
        common_lines = prefix_stmts + common_lines

    has_trailing_return = any(
        ln.strip().startswith("return ") or ln.strip() == "return"
        for ln in common_lines[-3:]
    )
    if not has_trailing_return and helper_outputs and not scope.get("has_yield"):
        if len(helper_outputs) >= 2:
            common_lines = common_lines + [f"return {', '.join(helper_outputs)}"]
        else:
            common_lines = common_lines + [f"return {helper_outputs[0]}"]

    docstring_lines = ["    \"\"\"Auto-extracted shared helper for duplicate logic."]
    if helper_outputs:
        if len(helper_outputs) >= 2:
            call_site = f"{', '.join(helper_outputs)} = {await_prefix}{helper_name}(...)"
        else:
            call_site = f"{helper_outputs[0]} = {await_prefix}{helper_name}(...)"
        docstring_lines.append("")
        docstring_lines.append("    Call site:")
        docstring_lines.append(f"        {call_site}")
    else:
        docstring_lines.append("")
        docstring_lines.append("    Call site:")
        docstring_lines.append(f"        {await_prefix}{helper_name}(...)")

    hazards = scope.get("control_flow_hazards", [])
    if hazards:
        hazard_str = ", ".join(hazards)
        docstring_lines.append("")
        docstring_lines.append(
            f"    WARNING: Non-local control flow hazard detected ({hazard_str}). "
            "Direct extraction alters caller control flow semantics."
        )
    docstring_lines.append("    \"\"\"")
    docstring_str = "\n".join(docstring_lines)

    indented_body = "\n".join(f"    {ln}" if ln.strip() else "" for ln in common_lines)
    helper_def = (
        f"{func_keyword} {helper_name}({params_str}) -> {return_type}:\n"
        f"{docstring_str}\n"
        f"{indented_body}\n"
    )

    if include_imports:
        needed_typing = _extract_required_typing_imports(f"{params_str} -> {return_type}")
        import_header_parts: List[str] = []
        if needed_typing:
            import_header_parts.append(f"from typing import {', '.join(sorted(needed_typing))}")
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in import_header_parts:
                import_header_parts.append(loc_imp)
        if import_header_parts:
            return "\n".join(import_header_parts) + "\n\n\n" + helper_def

    return helper_def


def replace_unit_in_source(
    source_text: str,
    unit: Dict[str, Any],
    replacement_text: str,
) -> str:
    """Replaces an AST code unit in source text with replacement text, preserving comments and formatting.

    Args:
        source_text: The complete original Python source code.
        unit: The AST unit mapping containing 1-indexed 'start' and 'end' line numbers.
        replacement_text: The text to substitute in place of the unit lines.

    Returns:
        The modified source code with comments, type ignores, and whitespace outside the unit preserved.
    """
    lines = source_text.splitlines(keepends=True)
    start = max(1, int(unit.get("start", 1)))
    end = min(len(lines), int(unit.get("end", len(lines))))
    if start > len(lines) or start > end:
        return source_text

    rep = replacement_text
    if rep and not rep.endswith("\n"):
        rep += "\n"

    prefix_lines = lines[: start - 1]
    suffix_lines = lines[end:]
    return "".join(prefix_lines) + rep + "".join(suffix_lines)


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
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

        needed_typing = _extract_required_typing_imports(helper_code)
        existing_imports = _get_module_imported_names(orig_text)
        missing_typing = [s for s in needed_typing if s not in existing_imports]

        missing_import_lines: List[str] = []
        if missing_typing:
            missing_import_lines.append(f"from typing import {', '.join(sorted(missing_typing))}")
        scope = analyze_unit_variable_scope(u1, u2)
        for loc_imp in scope.get("local_imports", []):
            if loc_imp not in orig_text and loc_imp not in missing_import_lines:
                missing_import_lines.append(loc_imp)

        current_text = orig_text
        if replace_clones:
            base_name1 = u1["name"].split(":")[0].lstrip("_")
            base_name2 = u2["name"].split(":")[0].lstrip("_")
            helper_name = (
                f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
            )
            await_prefix = "await " if scope.get("is_async") else ""
            inputs = scope.get("inputs", [])
            outputs = scope.get("outputs", [])
            args_str = ", ".join(inputs)
            u1_start = int(u1.get("start", 1))
            if 1 <= u1_start <= len(orig_lines):
                lead = orig_lines[u1_start - 1]
                indent = lead[: len(lead) - len(lead.lstrip())]
                if outputs:
                    if len(outputs) >= 2:
                        call_stmt = f"{indent}{', '.join(outputs)} = {await_prefix}{helper_name}({args_str})\n"
                    else:
                        call_stmt = f"{indent}{outputs[0]} = {await_prefix}{helper_name}({args_str})\n"
                else:
                    call_stmt = f"{indent}{await_prefix}{helper_name}({args_str})\n"
                current_text = replace_unit_in_source(current_text, u1, call_stmt)

        current_lines = current_text.splitlines(keepends=True)
        lines_with_imports = _insert_imports_into_module(current_lines, missing_import_lines)
        modified_lines = [helper_code + "\n\n"] + lines_with_imports

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
