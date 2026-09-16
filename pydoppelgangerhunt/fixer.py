"""Automated helper extraction and git-apply compatible patch synthesizer for code clones."""

from __future__ import annotations

import ast
import builtins
import difflib
import io
import logging
import os
from pathlib import Path
import re
import textwrap
import tokenize
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydoppelgangerhunt.reporters import extract_unit_source_code

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))


class _ScopeVisitor(ast.NodeVisitor):
    """Inspects AST loads, stores, function parameters, returns, nonlocals, globals, and attributes."""

    def __init__(
        self,
        loop_offset: int = 0,
        source_lines: Optional[Sequence[str]] = None,
    ) -> None:
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
        self.source_lines: Optional[List[str]] = list(source_lines) if source_lines is not None else None

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
        stmt: Optional[str] = None
        if self.source_lines is not None:
            lineno = getattr(node, "lineno", None)
            end_lineno = getattr(node, "end_lineno", lineno)
            if lineno is not None and end_lineno is not None:
                if 1 <= lineno <= len(self.source_lines) and 1 <= end_lineno <= len(self.source_lines):
                    raw_slice = self.source_lines[lineno - 1 : end_lineno]
                    raw_str = "".join(raw_slice).strip()
                    if raw_str:
                        stmt = raw_str

        if stmt is None:
            try:
                stmt = ast.unparse(node)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to unparse import node %r: %s", node, exc)
                return

        if stmt not in self.local_imports:
            self.local_imports.append(stmt)
        for alias in node.names:
            name = alias.asname or alias.name
            self.imported_names[name] = stmt

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


def _determine_binding_kind(
    has_instance_binding: bool,
    has_class_binding: bool,
) -> Optional[str]:
    """Determines the scope binding category from instance and class binding flags."""
    if has_instance_binding and has_class_binding:
        return "mixed"
    if has_instance_binding:
        return "instance"
    if has_class_binding:
        return "class"
    return None


def _normalize_receiver_order(
    inputs: List[str],
    has_instance_binding: bool,
    has_class_binding: bool,
) -> None:
    """Ensures self or cls is positioned as the primary receiver parameter in inputs."""
    if has_instance_binding and has_class_binding:
        if "cls" in inputs:
            inputs.remove("cls")
            inputs.insert(0, "cls")
        if "self" in inputs:
            inputs.remove("self")
            inputs.insert(0, "self")
    elif has_instance_binding:
        if "self" in inputs:
            inputs.remove("self")
            inputs.insert(0, "self")
    elif has_class_binding:
        if "cls" in inputs:
            inputs.remove("cls")
            inputs.insert(0, "cls")


def _format_call_arguments(
    inputs: List[str],
    param_details: List[Dict[str, Any]],
    receiver_to_omit: Optional[str] = None,
) -> str:
    """Formats argument strings for helper call sites, preserving keyword-only, vararg, and kwarg syntax."""
    param_map: Dict[str, str] = {
        p["name"].lstrip("*"): p.get("kind", "pos") for p in param_details
    }
    formatted_args: List[str] = []
    for raw_var in inputs:
        clean_var = raw_var.lstrip("*")
        if receiver_to_omit and clean_var == receiver_to_omit:
            continue
        kind = param_map.get(clean_var, "")
        if kind == "kwonly":
            formatted_args.append(f"{clean_var}={clean_var}")
        elif kind == "vararg" or (raw_var.startswith("*") and not raw_var.startswith("**")):
            formatted_args.append(f"*{clean_var}")
        elif kind == "kwarg" or raw_var.startswith("**"):
            formatted_args.append(f"**{clean_var}")
        else:
            formatted_args.append(clean_var)
    return ", ".join(formatted_args)


def _extract_unit_body_lines(unit: Dict[str, Any], raw_lines: List[str]) -> List[str]:
    """Extracts executable body lines for a unit, stripping function headers and docstrings for whole functions."""
    if not (
        unit.get("kind") in ("function", "closure")
        and ":" not in unit.get("name", "")
    ):
        return raw_lines

    code_block = "\n".join(raw_lines)
    dedented = textwrap.dedent(code_block)
    try:
        tree = ast.parse(dedented)
        if len(tree.body) == 1 and isinstance(
            tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            fn_node = tree.body[0]
            body_nodes = list(fn_node.body)
            if (
                body_nodes
                and isinstance(body_nodes[0], ast.Expr)
                and isinstance(body_nodes[0].value, ast.Constant)
                and isinstance(body_nodes[0].value.value, str)
            ):
                body_nodes = body_nodes[1:]
            if body_nodes:
                d_lines = dedented.splitlines()
                start_l = body_nodes[0].lineno
                end_l = getattr(body_nodes[-1], "end_lineno", len(d_lines))
                extracted = d_lines[start_l - 1 : end_l]
                if extracted:
                    return textwrap.dedent("\n".join(extracted)).splitlines()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return raw_lines


def _find_module_helper_insertion_index(lines: List[str]) -> int:
    """Finds the line index after module docstring, future imports, and all module imports."""
    min_insert_idx = 0
    if min_insert_idx < len(lines) and lines[min_insert_idx].startswith("#!"):
        min_insert_idx += 1
    if min_insert_idx < len(lines) and (
        "coding:" in lines[min_insert_idx] or "coding=" in lines[min_insert_idx]
    ):
        min_insert_idx += 1

    try:
        tree = ast.parse("".join(lines))
    except SyntaxError:
        return min_insert_idx

    last_import_line = 0
    docstring_line = 0

    for idx, node in enumerate(tree.body):
        if (
            idx == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docstring_line = getattr(node, "end_lineno", node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            end_l = getattr(node, "end_lineno", node.lineno)
            last_import_line = max(last_import_line, end_l)

    target_line = max(last_import_line, docstring_line, min_insert_idx)
    return min(target_line, len(lines))


def _inspect_unit_scope(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts lexical and AST scope metadata for a single unit."""
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
        "has_instance_binding": False,
        "has_class_binding": False,
        "binding_kind": None,
        "instance_attrs": [],
        "class_attrs": [],
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

    cand_lines = cand_text.splitlines(keepends=True)
    visitor = _ScopeVisitor(loop_offset=loop_offset, source_lines=cand_lines)
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

    has_instance_binding = (
        "self" in visitor.loads
        or "self" in visitor.params
        or any(a.startswith("self.") for a in visitor.attrs_read + visitor.attrs_written)
    )
    has_class_binding = (
        "cls" in visitor.loads
        or "cls" in visitor.params
        or any(a.startswith("cls.") for a in visitor.attrs_read + visitor.attrs_written)
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

    _normalize_receiver_order(inputs, has_instance_binding, has_class_binding)

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
        "has_instance_binding": has_instance_binding,
        "has_class_binding": has_class_binding,
        "binding_kind": binding_kind,
        "instance_attrs": [a for a in visitor.attrs_read + visitor.attrs_written if a.startswith("self.")],
        "class_attrs": [a for a in visitor.attrs_read + visitor.attrs_written if a.startswith("cls.")],
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

    has_instance_binding = info1.get("has_instance_binding", False) or (
        info2.get("has_instance_binding", False) if u2 is not None else False
    )
    has_class_binding = info1.get("has_class_binding", False) or (
        info2.get("has_class_binding", False) if u2 is not None else False
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

    _normalize_receiver_order(inputs, has_instance_binding, has_class_binding)

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
        "has_instance_binding": has_instance_binding,
        "has_class_binding": has_class_binding,
        "binding_kind": binding_kind,
        "instance_attrs": list(dict.fromkeys(
            info1.get("instance_attrs", []) + (info2.get("instance_attrs", []) if u2 is not None else [])
        )),
        "class_attrs": list(dict.fromkeys(
            info1.get("class_attrs", []) + (info2.get("class_attrs", []) if u2 is not None else [])
        )),
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
    if insert_idx < len(orig_lines) and (
        "coding:" in orig_lines[insert_idx] or "coding=" in orig_lines[insert_idx]
    ):
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


def slice_source_by_token_range(
    source_text: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: Optional[int] = None,
) -> str:
    """Extracts a precise slice of source code spanning 1-indexed lines and 0-indexed columns.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number.
        start_col: 0-indexed start column offset.
        end_line: 1-indexed end line number.
        end_col: 0-indexed end column offset (exclusive). If None, includes through the end of end_line.

    Returns:
        The exact string slice of source code between the specified coordinates.
    """
    if not source_text:
        return ""
    lines = source_text.splitlines(keepends=True)
    if not lines or start_line < 1 or start_line > len(lines) or start_line > end_line:
        return ""

    effective_end_line = min(len(lines), end_line)
    if start_line == effective_end_line:
        line_str = lines[start_line - 1]
        start_c = max(0, min(len(line_str), start_col))
        end_c = len(line_str) if end_col is None else max(start_c, min(len(line_str), end_col))
        return line_str[start_c:end_c]

    first_line = lines[start_line - 1]
    start_c = max(0, min(len(first_line), start_col))
    part_first = first_line[start_c:]
    middle_parts = lines[start_line : effective_end_line - 1]
    last_line = lines[effective_end_line - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), end_col))
    part_last = last_line[:end_c]
    return part_first + "".join(middle_parts) + part_last


def extract_unit_comments_and_pragmas(
    source_text: str,
    start_line: int,
    end_line: int,
    start_col: int = 0,
    end_col: Optional[int] = None,
    include_leading: bool = False,
) -> List[Dict[str, Any]]:
    """Extracts comments and pragmas within or attached to a unit's coordinate bounds.

    Uses Python's standard library `tokenize` module to harvest exact token positions
    for comments, identifying `# type: ignore`, `# noqa`, and `# pylint:` annotations.

    Args:
        source_text: The complete original Python source code.
        start_line: 1-indexed start line number of the unit.
        end_line: 1-indexed end line number of the unit.
        start_col: 0-indexed start column offset.
        end_col: 0-indexed end column offset.
        include_leading: If True, also harvests consecutive comment lines immediately
            preceding start_line.

    Returns:
        A list of dictionaries representing discovered comments:
        [{
            "line": int,
            "col": int,
            "end_col": int,
            "text": str,
            "is_pragma": bool,
            "pragma_kind": Optional[str],
        }]
    """
    if not source_text:
        return []

    lines = source_text.splitlines(keepends=True)
    min_check_line = start_line
    if include_leading and start_line > 1:
        curr = start_line - 1
        while curr >= 1:
            stripped = lines[curr - 1].strip()
            if stripped.startswith("#"):
                min_check_line = curr
                curr -= 1
            else:
                break

    results: List[Dict[str, Any]] = []
    try:
        token_gen = tokenize.generate_tokens(io.StringIO(source_text).readline)
        for tok in token_gen:
            if tok.type == tokenize.COMMENT:
                tok_line, tok_col = tok.start
                _, tok_end_col = tok.end

                if min_check_line <= tok_line <= end_line:
                    if tok_line == start_line and tok_line == min_check_line and tok_col < start_col:
                        continue
                    comment_text = tok.string
                    lower = comment_text.lower()
                    is_pragma = False
                    pragma_kind = None
                    if "type: ignore" in lower:
                        is_pragma = True
                        pragma_kind = "type_ignore"
                    elif "noqa" in lower:
                        is_pragma = True
                        pragma_kind = "noqa"
                    elif "pylint:" in lower:
                        is_pragma = True
                        pragma_kind = "pylint"

                    results.append({
                        "line": tok_line,
                        "col": tok_col,
                        "end_col": tok_end_col,
                        "text": comment_text,
                        "is_pragma": is_pragma,
                        "pragma_kind": pragma_kind,
                    })
    except (tokenize.TokenError, IndentationError) as exc:
        logger.debug("Failed to tokenize source for comments: %s", exc)

    return results


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
    except SyntaxError:
        return None

    u_start = int(unit.get("start", 0))
    u_end = int(unit.get("end", u_start))
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

    candidates.sort(key=lambda item: item[0])
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


def find_enclosing_class(
    source_text: str,
    unit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Locates the enclosing class definition for an AST code unit within the source text.

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
    meta["method_indent"] = indent + "    "
    del meta["node"]
    del meta["decorators"]
    del meta["def_start"]
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

    decs = meta.pop("decorators")
    meta.pop("node")
    meta["is_static"] = any(
        (isinstance(d, ast.Name) and d.id == "staticmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
        for d in decs
    )
    meta["is_class_method"] = any(
        (isinstance(d, ast.Name) and d.id == "classmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "classmethod")
        for d in decs
    )
    return meta


def _get_enclosing_receiver_kind(fn_info: Optional[Dict[str, Any]]) -> str:
    """Returns the receiver kind for an enclosing function: static, class, instance, or none."""
    if not fn_info:
        return "none"
    if fn_info.get("is_static"):
        return "static"
    if fn_info.get("is_class_method"):
        return "class"
    return "instance"


def _resolve_effective_binding(
    method_binding: str,
    is_same_class: bool,
    is_static: bool = False,
    receiver_kinds_differ: bool = False,
) -> str:
    """Resolve the effective helper binding mode ('method' vs 'module')."""
    if method_binding in ("method", "module"):
        return method_binding
    if is_static or receiver_kinds_differ:
        return "module"
    return "method" if is_same_class else "module"


def synthesize_shared_helper_code(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    type_merge_strategy: str = "fallback_any",
    include_imports: bool = False,
    preserve_pragmas: bool = True,
    method_binding: str = "auto",
    indent: str = "",
    is_static: bool = False,
) -> str:
    """Synthesizes a proposed shared helper function stub from two clone units.

    Args:
        u1: First clone unit dictionary.
        u2: Second clone unit dictionary.
        type_merge_strategy: Strategy for merging conflicting types.
            Options: "fallback_any" (default, falls back to Any)
            or "union" (suggests typing.Union[T1, T2]).
        include_imports: Whether to prepend required typing and local imports.
        preserve_pragmas: Whether to preserve comments and pragmas (# type: ignore / # noqa)
            from the original clone units in the extracted helper function.
        method_binding: Scope binding strategy ("auto", "method", "module").
            "auto" emits a private class method if both clones reside in the same class,
            or a module-level helper with explicit receiver injection otherwise.
        indent: Base indentation prefix for helper definition lines (defaults to 4 spaces for methods).
        is_static: Whether the cloned methods are static methods.
    """
    lines1 = _extract_unit_body_lines(u1, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u1)])
    lines2 = _extract_unit_body_lines(u2, [ln.rstrip("\r\n") for ln in extract_unit_source_code(u2)])

    base_name1 = u1["name"].split(":")[0].lstrip("_")
    base_name2 = u2["name"].split(":")[0].lstrip("_")
    helper_name = f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"

    # Find common lines using difflib matching
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    common_lines: List[str] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            common_lines.extend(lines1[i1:i2])

    if not common_lines or (preserve_pragmas and len(common_lines) < max(len(lines1), len(lines2)) * 0.5):
        common_lines = lines1

    if preserve_pragmas:
        raw1 = "\n".join(lines1)
        raw2 = "\n".join(lines2)
        pragmas1 = extract_unit_comments_and_pragmas(raw1, 1, len(lines1))
        pragmas2 = extract_unit_comments_and_pragmas(raw2, 1, len(lines2))
        for p in pragmas1 + pragmas2:
            if p["is_pragma"]:
                p_text = p["text"].strip()
                if not any(p_text in ln for ln in common_lines):
                    for idx, ln in enumerate(common_lines):
                        if ln.strip() and not ln.strip().startswith("#"):
                            common_lines[idx] = ln.rstrip() + f"  {p_text}"
                            break

    # Variable scope analysis for concrete parameter signatures
    scope1 = analyze_unit_variable_scope(u1)
    scope2 = analyze_unit_variable_scope(u2)
    scope = analyze_unit_variable_scope(u1, u2)

    enc1 = u1.get("enclosing_class")
    enc2 = u2.get("enclosing_class")
    f1 = u1.get("file", "").split("#")[0].replace("\\", "/")
    f2 = u2.get("file", "").split("#")[0].replace("\\", "/")
    is_same_class = bool(enc1 and enc2 and enc1 == enc2 and f1 == f2)

    inputs = list(scope["inputs"])
    is_static_clone = bool(
        is_static
        or scope.get("binding_kind") == "static"
        or u1.get("is_static")
        or u2.get("is_static")
    )

    effective_binding = _resolve_effective_binding(
        method_binding, is_same_class, is_static=is_static_clone
    )

    if effective_binding == "method" and not indent:
        indent = "    "

    meta1 = {p["name"].lstrip("*"): p for p in scope1.get("param_details", [])}
    meta2 = {p["name"].lstrip("*"): p for p in scope2.get("param_details", [])}

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

    # In method binding mode, ensure receiver parameter exists (unless static)
    if (
        effective_binding == "method"
        and not is_static_clone
        and not any(desc["var"] in ("self", "cls") for desc in descriptors)
    ):
        rec_var = "cls" if scope.get("binding_kind") == "class" else "self"
        descriptors.insert(0, {
            "var": rec_var,
            "type": "Any",
            "default": None,
            "kind": "pos",
        })

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

        if var in ("self", "cls") and effective_binding == "method":
            params.append(var)
        elif resolved_default is not None:
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

    body_indent = indent + "    "
    doc_indent = indent + "    "
    sub_indent = indent + "        "
    if effective_binding == "method":
        if scope.get("binding_kind") == "class":
            call_target = f"cls.{helper_name}"
        elif is_static_clone:
            call_target = f"{enc1 or 'ClassName'}.{helper_name}"
        else:
            call_target = f"self.{helper_name}"
    else:
        call_target = helper_name

    docstring_lines = [f"{doc_indent}\"\"\"Auto-extracted shared helper for duplicate logic."]
    if helper_outputs:
        if len(helper_outputs) >= 2:
            call_site = f"{', '.join(helper_outputs)} = {await_prefix}{call_target}(...)"
        else:
            call_site = f"{helper_outputs[0]} = {await_prefix}{call_target}(...)"
        docstring_lines.append("")
        docstring_lines.append(f"{doc_indent}Call site:")
        docstring_lines.append(f"{sub_indent}{call_site}")
    else:
        docstring_lines.append("")
        docstring_lines.append(f"{doc_indent}Call site:")
        docstring_lines.append(f"{sub_indent}{await_prefix}{call_target}(...)")

    hazards = scope.get("control_flow_hazards", [])
    if hazards:
        hazard_str = ", ".join(hazards)
        docstring_lines.append("")
        docstring_lines.append(
            f"{doc_indent}WARNING: Non-local control flow hazard detected ({hazard_str}). "
            "Direct extraction alters caller control flow semantics."
        )
    docstring_lines.append(f"{doc_indent}\"\"\"")
    docstring_str = "\n".join(docstring_lines)

    indented_body = "\n".join(f"{body_indent}{ln}" if ln.strip() else "" for ln in common_lines)
    if effective_binding == "method":
        if scope.get("binding_kind") == "class":
            dec_prefix = f"{indent}@classmethod\n"
        elif is_static_clone:
            dec_prefix = f"{indent}@staticmethod\n"
        else:
            dec_prefix = ""
    else:
        dec_prefix = ""
    helper_def = (
        f"{dec_prefix}"
        f"{indent}{func_keyword} {helper_name}({params_str}) -> {return_type}:\n"
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
    preserve_boundary_pragmas: bool = True,
) -> str:
    """Replaces an AST code unit in source text with replacement text, preserving comments and formatting.

    Supports sub-line and token-range precision slicing when 'start_col' and 'end_col'
    are present in the unit dictionary. Preserves surrounding indentation, preceding code
    on the start line, trailing pragmas/comments on the end line, and attached pragmas.

    Args:
        source_text: The complete original Python source code.
        unit: The AST unit mapping containing 1-indexed 'start' and 'end' line numbers,
            and optional 0-indexed 'start_col' and 'end_col' column offsets.
        replacement_text: The text to substitute in place of the unit lines.
        preserve_boundary_pragmas: If True, retains boundary pragmas (# type: ignore / # noqa)
            attached to the original code unit and appends them to the replacement statement.

    Returns:
        The modified source code with comments, type ignores, and whitespace outside the unit preserved.
    """
    lines = source_text.splitlines(keepends=True)
    start = max(1, int(unit.get("start", 1)))
    end = min(len(lines), int(unit.get("end", len(lines))))
    if start > len(lines) or start > end:
        return source_text

    start_col = unit.get("start_col")
    end_col = unit.get("end_col")

    rep = replacement_text

    # Extract boundary pragmas if requested
    attached_pragmas: List[str] = []
    if preserve_boundary_pragmas:
        boundary_comments = extract_unit_comments_and_pragmas(
            source_text, start_line=start, end_line=end, start_col=start_col or 0, end_col=end_col
        )
        for c in boundary_comments:
            if c["is_pragma"] and c["line"] in (start, end):
                p_text = c["text"].strip()
                if p_text not in rep and p_text not in attached_pragmas:
                    attached_pragmas.append(p_text)

    first_line = lines[start - 1]
    start_c = max(0, min(len(first_line), int(start_col or 0)))
    last_line = lines[end - 1]
    end_c = len(last_line) if end_col is None else max(0, min(len(last_line), int(end_col)))

    prefix_is_whitespace = not first_line[:start_c].strip()
    suffix_stripped = last_line[end_c:].strip()
    suffix_is_boundary_only = not suffix_stripped or suffix_stripped.startswith("#")
    is_column_bounded = (start_col is not None or end_col is not None) and not (
        prefix_is_whitespace and suffix_is_boundary_only
    )

    if is_column_bounded:
        prefix_line = first_line[:start_c]
        suffix_line = last_line[end_c:]

        prefix_all = "".join(lines[: start - 1]) + prefix_line
        suffix_all = suffix_line + "".join(lines[end:])

        final_rep = rep
        if attached_pragmas and not any(p in suffix_line for p in attached_pragmas):
            pragma_suffix = "  " + "  ".join(attached_pragmas)
            if final_rep.endswith("\n"):
                final_rep = final_rep[:-1] + pragma_suffix + "\n"
            else:
                final_rep += pragma_suffix

        return prefix_all + final_rep + suffix_all

    # Whole-line replacement
    if attached_pragmas and rep:
        pragma_suffix = "  " + "  ".join(attached_pragmas)
        rep_lines = rep.splitlines(keepends=True)
        if rep_lines:
            last_rep = rep_lines[-1]
            if last_rep.endswith("\n"):
                rep_lines[-1] = last_rep[:-1] + pragma_suffix + "\n"
            else:
                rep_lines[-1] = last_rep + pragma_suffix
            rep = "".join(rep_lines)

    if rep and not rep.endswith("\n"):
        rep += "\n"

    prefix_lines = lines[: start - 1]
    suffix_lines = lines[end:]
    return "".join(prefix_lines) + rep + "".join(suffix_lines)


def check_units_overlap(u1: Dict[str, Any], u2: Dict[str, Any]) -> bool:
    """Determines whether two AST code units in the same file share overlapping line ranges.

    Args:
        u1: First AST unit dictionary with 'file', 'start', and 'end'.
        u2: Second AST unit dictionary with 'file', 'start', and 'end'.

    Returns:
        True if both units reside in the same normalized file path and their [start, end]
        intervals overlap; False otherwise.
    """
    f1 = u1.get("file", "").split("#")[0].replace("\\", "/")
    f2 = u2.get("file", "").split("#")[0].replace("\\", "/")
    if not f1 or not f2 or f1 != f2:
        return False
    start1, end1 = int(u1.get("start", 1)), int(u1.get("end", 1))
    start2, end2 = int(u2.get("start", 1)), int(u2.get("end", 1))
    return max(start1, start2) <= min(end1, end2)


def filter_overlapping_clone_units(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters a sequence of candidate code units to retain a maximal non-overlapping subset.

    When candidates share overlapping lines/tokens within the same module, candidates
    spanning a larger line range are prioritized, with ties broken by earlier start line.

    Args:
        units: Sequence of AST unit dictionaries proposed for refactoring.

    Returns:
        List of non-overlapping units safe for concurrent refactoring within the same pass.
    """
    if not units:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for u in units:
        norm_file = u.get("file", "").split("#")[0].replace("\\", "/")
        grouped.setdefault(norm_file, []).append(u)

    retained: List[Dict[str, Any]] = []
    for file_units in grouped.values():
        sorted_candidates = sorted(
            file_units,
            key=lambda u: (
                -(int(u.get("end", 0)) - int(u.get("start", 0))),
                int(u.get("start", 0)),
                str(u.get("name", "")),
            ),
        )
        file_retained: List[Dict[str, Any]] = []
        for cand in sorted_candidates:
            if not any(check_units_overlap(cand, prev) for prev in file_retained):
                file_retained.append(cand)
        retained.extend(file_retained)

    return retained


def refactor_module_units(
    source_text: str,
    replacements: Sequence[Tuple[Dict[str, Any], str]],
) -> str:
    """Applies multiple non-overlapping unit replacements in reverse source order.

    Applying substitutions bottom-to-top (descending by unit 'start' line) guarantees
    that line count changes downstream never invalidate the 1-indexed source line
    coordinates of earlier units in the same file.

    Args:
        source_text: The complete original Python source code.
        replacements: Sequence of (unit, replacement_text) tuples.

    Returns:
        The refactored module source text.

    Raises:
        ValueError: If any pair of units in replacements shares overlapping line ranges.
    """
    if not replacements:
        return source_text

    rep_list = list(replacements)
    for i, (u1, _) in enumerate(rep_list):
        for u2, _ in rep_list[i + 1:]:
            if check_units_overlap(u1, u2):
                raise ValueError(
                    f"Overlapping unit collision detected between "
                    f"'{u1.get('name')}' ({u1.get('start')}-{u1.get('end')}) and "
                    f"'{u2.get('name')}' ({u2.get('start')}-{u2.get('end')}) in {u1.get('file')}."
                )

    sorted_replacements = sorted(
        rep_list,
        key=lambda item: int(item[0].get("start", 0)),
        reverse=True,
    )

    current_text = source_text
    for unit, replacement in sorted_replacements:
        current_text = replace_unit_in_source(current_text, unit, replacement)

    return current_text


def _build_whole_method_delegation(
    source_text: str,
    unit: Dict[str, Any],
    call_prefix: str,
    helper_name: str,
    args_str: str,
    await_prefix: str = "",
    has_return: bool = True,
    is_async: bool = False,
    has_yield: bool = False,
) -> str:
    """Builds a delegated method replacement body preserving method signature and docstring."""
    lines = source_text.splitlines(keepends=True)
    u_start = int(unit.get("start", 1))
    u_end = int(unit.get("end", len(lines)))

    lead = lines[u_start - 1] if 1 <= u_start <= len(lines) else ""
    indent = lead[: len(lead) - len(lead.lstrip())]
    body_indent = indent + "    "

    sig_end_line = u_start
    docstring_end_line: Optional[int] = None

    try:
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n_start = getattr(node, "lineno", 0)
                n_end = getattr(node, "end_lineno", n_start)
                if n_start == u_start or (node.name == unit.get("name") and n_start <= u_start <= n_end):
                    if node.body:
                        first_body = node.body[0]
                        sig_end_line = first_body.lineno - 1
                        if (
                            isinstance(first_body, ast.Expr)
                            and isinstance(first_body.value, ast.Constant)
                            and isinstance(first_body.value.value, str)
                        ):
                            docstring_end_line = getattr(first_body, "end_lineno", first_body.lineno)
                    break
    except SyntaxError:
        pass

    if docstring_end_line is not None and docstring_end_line >= u_start:
        header = "".join(lines[u_start - 1 : docstring_end_line])
    elif sig_end_line >= u_start:
        header = "".join(lines[u_start - 1 : sig_end_line])
    else:
        hdr_lines = []
        for l_num in range(u_start, min(u_end + 1, len(lines) + 1)):
            ln = lines[l_num - 1]
            hdr_lines.append(ln)
            if ln.rstrip().endswith(":"):
                break
        header = "".join(hdr_lines)

    if not header.endswith("\n"):
        header += "\n"

    effective_async = is_async or bool(await_prefix.strip())
    if effective_async and has_yield:
        delegation_stmt = (
            f"{body_indent}async for _item in {call_prefix}{helper_name}({args_str}):\n"
            f"{body_indent}    yield _item\n"
        )
    elif has_yield:
        delegation_stmt = f"{body_indent}yield from {call_prefix}{helper_name}({args_str})\n"
    elif effective_async:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}await {call_prefix}{helper_name}({args_str})\n"
    else:
        ret_prefix = "return " if has_return else ""
        delegation_stmt = f"{body_indent}{ret_prefix}{call_prefix}{helper_name}({args_str})\n"
    return header + delegation_stmt


def generate_refactoring_patch(
    clones: List[Tuple[float, Dict[str, Any], Dict[str, Any]]],
    repo_root: Optional[str] = None,
    type_merge_strategy: str = "fallback_any",
    replace_clones: bool = False,
    method_binding: str = "auto",
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

        enc1 = find_enclosing_class(orig_text, u1)
        f2_raw = u2.get("file", "").split("#")[0]
        enc2 = find_enclosing_class(orig_text, u2) if (f2_raw == f1_raw) else None

        is_same_class = bool(
            enc1 and enc2 and enc1["name"] == enc2["name"] and enc1["start"] == enc2["start"]
        )

        fn1 = find_enclosing_function(orig_text, u1)
        fn2 = find_enclosing_function(orig_text, u2) if (f2_raw == f1_raw) else None
        fn1_kind = _get_enclosing_receiver_kind(fn1)
        fn2_kind = _get_enclosing_receiver_kind(fn2) if fn2 else fn1_kind
        receiver_kinds_differ = bool(fn2 and fn1_kind != fn2_kind)
        is_static = bool((fn1 and fn1.get("is_static")) or (fn2 and fn2.get("is_static")))
        if is_static:
            u1["is_static"] = True
            u2["is_static"] = True

        effective_binding = _resolve_effective_binding(
            method_binding,
            is_same_class,
            is_static=is_static,
            receiver_kinds_differ=receiver_kinds_differ,
        )

        helper_indent = (
            enc1["method_indent"]
            if (effective_binding == "method" and enc1)
            else ("    " if effective_binding == "method" else "")
        )

        helper_code = synthesize_shared_helper_code(
            u1,
            u2,
            type_merge_strategy=type_merge_strategy,
            method_binding=effective_binding,
            indent=helper_indent,
            is_static=is_static,
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

        base_name1 = u1["name"].split(":")[0].lstrip("_")
        base_name2 = u2["name"].split(":")[0].lstrip("_")
        helper_name = (
            f"_shared_{base_name1}" if base_name1 == base_name2 else f"_shared_{base_name1}_{base_name2}"
        )
        await_prefix = "await " if scope.get("is_async") else ""
        inputs = list(scope.get("inputs", []))
        outputs = list(scope.get("outputs", []))

        candidate_units = [u1]
        if f2_raw and f2_raw == f1_raw and not check_units_overlap(u1, u2):
            candidate_units.append(u2)

        earliest_unit = min(candidate_units, key=lambda u: int(u.get("start", 1)))
        enc_fn_earliest = (
            find_enclosing_function(orig_text, earliest_unit)
            if effective_binding == "method"
            else None
        )
        insert_line = (
            enc_fn_earliest["start"] if enc_fn_earliest else int(earliest_unit.get("start", 1))
        )
        current_text = orig_text

        if replace_clones:
            units_to_replace: List[Tuple[Dict[str, Any], str]] = []
            for target_unit in candidate_units:
                u_start = int(target_unit.get("start", 1))
                if 1 <= u_start <= len(orig_lines):
                    lead = orig_lines[u_start - 1]
                    indent = lead[: len(lead) - len(lead.lstrip())]

                    if effective_binding == "method":
                        t_fn = find_enclosing_function(orig_text, target_unit)
                        t_kind = _get_enclosing_receiver_kind(t_fn)
                        if t_kind == "static":
                            cls_name = enc1["name"] if enc1 else ""
                            unit_call_prefix = f"{cls_name}." if cls_name else ""
                            unit_receiver_omit = None
                        elif t_kind == "class":
                            unit_call_prefix = "cls."
                            unit_receiver_omit = "cls"
                        else:
                            unit_call_prefix = "self."
                            unit_receiver_omit = "self"
                    else:
                        unit_call_prefix = ""
                        unit_receiver_omit = None

                    unit_args_str = _format_call_arguments(
                        inputs,
                        scope.get("param_details", []),
                        receiver_to_omit=unit_receiver_omit,
                    )

                    is_whole_method = (
                        target_unit.get("kind") in ("function", "closure")
                        and ":" not in target_unit.get("name", "")
                    )
                    if is_whole_method:
                        rep_stmt = _build_whole_method_delegation(
                            orig_text,
                            target_unit,
                            call_prefix=unit_call_prefix,
                            helper_name=helper_name,
                            args_str=unit_args_str,
                            await_prefix=await_prefix,
                            has_return=bool(outputs or scope.get("has_return", True)),
                            is_async=bool(scope.get("is_async")),
                            has_yield=bool(scope.get("has_yield")),
                        )
                    else:
                        if bool(scope.get("has_yield")):
                            if scope.get("is_async"):
                                rep_stmt = (
                                    f"{indent}async for _item in "
                                    f"{unit_call_prefix}{helper_name}({unit_args_str}):\n"
                                    f"{indent}    yield _item\n"
                                )
                            else:
                                rep_stmt = (
                                    f"{indent}yield from "
                                    f"{unit_call_prefix}{helper_name}({unit_args_str})\n"
                                )
                        elif outputs:
                            assign_target = (
                                ", ".join(outputs) if len(outputs) >= 2 else outputs[0]
                            )
                            rep_stmt = (
                                f"{indent}{assign_target} = "
                                f"{await_prefix}{unit_call_prefix}{helper_name}({unit_args_str})\n"
                            )
                        else:
                            rep_stmt = (
                                f"{indent}{await_prefix}{unit_call_prefix}"
                                f"{helper_name}({unit_args_str})\n"
                            )

                    units_to_replace.append((target_unit, rep_stmt))

            current_text = refactor_module_units(orig_text, units_to_replace)

        if effective_binding == "method":
            current_lines = current_text.splitlines(keepends=True)
            insert_idx = max(0, insert_line - 1)
            current_lines = current_lines[:insert_idx] + [helper_code + "\n"] + current_lines[insert_idx:]
            modified_lines = _insert_imports_into_module(current_lines, missing_import_lines)
        else:
            current_lines = current_text.splitlines(keepends=True)
            lines_with_imports = _insert_imports_into_module(current_lines, missing_import_lines)
            ins_idx = _find_module_helper_insertion_index(lines_with_imports)
            modified_lines = (
                lines_with_imports[:ins_idx]
                + ["\n", helper_code + "\n\n"]
                + lines_with_imports[ins_idx:]
            )

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
