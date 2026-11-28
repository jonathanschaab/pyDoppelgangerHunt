"""Host-module import resolution, helper symbol extraction, and future-annotation checks."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt._ast_compat import safe_parse_ast
from pydoppelgangerhunt.fixer._df_cfg import (
    _collect_scope_direct_bindings,
    _extract_arg_names,
    _extract_import_bound_names,
    _extract_pattern_bound_name,
    _extract_target_bound_names,
    _extract_type_alias_name,
    _extract_type_param_names,
    _iter_target_leaves,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _extract_comprehension_walrus_names,
    iter_nested_scope_header_nodes,
)
from pydoppelgangerhunt.fixer._patch_delegation import _FilePatchPlan
from pydoppelgangerhunt.fixer.depgraph import (
    _collect_top_level_import_nodes,
    _resolve_relative_import_path,
)
from pydoppelgangerhunt.fixer.source import (
    _get_module_imported_names,
    detect_line_ending,
    split_source_lines,
)
from pydoppelgangerhunt.fixer._synthesis_types import _extract_required_typing_imports

_BUILTIN_NAMES: Set[str] = set(dir(builtins))


def _is_or_contains_bitor(node: ast.AST) -> bool:
    """Checks if an AST expression node contains a bitwise OR (|) operator."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return True
    return False


def _has_future_annotations(source_text: str) -> bool:
    """Checks whether source_text enables postponed evaluation of annotations via __future__."""
    if not source_text or "__future__" not in source_text:
        return False
    tree = safe_parse_ast(source_text)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    return True
    return False


def _helper_requires_future_annotations(helper_code: str) -> bool:
    """Detects whether helper annotations use PEP 604 union syntax (|)."""
    if not helper_code or "|" not in helper_code:
        return False
    tree = safe_parse_ast(helper_code)
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_args = [
                *getattr(node.args, "posonlyargs", []),
                *node.args.args,
                *getattr(node.args, "kwonlyargs", []),
                *(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None),
            ]
            for arg in all_args:
                if arg.annotation and _is_or_contains_bitor(arg.annotation):
                    return True
            if node.returns and _is_or_contains_bitor(node.returns):
                return True
    return False


class _ModuleImportsResult(Tuple[Dict[str, str], List[str]]):
    """Encapsulates extracted module imports with binding classification.

    Inherits from Tuple[Dict[str, str], List[str]] so callers unpacking
    (stmts, wildcards) maintain 100% backwards compatibility.
    """

    stmts: Dict[str, str]
    wildcards: List[str]
    unconditional_names: Set[str]
    guarded_names: Set[str]
    rebound_conflicts: Set[str]

    def __new__(
        cls,
        stmts: Dict[str, str],
        wildcards: List[str],
        unconditional_names: Optional[Set[str]] = None,
        guarded_names: Optional[Set[str]] = None,
        rebound_conflicts: Optional[Set[str]] = None,
    ) -> "_ModuleImportsResult":
        instance = super().__new__(cls, (stmts, wildcards))
        instance.stmts = stmts
        instance.wildcards = wildcards
        instance.unconditional_names = (
            set() if unconditional_names is None else unconditional_names
        )
        instance.guarded_names = (
            set() if guarded_names is None else guarded_names
        )
        instance.rebound_conflicts = (
            set() if rebound_conflicts is None else rebound_conflicts
        )
        return instance


def _parse_import_node_statements(
    node: Union[ast.Import, ast.ImportFrom],
    source_mod: Optional[str] = None,
    is_package: bool = False,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Extracts (bound_name, import_stmt) pairs and wildcard strings from an AST import node."""
    name_stmts: List[Tuple[str, str]] = []
    wildcards: List[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if name != "*":
                if alias.asname:
                    stmt = f"import {alias.name} as {alias.asname}"
                else:
                    stmt = f"import {alias.name}"
                name_stmts.append((name, stmt))
    elif isinstance(node, ast.ImportFrom):
        level = getattr(node, "level", 0) or 0
        target_module = ""
        if level == 0 and node.module:
            target_module = node.module
        elif level > 0:
            if source_mod:
                target_module = _resolve_relative_import_path(
                    source_mod, level, node.module, is_package=is_package
                )
            else:
                dots = "." * level
                target_module = f"{dots}{node.module}" if node.module else dots

        if target_module:
            for alias in node.names:
                if alias.name == "*":
                    wildcards.append(f"from {target_module} import *")
                else:
                    name = alias.asname or alias.name
                    if alias.asname:
                        stmt = (
                            f"from {target_module} import {alias.name} as {alias.asname}"
                        )
                    else:
                        stmt = f"from {target_module} import {alias.name}"
                    name_stmts.append((name, stmt))
    return name_stmts, wildcards


def _extract_module_import_statements(
    source_text: str,
    source_mod: Optional[str] = None,
    is_package: bool = False,
) -> _ModuleImportsResult:
    """Maps imported symbol/module names to clean import statements from source text.

    Tracks final top-level bindings for rebound symbols, distinguishes unconditional
    top-level imports from runtime-guarded imports, and detects conflicting rebindings.
    """
    stmts: Dict[str, str] = {}
    wildcards: List[str] = []
    unconditional_names: Set[str] = set()
    guarded_names: Set[str] = set()
    rebound_conflicts: Set[str] = set()

    if not source_text.strip():
        return _ModuleImportsResult(stmts, wildcards)
    tree = safe_parse_ast(source_text)
    if tree is None:
        return _ModuleImportsResult({}, [])
    try:
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                name_stmts, wilds = _parse_import_node_statements(
                    stmt, source_mod=source_mod, is_package=is_package
                )
                wildcards.extend(wilds)
                for name, stmt_str in name_stmts:
                    if name in unconditional_names and stmts.get(name) != stmt_str:
                        rebound_conflicts.add(name)
                    stmts[name] = stmt_str
                    unconditional_names.add(name)
            else:
                guarded_nodes = _collect_top_level_import_nodes(
                    [stmt], include_classes=False
                )
                for g_node in guarded_nodes:
                    name_stmts, wilds = _parse_import_node_statements(
                        g_node, source_mod=source_mod, is_package=is_package
                    )
                    wildcards.extend(wilds)
                    for name, stmt_str in name_stmts:
                        guarded_names.add(name)
                        if name in unconditional_names and stmts.get(name) != stmt_str:
                            rebound_conflicts.add(name)
                        elif name not in stmts:
                            stmts[name] = stmt_str
    except RecursionError:
        return _ModuleImportsResult({}, [])

    return _ModuleImportsResult(
        stmts=stmts,
        wildcards=wildcards,
        unconditional_names=unconditional_names,
        guarded_names=guarded_names,
        rebound_conflicts=rebound_conflicts,
    )


def _canonicalize_helper_relative_imports(
    helper_code: str,
    source_mods: Sequence[Union[str, Tuple[Optional[str], bool], None]] = (),
) -> str:
    """Translates relative imports inside a helper body to canonical absolute imports."""
    if not helper_code:
        return helper_code
    valid_mods: List[Tuple[str, bool]] = []
    for item in source_mods:
        if not item:
            continue
        if isinstance(item, tuple):
            if item[0]:
                valid_mods.append((item[0], bool(item[1])))
        else:
            valid_mods.append((item, False))
    if not valid_mods:
        return helper_code
    tree = safe_parse_ast(helper_code)
    if tree is None:
        return helper_code

    import_nodes: List[ast.ImportFrom] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (getattr(node, "level", 0) or 0) > 0:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                import_nodes.append(node)

    if not import_nodes:
        return helper_code

    import_nodes.sort(
        key=lambda n: (
            n.lineno if n.lineno is not None else 0,
            getattr(n, "col_offset", 0) or 0,
        ),
        reverse=True,
    )
    nl = detect_line_ending(helper_code)
    lines = [ln.rstrip("\r\n") for ln in split_source_lines(helper_code)]

    for node in import_nodes:
        resolved_targets: Set[str] = set()
        for s_mod, s_ispkg in valid_mods:
            resolved = _resolve_relative_import_path(
                s_mod, node.level, node.module, is_package=s_ispkg
            )
            if resolved:
                resolved_targets.add(resolved)
        if len(resolved_targets) > 1:
            raise ValueError(
                "conflicting relative import in helper across clone sources"
            )
        if not resolved_targets:
            continue
        resolved = next(iter(resolved_targets))

        if node.lineno is None:
            continue
        start_idx = node.lineno - 1
        end_idx = node.end_lineno if node.end_lineno is not None else node.lineno
        if start_idx < 0 or start_idx >= len(lines) or end_idx < start_idx:
            continue

        names_str = ", ".join(
            f"{a.name} as {a.asname}" if a.asname else a.name
            for a in node.names
        )

        col_start = getattr(node, "col_offset", None)
        col_end = getattr(node, "end_col_offset", None)
        if (
            start_idx == end_idx - 1
            and col_start is not None
            and col_end is not None
            and 0 <= col_start <= col_end <= len(lines[start_idx])
        ):
            line = lines[start_idx]
            new_slice = f"from {resolved} import {names_str}"
            lines[start_idx] = line[:col_start] + new_slice + line[col_end:]
        else:
            first_line = lines[start_idx]
            indent = first_line[: len(first_line) - len(first_line.lstrip())]
            trailing_comment = ""
            last_line = lines[end_idx - 1] if 0 <= end_idx - 1 < len(lines) else first_line
            hash_pos = last_line.find("#")
            if hash_pos != -1:
                trailing_comment = f"  {last_line[hash_pos:].strip()}"
            new_stmt = f"{indent}from {resolved} import {names_str}{trailing_comment}"
            lines[start_idx:end_idx] = [new_stmt]

    result = nl.join(lines)
    if helper_code.endswith(("\r", "\n")) and not result.endswith(("\r", "\n")):
        result += nl
    return result


def _unpack_source_item(
    item: Union[str, Tuple[Any, ...], List[Any]]
) -> Tuple[str, Optional[str], bool]:
    if isinstance(item, (tuple, list)):
        text = str(item[0]) if len(item) > 0 else ""
        mod = str(item[1]) if len(item) > 1 and item[1] is not None else None
        ispkg = bool(item[2]) if len(item) > 2 else False
        return text, mod, ispkg
    return str(item), None, False


def _extract_module_defined_names(source_text: str) -> Set[str]:
    """Extracts top-level defined names (classes, functions, variables) from source text."""
    if not source_text.strip():
        return set()
    tree = safe_parse_ast(source_text)
    if tree is None:
        return set()
    try:
        direct = _collect_scope_direct_bindings(
            tree.body,
            include_imports=False,
            include_except_names=False,
            exclude_declared_globals=False,
        )
    except RecursionError:
        return set()
    return set(direct.locals)


@dataclass
class _HelperScopeFrame:
    """Typed lexical scope frame used while analyzing helper symbol references."""

    locals: Set[str] = field(default_factory=set)
    globals: Set[str] = field(default_factory=set)
    nonlocals: Set[str] = field(default_factory=set)
    is_class: bool = False
    is_comp: bool = False


class _HelperSymbolVisitor(ast.NodeVisitor):
    """Scope-aware visitor collecting free, defined, and locally imported names in a helper AST."""

    def __init__(self, max_helper_scope_depth: int) -> None:
        self._max_helper_scope_depth = max_helper_scope_depth
        self.free_names: Set[str] = set()
        self.defined_names: Set[str] = set()
        self.local_imported_names: Set[str] = set()
        self._scope_stack: List[_HelperScopeFrame] = [_HelperScopeFrame()]
        self._type_param_scopes: List[Set[str]] = []

    def _bind_to_frame(
        self, frame: _HelperScopeFrame, names: Union[Set[str], Sequence[str]]
    ) -> None:
        for n in names:
            if n in frame.globals:
                self.free_names.add(n)
            elif n in frame.nonlocals:
                self._check_load_name(n)
            else:
                frame.locals.add(n)
                if frame is self._scope_stack[0]:
                    self.defined_names.add(n)

    def _bind_names(
        self, names: Union[Set[str], Sequence[str]], *, is_walrus: bool = False
    ) -> None:
        if is_walrus:
            for frame in reversed(self._scope_stack):
                if not frame.is_comp:
                    if not (frame.is_class and self._scope_stack[-1].is_comp):
                        self._bind_to_frame(frame, names)
                    break
            return
        self._bind_to_frame(self._scope_stack[-1], names)

    def _check_load_name(self, name: str) -> None:
        if not any(name in tp for tp in self._type_param_scopes) and not any(
            name in frame.locals
            for frame in reversed(self._scope_stack)
            if not frame.is_class or frame is self._scope_stack[-1]
        ):
            self.free_names.add(name)

    def _visit_with_type_params(
        self,
        node: ast.AST,
        inner_nodes: Sequence[Optional[ast.AST]],
        *,
        record_def: bool = False,
    ) -> Set[str]:
        if record_def:
            for d in getattr(node, "decorator_list", []):
                self.visit(d)
            fn_args = getattr(node, "args", None)
            if fn_args is not None:
                for default in list(getattr(fn_args, "defaults", [])) + list(
                    getattr(fn_args, "kw_defaults", [])
                ):
                    if default is not None:
                        self.visit(default)
        tp_list = getattr(node, "type_params", [])
        tp_names = set(_extract_type_param_names(node))
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            for sub in list(tp_list) + [n for n in inner_nodes if n is not None]:
                self.visit(sub)
        finally:
            if tp_names:
                self._type_param_scopes.pop()
        if record_def:
            name_str = getattr(node, "name", "")
            self._bind_names({name_str})
        return tp_names

    def _handle_function_def(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        all_args = (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        )
        sig_nodes: List[Optional[ast.AST]] = [a.annotation for a in all_args]
        sig_nodes.append(node.returns)
        tp_names = self._visit_with_type_params(node, sig_nodes, record_def=True)
        direct = _collect_scope_direct_bindings(
            node.body,
            iter_header_nodes=iter_nested_scope_header_nodes,
            extract_comp_walrus=_extract_comprehension_walrus_names,
        )
        fn_frame = _HelperScopeFrame(
            locals=set(_extract_arg_names(node.args)) | tp_names | set(direct.locals),
            globals=set(direct.globals),
            nonlocals=set(direct.nonlocals),
        )
        self._enter_block_scope(fn_frame, node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_function_def(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        tp_names = self._visit_with_type_params(
            node, list(node.bases) + list(node.keywords), record_def=True
        )
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            self._enter_block_scope(_HelperScopeFrame(is_class=True), node.body)
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_TypeAlias(self, node: ast.AST) -> None:
        alias_name = _extract_type_alias_name(node)
        if alias_name is not None:
            self._bind_names({alias_name})
        self._visit_with_type_params(node, [getattr(node, "value", None)])

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for t in node.targets:
            self.visit(t)
            self._bind_names(_extract_target_bound_names(t))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)
        if node.value is not None or (
            len(self._scope_stack) > 1 and not self._scope_stack[-1].is_class
        ):
            self._bind_names(_extract_target_bound_names(node.target))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._check_load_name(node.target.id)
        self.visit(node.target)
        self.visit(node.value)
        self._bind_names(_extract_target_bound_names(node.target))

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)
        if isinstance(node.target, ast.Name):
            self._bind_names({node.target.id}, is_walrus=True)

    def visit_For(self, node: Union[ast.For, ast.AsyncFor]) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        self._bind_names(_extract_target_bound_names(node.target))
        for s in list(node.body) + list(node.orelse):
            self.visit(s)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: Union[ast.With, ast.AsyncWith]) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
                self._bind_names(_extract_target_bound_names(item.optional_vars))
        for s in node.body:
            self.visit(s)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if isinstance(node.name, str):
            self._bind_names({node.name})
        for s in node.body:
            self.visit(s)

    def _visit_pattern_binding(self, node: ast.AST) -> None:
        self.generic_visit(node)
        bound_name = _extract_pattern_bound_name(node)
        if bound_name is not None:
            self._bind_names({bound_name})

    def visit_MatchAs(self, node: ast.AST) -> None:
        self._visit_pattern_binding(node)

    def visit_MatchStar(self, node: ast.AST) -> None:
        self._visit_pattern_binding(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:
        self._visit_pattern_binding(node)

    def _enter_block_scope(
        self, child_frame: _HelperScopeFrame, body: Sequence[ast.stmt]
    ) -> None:
        is_top_block = len(self._scope_stack) == 1
        self._scope_stack.append(child_frame)
        try:
            for body_stmt in body:
                self.visit(body_stmt)
        finally:
            self._scope_stack.pop()
        if is_top_block:
            self.defined_names.update(child_frame.locals)

    def _handle_comprehension(
        self, elts: Sequence[ast.expr], generators: Sequence[ast.comprehension]
    ) -> None:
        if generators:
            self.visit(generators[0].iter)
        comp_frame = _HelperScopeFrame(is_comp=True)
        self._scope_stack.append(comp_frame)
        try:
            for gen in generators:
                if gen is not generators[0]:
                    self.visit(gen.iter)
                for leaf in _iter_target_leaves(gen.target):
                    if isinstance(leaf, ast.Name) and isinstance(
                        leaf.ctx, ast.Store
                    ):
                        comp_frame.locals.add(leaf.id)
                    else:
                        self.visit(leaf)
                for if_clause in gen.ifs:
                    self.visit(if_clause)
            for e in elts:
                self.visit(e)
        finally:
            self._scope_stack.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._handle_comprehension([node.elt], node.generators)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._handle_comprehension([node.elt], node.generators)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._handle_comprehension([node.elt], node.generators)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._handle_comprehension([node.key, node.value], node.generators)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for d in list(node.args.defaults) + list(node.args.kw_defaults):
            if d is not None:
                self.visit(d)
        self._scope_stack.append(
            _HelperScopeFrame(locals=set(_extract_arg_names(node.args)))
        )
        try:
            self.visit(node.body)
        finally:
            self._scope_stack.pop()

    def _record_import_bindings(
        self, node: Union[ast.Import, ast.ImportFrom]
    ) -> None:
        for name in _extract_import_bound_names(node):
            self._bind_names({name})
            if len(self._scope_stack) <= self._max_helper_scope_depth:
                self.local_imported_names.add(name)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_bindings(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_bindings(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._scope_stack[-1].globals.add(name)
            self.free_names.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._scope_stack[-1].nonlocals.add(name)
            self._check_load_name(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Load, ast.Del)):
            self._check_load_name(node.id)


def _extract_helper_symbols(
    helper_tree: ast.AST,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (free_names, defined_names, helper_local_imported_names) with lexical scope awareness.

    Parameters and local stores from nested functions/classes/comprehensions do not bleed
    into the outer helper scope, preventing false shadowing of module-level dependencies.
    """
    has_top_level_fn = isinstance(
        helper_tree, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) or (
        isinstance(helper_tree, ast.Module)
        and any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            for stmt in getattr(helper_tree, "body", [])
        )
    )
    visitor = _HelperSymbolVisitor(2 if has_top_level_fn else 1)
    try:
        visitor.visit(helper_tree)
    except RecursionError:
        return set(), set(), set()
    return visitor.free_names, visitor.defined_names, visitor.local_imported_names


def _parse_local_import_statements(
    loc_imports: Sequence[str],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    source_idx: Optional[int] = None,
) -> Dict[str, str]:
    stmts: Dict[str, str] = {}

    def _add_target(
        s_item: Any,
        lvl: int,
        mod: Optional[str],
        targets: Set[str],
    ) -> None:
        _, sm, s_pkg = _unpack_source_item(s_item)
        if sm:
            tm = _resolve_relative_import_path(sm, lvl, mod, is_package=s_pkg)
            if tm:
                targets.add(tm)

    for loc_imp in loc_imports:
        s_loc = loc_imp.strip()
        if not s_loc:
            continue
        loc_tree = safe_parse_ast(s_loc)
        if loc_tree is None:
            continue

        for node in loc_tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    stmt = (
                        f"import {alias.name} as {alias.asname}"
                        if alias.asname
                        else f"import {alias.name}"
                    )
                    if name in stmts and stmts[name] != stmt:
                        raise ValueError(
                            f"conflicting local import symbol '{name}' across clone sources"
                        )
                    stmts[name] = stmt
            elif isinstance(node, ast.ImportFrom):
                level = getattr(node, "level", 0) or 0
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    if level == 0 and node.module:
                        stmt = (
                            f"from {node.module} import {alias.name} as {alias.asname}"
                            if alias.asname
                            else f"from {node.module} import {alias.name}"
                        )
                    elif level > 0:
                        resolved_targets: Set[str] = set()
                        if source_idx is not None and 0 <= source_idx < len(source_texts):
                            _add_target(
                                source_texts[source_idx],
                                level,
                                node.module,
                                resolved_targets,
                            )
                        if not resolved_targets and source_texts:
                            matching_items = [
                                item for item in source_texts
                                if bool(_unpack_source_item(item)[0] and s_loc in (_unpack_source_item(item)[0] or ""))
                            ]
                            for item in matching_items or source_texts:
                                _add_target(
                                    item,
                                    level,
                                    node.module,
                                    resolved_targets,
                                )

                        if len(resolved_targets) > 1:
                            raise ValueError(
                                f"conflicting local import symbol '{name}' across clone sources"
                            )
                        if not resolved_targets:
                            continue
                        target_mod = next(iter(resolved_targets))
                        stmt = (
                            f"from {target_mod} import {alias.name} as {alias.asname}"
                            if alias.asname
                            else f"from {target_mod} import {alias.name}"
                        )
                    else:
                        continue

                    if name in stmts and stmts[name] != stmt:
                        raise ValueError(
                            f"conflicting local import symbol '{name}' across clone sources"
                        )
                    stmts[name] = stmt
    return stmts


def _collect_unit_local_import_maps(
    scope: Dict[str, Any],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Extracts per-unit local import maps and merges them, raising on cross-unit conflicts."""
    raw_units_local = scope.get("local_imports_by_unit")
    unit_local_maps: List[Dict[str, str]] = []
    if raw_units_local is not None:
        for u_idx, raw_list in enumerate(raw_units_local):
            s_idx = u_idx if u_idx < len(source_texts) else 0
            unit_local_maps.append(
                _parse_local_import_statements(raw_list, source_texts, source_idx=s_idx)
            )
    else:
        flat_loc = scope.get("local_imports", [])
        if len(source_texts) <= 1:
            unit_local_maps = [
                _parse_local_import_statements(flat_loc, source_texts, source_idx=0)
            ]
        else:
            for s_idx, item in enumerate(source_texts):
                s_text = _unpack_source_item(item)[0] or ""
                attributed = [
                    stmt_str for stmt_str in flat_loc if stmt_str.strip() in s_text
                ]
                if not attributed and flat_loc:
                    attributed = list(flat_loc)
                unit_local_maps.append(
                    _parse_local_import_statements(
                        attributed, source_texts, source_idx=s_idx
                    )
                )

    local_import_stmts: Dict[str, str] = {}
    for u_map in unit_local_maps:
        for name, stmt in u_map.items():
            if name in local_import_stmts and local_import_stmts[name] != stmt:
                raise ValueError(
                    f"conflicting local import symbol '{name}' across clone sources"
                )
            local_import_stmts[name] = stmt
    return unit_local_maps, local_import_stmts


def _resolve_symbol_unit_bindings(
    loaded_name: str,
    unit_local_maps: Sequence[Dict[str, str]],
    source_info: Sequence[Tuple[_ModuleImportsResult, List[str], Set[str]]],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    *,
    is_same_file_host: bool,
) -> List[Tuple[str, Optional[str], Optional[str], int, int]]:
    """Classifies how loaded_name is bound (import/def/wildcard/builtin/unresolved) per unit."""
    num_units = max(len(unit_local_maps), len(source_texts))
    unit_bindings: List[Tuple[str, Optional[str], Optional[str], int, int]] = []
    for u_idx in range(num_units):
        u_loc = unit_local_maps[u_idx] if u_idx < len(unit_local_maps) else {}
        s_idx = u_idx if u_idx < len(source_texts) else 0
        s_res, s_wild, s_defs = source_info[s_idx]
        _, s_mod, _ = _unpack_source_item(source_texts[s_idx])

        if loaded_name in u_loc:
            b_type, b_val = "import", u_loc[loaded_name]
        elif loaded_name in s_res.stmts:
            if loaded_name in s_defs:
                raise ValueError(
                    f"conflicting imported symbol '{loaded_name}' rebound by local definition within source module"
                )
            if loaded_name in s_res.rebound_conflicts:
                raise ValueError(
                    f"conflicting imported symbol '{loaded_name}' rebound within source module"
                )
            if not is_same_file_host and (
                loaded_name in s_res.guarded_names
                and loaded_name not in s_res.unconditional_names
            ):
                raise ValueError(
                    f"symbol '{loaded_name}' relies on runtime-guarded import in source module"
                )
            b_type, b_val = "import", s_res.stmts[loaded_name]
        elif loaded_name in s_defs:
            b_type, b_val = "def", s_mod or f"source_{s_idx}"
        elif bool(s_wild):
            b_type, b_val = "wildcard", "*"
        elif loaded_name in _BUILTIN_NAMES:
            b_type, b_val = "builtin", loaded_name
        else:
            b_type, b_val = "unresolved", None
        unit_bindings.append((b_type, b_val, s_mod, s_idx, u_idx))
    return unit_bindings


def _resolve_agreed_symbol_binding(
    loaded_name: str,
    unit_bindings: Sequence[Tuple[str, Optional[str], Optional[str], int, int]],
    *,
    host_plan: _FilePatchPlan,
    is_same_file_host: bool,
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    source_info: Sequence[Tuple[_ModuleImportsResult, List[str], Set[str]]],
    unit_local_maps: Sequence[Dict[str, str]],
    host_imported: Set[str],
    host_defs: Optional[Set[str]],
    host_mod: Optional[str],
) -> Tuple[Optional[str], Optional[Set[str]]]:
    """Validates cross-unit binding agreement for loaded_name and returns (agreed_stmt, host_defs)."""
    has_wildcard = any(b[0] == "wildcard" for b in unit_bindings)
    has_explicit = any(b[0] in ("import", "def") for b in unit_bindings)
    if has_wildcard and (not is_same_file_host or has_explicit):
        raise ValueError(
            f"symbol '{loaded_name}' potentially relies on wildcard import"
        )
    has_builtin = any(b[0] == "builtin" for b in unit_bindings)
    if has_builtin and has_explicit:
        raise ValueError(
            f"conflicting imported symbol '{loaded_name}' across clone sources"
        )
    has_import = any(b[0] == "import" for b in unit_bindings)
    if host_plan.is_new_file and not has_import:
        raise ValueError(f"unresolved symbol '{loaded_name}' in shared module")
    def_mods = {b[1] for b in unit_bindings if b[0] == "def"}
    if len(def_mods) > 1:
        raise ValueError(
            f"conflicting local definition '{loaded_name}' across clone sources"
        )
    has_unresolved = any(b[0] == "unresolved" for b in unit_bindings)
    if has_import and (has_builtin or has_unresolved):
        raise ValueError(
            f"conflicting imported symbol '{loaded_name}' across clone sources"
        )

    agreed_stmt: Optional[str] = None
    for b_type, b_val, _, _, _ in unit_bindings:
        if b_type == "import" and b_val is not None:
            if def_mods:
                def_mod = next(iter(def_mods))
                if not (bool(def_mod) and b_val.startswith(f"from {def_mod} import ")):
                    raise ValueError(
                        f"conflicting imported symbol '{loaded_name}' across clone sources"
                    )
            if agreed_stmt is None:
                agreed_stmt = b_val
            elif agreed_stmt != b_val:
                raise ValueError(
                    f"conflicting imported symbol '{loaded_name}' across clone sources"
                )
    if agreed_stmt is not None:
        return agreed_stmt, host_defs

    if not is_same_file_host and host_plan.is_new_file:
        raise ValueError(f"unresolved symbol '{loaded_name}' in shared module")
    if host_defs is None:
        host_defs = _extract_module_defined_names(host_plan.orig_text or "")
    if loaded_name not in host_imported and loaded_name not in host_defs:
        raise ValueError(f"unresolved symbol '{loaded_name}' in host module")
    if not is_same_file_host:
        for item, (s_res, _, s_defs) in zip(source_texts, source_info):
            src_text, s_mod, _ = _unpack_source_item(item)
            if (s_mod is not None and host_mod is not None and s_mod == host_mod) or (
                (s_mod is None or host_mod is None) and src_text == host_plan.orig_text
            ):
                continue
            if loaded_name in s_defs:
                raise ValueError(
                    f"conflicting local definition '{loaded_name}' across clone sources"
                )
            if loaded_name not in s_res.stmts and not any(
                loaded_name in u for u in unit_local_maps
            ):
                raise ValueError(
                    f"unresolved symbol '{loaded_name}' in caller module"
                )
    return None, host_defs


def _reconcile_symbol_imports_with_host(
    symbol_to_stmt: Dict[str, str],
    source_hoisted_symbols: Set[str],
    *,
    host_plan: _FilePatchPlan,
    source_texts: Sequence[Union[str, Tuple[Any, ...]]],
    host_mod: Optional[str],
    is_host_pkg: bool,
    host_imported: Set[str],
    host_defs: Optional[Set[str]],
    missing: List[str],
) -> None:
    """Eliminates self-imports, verifies host-module import conflicts, and appends missing statements."""
    if not host_plan.is_new_file:
        if host_defs is None:
            host_defs = _extract_module_defined_names(host_plan.orig_text or "")
        effective_host_mod = host_mod
        if effective_host_mod is None:
            for item in source_texts:
                src_text, s_mod, _ = _unpack_source_item(item)
                if src_text == host_plan.orig_text and s_mod:
                    effective_host_mod = s_mod
                    break
        self_imported = [
            sname
            for sname, stmt in symbol_to_stmt.items()
            if sname in host_defs
            and (
                (
                    bool(effective_host_mod)
                    and (
                        stmt.startswith(f"from {effective_host_mod} import ")
                        or stmt == f"import {effective_host_mod}"
                        or stmt.startswith(f"import {effective_host_mod} as ")
                    )
                )
                or not effective_host_mod
            )
        ]
        for sname in self_imported:
            symbol_to_stmt.pop(sname, None)

    host_existing_res = _extract_module_import_statements(
        host_plan.orig_text or "", source_mod=host_mod, is_package=is_host_pkg
    )
    host_queued_res = _extract_module_import_statements(
        "\n".join(host_plan.missing_imports), source_mod=host_mod, is_package=is_host_pkg
    )
    all_host_stmts: Dict[str, str] = {
        **host_existing_res.stmts,
        **host_queued_res.stmts,
    }
    for hname, hstmt in all_host_stmts.items():
        if hname in symbol_to_stmt:
            if host_defs is not None and hname in host_defs:
                raise ValueError(
                    f"conflicting imported symbol '{hname}' rebound by local definition within host module"
                )
            if hname in host_existing_res.rebound_conflicts:
                raise ValueError(
                    f"conflicting imported symbol '{hname}' rebound within host module"
                )
            if symbol_to_stmt.pop(hname) != hstmt:
                raise ValueError(
                    f"conflicting imported symbol '{hname}' across clone sources"
                )

    existing_and_queued = (
        set(all_host_stmts.values()) | set(host_plan.missing_imports)
    )
    for sname in set(symbol_to_stmt.keys()) & host_imported:
        cand_stmt = symbol_to_stmt.pop(sname)
        if cand_stmt not in existing_and_queued:
            raise ValueError(
                f"conflicting imported symbol '{sname}' with host module"
            )

    for sname, stmt in symbol_to_stmt.items():
        if (
            sname in source_hoisted_symbols
            and stmt not in missing
            and stmt not in existing_and_queued
        ):
            missing.append(stmt)


def _collect_host_missing_imports(
    host_plan: _FilePatchPlan,
    helper_code: str,
    scope: Dict[str, Any],
    source_texts: Sequence[Union[str, Tuple[Any, ...]]] = (),
    host_mod: Optional[str] = None,
    is_host_pkg: bool = False,
) -> List[str]:
    """Computes all required imports for a host plan receiving extracted helper_code.

    Considers typing symbols, scope local imports, and module-level source dependencies
    against the destination host plan's actual content and existing missing_imports.
    """
    host_content = (host_plan.orig_text or "") + "\n" + "\n".join(host_plan.missing_imports)
    host_imported = _get_module_imported_names(host_content, include_conditional=False)
    missing: List[str] = []

    raw_sources = [
        _unpack_source_item(s)[0] for s in source_texts if _unpack_source_item(s)[0]
    ]
    needs_future = any(_has_future_annotations(s) for s in raw_sources) or (
        _helper_requires_future_annotations(helper_code)
    )
    if (
        needs_future
        and not _has_future_annotations(host_plan.orig_text or "")
        and "from __future__ import annotations" not in host_plan.missing_imports
    ):
        missing.append("from __future__ import annotations")

    needed_typing = _extract_required_typing_imports(helper_code)
    missing_typing = [s for s in needed_typing if s not in host_imported]
    if missing_typing:
        missing.append(f"from typing import {', '.join(sorted(missing_typing))}")

    unit_local_maps, local_import_stmts = _collect_unit_local_import_maps(
        scope, source_texts
    )
    helper_tree = safe_parse_ast(helper_code)
    if helper_tree is None or not source_texts:
        return missing

    free_names, defined_in_helper, helper_local_imported_names = (
        _extract_helper_symbols(helper_tree)
    )
    is_same_file_host = (
        not host_plan.is_new_file
        and len(source_texts) == 1
        and _unpack_source_item(source_texts[0])[0] == host_plan.orig_text
    )
    source_info: List[Tuple[_ModuleImportsResult, List[str], Set[str]]] = []
    for item in source_texts:
        src_text, src_mod, src_ispkg = _unpack_source_item(item)
        if not src_text:
            continue
        s_res = _extract_module_import_statements(
            src_text, source_mod=src_mod, is_package=src_ispkg
        )
        source_info.append((s_res, s_res.wildcards, _extract_module_defined_names(src_text)))

    source_bound_names: Set[str] = (
        set(local_import_stmts.keys()) | helper_local_imported_names
    )
    for s_res, _, s_defs in source_info:
        source_bound_names.update(s_res.stmts.keys())
        source_bound_names.update(s_defs)

    ignored_names = (
        (_BUILTIN_NAMES - source_bound_names)
        | defined_in_helper
        | (set(needed_typing) - source_bound_names)
    )
    symbol_to_stmt: Dict[str, str] = dict(local_import_stmts)
    source_hoisted_symbols: Set[str] = set()
    host_defs: Optional[Set[str]] = None

    symbols_to_check = set(free_names) | (helper_local_imported_names & source_bound_names)
    for loaded_name in sorted(symbols_to_check):
        if loaded_name in ignored_names:
            continue
        unit_bindings = _resolve_symbol_unit_bindings(
            loaded_name,
            unit_local_maps,
            source_info,
            source_texts,
            is_same_file_host=is_same_file_host,
        )
        agreed_stmt, host_defs = _resolve_agreed_symbol_binding(
            loaded_name,
            unit_bindings,
            host_plan=host_plan,
            is_same_file_host=is_same_file_host,
            source_texts=source_texts,
            source_info=source_info,
            unit_local_maps=unit_local_maps,
            host_imported=host_imported,
            host_defs=host_defs,
            host_mod=host_mod,
        )
        if agreed_stmt is not None:
            symbol_to_stmt[loaded_name] = agreed_stmt
            if (
                loaded_name not in helper_local_imported_names
                and not (
                    is_same_file_host
                    and any(
                        loaded_name in s_info[0].guarded_names
                        and loaded_name not in s_info[0].unconditional_names
                        for s_info in source_info
                    )
                )
            ):
                source_hoisted_symbols.add(loaded_name)

    _reconcile_symbol_imports_with_host(
        symbol_to_stmt,
        source_hoisted_symbols,
        host_plan=host_plan,
        source_texts=source_texts,
        host_mod=host_mod,
        is_host_pkg=is_host_pkg,
        host_imported=host_imported,
        host_defs=host_defs,
        missing=missing,
    )
    return missing
