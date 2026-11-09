"""Lexical variable scope analysis, AST visitor, and control flow hazard tracking."""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
import logging
import re
import textwrap
import tokenize
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.config import (
    collect_heuristic_ctx_managers,
    normalize_non_suppressing_ctx_managers,
    resolve_pair_closure_flags,
    resolve_unit_closure_flags,
)
from pydoppelgangerhunt.fixer._ast_compat import (
    is_match_as_node,
    is_match_mapping_node,
    is_match_node,
    is_match_star_node,
    is_trystar_node,
    is_type_alias_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _ScopeCacheKey,
    _get_cached_tree_source_lines,
    _get_or_compute_scope_cached,
    _get_or_parse_tree,
    _get_scope_parent_map,
    _register_tree_source_lines,
    _scope_cell_info_cache,
    _suppress_syntax_warnings,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context
from pydoppelgangerhunt.fixer._df_cell_hazards import (
    _UnitClosureHazardContext,
    _evaluate_severed_closure_cell_hazard,
)
from pydoppelgangerhunt.fixer._df_cell_traversal import (
    _find_enclosing_chain_definite_pre_bound,
)
from pydoppelgangerhunt.fixer._df_closures import (  # pylint: disable=unused-import
    _COMPREHENSION_TYPES,
    _extract_comprehension_walrus_names,
    _extract_scope_nonlocals_and_outer_globals,
    _extract_single_node_local_names,
    _extract_strict_fn_local_names,
    _get_tree_scope_intervals,
    _is_irrefutable_case,
    _is_irrefutable_pattern,
    _is_span_in_scope_header,
    _node_earliest_start_pos,
    _node_strictly_encloses_bounds,
    _resolve_immediate_scope_in_fn,
    iter_nested_scope_header_nodes,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _find_enclosing_function_chain,
    _get_span_line_index,
    _has_invalid_lexical_scope_declarations,
    _resolve_unit_col_bounds,
)
from pydoppelgangerhunt.fixer._df_with_provenance import (  # pylint: disable=unused-import
    _EMPTY_SUPPRESS_PROVENANCE,
    _SuppressProvenance,
    _analyze_block_assignment,
    _block_terminates,
    _demote_deletions_to_conditional,
    _extract_deleted_names,
    _seed_scope_suppress_aliases,
    _walrus_assignment_in_expr,
    _walrus_in_sequence,
)
from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names
from pydoppelgangerhunt.fixer._df_visitors import (
    _OuterExprVisitor,
    _UnitLiveInVisitor,
    _is_ast_node_within_unit_bounds,
)
from pydoppelgangerhunt.reporters import (
    extract_unit_source_lines_with_state,
)
from pydoppelgangerhunt.fixer.source import (
    _SUBROUTINE_KINDS,
    _WHOLE_UNIT_KINDS,
    _extract_slice_col_offset,
    _is_valid_eval_expr,
    _load_unit_file_text,
    _slice_unit_token_lines,
    _slice_utf8_line,
    dedent_source_text,
    find_enclosing_function_is_async,
    is_subroutine_unit,
    parse_unit_coord,
    resolve_unit_declared_outputs,
    split_source_lines,
)

__all__ = [
    "BUILTIN_NAMES",
    "SCOPE_SCHEMA_VERSION",
    "analyze_unit_variable_scope",
    "ensure_unit_scope_closure_mode",
    "inspect_single_unit_scope",
    "is_subroutine_unit",
    "scope_matches_closure_mode",
]

logger = logging.getLogger(__name__)

BUILTIN_NAMES: Set[str] = set(dir(builtins))

SCOPE_SCHEMA_VERSION: int = 2

_SYNTHETIC_WRAPPER_NAME: str = "__pdh_wrapper__"


def _is_mangled_name(name: str) -> bool:
    """Checks if an identifier is subject to Python private name mangling (__foo, not __foo__)."""
    return name.startswith("__") and not name.endswith("__") and len(name) > 2


def _unfold_receiver_attribute(
    node: ast.AST,
    receiver_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Unfolds chained attribute access on receiver parameters (e.g. self.config.timeout -> 'self.config.timeout')."""
    parts: List[str] = []
    curr: ast.AST = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    valid_receivers = tuple(receiver_names) if receiver_names is not None else ("self", "cls")
    if isinstance(curr, ast.Name) and curr.id in valid_receivers:
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


def _normalize_receiver_attr_name(
    attr: str,
    receiver_param: Optional[str] = None,
) -> str:
    """Normalizes receiver attribute prefix (e.g. 'self.x', 'this.x', 'cls.x') to canonical '<rec>.x'."""
    pfxs = ["self.", "cls."]
    if receiver_param:
        pfxs.append(f"{receiver_param}.")
    for pfx in pfxs:
        if attr.startswith(pfx):
            return "<rec>." + attr[len(pfx):]
    return attr


def _normalize_receiver_attrs(
    attrs: Sequence[str],
    receiver_param: Optional[str] = None,
) -> Set[str]:
    """Normalizes a collection of receiver attribute strings to canonical '<rec>.*' form."""
    return {_normalize_receiver_attr_name(a, receiver_param) for a in attrs}

def _extract_arg_names(args: ast.arguments) -> Set[str]:
    """Extracts all parameter names from an ast.arguments node."""
    arg_set = {a.arg for a in (args.posonlyargs + args.args + args.kwonlyargs)}
    if args.vararg:
        arg_set.add(args.vararg.arg)
    if args.kwarg:
        arg_set.add(args.kwarg.arg)
    return arg_set


def _extract_type_param_names(node: ast.AST) -> Set[str]:
    """Extracts PEP 695 type parameter names from an AST node."""
    return {
        tp.name
        for tp in getattr(node, "type_params", [])
        if isinstance(getattr(tp, "name", None), str)
    }


def _block_contains_yield(node: ast.AST) -> bool:
    """Returns True if the node contains a Yield or YieldFrom at the current scope level."""
    stack = [node]
    while stack:
        curr = stack.pop()
        for child in ast.iter_child_nodes(curr):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                stack.append(child)
    return False


class _ScopeVisitor(ast.NodeVisitor):

    """Inspects AST loads, stores, parameters, returns, nonlocals, globals, and attrs."""

    def __init__(
        self,
        loop_offset: int = 0,
        source_lines: Optional[Sequence[str]] = None,
        is_subroutine: bool = False,
        receiver_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.loads: List[str] = []
        self.stores: List[str] = []
        self.read_before_write: List[str] = []
        self.params: List[str] = []
        self.param_details: List[Dict[str, Any]] = []
        self.returns: List[str] = []
        self.return_type: Optional[str] = None
        self.nonlocals: List[str] = []
        self.globals: List[str] = []
        self.attrs_read: List[str] = []
        self.attrs_written: List[str] = []
        self.except_vars: Set[str] = set()
        self.deleted_names: Set[str] = set()
        self.type_aliases: Set[str] = set()
        self.type_params: List[str] = []
        self.type_param_free_vars: List[str] = []
        self._scope_stack: List[Set[str]] = []
        self._type_param_scopes: List[Set[str]] = []
        self._comprehension_depth: int = 0
        self.class_depth: int = 0
        self.loop_offset: int = loop_offset
        self.loop_depth: int = 0
        self.has_return: bool = False
        self.has_return_value: bool = False
        self.has_yield: bool = False
        self.has_yield_assignment: bool = False
        self.has_try_or_with: bool = False
        self._ancestors: List[ast.AST] = []
        self.has_super: bool = False
        self.has_mangled_names: bool = False
        self.naked_breaks: int = 0
        self.naked_continues: int = 0
        self.local_imports: List[str] = []
        self.imported_names: Dict[str, str] = {}
        self.yield_expr_names: List[Tuple[str, str]] = []
        self.is_async: bool = False
        self.is_subroutine: bool = is_subroutine
        self._has_processed_top_func: bool = False
        self.source_lines: Optional[List[str]] = list(source_lines) if source_lines is not None else None
        self.receiver_names: Tuple[str, ...] = (
            tuple(dict.fromkeys(list(receiver_names) + ["self", "cls"]))
            if receiver_names is not None
            else ("self", "cls")
        )
        self.unit_top_locals: Set[str] = set()
        self._unit_top_stored_locals: Set[str] = set()

    def _init_unit_top_locals(self, target_node: ast.AST) -> None:
        self.unit_top_locals = _extract_strict_fn_local_names(target_node)
        self._unit_top_stored_locals = _extract_single_node_local_names(
            target_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
            include_bare_annotations=False,
        )

    def visit_Module(self, node: ast.Module) -> None:
        if self.is_subroutine and not self.unit_top_locals:
            if (
                len(node.body) == 1
                and isinstance(node.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body[0].name == _SYNTHETIC_WRAPPER_NAME
            ):
                wrapper_body = node.body[0].body
                if (
                    self.loop_offset > 0
                    and len(wrapper_body) == 1
                    and isinstance(wrapper_body[0], (ast.For, ast.AsyncFor, ast.While))
                ):
                    wrapper_body = wrapper_body[0].body
                target_mod = ast.Module(body=wrapper_body, type_ignores=[])
            else:
                target_mod = node
            self._init_unit_top_locals(target_mod)
        self.generic_visit(node)

    def visit(self, node: ast.AST) -> Any:
        self._ancestors.append(node)
        try:
            return super().visit(node)
        finally:
            self._ancestors.pop()

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
            except Exception as exc:
                logger.debug("Failed to unparse argument annotation %r: %s", arg_node.annotation, exc)
                type_val = None

        default_val = None
        if default_node is not None:
            try:
                default_val = ast.unparse(default_node)
            except Exception as exc:
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
            except Exception as exc:
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

    _extract_arg_names = staticmethod(_extract_arg_names)


    def _record_store_name(self, name: str, comp_depth: int = 0) -> None:
        if self._comprehension_depth > 0 and comp_depth == 0:
            self._scope_stack[-1].add(name)
            return
        eff_len = len(self._scope_stack) - comp_depth
        bound_in_intermediate_fn = eff_len > 2 and any(
            name in s
            for s in self._scope_stack[1 : -1 - comp_depth]
            if ":class_scope:" not in s
        )
        is_active_nonlocal = (
            name in self.nonlocals
            if eff_len <= 1
            else (
                f"nonlocal:{name}" in self._scope_stack[-1 - comp_depth]
                and not bound_in_intermediate_fn
            )
        )
        is_active_global = (
            name in self.globals
            if eff_len <= 1
            else f"global:{name}" in self._scope_stack[-1 - comp_depth]
        )
        if eff_len <= 1 or is_active_nonlocal:
            self.deleted_names.discard(name)
        if is_active_nonlocal:
            if (
                name not in BUILTIN_NAMES
                and name not in self.stores
                and name not in self._unit_top_stored_locals
            ):
                self.stores.append(name)
        elif is_active_global:
            pass
        elif eff_len > 1:
            target_frame = self._scope_stack[-1 - comp_depth]
            if (
                f"nonlocal:{name}" not in target_frame
                and not (comp_depth > 0 and ":class_scope:" in target_frame)
            ):
                target_frame.add(name)
        elif name not in BUILTIN_NAMES and name not in self.stores:
            self.stores.append(name)

    def _process_func(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        if (
            self.is_subroutine
            and len(self._scope_stack) == 0
            and node.name == _SYNTHETIC_WRAPPER_NAME
        ):
            body = node.body
            if (
                self.loop_offset > 0
                and len(body) == 1
                and isinstance(body[0], (ast.For, ast.AsyncFor, ast.While))
            ):
                body = body[0].body
                self.loop_offset = 0
            if not self.unit_top_locals:
                self._init_unit_top_locals(ast.Module(body=body, type_ignores=[]))
            if isinstance(node, ast.AsyncFunctionDef):
                self.is_async = True
            for stmt in body:
                self.visit(stmt)
            return

        tp_names = _extract_type_param_names(node)
        is_top = (
            len(self._scope_stack) == 0
            and not self.is_subroutine
            and not self._has_processed_top_func
        )
        created_outer = False
        if is_top:
            self._has_processed_top_func = True
            for tp in getattr(node, "type_params", []):
                try:
                    self.type_params.append(ast.unparse(tp))
                except Exception as exc:
                    logger.debug("Failed to unparse type_param %r: %s", tp, exc)
                tp_visitor = _OuterExprVisitor()
                tp_visitor.visit(tp)
                for name in tp_visitor.ordered_names:
                    if (
                        name not in tp_names
                        and name not in BUILTIN_NAMES
                        and name not in self.type_param_free_vars
                    ):
                        self.type_param_free_vars.append(name)
            self._process_args(node)
        else:
            for dec in node.decorator_list:
                self.visit(dec)
            for default in node.args.defaults:
                self.visit(default)
            for kw_default in node.args.kw_defaults:
                if kw_default is not None:
                    self.visit(kw_default)

        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            if is_top:
                self._init_unit_top_locals(node)
                self._scope_stack.append(
                    set(self.params) | tp_names | self.unit_top_locals
                )
            else:
                for tp in getattr(node, "type_params", []):
                    self.visit(tp)
                for arg in (
                    list(getattr(node.args, "posonlyargs", []))
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                    + ([node.args.vararg] if node.args.vararg else [])
                    + ([node.args.kwarg] if node.args.kwarg else [])
                ):
                    if arg.annotation is not None:
                        self.visit(arg.annotation)
                if node.returns is not None:
                    self.visit(node.returns)
                self._record_store_name(node.name)
                if not self._scope_stack:
                    self._scope_stack.append(set(self.unit_top_locals))
                    created_outer = True
                nested_locals = _extract_single_node_local_names(
                    node,
                    include_globals=False,
                    include_nonlocals=False,
                    include_self_name=False,
                )
                self._scope_stack.append(
                    self._extract_arg_names(node.args) | tp_names | nested_locals
                )
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self._scope_stack.pop()
                if created_outer:
                    self._scope_stack.pop()
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self.visit(kw_default)
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set(self.unit_top_locals))
            created_outer = True
        lambda_locals = _extract_single_node_local_names(
            node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
        )
        self._scope_stack.append(self._extract_arg_names(node.args) | lambda_locals)
        saved_comp_depth = self._comprehension_depth
        self._comprehension_depth = 0
        try:
            self.visit(node.body)
        finally:
            self._comprehension_depth = saved_comp_depth
            self._scope_stack.pop()
            if created_outer:
                self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if len(self._scope_stack) == 0 and not self.is_subroutine:
            self.is_async = True
        self._process_func(node)

    def visit_Await(self, node: ast.Await) -> None:
        if (len(self._scope_stack) - self._comprehension_depth) <= 1:
            self.is_async = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        tp_names = _extract_type_param_names(node)
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            for tp in getattr(node, "type_params", []):
                self.visit(tp)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword)
            self._record_store_name(node.name)
            created_outer = False
            if not self._scope_stack:
                self._scope_stack.append(set(self.unit_top_locals))
                created_outer = True
            self._scope_stack.append({":class_scope:", *tp_names})
            self.class_depth += 1
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self.class_depth -= 1
                self._scope_stack.pop()
                if created_outer:
                    self._scope_stack.pop()
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_TypeAlias(self, node: ast.AST) -> None:  # pragma: no cover (py312+)
        name_node = getattr(node, "name", None)
        alias_name = (
            name_node.id
            if isinstance(name_node, ast.Name)
            else (name_node if isinstance(name_node, str) else None)
        )
        if isinstance(alias_name, str):
            is_nested = (
                len(self._scope_stack) > 0
                if self.is_subroutine
                else len(self._scope_stack) > 1
            )
            if not is_nested:
                self.type_aliases.add(alias_name)
            self._record_store_name(alias_name)
        tp_names = _extract_type_param_names(node)
        if tp_names:
            self._type_param_scopes.append(tp_names)
        try:
            for tp in getattr(node, "type_params", []):
                self.visit(tp)
            val_node = getattr(node, "value", None)
            if isinstance(val_node, ast.AST):
                self.visit(val_node)
        finally:
            if tp_names:
                self._type_param_scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.class_depth == 0:
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "super"
                and "super" not in self.stores
                and "super" not in self.params
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "super"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "builtins"
            ):
                self.has_super = True
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        name = getattr(node, "name", None)
        if name and isinstance(name, str) and name != "_":
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        name = getattr(node, "name", None)
        if name and isinstance(name, str) and name != "_":
            self._record_store_name(name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node: ast.AST) -> None:  # pragma: no cover (py310+)
        rest = getattr(node, "rest", None)
        if rest and isinstance(rest, str) and rest != "_":
            self._record_store_name(rest)
        self.generic_visit(node)

    def _deferred_scope_depth(self) -> int:
        return (
            sum(1 for s in self._scope_stack if ":class_scope:" not in s)
            - self._comprehension_depth
        )

    def _record_load_name(self, name: str) -> None:
        is_inner = any(
            name in s
            for s in self._scope_stack[1:]
            if ":class_scope:" not in s or s is self._scope_stack[-1]
        ) or any(name in s for s in self._type_param_scopes)
        if not is_inner and name not in BUILTIN_NAMES:
            if name not in self.loads:
                self.loads.append(name)
            if (
                self._deferred_scope_depth() <= 1
                and name not in self.params
                and name not in self.stores
                and name not in self.read_before_write
            ):
                self.read_before_write.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.id):
            self.has_mangled_names = True
        if isinstance(node.ctx, ast.Load):
            self._record_load_name(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._record_store_name(node.id)
        elif isinstance(node.ctx, ast.Del):
            if (
                self._deferred_scope_depth() <= 1
                and self.class_depth == 0
            ):
                self._record_load_name(node.id)
            if len(self._scope_stack) <= 1 or (
                f"nonlocal:{node.id}" in self._scope_stack[-1]
                and not any(
                    node.id in s
                    for s in self._scope_stack[1:-1]
                    if ":class_scope:" not in s
                )
            ):
                self.deleted_names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.class_depth == 0 and _is_mangled_name(node.attr):
            self.has_mangled_names = True
        attr_name = _unfold_receiver_attribute(node, receiver_names=self.receiver_names)
        if attr_name is not None:
            receiver_id = attr_name.split(".", 1)[0]
            is_inner = any(
                receiver_id in s
                for s in self._scope_stack[1:]
                if ":class_scope:" not in s or s is self._scope_stack[-1]
            )
            if not is_inner and self.class_depth == 0:
                if isinstance(node.ctx, ast.Load):
                    if attr_name not in self.attrs_read:
                        self.attrs_read.append(attr_name)
                elif isinstance(node.ctx, (ast.Store, ast.Del)):
                    if attr_name not in self.attrs_written:
                        self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name and isinstance(node.name, str):
            self.except_vars.add(node.name)
            self._record_store_name(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # AugAssign both loads and stores the target
        if isinstance(node.target, ast.Name):
            self._record_load_name(node.target.id)
            self._record_store_name(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            attr_name = _unfold_receiver_attribute(node.target, receiver_names=self.receiver_names)
            if attr_name is not None:
                receiver_id = attr_name.split(".", 1)[0]
                is_inner = any(
                    receiver_id in s
                    for s in self._scope_stack[1:]
                    if ":class_scope:" not in s or s is self._scope_stack[-1]
                )
                if not is_inner and self.class_depth == 0:
                    if attr_name not in self.attrs_read:
                        self.attrs_read.append(attr_name)
                    if attr_name not in self.attrs_written:
                        self.attrs_written.append(attr_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if isinstance(node.target, ast.Name):
                self._record_store_name(node.target.id)
            else:
                self.visit(node.target)
        elif not isinstance(node.target, ast.Name):
            self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._record_store_name(
                node.target.id, comp_depth=self._comprehension_depth
            )

    def _visit_comprehension(
        self,
        node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp],
    ) -> None:
        if node.generators:
            self.visit(node.generators[0].iter)
        self._comprehension_depth += 1
        created_outer = False
        if not self._scope_stack:
            self._scope_stack.append(set(self.unit_top_locals))
            created_outer = True
        inner_vars: Set[str] = set()
        self._scope_stack.append(inner_vars)
        if (
            any(getattr(g, "is_async", False) for g in node.generators)
            and (len(self._scope_stack) - self._comprehension_depth) <= 1
        ):
            self.is_async = True
        try:
            for gen in node.generators:
                if gen is not node.generators[0]:
                    self.visit(gen.iter)
                for sub in (gen.target, *gen.ifs):
                    self.visit(sub)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self._scope_stack.pop()
            if created_outer:
                self._scope_stack.pop()
            self._comprehension_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        eff_len = len(self._scope_stack) - self._comprehension_depth
        enclosing_frames = self._scope_stack[: -1 - self._comprehension_depth]
        for name in node.names:
            is_unit_bound = (
                any(
                    name in s and ":class_scope:" not in s
                    for s in enclosing_frames
                )
                or name in self.stores
                or name in self.params
                or name in self.unit_top_locals
            )
            if (eff_len <= 1 or not is_unit_bound) and name not in self.nonlocals:
                self.nonlocals.append(name)
            if eff_len > 1:
                self._scope_stack[-1 - self._comprehension_depth].add(
                    f"nonlocal:{name}"
                )
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        eff_len = len(self._scope_stack) - self._comprehension_depth
        for name in node.names:
            if eff_len <= 1:
                if name not in self.globals:
                    self.globals.append(name)
            else:
                self._scope_stack[-1 - self._comprehension_depth].add(
                    f"global:{name}"
                )
        self.generic_visit(node)

    def _visit_loop(self, node: Union[ast.For, ast.AsyncFor, ast.While]) -> None:
        self.loop_depth += 1
        try:
            if isinstance(node, (ast.For, ast.AsyncFor)):
                self.visit(node.iter)
                self.visit(node.target)
            else:
                self.visit(node.test)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)
        finally:
            self.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
        self._visit_loop(node)

    def visit_With(self, node: ast.With) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        if len(self._scope_stack) <= 1:
            self.is_async = True
            if _block_contains_yield(node):
                self.has_try_or_with = True
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.AST) -> None:
        if len(self._scope_stack) <= 1 and _block_contains_yield(node):
            self.has_try_or_with = True
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
        if len(self._scope_stack) <= 1:
            self.has_yield = True
            if node.value is not None:
                if isinstance(node.value, ast.Name):
                    self.yield_expr_names.append((kind, node.value.id))
                elif isinstance(node.value, ast.Constant) and node.value.value is not None:
                    self.yield_expr_names.append((kind, f":literal:{type(node.value.value).__name__}"))
                elif (
                    kind == "yield_from"
                    and isinstance(node.value, (ast.List, ast.Tuple, ast.Set))
                    and node.value.elts
                ):
                    first_elt = node.value.elts[0]
                    if isinstance(first_elt, ast.Constant) and first_elt.value is not None:
                        t_name = type(first_elt.value).__name__
                        if all(
                            isinstance(e, ast.Constant)
                            and e.value is not None
                            and type(e.value).__name__ == t_name
                            for e in node.value.elts
                        ):
                            self.yield_expr_names.append((kind, f":literal:{t_name}"))

    def _inspect_yield_assignment(self) -> None:
        if len(self._scope_stack) <= 1 and len(self._ancestors) >= 2:
            immediate_parent = self._ancestors[-2]
            if not isinstance(immediate_parent, ast.Expr):
                self.has_yield_assignment = True

    def visit_Yield(self, node: ast.Yield) -> None:
        self._record_yield_expr("yield", node)
        if len(self._scope_stack) <= 1:
            self._inspect_yield_assignment()
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._record_yield_expr("yield_from", node)
        self.generic_visit(node)


    def visit_Return(self, node: ast.Return) -> None:
        if len(self._scope_stack) <= 1:
            self.has_return = True
            if node.value is not None:
                self.has_return_value = True
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
            except Exception as exc:
                logger.debug("Failed to unparse import node %r: %s", node, exc)
                return

        is_nested = (
            len(self._scope_stack) > 0
            if self.is_subroutine
            else len(self._scope_stack) > 1
        )
        if not is_nested and stmt not in self.local_imports:
            self.local_imports.append(stmt)
        for alias in node.names:
            if isinstance(node, ast.Import):
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            else:
                bound_name = alias.asname or alias.name
            if bound_name != "*":
                if not is_nested:
                    self.imported_names[bound_name] = stmt
                self._record_store_name(bound_name)

    def visit_Import(self, node: ast.Import) -> None:
        self._record_import_node(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record_import_node(node)
        self.generic_visit(node)


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
    primary_receiver: Optional[str] = None,
) -> None:
    """Ensures self or cls is positioned as the primary receiver parameter in inputs."""
    def _bring_to_front(name: str) -> None:
        if name in inputs:
            inputs.remove(name)
            inputs.insert(0, name)

    if has_instance_binding and has_class_binding:
        first = primary_receiver or "self"
        second = "cls" if first == "self" else "self"
        _bring_to_front(second)
        _bring_to_front(first)
    elif has_instance_binding or has_class_binding:
        default_rec = "self" if has_instance_binding else "cls"
        target = primary_receiver if (primary_receiver and primary_receiver in inputs) else default_rec
        _bring_to_front(target)



def _rank_param_kind(
    var_name: str,
    param_map: Dict[str, str],
    receiver_names: Optional[Sequence[str]] = None,
) -> int:
    """Returns canonical parameter sorting rank: receiver=0, pos=1, vararg=2, kwonly=3, kwarg=4."""
    clean = var_name.lstrip("*")
    valid_receivers = (
        tuple(receiver_names)
        if receiver_names is not None
        else ("self", "cls", "this", "klass")
    )
    if clean in valid_receivers:
        return 0
    kind = param_map.get(clean, "pos")
    if var_name.startswith("**") or kind == "kwarg":
        return 4
    if kind == "kwonly":
        return 3
    if var_name.startswith("*") or kind == "vararg":
        return 2
    return 1


def _find_enclosing_pre_bound_names(
    mod_tree: ast.AST,
    s_line: int,
    e_line: int,
    u_start_col: Optional[int] = None,
    u_end_col: Optional[int] = None,
) -> Tuple[Set[str], bool]:
    """Finds names may-bound before s_line in enclosing module/class/function scopes.

    Unlike ``_find_enclosing_chain_definite_pre_bound`` (which computes must-bound
    definite assignment across every control-flow path in enclosing functions), this
    function collects all names that have a lexical binding on at least one path prior
    to the unit across module, class, and function scopes, and reports whether the unit
    is enclosed inside a loop. Comparing the may-bound set against the must-bound set
    identifies variables that are only conditionally initialized before the unit.
    """
    bound_before: Set[str] = set()
    in_loop = False

    def _is_before_unit(c_line: Optional[int], c_col: Optional[int]) -> bool:
        if not isinstance(c_line, int):
            return False
        if c_line < s_line:
            return True
        return (
            c_line == s_line
            and u_start_col is not None
            and isinstance(c_col, int)
            and c_col < u_start_col
        )

    def _node_contains_or_encloses_unit(node: ast.AST) -> bool:
        return _is_ast_node_within_unit_bounds(
            node,
            s_line,
            e_line,
            u_end_col=u_end_col,
            u_start_col=u_start_col,
        ) or _node_strictly_encloses_bounds(
            node,
            s_line,
            e_line,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )

    intervals = [
        (earliest_start, n_end, span, node)
        for earliest_start, n_end, span, node in _get_tree_scope_intervals(mod_tree)
        if earliest_start <= s_line
        and e_line <= n_end
        and (
            _node_strictly_encloses_bounds(
                node,
                s_line,
                e_line,
                u_start_col=u_start_col,
                u_end_col=u_end_col,
            )
            or (
                u_start_col is None
                and u_end_col is None
                and (
                    earliest_start < s_line
                    or n_end > e_line
                    or earliest_start == n_end
                )
            )
        )
        and not _is_span_in_scope_header(
            node,
            s_line,
            e_line,
            u_start_col=u_start_col,
            u_end_col=u_end_col,
        )
    ]
    intervals.sort(key=lambda item: (item[0], -item[1], -item[2]))
    innermost_scope = intervals[-1][3] if intervals else mod_tree
    active_scopes: List[ast.AST] = []
    if isinstance(mod_tree, ast.Module):
        active_scopes.append(mod_tree)
    for _estart, _nend, _span, s_node in intervals:
        if isinstance(s_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            active_scopes.append(s_node)
        elif isinstance(s_node, ast.ClassDef) and s_node is innermost_scope:
            active_scopes.append(s_node)

    for scope_node in active_scopes:
        is_class_scope = isinstance(scope_node, ast.ClassDef)
        if isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound_before.difference_update(
                _extract_strict_fn_local_names(scope_node)
            )
            for arg in (
                scope_node.args.posonlyargs
                + scope_node.args.args
                + scope_node.args.kwonlyargs
                + ([scope_node.args.vararg] if scope_node.args.vararg else [])
                + ([scope_node.args.kwarg] if scope_node.args.kwarg else [])
            ):
                bound_before.add(arg.arg)
        stack: List[Tuple[ast.AST, bool]] = [
            (stmt, False) for stmt in reversed(getattr(scope_node, "body", []))
        ]
        while stack:
            curr, is_branched = stack.pop()
            c_start = getattr(curr, "lineno", None)
            c_col = getattr(curr, "col_offset", None)
            c_end = getattr(curr, "end_lineno", c_start)
            c_end_col = getattr(curr, "end_col_offset", c_col)
            if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                estart, ecol = _node_earliest_start_pos(curr)
                if _is_before_unit(estart, ecol) and (
                    (
                        curr in active_scopes
                        and not isinstance(curr, ast.ClassDef)
                    )
                    or _is_before_unit(c_end, c_end_col)
                ):
                    bound_before.add(curr.name)
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(curr, ast.Lambda):
                if not is_class_scope:
                    stack.extend(
                        (hdr, is_branched)
                        for hdr in reversed(list(iter_nested_scope_header_nodes(curr)))
                    )
                continue
            if isinstance(
                curr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
            ):
                if _is_before_unit(c_start, c_col) and not is_class_scope:
                    bound_before.update(_extract_comprehension_walrus_names(curr))
                continue
            if is_type_alias_node(curr):
                if _is_before_unit(c_start, c_col):
                    alias_name = getattr(curr, "name", None)
                    if isinstance(alias_name, ast.Name):
                        bound_before.add(alias_name.id)
                continue
            if (
                isinstance(curr, ast.AnnAssign)
                and curr.value is None
                and isinstance(curr.target, ast.Name)
            ):
                stack.append((curr.annotation, is_branched))
                continue
            if isinstance(
                curr,
                (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr),
            ):
                rhs_val = curr.value
                if rhs_val is not None and _node_contains_or_encloses_unit(
                    rhs_val
                ):
                    stack.append((rhs_val, is_branched))
                    if isinstance(curr, ast.AnnAssign):
                        stack.append((curr.annotation, is_branched))
                    elif isinstance(curr, ast.AugAssign) and not isinstance(
                        curr.target, ast.Name
                    ):
                        stack.append((curr.target, is_branched))
                    continue
            if isinstance(curr, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(
                    curr, (ast.For, ast.AsyncFor)
                ) and _node_contains_or_encloses_unit(curr.iter):
                    stack.append((curr.iter, is_branched))
                    continue
                if _node_strictly_encloses_bounds(
                    curr,
                    s_line,
                    e_line,
                    u_start_col=u_start_col,
                    u_end_col=u_end_col,
                ):
                    in_loop = True
            if _is_before_unit(c_start, c_col):
                if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Store):
                    bound_before.add(curr.id)
                elif (
                    isinstance(curr, ast.Name)
                    and isinstance(curr.ctx, ast.Del)
                    and not is_branched
                ):
                    bound_before.discard(curr.id)
                elif isinstance(curr, (ast.Import, ast.ImportFrom)):
                    for alias in curr.names:
                        b_name = alias.asname or alias.name.split(".", 1)[0]
                        if b_name != "*":
                            bound_before.add(b_name)
                elif (
                    isinstance(curr, ast.ExceptHandler)
                    and isinstance(curr.name, str)
                    and _node_strictly_encloses_bounds(
                        curr,
                        s_line,
                        e_line,
                        u_start_col=u_start_col,
                        u_end_col=u_end_col,
                    )
                ):
                    bound_before.add(curr.name)
                elif (
                    is_match_as_node(curr)
                    or is_match_star_node(curr)
                    or is_match_mapping_node(curr)
                ):
                    m_name = getattr(curr, "name", None) or getattr(curr, "rest", None)
                    if isinstance(m_name, str) and m_name != "_":
                        bound_before.add(m_name)
            child_branched = is_branched or (
                isinstance(
                    curr,
                    (
                        ast.If,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.Try,
                        ast.ExceptHandler,
                    ),
                )
                or is_trystar_node(curr)
                or is_match_node(curr)
            )
            stack.extend(
                (ch, child_branched)
                for ch in reversed(list(ast.iter_child_nodes(curr)))
            )
    return bound_before, in_loop


def _infer_missing_col_from_line_ast(
    lines: Union[str, Sequence[str]],
    s_col: Optional[int],
    e_col: Optional[int],
    unit_kind: Optional[str] = None,
    *,
    allow_expr_fallback: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Infers a missing start_col or end_col from single-line or multiline AST nodes."""
    line_seq: Sequence[str] = (lines,) if isinstance(lines, str) else lines
    if not line_seq:
        return s_col, e_col
    n_lines = len(line_seq)
    if n_lines == 1:
        line0 = line_seq[0]
        stripped = line0.lstrip()
        if not stripped:
            return s_col, e_col
        pad_first = len(line0[: len(line0) - len(stripped)].encode("utf-8"))
        pad_last = pad_first
        parse_src = stripped
    else:
        raw_block = "".join(
            ln if ln.endswith("\n") else ln + "\n" for ln in line_seq
        )
        dedented = dedent_source_text(raw_block)
        d_lines = split_source_lines(dedented)
        if len(d_lines) != n_lines:
            return s_col, e_col
        pad_first = len(line_seq[0].rstrip("\r\n").encode("utf-8")) - len(
            d_lines[0].rstrip("\r\n").encode("utf-8")
        )
        pad_last = len(line_seq[-1].rstrip("\r\n").encode("utf-8")) - len(
            d_lines[-1].rstrip("\r\n").encode("utf-8")
        )
        parse_src = dedented
    try:
        with _suppress_syntax_warnings():
            mod = ast.parse(parse_src)
    except SyntaxError:
        return s_col, e_col
    if unit_kind == "comprehension":
        candidate_groups: Tuple[Tuple[type, ...], ...] = (_COMPREHENSION_TYPES,)
    elif unit_kind == "complex_expr":
        candidate_groups = ((ast.expr,),)
    elif allow_expr_fallback:
        candidate_groups = ((ast.stmt,), (ast.expr,))
    else:
        candidate_groups = ((ast.stmt,),)
    for target_types in candidate_groups:
        matched_spans: Set[Tuple[int, int]] = set()
        for node in ast.walk(mod):
            if not isinstance(node, target_types) or isinstance(
                node, (ast.Name, ast.Constant)
            ):
                continue
            if (
                getattr(node, "lineno", None) != 1
                or getattr(node, "end_lineno", None) != n_lines
            ):
                continue
            nc = getattr(node, "col_offset", None)
            nec = getattr(node, "end_col_offset", None)
            if not isinstance(nc, int) or not isinstance(nec, int):
                continue
            abs_s = pad_first + nc
            abs_e = pad_last + nec
            if (n_lines > 1 or abs_s < abs_e) and (
                (s_col is not None and abs_s == s_col)
                or (e_col is not None and abs_e == e_col)
            ):
                matched_spans.add((abs_s, abs_e))
        if len(matched_spans) == 1:
            return next(iter(matched_spans))
        if len(matched_spans) > 1:
            return s_col, e_col
    return s_col, e_col


def _unit_cols(
    unit: Dict[str, Any],
    unit_code_lines: Optional[Sequence[str]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Extracts (start_col, end_col) coordinates from a unit dictionary."""
    s_col = parse_unit_coord(unit, "start_col", default=None)
    if s_col is None:
        s_col = parse_unit_coord(unit, "col_offset", default=None)
    e_col = parse_unit_coord(unit, "end_col", default=None)
    if e_col is None:
        e_col = parse_unit_coord(unit, "end_col_offset", default=None)
    u_kind_str = str(unit.get("kind") or "")
    is_inline_expr_kind = u_kind_str in (
        "comprehension",
        "complex_expr",
        "closure",
    )
    if (
        (s_col is None or e_col is None)
        and unit_code_lines
        and isinstance(unit.get("source_code"), str)
    ):
        snippet = unit["source_code"].strip()
        if len(unit_code_lines) == 1 and snippet and "\n" not in snippet:
            line0 = unit_code_lines[0]
            if (
                s_col is not None
                or e_col is not None
                or (
                    snippet != line0.strip()
                    and (";" in line0 or is_inline_expr_kind)
                )
            ):
                start_id = snippet[0].isalnum() or snippet[0] == "_"
                end_id = snippet[-1].isalnum() or snippet[-1] == "_"
                search_idx = 0
                exact_matches: List[Tuple[int, int]] = []
                while True:
                    pos = line0.find(snippet, search_idx)
                    if pos < 0:
                        break
                    end_pos = pos + len(snippet)
                    left_ok = not start_id or pos == 0 or not (
                        line0[pos - 1].isalnum() or line0[pos - 1] == "_"
                    )
                    right_ok = not end_id or end_pos >= len(line0) or not (
                        line0[end_pos].isalnum() or line0[end_pos] == "_"
                    )
                    if left_ok and right_ok:
                        cand_s = len(line0[:pos].encode("utf-8"))
                        cand_e = len(line0[:end_pos].encode("utf-8"))
                        if (s_col is None or cand_s == s_col) and (
                            e_col is None or cand_e == e_col
                        ):
                            exact_matches.append((cand_s, cand_e))
                    search_idx = pos + 1
                if len(exact_matches) == 1:
                    best_pair = exact_matches[0]
                    if s_col is None:
                        s_col = best_pair[0]
                    if e_col is None:
                        e_col = best_pair[1]
        elif len(unit_code_lines) > 1 and snippet and "\n" in snippet:
            snip_lines = [
                ln.rstrip("\r\n") for ln in split_source_lines(snippet)
            ]
            if (
                len(snip_lines) == len(unit_code_lines)
                and snip_lines[0]
                and snip_lines[-1]
                and all(
                    snip_lines[i].strip() == unit_code_lines[i].strip()
                    for i in range(1, len(snip_lines) - 1)
                )
            ):
                had_explicit_col = s_col is not None or e_col is not None
                line_first = unit_code_lines[0].rstrip("\r\n")
                line_last = unit_code_lines[-1].rstrip("\r\n")
                if (
                    had_explicit_col
                    or ";" in line_first
                    or ";" in line_last
                    or is_inline_expr_kind
                ):
                    ml_cand_s: Optional[int] = s_col
                    ml_cand_e: Optional[int] = e_col
                    snip_first = snip_lines[0]
                    if ml_cand_s is None:
                        start_id = (
                            snip_first[0].isalnum() or snip_first[0] == "_"
                        )
                        s_matches: List[int] = []
                        idx = 0
                        while True:
                            pos = line_first.find(snip_first, idx)
                            if pos < 0:
                                break
                            tail = line_first[
                                pos + len(snip_first) :
                            ].strip()
                            left_ok = not start_id or pos == 0 or not (
                                line_first[pos - 1].isalnum()
                                or line_first[pos - 1] == "_"
                            )
                            if left_ok and (not tail or tail.startswith("#")):
                                s_matches.append(
                                    len(line_first[:pos].encode("utf-8"))
                                )
                            idx = pos + 1
                        if len(s_matches) == 1:
                            ml_cand_s = s_matches[0]
                    snip_last = snip_lines[-1].lstrip()
                    if ml_cand_e is None and snip_last:
                        end_id = (
                            snip_last[-1].isalnum() or snip_last[-1] == "_"
                        )
                        l_stripped = line_last.lstrip()
                        if l_stripped.startswith(snip_last):
                            end_pos = (
                                len(line_last)
                                - len(l_stripped)
                                + len(snip_last)
                            )
                            right_ok = (
                                not end_id
                                or end_pos >= len(line_last)
                                or not (
                                    line_last[end_pos].isalnum()
                                    or line_last[end_pos] == "_"
                                )
                            )
                            if right_ok:
                                ml_cand_e = len(
                                    line_last[:end_pos].encode("utf-8")
                                )
                    if had_explicit_col or (
                        ml_cand_s is not None and ml_cand_e is not None
                    ):
                        s_col, e_col = ml_cand_s, ml_cand_e
    if (s_col is None) != (e_col is None) and unit_code_lines:
        s_col, e_col = _infer_missing_col_from_line_ast(
            unit_code_lines,
            s_col,
            e_col,
            unit_kind=u_kind_str,
            allow_expr_fallback=is_inline_expr_kind,
        )
    return s_col, e_col


def _is_word_or_attr_char(ch: str) -> bool:
    """Returns True if ch is an identifier character or attribute dot."""
    return ch.isalnum() or ch in ("_", ".")


def _has_token_boundary_subslice(
    sub: str,
    line: str,
    *,
    anchor_start: bool = False,
    anchor_end: bool = False,
    require_stmt_delimiters: bool = False,
) -> bool:
    """Checks whether sub occurs in line at valid lexical token boundaries."""
    if not sub:
        return False
    if sub == line:
        return True
    search_start = 0
    while True:
        pos = line.find(sub, search_start)
        if pos == -1:
            return False
        search_start = pos + 1
        end_pos = pos + len(sub)
        if anchor_start and pos != 0:
            continue
        if anchor_end and end_pos != len(line):
            continue
        if pos > 0 and (sub[0].isalnum() or sub[0] == "_"):
            if _is_word_or_attr_char(line[pos - 1]):
                continue
        if end_pos < len(line) and (sub[-1].isalnum() or sub[-1] == "_"):
            if _is_word_or_attr_char(line[end_pos]):
                continue
        if require_stmt_delimiters:
            prefix = line[:pos].rstrip()
            suffix = line[end_pos:].lstrip()
            if prefix and not prefix.endswith((";", ":")):
                continue
            if suffix and not suffix.startswith((";", "#")):
                continue
        return True


def _is_ordered_contiguous_source_match(
    src_code_lines: Sequence[str],
    raw_lines: Sequence[str],
    *,
    allow_subwindow: bool = True,
) -> bool:
    """Checks whether src_code_lines matches raw_lines (with token-boundary boundary lines when allow_subwindow is True)."""
    raw_nonempty = [ln.strip() for ln in raw_lines if ln.strip()]
    k = len(src_code_lines)
    if k == 0 or k != len(raw_nonempty):
        return False
    if not allow_subwindow:
        return list(src_code_lines) == raw_nonempty
    if k == 1:
        target = src_code_lines[0]
        is_stmt_like = not _is_valid_eval_expr(target)
        return _has_token_boundary_subslice(
            target,
            raw_nonempty[0],
            require_stmt_delimiters=is_stmt_like,
        )
    if not _has_token_boundary_subslice(
        src_code_lines[0], raw_nonempty[0], anchor_end=True
    ):
        return False
    if not _has_token_boundary_subslice(
        src_code_lines[-1], raw_nonempty[-1], anchor_start=True
    ):
        return False
    return all(
        raw_nonempty[i] == src_code_lines[i] for i in range(1, k - 1)
    )


def _normalize_clause_header_snippet(dedented: str) -> str:
    """Normalizes standalone elif/else/except/finally/case clause headers into parseable statements."""
    stripped = dedented.lstrip(" \t")
    if re.match(r"^elif(?:\s|\(|\[|:)", stripped):
        return "if" + stripped[4:]
    if re.match(r"^else\s*:", stripped):
        return "if 1" + stripped[4:]
    if re.match(r"^(?:except(?:\*|\s|\(|:)|finally\s*:)", stripped):
        return f"try:\n    pass\n{stripped}"
    if re.match(r"^case(?:\s|\(|\[)", stripped):
        return f"match 0:\n{textwrap.indent(stripped, '    ')}"
    return dedented


def _parse_normalized_stmt_dumps(text: str) -> Optional[List[str]]:
    """Parses a code snippet into position-independent AST statement dumps."""
    raw_dedented = dedent_source_text(text).strip()
    if not raw_dedented:
        return None
    dedented = _normalize_clause_header_snippet(raw_dedented)
    for cand in (
        dedented,
        f"async def _norm_wrap():\n{textwrap.indent(dedented, '    ')}",
    ):
        try:
            with _suppress_syntax_warnings():
                parsed = ast.parse(cand)
            stmts = (
                parsed.body
                if cand == dedented
                else getattr(parsed.body[0], "body", [])
            )
            return (
                [ast.dump(s, include_attributes=False) for s in stmts]
                if stmts
                else None
            )
        except (SyntaxError, ValueError, UnicodeDecodeError, RecursionError, MemoryError):
            continue
    return None


def _scan_line_unquoted_comment(
    line: str,
    in_quote: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Strips an unquoted trailing '#' comment while tracking triple-quote state across lines."""
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_quote is not None:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(in_quote, i):
                i += len(in_quote)
                in_quote = None
                continue
            i += 1
            continue
        if ch in ("'", '"'):
            triple = ch * 3
            if line.startswith(triple, i):
                in_quote = triple
                i += 3
            else:
                in_quote = ch
                i += 1
            continue
        if ch == "#":
            return line[:i].strip(), None
        i += 1
    if in_quote in ("'", '"') and not line.rstrip("\r\n").endswith("\\"):
        in_quote = None
    return line.strip(), in_quote


def _strip_unquoted_comment(line: str) -> str:
    """Strips a trailing '#' comment outside of single, double, or triple-quoted strings."""
    stripped, _ = _scan_line_unquoted_comment(line, None)
    return stripped


def _strip_inline_comment_line(line: str) -> str:
    """Strips trailing Python comments and normalizes intra-line token spacing."""
    toks: List[str] = []
    try:
        for tok in tokenize.generate_tokens(
            io.StringIO(line.lstrip(" \t")).readline
        ):
            if tok.type in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            ):
                continue
            toks.append(tok.string)
        return " ".join(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        if toks and "multi-line statement" in str(exc).lower():
            return " ".join(toks)
        return _strip_unquoted_comment(line)


def _strip_comments_from_lines(lines: Sequence[str]) -> List[str]:
    """Strips comments across a sequence of lines while preserving '#' inside multiline strings."""
    result: List[str] = []
    in_quote: Optional[str] = None
    for ln in lines:
        prev_quote = in_quote
        stripped_ln, next_quote = _scan_line_unquoted_comment(ln, prev_quote)
        in_quote = next_quote
        if prev_quote is None and next_quote is None:
            norm = _strip_inline_comment_line(ln)
        else:
            norm = stripped_ln
        if norm:
            result.append(norm)
    return result


def _is_normalized_source_equivalent(
    src_code: str,
    raw_lines: Sequence[str],
    *,
    allow_subwindow: bool = False,
) -> bool:
    """Checks whether normalized or comment-stripped src_code matches raw_lines."""
    src_dumps = _parse_normalized_stmt_dumps(src_code)
    raw_dumps = _parse_normalized_stmt_dumps("".join(raw_lines))
    has_boundary_semicolon = bool(
        raw_lines and (";" in raw_lines[0] or ";" in raw_lines[-1])
    )
    if src_dumps and raw_dumps:
        if not (allow_subwindow and has_boundary_semicolon):
            if src_dumps == raw_dumps:
                return True
        elif len(src_dumps) <= len(raw_dumps):
            k = len(src_dumps)
            if any(
                raw_dumps[i : i + k] == src_dumps
                for i in range(len(raw_dumps) - k + 1)
            ):
                return True
    norm_src_lines = _strip_comments_from_lines(split_source_lines(src_code))
    norm_raw_lines = _strip_comments_from_lines(raw_lines)
    if norm_src_lines and norm_raw_lines:
        return _is_ordered_contiguous_source_match(
            norm_src_lines,
            norm_raw_lines,
            allow_subwindow=allow_subwindow,
        )
    return False


def _get_tree_sha256_digest(mod_tree: ast.AST) -> str:
    """Returns a cached SHA-256 hex digest of mod_tree for downstream cache keys when source_text is absent."""
    def _compute() -> str:
        cached_lines = _get_cached_tree_source_lines(mod_tree)
        hasher = hashlib.sha256()
        if cached_lines is not None:
            for line in cached_lines:
                hasher.update(line.encode("utf-8"))
                hasher.update(b"\n")
            return hasher.hexdigest()
        hasher.update(
            ast.dump(mod_tree, include_attributes=True).encode("utf-8")
        )
        return hasher.hexdigest()

    if getattr(mod_tree, "_is_ephemeral_unit_mod", False):
        return _compute()
    cache_key = _ScopeCacheKey(mod_tree, "tree_sha256_digest")
    return cast(
        str,
        _get_or_compute_scope_cached(_scope_cell_info_cache, cache_key, _compute),
    )


def _unit_source_code_matches_tree(
    unit: Dict[str, Any],
    src_code: str,
    file_tree: ast.AST,
) -> bool:
    """Checks whether unit['source_code'] matches the ordered AST nodes in file_tree spanning (start, end)."""
    u_start = parse_unit_coord(unit, "start", default=1)
    u_end = parse_unit_coord(unit, "end", default=u_start)
    s_col = parse_unit_coord(
        unit,
        "start_col",
        default=parse_unit_coord(unit, "col_offset", default=None),
    )
    e_col = parse_unit_coord(
        unit,
        "end_col",
        default=parse_unit_coord(unit, "end_col_offset", default=None),
    )
    has_explicit_cols = s_col is not None or e_col is not None
    u_kind_str = str(unit.get("kind") or "")
    is_inline_expr_kind = u_kind_str in (
        "comprehension",
        "complex_expr",
        "closure",
    )
    span_idx = _get_span_line_index(file_tree)
    end_entries = [
        entry
        for entry in (
            *span_idx.by_start_line.get(u_end, ()),
            *span_idx.by_end_line.get(u_end, ()),
        )
        if entry.n_end == u_end
    ]
    src_dumps = _parse_normalized_stmt_dumps(src_code)
    if src_dumps:
        candidate_dump_seqs: List[List[str]] = [src_dumps]
        raw_dedented = dedent_source_text(src_code).strip()
        norm_dedented = _normalize_clause_header_snippet(raw_dedented)
        if norm_dedented != raw_dedented:
            try:
                with _suppress_syntax_warnings():
                    norm_mod = ast.parse(norm_dedented)
                inner_nodes: List[ast.AST] = []
                for top_s in norm_mod.body:
                    if isinstance(top_s, ast.If):
                        inner_nodes.extend(top_s.body)
                    elif isinstance(top_s, ast.Try):
                        inner_nodes.extend(top_s.handlers)
                        inner_nodes.extend(top_s.finalbody)
                    elif is_match_node(top_s):
                        for c_node in getattr(top_s, "cases", ()):
                            inner_nodes.extend(getattr(c_node, "body", ()))
                if inner_nodes:
                    candidate_dump_seqs.append(
                        [
                            ast.dump(n, include_attributes=False)
                            for n in inner_nodes
                        ]
                    )
            except (
                SyntaxError,
                ValueError,
                UnicodeDecodeError,
                RecursionError,
                MemoryError,
            ):
                pass
        parent_map = _get_scope_parent_map(file_tree)
        candidate_parents: List[ast.AST] = []
        seen_parent_ids: Set[int] = set()
        for entry in end_entries:
            node = entry.node
            if isinstance(node, (ast.stmt, ast.ExceptHandler)):
                n_s = getattr(node, "lineno", entry.n_start)
                if u_start in (n_s, entry.n_start):
                    single_dump = [span_idx.get_dump(node)]
                    if single_dump in candidate_dump_seqs:
                        return True
                parent_node = parent_map.get(node)
                if parent_node is not None and id(parent_node) not in seen_parent_ids:
                    seen_parent_ids.add(id(parent_node))
                    candidate_parents.append(parent_node)
        for node in candidate_parents:
            for attr_name in ("body", "orelse", "finalbody", "handlers"):
                seq = getattr(node, attr_name, None)
                if not isinstance(seq, list) or not seq:
                    continue
                in_span: List[ast.AST] = []
                for s in seq:
                    s_estart, _ = _node_earliest_start_pos(s)
                    s_lineno = getattr(s, "lineno", s_estart)
                    s_end = getattr(s, "end_lineno", s_lineno)
                    if (
                        isinstance(s_lineno, int)
                        and isinstance(s_end, int)
                        and (s_lineno >= u_start or (isinstance(s_estart, int) and s_estart >= u_start))
                        and s_end <= u_end
                    ):
                        in_span.append(s)
                if not in_span:
                    continue
                first_estart, _ = _node_earliest_start_pos(in_span[0])
                first_lineno = getattr(in_span[0], "lineno", first_estart)
                last_end = getattr(
                    in_span[-1],
                    "end_lineno",
                    getattr(in_span[-1], "lineno", None),
                )
                starts_at_u_start = u_start in (first_lineno, first_estart) or (
                    attr_name in ("body", "orelse", "finalbody")
                    and first_lineno == u_start + 1
                )
                if not (starts_at_u_start and last_end == u_end):
                    continue
                span_dumps = [span_idx.get_dump(s) for s in in_span]
                for target_dumps in candidate_dump_seqs:
                    if span_dumps == target_dumps:
                        return True
                    if (
                        (has_explicit_cols or (u_start == u_end and len(span_dumps) > 1))
                        and len(target_dumps) <= len(span_dumps)
                    ):
                        k = len(target_dumps)
                        if any(
                            span_dumps[i : i + k] == target_dumps
                            for i in range(len(span_dumps) - k + 1)
                        ):
                            return True
    dedented = dedent_source_text(src_code).strip()
    if dedented and (is_inline_expr_kind or has_explicit_cols or not src_dumps):
        try:
            with _suppress_syntax_warnings():
                expr_node = ast.parse(dedented, mode="eval").body
            expr_dump = ast.dump(expr_node, include_attributes=False)
            for entry in end_entries:
                node = entry.node
                if (
                    isinstance(node, ast.expr)
                    and entry.n_start == u_start
                    and span_idx.get_dump(node) == expr_dump
                ):
                    return True
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError):
            pass
    return False


def _has_out_of_bounds_unit_cols(
    unit: Dict[str, Any],
    unit_code_lines: Sequence[str],
    *,
    is_pre_sliced: bool,
    is_placeholder_origin: bool,
    explicit_col_sliced: bool,
) -> bool:
    """Returns True if explicit column coordinates on unit are invalid or out of bounds for unit_code_lines."""
    raw_s_col = _extract_slice_col_offset(
        unit, ("start_col", "start_col_offset", "col_offset"), default=None
    )
    raw_e_col = _extract_slice_col_offset(
        unit, ("end_col", "end_col_offset"), default=None
    )
    if raw_s_col == -1 or raw_e_col == -1:
        return True
    if (
        is_placeholder_origin
        or explicit_col_sliced
        or not unit_code_lines
        or (raw_s_col is None and raw_e_col is None)
    ):
        return False
    first_b = len(unit_code_lines[0].rstrip("\r\n").encode("utf-8"))
    last_b = len(unit_code_lines[-1].rstrip("\r\n").encode("utf-8"))
    s_in_bounds = raw_s_col is None or (0 <= raw_s_col < first_b)
    e_in_bounds = raw_e_col is None or (0 < raw_e_col <= last_b)
    order_ok = (
        len(unit_code_lines) > 1
        or raw_s_col is None
        or raw_e_col is None
        or raw_s_col < raw_e_col
    )
    if s_in_bounds and e_in_bounds and order_ok:
        return False
    if is_pre_sliced and not unit_code_lines[0][:1].isspace():
        if len(unit_code_lines) == 1:
            if (
                raw_s_col is not None
                and raw_e_col is not None
                and 0 < raw_s_col < raw_e_col
                and first_b == raw_e_col - raw_s_col
            ) or (
                "source_lines" not in unit
                and raw_e_col is None
                and _is_valid_eval_expr(unit_code_lines[0])
            ):
                return False
        elif (
            raw_e_col is not None
            and last_b == raw_e_col
            and _is_valid_eval_expr("".join(unit_code_lines))
        ):
            return False
    return True


def _inspect_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    file_tree: Optional[ast.AST] = None,
) -> Dict[str, Any]:
    cached_full_source: List[Optional[str]] = [None]
    full_source_loaded: List[bool] = [False]

    def _get_full_source() -> Optional[str]:
        if not full_source_loaded[0]:
            cached_full_source[0] = _load_unit_file_text(unit, repo_root=repo_root)
            full_source_loaded[0] = True
        return cached_full_source[0]

    unit_code_lines, line_origin = extract_unit_source_lines_with_state(
        unit, repo_root=repo_root
    )
    is_pre_sliced = line_origin.is_pre_sliced or line_origin.is_placeholder
    is_placeholder_origin = line_origin.is_placeholder
    explicit_col_sliced = bool(unit.get("source_lines_is_col_sliced"))
    col_bounds_invalid = _has_out_of_bounds_unit_cols(
        unit,
        unit_code_lines,
        is_pre_sliced=is_pre_sliced,
        is_placeholder_origin=is_placeholder_origin,
        explicit_col_sliced=explicit_col_sliced,
    )
    src_code_str = (
        unit["source_code"].strip()
        if isinstance(unit.get("source_code"), str)
        else ""
    )
    is_placeholder_source = not src_code_str and is_placeholder_origin
    inferred_s_col, inferred_e_col = _unit_cols(
        unit, None if is_placeholder_origin else unit_code_lines
    )
    slice_unit = (
        {**unit, "start_col": inferred_s_col, "end_col": inferred_e_col}
        if inferred_s_col is not None or inferred_e_col is not None
        else unit
    )
    raw_lines = (
        []
        if col_bounds_invalid
        else (
            split_source_lines(str(unit["source_code"]))
            if is_placeholder_origin and src_code_str
            else _slice_unit_token_lines(
                slice_unit,
                unit_code_lines,
                is_pre_sliced=is_pre_sliced,
                explicit_col_sliced=explicit_col_sliced,
            )
        )
    )
    if not is_placeholder_origin and unit_code_lines and not raw_lines:
        col_bounds_invalid = True
    if (
        not is_placeholder_origin
        and raw_lines
        and unit.get("kind") not in ("comprehension", "complex_expr")
        and (not src_code_str or "".join(raw_lines).strip() != src_code_str)
    ):
        if (
            len(raw_lines) == 1
            and (";" in raw_lines[0] or unit.get("kind") == "closure")
            and inferred_s_col is not None
            and inferred_e_col is not None
            and 0 <= inferred_s_col < inferred_e_col
            <= len(raw_lines[0].rstrip("\r\n").encode("utf-8"))
        ):
            sliced_stmt = _slice_utf8_line(
                raw_lines[0], inferred_s_col, inferred_e_col
            )
            if sliced_stmt.strip() and (
                ";" in raw_lines[0]
                or sliced_stmt.lstrip().startswith(("lambda", "(lambda"))
            ):
                raw_lines = [sliced_stmt]
        elif len(raw_lines) > 1 and (
            ";" in raw_lines[0] or ";" in raw_lines[-1]
        ):
            raw_lines = list(raw_lines)
            if (
                inferred_s_col is not None
                and 0
                < inferred_s_col
                < len(raw_lines[0].rstrip("\r\n").encode("utf-8"))
                and ";" in raw_lines[0]
            ):
                lead_ws = raw_lines[0][
                    : len(raw_lines[0]) - len(raw_lines[0].lstrip(" \t"))
                ]
                raw_lines[0] = lead_ws + _slice_utf8_line(
                    raw_lines[0], inferred_s_col, None
                ).lstrip(" \t")
            if (
                inferred_e_col is not None
                and 0
                < inferred_e_col
                <= len(raw_lines[-1].rstrip("\r\n").encode("utf-8"))
                and ";" in raw_lines[-1]
            ):
                raw_lines[-1] = _slice_utf8_line(
                    raw_lines[-1], 0, inferred_e_col
                )
    # Stale-source check: when both a backing module (file_tree or _get_full_source())
    # and an embedded unit["source_code"] are present, verify that source_code matches
    # the module lines at (start, end, start_col, end_col). When no backing module is
    # available (e.g., an in-memory unit carrying only source_code), raw_lines falls
    # back to unit["source_code"].
    stale_source_code_mismatch = col_bounds_invalid
    if stale_source_code_mismatch:
        pass
    elif is_placeholder_origin and src_code_str:
        if file_tree is not None and not _unit_source_code_matches_tree(
            unit, str(unit["source_code"]), file_tree
        ):
            stale_source_code_mismatch = True
    elif (
        src_code_str
        and dedent_source_text("".join(raw_lines)).strip()
        != dedent_source_text(str(unit["source_code"])).strip()
    ):
        src_code_lines = [
            ln.strip()
            for ln in split_source_lines(str(unit["source_code"]))
            if ln.strip()
        ]
        has_boundary_semicolon = bool(
            raw_lines and (";" in raw_lines[0] or ";" in raw_lines[-1])
        )
        allow_subwindow = (
            inferred_s_col is not None
            or inferred_e_col is not None
            or has_boundary_semicolon
            or str(unit.get("kind") or "")
            in ("comprehension", "complex_expr", "closure")
        )
        candidate_line_windows: List[Sequence[str]] = [raw_lines]
        if (
            inferred_s_col is not None or inferred_e_col is not None
        ) and raw_lines:
            col_sliced = list(raw_lines)
            first_b_len = len(col_sliced[0].rstrip("\r\n").encode("utf-8"))
            if len(col_sliced) == 1:
                s_c = inferred_s_col if inferred_s_col is not None and 0 <= inferred_s_col <= first_b_len else 0
                e_c = (
                    inferred_e_col
                    if inferred_e_col is not None and s_c <= inferred_e_col <= first_b_len
                    else None
                )
                col_sliced[0] = _slice_utf8_line(col_sliced[0], s_c, e_c)
            else:
                last_b_len = len(col_sliced[-1].rstrip("\r\n").encode("utf-8"))
                if inferred_s_col is not None and 0 < inferred_s_col < first_b_len:
                    col_sliced[0] = _slice_utf8_line(col_sliced[0], inferred_s_col, None)
                if inferred_e_col is not None and 0 < inferred_e_col <= last_b_len:
                    col_sliced[-1] = _slice_utf8_line(col_sliced[-1], 0, inferred_e_col)
            candidate_line_windows.append(col_sliced)
        matches_backing_source = any(
            dedent_source_text("".join(win)).strip()
            == dedent_source_text(str(unit["source_code"])).strip()
            or _is_ordered_contiguous_source_match(
                src_code_lines, win, allow_subwindow=allow_subwindow
            )
            or _is_normalized_source_equivalent(
                str(unit["source_code"]), win, allow_subwindow=allow_subwindow
            )
            for win in candidate_line_windows
        )
        if (
            (file_tree is not None or _get_full_source() is not None)
            and src_code_lines
            and not matches_backing_source
        ):
            stale_source_code_mismatch = True
        elif not raw_lines or allow_subwindow:
            raw_lines = split_source_lines(str(unit["source_code"]))
    dedented = dedent_source_text("".join(raw_lines))
    skip_pre_unit_closures_flag, strict_with_flag = resolve_unit_closure_flags(
        unit
    )
    extra_non_sup = frozenset(
        normalize_non_suppressing_ctx_managers(
            unit.get("non_suppressing_ctx_managers")
        )
    )
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
        "closure_hazard_reasons": [],
        "is_control_flow_safe": True,
        "has_yield": False,
        "has_yield_assignment": False,
        "has_try_or_with": False,
        "has_return": False,
        "has_return_value": False,
        "has_super": False,
        "has_mangled_names": False,
        "local_imports": [],
        "yield_expr_names": [],
        "is_async": False,
        "conditional_outputs": [],
        "definite_stores": [],
        "has_instance_binding": False,
        "has_class_binding": False,
        "binding_kind": None,
        "has_receiver_access": False,
        "instance_attrs": [],
        "class_attrs": [],
        "outputs_ambiguous": False,
        "type_params": [],
        "type_param_free_vars": [],
        "has_class_scoped_type_params": False,
        "deleted_names": [],
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "skip_pre_unit_closures": skip_pre_unit_closures_flag,
        "strict_with_suppression": strict_with_flag,
        "non_suppressing_ctx_managers": sorted(extra_non_sup),
        "heuristic_ctx_managers": [],
    }
    if stale_source_code_mismatch:
        return {
            **empty_res,
            "control_flow_hazards": ["stale_source_code"],
            "is_control_flow_safe": False,
        }
    if is_placeholder_source and file_tree is None:
        unit_kind_str = str(unit.get("kind") or "")
        if (
            bool(unit.get("is_subroutine"))
            or unit_kind_str in _SUBROUTINE_KINDS
            or unit_kind_str
            in (
                "subroutine",
                "basic_block",
                "linear_block",
                "stmt_seq",
                "statement",
                "closure",
                "comprehension",
                "complex_expr",
            )
        ):
            return {
                **empty_res,
                "control_flow_hazards": ["severed_closure_cell"],
                "closure_hazard_reasons": ["unavailable_source_placeholder"],
                "is_control_flow_safe": False,
            }
    if not dedented.strip():
        return empty_res

    tree: Optional[ast.AST] = None
    loop_offset = 0
    candidate_stmts: List[ast.stmt] = []
    clause_dedented = _normalize_clause_header_snippet(dedented)

    parse_candidates = [
        (dedented, 0),
        *([(clause_dedented, 0)] if clause_dedented != dedented else []),
        (f"if 1:\n{dedented}", 0),
        (f"def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        (f"async def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(dedented, '    ')}", 0),
        *(
            [
                (
                    f"def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(clause_dedented, '    ')}",
                    0,
                ),
                (
                    f"async def {_SYNTHETIC_WRAPPER_NAME}():\n{textwrap.indent(clause_dedented, '    ')}",
                    0,
                ),
            ]
            if clause_dedented != dedented
            else []
        ),
        (
            f"def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
        (
            f"async def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
            f"{textwrap.indent(dedented, '        ')}",
            1,
        ),
        *(
            [
                (
                    f"def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
                    f"{textwrap.indent(clause_dedented, '        ')}",
                    1,
                ),
                (
                    f"async def {_SYNTHETIC_WRAPPER_NAME}():\n    for _ in (0,):\n"
                    f"{textwrap.indent(clause_dedented, '        ')}",
                    1,
                ),
            ]
            if clause_dedented != dedented
            else []
        ),
    ]

    had_resource_error = False
    cand_text = dedented
    for cand_text, offset in parse_candidates:
        try:
            with _suppress_syntax_warnings():
                tree = ast.parse(cand_text)
            loop_offset = offset
            if cand_text in (dedented, clause_dedented):
                if isinstance(tree, ast.Module):
                    if len(tree.body) == 1 and isinstance(
                        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = tree.body[0].body
                    else:
                        candidate_stmts = tree.body
            elif cand_text.startswith("if 1:\n"):
                if (
                    isinstance(tree, ast.Module)
                    and len(tree.body) == 1
                    and isinstance(tree.body[0], ast.If)
                ):
                    inner_body = tree.body[0].body
                    tree = ast.Module(body=inner_body, type_ignores=[])
                    if len(inner_body) == 1 and isinstance(
                        inner_body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        candidate_stmts = inner_body[0].body
                    else:
                        candidate_stmts = inner_body
            elif offset == 1 and isinstance(tree, ast.Module):
                wrapper_fn = tree.body[0]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for_loop = wrapper_fn.body[0]
                    if isinstance(for_loop, (ast.For, ast.AsyncFor, ast.While)):
                        candidate_stmts = for_loop.body
            elif isinstance(tree, ast.Module):
                wrapper_fn = tree.body[0]
                if isinstance(wrapper_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    candidate_stmts = wrapper_fn.body
            break
        except (RecursionError, MemoryError):
            had_resource_error = True
            break
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue

    if tree is None:
        if had_resource_error:
            return {
                **empty_res,
                "control_flow_hazards": ["pathological_ast"],
                "is_control_flow_safe": False,
            }
        return empty_res

    try:
        if "is_subroutine" in unit and unit.get("is_subroutine") is not None:
            is_subroutine = bool(unit["is_subroutine"])
        elif str(unit.get("kind") or "") in _SUBROUTINE_KINDS:
            is_subroutine = True
        elif str(unit.get("kind") or "") in _WHOLE_UNIT_KINDS:
            is_subroutine = False
        else:
            is_subroutine = is_subroutine_unit(
                unit, source_text=_get_full_source(), tree=file_tree
            )

        rec_param = unit.get("receiver_param")
        rec_kind = unit.get("receiver_kind")
        is_class_rec = rec_kind == "class"

        inst_receivers: Set[str] = {"self", "this"}
        class_receivers: Set[str] = {"cls", "klass"}
        if rec_param:
            if is_class_rec:
                class_receivers.add(rec_param)
            else:
                inst_receivers.add(rec_param)

        all_receivers = inst_receivers | class_receivers

        cand_lines = split_source_lines(cand_text)
        visitor = _ScopeVisitor(
            loop_offset=loop_offset,
            source_lines=cand_lines,
            is_subroutine=is_subroutine,
            receiver_names=tuple(all_receivers),
        )
        visitor.visit(tree)

        hazards: List[str] = []
        if visitor.naked_breaks > 0:
            hazards.append("naked_break")
        if visitor.naked_continues > 0:
            hazards.append("naked_continue")
        if visitor.has_yield:
            hazards.append("generator_yield")
        if is_subroutine and visitor.has_return:
            hazards.append("embedded_return")

        has_class_scoped_type_params = False
        if visitor.type_param_free_vars and not is_subroutine:
            s_line = parse_unit_coord(unit, "start", default=1)
            e_line = parse_unit_coord(unit, "end", default=s_line)
            u_col = inferred_s_col
            top_fn = (
                tree.body[0]
                if isinstance(tree, ast.Module)
                and len(tree.body) == 1
                and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
                else None
            )
            top_fn_name = getattr(top_fn, "name", None) if top_fn is not None else None
            top_fn_count = (
                sum(
                    1
                    for n in ast.walk(top_fn)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                if top_fn is not None
                else None
            )
            parsed_tree = file_tree
            if parsed_tree is None:
                full_src = _get_full_source()
                if full_src:
                    parsed_tree = _get_or_parse_tree(full_src)
            enclosing_fn_locals: Set[str] = set()
            enclosing_cls_locals: Set[str] = set()
            has_enclosing_fn = False
            has_enclosing_cls = False
            prev_enclosed_cls_name: Optional[str] = None
            if parsed_tree is not None:
                full_src_for_lines = _get_full_source()
                mod_src_lines: Optional[Tuple[str, ...]] = (
                    tuple(
                        ln.rstrip("\r\n")
                        for ln in split_source_lines(full_src_for_lines)
                    )
                    if full_src_for_lines
                    else None
                )
                matched_self = False
                for earliest_start, n_end, _span, scope_node in _get_tree_scope_intervals(
                    parsed_tree, source_lines=mod_src_lines
                ):
                    if earliest_start > s_line or e_line > n_end:
                        continue
                    if isinstance(scope_node, ast.ClassDef):
                        if not has_enclosing_cls:
                            has_enclosing_cls = True
                            prev_enclosed_cls_name = getattr(scope_node, "name", None)
                            enclosing_cls_locals.update(
                                _extract_single_node_local_names(
                                    scope_node, include_globals=False
                                )
                            )
                        else:
                            if isinstance(prev_enclosed_cls_name, str):
                                enclosing_cls_locals.add(prev_enclosed_cls_name)
                            prev_enclosed_cls_name = getattr(scope_node, "name", None)
                            enclosing_cls_locals.update(
                                _extract_type_param_names(scope_node)
                            )
                    elif isinstance(scope_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if (
                            not matched_self
                            and s_line
                            in (
                                earliest_start,
                                getattr(scope_node, "lineno", earliest_start),
                            )
                            and (
                                e_line == n_end
                                or top_fn_name is not None
                                or u_col is not None
                            )
                            and (
                                u_col is None
                                or getattr(scope_node, "col_offset", 0) == u_col
                            )
                            and (
                                top_fn_name is None
                                or getattr(scope_node, "name", None) == top_fn_name
                            )
                            and (
                                top_fn_count is None
                                or sum(
                                    1
                                    for n in ast.walk(scope_node)
                                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                )
                                == top_fn_count
                            )
                        ):
                            matched_self = True
                            continue
                        if (
                            earliest_start < s_line
                            or n_end > e_line
                            or (u_col is None and matched_self)
                            or (
                                u_col is not None
                                and getattr(scope_node, "col_offset", 0) < u_col
                            )
                        ):
                            has_enclosing_fn = True
                            enclosing_fn_locals.update(
                                _extract_single_node_local_names(
                                    scope_node, include_globals=False
                                )
                            )
            if has_enclosing_fn and any(
                fv in enclosing_fn_locals for fv in visitor.type_param_free_vars
            ):
                hazards.append("type_param_closure_capture")
            elif (
                not has_enclosing_fn
                and not has_enclosing_cls
                and unit.get("kind") == "closure"
            ):
                hazards.append("type_param_closure_capture")
            if has_enclosing_cls and any(
                fv in enclosing_cls_locals for fv in visitor.type_param_free_vars
            ):
                has_class_scoped_type_params = True
                if (
                    has_enclosing_fn
                    or unit.get("kind") == "closure"
                ) and "type_param_closure_capture" not in hazards:
                    hazards.append("type_param_closure_capture")
            elif not has_enclosing_cls and (
                unit.get("enclosing_class")
                or unit.get("kind") in ("method", "async_method")
            ):
                has_class_scoped_type_params = True

        # Resolve module AST, unit column bounds, and enclosing function chain once per unit
        closure_hazard_reasons: List[str] = []
        u_start = parse_unit_coord(unit, "start", default=0)
        u_end = parse_unit_coord(unit, "end", default=u_start)
        mod_tree: Optional[ast.AST] = file_tree
        if mod_tree is None and u_start > 0:
            full_src = _get_full_source()
            if full_src:
                mod_tree = _get_or_parse_tree(full_src)
        resolved_s_col, resolved_e_col = inferred_s_col, inferred_e_col
        cols_resolved = False
        enclosing_chain: List[Tuple[int, int, ast.AST]] = []
        full_mod_src = _get_full_source()
        full_mod_lines: Optional[Tuple[str, ...]] = (
            tuple(ln.rstrip("\r\n") for ln in split_source_lines(full_mod_src))
            if full_mod_src
            else None
        )
        if mod_tree is not None and full_mod_lines is not None:
            _register_tree_source_lines(mod_tree, full_mod_lines)
        if mod_tree is not None and u_start > 0:
            resolved_s_col, resolved_e_col = _resolve_unit_col_bounds(
                candidate_stmts,
                mod_tree,
                u_start,
                u_end,
                resolved_s_col,
                resolved_e_col,
            )
            cols_resolved = True
            enclosing_chain = _find_enclosing_function_chain(
                mod_tree,
                u_start,
                u_end,
                resolved_s_col,
                resolved_e_col,
                source_lines=full_mod_lines,
            )

        # Definite, conditional, and branch-aware deletion analysis across candidate statements
        may_deleted: Set[str] = set()
        live_in_deleted: Set[str] = set()
        initial_sup_prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE
        if mod_tree is not None and any(
            isinstance(n, (ast.With, ast.AsyncWith))
            for st in candidate_stmts
            for n in ast.walk(st)
        ):
            sup_fn = enclosing_chain[-1][2] if enclosing_chain else mod_tree
            sup_scope_ctx = (
                _get_fn_scope_context(
                    sup_fn,
                    mod_tree=mod_tree,
                    enclosing_scopes=[f for _, _, f in enclosing_chain[:-1]],
                )
                if isinstance(sup_fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                else None
            )
            initial_sup_prov = _seed_scope_suppress_aliases(
                sup_fn,
                mod_tree=mod_tree,
                u_start=u_start,
                scope_ctx=sup_scope_ctx,
            )
        if "open" in visitor.params:
            initial_sup_prov = initial_sup_prov.with_shadowed_builtins(
                ("open",)
            )
        def_assigned, cond_assigned = _analyze_block_assignment(
            candidate_stmts,
            may_deleted=may_deleted,
            live_in_deleted=live_in_deleted,
            suppress_aliases=initial_sup_prov,
            extra_non_suppressing_ctx_managers=extra_non_sup,
            strict_with_suppression=strict_with_flag,
        )
        for del_name in sorted(live_in_deleted):
            if (
                del_name not in visitor.params
                and del_name not in visitor.globals
                and del_name not in BUILTIN_NAMES
            ):
                if del_name not in visitor.loads:
                    visitor.loads.append(del_name)
                if del_name not in visitor.read_before_write:
                    visitor.read_before_write.append(del_name)

        if is_subroutine:
            live_in_visitor = _UnitLiveInVisitor(1, 10**9, include_annotations=False)
            for stmt in candidate_stmts:
                live_in_visitor.visit(stmt)
            excluded_cond_vars = (
                set(visitor.params) | set(visitor.globals) | BUILTIN_NAMES
            )
            cond_live_ins = [
                v
                for v in sorted(live_in_visitor.live_ins)
                if v in visitor.loads
                and v not in excluded_cond_vars
                and v not in visitor.read_before_write
            ]
            cond_escaping_cands = [
                v
                for v in sorted(
                    (cond_assigned - def_assigned)
                    | set(cond_live_ins)
                    | (live_in_visitor.live_ins & set(visitor.stores))
                )
                if v not in excluded_cond_vars
            ]
            if cond_live_ins or cond_escaping_cands:
                pre_bound: Set[str] = set()
                rejected_escaping_vars: Set[str] = set()
                if mod_tree is not None and full_mod_lines is not None:
                    _register_tree_source_lines(mod_tree, full_mod_lines)
                if mod_tree is not None and u_start > 0:
                    u_start_col, u_end_col = resolved_s_col, resolved_e_col
                    pre_bound, _ = _find_enclosing_pre_bound_names(
                        mod_tree,
                        u_start,
                        u_end,
                        u_start_col=u_start_col,
                        u_end_col=u_end_col,
                    )
                    if enclosing_chain:
                        in_enclosing_class_body = isinstance(
                            _resolve_immediate_scope_in_fn(
                                enclosing_chain[-1][2],
                                u_start,
                                u_end,
                                u_start_col=u_start_col,
                                u_end_col=u_end_col,
                            ),
                            ast.ClassDef,
                        )
                        enc_nonlocals, enc_outer_globals = (
                            _extract_scope_nonlocals_and_outer_globals(
                                enclosing_chain[-1][2],
                                u_start,
                                u_end,
                                u_start_col=u_start_col,
                                u_end_col=u_end_col,
                            )
                        )
                        enc_escaping = enc_nonlocals | enc_outer_globals
                        def_pre_bound = (
                            _find_enclosing_chain_definite_pre_bound(
                                enclosing_chain,
                                u_start,
                                u_end,
                                u_start_col=u_start_col,
                                u_end_col=u_end_col,
                                mod_tree=mod_tree,
                                strict_with_suppression=strict_with_flag,
                                extra_non_suppressing_ctx_managers=extra_non_sup,
                            )
                        )
                        write_only_cond_cands = {
                            v
                            for v in cond_escaping_cands
                            if v not in enc_escaping
                            and v not in visitor.loads
                            and v not in live_in_visitor.live_ins
                            and v in pre_bound
                            and v in def_pre_bound
                        }
                        live_write_only_cond: Set[str] = set()
                        if write_only_cond_cands:
                            pre_outs_decl = resolve_unit_declared_outputs(unit)
                            if isinstance(
                                pre_outs_decl, (list, tuple, set, frozenset)
                            ):
                                live_write_only_cond = (
                                    write_only_cond_cands & set(pre_outs_decl)
                                )
                            elif visitor.returns and not is_subroutine:
                                live_write_only_cond = (
                                    write_only_cond_cands & set(visitor.returns)
                                )
                            else:
                                full_src_text = _get_full_source() or ""
                                ds_unit: Dict[str, Any] = {
                                    **unit,
                                    "start": u_start,
                                    "end": u_end,
                                }
                                if u_start_col is not None:
                                    ds_unit["start_col"] = u_start_col
                                if u_end_col is not None:
                                    ds_unit["end_col"] = u_end_col
                                # Always include pre-unit closures when checking whether a
                                # write-only conditional variable is live downstream: pruning
                                # a required output in lenient mode would break callers.
                                ds_live = collect_downstream_read_names(
                                    full_src_text,
                                    ds_unit,
                                    candidates=write_only_cond_cands,
                                    tree=mod_tree,
                                    skip_pre_unit_closures=False,
                                    source_digest=(
                                        None
                                        if full_src_text
                                        else _get_tree_sha256_digest(mod_tree)
                                    ),
                                )
                                if ds_live is None:
                                    live_write_only_cond = set(
                                        write_only_cond_cands
                                    )
                                else:
                                    live_write_only_cond = (
                                        write_only_cond_cands & ds_live
                                    )
                        for v in cond_escaping_cands:
                            if v in enc_escaping:
                                if (
                                    v in enc_nonlocals
                                    and v in pre_bound
                                    and v in def_pre_bound
                                ):
                                    if v not in visitor.loads:
                                        visitor.loads.append(v)
                                    if v not in visitor.read_before_write:
                                        visitor.read_before_write.append(v)
                                else:
                                    rejected_escaping_vars.add(v)
                                    if (
                                        v in visitor.read_before_write
                                        and v not in def_pre_bound
                                    ):
                                        visitor.read_before_write.remove(v)
                                    if "severed_closure_cell" not in hazards:
                                        hazards.append("severed_closure_cell")
                                    sub_r = (
                                        "global_escape"
                                        if v in enc_outer_globals
                                        else "cell_severing"
                                    )
                                    if sub_r not in closure_hazard_reasons:
                                        closure_hazard_reasons.append(sub_r)
                            elif (
                                not in_enclosing_class_body
                                and not visitor.has_yield
                                and v in pre_bound
                                and v in def_pre_bound
                                and (
                                    v in visitor.loads
                                    or v in live_in_visitor.live_ins
                                    or v in live_write_only_cond
                                )
                            ):
                                if v not in visitor.loads:
                                    visitor.loads.append(v)
                                if v not in visitor.read_before_write:
                                    visitor.read_before_write.append(v)
                            elif (
                                in_enclosing_class_body
                                or v not in def_pre_bound
                            ):
                                rejected_escaping_vars.add(v)
                        for v in cond_live_ins:
                            if v not in def_pre_bound or (
                                in_enclosing_class_body
                                and v not in enc_escaping
                            ):
                                rejected_escaping_vars.add(v)
                                if v in pre_bound:
                                    if "severed_closure_cell" not in hazards:
                                        hazards.append("severed_closure_cell")
                                    if (
                                        "conditional_live_in"
                                        not in closure_hazard_reasons
                                    ):
                                        closure_hazard_reasons.append(
                                            "conditional_live_in"
                                        )
                has_enclosing_context = mod_tree is not None and (
                    u_start > 1 or bool(_get_tree_scope_intervals(mod_tree))
                )
                for v in cond_live_ins:
                    if (
                        v not in rejected_escaping_vars
                        and (
                            v in pre_bound
                            or (not has_enclosing_context and v in def_assigned)
                        )
                        and v not in visitor.read_before_write
                    ):
                        visitor.read_before_write.append(v)

        free_vars = [
            name for name in visitor.loads
            if name not in visitor.params
            and (name not in visitor.stores or name in visitor.read_before_write)
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

        param_map = {p["name"].lstrip("*"): p.get("kind", "pos") for p in visitor.param_details}
        inputs.sort(key=lambda v: _rank_param_kind(v, param_map, receiver_names=tuple(all_receivers)))

        has_instance_binding = any(
            r in visitor.loads
            or r in visitor.params
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            for r in inst_receivers
            if r in ("self", "this")
            or (rec_param and r == rec_param and rec_kind == "instance")
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
        )
        has_class_binding = any(
            r in visitor.loads
            or r in visitor.params
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            for r in class_receivers
            if r in ("cls", "klass")
            or (rec_param and r == rec_param and is_class_rec)
            or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
        )
        binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

        primary_rec = rec_param or ("cls" if is_class_rec else None)
        _normalize_receiver_order(inputs, has_instance_binding, has_class_binding, primary_receiver=primary_rec)

        # Unit outputs: explicit returns if present; for subroutines without explicit
        # returns, all local variable stores act as unit outputs to preserve caller mutations.
        pre_outs = resolve_unit_declared_outputs(unit)

        outputs_ambiguous = False
        alias_cands = visitor.type_aliases & (set(visitor.stores) | set(visitor.nonlocals))
        downstream_aliases: Set[str] = set()
        if is_subroutine and alias_cands:
            full_src = _get_full_source()
            if full_src:
                ds_reads = collect_downstream_read_names(
                    full_src,
                    unit,
                    candidates=alias_cands,
                    tree=file_tree,
                    skip_pre_unit_closures=skip_pre_unit_closures_flag,
                    unit_inputs=set(inputs),
                )
                downstream_aliases = (
                    set(alias_cands) if ds_reads is None else ds_reads
                )
            else:
                downstream_aliases = set(alias_cands)
        excluded_local_defs = (visitor.type_aliases - downstream_aliases) | set(
            visitor.imported_names
        )
        all_deleted = _extract_deleted_names(candidate_stmts) | visitor.deleted_names
        uncond_deleted = all_deleted - (def_assigned | cond_assigned)

        if is_subroutine and isinstance(pre_outs, (list, tuple)):
            outputs = list(dict.fromkeys(pre_outs))
        elif is_subroutine and isinstance(pre_outs, (set, frozenset)):
            if len(pre_outs) > 1:
                logger.debug(
                    "Rejecting set-valued outputs (%s): unordered set cannot establish "
                    "positional order",
                    pre_outs,
                )
                outputs = []
                outputs_ambiguous = True
            else:
                outputs = list(pre_outs)
        elif visitor.returns:
            outputs = list(visitor.returns)
        elif is_subroutine:
            seen: Set[str] = set()
            outputs = []
            for v in visitor.stores:
                if (
                    v not in visitor.globals
                    and v not in BUILTIN_NAMES
                    and v not in visitor.except_vars
                    and v not in uncond_deleted
                    and v not in excluded_local_defs
                    and v not in seen
                ):
                    seen.add(v)
                    outputs.append(v)
        else:
            outputs = []

        for nl in visitor.nonlocals:
            if (
                nl in visitor.stores
                and nl not in outputs
                and nl not in visitor.except_vars
                and nl not in uncond_deleted
                and nl not in excluded_local_defs
            ):
                outputs.append(nl)

        # Detect conditional variable escapes (outputs assigned conditionally without prior
        # unconditional assignment in this block, or potentially deleted along a path).
        conditional_outputs = [
            v for v in outputs
            if (v not in inputs or v in may_deleted) and v not in def_assigned
        ]

        if (
            is_subroutine
            and bool((set(outputs) | set(inputs)) & may_deleted)
            and "deleted_output" not in hazards
        ):
            hazards.append("deleted_output")

        lex_stmts: Sequence[ast.stmt] = (
            tree.body
            if isinstance(tree, ast.Module)
            and len(tree.body) == 1
            and isinstance(
                tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and getattr(tree.body[0], "name", None) != _SYNTHETIC_WRAPPER_NAME
            else candidate_stmts
        )
        if (
            "severed_closure_cell" not in hazards
            and _has_invalid_lexical_scope_declarations(
                lex_stmts,
                enclosing_chain=enclosing_chain,
                mod_tree=mod_tree,
                source_lines=full_mod_lines,
            )
        ):
            hazards.append("severed_closure_cell")
            if "invalid_lexical_declaration" not in closure_hazard_reasons:
                closure_hazard_reasons.append("invalid_lexical_declaration")

        unit_heuristic_mgrs: Set[str] = set()
        if is_subroutine or unit.get("kind") in (
            "closure",
            "comprehension",
            "complex_expr",
        ):
            if u_start > 0 and mod_tree is not None:
                if "severed_closure_cell" not in hazards:
                    with collect_heuristic_ctx_managers() as unit_heuristic_mgrs:
                        hazard_res = _evaluate_severed_closure_cell_hazard(
                            _UnitClosureHazardContext(
                                candidate_stmts=candidate_stmts,
                                mod_tree=mod_tree,
                                u_start=u_start,
                                u_end=u_end,
                                u_start_col=resolved_s_col,
                                u_end_col=resolved_e_col,
                                is_subroutine=is_subroutine,
                                unit_kind=str(unit.get("kind") or ""),
                                inputs=inputs,
                                outputs=outputs,
                                stores=visitor.stores,
                                deleted_names=all_deleted,
                                unit_top_locals=visitor.unit_top_locals,
                                free_vars=free_vars,
                                nonlocals=visitor.nonlocals,
                                has_yield=visitor.has_yield,
                                is_async=visitor.is_async,
                                skip_pre_unit_closures=skip_pre_unit_closures_flag,
                                strict_with_suppression=strict_with_flag,
                                extra_non_suppressing_ctx_managers=extra_non_sup,
                                definite_stores=def_assigned,
                                cols_resolved=cols_resolved,
                                enclosing_chain=enclosing_chain,
                                lex_validated=True,
                            )
                        )
                    if hazard_res.has_hazard:
                        hazards.append("severed_closure_cell")
                        for r in hazard_res.reasons:
                            if r not in closure_hazard_reasons:
                                closure_hazard_reasons.append(r)
            elif "severed_closure_cell" not in hazards and (
                bool(_get_full_source())
                or (unit.get("kind") == "closure" and bool(free_vars))
            ):
                # Fail closed when module source exists but failed AST parsing, or when
                # a closure unit captures free variables without an enclosing module AST.
                # Standalone snippet-only subroutine units (source_code/source_lines with
                # no enclosing module source) are an intentional exception for isolated
                # snippet scope inspection.
                hazards.append("severed_closure_cell")
                if "unresolved_span_bounds" not in closure_hazard_reasons:
                    closure_hazard_reasons.append("unresolved_span_bounds")

        is_control_flow_safe = not any(
            h
            in (
                "naked_break",
                "naked_continue",
                "embedded_return",
                "type_param_closure_capture",
                "pathological_ast",
                "deleted_output",
                "severed_closure_cell",
                "unavailable_source_for_closure_mode",
            )
            for h in hazards
        )

        has_receiver_access = bool(
            any(
                r in visitor.loads
                or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
                for r in all_receivers
                if r in ("self", "cls", "this", "klass")
                or any(a.startswith(f"{r}.") for a in visitor.attrs_read + visitor.attrs_written)
            )
        )

        is_sub_or_inner = is_subroutine or unit.get("kind") in (
            "comprehension",
            "complex_expr",
            "class",
            "data_table",
        )
        unit_is_async = unit.get("is_async")
        if unit_is_async is None:
            if is_sub_or_inner and not visitor.has_yield:
                unit_is_async = visitor.is_async
            else:
                s_line = parse_unit_coord(unit, "start", default=1)
                e_line = parse_unit_coord(unit, "end", default=s_line)
                full_src = _get_full_source()
                if full_src:
                    unit_is_async = find_enclosing_function_is_async(
                        full_src, s_line, e_line, tree=file_tree
                    )
                else:
                    unit_is_async = None

        effective_is_async: Optional[bool]
        if visitor.is_async or unit.get("kind") in ("async_function", "async_method"):
            effective_is_async = True
        elif "is_async" in unit and unit.get("is_async") is not None:
            effective_is_async = bool(unit.get("is_async"))
        elif visitor.has_yield:
            effective_is_async = bool(unit_is_async) if unit_is_async is not None else None
        elif not is_sub_or_inner and unit.get("kind") not in (
            "function",
            "method",
            "async_function",
            "async_method",
        ):
            effective_is_async = bool(unit_is_async)
        else:
            effective_is_async = False

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
            "closure_hazard_reasons": closure_hazard_reasons,
            "is_control_flow_safe": is_control_flow_safe,
            "has_yield": visitor.has_yield,
            "has_yield_assignment": visitor.has_yield_assignment,
            "has_try_or_with": visitor.has_try_or_with,
            "has_return": visitor.has_return,
            "has_return_value": visitor.has_return_value,
            "has_super": visitor.has_super,
            "has_mangled_names": visitor.has_mangled_names,
            "local_imports": visitor.local_imports,
            "yield_expr_names": visitor.yield_expr_names,
            "is_async": effective_is_async,
            "conditional_outputs": conditional_outputs,
            "definite_stores": sorted(def_assigned),
            "has_instance_binding": has_instance_binding,
            "has_class_binding": has_class_binding,
            "binding_kind": binding_kind,
            "has_receiver_access": has_receiver_access,
            "instance_attrs": [
                a for a in visitor.attrs_read + visitor.attrs_written
                if any(a.startswith(f"{r}.") for r in inst_receivers)
            ],
            "class_attrs": [
                a for a in visitor.attrs_read + visitor.attrs_written
                if any(a.startswith(f"{r}.") for r in class_receivers)
            ],
            "outputs_ambiguous": outputs_ambiguous,
            "type_params": list(visitor.type_params),
            "type_param_free_vars": list(visitor.type_param_free_vars),
            "has_class_scoped_type_params": has_class_scoped_type_params,
            "deleted_names": sorted(may_deleted),
            "scope_schema_version": SCOPE_SCHEMA_VERSION,
            "skip_pre_unit_closures": skip_pre_unit_closures_flag,
            "strict_with_suppression": strict_with_flag,
            "non_suppressing_ctx_managers": sorted(extra_non_sup),
            "heuristic_ctx_managers": sorted(unit_heuristic_mgrs),
        }
    except (RecursionError, MemoryError):
        return {
            **empty_res,
            "control_flow_hazards": ["pathological_ast"],
            "is_control_flow_safe": False,
        }


def inspect_single_unit_scope(
    unit: Dict[str, Any],
    repo_root: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    *,
    skip_pre_unit_closures: Optional[bool] = None,
    strict_with_suppression: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping for an individual code unit."""
    if (
        skip_pre_unit_closures is not None
        or strict_with_suppression is not None
        or non_suppressing_ctx_managers is not None
    ):
        resolved_flags = resolve_unit_closure_flags(
            unit,
            skip_pre_unit_closures=skip_pre_unit_closures,
            strict_with_suppression=strict_with_suppression,
        )
        unit = dict(unit)
        if (
            skip_pre_unit_closures is not None
            or strict_with_suppression is not None
        ):
            unit.pop("closure_strictness", None)
            unit["skip_pre_unit_closures"] = resolved_flags.skip_pre_unit_closures
            unit["strict_with_suppression"] = resolved_flags.strict_with_suppression
        if non_suppressing_ctx_managers is not None:
            unit["non_suppressing_ctx_managers"] = list(
                normalize_non_suppressing_ctx_managers(
                    non_suppressing_ctx_managers,
                    unit.get("non_suppressing_ctx_managers"),
                )
            )
    return _inspect_unit_scope(unit, repo_root=repo_root, file_tree=tree)


def _align_symmetric_var_list(list1: List[str], list2: List[str]) -> List[str]:
    """Aligns shared variables between two units, preserving side 1 order if equal length."""
    common = [var for var in list1 if var in list2]
    if len(list1) == len(list2):
        return common if len(common) == len(list1) else list1
    return common if common else list1


def _has_type_param_default(tp_str: str) -> bool:
    """Checks whether a PEP 695/696 type parameter string has a top-level default."""
    cleaned = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", '""', tp_str)
    depth = 0
    for ch in cleaned:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            return True
    return False


def _merge_type_params(tp1: Sequence[str], tp2: Sequence[str]) -> List[str]:
    """Merges PEP 695 type parameters across two units without duplicate parameter names."""
    if not tp2:
        return list(tp1)
    if not tp1:
        return list(tp2)
    seen: Dict[str, str] = {}
    order: List[str] = []
    for raw_tp in list(tp1) + list(tp2):
        head = raw_tp.split(":", 1)[0].split("=", 1)[0].strip()
        ident = head.lstrip("*").strip()
        if not ident:
            continue
        if ident not in seen:
            seen[ident] = raw_tp
            order.append(ident)
        elif seen[ident] != raw_tp:
            prev_head = seen[ident].split(":", 1)[0].split("=", 1)[0].strip()
            seen[ident] = head if prev_head == head else ident
    merged = [seen[ident] for ident in order]
    return [tp for tp in merged if not _has_type_param_default(tp)] + [
        tp for tp in merged if _has_type_param_default(tp)
    ]


def scope_matches_closure_mode(
    scope: Optional[Dict[str, Any]],
    expected_skip: bool,
    *,
    strict_with_suppression: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> bool:
    """Checks whether a precomputed scope dictionary matches the current schema and mode.

    Requires ``scope_schema_version == SCOPE_SCHEMA_VERSION`` and an explicit boolean
    ``skip_pre_unit_closures`` matching ``expected_skip`` so unmarked or legacy
    serialized scopes are never trusted without recomputation.
    """
    if scope is None:
        return False
    if scope.get("scope_schema_version") != SCOPE_SCHEMA_VERSION:
        return False
    if "skip_pre_unit_closures" not in scope:
        return False
    for flag_key, expected_val in (
        ("skip_pre_unit_closures", bool(expected_skip)),
        ("strict_with_suppression", strict_with_suppression),
    ):
        if expected_val is None:
            continue
        if flag_key in scope:
            raw_flag = scope.get(flag_key)
            if not (isinstance(raw_flag, bool) and raw_flag is bool(expected_val)):
                return False
        elif bool(expected_val) is not False:
            return False
    scope_non_sup = frozenset(
        normalize_non_suppressing_ctx_managers(
            scope.get("non_suppressing_ctx_managers")
        )
    )
    target_non_sup = frozenset(
        normalize_non_suppressing_ctx_managers(non_suppressing_ctx_managers)
    )
    if scope_non_sup != target_non_sup:
        return False
    return True


def ensure_unit_scope_closure_mode(
    unit: Dict[str, Any],
    scope: Optional[Dict[str, Any]],
    skip_pre_unit_closures: bool,
    *,
    strict_with_suppression: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
    source_text: Optional[str] = None,
    tree: Optional[ast.AST] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Returns scope if its schema version and closure mode match, else recomputes it.

    Unmarked or legacy scopes (missing ``scope_schema_version == SCOPE_SCHEMA_VERSION``)
    and scopes whose mode flags differ from the requested configuration are recomputed
    when unit source or AST is available, and fail closed when source is unavailable.

    Note:
        Scopes that already record terminal ``"pathological_ast"`` or ``"stale_source_code"``
        in ``control_flow_hazards`` are returned directly without re-parsing or mutating
        the terminal hazard dict, as they are unconditionally rejected downstream regardless
        of closure mode.
    """
    resolved_flags = resolve_unit_closure_flags(
        unit,
        skip_pre_unit_closures=bool(skip_pre_unit_closures),
        strict_with_suppression=strict_with_suppression,
    )
    eff_skip = resolved_flags.skip_pre_unit_closures
    eff_strict_with = resolved_flags.strict_with_suppression
    eff_non_sup = normalize_non_suppressing_ctx_managers(
        non_suppressing_ctx_managers,
        unit.get("non_suppressing_ctx_managers"),
    )
    if scope is not None and (
        scope_matches_closure_mode(
            scope,
            eff_skip,
            strict_with_suppression=eff_strict_with,
            non_suppressing_ctx_managers=eff_non_sup,
        )
        or any(
            h in scope.get("control_flow_hazards", ())
            for h in ("pathological_ast", "stale_source_code")
        )
    ):
        return scope
    u_copy = {
        **unit,
        "skip_pre_unit_closures": eff_skip,
        "strict_with_suppression": eff_strict_with,
    }
    u_copy.pop("closure_strictness", None)
    if eff_non_sup or "non_suppressing_ctx_managers" in u_copy:
        u_copy["non_suppressing_ctx_managers"] = list(eff_non_sup)
    if source_text is not None and not str(u_copy.get("source_text") or "").strip():
        u_copy["source_text"] = source_text
    if scope is not None:
        src_text_val = u_copy.get("source_text")
        src_code_val = u_copy.get("source_code")
        src_lines_val = u_copy.get("source_lines")
        has_unit_source = (
            tree is not None
            or (isinstance(src_text_val, str) and bool(src_text_val.strip()))
            or (isinstance(src_code_val, str) and bool(src_code_val.strip()))
            or (
                isinstance(src_lines_val, (list, tuple))
                and any(
                    isinstance(ln, str) and bool(ln.strip())
                    for ln in src_lines_val
                )
            )
            or bool(_load_unit_file_text(u_copy, repo_root=repo_root))
        )
        if not has_unit_source:
            logger.debug(
                "Precomputed scope for unit %r does not match active closure mode "
                "and backing source is unavailable; failing closed",
                unit.get("name"),
            )
            hazards = list(scope.get("control_flow_hazards", ()))
            if "unavailable_source_for_closure_mode" not in hazards:
                hazards.append("unavailable_source_for_closure_mode")
            if "severed_closure_cell" not in hazards:
                hazards.append("severed_closure_cell")
            reasons = list(scope.get("closure_hazard_reasons", ()))
            if "unavailable_source_for_closure_mode" not in reasons:
                reasons.append("unavailable_source_for_closure_mode")
            return {
                **scope,
                "control_flow_hazards": hazards,
                "closure_hazard_reasons": reasons,
                "is_control_flow_safe": False,
                "scope_schema_version": SCOPE_SCHEMA_VERSION,
                "skip_pre_unit_closures": eff_skip,
                "strict_with_suppression": eff_strict_with,
                "non_suppressing_ctx_managers": list(eff_non_sup),
                "heuristic_ctx_managers": list(
                    scope.get("heuristic_ctx_managers", ())
                ),
            }
        logger.debug(
            "Recomputing precomputed scope for unit %r to match active closure mode",
            unit.get("name"),
        )
    return inspect_single_unit_scope(u_copy, repo_root=repo_root, tree=tree)


def analyze_unit_variable_scope(
    u1: Dict[str, Any],
    u2: Optional[Dict[str, Any]] = None,
    repo_root: Optional[str] = None,
    tree1: Optional[ast.AST] = None,
    tree2: Optional[ast.AST] = None,
    scope1: Optional[Dict[str, Any]] = None,
    scope2: Optional[Dict[str, Any]] = None,
    *,
    skip_pre_unit_closures: Optional[bool] = None,
    closure_strictness: Optional[str] = None,
    strict_with_suppression: Optional[bool] = None,
    non_suppressing_ctx_managers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Analyzes AST variable scoping to determine inputs, outputs, closures, and attributes."""
    if u2 is not None:
        pair_flags = resolve_pair_closure_flags(
            u1,
            u2,
            closure_strictness=closure_strictness,
            skip_pre_unit_closures=skip_pre_unit_closures,
            strict_with_suppression=strict_with_suppression,
            non_suppressing_ctx_managers=non_suppressing_ctx_managers,
        )
        eff_skip = pair_flags.effective_skip_pre_unit_closures
        eff_strict_with = pair_flags.effective_strict_with_suppression
        info1 = ensure_unit_scope_closure_mode(
            u1,
            scope1,
            eff_skip,
            strict_with_suppression=eff_strict_with,
            non_suppressing_ctx_managers=pair_flags.unit1.non_suppressing_ctx_managers,
            repo_root=repo_root,
            tree=tree1,
        )
        info2 = ensure_unit_scope_closure_mode(
            u2,
            scope2,
            eff_skip,
            strict_with_suppression=eff_strict_with,
            non_suppressing_ctx_managers=pair_flags.unit2.non_suppressing_ctx_managers,
            repo_root=repo_root,
            tree=tree2,
        )
        inputs = _align_symmetric_var_list(info1["inputs"], info2["inputs"])
        outputs = _align_symmetric_var_list(info1["outputs"], info2["outputs"])
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
        has_yield_assignment = bool(
            info1.get("has_yield_assignment", False)
            or info2.get("has_yield_assignment", False)
        )
        has_try_or_with = bool(
            info1.get("has_try_or_with", False)
            or info2.get("has_try_or_with", False)
        )
        has_return = info1["has_return"] or info2["has_return"]
        has_return_value = bool(info1.get("has_return_value") or info2.get("has_return_value"))
        has_super = bool(info1.get("has_super", False) or info2.get("has_super", False))
        has_mangled = bool(
            info1.get("has_mangled_names", False) or info2.get("has_mangled_names", False)
        )
        local_imports = list(dict.fromkeys(info1["local_imports"] + info2["local_imports"]))
        local_imports_by_unit = [info1["local_imports"], info2["local_imports"]]
        yield_expr_names = info1["yield_expr_names"] + info2["yield_expr_names"]
        is_async = info1.get("is_async", False) or info2.get("is_async", False)
        conditional_outputs = list(dict.fromkeys(
            info1.get("conditional_outputs", []) + info2.get("conditional_outputs", [])
        ))
        definite_stores = sorted(
            set(info1.get("definite_stores", [])) & set(info2.get("definite_stores", []))
        )
        outputs_ambiguous = bool(
            info1.get("outputs_ambiguous", False) or info2.get("outputs_ambiguous", False)
        )
        type_params = _merge_type_params(
            info1.get("type_params", []), info2.get("type_params", [])
        )
        deleted_names = list(
            dict.fromkeys(
                info1.get("deleted_names", []) + info2.get("deleted_names", [])
            )
        )
    else:
        expected_skip1, expected_strict_with1 = resolve_unit_closure_flags(
            u1,
            closure_strictness=closure_strictness,
            skip_pre_unit_closures=skip_pre_unit_closures,
            strict_with_suppression=strict_with_suppression,
        )
        expected_non_sup1 = normalize_non_suppressing_ctx_managers(
            non_suppressing_ctx_managers,
            u1.get("non_suppressing_ctx_managers"),
        )
        info1 = ensure_unit_scope_closure_mode(
            u1,
            scope1,
            expected_skip1,
            strict_with_suppression=expected_strict_with1,
            non_suppressing_ctx_managers=expected_non_sup1,
            repo_root=repo_root,
            tree=tree1,
        )
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
        has_yield_assignment = bool(info1.get("has_yield_assignment", False))
        has_try_or_with = bool(info1.get("has_try_or_with", False))
        has_return = info1["has_return"]
        has_return_value = bool(info1.get("has_return_value"))
        has_super = bool(info1.get("has_super", False))
        has_mangled = bool(info1.get("has_mangled_names", False))
        local_imports = info1["local_imports"]
        local_imports_by_unit = [info1["local_imports"]]
        yield_expr_names = info1["yield_expr_names"]
        is_async = info1.get("is_async", False)
        conditional_outputs = info1.get("conditional_outputs", [])
        definite_stores = info1.get("definite_stores", [])
        outputs_ambiguous = bool(info1.get("outputs_ambiguous", False))
        type_params = list(info1.get("type_params", []))
        deleted_names = list(info1.get("deleted_names", []))

    has_instance_binding = info1.get("has_instance_binding", False) or (
        info2.get("has_instance_binding", False) if u2 is not None else False
    )
    has_class_binding = info1.get("has_class_binding", False) or (
        info2.get("has_class_binding", False) if u2 is not None else False
    )
    has_receiver_access = info1.get("has_receiver_access", False) or (
        info2.get("has_receiver_access", False) if u2 is not None else False
    )
    binding_kind = _determine_binding_kind(has_instance_binding, has_class_binding)

    primary_rec = (
        u1.get("receiver_param")
        or (u2.get("receiver_param") if u2 else None)
        or (
            "cls"
            if (
                u1.get("receiver_kind") == "class"
                or (u2 and u2.get("receiver_kind") == "class")
            )
            else None
        )
    )
    _normalize_receiver_order(
        inputs,
        has_instance_binding,
        has_class_binding,
        primary_receiver=primary_rec,
    )

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
        "has_yield_assignment": has_yield_assignment,
        "has_try_or_with": has_try_or_with,
        "has_return": has_return,
        "has_return_value": has_return_value,
        "has_super": has_super,
        "has_mangled_names": has_mangled,
        "local_imports": local_imports,
        "local_imports_by_unit": local_imports_by_unit,
        "yield_expr_names": yield_expr_names,
        "is_async": is_async,
        "conditional_outputs": conditional_outputs,
        "definite_stores": definite_stores,
        "has_instance_binding": has_instance_binding,
        "has_class_binding": has_class_binding,
        "binding_kind": binding_kind,
        "has_receiver_access": has_receiver_access,
        "instance_attrs": list(
            dict.fromkeys(
                info1.get("instance_attrs", [])
                + (info2.get("instance_attrs", []) if u2 is not None else [])
            )
        ),
        "class_attrs": list(
            dict.fromkeys(
                info1.get("class_attrs", [])
                + (info2.get("class_attrs", []) if u2 is not None else [])
            )
        ),
        "outputs_ambiguous": outputs_ambiguous,
        "type_params": type_params,
        "type_param_free_vars": list(
            dict.fromkeys(
                info1.get("type_param_free_vars", [])
                + (info2.get("type_param_free_vars", []) if u2 is not None else [])
            )
        ),
        "has_class_scoped_type_params": bool(
            info1.get("has_class_scoped_type_params", False)
            or (
                info2.get("has_class_scoped_type_params", False)
                if u2 is not None
                else False
            )
        ),
        "deleted_names": deleted_names,
        "closure_hazard_reasons": list(
            dict.fromkeys(
                info1.get("closure_hazard_reasons", [])
                + (
                    info2.get("closure_hazard_reasons", [])
                    if u2 is not None
                    else []
                )
            )
        ),
        "scope_schema_version": SCOPE_SCHEMA_VERSION,
        "skip_pre_unit_closures": bool(
            info1.get("skip_pre_unit_closures", False)
        )
        and (
            u2 is None
            or bool(info2.get("skip_pre_unit_closures", False))
        ),
        "strict_with_suppression": bool(
            info1.get("strict_with_suppression", False)
        )
        or (
            u2 is not None
            and bool(info2.get("strict_with_suppression", False))
        ),
        "non_suppressing_ctx_managers": sorted(
            set(info1.get("non_suppressing_ctx_managers", ()))
            & set(info2.get("non_suppressing_ctx_managers", ()))
            if u2 is not None
            else set(info1.get("non_suppressing_ctx_managers", ()))
        ),
        "heuristic_ctx_managers": sorted(
            set(info1.get("heuristic_ctx_managers", ()))
            | (
                set(info2.get("heuristic_ctx_managers", ()))
                if u2 is not None
                else set()
            )
        ),
    }
