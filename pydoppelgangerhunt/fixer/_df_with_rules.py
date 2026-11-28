"""Context-manager suppression rules, allowlists, _SuppressProvenance, and local exit proofs."""

from __future__ import annotations

import ast
from dataclasses import replace
import logging
from types import MappingProxyType
from typing import (
    AbstractSet,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

from pydoppelgangerhunt.config import (
    record_heuristic_non_suppressing_ctx_manager,
    should_warn_closure_event,
)
from pydoppelgangerhunt._ast_compat import is_match_node
from pydoppelgangerhunt.fixer._df_cfg import (
    _extract_import_bound_names,
    _extract_type_alias_or_pattern_name,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_single_node_local_names,
    _iter_shallow_scope_nodes,
    _node_has_star_import,
)
from pydoppelgangerhunt.fixer._df_exit_protocol import (
    _STDLIB_NON_SUPPRESSING_MODULES,
    scan_exit_protocol_patches,
)
from pydoppelgangerhunt.fixer._df_suppress_prov import (
    _EMPTY_SUPPRESS_PROVENANCE,
    _PATHLIB_CLASS_NAMES,
    _SAFE_EXIT_BASE_CLASS_NAMES,
    _SAFE_METHOD_DECORATORS,
    _SHADOWABLE_BUILTIN_CTX_NAMES,
    _SUPPRESSING_CTX_MANAGER_NAMES,
    _SuppressProvenance,
    _coerce_suppress_provenance,
    _extract_ctx_manager_call_name,
    _has_rebound_imported_factory,
    _has_rebound_parent_package,
    _ident_has_suspicious_suppress_token,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _MAX_AST_RECURSION_DEPTH,
    _extract_assigned_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes


logger = logging.getLogger(__name__)


def _clear_store_target_provenance(
    target: ast.AST,
    prov: _SuppressProvenance,
    *,
    is_conditional: bool = False,
) -> _SuppressProvenance:
    """Clears provenance for direct destructuring target names, merging conservatively when conditional."""
    for t_name in _extract_definite_target_names(target):
        cleared = prov.clear_name(t_name)
        prov = prov.merge_branch_conservative(cleared) if is_conditional else cleared
    return prov


def _is_suppress_alias_target_expr(
    expr: Optional[ast.AST],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if expr references a known suppressing context manager or alias."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    if isinstance(expr, ast.Name):
        return expr.id in _SUPPRESSING_CTX_MANAGER_NAMES or expr.id in prov.suppressors
    if isinstance(expr, ast.Attribute):
        return expr.attr in _SUPPRESSING_CTX_MANAGER_NAMES
    return False


class _ExtraAllowlistMatch(NamedTuple):
    """Outcome of matching a symbol or attribute expression against explicit context-manager allowlists."""

    matched: bool
    wildcard_pattern: Optional[str] = None


def _match_prefix_wildcard_pattern(
    qualified_name: str,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> Optional[str]:
    """Returns matching '<module>.*.<attr>' pattern for qualified_name if allowed, else None."""
    if not extra_non_suppressing_ctx_managers or "." not in qualified_name:
        return None
    dots = len(qualified_name) - len(qualified_name.lstrip("."))
    rest = qualified_name[dots:]
    parts = rest.split(".")
    if len(parts) < 3 or any(not seg for seg in parts):
        return None
    attr = parts[-1]
    if attr in _SUPPRESSING_CTX_MANAGER_NAMES:
        return None
    for split_idx in range(len(parts) - 2, 0, -1):
        prefix = "." * dots + ".".join(parts[:split_idx])
        pat = f"{prefix}.*.{attr}"
        if pat in extra_non_suppressing_ctx_managers:
            return pat
    return None


def _match_extra_non_suppressing_symbol(
    base_id: str,
    tail_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> _ExtraAllowlistMatch:
    """Matches a base Name + attribute tail against extra_non_suppressing_ctx_managers, distinguishing wildcards."""
    if (
        not extra_non_suppressing_ctx_managers
        or base_id in prov.suppressors
        or base_id in prov.shadowed_builtins
        or base_id in prov.shadowed_names
    ):
        return _ExtraAllowlistMatch(False)
    has_relative_base = any(
        alias == base_id and qual.startswith(".")
        for alias, qual in prov.imported_symbols
    )
    candidate = ".".join((base_id, *tail_parts))
    if (
        not prov.has_star_import
        and (not has_relative_base or not tail_parts)
        and candidate in extra_non_suppressing_ctx_managers
    ):
        return _ExtraAllowlistMatch(True)
    resolved_variants: List[Tuple[str, str]] = []
    for alias, qual in prov.imported_symbols:
        if alias == base_id:
            resolved = ".".join((qual, *tail_parts))
            stripped = resolved.lstrip(".")
            last_part = resolved.rsplit(".", 1)[-1]
            if (
                resolved in extra_non_suppressing_ctx_managers
                or (
                    stripped
                    and "." not in stripped
                    and stripped in extra_non_suppressing_ctx_managers
                )
                or (
                    resolved.startswith(".")
                    and last_part
                    and last_part not in _SUPPRESSING_CTX_MANAGER_NAMES
                    and f".{last_part}" in extra_non_suppressing_ctx_managers
                )
            ):
                return _ExtraAllowlistMatch(True)
            resolved_variants.append((resolved, last_part))
    for resolved, _ in resolved_variants:
        prefix_pat = _match_prefix_wildcard_pattern(
            resolved, extra_non_suppressing_ctx_managers
        )
        if prefix_pat is not None:
            return _ExtraAllowlistMatch(True, prefix_pat)
    if not prov.has_star_import and not has_relative_base:
        cand_prefix_pat = _match_prefix_wildcard_pattern(
            candidate, extra_non_suppressing_ctx_managers
        )
        if cand_prefix_pat is not None:
            return _ExtraAllowlistMatch(True, cand_prefix_pat)
    for resolved, last_part in resolved_variants:
        if (
            "." in resolved
            and last_part
            and last_part not in _SUPPRESSING_CTX_MANAGER_NAMES
            and f"*.{last_part}" in extra_non_suppressing_ctx_managers
        ):
            return _ExtraAllowlistMatch(True, f"*.{last_part}")
    return _ExtraAllowlistMatch(False)


def _matches_extra_non_suppressing_symbol(
    base_id: str,
    tail_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> bool:
    """Returns True if base_id (with optional attribute tail_parts) matches extra_non_suppressing_ctx_managers.

    Note:
        Entries prefixed with ``*.`` (e.g. ``"*.atomic"``) match any attribute chain or
        imported symbol ending in that suffix, and ``"<module>.*.<attr>"`` matches
        qualified symbols or attribute chains rooted under ``<module>``. A leading
        single-dot entry ``".Name"`` matches relative imports.
    """
    return _match_extra_non_suppressing_symbol(
        base_id, tail_parts, prov, extra_non_suppressing_ctx_managers
    ).matched


def _unwind_attribute_chain(
    expr: ast.Attribute,
) -> Tuple[ast.AST, str, Tuple[str, ...]]:
    """Unwinds an Attribute node into (root_value, terminal_attr, chain_parts)."""
    attr_name = expr.attr
    parts: List[str] = [attr_name]
    curr_val: ast.AST = expr.value
    while isinstance(curr_val, ast.Attribute):
        parts.append(curr_val.attr)
        curr_val = curr_val.value
    return curr_val, attr_name, tuple(reversed(parts))


def _name_matches_non_suppressing_set(
    fn_id: str,
    allowed_names: AbstractSet[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> bool:
    """Returns True when fn_id is not a suppressor/rebound factory and belongs to allowed_names or extra symbols."""
    if fn_id in prov.suppressors or fn_id in prov.suspicious_aliases:
        return False
    if _matches_extra_non_suppressing_symbol(
        fn_id, (), prov, extra_non_suppressing_ctx_managers
    ):
        return not _has_rebound_imported_factory(
            fn_id, prov, include_untargeted=False
        )
    if _has_rebound_imported_factory(fn_id, prov):
        return False
    return fn_id in allowed_names


def _collect_attr_rebind_target_names(
    recv_root: ast.Name,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
) -> Set[str]:
    """Collects canonical module/symbol names and aliases checked for attribute factory rebinds."""
    target_names: Set[str] = {recv_root.id}
    for alias, mod_name in prov.stdlib_modules:
        if alias == recv_root.id:
            target_names |= prov.module_aliases_for(mod_name)
    matched_qual = False
    for alias, qual in prov.imported_symbols:
        if alias == recv_root.id:
            matched_qual = True
            if len(chain_parts) >= 2:
                mod_candidate = ".".join((qual, *chain_parts[:-1]))
                target_names |= prov.module_aliases_for(mod_candidate) | {
                    chain_parts[-2]
                }
            elif qual:
                target_names |= prov.module_aliases_for(qual)
    if (
        len(chain_parts) >= 2
        and not matched_qual
        and recv_root.id not in prov.shadowed_names
    ):
        raw_mod_candidate = ".".join((recv_root.id, *chain_parts[:-1]))
        target_names |= prov.module_aliases_for(raw_mod_candidate) | {chain_parts[-2]}
    return target_names


def _is_attribute_factory_rebound(
    recv_root: ast.Name,
    attr_name: str,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
    *,
    include_untargeted: bool,
) -> bool:
    """Returns True when an attribute chain's receiver/module factory attribute was rebound."""
    target_names = _collect_attr_rebind_target_names(recv_root, chain_parts, prov)
    eff_untargeted = include_untargeted and not (
        attr_name == "open"
        and bool(target_names & frozenset({"builtins", "__builtins__", "io"}))
    )
    if prov.is_factory_rebound(
        target_names, attr_name, include_untargeted=eff_untargeted
    ):
        return True
    return any(
        _has_rebound_parent_package(t_name, prov)
        for t_name in target_names
        if "." in t_name
    )


def _extract_compound_expr_subnodes(
    node: ast.AST,
    *,
    include_named_target: bool = False,
    include_container_elts: bool = False,
    include_lambda_body: bool = False,
) -> Tuple[ast.AST, ...]:
    """Returns inner expression subnodes unwrapped from compound receiver wrappers."""
    if isinstance(node, (ast.Subscript, ast.Await)):
        return (node.value,)
    if isinstance(node, ast.NamedExpr):
        return (node.target, node.value) if include_named_target else (node.value,)
    if isinstance(node, ast.IfExp):
        return (node.body, node.orelse)
    if isinstance(node, ast.BoolOp):
        return tuple(node.values)
    if include_container_elts and isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return tuple(node.elts)
    if include_lambda_body and isinstance(node, ast.Lambda):
        return (node.body,)
    return ()


def _receiver_chain_has_unsafe_root(
    root_expr: ast.AST,
    attr_name: str,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True when a receiver chain roots in a suppressor, suspicious alias, or patched/rebound target."""
    if any(part in _SUPPRESSING_CTX_MANAGER_NAMES for part in chain_parts):
        return True
    worklist: List[ast.AST] = [root_expr]
    steps = 0
    while worklist and steps < 64:
        steps += 1
        curr = worklist.pop()
        if isinstance(curr, ast.Name):
            if (
                curr.id in prov.suppressors
                or curr.id in prov.suspicious_aliases
                or curr.id in prov.exit_patch_targets
                or curr.id in _SUPPRESSING_CTX_MANAGER_NAMES
                or _is_attribute_factory_rebound(
                    curr,
                    attr_name,
                    chain_parts,
                    prov,
                    include_untargeted=False,
                )
            ):
                return True
        elif isinstance(curr, ast.Attribute):
            inner_root, inner_attr, inner_parts = _unwind_attribute_chain(curr)
            if any(part in _SUPPRESSING_CTX_MANAGER_NAMES for part in inner_parts):
                return True
            if isinstance(inner_root, ast.Name) and _is_attribute_factory_rebound(
                inner_root,
                inner_attr,
                inner_parts,
                prov,
                include_untargeted=False,
            ):
                return True
            worklist.append(inner_root)
        elif isinstance(curr, ast.Call):
            worklist.append(curr.func)
        else:
            worklist.extend(
                _extract_compound_expr_subnodes(curr, include_lambda_body=True)
            )
    return steps >= 64


def _match_attribute_extra_non_suppressing(
    curr_val: ast.AST,
    attr_name: str,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> _ExtraAllowlistMatch:
    """Matches an Attribute chain against extra_non_suppressing_ctx_managers, preferring exact over wildcards."""
    if not extra_non_suppressing_ctx_managers or attr_name in _SUPPRESSING_CTX_MANAGER_NAMES:
        return _ExtraAllowlistMatch(False)
    if _receiver_chain_has_unsafe_root(curr_val, attr_name, chain_parts, prov):
        return _ExtraAllowlistMatch(False)
    if isinstance(curr_val, ast.Name):
        sym_match = _match_extra_non_suppressing_symbol(
            curr_val.id, chain_parts, prov, extra_non_suppressing_ctx_managers
        )
        if sym_match.matched:
            return sym_match
    if (
        not prov.has_star_import
        and f"*.{attr_name}" in extra_non_suppressing_ctx_managers
    ):
        return _ExtraAllowlistMatch(True, f"*.{attr_name}")
    return _ExtraAllowlistMatch(False)


def _attribute_matches_extra_non_suppressing(
    curr_val: ast.AST,
    attr_name: str,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> bool:
    """Returns True when an Attribute chain matches extra_non_suppressing_ctx_managers."""
    return _match_attribute_extra_non_suppressing(
        curr_val, attr_name, chain_parts, prov, extra_non_suppressing_ctx_managers
    ).matched


def _is_proven_non_suppressing_target_expr(
    expr: Optional[ast.AST],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if expr is a proven unshadowed non-suppressing context manager factory."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    if _is_builtin_or_io_open_target(expr, prov):
        return True
    if isinstance(expr, ast.Name):
        return _name_matches_non_suppressing_set(
            expr.id, prov.non_suppressors, prov, extra_non_suppressing_ctx_managers
        )
    if isinstance(expr, ast.Attribute):
        curr_val, attr_name, chain_parts = _unwind_attribute_chain(expr)
        if _attribute_matches_extra_non_suppressing(
            curr_val, attr_name, chain_parts, prov, extra_non_suppressing_ctx_managers
        ):
            return True
        if isinstance(curr_val, ast.Name) and len(chain_parts) >= 2 and not prov.has_exit_patch:
            for alias, qual in prov.imported_symbols:
                if alias == curr_val.id:
                    mod_candidate = ".".join((qual, *chain_parts[:-1]))
                    allowed_sub_attrs = _STDLIB_NON_SUPPRESSING_MODULES.get(
                        mod_candidate, frozenset()
                    )
                    if attr_name in allowed_sub_attrs:
                        target_names = prov.module_aliases_for(mod_candidate) | {
                            curr_val.id,
                            chain_parts[-2],
                        }
                        if not prov.is_factory_rebound(target_names, attr_name):
                            return True
        if isinstance(expr.value, ast.Name) and not prov.has_exit_patch:
            base_id = expr.value.id
            for alias, mod_name in prov.stdlib_modules:
                if alias == base_id:
                    allowed_attrs = _STDLIB_NON_SUPPRESSING_MODULES.get(
                        mod_name, frozenset()
                    )
                    if attr_name in allowed_attrs:
                        target_names = prov.module_aliases_for(mod_name) | {base_id}
                        if not prov.is_factory_rebound(target_names, attr_name):
                            return True
    return False


def _is_builtin_or_io_open_target(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True when expr resolves to unshadowed builtin open, builtins.open, or io.open."""
    if isinstance(expr, ast.Name):
        fn_id = expr.id
        if (
            fn_id == "open"
            and "open" not in prov.shadowed_builtins
            and "open" not in prov.shadowed_names
            and not prov.has_star_import
            and not prov.is_factory_rebound(
                prov.module_aliases_for("builtins")
                | prov.module_aliases_for("__builtins__"),
                "open",
                include_untargeted=False,
            )
        ):
            return True
        for alias, qual in prov.imported_symbols:
            if (
                alias == fn_id
                and fn_id not in prov.exit_patch_targets
                and fn_id not in prov.suspicious_aliases
                and qual in ("builtins.open", "__builtins__.open", "io.open")
            ):
                imp_mod = qual.rsplit(".", 1)[0]
                if not prov.is_factory_rebound(
                    prov.module_aliases_for(imp_mod),
                    "open",
                    include_untargeted=False,
                ):
                    return True
        return False
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.attr == "open"
    ):
        base_id = expr.value.id
        if (
            base_id in ("builtins", "__builtins__")
            and base_id not in prov.shadowed_builtins
            and base_id not in prov.shadowed_names
            and not prov.has_star_import
            and not prov.is_factory_rebound(
                prov.module_aliases_for("builtins")
                | prov.module_aliases_for("__builtins__"),
                "open",
                include_untargeted=False,
            )
        ):
            return True
        for alias, mod_name in prov.stdlib_modules:
            if alias == base_id and mod_name in ("builtins", "__builtins__", "io"):
                check_names = (
                    prov.module_aliases_for("builtins")
                    | prov.module_aliases_for("__builtins__")
                    if mod_name in ("builtins", "__builtins__")
                    else prov.module_aliases_for("io")
                ) | {base_id}
                if not prov.is_factory_rebound(
                    check_names, "open", include_untargeted=False
                ):
                    return True
    return False


def _is_proven_non_suppressing_call(
    expr: ast.AST,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if expr is a valid Call to a proven non-suppressing context manager factory."""
    if not isinstance(expr, ast.Call):
        return False
    prov = _coerce_suppress_provenance(suppress_aliases)
    if not _is_proven_non_suppressing_target_expr(
        expr.func,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    ):
        return False
    if _is_builtin_or_io_open_target(expr.func, prov):
        return bool(expr.args or expr.keywords)
    return True


def _is_proven_non_suppressing_instance_expr(
    expr: Optional[ast.AST],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if a bare Name or Attribute expression is a proven non-suppressing context manager instance."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    if isinstance(expr, ast.Name):
        return _name_matches_non_suppressing_set(
            expr.id, prov.non_suppressor_instances, prov, extra_non_suppressing_ctx_managers
        )
    if isinstance(expr, ast.Attribute) and extra_non_suppressing_ctx_managers:
        curr_val, attr_name, chain_parts = _unwind_attribute_chain(expr)
        return _attribute_matches_extra_non_suppressing(
            curr_val, attr_name, chain_parts, prov, extra_non_suppressing_ctx_managers
        )
    return False


def _is_pathlib_path_expr(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True when expr is a pathlib.Path class reference or constructor call."""
    target = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(target, ast.Name):
        if (
            target.id in _PATHLIB_CLASS_NAMES
            and target.id not in prov.non_suppressors
            and target.id not in prov.non_suppressor_instances
        ):
            return True
        for alias, qual in prov.imported_symbols:
            if alias == target.id and any(
                qual == f"pathlib.{cls}" for cls in _PATHLIB_CLASS_NAMES
            ):
                return True
        return False
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.attr in _PATHLIB_CLASS_NAMES
    ):
        base_id = target.value.id
        if base_id == "pathlib" and "pathlib" not in prov.non_suppressors:
            return True
        for alias, qual in prov.imported_symbols:
            if alias == base_id and qual == "pathlib":
                return True
    return False


_NON_CONTEXT_MANAGER_EXPR_TYPES = (
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Lambda,
    *_COMPREHENSION_TYPES,
)


def _is_invalid_with_context_expr(
    expr: ast.AST,
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True when a with-item context_expr is an uncalled factory, 0-arg open(), literal, or pathlib.Path."""
    while isinstance(expr, ast.NamedExpr):
        expr = expr.value
    if isinstance(expr, _NON_CONTEXT_MANAGER_EXPR_TYPES):
        return True
    if isinstance(expr, ast.Call):
        if (
            not expr.args
            and not expr.keywords
            and _is_builtin_or_io_open_target(expr.func, prov)
        ):
            return True
        if _is_pathlib_path_expr(
            expr, prov
        ) and not _is_proven_non_suppressing_target_expr(
            expr.func,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            return True
        return False
    if isinstance(expr, (ast.Name, ast.Attribute)):
        if _is_proven_non_suppressing_instance_expr(
            expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            return False
        if _is_proven_non_suppressing_target_expr(expr, prov) or _is_pathlib_path_expr(
            expr, prov
        ):
            return True
    return False


_CLASS_DISALLOWED_META_DUNDERS: frozenset[str] = frozenset(
    {
        "__new__",
        "__getattribute__",
        "__getattr__",
    }
)


_DEFERRED_DEF_SCOPE_TYPES: Tuple[type, ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
)


def _iter_shallow_scope_body_nodes(
    scope_node: ast.AST,
    *,
    deferred_scope_types: Tuple[type, ...] = _DEFERRED_DEF_SCOPE_TYPES,
) -> Iterator[ast.AST]:
    """Yields AST nodes in scope_node.body without descending into nested deferred_scope_types."""
    body_attr = getattr(scope_node, "body", ())
    if not isinstance(body_attr, (list, tuple)):
        return
    yield from _iter_shallow_scope_nodes(
        body_attr, deferred_scope_types=deferred_scope_types
    )


_DYNAMIC_CLASS_NAMESPACE_CALLS: frozenset[str] = frozenset(
    {"locals", "vars", "globals"}
)


_DYNAMIC_FRAME_ATTR_CALLS: frozenset[str] = frozenset({"_getframe", "currentframe"})


_DISALLOWED_ANNOTATION_NODE_TYPES: Tuple[type, ...] = (
    ast.Call,
    ast.Lambda,
    ast.NamedExpr,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)


def _is_static_class_attr_value(node: Optional[ast.AST]) -> bool:
    """Returns True if an expression is a side-effect-free constant or literal container."""
    stack: List[Optional[ast.AST]] = [node]
    while stack:
        curr = stack.pop()
        if curr is None or isinstance(curr, ast.Constant):
            continue
        if isinstance(curr, ast.UnaryOp) and isinstance(
            curr.op, (ast.UAdd, ast.USub, ast.Invert, ast.Not)
        ):
            stack.append(curr.operand)
            continue
        if isinstance(curr, (ast.Tuple, ast.List, ast.Set)):
            stack.extend(curr.elts)
            continue
        if isinstance(curr, ast.Dict):
            for k, v in zip(curr.keys, curr.values):
                if k is not None:
                    stack.append(k)
                stack.append(v)
            continue
        return False
    return True


_SAFE_EXIT_BASE_ALLOWED_QUALS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "object": frozenset({"builtins.object"}),
        "AbstractContextManager": frozenset({"contextlib.AbstractContextManager"}),
        "AsyncContextManager": frozenset(
            {"contextlib.AsyncContextManager", "typing.AsyncContextManager"}
        ),
        "ContextManager": frozenset({"typing.ContextManager"}),
    }
)


_SAFE_EXIT_ATTR_MODULES: frozenset[str] = frozenset({"contextlib", "typing"})


_BUILTIN_MODULE_ALIASES: frozenset[str] = frozenset({"builtins", "__builtins__"})


def _is_name_shadowed_or_non_suppressor_bound(
    name: str, prov: _SuppressProvenance
) -> bool:
    """Returns True when star-import, shadowing, or non-suppressor binding invalidates a bare name."""
    return (
        prov.has_star_import
        or name in prov.shadowed_names
        or name in prov.shadowed_builtins
        or name in prov.non_suppressors
        or name in prov.non_suppressor_instances
    )


def _is_unshadowed_builtin_symbol(
    name: str,
    allowed_names: AbstractSet[str],
    prov: _SuppressProvenance,
) -> bool:
    """Checks that a bare builtin symbol is unshadowed and not rebound by any import or def."""
    if (
        name not in allowed_names
        or _is_name_shadowed_or_non_suppressor_bound(name, prov)
        or any(
            alias == name and mod_name != "builtins"
            for alias, mod_name in prov.stdlib_modules
        )
    ):
        return False
    allowed_imps = (
        {"builtins", "builtins.builtins"}
        if name in ("builtins", "__builtins__")
        else {f"builtins.{name}"}
    )
    return not any(
        alias == name and qual not in allowed_imps
        for alias, qual in prov.imported_symbols
    )


_TRACKED_FACTORY_MOD_NAMES: frozenset[str] = frozenset(
    _STDLIB_NON_SUPPRESSING_MODULES.keys()
) | frozenset({"contextlib", "builtins"})


def _resolve_unshadowed_stdlib_module_name(
    ident: str,
    allowed_mods: AbstractSet[str],
    prov: _SuppressProvenance,
    *,
    implicit_mods: AbstractSet[str] = frozenset(),
) -> Optional[str]:
    """Resolves ident to a verified stdlib module name in allowed_mods without implicit bypass."""
    bound_mods = {
        mod_name for alias, mod_name in prov.stdlib_modules if alias == ident
    }
    if bound_mods:
        if len(bound_mods) == 1:
            resolved = next(iter(bound_mods))
            if resolved in allowed_mods:
                return resolved
        return None
    if (
        ident not in implicit_mods
        or _is_name_shadowed_or_non_suppressor_bound(ident, prov)
        or any(alias == ident for alias, _ in prov.imported_symbols)
    ):
        return None
    return ident


def _is_unshadowed_exit_base_name(name: str, prov: _SuppressProvenance) -> bool:
    """Checks that a bare class base Name is a genuine stdlib/builtin safe base class."""
    allowed_quals = _SAFE_EXIT_BASE_ALLOWED_QUALS.get(name)
    if (
        allowed_quals is None
        or _is_name_shadowed_or_non_suppressor_bound(name, prov)
        or any(alias == name for alias, _ in prov.stdlib_modules)
    ):
        return False
    return not any(
        alias == name and qual not in allowed_quals
        for alias, qual in prov.imported_symbols
    )


def _annotation_is_call_free(node: Optional[ast.AST]) -> bool:
    """Returns True if a type annotation expression contains no calls, lambdas, walrus, or comprehensions."""
    return node is None or not any(
        isinstance(sub, _DISALLOWED_ANNOTATION_NODE_TYPES) for sub in ast.walk(node)
    )


def _method_signature_is_side_effect_free(
    stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> bool:
    """Returns True when method defaults are static and annotations contain no calls or comprehensions."""
    defaults_ok = all(
        _is_static_class_attr_value(val) for val in stmt.args.defaults
    ) and all(
        val is None or _is_static_class_attr_value(val)
        for val in stmt.args.kw_defaults
    )
    if not defaults_ok:
        return False
    arg_nodes: List[Optional[ast.arg]] = [
        *stmt.args.posonlyargs,
        *stmt.args.args,
        stmt.args.vararg,
        *stmt.args.kwonlyargs,
        stmt.args.kwarg,
    ]
    return _annotation_is_call_free(stmt.returns) and all(
        _annotation_is_call_free(arg.annotation)
        for arg in arg_nodes
        if arg is not None
    )


def _is_safe_non_suppressing_class_body_stmt(
    stmt: ast.stmt,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if a class-body statement belongs to the strict syntactic subset for proven non-suppressing classes."""
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not _method_signature_is_side_effect_free(stmt):
            return False
        if stmt.name in ("__exit__", "__aexit__"):
            return not stmt.decorator_list
        return all(
            isinstance(dec, ast.Name)
            and _is_unshadowed_builtin_symbol(dec.id, _SAFE_METHOD_DECORATORS, prov)
            for dec in stmt.decorator_list
        )
    if isinstance(stmt, ast.Assign):
        if not stmt.targets or not _is_static_class_attr_value(stmt.value):
            return False
        return all(
            isinstance(tgt, ast.Name) and not tgt.id.startswith("__")
            for tgt in stmt.targets
        )
    if isinstance(stmt, ast.AnnAssign):
        return (
            isinstance(stmt.target, ast.Name)
            and not stmt.target.id.startswith("__")
            and _is_static_class_attr_value(stmt.value)
            and _annotation_is_call_free(stmt.annotation)
        )
    return False


def _is_dynamic_namespace_call(node: ast.AST, call_names: frozenset[str]) -> bool:
    """Returns True if node is a call to a dynamic reflection/mutation function or frame getter."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name) and node.func.id in call_names:
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _DYNAMIC_FRAME_ATTR_CALLS
    )


def _class_def_has_dynamic_exit_mutation(cls_node: ast.ClassDef) -> bool:
    """Returns True if a ClassDef contains dynamic namespace reflection or runtime __exit__/__aexit__ mutation.

    Delegates direct exit-protocol patching (``__exit__``/``__aexit__``/``__class__``/``__bases__``,
    ``exec``/``eval``, and ``setattr``/``delattr``) to ``scan_exit_protocol_patches`` so there is
    a single source of truth, and layers only class-local namespace reflection (``locals``/``vars``/
    ``globals``), frame getters, and ``__dict__`` mutations on top.
    """
    if scan_exit_protocol_patches(cls_node).has_patch:
        return True
    if any(
        _is_dynamic_namespace_call(sub, _DYNAMIC_CLASS_NAMESPACE_CALLS)
        for sub in _iter_shallow_scope_body_nodes(cls_node)
    ):
        return True
    return any(
        _is_dynamic_namespace_call(node, frozenset())
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr == "__dict__"
        )
        for node in ast.walk(cls_node)
    )


def _class_def_may_suppress_in_exit(
    cls_node: ast.ClassDef,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if a ClassDef inherits from a suppressor or defines/binds an __exit__/__aexit__ that may return truthy."""
    if _class_def_has_dynamic_exit_mutation(cls_node):
        return True
    for b in cls_node.bases:
        base_target = b.value if isinstance(b, ast.Subscript) else b
        if _is_suppress_alias_target_expr(base_target, prov) or (
            isinstance(base_target, ast.Call)
            and _is_suppress_alias_target_expr(base_target.func, prov)
        ):
            return True
    for sub in _iter_shallow_scope_body_nodes(cls_node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            sub.name in ("__exit__", "__aexit__")
        ):
            if sub.decorator_list:
                return True
            if sub.name == "__exit__" and isinstance(sub, ast.AsyncFunctionDef):
                return True
            for m_node in _iter_shallow_scope_body_nodes(sub):
                if isinstance(m_node, (ast.Yield, ast.YieldFrom)):
                    return True
                if isinstance(m_node, ast.Return) and m_node.value is not None:
                    if not (
                        isinstance(m_node.value, ast.Constant)
                        and not bool(m_node.value.value)
                    ):
                        return True
        elif (
            isinstance(sub, ast.Name)
            and isinstance(sub.ctx, (ast.Store, ast.Del))
            and sub.id in ("__exit__", "__aexit__")
        ):
            return True
        elif isinstance(sub, (ast.Import, ast.ImportFrom)) and any(
            bound in ("__exit__", "__aexit__")
            for bound in _extract_import_bound_names(sub)
        ):
            return True
        elif (
            isinstance(sub, (ast.ClassDef, ast.ExceptHandler))
            and getattr(sub, "name", None) in ("__exit__", "__aexit__")
        ):
            return True
        elif _extract_type_alias_or_pattern_name(sub) in ("__exit__", "__aexit__"):
            return True
    return False


def _is_safe_context_manager_base_expr(
    base: ast.AST,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if a class base expression is an unshadowed object or abstract context-manager protocol base."""
    if prov.has_star_import:
        return False
    if isinstance(base, ast.Subscript):
        base = base.value
    if isinstance(base, ast.Name):
        return _is_unshadowed_exit_base_name(base.id, prov)
    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
        return (
            _resolve_unshadowed_stdlib_module_name(
                base.value.id,
                _SAFE_EXIT_ATTR_MODULES,
                prov,
                implicit_mods=_SAFE_EXIT_ATTR_MODULES,
            )
            is not None
            and base.attr in _SAFE_EXIT_BASE_CLASS_NAMES
        )
    return False


# INVARIANT (CONVERGENCE BOUNDARY — LOCAL CLASS EXIT PROOF):
# Positive proof that a locally defined ClassDef cannot suppress exceptions is
# intentionally limited to strictly static, undecorated classes inheriting only
# from unshadowed object / abstract context-manager bases with a single static
# ``__exit__``/``__aexit__`` whose returns are falsy constants and no dynamic
# namespace or exit-protocol mutation. Do NOT extend this proof with MRO
# resolution across imported base classes, decorator execution modeling, or
# interprocedural method-body abstract interpretation; unprovable local classes
# must fail closed (or be declared via ``non_suppressing_ctx_managers``).
def _class_def_is_proven_non_suppressing_in_exit(
    cls_node: ast.ClassDef,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if a ClassDef defines __exit__/__aexit__ that only returns falsy constants and has a strictly static class body."""
    if (
        cls_node.decorator_list
        or cls_node.keywords
        or prov.has_star_import
        or not all(_is_safe_context_manager_base_expr(b, prov) for b in cls_node.bases)
        or not all(
            _is_safe_non_suppressing_class_body_stmt(s, prov) for s in cls_node.body
        )
    ):
        return False
    cls_locals = _extract_single_node_local_names(cls_node)
    if bool(
        cls_locals & _CLASS_DISALLOWED_META_DUNDERS
    ) or _scope_has_direct_star_import(cls_node):
        return False
    top_exit_defs = [
        item
        for item in cls_node.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "__exit__"
        and not item.decorator_list
    ]
    top_aexit_defs = [
        item
        for item in cls_node.body
        if isinstance(item, ast.AsyncFunctionDef)
        and item.name == "__aexit__"
        and not item.decorator_list
    ]
    all_exit_fn_defs = [
        sub
        for sub in _iter_shallow_scope_body_nodes(cls_node)
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
        and sub.name in ("__exit__", "__aexit__")
    ]
    if (
        len(top_exit_defs) > 1
        or len(top_aexit_defs) > 1
        or (len(top_exit_defs) + len(top_aexit_defs)) == 0
        or len(all_exit_fn_defs) != (len(top_exit_defs) + len(top_aexit_defs))
    ):
        return False
    return not _class_def_may_suppress_in_exit(cls_node, prov)


class _ShallowAstFlowSummary(NamedTuple):
    """Shallow iterative AST summary used by depth-limit fallback handlers."""

    assigned: Set[str]
    deleted: Set[str]
    except_names: Set[str]
    has_star_import: bool
    has_dynamic_store: bool
    has_break: bool
    has_continue: bool


def _record_shallow_extra_binding(
    curr: ast.AST,
    assigned: Set[str],
    except_names: Set[str],
) -> None:
    """Collect ExceptHandler, TypeAlias, and pattern bindings during shallow AST traversal."""
    if isinstance(curr, ast.ExceptHandler):
        if isinstance(curr.name, str) and curr.name:
            except_names.add(curr.name)
        return
    bound = _extract_type_alias_or_pattern_name(curr)
    if bound is not None:
        assigned.add(bound)


def _push_shallow_ast_children(
    curr: ast.AST,
    in_nested_loop: bool,
    stack: List[Tuple[ast.AST, bool]],
    assigned: Set[str],
) -> None:
    """Push shallow child expressions/headers onto stack without descending into inner scope bodies."""
    if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        assigned.add(curr.name)
        stack.extend((h, in_nested_loop) for h in iter_nested_scope_header_nodes(curr))
        return
    if isinstance(curr, ast.Lambda):
        stack.extend((h, in_nested_loop) for h in iter_nested_scope_header_nodes(curr))
        return
    if isinstance(curr, _COMPREHENSION_TYPES):
        for gen in curr.generators:
            stack.append((gen.iter, in_nested_loop))
            stack.extend((if_e, in_nested_loop) for if_e in gen.ifs)
        elts: Tuple[ast.AST, ...] = (
            (curr.key, curr.value) if isinstance(curr, ast.DictComp) else (curr.elt,)
        )
        stack.extend((el, in_nested_loop) for el in elts)
        return
    next_in_loop = in_nested_loop or isinstance(
        curr, (ast.For, ast.AsyncFor, ast.While)
    )
    stack.extend((ch, next_in_loop) for ch in ast.iter_child_nodes(curr))


def _collect_shallow_ast_flow_summary(
    nodes: Iterable[ast.AST],
) -> _ShallowAstFlowSummary:
    """Iteratively collect shallow assignments, deletions, imports, and loop exits across nodes."""
    assigned: Set[str] = set()
    deleted: Set[str] = set()
    except_names: Set[str] = set()
    flags = [False, False, False, False]  # [star_import, dynamic_store, break, continue]
    stack: List[Tuple[ast.AST, bool]] = [(s, False) for s in nodes]
    while stack:
        curr, in_nested_loop = stack.pop()
        if isinstance(curr, (ast.Break, ast.Continue)):
            if not in_nested_loop:
                flags[2 if isinstance(curr, ast.Break) else 3] = True
            continue
        if isinstance(curr, ast.Name):
            if isinstance(curr.ctx, ast.Store):
                assigned.add(curr.id)
            elif isinstance(curr.ctx, ast.Del):
                deleted.add(curr.id)
            continue
        if isinstance(curr, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "*" for alias in curr.names):
                flags[0] = True
            assigned.update(_extract_import_bound_names(curr))
            continue
        if isinstance(curr, (ast.Attribute, ast.Subscript)) and isinstance(
            curr.ctx, (ast.Store, ast.Del)
        ):
            flags[1] = True
        _record_shallow_extra_binding(curr, assigned, except_names)
        _push_shallow_ast_children(curr, in_nested_loop, stack, assigned)
    return _ShallowAstFlowSummary(
        assigned=assigned,
        deleted=deleted,
        except_names=except_names,
        has_star_import=flags[0],
        has_dynamic_store=flags[1],
        has_break=flags[2],
        has_continue=flags[3],
    )


def _conservatively_update_prov_at_depth_limit(
    nodes: Iterable[ast.AST],
    prov: _SuppressProvenance,
) -> _SuppressProvenance:
    """Fail-closed provenance update using iterative traversal past the AST depth limit."""
    summary = _collect_shallow_ast_flow_summary(nodes)
    touched = summary.assigned | summary.deleted | summary.except_names
    shad_b = frozenset(n for n in touched if n in _SHADOWABLE_BUILTIN_CTX_NAMES)
    return replace(
        prov,
        suppressors=prov.suppressors | summary.assigned,
        non_suppressors=frozenset(),
        non_suppressor_instances=frozenset(),
        stdlib_modules=frozenset(),
        imported_symbols=frozenset(),
        shadowed_builtins=prov.shadowed_builtins | shad_b,
        shadowed_names=prov.shadowed_names | touched,
        has_star_import=prov.has_star_import or summary.has_star_import,
        has_exit_patch=prov.has_exit_patch or summary.has_dynamic_store,
    )


def _extract_shallow_deleted_names(node: ast.AST) -> Set[str]:
    """Extracts variable names deleted or unbound by ExceptHandler in node outside inner scopes."""
    deleted: Set[str] = set()
    stack: List[ast.AST] = [node]
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            stack.extend(iter_nested_scope_header_nodes(curr))
            continue
        if isinstance(curr, ast.Name) and isinstance(curr.ctx, ast.Del):
            deleted.add(curr.id)
        elif isinstance(curr, ast.ExceptHandler) and isinstance(curr.name, str):
            deleted.add(curr.name)
        stack.extend(ast.iter_child_nodes(curr))
    return deleted


def _resolve_tracked_factory_qualname(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> Optional[str]:
    """Returns canonical '<module>.<attr>' qualname if expr resolves to a tracked stdlib/suppress factory."""
    if isinstance(expr, ast.Name):
        fn_id = expr.id
        for alias, qual in prov.imported_symbols:
            if alias == fn_id and qual:
                return qual
        if (
            fn_id not in prov.shadowed_names
            and not prov.has_star_import
            and not any(alias == fn_id for alias, _ in prov.stdlib_modules)
        ):
            if fn_id == "nullcontext" and fn_id in prov.non_suppressors:
                return "contextlib.nullcontext"
            if fn_id == "open" and "open" not in prov.shadowed_builtins:
                return "builtins.open"
    elif isinstance(expr, ast.Attribute):
        curr_val, attr_name, chain_parts = _unwind_attribute_chain(expr)
        if isinstance(curr_val, ast.Name):
            if len(chain_parts) == 1:
                resolved_mod = _resolve_unshadowed_stdlib_module_name(
                    curr_val.id,
                    _TRACKED_FACTORY_MOD_NAMES,
                    prov,
                )
                if resolved_mod is not None:
                    if attr_name in _STDLIB_NON_SUPPRESSING_MODULES.get(
                        resolved_mod, ()
                    ):
                        return f"{resolved_mod}.{attr_name}"
                    if resolved_mod == "contextlib" and attr_name == "suppress":
                        return "contextlib.suppress"
                    if resolved_mod == "builtins" and attr_name == "open":
                        return "builtins.open"
                if (
                    curr_val.id in _BUILTIN_MODULE_ALIASES
                    and _is_unshadowed_builtin_symbol(
                        curr_val.id, _BUILTIN_MODULE_ALIASES, prov
                    )
                    and attr_name == "open"
                ):
                    return f"{curr_val.id}.open"
            if len(chain_parts) >= 2:
                for alias, qual in prov.imported_symbols:
                    if alias == curr_val.id:
                        mod_candidate = ".".join((qual, *chain_parts[:-1]))
                        if attr_name in _STDLIB_NON_SUPPRESSING_MODULES.get(
                            mod_candidate, ()
                        ) or (
                            mod_candidate == "contextlib" and attr_name == "suppress"
                        ):
                            return f"{mod_candidate}.{attr_name}"
    return None


_UNPACK_PASSTHROUGH_CALL_NAMES: frozenset[str] = frozenset(
    {"tuple", "list", "iter", "reversed", "set", "frozenset"}
)


def _is_direct_suppressor_expr(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True if expr is a direct suppressor reference or call to a suppressor factory."""
    return _is_suppress_alias_target_expr(expr, prov) or (
        isinstance(expr, ast.Call) and _is_suppress_alias_target_expr(expr.func, prov)
    )


def _match_sequence_unpack_pairs(
    target: ast.AST,
    value: Optional[ast.AST],
    *,
    allow_starred_target: bool = False,
) -> Optional[List[Tuple[ast.AST, ast.AST]]]:
    """Returns left-to-right (target_elt, value_elt) pairs when sequence unpack shapes match."""
    if not (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and not any(isinstance(e, ast.Starred) for e in value.elts)
    ):
        return None
    star_indices = [
        idx for idx, elt in enumerate(target.elts) if isinstance(elt, ast.Starred)
    ]
    if not star_indices:
        if len(target.elts) != len(value.elts):
            return None
        return list(zip(target.elts, value.elts))
    if (
        not allow_starred_target
        or len(star_indices) != 1
        or len(value.elts) < len(target.elts) - 1
    ):
        return None
    s_idx = star_indices[0]
    suffix_len = len(target.elts) - s_idx - 1
    mid_end = len(value.elts) - suffix_len
    star_elt = cast(ast.Starred, target.elts[s_idx])
    mid_list = ast.List(elts=list(value.elts[s_idx:mid_end]), ctx=ast.Load())
    pairs: List[Tuple[ast.AST, ast.AST]] = list(
        zip(target.elts[:s_idx], value.elts[:s_idx])
    )
    pairs.append((star_elt.value, mid_list))
    if suffix_len > 0:
        pairs.extend(zip(target.elts[s_idx + 1 :], value.elts[mid_end:]))
    return pairs


def _unpack_fallback_rhs_is_suspicious(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True if an unmatched unpacking RHS carries a suppressor or suspicious manager."""
    if expr is None:
        return False
    stack: List[ast.AST] = [expr]
    while stack:
        sub = stack.pop()
        while isinstance(sub, ast.NamedExpr):
            sub = sub.value
        if _is_direct_suppressor_expr(sub, prov) or _looks_like_suppressing_ctx_manager(
            sub, prov
        ):
            return True
        if isinstance(sub, ast.Starred):
            stack.append(sub.value)
        elif isinstance(sub, (ast.Tuple, ast.List, ast.Set)):
            stack.extend(sub.elts)
        elif isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Add, ast.Mult)):
            stack.extend((sub.left, sub.right))
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            stack.append(sub.elt)
        elif isinstance(sub, ast.Dict):
            stack.extend(sub.values)
        elif isinstance(sub, ast.IfExp):
            stack.extend((sub.body, sub.orelse))
        elif isinstance(sub, ast.BoolOp):
            stack.extend(sub.values)
        elif isinstance(sub, ast.Call):
            if (
                isinstance(sub.func, ast.Name)
                and sub.func.id in _UNPACK_PASSTHROUGH_CALL_NAMES
            ):
                stack.extend(sub.args)
            elif any(
                _is_direct_suppressor_expr(arg, prov)
                for arg in (*sub.args, *(kw.value for kw in sub.keywords))
            ):
                return True
    return False


def _has_shallow_unproven_call_or_await(
    nodes: Iterable[ast.AST],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if any node in nodes is an Await or an unproven Call."""
    for sub in nodes:
        if isinstance(sub, ast.Await):
            return True
        if isinstance(sub, ast.Call) and not _is_proven_non_suppressing_call(
            sub,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            return True
    return False


def _exprs_may_invoke_cross_scope_mutator(
    roots: Iterable[ast.AST],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if shallow expressions contain Await or an unproven Call."""
    if not prov.cross_scope_mutated:
        return False
    return _has_shallow_unproven_call_or_await(
        _iter_shallow_scope_nodes(
            tuple(roots), deferred_scope_types=_DEFERRED_DEF_SCOPE_TYPES
        ),
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _scope_has_direct_star_import(scope_node: ast.AST) -> bool:
    """Checks whether scope_node contains a wildcard 'from ... import *' outside nested scopes."""
    if isinstance(scope_node, ast.Module):
        return _node_has_star_import(scope_node)
    body_attr = getattr(scope_node, "body", ())
    if not isinstance(body_attr, (list, tuple)):
        return False
    stack: List[ast.AST] = list(body_attr)
    while stack:
        curr = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            continue
        if isinstance(curr, ast.ImportFrom) and any(
            alias.name == "*" for alias in curr.names
        ):
            return True
        stack.extend(ast.iter_child_nodes(curr))
    return False


def _classify_with_item_suppression(
    expr: ast.AST,
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> str:
    """Classifies a single with-item context_expr as 'suppressing', 'non_suppressing', or 'unknown'."""
    while isinstance(expr, ast.NamedExpr):
        expr = expr.value
    call_name = _extract_ctx_manager_call_name(expr)
    if call_name is not None and (
        call_name in _SUPPRESSING_CTX_MANAGER_NAMES or call_name in prov.suppressors
    ):
        return "suppressing"
    if isinstance(expr, ast.Name) and expr.id in prov.suppressors:
        return "suppressing"
    if _is_proven_non_suppressing_call(
        expr,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    ) or (
        isinstance(expr, (ast.Name, ast.Attribute))
        and _is_proven_non_suppressing_instance_expr(
            expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    ):
        return "non_suppressing"
    return "unknown"


def _looks_like_suppressing_ctx_manager(
    expr: ast.AST,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if an unknown context manager callee/target has a suppressor-like name or shadowed builtin.

    Attribute chains only inspect the terminal attribute name (plus proven suppressor
    roots in ``prov.suppressors``) so intermediate receiver attributes such as
    ``self.retry_budget.acquire()`` or ``svc.skip_list.lock`` do not trigger false
    positives.
    """
    prov = _coerce_suppress_provenance(suppress_aliases)
    call_name = _extract_ctx_manager_call_name(expr)
    if call_name == "open" and (
        "open" in prov.shadowed_builtins
        or "open" in prov.shadowed_names
        or prov.has_star_import
        or prov.is_factory_rebound(
            prov.module_aliases_for("builtins")
            | prov.module_aliases_for("__builtins__"),
            "open",
            include_untargeted=False,
        )
    ):
        return True
    stack: List[ast.AST] = [expr]
    while stack:
        sub = stack.pop()
        ident: Optional[str] = None
        if isinstance(sub, ast.Name):
            ident = sub.id
            if ident in prov.suppressors or ident in prov.suspicious_aliases:
                return True
            if _has_rebound_imported_factory(ident, prov):
                return True
            if (
                ident not in prov.shadowed_names
                and ident not in prov.shadowed_builtins
            ):
                for alias, qual in prov.imported_symbols:
                    if alias == ident and qual:
                        q_tail = qual.rsplit(".", 1)[-1]
                        if (
                            q_tail != "catch_warnings"
                            and _ident_has_suspicious_suppress_token(q_tail)
                        ):
                            return True
        elif isinstance(sub, ast.Attribute):
            ident = sub.attr
            recv_root, _, chain_parts = _unwind_attribute_chain(sub)
            if isinstance(recv_root, ast.Name) and (
                recv_root.id in prov.suppressors
                or recv_root.id in prov.suspicious_aliases
            ):
                return True
            if isinstance(recv_root, ast.Name) and _is_attribute_factory_rebound(
                recv_root, ident, chain_parts, prov, include_untargeted=True
            ):
                return True
        elif isinstance(sub, ast.Call):
            stack.append(sub.func)
            is_enter_ctx = (
                isinstance(sub.func, ast.Name)
                and sub.func.id in ("enter_context", "enter_async_context")
            ) or (
                isinstance(sub.func, ast.Attribute)
                and sub.func.attr in ("enter_context", "enter_async_context")
            )
            if is_enter_ctx:
                stack.extend(sub.args)
                stack.extend(kw.value for kw in sub.keywords)
        else:
            stack.extend(
                _extract_compound_expr_subnodes(
                    sub,
                    include_named_target=True,
                    include_container_elts=True,
                )
            )
        if ident is not None and ident != "catch_warnings":
            if _ident_has_suspicious_suppress_token(ident):
                return True
    return False


def _find_ctx_manager_wildcard_match(
    expr: ast.AST,
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> Optional[str]:
    """Returns the matching wildcard pattern when expr relied on a wildcard allowlist entry."""
    if not any("*" in pat for pat in extra_non_suppressing_ctx_managers):
        return None
    while isinstance(expr, ast.NamedExpr):
        expr = expr.value
    non_wildcards = frozenset(
        pat for pat in extra_non_suppressing_ctx_managers if "*" not in pat
    )
    if (
        _classify_with_item_suppression(
            expr,
            prov,
            extra_non_suppressing_ctx_managers=non_wildcards,
        )
        == "non_suppressing"
    ):
        return None
    target = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(target, ast.Name):
        return _match_extra_non_suppressing_symbol(
            target.id, (), prov, extra_non_suppressing_ctx_managers
        ).wildcard_pattern
    if isinstance(target, ast.Attribute):
        cur_val, _, ch_parts = _unwind_attribute_chain(target)
        return _match_attribute_extra_non_suppressing(
            cur_val, ch_parts[-1], ch_parts, prov, extra_non_suppressing_ctx_managers
        ).wildcard_pattern
    return None


class _WithItemProvenanceStep(NamedTuple):
    """Per-item suppression provenance state produced while stepping through a with header."""

    item: ast.withitem
    after_prior_sup: bool
    post_expr_prov: _SuppressProvenance
    item_may_suppress: bool
    opt_vars_conditional: bool
    post_item_prov: _SuppressProvenance
    item_is_proven_non_suppressing: bool = False
    item_mode: str = "unknown"
    is_invalid_ctx: bool = False
    item_looks_suppressing: bool = False


def _resolve_unique_imported_qualname(
    base_id: str,
    prov: _SuppressProvenance,
) -> Optional[str]:
    """Returns the unique imported qualname for base_id when unshadowed, or None."""
    if (
        base_id in prov.shadowed_names
        or base_id in prov.shadowed_builtins
    ):
        return None
    matches = sorted(
        {qual for alias, qual in prov.imported_symbols if alias == base_id and qual}
    )
    return matches[0] if len(matches) == 1 else None


def _format_ctx_manager_display_name(
    expr: ast.AST,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> str:
    """Formats a context-manager expression into an unambiguous qualified display name."""
    while isinstance(expr, (ast.NamedExpr, ast.Await)):
        expr = expr.value
    target = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(target, ast.Name):
        return _resolve_unique_imported_qualname(target.id, prov) or target.id
    if isinstance(target, ast.Attribute):
        curr_val, _, chain_parts = _unwind_attribute_chain(target)
        if isinstance(curr_val, ast.Name):
            root_qual = (
                _resolve_unique_imported_qualname(curr_val.id, prov) or curr_val.id
            )
            return ".".join((root_qual, *chain_parts))
        if isinstance(curr_val, ast.Call):
            if isinstance(curr_val.func, ast.Name):
                fn_qual = (
                    _resolve_unique_imported_qualname(curr_val.func.id, prov)
                    or curr_val.func.id
                )
                return f"{fn_qual}().{'.'.join(chain_parts)}"
            if isinstance(curr_val.func, ast.Attribute):
                r_val, _, r_parts = _unwind_attribute_chain(curr_val.func)
                if isinstance(r_val, ast.Name):
                    r_qual = (
                        _resolve_unique_imported_qualname(r_val.id, prov) or r_val.id
                    )
                    return f"{'.'.join((r_qual, *r_parts))}().{'.'.join(chain_parts)}"
        return f"*.{'.'.join(chain_parts)}"
    return "<expr>"


def _warn_heuristic_non_suppressing_ctx_manager(
    expr: ast.AST,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> None:
    """Logs a deduplicated info message when an unproven context manager is treated as non-suppressing via name heuristic."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    mgr_name = _format_ctx_manager_display_name(expr, prov)
    record_heuristic_non_suppressing_ctx_manager(mgr_name)
    if should_warn_closure_event(f"heuristic_safe_ctx:{mgr_name}"):
        logger.info(
            "Treating unproven context manager %r as non-suppressing via name heuristic in "
            "strict mode; use closure_strictness='pedantic' to fail closed on unknown context "
            "managers or add %r to non_suppressing_ctx_managers.",
            mgr_name,
            mgr_name,
        )


def _is_non_raising_store_target(
    target: ast.AST,
    value: Optional[ast.AST] = None,
    *,
    leaf_predicate: Optional[Callable[[ast.AST], bool]] = None,
) -> bool:
    """Returns True if storing value into target avoids unpacking errors and satisfies any leaf predicate."""
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(target, value)]
    while stack:
        curr_target, curr_value = stack.pop()
        while isinstance(curr_value, ast.NamedExpr):
            curr_value = curr_value.value
        if isinstance(curr_target, ast.Name):
            if leaf_predicate is not None and (
                curr_value is None or not leaf_predicate(curr_value)
            ):
                return False
            continue
        if (
            isinstance(curr_target, (ast.Tuple, ast.List))
            and isinstance(curr_value, (ast.Tuple, ast.List))
            and len(curr_target.elts) == len(curr_value.elts)
            and not any(
                isinstance(e, ast.Starred) for e in (*curr_target.elts, *curr_value.elts)
            )
        ):
            stack.extend(zip(curr_target.elts, curr_value.elts))
            continue
        return False
    return True


def _extract_definite_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names unconditionally bound by an assignment target."""
    return set(_extract_assigned_names(target))


def _walrus_assignment_in_expr(
    expr: Optional[ast.AST],
    is_conditional: bool = False,
    *,
    _depth: int = 0,
) -> Tuple[Set[str], Set[str]]:
    """Analyzes walrus assignments in an expression to distinguish definite from conditional stores."""
    definite: Set[str] = set()
    conditional: Set[str] = set()
    if expr is None:
        return definite, conditional

    work: List[Tuple[ast.AST, bool]] = [(expr, is_conditional)]
    while work:
        curr, curr_cond = work.pop()
        if isinstance(curr, _COMPREHENSION_TYPES):
            for idx, gen in enumerate(curr.generators):
                work.append((gen.iter, curr_cond if idx == 0 else True))
                for if_expr in gen.ifs:
                    work.append((if_expr, True))
            if isinstance(curr, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                work.append((curr.elt, True))
            elif isinstance(curr, ast.DictComp):
                work.append((curr.key, True))
                work.append((curr.value, True))
            continue

        if isinstance(curr, _INNER_SCOPE_TYPES):
            for hdr in iter_nested_scope_header_nodes(curr):
                work.append((hdr, curr_cond))
            continue

        if isinstance(curr, ast.NamedExpr):
            if isinstance(curr.target, ast.Name):
                if curr_cond:
                    conditional.add(curr.target.id)
                else:
                    definite.add(curr.target.id)
            work.append((curr.value, curr_cond))
            continue

        if isinstance(curr, ast.BoolOp):
            for idx, val in enumerate(curr.values):
                work.append((val, curr_cond if idx == 0 else True))
            continue

        if isinstance(curr, ast.IfExp):
            work.append((curr.test, curr_cond))
            if _depth >= _MAX_AST_RECURSION_DEPTH:
                work.append((curr.body, True))
                work.append((curr.orelse, True))
            else:
                d_b, c_b = _walrus_assignment_in_expr(
                    curr.body, False, _depth=_depth + 1
                )
                d_o, c_o = _walrus_assignment_in_expr(
                    curr.orelse, False, _depth=_depth + 1
                )
                both_def = (d_b & d_o) if not curr_cond else set()
                definite.update(both_def)
                conditional.update((d_b | c_b | d_o | c_o) - both_def)
            continue

        if isinstance(curr, ast.Compare):
            work.append((curr.left, curr_cond))
            for idx, comp in enumerate(curr.comparators):
                work.append((comp, curr_cond if idx == 0 else True))
            continue

        for child in ast.iter_child_nodes(curr):
            target_child = child.value if isinstance(child, ast.keyword) else child
            if isinstance(target_child, ast.AST):
                work.append((target_child, curr_cond))

    return definite, conditional - definite


def _extract_unconditional_walrus_names(node: ast.AST) -> Set[str]:
    """Extracts NamedExpr target names unconditionally evaluated in an AST node."""
    return _walrus_assignment_in_expr(node, is_conditional=False)[0]


def _extract_deleted_names(statements: Sequence[ast.AST]) -> Set[str]:
    """Finds all variable names deleted within a statement sequence without recursing into nested functions."""
    deleted: Set[str] = set()
    stack: List[ast.AST] = list(statements)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, ast.Delete):
            deleted.update(
                node.id
                for target in stmt.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del)
            )
        elif isinstance(stmt, _INNER_SCOPE_TYPES):
            continue
        elif isinstance(stmt, ast.ExceptHandler):
            stack.extend(stmt.body)
        elif is_match_node(stmt):  # pragma: no cover (py310+)
            for case in getattr(stmt, "cases", []):
                stack.extend(getattr(case, "body", []))
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, (ast.stmt, ast.ExceptHandler)):
                    stack.append(child)
    return deleted
