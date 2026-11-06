"""Context-manager suppression provenance and unified definite-assignment analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _FnScopeContext,
    _can_bind_in_fn_scope,
    _extract_store_names_in_node,
    _extract_type_alias_or_pattern_name,
    _find_enclosing_function_chain,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _is_irrefutable_case,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_pattern_bound_name,
    _record_import_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

_SUPPRESSING_CTX_MANAGER_NAMES: frozenset[str] = frozenset(
    {"suppress", "raises", "assertRaises", "assertRaisesRegex", "ignored"}
)

_NON_SUPPRESSING_CTX_MANAGER_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "nullcontext",
        "closing",
        "aclosing",
        "redirect_stdout",
        "redirect_stderr",
        "chdir",
        "TemporaryDirectory",
        "NamedTemporaryFile",
        "SpooledTemporaryFile",
        "TemporaryFile",
    }
)

_STDLIB_NON_SUPPRESSING_MODULES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "builtins": frozenset({"open"}),
        "__builtins__": frozenset({"open"}),
        "io": frozenset({"open"}),
        "contextlib": frozenset(
            {
                "nullcontext",
                "closing",
                "aclosing",
                "redirect_stdout",
                "redirect_stderr",
                "chdir",
            }
        ),
        "tempfile": frozenset(
            {
                "TemporaryDirectory",
                "NamedTemporaryFile",
                "SpooledTemporaryFile",
                "TemporaryFile",
            }
        ),
    }
)

_SAFE_EXCEPTION_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "BufferError",
        "EOFError",
        "Exception",
        "FileExistsError",
        "FileNotFoundError",
        "FloatingPointError",
        "GeneratorExit",
        "IOError",
        "ImportError",
        "IndentationError",
        "IndexError",
        "KeyError",
        "KeyboardInterrupt",
        "LookupError",
        "MemoryError",
        "ModuleNotFoundError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "RecursionError",
        "ReferenceError",
        "RuntimeError",
        "StopAsyncIteration",
        "StopIteration",
        "SyntaxError",
        "SystemError",
        "SystemExit",
        "TabError",
        "TypeError",
        "UnboundLocalError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "UnicodeError",
        "UnicodeTranslateError",
        "ValueError",
        "ZeroDivisionError",
    }
)

_SHADOWABLE_BUILTIN_CTX_NAMES: frozenset[str] = frozenset(
    {"open", "builtins", "__builtins__"}
)


@dataclass(frozen=True)
class _SuppressProvenance(AbstractSet[str]):
    """Structured provenance for context-manager suppression and stdlib non-suppression."""

    suppressors: frozenset[str] = frozenset()
    non_suppressors: frozenset[str] = frozenset()
    stdlib_modules: frozenset[Tuple[str, str]] = frozenset()
    shadowed_builtins: frozenset[str] = frozenset()
    has_star_import: bool = False

    @property
    def suppress_names(self) -> frozenset[str]:
        """Returns suppressing context-manager names."""
        return self.suppressors

    @property
    def non_suppress_names(self) -> frozenset[str]:
        """Returns non-suppressing context-manager names."""
        return self.non_suppressors

    def to_tokens(self) -> Set[str]:
        """Serializes provenance into canonical token strings for set compatibility."""
        tokens: Set[str] = set(self.suppressors)
        for name in self.non_suppressors:
            tokens.add(f"!nonsup:{name}")
        for alias, mod_name in self.stdlib_modules:
            tokens.add(f"!mod:{alias}:{mod_name}")
        for b_name in self.shadowed_builtins:
            tokens.add(f"!shadow:{b_name}")
        if self.has_star_import:
            tokens.add("!star_import")
        return tokens

    @classmethod
    def from_tokens(
        cls, tokens: Union[_SuppressProvenance, Iterable[str]]
    ) -> _SuppressProvenance:
        """Constructs a structured _SuppressProvenance from an existing instance or token iterable."""
        if isinstance(tokens, _SuppressProvenance):
            return tokens
        suppressors: Set[str] = set()
        non_suppressors: Set[str] = set()
        stdlib_modules: Set[Tuple[str, str]] = set()
        shadowed_builtins: Set[str] = set()
        has_star_import = False
        for tok in tokens:
            if not isinstance(tok, str) or not tok:
                continue
            if tok == "!star_import":
                has_star_import = True
            elif tok.startswith("!nonsup:"):
                name = tok[len("!nonsup:") :]
                if name:
                    non_suppressors.add(name)
            elif tok.startswith("!mod:"):
                rest = tok[len("!mod:") :]
                parts = rest.split(":", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    stdlib_modules.add((parts[0], parts[1]))
            elif tok.startswith("!shadow:"):
                b_name = tok[len("!shadow:") :]
                if b_name:
                    shadowed_builtins.add(b_name)
            elif not tok.startswith("!"):
                suppressors.add(tok)
        return cls(
            suppressors=frozenset(suppressors),
            non_suppressors=frozenset(non_suppressors),
            stdlib_modules=frozenset(stdlib_modules),
            shadowed_builtins=frozenset(shadowed_builtins),
            has_star_import=has_star_import,
        )

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str) or not item:
            return False
        if item == "!star_import":
            return self.has_star_import
        if item.startswith("!nonsup:"):
            return item[len("!nonsup:") :] in self.non_suppressors
        if item.startswith("!mod:"):
            parts = item[len("!mod:") :].split(":", 1)
            return (
                len(parts) == 2
                and (parts[0], parts[1]) in self.stdlib_modules
            )
        if item.startswith("!shadow:"):
            return item[len("!shadow:") :] in self.shadowed_builtins
        if item.startswith("!"):
            return False
        return item in self.suppressors

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_tokens())

    def __len__(self) -> int:
        return (
            len(self.suppressors)
            + len(self.non_suppressors)
            + len(self.stdlib_modules)
            + len(self.shadowed_builtins)
            + (1 if self.has_star_import else 0)
        )

    def clear_name(
        self, name: str, *, mark_shadow: bool = True
    ) -> _SuppressProvenance:
        """Returns provenance with suppressor and non-suppressing bindings cleared for name."""
        if not name or name.startswith("!"):
            return self
        new_sup = self.suppressors - {name}
        new_nonsup = self.non_suppressors - {name}
        new_mods = frozenset(
            pair for pair in self.stdlib_modules if pair[0] != name
        )
        new_shadow = (
            (self.shadowed_builtins | {name})
            if (mark_shadow and name in _SHADOWABLE_BUILTIN_CTX_NAMES)
            else self.shadowed_builtins
        )
        if (
            new_sup == self.suppressors
            and new_nonsup == self.non_suppressors
            and new_mods == self.stdlib_modules
            and new_shadow == self.shadowed_builtins
        ):
            return self
        return _SuppressProvenance(
            suppressors=new_sup,
            non_suppressors=new_nonsup,
            stdlib_modules=new_mods,
            shadowed_builtins=new_shadow,
            has_star_import=self.has_star_import,
        )

    def _bind_provenance_entry(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        is_suppressor: bool = False,
        is_non_suppressor: bool = False,
        stdlib_mod: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Clears prior bindings for name and optionally records a new provenance entry."""
        cleared = self.clear_name(name)
        if not allow_positive or not name or name.startswith("!"):
            return cleared
        return _SuppressProvenance(
            suppressors=(
                cleared.suppressors | {name}
                if is_suppressor
                else cleared.suppressors
            ),
            non_suppressors=(
                cleared.non_suppressors | {name}
                if is_non_suppressor
                else cleared.non_suppressors
            ),
            stdlib_modules=(
                cleared.stdlib_modules | {(name, stdlib_mod)}
                if stdlib_mod is not None
                else cleared.stdlib_modules
            ),
            shadowed_builtins=cleared.shadowed_builtins,
            has_star_import=cleared.has_star_import,
        )

    def with_suppressor(self, name: str) -> _SuppressProvenance:
        """Returns provenance after binding name to a known suppressing context manager."""
        return self._bind_provenance_entry(name, is_suppressor=True)

    def with_non_suppressor(
        self, name: str, *, allow_positive: bool = True
    ) -> _SuppressProvenance:
        """Returns provenance after binding name to a proven non-suppressing context manager."""
        return self._bind_provenance_entry(
            name, allow_positive=allow_positive, is_non_suppressor=True
        )

    def with_stdlib_module(
        self, alias: str, mod_name: str, *, allow_positive: bool = True
    ) -> _SuppressProvenance:
        """Returns provenance after binding alias to a trusted non-suppressing stdlib module."""
        return self._bind_provenance_entry(
            alias, allow_positive=allow_positive, stdlib_mod=mod_name
        )

    def with_shadowed_builtins(
        self, names: Iterable[str]
    ) -> _SuppressProvenance:
        """Returns provenance with the given builtin names marked shadowed."""
        valid = frozenset(
            n for n in names if n in _SHADOWABLE_BUILTIN_CTX_NAMES
        )
        if not valid:
            return self
        return _SuppressProvenance(
            suppressors=self.suppressors,
            non_suppressors=self.non_suppressors - valid,
            stdlib_modules=frozenset(
                pair for pair in self.stdlib_modules if pair[0] not in valid
            ),
            shadowed_builtins=self.shadowed_builtins | valid,
            has_star_import=self.has_star_import,
        )

    def with_star_import(self) -> _SuppressProvenance:
        """Returns provenance with has_star_import set to True."""
        if self.has_star_import:
            return self
        return _SuppressProvenance(
            suppressors=self.suppressors,
            non_suppressors=self.non_suppressors,
            stdlib_modules=self.stdlib_modules,
            shadowed_builtins=self.shadowed_builtins,
            has_star_import=True,
        )

    def merge_branch_conservative(
        self, branch_prov: _SuppressProvenance
    ) -> _SuppressProvenance:
        """Merges provenance from a branch, keeping only surviving positive proofs and all hazards."""
        return _SuppressProvenance(
            suppressors=self.suppressors | branch_prov.suppressors,
            non_suppressors=self.non_suppressors & branch_prov.non_suppressors,
            stdlib_modules=self.stdlib_modules & branch_prov.stdlib_modules,
            shadowed_builtins=(
                self.shadowed_builtins | branch_prov.shadowed_builtins
            ),
            has_star_import=(
                self.has_star_import or branch_prov.has_star_import
            ),
        )


_EMPTY_SUPPRESS_PROVENANCE = _SuppressProvenance()


def _coerce_suppress_provenance(
    suppress_aliases: Optional[Union[_SuppressProvenance, AbstractSet[str]]],
) -> _SuppressProvenance:
    """Coerces an optional token set or _SuppressProvenance into a _SuppressProvenance."""
    if suppress_aliases is None:
        return _EMPTY_SUPPRESS_PROVENANCE
    if isinstance(suppress_aliases, _SuppressProvenance):
        return suppress_aliases
    if not suppress_aliases:
        return _EMPTY_SUPPRESS_PROVENANCE
    return _SuppressProvenance.from_tokens(suppress_aliases)


def _sync_mutable_suppress_aliases(
    target: Union[Set[str], _SuppressProvenance],
    updated: _SuppressProvenance,
) -> _SuppressProvenance:
    """Synchronizes a mutable Set[str] in-place when used by legacy callers."""
    if isinstance(target, set):
        target.clear()
        target.update(updated.to_tokens())
    return updated


def _extract_ctx_manager_call_name(expr: ast.AST) -> Optional[str]:
    """Returns the target function or attribute name of a context manager Call node."""
    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Name):
            return expr.func.id
        if isinstance(expr.func, ast.Attribute):
            return expr.func.attr
    return None


def _clear_name_provenance(
    name: str,
    suppress_aliases: Union[Set[str], _SuppressProvenance],
    *,
    mark_shadow: bool = True,
) -> _SuppressProvenance:
    """Clears suppressor and non-suppressing stdlib provenance for a rebound name."""
    return _sync_mutable_suppress_aliases(
        suppress_aliases,
        _coerce_suppress_provenance(suppress_aliases).clear_name(
            name, mark_shadow=mark_shadow
        ),
    )


def _is_suppress_alias_target_expr(
    expr: Optional[ast.AST],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if expr references a known suppressing context manager or alias."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    if isinstance(expr, ast.Name):
        return not expr.id.startswith("!") and (
            expr.id in _SUPPRESSING_CTX_MANAGER_NAMES
            or expr.id in prov.suppressors
        )
    if isinstance(expr, ast.Attribute):
        return expr.attr in _SUPPRESSING_CTX_MANAGER_NAMES
    return False


def _is_proven_non_suppressing_target_expr(
    expr: Optional[ast.AST],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if expr is a proven unshadowed non-suppressing context manager factory."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    if isinstance(expr, ast.Name):
        fn_id = expr.id
        if fn_id.startswith("!") or fn_id in prov.suppressors:
            return False
        if fn_id in prov.non_suppressors:
            return True
        if (
            extra_non_suppressing_ctx_managers
            and fn_id in extra_non_suppressing_ctx_managers
            and fn_id not in prov.shadowed_builtins
        ):
            return True
        return (
            fn_id == "open"
            and "open" not in prov.shadowed_builtins
            and not prov.has_star_import
        )
    if isinstance(expr, ast.Attribute):
        attr_name = expr.attr
        if (
            extra_non_suppressing_ctx_managers
            and attr_name in extra_non_suppressing_ctx_managers
            and attr_name not in _SUPPRESSING_CTX_MANAGER_NAMES
        ):
            return True
        if isinstance(expr.value, ast.Name):
            base_id = expr.value.id
            if (
                base_id in ("builtins", "__builtins__")
                and attr_name == "open"
                and base_id not in prov.shadowed_builtins
                and not prov.has_star_import
            ):
                return True
            for alias, mod_name in prov.stdlib_modules:
                if alias == base_id:
                    allowed_attrs = _STDLIB_NON_SUPPRESSING_MODULES.get(
                        mod_name, frozenset()
                    )
                    if attr_name in allowed_attrs:
                        return True
    return False


def _is_proven_non_suppressing_call(
    expr: ast.AST,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if expr is a Call to a proven non-suppressing context manager."""
    return isinstance(expr, ast.Call) and _is_proven_non_suppressing_target_expr(
        expr.func,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _class_def_may_suppress_in_exit(cls_node: ast.ClassDef) -> bool:
    """Returns True if a ClassDef defines an __exit__/__aexit__ method that can return truthy."""
    for item in cls_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            item.name in ("__exit__", "__aexit__")
        ):
            for sub in ast.walk(item):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    if not (
                        isinstance(sub.value, ast.Constant)
                        and not bool(sub.value.value)
                    ):
                        return True
    return False


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


def _update_suppress_provenance_from_expr_walrus(
    expr_roots: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool = True,
) -> _SuppressProvenance:
    """Updates provenance from NamedExpr (walrus) bindings inside expression trees."""
    w_stack: List[Tuple[ast.AST, bool]] = [
        (root, allow_positive) for root in expr_roots
    ]
    while w_stack:
        w_curr, cur_allow_pos = w_stack.pop()
        if isinstance(w_curr, _INNER_SCOPE_TYPES):
            for hdr in iter_nested_scope_header_nodes(w_curr):
                w_stack.append((hdr, cur_allow_pos))
            continue
        if isinstance(w_curr, _COMPREHENSION_TYPES):
            if w_curr.generators:
                w_stack.append((w_curr.generators[0].iter, cur_allow_pos))
                for gen in w_curr.generators[1:]:
                    w_stack.append((gen.iter, False))
                for gen in w_curr.generators:
                    for if_e in gen.ifs:
                        w_stack.append((if_e, False))
            if isinstance(w_curr, ast.DictComp):
                w_stack.append((w_curr.key, False))
                w_stack.append((w_curr.value, False))
            else:
                w_stack.append((w_curr.elt, False))
            continue
        if isinstance(w_curr, ast.NamedExpr) and isinstance(
            w_curr.target, ast.Name
        ):
            prov = _bind_expr_target_provenance(
                prov,
                w_curr.target.id,
                w_curr.value,
                allow_positive=cur_allow_pos,
            )
        if isinstance(w_curr, ast.BoolOp):
            for idx, val in enumerate( reversed(w_curr.values) ):
                orig_idx = len(w_curr.values) - 1 - idx
                w_stack.append((val, cur_allow_pos and orig_idx == 0))
            continue
        if isinstance(w_curr, ast.IfExp):
            w_stack.append((w_curr.orelse, False))
            w_stack.append((w_curr.body, False))
            w_stack.append((w_curr.test, cur_allow_pos))
            continue
        for ch in ast.iter_child_nodes(w_curr):
            if not isinstance(ch, ast.stmt):
                w_stack.append((ch, cur_allow_pos))
    return prov


def _bind_expr_target_provenance(
    prov: _SuppressProvenance,
    name: str,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool = True,
) -> _SuppressProvenance:
    """Updates provenance when name is directly bound to value_expr."""
    if _is_suppress_alias_target_expr(value_expr, prov):
        return prov.with_suppressor(name)
    if _is_proven_non_suppressing_target_expr(value_expr, prov):
        return prov.with_non_suppressor(name, allow_positive=allow_positive)
    return prov.clear_name(name)


def _update_suppress_provenance_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
) -> _SuppressProvenance:
    """Applies _update_suppress_provenance_from_stmt across a statement sequence."""
    for sub_stmt in stmts:
        prov = _update_suppress_provenance_from_stmt(
            sub_stmt, prov, in_branch=in_branch
        )
    return prov


def _update_clause_branch_provenance(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    body: Iterable[ast.AST],
) -> _SuppressProvenance:
    """Updates provenance for an except handler or match case branch."""
    if isinstance(header_expr, ast.AST):
        prov = _update_suppress_provenance_from_expr_walrus(
            (header_expr,), prov, allow_positive=False
        )
    for b_name in bound_names:
        prov = prov.clear_name(b_name)
    return _update_suppress_provenance_from_stmts(body, prov, in_branch=True)


def _update_suppress_provenance_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
) -> _SuppressProvenance:
    """Returns updated _SuppressProvenance from a statement, recursing into compound blocks."""
    allow_pos = not in_branch
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if bound == "*":
                prov = prov.with_star_import()
                continue
            if alias.name in _STDLIB_NON_SUPPRESSING_MODULES and (
                alias.asname is not None or "." not in alias.name
            ):
                prov = prov.with_stdlib_module(
                    bound, alias.name, allow_positive=allow_pos
                )
            else:
                prov = prov.clear_name(bound)
        return prov

    if isinstance(stmt, ast.ImportFrom):
        mod_name = stmt.module or ""
        allowed_nonsup = (
            _STDLIB_NON_SUPPRESSING_MODULES.get(mod_name, frozenset())
            if getattr(stmt, "level", 0) == 0
            else frozenset()
        )
        for alias in stmt.names:
            if alias.name == "*":
                prov = prov.with_star_import()
                continue
            bound = alias.asname or alias.name
            if alias.name in _SUPPRESSING_CTX_MANAGER_NAMES:
                prov = prov.with_suppressor(bound)
            elif alias.name in allowed_nonsup:
                prov = prov.with_non_suppressor(
                    bound, allow_positive=allow_pos
                )
            else:
                prov = prov.clear_name(bound)
        return prov

    if isinstance(stmt, ast.Assign):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.value, *stmt.targets), prov, allow_positive=allow_pos
        )
        for tgt in stmt.targets:
            for t_name in _extract_store_names_in_node(tgt):
                prov = (
                    _bind_expr_target_provenance(
                        prov, t_name, stmt.value, allow_positive=allow_pos
                    )
                    if isinstance(tgt, ast.Name)
                    else prov.clear_name(t_name)
                )
        return prov

    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        exprs = [
            e
            for e in (
                getattr(stmt, "annotation", None),
                stmt.value,
                stmt.target,
            )
            if isinstance(e, ast.AST)
        ]
        prov = _update_suppress_provenance_from_expr_walrus(
            exprs, prov, allow_positive=allow_pos
        )
        if stmt.value is not None:
            for t_name in _extract_store_names_in_node(stmt.target):
                prov = prov.clear_name(t_name)
        return prov

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prov = _update_suppress_provenance_from_expr_walrus(
            iter_nested_scope_header_nodes(stmt), prov, allow_positive=allow_pos
        )
        return prov.clear_name(stmt.name)

    if isinstance(stmt, ast.ClassDef):
        prov = _update_suppress_provenance_from_expr_walrus(
            iter_nested_scope_header_nodes(stmt), prov, allow_positive=allow_pos
        )
        if _class_def_may_suppress_in_exit(stmt):
            return prov.with_suppressor(stmt.name)
        return prov.clear_name(stmt.name)

    if isinstance(stmt, ast.Delete):
        prov = _update_suppress_provenance_from_expr_walrus(
            stmt.targets, prov, allow_positive=allow_pos
        )
        for d_name in _extract_shallow_deleted_names(stmt):
            prov = prov.clear_name(d_name)
        return prov

    if type(stmt).__name__ == "TypeAlias":
        ta_val = getattr(stmt, "value", None)
        if isinstance(ta_val, ast.AST):
            prov = _update_suppress_provenance_from_expr_walrus(
                (ta_val,), prov, allow_positive=allow_pos
            )
        ta_name = _extract_type_alias_or_pattern_name(stmt)
        if ta_name is not None:
            prov = prov.clear_name(ta_name)
        return prov

    if isinstance(stmt, (ast.If, ast.While)):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.test,), prov, allow_positive=allow_pos
        )
        return _update_suppress_provenance_from_stmts(
            (*stmt.body, *stmt.orelse), prov, in_branch=True
        )

    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.iter, stmt.target), prov, allow_positive=allow_pos
        )
        for t_name in _extract_store_names_in_node(stmt.target):
            prov = prov.clear_name(t_name)
        return _update_suppress_provenance_from_stmts(
            (*stmt.body, *stmt.orelse), prov, in_branch=True
        )

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            prov = _update_suppress_provenance_from_expr_walrus(
                (item.context_expr,), prov, allow_positive=allow_pos
            )
            if item.optional_vars is not None:
                for t_name in _extract_store_names_in_node(item.optional_vars):
                    prov = prov.clear_name(t_name)
        return _update_suppress_provenance_from_stmts(
            stmt.body, prov, in_branch=True
        )

    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        prov = _update_suppress_provenance_from_stmts(
            (*getattr(stmt, "body", ()), *getattr(stmt, "orelse", ())),
            prov,
            in_branch=True,
        )
        for h_node in getattr(stmt, "handlers", ()):
            h_name = getattr(h_node, "name", None)
            prov = _update_clause_branch_provenance(
                prov,
                getattr(h_node, "type", None),
                (h_name,) if isinstance(h_name, str) else (),
                getattr(h_node, "body", ()),
            )
        return _update_suppress_provenance_from_stmts(
            getattr(stmt, "finalbody", ()), prov, in_branch=in_branch
        )

    if type(stmt).__name__ == "Match":
        subject = getattr(stmt, "subject", None)
        if isinstance(subject, ast.AST):
            prov = _update_suppress_provenance_from_expr_walrus(
                (subject,), prov, allow_positive=allow_pos
            )
        for case in getattr(stmt, "cases", ()):
            pattern = getattr(case, "pattern", None)
            pat_names = (
                [
                    p_name
                    for p_node in ast.walk(pattern)
                    for p_name in (_extract_pattern_bound_name(p_node),)
                    if p_name is not None
                ]
                if isinstance(pattern, ast.AST)
                else []
            )
            prov = _update_clause_branch_provenance(
                prov,
                getattr(case, "guard", None),
                pat_names,
                getattr(case, "body", ()),
            )
        return prov

    if not isinstance(stmt, _INNER_SCOPE_TYPES):
        child_exprs = [
            ch
            for ch in ast.iter_child_nodes(stmt)
            if not isinstance(ch, ast.stmt)
        ]
        if child_exprs:
            prov = _update_suppress_provenance_from_expr_walrus(
                child_exprs, prov, allow_positive=allow_pos
            )
    return prov


def _update_suppress_aliases_from_stmt(
    stmt: ast.AST,
    suppress_aliases: Union[Set[str], _SuppressProvenance],
    *,
    in_branch: bool = False,
) -> _SuppressProvenance:
    """Updates suppress_aliases from import, assignment, definition, or compound statements."""
    return _sync_mutable_suppress_aliases(
        suppress_aliases,
        _update_suppress_provenance_from_stmt(
            stmt,
            _coerce_suppress_provenance(suppress_aliases),
            in_branch=in_branch,
        ),
    )


def _scope_has_direct_star_import(scope_node: ast.AST) -> bool:
    """Checks whether scope_node contains a wildcard 'from ... import *' outside nested scopes."""
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


def _seed_scope_suppress_aliases(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    u_start: int = 0,
) -> _SuppressProvenance:
    """Seeds initial context-manager suppression and shadowing provenance for fn_node."""
    prov = _EMPTY_SUPPRESS_PROVENANCE
    eff_mod = (
        mod_tree
        if isinstance(mod_tree, ast.Module)
        else (fn_node if isinstance(fn_node, ast.Module) else None)
    )
    if eff_mod is not None:
        reached_cutoff = False
        for m_stmt in eff_mod.body:
            if m_stmt is fn_node:
                reached_cutoff = True
                continue
            if (
                not reached_cutoff
                and fn_node is eff_mod
                and 0 < u_start <= (getattr(m_stmt, "lineno", 0) or 0)
            ):
                reached_cutoff = True
            prov = _update_suppress_provenance_from_stmt(
                m_stmt, prov, in_branch=reached_cutoff
            )

    enclosing_nodes: Sequence[ast.AST] = (
        scope_ctx.enclosing_scopes
        if scope_ctx is not None and scope_ctx.enclosing_scopes
        else tuple(
            enc_n
            for _, _, enc_n in _find_enclosing_function_chain(
                eff_mod,
                int(getattr(fn_node, "lineno", 0) or 0),
                int(getattr(fn_node, "end_lineno", 0) or 0),
                getattr(fn_node, "col_offset", None),
                getattr(fn_node, "end_col_offset", None),
            )
            if enc_n is not fn_node
        )
        if eff_mod is not None and fn_node is not eff_mod
        else ()
    )
    for enc in enclosing_nodes:
        enc_args = getattr(enc, "args", None)
        if isinstance(enc_args, ast.arguments):
            for p_name in _extract_arg_names(enc_args):
                prov = prov.clear_name(p_name)
        for tp_name in _extract_node_type_param_names(enc):
            prov = prov.clear_name(tp_name)
        enc_body = getattr(enc, "body", None)
        if isinstance(enc_body, (list, tuple)):
            reached_fn = False
            for e_stmt in enc_body:
                if e_stmt is fn_node:
                    reached_fn = True
                    continue
                prov = _update_suppress_provenance_from_stmt(
                    e_stmt, prov, in_branch=reached_fn
                )

    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        for p_name in _extract_arg_names(fn_args):
            prov = prov.clear_name(p_name)
    for tp_name in _extract_node_type_param_names(fn_node):
        prov = prov.clear_name(tp_name)

    if u_start > 0 and fn_node is not eff_mod:
        fn_body = getattr(fn_node, "body", None)
        if isinstance(fn_body, (list, tuple)):
            for f_stmt in fn_body:
                if (getattr(f_stmt, "lineno", 0) or 0) >= u_start:
                    break
                prov = _update_suppress_provenance_from_stmt(f_stmt, prov)

    # Soundness check: any binding to 'open', 'builtins', or '__builtins__' anywhere
    # in the module, enclosing scopes, or current scope marks it shadowed.
    scope_chain: Tuple[ast.AST, ...] = (
        *((eff_mod,) if eff_mod is not None else ()),
        *enclosing_nodes,
        *((fn_node,) if fn_node is not eff_mod else ()),
    )
    bound_in_chain: Set[str] = set()
    has_star = prov.has_star_import
    if scope_ctx is not None:
        bound_in_chain.update(scope_ctx.shadowed_names | scope_ctx.fn_locals)
        if scope_ctx.has_star_import:
            has_star = True
    for sc_node in scope_chain:
        bound_in_chain.update(_extract_scope_local_names(sc_node))
        if not has_star and _scope_has_direct_star_import(sc_node):
            has_star = True
    if has_star:
        prov = prov.with_star_import()
    shadowed_b = bound_in_chain & _SHADOWABLE_BUILTIN_CTX_NAMES
    if shadowed_b:
        prov = prov.with_shadowed_builtins(shadowed_b)
    return prov


def _classify_with_suppression(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> str:
    """Classifies a with/async with statement as 'suppressing', 'non_suppressing', or 'unknown'."""
    if not stmt.items:
        return "non_suppressing"
    prov = _coerce_suppress_provenance(suppress_aliases)
    all_non_suppressing = True
    for item in stmt.items:
        call_name = _extract_ctx_manager_call_name(item.context_expr)
        if (
            call_name is not None
            and not call_name.startswith("!")
            and (
                call_name in _SUPPRESSING_CTX_MANAGER_NAMES
                or call_name in prov.suppressors
            )
        ):
            return "suppressing"
        if (
            isinstance(item.context_expr, ast.Name)
            and not item.context_expr.id.startswith("!")
            and item.context_expr.id in prov.suppressors
        ):
            return "suppressing"
        if not _is_proven_non_suppressing_call(
            item.context_expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            all_non_suppressing = False
    return "non_suppressing" if all_non_suppressing else "unknown"


def _is_exception_suppressing_with(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Checks whether a with/async with statement uses a known exception-suppressing manager."""
    return _classify_with_suppression(stmt, suppress_aliases) == "suppressing"


def _may_suppress_exceptions_with(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True unless all context managers in stmt are known non-suppressing."""
    return (
        _classify_with_suppression(
            stmt,
            suppress_aliases,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        != "non_suppressing"
    )


def _is_non_raising_bool_expr(
    expr: Optional[ast.AST],
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    strict_names: bool = True,
) -> bool:
    """Returns True if evaluating bool(expr) cannot raise an exception."""
    return _is_non_raising_expr(
        expr,
        safe_names,
        safe_bool_names,
        require_bool_safe=True,
        strict_names=strict_names,
    )


def _is_non_raising_expr(
    expr: Optional[ast.AST],
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    require_bool_safe: bool = False,
    strict_names: bool = True,
) -> bool:
    """Returns True if evaluating expr cannot raise an exception caught by a with block."""
    if expr is None or isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        if require_bool_safe:
            return expr.id in safe_bool_names
        return (not strict_names) or (expr.id in safe_names)
    if isinstance(expr, ast.Lambda):
        return all(
            _is_non_raising_expr(
                d, safe_names, safe_bool_names, strict_names=strict_names
            )
            for d in (
                *expr.args.defaults,
                *(d for d in expr.args.kw_defaults if d is not None),
            )
        )
    if isinstance(expr, (ast.Tuple, ast.List)):
        return not any(isinstance(e, ast.Starred) for e in expr.elts) and all(
            _is_non_raising_expr(
                e, safe_names, safe_bool_names, strict_names=strict_names
            )
            for e in expr.elts
        )
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, ast.Not):
            return _is_non_raising_bool_expr(
                expr.operand,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
        if isinstance(expr.op, ast.Invert):
            return isinstance(expr.operand, ast.Constant) and isinstance(
                expr.operand.value, (int, bool)
            )
        return (
            isinstance(expr.op, (ast.UAdd, ast.USub))
            and isinstance(expr.operand, ast.Constant)
            and isinstance(expr.operand.value, (int, float, bool))
        )
    if isinstance(expr, ast.BoolOp):
        return bool(expr.values) and (
            all(
                _is_non_raising_bool_expr(
                    v, safe_names, safe_bool_names, strict_names=strict_names
                )
                for v in expr.values[:-1]
            )
            and _is_non_raising_expr(
                expr.values[-1],
                safe_names,
                safe_bool_names,
                require_bool_safe=require_bool_safe,
                strict_names=strict_names,
            )
        )
    if isinstance(expr, ast.IfExp):
        return _is_non_raising_bool_expr(
            expr.test, safe_names, safe_bool_names, strict_names=strict_names
        ) and all(
            _is_non_raising_expr(
                b,
                safe_names,
                safe_bool_names,
                require_bool_safe=require_bool_safe,
                strict_names=strict_names,
            )
            for b in (expr.body, expr.orelse)
        )
    if isinstance(expr, ast.Compare):
        return (
            all(isinstance(op, (ast.Is, ast.IsNot)) for op in expr.ops)
            and _is_non_raising_expr(
                expr.left,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            and all(
                _is_non_raising_expr(
                    c, safe_names, safe_bool_names, strict_names=strict_names
                )
                for c in expr.comparators
            )
        )
    if isinstance(expr, ast.NamedExpr):
        return isinstance(expr.target, ast.Name) and _is_non_raising_expr(
            expr.value,
            safe_names,
            safe_bool_names,
            require_bool_safe=require_bool_safe,
            strict_names=strict_names,
        )
    return False


def _is_non_raising_store_target(
    target: ast.AST,
    value: Optional[ast.AST] = None,
) -> bool:
    """Returns True if storing value into target cannot raise a runtime unpacking/target error."""
    if isinstance(target, ast.Name):
        return True
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(
            isinstance(e, ast.Starred) for e in (*target.elts, *value.elts)
        )
    ):
        return all(
            _is_non_raising_store_target(t, v)
            for t, v in zip(target.elts, value.elts)
        )
    return False


def _extract_definite_target_names(target: ast.AST) -> Set[str]:
    """Extracts variable names unconditionally bound by an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: Set[str] = set()
        for elt in target.elts:
            names.update(_extract_definite_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _extract_definite_target_names(target.value)
    return set()


def _stmt_cannot_raise_in_suppress(
    stmt: ast.AST,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    strict_names: bool = True,
) -> bool:
    """Returns True if stmt executes and completes without raising a catchable exception."""
    if isinstance(
        stmt, (ast.Pass, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)
    ):
        return True
    if isinstance(stmt, ast.Delete):
        return bool(stmt.targets) and all(
            isinstance(t, ast.Name) and t.id in safe_names
            for t in stmt.targets
        )
    if isinstance(stmt, (ast.Return, ast.Expr)):
        return _is_non_raising_expr(
            stmt.value, safe_names, safe_bool_names, strict_names=strict_names
        )
    if isinstance(stmt, ast.Assign):
        return (
            stmt.value is not None
            and _is_non_raising_expr(
                stmt.value,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            and all(
                _is_non_raising_store_target(t, stmt.value)
                for t in stmt.targets
            )
        )
    if isinstance(stmt, ast.AnnAssign):
        return _is_non_raising_expr(
            stmt.annotation,
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
        ) and (
            stmt.value is None
            or (
                _is_non_raising_expr(
                    stmt.value,
                    safe_names,
                    safe_bool_names,
                    strict_names=strict_names,
                )
                and _is_non_raising_store_target(stmt.target, stmt.value)
            )
        )
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return not stmt.decorator_list and all(
            _is_non_raising_expr(
                h, safe_names, safe_bool_names, strict_names=strict_names
            )
            for h in iter_nested_scope_header_nodes(stmt)
        )
    if isinstance(stmt, ast.If):
        return (
            _is_non_raising_bool_expr(
                stmt.test,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            and not _split_suppress_with_body(
                stmt.body,
                safe_names=safe_names,
                safe_bool_names=safe_bool_names,
                strict_names=strict_names,
            )[1]
            and not _split_suppress_with_body(
                stmt.orelse,
                safe_names=safe_names,
                safe_bool_names=safe_bool_names,
                strict_names=strict_names,
            )[1]
        )
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        if _is_exception_suppressing_with(stmt) and all(
            (
                item.optional_vars is None
                or isinstance(item.optional_vars, ast.Name)
            )
            and isinstance(item.context_expr, ast.Call)
            and not any(
                isinstance(a, ast.Starred) or getattr(a, "arg", "k") is None
                for a in (
                    *item.context_expr.args,
                    *item.context_expr.keywords,
                )
            )
            and all(
                _is_non_raising_expr(
                    a.value if isinstance(a, ast.keyword) else a,
                    safe_names | _SAFE_EXCEPTION_TYPE_NAMES,
                    safe_bool_names,
                    strict_names=strict_names,
                )
                for a in (
                    *item.context_expr.args,
                    *item.context_expr.keywords,
                )
            )
            for item in stmt.items
        ):
            return not _split_suppress_with_body(
                stmt.body,
                safe_names=safe_names,
                safe_bool_names=safe_bool_names,
                strict_names=strict_names,
            )[1]
        return False
    return False


def _split_suppress_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a potentially suppressing with-block body into (non_raising_prefix, potentially_raising_suffix)."""
    safe_prefix: List[ast.stmt] = []
    cur_safe: Set[str] = set(safe_names)
    cur_bool: Set[str] = set(safe_bool_names)
    for idx, b_stmt in enumerate(body):
        if _stmt_cannot_raise_in_suppress(
            b_stmt, cur_safe, cur_bool, strict_names=strict_names
        ):
            safe_prefix.append(b_stmt)
            if isinstance(b_stmt, (ast.Return, ast.Break, ast.Continue)):
                return safe_prefix, []
            if isinstance(b_stmt, ast.Delete):
                for tgt in b_stmt.targets:
                    if isinstance(tgt, ast.Name):
                        cur_safe.discard(tgt.id)
                        cur_bool.discard(tgt.id)
            elif isinstance(b_stmt, ast.Assign):
                val_bool = _is_non_raising_bool_expr(
                    b_stmt.value, cur_safe, cur_bool, strict_names=strict_names
                )
                for tgt in b_stmt.targets:
                    t_names = _extract_definite_target_names(tgt)
                    cur_safe.update(t_names)
                    if val_bool:
                        cur_bool.update(t_names)
                    else:
                        cur_bool.difference_update(t_names)
            elif isinstance(b_stmt, ast.AnnAssign) and b_stmt.value is not None:
                t_names = _extract_definite_target_names(b_stmt.target)
                cur_safe.update(t_names)
                if _is_non_raising_bool_expr(
                    b_stmt.value, cur_safe, cur_bool, strict_names=strict_names
                ):
                    cur_bool.update(t_names)
                else:
                    cur_bool.difference_update(t_names)
            elif isinstance(b_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cur_safe.add(b_stmt.name)
                cur_bool.add(b_stmt.name)
        else:
            return safe_prefix, list(body[idx:])
    return safe_prefix, []


def _stmt_contains_direct_raise(stmt: ast.AST) -> bool:
    """Returns True if stmt or any nested control-flow block (excluding nested defs/classes) contains ast.Raise."""
    stack: List[ast.AST] = [stmt]
    while stack:
        curr = stack.pop()
        if isinstance(curr, ast.Raise):
            return True
        if isinstance(
            curr,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        ):
            continue
        stack.extend(ast.iter_child_nodes(curr))
    return False


def _split_unknown_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits an unknown context manager's body into (guaranteed_non_raising_prefix, potentially_raising_suffix)."""
    return _split_suppress_with_body(
        body, safe_names=safe_names, strict_names=strict_names
    )


def _split_with_body_by_mode(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a with-block body into (guaranteed_prefix, potentially_suppressed_suffix) across 3 suppression states."""
    w_mode = _classify_with_suppression(
        stmt,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    if w_mode == "suppressing":
        return _split_suppress_with_body(stmt.body, safe_names=safe_names)
    if w_mode == "unknown":
        return _split_unknown_with_body(
            stmt.body, safe_names=safe_names, strict_names=strict_names
        )
    return list(stmt.body), []


def _walrus_in_sequence(
    items: Sequence[Optional[ast.AST]],
    is_conditional: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Evaluates walrus assignments sequentially across expressions with short-circuiting tail conditions."""
    definite: Set[str] = set()
    conditional: Set[str] = set()
    for idx, item in enumerate(items):
        if item is not None:
            d, c = _walrus_assignment_in_expr(
                item, is_conditional if idx == 0 else True
            )
            definite.update(d)
            conditional.update(c)
    return definite, conditional


def _walrus_assignment_in_expr(
    expr: Optional[ast.AST],
    is_conditional: bool = False,
) -> Tuple[Set[str], Set[str]]:
    """Analyzes walrus assignments in an expression to distinguish definite from conditional stores."""
    definite: Set[str] = set()
    conditional: Set[str] = set()
    if expr is None:
        return definite, conditional

    if isinstance(expr, _INNER_SCOPE_TYPES):
        for hdr in iter_nested_scope_header_nodes(expr):
            d_h, c_h = _walrus_assignment_in_expr(hdr, is_conditional)
            definite.update(d_h)
            conditional.update(c_h)
        if isinstance(expr, ast.Lambda):
            d_l, c_l = _walrus_assignment_in_expr(expr.body, True)
            conditional.update(d_l | c_l)
        return definite, conditional - definite

    if isinstance(expr, ast.NamedExpr) and isinstance(expr.target, ast.Name):
        if is_conditional:
            conditional.add(expr.target.id)
        else:
            definite.add(expr.target.id)

    if isinstance(expr, ast.BoolOp):
        d_seq, c_seq = _walrus_in_sequence(expr.values, is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        return definite, conditional - definite

    if isinstance(expr, ast.IfExp):
        d_t, c_t = _walrus_assignment_in_expr(expr.test, is_conditional)
        d_b, c_b = _walrus_assignment_in_expr(expr.body, True)
        d_o, c_o = _walrus_assignment_in_expr(expr.orelse, True)
        definite.update(d_t)
        conditional.update(c_t | d_b | c_b | (d_b & d_o) | d_o | c_o)
        if not is_conditional:
            definite.update(d_b & d_o)
        return definite, conditional - definite

    if isinstance(expr, ast.Compare):
        d_l, c_l = _walrus_assignment_in_expr(expr.left, is_conditional)
        definite.update(d_l)
        conditional.update(c_l)
        d_seq, c_seq = _walrus_in_sequence(expr.comparators, is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        return definite, conditional - definite

    if isinstance(expr, _COMPREHENSION_TYPES):
        d_seq, c_seq = _walrus_in_sequence(
            [g.iter for g in expr.generators], is_conditional
        )
        definite.update(d_seq)
        conditional.update(c_seq)
        for child in ast.iter_child_nodes(expr):
            if (
                isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp))
                and child is expr.elt
            ):
                d_c, c_c = _walrus_assignment_in_expr(child, True)
                conditional.update(d_c | c_c)
            elif isinstance(expr, ast.DictComp) and child in (
                expr.key,
                expr.value,
            ):
                d_c, c_c = _walrus_assignment_in_expr(child, True)
                conditional.update(d_c | c_c)
            elif isinstance(child, ast.comprehension):
                for if_expr in child.ifs:
                    d_c, c_c = _walrus_assignment_in_expr(if_expr, True)
                    conditional.update(d_c | c_c)
        return definite, conditional - definite

    for child in ast.iter_child_nodes(expr):
        target_child = (
            child.value if isinstance(child, ast.keyword) else child
        )
        if isinstance(target_child, ast.AST):
            d_ch, c_ch = _walrus_assignment_in_expr(
                target_child, is_conditional
            )
            definite.update(d_ch)
            conditional.update(c_ch)

    return definite, conditional - definite


def _extract_unconditional_walrus_names(node: ast.AST) -> Set[str]:
    """Extracts NamedExpr target names unconditionally evaluated in an AST node."""
    return _walrus_assignment_in_expr(node, is_conditional=False)[0]


def _extract_deleted_names(statements: Sequence[ast.AST]) -> Set[str]:
    """Finds all variable names deleted within a statement sequence without recursing into nested functions."""
    deleted: Set[str] = set()
    for stmt in statements:
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
            deleted.update(_extract_deleted_names(stmt.body))
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            for case in getattr(stmt, "cases", []):
                deleted.update(
                    _extract_deleted_names(getattr(case, "body", []))
                )
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, (ast.stmt, ast.ExceptHandler)):
                    deleted.update(_extract_deleted_names([child]))
    return deleted


def _demote_deletions_to_conditional(
    deleted_names: Iterable[str],
    definite: Set[str],
    conditional: Set[str],
) -> None:
    """Discards deleted variables from definite and promotes previously definite variables to conditional."""
    for v_del in deleted_names:
        if v_del in definite:
            definite.discard(v_del)
            conditional.add(v_del)


def _block_terminates(
    statements: Sequence[ast.AST],
    *,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    loop_control_terminates: bool = False,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if the statement sequence unconditionally terminates."""
    local_prov = _coerce_suppress_provenance(suppress_aliases)
    cur_safe: Set[str] = set(safe_names)
    for stmt in statements:
        if isinstance(stmt, (ast.Return, ast.Raise)) or (
            loop_control_terminates
            and isinstance(stmt, (ast.Break, ast.Continue))
        ):
            return True
        if isinstance(stmt, ast.If):
            test_prov = _update_suppress_provenance_from_expr_walrus(
                (stmt.test,), local_prov
            )
            if (
                stmt.orelse
                and _block_terminates(
                    stmt.body,
                    suppress_aliases=test_prov,
                    loop_control_terminates=loop_control_terminates,
                    safe_names=cur_safe,
                    strict_names=strict_names,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                and _block_terminates(
                    stmt.orelse,
                    suppress_aliases=test_prov,
                    loop_control_terminates=loop_control_terminates,
                    safe_names=cur_safe,
                    strict_names=strict_names,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
            ):
                return True
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            item_prov = local_prov
            for item in stmt.items:
                item_prov = _update_suppress_provenance_from_expr_walrus(
                    (item.context_expr,), item_prov
                )
                if item.optional_vars is not None:
                    cur_safe.update(
                        _extract_definite_target_names(item.optional_vars)
                    )
                    for t_n in _extract_store_names_in_node(
                        item.optional_vars
                    ):
                        item_prov = item_prov.clear_name(t_n)
            w_body = _split_with_body_by_mode(
                stmt,
                item_prov,
                safe_names=cur_safe,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )[0]
            if _block_terminates(
                w_body,
                suppress_aliases=item_prov,
                loop_control_terminates=loop_control_terminates,
                safe_names=cur_safe,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            ):
                return True
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            cases = getattr(stmt, "cases", [])
            if cases:
                reachable_cases: List[ast.AST] = []
                has_irrefutable = False
                for c in cases:
                    reachable_cases.append(c)
                    if _is_irrefutable_case(c):
                        has_irrefutable = True
                        break
                if has_irrefutable and all(
                    _block_terminates(
                        getattr(c, "body", ()),
                        suppress_aliases=local_prov,
                        loop_control_terminates=loop_control_terminates,
                        safe_names=cur_safe,
                        strict_names=strict_names,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                    for c in reachable_cases
                ):
                    return True
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            t_body_seq = getattr(stmt, "body", [])
            try_prov = local_prov
            for b_stmt in t_body_seq:
                try_prov = _update_suppress_provenance_from_stmt(
                    b_stmt, try_prov
                )
            handler_prov = local_prov.merge_branch_conservative(try_prov)
            f_body = getattr(stmt, "finalbody", [])
            if f_body and _block_terminates(
                f_body,
                suppress_aliases=handler_prov,
                loop_control_terminates=loop_control_terminates,
                safe_names=cur_safe,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            ):
                return True
            handlers = getattr(stmt, "handlers", [])
            o_stmts = getattr(stmt, "orelse", [])
            body_terminates = _block_terminates(
                t_body_seq,
                suppress_aliases=local_prov,
                loop_control_terminates=loop_control_terminates,
                safe_names=cur_safe,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            ) or (
                bool(o_stmts)
                and _block_terminates(
                    o_stmts,
                    suppress_aliases=try_prov,
                    loop_control_terminates=loop_control_terminates,
                    safe_names=cur_safe,
                    strict_names=strict_names,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
            )
            if not handlers and body_terminates:
                return True
            if (
                handlers
                and type(stmt).__name__ != "TryStar"
                and body_terminates
                and all(
                    _block_terminates(
                        getattr(h, "body", []),
                        suppress_aliases=(
                            handler_prov.clear_name(h.name)
                            if isinstance(getattr(h, "name", None), str)
                            else handler_prov
                        ),
                        loop_control_terminates=loop_control_terminates,
                        safe_names=cur_safe,
                        strict_names=strict_names,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                    for h in handlers
                )
            ):
                return True
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
        else:
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    cur_safe.update(_extract_definite_target_names(tgt))
            elif (
                isinstance(stmt, (ast.AnnAssign, ast.AugAssign))
                and stmt.value is not None
            ):
                cur_safe.update(_extract_definite_target_names(stmt.target))
            elif isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                cur_safe.add(stmt.name)
            elif isinstance(stmt, ast.Delete):
                cur_safe.difference_update(_extract_shallow_deleted_names(stmt))
    return False


def _filter_scope_names(
    names: Iterable[str],
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
) -> Set[str]:
    """Filters candidate bound names by scope nonlocal/global rules when active."""
    if active_nonlocals is None and not fn_globals:
        return set(names)
    return {
        n
        for n in names
        if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
    }


def _analyze_block_assignment_impl(
    statements: Sequence[ast.AST],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    active_nonlocals: Optional[Set[str]] = None,
    fn_globals: AbstractSet[str] = frozenset(),
    loop_control_terminates: bool = False,
    known_safe_names: AbstractSet[str] = frozenset(),
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[Set[str], Set[str], bool, _SuppressProvenance]:
    """Unified single-pass analyzer for definite/conditional bindings, deletions, and termination."""
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()
    may_del = may_deleted if may_deleted is not None else set()
    live_in_del = live_in_deleted if live_in_deleted is not None else set()
    local_prov = _coerce_suppress_provenance(suppress_aliases)

    def _bind_def_cond(
        d_raw: Iterable[str], c_raw: Iterable[str] = ()
    ) -> None:
        d_filt = _filter_scope_names(d_raw, active_nonlocals, fn_globals)
        c_filt = _filter_scope_names(c_raw, active_nonlocals, fn_globals)
        if d_filt:
            definite.update(d_filt)
            may_del.difference_update(d_filt)
        if c_filt:
            conditional.update(c_filt)

    def _apply_expr_effects(
        expr_node: Optional[ast.AST], *, is_cond: bool = False
    ) -> None:
        if not isinstance(expr_node, ast.AST):
            return
        d_expr, c_expr = _walrus_assignment_in_expr(
            expr_node, is_conditional=is_cond
        )
        _bind_def_cond(d_expr, c_expr)

    for stmt in statements:
        before_def = set(definite)
        if isinstance(stmt, ast.Assign):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            for target in stmt.targets:
                _bind_def_cond(_extract_definite_target_names(target))
                _apply_expr_effects(target)
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            if stmt.value is not None:
                _bind_def_cond(_extract_definite_target_names(stmt.target))
                _apply_expr_effects(stmt.target)
                _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            _bind_def_cond(_extract_definite_target_names(stmt.target))
            _apply_expr_effects(stmt.target)
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.Expr):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.Delete):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            for target in stmt.targets:
                _apply_expr_effects(target)
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(
                        node.ctx, ast.Del
                    ):
                        if node.id not in definite:
                            live_in_del.add(node.id)
                        definite.discard(node.id)
                        conditional.discard(node.id)
                        may_del.add(node.id)
        elif isinstance(stmt, ast.Raise):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            _apply_expr_effects(stmt.exc)
            _apply_expr_effects(stmt.cause)
            may_del.difference_update(definite - before_def)
            return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, ast.Return):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            _apply_expr_effects(stmt.value)
            may_del.difference_update(definite - before_def)
            return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            if loop_control_terminates:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, ast.If):
            local_prov = _update_suppress_provenance_from_expr_walrus(
                (stmt.test,), local_prov
            )
            _apply_expr_effects(stmt.test)
            cur_safe = definite | set(known_safe_names)
            b_may: Set[str] = set()
            b_live: Set[str] = set()
            o_may: Set[str] = set()
            o_live: Set[str] = set()
            b_def, b_cond, b_term, b_prov = _analyze_block_assignment_impl(
                stmt.body,
                may_deleted=b_may,
                live_in_deleted=b_live,
                suppress_aliases=local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=cur_safe,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if stmt.orelse:
                o_def, o_cond, o_term, o_prov = _analyze_block_assignment_impl(
                    stmt.orelse,
                    may_deleted=o_may,
                    live_in_deleted=o_live,
                    suppress_aliases=local_prov,
                    active_nonlocals=active_nonlocals,
                    fn_globals=fn_globals,
                    loop_control_terminates=loop_control_terminates,
                    known_safe_names=cur_safe,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
            else:
                o_def, o_cond, o_term, o_prov = (
                    set(),
                    set(),
                    False,
                    local_prov,
                )
            local_prov = local_prov.merge_branch_conservative(
                b_prov
            ).merge_branch_conservative(o_prov)
            live_in_del.update((b_live | o_live) - definite)
            eff_b_del = (
                _extract_deleted_names(stmt.body)
                | set().union(
                    *(_extract_shallow_deleted_names(s) for s in stmt.body)
                )
                if stmt.body
                else set()
            ) - b_def
            eff_o_del = (
                _extract_deleted_names(stmt.orelse)
                | set().union(
                    *(_extract_shallow_deleted_names(s) for s in stmt.orelse)
                )
                if stmt.orelse
                else set()
            ) - o_def
            if stmt.orelse:
                if o_term and not b_term:
                    definite.update(b_def)
                    conditional.update((b_def | b_cond) - definite)
                    may_del.difference_update(b_def)
                    may_del.update(b_may)
                elif b_term and not o_term:
                    definite.update(o_def)
                    conditional.update((o_def | o_cond) - definite)
                    may_del.difference_update(o_def)
                    may_del.update(o_may)
                elif not b_term and not o_term:
                    both = b_def & o_def
                    definite.update(both)
                    conditional.update(
                        (b_def | b_cond | o_def | o_cond) - definite
                    )
                    may_del.difference_update(both)
                    may_del.update(b_may | o_may)
                else:
                    both = b_def & o_def
                    definite.update(both)
                    conditional.update(
                        (b_def | b_cond | o_def | o_cond) - definite
                    )
                    may_del.difference_update(both)
                    may_del.update(b_may | o_may)
                for v_del in eff_b_del & eff_o_del:
                    definite.discard(v_del)
                    conditional.discard(v_del)
                _demote_deletions_to_conditional(
                    eff_b_del ^ eff_o_del, definite, conditional
                )
                if b_term and o_term:
                    may_del.difference_update(definite - before_def)
                    return definite, conditional - definite, True, local_prov
            else:
                if not b_term:
                    conditional.update((b_def | b_cond) - definite)
                    may_del.update(b_may)
                    _demote_deletions_to_conditional(
                        eff_b_del, definite, conditional
                    )
        elif isinstance(stmt, ast.While):
            local_prov = _update_suppress_provenance_from_expr_walrus(
                (stmt.test,), local_prov
            )
            _apply_expr_effects(stmt.test)
            sub_stmts = list(stmt.body)
            if stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            sub_may: Set[str] = set()
            sub_live: Set[str] = set()
            sub_def, sub_cond, _, sub_prov = _analyze_block_assignment_impl(
                sub_stmts,
                may_deleted=sub_may,
                live_in_deleted=sub_live,
                suppress_aliases=local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=definite | set(known_safe_names),
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            local_prov = local_prov.merge_branch_conservative(sub_prov)
            live_in_del.update(sub_live - definite)
            may_del.update(sub_may)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(
                _extract_deleted_names(sub_stmts), definite, conditional
            )
            definite.difference_update(_extract_shallow_deleted_names(stmt))
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            local_prov = _update_suppress_provenance_from_expr_walrus(
                (stmt.iter, stmt.target), local_prov
            )
            _apply_expr_effects(stmt.iter)
            t_names: Set[str] = set()
            for p_node in ast.walk(stmt.target):
                if isinstance(p_node, ast.Name) and isinstance(
                    p_node.ctx, ast.Store
                ):
                    t_names.add(p_node.id)
                    local_prov = local_prov.clear_name(p_node.id)
            _bind_def_cond((), t_names)
            sub_stmts = list(stmt.body)
            if stmt.orelse:
                sub_stmts.extend(stmt.orelse)
            for_may: Set[str] = set()
            for_live: Set[str] = set()
            sub_def, sub_cond, _, sub_prov = _analyze_block_assignment_impl(
                sub_stmts,
                may_deleted=for_may,
                live_in_deleted=for_live,
                suppress_aliases=local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=definite | set(known_safe_names) | t_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            local_prov = local_prov.merge_branch_conservative(sub_prov)
            live_in_del.update((for_live - t_names) - definite)
            may_del.update(for_may)
            conditional.update((sub_def | sub_cond) - definite)
            _demote_deletions_to_conditional(
                _extract_deleted_names(sub_stmts), definite, conditional
            )
            definite.difference_update(_extract_shallow_deleted_names(stmt))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                local_prov = _update_suppress_provenance_from_expr_walrus(
                    (item.context_expr,), local_prov
                )
                _apply_expr_effects(item.context_expr)
                if item.optional_vars is not None:
                    opt_names = _extract_definite_target_names(
                        item.optional_vars
                    )
                    _bind_def_cond(opt_names)
                    for o_n in _extract_store_names_in_node(
                        item.optional_vars
                    ):
                        local_prov = local_prov.clear_name(o_n)
            cur_safe = definite | set(known_safe_names)
            safe_prefix, risky_suffix = _split_with_body_by_mode(
                stmt,
                local_prov,
                safe_names=cur_safe,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            _, _, w_term, local_prov = _analyze_block_assignment_impl(
                safe_prefix,
                definite,
                conditional,
                may_deleted=may_del,
                live_in_deleted=live_in_del,
                suppress_aliases=local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=cur_safe,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if w_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
            if risky_suffix:
                w_may: Set[str] = set()
                w_live: Set[str] = set()
                w_def, w_cond, _, w_prov = _analyze_block_assignment_impl(
                    risky_suffix,
                    may_deleted=w_may,
                    live_in_deleted=w_live,
                    suppress_aliases=local_prov,
                    active_nonlocals=active_nonlocals,
                    fn_globals=fn_globals,
                    loop_control_terminates=loop_control_terminates,
                    known_safe_names=definite | set(known_safe_names),
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                local_prov = local_prov.merge_branch_conservative(w_prov)
                live_in_del.update(w_live - definite)
                may_del.update(w_may)
                conditional.update((w_def | w_cond) - definite)
                _demote_deletions_to_conditional(
                    _extract_deleted_names(risky_suffix),
                    definite,
                    conditional,
                )
                for b_stmt in risky_suffix:
                    definite.difference_update(
                        _extract_shallow_deleted_names(b_stmt)
                    )
        elif isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
            t_body = list(getattr(stmt, "body", []))
            raw_try_del = _extract_deleted_names(t_body) | (
                set().union(*(_extract_shallow_deleted_names(s) for s in t_body))
                if t_body
                else set()
            )
            is_trystar = type(stmt).__name__ == "TryStar"
            cur_safe = definite | set(known_safe_names)
            t_may: Set[str] = set()
            t_live: Set[str] = set()
            t_def, t_cond, t_term, try_body_prov = (
                _analyze_block_assignment_impl(
                    t_body,
                    may_deleted=t_may,
                    live_in_deleted=t_live,
                    suppress_aliases=local_prov,
                    active_nonlocals=active_nonlocals,
                    fn_globals=fn_globals,
                    loop_control_terminates=loop_control_terminates,
                    known_safe_names=cur_safe,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
            )
            for b_stmt in t_body:
                try_body_prov = _update_suppress_provenance_from_stmt(
                    b_stmt, try_body_prov, in_branch=True
                )
            handler_base_prov = local_prov.merge_branch_conservative(
                try_body_prov
            )
            live_in_del.update(t_live - definite)
            o_stmts = list(getattr(stmt, "orelse", []))
            o_try_may: Set[str] = set()
            o_def = set()
            if o_stmts:
                o_try_live: Set[str] = set()
                o_def, o_cond, o_term, try_body_prov = (
                    _analyze_block_assignment_impl(
                        o_stmts,
                        may_deleted=o_try_may,
                        live_in_deleted=o_try_live,
                        suppress_aliases=try_body_prov,
                        active_nonlocals=active_nonlocals,
                        fn_globals=fn_globals,
                        loop_control_terminates=loop_control_terminates,
                        known_safe_names=cur_safe | t_def,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                )
                for o_stmt in o_stmts:
                    try_body_prov = _update_suppress_provenance_from_stmt(
                        o_stmt, try_body_prov, in_branch=True
                    )
                if not t_term:
                    live_in_del.update(o_try_live - (definite | t_def))
                if not o_term:
                    try_def = t_def | o_def
                    try_cond = (t_cond | o_cond) - try_def
                    succ_may = (t_may - o_def) | o_try_may
                else:
                    try_def = t_def | o_def
                    try_cond = (t_cond | o_cond) - try_def
                    succ_may = set()
                    t_term = True
            else:
                try_def = t_def
                try_cond = t_cond
                succ_may = t_may

            handlers = list(getattr(stmt, "handlers", []))
            h_defs: List[Set[str]] = []
            h_dels: List[Tuple[Set[str], Set[str], bool]] = []
            h_mays: List[Set[str]] = []
            final_prov = handler_base_prov.merge_branch_conservative(
                try_body_prov
            )

            for h in handlers:
                h_body = list(getattr(h, "body", []))
                if getattr(h, "type", None) is not None:
                    _apply_expr_effects(h.type, is_cond=True)
                h_prov = handler_base_prov
                h_name = getattr(h, "name", None)
                h_bound: Set[str] = set()
                if isinstance(h_name, str):
                    h_prov = h_prov.clear_name(h_name)
                    if _can_bind_in_fn_scope(
                        h_name, active_nonlocals, fn_globals
                    ):
                        h_bound.add(h_name)
                h_may: Set[str] = set()
                h_live: Set[str] = set()
                h_d, h_c, h_raw_term, h_out_prov = (
                    _analyze_block_assignment_impl(
                        h_body,
                        may_deleted=h_may,
                        live_in_deleted=h_live,
                        suppress_aliases=h_prov,
                        active_nonlocals=active_nonlocals,
                        fn_globals=fn_globals,
                        loop_control_terminates=loop_control_terminates,
                        known_safe_names=(cur_safe - raw_try_del) | h_bound,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                )
                for h_stmt in h_body:
                    h_out_prov = _update_suppress_provenance_from_stmt(
                        h_stmt, h_out_prov, in_branch=True
                    )
                final_prov = final_prov.merge_branch_conservative(h_out_prov)
                h_d.difference_update(h_bound)
                h_raw_bound = {h_name} if isinstance(h_name, str) else set()
                h_may.update(h_raw_bound)
                live_in_del.update((h_live - h_raw_bound) - definite)
                h_term = not is_trystar and h_raw_term
                h_body_dels = _extract_deleted_names(h_body) | (
                    set().union(
                        *(_extract_shallow_deleted_names(s) for s in h_body)
                    )
                    if h_body
                    else set()
                )
                eff_h_del = (h_body_dels - h_d) | h_raw_bound
                reach_h_del = eff_h_del | (raw_try_del - h_d)
                h_dels.append((eff_h_del, reach_h_del, h_term))
                if not h_term:
                    h_defs.append(h_d)
                    h_mays.append(h_may | ((raw_try_del | t_may) - h_d))
                conditional.update(((h_d | h_c) - h_raw_bound) - definite)

            all_handlers_terminate = bool(
                handlers
                and not is_trystar
                and all(h_term for _, _, h_term in h_dels)
            )
            if is_trystar and handlers:
                conditional.update(
                    (t_def | t_cond | try_def | try_cond) - definite
                )
                may_del.update(t_may | o_try_may | raw_try_del)
                for hm in h_mays:
                    may_del.update(hm)
            elif not t_term:
                if all_handlers_terminate:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
                    may_del.difference_update(try_def)
                    may_del.update(succ_may)
                elif handlers and h_defs:
                    common = try_def.intersection(*h_defs)
                    definite.update(common)
                    conditional.update((try_def | try_cond) - definite)
                    may_del.difference_update(common)
                    may_del.update(succ_may)
                    for hm in h_mays:
                        may_del.update(hm)
                elif not handlers:
                    definite.update(try_def)
                    conditional.update(try_cond - definite)
                    may_del.difference_update(try_def)
                    may_del.update(succ_may)
            elif not handlers:
                definite.update(try_def)
                conditional.update(try_cond - definite)
                may_del.difference_update(try_def)
            elif h_defs:
                common = set.intersection(*h_defs)
                definite.update(common)
                conditional.update(try_cond - definite)
                may_del.difference_update(common)
                for hm in h_mays:
                    may_del.update(hm)

            eff_try_del = raw_try_del - t_def
            eff_o_del = (
                (
                    _extract_deleted_names(o_stmts)
                    | set().union(
                        *(_extract_shallow_deleted_names(s) for s in o_stmts)
                    )
                )
                - o_def
                if o_stmts
                else set()
            )
            succ_del = eff_try_del | eff_o_del

            surviving_uncond_dels: List[Set[str]] = []
            surviving_reach_dels: List[Set[str]] = []
            all_try_dels = succ_del | raw_try_del
            if not t_term and not (is_trystar and handlers):
                surviving_uncond_dels.append(succ_del)
                surviving_reach_dels.append(succ_del)

            for eff_h_del, reach_h_del, h_term in h_dels:
                all_try_dels.update(reach_h_del)
                if not h_term and not is_trystar:
                    surviving_uncond_dels.append(eff_h_del)
                    surviving_reach_dels.append(reach_h_del)

            f_body = list(getattr(stmt, "finalbody", []))
            reach_dels: Set[str] = (
                all_try_dels
                if (f_body or is_trystar or not surviving_reach_dels)
                else set().union(*surviving_reach_dels)
            )
            if surviving_uncond_dels and not is_trystar:
                uncond_dels = set.intersection(*surviving_uncond_dels)
                for v_del in uncond_dels:
                    definite.discard(v_del)
                    conditional.discard(v_del)
                _demote_deletions_to_conditional(
                    reach_dels - uncond_dels, definite, conditional
                )
            else:
                _demote_deletions_to_conditional(
                    all_try_dels, definite, conditional
                )
            if is_trystar and handlers:
                definite.difference_update(
                    _extract_shallow_deleted_names(stmt)
                )

            f_term = False
            if f_body:
                _, _, f_term, f_out_prov = _analyze_block_assignment_impl(
                    f_body,
                    definite,
                    conditional,
                    may_deleted=may_del,
                    live_in_deleted=live_in_del,
                    suppress_aliases=final_prov,
                    active_nonlocals=active_nonlocals,
                    fn_globals=fn_globals,
                    loop_control_terminates=loop_control_terminates,
                    known_safe_names=definite | set(known_safe_names),
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                local_prov = final_prov.merge_branch_conservative(f_out_prov)
            else:
                local_prov = final_prov

            try_terminates = f_term or (
                t_term and (not handlers or all_handlers_terminate)
            )
            if try_terminates:
                if handlers and not f_body and loop_control_terminates:
                    definite.clear()
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif type(stmt).__name__ == "Match":  # pragma: no cover (py310+)
            subject = getattr(stmt, "subject", None)
            if isinstance(subject, ast.AST):
                local_prov = _update_suppress_provenance_from_expr_walrus(
                    (subject,), local_prov
                )
                _apply_expr_effects(subject)
            cur_safe = definite | set(known_safe_names)
            case_defs: List[Set[str]] = []
            non_term_case_defs: List[Set[str]] = []
            case_conds: Set[str] = set()
            case_dels: Set[str] = set()
            non_term_case_dels: List[Set[str]] = []
            has_irrefutable_default = False
            merged_match_prov = local_prov
            for case in getattr(stmt, "cases", []):
                raw_pattern_stores: Set[str] = set()
                pattern = getattr(case, "pattern", None)
                if isinstance(pattern, ast.AST):
                    for p_node in ast.walk(pattern):
                        p_name = _extract_pattern_bound_name(p_node)
                        if p_name is not None:
                            raw_pattern_stores.add(p_name)
                pattern_stores = _filter_scope_names(
                    raw_pattern_stores, active_nonlocals, fn_globals
                )
                guard = getattr(case, "guard", None)
                d_g: Set[str] = set()
                c_g: Set[str] = set()
                if isinstance(guard, ast.AST):
                    d_g_raw, c_g_raw = _walrus_assignment_in_expr(
                        guard, is_conditional=False
                    )
                    d_g = _filter_scope_names(
                        d_g_raw, active_nonlocals, fn_globals
                    )
                    c_g = _filter_scope_names(
                        c_g_raw, active_nonlocals, fn_globals
                    )
                case_init_def = pattern_stores | d_g
                case_prov = local_prov
                for p_s in raw_pattern_stores:
                    case_prov = case_prov.clear_name(p_s)
                c_may: Set[str] = set()
                c_live: Set[str] = set()
                case_body = getattr(case, "body", [])
                c_def, c_cond, c_term, c_out_prov = (
                    _analyze_block_assignment_impl(
                        case_body,
                        definite=case_init_def.copy(),
                        conditional=c_g.copy() if c_g else None,
                        may_deleted=c_may,
                        live_in_deleted=c_live,
                        suppress_aliases=case_prov,
                        active_nonlocals=active_nonlocals,
                        fn_globals=fn_globals,
                        loop_control_terminates=loop_control_terminates,
                        known_safe_names=cur_safe | case_init_def,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                )
                merged_match_prov = (
                    merged_match_prov.merge_branch_conservative(c_out_prov)
                )
                live_in_del.update((c_live - case_init_def) - definite)
                case_defs.append(c_def)
                case_conds.update(c_def | c_cond)
                c_raw_del = _extract_deleted_names(case_body) | (
                    set().union(
                        *(
                            _extract_shallow_deleted_names(s)
                            for s in case_body
                        )
                    )
                    if case_body
                    else set()
                )
                c_del = c_raw_del - c_def
                if not c_term:
                    case_dels.update(c_del)
                    non_term_case_defs.append(c_def)
                    non_term_case_dels.append(c_del)
                    may_del.update(c_may - raw_pattern_stores)
                if _is_irrefutable_case(case):
                    has_irrefutable_default = True
                    break
            local_prov = merged_match_prov
            if case_defs and has_irrefutable_default:
                if non_term_case_defs:
                    common = set.intersection(*non_term_case_defs)
                    definite.update(common)
                    may_del.difference_update(common)
                conditional.update(case_conds - definite)
            else:
                conditional.update(case_conds - definite)
            all_del = (
                {
                    v
                    for v in case_dels
                    if all(v in d for d in non_term_case_dels)
                }
                if (has_irrefutable_default and non_term_case_dels)
                else set()
            )
            for v_del in all_del:
                definite.discard(v_del)
                conditional.discard(v_del)
            _demote_deletions_to_conditional(
                case_dels - all_del, definite, conditional
            )
            if (
                case_defs
                and has_irrefutable_default
                and not non_term_case_defs
            ):
                if loop_control_terminates:
                    definite.clear()
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            for hdr in iter_nested_scope_header_nodes(stmt):
                _apply_expr_effects(hdr)
            _bind_def_cond((stmt.name,))
        elif type(stmt).__name__ == "TypeAlias":  # pragma: no cover (py312+)
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            ta_name = _extract_type_alias_or_pattern_name(stmt)
            if ta_name is not None:
                _bind_def_cond((ta_name,))
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            imp_names: Set[str] = set()
            _record_import_names(stmt, imp_names)
            _bind_def_cond(imp_names)
        elif isinstance(stmt, ast.Assert):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )
            _apply_expr_effects(stmt.test)
            if stmt.msg is not None:
                _apply_expr_effects(stmt.msg, is_cond=True)
        else:
            local_prov = _update_suppress_provenance_from_stmt(
                stmt, local_prov
            )

        may_del.difference_update(definite - before_def)

    return definite, conditional - definite, False, local_prov


def _analyze_block_assignment(
    statements: Sequence[ast.stmt],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[Set[str], Set[str]]:
    """Analyzes a sequence of statements to determine unconditionally and conditionally assigned variables."""
    def_out, cond_out, _, _ = _analyze_block_assignment_impl(
        statements,
        definite=definite,
        conditional=conditional,
        may_deleted=may_deleted,
        live_in_deleted=live_in_deleted,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    return def_out, cond_out


def _extract_block_definite_bindings_and_term(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[Set[str], bool]:
    """Extracts (unconditionally_bound_names, block_terminates) across a statement sequence."""
    def_out, _, term, _ = _analyze_block_assignment_impl(
        stmts,
        definite=set(initial_bound),
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=True,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    return def_out, term


def _extract_block_definite_bindings(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Set[str]:
    """Extracts variable names unconditionally bound across a statement sequence."""
    return _extract_block_definite_bindings_and_term(
        stmts,
        active_nonlocals,
        initial_bound,
        fn_globals,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )[0]


def _apply_expr_definite_effects(
    target_bound: Set[str],
    expr_node: Any,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
) -> None:
    """Applies shallow deletions and unconditional walrus bindings from an AST node."""
    if not isinstance(expr_node, ast.AST):
        return
    target_bound.difference_update(_extract_shallow_deleted_names(expr_node))
    target_bound.update(
        n
        for n in _extract_unconditional_walrus_names(expr_node)
        if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)
    )


def _eval_try_definite_bindings(
    stmt: ast.AST,
    active_nonlocals: Optional[Set[str]],
    bound: Set[str],
    fn_globals: AbstractSet[str],
    *,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> Tuple[Set[str], bool]:
    """Computes definite bindings and termination across a Try/TryStar statement."""
    return _extract_block_definite_bindings_and_term(
        (stmt,),
        active_nonlocals,
        bound,
        fn_globals,
        suppress_aliases=suppress_aliases,
    )


def _eval_match_definite_bindings(
    stmt: ast.AST,
    active_nonlocals: Optional[Set[str]],
    bound: Set[str],
    fn_globals: AbstractSet[str],
    *,
    suppress_aliases: Union[
        _SuppressProvenance, AbstractSet[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> Tuple[Set[str], bool]:
    """Computes definite bindings and termination across a Match statement."""
    return _extract_block_definite_bindings_and_term(
        (stmt,),
        active_nonlocals,
        bound,
        fn_globals,
        suppress_aliases=suppress_aliases,
    )
