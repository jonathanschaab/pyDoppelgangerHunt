"""Context-manager suppression provenance, block termination, and unified definite-assignment engine."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import logging
import re
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
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
from pydoppelgangerhunt.fixer._ast_compat import (
    is_match_node,
    is_try_or_trystar_node,
    is_trystar_node,
    is_type_alias_node,
)
from pydoppelgangerhunt.fixer._df_cache import (
    _get_or_compute_scope_cached,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _scope_cell_info_cache,
    _tree_contains_scope_node,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _FnScopeContext,
    _can_bind_in_fn_scope,
    _extract_store_names_in_node,
)
from pydoppelgangerhunt.fixer._df_span_index import (
    _extract_type_alias_or_pattern_name,
    _find_enclosing_function_chain,
)
from pydoppelgangerhunt.fixer._df_closures import (
    _COMPREHENSION_TYPES,
    _INNER_SCOPE_TYPES,
    _extract_node_type_param_names,
    _extract_scope_local_names,
    _extract_single_node_local_names,
    _is_irrefutable_case,
    _iter_shallow_scope_nodes,
    _node_has_star_import,
)
from pydoppelgangerhunt.fixer._df_visitors import (
    _collect_func_params as _extract_arg_names,
    _extract_pattern_bound_name,
    _record_import_names,
)
from pydoppelgangerhunt.parser import iter_nested_scope_header_nodes

logger = logging.getLogger(__name__)

_SUPPRESSING_CTX_MANAGER_NAMES: frozenset[str] = frozenset(
    {
        "suppress",
        "raises",
        "assertRaises",
        "assertRaisesRegex",
        "ignore",
        "ignored",
        "subTest",
        "ExitStack",
        "AsyncExitStack",
        "enter_context",
        "CancelScope",
        "move_on_after",
        "fail_after",
        "catch",
    }
)

_STDLIB_NON_SUPPRESSING_MODULES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "builtins": frozenset({"open"}),
        "__builtins__": frozenset({"open"}),
        "io": frozenset(
            {
                "open",
                "StringIO",
                "BytesIO",
                "TextIOWrapper",
                "BufferedReader",
                "BufferedWriter",
                "BufferedRandom",
                "BufferedRWPair",
                "FileIO",
            }
        ),
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
        "warnings": frozenset({"catch_warnings"}),
        "threading": frozenset(
            {
                "Lock",
                "RLock",
                "Condition",
                "Semaphore",
                "BoundedSemaphore",
            }
        ),
        "asyncio": frozenset(
            {
                "Lock",
                "Condition",
                "Semaphore",
                "BoundedSemaphore",
            }
        ),
        "decimal": frozenset({"localcontext"}),
        "unittest.mock": frozenset({"patch"}),
        "mock": frozenset({"patch"}),
        "os": frozenset({"scandir"}),
        "mmap": frozenset({"mmap"}),
        "socket": frozenset({"socket", "create_connection"}),
        "sqlite3": frozenset({"connect"}),
        "zipfile": frozenset({"ZipFile"}),
        "tarfile": frozenset({"TarFile", "open"}),
        "gzip": frozenset({"open", "GzipFile"}),
        "bz2": frozenset({"open", "BZ2File"}),
        "lzma": frozenset({"open", "LZMAFile"}),
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

_SAFE_EXIT_BASE_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "object",
        "AbstractContextManager",
        "AsyncContextManager",
        "ContextManager",
    }
)

_PATHLIB_CLASS_NAMES: frozenset[str] = frozenset(
    {"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"}
)

_TRACKED_SHADOW_NAMES: frozenset[str] = frozenset(
    {
        *_SHADOWABLE_BUILTIN_CTX_NAMES,
        *_SUPPRESSING_CTX_MANAGER_NAMES,
        *_STDLIB_NON_SUPPRESSING_MODULES.keys(),
        *_SAFE_EXIT_BASE_CLASS_NAMES,
        *_PATHLIB_CLASS_NAMES,
        "pathlib",
        "nullcontext",
        "contextlib",
        "typing",
    }
)

_SUSPICIOUS_SUPPRESS_SUBSTRINGS: Tuple[str, ...] = (
    "suppress",
    "raises",
    "ignor",
    "subtest",
    "exitstack",
    "enter_context",
    "enter_async_context",
    "cancelscope",
    "move_on_after",
    "fail_after",
    "catch",
    "swallow",
    "silen",
    "quiet",
    "toler",
    "best_effort",
    "absorb",
    "trap",
    "fallback",
    "noop",
    "no_op",
    "shield",
    "retry",
    "attempt",
    "skip",
    "mute",
    "muting",
    "maybe",
    "hush",
    "discard",
    "forgiv",
    "squelch",
    "quench",
    "mask",
)

_IDENT_WORD_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+")

_SUSPICIOUS_PREFIX_STEMS: Tuple[str, ...] = (
    "suppress",
    "ignor",
    "swallow",
    "silen",
    "quiet",
    "absorb",
    "forgiv",
    "squelch",
    "quench",
)

_SUSPICIOUS_COMPOUND_TOKEN_SPLITS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "subtest": ("sub", "test"),
        "exitstack": ("exit", "stack"),
        "cancelscope": ("cancel", "scope"),
        "fallback": ("fall", "back"),
        "noop": ("no", "op"),
    }
)

_SUSPICIOUS_EXACT_INFLECTIONS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "toler": ("tolerate", "tolerates", "tolerated", "tolerating", "tolerant"),
        "catch": ("catches", "catching", "catcher"),
        "trap": ("traps", "trapped", "trapping", "trapper"),
        "shield": ("shields", "shielded", "shielding"),
        "retry": ("retries", "retrying", "retried", "retrier"),
        "attempt": ("attempts", "attempted", "attempting"),
        "skip": ("skips", "skipped", "skipping", "skipper"),
        "mute": ("mutes", "muted"),
        "hush": ("hushes", "hushed", "hushing"),
        "discard": ("discards", "discarded", "discarding"),
        "mask": ("masks", "masking"),
        "fallback": ("fallbacks",),
        "noop": ("noops",),
        "subtest": ("subtests",),
        "exitstack": ("exitstacks",),
        "cancelscope": ("cancelscopes",),
    }
)


def _build_suspicious_token_tables() -> Tuple[frozenset[str], Tuple[Tuple[str, ...], ...]]:
    """Derives exact single-token and multi-token match tables from _SUSPICIOUS_SUPPRESS_SUBSTRINGS."""
    prefix_set = frozenset(_SUSPICIOUS_PREFIX_STEMS)
    exact_tokens: Set[str] = set()
    token_seqs: List[Tuple[str, ...]] = []
    seen_seqs: Set[Tuple[str, ...]] = set()
    for entry in _SUSPICIOUS_SUPPRESS_SUBSTRINGS:
        if "_" in entry:
            seq = tuple(entry.split("_"))
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                token_seqs.append(seq)
            continue
        if entry not in prefix_set:
            exact_tokens.add(entry)
            exact_tokens.update(_SUSPICIOUS_EXACT_INFLECTIONS.get(entry, ()))
        comp_seq = _SUSPICIOUS_COMPOUND_TOKEN_SPLITS.get(entry)
        if comp_seq is not None and comp_seq not in seen_seqs:
            seen_seqs.add(comp_seq)
            token_seqs.append(comp_seq)
    return frozenset(exact_tokens), tuple(token_seqs)


_SUSPICIOUS_EXACT_TOKENS, _SUSPICIOUS_TOKEN_SEQUENCES = _build_suspicious_token_tables()


def _ident_has_suspicious_suppress_token(ident: str) -> bool:
    """Returns True when snake_case/CamelCase tokens of ident match a suspicious suppressor stem."""
    words = tuple(tok.lower() for tok in _IDENT_WORD_TOKEN_RE.findall(ident))
    if not words:
        return False
    for word in words:
        if word in _SUSPICIOUS_EXACT_TOKENS or word.startswith(_SUSPICIOUS_PREFIX_STEMS):
            return True
    n_words = len(words)
    for seq in _SUSPICIOUS_TOKEN_SEQUENCES:
        seq_len = len(seq)
        if n_words < seq_len:
            continue
        seq_prefix = seq[:-1]
        seq_last = seq[-1]
        seq_last_plural = f"{seq_last}s"
        for idx in range(n_words - seq_len + 1):
            if words[idx : idx + seq_len - 1] == seq_prefix and words[
                idx + seq_len - 1
            ] in (seq_last, seq_last_plural):
                return True
    return False


@dataclass(frozen=True)
class _SuppressProvenance:
    """Structured provenance for context-manager suppression and stdlib non-suppression."""

    suppressors: frozenset[str] = frozenset()
    non_suppressors: frozenset[str] = frozenset()
    stdlib_modules: frozenset[Tuple[str, str]] = frozenset()
    shadowed_builtins: frozenset[str] = frozenset()
    has_star_import: bool = False
    imported_symbols: frozenset[Tuple[str, str]] = frozenset()
    shadowed_names: frozenset[str] = frozenset()
    non_suppressor_instances: frozenset[str] = frozenset()

    @property
    def suppress_names(self) -> frozenset[str]:
        """Returns suppressing context-manager names."""
        return self.suppressors

    @property
    def non_suppress_names(self) -> frozenset[str]:
        """Returns non-suppressing context-manager names."""
        return self.non_suppressors

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()

    def as_cache_key_tuple(self) -> Tuple[Any, ...]:
        """Returns a deterministic hashable tuple representation for cache keys."""
        return (
            tuple(sorted(self.suppressors)),
            tuple(sorted(self.non_suppressors)),
            tuple(sorted(self.stdlib_modules)),
            tuple(sorted(self.shadowed_builtins)),
            self.has_star_import,
            tuple(sorted(self.imported_symbols)),
            tuple(sorted(self.shadowed_names)),
            tuple(sorted(self.non_suppressor_instances)),
        )

    def __bool__(self) -> bool:
        return bool(
            self.suppressors
            or self.non_suppressors
            or self.stdlib_modules
            or self.shadowed_builtins
            or self.has_star_import
            or self.imported_symbols
            or self.shadowed_names
            or self.non_suppressor_instances
        )

    def clear_name(self, name: str, *, mark_shadow: bool = True) -> _SuppressProvenance:
        """Returns provenance with suppressor and non-suppressing bindings cleared for name."""
        if not name:
            return self
        new_sup = self.suppressors - {name}
        new_nonsup = self.non_suppressors - {name}
        new_nonsup_inst = self.non_suppressor_instances - {name}
        new_mods = frozenset(pair for pair in self.stdlib_modules if pair[0] != name)
        new_imps = frozenset(pair for pair in self.imported_symbols if pair[0] != name)
        new_shadow = (
            (self.shadowed_builtins | {name})
            if (mark_shadow and name in _SHADOWABLE_BUILTIN_CTX_NAMES)
            else self.shadowed_builtins
        )
        new_shadow_names = (
            (self.shadowed_names | {name}) if mark_shadow else self.shadowed_names
        )
        if (
            new_sup == self.suppressors
            and new_nonsup == self.non_suppressors
            and new_nonsup_inst == self.non_suppressor_instances
            and new_mods == self.stdlib_modules
            and new_imps == self.imported_symbols
            and new_shadow == self.shadowed_builtins
            and new_shadow_names == self.shadowed_names
        ):
            return self
        return _SuppressProvenance(
            suppressors=new_sup,
            non_suppressors=new_nonsup,
            stdlib_modules=new_mods,
            shadowed_builtins=new_shadow,
            has_star_import=self.has_star_import,
            imported_symbols=new_imps,
            shadowed_names=new_shadow_names,
            non_suppressor_instances=new_nonsup_inst,
        )

    def _bind_provenance_entry(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        is_suppressor: bool = False,
        is_non_suppressor: bool = False,
        is_non_suppressor_instance: bool = False,
        stdlib_mod: Optional[str] = None,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Clears prior bindings for name and optionally records a new provenance entry."""
        is_shadowable_builtin = name in _SHADOWABLE_BUILTIN_CTX_NAMES
        has_pos_binding = bool(
            allow_positive
            and (
                is_non_suppressor
                or is_non_suppressor_instance
                or stdlib_mod is not None
                or (imported_qualname is not None and not is_shadowable_builtin)
            )
        )
        cleared = self.clear_name(name, mark_shadow=not has_pos_binding)
        if not (has_pos_binding or is_suppressor) or not name:
            return cleared
        return _SuppressProvenance(
            suppressors=(
                cleared.suppressors | {name} if is_suppressor else cleared.suppressors
            ),
            non_suppressors=(
                cleared.non_suppressors | {name}
                if (allow_positive and is_non_suppressor)
                else cleared.non_suppressors
            ),
            stdlib_modules=(
                cleared.stdlib_modules | {(name, stdlib_mod)}
                if (allow_positive and stdlib_mod is not None)
                else cleared.stdlib_modules
            ),
            shadowed_builtins=(
                cleared.shadowed_builtins - {name}
                if has_pos_binding
                else cleared.shadowed_builtins
            ),
            has_star_import=cleared.has_star_import,
            imported_symbols=(
                cleared.imported_symbols | {(name, imported_qualname)}
                if (
                    allow_positive
                    and imported_qualname is not None
                    and (has_pos_binding or is_suppressor)
                )
                else cleared.imported_symbols
            ),
            shadowed_names=(
                cleared.shadowed_names - {name}
                if has_pos_binding
                else cleared.shadowed_names
            ),
            non_suppressor_instances=(
                cleared.non_suppressor_instances | {name}
                if (allow_positive and is_non_suppressor_instance)
                else cleared.non_suppressor_instances
            ),
        )

    def with_suppressor(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Returns provenance after binding name to a known suppressing context manager."""
        return self._bind_provenance_entry(
            name,
            allow_positive=allow_positive,
            is_suppressor=True,
            imported_qualname=imported_qualname,
        )

    def with_non_suppressor(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Returns provenance after binding name to a proven non-suppressing context manager factory."""
        return self._bind_provenance_entry(
            name,
            allow_positive=allow_positive,
            is_non_suppressor=True,
            imported_qualname=imported_qualname,
        )

    def with_non_suppressor_instance(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Returns provenance after binding name to a proven non-suppressing context manager instance."""
        return self._bind_provenance_entry(
            name,
            allow_positive=allow_positive,
            is_non_suppressor_instance=True,
            imported_qualname=imported_qualname,
        )

    def with_stdlib_module(
        self, alias: str, mod_name: str, *, allow_positive: bool = True
    ) -> _SuppressProvenance:
        """Returns provenance after binding alias to a trusted non-suppressing stdlib module."""
        return self._bind_provenance_entry(
            alias,
            allow_positive=allow_positive,
            stdlib_mod=mod_name,
            imported_qualname=mod_name,
        )

    def with_imported_symbol(
        self, alias: str, qualname: str, *, allow_positive: bool = True
    ) -> _SuppressProvenance:
        """Returns provenance after binding alias to an imported module or symbol qualname."""
        return self._bind_provenance_entry(
            alias,
            allow_positive=allow_positive,
            imported_qualname=qualname,
        )

    def with_shadowed_builtins(self, names: Iterable[str]) -> _SuppressProvenance:
        """Returns provenance with the given builtin names marked shadowed."""
        valid = frozenset(n for n in names if n in _SHADOWABLE_BUILTIN_CTX_NAMES)
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
            imported_symbols=frozenset(
                pair for pair in self.imported_symbols if pair[0] not in valid
            ),
            shadowed_names=self.shadowed_names | valid,
            non_suppressor_instances=self.non_suppressor_instances - valid,
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
            imported_symbols=self.imported_symbols,
            shadowed_names=self.shadowed_names,
            non_suppressor_instances=self.non_suppressor_instances,
        )

    def merge_branch_conservative(
        self, branch_prov: _SuppressProvenance
    ) -> _SuppressProvenance:
        """Merges provenance from a branch, keeping only surviving positive proofs and all hazards."""
        return _SuppressProvenance(
            suppressors=self.suppressors | branch_prov.suppressors,
            non_suppressors=self.non_suppressors & branch_prov.non_suppressors,
            stdlib_modules=self.stdlib_modules & branch_prov.stdlib_modules,
            shadowed_builtins=(self.shadowed_builtins | branch_prov.shadowed_builtins),
            has_star_import=(self.has_star_import or branch_prov.has_star_import),
            imported_symbols=(self.imported_symbols & branch_prov.imported_symbols),
            shadowed_names=self.shadowed_names | branch_prov.shadowed_names,
            non_suppressor_instances=(
                self.non_suppressor_instances & branch_prov.non_suppressor_instances
            ),
        )

    def apply_branch_delta(
        self,
        shadowed_names: AbstractSet[str],
        shadowed_builtins: AbstractSet[str],
        new_suppressors: AbstractSet[str],
        has_star: bool,
    ) -> _SuppressProvenance:
        """Applies a precomputed branch/suffix delta (shadowed names, suppressors, star import)."""
        if (
            not shadowed_names
            and not shadowed_builtins
            and not new_suppressors
            and not has_star
        ):
            return self
        return _SuppressProvenance(
            suppressors=self.suppressors | frozenset(new_suppressors),
            non_suppressors=self.non_suppressors - shadowed_names,
            stdlib_modules=frozenset(
                pair for pair in self.stdlib_modules if pair[0] not in shadowed_names
            ),
            shadowed_builtins=(self.shadowed_builtins | frozenset(shadowed_builtins)),
            has_star_import=self.has_star_import or has_star,
            imported_symbols=frozenset(
                pair for pair in self.imported_symbols if pair[0] not in shadowed_names
            ),
            shadowed_names=self.shadowed_names | frozenset(shadowed_names),
            non_suppressor_instances=self.non_suppressor_instances - shadowed_names,
        )


_EMPTY_SUPPRESS_PROVENANCE = _SuppressProvenance()


def _coerce_suppress_provenance(
    suppress_aliases: Optional[Union[_SuppressProvenance, Iterable[str]]],
) -> _SuppressProvenance:
    """Coerces an optional suppressor iterable or _SuppressProvenance into a _SuppressProvenance."""
    if suppress_aliases is None:
        return _EMPTY_SUPPRESS_PROVENANCE
    if isinstance(suppress_aliases, _SuppressProvenance):
        return suppress_aliases
    if isinstance(suppress_aliases, (str, bytes)):
        raise TypeError(
            "suppress_aliases must be a _SuppressProvenance or non-string iterable of names, not a bare str/bytes"
        )
    names = frozenset(s for s in suppress_aliases if isinstance(s, str) and s)
    if not names:
        return _EMPTY_SUPPRESS_PROVENANCE
    return _SuppressProvenance(suppressors=names)


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
    suppress_aliases: Union[Iterable[str], _SuppressProvenance],
    *,
    mark_shadow: bool = True,
) -> _SuppressProvenance:
    """Clears suppressor and non-suppressing stdlib provenance for a rebound name."""
    return _coerce_suppress_provenance(suppress_aliases).clear_name(
        name, mark_shadow=mark_shadow
    )


def _clear_store_target_provenance(
    target: ast.AST,
    prov: _SuppressProvenance,
    *,
    is_conditional: bool = False,
) -> _SuppressProvenance:
    """Clears provenance for store names in target, merging conservatively when conditional."""
    for t_name in _extract_store_names_in_node(target):
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


def _matches_extra_non_suppressing_symbol(
    base_id: str,
    tail_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> bool:
    """Returns True if base_id (with optional attribute tail_parts) matches extra_non_suppressing_ctx_managers.

    Note:
        Entries prefixed with ``*.`` (e.g. ``"*.atomic"``) match any attribute chain or
        imported symbol ending in that suffix. A leading single-dot entry ``".Name"``
        matches relative imports (``from . import Name`` or ``from .mod import Name``).
    """
    if (
        not extra_non_suppressing_ctx_managers
        or base_id in prov.suppressors
        or base_id in prov.shadowed_builtins
        or base_id in prov.shadowed_names
        or prov.has_star_import
    ):
        return False
    candidate = ".".join((base_id, *tail_parts))
    if candidate in extra_non_suppressing_ctx_managers:
        return True
    for alias, qual in prov.imported_symbols:
        if alias == base_id:
            resolved = ".".join((qual, *tail_parts))
            stripped = resolved.lstrip(".")
            last_part = resolved.rsplit(".", 1)[-1]
            if (
                resolved in extra_non_suppressing_ctx_managers
                or (stripped and stripped in extra_non_suppressing_ctx_managers)
                or candidate in extra_non_suppressing_ctx_managers
                or (
                    resolved.startswith(".")
                    and last_part
                    and last_part not in _SUPPRESSING_CTX_MANAGER_NAMES
                    and f".{last_part}" in extra_non_suppressing_ctx_managers
                )
                or (
                    "." in resolved
                    and last_part
                    and last_part not in _SUPPRESSING_CTX_MANAGER_NAMES
                    and f"*.{last_part}" in extra_non_suppressing_ctx_managers
                )
            ):
                return True
    return False


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
    """Returns True when fn_id is not a suppressor and belongs to allowed_names or extra symbols."""
    if fn_id in prov.suppressors:
        return False
    return fn_id in allowed_names or _matches_extra_non_suppressing_symbol(
        fn_id, (), prov, extra_non_suppressing_ctx_managers
    )


def _attribute_matches_extra_non_suppressing(
    curr_val: ast.AST,
    attr_name: str,
    chain_parts: Sequence[str],
    prov: _SuppressProvenance,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> bool:
    """Returns True when an Attribute chain matches extra_non_suppressing_ctx_managers."""
    if not extra_non_suppressing_ctx_managers or attr_name in _SUPPRESSING_CTX_MANAGER_NAMES:
        return False
    if (
        not (isinstance(curr_val, ast.Name) and curr_val.id in prov.suppressors)
        and f"*.{attr_name}" in extra_non_suppressing_ctx_managers
    ):
        return True
    return isinstance(curr_val, ast.Name) and _matches_extra_non_suppressing_symbol(
        curr_val.id, chain_parts, prov, extra_non_suppressing_ctx_managers
    )


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
        if isinstance(curr_val, ast.Name) and len(chain_parts) >= 2:
            for alias, qual in prov.imported_symbols:
                if alias == curr_val.id:
                    mod_candidate = ".".join((qual, *chain_parts[:-1]))
                    allowed_sub_attrs = _STDLIB_NON_SUPPRESSING_MODULES.get(
                        mod_candidate, frozenset()
                    )
                    if attr_name in allowed_sub_attrs:
                        return True
        if isinstance(expr.value, ast.Name):
            base_id = expr.value.id
            for alias, mod_name in prov.stdlib_modules:
                if alias == base_id:
                    allowed_attrs = _STDLIB_NON_SUPPRESSING_MODULES.get(
                        mod_name, frozenset()
                    )
                    if attr_name in allowed_attrs:
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
        ):
            return True
        for alias, qual in prov.imported_symbols:
            if alias == fn_id and qual in (
                "builtins.open",
                "__builtins__.open",
                "io.open",
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
        ):
            return True
        for alias, mod_name in prov.stdlib_modules:
            if alias == base_id and mod_name in ("builtins", "__builtins__", "io"):
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


def _iter_shallow_scope_body_nodes(scope_node: ast.AST) -> Iterator[ast.AST]:
    """Yields AST nodes in scope_node.body without descending into nested _INNER_SCOPE_TYPES bodies."""
    body_attr = getattr(scope_node, "body", ())
    if not isinstance(body_attr, (list, tuple)):
        return
    yield from _iter_shallow_scope_nodes(body_attr)


_DYNAMIC_CLASS_NAMESPACE_CALLS: frozenset[str] = frozenset(
    {"locals", "vars", "exec", "eval", "setattr", "delattr", "globals"}
)
_DYNAMIC_METHOD_MUTATION_CALLS: frozenset[str] = frozenset(
    {"exec", "eval", "setattr", "delattr"}
)
_DYNAMIC_FRAME_ATTR_CALLS: frozenset[str] = frozenset({"_getframe", "currentframe"})
_SAFE_METHOD_DECORATORS: frozenset[str] = frozenset(
    {"staticmethod", "classmethod", "property"}
)
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
    if node is None or isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert, ast.Not)
    ):
        return _is_static_class_attr_value(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_static_class_attr_value(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or _is_static_class_attr_value(k))
            and _is_static_class_attr_value(v)
            for k, v in zip(node.keys, node.values)
        )
    return False


def _annotation_is_call_free(node: Optional[ast.AST]) -> bool:
    """Returns True if a type annotation expression contains no calls, lambdas, walrus, or comprehensions."""
    return node is None or not any(
        isinstance(sub, _DISALLOWED_ANNOTATION_NODE_TYPES) for sub in ast.walk(node)
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
        if stmt.name in ("__exit__", "__aexit__"):
            return not stmt.decorator_list
        return all(
            isinstance(dec, ast.Name)
            and dec.id in _SAFE_METHOD_DECORATORS
            and dec.id not in prov.shadowed_names
            and dec.id not in prov.shadowed_builtins
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
    """Returns True if a ClassDef contains dynamic namespace reflection or runtime __exit__/__aexit__ mutation."""
    if any(
        _is_dynamic_namespace_call(sub, _DYNAMIC_CLASS_NAMESPACE_CALLS)
        for sub in _iter_shallow_scope_body_nodes(cls_node)
    ):
        return True
    return any(
        _is_dynamic_namespace_call(node, _DYNAMIC_METHOD_MUTATION_CALLS)
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr in ("__exit__", "__aexit__", "__class__", "__dict__")
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
            (alias.asname or alias.name.split(".", 1)[0]) in ("__exit__", "__aexit__")
            for alias in sub.names
        ):
            return True
        elif isinstance(sub, ast.ClassDef) and sub.name in (
            "__exit__",
            "__aexit__",
        ):
            return True
        elif isinstance(sub, ast.ExceptHandler) and sub.name in (
            "__exit__",
            "__aexit__",
        ):
            return True
        elif _extract_pattern_bound_name(sub) in ("__exit__", "__aexit__"):
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
        return (
            base.id in _SAFE_EXIT_BASE_CLASS_NAMES
            and base.id not in prov.shadowed_names
            and base.id not in prov.shadowed_builtins
        )
    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
        return (
            base.value.id in ("contextlib", "typing")
            and base.attr in _SAFE_EXIT_BASE_CLASS_NAMES
            and base.value.id not in prov.shadowed_names
        )
    return False


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


def _func_def_may_return_suppressor(
    fn_stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    prov: _SuppressProvenance,
) -> bool:
    """Returns True if a function definition may return a suppressor or suppress exceptions around yield."""
    local_prov = prov
    for p_name in _extract_arg_names(fn_stmt.args):
        local_prov = local_prov.clear_name(p_name)
    for tp_name in _extract_node_type_param_names(fn_stmt):
        local_prov = local_prov.clear_name(tp_name)
    body_prov = _update_suppress_provenance_from_stmts(
        fn_stmt.body, local_prov, in_branch=True
    )
    fn_prov = local_prov.merge_branch_conservative(body_prov)

    stack: List[Tuple[ast.AST, bool]] = [
        (stmt, False) for stmt in reversed(fn_stmt.body)
    ]
    while stack:
        curr, in_try_or_with = stack.pop()
        if isinstance(curr, _INNER_SCOPE_TYPES):
            for hdr in reversed(tuple(iter_nested_scope_header_nodes(curr))):
                stack.append((hdr, in_try_or_with))
            continue
        if isinstance(curr, (ast.Yield, ast.YieldFrom)) and in_try_or_with:
            return True
        if isinstance(curr, ast.Return) and curr.value is not None:
            val_stack: List[ast.AST] = [curr.value]
            while val_stack:
                v_curr = val_stack.pop()
                if _is_suppress_alias_target_expr(v_curr, fn_prov):
                    return True
                if isinstance(v_curr, _INNER_SCOPE_TYPES):
                    val_stack.extend(iter_nested_scope_header_nodes(v_curr))
                    continue
                val_stack.extend(ast.iter_child_nodes(v_curr))
        next_in_try_or_with = (
            in_try_or_with
            or isinstance(curr, (ast.With, ast.AsyncWith))
            or (
                is_try_or_trystar_node(curr)
                and bool(
                    getattr(curr, "handlers", None) or getattr(curr, "finalbody", None)
                )
            )
        )
        for ch in reversed(tuple(ast.iter_child_nodes(curr))):
            stack.append((ch, next_in_try_or_with))
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
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance from NamedExpr (walrus) bindings inside expression trees in left-to-right order."""
    w_stack: List[Tuple[ast.AST, bool, bool]] = [
        (root, allow_positive, False) for root in reversed(tuple(expr_roots))
    ]
    while w_stack:
        w_curr, cur_allow_pos, is_post_bind = w_stack.pop()
        if is_post_bind:
            if isinstance(w_curr, ast.NamedExpr) and isinstance(
                w_curr.target, ast.Name
            ):
                bound_prov = _bind_expr_target_provenance(
                    prov,
                    w_curr.target.id,
                    w_curr.value,
                    allow_positive=cur_allow_pos,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                prov = (
                    bound_prov
                    if cur_allow_pos
                    else prov.merge_branch_conservative(bound_prov)
                )
            continue
        if isinstance(w_curr, _COMPREHENSION_TYPES):
            if isinstance(w_curr, ast.DictComp):
                w_stack.append((w_curr.value, False, False))
                w_stack.append((w_curr.key, False, False))
            else:
                w_stack.append((w_curr.elt, False, False))
            if w_curr.generators:
                for gen in reversed(w_curr.generators):
                    for if_e in reversed(gen.ifs):
                        w_stack.append((if_e, False, False))
                for gen in reversed(w_curr.generators[1:]):
                    w_stack.append((gen.iter, False, False))
                w_stack.append((w_curr.generators[0].iter, cur_allow_pos, False))
            continue
        if isinstance(w_curr, _INNER_SCOPE_TYPES):
            for hdr in reversed(tuple(iter_nested_scope_header_nodes(w_curr))):
                w_stack.append((hdr, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.NamedExpr):
            if isinstance(w_curr.target, ast.Name):
                w_stack.append((w_curr, cur_allow_pos, True))
            w_stack.append((w_curr.value, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.BoolOp):
            for idx in range(len(w_curr.values) - 1, -1, -1):
                w_stack.append((w_curr.values[idx], cur_allow_pos and idx == 0, False))
            continue
        if isinstance(w_curr, ast.Compare):
            for idx in range(len(w_curr.comparators) - 1, -1, -1):
                w_stack.append(
                    (
                        w_curr.comparators[idx],
                        cur_allow_pos and idx == 0,
                        False,
                    )
                )
            w_stack.append((w_curr.left, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.IfExp):
            w_stack.append((w_curr.orelse, False, False))
            w_stack.append((w_curr.body, False, False))
            w_stack.append((w_curr.test, cur_allow_pos, False))
            continue
        if isinstance(w_curr, ast.Dict):
            for k_node, v_node in reversed(list(zip(w_curr.keys, w_curr.values))):
                if v_node is not None:
                    w_stack.append((v_node, cur_allow_pos, False))
                if k_node is not None:
                    w_stack.append((k_node, cur_allow_pos, False))
            continue
        for ch in reversed(tuple(ast.iter_child_nodes(w_curr))):
            if not isinstance(ch, ast.stmt):
                w_stack.append((ch, cur_allow_pos, False))
    return prov


def _resolve_suppress_or_null_qualname(
    expr: Optional[ast.AST],
    prov: _SuppressProvenance,
) -> Optional[str]:
    """Returns 'contextlib.suppress' or 'contextlib.nullcontext' if expr is an unshadowed reference."""
    if isinstance(expr, ast.Name):
        fn_id = expr.id
        if fn_id not in prov.shadowed_names and not prov.has_star_import:
            if fn_id == "suppress":
                return "contextlib.suppress"
            if fn_id == "nullcontext" and fn_id in prov.non_suppressors:
                return "contextlib.nullcontext"
            for alias, qual in prov.imported_symbols:
                if alias == fn_id and qual in (
                    "contextlib.suppress",
                    "contextlib.nullcontext",
                ):
                    return qual
    elif (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "contextlib"
        and "contextlib" not in prov.shadowed_names
        and not prov.has_star_import
        and expr.attr in ("suppress", "nullcontext")
    ):
        return f"contextlib.{expr.attr}"
    return None


def _bind_expr_target_provenance(
    prov: _SuppressProvenance,
    name: str,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance when name is directly bound to value_expr."""
    while isinstance(value_expr, ast.NamedExpr):
        value_expr = value_expr.value
    target_qual = (
        _resolve_suppress_or_null_qualname(value_expr, prov) if allow_positive else None
    )
    if _is_suppress_alias_target_expr(value_expr, prov) or (
        isinstance(value_expr, ast.Call)
        and _is_suppress_alias_target_expr(value_expr.func, prov)
    ):
        return prov.with_suppressor(
            name, allow_positive=allow_positive, imported_qualname=target_qual
        )
    if value_expr is not None:
        if _is_proven_non_suppressing_call(
            value_expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ) or (
            isinstance(value_expr, ast.Name)
            and value_expr.id in prov.non_suppressor_instances
            and value_expr.id not in prov.suppressors
        ):
            return prov.with_non_suppressor_instance(
                name, allow_positive=allow_positive, imported_qualname=target_qual
            )
        if _is_proven_non_suppressing_target_expr(
            value_expr,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            open_qual = (
                "builtins.open"
                if (
                    allow_positive
                    and target_qual is None
                    and _is_builtin_or_io_open_target(value_expr, prov)
                )
                else target_qual
            )
            return prov.with_non_suppressor(
                name, allow_positive=allow_positive, imported_qualname=open_qual
            )
        if allow_positive and _is_pathlib_path_expr(value_expr, prov):
            return prov.with_imported_symbol(name, "pathlib.Path", allow_positive=True)
    return prov.clear_name(name)


def _update_suppress_provenance_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Applies _update_suppress_provenance_from_stmt across a statement sequence."""
    for sub_stmt in stmts:
        prov = _update_suppress_provenance_from_stmt(
            sub_stmt,
            prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    return prov


def _update_clause_branch_provenance(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    body: Iterable[ast.AST],
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance for an except handler or match case branch."""
    if isinstance(header_expr, ast.AST):
        prov = _update_suppress_provenance_from_expr_walrus(
            (header_expr,), prov, allow_positive=False
        )
    for b_name in bound_names:
        prov = prov.clear_name(b_name)
    return _update_suppress_provenance_from_stmts(
        body,
        prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _update_suppress_provenance_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Returns updated _SuppressProvenance from a statement, recursing into compound blocks."""
    allow_pos = not in_branch
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            if bound == "*":
                prov = prov.with_star_import()
                continue
            target_qual = (
                alias.name if alias.asname is not None else alias.name.split(".", 1)[0]
            )
            if alias.name in _STDLIB_NON_SUPPRESSING_MODULES and (
                alias.asname is not None or "." not in alias.name
            ):
                prov = prov.with_stdlib_module(
                    bound, alias.name, allow_positive=allow_pos
                )
            elif allow_pos and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES:
                prov = prov.with_imported_symbol(
                    bound, target_qual, allow_positive=allow_pos
                )
            else:
                prov = prov.clear_name(bound)
        return prov

    if isinstance(stmt, ast.ImportFrom):
        level = int(getattr(stmt, "level", 0) or 0)
        mod_name = stmt.module or ""
        is_abs_mod = level == 0 and bool(mod_name)
        rel_prefix = "." * level if level > 0 else ""
        allowed_nonsup = (
            _STDLIB_NON_SUPPRESSING_MODULES.get(mod_name, frozenset())
            if is_abs_mod
            else frozenset()
        )
        for alias in stmt.names:
            if alias.name == "*":
                prov = prov.with_star_import()
                continue
            bound = alias.asname or alias.name
            if is_abs_mod:
                imp_qual: Optional[str] = f"{mod_name}.{alias.name}"
            elif level > 0:
                imp_qual = (
                    f"{rel_prefix}{mod_name}.{alias.name}"
                    if mod_name
                    else f"{rel_prefix}{alias.name}"
                )
            else:
                imp_qual = None
            if alias.name in allowed_nonsup:
                prov = prov.with_non_suppressor(
                    bound,
                    allow_positive=allow_pos,
                    imported_qualname=imp_qual,
                )
            elif alias.name in _SUPPRESSING_CTX_MANAGER_NAMES:
                prov = prov.with_suppressor(
                    bound,
                    allow_positive=allow_pos,
                    imported_qualname=imp_qual,
                )
            elif (
                imp_qual is not None
                and allow_pos
                and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES
            ):
                prov = prov.with_imported_symbol(
                    bound,
                    imp_qual,
                    allow_positive=allow_pos,
                )
            else:
                prov = prov.clear_name(bound)
        return prov

    if isinstance(stmt, ast.Assign):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.value, *stmt.targets),
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        for tgt in stmt.targets:
            for t_name in _extract_store_names_in_node(tgt):
                prov = (
                    _bind_expr_target_provenance(
                        prov,
                        t_name,
                        stmt.value,
                        allow_positive=allow_pos,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
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
            exprs,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if stmt.value is not None:
            for t_name in _extract_store_names_in_node(stmt.target):
                prov = (
                    _bind_expr_target_provenance(
                        prov,
                        t_name,
                        stmt.value,
                        allow_positive=allow_pos and isinstance(stmt, ast.AnnAssign),
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                    )
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    else prov.clear_name(t_name)
                )
        return prov

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prov = _update_suppress_provenance_from_expr_walrus(
            iter_nested_scope_header_nodes(stmt),
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if _func_def_may_return_suppressor(stmt, prov):
            return prov.with_suppressor(stmt.name)
        return prov.clear_name(stmt.name)

    if isinstance(stmt, ast.ClassDef):
        prov = _update_suppress_provenance_from_expr_walrus(
            iter_nested_scope_header_nodes(stmt),
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if _class_def_may_suppress_in_exit(stmt, prov):
            return prov.with_suppressor(stmt.name)
        if (
            _class_def_is_proven_non_suppressing_in_exit(stmt, prov)
            and stmt.name not in _SUPPRESSING_CTX_MANAGER_NAMES
        ):
            return prov.with_non_suppressor(
                stmt.name,
                allow_positive=allow_pos,
                imported_qualname=stmt.name if allow_pos else None,
            )
        if allow_pos and stmt.name not in _SHADOWABLE_BUILTIN_CTX_NAMES:
            return prov.with_imported_symbol(stmt.name, stmt.name, allow_positive=True)
        return prov.clear_name(stmt.name)

    if isinstance(stmt, ast.Delete):
        prov = _update_suppress_provenance_from_expr_walrus(
            stmt.targets,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        for d_name in _extract_shallow_deleted_names(stmt):
            prov = prov.clear_name(d_name)
        return prov

    if is_type_alias_node(stmt):
        ta_val = getattr(stmt, "value", None)
        if isinstance(ta_val, ast.AST):
            prov = _update_suppress_provenance_from_expr_walrus(
                (ta_val,),
                prov,
                allow_positive=allow_pos,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        ta_name = _extract_type_alias_or_pattern_name(stmt)
        if ta_name is not None:
            prov = prov.clear_name(ta_name)
        return prov

    if isinstance(stmt, (ast.If, ast.While)):
        is_while = isinstance(stmt, ast.While)
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.test,),
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        branch_flag = True if is_while else in_branch
        b_prov = _update_suppress_provenance_from_stmts(
            stmt.body,
            prov,
            in_branch=branch_flag,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        else_base = b_prov if is_while else prov
        o_prov = (
            _update_suppress_provenance_from_stmts(
                stmt.orelse,
                else_base,
                in_branch=branch_flag,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if stmt.orelse
            else else_base
        )
        merged = b_prov.merge_branch_conservative(o_prov)
        return prov.merge_branch_conservative(merged) if is_while else merged

    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.iter,),
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        iter_prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.target,),
            prov,
            allow_positive=False,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        iter_is_suspicious = _looks_like_suppressing_ctx_manager(stmt.iter, prov)
        for t_name in _extract_store_names_in_node(stmt.target):
            if iter_is_suspicious and isinstance(stmt.target, ast.Name):
                iter_prov = iter_prov.with_suppressor(t_name, allow_positive=True)
            else:
                iter_prov = iter_prov.clear_name(t_name)
        b_prov = _update_suppress_provenance_from_stmts(
            stmt.body,
            iter_prov,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        o_base = prov.merge_branch_conservative(b_prov)
        o_prov = (
            _update_suppress_provenance_from_stmts(
                stmt.orelse,
                o_base,
                in_branch=True,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if stmt.orelse
            else b_prov
        )
        return o_base.merge_branch_conservative(o_prov)

    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        header_prov = prov
        item_prov = prov
        after_sup = False
        header_risky = False
        for step in _iter_with_item_provenance_steps(
            stmt,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        ):
            if step.after_prior_sup or step.opt_vars_conditional:
                header_risky = True
            if step.item_may_suppress:
                after_sup = True
            item_prov = step.post_item_prov
        if (
            not header_risky
            and not after_sup
            and _classify_with_suppression(
                stmt,
                header_prov,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            == "non_suppressing"
        ):
            return _update_suppress_provenance_from_stmts(
                stmt.body,
                item_prov,
                in_branch=in_branch,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        safe_prefix, risky_suffix = (
            ([], list(stmt.body))
            if header_risky
            else _split_suppress_with_body(stmt.body, suppress_aliases=item_prov)
        )
        prefix_prov = _update_suppress_provenance_from_stmts(
            safe_prefix,
            item_prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if not risky_suffix:
            return prefix_prov
        # Statements in risky_suffix execute after a potentially raising expression inside a
        # potentially exception-suppressing with-block; if an exception is raised and suppressed,
        # the remainder of risky_suffix is skipped, so bindings in risky_suffix are conditional
        # (in_branch=True) relative to code following the with-block.
        suffix_prov = _update_suppress_provenance_from_stmts(
            risky_suffix,
            prefix_prov,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        return prefix_prov.merge_branch_conservative(suffix_prov)

    if is_try_or_trystar_node(stmt):
        t_body = getattr(stmt, "body", ())
        t_safe_prefix, t_risky_suffix = _split_suppress_with_body(
            t_body, suppress_aliases=prov
        )
        prefix_prov = _update_suppress_provenance_from_stmts(
            t_safe_prefix,
            prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        try_mid_prov = _update_suppress_provenance_from_stmts(
            t_risky_suffix,
            prefix_prov,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        try_prov = _update_suppress_provenance_from_stmts(
            t_risky_suffix,
            prefix_prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        handler_base = prefix_prov.merge_branch_conservative(
            try_mid_prov
        ).merge_branch_conservative(try_prov)
        o_stmts = getattr(stmt, "orelse", ())
        normal_prov = (
            _update_suppress_provenance_from_stmts(
                o_stmts,
                try_prov,
                in_branch=in_branch,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if o_stmts
            else try_prov
        )
        merged_try = normal_prov
        for h_node in getattr(stmt, "handlers", ()):
            h_name = getattr(h_node, "name", None)
            h_prov = _update_clause_branch_provenance(
                handler_base,
                getattr(h_node, "type", None),
                (h_name,) if isinstance(h_name, str) else (),
                getattr(h_node, "body", ()),
                in_branch=in_branch,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            merged_try = merged_try.merge_branch_conservative(h_prov)
        f_body = getattr(stmt, "finalbody", ())
        if f_body:
            return _update_suppress_provenance_from_stmts(
                f_body,
                merged_try,
                in_branch=in_branch,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        return merged_try

    if is_match_node(stmt):
        subject = getattr(stmt, "subject", None)
        if isinstance(subject, ast.AST):
            prov = _update_suppress_provenance_from_expr_walrus(
                (subject,), prov, allow_positive=allow_pos
            )
        cases = getattr(stmt, "cases", ())
        if not cases:
            return prov
        case_provs: List[_SuppressProvenance] = []
        has_irrefutable = False
        for case in cases:
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
            c_prov = _update_clause_branch_provenance(
                prov,
                getattr(case, "guard", None),
                pat_names,
                getattr(case, "body", ()),
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            case_provs.append(c_prov)
            if _is_irrefutable_case(case):
                has_irrefutable = True
                break
        merged_match = case_provs[0]
        for cp in case_provs[1:]:
            merged_match = merged_match.merge_branch_conservative(cp)
        if not has_irrefutable:
            merged_match = prov.merge_branch_conservative(merged_match)
        return merged_match

    if not isinstance(stmt, _INNER_SCOPE_TYPES):
        child_exprs = [
            ch for ch in ast.iter_child_nodes(stmt) if not isinstance(ch, ast.stmt)
        ]
        if child_exprs:
            prov = _update_suppress_provenance_from_expr_walrus(
                child_exprs, prov, allow_positive=allow_pos
            )
    return prov


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


def _extract_stmt_referenced_and_bound_names(stmt: ast.AST) -> Set[str]:
    """Extracts identifier strings referenced or bound anywhere inside stmt."""
    names: Set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            if node.asname:
                names.add(node.asname)
            if node.name:
                names.add(node.name.split(".", 1)[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        else:
            node_name = getattr(node, "name", None)
            if isinstance(node_name, str) and node_name:
                names.add(node_name)
    return names


class _ModuleStmtDelta(NamedTuple):
    """Per-statement forward and branch provenance delta for linear-time module seeding."""

    stmt_id: int
    fwd_add_sup: frozenset[str]
    fwd_rem_sup: frozenset[str]
    fwd_add_nonsup: frozenset[str]
    fwd_rem_nonsup: frozenset[str]
    fwd_add_nonsup_inst: frozenset[str]
    fwd_rem_nonsup_inst: frozenset[str]
    fwd_add_mods: frozenset[Tuple[str, str]]
    fwd_rem_mods: frozenset[Tuple[str, str]]
    fwd_add_imps: frozenset[Tuple[str, str]]
    fwd_rem_imps: frozenset[Tuple[str, str]]
    fwd_add_shad_b: frozenset[str]
    fwd_rem_shad_b: frozenset[str]
    fwd_add_shad_n: frozenset[str]
    fwd_rem_shad_n: frozenset[str]
    fwd_star: bool
    branch_new_sup: frozenset[str]
    branch_shad_b: frozenset[str]
    branch_shad_n: frozenset[str]
    branch_star: bool


class _LazyModuleStmtSeedMap(Mapping[int, _SuppressProvenance]):
    """On-demand mapping from top-level statement id() to its prefix, suffix, or combined provenance."""

    __slots__ = (
        "_deltas",
        "_idx_by_id",
        "_kind",
        "_cache",
        "_prefix_table",
        "_suffix_table",
    )

    def __init__(
        self,
        deltas: Tuple[_ModuleStmtDelta, ...],
        idx_by_id: Mapping[int, int],
        kind: str,
    ) -> None:
        self._deltas = deltas
        self._idx_by_id = idx_by_id
        self._kind = kind
        self._cache: dict[int, _SuppressProvenance] = {}
        self._prefix_table: Optional[Tuple[_SuppressProvenance, ...]] = None
        self._suffix_table: Optional[
            Tuple[Tuple[frozenset[str], frozenset[str], frozenset[str], bool], ...]
        ] = None

    def __len__(self) -> int:
        return len(self._idx_by_id)

    def __iter__(self) -> Iterator[int]:
        return iter(self._idx_by_id)

    def __contains__(self, key: object) -> bool:
        return key in self._idx_by_id

    def _ensure_tables(self) -> None:
        deltas = self._deltas
        n = len(deltas)
        if self._kind in ("prefix", "combined") and self._prefix_table is None:
            p_sup: Set[str] = set()
            p_nonsup: Set[str] = set()
            p_nonsup_inst: Set[str] = set()
            p_mods: Set[Tuple[str, str]] = set()
            p_imps: Set[Tuple[str, str]] = set()
            p_shad_b: Set[str] = set()
            p_shad_n: Set[str] = set()
            p_star = False
            pref_list: List[_SuppressProvenance] = []
            for idx in range(n):
                pref_list.append(
                    _SuppressProvenance(
                        frozenset(p_sup),
                        frozenset(p_nonsup),
                        frozenset(p_mods),
                        frozenset(p_shad_b),
                        p_star,
                        frozenset(p_imps),
                        frozenset(p_shad_n),
                        frozenset(p_nonsup_inst),
                    )
                )
                d = deltas[idx]
                if d.fwd_rem_sup:
                    p_sup.difference_update(d.fwd_rem_sup)
                if d.fwd_add_sup:
                    p_sup.update(d.fwd_add_sup)
                if d.fwd_rem_nonsup:
                    p_nonsup.difference_update(d.fwd_rem_nonsup)
                if d.fwd_add_nonsup:
                    p_nonsup.update(d.fwd_add_nonsup)
                if d.fwd_rem_nonsup_inst:
                    p_nonsup_inst.difference_update(d.fwd_rem_nonsup_inst)
                if d.fwd_add_nonsup_inst:
                    p_nonsup_inst.update(d.fwd_add_nonsup_inst)
                if d.fwd_rem_mods:
                    p_mods.difference_update(d.fwd_rem_mods)
                if d.fwd_add_mods:
                    p_mods.update(d.fwd_add_mods)
                if d.fwd_rem_imps:
                    p_imps.difference_update(d.fwd_rem_imps)
                if d.fwd_add_imps:
                    p_imps.update(d.fwd_add_imps)
                if d.fwd_rem_shad_b:
                    p_shad_b.difference_update(d.fwd_rem_shad_b)
                if d.fwd_add_shad_b:
                    p_shad_b.update(d.fwd_add_shad_b)
                if d.fwd_rem_shad_n:
                    p_shad_n.difference_update(d.fwd_rem_shad_n)
                if d.fwd_add_shad_n:
                    p_shad_n.update(d.fwd_add_shad_n)
                if d.fwd_star:
                    p_star = True
            self._prefix_table = tuple(pref_list)

        if self._kind in ("suffix", "combined") and self._suffix_table is None:
            s_sup: Set[str] = set()
            s_shad_b: Set[str] = set()
            s_shad_n: Set[str] = set()
            s_star = False
            suff_list: List[
                Tuple[frozenset[str], frozenset[str], frozenset[str], bool]
            ] = [(frozenset(), frozenset(), frozenset(), False)] * n
            for idx in range(n - 1, -1, -1):
                suff_list[idx] = (
                    frozenset(s_sup),
                    frozenset(s_shad_b),
                    frozenset(s_shad_n),
                    s_star,
                )
                d = deltas[idx]
                if d.branch_new_sup:
                    s_sup.update(d.branch_new_sup)
                if d.branch_shad_b:
                    s_shad_b.update(d.branch_shad_b)
                if d.branch_shad_n:
                    s_shad_n.update(d.branch_shad_n)
                if d.branch_star:
                    s_star = True
            self._suffix_table = tuple(suff_list)

    def __getitem__(self, stmt_id: int) -> _SuppressProvenance:
        cached = self._cache.get(stmt_id)
        if cached is not None:
            return cached
        target_idx = self._idx_by_id[stmt_id]
        self._ensure_tables()
        if self._kind == "prefix":
            assert self._prefix_table is not None
            res = self._prefix_table[target_idx]
            self._cache[stmt_id] = res
            return res

        assert self._suffix_table is not None
        s_sup, s_shad_b, s_shad_n, s_star = self._suffix_table[target_idx]
        if self._kind == "suffix":
            res = _SuppressProvenance(
                suppressors=s_sup,
                shadowed_builtins=s_shad_b,
                has_star_import=s_star,
                shadowed_names=s_shad_n,
            )
            self._cache[stmt_id] = res
            return res

        assert self._prefix_table is not None
        pref = self._prefix_table[target_idx]
        if not s_sup and not s_shad_b and not s_shad_n and not s_star:
            self._cache[stmt_id] = pref
            return pref
        p_nonsup = pref.non_suppressors
        p_nonsup_inst = pref.non_suppressor_instances
        p_mods = pref.stdlib_modules
        p_imps = pref.imported_symbols
        p_shad_n = pref.shadowed_names
        if s_shad_n:
            p_nonsup = p_nonsup - s_shad_n
            p_nonsup_inst = p_nonsup_inst - s_shad_n
            if p_mods:
                p_mods = frozenset(pair for pair in p_mods if pair[0] not in s_shad_n)
            if p_imps:
                p_imps = frozenset(pair for pair in p_imps if pair[0] not in s_shad_n)
            p_shad_n = p_shad_n | s_shad_n
        res = _SuppressProvenance(
            suppressors=pref.suppressors | s_sup if s_sup else pref.suppressors,
            non_suppressors=p_nonsup,
            stdlib_modules=p_mods,
            shadowed_builtins=(
                pref.shadowed_builtins | s_shad_b
                if s_shad_b
                else pref.shadowed_builtins
            ),
            has_star_import=pref.has_star_import or s_star,
            imported_symbols=p_imps,
            shadowed_names=p_shad_n,
            non_suppressor_instances=p_nonsup_inst,
        )
        self._cache[stmt_id] = res
        return res


_EMPTY_STMT_SEED_MAP: Mapping[int, _SuppressProvenance] = MappingProxyType({})


@dataclass(frozen=True)
class _ModuleSuppressSeed:
    """Cached per-module top-level suppression provenance summary."""

    full_forward_prov: _SuppressProvenance
    by_top_stmt_id: Mapping[int, _SuppressProvenance]
    has_any_positive_or_suppressor: bool
    prefix_by_top_stmt_id: Mapping[int, _SuppressProvenance] = field(
        default_factory=lambda: _EMPTY_STMT_SEED_MAP
    )
    suffix_delta_by_top_stmt_id: Mapping[int, _SuppressProvenance] = field(
        default_factory=lambda: _EMPTY_STMT_SEED_MAP
    )

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()


def _build_module_suppress_seed(eff_mod: ast.Module) -> _ModuleSuppressSeed:
    """Builds a reusable module-level suppression provenance summary in O(N) time."""
    body = eff_mod.body
    deltas: List[_ModuleStmtDelta] = []
    idx_by_id: dict[int, int] = {}
    cur_sup: Set[str] = set()
    cur_all_sup: Set[str] = set()
    cur_nonsup: Set[str] = set()
    cur_nonsup_inst: Set[str] = set()
    cur_mods: dict[str, str] = {}
    cur_imps: dict[str, str] = {}
    cur_shad_b: Set[str] = set()
    cur_shad_n: Set[str] = set()
    cur_star = False
    has_pos_or_sup = False

    for idx, m_stmt in enumerate(body):
        stmt_id = id(m_stmt)
        idx_by_id[stmt_id] = idx
        stmt_names = _extract_stmt_referenced_and_bound_names(m_stmt)
        local_all_sup = frozenset(n for n in stmt_names if n in cur_all_sup)
        seed_in = (
            _SuppressProvenance(suppressors=local_all_sup)
            if local_all_sup
            else _EMPTY_SUPPRESS_PROVENANCE
        )
        b_delta = _update_suppress_provenance_from_stmt(m_stmt, seed_in, in_branch=True)
        rebound_existing = local_all_sup & b_delta.shadowed_names
        if rebound_existing:
            b_without_rebound = _update_suppress_provenance_from_stmt(
                m_stmt,
                _SuppressProvenance(suppressors=local_all_sup - rebound_existing),
                in_branch=True,
            )
            new_sup = (b_delta.suppressors - local_all_sup) | (
                b_without_rebound.suppressors & rebound_existing
            )
        else:
            new_sup = b_delta.suppressors - local_all_sup
        if new_sup:
            cur_all_sup.update(new_sup)

        local_sup = frozenset(n for n in stmt_names if n in cur_sup)
        local_nonsup = frozenset(n for n in stmt_names if n in cur_nonsup)
        local_nonsup_inst = frozenset(n for n in stmt_names if n in cur_nonsup_inst)
        local_mods = frozenset((n, cur_mods[n]) for n in stmt_names if n in cur_mods)
        local_imps = frozenset((n, cur_imps[n]) for n in stmt_names if n in cur_imps)
        local_shad_b = frozenset(cur_shad_b)
        local_shad_n = frozenset(
            n for n in (stmt_names | _TRACKED_SHADOW_NAMES) if n in cur_shad_n
        )
        local_cur = _SuppressProvenance(
            suppressors=local_sup,
            non_suppressors=local_nonsup,
            stdlib_modules=local_mods,
            shadowed_builtins=local_shad_b,
            has_star_import=cur_star,
            imported_symbols=local_imps,
            shadowed_names=local_shad_n,
            non_suppressor_instances=local_nonsup_inst,
        )
        out_cur = _update_suppress_provenance_from_stmt(
            m_stmt, local_cur, in_branch=False
        )

        fwd_add_sup = out_cur.suppressors - local_sup
        fwd_rem_sup = local_sup - out_cur.suppressors
        if fwd_rem_sup:
            cur_sup.difference_update(fwd_rem_sup)
        if fwd_add_sup:
            cur_sup.update(fwd_add_sup)

        fwd_add_nonsup = out_cur.non_suppressors - local_nonsup
        fwd_rem_nonsup = local_nonsup - out_cur.non_suppressors
        if fwd_rem_nonsup:
            cur_nonsup.difference_update(fwd_rem_nonsup)
        if fwd_add_nonsup:
            cur_nonsup.update(fwd_add_nonsup)

        fwd_add_nonsup_inst = out_cur.non_suppressor_instances - local_nonsup_inst
        fwd_rem_nonsup_inst = local_nonsup_inst - out_cur.non_suppressor_instances
        if fwd_rem_nonsup_inst:
            cur_nonsup_inst.difference_update(fwd_rem_nonsup_inst)
        if fwd_add_nonsup_inst:
            cur_nonsup_inst.update(fwd_add_nonsup_inst)

        fwd_add_mods = out_cur.stdlib_modules - local_mods
        fwd_rem_mods = local_mods - out_cur.stdlib_modules
        for k, _ in fwd_rem_mods:
            cur_mods.pop(k, None)
        for k, v in fwd_add_mods:
            cur_mods[k] = v

        fwd_add_imps = out_cur.imported_symbols - local_imps
        fwd_rem_imps = local_imps - out_cur.imported_symbols
        for k, _ in fwd_rem_imps:
            cur_imps.pop(k, None)
        for k, v in fwd_add_imps:
            cur_imps[k] = v

        fwd_add_shad_b = out_cur.shadowed_builtins - local_shad_b
        fwd_rem_shad_b = local_shad_b - out_cur.shadowed_builtins
        if fwd_rem_shad_b:
            cur_shad_b.difference_update(fwd_rem_shad_b)
        if fwd_add_shad_b:
            cur_shad_b.update(fwd_add_shad_b)

        fwd_add_shad_n = out_cur.shadowed_names - local_shad_n
        fwd_rem_shad_n = local_shad_n - out_cur.shadowed_names
        if fwd_rem_shad_n:
            cur_shad_n.difference_update(fwd_rem_shad_n)
        if fwd_add_shad_n:
            cur_shad_n.update(fwd_add_shad_n)

        if out_cur.has_star_import:
            cur_star = True

        if (
            cur_sup
            or cur_nonsup
            or cur_nonsup_inst
            or cur_mods
            or cur_imps
            or cur_all_sup
        ):
            has_pos_or_sup = True

        deltas.append(
            _ModuleStmtDelta(
                stmt_id=stmt_id,
                fwd_add_sup=fwd_add_sup,
                fwd_rem_sup=fwd_rem_sup,
                fwd_add_nonsup=fwd_add_nonsup,
                fwd_rem_nonsup=fwd_rem_nonsup,
                fwd_add_nonsup_inst=fwd_add_nonsup_inst,
                fwd_rem_nonsup_inst=fwd_rem_nonsup_inst,
                fwd_add_mods=fwd_add_mods,
                fwd_rem_mods=fwd_rem_mods,
                fwd_add_imps=fwd_add_imps,
                fwd_rem_imps=fwd_rem_imps,
                fwd_add_shad_b=fwd_add_shad_b,
                fwd_rem_shad_b=fwd_rem_shad_b,
                fwd_add_shad_n=fwd_add_shad_n,
                fwd_rem_shad_n=fwd_rem_shad_n,
                fwd_star=out_cur.has_star_import,
                branch_new_sup=new_sup,
                branch_shad_b=b_delta.shadowed_builtins,
                branch_shad_n=b_delta.shadowed_names,
                branch_star=b_delta.has_star_import,
            )
        )

    full_forward_prov = _SuppressProvenance(
        suppressors=frozenset(cur_sup),
        non_suppressors=frozenset(cur_nonsup),
        stdlib_modules=frozenset(cur_mods.items()),
        shadowed_builtins=frozenset(cur_shad_b),
        has_star_import=cur_star,
        imported_symbols=frozenset(cur_imps.items()),
        shadowed_names=frozenset(cur_shad_n),
        non_suppressor_instances=frozenset(cur_nonsup_inst),
    )
    if not has_pos_or_sup:
        return _ModuleSuppressSeed(
            full_forward_prov=full_forward_prov,
            by_top_stmt_id=MappingProxyType({}),
            has_any_positive_or_suppressor=False,
        )
    deltas_tuple = tuple(deltas)
    idx_map = MappingProxyType(idx_by_id)
    return _ModuleSuppressSeed(
        full_forward_prov=full_forward_prov,
        by_top_stmt_id=_LazyModuleStmtSeedMap(deltas_tuple, idx_map, "combined"),
        has_any_positive_or_suppressor=True,
        prefix_by_top_stmt_id=_LazyModuleStmtSeedMap(deltas_tuple, idx_map, "prefix"),
        suffix_delta_by_top_stmt_id=_LazyModuleStmtSeedMap(
            deltas_tuple, idx_map, "suffix"
        ),
    )


def _get_module_suppress_seed(eff_mod: ast.Module) -> _ModuleSuppressSeed:
    """Returns the cached _ModuleSuppressSeed for eff_mod."""
    if _is_ephemeral_unit_ast(eff_mod):
        return _build_module_suppress_seed(eff_mod)
    cache_key = _make_trusted_scope_cache_key(eff_mod, "module_suppress_seed")
    return cast(
        _ModuleSuppressSeed,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _build_module_suppress_seed(eff_mod),
        ),
    )


def _advance_enclosing_compound_stmt_seed(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    target_node: Optional[ast.AST],
    target_lineno: int,
) -> _SuppressProvenance:
    """Advances provenance into a compound statement enclosing target_node or target_lineno."""
    is_loop = isinstance(stmt, (ast.While, ast.For, ast.AsyncFor))
    if isinstance(stmt, (ast.If, ast.While)):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.test,), prov, allow_positive=not is_loop
        )
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.iter,), prov, allow_positive=True
        )
        prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.target,), prov, allow_positive=False
        )
        iter_is_suspicious = _looks_like_suppressing_ctx_manager(stmt.iter, prov)
        for t_name in _extract_store_names_in_node(stmt.target):
            if iter_is_suspicious and isinstance(stmt.target, ast.Name):
                prov = prov.with_suppressor(t_name, allow_positive=True)
            else:
                prov = prov.clear_name(t_name)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        prov = _advance_prov_through_with_items(stmt, prov)
    elif is_match_node(stmt):
        subject = getattr(stmt, "subject", None)
        if isinstance(subject, ast.AST):
            prov = _update_suppress_provenance_from_expr_walrus(
                (subject,), prov, allow_positive=True
            )

    clauses: List[Tuple[Sequence[ast.AST], Tuple[str, ...], Optional[ast.AST]]] = []
    if isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        clauses.append((stmt.body, (), None))
        if stmt.orelse:
            clauses.append((stmt.orelse, (), None))
    elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.ClassDef)):
        clauses.append((stmt.body, (), None))
    elif is_try_or_trystar_node(stmt):
        clauses.append((getattr(stmt, "body", ()), (), None))
        for h_node in getattr(stmt, "handlers", ()):
            h_name = getattr(h_node, "name", None)
            clauses.append(
                (
                    getattr(h_node, "body", ()),
                    (h_name,) if isinstance(h_name, str) else (),
                    getattr(h_node, "type", None),
                )
            )
        o_stmts = getattr(stmt, "orelse", ())
        if o_stmts:
            clauses.append((o_stmts, (), None))
        f_stmts = getattr(stmt, "finalbody", ())
        if f_stmts:
            clauses.append((f_stmts, (), None))
    elif is_match_node(stmt):
        for case in getattr(stmt, "cases", ()):
            pattern = getattr(case, "pattern", None)
            pat_names = (
                tuple(
                    p_name
                    for p_node in ast.walk(pattern)
                    for p_name in (_extract_pattern_bound_name(p_node),)
                    if p_name is not None
                )
                if isinstance(pattern, ast.AST)
                else ()
            )
            clauses.append(
                (
                    getattr(case, "body", ()),
                    pat_names,
                    getattr(case, "guard", None),
                )
            )

    enclosing_clause_idx: Optional[int] = None
    for c_idx, (c_body, _, _) in enumerate(clauses):
        if any(
            (target_node is not None and s is target_node)
            or (
                target_lineno > 0
                and int(getattr(s, "lineno", 0) or 0)
                <= target_lineno
                <= int(getattr(s, "end_lineno", getattr(s, "lineno", 0)) or 0)
            )
            for s in c_body
        ):
            enclosing_clause_idx = c_idx
            break
    if enclosing_clause_idx is None:
        return prov.merge_branch_conservative(
            _update_suppress_provenance_from_stmt(stmt, prov, in_branch=True)
        )

    c_body, c_bound, c_hdr = clauses[enclosing_clause_idx]
    if is_try_or_trystar_node(stmt) and enclosing_clause_idx > 0:
        t_body = getattr(stmt, "body", ())
        is_orelse = bool(
            getattr(stmt, "orelse", ()) and c_body is getattr(stmt, "orelse", ())
        )
        for tb_stmt in t_body:
            tb_p = _update_suppress_provenance_from_stmt(
                tb_stmt, prov, in_branch=not is_orelse
            )
            prov = tb_p if is_orelse else prov.merge_branch_conservative(tb_p)
    if isinstance(c_hdr, ast.AST):
        prov = _update_suppress_provenance_from_expr_walrus(
            (c_hdr,), prov, allow_positive=False
        )
    for b_name in c_bound:
        prov = prov.clear_name(b_name)

    reached = False
    for sub_stmt in c_body:
        if target_node is not None and sub_stmt is target_node:
            reached = True
            continue
        s_start = int(getattr(sub_stmt, "lineno", 0) or 0)
        s_end = int(getattr(sub_stmt, "end_lineno", s_start) or s_start)
        if not reached and target_lineno > 0 and s_start <= target_lineno <= s_end:
            if target_node is None and s_start >= target_lineno:
                reached = True
            else:
                prov = _advance_enclosing_compound_stmt_seed(
                    sub_stmt, prov, target_node, target_lineno
                )
                reached = True
                continue
        if not reached and 0 < target_lineno <= s_start:
            reached = True
        next_sub = _update_suppress_provenance_from_stmt(
            sub_stmt, prov, in_branch=reached
        )
        prov = prov.merge_branch_conservative(next_sub) if reached else next_sub

    for c_idx, (sib_body, sib_bound, sib_hdr) in enumerate(clauses):
        if c_idx == enclosing_clause_idx and not is_loop:
            continue
        if isinstance(stmt, ast.If) or is_match_node(stmt):
            continue
        sib_prov = prov
        if isinstance(sib_hdr, ast.AST):
            sib_prov = _update_suppress_provenance_from_expr_walrus(
                (sib_hdr,), sib_prov, allow_positive=False
            )
        for b_name in sib_bound:
            sib_prov = sib_prov.clear_name(b_name)
        sib_prov = _update_suppress_provenance_from_stmts(
            sib_body, sib_prov, in_branch=True
        )
        prov = prov.merge_branch_conservative(sib_prov)
    return prov


def _resolve_top_level_ancestor_seed(
    fn_node: ast.AST,
    eff_mod: ast.Module,
    enclosing_nodes: Sequence[ast.AST],
    mod_seed: _ModuleSuppressSeed,
) -> _SuppressProvenance:
    """Resolves module-level suppression seed using the top-level ancestor of fn_node."""
    top_target = enclosing_nodes[0] if enclosing_nodes else fn_node
    if id(top_target) in mod_seed.by_top_stmt_id:
        return mod_seed.by_top_stmt_id[id(top_target)]
    target_lineno = int(getattr(top_target, "lineno", 0) or 0)
    if target_lineno > 0:
        for m_stmt in eff_mod.body:
            m_start = int(getattr(m_stmt, "lineno", 0) or 0)
            m_end = int(getattr(m_stmt, "end_lineno", m_start) or m_start)
            stmt_id = id(m_stmt)
            if m_start <= target_lineno <= m_end and stmt_id in mod_seed.by_top_stmt_id:
                if top_target is m_stmt:
                    return mod_seed.by_top_stmt_id[stmt_id]
                pre_prov = mod_seed.prefix_by_top_stmt_id.get(
                    stmt_id, mod_seed.by_top_stmt_id[stmt_id]
                )
                inner_prov = _advance_enclosing_compound_stmt_seed(
                    m_stmt, pre_prov, top_target, target_lineno
                )
                suf_delta = mod_seed.suffix_delta_by_top_stmt_id.get(stmt_id)
                if suf_delta is not None:
                    inner_prov = inner_prov.apply_branch_delta(
                        suf_delta.shadowed_names,
                        suf_delta.shadowed_builtins,
                        suf_delta.suppressors,
                        suf_delta.has_star_import,
                    )
                return inner_prov
    return mod_seed.full_forward_prov


def _seed_scope_suppress_aliases_uncached(
    fn_node: ast.AST,
    eff_mod: Optional[ast.Module],
    enclosing_nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext],
    u_start: int,
) -> _SuppressProvenance:
    """Uncached implementation of _seed_scope_suppress_aliases."""
    prov = _EMPTY_SUPPRESS_PROVENANCE
    if eff_mod is not None:
        mod_seed = _get_module_suppress_seed(eff_mod)
        if not mod_seed.has_any_positive_or_suppressor:
            prov = mod_seed.full_forward_prov
        elif fn_node is not eff_mod:
            prov = _resolve_top_level_ancestor_seed(
                fn_node, eff_mod, enclosing_nodes, mod_seed
            )
        elif u_start <= 0:
            prov = mod_seed.full_forward_prov
        else:
            reached_cutoff = False
            for m_stmt in eff_mod.body:
                m_start = int(getattr(m_stmt, "lineno", 0) or 0)
                m_end = int(getattr(m_stmt, "end_lineno", m_start) or m_start)
                if not reached_cutoff and 0 < u_start <= m_end and m_start < u_start:
                    prov = _advance_enclosing_compound_stmt_seed(
                        m_stmt, prov, None, u_start
                    )
                    reached_cutoff = True
                    continue
                if not reached_cutoff and 0 < u_start <= m_start:
                    reached_cutoff = True
                next_p = _update_suppress_provenance_from_stmt(
                    m_stmt, prov, in_branch=reached_cutoff
                )
                prov = (
                    prov.merge_branch_conservative(next_p) if reached_cutoff else next_p
                )

    for idx_enc, enc in enumerate(enclosing_nodes):
        enc_args = getattr(enc, "args", None)
        if isinstance(enc_args, ast.arguments):
            for p_name in _extract_arg_names(enc_args):
                prov = prov.clear_name(p_name)
        for tp_name in _extract_node_type_param_names(enc):
            prov = prov.clear_name(tp_name)
        for loc_name in _extract_scope_local_names(enc):
            prov = prov.clear_name(loc_name)
        enc_body = getattr(enc, "body", None)
        if isinstance(enc_body, (list, tuple)):
            next_target = (
                enclosing_nodes[idx_enc + 1]
                if idx_enc + 1 < len(enclosing_nodes)
                else fn_node
            )
            next_target_lineno = int(getattr(next_target, "lineno", 0) or 0)
            reached_fn = False
            for e_stmt in enc_body:
                if e_stmt is next_target:
                    reached_fn = True
                    continue
                e_start = int(getattr(e_stmt, "lineno", 0) or 0)
                e_end = int(getattr(e_stmt, "end_lineno", e_start) or e_start)
                if (
                    not reached_fn
                    and next_target_lineno > 0
                    and e_start <= next_target_lineno <= e_end
                ):
                    prov = _advance_enclosing_compound_stmt_seed(
                        e_stmt, prov, next_target, next_target_lineno
                    )
                    reached_fn = True
                    continue
                next_e = _update_suppress_provenance_from_stmt(
                    e_stmt, prov, in_branch=reached_fn
                )
                prov = prov.merge_branch_conservative(next_e) if reached_fn else next_e

    fn_args = getattr(fn_node, "args", None)
    if isinstance(fn_args, ast.arguments):
        for p_name in _extract_arg_names(fn_args):
            prov = prov.clear_name(p_name)
    for tp_name in _extract_node_type_param_names(fn_node):
        prov = prov.clear_name(tp_name)

    if fn_node is not eff_mod:
        fn_local_names = (
            set(scope_ctx.fn_locals)
            if scope_ctx is not None
            else _extract_scope_local_names(fn_node)
        )
        for loc_name in fn_local_names:
            prov = prov.clear_name(loc_name)
        fn_body = getattr(fn_node, "body", None)
        if isinstance(fn_body, (list, tuple)) and u_start > 0:
            reached_u = False
            for f_stmt in fn_body:
                f_start = int(getattr(f_stmt, "lineno", 0) or 0)
                f_end = int(getattr(f_stmt, "end_lineno", f_start) or f_start)
                if not reached_u and f_start < u_start <= f_end:
                    prov = _advance_enclosing_compound_stmt_seed(
                        f_stmt, prov, None, u_start
                    )
                    reached_u = True
                    continue
                if not reached_u and f_start >= u_start:
                    reached_u = True
                if not reached_u:
                    prov = _update_suppress_provenance_from_stmt(f_stmt, prov)
                else:
                    next_f = _update_suppress_provenance_from_stmt(
                        f_stmt, prov, in_branch=True
                    )
                    prov = prov.merge_branch_conservative(next_f)

    # Soundness check: any binding to 'open', 'builtins', or '__builtins__' anywhere
    # in the module, enclosing scopes, or current scope marks it shadowed unless
    # already proven as an unshadowed non-suppressor or stdlib module in prov.
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
    if not has_star and eff_mod is not None and _node_has_star_import(eff_mod):
        has_star = True
    for sc_node in scope_chain:
        bound_in_chain.update(_extract_scope_local_names(sc_node))
        if (
            not has_star
            and sc_node is not eff_mod
            and _scope_has_direct_star_import(sc_node)
        ):
            has_star = True
    if has_star:
        prov = prov.with_star_import()
    proven_stdlib_aliases = {alias for alias, _ in prov.stdlib_modules}
    fn_own_locals = (
        _extract_scope_local_names(fn_node)
        if (fn_node is not eff_mod and u_start <= 0)
        else set()
    )
    shadowed_b = {
        b_name
        for b_name in (bound_in_chain & _SHADOWABLE_BUILTIN_CTX_NAMES)
        if b_name in fn_own_locals
        or (b_name not in prov.non_suppressors and b_name not in proven_stdlib_aliases)
    }
    if shadowed_b:
        prov = prov.with_shadowed_builtins(shadowed_b)
    return prov


def _seed_scope_suppress_aliases(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    u_start: int = 0,
) -> _SuppressProvenance:
    """Seeds initial context-manager suppression and shadowing provenance for fn_node."""
    eff_mod = (
        mod_tree
        if isinstance(mod_tree, ast.Module)
        else (fn_node if isinstance(fn_node, ast.Module) else None)
    )
    enclosing_nodes: Tuple[ast.AST, ...] = (
        tuple(scope_ctx.enclosing_scopes)
        if scope_ctx is not None and scope_ctx.enclosing_scopes
        else (
            tuple(
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
    )
    aux_nodes: Tuple[ast.AST, ...] = (
        *((eff_mod,) if eff_mod is not None and eff_mod is not fn_node else ()),
        *enclosing_nodes,
    )
    can_cache = not _is_ephemeral_unit_ast(fn_node) and all(
        not _is_ephemeral_unit_ast(n) for n in aux_nodes
    )
    if can_cache and eff_mod is not None and fn_node is not eff_mod:
        if not _tree_contains_scope_node(eff_mod, fn_node) or any(
            n is not eff_mod and not _tree_contains_scope_node(eff_mod, n)
            for n in aux_nodes
        ):
            can_cache = False
    if not can_cache:
        return _seed_scope_suppress_aliases_uncached(
            fn_node, eff_mod, enclosing_nodes, scope_ctx, int(u_start)
        )
    cache_key = _make_trusted_scope_cache_key(
        fn_node,
        "suppress_seed",
        params=(int(u_start),),
        aux_ast_nodes=aux_nodes,
    )
    return cast(
        _SuppressProvenance,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _seed_scope_suppress_aliases_uncached(
                fn_node, eff_mod, enclosing_nodes, scope_ctx, int(u_start)
            ),
        ),
    )


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


def _classify_with_suppression(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> str:
    """Classifies a with/async with statement as 'suppressing', 'non_suppressing', or 'unknown'."""
    if not stmt.items:
        return "non_suppressing"
    prov = _coerce_suppress_provenance(suppress_aliases)
    all_non_suppressing = True
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    ):
        item_mode = _classify_with_item_suppression(
            step.item.context_expr,
            step.post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if item_mode == "suppressing":
            return "suppressing"
        if item_mode != "non_suppressing":
            all_non_suppressing = False
    return "non_suppressing" if all_non_suppressing else "unknown"


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
    ):
        return True
    stack: List[ast.AST] = [expr]
    while stack:
        sub = stack.pop()
        ident: Optional[str] = None
        if isinstance(sub, ast.Name):
            ident = sub.id
            if ident in prov.suppressors:
                return True
        elif isinstance(sub, ast.Attribute):
            ident = sub.attr
            recv_root, _, _ = _unwind_attribute_chain(sub)
            if isinstance(recv_root, ast.Name) and recv_root.id in prov.suppressors:
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
        elif isinstance(sub, (ast.Subscript, ast.Await)):
            stack.append(sub.value)
        elif isinstance(sub, ast.NamedExpr):
            stack.append(sub.target)
            stack.append(sub.value)
        elif isinstance(sub, ast.IfExp):
            stack.append(sub.body)
            stack.append(sub.orelse)
        elif isinstance(sub, ast.BoolOp):
            stack.extend(sub.values)
        elif isinstance(sub, (ast.List, ast.Tuple, ast.Set)):
            stack.extend(sub.elts)
        if ident is not None and ident != "catch_warnings":
            if _ident_has_suspicious_suppress_token(ident):
                return True
    return False


def _with_item_may_suppress_exceptions(
    expr: ast.AST,
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> bool:
    """Returns True if a single withitem context_expr may suppress exceptions under active policy."""
    item_mode = _classify_with_item_suppression(
        expr,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    return item_mode == "suppressing" or (
        item_mode != "non_suppressing"
        and (strict_with_suppression or _looks_like_suppressing_ctx_manager(expr, prov))
    )


class _WithItemProvenanceStep(NamedTuple):
    """Per-item suppression provenance state produced while stepping through a with header."""

    item: ast.withitem
    after_prior_sup: bool
    post_expr_prov: _SuppressProvenance
    item_may_suppress: bool
    opt_vars_conditional: bool
    post_item_prov: _SuppressProvenance


def _iter_with_item_provenance_steps(
    stmt: Union[ast.With, ast.AsyncWith],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool = True,
    encloses_span: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Iterator[_WithItemProvenanceStep]:
    """Yields per-item suppression provenance progression across a With/AsyncWith header."""
    item_prov = prov
    after_sup = False
    for item in stmt.items:
        after_prior_sup = after_sup and not encloses_span
        w_prov = _update_suppress_provenance_from_expr_walrus(
            (item.context_expr,),
            item_prov,
            allow_positive=allow_positive and not after_prior_sup,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        post_expr_prov = (
            item_prov.merge_branch_conservative(w_prov) if after_prior_sup else w_prov
        )
        is_invalid_ctx = _is_invalid_with_context_expr(
            item.context_expr,
            post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        item_sup = is_invalid_ctx or _with_item_may_suppress_exceptions(
            item.context_expr,
            post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        opt_cond = (
            after_prior_sup
            or is_invalid_ctx
            or (
                item_sup
                and item.optional_vars is not None
                and not _is_non_raising_store_target(item.optional_vars)
            )
        )
        post_item_prov = (
            _clear_store_target_provenance(
                item.optional_vars, post_expr_prov, is_conditional=opt_cond
            )
            if item.optional_vars is not None
            else post_expr_prov
        )
        yield _WithItemProvenanceStep(
            item=item,
            after_prior_sup=after_prior_sup,
            post_expr_prov=post_expr_prov,
            item_may_suppress=item_sup,
            opt_vars_conditional=opt_cond,
            post_item_prov=post_item_prov,
        )
        if item_sup:
            after_sup = True
        item_prov = post_item_prov


def _with_header_has_post_suppress_risk(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> bool:
    """Returns True if a multi-item with has an item after a suppressing item or a raising optional_vars on a suppressing item."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    return any(
        step.after_prior_sup or step.opt_vars_conditional
        for step in _iter_with_item_provenance_steps(
            stmt,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
    )


def _resolve_unique_imported_qualname(
    base_id: str,
    prov: _SuppressProvenance,
) -> Optional[str]:
    """Returns the unique imported qualname for base_id when unshadowed, or None."""
    if (
        base_id in prov.shadowed_names
        or base_id in prov.shadowed_builtins
        or prov.has_star_import
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


def _with_stmt_looks_suppressing(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> bool:
    """Returns True if any unproven item in stmt looks like an exception-suppressing manager."""
    prov = _coerce_suppress_provenance(suppress_aliases)
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    ):
        item_mode = _classify_with_item_suppression(
            step.item.context_expr,
            step.post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if item_mode == "suppressing":
            return True
        if item_mode == "unknown" and _looks_like_suppressing_ctx_manager(
            step.item.context_expr, step.post_expr_prov
        ):
            return True
    return False


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


def _apply_non_raising_if_branch_updates(
    node: Union[ast.If, ast.IfExp],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    is_conditional: bool = False,
) -> None:
    """Applies walrus/statement safe updates across test, body, and orelse of If/IfExp."""
    _apply_non_raising_expr_walrus_updates(
        node.test,
        cur_safe,
        cur_bool,
        strict_names=strict_names,
        is_conditional=is_conditional,
    )
    then_safe, then_bool = set(cur_safe), set(cur_bool)
    else_safe, else_bool = set(cur_safe), set(cur_bool)
    if isinstance(node, ast.IfExp):
        _apply_non_raising_expr_walrus_updates(
            node.body,
            then_safe,
            then_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
        )
        _apply_non_raising_expr_walrus_updates(
            node.orelse,
            else_safe,
            else_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
        )
    else:
        _apply_non_raising_seq_safe_updates(
            node.body, then_safe, then_bool, strict_names=strict_names
        )
        _apply_non_raising_seq_safe_updates(
            node.orelse, else_safe, else_bool, strict_names=strict_names
        )
    cur_safe.clear()
    cur_safe.update(then_safe & else_safe)
    cur_bool.clear()
    cur_bool.update(then_bool & else_bool)


def _apply_non_raising_expr_walrus_updates(
    expr: Optional[ast.AST],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    is_conditional: bool = False,
) -> None:
    """Applies left-to-right cur_safe and cur_bool updates from NamedExpr nodes in a non-raising expr."""
    if expr is None or isinstance(expr, ast.Constant):
        return
    if isinstance(expr, ast.IfExp):
        _apply_non_raising_if_branch_updates(
            expr,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
        )
        return
    if isinstance(expr, ast.NamedExpr):
        _apply_non_raising_expr_walrus_updates(
            expr.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
        )
        if isinstance(expr.target, ast.Name):
            t_name = expr.target.id
            val_bool = _is_non_raising_bool_expr(
                expr.value, cur_safe, cur_bool, strict_names=strict_names
            )
            if not is_conditional:
                cur_safe.add(t_name)
                if val_bool:
                    cur_bool.add(t_name)
                else:
                    cur_bool.discard(t_name)
            elif not val_bool:
                cur_bool.discard(t_name)
        return
    sub_exprs: Sequence[Tuple[ast.AST, bool]] = ()
    if isinstance(expr, ast.Lambda):
        sub_exprs = tuple(
            (d, is_conditional)
            for d in (
                *expr.args.defaults,
                *(d for d in expr.args.kw_defaults if d is not None),
            )
        )
    elif isinstance(expr, (ast.Tuple, ast.List)):
        sub_exprs = tuple((elt, is_conditional) for elt in expr.elts)
    elif isinstance(expr, ast.UnaryOp):
        sub_exprs = ((expr.operand, is_conditional),)
    elif isinstance(expr, ast.BoolOp):
        sub_exprs = tuple(
            (val, is_conditional or idx > 0) for idx, val in enumerate(expr.values)
        )
    elif isinstance(expr, ast.Compare):
        sub_exprs = (
            (expr.left, is_conditional),
            *(
                (comp, is_conditional or idx > 0)
                for idx, comp in enumerate(expr.comparators)
            ),
        )
    elif isinstance(expr, ast.Call):
        sub_exprs = (
            (expr.func, is_conditional),
            *((arg, is_conditional) for arg in expr.args),
            *((kw.value, is_conditional) for kw in expr.keywords),
        )
    for sub_node, sub_cond in sub_exprs:
        _apply_non_raising_expr_walrus_updates(
            sub_node, cur_safe, cur_bool, strict_names=strict_names, is_conditional=sub_cond
        )


def _check_and_advance_non_raising_seq(
    nodes: Sequence[ast.AST],
    safe_names: AbstractSet[str],
    safe_bool_names: AbstractSet[str],
    *,
    strict_names: bool,
    require_bool_safe: bool = False,
    conditional_from_idx: Optional[int] = None,
) -> Optional[Tuple[Set[str], Set[str]]]:
    """Verifies each node in nodes is non-raising left-to-right and returns updated (safe, bool) sets."""
    cur_safe = set(safe_names)
    cur_bool = set(safe_bool_names)
    for idx, node in enumerate(nodes):
        ok = (
            _is_non_raising_bool_expr(
                node, cur_safe, cur_bool, strict_names=strict_names
            )
            if require_bool_safe
            else _is_non_raising_expr(
                node, cur_safe, cur_bool, strict_names=strict_names
            )
        )
        if not ok:
            return None
        _apply_non_raising_expr_walrus_updates(
            node,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=(
                conditional_from_idx is not None and idx >= conditional_from_idx
            ),
        )
    return cur_safe, cur_bool


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
        defaults = (
            *expr.args.defaults,
            *(d for d in expr.args.kw_defaults if d is not None),
        )
        return (
            _check_and_advance_non_raising_seq(
                defaults,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            is not None
        )
    if isinstance(expr, (ast.Tuple, ast.List)):
        if any(isinstance(e, ast.Starred) for e in expr.elts):
            return False
        return (
            _check_and_advance_non_raising_seq(
                expr.elts,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            is not None
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
        if not expr.values:
            return False
        advanced = _check_and_advance_non_raising_seq(
            expr.values[:-1],
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
            conditional_from_idx=1,
        )
        return advanced is not None and _is_non_raising_expr(
            expr.values[-1],
            advanced[0],
            advanced[1],
            require_bool_safe=require_bool_safe,
            strict_names=strict_names,
        )
    if isinstance(expr, ast.IfExp):
        advanced = _check_and_advance_non_raising_seq(
            (expr.test,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
        )
        return advanced is not None and all(
            _is_non_raising_expr(
                b,
                advanced[0],
                advanced[1],
                require_bool_safe=require_bool_safe,
                strict_names=strict_names,
            )
            for b in (expr.body, expr.orelse)
        )
    if isinstance(expr, ast.Compare):
        if not all(isinstance(op, (ast.Is, ast.IsNot)) for op in expr.ops):
            return False
        return (
            _check_and_advance_non_raising_seq(
                (expr.left, *expr.comparators),
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                conditional_from_idx=2,
            )
            is not None
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
    *,
    leaf_predicate: Optional[Callable[[ast.AST], bool]] = None,
) -> bool:
    """Returns True if storing value into target avoids unpacking errors and satisfies any leaf predicate."""
    while isinstance(value, ast.NamedExpr):
        value = value.value
    if isinstance(target, ast.Name):
        return True if leaf_predicate is None else (value is not None and leaf_predicate(value))
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))
    ):
        return all(
            _is_non_raising_store_target(t, v, leaf_predicate=leaf_predicate)
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


def _is_non_raising_suppress_or_null_with(
    stmt: ast.AST,
    prov: _SuppressProvenance = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if stmt is a sync With (suppress/nullcontext) or AsyncWith (nullcontext) with proven non-raising entry/exit."""
    if not isinstance(stmt, (ast.With, ast.AsyncWith)) or not stmt.items:
        return False
    is_async = isinstance(stmt, ast.AsyncWith)
    cur_prov = prov
    for item in stmt.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            return False
        cur_prov = _update_suppress_provenance_from_expr_walrus((call,), cur_prov)
        qual = _resolve_suppress_or_null_qualname(call.func, cur_prov)
        if qual is None or (is_async and qual != "contextlib.nullcontext"):
            return False
        if item.optional_vars is not None:
            cur_prov = _clear_store_target_provenance(item.optional_vars, cur_prov)
    return True


def _non_raising_stmt_terminates(stmt: ast.AST) -> bool:
    """Returns True if a proven non-raising statement unconditionally exits the block."""
    if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return any(_non_raising_stmt_terminates(s) for s in stmt.body)
    if isinstance(stmt, ast.If) and stmt.orelse:
        return any(_non_raising_stmt_terminates(s) for s in stmt.body) and any(
            _non_raising_stmt_terminates(s) for s in stmt.orelse
        )
    return False


def _stmt_cannot_raise_in_suppress(
    stmt: ast.AST,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> bool:
    """Returns True if stmt executes and completes without raising a catchable exception."""
    if isinstance(stmt, (ast.Pass, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, ast.Delete):
        return bool(stmt.targets) and all(
            isinstance(t, ast.Name) and t.id in safe_names for t in stmt.targets
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
            and all(_is_non_raising_store_target(t, stmt.value) for t in stmt.targets)
        )
    if isinstance(stmt, ast.AnnAssign):
        ann_sets = _check_and_advance_non_raising_seq(
            (stmt.annotation,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
        )
        if ann_sets is None:
            return False
        return stmt.value is None or (
            _is_non_raising_expr(
                stmt.value,
                ann_sets[0],
                ann_sets[1],
                strict_names=strict_names,
            )
            and _is_non_raising_store_target(stmt.target, stmt.value)
        )
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.decorator_list:
            return False
        return (
            _check_and_advance_non_raising_seq(
                tuple(iter_nested_scope_header_nodes(stmt)),
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
            )
            is not None
        )
    if isinstance(stmt, ast.If):
        test_sets = _check_and_advance_non_raising_seq(
            (stmt.test,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
        )
        if test_sets is None:
            return False
        test_safe, test_bool = test_sets
        test_prov = _update_suppress_provenance_from_expr_walrus(
            (stmt.test,), _coerce_suppress_provenance(suppress_aliases)
        )
        return (
            not _split_suppress_with_body(
                stmt.body,
                safe_names=test_safe,
                safe_bool_names=test_bool,
                strict_names=strict_names,
                suppress_aliases=test_prov,
            )[1]
            and not _split_suppress_with_body(
                stmt.orelse,
                safe_names=test_safe,
                safe_bool_names=test_bool,
                strict_names=strict_names,
                suppress_aliases=test_prov,
            )[1]
        )
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        prov = _coerce_suppress_provenance(suppress_aliases)
        if not _is_non_raising_suppress_or_null_with(stmt, prov):
            return False
        inner_safe = set(safe_names)
        inner_bool = set(safe_bool_names)
        inner_prov = prov
        for item in stmt.items:
            if not (
                (item.optional_vars is None or isinstance(item.optional_vars, ast.Name))
                and isinstance(item.context_expr, ast.Call)
            ):
                return False
            call_args = (*item.context_expr.args, *item.context_expr.keywords)
            if any(
                isinstance(a, ast.Starred) or getattr(a, "arg", "k") is None
                for a in call_args
            ):
                return False
            arg_exprs = tuple(
                a.value if isinstance(a, ast.keyword) else a for a in call_args
            )
            if (
                _check_and_advance_non_raising_seq(
                    arg_exprs,
                    inner_safe | _SAFE_EXCEPTION_TYPE_NAMES,
                    inner_bool,
                    strict_names=strict_names,
                )
                is None
            ):
                return False
            _apply_non_raising_expr_walrus_updates(
                item.context_expr,
                inner_safe,
                inner_bool,
                strict_names=strict_names,
            )
            inner_prov = _update_suppress_provenance_from_expr_walrus(
                (item.context_expr,), inner_prov
            )
            if isinstance(item.optional_vars, ast.Name):
                inner_safe.add(item.optional_vars.id)
                inner_bool.discard(item.optional_vars.id)
                inner_prov = _clear_store_target_provenance(
                    item.optional_vars, inner_prov
                )
        return not _split_suppress_with_body(
            stmt.body,
            safe_names=inner_safe,
            safe_bool_names=inner_bool,
            strict_names=strict_names,
            suppress_aliases=inner_prov,
        )[1]
    return False


def _collect_target_unpack_bool_flags(
    target: ast.AST,
    value: Optional[ast.AST],
    rhs_safe: AbstractSet[str],
    rhs_bool: AbstractSet[str],
    *,
    strict_names: bool,
) -> List[Tuple[str, bool]]:
    """Collects left-to-right (target_name, is_bool_safe) pairs evaluated against RHS state."""
    while isinstance(value, ast.NamedExpr):
        value = value.value
    if isinstance(target, ast.Name):
        return [
            (
                target.id,
                _is_non_raising_bool_expr(
                    value, rhs_safe, rhs_bool, strict_names=strict_names
                ),
            )
        ]
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))
    ):
        pairs: List[Tuple[str, bool]] = []
        for t_elt, v_elt in zip(target.elts, value.elts):
            pairs.extend(
                _collect_target_unpack_bool_flags(
                    t_elt,
                    v_elt,
                    rhs_safe,
                    rhs_bool,
                    strict_names=strict_names,
                )
            )
        return pairs
    return [(t_name, False) for t_name in _extract_definite_target_names(target)]


def _apply_target_unpack_safe_updates(
    target: ast.AST,
    value: Optional[ast.AST],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
) -> None:
    """Updates cur_safe and cur_bool for an assignment target matched against value."""
    for t_name, is_bool_safe in _collect_target_unpack_bool_flags(
        target,
        value,
        frozenset(cur_safe),
        frozenset(cur_bool),
        strict_names=strict_names,
    ):
        cur_safe.add(t_name)
        if is_bool_safe:
            cur_bool.add(t_name)
        else:
            cur_bool.discard(t_name)


def _apply_non_raising_stmt_safe_updates(
    b_stmt: ast.stmt,
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
) -> None:
    """Updates cur_safe and cur_bool after a statement proven non-raising by _stmt_cannot_raise_in_suppress."""
    if isinstance(b_stmt, ast.Delete):
        for tgt in b_stmt.targets:
            if isinstance(tgt, ast.Name):
                cur_safe.discard(tgt.id)
                cur_bool.discard(tgt.id)
    elif isinstance(b_stmt, (ast.Return, ast.Expr)):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value, cur_safe, cur_bool, strict_names=strict_names
        )
    elif isinstance(b_stmt, ast.Assign):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value, cur_safe, cur_bool, strict_names=strict_names
        )
        for tgt in b_stmt.targets:
            _apply_target_unpack_safe_updates(
                tgt,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
            )
    elif isinstance(b_stmt, ast.AnnAssign):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.annotation, cur_safe, cur_bool, strict_names=strict_names
        )
        if b_stmt.value is not None:
            _apply_non_raising_expr_walrus_updates(
                b_stmt.value, cur_safe, cur_bool, strict_names=strict_names
            )
            _apply_target_unpack_safe_updates(
                b_stmt.target,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
            )
    elif isinstance(b_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for h in iter_nested_scope_header_nodes(b_stmt):
            _apply_non_raising_expr_walrus_updates(
                h, cur_safe, cur_bool, strict_names=strict_names
            )
        cur_safe.add(b_stmt.name)
        cur_bool.add(b_stmt.name)
    elif isinstance(b_stmt, (ast.With, ast.AsyncWith)):
        for item in b_stmt.items:
            _apply_non_raising_expr_walrus_updates(
                item.context_expr,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
            )
            if isinstance(item.optional_vars, ast.Name):
                cur_safe.add(item.optional_vars.id)
                cur_bool.discard(item.optional_vars.id)
        _apply_non_raising_seq_safe_updates(
            b_stmt.body, cur_safe, cur_bool, strict_names=strict_names
        )
    elif isinstance(b_stmt, ast.If):
        _apply_non_raising_if_branch_updates(
            b_stmt, cur_safe, cur_bool, strict_names=strict_names
        )


def _apply_non_raising_seq_safe_updates(
    stmts: Sequence[ast.stmt],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
) -> None:
    """Propagates safe bindings sequentially across a non-raising statement list until termination."""
    for sub_s in stmts:
        if _non_raising_stmt_terminates(sub_s):
            break
        _apply_non_raising_stmt_safe_updates(
            sub_s, cur_safe, cur_bool, strict_names=strict_names
        )


def _split_suppress_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a potentially suppressing with-block body into (non_raising_prefix, potentially_raising_suffix)."""
    safe_prefix: List[ast.stmt] = []
    cur_prov = _coerce_suppress_provenance(suppress_aliases)
    cur_safe: Set[str] = (
        set(safe_names)
        | set(cur_prov.suppressors)
        | set(cur_prov.non_suppressors)
        | set(cur_prov.non_suppressor_instances)
        | {alias for alias, _ in cur_prov.stdlib_modules}
        | {alias for alias, _ in cur_prov.imported_symbols}
    )
    cur_bool: Set[str] = set(safe_bool_names)
    for idx, b_stmt in enumerate(body):
        if _stmt_cannot_raise_in_suppress(
            b_stmt,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            suppress_aliases=cur_prov,
        ):
            safe_prefix.append(b_stmt)
            if _non_raising_stmt_terminates(b_stmt):
                return safe_prefix, []
            _apply_non_raising_stmt_safe_updates(
                b_stmt, cur_safe, cur_bool, strict_names=strict_names
            )
            cur_prov = _update_suppress_provenance_from_stmt(b_stmt, cur_prov)
        else:
            return safe_prefix, list(body[idx:])
    return safe_prefix, []


def _split_unknown_with_body(
    body: Sequence[ast.stmt],
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits an unknown context manager's body into (guaranteed_non_raising_prefix, potentially_raising_suffix)."""
    return _split_suppress_with_body(
        body,
        safe_names=safe_names,
        strict_names=strict_names,
        suppress_aliases=suppress_aliases,
    )


def _advance_prov_through_with_items(
    stmt: Union[ast.With, ast.AsyncWith],
    prov: _SuppressProvenance,
    *,
    safe_names_out: Optional[Set[str]] = None,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Advances provenance across with-header items (walrus and as-targets) for body analysis."""
    item_prov = prov
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    ):
        if (
            step.item.optional_vars is not None
            and not step.opt_vars_conditional
            and safe_names_out is not None
        ):
            safe_names_out.update(
                _extract_definite_target_names(step.item.optional_vars)
            )
        item_prov = step.post_item_prov
    return item_prov


def _record_with_items_heuristic_ctx_managers(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> None:
    """Records any unknown context-manager items in stmt that rely on the strict-mode name heuristic."""
    if strict_with_suppression:
        return
    prov = _coerce_suppress_provenance(suppress_aliases)
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    ):
        expr = step.item.context_expr
        item_mode = _classify_with_item_suppression(
            expr,
            step.post_expr_prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        if (
            item_mode == "unknown"
            and not _is_invalid_with_context_expr(
                expr,
                step.post_expr_prov,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            and not _looks_like_suppressing_ctx_manager(expr, step.post_expr_prov)
        ):
            _warn_heuristic_non_suppressing_ctx_manager(expr, step.post_expr_prov)


def _split_with_body_by_mode(
    stmt: Union[ast.With, ast.AsyncWith],
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    *,
    body_suppress_aliases: Optional[Union[_SuppressProvenance, Iterable[str]]] = None,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a with-block body into (guaranteed_prefix, potentially_suppressed_suffix) across 3 suppression states."""
    _record_with_items_heuristic_ctx_managers(
        stmt,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    if _with_header_has_post_suppress_risk(
        stmt,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    ):
        return [], list(stmt.body)
    w_mode = _classify_with_suppression(
        stmt,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    body_prov = (
        _coerce_suppress_provenance(body_suppress_aliases)
        if body_suppress_aliases is not None
        else _advance_prov_through_with_items(
            stmt,
            _coerce_suppress_provenance(suppress_aliases),
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
    )
    if w_mode == "suppressing":
        return _split_suppress_with_body(
            stmt.body,
            safe_names=safe_names,
            strict_names=strict_names,
            suppress_aliases=body_prov,
        )
    if w_mode == "unknown":
        if strict_with_suppression or _with_stmt_looks_suppressing(
            stmt,
            suppress_aliases,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        ):
            return _split_unknown_with_body(
                stmt.body,
                safe_names=safe_names,
                strict_names=strict_names,
                suppress_aliases=body_prov,
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

    if isinstance(expr, _INNER_SCOPE_TYPES):
        for hdr in iter_nested_scope_header_nodes(expr):
            d_h, c_h = _walrus_assignment_in_expr(hdr, is_conditional)
            definite.update(d_h)
            conditional.update(c_h)
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
        d_b, c_b = _walrus_assignment_in_expr(expr.body, False)
        d_o, c_o = _walrus_assignment_in_expr(expr.orelse, False)
        definite.update(d_t)
        both_def = (d_b & d_o) if not is_conditional else set()
        definite.update(both_def)
        conditional.update(c_t | ((d_b | c_b | d_o | c_o) - both_def))
        return definite, conditional - definite

    if isinstance(expr, ast.Compare):
        d_l, c_l = _walrus_assignment_in_expr(expr.left, is_conditional)
        definite.update(d_l)
        conditional.update(c_l)
        d_seq, c_seq = _walrus_in_sequence(expr.comparators, is_conditional)
        definite.update(d_seq)
        conditional.update(c_seq)
        return definite, conditional - definite

    for child in ast.iter_child_nodes(expr):
        target_child = child.value if isinstance(child, ast.keyword) else child
        if isinstance(target_child, ast.AST):
            d_ch, c_ch = _walrus_assignment_in_expr(target_child, is_conditional)
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
        elif is_match_node(stmt):  # pragma: no cover (py310+)
            for case in getattr(stmt, "cases", []):
                deleted.update(_extract_deleted_names(getattr(case, "body", [])))
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
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    loop_control_terminates: bool = False,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> bool:
    """Returns True if the statement sequence unconditionally terminates."""
    _, _, term, _ = _analyze_block_assignment_impl(
        statements,
        definite=set(safe_names),
        suppress_aliases=suppress_aliases,
        loop_control_terminates=loop_control_terminates,
        known_safe_names=safe_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    return term


def _filter_scope_names(
    names: Iterable[str],
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
) -> Set[str]:
    """Filters candidate bound names by scope nonlocal/global rules when active."""
    if active_nonlocals is None and not fn_globals:
        return set(names)
    return {n for n in names if _can_bind_in_fn_scope(n, active_nonlocals, fn_globals)}


def _extract_block_combined_deletions(stmts: Sequence[ast.AST]) -> Set[str]:
    """Returns combined deep and shallow deleted/unbound variable names for stmts."""
    if not stmts:
        return set()
    return _extract_deleted_names(stmts) | set().union(
        *(_extract_shallow_deleted_names(s) for s in stmts)
    )


def _analyze_if_stmt_assignment(
    stmt: ast.If,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes an ast.If statement and updates definite, conditional, may_del, and live_in_del in place."""
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
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
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
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
    else:
        o_def, o_cond, o_term, o_prov = set(), set(), False, local_prov
    local_prov = b_prov.merge_branch_conservative(o_prov)
    live_in_del.update((b_live | o_live) - definite)
    eff_b_del = _extract_block_combined_deletions(stmt.body) - b_def
    eff_o_del = _extract_block_combined_deletions(stmt.orelse) - o_def
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
        else:
            both = b_def & o_def
            definite.update(both)
            conditional.update((b_def | b_cond | o_def | o_cond) - definite)
            may_del.difference_update(both)
            may_del.update(b_may | o_may)
        for v_del in eff_b_del & eff_o_del:
            definite.discard(v_del)
            conditional.discard(v_del)
        _demote_deletions_to_conditional(eff_b_del ^ eff_o_del, definite, conditional)
        if b_term and o_term:
            return True, local_prov
    elif not b_term:
        conditional.update((b_def | b_cond) - definite)
        may_del.update(b_may)
        _demote_deletions_to_conditional(eff_b_del, definite, conditional)
    return False, local_prov


def _analyze_with_stmt_assignment(
    stmt: Union[ast.With, ast.AsyncWith],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    bind_def_cond: Any,
    apply_expr_effects: Any,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a With/AsyncWith statement and updates definite, conditional, may_del, and live_in_del in place."""
    header_prov = local_prov
    for step in _iter_with_item_provenance_steps(
        stmt,
        local_prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    ):
        item = step.item
        apply_expr_effects(item.context_expr, is_cond=step.after_prior_sup)
        if item.optional_vars is not None:
            if step.opt_vars_conditional:
                bind_def_cond((), _extract_store_names_in_node(item.optional_vars))
            else:
                bind_def_cond(_extract_definite_target_names(item.optional_vars))
            apply_expr_effects(item.optional_vars, is_cond=step.opt_vars_conditional)
        local_prov = step.post_item_prov
    cur_safe = definite | set(known_safe_names)
    safe_prefix, risky_suffix = _split_with_body_by_mode(
        stmt,
        header_prov,
        body_suppress_aliases=local_prov,
        safe_names=cur_safe,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
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
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    if w_term:
        return True, local_prov
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
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        local_prov = local_prov.merge_branch_conservative(w_prov)
        live_in_del.update(w_live - definite)
        may_del.update(w_may)
        conditional.update((w_def | w_cond) - definite)
        _demote_deletions_to_conditional(
            _extract_deleted_names(risky_suffix), definite, conditional
        )
        for b_stmt in risky_suffix:
            definite.difference_update(_extract_shallow_deleted_names(b_stmt))
    return False, local_prov


def _analyze_try_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    apply_expr_effects: Any,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a Try/TryStar statement and updates definite, conditional, may_del, and live_in_del in place."""
    t_body = list(getattr(stmt, "body", []))
    raw_try_del = _extract_block_combined_deletions(t_body)
    is_trystar = is_trystar_node(stmt)
    cur_safe = definite | set(known_safe_names)
    t_safe_prefix, t_risky_suffix = _split_suppress_with_body(
        t_body,
        safe_names=cur_safe,
        strict_names=strict_names,
        suppress_aliases=local_prov,
    )
    try_prefix_prov = _update_suppress_provenance_from_stmts(
        t_safe_prefix,
        local_prov,
        in_branch=False,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    try_mid_prov = _update_suppress_provenance_from_stmts(
        t_risky_suffix,
        try_prefix_prov,
        in_branch=True,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    t_may: Set[str] = set()
    t_live: Set[str] = set()
    t_def, t_cond, t_term, try_body_prov = _analyze_block_assignment_impl(
        t_body,
        may_deleted=t_may,
        live_in_deleted=t_live,
        suppress_aliases=local_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        known_safe_names=cur_safe,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    handler_base_prov = try_prefix_prov.merge_branch_conservative(
        try_mid_prov
    ).merge_branch_conservative(try_body_prov)
    live_in_del.update(t_live - definite)
    o_stmts = list(getattr(stmt, "orelse", []))
    o_try_may: Set[str] = set()
    o_def: Set[str] = set()
    try_body_term = t_term
    if o_stmts and not try_body_term:
        o_try_live: Set[str] = set()
        o_def, o_cond, o_term, try_body_prov = _analyze_block_assignment_impl(
            o_stmts,
            may_deleted=o_try_may,
            live_in_deleted=o_try_live,
            suppress_aliases=try_body_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            known_safe_names=cur_safe | t_def,
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        live_in_del.update(o_try_live - (definite | t_def))
        try_def = t_def | o_def
        try_cond = (t_cond | o_cond) - try_def
        if not o_term:
            succ_may = (t_may - o_def) | o_try_may
        else:
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
    surviving_provs: List[_SuppressProvenance] = []
    if not t_term and not (is_trystar and handlers):
        surviving_provs.append(try_body_prov)
    final_prov = handler_base_prov.merge_branch_conservative(try_body_prov)

    for h in handlers:
        h_body = list(getattr(h, "body", []))
        if getattr(h, "type", None) is not None:
            apply_expr_effects(h.type, is_cond=True)
        h_prov = handler_base_prov
        h_name = getattr(h, "name", None)
        h_bound: Set[str] = set()
        if isinstance(h_name, str):
            h_prov = h_prov.clear_name(h_name)
            if _can_bind_in_fn_scope(h_name, active_nonlocals, fn_globals):
                h_bound.add(h_name)
        h_may: Set[str] = set()
        h_live: Set[str] = set()
        h_d, h_c, h_raw_term, h_out_prov = _analyze_block_assignment_impl(
            h_body,
            may_deleted=h_may,
            live_in_deleted=h_live,
            suppress_aliases=h_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            known_safe_names=(cur_safe - raw_try_del) | h_bound,
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        final_prov = final_prov.merge_branch_conservative(h_out_prov)
        h_d.difference_update(h_bound)
        h_raw_bound = {h_name} if isinstance(h_name, str) else set()
        h_may.update(h_raw_bound)
        live_in_del.update((h_live - h_raw_bound) - definite)
        h_term = not is_trystar and h_raw_term
        h_body_dels = _extract_block_combined_deletions(h_body)
        eff_h_del = (h_body_dels - h_d) | h_raw_bound
        reach_h_del = eff_h_del | (raw_try_del - h_d)
        h_dels.append((eff_h_del, reach_h_del, h_term))
        if not h_term:
            h_defs.append(h_d)
            h_mays.append(h_may | ((raw_try_del | t_may) - h_d))
            surviving_provs.append(h_out_prov)
        conditional.update(((h_d | h_c) - h_raw_bound) - definite)

    all_handlers_terminate = bool(
        handlers and not is_trystar and all(h_term for _, _, h_term in h_dels)
    )
    if is_trystar and handlers:
        conditional.update((t_def | t_cond | try_def | try_cond) - definite)
        may_del.update(t_may | o_try_may | raw_try_del)
        for hm in h_mays:
            may_del.update(hm)
    else:
        if not handlers or (not t_term and all_handlers_terminate):
            definite.update(try_def)
            conditional.update(try_cond - definite)
            may_del.difference_update(try_def)
            if not t_term:
                may_del.update(succ_may)
        elif h_defs:
            common = (
                try_def.intersection(*h_defs)
                if not t_term
                else set.intersection(*h_defs)
            )
            definite.update(common)
            cond_pool = try_def | try_cond
            conditional.update(cond_pool - definite)
            may_del.difference_update(common)
            if not t_term:
                may_del.update(succ_may)
            for hm in h_mays:
                may_del.update(hm)

    eff_try_del = raw_try_del - t_def
    eff_o_del = (
        (_extract_block_combined_deletions(o_stmts) - o_def)
        if not try_body_term
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
        _demote_deletions_to_conditional(all_try_dels, definite, conditional)
    if is_trystar and handlers:
        definite.difference_update(_extract_shallow_deleted_names(stmt))

    post_try_prov = final_prov
    if surviving_provs and not (is_trystar and handlers):
        post_try_prov = surviving_provs[0]
        for sp in surviving_provs[1:]:
            post_try_prov = post_try_prov.merge_branch_conservative(sp)
        post_try_prov = _SuppressProvenance(
            suppressors=post_try_prov.suppressors | final_prov.suppressors,
            non_suppressors=post_try_prov.non_suppressors,
            stdlib_modules=post_try_prov.stdlib_modules,
            shadowed_builtins=post_try_prov.shadowed_builtins
            | final_prov.shadowed_builtins,
            has_star_import=post_try_prov.has_star_import or final_prov.has_star_import,
            imported_symbols=post_try_prov.imported_symbols,
            shadowed_names=post_try_prov.shadowed_names,
            non_suppressor_instances=post_try_prov.non_suppressor_instances,
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
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        post_f_prov = _update_suppress_provenance_from_stmts(
            f_body,
            post_try_prov,
            in_branch=False,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        local_prov = _SuppressProvenance(
            suppressors=post_f_prov.suppressors | f_out_prov.suppressors,
            non_suppressors=post_f_prov.non_suppressors,
            stdlib_modules=post_f_prov.stdlib_modules,
            shadowed_builtins=post_f_prov.shadowed_builtins
            | f_out_prov.shadowed_builtins,
            has_star_import=post_f_prov.has_star_import or f_out_prov.has_star_import,
            imported_symbols=post_f_prov.imported_symbols,
            shadowed_names=post_f_prov.shadowed_names,
            non_suppressor_instances=post_f_prov.non_suppressor_instances,
        )
    else:
        local_prov = post_try_prov

    try_terminates = f_term or (t_term and (not handlers or all_handlers_terminate))
    return try_terminates, local_prov


def _analyze_match_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    apply_expr_effects: Any,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a Match statement and updates definite, conditional, may_del, and live_in_del in place."""
    subject = getattr(stmt, "subject", None)
    if isinstance(subject, ast.AST):
        local_prov = _update_suppress_provenance_from_expr_walrus(
            (subject,), local_prov
        )
        apply_expr_effects(subject)
    cur_safe = definite | set(known_safe_names)
    case_defs: List[Set[str]] = []
    non_term_case_defs: List[Set[str]] = []
    case_conds: Set[str] = set()
    case_dels: Set[str] = set()
    all_case_dels: Set[str] = set()
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
            d_g_raw, c_g_raw = _walrus_assignment_in_expr(guard, is_conditional=False)
            d_g = _filter_scope_names(d_g_raw, active_nonlocals, fn_globals)
            c_g = _filter_scope_names(c_g_raw, active_nonlocals, fn_globals)
        case_init_def = pattern_stores | d_g
        case_prov = local_prov
        for p_s in raw_pattern_stores:
            case_prov = case_prov.clear_name(p_s)
        c_may: Set[str] = set()
        c_live: Set[str] = set()
        case_body = getattr(case, "body", [])
        c_def, c_cond, c_term, c_out_prov = _analyze_block_assignment_impl(
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
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        merged_match_prov = merged_match_prov.merge_branch_conservative(c_out_prov)
        live_in_del.update((c_live - case_init_def) - definite)
        case_defs.append(c_def)
        case_conds.update(c_def | c_cond)
        c_del = _extract_block_combined_deletions(case_body) - c_def
        all_case_dels.update(c_del)
        if not c_term:
            case_dels.update(c_del)
            non_term_case_defs.append(c_def)
            non_term_case_dels.append(c_del)
            may_del.update(c_may - raw_pattern_stores)
        elif loop_control_terminates:
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
        {v for v in case_dels if all(v in d for d in non_term_case_dels)}
        if (has_irrefutable_default and non_term_case_dels)
        else set()
    )
    for v_del in all_del:
        definite.discard(v_del)
        conditional.discard(v_del)
    _demote_deletions_to_conditional(case_dels - all_del, definite, conditional)
    if case_defs and has_irrefutable_default and not non_term_case_defs:
        _demote_deletions_to_conditional(all_case_dels, definite, conditional)
        return True, local_prov
    return False, local_prov


def _analyze_block_assignment_impl(
    statements: Sequence[ast.AST],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    active_nonlocals: Optional[Set[str]] = None,
    fn_globals: AbstractSet[str] = frozenset(),
    loop_control_terminates: bool = False,
    known_safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Tuple[Set[str], Set[str], bool, _SuppressProvenance]:
    """Unified single-pass analyzer for definite/conditional bindings, deletions, and termination."""
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()
    may_del = may_deleted if may_deleted is not None else set()
    live_in_del = live_in_deleted if live_in_deleted is not None else set()
    local_prov = _coerce_suppress_provenance(suppress_aliases)

    def _bind_def_cond(d_raw: Iterable[str], c_raw: Iterable[str] = ()) -> None:
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
        d_expr, c_expr = _walrus_assignment_in_expr(expr_node, is_conditional=is_cond)
        _bind_def_cond(d_expr, c_expr)

    for stmt in statements:
        before_def = set(definite)
        if isinstance(stmt, ast.Assign):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            for target in stmt.targets:
                _bind_def_cond(_extract_definite_target_names(target))
                _apply_expr_effects(target)
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            if stmt.value is not None:
                _bind_def_cond(_extract_definite_target_names(stmt.target))
                _apply_expr_effects(stmt.target)
                _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            _bind_def_cond(_extract_definite_target_names(stmt.target))
            _apply_expr_effects(stmt.target)
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.Expr):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            _apply_expr_effects(stmt.value)
        elif isinstance(stmt, ast.Delete):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            for target in stmt.targets:
                _apply_expr_effects(target)
                for node in ast.walk(target):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                        if node.id not in definite:
                            live_in_del.add(node.id)
                        definite.discard(node.id)
                        conditional.discard(node.id)
                        may_del.add(node.id)
        elif isinstance(stmt, ast.Raise):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            _apply_expr_effects(stmt.exc)
            _apply_expr_effects(stmt.cause)
            may_del.difference_update(definite - before_def)
            return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, ast.Return):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            _apply_expr_effects(stmt.value)
            may_del.difference_update(definite - before_def)
            return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            if loop_control_terminates:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, ast.If):
            local_prov = _update_suppress_provenance_from_expr_walrus(
                (stmt.test,), local_prov
            )
            _apply_expr_effects(stmt.test)
            if_term, local_prov = _analyze_if_stmt_assignment(
                stmt,
                definite,
                conditional,
                may_del,
                live_in_del,
                local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=known_safe_names,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
            if if_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
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
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
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
                if isinstance(p_node, ast.Name) and isinstance(p_node.ctx, ast.Store):
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
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
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
            w_term, local_prov = _analyze_with_stmt_assignment(
                stmt,
                definite,
                conditional,
                may_del,
                live_in_del,
                local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=known_safe_names,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
                bind_def_cond=_bind_def_cond,
                apply_expr_effects=_apply_expr_effects,
            )
            if w_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif is_try_or_trystar_node(stmt):
            t_term, local_prov = _analyze_try_stmt_assignment(
                stmt,
                definite,
                conditional,
                may_del,
                live_in_del,
                local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=known_safe_names,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
                apply_expr_effects=_apply_expr_effects,
            )
            if t_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif is_match_node(stmt):  # pragma: no cover (py310+)
            m_term, local_prov = _analyze_match_stmt_assignment(
                stmt,
                definite,
                conditional,
                may_del,
                live_in_del,
                local_prov,
                active_nonlocals=active_nonlocals,
                fn_globals=fn_globals,
                loop_control_terminates=loop_control_terminates,
                known_safe_names=known_safe_names,
                strict_names=strict_names,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
                apply_expr_effects=_apply_expr_effects,
            )
            if m_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            for hdr in iter_nested_scope_header_nodes(stmt):
                _apply_expr_effects(hdr)
            _bind_def_cond((stmt.name,))
        elif is_type_alias_node(stmt):  # pragma: no cover (py312+)
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            ta_name = _extract_type_alias_or_pattern_name(stmt)
            if ta_name is not None:
                _bind_def_cond((ta_name,))
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            imp_names: Set[str] = set()
            _record_import_names(stmt, imp_names)
            _bind_def_cond(imp_names)
        elif isinstance(stmt, ast.Assert):
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)
            _apply_expr_effects(stmt.test)
            if stmt.msg is not None:
                _apply_expr_effects(stmt.msg, is_cond=True)
        else:
            local_prov = _update_suppress_provenance_from_stmt(stmt, local_prov)

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
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
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
        strict_with_suppression=strict_with_suppression,
    )
    return def_out, cond_out


def _extract_block_definite_bindings_and_term(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Tuple[Set[str], bool]:
    """Extracts (unconditionally_bound_names, block_terminates) across a statement sequence.

    Passes ``loop_control_terminates=True`` because callers inspect pre-unit or
    intra-loop compound statements (``if``/``try``/``match``) where a ``break`` or
    ``continue`` branch exits the clause without falling through to statements
    immediately following the compound statement.
    """
    def_out, _, term, _ = _analyze_block_assignment_impl(
        stmts,
        definite=set(initial_bound),
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=True,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    return def_out, term


def _extract_block_definite_bindings(
    stmts: Sequence[ast.AST],
    active_nonlocals: Optional[Set[str]],
    initial_bound: AbstractSet[str] = frozenset(),
    fn_globals: AbstractSet[str] = frozenset(),
    *,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> Set[str]:
    """Extracts variable names unconditionally bound across a statement sequence."""
    return _extract_block_definite_bindings_and_term(
        stmts,
        active_nonlocals,
        initial_bound,
        fn_globals,
        suppress_aliases=suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )[0]
