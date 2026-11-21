"""Context-manager suppression provenance, block termination, and unified definite-assignment engine."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from enum import Enum
import logging
import re
import threading
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
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
    record_wildcard_ctx_manager_match,
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
    scope_fingerprint_verification_session,
)
from pydoppelgangerhunt.fixer._df_cell_carriers import (
    _FnScopeContext,
    _can_bind_in_fn_scope,
    _extract_store_names_in_node,
)
from pydoppelgangerhunt.fixer._df_exit_protocol import (
    _NO_PATCH_SCAN,
    _STDLIB_NON_SUPPRESSING_MODULES,
    ExitProtocolPatchScan,
    is_factory_rebound_target,
    scan_exit_protocol_patches,
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
    _MAX_AST_RECURSION_DEPTH,
    _collect_func_params as _extract_arg_names,
    _extract_assigned_names,
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

_SAFE_METHOD_DECORATORS: frozenset[str] = frozenset(
    {"staticmethod", "classmethod", "property"}
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
        *_SAFE_METHOD_DECORATORS,
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
    "handl",
    "safe",
    "failsafe",
    "rescue",
    "recover",
    "allow",
    "continue_on",
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
    "handl",
)

_SUSPICIOUS_COMPOUND_TOKEN_SPLITS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "subtest": ("sub", "test"),
        "exitstack": ("exit", "stack"),
        "cancelscope": ("cancel", "scope"),
        "fallback": ("fall", "back"),
        "noop": ("no", "op"),
        "failsafe": ("fail", "safe"),
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
        "mute": ("mutes", "muted", "muter", "muters"),
        "hush": ("hushes", "hushed", "hushing"),
        "discard": ("discards", "discarded", "discarding"),
        "mask": ("masks", "masking", "masker", "maskers"),
        "safe": ("safely", "safeguard"),
        "failsafe": ("failsafes",),
        "rescue": ("rescues", "rescued", "rescuing", "rescuer", "rescuers"),
        "recover": ("recovers", "recovered", "recovering", "recovery", "recoverer"),
        "allow": ("allows", "allowed", "allowing"),
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
    """Structured provenance for context-manager suppression and stdlib non-suppression.

    Note:
        ``has_exit_patch`` records that the analyzed module patches the
        ``__exit__`` / ``__aexit__`` protocol at runtime (or executes dynamic code).
        While set, positive non-suppression proofs derived from class bodies or
        pure-Python stdlib managers are withheld (only unpatchable builtin ``open``
        and explicit ``non_suppressing_ctx_managers`` entries remain trusted), and
        names in ``exit_patch_targets`` bind as suspicious aliases.
    """

    suppressors: frozenset[str] = frozenset()
    non_suppressors: frozenset[str] = frozenset()
    stdlib_modules: frozenset[Tuple[str, str]] = frozenset()
    shadowed_builtins: frozenset[str] = frozenset()
    has_star_import: bool = False
    imported_symbols: frozenset[Tuple[str, str]] = frozenset()
    shadowed_names: frozenset[str] = frozenset()
    non_suppressor_instances: frozenset[str] = frozenset()
    suspicious_aliases: frozenset[str] = frozenset()
    cross_scope_mutated: frozenset[str] = frozenset()
    cross_scope_suppressors: frozenset[str] = frozenset()
    cross_scope_suspicious: frozenset[str] = frozenset()
    exit_patch_targets: frozenset[str] = frozenset()
    has_exit_patch: bool = False
    rebound_factory_targets: frozenset[Tuple[Optional[str], Optional[str]]] = frozenset()

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
            tuple(sorted(self.suspicious_aliases)),
            tuple(sorted(self.cross_scope_mutated)),
            tuple(sorted(self.cross_scope_suppressors)),
            tuple(sorted(self.cross_scope_suspicious)),
            tuple(sorted(self.exit_patch_targets)),
            self.has_exit_patch,
            tuple(
                sorted(
                    (t or "", a or "") for t, a in self.rebound_factory_targets
                )
            ),
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
            or self.suspicious_aliases
            or self.cross_scope_mutated
            or self.has_exit_patch
            or self.rebound_factory_targets
        )

    def module_aliases_for(self, mod_name: str) -> Set[str]:
        """Returns mod_name plus any local aliases bound to mod_name in stdlib_modules."""
        names: Set[str] = {mod_name}
        for alias, bound_mod in self.stdlib_modules:
            if bound_mod == mod_name:
                names.add(alias)
        return names

    def is_factory_rebound(
        self,
        target_names: AbstractSet[str],
        attr_name: str,
        *,
        include_untargeted: bool = True,
    ) -> bool:
        """Returns True if any matching module/alias factory attribute or untargeted dynamic patch was seen."""
        return is_factory_rebound_target(
            self.rebound_factory_targets,
            target_names,
            attr_name,
            include_untargeted=include_untargeted,
        )

    def clear_name(
        self,
        name: str,
        *,
        mark_shadow: bool = True,
        clear_cross_scope: bool = False,
    ) -> _SuppressProvenance:
        """Returns provenance with suppressor and non-suppressing bindings cleared for name."""
        if not name:
            return self
        new_sup = self.suppressors - {name}
        new_nonsup = self.non_suppressors - {name}
        new_nonsup_inst = self.non_suppressor_instances - {name}
        new_susp = self.suspicious_aliases - {name}
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
        new_cs_mut = (
            self.cross_scope_mutated - {name}
            if clear_cross_scope
            else self.cross_scope_mutated
        )
        new_cs_sup = (
            self.cross_scope_suppressors - {name}
            if clear_cross_scope
            else self.cross_scope_suppressors
        )
        new_cs_susp = (
            self.cross_scope_suspicious - {name}
            if clear_cross_scope
            else self.cross_scope_suspicious
        )
        if (
            new_sup == self.suppressors
            and new_nonsup == self.non_suppressors
            and new_nonsup_inst == self.non_suppressor_instances
            and new_susp == self.suspicious_aliases
            and new_mods == self.stdlib_modules
            and new_imps == self.imported_symbols
            and new_shadow == self.shadowed_builtins
            and new_shadow_names == self.shadowed_names
            and new_cs_mut == self.cross_scope_mutated
            and new_cs_sup == self.cross_scope_suppressors
            and new_cs_susp == self.cross_scope_suspicious
        ):
            return self
        return replace(
            self,
            suppressors=new_sup,
            non_suppressors=new_nonsup,
            stdlib_modules=new_mods,
            shadowed_builtins=new_shadow,
            imported_symbols=new_imps,
            shadowed_names=new_shadow_names,
            non_suppressor_instances=new_nonsup_inst,
            suspicious_aliases=new_susp,
            cross_scope_mutated=new_cs_mut,
            cross_scope_suppressors=new_cs_sup,
            cross_scope_suspicious=new_cs_susp,
        )

    def _bind_provenance_entry(
        self,
        name: str,
        *,
        allow_positive: bool = True,
        is_suppressor: bool = False,
        is_non_suppressor: bool = False,
        is_non_suppressor_instance: bool = False,
        is_suspicious_alias: bool = False,
        stdlib_mod: Optional[str] = None,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Clears prior bindings for name and optionally records a new provenance entry."""
        if self.has_exit_patch and not is_suppressor:
            if name in self.exit_patch_targets:
                is_suspicious_alias = True
            is_non_suppressor = False
            is_non_suppressor_instance = False
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
        if not (has_pos_binding or is_suppressor or is_suspicious_alias) or not name:
            return cleared
        return replace(
            cleared,
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
            suspicious_aliases=(
                cleared.suspicious_aliases | {name}
                if is_suspicious_alias
                else cleared.suspicious_aliases
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

    def with_suspicious_alias(self, name: str) -> _SuppressProvenance:
        """Returns provenance after binding name to an unproven suspicious-named manager."""
        return self._bind_provenance_entry(
            name,
            allow_positive=False,
            is_suspicious_alias=True,
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
        return replace(
            self,
            non_suppressors=self.non_suppressors - valid,
            stdlib_modules=frozenset(
                pair for pair in self.stdlib_modules if pair[0] not in valid
            ),
            shadowed_builtins=self.shadowed_builtins | valid,
            imported_symbols=frozenset(
                pair for pair in self.imported_symbols if pair[0] not in valid
            ),
            shadowed_names=self.shadowed_names | valid,
            non_suppressor_instances=self.non_suppressor_instances - valid,
        )

    def _positive_binding_names(self) -> frozenset[str]:
        """Returns all binding names currently carrying positive non-suppression proofs."""
        return (
            self.non_suppressors
            | self.non_suppressor_instances
            | frozenset(alias for alias, _ in self.stdlib_modules)
            | frozenset(alias for alias, _ in self.imported_symbols)
        )

    def with_star_import(self) -> _SuppressProvenance:
        """Withdraws prior positive non-suppression proofs that a wildcard import may overwrite."""
        prior_positive_names = self._positive_binding_names() - self.suppressors
        if self.has_star_import and not prior_positive_names:
            return self
        return replace(
            self,
            non_suppressors=frozenset(),
            non_suppressor_instances=frozenset(),
            stdlib_modules=frozenset(),
            imported_symbols=frozenset(),
            suspicious_aliases=self.suspicious_aliases | prior_positive_names,
            has_star_import=True,
        )

    def with_exit_protocol_patches(
        self,
        patched_names: AbstractSet[str],
        *,
        has_patch: bool,
        rebound_attrs: AbstractSet[Tuple[Optional[str], Optional[str]]] = frozenset(),
    ) -> _SuppressProvenance:
        """Withdraws patchable positive non-suppression proofs after runtime exit-protocol or factory patching.

        Clears every proven non-suppressing factory and instance binding (pure-Python
        classes, including stdlib managers such as ``contextlib.nullcontext``, can have
        ``__exit__`` reassigned after definition) and rebinds each name in
        ``patched_names`` that is not already a known suppressor as a suspicious alias.
        """
        if not has_patch and not rebound_attrs:
            return self
        merged_rebound = self.rebound_factory_targets | frozenset(rebound_attrs)
        if not has_patch:
            updated = replace(self, rebound_factory_targets=merged_rebound)
            invalidated = frozenset(
                n
                for n in (updated.non_suppressors | updated.non_suppressor_instances)
                if _has_rebound_imported_factory(n, updated)
            )
            if not invalidated:
                return updated
            return replace(
                updated,
                non_suppressors=updated.non_suppressors - invalidated,
                non_suppressor_instances=updated.non_suppressor_instances - invalidated,
            )
        demoted = replace(
            self,
            non_suppressors=frozenset(),
            non_suppressor_instances=frozenset(),
            exit_patch_targets=self.exit_patch_targets | frozenset(patched_names),
            has_exit_patch=True,
            rebound_factory_targets=merged_rebound,
        )
        for name in sorted(patched_names):
            if name not in demoted.suppressors:
                demoted = demoted.with_suspicious_alias(name)
        return demoted

    def with_cross_scope_mutations(
        self,
        mutated: AbstractSet[str],
        suppressor_targets: AbstractSet[str] = frozenset(),
        suspicious_targets: AbstractSet[str] = frozenset(),
    ) -> _SuppressProvenance:
        """Records global/nonlocal target mutations defined inside nested callables."""
        if not mutated:
            return self
        return replace(
            self,
            cross_scope_mutated=self.cross_scope_mutated | frozenset(mutated),
            cross_scope_suppressors=(
                self.cross_scope_suppressors | frozenset(suppressor_targets)
            ),
            cross_scope_suspicious=(
                self.cross_scope_suspicious | frozenset(suspicious_targets)
            ),
        )

    def invalidate_cross_scope_targets(
        self,
        mutated: AbstractSet[str],
        suppressor_targets: AbstractSet[str] = frozenset(),
        suspicious_targets: AbstractSet[str] = frozenset(),
    ) -> _SuppressProvenance:
        """Invalidates positive bindings and merges suppressor provenance for mutated names."""
        if not mutated:
            return self
        shad_b = frozenset(n for n in mutated if n in _SHADOWABLE_BUILTIN_CTX_NAMES)
        return replace(
            self,
            suppressors=self.suppressors | frozenset(suppressor_targets),
            non_suppressors=self.non_suppressors - mutated,
            stdlib_modules=frozenset(
                pair for pair in self.stdlib_modules if pair[0] not in mutated
            ),
            shadowed_builtins=self.shadowed_builtins | shad_b,
            imported_symbols=frozenset(
                pair for pair in self.imported_symbols if pair[0] not in mutated
            ),
            shadowed_names=self.shadowed_names | frozenset(mutated),
            non_suppressor_instances=self.non_suppressor_instances - mutated,
            suspicious_aliases=(
                (self.suspicious_aliases - frozenset(suppressor_targets))
                | frozenset(suspicious_targets)
            ),
        )

    def apply_cross_scope_invalidation(self) -> _SuppressProvenance:
        """Invalidates positive bindings for names mutable via global/nonlocal helpers."""
        return self.invalidate_cross_scope_targets(
            self.cross_scope_mutated,
            self.cross_scope_suppressors,
            self.cross_scope_suspicious,
        )

    def merge_branch_conservative(
        self, branch_prov: _SuppressProvenance
    ) -> _SuppressProvenance:
        """Merges provenance from a branch, keeping only surviving positive proofs and all hazards."""
        divergent_aliases = (
            frozenset(
                alias
                for alias, _ in (self.imported_symbols ^ branch_prov.imported_symbols)
            )
            | frozenset(
                alias
                for alias, _ in (self.stdlib_modules ^ branch_prov.stdlib_modules)
            )
            | (self.non_suppressors ^ branch_prov.non_suppressors)
            | (self.non_suppressor_instances ^ branch_prov.non_suppressor_instances)
        )
        divergent_builtins = divergent_aliases & _SHADOWABLE_BUILTIN_CTX_NAMES
        return _SuppressProvenance(
            suppressors=self.suppressors | branch_prov.suppressors,
            non_suppressors=self.non_suppressors & branch_prov.non_suppressors,
            stdlib_modules=self.stdlib_modules & branch_prov.stdlib_modules,
            shadowed_builtins=(
                self.shadowed_builtins
                | branch_prov.shadowed_builtins
                | divergent_builtins
            ),
            has_star_import=(self.has_star_import or branch_prov.has_star_import),
            imported_symbols=(self.imported_symbols & branch_prov.imported_symbols),
            shadowed_names=(
                self.shadowed_names | branch_prov.shadowed_names | divergent_aliases
            ),
            non_suppressor_instances=(
                self.non_suppressor_instances & branch_prov.non_suppressor_instances
            ),
            suspicious_aliases=(
                self.suspicious_aliases | branch_prov.suspicious_aliases
            ),
            cross_scope_mutated=(
                self.cross_scope_mutated | branch_prov.cross_scope_mutated
            ),
            cross_scope_suppressors=(
                self.cross_scope_suppressors | branch_prov.cross_scope_suppressors
            ),
            cross_scope_suspicious=(
                self.cross_scope_suspicious | branch_prov.cross_scope_suspicious
            ),
            exit_patch_targets=(
                self.exit_patch_targets | branch_prov.exit_patch_targets
            ),
            has_exit_patch=self.has_exit_patch or branch_prov.has_exit_patch,
            rebound_factory_targets=(
                self.rebound_factory_targets | branch_prov.rebound_factory_targets
            ),
        )

    def apply_branch_delta(
        self,
        shadowed_names: AbstractSet[str],
        shadowed_builtins: AbstractSet[str],
        new_suppressors: AbstractSet[str],
        has_star: bool,
        new_suspicious: AbstractSet[str] = frozenset(),
    ) -> _SuppressProvenance:
        """Applies a precomputed branch/suffix delta (shadowed names, suppressors, star import)."""
        if (
            not shadowed_names
            and not shadowed_builtins
            and not new_suppressors
            and not has_star
            and not new_suspicious
        ):
            return self
        merged_sup = self.suppressors | frozenset(new_suppressors)
        star_demoted = (
            self._positive_binding_names() - merged_sup if has_star else frozenset()
        )
        return replace(
            self,
            suppressors=merged_sup,
            non_suppressors=(
                frozenset() if has_star else self.non_suppressors - shadowed_names
            ),
            stdlib_modules=(
                frozenset()
                if has_star
                else frozenset(
                    pair for pair in self.stdlib_modules if pair[0] not in shadowed_names
                )
            ),
            shadowed_builtins=(self.shadowed_builtins | frozenset(shadowed_builtins)),
            has_star_import=self.has_star_import or has_star,
            imported_symbols=(
                frozenset()
                if has_star
                else frozenset(
                    pair
                    for pair in self.imported_symbols
                    if pair[0] not in shadowed_names
                )
            ),
            shadowed_names=self.shadowed_names | frozenset(shadowed_names),
            non_suppressor_instances=(
                frozenset()
                if has_star
                else self.non_suppressor_instances - shadowed_names
            ),
            suspicious_aliases=(
                self.suspicious_aliases | frozenset(new_suspicious) | star_demoted
            ),
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
        if dots > 0:
            stripped_pat = f"{'.'.join(parts[:split_idx])}.*.{attr}"
            if stripped_pat in extra_non_suppressing_ctx_managers:
                return stripped_pat
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
        or prov.has_star_import
    ):
        return _ExtraAllowlistMatch(False)
    candidate = ".".join((base_id, *tail_parts))
    if candidate in extra_non_suppressing_ctx_managers:
        return _ExtraAllowlistMatch(True)
    resolved_variants: List[Tuple[str, str]] = []
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
            ):
                return _ExtraAllowlistMatch(True)
            resolved_variants.append((resolved, last_part))
    for resolved, _ in resolved_variants:
        prefix_pat = _match_prefix_wildcard_pattern(
            resolved, extra_non_suppressing_ctx_managers
        )
        if prefix_pat is not None:
            return _ExtraAllowlistMatch(True, prefix_pat)
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
    if isinstance(curr_val, ast.Name):
        sym_match = _match_extra_non_suppressing_symbol(
            curr_val.id, chain_parts, prov, extra_non_suppressing_ctx_managers
        )
        if sym_match.matched:
            return sym_match
    if (
        not (isinstance(curr_val, ast.Name) and curr_val.id in prov.suppressors)
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


def _has_rebound_imported_factory(
    ident: str,
    prov: _SuppressProvenance,
    *,
    include_untargeted: bool = True,
) -> bool:
    """Returns True when ident aliases an imported symbol whose factory attribute was rebound."""
    for alias, qual in prov.imported_symbols:
        if alias == ident and "." in qual:
            imp_mod, imp_attr = qual.rsplit(".", 1)
            eff_untargeted = include_untargeted and qual not in (
                "builtins.open",
                "__builtins__.open",
                "io.open",
            )
            if prov.is_factory_rebound(
                prov.module_aliases_for(imp_mod),
                imp_attr,
                include_untargeted=eff_untargeted,
            ):
                return True
    return False


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
            if alias == fn_id and qual in (
                "builtins.open",
                "__builtins__.open",
                "io.open",
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
    {"locals", "vars", "exec", "eval", "setattr", "delattr", "globals"}
)
_DYNAMIC_METHOD_MUTATION_CALLS: frozenset[str] = frozenset(
    {"exec", "eval", "setattr", "delattr"}
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
    """Returns True if a ClassDef contains dynamic namespace reflection or runtime __exit__/__aexit__ mutation."""
    if scan_exit_protocol_patches(cls_node).has_patch:
        return True
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
            and node.attr in ("__exit__", "__aexit__", "__class__", "__bases__", "__dict__")
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
    if is_type_alias_node(curr):  # pragma: no cover (py312+)
        bound = _extract_type_alias_or_pattern_name(curr)
    else:
        bound = _extract_pattern_bound_name(curr)
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
            _record_import_names(curr, assigned)
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


class _FuncDefCtxKind(Enum):
    """Classification of context-manager factory behavior returned or yielded by a function definition."""

    NONE = "none"
    SUSPICIOUS = "suspicious"
    SUPPRESSOR = "suppressor"


def _classify_func_def_ctx_manager_kind(
    fn_stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _FuncDefCtxKind:
    """Classifies whether a function definition returns/yields a suppressor or suspicious manager."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _FuncDefCtxKind.SUPPRESSOR
    local_prov = prov
    for p_name in _extract_arg_names(fn_stmt.args):
        local_prov = local_prov.clear_name(p_name)
    for tp_name in _extract_node_type_param_names(fn_stmt):
        local_prov = local_prov.clear_name(tp_name)
    body_prov = _update_suppress_provenance_from_stmts(
        fn_stmt.body,
        local_prov,
        in_branch=True,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    fn_prov = local_prov.merge_branch_conservative(body_prov)

    found_suspicious = False
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
            return _FuncDefCtxKind.SUPPRESSOR
        if isinstance(curr, ast.Return) and curr.value is not None:
            val_stack: List[ast.AST] = [curr.value]
            while val_stack:
                v_curr = val_stack.pop()
                if _is_suppress_alias_target_expr(v_curr, fn_prov):
                    return _FuncDefCtxKind.SUPPRESSOR
                if isinstance(v_curr, _INNER_SCOPE_TYPES):
                    val_stack.extend(iter_nested_scope_header_nodes(v_curr))
                    continue
                val_stack.extend(ast.iter_child_nodes(v_curr))
            if (
                not found_suspicious
                and not _is_proven_non_suppressing_call(
                    curr.value,
                    fn_prov,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                and not _is_proven_non_suppressing_target_expr(
                    curr.value,
                    fn_prov,
                    extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                )
                and _looks_like_suppressing_ctx_manager(curr.value, fn_prov)
            ):
                found_suspicious = True
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
    return (
        _FuncDefCtxKind.SUSPICIOUS
        if found_suspicious
        else _FuncDefCtxKind.NONE
    )


def _func_def_may_return_suppressor(
    fn_stmt: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    prov: _SuppressProvenance,
    *,
    _depth: int = 0,
) -> bool:
    """Returns True if a function definition may return a suppressor or suppress exceptions around yield."""
    return (
        _classify_func_def_ctx_manager_kind(fn_stmt, prov, _depth=_depth)
        is _FuncDefCtxKind.SUPPRESSOR
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


def _bind_expr_target_provenance(
    prov: _SuppressProvenance,
    name: str,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    target_prov: Optional[_SuppressProvenance] = None,
) -> _SuppressProvenance:
    """Updates target_prov (defaulting to prov) when name is directly bound to value_expr evaluated in prov."""
    out_prov = prov if target_prov is None else target_prov
    while isinstance(value_expr, ast.NamedExpr):
        value_expr = value_expr.value
    target_src_expr = (
        value_expr.func if isinstance(value_expr, ast.Call) else value_expr
    )
    target_qual = (
        _resolve_tracked_factory_qualname(target_src_expr, prov)
        if allow_positive
        else None
    )
    if _is_direct_suppressor_expr(value_expr, prov):
        return out_prov.with_suppressor(
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
            return out_prov.with_non_suppressor_instance(
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
            return out_prov.with_non_suppressor(
                name, allow_positive=allow_positive, imported_qualname=open_qual
            )
        if allow_positive and _is_pathlib_path_expr(value_expr, prov):
            return out_prov.with_imported_symbol(
                name, "pathlib.Path", allow_positive=True
            )
        if _looks_like_suppressing_ctx_manager(value_expr, prov):
            return out_prov.with_suspicious_alias(name)
    return out_prov.clear_name(name)


def _bind_assign_target_provenance(
    rhs_prov: _SuppressProvenance,
    target: ast.AST,
    value_expr: Optional[ast.AST],
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    target_prov: _SuppressProvenance,
) -> _SuppressProvenance:
    """Binds assignment target provenance against value_expr, supporting nested and starred unpacking."""
    prov = target_prov
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(target, value_expr)]
    while stack:
        curr_target, curr_value = stack.pop()
        while isinstance(curr_value, ast.NamedExpr):
            curr_value = curr_value.value
        if isinstance(curr_target, ast.Name):
            prov = _bind_expr_target_provenance(
                rhs_prov,
                curr_target.id,
                curr_value,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                target_prov=prov,
            )
            continue
        matched_pairs = _match_sequence_unpack_pairs(
            curr_target, curr_value, allow_starred_target=True
        )
        if matched_pairs is not None:
            stack.extend(reversed(matched_pairs))
            continue
        is_susp_rhs = _unpack_fallback_rhs_is_suspicious(curr_value, rhs_prov)
        for t_name in sorted(_extract_definite_target_names(curr_target)):
            prov = (
                prov.with_suspicious_alias(t_name)
                if is_susp_rhs
                else prov.clear_name(t_name)
            )
    return prov


class _CrossScopeMutationAcc(NamedTuple):
    """Mutable sets accumulating global/nonlocal target names and provenance tags."""

    mutated: Set[str]
    sup_targets: Set[str]
    susp_targets: Set[str]


def _collect_scope_escaping_into(
    shallow_nodes: Sequence[ast.AST],
    anc_locals: AbstractSet[str],
    prov: _SuppressProvenance,
    acc: _CrossScopeMutationAcc,
) -> Tuple[Set[str], Set[str]]:
    """Collects global/nonlocal escaping mutations from shallow_nodes into acc, returning (globs, nonlocs)."""
    scope_globs: Set[str] = set()
    scope_nonlocs: Set[str] = set()
    for sub in shallow_nodes:
        if isinstance(sub, ast.Global):
            scope_globs.update(sub.names)
        elif isinstance(sub, ast.Nonlocal):
            scope_nonlocs.update(sub.names)
    escaping = scope_globs | (scope_nonlocs - anc_locals)
    if escaping:
        def _record_escaping_value(
            targets: AbstractSet[str], val_expr: ast.AST
        ) -> None:
            if not targets:
                return
            acc.mutated.update(targets)
            if _is_direct_suppressor_expr(val_expr, prov):
                acc.sup_targets.update(targets)
            elif _unpack_fallback_rhs_is_suspicious(val_expr, prov):
                acc.susp_targets.update(targets)

        for sub in shallow_nodes:
            if isinstance(sub, ast.Assign):
                _record_escaping_value(
                    {
                        n
                        for tgt in sub.targets
                        for n in _extract_definite_target_names(tgt)
                        if n in escaping
                    },
                    sub.value,
                )
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                _record_escaping_value(
                    _extract_definite_target_names(sub.target) & escaping,
                    sub.value,
                )
            elif (
                isinstance(sub, ast.NamedExpr)
                and isinstance(sub.target, ast.Name)
                and sub.target.id in escaping
            ):
                _record_escaping_value({sub.target.id}, sub.value)
            elif isinstance(sub, ast.Name) and isinstance(
                sub.ctx, (ast.Store, ast.Del)
            ):
                if sub.id in escaping:
                    acc.mutated.add(sub.id)
            elif isinstance(
                sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                if sub.name in escaping:
                    acc.mutated.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                imp_s: Set[str] = set()
                _record_import_names(sub, imp_s)
                acc.mutated.update(imp_s & escaping)
            elif isinstance(sub, ast.ExceptHandler) and isinstance(sub.name, str):
                if sub.name in escaping:
                    acc.mutated.add(sub.name)
            else:
                p_name = _extract_pattern_bound_name(sub)
                if p_name is not None and p_name in escaping:
                    acc.mutated.add(p_name)
    return scope_globs, scope_nonlocs


def _extract_nested_cross_scope_mutations(
    scope_stmt: ast.AST,
    prov: _SuppressProvenance,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extracts (mutated, suppressor_targets, suspicious_targets) for global/nonlocal stores in nested scopes."""
    acc = _CrossScopeMutationAcc(set(), set(), set())
    work: List[Tuple[ast.AST, frozenset[str]]] = [(scope_stmt, frozenset())]
    while work:
        curr_scope, anc_locals = work.pop()
        if not isinstance(
            curr_scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        shallow_nodes = tuple(_iter_shallow_scope_body_nodes(curr_scope))
        sc_globs, sc_nonlocs = _collect_scope_escaping_into(
            shallow_nodes, anc_locals, prov, acc
        )
        if isinstance(curr_scope, ast.ClassDef):
            next_anc = anc_locals
        else:
            own_locals = (
                (
                    _extract_arg_names(curr_scope.args)
                    | _extract_single_node_local_names(curr_scope)
                )
                - sc_globs
                - sc_nonlocs
            )
            next_anc = anc_locals | frozenset(own_locals)
        for sub in shallow_nodes:
            if isinstance(sub, _DEFERRED_DEF_SCOPE_TYPES) and not isinstance(
                sub, ast.Lambda
            ):
                work.append((sub, next_anc))
    return acc.mutated, acc.sup_targets, acc.susp_targets


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


class _ProvWalkResult(NamedTuple):
    """Single-pass provenance, intermediate-hazard provenance, and termination status."""

    prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    terminated: bool


def _update_prov_and_term_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Applies provenance and termination analysis across a statement sequence in one pass."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        out = _conservatively_update_prov_at_depth_limit(stmts, prov)
        return _ProvWalkResult(out, prov.merge_branch_conservative(out), False)
    cur_prov = prov
    mid_prov = prov
    for sub_stmt in stmts:
        step = _update_prov_and_term_from_stmt(
            sub_stmt,
            cur_prov,
            in_branch=in_branch,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        cur_prov = step.prov
        mid_prov = mid_prov.merge_branch_conservative(step.mid_prov)
        if step.terminated:
            return _ProvWalkResult(cur_prov, mid_prov, True)
    return _ProvWalkResult(cur_prov, mid_prov, False)


def _update_suppress_provenance_from_stmts(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Applies _update_suppress_provenance_from_stmt across a statement sequence."""
    return _update_prov_and_term_from_stmts(
        stmts,
        prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth,
    ).prov


def _advance_clause_entry_prov(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Advances provenance through a clause header expression and clears clause-bound names."""
    prov = _advance_prov_through_opt_expr(
        header_expr,
        prov,
        allow_positive=False,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    for b_name in bound_names:
        prov = prov.clear_name(b_name)
    return prov


def _merge_prior_clause_guard_and_binds(
    prov: _SuppressProvenance,
    prior_hdr: Any,
    prior_bound: Iterable[str],
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Conservatively merges effects of evaluating a prior failed clause's bindings and guard/type."""
    branch = prov
    for b_name in prior_bound:
        branch = branch.clear_name(b_name)
    branch = _advance_clause_entry_prov(
        branch,
        prior_hdr,
        (),
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    return prov if branch is prov else prov.merge_branch_conservative(branch)


def _update_clause_branch_provenance_with_term(
    prov: _SuppressProvenance,
    header_expr: Any,
    bound_names: Iterable[str],
    body: Iterable[ast.AST],
    *,
    in_branch: bool = True,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Updates provenance and termination status for an except handler or match case branch."""
    entry_prov = _advance_clause_entry_prov(
        prov,
        header_expr,
        bound_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    res = _update_prov_and_term_from_stmts(
        body,
        entry_prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    return _ProvWalkResult(
        res.prov, entry_prov.merge_branch_conservative(res.mid_prov), res.terminated
    )


def _advance_prov_through_exprs(
    exprs: Sequence[ast.AST],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance across shallow expression roots (walrus + cross-scope call check)."""
    if not exprs:
        return prov
    updated = _update_suppress_provenance_from_expr_walrus(
        exprs,
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    if _exprs_may_invoke_cross_scope_mutator(
        exprs, updated, extra_non_suppressing_ctx_managers
    ):
        return updated.apply_cross_scope_invalidation()
    return updated


def _advance_prov_through_opt_expr(
    expr: Any,
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> _SuppressProvenance:
    """Updates provenance across an optional header expression (walrus + cross-scope call check)."""
    if not isinstance(expr, ast.AST):
        return prov
    return _advance_prov_through_exprs(
        (expr,),
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )


def _advance_prov_through_for_header(
    stmt: Union[ast.For, ast.AsyncFor],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
) -> Tuple[_SuppressProvenance, _SuppressProvenance]:
    """Returns (post_iter_prov, post_target_prov) after evaluating a For/AsyncFor header."""
    post_iter = _advance_prov_through_opt_expr(
        stmt.iter,
        prov,
        allow_positive=allow_positive,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    post_target = _update_suppress_provenance_from_expr_walrus(
        (stmt.target,),
        post_iter,
        allow_positive=False,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )
    iter_is_proven_sup = _is_suppress_alias_target_expr(stmt.iter, post_iter)
    iter_is_suspicious = (
        not iter_is_proven_sup
        and _unpack_fallback_rhs_is_suspicious(stmt.iter, post_iter)
    )
    for t_name in sorted(_extract_definite_target_names(stmt.target)):
        if iter_is_proven_sup and isinstance(stmt.target, ast.Name):
            post_target = post_target.with_suppressor(t_name, allow_positive=True)
        elif iter_is_proven_sup or iter_is_suspicious:
            post_target = post_target.with_suspicious_alias(t_name)
        else:
            post_target = post_target.clear_name(t_name)
    return post_iter, post_target


def _update_import_stmt_provenance(
    stmt: Union[ast.Import, ast.ImportFrom],
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
) -> _SuppressProvenance:
    """Updates provenance for ast.Import and ast.ImportFrom statements."""
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
                    bound, alias.name, allow_positive=allow_positive
                )
            elif allow_positive and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES:
                prov = prov.with_imported_symbol(
                    bound, target_qual, allow_positive=allow_positive
                )
            else:
                prov = prov.clear_name(bound)
        return prov

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
                allow_positive=allow_positive,
                imported_qualname=imp_qual,
            )
        elif alias.name in _SUPPRESSING_CTX_MANAGER_NAMES:
            prov = prov.with_suppressor(
                bound,
                allow_positive=allow_positive,
                imported_qualname=imp_qual,
            )
        elif is_abs_mod and imp_qual in _STDLIB_NON_SUPPRESSING_MODULES:
            prov = prov.with_stdlib_module(
                bound,
                cast(str, imp_qual),
                allow_positive=allow_positive,
            )
        elif (
            imp_qual is not None
            and allow_positive
            and bound not in _SHADOWABLE_BUILTIN_CTX_NAMES
        ):
            prov = prov.with_imported_symbol(
                bound,
                imp_qual,
                allow_positive=allow_positive,
            )
        else:
            prov = prov.clear_name(bound)
    return prov


def _update_assign_or_delete_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    allow_positive: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
) -> _SuppressProvenance:
    """Updates provenance for Assign, AnnAssign, AugAssign, Delete, and TypeAlias statements."""
    if isinstance(stmt, ast.Assign):
        rhs_prov = _advance_prov_through_exprs(
            (stmt.value,),
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        prov = rhs_prov
        for tgt in stmt.targets:
            prov = _advance_prov_through_exprs(
                (tgt,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            prov = _bind_assign_target_provenance(
                rhs_prov,
                tgt,
                stmt.value,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                target_prov=prov,
            )
        return prov

    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        rhs_prov = (
            _advance_prov_through_exprs(
                (stmt.value,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            if isinstance(stmt.value, ast.AST)
            else prov
        )
        ann_expr = getattr(stmt, "annotation", None)
        prov = rhs_prov
        if isinstance(ann_expr, ast.AST):
            prov = _advance_prov_through_exprs(
                (ann_expr,),
                prov,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        if isinstance(stmt.target, ast.AST):
            prov = _advance_prov_through_exprs(
                (stmt.target,),
                prov,
                allow_positive=allow_positive,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        if stmt.value is not None:
            is_plain_ann = isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name
            )
            for t_name in _extract_definite_target_names(stmt.target):
                prov = (
                    _bind_expr_target_provenance(
                        rhs_prov,
                        t_name,
                        stmt.value,
                        allow_positive=allow_positive and isinstance(stmt, ast.AnnAssign),
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                        target_prov=prov,
                    )
                    if is_plain_ann
                    else prov.clear_name(t_name)
                )
        return prov

    if isinstance(stmt, ast.Delete):
        prov = _advance_prov_through_exprs(
            stmt.targets,
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        for d_name in _extract_shallow_deleted_names(stmt):
            prov = prov.clear_name(d_name)
        return prov

    ta_val = getattr(stmt, "value", None)
    if isinstance(ta_val, ast.AST):
        prov = _update_suppress_provenance_from_expr_walrus(
            (ta_val,),
            prov,
            allow_positive=allow_positive,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    ta_name = _extract_type_alias_or_pattern_name(stmt)
    return prov.clear_name(ta_name) if ta_name is not None else prov


class _ProvenanceWalkContext(NamedTuple):
    """Policy and recursion depth context passed across compound provenance updaters."""

    in_branch: bool = False
    strict_with_suppression: bool = False
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset()
    depth: int = 0


def _walk_child_stmts_prov(
    stmts: Iterable[ast.AST],
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
    *,
    in_branch: Optional[bool] = None,
) -> _ProvWalkResult:
    """Steps into child statement sequences at ctx.depth + 1 using ctx policy flags."""
    return _update_prov_and_term_from_stmts(
        stmts,
        prov,
        in_branch=ctx.in_branch if in_branch is None else in_branch,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        _depth=ctx.depth + 1,
    )


class _TryBodySplitProvenance(NamedTuple):
    """Remaining suffix, provenance states, and termination status across a split try body."""

    risky_suffix: List[ast.stmt]
    prefix_prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    try_prov: _SuppressProvenance
    terminated: bool


def _compute_try_body_split_provenance(
    t_body: Sequence[ast.stmt],
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
    *,
    safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
) -> _TryBodySplitProvenance:
    """Split a try body at its first potentially raising statement and advance provenance in one pass."""
    safe_prefix, risky_suffix = _split_suppress_with_body(
        t_body,
        safe_names=safe_names,
        strict_names=strict_names,
        suppress_aliases=prov,
        _depth=ctx.depth + 1,
    )
    pre_res = _walk_child_stmts_prov(safe_prefix, prov, ctx)
    if pre_res.terminated or not risky_suffix:
        return _TryBodySplitProvenance(
            [], pre_res.prov, pre_res.prov, pre_res.prov, pre_res.terminated
        )
    suf_res = _walk_child_stmts_prov(risky_suffix, pre_res.prov, ctx)
    return _TryBodySplitProvenance(
        risky_suffix,
        pre_res.prov,
        suf_res.mid_prov,
        suf_res.prov,
        suf_res.terminated,
    )


def _apply_immediate_class_body_provenance(
    cls_stmt: ast.ClassDef,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Applies immediate global/nonlocal rebinds and cross-scope call invalidations for ClassDef bodies."""
    imm_acc = _CrossScopeMutationAcc(set(), set(), set())
    has_unproven_call = False
    cls_work: List[ast.ClassDef] = [cls_stmt]
    while cls_work:
        curr_cls = cls_work.pop()
        shallow_nodes = tuple(_iter_shallow_scope_body_nodes(curr_cls))
        _collect_scope_escaping_into(shallow_nodes, frozenset(), prov, imm_acc)
        if prov.cross_scope_mutated and not has_unproven_call:
            has_unproven_call = _has_shallow_unproven_call_or_await(
                shallow_nodes,
                prov,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
        for sub in shallow_nodes:
            if isinstance(sub, ast.ClassDef):
                cls_work.append(sub)
    if imm_acc.mutated:
        prov = prov.invalidate_cross_scope_targets(
            imm_acc.mutated, imm_acc.sup_targets, imm_acc.susp_targets
        )
    if has_unproven_call:
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _apply_class_def_external_effects(
    cls_stmt: ast.ClassDef,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Applies outer header, cross-scope, and global/nonlocal effects of ClassDef without leaking class locals."""
    allow_pos = not ctx.in_branch
    hdr_nodes = tuple(iter_nested_scope_header_nodes(cls_stmt))
    prov = _advance_prov_through_exprs(
        hdr_nodes,
        prov,
        allow_positive=allow_pos,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    cs_mut, cs_sup, cs_susp = _extract_nested_cross_scope_mutations(cls_stmt, prov)
    prov = prov.with_cross_scope_mutations(cs_mut, cs_sup, cs_susp)
    prov = _apply_immediate_class_body_provenance(cls_stmt, prov, ctx)
    if ctx.in_branch and cs_mut:
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _update_scope_def_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _SuppressProvenance:
    """Updates provenance from a FunctionDef, AsyncFunctionDef, or ClassDef statement."""
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return prov
    allow_pos = not ctx.in_branch
    if isinstance(stmt, ast.ClassDef):
        prov = _apply_class_def_external_effects(stmt, prov, ctx)
    else:
        hdr_nodes = tuple(iter_nested_scope_header_nodes(stmt))
        prov = _advance_prov_through_exprs(
            hdr_nodes,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        cs_mut, cs_sup, cs_susp = _extract_nested_cross_scope_mutations(stmt, prov)
        prov = prov.with_cross_scope_mutations(cs_mut, cs_sup, cs_susp)
        if ctx.in_branch and cs_mut:
            prov = prov.apply_cross_scope_invalidation()
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn_kind = _classify_func_def_ctx_manager_kind(
            stmt,
            prov,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            _depth=ctx.depth + 1,
        )
        if fn_kind is _FuncDefCtxKind.SUPPRESSOR:
            return prov.with_suppressor(stmt.name)
        if fn_kind is _FuncDefCtxKind.SUSPICIOUS:
            return prov.with_suspicious_alias(stmt.name)
    elif _class_def_may_suppress_in_exit(stmt, prov):
        return prov.with_suppressor(stmt.name)
    elif (
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


def _update_scope_def_stmt_prov_with_term(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Wraps _update_scope_def_stmt_provenance into a non-terminating _ProvWalkResult."""
    out = _update_scope_def_stmt_provenance(stmt, prov, ctx)
    return _ProvWalkResult(out, out, False)


def _update_loop_or_if_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination status across If, While, For, and AsyncFor statements."""
    if not isinstance(stmt, (ast.If, ast.While, ast.For, ast.AsyncFor)):
        return _ProvWalkResult(prov, prov, False)
    allow_pos = not ctx.in_branch
    if isinstance(stmt, (ast.If, ast.While)):
        is_while = isinstance(stmt, ast.While)
        prov = _advance_prov_through_opt_expr(
            stmt.test,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        branch_flag = True if is_while else ctx.in_branch
        b_res = _walk_child_stmts_prov(stmt.body, prov, ctx, in_branch=branch_flag)
        else_base = b_res.prov if is_while else prov
        o_res = (
            _walk_child_stmts_prov(stmt.orelse, else_base, ctx, in_branch=branch_flag)
            if stmt.orelse
            else _ProvWalkResult(else_base, else_base, False)
        )
        merged = b_res.prov.merge_branch_conservative(o_res.prov)
        out_prov = prov.merge_branch_conservative(merged) if is_while else merged
        out_mid = (
            prov.merge_branch_conservative(b_res.mid_prov)
            .merge_branch_conservative(o_res.mid_prov)
        )
        out_term = (
            (
                bool(stmt.orelse)
                and o_res.terminated
                and not _scan_stmts_shallow_exits(stmt.body)[1]
            )
            if is_while
            else (b_res.terminated and o_res.terminated)
        )
        return _ProvWalkResult(out_prov, out_mid, out_term)

    prov, iter_prov = _advance_prov_through_for_header(
        stmt,
        prov,
        allow_positive=allow_pos,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    b_res = _walk_child_stmts_prov(stmt.body, iter_prov, ctx, in_branch=True)
    o_base = prov.merge_branch_conservative(b_res.prov)
    o_res = (
        _walk_child_stmts_prov(stmt.orelse, o_base, ctx, in_branch=True)
        if stmt.orelse
        else _ProvWalkResult(b_res.prov, b_res.mid_prov, False)
    )
    out_prov = o_base.merge_branch_conservative(o_res.prov)
    out_mid = (
        prov.merge_branch_conservative(b_res.mid_prov)
        .merge_branch_conservative(o_res.mid_prov)
    )
    out_term = (
        bool(stmt.orelse)
        and o_res.terminated
        and not _scan_stmts_shallow_exits(stmt.body)[1]
    )
    return _ProvWalkResult(out_prov, out_mid, out_term)


def _update_with_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across a With/AsyncWith header and body prefix/suffix."""
    if not isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _ProvWalkResult(prov, prov, False)
    allow_pos = not ctx.in_branch
    header_prov = prov
    item_prov = prov
    after_sup = False
    header_risky = False
    for step in _iter_with_item_provenance_steps(
        stmt,
        prov,
        allow_positive=allow_pos,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        strict_with_suppression=ctx.strict_with_suppression,
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
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        == "non_suppressing"
    ):
        b_res = _walk_child_stmts_prov(stmt.body, item_prov, ctx)
        return _ProvWalkResult(
            b_res.prov,
            item_prov.merge_branch_conservative(b_res.mid_prov),
            b_res.terminated,
        )
    safe_prefix, risky_suffix = (
        ([], list(stmt.body))
        if header_risky
        else _split_suppress_with_body(
            stmt.body, suppress_aliases=item_prov, _depth=ctx.depth + 1
        )
    )
    pre_res = _walk_child_stmts_prov(safe_prefix, item_prov, ctx)
    pre_mid = item_prov.merge_branch_conservative(pre_res.mid_prov)
    if pre_res.terminated or not risky_suffix:
        return _ProvWalkResult(pre_res.prov, pre_mid, pre_res.terminated)
    # Statements in risky_suffix execute after a potentially raising expression inside a
    # potentially exception-suppressing with-block; if an exception is raised and suppressed,
    # the remainder of risky_suffix is skipped, so bindings in risky_suffix are conditional
    # (in_branch=True) relative to code following the with-block.
    suf_res = _walk_child_stmts_prov(risky_suffix, pre_res.prov, ctx, in_branch=True)
    return _ProvWalkResult(
        pre_res.prov.merge_branch_conservative(suf_res.prov),
        pre_mid.merge_branch_conservative(suf_res.mid_prov),
        False,
    )


def _scan_stmts_shallow_exits(stmts: Sequence[ast.AST]) -> Tuple[bool, bool, bool]:
    """Returns (has_return, has_unescaped_break, has_unescaped_continue) outside nested scopes."""
    has_ret = False
    has_brk = False
    has_cont = False
    stack: List[Tuple[ast.AST, int]] = [(s, 0) for s in stmts]
    while stack:
        curr, loop_depth = stack.pop()
        if isinstance(
            curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(curr, ast.Return):
            has_ret = True
        elif loop_depth == 0:
            if isinstance(curr, ast.Break):
                has_brk = True
            elif isinstance(curr, ast.Continue):
                has_cont = True
        if has_ret and has_brk and has_cont:
            break
        next_depth = (
            loop_depth + 1
            if isinstance(curr, (ast.For, ast.AsyncFor, ast.While))
            else loop_depth
        )
        for child in ast.iter_child_nodes(curr):
            stack.append((child, next_depth))
    return has_ret, has_brk, has_cont


def _finally_may_suppress_active_exception(f_body: Sequence[ast.AST]) -> bool:
    """Returns True if f_body contains return/break/continue targeting an enclosing scope."""
    has_ret, has_brk, has_cont = _scan_stmts_shallow_exits(f_body)
    return has_ret or has_brk or has_cont


def _update_try_finally_provenance(
    f_body: Sequence[ast.AST],
    merged_try: _SuppressProvenance,
    handler_base: _SuppressProvenance,
    *,
    has_abrupt_entry: bool,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Evaluates finally-body provenance and termination including unhandled-exception entry."""
    f_entry_prov = (
        merged_try.merge_branch_conservative(handler_base)
        if has_abrupt_entry
        else merged_try
    )
    walk_entry = (
        f_entry_prov
        if (not has_abrupt_entry or _finally_may_suppress_active_exception(f_body))
        else merged_try
    )
    return _walk_child_stmts_prov(f_body, walk_entry, ctx)


class _TryHandlersProvenance(NamedTuple):
    """Merged fallthrough provenance, handler scan state, mid-hazards, and termination flag."""

    post_try_prov: _SuppressProvenance
    handler_scan_prov: _SuppressProvenance
    mid_prov: _SuppressProvenance
    all_handlers_terminated: bool


def _update_try_handlers_provenance(
    stmt: ast.AST,
    handler_base: _SuppressProvenance,
    succ_prov: _SuppressProvenance,
    succ_term: bool,
    ctx: _ProvenanceWalkContext,
) -> _TryHandlersProvenance:
    """Evaluates except handlers in one pass, omitting terminating paths from fallthrough provenance."""
    handlers = getattr(stmt, "handlers", ())
    is_trystar = is_trystar_node(stmt)
    final_prov = handler_base.merge_branch_conservative(succ_prov)
    mid_prov = final_prov
    all_handlers_terminated = True
    surviving: List[_SuppressProvenance] = (
        [succ_prov] if (not succ_term and not (is_trystar and handlers)) else []
    )
    scan_prov = handler_base
    for h_node in handlers:
        h_name = getattr(h_node, "name", None)
        h_type = getattr(h_node, "type", None)
        h_entry = _advance_clause_entry_prov(
            scan_prov,
            h_type,
            (h_name,) if isinstance(h_name, str) else (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        h_body = getattr(h_node, "body", ())
        h_res = _walk_child_stmts_prov(h_body, h_entry, ctx)
        h_exit_prov = (
            h_res.prov.clear_name(h_name) if isinstance(h_name, str) else h_res.prov
        )
        final_prov = final_prov.merge_branch_conservative(h_exit_prov)
        mid_prov = mid_prov.merge_branch_conservative(
            h_res.mid_prov
        ).merge_branch_conservative(h_exit_prov)
        if not h_res.terminated:
            all_handlers_terminated = False
            if not is_trystar:
                surviving.append(h_exit_prov)
        scan_prov = _merge_prior_clause_guard_and_binds(
            scan_prov,
            h_type,
            (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
    post_prov = final_prov
    if surviving:
        post_prov = surviving[0]
        for sp in surviving[1:]:
            post_prov = post_prov.merge_branch_conservative(sp)
    return _TryHandlersProvenance(
        post_prov, scan_prov, mid_prov, all_handlers_terminated
    )


def _update_try_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across Try/TryStar body, handlers, orelse, and finalbody in one pass."""
    t_body = getattr(stmt, "body", ())
    split = _compute_try_body_split_provenance(t_body, prov, ctx)
    try_prov = split.try_prov
    handler_base = split.prefix_prov.merge_branch_conservative(
        split.mid_prov
    ).merge_branch_conservative(try_prov)
    t_term = split.terminated
    o_stmts = getattr(stmt, "orelse", ())
    if o_stmts and not t_term:
        o_res = _walk_child_stmts_prov(o_stmts, try_prov, ctx)
        succ_prov, succ_mid, succ_term = o_res.prov, o_res.mid_prov, o_res.terminated
    else:
        succ_prov, succ_mid, succ_term = try_prov, split.mid_prov, t_term
    h_res = _update_try_handlers_provenance(
        stmt, handler_base, succ_prov, succ_term, ctx
    )
    pre_finally_term = succ_term and h_res.all_handlers_terminated
    try_mid = (
        prov.merge_branch_conservative(handler_base)
        .merge_branch_conservative(succ_mid)
        .merge_branch_conservative(h_res.mid_prov)
    )
    f_body = getattr(stmt, "finalbody", ())
    if not f_body:
        return _ProvWalkResult(h_res.post_try_prov, try_mid, pre_finally_term)
    o_has_risky = bool(
        o_stmts
        and not t_term
        and _split_suppress_with_body(
            o_stmts, suppress_aliases=try_prov, _depth=ctx.depth + 1
        )[1]
    )
    f_res = _update_try_finally_provenance(
        f_body,
        h_res.post_try_prov,
        h_res.handler_scan_prov.merge_branch_conservative(h_res.mid_prov),
        has_abrupt_entry=bool(split.risky_suffix or o_has_risky),
        ctx=ctx,
    )
    return _ProvWalkResult(
        f_res.prov,
        try_mid.merge_branch_conservative(f_res.mid_prov),
        f_res.terminated or pre_finally_term,
    )


def _update_match_stmt_provenance(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    ctx: _ProvenanceWalkContext,
) -> _ProvWalkResult:
    """Updates provenance and termination across a Python 3.10+ Match statement."""
    prov = _advance_prov_through_opt_expr(
        getattr(stmt, "subject", None),
        prov,
        allow_positive=not ctx.in_branch,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
    )
    cases = getattr(stmt, "cases", ())
    if not cases:
        return _ProvWalkResult(prov, prov, False)
    case_provs: List[_SuppressProvenance] = []
    mid_prov = prov
    match_scan_prov = prov
    has_irrefutable = False
    all_cases_terminated = True
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
        guard = getattr(case, "guard", None)
        c_res = _update_clause_branch_provenance_with_term(
            match_scan_prov,
            guard,
            pat_names,
            getattr(case, "body", ()),
            strict_with_suppression=ctx.strict_with_suppression,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            _depth=ctx.depth + 1,
        )
        case_provs.append(c_res.prov)
        mid_prov = mid_prov.merge_branch_conservative(c_res.mid_prov)
        if not c_res.terminated:
            all_cases_terminated = False
        if _is_irrefutable_case(case):
            has_irrefutable = True
            break
        match_scan_prov = _merge_prior_clause_guard_and_binds(
            match_scan_prov,
            guard,
            pat_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
    merged_match = case_provs[0]
    for cp in case_provs[1:]:
        merged_match = merged_match.merge_branch_conservative(cp)
    if not has_irrefutable:
        merged_match = match_scan_prov.merge_branch_conservative(merged_match)
    return _ProvWalkResult(
        merged_match, mid_prov, has_irrefutable and all_cases_terminated
    )


_COMPOUND_STMT_PROVENANCE_HANDLERS: Tuple[
    Tuple[
        Callable[[ast.AST], bool],
        Callable[
            [ast.AST, _SuppressProvenance, _ProvenanceWalkContext],
            _ProvWalkResult,
        ],
    ],
    ...,
] = (
    (
        lambda s: isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)),
        _update_scope_def_stmt_prov_with_term,
    ),
    (
        lambda s: isinstance(s, (ast.If, ast.While, ast.For, ast.AsyncFor)),
        _update_loop_or_if_stmt_provenance,
    ),
    (lambda s: isinstance(s, (ast.With, ast.AsyncWith)), _update_with_stmt_provenance),
    (is_try_or_trystar_node, _update_try_stmt_provenance),
    (is_match_node, _update_match_stmt_provenance),
)


def _update_prov_and_term_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _ProvWalkResult:
    """Returns updated provenance, intermediate-hazard provenance, and termination for stmt."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        out = _conservatively_update_prov_at_depth_limit((stmt,), prov)
        return _ProvWalkResult(out, prov.merge_branch_conservative(out), False)
    allow_pos = not in_branch
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        out = _update_import_stmt_provenance(stmt, prov, allow_positive=allow_pos)
        return _ProvWalkResult(out, out, False)
    if isinstance(
        stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)
    ) or is_type_alias_node(stmt):
        out = _update_assign_or_delete_stmt_provenance(
            stmt,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        return _ProvWalkResult(out, out, False)
    walk_ctx = _ProvenanceWalkContext(
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        depth=_depth,
    )
    for matcher, handler in _COMPOUND_STMT_PROVENANCE_HANDLERS:
        if matcher(stmt):
            return handler(stmt, prov, walk_ctx)
    if not isinstance(stmt, _INNER_SCOPE_TYPES):
        child_exprs = tuple(
            ch for ch in ast.iter_child_nodes(stmt) if not isinstance(ch, ast.stmt)
        )
        out = _advance_prov_through_exprs(
            child_exprs,
            prov,
            allow_positive=allow_pos,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        return _ProvWalkResult(out, out, isinstance(stmt, (ast.Return, ast.Raise)))
    return _ProvWalkResult(prov, prov, False)


def _update_suppress_provenance_from_stmt(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    *,
    in_branch: bool = False,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Returns updated _SuppressProvenance from a statement, recursing into compound blocks."""
    return _update_prov_and_term_from_stmt(
        stmt,
        prov,
        in_branch=in_branch,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth,
    ).prov


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
            for attr_name in ("name", "rest"):
                bound_val = getattr(node, attr_name, None)
                if isinstance(bound_val, str) and bound_val:
                    names.add(bound_val)
    return names


class _ModuleStmtDelta(NamedTuple):
    """Per-statement forward and branch provenance delta for linear-time module seeding."""

    stmt_id: int
    fwd_add_sup: frozenset[str]
    fwd_rem_sup: frozenset[str]
    fwd_add_susp: frozenset[str]
    fwd_rem_susp: frozenset[str]
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
    fwd_cs_mut: frozenset[str]
    fwd_cs_sup: frozenset[str]
    fwd_cs_susp: frozenset[str]
    branch_new_sup: frozenset[str]
    branch_new_susp: frozenset[str]
    branch_shad_b: frozenset[str]
    branch_shad_n: frozenset[str]
    branch_star: bool


def _with_patch_scan_state(
    prov: _SuppressProvenance, patch_scan: ExitProtocolPatchScan
) -> _SuppressProvenance:
    """Attaches module-wide exit-protocol patch state without mutating existing bindings."""
    if not patch_scan.has_patch and not patch_scan.rebound_attrs:
        return prov
    return replace(
        prov,
        exit_patch_targets=prov.exit_patch_targets | patch_scan.patched_names,
        has_exit_patch=prov.has_exit_patch or patch_scan.has_patch,
        rebound_factory_targets=(
            prov.rebound_factory_targets | patch_scan.rebound_attrs
        ),
    )


def _apply_exit_protocol_patch_scan(
    prov: _SuppressProvenance, patch_scan: ExitProtocolPatchScan
) -> _SuppressProvenance:
    """Applies exit-protocol and factory-rebind invalidations from patch_scan onto prov."""
    return prov.with_exit_protocol_patches(
        patch_scan.patched_names,
        has_patch=patch_scan.has_patch,
        rebound_attrs=patch_scan.rebound_attrs,
    )


class _LazyModuleStmtSeedMap(Mapping[int, _SuppressProvenance]):
    """On-demand mapping from top-level statement id() to its prefix, suffix, or combined provenance."""

    __slots__ = (
        "_deltas",
        "_idx_by_id",
        "_kind",
        "_patch_scan",
        "_cache",
        "_prefix_table",
        "_suffix_table",
        "_lock",
    )

    def __init__(
        self,
        deltas: Tuple[_ModuleStmtDelta, ...],
        idx_by_id: Mapping[int, int],
        kind: str,
        patch_scan: ExitProtocolPatchScan = _NO_PATCH_SCAN,
    ) -> None:
        self._deltas = deltas
        self._idx_by_id = idx_by_id
        self._kind = kind
        self._patch_scan = patch_scan
        self._cache: dict[int, _SuppressProvenance] = {}
        self._prefix_table: Optional[Tuple[_SuppressProvenance, ...]] = None
        self._suffix_table: Optional[
            Tuple[
                Tuple[
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                    bool,
                ],
                ...,
            ]
        ] = None
        self._lock: threading.RLock = threading.RLock()

    def __len__(self) -> int:
        return len(self._idx_by_id)

    def __iter__(self) -> Iterator[int]:
        return iter(self._idx_by_id)

    def __contains__(self, key: object) -> bool:
        return key in self._idx_by_id

    def _ensure_tables(self) -> None:
        with self._lock:
            deltas = self._deltas
            n = len(deltas)
            if self._kind in ("prefix", "combined") and self._prefix_table is None:
                p_sup: Set[str] = set()
                p_susp: Set[str] = set()
                p_nonsup: Set[str] = set()
                p_nonsup_inst: Set[str] = set()
                p_mods: Set[Tuple[str, str]] = set()
                p_imps: Set[Tuple[str, str]] = set()
                p_shad_b: Set[str] = set()
                p_shad_n: Set[str] = set()
                p_cs_mut: Set[str] = set()
                p_cs_sup: Set[str] = set()
                p_cs_susp: Set[str] = set()
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
                            frozenset(p_susp),
                            frozenset(p_cs_mut),
                            frozenset(p_cs_sup),
                            frozenset(p_cs_susp),
                            self._patch_scan.patched_names,
                            self._patch_scan.has_patch,
                            self._patch_scan.rebound_attrs,
                        )
                    )
                    d = deltas[idx]
                    for target_set, rem_items, add_items in (
                        (p_sup, d.fwd_rem_sup, d.fwd_add_sup),
                        (p_susp, d.fwd_rem_susp, d.fwd_add_susp),
                        (p_nonsup, d.fwd_rem_nonsup, d.fwd_add_nonsup),
                        (p_nonsup_inst, d.fwd_rem_nonsup_inst, d.fwd_add_nonsup_inst),
                        (p_shad_b, d.fwd_rem_shad_b, d.fwd_add_shad_b),
                        (p_shad_n, d.fwd_rem_shad_n, d.fwd_add_shad_n),
                    ):
                        if rem_items:
                            target_set.difference_update(rem_items)
                        if add_items:
                            target_set.update(add_items)
                    if d.fwd_rem_mods:
                        p_mods.difference_update(d.fwd_rem_mods)
                    if d.fwd_add_mods:
                        p_mods.update(d.fwd_add_mods)
                    if d.fwd_rem_imps:
                        p_imps.difference_update(d.fwd_rem_imps)
                    if d.fwd_add_imps:
                        p_imps.update(d.fwd_add_imps)
                    if d.fwd_cs_mut:
                        p_cs_mut.update(d.fwd_cs_mut)
                    if d.fwd_cs_sup:
                        p_cs_sup.update(d.fwd_cs_sup)
                    if d.fwd_cs_susp:
                        p_cs_susp.update(d.fwd_cs_susp)
                    if d.fwd_star:
                        p_star = True
                self._prefix_table = tuple(pref_list)

            if self._kind in ("suffix", "combined") and self._suffix_table is None:
                s_sup: Set[str] = set()
                s_susp: Set[str] = set()
                s_shad_b: Set[str] = set()
                s_shad_n: Set[str] = set()
                s_star = False
                empty_entry: Tuple[
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                    frozenset[str],
                    bool,
                ] = (
                    frozenset(),
                    frozenset(),
                    frozenset(),
                    frozenset(),
                    False,
                )
                suff_list: List[
                    Tuple[
                        frozenset[str],
                        frozenset[str],
                        frozenset[str],
                        frozenset[str],
                        bool,
                    ]
                ] = [empty_entry] * n
                for idx in range(n - 1, -1, -1):
                    suff_list[idx] = (
                        frozenset(s_sup),
                        frozenset(s_susp),
                        frozenset(s_shad_b),
                        frozenset(s_shad_n),
                        s_star,
                    )
                    d = deltas[idx]
                    if d.branch_new_sup:
                        s_sup.update(d.branch_new_sup)
                    if d.branch_new_susp:
                        s_susp.update(d.branch_new_susp)
                    if d.branch_shad_b:
                        s_shad_b.update(d.branch_shad_b)
                    if d.branch_shad_n:
                        s_shad_n.update(d.branch_shad_n)
                    if d.branch_star:
                        s_star = True
                self._suffix_table = tuple(suff_list)

    def __getitem__(self, stmt_id: int) -> _SuppressProvenance:
        target_idx = self._idx_by_id[stmt_id]
        with self._lock:
            cached = self._cache.get(stmt_id)
            if cached is not None:
                return cached
            self._ensure_tables()
            if self._kind == "prefix":
                if self._prefix_table is None:
                    raise RuntimeError("prefix provenance table not initialized")
                res = _apply_exit_protocol_patch_scan(
                    self._prefix_table[target_idx], self._patch_scan
                )
                self._cache[stmt_id] = res
                return res

            if self._suffix_table is None:
                raise RuntimeError("suffix provenance table not initialized")
            s_sup, s_susp, s_shad_b, s_shad_n, s_star = self._suffix_table[target_idx]
            if self._kind == "suffix":
                res = _SuppressProvenance(
                    suppressors=s_sup,
                    shadowed_builtins=s_shad_b,
                    has_star_import=s_star,
                    shadowed_names=s_shad_n,
                    suspicious_aliases=s_susp,
                )
                self._cache[stmt_id] = res
                return res

            if self._prefix_table is None:
                raise RuntimeError("prefix provenance table not initialized")
            pref = self._prefix_table[target_idx]
            res = _apply_exit_protocol_patch_scan(
                pref.apply_branch_delta(
                    s_shad_n,
                    s_shad_b,
                    s_sup,
                    s_star,
                    new_suspicious=s_susp,
                ),
                self._patch_scan,
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
    module_cross_scope_mutated: frozenset[str] = frozenset()
    module_cross_scope_suppressors: frozenset[str] = frozenset()
    module_cross_scope_suspicious: frozenset[str] = frozenset()

    def cached_ast_nodes(self) -> Tuple[ast.AST, ...]:
        """Returns AST nodes retained by this cached value (none)."""
        return ()


def _build_module_suppress_seed(
    eff_mod: ast.Module,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _ModuleSuppressSeed:
    """Builds a reusable module-level suppression provenance summary in O(N) time."""
    patch_scan = scan_exit_protocol_patches(eff_mod)
    body = eff_mod.body
    deltas: List[_ModuleStmtDelta] = []
    idx_by_id: dict[int, int] = {}
    cur_sup: Set[str] = set()
    cur_susp: Set[str] = set()
    cur_all_sup: Set[str] = set()
    cur_all_susp: Set[str] = set()
    cur_nonsup: Set[str] = set()
    cur_nonsup_inst: Set[str] = set()
    cur_mods: dict[str, str] = {}
    cur_imps: dict[str, str] = {}
    cur_shad_b: Set[str] = set()
    cur_shad_n: Set[str] = set()
    cur_cs_mut: Set[str] = set()
    cur_cs_sup: Set[str] = set()
    cur_cs_susp: Set[str] = set()
    all_cs_mut: Set[str] = set()
    all_cs_sup: Set[str] = set()
    all_cs_susp: Set[str] = set()
    cur_star = False
    has_pos_or_sup = False

    for idx, m_stmt in enumerate(body):
        stmt_id = id(m_stmt)
        idx_by_id[stmt_id] = idx
        stmt_names = _extract_stmt_referenced_and_bound_names(m_stmt)
        local_all_sup = frozenset(n for n in stmt_names if n in cur_all_sup)
        local_all_susp = frozenset(n for n in stmt_names if n in cur_all_susp)
        seed_in = _with_patch_scan_state(
            _SuppressProvenance(
                suppressors=local_all_sup, suspicious_aliases=local_all_susp
            )
            if (local_all_sup or local_all_susp)
            else _EMPTY_SUPPRESS_PROVENANCE,
            patch_scan,
        )
        b_delta = _update_suppress_provenance_from_stmt(
            m_stmt,
            seed_in,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        rebound_existing = (local_all_sup | local_all_susp) & b_delta.shadowed_names
        if rebound_existing:
            b_without_rebound = _update_suppress_provenance_from_stmt(
                m_stmt,
                _with_patch_scan_state(
                    _SuppressProvenance(
                        suppressors=local_all_sup - rebound_existing,
                        suspicious_aliases=local_all_susp - rebound_existing,
                    ),
                    patch_scan,
                ),
                in_branch=True,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            new_sup = (b_delta.suppressors - local_all_sup) | (
                b_without_rebound.suppressors & rebound_existing
            )
            new_susp = (b_delta.suspicious_aliases - local_all_susp) | (
                b_without_rebound.suspicious_aliases & rebound_existing
            )
        else:
            new_sup = b_delta.suppressors - local_all_sup
            new_susp = b_delta.suspicious_aliases - local_all_susp
        if new_sup:
            cur_all_sup.update(new_sup)
        if new_susp:
            cur_all_susp.update(new_susp)
        if b_delta.cross_scope_mutated:
            all_cs_mut.update(b_delta.cross_scope_mutated)
            all_cs_sup.update(b_delta.cross_scope_suppressors)
            all_cs_susp.update(b_delta.cross_scope_suspicious)

        if b_delta.has_star_import:
            eff_names = (
                stmt_names
                | cur_cs_mut
                | cur_nonsup
                | cur_nonsup_inst
                | cur_mods.keys()
                | cur_imps.keys()
                | cur_sup
            )
        else:
            eff_names = stmt_names | cur_cs_mut if cur_cs_mut else stmt_names
        local_sup = frozenset(n for n in eff_names if n in cur_sup)
        local_susp = frozenset(n for n in eff_names if n in cur_susp)
        local_nonsup = frozenset(n for n in eff_names if n in cur_nonsup)
        local_nonsup_inst = frozenset(n for n in eff_names if n in cur_nonsup_inst)
        local_mods = frozenset((n, cur_mods[n]) for n in eff_names if n in cur_mods)
        local_imps = frozenset((n, cur_imps[n]) for n in eff_names if n in cur_imps)
        local_shad_b = frozenset(cur_shad_b)
        local_shad_n = frozenset(
            n for n in (eff_names | _TRACKED_SHADOW_NAMES) if n in cur_shad_n
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
            suspicious_aliases=local_susp,
            cross_scope_mutated=frozenset(cur_cs_mut),
            cross_scope_suppressors=frozenset(cur_cs_sup),
            cross_scope_suspicious=frozenset(cur_cs_susp),
            exit_patch_targets=patch_scan.patched_names,
            has_exit_patch=patch_scan.has_patch,
            rebound_factory_targets=patch_scan.rebound_attrs,
        )
        out_cur = _update_suppress_provenance_from_stmt(
            m_stmt,
            local_cur,
            in_branch=False,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )

        fwd_add_sup = out_cur.suppressors - local_sup
        fwd_rem_sup = local_sup - out_cur.suppressors
        if fwd_rem_sup:
            cur_sup.difference_update(fwd_rem_sup)
        if fwd_add_sup:
            cur_sup.update(fwd_add_sup)

        fwd_add_susp = out_cur.suspicious_aliases - local_susp
        fwd_rem_susp = local_susp - out_cur.suspicious_aliases
        if fwd_rem_susp:
            cur_susp.difference_update(fwd_rem_susp)
        if fwd_add_susp:
            cur_susp.update(fwd_add_susp)

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

        fwd_cs_mut = out_cur.cross_scope_mutated - cur_cs_mut
        fwd_cs_sup = out_cur.cross_scope_suppressors - cur_cs_sup
        fwd_cs_susp = out_cur.cross_scope_suspicious - cur_cs_susp
        if fwd_cs_mut:
            cur_cs_mut.update(fwd_cs_mut)
            all_cs_mut.update(fwd_cs_mut)
        if fwd_cs_sup:
            cur_cs_sup.update(fwd_cs_sup)
            all_cs_sup.update(fwd_cs_sup)
        if fwd_cs_susp:
            cur_cs_susp.update(fwd_cs_susp)
            all_cs_susp.update(fwd_cs_susp)

        if out_cur.has_star_import:
            cur_star = True

        if (
            cur_sup
            or cur_susp
            or cur_nonsup
            or cur_nonsup_inst
            or cur_mods
            or cur_imps
            or cur_all_sup
            or cur_all_susp
            or cur_cs_mut
            or all_cs_mut
        ):
            has_pos_or_sup = True

        deltas.append(
            _ModuleStmtDelta(
                stmt_id=stmt_id,
                fwd_add_sup=fwd_add_sup,
                fwd_rem_sup=fwd_rem_sup,
                fwd_add_susp=fwd_add_susp,
                fwd_rem_susp=fwd_rem_susp,
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
                fwd_cs_mut=fwd_cs_mut,
                fwd_cs_sup=fwd_cs_sup,
                fwd_cs_susp=fwd_cs_susp,
                branch_new_sup=new_sup,
                branch_new_susp=new_susp,
                branch_shad_b=b_delta.shadowed_builtins,
                branch_shad_n=b_delta.shadowed_names,
                branch_star=b_delta.has_star_import,
            )
        )

    cs_mut_f = frozenset(all_cs_mut)
    cs_sup_f = frozenset(all_cs_sup)
    cs_susp_f = frozenset(all_cs_susp)
    full_forward_prov = _apply_exit_protocol_patch_scan(
        _SuppressProvenance(
            suppressors=frozenset(cur_sup),
            non_suppressors=frozenset(cur_nonsup),
            stdlib_modules=frozenset(cur_mods.items()),
            shadowed_builtins=frozenset(cur_shad_b),
            has_star_import=cur_star,
            imported_symbols=frozenset(cur_imps.items()),
            shadowed_names=frozenset(cur_shad_n),
            non_suppressor_instances=frozenset(cur_nonsup_inst),
            suspicious_aliases=frozenset(cur_susp),
            cross_scope_mutated=cs_mut_f,
            cross_scope_suppressors=cs_sup_f,
            cross_scope_suspicious=cs_susp_f,
        ),
        patch_scan,
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
        by_top_stmt_id=_LazyModuleStmtSeedMap(
            deltas_tuple, idx_map, "combined", patch_scan
        ),
        has_any_positive_or_suppressor=True,
        prefix_by_top_stmt_id=_LazyModuleStmtSeedMap(
            deltas_tuple, idx_map, "prefix", patch_scan
        ),
        suffix_delta_by_top_stmt_id=_LazyModuleStmtSeedMap(
            deltas_tuple, idx_map, "suffix", patch_scan
        ),
        module_cross_scope_mutated=cs_mut_f,
        module_cross_scope_suppressors=cs_sup_f,
        module_cross_scope_suspicious=cs_susp_f,
    )


def _get_module_suppress_seed(
    eff_mod: ast.Module,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _ModuleSuppressSeed:
    """Returns the cached _ModuleSuppressSeed for eff_mod."""
    extra_f = frozenset(extra_non_suppressing_ctx_managers)
    strict_w = bool(strict_with_suppression)
    if _is_ephemeral_unit_ast(eff_mod):
        return _build_module_suppress_seed(
            eff_mod,
            extra_non_suppressing_ctx_managers=extra_f,
            strict_with_suppression=strict_w,
        )
    seed_params: Tuple[Any, ...] = (
        (strict_w, extra_f) if (strict_w or extra_f) else ()
    )
    cache_key = _make_trusted_scope_cache_key(
        eff_mod, "module_suppress_seed", params=seed_params
    )
    return cast(
        _ModuleSuppressSeed,
        _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            cache_key,
            lambda: _build_module_suppress_seed(
                eff_mod,
                extra_non_suppressing_ctx_managers=extra_f,
                strict_with_suppression=strict_w,
            ),
        ),
    )


def _merge_enclosing_prior_clause_effects(
    stmt: ast.AST,
    clauses: Sequence[Tuple[Sequence[ast.AST], Tuple[str, ...], Optional[ast.AST]]],
    enclosing_clause_idx: int,
    prov: _SuppressProvenance,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Merges try-body and prior clause header/binding effects before enclosing_clause_idx."""
    if enclosing_clause_idx <= 0:
        return prov
    c_body = clauses[enclosing_clause_idx][0]
    if is_try_or_trystar_node(stmt):
        t_body = getattr(stmt, "body", ())
        o_stmts = getattr(stmt, "orelse", ())
        is_orelse = bool(o_stmts and c_body is o_stmts)
        for tb_stmt in t_body:
            tb_p = _update_suppress_provenance_from_stmt(
                tb_stmt,
                prov,
                in_branch=not is_orelse,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                _depth=_depth + 1,
            )
            prov = tb_p if is_orelse else prov.merge_branch_conservative(tb_p)
        if is_orelse:
            return prov
        for _, _, prior_hdr in clauses[1:enclosing_clause_idx]:
            prov = _merge_prior_clause_guard_and_binds(
                prov,
                prior_hdr,
                (),
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
        return prov
    if is_match_node(stmt):
        for _, prior_bound, prior_hdr in clauses[:enclosing_clause_idx]:
            prov = _merge_prior_clause_guard_and_binds(
                prov,
                prior_hdr,
                prior_bound,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
    return prov


def _advance_enclosing_compound_stmt_seed(
    stmt: ast.AST,
    prov: _SuppressProvenance,
    target_node: Optional[ast.AST],
    target_lineno: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Advances provenance into a compound statement enclosing target_node or target_lineno."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _conservatively_update_prov_at_depth_limit((stmt,), prov)
    if isinstance(stmt, ast.ClassDef) and target_node is not None:
        # Per PEP 227, ClassDef bodies do not form a lexical scope for nested
        # methods/classes/callables; only header, global/nonlocal, and cross-scope
        # mutations propagate across the class boundary.
        cls_ctx = _ProvenanceWalkContext(
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            depth=_depth + 1,
        )
        return _apply_class_def_external_effects(stmt, prov, cls_ctx).clear_name(
            stmt.name
        )
    is_loop = isinstance(stmt, (ast.While, ast.For, ast.AsyncFor))
    if isinstance(stmt, (ast.If, ast.While)):
        prov = _advance_prov_through_opt_expr(
            stmt.test,
            prov,
            allow_positive=not is_loop,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        _, prov = _advance_prov_through_for_header(
            stmt,
            prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        prov = _advance_prov_through_with_items(
            stmt,
            prov,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
    elif is_match_node(stmt):
        prov = _advance_prov_through_opt_expr(
            getattr(stmt, "subject", None),
            prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
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
            _update_suppress_provenance_from_stmt(
                stmt,
                prov,
                in_branch=True,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                _depth=_depth + 1,
            )
        )

    c_body, c_bound, c_hdr = clauses[enclosing_clause_idx]
    prov = _merge_enclosing_prior_clause_effects(
        stmt,
        clauses,
        enclosing_clause_idx,
        prov,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth,
    )
    prov = _advance_clause_entry_prov(
        prov,
        c_hdr,
        c_bound,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
    )

    prov = _advance_prov_across_stmts_until_target(
        c_body,
        prov,
        target_node,
        target_lineno,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )

    for c_idx, (sib_body, sib_bound, sib_hdr) in enumerate(clauses):
        if c_idx == enclosing_clause_idx and not is_loop:
            continue
        if isinstance(stmt, ast.If) or is_match_node(stmt):
            continue
        sib_prov = _advance_prov_through_opt_expr(
            sib_hdr,
            prov,
            allow_positive=False,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        for b_name in sib_bound:
            sib_prov = sib_prov.clear_name(b_name)
        sib_prov = _update_suppress_provenance_from_stmts(
            sib_body,
            sib_prov,
            in_branch=True,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        prov = prov.merge_branch_conservative(sib_prov)
    return prov


def _advance_prov_across_stmts_until_target(
    stmts: Sequence[ast.AST],
    prov: _SuppressProvenance,
    target_node: Optional[ast.AST],
    target_lineno: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> _SuppressProvenance:
    """Advances provenance forward to (target_node, target_lineno) and conservatively merges trailing statements."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _conservatively_update_prov_at_depth_limit(stmts, prov)
    reached = False
    for sub_stmt in stmts:
        if target_node is not None and sub_stmt is target_node:
            reached = True
            continue
        s_start = int(getattr(sub_stmt, "lineno", 0) or 0)
        s_end = int(getattr(sub_stmt, "end_lineno", s_start) or s_start)
        if not reached and target_lineno > 0 and s_start <= target_lineno <= s_end:
            if target_node is None and s_start >= target_lineno:
                return prov
            prov = _advance_enclosing_compound_stmt_seed(
                sub_stmt,
                prov,
                target_node,
                target_lineno,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
                _depth=_depth + 1,
            )
            if target_node is None:
                return prov
            reached = True
            continue
        if not reached and 0 < target_lineno <= s_start:
            if target_node is None:
                return prov
            reached = True
        next_sub = _update_suppress_provenance_from_stmt(
            sub_stmt,
            prov,
            in_branch=reached,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            _depth=_depth + 1,
        )
        prov = prov.merge_branch_conservative(next_sub) if reached else next_sub
    return prov


def _resolve_top_level_ancestor_seed(
    fn_node: ast.AST,
    eff_mod: ast.Module,
    enclosing_nodes: Sequence[ast.AST],
    mod_seed: _ModuleSuppressSeed,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Resolves module-level suppression seed using the top-level ancestor of fn_node."""
    top_target = enclosing_nodes[0] if enclosing_nodes else fn_node
    res: Optional[_SuppressProvenance] = None
    if id(top_target) in mod_seed.by_top_stmt_id:
        res = mod_seed.by_top_stmt_id[id(top_target)]
    else:
        target_lineno = int(getattr(top_target, "lineno", 0) or 0)
        if target_lineno > 0:
            for m_stmt in eff_mod.body:
                m_start = int(getattr(m_stmt, "lineno", 0) or 0)
                m_end = int(getattr(m_stmt, "end_lineno", m_start) or m_start)
                stmt_id = id(m_stmt)
                if m_start <= target_lineno <= m_end and stmt_id in mod_seed.by_top_stmt_id:
                    if top_target is m_stmt:
                        res = mod_seed.by_top_stmt_id[stmt_id]
                        break
                    pre_prov = mod_seed.prefix_by_top_stmt_id.get(
                        stmt_id, mod_seed.by_top_stmt_id[stmt_id]
                    )
                    inner_prov = _advance_enclosing_compound_stmt_seed(
                        m_stmt,
                        pre_prov,
                        top_target,
                        target_lineno,
                        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                        strict_with_suppression=strict_with_suppression,
                    )
                    suf_delta = mod_seed.suffix_delta_by_top_stmt_id.get(stmt_id)
                    if suf_delta is not None:
                        inner_prov = inner_prov.apply_branch_delta(
                            suf_delta.shadowed_names,
                            suf_delta.shadowed_builtins,
                            suf_delta.suppressors,
                            suf_delta.has_star_import,
                            new_suspicious=suf_delta.suspicious_aliases,
                        )
                    res = inner_prov
                    break
    if res is None:
        res = mod_seed.full_forward_prov
    return res.with_cross_scope_mutations(
        mod_seed.module_cross_scope_mutated,
        mod_seed.module_cross_scope_suppressors,
        mod_seed.module_cross_scope_suspicious,
    )


def _clear_callable_scope_locals_and_invalidate(
    prov: _SuppressProvenance,
    scope_node: ast.AST,
    *,
    explicit_locals: Optional[AbstractSet[str]] = None,
) -> _SuppressProvenance:
    """Clears callable parameters/locals on entry and applies outer cross-scope invalidation."""
    s_args = getattr(scope_node, "args", None)
    if isinstance(s_args, ast.arguments):
        for p_name in _extract_arg_names(s_args):
            prov = prov.clear_name(p_name, clear_cross_scope=True)
    for tp_name in _extract_node_type_param_names(scope_node):
        prov = prov.clear_name(tp_name, clear_cross_scope=True)
    true_locals = (
        set(explicit_locals)
        if explicit_locals is not None
        else _extract_single_node_local_names(
            scope_node,
            include_globals=False,
            include_nonlocals=False,
            include_self_name=False,
        )
    )
    for loc_name in true_locals:
        prov = prov.clear_name(loc_name, clear_cross_scope=True)
    for decl_name in (
        _extract_scope_local_names(scope_node, include_self_name=False) - true_locals
    ):
        prov = prov.clear_name(decl_name)
    if not isinstance(scope_node, ast.ClassDef):
        prov = prov.apply_cross_scope_invalidation()
    return prov


def _seed_scope_suppress_aliases_uncached(
    fn_node: ast.AST,
    eff_mod: Optional[ast.Module],
    enclosing_nodes: Sequence[ast.AST],
    scope_ctx: Optional[_FnScopeContext],
    u_start: int,
    *,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
) -> _SuppressProvenance:
    """Uncached implementation of _seed_scope_suppress_aliases."""
    patch_scan = scan_exit_protocol_patches(eff_mod if eff_mod is not None else fn_node)
    prov = _apply_exit_protocol_patch_scan(_EMPTY_SUPPRESS_PROVENANCE, patch_scan)
    if eff_mod is not None:
        mod_seed = _get_module_suppress_seed(
            eff_mod,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
        )
        if not mod_seed.has_any_positive_or_suppressor:
            prov = mod_seed.full_forward_prov
        elif fn_node is not eff_mod:
            prov = _resolve_top_level_ancestor_seed(
                fn_node,
                eff_mod,
                enclosing_nodes,
                mod_seed,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
        elif u_start <= 0:
            prov = mod_seed.full_forward_prov
        else:
            prov = _advance_prov_across_stmts_until_target(
                eff_mod.body,
                prov,
                None,
                u_start,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
        prov = _apply_exit_protocol_patch_scan(prov, patch_scan)

    for idx_enc, enc in enumerate(enclosing_nodes):
        if isinstance(enc, ast.ClassDef):
            # Per PEP 227, enclosing class attributes/methods do not bind lexical
            # names inside nested methods or inner classes.
            cls_ctx = _ProvenanceWalkContext(
                in_branch=True,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            )
            prov = _apply_class_def_external_effects(enc, prov, cls_ctx).clear_name(
                enc.name
            )
            continue
        prov = _clear_callable_scope_locals_and_invalidate(prov, enc)
        enc_body = getattr(enc, "body", None)
        if isinstance(enc_body, (list, tuple)):
            next_target = (
                enclosing_nodes[idx_enc + 1]
                if idx_enc + 1 < len(enclosing_nodes)
                else fn_node
            )
            next_target_lineno = int(getattr(next_target, "lineno", 0) or 0)
            prov = _advance_prov_across_stmts_until_target(
                enc_body,
                prov,
                next_target,
                next_target_lineno,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )

    if fn_node is not eff_mod:
        prov = _clear_callable_scope_locals_and_invalidate(
            prov,
            fn_node,
            explicit_locals=(scope_ctx.fn_locals if scope_ctx is not None else None),
        )
        fn_body = getattr(fn_node, "body", None)
        if isinstance(fn_body, (list, tuple)) and u_start > 0:
            prov = _advance_prov_across_stmts_until_target(
                fn_body,
                prov,
                None,
                u_start,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                strict_with_suppression=strict_with_suppression,
            )
    else:
        fn_args = getattr(fn_node, "args", None)
        if isinstance(fn_args, ast.arguments):
            for p_name in _extract_arg_names(fn_args):
                prov = prov.clear_name(p_name)
        for tp_name in _extract_node_type_param_names(fn_node):
            prov = prov.clear_name(tp_name)

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
        if isinstance(sc_node, ast.ClassDef) and sc_node is not fn_node:
            continue
        bound_in_chain.update(
            _extract_scope_local_names(
                sc_node,
                module_tree=eff_mod if sc_node is fn_node else None,
                enclosing_scopes=enclosing_nodes if sc_node is fn_node else (),
                include_self_name=(sc_node is fn_node),
            )
        )
        if (
            not has_star
            and sc_node is not eff_mod
            and _scope_has_direct_star_import(sc_node)
        ):
            has_star = True
    if has_star and not prov.has_star_import:
        prov = prov.with_star_import()
    proven_stdlib_aliases = {alias for alias, _ in prov.stdlib_modules}
    fn_own_locals = (
        _extract_scope_local_names(
            fn_node,
            module_tree=eff_mod,
            enclosing_scopes=enclosing_nodes,
        )
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
    return _apply_exit_protocol_patch_scan(prov, patch_scan)


def _seed_scope_suppress_aliases(
    fn_node: ast.AST,
    mod_tree: Optional[ast.AST] = None,
    scope_ctx: Optional[_FnScopeContext] = None,
    *,
    u_start: int = 0,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
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
    extra_f = frozenset(extra_non_suppressing_ctx_managers)
    strict_w = bool(strict_with_suppression)
    can_cache = not _is_ephemeral_unit_ast(fn_node) and all(
        not _is_ephemeral_unit_ast(n) for n in aux_nodes
    )
    with scope_fingerprint_verification_session():
        if can_cache and eff_mod is not None and fn_node is not eff_mod:
            if not _tree_contains_scope_node(eff_mod, fn_node) or any(
                n is not eff_mod and not _tree_contains_scope_node(eff_mod, n)
                for n in aux_nodes
            ):
                can_cache = False
        if not can_cache:
            return _seed_scope_suppress_aliases_uncached(
                fn_node,
                eff_mod,
                enclosing_nodes,
                scope_ctx,
                int(u_start),
                extra_non_suppressing_ctx_managers=extra_f,
                strict_with_suppression=strict_w,
            )
        seed_params: Tuple[Any, ...] = (
            (int(u_start), strict_w, extra_f)
            if (strict_w or extra_f)
            else (int(u_start),)
        )
        cache_key = _make_trusted_scope_cache_key(
            fn_node,
            "suppress_seed",
            params=seed_params,
            aux_ast_nodes=aux_nodes,
        )
        return cast(
            _SuppressProvenance,
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                cache_key,
                lambda: _seed_scope_suppress_aliases_uncached(
                    fn_node,
                    eff_mod,
                    enclosing_nodes,
                    scope_ctx,
                    int(u_start),
                    extra_non_suppressing_ctx_managers=extra_f,
                    strict_with_suppression=strict_w,
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
                and not prov.has_star_import
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
            if isinstance(recv_root, ast.Name):
                target_names: Set[str] = {recv_root.id}
                for alias, mod_name in prov.stdlib_modules:
                    if alias == recv_root.id:
                        target_names |= prov.module_aliases_for(mod_name)
                if len(chain_parts) >= 2:
                    matched_qual = False
                    for alias, qual in prov.imported_symbols:
                        if alias == recv_root.id:
                            matched_qual = True
                            mod_candidate = ".".join((qual, *chain_parts[:-1]))
                            if mod_candidate in _STDLIB_NON_SUPPRESSING_MODULES:
                                target_names |= prov.module_aliases_for(
                                    mod_candidate
                                ) | {chain_parts[-2]}
                    if not matched_qual and recv_root.id not in prov.shadowed_names:
                        raw_mod_candidate = ".".join(
                            (recv_root.id, *chain_parts[:-1])
                        )
                        if raw_mod_candidate in _STDLIB_NON_SUPPRESSING_MODULES:
                            target_names |= prov.module_aliases_for(
                                raw_mod_candidate
                            ) | {chain_parts[-2]}
                include_untargeted = not (
                    ident == "open"
                    and bool(
                        target_names & frozenset({"builtins", "__builtins__", "io"})
                    )
                )
                if prov.is_factory_rebound(
                    target_names, ident, include_untargeted=include_untargeted
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
        elif isinstance(sub, (ast.Subscript, ast.Await)):
            stack.append(sub.value)
        elif isinstance(sub, ast.NamedExpr):
            stack.append(sub.target)
            stack.append(sub.value)
        elif isinstance(sub, ast.IfExp):
            stack.extend((sub.body, sub.orelse))
        elif isinstance(sub, ast.BoolOp):
            stack.extend(sub.values)
        elif isinstance(sub, (ast.List, ast.Tuple, ast.Set)):
            stack.extend(sub.elts)
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
    if item_mode == "non_suppressing" and extra_non_suppressing_ctx_managers:
        matched_wc = _find_ctx_manager_wildcard_match(
            expr, prov, extra_non_suppressing_ctx_managers
        )
        if matched_wc is not None:
            record_wildcard_ctx_manager_match(
                matched_wc, _format_ctx_manager_display_name(expr, prov)
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
        pre_call_subexprs: Tuple[ast.AST, ...] = (
            (
                item.context_expr.func,
                *item.context_expr.args,
                *(kw.value for kw in item.context_expr.keywords),
            )
            if isinstance(item.context_expr, ast.Call)
            else ()
        )
        if pre_call_subexprs and _exprs_may_invoke_cross_scope_mutator(
            pre_call_subexprs,
            post_expr_prov,
            extra_non_suppressing_ctx_managers,
        ):
            post_expr_prov = post_expr_prov.apply_cross_scope_invalidation()
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
        if _exprs_may_invoke_cross_scope_mutator(
            (item.context_expr,),
            post_expr_prov,
            extra_non_suppressing_ctx_managers,
        ):
            post_item_prov = post_item_prov.apply_cross_scope_invalidation()
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
    _depth: int = 0,
) -> bool:
    """Returns True if evaluating bool(expr) cannot raise an exception."""
    return _is_non_raising_expr(
        expr,
        safe_names,
        safe_bool_names,
        require_bool_safe=True,
        strict_names=strict_names,
        _depth=_depth,
    )


def _apply_non_raising_if_branch_updates(
    node: Union[ast.If, ast.IfExp],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    is_conditional: bool = False,
    _depth: int = 0,
) -> None:
    """Applies walrus/statement safe updates across test, body, and orelse of If/IfExp."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    _apply_non_raising_expr_walrus_updates(
        node.test,
        cur_safe,
        cur_bool,
        strict_names=strict_names,
        is_conditional=is_conditional,
        _depth=_depth + 1,
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
            _depth=_depth + 1,
        )
        _apply_non_raising_expr_walrus_updates(
            node.orelse,
            else_safe,
            else_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
    else:
        _apply_non_raising_seq_safe_updates(
            node.body,
            then_safe,
            then_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
        _apply_non_raising_seq_safe_updates(
            node.orelse,
            else_safe,
            else_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
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
    _depth: int = 0,
) -> None:
    """Applies left-to-right cur_safe and cur_bool updates from NamedExpr nodes in a non-raising expr."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    if expr is None or isinstance(expr, ast.Constant):
        return
    if isinstance(expr, ast.IfExp):
        _apply_non_raising_if_branch_updates(
            expr,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
        return
    if isinstance(expr, ast.NamedExpr):
        _apply_non_raising_expr_walrus_updates(
            expr.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=is_conditional,
            _depth=_depth + 1,
        )
        if isinstance(expr.target, ast.Name):
            t_name = expr.target.id
            val_bool = _is_non_raising_bool_expr(
                expr.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
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
            sub_node,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            is_conditional=sub_cond,
            _depth=_depth + 1,
        )


def _check_and_advance_non_raising_seq(
    nodes: Sequence[ast.AST],
    safe_names: AbstractSet[str],
    safe_bool_names: AbstractSet[str],
    *,
    strict_names: bool,
    require_bool_safe: bool = False,
    conditional_from_idx: Optional[int] = None,
    _depth: int = 0,
) -> Optional[Tuple[Set[str], Set[str]]]:
    """Verifies each node in nodes is non-raising left-to-right and returns updated (safe, bool) sets."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return None
    cur_safe = set(safe_names)
    cur_bool = set(safe_bool_names)
    for idx, node in enumerate(nodes):
        ok = (
            _is_non_raising_bool_expr(
                node,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            if require_bool_safe
            else _is_non_raising_expr(
                node,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
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
            _depth=_depth + 1,
        )
    return cur_safe, cur_bool


def _is_non_raising_expr(
    expr: Optional[ast.AST],
    safe_names: AbstractSet[str] = frozenset(),
    safe_bool_names: AbstractSet[str] = frozenset(),
    *,
    require_bool_safe: bool = False,
    strict_names: bool = True,
    _depth: int = 0,
) -> bool:
    """Returns True if evaluating expr cannot raise an exception caught by a with block."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    while isinstance(expr, ast.NamedExpr):
        if not isinstance(expr.target, ast.Name):
            return False
        expr = expr.value
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
                _depth=_depth + 1,
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
                _depth=_depth + 1,
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
                _depth=_depth + 1,
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
            _depth=_depth + 1,
        )
        return advanced is not None and _is_non_raising_expr(
            expr.values[-1],
            advanced[0],
            advanced[1],
            require_bool_safe=require_bool_safe,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    if isinstance(expr, ast.IfExp):
        advanced = _check_and_advance_non_raising_seq(
            (expr.test,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            require_bool_safe=True,
            _depth=_depth + 1,
        )
        return advanced is not None and all(
            _is_non_raising_expr(
                b,
                advanced[0],
                advanced[1],
                require_bool_safe=require_bool_safe,
                strict_names=strict_names,
                _depth=_depth + 1,
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
                _depth=_depth + 1,
            )
            is not None
        )
    return False


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


def _is_valid_suppress_or_null_call_signature(
    qual: str,
    call: ast.Call,
) -> bool:
    """Validates positional/keyword arity and names for contextlib.suppress/nullcontext calls."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        kw.arg is None for kw in call.keywords
    ):
        return False
    if qual == "contextlib.suppress":
        return not call.keywords
    if qual == "contextlib.nullcontext":
        return len(call.args) + len(call.keywords) <= 1 and all(
            kw.arg == "enter_result" for kw in call.keywords
        )
    return False


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
        qual = _resolve_tracked_factory_qualname(call.func, cur_prov)
        allowed = (
            ("contextlib.nullcontext",)
            if is_async
            else ("contextlib.suppress", "contextlib.nullcontext")
        )
        if (
            qual is None
            or qual not in allowed
            or not _is_valid_suppress_or_null_call_signature(qual, call)
            or cur_prov.has_exit_patch
            or cur_prov.is_factory_rebound(
                cur_prov.module_aliases_for("contextlib"),
                qual.rsplit(".", 1)[1],
            )
        ):
            return False
        if item.optional_vars is not None:
            cur_prov = _clear_store_target_provenance(item.optional_vars, cur_prov)
    return True


def _non_raising_stmt_terminates(stmt: ast.AST, *, _depth: int = 0) -> bool:
    """Returns True if a proven non-raising statement unconditionally exits the block."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.body
        )
    if isinstance(stmt, ast.If) and stmt.orelse:
        return any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.body
        ) and any(
            _non_raising_stmt_terminates(s, _depth=_depth + 1) for s in stmt.orelse
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
    _depth: int = 0,
) -> bool:
    """Returns True if stmt executes and completes without raising a catchable exception."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return False
    if isinstance(stmt, (ast.Pass, ast.Global, ast.Nonlocal, ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, ast.Delete):
        seen_del: Set[str] = set()
        for t in stmt.targets:
            if (
                not isinstance(t, ast.Name)
                or t.id not in safe_names
                or t.id in seen_del
            ):
                return False
            seen_del.add(t.id)
        return bool(seen_del)
    if isinstance(stmt, (ast.Return, ast.Expr)):
        return _is_non_raising_expr(
            stmt.value,
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    if isinstance(stmt, ast.Assign):
        return (
            stmt.value is not None
            and _is_non_raising_expr(
                stmt.value,
                safe_names,
                safe_bool_names,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            and all(_is_non_raising_store_target(t, stmt.value) for t in stmt.targets)
        )
    if isinstance(stmt, ast.AnnAssign):
        ann_sets = _check_and_advance_non_raising_seq(
            (stmt.annotation,),
            safe_names,
            safe_bool_names,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
        if ann_sets is None:
            return False
        return stmt.value is None or (
            _is_non_raising_expr(
                stmt.value,
                ann_sets[0],
                ann_sets[1],
                strict_names=strict_names,
                _depth=_depth + 1,
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
                _depth=_depth + 1,
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
            _depth=_depth + 1,
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
                _depth=_depth + 1,
            )[1]
            and not _split_suppress_with_body(
                stmt.orelse,
                safe_names=test_safe,
                safe_bool_names=test_bool,
                strict_names=strict_names,
                suppress_aliases=test_prov,
                _depth=_depth + 1,
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
            unshadowed_exc_names = (
                frozenset()
                if inner_prov.has_star_import
                else (_SAFE_EXCEPTION_TYPE_NAMES - inner_prov.shadowed_names)
            )
            if (
                _check_and_advance_non_raising_seq(
                    arg_exprs,
                    inner_safe | unshadowed_exc_names,
                    inner_bool,
                    strict_names=strict_names,
                    _depth=_depth + 1,
                )
                is None
            ):
                return False
            _apply_non_raising_expr_walrus_updates(
                item.context_expr,
                inner_safe,
                inner_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
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
            _depth=_depth + 1,
        )[1]
    return False


def _collect_target_unpack_bool_flags(
    target: ast.AST,
    value: Optional[ast.AST],
    rhs_safe: AbstractSet[str],
    rhs_bool: AbstractSet[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> List[Tuple[str, bool]]:
    """Collects left-to-right (target_name, is_bool_safe) pairs evaluated against RHS state."""
    pairs: List[Tuple[str, bool]] = []
    stack: List[Tuple[ast.AST, Optional[ast.AST]]] = [(target, value)]
    while stack:
        curr_target, curr_value = stack.pop()
        while isinstance(curr_value, ast.NamedExpr):
            curr_value = curr_value.value
        if isinstance(curr_target, ast.Name):
            pairs.append(
                (
                    curr_target.id,
                    _is_non_raising_bool_expr(
                        curr_value,
                        rhs_safe,
                        rhs_bool,
                        strict_names=strict_names,
                        _depth=_depth + 1,
                    ),
                )
            )
        else:
            matched_pairs = _match_sequence_unpack_pairs(
                curr_target, curr_value, allow_starred_target=False
            )
            if matched_pairs is not None:
                stack.extend(reversed(matched_pairs))
            else:
                for t_name in _extract_definite_target_names(curr_target):
                    pairs.append((t_name, False))
    return pairs


def _apply_target_unpack_safe_updates(
    target: ast.AST,
    value: Optional[ast.AST],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> None:
    """Updates cur_safe and cur_bool for an assignment target matched against value."""
    for t_name, is_bool_safe in _collect_target_unpack_bool_flags(
        target,
        value,
        frozenset(cur_safe),
        frozenset(cur_bool),
        strict_names=strict_names,
        _depth=_depth,
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
    _depth: int = 0,
) -> None:
    """Updates cur_safe and cur_bool after a statement proven non-raising by _stmt_cannot_raise_in_suppress."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    if isinstance(b_stmt, ast.Delete):
        for tgt in b_stmt.targets:
            if isinstance(tgt, ast.Name):
                cur_safe.discard(tgt.id)
                cur_bool.discard(tgt.id)
    elif isinstance(b_stmt, (ast.Return, ast.Expr)):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    elif isinstance(b_stmt, ast.Assign):
        _apply_non_raising_expr_walrus_updates(
            b_stmt.value,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
        for tgt in b_stmt.targets:
            _apply_target_unpack_safe_updates(
                tgt,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
    elif isinstance(b_stmt, ast.AnnAssign):
        if b_stmt.value is not None:
            _apply_non_raising_expr_walrus_updates(
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            _apply_target_unpack_safe_updates(
                b_stmt.target,
                b_stmt.value,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
    elif isinstance(b_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for h in iter_nested_scope_header_nodes(b_stmt):
            _apply_non_raising_expr_walrus_updates(
                h,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
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
                _depth=_depth + 1,
            )
            if isinstance(item.optional_vars, ast.Name):
                cur_safe.add(item.optional_vars.id)
                cur_bool.discard(item.optional_vars.id)
        _apply_non_raising_seq_safe_updates(
            b_stmt.body,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )
    elif isinstance(b_stmt, ast.If):
        _apply_non_raising_if_branch_updates(
            b_stmt,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
        )


def _apply_non_raising_seq_safe_updates(
    stmts: Sequence[ast.stmt],
    cur_safe: Set[str],
    cur_bool: Set[str],
    *,
    strict_names: bool,
    _depth: int = 0,
) -> None:
    """Propagates safe bindings sequentially across a non-raising statement list until termination."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        cur_safe.clear()
        cur_bool.clear()
        return
    for sub_s in stmts:
        if _non_raising_stmt_terminates(sub_s, _depth=_depth + 1):
            break
        _apply_non_raising_stmt_safe_updates(
            sub_s,
            cur_safe,
            cur_bool,
            strict_names=strict_names,
            _depth=_depth + 1,
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
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a potentially suppressing with-block body into (non_raising_prefix, potentially_raising_suffix)."""
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return [], list(body)
    safe_prefix: List[ast.stmt] = []
    cur_prov = _coerce_suppress_provenance(suppress_aliases)
    cur_safe: Set[str] = (
        set(safe_names)
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
            _depth=_depth + 1,
        ):
            safe_prefix.append(b_stmt)
            if _non_raising_stmt_terminates(b_stmt, _depth=_depth + 1):
                return safe_prefix, []
            _apply_non_raising_stmt_safe_updates(
                b_stmt,
                cur_safe,
                cur_bool,
                strict_names=strict_names,
                _depth=_depth + 1,
            )
            cur_prov = _update_suppress_provenance_from_stmt(
                b_stmt, cur_prov, _depth=_depth + 1
            )
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
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits an unknown context manager's body into (guaranteed_non_raising_prefix, potentially_raising_suffix)."""
    return _split_suppress_with_body(
        body,
        safe_names=safe_names,
        strict_names=strict_names,
        suppress_aliases=suppress_aliases,
        _depth=_depth,
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
    _depth: int = 0,
) -> Tuple[List[ast.stmt], List[ast.stmt]]:
    """Splits a with-block body into (guaranteed_prefix, potentially_suppressed_suffix) across 3 suppression states."""
    _record_with_items_heuristic_ctx_managers(
        stmt,
        suppress_aliases,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
    )
    if _depth > _MAX_AST_RECURSION_DEPTH or _with_header_has_post_suppress_risk(
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
            _depth=_depth,
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
                _depth=_depth,
            )
    return list(stmt.body), []


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
    _depth: int = 0,
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
        _depth=_depth,
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


class _BranchFlowState(NamedTuple):
    """Snapshot of definite, conditional, may_deleted, and must_deleted sets on a control-flow path."""

    definite: Set[str]
    conditional: Set[str]
    may_del: Set[str]
    must_del: Set[str]


@dataclass
class _LoopJumpSummary:
    """Tracks variable state across break/continue jumps targeting the innermost enclosing loop."""

    has_jump: bool = False
    has_break: bool = False
    may_bound: Set[str] = field(default_factory=set)
    may_deleted: Set[str] = field(default_factory=set)


def _join_branch_flow_states(states: Sequence[_BranchFlowState]) -> _BranchFlowState:
    """Computes the conservative lattice join across one or more control-flow branch states."""
    def_out = set.intersection(*(s.definite for s in states))
    may_bound_out = set.union(*(s.definite | s.conditional for s in states))
    may_del_out = set.union(*(s.may_del for s in states))
    must_del_out = set.intersection(*(s.must_del for s in states))
    return _BranchFlowState(
        definite=def_out,
        conditional=may_bound_out - def_out,
        may_del=may_del_out,
        must_del=must_del_out,
    )


def _overwrite_flow_state(
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    state: _BranchFlowState,
) -> None:
    """Replaces in-place flow tracking sets with the joined _BranchFlowState."""
    definite.clear()
    definite.update(state.definite)
    conditional.clear()
    conditional.update(state.conditional)
    may_del.clear()
    may_del.update(state.may_del)
    must_del.clear()
    must_del.update(state.must_del)


def _merge_hazard_provenance(
    base: _SuppressProvenance,
    hazard_src: _SuppressProvenance,
) -> _SuppressProvenance:
    """Preserves base's positive bindings while unioning suppressor, shadowing, exit-patch, and cross-scope hazards."""
    merged = replace(
        base,
        suppressors=base.suppressors | hazard_src.suppressors,
        shadowed_builtins=base.shadowed_builtins | hazard_src.shadowed_builtins,
        has_star_import=base.has_star_import or hazard_src.has_star_import,
        suspicious_aliases=base.suspicious_aliases | hazard_src.suspicious_aliases,
        cross_scope_mutated=base.cross_scope_mutated | hazard_src.cross_scope_mutated,
        cross_scope_suppressors=(
            base.cross_scope_suppressors | hazard_src.cross_scope_suppressors
        ),
        cross_scope_suspicious=(
            base.cross_scope_suspicious | hazard_src.cross_scope_suspicious
        ),
        exit_patch_targets=base.exit_patch_targets | hazard_src.exit_patch_targets,
        has_exit_patch=base.has_exit_patch or hazard_src.has_exit_patch,
        rebound_factory_targets=(
            base.rebound_factory_targets | hazard_src.rebound_factory_targets
        ),
    )
    if hazard_src.has_exit_patch or hazard_src.rebound_factory_targets:
        return merged.with_exit_protocol_patches(
            hazard_src.exit_patch_targets,
            has_patch=hazard_src.has_exit_patch,
            rebound_attrs=hazard_src.rebound_factory_targets,
        )
    return merged


def _finalize_surviving_branch_flow(
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    candidates: Sequence[Tuple[bool, _BranchFlowState]],
    terminated_fallback: _BranchFlowState,
    out_prov: _SuppressProvenance,
) -> Tuple[bool, _SuppressProvenance]:
    """Commits joined surviving branch flow states or terminated_fallback when all paths terminate."""
    surviving = [state for is_alive, state in candidates if is_alive]
    target_state = (
        _join_branch_flow_states(surviving) if surviving else terminated_fallback
    )
    _overwrite_flow_state(definite, conditional, may_del, must_del, target_state)
    return not surviving, out_prov


def _collect_block_shallow_flow_at_depth_limit(
    statements: Sequence[ast.AST],
) -> Tuple[Set[str], Set[str], bool, bool]:
    """Iteratively collects (raw_assigned, raw_deleted, has_break, has_continue) for depth fallback."""
    summary = _collect_shallow_ast_flow_summary(statements)
    return (
        summary.assigned | summary.except_names,
        summary.deleted | summary.except_names,
        summary.has_break,
        summary.has_continue,
    )


def _apply_depth_bailout_block_flow(
    statements: Sequence[ast.AST],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
) -> Tuple[Set[str], Set[str], bool, _SuppressProvenance]:
    """Conservatively summarizes a statement sequence when recursion reaches _MAX_AST_RECURSION_DEPTH."""
    raw_assigned, raw_deleted, has_break, has_continue = (
        _collect_block_shallow_flow_at_depth_limit(statements)
    )
    assigned = _filter_scope_names(raw_assigned, active_nonlocals, fn_globals)
    deleted = _filter_scope_names(raw_deleted, active_nonlocals, fn_globals)
    before_def = set(definite)
    live_in_del.update(deleted - before_def)
    definite.difference_update(deleted)
    conditional.update(assigned | (before_def & deleted))
    may_del.update(deleted)
    must_del.difference_update(assigned)
    if ever_bound is not None:
        ever_bound.update(assigned)
    if loop_jumps is not None and (has_break or has_continue):
        loop_jumps.has_jump = True
        if has_break:
            loop_jumps.has_break = True
        loop_jumps.may_bound.update(definite | conditional | assigned)
        loop_jumps.may_deleted.update(may_del)
    local_prov = _conservatively_update_prov_at_depth_limit(statements, local_prov)
    return definite, conditional - definite, False, local_prov


def _analyze_if_stmt_assignment(
    stmt: ast.If,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    _depth: int = 0,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes an ast.If statement and updates flow sets in place."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    cur_safe = d_in | set(known_safe_names)

    b_may = set(may_in)
    b_must = set(must_in)
    b_live: Set[str] = set()
    b_def, b_cond, b_term, b_prov = _analyze_block_assignment_impl(
        stmt.body,
        definite=set(d_in),
        conditional=set(c_in),
        may_deleted=b_may,
        must_deleted=b_must,
        live_in_deleted=b_live,
        suppress_aliases=local_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=cur_safe,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )
    b_state = _BranchFlowState(b_def, b_cond, b_may, b_must)

    o_live: Set[str] = set()
    if stmt.orelse:
        o_may = set(may_in)
        o_must = set(must_in)
        o_def, o_cond, o_term, o_prov = _analyze_block_assignment_impl(
            stmt.orelse,
            definite=set(d_in),
            conditional=set(c_in),
            may_deleted=o_may,
            must_deleted=o_must,
            live_in_deleted=o_live,
            suppress_aliases=local_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            loop_jumps=loop_jumps,
            ever_bound=ever_bound,
            known_safe_names=cur_safe,
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
            _depth=_depth + 1,
        )
        o_state = _BranchFlowState(o_def, o_cond, o_may, o_must)
    else:
        o_term = False
        o_prov = local_prov
        o_state = _BranchFlowState(d_in, c_in, may_in, must_in)

    local_prov = b_prov.merge_branch_conservative(o_prov)
    live_in_del.update((b_live | o_live) - d_in)

    return _finalize_surviving_branch_flow(
        definite,
        conditional,
        may_del,
        must_del,
        ((not b_term, b_state), (not o_term, o_state)),
        _join_branch_flow_states((b_state, o_state)),
        local_prov,
    )


def _analyze_with_stmt_assignment(
    stmt: Union[ast.With, ast.AsyncWith],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    bind_def_cond: Any,
    apply_expr_effects: Any,
    _depth: int = 0,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a With/AsyncWith statement and updates flow sets in place."""
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
        _depth=_depth + 1,
    )
    _, _, w_term, local_prov = _analyze_block_assignment_impl(
        safe_prefix,
        definite,
        conditional,
        may_deleted=may_del,
        must_deleted=must_del,
        live_in_deleted=live_in_del,
        suppress_aliases=local_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=cur_safe,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )
    if w_term:
        return True, local_prov
    if risky_suffix:
        d_pre = set(definite)
        c_pre = set(conditional)
        may_pre = set(may_del)
        must_pre = set(must_del)
        w_may = set(may_pre)
        w_must = set(must_pre)
        w_live: Set[str] = set()
        w_ever: Set[str] = set()
        w_def, w_cond, w_sub_term, w_prov = _analyze_block_assignment_impl(
            risky_suffix,
            definite=set(d_pre),
            conditional=set(c_pre),
            may_deleted=w_may,
            must_deleted=w_must,
            live_in_deleted=w_live,
            suppress_aliases=local_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            loop_jumps=loop_jumps,
            ever_bound=w_ever,
            known_safe_names=d_pre | set(known_safe_names),
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
            _depth=_depth + 1,
        )
        if ever_bound is not None:
            ever_bound.update(w_ever)
        local_prov = local_prov.merge_branch_conservative(w_prov)
        live_in_del.update(w_live - d_pre)
        all_risky_dels = _extract_block_combined_deletions(risky_suffix)
        sup_def = d_pre - all_risky_dels
        sup_state = _BranchFlowState(
            definite=sup_def,
            conditional=(d_pre | c_pre | w_ever) - sup_def,
            may_del=may_pre | all_risky_dels,
            must_del=must_pre - w_ever,
        )
        branches: List[_BranchFlowState] = [sup_state]
        if not w_sub_term:
            branches.append(_BranchFlowState(w_def, w_cond, w_may, w_must))
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(branches)
        )
    return False, local_prov


class _TryFlowContext(NamedTuple):
    """Shared invariant analysis context passed across try/handler/finally/compound dispatch."""

    active_nonlocals: Optional[Set[str]]
    fn_globals: AbstractSet[str]
    loop_control_terminates: bool
    loop_jumps: Optional[_LoopJumpSummary]
    ever_bound: Optional[Set[str]]
    known_safe_names: AbstractSet[str]
    strict_names: bool
    extra_non_suppressing_ctx_managers: AbstractSet[str]
    strict_with_suppression: bool
    depth: int


class _TryHandlersResult(NamedTuple):
    """Aggregated flow state and provenance across all except/except* handlers."""

    target_branches: List[_BranchFlowState]
    pre_finally_term: bool
    all_branches: List[_BranchFlowState]
    final_prov: _SuppressProvenance
    post_try_prov: _SuppressProvenance
    h_ever_all: Set[str]


def _analyze_try_except_handlers(
    handlers: Sequence[ast.AST],
    *,
    is_trystar: bool,
    h_entry: _BranchFlowState,
    succ_state: _BranchFlowState,
    succ_term: bool,
    try_body_prov: _SuppressProvenance,
    handler_base_prov: _SuppressProvenance,
    d_in: Set[str],
    raw_try_del: Set[str],
    live_in_del: Set[str],
    cur_safe: AbstractSet[str],
    inner_jumps: Optional[_LoopJumpSummary],
    ctx: _TryFlowContext,
) -> _TryHandlersResult:
    """Evaluates except / except* handler branches and merges surviving provenance."""
    surviving: List[_BranchFlowState] = []
    all_branches: List[_BranchFlowState] = [succ_state]
    surviving_provs: List[_SuppressProvenance] = []
    if not succ_term:
        surviving.append(succ_state)
        if not (is_trystar and handlers):
            surviving_provs.append(try_body_prov)
    final_prov = handler_base_prov.merge_branch_conservative(try_body_prov)
    handler_scan_prov = handler_base_prov

    trystar_acc: Optional[_BranchFlowState] = h_entry if is_trystar else None
    any_trystar_survives = False
    h_ever_all: Set[str] = set()
    for h in handlers:
        h_body = list(getattr(h, "body", []))
        h_type = getattr(h, "type", None)
        if isinstance(h_type, ast.AST):
            d_ht, c_ht = _walrus_assignment_in_expr(h_type, is_conditional=True)
            ht_filt = _filter_scope_names(
                d_ht | c_ht, ctx.active_nonlocals, ctx.fn_globals
            )
            h_ever_all.update(ht_filt)
            if ctx.ever_bound is not None:
                ctx.ever_bound.update(ht_filt)
            h_entry = _BranchFlowState(
                definite=h_entry.definite,
                conditional=(h_entry.conditional | ht_filt) - h_entry.definite,
                may_del=h_entry.may_del,
                must_del=h_entry.must_del - ht_filt,
            )
        base_h = (
            _join_branch_flow_states((h_entry, trystar_acc))
            if (is_trystar and trystar_acc is not None)
            else h_entry
        )
        h_name = getattr(h, "name", None)
        h_bound: Set[str] = set()
        if isinstance(h_name, str) and _can_bind_in_fn_scope(
            h_name, ctx.active_nonlocals, ctx.fn_globals
        ):
            h_bound.add(h_name)
        h_prov = _advance_clause_entry_prov(
            handler_scan_prov,
            h_type,
            (h_name,) if isinstance(h_name, str) else (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        handler_scan_prov = _merge_prior_clause_guard_and_binds(
            handler_scan_prov,
            h_type,
            (),
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        h_live: Set[str] = set()
        h_ever: Set[str] = set()
        hmay_in = base_h.may_del - h_bound
        hmust_in = base_h.must_del - h_bound
        hd_in = base_h.definite | h_bound
        hd_out, hc_out, h_raw_term, h_out_prov = _analyze_block_assignment_impl(
            h_body,
            definite=hd_in,
            conditional=base_h.conditional - hd_in,
            may_deleted=hmay_in,
            must_deleted=hmust_in,
            live_in_deleted=h_live,
            suppress_aliases=h_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            loop_control_terminates=ctx.loop_control_terminates,
            loop_jumps=inner_jumps,
            ever_bound=h_ever,
            known_safe_names=(cur_safe - raw_try_del) | h_bound,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth + 1,
        )
        if isinstance(h_name, str):
            h_out_prov = h_out_prov.clear_name(h_name)
        final_prov = final_prov.merge_branch_conservative(h_out_prov)
        h_ever_clean = h_ever - h_bound
        h_ever_all.update(h_ever_clean)
        if ctx.ever_bound is not None:
            ctx.ever_bound.update(h_ever_clean)
        live_in_del.update((h_live - h_bound) - d_in)

        hd_out.difference_update(h_bound)
        hc_out.difference_update(h_bound)
        if h_bound:
            hmay_in.update(h_bound)
            hmust_in.update(h_bound)
        h_state = _BranchFlowState(hd_out, hc_out, hmay_in, hmust_in)
        all_branches.append(h_state)
        if not h_raw_term:
            if is_trystar:
                any_trystar_survives = True
                trystar_acc = (
                    h_state
                    if trystar_acc is None
                    else _join_branch_flow_states((trystar_acc, h_state))
                )
            else:
                surviving.append(h_state)
                surviving_provs.append(h_out_prov)

    if is_trystar and any_trystar_survives and trystar_acc is not None:
        surviving.append(trystar_acc)

    post_try_prov = final_prov
    if surviving_provs and not (is_trystar and handlers):
        post_try_prov = surviving_provs[0]
        for sp in surviving_provs[1:]:
            post_try_prov = post_try_prov.merge_branch_conservative(sp)
        post_try_prov = _merge_hazard_provenance(post_try_prov, final_prov)
    final_prov = final_prov.merge_branch_conservative(handler_scan_prov)

    return _TryHandlersResult(
        target_branches=surviving if surviving else all_branches,
        pre_finally_term=not surviving,
        all_branches=all_branches,
        final_prov=final_prov,
        post_try_prov=post_try_prov,
        h_ever_all=h_ever_all,
    )


class _TryFinallyAbruptEntry(NamedTuple):
    """Abrupt-exception entry flow state for a try-statement finally block."""

    pre_try_def: Set[str]
    definite: Set[str]
    conditional: Set[str]
    may_del: Set[str]


def _compute_try_finally_abrupt_entry(
    t_body: Sequence[ast.stmt],
    o_stmts: Sequence[ast.stmt],
    f_body: Sequence[ast.stmt],
    *,
    split: _TryBodySplitProvenance,
    t_def: Set[str],
    t_ever: Set[str],
    o_ever: Set[str],
    d_in: Set[str],
    c_in: Set[str],
    may_in: Set[str],
    raw_try_del: Set[str],
    t_may: Set[str],
    h_res: _TryHandlersResult,
    local_prov: _SuppressProvenance,
    cur_safe: Set[str],
    ctx: _TryFlowContext,
) -> _TryFinallyAbruptEntry:
    """Computes the conservative flow state when entering finally via an abrupt exit."""
    abrupt_cond = d_in | c_in | t_ever | o_ever | h_res.h_ever_all
    abrupt_may = (
        may_in
        | raw_try_del
        | t_may
        | _extract_block_combined_deletions(o_stmts)
        | set().union(*(s.may_del for s in h_res.all_branches))
    )
    safe_prefix_len = len(t_body) - len(split.risky_suffix)
    guaranteed_prefix_len = 0
    for s_item in t_body[:safe_prefix_len]:
        if any(_scan_stmts_shallow_exits([s_item])):
            break
        guaranteed_prefix_len += 1
    if (
        not split.risky_suffix
        and not split.terminated
        and guaranteed_prefix_len == len(t_body)
    ):
        abrupt_def = set(t_def) - abrupt_may
    elif (
        guaranteed_prefix_len > 0
        and ctx.depth < 8
        and _extract_block_combined_deletions(f_body)
    ):
        pre_def, _, _, _ = _analyze_block_assignment_impl(
            t_body[:guaranteed_prefix_len],
            definite=set(d_in),
            suppress_aliases=local_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            known_safe_names=cur_safe,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth + 1,
        )
        abrupt_def = pre_def - abrupt_may
    else:
        abrupt_def = set(d_in) - abrupt_may
    return _TryFinallyAbruptEntry(
        pre_try_def=d_in,
        definite=abrupt_def,
        conditional=abrupt_cond,
        may_del=abrupt_may,
    )


def _analyze_try_finally_clause(
    f_body: Sequence[ast.AST],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    final_prov: _SuppressProvenance,
    post_try_prov: _SuppressProvenance,
    *,
    abrupt_entry: _TryFinallyAbruptEntry,
    inner_jumps: Optional[_LoopJumpSummary],
    ctx: _TryFlowContext,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a try-statement finally block and propagates loop jump effects."""
    entry_def = set(definite)
    post_known_safe = definite | set(ctx.known_safe_names)
    has_ret, has_brk, has_cont = _scan_stmts_shallow_exits(f_body)
    has_unescaped_jump = has_brk or has_cont
    may_suppress_exc = has_ret or has_unescaped_jump

    def _run_finally(
        d_set: Set[str],
        c_set: Set[str],
        may_set: Set[str],
        must_set: Optional[Set[str]] = None,
        *,
        live_del: Optional[Set[str]] = None,
        jumps: Optional[_LoopJumpSummary] = None,
        ever: Optional[Set[str]] = None,
    ) -> Tuple[Set[str], Set[str], bool, _SuppressProvenance]:
        return _analyze_block_assignment_impl(
            f_body,
            definite=d_set,
            conditional=c_set,
            may_deleted=may_set,
            must_deleted=must_set,
            live_in_deleted=live_del,
            suppress_aliases=final_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            loop_control_terminates=ctx.loop_control_terminates,
            loop_jumps=jumps,
            ever_bound=ever,
            known_safe_names=post_known_safe,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth + 1,
        )

    f_ever: Set[str] = set()
    _, _, f_term, f_out_prov = _run_finally(
        definite,
        conditional,
        may_del,
        must_del,
        live_del=live_in_del,
        ever=f_ever,
    )
    if ctx.ever_bound is not None:
        ctx.ever_bound.update(f_ever)
    f_raw_dels = _filter_scope_names(
        _extract_block_combined_deletions(f_body),
        ctx.active_nonlocals,
        ctx.fn_globals,
    )
    if f_raw_dels & (entry_def - abrupt_entry.definite):
        if ctx.depth < 8:
            abrupt_live: Set[str] = set()
            _run_finally(
                set(abrupt_entry.definite),
                set(abrupt_entry.conditional),
                set(abrupt_entry.may_del),
                live_del=abrupt_live,
            )
            live_in_del.update(abrupt_live - abrupt_entry.pre_try_def)
        else:
            live_in_del.update(
                (f_raw_dels - abrupt_entry.definite) - abrupt_entry.pre_try_def
            )
    if ctx.loop_jumps is not None:
        if has_unescaped_jump:
            if ctx.depth < 8:
                f_jumps = _LoopJumpSummary()
                _run_finally(
                    set(),
                    set(abrupt_entry.conditional),
                    set(abrupt_entry.may_del),
                    jumps=f_jumps,
                )
                if f_jumps.has_jump:
                    ctx.loop_jumps.has_jump = True
                    if f_jumps.has_break:
                        ctx.loop_jumps.has_break = True
                    ctx.loop_jumps.may_bound.update(f_jumps.may_bound)
                    ctx.loop_jumps.may_deleted.update(f_jumps.may_deleted)
            else:
                ctx.loop_jumps.has_jump = True
                if has_brk:
                    ctx.loop_jumps.has_break = True
                ctx.loop_jumps.may_bound.update(
                    abrupt_entry.conditional | definite | conditional | f_ever
                )
                ctx.loop_jumps.may_deleted.update(
                    abrupt_entry.may_del | may_del | f_raw_dels
                )
        if inner_jumps is not None and inner_jumps.has_jump and not f_term:
            ctx.loop_jumps.has_jump = True
            if inner_jumps.has_break:
                ctx.loop_jumps.has_break = True
            if ctx.depth < 8:
                f_standalone_may: Set[str] = set()
                f_standalone_must: Set[str] = set()
                f_standalone_def, f_standalone_cond, _, _ = _run_finally(
                    set(), set(), f_standalone_may, f_standalone_must
                )
                ctx.loop_jumps.may_bound.update(
                    (inner_jumps.may_bound - f_standalone_must)
                    | f_standalone_def
                    | f_standalone_cond
                )
                ctx.loop_jumps.may_deleted.update(
                    (inner_jumps.may_deleted - f_standalone_def) | f_standalone_may
                )
            else:
                ctx.loop_jumps.may_bound.update(
                    inner_jumps.may_bound | definite | conditional | f_ever
                )
                ctx.loop_jumps.may_deleted.update(
                    inner_jumps.may_deleted | may_del | f_raw_dels
                )
    post_f_prov = (
        f_out_prov
        if post_try_prov == final_prov
        else _update_suppress_provenance_from_stmts(
            f_body,
            post_try_prov,
            in_branch=False,
            strict_with_suppression=ctx.strict_with_suppression,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            _depth=ctx.depth + 1,
        )
    )
    if may_suppress_exc:
        return f_term, post_f_prov.merge_branch_conservative(f_out_prov)
    return f_term, _merge_hazard_provenance(post_f_prov, f_out_prov)


def _analyze_try_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    apply_expr_effects: Any,
    _depth: int = 0,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a Try/TryStar statement and updates flow sets in place."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    t_body = list(getattr(stmt, "body", []))
    o_stmts = list(getattr(stmt, "orelse", []))
    f_body = list(getattr(stmt, "finalbody", []))
    raw_try_del = _extract_block_combined_deletions(t_body)
    is_trystar = is_trystar_node(stmt)
    cur_safe = d_in | set(known_safe_names)
    ctx = _TryFlowContext(
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=known_safe_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        depth=_depth,
    )

    split = _compute_try_body_split_provenance(
        t_body,
        local_prov,
        _ProvenanceWalkContext(
            in_branch=False,
            strict_with_suppression=strict_with_suppression,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            depth=_depth,
        ),
        safe_names=cur_safe,
        strict_names=strict_names,
    )

    inner_jumps: Optional[_LoopJumpSummary] = (
        _LoopJumpSummary() if (loop_jumps is not None and f_body) else loop_jumps
    )
    t_may = set(may_in)
    t_must = set(must_in)
    t_live: Set[str] = set()
    t_ever: Set[str] = set()
    t_def, t_cond, t_term, try_body_prov = _analyze_block_assignment_impl(
        t_body,
        definite=set(d_in),
        conditional=set(c_in),
        may_deleted=t_may,
        must_deleted=t_must,
        live_in_deleted=t_live,
        suppress_aliases=local_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=inner_jumps,
        ever_bound=t_ever,
        known_safe_names=cur_safe,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )
    if ever_bound is not None:
        ever_bound.update(t_ever)
    handler_base_prov = split.prefix_prov.merge_branch_conservative(
        split.mid_prov
    ).merge_branch_conservative(try_body_prov)
    live_in_del.update(t_live - d_in)

    o_ever: Set[str] = set()
    succ_term = t_term
    if o_stmts and not t_term:
        o_may = set(t_may)
        o_must = set(t_must)
        o_live: Set[str] = set()
        succ_def, succ_cond, o_term, try_body_prov = _analyze_block_assignment_impl(
            o_stmts,
            definite=set(t_def),
            conditional=set(t_cond),
            may_deleted=o_may,
            must_deleted=o_must,
            live_in_deleted=o_live,
            suppress_aliases=try_body_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            loop_jumps=inner_jumps,
            ever_bound=o_ever,
            known_safe_names=cur_safe | t_def,
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
            _depth=_depth + 1,
        )
        if ever_bound is not None:
            ever_bound.update(o_ever)
        live_in_del.update(o_live - t_def)
        succ_state = _BranchFlowState(succ_def, succ_cond, o_may, o_must)
        if o_term:
            succ_term = True
    else:
        succ_state = _BranchFlowState(t_def, t_cond, t_may, t_must)

    h_entry_def = d_in - (raw_try_del | t_may)
    h_entry = _BranchFlowState(
        definite=h_entry_def,
        conditional=(d_in | c_in | t_ever) - h_entry_def,
        may_del=may_in | raw_try_del | t_may,
        must_del=must_in - t_ever,
    )
    h_res = _analyze_try_except_handlers(
        list(getattr(stmt, "handlers", [])),
        is_trystar=is_trystar,
        h_entry=h_entry,
        succ_state=succ_state,
        succ_term=succ_term,
        try_body_prov=try_body_prov,
        handler_base_prov=handler_base_prov,
        d_in=d_in,
        raw_try_del=raw_try_del,
        live_in_del=live_in_del,
        cur_safe=cur_safe,
        inner_jumps=inner_jumps,
        ctx=ctx,
    )
    _overwrite_flow_state(
        definite,
        conditional,
        may_del,
        must_del,
        _join_branch_flow_states(h_res.target_branches),
    )

    f_term = False
    if f_body:
        abrupt_entry = _compute_try_finally_abrupt_entry(
            t_body,
            o_stmts,
            f_body,
            split=split,
            t_def=t_def,
            t_ever=t_ever,
            o_ever=o_ever,
            d_in=d_in,
            c_in=c_in,
            may_in=may_in,
            raw_try_del=raw_try_del,
            t_may=t_may,
            h_res=h_res,
            local_prov=local_prov,
            cur_safe=cur_safe,
            ctx=ctx,
        )
        f_term, local_prov = _analyze_try_finally_clause(
            f_body,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            h_res.final_prov,
            h_res.post_try_prov,
            abrupt_entry=abrupt_entry,
            inner_jumps=inner_jumps,
            ctx=ctx,
        )
    else:
        local_prov = h_res.post_try_prov

    return f_term or h_res.pre_finally_term, local_prov


def _analyze_match_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
    known_safe_names: AbstractSet[str],
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    apply_expr_effects: Any,
    _depth: int = 0,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a Match statement and updates flow sets in place."""
    subject = getattr(stmt, "subject", None)
    if isinstance(subject, ast.AST):
        local_prov = _advance_prov_through_opt_expr(
            subject,
            local_prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        apply_expr_effects(subject)
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    cur_safe = d_in | set(known_safe_names)

    surviving: List[_BranchFlowState] = []
    all_cases: List[_BranchFlowState] = []
    prior_guard_cond: Set[str] = set()
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
        if ever_bound is not None:
            ever_bound.update(case_init_def | c_g)
        case_prov = local_prov
        for p_s in raw_pattern_stores:
            case_prov = case_prov.clear_name(p_s)
        case_prov = _advance_clause_entry_prov(
            case_prov,
            guard,
            (),
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )
        c_def_in = d_in | case_init_def
        c_cond_in = (c_in | prior_guard_cond | c_g) - c_def_in
        c_may = may_in - case_init_def
        c_must = must_in - case_init_def - c_g - prior_guard_cond
        c_live: Set[str] = set()
        case_body = getattr(case, "body", [])
        c_def, c_cond, c_term, c_out_prov = _analyze_block_assignment_impl(
            case_body,
            definite=c_def_in,
            conditional=c_cond_in,
            may_deleted=c_may,
            must_deleted=c_must,
            live_in_deleted=c_live,
            suppress_aliases=case_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_control_terminates=loop_control_terminates,
            loop_jumps=loop_jumps,
            ever_bound=ever_bound,
            known_safe_names=cur_safe | case_init_def,
            strict_names=strict_names,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
            strict_with_suppression=strict_with_suppression,
            _depth=_depth + 1,
        )
        merged_match_prov = merged_match_prov.merge_branch_conservative(c_out_prov)
        live_in_del.update((c_live - case_init_def) - d_in)
        c_state = _BranchFlowState(c_def, c_cond, c_may, c_must)
        all_cases.append(c_state)
        if not c_term:
            surviving.append(c_state)
        if guard is not None:
            prior_guard_cond.update(case_init_def | c_g)
        if _is_irrefutable_case(case):
            has_irrefutable_default = True
            break
        local_prov = _merge_prior_clause_guard_and_binds(
            local_prov,
            guard,
            raw_pattern_stores,
            extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        )

    local_prov = (
        merged_match_prov
        if has_irrefutable_default
        else local_prov.merge_branch_conservative(merged_match_prov)
    )
    if not has_irrefutable_default:
        fallthrough_state = _BranchFlowState(
            definite=d_in,
            conditional=(c_in | prior_guard_cond) - d_in,
            may_del=may_in,
            must_del=must_in - prior_guard_cond,
        )
        surviving.append(fallthrough_state)
        all_cases.append(fallthrough_state)

    if surviving:
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(surviving)
        )
        return False, local_prov
    if all_cases:
        _overwrite_flow_state(
            definite, conditional, may_del, must_del, _join_branch_flow_states(all_cases)
        )
        return True, local_prov
    return False, local_prov


def _merge_loop_carried_suppress_provenance(
    body_stmts: Sequence[ast.AST],
    entry_prov: _SuppressProvenance,
    *,
    strict_with_suppression: bool = False,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    _depth: int = 0,
) -> _SuppressProvenance:
    """Conservatively merges loop-body provenance hazards back into loop-entry provenance."""
    if not body_stmts:
        return entry_prov
    body_carried_mid = _update_prov_and_term_from_stmts(
        body_stmts,
        entry_prov,
        in_branch=True,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth,
    ).mid_prov
    return entry_prov.merge_branch_conservative(body_carried_mid)


def _analyze_loop_stmt_assignment(
    body_stmts: Sequence[ast.AST],
    orelse_stmts: Sequence[ast.AST],
    target_raw_names: AbstractSet[str],
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    body_prov: Optional[_SuppressProvenance] = None,
    active_nonlocals: Optional[Set[str]],
    fn_globals: AbstractSet[str],
    loop_control_terminates: bool,
    loop_jumps: Optional[_LoopJumpSummary],
    ever_bound: Optional[Set[str]],
    known_safe_names: AbstractSet[str],
    strict_names: bool,
    extra_non_suppressing_ctx_managers: AbstractSet[str],
    strict_with_suppression: bool,
    _depth: int = 0,
) -> Tuple[bool, _SuppressProvenance]:
    """Analyzes a While/For/AsyncFor loop with break/continue jump tracking and orelse composition."""
    d_in = set(definite)
    c_in = set(conditional)
    may_in = set(may_del)
    must_in = set(must_del)
    target_filt = _filter_scope_names(target_raw_names, active_nonlocals, fn_globals)
    if ever_bound is not None and target_filt:
        ever_bound.update(target_filt)

    body_jumps = _LoopJumpSummary()
    b_def_in = d_in | target_filt
    b_cond_in = c_in - b_def_in
    b_may = may_in - target_filt
    b_must = must_in - target_filt
    b_live: Set[str] = set()
    eff_body_prov = body_prov if body_prov is not None else local_prov
    loop_entry_prov = _merge_loop_carried_suppress_provenance(
        body_stmts,
        eff_body_prov,
        strict_with_suppression=strict_with_suppression,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        _depth=_depth + 1,
    )
    b_def, b_cond, _, b_prov = _analyze_block_assignment_impl(
        body_stmts,
        definite=b_def_in,
        conditional=b_cond_in,
        may_deleted=b_may,
        must_deleted=b_must,
        live_in_deleted=b_live,
        suppress_aliases=loop_entry_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=True,
        loop_jumps=body_jumps,
        ever_bound=ever_bound,
        known_safe_names=d_in | set(known_safe_names) | target_raw_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )
    live_in_del.update((b_live - target_raw_names) - d_in)
    merged_prov = local_prov.merge_branch_conservative(b_prov)

    iter_dels = (b_may - may_in) | body_jumps.may_deleted
    iter_bounds = target_filt | b_def | b_cond | body_jumps.may_bound
    exh_def = d_in - iter_dels
    exh_state = _BranchFlowState(
        definite=exh_def,
        conditional=(d_in | c_in | iter_bounds) - exh_def,
        may_del=may_in | iter_dels,
        must_del=must_in - iter_bounds,
    )
    if not orelse_stmts:
        _overwrite_flow_state(definite, conditional, may_del, must_del, exh_state)
        return False, merged_prov

    o_may = set(exh_state.may_del)
    o_must = set(exh_state.must_del)
    o_live: Set[str] = set()
    o_def, o_cond, o_term, o_prov = _analyze_block_assignment_impl(
        orelse_stmts,
        definite=set(exh_state.definite),
        conditional=set(exh_state.conditional),
        may_deleted=o_may,
        must_deleted=o_must,
        live_in_deleted=o_live,
        suppress_aliases=merged_prov,
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=exh_state.definite | set(known_safe_names),
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        _depth=_depth + 1,
    )
    live_in_del.update(o_live - exh_state.definite)
    merged_prov = merged_prov.merge_branch_conservative(o_prov)
    after_orelse = _BranchFlowState(o_def, o_cond, o_may, o_must)

    return _finalize_surviving_branch_flow(
        definite,
        conditional,
        may_del,
        must_del,
        ((not o_term, after_orelse), (body_jumps.has_break, exh_state)),
        after_orelse,
        merged_prov,
    )


def _analyze_simple_non_terminating_stmt(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    bind_def_cond: Callable[[Iterable[str], Iterable[str]], None],
    apply_expr_effects: Any,
) -> _SuppressProvenance:
    """Updates provenance, definite/conditional bindings, and deletions for non-compound stmts."""
    next_prov = _update_suppress_provenance_from_stmt(
        stmt,
        local_prov,
        strict_with_suppression=ctx.strict_with_suppression,
        extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        _depth=ctx.depth,
    )
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            bind_def_cond(_extract_definite_target_names(target), ())
            apply_expr_effects(target)
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.AnnAssign):
        apply_expr_effects(stmt.annotation, is_cond=True)
        if stmt.value is not None:
            bind_def_cond(_extract_definite_target_names(stmt.target), ())
            apply_expr_effects(stmt.target)
            apply_expr_effects(stmt.value)
        elif not isinstance(stmt.target, ast.Name):
            apply_expr_effects(stmt.target)
    elif isinstance(stmt, ast.AugAssign):
        bind_def_cond(_extract_definite_target_names(stmt.target), ())
        apply_expr_effects(stmt.target)
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.Expr):
        apply_expr_effects(stmt.value)
    elif isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            apply_expr_effects(target)
            for node in ast.walk(target):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Del)
                    and _can_bind_in_fn_scope(
                        node.id, ctx.active_nonlocals, ctx.fn_globals
                    )
                ):
                    if node.id not in definite:
                        live_in_del.add(node.id)
                    definite.discard(node.id)
                    conditional.discard(node.id)
                    may_del.add(node.id)
                    must_del.add(node.id)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for hdr in iter_nested_scope_header_nodes(stmt):
            apply_expr_effects(hdr)
        bind_def_cond((stmt.name,), ())
    elif is_type_alias_node(stmt):  # pragma: no cover (py312+)
        ta_name = _extract_type_alias_or_pattern_name(stmt)
        if ta_name is not None:
            bind_def_cond((ta_name,), ())
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        imp_names: Set[str] = set()
        _record_import_names(stmt, imp_names)
        bind_def_cond(imp_names, ())
    elif isinstance(stmt, ast.Assert):
        apply_expr_effects(stmt.test)
        if stmt.msg is not None:
            apply_expr_effects(stmt.msg, is_cond=True)
    return next_prov


def _analyze_compound_stmt_assignment(
    stmt: ast.AST,
    definite: Set[str],
    conditional: Set[str],
    may_del: Set[str],
    must_del: Set[str],
    live_in_del: Set[str],
    local_prov: _SuppressProvenance,
    *,
    ctx: _TryFlowContext,
    bind_def_cond: Callable[[Iterable[str], Iterable[str]], None],
    apply_expr_effects: Any,
) -> Tuple[bool, _SuppressProvenance, bool]:
    """Dispatches compound statements (If/While/For/With/Try/Match); returns (term, prov, handled)."""
    if isinstance(stmt, ast.If):
        local_prov = _advance_prov_through_opt_expr(
            stmt.test,
            local_prov,
            allow_positive=True,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
        )
        apply_expr_effects(stmt.test)
        term, local_prov = _analyze_if_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            loop_control_terminates=ctx.loop_control_terminates,
            loop_jumps=ctx.loop_jumps,
            ever_bound=ctx.ever_bound,
            known_safe_names=ctx.known_safe_names,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth,
        )
        return term, local_prov, True

    if isinstance(stmt, (ast.While, ast.For, ast.AsyncFor)):
        if isinstance(stmt, ast.While):
            local_prov = _advance_prov_through_opt_expr(
                stmt.test,
                local_prov,
                allow_positive=False,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
            apply_expr_effects(stmt.test)
            loop_targets: AbstractSet[str] = frozenset()
            loop_body_prov = local_prov
        else:
            local_prov, loop_body_prov = _advance_prov_through_for_header(
                stmt,
                local_prov,
                allow_positive=True,
                extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            )
            apply_expr_effects(stmt.iter)
            apply_expr_effects(stmt.target, is_cond=True)
            loop_targets = _extract_definite_target_names(stmt.target)
        term, local_prov = _analyze_loop_stmt_assignment(
            stmt.body,
            stmt.orelse,
            loop_targets,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            body_prov=loop_body_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            loop_control_terminates=ctx.loop_control_terminates,
            loop_jumps=ctx.loop_jumps,
            ever_bound=ctx.ever_bound,
            known_safe_names=ctx.known_safe_names,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth,
        )
        return term, local_prov, True

    extra_kwargs: Dict[str, Any] = {"apply_expr_effects": apply_expr_effects}
    stmt_handler: Optional[Callable[..., Tuple[bool, _SuppressProvenance]]] = None
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        stmt_handler = _analyze_with_stmt_assignment
        extra_kwargs["bind_def_cond"] = bind_def_cond
    elif is_try_or_trystar_node(stmt):
        stmt_handler = _analyze_try_stmt_assignment
    elif is_match_node(stmt):  # pragma: no cover (py310+)
        stmt_handler = _analyze_match_stmt_assignment
    if stmt_handler is not None:
        term, local_prov = stmt_handler(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            active_nonlocals=ctx.active_nonlocals,
            fn_globals=ctx.fn_globals,
            loop_control_terminates=ctx.loop_control_terminates,
            loop_jumps=ctx.loop_jumps,
            ever_bound=ctx.ever_bound,
            known_safe_names=ctx.known_safe_names,
            strict_names=ctx.strict_names,
            extra_non_suppressing_ctx_managers=ctx.extra_non_suppressing_ctx_managers,
            strict_with_suppression=ctx.strict_with_suppression,
            _depth=ctx.depth,
            **extra_kwargs,
        )
        return term, local_prov, True

    return False, local_prov, False


def _analyze_block_assignment_impl(
    statements: Sequence[ast.AST],
    definite: Optional[Set[str]] = None,
    conditional: Optional[Set[str]] = None,
    *,
    may_deleted: Optional[Set[str]] = None,
    must_deleted: Optional[Set[str]] = None,
    live_in_deleted: Optional[Set[str]] = None,
    suppress_aliases: Union[
        _SuppressProvenance, Iterable[str]
    ] = _EMPTY_SUPPRESS_PROVENANCE,
    active_nonlocals: Optional[Set[str]] = None,
    fn_globals: AbstractSet[str] = frozenset(),
    loop_control_terminates: bool = False,
    loop_jumps: Optional[_LoopJumpSummary] = None,
    ever_bound: Optional[Set[str]] = None,
    known_safe_names: AbstractSet[str] = frozenset(),
    strict_names: bool = True,
    extra_non_suppressing_ctx_managers: AbstractSet[str] = frozenset(),
    strict_with_suppression: bool = False,
    _depth: int = 0,
) -> Tuple[Set[str], Set[str], bool, _SuppressProvenance]:
    """Unified single-pass analyzer for definite/conditional bindings, deletions, and termination."""
    if definite is None:
        definite = set()
    if conditional is None:
        conditional = set()
    may_del = may_deleted if may_deleted is not None else set()
    must_del = must_deleted if must_deleted is not None else set()
    live_in_del = live_in_deleted if live_in_deleted is not None else set()
    local_prov = _coerce_suppress_provenance(suppress_aliases)
    if _depth > _MAX_AST_RECURSION_DEPTH:
        return _apply_depth_bailout_block_flow(
            statements,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            active_nonlocals=active_nonlocals,
            fn_globals=fn_globals,
            loop_jumps=loop_jumps,
            ever_bound=ever_bound,
        )

    def _bind_def_cond(d_raw: Iterable[str], c_raw: Iterable[str] = ()) -> None:
        d_filt = _filter_scope_names(d_raw, active_nonlocals, fn_globals)
        c_filt = _filter_scope_names(c_raw, active_nonlocals, fn_globals)
        if d_filt:
            definite.update(d_filt)
            conditional.difference_update(d_filt)
            may_del.difference_update(d_filt)
            must_del.difference_update(d_filt)
            if ever_bound is not None:
                ever_bound.update(d_filt)
        if c_filt:
            conditional.update(c_filt - definite)
            must_del.difference_update(c_filt)
            if ever_bound is not None:
                ever_bound.update(c_filt)

    def _apply_expr_effects(
        expr_node: Optional[ast.AST], *, is_cond: bool = False
    ) -> None:
        if not isinstance(expr_node, ast.AST):
            return
        d_expr, c_expr = _walrus_assignment_in_expr(expr_node, is_conditional=is_cond)
        _bind_def_cond(d_expr, c_expr)

    ctx = _TryFlowContext(
        active_nonlocals=active_nonlocals,
        fn_globals=fn_globals,
        loop_control_terminates=loop_control_terminates,
        loop_jumps=loop_jumps,
        ever_bound=ever_bound,
        known_safe_names=known_safe_names,
        strict_names=strict_names,
        extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
        strict_with_suppression=strict_with_suppression,
        depth=_depth,
    )

    for stmt in statements:
        before_def = set(definite)
        if isinstance(stmt, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
            local_prov = _update_suppress_provenance_from_stmt(
                stmt,
                local_prov,
                strict_with_suppression=strict_with_suppression,
                extra_non_suppressing_ctx_managers=extra_non_suppressing_ctx_managers,
                _depth=_depth,
            )
            if isinstance(stmt, ast.Raise):
                _apply_expr_effects(stmt.exc)
                _apply_expr_effects(stmt.cause)
                term_flag = True
            elif isinstance(stmt, ast.Return):
                _apply_expr_effects(stmt.value)
                term_flag = True
            else:
                if loop_jumps is not None:
                    loop_jumps.has_jump = True
                    if isinstance(stmt, ast.Break):
                        loop_jumps.has_break = True
                    loop_jumps.may_bound.update(definite | conditional)
                    loop_jumps.may_deleted.update(may_del)
                term_flag = bool(loop_control_terminates)
            may_del.difference_update(definite - before_def)
            return definite, conditional - definite, term_flag, local_prov

        c_term, local_prov, is_compound = _analyze_compound_stmt_assignment(
            stmt,
            definite,
            conditional,
            may_del,
            must_del,
            live_in_del,
            local_prov,
            ctx=ctx,
            bind_def_cond=_bind_def_cond,
            apply_expr_effects=_apply_expr_effects,
        )
        if is_compound:
            if c_term:
                may_del.difference_update(definite - before_def)
                return definite, conditional - definite, True, local_prov
        else:
            local_prov = _analyze_simple_non_terminating_stmt(
                stmt,
                definite,
                conditional,
                may_del,
                must_del,
                live_in_del,
                local_prov,
                ctx=ctx,
                bind_def_cond=_bind_def_cond,
                apply_expr_effects=_apply_expr_effects,
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
