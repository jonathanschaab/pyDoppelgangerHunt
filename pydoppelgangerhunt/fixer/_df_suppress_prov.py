"""Context-manager suppression provenance dataclass, stem rules, and rebound checks."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import re
from typing import (
    AbstractSet,
    Any,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydoppelgangerhunt.fixer._df_exit_protocol import (
    _STDLIB_NON_SUPPRESSING_MODULES,
    is_factory_rebound_target,
)

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


class _SuspiciousStemRule(NamedTuple):
    """Declarative rule for one suspicious exception-suppressing context manager stem."""

    stem: str
    is_prefix: bool = False
    split: Tuple[str, ...] = ()
    inflections: Tuple[str, ...] = ()


# INVARIANT (CONVERGENCE BOUNDARY — CLOSED HEURISTIC TABLE):
# `_SUSPICIOUS_SUPPRESS_SPECS` is a single frozen, closed name-heuristic table
# used ONLY in `strict` mode to flag common exception-suppressing context
# manager naming idioms. Naming heuristics cannot enumerate every custom
# context-manager vocabulary and must NOT be expanded incrementally for
# individual third-party APIs; callers requiring fail-closed guarantees on
# unproven context managers must use `closure_strictness = "pedantic"` (the
# default for `--patch`) or configure `non_suppressing_ctx_managers`.
_SUSPICIOUS_SUPPRESS_SPECS: Tuple[_SuspiciousStemRule, ...] = (
    _SuspiciousStemRule("suppress", is_prefix=True),
    _SuspiciousStemRule("raises"),
    _SuspiciousStemRule("ignor", is_prefix=True),
    _SuspiciousStemRule("subtest", split=("sub", "test"), inflections=("subtests",)),
    _SuspiciousStemRule("exitstack", split=("exit", "stack"), inflections=("exitstacks",)),
    _SuspiciousStemRule("enter_context"),
    _SuspiciousStemRule("enter_async_context"),
    _SuspiciousStemRule("cancelscope", split=("cancel", "scope"), inflections=("cancelscopes",)),
    _SuspiciousStemRule("move_on_after"),
    _SuspiciousStemRule("fail_after"),
    _SuspiciousStemRule("catch", inflections=("catches", "catching", "catcher")),
    _SuspiciousStemRule("swallow", is_prefix=True),
    _SuspiciousStemRule("silen", is_prefix=True),
    _SuspiciousStemRule("quiet", is_prefix=True),
    _SuspiciousStemRule(
        "toler",
        inflections=("tolerate", "tolerates", "tolerated", "tolerating", "tolerant"),
    ),
    _SuspiciousStemRule("best_effort"),
    _SuspiciousStemRule("absorb", is_prefix=True),
    _SuspiciousStemRule("trap", inflections=("traps", "trapped", "trapping", "trapper")),
    _SuspiciousStemRule("fallback", split=("fall", "back"), inflections=("fallbacks",)),
    _SuspiciousStemRule("noop", split=("no", "op"), inflections=("noops",)),
    _SuspiciousStemRule("no_op"),
    _SuspiciousStemRule("shield", inflections=("shields", "shielded", "shielding")),
    _SuspiciousStemRule("retry", inflections=("retries", "retrying", "retried", "retrier")),
    _SuspiciousStemRule("attempt", inflections=("attempts", "attempted", "attempting")),
    _SuspiciousStemRule("skip", inflections=("skips", "skipped", "skipping", "skipper")),
    _SuspiciousStemRule("mute", inflections=("mutes", "muted", "muter", "muters")),
    _SuspiciousStemRule("muting"),
    _SuspiciousStemRule("maybe"),
    _SuspiciousStemRule("hush", inflections=("hushes", "hushed", "hushing")),
    _SuspiciousStemRule("discard", inflections=("discards", "discarded", "discarding")),
    _SuspiciousStemRule("forgiv", is_prefix=True),
    _SuspiciousStemRule("squelch", is_prefix=True),
    _SuspiciousStemRule("quench", is_prefix=True),
    _SuspiciousStemRule("mask", inflections=("masks", "masking", "masker", "maskers")),
    _SuspiciousStemRule("handl", is_prefix=True),
    _SuspiciousStemRule("safe", inflections=("safely", "safeguard")),
    _SuspiciousStemRule("failsafe", split=("fail", "safe"), inflections=("failsafes",)),
    _SuspiciousStemRule(
        "rescue", inflections=("rescues", "rescued", "rescuing", "rescuer", "rescuers")
    ),
    _SuspiciousStemRule(
        "recover",
        inflections=("recovers", "recovered", "recovering", "recovery", "recoverer"),
    ),
    _SuspiciousStemRule("allow", inflections=("allows", "allowed", "allowing")),
    _SuspiciousStemRule("continue_on"),
)


_SUSPICIOUS_SUPPRESS_SUBSTRINGS: Tuple[str, ...] = tuple(
    spec.stem for spec in _SUSPICIOUS_SUPPRESS_SPECS
)


_SUSPICIOUS_PREFIX_STEMS: Tuple[str, ...] = tuple(
    spec.stem for spec in _SUSPICIOUS_SUPPRESS_SPECS if spec.is_prefix
)


_IDENT_WORD_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+")


def _build_suspicious_token_tables() -> Tuple[frozenset[str], Tuple[Tuple[str, ...], ...]]:
    """Derives exact single-token and multi-token match tables from _SUSPICIOUS_SUPPRESS_SPECS."""
    exact_tokens: Set[str] = set()
    token_seqs: List[Tuple[str, ...]] = []
    seen_seqs: Set[Tuple[str, ...]] = set()
    for spec in _SUSPICIOUS_SUPPRESS_SPECS:
        if "_" in spec.stem:
            seq = tuple(spec.stem.split("_"))
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                token_seqs.append(seq)
            continue
        if not spec.is_prefix:
            exact_tokens.add(spec.stem)
            exact_tokens.update(spec.inflections)
        if spec.split and spec.split not in seen_seqs:
            seen_seqs.add(spec.split)
            token_seqs.append(spec.split)
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
        """Returns mod_name plus any local aliases bound to mod_name in stdlib_modules or imported_symbols."""
        names: Set[str] = {mod_name}
        for alias, bound_mod in self.stdlib_modules:
            if bound_mod == mod_name:
                names.add(alias)
        for alias, bound_qual in self.imported_symbols:
            if bound_qual == mod_name:
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
        if not is_suppressor and (
            self.has_exit_patch or name in self.exit_patch_targets
        ):
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

    def with_suspicious_alias(
        self,
        name: str,
        *,
        allow_positive: bool = False,
        imported_qualname: Optional[str] = None,
    ) -> _SuppressProvenance:
        """Returns provenance after binding name to an unproven suspicious-named manager."""
        return self._bind_provenance_entry(
            name,
            allow_positive=allow_positive,
            is_suspicious_alias=True,
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
        if not has_patch and not rebound_attrs and not patched_names:
            return self
        merged_rebound = self.rebound_factory_targets | frozenset(rebound_attrs)
        merged_patched = self.exit_patch_targets | frozenset(patched_names)
        if not has_patch:
            updated = replace(
                self,
                exit_patch_targets=merged_patched,
                rebound_factory_targets=merged_rebound,
            )
            invalidated = frozenset(
                n
                for n in (updated.non_suppressors | updated.non_suppressor_instances)
                if n in patched_names or _has_rebound_imported_factory(n, updated)
            )
            if invalidated:
                updated = replace(
                    updated,
                    non_suppressors=updated.non_suppressors - invalidated,
                    non_suppressor_instances=(
                        updated.non_suppressor_instances - invalidated
                    ),
                )
            for name in sorted(patched_names):
                if name not in updated.suppressors:
                    updated = updated.with_suspicious_alias(name)
            return updated
        demoted = replace(
            self,
            non_suppressors=frozenset(),
            non_suppressor_instances=frozenset(),
            exit_patch_targets=merged_patched,
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
        cleared_pos = mutated | suppressor_targets
        return replace(
            self,
            suppressors=self.suppressors | frozenset(suppressor_targets),
            non_suppressors=self.non_suppressors - cleared_pos,
            stdlib_modules=frozenset(
                pair for pair in self.stdlib_modules if pair[0] not in mutated
            ),
            shadowed_builtins=self.shadowed_builtins | shad_b,
            imported_symbols=frozenset(
                pair for pair in self.imported_symbols if pair[0] not in mutated
            ),
            shadowed_names=self.shadowed_names | frozenset(mutated),
            non_suppressor_instances=self.non_suppressor_instances - cleared_pos,
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
        merged_sup = self.suppressors | branch_prov.suppressors
        return _SuppressProvenance(
            suppressors=merged_sup,
            non_suppressors=(
                (self.non_suppressors & branch_prov.non_suppressors) - merged_sup
            ),
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
                (self.non_suppressor_instances & branch_prov.non_suppressor_instances)
                - merged_sup
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
        cleared_pos = shadowed_names | merged_sup
        star_demoted = (
            self._positive_binding_names() - merged_sup if has_star else frozenset()
        )
        return replace(
            self,
            suppressors=merged_sup,
            non_suppressors=(
                frozenset() if has_star else self.non_suppressors - cleared_pos
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
                else self.non_suppressor_instances - cleared_pos
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


def _has_rebound_parent_package(
    mod_qual: str,
    prov: _SuppressProvenance,
) -> bool:
    """Returns True when any enclosing package prefix of mod_qual was rebound or replaced."""
    cur_mod = mod_qual
    while "." in cur_mod:
        parent_mod, child_attr = cur_mod.rsplit(".", 1)
        if parent_mod and prov.is_factory_rebound(
            prov.module_aliases_for(parent_mod),
            child_attr,
            include_untargeted=False,
        ):
            return True
        cur_mod = parent_mod
    return False


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
            mod_targets = (
                {qual} if not imp_mod.strip(".") else prov.module_aliases_for(imp_mod)
            )
            if prov.is_factory_rebound(
                mod_targets,
                imp_attr,
                include_untargeted=eff_untargeted,
            ) or _has_rebound_parent_package(imp_mod, prov):
                return True
    return False
