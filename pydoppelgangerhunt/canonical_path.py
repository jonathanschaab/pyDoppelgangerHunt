"""Unified Canonical Path Model for pyDoppelgangerHunt.

Provides immutable path representations and worktree-scoped coordinate resolution
distinguishing lexical identity from physical identity across Git repositories,
subdirectories, differential scans, and baseline caches.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import unicodedata


def normalize_lexical_posix(path_str: Optional[str], strip_anchor: bool = False) -> str:
    """Normalizes a path string to forward slashes with drive and anchor handling.

    Args:
        path_str: Raw path string or stringifiable object.
        strip_anchor: When True, strips trailing `#anchor` (e.g. notebook cell fragments).
            When False, preserves `#` characters in filesystem directory/file names.

    Returns:
        Lexically normalized forward-slash path string.
    """
    if path_str is None:
        return ""
    raw = str(path_str).rstrip("\r\n")
    if not raw or not raw.strip():
        return ""

    if "\x00" in raw:
        raw = raw.replace("\x00", "")
    if not raw:
        return ""

    if strip_anchor and "#" in raw:
        last_seg = raw.replace("\\", "/").rsplit("/", 1)[-1]
        if "#" in last_seg:
            fname, fragment = last_seg.split("#", 1)
            if (
                ("." in fname and not fname.startswith("."))
                or fragment.lower().startswith("cell")
                or ".ipynb#" in raw
            ):
                raw = raw[: len(raw) - len(fragment) - 1]

    norm = raw.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]

    if sys.platform == "darwin":
        norm = unicodedata.normalize("NFC", norm)

    # Normalize Windows drive letters (e.g. c:/ -> C:/)
    if len(norm) >= 2 and norm[1] == ":" and norm[0].isalpha():
        norm = norm[0].upper() + norm[1:]

    return norm


def _resolve_relative_segments(rel: str) -> Optional[str]:
    """Resolves '.' and '..' segments in a relative path, returning None on parent escape."""
    rel_parts = rel.split("/")
    if ".." not in rel_parts and "." not in rel_parts:
        return rel
    resolved_segments: List[str] = []
    for seg in rel_parts:
        if not seg or seg == ".":
            continue
        if seg == "..":
            if not resolved_segments:
                return None
            resolved_segments.pop()
        else:
            resolved_segments.append(seg)
    return "/".join(resolved_segments)


def lexical_relative_to(
    path_posix: str,
    base_posix: str,
    case_fold: bool = False,
) -> Optional[str]:
    """Computes pure lexical relative path without requiring filesystem resolution.

    Args:
        path_posix: Normalized posix path string.
        base_posix: Normalized posix base directory string.
        case_fold: When True, performs case-insensitive segment comparisons.

    Returns:
        Posix relative path string if path_posix is located under base_posix, else None.
    """
    if not path_posix:
        return None

    p_clean = path_posix.rstrip("/")
    b_clean = base_posix.rstrip("/")

    # Handle filesystem root base ("/") as a special case
    if base_posix and set(base_posix) == {"/"}:
        if not p_clean:
            return ""
        if len(p_clean) >= 2 and p_clean[1] == ":":
            return None
        return _resolve_relative_segments(p_clean.lstrip("/"))

    if not b_clean:
        return p_clean if not p_clean.startswith("/") else None

    p_compare = p_clean.lower() if case_fold else p_clean
    b_compare = b_clean.lower() if case_fold else b_clean

    if p_compare == b_compare:
        return ""

    prefix = b_compare + "/"
    if p_compare.startswith(prefix):
        return _resolve_relative_segments(p_clean[len(prefix):])

    return None


@dataclass(frozen=True)
class CanonicalPath:
    """Immutable representation of a path across multiple coordinate systems.

    Attributes:
        raw: Original path string as provided.
        repo_relative: Path relative to Git worktree root (if inside repo), posix-normalized.
        target_relative: Path relative to scan target root (if inside target), posix-normalized.
        absolute_lexical: Normalized absolute path string without resolving symlinks.
    """

    raw: str
    repo_relative: Optional[str] = None
    target_relative: Optional[str] = None
    absolute_lexical: Optional[str] = None

    @property
    def display_path(self) -> str:
        """Returns the most human-readable relative path available."""
        return self.target_relative or self.repo_relative or self.raw

    def __str__(self) -> str:
        return self.display_path


MAX_RESOLVER_CACHE_ENTRIES: int = 50_000


class CanonicalPathResolver:
    """Resolves and compares paths across Git repository root and target directory boundaries.

    Encapsulates worktree-scoped coordinate translation and memoized equivalence checks,
    preventing ambiguous basename collisions and suffix matching errors.
    """

    def __init__(
        self,
        target_root: Union[str, Path],
        repo_root: Optional[Union[str, Path]] = None,
        case_fold: Optional[bool] = None,
    ) -> None:
        """Initializes resolver with target directory and optional repo root.

        Args:
            target_root: Directory being scanned.
            repo_root: Git worktree root directory. If omitted or None, defaults to target_root.
            case_fold: Whether to fold case during comparisons. If None, defaults to True
                on Windows and False on other platforms.
        """
        self.target_path = Path(target_root)
        try:
            self.target_resolved = self.target_path.resolve()
        except (ValueError, OSError, RuntimeError):
            self.target_resolved = self.target_path
        if self.target_resolved.is_file():
            self.target_resolved = self.target_resolved.parent
            self.target_path = self.target_path.parent

        if repo_root is not None:
            self.repo_path: Optional[Path] = Path(repo_root)
            try:
                self.repo_resolved: Optional[Path] = self.repo_path.resolve()
            except (ValueError, OSError, RuntimeError):
                self.repo_resolved = self.repo_path
        else:
            self.repo_path = self.target_path
            self.repo_resolved = self.target_resolved

        if case_fold is None:
            self.case_fold = os.name == "nt" or sys.platform == "win32"
        else:
            self.case_fold = bool(case_fold)

        self.target_lexical = normalize_lexical_posix(str(self.target_resolved))
        self.repo_lexical = normalize_lexical_posix(str(self.repo_resolved))

        # Determine target's repo-relative offset if target is inside repo
        self.target_in_repo = lexical_relative_to(
            self.target_lexical, self.repo_lexical, case_fold=self.case_fold
        )

        self._cache: Dict[Tuple[str, str, bool], CanonicalPath] = {}

    def _cache_set(self, key: Tuple[str, str, bool], value: CanonicalPath) -> None:
        if len(self._cache) >= MAX_RESOLVER_CACHE_ENTRIES:
            self._cache.clear()
        self._cache[key] = value

    def resolve(
        self,
        path: Union[str, Path, CanonicalPath],
        basis: str = "auto",
        strip_anchor: bool = False,
    ) -> CanonicalPath:
        """Resolves raw path into a CanonicalPath dataclass.

        Args:
            path: Input path string, Path object, or existing CanonicalPath.
            basis: Coordinate basis for relative paths:
                - "auto": infers basis from path format and prefixes
                - "target" / "target_relative": relative to target directory
                - "repo" / "repo_relative": relative to repository worktree root
            strip_anchor: Whether to strip `#fragment` anchors.

        Returns:
            Resolved CanonicalPath.
        """
        if isinstance(path, CanonicalPath):
            return path

        raw_str = str(path).rstrip("\r\n")
        cache_key = (raw_str, basis, bool(strip_anchor))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        norm = normalize_lexical_posix(raw_str, strip_anchor=strip_anchor)
        if not norm:
            res = CanonicalPath(raw=raw_str, repo_relative="", target_relative="", absolute_lexical="")
            self._cache_set(cache_key, res)
            return res

        p = Path(norm)
        is_abs = (
            p.is_absolute()
            or norm.startswith("/")
            or (len(norm) >= 3 and norm[1] == ":" and norm[2] == "/")
        )

        if is_abs:
            abs_lex = norm
            if (
                os.name == "nt"
                and abs_lex.startswith("/")
                and not (len(abs_lex) >= 2 and abs_lex[1] == ":")
                and len(self.target_lexical) >= 2
                and self.target_lexical[1] == ":"
            ):
                abs_lex = self.target_lexical[:2] + abs_lex
            rel_target = lexical_relative_to(abs_lex, self.target_lexical, case_fold=self.case_fold)
            rel_repo = lexical_relative_to(abs_lex, self.repo_lexical, case_fold=self.case_fold)
            res = CanonicalPath(
                raw=raw_str,
                repo_relative=rel_repo,
                target_relative=rel_target,
                absolute_lexical=abs_lex,
            )
            self._cache_set(cache_key, res)
            return res

        # Handle relative path
        if basis in ("repo", "repo_relative"):
            repo_rel = norm
            if self.target_in_repo is not None:
                if self.target_in_repo:
                    target_rel = lexical_relative_to(
                        norm, self.target_in_repo, case_fold=self.case_fold
                    )
                else:
                    target_rel = norm
            else:
                target_rel = None
            abs_lex = f"{self.repo_lexical}/{norm}".rstrip("/")
        elif basis in ("target", "target_relative"):
            target_rel = norm
            if self.target_in_repo is not None:
                repo_rel = f"{self.target_in_repo}/{norm}" if self.target_in_repo else norm
            else:
                repo_rel = None
            abs_lex = f"{self.target_lexical}/{norm}".rstrip("/")
        else:
            # "auto" basis
            target_in_repo = self.target_in_repo
            if target_in_repo is not None:
                norm_cmp = norm.lower() if self.case_fold else norm
                target_in_repo_cmp = target_in_repo.lower() if self.case_fold else target_in_repo
                if target_in_repo_cmp and (
                    norm_cmp == target_in_repo_cmp
                    or norm_cmp.startswith(target_in_repo_cmp + "/")
                ):
                    # Path includes target_in_repo prefix; treat as repo-relative
                    repo_rel = norm
                    target_rel = lexical_relative_to(
                        norm, target_in_repo, case_fold=self.case_fold
                    )
                    abs_lex = f"{self.repo_lexical}/{norm}".rstrip("/")
                else:
                    # Default relative path from scanner/units is target-relative
                    target_rel = norm
                    repo_rel = f"{target_in_repo}/{norm}" if target_in_repo else norm
                    abs_lex = f"{self.target_lexical}/{norm}".rstrip("/")
            else:
                target_rel = norm
                repo_rel = None
                abs_lex = f"{self.target_lexical}/{norm}".rstrip("/")

        res = CanonicalPath(
            raw=raw_str,
            repo_relative=repo_rel,
            target_relative=target_rel,
            absolute_lexical=abs_lex,
        )
        self._cache_set(cache_key, res)
        return res

    def _coordinate_key(
        self,
        path: Union[str, Path, CanonicalPath],
        attr_name: str,
        basis: str = "auto",
        strip_anchor: bool = False,
    ) -> Optional[str]:
        """Resolves path and extracts case-folded coordinate attribute value."""
        cp = self.resolve(path, basis=basis, strip_anchor=strip_anchor)
        val: Optional[str] = getattr(cp, attr_name)
        if val is None:
            return None
        return val.lower() if self.case_fold else val

    def repo_key(
        self,
        path: Union[str, Path, CanonicalPath],
        basis: str = "auto",
        strip_anchor: bool = False,
    ) -> Optional[str]:
        """Returns case-folded normalized repo_relative key for set/dict diff matching."""
        return self._coordinate_key(path, "repo_relative", basis=basis, strip_anchor=strip_anchor)

    def target_key(
        self,
        path: Union[str, Path, CanonicalPath],
        basis: str = "auto",
        strip_anchor: bool = False,
    ) -> Optional[str]:
        """Returns case-folded normalized target_relative key for set/dict baseline matching."""
        return self._coordinate_key(path, "target_relative", basis=basis, strip_anchor=strip_anchor)

    def canonical_key(
        self,
        path: Union[str, Path, CanonicalPath],
        basis: str = "auto",
        strip_anchor: bool = False,
    ) -> str:
        """Returns standard canonical lookup string for general dictionary indexing."""
        cp = self.resolve(path, basis=basis, strip_anchor=strip_anchor)
        key = cp.target_relative or cp.repo_relative or cp.raw
        return key.lower() if self.case_fold else key

    def equivalent(
        self,
        path_a: Union[str, Path, CanonicalPath],
        path_b: Union[str, Path, CanonicalPath],
        allow_suffix_fallback: bool = True,
    ) -> bool:
        """Determines whether two paths refer to the same file in this resolver's scope.

        Checks:
        1. target_relative equivalence
        2. repo_relative equivalence
        3. absolute_lexical equivalence
        4. physical resolution (if both files exist on disk)
        5. optional boundary suffix matching (for legacy unanchored baseline strings)
        """
        s_a = str(path_a or "").rstrip("\r\n")
        s_b = str(path_b or "").rstrip("\r\n")
        if not s_a or not s_b:
            return False

        ca = self.resolve(path_a)
        cb = self.resolve(path_b)

        if ca.target_relative is not None and cb.target_relative is not None:
            t_a = ca.target_relative.lower() if self.case_fold else ca.target_relative
            t_b = cb.target_relative.lower() if self.case_fold else cb.target_relative
            if t_a == t_b:
                return True

        if ca.repo_relative is not None and cb.repo_relative is not None:
            r_a = ca.repo_relative.lower() if self.case_fold else ca.repo_relative
            r_b = cb.repo_relative.lower() if self.case_fold else cb.repo_relative
            if r_a == r_b:
                return True

        if ca.absolute_lexical and cb.absolute_lexical:
            a_a = ca.absolute_lexical.lower() if self.case_fold else ca.absolute_lexical
            a_b = cb.absolute_lexical.lower() if self.case_fold else cb.absolute_lexical
            if a_a == a_b:
                return True

        # Physical resolution fallback if files exist
        try:
            pa = Path(ca.raw)
            pb = Path(cb.raw)

            def _find_file(p: Path) -> Optional[Path]:
                if p.is_absolute():
                    return p if p.is_file() else None
                t_cand = self.target_path / p
                if t_cand.is_file():
                    return t_cand
                if self.repo_path is not None and self.repo_path != self.target_path:
                    r_cand = self.repo_path / p
                    if r_cand.is_file():
                        return r_cand
                curr = self.target_path.parent
                while curr != curr.parent:
                    cand = curr / p
                    if cand.is_file():
                        return cand
                    if curr == self.repo_path:
                        break
                    curr = curr.parent
                return None

            pa_file = _find_file(pa)
            pb_file = _find_file(pb)
            if pa_file and pb_file and pa_file.resolve() == pb_file.resolve():
                return True
        except (ValueError, OSError, RuntimeError):
            pass

        if allow_suffix_fallback:
            return self._suffix_matches_boundary(ca.raw, cb.raw)

        return False

    def _probe_keys_in_diff(
        self,
        t_key: Optional[str],
        r_key: Optional[str],
        c_key: Optional[str],
        diff_keys: Set[str],
        has_tagged: bool,
    ) -> bool:
        """Probes coordinate keys against diff set respecting coordinate tag isolation."""
        if has_tagged:
            return bool(
                (t_key and f"target:{t_key}" in diff_keys)
                or (r_key and f"repo:{r_key}" in diff_keys)
            )
        return bool(
            (t_key and t_key in diff_keys)
            or (r_key and r_key in diff_keys)
            or (c_key and c_key in diff_keys)
        )

    def matches_diff(
        self,
        unit_file: Optional[Union[str, Path, CanonicalPath]],
        diff_keys: Set[str],
        basis: str = "target",
        strip_anchor: bool = True,
    ) -> bool:
        """Checks if a unit file path matches any diff key without ambiguous suffix matching."""
        if not unit_file or not diff_keys:
            return False

        has_tagged = any(k.startswith(("repo:", "target:")) for k in diff_keys)
        if self._probe_keys_in_diff(
            self.target_key(unit_file, basis=basis, strip_anchor=False),
            self.repo_key(unit_file, basis=basis, strip_anchor=False),
            self.canonical_key(unit_file, basis=basis, strip_anchor=False),
            diff_keys,
            has_tagged,
        ):
            return True

        # Stripped anchor match (for notebook cell anchors like .ipynb#cell_1 or #cell1)
        if strip_anchor:
            raw_str = str(getattr(unit_file, "raw", unit_file) or "")
            if "#" in raw_str:
                last_hash = raw_str.rfind("#")
                fragment = raw_str[last_hash + 1:]
                if fragment.lower().startswith("cell") or ".ipynb#" in raw_str:
                    return self._probe_keys_in_diff(
                        self.target_key(unit_file, basis=basis, strip_anchor=True),
                        self.repo_key(unit_file, basis=basis, strip_anchor=True),
                        self.canonical_key(unit_file, basis=basis, strip_anchor=True),
                        diff_keys,
                        has_tagged,
                    )

        return False

    def _suffix_matches_boundary(self, p1_raw: str, p2_raw: str) -> bool:
        """Fallback boundary suffix match for unanchored legacy baseline paths."""
        n1 = normalize_lexical_posix(p1_raw)
        n2 = normalize_lexical_posix(p2_raw)
        if not n1 or not n2:
            return False
        if self.case_fold:
            n1 = n1.lower()
            n2 = n2.lower()
        if n1 == n2:
            return True
        return n1.endswith("/" + n2) or n2.endswith("/" + n1)


def build_diff_path_keys(
    diff_files: Sequence[str],
    resolver: CanonicalPathResolver,
) -> Set[str]:
    """Builds canonical lookup keys for diff files using the canonical path resolver."""
    keys: Set[str] = set()
    for d in diff_files:
        if not d:
            continue
        # Git diff outputs paths relative to repo root; preserve literal # and spaces
        cp = resolver.resolve(d, basis="repo", strip_anchor=False)
        if resolver.target_in_repo is not None and cp.target_relative is None:
            continue
        r_key = resolver.repo_key(cp, strip_anchor=False)
        if r_key:
            keys.add(f"repo:{r_key}")
        if cp.target_relative is not None:
            t_key = resolver.target_key(cp, strip_anchor=False)
            if t_key:
                keys.add(f"target:{t_key}")
    return keys
