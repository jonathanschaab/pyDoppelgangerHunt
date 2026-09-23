"""Exhaustive unit and integration tests for the Unified Canonical Path Model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unicodedata
import pytest

from pydoppelgangerhunt.canonical_path import (
    CanonicalPath,
    CanonicalPathResolver,
    build_diff_path_keys,
    lexical_relative_to,
    normalize_lexical_posix,
)
from pydoppelgangerhunt.baseline import (
    load_baseline,
    record_baseline,
)


def test_normalize_lexical_posix_separators_and_anchors() -> None:
    """Verifies normalize_lexical_posix converts separators, strips ./, and handles anchors."""
    assert normalize_lexical_posix(None) == ""
    assert normalize_lexical_posix("") == ""
    assert normalize_lexical_posix("   ") == ""
    assert normalize_lexical_posix("././foo/bar.py") == "foo/bar.py"
    assert normalize_lexical_posix(".\\.\\foo\\bar.py") == "foo/bar.py"
    assert normalize_lexical_posix("foo/bar.ipynb#cell_1", strip_anchor=True) == "foo/bar.ipynb"
    assert normalize_lexical_posix("foo/bar.ipynb#cell_1", strip_anchor=False) == "foo/bar.ipynb#cell_1"
    assert normalize_lexical_posix("foo/bar.py#cell_1", strip_anchor=True) == "foo/bar.py#cell_1"
    assert normalize_lexical_posix("worker.py#cell_data.py", strip_anchor=True) == "worker.py#cell_data.py"
    assert normalize_lexical_posix("worker.py#v1", strip_anchor=True) == "worker.py#v1"
    assert normalize_lexical_posix("repo#1/pkg#2/mod.py", strip_anchor=False) == "repo#1/pkg#2/mod.py"


def test_normalize_lexical_posix_windows_drive_letters() -> None:
    """Verifies Windows drive letters are canonicalized to uppercase."""
    assert normalize_lexical_posix("c:/users/dev/repo") == "C:/users/dev/repo"
    assert normalize_lexical_posix("d:\\projects\\repo") == "D:/projects/repo"
    assert normalize_lexical_posix("E:/already/upper") == "E:/already/upper"


def test_normalize_lexical_posix_macos_nfc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies macOS decomposes Unicode (NFD) is normalized to NFC."""
    monkeypatch.setattr(sys, "platform", "darwin")
    decomposed = unicodedata.normalize("NFD", "café/résumé.py")
    normalized = normalize_lexical_posix(decomposed)
    assert normalized == unicodedata.normalize("NFC", "café/résumé.py")


def test_lexical_relative_to_computations() -> None:
    """Verifies pure lexical relative path resolution without touching disk."""
    assert lexical_relative_to("", "") is None
    assert lexical_relative_to("foo/bar.py", "") == "foo/bar.py"
    assert lexical_relative_to("/repo/src/pkg/mod.py", "/repo/src") == "pkg/mod.py"
    assert lexical_relative_to("/repo/src/pkg/mod.py", "/repo/src/") == "pkg/mod.py"
    assert lexical_relative_to("/repo/src", "/repo/src") == ""
    assert lexical_relative_to("/repo/tests/test_mod.py", "/repo/src") is None
    assert lexical_relative_to("C:/Repo/Src/mod.py", "c:/repo/src", case_fold=True) == "mod.py"
    assert lexical_relative_to("C:/Repo/Src/mod.py", "c:/repo/src", case_fold=False) is None


def test_canonical_path_dataclass_properties() -> None:
    """Verifies CanonicalPath immutability, display properties, and hashing."""
    cp = CanonicalPath(
        raw="./src/mod.py",
        repo_relative="packages/foo/src/mod.py",
        target_relative="src/mod.py",
        absolute_lexical="/app/packages/foo/src/mod.py",
    )
    assert cp.display_path == "src/mod.py"
    assert str(cp) == "src/mod.py"

    cp_no_target = CanonicalPath(
        raw="other/mod.py",
        repo_relative="packages/other/mod.py",
        target_relative=None,
    )
    assert cp_no_target.display_path == "packages/other/mod.py"

    # Dataclass is hashable and can be stored in sets
    s = {cp, cp_no_target}
    assert len(s) == 2


def test_resolver_subdirectory_and_repo_root_coordinates(tmp_path: Path) -> None:
    """Verifies CanonicalPathResolver correctly translates between repo and target subdirectories."""
    repo = tmp_path / "my_project"
    target = repo / "packages" / "sub_pkg"
    target.mkdir(parents=True)

    resolver = CanonicalPathResolver(target_root=target, repo_root=repo)
    assert resolver.target_in_repo == "packages/sub_pkg"

    # 1. Target-relative unit file resolution
    cp_unit = resolver.resolve("src/engine.py", basis="target")
    assert cp_unit.target_relative == "src/engine.py"
    assert cp_unit.repo_relative == "packages/sub_pkg/src/engine.py"
    assert resolver.target_key(cp_unit) == (
        "src/engine.py".lower() if resolver.case_fold else "src/engine.py"
    )
    assert resolver.repo_key(cp_unit) == (
        "packages/sub_pkg/src/engine.py".lower()
        if resolver.case_fold
        else "packages/sub_pkg/src/engine.py"
    )

    # 2. Git diff repo-relative file resolution
    cp_diff = resolver.resolve("packages/sub_pkg/src/engine.py", basis="repo")
    assert cp_diff.repo_relative == "packages/sub_pkg/src/engine.py"
    assert cp_diff.target_relative == "src/engine.py"

    # 3. Equivalence across coordinate systems
    assert resolver.equivalent("src/engine.py", "packages/sub_pkg/src/engine.py")

    # 4. External file in different subpackage
    cp_external = resolver.resolve("packages/other_pkg/src/engine.py", basis="repo")
    assert cp_external.repo_relative == "packages/other_pkg/src/engine.py"
    assert cp_external.target_relative is None
    # Crucial: Basename collision avoided! "src/engine.py" != "packages/other_pkg/src/engine.py"
    assert not resolver.equivalent("src/engine.py", "packages/other_pkg/src/engine.py", allow_suffix_fallback=False)


def test_resolver_diff_matching_and_basename_collision_prevention(tmp_path: Path) -> None:
    """Verifies diff matching matches true modified files and suppresses suffix collisions."""
    repo = tmp_path / "repo"
    target = repo / "pkg_a"
    target.mkdir(parents=True)

    resolver = CanonicalPathResolver(target_root=target, repo_root=repo)

    # Git diff reports pkg_a/worker.py changed
    diff_files = ["pkg_a/worker.py"]
    diff_keys = build_diff_path_keys(diff_files, resolver)

    # True unit under pkg_a matches
    assert resolver.matches_diff("worker.py", diff_keys)

    # Unit in unrelated pkg_b with same basename does NOT match
    assert not resolver.matches_diff("pkg_b/worker.py", diff_keys)

    # Empty inputs handled safely
    assert not resolver.matches_diff(None, diff_keys)
    assert not resolver.matches_diff("", diff_keys)
    assert not resolver.matches_diff("worker.py", set())


def test_resolver_deleted_and_nonexistent_files(tmp_path: Path) -> None:
    """Verifies resolver does not crash or fail when handling deleted or missing diff files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    resolver = CanonicalPathResolver(target_root=repo, repo_root=repo)

    # Deleted diff file path
    deleted_path = "deleted_module.py"
    cp_del = resolver.resolve(deleted_path, basis="repo")
    assert cp_del.repo_relative == "deleted_module.py"
    assert cp_del.target_relative == "deleted_module.py"

    # Diff keys build cleanly
    diff_keys = build_diff_path_keys([deleted_path, "/dev/null"], resolver)
    assert len(diff_keys) > 0


def test_baseline_schema_v15_metadata_round_trip(tmp_path: Path) -> None:
    """Verifies record_baseline persists schema 1.5.0 with path_basis and target_repo_relative."""
    base_file = tmp_path / "baseline_v15.json"
    u1 = {"file": "core/engine.py", "name": "fn1", "tokens": ["tok1", "tok2"]}
    u2 = {"file": "core/engine.py", "name": "fn2", "tokens": ["tok1", "tok2"]}

    saved = record_baseline([(1.0, u1, u2)], str(base_file), str(tmp_path), threshold=0.85)
    assert Path(saved).is_file()

    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert data["version"] == "1.5.0"
    assert data["path_basis"] == "target_relative"

    loaded = load_baseline(str(base_file))
    assert loaded.path_basis == "target_relative"
    assert len(loaded) > 0


def test_baseline_cross_root_equivalence_matching(tmp_path: Path) -> None:
    """Verifies baseline records match across repo root and subdirectory target scans."""
    from pydoppelgangerhunt.baseline import filter_clones_by_baseline

    repo = tmp_path / "my_project"
    sub_pkg = repo / "sub_pkg"
    sub_pkg.mkdir(parents=True)

    base_file = repo / "baseline.json"

    # Clone recorded with repo-relative paths
    u1_repo = {"file": "sub_pkg/service.py", "name": "run", "structural_hash": "abcd1234efgh5678"}
    u2_repo = {"file": "sub_pkg/worker.py", "name": "run", "structural_hash": "1234abcd5678efgh"}
    record_baseline([(1.0, u1_repo, u2_repo)], str(base_file), str(repo), threshold=0.90)

    loaded_base = load_baseline(str(base_file))

    # Current scan running inside sub_pkg: units have target-relative paths
    u1_sub = {"file": "service.py", "name": "run", "structural_hash": "abcd1234efgh5678"}
    u2_sub = {"file": "worker.py", "name": "run", "structural_hash": "1234abcd5678efgh"}
    current_clones = [(1.0, u1_sub, u2_sub)]

    # Filter with repo_root specified
    unsuppressed, suppressed_count = filter_clones_by_baseline(
        current_clones, loaded_base, repo_root=str(repo)
    )
    assert suppressed_count == 1
    assert len(unsuppressed) == 0


def test_canonical_path_security_and_cache_isolation() -> None:
    """Verifies null byte sanitization, cache isolation across strip_anchor, and traversal guards."""
    # 1. Null byte sanitization
    assert normalize_lexical_posix("foo\x00bar.py") == "foobar.py"
    assert normalize_lexical_posix("\x00") == ""

    # 2. Cache isolation across strip_anchor
    resolver = CanonicalPathResolver(target_root=".")
    p1 = resolver.resolve("script.ipynb#cell_1", strip_anchor=False)
    assert p1.target_relative == "script.ipynb#cell_1"
    p2 = resolver.resolve("script.ipynb#cell_1", strip_anchor=True)
    assert p2.target_relative == "script.ipynb"

    # 3. Path traversal escape guard in lexical_relative_to
    assert lexical_relative_to("/repo/src/../../etc/passwd", "/repo/src") is None
    assert lexical_relative_to("/repo/src/pkg/../pkg/mod.py", "/repo/src") == "pkg/mod.py"
    assert lexical_relative_to("C:/repo/src/../../windows/system32", "C:/repo/src") is None


def test_matches_diff_notebook_anchor_support() -> None:
    """Verifies matches_diff correctly matches notebook units with #cell anchors against diff keys."""
    resolver = CanonicalPathResolver(target_root="packages/subpkg", repo_root=".")
    diff_files = ["packages/subpkg/analysis.ipynb"]
    diff_keys = build_diff_path_keys(diff_files, resolver)

    assert resolver.matches_diff("analysis.ipynb#cell_1", diff_keys)
    assert resolver.matches_diff("analysis.ipynb#cell_99", diff_keys)
    assert not resolver.matches_diff("other_notebook.ipynb#cell_1", diff_keys)
    assert not resolver.matches_diff("packages/other_pkg/analysis.ipynb#cell_1", diff_keys)


def test_canonical_path_preserves_significant_spaces() -> None:
    """Verifies that leading and trailing spaces in filesystem paths are preserved."""
    assert normalize_lexical_posix(" foo.py") == " foo.py"
    assert normalize_lexical_posix("dir /bar.py") == "dir /bar.py"

    resolver = CanonicalPathResolver(target_root="/repo")
    cp = resolver.resolve(" pkg/ foo.py", basis="target")
    assert cp.target_relative == " pkg/ foo.py"

    assert resolver.equivalent(" pkg/ foo.py", " pkg/ foo.py")
    assert not resolver.equivalent(" pkg/ foo.py", "pkg/foo.py")


def test_build_diff_path_keys_preserves_literal_hashes() -> None:
    """Verifies that literal # in filenames are preserved in diff keys and do not truncate to prefix."""
    resolver = CanonicalPathResolver(target_root="/repo/sub", repo_root="/repo")
    diff_files = ["sub/pkg#2/worker.py"]
    diff_keys = build_diff_path_keys(diff_files, resolver)

    # Unit matching: exact file with literal hash matches
    assert resolver.matches_diff("pkg#2/worker.py", diff_keys)

    # Different file without hash does NOT collide or falsely match
    assert not resolver.matches_diff("pkg/worker.py", diff_keys)
    assert not resolver.matches_diff("pkg", diff_keys)


def test_build_diff_path_keys_repo_relative_does_not_leak_target_key() -> None:
    """Verifies that repo-relative diff files outside target directory do not emit target keys."""
    resolver = CanonicalPathResolver(target_root="/repo/pkg", repo_root="/repo")
    diff_files = ["foo.py"]
    diff_keys = build_diff_path_keys(diff_files, resolver)

    # "foo.py" is at repo root; it should NOT match a unit in pkg/foo.py (which has target_key "foo.py")
    assert not resolver.matches_diff("foo.py", diff_keys)

    # However, if pkg/foo.py was changed in repo diff, it matches
    pkg_diff_keys = build_diff_path_keys(["pkg/foo.py"], resolver)
    assert resolver.matches_diff("foo.py", pkg_diff_keys)


def test_matches_diff_nested_coincident_target_path() -> None:
    """Verifies matches_diff with basis='target' matches when target_in_repo equals a subdirectory name."""
    resolver = CanonicalPathResolver(target_root="/repo/src", repo_root="/repo")
    # Harvested unit from /repo/src/src/foo.py has unit_file = "src/foo.py"
    diff_keys = {"src/src/foo.py"}
    assert resolver.matches_diff("src/foo.py", diff_keys, basis="target")
    assert resolver.matches_diff("src/foo.py", diff_keys)


def test_resolver_normalizes_single_file_target_root_to_parent(tmp_path: Path) -> None:
    """Verifies that CanonicalPathResolver normalizes a file target to its parent directory."""
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    file_path = src / "worker.py"
    file_path.write_text("print(1)\n", encoding="utf-8")

    resolver = CanonicalPathResolver(target_root=file_path, repo_root=repo)
    assert resolver.target_resolved == src.resolve()
    assert resolver.target_in_repo == "src"

    diff_keys = build_diff_path_keys(["src/worker.py"], resolver)
    assert "target:worker.py" in diff_keys
    assert "repo:src/worker.py" in diff_keys
    assert resolver.matches_diff("worker.py", diff_keys, basis="target")


def test_lexical_relative_to_filesystem_root() -> None:
    """Verifies that lexical_relative_to correctly handles root '/' base directory without returning absolute paths."""
    assert lexical_relative_to("/tmp/x.py", "/") == "tmp/x.py"
    assert lexical_relative_to("tmp/x.py", "/") == "tmp/x.py"
    assert lexical_relative_to("/", "/") == ""
    assert lexical_relative_to("/tmp/../../escaped.py", "/") is None
    assert lexical_relative_to("/a/b/c.py", "/") == "a/b/c.py"

    resolver = CanonicalPathResolver(target_root="/tmp", repo_root="/")
    assert resolver.target_in_repo == "tmp"
    diff_keys = build_diff_path_keys(["/tmp/worker.py"], resolver)
    assert "target:worker.py" in diff_keys
    assert "repo:tmp/worker.py" in diff_keys
    assert resolver.matches_diff("worker.py", diff_keys, basis="target")
    assert resolver.matches_diff("/tmp/worker.py", diff_keys, basis="repo")


def test_canonical_path_preserves_unc_network_paths_on_windows() -> None:
    """Verifies that CanonicalPathResolver does not corrupt UNC network paths with drive letters on Windows."""
    resolver = CanonicalPathResolver(target_root="C:/repo/src", repo_root="C:/repo", is_windows=True)

    unc_path = "//server/share/folder/file.py"
    res = resolver.resolve(unc_path)
    assert res.absolute_lexical == unc_path
    assert not res.absolute_lexical.startswith("C:")
    assert not res.absolute_lexical.startswith("C://")

    # Also test backslash UNC path input
    unc_backslash = r"\\server\share\folder\file.py"
    res_bs = resolver.resolve(unc_backslash)
    assert res_bs.absolute_lexical == unc_path
    assert not res_bs.absolute_lexical.startswith("C:")

    # Verify regular absolute path with leading slash DOES get drive letter on Windows
    res_drive_rel = resolver.resolve("/folder/file.py")
    assert res_drive_rel.absolute_lexical == "C:/folder/file.py"

    # Verify coordinate translation with Windows drive paths
    res_file = resolver.resolve("C:/repo/src/sub/file.py")
    assert res_file.target_relative == "sub/file.py"
    assert res_file.repo_relative == "src/sub/file.py"

    # Verify UNC target and repo roots
    resolver_unc = CanonicalPathResolver(
        target_root="//server/share/src", repo_root="//server/share", is_windows=True
    )
    assert resolver_unc.target_in_repo == "src"
    res_unc = resolver_unc.resolve("//server/share/src/sub/file.py")
    assert res_unc.target_relative == "sub/file.py"
    assert res_unc.repo_relative == "src/sub/file.py"



def test_lexical_relative_to_empty_base_windows_drive() -> None:
    """Verifies that lexical_relative_to guards Windows drive paths when base is empty."""
    assert lexical_relative_to("C:/project/foo.py", "") is None
    assert lexical_relative_to("d:/project/bar.py", "") is None
    assert lexical_relative_to("/project/foo.py", "") is None
    assert lexical_relative_to("project/foo.py", "") == "project/foo.py"


def test_matches_diff_has_tagged_caching() -> None:
    """Verifies matches_diff caches has_tagged per diff_keys and respects explicit parameter."""
    resolver = CanonicalPathResolver(target_root="/repo/src", repo_root="/repo")
    diff_keys = {"target:mod.py", "repo:src/mod.py"}

    # Initially cache is None
    assert resolver._tagged_keys_cache is None  # pylint: disable=protected-access

    # First call populates cache
    assert resolver.matches_diff("mod.py", diff_keys)
    assert resolver._tagged_keys_cache is not None  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[0] is diff_keys  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[1] is True  # pylint: disable=protected-access

    # Subsequent call reuses cached value
    assert resolver.matches_diff("mod.py", diff_keys)

    # Explicit parameter overrides/bypasses cache check
    assert resolver.matches_diff("mod.py", diff_keys, has_tagged=True)
    assert not resolver.matches_diff("mod.py", diff_keys, has_tagged=False)


def test_resolver_root_directories_no_double_slash() -> None:
    """Verifies that relative path resolution under root directories does not introduce double slashes."""
    resolver = CanonicalPathResolver(target_root="/", repo_root="/")
    res = resolver.resolve("worker.py", basis="target")
    assert res.absolute_lexical is not None
    assert not res.absolute_lexical.startswith("//")
    assert "://" not in res.absolute_lexical

    res_repo = resolver.resolve("worker.py", basis="repo")
    assert res_repo.absolute_lexical is not None
    assert not res_repo.absolute_lexical.startswith("//")
    assert "://" not in res_repo.absolute_lexical

    # Windows drive root
    resolver_win = CanonicalPathResolver(target_root="C:/", repo_root="C:/", is_windows=True)
    res_win = resolver_win.resolve("worker.py", basis="target")
    assert res_win.absolute_lexical == "C:/worker.py"
    assert "C://" not in res_win.absolute_lexical

    res_win_repo = resolver_win.resolve("worker.py", basis="repo")
    assert res_win_repo.absolute_lexical == "C:/worker.py"
    assert "C://" not in res_win_repo.absolute_lexical


def test_resolver_simulated_windows_file_target_normalization() -> None:
    """Verifies that _normalize_root_directory normalizes simulated Windows file targets to parents."""
    from unittest.mock import patch
    from pydoppelgangerhunt.canonical_path import _normalize_root_directory

    with patch("os.name", "posix"):
        _, _, sim_lex = _normalize_root_directory("C:/repo/src/worker.py", is_windows=True)
        assert sim_lex == "C:/repo/src"
        _, _, sim_nb = _normalize_root_directory("C:/repo/src/analysis.ipynb", is_windows=True)
        assert sim_nb == "C:/repo/src"
        _, _, sim_drive_file = _normalize_root_directory("C:/worker.py", is_windows=True)
        assert sim_drive_file == "C:/"


def test_resolver_equivalent_disjoint_repo_bounded_traversal() -> None:
    """Verifies that equivalent does not perform unbounded upward traversal when target is not in repo."""
    resolver = CanonicalPathResolver(target_root="/some/target", repo_root="/other/repo")
    assert resolver.target_in_repo is None
    assert not resolver.equivalent("nonexistent_a.py", "nonexistent_b.py")


