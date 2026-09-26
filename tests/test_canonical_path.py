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
    DiffPathKeySet,
    _join_lexical_posix,
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
    assert normalize_lexical_posix("foo/bar.ipynb#cell1", strip_anchor=True) == "foo/bar.ipynb"
    assert normalize_lexical_posix("foo/bar.ipynb#cell_1", strip_anchor=False) == "foo/bar.ipynb#cell_1"
    assert normalize_lexical_posix("foo/experiment#1.ipynb#cell_4", strip_anchor=True) == "foo/experiment#1.ipynb"
    assert normalize_lexical_posix("foo/experiment#1.ipynb#cell_4", strip_anchor=False) == "foo/experiment#1.ipynb#cell_4"
    # Literal hash paths with fragments starting with cell but not digits must NOT be stripped
    assert normalize_lexical_posix("foo/report.ipynb#cellular", strip_anchor=True) == "foo/report.ipynb#cellular"
    assert normalize_lexical_posix("foo/report.ipynb#cell", strip_anchor=True) == "foo/report.ipynb#cell"
    assert normalize_lexical_posix("foo/report.ipynb#cell_", strip_anchor=True) == "foo/report.ipynb#cell_"
    assert normalize_lexical_posix("foo/report.ipynb#cell_abc", strip_anchor=True) == "foo/report.ipynb#cell_abc"
    assert normalize_lexical_posix("foo/report#cellular.ipynb#cell_2", strip_anchor=True) == "foo/report#cellular.ipynb"
    assert normalize_lexical_posix("foo/bar.py#cell_1", strip_anchor=True) == "foo/bar.py#cell_1"
    assert normalize_lexical_posix("worker.py#cell_data.py", strip_anchor=True) == "worker.py#cell_data.py"
    assert normalize_lexical_posix("worker.py#v1", strip_anchor=True) == "worker.py#v1"
    assert normalize_lexical_posix("repo#1/pkg#2/mod.py", strip_anchor=False) == "repo#1/pkg#2/mod.py"


def test_parse_notebook_cell_anchor() -> None:
    """Verifies that parse_notebook_cell_anchor extracts base path and 0-based index strictly for digit suffixes."""
    from pydoppelgangerhunt.canonical_path import parse_notebook_cell_anchor  # pylint: disable=import-outside-toplevel

    assert parse_notebook_cell_anchor(None) is None
    assert parse_notebook_cell_anchor("") is None
    assert parse_notebook_cell_anchor("script.py") is None
    assert parse_notebook_cell_anchor("notebook.ipynb") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cellular") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cell") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_abc") is None
    assert parse_notebook_cell_anchor("notebook.py#cell_1") is None

    # Valid anchors
    assert parse_notebook_cell_anchor("analysis.ipynb#cell_1") == ("analysis.ipynb", 0)
    assert parse_notebook_cell_anchor("analysis.ipynb#cell1") == ("analysis.ipynb", 0)
    assert parse_notebook_cell_anchor("dir/sub/analysis.ipynb#cell_10") == ("dir/sub/analysis.ipynb", 9)
    assert parse_notebook_cell_anchor("dir\\sub\\analysis.ipynb#cell_5") == ("dir\\sub\\analysis.ipynb", 4)
    assert parse_notebook_cell_anchor("report#cellular.ipynb#cell_3") == ("report#cellular.ipynb", 2)


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

    # Re-resolving CanonicalPath instance with strip_anchor=True
    p3 = resolver.resolve(p1, strip_anchor=True)
    assert p3.target_relative == "script.ipynb"
    assert resolver.target_key(p1, strip_anchor=True) == "script.ipynb"
    assert resolver.repo_key(p1, strip_anchor=True) == "script.ipynb"

    # CanonicalPath instance without anchors returns identical instance
    p_no_anchor = resolver.resolve("script.py", strip_anchor=False)
    assert resolver.resolve(p_no_anchor, strip_anchor=True) is p_no_anchor
    assert resolver.resolve(p_no_anchor, strip_anchor=False) is p_no_anchor

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
    # Passing pre-resolved CanonicalPath instance
    cp_anchored = resolver.resolve("analysis.ipynb#cell_1", strip_anchor=False)
    assert resolver.matches_diff(cp_anchored, diff_keys, strip_anchor=True)
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
    assert lexical_relative_to("//server/share/file.py", "/") is None
    assert lexical_relative_to("C:/repo/file.py", "/") is None

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

    # First call populates cache without strongly referencing diff_keys
    assert resolver.matches_diff("mod.py", diff_keys)
    assert resolver._tagged_keys_cache is not None  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[0] == id(diff_keys)  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[1] == len(diff_keys)  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[2] in diff_keys  # pylint: disable=protected-access
    assert resolver._tagged_keys_cache[3] is True  # pylint: disable=protected-access

    # Subsequent call reuses cached value
    assert resolver.matches_diff("mod.py", diff_keys)

    # Simulated memory ID reuse: same id & len but sample key not in new set causes safe cache miss
    untagged_keys = {"mod.py", "other.py"}
    resolver._tagged_keys_cache = (id(untagged_keys), len(untagged_keys), "stale_key", True)  # pylint: disable=protected-access
    # Because "stale_key" not in untagged_keys, cache is invalidated and recomputed to False
    assert resolver.matches_diff("mod.py", untagged_keys)
    assert resolver._tagged_keys_cache[3] is False  # pylint: disable=protected-access

    # Explicit parameter overrides/bypasses cache check
    assert resolver.matches_diff("mod.py", diff_keys, has_tagged=True)
    assert not resolver.matches_diff("mod.py", diff_keys, has_tagged=False)

    # clear_cache empties both resolution cache and diff key metadata cache
    resolver.clear_cache()
    assert resolver._tagged_keys_cache is None  # pylint: disable=protected-access


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


def test_resolver_native_nonexistent_file_target_normalization(tmp_path: Path) -> None:
    """Verifies that _normalize_root_directory normalizes non-existent .py and .ipynb files to their parent."""
    from pydoppelgangerhunt.canonical_path import _normalize_root_directory  # pylint: disable=import-outside-toplevel

    # Non-existent python file in existing parent directory
    nonexistent_file = tmp_path / "phantom_module.py"
    assert not nonexistent_file.exists()
    _, _, lex_path = _normalize_root_directory(str(nonexistent_file), is_windows=(os.name == "nt"))
    expected_parent = normalize_lexical_posix(str(tmp_path.resolve()))
    assert lex_path == expected_parent

    # Non-existent notebook file in existing parent directory
    nonexistent_nb = tmp_path / "phantom_notebook.ipynb"
    assert not nonexistent_nb.exists()
    _, _, lex_nb = _normalize_root_directory(str(nonexistent_nb), is_windows=(os.name == "nt"))
    assert lex_nb == expected_parent


def test_resolver_matches_diff_notebook_anchor_with_literal_hash() -> None:
    """Verifies matches_diff strips notebook anchors properly when filename contains literal #."""
    resolver = CanonicalPathResolver(target_root="/repo", repo_root="/repo")
    diff_keys = {"target:experiments/run#1.ipynb"}
    assert resolver.matches_diff("experiments/run#1.ipynb#cell_3", diff_keys)
    assert not resolver.matches_diff("experiments/run#2.ipynb#cell_3", diff_keys)


def test_resolver_invalidate_path() -> None:
    """Verifies that invalidate_path removes matching path entries across separators, case, and notebook anchors."""
    resolver = CanonicalPathResolver(target_root="/repo", repo_root="/repo")
    resolver.resolve("foo.py")
    resolver.resolve("bar.py")
    resolver.resolve("pkg/worker.py")
    resolver.resolve("experiments/run.ipynb#cell_1")
    resolver.resolve("experiments/run.ipynb#cell_2")

    assert len(resolver._cache) == 5  # pylint: disable=protected-access

    # 1. Invalidate foo.py: only foo.py is removed
    resolver.invalidate_path("foo.py")
    assert len(resolver._cache) == 4  # pylint: disable=protected-access
    assert not any(k[0] == "foo.py" for k in resolver._cache)  # pylint: disable=protected-access

    # 2. Windows backslash invalidation evicts POSIX forward slash entry (pkg\worker.py -> pkg/worker.py)
    resolver.invalidate_path(r"pkg\worker.py")
    assert len(resolver._cache) == 3  # pylint: disable=protected-access
    assert not any("worker.py" in k[0] for k in resolver._cache)  # pylint: disable=protected-access

    # 3. Invalidating parent notebook without anchor evicts all child cell entries
    resolver.invalidate_path(r"experiments\run.ipynb")
    assert len(resolver._cache) == 1  # pylint: disable=protected-access
    assert any(k[0] == "bar.py" for k in resolver._cache)  # pylint: disable=protected-access

    # 4. Invalidate empty or non-existent path does nothing
    resolver.invalidate_path("")
    resolver.invalidate_path("non_existent.py")
    assert len(resolver._cache) == 1  # pylint: disable=protected-access

    # 5. Cell-specific invalidation evicts only that specific cell
    resolver.resolve("demo.ipynb#cell_1")
    resolver.resolve("demo.ipynb#cell_2")
    resolver.invalidate_path("demo.ipynb#cell_1")
    assert any(k[0] == "demo.ipynb#cell_2" for k in resolver._cache)  # pylint: disable=protected-access
    assert not any(k[0] == "demo.ipynb#cell_1" for k in resolver._cache)  # pylint: disable=protected-access


def test_resolver_raw_lexical_root_symlink_fallback(tmp_path: Path) -> None:
    """Verifies that absolute paths referencing raw symlink roots resolve when files do not exist physically."""
    real_repo = tmp_path / "real_repo"
    real_repo.mkdir()
    sym_repo = tmp_path / "sym_repo"

    try:
        sym_repo.symlink_to(real_repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in this environment")

    resolver = CanonicalPathResolver(target_root=sym_repo, repo_root=sym_repo)
    # File does not exist on disk (e.g. deleted file from diff)
    deleted_abs_symlink = normalize_lexical_posix(str(sym_repo / "deleted_file.py"))
    cp = resolver.resolve(deleted_abs_symlink)
    assert cp.target_relative == "deleted_file.py"
    assert cp.repo_relative == "deleted_file.py"


def test_resolver_re_resolve_canonical_path_with_explicit_basis(tmp_path: Path) -> None:
    """Verifies that re-resolving an existing CanonicalPath with an explicit basis re-evaluates coordinates."""
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)

    resolver = CanonicalPathResolver(target_root=pkg, repo_root=repo)

    # 1. Resolve with default basis="auto" (treated as target-relative)
    cp_auto = resolver.resolve("worker.py", basis="auto")
    assert cp_auto.target_relative == "worker.py"
    assert cp_auto.repo_relative == "pkg/worker.py"

    # 2. Re-resolve the CanonicalPath instance with basis="repo"
    cp_repo = resolver.resolve(cp_auto, basis="repo")
    assert cp_repo.repo_relative == "worker.py"
    assert cp_repo.target_relative is None

    # 3. Passing CanonicalPath with basis="auto" and no anchor returns same instance (fast-path)
    cp_cached = resolver.resolve(cp_auto, basis="auto")
    assert cp_cached is cp_auto

    # 4. Passing anchored CanonicalPath with strip_anchor=True extracts raw and strips anchor
    cp_anchored = resolver.resolve("notebook.ipynb#cell_3", basis="auto")
    assert "#cell_3" in cp_anchored.raw
    cp_stripped = resolver.resolve(cp_anchored, basis="auto", strip_anchor=True)
    assert cp_stripped.target_relative == "notebook.ipynb"
    assert cp_stripped.repo_relative == "pkg/notebook.ipynb"
    assert cp_stripped.raw == "notebook.ipynb#cell_3"


def test_diff_path_key_set_and_zero_allocation_probing(tmp_path: Path) -> None:
    """Verifies DiffPathKeySet sub-sets and zero-allocation diff probing in CanonicalPathResolver."""
    from pydoppelgangerhunt.canonical_path import (  # pylint: disable=import-outside-toplevel
        CanonicalPathResolver,
        DiffPathKeySet,
        build_diff_path_keys,
    )

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f1 = src / "worker.py"
    f1.write_text("x = 1\n", encoding="utf-8")

    resolver = CanonicalPathResolver(target_root=src, repo_root=repo)

    # 1. build_diff_path_keys produces DiffPathKeySet with separate target and repo keys
    diff_keys = build_diff_path_keys(["src/worker.py"], resolver)
    assert isinstance(diff_keys, DiffPathKeySet)
    assert isinstance(diff_keys, set)
    assert "target:worker.py" in diff_keys
    assert "repo:src/worker.py" in diff_keys
    assert diff_keys.diff_target_keys == {"worker.py"}
    assert diff_keys.diff_repo_keys == {"src/worker.py"}

    # 2. resolver stores separate diff_target_keys and diff_repo_keys
    assert resolver.diff_target_keys == {"worker.py"}
    assert resolver.diff_repo_keys == {"src/worker.py"}

    # 3. matches_diff uses separate sets
    assert resolver.matches_diff("worker.py", diff_keys, basis="target")
    assert resolver.matches_diff("src/worker.py", diff_keys, basis="repo")
    assert not resolver.matches_diff("other.py", diff_keys)

    # 4. DiffPathKeySet copy, add, update methods
    copy_keys = diff_keys.copy()
    assert isinstance(copy_keys, DiffPathKeySet)
    assert copy_keys.diff_target_keys == {"worker.py"}
    assert copy_keys.diff_repo_keys == {"src/worker.py"}

    copy_keys.add("target:helper.py")
    copy_keys.add("repo:src/helper.py")
    assert "helper.py" in copy_keys.diff_target_keys
    assert "src/helper.py" in copy_keys.diff_repo_keys

    copy_keys.update(["target:util.py", "repo:src/util.py"])
    assert "util.py" in copy_keys.diff_target_keys
    assert "src/util.py" in copy_keys.diff_repo_keys

    # Mutation parity: discard
    copy_keys.discard("target:util.py")
    assert "target:util.py" not in copy_keys
    assert "util.py" not in copy_keys.diff_target_keys
    copy_keys.discard("repo:src/util.py")
    assert "repo:src/util.py" not in copy_keys
    assert "src/util.py" not in copy_keys.diff_repo_keys
    # Non-string object discard should safely no-op
    copy_keys.discard(12345)

    # Mutation parity: remove
    copy_keys.remove("target:helper.py")
    assert "target:helper.py" not in copy_keys
    assert "helper.py" not in copy_keys.diff_target_keys

    # Mutation parity: pop
    popped = copy_keys.pop()
    assert popped not in copy_keys
    if popped.startswith("target:"):
        assert popped[7:] not in copy_keys.diff_target_keys
    elif popped.startswith("repo:"):
        assert popped[5:] not in copy_keys.diff_repo_keys

    # Mutation parity: clear
    copy_keys.clear()
    assert len(copy_keys) == 0
    assert len(copy_keys.diff_target_keys) == 0
    assert len(copy_keys.diff_repo_keys) == 0

    # Mutation parity: difference_update and -=
    mut_keys = DiffPathKeySet(["target:a.py", "target:b.py", "repo:src/a.py"])
    mut_keys.difference_update(["target:b.py"])
    assert "target:b.py" not in mut_keys
    assert "b.py" not in mut_keys.diff_target_keys

    mut_keys -= {"target:a.py"}
    assert "target:a.py" not in mut_keys
    assert "a.py" not in mut_keys.diff_target_keys
    assert mut_keys.diff_repo_keys == {"src/a.py"}

    # Mutation parity: intersection_update and &=
    mut_keys2 = DiffPathKeySet(["target:x.py", "target:y.py", "repo:src/x.py"])
    mut_keys2.intersection_update(["target:x.py", "repo:src/x.py"])
    assert "target:y.py" not in mut_keys2
    assert "y.py" not in mut_keys2.diff_target_keys
    assert mut_keys2.diff_target_keys == {"x.py"}

    mut_keys2 &= {"target:x.py"}
    assert "repo:src/x.py" not in mut_keys2
    assert len(mut_keys2.diff_repo_keys) == 0

    # Mutation parity: symmetric_difference_update
    mut_keys3 = DiffPathKeySet(["target:m.py"])
    mut_keys3.symmetric_difference_update(["target:m.py", "target:n.py"])
    assert "target:m.py" not in mut_keys3
    assert "m.py" not in mut_keys3.diff_target_keys
    assert "target:n.py" in mut_keys3
    assert "n.py" in mut_keys3.diff_target_keys

    # Mutation parity: symmetric_difference_update with duplicate elements in operand
    mut_keys_dup = DiffPathKeySet(["target:a.py"])
    mut_keys_dup.symmetric_difference_update(["target:a.py", "target:a.py"])
    assert "target:a.py" not in mut_keys_dup
    assert "a.py" not in mut_keys_dup.diff_target_keys

    mut_keys_dup2 = DiffPathKeySet()
    mut_keys_dup2.symmetric_difference_update(["target:b.py", "target:b.py"])
    assert "target:b.py" in mut_keys_dup2
    assert "b.py" in mut_keys_dup2.diff_target_keys

    # Mutation parity: in-place operators |= and ^=
    mut_op = DiffPathKeySet(["target:orig.py"])
    mut_op |= {"target:added.py", "repo:src/added.py"}
    assert "target:orig.py" in mut_op
    assert "target:added.py" in mut_op
    assert mut_op.diff_target_keys == {"orig.py", "added.py"}
    assert mut_op.diff_repo_keys == {"src/added.py"}

    mut_op ^= {"target:orig.py", "target:flipped.py"}
    assert "target:orig.py" not in mut_op
    assert "orig.py" not in mut_op.diff_target_keys
    assert "target:flipped.py" in mut_op
    assert "flipped.py" in mut_op.diff_target_keys

    # Self-mutation tests: s -= s, s ^= s, s &= s, s |= s and method calls
    self_mut = DiffPathKeySet(["target:item.py", "repo:src/item.py"])
    self_mut -= self_mut
    assert len(self_mut) == 0
    assert len(self_mut.diff_target_keys) == 0
    assert len(self_mut.diff_repo_keys) == 0

    self_mut = DiffPathKeySet(["target:item.py", "repo:src/item.py"])
    self_mut ^= self_mut
    assert len(self_mut) == 0
    assert len(self_mut.diff_target_keys) == 0
    assert len(self_mut.diff_repo_keys) == 0

    self_mut = DiffPathKeySet(["target:item.py", "repo:src/item.py"])
    self_mut &= self_mut
    assert self_mut == {"target:item.py", "repo:src/item.py"}
    assert self_mut.diff_target_keys == {"item.py"}
    assert self_mut.diff_repo_keys == {"src/item.py"}

    self_mut |= self_mut
    assert self_mut == {"target:item.py", "repo:src/item.py"}
    assert self_mut.diff_target_keys == {"item.py"}
    assert self_mut.diff_repo_keys == {"src/item.py"}

    self_mut.difference_update(self_mut)
    assert len(self_mut) == 0
    assert len(self_mut.diff_target_keys) == 0
    assert len(self_mut.diff_repo_keys) == 0

    self_mut = DiffPathKeySet(["target:item.py"])
    self_mut.symmetric_difference_update(self_mut)
    assert len(self_mut) == 0
    assert len(self_mut.diff_target_keys) == 0

    self_mut = DiffPathKeySet(["target:item.py"])
    self_mut.intersection_update(self_mut)
    assert self_mut == {"target:item.py"}
    assert self_mut.diff_target_keys == {"item.py"}

    self_mut.update(self_mut)
    assert self_mut == {"target:item.py"}
    assert self_mut.diff_target_keys == {"item.py"}

    # Binary set operators: |, -, &, ^ and methods union, difference, intersection, symmetric_difference
    bin1 = DiffPathKeySet(["target:a.py", "repo:b.py"])
    bin2 = DiffPathKeySet(["target:a.py", "target:c.py"])

    # Union |
    res_or = bin1 | bin2
    assert isinstance(res_or, DiffPathKeySet)
    assert res_or == {"target:a.py", "repo:b.py", "target:c.py"}
    assert res_or.diff_target_keys == {"a.py", "c.py"}
    assert res_or.diff_repo_keys == {"b.py"}
    res_or_meth = bin1.union(["target:d.py"])
    assert isinstance(res_or_meth, DiffPathKeySet)
    assert res_or_meth.diff_target_keys == {"a.py", "d.py"}

    # Difference -
    res_sub = bin1 - bin2
    assert isinstance(res_sub, DiffPathKeySet)
    assert res_sub == {"repo:b.py"}
    assert res_sub.diff_target_keys == set()
    assert res_sub.diff_repo_keys == {"b.py"}
    res_sub_meth = bin1.difference(["target:a.py"])
    assert isinstance(res_sub_meth, DiffPathKeySet)
    assert res_sub_meth == {"repo:b.py"}

    # Intersection &
    res_and = bin1 & bin2
    assert isinstance(res_and, DiffPathKeySet)
    assert res_and == {"target:a.py"}
    assert res_and.diff_target_keys == {"a.py"}
    assert res_and.diff_repo_keys == set()
    res_and_meth = bin1.intersection(["target:a.py"])
    assert isinstance(res_and_meth, DiffPathKeySet)
    assert res_and_meth == {"target:a.py"}

    # Symmetric Difference ^
    res_xor = bin1 ^ bin2
    assert isinstance(res_xor, DiffPathKeySet)
    assert res_xor == {"repo:b.py", "target:c.py"}
    assert res_xor.diff_target_keys == {"c.py"}
    assert res_xor.diff_repo_keys == {"b.py"}
    res_xor_meth = bin1.symmetric_difference(["target:a.py", "target:c.py"])
    assert isinstance(res_xor_meth, DiffPathKeySet)
    assert res_xor_meth == {"repo:b.py", "target:c.py"}

    # Self binary operations
    assert isinstance(bin1 - bin1, DiffPathKeySet) and len(bin1 - bin1) == 0
    assert isinstance(bin1 ^ bin1, DiffPathKeySet) and len(bin1 ^ bin1) == 0
    assert isinstance(bin1 & bin1, DiffPathKeySet) and bin1 & bin1 == bin1
    assert isinstance(bin1 | bin1, DiffPathKeySet) and bin1 | bin1 == bin1

    # TypeError on invalid operand types for operators
    with pytest.raises(TypeError):
        _ = bin1 - ["target:a.py"]  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = bin1 | ["target:a.py"]  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = bin1 & ["target:a.py"]  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = bin1 ^ ["target:a.py"]  # type: ignore[operator]

    # 5. set_diff_keys on resolver
    resolver.clear_cache()
    assert resolver.diff_target_keys is None
    assert resolver.diff_repo_keys is None

    # Set from DiffPathKeySet
    resolver.set_diff_keys(diff_keys)
    assert resolver.diff_target_keys == {"worker.py"}
    assert resolver.diff_repo_keys == {"src/worker.py"}

    # Set from raw set
    raw_set = {"target:mod.py", "repo:pkg/mod.py"}
    resolver.set_diff_keys(raw_set)
    assert resolver.diff_target_keys == {"mod.py"}
    assert resolver.diff_repo_keys == {"pkg/mod.py"}

    # Explicit sets
    resolver.set_diff_keys(set(), diff_target_keys={"a.py"}, diff_repo_keys={"pkg/a.py"})
    assert resolver.diff_target_keys == {"a.py"}
    assert resolver.diff_repo_keys == {"pkg/a.py"}

    # 6. _probe_keys_in_diff with explicit sets
    assert resolver._probe_keys_in_diff(
        t_key="a.py",
        r_key=None,
        c_key=None,
        diff_keys=set(),
        has_tagged=True,
        diff_target_keys={"a.py"},
        diff_repo_keys=set(),
    )
    assert not resolver._probe_keys_in_diff(
        t_key="b.py",
        r_key=None,
        c_key=None,
        diff_keys=set(),
        has_tagged=True,
        diff_target_keys={"a.py"},
        diff_repo_keys=set(),
    )


def test_matches_diff_state_isolation_across_different_diff_keys(tmp_path: Path) -> None:
    """Verifies that consecutive matches_diff calls across different diff_keys do not leak state."""
    repo = tmp_path / "repo"
    repo.mkdir()
    f_a = repo / "a.py"
    f_a.write_text("x = 1\n", encoding="utf-8")
    f_b = repo / "b.py"
    f_b.write_text("y = 2\n", encoding="utf-8")

    resolver = CanonicalPathResolver(target_root=repo)
    assert resolver.diff_target_keys is None
    assert resolver.diff_repo_keys is None

    # Call 1 with diff containing target:a.py
    diff_a = {"target:a.py"}
    assert resolver.matches_diff("a.py", diff_a) is True
    assert resolver.matches_diff("b.py", diff_a) is False
    # Verify matches_diff does NOT mutate resolver.diff_target_keys
    assert resolver.diff_target_keys is None
    assert resolver.diff_repo_keys is None

    # Call 2 with completely separate diff containing target:b.py
    diff_b = {"target:b.py"}
    assert resolver.matches_diff("b.py", diff_b) is True
    # If state leaked from Call 1, "a.py" might evaluate to True; verify strict isolation
    assert resolver.matches_diff("a.py", diff_b) is False
    assert resolver.diff_target_keys is None

    # Call 3 with DiffPathKeySet
    dp_keys = DiffPathKeySet({"target:a.py"})
    assert resolver.matches_diff("a.py", dp_keys) is True
    assert resolver.matches_diff("b.py", dp_keys) is False

    # Now mutate dp_keys in place using discard
    dp_keys.discard("target:a.py")
    dp_keys.add("target:b.py")
    assert resolver.matches_diff("a.py", dp_keys) is False
    assert resolver.matches_diff("b.py", dp_keys) is True

    # Call 4: set_diff_keys binds to specific diff_keys
    resolver.set_diff_keys(diff_a)
    assert resolver.diff_target_keys == {"a.py"}
    assert resolver.matches_diff("a.py", diff_a) is True
    assert resolver.matches_diff("b.py", diff_a) is False

    # Calling with diff_b ignores resolver.diff_target_keys bound to diff_a because id() doesn't match
    assert resolver.matches_diff("b.py", diff_b) is True
    assert resolver.matches_diff("a.py", diff_b) is False


def test_cache_set_eviction_preserves_bound_diff_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _cache_set capacity eviction clears path cache without resetting bound diff keys."""
    repo = tmp_path / "repo"
    repo.mkdir()
    f1 = repo / "a.py"
    f1.write_text("x = 1\n", encoding="utf-8")

    resolver = CanonicalPathResolver(target_root=repo)
    resolver.set_diff_keys({"target:a.py"}, diff_target_keys={"a.py"}, diff_repo_keys=set())
    assert resolver.diff_target_keys == {"a.py"}
    assert resolver._bound_diff_id is not None

    # Resolve a path to populate _cache
    resolver.resolve("a.py")
    assert len(resolver._cache) > 0

    # Lower MAX_RESOLVER_CACHE_ENTRIES to force eviction on next _cache_set
    monkeypatch.setattr("pydoppelgangerhunt.canonical_path.MAX_RESOLVER_CACHE_ENTRIES", 1)

    # Resolve another path causing cache eviction
    resolver.resolve("b.py")

    # Verify path cache rolled over but bound diff keys survived
    assert len(resolver._cache) == 1
    assert resolver.diff_target_keys == {"a.py"}
    assert resolver._bound_diff_id is not None


def test_join_lexical_posix_empty_and_root_bases() -> None:
    """Verifies _join_lexical_posix distinguishes relative empty base from filesystem root."""
    assert _join_lexical_posix("", "foo/bar.py") == "foo/bar.py"
    assert _join_lexical_posix("/", "foo/bar.py") == "/foo/bar.py"
    assert _join_lexical_posix("/root", "foo/bar.py") == "/root/foo/bar.py"
    assert _join_lexical_posix("rel/dir", "foo/bar.py") == "rel/dir/foo/bar.py"


def test_resolver_resolve_none(tmp_path: Path) -> None:
    """Verifies that CanonicalPathResolver.resolve(None) safely returns empty CanonicalPath."""
    repo = tmp_path / "repo"
    repo.mkdir()
    resolver = CanonicalPathResolver(target_root=repo)
    resolved = resolver.resolve(None)
    assert resolved == CanonicalPath(raw="", repo_relative="", target_relative="", absolute_lexical="")


def test_parse_notebook_cell_anchor_canonical_path_object_transparency(tmp_path: Path) -> None:
    """Verifies parse_notebook_cell_anchor accepts CanonicalPath instances unwrapping .raw."""
    from pydoppelgangerhunt.canonical_path import parse_notebook_cell_anchor  # pylint: disable=import-outside-toplevel

    resolver = CanonicalPathResolver(target_root=tmp_path)
    cp = resolver.resolve("notebooks/analysis.ipynb#cell_4")
    parsed = parse_notebook_cell_anchor(cp)
    assert parsed == ("notebooks/analysis.ipynb", 3)

    cp_plain = resolver.resolve("notebooks/analysis.ipynb")
    assert parse_notebook_cell_anchor(cp_plain) is None

    cp_literal = resolver.resolve("notebooks/analysis.ipynb#cellular")
    assert parse_notebook_cell_anchor(cp_literal) is None


def test_resolve_coordinate_diff_keys_extracts_from_diff_path_key_set(tmp_path: Path) -> None:
    """Verifies that _resolve_coordinate_diff_keys extracts dt_keys and dr_keys from DiffPathKeySet."""
    repo = tmp_path / "repo"
    repo.mkdir()
    resolver = CanonicalPathResolver(target_root=repo)
    keys = DiffPathKeySet(["target:foo.py", "repo:src/foo.py"])
    dt_keys, dr_keys = resolver._resolve_coordinate_diff_keys(keys, None, None)
    assert dt_keys is keys.diff_target_keys
    assert dr_keys is keys.diff_repo_keys
    assert dt_keys == {"foo.py"}
    assert dr_keys == {"src/foo.py"}


def test_parse_notebook_cell_anchor_rejects_zero_and_negative_indices() -> None:
    """Verifies that parse_notebook_cell_anchor requires positive 1-based indices and rejects zero/negative/overflow."""
    from pydoppelgangerhunt.canonical_path import (  # pylint: disable=import-outside-toplevel
        normalize_lexical_posix,
        parse_notebook_cell_anchor,
    )

    # Valid positive 1-based anchors
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_1") == ("notebook.ipynb", 0)
    assert parse_notebook_cell_anchor("notebook.ipynb#cell1") == ("notebook.ipynb", 0)
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_42") == ("notebook.ipynb", 41)
    assert parse_notebook_cell_anchor("notebook.ipynb#cell01") == ("notebook.ipynb", 0)

    # Invalid zero-based or non-positive anchors
    assert parse_notebook_cell_anchor("notebook.ipynb#cell0") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_0") is None
    assert parse_notebook_cell_anchor("notebook.ipynb#cell_00") is None

    # Literal filename hashes containing #cell0 or #cell_0 are preserved and not stripped
    assert normalize_lexical_posix("notebook.ipynb#cell0", strip_anchor=True) == "notebook.ipynb#cell0"
    assert normalize_lexical_posix("notebook.ipynb#cell_0", strip_anchor=True) == "notebook.ipynb#cell_0"

    # Overflow / huge integers guarded safely
    huge_anchor = "notebook.ipynb#cell_" + ("9" * 5000)
    assert parse_notebook_cell_anchor(huge_anchor) is None


def test_matches_diff_in_place_set_mutation_invalidates_cached_coordinates(tmp_path: Path) -> None:
    """Verifies that in-place mutations of plain sets invalidate cached coordinate sets."""
    repo = tmp_path / "repo"
    repo.mkdir()
    resolver = CanonicalPathResolver(target_root=repo)

    # Initial plain set with two elements
    diff_keys = {"target:a.py", "target:b.py"}
    assert resolver.matches_diff("a.py", diff_keys) is True
    assert resolver.matches_diff("b.py", diff_keys) is True
    assert resolver.matches_diff("c.py", diff_keys) is False

    # Mutate plain set in-place: swap b.py for c.py (preserves set id, length, and sample key a.py)
    diff_keys.remove("target:b.py")
    diff_keys.add("target:c.py")

    # If stale coordinates were reused, c.py would be False and b.py would be True.
    # With content snapshot validation, cache invalidates and reflects true set state.
    assert resolver.matches_diff("c.py", diff_keys) is True
    assert resolver.matches_diff("b.py", diff_keys) is False
    assert resolver.matches_diff("a.py", diff_keys) is True



