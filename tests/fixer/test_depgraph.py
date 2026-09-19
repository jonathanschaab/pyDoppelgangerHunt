"""Tests for module dependency graph construction, import resolution, and cycle detection."""

from __future__ import annotations

from pathlib import Path
import pytest

from pydoppelgangerhunt.fixer.depgraph import (
    ModuleDependencyGraph,
    build_module_graph,
    derive_module_import_path,
    derive_shared_module_import,
    find_nearest_common_package,
    resolve_shared_module_file,
)


def test_derive_module_import_path_layouts(tmp_path: Path) -> None:
    """Verifies module dot-path derivation across flat, package, src, and init layouts."""
    root = tmp_path / "repo"
    root.mkdir()

    # Flat layout: root/module.py -> module
    flat_file = root / "simple_mod.py"
    flat_file.write_text("x = 1\n", encoding="utf-8")
    assert derive_module_import_path(flat_file, root) == "simple_mod"

    # Package layout: root/pkg/sub/worker.py -> pkg.sub.worker
    pkg_sub = root / "pkg" / "sub"
    pkg_sub.mkdir(parents=True)
    worker_file = pkg_sub / "worker.py"
    worker_file.write_text("x = 2\n", encoding="utf-8")
    assert derive_module_import_path(worker_file, root) == "pkg.sub.worker"

    # Init file: root/pkg/sub/__init__.py -> pkg.sub
    init_file = pkg_sub / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    assert derive_module_import_path(init_file, root) == "pkg.sub"

    # Src layout: root/src/mypkg/core.py -> mypkg.core
    src_core = root / "src" / "mypkg" / "core.py"
    src_core.parent.mkdir(parents=True)
    src_core.write_text("x = 3\n", encoding="utf-8")
    assert derive_module_import_path(src_core, root) == "mypkg.core"

    # Outside root or unresolvable
    other = tmp_path / "outside.py"
    assert derive_module_import_path(other, root) == "outside"

    # Relative path inputs resolved against repo root (regardless of process CWD)
    assert derive_module_import_path("pkg/sub/worker.py", root) == "pkg.sub.worker"
    assert derive_module_import_path("pkg\\sub\\worker.py", root) == "pkg.sub.worker"


def test_find_nearest_common_package(tmp_path: Path) -> None:
    """Verifies nearest common package directory resolution between files."""
    root = tmp_path / "project"
    root.mkdir()

    f1 = root / "pkg" / "mod_a.py"
    f2 = root / "pkg" / "mod_b.py"
    f1.parent.mkdir(parents=True)
    f1.write_text("", encoding="utf-8")
    f2.write_text("", encoding="utf-8")

    common = find_nearest_common_package(f1, f2, root)
    assert common == root / "pkg"

    # Sibling packages
    f_sub1 = root / "pkg" / "sub1" / "a.py"
    f_sub2 = root / "pkg" / "sub2" / "b.py"
    f_sub1.parent.mkdir(parents=True)
    f_sub2.parent.mkdir(parents=True)
    f_sub1.write_text("", encoding="utf-8")
    f_sub2.write_text("", encoding="utf-8")

    common_subs = find_nearest_common_package(f_sub1, f_sub2, root)
    assert common_subs == root / "pkg"

    # Divergent packages at repo root
    f_root1 = root / "alpha" / "a.py"
    f_root2 = root / "beta" / "b.py"
    f_root1.parent.mkdir(parents=True)
    f_root2.parent.mkdir(parents=True)
    common_root = find_nearest_common_package(f_root1, f_root2, root)
    assert common_root == root

    # Relative path inputs resolved against repo root (regardless of process CWD)
    assert find_nearest_common_package("pkg/mod_a.py", "pkg/mod_b.py", root) == root / "pkg"
    assert find_nearest_common_package("pkg\\mod_a.py", "pkg\\mod_b.py", root) == root / "pkg"
    assert resolve_shared_module_file("pkg/mod_a.py", "pkg/mod_b.py", root) == root / "pkg" / "_common.py"


def test_resolve_shared_module_file(tmp_path: Path) -> None:
    """Verifies target shared utility module file path resolution."""
    root = tmp_path / "app"
    root.mkdir()

    f1 = root / "services" / "srv1.py"
    f2 = root / "services" / "srv2.py"
    f1.parent.mkdir(parents=True)

    shared_path = resolve_shared_module_file(f1, f2, root)
    assert shared_path == root / "services" / "_common.py"

    custom_shared = resolve_shared_module_file(f1, f2, root, shared_module_name="utils")
    assert custom_shared == root / "services" / "utils.py"


def test_derive_shared_module_import_modes(tmp_path: Path) -> None:
    """Verifies canonical absolute and relative import derivation for shared modules."""
    root = tmp_path / "repo"
    root.mkdir()

    src_file = root / "pkg" / "services" / "worker.py"
    src_file.parent.mkdir(parents=True)
    shared_file = root / "pkg" / "_common.py"

    # Default absolute module path
    abs_imp = derive_shared_module_import(src_file, shared_file, root, prefer_relative=False)
    assert abs_imp == "pkg._common"

    # Relative import from subpackage to parent package
    rel_imp = derive_shared_module_import(src_file, shared_file, root, prefer_relative=True)
    assert rel_imp == ".._common"

    # Relative import within same directory
    sibling_file = root / "pkg" / "helper.py"
    sibling_rel = derive_shared_module_import(sibling_file, shared_file, root, prefer_relative=True)
    assert sibling_rel == "._common"

    # Relative string path inputs resolved against repo root (regardless of process CWD)
    assert derive_shared_module_import("pkg/services/worker.py", "pkg/_common.py", root) == "pkg._common"
    assert derive_shared_module_import("pkg\\services\\worker.py", "pkg\\_common.py", root) == "pkg._common"
    assert derive_shared_module_import("pkg/services/worker.py", "pkg/_common.py", root, prefer_relative=True) == ".._common"
    assert derive_shared_module_import("pkg\\services\\worker.py", "pkg\\_common.py", root, prefer_relative=True) == ".._common"


def test_resolve_shared_module_file_safe_sanitization(tmp_path: Path) -> None:
    """Verifies that malicious or tricky shared_module_name inputs cannot escape the common directory."""
    root = tmp_path / "app"
    root.mkdir()
    f1 = root / "services" / "srv1.py"
    f2 = root / "services" / "srv2.py"
    f1.parent.mkdir(parents=True)

    # Absolute path input does not escape common directory
    res_abs = resolve_shared_module_file(f1, f2, root, shared_module_name="/tmp/secret.py")
    assert res_abs == root / "services" / "secret.py"

    # Directory traversal input does not escape common directory
    res_trav = resolve_shared_module_file(f1, f2, root, shared_module_name="../../evil.py")
    assert res_trav == root / "services" / "evil.py"

    # Empty string, dots, and bare extension fall back safely to _common.py
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="..") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name=".") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name=".py") == root / "services" / "_common.py"

    # Invalid Python identifiers and reserved keywords fall back safely to _common.py
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="shared-module.py") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="helpers.py.py") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="123helper.py") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="class.py") == root / "services" / "_common.py"
    assert resolve_shared_module_file(f1, f2, root, shared_module_name="def") == root / "services" / "_common.py"


def test_derive_shared_module_import_top_level_fallback(tmp_path: Path) -> None:
    """Verifies that top-level modules without package context fall back to absolute imports."""
    root = tmp_path / "repo"
    root.mkdir()
    top_file = root / "main.py"
    shared_file = root / "pkg" / "sub" / "_common.py"
    shared_file.parent.mkdir(parents=True)

    # Even with prefer_relative=True, a top-level file at repo root must not use leading dots
    rel_imp = derive_shared_module_import(top_file, shared_file, root, prefer_relative=True)
    assert rel_imp == "pkg.sub._common"

    # Top-level file under src/ root also falls back to absolute import
    src_top = root / "src" / "cli.py"
    src_top.parent.mkdir(parents=True)
    src_shared = root / "src" / "app" / "_common.py"
    src_shared.parent.mkdir(parents=True)
    src_rel_imp = derive_shared_module_import(src_top, src_shared, root, prefer_relative=True)
    assert src_rel_imp == "app._common"


def test_module_dependency_graph_reachability_and_cycles(tmp_path: Path) -> None:
    """Verifies BFS reachability, cycle path extraction, and safety checks on directed graph."""
    graph = ModuleDependencyGraph(tmp_path)

    # Empty graph queries
    assert not graph.has_transitive_path("a", "b")
    assert graph.find_cycle_path("a", "b") is None
    assert graph.check_cycle_if_added("a", "a") == ["a", "a"]

    # Build chain: A -> B -> C -> D
    graph.add_module("mod_a", tmp_path / "mod_a.py")
    graph.add_module("mod_b", tmp_path / "mod_b.py")
    graph.add_module("mod_c", tmp_path / "mod_c.py")
    graph.add_module("mod_d", tmp_path / "mod_d.py")

    graph.add_dependency("mod_a", "mod_b")
    graph.add_dependency("mod_b", "mod_c")
    graph.add_dependency("mod_c", "mod_d")

    # Direct and transitive reachability
    assert graph.has_transitive_path("mod_a", "mod_b")
    assert graph.has_transitive_path("mod_a", "mod_d")
    assert not graph.has_transitive_path("mod_d", "mod_a")

    # Path reconstruction
    path_ad = graph.find_cycle_path("mod_a", "mod_d")
    assert path_ad == ["mod_a", "mod_b", "mod_c", "mod_d"]

    # Safe edge addition: D -> E (acyclic)
    assert graph.check_cycle_if_added("mod_d", "mod_e") is None

    # Dangerous edge addition: D -> A (would close cycle A -> B -> C -> D -> A)
    cycle = graph.check_cycle_if_added("mod_d", "mod_a")
    assert cycle == ["mod_d", "mod_a", "mod_b", "mod_c", "mod_d"]


def test_build_module_graph_from_repository_ast(tmp_path: Path) -> None:
    """Verifies AST-based repository parsing across absolute and relative imports."""
    root = tmp_path / "full_repo"
    pkg = root / "my_pkg"
    pkg.mkdir(parents=True)

    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_util = (
        "def helper() -> int:\n"
        "    return 42\n"
    )
    src_alpha = (
        "from .util import helper\n"
        "import my_pkg.beta\n"
        "\n"
        "def run_alpha() -> int:\n"
        "    return helper()\n"
    )
    src_beta = (
        "import os\n"
        "from my_pkg.util import helper\n"
        "\n"
        "def run_beta() -> int:\n"
        "    return helper() * 2\n"
    )
    (pkg / "util.py").write_text(src_util, encoding="utf-8")
    (pkg / "alpha.py").write_text(src_alpha, encoding="utf-8")
    (pkg / "beta.py").write_text(src_beta, encoding="utf-8")

    graph = build_module_graph(root)

    # Verify registered modules
    assert "my_pkg.util" in graph.mod_to_file
    assert "my_pkg.alpha" in graph.mod_to_file
    assert "my_pkg.beta" in graph.mod_to_file

    # Verify dependencies
    alpha_deps = graph.get_dependencies("my_pkg.alpha")
    assert "my_pkg.util" in alpha_deps
    assert "my_pkg.beta" in alpha_deps

    beta_deps = graph.get_dependencies("my_pkg.beta")
    assert "my_pkg.util" in beta_deps

    # Verify transitive paths
    assert graph.has_transitive_path("my_pkg.alpha", "my_pkg.util")
    assert graph.has_transitive_path("my_pkg.alpha", "my_pkg.beta")
    assert not graph.has_transitive_path("my_pkg.beta", "my_pkg.alpha")


def test_build_module_graph_ignores_type_checking_import_cycles(tmp_path: Path) -> None:
    """Verifies that TYPE_CHECKING-only imports do not create runtime dependency edges."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    import pkg.b\n",
        encoding="utf-8",
    )
    (pkg / "b.py").write_text("VALUE = 1\n", encoding="utf-8")

    graph = build_module_graph(root)

    assert "pkg.b" not in graph.get_dependencies("pkg.a")
    assert graph.check_cycle_if_added("pkg.b", "pkg.a") is None


def test_build_module_graph_package_root_preserves_package_prefix(tmp_path: Path) -> None:
    """Verifies graphs built from a package directory keep fully qualified module names."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    pkg = project_root / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "caller.py").write_text("def compute_c(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    (pkg / "mid.py").write_text(
        "import mypkg.caller\n\n"
        "def compute_m(x: int) -> int:\n"
        "    return mypkg.caller.compute_c(x)\n",
        encoding="utf-8",
    )
    (pkg / "host.py").write_text(
        "import mypkg.mid\n\n"
        "def compute_h(x: int) -> int:\n"
        "    return x + 1\n",
        encoding="utf-8",
    )

    graph = build_module_graph(pkg)

    assert "mypkg.host" in graph.mod_to_file
    assert "mypkg.mid" in graph.get_dependencies("mypkg.host")
    assert (
        graph.check_cycle_if_added("mypkg.caller", "mypkg.host")
        == ["mypkg.caller", "mypkg.host", "mypkg.mid", "mypkg.caller"]
    )


def test_depgraph_edge_cases(tmp_path: Path) -> None:
    """Verifies edge cases in depgraph for branch coverage."""
    # pylint: disable=protected-access
    from pydoppelgangerhunt.fixer.depgraph import (
        _parse_source_imports,
        _resolve_relative_import_path,
    )

    root = tmp_path / "edge_repo"
    root.mkdir()

    # 1. Directory inputs to find_nearest_common_package
    d1 = root / "dir1"
    d2 = root / "dir2"
    d1.mkdir()
    d2.mkdir()
    assert find_nearest_common_package(d1, d2, root) == root

    # 2. Path outside root for find_nearest_common_package
    outside = tmp_path.parent / "outside" / "path.py"
    assert find_nearest_common_package(outside, d1, root) == root

    # 3. derive_shared_module_import with down_parts and error fallback
    src_parent = root / "a.py"
    shared_child = root / "sub" / "pkg" / "_common.py"
    rel_down = derive_shared_module_import(src_parent, shared_child, root, prefer_relative=True)
    assert "sub.pkg._common" in rel_down

    assert derive_shared_module_import(src_parent, outside, root, prefer_relative=True) == "path"

    # 4. _resolve_relative_import_path
    assert _resolve_relative_import_path("", 1, "foo") == "foo"
    assert _resolve_relative_import_path("pkg.mod", 5, "foo") == "foo"
    assert _resolve_relative_import_path("pkg.sub.mod", 1, "util") == "pkg.sub.util"

    # 5. _parse_source_imports with syntax error and relative imports with no module name
    assert not _parse_source_imports("def broken(:\n", "mod")
    assert "pkg.helper" in _parse_source_imports("from . import helper\n", "pkg.mod")
    assert "pkg.sub.helper" in _parse_source_imports("from .sub import helper\n", "pkg.mod")

    # 6. derive_module_import_path for top-level or src __init__.py
    root_init = root / "__init__.py"
    root_init.write_text("", encoding="utf-8")
    assert derive_module_import_path(root_init, root) == ""

    src_init = root / "src" / "__init__.py"
    src_init.parent.mkdir(parents=True, exist_ok=True)
    src_init.write_text("", encoding="utf-8")
    assert derive_module_import_path(src_init, root) == ""

    # 7. _resolve_relative_import_path empty base/module
    assert _resolve_relative_import_path("pkg.mod", 1, "") == "pkg"
    assert _resolve_relative_import_path("", 1, "") == ""

    # 8. ModuleDependencyGraph edge cases
    graph = ModuleDependencyGraph(root)
    graph.add_module("", root / "empty.py")
    assert "" not in graph.adjacency

    graph.add_dependency("", "b")
    graph.add_dependency("a", "")
    graph.add_dependency("a", "a")
    assert "a" not in graph.adjacency

    assert not graph.has_transitive_path("", "b")
    assert not graph.has_transitive_path("a", "")
    assert not graph.has_transitive_path("a", "nonexistent")
    assert graph.has_transitive_path("same", "same")

    assert graph.find_cycle_path("", "b") is None
    assert graph.find_cycle_path("a", "") is None
    assert graph.find_cycle_path("a", "nonexistent") is None
    assert graph.find_cycle_path("same", "same") == ["same"]

    # Multi-path BFS exploration where neighbor is visited before target
    graph.add_module("x1", root / "x1.py")
    graph.add_module("x2", root / "x2.py")
    graph.add_module("x3", root / "x3.py")
    graph.add_dependency("x1", "x2")
    graph.add_dependency("x1", "x3")
    graph.add_dependency("x2", "x3")
    assert graph.find_cycle_path("x1", "x3") == ["x1", "x3"]

    # 9. build_from_repository with explicit file_paths and unreadable file
    f_ok = root / "ok.py"
    f_ok.write_text("x = 1\n", encoding="utf-8")
    f_not_py = root / "data.txt"
    f_not_py.write_text("hello", encoding="utf-8")
    g_explicit = build_module_graph(root, file_paths=[f_ok, f_not_py, root_init])
    assert "edge_repo.ok" in g_explicit.mod_to_file


def test_derive_shared_module_import_climbs_above_package(tmp_path: Path) -> None:
    """Verifies that relative imports climbing above top-level package fall back to absolute path."""
    root = tmp_path / "repo"
    root.mkdir()
    shared_at_root = root / "_common.py"
    pkg_sub = root / "pkg" / "sub"
    pkg_sub.mkdir(parents=True)
    src_mod = pkg_sub / "worker.py"
    src_mod.write_text("x = 1\n", encoding="utf-8")

    # Shared module at repo root
    assert (
        derive_shared_module_import(src_mod, shared_at_root, root, prefer_relative=True)
        == "_common"
    )

    # Source in top-level package and shared module in another top-level package
    other_pkg = root / "other"
    other_pkg.mkdir()
    shared_in_other = other_pkg / "_common.py"
    assert (
        derive_shared_module_import(src_mod, shared_in_other, root, prefer_relative=True)
        == "other._common"
    )

    # Relative file_paths in build_from_repository resolved against repo root
    g_rel = build_module_graph(root, file_paths=[Path("pkg/sub/worker.py")])
    assert "pkg.sub.worker" in g_rel.mod_to_file


def test_resolve_shared_module_file_unsafe_fallback_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if fallback _common.py resolves outside common_dir, ValueError is raised."""
    root = tmp_path / "app"
    root.mkdir()
    f1 = root / "services" / "srv1.py"
    f2 = root / "services" / "srv2.py"
    f1.parent.mkdir(parents=True)

    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")

    orig_resolve = Path.resolve

    def mock_resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "_common.py":
            return outside
        return orig_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", mock_resolve)
    with pytest.raises(
        ValueError, match="Shared module path resolves outside common package directory"
    ):
        resolve_shared_module_file(f1, f2, root, shared_module_name="_common.py")


def test_package_ancestor_initializer_cycle_detected(tmp_path: Path) -> None:
    """Verifies that submodule imports account for package initializer execution cycles."""
    root = tmp_path / "pkg_cycle_repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)

    # pkg/__init__.py imports caller module
    (pkg / "__init__.py").write_text("import caller\n", encoding="utf-8")
    (pkg / "worker.py").write_text("def do_work(): pass\n", encoding="utf-8")
    (root / "caller.py").write_text("def run(): pass\n", encoding="utf-8")

    graph = build_module_graph(root)

    # Submodule should model dependency on its parent package initializer
    assert "pkg" in graph.get_dependencies("pkg.worker")

    # caller -> pkg.worker would execute pkg/__init__.py, which imports caller -> CYCLE!
    cycle = graph.check_cycle_if_added("caller", "pkg.worker")
    assert cycle is not None
    assert cycle[0] == "caller"
    assert "pkg.worker" in cycle
    assert "pkg" in cycle
    assert cycle[-1] == "caller"


def test_depgraph_bfs_neighbor_caching_deterministic(tmp_path: Path) -> None:
    """Verifies that _get_sorted_neighbors caches sorted neighbors and invalidates on mutation."""
    # pylint: disable=protected-access
    graph = ModuleDependencyGraph(tmp_path)
    graph.add_module("a", tmp_path / "a.py")
    graph.add_module("b", tmp_path / "b.py")
    graph.add_module("c", tmp_path / "c.py")

    graph.add_dependency("a", "c")
    graph.add_dependency("a", "b")

    # First call caches sorted neighbors
    neighbors = graph._get_sorted_neighbors("a")
    assert neighbors == ["b", "c"]
    assert "a" in graph._sorted_adjacency

    # Adding a redundant edge does not invalidate cache
    graph.add_dependency("a", "b")
    assert "a" in graph._sorted_adjacency

    # Adding a new edge invalidates cache
    graph.add_dependency("a", "a0")
    assert "a" not in graph._sorted_adjacency
    assert graph._get_sorted_neighbors("a") == ["a0", "b", "c"]


def test_resolve_shared_module_file_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if target or fallback is an existing symlink, ValueError is raised."""
    root = tmp_path / "sym_repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    f1 = pkg / "f1.py"
    f2 = pkg / "f2.py"
    f1.write_text("x = 1\n", encoding="utf-8")
    f2.write_text("x = 2\n", encoding="utf-8")

    # Mock Path.is_symlink to simulate existing symlink
    orig_is_symlink = Path.is_symlink

    def mock_is_symlink(self: Path) -> bool:
        if self.name in ("_common.py", "custom.py"):
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    with pytest.raises(ValueError, match="existing symlink"):
        resolve_shared_module_file(f1, f2, root, shared_module_name="custom.py")

    with pytest.raises(ValueError, match="existing symlink"):
        resolve_shared_module_file(f1, f2, root, shared_module_name="_common.py")


def test_depgraph_pending_descendants_wiring(tmp_path: Path) -> None:
    """Verifies that pending package descendants are wired without quadratic scans."""
    # pylint: disable=protected-access
    repo = tmp_path / "repo"
    repo.mkdir()

    # Case 1: Submodule registered BEFORE ancestor packages
    g1 = ModuleDependencyGraph(repo)
    g1.add_module("pkg.sub.worker", repo / "pkg" / "sub" / "worker.py")
    assert "pkg" in g1._pending_descendants
    assert "pkg.sub" in g1._pending_descendants
    assert g1._pending_descendants["pkg"] == {"pkg.sub.worker"}
    assert g1._pending_descendants["pkg.sub"] == {"pkg.sub.worker"}

    g1.add_module("pkg.sub", repo / "pkg" / "sub" / "__init__.py")
    assert "pkg.sub" not in g1._pending_descendants
    assert "pkg.sub" in g1.adjacency["pkg.sub.worker"]

    g1.add_module("pkg", repo / "pkg" / "__init__.py")
    assert "pkg" not in g1._pending_descendants
    assert "pkg" in g1.adjacency["pkg.sub.worker"]
    assert "pkg" in g1.adjacency["pkg.sub"]
    assert not g1._pending_descendants

    # Case 2: Ancestor packages registered BEFORE submodule
    g2 = ModuleDependencyGraph(repo)
    g2.add_module("pkg", repo / "pkg" / "__init__.py")
    g2.add_module("pkg.sub", repo / "pkg" / "sub" / "__init__.py")
    g2.add_module("pkg.sub.worker", repo / "pkg" / "sub" / "worker.py")
    assert "pkg" in g2.adjacency["pkg.sub"]
    assert "pkg.sub" in g2.adjacency["pkg.sub.worker"]
    assert "pkg" in g2.adjacency["pkg.sub.worker"]
    assert not g2._pending_descendants


def test_derive_shared_module_import_preserves_package_prefix_for_package_root(tmp_path: Path) -> None:
    """Verifies derive_shared_module_import retains top-level package prefix when repo_root is a package."""
    proj = tmp_path / "project"
    pkg = proj / "mypkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub / "__init__.py").write_text("", encoding="utf-8")
    mod = sub / "worker.py"
    mod.write_text("x = 1\n", encoding="utf-8")
    shared_sub = sub / "_common.py"
    shared_pkg = pkg / "_common.py"

    imp1 = derive_shared_module_import(mod, shared_sub, repo_root=pkg, prefer_relative=False)
    assert imp1 == "mypkg.sub._common"

    imp2 = derive_shared_module_import(mod, shared_pkg, repo_root=pkg, prefer_relative=False)
    assert imp2 == "mypkg._common"


def test_resolve_shared_module_file_symlink_fallback_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that existing symlinks at target or fallback shared module path raise ValueError."""
    repo = tmp_path / "repo"
    repo.mkdir()
    f1 = repo / "pkg" / "mod1.py"
    f2 = repo / "pkg" / "mod2.py"
    f1.parent.mkdir()
    f1.write_text("", encoding="utf-8")
    f2.write_text("", encoding="utf-8")

    orig_is_symlink = Path.is_symlink

    # 1. Target itself is a symlink
    def mock_target_symlink(self: Path) -> bool:
        if self.name == "custom.py":
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", mock_target_symlink)
    with pytest.raises(ValueError, match="Shared module target path is an existing symlink"):
        resolve_shared_module_file(f1, f2, repo, shared_module_name="custom.py")

    # 2. Target fails containment and fallback _common.py is a symlink
    orig_resolve = Path.resolve

    def mock_resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "outside.py":
            return tmp_path / "other_dir" / "outside.py"
        return orig_resolve(self, strict=strict)

    def mock_fallback_symlink(self: Path) -> bool:
        if self.name == "_common.py":
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "resolve", mock_resolve)
    monkeypatch.setattr(Path, "is_symlink", mock_fallback_symlink)
    with pytest.raises(ValueError, match="Shared module fallback path is an existing symlink"):
        resolve_shared_module_file(f1, f2, repo, shared_module_name="outside.py")

    # 3. Target fails containment and fallback also resolves outside common package directory
    def mock_resolve_all_outside(self: Path, strict: bool = False) -> Path:
        return tmp_path / "other_dir" / self.name

    monkeypatch.setattr(Path, "resolve", mock_resolve_all_outside)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    with pytest.raises(ValueError, match="Shared module path resolves outside common package directory"):
        resolve_shared_module_file(f1, f2, repo, shared_module_name="outside.py")


def test_derive_shared_module_import_relative_and_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies prefer_relative behavior within subpackages and graceful fallback."""
    root = tmp_path / "repo"
    pkg = root / "mypkg"
    sub1 = pkg / "sub1"
    sub2 = pkg / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "__init__.py").write_text("", encoding="utf-8")

    f1 = sub1 / "caller.py"
    shared_same = sub1 / "_common.py"
    shared_sibling = sub2 / "_common.py"
    shared_parent = pkg / "_common.py"
    f1.write_text("", encoding="utf-8")
    shared_same.write_text("", encoding="utf-8")
    shared_sibling.write_text("", encoding="utf-8")
    shared_parent.write_text("", encoding="utf-8")

    # Same directory relative import
    imp_same = derive_shared_module_import(f1, shared_same, root, prefer_relative=True)
    assert imp_same == "._common"

    # Parent directory relative import
    imp_parent = derive_shared_module_import(f1, shared_parent, root, prefer_relative=True)
    assert imp_parent == ".._common"

    # Sibling directory relative import
    imp_sibling = derive_shared_module_import(f1, shared_sibling, root, prefer_relative=True)
    assert imp_sibling == "..sub2._common"

    # Cross-drive / relpath ValueError fallback to absolute
    import os  # pylint: disable=import-outside-toplevel
    def mock_relpath(path: str, start: str = ".") -> str:
        raise ValueError("Cannot compute relative path across drives")

    monkeypatch.setattr(os.path, "relpath", mock_relpath)
    imp_fallback = derive_shared_module_import(f1, shared_same, root, prefer_relative=True)
    assert imp_fallback == "mypkg.sub1._common"


def test_depgraph_detects_cycle_from_try_guarded_import(tmp_path: Path) -> None:
    """Verifies that imports inside top-level try blocks are captured and detect cycles."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "host.py").write_text(
        "try:\n"
        "    import pkg.caller\n"
        "except ImportError:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (pkg / "caller.py").write_text("VALUE = 1\n", encoding="utf-8")

    graph = build_module_graph(root)

    assert "pkg.caller" in graph.get_dependencies("pkg.host")
    cycle = graph.check_cycle_if_added("pkg.caller", "pkg.host")
    assert cycle == ["pkg.caller", "pkg.host", "pkg.caller"]


def test_depgraph_detects_cycle_from_class_body_import(tmp_path: Path) -> None:
    """Verifies that imports inside top-level class bodies are captured as runtime edges."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "host.py").write_text(
        "class Registry:\n"
        "    import pkg.caller\n",
        encoding="utf-8",
    )
    (pkg / "caller.py").write_text("VALUE = 1\n", encoding="utf-8")

    graph = build_module_graph(root)

    assert "pkg.caller" in graph.get_dependencies("pkg.host")
    cycle = graph.check_cycle_if_added("pkg.caller", "pkg.host")
    assert cycle == ["pkg.caller", "pkg.host", "pkg.caller"]


def test_depgraph_conditional_and_type_checking_matrix() -> None:
    """Verifies that _parse_source_imports correctly handles TYPE_CHECKING and runtime conditionals."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _parse_source_imports,
    )

    # 1. TYPE_CHECKING body skipped, orelse included
    src_tc = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import type_only\n"
        "else:\n"
        "    import runtime_fallback\n"
    )
    imports_tc = _parse_source_imports(src_tc, "pkg.mod")
    assert "type_only" not in imports_tc
    assert "runtime_fallback" in imports_tc

    # 2. not TYPE_CHECKING body included, orelse skipped
    src_not_tc = (
        "import typing\n"
        "if not typing.TYPE_CHECKING:\n"
        "    import runtime_active\n"
        "else:\n"
        "    import type_ignored\n"
    )
    imports_not_tc = _parse_source_imports(src_not_tc, "pkg.mod")
    assert "runtime_active" in imports_not_tc
    assert "type_ignored" not in imports_not_tc

    # 3. Runtime conditional: both branches conservatively included
    src_cond = (
        "import sys\n"
        "if sys.platform == 'win32':\n"
        "    import win32_backend\n"
        "else:\n"
        "    import unix_backend\n"
    )
    imports_cond = _parse_source_imports(src_cond, "pkg.mod")
    assert "win32_backend" in imports_cond
    assert "unix_backend" in imports_cond

    # 4. Context manager with block
    src_with = (
        "import contextlib\n"
        "with contextlib.suppress(ImportError):\n"
        "    import optional_dep\n"
    )
    imports_with = _parse_source_imports(src_with, "pkg.mod")
    assert "optional_dep" in imports_with


def test_check_cycle_if_imports_added_transactional_isolation(tmp_path: Path) -> None:
    """Verifies that check_cycle_if_imports_added does not mutate the graph."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text("import pkg.b\n", encoding="utf-8")
    (pkg / "b.py").write_text("x = 1\n", encoding="utf-8")

    graph = build_module_graph(root)
    assert graph.get_dependencies("pkg.b") == {"pkg"}

    # Propose adding import of pkg.a to pkg.b (would create cycle: b -> a -> b)
    cycle = graph.check_cycle_if_imports_added("pkg.b", ["import pkg.a"])
    assert cycle == ["pkg.b", "pkg.a", "pkg.b"]

    # Verify graph is NOT mutated by the check
    assert graph.get_dependencies("pkg.b") == {"pkg"}
    assert graph.check_cycle_if_added("pkg.b", "pkg.a") == ["pkg.b", "pkg.a", "pkg.b"]


def test_check_cycle_if_imports_added_reports_self_import_cycle(tmp_path: Path) -> None:
    """Verifies that transactional import checks reject synthesized self-imports."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    host_file = pkg / "host.py"
    host_file.write_text("class LocalDep:\n    pass\n", encoding="utf-8")

    graph = build_module_graph(root)

    cycle = graph.check_cycle_if_imports_added(
        "pkg.host",
        ["from pkg.host import LocalDep"],
        file_path=host_file,
    )
    assert cycle == ["pkg.host", "pkg.host"]


def test_check_cycle_if_imports_added_detects_self_import(tmp_path: Path) -> None:
    """Verifies that check_cycle_if_imports_added identifies self-imports as cycles."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "service.py").write_text("def run() -> None: pass\n", encoding="utf-8")

    graph = build_module_graph(root)

    # Propose self-import: pkg.service importing pkg.service
    self_cycle = graph.check_cycle_if_imports_added(
        "pkg.service", ["from pkg.service import run"]
    )
    assert self_cycle == ["pkg.service", "pkg.service"]


def test_depgraph_collect_top_level_import_nodes_in_class_def(tmp_path: Path) -> None:
    """Verifies that top-level class body imports are collected, while methods are skipped."""
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    (pkg / "caller.py").write_text("x = 10\n", encoding="utf-8")
    (pkg / "host.py").write_text(
        "class Registry:\n"
        "    from pkg.caller import x\n"
        "    def method(self) -> None:\n"
        "        import pkg.lazy_ignored\n",
        encoding="utf-8",
    )

    graph = build_module_graph(root)
    assert "pkg.caller" in graph.get_dependencies("pkg.host")
    assert "pkg.lazy_ignored" not in graph.get_dependencies("pkg.host")

    # Propose caller -> host: should detect cycle host -> caller -> host
    cycle = graph.check_cycle_if_added("pkg.caller", "pkg.host")
    assert cycle == ["pkg.caller", "pkg.host", "pkg.caller"]


def test_depgraph_conditional_comparisons_and_loops() -> None:
    """Verifies ast comparison expressions for TYPE_CHECKING and loop traversal."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _parse_source_imports,
    )

    src = (
        "import typing\n"
        "if typing.TYPE_CHECKING is True:\n"
        "    import tc_cmp_skipped\n"
        "else:\n"
        "    import tc_cmp_active\n"
        "if typing.TYPE_CHECKING is False:\n"
        "    import tc_false_active\n"
        "for _ in range(2):\n"
        "    import loop_dep\n"
    )
    imports = _parse_source_imports(src, "pkg.mod")
    assert "tc_cmp_skipped" not in imports
    assert "tc_cmp_active" in imports
    assert "tc_false_active" in imports
    assert "loop_dep" in imports

    # Null bytes do not raise ValueError and return empty set
    assert _parse_source_imports("x = 1\x00\nimport bad", "pkg.mod") == set()


def test_find_project_filesystem_root(tmp_path: Path) -> None:
    """Verifies that _find_project_filesystem_root accurately discovers project boundaries."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _find_project_filesystem_root,
    )

    # 1. Project with .git directory
    git_repo = tmp_path / "git_project"
    git_repo.mkdir()
    (git_repo / ".git").mkdir()
    sub_dir = git_repo / "sub" / "deep"
    sub_dir.mkdir(parents=True)
    assert _find_project_filesystem_root(sub_dir) == git_repo

    # 2. Project with pyproject.toml
    pyproject_repo = tmp_path / "pyproject_project"
    pyproject_repo.mkdir()
    (pyproject_repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    pkg_dir = pyproject_repo / "src" / "pkg"
    pkg_dir.mkdir(parents=True)
    assert _find_project_filesystem_root(pkg_dir) == pyproject_repo

    # 3. src layout without markers
    src_repo = tmp_path / "src_layout_project"
    src_repo.mkdir()
    src_pkg = src_repo / "src" / "pkg"
    src_pkg.mkdir(parents=True)
    assert _find_project_filesystem_root(src_pkg) == src_repo

    # 4. Nested package directory without __init__.py inside an ancestor package
    pkg_repo = tmp_path / "pkg_repo"
    pkg_repo.mkdir()
    top_pkg = pkg_repo / "my_pkg"
    top_pkg.mkdir()
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")
    nested_dir = top_pkg / "tools" / "utils"
    nested_dir.mkdir(parents=True)
    assert _find_project_filesystem_root(nested_dir) == pkg_repo

    # 5. Standalone directory without VCS or package context falls back to itself
    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir()
    assert _find_project_filesystem_root(plain_dir) == plain_dir


def test_resolve_repo_relative_path_prioritizes_repo_root_over_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that _resolve_repo_relative_path prioritizes repo_root over process CWD."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _resolve_repo_relative_path,
    )

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    (cwd_dir / "target.py").write_text("# cwd version\n", encoding="utf-8")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo_target = repo_dir / "target.py"
    repo_target.write_text("# repo version\n", encoding="utf-8")

    monkeypatch.chdir(cwd_dir)
    resolved = _resolve_repo_relative_path("target.py", repo_dir)
    assert resolved == repo_target
    assert resolved.read_text(encoding="utf-8") == "# repo version\n"


def test_resolve_repo_relative_path_rejects_parent_escape(tmp_path: Path) -> None:
    """Verifies that parent-relative paths cannot escape the provided repo root."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _resolve_repo_relative_path,
    )

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    safe_target = repo_dir / "safe.py"
    safe_target.write_text("# safe\n", encoding="utf-8")

    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")

    resolved_safe = _resolve_repo_relative_path("pkg/../safe.py", repo_dir)
    assert resolved_safe == safe_target

    escaped = _resolve_repo_relative_path("../outside.py", repo_dir)
    assert escaped != outside
    assert not escaped.exists()
    assert escaped.is_absolute()
    assert escaped.is_relative_to(repo_dir)


def test_find_enclosing_package_root_nested_subdirectory_without_init(tmp_path: Path) -> None:
    """Verifies that nested package directories without __init__.py discover the enclosing import root."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _find_enclosing_package_root,
        derive_module_import_path,
    )

    repo_root = tmp_path / "project"
    pkg = repo_root / "my_pkg"
    tools = pkg / "tools"
    tools.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mod_file = tools / "runner.py"
    mod_file.write_text("x = 1\n", encoding="utf-8")

    # tools has no __init__.py, but enclosing package is my_pkg with import root repo_root
    import_root = _find_enclosing_package_root(tools)
    assert import_root == repo_root
    assert derive_module_import_path(mod_file, import_root) == "my_pkg.tools.runner"


def test_find_enclosing_package_root_non_src_container(tmp_path: Path) -> None:
    """Verifies that custom non-src containers like lib/ derive the correct import root."""
    from pydoppelgangerhunt.fixer.depgraph import (  # pylint: disable=import-outside-toplevel
        _find_enclosing_package_root,
        derive_module_import_path,
    )

    project_root = tmp_path / "project"
    lib_dir = project_root / "lib"
    pkg = lib_dir / "custom_lib"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mod_file = pkg / "core.py"
    mod_file.write_text("y = 2\n", encoding="utf-8")

    import_root = _find_enclosing_package_root(pkg)
    assert import_root == lib_dir
    assert derive_module_import_path(mod_file, import_root) == "custom_lib.core"
