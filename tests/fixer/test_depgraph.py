"""Tests for module dependency graph construction, import resolution, and cycle detection."""

from __future__ import annotations

from pathlib import Path

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
    assert "ok" in g_explicit.mod_to_file
