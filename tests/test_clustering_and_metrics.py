"""Unit tests for UnionFind clustering, linkage strategies, and DRY score metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pydoppelgangerhunt import (
    UnionFind,
    analyze_unit_variable_scope,
    cluster_clone_families,
    compute_repository_dry_stats,
    compute_unit_coverage,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _detect_indent_step,
    _inspect_unit_scope,
)


def test_union_find_clone_family_clustering() -> None:
    """Test UnionFind data structure and connected-component Clone Family clustering."""
    uf = UnionFind()
    assert uf.find("A") == "A"
    assert uf.find("B") == "B"
    root_ab = uf.union("A", "B")
    assert uf.find("A") == uf.find("B") == root_ab

    uf.union("B", "C")
    assert uf.find("C") == root_ab

    u_a = {"file": "mod_a.py", "start": 10, "end": 20, "name": "fn_a"}
    u_b = {"file": "mod_b.py", "start": 15, "end": 25, "name": "fn_b"}
    u_c = {"file": "mod_c.py", "start": 30, "end": 40, "name": "fn_c"}
    u_d = {"file": "mod_d.py", "start": 5, "end": 15, "name": "fn_d"}
    u_e = {"file": "mod_e.py", "start": 50, "end": 60, "name": "fn_e"}

    mock_clones = [
        (0.95, u_a, u_b),
        (0.92, u_b, u_c),
        (0.88, u_d, u_e),
    ]

    families = cluster_clone_families(mock_clones)
    assert len(families) == 2

    fam_3 = next(f for f in families if f["member_count"] == 3)
    assert len(fam_3["unique_files"]) == 3
    assert fam_3["member_count"] == 3
    assert 0.93 <= fam_3["avg_similarity"] <= 0.94
    assert fam_3["max_similarity"] == 0.95
    assert fam_3["total_lines"] == 33

    fam_2 = next(f for f in families if f["member_count"] == 2)
    assert fam_2["member_count"] == 2
    assert fam_2["avg_similarity"] == 0.88

def test_clustering_tie_breaking_determinism() -> None:
    """Verifies that clone family aggregation and tie-breaking are deterministic across edge orderings."""
    u1 = {"file": "pkg/a.py", "start": 10, "end": 20, "name": "fn_1"}
    u2 = {"file": "pkg/b.py", "start": 10, "end": 20, "name": "fn_2"}
    u3 = {"file": "pkg/c.py", "start": 10, "end": 20, "name": "fn_3"}
    u4 = {"file": "pkg/d.py", "start": 10, "end": 20, "name": "fn_4"}
    u5 = {"file": "pkg/e.py", "start": 10, "end": 20, "name": "fn_5"}
    u6 = {"file": "pkg/f.py", "start": 10, "end": 20, "name": "fn_6"}

    # Two independent triangles with identical similarity scores across all edges
    # (testing strict tie-breaking across multiple components)
    clones_forward = [
        (0.85, u1, u2),
        (0.85, u2, u3),
        (0.85, u1, u3),
        (0.85, u4, u5),
        (0.85, u5, u6),
        (0.85, u4, u6),
    ]

    # Reversed edge list
    clones_reversed = list(reversed(clones_forward))

    # Inverted unit pairs (u2, u1) instead of (u1, u2)
    clones_inverted = [(sim, u_b, u_a) for sim, u_a, u_b in clones_forward]

    for strategy in ("single", "complete", "quasi_complete", "average", "medoid"):
        fams_fwd = cluster_clone_families(clones_forward, linkage=strategy, min_similarity_floor=0.80)
        fams_rev = cluster_clone_families(clones_reversed, linkage=strategy, min_similarity_floor=0.80)
        fams_inv = cluster_clone_families(clones_inverted, linkage=strategy, min_similarity_floor=0.80)

        assert len(fams_fwd) == 2
        assert len(fams_rev) == 2
        assert len(fams_inv) == 2

        for f1, f2, f3 in zip(fams_fwd, fams_rev, fams_inv):
            assert f1["family_id"] == f2["family_id"] == f3["family_id"]
            assert f1["member_count"] == f2["member_count"] == f3["member_count"] == 3
            assert [m["name"] for m in f1["members"]] == [m["name"] for m in f2["members"]] == [m["name"] for m in f3["members"]]
            assert f1["medoid"]["name"] == f2["medoid"]["name"] == f3["medoid"]["name"]
            assert round(f1["avg_similarity"], 6) == round(f2["avg_similarity"], 6) == round(f3["avg_similarity"], 6)

def test_complete_linkage_sensitivity_and_quasi_complete(tmp_path: Path) -> None:
    """Verifies complete-linkage sensitivity tolerance and quasi_complete clustering on Type-3 clones."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    u_a = {"file": "mod_a.py", "start": 1, "end": 15, "name": "fn_a"}
    u_b = {"file": "mod_b.py", "start": 1, "end": 15, "name": "fn_b"}
    u_c = {"file": "mod_c.py", "start": 1, "end": 15, "name": "fn_c"}
    u_d = {"file": "mod_d.py", "start": 1, "end": 15, "name": "fn_d"}
    u_e = {"file": "mod_e.py", "start": 1, "end": 15, "name": "fn_e"}
    u_f = {"file": "mod_f.py", "start": 1, "end": 15, "name": "fn_f"}

    # Type-3 clone family {A, B, C, D} where A~D fluctuates slightly below 0.80 (0.77)
    # Nodes E and F form a separate cluster {E, F} with D~E being 0.80 (potential chaining candidate)
    clones = [
        (0.90, u_a, u_b),
        (0.88, u_b, u_c),
        (0.86, u_c, u_d),
        (0.85, u_e, u_f),
        (0.85, u_b, u_d),
        (0.82, u_a, u_c),
        (0.77, u_a, u_d),
        (0.80, u_d, u_e),
    ]

    # 1. Strict complete linkage (tolerance=0.0): over-partitions {A, B, C, D}
    # Because A~D is 0.77 < 0.80, D cannot merge into {A, B, C}. Nor can D merge into {E, F} (D~F is 0.0).
    f_strict = cluster_clone_families(clones, linkage="complete", min_similarity_floor=0.80, linkage_tolerance=0.0)
    # Result: {fn_a, fn_b, fn_c} of size 3, {fn_e, fn_f} of size 2. D remains an unclustered singleton.
    assert len(f_strict) == 2
    assert f_strict[0]["member_count"] == 3
    assert [m["name"] for m in f_strict[0]["members"]] == ["fn_a", "fn_b", "fn_c"]
    assert f_strict[1]["member_count"] == 2
    assert [m["name"] for m in f_strict[1]["members"]] == ["fn_e", "fn_f"]

    # 2. Complete linkage with tolerance 0.05:
    # 0.80 - 0.05 = 0.75. All cross-cluster edges between {A, B, C} and {D} are >= 0.75 (0.77, 0.85, 0.86)
    # and average cross-similarity is (0.77 + 0.85 + 0.86) / 3 = 0.8267 >= 0.80.
    # Therefore, {A, B, C, D} merges successfully!
    # {E, F} does NOT merge into {A, B, C, D} because edges E~A, E~B, E~C, F~A, etc. are 0.0 < 0.75.
    f_tolerant = cluster_clone_families(clones, linkage="complete", min_similarity_floor=0.80, linkage_tolerance=0.05)
    assert len(f_tolerant) == 2
    assert f_tolerant[0]["member_count"] == 4
    assert [m["name"] for m in f_tolerant[0]["members"]] == ["fn_a", "fn_b", "fn_c", "fn_d"]
    assert f_tolerant[1]["member_count"] == 2
    assert [m["name"] for m in f_tolerant[1]["members"]] == ["fn_e", "fn_f"]

    # 3. quasi_complete linkage: defaults to tolerance 0.05
    f_quasi = cluster_clone_families(clones, linkage="quasi_complete", min_similarity_floor=0.80)
    assert len(f_quasi) == 2
    assert f_quasi[0]["member_count"] == 4
    assert [m["name"] for m in f_quasi[0]["members"]] == ["fn_a", "fn_b", "fn_c", "fn_d"]
    assert f_quasi[1]["member_count"] == 2
    assert [m["name"] for m in f_quasi[1]["members"]] == ["fn_e", "fn_f"]

    # 4. CLI invocation with --linkage quasi_complete and --linkage-tolerance
    f1 = tmp_path / "mod1.py"
    f1.write_text("def worker(x, y):\n    v1 = x * 2\n    v2 = y * 2\n    return v1 + v2\n", encoding="utf-8")
    f2 = tmp_path / "mod2.py"
    f2.write_text("def worker(x, y):\n    v1 = x * 2\n    v2 = y * 2\n    return v1 + v2\n", encoding="utf-8")

    exit_code = main([
        str(tmp_path),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--cluster",
        "--linkage", "quasi_complete",
        "--linkage-tolerance", "0.05",
    ])
    assert exit_code == 1

def test_batch_37_directory_boundary_path_matching_and_exclude_filtering(tmp_path: Any) -> None:
    """Tests Batch 37: directory-boundary path matching and exclude filter safety."""
    from pydoppelgangerhunt.clustering import unit_key
    from pydoppelgangerhunt.config import find_python_files
    from pydoppelgangerhunt.coverage import compute_unit_coverage
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats

    # 1. compute_unit_diff_overlap avoids suffix false positives without directory boundary
    u_engine = {"file": "engine.py", "start": 10, "end": 20}
    # computation_engine.py shares suffix 'engine.py' but lacks '/' boundary
    overlap_false, ratio_false = compute_unit_diff_overlap(
        u_engine, {"computation_engine.py": [(10, 20)]}
    )
    assert overlap_false == 0
    assert ratio_false == 0.0

    # True matches with directory boundary and dot-slash
    overlap_true_sub, _ = compute_unit_diff_overlap(u_engine, {"sub/engine.py": [(10, 20)]})
    assert overlap_true_sub == 11
    overlap_true_dot, _ = compute_unit_diff_overlap(u_engine, {"./engine.py": [(10, 20)]})
    assert overlap_true_dot == 11

    # 2. compute_unit_coverage respects directory boundary
    cov_false = compute_unit_coverage(u_engine, {"computation_engine.py": {10, 11, 12}})
    assert cov_false == 0.0
    cov_sub = compute_unit_coverage(u_engine, {"sub/engine.py": set(range(10, 21))})
    assert cov_sub == 1.0
    cov_dot = compute_unit_coverage(u_engine, {"./engine.py": {10, 11}})
    assert round(cov_dot, 2) == 0.18

    # 3. find_python_files handles empty and slash-only exclude patterns safely
    py_file = tmp_path / "app.py"
    py_file.write_text("x = 1\n", encoding="utf-8")
    found = find_python_files(tmp_path, excludes=["", "   ", "/", "./"])
    assert len(found) == 1
    assert found[0].name == "app.py"

    # 4. unit_key normalizes leading dot-slash while preserving sub-document cell anchors
    u_dot = {"file": "./sub/app.py", "start": 1, "end": 5, "name": "f"}
    u_plain = {"file": "sub/app.py", "start": 1, "end": 5, "name": "f"}
    assert unit_key(u_dot) == unit_key(u_plain)
    u_cell1 = {"file": "./sub/app.py#cell_1", "start": 1, "end": 5, "name": "f"}
    u_cell2 = {"file": "sub/app.py#cell_2", "start": 1, "end": 5, "name": "f"}
    assert unit_key(u_cell1) != unit_key(u_cell2)
    assert unit_key(u_cell1) == "sub/app.py#cell_1:1-5:f"

    # 5. compute_repository_dry_stats de-duplicates lines across dot-slash file representations
    u_mod1 = {"file": "./mod.py", "start": 1, "end": 10, "name": "f1"}
    u_mod2 = {"file": "mod.py", "start": 1, "end": 10, "name": "f2"}
    mod_file = tmp_path / "mod.py"
    mod_file.write_text("x = 1\n" * 10, encoding="utf-8")
    stats = compute_repository_dry_stats(str(tmp_path), [(1.0, u_mod1, u_mod2)])
    assert stats["dloc"] == 10

def test_batch_39_notebook_cell_clustering_and_metrics_isolation(tmp_path: Any) -> None:
    """Tests Batch 39: notebook cell clustering isolation, notebook SLOC/DLOC metrics, and indent step detection."""
    import json
    from pydoppelgangerhunt.clustering import cluster_clone_families
    from pydoppelgangerhunt.fixer import _detect_indent_step
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats

    # 1. Notebook clone pairs across different cells form distinct members and families
    u_nb1 = {"file": "analysis.ipynb#cell_1", "start": 1, "end": 5, "name": "process_data"}
    u_nb2 = {"file": "analysis.ipynb#cell_2", "start": 1, "end": 5, "name": "process_data"}
    clones = [(0.95, u_nb1, u_nb2)]
    families = cluster_clone_families(clones)
    assert len(families) == 1
    fam = families[0]
    assert fam["member_count"] == 2
    assert fam["unique_files"] == ["analysis.ipynb"]
    assert fam["total_lines"] == 10
    assert fam["members"][0]["file"] == "analysis.ipynb#cell_1"
    assert fam["members"][1]["file"] == "analysis.ipynb#cell_2"

    # 2. compute_repository_dry_stats isolates DLOC lines across distinct notebook cells
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["def process_data():\n", "    # comment\n", "    return 42\n"],
            },
            {
                "cell_type": "code",
                "source": "def helper():\n    return 10\n",
            },
            {
                "cell_type": "markdown",
                "source": ["# Documentation\n"],
            },
        ]
    }
    nb_file = tmp_path / "analysis.ipynb"
    nb_file.write_text(json.dumps(nb_content), encoding="utf-8")

    stats = compute_repository_dry_stats(str(tmp_path), clones, include_notebooks=True)
    assert stats["dloc"] == 10
    assert stats["sloc"] == 4
    assert stats["clone_families"] == 1

    # 3. Corrupted notebook JSON handling
    corrupt_nb = tmp_path / "corrupt.ipynb"
    corrupt_nb.write_text("{broken json", encoding="utf-8")
    stats_corrupt = compute_repository_dry_stats(str(tmp_path), [], include_notebooks=True)
    assert stats_corrupt["sloc"] == 4

    # 4. _detect_indent_step coverage across tab, 2-space, 4-space, 6-space, and non-standard indents
    assert _detect_indent_step("\t") == "\t"
    assert _detect_indent_step("  ") == "  "
    assert _detect_indent_step("    ") == "    "
    assert _detect_indent_step("      ") == "  "
    assert _detect_indent_step("        ") == "    "
    assert _detect_indent_step("   ") == "    "
    assert _detect_indent_step("") == "    "

def test_batch_57_empty_res_has_receiver_access_and_metrics_package_sloc(tmp_path: Path) -> None:
    """Batch 57: Test empty_res has_receiver_access key parity and cross-platform package SLOC."""
    # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import _inspect_unit_scope, analyze_unit_variable_scope
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats

    # 1. Test _inspect_unit_scope on empty / whitespace units
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("   \n\n\t  \n", encoding="utf-8")
    u_empty = {"file": str(empty_file), "start": 1, "end": 3, "name": "empty_unit"}
    info_empty = _inspect_unit_scope(u_empty, repo_root=str(tmp_path))

    # Verify has_receiver_access is present and is False
    assert "has_receiver_access" in info_empty
    assert info_empty["has_receiver_access"] is False
    assert info_empty["has_instance_binding"] is False
    assert info_empty["has_class_binding"] is False
    assert info_empty["binding_kind"] is None

    # Verify key schema parity with populated unit
    valid_file = tmp_path / "valid.py"
    valid_file.write_text("def hello(self, x: int) -> int:\n    return self.val + x\n", encoding="utf-8")
    u_valid = {"file": str(valid_file), "start": 1, "end": 2, "name": "hello"}
    info_valid = _inspect_unit_scope(u_valid, repo_root=str(tmp_path))
    assert set(info_empty.keys()) == set(info_valid.keys())
    assert info_valid["has_receiver_access"] is True

    # 2. Test analyze_unit_variable_scope on empty unit
    scope_empty = analyze_unit_variable_scope(u_empty, repo_root=str(tmp_path))
    assert "has_receiver_access" in scope_empty
    assert scope_empty["has_receiver_access"] is False

    # 3. Test compute_repository_dry_stats package SLOC resolution with subpackages
    pkg_dir = tmp_path / "core_pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    mod_file = pkg_dir / "worker.py"
    mod_file.write_text("def work():\n    return 42\n", encoding="utf-8")

    stats = compute_repository_dry_stats(str(tmp_path), [])
    assert "core_pkg" in stats["package_sloc"]
    assert stats["package_sloc"]["core_pkg"] == 2
