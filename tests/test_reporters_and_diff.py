"""Unit tests for reporters and diff."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

import pydoppelgangerhunt

from pydoppelgangerhunt import (
    UnionFind,
    analyze_unit_variable_scope,
    check_asymmetric_coverage,
    check_temporal_divergence,
    check_units_overlap,
    cluster_clone_families,
    compute_medoid,
    colorize,
    compute_repository_dry_stats,
    compute_unit_coverage,
    extract_unit_source_code,
    filter_clones_by_baseline,
    filter_clones_by_git_diff,
    find_enclosing_class,
    find_enclosing_function,
    format_github_annotations,
    format_json_report,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    generate_refactoring_patch,
    is_unit_in_modified_ranges,
    load_baseline,
    parse_git_diff_hunks,
    read_coverage_data,
    record_baseline,
    refactor_module_units,
    replace_unit_in_source,
    scan_target,
    supports_color,
    synthesize_refactoring_suggestion,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _analyze_block_assignment,
    _block_terminates,
    _build_whole_method_delegation,
    _detect_indent_step,
    _extract_deleted_names,
    _extract_required_typing_imports,
    _extract_unit_body_lines,
    _find_module_helper_insertion_index,
    _format_call_arguments,
    _insert_imports_into_module,
    _inspect_unit_scope,
    _is_method_of_class,
    _populate_unit_receiver_metadata,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.parser import harvest_file_units


def test_sarif_210_report_generation() -> None:
    """Test generation and structure of OASIS SARIF 2.1.0 report."""
    mock_clones = [
        (
            0.95,
            {
                "name": "func_alpha",
                "file": "src/module_a.py",
                "start": 10,
                "end": 25,
                "kind": "function",
                "token_count": 35,
            },
            {
                "name": "func_beta",
                "file": "src/module_b.py",
                "start": 30,
                "end": 45,
                "kind": "function",
                "token_count": 35,
            },
        )
    ]
    sarif = format_sarif_report(mock_clones, target="src", threshold=0.90)
    assert sarif["version"] == "2.1.0"
    assert "$schema" in sarif
    runs = sarif["runs"]
    assert len(runs) == 1
    driver = runs[0]["tool"]["driver"]
    assert driver["name"] == "pyDoppelgangerHunt"
    assert driver["rules"][0]["id"] == "PYDOPPEL001"
    results = runs[0]["results"]
    assert len(results) == 1
    assert results[0]["ruleId"] == "PYDOPPEL001"
    assert (
        results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == "src/module_a.py"
    )
    assert (
        results[0]["relatedLocations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == "src/module_b.py"
    )



def test_json_report_generation() -> None:
    """Test generation of structured JSON report representation."""
    mock_clones = [
        (
            0.88,
            {
                "name": "worker_one",
                "file": "worker.py",
                "start": 5,
                "end": 20,
                "kind": "compound_block",
                "token_count": 40,
            },
            {
                "name": "worker_two",
                "file": "worker.py",
                "start": 30,
                "end": 45,
                "kind": "compound_block",
                "token_count": 40,
            },
        )
    ]
    data = format_json_report(mock_clones, target="worker.py", threshold=0.85)
    assert data["tool"] == "pyDoppelgangerHunt"
    assert data["clone_count"] == 1
    assert data["clones"][0]["similarity"] == 0.88
    assert data["clones"][0]["unit_a"]["name"] == "worker_one"
    assert data["clones"][0]["unit_b"]["name"] == "worker_two"



def test_audit_tests_parametrize_candidate_detection(tmp_path: Path) -> None:
    """Test detection of copy-pasted test functions as @pytest.mark.parametrize candidates."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_example.py"
    test_file.write_text(
        """
def test_calc_tier_low():
    val = 10.0
    res = val * 2.0
    assert res == 20.0

def test_calc_tier_high():
    val = 50.0
    res = val * 2.0
    assert res == 100.0
""",
        encoding="utf-8",
    )
    clones = scan_target(
        str(test_dir),
        threshold=0.80,
        min_lines=3,
        min_tokens=5,
        audit_tests=True,
        blind_literals=True,
    )
    assert len(clones) >= 1
    u1, u2 = clones[0][1], clones[0][2]
    assert u1["name"] == "test_calc_tier_low" or u2["name"] == "test_calc_tier_low"



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



def test_generate_clone_diff_and_refactoring_suggestions(tmp_path: Path) -> None:
    """Test generation of unified clone diffs and synthesized refactoring suggestions."""
    file_a = tmp_path / "service_a.py"
    file_b = tmp_path / "service_b.py"

    file_a.write_text(
        "def process_order_v1(order_id):\n"
        "    tax = 0.05\n"
        "    total = calculate_total(order_id)\n"
        "    return total * (1.0 + tax)\n",
        encoding="utf-8",
    )

    file_b.write_text(
        "def process_order_v2(order_id):\n"
        "    tax = 0.08\n"
        "    total = calculate_total(order_id)\n"
        "    return total * (1.0 + tax)\n",
        encoding="utf-8",
    )

    u1 = {"file": str(file_a), "start": 1, "end": 4, "name": "process_order_v1"}
    u2 = {"file": str(file_b), "start": 1, "end": 4, "name": "process_order_v2"}

    lines1 = extract_unit_source_code(u1)
    assert len(lines1) == 4
    assert "process_order_v1" in lines1[0]

    missing_lines = extract_unit_source_code({"file": "non_existent.py", "start": 1, "end": 5, "name": "mock"})
    assert "Source for mock" in missing_lines[0]

    diff = generate_clone_diff(u1, u2)
    assert "---" in diff and "+++" in diff
    assert "-    tax = 0.05" in diff
    assert "+    tax = 0.08" in diff

    suggestion = synthesize_refactoring_suggestion(u1, u2)
    assert "[REFACTOR SUGGESTION]" in suggestion
    assert "_shared_process_order" in suggestion

    u_t1 = {"file": str(file_a), "start": 1, "end": 4, "name": "test_service_alpha"}
    u_t2 = {"file": str(file_b), "start": 1, "end": 4, "name": "test_service_beta"}
    test_suggestion = synthesize_refactoring_suggestion(u_t1, u_t2)
    assert "@pytest.mark.parametrize" in test_suggestion



def test_git_incremental_diff_filtering() -> None:
    """Test parsing of git diff hunk headers and incremental clone filtering."""
    sample_diff = """diff --git a/pkg/mod_one.py b/pkg/mod_one.py
--- a/pkg/mod_one.py
+++ b/pkg/mod_one.py
@@ -10,0 +12,8 @@
+def new_feature():
+    pass
diff --git a/pkg/mod_two.py b/pkg/mod_two.py
--- a/pkg/mod_two.py
+++ b/pkg/mod_two.py
@@ -50,3 +50,1 @@
-old
+new
"""
    hunks = parse_git_diff_hunks(sample_diff)
    assert "pkg/mod_one.py" in hunks
    assert (12, 19) in hunks["pkg/mod_one.py"]
    assert "pkg/mod_two.py" in hunks
    assert (50, 50) in hunks["pkg/mod_two.py"]

    unit_hit = {"file": "pkg/mod_one.py", "start": 15, "end": 22, "name": "feature"}
    unit_miss = {"file": "pkg/mod_one.py", "start": 1, "end": 10, "name": "header"}
    unit_other_file = {"file": "pkg/mod_unmodified.py", "start": 12, "end": 15, "name": "clean"}

    assert is_unit_in_modified_ranges(unit_hit, hunks) is True
    assert is_unit_in_modified_ranges(unit_miss, hunks) is False
    assert is_unit_in_modified_ranges(unit_other_file, hunks) is False

    mock_clones = [
        (0.95, unit_hit, unit_other_file),
        (0.90, unit_miss, unit_other_file),
    ]
    filtered = filter_clones_by_git_diff(mock_clones, hunks)
    assert len(filtered) == 1
    assert filtered[0][1]["name"] == "feature"



def test_dry_score_calculation_and_markdown_summary(tmp_path: Path) -> None:
    """Test calculation of repository DRY score, DLOC, and Markdown summary formatting."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    f1 = src_dir / "file1.py"
    f2 = src_dir / "file2.py"

    f1.write_text("\n".join([f"line_{i} = {i}" for i in range(1, 51)]), encoding="utf-8")
    f2.write_text("\n".join([f"line_{i} = {i}" for i in range(1, 51)]), encoding="utf-8")

    u1 = {"file": str(f1), "start": 10, "end": 19, "name": "block_a"}
    u2 = {"file": str(f2), "start": 20, "end": 29, "name": "block_b"}
    clones = [(0.95, u1, u2)]

    stats = compute_repository_dry_stats(str(src_dir), clones)
    assert stats["sloc"] == 100
    assert stats["dloc"] == 20
    assert stats["duplication_pct"] == 20.0
    assert stats["dry_score"] == 80.0
    assert stats["grade"] == "C"
    assert stats["clone_pairs"] == 1
    assert stats["clone_families"] == 1

    md_summary = format_markdown_summary(stats, target=str(src_dir))
    assert "## pyDoppelgangerHunt DRY Quality Audit Summary" in md_summary
    assert "| **Repository DRY Score** | **80.0% (Grade: C)** |" in md_summary
    assert "| **Total Source Lines (SLOC)** | 100 |" in md_summary
    assert "| **Duplicated Lines (DLOC)** | 20 |" in md_summary



def test_ansi_color_formatting(monkeypatch: Any) -> None:
    """Test ANSI terminal color formatting and NO_COLOR detection."""
    red_text = colorize("error", "\033[31m", enabled=True)
    assert "\033[31m" in red_text and "\033[0m" in red_text
    plain_text = colorize("error", "\033[31m", enabled=False)
    assert plain_text == "error"

    assert supports_color(color_override=True) is True
    assert supports_color(color_override=False) is False

    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(color_override=None) is False
    monkeypatch.delenv("NO_COLOR", raising=False)



def test_modular_pydoppelgangerhunt_exports() -> None:
    """Test that pydoppelgangerhunt package exports clean public API and metadata."""
    assert pydoppelgangerhunt.__version__ == "1.0.0"
    assert callable(pydoppelgangerhunt.main)
    assert callable(pydoppelgangerhunt.scan_target)
    assert callable(pydoppelgangerhunt.cluster_clone_families)
    assert callable(pydoppelgangerhunt.record_baseline)
    assert callable(pydoppelgangerhunt.load_baseline)
    assert callable(pydoppelgangerhunt.pure_structural_fingerprint)
    assert callable(pydoppelgangerhunt.extract_unit_namespace)
    assert callable(pydoppelgangerhunt.namespaced_structural_fingerprint)
    assert callable(pydoppelgangerhunt.prune_baseline)
    assert isinstance(pydoppelgangerhunt.DEFAULT_STOP_SHINGLES, set)
    assert callable(pydoppelgangerhunt.get_boilerplate_stop_shingles)
    assert callable(pydoppelgangerhunt.compute_unit_diff_overlap)
    assert pydoppelgangerhunt.MAJOR_POLICY_THRESHOLD == 0.50
    assert pydoppelgangerhunt.NEW_POLICY_THRESHOLD == 0.80
    assert callable(pydoppelgangerhunt.replace_unit_in_source)
    assert callable(pydoppelgangerhunt.check_units_overlap)
    assert callable(pydoppelgangerhunt.filter_overlapping_clone_units)
    assert callable(pydoppelgangerhunt.refactor_module_units)



def test_html_report_generation() -> None:
    """Test generating interactive standalone HTML reports."""
    u1 = {
        "file": "pkg/mod_a.py",
        "name": "func_a",
        "type": "function",
        "start": 10,
        "end": 25,
        "sloc": 16,
        "tokens": 45,
        "complexity": 3,
        "ast_node": None,
    }
    u2 = {
        "file": "pkg/mod_b.py",
        "name": "func_b",
        "type": "function",
        "start": 50,
        "end": 65,
        "sloc": 16,
        "tokens": 45,
        "complexity": 3,
        "ast_node": None,
    }
    clones = [(0.95, u1, u2)]
    families = cluster_clone_families(clones)
    stats = {
        "dry_score": 92.5,
        "grade": "A",
        "sloc": 2000,
        "dloc": 150,
        "duplication_pct": 7.5,
        "clone_pairs": 1,
        "clone_families": 1,
        "total_files": 10,
    }

    html = generate_html_report(clones, "pkg", 0.90, families=families, stats=stats)
    assert "<!DOCTYPE html>" in html
    assert "pyDoppelgangerHunt Audit Report" in html
    assert "92.5%" in html
    assert "DRY Score" in html
    assert "func_a" in html
    assert "func_b" in html
    assert "svg" in html



def test_github_actions_annotations() -> None:
    """Test emission of GitHub Actions workflow warning annotations."""
    u1 = {
        "file": "pkg/foo.py",
        "name": "calc",
        "type": "function",
        "start": 12,
        "end": 30,
    }
    u2 = {
        "file": "pkg/bar.py",
        "name": "compute",
        "type": "function",
        "start": 45,
        "end": 63,
    }
    clones = [(0.92, u1, u2)]
    annotations = format_github_annotations(clones)
    assert any("file=pkg/foo.py,line=12" in a for a in annotations)
    assert any("file=pkg/bar.py,line=45" in a for a in annotations)
    assert any("92.0%" in a for a in annotations)



def test_temporal_divergence_detection() -> None:
    """Test temporal divergence detection between commit dates."""
    u1 = {"file": "mod1.py", "start": 1, "end": 10}
    u2 = {"file": "mod2.py", "start": 1, "end": 10}

    now = 1700000000
    b1 = {"timestamp": now, "author": "Alice", "commit": "abc1234"}
    b2 = {"timestamp": now - (120 * 86400), "author": "Bob", "commit": "def5678"}

    with mock.patch("pydoppelgangerhunt.git_diff.get_git_blame_info", side_effect=[b1, b2]):
        res = check_temporal_divergence(u1, u2, max_divergence_days=90)
        assert res is not None
        assert res["divergence_days"] == 120.0
        assert res["newer"] == u1
        assert res["older"] == u2

    with mock.patch("pydoppelgangerhunt.git_diff.get_git_blame_info", side_effect=[b1, b1]):
        res_same = check_temporal_divergence(u1, u2, max_divergence_days=90)
        assert res_same is None



def test_coverage_readers_and_asymmetry(tmp_path: Path) -> None:
    """Test Cobertura XML and SQLite coverage readers and asymmetric coverage risk detection."""
    import sqlite3

    # 1. Cobertura XML
    xml_content = (
        '<?xml version="1.0" ?>\n'
        '<coverage version="7.0">\n'
        '  <packages>\n'
        '    <package name="pkg">\n'
        '      <classes>\n'
        '        <class name="mod_a" filename="pkg/mod_a.py">\n'
        '          <lines>\n'
        '            <line number="10" hits="1"/>\n'
        '            <line number="11" hits="1"/>\n'
        '            <line number="12" hits="0"/>\n'
        '          </lines>\n'
        '        </class>\n'
        '      </classes>\n'
        '    </package>\n'
        '  </packages>\n'
        '</coverage>\n'
    )
    xml_path = tmp_path / "coverage.xml"
    xml_path.write_text(xml_content, encoding="utf-8")

    cov_data = read_coverage_data(str(xml_path))
    assert "pkg/mod_a.py" in cov_data
    assert cov_data["pkg/mod_a.py"] == {10, 11}

    # 2. Unit coverage and asymmetry
    u1 = {"file": "pkg/mod_a.py", "start": 10, "end": 11}
    u2 = {"file": "pkg/mod_b.py", "start": 10, "end": 11}
    cov1 = compute_unit_coverage(u1, cov_data)
    cov2 = compute_unit_coverage(u2, cov_data)
    assert cov1 == 1.0
    assert cov2 == 0.0

    asym = check_asymmetric_coverage(u1, u2, cov_data, min_diff=0.40)
    assert asym is not None
    assert asym == (1.0, 0.0)

    # 3. SQLite .coverage reader
    db_path = tmp_path / ".coverage"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    cursor.execute("CREATE TABLE line_bits (file_id INTEGER, num_bits INTEGER, bits BLOB)")
    cursor.execute("INSERT INTO file VALUES (1, ?)", (str(tmp_path / "mod_sql.py"),))
    cursor.execute("INSERT INTO line_bits VALUES (1, 8, ?)", (bytes([1]),))
    conn.commit()
    conn.close()

    sql_data = read_coverage_data(str(db_path))
    matched_key = [k for k in sql_data if "mod_sql.py" in k]
    assert len(matched_key) == 1
    assert 1 in sql_data[matched_key[0]]

    # 4. SQLite .coverage reader with arc branch table
    db_arc = tmp_path / ".coverage_arc"
    conn_arc = sqlite3.connect(str(db_arc))
    cur_arc = conn_arc.cursor()
    cur_arc.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    cur_arc.execute("CREATE TABLE arc (file_id INTEGER, fromno INTEGER, tono INTEGER)")
    cur_arc.execute("INSERT INTO file VALUES (1, ?)", (str(tmp_path / "mod_arc.py"),))
    cur_arc.execute("INSERT INTO arc VALUES (1, 10, 12)")
    cur_arc.execute("INSERT INTO arc VALUES (1, -1, 10)")
    conn_arc.commit()
    conn_arc.close()

    sql_arc_data = read_coverage_data(str(db_arc))
    arc_key = [k for k in sql_arc_data if "mod_arc.py" in k]
    assert len(arc_key) == 1
    assert 10 in sql_arc_data[arc_key[0]]



def test_fixer_synthesis_and_patch(tmp_path: Path) -> None:
    """Test refactoring helper synthesis and unified patch generation."""
    code_a = (
        "def process_alpha(x, y):\n"
        "    v = x + y\n"
        "    return v * 2\n"
    )
    code_b = (
        "def process_beta(x, y):\n"
        "    v = x + y\n"
        "    return v * 3\n"
    )
    file_a = tmp_path / "proc_a.py"
    file_b = tmp_path / "proc_b.py"
    file_a.write_text(code_a, encoding="utf-8")
    file_b.write_text(code_b, encoding="utf-8")

    u1 = {
        "file": str(file_a),
        "name": "process_alpha",
        "type": "function",
        "start": 1,
        "end": 3,
        "params": ["x", "y"],
        "lines": 3,
    }
    u2 = {
        "file": str(file_b),
        "name": "process_beta",
        "type": "function",
        "start": 1,
        "end": 3,
        "params": ["x", "y"],
        "lines": 3,
    }

    helper = synthesize_shared_helper_code(u1, u2)
    assert "def _shared_process_alpha" in helper
    assert "v = x + y" in helper

    patch = generate_refactoring_patch([(0.85, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch
    assert "def _shared_process_alpha" in patch



def test_git_blame_porcelain_parsing(monkeypatch: Any) -> None:
    """Test get_git_blame_info parsing porcelain git blame outputs."""
    porcelain_sample = (
        "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 1 1 1\n"
        "author Ada Lovelace\n"
        "author-mail <ada@example.com>\n"
        "author-time 1700000000\n"
        "author-tz +0000\n"
        "committer Ada Lovelace\n"
        "committer-mail <ada@example.com>\n"
        "committer-time 1700000000\n"
        "committer-tz +0000\n"
        "summary Initial computation engine\n"
        "filename engine.py\n"
        "\tprint('hello')\n"
    )
    from pydoppelgangerhunt import git_diff
    monkeypatch.setattr(git_diff, "_run_git_command", lambda args, cwd=None: porcelain_sample)
    info = git_diff.get_git_blame_info("engine.py", 1, 10)
    assert info["author"] == "Ada Lovelace"
    assert info["commit"] == "a1b2c3d4"
    assert info["timestamp"] == 1700000000
    assert info["summary"] == "Initial computation engine"



def test_xml_coverage_parsing(tmp_path: Path) -> None:
    """Test read_coverage_data parsing standard Cobertura XML files."""
    cov_xml = tmp_path / "coverage.xml"
    cov_xml.write_text(
        """<?xml version="1.0" ?>
<coverage version="7.0">
  <packages>
    <package name="pkg">
      <classes>
        <class name="mod.py" filename="src/mod.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="0"/>
            <line number="3" hits="5"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )
    cov_map = read_coverage_data(str(cov_xml))
    assert "src/mod.py" in cov_map
    assert cov_map["src/mod.py"] == {1, 3}



def test_colored_clone_diff(tmp_path: Path) -> None:
    """Test generate_clone_diff with color=True and color=False."""
    f1 = tmp_path / "u1.py"
    f2 = tmp_path / "u2.py"
    f1.write_text("def f1():\n    a = 1\n    return a\n", encoding="utf-8")
    f2.write_text("def f2():\n    a = 2\n    return a\n", encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 3, "name": "f1"}
    u2 = {"file": str(f2), "start": 1, "end": 3, "name": "f2"}

    diff_plain = generate_clone_diff(u1, u2, color=False)
    assert "---" in diff_plain
    assert "+++" in diff_plain

    diff_colored = generate_clone_diff(u1, u2, color=True)
    assert "\033[" in diff_colored



def test_asymmetric_and_symmetric_coverage(tmp_path: Path) -> None:
    """Test check_asymmetric_coverage returns None on symmetric and tuple on asymmetric."""
    f1 = str((tmp_path / "u1.py").resolve()).replace("\\", "/")
    f2 = str((tmp_path / "u2.py").resolve()).replace("\\", "/")
    u1 = {"file": f1, "start": 1, "end": 10, "name": "f1"}
    u2 = {"file": f2, "start": 1, "end": 10, "name": "f2"}

    # Both 100% covered -> symmetric (None)
    cov_symmetric = {f1: set(range(1, 11)), f2: set(range(1, 11))}
    assert check_asymmetric_coverage(u1, u2, cov_symmetric, min_diff=0.40) is None

    # Empty coverage map
    assert check_asymmetric_coverage(u1, u2, {}) is None

    # Asymmetric: u1 has 10/10 (1.0), u2 has 1/10 (0.1)
    cov_asymmetric = {f1: set(range(1, 11)), f2: {1}}
    res = check_asymmetric_coverage(u1, u2, cov_asymmetric, min_diff=0.40)
    assert res is not None
    assert res[0] == 1.0
    assert res[1] == 0.1



def test_temporal_divergence_edge_cases(monkeypatch: Any) -> None:
    """Test check_temporal_divergence returns None when within divergence threshold or invalid timestamp."""
    from pydoppelgangerhunt import git_diff
    u1 = {"file": "a.py", "start": 1, "end": 5, "name": "a"}
    u2 = {"file": "b.py", "start": 1, "end": 5, "name": "b"}

    # Timestamps 10 days apart (< 90 days default)
    b1 = {"timestamp": 1700000000, "author": "Alice", "commit": "111", "summary": "s1"}
    b2 = {"timestamp": 1700000000 + 10 * 86400, "author": "Bob", "commit": "222", "summary": "s2"}
    monkeypatch.setattr(git_diff, "get_git_blame_info", lambda f, s, e, repo_root=None: b1 if f == "a.py" else b2)
    assert check_temporal_divergence(u1, u2) is None

    # Zero / missing timestamp
    b_zero = {"timestamp": 0, "author": "Unknown", "commit": "000", "summary": ""}
    monkeypatch.setattr(git_diff, "get_git_blame_info", lambda f, s, e, repo_root=None: b_zero)
    assert check_temporal_divergence(u1, u2) is None



def test_dry_score_grade_tiers(tmp_path: Path) -> None:
    """Test compute_repository_dry_stats letter grading from A+ down to F."""
    f = tmp_path / "code.py"
    f.write_text("x = 1\n" * 100, encoding="utf-8")

    # A+ (0 duplicates)
    s_a_plus = compute_repository_dry_stats(str(tmp_path), [])
    assert s_a_plus["grade"] == "A+"

    # A (4% duplication -> 96% dry)
    u1 = {"file": str(f), "start": 1, "end": 2, "name": "u1"}
    u2 = {"file": str(f), "start": 3, "end": 4, "name": "u2"}
    s_a = compute_repository_dry_stats(str(tmp_path), [(0.95, u1, u2)])
    assert s_a["grade"] == "A"

    # B (8% duplication -> 92% dry)
    u_b1 = {"file": str(f), "start": 1, "end": 4, "name": "u1"}
    u_b2 = {"file": str(f), "start": 5, "end": 8, "name": "u2"}
    s_b = compute_repository_dry_stats(str(tmp_path), [(0.95, u_b1, u_b2)])
    assert s_b["grade"] == "B"

    # C (16% duplication -> 84% dry)
    u_c1 = {"file": str(f), "start": 1, "end": 8, "name": "u1"}
    u_c2 = {"file": str(f), "start": 9, "end": 16, "name": "u2"}
    s_c = compute_repository_dry_stats(str(tmp_path), [(0.95, u_c1, u_c2)])
    assert s_c["grade"] == "C"

    # F (26% duplication -> 74% dry)
    u_f1 = {"file": str(f), "start": 1, "end": 13, "name": "u1"}
    u_f2 = {"file": str(f), "start": 14, "end": 26, "name": "u2"}
    s_f = compute_repository_dry_stats(str(tmp_path), [(0.95, u_f1, u_f2)])
    assert s_f["grade"] == "F"


def test_baseline_drift_immunity_across_line_shifts(tmp_path: Path) -> None:
    """Test that grandfathered baseline matches via structural hash despite line position shifts."""
    pkg = tmp_path / "baseline_pkg"
    pkg.mkdir()

    file_a = pkg / "worker_a.py"
    file_b = pkg / "worker_b.py"

    code_body = (
        "def run_computation(x, y):\n"
        "    acc = 0\n"
        "    for val in range(x):\n"
        "        acc += val * y\n"
        "    return acc\n"
    )
    file_a.write_text(code_body, encoding="utf-8")
    file_b.write_text(code_body, encoding="utf-8")

    initial_clones = scan_target(str(pkg), min_lines=3, min_tokens=10, threshold=0.90)
    assert len(initial_clones) >= 1
    baseline_path = tmp_path / "baseline.json"
    record_baseline(initial_clones, str(baseline_path), target=str(pkg), threshold=0.90)
    fps = load_baseline(str(baseline_path))

    # Shift lines in file_a by inserting 15 comment lines at top
    shifted_code = ("# Upstream edit shifting lines down\n" * 15) + code_body
    file_a.write_text(shifted_code, encoding="utf-8")

    shifted_clones = scan_target(str(pkg), min_lines=3, min_tokens=10, threshold=0.90)
    assert len(shifted_clones) >= 1
    _, u1, u2 = shifted_clones[0]
    # Verify line numbers indeed shifted
    assert u1["start"] >= 16 or u2["start"] >= 16

    # Verify structural fingerprint suppresses the clone despite the line shift
    remaining, suppressed = filter_clones_by_baseline(shifted_clones, fps)
    assert suppressed >= 1
    assert len(remaining) == 0


def test_fixer_scope_analysis_and_signature_synthesis(tmp_path: Path) -> None:
    """Test AST variable scope analysis and concrete parameter signature synthesis."""
    file_a = tmp_path / "math_a.py"
    file_b = tmp_path / "math_b.py"

    file_a.write_text(
        "def compute_coords(x: float, y: float) -> float:\n"
        "    scaled = (x * 2.0) + (y * 3.0)\n"
        "    return scaled\n",
        encoding="utf-8",
    )
    file_b.write_text(
        "def compute_coords_v2(x: float, y: float) -> float:\n"
        "    scaled = (x * 2.0) + (y * 3.0)\n"
        "    return scaled\n",
        encoding="utf-8",
    )

    u1 = {"file": str(file_a), "start": 1, "end": 3, "name": "compute_coords"}
    u2 = {"file": str(file_b), "start": 1, "end": 3, "name": "compute_coords_v2"}

    scope = analyze_unit_variable_scope(u1, u2)
    assert scope["inputs"] == ["x", "y"]
    assert "scaled" in scope["outputs"]
    assert "scaled" in scope["locals"]

    helper = synthesize_shared_helper_code(u1, u2)
    assert "def _shared_compute_coords" in helper
    assert "x: float, y: float" in helper
    assert "-> float:" in helper
    assert "*args" not in helper

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch
    assert "def _shared_compute_coords" in patch

    # Test single-unit scope analysis
    scope_single = analyze_unit_variable_scope(u1)
    assert scope_single["inputs"] == ["x", "y"]
    assert scope_single["return_type"] == "float"

    # Test async function with *args and **kwargs
    file_async = tmp_path / "async_service.py"
    file_async.write_text(
        "async def fetch_record(record_id: int, *extra_args: Any, **options: Any) -> int:\n"
        "    return record_id\n",
        encoding="utf-8",
    )
    u_async = {"file": str(file_async), "start": 1, "end": 2, "name": "fetch_record"}
    scope_async = analyze_unit_variable_scope(u_async)
    assert "record_id" in scope_async["inputs"]
    assert "extra_args" in scope_async["inputs"]
    assert "options" in scope_async["inputs"]
    assert "record_id" in scope_async["outputs"]
    assert scope_async["return_type"] == "int"

    # Test closure-captured variables from outer lexical scopes
    file_closure = tmp_path / "closure_service.py"
    file_closure.write_text(
        "def make_multiplier(factor: int):\n"
        "    def inner_mult(x: int) -> int:\n"
        "        return x * factor\n"
        "    return inner_mult\n",
        encoding="utf-8",
    )
    u_closure = {"file": str(file_closure), "start": 2, "end": 3, "name": "inner_mult"}
    scope_closure = analyze_unit_variable_scope(u_closure)
    assert "x" in scope_closure["inputs"]
    assert "factor" in scope_closure["inputs"]
    assert "factor" in scope_closure["free_vars"]
    assert scope_closure["return_type"] == "int"

    # Test nonlocal and global variable declarations
    file_scope_vars = tmp_path / "scope_vars.py"
    file_scope_vars.write_text(
        "def stateful_step():\n"
        "    global global_cache\n"
        "    nonlocal outer_counter\n"
        "    outer_counter += 1\n"
        "    global_cache = outer_counter\n"
        "    return outer_counter\n",
        encoding="utf-8",
    )
    u_state = {"file": str(file_scope_vars), "start": 1, "end": 6, "name": "stateful_step"}
    scope_state = analyze_unit_variable_scope(u_state)
    assert "outer_counter" in scope_state["nonlocals"]
    assert "global_cache" in scope_state["globals"]
    assert "outer_counter" in scope_state["inputs"]
    assert "outer_counter" in scope_state["outputs"]

    # Test class instance attributes (self.x, self.y)
    file_cls = tmp_path / "point_cls.py"
    file_cls.write_text(
        "class Point:\n"
        "    def translate(self, dx: float, dy: float = 0.0) -> None:\n"
        "        self.x += dx\n"
        "        self.y += dy\n",
        encoding="utf-8",
    )
    u_cls = {"file": str(file_cls), "start": 2, "end": 4, "name": "translate"}
    scope_cls = analyze_unit_variable_scope(u_cls)
    assert scope_cls["inputs"][0] == "self"
    assert "self.x" in scope_cls["attrs_read"]
    assert "self.x" in scope_cls["attrs_written"]
    assert "self.y" in scope_cls["attrs_read"]
    assert "self.y" in scope_cls["attrs_written"]
    assert scope_cls["return_type"] == "None"

    # Test keyword-only parameters and defaults preservation
    file_kw = tmp_path / "kw_service.py"
    file_kw.write_text(
        "def fetch_api(url: str, timeout: float = 5.0, *, retries: int = 3) -> dict:\n"
        "    return {'url': url}\n",
        encoding="utf-8",
    )
    u_kw = {"file": str(file_kw), "start": 1, "end": 2, "name": "fetch_api"}
    scope_kw = analyze_unit_variable_scope(u_kw)
    assert "url" in scope_kw["inputs"]
    assert "timeout" in scope_kw["inputs"]
    assert "retries" in scope_kw["inputs"]
    assert scope_kw["return_type"] == "dict"

    helper_kw = synthesize_shared_helper_code(u_kw, u_kw)
    assert "url: str" in helper_kw
    assert "timeout: float = 5.0" in helper_kw
    assert "*, retries: int = 3" in helper_kw
    assert "-> dict:" in helper_kw

    # Test conflicting defaults and conflicting types between clone units
    file_conf1 = tmp_path / "conf1.py"
    file_conf2 = tmp_path / "conf2.py"
    file_conf1.write_text(
        "def process_data(val: int, timeout: float = 10.0) -> int:\n"
        "    return val + int(timeout)\n",
        encoding="utf-8",
    )
    file_conf2.write_text(
        "def process_data_v2(val: float, timeout: float = 20.0) -> str:\n"
        "    return str(val + timeout)\n",
        encoding="utf-8",
    )
    u_c1 = {"file": str(file_conf1), "start": 1, "end": 2, "name": "process_data"}
    u_c2 = {"file": str(file_conf2), "start": 1, "end": 2, "name": "process_data_v2"}
    helper_conf = synthesize_shared_helper_code(u_c1, u_c2)
    sig_conf = helper_conf.split("\n")[0]
    # Conflicting types val: int vs float fallback to Any
    assert "val: Any" in sig_conf
    # Conflicting defaults 10.0 vs 20.0 are safely omitted from signature
    assert "timeout: float," in sig_conf or "timeout: float)" in sig_conf
    assert "= 10.0" not in sig_conf
    assert "= 20.0" not in sig_conf
    # Test nested function handling (top-level function guard)
    file_nested = tmp_path / "nested_func.py"
    file_nested.write_text(
        "def outer_calc(base: int) -> int:\n"
        "    def inner_step(step: int) -> int:\n"
        "        return base + step\n"
        "    return inner_step(5)\n",
        encoding="utf-8",
    )
    u_nested = {"file": str(file_nested), "start": 1, "end": 4, "name": "outer_calc"}
    scope_nested = analyze_unit_variable_scope(u_nested)
    assert scope_nested["inputs"] == ["base"]
    assert scope_nested["return_type"] == "int"

    # Test classmethod with cls parameter ordering
    file_cm = tmp_path / "cm_service.py"
    file_cm.write_text(
        "class Builder:\n"
        "    @classmethod\n"
        "    def build(cls, tag: str = 'item') -> str:\n"
        "        return f'{tag}'\n",
        encoding="utf-8",
    )
    u_cm = {"file": str(file_cm), "start": 2, "end": 4, "name": "build"}
    scope_cm = analyze_unit_variable_scope(u_cm)
    assert scope_cm["inputs"][0] == "cls"
    helper_cm = synthesize_shared_helper_code(u_cm, u_cm)
    assert (
        "def _shared_build(cls, tag: str = 'item') -> str:" in helper_cm
        or "def _shared_build(cls: Any, tag: str = 'item') -> str:" in helper_cm
    )

    # Test positional default order invalidation (clearing preceding defaults)
    file_ord1 = tmp_path / "ord1.py"
    file_ord2 = tmp_path / "ord2.py"
    file_ord1.write_text(
        "def order_func(a: int = 1, b: int = 2) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    file_ord2.write_text(
        "def order_func_v2(a: int = 1, b: int = 3) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    u_o1 = {"file": str(file_ord1), "start": 1, "end": 2, "name": "order_func"}
    u_o2 = {"file": str(file_ord2), "start": 1, "end": 2, "name": "order_func_v2"}
    helper_ord = synthesize_shared_helper_code(u_o1, u_o2)
    sig_ord = helper_ord.split("\n")[0]
    assert "a: int, b: int" in sig_ord
    assert "= 1" not in sig_ord

    # Test type_merge_strategy="union"
    helper_union = synthesize_shared_helper_code(u_c1, u_c2, type_merge_strategy="union")
    sig_union = helper_union.split("\n")[0]
    assert "Union[int, float]" in sig_union
    assert "Union[int, str]" in sig_union

    # Test read-only nonlocal variable (in inputs, but NOT in outputs)
    file_ro_nl = tmp_path / "ro_nonlocal.py"
    file_ro_nl.write_text(
        "def read_only_step(x: int) -> int:\n"
        "    nonlocal scale_factor\n"
        "    return x * scale_factor\n",
        encoding="utf-8",
    )
    u_ro_nl = {"file": str(file_ro_nl), "start": 1, "end": 3, "name": "read_only_step"}
    scope_ro_nl = analyze_unit_variable_scope(u_ro_nl)
    assert "scale_factor" in scope_ro_nl["inputs"]
    assert "scale_factor" in scope_ro_nl["nonlocals"]
    # Conservative check: scale_factor was only read, never stored, so NOT in outputs
    assert "scale_factor" not in scope_ro_nl["outputs"]

    # Test canonical argument kind ordering (pos -> vararg -> kwonly -> kwarg)
    file_full_args = tmp_path / "full_args.py"
    file_full_args.write_text(
        "def complex_sig(self, a: int, *args: Any, b: int = 1, **kwargs: Any) -> bool:\n"
        "    return True\n",
        encoding="utf-8",
    )
    u_fa = {"file": str(file_full_args), "start": 1, "end": 2, "name": "complex_sig"}
    helper_fa = synthesize_shared_helper_code(u_fa, u_fa)
    sig_fa = helper_fa.split("\n")[0]
    assert "self: Any, a: int, *args: Any, b: int = 1, **kwargs: Any" in sig_fa
    assert "-> bool:" in sig_fa


    # Test empty / invalid source unit
    scope_empty = analyze_unit_variable_scope({"file": "non_existent.py", "start": 1, "end": 1, "name": "bad"})
    assert scope_empty["inputs"] == []

    # Test vector fallback and invalid baseline JSON
    from pydoppelgangerhunt.baseline import compute_unit_structural_hash  # pylint: disable=import-outside-toplevel
    assert compute_unit_structural_hash({"structural_hash": "existing12345678"}) == "existing12345678"
    assert len(compute_unit_structural_hash({"tokens": ["alpha", "beta"]})) == 16
    h_vec = compute_unit_structural_hash({"vector": {"token_x": 3, "token_y": 1}})
    assert len(h_vec) == 16

    corrupt_json = tmp_path / "corrupt.json"
    corrupt_json.write_text("NOT_JSON_DATA", encoding="utf-8")
    assert load_baseline(str(corrupt_json)) == set()
    assert load_baseline(str(tmp_path / "absent.json")) == set()

    # Test fixer with no common lines fallback and non-existent patch targets
    file_diff1 = tmp_path / "diff1.py"
    file_diff2 = tmp_path / "diff2.py"
    file_diff1.write_text("def unique_one():\n    return 1\n", encoding="utf-8")
    file_diff2.write_text("def unique_two():\n    return 2\n", encoding="utf-8")
    u_d1 = {"file": str(file_diff1), "start": 1, "end": 2, "name": "unique_one"}
    u_d2 = {"file": str(file_diff2), "start": 1, "end": 2, "name": "unique_two"}
    helper_uncommon = synthesize_shared_helper_code(u_d1, u_d2)
    assert "_shared_unique_one_unique_two" in helper_uncommon

    u_missing = {"file": "no_such_file_on_disk.py", "start": 1, "end": 2, "name": "miss"}
    assert generate_refactoring_patch([(0.9, u_missing, u_d2)]) == ""

    # Test cli risk warnings helper
    from pydoppelgangerhunt import cli  # pylint: disable=import-outside-toplevel
    warn_u1 = {"file": "mod1.py", "start": 1, "end": 10, "name": "f1"}
    warn_u2 = {"file": "mod2.py", "start": 1, "end": 10, "name": "f2"}
    risk_lines = cli._audit_clone_risk_warnings(  # pylint: disable=protected-access
        warn_u1,
        warn_u2,
        audit_blame=True,
        cov_data={"mod1.py": {1, 2}, "mod2.py": set(range(1, 10))},
        use_color=False,
    )
    assert isinstance(risk_lines, list)


def test_fixer_control_flow_and_side_effect_safety(tmp_path: Path) -> None:
    """Verifies control flow hazard detection, multi-value returns, and import propagation."""
    # 1. Test naked break and continue detection in compound block
    file_ctrl = tmp_path / "ctrl_hazards.py"
    file_ctrl.write_text(
        "def outer_loop(items):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            break\n"
        "        if x == 0:\n"
        "            continue\n",
        encoding="utf-8",
    )
    # Unit representing the if block with break
    u_break = {
        "file": str(file_ctrl),
        "start": 3,
        "end": 4,
        "name": "outer_loop:If",
        "kind": "compound_block",
    }
    scope_break = analyze_unit_variable_scope(u_break)
    assert "naked_break" in scope_break["control_flow_hazards"]
    assert scope_break["is_control_flow_safe"] is False

    helper_break = synthesize_shared_helper_code(u_break, u_break)
    assert "WARNING: Non-local control flow hazard detected" in helper_break
    assert "naked_break" in helper_break

    # Unit representing the for loop with internal break
    u_loop = {
        "file": str(file_ctrl),
        "start": 2,
        "end": 6,
        "name": "outer_loop:For",
        "kind": "compound_block",
    }
    scope_loop = analyze_unit_variable_scope(u_loop)
    # Inside loop, break and continue are enclosed, so not naked
    assert "naked_break" not in scope_loop["control_flow_hazards"]
    assert "naked_continue" not in scope_loop["control_flow_hazards"]

    # 2. Test embedded return in compound block
    file_ret = tmp_path / "embedded_ret.py"
    file_ret.write_text(
        "def compute(val: int) -> int:\n"
        "    if val < 0:\n"
        "        return -1\n"
        "    return val * 2\n",
        encoding="utf-8",
    )
    u_emb_ret = {
        "file": str(file_ret),
        "start": 2,
        "end": 3,
        "name": "compute:If",
        "kind": "compound_block",
    }
    scope_emb_ret = analyze_unit_variable_scope(u_emb_ret)
    assert "embedded_return" in scope_emb_ret["control_flow_hazards"]
    assert scope_emb_ret["is_control_flow_safe"] is False

    helper_emb_ret = synthesize_shared_helper_code(u_emb_ret, u_emb_ret)
    assert "embedded_return" in helper_emb_ret

    # 3. Test generator yield detection
    file_gen = tmp_path / "gen_func.py"
    file_gen.write_text(
        "def number_stream(n: int):\n"
        "    for i in range(n):\n"
        "        yield i\n",
        encoding="utf-8",
    )
    u_gen = {"file": str(file_gen), "start": 1, "end": 3, "name": "number_stream", "kind": "function"}
    scope_gen = analyze_unit_variable_scope(u_gen)
    assert scope_gen["has_yield"] is True
    assert "generator_yield" in scope_gen["control_flow_hazards"]
    helper_gen = synthesize_shared_helper_code(u_gen, u_gen)
    assert "-> Iterator[Any]:" in helper_gen

    # 4. Test multiple local reassignments and tuple return synthesis
    file_mut = tmp_path / "mut_locals.py"
    file_mut.write_text(
        "def process_batch(items, delta: int):\n"
        "    total = 0\n"
        "    count = 0\n"
        "    total += delta\n"
        "    count += 1\n"
        "    return total, count\n",
        encoding="utf-8",
    )
    # Unit representing the two reassignments in lines 4-5
    u_mut = {
        "file": str(file_mut),
        "start": 4,
        "end": 5,
        "name": "process_batch:stmts_4-5",
        "kind": "sliding_window",
    }
    scope_mut = analyze_unit_variable_scope(u_mut)
    assert "total" in scope_mut["outputs"]
    assert "count" in scope_mut["outputs"]
    assert len(scope_mut["outputs"]) >= 2

    helper_mut = synthesize_shared_helper_code(u_mut, u_mut)
    assert "Tuple[" in helper_mut
    assert "return total, count" in helper_mut
    assert "Call site:" in helper_mut
    assert "total, count = _shared_process_batch(...)" in helper_mut

    # 5. Test multi-value return in whole function (return a, b)
    u_fn_ret = {
        "file": str(file_mut),
        "start": 1,
        "end": 6,
        "name": "process_batch",
        "kind": "function",
    }
    scope_fn_ret = analyze_unit_variable_scope(u_fn_ret)
    assert "total" in scope_fn_ret["outputs"]
    assert "count" in scope_fn_ret["outputs"]

    # 6. Test local import tracking and include_imports
    file_imp = tmp_path / "local_imports.py"
    file_imp.write_text(
        "def parse_config(path: str) -> dict:\n"
        "    import json\n"
        "    from collections import deque\n"
        "    with open(path) as fh:\n"
        "        return json.load(fh)\n",
        encoding="utf-8",
    )
    u_imp = {"file": str(file_imp), "start": 1, "end": 5, "name": "parse_config", "kind": "function"}
    scope_imp = analyze_unit_variable_scope(u_imp)
    assert "import json" in scope_imp["local_imports"]
    assert "from collections import deque" in scope_imp["local_imports"]

    helper_with_imports = synthesize_shared_helper_code(u_mut, u_mut, include_imports=True)
    assert "from typing import" in helper_with_imports
    assert "Tuple" in helper_with_imports

    # 7. Test missing import propagation in generate_refactoring_patch
    file_target = tmp_path / "patch_target.py"
    file_target.write_text(
        '"""Target module docstring."""\n\n'
        'from __future__ import annotations\n\n'
        'def run_step(a: int, b: int) -> int:\n'
        '    return a + b\n',
        encoding="utf-8",
    )
    u_patch1 = {"file": str(file_target), "start": 5, "end": 6, "name": "run_step"}
    patch = generate_refactoring_patch([(0.95, u_mut, u_patch1)], repo_root=str(tmp_path))
    assert "+from typing import" in patch
    assert "Tuple" in patch


def test_fixer_feedback_and_advanced_robustness(tmp_path: Path) -> None:
    """Verifies nested nonlocals, typed generator inference, import dedup, and AST typing extraction."""
    # 1. Nested function scoping with nonlocals (mutated vs read-only)
    file_nested = tmp_path / "nested_nonlocal.py"
    file_nested.write_text(
        "def outer():\n"
        "    count = 0\n"
        "    def inner():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    return inner\n",
        encoding="utf-8",
    )
    u_inner = {
        "file": str(file_nested),
        "start": 3,
        "end": 5,
        "name": "outer:inner",
        "kind": "function",
    }
    scope_inner = analyze_unit_variable_scope(u_inner)
    assert "count" in scope_inner["outputs"]

    file_ro = tmp_path / "nested_nonlocal_ro.py"
    file_ro.write_text(
        "def outer():\n"
        "    base = 10\n"
        "    def inner(x):\n"
        "        nonlocal base\n"
        "        return x + base\n"
        "    return inner\n",
        encoding="utf-8",
    )
    u_ro = {
        "file": str(file_ro),
        "start": 3,
        "end": 5,
        "name": "outer:inner",
        "kind": "function",
    }
    scope_ro = analyze_unit_variable_scope(u_ro)
    assert "base" in scope_ro["inputs"]
    assert "base" not in scope_ro["outputs"]

    # 2. Generator return type inference for yield from and typed yield
    file_gen_from = tmp_path / "gen_yield_from.py"
    file_gen_from.write_text(
        "def stream_items(src: List[str]):\n"
        "    yield from src\n",
        encoding="utf-8",
    )
    u_gen_from = {
        "file": str(file_gen_from),
        "start": 1,
        "end": 2,
        "name": "stream_items",
        "kind": "function",
    }
    helper_gen_from = synthesize_shared_helper_code(u_gen_from, u_gen_from, include_imports=True)
    assert "-> Iterator[str]:" in helper_gen_from
    assert "from typing import Iterator, List" in helper_gen_from

    file_gen_val = tmp_path / "gen_yield_val.py"
    file_gen_val.write_text(
        "def yield_single(val: int):\n"
        "    yield val\n",
        encoding="utf-8",
    )
    u_gen_val = {
        "file": str(file_gen_val),
        "start": 1,
        "end": 2,
        "name": "yield_single",
        "kind": "function",
    }
    helper_gen_val = synthesize_shared_helper_code(u_gen_val, u_gen_val)
    assert "-> Iterator[int]:" in helper_gen_val

    # 3. Import deduplication in _insert_imports_into_module
    orig_lines = ["from typing import Tuple\n", "import os\n"]
    res_dedup = _insert_imports_into_module(orig_lines, ["from typing import Tuple"])
    assert res_dedup == orig_lines

    new_lines = _insert_imports_into_module(orig_lines, ["from typing import List"])
    assert "from typing import List\n" in new_lines

    # 4. AST-based typing symbol extraction with shadowing parameter immunity
    shadow_func = "def helper(Tuple: int, List: str) -> None:\n    pass\n"
    extracted_shadow = _extract_required_typing_imports(shadow_func)
    assert "Tuple" not in extracted_shadow
    assert "List" not in extracted_shadow

    real_func = "def helper(items: List[str]) -> Tuple[int, Optional[str]]:\n    pass\n"
    extracted_real = _extract_required_typing_imports(real_func)
    assert extracted_real == ["List", "Optional", "Tuple"]

    # Raw signature without wrapper
    sig_raw = "x: Dict[str, Any]"
    extracted_sig = _extract_required_typing_imports(sig_raw)
    assert "Dict" in extracted_sig
    assert "Any" in extracted_sig


def test_clustering_transitivity_and_medoid_coherence(tmp_path: Path) -> None:
    """Verifies complete/average/medoid linkage strategies, chaining prevention, and medoid coherence."""
    import pytest  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    u_a = {"file": "mod_a.py", "start": 1, "end": 10, "name": "fn_a"}
    u_b = {"file": "mod_b.py", "start": 1, "end": 10, "name": "fn_b"}
    u_c = {"file": "mod_c.py", "start": 1, "end": 10, "name": "fn_c"}

    # 1. Transitive chaining test: A ~ B (0.95), B ~ C (0.90), no edge A ~ C
    clones_chain = [
        (0.95, u_a, u_b),
        (0.90, u_b, u_c),
    ]

    # Single linkage chains transitively into 1 family with 3 members
    f_single = cluster_clone_families(clones_chain, linkage="single")
    assert len(f_single) == 1
    assert f_single[0]["member_count"] == 3

    # Complete linkage (clique partitioning) isolates the strongest clique and drops chaining
    f_complete = cluster_clone_families(clones_chain, linkage="complete", min_similarity_floor=0.80)
    assert len(f_complete) == 1
    assert f_complete[0]["member_count"] == 2
    assert [m["name"] for m in f_complete[0]["members"]] == ["fn_a", "fn_b"]

    # Average linkage with floor 0.80 prevents merge because cross-cluster similarity is (0.0 + 0.90)/2 = 0.45 < 0.80
    f_average = cluster_clone_families(clones_chain, linkage="average", min_similarity_floor=0.80)
    assert len(f_average) == 1
    assert f_average[0]["member_count"] == 2

    # 2. Triangle clique: all edges present
    clones_triangle = [
        (0.95, u_a, u_b),
        (0.90, u_b, u_c),
        (0.85, u_a, u_c),
    ]
    for strategy in ("single", "complete", "average", "medoid"):
        f_tri = cluster_clone_families(clones_triangle, linkage=strategy, min_similarity_floor=0.80)
        assert len(f_tri) == 1
        assert f_tri[0]["member_count"] == 3
        assert f_tri[0]["medoid"]["name"] == "fn_b"
        assert f_tri[0]["min_similarity"] == 0.85
        assert f_tri[0]["max_similarity"] == 0.95
        assert round(f_tri[0]["coherence"], 4) == round((1.0 + 0.95 + 0.90) / 3, 4)

    # 3. compute_medoid standalone validation
    sim_map = {
        ("k1", "k2"): 0.9,
        ("k2", "k1"): 0.9,
        ("k2", "k3"): 0.8,
        ("k3", "k2"): 0.8,
    }
    m_key, m_score = compute_medoid(["k1", "k2", "k3"], sim_map)
    assert m_key == "k2"
    assert m_score > 0.8

    # Memoization validation
    memo: Dict[Tuple[str, ...], Tuple[str, float]] = {}
    m_key_c, m_score_c = compute_medoid(["k1", "k2", "k3"], sim_map, cache=memo)
    assert m_key_c == "k2"
    assert len(memo) == 1
    m_key_hit, m_score_hit = compute_medoid(["k3", "k2", "k1"], sim_map, cache=memo)
    assert (m_key_hit, m_score_hit) == (m_key_c, m_score_c)

    # Edge cases in compute_medoid
    single_key, single_score = compute_medoid(["only_one"], {})
    assert single_key == "only_one"
    assert single_score == 1.0

    with pytest.raises(ValueError, match="empty member set"):
        compute_medoid([], {})

    # Invalid linkage strategy raises ValueError
    with pytest.raises(ValueError, match="Unsupported linkage"):
        cluster_clone_families(clones_triangle, linkage="invalid_strategy")

    # Empty clones list returns empty families
    assert not cluster_clone_families([])

    # 4. JSON report serialization with medoid and coherence
    tri_fams = cluster_clone_families(clones_triangle, linkage="complete", min_similarity_floor=0.80)
    json_rep = format_json_report(clones_triangle, "target_pkg", 0.80, families=tri_fams)
    assert "families" in json_rep
    fam_entry = json_rep["families"][0]
    assert fam_entry["medoid"]["name"] == "fn_b"
    assert fam_entry["min_similarity"] == 0.85
    assert "coherence" in fam_entry

    # 5. HTML report serialization with medoid badge and coherence
    html_rep = generate_html_report(clones_triangle, "target_pkg", 0.80, families=tri_fams)
    assert "[medoid]" in html_rep
    assert "coherence" in html_rep

    # 6. CLI execution with --linkage complete and --min-cluster-similarity
    f1 = tmp_path / "cli_mod1.py"
    f1.write_text("def run_task(a, b):\n    print(a)\n    print(b)\n    return a + b\n", encoding="utf-8")
    f2 = tmp_path / "cli_mod2.py"
    f2.write_text("def run_task(a, b):\n    print(a)\n    print(b)\n    return a + b\n", encoding="utf-8")

    exit_code = main([
        str(tmp_path),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--cluster",
        "--linkage", "complete",
        "--min-cluster-similarity", "0.75",
    ])
    assert exit_code == 1


def test_path_independent_structural_fingerprint_rename_and_fallback(tmp_path: Path) -> None:
    """Test pure structural fingerprinting across file renames and legacy baseline fallback."""
    u1 = {"file": "dir_a/worker.py", "name": "do_work", "structural_hash": "hash_worker_01"}
    u2 = {"file": "dir_b/worker.py", "name": "do_work", "structural_hash": "hash_worker_02"}

    # Pure structural fingerprint is independent of file paths and order-invariant
    pure_fp = pydoppelgangerhunt.pure_structural_fingerprint(u1, u2)
    pure_fp_rev = pydoppelgangerhunt.pure_structural_fingerprint(u2, u1)
    assert pure_fp == pure_fp_rev
    assert "hash_worker_01" in pure_fp and "hash_worker_02" in pure_fp
    assert "dir_a" not in pure_fp and "dir_b" not in pure_fp

    # Simulate baseline recording
    baseline_path = tmp_path / "baseline_v120.json"
    pydoppelgangerhunt.record_baseline([(0.95, u1, u2)], str(baseline_path), "test_target", 0.90)

    # Move/rename dir_a/worker.py to dir_renamed/worker.py
    u1_renamed = {"file": "dir_renamed/worker.py", "name": "do_work", "structural_hash": "hash_worker_01"}
    active_clones = [(0.95, u1_renamed, u2)]

    # Old fingerprints (fp, sfp) won't match because path changed, but pure_sfp will match!
    loaded_fps = pydoppelgangerhunt.load_baseline(str(baseline_path))
    remaining, suppressed = pydoppelgangerhunt.filter_clones_by_baseline(active_clones, loaded_fps)
    assert suppressed == 1
    assert len(remaining) == 0

    # Test backward-compatibility with v1.0/v1.1 legacy baseline JSON lacking pure_structural_fingerprint
    legacy_baseline = tmp_path / "baseline_legacy.json"
    legacy_data = {
        "version": "1.1.0",
        "fingerprints": [
            {
                "fingerprint": "old_dir/a.py:calc <===> old_dir/b.py:calc",
                "structural_fingerprint": "old_dir/a.py#hash_c1 <===> old_dir/b.py#hash_c2",
                "hash_a": "hash_c1",
                "hash_b": "hash_c2",
            }
        ]
    }
    legacy_baseline.write_text(json.dumps(legacy_data), encoding="utf-8")
    loaded_legacy_fps = pydoppelgangerhunt.load_baseline(str(legacy_baseline))
    # Check that synthesized pure structural fingerprint is present
    expected_pure = "hash_c1 <===> hash_c2"
    assert expected_pure in loaded_legacy_fps

    # Verify a renamed clone matching hash_c1 and hash_c2 is suppressed using legacy baseline
    u_renamed_a = {"file": "new_dir/a.py", "name": "calc", "structural_hash": "hash_c1"}
    u_renamed_b = {"file": "new_dir/b.py", "name": "calc", "structural_hash": "hash_c2"}
    rem_legacy, supp_legacy = pydoppelgangerhunt.filter_clones_by_baseline(
        [(0.92, u_renamed_a, u_renamed_b)], loaded_legacy_fps
    )
    assert supp_legacy == 1
    assert len(rem_legacy) == 0


def test_baseline_pruning_and_cli(tmp_path: Path) -> None:
    """Test prune_baseline removing orphaned entries, synchronizing renames, and CLI integration."""
    u1 = {"file": "src/active_a.py", "name": "active_a", "structural_hash": "hash_act_a"}
    u2 = {"file": "src/active_b.py", "name": "active_b", "structural_hash": "hash_act_b"}
    u_dead1 = {"file": "src/dead_a.py", "name": "dead_a", "structural_hash": "hash_dead_a"}
    u_dead2 = {"file": "src/dead_b.py", "name": "dead_b", "structural_hash": "hash_dead_b"}
    u_moved1 = {"file": "src/old_loc.py", "name": "calc", "structural_hash": "hash_moved"}
    u_moved2 = {"file": "src/fixed_loc.py", "name": "calc", "structural_hash": "hash_fixed"}

    baseline_file = tmp_path / "prune_test_baseline.json"
    initial_clones = [
        (0.95, u1, u2),
        (0.92, u_dead1, u_dead2),
        (0.90, u_moved1, u_moved2),
    ]
    pydoppelgangerhunt.record_baseline(initial_clones, str(baseline_file), "test_pkg", 0.90)

    # Now simulate u_dead1/u_dead2 being deleted/refactored (not active)
    # and u_moved1 being moved to src/new_loc.py
    u_moved1_new = {"file": "src/new_loc.py", "name": "calc", "structural_hash": "hash_moved"}
    current_active = [
        (0.95, u1, u2),
        (0.90, u_moved1_new, u_moved2),
    ]

    pruned_count, retained_count = pydoppelgangerhunt.prune_baseline(str(baseline_file), current_active)
    assert pruned_count == 1
    assert retained_count == 2

    # Verify pruned baseline content
    content = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert content["clone_count"] == 2
    assert len(content["fingerprints"]) == 2
    # Verify moved clone was retained and synchronized to new file path
    fps_files = [(item["file_a"], item["file_b"]) for item in content["fingerprints"]]
    assert any("src/new_loc.py" in (f1, f2) for f1, f2 in fps_files)

    # Test edge cases: non-existent file, corrupt file, invalid JSON structure
    assert pydoppelgangerhunt.prune_baseline(str(tmp_path / "missing.json"), current_active) == (0, 0)
    corrupt_f = tmp_path / "corrupt_baseline.json"
    corrupt_f.write_text("invalid json content", encoding="utf-8")
    assert pydoppelgangerhunt.prune_baseline(str(corrupt_f), current_active) == (0, 0)
    not_dict_f = tmp_path / "not_dict.json"
    not_dict_f.write_text("[1, 2, 3]", encoding="utf-8")
    assert pydoppelgangerhunt.prune_baseline(str(not_dict_f), current_active) == (0, 0)

    # CLI test: --prune-baseline without baseline specified
    code_missing_base = pydoppelgangerhunt.main([str(tmp_path), "--prune-baseline"])
    assert code_missing_base == 1

    # CLI test: --prune-baseline with non-existent baseline file
    code_not_found = pydoppelgangerhunt.main([str(tmp_path), "--baseline", str(tmp_path / "no_such_file.json"), "--prune-baseline"])
    assert code_not_found == 1

    # CLI test: valid execution with --prune-baseline
    src_dir = tmp_path / "cli_src"
    src_dir.mkdir()
    f1 = src_dir / "mod1.py"
    f2 = src_dir / "mod2.py"
    shared_code = "def compute_val(x, y):\n    res = x * y\n    print(res)\n    return res + 1\n"
    f1.write_text(shared_code, encoding="utf-8")
    f2.write_text(shared_code, encoding="utf-8")

    cli_base = tmp_path / "cli_baseline.json"
    # First record baseline
    record_ret = pydoppelgangerhunt.main([str(src_dir), "--threshold", "0.70", "--min-lines", "3", "--min-tokens", "5", "--record-baseline", str(cli_base)])
    assert record_ret == 0
    assert cli_base.exists()

    # Now prune baseline (nothing pruned, 1 retained, clone suppressed)
    prune_ret = pydoppelgangerhunt.main([str(src_dir), "--threshold", "0.70", "--min-lines", "3", "--min-tokens", "5", "--baseline", str(cli_base), "--prune-baseline"])
    assert prune_ret == 0


def test_partial_hunk_policy_and_diff_overlap(tmp_path: Path) -> None:
    """Test compute_unit_diff_overlap, partial hunk policies ('any', 'major', 'new'), and CLI flags."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap, is_unit_in_modified_ranges, filter_clones_by_git_diff  # pylint: disable=import-outside-toplevel

    u_target = {"file": "pkg/service.py", "start": 10, "end": 29, "name": "process_records"}
    # 20 lines total: [10 .. 29]

    # 1. Incidental 1-line edit (line 15)
    ranges_incidental = {"pkg/service.py": [(15, 15)]}
    count_1, ratio_1 = compute_unit_diff_overlap(u_target, ranges_incidental)
    assert count_1 == 1
    assert ratio_1 == 0.05

    # 2. Major 12-line edit (lines 10..21)
    ranges_major = {"pkg/service.py": [(10, 21)]}
    count_12, ratio_12 = compute_unit_diff_overlap(u_target, ranges_major)
    assert count_12 == 12
    assert ratio_12 == 0.60

    # 3. Disjoint / non-matching file
    ranges_other = {"pkg/other.py": [(10, 20)]}
    assert compute_unit_diff_overlap(u_target, ranges_other) == (0, 0.0)

    # 4. Evaluate policy thresholds on incidental 1-line edit
    assert is_unit_in_modified_ranges(u_target, ranges_incidental, policy="any") is True
    assert is_unit_in_modified_ranges(u_target, ranges_incidental, policy="major") is False
    assert is_unit_in_modified_ranges(u_target, ranges_incidental, policy="new") is False
    assert is_unit_in_modified_ranges(u_target, ranges_incidental, min_overlap_ratio=0.10) is False

    # 5. Evaluate policy thresholds on major 60% edit
    assert is_unit_in_modified_ranges(u_target, ranges_major, policy="any") is True
    assert is_unit_in_modified_ranges(u_target, ranges_major, policy="major") is True
    assert is_unit_in_modified_ranges(u_target, ranges_major, policy="new") is False

    # 6. Evaluate policy thresholds on 90% rewrite
    ranges_new = {"pkg/service.py": [(10, 27)]}
    assert is_unit_in_modified_ranges(u_target, ranges_new, policy="new") is True

    # 7. Test filter_clones_by_git_diff with clone pair
    u_other = {"file": "pkg/other.py", "start": 1, "end": 20, "name": "legacy_helper"}
    mock_clones = [(0.95, u_target, u_other)]

    # Under "any", incidental edit triggers violation
    clones_any = filter_clones_by_git_diff(mock_clones, ranges_incidental, policy="any")
    assert len(clones_any) == 1

    # Under "major" or "new", incidental edit is ignored
    clones_major = filter_clones_by_git_diff(mock_clones, ranges_incidental, policy="major")
    assert len(clones_major) == 0
    clones_new = filter_clones_by_git_diff(mock_clones, ranges_incidental, policy="new")
    assert len(clones_new) == 0

    # Test both_units requirement
    clones_both = filter_clones_by_git_diff(mock_clones, ranges_major, policy="major", both_units=True)
    assert len(clones_both) == 0  # u_other is untouched

    # 8. CLI integration test with --diff-only, --partial-hunk-policy, --min-diff-overlap, and --stop-shingles
    cli_src = tmp_path / "diff_src"
    cli_src.mkdir()
    f_a = cli_src / "task_a.py"
    f_b = cli_src / "task_b.py"
    f_a.write_text("def run():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")
    f_b.write_text("def run():\n    x = 1\n    y = 2\n    return x + y\n", encoding="utf-8")

    cli_exit = pydoppelgangerhunt.main([
        str(cli_src),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--diff-only",
        "--partial-hunk-policy", "major",
        "--min-diff-overlap", "0.5",
        "--stop-shingles",
    ])
    # No git repo initialized in cli_src so modified_ranges is empty -> exits 0 (clean)
    assert cli_exit == 0


def test_conditional_variable_escape_prevention(tmp_path: Path) -> None:
    """Test that conditionally assigned variables are safely initialized to avoid runtime UnboundLocalError."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "def compute_summary(flag: bool):\n"
        "    if flag:\n"
        "        metric = 42\n"
        "    return flag\n"
    )
    src_file = tmp_path / "cond_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_block = {"file": str(src_file), "start": 2, "end": 3, "name": "compute_summary:If", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_block)
    assert "metric" in scope["conditional_outputs"]
    assert "flag" in scope["inputs"]

    helper_code = synthesize_shared_helper_code(u_block, u_block)
    assert "metric = None" in helper_code
    assert "Optional" in helper_code

    # Execute synthesized helper dynamically in namespace to verify no UnboundLocalError occurs when flag=False
    namespace: Dict[str, Any] = {}
    exec(helper_code, namespace)  # nosec
    helper_fn = namespace["_shared_compute_summary"]
    assert helper_fn(False) is None
    assert helper_fn(True) == 42


def test_async_coroutine_context_synthesis(tmp_path: Path) -> None:
    """Test that extracted blocks with await or async context synthesize async def and await call sites."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "async def process_item(item_id: int):\n"
        "    async with acquire_lock(item_id):\n"
        "        data = await fetch_remote(item_id)\n"
        "        return data\n"
    )
    src_file = tmp_path / "async_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_async = {"file": str(src_file), "start": 2, "end": 3, "name": "process_item:AsyncWith", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_async)
    assert scope["is_async"] is True
    assert "item_id" in scope["inputs"]

    helper_code = synthesize_shared_helper_code(u_async, u_async)
    assert helper_code.startswith("async def _shared_process_item")
    assert "await _shared_process_item" in helper_code
    # Must compile cleanly as an async function without SyntaxError
    compiled = compile(helper_code, "<test_async_helper>", "exec")
    assert compiled is not None


def test_global_and_nonlocal_scope_preservation(tmp_path: Path) -> None:
    """Test that variables altered via global/nonlocal preserve scope modifiers and avoid return-tuple leakage."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "def outer():\n"
        "    global global_metric\n"
        "    nonlocal captured_scale\n"
        "    global_metric += 1\n"
        "    captured_scale *= 2\n"
    )
    src_file = tmp_path / "scope_mod_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_scope = {"file": str(src_file), "start": 2, "end": 5, "name": "outer:compound", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_scope)
    assert "global_metric" in scope["globals"]
    assert "captured_scale" in scope["nonlocals"]

    helper_code = synthesize_shared_helper_code(u_scope, u_scope)
    # Synthesis must be declined for nonlocal-dependent units to avoid compile-time SyntaxError
    assert helper_code == ""

    # Global-only units must preserve global modifiers and avoid return-tuple leakage
    code_glob = (
        "def outer():\n"
        "    global global_metric\n"
        "    global_metric += 1\n"
    )
    src_glob = tmp_path / "scope_glob_src.py"
    src_glob.write_text(code_glob, encoding="utf-8")
    u_glob = {"file": str(src_glob), "start": 2, "end": 3, "name": "outer:compound", "kind": "compound_block"}
    helper_glob = synthesize_shared_helper_code(u_glob, u_glob)
    assert "global global_metric" in helper_glob
    assert "return" not in helper_glob


def test_comment_and_formatting_preservation(tmp_path: Path) -> None:
    """Test that replace_unit_in_source and generate_refactoring_patch preserve comments and formatting."""
    from pydoppelgangerhunt.fixer import generate_refactoring_patch, replace_unit_in_source  # pylint: disable=import-outside-toplevel

    source_code = (
        "# Top header license comment\n"
        "# type: ignore[module-attr]\n"
        "\n"
        "def calculate_tax(income: float) -> float:  # inline comment\n"
        "    # Step 1: Base deduction\n"
        "    rate = 0.20\n"
        "    return income * rate\n"
        "\n"
        "# Section divider comment\n"
        "def other_worker():\n"
        "    # worker notes\n"
        "    pass  # noqa: E501\n"
    )
    src_file = tmp_path / "tax_calc.py"
    src_file.write_text(source_code, encoding="utf-8")

    u_calc = {"file": str(src_file), "start": 4, "end": 7, "name": "calculate_tax", "kind": "function"}

    # 1. Test replace_unit_in_source
    call_stmt = "def calculate_tax(income: float) -> float:\n    return _shared_calculate_tax(income)\n"
    replaced = replace_unit_in_source(source_code, u_calc, call_stmt)

    assert "# Top header license comment\n" in replaced
    assert "# type: ignore[module-attr]\n" in replaced
    assert "# Section divider comment\n" in replaced
    assert "# worker notes\n" in replaced
    assert "pass  # noqa: E501\n" in replaced
    assert "return _shared_calculate_tax(income)" in replaced

    # 2. Test generate_refactoring_patch with replace_clones=True
    patch = generate_refactoring_patch([(0.95, u_calc, u_calc)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_calculate_tax" in patch
    assert "# Clone Pair" in patch


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


def test_corpus_sensitivity_stop_shingle_pruning_on_differential_runs(tmp_path: Path) -> None:
    """Verifies that dynamic stop-shingle pruning activates on small corpora (differential PR runs)."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # Create a small module with 6 distinct utility functions sharing common boilerplate logging/guards
    code_lines = []
    for i in range(6):
        code_lines.append(
            f"def compute_metric_{i}(data: int) -> int:\n"
            f"    # Standard boilerplate logger invocation\n"
            f"    if __name__ == '__main__':\n"
            f"        pass\n"
            f"    # Distinct algorithmic computation\n"
            f"    return data ** {i + 2} + {i * 100}\n"
        )
    src_file = tmp_path / "pr_diff_sample.py"
    src_file.write_text("\n".join(code_lines), encoding="utf-8")

    # With min_corpus_size=4 and max_index_frequency=0.25 on a 6-unit corpus:
    # max_posting_len = max(2, ceil(6 * 0.25)) = 2.
    # The boilerplate guard appearing in all 6 functions has posting len 6 > 2, so it is pruned!
    # Because each function's algorithm is distinct, no spurious clones are reported:
    clones_pruned = scan_target(
        str(tmp_path),
        threshold=0.85,
        min_lines=3,
        min_tokens=5,
        max_index_frequency=0.25,
        min_corpus_size=4,
    )
    assert len(clones_pruned) == 0


def test_inter_block_overlap_collision_detection_and_reverse_offset_refactoring(tmp_path: Path) -> None:
    """Verifies overlap collision detection, maximal subset filtering, and reverse-order refactoring."""
    import pytest  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        check_units_overlap,
        filter_overlapping_clone_units,
        refactor_module_units,
    )

    u_outer = {"file": "mod.py", "start": 10, "end": 30, "name": "outer_block"}
    u_inner = {"file": "mod.py", "start": 15, "end": 25, "name": "inner_block"}
    u_disjoint = {"file": "mod.py", "start": 40, "end": 50, "name": "disjoint_block"}
    u_other_file = {"file": "other.py", "start": 10, "end": 30, "name": "other_outer"}

    # 1. Test check_units_overlap
    assert check_units_overlap(u_outer, u_inner) is True
    assert check_units_overlap(u_inner, u_outer) is True
    assert check_units_overlap(u_outer, u_disjoint) is False
    assert check_units_overlap(u_outer, u_other_file) is False

    # 2. Test filter_overlapping_clone_units
    # Should prioritize u_outer (21 lines) over u_inner (11 lines), while keeping u_disjoint
    filtered = filter_overlapping_clone_units([u_inner, u_outer, u_disjoint])
    assert len(filtered) == 2
    names = [u["name"] for u in filtered]
    assert "outer_block" in names
    assert "disjoint_block" in names
    assert "inner_block" not in names

    # Empty list edge case
    assert filter_overlapping_clone_units([]) == []

    # 3. Test refactor_module_units overlap collision exception
    source_sample = (
        "line 1\nline 2\nline 3\nline 4\nline 5\n"
        "line 6\nline 7\nline 8\nline 9\nline 10\n"
    )
    rep_col1 = ({"file": "test.py", "start": 2, "end": 5, "name": "c1"}, "NEW_C1\n")
    rep_col2 = ({"file": "test.py", "start": 4, "end": 7, "name": "c2"}, "NEW_C2\n")

    with pytest.raises(ValueError, match="Overlapping unit collision detected"):
        refactor_module_units(source_sample, [rep_col1, rep_col2])

    # 4. Test refactor_module_units reverse line order application (no offset drift)
    # Block A: lines 2-4 (replaced with 1 line)
    # Block B: lines 7-9 (replaced with 1 line)
    rep_a = ({"file": "test.py", "start": 2, "end": 4, "name": "block_a"}, "REPLACED_A\n")
    rep_b = ({"file": "test.py", "start": 7, "end": 9, "name": "block_b"}, "REPLACED_B\n")

    # Pass in forward order: refactor_module_units should internally sort reverse
    refactored = refactor_module_units(source_sample, [rep_a, rep_b])
    expected_lines = [
        "line 1\n",
        "REPLACED_A\n",
        "line 5\n",
        "line 6\n",
        "REPLACED_B\n",
        "line 10\n",
    ]
    assert refactored == "".join(expected_lines)

    # Empty replacements edge case
    assert refactor_module_units(source_sample, []) == source_sample

    # 5. Test generate_refactoring_patch with overlapping clone units in the same file
    f_multi = tmp_path / "multi_clone.py"
    f_multi.write_text(
        "def first_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n"
        "\n"
        "def second_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n",
        encoding="utf-8",
    )
    u_t1 = {"file": str(f_multi), "start": 1, "end": 4, "name": "first_task"}
    u_t2 = {"file": str(f_multi), "start": 6, "end": 9, "name": "second_task"}

    # Non-overlapping clones in the same file are both safely refactored in reverse order
    patch = generate_refactoring_patch([(0.95, u_t1, u_t2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_first_task_second_task" in patch
    assert "_shared_first_task_second_task(x)" in patch


def test_cross_namespace_collision_prevention_and_granular_tracking(tmp_path: Path) -> None:
    """Test that identical boilerplate across different modules is disambiguated and tracked granularly."""
    bp_hash = "hash_boilerplate_common"
    u_auth1 = {"file": "auth/views.py", "name": "to_dict", "structural_hash": bp_hash}
    u_auth2 = {"file": "auth/helpers.py", "name": "to_dict", "structural_hash": bp_hash}

    u_bill1 = {"file": "billing/views.py", "name": "to_dict", "structural_hash": bp_hash}
    u_bill2 = {"file": "billing/helpers.py", "name": "to_dict", "structural_hash": bp_hash}

    # Verify extract_unit_namespace
    assert pydoppelgangerhunt.extract_unit_namespace("auth/views.py") == "auth"
    assert pydoppelgangerhunt.extract_unit_namespace("billing/helpers.py") == "billing"
    assert pydoppelgangerhunt.extract_unit_namespace("standalone.py") == "."
    assert pydoppelgangerhunt.extract_unit_namespace("./nested/dir/mod.py") == "nested/dir"

    # Verify namespaced structural fingerprints
    ns_auth = pydoppelgangerhunt.namespaced_structural_fingerprint(u_auth1, u_auth2)
    ns_bill = pydoppelgangerhunt.namespaced_structural_fingerprint(u_bill1, u_bill2)
    assert ns_auth != ns_bill
    assert "auth#hash_boilerplate_common" in ns_auth
    assert "billing#hash_boilerplate_common" in ns_bill

    # Pure structural fingerprint is identical
    pure_auth = pydoppelgangerhunt.pure_structural_fingerprint(u_auth1, u_auth2)
    pure_bill = pydoppelgangerhunt.pure_structural_fingerprint(u_bill1, u_bill2)
    assert pure_auth == pure_bill

    # Record baseline grandfathering ONLY the auth/ boilerplate pair
    base_file = tmp_path / "baseline_disambig.json"
    pydoppelgangerhunt.record_baseline([(0.98, u_auth1, u_auth2)], str(base_file), "test_ns", 0.90)

    loaded_base = pydoppelgangerhunt.load_baseline(str(base_file))

    # Test 1: Scan both clone pairs -> auth is suppressed, billing is NOT (detected as new)
    active_clones = [
        (0.98, u_auth1, u_auth2),
        (0.98, u_bill1, u_bill2),
    ]
    new_clones, suppressed = pydoppelgangerhunt.filter_clones_by_baseline(active_clones, loaded_base)
    assert suppressed == 1
    assert len(new_clones) == 1
    assert new_clones[0][1]["file"] == "billing/views.py"

    # Test 2: Even if auth/ pair is absent, billing/ cannot hijack auth's grandfathered record
    active_bill_only = [(0.98, u_bill1, u_bill2)]
    new_clones_bill, suppressed_bill = pydoppelgangerhunt.filter_clones_by_baseline(active_bill_only, loaded_base)
    assert suppressed_bill == 0
    assert len(new_clones_bill) == 1

    # Test 3: Granular 1-to-1 matching within the same namespace
    u_auth3 = {"file": "auth/extra.py", "name": "to_dict", "structural_hash": bp_hash}
    u_auth4 = {"file": "auth/other.py", "name": "to_dict", "structural_hash": bp_hash}
    two_auth_pairs = [
        (0.98, u_auth1, u_auth2),
        (0.98, u_auth3, u_auth4),
    ]
    new_two, supp_two = pydoppelgangerhunt.filter_clones_by_baseline(two_auth_pairs, loaded_base)
    assert supp_two == 1
    assert len(new_two) == 1
    assert new_two[0][1]["file"] == "auth/extra.py"

    # Test 4: File rename resilience within the same namespace
    u_auth1_renamed = {"file": "auth/auth_views.py", "name": "to_dict", "structural_hash": bp_hash}
    renamed_active = [(0.98, u_auth1_renamed, u_auth2)]
    new_renamed, supp_renamed = pydoppelgangerhunt.filter_clones_by_baseline(renamed_active, loaded_base)
    assert supp_renamed == 1
    assert len(new_renamed) == 0


def test_stale_unstaged_diff_protection_during_baseline_pruning(tmp_path: Path) -> None:
    """Test that prune_baseline protects inactive entries touching unstaged git changes."""
    u_auth1 = {"file": "src/auth/views.py", "name": "login", "structural_hash": "hash_auth_login"}
    u_auth2 = {"file": "src/auth/helpers.py", "name": "login", "structural_hash": "hash_auth_login"}
    u_clean1 = {"file": "src/calc/math.py", "name": "add", "structural_hash": "hash_clean_add"}
    u_clean2 = {"file": "src/calc/utils.py", "name": "add", "structural_hash": "hash_clean_add"}

    base_file = tmp_path / "prune_dirty_baseline.json"
    pydoppelgangerhunt.record_baseline(
        [(0.95, u_auth1, u_auth2), (0.95, u_clean1, u_clean2)],
        str(base_file),
        "test_prune",
        0.90,
    )

    active_clones: List[Any] = []
    dirty_unstaged = {"src/auth/views.py": [(10, 20)]}

    prune_res = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_clones,
        unstaged_modified_ranges=dirty_unstaged,
    )
    assert isinstance(prune_res, tuple)
    pruned, retained = prune_res
    assert pruned == 1
    assert retained == 1
    assert prune_res.skipped_dirty_count == 1
    assert prune_res.pruned_count == 1
    assert prune_res.retained_count == 1

    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert data["clone_count"] == 1
    assert data["fingerprints"][0]["file_a"] == "src/auth/views.py"

    # Clean worktree prunes the remaining entry
    prune_res_clean = pydoppelgangerhunt.prune_baseline(
        str(base_file),
        active_clones,
        unstaged_modified_ranges={},
    )
    assert prune_res_clean.pruned_count == 1
    assert prune_res_clean.retained_count == 0
    assert prune_res_clean.skipped_dirty_count == 0

    data_clean = json.loads(base_file.read_text(encoding="utf-8"))
    assert data_clean["clone_count"] == 0
    assert len(data_clean["fingerprints"]) == 0


def test_cli_prune_baseline_with_unstaged_diff(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    """Test CLI output for --prune-baseline when unstaged git changes protect entries."""
    src_dir = tmp_path / "cli_dirty"
    src_dir.mkdir()
    f1 = src_dir / "mod1.py"
    f2 = src_dir / "mod2.py"
    code = "def compute_val(x, y):\n    res = x * y\n    print(res)\n    return res + 1\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    cli_base = tmp_path / "cli_prune_test.json"
    pydoppelgangerhunt.main([
        str(src_dir),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--record-baseline", str(cli_base),
    ])
    assert cli_base.exists()

    # Now mod2 has different function, mod1 has unstaged changes
    f2.write_text("def different_function():\n    return 42\n", encoding="utf-8")
    norm_f1 = str(f1.resolve()).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff.get_git_modified_line_ranges",
        lambda since_ref=None, repo_root=None: {norm_f1: [(1, 5)]},
    )

    exit_code = pydoppelgangerhunt.main([
        str(src_dir),
        "--threshold", "0.70",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--baseline", str(cli_base),
        "--prune-baseline",
    ])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "skipped due to unstaged git changes" in captured


def test_slice_source_by_token_range() -> None:
    """Tests precise character/token-range slicing across lines and columns."""
    from pydoppelgangerhunt import slice_source_by_token_range  # pylint: disable=import-outside-toplevel

    code = (
        "def example(alpha: int, beta: str) -> None:\n"
        "    first_val = 10; second_val = 20  # inline\n"
        "    return None\n"
    )

    # 1. Single-line slice: "first_val = 10" is columns 4 to 18 on line 2
    sl1 = slice_source_by_token_range(code, 2, 4, 2, 18)
    assert sl1 == "first_val = 10"

    # 2. Multi-line slice
    sl2 = slice_source_by_token_range(code, 1, 4, 2, 18)
    assert sl2.startswith("example(alpha: int, beta: str) -> None:\n    first_val = 10")

    # 3. None end_col includes through end of line
    sl3 = slice_source_by_token_range(code, 3, 4, 3, None)
    assert sl3 == "return None\n"

    # 4. Out of bounds and edge cases
    assert slice_source_by_token_range("", 1, 0, 1, 5) == ""
    assert slice_source_by_token_range(code, 0, 0, 1, 5) == ""
    assert slice_source_by_token_range(code, 5, 0, 10, 5) == ""
    assert slice_source_by_token_range(code, 3, 0, 2, 5) == ""


def test_extract_unit_comments_and_pragmas() -> None:
    """Tests extraction and classification of comments, # type: ignore, and # noqa pragmas."""
    from pydoppelgangerhunt import extract_unit_comments_and_pragmas  # pylint: disable=import-outside-toplevel

    code = (
        "# Top comment\n"
        "# Another header\n"
        "def compute():\n"
        "    x = 1  # inline note\n"
        "    y = 2  # type: ignore[assignment]\n"
        "    z = 3  # noqa: E501\n"
        "    w = 4  # pylint: disable=unused-variable\n"
    )

    # Harvest lines 3 through 7
    comments = extract_unit_comments_and_pragmas(code, 3, 7)
    assert len(comments) == 4
    texts = [c["text"] for c in comments]
    assert "# inline note" in texts
    assert "# type: ignore[assignment]" in texts
    assert "# noqa: E501" in texts
    assert "# pylint: disable=unused-variable" in texts

    # Verify classification
    pragmas = [c for c in comments if c["is_pragma"]]
    assert len(pragmas) == 3
    kinds = {c["pragma_kind"] for c in pragmas}
    assert kinds == {"type_ignore", "noqa", "pylint"}

    # Leading comments harvesting
    leading = extract_unit_comments_and_pragmas(code, 3, 4, include_leading=True)
    leading_texts = [c["text"] for c in leading]
    assert "# Top comment" in leading_texts
    assert "# Another header" in leading_texts

    # Empty code
    assert extract_unit_comments_and_pragmas("", 1, 5) == []


def test_replace_unit_in_source_subline_and_token_slicing() -> None:
    """Verifies sub-line and token-range replacement preserving surrounding code and pragmas."""
    from pydoppelgangerhunt import replace_unit_in_source  # pylint: disable=import-outside-toplevel

    # 1. Statement sharing a line with prefix code and trailing pragma
    code1 = "x = 1; y = (a + b)  # type: ignore[operator]\n"
    unit_subline = {
        "file": "sub.py",
        "start": 1,
        "end": 1,
        "start_col": 7,
        "end_col": 18,
        "name": "y_assign",
    }
    replaced1 = replace_unit_in_source(code1, unit_subline, "y = helper(a, b)")
    assert replaced1 == "x = 1; y = helper(a, b)  # type: ignore[operator]\n"

    # 2. Multi-line unit with boundary pragma preservation
    code2 = (
        "def run_job():\n"
        "    val = compute_step()\n"
        "    return val  # type: ignore[no-any-return]\n"
    )
    unit_whole = {"file": "job.py", "start": 2, "end": 3, "name": "run_job_body"}
    replaced2 = replace_unit_in_source(code2, unit_whole, "    return _shared_compute_step()")
    assert "return _shared_compute_step()  # type: ignore[no-any-return]\n" in replaced2

    # 3. Whole-line with preserve_boundary_pragmas=False
    replaced3 = replace_unit_in_source(
        code2, unit_whole, "    return _shared_compute_step()", preserve_boundary_pragmas=False
    )
    assert "return _shared_compute_step()\n" in replaced3
    assert "# type: ignore" not in replaced3


def test_replace_unit_in_source_harvested_block_keeps_indentation(tmp_path: Path) -> None:
    """Verifies harvested indented multiline blocks are replaced as whole lines."""
    from pydoppelgangerhunt import replace_unit_in_source  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    code = (
        "def run(value: int) -> int:\n"
        "    if value > 0:\n"
        "        result = value + 1\n"
        "        return result\n"
        "    fallback = 0\n"
        "    return fallback\n"
    )
    src = tmp_path / "sample.py"
    src.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(src), str(tmp_path), min_lines=2, min_tokens=3)
    block_unit = next(unit for unit in units if unit.get("kind") == "compound_block")

    replaced = replace_unit_in_source(code, block_unit, "    helper_result = shared(value)\n    return helper_result\n")

    assert replaced == (
        "def run(value: int) -> int:\n"
        "    helper_result = shared(value)\n"
        "    return helper_result\n"
        "    fallback = 0\n"
        "    return fallback\n"
    )


def test_synthesize_helper_with_pragma_and_import_preservation(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code and local import capture preserve pragmas."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
        synthesize_shared_helper_code,
    )

    f1 = tmp_path / "mod_a.py"
    f2 = tmp_path / "mod_b.py"

    code_a = (
        "def process_a(val):\n"
        "    from typing_extensions import Buffer  # type: ignore\n"
        "    # Step 1: calculate\n"
        "    result = val * 10  # type: ignore[operator]\n"
        "    return result\n"
    )
    code_b = (
        "def process_b(val):\n"
        "    # Step 1: calculate\n"
        "    result = val * 10  # type: ignore[operator]\n"
        "    return result\n"
    )
    f1.write_text(code_a, encoding="utf-8")
    f2.write_text(code_b, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 5, "name": "process_a", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 5, "name": "process_b", "kind": "function"}

    # Scope analysis captures local import with # type: ignore intact
    scope = analyze_unit_variable_scope(u1)
    assert any("# type: ignore" in imp for imp in scope.get("local_imports", []))

    # Shared helper synthesis preserves comments and # type: ignore pragma
    helper = synthesize_shared_helper_code(u1, u2, preserve_pragmas=True)
    assert "# type: ignore[operator]" in helper


def test_parser_harvests_token_columns(tmp_path: Path) -> None:
    """Verifies that parser._record_unit stores start_col and end_col on harvested units."""
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    code = (
        "def add(x: int, y: int) -> int:\n"
        "    # Some logic\n"
        "    total = x + y\n"
        "    return total\n"
    )
    f = tmp_path / "sample.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=3)
    assert units
    for u in units:
        assert "start_col" in u
        assert "end_col" in u
        assert isinstance(u["start_col"], int)


def test_scope_binding_instance_and_class_detection(tmp_path: Path) -> None:
    """Verifies that scope analysis identifies instance and class method bindings."""
    code = (
        "class Service:\n"
        "    def run_instance(self, delta: int) -> int:\n"
        "        self.total = self.base_value + delta\n"
        "        return self.total\n"
        "\n"
        "    @classmethod\n"
        "    def run_class(cls, factor: int) -> int:\n"
        "        cls.count += factor\n"
        "        return cls.count\n"
        "\n"
        "    def run_plain(x: int) -> int:\n"
        "        return x * 2\n"
    )
    src = tmp_path / "service.py"
    src.write_text(code, encoding="utf-8")

    u_inst = {"file": str(src), "start": 2, "end": 4, "name": "run_instance", "kind": "function"}
    scope_inst = analyze_unit_variable_scope(u_inst)
    assert scope_inst["has_instance_binding"] is True
    assert scope_inst["has_class_binding"] is False
    assert scope_inst["binding_kind"] == "instance"
    assert scope_inst["inputs"][0] == "self"
    assert "self.base_value" in scope_inst["instance_attrs"] or "self.total" in scope_inst["instance_attrs"]

    u_cls = {"file": str(src), "start": 6, "end": 9, "name": "run_class", "kind": "function"}
    scope_cls = analyze_unit_variable_scope(u_cls)
    assert scope_cls["has_class_binding"] is True
    assert scope_cls["has_instance_binding"] is False
    assert scope_cls["binding_kind"] == "class"
    assert scope_cls["inputs"][0] == "cls"
    assert "cls.count" in scope_cls["class_attrs"]

    u_plain = {"file": str(src), "start": 11, "end": 12, "name": "run_plain", "kind": "function"}
    scope_plain = analyze_unit_variable_scope(u_plain)
    assert scope_plain["has_instance_binding"] is False
    assert scope_plain["has_class_binding"] is False
    assert scope_plain["binding_kind"] is None


def test_find_enclosing_class_and_function(tmp_path: Path) -> None:
    """Verifies locating enclosing class definitions and functions including nested scopes."""
    code = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def compute(self, x: int) -> int:\n"
        "            total = x + 1\n"
        "            return total\n"
        "\n"
        "def standalone(a: int) -> int:\n"
        "    return a * 2\n"
    )
    src = tmp_path / "nested.py"
    src.write_text(code, encoding="utf-8")

    # Inside Inner method
    u_inner = {"file": str(src), "start": 3, "end": 5, "name": "compute", "kind": "function"}
    enc_class = find_enclosing_class(code, u_inner)
    assert enc_class is not None
    assert enc_class["name"] == "Inner"
    assert enc_class["start"] == 2
    assert enc_class["indent"] == "    "
    assert enc_class["method_indent"] == "        "

    enc_fn = find_enclosing_function(code, u_inner)
    assert enc_fn is not None
    assert enc_fn["name"] == "compute"
    assert enc_fn["start"] == 3

    # Compound block inside compute
    u_block = {"file": str(src), "start": 4, "end": 4, "name": "compute:Assign", "kind": "compound_block"}
    enc_block_cls = find_enclosing_class(code, u_block)
    assert enc_block_cls is not None
    assert enc_block_cls["name"] == "Inner"
    enc_block_fn = find_enclosing_function(code, u_block)
    assert enc_block_fn is not None
    assert enc_block_fn["name"] == "compute"

    # Top-level standalone function has no enclosing class
    u_standalone = {"file": str(src), "start": 7, "end": 8, "name": "standalone", "kind": "function"}
    assert find_enclosing_class(code, u_standalone) is None
    standalone_fn = find_enclosing_function(code, u_standalone)
    assert standalone_fn is not None
    assert standalone_fn["name"] == "standalone"

    # Edge cases: empty text, syntax error, zero start line
    assert find_enclosing_class("", u_inner) is None
    assert find_enclosing_class("def broken(: pass", u_inner) is None
    assert find_enclosing_class(code, {"start": 0, "end": 0}) is None
    assert find_enclosing_function("", u_inner) is None
    assert find_enclosing_function("def broken(: pass", u_inner) is None
    assert find_enclosing_function(code, {"start": 0, "end": 0}) is None


def test_synthesize_shared_helper_code_same_class_and_class_binding(tmp_path: Path) -> None:
    """Verifies synthesizing private instance and class methods with appropriate indentation."""
    f = tmp_path / "calc.py"
    code = (
        "class Calculator:\n"
        "    def add_tax_a(self, amount: float) -> float:\n"
        "        rate = 0.05\n"
        "        return amount * (1.0 + rate)\n"
        "\n"
        "    def add_tax_b(self, amount: float) -> float:\n"
        "        rate = 0.05\n"
        "        return amount * (1.0 + rate)\n"
        "\n"
        "    @classmethod\n"
        "    def create_a(cls, val: int) -> int:\n"
        "        cls.total += val\n"
        "        return cls.total\n"
        "\n"
        "    @classmethod\n"
        "    def create_b(cls, val: int) -> int:\n"
        "        cls.total += val\n"
        "        return cls.total\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "add_tax_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "add_tax_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }

    # Auto mode detects same class in same file -> method mode
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto")
    assert "    def _shared_add_tax_a_add_tax_b(self, amount: float) -> float:" in helper
    assert "self: Any" not in helper  # self must not have : Any annotation
    assert "Call site:\n            self._shared_add_tax_a_add_tax_b(...)" in helper

    # Class method binding
    u_c1 = {
        "file": str(f),
        "start": 10,
        "end": 13,
        "name": "create_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u_c2 = {
        "file": str(f),
        "start": 15,
        "end": 18,
        "name": "create_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    cls_helper = synthesize_shared_helper_code(u_c1, u_c2, method_binding="method")
    assert "    @classmethod\n    def _shared_create_a_create_b(cls, val: int) -> int:" in cls_helper
    assert "Call site:\n            cls._shared_create_a_create_b(...)" in cls_helper

    # Explicit module mode falls back to module-level helper with self: Any
    mod_helper = synthesize_shared_helper_code(u1, u2, method_binding="module")
    assert "def _shared_add_tax_a_add_tax_b(self: Any, amount: float) -> float:" in mod_helper


def test_generate_refactoring_patch_same_class_whole_methods(tmp_path: Path) -> None:
    """Verifies that refactoring whole methods in the same class preserves headers and delegates."""
    f = tmp_path / "bank.py"
    code = (
        "class BankAccount:\n"
        "    def calc_interest_a(self, principal: float) -> float:\n"
        "        rate = 0.05\n"
        "        return principal * rate\n"
        "\n"
        "    def calc_interest_b(self, principal: float) -> float:\n"
        "        rate = 0.05\n"
        "        return principal * rate\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "calc_interest_a",
        "kind": "function",
        "enclosing_class": "BankAccount",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "calc_interest_b",
        "kind": "function",
        "enclosing_class": "BankAccount",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Private helper should be inserted inside BankAccount
    assert "+    def _shared_calc_interest_a_calc_interest_b(self, principal: float) -> float:" in patch
    # Signatures preserved and bodies delegate via self._shared_helper
    assert "+        return self._shared_calc_interest_a_calc_interest_b(principal)" in patch
    assert "-        rate = 0.05" in patch

    # Check replace_clones=False inserting helper inside class
    patch_preview = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=False)
    assert patch_preview
    assert "+    def _shared_calc_interest_a_calc_interest_b(self, principal: float) -> float:" in patch_preview
    assert "-        rate = 0.05" not in patch_preview


def test_generate_refactoring_patch_same_class_compound_blocks(tmp_path: Path) -> None:
    """Verifies refactoring compound blocks within class methods invokes self._shared_helper."""
    f = tmp_path / "processor.py"
    code = (
        "class DataProcessor:\n"
        "    def process_one(self, items: list) -> int:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            total += x\n"
        "        return total\n"
        "\n"
        "    def process_two(self, items: list) -> int:\n"
        "        total = 0\n"
        "        for x in items:\n"
        "            total += x\n"
        "        return total * 2\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "process_one:For",
        "kind": "compound_block",
        "enclosing_class": "DataProcessor",
    }
    u2 = {
        "file": str(f),
        "start": 10,
        "end": 11,
        "name": "process_two:For",
        "kind": "compound_block",
        "enclosing_class": "DataProcessor",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Private method in class body
    assert "+    def _shared_process_one_process_two(" in patch
    # Invocation via self._shared_helper omitting self from args
    assert "self._shared_process_one_process_two(" in patch


def test_generate_refactoring_patch_cross_class_receiver_injection(tmp_path: Path) -> None:
    """Verifies cross-class clones extract to module-level helper with explicit receiver injection."""
    f = tmp_path / "workers.py"
    code = (
        "class WorkerA:\n"
        "    def work(self, speed: int) -> int:\n"
        "        return speed * 10\n"
        "\n"
        "class WorkerB:\n"
        "    def work(self, speed: int) -> int:\n"
        "        return speed * 10\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "work",
        "kind": "function",
        "enclosing_class": "WorkerA",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "work",
        "kind": "function",
        "enclosing_class": "WorkerB",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    # Module-level helper with self: Any
    assert "+def _shared_work(self: Any, speed: int) -> int:" in patch
    # Call site delegates with self passed explicitly
    assert "+        return _shared_work(self, speed)" in patch


def test_parser_harvests_enclosing_class(tmp_path: Path) -> None:
    """Verifies that parser sets enclosing_class across methods, blocks, and sliding windows."""
    from pydoppelgangerhunt.parser import harvest_file_units  # pylint: disable=import-outside-toplevel

    code = (
        "class OrderService:\n"
        "    def process_order(self, order_id: int) -> bool:\n"
        "        if order_id > 0:\n"
        "            status = True\n"
        "            return status\n"
        "        return False\n"
        "\n"
        "def top_level(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y\n"
    )
    f = tmp_path / "order.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=3)
    method_unit = next(u for u in units if u["name"] == "process_order")
    assert method_unit.get("enclosing_class") == "OrderService"

    top_unit = next(u for u in units if u["name"] == "top_level")
    assert top_unit.get("enclosing_class") is None


def test_same_named_whole_method_helper_synthesis(tmp_path: Path) -> None:
    """Verifies that whole-method clones with identical names do not nest def inside helper."""
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    c1 = "class ServiceA:\n    def compute(self, n: int) -> int:\n        \"\"\"Docs.\"\"\"\n        ans = n * 10\n        return ans\n"
    c2 = "class ServiceB:\n    def compute(self, n: int) -> int:\n        \"\"\"Docs.\"\"\"\n        ans = n * 10\n        return ans\n"
    f1.write_text(c1, encoding="utf-8")
    f2.write_text(c2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "compute", "kind": "function", "enclosing_class": "ServiceA"}
    u2 = {"file": str(f2), "start": 2, "end": 5, "name": "compute", "kind": "function", "enclosing_class": "ServiceB"}
    code = synthesize_shared_helper_code(u1, u2)
    assert "def compute(" not in code
    assert "ans = n * 10" in code
    assert "return ans" in code


def test_module_helper_placed_after_docstring_and_future_imports(tmp_path: Path) -> None:
    """Verifies that cross-class module helpers are inserted after docstrings and future imports."""
    f = tmp_path / "order_proc.py"
    content = (
        '"""Module docstring."""\n'
        'from __future__ import annotations\n'
        '\n'
        'class Handler1:\n'
        '    def handle(self, num: int) -> int:\n'
        '        return num + 42\n'
        '\n'
        'class Handler2:\n'
        '    def handle(self, num: int) -> int:\n'
        '        return num + 42\n'
    )
    f.write_text(content, encoding="utf-8")
    u1 = {"file": str(f), "start": 5, "end": 6, "name": "handle", "kind": "function", "enclosing_class": "Handler1"}
    u2 = {"file": str(f), "start": 9, "end": 10, "name": "handle", "kind": "function", "enclosing_class": "Handler2"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert '"""Module docstring."""' in patch
    assert 'from __future__ import annotations' in patch
    assert "+def _shared_handle(self: Any, num: int) -> int:" in patch


def test_mixed_binding_preserves_cls_argument(tmp_path: Path) -> None:
    """Verifies that mixed instance and class binding retains cls in helper call arguments."""
    f = tmp_path / "factory.py"
    code = (
        "class WidgetFactory:\n"
        "    def build_item(self, cls: type, item_id: int) -> object:\n"
        "        self.count += 1\n"
        "        return cls(self.prefix, item_id)\n"
    )
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 4, "name": "build_item", "kind": "function", "enclosing_class": "WidgetFactory"}
    scope = analyze_unit_variable_scope(u)
    assert scope.get("binding_kind") == "mixed"
    assert scope.get("has_instance_binding") is True
    assert scope.get("has_class_binding") is True


def test_static_methods_avoid_self_injection(tmp_path: Path) -> None:
    """Verifies static method clones don't inject self and delegate cleanly."""
    f = tmp_path / "math_util.py"
    code = (
        "class MathUtils:\n"
        "    @staticmethod\n"
        "    def add_sq1(x: int, y: int) -> int:\n"
        "        return (x + y) ** 2\n"
        "\n"
        "    @staticmethod\n"
        "    def add_sq2(x: int, y: int) -> int:\n"
        "        return (x + y) ** 2\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 3, "end": 4, "name": "add_sq1", "kind": "function", "enclosing_class": "MathUtils"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "add_sq2", "kind": "function", "enclosing_class": "MathUtils"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "+def _shared_add_sq1_add_sq2(x: int, y: int) -> int:" in patch
    assert "+        return _shared_add_sq1_add_sq2(x, y)" in patch
    assert "self." not in patch


def test_parameter_kinds_forwarding_in_calls(tmp_path: Path) -> None:
    """Verifies that keyword-only, vararg, and kwarg parameters are forwarded with correct syntax."""
    f = tmp_path / "dispatcher.py"
    code = (
        "class EventDispatcher:\n"
        "    def dispatch_a(self, event: str, *args: object, retries: int = 3, **kwargs: object) -> bool:\n"
        "        self.sent.append(event)\n"
        "        return len(self.sent) > 0\n"
        "\n"
        "    def dispatch_b(self, event: str, *args: object, retries: int = 3, **kwargs: object) -> bool:\n"
        "        self.sent.append(event)\n"
        "        return len(self.sent) > 0\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 4, "name": "dispatch_a", "kind": "function", "enclosing_class": "EventDispatcher"}
    u2 = {"file": str(f), "start": 6, "end": 8, "name": "dispatch_b", "kind": "function", "enclosing_class": "EventDispatcher"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "retries=retries" in patch
    assert "*args" in patch
    assert "**kwargs" in patch


def test_helper_inserted_before_decorators(tmp_path: Path) -> None:
    """Verifies helper is inserted before decorators on earliest method."""
    f = tmp_path / "repo.py"
    code = (
        "class Repository:\n"
        "    @classmethod\n"
        "    def get_first(cls, name: str) -> str:\n"
        "        return f'item:{name}'\n"
        "\n"
        "    @classmethod\n"
        "    def get_second(cls, name: str) -> str:\n"
        "        return f'item:{name}'\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 3, "end": 4, "name": "get_first", "kind": "function", "enclosing_class": "Repository"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "get_second", "kind": "function", "enclosing_class": "Repository"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "_shared_get_first_get_second(cls, name: str) -> str:" in patch
    assert "return cls._shared_get_first_get_second(name)" in patch


def test_find_module_helper_insertion_index_shebang_and_encoding() -> None:
    """Verifies that module helper insertion respects shebang and encoding cookies."""
    lines_both = [
        "#!/usr/bin/env python3\n",
        "# -*- coding: utf-8 -*-\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_both) == 2

    lines_shebang = [
        "#!/usr/bin/env python3\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_shebang) == 1

    lines_encoding = [
        "# coding=utf-8\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_encoding) == 1

    lines_comment_then_encoding = [
        "# Ordinary comment\n",
        "# -*- coding: utf-8 -*-\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_comment_then_encoding) == 2

    lines_syntax_error = [
        "#!/usr/bin/env python3\n",
        "# coding=utf-8\n",
        "invalid ? ? ?\n",
    ]
    assert _find_module_helper_insertion_index(lines_syntax_error) == 2


def test_build_whole_method_delegation_sync_and_async_generators() -> None:
    """Verifies whole-method delegation syntax for sync and async generators."""
    src = (
        "class Streamer:\n"
        "    def gen(self, count: int):\n"
        "        \"\"\"Docstring.\"\"\"\n"
        "        for i in range(count):\n"
        "            yield i\n"
    )
    unit = {"name": "gen", "start": 2, "end": 5}
    sync_del = _build_whole_method_delegation(
        src,
        unit,
        call_prefix="self.",
        helper_name="_shared_gen",
        args_str="count",
        has_yield=True,
        is_async=False,
        has_return=False,
    )
    assert "yield from self._shared_gen(count)" in sync_del

    sync_del_ret = _build_whole_method_delegation(
        src,
        unit,
        call_prefix="self.",
        helper_name="_shared_gen",
        args_str="count",
        has_yield=True,
        is_async=False,
        has_return=True,
    )
    assert "return (yield from self._shared_gen(count))" in sync_del_ret

    async_src = (
        "class AsyncStreamer:\n"
        "    async def stream(self, count: int):\n"
        "        for i in range(count):\n"
        "            yield i\n"
    )
    async_unit = {"name": "stream", "start": 2, "end": 4}
    async_del = _build_whole_method_delegation(
        async_src,
        async_unit,
        call_prefix="self.",
        helper_name="_shared_stream",
        args_str="count",
        has_yield=True,
        is_async=True,
    )
    assert "async for _item in self._shared_stream(count):\n            yield _item" in async_del


def test_static_methods_propagation_in_synthesis_and_patch(tmp_path: Path) -> None:
    """Verifies is_static propagates into synthesize_shared_helper_code and method_binding='method'."""
    u1: Dict[str, Any] = {
        "file": "calc.py",
        "start": 3,
        "end": 4,
        "name": "calc_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u2: Dict[str, Any] = {
        "file": "calc.py",
        "start": 7,
        "end": 8,
        "name": "calc_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    code = synthesize_shared_helper_code(
        u1,
        u2,
        method_binding="method",
        is_static=True,
    )
    assert "@staticmethod" in code
    assert "(self" not in code
    assert "(cls" not in code
    assert "__class__._shared_calc_a_calc_b" in code

    f = tmp_path / "calc.py"
    calc_code = (
        "class Calculator:\n"
        "    @staticmethod\n"
        "    def add_a(x: int, y: int) -> int:\n"
        "        return x + y\n"
        "\n"
        "    @staticmethod\n"
        "    def add_b(x: int, y: int) -> int:\n"
        "        return x + y\n"
    )
    f.write_text(calc_code, encoding="utf-8")
    u1["file"] = str(f)
    u2["file"] = str(f)
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@staticmethod" in patch
    assert "__class__._shared" in patch


def test_differing_receiver_kinds_in_same_class(tmp_path: Path) -> None:
    """Verifies that clones with different receiver kinds fall back to module helper."""
    f = tmp_path / "mixed_receivers.py"
    code = (
        "class Handler:\n"
        "    def inst_worker(self, key: str) -> str:\n"
        "        res = f'key:{key}'\n"
        "        return res\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, key: str) -> str:\n"
        "        res = f'key:{key}'\n"
        "        return res\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "inst_worker",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    u2 = {
        "file": str(f),
        "start": 7,
        "end": 9,
        "name": "cls_worker",
        "kind": "function",
        "enclosing_class": "Handler",
    }

    # In auto mode, differing receivers must fall back to module helper
    patch_auto = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch_auto
    assert "+def _shared_inst_worker_cls_worker(key: str) -> str:" in patch_auto
    assert "self." not in patch_auto.split("def cls_worker")[1]

    # Explicit method binding must also fall back to module helper because differing receivers cannot share one descriptor
    patch_method = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch_method
    assert "+def _shared_inst_worker_cls_worker(key: str) -> str:" in patch_method
    assert "return _shared_inst_worker_cls_worker(key)" in patch_method


def test_same_file_cross_class_method_binding_fallback(tmp_path: Path) -> None:
    """Verifies that clones across different classes fall back to module helper even with method_binding='method'."""
    f = tmp_path / "cross_class.py"
    code = (
        "class Alpha:\n"
        "    def work(self, v: int) -> int:\n"
        "        return v * 10\n"
        "\n"
        "class Beta:\n"
        "    def work(self, v: int) -> int:\n"
        "        return v * 10\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "work", "kind": "function", "enclosing_class": "Alpha"}
    u2 = {"file": str(f), "start": 6, "end": 7, "name": "work", "kind": "function", "enclosing_class": "Beta"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "+def _shared_work(" in patch
    # Helper must be at module level, not inside Beta
    assert "class Beta" in patch


def test_classmethod_compound_block_injects_cls(tmp_path: Path) -> None:
    """Verifies that compound blocks inside @classmethod that don't reference cls still inject cls and @classmethod."""
    f = tmp_path / "factory.py"
    code = (
        "class Factory:\n"
        "    @classmethod\n"
        "    def make_a(cls, x: int, y: int) -> int:\n"
        "        val = (x + y) * 2\n"
        "        return val\n"
        "\n"
        "    @classmethod\n"
        "    def make_b(cls, x: int, y: int) -> int:\n"
        "        val = (x + y) * 2\n"
        "        return val\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "make_a:block",
        "kind": "compound_block",
        "enclosing_class": "Factory",
    }
    u2 = {
        "file": str(f),
        "start": 9,
        "end": 10,
        "name": "make_b:block",
        "kind": "compound_block",
        "enclosing_class": "Factory",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@classmethod" in patch
    assert "_shared_make_a_make_b(cls, " in patch
    assert "cls._shared_make_a_make_b" in patch
    assert "self." not in patch


def test_class_body_comprehension_clones_use_module_binding(tmp_path: Path) -> None:
    """Verifies that clones directly in class body fall back to module helper and avoid self._shared in class body."""
    f = tmp_path / "table.py"
    code = (
        "class ConfigTable:\n"
        "    A = [k.upper() for k in ('x', 'y')]\n"
        "    B = [k.upper() for k in ('x', 'y')]\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 2,
        "name": "ConfigTable:listcomp_L2",
        "kind": "comprehension",
        "enclosing_class": "ConfigTable",
    }
    u2 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "ConfigTable:listcomp_L3",
        "kind": "comprehension",
        "enclosing_class": "ConfigTable",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "self." not in patch
    assert "+def _shared" in patch


def test_class_level_comprehensions_harvest_enclosing_class(tmp_path: Path) -> None:
    """Verifies that comprehensions defined directly in class body have enclosing_class set."""
    f = tmp_path / "settings.py"
    code = (
        "class Settings:\n"
        "    KEYS = [k.upper() for k in ('a', 'b', 'c')]\n"
        "    MAP = {k: k * 2 for k in range(5)}\n"
    )
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), comprehensions=True)
    comps = [u for u in units if u.get("kind") == "comprehension"]
    assert len(comps) >= 2
    for comp in comps:
        assert comp.get("enclosing_class") == "Settings"
        assert comp.get("name", "").startswith("Settings:")


def test_class_suite_tab_and_two_space_indentation(tmp_path: Path) -> None:
    """Verifies that suite indentation is dynamically derived for tab and 2-space classes."""
    tab_code = (
        "class TabbedClass:\n"
        "\tdef m1(self, x: int) -> int:\n"
        "\t\treturn x + 1\n"
        "\n"
        "\tdef m2(self, x: int) -> int:\n"
        "\t\treturn x + 1\n"
    )
    f_tab = tmp_path / "tabbed.py"
    f_tab.write_text(tab_code, encoding="utf-8")
    u_tab = {"file": str(f_tab), "start": 2, "end": 3, "name": "m1", "kind": "function"}
    meta_tab = find_enclosing_class(tab_code, u_tab)
    assert meta_tab is not None
    assert meta_tab["method_indent"] == "\t"

    u_tab2 = {"file": str(f_tab), "start": 5, "end": 6, "name": "m2", "kind": "function"}
    patch_tab = generate_refactoring_patch(
        [(0.95, u_tab, u_tab2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch_tab
    assert "+\tdef _shared_m1_m2(self" in patch_tab
    assert "+\t\treturn " in patch_tab

    two_space_code = (
        "class TwoSpaceClass:\n"
        "  def m1(self, x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "  def m2(self, x: int) -> int:\n"
        "    return x + 1\n"
    )
    f_two = tmp_path / "two.py"
    f_two.write_text(two_space_code, encoding="utf-8")
    u_two = {"file": str(f_two), "start": 2, "end": 3, "name": "m1", "kind": "function"}
    meta_two = find_enclosing_class(two_space_code, u_two)
    assert meta_two is not None
    assert meta_two["method_indent"] == "  "


def test_receiver_bound_mixed_kind_declines_replacement(tmp_path: Path) -> None:
    """Verifies that differing receiver kinds with referenced receivers decline replacement."""
    code = (
        "class Handler:\n"
        "    def inst_worker(self, x: int) -> int:\n"
        "        return self.value + x\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, x: int) -> int:\n"
        "        return cls.value + x\n"
    )
    f = tmp_path / "mixed_referenced.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "inst_worker",
        "kind": "function",
        "enclosing_class": "Handler",
        "receiver_kind": "instance",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "cls_worker",
        "kind": "function",
        "enclosing_class": "Handler",
        "receiver_kind": "class",
    }

    # Synthesis must return empty string
    helper = synthesize_shared_helper_code(u1, u2)
    assert helper == ""

    # Patch generation must skip / return empty patch
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch == ""


def test_class_body_in_factory_function_uses_module_binding(tmp_path: Path) -> None:
    """Verifies that class body units in a local class defined inside a factory function use module binding."""
    code = (
        "def make_class():\n"
        "    class LocalClass:\n"
        "        vals1 = [x * 2 for x in range(10)]\n"
        "        vals2 = [x * 2 for x in range(10)]\n"
        "    return LocalClass\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "vals1",
        "kind": "comprehension",
        "enclosing_class": "LocalClass",
    }
    u2 = {
        "file": str(f),
        "start": 4,
        "end": 4,
        "name": "vals2",
        "kind": "comprehension",
        "enclosing_class": "LocalClass",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch
    # Must NOT emit self. inside class creation body
    assert "self._shared" not in patch
    # Must synthesize module helper
    assert "+def _shared" in patch


def test_nested_class_static_method_uses_dunder_class(tmp_path: Path) -> None:
    """Verifies that nested static methods in Outer.Inner delegate via __class__._shared."""
    code = (
        "class Outer:\n"
        "    class Inner:\n"
        "        @staticmethod\n"
        "        def add1(a: int, b: int) -> int:\n"
        "            return a + b\n"
        "\n"
        "        @staticmethod\n"
        "        def add2(a: int, b: int) -> int:\n"
        "            return a + b\n"
    )
    f = tmp_path / "nested_static.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "add1",
        "kind": "function",
        "enclosing_class": "Inner",
        "receiver_kind": "static",
        "is_static": True,
    }
    u2 = {
        "file": str(f),
        "start": 8,
        "end": 9,
        "name": "add2",
        "kind": "function",
        "enclosing_class": "Inner",
        "receiver_kind": "static",
        "is_static": True,
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@staticmethod" in patch
    assert "__class__._shared" in patch
    assert "Inner._shared" not in patch


def test_comprehension_inside_method_uses_module_binding(tmp_path: Path) -> None:
    """Verifies that duplicate comprehensions inside methods use module binding without self._shared."""
    code = (
        "class Worker:\n"
        "    def run_a(self, data: list) -> list:\n"
        "        return [x * 2 for x in data]\n"
        "\n"
        "    def run_b(self, data: list) -> list:\n"
        "        return [x * 2 for x in data]\n"
    )
    f = tmp_path / "worker_comp.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "run_a:listcomp_L3",
        "kind": "comprehension",
        "enclosing_class": "Worker",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 6,
        "name": "run_b:listcomp_L6",
        "kind": "comprehension",
        "enclosing_class": "Worker",
    }
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch
    # Must synthesize module helper, not method helper inside Worker
    assert "+def _shared_run_a_run_b(" in patch
    # Must not invoke self._shared inside run_a or run_b
    assert "self._shared_run_a_run_b" not in patch
    assert "_shared_run_a_run_b(" in patch


def test_harvest_file_units_records_receiver_kind_and_is_static(tmp_path: Path) -> None:
    """Verifies that harvest_file_units records receiver_kind and is_static."""
    code = (
        "class Service:\n"
        "    def inst_m(self, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    @classmethod\n"
        "    def cls_m(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    @staticmethod\n"
        "    def stat_m(x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
    )
    f = tmp_path / "svc.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1)
    by_name = {u["name"]: u for u in units}

    assert by_name["inst_m"]["receiver_kind"] == "instance"
    assert by_name["inst_m"]["is_static"] is False

    assert by_name["cls_m"]["receiver_kind"] == "class"
    assert by_name["cls_m"]["is_static"] is False

    assert by_name["stat_m"]["receiver_kind"] == "static"
    assert by_name["stat_m"]["is_static"] is True


def test_synthesize_shared_helper_harvested_units_declines_mixed_receiver(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code on real harvested units declines receiver-bound mixed-kind pairs."""
    code = (
        "class Handler:\n"
        "    def inst_worker(self, x: int) -> int:\n"
        "        y = x\n"
        "        return self.value + y\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return cls.value + y\n"
    )
    f = tmp_path / "mixed_harvest.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1)
    by_name = {u["name"]: u for u in units}

    # Direct call to synthesize_shared_helper_code without pre-configured receiver_kind
    helper = synthesize_shared_helper_code(
        by_name["inst_worker"], by_name["cls_worker"], repo_root=str(tmp_path)
    )
    assert helper == ""


def test_closure_whole_function_refactoring_and_body_extraction(tmp_path: Path) -> None:
    """Verifies that whole closure units strip their header in synthesis and delegate properly in patches."""
    code = (
        "def factory_a():\n"
        "    def inner_a(x: int) -> int:\n"
        "        step_val = 1\n"
        "        return x + step_val\n"
        "    return inner_a\n"
        "\n"
        "def factory_b():\n"
        "    def inner_b(x: int) -> int:\n"
        "        step_val = 1\n"
        "        return x + step_val\n"
        "    return inner_b\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "factory_a:inner_a",
        "kind": "closure",
    }
    u2 = {
        "file": str(f),
        "start": 8,
        "end": 10,
        "name": "factory_b:inner_b",
        "kind": "closure",
    }
    helper = synthesize_shared_helper_code(u1, u2)
    assert helper
    # def inner_a must NOT be nested inside the helper body
    assert "def inner_a" not in helper
    assert "def inner_b" not in helper

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    # Closure definitions must be preserved and delegate to helper
    assert "def inner_a(x: int) -> int:" in patch
    assert "return _shared_inner_a_inner_b(x)" in patch


def test_two_space_async_generator_delegation(tmp_path: Path) -> None:
    """Verifies that 2-space indented files use 2-space step for async generator delegation."""
    code = (
        "async def gen1(src):\n"
        "  async for item in src:\n"
        "    yield item\n"
        "\n"
        "async def gen2(src):\n"
        "  async for item in src:\n"
        "    yield item\n"
    )
    f = tmp_path / "two_gen.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "gen1:asyncfor", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 6, "end": 7, "name": "gen2:asyncfor", "kind": "compound_block"}
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    assert "+  async for _item in _shared_gen1_gen2(src):\n+    yield _item\n" in patch


def test_generate_refactoring_patch_does_not_mutate_unit_is_static(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not mutate caller unit dicts with is_static."""
    code = (
        "class Handler:\n"
        "    @staticmethod\n"
        "    def s_work(x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    def i_work(self, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
    )
    f = tmp_path / "mutate_check.py"
    f.write_text(code, encoding="utf-8")
    u_inst = {"file": str(f), "start": 8, "end": 9, "name": "i_work", "kind": "function"}
    u_stat = {"file": str(f), "start": 3, "end": 4, "name": "s_work", "kind": "function"}

    _ = generate_refactoring_patch(
        [(0.95, u_inst, u_stat)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "is_static" not in u_inst


def test_closure_returns_not_classified_as_control_flow_hazards(tmp_path: Path) -> None:
    """Verifies that whole closure units with return statements are not flagged as embedded_return hazards."""
    code = (
        "def factory_calc(multiplier: int):\n"
        "    def step_fn(val: int) -> int:\n"
        "        offset = 10\n"
        "        return val * multiplier + offset\n"
        "    return step_fn\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    u = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "factory_calc:step_fn",
        "kind": "closure",
    }
    info = _inspect_unit_scope(u, repo_root=str(tmp_path))
    assert info["is_control_flow_safe"] is True
    assert "embedded_return" not in info["control_flow_hazards"]

    helper = synthesize_shared_helper_code(u, u, repo_root=str(tmp_path))
    assert "WARNING: Non-local control flow hazard" not in helper


def test_synthesize_shared_helper_independent_receiver_kinds(tmp_path: Path) -> None:
    """Verifies that units with differing is_static flags are not cross-contaminated into identical receiver kinds."""
    code_stat = (
        "def stat_fn(x: int) -> int:\n"
        "    return x + 1\n"
    )
    code_inst = (
        "class Worker:\n"
        "    def inst_fn(self, x: int) -> int:\n"
        "        return self.value + x\n"
    )
    f1 = tmp_path / "mod_stat.py"
    f2 = tmp_path / "mod_inst.py"
    f1.write_text(code_stat, encoding="utf-8")
    f2.write_text(code_inst, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 2, "name": "stat_fn", "kind": "function", "is_static": True}
    u2 = {"file": str(f2), "start": 2, "end": 3, "name": "inst_fn", "kind": "function", "is_static": False}

    # Should detect differing receiver kinds and decline helper replacement due to self.value reference
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""


def test_generate_refactoring_patch_cross_file_receiver_difference(tmp_path: Path) -> None:
    """Verifies that cross-file pairs with differing receiver kinds decline replacement if receiver is referenced."""
    code1 = (
        "class ServiceA:\n"
        "    def run(self, x: int) -> int:\n"
        "        y = x\n"
        "        return self.factor + y\n"
    )
    code2 = (
        "class ServiceB:\n"
        "    @classmethod\n"
        "    def run(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return cls.factor + y\n"
    )
    f1 = tmp_path / "srv_a.py"
    f2 = tmp_path / "srv_b.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "run", "kind": "function", "enclosing_class": "ServiceA", "receiver_kind": "instance"}
    u2 = {"file": str(f2), "start": 3, "end": 5, "name": "run", "kind": "function", "enclosing_class": "ServiceB", "receiver_kind": "class"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_generate_refactoring_patch_slash_normalization(tmp_path: Path) -> None:
    """Verifies that differing path slash formats normalize to the same file during patch generation."""
    code = (
        "class Normalizer:\n"
        "    def worker_one(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
        "\n"
        "    def worker_two(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
    )
    sub = tmp_path / "pkg"
    sub.mkdir()
    f = sub / "norm.py"
    f.write_text(code, encoding="utf-8")

    # One unit with POSIX slash, one with Windows backslash
    u1 = {"file": "pkg/norm.py", "start": 2, "end": 4, "name": "worker_one", "kind": "function", "enclosing_class": "Normalizer"}
    u2 = {"file": "pkg\\norm.py", "start": 6, "end": 8, "name": "worker_two", "kind": "function", "enclosing_class": "Normalizer"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    # Both units must be replaced in the single file and delegate to the helper
    assert patch.count("+        return self._shared_worker_one_worker_two(x)") == 2


def test_parentheses_on_staticmethod_and_classmethod_decorators(tmp_path: Path) -> None:
    """Verifies that @staticmethod() and @classmethod() call decorators are recognized properly."""
    code = (
        "class CallDec:\n"
        "    @staticmethod()\n"
        "    def s_fn(x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    @classmethod()\n"
        "    def c_fn(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
    )
    f = tmp_path / "calldec.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1)
    by_name = {u["name"]: u for u in units}

    assert by_name["s_fn"]["is_static"] is True
    assert by_name["s_fn"]["receiver_kind"] == "static"

    assert by_name["c_fn"]["is_static"] is False
    assert by_name["c_fn"]["receiver_kind"] == "class"

    enc_s = find_enclosing_function(code, by_name["s_fn"])
    assert enc_s is not None
    assert enc_s["is_static"] is True

    enc_c = find_enclosing_function(code, by_name["c_fn"])
    assert enc_c is not None
    assert enc_c["is_class_method"] is True


def test_nested_closure_control_flow_and_returns_do_not_leak_to_outer(tmp_path: Path) -> None:
    """Verifies that returns, yields, and awaits in nested closures do not leak into outer function scope."""
    code = (
        "def outer_factory(base: int):\n"
        "    multiplier = 2\n"
        "    async def inner_worker(val: int) -> int:\n"
        "        await do_async_op()\n"
        "        if val < 0:\n"
        "            return 0\n"
        "        return val * multiplier\n"
        "    return inner_worker\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1, harvest_closures=True)
    outer_u = next(u for u in units if u["name"] == "outer_factory")
    scope = analyze_unit_variable_scope(outer_u, repo_root=str(tmp_path))

    # Outer factory must be synchronous, non-generator, and only return inner_worker
    assert scope["is_async"] is False
    assert scope["has_yield"] is False
    assert scope["outputs"] == ["inner_worker"]


def test_local_class_methods_not_scoped_as_closures_of_factory_function(tmp_path: Path) -> None:
    """Verifies that methods of classes defined in functions are scoped to the class rather than as closures."""
    code = (
        "def make_handler():\n"
        "    class LocalHandler:\n"
        "        def handle(self, item: str) -> str:\n"
        "            def nested_sub():\n"
        "                item_clean = item.strip()\n"
        "                temp = item_clean.lower()\n"
        "                return temp\n"
        "            return nested_sub()\n"
        "    return LocalHandler\n"
    )
    f = tmp_path / "local_cls.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1, harvest_closures=True)
    by_name = {u["name"]: u for u in units}

    # handle is a method of LocalHandler, not a closure of make_handler
    assert "handle" in by_name
    handle_u = by_name["handle"]
    assert handle_u["kind"] == "function"
    assert handle_u["enclosing_class"] == "LocalHandler"
    assert handle_u["receiver_kind"] == "instance"

    # nested_sub is a closure of handle
    assert "handle:nested_sub" in by_name
    sub_u = by_name["handle:nested_sub"]
    assert sub_u["kind"] == "closure"
    assert sub_u["enclosing_class"] == "LocalHandler"


def test_instance_method_paired_with_module_function_declines_replacement_when_receiver_accessed(tmp_path: Path) -> None:
    """Verifies that pairing an instance method accessing self.val with a module function declines replacement."""
    code1 = (
        "class Service:\n"
        "    def compute(self, x: int) -> int:\n"
        "        val = self.offset\n"
        "        return val + x * 2\n"
    )
    code2 = (
        "def compute(x: int) -> int:\n"
        "    val = 10\n"
        "    return val + x * 2\n"
    )
    f1 = tmp_path / "srv.py"
    f1.write_text(code1, encoding="utf-8")
    f2 = tmp_path / "util.py"
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "srv.py", "start": 2, "end": 4, "name": "Service:compute",
        "kind": "function", "enclosing_class": "Service", "receiver_kind": "instance"
    }
    u2 = {
        "file": "util.py", "start": 1, "end": 3, "name": "compute",
        "kind": "function", "enclosing_class": None, "receiver_kind": None
    }

    # Synthesis must return "" because receiver kinds differ and u1 accesses self
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    # Patch generation must skip this clone pair
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_detect_indent_step_multi_level_two_spaces() -> None:
    """Verifies that _detect_indent_step correctly handles 2-space indentation at various nesting levels."""
    assert _detect_indent_step("  ") == "  "
    assert _detect_indent_step("    ") == "    "
    assert _detect_indent_step("      ") == "  "
    assert _detect_indent_step("        ") == "    "
    assert _detect_indent_step("          ") == "  "
    assert _detect_indent_step("\t\t") == "\t"


def test_insert_imports_raw_and_unicode_docstrings() -> None:
    """Verifies that raw (r\"\"\") and unicode (u\"\"\") docstrings are preserved ahead of inserted imports."""
    raw_lines = [
        'r"""Raw module docstring with \\s+ escapes."""\n',
        "x = 1\n",
    ]
    res_raw = _insert_imports_into_module(raw_lines, ["from typing import Any"])
    assert res_raw[0].startswith('r"""')
    assert "from typing import Any\n" in res_raw
    assert res_raw.index("from typing import Any\n") > 0

    uni_lines = [
        'u"""Unicode module docstring with unicode text."""\n',
        "x = 1\n",
    ]
    res_uni = _insert_imports_into_module(uni_lines, ["from typing import Any"])
    assert res_uni[0].startswith('u"""')
    assert res_uni.index("from typing import Any\n") > 0


def test_extract_required_typing_imports_comprehensive() -> None:
    """Verifies that standard library typing symbols such as Mapping and Literal are extracted."""
    sig = "(config: Mapping[str, Any], mode: Literal['fast', 'slow']) -> Optional[Tuple[int, ...]]"
    needed = _extract_required_typing_imports(sig)
    for expected in ["Any", "Literal", "Mapping", "Optional", "Tuple"]:
        assert expected in needed


def test_same_name_classes_in_different_factories_not_same_class(tmp_path: Path) -> None:
    """Verifies that two identically named local classes in different factory functions are not treated as same class."""
    code = (
        "def factory_one():\n"
        "    class Config:\n"
        "        def process(self, x: int) -> int:\n"
        "            y = x * 2\n"
        "            return y + 1\n"
        "    return Config\n"
        "\n"
        "def factory_two():\n"
        "    class Config:\n"
        "        def process(self, x: int) -> int:\n"
        "            y = x * 2\n"
        "            return y + 1\n"
        "    return Config\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": "factories.py", "start": 3, "end": 5, "name": "process", "kind": "function"}
    u2 = {"file": "factories.py", "start": 10, "end": 12, "name": "process", "kind": "function"}

    # In auto mode, because start lines differ (line 2 vs line 9), they cannot share a private class method
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto", repo_root=str(tmp_path))
    # Helper should fall back to module-level helper (no self parameter omission or @classmethod/method indent)
    assert helper.startswith("def _shared_process(self: Any, x: int) -> int:")


def test_lambda_parameter_scoping_no_leakage(tmp_path: Path) -> None:
    """Verifies that parameters of lambda expressions do not leak into outer inputs."""
    code = (
        "def transform_items(items: list[int]) -> list[int]:\n"
        "    return sorted(items, key=lambda x: x.val)\n"
    )
    f = tmp_path / "lambda_mod.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 2, "name": "transform_items"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "items" in scope["inputs"]
    assert "x" not in scope["inputs"]


def test_mixed_async_and_sync_pair_declined(tmp_path: Path) -> None:
    """Verifies that clone pairs with mixed async and sync execution models are declined."""
    code = (
        "async def async_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
        "\n"
        "def sync_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
    )
    f = tmp_path / "mixed_async.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 3, "name": "async_worker", "kind": "function"}
    u2 = {"file": str(f), "start": 5, "end": 7, "name": "sync_worker", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_mixed_generator_and_function_pair_declined(tmp_path: Path) -> None:
    """Verifies that clone pairs with mixed generator (yield) and normal function models are declined."""
    code = (
        "def gen_worker(x: int):\n"
        "    y = x * 2\n"
        "    yield y + 1\n"
        "\n"
        "def fn_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
    )
    f = tmp_path / "mixed_gen.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 3, "name": "gen_worker", "kind": "function"}
    u2 = {"file": str(f), "start": 5, "end": 7, "name": "fn_worker", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_instance_method_paired_with_module_function_omits_receiver_parameter(tmp_path: Path) -> None:
    """Verifies that an instance method paired with a module function without receiver references omits self."""
    code = (
        "class Service:\n"
        "    def foo(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
        "\n"
        "def bar(x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
    )
    f = tmp_path / "service_mod.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "foo",
        "kind": "function",
        "enclosing_class": "Service",
        "receiver_kind": "instance",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "bar",
        "kind": "function",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper.startswith("def _shared_foo_bar(x: int) -> int:")
    assert "self" not in helper.splitlines()[0]

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_foo_bar(x)" in patch
    assert "_shared_foo_bar(self" not in patch


def test_direct_method_verification_distinguishes_closures(tmp_path: Path) -> None:
    """Verifies that direct method verification distinguishes class methods from nested closures."""
    code = (
        "class Worker:\n"
        "    def outer_method(self, a: int) -> int:\n"
        "        def inner_closure(b: int) -> int:\n"
        "            y = b * 2\n"
        "            z = y + 1\n"
        "            w = z * 3\n"
        "            return w\n"
        "        return inner_closure(a)\n"
    )
    f = tmp_path / "worker.py"
    f.write_text(code, encoding="utf-8")

    # Harvest units to check parser hierarchy scoping
    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=2, min_tokens=5, harvest_closures=True)
    by_name = {u["name"]: u for u in units}
    assert by_name["outer_method"]["enclosing_class"] == "Worker"
    assert by_name["outer_method"]["receiver_kind"] == "instance"
    assert by_name["outer_method:inner_closure"]["enclosing_class"] == "Worker"
    assert by_name["outer_method:inner_closure"]["receiver_kind"] is None

    # Verify find_enclosing_class and _is_method_of_class
    cls_meta = find_enclosing_class(code, by_name["outer_method"])
    fn_outer = find_enclosing_function(code, by_name["outer_method"])
    fn_inner = find_enclosing_function(code, by_name["outer_method:inner_closure"])

    assert _is_method_of_class(fn_outer, cls_meta) is True
    assert _is_method_of_class(fn_inner, cls_meta) is False


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_pattern_bound_variables_not_treated_as_inputs(tmp_path: Path) -> None:
    """Verifies that pattern-bound variables in match statements are recognized as stores, not inputs."""
    code = (
        "def process_data(data: list) -> int:\n"
        "    match data:\n"
        "        case [head, *tail]:\n"
        "            total = head + len(tail)\n"
        "            return total\n"
        "        case _:\n"
        "            return 0\n"
    )
    f = tmp_path / "matcher_unit.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 7, "name": "process_data", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "data" in scope["inputs"]
    assert "head" not in scope["inputs"]
    assert "tail" not in scope["inputs"]
    assert "total" in scope["outputs"]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_definite_assignment_requires_irrefutable_default() -> None:
    """Verifies that match block assignments require an irrefutable pattern to be definite."""
    code_no_default = (
        "match x:\n"
        "    case 1:\n"
        "        val = 10\n"
        "    case 2:\n"
        "        val = 20\n"
    )
    tree1 = ast.parse(code_no_default)
    def1, cond1 = _analyze_block_assignment(tree1.body)
    assert "val" not in def1
    assert "val" in cond1

    code_with_default = (
        "match x:\n"
        "    case 1:\n"
        "        val = 10\n"
        "    case _:\n"
        "        val = 20\n"
    )
    tree2 = ast.parse(code_with_default)
    def2, cond2 = _analyze_block_assignment(tree2.body)
    assert "val" in def2
    assert "val" not in cond2


def test_inner_function_decorators_and_defaults_lexical_scoping(tmp_path: Path) -> None:
    """Verifies that variables in inner function decorators and defaults are captured into outer inputs."""
    code = (
        "def make_pipeline(outer_mult: int, threshold: int) -> object:\n"
        "    @transform_decorator\n"
        "    def step(x: int = threshold) -> int:\n"
        "        return x * outer_mult\n"
        "    return step\n"
    )
    f = tmp_path / "pipeline.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 5, "name": "make_pipeline", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "outer_mult" in scope["inputs"]
    assert "threshold" in scope["inputs"]
    assert "transform_decorator" in scope["inputs"]
    assert "x" not in scope["inputs"]


def test_separate_scopes_identical_class_name_enclosing_class_start(tmp_path: Path) -> None:
    """Verifies that identically named classes in different scopes have distinct enclosing_class_start."""
    code = (
        "def factory_one():\n"
        "    class Config:\n"
        "        def process(self, a: int, b: int) -> int:\n"
        "            x = a * 10\n"
        "            y = b * 20\n"
        "            z = x + y\n"
        "            return z\n"
        "    return Config\n\n"
        "def factory_two():\n"
        "    class Config:\n"
        "        def process(self, a: int, b: int) -> int:\n"
        "            x = a * 10\n"
        "            y = b * 20\n"
        "            z = x + y\n"
        "            return z\n"
        "    return Config\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=3, min_tokens=5)
    processes = [u for u in units if u["name"] == "process"]
    assert len(processes) == 2
    u1, u2 = processes[0], processes[1]
    assert u1["enclosing_class"] == "Config"
    assert u2["enclosing_class"] == "Config"
    assert u1["enclosing_class_start"] is not None
    assert u2["enclosing_class_start"] is not None
    assert u1["enclosing_class_start"] != u2["enclosing_class_start"]

    # In auto mode, separate classes must NOT synthesize a private instance method
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto", repo_root=str(tmp_path))
    assert not helper.startswith("    def _shared_process(self")
    assert "def _shared_process(self" in helper or "def _shared_process(" in helper


def test_closures_in_classes_synthesize_module_helper_auto_mode(tmp_path: Path) -> None:
    """Verifies that closures inside classes synthesize module-level helpers in auto mode."""
    code = (
        "class Service:\n"
        "    def run_a(self, items: list) -> list:\n"
        "        def transform(val: int) -> int:\n"
        "            x = val * 2\n"
        "            y = x + 3\n"
        "            return y\n"
        "        return [transform(i) for i in items]\n"
        "    def run_b(self, items: list) -> list:\n"
        "        def transform(val: int) -> int:\n"
        "            x = val * 2\n"
        "            y = x + 3\n"
        "            return y\n"
        "        return [transform(i) for i in items]\n"
    )
    f = tmp_path / "service.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=2, min_tokens=5, harvest_closures=True)
    closures = [u for u in units if u["kind"] == "closure"]
    assert len(closures) == 2
    c1, c2 = closures[0], closures[1]

    helper = synthesize_shared_helper_code(c1, c2, method_binding="auto", repo_root=str(tmp_path))
    # Helper must be module-level (no indentation, no self receiver)
    assert helper.startswith("def _shared_transform(")
    assert "self" not in helper


def test_aug_assign_compound_block_inputs_and_outputs(tmp_path: Path) -> None:
    """Verifies that AugAssign targets without prior assignment in the block are tracked as both inputs and outputs."""
    code = (
        "def compute_a(items: list) -> int:\n"
        "    total = 0\n"
        "    total += len(items)\n"
        "    return total\n\n"
        "def compute_b(items: list) -> int:\n"
        "    total = 0\n"
        "    total += len(items)\n"
        "    return total\n"
    )
    f = tmp_path / "calc.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": str(f), "start": 3, "end": 3, "name": "aug_a", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 8, "end": 8, "name": "aug_b", "kind": "compound_block"}

    scope = analyze_unit_variable_scope(u1, repo_root=str(tmp_path))
    assert "total" in scope["inputs"]
    assert "total" in scope["outputs"]
    assert "items" in scope["inputs"]

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "total: Any" in helper
    assert "return total" in helper

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "total = _shared_aug_a" in patch
    assert "total," in patch or "(total," in patch or "(items, total)" in patch or "(total)" in patch


def test_module_helper_insertion_index_stops_at_first_non_import() -> None:
    """Verifies that module helper insertion index terminates at the initial import block boundary."""
    code = (
        '"""Module docstring."""\n'
        "import sys\n"
        "import os\n\n"
        "class Config:\n"
        "    val = 42\n\n"
        "import math\n"
    )
    lines = code.splitlines(keepends=True)
    idx = _find_module_helper_insertion_index(lines)
    # Must be after import os (line 3), before class Config (line 5), not after import math (line 8)
    assert idx == 3


def test_comprehension_token_span_and_bare_call_replacement(tmp_path: Path) -> None:
    """Verifies that comprehension units return the evaluated expression and replace exact token spans."""
    code = (
        "def make_lists(items: list):\n"
        "    a_list = [k * 2 for k in items if k > 0]\n"
        "    b_list = [k * 2 for k in items if k > 0]\n"
        "    return a_list, b_list\n"
    )
    f = tmp_path / "comps.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=1, min_tokens=5, comprehensions=True)
    comps = [u for u in units if u["kind"] == "comprehension"]
    assert len(comps) == 2
    c1, c2 = comps[0], comps[1]

    scope = analyze_unit_variable_scope(c1, repo_root=str(tmp_path))
    assert "items" in scope["inputs"]
    assert "k" not in scope["inputs"]
    assert "k" not in scope["outputs"]

    helper = synthesize_shared_helper_code(c1, c2, repo_root=str(tmp_path))
    assert "return [k * 2 for k in items if k > 0]" in helper
    assert "return k" not in helper

    patch = generate_refactoring_patch([(1.0, c1, c2)], repo_root=str(tmp_path), replace_clones=True)
    assert "a_list = _shared_" in patch
    assert "b_list = _shared_" in patch


def test_receiver_isolation_in_factory_function(tmp_path: Path) -> None:
    """Verifies that nested class/method receiver accesses do not pollute the outer function's scope."""
    code = (
        "def build_service(mult: int):\n"
        "    class Inner:\n"
        "        def calc(self, v: int) -> int:\n"
        "            return self.mult * v\n"
        "    return Inner()\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")

    u = {"file": str(f), "start": 1, "end": 5, "name": "build_service", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert scope["has_instance_binding"] is False
    assert scope["has_class_binding"] is False
    assert "self" not in scope["inputs"]
    assert "mult" in scope["inputs"]


def test_backslash_relative_path_normalization_on_unit(tmp_path: Path) -> None:
    """Verifies that units carrying Windows-style backslashes resolve correctly across platforms."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    f = pkg_dir / "service.py"
    f.write_text("def run_a(x: int) -> int:\n    return x + 10\n\ndef run_b(x: int) -> int:\n    return x + 10\n", encoding="utf-8")

    u1 = {"file": "pkg\\service.py", "start": 1, "end": 2, "name": "run_a", "kind": "function"}
    u2 = {"file": "pkg/service.py", "start": 4, "end": 5, "name": "run_b", "kind": "function"}

    extracted = extract_unit_source_code(u1, repo_root=str(tmp_path))
    assert len(extracted) == 2
    assert "def run_a" in extracted[0]

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/pkg/service.py" in patch
    assert "+++ b/pkg/service.py" in patch


def test_cross_class_clone_call_site_receiver_arguments(tmp_path: Path) -> None:
    """Verifies that cross-class method clones preserve receiver arguments at both call sites."""
    code = (
        "class WorkerA:\n"
        "    def execute(self, payload: str) -> str:\n"
        "        return payload.strip().lower()\n"
        "\n"
        "class WorkerB:\n"
        "    def run_task(self, payload: str) -> str:\n"
        "        return payload.strip().lower()\n"
    )
    f = tmp_path / "workers.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": "workers.py", "start": 2, "end": 3, "name": "execute", "kind": "function"}
    u2 = {"file": "workers.py", "start": 6, "end": 7, "name": "run_task", "kind": "function"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_execute_run_task(self: Any, payload: str)" in patch
    assert "+        return _shared_execute_run_task(self, payload)" in patch
    # Verify WorkerB also passes self, not just WorkerA
    count_self_call = patch.count("_shared_execute_run_task(self, payload)")
    assert count_self_call == 2


def test_annassign_without_value_not_treated_as_definite_assignment(tmp_path: Path) -> None:
    """Verifies that type annotations without values do not prevent conditional output initialization."""
    from pydoppelgangerhunt.fixer import _analyze_block_assignment, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    tree = ast.parse("res: int\nif flag:\n    res = 42\n")
    definite, conditional = _analyze_block_assignment(tree.body)
    assert "res" not in definite
    assert "res" in conditional

    code = "res: int\nif flag:\n    res = 42\n"
    f = tmp_path / "ann.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": "ann.py", "start": 1, "end": 3, "name": "block", "kind": "compound_block"}
    helper = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    assert "res = None" in helper


def test_comprehension_multi_generator_scoping(tmp_path: Path) -> None:
    """Verifies that multi-generator comprehensions do not leak inner loop variables into inputs."""
    code = "matrix = [[1, 2], [3, 4]]\nflat = [x for row in matrix for x in row]\n"
    f = tmp_path / "comp_matrix.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "comp_matrix.py",
        "start": 2,
        "end": 2,
        "start_col": 7,
        "end_col": 42,
        "name": "flat_comp",
        "kind": "comprehension",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "matrix" in scope["inputs"]
    assert "row" not in scope["inputs"]
    assert "x" not in scope["inputs"]


def test_walrus_operator_comprehension_enclosing_scope(tmp_path: Path) -> None:
    """Verifies that walrus expressions in comprehensions register as stores in the enclosing scope."""
    code = "def process_data(items):\n    squared = [val := x * 2 for x in items]\n    return squared, val\n"
    f = tmp_path / "walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {"file": "walrus.py", "start": 1, "end": 3, "name": "process_data", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "val" in scope["locals"]
    assert "val" in scope["outputs"]
    assert "items" in scope["inputs"]
    assert "val" not in scope["inputs"]


def test_async_comprehension_detected_as_async(tmp_path: Path) -> None:
    """Verifies that an async comprehension correctly tags its unit scope as async."""
    code = "async def fetch():\n    results = [x async for x in aiter]\n    return results\n"
    f = tmp_path / "async_comp.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "async_comp.py",
        "start": 2,
        "end": 2,
        "start_col": 14,
        "end_col": 38,
        "name": "async_listcomp",
        "kind": "comprehension",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert scope["is_async"] is True


def test_except_handler_as_err_variable_not_treated_as_input_or_output(tmp_path: Path) -> None:
    """Verifies that 'except Exception as err:' does not leak 'err' into inputs or subroutine outputs."""
    code = (
        "def run_task():\n"
        "    try:\n"
        "        action()\n"
        "    except Exception as err:\n"
        "        logger.error(err)\n"
    )
    f = tmp_path / "except_scope.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "except_scope.py",
        "start": 2,
        "end": 5,
        "name": "run_task:try_block",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "err" not in scope["inputs"]
    assert "err" not in scope["outputs"]
    assert "logger" in scope["inputs"]


def test_nested_closure_comprehension_owner_attribution(tmp_path: Path) -> None:
    """Verifies that comprehensions inside nested closures are accurately attributed to their lexical parent path."""
    code = (
        "def outer():\n"
        "    def inner():\n"
        "        return [x for x in [1, 2, 3]]\n"
        "    return inner\n"
    )
    f = tmp_path / "nested_comp.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(
        str(f),
        repo_root=str(tmp_path),
        min_lines=1,
        min_tokens=1,
        harvest_closures=True,
        comprehensions=True,
    )
    comp_units = [u for u in units if u.get("kind") == "comprehension"]
    assert len(comp_units) == 1
    assert comp_units[0]["name"].startswith("outer:inner:listcomp_L")


def test_attribute_del_tracked_as_attribute_written(tmp_path: Path) -> None:
    """Verifies that 'del self.attr' is tracked in attrs_written and triggers instance binding."""
    code = (
        "class CacheManager:\n"
        "    def clear_cache(self):\n"
        "        del self._cache\n"
    )
    f = tmp_path / "del_attr.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "del_attr.py",
        "start": 2,
        "end": 3,
        "name": "CacheManager:clear_cache",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "self._cache" in scope["attrs_written"]
    assert scope["has_instance_binding"] is True
    assert scope["binding_kind"] == "instance"


def test_deleted_variable_not_treated_as_subroutine_output(tmp_path: Path) -> None:
    """Verifies that a locally deleted variable ('del x') is not returned in subroutine outputs."""
    code = (
        "def do_calc():\n"
        "    x = 10\n"
        "    del x\n"
        "    y = 20\n"
    )
    f = tmp_path / "del_var.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "del_var.py",
        "start": 2,
        "end": 4,
        "name": "do_calc:stmts",
        "kind": "sliding_window",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "x" not in scope["outputs"]
    assert "y" in scope["outputs"]


def test_local_import_not_treated_as_free_var_input_or_subroutine_output(tmp_path: Path) -> None:
    """Verifies that locally imported modules and functions are not treated as inputs or subroutine outputs."""
    code = (
        "def compute(val):\n"
        "    import math\n"
        "    from os.path import join\n"
        "    res = math.sqrt(val)\n"
        "    path = join('dir', str(res))\n"
    )
    f = tmp_path / "local_import.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "local_import.py",
        "start": 2,
        "end": 5,
        "name": "compute:stmts",
        "kind": "sliding_window",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "math" not in scope["inputs"]
    assert "join" not in scope["inputs"]
    assert "val" in scope["inputs"]
    assert "math" not in scope["outputs"]
    assert "join" not in scope["outputs"]
    assert "res" in scope["outputs"]
    assert "path" in scope["outputs"]


def test_compute_unit_diff_overlap_with_notebook_cell_fragment() -> None:
    """Verifies that notebook cell fragments (#cell_1) are stripped when matching diff ranges."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap  # pylint: disable=import-outside-toplevel

    unit = {"file": "notebooks/analysis.ipynb#cell_3", "start": 10, "end": 20}
    modified_ranges = {"notebooks/analysis.ipynb": [(12, 16)]}
    count, ratio = compute_unit_diff_overlap(unit, modified_ranges)
    assert count == 5
    assert ratio == round(5 / 11, 4)


def test_base_unit_name_fallback_for_underscore_or_empty_name() -> None:
    """Verifies that _base_unit_name falls back to 'helper' when unit name has only underscores."""
    from pydoppelgangerhunt.fixer import _base_unit_name  # pylint: disable=protected-access

    assert _base_unit_name({"name": "_", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "___", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "outer:_", "kind": "closure"}) == "helper"
    assert _base_unit_name({"name": "calculate", "kind": "function"}) == "calculate"


def test_for_loop_iter_evaluated_before_target_binding(tmp_path: Path) -> None:
    """Verifies that variables referenced in loop iterators are evaluated before target assignment."""
    code = (
        "def process(items):\n"
        "    for item in transform(items, item):\n"
        "        print(item)\n"
    )
    f = tmp_path / "for_order.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "for_order.py",
        "start": 2,
        "end": 3,
        "name": "process:for_loop",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "item" in scope["inputs"]
    assert "transform" in scope["inputs"]
    assert "items" in scope["inputs"]


def test_annassign_type_annotation_does_not_leak_into_inputs(tmp_path: Path) -> None:
    """Verifies that type annotations on local variables are not treated as runtime inputs."""
    code = (
        "def run():\n"
        "    data: CustomType = fetch_data()\n"
        "    entries: List[str] = parse(data)\n"
        "    return entries\n"
    )
    f = tmp_path / "ann_leak.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "ann_leak.py",
        "start": 1,
        "end": 4,
        "name": "run",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "CustomType" not in scope["inputs"]
    assert "List" not in scope["inputs"]
    assert "fetch_data" in scope["inputs"]
    assert "parse" in scope["inputs"]
    assert "entries" in scope["outputs"]


def test_while_test_walrus_treated_as_definite_assignment(tmp_path: Path) -> None:
    """Verifies that walrus expressions in while loop tests are marked as definite assignments."""
    code = (
        "def drain_stream(stream):\n"
        "    while (chunk := stream.read(1024)):\n"
        "        process(chunk)\n"
        "    return chunk\n"
    )
    f = tmp_path / "while_walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "while_walrus.py",
        "start": 2,
        "end": 3,
        "name": "drain_stream:while_loop",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "chunk" in scope["definite_stores"]


def test_nested_function_walrus_does_not_leak_to_outer_scope(tmp_path: Path) -> None:
    """Verifies that walrus expressions inside nested functions do not register in outer unit stores."""
    code = (
        "def outer(data):\n"
        "    def inner():\n"
        "        if (inner_val := process(data)):\n"
        "            return inner_val\n"
        "    return inner()\n"
    )
    f = tmp_path / "nested_walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "nested_walrus.py",
        "start": 1,
        "end": 5,
        "name": "outer",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "inner_val" not in scope["locals"]
    assert "inner_val" not in scope["outputs"]


def test_extract_unit_source_code_empty_and_directory_path(tmp_path: Path) -> None:
    """Verifies that extract_unit_source_code safely handles empty paths and directory paths."""
    u_empty = {"file": "", "name": "dummy", "start": 1, "end": 5}
    assert extract_unit_source_code(u_empty) == ["# Source for dummy lines 1-5\n"]

    u_dir = {"file": str(tmp_path), "name": "dir_unit", "start": 1, "end": 3}
    assert extract_unit_source_code(u_dir) == ["# Source for dir_unit lines 1-3\n"]


def test_assign_rhs_evaluation_order_captures_inputs(tmp_path: Path) -> None:
    """Verifies that Assign statements visit RHS values before LHS targets, capturing free variables."""
    code = (
        "def compute(item: int):\n"
        "    total = total + item\n"
        "    a, b = b, a\n"
        "    return total, a, b\n"
    )
    f = tmp_path / "assign_order.py"
    f.write_text(code, encoding="utf-8")

    u_sub = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "compute:sub",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u_sub, repo_root=str(tmp_path))
    assert "total" in scope["inputs"]
    assert "b" in scope["inputs"]
    assert "a" in scope["inputs"]
    assert "item" in scope["inputs"]


def test_standalone_comprehension_replacement_preserves_indentation() -> None:
    """Verifies that an expression unit occupying its own line retains its leading indentation."""
    source = "def foo():\n    [x for x in data]\n"
    unit = {
        "kind": "comprehension",
        "start": 2,
        "end": 2,
        "start_col": 4,
        "end_col": 21,
    }
    rep = "_shared(data)"
    result = replace_unit_in_source(source, unit, rep)
    assert result == "def foo():\n    _shared(data)\n"


def test_expression_unit_midline_pragma_placement() -> None:
    """Verifies that boundary pragmas are appended to the line end when trailing code is present."""
    source = "call([x for x in data], extra_arg)  # type: ignore\n"
    unit = {
        "kind": "comprehension",
        "start": 1,
        "end": 1,
        "start_col": 5,
        "end_col": 22,
    }
    rep = "_shared(data)"
    result = replace_unit_in_source(source, unit, rep, preserve_boundary_pragmas=True)
    assert "extra_arg" in result
    assert result.endswith("  # type: ignore\n")
    assert result == "call(_shared(data), extra_arg)  # type: ignore\n"


def test_check_units_overlap_same_line_column_bounded() -> None:
    """Verifies that distinct non-overlapping expressions on the same line are not marked as overlapping."""
    u1 = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 4,
        "end_col": 15,
    }
    u2 = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 18,
        "end_col": 29,
    }
    assert check_units_overlap(u1, u2) is False

    u_overlap = {
        "file": "test.py",
        "start": 5,
        "end": 5,
        "start_col": 10,
        "end_col": 20,
    }
    assert check_units_overlap(u1, u_overlap) is True


def test_refactor_module_units_same_line_column_order() -> None:
    """Verifies that refactor_module_units applies same-line replacements from right to left."""
    source = "val = [a for a in b] + [c for c in d]\n"
    u1 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 6,
        "end_col": 20,
    }
    u2 = {
        "file": "test.py",
        "start": 1,
        "end": 1,
        "start_col": 23,
        "end_col": 37,
    }
    replacements = [(u1, "helper(b)"), (u2, "helper(d)")]
    refactored = refactor_module_units(source, replacements)
    assert refactored == "val = helper(b) + helper(d)\n"


def test_scope_hierarchy_visitor_decorator_and_default_isolation(tmp_path: Path) -> None:
    """Verifies that decorator expressions and default arguments are scoped in enclosing outer frames."""
    code = (
        "@wrap([x for x in outer_list])\n"
        "class Service:\n"
        "    @route([y for y in class_list])\n"
        "    def handle(self, val=[z for z in def_list]):\n"
        "        pass\n"
    )
    f = tmp_path / "hierarchy.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=1, min_tokens=1, comprehensions=True)
    comps = {u["name"]: u for u in units if u["kind"] == "comprehension"}

    outer_c = comps.get("module:listcomp_L1")
    assert outer_c is not None
    assert outer_c.get("enclosing_class") is None

    class_c = comps.get("Service:listcomp_L3")
    assert class_c is not None
    assert class_c.get("enclosing_class") == "Service"
    assert "handle" not in class_c["name"]

    def_c = comps.get("Service:listcomp_L4")
    assert def_c is not None
    assert def_c.get("enclosing_class") == "Service"
    assert "handle" not in def_c["name"]


def test_harvest_file_units_nested_function_receiver_kind_isolation(tmp_path: Path) -> None:
    """Verifies that nested functions inside methods do not acquire receiver_kind='instance' when closures are disabled."""
    code = (
        "class Worker:\n"
        "    def run(self, data: list):\n"
        "        def inner(x: int) -> int:\n"
        "            y = x * 2\n"
        "            return y\n"
        "        return [inner(v) for v in data]\n"
    )
    f = tmp_path / "nested_worker.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=1, min_tokens=1, harvest_closures=False)
    inner_u = next((u for u in units if u["name"] == "inner"), None)
    assert inner_u is not None
    assert inner_u.get("receiver_kind") is None
    assert inner_u.get("is_static") is False


def test_populate_unit_receiver_metadata_enclosing_class_population(tmp_path: Path) -> None:
    """Verifies that _populate_unit_receiver_metadata populates missing enclosing_class."""
    code = (
        "class Model:\n"
        "    def predict(self, val: int) -> int:\n"
        "        return val * 10\n"
    )
    f = tmp_path / "model.py"
    f.write_text(code, encoding="utf-8")

    unit = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "predict",
        "kind": "function",
    }
    _populate_unit_receiver_metadata(unit, repo_root=str(tmp_path))
    assert unit.get("enclosing_class") == "Model"
    assert unit.get("enclosing_class_start") == 1
    assert unit.get("receiver_kind") == "instance"


def test_generate_refactoring_patch_mixed_static_instance_does_not_mark_static(tmp_path: Path) -> None:
    """Verifies that a mixed static/instance clone pair does not evaluate to is_static=True."""
    code = (
        "class Handler:\n"
        "    @staticmethod\n"
        "    def static_calc(a: int, b: int) -> int:\n"
        "        return a * 10 + b\n"
        "\n"
        "    def inst_calc(self, a: int, b: int) -> int:\n"
        "        return a * 10 + b\n"
    )
    f = tmp_path / "mixed.py"
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 3,
        "end": 4,
        "name": "static_calc",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "inst_calc",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "@staticmethod\n    def _shared" not in patch
    assert "@staticmethod\ndef _shared" not in patch
    assert "def _shared_static_calc_inst_calc(a: int, b: int) -> int:" in patch


def test_format_call_arguments_vararg_kwarg_ordering() -> None:
    """Verifies that _format_call_arguments orders positional arguments and free variables before varargs/kwargs."""
    inputs = ["self", "*args", "**kwargs", "extra_pos"]
    param_details = [
        {"name": "self", "kind": "pos"},
        {"name": "*args", "kind": "vararg"},
        {"name": "**kwargs", "kind": "kwarg"},
        {"name": "extra_pos", "kind": "pos"},
    ]
    args_str = _format_call_arguments(inputs, param_details, receiver_to_omit="self")
    assert args_str == "extra_pos, *args, **kwargs"
    # Ensure it parses cleanly without SyntaxError: positional argument follows keyword argument unpacking
    ast.parse(f"call({args_str})")


def test_analyze_block_assignment_short_circuit_walrus() -> None:
    """Verifies that short-circuit walrus expressions are correctly identified as conditional."""
    code_and = "if cond and (x := 1):\n    pass\n"
    d1, c1 = _analyze_block_assignment(ast.parse(code_and).body)
    assert "x" in c1
    assert "x" not in d1

    code_or = "if cond or (x := 1):\n    pass\n"
    d2, c2 = _analyze_block_assignment(ast.parse(code_or).body)
    assert "x" in c2
    assert "x" not in d2

    code_ifexp = "val = (x := 1) if cond else 0\n"
    d3, c3 = _analyze_block_assignment(ast.parse(code_ifexp).body)
    assert "val" in d3
    assert "x" in c3
    assert "x" not in d3


def test_analyze_block_assignment_unconditional_walrus() -> None:
    """Verifies that unconditionally evaluated walrus expressions are identified as definite."""
    code_assign = "val = (x := 1)\n"
    d1, c1 = _analyze_block_assignment(ast.parse(code_assign).body)
    assert "val" in d1
    assert "x" in d1
    assert "x" not in c1

    code_head = "if (x := 1) and cond:\n    pass\n"
    d2, c2 = _analyze_block_assignment(ast.parse(code_head).body)
    assert "x" in d2
    assert "x" not in c2


def test_analyze_block_assignment_with_target_variables() -> None:
    """Verifies that with/async with optional_vars targets are captured as definite assignments."""
    code_with = "with open('file') as fh:\n    data = fh.read()\n"
    d, _ = _analyze_block_assignment(ast.parse(code_with).body)
    assert "fh" in d


def test_analyze_block_assignment_try_finally_definite() -> None:
    """Verifies that statements in finally: blocks are recognized as definite assignments."""
    code_try = "try:\n    risky()\nfinally:\n    cleaned = True\n"
    d, _ = _analyze_block_assignment(ast.parse(code_try).body)
    assert "cleaned" in d


def test_extracted_helper_short_circuit_walrus_initialization(tmp_path: Path) -> None:
    """Verifies that short-circuit walrus inside extracted clone initializes output = None."""
    code = (
        "def worker_a(cond: bool, data: int) -> int:\n"
        "    if cond and (res := data * 2):\n"
        "        pass\n"
        "    return res\n"
        "\n"
        "def worker_b(cond: bool, data: int) -> int:\n"
        "    if cond and (res := data * 2):\n"
        "        pass\n"
        "    return res\n"
    )
    f = tmp_path / "walrus_sub.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "worker_a:block_L2", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "worker_b:block_L7", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "res = None" in patch


def test_extracted_helper_unconditional_walrus_no_redundant_init(tmp_path: Path) -> None:
    """Verifies that unconditionally evaluated walrus inside extracted clone does not initialize with None."""
    code = (
        "def worker_a(data: int) -> int:\n"
        "    total = (res := data * 2)\n"
        "    return total + res\n"
        "\n"
        "def worker_b(data: int) -> int:\n"
        "    total = (res := data * 2)\n"
        "    return total + res\n"
    )
    f = tmp_path / "walrus_uncond.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 2, "name": "worker_a:block_L2", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 6, "end": 6, "name": "worker_b:block_L6", "kind": "compound_block"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "res = None" not in patch


def test_nested_closure_receiver_attribute_tracking(tmp_path: Path) -> None:
    """Verifies that attributes accessed or mutated inside nested closures are attributed to the outer receiver."""
    code = (
        "class Pipeline:\n"
        "    def run(self, items: list) -> list:\n"
        "        def transform(x: int) -> int:\n"
        "            self.count += 1\n"
        "            return x + self.offset\n"
        "        return [transform(i) for i in items]\n"
    )
    f = tmp_path / "closure_receiver.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 6, "name": "run", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.count" in scope["attrs_read"]
    assert "self.count" in scope["attrs_written"]
    assert "self.offset" in scope["attrs_read"]
    assert "self.count" in scope["instance_attrs"]
    assert "self.offset" in scope["instance_attrs"]
    assert scope["has_receiver_access"] is True


def test_nested_class_receiver_attribute_isolation(tmp_path: Path) -> None:
    """Verifies that nested class methods do not leak their own self attributes to the enclosing method."""
    code = (
        "class Outer:\n"
        "    def execute(self, val: int) -> int:\n"
        "        class Inner:\n"
        "            def helper(self, x: int) -> int:\n"
        "                self.inner_data = x * 2\n"
        "                return self.inner_data\n"
        "        return Inner().helper(val)\n"
    )
    f = tmp_path / "nested_class_receiver.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 7, "name": "execute", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.inner_data" not in scope["attrs_read"]
    assert "self.inner_data" not in scope["attrs_written"]
    assert "self.inner_data" not in scope["instance_attrs"]


def test_nested_closure_shadowed_self_param_isolation(tmp_path: Path) -> None:
    """Verifies that local functions taking a parameter named 'self' do not leak receiver attributes."""
    code = (
        "class Service:\n"
        "    def process(self, data: int) -> int:\n"
        "        def inner(self, v: int) -> int:\n"
        "            return self.shadowed + v\n"
        "        return inner(self, data)\n"
    )
    f = tmp_path / "shadowed_self.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 5, "name": "process", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.shadowed" not in scope["attrs_read"]
    assert "self.shadowed" not in scope["instance_attrs"]


def test_synthesize_shared_helper_code_body_typing_imports(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code captures typing constructs inside the helper body."""
    code = (
        "def transform(data: list) -> list:\n"
        "    res: List[Dict[str, Any]] = []\n"
        "    for x in data:\n"
        "        res.append({'val': x})\n"
        "    return res\n"
    )
    f = tmp_path / "typing_body.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 5, "name": "transform", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u1, include_imports=True, repo_root=str(tmp_path))
    assert "from typing import" in helper
    assert "Dict" in helper
    assert "List" in helper
    assert "Any" in helper


def test_nonlocal_variable_mutation_recorded_in_stores(tmp_path: Path) -> None:
    """Verifies that nonlocal mutations within inner closures are recorded in unit stores and outputs."""
    code = (
        "def outer(limit: int) -> int:\n"
        "    total = 0\n"
        "    def increment(step: int) -> None:\n"
        "        nonlocal total\n"
        "        total += step\n"
        "    increment(limit)\n"
        "    return total\n"
    )
    f = tmp_path / "nonlocal_stores.py"
    f.write_text(code, encoding="utf-8")
    u_block = {
        "file": str(f),
        "start": 3,
        "end": 6,
        "name": "outer:sub",
        "kind": "compound_block",
    }
    raw_info = _inspect_unit_scope(u_block, repo_root=str(tmp_path))
    assert "total" in raw_info["stores"]
    assert "total" in raw_info["outputs"]

    scope = analyze_unit_variable_scope(u_block, repo_root=str(tmp_path))
    assert "total" in scope["outputs"]
    assert "total" in scope["nonlocals"]

def test_ast_delete_in_block_assignment() -> None:
    """Verifies that ast.Delete discards variables from definite stores unless reassigned."""
    tree1 = ast.parse("x = 1\ny = 2\ndel x\n")
    d1, _ = _analyze_block_assignment(tree1.body)
    assert "x" not in d1
    assert "y" in d1

    tree2 = ast.parse("x = 1\ndel x\nx = 3\n")
    d2, _ = _analyze_block_assignment(tree2.body)
    assert "x" in d2

    tree_del_tuple = ast.parse("a = 1\nb = 2\ndel a, b\n")
    d3, _ = _analyze_block_assignment(tree_del_tuple.body)
    assert "a" not in d3
    assert "b" not in d3


def test_inner_closure_delete_isolation(tmp_path: Path) -> None:
    """Verifies that del on an inner closure local variable does not delete outer variables of the same name."""
    code = (
        "def compute() -> int:\n"
        "    x = 10\n"
        "    def inner_cleanup() -> None:\n"
        "        x = 20\n"
        "        del x\n"
        "    inner_cleanup()\n"
        "    return x\n"
    )
    f = tmp_path / "inner_del.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 7, "name": "compute", "kind": "function"}
    raw_info = _inspect_unit_scope(u, repo_root=str(tmp_path))
    assert "x" not in raw_info.get("deleted_names", set())

    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "x" in scope["outputs"]


def test_if_else_terminating_branch_definite_assignment(tmp_path: Path) -> None:
    """Verifies that if-else with a terminating branch correctly tracks definite assignment."""
    code = (
        "def parse_or_raise(flag: bool, num: int) -> int:\n"
        "    if flag:\n"
        "        val = num * 2\n"
        "    else:\n"
        "        raise ValueError('invalid flag')\n"
        "    return val\n"
    )
    f = tmp_path / "terminating_else.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 6, "name": "parse_or_raise", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "val" in scope["definite_stores"]
    assert "val" not in scope["conditional_outputs"]

    helper = synthesize_shared_helper_code(u, u, repo_root=str(tmp_path))
    assert "Optional[" not in helper
    assert "val = None" not in helper


def test_if_terminating_no_else_conditional_isolation(tmp_path: Path) -> None:
    """Verifies that if-body assignments in a terminating if block do not leak into conditional outputs."""
    code = (
        "def guard_step(error: bool, data: str) -> str:\n"
        "    if error:\n"
        "        err_msg = f'Failed on {data}'\n"
        "        raise RuntimeError(err_msg)\n"
        "    res = data.upper()\n"
        "    return res\n"
    )
    f = tmp_path / "terminating_guard.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 6, "name": "guard_step", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "err_msg" not in scope["conditional_outputs"]
    assert "res" in scope["definite_stores"]
    assert "res" not in scope["conditional_outputs"]


def test_with_block_definite_assignment(tmp_path: Path) -> None:
    """Verifies that sequential assignments inside with blocks are recorded as definite stores."""
    code = (
        "def read_record(path: str) -> str:\n"
        "    with open(path, 'r', encoding='utf-8') as fh:\n"
        "        content = fh.read()\n"
        "    return content\n"
    )
    f = tmp_path / "with_block.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 4, "name": "read_record", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "content" in scope["definite_stores"]
    assert "content" not in scope["conditional_outputs"]


def test_try_terminating_handlers_definite_assignment(tmp_path: Path) -> None:
    """Verifies that try-except with terminating handlers or dual assignments tracks definite assignment."""
    code1 = (
        "def safe_convert(s: str) -> int:\n"
        "    try:\n"
        "        val = int(s)\n"
        "    except ValueError:\n"
        "        return 0\n"
        "    return val\n"
    )
    f1 = tmp_path / "try_term.py"
    f1.write_text(code1, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 6, "name": "safe_convert", "kind": "function"}
    scope1 = analyze_unit_variable_scope(u1, repo_root=str(tmp_path))
    assert "val" in scope1["definite_stores"]
    assert "val" not in scope1["conditional_outputs"]

    code2 = (
        "def fallback_convert(s: str) -> int:\n"
        "    try:\n"
        "        val = int(s)\n"
        "    except ValueError:\n"
        "        val = -1\n"
        "    return val\n"
    )
    f2 = tmp_path / "try_fallback.py"
    f2.write_text(code2, encoding="utf-8")
    u2 = {"file": str(f2), "start": 1, "end": 6, "name": "fallback_convert", "kind": "function"}
    scope2 = analyze_unit_variable_scope(u2, repo_root=str(tmp_path))
    assert "val" in scope2["definite_stores"]
    assert "val" not in scope2["conditional_outputs"]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_terminating_case_definite_assignment() -> None:
    """Verifies that Match with a terminating case preserves definite assignment across non-terminating branches."""
    code = (
        "match cmd:\n"
        "    case 'start':\n"
        "        res = 1\n"
        "    case 'stop':\n"
        "        res = 2\n"
        "    case _:\n"
        "        raise ValueError('unknown')\n"
    )
    tree = ast.parse(code)
    def_assigned, _ = _analyze_block_assignment(tree.body)
    assert "res" in def_assigned


def test_extract_deleted_names_inside_except_handler() -> None:
    """Verifies that _extract_deleted_names extracts deletions inside except handlers and preserves inner closures."""
    code = (
        "try:\n"
        "    val = 10\n"
        "except ValueError:\n"
        "    del err_val\n"
        "    def inner():\n"
        "        del inner_scoped\n"
    )
    tree = ast.parse(code)
    dels = _extract_deleted_names(tree.body)
    assert "err_val" in dels
    assert "inner_scoped" not in dels


def test_try_finally_deletion_discards_from_definite() -> None:
    """Verifies that _analyze_block_assignment discards variables deleted in finally blocks."""
    code = (
        "x = 10\n"
        "y = 20\n"
        "try:\n"
        "    z = 30\n"
        "finally:\n"
        "    del x\n"
    )
    tree = ast.parse(code)
    definite, _ = _analyze_block_assignment(tree.body)
    assert "x" not in definite
    assert "y" in definite


def test_chained_comparison_walrus_short_circuit() -> None:
    """Verifies that _walrus_assignment_in_expr treats chained comparison tails as conditional."""
    code = "res = (a < (b := 1) < (c := 2))"
    tree = ast.parse(code)
    assign_stmt = tree.body[0]
    assert isinstance(assign_stmt, ast.Assign)
    definite, conditional = _walrus_assignment_in_expr(assign_stmt.value)
    assert "b" in definite
    assert "c" in conditional


def test_find_enclosing_function_defensive(tmp_path: Path) -> None:
    """Verifies that find_enclosing_function handles functions without decorators or metadata safely."""
    code = "def sample():\n    pass\n"
    f = tmp_path / "sample_mod.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 2, "name": "sample", "kind": "function"}
    meta = find_enclosing_function(code, u)
    assert meta is not None
    assert meta["name"] == "sample"
    assert not meta["is_static"]
    assert not meta["is_class_method"]


def test_build_whole_method_delegation_single_line_function() -> None:
    """Verifies that _build_whole_method_delegation cleanly handles single-line functions without IndentationError."""
    code = "def add(x: int, y: int) -> int: return x + y\n"
    unit = {"file": "t.py", "start": 1, "end": 1, "name": "add", "kind": "function"}
    res = _build_whole_method_delegation(code, unit, "", "_shared_add", "x, y")
    assert "def add(x: int, y: int) -> int:\n" in res
    assert "return _shared_add(x, y)" in res
    parsed = ast.parse(res)
    assert len(parsed.body) == 1
    assert isinstance(parsed.body[0], ast.FunctionDef)


def test_compute_unit_diff_overlap_missing_file() -> None:
    """Verifies that compute_unit_diff_overlap safely returns (0, 0.0) when file key is missing or empty."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap  # pylint: disable=import-outside-toplevel

    assert compute_unit_diff_overlap({}, {"foo.py": [(1, 10)]}) == (0, 0.0)
    assert compute_unit_diff_overlap({"file": ""}, {"foo.py": [(1, 10)]}) == (0, 0.0)


def test_subroutine_with_local_function_scope_isolation(tmp_path: Path) -> None:
    """Verifies that compound blocks with local helper functions isolate arguments and returns."""
    code = (
        "def outer(val):\n"
        "    def helper(a):\n"
        "        return a * 2\n"
        "    result = helper(val)\n"
        "    return result\n"
    )
    f = tmp_path / "sub_fn.py"
    f.write_text(code, encoding="utf-8")
    u_block = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "outer:2-4",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u_block, repo_root=str(tmp_path))
    assert "val" in scope["inputs"]
    assert "a" not in scope["inputs"]
    assert "helper" not in scope["inputs"]
    assert "result" in scope["outputs"]
    assert "helper" in scope["outputs"]
    assert scope["is_control_flow_safe"] is True
    assert "embedded_return" not in scope["control_flow_hazards"]


def test_with_block_unconditional_termination_and_deletion() -> None:
    """Verifies that terminating with blocks are recognized and deleted names in with blocks are discarded."""
    term_code = "with lock:\n    return val\nx = 1\n"
    tree_term = ast.parse(term_code)
    assert _block_terminates(tree_term.body) is True

    del_code = "x = 1\nwith lock:\n    del x\n"
    tree_del = ast.parse(del_code)
    definite, conditional = _analyze_block_assignment(tree_del.body)
    assert "x" not in definite
    assert "x" not in conditional


def test_loop_block_deletion_promoted_to_conditional() -> None:
    """Verifies that variable deletions inside for and while loops are discarded from definite."""
    for_code = "x = 1\nfor i in items:\n    del x\n"
    tree_for = ast.parse(for_code)
    def_for, cond_for = _analyze_block_assignment(tree_for.body)
    assert "x" not in def_for
    assert "x" in cond_for

    while_code = "x = 1\nwhile cond:\n    del x\n"
    tree_while = ast.parse(while_code)
    def_while, cond_while = _analyze_block_assignment(tree_while.body)
    assert "x" not in def_while
    assert "x" in cond_while


def test_walrus_assignment_in_call_keywords_and_targets() -> None:
    """Verifies walrus expressions inside function call keywords and assignment/augassign targets."""
    call_stmt = ast.parse("fn(x=(y := 1), **(z := {}))").body[0]
    assert isinstance(call_stmt, ast.Expr)
    expr_call = call_stmt.value
    def_call, _ = _walrus_assignment_in_expr(expr_call)
    assert "y" in def_call
    assert "z" in def_call

    assign_code = "data[(k := 1)] = val\n"
    tree_assign = ast.parse(assign_code)
    def_assign, _ = _analyze_block_assignment(tree_assign.body)
    assert "k" in def_assign

    augassign_code = "data[(m := 2)] += val\n"
    tree_augassign = ast.parse(augassign_code)
    def_augassign, _ = _analyze_block_assignment(tree_augassign.body)
    assert "m" in def_augassign


def test_extract_unit_body_lines_single_line_function() -> None:
    """Verifies that _extract_unit_body_lines strips single-line function headers."""
    unit = {"kind": "function", "start": 1, "end": 1, "name": "foo"}
    raw_lines = ["def foo(x: int) -> int: return x * 2\n"]
    extracted = _extract_unit_body_lines(unit, raw_lines)
    assert extracted == ["return x * 2"]


def test_build_whole_method_delegation_decorated_single_line() -> None:
    """Verifies that _build_whole_method_delegation retains decorators on single-line functions."""
    code = "@decorator\ndef compute(x: int) -> int: return x * 2\n"
    unit = {"file": "t.py", "start": 1, "end": 2, "name": "compute", "kind": "function"}
    res = _build_whole_method_delegation(code, unit, "", "_shared_compute", "x")
    assert "@decorator\n" in res
    assert "def compute(x: int) -> int:\n" in res
    assert "return _shared_compute(x)" in res
    parsed = ast.parse(res)
    assert len(parsed.body) == 1
    fn_def = parsed.body[0]
    assert isinstance(fn_def, ast.FunctionDef)
    assert len(fn_def.decorator_list) == 1


def test_defensive_unit_file_and_name_none_handling() -> None:
    """Verifies that units with None or missing file/name attributes are handled without exceptions."""
    from pydoppelgangerhunt.fixer import filter_overlapping_clone_units  # pylint: disable=import-outside-toplevel

    u_none_file: Dict[str, Any] = {"file": None, "start": 1, "end": 5, "name": "test"}
    assert filter_overlapping_clone_units([u_none_file]) == [u_none_file]
    assert not check_units_overlap(u_none_file, {"file": "a.py", "start": 1, "end": 5})

    u_none_name: Dict[str, Any] = {"file": "a.py", "start": 1, "end": 5, "name": None}
    suggestion = synthesize_refactoring_suggestion(u_none_name, u_none_name)
    assert "[REFACTOR SUGGESTION]" in suggestion


def test_if_else_single_and_dual_branch_deletion() -> None:
    """Verifies that if/else deletions in only one branch demote to conditional, while both branches discard."""
    code_single = "x = 1\nif cond:\n    del x\nelse:\n    y = 2\n"
    d_single, c_single = _analyze_block_assignment(ast.parse(code_single).body)
    assert "x" not in d_single
    assert "x" in c_single
    assert "y" in c_single

    code_both = "x = 1\nif cond:\n    del x\nelse:\n    del x\n"
    d_both, c_both = _analyze_block_assignment(ast.parse(code_both).body)
    assert "x" not in d_both
    assert "x" not in c_both


def test_with_and_finally_conditional_deletion_demotion() -> None:
    """Verifies that deletions under conditional sub-blocks within with and finally demote to conditional."""
    code_with = "x = 1\nwith lock:\n    if cond:\n        del x\n"
    d_with, c_with = _analyze_block_assignment(ast.parse(code_with).body)
    assert "x" not in d_with
    assert "x" in c_with

    code_fin_cond = "x = 1\ntry:\n    pass\nfinally:\n    if cond:\n        del x\n"
    d_fin, c_fin = _analyze_block_assignment(ast.parse(code_fin_cond).body)
    assert "x" not in d_fin
    assert "x" in c_fin

    code_fin_uncond = "x = 1\ntry:\n    pass\nfinally:\n    del x\n"
    d_fin_u, c_fin_u = _analyze_block_assignment(ast.parse(code_fin_uncond).body)
    assert "x" not in d_fin_u
    assert "x" not in c_fin_u


def test_undefined_variable_deletion_does_not_create_conditional() -> None:
    """Verifies that deleting a variable that was never defined does not add it to conditional stores."""
    code = "if cond:\n    del z\n"
    d, c = _analyze_block_assignment(ast.parse(code).body)
    assert "z" not in d
    assert "z" not in c


def test_conditional_output_which_is_also_input_not_overwritten_with_none(tmp_path: Path) -> None:
    """Verifies that when a conditional output is also an input parameter, it is not initialized to None."""
    code1 = (
        "def compute(flag, total):\n"
        "    if flag:\n"
        "        total = total + 10\n"
        "    return total\n"
    )
    f1 = tmp_path / "mod1.py"
    f1.write_text(code1, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 3, "name": "compute:If", "kind": "compound_block"}
    helper = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    assert "total = None" not in helper
    assert "total: Any" in helper or "total" in helper


def test_refactoring_patch_global_and_nonlocal_not_assigned_at_call_site(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not generate call-site assignments for globals and declines nonlocals."""
    code_glob = (
        "total = 0\n"
        "def outer():\n"
        "    def inner1():\n"
        "        global total\n"
        "        total += 1\n"
        "    def inner2():\n"
        "        global total\n"
        "        total += 1\n"
    )
    f = tmp_path / "mod_glob.py"
    f.write_text(code_glob, encoding="utf-8")
    u1 = {"file": str(f), "start": 4, "end": 5, "name": "outer:inner1", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "outer:inner2", "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "total = _shared" not in patch
    assert "_shared" in patch

    # Nonlocal-dependent units must decline patch generation to avoid compile-time SyntaxError
    code_nl = (
        "def outer():\n"
        "    acc = 10\n"
        "    def inner1():\n"
        "        nonlocal acc\n"
        "        acc += 2\n"
        "    def inner2():\n"
        "        nonlocal acc\n"
        "        acc += 2\n"
    )
    f_nl = tmp_path / "mod_nl.py"
    f_nl.write_text(code_nl, encoding="utf-8")
    u1_nl = {"file": str(f_nl), "start": 4, "end": 5, "name": "outer:inner1", "kind": "compound_block"}
    u2_nl = {"file": str(f_nl), "start": 7, "end": 8, "name": "outer:inner2", "kind": "compound_block"}
    patch_nl = generate_refactoring_patch(
        [(1.0, u1_nl, u2_nl)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch_nl == ""


def test_class_level_comprehension_receiver_kind_is_none(tmp_path: Path) -> None:
    """Verifies that a comprehension in a class scope has receiver kind 'none'."""
    code = (
        "class Container:\n"
        "    items = [x * 2 for x in range(5)]\n"
    )
    f = tmp_path / "cls_comp.py"
    f.write_text(code, encoding="utf-8")
    u_comp = {
        "file": str(f),
        "start": 2,
        "end": 2,
        "name": "Container:listcomp_L2",
        "kind": "comprehension",
        "enclosing_class": "Container",
        "enclosing_class_start": 1,
    }
    helper = synthesize_shared_helper_code(u_comp, u_comp, repo_root=str(tmp_path))
    assert "self: Any" not in helper
    assert "def _shared" in helper


def test_defensive_unit_file_handling_in_diff_and_reporters(tmp_path: Path) -> None:
    """Verifies that diff, reporter, and coverage tools gracefully handle units with None or empty file."""
    from pydoppelgangerhunt.coverage import compute_unit_coverage
    from pydoppelgangerhunt.git_diff import check_temporal_divergence, compute_unit_diff_overlap
    from pydoppelgangerhunt.reporters import format_github_annotations

    u_none: Dict[str, Any] = {"file": None, "start": 1, "end": 2, "name": "dummy"}
    u_empty: Dict[str, Any] = {"file": "", "start": 1, "end": 2, "name": "dummy"}
    u_missing: Dict[str, Any] = {"start": 1, "end": 2}

    # Should not raise AttributeError or KeyError
    assert compute_unit_diff_overlap(u_none, {}) == (0, 0.0)
    assert compute_unit_diff_overlap(u_empty, {}) == (0, 0.0)
    assert compute_unit_diff_overlap(u_missing, {}) == (0, 0.0)

    lines = extract_unit_source_code(u_none, repo_root=str(tmp_path))
    assert len(lines) >= 1

    annots = format_github_annotations([(1.0, u_none, u_missing)])
    assert len(annots) == 2

    cov = compute_unit_coverage(u_none, {})
    assert cov == 0.0

    divergence = check_temporal_divergence(u_none, u_missing, repo_root=str(tmp_path))
    assert divergence is None


def test_defensive_unit_start_end_none_handling(tmp_path: Path) -> None:
    """Verifies that units with None start or end lines do not crash with TypeError."""
    from pydoppelgangerhunt.baseline import clone_pair_fingerprint, record_baseline
    from pydoppelgangerhunt.clustering import unit_key
    from pydoppelgangerhunt.fixer import check_units_overlap, filter_overlapping_clone_units, refactor_module_units

    u_null_bounds: Dict[str, Any] = {
        "file": str(tmp_path / "mod.py"),
        "start": None,
        "end": None,
        "name": None,
    }
    u_null_bounds2: Dict[str, Any] = {
        "file": str(tmp_path / "mod.py"),
        "start": None,
        "end": None,
        "name": None,
    }

    # Should not raise TypeError: int() argument must be ... not 'NoneType'
    key = unit_key(u_null_bounds)
    assert ":1-1:unit" in key

    assert check_units_overlap(u_null_bounds, u_null_bounds2) is True

    retained = filter_overlapping_clone_units([u_null_bounds, u_null_bounds2])
    assert len(retained) == 1

    fp = clone_pair_fingerprint(u_null_bounds, u_null_bounds2)
    assert "unit" in fp

    base_file = str(tmp_path / "base.json")
    saved = record_baseline([(1.0, u_null_bounds, u_null_bounds2)], base_file, "target", 0.8)
    assert Path(saved).is_file()

    out = refactor_module_units("x = 1\n", [])
    assert out == "x = 1\n"


def test_defensive_reporting_and_metrics_with_none_fields(tmp_path: Path) -> None:
    """Verifies that reporters, clustering, and metrics do not crash with NoneType/KeyError on synthetic units."""
    from pydoppelgangerhunt.baseline import extract_unit_namespace
    from pydoppelgangerhunt.clustering import cluster_clone_families
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats
    from pydoppelgangerhunt.reporters import (
        format_json_report,
        format_sarif_report,
        generate_clone_diff,
        generate_html_report,
    )

    # 1. extract_unit_namespace handles None or empty string gracefully
    assert extract_unit_namespace(None) == "."  # type: ignore[arg-type]
    assert extract_unit_namespace("") == "."

    u1: Dict[str, Any] = {
        "file": None,
        "name": None,
        "start": None,
        "end": None,
    }
    u2: Dict[str, Any] = {
        "file": None,
        "name": None,
        "start": None,
        "end": None,
    }

    # 2. generate_clone_diff
    diff = generate_clone_diff(u1, u2)
    assert isinstance(diff, str)

    # 3. format_sarif_report
    sarif = format_sarif_report([(0.9, u1, u2)], "target", 0.8)
    assert len(sarif["runs"][0]["results"]) == 1

    # 4. format_json_report with mock families
    mock_family = {
        "family_id": "CF-001",
        "member_count": 2,
        "unique_files": ["."],
        "avg_similarity": 0.9,
        "max_similarity": 0.9,
        "min_similarity": 0.9,
        "coherence": 1.0,
        "total_lines": 1,
        "medoid": {"name": None, "file": None, "start": None, "end": None},
        "members": [u1, u2],
    }
    json_rep = format_json_report([(0.9, u1, u2)], "target", 0.8, families=[mock_family])
    assert json_rep["clone_count"] == 1
    assert json_rep["clones"][0]["unit_a"]["name"] == "unit1"
    assert json_rep["families"][0]["medoid"]["name"] == "medoid"
    assert json_rep["families"][0]["members"][0]["name"] == "member"

    # 5. generate_html_report
    html = generate_html_report([(0.9, u1, u2)], "target", 0.8, families=[mock_family])
    assert "unit1" in html
    assert "unit2" in html

    # 6. compute_repository_dry_stats
    f = tmp_path / "sample.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    stats = compute_repository_dry_stats(str(tmp_path), [(0.9, u1, u2)])
    assert stats["sloc"] == 2
    assert stats["clone_pairs"] == 1

    # 7. cluster_clone_families
    u_a: Dict[str, Any] = {"file": None, "name": "a", "start": None, "end": None}
    u_b: Dict[str, Any] = {"file": None, "name": "b", "start": None, "end": None}
    families = cluster_clone_families([(0.95, u_a, u_b)], min_similarity_floor=0.8)
    assert len(families) == 1


def test_single_line_delegation_with_type_annotations_and_none_bounds() -> None:
    """Verifies that single-line functions with type annotations preserve their signatures during delegation."""
    from pydoppelgangerhunt.fixer import _build_whole_method_delegation, _find_sig_colon  # pylint: disable=protected-access
    from pydoppelgangerhunt.git_diff import check_temporal_divergence
    from pydoppelgangerhunt.reporters import format_github_annotations

    # 1. Single-line function with argument annotations and return annotation
    src = "def add(x: int, y: int = 1) -> int: return x + y\n"
    unit = {"file": "calc.py", "name": "add", "start": 1, "end": 1, "kind": "function"}
    delegation = _build_whole_method_delegation(
        src,
        unit,
        call_prefix="",
        helper_name="_shared_add",
        args_str="x, y=y",
        has_return=True,
    )
    assert "def add(x: int, y: int = 1) -> int:\n" in delegation
    assert "    return _shared_add(x, y=y)\n" in delegation

    # 2. _find_sig_colon unit tests
    assert _find_sig_colon("def f(x: int, y: int = 1) -> int: return x + y") == 32
    assert _find_sig_colon('def g(msg: str = "val:1") -> str: return msg') == 32
    assert _find_sig_colon("def h(): return {'a': 1, 'b': 2}") == 7

    # 3. _build_whole_method_delegation with all-None synthetic unit
    null_unit = {"file": None, "name": None, "start": None, "end": None}
    delegation_null = _build_whole_method_delegation(
        "def fallback(): pass\n",
        null_unit,
        call_prefix="",
        helper_name="_shared_fallback",
        args_str="",
    )
    assert isinstance(delegation_null, str)
    assert "_shared_fallback()" in delegation_null

    # 4. format_github_annotations defensive end bounds
    u_open = {"file": "target.py", "name": "open_unit", "start": 42, "end": None}
    u_other = {"file": "target.py", "name": "other_unit", "start": 100, "end": 105}
    ann = format_github_annotations([(0.9, u_open, u_other)])
    assert "line=42,endLine=42" in ann[0]

    # 5. check_temporal_divergence defensive None end handling
    res = check_temporal_divergence(u_open, u_other)
    assert res is None or isinstance(res, dict)


def test_syntax_error_fallback_delegation_and_defensive_extract_source_code() -> None:
    """Verifies scanner fallback delegation on SyntaxError and defensive source code / HTML report bounds."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
        _build_whole_method_delegation,
        _scan_sig_line,
    )
    from pydoppelgangerhunt.reporters import extract_unit_source_code, generate_html_report

    # 1. _scan_sig_line paren and colon detection
    pos, depth = _scan_sig_line("def f(x: int = 1) -> int: return x", 0)
    assert pos == 24
    assert depth == 0

    pos_m1, depth_m1 = _scan_sig_line("def f(", 0)
    assert pos_m1 == -1
    assert depth_m1 == 1

    pos_m2, depth_m2 = _scan_sig_line("    x: int = 1,", 1)
    assert pos_m2 == -1
    assert depth_m2 == 1

    pos_m3, depth_m3 = _scan_sig_line(") -> int: return x", 1)
    assert pos_m3 == 8
    assert depth_m3 == 0

    pos_c, depth_c = _scan_sig_line("def g():  # note: colon", 0)
    assert pos_c == 7
    assert depth_c == 0

    pos_bs, depth_bs = _scan_sig_line('def h(s: str = "\\\\"): pass', 0)
    assert pos_bs == 20
    assert depth_bs == 0

    # 2. _build_whole_method_delegation fallback on SyntaxError
    bad_src = "def add(x: int, y: int = 1) -> int: return x + y\nif {\n"
    unit = {"file": "mod.py", "name": "add", "start": 1, "end": 1}
    del_res = _build_whole_method_delegation(bad_src, unit, "", "_shared_add", "x, y=y")
    assert "def add(x: int, y: int = 1) -> int:\n" in del_res
    assert "return _shared_add(x, y=y)\n" in del_res
    assert "return x + y" not in del_res

    bad_multi = "def multi(\n    x: int,\n) -> int: return x + 1\nif {\n"
    unit_m = {"file": "mod.py", "name": "multi", "start": 1, "end": 3}
    del_m = _build_whole_method_delegation(bad_multi, unit_m, "", "_shared_multi", "x")
    assert ") -> int:\n" in del_m
    assert "return _shared_multi(x)\n" in del_m
    assert "return x + 1" not in del_m

    # 3. extract_unit_source_code with empty or missing files and None bounds/names
    u_none: Dict[str, Any] = {"file": "", "name": None, "start": 40, "end": None}
    lines_none = extract_unit_source_code(u_none)
    assert lines_none == ["# Source for unit lines 40-40\n"]

    u_missing: Dict[str, Any] = {"file": "nonexistent_file_xyz.py", "name": None, "start": 55, "end": None}
    lines_missing = extract_unit_source_code(u_missing)
    assert lines_missing == ["# Source for unit lines 55-55\n"]

    # 4. generate_html_report with family member None values
    fam: Dict[str, Any] = {
        "family_id": "CF-001",
        "member_count": 1,
        "avg_similarity": 0.9,
        "members": [{"file": None, "start": 10, "end": None, "name": None}],
        "medoid": None,
    }
    html = generate_html_report([], "target", 0.9, families=[fam])
    assert "CF-001" in html
    assert ":10-10</code> (member)" in html


def test_batch_32_exhaustive_review_hardening(tmp_path: Path) -> None:
    """Tests Batch 32 edge case fixes across fixer, diff, baseline, and reporters."""
    from pydoppelgangerhunt.fixer import (
        _block_terminates,
        _analyze_block_assignment,
        _walrus_assignment_in_expr,
    )
    from pydoppelgangerhunt.git_diff import get_git_blame_info
    from pydoppelgangerhunt.baseline import _match_clone_record, prune_baseline
    from pydoppelgangerhunt.reporters import generate_html_report
    import unittest.mock as mock

    # 1. _block_terminates on try-except-else terminating branches
    tree_term = ast.parse(
        "try:\n    x = 1\nexcept Exception:\n    return 2\nelse:\n    return 3\n"
    )
    assert _block_terminates(tree_term.body) is True

    tree_nonterm1 = ast.parse(
        "try:\n    return 1\nexcept Exception:\n    pass\n"
    )
    assert _block_terminates(tree_nonterm1.body) is False

    tree_nonterm2 = ast.parse(
        "try:\n    x = 1\nexcept Exception:\n    return 2\nelse:\n    pass\n"
    )
    assert _block_terminates(tree_nonterm2.body) is False

    # 2. _analyze_block_assignment with orelse deletions and unconditional deletion discarding
    # 2a. Deletion in else clause demoted to conditional
    tree_del_else = ast.parse(
        "x = 10\ntry:\n    pass\nexcept Exception:\n    pass\nelse:\n    del x\n"
    )
    d_else, c_else = _analyze_block_assignment(tree_del_else.body)
    assert "x" not in d_else
    assert "x" in c_else

    # 2b. Unconditional deletion in both try and except discarded from definite and conditional
    tree_del_both = ast.parse(
        "x = 10\ntry:\n    del x\nexcept Exception:\n    del x\n"
    )
    d_both, c_both = _analyze_block_assignment(tree_del_both.body)
    assert "x" not in d_both
    assert "x" not in c_both

    # 2c. Unconditional deletion in try with terminating except handler
    tree_del_term = ast.parse(
        "x = 10\ntry:\n    del x\nexcept Exception:\n    return 1\n"
    )
    d_term, c_term = _analyze_block_assignment(tree_del_term.body)
    assert "x" not in d_term
    assert "x" not in c_term

    # 3. _walrus_assignment_in_expr in AST slice bounds
    tree_slice = ast.parse("y = a[(b := 1) : (c := 2)]")
    assert isinstance(tree_slice.body[0], ast.Assign)
    d_sl, c_sl = _walrus_assignment_in_expr(tree_slice.body[0].value)
    assert "b" in (d_sl | c_sl)
    assert "c" in (d_sl | c_sl)

    # 4. get_git_blame_info with out-of-order headers without author line
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command") as mock_git:
        mock_git.return_value = (
            "0000000000000000000000000000000000000001 1 1 1\n"
            "author-time 1700000000\n"
            "summary Initial commit\n"
        )
        blame = get_git_blame_info("test.py", 1, 5)
        assert blame["author"] == "Unknown"
        assert blame["timestamp"] == 1700000000

    # 5. _match_clone_record and prune_baseline with None name_a and name_b values
    c_keys = {
        "fp": "fp",
        "sfp": "sfp",
        "ns_sfp": "ns_sfp",
        "pure_sfp": "pure_sfp",
        "namespaces": ["ns_a", "ns_b"],
        "names": ["name_a", "name_b"],
    }
    unconsumed = [
        {
            "pure_structural_fingerprint": "pure_sfp",
            "hash_a": "h1",
            "hash_b": "h2",
            "name_a": None,
            "name_b": None,
        }
    ]
    # Verify no TypeError '<' not supported between instances of 'NoneType'
    matched = _match_clone_record(c_keys, unconsumed)
    assert matched is not None

    b_file = tmp_path / "baseline.json"
    b_file.write_text(
        '{"fingerprints": [{"pure_structural_fingerprint": "pure_sfp", "hash_a": "h1", "hash_b": "h2", "name_a": null, "name_b": null}]}',
        encoding="utf-8",
    )
    pruned_cnt, retained_cnt = prune_baseline(str(b_file), [])
    assert pruned_cnt == 1
    assert retained_cnt == 0

    # 6. HTML entity escaping in generate_html_report
    u_special_1 = {
        "file": "pkg/foo.py",
        "start": 1,
        "end": 2,
        "name": "<lambda>",
    }
    u_special_2 = {
        "file": "pkg/bar.py",
        "start": 1,
        "end": 2,
        "name": "<listcomp>",
    }
    html_out = generate_html_report([(0.9, u_special_1, u_special_2)], "My <Package>", 0.8)
    assert "&lt;lambda&gt;" in html_out
    assert "&lt;listcomp&gt;" in html_out
    assert "My &lt;Package&gt;" in html_out


def test_batch_34_local_node_traversal_and_delegation_hardening(tmp_path: Path) -> None:
    """Tests Batch 34: _iter_local_nodes isolation and delegation tie-breaking."""
    from pydoppelgangerhunt.parser import harvest_file_units
    from pydoppelgangerhunt.fixer import _build_whole_method_delegation

    code = (
        "class Handler:\n"
        "    def handle(self, req):\n"
        "        def validate(data):\n"
        "            if not data:\n"
        "                a = 1\n"
        "                b = 2\n"
        "                c = 3\n"
        "                d = 4\n"
        "                e = 5\n"
        "                return False\n"
        "            return True\n"
        "        return validate(req)\n"
    )
    src_file = tmp_path / "handler.py"
    src_file.write_text(code, encoding="utf-8")

    units_with_closures = harvest_file_units(
        str(src_file),
        repo_root=str(tmp_path),
        min_lines=4,
        min_tokens=5,
        harvest_closures=True,
    )
    names = [u["name"] for u in units_with_closures]
    assert "handle:If" not in names
    assert "Handler.handle:If" not in names
    assert any("validate:If" in name for name in names)

    validate_if = next(u for u in units_with_closures if "validate:If" in u["name"])
    assert validate_if.get("receiver_kind") is None
    assert validate_if.get("is_static") is False

    clause_units = harvest_file_units(
        str(src_file),
        repo_root=str(tmp_path),
        min_lines=4,
        min_tokens=5,
        clause_level=True,
        harvest_closures=True,
    )
    c_names = [u["name"] for u in clause_units]
    assert "handle:if_branch" not in c_names

    delegation_code = (
        "class A:\n"
        "    def method(self):\n"
        "        return 42\n"
    )
    u_delegation = {
        "file": str(src_file),
        "start": 2,
        "end": 3,
        "name": "method",
        "kind": "function",
    }
    del_res = _build_whole_method_delegation(
        delegation_code,
        u_delegation,
        call_prefix="self.",
        helper_name="_shared_method",
        args_str="",
    )
    assert "def method(self):" in del_res
    assert "return self._shared_method()" in del_res


def test_batch_35_git_blame_dictionary_porcelain_parsing() -> None:
    """Tests Batch 35: git blame porcelain parsing with out-of-order and repeated commit headers."""
    import unittest.mock as mock
    from pydoppelgangerhunt.git_diff import get_git_blame_info

    mock_out_1 = (
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1 1 1\n"
        "summary Add new feature\n"
        "author Alice\n"
        "author-time 1710000000\n"
    )
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=mock_out_1):
        blame_1 = get_git_blame_info("mod.py", 1, 1)
        assert blame_1["author"] == "Alice"
        assert blame_1["commit"] == "aaaaaaaa"
        assert blame_1["timestamp"] == 1710000000
        assert blame_1["summary"] == "Add new feature"

    mock_out_2 = (
        "1111111111111111111111111111111111111111 1 1 1\n"
        "author-time 1720000000\n"
        "author Bob\n"
        "summary Bob commit\n"
        "2222222222222222222222222222222222222222 2 2 1\n"
        "author-time 1710000000\n"
        "author Charlie\n"
        "summary Charlie commit\n"
        "1111111111111111111111111111111111111111 3 3 1\n"
    )
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=mock_out_2):
        blame_2 = get_git_blame_info("mod.py", 1, 3)
        assert blame_2["author"] == "Bob"
        assert blame_2["commit"] == "11111111"
        assert blame_2["timestamp"] == 1720000000
        assert blame_2["summary"] == "Bob commit"


def test_batch_36_path_resolution_and_same_file_matching(tmp_path: Any) -> None:
    """Tests Batch 36: robust path normalization and resolution across duplicate units."""
    from pydoppelgangerhunt.fixer import (
        _is_same_file_path,
        _normalize_file_path,
        check_units_overlap,
        filter_overlapping_clone_units,
        generate_refactoring_patch,
        synthesize_shared_helper_code,
    )

    # 1. Base string equivalence and anchor handling
    assert not _is_same_file_path("", "foo.py")
    assert not _is_same_file_path("foo.py", "")
    assert _is_same_file_path("foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo/bar.py#hash1", "foo/bar.py#hash2")
    assert _is_same_file_path("./foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo\\bar.py", "foo/bar.py")
    assert _normalize_file_path("") == ""
    assert _normalize_file_path("./mod.py#h").endswith("mod.py")

    # 2. Filesystem-based relative vs absolute path equivalence
    target_file = tmp_path / "sample.py"
    target_file.write_text(
        "class Worker:\n"
        "    def task_a(self, x):\n"
        "        val = x * 2 + 1\n"
        "        return val\n"
        "    def task_b(self, x):\n"
        "        val = x * 2 + 1\n"
        "        return val\n",
        encoding="utf-8",
    )

    abs_path_str = str(target_file)
    rel_path_dot = "./sample.py"
    rel_path_plain = "sample.py"

    assert _is_same_file_path(abs_path_str, rel_path_dot, repo_root=str(tmp_path))
    assert _is_same_file_path(rel_path_plain, rel_path_dot, repo_root=str(tmp_path))

    # 3. check_units_overlap with differing path formats
    u_base = {"file": abs_path_str, "start": 2, "end": 4}
    u_overlap = {"file": rel_path_dot, "start": 3, "end": 5}
    u_disjoint = {"file": rel_path_dot, "start": 5, "end": 7}
    assert check_units_overlap(u_base, u_overlap)
    assert not check_units_overlap(u_base, u_disjoint)

    # 4. filter_overlapping_clone_units groups identical files despite leading dot-slash
    filtered = filter_overlapping_clone_units([u_base, u_overlap])
    assert len(filtered) == 1

    # 5. synthesize_shared_helper_code recognizes same-class across relative and absolute paths
    u1 = {
        "file": abs_path_str,
        "name": "task_a",
        "type": "function",
        "kind": "function",
        "start": 2,
        "end": 4,
        "params": ["self", "x"],
        "enclosing_class": "Worker",
        "enclosing_class_start": 1,
        "receiver_kind": "method",
    }
    u2 = {
        "file": rel_path_dot,
        "name": "task_b",
        "type": "function",
        "kind": "function",
        "start": 5,
        "end": 7,
        "params": ["self", "x"],
        "enclosing_class": "Worker",
        "enclosing_class_start": 1,
        "receiver_kind": "method",
    }
    helper_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_task_a_task_b(self, x: Any) -> Any:" in helper_code

    # 6. generate_refactoring_patch succeeds with mixed path formats
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_task_a_task_b(self, x: Any) -> Any:" in patch
    assert "self._shared_task_a_task_b(x)" in patch


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


def test_batch_38_windows_path_case_insensitivity_and_unicode_resilience(tmp_path: Any, monkeypatch: Any) -> None:
    """Tests Batch 38: Windows case-insensitive path matching, UnicodeDecodeError resilience, and markdown table escaping."""
    import sys
    from pydoppelgangerhunt.baseline import load_baseline, prune_baseline
    from pydoppelgangerhunt.config import (
        find_matching_path_value,
        init_tool_configuration,
        paths_match_boundary,
    )
    from pydoppelgangerhunt.fixer import (
        _is_same_file_path,
        _populate_unit_receiver_metadata,
        generate_refactoring_patch,
    )
    from pydoppelgangerhunt.reporters import format_markdown_summary

    # 1. Windows case-insensitive boundary path matching
    monkeypatch.setattr(sys, "platform", "win32")
    assert paths_match_boundary("C:/Project/Sub/Engine.py", "c:/project/sub/engine.py")
    assert paths_match_boundary("C:/Project/Sub/Engine.py", "sub/engine.py")
    assert paths_match_boundary("sub/engine.py", "C:/Project/Sub/Engine.py")
    assert not paths_match_boundary("C:/Project/Sub/Other.py", "sub/engine.py")

    # 2. find_matching_path_value with Windows case insensitivity
    data_map = {"c:/repo/sub/module.py": 42}
    val = find_matching_path_value("C:/Repo/Sub/Module.py", data_map)
    assert val == 42
    val_rel = find_matching_path_value("sub/module.py", data_map)
    assert val_rel == 42

    # 3. _is_same_file_path using boundary matching
    assert _is_same_file_path("./Sub/Module.py#cell1", "sub/module.py")
    assert _is_same_file_path("C:/Repo/Sub/Module.py", "sub/module.py")

    # 4. UnicodeDecodeError resilience across file reading
    bad_file = tmp_path / "corrupted.py"
    bad_file.write_bytes(b"\x80\xff\xfe\x00\x01not-utf8")
    u_corrupt = {"file": str(bad_file), "start": 1, "end": 2, "name": "bad"}

    # _populate_unit_receiver_metadata does not raise
    _populate_unit_receiver_metadata(u_corrupt, repo_root=str(tmp_path))
    assert u_corrupt.get("receiver_kind") is None

    # generate_refactoring_patch skips corrupted files gracefully
    patch = generate_refactoring_patch([(1.0, u_corrupt, u_corrupt)], repo_root=str(tmp_path))
    assert patch == ""

    # load_baseline and prune_baseline with non-UTF8 baseline file
    bad_baseline = tmp_path / "bad_baseline.json"
    bad_baseline.write_bytes(b"\xff\xfe\x00corrupt")
    base_fps = load_baseline(str(bad_baseline))
    assert len(base_fps) == 0

    pruned = prune_baseline(str(bad_baseline), [])
    assert pruned.pruned_count == 0

    # init_tool_configuration with non-UTF8 pyproject.toml leaves file untouched and uses standalone config
    bad_pyproject = tmp_path / "pyproject.toml"
    bad_bytes = b"\x80\xff[project]\nname='bad'\n"
    bad_pyproject.write_bytes(bad_bytes)
    res_cfg = init_tool_configuration(str(tmp_path))
    assert bad_pyproject.read_bytes() == bad_bytes
    assert ".pydoppelgangerhunt.toml" in res_cfg

    # 5. Markdown table escaping for pipe characters in package names
    stats_with_pipe = {
        "dry_score": 95.0,
        "grade": "A",
        "sloc": 1000,
        "dloc": 50,
        "duplication_pct": 5.0,
        "clone_pairs": 1,
        "clone_families": 1,
        "package_sloc": {"pkg|with|pipes": 500, "normal_pkg": 500},
    }
    summary = format_markdown_summary(stats_with_pipe, "test_repo")
    assert "| `pkg\\|with\\|pipes` | 500 |" in summary
    assert "| `normal_pkg` | 500 |" in summary


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


def test_batch_40_notebook_source_code_and_matcher_path_robustness(tmp_path: Any) -> None:
    """Tests Batch 40: notebook cell source extraction, clone diffing, and matcher/baseline path equivalence."""
    import json
    from pydoppelgangerhunt.baseline import (
        clone_pair_fingerprint,
        clone_pair_structural_fingerprint,
        extract_unit_namespace,
        record_baseline,
    )
    from pydoppelgangerhunt.matcher import merge_adjacent_clones, suppress_subclones
    from pydoppelgangerhunt.reporters import (
        extract_unit_source_code,
        format_sarif_report,
        generate_clone_diff,
    )

    # 1. extract_unit_source_code and generate_clone_diff from Jupyter Notebook code cells
    nb_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["def calc(x):\n", "    y = x * 2\n", "    return y\n"],
            },
            {
                "cell_type": "code",
                "source": ["def calc(x):\n", "    y = x * 3\n", "    return y\n"],
            },
        ]
    }
    nb_path = tmp_path / "notebook.ipynb"
    nb_path.write_text(json.dumps(nb_content), encoding="utf-8")

    u1 = {"file": f"{nb_path}#cell_1", "start": 1, "end": 3, "name": "calc"}
    u2 = {"file": f"{nb_path}#cell_2", "start": 1, "end": 3, "name": "calc"}

    lines1 = extract_unit_source_code(u1)
    assert lines1 == ["def calc(x):\n", "    y = x * 2\n", "    return y\n"]

    diff_text = generate_clone_diff(u1, u2)
    assert "-    y = x * 2" in diff_text
    assert "+    y = x * 3" in diff_text

    # Out-of-bounds cell index fallback
    u_oob = {"file": f"{nb_path}#cell_99", "start": 1, "end": 3, "name": "oob"}
    lines_oob = extract_unit_source_code(u_oob)
    assert lines_oob == []

    # 2. format_sarif_report normalizes leading dot-slash while retaining cell anchors
    u_dot_cell = {"file": "./notebook.ipynb#cell_1", "start": 1, "end": 3, "name": "calc"}
    sarif = format_sarif_report([(1.0, u_dot_cell, u_dot_cell)], "repo", 0.9)
    uri = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri == "notebook.ipynb#cell_1"

    # 3. merge_adjacent_clones consolidates hits across dot-slash file representations
    u1_w1 = {"file": "./mod.py", "name": "f:w1", "start": 10, "end": 15, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}
    u2_w1 = {"file": "mod.py", "name": "g:w1", "start": 30, "end": 35, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}
    u1_w2 = {"file": "mod.py", "name": "f:w2", "start": 11, "end": 16, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}
    u2_w2 = {"file": "./mod.py", "name": "g:w2", "start": 31, "end": 36, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}

    merged = merge_adjacent_clones([(1.0, u1_w1, u2_w1), (1.0, u1_w2, u2_w2)], line_tolerance=2)
    assert len(merged) == 1
    assert merged[0][1]["start"] == 10
    assert merged[0][1]["end"] == 16

    # 4. suppress_subclones eliminates redundant sub-clones across dot-slash file representations
    u_parent1 = {"file": "./worker.py", "name": "worker", "start": 1, "end": 50, "token_count": 100}
    u_parent2 = {"file": "worker.py", "name": "worker2", "start": 1, "end": 50, "token_count": 100}
    u_child1 = {"file": "worker.py", "name": "worker:w1", "start": 10, "end": 25, "token_count": 30}
    u_child2 = {"file": "./worker.py", "name": "worker2:w1", "start": 10, "end": 25, "token_count": 30}

    suppressed = suppress_subclones([(1.0, u_parent1, u_parent2), (0.98, u_child1, u_child2)])
    assert len(suppressed) == 1
    assert suppressed[0][1]["name"] == "worker"

    # 5. baseline fingerprint and recording path normalization
    u_b1 = {"file": "./core/engine.py", "start": 5, "end": 15, "name": "run", "tokens": ["a", "b"]}
    u_b2 = {"file": "core/engine.py", "start": 5, "end": 15, "name": "run", "tokens": ["a", "b"]}
    assert clone_pair_fingerprint(u_b1, u_b1) == clone_pair_fingerprint(u_b2, u_b2)
    assert clone_pair_structural_fingerprint(u_b1, u_b1) == clone_pair_structural_fingerprint(u_b2, u_b2)
    assert extract_unit_namespace("./core/engine.py") == "core"

    base_json = tmp_path / "baseline.json"
    record_baseline([(1.0, u_b1, u_b2)], str(base_json), "repo", 0.9)
    base_data = json.loads(base_json.read_text(encoding="utf-8"))
    assert base_data["fingerprints"][0]["file_a"] == "core/engine.py"
    assert base_data["fingerprints"][0]["file_b"] == "core/engine.py"


def test_batch_41_unified_path_normalization_and_html_escaping(tmp_path: Path) -> None:
    """Verifies Batch 41 unified path normalization across reporters, diff, coverage, and HTML escaping."""
    from pydoppelgangerhunt.cli import _audit_clone_risk_warnings  # pylint: disable=protected-access
    from pydoppelgangerhunt.config import normalize_path_string
    from pydoppelgangerhunt.coverage import _read_xml_coverage  # pylint: disable=protected-access
    from pydoppelgangerhunt.fixer import generate_refactoring_patch
    from pydoppelgangerhunt.git_diff import check_temporal_divergence, parse_git_diff_hunks
    from pydoppelgangerhunt.reporters import (
        format_github_annotations,
        format_json_report,
        generate_html_report,
    )

    # 1. normalize_path_string handles repetitive dot-slash and backslashes
    assert normalize_path_string("././foo/bar.py") == "foo/bar.py"
    assert normalize_path_string(".\\.\\foo\\bar.py") == "foo/bar.py"
    assert normalize_path_string("./foo/bar.py#cell_1", strip_anchor=False) == "foo/bar.py#cell_1"
    assert normalize_path_string("./foo/bar.py#cell_1", strip_anchor=True) == "foo/bar.py"

    # 2. format_json_report normalizes paths while preserving anchors
    u1 = {"file": "./pkg/mod.py#cell_1", "start": 5, "end": 10, "name": "fn1", "token_count": 25}
    u2 = {"file": ".\\pkg\\mod.py#cell_2", "start": 15, "end": 20, "name": "fn2", "token_count": 25}
    mock_fam = {
        "family_id": "CF-100",
        "member_count": 2,
        "unique_files": ["pkg/mod.py"],
        "avg_similarity": 0.95,
        "max_similarity": 0.95,
        "min_similarity": 0.95,
        "coherence": 1.0,
        "total_lines": 10,
        "medoid": {"file": "./pkg/mod.py#cell_1", "name": "fn1", "start": 5, "end": 10},
        "members": [u1, u2],
    }
    json_out = format_json_report([(0.95, u1, u2)], "repo", 0.9, families=[mock_fam])
    assert json_out["clones"][0]["unit_a"]["file"] == "pkg/mod.py#cell_1"
    assert json_out["clones"][0]["unit_b"]["file"] == "pkg/mod.py#cell_2"
    assert json_out["families"][0]["medoid"]["file"] == "pkg/mod.py#cell_1"
    assert json_out["families"][0]["members"][1]["file"] == "pkg/mod.py#cell_2"

    # 3. format_github_annotations strips anchors and leading dot-slash
    annots = format_github_annotations([(0.95, u1, u2)])
    assert len(annots) == 2
    assert "file=pkg/mod.py," in annots[0]
    assert "#cell_" not in annots[0]

    # 4. generate_html_report escapes target and renders normalized paths
    dangerous_target = "<script>alert('pwned')</script>&foo"
    html_report = generate_html_report([(0.95, u1, u2)], dangerous_target, 0.9, families=[mock_fam])
    assert "<script>alert('pwned')</script>" not in html_report
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;&amp;foo" in html_report
    assert "pkg/mod.py#cell_1" in html_report
    assert "pkg/mod.py#cell_2" in html_report

    # 5. _audit_clone_risk_warnings path normalization
    cov_data = {"pkg/mod.py#cell_1": {5, 6, 7, 8, 9, 10}, "pkg/mod.py#cell_2": set()}
    warn_lines = _audit_clone_risk_warnings(u1, u2, cov_data=cov_data, use_color=False)
    assert any("Asymmetric test coverage: pkg/mod.py#cell_1 (100%) vs pkg/mod.py#cell_2 (0%)" in w for w in warn_lines)

    # 6. parse_git_diff_hunks path normalization
    diff_raw = "--- a/pkg/mod.py\n+++ b/./pkg/mod.py\n@@ -1,5 +1,5 @@\n+line\n"
    hunks = parse_git_diff_hunks(diff_raw)
    assert "pkg/mod.py" in hunks

    # 7. check_temporal_divergence path normalization
    assert check_temporal_divergence(u1, u2, repo_root=str(tmp_path)) is None

    # 8. generate_refactoring_patch patch header comment path normalization
    src_file = tmp_path / "calc.py"
    src_file.write_text("def a():\n    return 42\ndef b():\n    return 42\n", encoding="utf-8")
    u_c1 = {"file": f"./{src_file.name}", "start": 1, "end": 2, "name": "a"}
    u_c2 = {"file": str(src_file), "start": 3, "end": 4, "name": "b"}
    patch = generate_refactoring_patch([(1.0, u_c1, u_c2)], repo_root=str(tmp_path))
    assert f"# Clone Pair (100.0%): {src_file.name} <===>" in patch

    # 9. _read_xml_coverage path normalization
    xml_file = tmp_path / "coverage.xml"
    xml_file.write_text(
        '<coverage><packages><package><classes>'
        f'<class filename="./sub/module.py"><lines><line number="10" hits="1"/></lines></class>'
        '</classes></package></packages></coverage>',
        encoding="utf-8",
    )
    cov_xml = _read_xml_coverage(str(xml_file))
    assert "sub/module.py" in cov_xml

    # 10. canonical_path_key and _normalize_exemption_endpoint
    import os
    from pydoppelgangerhunt.config import canonical_path_key
    from pydoppelgangerhunt.matcher import _normalize_exemption_endpoint, scan_target
    from pydoppelgangerhunt.clustering import unit_key, _normalize_unit_file  # pylint: disable=protected-access
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats

    ep = _normalize_exemption_endpoint("./pkg/Engine.py:ComputeData")
    assert ep.endswith(":ComputeData")
    if os.name == "nt" or sys.platform == "win32":
        assert ep.startswith("pkg/engine.py:")
    else:
        assert ep.startswith("pkg/Engine.py:")

    # 11. scan_target with symbol-level and whole-file exemptions
    ex_dir = tmp_path / "ex_dir"
    ex_dir.mkdir()
    mod_file = ex_dir / "service.py"
    mod_file.write_text(
        "def ProcessTask(x):\n    a = x * 2\n    b = a + 1\n    c = b * 3\n    return c\n\n"
        "def HandleTask(x):\n    a = x * 2\n    b = a + 1\n    c = b * 3\n    return c\n",
        encoding="utf-8",
    )
    exemptions_sym = [(f"{mod_file}:ProcessTask", f"{mod_file}:HandleTask")]
    clones_sym = scan_target(str(ex_dir), min_lines=4, threshold=0.8, exemptions=exemptions_sym)
    assert len(clones_sym) == 0

    exemptions_file = [(str(mod_file), str(mod_file))]
    clones_file = scan_target(str(ex_dir), min_lines=4, threshold=0.8, exemptions=exemptions_file)
    assert len(clones_file) == 0

    # 12. clustering unit_key and _normalize_unit_file with case variants
    u_lower = {"file": "c:/repo/mod.py", "name": "fn", "start": 1, "end": 5}
    u_upper = {"file": "C:/REPO/MOD.PY", "name": "fn", "start": 1, "end": 5}
    if os.name == "nt" or sys.platform == "win32":
        assert unit_key(u_lower) == unit_key(u_upper)
        assert _normalize_unit_file(u_lower) == _normalize_unit_file(u_upper)

    # 13. metrics compute_repository_dry_stats DLOC deduplication
    stats_case = compute_repository_dry_stats(
        str(tmp_path),
        [(1.0, u_lower, u_upper)],
    )
    if os.name == "nt" or sys.platform == "win32":
        assert stats_case["dloc"] == 5


def test_batch_47_git_diff_metrics_matcher_clustering(tmp_path: Path) -> None:
    """Batch 47: Test git diff hunk parsing, blame parsing, UnionFind rank orders,

    metrics relative package path resolution, matcher edge cases, and coverage readers.
    """
    # pylint: disable=protected-access,import-outside-toplevel
    import sqlite3
    from pydoppelgangerhunt.baseline import clone_pair_fingerprint, prune_baseline
    from pydoppelgangerhunt.coverage import _read_sqlite_coverage, _read_xml_coverage
    from pydoppelgangerhunt.git_diff import (
        _run_git_command,
        get_git_blame_info,
        get_git_modified_line_ranges,
    )
    from pydoppelgangerhunt.matcher import (
        _split_exemption_endpoint,
        call_sequence_similarity,
        compute_pair_similarity,
        jaccard_similarity,
        merge_adjacent_clones,
        tfidf_multiset_jaccard_similarity,
    )

    # 1. git_diff: parse_git_diff_hunks single-line hunks and deleted-file bleed prevention
    diff_data = (
        "--- a/file_one.py\n"
        "+++ b/file_one.py\n"
        "@@ -10 +10 @@\n"
        "+added_single_line\n"
        "--- a/file_deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,5 +0,0 @@\n"
        "-deleted_line\n"
        "--- a/file_two.py\n"
        "+++ b/file_two.py\n"
        "@@ -20,2 +20,2 @@\n"
        "+line20\n+line21\n"
    )
    parsed_hunks = parse_git_diff_hunks(diff_data)
    assert parsed_hunks.get("file_one.py") == [(10, 10)]
    assert "file_deleted.py" not in parsed_hunks
    assert parsed_hunks.get("file_two.py") == [(20, 21)]

    # 2. git_diff: _run_git_command exception safety and range queries
    with mock.patch("subprocess.run", side_effect=OSError("binary failure")):
        assert _run_git_command(["status"]) is None

    mock_diff = "+++ b/target_mod.py\n@@ -5,3 +5,3 @@\n"
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=mock_diff):
        ranges = get_git_modified_line_ranges(since_ref="HEAD~1")
        assert "target_mod.py" in ranges

    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=None):
        assert get_git_modified_line_ranges() == {}

    # 3. git_diff: filter_clones_by_git_diff empty ranges & both_units flag
    unit_alpha = {"file": "mod_a.py", "start": 5, "end": 15}
    unit_beta = {"file": "mod_b.py", "start": 5, "end": 15}
    clone_candidate = [(0.92, unit_alpha, unit_beta)]
    assert filter_clones_by_git_diff(clone_candidate, {}) == []

    matched_ranges = {"mod_a.py": [(5, 10)], "mod_b.py": [(5, 10)]}
    both_filtered = filter_clones_by_git_diff(
        clone_candidate, matched_ranges, both_units=True
    )
    assert len(both_filtered) == 1

    # 4. git_diff: get_git_blame_info empty file and invalid timestamp handling
    blame_empty = get_git_blame_info("", 1, 5)
    assert blame_empty["author"] == "Unknown"

    porcelain_sample = (
        "fedcba9876543210fedcba9876543210fedcba98 1 1 1\n"
        "author Dev Master\n"
        "author-time not_a_valid_integer\n"
        "summary Patch update\n"
        "\tval = 42\n"
    )
    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command", return_value=porcelain_sample):
        blame_res = get_git_blame_info("mod_a.py", 1, 1)
        assert blame_res["commit"] == "fedcba98"
        assert blame_res["author"] == "Dev Master"
        assert blame_res["timestamp"] == 0

    # 5. UnionFind: rank orders and self-union branch coverage
    uf = UnionFind()
    assert uf.find("item_solo") == "item_solo"
    root_1 = uf.union("elem_a", "elem_b")
    root_2 = uf.union("elem_c", "elem_d")
    assert uf.union("elem_e", "elem_c") == root_2
    assert uf.union("elem_c", "elem_f") == root_2
    assert uf.union("elem_a", "elem_a") == root_1

    # 6. clustering: tolerance-floor rejection and medoid sub-threshold rejection
    cand_1 = {"file": "c1.py", "name": "f1", "start": 1, "end": 10}
    cand_2 = {"file": "c2.py", "name": "f2", "start": 1, "end": 10}
    fams_tol = cluster_clone_families(
        [(0.77, cand_1, cand_2)],
        linkage="complete",
        min_similarity_floor=0.80,
        linkage_tolerance=0.05,
    )
    assert len(fams_tol) == 0

    cand_3 = {"file": "c3.py", "name": "f3", "start": 1, "end": 10}
    # In medoid linkage, cand_3 cannot merge with {cand_1, cand_2} because sim(cand_2, cand_3) < floor
    fams_med = cluster_clone_families(
        [(0.95, cand_1, cand_2), (0.75, cand_2, cand_3)],
        linkage="medoid",
        min_similarity_floor=0.80,
    )
    assert len(fams_med) == 1
    assert fams_med[0]["member_count"] == 2

    # 7. metrics: package_sloc relative to target_dir and OSError handling
    stat_root = tmp_path / "repo_metrics"
    svc_sub = stat_root / "services_pkg"
    svc_sub.mkdir(parents=True)
    (svc_sub / "handler.py").write_text("def run_job():\n    return 100\n", encoding="utf-8")
    (stat_root / "entrypoint.py").write_text("import sys\n", encoding="utf-8")

    stats = compute_repository_dry_stats(str(stat_root), [])
    assert "services_pkg" in stats["package_sloc"]

    with mock.patch("builtins.open", side_effect=OSError("Disk read failure")):
        stats_error = compute_repository_dry_stats(str(stat_root), [])
        assert stats_error["sloc"] == 0

    # 8. matcher: empty fallbacks, exemptions, and alternative similarity modes
    assert jaccard_similarity(set(), set()) == 0.0
    assert call_sequence_similarity([], []) == 0.0
    assert tfidf_multiset_jaccard_similarity({}, {}, {}) == 0.0
    assert merge_adjacent_clones([]) == []

    assert _split_exemption_endpoint("no_colon_path") == ("no_colon_path", None)
    assert _split_exemption_endpoint("file.py:sym") == ("file.py", "sym")

    unit_trace_1 = {"calls": ["step_a", "step_b", "step_c"]}
    unit_trace_2 = {"calls": ["step_a", "step_b", "step_c"]}
    assert compute_pair_similarity(unit_trace_1, unit_trace_2, call_sequences=True) == 1.0

    unit_vec_1 = {"vector": {"tok_x": 3, "tok_y": 1}, "shingles": set()}
    unit_vec_2 = {"vector": {"tok_x": 3, "tok_y": 1}, "shingles": set()}
    idf_map = {"tok_x": 1.2, "tok_y": 1.8}
    assert compute_pair_similarity(
        unit_vec_1, unit_vec_2, tfidf=True, bag_of_tokens=True, idf_weights=idf_map
    ) == 1.0

    # 9. baseline: plain set consumable suppression and prune error recovery
    u_base_1 = {"file": "b1.py", "name": "fn1", "tokens": ["tok_a", "tok_b"], "start": 1, "end": 10}
    u_base_2 = {"file": "b2.py", "name": "fn2", "tokens": ["tok_a", "tok_b"], "start": 1, "end": 10}
    fp_base = clone_pair_fingerprint(u_base_1, u_base_2)
    plain_fp_set = {fp_base}
    rem, supp = filter_clones_by_baseline([(0.95, u_base_1, u_base_2)], plain_fp_set)
    assert supp == 1
    assert len(rem) == 0

    with mock.patch(
        "pydoppelgangerhunt.git_diff.get_git_modified_line_ranges",
        side_effect=RuntimeError("git failure"),
    ):
        base_doc = {"fingerprints": [99999, "legacy_raw_str", {"fingerprint": "dict_raw"}]}
        base_file = tmp_path / "mock_baseline.json"
        base_file.write_text(json.dumps(base_doc), encoding="utf-8")
        prune_res = prune_baseline(str(base_file), active_clones=[])
        assert isinstance(prune_res, tuple)

    # 10. coverage: SQLite bit bounds & unmapped IDs, and XML parse exceptions
    db_cov = tmp_path / ".coverage_bit_bounds"
    db_conn = sqlite3.connect(str(db_cov))
    db_c = db_conn.cursor()
    db_c.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    db_c.execute("CREATE TABLE line_bits (file_id INTEGER, num_bits INTEGER, bits BLOB)")
    db_c.execute("INSERT INTO file VALUES (1, 'recorded.py')")
    db_c.execute("INSERT INTO line_bits VALUES (1, 2, ?)", (b"\xff",))
    db_c.execute("INSERT INTO line_bits VALUES (888, 5, ?)", (b"\xff",))
    db_conn.commit()
    db_conn.close()

    cov_result = _read_sqlite_coverage(str(db_cov))
    assert cov_result.get("recorded.py") == {1, 2}

    xml_cov = tmp_path / "cov_malformed.xml"
    xml_cov.write_text(
        "<coverage><project><package><classes><class filename='m.py'>"
        "<lines><line number='not_a_num' hits='1'/></lines></class></classes></package></project></coverage>",
        encoding="utf-8",
    )
    assert _read_xml_coverage(str(xml_cov)) == {"m.py": set()}
    xml_cov.write_text("<<<NOT_XML>>>", encoding="utf-8")
    assert _read_xml_coverage(str(xml_cov)) == {}


def test_batch_54_fixer_path_resolution_and_receiver_metadata(tmp_path: Path) -> None:
    """Batch 54: Test _normalize_file_path and _populate_unit_receiver_metadata edge cases."""
    # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (
        _normalize_file_path,
        _populate_unit_receiver_metadata,
        find_enclosing_class,
        find_enclosing_function,
    )

    # 1. Test _normalize_file_path with existing cwd-relative file and repo_root
    pyproject = "pyproject.toml"
    assert _normalize_file_path(pyproject, repo_root=str(tmp_path)).endswith("pyproject.toml")
    assert _normalize_file_path("") == ""

    # 2. Test _populate_unit_receiver_metadata with file existing relative to tmp_path
    mod_code = (
        "class Service:\n"
        "    @classmethod\n"
        "    def create(cls, data):\n"
        "        return cls(data)\n"
    )
    mod_file = tmp_path / "service.py"
    mod_file.write_text(mod_code, encoding="utf-8")

    unit_cls = {"file": "service.py", "start": 3, "end": 4, "name": "create"}
    _populate_unit_receiver_metadata(unit_cls, repo_root=str(tmp_path))
    assert unit_cls.get("enclosing_class") == "Service"
    assert unit_cls.get("receiver_kind") == "class"

    # 3. Test find_enclosing_class and find_enclosing_function edge cases
    assert find_enclosing_class("", unit_cls) is None
    assert find_enclosing_function("", unit_cls) is None


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


def test_batch_58_parenthesized_return_and_relative_path_resolution(tmp_path: Path) -> None:
    """Batch 58: Test parenthesized/tabbed return detection and relative path resolution in patch/harvest."""
    # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import generate_refactoring_patch, synthesize_shared_helper_code
    from pydoppelgangerhunt.parser import harvest_file_units

    # 1. Test parenthesized return 'return(res)' in synthesize_shared_helper_code
    code_paren = (
        "def compute_paren(val: int) -> int:\n"
        "    res = val * 2\n"
        "    return(res)\n"
    )
    f1 = tmp_path / "mod_paren.py"
    f1.write_text(code_paren, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 3, "name": "compute_paren", "kind": "function"}

    helper_paren = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    # Must contain return(res) and not append a duplicate 'return res'
    assert "return(res)" in helper_paren
    assert "return res" not in helper_paren

    # Test tabbed return 'return\tres'
    code_tab = (
        "def compute_tab(val: int) -> int:\n"
        "    res = val * 2\n"
        "    return\tres\n"
    )
    f2 = tmp_path / "mod_tab.py"
    f2.write_text(code_tab, encoding="utf-8")
    u2 = {"file": str(f2), "start": 1, "end": 3, "name": "compute_tab", "kind": "function"}

    helper_tab = synthesize_shared_helper_code(u2, u2, repo_root=str(tmp_path))
    assert "return\tres" in helper_tab
    assert "return res" not in helper_tab

    # 2. Test generate_refactoring_patch with subpackage path and relative repo_root
    subpkg = tmp_path / "nested_subpkg"
    subpkg.mkdir(parents=True, exist_ok=True)
    sub_file = subpkg / "logic.py"
    sub_code = (
        "def process(a: int, b: int) -> int:\n"
        "    c = a + b\n"
        "    return c\n"
    )
    sub_file.write_text(sub_code, encoding="utf-8")
    u_sub = {
        "file": str(sub_file),
        "start": 1,
        "end": 3,
        "name": "process",
        "kind": "function",
    }
    patch = generate_refactoring_patch([(1.0, u_sub, u_sub)], repo_root=str(tmp_path))
    assert "--- a/nested_subpkg/logic.py" in patch
    assert "+++ b/nested_subpkg/logic.py" in patch

    # 3. Test harvest_file_units with absolute file_path and relative repo_root
    units = harvest_file_units(
        str(sub_file.resolve()),
        repo_root=str(tmp_path.resolve()),
        min_lines=1,
        min_tokens=1,
    )
    assert len(units) >= 1
    assert units[0]["file"] == "nested_subpkg/logic.py"


def test_batch_60_generator_docstring_call_site_and_overlap(tmp_path: Path) -> None:
    """Test generator call site docstring synthesis and check_units_overlap column boundary validation."""
    from pydoppelgangerhunt.fixer import check_units_overlap, synthesize_shared_helper_code

    # 1. Sync generator without return value
    gen_file = tmp_path / "gen_logic.py"
    gen_file.write_text(
        "def produce_stream(limit: int):\n"
        "    for val in range(limit):\n"
        "        yield val * 2\n",
        encoding="utf-8",
    )
    u_gen = {"file": str(gen_file), "start": 1, "end": 3, "name": "produce_stream", "kind": "function"}
    helper_gen = synthesize_shared_helper_code(u_gen, u_gen, repo_root=str(tmp_path))
    assert "yield from _shared_produce_stream(...)" in helper_gen

    # 2. Sync generator with return value (StopIteration.value)
    gen_ret_file = tmp_path / "gen_ret.py"
    gen_ret_file.write_text(
        "def accumulate_stream(limit: int):\n"
        "    accum = 0\n"
        "    for val in range(limit):\n"
        "        accum += val\n"
        "        yield val\n"
        "    return accum\n",
        encoding="utf-8",
    )
    u_gen_ret = {"file": str(gen_ret_file), "start": 1, "end": 6, "name": "accumulate_stream", "kind": "function"}
    helper_gen_ret = synthesize_shared_helper_code(u_gen_ret, u_gen_ret, repo_root=str(tmp_path))
    assert "(yield from _shared_accumulate_stream(...))" in helper_gen_ret

    # 3. Async generator
    agen_file = tmp_path / "agen_logic.py"
    agen_file.write_text(
        "async def produce_async(limit: int):\n"
        "    for val in range(limit):\n"
        "        yield val * 3\n",
        encoding="utf-8",
    )
    u_agen = {"file": str(agen_file), "start": 1, "end": 3, "name": "produce_async", "kind": "function"}
    helper_agen = synthesize_shared_helper_code(u_agen, u_agen, repo_root=str(tmp_path))
    assert "async for _item in _shared_produce_async(...):" in helper_agen
    assert "yield _item" in helper_agen

    # 4. check_units_overlap column boundary validation
    u_base = {"file": "core.py", "start": 10, "end": 10}
    # Overlapping columns (0..20 and 15..30)
    assert check_units_overlap(
        dict(u_base, start_col=0, end_col=20),
        dict(u_base, start_col=15, end_col=30),
    ) is True
    # Disjoint columns (0..10 and 15..30)
    assert check_units_overlap(
        dict(u_base, start_col=0, end_col=10),
        dict(u_base, start_col=15, end_col=30),
    ) is False
    # Malformed / inverted column bounds (e.g. start_col > end_col)
    assert check_units_overlap(
        dict(u_base, start_col=25, end_col=10),
        dict(u_base, start_col=15, end_col=30),
    ) is False


def test_batch_61_column_precision_and_diff_robustness() -> None:
    """Verifies SARIF/annotation column precision, matcher column handling, and diff parser robustness."""
    from pydoppelgangerhunt.git_diff import parse_git_diff_hunks
    from pydoppelgangerhunt.matcher import merge_adjacent_clones, suppress_subclones
    from pydoppelgangerhunt.reporters import format_github_annotations, format_sarif_report

    # 1. format_sarif_report with column bounds
    u1 = {
        "file": "src/calc.py",
        "name": "calc_a",
        "start": 10,
        "end": 12,
        "start_col": 4,
        "end_col": 28,
    }
    u2 = {
        "file": "src/calc.py",
        "name": "calc_b",
        "start": 20,
        "end": 22,
        "start_col": 8,
        "end_col": 32,
    }
    sarif = format_sarif_report([(0.95, u1, u2)], target="src", threshold=0.90)
    loc_region = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    rel_region = sarif["runs"][0]["results"][0]["relatedLocations"][0]["physicalLocation"]["region"]
    assert loc_region["startLine"] == 10
    assert loc_region["endLine"] == 12
    assert loc_region["startColumn"] == 5
    assert loc_region["endColumn"] == 29
    assert rel_region["startLine"] == 20
    assert rel_region["endLine"] == 22
    assert rel_region["startColumn"] == 9
    assert rel_region["endColumn"] == 33

    # Omitted columns
    u_no_col = {"file": "src/calc.py", "name": "c", "start": 5, "end": 6}
    sarif_no_col = format_sarif_report([(0.90, u_no_col, u_no_col)], target="src", threshold=0.90)
    region_plain = sarif_no_col["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert "startColumn" not in region_plain
    assert "endColumn" not in region_plain

    # 2. format_github_annotations with column bounds
    annots = format_github_annotations([(0.95, u1, u2)])
    assert len(annots) == 2
    assert "col=5,endColumn=29" in annots[0]
    assert "col=9,endColumn=33" in annots[1]

    annots_no_col = format_github_annotations([(0.90, u_no_col, u_no_col)])
    assert "col=" not in annots_no_col[0]

    # 3. merge_adjacent_clones column propagation
    unit_a1 = {
        "file": "pkg/mod.py",
        "name": "foo:block1",
        "start": 10,
        "end": 15,
        "start_col": 4,
        "end_col": 20,
        "shingles": {"s1"},
        "tokens": ["a", "b"],
        "token_count": 2,
    }
    unit_a2 = {
        "file": "pkg/mod.py",
        "name": "foo:block2",
        "start": 16,
        "end": 20,
        "start_col": 0,
        "end_col": 35,
        "shingles": {"s2"},
        "tokens": ["c", "d"],
        "token_count": 2,
    }
    unit_b1 = {
        "file": "pkg/other.py",
        "name": "bar:block1",
        "start": 30,
        "end": 35,
        "start_col": 8,
        "end_col": 25,
        "shingles": {"s1"},
        "tokens": ["a", "b"],
        "token_count": 2,
    }
    unit_b2 = {
        "file": "pkg/other.py",
        "name": "bar:block2",
        "start": 36,
        "end": 40,
        "start_col": 2,
        "end_col": 40,
        "shingles": {"s2"},
        "tokens": ["c", "d"],
        "token_count": 2,
    }
    merged = merge_adjacent_clones([(0.9, unit_a1, unit_b1), (0.9, unit_a2, unit_b2)])
    assert len(merged) == 1
    m_sim, m_u1, m_u2 = merged[0]
    assert m_sim >= 0.0
    assert m_u1["start"] == 10 and m_u1["end"] == 20
    assert m_u1["start_col"] == 4 and m_u1["end_col"] == 35
    assert m_u2["start"] == 30 and m_u2["end"] == 40
    assert m_u2["start_col"] == 8 and m_u2["end_col"] == 40

    # 4. suppress_subclones column awareness
    p1 = {"file": "mod.py", "name": "stmt1", "start": 10, "end": 10, "start_col": 0, "end_col": 80}
    p2 = {"file": "mod.py", "name": "stmt2", "start": 20, "end": 20, "start_col": 0, "end_col": 80}
    c1 = {"file": "mod.py", "name": "expr1", "start": 10, "end": 10, "start_col": 15, "end_col": 40}
    c2 = {"file": "mod.py", "name": "expr2", "start": 20, "end": 20, "start_col": 15, "end_col": 40}

    # Child is strictly inside column bounds of parent -> suppressed
    surviving = suppress_subclones([(0.95, p1, p2), (0.92, c1, c2)])
    assert len(surviving) == 1
    assert surviving[0][1]["name"] == "stmt1"

    # Child with wider column bounds than parent -> NOT suppressed
    c_wide = {"file": "mod.py", "name": "wide1", "start": 10, "end": 10, "start_col": 0, "end_col": 90}
    not_suppressed = suppress_subclones([(0.95, p1, p2), (0.92, c_wide, c2)])
    assert len(not_suppressed) == 2

    # 5. parse_git_diff_hunks robustness
    diff_sample = (
        "--- a/src/util.py\t2026-09-17 10:00:00\n"
        "+++ b/src/util.py\t2026-09-17 11:00:00\n"
        "@@ -10,2 +10,5 @@ def f():\n"
        "--- /dev/null\n"
        '+++ "b/path with space/module.py"\n'
        "@@ -0,0 +1,10 @@\n"
        "--- a/corrupt.py\n"
        "+++ b/corrupt.py\n"
        "@@ -bad +corrupt,spec @@\n"
    )
    hunks = parse_git_diff_hunks(diff_sample)
    assert "src/util.py" in hunks
    assert hunks["src/util.py"] == [(10, 14)]
    assert "path with space/module.py" in hunks
    assert hunks["path with space/module.py"] == [(1, 10)]
    assert "corrupt.py" not in hunks


def test_batch_62_toml_inline_comments_and_coverage_cleanup(tmp_path: Path) -> None:
    """Verifies TOML inline comment parsing, coverage SQLite cleanup, and metrics single-file resolution."""
    import sqlite3
    from pydoppelgangerhunt.config import _strip_toml_inline_comment, load_toml_section
    from pydoppelgangerhunt.coverage import _read_sqlite_coverage, _read_xml_coverage
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats

    # 1. _strip_toml_inline_comment and load_toml_section
    assert _strip_toml_inline_comment('threshold = 0.85 # comment') == "threshold = 0.85"
    assert _strip_toml_inline_comment('key = "value # not comment"') == 'key = "value # not comment"'
    assert _strip_toml_inline_comment("key = 'single # quote'") == "key = 'single # quote'"
    assert _strip_toml_inline_comment("# full line comment") == ""

    toml_file = tmp_path / "pyproject.toml"
    toml_file.write_text(
        """
[tool.pydoppelgangerhunt]
threshold = 0.85 # Minimum similarity threshold
min_lines = 10 # Minimal lines
call_sequences = true # Call sequence analysis
exclude = ["venv", "build#dir", ".git"] # Exclude patterns
""",
        encoding="utf-8",
    )
    cfg = load_toml_section(toml_file, "pydoppelgangerhunt")
    assert cfg.get("threshold") == 0.85
    assert cfg.get("min_lines") == 10
    assert cfg.get("call_sequences") is True
    assert cfg.get("exclude") == ["venv", "build#dir", ".git"]

    # 2. _read_sqlite_coverage connection cleanup and file unlink
    db_file = tmp_path / "test.coverage"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE file (id INTEGER PRIMARY KEY, path TEXT)")
    conn.execute("CREATE TABLE line_bits (file_id INTEGER, num_bits INTEGER, bits BLOB)")
    conn.execute("INSERT INTO file VALUES (1, 'src/main.py')")
    # Bit 0 of byte 0 set -> line 1 covered
    conn.execute("INSERT INTO line_bits VALUES (1, 8, ?)", (b"\x01",))
    conn.commit()
    conn.close()

    cov_map = _read_sqlite_coverage(str(db_file))
    assert "src/main.py" in cov_map
    assert 1 in cov_map["src/main.py"]
    # File should be completely closed and un-lockable on Windows
    db_file.unlink()
    assert not db_file.exists()

    # 3. _read_xml_coverage positive line check
    xml_file = tmp_path / "coverage.xml"
    xml_file.write_text(
        """<?xml version="1.0" ?>
<coverage version="7.0">
  <packages>
    <package name="pkg">
      <classes>
        <class name="mod" filename="pkg/mod.py">
          <lines>
            <line number="0" hits="1" />
            <line number="-5" hits="1" />
            <line number="12" hits="0" />
            <line number="42" hits="3" />
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )
    xml_cov = _read_xml_coverage(str(xml_file))
    assert "pkg/mod.py" in xml_cov
    assert 42 in xml_cov["pkg/mod.py"]
    assert 12 not in xml_cov["pkg/mod.py"]
    assert 0 not in xml_cov["pkg/mod.py"]
    assert -5 not in xml_cov["pkg/mod.py"]

    # 4. compute_repository_dry_stats on a single file target
    single_script = tmp_path / "script.py"
    single_script.write_text("x = 1\ny = 2\n# comment\nz = x + y\n", encoding="utf-8")
    stats = compute_repository_dry_stats(str(single_script), clones=[])
    assert stats["sloc"] == 3
    assert stats["dloc"] == 0
    assert stats["dry_score"] == 100.0
    assert "script.py" in stats["package_sloc"]


def test_batch_63_multiline_toml_and_baseline_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests multi-line TOML array parsing in zero-dep fallback, baseline structural reconstruction, and record disambiguation."""
    # pylint: disable=import-outside-toplevel
    import builtins
    from pydoppelgangerhunt.baseline import (
        _match_clone_record,
        _record_matches_names,
        load_baseline,
        prune_baseline,
    )
    from pydoppelgangerhunt.config import _parse_toml_array_value, load_toml_section

    # 1. Direct _parse_toml_array_value validation
    assert _parse_toml_array_value("['alpha', 'beta']") == ["alpha", "beta"]
    assert _parse_toml_array_value("[ 10, 2.5, true, false, 'text' ]") == [
        10,
        2.5,
        True,
        False,
        "text",
    ]
    assert not _parse_toml_array_value("")

    # 2. Multi-line TOML array parsing via zero-dependency line fallback
    multiline_toml = tmp_path / "multiline.toml"
    multiline_toml.write_text(
        """[tool.pydoppelgangerhunt]
threshold = 0.88 # inline comment
exclude = [
    "venv", # comment 1
    ".venv",
    "build",
    "dist",
]
flags = [
    true,
    false,
]
values = [ 10, 20, 30 ]
""",
        encoding="utf-8",
    )
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomli", "tomllib"):
            raise ImportError("mocked no toml parser")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    cfg = load_toml_section(multiline_toml, "pydoppelgangerhunt")
    assert cfg.get("threshold") == 0.88
    assert cfg.get("exclude") == ["venv", ".venv", "build", "dist"]
    assert cfg.get("flags") == [True, False]
    assert cfg.get("values") == [10, 20, 30]
    monkeypatch.undo()

    # 3. Baseline structural fingerprint reconstruction in load_baseline and prune_baseline
    base_json = tmp_path / "legacy_base.json"
    base_json.write_text(
        json.dumps({
            "version": "1.3.0",
            "fingerprints": [
                {
                    "file_a": "pkg/mod_a.py",
                    "name_a": "compute",
                    "hash_a": "aaaa111122223333",
                    "file_b": "pkg/mod_b.py",
                    "name_b": "calculate",
                    "hash_b": "bbbb111122223333",
                }
            ],
        }),
        encoding="utf-8",
    )
    loaded_base = load_baseline(str(base_json))
    assert len(loaded_base.records) == 1
    rec = loaded_base.records[0]
    expected_sfp = "pkg/mod_a.py#aaaa111122223333 <===> pkg/mod_b.py#bbbb111122223333"
    assert rec.get("structural_fingerprint") == expected_sfp
    assert expected_sfp in loaded_base

    u1 = {"file": "pkg/mod_a.py", "name": "new_compute", "structural_hash": "aaaa111122223333"}
    u2 = {"file": "pkg/mod_b.py", "name": "new_calc", "structural_hash": "bbbb111122223333"}
    prune_res = prune_baseline(str(base_json), [(0.95, u1, u2)], unstaged_modified_ranges={})
    assert prune_res.retained_count == 1
    assert prune_res.pruned_count == 0

    # 4. _match_clone_record Pass 1 disambiguation prioritizing matching structural hash
    rec_stale = {
        "fingerprint": "service.py:worker <===> client.py:worker",
        "structural_fingerprint": "service.py#old1 <===> client.py#old2",
        "name_a": "worker",
        "name_b": "worker",
    }
    rec_fresh = {
        "fingerprint": "service.py:worker <===> client.py:worker",
        "structural_fingerprint": "service.py#new1 <===> client.py#new2",
        "name_a": "worker",
        "name_b": "worker",
    }
    keys_p1 = {
        "fp": "service.py:worker <===> client.py:worker",
        "sfp": "service.py#new1 <===> client.py#new2",
        "ns_sfp": "dummy",
        "pure_sfp": "dummy",
        "namespaces": [".", "."],
        "names": ["worker", "worker"],
    }
    matched_p1 = _match_clone_record(keys_p1, [rec_stale, rec_fresh])
    assert matched_p1 is rec_fresh

    # 5. _match_clone_record Pass 3/4 disambiguation prioritizing symbol names
    rec_diff_names = {
        "namespaced_structural_fingerprint": "pkg#h1 <===> pkg#h2",
        "pure_structural_fingerprint": "h1 <===> h2",
        "namespace_a": "pkg",
        "namespace_b": "pkg",
        "name_a": "alpha",
        "name_b": "beta",
    }
    rec_same_names = {
        "namespaced_structural_fingerprint": "pkg#h1 <===> pkg#h2",
        "pure_structural_fingerprint": "h1 <===> h2",
        "namespace_a": "pkg",
        "namespace_b": "pkg",
        "name_a": "worker",
        "name_b": "worker",
    }
    assert _record_matches_names(rec_same_names, ["worker", "worker"]) is True
    assert _record_matches_names(rec_diff_names, ["worker", "worker"]) is False

    keys_ns = {
        "fp": "nomatch",
        "sfp": "nomatch",
        "ns_sfp": "pkg#h1 <===> pkg#h2",
        "pure_sfp": "h1 <===> h2",
        "namespaces": ["pkg", "pkg"],
        "names": ["worker", "worker"],
    }
    matched_ns = _match_clone_record(keys_ns, [rec_diff_names, rec_same_names])
    assert matched_ns is rec_same_names


def test_batch_64_control_flow_hazards_and_baseline_pruning(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch declines units with naked loop controls,

    and prune_baseline correctly uses sfp_to_clone and name disambiguation across identical hashes.
    """
    from pydoppelgangerhunt.baseline import prune_baseline  # pylint: disable=import-outside-toplevel

    # 1. Test generate_refactoring_patch with naked break and naked continue
    loop_code = (
        "def process_items(items: list) -> None:\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            break\n"
        "        if x == 0:\n"
        "            continue\n"
        "        print(x)\n"
    )
    src_file = tmp_path / "proc.py"
    src_file.write_text(loop_code, encoding="utf-8")

    u_break = {
        "file": str(src_file),
        "start": 3,
        "end": 4,
        "name": "process_items:If_break",
        "kind": "compound_block",
    }
    u_continue = {
        "file": str(src_file),
        "start": 5,
        "end": 6,
        "name": "process_items:If_continue",
        "kind": "compound_block",
    }

    # Patch generation should decline (return empty string) for naked_break and naked_continue
    patch_break = generate_refactoring_patch([(0.95, u_break, u_break)], repo_root=str(tmp_path))
    assert patch_break == ""

    patch_continue = generate_refactoring_patch([(0.95, u_continue, u_continue)], repo_root=str(tmp_path))
    assert patch_continue == ""

    # 2. Test prune_baseline sfp_to_clone and name-disambiguated ns_sfp_to_clones
    base_json = tmp_path / "baseline_multi.json"
    data = {
        "version": "1.3.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "target": ".",
        "threshold": 0.8,
        "clone_count": 2,
        "fingerprints": [
            {
                "file_a": "pkg/mod1.py",
                "name_a": "alpha",
                "hash_a": "1111222233334444",
                "file_b": "pkg/mod1.py",
                "name_b": "beta",
                "hash_b": "1111222233334444",
            },
            {
                "file_a": "pkg/mod2.py",
                "name_a": "gamma",
                "hash_a": "1111222233334444",
                "file_b": "pkg/mod2.py",
                "name_b": "delta",
                "hash_b": "1111222233334444",
            },
        ],
    }
    base_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Active clones: pair in mod1 was renamed from beta -> beta_renamed, while mod2 pair is active as-is
    u_m1_a = {"file": "pkg/mod1.py", "name": "alpha", "structural_hash": "1111222233334444"}
    u_m1_b = {"file": "pkg/mod1.py", "name": "beta_renamed", "structural_hash": "1111222233334444"}
    u_m2_a = {"file": "pkg/mod2.py", "name": "gamma", "structural_hash": "1111222233334444"}
    u_m2_b = {"file": "pkg/mod2.py", "name": "delta", "structural_hash": "1111222233334444"}

    active = [
        (0.95, u_m1_a, u_m1_b),
        (0.95, u_m2_a, u_m2_b),
    ]
    res = prune_baseline(str(base_json), active, unstaged_modified_ranges={})
    assert res.retained_count == 2
    assert res.pruned_count == 0

    saved_data = json.loads(base_json.read_text(encoding="utf-8"))
    recs = saved_data["fingerprints"]
    # mod1 record should be updated to beta_renamed, without adopting mod2's file path
    m1_rec = next(r for r in recs if r["file_a"] == "pkg/mod1.py")
    assert m1_rec["name_b"] == "beta_renamed"
    assert m1_rec["file_b"] == "pkg/mod1.py"

    m2_rec = next(r for r in recs if r["file_a"] == "pkg/mod2.py")
    assert m2_rec["name_b"] == "delta"
    assert m2_rec["file_b"] == "pkg/mod2.py"


def test_batch_65_git_sha256_diff_prefixes_and_quoted_toml_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies SHA-256 git blame parsing, git diff prefix determinism, and quote-aware TOML arrays."""
    from typing import Optional, Sequence  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.config import _parse_toml_array_value  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.coverage import compute_unit_coverage  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import (  # pylint: disable=import-outside-toplevel
        get_git_blame_info,
        get_git_modified_line_ranges,
    )

    # 1. Quotation-aware TOML array tokenization with internal commas
    arr_with_commas = '["path,with,comma", "simple", \'single,quote,comma\', true, false, 42, 3.14]'
    parsed = _parse_toml_array_value(arr_with_commas)
    assert parsed == [
        "path,with,comma",
        "simple",
        "single,quote,comma",
        True,
        False,
        42,
        3.14,
    ]

    # 2. Git blame parsing on modern SHA-256 (64 hex characters) repositories
    sha256_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    mock_blame = (
        f"{sha256_hash} 1 1 1\n"
        f"author Jane Doe\n"
        f"author-time 1700000000\n"
        f"summary Feature implementation\n"
        f"filename test.py\n"
        f"\tprint('hello')\n"
    )

    def mock_run_blame(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
        assert "blame" in args
        return mock_blame

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_blame)
    blame = get_git_blame_info("test.py", 1, 1)
    assert blame["author"] == "Jane Doe"
    assert blame["commit"] == sha256_hash[:8]
    assert blame["timestamp"] == 1700000000
    assert blame["summary"] == "Feature implementation"

    # 3. Deterministic diff prefixes passed to git diff
    captured_args: List[Sequence[str]] = []

    def mock_run_diff(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
        captured_args.append(args)
        return "--- a/test.py\n+++ b/test.py\n@@ -1,0 +1,5 @@\n"

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_diff)
    diff_ranges = get_git_modified_line_ranges(since_ref="HEAD~1")
    assert len(captured_args) == 1
    assert "--src-prefix=a/" in captured_args[0]
    assert "--dst-prefix=b/" in captured_args[0]
    assert "test.py" in diff_ranges

    # 4. Zero-allocation compute_unit_coverage calculation
    u_cov = {"file": "mod.py", "start": 10, "end": 14}  # 5 lines total
    cov_data = {"mod.py": {10, 11, 12}}  # 3 of 5 lines covered
    ratio = compute_unit_coverage(u_cov, cov_data)
    assert ratio == 0.6


def test_batch_66_artifact_dirs_clustering_helper_and_defensive_scoring(tmp_path: Path) -> None:
    """Batch 66: Verify artifact directory creation, clustering deduplication helper, and defensive scoring."""
    from pydoppelgangerhunt.cli import (  # pylint: disable=import-outside-toplevel
        _write_artifact_file,
        main as cli_main,
    )
    from pydoppelgangerhunt.clustering import (  # pylint: disable=import-outside-toplevel,protected-access
        _extract_unique_clusters,
    )
    from pydoppelgangerhunt.matcher import compute_priority_score  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.reporters import (  # pylint: disable=import-outside-toplevel
        emit_structured_report,
        format_markdown_summary,
    )

    # 1. _write_artifact_file creates nested missing parent directories
    nested_art = tmp_path / "nested" / "deep" / "summary.md"
    _write_artifact_file(str(nested_art), "# Title\n", "SUMMARY", verbose=False)
    assert nested_art.is_file()
    assert nested_art.read_text(encoding="utf-8") == "# Title\n"

    # 2. emit_structured_report creates nested missing parent directories
    nested_json = tmp_path / "build" / "reports" / "report.json"
    emit_structured_report({"test": True}, "JSON", str(nested_json))
    assert nested_json.is_file()
    assert '"test": true' in nested_json.read_text(encoding="utf-8")

    # 3. cli_main creates nested missing parent directories for --output
    py_file = tmp_path / "sample.py"
    py_file.write_text("def unique_one():\n    return 1\n", encoding="utf-8")
    nested_out = tmp_path / "out" / "cli" / "result.txt"
    code = cli_main([str(tmp_path), "--output", str(nested_out)])
    assert code == 0
    assert nested_out.is_file()
    assert "No structural code clones found" in nested_out.read_text(encoding="utf-8")

    # 4. _extract_unique_clusters dedupes and deterministically sorts clusters
    shared_c1 = {"node_b", "node_a"}
    shared_c2 = {"node_c"}
    c_map = {
        "node_a": shared_c1,
        "node_b": shared_c1,
        "node_c": shared_c2,
    }
    extracted = _extract_unique_clusters(c_map, ["node_a", "node_b", "node_c"])
    assert extracted == [["node_a", "node_b"], ["node_c"]]

    # 5. compute_priority_score defensive bounds on partial/missing unit keys
    u_partial1 = {"name": "partial1"}
    u_partial2 = {"name": "partial2"}
    score = compute_priority_score(0.95, u_partial1, u_partial2)
    assert score > 0.0
    assert isinstance(score, float)

    # 6. format_markdown_summary with empty stats and backticks / pipes in names
    empty_stats: Dict[str, Any] = {
        "package_sloc": {"pkg|sub`dir": 120},
    }
    md_summary = format_markdown_summary(empty_stats, target="target`pkg")
    assert "target'pkg" in md_summary
    assert "`target`pkg`" not in md_summary
    assert "pkg\\|sub'dir" in md_summary
    assert "Repository DRY Score" in md_summary


def test_batch_71_type2_clone_parameter_renaming_and_body_retention(tmp_path: Path) -> None:
    """Verifies that Type-2 clones with renamed parameters preserve full helper body and map arguments at call sites."""
    f = tmp_path / "calc.py"
    code = (
        "def calc_alpha(x: int) -> int:\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    d = c + 4\n"
        "    return d\n"
        "\n"
        "def calc_beta(y: int) -> int:\n"
        "    a = y * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    d = c + 4\n"
        "    return d\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {"file": str(f), "start": 1, "end": 6, "name": "calc_alpha", "kind": "function"}
    u2 = {"file": str(f), "start": 8, "end": 13, "name": "calc_beta", "kind": "function"}

    # 1. Helper synthesis must retain all statements despite variable renaming on line 1
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "a = x * 2" in helper
    assert "b = a + 1" in helper
    assert "c = b * 3" in helper
    assert "d = c + 4" in helper
    assert "return d" in helper

    # 2. Refactoring patch must delegate with x in calc_alpha and y in calc_beta
    patch = generate_refactoring_patch([(0.90, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_calc_alpha_calc_beta(x)" in patch
    assert "return _shared_calc_alpha_calc_beta(y)" in patch
    # Ensure calc_beta does not reference x
    beta_section = patch.split("def calc_beta(y: int) -> int:")[1]
    assert "(x)" not in beta_section
    assert "(y)" in beta_section

    # 3. Compound blocks with renamed outputs
    f_block = tmp_path / "block.py"
    b_code = (
        "def proc1(p: int) -> int:\n"
        "    v = p * 2\n"
        "    out1 = v + 10\n"
        "    return out1\n"
        "\n"
        "def proc2(q: int) -> int:\n"
        "    v = q * 2\n"
        "    out2 = v + 10\n"
        "    return out2\n"
    )
    f_block.write_text(b_code, encoding="utf-8")
    u_b1 = {"file": str(f_block), "start": 2, "end": 3, "name": "b1:1", "kind": "compound_block"}
    u_b2 = {"file": str(f_block), "start": 7, "end": 8, "name": "b2:1", "kind": "compound_block"}
    patch_block = generate_refactoring_patch([(0.90, u_b1, u_b2)], repo_root=str(tmp_path), replace_clones=True)
    assert "v, out1 = _shared_b1_b2(p)" in patch_block
    assert "v, out2 = _shared_b1_b2(q)" in patch_block


def test_generate_refactoring_patch_multi_clone_same_file_git_apply(tmp_path: Path) -> None:
    """Verifies that multiple clone pairs in the same file generate a single unified diff that applies cleanly via git apply."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "def alpha(x: int) -> int:\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    return b\n\n"
        "def beta(y: int) -> int:\n"
        "    a = y * 2\n"
        "    b = a + 1\n"
        "    return b\n\n"
        "def gamma(z: int) -> int:\n"
        "    c = z * 3\n"
        "    d = c + 5\n"
        "    return d\n\n"
        "def delta(w: int) -> int:\n"
        "    c = w * 3\n"
        "    d = c + 5\n"
        "    return d\n"
    )
    mod_file = tmp_path / "multi_mod.py"
    mod_file.write_text(src, encoding="utf-8")

    u_alpha = {"name": "alpha", "file": "multi_mod.py", "start": 1, "end": 4, "kind": "function", "receiver_kind": "none"}
    u_beta = {"name": "beta", "file": "multi_mod.py", "start": 6, "end": 9, "kind": "function", "receiver_kind": "none"}
    u_gamma = {"name": "gamma", "file": "multi_mod.py", "start": 11, "end": 14, "kind": "function", "receiver_kind": "none"}
    u_delta = {"name": "delta", "file": "multi_mod.py", "start": 16, "end": 19, "kind": "function", "receiver_kind": "none"}

    patch = generate_refactoring_patch(
        [(0.95, u_alpha, u_beta), (0.95, u_gamma, u_delta)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    # 1. Exactly one unified diff header for multi_mod.py
    assert patch.count("--- a/multi_mod.py") == 1
    assert patch.count("+++ b/multi_mod.py") == 1
    # 2. Both clone pairs reflected and helpers synthesized
    assert "# Clone Pair (95.0%): multi_mod.py <===> multi_mod.py" in patch
    assert "def _shared_alpha_beta(x: int) -> int:" in patch
    assert "def _shared_gamma_delta(z: int) -> int:" in patch
    assert "return _shared_alpha_beta(x)" in patch
    assert "return _shared_gamma_delta(z)" in patch

    # 3. Verify clean git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    # Verify execution
    run_proc = subprocess.run(
        ["python", "-c", "from multi_mod import alpha, beta, gamma, delta; print(alpha(3), beta(3), gamma(2), delta(2))"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "7 7 11 11"


def test_generate_refactoring_patch_cross_module_with_import_and_execution(tmp_path: Path) -> None:
    """Verifies that cross-module clone refactoring injects the import into file 2 and both run."""
    import subprocess  # pylint: disable=import-outside-toplevel

    pkg_dir = tmp_path / "service_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "def compute_alpha(x: int) -> int:\n"
        "    step1 = x * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    src_b = (
        "def compute_beta(v: int) -> int:\n"
        "    step1 = v * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    f_a = pkg_dir / "srv_a.py"
    f_b = pkg_dir / "srv_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_alpha", "file": "service_pkg/srv_a.py", "start": 1, "end": 4, "kind": "function"}
    u_b = {"name": "compute_beta", "file": "service_pkg/srv_b.py", "start": 1, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    # 1. Diff includes both files
    assert "--- a/service_pkg/srv_a.py" in patch
    assert "--- a/service_pkg/srv_b.py" in patch
    # 2. File 2 imports helper from file 1
    assert "from service_pkg.srv_a import _shared_compute_alpha_compute_beta" in patch

    # 3. Verify git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        ["python", "-c", "from service_pkg.srv_a import compute_alpha; from service_pkg.srv_b import compute_beta; print(compute_alpha(5), compute_beta(5))"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "57 57"


def test_generate_refactoring_patch_cross_module_circular_import_safety(tmp_path: Path) -> None:
    """Verifies that circular module imports are detected and prevented during cross-module patch generation."""
    pkg_dir = tmp_path / "cyclic_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "import cyclic_pkg.mod_b\n\n"
        "def work_a(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    src_b = (
        "def work_b(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    f_a = pkg_dir / "mod_a.py"
    f_b = pkg_dir / "mod_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "work_a", "file": "cyclic_pkg/mod_a.py", "start": 3, "end": 5, "kind": "function"}
    u_b = {"name": "work_b", "file": "cyclic_pkg/mod_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    # File 2 is NOT refactored with an unsafe circular import
    assert "--- a/cyclic_pkg/mod_b.py" not in patch
    assert "Circular import or unresolvable module path" in patch


def test_generate_refactoring_patch_overlapping_units_filtered(tmp_path: Path) -> None:
    """Verifies that overlapping candidate units are filtered without crashing ValueError."""
    src = (
        "def outer(x: int) -> int:\n"
        "    y = x * 2\n"
        "    z = y + 1\n"
        "    return z\n"
    )
    f = tmp_path / "overlap.py"
    f.write_text(src, encoding="utf-8")

    u_outer = {"name": "outer", "file": "overlap.py", "start": 1, "end": 4, "kind": "function"}
    u_inner = {"name": "outer:inner", "file": "overlap.py", "start": 2, "end": 3, "kind": "compound_block"}

    # Pair proposing outer and inner concurrently
    patch = generate_refactoring_patch(
        [(0.90, u_outer, u_outer), (0.85, u_inner, u_inner)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def outer(x: int) -> int:" in patch


def test_multiline_comprehension_helper_synthesis_and_apply(tmp_path: Path) -> None:
    """Verifies that multiline comprehensions with comments are wrapped in return (...) and run cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "def parse_items(items: list) -> list:\n"
        "    return [\n"
        "        # double item value\n"
        "        x * 2\n"
        "        for x in items\n"
        "        if x > 0\n"
        "    ]\n"
        "\n"
        "def process_items(elements: list) -> list:\n"
        "    return [\n"
        "        # double item value\n"
        "        e * 2\n"
        "        for e in elements\n"
        "        if e > 0\n"
        "    ]\n"
    )
    f = tmp_path / "comp.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "parse_items:listcomp",
        "file": "comp.py",
        "start": 2,
        "end": 7,
        "start_col": 11,
        "end_col": 5,
        "kind": "comprehension",
    }
    u2 = {
        "name": "process_items:listcomp",
        "file": "comp.py",
        "start": 10,
        "end": 15,
        "start_col": 11,
        "end_col": 5,
        "kind": "comprehension",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "return (" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from comp import parse_items, process_items; "
            "print(parse_items([1, -1, 3]), process_items([2, -5, 4]))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "[2, 6] [4, 8]"


def test_subroutine_nonterminal_embedded_return_hazard_rejected(tmp_path: Path) -> None:
    """Verifies that subroutine compound blocks with non-terminal returns are rejected as hazards."""
    src = (
        "def func_a(x: int) -> int:\n"
        "    if x < 0:\n"
        "        return 0\n"
        "    y = x * 2\n"
        "    return y\n"
        "\n"
        "def func_b(x: int) -> int:\n"
        "    if x < 0:\n"
        "        return 0\n"
        "    y = x * 2\n"
        "    return y\n"
    )
    f = tmp_path / "hazard.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "func_a:blk", "file": "hazard.py", "start": 2, "end": 4, "kind": "compound_block"}
    u2 = {"name": "func_b:blk", "file": "hazard.py", "start": 8, "end": 10, "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_init_method_delegation_suppresses_return(tmp_path: Path) -> None:
    """Verifies that __init__ method refactoring never emits return in delegation calls."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class ModelA:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.val = val * 10\n"
        "        self.active = True\n"
        "\n"
        "class ModelB:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.val = val * 10\n"
        "        self.active = True\n"
    )
    f = tmp_path / "models.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "__init__", "file": "models.py", "start": 2, "end": 4, "kind": "function", "enclosing_class": "ModelA"}
    u2 = {"name": "__init__", "file": "models.py", "start": 7, "end": 9, "kind": "function", "enclosing_class": "ModelB"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    lines = [
        line for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    init_delegations = [line for line in lines if "_shared_init" in line]
    assert len(init_delegations) >= 2
    for line in init_delegations:
        assert "return " not in line

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from models import ModelA, ModelB; "
            "a = ModelA(5); b = ModelB(7); "
            "print(a.val, a.active, b.val, b.active)",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "50 True 70 True"


def test_multi_class_method_helper_line_adjustment(tmp_path: Path) -> None:
    """Verifies that line shifts from earlier method refactorings do not corrupt helper insertion in later classes."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class ServiceFirst:\n"
        "    def action_one(self, x: int) -> int:\n"
        "        a = x * 2\n"
        "        b = a + 3\n"
        "        return b\n"
        "\n"
        "    def action_two(self, x: int) -> int:\n"
        "        a = x * 2\n"
        "        b = a + 3\n"
        "        return b\n"
        "\n"
        "class ServiceSecond:\n"
        "    def execute_one(self, y: int) -> int:\n"
        "        m = y * 5\n"
        "        n = m + 7\n"
        "        return n\n"
        "\n"
        "    def execute_two(self, y: int) -> int:\n"
        "        m = y * 5\n"
        "        n = m + 7\n"
        "        return n\n"
    )
    f = tmp_path / "services.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "action_one", "file": "services.py", "start": 2, "end": 5, "kind": "function", "enclosing_class": "ServiceFirst"}
    u2 = {"name": "action_two", "file": "services.py", "start": 7, "end": 10, "kind": "function", "enclosing_class": "ServiceFirst"}
    u3 = {"name": "execute_one", "file": "services.py", "start": 13, "end": 16, "kind": "function", "enclosing_class": "ServiceSecond"}
    u4 = {"name": "execute_two", "file": "services.py", "start": 18, "end": 21, "kind": "function", "enclosing_class": "ServiceSecond"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2), (1.0, u3, u4)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from services import ServiceFirst, ServiceSecond; "
            "s1 = ServiceFirst(); s2 = ServiceSecond(); "
            "print(s1.action_one(4), s2.execute_one(3))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "11 22"


def test_type2_arity_mismatch_skips_delegation(tmp_path: Path) -> None:
    """Verifies that clone pairs with differing input or output variable arities skip replacement."""
    src = (
        "def compute_first(a: int, b: int) -> int:\n"
        "    res = a * b + 1\n"
        "    return res\n"
        "\n"
        "def compute_second(a: int) -> int:\n"
        "    res = a * 10 + 1\n"
        "    return res\n"
    )
    f = tmp_path / "arity.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "compute_first", "file": "arity.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_second", "file": "arity.py", "start": 5, "end": 7, "kind": "function"}

    real_scope = analyze_unit_variable_scope

    def fake_scope(unit: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        res = real_scope(unit, *args, **kwargs)
        if unit.get("name") == "compute_second":
            res["inputs"] = ["a"]
        return res

    with mock.patch("pydoppelgangerhunt.fixer.analyze_unit_variable_scope", side_effect=fake_scope):
        patch = generate_refactoring_patch(
            [(0.90, u1, u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
        assert patch == ""


def test_cross_module_helper_name_collision_deduplication(tmp_path: Path) -> None:
    """Verifies that if File 2 already defines the candidate helper name, an indexed suffix is generated."""
    pkg = tmp_path / "coll_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def handle_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    src2 = (
        "_shared_handle_data_process_data = 'existing_symbol'\n"
        "\n"
        "def process_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    (pkg / "f1.py").write_text(src1, encoding="utf-8")
    (pkg / "f2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "handle_data", "file": "coll_pkg/f1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "process_data", "file": "coll_pkg/f2.py", "start": 3, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "_shared_handle_data_process_data_2" in patch
    assert "from coll_pkg.f1 import _shared_handle_data_process_data_2" in patch


def test_super_call_rejected_for_module_binding(tmp_path: Path) -> None:
    """Verifies that methods containing super() are rejected when effective binding is module."""
    src = (
        "class Base:\n"
        "    def run(self) -> str:\n"
        "        return 'base'\n"
        "\n"
        "class ChildA(Base):\n"
        "    def execute(self) -> str:\n"
        "        val = super().run() + '_a'\n"
        "        return val\n"
        "\n"
        "class ChildB(Base):\n"
        "    def execute(self) -> str:\n"
        "        val = super().run() + '_b'\n"
        "        return val\n"
    )
    f = tmp_path / "super_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "execute", "file": "super_mod.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "ChildA"}
    u2 = {"name": "execute", "file": "super_mod.py", "start": 11, "end": 13, "kind": "function", "enclosing_class": "ChildB"}

    # Cross-class clones resolve to module binding -> should reject
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

    # Explicit module binding on same class -> should also reject
    u3 = {"name": "execute", "file": "super_mod.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "ChildA"}
    patch_module = generate_refactoring_patch(
        [(1.0, u1, u3)],
        repo_root=str(tmp_path),
        method_binding="module",
        replace_clones=True,
    )
    assert patch_module == ""


def test_super_call_accepted_for_same_class_method_binding(tmp_path: Path) -> None:
    """Verifies that methods containing super() within the same class generate a valid class helper method."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class BaseHandler:\n"
        "    def perform(self, x: int) -> int:\n"
        "        return x * 10\n"
        "\n"
        "class CustomHandler(BaseHandler):\n"
        "    def handle_primary(self, x: int) -> int:\n"
        "        base_val = super().perform(x)\n"
        "        return base_val + 5\n"
        "\n"
        "    def handle_secondary(self, x: int) -> int:\n"
        "        base_val = super().perform(x)\n"
        "        return base_val + 5\n"
    )
    f = tmp_path / "handlers.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "handle_primary", "file": "handlers.py", "start": 6, "end": 8, "kind": "function", "enclosing_class": "CustomHandler"}
    u2 = {"name": "handle_secondary", "file": "handlers.py", "start": 10, "end": 12, "kind": "function", "enclosing_class": "CustomHandler"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "def _shared_handle_primary" in patch
    assert "super().perform(x)" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from handlers import CustomHandler; "
            "h = CustomHandler(); "
            "print(h.handle_primary(3), h.handle_secondary(7))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "35 75"


def test_cross_module_global_rejected(tmp_path: Path) -> None:
    """Verifies that clone pairs using global variables across different files are rejected."""
    pkg = tmp_path / "state_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "COUNTER = 0\n"
        "\n"
        "def increment_counter() -> int:\n"
        "    global COUNTER\n"
        "    COUNTER += 1\n"
        "    return COUNTER\n"
    )
    src2 = (
        "COUNTER = 100\n"
        "\n"
        "def advance_counter() -> int:\n"
        "    global COUNTER\n"
        "    COUNTER += 1\n"
        "    return COUNTER\n"
    )
    (pkg / "mod_a.py").write_text(src1, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "increment_counter", "file": "state_pkg/mod_a.py", "start": 3, "end": 6, "kind": "function"}
    u2 = {"name": "advance_counter", "file": "state_pkg/mod_b.py", "start": 3, "end": 6, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_receiver_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing different receiver attributes are rejected."""
    src = (
        "class StateManager:\n"
        "    def update_alpha(self, val: int) -> int:\n"
        "        self.alpha = val * 2\n"
        "        return self.alpha\n"
        "\n"
        "    def update_beta(self, val: int) -> int:\n"
        "        self.beta = val * 2\n"
        "        return self.beta\n"
    )
    f = tmp_path / "state_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "update_alpha",
        "file": "state_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "StateManager",
    }
    u2 = {
        "name": "update_beta",
        "file": "state_mismatch.py",
        "start": 6,
        "end": 8,
        "kind": "function",
        "enclosing_class": "StateManager",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_receiver_chained_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing different chained receiver attributes are rejected."""
    src = (
        "class ServiceDriver:\n"
        "    def configure_timeout(self, setting: int) -> int:\n"
        "        self.config.timeout = setting\n"
        "        return self.config.timeout\n"
        "\n"
        "    def configure_retries(self, setting: int) -> int:\n"
        "        self.config.retries = setting\n"
        "        return self.config.retries\n"
    )
    f = tmp_path / "service_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "configure_timeout",
        "file": "service_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ServiceDriver",
    }
    u2 = {
        "name": "configure_retries",
        "file": "service_mismatch.py",
        "start": 6,
        "end": 8,
        "kind": "function",
        "enclosing_class": "ServiceDriver",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_receiver_cross_class_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that cross-class method clones accessing different attributes are rejected."""
    src = (
        "class ModelA:\n"
        "    def fetch(self) -> int:\n"
        "        res = self.primary_data + 1\n"
        "        return res\n"
        "\n"
        "class ModelB:\n"
        "    def fetch(self) -> int:\n"
        "        res = self.secondary_data + 1\n"
        "        return res\n"
    )
    f = tmp_path / "models_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "fetch",
        "file": "models_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ModelA",
    }
    u2 = {
        "name": "fetch",
        "file": "models_mismatch.py",
        "start": 7,
        "end": 9,
        "kind": "function",
        "enclosing_class": "ModelB",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_receiver_identical_attributes_accepted_and_executed(tmp_path: Path) -> None:
    """Verifies that methods accessing identical receiver attributes are refactored cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class Ledger:\n"
        "    def __init__(self, initial: int) -> None:\n"
        "        self.balance: int = initial\n"
        "\n"
        "    def credit_direct(self, amount: int) -> int:\n"
        "        self.balance += amount\n"
        "        return self.balance\n"
        "\n"
        "    def credit_wire(self, amount: int) -> int:\n"
        "        self.balance += amount\n"
        "        return self.balance\n"
    )
    f = tmp_path / "ledger.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "credit_direct",
        "file": "ledger.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "Ledger",
    }
    u2 = {
        "name": "credit_wire",
        "file": "ledger.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "Ledger",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "_shared_credit_direct_credit_wire(self, amount: int) -> int:" in helper
    assert "self.balance += amount" in helper

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "return self._shared_credit_direct_credit_wire(amount)" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ledger import Ledger; "
            "led = Ledger(100); "
            "r1 = led.credit_direct(50); "
            "r2 = led.credit_wire(25); "
            "print(r1, r2, led.balance)",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "150 175 175"


def test_mangled_private_attribute_module_binding_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing mangled private attributes reject module-level binding."""
    src = (
        "class Vault:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.__key = key\n"
        "\n"
        "    def access_alpha(self) -> str:\n"
        "        tag = 'v1_'\n"
        "        return tag + self.__key\n"
        "\n"
        "    def access_beta(self) -> str:\n"
        "        tag = 'v1_'\n"
        "        return tag + self.__key\n"
    )
    f = tmp_path / "vault_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "access_alpha",
        "file": "vault_mod.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "Vault",
    }
    u2 = {
        "name": "access_beta",
        "file": "vault_mod.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "Vault",
    }

    helper = synthesize_shared_helper_code(u1, u2, method_binding="module", repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="module",
        replace_clones=True,
    )
    assert patch == ""


def test_mangled_private_attribute_cross_class_rejected(tmp_path: Path) -> None:
    """Verifies that cross-class clones accessing mangled private attributes are rejected."""
    src = (
        "class NodeAlpha:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.__val = val\n"
        "    def get_metric(self) -> int:\n"
        "        scale = 10\n"
        "        return self.__val * scale\n"
        "\n"
        "class NodeBeta:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.__val = val\n"
        "    def get_metric(self) -> int:\n"
        "        scale = 10\n"
        "        return self.__val * scale\n"
    )
    f = tmp_path / "cross_vault.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "get_metric",
        "file": "cross_vault.py",
        "start": 4,
        "end": 6,
        "kind": "function",
        "enclosing_class": "NodeAlpha",
    }
    u2 = {
        "name": "get_metric",
        "file": "cross_vault.py",
        "start": 11,
        "end": 13,
        "kind": "function",
        "enclosing_class": "NodeBeta",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""


def test_mangled_private_attribute_same_class_method_binding_accepted_and_executed(
    tmp_path: Path,
) -> None:
    """Verifies that methods accessing mangled attributes within the same class refactor and execute cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class SecretHolder:\n"
        "    def __init__(self, token: str) -> None:\n"
        "        self.__token = token\n"
        "\n"
        "    def reveal_first(self) -> str:\n"
        "        prefix = 'token:'\n"
        "        return prefix + self.__token\n"
        "\n"
        "    def reveal_second(self) -> str:\n"
        "        prefix = 'token:'\n"
        "        return prefix + self.__token\n"
    )
    f = tmp_path / "secret_holder.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "reveal_first",
        "file": "secret_holder.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "SecretHolder",
    }
    u2 = {
        "name": "reveal_second",
        "file": "secret_holder.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "SecretHolder",
    }

    helper = synthesize_shared_helper_code(u1, u2, method_binding="method", repo_root=str(tmp_path))
    assert "def _shared_reveal_first_reveal_second(self) -> str:" in helper
    assert "return prefix + self.__token" in helper

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "return self._shared_reveal_first_reveal_second()" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from secret_holder import SecretHolder; "
            "h = SecretHolder('super_secret_xyz'); "
            "print(h.reveal_first(), h.reveal_second())",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "token:super_secret_xyz token:super_secret_xyz"


def test_init_colon_notation_no_return_delegation(tmp_path: Path) -> None:
    """Verifies that __init__ methods with colon-prefixed names delegate without return statements."""
    src = (
        "class ConfigRecord:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.key = key\n"
        "        self.active = True\n"
        "\n"
        "class DataRecord:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.key = key\n"
        "        self.active = True\n"
    )
    f = tmp_path / "records.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "ConfigRecord:__init__",
        "file": "records.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ConfigRecord",
    }
    u2 = {
        "name": "DataRecord:__init__",
        "file": "records.py",
        "start": 7,
        "end": 9,
        "kind": "function",
        "enclosing_class": "DataRecord",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch != ""
    assert "return _shared" not in patch
    assert "_shared_ConfigRecord_DataRecord(self, key)" in patch


def test_partially_overlapping_inputs_preserves_full_signature(tmp_path: Path) -> None:
    """Verifies that Type-2 clone pairs with partially overlapping parameter names preserve all inputs."""
    code1 = (
        "def compute_delta(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def compute_delta_v2(a: int, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "calc1.py"
    f2 = tmp_path / "calc2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "calc1.py", "start": 1, "end": 2, "name": "compute_delta", "kind": "function"}
    u2 = {"file": "calc2.py", "start": 1, "end": 2, "name": "compute_delta_v2", "kind": "function"}

    scope = analyze_unit_variable_scope(u1, u2, repo_root=str(tmp_path))
    assert scope["inputs"] == ["x", "y", "timeout"]

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_compute_delta_compute_delta_v2(x: int, y: int, timeout: float = 1.0) -> int:" in helper
    assert "return x * 2 + y" in helper

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_compute_delta_compute_delta_v2(x, y, timeout)" in patch
    assert "return _shared_compute_delta_compute_delta_v2(a, b, timeout)" in patch


def test_partially_overlapping_inputs_subprocess_execution(tmp_path: Path) -> None:
    """Verifies runtime correctness of refactored Type-2 clones with partially overlapping parameters."""
    code1 = (
        "def eval_math(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def eval_math_alt(a: int, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "math1.py"
    f2 = tmp_path / "math2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "math1.py", "start": 1, "end": 2, "name": "eval_math", "kind": "function"}
    u2 = {"file": "math2.py", "start": 1, "end": 2, "name": "eval_math_alt", "kind": "function"}

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""

    patch_file = tmp_path / "refactor.patch"
    patch_file.write_text(patch, encoding="utf-8")

    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_file)],
        cwd=tmp_path,
        check=True,
    )

    runner_code = (
        "from math1 import eval_math\n"
        "from math2 import eval_math_alt\n"
        "r1 = eval_math(3, 4)\n"
        "r2 = eval_math_alt(5, 6)\n"
        "assert r1 == 10, f'Expected 10, got {r1}'\n"
        "assert r2 == 16, f'Expected 16, got {r2}'\n"
        "print('PARTIAL_INPUTS_SUCCESS')\n"
    )
    runner_file = tmp_path / "run_test.py"
    runner_file.write_text(runner_code, encoding="utf-8")

    res = subprocess.run(
        [sys.executable, str(runner_file)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Execution failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "PARTIAL_INPUTS_SUCCESS" in res.stdout


def test_type_merge_positional_alignment_renamed_parameters(tmp_path: Path) -> None:
    """Verifies that type annotations merge positionally across renamed parameter names in Type-2 clones."""
    code1 = (
        "def process_val(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def process_val_v2(a: str, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "proc1.py"
    f2 = tmp_path / "proc2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "proc1.py", "start": 1, "end": 2, "name": "process_val", "kind": "function"}
    u2 = {"file": "proc2.py", "start": 1, "end": 2, "name": "process_val_v2", "kind": "function"}

    helper = synthesize_shared_helper_code(
        u1, u2, repo_root=str(tmp_path), type_merge_strategy="union"
    )
    assert "def _shared_process_val_process_val_v2(x: Union[int, str], y: int, timeout: float = 1.0) -> int:" in helper


def test_insert_imports_into_module_deduplicates_within_import_lines() -> None:
    """Verifies that _insert_imports_into_module eliminates duplicate imports present in import_lines."""
    from pydoppelgangerhunt.fixer import _insert_imports_into_module  # pylint: disable=import-outside-toplevel

    orig = ["def foo(): pass\n"]
    imports = ["from typing import Any", "from typing import Any", "from typing import Tuple"]
    res = _insert_imports_into_module(orig, imports)
    assert res.count("from typing import Any\n") == 1
    assert res.count("from typing import Tuple\n") == 1


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_is_irrefutable_pattern_and_case() -> None:
    """Verifies that _is_irrefutable_case identifies wildcard, as, and alternative irrefutable patterns."""
    from pydoppelgangerhunt.fixer import _is_irrefutable_case  # pylint: disable=import-outside-toplevel

    code = (
        "match x:\n"
        "    case _:\n"
        "        pass\n"
        "    case val:\n"
        "        pass\n"
        "    case _ as y:\n"
        "        pass\n"
        "    case 1 | _:\n"
        "        pass\n"
        "    case (1 | _) as y:\n"
        "        pass\n"
        "    case 1 as y:\n"
        "        pass\n"
        "    case 1 | 2:\n"
        "        pass\n"
        "    case _ if False:\n"
        "        pass\n"
        "    case [a, b]:\n"
        "        pass\n"
    )
    tree = ast.parse(code)
    cases = tree.body[0].cases  # type: ignore[attr-defined]

    # case _:
    assert _is_irrefutable_case(cases[0]) is True
    # case val:
    assert _is_irrefutable_case(cases[1]) is True
    # case _ as y:
    assert _is_irrefutable_case(cases[2]) is True
    # case 1 | _:
    assert _is_irrefutable_case(cases[3]) is True
    # case (1 | _) as y:
    assert _is_irrefutable_case(cases[4]) is True
    # case 1 as y:
    assert _is_irrefutable_case(cases[5]) is False
    # case 1 | 2:
    assert _is_irrefutable_case(cases[6]) is False
    # case _ if False:
    assert _is_irrefutable_case(cases[7]) is False
    # case [a, b]:
    assert _is_irrefutable_case(cases[8]) is False


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_pattern_stores_deletion_and_as_pattern_definite_assignment() -> None:
    """Verifies that variable deletions inside match bodies discard pattern-bound stores, and as-patterns are definite."""
    code = (
        "match x:\n"
        "    case (a, b):\n"
        "        del a\n"
        "        res = 1\n"
        "    case _ as fallback:\n"
        "        res = 2\n"
    )
    tree = ast.parse(code)
    definite, conditional = _analyze_block_assignment(tree.body)

    # res is assigned in all non-terminating branches of an irrefutable match
    assert "res" in definite
    assert "res" not in conditional

    # a was bound by pattern but deleted in case 1 body
    assert "a" not in definite
    assert "a" not in conditional

    # b and fallback were conditionally bound in respective branches
    assert "b" in conditional
    assert "b" not in definite
    assert "fallback" in conditional
    assert "fallback" not in definite


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_guard_walrus_definite_in_case_body() -> None:
    """Verifies that walrus variables defined in match guards are recognized as definite within the case body."""
    code = (
        "match x:\n"
        "    case items if (n := len(items)) > 0:\n"
        "        del n\n"
        "        res = 1\n"
        "    case _:\n"
        "        res = 2\n"
    )
    tree = ast.parse(code)
    definite, conditional = _analyze_block_assignment(tree.body)

    # res is definitely assigned across the whole match
    assert "res" in definite
    # n was walrus-assigned in the guard and deleted inside the case body, so it should not be definite or conditional
    assert "n" not in definite
    assert "n" not in conditional


def test_extract_unit_body_lines_and_scope_method_kind(tmp_path: Path) -> None:
    """Verifies that units with kind='method' strip headers/docstrings and preserve method parameters."""
    from pydoppelgangerhunt.fixer import _extract_unit_body_lines  # pylint: disable=import-outside-toplevel

    code = (
        "class Calculator:\n"
        "    def compute(self, a: int, b: int) -> int:\n"
        "        '''Compute sum.'''\n"
        "        total = a + b\n"
        "        return total\n"
    )
    f = tmp_path / "calc.py"
    f.write_text(code, encoding="utf-8")

    unit = {
        "file": str(f),
        "start": 2,
        "end": 5,
        "name": "Calculator:compute",
        "kind": "method",
    }
    raw_lines = code.splitlines()[1:]
    body_lines = _extract_unit_body_lines(unit, raw_lines)
    assert body_lines == ["total = a + b", "return total"]

    scope = analyze_unit_variable_scope(unit, repo_root=str(tmp_path))
    assert "self" in scope["inputs"]
    assert "a" in scope["inputs"]
    assert "b" in scope["inputs"]
    assert "total" in scope["outputs"]
    assert scope["return_type"] == "int"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_refactoring_subprocess_execution(tmp_path: Path) -> None:
    """End-to-end integration test verifying that refactored pattern matching methods execute cleanly via subprocess."""
    code1 = (
        "def evaluate_code_v1(payload: dict) -> str:\n"
        "    match payload:\n"
        "        case {'status': 'success', 'code': c}:\n"
        "            res = f'OK:{c}'\n"
        "        case _ as fallback:\n"
        "            res = f'FAIL:{fallback.get(\"status\")}'\n"
        "    return res\n"
    )
    code2 = (
        "def evaluate_code_v2(payload: dict) -> str:\n"
        "    match payload:\n"
        "        case {'status': 'success', 'code': c}:\n"
        "            res = f'OK:{c}'\n"
        "        case _ as fallback:\n"
        "            res = f'FAIL:{fallback.get(\"status\")}'\n"
        "    return res\n"
    )
    entry_script = (
        "from mod1 import evaluate_code_v1\n"
        "from mod2 import evaluate_code_v2\n"
        "assert evaluate_code_v1({'status': 'success', 'code': 200}) == 'OK:200'\n"
        "assert evaluate_code_v1({'status': 'error'}) == 'FAIL:error'\n"
        "assert evaluate_code_v2({'status': 'success', 'code': 200}) == 'OK:200'\n"
        "assert evaluate_code_v2({'status': 'error'}) == 'FAIL:error'\n"
        "print('ALL_EVAL_TESTS_PASSED')\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    main_py = tmp_path / "run_eval.py"

    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    main_py.write_text(entry_script, encoding="utf-8")

    u1 = {"file": "mod1.py", "start": 1, "end": 7, "name": "evaluate_code_v1", "kind": "function"}
    u2 = {"file": "mod2.py", "start": 1, "end": 7, "name": "evaluate_code_v2", "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch != ""

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    proc = subprocess.run(
        [sys.executable, str(main_py)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ALL_EVAL_TESTS_PASSED" in proc.stdout


def test_find_module_helper_insertion_index_with_try_except_and_conditional_imports() -> None:
    """Verifies that _find_module_helper_insertion_index places helpers after try/except and conditional imports."""
    from pydoppelgangerhunt.fixer import _find_module_helper_insertion_index  # pylint: disable=import-outside-toplevel

    lines = [
        '"""Module docstring."""\n',
        "import os\n",
        "try:\n",
        "    import tomllib\n",
        "except ImportError:\n",
        "    import tomli as tomllib\n",
        "import sys\n",
        "\n",
        "def existing_fn():\n",
        "    pass\n",
    ]
    idx = _find_module_helper_insertion_index(lines)
    # The last import is 'import sys' at line 7 (1-indexed). The helper should be inserted at index 7 (line 7), before line 9.
    assert idx == 7


def test_infer_helper_return_type_generator_literal_and_explicit_types() -> None:
    """Verifies that _infer_helper_return_type infers literal yield types and preserves explicit annotations."""
    from pydoppelgangerhunt.fixer import _infer_helper_return_type  # pylint: disable=import-outside-toplevel

    # 1. Sync generator with literal integer yield
    scope_int = {"has_yield": True, "yield_expr_names": [("yield", ":literal:int")]}
    res_int = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope_int,
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_int == "Iterator[int]"

    # 2. Async generator with literal string yield
    scope_str = {"has_yield": True, "yield_expr_names": [("yield", ":literal:str")]}
    res_str = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope_str,
        meta1={},
        meta2={},
        is_async=True,
    )
    assert res_str == "AsyncIterator[str]"

    # 3. Explicit return type preserved for async generator
    res_explicit = _infer_helper_return_type(
        resolved_ret="AsyncIterator[float]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": []},
        meta1={},
        meta2={},
        is_async=True,
    )
    assert res_explicit == "AsyncIterator[float]"


def test_generator_helper_synthesis_literal_yields(tmp_path: Path) -> None:
    """Verifies that synthesizing helpers from generator units infers Iterator types and executes cleanly."""
    code1 = (
        "def num_gen1(limit: int):\n"
        "    for i in range(limit):\n"
        "        yield 42\n"
    )
    code2 = (
        "def num_gen2(limit: int):\n"
        "    for i in range(limit):\n"
        "        yield 42\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "g1.py", "start": 1, "end": 3, "name": "num_gen1", "kind": "function"}
    u2 = {"file": "g2.py", "start": 1, "end": 3, "name": "num_gen2", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "-> Iterator[int]:" in helper


def test_is_docstring_node_and_extract_docstring_end_line() -> None:
    """Verifies that _is_docstring_node and _extract_docstring_end_line accurately detect docstrings."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        _extract_docstring_end_line,
        _is_docstring_node,
    )

    # 1. Non-docstrings: None, Call, Constant int
    assert not _is_docstring_node(None)
    assert not _is_docstring_node(ast.parse("pass").body[0])
    assert not _is_docstring_node(ast.parse("42").body[0])
    assert not _is_docstring_node(ast.parse("print('hi')").body[0])

    # 2. String constant docstring node
    doc_node = ast.parse("'''docstring'''").body[0]
    assert _is_docstring_node(doc_node)

    # 3. _extract_docstring_end_line on module and function with multiline docstring
    tree_multiline = ast.parse('"""Line 1\nLine 2\nLine 3"""\nx = 1\n')
    assert _extract_docstring_end_line(tree_multiline) == 3

    # 4. _extract_docstring_end_line with no docstring
    tree_no_doc = ast.parse("x = 1\ny = 2\n")
    assert _extract_docstring_end_line(tree_no_doc) == 0

    # 5. _extract_docstring_end_line on a function node
    fn_tree = ast.parse("def f():\n    '''Function docstring.'''\n    return 10\n")
    assert _extract_docstring_end_line(fn_tree.body[0]) == 2

