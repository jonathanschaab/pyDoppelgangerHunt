"""Unit tests for reporters and diff."""

from __future__ import annotations

import ast
import json
from pathlib import Path
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
    scan_target,
    supports_color,
    synthesize_refactoring_suggestion,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _analyze_block_assignment,
    _build_whole_method_delegation,
    _detect_indent_step,
    _extract_required_typing_imports,
    _find_module_helper_insertion_index,
    _insert_imports_into_module,
    _inspect_unit_scope,
    _is_method_of_class,
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
    assert "def _shared_build(cls: Any, tag: str = 'item') -> str:" in helper_cm

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
    # The return-tuple generator must not return globals or nonlocals; scope modifiers must be preserved
    assert "global global_metric" in helper_code
    assert "nonlocal captured_scale" in helper_code
    assert "return" not in helper_code


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
















