"""Unit tests for reporters and diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest import mock
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
    _extract_required_typing_imports,
    _insert_imports_into_module,
)


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
    assert callable(pydoppelgangerhunt.prune_baseline)
    assert isinstance(pydoppelgangerhunt.DEFAULT_STOP_SHINGLES, set)
    assert callable(pydoppelgangerhunt.get_boilerplate_stop_shingles)
    assert callable(pydoppelgangerhunt.compute_unit_diff_overlap)
    assert pydoppelgangerhunt.MAJOR_POLICY_THRESHOLD == 0.50
    assert pydoppelgangerhunt.NEW_POLICY_THRESHOLD == 0.80



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


