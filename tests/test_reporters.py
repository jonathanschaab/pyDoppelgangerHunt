"""Unit tests for report generation (SARIF, JSON, HTML, Markdown, and ANSI color)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from pydoppelgangerhunt import (
    cluster_clone_families,
    colorize,
    compute_medoid,
    compute_repository_dry_stats,
    extract_unit_source_code,
    format_github_annotations,
    format_json_report,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    supports_color,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _build_whole_method_delegation,
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
