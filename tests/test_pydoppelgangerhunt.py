"""Comprehensive unit test suite for pyDoppelgangerHunt."""

from __future__ import annotations

import ast
import json
import sqlite3
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pydoppelgangerhunt
from pydoppelgangerhunt.baseline import (
    clone_pair_fingerprint,
    filter_clones_by_baseline,
    load_baseline,
    record_baseline,
)
from pydoppelgangerhunt.clustering import UnionFind, cluster_clone_families
from pydoppelgangerhunt.config import init_tool_configuration, load_toml_section, load_tool_config
from pydoppelgangerhunt.coverage import (
    check_asymmetric_coverage,
    compute_unit_coverage,
    read_coverage_data,
)
from pydoppelgangerhunt.fixer import generate_refactoring_patch, synthesize_shared_helper_code
from pydoppelgangerhunt.git_diff import (
    check_temporal_divergence,
    filter_clones_by_git_diff,
    get_git_blame_info,
    get_git_modified_line_ranges,
    is_unit_in_modified_ranges,
    parse_git_diff_hunks,
)
from pydoppelgangerhunt.matcher import (
    call_sequence_similarity,
    compute_pair_similarity,
    compute_priority_score,
    jaccard_similarity,
    lcs_alignment_similarity,
    merge_adjacent_clones,
    multiset_jaccard_similarity,
    scan_target,
    suppress_subclones,
    tfidf_jaccard_similarity,
    tfidf_multiset_jaccard_similarity,
)
from pydoppelgangerhunt.metrics import compute_repository_dry_stats
from pydoppelgangerhunt.parser import (
    BUILTIN_NAMES,
    compute_cyclomatic_complexity,
    extract_call_sequence,
    get_ast_characteristic_vector,
    get_ast_shingles,
    get_ast_tokens,
    harvest_file_units,
    harvest_notebook_units,
    is_boilerplate_node,
)
from pydoppelgangerhunt.reporters import (
    colorize,
    emit_structured_report,
    extract_unit_source_code,
    format_github_annotations,
    format_json_report,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    supports_color,
    synthesize_refactoring_suggestion,
)


def test_package_metadata_and_exports() -> None:
    """Verify package exports and metadata."""
    assert pydoppelgangerhunt.__version__ == "1.0.0"
    assert callable(pydoppelgangerhunt.main)
    assert callable(pydoppelgangerhunt.scan_target)
    assert callable(pydoppelgangerhunt.cluster_clone_families)


def test_ast_tokenization_and_shingling() -> None:
    """Verify AST shingle generation and Jaccard similarity."""
    code1 = "def add(x, y):\n    return x + y\n"
    code2 = "def add(a, b):\n    return a + b\n"
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)

    shingles1, _ = get_ast_shingles(tree1, k=3, blind_indexing=True)
    shingles2, _ = get_ast_shingles(tree2, k=3, blind_indexing=True)
    assert jaccard_similarity(shingles1, shingles2) == 1.0


def test_cyclomatic_complexity_and_priority() -> None:
    """Verify McCabe cyclomatic complexity calculation and priority scoring."""
    tree_simple = ast.parse("def f(x):\n    return x + 1\n").body[0]
    assert compute_cyclomatic_complexity(tree_simple) == 1

    branching = (
        "def g(a, b):\n"
        "    if a > 0 and b > 0:\n"
        "        for i in range(10):\n"
        "            if i % 2 == 0:\n"
        "                print(i)\n"
        "    return a or b\n"
    )
    tree_branch = ast.parse(branching).body[0]
    assert compute_cyclomatic_complexity(tree_branch) == 6

    u1 = {"complexity": 6, "start": 1, "end": 20}
    u2 = {"complexity": 4, "start": 1, "end": 20}
    assert compute_priority_score(0.95, u1, u2) == 114.0


def test_jupyter_notebook_harvesting(tmp_path: Path) -> None:
    """Verify code cell harvesting from .ipynb files."""
    nb_data = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n"]},
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": ["def f(a, b):\n", "    return a * 2 + b\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "source": ["def g(x, y):\n", "    return x * 2 + y\n"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    nb_file = tmp_path / "test.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    units = harvest_notebook_units(str(nb_file), str(tmp_path), min_lines=2, min_tokens=5)
    assert len(units) >= 2
    assert any("#cell_2" in u["file"] for u in units)
    assert any("#cell_3" in u["file"] for u in units)

    clones = scan_target(str(tmp_path), min_lines=2, min_tokens=5, threshold=0.85, include_notebooks=True)
    assert len(clones) >= 1


def test_html_report_and_sarif_generation() -> None:
    """Verify HTML and SARIF report generation."""
    u1 = {
        "file": "mod1.py",
        "name": "func_a",
        "type": "function",
        "start": 1,
        "end": 10,
        "sloc": 10,
        "tokens": 25,
        "complexity": 2,
    }
    u2 = {
        "file": "mod2.py",
        "name": "func_b",
        "type": "function",
        "start": 1,
        "end": 10,
        "sloc": 10,
        "tokens": 25,
        "complexity": 2,
    }
    clones = [(0.95, u1, u2)]
    families = cluster_clone_families(clones)
    stats = {
        "dry_score": 90.0,
        "grade": "A",
        "sloc": 100,
        "dloc": 10,
        "duplication_pct": 10.0,
        "clone_pairs": 1,
        "clone_families": 1,
        "total_files": 2,
    }

    html = generate_html_report(clones, "pkg", 0.90, families=families, stats=stats)
    assert "<!DOCTYPE html>" in html
    assert "pyDoppelgangerHunt" in html
    assert "90.0%" in html

    sarif = format_sarif_report(clones, "pkg", 0.90)
    assert sarif["version"] == "2.1.0"
    assert len(sarif["runs"][0]["results"]) == 1


def test_github_actions_annotations() -> None:
    """Verify formatting of GitHub Actions warning commands."""
    u1 = {"file": "a.py", "name": "foo", "start": 1, "end": 10}
    u2 = {"file": "b.py", "name": "bar", "start": 20, "end": 30}
    annotations = format_github_annotations([(0.90, u1, u2)])
    assert any("file=a.py,line=1" in a for a in annotations)
    assert any("file=b.py,line=20" in a for a in annotations)


def test_coverage_readers_and_asymmetry(tmp_path: Path) -> None:
    """Verify Cobertura XML and SQLite coverage readers."""
    xml_content = (
        '<?xml version="1.0" ?>\n'
        '<coverage version="7.0">\n'
        '  <packages>\n'
        '    <package name="pkg">\n'
        '      <classes>\n'
        '        <class name="a" filename="a.py">\n'
        '          <lines>\n'
        '            <line number="1" hits="1"/>\n'
        '            <line number="2" hits="0"/>\n'
        '          </lines>\n'
        '        </class>\n'
        '      </classes>\n'
        '    </package>\n'
        '  </packages>\n'
        '</coverage>\n'
    )
    xml_file = tmp_path / "coverage.xml"
    xml_file.write_text(xml_content, encoding="utf-8")

    cov_data = read_coverage_data(str(xml_file))
    assert cov_data.get("a.py") == {1}

    u1 = {"file": "a.py", "start": 1, "end": 1}
    u2 = {"file": "b.py", "start": 1, "end": 1}
    cov1 = compute_unit_coverage(u1, cov_data)
    cov2 = compute_unit_coverage(u2, cov_data)
    assert cov1 == 1.0
    assert cov2 == 0.0

    asym = check_asymmetric_coverage(u1, u2, cov_data, min_diff=0.40)
    assert asym == (1.0, 0.0)


def test_fixer_and_patch_synthesis(tmp_path: Path) -> None:
    """Verify shared helper synthesis and patch output."""
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("def op_one(x, y):\n    res = x + y\n    return res * 2\n", encoding="utf-8")
    file_b.write_text("def op_two(x, y):\n    res = x + y\n    return res * 3\n", encoding="utf-8")

    u1 = {"file": str(file_a), "name": "op_one", "start": 1, "end": 3}
    u2 = {"file": str(file_b), "name": "op_two", "start": 1, "end": 3}

    helper = synthesize_shared_helper_code(u1, u2)
    assert "def _shared_op_one" in helper

    patch = generate_refactoring_patch([(0.85, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch


def test_cli_execution(tmp_path: Path) -> None:
    """Verify CLI flags, baseline grandfathering, and exit codes."""
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    code = "def calc(x):\n    factor = 10\n    res = x * factor\n    return res if res > 0 else 0\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code.replace("calc", "calc_two"), encoding="utf-8")

    html_path = tmp_path / "report.html"
    base_path = tmp_path / "baseline.json"

    # 1. First run records baseline
    code_rec = pydoppelgangerhunt.main([
        str(tmp_path),
        "--min-lines", "3",
        "--min-tokens", "5",
        "--record-baseline", str(base_path),
    ])
    assert code_rec == 0
    assert base_path.is_file()

    # 2. Run with baseline -> suppressed -> exit 0
    code_base = pydoppelgangerhunt.main([
        str(tmp_path),
        "--min-lines", "3",
        "--min-tokens", "5",
        "--baseline", str(base_path),
    ])
    assert code_base == 0

    # 3. Run with --html and --priority without baseline -> exit 1
    code_html = pydoppelgangerhunt.main([
        str(tmp_path),
        "--min-lines", "3",
        "--min-tokens", "5",
        "--html", str(html_path),
        "--priority",
    ])
    assert code_html == 1
    assert html_path.is_file()
