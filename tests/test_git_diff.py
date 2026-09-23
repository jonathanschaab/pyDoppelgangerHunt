"""Unit tests for git diff hunks, clone diffs, blame porcelain, and temporal divergence."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

import pydoppelgangerhunt

from pydoppelgangerhunt import (
    check_temporal_divergence,
    compute_repository_dry_stats,
    compute_unit_coverage,
    extract_unit_source_code,
    filter_clones_by_git_diff,
    format_github_annotations,
    format_json_report,
    format_sarif_report,
    generate_html_report,
    generate_refactoring_patch,
    is_unit_in_modified_ranges,
    parse_git_diff_hunks,
    scan_target,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _build_whole_method_delegation,
)


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

def test_compute_unit_diff_overlap_with_notebook_cell_fragment() -> None:
    """Verifies that notebook cell fragments (#cell_1) are stripped when matching diff ranges."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap  # pylint: disable=import-outside-toplevel

    unit = {"file": "notebooks/analysis.ipynb#cell_3", "start": 10, "end": 20}
    modified_ranges = {"notebooks/analysis.ipynb": [(12, 16)]}
    count, ratio = compute_unit_diff_overlap(unit, modified_ranges)
    assert count == 5
    assert ratio == round(5 / 11, 4)


def test_compute_unit_diff_overlap_with_literal_hash_in_path() -> None:
    """Verifies that filenames with literal # (e.g. c#_repo or pkg#2) do not truncate and match properly."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap  # pylint: disable=import-outside-toplevel

    unit = {"file": "c#_repo/worker.py", "start": 10, "end": 20}
    modified_ranges = {"c#_repo/worker.py": [(12, 16)]}
    count, ratio = compute_unit_diff_overlap(unit, modified_ranges)
    assert count == 5
    assert ratio == round(5 / 11, 4)

    # Different path without hash does not match
    assert compute_unit_diff_overlap(unit, {"c/worker.py": [(12, 16)]}) == (0, 0.0)

def test_compute_unit_diff_overlap_missing_file() -> None:
    """Verifies that compute_unit_diff_overlap safely returns (0, 0.0) when file key is missing or empty."""
    from pydoppelgangerhunt.git_diff import compute_unit_diff_overlap  # pylint: disable=import-outside-toplevel

    assert compute_unit_diff_overlap({}, {"foo.py": [(1, 10)]}) == (0, 0.0)
    assert compute_unit_diff_overlap({"file": ""}, {"foo.py": [(1, 10)]}) == (0, 0.0)

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
    assert normalize_path_string("./foo/bar.ipynb#cell_1", strip_anchor=False) == "foo/bar.ipynb#cell_1"
    assert normalize_path_string("./foo/bar.ipynb#cell_1", strip_anchor=True) == "foo/bar.ipynb"
    assert normalize_path_string("./foo/bar.py#cell_1", strip_anchor=True) == "foo/bar.py#cell_1"
    assert normalize_path_string("worker.py#cell_data.py", strip_anchor=True) == "worker.py#cell_data.py"

    # 2. format_json_report normalizes paths while preserving anchors
    u1 = {"file": "./pkg/mod.ipynb#cell_1", "start": 5, "end": 10, "name": "fn1", "token_count": 25}
    u2 = {"file": ".\\pkg\\mod.ipynb#cell_2", "start": 15, "end": 20, "name": "fn2", "token_count": 25}
    mock_fam = {
        "family_id": "CF-100",
        "member_count": 2,
        "unique_files": ["pkg/mod.ipynb"],
        "avg_similarity": 0.95,
        "max_similarity": 0.95,
        "min_similarity": 0.95,
        "coherence": 1.0,
        "total_lines": 10,
        "medoid": {"file": "./pkg/mod.ipynb#cell_1", "name": "fn1", "start": 5, "end": 10},
        "members": [u1, u2],
    }
    json_out = format_json_report([(0.95, u1, u2)], "repo", 0.9, families=[mock_fam])
    assert json_out["clones"][0]["unit_a"]["file"] == "pkg/mod.ipynb#cell_1"
    assert json_out["clones"][0]["unit_b"]["file"] == "pkg/mod.ipynb#cell_2"
    assert json_out["families"][0]["medoid"]["file"] == "pkg/mod.ipynb#cell_1"
    assert json_out["families"][0]["members"][1]["file"] == "pkg/mod.ipynb#cell_2"

    # 3. format_github_annotations strips anchors and leading dot-slash
    annots = format_github_annotations([(0.95, u1, u2)])
    assert len(annots) == 2
    assert "file=pkg/mod.ipynb," in annots[0]
    assert "#cell_" not in annots[0]

    # 4. generate_html_report escapes target and renders normalized paths
    dangerous_target = "<script>alert('pwned')</script>&foo"
    html_report = generate_html_report([(0.95, u1, u2)], dangerous_target, 0.9, families=[mock_fam])
    assert "<script>alert('pwned')</script>" not in html_report
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;&amp;foo" in html_report
    assert "pkg/mod.ipynb#cell_1" in html_report
    assert "pkg/mod.ipynb#cell_2" in html_report

    # 5. _audit_clone_risk_warnings path normalization
    cov_data = {"pkg/mod.ipynb#cell_1": {5, 6, 7, 8, 9, 10}, "pkg/mod.ipynb#cell_2": set()}
    warn_lines = _audit_clone_risk_warnings(u1, u2, cov_data=cov_data, use_color=False)
    assert any("Asymmetric test coverage: pkg/mod.ipynb#cell_1 (100%) vs pkg/mod.ipynb#cell_2 (0%)" in w for w in warn_lines)

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


def test_get_git_modified_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test extracting normalized modified files list from git diff --name-only."""
    from typing import Sequence  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import get_git_modified_files  # pylint: disable=import-outside-toplevel

    captured_args: List[Sequence[str]] = []

    def mock_run_diff(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
        captured_args.append(args)
        return "pkg/mod1.py\npkg/sub/mod2.py\n\npkg/mod1.py\n"

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_diff)
    files = get_git_modified_files(since_ref="main", repo_root="/repo")
    assert "--name-only" in captured_args[0]
    assert "main" in captured_args[0]
    assert files == ["pkg/mod1.py", "pkg/sub/mod2.py"]

    # Quoted filenames from git diff (e.g. core.quotepath with spaces, quotes, octal UTF-8)
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: '"pkg/mod with spaces.py"\n"pkg/sub/mod2.py"\n"pkg/\\\"quoted\\\".py"\n"pkg/caf\\303\\251.py"\n',
    )
    assert get_git_modified_files() == [
        "pkg/mod with spaces.py",
        "pkg/sub/mod2.py",
        'pkg/"quoted".py',
        "pkg/café.py",
    ]

    # When git returns None
    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", lambda args, cwd=None: None)
    assert get_git_modified_files() == []

    # Dash-prefixed since_ref must be rejected immediately to prevent option injection
    from pydoppelgangerhunt.git_diff import get_git_modified_line_ranges  # pylint: disable=import-outside-toplevel
    assert get_git_modified_files(since_ref="--output=/tmp/pwned") == []
    assert get_git_modified_files(since_ref="-h") == []
    assert get_git_modified_line_ranges(since_ref="--diff-filter=A") == {}
    assert get_git_modified_line_ranges(since_ref="-R") == {}


def test_parse_git_diff_hunks_cstyle_and_blame_dash_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parse_git_diff_hunks decodes C-style octal paths and blame porcelain uses '--'."""
    from typing import Sequence  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import get_git_blame_info  # pylint: disable=import-outside-toplevel

    diff_sample = (
        'diff --git "a/pkg/caf\\303\\251.py" "b/pkg/caf\\303\\251.py"\n'
        '--- "a/pkg/caf\\303\\251.py"\n'
        '+++ "b/pkg/caf\\303\\251.py"\n'
        '@@ -10,3 +10,5 @@\n'
        '+def cafe():\n'
    )
    hunks = parse_git_diff_hunks(diff_sample)
    assert "pkg/café.py" in hunks
    assert (10, 14) in hunks["pkg/café.py"]

    blame_args: List[Sequence[str]] = []

    def mock_run_blame(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
        blame_args.append(args)
        return (
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2 1 1 1\n"
            "author Ada\nauthor-time 1700000000\nsummary test\n"
        )

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_blame)
    get_git_blame_info("pkg/café.py", 1, 5, repo_root="/repo")
    assert blame_args
    assert "--" in blame_args[0]
    assert blame_args[0][blame_args[0].index("--") + 1] == "pkg/café.py"


def test_git_diff_args_contain_double_dash_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that get_git_modified_files and get_git_modified_line_ranges isolate revisions with '--'."""
    from typing import Sequence  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import (  # pylint: disable=import-outside-toplevel
        get_git_modified_files,
        get_git_modified_line_ranges,
    )

    captured_cmds: List[Sequence[str]] = []

    def mock_run(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
        captured_cmds.append(args)
        return ""

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run)

    # 1. With since_ref
    get_git_modified_files(since_ref="release_branch")
    assert len(captured_cmds) == 1
    assert captured_cmds[0] == ["diff", "--name-only", "release_branch", "--"]

    get_git_modified_line_ranges(since_ref="release_branch")
    assert len(captured_cmds) == 2
    assert captured_cmds[1] == ["diff", "--unified=0", "--src-prefix=a/", "--dst-prefix=b/", "release_branch", "--"]

    # 2. Without since_ref (unstaged working tree diff)
    get_git_modified_files()
    assert len(captured_cmds) == 3
    assert captured_cmds[2] == ["diff", "--name-only", "--"]

    get_git_modified_line_ranges()
    assert len(captured_cmds) == 4
    assert captured_cmds[3] == ["diff", "--unified=0", "--src-prefix=a/", "--dst-prefix=b/", "--"]


def test_run_git_command_utf8_decoding_and_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _run_git_command enforces utf-8 encoding and replace error handling."""
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.git_diff import _run_git_command  # pylint: disable=import-outside-toplevel

    captured_kwargs: Dict[str, Any] = {}

    def mock_subprocess_run(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "pkg/café.py\npkg/🚀_launch.py\n"
        return mock_proc

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    out = _run_git_command(["diff", "--name-only"])
    assert captured_kwargs.get("encoding") == "utf-8"
    assert captured_kwargs.get("errors") == "replace"
    assert captured_kwargs.get("text") is True
    assert out is not None
    assert "pkg/café.py" in out
    assert "pkg/🚀_launch.py" in out


def test_decode_git_cstyle_path_quoted_literal_unicode() -> None:
    """Verifies that _decode_git_cstyle_path preserves literal Unicode and parses C-style octal escapes."""
    from pydoppelgangerhunt.git_diff import (  # pylint: disable=import-outside-toplevel
        _decode_git_cstyle_path,
        parse_git_diff_hunks,
    )

    # 1. Literal Unicode in quoted path (core.quotepath = false or non-ASCII characters quoted due to space)
    p1 = '"pkg/café file.py"'
    assert _decode_git_cstyle_path(p1) == "pkg/café file.py"

    # 2. Octal escaped Unicode (core.quotepath = true default)
    p2 = '"pkg/caf\\303\\251 file.py"'
    assert _decode_git_cstyle_path(p2) == "pkg/café file.py"

    # 3. Mixed quotes, backslashes, tabs, and octal escapes
    p3 = '"pkg/\\"quoted\\"\\\\file\\t\\303\\251.py"'
    assert _decode_git_cstyle_path(p3) == 'pkg/"quoted"\\file\té.py'

    # 4. Standard C escapes
    p4 = '"pkg/line\\nbreak\\rreturn.py"'
    assert _decode_git_cstyle_path(p4) == "pkg/line\nbreak\rreturn.py"

    # 5. Unquoted plain path
    p5 = "pkg/plain_path.py"
    assert _decode_git_cstyle_path(p5) == "pkg/plain_path.py"

    # 6. Integration with parse_git_diff_hunks
    diff_sample = (
        'diff --git "a/pkg/café file.py" "b/pkg/café file.py"\n'
        '--- "a/pkg/café file.py"\n'
        '+++ "b/pkg/café file.py"\n'
        "@@ -1,5 +1,5 @@\n"
        "+# modified line\n"
    )
    hunks = parse_git_diff_hunks(diff_sample)
    assert any("café file.py" in k for k in hunks)


def test_parse_git_diff_hunks_preserves_literal_quotes_in_filename() -> None:
    """Verifies that parse_git_diff_hunks does not strip legitimate quotes from decoded filenames."""
    from pydoppelgangerhunt.git_diff import parse_git_diff_hunks  # pylint: disable=import-outside-toplevel

    # Git quotes a file containing leading/trailing quotes, escaping internal quotes:
    # e.g., file `"pkg.py"` is output by git as `"\"pkg.py\""`
    diff_with_quotes = (
        'diff --git "a/\\"pkg.py\\"" "b/\\"pkg.py\\""\n'
        '--- "a/\\"pkg.py\\""\n'
        '+++ "b/\\"pkg.py\\""\n'
        "@@ -10,3 +10,3 @@\n"
        "+# modified line\n"
    )
    hunks = parse_git_diff_hunks(diff_with_quotes)
    # The parsed filename should retain the literal quote character rather than having it stripped
    assert any('"pkg.py"' in k or '"pkg.py' in k for k in hunks)


def test_parse_git_diff_hunks_quoted_paths_with_timestamps_and_escapes() -> None:
    """Verifies that parse_git_diff_hunks strips trailing timestamps outside quoted filenames without breaking path content."""
    from pydoppelgangerhunt.git_diff import parse_git_diff_hunks  # pylint: disable=import-outside-toplevel

    diff_timestamp = (
        'diff --git "a/path with space.py" "b/path with space.py"\n'
        '--- "a/path with space.py"\t2026-09-20 12:00:00.000000000 +0000\n'
        '+++ "b/path with space.py"\t2026-09-20 12:00:00.000000000 +0000\n'
        "@@ -5,2 +5,2 @@\n"
        "+# change\n"
    )
    hunks = parse_git_diff_hunks(diff_timestamp)
    assert any("path with space.py" in k for k in hunks)
    assert not any("2026-09-20" in k for k in hunks)

    # Unquoted with timestamp
    diff_unquoted = (
        "diff --git a/simple.py b/simple.py\n"
        "--- a/simple.py\t2026-09-20 12:00:00.000000000 +0000\n"
        "+++ b/simple.py\t2026-09-20 12:00:00.000000000 +0000\n"
        "@@ -1,1 +1,1 @@\n"
        "+# change\n"
    )
    hunks_unquoted = parse_git_diff_hunks(diff_unquoted)
    assert any("simple.py" in k for k in hunks_unquoted)
    assert not any("2026-09-20" in k for k in hunks_unquoted)


def test_get_git_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies get_git_repo_root resolves normalized worktree top-level path and handles non-git fallbacks."""
    from pydoppelgangerhunt.git_diff import get_git_repo_root  # pylint: disable=import-outside-toplevel

    sub_dir = tmp_path / "src" / "pkg"
    sub_dir.mkdir(parents=True)
    expected_root = str(tmp_path).replace("\\", "/")

    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: expected_root + "\n" if args == ["rev-parse", "--show-toplevel"] else None,
    )
    resolved = get_git_repo_root(repo_root=sub_dir)
    assert resolved == expected_root

    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: None,
    )
    assert get_git_repo_root(repo_root=sub_dir) is None


def test_subdirectory_scan_in_git_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies CLI differential scan targeting a subdirectory inside a Git worktree resolves worktree root."""
    repo = tmp_path / "worktree_repo"
    repo.mkdir()
    subpkg = repo / "src" / "subpkg"
    subpkg.mkdir(parents=True)

    foo_code = (
        "def execute_pipeline(data_batch, multiplier):\n"
        "    res = 0\n"
        "    for val in data_batch:\n"
        "        res += val * multiplier + 10\n"
        "    return res\n"
    )
    clone_code = (
        "def execute_pipeline(data_batch, multiplier):\n"
        "    res = 0\n"
        "    for val in data_batch:\n"
        "        res += val * multiplier + 10\n"
        "    return res\n"
    )

    (subpkg / "worker.py").write_text(foo_code, encoding="utf-8")
    (subpkg / "worker_clone.py").write_text(clone_code, encoding="utf-8")

    norm_repo = str(repo).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: (
            norm_repo + "\n"
            if args == ["rev-parse", "--show-toplevel"]
            else (
                "src/subpkg/worker.py\n"
                if args == ["diff", "--name-only", "--"]
                else (
                    "diff --git a/src/subpkg/worker.py b/src/subpkg/worker.py\n"
                    "--- a/src/subpkg/worker.py\n"
                    "+++ b/src/subpkg/worker.py\n"
                    "@@ -1,5 +1,5 @@\n"
                    "+# change\n"
                    if "--unified=0" in args
                    else None
                )
            )
        ),
    )

    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    test_args = [
        "pydoppelgangerhunt",
        str(subpkg),
        "--diff-only",
        "--threshold",
        "0.90",
        "--min-lines",
        "5",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    exit_code = main()
    assert exit_code == 1


def test_get_git_repo_root_security_and_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies get_git_repo_root handles non-existent paths, null bytes, and corrupted git metadata safely."""
    from pydoppelgangerhunt.git_diff import get_git_repo_root  # pylint: disable=import-outside-toplevel

    # 1. Non-existent path returns None without raising exceptions
    non_existent = tmp_path / "definitely_does_not_exist_12345"
    assert get_git_repo_root(repo_root=non_existent) is None

    # 2. Path with null byte is safely handled
    null_byte_path = str(tmp_path) + "\x00invalid"
    assert get_git_repo_root(repo_root=null_byte_path) is None

    # 3. Empty or whitespace-only git output returns None
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: "   \n\t\n  ",
    )
    assert get_git_repo_root(repo_root=tmp_path) is None

    # 4. Git error / non-zero exit returns None
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: None,
    )
    assert get_git_repo_root(repo_root=tmp_path) is None

    # 5. Repository path containing hash characters is preserved (not stripped as an anchor)
    hash_repo = "/tmp/repo#1/subproject"
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: hash_repo + "\n" if args == ["rev-parse", "--show-toplevel"] else None,
    )
    assert get_git_repo_root(repo_root=tmp_path) == hash_repo

    # 6. Repository path containing significant trailing/leading space is preserved
    spaced_repo = "/tmp/repo "
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: spaced_repo + "\r\n" if args == ["rev-parse", "--show-toplevel"] else None,
    )
    assert get_git_repo_root(repo_root=tmp_path) == spaced_repo



def test_diff_unit_matching_subprocess_efficiency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies scan_target resolves git root once rather than executing git subprocess for every unit."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    code_file = tmp_path / "sample.py"
    # Generate multiple functions
    functions_code = "\n\n".join(
        f"def fn_{i}(a, b):\n    val = a * 2 + b * {i}\n    return val + 10\n"
        for i in range(20)
    )
    code_file.write_text(functions_code, encoding="utf-8")

    git_call_count = 0

    def counting_git_command(args: Any, cwd: Any = None) -> Optional[str]:  # pylint: disable=unused-argument
        nonlocal git_call_count
        if args == ["rev-parse", "--show-toplevel"]:
            git_call_count += 1
            return str(tmp_path).replace("\\", "/") + "\n"
        return None

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", counting_git_command)

    # Run scan_target with diff_files specifying a non-matching file
    scan_target(
        str(tmp_path),
        min_lines=2,
        min_tokens=3,
        threshold=0.90,
        diff_files=["other_unrelated_file.py"],
    )

    # Crucial assertion: get_git_repo_root must not be invoked per unit
    assert git_call_count <= 2


def test_normalize_git_paths_and_ranges_excludes_out_of_target_files(tmp_path: Path) -> None:
    """Verifies that git paths and modified ranges outside target directory are excluded."""
    from pydoppelgangerhunt.cli import (  # pylint: disable=import-outside-toplevel
        _normalize_git_paths_for_target,
        _normalize_modified_ranges_for_target,
    )

    repo_dir = tmp_path / "repo"
    target_dir = repo_dir / "pkg"
    target_dir.mkdir(parents=True)

    raw_paths = ["foo.py", "pkg/bar.py", "other/baz.py"]
    norm_paths = _normalize_git_paths_for_target(raw_paths, str(repo_dir), str(target_dir))
    assert "foo.py" not in norm_paths
    assert "other/baz.py" not in norm_paths
    assert "pkg/bar.py" in norm_paths
    assert "bar.py" not in norm_paths

    ranges = {
        "foo.py": [(1, 10)],
        "pkg/bar.py": [(20, 30)],
        "other/baz.py": [(5, 15)],
    }
    norm_ranges = _normalize_modified_ranges_for_target(ranges, str(repo_dir), str(target_dir))
    assert "foo.py" not in norm_ranges
    assert "other/baz.py" not in norm_ranges
    assert norm_ranges["bar.py"] == [(20, 30)]
    assert norm_ranges["pkg/bar.py"] == [(20, 30)]


def test_normalize_git_paths_prevents_unchanged_same_named_file_collision(tmp_path: Path) -> None:
    """Verifies that _normalize_git_paths_for_target does not add target-relative alias which collides with unchanged files."""
    from pydoppelgangerhunt.cli import _normalize_git_paths_for_target  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.canonical_path import CanonicalPathResolver, build_diff_path_keys  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    src = repo / "src"
    nested_src = src / "src"
    nested_src.mkdir(parents=True)

    # Modified file is repo/src/src/foo.py
    raw_paths = ["src/src/foo.py"]
    norm_paths = _normalize_git_paths_for_target(raw_paths, str(repo), str(src))
    assert norm_paths == ["src/src/foo.py"]
    assert "src/foo.py" not in norm_paths

    resolver = CanonicalPathResolver(target_root=src, repo_root=repo)
    diff_keys = build_diff_path_keys(norm_paths, resolver)
    # Unchanged file at repo/src/foo.py (target-relative 'foo.py', repo-relative 'src/foo.py') must NOT match
    assert not resolver.matches_diff("foo.py", diff_keys, basis="target")
    # Modified file at repo/src/src/foo.py (target-relative 'src/foo.py', repo-relative 'src/src/foo.py') MUST match
    assert resolver.matches_diff("src/foo.py", diff_keys, basis="target")


def test_decode_git_cstyle_path_preserves_spaces() -> None:
    """Verifies that _decode_git_cstyle_path preserves significant leading/trailing whitespace in unquoted paths."""
    from pydoppelgangerhunt.git_diff import _decode_git_cstyle_path  # pylint: disable=import-outside-toplevel

    assert _decode_git_cstyle_path(" foo.py\n") == " foo.py"
    assert _decode_git_cstyle_path("bar.py \r\n") == "bar.py "
    assert _decode_git_cstyle_path("  pkg/baz.py  \n") == "  pkg/baz.py  "
    assert _decode_git_cstyle_path('" foo.py"\n') == " foo.py"
    assert _decode_git_cstyle_path("   \n") == ""


def test_parse_git_diff_hunks_preserves_trailing_spaces_unquoted() -> None:
    """Verifies that parse_git_diff_hunks preserves trailing spaces in unquoted diff filenames."""
    from pydoppelgangerhunt.git_diff import parse_git_diff_hunks  # pylint: disable=import-outside-toplevel

    # 1. Unquoted diff header with trailing space before newline
    diff_unquoted_trailing_space = (
        "diff --git a/foo.py  b/foo.py \n"
        "--- a/foo.py \n"
        "+++ b/foo.py \n"
        "@@ -1,3 +1,3 @@\n"
        "+# modified line\n"
    )
    hunks = parse_git_diff_hunks(diff_unquoted_trailing_space)
    assert "foo.py " in hunks
    assert hunks["foo.py "] == [(1, 3)]

    # 2. Unquoted diff header with trailing space before timestamp tab
    diff_unquoted_timestamp = (
        "diff --git a/bar.py  b/bar.py \n"
        "--- a/bar.py \t2026-09-22 12:00:00.000000000 +0000\n"
        "+++ b/bar.py \t2026-09-22 12:00:00.000000000 +0000\n"
        "@@ -10,2 +10,2 @@\n"
        "+# modified line\n"
    )
    hunks_ts = parse_git_diff_hunks(diff_unquoted_timestamp)
    assert "bar.py " in hunks_ts
    assert hunks_ts["bar.py "] == [(10, 11)]


def test_run_git_diff_clean_since_ref() -> None:
    """Verifies that _run_git_diff cleans since_ref, rejects leading hyphens, and ignores whitespace-only refs."""
    from pydoppelgangerhunt.git_diff import _run_git_diff  # pylint: disable=import-outside-toplevel

    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command") as mock_cmd:
        mock_cmd.return_value = ""
        # 1. Whitespace-only ref behaves like unstaged diff without passing whitespace to git
        _run_git_diff(["--name-only"], since_ref="   ")
        mock_cmd.assert_called_once_with(["diff", "--name-only", "--"], cwd=None)

    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command") as mock_cmd:
        # 2. Leading hyphen rejection
        res = _run_git_diff(["--name-only"], since_ref="  --invalid-flag  ")
        assert res is None
        mock_cmd.assert_not_called()

    with mock.patch("pydoppelgangerhunt.git_diff._run_git_command") as mock_cmd:
        mock_cmd.return_value = ""
        # 3. Valid ref with surrounding whitespace is stripped
        _run_git_diff(["--name-only"], since_ref="  HEAD~1  ")
        mock_cmd.assert_called_once_with(["diff", "--name-only", "HEAD~1", "--"], cwd=None)

