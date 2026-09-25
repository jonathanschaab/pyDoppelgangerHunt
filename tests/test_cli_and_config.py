"""Unit tests for cli and config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock
import pytest

import pydoppelgangerhunt

from pydoppelgangerhunt import (
    clone_pair_fingerprint,
    clone_pair_structural_fingerprint,
    filter_clones_by_baseline,
    init_tool_configuration,
    load_baseline,
    load_toml_section,
    load_tool_config,
    namespaced_structural_fingerprint,
    pure_structural_fingerprint,
    record_baseline,
)


def test_load_tool_config_and_toml_section(tmp_path: Path) -> None:
    """Test loading [tool.pydoppelgangerhunt] from pyproject.toml and custom paths."""
    toml_file = tmp_path / "custom_config.toml"
    toml_file.write_text(
        """
[tool.pydoppelgangerhunt]
threshold = 0.72
min_lines = 12
idioms = true
exclude = ["custom_dir"]
exemptions = [["a.py:f1", "b.py:f2"]]
""",
        encoding="utf-8",
    )
    cfg = load_toml_section(toml_file, "pydoppelgangerhunt")
    assert cfg.get("threshold") == 0.72
    assert cfg.get("min_lines") == 12
    assert cfg.get("idioms") is True

    # Test load_tool_config on custom path
    loaded = load_tool_config(str(toml_file))
    assert loaded.get("threshold") == 0.72
    assert loaded.get("exclude") == ["custom_dir"]

    # Test load_tool_config on repository pyproject.toml
    repo_cfg = load_tool_config()
    assert "threshold" in repo_cfg
    assert repo_cfg.get("idioms") is True



def test_init_configuration_generation(tmp_path: Path) -> None:
    """Test --init generation for pyproject.toml and standalone .pydoppelgangerhunt.toml."""
    dir_empty = tmp_path / "empty_proj"
    dir_empty.mkdir()
    res_standalone = init_tool_configuration(str(dir_empty))
    assert ".pydoppelgangerhunt.toml" in res_standalone
    assert (dir_empty / ".pydoppelgangerhunt.toml").exists()
    content_standalone = (dir_empty / ".pydoppelgangerhunt.toml").read_text(encoding="utf-8")
    assert "[tool.pydoppelgangerhunt]" in content_standalone
    assert "threshold = 0.90" in content_standalone

    dir_pyproj = tmp_path / "pyproj_proj"
    dir_pyproj.mkdir()
    pyproject_file = dir_pyproj / "pyproject.toml"
    pyproject_file.write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    res_pyproj = init_tool_configuration(str(dir_pyproj))
    assert "pyproject.toml" in res_pyproj
    content_pyproj = pyproject_file.read_text(encoding="utf-8")
    assert "[project]" in content_pyproj
    assert "[tool.pydoppelgangerhunt]" in content_pyproj

    res_idempotent = init_tool_configuration(str(dir_pyproj))
    assert "already configured" in res_idempotent



def test_baseline_record_load_and_filter(tmp_path: Path) -> None:
    """Test recording clone pairs to a baseline JSON file and filtering grandfathered hits."""
    baseline_file = tmp_path / "clones_baseline.json"
    u1 = {"file": "service/alpha.py", "start": 10, "end": 20, "name": "handler_a", "structural_hash": "hash_a1"}
    u2 = {"file": "service/beta.py", "start": 15, "end": 25, "name": "handler_b", "structural_hash": "hash_b1"}
    u3 = {"file": "service/gamma.py", "start": 30, "end": 40, "name": "handler_c", "structural_hash": "hash_c1"}
    u4 = {"file": "service/delta.py", "start": 35, "end": 45, "name": "handler_d", "structural_hash": "hash_d1"}

    mock_clones = [
        (0.95, u1, u2),
        (0.91, u3, u4),
    ]

    saved_path = record_baseline(mock_clones, str(baseline_file), target="service", threshold=0.90)
    assert Path(saved_path).exists()

    loaded_fps = load_baseline(str(baseline_file))
    assert len(loaded_fps) == 8
    fp1 = clone_pair_fingerprint(u1, u2)
    assert fp1 in loaded_fps
    assert clone_pair_structural_fingerprint(u1, u2) in loaded_fps
    assert namespaced_structural_fingerprint(u1, u2) in loaded_fps
    assert pure_structural_fingerprint(u1, u2) in loaded_fps

    new_clones, suppressed = filter_clones_by_baseline(mock_clones, loaded_fps)
    assert len(new_clones) == 0
    assert suppressed == 2

    u5 = {"file": "service/new.py", "start": 5, "end": 15, "name": "new_func"}
    mock_clones_with_new = mock_clones + [(0.98, u1, u5)]
    new_clones_2, suppressed_2 = filter_clones_by_baseline(mock_clones_with_new, loaded_fps)
    assert len(new_clones_2) == 1
    assert suppressed_2 == 2
    assert new_clones_2[0][2]["name"] == "new_func"



def test_cli_advanced_flags(tmp_path: Path) -> None:
    """Test CLI flags --html, --patch, --sort-by priority, and --top."""
    code = (
        "def compute_one(a, b):\n"
        "    res = a * 2 + b\n"
        "    return res if res > 0 else 0\n"
    )
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code.replace("compute_one", "compute_two"), encoding="utf-8")

    html_out = tmp_path / "report.html"
    patch_out = tmp_path / "fix.patch"

    exit_code = pydoppelgangerhunt.main([
        str(tmp_path),
        "--min-lines", "3",
        "--min-tokens", "5",
        "--html", str(html_out),
        "--patch", str(patch_out),
        "--sort-by", "priority",
        "--top", "5",
        "--github-annotations",
    ])
    assert exit_code == 1
    assert html_out.is_file()
    assert "<!DOCTYPE html>" in html_out.read_text(encoding="utf-8")
    assert patch_out.is_file()
    assert len(patch_out.read_text(encoding="utf-8")) > 0



def test_cli_cluster_suggest_diff_and_formats(tmp_path: Path) -> None:
    """Test CLI flags --cluster, --suggest, --diff, --format sarif, and --stats."""
    code = (
        "def compute_alpha(x, y):\n"
        "    v1 = x * 10\n"
        "    v2 = y * 20\n"
        "    return v1 + v2\n"
    )
    f1 = tmp_path / "mod_a.py"
    f2 = tmp_path / "mod_b.py"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code.replace("compute_alpha", "compute_beta"), encoding="utf-8")

    sarif_out = tmp_path / "out.sarif"
    json_out = tmp_path / "out.json"
    summary_out = tmp_path / "summary.md"
    text_out = tmp_path / "out.txt"

    # Test cluster, suggest, diff
    code_cluster = pydoppelgangerhunt.main([
        str(tmp_path),
        "--threshold", "0.50",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--cluster",
        "--suggest",
        "--diff",
        "--stats",
        "--summary", str(summary_out),
    ])
    assert code_cluster == 1
    assert summary_out.is_file()

    # Test format sarif and json
    code_sarif = pydoppelgangerhunt.main([
        str(tmp_path),
        "--threshold", "0.50",
        "--min-lines", "3",
        "--format", "sarif",
        "--output", str(sarif_out),
    ])
    assert code_sarif == 1
    assert sarif_out.is_file()

    code_json = pydoppelgangerhunt.main([
        str(tmp_path),
        "--threshold", "0.50",
        "--min-lines", "3",
        "--format", "json",
        "--output", str(json_out),
    ])
    assert code_json == 1
    assert json_out.is_file()

    # Test clean output with high threshold
    code_clean = pydoppelgangerhunt.main([
        str(tmp_path),
        "--threshold", "0.99",
        "--min-lines", "50",
        "--output", str(text_out),
    ])
    assert code_clean == 0
    assert text_out.is_file()
    assert "[OK]" in text_out.read_text(encoding="utf-8")



def test_cli_init_and_diff_only(tmp_path: Path, monkeypatch: Any) -> None:
    """Test CLI execution for --init and --diff-only."""
    # Test --init
    init_dir = tmp_path / "init_proj"
    init_dir.mkdir()
    code_init = pydoppelgangerhunt.main(["--init", str(init_dir)])
    assert code_init == 0
    assert (init_dir / ".pydoppelgangerhunt.toml").is_file()

    # Test --diff-only with monkeypatched git modified ranges
    from pydoppelgangerhunt import cli
    f = init_dir / "target.py"
    f.write_text("def a(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "get_git_modified_line_ranges",
        lambda since_ref=None, repo_root=None, cwd=None, **kwargs: {str(f.resolve()): [(1, 1)]},
    )
    code_diff = pydoppelgangerhunt.main([str(init_dir), "--diff-only", "--threshold", "0.90", "--min-lines", "1"])
    assert code_diff == 0



def test_toml_fallback_line_parser(tmp_path: Path, monkeypatch: Any) -> None:
    """Test fallback line-by-line parser when standard toml libraries raise error."""
    toml_file = tmp_path / "fallback.toml"
    toml_file.write_text(
        '[tool.pydoppelgangerhunt]\nthreshold = 0.85\nmin_lines = 10\nidioms = true\nflag = false\nname = "test"\n',
        encoding="utf-8",
    )
    import builtins
    from pydoppelgangerhunt import config
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomli", "tomllib"):
            raise ImportError("mocked no toml parser")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    res = config.load_toml_section(toml_file, "pydoppelgangerhunt")
    assert res.get("threshold") == 0.85
    assert res.get("min_lines") == 10
    assert res.get("idioms") is True
    assert res.get("flag") is False
    assert res.get("name") == "test"


def test_cli_normalization_and_frequency_flags(tmp_path: Path) -> None:
    """Test CLI flags --preserve-docstrings, --preserve-annotations, and --max-index-frequency."""
    f = tmp_path / "sample.py"
    f.write_text(
        'def compute(x: int) -> int:\n'
        '    """Sample docstring."""\n'
        '    return x * 2\n',
        encoding="utf-8",
    )
    code = pydoppelgangerhunt.main([
        str(tmp_path),
        "--preserve-docstrings",
        "--preserve-annotations",
        "--max-index-frequency", "0.20",
        "--threshold", "0.90",
    ])
    assert code == 0


def test_public_api_exports() -> None:
    """Verify that all declared exports in __all__ exist on pydoppelgangerhunt."""
    for name in pydoppelgangerhunt.__all__:
        assert hasattr(pydoppelgangerhunt, name), f"Missing export: {name}"
    assert "canonical_path_key" in pydoppelgangerhunt.__all__
    assert "normalize_path_string" in pydoppelgangerhunt.__all__
    assert "COMPOUND_BLOCK_TYPES" in pydoppelgangerhunt.__all__
    assert "BRANCH_NODE_TYPES" in pydoppelgangerhunt.__all__


def test_cli_dynamic_target_resolution(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify CLI target fallback resolution from tool config and directory detection."""
    monkeypatch.chdir(tmp_path)

    # 1. No target, no pyproject.toml -> defaults to '.'
    code_empty = pydoppelgangerhunt.main([])
    assert code_empty == 0

    # 2. Configured target in pyproject.toml
    sub_dir = tmp_path / "custom_module"
    sub_dir.mkdir()
    (sub_dir / "code.py").write_text("def f() -> int:\n    return 10\n", encoding="utf-8")
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text('[tool.pydoppelgangerhunt]\ntarget = "custom_module"\n', encoding="utf-8")
    code_cfg = pydoppelgangerhunt.main([])
    assert code_cfg == 0

    # 3. CLI --init without target defaults to '.'
    init_dir = tmp_path / "new_repo"
    init_dir.mkdir()
    monkeypatch.chdir(init_dir)
    code_init = pydoppelgangerhunt.main(["--init"])
    assert code_init == 0
    assert (init_dir / ".pydoppelgangerhunt.toml").exists()


def test_batch_48_cli_and_config_edge_cases(tmp_path: Path) -> None:
    """Batch 48: Test CLI diff/suggest output, warnings, Type-4 execution, and config fallbacks."""
    # pylint: disable=protected-access,import-outside-toplevel
    import os
    import sys
    from unittest import mock
    from pydoppelgangerhunt.cli import _audit_clone_risk_warnings, _write_artifact_file
    from pydoppelgangerhunt.config import (
        canonical_path_key,
        find_python_files,
        paths_match_boundary,
    )

    # 1. cli: _write_artifact_file with verbose=False
    dest_file = tmp_path / "silent_artifact.txt"
    _write_artifact_file(str(dest_file), "content", "TEST", verbose=False)
    assert dest_file.read_text(encoding="utf-8") == "content"

    # 2. cli: _audit_clone_risk_warnings divergence and coverage warnings
    u_mock1 = {"file": "svc/a.py", "start": 1, "end": 10}
    u_mock2 = {"file": "svc/b.py", "start": 1, "end": 10}
    mock_div = {"divergence_days": 120.5}
    with mock.patch("pydoppelgangerhunt.cli.check_temporal_divergence", return_value=mock_div):
        warns = _audit_clone_risk_warnings(u_mock1, u_mock2, audit_blame=True)
        assert any("Divergent clone risk" in w for w in warns)

    mock_cov_data = {"svc/a.py": {1, 2, 3}}
    with mock.patch("pydoppelgangerhunt.cli.check_asymmetric_coverage", return_value=(0.9, 0.1)):
        warns_cov = _audit_clone_risk_warnings(u_mock1, u_mock2, cov_data=mock_cov_data)
        assert any("Asymmetric test coverage" in w for w in warns_cov)

    # 3. cli: --diff, --suggest, --output, and @pytest.mark.parametrize tip
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    code_content = (
        "def test_foo(x):\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    return c\n\n"
        "def test_bar(x):\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    return c\n"
    )
    (test_repo / "test_sample.py").write_text(code_content, encoding="utf-8")
    out_file = tmp_path / "cli_report.txt"

    code_res = pydoppelgangerhunt.main([
        str(test_repo),
        "--threshold", "0.80",
        "--min-lines", "4",
        "--diff",
        "--suggest",
        "--audit-tests",
        "--output", str(out_file),
    ])
    assert code_res == 1
    assert out_file.exists()
    out_text = out_file.read_text(encoding="utf-8")
    assert "pytest.mark.parametrize" in out_text
    assert "Diff" in out_text

    # 4. cli: --type4 fallback when module missing and strict-type4 failure
    code_type4_missing = pydoppelgangerhunt.main([str(test_repo), "--type4"])
    assert code_type4_missing == 0

    fake_mod = mock.MagicMock()
    fake_mod.find_semantic_clones.return_value = [{"dummy": 1}]
    fake_mod.report_semantic_results.return_value = True
    with mock.patch.dict("sys.modules", {"check_semantic_clones": fake_mod}):
        clean_dir = tmp_path / "clean_empty"
        clean_dir.mkdir()
        code_strict = pydoppelgangerhunt.main([
            str(clean_dir),
            "--type4",
            "--strict-type4",
        ])
        assert code_strict == 1

    # 5. config: canonical_path_key non-Windows & paths_match_boundary edge cases
    with mock.patch.object(os, "name", "posix"):
        with mock.patch.object(sys, "platform", "linux"):
            assert canonical_path_key("Some/Path/File.py") == "Some/Path/File.py"

    assert not paths_match_boundary(None, "mod.py")
    assert not paths_match_boundary("mod.py", None)
    assert not paths_match_boundary("foo/a.py", "bar/b.py")

    # 6. config: find_python_files edge cases
    assert find_python_files(tmp_path / "non_existent_folder") == []
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("hello", encoding="utf-8")
    assert find_python_files(txt_file) == []
    py_single = test_repo / "test_sample.py"
    assert len(find_python_files(py_single)) == 1

    # 7. config: load_toml_section edge cases (flat toml, section transitions, unquoted values)
    assert load_toml_section(tmp_path / "missing.toml", "tool") == {}

    flat_toml = tmp_path / "flat.toml"
    flat_toml.write_text("threshold = 0.77\nmin_lines = 8\nunquoted = some_value\n", encoding="utf-8")
    flat_cfg = load_toml_section(flat_toml, "pydoppelgangerhunt")
    assert flat_cfg.get("threshold") == 0.77

    multi_section = tmp_path / "multi.toml"
    multi_section.write_text(
        "[tool.pydoppelgangerhunt]\n"
        "threshold = 0.88\n"
        "[tool.other]\n"
        "threshold = 0.11\n",
        encoding="utf-8",
    )
    multi_cfg = load_toml_section(multi_section, "pydoppelgangerhunt")
    assert multi_cfg.get("threshold") == 0.88

    # 8. config: load_tool_config fallback between candidates
    fallback_dir = tmp_path / "fallback_dir"
    fallback_dir.mkdir()
    (fallback_dir / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (fallback_dir / ".pydoppelgangerhunt.toml").write_text(
        "[tool.pydoppelgangerhunt]\nthreshold = 0.93\n", encoding="utf-8"
    )
    fallback_cfg = load_tool_config(repo_root=str(fallback_dir))
    assert fallback_cfg.get("threshold") == 0.93


def test_batch_52_toml_lists_and_quotes(tmp_path: Path, monkeypatch: Any) -> None:
    """Batch 52: Test zero-dependency TOML parser with single quotes, lists, and typed elements."""
    # pylint: disable=import-outside-toplevel
    import builtins
    from pydoppelgangerhunt import config
    from pydoppelgangerhunt.coverage import check_asymmetric_coverage, compute_unit_coverage

    toml_file = tmp_path / "extended.toml"
    toml_content = (
        "[tool.pydoppelgangerhunt]\n"
        "single_quoted = 'literal_value'\n"
        'double_quoted = "standard_value"\n'
        'exclude = ["dist", \'build\', "venv"]\n'
        'mixed_list = [1, 2.5, true, false, "alpha", \'beta\']\n'
        "flag_true = true\n"
        "flag_false = false\n"
        "int_val = 42\n"
        "float_val = 3.14\n"
    )
    toml_file.write_text(toml_content, encoding="utf-8")

    # 1. Test via standard parser
    std_cfg = config.load_toml_section(toml_file, "pydoppelgangerhunt")
    assert std_cfg.get("single_quoted") == "literal_value"
    assert std_cfg.get("double_quoted") == "standard_value"
    assert "dist" in std_cfg.get("exclude", [])

    # 2. Test via zero-dependency line parser fallback
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomli", "tomllib"):
            raise ImportError("mocked no toml parser")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    line_cfg = config.load_toml_section(toml_file, "pydoppelgangerhunt")
    assert line_cfg.get("single_quoted") == "literal_value"
    assert line_cfg.get("double_quoted") == "standard_value"
    assert line_cfg.get("exclude") == ["dist", "build", "venv"]
    assert line_cfg.get("mixed_list") == [1, 2.5, True, False, "alpha", "beta"]
    assert line_cfg.get("flag_true") is True
    assert line_cfg.get("flag_false") is False
    assert line_cfg.get("int_val") == 42
    assert line_cfg.get("float_val") == 3.14

    # 3. Coverage helper edge cases
    assert compute_unit_coverage({}, {"mod.py": {1, 2}}) == 0.0
    assert compute_unit_coverage({"file": "mod.py", "start": 1, "end": 5}, {}) == 0.0
    assert compute_unit_coverage({"file": "mod.py", "start": 1, "end": 5}, {"other.py": {1}}) == 0.0

    u_a = {"file": "mod.py", "start": 1, "end": 10}
    u_b = {"file": "mod.py", "start": 1, "end": 10}
    cov_data = {"mod.py": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
    assert check_asymmetric_coverage(u_a, u_b, cov_data, min_diff=0.40) is None


def test_batch_53_cli_method_binding_and_repo_root(tmp_path: Path) -> None:
    """Batch 53: Test CLI --method-binding, repo_root forwarding, and metrics clamping."""
    # pylint: disable=import-outside-toplevel
    from unittest import mock
    import pytest
    from pydoppelgangerhunt.cli import _audit_clone_risk_warnings, build_arg_parser
    from pydoppelgangerhunt.fixer import _is_same_file_path
    from pydoppelgangerhunt.metrics import compute_repository_dry_stats
    from pydoppelgangerhunt.reporters import extract_unit_source_code

    # 1. Test CLI argument parser for --method-binding
    parser = build_arg_parser()
    args_auto = parser.parse_args(["target_dir", "--method-binding", "auto"])
    assert args_auto.method_binding == "auto"
    args_method = parser.parse_args(["target_dir", "--method-binding", "method"])
    assert args_method.method_binding == "method"
    args_module = parser.parse_args(["target_dir", "--method-binding", "module"])
    assert args_module.method_binding == "module"
    with pytest.raises(SystemExit):
        parser.parse_args(["target_dir", "--method-binding", "invalid_mode"])

    # 2. Test CLI execution with --method-binding and repo_root forwarding
    sub_repo = tmp_path / "sub_repo"
    sub_repo.mkdir()
    code_a = (
        "def compute_alpha(x, y):\n"
        "    res = x * 2 + y * 3\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    code_b = (
        "def compute_beta(x, y):\n"
        "    res = x * 2 + y * 3\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    (sub_repo / "mod_a.py").write_text(code_a, encoding="utf-8")
    (sub_repo / "mod_b.py").write_text(code_b, encoding="utf-8")

    patch_file = tmp_path / "refactor.patch"
    cli_code = pydoppelgangerhunt.main([
        str(sub_repo),
        "--threshold", "0.80",
        "--min-lines", "4",
        "--method-binding", "module",
        "--patch", str(patch_file),
        "--suggest",
        "--diff",
    ])
    assert cli_code == 1
    assert patch_file.exists()

    # 3. Test extract_unit_source_code with relative paths and repo_root
    unit_rel = {"file": "mod_a.py", "start": 1, "end": 4, "name": "compute_alpha"}
    src_lines = extract_unit_source_code(unit_rel, repo_root=str(sub_repo))
    assert len(src_lines) == 4
    assert "compute_alpha" in src_lines[0]

    # Test that existing file in CWD is contained and does not escape repo_root
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        unit_cwd = {"file": "pyproject.toml", "start": 1, "end": 2, "name": "root"}
        cwd_lines = extract_unit_source_code(unit_cwd, repo_root=str(sub_repo))
        assert cwd_lines == ["# Source for root lines 1-2\n"]

    # 4. Test _is_same_file_path with repo_root
    assert _is_same_file_path("mod_a.py", "mod_a.py", repo_root=str(sub_repo))
    assert not _is_same_file_path("mod_a.py", "mod_b.py", repo_root=str(sub_repo))

    # 5. Test _audit_clone_risk_warnings forwarding repo_root
    with mock.patch("pydoppelgangerhunt.cli.check_temporal_divergence") as mock_div:
        mock_div.return_value = {"divergence_days": 120.0}
        warns = _audit_clone_risk_warnings(
            unit_rel, unit_rel, audit_blame=True, repo_root=str(sub_repo)
        )
        assert len(warns) == 1
        assert "Divergent clone risk" in warns[0]
        mock_div.assert_called_once_with(unit_rel, unit_rel, repo_root=str(sub_repo))

    # 6. Test defensive clamping in compute_repository_dry_stats (dloc > sloc)
    huge_clone = (
        0.95,
        {"file": str(sub_repo / "mod_a.py"), "start": 1, "end": 500},
        {"file": str(sub_repo / "mod_b.py"), "start": 1, "end": 500},
    )
    dry_stats = compute_repository_dry_stats(str(sub_repo), [huge_clone])
    assert dry_stats["duplication_pct"] == 100.0
    assert dry_stats["dry_score"] == 0.0
    assert dry_stats["grade"] == "F"


def test_batch_55_baseline_prune_repo_root_and_html_reporter(tmp_path: Path) -> None:
    """Batch 55: Test prune_baseline repo_root, generate_html_report repo_root, and deque AST walk."""
    # pylint: disable=import-outside-toplevel
    import ast
    from unittest import mock
    from pydoppelgangerhunt.baseline import prune_baseline, record_baseline
    from pydoppelgangerhunt.parser import _walk_ast_nodes
    from pydoppelgangerhunt.reporters import generate_html_report

    # 1. Test _walk_ast_nodes with deque BFS traversal
    sample_ast = ast.parse("def sample(a):\n    if a > 0:\n        return a * 2\n    return 0\n")
    walked_nodes = _walk_ast_nodes(sample_ast)
    assert len(walked_nodes) > 5
    assert isinstance(walked_nodes[0], ast.Module)
    assert isinstance(walked_nodes[1], ast.FunctionDef)

    # 2. Test prune_baseline forwarding repo_root to get_git_modified_line_ranges
    sub_repo = tmp_path / "batch55_sub"
    sub_repo.mkdir()
    f1 = sub_repo / "src_a.py"
    f2 = sub_repo / "src_b.py"
    code1 = (
        "def calc_one(x, y):\n"
        "    res = x * 10 + y * 20\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    code2 = (
        "def calc_two(x, y):\n"
        "    res = x * 10 + y * 20\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "src_a.py", "start": 1, "end": 4, "name": "calc_one", "tokens": ["def", "calc_one"]}
    u2 = {"file": "src_b.py", "start": 1, "end": 4, "name": "calc_two", "tokens": ["def", "calc_two"]}
    clones = [(1.0, u1, u2)]

    base_file = sub_repo / "baseline.json"
    record_baseline(clones, str(base_file), str(sub_repo), 0.90)
    assert base_file.exists()

    with mock.patch("pydoppelgangerhunt.git_diff.get_git_modified_line_ranges") as mock_git:
        mock_git.return_value = {"src_a.py": [(1, 4)]}
        res = prune_baseline(str(base_file), active_clones=[], repo_root=str(sub_repo))
        assert res.skipped_dirty_count == 1
        assert res.retained_count == 1
        mock_git.assert_called_once_with(since_ref=None, repo_root=str(sub_repo))

    # Test prune_baseline fallback when git function raises TypeError
    with mock.patch("pydoppelgangerhunt.git_diff.get_git_modified_line_ranges") as mock_git_err:
        def _err_call(since_ref: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            if "repo_root" in kwargs:
                raise TypeError("unexpected keyword argument 'repo_root'")
            return {}
        mock_git_err.side_effect = _err_call
        res_fallback = prune_baseline(str(base_file), active_clones=[], repo_root=str(sub_repo))
        assert res_fallback.pruned_count == 1
        assert res_fallback.retained_count == 0

    # 3. Test generate_html_report with explicit repo_root and target subfolder
    record_baseline(clones, str(base_file), str(sub_repo), 0.90)
    html_out = generate_html_report(clones, target=str(sub_repo), threshold=0.90, repo_root=str(sub_repo))
    assert "calc_one" in html_out
    assert "calc_two" in html_out
    assert "res ** 2" in html_out

    # 4. Test CLI integration with --prune-baseline and --html forwarding target_repo_root
    html_file = sub_repo / "report.html"
    cli_code = pydoppelgangerhunt.main([
        str(sub_repo),
        "--threshold", "0.80",
        "--min-lines", "3",
        "--min-tokens", "5",
        "--html", str(html_file),
    ])
    assert cli_code == 1
    assert html_file.exists()
    html_content = html_file.read_text(encoding="utf-8")
    assert "calc_one" in html_content
    assert "calc_two" in html_content

    # 5. Test CLI prune-baseline with dirty unstaged check
    f2.write_text("def different_function():\n    return 42\n", encoding="utf-8")
    norm_f1 = str(f1.resolve()).replace("\\", "/")
    with mock.patch("pydoppelgangerhunt.git_diff.get_git_modified_line_ranges") as mock_git_cli:
        mock_git_cli.return_value = {norm_f1: [(1, 4)]}
        cli_prune_code = pydoppelgangerhunt.main([
            str(sub_repo),
            "--threshold", "0.80",
            "--min-lines", "3",
            "--min-tokens", "5",
            "--baseline", str(base_file),
            "--prune-baseline",
        ])
        assert cli_prune_code == 0


def test_batch_56_target_dir_config_and_pruneresult_export(tmp_path: Path) -> None:
    """Batch 56: Test PruneResult public export and target directory config discovery."""
    # pylint: disable=import-outside-toplevel
    from unittest import mock
    from pydoppelgangerhunt import PruneResult
    from pydoppelgangerhunt.baseline import PruneResult as BasePruneResult

    # 1. Test PruneResult export and attribute behavior
    assert PruneResult is BasePruneResult
    res = PruneResult(pruned_count=3, retained_count=7, skipped_dirty_count=2)
    assert res.pruned_count == 3
    assert res.retained_count == 7
    assert res.skipped_dirty_count == 2
    # 2-tuple unpacking backward compatibility
    pruned, retained = res
    assert pruned == 3
    assert retained == 7

    # 2. Test CLI automatic config discovery from target directory
    sub_proj = tmp_path / "sub_project_cfg"
    sub_proj.mkdir()
    (sub_proj / "pyproject.toml").write_text(
        """
[tool.pydoppelgangerhunt]
threshold = 0.77
min_lines = 11
""",
        encoding="utf-8",
    )
    (sub_proj / "dummy.py").write_text("x = 1\n", encoding="utf-8")

    # Run CLI without --config pointing to sub_proj
    # Should automatically discover pyproject.toml in sub_proj
    with mock.patch("pydoppelgangerhunt.cli.scan_target") as mock_scan:
        mock_scan.return_value = []
        exit_code = pydoppelgangerhunt.main([str(sub_proj)])
        assert exit_code == 0
        mock_scan.assert_called_once()
        _, kwargs = mock_scan.call_args
        assert kwargs.get("threshold") == 0.77
        assert kwargs.get("min_lines") == 11


def test_batch_67_cli_and_fixer_decomposition(tmp_path: Path) -> None:
    """Batch 67: Test extracted helper functions in cli.py and fixer.py."""
    # pylint: disable=import-outside-toplevel
    import argparse
    from unittest import mock
    from pydoppelgangerhunt.cli import (
        _format_scan_mode_description,
        _render_dry_scorecard,
        _render_pair_diff_and_suggestions,
        _run_type4_semantic_audit,
    )
    from pydoppelgangerhunt.fixer import _format_helper_parameters

    # 1. Test _format_scan_mode_description
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-only", action="store_true")
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--complex-expressions", action="store_true")
    parser.add_argument("--clause-level", action="store_true")
    parser.add_argument("--data-tables", action="store_true")
    parser.add_argument("--class-level", action="store_true")
    parser.add_argument("--merge-subtrees", action="store_true")
    parser.add_argument("--blind-indexing", action="store_true")
    parser.add_argument("--blind-literals", action="store_true")
    parser.add_argument("--bag-of-tokens", action="store_true")
    parser.add_argument("--filter-boilerplate", action="store_true")
    parser.add_argument("--consistent-renaming", action="store_true")
    parser.add_argument("--tfidf", action="store_true")
    parser.add_argument("--harvest-closures", action="store_true")
    parser.add_argument("--commutative", action="store_true")
    parser.add_argument("--comprehensions", action="store_true")
    parser.add_argument("--abstract-expressions", action="store_true")
    parser.add_argument("--gapped-tolerance", action="store_true")
    parser.add_argument("--nms", action="store_true")
    args = parser.parse_args([
        "--sliding-window",
        "--comprehensions",
        "--nms",
    ])
    desc = _format_scan_mode_description(
        args,
        strip_annotations=True,
        strip_docstrings=False,
        idioms_enabled=True,
        call_seq_enabled=False,
        audit_tests_enabled=True,
        stop_shingles_enabled=False,
    )
    assert "functions + compound blocks" in desc
    assert "sliding windows" in desc
    assert "comprehensions" in desc
    assert "untyped" in desc
    assert "idioms canonicalized" in desc
    assert "audit tests" in desc
    assert "nms suppressed" in desc

    # 2. Test _render_dry_scorecard
    stats_good = {
        "grade": "A+",
        "dry_score": 98.5,
        "sloc": 1200,
        "dloc": 10,
        "duplication_pct": 0.83,
        "clone_pairs": 1,
        "clone_families": 1,
    }
    _render_dry_scorecard(stats_good, use_color=True)
    _render_dry_scorecard(stats_good, use_color=False)

    stats_poor = {
        "grade": "D",
        "dry_score": 55.0,
        "sloc": 1000,
        "dloc": 450,
        "duplication_pct": 45.0,
        "clone_pairs": 10,
        "clone_families": 4,
    }
    _render_dry_scorecard(stats_poor, use_color=False)

    # Empty stats defensive defaults
    _render_dry_scorecard({}, use_color=False)

    # 3. Test _render_pair_diff_and_suggestions
    dummy_u1 = {"file": "mod1.py", "start": 1, "end": 4, "name": "fn1", "source": "def fn1(): pass\n"}
    dummy_u2 = {"file": "mod2.py", "start": 1, "end": 4, "name": "fn2", "source": "def fn2(): pass\n"}
    pair_parser = argparse.ArgumentParser()
    pair_parser.add_argument("--suggest", action="store_true")
    pair_parser.add_argument("--diff", action="store_true")
    pair_args = pair_parser.parse_args(["--suggest", "--diff"])

    with mock.patch("pydoppelgangerhunt.cli.synthesize_refactoring_suggestion", return_value="suggestion block"), \
         mock.patch("pydoppelgangerhunt.cli.generate_clone_diff", return_value="--- diff block"):
        rendered_lines = _render_pair_diff_and_suggestions(
            dummy_u1,
            dummy_u2,
            args=pair_args,
            target_repo_root=str(tmp_path),
            use_color=False,
            indent="  ",
        )
        assert len(rendered_lines) == 2
        assert "suggestion block" in rendered_lines[0]
        assert "diff block" in rendered_lines[1]

    # When both suggest and diff are disabled
    no_args = pair_parser.parse_args([])
    assert _render_pair_diff_and_suggestions(
        dummy_u1,
        dummy_u2,
        args=no_args,
        target_repo_root=str(tmp_path),
        use_color=False,
    ) == []

    # 4. Test _run_type4_semantic_audit
    # Missing module
    assert not _run_type4_semantic_audit("target", [], strict_type4=True, use_color=False)

    # Mock module present with violations
    mock_mod = mock.MagicMock()
    mock_mod.find_semantic_clones.return_value = ["violation"]
    mock_mod.report_semantic_results.return_value = True
    with mock.patch.dict("sys.modules", {"check_semantic_clones": mock_mod}):
        # strict_type4=True returns True on violations
        assert _run_type4_semantic_audit("target", [], strict_type4=True, use_color=False)
        # strict_type4=False returns False even with violations
        assert not _run_type4_semantic_audit("target", [], strict_type4=False, use_color=False)

    # 5. Test _format_helper_parameters from fixer.py
    # Ordering: self/cls -> pos -> vararg -> kwonly -> kwarg
    inputs = ["kw", "args", "kwargs", "x", "self"]
    meta1 = {
        "self": {"name": "self", "type": "Any", "default": None, "kind": "pos"},
        "x": {"name": "x", "type": "int", "default": "0", "kind": "pos"},
        "args": {"name": "*args", "type": "Any", "default": None, "kind": "vararg"},
        "kw": {"name": "kw", "type": "str", "default": None, "kind": "kwonly"},
        "kwargs": {"name": "**kwargs", "type": "Any", "default": None, "kind": "kwarg"},
    }
    meta2 = dict(meta1)
    params = _format_helper_parameters(
        inputs,
        meta1,
        meta2,
        effective_binding="method",
        is_static_clone=False,
        is_class_receiver=False,
        type_merge_strategy="fallback_any",
    )
    assert params[0] == "self"
    assert params[1] == "x: int = 0"
    assert params[2] == "*args: Any"
    assert params[3] == "kw: str"
    assert params[4] == "**kwargs: Any"

    # Test receiver parameter injection when absent in method binding
    inputs_no_receiver = ["a", "b"]
    meta_no_rec = {
        "a": {"name": "a", "type": "int", "default": None, "kind": "pos"},
        "b": {"name": "b", "type": "int", "default": None, "kind": "pos"},
    }
    params_injected_cls = _format_helper_parameters(
        inputs_no_receiver,
        meta_no_rec,
        meta_no_rec,
        effective_binding="method",
        is_static_clone=False,
        is_class_receiver=True,
        type_merge_strategy="fallback_any",
    )
    assert params_injected_cls[0] == "cls"
    assert params_injected_cls[1] == "a: int"
    assert params_injected_cls[2] == "b: int"

    # Test positional argument default invalidation (non-default follows default)
    meta_invalid_order = {
        "p1": {"name": "p1", "type": "int", "default": "10", "kind": "pos"},
        "p2": {"name": "p2", "type": "int", "default": None, "kind": "pos"},
    }
    params_reset_defaults = _format_helper_parameters(
        ["p1", "p2"],
        meta_invalid_order,
        meta_invalid_order,
        effective_binding="module",
        is_static_clone=False,
        is_class_receiver=False,
        type_merge_strategy="fallback_any",
    )
    assert params_reset_defaults[0] == "p1: int"
    assert params_reset_defaults[1] == "p2: int"

    # Test kwonly marker '*' insertion when no *args is present
    meta_kwonly = {
        "k1": {"name": "k1", "type": "int", "default": "1", "kind": "kwonly"},
    }
    params_kwonly = _format_helper_parameters(
        ["k1"],
        meta_kwonly,
        meta_kwonly,
        effective_binding="module",
        is_static_clone=False,
        is_class_receiver=False,
        type_merge_strategy="fallback_any",
    )
    assert params_kwonly == ["*", "k1: int = 1"]


def test_batch_68_cli_patch_replace_and_type_merge_strategy(tmp_path: Path) -> None:
    """Batch 68: Test CLI --replace-clones and --type-merge-strategy flags and config propagation."""
    # pylint: disable=import-outside-toplevel
    from unittest import mock
    from pydoppelgangerhunt.cli import build_arg_parser

    # 1. Test argument parser
    parser = build_arg_parser()
    args_def = parser.parse_args(["target_dir"])
    assert args_def.replace_clones is False
    assert args_def.type_merge_strategy is None

    args_custom = parser.parse_args([
        "target_dir",
        "--replace-clones",
        "--type-merge-strategy",
        "union",
    ])
    assert args_custom.replace_clones is True
    assert args_custom.type_merge_strategy == "union"

    # 2. Test CLI forwarding to generate_refactoring_patch
    sub_repo = tmp_path / "sub_repo_68"
    sub_repo.mkdir()
    code_a = (
        "def compute_1(x: int, y: int) -> int:\n"
        "    res = x * 10 + y * 20\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    code_b = (
        "def compute_2(x: float, y: float) -> float:\n"
        "    res = x * 10 + y * 20\n"
        "    val = res ** 2\n"
        "    return val + 1\n"
    )
    (sub_repo / "mod_a.py").write_text(code_a, encoding="utf-8")
    (sub_repo / "mod_b.py").write_text(code_b, encoding="utf-8")

    patch_file = tmp_path / "replace.patch"
    with mock.patch("pydoppelgangerhunt.cli.generate_refactoring_patch") as mock_patch:
        mock_patch.return_value = "patch content"
        exit_code = pydoppelgangerhunt.main([
            str(sub_repo),
            "--threshold", "0.80",
            "--min-lines", "4",
            "--patch", str(patch_file),
            "--replace-clones",
            "--type-merge-strategy", "union",
        ])
        assert exit_code == 1
        mock_patch.assert_called_once()
        _, kwargs = mock_patch.call_args
        assert kwargs.get("replace_clones") is True
        assert kwargs.get("type_merge_strategy") == "union"
        assert kwargs.get("method_binding") == "auto"
        assert kwargs.get("cross_file_strategy") == "auto"
        assert kwargs.get("shared_module_name") == "_common.py"

    # 3. Test CLI explicit cross-file flags
    with mock.patch("pydoppelgangerhunt.cli.generate_refactoring_patch") as mock_patch_custom:
        mock_patch_custom.return_value = "custom patch"
        exit_code = pydoppelgangerhunt.main([
            str(sub_repo),
            "--threshold", "0.80",
            "--min-lines", "4",
            "--patch", str(patch_file),
            "--cross-file-strategy", "host_module",
            "--shared-module-name", "_shared_helpers.py",
        ])
        assert exit_code == 1
        mock_patch_custom.assert_called_once()
        _, kwargs_custom = mock_patch_custom.call_args
        assert kwargs_custom.get("cross_file_strategy") == "host_module"
        assert kwargs_custom.get("shared_module_name") == "_shared_helpers.py"

    # 4. Test CLI explicit cross-file skip flag
    with mock.patch("pydoppelgangerhunt.cli.generate_refactoring_patch") as mock_patch_skip:
        mock_patch_skip.return_value = "skip patch"
        exit_code = pydoppelgangerhunt.main([
            str(sub_repo),
            "--threshold", "0.80",
            "--min-lines", "4",
            "--patch", str(patch_file),
            "--cross-file-strategy", "skip",
        ])
        assert exit_code == 1
        mock_patch_skip.assert_called_once()
        _, kwargs_skip = mock_patch_skip.call_args
        assert kwargs_skip.get("cross_file_strategy") == "skip"

    # 5. Test config file defaults when flags omitted
    (sub_repo / "pyproject.toml").write_text(
        """
[tool.pydoppelgangerhunt]
threshold = 0.80
min_lines = 4
replace_clones = true
type_merge_strategy = "union"
cross_file_strategy = "shared_module"
shared_module_name = "_my_common.py"
""",
        encoding="utf-8",
    )
    with mock.patch("pydoppelgangerhunt.cli.generate_refactoring_patch") as mock_patch_cfg:
        mock_patch_cfg.return_value = "patch cfg content"
        exit_code = pydoppelgangerhunt.main([
            str(sub_repo),
            "--patch", str(patch_file),
        ])
        assert exit_code == 1
        mock_patch_cfg.assert_called_once()
        _, kwargs_cfg = mock_patch_cfg.call_args
        assert kwargs_cfg.get("replace_clones") is True
        assert kwargs_cfg.get("type_merge_strategy") == "union"
        assert kwargs_cfg.get("cross_file_strategy") == "shared_module"
        assert kwargs_cfg.get("shared_module_name") == "_my_common.py"


def test_batch_69_modular_decomposition(tmp_path: Path) -> None:
    """Batch 69: Validate extracted single-responsibility helpers in cli.py and fixer.py."""
    # pylint: disable=import-outside-toplevel
    import argparse
    from unittest import mock
    from pydoppelgangerhunt.cli import (
        _apply_baseline_and_diff_filters,
        _render_text_violations,
    )
    from pydoppelgangerhunt.fixer import (
        _infer_helper_return_type,
        _format_helper_docstring,
        _build_unit_delegation_call,
    )

    # 1. Test _apply_baseline_and_diff_filters
    sample_clones = [
        (0.95, {"file": "mod1.py", "start": 1, "end": 10, "name": "fn1"}, {"file": "mod2.py", "start": 1, "end": 10, "name": "fn2"})
    ]
    # Prune without baseline path -> exit code 1
    args_err1 = argparse.Namespace(
        prune_baseline=True, baseline=None, diff_only=False, format="text"
    )
    clones_res, exit_code = _apply_baseline_and_diff_filters(
        sample_clones, args_err1, {}, "."
    )
    assert exit_code == 1

    # Prune with missing file -> exit code 1
    args_err2 = argparse.Namespace(
        prune_baseline=True, baseline=str(tmp_path / "nonexistent.json"), diff_only=False, format="text"
    )
    _, exit_code = _apply_baseline_and_diff_filters(
        sample_clones, args_err2, {}, "."
    )
    assert exit_code == 1

    # Normal baseline loading & suppression
    base_file = tmp_path / "base.json"
    base_file.write_text("{}", encoding="utf-8")
    args_normal = argparse.Namespace(
        prune_baseline=False,
        baseline=str(base_file),
        diff_only=False,
        format="text",
    )
    with mock.patch("pydoppelgangerhunt.cli.load_baseline", return_value=set()):
        with mock.patch("pydoppelgangerhunt.cli.filter_clones_by_baseline", return_value=(sample_clones, 0)):
            res_clones, early_exit = _apply_baseline_and_diff_filters(
                sample_clones, args_normal, {}, "."
            )
            assert early_exit is None
            assert len(res_clones) == 1

    # 2. Test _render_text_violations (families and pairwise)
    u1 = {"file": "test_foo.py", "start": 1, "end": 5, "name": "test_one"}
    u2 = {"file": "test_bar.py", "start": 1, "end": 5, "name": "test_two"}
    clones = [(0.90, u1, u2)]

    # Flat clone report
    args_flat = argparse.Namespace(cluster=False, blame=False, suggest=False, diff=False)
    lines_flat = _render_text_violations(
        clones, 0.85, "priority", args=args_flat, audit_tests_enabled=True
    )
    assert any("[Priority" in ln for ln in lines_flat)
    assert any("@pytest.mark.parametrize" in ln for ln in lines_flat)
    assert any("Please refactor structural duplicates" in ln for ln in lines_flat)

    # Clustered family report
    fam = {
        "family_id": "fam_1",
        "member_count": 2,
        "avg_similarity": 0.90,
        "max_similarity": 0.90,
        "coherence": 1.0,
        "total_lines": 10,
        "unique_files": ["test_foo.py", "test_bar.py"],
        "medoid": u1,
        "members": [u1, u2],
    }
    args_cluster = argparse.Namespace(cluster=True, blame=False, suggest=False, diff=False)
    lines_fam = _render_text_violations(
        clones, 0.85, "similarity", args=args_cluster, families=[fam]
    )
    assert any("[fam_1]" in ln for ln in lines_fam)
    assert any("[medoid]" in ln for ln in lines_fam)

    # 3. Test _infer_helper_return_type
    # Async generator
    ret_async_gen = _infer_helper_return_type(
        "Any", [], set(), {"has_yield": True}, {}, {}, is_async=True
    )
    assert ret_async_gen == "AsyncIterator[Any]"

    # Sync generator with inferred item type
    ret_sync_gen = _infer_helper_return_type(
        "Any",
        [],
        set(),
        {"has_yield": True, "yield_expr_names": [("yield", "item")]},
        {"item": {"type": "int"}},
        {},
        is_async=False,
    )
    assert ret_sync_gen == "Iterator[int]"

    # Sync generator with yield from List[str]
    ret_sync_yf = _infer_helper_return_type(
        "Any",
        [],
        set(),
        {"has_yield": True, "yield_expr_names": [("yield_from", "items")]},
        {"items": {"type": "List[str]"}},
        {},
        is_async=False,
    )
    assert ret_sync_yf == "Iterator[str]"

    # Multi-output tuple with conditional output
    ret_tuple = _infer_helper_return_type(
        "Any",
        ["a", "b"],
        {"b"},
        {},
        {"a": {"type": "int"}, "b": {"type": "str"}},
        {},
    )
    assert ret_tuple == "Tuple[int, Optional[str]]"

    # Single output
    ret_single = _infer_helper_return_type(
        "Any",
        ["res"],
        set(),
        {},
        {"res": {"type": "float"}},
        {},
    )
    assert ret_single == "float"

    # Comprehension fallback
    ret_comp = _infer_helper_return_type(
        "Any", [], set(), {"has_return": False}, {}, {}, unit_kind="comprehension"
    )
    assert ret_comp == "Any"

    # Void fallback
    ret_void = _infer_helper_return_type(
        "Any", [], set(), {"has_return": False}, {}, {}, unit_kind="block"
    )
    assert ret_void == "None"

    # 4. Test _format_helper_docstring
    # Async generator call site
    doc_async = _format_helper_docstring(
        "helper", [], {"has_yield": True}, is_async=True
    )
    assert "async for _item in helper(...):" in doc_async

    # Sync generator call site with assignment
    doc_sync = _format_helper_docstring(
        "helper", ["res1", "res2"], {"has_yield": True}, is_async=False
    )
    assert "res1, res2 = (yield from helper(...))" in doc_sync

    # Hazard warning
    doc_hazard = _format_helper_docstring(
        "helper", [], {"control_flow_hazards": ["naked_break"]}
    )
    assert "WARNING: Non-local control flow hazard detected" in doc_hazard

    # 5. Test _build_unit_delegation_call
    orig_code = (
        "class MyService:\n"
        "    def do_work(self, a: int) -> int:\n"
        "        x = a * 2\n"
        "        return x\n"
    )
    orig_lines = orig_code.splitlines(keepends=True)
    target_unit = {"kind": "block", "start": 3, "end": 3}
    scope_inst = {
        "inputs": ["self", "a"],
        "outputs": ["x"],
        "param_details": [{"name": "self"}, {"name": "a"}],
    }
    call_inst = _build_unit_delegation_call(
        target_unit,
        orig_code,
        orig_lines,
        effective_binding="method",
        helper_name="_shared_calc",
        inputs=["self", "a"],
        outputs=["x"],
        scope=scope_inst,
    )
    assert "x = self._shared_calc(a)" in call_inst

    # Comprehension call
    target_comp = {"kind": "comprehension", "start": 3, "end": 3}
    call_comp = _build_unit_delegation_call(
        target_comp,
        orig_code,
        orig_lines,
        effective_binding="module",
        helper_name="_shared_comp",
        inputs=["a"],
        outputs=[],
        scope=scope_inst,
    )
    assert call_comp == "_shared_comp(a)"


def test_toml_array_value_nested_2d_arrays(tmp_path: Path) -> None:
    """Verifies that _parse_toml_array_value correctly parses nested 2D TOML arrays and exemptions."""
    from pydoppelgangerhunt.config import _parse_toml_array_value, load_toml_section  # pylint: disable=import-outside-toplevel

    # 1. 2D array of strings
    raw_2d = '[["pkg/a.py:f", "pkg/b.py:g"], ["pkg/c.py:h", "pkg/d.py:k"]]'
    parsed_2d = _parse_toml_array_value(raw_2d)
    assert parsed_2d == [["pkg/a.py:f", "pkg/b.py:g"], ["pkg/c.py:h", "pkg/d.py:k"]]

    # 2. Mixed types in nested arrays
    raw_mixed = '[["a", 1, True], ["b", 2.5, False]]'
    parsed_mixed = _parse_toml_array_value(raw_mixed)
    assert parsed_mixed == [["a", 1, True], ["b", 2.5, False]]

    # 3. Via load_toml_section fallback parser
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[tool.pydoppelgangerhunt]\n'
        'exemptions = [["pkg/a.py:func1", "pkg/b.py:func2"]]\n',
        encoding="utf-8",
    )
    data = load_toml_section(cfg_file, "pydoppelgangerhunt")
    assert "exemptions" in data
    assert data["exemptions"] == [["pkg/a.py:func1", "pkg/b.py:func2"]]


def test_cli_replace_clones_without_patch_warning(tmp_path: Path, capsys: Any) -> None:
    """Verifies that CLI emits a friendly warning when --replace-clones is passed without --patch."""
    f = tmp_path / "target.py"
    f.write_text("x = 1\n", encoding="utf-8")

    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    code = main([str(f), "--replace-clones"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Warning: --replace-clones specified without --patch; no patch will be generated." in captured.out


def test_load_toml_section_multiline_nested_arrays_fallback(tmp_path: Path) -> None:
    """Verifies that the zero-dependency TOML fallback parser tracks bracket depth across multiline 2D arrays."""
    import builtins  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.config import load_toml_section  # pylint: disable=import-outside-toplevel

    cfg = tmp_path / "pyproject.toml"
    cfg.write_text(
        "[tool.pydoppelgangerhunt]\n"
        "exemptions = [\n"
        '    ["pkg/a.py:func1", "pkg/b.py:func2"],\n'
        '    ["pkg/c.py:func3", "pkg/d.py:func4"]\n'
        "]\n",
        encoding="utf-8",
    )

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomllib", "tomli"):
            raise ImportError("Simulated missing toml library")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        data = load_toml_section(cfg, "pydoppelgangerhunt")
        assert data["exemptions"] == [
            ["pkg/a.py:func1", "pkg/b.py:func2"],
            ["pkg/c.py:func3", "pkg/d.py:func4"],
        ]


def test_toml_line_parser_escaped_quote_inline_comment(tmp_path: Path) -> None:
    """Verifies that escaped quotes in strings don't cause trailing text to be misidentified as inline comments."""
    import builtins  # pylint: disable=import-outside-toplevel
    from unittest import mock  # pylint: disable=import-outside-toplevel

    cfg = tmp_path / "pyproject.toml"
    cfg.write_text(
        '[tool.pydoppelgangerhunt]\n'
        'pattern = "hello \\" # not a comment"\n'
        'threshold = 0.85 # this is a comment\n',
        encoding="utf-8",
    )

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("tomllib", "tomli"):
            raise ImportError("Simulated missing toml library")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        data = load_toml_section(cfg, "pydoppelgangerhunt")
        assert data["pattern"] == 'hello \\" # not a comment'
        assert data["threshold"] == 0.85


def test_load_toml_section_non_mapping_resilience(tmp_path: Path) -> None:
    """Verifies that non-dict or malformed TOML sections do not raise TypeError and return empty dict."""
    cfg = tmp_path / "pyproject.toml"
    cfg.write_text(
        '[tool]\n'
        'pydoppelgangerhunt = 1\n',
        encoding="utf-8",
    )
    assert not load_toml_section(cfg, "pydoppelgangerhunt")

    cfg.write_text(
        '[tool]\n'
        'pydoppelgangerhunt = "scalar_value"\n',
        encoding="utf-8",
    )
    assert not load_toml_section(cfg, "pydoppelgangerhunt")

    cfg.write_text(
        'tool = 42\n',
        encoding="utf-8",
    )
    assert not load_toml_section(cfg, "pydoppelgangerhunt")


def test_parse_toml_array_value_escaped_quotes_and_commas() -> None:
    """Verifies that _parse_toml_array_value handles escaped quotes containing commas inside string elements."""
    from pydoppelgangerhunt.config import _parse_toml_array_value  # pylint: disable=import-outside-toplevel

    raw = '["hello \\"world\\", here", "item2", \'another \\\'escaped\\\', comma\']'
    parsed = _parse_toml_array_value(raw)
    assert parsed == ['hello "world", here', 'item2', "another 'escaped', comma"]


def test_cli_record_baseline_and_differential_scan_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that main() records baseline with corpus_calibration and passes it during --baseline runs."""
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "cli_repo"
    repo.mkdir()

    # Create dummy python files
    shared = (
        "def compute_total(a, b, c):\n"
        "    res = 0\n"
        "    for val in [a, b, c]:\n"
        "        if val > 0:\n"
        "            res += val * 2\n"
        "        else:\n"
        "            res -= val\n"
        "    return res\n"
    )
    (repo / "m1.py").write_text(shared, encoding="utf-8")
    (repo / "m2.py").write_text(shared, encoding="utf-8")

    baseline_json = tmp_path / "baseline.json"

    # Step 1: Record baseline via CLI
    test_args_record = [
        "pydoppelgangerhunt",
        str(repo),
        "--record-baseline",
        str(baseline_json),
        "--threshold",
        "0.90",
        "--min-lines",
        "6",
    ]
    monkeypatch.setattr("sys.argv", test_args_record)
    exit_code_record = main()
    assert exit_code_record == 0
    assert baseline_json.is_file()

    # Verify baseline contents
    loaded = load_baseline(str(baseline_json))
    assert loaded.corpus_calibration is not None
    assert loaded.corpus_calibration["total_units"] >= 2

    # Step 2: Run scan with --baseline to ensure calibration forwarding to scan_target
    import pydoppelgangerhunt.cli as cli_mod  # pylint: disable=import-outside-toplevel
    captured_kwargs: List[Dict[str, Any]] = []
    orig_scan = cli_mod.scan_target

    def spy_scan_target(*args: Any, **kwargs: Any) -> Any:
        captured_kwargs.append(dict(kwargs))
        return orig_scan(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "scan_target", spy_scan_target)

    test_args_scan = [
        "pydoppelgangerhunt",
        str(repo),
        "--baseline",
        str(baseline_json),
        "--threshold",
        "0.90",
        "--min-lines",
        "6",
    ]
    monkeypatch.setattr("sys.argv", test_args_scan)
    exit_code_scan = main()
    assert exit_code_scan == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("corpus_calibration") is not None
    assert captured_kwargs[0]["corpus_calibration"]["total_units"] >= 2

    # Step 3: Run differential scan with --baseline and --diff-only to verify differential forwarding
    captured_kwargs.clear()
    monkeypatch.setattr(
        "pydoppelgangerhunt.cli.get_git_modified_files",
        lambda since_ref=None, repo_root=None: [str(repo / "m1.py")],
    )
    monkeypatch.setattr(
        "pydoppelgangerhunt.cli.get_git_modified_line_ranges",
        lambda since_ref=None, repo_root=None: {str(repo / "m1.py"): [(1, 10)]},
    )
    test_args_diff = [
        "pydoppelgangerhunt",
        str(repo),
        "--baseline",
        str(baseline_json),
        "--diff-only",
        "--threshold",
        "0.90",
        "--min-lines",
        "6",
    ]
    monkeypatch.setattr("sys.argv", test_args_diff)
    exit_code_diff = main()
    assert exit_code_diff == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("corpus_calibration") is not None
    assert captured_kwargs[0]["corpus_calibration"]["total_units"] >= 2
    assert captured_kwargs[0].get("diff_files") is not None
    assert str(repo / "m1.py") in captured_kwargs[0]["diff_files"]


def test_normalize_path_string_nfc_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies normalize_path_string converts macOS NFD decomposed Unicode into canonical NFC while preserving POSIX identity."""
    import unicodedata
    from pydoppelgangerhunt.config import normalize_path_string, paths_match_boundary

    nfc_path = "src/café_module.py"
    nfd_path = unicodedata.normalize("NFD", nfc_path)
    assert nfc_path != nfd_path

    # 1. On macOS (darwin), NFD paths are normalized to canonical NFC to align with Git precomposeunicode
    monkeypatch.setattr("sys.platform", "darwin")
    norm_darwin_nfd = normalize_path_string(nfd_path)
    norm_darwin_nfc = normalize_path_string(nfc_path)
    assert norm_darwin_nfd == norm_darwin_nfc == "src/café_module.py"
    assert paths_match_boundary(nfd_path, nfc_path)
    assert paths_match_boundary(nfc_path, nfd_path)

    # 2. On standard POSIX (linux), distinct composed and decomposed filenames preserve filesystem identity
    monkeypatch.setattr("sys.platform", "linux")
    norm_posix_nfd = normalize_path_string(nfd_path)
    norm_posix_nfc = normalize_path_string(nfc_path)
    assert norm_posix_nfd != norm_posix_nfc
    assert norm_posix_nfd == nfd_path
    assert norm_posix_nfc == nfc_path
    assert not paths_match_boundary(nfd_path, nfc_path)


def test_cli_warns_on_calibration_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that main() emits an advisory warning when scan configuration differs from baseline calibration."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_cfg_mismatch"
    repo.mkdir()
    code = (
        "def sample_func(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    c = b - 3\n"
        "    return c * 4\n"
    )
    (repo / "f1.py").write_text(code, encoding="utf-8")
    baseline_file = tmp_path / "baseline_mismatch.json"

    # 1. Record baseline with min_lines=4
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--record-baseline",
            str(baseline_file),
            "--min-lines",
            "4",
            "--threshold",
            "0.90",
        ],
    )
    assert main() == 0
    capsys.readouterr()

    # 2. Run scan with min_lines=10 (config mismatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "10",
            "--threshold",
            "0.90",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" in out


def test_cli_warns_on_calibration_unit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that main() emits a calibration drift warning when repository units drift >= 20% from baseline."""
    import json  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_unit_drift"
    repo.mkdir()
    code = (
        "def helper_action(val):\n"
        "    res = [x * 2 for x in val if x > 0]\n"
        "    return sum(res)\n"
    )
    (repo / "f1.py").write_text(code, encoding="utf-8")
    baseline_file = tmp_path / "baseline_drift.json"

    # 1. Record baseline with CLI to guarantee identical configuration settings
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--record-baseline",
            str(baseline_file),
            "--threshold",
            "0.90",
            "--min-lines",
            "3",
        ],
    )
    assert main() == 0
    capsys.readouterr()

    # 2. Artificially simulate repository unit drift in the recorded baseline
    raw_data = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert "corpus_calibration" in raw_data
    raw_data["corpus_calibration"]["total_units"] = 100
    baseline_file.write_text(json.dumps(raw_data), encoding="utf-8")

    # 3. Run scan with --baseline (same config, but drifted unit count)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--threshold",
            "0.90",
            "--min-lines",
            "3",
        ],
    )
    exit_code = main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Warning: Calibration drift detected: repository units drifted by" in out


def test_cli_inherits_baseline_min_corpus_size_without_mismatch_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that running --baseline without --min-corpus-units inherits min_corpus_size without spurious warnings."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_inherit_mcs"
    repo.mkdir()
    code = (
        "def compute_delta(x, y):\n"
        "    diff = x - y\n"
        "    return diff * 2 if diff > 0 else 0\n"
    )
    (repo / "f1.py").write_text(code, encoding="utf-8")
    baseline_file = tmp_path / "baseline_mcs.json"

    # 1. Record baseline with --min-corpus-units 50
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--record-baseline",
            str(baseline_file),
            "--min-lines",
            "3",
            "--threshold",
            "0.90",
            "--min-corpus-units",
            "50",
        ],
    )
    assert main() == 0
    capsys.readouterr()

    # 2. Run scan without --min-corpus-units (should inherit from baseline cleanly without config mismatch warning)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "3",
            "--threshold",
            "0.90",
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" not in out


def test_cli_verbose_flag_and_commit_divergence_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that -v/--verbose flag is recognized and reports commit differences without AttributeError."""
    import json  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.cli import build_arg_parser, main  # pylint: disable=import-outside-toplevel
    import pydoppelgangerhunt.cli as cli_mod  # pylint: disable=import-outside-toplevel

    parser = build_arg_parser()
    parsed_v = parser.parse_args(["-v"])
    assert parsed_v.verbose is True
    parsed_verbose = parser.parse_args(["--verbose"])
    assert parsed_verbose.verbose is True
    parsed_none = parser.parse_args([])
    assert parsed_none.verbose is False

    repo = tmp_path / "repo_verbose"
    repo.mkdir(parents=True)
    (repo / "mod.py").write_text("def fn():\n    return 1\n", encoding="utf-8")

    baseline_file = repo / "baseline.json"
    recorded_hash = "1111222233334444555566667777888899990000"
    current_hash = "aaaabbbbccccddddeeeeffff0000111122223333"

    # Record baseline with recorded_commit
    data = {
        "version": "1.5.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "target": str(repo),
        "path_basis": "target_relative",
        "threshold": 0.90,
        "clone_count": 0,
        "recorded_commit": recorded_hash,
        "fingerprints": [],
    }
    baseline_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    monkeypatch.setattr(cli_mod, "get_git_head_commit", lambda repo_root=None: current_hash)

    # 1. Run without -v: should not report commit divergence, should not raise AttributeError
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "3",
        ],
    )
    assert main() == 0
    out_default = capsys.readouterr().out
    assert "differs from baseline recorded commit" not in out_default

    # 2. Run with -v: should report commit divergence
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "3",
            "-v",
        ],
    )
    assert main() == 0
    out_verbose = capsys.readouterr().out
    assert f"Repository HEAD commit {current_hash[:8]} differs from baseline recorded commit {recorded_hash[:8]}" in out_verbose


def test_cli_inherits_all_unspecified_bounds_from_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that cli.main inherits unspecified pruning/harvesting bounds from loaded calibration."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo_bounds"
    repo.mkdir(parents=True)
    code_fn = """
def sample_logic():
    a = 10
    b = 20
    c = 30
    d = 40
    e = 50
    f = 60
    g = 70
    h = 80
    i = 90
    j = 100
    k = 110
    l = 120
    return a + b + c + d + e + f + g + h + i + j + k + l
"""
    (repo / "sample1.py").write_text(code_fn, encoding="utf-8")
    (repo / "sample2.py").write_text(code_fn, encoding="utf-8")

    baseline_file = repo / "baseline.json"

    # 1. Record baseline with non-default bounds: min-lines=12, min-tokens=20, max-index-frequency=0.15, min-corpus-units=50
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--record-baseline",
            str(baseline_file),
            "--min-lines",
            "12",
            "--min-tokens",
            "20",
            "--max-index-frequency",
            "0.15",
            "--min-corpus-units",
            "50",
            "--threshold",
            "0.90",
        ],
    )
    assert main() == 0
    capsys.readouterr()

    # 2. Run scan with ONLY --baseline (no --min-lines, no --min-tokens, no --max-index-frequency, no --min-corpus-units)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--threshold",
            "0.90",
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    # The scan banner must reflect inherited min_lines=12
    assert "min_lines=12" in out
    # Calibration must be accepted without mismatch warning
    assert "Warning: Active scan configuration does not match baseline calibration config" not in out


def test_cli_inherits_representation_and_harvesting_modes_from_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that representation and harvesting modes are inherited from baseline calibration unless overridden."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    pkg_dir = tmp_path / "custom_modes_repo"
    pkg_dir.mkdir(parents=True)
    file_content = (
        "def compute_alpha(vals: list) -> int:\n"
        "    total = 0\n"
        "    for v in vals:\n"
        "        total += v\n"
        "    return total\n"
    )
    (pkg_dir / "mod_a.py").write_text(file_content, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(file_content, encoding="utf-8")

    bl_path = pkg_dir / "baseline.json"

    # Step 1: Record baseline with --idioms, --bag-of-tokens, and --preserve-annotations
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(pkg_dir),
            "--record-baseline",
            str(bl_path),
            "--idioms",
            "--bag-of-tokens",
            "--preserve-annotations",
            "--min-lines",
            "3",
            "--threshold",
            "0.80",
        ],
    )
    exit_code_rec = main()
    assert exit_code_rec == 0
    capsys.readouterr()

    # Step 2: Scan with --baseline only (unspecified --idioms/--bag-of-tokens/--preserve-annotations)
    # Options should be inherited from active calibration, matching hash and avoiding warning
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(pkg_dir),
            "--baseline",
            str(bl_path),
            "--threshold",
            "0.80",
        ],
    )
    exit_code_scan = main()
    assert exit_code_scan == 0
    scan_out = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" not in scan_out
    assert "idioms canonicalized" in scan_out
    assert "bag of tokens" in scan_out
    assert "untyped" not in scan_out  # preserve-annotations was active

    # Step 3: Explicit override (--no-idioms) must trigger configuration mismatch warning
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(pkg_dir),
            "--baseline",
            str(bl_path),
            "--no-idioms",
            "--threshold",
            "0.80",
        ],
    )
    exit_code_override = main()
    assert exit_code_override == 0
    override_out = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" in override_out


def test_cli_inherits_null_max_index_frequency_from_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that max_index_frequency: null in calibration is preserved as None without reverting to 0.25."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_corpus_calibration, record_baseline  # pylint: disable=import-outside-toplevel

    code = (
        "def sample_operation_handler():\n"
        "    val_one = 100\n"
        "    val_two = 200\n"
        "    val_three = 300\n"
        "    return val_one + val_two + val_three\n"
    )
    repo = tmp_path / "null_freq_repo"
    repo.mkdir(parents=True)
    (repo / "comp_a.py").write_text(code, encoding="utf-8")
    (repo / "comp_b.py").write_text(code, encoding="utf-8")

    bl_file = repo / "baseline_null_freq.json"

    # 1. Create and record baseline with max_index_frequency=None (disabled frequency pruning)
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel
    clones = scan_target(str(repo), min_lines=3, threshold=0.80)
    calib = compute_corpus_calibration([], max_index_frequency=None, min_lines=3, min_tokens=15, min_corpus_size=4)
    record_baseline(
        clones,
        str(bl_file),
        str(repo),
        0.80,
        repo_root=str(repo),
        corpus_calibration=calib,
    )

    # 2. Run CLI scan with ONLY --baseline (unspecified max_index_frequency)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(bl_file),
            "--min-lines",
            "3",
            "--threshold",
            "0.80",
        ],
    )
    res = main()
    assert res == 0
    out = capsys.readouterr().out
    # Calibration should be accepted without configuration mismatch warning
    assert "Warning: Active scan configuration does not match baseline calibration config" not in out

    # 3. Explicit override (--max-index-frequency 0.20) should trigger configuration mismatch warning
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(bl_file),
            "--max-index-frequency",
            "0.20",
            "--min-lines",
            "3",
            "--threshold",
            "0.80",
        ],
    )
    res_override = main()
    assert res_override == 0
    out_override = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" in out_override


def test_cli_diff_only_empty_diff_skips_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that --diff-only with an empty diff passes diff_files=[] to scan_target and skips scanning."""
    repo = tmp_path / "repo"
    repo.mkdir()
    f1 = repo / "a.py"
    f2 = repo / "b.py"
    code = "def duplicate():\n    x = 1\n    y = 2\n    return x + y\n"
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(
        "pydoppelgangerhunt.cli._safe_call_git_diff_helper",
        lambda helper, *args, **kwargs: [] if "modified" in helper.__name__ else str(repo),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--diff-only",
            "--threshold",
            "0.80",
            "--min-lines",
            "3",
        ],
    )
    with mock.patch("pydoppelgangerhunt.cli.scan_target", wraps=pydoppelgangerhunt.cli.scan_target) as mock_scan:
        res = main()
        assert res == 0
        mock_scan.assert_called_once()
        _, kwargs = mock_scan.call_args
        assert kwargs.get("diff_files") == []
    out = capsys.readouterr().out
    assert "No structural code clones found" in out


def test_cli_main_passes_notebooks_and_exemptions_to_scan_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that cli.main passes include_notebooks and config exemptions to scan_target."""
    import pydoppelgangerhunt.cli
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "repo"
    repo.mkdir()
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        '[tool.pydoppelgangerhunt]\nexemptions = [["a.py:foo", "b.py:foo"]]\n',
        encoding="utf-8",
    )
    (repo / "dummy.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--notebooks",
        ],
    )
    with mock.patch("pydoppelgangerhunt.cli.scan_target", wraps=pydoppelgangerhunt.cli.scan_target) as mock_scan:
        res = main()
        assert res == 0
        mock_scan.assert_called_once()
        _, kwargs = mock_scan.call_args
        assert kwargs.get("include_notebooks") is True
        assert kwargs.get("exemptions") == [("a.py:foo", "b.py:foo")]


def test_cli_skipped_novel_shingles_verbose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies that cli.main prints an informational notice in text mode when novel shingles exceed budget under -v."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import compute_corpus_calibration, record_baseline  # pylint: disable=import-outside-toplevel
    import pydoppelgangerhunt.matcher

    repo = tmp_path / "repo_novel"
    repo.mkdir(parents=True)
    f = repo / "module.py"
    f.write_text(
        "def f1():\n    x = 1\n    y = 2\n    return x + y\n\n"
        "def f2():\n    x = 1\n    y = 2\n    return x + y\n\n"
        "def f3():\n    x = 1\n    y = 2\n    return x + y\n",
        encoding="utf-8",
    )

    baseline_file = tmp_path / "baseline.json"
    calib = compute_corpus_calibration([], min_lines=3)
    calib["total_units"] = 100
    calib["shingle_frequencies"] = {}
    record_baseline([], str(baseline_file), str(repo), 0.90, corpus_calibration=calib)

    monkeypatch.setattr(pydoppelgangerhunt.matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 1)

    # Run without -v: should NOT print advisory
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "3",
            "--format",
            "text",
            "--no-color",
        ],
    )
    assert main() == 0
    out_default = capsys.readouterr().out
    assert "high-density novel shingle(s) exceeded candidate pair budget" not in out_default

    # Run with -v: should print advisory
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--baseline",
            str(baseline_file),
            "--min-lines",
            "3",
            "--format",
            "text",
            "--no-color",
            "-v",
        ],
    )
    assert main() == 0
    out_verbose = capsys.readouterr().out
    assert "high-density novel shingle(s) exceeded candidate pair budget during differential scan." in out_verbose


def test_cli_min_calibration_frequency_flag_and_pyproject_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies --min-calibration-frequency CLI flag and pyproject.toml loading."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.baseline import load_baseline  # pylint: disable=import-outside-toplevel

    repo = tmp_path / "min_calib_freq_repo"
    repo.mkdir(parents=True)
    f1 = repo / "a.py"
    f2 = repo / "b.py"
    f3 = repo / "c.py"

    code_shared = "def common_task():\n    return 42\n"
    code_singleton = "def unique_task():\n    return 9999\n"

    f1.write_text(code_shared, encoding="utf-8")
    f2.write_text(code_shared, encoding="utf-8")
    f3.write_text(code_singleton, encoding="utf-8")

    bl_file = repo / "baseline_cutoff.json"

    # 1. Record baseline with --min-calibration-frequency 2
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--min-lines",
            "2",
            "--threshold",
            "0.80",
            "--record-baseline",
            str(bl_file),
            "--min-calibration-frequency",
            "2",
        ],
    )
    assert main() == 0

    base_obj = load_baseline(str(bl_file))
    calib = base_obj.corpus_calibration
    assert calib is not None
    assert calib.get("min_frequency") == 2
    freqs = calib.get("shingle_frequencies", {})
    # All retained shingles must have frequency >= 2
    for count in freqs.values():
        assert count >= 2

    # 2. Re-run scan with --baseline (no explicit flag) -> inherits min_frequency=2 without warning
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--min-lines",
            "2",
            "--threshold",
            "0.80",
            "--baseline",
            str(bl_file),
            "--no-color",
        ],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "Warning: Active scan configuration does not match baseline calibration config" not in out

    # 3. pyproject.toml configuration test
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        "[tool.pydoppelgangerhunt]\nmin_calibration_frequency = 3\nmin_lines = 2\nthreshold = 0.80\n",
        encoding="utf-8",
    )
    bl_file_toml = repo / "baseline_toml.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pydoppelgangerhunt",
            str(repo),
            "--config",
            str(pyproject),
            "--record-baseline",
            str(bl_file_toml),
        ],
    )
    assert main() == 0
    base_toml = load_baseline(str(bl_file_toml))
    assert base_toml.corpus_calibration is not None
    assert base_toml.corpus_calibration.get("min_frequency") == 3


def test_paths_match_boundary_notebook_anchors_default() -> None:
    """Verifies that paths_match_boundary defaults to strip_anchor=True for notebook cells while preserving literal hashes."""
    from pydoppelgangerhunt.config import paths_match_boundary  # pylint: disable=import-outside-toplevel

    # 1. Default strip_anchor=True matches different cells of the same notebook
    assert paths_match_boundary("analysis.ipynb#cell_1", "analysis.ipynb#cell_2")
    assert paths_match_boundary("sub/analysis.ipynb#cell_1", "analysis.ipynb#cell_2")
    assert paths_match_boundary("analysis.ipynb#cell_1", "analysis.ipynb")

    # 2. Explicit strip_anchor=False compares full strings with anchors
    assert not paths_match_boundary("analysis.ipynb#cell_1", "analysis.ipynb#cell_2", strip_anchor=False)
    assert paths_match_boundary("analysis.ipynb#cell_1", "sub/analysis.ipynb#cell_1", strip_anchor=False)

    # 3. Literal # in filename is NOT stripped as a cell anchor
    assert paths_match_boundary("report#cellular.ipynb", "other/report#cellular.ipynb")
    assert not paths_match_boundary("report#1.py", "report#2.py")


def test_cli_warns_on_conflicting_strip_and_preserve_flags(tmp_path: Path, capsys: Any) -> None:
    """Verifies that CLI warns when both strip and preserve flags are supplied and prioritizes preserve."""
    from pydoppelgangerhunt.cli import main  # pylint: disable=import-outside-toplevel

    target_file = tmp_path / "sample.py"
    target_file.write_text("x: int = 1\n", encoding="utf-8")

    # 1. Conflicting annotations flags
    code = main([str(target_file), "--strip-annotations", "--preserve-annotations"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Conflicting flags --strip-annotations and --preserve-annotations specified" in captured.err

    # 2. Conflicting docstring flags
    code_doc = main([str(target_file), "--strip-docstrings", "--preserve-docstrings"])
    assert code_doc == 0
    captured_doc = capsys.readouterr()
    assert "Conflicting flags --strip-docstrings and --preserve-docstrings specified" in captured_doc.err


def test_resolve_effective_config_precedence_and_calibration_export() -> None:
    """Verifies that _resolve_effective_config properly handles precedence hierarchy and calibration dict export."""
    from pydoppelgangerhunt.cli import (  # pylint: disable=import-outside-toplevel
        build_arg_parser,
        _resolve_effective_config,
    )

    parser = build_arg_parser()

    # 1. Defaults with empty tool_cfg and empty calib_dict
    args = parser.parse_args([])
    eff = _resolve_effective_config(args, tool_cfg={}, calib_dict={})
    assert eff.min_lines == 8
    assert eff.min_tokens == 15
    assert eff.window_size == 5
    assert eff.min_expr_complexity == 4
    assert eff.min_frequency == 1
    assert eff.max_index_frequency == 0.25
    assert not eff.call_sequences
    assert not eff.audit_tests
    assert not eff.idioms
    assert not eff.stop_shingles

    # 2. Calibration inheritance when CLI and tool_cfg are absent
    calib = {
        "min_lines": 14,
        "min_tokens": 25,
        "window_size": 7,
        "min_expr_complexity": 6,
        "min_frequency": 3,
        "max_index_frequency": 0.12,
        "excludes": ["custom_calib_exclude"],
        "call_sequences": True,
        "audit_tests": True,
        "idioms": True,
        "filter_stop_shingles": True,
    }
    args_calib = parser.parse_args([])
    eff_calib = _resolve_effective_config(args_calib, tool_cfg={}, calib_dict=calib)
    assert eff_calib.min_lines == 14
    assert eff_calib.min_tokens == 25
    assert eff_calib.window_size == 7
    assert eff_calib.min_expr_complexity == 6
    assert eff_calib.min_frequency == 3
    assert eff_calib.max_index_frequency == 0.12
    assert "custom_calib_exclude" in eff_calib.excludes
    assert eff_calib.call_sequences
    assert eff_calib.audit_tests
    assert eff_calib.idioms
    assert eff_calib.stop_shingles

    # 3. CLI override takes highest precedence
    args_override = parser.parse_args([
        "--min-lines",
        "22",
        "--min-calibration-frequency",
        "5",
        "--call-sequences",
        "--no-idioms",
    ])
    eff_override = _resolve_effective_config(args_override, tool_cfg={"min_lines": 18}, calib_dict=calib)
    assert eff_override.min_lines == 22
    assert eff_override.min_frequency == 5
    assert eff_override.call_sequences
    assert not eff_override.idioms

    # 4. to_calibration_config export
    calib_export = eff_override.to_calibration_config(args_override, scope="sub/pkg")
    assert calib_export["min_lines"] == 22
    assert calib_export["min_frequency"] == 5
    assert calib_export["call_sequences"] is True
    assert calib_export["idioms"] is False
    assert calib_export["scope"] == "sub/pkg"
