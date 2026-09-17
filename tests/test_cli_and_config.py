"""Unit tests for cli and config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
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

    # Test that existing file in CWD is not mangled by repo_root
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        unit_cwd = {"file": "pyproject.toml", "start": 1, "end": 2, "name": "root"}
        cwd_lines = extract_unit_source_code(unit_cwd, repo_root=str(sub_repo))
        assert len(cwd_lines) == 2

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

    # 3. Test config file defaults when flags omitted
    (sub_repo / "pyproject.toml").write_text(
        """
[tool.pydoppelgangerhunt]
threshold = 0.80
min_lines = 4
replace_clones = true
type_merge_strategy = "union"
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




