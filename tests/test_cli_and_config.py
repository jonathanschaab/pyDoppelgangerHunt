"""Unit tests for cli and config."""

from __future__ import annotations

from pathlib import Path
from typing import Any
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
    monkeypatch.setattr(cli, "get_git_modified_line_ranges", lambda since_ref=None, cwd=None: {str(f.resolve()): [(1, 1)]})
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




