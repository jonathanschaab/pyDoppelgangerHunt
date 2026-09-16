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


