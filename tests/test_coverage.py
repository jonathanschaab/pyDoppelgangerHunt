"""Unit tests for test coverage readers (Cobertura XML, SQLite) and asymmetric coverage checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydoppelgangerhunt import (
    check_asymmetric_coverage,
    compute_repository_dry_stats,
    compute_unit_coverage,
    read_coverage_data,
)


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
