"""Pure standard library test coverage reader and asymmetric coverage analyzer."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Optional, Set, Tuple
import xml.etree.ElementTree as ET


def _read_sqlite_coverage(coverage_path: str) -> Dict[str, Set[int]]:
    """Reads covered line sets per file from SQLite-based .coverage file."""
    coverage_map: Dict[str, Set[int]] = {}
    if not os.path.isfile(coverage_path):
        return coverage_map

    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(coverage_path)}?mode=ro", uri=True)
        cursor = conn.cursor()

        cursor.execute("SELECT id, path FROM file")
        file_map = {row[0]: row[1].replace("\\", "/") for row in cursor.fetchall()}

        try:
            cursor.execute("SELECT file_id, num_bits, bits FROM line_bits")
            for file_id, num_bits, bit_blob in cursor.fetchall():
                fpath = file_map.get(file_id)
                if not fpath:
                    continue
                covered_lines = coverage_map.setdefault(fpath, set())
                if isinstance(bit_blob, (bytes, bytearray)):
                    for byte_idx, byte_val in enumerate(bit_blob):
                        for bit_idx in range(8):
                            line_num = byte_idx * 8 + bit_idx + 1
                            if line_num > num_bits:
                                break
                            if (byte_val >> bit_idx) & 1:
                                covered_lines.add(line_num)
        except sqlite3.OperationalError:
            pass

        if not any(coverage_map.values()):
            try:
                cursor.execute("SELECT file_id, fromno FROM arc WHERE fromno > 0")
                for file_id, line_num in cursor.fetchall():
                    fpath = file_map.get(file_id)
                    if fpath:
                        coverage_map.setdefault(fpath, set()).add(line_num)
            except sqlite3.OperationalError:
                pass

        conn.close()
    except (sqlite3.Error, OSError):
        pass

    return coverage_map


def _read_xml_coverage(xml_path: str) -> Dict[str, Set[int]]:
    """Reads covered line sets per file from Cobertura coverage.xml."""
    coverage_map: Dict[str, Set[int]] = {}
    if not os.path.isfile(xml_path):
        return coverage_map

    try:
        tree = ET.parse(xml_path)  # nosec B314
        root = tree.getroot()
        for class_node in root.findall(".//class"):
            filename = class_node.get("filename")
            if not filename:
                continue
            norm_file = filename.replace("\\", "/")
            covered_lines = coverage_map.setdefault(norm_file, set())
            for line_node in class_node.findall("./lines/line"):
                hits = line_node.get("hits", "0")
                try:
                    if int(hits) > 0:
                        covered_lines.add(int(line_node.get("number", 0)))
                except ValueError:
                    pass
    except (ET.ParseError, OSError):
        pass

    return coverage_map


def read_coverage_data(coverage_path: str) -> Dict[str, Set[int]]:
    """Dispatches to SQLite or XML coverage reader based on file format."""
    if coverage_path.endswith(".xml"):
        return _read_xml_coverage(coverage_path)
    return _read_sqlite_coverage(coverage_path)


def compute_unit_coverage(
    unit: Dict[str, Any], coverage_data: Dict[str, Set[int]]
) -> float:
    """Computes statement coverage percentage for an AST unit's line range."""
    if not coverage_data:
        return 0.0

    target = str(unit.get("file") or "").replace("\\", "/").split("#", maxsplit=1)[0]
    if not target:
        return 0.0
    covered_lines = coverage_data.get(target)
    if covered_lines is None:
        covered_lines = next(
            (lines for f, lines in coverage_data.items() if f and (f.endswith(target) or target.endswith(f))),
            set(),
        )

    s = int(unit.get("start") or 1)
    e = int(unit.get("end") or s)
    lines_total = e - s + 1
    if lines_total <= 0 or not covered_lines:
        return 0.0

    hits = sum(1 for ln in range(s, e + 1) if ln in covered_lines)
    return min(1.0, hits / float(lines_total))


def check_asymmetric_coverage(
    u1: Dict[str, Any],
    u2: Dict[str, Any],
    coverage_data: Dict[str, Set[int]],
    min_diff: float = 0.40,
) -> Optional[Tuple[float, float]]:
    """Detects if a clone pair has asymmetrical test coverage, posing a silent bug risk."""
    cov1 = compute_unit_coverage(u1, coverage_data)
    cov2 = compute_unit_coverage(u2, coverage_data)
    if abs(cov1 - cov2) >= min_diff:
        return cov1, cov2
    return None
