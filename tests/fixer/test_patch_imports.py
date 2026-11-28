"""Unit tests for cross-file patch imports, shared helper modules, symbol hoisting, and collision safety."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    extract_unit_source_code,
    generate_refactoring_patch,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import patch as patch_mod
from pydoppelgangerhunt.fixer.depgraph import build_module_graph
from tests.helpers import assert_valid_python


def test_inter_block_overlap_collision_detection_and_reverse_offset_refactoring(tmp_path: Path) -> None:
    """Verifies overlap collision detection, maximal subset filtering, and reverse-order refactoring."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        check_units_overlap,
        filter_overlapping_clone_units,
        refactor_module_units,
    )

    u_outer = {"file": "mod.py", "start": 10, "end": 30, "name": "outer_block"}
    u_inner = {"file": "mod.py", "start": 15, "end": 25, "name": "inner_block"}
    u_disjoint = {"file": "mod.py", "start": 40, "end": 50, "name": "disjoint_block"}
    u_other_file = {"file": "other.py", "start": 10, "end": 30, "name": "other_outer"}

    # 1. Test check_units_overlap
    assert check_units_overlap(u_outer, u_inner) is True
    assert check_units_overlap(u_inner, u_outer) is True
    assert check_units_overlap(u_outer, u_disjoint) is False
    assert check_units_overlap(u_outer, u_other_file) is False

    # 2. Test filter_overlapping_clone_units
    # Should prioritize u_outer (21 lines) over u_inner (11 lines), while keeping u_disjoint
    filtered = filter_overlapping_clone_units([u_inner, u_outer, u_disjoint])
    assert len(filtered) == 2
    names = [u["name"] for u in filtered]
    assert "outer_block" in names
    assert "disjoint_block" in names
    assert "inner_block" not in names

    # Empty list edge case
    assert filter_overlapping_clone_units([]) == []

    # 3. Test refactor_module_units overlap collision exception
    source_sample = (
        "line 1\nline 2\nline 3\nline 4\nline 5\n"
        "line 6\nline 7\nline 8\nline 9\nline 10\n"
    )
    rep_col1 = ({"file": "test.py", "start": 2, "end": 5, "name": "c1"}, "NEW_C1\n")
    rep_col2 = ({"file": "test.py", "start": 4, "end": 7, "name": "c2"}, "NEW_C2\n")

    with pytest.raises(ValueError, match="Overlapping unit collision detected"):
        refactor_module_units(source_sample, [rep_col1, rep_col2])

    # 4. Test refactor_module_units reverse line order application (no offset drift)
    # Block A: lines 2-4 (replaced with 1 line)
    # Block B: lines 7-9 (replaced with 1 line)
    rep_a = ({"file": "test.py", "start": 2, "end": 4, "name": "block_a"}, "REPLACED_A\n")
    rep_b = ({"file": "test.py", "start": 7, "end": 9, "name": "block_b"}, "REPLACED_B\n")

    # Pass in forward order: refactor_module_units should internally sort reverse
    refactored = refactor_module_units(source_sample, [rep_a, rep_b])
    expected_lines = [
        "line 1\n",
        "REPLACED_A\n",
        "line 5\n",
        "line 6\n",
        "REPLACED_B\n",
        "line 10\n",
    ]
    assert refactored == "".join(expected_lines)

    # Empty replacements edge case
    assert refactor_module_units(source_sample, []) == source_sample

    # 5. Test generate_refactoring_patch with overlapping clone units in the same file
    f_multi = tmp_path / "multi_clone.py"
    f_multi.write_text(
        "def first_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n"
        "\n"
        "def second_task(x: int) -> int:\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n",
        encoding="utf-8",
    )
    u_t1 = {"file": str(f_multi), "start": 1, "end": 4, "name": "first_task"}
    u_t2 = {"file": str(f_multi), "start": 6, "end": 9, "name": "second_task"}

    # Non-overlapping clones in the same file are both safely refactored in reverse order
    patch = generate_refactoring_patch([(0.95, u_t1, u_t2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_first_task_second_task" in patch
    assert "_shared_first_task_second_task(x)" in patch

def test_generate_refactoring_patch_cross_file_receiver_difference(tmp_path: Path) -> None:
    """Verifies that cross-file pairs with differing receiver kinds decline replacement if receiver is referenced."""
    code1 = (
        "class ServiceA:\n"
        "    def run(self, x: int) -> int:\n"
        "        y = x\n"
        "        return self.factor + y\n"
    )
    code2 = (
        "class ServiceB:\n"
        "    @classmethod\n"
        "    def run(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return cls.factor + y\n"
    )
    f1 = tmp_path / "srv_a.py"
    f2 = tmp_path / "srv_b.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 4, "name": "run", "kind": "function", "enclosing_class": "ServiceA", "receiver_kind": "instance"}
    u2 = {"file": str(f2), "start": 3, "end": 5, "name": "run", "kind": "function", "enclosing_class": "ServiceB", "receiver_kind": "class"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_backslash_relative_path_normalization_on_unit(tmp_path: Path) -> None:
    """Verifies that units carrying Windows-style backslashes resolve correctly across platforms."""
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    f = pkg_dir / "service.py"
    f.write_text("def run_a(x: int) -> int:\n    return x + 10\n\ndef run_b(x: int) -> int:\n    return x + 10\n", encoding="utf-8")

    u1 = {"file": "pkg\\service.py", "start": 1, "end": 2, "name": "run_a", "kind": "function"}
    u2 = {"file": "pkg/service.py", "start": 4, "end": 5, "name": "run_b", "kind": "function"}

    extracted = extract_unit_source_code(u1, repo_root=str(tmp_path))
    assert len(extracted) == 2
    assert "def run_a" in extracted[0]

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "--- a/pkg/service.py" in patch
    assert "+++ b/pkg/service.py" in patch

def test_generate_refactoring_patch_cross_module_with_import_and_execution(tmp_path: Path) -> None:
    """Verifies that cross-module clone refactoring injects the import into file 2 and both run."""
    import subprocess  # pylint: disable=import-outside-toplevel

    pkg_dir = tmp_path / "service_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "def compute_alpha(x: int) -> int:\n"
        "    step1 = x * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    src_b = (
        "def compute_beta(v: int) -> int:\n"
        "    step1 = v * 10\n"
        "    step2 = step1 + 7\n"
        "    return step2\n"
    )
    f_a = pkg_dir / "srv_a.py"
    f_b = pkg_dir / "srv_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_alpha", "file": "service_pkg/srv_a.py", "start": 1, "end": 4, "kind": "function"}
    u_b = {"name": "compute_beta", "file": "service_pkg/srv_b.py", "start": 1, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    # 1. Diff includes both files
    assert "--- a/service_pkg/srv_a.py" in patch
    assert "--- a/service_pkg/srv_b.py" in patch
    # 2. File 2 imports helper from file 1
    assert "from service_pkg.srv_a import _shared_compute_alpha_compute_beta" in patch

    # 3. Verify git apply and execution
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        ["python", "-c", "from service_pkg.srv_a import compute_alpha; from service_pkg.srv_b import compute_beta; print(compute_alpha(5), compute_beta(5))"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "57 57"

def test_generate_refactoring_patch_cross_module_circular_import_safety(tmp_path: Path) -> None:
    """Verifies that circular module imports are detected and prevented during cross-module patch generation."""
    pkg_dir = tmp_path / "cyclic_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "import cyclic_pkg.mod_b\n\n"
        "def work_a(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    src_b = (
        "def work_b(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    f_a = pkg_dir / "mod_a.py"
    f_b = pkg_dir / "mod_b.py"
    f_a.write_text(src_a, encoding="utf-8")
    f_b.write_text(src_b, encoding="utf-8")

    u_a = {"name": "work_a", "file": "cyclic_pkg/mod_a.py", "start": 3, "end": 5, "kind": "function"}
    u_b = {"name": "work_b", "file": "cyclic_pkg/mod_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    # File 2 is NOT refactored with an unsafe circular import
    assert "--- a/cyclic_pkg/mod_b.py" not in patch
    assert "Circular import or unresolvable module path" in patch

def test_cross_module_helper_name_collision_deduplication(tmp_path: Path) -> None:
    """Verifies that if File 2 already defines the candidate helper name, an indexed suffix is generated."""
    pkg = tmp_path / "coll_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def handle_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    src2 = (
        "_shared_handle_data_process_data = 'existing_symbol'\n"
        "\n"
        "def process_data(x: int) -> int:\n"
        "    res = x * 2 + 5\n"
        "    return res\n"
    )
    (pkg / "f1.py").write_text(src1, encoding="utf-8")
    (pkg / "f2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "handle_data", "file": "coll_pkg/f1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "process_data", "file": "coll_pkg/f2.py", "start": 3, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )
    assert "_shared_handle_data_process_data_2" in patch
    assert "from coll_pkg.f1 import _shared_handle_data_process_data_2" in patch

def test_generate_refactoring_patch_cross_module_auto_prevents_circular_import(tmp_path: Path) -> None:
    """Verifies that synthesizing _common.py eliminates sibling circular dependencies even when imports exist."""
    pkg_dir = tmp_path / "cyclic_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "import cyclic_pkg.mod_b\n\n"
        "def work_a(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    src_b = (
        "def work_b(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y * 2\n"
    )
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "work_a", "file": "cyclic_pkg/mod_a.py", "start": 3, "end": 5, "kind": "function"}
    u_b = {"name": "work_b", "file": "cyclic_pkg/mod_b.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # In auto mode, _common.py is synthesized and neither sibling imports the other for the helper
    assert "--- a/cyclic_pkg/mod_a.py" in patch
    assert "--- a/cyclic_pkg/mod_b.py" in patch
    assert "from cyclic_pkg._common import _shared_work_a_work_b" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cyclic_pkg.mod_a; import cyclic_pkg.mod_b; "
            "print(cyclic_pkg.mod_a.work_a(5), cyclic_pkg.mod_b.work_b(5))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "12 12"

@pytest.mark.parametrize("reverse_order", [False, True])
def test_generate_refactoring_patch_cross_module_when_one_file_is_already_shared_module(
    tmp_path: Path, reverse_order: bool
) -> None:
    """Verifies consolidating when either duplicate unit is already located in the shared module."""
    pkg_dir = tmp_path / "shared_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_worker = (
        "def compute_task(x: int) -> int:\n"
        "    r = x * 4 + 3\n"
        "    return r\n"
    )
    src_common = (
        "def helper_seed() -> int:\n"
        "    return 1\n"
        "\n"
        "def compute_common(x: int) -> int:\n"
        "    r = x * 4 + 3\n"
        "    return r\n"
    )
    (pkg_dir / "worker.py").write_text(src_worker, encoding="utf-8")
    (pkg_dir / "_common.py").write_text(src_common, encoding="utf-8")

    u_worker = {"name": "compute_task", "file": "shared_pkg/worker.py", "start": 1, "end": 3, "kind": "function"}
    u_common = {"name": "compute_common", "file": "shared_pkg/_common.py", "start": 4, "end": 6, "kind": "function"}

    pair = (1.0, u_common, u_worker) if reverse_order else (1.0, u_worker, u_common)

    patch = generate_refactoring_patch(
        [pair],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    helper_name = "_shared_compute_common_compute_task" if reverse_order else "_shared_compute_task_compute_common"
    # worker.py imports from shared_pkg._common
    assert "--- a/shared_pkg/worker.py" in patch
    assert f"from shared_pkg._common import {helper_name}" in patch
    # _common.py defines the helper
    assert "--- a/shared_pkg/_common.py" in patch
    assert f"def {helper_name}" in patch

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True)

    apply_proc = subprocess.run(
        ["git", "apply"], input=patch, text=True, cwd=str(tmp_path), capture_output=True, check=False
    )
    assert apply_proc.returncode == 0, f"git apply failed: {apply_proc.stderr}"

    run_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import shared_pkg.worker; import shared_pkg._common; "
            "print(shared_pkg.worker.compute_task(7), shared_pkg._common.compute_common(7))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "31 31"

def test_generate_refactoring_patch_shared_module_helper_name_collision_avoidance(
    tmp_path: Path,
) -> None:
    """Verifies that two distinct clone pairs sharing the same base name do not collide in _common.py."""
    pkg_dir = tmp_path / "collision_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_a = (
        "def compute_val(x: int) -> int:\n"
        "    return x * 10\n"
        "\n"
        "def calculate_val(x: int) -> int:\n"
        "    return x + 99\n"
    )
    src_b = (
        "def compute_val(y: int) -> int:\n"
        "    return y * 10\n"
        "\n"
        "def calculate_val(y: int) -> int:\n"
        "    return y + 99\n"
    )
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a1 = {"name": "compute_val", "file": "collision_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b1 = {"name": "compute_val", "file": "collision_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    u_a2 = {"name": "calculate_val", "file": "collision_pkg/mod_a.py", "start": 4, "end": 5, "kind": "function"}
    u_b2 = {"name": "calculate_val", "file": "collision_pkg/mod_b.py", "start": 4, "end": 5, "kind": "function"}

    # Pre-existing _common.py that already contains _shared_compute_val
    src_existing_common = (
        "def _shared_compute_val(x: int) -> int:\n"
        "    return x * 999\n"
    )
    (pkg_dir / "_common.py").write_text(src_existing_common, encoding="utf-8")

    patch = generate_refactoring_patch(
        [(1.0, u_a1, u_b1), (1.0, u_a2, u_b2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # Pre-existing _shared_compute_val was in _common.py, so helper for u_a1/u_b1 gets _shared_compute_val_2
    assert "def _shared_compute_val_2" in patch
    assert "def _shared_calculate_val" in patch
    assert "from collision_pkg._common import _shared_compute_val_2" in patch

def test_generate_refactoring_patch_auto_detects_existing_shared_module_cycle(
    tmp_path: Path,
) -> None:
    """Verifies that auto strategy detects when existing _common.py already imports caller and avoids cycle."""
    pkg_dir = tmp_path / "precycle_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    # _common.py already imports mod_a
    src_common = (
        "import precycle_pkg.mod_a\n\n"
        "def existing_seed() -> int:\n"
        "    return 1\n"
    )
    src_a = (
        "def run_task(x: int) -> int:\n"
        "    return x * 5\n"
    )
    src_b = (
        "def run_other(x: int) -> int:\n"
        "    return x * 5\n"
    )
    (pkg_dir / "_common.py").write_text(src_common, encoding="utf-8")
    (pkg_dir / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg_dir / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "run_task", "file": "precycle_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "run_other", "file": "precycle_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="auto",
    )

    # Circular import detected for mod_a -> _common -> mod_a
    assert (
        "Circular import detected "
        "(cycle: precycle_pkg.mod_a -> precycle_pkg._common -> precycle_pkg.mod_a)"
    ) in patch
    # mod_a must NOT be patched with a circular import
    assert "--- a/precycle_pkg/mod_a.py" not in patch
    # mod_b (which has no cycle) safely imports from _common
    assert "--- a/precycle_pkg/mod_b.py" in patch
    assert "from precycle_pkg._common import _shared_run_task_run_other" in patch

def test_generate_refactoring_patch_lazy_depgraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that build_module_graph is evaluated lazily only when needed."""
    mock_build = mock.MagicMock()
    monkeypatch.setattr(patch_mod, "build_module_graph", mock_build)

    f = tmp_path / "single.py"
    f.write_text("def a(): return 1\ndef b(): return 1\n", encoding="utf-8")
    u1 = {"name": "a", "file": str(f), "start": 1, "end": 1, "kind": "function"}
    u2 = {"name": "b", "file": str(f), "start": 2, "end": 2, "kind": "function"}

    # Intra-file patch generation does not consult or build the graph
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
    )
    assert "def _shared_a_b" in patch
    mock_build.assert_not_called()

def test_evolving_depgraph_detects_multi_pair_cycles(tmp_path: Path) -> None:
    """Verifies that accepted cross-module dependencies update the evolving graph, catching multi-pair cycles."""
    pkg = tmp_path / "cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod_a.py").write_text(
        "def a1(x: int) -> int:\n    return x + 1\ndef a2(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    (pkg / "mod_b.py").write_text(
        "def b1(x: int) -> int:\n    return x + 1\ndef b2(z: int) -> int:\n    return z * 3\n",
        encoding="utf-8",
    )
    (pkg / "mod_c.py").write_text(
        "def c1(z: int) -> int:\n    return z * 3\ndef c2(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )

    u_a1 = {"name": "a1", "file": "cycle_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b1 = {"name": "b1", "file": "cycle_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}
    u_b2 = {"name": "b2", "file": "cycle_pkg/mod_b.py", "start": 3, "end": 4, "kind": "function"}
    u_c1 = {"name": "c1", "file": "cycle_pkg/mod_c.py", "start": 1, "end": 2, "kind": "function"}
    u_c2 = {"name": "c2", "file": "cycle_pkg/mod_c.py", "start": 3, "end": 4, "kind": "function"}
    u_a2 = {"name": "a2", "file": "cycle_pkg/mod_a.py", "start": 3, "end": 4, "kind": "function"}

    # Pair 1: a1(host) <-> b1(caller) -> b imports a (b -> a)
    # Pair 2: b2(host) <-> c1(caller) -> c imports b (c -> b)
    # Pair 3: c2(host) <-> a2(caller) -> a imports c (a -> c), which closes a -> c -> b -> a cycle!
    clones = [
        (1.0, u_a1, u_b1),
        (1.0, u_b2, u_c1),
        (1.0, u_c2, u_a2),
    ]

    patch = generate_refactoring_patch(
        clones,
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    # Pair 1 & Pair 2 succeed and are wired
    assert "from cycle_pkg.mod_a import _shared_a1_b1" in patch
    assert "from cycle_pkg.mod_b import _shared_b2_c1" in patch

    # Pair 3 is detected as closing a cycle: cycle_pkg.mod_a -> cycle_pkg.mod_c -> cycle_pkg.mod_b -> cycle_pkg.mod_a
    assert "Circular import or unresolvable module path" in patch
    assert "cycle_pkg.mod_a -> cycle_pkg.mod_c -> cycle_pkg.mod_b -> cycle_pkg.mod_a" in patch
    # mod_a must NOT import from mod_c
    assert "from cycle_pkg.mod_c import" not in patch

def test_shared_module_computes_required_imports_for_host(tmp_path: Path) -> None:
    """Verifies that shared modules receive typing and source dependencies even if already imported in f1."""
    pkg = tmp_path / "imports_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "from typing import List\n"
        "\n"
        "def calc_roots(vals: List[float]) -> float:\n"
        "    import math\n"
        "    return math.sqrt(sum(vals))\n"
    )
    src_f2 = (
        "from typing import List\n"
        "\n"
        "def compute_roots(items: List[float]) -> float:\n"
        "    import math\n"
        "    return math.sqrt(sum(items))\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "calc_roots", "file": "imports_pkg/f1.py", "start": 3, "end": 5, "kind": "function"}
    u2 = {"name": "compute_roots", "file": "imports_pkg/f2.py", "start": 3, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Verify that the synthesized _common.py patch contains typing imports and preserves local math import
    assert "diff --git a/imports_pkg/_common.py b/imports_pkg/_common.py" in patch
    assert "+from typing import List" in patch
    assert "+    import math" in patch
    assert "\n+import math\n" not in patch

def test_unreadable_existing_shared_module_skips_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that an existing shared module that fails to read does not emit new-file patch."""
    pkg = tmp_path / "unread_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "f1.py").write_text("def run_a(x):\n    return x * 2\n", encoding="utf-8")
    (pkg / "f2.py").write_text("def run_b(x):\n    return x * 2\n", encoding="utf-8")
    shared_file = pkg / "_common.py"
    shared_file.write_text("# existing unreadable\n", encoding="utf-8")

    orig_read_text = Path.read_text

    def mock_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.resolve() == shared_file.resolve():
            raise OSError("Permission denied: unreadable")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    u1 = {"name": "run_a", "file": "unread_pkg/f1.py", "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "run_b", "file": "unread_pkg/f2.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Must NOT emit a new-file patch for _common.py
    assert "new file mode 100644" not in patch
    assert "--- /dev/null" not in patch
    # Must emit advisory notice about existing unreadable file
    expected_note = (
        "shared module file unread_pkg/_common.py exists but could not be read; skipping extraction."
    )
    assert expected_note in patch

def test_relative_imports_preserved_and_translated_in_shared_module(tmp_path: Path) -> None:
    """Verifies that relative imports in clone files are translated to canonical imports in _common.py."""
    pkg = tmp_path / "rel_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    src_f1 = (
        "from .helpers import CustomType\n"
        "\n"
        "def compute_val(v: CustomType) -> int:\n"
        "    return 42\n"
    )
    src_f2 = (
        "from .helpers import CustomType\n"
        "\n"
        "def compute_other(v: CustomType) -> int:\n"
        "    return 42\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "compute_val", "file": "rel_pkg/f1.py", "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_other", "file": "rel_pkg/f2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py should receive translated canonical import: from rel_pkg.helpers import CustomType
    assert "diff --git a/rel_pkg/_common.py b/rel_pkg/_common.py" in patch
    assert "+from rel_pkg.helpers import CustomType" in patch

def test_shared_module_host_imports_cycle_detection(tmp_path: Path) -> None:
    """Verifies that when _common.py gains dependencies on a caller module, cycle is detected and prevented."""
    pkg = tmp_path / "host_cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "class HelperDep:\n"
        "    pass\n"
        "\n"
        "def clone_a(x: HelperDep) -> int:\n"
        "    return 42\n"
    )
    src_f2 = (
        "from host_cycle_pkg.f1 import HelperDep\n"
        "\n"
        "def clone_b(x: HelperDep) -> int:\n"
        "    return 42\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "clone_a", "file": "host_cycle_pkg/f1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "clone_b", "file": "host_cycle_pkg/f2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # Because helper uses HelperDep from host_cycle_pkg.f1, _common.py imports host_cycle_pkg.f1.
    # Therefore, f1 importing _common.py would form f1 -> _common -> f1 cycle!
    assert "Circular import detected" in patch
    assert "host_cycle_pkg.f1 -> host_cycle_pkg._common -> host_cycle_pkg.f1" in patch

def test_future_annotations_propagation_in_shared_module(tmp_path: Path) -> None:
    """Verifies that from __future__ import annotations is carried over to synthesized shared module."""
    pkg = tmp_path / "future_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "from __future__ import annotations\n"
        "from typing import List\n"
        "\n"
        "def process_models(items: List[str] | None) -> int:\n"
        "    return len(items) if items else 0\n"
    )
    src_f2 = (
        "from __future__ import annotations\n"
        "from typing import List\n"
        "\n"
        "def handle_models(items: List[str] | None) -> int:\n"
        "    return len(items) if items else 0\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "process_models", "file": "future_pkg/f1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "handle_models", "file": "future_pkg/f2.py", "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/future_pkg/_common.py b/future_pkg/_common.py" in patch
    assert "+from __future__ import annotations" in patch
    # Verify from __future__ import annotations comes before other imports in the diff
    future_idx = patch.find("+from __future__ import annotations")
    typing_idx = patch.find("+from typing import List")
    assert future_idx != -1
    assert typing_idx != -1
    assert future_idx < typing_idx

def test_shared_module_dry_run_no_ghost_edge_or_caller_import(tmp_path: Path) -> None:
    """Verifies that replace_clones=False with shared_module does not emit caller imports or ghost edges."""
    pkg = tmp_path / "dry_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_a = "def compute_a(x: int) -> int:\n    return x * 10 + 1\n"
    src_b = "def compute_b(x: int) -> int:\n    return x * 10 + 1\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_a", "file": "dry_shared_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "compute_b", "file": "dry_shared_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    # _common.py is synthesized with the helper
    assert "diff --git a/dry_shared_pkg/_common.py b/dry_shared_pkg/_common.py" in patch
    assert "+def _shared_compute_a_compute_b(x: int) -> int:" in patch

    # Callers receive the proposed import from _common
    assert "+from dry_shared_pkg._common import _shared_compute_a_compute_b" in patch
    # Function bodies are NOT replaced because replace_clones=False
    assert "return _shared_compute_a_compute_b" not in patch
    # Callers receive advisory comments explaining where helper was extracted
    assert "Complete refactoring by replacing the clone with a call in dry_shared_pkg/mod_a.py" in patch
    assert "Complete refactoring by replacing the clone with a call in dry_shared_pkg/mod_b.py" in patch

def test_shared_module_existing_directory_skipped(tmp_path: Path) -> None:
    """Verifies that when the target shared module path is an existing directory, extraction skips gracefully."""
    pkg = tmp_path / "dir_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Create a directory named _common.py
    dir_conflict = pkg / "_common.py"
    dir_conflict.mkdir()

    src_a = "def run_a(x: int) -> int:\n    return x + 5\n"
    src_b = "def run_b(x: int) -> int:\n    return x + 5\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "run_a", "file": "dir_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "run_b", "file": "dir_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "is an existing directory; skipping extraction" in patch
    assert "new file mode 100644" not in patch

def test_host_module_detects_cycle_from_synthesized_host_imports(tmp_path: Path) -> None:
    """Verifies that host_module detects cycle when host gains an import pointing back to caller."""
    pkg = tmp_path / "host_cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # mod_b has a helper function used by the clone in mod_b
    src_b = (
        "def helper_b(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "def compute_b(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    src_a = (
        "def helper_b(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "def compute_a(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "compute_a", "file": "host_cycle_pkg/mod_a.py", "start": 4, "end": 6, "kind": "function"}
    u_b = {"name": "compute_b", "file": "host_cycle_pkg/mod_b.py", "start": 4, "end": 6, "kind": "function"}

    # Pre-wire mod_a to import mod_b so mod_a -> mod_b
    src_a_with_import = (
        "from host_cycle_pkg.mod_b import helper_b\n"
        "\n"
        "def compute_a(x: int) -> int:\n"
        "    y = helper_b(x)\n"
        "    return y * 2\n"
    )
    (pkg / "mod_a.py").write_text(src_a_with_import, encoding="utf-8")
    u_a["start"] = 3
    u_a["end"] = 5

    # If mod_b were to import mod_a, cycle mod_b -> mod_a -> mod_b would form
    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    assert "Circular import detected" in patch or "Circular import or unresolvable module path" in patch
    assert "from host_cycle_pkg.mod_a import _shared_compute_a_compute_b" not in patch

def test_host_module_ignores_type_checking_guarded_direct_import(tmp_path: Path) -> None:
    """Verifies that TYPE_CHECKING-only imports do not block safe host-module extraction."""
    pkg_dir = tmp_path / "guard_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    import guard_pkg.caller\n\n"
        "def compute_host(x: int) -> int:\n"
        "    return x * 2 + 5\n"
    )
    src_caller = (
        "def compute_caller(x: int) -> int:\n"
        "    return x * 2 + 5\n"
    )
    (pkg_dir / "host.py").write_text(src_host, encoding="utf-8")
    (pkg_dir / "caller.py").write_text(src_caller, encoding="utf-8")

    u_host = {"name": "compute_host", "file": "guard_pkg/host.py", "start": 6, "end": 7, "kind": "function"}
    u_caller = {"name": "compute_caller", "file": "guard_pkg/caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
    )

    assert "Circular import or unresolvable module path" not in patch
    assert "from guard_pkg.host import _shared_compute_host_compute_caller" in patch

def test_shared_module_translates_local_relative_imports(tmp_path: Path) -> None:
    """Verifies that relative function-local imports in clones are translated to canonical absolute paths."""
    pkg = tmp_path / "local_rel_pkg"
    sub1 = pkg / "sub1"
    sub2 = pkg / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "__init__.py").write_text("", encoding="utf-8")

    (pkg / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 42\n", encoding="utf-8")

    src_1 = (
        "def compute_1(x: int) -> int:\n"
        "    from ..helpers import transform\n"
        "    return transform(x)\n"
    )
    src_2 = (
        "def compute_2(x: int) -> int:\n"
        "    from ..helpers import transform\n"
        "    return transform(x)\n"
    )
    (sub1 / "mod1.py").write_text(src_1, encoding="utf-8")
    (sub2 / "mod2.py").write_text(src_2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": "local_rel_pkg/sub1/mod1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_2", "file": "local_rel_pkg/sub2/mod2.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In local_rel_pkg/_common.py, the import should be canonicalized to from local_rel_pkg.helpers import transform
    assert "from local_rel_pkg.helpers import transform" in patch
    # Should NOT have verbatim 'from ..helpers import transform' added in the shared module
    assert "+from ..helpers import transform" not in patch
    assert "+    from ..helpers import transform" not in patch

def test_shared_module_translates_local_relative_imports_package_init(tmp_path: Path) -> None:
    """Verifies that relative function-local imports in pkg/__init__.py are translated with package context."""
    pkg = tmp_path / "pkg_init_test"
    pkg.mkdir()
    (pkg / "deps.py").write_text("def helper(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    src_init = (
        "def compute_init(x: int) -> int:\n"
        "    from .deps import helper\n"
        "    return helper(x)\n"
    )
    src_mod = (
        "def compute_mod(x: int) -> int:\n"
        "    from .deps import helper\n"
        "    return helper(x)\n"
    )
    (pkg / "__init__.py").write_text(src_init, encoding="utf-8")
    (pkg / "mod.py").write_text(src_mod, encoding="utf-8")

    u1 = {"name": "compute_init", "file": "pkg_init_test/__init__.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_mod", "file": "pkg_init_test/mod.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In pkg_init_test/_common.py, the import should be translated to from pkg_init_test.deps import helper
    assert "from pkg_init_test.deps import helper" in patch
    # Should NOT have resolved to from deps import helper (which would be missing pkg_init_test.)
    assert "+from deps import helper" not in patch
    assert "+    from deps import helper" not in patch

def test_existing_shared_module_gains_future_annotations_from_source(tmp_path: Path) -> None:
    """Verifies that an existing shared module gains future annotations when clone source has them."""
    pkg = tmp_path / "future_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # Pre-existing _common.py without future annotations
    (pkg / "_common.py").write_text("def existing_helper() -> None:\n    pass\n", encoding="utf-8")

    src1 = (
        "from __future__ import annotations\n"
        "from dep_pkg import Later\n\n"
        "def run_calc(x: Later) -> Later:\n"
        "    return x\n"
    )
    src2 = (
        "from __future__ import annotations\n"
        "from dep_pkg import Later\n\n"
        "def run_calc2(x: Later) -> Later:\n"
        "    return x\n"
    )
    (pkg / "mod1.py").write_text(src1, encoding="utf-8")
    (pkg / "mod2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "run_calc", "file": "future_pkg/mod1.py", "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "run_calc2", "file": "future_pkg/mod2.py", "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py should receive from __future__ import annotations
    assert "+from __future__ import annotations" in patch

def test_collect_host_missing_imports_preserves_aliased_typing(tmp_path: Path) -> None:
    """Verifies that aliased typing imports (e.g. from typing import List as MyList) are preserved."""
    pkg = tmp_path / "alias_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from typing import List as MyList\n\n"
        "def process(items: MyList[int]) -> int:\n"
        "    return len(items)\n"
    )
    src2 = (
        "from typing import List as MyList\n\n"
        "def process2(items: MyList[int]) -> int:\n"
        "    return len(items)\n"
    )
    (pkg / "mod1.py").write_text(src1, encoding="utf-8")
    (pkg / "mod2.py").write_text(src2, encoding="utf-8")

    u1 = {"name": "process", "file": "alias_pkg/mod1.py", "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "process2", "file": "alias_pkg/mod2.py", "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # In _common.py, the aliased typing import should be present
    assert "from typing import List as MyList" in patch

def test_generate_patch_relative_repo_root_no_path_doubling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that passing a relative repo_root does not duplicate path components."""
    monkeypatch.chdir(tmp_path)
    rel_root = Path("sub_proj")
    pkg = rel_root / "my_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src = "def util(a: int) -> int:\n    return a * 2\n"
    (pkg / "a.py").write_text(src, encoding="utf-8")
    (pkg / "b.py").write_text(src, encoding="utf-8")

    u1 = {"name": "util", "file": str(pkg / "a.py"), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "util", "file": str(pkg / "b.py"), "start": 1, "end": 2, "kind": "function"}

    # Pass genuinely relative repo_root
    rel_repo_root = "sub_proj"
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=rel_repo_root,
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "sub_proj/sub_proj" not in patch.replace("\\", "/")
    assert "my_pkg/_common.py" in patch.replace("\\", "/")

def test_generate_patch_conflicting_import_symbols_skips_extraction(tmp_path: Path) -> None:
    """Verifies that clones binding the same symbol to different modules skip extraction."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    sub1 = pkg / "sub1"
    sub1.mkdir()
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    sub2 = pkg / "sub2"
    sub2.mkdir()
    (sub2 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    src1 = (
        "from .helpers import CustomType\n\n"
        "def compute(item: CustomType) -> int:\n"
        "    return len(item)\n"
    )
    src2 = (
        "from .helpers import CustomType\n\n"
        "def compute(item: CustomType) -> int:\n"
        "    return len(item)\n"
    )
    f1 = sub1 / "worker.py"
    f2 = sub2 / "worker.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType'" in patch
    assert "skipping extraction" in patch
    assert "_common.py" not in patch

def test_generate_patch_skips_guarded_imports_in_shared_module(tmp_path: Path) -> None:
    """Verifies that imports inside conditional or guarded blocks are not emitted into shared modules."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from typing import List\n"
        "if True:\n"
        "    from typing_extensions import Buffer\n\n"
        "def run_job(x: List[int]) -> int:\n"
        "    return len(x)\n"
    )
    src2 = (
        "from typing import List\n"
        "try:\n"
        "    import optional_dep\n"
        "except ImportError:\n"
        "    pass\n\n"
        "def run_job(x: List[int]) -> int:\n"
        "    return len(x)\n"
    )
    f1 = pkg / "a.py"
    f2 = pkg / "b.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_job", "file": str(f1), "start": 5, "end": 6, "kind": "function"}
    u2 = {"name": "run_job", "file": str(f2), "start": 7, "end": 8, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    common_section = patch.split("diff --git a/pkg/_common.py")[-1]
    assert "from typing import List" in common_section
    assert "Buffer" not in common_section
    assert "optional_dep" not in common_section

def test_generate_patch_wildcard_import_unresolved_symbol(tmp_path: Path) -> None:
    """Verifies that an unresolved symbol in a helper where source used wildcard import skips extraction."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import *\n\n"
        "def calc(x: CustomType) -> float:\n"
        "    return float(len(str(x)))\n"
    )
    src2 = (
        "from math import *\n\n"
        "def calc(x: CustomType) -> float:\n"
        "    return float(len(str(x)))\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "potentially relies on wildcard import" in patch
    assert "skipping extraction" in patch

def test_cross_file_wildcard_with_explicit_import_rejected(tmp_path: Path) -> None:
    """Verifies that cross-file extraction is rejected when one clone uses wildcard and another has explicit import."""
    pkg = tmp_path / "pkg_wildcard_mixed"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep_a import *\n\n"
        "def process(x: TransformType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "from dep_b import TransformType\n\n"
        "def process(x: TransformType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "process", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "process", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "potentially relies on wildcard import" in patch
    assert "skipping extraction" in patch
    assert "from dep_b import TransformType" not in patch

def test_same_file_wildcard_import_allowed(tmp_path: Path) -> None:
    """Verifies that same-file clones sharing a module-level wildcard import are extracted."""
    pkg = tmp_path / "pkg_same_file_wildcard"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src = (
        "from math import *\n\n"
        "def run_a(x: int) -> float:\n"
        "    return float(x) * 2.0\n\n"
        "def run_b(x: int) -> float:\n"
        "    return float(x) * 2.0\n"
    )
    f = pkg / "mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "run_a", "file": str(f), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "run_b", "file": str(f), "start": 6, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    assert "potentially relies on wildcard import" not in patch
    assert "def _shared_run_a" in patch

def test_shared_module_rejects_unresolved_nonbuiltin_dependency(tmp_path: Path) -> None:
    """Verifies that a helper referencing an unimported module-local type rejects shared module extraction."""
    pkg = tmp_path / "pkg_unresolved"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def compute_1(x: LocalDep) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def compute_2(x: LocalDep) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": str(f1), "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "compute_2", "file": str(f2), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "unresolved symbol 'LocalDep' in shared module" in patch
    assert "skipping extraction" in patch

def test_shared_module_rejects_conflicting_local_imports(tmp_path: Path) -> None:
    """Verifies that clones with conflicting local imports for the same symbol are rejected."""
    pkg = tmp_path / "pkg_conflicting_locals"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def calc_1(x: float) -> float:\n"
        "    from math import sin as f\n"
        "    return f(x)\n"
    )
    src2 = (
        "def calc_2(x: float) -> float:\n"
        "    from cmath import sin as f\n"
        "    return f(x).real\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc_1", "file": str(f1), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "calc_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting local import symbol 'f' across clone sources" in patch
    assert "skipping extraction" in patch

def test_host_module_detects_conflicting_imported_symbol_with_host(tmp_path: Path) -> None:
    """Verifies that extraction is skipped if host module already imports the symbol from another package."""
    pkg = tmp_path / "pkg_host_conflict"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from pkg_a import transform\n\n"
        "def run_1(x: int) -> int:\n"
        "    from pkg_b import transform\n"
        "    return transform(x)\n"
    )
    src2 = (
        "def run_2(x: int) -> int:\n"
        "    from pkg_b import transform\n"
        "    return transform(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_1", "file": str(f1), "start": 3, "end": 5, "kind": "function"}
    u2 = {"name": "run_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting imported symbol 'transform' across clone sources" in patch
    assert "skipping extraction" in patch

def test_canonicalize_helper_relative_imports_with_nonstandard_whitespace() -> None:
    """Verifies that relative imports with irregular whitespace are parsed and canonicalized."""
    from pydoppelgangerhunt.fixer.patch import _canonicalize_helper_relative_imports  # pylint: disable=import-outside-toplevel

    code = (
        "def helper(x: int) -> int:\n"
        "    from   .helpers   import   transform\n"
        "    return transform(x)\n"
    )
    res = _canonicalize_helper_relative_imports(code, [("my_pkg.mod", False)])
    assert "from my_pkg.helpers import transform" in res
    assert "from   ." not in res

def test_shared_module_rejects_conflicting_local_relative_imports(tmp_path: Path) -> None:
    """Verifies that clones in different subpackages using conflicting relative imports are rejected."""
    pkg = tmp_path / "conflict_rel_pkg"
    sub1 = pkg / "sub1"
    sub2 = pkg / "sub2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub1 / "__init__.py").write_text("", encoding="utf-8")
    (sub2 / "__init__.py").write_text("", encoding="utf-8")

    (sub1 / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    (sub2 / "helpers.py").write_text("def transform(x: int) -> int:\n    return x + 2\n", encoding="utf-8")

    src_1 = (
        "def compute_1(x: int) -> int:\n"
        "    from .helpers import transform\n"
        "    return transform(x)\n"
    )
    src_2 = (
        "def compute_2(x: int) -> int:\n"
        "    from .helpers import transform\n"
        "    return transform(x)\n"
    )
    (sub1 / "mod1.py").write_text(src_1, encoding="utf-8")
    (sub2 / "mod2.py").write_text(src_2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": "conflict_rel_pkg/sub1/mod1.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_2", "file": "conflict_rel_pkg/sub2/mod2.py", "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert (
        "conflicting relative import in helper across clone sources" in patch
        or "conflicting local import symbol 'transform' across clone sources" in patch
    )
    assert "from conflict_rel_pkg.sub1.helpers import transform" not in patch

def test_same_file_conflicting_local_imports_skips_without_corrupting_plan(tmp_path: Path) -> None:
    """Verifies same-file clones with conflicting local imports skip extraction cleanly without dirtying plan."""
    f = tmp_path / "same_file_mod.py"
    f.write_text(
        "def foo(x: int) -> int:\n"
        "    from math import sqrt as f\n"
        "    return f(x)\n\n"
        "def bar(x: int) -> int:\n"
        "    from cmath import sqrt as f\n"
        "    return f(x).real\n",
        encoding="utf-8",
    )
    u1 = {"name": "foo", "file": str(f), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "bar", "file": str(f), "start": 5, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )

    assert "conflicting local import symbol 'f' across clone sources" in patch
    assert "skipping extraction" in patch
    assert "_shared_foo_bar" not in patch

def test_local_imports_not_hoisted_to_module_scope(tmp_path: Path) -> None:
    """Verifies that lazy/guarded local imports stay inside the helper body and are not hoisted to module scope."""
    pkg = tmp_path / "lazy_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "def do_calc1(x: int) -> int:\n"
        "    import math\n"
        "    return math.isqrt(x)\n"
    )
    src2 = (
        "def do_calc2(x: int) -> int:\n"
        "    import math\n"
        "    return math.isqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "do_calc1", "file": str(f1), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "do_calc2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/lazy_pkg/_common.py b/lazy_pkg/_common.py" in patch
    assert "+    import math" in patch
    assert "\n+import math\n" not in patch

def test_host_existing_relative_imports_canonicalized_without_false_conflict(tmp_path: Path) -> None:
    """Verifies that existing relative imports in host module match canonicalized clone imports without conflict."""
    pkg = tmp_path / "rel_host_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers.py").write_text("class CustomType:\n    pass\n", encoding="utf-8")

    host_src = (
        "from .helpers import CustomType\n\n"
        "def run_host(val: CustomType) -> CustomType:\n"
        "    return val\n"
    )
    caller_src = (
        "from rel_host_pkg.helpers import CustomType\n\n"
        "def run_caller(val: CustomType) -> CustomType:\n"
        "    return val\n"
    )

    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(host_src, encoding="utf-8")
    f_caller.write_text(caller_src, encoding="utf-8")

    u1 = {"name": "run_host", "file": str(f_host), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "run_caller", "file": str(f_caller), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting imported symbol" not in patch
    assert "skipping extraction" not in patch
    assert "from rel_host_pkg.host_mod import _shared_run_host_run_caller" in patch

def test_shared_module_resolution_failure_adds_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if resolve_shared_module_file raises ValueError, patch adds skip advisory comment."""
    pkg = tmp_path / "unsafe_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 1\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    from pydoppelgangerhunt.fixer import patch as patch_mod  # pylint: disable=import-outside-toplevel

    def mock_resolve_shared(*args: Any, **kwargs: Any) -> Path:
        raise ValueError("Shared module path resolves outside common package directory")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_shared)

    patch = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Shared module path resolves outside common package directory" in patch
    assert "skipping extraction" in patch
    assert "_common.py" not in patch

def test_shared_module_resolution_filesystem_error_adds_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if resolve_shared_module_file raises OSError or RuntimeError, patch adds skip advisory comment."""
    pkg = tmp_path / "oserr_shared_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 1\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    from pydoppelgangerhunt.fixer import patch as patch_mod  # pylint: disable=import-outside-toplevel

    def mock_resolve_oserror(*args: Any, **kwargs: Any) -> Path:
        raise OSError("Permission denied: cannot access directory")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_oserror)

    patch_os = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Permission denied: cannot access directory" in patch_os
    assert "skipping extraction" in patch_os
    assert "_common.py" not in patch_os

    def mock_resolve_runtime_err(*args: Any, **kwargs: Any) -> Path:
        raise RuntimeError("Symlink loop detected")

    monkeypatch.setattr(patch_mod, "resolve_shared_module_file", mock_resolve_runtime_err)

    patch_rt = patch_mod.generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "Symlink loop detected" in patch_rt
    assert "skipping extraction" in patch_rt
    assert "_common.py" not in patch_rt

def test_clone_sources_conflicting_definition_and_import_rejected(tmp_path: Path) -> None:
    """Verifies that if one clone imports a symbol while another defines it locally, extraction is rejected."""
    pkg = tmp_path / "conflict_def_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep import CustomType\n\n"
        "def calc1(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "class CustomType:\n"
        "    pass\n\n"
        "def calc2(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType' across clone sources" in patch
    assert "skipping extraction" in patch

def test_clone_sources_inconsistent_binding_missing_import_rejected(tmp_path: Path) -> None:
    """Verifies that if one clone imports a symbol while another has no binding for it, extraction is rejected."""
    pkg = tmp_path / "inconsistent_bind_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from dep import CustomType\n\n"
        "def calc1(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    src2 = (
        "def calc2(x: CustomType) -> int:\n"
        "    return 42\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'CustomType' across clone sources" in patch
    assert "skipping extraction" in patch

def test_helper_retains_module_import_when_local_import_not_selected(tmp_path: Path) -> None:
    """Verifies that module-level import is hoisted when clone 2 had local import not present in clone 1 helper."""
    pkg = tmp_path / "retained_local_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import sqrt\n\n"
        "def calc1(x: int) -> float:\n"
        "    return sqrt(x)\n"
    )
    src2 = (
        "def calc2(x: int) -> float:\n"
        "    from math import sqrt\n"
        "    return sqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "calc1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "calc2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    assert "from math import sqrt" in patch
    assert "skipping extraction" not in patch
    assert "diff --git a/retained_local_pkg/_common.py b/retained_local_pkg/_common.py" in patch

def test_nested_function_parameters_do_not_shadow_outer_helper_imports(tmp_path: Path) -> None:
    """Verifies that nested function parameters in helper do not prevent module-level imports from hoisting."""
    pkg = tmp_path / "nested_shadow_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import sqrt as transform\n\n"
        "def runner1(x: int) -> float:\n"
        "    res = transform(x)\n"
        "    def inner(transform: int) -> int:\n"
        "        return transform + 1\n"
        "    return res + inner(1)\n"
    )
    src2 = (
        "from math import sqrt as transform\n\n"
        "def runner2(x: int) -> float:\n"
        "    res = transform(x)\n"
        "    def inner(transform: int) -> int:\n"
        "        return transform + 1\n"
        "    return res + inner(1)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "runner1", "file": str(f1), "start": 3, "end": 7, "kind": "function"}
    u2 = {"name": "runner2", "file": str(f2), "start": 3, "end": 7, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from math import sqrt as transform" in patch
    assert "skipping extraction" not in patch

def test_symlink_shared_module_target_rejected_with_advisory_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that if target shared module is a symlink, extraction is skipped with advisory comment."""
    pkg = tmp_path / "sym_target_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def f1(x: int) -> int:\n    return x + 1\n"
    src2 = "def f2(x: int) -> int:\n    return x + 1\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "f1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "f2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    orig_is_symlink = Path.is_symlink

    def mock_is_symlink(self: Path) -> bool:
        if self.name == "_common.py":
            return True
        return orig_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", mock_is_symlink)

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "is an existing symlink" in patch
    assert "skipping extraction" in patch
    assert "diff --git" not in patch

def test_generate_refactoring_patch_with_package_root_depgraph_preserves_cycle_detection(
    tmp_path: Path,
) -> None:
    """Verifies externally built depgraphs from package roots use the same module namespace."""
    from pydoppelgangerhunt.fixer.depgraph import build_module_graph  # pylint: disable=import-outside-toplevel

    project_root = tmp_path / "project"
    project_root.mkdir()
    top_pkg = project_root / "my_package"
    top_pkg.mkdir()
    (top_pkg / "__init__.py").write_text("", encoding="utf-8")
    (top_pkg / "caller.py").write_text(
        "def compute_c(x: int) -> int:\n"
        "    return x + 10\n",
        encoding="utf-8",
    )
    (top_pkg / "mid.py").write_text(
        "import my_package.caller\n\n"
        "def compute_m(x: int) -> int:\n"
        "    return my_package.caller.compute_c(x)\n",
        encoding="utf-8",
    )
    (top_pkg / "host.py").write_text(
        "import my_package.mid\n\n"
        "def compute_h(x: int) -> int:\n"
        "    return x + 10\n",
        encoding="utf-8",
    )

    depgraph = build_module_graph(top_pkg)
    u_host = {"name": "compute_h", "file": "host.py", "start": 3, "end": 4, "kind": "function"}
    u_caller = {"name": "compute_c", "file": "caller.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(top_pkg),
        replace_clones=True,
        cross_file_strategy="host_module",
        depgraph=depgraph,
    )

    assert (
        "Circular import or unresolvable module path "
        "(cycle: my_package.caller -> my_package.host -> my_package.mid -> my_package.caller)"
    ) in patch
    assert "from my_package.host import _shared_compute_h_compute_c" not in patch

def test_collect_host_missing_imports_handles_shadowed_builtins(tmp_path: Path) -> None:
    """Verifies that shadowed builtins are hoisted when consistent or rejected when conflicting."""
    pkg = tmp_path / "shadow_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Consistent shadowing: both modules import a custom `len`
    src1_consistent = (
        "from custom_utils import len\n\n"
        "def count_items1(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    src2_consistent = (
        "from custom_utils import len\n\n"
        "def count_items2(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    f1 = pkg / "c1.py"
    f2 = pkg / "c2.py"
    f1.write_text(src1_consistent, encoding="utf-8")
    f2.write_text(src2_consistent, encoding="utf-8")

    u1 = {"name": "count_items1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "count_items2", "file": str(f2), "start": 3, "end": 4, "kind": "function"}

    patch_consistent = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from custom_utils import len" in patch_consistent
    assert "from shadow_pkg._common import _shared_count_items1_count_items2" in patch_consistent

    # Inconsistent shadowing: one clone imports custom `len`, the other uses builtin `len`
    src2_inconsistent = (
        "def count_items2(items: list) -> int:\n"
        "    return len(items) + 1\n"
    )
    f2.write_text(src2_inconsistent, encoding="utf-8")
    u2_inconsistent = {"name": "count_items2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch_inconsistent = generate_refactoring_patch(
        [(1.0, u1, u2_inconsistent)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "conflicting imported symbol 'len'" in patch_inconsistent
    assert "skipping extraction" in patch_inconsistent

def test_generate_refactoring_patch_cross_file_strategy_skip(tmp_path: Path) -> None:
    """Verifies that cross_file_strategy='skip' bypasses cross-file clones while allowing same-file clones."""
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text("def a():\n    return 1\ndef b():\n    return 1\n", encoding="utf-8")
    f2.write_text("def c():\n    return 1\n", encoding="utf-8")

    u_a = {"name": "a", "file": "mod1.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "b", "file": "mod1.py", "start": 3, "end": 4, "kind": "function"}
    u_c = {"name": "c", "file": "mod2.py", "start": 1, "end": 2, "kind": "function"}

    # Pure cross-file pair with skip should produce no patch
    patch_cross = generate_refactoring_patch(
        [(1.0, u_a, u_c)],
        repo_root=str(tmp_path),
        cross_file_strategy="skip",
    )
    assert patch_cross == ""

    # Same-file pair should still be extracted even when cross_file_strategy='skip'
    patch_same = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        cross_file_strategy="skip",
    )
    assert "def _shared_a_b" in patch_same

def test_nested_imports_do_not_suppress_outer_helper_imports(tmp_path: Path) -> None:
    """Verifies that imports in nested functions/classes do not suppress outer helper imports."""
    # pylint: disable=protected-access
    code = (
        "def helper(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    tree = ast.parse(code)
    free_names, defined_names, local_imports = patch_mod._extract_helper_symbols(tree)
    assert "transform" in free_names
    assert "transform" not in local_imports
    assert "helper" in defined_names
    assert "x" in defined_names
    assert "inner" in defined_names
    assert "y" not in defined_names

    # End-to-end patch generation test:
    # Outer helper uses `transform`, while inner function locally imports a different `transform`.
    # Ensure `from mymod import transform` is hoisted to the shared module header.
    pkg = tmp_path / "nested_import_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from mymod import transform\n\n"
        "def run_step1(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    src2 = (
        "from mymod import transform\n\n"
        "def run_step2(x: int) -> int:\n"
        "    res = transform(x)\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import transform\n"
        "        return transform(y)\n"
        "    return res + inner(x)\n"
    )
    f1 = pkg / "s1.py"
    f2 = pkg / "s2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "run_step1", "file": str(f1), "start": 3, "end": 8, "kind": "function"}
    u2 = {"name": "run_step2", "file": str(f2), "start": 3, "end": 8, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "from mymod import transform" in patch
    assert "from nested_import_pkg._common import _shared_run_step1_run_step2" in patch

def test_type_checking_guarded_import_does_not_suppress_runtime_helper_typing(
    tmp_path: Path,
) -> None:
    """Verifies that imports under 'if TYPE_CHECKING:' do not suppress runtime helper typing imports."""
    pkg = tmp_path / "type_check_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Existing _common.py contains guarded TYPE_CHECKING import
    common_file = pkg / "_common.py"
    common_file.write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from typing import List\n",
        encoding="utf-8",
    )

    src1 = (
        "def compute_items1(vals: List[int]) -> int:\n"
        "    return sum(vals)\n"
    )
    src2 = (
        "def compute_items2(vals: List[int]) -> int:\n"
        "    return sum(vals)\n"
    )
    f1 = pkg / "c1.py"
    f2 = pkg / "c2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_items1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "compute_items2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    # _common.py patch MUST contain unconditional 'from typing import List'
    assert "+from typing import List" in patch

def test_cross_file_host_local_definition_conflict_rejected(
    tmp_path: Path,
) -> None:
    """Verifies that cross-file extraction into host module is rejected when callers define conflicting local dependencies."""
    # pylint: disable=protected-access
    pkg = tmp_path / "host_conflict_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_host = (
        "class LocalService:\n"
        "    pass\n\n"
        "def process_service1(svc: LocalService) -> int:\n"
        "    return 42\n"
    )
    src_caller = (
        "class LocalService:\n"
        "    pass\n\n"
        "def process_service2(svc: LocalService) -> int:\n"
        "    return 42\n"
    )
    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(src_host, encoding="utf-8")
    f_caller.write_text(src_caller, encoding="utf-8")

    u_host = {"name": "process_service1", "file": str(f_host), "start": 4, "end": 5, "kind": "function"}
    u_caller = {"name": "process_service2", "file": str(f_caller), "start": 4, "end": 5, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_host, u_caller)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    assert "conflicting local definition 'LocalService' across clone sources" in patch
    # Extraction must be safely skipped, no helper inserted
    assert "_shared_process_service1_process_service2" not in patch

    # Direct collector verification for conflicting and unresolved symbols
    host_plan = patch_mod._FilePatchPlan(file_path=f_host, rel_path="host_mod.py", orig_text=src_host)
    helper_code = "def _shared(svc: LocalService) -> int:\n    return 42\n"
    with pytest.raises(ValueError, match="conflicting local definition 'LocalService'"):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope={},
            source_texts=[(src_host, "host_mod", False), (src_caller, "caller_mod", False)],
            host_mod="host_mod",
        )

    with pytest.raises(ValueError, match="unresolved symbol 'LocalService' in caller module"):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope={},
            source_texts=[(src_host, "host_mod", False), ("# no defs\n", "caller_mod", False)],
            host_mod="host_mod",
        )

def test_host_local_definition_omits_self_import_when_caller_imports_from_host(
    tmp_path: Path,
) -> None:
    """Verifies that when a caller imports a dependency from host, host omits self-imports of its own local definition."""
    # pylint: disable=protected-access
    pkg = tmp_path / "self_import_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    host_src = (
        "class LocalDep:\n"
        "    pass\n\n"
        "def process_item1(item: LocalDep) -> int:\n"
        "    return 42\n"
    )
    caller_src = (
        "from self_import_pkg.host_mod import LocalDep\n\n"
        "def process_item2(item: LocalDep) -> int:\n"
        "    return 42\n"
    )
    f_host = pkg / "host_mod.py"
    f_caller = pkg / "caller_mod.py"
    f_host.write_text(host_src, encoding="utf-8")
    f_caller.write_text(caller_src, encoding="utf-8")

    # 1. Direct collector check
    host_plan = patch_mod._FilePatchPlan(file_path=f_host, rel_path="self_import_pkg/host_mod.py", orig_text=host_src)
    helper_code = "def _shared_process_item1_process_item2(item: LocalDep) -> int:\n    return 42\n"
    imports = patch_mod._collect_host_missing_imports(
        host_plan=host_plan,
        helper_code=helper_code,
        scope={},
        source_texts=[
            (host_src, "self_import_pkg.host_mod", False),
            (caller_src, "self_import_pkg.caller_mod", False),
        ],
        host_mod="self_import_pkg.host_mod",
    )
    # Must NOT contain self-import from self_import_pkg.host_mod
    assert not any("self_import_pkg.host_mod" in imp for imp in imports)
    assert not any("LocalDep" in imp for imp in imports)

    # 2. End-to-end patch generation check
    u1 = {"name": "process_item1", "file": str(f_host), "start": 4, "end": 5, "kind": "function"}
    u2 = {"name": "process_item2", "file": str(f_caller), "start": 3, "end": 4, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host",
    )

    # Host module diff must NOT contain a self-import
    assert "+from self_import_pkg.host_mod import LocalDep" not in patch
    assert "+import self_import_pkg.host_mod" not in patch
    # Host module must receive the extracted helper
    assert "def _shared_process_item1_process_item2(item: LocalDep) -> int:" in patch
    # Caller module must import the extracted helper from host
    assert "from self_import_pkg.host_mod import _shared_process_item1_process_item2" in patch

def test_clones_with_matching_local_imports_hoisted_to_shared_module(tmp_path: Path) -> None:
    """Verifies that when helper lacks a local import, an agreed local import from clones is safely hoisted."""
    pkg = tmp_path / "shared_local_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = (
        "from math import isqrt\n\n"
        "def compute_1(x: int) -> int:\n"
        "    return isqrt(x)\n"
    )
    src2 = (
        "def compute_2(x: int) -> int:\n"
        "    from math import isqrt\n"
        "    return isqrt(x)\n"
    )
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": str(f1), "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_2", "file": str(f2), "start": 1, "end": 3, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=False,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/shared_local_pkg/_common.py b/shared_local_pkg/_common.py" in patch
    assert "+from math import isqrt" in patch
    assert "def _shared_compute_1_compute_2(x: int) -> int:" in patch

def test_shared_module_rejects_subsequent_pair_creating_cycle_with_earlier_caller(
    tmp_path: Path,
) -> None:
    """Verifies that a subsequent clone pair cannot inject dependencies into a shared module

    that create a circular dependency with an earlier pair's caller.
    """
    root = tmp_path / "repo"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # caller_a defines CustomType and has clone 1
    src_a = (
        "class CustomType:\n"
        "    pass\n\n"
        "def clone_one_a(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    # caller_b has clone 1
    src_b = (
        "def clone_one_b(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    # caller_c imports CustomType from caller_a and has clone 2
    src_c = (
        "from pkg.caller_a import CustomType\n\n"
        "def clone_two_c(item: CustomType) -> int:\n"
        "    a = 10\n"
        "    b = a + 20\n"
        "    return b\n"
    )
    # caller_d imports CustomType from caller_a and has clone 2
    src_d = (
        "from pkg.caller_a import CustomType\n\n"
        "def clone_two_d(item: CustomType) -> int:\n"
        "    a = 10\n"
        "    b = a + 20\n"
        "    return b\n"
    )

    fa = pkg / "caller_a.py"
    fb = pkg / "caller_b.py"
    fc = pkg / "caller_c.py"
    fd = pkg / "caller_d.py"
    fa.write_text(src_a, encoding="utf-8")
    fb.write_text(src_b, encoding="utf-8")
    fc.write_text(src_c, encoding="utf-8")
    fd.write_text(src_d, encoding="utf-8")

    u1a = {"name": "clone_one_a", "file": str(fa), "start": 4, "end": 7, "kind": "function"}
    u1b = {"name": "clone_one_b", "file": str(fb), "start": 1, "end": 4, "kind": "function"}
    u2c = {"name": "clone_two_c", "file": str(fc), "start": 3, "end": 6, "kind": "function"}
    u2d = {"name": "clone_two_d", "file": str(fd), "start": 3, "end": 6, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1a, u1b), (1.0, u2c, u2d)],
        repo_root=str(root),
        cross_file_strategy="shared_module",
        replace_clones=True,
    )

    assert patch is not None
    # Pair 1 succeeded in creating _common.py
    assert "pkg/_common.py" in patch
    # Pair 1 callers were patched
    assert "pkg/caller_a.py" in patch
    assert "pkg/caller_b.py" in patch

    # _common.py must NOT import caller_a (which would cause a cycle)
    assert "+from pkg.caller_a import" not in patch

    # Pair 2 was rejected due to circular dependency
    assert "rejected due to circular dependency" in patch

def test_cross_module_shared_module_directory_collision_rejected(tmp_path: Path) -> None:
    """Verifies that if the shared module path is an existing directory, extraction is safely skipped."""
    pkg = tmp_path / "pkg_dir_collision"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src1 = "def fn1(x: int) -> int:\n    return x + 10\n"
    src2 = "def fn2(x: int) -> int:\n    return x + 10\n"
    f1 = pkg / "m1.py"
    f2 = pkg / "m2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    # Create directory at _common.py
    (pkg / "_common.py").mkdir()

    u1 = {"name": "fn1", "file": str(f1), "start": 1, "end": 2, "kind": "function"}
    u2 = {"name": "fn2", "file": str(f2), "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        cross_file_strategy="shared_module",
        replace_clones=True,
    )

    assert "existing directory" in patch
    assert "skipping extraction" in patch

def test_scope_nested_import_not_recorded_in_local_imports(tmp_path: Path) -> None:
    """Verifies that imports within nested functions are not recorded in unit local_imports."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
    )

    code = (
        "def outer(x: int) -> int:\n"
        "    def inner(y: int) -> int:\n"
        "        from inner_pkg import helper\n"
        "        return helper(y)\n"
        "    return inner(x)\n"
    )
    f = tmp_path / "test_nested.py"
    f.write_text(code, encoding="utf-8")
    unit = {"file": "test_nested.py", "start": 1, "end": 5, "kind": "function", "name": "outer"}
    scope_info = analyze_unit_variable_scope(unit, repo_root=str(tmp_path))
    assert "from inner_pkg import helper" not in scope_info["local_imports"]

def test_resolve_safe_clone_file_path_does_not_fallback_relative_path_to_patch_root(
    tmp_path: Path,
) -> None:
    """Verifies that relative clone paths are only resolved against the scan root, not patch_root."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _resolve_safe_clone_file_path,
    )

    project = tmp_path / "project"
    tools = project / "pkg" / "tools"
    tools.mkdir(parents=True)
    other = project / "pkg" / "other.py"
    other.write_text("x = 1\n", encoding="utf-8")

    res = _resolve_safe_clone_file_path("pkg/other.py", (tools, project))
    assert res is None

def test_extract_module_import_statements_compound_blocks() -> None:
    """Verifies that runtime imports inside top-level try/except and if blocks are extracted."""
    src = (
        "try:\n"
        "    import ujson as json\n"
        "except ImportError:\n"
        "    import json\n"
        "if True:\n"
        "    from math import sqrt\n"
    )
    stmts, wildcards = patch_mod._extract_module_import_statements(src)
    assert "json" in stmts
    assert "sqrt" in stmts
    assert not wildcards

def test_shared_module_strategy_skips_when_host_is_clone_and_cycle_detected(
    tmp_path: Path,
) -> None:
    """Verifies that shared_module strategy targeting a clone file skips transactionally on cycle."""
    pkg = tmp_path / "shared_host_cycle"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    src_f1 = (
        "import shared_host_cycle.f2\n\n"
        "def compute_1(x: int) -> int:\n"
        "    return x * 10\n"
    )
    src_f2 = (
        "def compute_2(x: int) -> int:\n"
        "    return x * 10\n"
    )
    (pkg / "f1.py").write_text(src_f1, encoding="utf-8")
    (pkg / "f2.py").write_text(src_f2, encoding="utf-8")

    u1 = {"name": "compute_1", "file": "shared_host_cycle/f1.py", "start": 3, "end": 4, "kind": "function"}
    u2 = {"name": "compute_2", "file": "shared_host_cycle/f2.py", "start": 1, "end": 2, "kind": "function"}

    # Set shared_module_name to f1.py so host_plan is f1_plan
    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
        shared_module_name="f1.py",
    )

    assert "--- a/shared_host_cycle/f1.py" not in patch
    assert "--- a/shared_host_cycle/f2.py" not in patch
    assert "Circular import or unresolvable module path" in patch

def test_shared_module_strategy_skips_when_all_callers_form_cycles(
    tmp_path: Path,
) -> None:
    """Verifies that if all callers form cycles with _common.py, extraction is transactionally skipped."""
    pkg = tmp_path / "all_cycle_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    # Create existing _common.py that already imports both mod_a and mod_b
    (pkg / "_common.py").write_text(
        "import all_cycle_pkg.mod_a\nimport all_cycle_pkg.mod_b\n", encoding="utf-8"
    )
    src_a = "def process_a(x: int) -> int:\n    return x + 5\n"
    src_b = "def process_b(x: int) -> int:\n    return x + 5\n"
    (pkg / "mod_a.py").write_text(src_a, encoding="utf-8")
    (pkg / "mod_b.py").write_text(src_b, encoding="utf-8")

    u_a = {"name": "process_a", "file": "all_cycle_pkg/mod_a.py", "start": 1, "end": 2, "kind": "function"}
    u_b = {"name": "process_b", "file": "all_cycle_pkg/mod_b.py", "start": 1, "end": 2, "kind": "function"}

    patch = generate_refactoring_patch(
        [(1.0, u_a, u_b)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "--- a/all_cycle_pkg/mod_a.py" not in patch
    assert "--- a/all_cycle_pkg/mod_b.py" not in patch
    assert "rejected due to circular dependency" in patch

def test_generate_refactoring_patch_init_and_submodule_shared_module(
    tmp_path: Path,
) -> None:
    """Verifies that clones between __init__.py and a submodule can share a module without false cycle."""
    pkg = tmp_path / "init_sub_pkg"
    pkg.mkdir()
    src_init = (
        "def common_calc(val: int) -> int:\n"
        "    res = val * 2 + 1\n"
        "    return res\n"
    )
    src_worker = (
        "def run_worker_calc(val: int) -> int:\n"
        "    res = val * 2 + 1\n"
        "    return res\n"
    )
    (pkg / "__init__.py").write_text(src_init, encoding="utf-8")
    (pkg / "worker.py").write_text(src_worker, encoding="utf-8")

    u1 = {
        "name": "common_calc",
        "file": "init_sub_pkg/__init__.py",
        "start": 1,
        "end": 3,
        "kind": "function",
    }
    u2 = {
        "name": "run_worker_calc",
        "file": "init_sub_pkg/worker.py",
        "start": 1,
        "end": 3,
        "kind": "function",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="shared_module",
    )

    assert "diff --git a/init_sub_pkg/_common.py b/init_sub_pkg/_common.py" in patch
    assert "--- a/init_sub_pkg/__init__.py" in patch
    assert "--- a/init_sub_pkg/worker.py" in patch
    assert "Circular import" not in patch

def test_collect_host_missing_imports_substring_safety() -> None:
    """Verifies that substring matches in comments or longer import names do not suppress required imports."""
    host_code = (
        "# Note: do not import os directly\n"
        "import os_helper\n\n"
        "def existing_fn():\n"
        "    return os_helper.run()\n"
    )
    host_plan = patch_mod._FilePatchPlan(
        file_path=Path("app/main.py"),
        rel_path="app/main.py",
        orig_text=host_code,
    )
    helper_code = "def helper_fn():\n    return os.path.exists('foo')\n"
    scope = {"module": "app.main"}
    source_code = "import os\ndef orig_fn():\n    return os.path.exists('foo')\n"

    missing = patch_mod._collect_host_missing_imports(
        host_plan=host_plan,
        helper_code=helper_code,
        scope=scope,
        source_texts=[source_code],
    )
    assert "import os" in missing

def test_extract_module_import_statements_final_binding_and_rebound_conflicts() -> None:
    """Verifies that later top-level imports override earlier bindings and flag conflicts."""
    src = (
        "from dep_a import transform\n"
        "from dep_b import transform\n"
        "import os\n"
        "import os\n"
    )
    res = patch_mod._extract_module_import_statements(src)
    assert res.stmts["transform"] == "from dep_b import transform"
    assert "transform" in res.rebound_conflicts
    assert "transform" in res.unconditional_names
    assert res.stmts["os"] == "import os"
    assert "os" not in res.rebound_conflicts
    assert "os" in res.unconditional_names

def test_collect_host_missing_imports_rejects_rebound_conflicts() -> None:
    """Verifies that clone extraction fails when referencing a rebound conflicting symbol."""
    source_code = (
        "from dep_a import transform\n"
        "from dep_b import transform\n\n"
        "def compute(x: int) -> int:\n"
        "    return transform(x)\n"
    )
    host_plan = patch_mod._FilePatchPlan(
        file_path=Path("common.py"),
        rel_path="common.py",
        orig_text="",
        is_new_file=True,
    )
    helper_code = "def helper_fn(x: int) -> int:\n    return transform(x)\n"
    scope = {"module": "source_mod"}

    with pytest.raises(
        ValueError,
        match="conflicting imported symbol 'transform' rebound within source module",
    ):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope=scope,
            source_texts=[(source_code, "source_mod", False)],
        )

def test_collect_host_missing_imports_rejects_cross_file_guarded_imports() -> None:
    """Verifies that cross-file extraction rejects symbols relying on runtime fallback imports."""
    source_code = (
        "try:\n"
        "    from fast_dep import transform\n"
        "except ImportError:\n"
        "    from fallback_dep import transform\n\n"
        "def compute(x: int) -> int:\n"
        "    return transform(x)\n"
    )
    host_plan = patch_mod._FilePatchPlan(
        file_path=Path("pkg/_common.py"),
        rel_path="pkg/_common.py",
        orig_text="",
        is_new_file=True,
    )
    helper_code = "def helper_fn(x: int) -> int:\n    return transform(x)\n"
    scope = {"module": "pkg.worker"}

    with pytest.raises(
        ValueError,
        match="symbol 'transform' relies on runtime-guarded import in source module",
    ):
        patch_mod._collect_host_missing_imports(
            host_plan=host_plan,
            helper_code=helper_code,
            scope=scope,
            source_texts=[(source_code, "pkg.worker", False)],
        )

def test_collect_host_missing_imports_same_file_preserves_guarded_imports() -> None:
    """Verifies that same-file host refactoring does not hoist guarded imports unconditionally."""
    source_code = (
        "try:\n"
        "    import ujson as json\n"
        "except ImportError:\n"
        "    import json\n\n"
        "def f1(d):\n"
        "    return json.dumps(d)\n"
    )
    host_plan = patch_mod._FilePatchPlan(
        file_path=Path("pkg/worker.py"),
        rel_path="pkg/worker.py",
        orig_text=source_code,
        is_new_file=False,
    )
    helper_code = "def helper_fn(d):\n    return json.dumps(d)\n"
    scope = {"module": "pkg.worker"}

    missing = patch_mod._collect_host_missing_imports(
        host_plan=host_plan,
        helper_code=helper_code,
        scope=scope,
        source_texts=[(source_code, "pkg.worker", False)],
    )
    assert not any("json" in s for s in missing)

def test_collect_host_missing_imports_rejects_import_rebound_by_definition(tmp_path: Path) -> None:
    """Verifies that attempting cross-file extraction of a symbol rebound by local definition is rejected."""
    plan = patch_mod._FilePatchPlan(
        file_path=tmp_path / "mod_target.py",
        rel_path="mod_target.py",
        orig_text="",
        is_new_file=True,
    )
    source_code = (
        "from dep_lib import process\n"
        "def process(x: int) -> int:\n"
        "    return x + 10\n"
    )
    helper_code = "def _shared(x: int) -> int:\n    return process(x)\n"

    with pytest.raises(ValueError, match="conflicting imported symbol 'process' rebound by local definition"):
        patch_mod._collect_host_missing_imports(
            host_plan=plan,
            helper_code=helper_code,
            scope={"local_imports": []},
            source_texts=[(source_code, "pkg.worker", False)],
        )

def test_generate_refactoring_patch_does_not_mutate_caller_depgraph(
    tmp_path: Path,
) -> None:
    """Verifies generate_refactoring_patch works on a copy of depgraph and does not mutate caller's graph."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mod_a = pkg / "mod_a.py"
    mod_b = pkg / "mod_b.py"

    mod_a.write_text("import pkg.mod_b\ndef func_a(): return 123\n", encoding="utf-8")
    mod_b.write_text("def func_b(): return 123\n", encoding="utf-8")

    u1 = {"name": "func_a", "file": str(mod_a), "start": 2, "end": 2, "kind": "function"}
    u2 = {"name": "func_b", "file": str(mod_b), "start": 1, "end": 1, "kind": "function"}

    depgraph = build_module_graph(tmp_path)
    snapshot_adj = {k: set(v) for k, v in depgraph.adjacency.items()}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
        cross_file_strategy="host_module",
        depgraph=depgraph,
    )

    assert "Circular import or unresolvable module path" in patch
    # Verify caller's depgraph adjacency remains identical
    assert depgraph.adjacency == snapshot_adj

def test_canonicalize_helper_relative_imports_semicolon_separated() -> None:
    """Verifies that multiple relative imports on the same line are both canonicalized."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _canonicalize_helper_relative_imports,
    )

    code = "from .a import x; from .b import y\n"
    canon = _canonicalize_helper_relative_imports(code, [("pkg.mod", False)])
    assert "from pkg.a import x" in canon
    assert "from pkg.b import y" in canon

def test_format_patch_relative_path_nested_directories(tmp_path: Path) -> None:
    """Verifies that _format_patch_relative_path does not strip nested directory components."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _format_patch_relative_path,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    rel_target = Path("sub/nested/file.py")
    res = _format_patch_relative_path(rel_target, repo)
    assert res == "sub/nested/file.py"

def test_delegate_unit_in_plan_tier2_collision_raises_and_rolls_back(tmp_path: Path) -> None:
    """Verifies that physical byte-span collisions in _delegate_unit_in_plan trigger transactional rollback."""
    from pydoppelgangerhunt.fixer import UnitCollisionError
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _delegate_unit_in_plan

    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text("def a():\n    v = 1\n    return v\n", encoding="utf-8")
    f2.write_text("def b():\n    v = 1\n    return v\n", encoding="utf-8")

    u1 = {"file": str(f1), "name": "a", "start": 2, "end": 3, "kind": "function"}
    u2 = {"file": str(f2), "name": "b", "start": 2, "end": 3, "kind": "function"}

    # Valid clone pair works cleanly via batch patch generation
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "_shared_a_b" in patch

    # Verify direct _delegate_unit_in_plan collision detection and transactional rollback
    plan = _FilePatchPlan(
        file_path=f1,
        rel_path="mod1.py",
        orig_text=f1.read_text(encoding="utf-8"),
        is_new_file=False,
    )
    # First unit delegation succeeds
    _delegate_unit_in_plan(
        u1,
        plan,
        helper_name="_shared_a_b",
        inputs=[],
        outputs=["v"],
        scope={"is_async": False, "has_yield": False},
        target_inputs=None,
        target_outputs=None,
        await_prefix="",
    )
    assert len(plan.replacements) == 1
    assert len(plan.claimed_units) == 1

    # Colliding unit overlapping with u1 triggers UnitCollisionError
    u_colliding = {"file": str(f1), "name": "inner", "start": 2, "end": 2, "kind": "statement"}
    with pytest.raises(UnitCollisionError):
        _delegate_unit_in_plan(
            u_colliding,
            plan,
            helper_name="_shared_inner",
            inputs=[],
            outputs=["v"],
            scope={"is_async": False, "has_yield": False},
            target_inputs=None,
            target_outputs=None,
            await_prefix="",
        )

    # Local transactional rollback: plan state is restored to pre-collision state
    assert len(plan.replacements) == 1
    assert plan.replacements[0][0] == u1
    assert len(plan.claimed_units) == 1
    assert plan.claimed_units[0] == u1

def test_render_file_patch_plan_tier2_collision_fallback() -> None:
    """Verifies that _render_file_patch_plan safely recovers and filters when replacements collide."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan, _render_file_patch_plan

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\nc = 3\n",
        is_new_file=False,
    )
    u1 = {"file": "test.py", "start": 1, "end": 2}
    u2 = {"file": "test.py", "start": 2, "end": 3}  # Collides physically on line 2
    plan.replacements.append((u1, "# replaced 1\n"))
    plan.replacements.append((u2, "# replaced 2\n"))

    # _render_file_patch_plan should not raise ValueError; it should filter the collision and render a valid patch
    diff = _render_file_patch_plan(plan, replace_clones=True)
    assert "--- a/test.py" in diff
    assert "# replaced 1" in diff
    assert "# replaced 2" not in diff

def test_unit_collision_error_hierarchy_and_render_fallback() -> None:
    """Verifies that UnitCollisionError subclasses ValueError and is raised on collisions."""
    from pydoppelgangerhunt.fixer import UnitCollisionError, refactor_module_units

    assert issubclass(UnitCollisionError, ValueError)

    collision_u1 = {"file": "collision_target.py", "name": "block_alpha", "start": 1, "end": 2}
    collision_u2 = {"file": "collision_target.py", "name": "block_beta", "start": 1, "end": 2}

    with pytest.raises(UnitCollisionError) as exc_info:
        refactor_module_units("x = 1\ny = 2\n", [(collision_u1, "x = 10\n"), (collision_u2, "y = 20\n")])
    assert "Overlapping unit collision detected" in str(exc_info.value)
    assert isinstance(exc_info.value, ValueError)

def test_package_reexport_unit_collision_error() -> None:
    """Verifies that UnitCollisionError is re-exported in root and fixer packages."""
    import pydoppelgangerhunt as pdgh
    import pydoppelgangerhunt.fixer as pdgh_fixer

    assert hasattr(pdgh, "UnitCollisionError")
    assert "UnitCollisionError" in pdgh.__all__
    assert issubclass(pdgh.UnitCollisionError, ValueError)

    assert hasattr(pdgh_fixer, "UnitCollisionError")
    assert "UnitCollisionError" in pdgh_fixer.__all__
    assert issubclass(pdgh_fixer.UnitCollisionError, ValueError)

def test_refactor_module_units_tier2_sweep_catches_nested_and_adjacent_collisions() -> None:
    """Verifies that O(N log N) Tier 2 physical sweep catches nested intervals and collisions."""
    from pydoppelgangerhunt.fixer import UnitCollisionError, refactor_module_units

    text = "alpha = 1\nbeta = 2\ngamma = 3\n"
    # c_wide covers lines 1-3, c_inner covers line 2 (nested within c_wide)
    c_wide = {"file": "f.py", "name": "wide", "start": 1, "end": 3}
    c_inner = {"file": "f.py", "name": "inner", "start": 2, "end": 2}

    with pytest.raises(UnitCollisionError) as err:
        refactor_module_units(text, [(c_wide, "# wide\n"), (c_inner, "# inner\n")])
    assert "Overlapping unit collision detected" in str(err.value)

def test_compute_replacement_line_deltas_sweep_catches_nested_interval_collision() -> None:
    """Verifies that _compute_replacement_line_deltas catches nested intervals via max_end_item sweep."""
    from pydoppelgangerhunt.fixer import UnitCollisionError
    from pydoppelgangerhunt.fixer.patch import _compute_replacement_line_deltas

    text = "line_one = 1\nline_two = 2\nline_three = 3\n"
    c_wide = {"file": "f.py", "name": "wide", "start": 1, "end": 3}
    c_inner = {"file": "f.py", "name": "inner", "start": 2, "end": 2}

    with pytest.raises(UnitCollisionError) as err:
        _compute_replacement_line_deltas([(c_wide, "# wide\n"), (c_inner, "# inner\n")], text)
    assert "Overlapping unit collision detected in line delta computation" in str(err.value)

def test_tier2_sweep_zero_width_slice_collision() -> None:
    """Verifies that _intervals_overlap and Tier 2 sweep detect collisions with zero-width slices."""
    from pydoppelgangerhunt.fixer.patch import _intervals_overlap

    # Zero-width point strictly inside active interval [10, 20)
    assert _intervals_overlap(10, 20, 15, 15) is True
    assert _intervals_overlap(15, 15, 10, 20) is True

    # Zero-width point touching interval boundaries exactly does not collide
    assert _intervals_overlap(10, 20, 10, 10) is False
    assert _intervals_overlap(10, 20, 20, 20) is False
    assert _intervals_overlap(10, 10, 10, 20) is False
    assert _intervals_overlap(20, 20, 10, 20) is False

    # Zero-width points at the exact same offset collide
    assert _intervals_overlap(10, 10, 10, 10) is True
    assert _intervals_overlap(10, 10, 15, 15) is False

    # Disjoint intervals
    assert _intervals_overlap(10, 20, 25, 25) is False
    assert _intervals_overlap(10, 20, 5, 5) is False

def test_refactor_module_units_zero_width_insertion_collision() -> None:
    """Verifies that refactor_module_units detects collisions between distinct zero-width insertions."""
    from pydoppelgangerhunt.fixer import UnitCollisionError, refactor_module_units

    code = "x = 1\ny = 2\n"
    # Two zero-width units at the exact same line and column offset collide in Tier 2
    u1 = {"file": "sample.py", "start": 1, "end": 1, "start_col": 2, "end_col": 2}
    u2 = {"file": "sample.py", "start": 1, "end": 1, "start_col": 2, "end_col": 2}

    with pytest.raises(UnitCollisionError, match="Overlapping unit collision detected"):
        refactor_module_units(code, [(u1, "# ins1 "), (u2, "# ins2 ")])

    # Two zero-width units at distinct column offsets succeed without collision
    u3 = {"file": "sample.py", "start": 1, "end": 1, "start_col": 4, "end_col": 4}
    res = refactor_module_units(code, [(u1, "#1 "), (u3, "#2 ")])
    assert res == "x #1 = #2 1\ny = 2\n"

def test_canonicalize_helper_relative_imports_form_feed_and_crlf() -> None:
    """Verifies that _canonicalize_helper_relative_imports preserves form feeds and line endings."""
    from pydoppelgangerhunt.fixer.patch import _canonicalize_helper_relative_imports

    helper_code = "\x0cdef foo():\r\n    from .bar import baz\r\n    x = 1\x0c + 2\r\n    return baz()\r\n"
    res = _canonicalize_helper_relative_imports(
        helper_code,
        [("pkg.sub", False)],
    )
    assert "\x0c" in res
    assert "from pkg.bar import baz" in res
    assert "\r\n" in res
    assert "x = 1\x0c + 2" in res

def test_delegate_unit_in_plan_incremental_scaling_and_collision() -> None:
    """Verifies incremental O(K) collision checking and snapshot rollback in _FilePatchPlan."""
    from pydoppelgangerhunt.fixer.patch import (
        UnitCollisionError,
        _FilePatchPlan,
        _delegate_unit_in_plan,
    )

    code = "def foo():\n    a = 1\n    b = 2\n    return a + b\n"
    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text=code,
    )

    u1 = {"file": "test.py", "start": 2, "end": 2, "start_col": 4, "end_col": 9}
    _delegate_unit_in_plan(
        unit=u1,
        plan=plan,
        helper_name="_h1",
        inputs=[],
        outputs=["a"],
        scope={},
        target_inputs=None,
        target_outputs=None,
        await_prefix="",
    )
    assert len(plan.replacements) == 1
    assert len(plan.replacement_items) == 1

    # Snapshot state
    snap = plan.snapshot()

    # Attempt colliding unit
    u_collide = {"file": "test.py", "start": 2, "end": 2, "start_col": 4, "end_col": 9}
    with pytest.raises(UnitCollisionError):
        _delegate_unit_in_plan(
            unit=u_collide,
            plan=plan,
            helper_name="_h2",
            inputs=[],
            outputs=["a"],
            scope={},
            target_inputs=None,
            target_outputs=None,
            await_prefix="",
        )

    # Verify plan was not mutated by failed delegation
    assert len(plan.replacements) == 1
    assert len(plan.replacement_items) == 1

    # Restore snapshot and verify parity
    plan.restore(snap)
    assert len(plan.replacements) == 1
    assert len(plan.replacement_items) == 1

def test_validate_helper_compilation_indented_method_with_imports() -> None:
    """Verifies that _validate_helper_compilation accepts an indented method helper
    combined with unindented imports without raising IndentationError."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _validate_helper_compilation,
    )

    combined_helper = (
        "from typing import Tuple\n\n\n"
        "    def _shared_calc(self, x: int) -> Tuple[int, int]:\n"
        "        return x, x * 2\n"
    )
    assert _validate_helper_compilation(combined_helper) is True

    # Truly invalid syntax should return False
    assert _validate_helper_compilation("def invalid syntax (:") is False

def test_validate_helper_compilation_indented_method_with_multiline_string_at_col_0() -> None:
    """Verifies that _validate_helper_compilation and assert_valid_python accept an indented
    method helper containing a triple-quoted multiline string with lines at column 0."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=import-outside-toplevel
        _validate_helper_compilation,
    )

    helper_code = (
        "    def _format_message(self, tag: str) -> str:\n"
        '        template = """header:\n'
        "col_0_line\n"
        'footer"""\n'
        "        return f'{tag}: {template}'\n"
    )
    assert _validate_helper_compilation(helper_code) is True
    assert_valid_python(helper_code)

    combined_helper = (
        "from typing import Optional\n\n\n"
        "    def _format_message(self, tag: str) -> Optional[str]:\n"
        '        template = """header:\n'
        "col_0_line\n"
        'footer"""\n'
        "        return f'{tag}: {template}'\n"
    )
    assert _validate_helper_compilation(combined_helper) is True
    assert_valid_python(combined_helper)

def test_validate_helper_compilation_multiline_imports() -> None:
    """Verifies that multiline parenthesized imports compile cleanly in helper validation."""
    from pydoppelgangerhunt.fixer.patch import _validate_helper_compilation
    from tests.helpers import assert_valid_python

    helper_code = (
        "from os import (\n"
        "    path,\n"
        ")\n\n\n"
        "def _shared_helper(a: int) -> int:\n"
        "    return a + 1\n"
    )
    assert _validate_helper_compilation(helper_code) is True
    assert_valid_python(helper_code)

def test_build_unit_delegation_call_cross_clone_name_collision_no_overprune() -> None:
    """Verifies that _build_unit_delegation_call does not mutate inputs when target
    name matches a remaining input variable."""
    from pydoppelgangerhunt.fixer.patch import (  # pylint: disable=protected-access
        _build_unit_delegation_call,
    )

    unit = {
        "kind": "complex_expr",
        "start": 1,
        "end": 1,
    }
    orig_lines = ["a += b + c\n"]
    call_str = _build_unit_delegation_call(
        unit,
        "a += b + c\n",
        orig_lines,
        effective_binding="module",
        helper_name="_shared_helper",
        inputs=["a", "b", "c"],
        outputs=[],
        scope={"inputs": ["a", "b", "c"]},
        target_inputs=["a", "b", "c"],
        await_prefix="",
    )
    assert "_shared_helper(a, b, c)" in call_str

def test_augassign_target_pruning_cross_variable_collision(tmp_path: Path) -> None:
    """Verifies that when an AugAssign target matches an active input in the partner clone,
    both unreferenced targets are pruned and call-site arguments preserve positional order."""
    src1 = (
        "def func1(a, b, c):\n"
        "    a += b - c\n"
        "    return a\n"
    )
    src2 = (
        "def func2(total, a, b):\n"
        "    total += a - b\n"
        "    return total\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "mod1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "mod2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(b: Any, c: Any) -> Any:" in helper
    assert "return b - c" in helper

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "def _shared_u1_u2(b: Any, c: Any) -> Any:" in patch
    assert "+    a += _shared_u1_u2(b, c)" in patch
    assert "+    total += _shared_u1_u2(a, b)" in patch
    assert "_shared_u1_u2(a, b, total)" not in patch
    assert "_shared_u1_u2(b, a)" not in patch

def test_lhs_load_pruning_cross_clone_collision_and_arity_mismatch(
    tmp_path: Path,
) -> None:
    """Verifies that whole-line complex_expr units whose LHS-only loads collide with the
    partner clone's RHS inputs preserve positional argument order without permuting operands,
    and fail closed when pruned RHS arities differ."""
    src1 = (
        "def store_one(arr, a, b, c):\n"
        "    arr[a] = b - c\n"
    )
    src2 = (
        "def store_two(arr, idx, a, b):\n"
        "    arr[idx] = a - b\n"
    )
    f1 = tmp_path / "lhs_col1.py"
    f2 = tmp_path / "lhs_col2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {"file": "lhs_col1.py", "start": 2, "end": 2, "name": "u1", "kind": "complex_expr"}
    u2 = {"file": "lhs_col2.py", "start": 2, "end": 2, "name": "u2", "kind": "complex_expr"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_u1_u2(b: Any, c: Any) -> Any:" in helper
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "+    arr[a] = _shared_u1_u2(b, c)" in patch
    assert "+    arr[idx] = _shared_u1_u2(a, b)" in patch
    assert "_shared_u1_u2(b, a)" not in patch

    # Asymmetric RHS arity after LHS pruning fails closed
    src3 = (
        "def store_three(arr, a, b):\n"
        "    arr[a] = a + b + 1\n"
    )
    f3 = tmp_path / "lhs_col3.py"
    f3.write_text(src3, encoding="utf-8")
    u3 = {"file": "lhs_col3.py", "start": 2, "end": 2, "name": "u3", "kind": "complex_expr"}
    u1_single = {
        "file": "lhs_col1.py",
        "start": 2,
        "end": 2,
        "name": "u1",
        "kind": "complex_expr",
    }
    f1.write_text("def store_one(arr, a, b):\n    arr[a] = b + 1\n", encoding="utf-8")
    assert synthesize_shared_helper_code(u1_single, u3, repo_root=str(tmp_path)) == ""
    assert (
        generate_refactoring_patch(
            [(0.95, u1_single, u3)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
        == ""
    )
