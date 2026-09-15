"""Unit tests for advanced AST structural clone detector features."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from pydoppelgangerhunt import (
    UnionFind,
    call_sequence_similarity,
    check_asymmetric_coverage,
    check_temporal_divergence,
    clone_pair_fingerprint,
    cluster_clone_families,
    colorize,
    compute_cyclomatic_complexity,
    compute_pair_similarity,
    compute_priority_score,
    compute_repository_dry_stats,
    compute_unit_coverage,
    emit_structured_report,
    extract_call_sequence,
    extract_unit_source_code,
    filter_clones_by_baseline,
    filter_clones_by_git_diff,
    format_github_annotations,
    format_json_report,
    format_markdown_summary,
    format_sarif_report,
    generate_clone_diff,
    generate_html_report,
    generate_refactoring_patch,
    get_ast_characteristic_vector,
    get_ast_shingles,
    get_ast_tokens,
    get_git_modified_line_ranges,
    harvest_file_units,
    harvest_notebook_units,
    init_tool_configuration,
    is_boilerplate_node,
    is_unit_in_modified_ranges,
    jaccard_similarity,
    lcs_alignment_similarity,
    load_baseline,
    load_toml_section,
    load_tool_config,
    merge_adjacent_clones,
    multiset_jaccard_similarity,
    parse_git_diff_hunks,
    read_coverage_data,
    record_baseline,
    scan_target,
    supports_color,
    suppress_subclones,
    synthesize_refactoring_suggestion,
    synthesize_shared_helper_code,
    tfidf_jaccard_similarity,
    tfidf_multiset_jaccard_similarity,
)
from pydoppelgangerhunt.parser import (
    _CommutativeCanonicalizer,
    _IdiomCanonicalizer,
    _compute_expression_complexity,
    _harvest_complex_expressions,
    check_inline_suppression,
)
import pydoppelgangerhunt

_extract_call_sequence = extract_call_sequence


def test_blind_indexing_distinguishes_variable_reuse() -> None:
    """Test that blind identifier indexing differentiates consistent reuse vs distinct variables."""
    # a + b + a vs x + y + x (both have reuse pattern: var_0, var_1, var_0)
    tree_reuse_1 = ast.parse("res = a + b + a")
    tree_reuse_2 = ast.parse("res = x + y + x")
    # x + y + z (no reuse: var_0, var_1, var_2)
    tree_distinct = ast.parse("res = x + y + z")

    # Without blind indexing, all Name nodes are normalized to 'VAR'
    shingles_default_1, _ = get_ast_shingles(tree_reuse_1, k=3, blind_indexing=False)
    shingles_default_distinct, _ = get_ast_shingles(tree_distinct, k=3, blind_indexing=False)
    assert jaccard_similarity(shingles_default_1, shingles_default_distinct) == 1.0

    # With blind indexing, reuse_1 and reuse_2 produce identical shingles
    shingles_blind_1, _ = get_ast_shingles(tree_reuse_1, k=3, blind_indexing=True)
    shingles_blind_2, _ = get_ast_shingles(tree_reuse_2, k=3, blind_indexing=True)
    assert jaccard_similarity(shingles_blind_1, shingles_blind_2) == 1.0

    # But reuse_1 and distinct produce different shingles
    shingles_blind_distinct, _ = get_ast_shingles(tree_distinct, k=3, blind_indexing=True)
    assert jaccard_similarity(shingles_blind_1, shingles_blind_distinct) < 1.0


def test_blind_indexing_attributes_and_arguments() -> None:
    """Test blind indexing on attributes and function arguments."""
    tree_attr_1 = ast.parse("def f(p1, p2): return self.a * self.b + self.a")
    tree_attr_2 = ast.parse("def g(q1, q2): return other.x * other.y + other.x")
    tree_attr_distinct = ast.parse("def h(r1, r2): return other.x * other.y + other.z")

    shingles_1, _ = get_ast_shingles(tree_attr_1, k=3, blind_indexing=True)
    shingles_2, _ = get_ast_shingles(tree_attr_2, k=3, blind_indexing=True)
    shingles_distinct, _ = get_ast_shingles(tree_attr_distinct, k=3, blind_indexing=True)

    assert jaccard_similarity(shingles_1, shingles_2) == 1.0
    assert jaccard_similarity(shingles_1, shingles_distinct) < 1.0


def test_compute_expression_complexity() -> None:
    """Test operator counting in expression complexity helper."""
    # Simple expression (1 op)
    tree_simple = ast.parse("x = a + b").body[0].value  # type: ignore[attr-defined]
    assert _compute_expression_complexity(tree_simple) == 1

    # Medium expression (3 ops: +, *, -)
    tree_med = ast.parse("x = (a + b) * (c - d)").body[0].value  # type: ignore[attr-defined]
    assert _compute_expression_complexity(tree_med) == 3

    # High complexity expression (5 ops: /, +, *, *, -)
    tree_high = ast.parse("x = (a * b + c * d) / (e - f)").body[0].value  # type: ignore[attr-defined]
    assert _compute_expression_complexity(tree_high) >= 4


def test_harvest_complex_expressions_outermost() -> None:
    """Test harvesting outermost complex expressions with complexity >= min_complexity."""
    source = '''
def calculate():
    # Simple expression (< 4 ops) -> ignored
    v1 = a + b
    # Complex expression (>= 4 ops) -> harvested as one unit
    v2 = (width * scale + pad_x * 2) / (aspect_ratio - margin)
    # Another complex comprehension
    v3 = [x * 2 + y * 3 for x, y in points if x > 0 and y > 0]
'''
    tree = ast.parse(source)
    func_node = tree.body[0]
    complex_units = _harvest_complex_expressions(func_node, min_complexity=4)
    assert len(complex_units) == 2

    # Verify line numbers are captured
    for node, start, end in complex_units:
        assert isinstance(node, ast.AST)
        assert start > 0
        assert end >= start


def test_merge_adjacent_clones_sliding_windows() -> None:
    """Test Deckard-style subtree/window merging on overlapping sliding window hits."""
    # Construct 3 simulated overlapping sliding window hits between func_a and func_b
    u1_w1 = {"file": "mod.py", "name": "func_a:stmts_1-5", "start": 10, "end": 15, "lines": 6, "shingles": {"s1", "s2", "s3"}, "token_count": 20, "kind": "sliding_window"}
    u2_w1 = {"file": "mod.py", "name": "func_b:stmts_1-5", "start": 30, "end": 35, "lines": 6, "shingles": {"s1", "s2", "s3"}, "token_count": 20, "kind": "sliding_window"}

    u1_w2 = {"file": "mod.py", "name": "func_a:stmts_2-6", "start": 11, "end": 16, "lines": 6, "shingles": {"s2", "s3", "s4"}, "token_count": 20, "kind": "sliding_window"}
    u2_w2 = {"file": "mod.py", "name": "func_b:stmts_2-6", "start": 31, "end": 36, "lines": 6, "shingles": {"s2", "s3", "s4"}, "token_count": 20, "kind": "sliding_window"}

    u1_w3 = {"file": "mod.py", "name": "func_a:stmts_3-7", "start": 12, "end": 17, "lines": 6, "shingles": {"s3", "s4", "s5"}, "token_count": 20, "kind": "sliding_window"}
    u2_w3 = {"file": "mod.py", "name": "func_b:stmts_3-7", "start": 32, "end": 37, "lines": 6, "shingles": {"s3", "s4", "s5"}, "token_count": 20, "kind": "sliding_window"}

    raw_clones = [
        (1.0, u1_w1, u2_w1),
        (1.0, u1_w2, u2_w2),
        (1.0, u1_w3, u2_w3),
    ]

    merged = merge_adjacent_clones(raw_clones, line_tolerance=2)
    # The 3 overlapping hits should be consolidated into 1 maximal clone region
    assert len(merged) == 1
    sim, m1, m2 = merged[0]
    assert sim == 1.0
    assert m1["start"] == 10
    assert m1["end"] == 17
    assert m2["start"] == 30
    assert m2["end"] == 37
    assert "merged_10-17" in m1["name"]
    assert "merged_30-37" in m2["name"]


def test_merge_adjacent_clones_does_not_merge_unrelated() -> None:
    """Test that distinct functions or non-overlapping line ranges are preserved."""
    u1_a = {"file": "mod.py", "name": "func_a:stmts_1-5", "start": 10, "end": 15, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}
    u2_a = {"file": "mod.py", "name": "func_b:stmts_1-5", "start": 30, "end": 35, "lines": 6, "shingles": {"s1"}, "token_count": 20, "kind": "sliding_window"}

    # Disjoint lines far away
    u1_b = {"file": "mod.py", "name": "func_a:stmts_20-25", "start": 80, "end": 85, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}
    u2_b = {"file": "mod.py", "name": "func_b:stmts_20-25", "start": 100, "end": 105, "lines": 6, "shingles": {"s2"}, "token_count": 20, "kind": "sliding_window"}

    raw_clones = [
        (1.0, u1_a, u2_a),
        (1.0, u1_b, u2_b),
    ]
    merged = merge_adjacent_clones(raw_clones, line_tolerance=2)
    assert len(merged) == 2


def test_scan_target_with_advanced_options(tmp_path: Path) -> None:
    """End-to-end scan_target integration test verifying all three advanced capabilities."""
    pkg = tmp_path / "sample_pkg"
    pkg.mkdir()

    file_a = pkg / "engine_a.py"
    file_a.write_text(
        '''"""Module A."""

def compute_structural_stiffness(span: float, depth: float, width: float, mod: float) -> float:
    """Calculate beam structural stiffness."""
    v1 = span * depth
    v2 = depth * depth
    v3 = width * v2
    v4 = v3 * mod
    v5 = v4 / 12.0
    v6 = v5 * 0.95
    v7 = v6 + 1.0
    v8 = v7 * 2.0
    return v8

def helper_formula_alpha(a: float, b: float, c: float, d: float, e: float) -> float:
    return (a * b + c * d) / (e - 1.0)
''',
        encoding="utf-8",
    )

    file_b = pkg / "engine_b.py"
    file_b.write_text(
        '''"""Module B."""

def compute_member_rigidity(length: float, height: float, thick: float, modulus: float) -> float:
    """Calculate member rigidity."""
    s1 = length * height
    s2 = height * height
    s3 = thick * s2
    s4 = s3 * modulus
    s5 = s4 / 12.0
    s6 = s5 * 0.95
    s7 = s6 + 1.0
    s8 = s7 * 2.0
    return s8

def helper_formula_beta(w: float, x: float, y: float, z: float, k: float) -> float:
    return (w * x + y * z) / (k - 1.0)
''',
        encoding="utf-8",
    )

    # 1. Sliding window alone yields multiple overlapping hits
    clones_raw = scan_target(
        str(pkg),
        min_lines=4,
        min_tokens=10,
        threshold=0.85,
        sliding_window=True,
        window_size=4,
        merge_subtrees=False,
    )
    assert len(clones_raw) >= 3

    # 2. Sliding window with merge_subtrees consolidates into a single maximal clone region
    clones_merged = scan_target(
        str(pkg),
        min_lines=4,
        min_tokens=10,
        threshold=0.85,
        sliding_window=True,
        window_size=4,
        merge_subtrees=True,
    )
    assert len(clones_merged) == 1
    sim, u1, u2 = clones_merged[0]
    assert sim >= 0.85
    assert u1["lines"] >= 8
    assert u2["lines"] >= 8

    # 3. Complex expressions audit detects helper_formula clone
    clones_expr = scan_target(
        str(pkg),
        min_lines=1,
        min_tokens=8,
        threshold=0.85,
        functions_only=True,
        complex_expressions=True,
        min_expr_complexity=4,
    )
    assert any("expr" in c[1]["name"] or "expr" in c[2]["name"] for c in clones_expr)

    # 4. Blind indexing audit verifies consistent reuse matching
    clones_blind = scan_target(
        str(pkg),
        min_lines=4,
        min_tokens=10,
        threshold=0.85,
        sliding_window=True,
        window_size=4,
        blind_indexing=True,
        merge_subtrees=True,
    )
    assert len(clones_blind) == 1


def test_strip_annotations_equivalence() -> None:
    """Test that strip_annotations normalizes typed vs untyped logic into identical AST shingles."""
    typed_src = """
def compute_metrics(x: float, y: Dict[str, Any], limit: Optional[int] = None) -> Tuple[bool, List[float]]:
    res: List[float] = []
    for item in y.values():
        if item > x:
            res.append(item * 2.0)
    return True, res
"""
    untyped_src = """
def compute_metrics(x, y, limit=None):
    res = []
    for item in y.values():
        if item > x:
            res.append(item * 2.0)
    return True, res
"""
    tree_typed = ast.parse(typed_src)
    tree_untyped = ast.parse(untyped_src)

    # Without strip_annotations, type nodes alter shingle tokens
    sh_typed_raw, _ = get_ast_shingles(tree_typed, k=3, strip_annotations=False)
    sh_untyped_raw, _ = get_ast_shingles(tree_untyped, k=3, strip_annotations=False)
    assert jaccard_similarity(sh_typed_raw, sh_untyped_raw) < 1.0

    # With strip_annotations, they produce exactly 1.0 similarity
    sh_typed_clean, _ = get_ast_shingles(tree_typed, k=3, strip_annotations=True)
    sh_untyped_clean, _ = get_ast_shingles(tree_untyped, k=3, strip_annotations=True)
    assert jaccard_similarity(sh_typed_clean, sh_untyped_clean) == 1.0


def test_clause_level_branch_detection(tmp_path: Path) -> None:
    """Test branch- and handler-level clone slicing on if/else and try/except clauses."""
    pkg = tmp_path / "clause_pkg"
    pkg.mkdir()

    file_a = pkg / "handler_a.py"
    file_a.write_text(
        '''"""Module A."""
def run_job_alpha(data):
    # Long distinct preface
    x = 10 + 20
    y = x * 30
    z = y / 4.0
    w = z + 100
    try:
        raw = data.get("key")
        step = raw * 2
        value = step + 1
        return value
    except KeyError as exc:
        log.error("Failed key lookup", exc)
        metrics.increment("errors")
        fallback = default_data()
        return fallback
''',
        encoding="utf-8",
    )

    file_b = pkg / "handler_b.py"
    file_b.write_text(
        '''"""Module B."""
def execute_step_beta(payload):
    # Completely different preface
    m1 = "starting"
    m2 = m1.upper()
    try:
        raw = payload.get("id")
        inv = 1.0 / raw
        value = math.sqrt(inv)
        return value
    except KeyError as err:
        log.error("Failed key lookup", err)
        metrics.increment("errors")
        fallback = default_data()
        return fallback
''',
        encoding="utf-8",
    )

    # Without clause_level, whole functions and compound blocks differ and no clones are found
    clones_default = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        clause_level=False,
    )
    assert len(clones_default) == 0

    # With clause_level=True, the except handler clause clone is discovered
    clones_clauses = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        clause_level=True,
    )
    assert len(clones_clauses) >= 1
    sim, u1, u2 = clones_clauses[0]
    assert sim >= 0.85
    assert "except" in u1["name"] and "except" in u2["name"]


def test_data_table_clone_detection(tmp_path: Path) -> None:
    """Test clone detection on module-level dictionary/list/tuple configuration tables."""
    pkg = tmp_path / "table_pkg"
    pkg.mkdir()

    file_a = pkg / "config_a.py"
    file_a.write_text(
        '''"""Configuration A."""
SCHEMA_DEFAULTS = {
    "wall_min_thickness": 4.5,
    "door_standard_width": 36.0,
    "window_height_offset": 42.0,
    "hallway_clearance": 48.0,
    "ceiling_default_height": 96.0,
}
''',
        encoding="utf-8",
    )

    file_b = pkg / "config_b.py"
    file_b.write_text(
        '''"""Configuration B."""
LAYOUT_SETTINGS = {
    "wall_min_thickness": 4.5,
    "door_standard_width": 36.0,
    "window_height_offset": 42.0,
    "hallway_clearance": 48.0,
    "ceiling_default_height": 96.0,
}
''',
        encoding="utf-8",
    )

    # Without data_tables, module level dictionary assignments are ignored
    clones_no_tables = scan_target(
        str(pkg),
        min_lines=1,
        min_tokens=10,
        threshold=0.85,
        data_tables=False,
    )
    assert len(clones_no_tables) == 0

    # With data_tables=True, table clone is detected
    clones_tables = scan_target(
        str(pkg),
        min_lines=1,
        min_tokens=10,
        threshold=0.85,
        data_tables=True,
    )
    assert len(clones_tables) == 1
    sim, u1, u2 = clones_tables[0]
    assert sim >= 0.85
    assert "table:" in u1["name"] and "table:" in u2["name"]


def test_nms_subclone_suppression() -> None:
    """Test Non-Maximum Suppression (subclone elimination) drops contained child clones."""
    # Parent clone covering lines 10-50 in file1.py and 20-60 in file2.py
    parent_u1 = {"file": "file1.py", "name": "func_a", "start": 10, "end": 50, "lines": 41, "shingles": {"s1"}, "token_count": 50, "kind": "function"}
    parent_u2 = {"file": "file2.py", "name": "func_b", "start": 20, "end": 60, "lines": 41, "shingles": {"s1"}, "token_count": 50, "kind": "function"}

    # Child subclone covering lines 15-25 in file1.py and 25-35 in file2.py (strictly enclosed)
    child_u1 = {"file": "file1.py", "name": "func_a:stmts_1-5", "start": 15, "end": 25, "lines": 11, "shingles": {"s2"}, "token_count": 15, "kind": "sliding_window"}
    child_u2 = {"file": "file2.py", "name": "func_b:stmts_1-5", "start": 25, "end": 35, "lines": 11, "shingles": {"s2"}, "token_count": 15, "kind": "sliding_window"}

    # Independent clone covering lines 80-100 in file1.py and 90-110 in file2.py (disjoint)
    indep_u1 = {"file": "file1.py", "name": "func_c", "start": 80, "end": 100, "lines": 21, "shingles": {"s3"}, "token_count": 30, "kind": "function"}
    indep_u2 = {"file": "file2.py", "name": "func_d", "start": 90, "end": 110, "lines": 21, "shingles": {"s3"}, "token_count": 30, "kind": "function"}

    clones = [
        (1.0, parent_u1, parent_u2),
        (0.95, child_u1, child_u2),
        (0.90, indep_u1, indep_u2),
    ]

    suppressed = suppress_subclones(clones, sim_tolerance=0.10)
    assert len(suppressed) == 2
    # Only parent and independent clone remain; child is suppressed
    assert suppressed[0][1]["name"] == "func_a"
    assert suppressed[1][1]["name"] == "func_c"


def test_class_level_clone_detection(tmp_path: Path) -> None:
    """Test that --class-level detects structural duplicates across class definitions."""
    pkg = tmp_path / "models_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    class_a = pkg / "model_a.py"
    class_a.write_text(
        '''"""Class A."""
class BuildingSensor:
    id: str
    pin: int
    threshold: float
    active: bool

    def read_metric(self) -> float:
        val = self.pin * 1.5
        return val if val > self.threshold else 0.0
''',
        encoding="utf-8",
    )

    class_b = pkg / "model_b.py"
    class_b.write_text(
        '''"""Class B."""
class EnvironmentalProbe:
    id: str
    pin: int
    threshold: float
    active: bool

    def read_metric(self) -> float:
        val = self.pin * 1.5
        return val if val > self.threshold else 0.0
''',
        encoding="utf-8",
    )

    # Without class_level, classes alone (with short method) are not harvested as class units
    clones_no_class = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        class_level=False,
    )
    assert not any("class:" in c[1]["name"] for c in clones_no_class)

    # With class_level=True, class unit clone is detected
    clones_class = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        class_level=True,
    )
    class_clones = [c for c in clones_class if "class:" in c[1]["name"]]
    assert len(class_clones) >= 1
    sim, u1, u2 = class_clones[0]
    assert sim >= 0.85
    assert "class:" in u1["name"] and "class:" in u2["name"]


def test_blind_literals_parameterized_clones() -> None:
    """Test that --blind-literals allows parameter-modified constants and error strings to match at 100%."""
    code_1 = '''
def validate_clearance_zone_a(dim: float) -> bool:
    """Validates clearance zone A."""
    if dim < 36.0:
        msg = "Zone A clearance violation: minimum 36.0 in required"
        raise ValueError(msg)
    offset = dim + 12.0
    return offset >= 48.0
'''
    code_2 = '''
def validate_clearance_zone_b(dim: float) -> bool:
    """Validates clearance zone B with different documentation."""
    if dim < 42.0:
        msg = "Zone B clearance error: min 42.0 in"
        raise ValueError(msg)
    offset = dim + 15.0
    return offset >= 57.0
'''
    tree_1 = ast.parse(code_1)
    tree_2 = ast.parse(code_2)

    # Without blind_literals, constant string lengths/contents and different numbers create different shingles
    shingles_default_1, _ = get_ast_shingles(tree_1, k=3, blind_literals=False)
    shingles_default_2, _ = get_ast_shingles(tree_2, k=3, blind_literals=False)
    sim_default = jaccard_similarity(shingles_default_1, shingles_default_2)

    # With blind_literals, all constants map to LITERAL and docstrings are stripped
    shingles_blind_1, _ = get_ast_shingles(tree_1, k=3, blind_literals=True)
    shingles_blind_2, _ = get_ast_shingles(tree_2, k=3, blind_literals=True)
    sim_blind = jaccard_similarity(shingles_blind_1, shingles_blind_2)

    assert sim_blind == 1.0
    assert sim_default <= sim_blind


def test_bag_of_tokens_permutation_invariance() -> None:
    """Test that multiset characteristic vectors (Deckard-style) are permutation-invariant."""
    # Two functions with the same intermediate calculations executed in commutative/reordered order
    code_seq_1 = '''
def compute_metrics_v1(x, y, z):
    a = x * 2.0
    b = y + 5.0
    c = z / 3.0
    return a + b - c
'''
    code_seq_2 = '''
def compute_metrics_v2(x, y, z):
    c = z / 3.0
    a = x * 2.0
    b = y + 5.0
    return a + b - c
'''
    tree_1 = ast.parse(code_seq_1)
    tree_2 = ast.parse(code_seq_2)

    vec_1 = get_ast_characteristic_vector(tree_1)
    vec_2 = get_ast_characteristic_vector(tree_2)

    # Permuted statements have identical multiset frequencies
    assert multiset_jaccard_similarity(vec_1, vec_2) == 1.0

    # But contiguous shingles are penalized due to reordered sequence boundary transitions
    shingles_1, _ = get_ast_shingles(tree_1, k=3)
    shingles_2, _ = get_ast_shingles(tree_2, k=3)
    shingle_sim = jaccard_similarity(shingles_1, shingles_2)
    assert shingle_sim < 1.0


def test_filter_boilerplate_telemetry_filtering() -> None:
    """Test that --filter-boilerplate removes logging, print, and assertions before comparison."""
    code_clean = '''
def calculate_airflow_cfm(area_sqft: float, ach: float) -> float:
    volume_cuft = area_sqft * 9.0
    cfm = (volume_cuft * ach) / 60.0
    return cfm
'''
    code_instrumented = '''
def calculate_airflow_cfm_instrumented(area_sqft: float, ach: float) -> float:
    logger.info("Computing airflow rate...")
    assert area_sqft > 0, "Invalid room area"
    volume_cuft = area_sqft * 9.0
    print(f"Debug: volume is {volume_cuft}")
    self._log(f"Volume: {volume_cuft}")
    cfm = (volume_cuft * ach) / 60.0
    logging.debug("Finished airflow calculation")
    return cfm
'''
    tree_clean = ast.parse(code_clean)
    tree_inst = ast.parse(code_instrumented)

    # Without filter_boilerplate, logging/prints/asserts alter shingles significantly
    shingles_no_filter_1, _ = get_ast_shingles(tree_clean, k=3, filter_boilerplate=False)
    shingles_no_filter_2, _ = get_ast_shingles(tree_inst, k=3, filter_boilerplate=False)
    sim_unfiltered = jaccard_similarity(shingles_no_filter_1, shingles_no_filter_2)
    assert sim_unfiltered < 0.70

    # With filter_boilerplate=True, the non-algorithmic telemetry is stripped
    shingles_filtered_1, _ = get_ast_shingles(tree_clean, k=3, filter_boilerplate=True)
    shingles_filtered_2, _ = get_ast_shingles(tree_inst, k=3, filter_boilerplate=True)
    sim_filtered = jaccard_similarity(shingles_filtered_1, shingles_filtered_2)
    assert sim_filtered == 1.0


def test_consistent_renaming_preserves_builtins_and_scopes() -> None:
    """Test that consistent alpha-renaming preserves Python builtins while scoping variables."""
    code_a = """
def calc_surface(width: float, height: float) -> float:
    area = width * height
    return round(area, 2)
"""
    code_b = """
def compute_area(w: float, h: float) -> float:
    total = w * h
    return round(total, 2)
"""
    code_different_builtin = """
def inspect_len(w: float, h: float) -> float:
    total = w * h
    return len(total, 2)
"""
    tree_a = ast.parse(code_a)
    tree_b = ast.parse(code_b)
    tree_diff = ast.parse(code_different_builtin)

    # Identical structure with renamed variables should match 100% under consistent renaming
    sh_a, _ = get_ast_shingles(tree_a, k=3, consistent_renaming=True)
    sh_b, _ = get_ast_shingles(tree_b, k=3, consistent_renaming=True)
    assert jaccard_similarity(sh_a, sh_b) == 1.0

    # Naive blind indexing conflates round() and len() because both become generic VAR_n
    sh_blind_a, _ = get_ast_shingles(tree_a, k=3, blind_indexing=True)
    sh_blind_diff, _ = get_ast_shingles(tree_diff, k=3, blind_indexing=True)
    assert jaccard_similarity(sh_blind_a, sh_blind_diff) == 1.0

    # Consistent renaming preserves BUILTIN_round vs BUILTIN_len, preventing false positives
    sh_diff, _ = get_ast_shingles(tree_diff, k=3, consistent_renaming=True)
    assert jaccard_similarity(sh_a, sh_diff) < 1.0


def test_tfidf_shingle_weighting_elevates_discriminative_tokens() -> None:
    """Test that TF-IDF weighted Jaccard downweights ubiquitous syntax and rewards rare shingles."""
    shingles_1 = {"boilerplate_1", "boilerplate_2", "rare_domain_logic"}
    shingles_2 = {"boilerplate_1", "boilerplate_2", "unrelated_rare_logic"}
    shingles_3 = {"boilerplate_x", "boilerplate_y", "rare_domain_logic"}

    # Ubiquitous boilerplate has low IDF (df=1000), rare domain logic has high IDF (df=2)
    idf_weights = {
        "boilerplate_1": 1.1,
        "boilerplate_2": 1.1,
        "boilerplate_x": 1.1,
        "boilerplate_y": 1.1,
        "rare_domain_logic": 6.5,
        "unrelated_rare_logic": 6.5,
    }

    # Unweighted Jaccard thinks (1, 2) is more similar than (1, 3) because of 2 shared boilerplate shingles
    unweighted_sim_1_2 = jaccard_similarity(shingles_1, shingles_2)  # 2/4 = 0.50
    unweighted_sim_1_3 = jaccard_similarity(shingles_1, shingles_3)  # 1/5 = 0.20
    assert unweighted_sim_1_2 > unweighted_sim_1_3

    # TF-IDF weighted Jaccard elevates the shared domain logic over ubiquitous boilerplate
    tfidf_sim_1_2 = tfidf_jaccard_similarity(shingles_1, shingles_2, idf_weights)
    tfidf_sim_1_3 = tfidf_jaccard_similarity(shingles_1, shingles_3, idf_weights)
    assert tfidf_sim_1_3 > tfidf_sim_1_2

    # Also test multiset TF-IDF similarity
    vec_1 = {"boilerplate_1": 2, "rare_domain_logic": 1}
    vec_2 = {"boilerplate_1": 2, "unrelated_rare_logic": 1}
    vec_3 = {"boilerplate_x": 2, "rare_domain_logic": 1}
    tfidf_vec_1_2 = tfidf_multiset_jaccard_similarity(vec_1, vec_2, idf_weights)
    tfidf_vec_1_3 = tfidf_multiset_jaccard_similarity(vec_1, vec_3, idf_weights)
    assert tfidf_vec_1_3 > tfidf_vec_1_2


def test_harvest_closures_scopes_inner_functions(tmp_path: Path) -> None:
    """Test that --harvest-closures captures inner closures with qualified parent names."""
    pkg = tmp_path / "closure_pkg"
    pkg.mkdir()

    factory_file = pkg / "factories.py"
    factory_file.write_text(
        '''
def make_threshold_rule(min_val: float):
    def evaluate(self, context) -> bool:
        v = context.get_val()
        if v < min_val:
            return False
        return True
    return evaluate

def make_other_rule(limit_val: float):
    def evaluate(self, context) -> bool:
        v = context.get_val()
        if v < limit_val:
            return False
        return True
    return evaluate
''',
        encoding="utf-8",
    )

    # When harvest_closures is True, the inner evaluate functions are harvested and matched
    clones = scan_target(
        str(pkg),
        threshold=0.85,
        min_lines=4,
        harvest_closures=True,
    )
    assert len(clones) >= 1
    found_closure_pair = any(
        "evaluate" in c[1]["name"] and "evaluate" in c[2]["name"]
        for c in clones
    )
    assert found_closure_pair


def test_commutative_canonicalizer_boolop() -> None:
    """Test that _CommutativeCanonicalizer normalizes boolean operands (and, or)."""
    t1 = ast.parse("def f(a, b):\n    return a.x and b.y")
    t2 = ast.parse("def f(a, b):\n    return b.y and a.x")

    # Without canonicalization, consistent renaming yields different ARG order
    tok1_raw = get_ast_tokens(t1.body[0], consistent_renaming=True)
    tok2_raw = get_ast_tokens(t2.body[0], consistent_renaming=True)
    assert tok1_raw != tok2_raw

    # With canonicalization, operands are sorted deterministically
    c1 = _CommutativeCanonicalizer().visit(t1)
    c2 = _CommutativeCanonicalizer().visit(t2)
    tok1_c = get_ast_tokens(c1.body[0], consistent_renaming=True)
    tok2_c = get_ast_tokens(c2.body[0], consistent_renaming=True)
    assert tok1_c == tok2_c


def test_commutative_canonicalizer_binop() -> None:
    """Test that _CommutativeCanonicalizer normalizes commutative binary arithmetic (+, *)."""
    t1 = ast.parse("res = x * y + z")
    t2 = ast.parse("res = y * x + z")

    c1 = _CommutativeCanonicalizer().visit(t1)
    c2 = _CommutativeCanonicalizer().visit(t2)
    tok1 = get_ast_tokens(c1.body[0])
    tok2 = get_ast_tokens(c2.body[0])
    assert tok1 == tok2


def test_commutative_canonicalizer_compare() -> None:
    """Test that _CommutativeCanonicalizer normalizes symmetric comparisons (==, !=)."""
    t1 = ast.parse("if u == v:\n    pass")
    t2 = ast.parse("if v == u:\n    pass")

    c1 = _CommutativeCanonicalizer().visit(t1)
    c2 = _CommutativeCanonicalizer().visit(t2)
    tok1 = get_ast_tokens(c1.body[0])
    tok2 = get_ast_tokens(c2.body[0])
    assert tok1 == tok2


def test_scan_target_commutative_flag(tmp_path: Path) -> None:
    """Test that scan_target with commutative=True detects commutative duplicates."""
    pkg = tmp_path / "comm_pkg"
    pkg.mkdir()

    f1 = pkg / "mod1.py"
    f1.write_text(
        '''
def compute_total(a, b, c, d):
    part1 = a + b
    part2 = c * d
    return part1 + part2
''',
        encoding="utf-8",
    )

    f2 = pkg / "mod2.py"
    f2.write_text(
        '''
def compute_total(a, b, c, d):
    part1 = b + a
    part2 = d * c
    return part2 + part1
''',
        encoding="utf-8",
    )

    # With commutative=True, both functions produce identical AST structures
    clones = scan_target(
        str(pkg),
        threshold=0.95,
        min_lines=4,
        commutative=True,
    )
    assert len(clones) >= 1
    sim, u1, u2 = clones[0]
    assert sim >= 0.95
    assert u1["name"] == "compute_total"
    assert u2["name"] == "compute_total"


def test_scan_target_comprehensions_flag(tmp_path: Path) -> None:
    """Test that scan_target with comprehensions=True harvests list, dict, set comprehensions."""
    pkg = tmp_path / "comp_pkg"
    pkg.mkdir()

    f1 = pkg / "rooms.py"
    f1.write_text(
        '''
def get_active_rooms(floor):
    active = [r for r in floor.rooms if r.area > 10.0 and r.is_active]
    return active
''',
        encoding="utf-8",
    )

    f2 = pkg / "spaces.py"
    f2.write_text(
        '''
def get_qualifying_spaces(building):
    active = [s for s in building.spaces if s.area > 10.0 and s.is_active]
    return active
''',
        encoding="utf-8",
    )

    clones = scan_target(
        str(pkg),
        threshold=0.85,
        min_lines=1,
        min_tokens=10,
        blind_indexing=True,
        comprehensions=True,
    )
    assert len(clones) >= 1
    found_comp = any(
        "listcomp" in c[1]["name"] and "listcomp" in c[2]["name"]
        for c in clones
    )
    assert found_comp


def test_dictcomp_and_setcomp_harvesting(tmp_path: Path) -> None:
    """Test that dict and set comprehensions are harvested and compared."""
    pkg = tmp_path / "dict_comp_pkg"
    pkg.mkdir()

    f1 = pkg / "indexer.py"
    f1.write_text(
        '''
def build_map(entities):
    lookup = {e.uid: e.value for e in entities if e.is_valid and e.weight > 0}
    return lookup
''',
        encoding="utf-8",
    )

    f2 = pkg / "other_indexer.py"
    f2.write_text(
        '''
def build_registry(items):
    lookup = {i.uid: i.value for i in items if i.is_valid and i.weight > 0}
    return lookup
''',
        encoding="utf-8",
    )

    clones = scan_target(
        str(pkg),
        threshold=0.85,
        min_lines=1,
        min_tokens=10,
        blind_indexing=True,
        comprehensions=True,
    )
    assert len(clones) >= 1
    found_dictcomp = any(
        "dictcomp" in c[1]["name"] and "dictcomp" in c[2]["name"]
        for c in clones
    )
    assert found_dictcomp


def test_idiom_canonicalizer_loop_to_comprehension() -> None:
    """Test that _IdiomCanonicalizer transforms loop accumulators into ListComp assignments."""
    s_loop = """
res = []
for x in items:
    if x > 0:
        res.append(x * 2)
"""
    s_comp = """
res = [x * 2 for x in items if x > 0]
"""
    t_loop = ast.parse(s_loop)
    _IdiomCanonicalizer().visit(t_loop)
    ast.fix_missing_locations(t_loop)

    t_comp = ast.parse(s_comp)

    tok_loop = get_ast_tokens(t_loop.body[1])
    tok_comp = get_ast_tokens(t_comp.body[0])
    assert tok_loop == tok_comp


def test_idiom_canonicalizer_loop_to_any_all() -> None:
    """Test that _IdiomCanonicalizer transforms search loops into any() / all() generator expressions."""
    s_any_loop = """
for item in items:
    if item.is_valid:
        return True
"""
    s_any_call = """
return any(item.is_valid for item in items)
"""
    t_any_loop = ast.parse(s_any_loop)
    _IdiomCanonicalizer().visit(t_any_loop)
    ast.fix_missing_locations(t_any_loop)
    t_any_call = ast.parse(s_any_call)

    assert get_ast_tokens(t_any_loop.body[0]) == get_ast_tokens(t_any_call.body[0])

    s_all_loop = """
for item in items:
    if not item.is_valid:
        return False
"""
    s_all_call = """
return all(item.is_valid for item in items)
"""
    t_all_loop = ast.parse(s_all_loop)
    _IdiomCanonicalizer().visit(t_all_loop)
    ast.fix_missing_locations(t_all_loop)
    t_all_call = ast.parse(s_all_call)

    assert get_ast_tokens(t_all_loop.body[0]) == get_ast_tokens(t_all_call.body[0])


def test_abstract_expressions_nicad_type32() -> None:
    """Test that NiCad Type-3-2 dynamic expression abstraction equates algorithmic twins."""
    s1 = """
def eval_policy_a(req, metrics, key):
    val = safe_float(req)
    if val is not None and metrics[key] > val:
        score = (val * 100.0 + 5.0) / 2.0
        return score
    return 0.0
"""
    s2 = """
def eval_policy_b(req, metrics, key):
    val = safe_float(req)
    if val is not None and metrics[key] > val:
        score = (val ** 1.85 - 12.3) * 4.5
        return score
    return 0.0
"""
    t1 = ast.parse(s1)
    t2 = ast.parse(s2)

    sh1_raw, _ = get_ast_shingles(t1, k=3, abstract_expressions=False)
    sh2_raw, _ = get_ast_shingles(t2, k=3, abstract_expressions=False)
    sim_raw = jaccard_similarity(sh1_raw, sh2_raw)

    sh1_abs, _ = get_ast_shingles(t1, k=3, abstract_expressions=True)
    sh2_abs, _ = get_ast_shingles(t2, k=3, abstract_expressions=True)
    sim_abs = jaccard_similarity(sh1_abs, sh2_abs)

    assert sim_abs > sim_raw
    assert sim_abs == 1.0


def test_lcs_alignment_gapped_similarity() -> None:
    """Test CCAligner-style Longest Common Subsequence alignment similarity for gapped clones."""
    tok_clean = ["Assign", "VAR", "CONST", "If", "VAR", "Return", "CONST"] * 4
    # Insert two extra statements in the middle of tok_clean
    tok_gapped = (
        tok_clean[:14]
        + ["EXTRA_LOG_A", "EXTRA_LOG_B"]
        + tok_clean[14:]
    )
    sim_lcs = lcs_alignment_similarity(tok_clean, tok_gapped)
    assert sim_lcs >= 0.85

    # Empty handling
    assert lcs_alignment_similarity([], tok_gapped) == 0.0
    assert lcs_alignment_similarity(tok_clean, []) == 0.0


def test_scan_target_with_idioms_flag(tmp_path: Path) -> None:
    """Test that scan_target with idioms=True detects equivalence between accumulator loop and listcomp."""
    pkg = tmp_path / "idioms_pkg"
    pkg.mkdir()

    f1 = pkg / "loop_mod.py"
    f1.write_text(
        '''
def process_data(items):
    out = []
    for x in items:
        if x > 0:
            out.append(x * 2)
    return out
''',
        encoding="utf-8",
    )

    f2 = pkg / "comp_mod.py"
    f2.write_text(
        '''
def process_data(items):
    out = []
    out = [x * 2 for x in items if x > 0]
    return out
''',
        encoding="utf-8",
    )

    clones = scan_target(
        str(pkg),
        threshold=0.85,
        min_lines=4,
        idioms=True,
    )
    assert len(clones) >= 1
    assert clones[0][0] >= 0.85


def test_scan_target_with_abstract_expressions(tmp_path: Path) -> None:
    """Test that scan_target with abstract_expressions=True detects algorithmic twins."""
    pkg = tmp_path / "abstract_pkg"
    pkg.mkdir()

    f1 = pkg / "algo_a.py"
    f1.write_text(
        '''
def compute_metric(val, thresh):
    if val > thresh:
        res = (val * 10.0 + 2.0) / 3.0
        return res
    return 0.0
''',
        encoding="utf-8",
    )

    f2 = pkg / "algo_b.py"
    f2.write_text(
        '''
def compute_metric(val, thresh):
    if val > thresh:
        res = (val ** 2.5 - 7.0) * 1.5
        return res
    return 0.0
''',
        encoding="utf-8",
    )

    clones = scan_target(
        str(pkg),
        threshold=0.90,
        min_lines=4,
        abstract_expressions=True,
    )
    assert len(clones) >= 1
    assert clones[0][0] >= 0.90


def test_scan_target_with_gapped_tolerance(tmp_path: Path) -> None:
    """Test that scan_target with gapped_tolerance=True detects clones with inserted statements."""
    pkg = tmp_path / "gapped_pkg"
    pkg.mkdir()

    f1 = pkg / "base.py"
    f1.write_text(
        '''
def run_pipeline(data):
    v1 = data["a"]
    v2 = data["b"]
    v3 = data["c"]
    v4 = data["d"]
    return v1 + v2 + v3 + v4
''',
        encoding="utf-8",
    )

    f2 = pkg / "gapped.py"
    f2.write_text(
        '''
def run_pipeline(data):
    v1 = data["a"]
    v2 = data["b"]
    intermediate_check = True
    v3 = data["c"]
    v4 = data["d"]
    return v1 + v2 + v3 + v4
''',
        encoding="utf-8",
    )

    clones = scan_target(
        str(pkg),
        threshold=0.80,
        min_lines=5,
        gapped_tolerance=True,
    )
    assert len(clones) >= 1
    assert clones[0][0] >= 0.80


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


def test_call_sequence_extraction_and_similarity() -> None:
    """Test extraction of call sequences and procedural pipeline trace matching."""
    code_a = """
def pipeline_alpha(data):
    v = fetch_data(data)
    clean = sanitize(v)
    res = compute_physics(clean)
    return format_output(res)
"""
    code_b = """
def pipeline_beta(payload):
    # Differs in structure but executes identical call pipeline
    d = fetch_data(payload)
    if not d:
        return None
    cleaned = sanitize(d)
    val = compute_physics(cleaned)
    out = format_output(val)
    return out
"""
    tree_a = ast.parse(code_a)
    tree_b = ast.parse(code_b)

    seq_a = _extract_call_sequence(tree_a)
    seq_b = _extract_call_sequence(tree_b)
    assert seq_a == ["fetch_data", "sanitize", "compute_physics", "format_output"]
    assert seq_b == ["fetch_data", "sanitize", "compute_physics", "format_output"]

    sim = call_sequence_similarity(seq_a, seq_b)
    assert sim == 1.0


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


def test_sourcerercc_length_bound_pruning() -> None:
    """Test mathematical upper bound pruning for LCS alignment."""
    # When tokens differ greatly in size, max possible LCS is 2 * min / (len1 + len2)
    # E.g. len 10 and len 100 -> 2 * 10 / 110 = 0.1818
    tokens_short = ["a", "b", "c", "d", "e"] * 2  # len 10
    tokens_long = ["a", "b", "c", "d", "e"] * 20  # len 100
    # With threshold = 0.50, max_possible (0.1818) is well below 0.50 -> returns 0.0 immediately
    sim = lcs_alignment_similarity(tokens_short, tokens_long, threshold=0.50)
    assert sim == 0.0

    # With threshold = None, computes actual LCS similarity
    sim_actual = lcs_alignment_similarity(tokens_short, tokens_long, threshold=None)
    assert 0.0 < sim_actual <= 0.20


def test_audit_tests_parametrize_candidate_detection(tmp_path: Path) -> None:
    """Test detection of copy-pasted test functions as @pytest.mark.parametrize candidates."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_example.py"
    test_file.write_text(
        """
def test_calc_tier_low():
    val = 10.0
    res = val * 2.0
    assert res == 20.0

def test_calc_tier_high():
    val = 50.0
    res = val * 2.0
    assert res == 100.0
""",
        encoding="utf-8",
    )
    clones = scan_target(
        str(test_dir),
        threshold=0.80,
        min_lines=3,
        min_tokens=5,
        audit_tests=True,
        blind_literals=True,
    )
    assert len(clones) >= 1
    u1, u2 = clones[0][1], clones[0][2]
    assert u1["name"] == "test_calc_tier_low" or u2["name"] == "test_calc_tier_low"


def test_union_find_clone_family_clustering() -> None:
    """Test UnionFind data structure and connected-component Clone Family clustering."""
    uf = UnionFind()
    assert uf.find("A") == "A"
    assert uf.find("B") == "B"
    root_ab = uf.union("A", "B")
    assert uf.find("A") == uf.find("B") == root_ab

    uf.union("B", "C")
    assert uf.find("C") == root_ab

    u_a = {"file": "mod_a.py", "start": 10, "end": 20, "name": "fn_a"}
    u_b = {"file": "mod_b.py", "start": 15, "end": 25, "name": "fn_b"}
    u_c = {"file": "mod_c.py", "start": 30, "end": 40, "name": "fn_c"}
    u_d = {"file": "mod_d.py", "start": 5, "end": 15, "name": "fn_d"}
    u_e = {"file": "mod_e.py", "start": 50, "end": 60, "name": "fn_e"}

    mock_clones = [
        (0.95, u_a, u_b),
        (0.92, u_b, u_c),
        (0.88, u_d, u_e),
    ]

    families = cluster_clone_families(mock_clones)
    assert len(families) == 2

    fam_3 = next(f for f in families if f["member_count"] == 3)
    assert len(fam_3["unique_files"]) == 3
    assert fam_3["member_count"] == 3
    assert 0.93 <= fam_3["avg_similarity"] <= 0.94
    assert fam_3["max_similarity"] == 0.95
    assert fam_3["total_lines"] == 33

    fam_2 = next(f for f in families if f["member_count"] == 2)
    assert fam_2["member_count"] == 2
    assert fam_2["avg_similarity"] == 0.88


def test_generate_clone_diff_and_refactoring_suggestions(tmp_path: Path) -> None:
    """Test generation of unified clone diffs and synthesized refactoring suggestions."""
    file_a = tmp_path / "service_a.py"
    file_b = tmp_path / "service_b.py"

    file_a.write_text(
        "def process_order_v1(order_id):\n"
        "    tax = 0.05\n"
        "    total = calculate_total(order_id)\n"
        "    return total * (1.0 + tax)\n",
        encoding="utf-8",
    )

    file_b.write_text(
        "def process_order_v2(order_id):\n"
        "    tax = 0.08\n"
        "    total = calculate_total(order_id)\n"
        "    return total * (1.0 + tax)\n",
        encoding="utf-8",
    )

    u1 = {"file": str(file_a), "start": 1, "end": 4, "name": "process_order_v1"}
    u2 = {"file": str(file_b), "start": 1, "end": 4, "name": "process_order_v2"}

    lines1 = extract_unit_source_code(u1)
    assert len(lines1) == 4
    assert "process_order_v1" in lines1[0]

    missing_lines = extract_unit_source_code({"file": "non_existent.py", "start": 1, "end": 5, "name": "mock"})
    assert "Source for mock" in missing_lines[0]

    diff = generate_clone_diff(u1, u2)
    assert "---" in diff and "+++" in diff
    assert "-    tax = 0.05" in diff
    assert "+    tax = 0.08" in diff

    suggestion = synthesize_refactoring_suggestion(u1, u2)
    assert "[REFACTOR SUGGESTION]" in suggestion
    assert "_shared_process_order" in suggestion

    u_t1 = {"file": str(file_a), "start": 1, "end": 4, "name": "test_service_alpha"}
    u_t2 = {"file": str(file_b), "start": 1, "end": 4, "name": "test_service_beta"}
    test_suggestion = synthesize_refactoring_suggestion(u_t1, u_t2)
    assert "@pytest.mark.parametrize" in test_suggestion


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


def test_inline_comment_suppression(tmp_path: Path) -> None:
    """Test that inline comments (# pydoppelgangerhunt: ignore / # noqa: clone) exempt code units."""
    test_dir = tmp_path / "inline_suppression"
    test_dir.mkdir()
    f1 = test_dir / "calc_a.py"
    f2 = test_dir / "calc_b.py"

    f1.write_text(
        "def compute_interest_alpha(principal, rate, time):\n"
        "    interest = principal * rate * time\n"
        "    total = principal + interest\n"
        "    return total\n",
        encoding="utf-8",
    )

    f2.write_text(
        "def compute_interest_beta(principal, rate, time):  # pydoppelgangerhunt: ignore\n"
        "    interest = principal * rate * time\n"
        "    total = principal + interest\n"
        "    return total\n",
        encoding="utf-8",
    )

    clones = scan_target(str(test_dir), min_lines=3, min_tokens=8, threshold=0.85)
    assert len(clones) == 0

    f2.write_text(
        "def compute_interest_beta(principal, rate, time):  # noqa: clone\n"
        "    interest = principal * rate * time\n"
        "    total = principal + interest\n"
        "    return total\n",
        encoding="utf-8",
    )
    clones_noqa = scan_target(str(test_dir), min_lines=3, min_tokens=8, threshold=0.85)
    assert len(clones_noqa) == 0

    lines = ["def foo():  # pydoppelgangerhunt: ignore", "    pass"]
    assert check_inline_suppression(lines, 1, 2) is True
    lines_clean = ["def foo():", "    pass"]
    assert check_inline_suppression(lines_clean, 1, 2) is False


def test_baseline_record_load_and_filter(tmp_path: Path) -> None:
    """Test recording clone pairs to a baseline JSON file and filtering grandfathered hits."""
    baseline_file = tmp_path / "clones_baseline.json"
    u1 = {"file": "service/alpha.py", "start": 10, "end": 20, "name": "handler_a"}
    u2 = {"file": "service/beta.py", "start": 15, "end": 25, "name": "handler_b"}
    u3 = {"file": "service/gamma.py", "start": 30, "end": 40, "name": "handler_c"}
    u4 = {"file": "service/delta.py", "start": 35, "end": 45, "name": "handler_d"}

    mock_clones = [
        (0.95, u1, u2),
        (0.91, u3, u4),
    ]

    saved_path = record_baseline(mock_clones, str(baseline_file), target="service", threshold=0.90)
    assert Path(saved_path).exists()

    loaded_fps = load_baseline(str(baseline_file))
    assert len(loaded_fps) == 2
    fp1 = clone_pair_fingerprint(u1, u2)
    assert fp1 in loaded_fps

    new_clones, suppressed = filter_clones_by_baseline(mock_clones, loaded_fps)
    assert len(new_clones) == 0
    assert suppressed == 2

    u5 = {"file": "service/new.py", "start": 5, "end": 15, "name": "new_func"}
    mock_clones_with_new = mock_clones + [(0.98, u1, u5)]
    new_clones_2, suppressed_2 = filter_clones_by_baseline(mock_clones_with_new, loaded_fps)
    assert len(new_clones_2) == 1
    assert suppressed_2 == 2
    assert new_clones_2[0][2]["name"] == "new_func"


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


def test_multi_core_parallel_scan(tmp_path: Path) -> None:
    """Test multi-core parallel AST unit harvesting with ProcessPoolExecutor."""
    scan_dir = tmp_path / "parallel_scan"
    scan_dir.mkdir()

    for i in range(12):
        file_path = scan_dir / f"mod_{i}.py"
        file_path.write_text(
            f"def common_computation_{i}(x, y):\n"
            f"    factor = 2.5\n"
            f"    res = (x + y) * factor\n"
            f"    return res if res > 0 else 0.0\n",
            encoding="utf-8",
        )

    clones_parallel = scan_target(
        str(scan_dir),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        workers=2,
    )
    assert len(clones_parallel) >= 1

    clones_seq = scan_target(
        str(scan_dir),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        workers=1,
    )
    assert len(clones_parallel) == len(clones_seq)


def test_modular_pydoppelgangerhunt_exports() -> None:
    """Test that pydoppelgangerhunt package exports clean public API and metadata."""
    assert pydoppelgangerhunt.__version__ == "1.0.0"
    assert callable(pydoppelgangerhunt.main)
    assert callable(pydoppelgangerhunt.scan_target)
    assert callable(pydoppelgangerhunt.cluster_clone_families)
    assert callable(pydoppelgangerhunt.record_baseline)
    assert callable(pydoppelgangerhunt.load_baseline)


def test_cyclomatic_complexity_and_priority() -> None:
    """Test McCabe cyclomatic complexity calculation and priority scoring."""
    simple_code = "def f(x):\n    return x + 1\n"
    tree_simple = ast.parse(simple_code).body[0]
    assert compute_cyclomatic_complexity(tree_simple) == 1

    branching_code = (
        "def g(a, b):\n"
        "    if a > 0 and b > 0:\n"
        "        for i in range(10):\n"
        "            if i % 2 == 0:\n"
        "                print(i)\n"
        "    return a or b\n"
    )
    tree_branching = ast.parse(branching_code).body[0]
    assert compute_cyclomatic_complexity(tree_branching) == 6

    u1 = {"complexity": 6, "start": 1, "end": 20}
    u2 = {"complexity": 4, "start": 1, "end": 20}
    p_score = compute_priority_score(0.95, u1, u2)
    assert p_score == 114.0


def test_jupyter_notebook_harvesting_and_clones(tmp_path: Path) -> None:
    """Test parsing and clone detection on Jupyter notebook (.ipynb) files."""
    import json

    nb_data = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Title\n", "Some documentation\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "source": [
                    "def compute_score(a, b, c):\n",
                    "    factor = 1.25\n",
                    "    val = (a + b) * factor\n",
                    "    return val / (c + 0.001)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "source": [
                    "def calculate_metric(x, y, z):\n",
                    "    multiplier = 1.25\n",
                    "    res = (x + y) * multiplier\n",
                    "    return res / (z + 0.001)\n",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    nb_file = tmp_path / "analysis.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    units = harvest_notebook_units(str(nb_file), str(tmp_path), min_lines=3, min_tokens=5)
    assert len(units) >= 2
    assert any("#cell_2" in u["file"] for u in units)
    assert any("#cell_3" in u["file"] for u in units)

    clones = scan_target(
        str(tmp_path),
        min_lines=3,
        min_tokens=10,
        threshold=0.85,
        include_notebooks=True,
    )
    assert len(clones) >= 1
    assert any("#cell_" in clones[0][1]["file"] for _ in [0])


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


def test_temporal_divergence_detection() -> None:
    """Test temporal divergence detection between commit dates."""
    u1 = {"file": "mod1.py", "start": 1, "end": 10}
    u2 = {"file": "mod2.py", "start": 1, "end": 10}

    now = 1700000000
    b1 = {"timestamp": now, "author": "Alice", "commit": "abc1234"}
    b2 = {"timestamp": now - (120 * 86400), "author": "Bob", "commit": "def5678"}

    import unittest.mock as mock

    with mock.patch("pydoppelgangerhunt.git_diff.get_git_blame_info", side_effect=[b1, b2]):
        res = check_temporal_divergence(u1, u2, max_divergence_days=90)
        assert res is not None
        assert res["divergence_days"] == 120.0
        assert res["newer"] == u1
        assert res["older"] == u2

    with mock.patch("pydoppelgangerhunt.git_diff.get_git_blame_info", side_effect=[b1, b1]):
        res_same = check_temporal_divergence(u1, u2, max_divergence_days=90)
        assert res_same is None


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


def test_fixer_synthesis_and_patch(tmp_path: Path) -> None:
    """Test refactoring helper synthesis and unified patch generation."""
    code_a = (
        "def process_alpha(x, y):\n"
        "    v = x + y\n"
        "    return v * 2\n"
    )
    code_b = (
        "def process_beta(x, y):\n"
        "    v = x + y\n"
        "    return v * 3\n"
    )
    file_a = tmp_path / "proc_a.py"
    file_b = tmp_path / "proc_b.py"
    file_a.write_text(code_a, encoding="utf-8")
    file_b.write_text(code_b, encoding="utf-8")

    u1 = {
        "file": str(file_a),
        "name": "process_alpha",
        "type": "function",
        "start": 1,
        "end": 3,
        "params": ["x", "y"],
        "lines": 3,
    }
    u2 = {
        "file": str(file_b),
        "name": "process_beta",
        "type": "function",
        "start": 1,
        "end": 3,
        "params": ["x", "y"],
        "lines": 3,
    }

    helper = synthesize_shared_helper_code(u1, u2)
    assert "def _shared_process_alpha" in helper
    assert "v = x + y" in helper

    patch = generate_refactoring_patch([(0.85, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch
    assert "def _shared_process_alpha" in patch


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


def test_coverage_missing_files_and_boilerplate_nodes() -> None:
    """Test read_coverage_data with missing files and is_boilerplate_node classifications."""
    assert read_coverage_data("nonexistent_path.coverage") == {}
    assert read_coverage_data("nonexistent_path.xml") == {}

    assert is_boilerplate_node(ast.Assert(test=ast.Constant(value=True))) is True
    assert is_boilerplate_node(ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[], keywords=[]))) is True
    assert is_boilerplate_node(ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=1))) is False
