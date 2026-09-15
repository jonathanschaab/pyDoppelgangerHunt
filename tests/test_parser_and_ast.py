"""Unit tests for parser and ast."""

from __future__ import annotations

import ast
from pathlib import Path

from pydoppelgangerhunt import (
    call_sequence_similarity,
    compute_cyclomatic_complexity,
    compute_priority_score,
    extract_call_sequence,
    get_ast_shingles,
    get_ast_tokens,
    harvest_file_units,
    harvest_notebook_units,
    is_boilerplate_node,
    jaccard_similarity,
    read_coverage_data,
    scan_target,
)

from pydoppelgangerhunt.parser import (
    _compute_expression_complexity,
    _harvest_complex_expressions,
    check_inline_suppression,
)

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



def test_coverage_missing_files_and_boilerplate_nodes() -> None:
    """Test read_coverage_data with missing files and is_boilerplate_node classifications."""
    assert read_coverage_data("nonexistent_path.coverage") == {}
    assert read_coverage_data("nonexistent_path.xml") == {}

    assert is_boilerplate_node(ast.Assert(test=ast.Constant(value=True))) is True
    assert is_boilerplate_node(ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[], keywords=[]))) is True
    assert is_boilerplate_node(ast.Assign(targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=1))) is False


def test_docstring_and_annotation_normalization() -> None:
    """Test decoupled docstring stripping and default annotation normalization."""
    code_a = (
        'def process_items(items: list[str]) -> int:\n'
        '    """Documentation for process items version A."""\n'
        '    total: int = len(items)\n'
        '    return total * 2\n'
    )
    code_b = (
        'def process_items(items):\n'
        '    total = len(items)\n'
        '    return total * 2\n'
    )
    tree_a = ast.parse(code_a)
    tree_b = ast.parse(code_b)

    tokens_a_default = get_ast_tokens(tree_a)
    tokens_b_default = get_ast_tokens(tree_b)
    # Default: both docstrings and annotations are stripped, yielding identical tokens
    assert tokens_a_default == tokens_b_default

    # With docstrings preserved, code_a has docstring tokens whereas code_b does not
    tokens_a_doc = get_ast_tokens(tree_a, strip_docstrings=False)
    tokens_b_doc = get_ast_tokens(tree_b, strip_docstrings=False)
    assert tokens_a_doc != tokens_a_default
    assert tokens_a_doc != tokens_b_doc

    # With annotations preserved, they differ due to annotations
    tokens_a_ann = get_ast_tokens(tree_a, strip_annotations=False)
    tokens_b_ann = get_ast_tokens(tree_b, strip_annotations=False)
    assert tokens_a_ann != tokens_b_ann


def test_harvest_units_structural_hash(tmp_path: Path) -> None:
    """Test deterministic structural content hash generation during AST harvesting."""
    src_file = tmp_path / "sample.py"
    src_file.write_text(
        "def compute_score(val: int) -> float:\n"
        "    scaled = val * 1.5\n"
        "    offset = scaled + 10.0\n"
        "    return offset / 2.0\n",
        encoding="utf-8",
    )
    units = harvest_file_units(str(src_file), str(tmp_path), min_lines=2, min_tokens=5)
    assert len(units) >= 1
    u = units[0]
    assert "structural_hash" in u
    assert isinstance(u["structural_hash"], str)
    assert len(u["structural_hash"]) == 16

    # Re-harvest to confirm hash determinism
    units2 = harvest_file_units(str(src_file), str(tmp_path), min_lines=2, min_tokens=5)
    assert units2[0]["structural_hash"] == u["structural_hash"]


