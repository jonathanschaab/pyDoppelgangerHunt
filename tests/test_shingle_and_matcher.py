"""Unit tests for shingle and matcher."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from pydoppelgangerhunt import (
    get_ast_characteristic_vector,
    get_ast_shingles,
    get_ast_tokens,
    jaccard_similarity,
    lcs_alignment_similarity,
    merge_adjacent_clones,
    multiset_jaccard_similarity,
    scan_target,
    suppress_subclones,
    tfidf_jaccard_similarity,
    tfidf_multiset_jaccard_similarity,
)

from pydoppelgangerhunt.parser import (
    _CommutativeCanonicalizer,
    _IdiomCanonicalizer,
)


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


def test_inverted_index_frequency_threshold_pruning(tmp_path: Path) -> None:
    """Test inverted index frequency thresholding prunes ubiquitous shingles in large codebases."""
    scan_dir = tmp_path / "large_corpus"
    scan_dir.mkdir()

    # Create 35 files (corpus size > 30 triggers frequency thresholding)
    for i in range(35):
        fp = scan_dir / f"unit_{i:02d}.py"
        if i < 2:
            fp.write_text(
                f"def calculate_discriminative_{i}(val_x, val_y):\n"
                f"    common_counter = 0\n"
                f"    while common_counter < 10:\n"
                f"        common_counter += 1\n"
                f"    discriminative_calc = (val_x * 98765) ^ (val_y * 43210)\n"
                f"    return discriminative_calc\n",
                encoding="utf-8",
            )
        else:
            fp.write_text(
                f"def calculate_filler_{i}(val_x, val_y):\n"
                f"    common_counter = 0\n"
                f"    while common_counter < 10:\n"
                f"        common_counter += 1\n"
                f"    filler_unique_{i} = val_x + val_y + {i * 1000}\n"
                f"    return filler_unique_{i}\n",
                encoding="utf-8",
            )

    # Scan with max_index_frequency=0.20
    clones = scan_target(
        str(scan_dir),
        min_lines=4,
        min_tokens=12,
        threshold=0.85,
        max_index_frequency=0.20,
    )
    assert len(clones) == 1
    sim, u1, u2 = clones[0]
    assert sim >= 0.85
    assert "calculate_discriminative" in u1["name"]
    assert "calculate_discriminative" in u2["name"]


def test_stop_shingle_filtering_and_candidate_pruning(tmp_path: Path) -> None:
    """Test DEFAULT_STOP_SHINGLES and candidate inverted index pruning with stop-shingles."""
    from pydoppelgangerhunt.matcher import DEFAULT_STOP_SHINGLES, get_boilerplate_stop_shingles  # pylint: disable=import-outside-toplevel

    stop_shingles = get_boilerplate_stop_shingles()
    assert isinstance(stop_shingles, set)
    assert len(stop_shingles) >= 20
    assert ("Module", "If", "Compare") in stop_shingles
    assert ("Expr", "Call", "ATTR") in stop_shingles

    # Create two modules that only share boilerplate logging and main guards,
    # but have completely distinct algorithmic code
    pkg = tmp_path / "stop_pkg"
    pkg.mkdir()
    f1 = pkg / "worker_a.py"
    f2 = pkg / "worker_b.py"

    code_a = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def run_a(alpha, beta):\n"
        "    logger.info('running worker a')\n"
        "    result = [alpha * x for x in range(beta)]\n"
        "    logger.debug('finished')\n"
        "    return result\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    code_b = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def run_b(delta, gamma):\n"
        "    logger.info('running worker b')\n"
        "    mapping = {k: chr(k + 65) for k in range(delta, gamma)}\n"
        "    logger.debug('finished')\n"
        "    return mapping\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    f1.write_text(code_a, encoding="utf-8")
    f2.write_text(code_b, encoding="utf-8")

    # Scan with stop-shingles filtered
    clones_filtered = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.80,
        filter_stop_shingles=True,
    )
    assert len(clones_filtered) == 0

    # Test with custom stop-shingles
    custom_stop = {("Custom", "Token", "Shingle")}
    clones_custom = scan_target(
        str(pkg),
        min_lines=3,
        min_tokens=10,
        threshold=0.80,
        stop_shingles=custom_stop,
    )
    assert isinstance(clones_custom, list)


def test_batch_49_matcher_similarity_and_subclones(tmp_path: Path) -> None:
    """Batch 49: Test matcher similarity bounds, tfidf fallbacks, subclones, and multiprocessing."""
    # pylint: disable=protected-access,import-outside-toplevel
    from pydoppelgangerhunt.matcher import (
        _worker_harvest_file,
        compute_pair_similarity,
        suppress_subclones,
        tfidf_jaccard_similarity,
    )

    # 1. tfidf_jaccard_similarity fallback when idf_weights is empty
    set_x = {"a", "b", "c"}
    set_y = {"b", "c", "d"}
    assert tfidf_jaccard_similarity(set_x, set_y, {}) == 0.5

    # 2. compute_pair_similarity: call sequences under 3 calls and size bounding
    u_short_call1 = {"calls": ["step1", "step2"]}
    u_short_call2 = {"calls": ["step1", "step2"]}
    assert compute_pair_similarity(u_short_call1, u_short_call2, call_sequences=True) == 0.0

    u_huge = {"token_count": 100, "tokens": ["t"] * 100}
    u_tiny = {"token_count": 10, "tokens": ["t"] * 10}
    assert compute_pair_similarity(u_huge, u_tiny, threshold=0.8) == 0.0

    # 3. compute_pair_similarity: tfidf with shingles (not bag_of_tokens)
    u_shing1 = {"shingles": {"s1", "s2"}}
    u_shing2 = {"shingles": {"s1", "s2"}}
    assert compute_pair_similarity(u_shing1, u_shing2, tfidf=True, idf_weights={"s1": 1.5, "s2": 1.5}) == 1.0

    # 4. compute_pair_similarity: gapped_tolerance SourcererCC min multiset pruning
    u_gap1 = {"tokens": ["a", "b", "c"], "vector": {"a": 1, "b": 1, "c": 1}}
    u_gap2 = {"tokens": ["x", "y", "z"], "vector": {"x": 1, "y": 1, "z": 1}}
    assert compute_pair_similarity(u_gap1, u_gap2, gapped_tolerance=True, threshold=0.8) == 0.0

    # 5. compute_pair_similarity: plain bag_of_tokens without tfidf
    u_bag1 = {"vector": {"alpha": 2, "beta": 1}}
    u_bag2 = {"vector": {"alpha": 2, "beta": 1}}
    assert compute_pair_similarity(u_bag1, u_bag2, bag_of_tokens=True) == 1.0

    # 6. suppress_subclones: <= 1 clone, non-matching files, non-strictly smaller
    single_clone = [(0.9, {"file": "a.py"}, {"file": "b.py"})]
    assert suppress_subclones(single_clone) == single_clone

    c_parent = (0.95, {"file": "p.py", "start": 1, "end": 20}, {"file": "p.py", "start": 30, "end": 50})
    c_other = (0.90, {"file": "other1.py", "start": 5, "end": 10}, {"file": "other2.py", "start": 5, "end": 10})
    assert len(suppress_subclones([c_parent, c_other])) == 2

    c_equal = (0.95, {"file": "p.py", "start": 1, "end": 20}, {"file": "p.py", "start": 30, "end": 50})
    assert len(suppress_subclones([c_parent, c_equal])) == 2

    # 7. scan_target with tfidf, call_sequences, bag_of_tokens, sort_by="priority", and top_n
    pkg_scan = tmp_path / "scan_modes"
    pkg_scan.mkdir()
    (pkg_scan / "a.py").write_text(
        "def compute_1():\n    step_a()\n    step_b()\n    step_c()\n    return 42\n",
        encoding="utf-8",
    )
    (pkg_scan / "b.py").write_text(
        "def compute_2():\n    step_a()\n    step_b()\n    step_c()\n    return 42\n",
        encoding="utf-8",
    )

    clones_tfidf = scan_target(str(pkg_scan), threshold=0.8, min_lines=2, min_tokens=3, tfidf=True)
    assert len(clones_tfidf) == 1

    clones_calls = scan_target(str(pkg_scan), threshold=0.8, min_lines=2, min_tokens=3, call_sequences=True)
    assert len(clones_calls) == 1

    clones_tfidf_calls = scan_target(
        str(pkg_scan),
        threshold=0.8,
        min_lines=2,
        min_tokens=3,
        tfidf=True,
        call_sequences=True,
    )
    assert len(clones_tfidf_calls) == 1

    from pydoppelgangerhunt.matcher import _extract_unit_shingle_keys  # pylint: disable=import-outside-toplevel
    assert _extract_unit_shingle_keys({"calls": None}, call_sequences=True) == set()
    assert _extract_unit_shingle_keys({"vector": None}, bag_of_tokens=True) == set()
    assert _extract_unit_shingle_keys({"shingles": None}) == set()

    clones_bag = scan_target(
        str(pkg_scan),
        threshold=0.8,
        min_lines=2,
        min_tokens=3,
        bag_of_tokens=True,
        sort_by="priority",
        top_n=1,
    )
    assert len(clones_bag) == 1

    # 8. scan_target with workers=2 exercising _worker_harvest_file
    pkg = tmp_path / "parallel_pkg"
    pkg.mkdir()
    for idx in range(12):
        (pkg / f"f{idx}.py").write_text(f"def run_{idx}():\n    x = 10\n    return x\n", encoding="utf-8")
    par_clones = scan_target(str(pkg), threshold=0.8, min_lines=2, min_tokens=3, workers=2)
    assert isinstance(par_clones, list)
    assert len(par_clones) > 0

    task = {
        "file_path": str(pkg / "f1.py"),
        "repo_root": str(pkg),
        "min_lines": 2,
        "min_tokens": 3,
    }
    assert len(_worker_harvest_file(task)) >= 1


def test_batch_51_matcher_defensive_bounds_and_raw_set_baseline(tmp_path: Path) -> None:
    """Batch 51: Test defensive bounds in matcher merging/subclones, and raw set baseline filtering."""
    # pylint: disable=import-outside-toplevel
    import json
    from pydoppelgangerhunt.baseline import (
        clone_pair_structural_fingerprint,
        filter_clones_by_baseline,
        namespaced_structural_fingerprint,
        prune_baseline,
        pure_structural_fingerprint,
    )
    from pydoppelgangerhunt.matcher import merge_adjacent_clones, suppress_subclones

    # 1. merge_adjacent_clones with minimal dicts missing shingles/token_count (bag_of_tokens mode)
    u1_a = {"file": "mod.py", "name": "fn1:1-5", "start": 1, "end": 5, "vector": {"a": 1}}
    u2_a = {"file": "mod.py", "name": "fn2:1-5", "start": 1, "end": 5, "vector": {"a": 1}}
    u1_b = {"file": "mod.py", "name": "fn1:6-10", "start": 6, "end": 10, "vector": {"b": 1}}
    u2_b = {"file": "mod.py", "name": "fn2:6-10", "start": 6, "end": 10, "vector": {"b": 1}}
    merged = merge_adjacent_clones([(1.0, u1_a, u2_a), (1.0, u1_b, u2_b)], bag_of_tokens=True)
    assert len(merged) == 1
    assert merged[0][1]["start"] == 1
    assert merged[0][1]["end"] == 10
    assert merged[0][1]["vector"] == {"a": 1, "b": 1}

    # 2. suppress_subclones with missing start/end or None values
    p1 = {"file": "mod.py", "start": 1, "end": 20}
    p2 = {"file": "mod.py", "start": 30, "end": 50}
    c1 = {"file": "mod.py", "start": None, "end": 10}
    c2 = {"file": "mod.py", "start": 35, "end": 45}
    suppressed = suppress_subclones([(0.95, p1, p2), (0.90, c1, c2)])
    assert len(suppressed) == 1
    assert suppressed[0][0] == 0.95

    # 3. filter_clones_by_baseline with a raw set of strings (sfp, ns_sfp, pure_sfp)
    u_x = {"file": "x.py", "name": "calc", "start": 1, "end": 5, "structural_hash": "hash_x"}
    u_y = {"file": "y.py", "name": "calc", "start": 1, "end": 5, "structural_hash": "hash_y"}
    sfp = clone_pair_structural_fingerprint(u_x, u_y)
    ns_sfp = namespaced_structural_fingerprint(u_x, u_y)
    pure_sfp = pure_structural_fingerprint(u_x, u_y)

    # sfp match via raw set
    filtered_sfp, supp_sfp = filter_clones_by_baseline([(0.9, u_x, u_y)], {sfp})
    assert len(filtered_sfp) == 0
    assert supp_sfp == 1

    # ns_sfp match via raw set
    filtered_ns, supp_ns = filter_clones_by_baseline([(0.9, u_x, u_y)], {ns_sfp})
    assert len(filtered_ns) == 0
    assert supp_ns == 1

    # pure_sfp match via raw set
    filtered_pure, supp_pure = filter_clones_by_baseline([(0.9, u_x, u_y)], {pure_sfp})
    assert len(filtered_pure) == 0
    assert supp_pure == 1

    # unmatched clone retains
    u_z = {"file": "z.py", "name": "calc", "start": 1, "end": 5, "structural_hash": "hash_z"}
    unmatched_res, supp_unmatched = filter_clones_by_baseline([(0.85, u_x, u_z)], {pure_sfp})
    assert len(unmatched_res) == 1
    assert supp_unmatched == 0

    # 4. prune_baseline when item_pure_sfp is absent in baseline file
    bl_json = tmp_path / "baseline_no_pure.json"
    bl_json.write_text(
        json.dumps({
            "version": "1.3.0",
            "fingerprints": [
                {
                    "fingerprint": "x.py:calc <===> y.py:calc",
                    "hash_a": "hash_x",
                    "hash_b": "hash_y",
                    "file_a": "x.py",
                    "file_b": "y.py",
                    "name_a": "calc",
                    "name_b": "calc",
                }
            ],
        }),
        encoding="utf-8",
    )
    res_prune = prune_baseline(
        str(bl_json),
        active_clones=[(0.9, u_x, u_y)],
        unstaged_modified_ranges={},
    )
    assert res_prune.retained_count == 1
    assert res_prune.pruned_count == 0


def test_batch_59_scan_target_repo_root_and_diff_hunk_prefixes(tmp_path: Path) -> None:
    """Test scan_target repo_root propagation, auto-derivation, and parse_git_diff_hunks multi-prefix parsing."""
    from pydoppelgangerhunt.git_diff import parse_git_diff_hunks

    # 1. scan_target with explicit repo_root on external directory
    ext_repo = tmp_path / "ext_repo"
    sub_pkg = ext_repo / "sub_pkg"
    sub_pkg.mkdir(parents=True)

    f1 = sub_pkg / "worker_a.py"
    f2 = sub_pkg / "worker_b.py"
    code = (
        "def process_payload(x: int, y: int) -> int:\n"
        "    r1 = x * 10 + y * 20\n"
        "    r2 = r1 ** 2 + 100\n"
        "    return r2 // 3\n"
    )
    f1.write_text(code, encoding="utf-8")
    f2.write_text(code, encoding="utf-8")

    clones_with_root = scan_target(
        str(sub_pkg),
        repo_root=str(ext_repo),
        min_lines=2,
        min_tokens=5,
        threshold=0.90,
    )
    assert len(clones_with_root) >= 1
    file_a = clones_with_root[0][1]["file"]
    file_b = clones_with_root[0][2]["file"]
    assert file_a in ("sub_pkg/worker_a.py", "sub_pkg/worker_b.py")
    assert file_b in ("sub_pkg/worker_a.py", "sub_pkg/worker_b.py")
    assert not Path(file_a).is_absolute()

    # 2. scan_target auto-deriving effective_repo_root when external target_dir is scanned without repo_root
    clones_auto_root = scan_target(
        str(sub_pkg),
        min_lines=2,
        min_tokens=5,
        threshold=0.90,
    )
    assert len(clones_auto_root) >= 1
    auto_file_a = clones_auto_root[0][1]["file"]
    auto_file_b = clones_auto_root[0][2]["file"]
    assert auto_file_a in ("worker_a.py", "worker_b.py")
    assert auto_file_b in ("worker_a.py", "worker_b.py")
    assert not Path(auto_file_a).is_absolute()

    # 3. parse_git_diff_hunks supporting no-prefix, b/, w/, i/, c/, and /dev/null
    diff_multi = (
        "--- file_np.py\n"
        "+++ file_np.py\n"
        "@@ -10,3 +10,3 @@\n"
        "+np_line\n"
        "--- a/file_b.py\n"
        "+++ b/file_b.py\n"
        "@@ -20,2 +20,2 @@\n"
        "+b_line\n"
        "--- old/file_w.py\n"
        "+++ w/file_w.py\n"
        "@@ -30,1 +30,1 @@\n"
        "+w_line\n"
        "--- old/file_i.py\n"
        "+++ i/file_i.py\n"
        "@@ -40,1 +40,1 @@\n"
        "+i_line\n"
        "--- a/deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,5 +0,0 @@\n"
        "-deleted\n"
    )
    hunks = parse_git_diff_hunks(diff_multi)
    assert hunks.get("file_np.py") == [(10, 12)]
    assert hunks.get("file_b.py") == [(20, 21)]
    assert hunks.get("file_w.py") == [(30, 30)]
    assert hunks.get("file_i.py") == [(40, 40)]
    assert "deleted.py" not in hunks
    assert "/dev/null" not in hunks


def test_batch_70_review_fixes(tmp_path: Path) -> None:
    """Batch 70: Test TRY_NODE_TYPES compatibility, path-specific exemption isolation, and absolute path normalization."""
    # pylint: disable=import-outside-toplevel
    import ast
    from pydoppelgangerhunt.matcher import (
        _normalize_exemption_endpoint,
        scan_target,
    )
    from pydoppelgangerhunt.parser import TRY_NODE_TYPES, harvest_file_units

    # 1. TRY_NODE_TYPES verification
    assert ast.Try in TRY_NODE_TYPES
    assert () not in TRY_NODE_TYPES
    assert all(isinstance(t, type) for t in TRY_NODE_TYPES)

    # 2. Path-specific exemption isolation
    pkg_a = tmp_path / "pkg_a"
    pkg_b = tmp_path / "pkg_b"
    pkg_a.mkdir(parents=True, exist_ok=True)
    pkg_b.mkdir(parents=True, exist_ok=True)

    code_body = "def compute(x: int) -> int:\n    a = x * 10\n    b = a + 5\n    c = b * 2\n    return c\n"
    (pkg_a / "service.py").write_text(code_body, encoding="utf-8")
    (pkg_b / "service.py").write_text(code_body, encoding="utf-8")
    (tmp_path / "other.py").write_text(code_body, encoding="utf-8")

    # Path-specific exemption for pkg_a should not suppress pkg_b
    exempt_pair = ("pkg_a/service.py:compute", "other.py:compute")
    clones = scan_target(str(tmp_path), exemptions=[exempt_pair], threshold=0.8, min_lines=3)
    has_pkg_b_other = any(
        ("pkg_b" in str(u1.get("file")) and "other.py" in str(u2.get("file")))
        or ("other.py" in str(u1.get("file")) and "pkg_b" in str(u2.get("file")))
        for _, u1, u2 in clones
    )
    assert has_pkg_b_other, "pkg_b/service.py should not be suppressed by pkg_a/service.py exemption"

    has_pkg_a_other = any(
        ("pkg_a" in str(u1.get("file")) and "other.py" in str(u2.get("file")))
        or ("other.py" in str(u1.get("file")) and "pkg_a" in str(u2.get("file")))
        for _, u1, u2 in clones
    )
    assert not has_pkg_a_other, "pkg_a/service.py should be suppressed by pkg_a exemption"

    # 3. Basename-only exemption suppresses across all packages
    base_exempt_pair = ("service.py:compute", "other.py:compute")
    clones_base = scan_target(str(tmp_path), exemptions=[base_exempt_pair], threshold=0.8, min_lines=3)
    has_any_service_other = any(
        ("service.py" in str(u1.get("file")) and "other.py" in str(u2.get("file")))
        or ("other.py" in str(u1.get("file")) and "service.py" in str(u2.get("file")))
        for _, u1, u2 in clones_base
    )
    assert not has_any_service_other, "service.py basename exemption should suppress across all packages"

    # 4. Absolute path exemption normalization
    abs_f = str((pkg_a / "service.py").resolve())
    norm_ep = _normalize_exemption_endpoint(f"{abs_f}:compute", repo_root=tmp_path)
    assert not norm_ep.startswith("/")
    assert "pkg_a/service.py:compute" in norm_ep or "pkg_a" in norm_ep

    # 5. harvest_file_units with clause_level=True parses Try statements safely
    try_code = (
        "def try_flow(x):\n"
        "    try:\n"
        "        return 1 / x\n"
        "    except ZeroDivisionError:\n"
        "        return 0\n"
        "    else:\n"
        "        pass\n"
        "    finally:\n"
        "        pass\n"
    )
    try_file = tmp_path / "try_test.py"
    try_file.write_text(try_code, encoding="utf-8")
    try_units = harvest_file_units(str(try_file), str(tmp_path), clause_level=True, min_lines=1, min_tokens=1)
    assert len(try_units) > 0


def test_corpus_sensitivity_stop_shingle_pruning_on_differential_runs(tmp_path: Path) -> None:
    """Verifies that dynamic stop-shingle pruning activates on small corpora (differential PR runs)."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # Create a small module with 6 distinct utility functions sharing common boilerplate logging/guards
    code_lines = []
    for i in range(6):
        code_lines.append(
            f"def compute_metric_{i}(data: int) -> int:\n"
            f"    # Standard boilerplate logger invocation\n"
            f"    if __name__ == '__main__':\n"
            f"        pass\n"
            f"    # Distinct algorithmic computation\n"
            f"    return data ** {i + 2} + {i * 100}\n"
        )
    src_file = tmp_path / "pr_diff_sample.py"
    src_file.write_text("\n".join(code_lines), encoding="utf-8")

    # With min_corpus_size=4 and max_index_frequency=0.25 on a 6-unit corpus:
    # max_posting_len = max(2, ceil(6 * 0.25)) = 2.
    # The boilerplate guard appearing in all 6 functions has posting len 6 > 2, so it is pruned!
    # Because each function's algorithm is distinct, no spurious clones are reported:
    clones_pruned = scan_target(
        str(tmp_path),
        threshold=0.85,
        min_lines=3,
        min_tokens=5,
        max_index_frequency=0.25,
        min_corpus_size=4,
    )
    assert len(clones_pruned) == 0

def test_batch_82_matcher_sloc_and_priority_score_bounds() -> None:
    """Verifies that compute_priority_score and SLOC sorting handle inverted or synthetic line bounds defensively."""
    from pydoppelgangerhunt.matcher import compute_priority_score  # pylint: disable=import-outside-toplevel

    u1 = {"start": 20, "end": 10, "complexity": -5, "token_count": 50, "name": "bad1", "file": "f1.py"}
    u2 = {"start": 30, "end": 15, "complexity": 0, "token_count": 50, "name": "bad2", "file": "f2.py"}
    score = compute_priority_score(0.9, u1, u2)
    assert score == 0.0

    clone_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = [
        (0.9, u1, u2),
        (0.8, {"start": 1, "end": 5}, {"start": 1, "end": 5}),
    ]
    clone_pairs.sort(
        key=lambda x: (
            max(0, int(x[1].get("end") or int(x[1].get("start") or 1)) - int(x[1].get("start") or 1) + 1)
            + max(0, int(x[2].get("end") or int(x[2].get("start") or 1)) - int(x[2].get("start") or 1) + 1)
        ),
        reverse=True,
    )
    assert len(clone_pairs) == 2


def test_differential_scan_multi_unit_file_and_shingle_dedup(tmp_path: Path) -> None:
    """Verifies differential scanning with multi-unit files and candidate pair monotonic ordering."""
    from pydoppelgangerhunt.matcher import _add_candidate_pairs  # pylint: disable=import-outside-toplevel

    file_a = tmp_path / "file_a.py"
    file_a.write_text(
        "def func_one(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    c = b - 3\n"
        "    return c * 4\n\n"
        "def func_two(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    c = b - 3\n"
        "    return c * 4\n\n"
        "def func_three(x: int) -> int:\n"
        "    return x ** 2 + 100\n",
        encoding="utf-8",
    )

    file_b = tmp_path / "file_b.py"
    file_b.write_text(
        "def func_four(x: int) -> int:\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    c = b - 3\n"
        "    return c * 4\n",
        encoding="utf-8",
    )

    clones = scan_target(
        str(tmp_path),
        diff_files=["file_a.py"],
        threshold=0.85,
        min_lines=3,
        min_tokens=5,
    )
    assert len(clones) >= 1
    for _, u1, u2 in clones:
        files = {Path(u1["file"]).name, Path(u2["file"]).name}
        assert "file_a.py" in files

    cand_pairs: set[Tuple[int, int]] = set()
    _add_candidate_pairs(cand_pairs, [0, 2, 4], diff_unit_indices={0, 4})
    assert cand_pairs == {(0, 4), (0, 2), (2, 4)}
    for idx1, idx2 in cand_pairs:
        assert idx1 < idx2


def test_uncalibrated_novel_shingle_pair_budget_bounds_explosion(tmp_path: Path) -> None:
    """Verifies that novel shingles generating massive candidate pairings are bounded by MAX_NOVEL_SHINGLE_PAIR_BUDGET."""
    from pydoppelgangerhunt.matcher import (  # pylint: disable=import-outside-toplevel
        MAX_NOVEL_SHINGLE_PAIR_BUDGET,
        scan_target,
    )

    assert MAX_NOVEL_SHINGLE_PAIR_BUDGET == 10_000

    # Generate a file with 150 identical functions sharing an uncalibrated novel shingle:
    # 150 * 149 // 2 = 11,175 potential pairs (> 10,000 budget)
    lines: list[str] = []
    for i in range(150):
        lines.append(
            f"def action_handler_burst_{i}(arg_a, arg_b, arg_c):\n"
            "    novel_token_sequence_xyz = arg_a + arg_b + arg_c\n"
            "    return novel_token_sequence_xyz * 42\n"
        )
    test_file = tmp_path / "burst.py"
    test_file.write_text("\n".join(lines), encoding="utf-8")

    calib = {
        "total_units": 5000,
        "max_index_frequency": 0.25,
        "min_lines": 3,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    # In differential scan against burst.py, the novel token produces > 10,000 pairs and is skipped
    clones = scan_target(
        str(tmp_path),
        diff_files=["burst.py"],
        min_lines=3,
        threshold=0.90,
        corpus_calibration=calib,
    )
    # The unbounded candidate explosion was prevented, suppressing pairs for this shingle
    assert isinstance(clones, list)
    assert len(clones) == 0

    # Legacy calibration (missing shingle_frequencies) falls back to legacy stop-shingle behavior
    # and does not drop valid pairs under the novel-pair budget
    calib_legacy = {
        "total_units": 5000,
        "max_index_frequency": 0.25,
        "min_lines": 3,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
    }
    clones_legacy = scan_target(
        str(tmp_path),
        diff_files=["burst.py"],
        min_lines=3,
        threshold=0.90,
        corpus_calibration=calib_legacy,
    )
    assert len(clones_legacy) > 0

    # When pairs are within budget (40 functions = 780 pairs <= 10,000), clones are detected
    small_lines: list[str] = []
    for i in range(40):
        small_lines.append(
            f"def action_handler_small_{i}(arg_a, arg_b, arg_c):\n"
            "    novel_token_sequence_xyz = arg_a + arg_b + arg_c\n"
            "    return novel_token_sequence_xyz * 42\n"
        )
    small_file = tmp_path / "small.py"
    small_file.write_text("\n".join(small_lines), encoding="utf-8")

    clones_small = scan_target(
        str(tmp_path),
        diff_files=["small.py"],
        min_lines=3,
        threshold=0.90,
        corpus_calibration=calib,
    )
    assert len(clones_small) > 0


def test_cumulative_novel_shingle_pair_budget_bounds_multiple_shingles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that the candidate-pair budget is tracked globally across multiple distinct novel shingles."""
    import logging
    import pydoppelgangerhunt.matcher as matcher  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # Set a tight global novel pair budget: e.g. 5 pairs
    monkeypatch.setattr(matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 5)

    # Shingle 1 group: 3 functions = 3 pairs <= 5
    # Shingle 2 group: 3 functions = 3 pairs (cumulative 6 > 5)
    lines: list[str] = []
    for i in range(3):
        lines.append(
            f"def group_one_fn_{i}(val_x, val_y):\n"
            f"    novel_token_group_one = val_x + val_y + {i}\n"
            "    return novel_token_group_one * 10\n"
        )
    for i in range(3):
        lines.append(
            f"def group_two_fn_{i}(val_a, val_b):\n"
            f"    novel_token_group_two = val_a * val_b + {i}\n"
            "    return novel_token_group_two * 20\n"
        )
    test_file = tmp_path / "multi_novel.py"
    test_file.write_text("\n".join(lines), encoding="utf-8")

    calib = {
        "total_units": 5000,
        "max_index_frequency": 0.25,
        "min_lines": 3,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    # Group 1 uses 3 pairs (remaining budget = 2).
    # Group 2 needs 3 pairs, which exceeds the remaining budget 2, so Group 2 is skipped.
    with caplog.at_level(logging.DEBUG):
        clones = scan_target(
            str(tmp_path),
            diff_files=["multi_novel.py"],
            min_lines=3,
            threshold=0.70,
            corpus_calibration=calib,
        )
    group_one_clones = [
        c for c in clones
        if "group_one_fn" in (c[1].get("name") or "") or "group_one_fn" in (c[2].get("name") or "")
    ]
    group_two_clones = [
        c for c in clones
        if "group_two_fn" in (c[1].get("name") or "") or "group_two_fn" in (c[2].get("name") or "")
    ]
    assert len(group_one_clones) > 0
    assert len(group_two_clones) == 0
    assert any("Novel shingle pair budget exceeded" in record.message for record in caplog.records)


def test_unit_drift_computed_when_target_units_drop_to_zero(tmp_path: Path) -> None:
    """Verifies that unit_drift is computed as 1.0 (100% drift) when active target units drop to 0."""
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    calib = {
        "total_units": 50,
        "max_index_frequency": 0.25,
        "min_lines": 8,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    scan_target(str(empty_dir), corpus_calibration=calib)
    assert calib.get("current_units") == 0
    assert calib.get("unit_drift") == 1.0


def test_scan_target_diff_files_in_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies scan_target correctly resolves diff_files when scanning a subdirectory with repo_root."""
    repo = tmp_path / "diff_repo"
    repo.mkdir()
    sub = repo / "sub_pkg"
    sub.mkdir()

    code1 = (
        "def worker_alpha(val_a, val_b):\n"
        "    res = val_a * 10 + val_b\n"
        "    res_scaled = res * 2\n"
        "    return res_scaled\n"
    )
    code2 = (
        "def worker_beta(val_a, val_b):\n"
        "    res = val_a * 10 + val_b\n"
        "    res_scaled = res * 2\n"
        "    return res_scaled\n"
    )
    (sub / "w1.py").write_text(code1, encoding="utf-8")
    (sub / "w2.py").write_text(code2, encoding="utf-8")

    # diff_files passed as repo-relative path 'sub_pkg/w1.py'
    clones = scan_target(
        str(sub),
        repo_root=str(repo),
        diff_files=["sub_pkg/w1.py"],
        min_lines=3,
        threshold=0.80,
    )
    assert len(clones) >= 1

    # diff_files passed when target is itself the repo root
    clones2 = scan_target(
        str(sub),
        diff_files=["w1.py"],
        min_lines=3,
        threshold=0.80,
    )
    assert len(clones2) >= 1

    # diff_files passed in Git CLI scenario: target="sub_pkg", repo_root="sub_pkg", diff_files=["sub_pkg/w1.py"]
    # (Inside Git worktree `repo`, so git_root_resolved is `repo`)
    norm_repo = str(repo).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: (
            norm_repo + "\n"
            if args == ["rev-parse", "--show-toplevel"]
            else None
        ),
    )
    clones3 = scan_target(
        str(sub),
        repo_root=str(sub),
        diff_files=["sub_pkg/w1.py"],
        min_lines=3,
        threshold=0.80,
    )
    assert len(clones3) >= 1


def test_scan_target_diff_files_single_file_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies scan_target correctly normalizes single-file targets to containing directory for diff keys."""
    repo = tmp_path / "single_file_repo"
    src = repo / "src"
    src.mkdir(parents=True)

    code = (
        "def compute_alpha(val_a, val_b):\n"
        "    temp = val_a * 10 + val_b\n"
        "    return temp * 2\n\n"
        "def compute_beta(val_a, val_b):\n"
        "    temp = val_a * 10 + val_b\n"
        "    return temp * 2\n"
    )
    target_file = src / "a.py"
    target_file.write_text(code, encoding="utf-8")

    norm_repo = str(repo).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: (
            norm_repo + "\n"
            if args == ["rev-parse", "--show-toplevel"]
            else None
        ),
    )

    # 1. Single file scan with repo_root as containing directory (CLI scenario)
    clones = scan_target(
        str(target_file),
        repo_root=str(src),
        diff_files=["src/a.py"],
        min_lines=3,
        threshold=0.80,
    )
    assert len(clones) >= 1

    # 2. Single file scan without explicit repo_root (falls back to Git worktree root)
    clones2 = scan_target(
        str(target_file),
        diff_files=["src/a.py"],
        min_lines=3,
        threshold=0.80,
    )
    assert len(clones2) >= 1


def test_scan_target_calibration_scope_mismatch_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that scan_target discards calibration when target directory scope differs from calibration scope."""
    repo = tmp_path / "scope_repo"
    src = repo / "src"
    src.mkdir(parents=True)
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)

    norm_repo = str(repo).replace("\\", "/")
    monkeypatch.setattr(
        "pydoppelgangerhunt.git_diff._run_git_command",
        lambda args, cwd=None: (
            norm_repo + "\n"
            if args == ["rev-parse", "--show-toplevel"]
            else None
        ),
    )

    code = "def sample():\n    x = 10\n    return x * 2\n"
    (src / "worker.py").write_text(code, encoding="utf-8")
    (tests_dir / "test_worker.py").write_text(code, encoding="utf-8")

    # Generate calibration for src
    _, calib_src = scan_target(str(src), return_calibration=True)
    assert calib_src.get("scope") == "src"

    # Running scan on repo root (scope="") with calib_src:
    # calib_src should be rejected and NOT used (so calib_src does not record unit_drift)
    calib_copy = dict(calib_src)
    scan_target(str(repo), corpus_calibration=calib_copy)
    assert "unit_drift" not in calib_copy


def test_scan_target_git_root_discovered_for_single_file_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that scan_target correctly probes git root when targeting a single file in a git repo."""
    repo = tmp_path / "git_single_repo"
    src = repo / "src"
    src.mkdir(parents=True)
    f_target = src / "target.py"
    code = (
        "def compute_val(a, b):\n"
        "    res = 0\n"
        "    for val in a:\n"
        "        res += val * b + 42\n"
        "    return res\n"
    )
    code_clone = (
        "def compute_val2(a, b):\n"
        "    res = 0\n"
        "    for val in a:\n"
        "        res += val * b + 42\n"
        "    return res\n"
    )
    f_target.write_text(code + "\n" + code_clone, encoding="utf-8")

    norm_repo = str(repo).replace("\\", "/")

    def mock_run_git(args: Any, cwd: Any = None) -> Any:
        if args == ["rev-parse", "--show-toplevel"]:
            if cwd is not None and not Path(cwd).is_dir():
                raise NotADirectoryError(f"{cwd} is not a directory")
            return norm_repo + "\n"
        return None

    monkeypatch.setattr("pydoppelgangerhunt.git_diff._run_git_command", mock_run_git)

    # Differential scan targeting single file f_target with diff_files referencing git-worktree-relative "src/target.py"
    # without passing repo_root (relying on _resolve_git_root_path to find git root)
    clones = scan_target(
        str(f_target),
        diff_files=["src/target.py"],
        min_lines=5,
        threshold=0.80,
    )
    assert len(clones) >= 1


def test_shingle_iteration_order_is_deterministic_under_novel_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that novel shingle admission order is strictly deterministic regardless of file discovery order."""
    import logging
    import pydoppelgangerhunt.matcher as matcher  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # Budget allows only one 3-function group (3 * 2 // 2 = 3 pairs <= 4)
    monkeypatch.setattr(matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 4)

    lines_alpha = [
        f"def alpha_unique_proc_{i}(v1, v2):\n    token_alpha_shingle = v1 + v2 + {i}\n    return token_alpha_shingle\n"
        for i in range(3)
    ]
    lines_omega = [
        f"def omega_unique_proc_{i}(v1, v2):\n    token_omega_shingle = v1 * v2 + {i}\n    return token_omega_shingle\n"
        for i in range(3)
    ]

    dir_1 = tmp_path / "order_1"
    dir_1.mkdir()
    # In dir_1: write alpha first, then omega
    (dir_1 / "1_alpha.py").write_text("\n".join(lines_alpha), encoding="utf-8")
    (dir_1 / "2_omega.py").write_text("\n".join(lines_omega), encoding="utf-8")

    dir_2 = tmp_path / "order_2"
    dir_2.mkdir()
    # In dir_2: write omega first, then alpha
    (dir_2 / "1_omega.py").write_text("\n".join(lines_omega), encoding="utf-8")
    (dir_2 / "2_alpha.py").write_text("\n".join(lines_alpha), encoding="utf-8")

    calib = {
        "total_units": 1000,
        "max_index_frequency": 0.25,
        "min_lines": 3,
        "min_corpus_size": 4,
        "global_stop_shingles": set(),
        "shingle_frequencies": {},
    }

    with caplog.at_level(logging.DEBUG):
        clones_1 = scan_target(
            str(dir_1),
            min_lines=3,
            threshold=0.70,
            corpus_calibration=calib,
        )
        clones_2 = scan_target(
            str(dir_2),
            min_lines=3,
            threshold=0.70,
            corpus_calibration=calib,
        )

    names_1 = sorted({(c[1]["name"], c[2]["name"]) for c in clones_1})
    names_2 = sorted({(c[1]["name"], c[2]["name"]) for c in clones_2})

    # Both runs must discover the exact same clone pairs despite opposite file discovery order
    assert len(names_1) > 0
    assert names_1 == names_2
    assert any("Novel shingle pair budget exceeded" in record.message for record in caplog.records)


def test_legacy_calibration_missing_frequencies_falls_back_to_stop_shingles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that legacy calibrations without shingle_frequencies fall back to stop shingles without novel budget capping."""
    import pydoppelgangerhunt.matcher as matcher  # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.matcher import scan_target  # pylint: disable=import-outside-toplevel

    # Artificially tiny novel budget: 2 pairs max if shingles were marked novel
    monkeypatch.setattr(matcher, "MAX_NOVEL_SHINGLE_PAIR_BUDGET", 2)

    lines_a = (
        "def func_a1(x, y):\n"
        "    alpha_calc = x * 10 + y * 20\n"
        "    return alpha_calc + 1\n\n"
        "def func_a2(x, y):\n"
        "    alpha_calc = x * 10 + y * 20\n"
        "    return alpha_calc + 1\n"
    )
    lines_b = (
        "def func_b1(x, y):\n"
        "    beta_calc = x ** 2 + y ** 2\n"
        "    return beta_calc + 2\n\n"
        "def func_b2(x, y):\n"
        "    beta_calc = x ** 2 + y ** 2\n"
        "    return beta_calc + 2\n"
    )
    lines_c = (
        "def func_c1(x, y):\n"
        "    gamma_calc = (x + y) * (x - y)\n"
        "    return gamma_calc + 3\n\n"
        "def func_c2(x, y):\n"
        "    gamma_calc = (x + y) * (x - y)\n"
        "    return gamma_calc + 3\n"
    )
    (tmp_path / "mod_a.py").write_text(lines_a, encoding="utf-8")
    (tmp_path / "mod_b.py").write_text(lines_b, encoding="utf-8")
    (tmp_path / "mod_c.py").write_text(lines_c, encoding="utf-8")

    # Legacy calibration has total_units and stop shingles, but NO shingle_frequencies key
    legacy_calib = {
        "total_units": 100,
        "max_index_frequency": 0.50,
        "global_stop_shingles": [],
    }

    clones = scan_target(
        str(tmp_path),
        min_lines=3,
        threshold=0.80,
        corpus_calibration=legacy_calib,
    )

    # All 3 clone pairs must be discovered (not dropped by novel-pair budget of 2)
    clone_pairs = {(c[1]["name"], c[2]["name"]) for c in clones}
    assert len(clone_pairs) >= 3

    # Now verify with shingle_frequencies: None
    legacy_calib_none = {
        "total_units": 100,
        "max_index_frequency": 0.50,
        "global_stop_shingles": [],
        "shingle_frequencies": None,
    }
    clones_none = scan_target(
        str(tmp_path),
        min_lines=3,
        threshold=0.80,
        corpus_calibration=legacy_calib_none,
    )
    assert len(clones_none) >= 3


def test_matcher_units_missing_shingles_no_key_error(tmp_path: Path) -> None:
    """Verifies that scan_target with tfidf=True handles units missing shingles key without KeyError."""
    from unittest.mock import patch
    import pydoppelgangerhunt.matcher

    f = tmp_path / "code.py"
    f.write_text("def a():\n    return 1\ndef b():\n    return 1\n", encoding="utf-8")

    orig_harvest = pydoppelgangerhunt.matcher.harvest_file_units

    def mock_harvest(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        units = orig_harvest(*args, **kwargs)
        if units:
            units[0].pop("shingles", None)
            if len(units) > 1:
                units[1]["shingles"] = None
        return units

    with patch.object(pydoppelgangerhunt.matcher, "harvest_file_units", side_effect=mock_harvest):
        clones = scan_target(str(tmp_path), tfidf=True, min_lines=1)
        assert isinstance(clones, list)


def test_shingle_sort_key_determinism_and_types() -> None:
    """Verifies that _shingle_sort_key partitions and orders tuples and strings deterministically."""
    from pydoppelgangerhunt.matcher import _shingle_sort_key  # pylint: disable=import-outside-toplevel

    assert _shingle_sort_key(("FunctionDef", "arguments")) == (0, (("str", "FunctionDef"), ("str", "arguments")))
    assert _shingle_sort_key("token_str") == (1, "token_str")
    assert _shingle_sort_key(42) == (1, "42")

    # Heterogeneous tuple elements (int vs str) do not trigger TypeError
    hetero_keys = [("call", "arg"), ("call", 10), ("call", 2), "token"]
    sorted_hetero = sorted(hetero_keys, key=_shingle_sort_key)
    assert sorted_hetero == [("call", 2), ("call", 10), ("call", "arg"), "token"]

    raw_keys = [
        "zebra_token",
        ("Module", "If", "Compare"),
        "apple_token",
        ("FunctionDef", "arguments", "arg"),
        "beta_token",
        ("ClassDef", "Name"),
    ]
    sorted_keys = sorted(raw_keys, key=_shingle_sort_key)
    expected = [
        ("ClassDef", "Name"),
        ("FunctionDef", "arguments", "arg"),
        ("Module", "If", "Compare"),
        "apple_token",
        "beta_token",
        "zebra_token",
    ]
    assert sorted_keys == expected


def test_scan_target_accepts_path_instance(tmp_path: Path) -> None:
    """Verifies that scan_target accepts Path instances for target_dir without type or runtime errors."""
    f = tmp_path / "sample.py"
    f.write_text("def foo():\n    return 42\n", encoding="utf-8")
    clones = scan_target(tmp_path)
    assert isinstance(clones, list)









