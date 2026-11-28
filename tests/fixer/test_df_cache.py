"""Unit tests for run-scoped AnalysisSession and immutable source-hash caches in _df_cache."""

from __future__ import annotations

import ast
from collections import OrderedDict
import concurrent.futures
import contextvars
import gc
import threading
from typing import Any
import weakref

import pytest

import pydoppelgangerhunt
import pydoppelgangerhunt.fixer as fixer_mod
import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
import pydoppelgangerhunt.fixer.dataflow as dataflow_mod
from pydoppelgangerhunt.config import should_warn_closure_event
from pydoppelgangerhunt.fixer._df_cache import (  # pylint: disable=protected-access
    AnalysisSession,
    _DownstreamCacheKey,
    _MAX_PARSED_TREE_CACHE_SIZE,
    _ScopeCacheKey,
    _SessionBucket,
    _build_downstream_cache_key,
    _downstream_reads_cache,
    _ephemeral_unit_ast_context,
    _extract_unit_end_col,
    _extract_unit_start_col,
    _get_cached_tree_source_lines,
    _get_or_compute_scope_cached,
    _get_or_parse_tree,
    _get_scope_parent_map,
    _get_scope_stmts_by_end_lineno,
    _get_scope_stmts_by_start_lineno,
    _get_valid_unit_bounds,
    _is_ephemeral_unit_ast,
    _make_trusted_scope_cache_key,
    _mark_ephemeral_unit_ast,
    _parse_source_tree,
    _parsed_tree_cache,
    _parsed_tree_source_lines,
    _register_tree_source_lines,
    _store_lru_cache_entry,
    _tree_contains_scope_node,
    analysis_session,
    clear_dataflow_cache,
    get_active_analysis_session,
)
from pydoppelgangerhunt.fixer._df_span_index import (  # pylint: disable=protected-access
    _resolve_unit_ast_end_col,
)
from pydoppelgangerhunt.fixer.dataflow import collect_downstream_read_names
from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope


def test_public_session_exports_and_absence_of_global_ast_eviction_api() -> None:
    """Verifies AnalysisSession/analysis_session/clear_dataflow_cache exports and removal of evict_tree_scopes."""
    for mod in (pydoppelgangerhunt, fixer_mod, dataflow_mod, df_cache_mod):
        assert hasattr(mod, "AnalysisSession")
        assert hasattr(mod, "analysis_session")
        assert hasattr(mod, "clear_dataflow_cache")
        assert not hasattr(mod, "evict_tree_scopes")
        assert not hasattr(mod, "_evict_tree_scopes")


def test_analysis_session_lifecycle_reentrancy_and_gc_release() -> None:
    """Verifies nesting/re-entrancy, explicit session override, and immediate AST GC release on exit."""
    clear_dataflow_cache()
    assert get_active_analysis_session() is None

    tree_ref: weakref.ReferenceType[ast.Module]
    with analysis_session() as outer:
        assert get_active_analysis_session() is outer
        with analysis_session() as reused:
            assert reused is outer

        custom = AnalysisSession(max_bucket_entries=24)
        with analysis_session(custom) as inner:
            assert inner is custom
            assert get_active_analysis_session() is custom
            local_tree = ast.parse("def work(x):\n    return x + 1\n")
            tree_ref = weakref.ref(local_tree)
            _get_scope_parent_map(local_tree)
            _get_scope_stmts_by_end_lineno(local_tree)
            assert local_tree in custom.parent_maps
            assert local_tree not in outer.parent_maps
            del local_tree

        assert get_active_analysis_session() is outer
        custom.clear()
        assert len(custom.parent_maps) == 0
        assert len(custom.buckets) == 0
        del custom

    assert get_active_analysis_session() is None
    gc.collect()
    assert tree_ref() is None


def test_inplace_ast_mutation_observed_immediately_without_poisoning_source_hash_cache() -> None:
    """Verifies standalone mutated ASTs reflect edits immediately and never poison _downstream_reads_cache."""
    clear_dataflow_cache()
    src = (
        "def compute(a):\n"
        "    x = a + 1\n"
        "    y = x * 2\n"
        "    return y\n"
    )
    unit: dict[str, Any] = {
        "start": 2,
        "end": 2,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "x = a + 1\n",
    }

    caller_tree = ast.parse(src)
    # 1. Baseline call with caller-owned tree sees downstream read of x
    reads_before = collect_downstream_read_names(src, unit, tree=caller_tree)
    assert reads_before == {"x"}
    assert len(_downstream_reads_cache) == 0

    # 2. Mutate caller_tree in place to remove `y = x * 2` and `return y`
    fn_node = caller_tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    fn_node.body = [fn_node.body[0]]

    # Standalone call immediately observes mutation without any manual eviction
    reads_after = collect_downstream_read_names(src, unit, tree=caller_tree)
    assert reads_after == set()
    # Mutated caller AST must NOT have polluted immutable _downstream_reads_cache
    assert len(_downstream_reads_cache) == 0

    # 3. Source-text path (tree=None) computes from immutable source_text and still sees {"x"}
    reads_from_source = collect_downstream_read_names(src, unit)
    assert reads_from_source == {"x"}
    assert len(_downstream_reads_cache) == 1

    # 4. inspect_single_unit_scope also observes caller AST mutation immediately
    scope_before = inspect_single_unit_scope(
        {
            "start": 1,
            "end": 4,
            "kind": "function",
            "source_text": src,
        },
        tree=ast.parse(src),
    )
    mut_fn_tree = ast.parse(src)
    mut_fn = mut_fn_tree.body[0]
    assert isinstance(mut_fn, ast.FunctionDef)
    mut_fn.body = [mut_fn.body[0]]
    scope_after = inspect_single_unit_scope(
        {
            "start": 1,
            "end": 2,
            "kind": "function",
            "source_text": src,
        },
        tree=mut_fn_tree,
    )
    assert scope_before["has_return"] is True
    assert scope_after["has_return"] is False


def test_multithreaded_session_isolation_and_shared_context_safety() -> None:
    """Verifies distinct threads get isolated sessions and shared-context threads synchronize cleanly."""
    clear_dataflow_cache()
    errors: list[BaseException] = []
    seen_sessions: list[AnalysisSession] = []
    seen_lock = threading.Lock()

    def _isolated_worker(idx: int) -> None:
        try:
            with analysis_session() as sess:
                with seen_lock:
                    seen_sessions.append(sess)
                tree = ast.parse(f"def f_{idx}(v):\n    return v + {idx}\n")
                pmap = _get_scope_parent_map(tree)
                assert tree.body[0] in pmap
                val = _get_or_compute_scope_cached(
                    _SessionBucket.CELL_INFO,
                    _make_trusted_scope_cache_key(tree, f"iso_{idx}"),
                    lambda: idx,
                )
                assert val == idx
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            with seen_lock:
                errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_isolated_worker, i) for i in range(12)]
        for fut in futs:
            fut.result()

    if errors:
        raise errors[0]
    assert len({id(sess) for sess in seen_sessions}) == 12

    # Shared AnalysisSession via contextvars.copy_context().run
    shared_tree = ast.parse("def shared(a, b):\n    c = a + b\n    return c\n")
    compute_calls = 0
    compute_lock = threading.Lock()

    def _shared_worker(worker_idx: int) -> tuple[int, int]:
        def _compute() -> int:
            nonlocal compute_calls
            with compute_lock:
                compute_calls += 1
            return 42

        pmap = _get_scope_parent_map(shared_tree)
        assert shared_tree.body[0] in pmap
        shared_val = _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            _make_trusted_scope_cache_key(shared_tree, "shared_key"),
            _compute,
        )
        own_val = _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO,
            _make_trusted_scope_cache_key(shared_tree, f"worker_{worker_idx}"),
            lambda: worker_idx,
        )
        return shared_val, own_val

    with analysis_session() as shared_sess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            shared_futs = [
                pool.submit(contextvars.copy_context().run, _shared_worker, idx)
                for idx in range(16)
            ]
            results = [fut.result() for fut in shared_futs]
        assert all(shared_v == 42 for shared_v, _ in results)
        assert [own_v for _, own_v in results] == list(range(16))
        assert shared_tree in shared_sess.parent_maps
        assert compute_calls >= 1


def test_session_lru_eviction_for_buckets_and_parent_maps_and_unhashable_fallback() -> None:
    """Verifies LRU capacity bounds, recency refresh, node_to_roots pruning, and unhashable key fallback."""
    # Minimum capacity floor is clamped to 16
    sess = AnalysisSession(max_bucket_entries=4)
    with analysis_session(sess):
        # 1. Bucket LRU with recency refresh
        keys = [ast.Pass() for _ in range(20)]
        for idx, node in enumerate(keys[:16]):
            def _make_val(val: int = idx) -> int:
                return val

            _get_or_compute_scope_cached(
                _SessionBucket.CELL_INFO, node, _make_val
            )
        # Touch keys[0] so keys[1] becomes the oldest entry
        assert (
            _get_or_compute_scope_cached(
                _SessionBucket.CELL_INFO, keys[0], lambda: -1
            )
            == 0
        )
        _get_or_compute_scope_cached(
            _SessionBucket.CELL_INFO, keys[16], lambda: 16
        )
        bucket = sess.get_bucket(_SessionBucket.CELL_INFO)
        assert len(bucket) == 16
        assert keys[0] in bucket
        assert keys[1] not in bucket
        assert keys[16] in bucket

        # 2. Parent map LRU evicts oldest tree and prunes its node_to_roots links
        trees = [ast.parse(f"x_{i} = {i}\n") for i in range(20)]
        for t in trees:
            _get_scope_parent_map(t)
        assert len(sess.parent_maps) == 16
        assert trees[0] not in sess.parent_maps
        assert trees[0].body[0] not in sess.node_to_roots
        assert trees[-1] in sess.parent_maps
        assert trees[-1].body[0] in sess.node_to_roots

        # 3. Sub-node lookup reuses the enclosing root's parent map
        tail_tree = trees[-1]
        tail_pmap = _get_scope_parent_map(tail_tree)
        assert _get_scope_parent_map(tail_tree.body[0]) is tail_pmap

        # 4. Unhashable keys and AST nodes fall back cleanly
        class _UnhashableAST(ast.AST):
            __hash__ = None  # type: ignore[assignment]

        bad_node = _UnhashableAST()
        assert (
            _get_or_compute_scope_cached(
                _SessionBucket.CELL_INFO, bad_node, lambda: "fallback"
            )
            == "fallback"
        )
        assert _get_scope_parent_map(bad_node) == {}


def test_immutable_source_hash_parsed_tree_cache_and_clear() -> None:
    """Verifies _get_or_parse_tree, _store_lru_cache_entry, and clear_dataflow_cache."""
    clear_dataflow_cache()
    assert _get_or_parse_tree("") is None
    assert _get_or_parse_tree("   \n\t  ") is None
    assert _get_or_parse_tree("def broken(:\n") is None

    src = "def alpha(x):\n    return x * 2\n"
    t1 = _get_or_parse_tree(src)
    t2 = _get_or_parse_tree(src)
    assert t1 is not None
    assert t1 is t2
    assert _get_cached_tree_source_lines(t1) == ("def alpha(x):", "    return x * 2")

    # _parse_source_tree returns caller tree directly and registers lines only inside active session
    caller_tree = ast.parse("z = 1\n")
    with analysis_session() as sess:
        assert _parse_source_tree("z = 1\n", tree=caller_tree) is caller_tree
        assert _get_cached_tree_source_lines(caller_tree) == ("z = 1",)
        assert caller_tree in sess.tree_source_lines

    # _store_lru_cache_entry preserves canonical value on duplicate insert and enforces max_size
    local_lru: OrderedDict[str, list[int]] = OrderedDict()
    first = [1]
    assert _store_lru_cache_entry(local_lru, "k1", first, max_size=2) is first
    assert _store_lru_cache_entry(local_lru, "k1", [999], max_size=2) is first
    _store_lru_cache_entry(local_lru, "k2", [2], max_size=2)
    _store_lru_cache_entry(local_lru, "k3", [3], max_size=2)
    assert list(local_lru.keys()) == ["k2", "k3"]

    # Parsed tree cache stays bounded by _MAX_PARSED_TREE_CACHE_SIZE
    for idx in range(_MAX_PARSED_TREE_CACHE_SIZE + 10):
        _get_or_parse_tree(f"v_{idx} = {idx}\n")
    assert len(_parsed_tree_cache) == _MAX_PARSED_TREE_CACHE_SIZE

    should_warn_closure_event("lenient_closure_mode")
    assert should_warn_closure_event("lenient_closure_mode") is False
    with analysis_session() as active_sess:
        _get_scope_parent_map(caller_tree)
        clear_dataflow_cache()
        assert len(_parsed_tree_cache) == 0
        assert len(_parsed_tree_source_lines) == 0
        assert len(_downstream_reads_cache) == 0
        assert should_warn_closure_event("lenient_closure_mode") is True
        assert len(active_sess.parent_maps) == 0


def test_parent_map_excludes_singletons_and_tree_containment_checks() -> None:
    """Verifies singleton AST operators/contexts are excluded from parent maps and containment is accurate."""
    tree_a = ast.parse("x = (a + b) and (not c == d)\n")
    tree_b = ast.parse("y = (p + q) and (not r == s)\n")

    with analysis_session():
        pmap_a = _get_scope_parent_map(tree_a)
        assert tree_a not in pmap_a
        for child in pmap_a:
            assert not isinstance(
                child,
                (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop),
            )
        assert _tree_contains_scope_node(tree_a, tree_a) is True
        assert _tree_contains_scope_node(tree_a, tree_a.body[0]) is True
        assert _tree_contains_scope_node(tree_a, tree_b.body[0]) is False


def test_ephemeral_unit_ast_context_and_mark_skip_session_caching() -> None:
    """Verifies ephemeral synthetic ASTs bypass session buckets, parent maps, and containment."""
    tree = ast.parse("x = 1\n")
    owner = ast.parse("def f():\n    return 1\n")
    stmt = tree.body[0]

    with analysis_session() as sess:
        assert _is_ephemeral_unit_ast(None) is False
        assert _is_ephemeral_unit_ast(tree) is False

        with _ephemeral_unit_ast_context(tree):
            assert _is_ephemeral_unit_ast(tree) is True
            assert _tree_contains_scope_node(tree, tree) is False
            calls = 0

            def _compute() -> str:
                nonlocal calls
                calls += 1
                return f"call_{calls}"

            # Direct AST key and aux_ast_nodes key both bypass caching
            assert _get_or_compute_scope_cached(_SessionBucket.CELL_INFO, tree, _compute) == "call_1"
            assert _get_or_compute_scope_cached(_SessionBucket.CELL_INFO, tree, _compute) == "call_2"
            aux_key = _ScopeCacheKey(owner=owner, kind="aux", aux_ast_nodes=(tree,))
            assert _get_or_compute_scope_cached(_SessionBucket.CELL_INFO, aux_key, _compute) == "call_3"
            assert _get_or_compute_scope_cached(_SessionBucket.CELL_INFO, aux_key, _compute) == "call_4"
            assert len(sess.get_bucket(_SessionBucket.CELL_INFO)) == 0
            assert _register_tree_source_lines(tree, source_text="x = 1\n") is None
            pmap = _get_scope_parent_map(tree)
            assert stmt in pmap
            assert tree not in sess.parent_maps

        # Context exit clears ephemeral marker
        assert _is_ephemeral_unit_ast(tree) is False

        # Session-scoped _mark_ephemeral_unit_ast skips caching until sess.clear()
        marked = ast.parse("y = 2\n")
        assert _mark_ephemeral_unit_ast(marked) is marked
        assert _is_ephemeral_unit_ast(marked) is True
        _get_scope_parent_map(marked)
        assert marked not in sess.parent_maps
        sess.clear()
        assert _is_ephemeral_unit_ast(marked) is False


def test_register_and_lookup_tree_source_lines() -> None:
    """Verifies line-ending normalization, descendant lookup, and TypeError guard for bare strings."""
    tree = ast.parse("def f():\r\n    a = 1\r    return a\n")
    fn = tree.body[0]

    with pytest.raises(TypeError, match="non-string Sequence"):
        _register_tree_source_lines(tree, "def f():\n    return 1\n")

    assert _register_tree_source_lines(None, source_text="x = 1\n") is None
    assert _register_tree_source_lines(tree, source_lines=[]) is None
    assert _get_cached_tree_source_lines(None) is None

    with analysis_session():
        norm = _register_tree_source_lines(
            tree, source_text="def f():\r\n    a = 1\r    return a\n"
        )
        assert norm == ("def f():", "    a = 1", "    return a")
        assert _get_cached_tree_source_lines(tree) == norm
        assert _get_cached_tree_source_lines(fn) == norm
        assert fn in _get_scope_stmts_by_start_lineno(tree)[1]


@pytest.mark.parametrize(
    ("unit", "expected_bounds", "expected_start_col", "expected_end_col"),
    [
        pytest.param({"start": 2, "end": 5, "start_col": 4, "end_col": 12}, (2, 5), 4, 12, id="canonical"),
        pytest.param({"start": 3, "col_offset": 8, "end_col_offset": 20}, (3, 3), 8, 20, id="alias_cols_default_end"),
        pytest.param({"start": 0, "end": 2}, None, None, None, id="zero_start_invalid"),
        pytest.param({"start": 5, "end": 2}, None, None, None, id="inverted_span_invalid"),
        pytest.param({"start": "bad", "end": 3}, None, None, None, id="non_numeric_invalid"),
        pytest.param("not_a_dict", None, None, None, id="non_dict_invalid"),
    ],
)
def test_unit_coordinate_helpers(
    unit: Any,
    expected_bounds: tuple[int, int] | None,
    expected_start_col: int | None,
    expected_end_col: int | None,
) -> None:
    """Verifies unit line bounds and column alias extraction across valid and malformed inputs."""
    assert _get_valid_unit_bounds(unit) == expected_bounds
    assert _extract_unit_start_col(unit) == expected_start_col
    assert _extract_unit_end_col(unit) == expected_end_col


@pytest.mark.parametrize(
    ("src", "unit", "expected_end_col"),
    [
        pytest.param(
            "def f():\n    x = 1\n    y = 2\n",
            {"start": 2, "end": 2, "start_col": 4},
            9,
            id="single_line_assign",
        ),
        pytest.param(
            "if True:\n    x = (\n        1 + 2\n    )\n",
            {"start": 2, "end": 4, "start_col": 4},
            5,
            id="multiline_assign",
        ),
        pytest.param(
            "x = 1; y = 2\n",
            {"start": 1, "end": 1, "start_col": 7},
            12,
            id="second_stmt_on_semicolon_line",
        ),
        pytest.param(
            "x = 1; y = 2\n",
            {"start": 1, "end": 1, "start_col": 0},
            None,
            id="first_stmt_on_semicolon_line_ambiguous",
        ),
        pytest.param(
            "res = [i for i in range(3)]\n",
            {"start": 1, "end": 1, "start_col": 6, "kind": "comprehension"},
            None,
            id="comprehension_defers_to_whole_line",
        ),
        pytest.param(
            "def f():\n    foo()\n",
            {"start": 2, "end": 2, "start_col": 4},
            None,
            id="expr_stmt_with_start_col_defers",
        ),
        pytest.param(
            "def f():\n    x = 1\n",
            {"start": 99, "end": 99},
            None,
            id="no_stmt_on_line",
        ),
    ],
)
def test_resolve_unit_ast_end_col(
    src: str,
    unit: dict[str, Any],
    expected_end_col: int | None,
) -> None:
    """Verifies statement boundary end-column resolution across single-line, multiline, and semicolon forms."""
    tree = ast.parse(src)
    assert _resolve_unit_ast_end_col(tree, unit) == expected_end_col


def test_build_downstream_cache_key_differentiates_subroutine_and_source_hash() -> None:
    """Verifies _build_downstream_cache_key distinguishes kind, candidates, inputs, and unit source hashes."""
    src = "def f(a):\n    x = a + 1\n    return x\n"
    k_sub = _build_downstream_cache_key(
        src,
        {"kind": "compound_block", "inputs": ["a"], "source_code": "x = a + 1\n"},
        u_start=2,
        u_end=2,
        candidates={"x"},
        is_lenient=False,
    )
    k_fn = _build_downstream_cache_key(
        src,
        {"kind": "function", "inputs": ["a"], "source_code": "x = a + 1\n"},
        u_start=2,
        u_end=2,
        candidates={"x"},
        is_lenient=False,
    )
    k_cols = _build_downstream_cache_key(
        src,
        {
            "kind": "compound_block",
            "start_col": 4,
            "end_col": 13,
            "inputs": ["a"],
            "source_code": "x = a + 1\n",
        },
        u_start=2,
        u_end=2,
        candidates={"x"},
        is_lenient=False,
    )
    assert isinstance(k_sub, _DownstreamCacheKey)
    assert k_sub.is_subroutine is True
    assert k_fn.is_subroutine is False
    assert k_sub != k_fn
    assert k_sub.unit_source_hash is not None
    assert k_cols.unit_source_hash is None
    assert k_sub.candidates == frozenset({"x"})
    assert k_sub.unit_inputs == frozenset({"a"})
