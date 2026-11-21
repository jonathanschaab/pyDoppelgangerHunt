"""Tests for _df_cache LRU structures, parent-map budgets, AST eviction, and warning suppression."""

from __future__ import annotations

import ast
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any

import pytest

from pydoppelgangerhunt.fixer import (
    analyze_unit_variable_scope,
    clear_dataflow_cache,
    inspect_single_unit_scope,
    resolve_clone_generator_subroutine_outputs,
    resolve_clone_pair_outputs,
    synthesize_shared_helper_code,
)


def test_heterogeneous_frozenset_eviction_and_scope_cache_key() -> None:
    """Covers heterogeneous frozenset cache eviction, nested AST nodes, and _ScopeCacheKey."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _SpanBounds,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

    clear_dataflow_cache()
    evict_mod = ast.parse(
        "def outer(items):\n"
        "    def inner(y):\n"
        "        return [ (lambda z: z + y)(x) for x in items ]\n"
        "    return inner\n"
    )
    keep_mod = ast.parse("def survivor(items):\n    return [x for x in items]\n")
    outer_fn = evict_mod.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    ret_stmt = inner_fn.body[0]
    assert isinstance(ret_stmt, ast.Return)
    listcomp_node = ret_stmt.value
    assert isinstance(listcomp_node, ast.ListComp)
    call_node = listcomp_node.elt
    assert isinstance(call_node, ast.Call)
    lambda_node = call_node.func
    assert isinstance(lambda_node, ast.Lambda)

    het_keys = [
        frozenset({"metadata", outer_fn}),
        frozenset({"alpha", "beta", "gamma", 42, inner_fn}),
        ("composite", frozenset({"tag_a", "tag_b", lambda_node})),
        (
            "nested_comp",
            (_SpanBounds(1, 4), frozenset({"meta", listcomp_node})),
        ),
        _ScopeCacheKey(
            owner=keep_mod,
            kind="cross_ref",
            params=("param",),
            aux_ast_nodes=(inner_fn,),
        ),
    ]
    keep_key = _ScopeCacheKey(
        owner=keep_mod,
        kind="survivor_key",
        params=(frozenset({"only_strings", "safe"}),),
    )
    for idx, hk in enumerate(het_keys):
        _scope_cell_info_cache[hk] = f"evict_{idx}"
        _scope_closures_cache[hk] = [(1, set())]
    _scope_cell_info_cache[keep_key] = "keep_val"

    tree_nodes = set(ast.walk(evict_mod))
    for hk in het_keys:
        assert _key_references_scope_nodes(hk, tree_nodes) is True
    assert _key_references_scope_nodes(keep_key, tree_nodes) is False

    _evict_tree_scopes(evict_mod)
    for hk in het_keys:
        assert hk not in _scope_cell_info_cache
        assert hk not in _scope_closures_cache
    assert _scope_cell_info_cache.get(keep_key) == "keep_val"

    info_before = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    _evict_tree_scopes(evict_mod)
    info_after = _analyze_enclosing_fn_cells(
        outer_fn, (2, 3, 4, 54), mod_tree=evict_mod
    )
    assert info_before == info_after


def test_cached_vs_uncached_helper_equivalence_and_multipath_runtime_oracle() -> None:
    """Verifies cached vs uncached helper equivalence and multi-path runtime oracle intersection."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
        _extract_closure_nonlocal_mutations_and_deletions,
        _extract_fn_outer_carrier_targets,
        _extract_fn_returned_closure_refs,
        _get_fn_scope_context,
        _is_closure_factory_scope,
        _is_self_escaping_def_node,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer.scope import _analyze_block_assignment

    clear_dataflow_cache()
    helper_eq_tree = ast.parse(
        "def outer(box, items):\n"
        "    x = 1\n"
        "    @dec\n"
        "    def inner(v):\n"
        "        nonlocal x\n"
        "        x += v\n"
        "        box.append(lambda: x + v)\n"
        "        return lambda: x\n"
        "    return inner\n"
    )
    outer_eq = helper_eq_tree.body[0]
    assert isinstance(outer_eq, ast.FunctionDef)
    inner_eq = outer_eq.body[1]
    assert isinstance(inner_eq, ast.FunctionDef)
    eq_ctx = _get_fn_scope_context(outer_eq, mod_tree=helper_eq_tree)
    assert _extract_closure_nonlocal_mutations_and_deletions(
        inner_eq, use_cache=True
    ) == _extract_closure_nonlocal_mutations_and_deletions(inner_eq, use_cache=False)
    assert _extract_cached_scope_free_reads(
        inner_eq, use_cache=True
    ) == _extract_cached_scope_free_reads(inner_eq, use_cache=False)
    assert _is_self_escaping_def_node(
        inner_eq, use_cache=True
    ) == _is_self_escaping_def_node(inner_eq, use_cache=False)
    assert _is_closure_factory_scope(
        inner_eq, eq_ctx, use_cache=True
    ) == _is_closure_factory_scope(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_returned_closure_refs(
        inner_eq, eq_ctx, use_cache=True
    ) == _extract_fn_returned_closure_refs(inner_eq, eq_ctx, use_cache=False)
    assert _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=True
    ) == _extract_fn_outer_carrier_targets(
        inner_eq, frozenset(), eq_ctx, use_cache=False
    )

    multipath_snippet = (
        "def _probe(cond, fail):\n"
        "    always = 1\n"
        "    survives = 2\n"
        "    killed_on_err = 3\n"
        "    if cond:\n"
        "        branch_both = 10\n"
        "        only_true = 11\n"
        "    else:\n"
        "        branch_both = 20\n"
        "    try:\n"
        "        del killed_on_err\n"
        "        if fail:\n"
        "            raise ValueError('boom')\n"
        "        return ('early', always, survives, branch_both)\n"
        "    except ValueError:\n"
        "        pass\n"
        "    return set(locals().keys())\n"
    )
    ns_mp: dict[str, Any] = {}
    exec(compile(multipath_snippet, "<multipath>", "exec"), ns_mp)  # noqa: S102
    probe_fn = ns_mp["_probe"]
    fallthrough_key_sets: list[set[str]] = []
    for c_val in (True, False):
        for f_val in (True, False):
            res_mp = probe_fn(c_val, f_val)
            if isinstance(res_mp, set):
                fallthrough_key_sets.append(res_mp - {"cond", "fail"})
    runtime_definite = set.intersection(*fallthrough_key_sets)
    probe_body = ast.parse(multipath_snippet).body[0].body[:-1]  # type: ignore[attr-defined]
    static_def_1, _ = _analyze_block_assignment(probe_body)
    static_def_2, _ = _extract_block_definite_bindings_and_term(probe_body, None)
    assert static_def_1 == runtime_definite == {"always", "survives", "branch_both"}
    assert static_def_2 == runtime_definite

    orig_hazard_fn_src = (
        "def run_orig(items):\n"
        "    scale = 2\n"
        "    cbs = [lambda x, i=i: x * scale + i for i in items]\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    forced_extracted_src = (
        "def _forced_helper(items, scale):\n"
        "    return [lambda x, i=i: x * scale + i for i in items]\n"
        "def run_forced(items):\n"
        "    scale = 2\n"
        "    cbs = _forced_helper(items, scale)\n"
        "    scale = 10\n"
        "    return [cb(3) for cb in cbs]\n"
    )
    ns_neg_ctrl: dict[str, Any] = {}
    exec(
        compile(
            orig_hazard_fn_src + "\n" + forced_extracted_src,
            "<neg_ctrl>",
            "exec",
        ),
        ns_neg_ctrl,
    )  # noqa: S102
    assert ns_neg_ctrl["run_orig"]([1, 2]) == [31, 32]
    assert ns_neg_ctrl["run_forced"]([1, 2]) == [7, 8]
    assert ns_neg_ctrl["run_orig"]([1, 2]) != ns_neg_ctrl["run_forced"]([1, 2])


def test_scope_cache_key_permitted_parameter_types() -> None:
    """Verifies _ScopeCacheKey enforces permitted immutable AST-free parameter schema, bool/int non-collision, and hidden AST detection."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _make_trusted_scope_cache_key,
    )

    mod = ast.parse("x = 1\n")
    stmt = mod.body[0]

    valid_key = _ScopeCacheKey(
        mod,
        "test_kind",
        params=(
            "str_val",
            42,
            3.14,
            True,
            b"bytes",
            None,
            ("nested_tuple", frozenset({"a", "b", 1})),
        ),
        aux_ast_nodes=(stmt,),
    )
    assert valid_key.owner is mod
    assert valid_key.aux_ast_nodes == (stmt,)

    assert not hasattr(_ScopeCacheKey, "make_trusted")
    trusted_key = _make_trusted_scope_cache_key(mod, "trusted_kind", params=("ok", 1))
    assert trusted_key.kind == "trusted_kind"
    assert trusted_key == _ScopeCacheKey(mod, "trusted_kind", params=("ok", 1))

    # 1 and True (and 0 and False) must not collide in key params
    key_int_1 = _ScopeCacheKey(mod, "bool_int", params=(1,))
    key_bool_true = _ScopeCacheKey(mod, "bool_int", params=(True,))
    key_int_0 = _ScopeCacheKey(mod, "bool_int", params=(0,))
    key_bool_false = _ScopeCacheKey(mod, "bool_int", params=(False,))
    assert key_int_1 != key_bool_true
    assert key_int_0 != key_bool_false
    assert len({key_int_1, key_bool_true, key_int_0, key_bool_false}) == 4

    for bad_ast_params in (
        (stmt,),
        ((stmt,),),
        (frozenset({stmt}),),
        ([stmt],),
        ({stmt},),
        ({"k": stmt},),
        ({stmt: "v"},),
        (object(), stmt),
        ([object(), (stmt,)],),
    ):
        with pytest.raises(ValueError, match="must not contain ast.AST"):
            _ScopeCacheKey(mod, "bad_ast", params=bad_ast_params)

    for bad_type_params in (
        (["a", "b"],),
        ({"k": "v"},),
        ({"a", "b"},),
        (object(),),
    ):
        with pytest.raises(TypeError, match="only permits immutable"):
            _ScopeCacheKey(mod, "bad_type", params=bad_type_params)


def test_cyclic_mutable_containers_in_scope_cache_key() -> None:
    """Verifies self-referential and mutually recursive containers do not hang validation."""
    from pydoppelgangerhunt.fixer._df_cache import _ScopeCacheKey

    owner = ast.parse("x = 1")

    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_list,))

    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=(cyclic_dict,))

    mut_a: list[Any] = []
    mut_b: dict[str, Any] = {"a": mut_a}
    mut_a.append(mut_b)
    with pytest.raises(TypeError, match="only permits immutable"):
        _ScopeCacheKey(owner, "test", params=((mut_a, mut_b),))

    cyclic_with_ast: list[Any] = []
    cyclic_with_ast.append(cyclic_with_ast)
    cyclic_with_ast.append(owner.body[0])
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _ScopeCacheKey(owner, "test", params=(cyclic_with_ast,))


def test_indexed_cache_eviction_multi_owner_node_reuse_and_central_invariant() -> None:
    """Verifies indexed eviction, multi-owner keys, AST node reuse, and the central cache invariant."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _key_references_scope_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import _FrozenClosureList

    clear_dataflow_cache()

    tree_a = ast.parse("def fa():\n    return 1\n")
    tree_b = ast.parse("def fb():\n    return 2\n")
    tree_c = ast.parse("def fc():\n    return 3\n")
    multi_key = _ScopeCacheKey(
        owner=tree_a.body[0],
        kind="multi_aux",
        params=("tag", 1),
        aux_ast_nodes=(tree_b.body[0], tree_c),
    )
    _scope_cell_info_cache[multi_key] = ("cached_val",)
    assert multi_key in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_b.body[0], set())
    assert multi_key in _scope_owner_keys_index.get(tree_c, set())

    _evict_tree_scopes(tree_b)
    assert multi_key not in _scope_cell_info_cache
    assert multi_key not in _scope_owner_keys_index.get(tree_a.body[0], set())
    assert multi_key not in _scope_owner_keys_index.get(tree_c, set())

    reused_fn = tree_a.body[0]
    assert isinstance(reused_fn, ast.FunctionDef)
    ctx_before = _get_fn_scope_context(reused_fn, mod_tree=tree_a)
    assert ctx_before.mod_tree is tree_a
    _evict_tree_scopes(tree_a)
    assert not any(
        _key_references_scope_nodes(k, {reused_fn, tree_a})
        for k in _scope_cell_info_cache
    )
    new_wrapper_tree = ast.Module(
        body=[
            ast.ImportFrom(module="math", names=[ast.alias(name="*")], level=0),
            reused_fn,
        ],
        type_ignores=[],
    )
    ctx_after = _get_fn_scope_context(reused_fn, mod_tree=new_wrapper_tree)
    assert ctx_after.mod_tree is new_wrapper_tree
    assert ctx_after.has_star_import is True

    clear_dataflow_cache()
    trees = [
        ast.parse(f"def fn_{i}(x):\n    y = x + {i}\n    return y\n") for i in range(4)
    ]
    tree_node_sets = [set(ast.walk(t)) for t in trees]
    all_supported_keys: list[Any] = []
    for idx, t in enumerate(trees):
        fn_node = t.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        stmt_node = fn_node.body[0]
        next_t = trees[(idx + 1) % len(trees)]
        k_ast = fn_node
        k_sck_single = _ScopeCacheKey(
            owner=fn_node, kind="single", params=(idx, ("nested", True))
        )
        k_sck_multi = _ScopeCacheKey(
            owner=stmt_node,
            kind="multi",
            params=(frozenset({"a", "b"}),),
            aux_ast_nodes=(t, next_t.body[0]),
        )
        k_tuple = ("custom_tuple", (fn_node, idx), frozenset({"meta"}))
        k_frozenset = frozenset({"heterogeneous_label", stmt_node, idx})
        for k in (k_ast, k_sck_single, k_sck_multi, k_tuple, k_frozenset):
            _scope_cell_info_cache[k] = (idx, "cell")
            _scope_closures_cache[k] = _FrozenClosureList(
                [(None, frozenset({f"v_{idx}"}))]
            )
            all_supported_keys.append(k)

    evicted_nodes = tree_node_sets[1]
    _evict_tree_scopes(trees[1])
    for k in all_supported_keys:
        refs_evicted = _key_references_scope_nodes(k, evicted_nodes)
        if refs_evicted:
            assert k not in _scope_cell_info_cache
            assert k not in _scope_closures_cache
        else:
            assert k in _scope_cell_info_cache
            assert k in _scope_closures_cache


def test_scope_cache_key_kind_validation_cross_cache_eviction_and_detached_subtree() -> (
    None
):
    """Covers kind validation, cross-cache removal orders, central invariant, and detached subtree eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _extract_key_ast_nodes,
        _extract_value_ast_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_owner_keys_index,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _FrozenClosureList,
        _collect_scope_closures,
    )

    def _assert_central_index_invariant() -> None:
        all_cached_keys = set(_scope_closures_cache.keys()) | set(
            _scope_cell_info_cache.keys()
        )
        for cache in (_scope_closures_cache, _scope_cell_info_cache):
            for k, val in cache.items():
                for node in (
                    *_extract_key_ast_nodes(k),
                    *_extract_value_ast_nodes(val),
                ):
                    assert k in _scope_owner_keys_index.get(node, set())
                    assert node in _scope_key_nodes_index.get(k, set())
        for node, keys in _scope_owner_keys_index.items():
            assert isinstance(node, ast.AST)
            assert keys, "Empty key set must be popped from _scope_owner_keys_index"
            for k in keys:
                assert k in all_cached_keys
        for k, nodes in _scope_key_nodes_index.items():
            assert k in all_cached_keys
            assert nodes

    mod = ast.parse("def f():\n    return 1\n")
    other_node = mod.body[0]
    with pytest.raises(TypeError, match="kind must be a string"):
        _ScopeCacheKey(owner=mod, kind=other_node)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        _ScopeCacheKey(owner=mod, kind="")

    clear_dataflow_cache()
    shared_tree = ast.parse("def shared_fn():\n    return 42\n")
    shared_fn = shared_tree.body[0]
    shared_key = _ScopeCacheKey(owner=shared_fn, kind="shared_entry")
    frozen_closures = _FrozenClosureList([(None, frozenset({"a"}))])

    # Order A
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _assert_central_index_invariant()
    del _scope_closures_cache[shared_key]
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    del _scope_cell_info_cache[shared_key]
    assert shared_fn not in _scope_owner_keys_index
    assert shared_key not in _scope_key_nodes_index
    _assert_central_index_invariant()

    # Order B
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.pop(shared_key)
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.pop(shared_key)
    assert shared_fn not in _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order C
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_closures_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_cell_info_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order D
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _scope_cell_info_cache.clear()
    assert shared_key in _scope_owner_keys_index[shared_fn]
    _assert_central_index_invariant()
    _scope_closures_cache.clear()
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Order E
    _scope_closures_cache[shared_key] = frozen_closures
    _scope_cell_info_cache[shared_key] = frozenset({"a"})
    _evict_tree_scopes(shared_tree)
    assert shared_key not in _scope_closures_cache
    assert shared_key not in _scope_cell_info_cache
    assert not _scope_owner_keys_index
    _assert_central_index_invariant()

    # Eviction of a detached AST subtree invalidates ancestor cached values (including value-less entries)
    clear_dataflow_cache()
    nested_tree = ast.parse(
        "def outer():\n"
        "    x = 1\n"
        "    def inner():\n"
        "        return x + ext_var\n"
        "    return inner\n"
    )
    outer_fn = nested_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[1]
    assert isinstance(inner_fn, ast.FunctionDef)

    ctx_outer = _get_fn_scope_context(outer_fn, mod_tree=nested_tree)
    closures_outer = _collect_scope_closures(outer_fn, module_tree=nested_tree)
    free_reads_outer = _extract_cached_scope_free_reads(outer_fn, use_cache=True)
    assert "ext_var" in free_reads_outer
    free_reads_key = _ScopeCacheKey(outer_fn, "free_reads")
    assert free_reads_key in _scope_cell_info_cache
    assert inner_fn in ctx_outer.parent_map
    assert len(closures_outer) == 1
    assert nested_tree in _scope_parent_maps
    assert outer_fn not in _scope_parent_maps
    _assert_central_index_invariant()

    outer_fn.body.remove(inner_fn)
    _evict_tree_scopes(inner_fn)
    _assert_central_index_invariant()
    assert nested_tree not in _scope_parent_maps
    assert outer_fn not in _scope_parent_maps
    assert free_reads_key not in _scope_cell_info_cache
    assert not any(
        inner_fn in _extract_value_ast_nodes(v)
        for v in (
            *_scope_cell_info_cache.values(),
            *_scope_closures_cache.values(),
        )
    )


def test_extract_value_ast_nodes_mandatory_protocol_and_tracked_dict_copy() -> None:
    """Verifies mandatory cached_ast_nodes() protocol and independent _TrackedCachedDict copies."""
    import copy
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _debug_ast_immutability,
        _evict_tree_scopes,
        _extract_value_ast_nodes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
    )

    class _UnindexedCustomValue:
        def __init__(self, node: ast.AST) -> None:
            self.node = node

    class _ProperCustomValue:
        def __init__(self, node: ast.AST) -> None:
            self._node = node

        def cached_ast_nodes(self) -> tuple[ast.AST, ...]:
            return (self._node,)

    mod = ast.parse("x = 1\n")
    stmt = mod.body[0]
    bad_val = _UnindexedCustomValue(stmt)
    with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
        _extract_value_ast_nodes(bad_val)
    # Default production mode safely bypasses caching without crashing end users
    prev_dbg = _debug_ast_immutability[0]
    _debug_ast_immutability[0] = False
    try:
        _scope_cell_info_cache[_ScopeCacheKey(mod, "bad_val")] = bad_val
        assert _ScopeCacheKey(mod, "bad_val") not in _scope_cell_info_cache
        bypassed = _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _ScopeCacheKey(mod, "bad_val_graceful"),
            lambda: bad_val,
        )
        assert bypassed is bad_val
        assert _ScopeCacheKey(mod, "bad_val_graceful") not in _scope_cell_info_cache
    finally:
        _debug_ast_immutability[0] = prev_dbg
    # Debug mode strictly raises TypeError on unregistered cached value types
    _debug_ast_immutability[0] = True
    try:
        with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
            _scope_cell_info_cache[_ScopeCacheKey(mod, "bad_val_dbg")] = bad_val
        with pytest.raises(TypeError, match="must implement cached_ast_nodes"):
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                _ScopeCacheKey(mod, "bad_val_graceful_dbg"),
                lambda: bad_val,
            )
    finally:
        _debug_ast_immutability[0] = prev_dbg

    good_val = _ProperCustomValue(stmt)
    assert list(_extract_value_ast_nodes(good_val)) == [stmt]

    clear_dataflow_cache()
    td_key = _ScopeCacheKey(mod, "tracked_dict_test")
    _scope_cell_info_cache[td_key] = {"init": 1}
    tracked = _scope_cell_info_cache[td_key]
    assert isinstance(tracked, _TrackedCachedDict)
    cp1 = tracked.copy()
    cp2 = copy.copy(tracked)
    assert isinstance(cp1, _TrackedCachedDict)
    assert isinstance(cp2, _TrackedCachedDict)
    other_mod = ast.parse("y = 2\n")
    other_stmt = other_mod.body[0]
    cp1[other_stmt] = 99
    cp2[other_stmt] = 100
    _evict_tree_scopes(other_mod)
    assert td_key in _scope_cell_info_cache


def test_fn_scope_context_parent_map_immutability_and_enclosing_scope_key_isolation() -> (
    None
):
    """Covers MappingProxyType parent_map immutability and non-collision of enclosing_scopes in _make_scope_ctx_cache_key."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
        _make_nested_fn_scope_context,
        _make_scope_ctx_cache_key,
    )

    clear_dataflow_cache()
    src = (
        "def enc_a():\n"
        "    x = 1\n"
        "    def target():\n"
        "        return x\n"
        "    return target\n"
        "\n"
        "def enc_b():\n"
        "    x = 1\n"
        "    def target():\n"
        "        return x\n"
        "    return target\n"
    )
    tree = ast.parse(src)
    enc_a = tree.body[0]
    enc_b = tree.body[1]
    assert isinstance(enc_a, ast.FunctionDef)
    assert isinstance(enc_b, ast.FunctionDef)
    target_a = enc_a.body[1]
    target_b = enc_b.body[1]
    assert isinstance(target_a, ast.FunctionDef)
    assert isinstance(target_b, ast.FunctionDef)

    ctx_a = _get_fn_scope_context(enc_a, mod_tree=tree)
    ctx_b = _get_fn_scope_context(enc_b, mod_tree=tree)

    with pytest.raises(TypeError):
        ctx_a.parent_map[target_a] = enc_b  # type: ignore[index]
    nested_ctx_a = _make_nested_fn_scope_context(target_a, frozenset(), scope_ctx=ctx_a)
    nested_ctx_b = _make_nested_fn_scope_context(target_b, frozenset(), scope_ctx=ctx_b)
    with pytest.raises(TypeError):
        nested_ctx_a.parent_map[target_a] = enc_b  # type: ignore[index]

    assert nested_ctx_a.cache_fingerprint() == nested_ctx_b.cache_fingerprint()
    assert nested_ctx_a.enclosing_scopes != nested_ctx_b.enclosing_scopes
    shared_expr = target_a.body[0]
    key_a = _make_scope_ctx_cache_key(shared_expr, "expr_free_reads", nested_ctx_a)
    key_b = _make_scope_ctx_cache_key(shared_expr, "expr_free_reads", nested_ctx_b)
    assert key_a != key_b
    assert hash(key_a) != hash(key_b) or key_a.aux_ast_nodes != key_b.aux_ast_nodes


def test_span_line_index_detached_subtree_eviction_and_recomputation() -> None:
    """Verifies _SpanLineIndex AST node indexing across all 4 maps, detached subtree eviction, and fresh rebuild."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _get_span_line_index,
        _resolve_unit_col_bounds,
    )

    clear_dataflow_cache()
    src = (
        "def keep_fn(x):\n"
        "    return x + 1\n"
        "\n"
        "def detach_fn(y):\n"
        "    if y > 0:\n"
        "        z = y * 2\n"
        "    return y\n"
    )
    tree = ast.parse(src)
    keep_fn = tree.body[0]
    detach_fn = tree.body[1]
    assert isinstance(detach_fn, ast.FunctionDef)
    inner_if = detach_fn.body[0]
    assert isinstance(inner_if, ast.If)
    inner_assign = inner_if.body[0]

    idx1 = _get_span_line_index(tree)
    span_key = _ScopeCacheKey(tree, "span_line_index")
    assert span_key in _scope_cell_info_cache

    # _SpanLineIndex indexes only the root via _ScopeCacheKey.owner to avoid per-node memory overhead
    assert idx1.cached_ast_nodes() == ()
    assert span_key in _scope_owner_keys_index.get(tree, set())
    assert detach_fn not in _scope_owner_keys_index

    tree.body.pop(1)
    _evict_tree_scopes(detach_fn)

    assert span_key not in _scope_cell_info_cache
    assert detach_fn not in _scope_owner_keys_index
    assert inner_if not in _scope_owner_keys_index
    assert inner_assign not in _scope_owner_keys_index

    idx2 = _get_span_line_index(tree)
    assert idx2 is not idx1
    assert idx2.cached_ast_nodes() == ()
    assert 4 not in idx2.by_start_line
    assert 7 not in idx2.by_end_line
    assert 6 not in idx2.stmts_by_start_line
    assert 6 not in idx2.stmts_by_end_line

    assert isinstance(keep_fn, ast.FunctionDef)
    cols = _resolve_unit_col_bounds([keep_fn.body[0]], tree, 2, 2, None, None)
    assert cols == (4, 16)


def test_tracked_scope_cache_stmt_deltas_incremental_mutation_indexing() -> None:
    """Verifies that incremental mutations to cached stmt_deltas dictionaries update _scope_owner_keys_index."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _evict_tree_scopes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
    )

    clear_dataflow_cache()
    src = (
        "def outer(a):\n"
        "    x = a + 1\n"
        "    y = x + 2\n"
        "    def sub():\n"
        "        return x + y\n"
        "    return sub()\n"
    )
    tree = ast.parse(src)
    outer_fn = tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    stmt_x = outer_fn.body[0]
    stmt_y = outer_fn.body[1]

    _analyze_enclosing_fn_cells(outer_fn, (4, 5, 4, 16), mod_tree=tree)
    deltas_key = _ScopeCacheKey(
        owner=outer_fn, kind="stmt_deltas", aux_ast_nodes=(tree,)
    )
    assert deltas_key in _scope_cell_info_cache
    cached_deltas = _scope_cell_info_cache[deltas_key]
    assert isinstance(cached_deltas, _TrackedCachedDict)
    assert (stmt_x, True) in cached_deltas
    assert (stmt_y, True) in cached_deltas

    assert deltas_key in _scope_owner_keys_index[stmt_x]
    assert deltas_key in _scope_owner_keys_index[stmt_y]

    cached_deltas.pop((stmt_x, True))
    assert deltas_key not in _scope_owner_keys_index.get(stmt_x, set())
    assert deltas_key in _scope_owner_keys_index[stmt_y]
    _evict_tree_scopes(tree)
    assert deltas_key not in _scope_owner_keys_index.get(stmt_x, set())
    assert deltas_key not in _scope_owner_keys_index.get(stmt_y, set())

    tree2 = ast.parse("def f(z):\n    s1 = z + 1\n    s2 = s1 + 1\n")
    fn2 = tree2.body[0]
    assert isinstance(fn2, ast.FunctionDef)
    s1, s2 = fn2.body[0], fn2.body[1]
    manual_key = _ScopeCacheKey(owner=fn2, kind="stmt_deltas")
    manual_deltas: dict[tuple[ast.AST, bool], Any] = _get_or_compute_scope_cached(
        _scope_cell_info_cache, manual_key, dict
    )
    assert s1 not in _scope_owner_keys_index
    manual_deltas.setdefault((s1, False), None)
    assert manual_key in _scope_owner_keys_index[s1]
    manual_deltas[(s2, True)] = None
    assert manual_key in _scope_owner_keys_index[s2]

    tree3 = ast.parse("def g(w):\n    st = w * 2\n")
    fn3 = tree3.body[0]
    assert isinstance(fn3, ast.FunctionDef)
    st = fn3.body[0]
    plain_key = _ScopeCacheKey(owner=fn3, kind="stmt_deltas")
    plain_dict: dict[tuple[ast.AST, bool], Any] = {}
    _scope_cell_info_cache[plain_key] = plain_dict
    _scope_cell_info_cache[plain_key][(st, False)] = None
    assert plain_key in _scope_owner_keys_index[st]

    _evict_tree_scopes(st)
    assert plain_key not in _scope_cell_info_cache


def test_trusted_scope_cache_key_immutability_and_enclosing_chain_and_modes() -> None:
    """Verifies trusted key construction, same-line/multiline-header enclosing chains, and mode propagation."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _make_trusted_scope_cache_key,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _make_scope_ctx_cache_key,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
    )
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    dummy = ast.parse("x = 1").body[0]

    # _ScopeCacheKey validates params, whereas _make_trusted_scope_cache_key skips validation on hot internal paths
    with pytest.raises(TypeError, match="immutable"):
        _ScopeCacheKey(dummy, "kind", params=([1, 2],))
    with pytest.raises(ValueError, match="must not contain ast.AST"):
        _ScopeCacheKey(dummy, "kind", params=(dummy,))
    fast_key = _make_trusted_scope_cache_key(dummy, "kind", params=("a", 1))
    ctx_key = _make_scope_ctx_cache_key(dummy, "kind", None, extra_params=("a", 1))
    assert fast_key == _ScopeCacheKey(dummy, "kind", params=("a", 1))
    assert ctx_key.kind == "kind"

    chain_src = (
        "@(\n"
        "    dec\n"
        ")\n"
        "def outer(\n"
        "    a: int,\n"
        "    b: int,\n"
        ") -> int:\n"
        "    def mid(c: int) -> int:\n"
        "        def inner(d: int) -> int: return a + b + c + d\n"
        "        return inner(c)\n"
        "    return mid(a)\n"
    )
    chain_tree = ast.parse(chain_src)
    chain = _find_enclosing_function_chain(
        chain_tree, 9, 9, u_start_col=34, u_end_col=54
    )
    assert [getattr(n, "name", "") for _, _, n in chain] == [
        "outer",
        "mid",
        "inner",
    ]
    hdr_chain = _find_enclosing_function_chain(chain_tree, 5, 6)
    assert hdr_chain == []

    mode_src = (
        "def f1(n):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    x = n + 1\n"
        "    yield x\n"
        "    return x\n"
        "\n"
        "def f2(n):\n"
        "    x = 1\n"
        "    cb = lambda: x\n"
        "    x = n + 1\n"
        "    yield x\n"
        "    return x\n"
    )
    u1 = {
        "name": "u1",
        "kind": "compound_block",
        "start": 4,
        "end": 5,
        "source_text": mode_src,
    }
    u2 = {
        "name": "u2",
        "kind": "compound_block",
        "start": 12,
        "end": 13,
        "source_text": mode_src,
    }

    s1_strict = inspect_single_unit_scope({**u1, "skip_pre_unit_closures": False})
    s2_strict = inspect_single_unit_scope({**u2, "skip_pre_unit_closures": False})
    s1_lenient = ensure_unit_scope_closure_mode(u1, s1_strict, True)
    assert s1_strict["skip_pre_unit_closures"] is False
    assert s1_lenient["skip_pre_unit_closures"] is True
    assert "severed_closure_cell" in s1_strict["control_flow_hazards"]
    assert "severed_closure_cell" not in s1_lenient["control_flow_hazards"]

    s2_lenient = inspect_single_unit_scope({**u2, "skip_pre_unit_closures": True})
    pair_strict = analyze_unit_variable_scope(
        u1,
        u2,
        scope1=s1_lenient,
        scope2=s2_lenient,
        skip_pre_unit_closures=False,
    )
    assert pair_strict["skip_pre_unit_closures"] is False
    assert "severed_closure_cell" in pair_strict["control_flow_hazards"]

    assert (
        resolve_clone_pair_outputs(
            u1,
            u2,
            scope1=s1_lenient,
            scope2=s2_lenient,
            closure_strictness="strict",
        )
        is None
    )
    outs_lenient = resolve_clone_pair_outputs(
        u1,
        u2,
        scope1=s1_strict,
        scope2=s2_strict,
        closure_strictness="lenient",
    )
    assert outs_lenient == (["x"], ["x"])

    helper_strict = synthesize_shared_helper_code(
        u1,
        u2,
        helper_name="_helper",
        scope1=s1_lenient,
        scope2=s2_lenient,
        closure_strictness="strict",
    )
    assert helper_strict == ""

    pair_lenient = analyze_unit_variable_scope(
        u1,
        u2,
        scope1=s1_lenient,
        scope2=s2_lenient,
        skip_pre_unit_closures=True,
    )
    helper_lenient = synthesize_shared_helper_code(
        u1,
        u2,
        helper_name="_helper",
        scope=pair_lenient,
        scope1=s1_lenient,
        scope2=s2_lenient,
        outputs=["x"],
        outputs2=["x"],
        closure_strictness="lenient",
    )
    assert "def _helper(" in helper_lenient

    assert (
        resolve_clone_generator_subroutine_outputs(
            u1,
            u2,
            s1_lenient,
            s2_lenient,
            source_text1=mode_src,
            source_text2=mode_src,
            closure_strictness="strict",
        )
        is None
    )
    gen_outs_lenient = resolve_clone_generator_subroutine_outputs(
        u1,
        u2,
        s1_strict,
        s2_strict,
        source_text1=mode_src,
        source_text2=mode_src,
        closure_strictness="lenient",
    )
    assert gen_outs_lenient == (["x"], ["x"])


def test_parent_map_and_stmt_line_map_detached_child_eviction() -> None:
    """Verifies child-subtree eviction for parent_maps and statement line maps via ancestor walk."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _get_scope_parent_map,
        _get_scope_stmts_by_end_lineno,
        _get_scope_stmts_by_start_lineno,
        _scope_parent_maps,
        _scope_stmts_by_end,
        _scope_stmts_by_start,
    )

    clear_dataflow_cache()
    aux_tree = ast.parse(
        "def top_fn(x):\n"
        "    def child_fn(y):\n"
        "        return y + 1\n"
        "    return child_fn(x)\n"
    )
    top_fn = aux_tree.body[0]
    assert isinstance(top_fn, ast.FunctionDef)
    child_fn = top_fn.body[0]
    assert isinstance(child_fn, ast.FunctionDef)
    _ = _get_scope_parent_map(aux_tree)
    _ = _get_scope_stmts_by_start_lineno(top_fn)
    _ = _get_scope_stmts_by_end_lineno(top_fn)
    assert aux_tree in _scope_parent_maps
    assert top_fn in _scope_stmts_by_start
    assert top_fn in _scope_stmts_by_end

    top_fn.body.remove(child_fn)
    _evict_tree_scopes(child_fn)
    assert aux_tree not in _scope_parent_maps
    assert top_fn not in _scope_stmts_by_start
    assert top_fn not in _scope_stmts_by_end


def test_cache_lock_reentrancy_and_builtin_call_shadowing() -> None:
    """Verifies lock reentrancy and _is_non_closure_builtin_call fail-closed behavior."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _downstream_cache_lock,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _FnScopeContext,
        _extract_expr_closure_and_load_info,
        _is_non_closure_builtin_call,
    )

    assert isinstance(_downstream_cache_lock, type(threading.RLock()))

    len_call = ast.parse("len(x)", mode="eval").body
    assert isinstance(len_call, ast.Call)
    assert _is_non_closure_builtin_call(len_call, None) is False
    unshadowed_ctx = _FnScopeContext(
        fn_locals=frozenset(),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset(),
        has_star_import=False,
    )
    assert _is_non_closure_builtin_call(len_call, unshadowed_ctx) is True

    direct_call = ast.parse("custom_callee(arg_val)", mode="eval").body
    _, _, loads_without_out = _extract_expr_closure_and_load_info(
        [direct_call], called_names_out=None
    )
    assert {"custom_callee", "arg_val"} <= loads_without_out


def test_enclosing_scope_sensitivity_and_parent_map_reverse_index_eviction() -> None:
    """Verifies enclosing-scope sensitivity in _get_fn_scope_context and _collect_scope_closures, plus O(1) reverse-index parent-map eviction and sibling isolation."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _evict_tree_scopes,
        _get_scope_parent_map,
        _node_to_parent_map_roots,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_parent_maps,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _get_fn_scope_context,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
    )

    clear_dataflow_cache()
    enc_src = (
        "def outer_shadow(list):\n"
        "    def inner(items):\n"
        "        return list(x for x in items)\n"
        "    return inner\n"
        "\n"
        "def outer_clean():\n"
        "    def inner(items):\n"
        "        return list(x for x in items)\n"
        "    return inner\n"
    )
    enc_tree = ast.parse(enc_src)
    outer_sh = enc_tree.body[0]
    outer_cl = enc_tree.body[1]
    assert isinstance(outer_sh, ast.FunctionDef)
    assert isinstance(outer_cl, ast.FunctionDef)
    shared_inner = outer_sh.body[0]
    assert isinstance(shared_inner, ast.FunctionDef)

    ctx_sh = _get_fn_scope_context(
        shared_inner, mod_tree=None, enclosing_scopes=(outer_sh,)
    )
    ctx_cl = _get_fn_scope_context(
        shared_inner, mod_tree=None, enclosing_scopes=(outer_cl,)
    )
    assert "list" in ctx_sh.shadowed_names
    assert "list" not in ctx_cl.shadowed_names
    # _FnScopeContext.cached_ast_nodes() only indexes cache_aux_nodes, not every AST node in parent_map
    assert set(ctx_sh.cached_ast_nodes()) == set(ctx_sh.cache_aux_nodes())

    clos_sh = _collect_scope_closures(
        shared_inner, module_tree=None, enclosing_scopes=(outer_sh,)
    )
    clos_cl = _collect_scope_closures(
        shared_inner, module_tree=None, enclosing_scopes=(outer_cl,)
    )
    assert len(clos_sh) == 1  # shadowed list -> lazy genexp closure
    assert len(clos_cl) == 0  # unshadowed list -> eager genexp

    _ = _get_scope_parent_map(enc_tree)
    _ = _get_scope_parent_map(outer_sh)
    _ = _get_scope_parent_map(outer_cl)
    clean_inner = outer_cl.body[0]
    assert isinstance(clean_inner, ast.FunctionDef)
    assert enc_tree in _node_to_parent_map_roots.get(shared_inner, set())
    assert outer_sh in _node_to_parent_map_roots.get(shared_inner, set())
    assert outer_cl in _node_to_parent_map_roots.get(clean_inner, set())

    # Detaching a statement inside outer_cl evicts outer_cl and enc_tree parent maps,
    # while preserving unrelated sibling outer_sh!
    outer_cl.body.remove(clean_inner)
    _evict_tree_scopes(clean_inner)
    assert outer_cl not in _scope_parent_maps
    assert enc_tree not in _scope_parent_maps
    assert outer_sh in _scope_parent_maps

    _evict_tree_scopes(enc_tree)
    assert not _scope_closures_cache
    assert not _scope_cell_info_cache
    assert not _scope_key_nodes_index
    assert not _scope_parent_maps
    assert not _node_to_parent_map_roots


def test_fn_scope_context_fingerprint_downstream_cache_key_and_multiline_decorator_start() -> (
    None
):
    """Verifies _FnScopeContext.cache_fingerprint, _build_downstream_cache_key, and _node_earliest_start_pos."""
    from pydoppelgangerhunt.fixer._df_cache import _build_downstream_cache_key
    from pydoppelgangerhunt.fixer._df_cell_carriers import _FnScopeContext
    from pydoppelgangerhunt.fixer._df_closures import _node_earliest_start_pos

    ml_dec_src = (
        "x = 1\n"
        "@(\n"
        "    decorator_factory\n"
        ")\n"
        "def decorated_fn():\n"
        "    return 1\n"
    )
    ml_dec_tree = ast.parse(ml_dec_src)
    dec_fn_node = ml_dec_tree.body[1]
    assert _node_earliest_start_pos(
        dec_fn_node, source_lines=ml_dec_src.splitlines()
    ) == (2, 0)

    fp_empty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    fp_nonempty = _FnScopeContext(
        fn_locals=frozenset({"x"}),
        parent_map=MappingProxyType({dec_fn_node: ml_dec_tree}),
        shadowed_names=frozenset({"x"}),
        has_star_import=False,
    ).cache_fingerprint()
    assert fp_empty != fp_nonempty
    assert fp_empty[-1] is False
    assert fp_nonempty[-1] is True

    dk_1 = _build_downstream_cache_key(
        ml_dec_src,
        {"start": 2, "end": 2, "source_code": "a = 1"},
        2,
        2,
        {"a"},
        False,
    )
    dk_2 = _build_downstream_cache_key(
        ml_dec_src,
        {"start": 2, "end": 2, "source_code": "b = 2"},
        2,
        2,
        {"a"},
        False,
    )
    assert dk_1 != dk_2


def test_get_scope_parent_map_excludes_expr_context_tokens() -> None:
    """Verifies _get_scope_parent_map skips shared ast.expr_context singleton tokens."""
    from pydoppelgangerhunt.fixer._df_cache import _get_scope_parent_map

    cls_tree = ast.parse("def enclosing(x):\n    y = x + 1\n    return y\n")
    fn_node = cls_tree.body[0]
    pmap = _get_scope_parent_map(cls_tree)
    assert not any(isinstance(k, ast.expr_context) for k in pmap)
    assert fn_node in pmap


def test_syntax_warning_propagates_across_library_parse_and_inference_paths() -> None:
    """Verifies library parse, symtable, column inference, and stmt dump paths do not suppress SyntaxWarning."""
    import warnings
    from pydoppelgangerhunt.fixer.scope import (
        _get_or_parse_tree,
        _infer_missing_col_from_line_ast,
        _inspect_unit_scope,
        _parse_normalized_stmt_dumps,
    )

    escape_src = (
        'PAT = "\\d+\\s+"\n'
        "def match_fn(x):\n"
        "    def inner():\n"
        '        return ("\\d+" + str(x))\n'
        "    return inner()\n"
    )
    with warnings.catch_warnings(record=True) as w_parse:
        warnings.simplefilter("always")
        esc_tree = _get_or_parse_tree(escape_src)
        assert isinstance(esc_tree, ast.Module)
    assert any(issubclass(w.category, (SyntaxWarning, DeprecationWarning)) for w in w_parse)

    u_esc = {
        "name": "match_fn",
        "kind": "function",
        "start": 2,
        "end": 5,
        "source_text": escape_src,
    }
    with warnings.catch_warnings(record=True) as w_scope:
        warnings.simplefilter("always")
        sc_esc = _inspect_unit_scope(u_esc, file_tree=esc_tree)
        assert sc_esc["is_control_flow_safe"] is True
    assert any(issubclass(w.category, (SyntaxWarning, DeprecationWarning)) for w in w_scope)

    with warnings.catch_warnings(record=True) as w_col:
        warnings.simplefilter("always")
        col, ecol = _infer_missing_col_from_line_ast(
            '    y = ("\\d+" + str(x))',
            None,
            24,
            allow_expr_fallback=True,
        )
        assert col == 4 and ecol == 24
    assert any(issubclass(w.category, (SyntaxWarning, DeprecationWarning)) for w in w_col)

    with warnings.catch_warnings(record=True) as w_dump:
        warnings.simplefilter("always")
        dumps = _parse_normalized_stmt_dumps('PAT = "\\d+\\s+"\n')
        assert dumps is not None and len(dumps) == 1
    assert any(issubclass(w.category, (SyntaxWarning, DeprecationWarning)) for w in w_dump)


def test_tracked_cached_dict_prunes_reverse_index_and_missing_cached_ast_nodes_raises() -> (
    None
):
    """Verifies _TrackedCachedDict removes stale reverse index entries on overwrite/pop/del and missing cached_ast_nodes() raises TypeError."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _TrackedCachedDict,
        _debug_ast_immutability,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
    )

    clear_dataflow_cache()
    mod = ast.parse("def f():\n    a = 1\n    b = 2\n")
    fn = mod.body[0]
    n1 = fn.body[0]  # type: ignore[attr-defined]
    n2 = fn.body[1]  # type: ignore[attr-defined]
    ck = _ScopeCacheKey(fn, "unit_test_tracked_dict")
    _scope_cell_info_cache[ck] = {}
    tracked = _scope_cell_info_cache[ck]
    assert isinstance(tracked, _TrackedCachedDict)
    tracked["k1"] = n1
    assert ck in _scope_owner_keys_index.get(n1, set())
    tracked["k1"] = n2
    assert ck not in _scope_owner_keys_index.get(n1, set())
    assert ck in _scope_owner_keys_index.get(n2, set())
    popped = tracked.pop("k1")
    assert popped is n2
    assert ck not in _scope_owner_keys_index.get(n2, set())

    class _UnindexedCustomBox:
        pass

    prev_dbg_box = _debug_ast_immutability[0]
    _debug_ast_immutability[0] = False
    try:
        unindexed_res = _get_or_compute_scope_cached(
            _scope_cell_info_cache,
            _ScopeCacheKey(fn, "unit_test_bad_box_cache"),
            _UnindexedCustomBox,
        )
        assert isinstance(unindexed_res, _UnindexedCustomBox)
        assert _ScopeCacheKey(fn, "unit_test_bad_box_cache") not in _scope_cell_info_cache
    finally:
        _debug_ast_immutability[0] = prev_dbg_box
    _debug_ast_immutability[0] = True
    try:
        with pytest.raises(TypeError, match="cached_ast_nodes"):
            _get_or_compute_scope_cached(
                _scope_cell_info_cache,
                _ScopeCacheKey(fn, "unit_test_bad_box_cache_dbg"),
                _UnindexedCustomBox,
            )
    finally:
        _debug_ast_immutability[0] = prev_dbg_box


def test_cross_cache_reverse_index_sync_and_aux_ast_node_eviction() -> None:
    """Verifies sibling cache reverse-index synchronization and aux_ast_nodes eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _evict_tree_scopes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_owner_keys_index,
    )

    clear_dataflow_cache()
    mod_a = ast.parse("def fa():\n    x = 1\n    y = 2\n")
    mod_b = ast.parse("def fb():\n    z = 3\n")
    fn_a = mod_a.body[0]
    stmt_x = fn_a.body[0]  # type: ignore[attr-defined]
    stmt_y = fn_a.body[1]  # type: ignore[attr-defined]
    fn_b = mod_b.body[0]

    shared_key = _ScopeCacheKey(fn_a, "shared_cross_cache_test")
    _scope_closures_cache[shared_key] = (stmt_x,)
    _scope_cell_info_cache[shared_key] = {}
    tracked_dict = _scope_cell_info_cache[shared_key]
    tracked_dict["node"] = stmt_y

    assert shared_key in _scope_owner_keys_index.get(stmt_x, set())
    assert shared_key in _scope_owner_keys_index.get(stmt_y, set())

    # Deleting from _scope_cell_info_cache must keep stmt_x indexed for _scope_closures_cache
    del _scope_cell_info_cache[shared_key]
    assert shared_key in _scope_owner_keys_index.get(stmt_x, set())
    assert shared_key not in _scope_owner_keys_index.get(stmt_y, set())

    # Key owned by mod_a with aux_ast_nodes in mod_b must be evicted when mod_b is evicted
    aux_key = _ScopeCacheKey(fn_a, "aux_evict_test", aux_ast_nodes=(fn_b,))
    _scope_cell_info_cache[aux_key] = ("ok",)
    assert aux_key in _scope_owner_keys_index.get(fn_b, set())
    _evict_tree_scopes(mod_b)
    assert aux_key not in _scope_cell_info_cache
    assert aux_key not in _scope_owner_keys_index.get(fn_a, set())


def test_scope_cache_key_frozenset_and_nested_int_float_bool_non_collision() -> None:
    """Verifies _ScopeCacheKey distinguishes int, float, and bool inside frozensets and nested tuples."""
    from pydoppelgangerhunt.fixer._df_cache import _ScopeCacheKey

    mod = ast.parse("x = 1\n")
    k_int = _ScopeCacheKey(mod, "fs", params=(frozenset({2}),))
    k_float = _ScopeCacheKey(mod, "fs", params=(frozenset({2.0}),))
    k_nested_int = _ScopeCacheKey(mod, "fs", params=(frozenset({(1, "a")}),))
    k_nested_bool = _ScopeCacheKey(mod, "fs", params=(frozenset({(True, "a")}),))

    assert k_int != k_float
    assert k_nested_int != k_nested_bool
    assert len({k_int, k_float, k_nested_int, k_nested_bool}) == 4


def test_stmt_deltas_cap_and_nan_cache_key_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _MAX_STMT_DELTAS_PER_FN LRU reuse and _ScopeCacheKey NaN equality/hashing."""
    import pydoppelgangerhunt.fixer._df_cell_traversal as trav_mod
    from pydoppelgangerhunt.fixer._df_cache import (
        _MAX_STMT_DELTAS_PER_FN,
        _ScopeCacheKey,
        _scope_cell_info_cache,
    )
    from pydoppelgangerhunt.fixer._df_cell_carriers import _SpanBounds
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells_uncached,
    )

    clear_dataflow_cache()
    many_lines = (
        ["def big_fn(x):\n"]
        + [f"    v_{i} = x + {i}\n" for i in range(_MAX_STMT_DELTAS_PER_FN + 20)]
        + ["    return v_0\n"]
    )
    big_tree = ast.parse("".join(many_lines))
    big_fn = big_tree.body[0]
    assert isinstance(big_fn, ast.FunctionDef)
    info = _analyze_enclosing_fn_cells_uncached(
        big_fn,
        _SpanBounds(2, 5),
        mod_tree=big_tree,
        use_stmt_deltas=True,
    )
    assert info.pre_bound == frozenset({"x"})
    assert info.pre_unbound == frozenset()
    stmt_deltas = _scope_cell_info_cache.get(
        _ScopeCacheKey(big_fn, "stmt_deltas", aux_ast_nodes=(big_tree,))
    )
    assert isinstance(stmt_deltas, dict)
    assert len(stmt_deltas) == _MAX_STMT_DELTAS_PER_FN
    assert all(
        isinstance(k, tuple)
        and len(k) == 2
        and k[0] in big_fn.body
        and isinstance(k[1], bool)
        for k in stmt_deltas
    )

    orig_apply = trav_mod._try_apply_cached_stmt_delta
    pass2_hits = 0

    def _counting_apply(*args: Any, **kwargs: Any) -> Any:
        nonlocal pass2_hits
        res = orig_apply(*args, **kwargs)
        if res is True:
            pass2_hits += 1
        return res

    monkeypatch.setattr(trav_mod, "_try_apply_cached_stmt_delta", _counting_apply)
    info2 = _analyze_enclosing_fn_cells_uncached(
        big_fn,
        _SpanBounds(2, 6),
        mod_tree=big_tree,
        use_stmt_deltas=True,
    )
    assert info2.pre_bound == frozenset({"x"})
    assert info2.pre_unbound == frozenset()
    assert len(stmt_deltas) == _MAX_STMT_DELTAS_PER_FN
    assert pass2_hits >= _MAX_STMT_DELTAS_PER_FN - 10

    owner = ast.parse("x = 1\n")
    k_nan_1 = _ScopeCacheKey(
        owner, "nan_test", (float("nan"), ("nested", float("nan")))
    )
    k_nan_2 = _ScopeCacheKey(
        owner, "nan_test", (float("nan"), ("nested", float("nan")))
    )
    assert k_nan_1 == k_nan_2
    assert hash(k_nan_1) == hash(k_nan_2)
    cache_probe = {k_nan_1: "hit"}
    assert cache_probe.get(k_nan_2) == "hit"


def test_public_evict_tree_scopes_invalidates_mutated_ast_caches() -> None:
    """Verifies public evict_tree_scopes invalidates cached scope intervals after in-place AST coordinate mutation."""
    from pydoppelgangerhunt import evict_tree_scopes as top_evict_tree_scopes
    from pydoppelgangerhunt.fixer import evict_tree_scopes
    from pydoppelgangerhunt.fixer.source import _get_tree_scope_intervals

    assert top_evict_tree_scopes is evict_tree_scopes
    clear_dataflow_cache()
    tree = ast.parse("def alpha():\n    return 1\n")
    intervals_before = _get_tree_scope_intervals(tree)
    assert len(intervals_before) == 1
    assert intervals_before[0][:2] == (1, 2)

    fn_node = tree.body[0]
    fn_node.lineno = 10
    fn_node.end_lineno = 20
    evict_tree_scopes(tree)
    intervals_after = _get_tree_scope_intervals(tree)
    assert intervals_after[0][:2] == (10, 20)


def test_make_scope_ctx_cache_key_rejects_mutable_extra_params() -> None:
    """Verifies _make_scope_ctx_cache_key rejects mutable extra_params."""
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _make_scope_ctx_cache_key,
    )

    mod = ast.parse("x = 1\n")
    with pytest.raises(TypeError, match="immutable"):
        _make_scope_ctx_cache_key(mod, "test_kind", None, extra_params=([1, 2],))
    with pytest.raises(TypeError, match="immutable"):
        _make_scope_ctx_cache_key(mod, "test_kind", None, extra_params=({"a": 1},))


def test_lazy_parent_map_eviction_and_total_node_budget() -> None:
    """Verifies _TrackedScopeCache skips duplicate nested parent maps and enforces _MAX_PARENT_MAP_TOTAL_NODES budget."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _MAX_PARENT_MAP_TOTAL_NODES,
        _ScopeCacheKey,
        _downstream_cache_lock,
        _evict_tree_scopes,
        _get_scope_parent_map,
        _parent_map_total_nodes,
        _scope_cell_info_cache,
        _scope_parent_maps,
        _store_parent_map_locked,
    )

    clear_dataflow_cache()
    mod = ast.parse("def outer():\n    def inner():\n        return 1\n")
    outer_fn = mod.body[0]
    inner_fn = outer_fn.body[0]  # type: ignore[attr-defined]
    mod_ck = _ScopeCacheKey(mod, "mod_test")
    _scope_cell_info_cache[mod_ck] = ("mod_val",)
    assert mod in _scope_parent_maps
    assert _get_scope_parent_map(inner_fn, use_cache=True) is _get_scope_parent_map(
        mod, use_cache=True
    )
    # Inserting a nested scope key when mod is already indexed skips building a duplicate parent map for inner_fn
    ck = _ScopeCacheKey(inner_fn, "lazy_evict_test")
    _scope_cell_info_cache[ck] = ("cached_val",)
    assert inner_fn not in _scope_parent_maps

    # Evicting outer_fn (non-Module subtree) evicts both inner_fn and ancestor mod keys
    _evict_tree_scopes(outer_fn)
    assert ck not in _scope_cell_info_cache
    assert mod_ck not in _scope_cell_info_cache

    clear_dataflow_cache()
    assert _parent_map_total_nodes[0] == 0
    big_pm: dict[ast.AST, ast.AST] = {ast.Name(id=f"n{i}"): mod for i in range(500)}
    with _downstream_cache_lock:
        _store_parent_map_locked(mod, big_pm)
        assert _parent_map_total_nodes[0] == 500
        # Exceeding _MAX_PARENT_MAP_TOTAL_NODES evicts older parent maps to stay within budget
        mod2 = ast.parse("x = 1\n")
        huge_pm: dict[ast.AST, ast.AST] = {
            ast.Name(id=f"m{i}"): mod2 for i in range(_MAX_PARENT_MAP_TOTAL_NODES - 100)
        }
        _store_parent_map_locked(mod2, huge_pm)
        assert mod not in _scope_parent_maps
        assert mod2 in _scope_parent_maps
        assert _parent_map_total_nodes[0] == len(huge_pm)
        # A single map exceeding _MAX_PARENT_MAP_TOTAL_NODES must not evict existing cached maps
        mod3 = ast.parse("y = 2\n")
        oversized_pm: dict[ast.AST, ast.AST] = {
            ast.Name(id=f"z{i}"): mod3 for i in range(_MAX_PARENT_MAP_TOTAL_NODES + 5)
        }
        _store_parent_map_locked(mod3, oversized_pm)
        assert mod3 not in _scope_parent_maps
        assert mod2 in _scope_parent_maps
        assert _parent_map_total_nodes[0] == len(huge_pm)
    clear_dataflow_cache()
    assert _parent_map_total_nodes[0] == 0


def test_ast_immutability_fingerprint_and_unavailable_source_closure_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies AST mutation detection, unavailable_source_for_closure_mode skip reason, receiver sibling filtering, and module SyntaxError isolation."""
    import logging
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _debug_ast_immutability,
        _evict_tree_scopes,
        _get_or_compute_scope_cached,
        _scope_cell_info_cache,
        _scope_stmts_by_end,
        _scope_stmts_by_start,
    )
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _build_enclosing_chain_cell_infos,
    )
    from pydoppelgangerhunt.fixer._df_pairing import resolve_clone_pair_outputs
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _has_invalid_lexical_scope_declarations,
    )
    from pydoppelgangerhunt.fixer._df_with_provenance import (
        _SuppressProvenance,
        _build_module_suppress_seed,
        _update_suppress_provenance_from_expr_walrus,
    )
    from pydoppelgangerhunt.fixer.scope import ensure_unit_scope_closure_mode

    # 1. _evict_tree_scopes cleans _scope_stmts_by_start/_end and fingerprint detects in-place AST mutation
    mod = ast.parse("def f():\n    x = 1\n    return x\n")
    fn = mod.body[0]
    _ = inspect_single_unit_scope(
        {"start": 2, "end": 2, "kind": "basic_block", "source_code": "    x = 1\n"},
        tree=mod,
    )
    _evict_tree_scopes(mod)
    assert fn not in _scope_stmts_by_start
    assert fn not in _scope_stmts_by_end

    # Production mode (default): mutation refreshes cached result without raising.
    prod_mod = ast.parse("x = 1\n")
    prod_ck = _ScopeCacheKey(prod_mod, "fp_prod_test")
    assert _get_or_compute_scope_cached(_scope_cell_info_cache, prod_ck, lambda: ("p1",)) == ("p1",)
    prod_mod.body.append(ast.Pass())
    assert _get_or_compute_scope_cached(_scope_cell_info_cache, prod_ck, lambda: ("p2",)) == ("p2",)
    _evict_tree_scopes(prod_mod)

    # Debug mode: enabling before caching causes in-place mutation to raise RuntimeError.
    mut_mod = ast.parse("x = 1\n")
    ck = _ScopeCacheKey(mut_mod, "fp_test")
    _debug_ast_immutability[0] = True
    try:
        _get_or_compute_scope_cached(_scope_cell_info_cache, ck, lambda: ("v1",))
        mut_mod.body.append(ast.Pass())
        with pytest.raises(
            RuntimeError,
            match="Cached AST node was mutated in-place without calling evict_tree_scopes",
        ):
            _get_or_compute_scope_cached(_scope_cell_info_cache, ck, lambda: ("v2",))
    finally:
        _debug_ast_immutability[0] = False
        _evict_tree_scopes(mut_mod)
    assert _get_or_compute_scope_cached(
        _scope_cell_info_cache, ck, lambda: ("v2",)
    ) == ("v2",)

    # 2. ensure_unit_scope_closure_mode logs DEBUG and sets unavailable_source_for_closure_mode
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        failed_sc = ensure_unit_scope_closure_mode(
            {"name": "no_src_unit", "start": 2, "end": 3, "kind": "compound_block"},
            {"inputs": ["a"], "outputs": ["b"], "control_flow_hazards": []},
            False,
        )
    assert "unavailable_source_for_closure_mode" in failed_sc["control_flow_hazards"]
    assert any("failing closed" in rec.message for rec in caplog.records)
    res_pair = resolve_clone_pair_outputs(
        {"name": "u1", "start": 2, "end": 3, "kind": "compound_block"},
        {"name": "u2", "start": 2, "end": 3, "kind": "compound_block"},
        failed_sc,
        failed_sc,
        return_details=True,
    )
    assert res_pair.outputs is None
    assert res_pair.skip_reason == "unavailable source for closure mode"

    # 3. Single-method class does not synthesize a self-referential receiver sibling closure
    cls_src = (
        "class Worker:\n"
        "    def compute(self, x):\n"
        "        val = self.base + x\n"
        "        return val\n"
    )
    cls_tree = ast.parse(cls_src)
    chain = _find_enclosing_function_chain(cls_tree, 3, 3)
    infos = _build_enclosing_chain_cell_infos(
        chain, 3, 3, None, None, set(), mod_tree=cls_tree
    )
    assert len(infos) == 1
    assert "self" not in infos[0].sibling_closures

    # 4. Unrelated function with invalid nonlocal in same module does not taint valid function
    mixed_mod_src = (
        "def bad_other():\n"
        "    nonlocal missing_cell\n"
        "    missing_cell = 1\n\n"
        "def good_target(a):\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    mixed_tree = ast.parse(mixed_mod_src)
    good_chain = _find_enclosing_function_chain(mixed_tree, 6, 6)
    good_fn = mixed_tree.body[1]
    assert isinstance(good_fn, ast.FunctionDef)
    assert (
        _has_invalid_lexical_scope_declarations(
            [good_fn.body[0]],
            enclosing_chain=good_chain,
            mod_tree=mixed_tree,
            source_lines=mixed_mod_src.splitlines(),
        )
        is False
    )

    # 5. ast.Dict walrus order (k1, v1, k2, v2) and _LazyModuleStmtSeedMap O(1) table lookup
    dict_expr = ast.parse(
        "{(k1 := suppress): (v1 := 1), (k2 := v1): (k1 := open)}",
        mode="eval",
    ).body
    prov_dict = _update_suppress_provenance_from_expr_walrus(
        [dict_expr],
        _SuppressProvenance(
            suppressors=frozenset({"suppress"}),
            non_suppressors=frozenset({"open"}),
        ),
        allow_positive=True,
    )
    assert "k1" not in prov_dict.suppressors
    assert "k1" in prov_dict.non_suppressors

    seed_mod = ast.parse(
        "from contextlib import suppress as s1\n"
        "x = 1\n"
        "from contextlib import nullcontext as nc1\n"
    )
    seed_res = _build_module_suppress_seed(seed_mod)
    last_id = id(seed_mod.body[-1])
    assert "s1" in seed_res.prefix_by_top_stmt_id[last_id].suppressors
    assert seed_res.suffix_delta_by_top_stmt_id[last_id].suppressors == frozenset()


def test_register_tree_source_lines_refreshes_and_invalidates_on_changed_source_text() -> (
    None
):
    """Verifies re-registering the same AST tree with updated source text refreshes cached lines and evicts stale scope entries."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _get_cached_tree_source_lines,
        _make_trusted_scope_cache_key,
        _register_tree_source_lines,
        _scope_cell_info_cache,
        _store_lru_cache_entry,
        clear_dataflow_cache,
    )

    clear_dataflow_cache()
    src_v1 = "def f():\n    return 1\n"
    src_v2 = "def f():\n    return 2\n"
    tree = ast.parse(src_v1)

    lines_v1 = _register_tree_source_lines(tree, source_text=src_v1)
    assert lines_v1 == ("def f():", "    return 1")
    assert _get_cached_tree_source_lines(tree) == ("def f():", "    return 1")

    dummy_ck = _make_trusted_scope_cache_key(owner=tree, kind="tree_scope_intervals")
    _store_lru_cache_entry(_scope_cell_info_cache, dummy_ck, ("stale_intervals",), 128)
    assert dummy_ck in _scope_cell_info_cache

    # Re-registering with identical text preserves dependent cache entries
    assert _register_tree_source_lines(tree, source_text=src_v1) == (
        "def f():",
        "    return 1",
    )
    assert dummy_ck in _scope_cell_info_cache

    # Re-registering the same AST object with different source text updates lines and evicts stale entries
    lines_v2 = _register_tree_source_lines(tree, source_text=src_v2)
    assert lines_v2 == ("def f():", "    return 2")
    assert _get_cached_tree_source_lines(tree) == ("def f():", "    return 2")
    assert dummy_ck not in _scope_cell_info_cache


def test_evict_tree_scopes_purges_detached_child_owner_only_cache_entries() -> None:
    """Verifies evict_tree_scopes(root) evicts owner-only cache entries for AST children detached prior to eviction."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _make_trusted_scope_cache_key,
        _scope_cell_info_cache,
        _scope_owner_keys_index,
        _store_lru_cache_entry,
        clear_dataflow_cache,
        evict_tree_scopes,
    )

    clear_dataflow_cache()
    mod_src = (
        "def outer_fn(x):\n"
        "    def inner_fn(y):\n"
        "        return x + y\n"
        "    return inner_fn(1)\n"
        "def sibling_fn(z):\n"
        "    return z * 2\n"
    )
    mod_tree = ast.parse(mod_src)
    outer_fn = mod_tree.body[0]
    sibling_fn = mod_tree.body[1]
    assert isinstance(outer_fn, ast.FunctionDef)
    assert isinstance(sibling_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)

    # Populate scope caches for outer_fn, inner_fn (owner-only), and sibling_fn
    outer_ck = _make_trusted_scope_cache_key(owner=outer_fn, kind="cell_info")
    inner_owner_only_ck = _make_trusted_scope_cache_key(
        owner=inner_fn, kind="cell_info"
    )
    sibling_owner_only_ck = _make_trusted_scope_cache_key(
        owner=sibling_fn, kind="cell_info"
    )
    _store_lru_cache_entry(_scope_cell_info_cache, outer_ck, ("outer",), 128)
    _store_lru_cache_entry(_scope_cell_info_cache, inner_owner_only_ck, ("inner",), 128)
    _store_lru_cache_entry(
        _scope_cell_info_cache, sibling_owner_only_ck, ("sibling",), 128
    )
    assert inner_owner_only_ck in _scope_cell_info_cache
    assert sibling_owner_only_ck in _scope_cell_info_cache

    # Detach inner_fn from outer_fn.body BEFORE calling evict_tree_scopes(outer_fn)
    outer_fn.body.remove(inner_fn)
    evict_tree_scopes(outer_fn)

    assert inner_owner_only_ck not in _scope_cell_info_cache
    assert inner_fn not in _scope_owner_keys_index
    # Unrelated sibling_fn in mod_tree remains cached
    assert sibling_owner_only_ck in _scope_cell_info_cache

    # Now populate mod_tree and detach sibling_fn from mod_tree.body BEFORE calling evict_tree_scopes(mod_tree)
    mod_ck = _make_trusted_scope_cache_key(owner=mod_tree, kind="cell_info")
    _store_lru_cache_entry(_scope_cell_info_cache, mod_ck, ("mod",), 128)
    mod_tree.body.remove(sibling_fn)
    evict_tree_scopes(mod_tree)
    assert sibling_owner_only_ck not in _scope_cell_info_cache
    assert sibling_fn not in _scope_owner_keys_index


def test_interior_ast_mutation_invalidation_and_cache_lifecycle_stress() -> None:
    """Verifies evict_tree_scopes invalidates interior AST mutations and stress-tests cache index invariants."""
    import concurrent.futures
    import threading
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _downstream_cache_lock,
        _node_to_parent_map_roots,
        _parent_map_root_nodes,
        _scope_cell_info_cache,
        _scope_closures_cache,
        _scope_key_nodes_index,
        _scope_owner_keys_index,
        _scope_parent_maps,
        _store_lru_cache_entry,
        clear_dataflow_cache,
        evict_tree_scopes,
    )

    clear_dataflow_cache()

    # 1. Interior AST mutation (leaving first and last statement unchanged) + evict_tree_scopes(tree)
    src = (
        "def outer(x):\n"
        "    a = 0\n"
        "    mid = lambda: 1\n"
        "    a = x + 1\n"
        "    res = mid() + a\n"
        "    return res\n"
    )
    tree = ast.parse(src)
    unit_lines_4_5 = {
        "start": 4,
        "end": 5,
        "kind": "compound_block",
        "source_code": "    a = x + 1\n    res = mid() + a\n",
    }
    sc_before = inspect_single_unit_scope(unit_lines_4_5, tree=tree)
    assert "x" in sc_before["inputs"]
    assert "mid" in sc_before["inputs"]
    assert sc_before["is_control_flow_safe"] is True

    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    # Mutate interior statement (index 1, line 3) in-place so `mid` captures `a`
    # without changing first (line 2) or last (line 6) statement id/lineno.
    mid_assign = fn_node.body[1]
    assert isinstance(mid_assign, ast.Assign) and isinstance(
        mid_assign.value, ast.Lambda
    )
    mid_assign.value.body = ast.Name(id="a", ctx=ast.Load())
    ast.fix_missing_locations(tree)
    evict_tree_scopes(tree)

    sc_after = inspect_single_unit_scope(unit_lines_4_5, tree=tree)
    assert sc_after["is_control_flow_safe"] is False
    assert "severed_closure_cell" in sc_after["control_flow_hazards"]

    # 2. Cache lifecycle stress scenarios:
    # (a) Replacing an existing key with a different value
    # (b) Deleting an entry shared by multiple AST owners (owner + aux_ast_nodes)
    # (c) Evicting a parent-map root while another root still references shared child nodes
    # (d) Clearing one tracked cache while the other retains entries
    # (e) Repeated eviction of already-evicted trees
    clear_dataflow_cache()
    t1 = ast.parse("def f1(a):\n    return a + 1\n")
    t2 = ast.parse("def f2(b):\n    return b + 2\n")
    f1_node = t1.body[0]
    f2_node = t2.body[0]
    shared_key = _ScopeCacheKey(
        owner=f1_node,
        kind="stress_shared",
        params=("v1",),
        aux_ast_nodes=(t1, f2_node),
    )
    _store_lru_cache_entry(_scope_closures_cache, shared_key, ("val1",), 128)
    _store_lru_cache_entry(_scope_cell_info_cache, shared_key, ("info1",), 128)
    # Overwrite with new value
    with _downstream_cache_lock:
        _scope_closures_cache[shared_key] = ("val2",)
    # Clear one tracked cache while the other retains shared_key
    with _downstream_cache_lock:
        _scope_closures_cache.clear()
        assert shared_key in _scope_cell_info_cache
        assert shared_key in _scope_key_nodes_index
        assert f1_node in _scope_owner_keys_index
        assert f2_node in _scope_owner_keys_index
    # Delete from the second cache and verify reverse indexes are clean
    with _downstream_cache_lock:
        del _scope_cell_info_cache[shared_key]
        assert shared_key not in _scope_key_nodes_index
        assert shared_key not in _scope_owner_keys_index.get(f1_node, set())
        assert shared_key not in _scope_owner_keys_index.get(f2_node, set())

    # Evicting a parent-map root while another root (e.g. function scope vs module scope) shares child nodes
    inspect_single_unit_scope({"start": 1, "end": 2, "kind": "function"}, tree=t1)
    _store_lru_cache_entry(
        _scope_cell_info_cache,
        _ScopeCacheKey(owner=f1_node, kind="fn_scope_ctx"),
        ("ctx",),
        128,
    )
    evict_tree_scopes(f1_node)
    # Repeated eviction on already-evicted subtree/tree is a safe no-op
    evict_tree_scopes(f1_node)
    evict_tree_scopes(t1)
    evict_tree_scopes(t1)

    # (f) & (g) Concurrent insertion, lookup, and synchronized AST mutation + eviction across worker threads
    trees = [
        ast.parse(f"def worker_{i}(x):\n    y = x + {i}\n    z = y * 2\n    return z\n")
        for i in range(6)
    ]
    tree_locks = [threading.Lock() for _ in trees]

    def _stress_worker(tid: int) -> None:
        for step in range(20):
            idx = (tid + step) % len(trees)
            with tree_locks[idx]:
                cur_tree = trees[idx]
                if step % 5 == 0:
                    fn = cur_tree.body[0]
                    if isinstance(fn, ast.FunctionDef) and isinstance(
                        fn.body[1], ast.Assign
                    ):
                        fn.body[1].value = ast.BinOp(
                            left=ast.Name(id="y", ctx=ast.Load()),
                            op=ast.Mult(),
                            right=ast.Constant(value=step + 1),
                        )
                        ast.fix_missing_locations(cur_tree)
                    evict_tree_scopes(cur_tree)
                sc = inspect_single_unit_scope(
                    {
                        "start": 2,
                        "end": 2,
                        "kind": "basic_block",
                        "source_code": f"    y = x + {idx}\n",
                    },
                    tree=cur_tree,
                )
                assert "x" in sc["inputs"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_stress_worker, i) for i in range(6)]
        for fut in futs:
            fut.result()

    with _downstream_cache_lock:
        for k, nodes in _scope_key_nodes_index.items():
            assert k in _scope_closures_cache or k in _scope_cell_info_cache
            for n in nodes:
                assert k in _scope_owner_keys_index.get(n, set())
        for root, rnodes in _parent_map_root_nodes.items():
            assert root in _scope_parent_maps
            for rn in rnodes:
                assert root in _node_to_parent_map_roots.get(rn, set())


def test_oversized_module_parent_map_built_once_across_repeated_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies ASTs exceeding _MAX_PARENT_MAP_TOTAL_NODES build their parent map once rather than per lookup."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer import evict_tree_scopes
    from pydoppelgangerhunt.fixer._df_cell_carriers import _is_node_attached_to_tree

    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 40)
    clear_dataflow_cache()

    lines = ["GLOBAL_BASE = 1\n"]
    for idx in range(12):
        lines.append(
            f"def worker_{idx}(items):\n"
            f"    acc = {idx}\n"
            f"    for v in items:\n"
            f"        acc = acc + v\n"
            f"    return acc\n"
        )
    src = "".join(lines)
    big_tree = ast.parse(src)
    fn0 = big_tree.body[1]
    assert isinstance(fn0, ast.FunctionDef)
    inner_stmt = fn0.body[1]
    detached_stmt = ast.parse("orphan = 99\n").body[0]

    build_calls = 0
    orig_build = df_cache_mod._build_parent_map_unlocked

    def _counting_build(owner: ast.AST) -> dict[ast.AST, ast.AST]:
        nonlocal build_calls
        if owner is big_tree:
            build_calls += 1
        return orig_build(owner)

    monkeypatch.setattr(df_cache_mod, "_build_parent_map_unlocked", _counting_build)

    for idx in range(6):
        base_line = 2 + idx * 5
        sc = inspect_single_unit_scope(
            {
                "start": base_line + 1,
                "end": base_line + 3,
                "kind": "compound_block",
                "source_text": src,
            },
            tree=big_tree,
        )
        assert sc["is_control_flow_safe"] is True

    assert _is_node_attached_to_tree(fn0, big_tree) is True
    assert _is_node_attached_to_tree(inner_stmt, big_tree) is True
    assert _is_node_attached_to_tree(detached_stmt, big_tree) is False
    assert df_cache_mod._tree_contains_scope_node(big_tree, fn0) is True
    assert df_cache_mod._tree_contains_scope_node(big_tree, detached_stmt) is False
    anc_map = df_cache_mod._get_scope_ancestor_map(big_tree)
    assert anc_map.get(fn0) is big_tree
    assert df_cache_mod._get_cached_tree_source_lines(inner_stmt) == tuple(
        src.splitlines()
    )

    assert big_tree not in df_cache_mod._scope_parent_maps
    assert build_calls == 1

    big_tree.body.append(ast.Pass())
    evict_tree_scopes(big_tree)
    assert df_cache_mod._tree_contains_scope_node(big_tree, fn0) is True
    assert build_calls == 2


def test_ephemeral_unit_ast_unified_tracking_and_non_weakref_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies ephemeral unit ASTs never receive ad hoc attributes and non-weakrefable ASTs still bypass caches."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _collect_escaped_and_executed_intra_unit_defs,
        _compute_scope_ctx_aux_nodes,
        _is_node_attached_to_tree,
    )

    src = (
        "def outer(items):\n"
        "    scale = 2\n"
        "    def inner(v):\n"
        "        return v * scale\n"
        "    res = inner(3)\n"
        "    return res\n"
    )
    mod = ast.parse(src)
    fn_outer = mod.body[0]
    assert isinstance(fn_outer, ast.FunctionDef)
    candidate_stmts = fn_outer.body[1:3]

    seen_marked: list[ast.AST] = []
    orig_mark = df_cache_mod._mark_ephemeral_unit_ast

    def _spy_mark(node: ast.AST) -> ast.AST:
        seen_marked.append(node)
        return orig_mark(node)

    monkeypatch.setattr(
        "pydoppelgangerhunt.fixer._df_cell_carriers._mark_ephemeral_unit_ast",
        _spy_mark,
    )
    _collect_escaped_and_executed_intra_unit_defs(
        candidate_stmts,
        frozenset(),
    )
    assert len(seen_marked) == 1
    ephem_unit_mod = seen_marked[0]
    assert df_cache_mod._is_ephemeral_unit_ast(ephem_unit_mod) is True
    for sub in ast.walk(ephem_unit_mod):
        assert not hasattr(sub, "_is_ephemeral_unit_mod")

    # Simulate a custom AST node whose weakref registration raises TypeError
    custom_mod = ast.parse("z = 1\n")

    class _NoWeakRefSet:
        def add(self, _item: Any) -> None:
            raise TypeError("cannot create weak reference")

        def __contains__(self, _item: object) -> bool:
            raise TypeError("cannot create weak reference")

        def clear(self) -> None:
            pass

    monkeypatch.setattr(df_cache_mod, "_ephemeral_unit_asts", _NoWeakRefSet())
    df_cache_mod._mark_ephemeral_unit_ast(custom_mod)
    assert df_cache_mod._is_ephemeral_unit_ast(custom_mod) is True
    assert _compute_scope_ctx_aux_nodes(custom_mod, custom_mod, (), False) == ()
    assert _compute_scope_ctx_aux_nodes(
        custom_mod.body[0], custom_mod, (), False
    ) == (custom_mod.body[0],)
    assert _is_node_attached_to_tree(custom_mod.body[0], custom_mod) is True
    assert custom_mod not in df_cache_mod._scope_parent_maps


def test_interior_ast_mutation_detected_without_header_change() -> None:
    """Verifies mutating an interior expression inside a cached AST refreshes in prod mode and raises RuntimeError in debug mode."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _debug_ast_immutability,
        evict_tree_scopes,
    )

    src = (
        "def compute(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    unit = {
        "start": 2,
        "end": 3,
        "kind": "compound_block",
        "source_text": src,
        "source_code": "    a = x + 1\n    b = a * 2\n",
    }

    prod_tree = ast.parse(src)
    assert inspect_single_unit_scope(unit, tree=prod_tree)["is_control_flow_safe"] is True
    prod_fn = prod_tree.body[0]
    assert isinstance(prod_fn, ast.FunctionDef)
    prod_assign = prod_fn.body[1]
    assert isinstance(prod_assign, ast.Assign) and isinstance(prod_assign.value, ast.BinOp)
    assert isinstance(prod_assign.value.right, ast.Constant)
    prod_assign.value.right.value = 99
    assert inspect_single_unit_scope(unit, tree=prod_tree)["is_control_flow_safe"] is True
    evict_tree_scopes(prod_tree)

    tree = ast.parse(src)
    prev_dbg = _debug_ast_immutability[0]
    _debug_ast_immutability[0] = True
    try:
        sc1 = inspect_single_unit_scope(unit, tree=tree)
        assert sc1["is_control_flow_safe"] is True

        # Mutate deep interior constant in middle statement without altering header or first/last stmt identity
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        mid_assign = fn.body[1]
        assert isinstance(mid_assign, ast.Assign) and isinstance(mid_assign.value, ast.BinOp)
        assert isinstance(mid_assign.value.right, ast.Constant)
        mid_assign.value.right.value = 99

        with pytest.raises(RuntimeError, match="Cached AST node was mutated in-place"):
            inspect_single_unit_scope(unit, tree=tree)
    finally:
        _debug_ast_immutability[0] = prev_dbg
        evict_tree_scopes(tree)

    evict_tree_scopes(tree)
    unit_updated = {
        **unit,
        "source_text": "def compute(x):\n    a = x + 1\n    b = a * 99\n    return b\n",
        "source_code": "    a = x + 1\n    b = a * 99\n",
    }
    sc2 = inspect_single_unit_scope(unit_updated, tree=tree)
    assert sc2["is_control_flow_safe"] is True


def test_evict_tree_scopes_detached_child_and_unrelated_preservation() -> None:
    """Verifies evict_tree_scopes evicts detached child scopes and with-summaries while preserving unrelated trees."""
    from pydoppelgangerhunt.fixer._df_cache import (
        _ScopeCacheKey,
        _get_scope_parent_map,
        _scope_cell_info_cache,
        _stmt_with_summary_cache,
        evict_tree_scopes,
    )

    clear_dataflow_cache()
    target_tree = ast.parse(
        "def outer():\n"
        "    with mgr():\n"
        "        def inner():\n"
        "            return 1\n"
        "    return inner\n"
    )
    other_tree = ast.parse(
        "def unrelated():\n"
        "    with other_mgr():\n"
        "        x = 2\n"
        "    return x\n"
    )
    _get_scope_parent_map(target_tree)
    _get_scope_parent_map(other_tree)
    outer_fn = target_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    with_target = outer_fn.body[0]
    assert isinstance(with_target, ast.With)
    inner_fn = with_target.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)

    other_fn = other_tree.body[0]
    assert isinstance(other_fn, ast.FunctionDef)
    with_other = other_fn.body[0]
    assert isinstance(with_other, ast.With)

    target_ck = _ScopeCacheKey(inner_fn, "detached_child_test")
    other_ck = _ScopeCacheKey(other_fn, "unrelated_tree_test")
    _scope_cell_info_cache[target_ck] = ("child_val",)
    _scope_cell_info_cache[other_ck] = ("other_val",)
    _stmt_with_summary_cache[with_target] = (False, frozenset())
    _stmt_with_summary_cache[with_other] = (False, frozenset())

    # Detach inner_fn and with_target from outer_fn before calling evict_tree_scopes
    outer_fn.body = [ast.Return(value=ast.Constant(value=0), lineno=2, col_offset=4)]
    evict_tree_scopes(target_tree)

    assert target_ck not in _scope_cell_info_cache
    assert with_target not in _stmt_with_summary_cache
    assert other_ck in _scope_cell_info_cache
    assert with_other in _stmt_with_summary_cache


def test_oversized_scope_context_unpins_parent_map_and_weakref_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies oversized _FnScopeContext leaves _stored_parent_map None and _OversizedParentEntry uses weakrefs."""
    import gc
    import weakref
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context

    clear_dataflow_cache()
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 4)
    mod = ast.parse(
        "def big_fn(x):\n"
        "    a = x + 1\n"
        "    b = a + 2\n"
        "    return b\n"
    )
    fn = mod.body[0]
    assert isinstance(fn, ast.FunctionDef)
    ctx = _get_fn_scope_context(fn, mod_tree=mod)
    assert ctx._stored_parent_map is None
    assert ctx.parent_map.get(fn.body[0]) is fn

    entry = df_cache_mod._oversized_scope_parent_maps[mod]
    assert not hasattr(entry, "attached_nodes")
    assert isinstance(entry.weak_parent_map, df_cache_mod._WeakParentMap)
    assert fn.body[0] in entry.weak_parent_map
    assert df_cache_mod._oversized_entry_contains_node(mod, entry, fn.body[0]) is True
    assert df_cache_mod._oversized_entry_contains_node(mod, entry, ast.Pass()) is False
    detached_stmt = fn.body.pop(1)
    assert detached_stmt in entry.weak_parent_map
    stmt_ref = weakref.ref(detached_stmt)
    del detached_stmt
    gc.collect()
    assert stmt_ref() is None



def test_ephemeral_unit_ast_context_handles_non_weakrefable_custom_ast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _ephemeral_unit_ast_context refcounts non-weakrefable ASTs and unregisters on exit."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    class _NoWeakRefSet:
        def add(self, _item: Any) -> None:
            raise TypeError("cannot create weak reference")

        def __contains__(self, _item: object) -> bool:
            raise TypeError("cannot create weak reference")

        def clear(self) -> None:
            pass

    monkeypatch.setattr(df_cache_mod, "_ephemeral_unit_asts", _NoWeakRefSet())
    ephem_mod = ast.Module(body=[], type_ignores=[])
    assert df_cache_mod._is_ephemeral_unit_ast(ephem_mod) is False
    with df_cache_mod._ephemeral_unit_ast_context(ephem_mod):
        assert df_cache_mod._is_ephemeral_unit_ast(ephem_mod) is True
        with df_cache_mod._ephemeral_unit_ast_context(ephem_mod):
            assert df_cache_mod._is_ephemeral_unit_ast(ephem_mod) is True
        assert df_cache_mod._is_ephemeral_unit_ast(ephem_mod) is True
    assert df_cache_mod._is_ephemeral_unit_ast(ephem_mod) is False


def test_span_line_index_has_closure_overlap_bisect_and_missing_end_fallback() -> None:
    """Verifies _SpanLineIndex.has_closure_overlap interval bisect and missing end_lineno fallback."""
    from pydoppelgangerhunt.fixer._df_span_index import _get_span_line_index

    src = (
        "def outer():\n"
        "    a = 1\n"
        "    f1 = lambda: a\n"
        "    b = 2\n"
        "    def f2():\n"
        "        return b\n"
        "    c = 3\n"
    )
    tree = ast.parse(src)
    outer_fn = tree.body[0]
    span_idx = _get_span_line_index(outer_fn)
    assert span_idx.has_closure_overlap(2, 2) == (False, False)
    assert span_idx.has_closure_overlap(3, 3) == (False, True)
    assert span_idx.has_closure_overlap(4, 4) == (False, False)
    assert span_idx.has_closure_overlap(5, 6) == (False, True)
    assert span_idx.has_closure_overlap(7, 7) == (False, False)

    # Synthetic closure missing end_lineno falls back conservatively when c_start <= u_end
    synth_fn = ast.FunctionDef(
        name="outer_synth",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[
            ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=ast.Constant(value=1),
                lineno=5,
                end_lineno=None,
                col_offset=4,
            )  # type: ignore[list-item]
        ],
        decorator_list=[],
        lineno=1,
        end_lineno=10,
        col_offset=0,
    )
    synth_idx = _get_span_line_index(synth_fn)
    assert synth_idx.has_closure_overlap(2, 4).has_missing_end is False
    assert synth_idx.has_closure_overlap(5, 8).has_missing_end is True


def test_oversized_ast_get_scope_parent_map_single_build_and_detached_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies oversized module parent map builds once across all _get_scope_parent_map callers and allows GC."""
    import gc
    import weakref
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures_uncached,
        _resolve_enclosing_scopes_by_coords,
    )
    from pydoppelgangerhunt.fixer.scope import (
        _unit_source_code_matches_tree,
        inspect_single_unit_scope,
    )

    clear_dataflow_cache()
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 6)
    orig_build = df_cache_mod._build_parent_map_unlocked
    root_builds: list[ast.AST] = []

    def _counting_build(node: ast.AST) -> dict[ast.AST, ast.AST]:
        root_builds.append(node)
        return orig_build(node)

    monkeypatch.setattr(df_cache_mod, "_build_parent_map_unlocked", _counting_build)
    src = (
        "def big_outer(items):\n"
        "    scale = 2\n"
        "    cb = lambda v: v * scale\n"
        "    out = [cb(x) for x in items]\n"
        "    extra = len(out) + 1\n"
        "    return out, extra\n"
    )
    mod = ast.parse(src)
    fn_node = mod.body[0]
    assert isinstance(fn_node, ast.FunctionDef)

    for _ in range(3):
        pmap_mod = df_cache_mod._get_scope_parent_map(mod)
        pmap_fn = df_cache_mod._get_scope_parent_map(fn_node)
        assert pmap_mod.get(fn_node) is mod
        assert pmap_fn.get(fn_node.body[0]) is fn_node
        _collect_scope_closures_uncached(mod)
        _resolve_enclosing_scopes_by_coords(fn_node, mod)
        assert _unit_source_code_matches_tree(
            {"start": 4, "end": 4, "kind": "basic_block"},
            "    out = [cb(x) for x in items]\n",
            mod,
        )
        sc = inspect_single_unit_scope(
            {
                "start": 4,
                "end": 4,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "    out = [cb(x) for x in items]\n",
            },
            tree=mod,
        )
        assert sc["is_control_flow_safe"] is True

    assert root_builds.count(mod) == 1

    # Verify _WeakParentMap does not retain detached AST child nodes while the oversized root stays cached
    mod_detach = ast.parse(src)
    fn_detach = mod_detach.body[0]
    assert isinstance(fn_detach, ast.FunctionDef)
    ephem_stmt = ast.parse("temp_detached = 1\n").body[0]
    fn_detach.body.insert(0, ephem_stmt)
    pmap_detach = df_cache_mod._get_scope_parent_map(mod_detach)
    assert pmap_detach.get(ephem_stmt) is fn_detach
    detached = fn_detach.body.pop(0)
    detached_ref = weakref.ref(detached)
    del ephem_stmt, detached
    gc.collect()
    assert detached_ref() is None


class TestProductionAstImmutabilityMode:
    """Exercises production AST immutability fingerprinting with _debug_ast_immutability disabled."""

    def test_production_mode_detects_statement_interval_mutation_past_bfs_prefix(self) -> None:
        """Verifies production fingerprinting hashes all body statement intervals past the 512-node BFS cap."""
        import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
        from pydoppelgangerhunt.fixer._df_span_index import _get_span_line_index

        clear_dataflow_cache()
        # Each statement has ~15 AST nodes; 50 statements exceed the 512-node production BFS prefix,
        # and statement index 43 is also non-aligned with 32-stride sampling.
        lines = [f"v_{idx} = (a + {idx}) * (b + {idx}) + (c - {idx})\n" for idx in range(50)]
        mod = ast.parse("".join(lines))
        assert sum(1 for _ in ast.walk(mod)) > df_cache_mod._MAX_PROD_FINGERPRINT_NODES

        prev_debug = df_cache_mod._debug_ast_immutability[0]
        df_cache_mod._debug_ast_immutability[0] = False
        try:
            idx_before = _get_span_line_index(mod)
            assert 44 in idx_before.stmts_by_start_line

            target_stmt = mod.body[43]
            target_stmt.lineno = 950
            target_stmt.end_lineno = 950

            # In production mode, mutation of immediate statement intervals silently evicts and rebuilds
            idx_after = _get_span_line_index(mod)
            assert idx_after is not idx_before
            assert 950 in idx_after.stmts_by_start_line
        finally:
            df_cache_mod._debug_ast_immutability[0] = prev_debug

    def test_production_mode_deep_expr_mutation_requires_evict_tree_scopes(self) -> None:
        """Verifies evict_tree_scopes refreshes caches when a deep constant beyond the 512-node prefix is mutated."""
        import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
        from pydoppelgangerhunt.fixer._df_span_index import _get_span_line_index
        from pydoppelgangerhunt.fixer import evict_tree_scopes

        clear_dataflow_cache()
        lines = [f"v_{idx} = (a + {idx}) * (b + {idx}) + (c - {idx})\n" for idx in range(50)]
        mod = ast.parse("".join(lines))

        prev_debug = df_cache_mod._debug_ast_immutability[0]
        df_cache_mod._debug_ast_immutability[0] = False
        try:
            idx_cached = _get_span_line_index(mod)
            tail_assign = mod.body[48]
            assert isinstance(tail_assign, ast.Assign)
            assert isinstance(tail_assign.value, ast.BinOp)
            assert isinstance(tail_assign.value.right, ast.BinOp)
            tail_assign.value.right.right = ast.Constant(value=999999, lineno=49, col_offset=30)

            evict_tree_scopes(mod)
            idx_refreshed = _get_span_line_index(mod)
            assert idx_refreshed is not idx_cached
        finally:
            df_cache_mod._debug_ast_immutability[0] = prev_debug


def test_tree_raw_source_cache_weakref_and_lru_eviction_releases_trees() -> None:
    """Verifies caller-supplied AST trees and raw source strings are not pinned by _tree_raw_source_cache."""
    import gc
    import weakref
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer import evict_tree_scopes
    from pydoppelgangerhunt.fixer.scope import inspect_single_unit_scope

    class _TrackableSourceStr(str):
        """Weakly-referenceable str subclass to verify raw source text is not retained."""

    clear_dataflow_cache()

    # 1. Raw source string is never retained by _tree_raw_source_cache even while tree remains cached
    tracked_src = _TrackableSourceStr("def tracked_fn(x):\n    y = x + 99\n    return y\n")
    src_ref = weakref.ref(tracked_src)
    pinned_tree = ast.parse(tracked_src)
    pinned_tree_ref = weakref.ref(pinned_tree)
    inspect_single_unit_scope(
        {"start": 2, "end": 2, "kind": "basic_block", "source_text": tracked_src},
        tree=pinned_tree,
    )
    del tracked_src
    gc.collect()
    assert src_ref() is None
    assert pinned_tree in df_cache_mod._tree_raw_source_cache

    evict_tree_scopes(pinned_tree)
    assert pinned_tree not in df_cache_mod._tree_raw_source_cache
    del pinned_tree
    gc.collect()
    assert pinned_tree_ref() is None

    # 2. LRU eviction of tree_source_lines pops _tree_raw_source_cache and releases unreferenced trees
    clear_dataflow_cache()
    cap = df_cache_mod._SCOPE_CACHE_KIND_CAPS["tree_source_lines"]
    reg_refs: list[weakref.ReferenceType[ast.Module]] = []
    seen_populated = False
    for idx in range(cap + 16):
        src = f"def fn_{idx}(x):\n    y = x + {idx}\n    return y\n"
        caller_tree = ast.parse(src)
        reg_refs.append(weakref.ref(caller_tree))
        df_cache_mod._register_tree_source_lines(caller_tree, source_text=src)
        if len(df_cache_mod._tree_raw_source_cache) > 0:
            seen_populated = True
        del caller_tree

    gc.collect()
    assert seen_populated
    assert 0 < len(df_cache_mod._tree_raw_source_cache) <= cap
    assert reg_refs[0]() is None

    # Negative control: bare str/bytes for source_lines must fail loudly
    bad_tree = ast.parse("def bad_arg():\n    return 1\n")
    for raw_bad in ("def bad_arg():\n    return 1\n", b"def bad_arg():\n    return 1\n"):
        with pytest.raises(TypeError, match="source_lines"):
            df_cache_mod._register_tree_source_lines(bad_tree, raw_bad)  # type: ignore[arg-type]


def test_batch_entrypoints_outer_syntax_warning_guard_preserves_warning_registry(
    tmp_path: Path,
) -> None:
    """Verifies batch scope/synthesis/patch entrypoints preserve warnings.filters and unrelated __warningregistry__."""
    import warnings
    from pydoppelgangerhunt.fixer import (
        generate_refactoring_patch,
        inspect_single_unit_scope,
        synthesize_shared_helper_code,
    )

    src = (
        "def alpha(x):\n"
        "    val = x + 1\n"
        "    return val * 2\n\n"
        "def beta(x):\n"
        "    val = x + 1\n"
        "    return val * 2\n"
    )
    mod_file = tmp_path / "batch_warn.py"
    mod_file.write_text(src, encoding="utf-8")
    u1 = {"name": "alpha", "file": str(mod_file), "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "beta", "file": str(mod_file), "start": 5, "end": 7, "kind": "function"}

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        pre_filters = list(warnings.filters)
        sentinel_reg: dict[Any, int] = {("external warning", UserWarning, 10): 1}
        globals()["__warningregistry__"] = sentinel_reg
        try:
            inspect_single_unit_scope(u1, repo_root=str(tmp_path))
            helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
            patch = generate_refactoring_patch(
                [(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True
            )
            assert "def _shared_" in helper
            assert "def _shared_" in patch
            assert list(warnings.filters) == pre_filters
            assert sentinel_reg == {("external warning", UserWarning, 10): 1}
        finally:
            globals().pop("__warningregistry__", None)


def test_fingerprint_horizon_deep_ast_mutation_in_production_and_debug_modes() -> None:
    """Verifies deep interior AST mutations beyond node 512 require evict_tree_scopes in prod and raise RuntimeError in debug mode."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
    )

    clear_dataflow_cache()
    filler_lines = [f"        step_{i} = {i}\n" for i in range(300)]
    src = (
        "def outer(obj):\n"
        "    first_var = 1\n"
        "    second_var = 2\n"
        "    def inner():\n"
        + "".join(filler_lines)
        + "        return obj.attribute(first_var)\n"
        "    return inner\n"
    )

    # 1. Production mode (512-node BFS horizon): deep mutation is outside fingerprint until evict_tree_scopes(tree)
    prod_tree = ast.parse(src)
    prod_outer = prod_tree.body[0]
    assert isinstance(prod_outer, ast.FunctionDef)
    prod_inner = prod_outer.body[2]
    assert isinstance(prod_inner, ast.FunctionDef)
    prod_ret = prod_inner.body[-1]
    assert isinstance(prod_ret, ast.Return) and isinstance(prod_ret.value, ast.Call)
    assert isinstance(prod_ret.value.func, ast.Attribute)
    assert isinstance(prod_ret.value.args[0], ast.Name)

    prev_debug = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = False
    try:
        initial_reads = _extract_cached_scope_free_reads(prod_inner)
        assert initial_reads == frozenset({"obj", "first_var"})
        fp_before = df_cache_mod._compute_scope_fingerprint(prod_inner)

        # Mutate deep interior Attribute and free-variable Name beyond the 512-node BFS horizon
        prod_ret.value.func.attr = "other_attribute"
        prod_ret.value.args[0].id = "second_var"
        fp_after_prod = df_cache_mod._compute_scope_fingerprint(prod_inner)
        assert fp_after_prod == fp_before
        assert _extract_cached_scope_free_reads(prod_inner) == frozenset(
            {"obj", "first_var"}
        )

        # Explicit eviction refreshes the cached result to reflect the mutated AST
        df_cache_mod.evict_tree_scopes(prod_tree)
        assert _extract_cached_scope_free_reads(prod_inner) == frozenset(
            {"obj", "second_var"}
        )

        # 2. Debug immutability mode (65,536-node BFS horizon): deep `obj.attribute` -> `obj.other_attribute` raises RuntimeError
        clear_dataflow_cache()
        df_cache_mod._debug_ast_immutability[0] = True
        dbg_tree = ast.parse(src)
        dbg_outer = dbg_tree.body[0]
        assert isinstance(dbg_outer, ast.FunctionDef)
        dbg_inner = dbg_outer.body[2]
        assert isinstance(dbg_inner, ast.FunctionDef)
        dbg_ret = dbg_inner.body[-1]
        assert isinstance(dbg_ret, ast.Return) and isinstance(dbg_ret.value, ast.Call)
        assert isinstance(dbg_ret.value.func, ast.Attribute)

        assert _extract_cached_scope_free_reads(dbg_inner) == frozenset(
            {"obj", "first_var"}
        )
        dbg_ret.value.func.attr = "other_attribute"
        with pytest.raises(
            RuntimeError,
            match="mutated in-place without calling evict_tree_scopes",
        ):
            _extract_cached_scope_free_reads(dbg_inner)

        df_cache_mod.evict_tree_scopes(dbg_tree)
        assert _extract_cached_scope_free_reads(dbg_inner) == frozenset(
            {"obj", "first_var"}
        )
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_debug


def test_evict_tree_scopes_purges_all_eight_cache_categories_selectively() -> None:
    """Verifies evict_tree_scopes(tree_a) purges all 8 cache categories for tree_a while preserving tree_b."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import _get_fn_scope_context
    from pydoppelgangerhunt.fixer._df_cell_traversal import (
        _analyze_enclosing_fn_cells,
        _get_stmt_with_summary,
    )
    from pydoppelgangerhunt.fixer._df_closures import (
        _collect_scope_closures,
        _get_tree_expr_scope_intervals,
        _get_tree_scope_intervals,
    )
    from pydoppelgangerhunt.fixer._df_hazards import (
        _enclosing_try_reads_outputs,
        collect_downstream_read_names,
    )
    from pydoppelgangerhunt.fixer._df_span_index import (
        _find_enclosing_function_chain,
        _get_span_line_index,
    )
    from pydoppelgangerhunt.fixer.scope import _get_tree_sha256_digest

    clear_dataflow_cache()

    def _build_rich_module_source(tag: str) -> str:
        return (
            "import contextlib\n"
            f"def workflow_{tag}(items):\n"
            "    seed = 1\n"
            "    reader = lambda: seed\n"
            "    comp = [x + seed for x in items]\n"
            "    def factory():\n"
            "        return reader\n"
            "    for item in items:\n"
            "        try:\n"
            "            with contextlib.nullcontext():\n"
            "                step = item + seed\n"
            "        except ValueError:\n"
            "            step = 0\n"
            "    return factory(), comp, step\n"
        )

    src_a = _build_rich_module_source("alpha")
    src_b = _build_rich_module_source("beta")
    tree_a = df_cache_mod._get_or_parse_tree(src_a)
    tree_b = df_cache_mod._get_or_parse_tree(src_b)
    assert isinstance(tree_a, ast.Module) and isinstance(tree_b, ast.Module)
    fn_a = tree_a.body[1]
    fn_b = tree_b.body[1]
    assert isinstance(fn_a, ast.FunctionDef) and isinstance(fn_b, ast.FunctionDef)

    for src, tree, fn in ((src_a, tree_a, fn_a), (src_b, tree_b, fn_b)):
        # Categories 1 & 2: scope intervals + expression-scope intervals
        assert _get_tree_scope_intervals(tree)
        assert _get_tree_expr_scope_intervals(tree)
        # Category 3: parent maps + children maps + reverse root index
        assert df_cache_mod._get_scope_parent_map(tree)
        # Category 4: enclosing-scope index, span-line index, stmts_by_start/end
        assert _find_enclosing_function_chain(tree, 11, 11)
        assert _get_span_line_index(fn)
        assert df_cache_mod._get_scope_stmts_by_start_lineno(fn)
        assert df_cache_mod._get_scope_stmts_by_end_lineno(fn)
        # Category 5: function-scope contexts
        assert _get_fn_scope_context(fn, tree)
        # Category 6: cell analysis & scope closures
        assert _collect_scope_closures(fn, tree)
        assert _analyze_enclosing_fn_cells(fn, (11, 11, None, None), mod_tree=tree)
        inspect_single_unit_scope(
            {
                "start": 11,
                "end": 11,
                "kind": "basic_block",
                "source_text": src,
                "source_code": "                step = item + seed\n",
            },
            tree=tree,
        )
        # Category 7: statement summaries, loop cache, try/with cache
        for sub in ast.walk(fn):
            if isinstance(sub, ast.With):
                assert _get_stmt_with_summary(sub)[0] is True
        collect_downstream_read_names(
            src, {"start": 11, "end": 11}, candidates={"step"}, tree=tree
        )
        _enclosing_try_reads_outputs(fn, 11, 11, {"step"})
        # Category 8: source-line registrations, raw source digest, tree sha256, parsed tree
        assert df_cache_mod._get_cached_tree_source_lines(tree)
        assert _get_tree_sha256_digest(tree)

    nodes_a = set(ast.walk(tree_a))
    nodes_b = set(ast.walk(tree_b))

    def _present_in_all_eight_categories(
        target_tree: ast.AST,
        target_fn: ast.FunctionDef,
        target_nodes: set[ast.AST],
    ) -> dict[str, bool]:
        cell_kinds = {
            k.kind
            for k in df_cache_mod._scope_cell_info_cache
            if isinstance(k, df_cache_mod._ScopeCacheKey) and k.owner in target_nodes
        }
        return {
            "1_scope_intervals": "tree_scope_intervals" in cell_kinds,
            "2_expr_scope_intervals": "tree_expr_scope_intervals" in cell_kinds,
            "3_parent_maps": (
                target_tree in df_cache_mod._scope_parent_maps
                and target_tree in df_cache_mod._scope_children_maps
                and bool(df_cache_mod._node_to_parent_map_roots.get(target_fn))
            ),
            "4_enclosing_and_stmt_indexes": (
                "tree_enclosing_index" in cell_kinds
                and "span_line_index" in cell_kinds
                and target_fn in df_cache_mod._scope_stmts_by_start
                and target_fn in df_cache_mod._scope_stmts_by_end
            ),
            "5_fn_scope_contexts": "fn_scope_ctx" in cell_kinds,
            "6_cell_and_closure_caches": (
                bool(
                    {
                        "enclosing_fn_cells",
                        "stmt_deltas",
                        "free_reads",
                        "is_closure_factory",
                        "suppress_seed",
                    }
                    <= cell_kinds
                )
                and any(
                    df_cache_mod._key_references_scope_nodes(k, target_nodes)
                    for k in df_cache_mod._scope_closures_cache
                )
            ),
            "7_stmt_loop_try_with_summaries": (
                any(n in df_cache_mod._stmt_with_summary_cache for n in target_nodes if isinstance(n, ast.stmt))
                and target_fn in df_cache_mod._scope_loops_cache
                and target_fn in df_cache_mod._scope_try_with_cache
            ),
            "8_source_lines_and_parsed_trees": (
                "tree_source_lines" in cell_kinds
                and "tree_sha256_digest" in cell_kinds
                and target_tree in df_cache_mod._tree_raw_source_cache
                and any(pv in target_nodes for pv in df_cache_mod._parsed_tree_cache.values() if pv is not None)
            ),
        }

    pre_a = _present_in_all_eight_categories(tree_a, fn_a, nodes_a)
    pre_b = _present_in_all_eight_categories(tree_b, fn_b, nodes_b)
    assert all(pre_a.values()), pre_a
    assert all(pre_b.values()), pre_b

    df_cache_mod.evict_tree_scopes(tree_a)

    post_a = _present_in_all_eight_categories(tree_a, fn_a, nodes_a)
    post_b = _present_in_all_eight_categories(tree_b, fn_b, nodes_b)
    assert not any(post_a.values()), post_a
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, nodes_a)
        for k in df_cache_mod._scope_cell_info_cache
    )
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, nodes_a)
        for k in df_cache_mod._scope_closures_cache
    )
    assert not any(n in df_cache_mod._cached_scope_fingerprints for n in nodes_a)
    assert all(post_b.values()), post_b


def test_evict_tree_scopes_evicts_oversized_ast_from_interior_stmt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies evict_tree_scopes(interior_stmt) evicts enclosing scopes for oversized ASTs."""
    from pydoppelgangerhunt.fixer import _df_cache as df_cache_mod

    from pydoppelgangerhunt.fixer._df_closures import _collect_scope_closures
    from pydoppelgangerhunt.fixer._df_hazards import collect_downstream_read_names

    clear_dataflow_cache()
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 20)
    body_stmts = "\n".join(f"    v_{i} = {i}" for i in range(15))
    oversized_src = f"def huge(x):\n{body_stmts}\n    return v_0 + x\n"
    oversized_tree = df_cache_mod._get_or_parse_tree(oversized_src)
    assert isinstance(oversized_tree, ast.Module)
    fn_node = oversized_tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    interior_stmt = fn_node.body[5]

    other_src = "def small(y):\n    return y + 1\n"
    other_tree = df_cache_mod._get_or_parse_tree(other_src)
    assert isinstance(other_tree, ast.Module)
    other_fn = other_tree.body[0]
    assert isinstance(other_fn, ast.FunctionDef)

    # Populate oversized parent map and enclosing scope caches through real analyzers
    pmap = df_cache_mod._get_scope_ancestor_map(oversized_tree)
    assert pmap.get(fn_node) is oversized_tree
    assert oversized_tree in df_cache_mod._oversized_scope_parent_maps
    assert oversized_tree not in df_cache_mod._scope_parent_maps

    _collect_scope_closures(oversized_tree, oversized_tree)
    _collect_scope_closures(other_tree, other_tree)
    collect_downstream_read_names(
        oversized_src, {"start": 6, "end": 6}, candidates={"v_0"}, tree=oversized_tree
    )
    collect_downstream_read_names(
        other_src, {"start": 2, "end": 2}, candidates={"y"}, tree=other_tree
    )

    df_cache_mod.evict_tree_scopes(interior_stmt)

    assert oversized_tree not in df_cache_mod._oversized_scope_parent_maps
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, {oversized_tree, fn_node, interior_stmt})
        for k in df_cache_mod._scope_cell_info_cache
    )
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, {oversized_tree, fn_node})
        for k in df_cache_mod._scope_closures_cache
    )
    assert any(
        df_cache_mod._key_references_scope_nodes(k, {other_tree, other_fn})
        for k in df_cache_mod._scope_closures_cache
    )


def test_ast_fingerprint_detects_excephandler_name_and_keyword_arg_mutations() -> None:
    """Verifies ExceptHandler.name and keyword.arg mutations raise in debug mode, refresh in prod, and survive prod-to-debug switches."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
    )

    clear_dataflow_cache()
    src = (
        "def outer(fn, handler_flag):\n"
        "    alpha = 1\n"
        "    beta = 2\n"
        "    def inner():\n"
        "        try:\n"
        "            val = fn(alpha=alpha)\n"
        "        except ValueError as err_a:\n"
        "            val = handler_flag(err_a)\n"
        "        return val\n"
        "    return inner\n"
    )

    # 1. Debug mode detects keyword.arg and ExceptHandler.name mutations via RuntimeError
    dbg_tree = ast.parse(src)
    dbg_outer = dbg_tree.body[0]
    assert isinstance(dbg_outer, ast.FunctionDef)
    dbg_inner = dbg_outer.body[2]
    assert isinstance(dbg_inner, ast.FunctionDef)
    try_stmt = dbg_inner.body[0]
    assert isinstance(try_stmt, ast.Try)
    try_assign = try_stmt.body[0]
    assert isinstance(try_assign, ast.Assign) and isinstance(try_assign.value, ast.Call)
    kw_node = try_assign.value.keywords[0]
    handler_node = try_stmt.handlers[0]

    prev_dbg = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = True
    try:
        assert _extract_cached_scope_free_reads(dbg_inner) == frozenset(
            {"fn", "alpha", "ValueError", "handler_flag"}
        )
        kw_node.arg = "beta"
        with pytest.raises(RuntimeError, match="Cached AST node was mutated in-place"):
            _extract_cached_scope_free_reads(dbg_inner)
        df_cache_mod.evict_tree_scopes(dbg_tree)
        assert _extract_cached_scope_free_reads(dbg_inner) == frozenset(
            {"fn", "alpha", "ValueError", "handler_flag"}
        )

        handler_node.name = "err_b"
        with pytest.raises(RuntimeError, match="Cached AST node was mutated in-place"):
            _extract_cached_scope_free_reads(dbg_inner)
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_dbg
        df_cache_mod.evict_tree_scopes(dbg_tree)

    # 2. Production mode refreshes silently on mutation, and switching prod -> debug never spuriously raises
    prod_tree = ast.parse(src)
    prod_outer = prod_tree.body[0]
    assert isinstance(prod_outer, ast.FunctionDef)
    prod_inner = prod_outer.body[2]
    assert isinstance(prod_inner, ast.FunctionDef)
    prod_try = prod_inner.body[0]
    assert isinstance(prod_try, ast.Try)
    prod_call = prod_try.body[0]
    assert isinstance(prod_call, ast.Assign) and isinstance(prod_call.value, ast.Call)
    prod_kw = prod_call.value.keywords[0]
    prod_handler = prod_try.handlers[0]

    ck = df_cache_mod._ScopeCacheKey(prod_inner, "excep_kw_fp_test")
    df_cache_mod._debug_ast_immutability[0] = False
    try:
        assert df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, ck, lambda: ("v1",)
        ) == ("v1",)
        prod_kw.arg = "gamma"
        assert df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, ck, lambda: ("v2",)
        ) == ("v2",)

        # Mode transition (False -> True) refreshes rather than raising false RuntimeError
        prod_handler.name = "err_c"
        df_cache_mod._debug_ast_immutability[0] = True
        assert df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, ck, lambda: ("v3",)
        ) == ("v3",)
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_dbg
        df_cache_mod.evict_tree_scopes(prod_tree)


def test_df_cache_deep_mutation_interior_swap_and_selective_eviction() -> None:
    """Verifies deep expr mutation, same-shape interior swap, detachment, and targeted tree eviction."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_traversal import _extract_cached_scope_free_reads

    clear_dataflow_cache()
    src_main = (
        "def outer(items):\n"
        "    def inner(seq):\n"
        "        return [seq[alpha:] for x in items if ok_flag]\n"
        "    x = init1\n"
        "    return inner\n"
    )
    src_other = "def other_fn(items):\n    return [keep_var for _ in items]\n"

    mod_main = ast.parse(src_main)
    mod_other = ast.parse(src_other)
    outer_fn = mod_main.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    inner_fn = outer_fn.body[0]
    assert isinstance(inner_fn, ast.FunctionDef)
    other_fn = mod_other.body[0]
    assert isinstance(other_fn, ast.FunctionDef)

    # Seed both trees cleanly
    assert _extract_cached_scope_free_reads(other_fn) == frozenset({"keep_var"})
    assert _extract_cached_scope_free_reads(inner_fn) == frozenset(
        {"alpha", "items", "ok_flag"}
    )

    # 1. Deep nested expression mutation (Subscript slice Name inside ListComp)
    ret = inner_fn.body[0]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.ListComp)
    sub = ret.value.elt
    assert isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice)
    assert isinstance(sub.slice.lower, ast.Name)

    prev_dbg = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = True
    try:
        # Switching debug immutability mode on after prod seeding re-fingerprints cleanly without false positives.
        assert _extract_cached_scope_free_reads(inner_fn) == frozenset(
            {"alpha", "items", "ok_flag"}
        )
        sub.slice.lower.id = "omega"
        with pytest.raises(RuntimeError, match="Cached AST node was mutated in-place"):
            _extract_cached_scope_free_reads(inner_fn)
    finally:
        df_cache_mod._debug_ast_immutability[0] = False

    # Production mode refreshes silently on deep mutation
    assert _extract_cached_scope_free_reads(inner_fn) == frozenset(
        {"omega", "items", "ok_flag"}
    )

    # 2. Interior statement replacement with identical body length and identical coordinates
    assert _extract_cached_scope_free_reads(outer_fn) == frozenset(
        {"init1", "ok_flag", "omega"}
    )
    orig_assign = outer_fn.body[1]
    assert isinstance(orig_assign, ast.Assign)
    new_assign = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())],
        value=ast.Name(id="init2", ctx=ast.Load()),
    )
    ast.copy_location(new_assign, orig_assign)
    ast.copy_location(new_assign.targets[0], orig_assign.targets[0])
    ast.copy_location(new_assign.value, orig_assign.value)
    outer_fn.body[1] = new_assign
    assert _extract_cached_scope_free_reads(outer_fn) == frozenset(
        {"init2", "ok_flag", "omega"}
    )

    # 3. Selective eviction removes mod_main scopes (including detached inner_fn) while retaining mod_other scopes
    df_cache_mod._debug_ast_immutability[0] = prev_dbg
    detached = outer_fn.body.pop(0)
    assert isinstance(detached, ast.FunctionDef)
    df_cache_mod.evict_tree_scopes(mod_main)
    cached_owners = {
        k.owner
        for k in df_cache_mod._scope_cell_info_cache
        if isinstance(k, df_cache_mod._ScopeCacheKey)
    }
    assert other_fn in cached_owners
    assert outer_fn not in cached_owners
    assert inner_fn not in cached_owners
    clear_dataflow_cache()


def test_df_cache_concurrent_workers_and_nested_cm_delta_modes() -> None:
    """Verifies thread-safe concurrent scope inspection under eviction and per-mode nested CM delta isolation."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    src = (
        "import contextlib\n"
        "def run(CustomLock, compute):\n"
        "    with contextlib.nullcontext():\n"
        "        with CustomLock():\n"
        "            val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    src_sup = (
        "import contextlib\n"
        "def run(flag, compute):\n"
        "    with contextlib.nullcontext():\n"
        "        with contextlib.suppress(ValueError):\n"
        "            val = compute()\n"
        "    cb = lambda: val\n"
        "    return cb()\n"
    )
    shared_tree = ast.parse(src)
    sup_tree = ast.parse(src_sup)
    unrelated_tree = ast.parse("def u():\n    return 1\n")
    unit = {
        "name": "run:cb",
        "start": 6,
        "end": 6,
        "kind": "basic_block",
        "source_text": src,
        "closure_strictness": "pedantic",
    }

    # Verify nested CM modes remain isolated across repeated cached analyses
    results_by_mode: dict[str, bool] = {}
    for mode in ("lenient", "strict", "pedantic"):
        sc = inspect_single_unit_scope({**unit, "closure_strictness": mode}, tree=shared_tree)
        results_by_mode[mode] = bool(sc["is_control_flow_safe"])
    assert results_by_mode == {"lenient": True, "strict": True, "pedantic": False}

    sup_unit = {**unit, "source_text": src_sup}
    sup_results: dict[str, bool] = {}
    for mode in ("lenient", "strict", "pedantic"):
        sc_sup = inspect_single_unit_scope({**sup_unit, "closure_strictness": mode}, tree=sup_tree)
        sup_results[mode] = bool(sc_sup["is_control_flow_safe"])
    assert sup_results == {"lenient": False, "strict": False, "pedantic": False}

    # Concurrent threads inspecting shared_tree while evicting/clearing cache
    num_workers = 6
    barrier = threading.Barrier(num_workers)
    worker_errors: list[BaseException] = []
    worker_outputs: list[tuple[str, bool]] = []
    out_lock = threading.Lock()

    def _worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5)
            for step in range(8):
                mode = ("strict", "pedantic", "lenient")[(idx + step) % 3]
                sc_w = inspect_single_unit_scope(
                    {**unit, "closure_strictness": mode},
                    tree=shared_tree,
                )
                with out_lock:
                    worker_outputs.append((mode, bool(sc_w["is_control_flow_safe"])))
                if step % 3 == 0:
                    df_cache_mod.evict_tree_scopes(unrelated_tree)
                elif idx == 0 and step == 4:
                    clear_dataflow_cache()
        except BaseException as exc:  # pragma: no cover - captured for re-raise
            with out_lock:
                worker_errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)
    if worker_errors:
        raise worker_errors[0]
    assert len(worker_outputs) == num_workers * 8
    for mode, is_safe in worker_outputs:
        assert is_safe is (mode != "pedantic"), (mode, is_safe)
    clear_dataflow_cache()


def test_evict_tree_scopes_evicts_detached_with_stmt_on_oversized_ast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies evict_tree_scopes removes detached non-scope ast.With summaries attached to oversized parent maps."""
    from pydoppelgangerhunt.fixer import _df_cache as df_cache_mod

    clear_dataflow_cache()
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 20)

    filler = "\n".join(f"    v_{i} = {i}" for i in range(15))
    target_src = (
        "def run_oversized(ctx, item):\n"
        f"{filler}\n"
        "    with ctx() as val:\n"
        "        res = val + item\n"
        "    cb = lambda: res\n"
        "    return cb()\n"
    )
    other_src = (
        "def run_other(ctx, item):\n"
        "    with ctx() as val:\n"
        "        res = val + item\n"
        "    cb = lambda: res\n"
        "    return cb()\n"
    )
    target_tree = df_cache_mod._get_or_parse_tree(target_src)
    other_tree = df_cache_mod._get_or_parse_tree(other_src)
    assert isinstance(target_tree, ast.Module)
    assert isinstance(other_tree, ast.Module)
    outer_fn = target_tree.body[0]
    other_fn = other_tree.body[0]
    assert isinstance(outer_fn, ast.FunctionDef)
    assert isinstance(other_fn, ast.FunctionDef)
    with_target = next(s for s in outer_fn.body if isinstance(s, ast.With))
    with_other = next(s for s in other_fn.body if isinstance(s, ast.With))

    cb_line = next(
        idx for idx, ln in enumerate(target_src.splitlines(), start=1) if "cb = lambda" in ln
    )
    other_cb_line = next(
        idx for idx, ln in enumerate(other_src.splitlines(), start=1) if "cb = lambda" in ln
    )
    inspect_single_unit_scope(
        {
            "name": "run_oversized:cb",
            "start": cb_line,
            "end": cb_line,
            "kind": "basic_block",
            "source_text": target_src,
            "closure_strictness": "pedantic",
        },
        tree=target_tree,
    )
    inspect_single_unit_scope(
        {
            "name": "run_other:cb",
            "start": other_cb_line,
            "end": other_cb_line,
            "kind": "basic_block",
            "source_text": other_src,
            "closure_strictness": "pedantic",
        },
        tree=other_tree,
    )
    assert target_tree in df_cache_mod._oversized_scope_parent_maps
    assert with_target in df_cache_mod._stmt_with_summary_cache
    assert with_other in df_cache_mod._stmt_with_summary_cache

    # Detach with_target from outer_fn before evicting target_tree
    outer_fn.body = [s for s in outer_fn.body if s is not with_target]
    df_cache_mod.evict_tree_scopes(target_tree)

    assert target_tree not in df_cache_mod._oversized_scope_parent_maps
    assert with_target not in df_cache_mod._stmt_with_summary_cache
    assert with_other in df_cache_mod._stmt_with_summary_cache
    clear_dataflow_cache()


def test_register_tree_source_lines_computes_sha256_outside_downstream_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _register_tree_source_lines releases _downstream_cache_lock before computing raw digest SHA-256."""
    from pydoppelgangerhunt.fixer import _df_cache as df_cache_mod

    clear_dataflow_cache()
    src = "def alpha(x):\n    return x + 1\n"
    tree = ast.parse(src)
    orig_resolve = df_cache_mod._resolve_raw_text_digest
    observed_lock_free: list[bool] = []
    worker_errors: list[BaseException] = []

    def _spy_resolve(
        raw_text: str, source_digest: object = None
    ) -> object:
        probe_res: list[bool] = []

        def _probe_lock() -> None:
            try:
                acquired = df_cache_mod._downstream_cache_lock.acquire(blocking=False)
                if acquired:
                    df_cache_mod._downstream_cache_lock.release()
                probe_res.append(acquired)
            except BaseException as exc:  # pragma: no cover
                worker_errors.append(exc)

        probe_thread = threading.Thread(target=_probe_lock)
        probe_thread.start()
        probe_thread.join(timeout=5)
        assert not probe_thread.is_alive()
        observed_lock_free.extend(probe_res)
        return orig_resolve(raw_text, source_digest)  # type: ignore[arg-type]

    monkeypatch.setattr(df_cache_mod, "_resolve_raw_text_digest", _spy_resolve)

    # First registration populates _tree_raw_source_cache; second registration hits hash/len check and refreshes digest
    df_cache_mod._register_tree_source_lines(tree, source_text=src)
    df_cache_mod._tree_raw_source_cache[tree] = (hash(src), len(src), "stale_digest")
    df_cache_mod._register_tree_source_lines(tree, source_text=src)

    assert not worker_errors
    assert len(observed_lock_free) == 2
    assert all(observed_lock_free)

    # Concurrent registrations remain consistent without deadlocks
    trees = [ast.parse(f"def fn_{idx}(x):\n    return x + {idx}\n") for idx in range(8)]
    srcs = [f"def fn_{idx}(x):\n    return x + {idx}\n" for idx in range(8)]
    concurrent_errors: list[BaseException] = []

    def _register_worker(idx: int) -> None:
        try:
            for iter_idx in range(4):
                pos = (idx + iter_idx) % len(trees)
                df_cache_mod._register_tree_source_lines(trees[pos], source_text=srcs[pos])
        except BaseException as exc:  # pragma: no cover
            concurrent_errors.append(exc)

    threads = [threading.Thread(target=_register_worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()
    assert not concurrent_errors
    for t_node, expected_src in zip(trees, srcs):
        assert df_cache_mod._tree_raw_source_cache[t_node] == (
            hash(expected_src),
            len(expected_src),
            df_cache_mod._resolve_raw_text_digest(expected_src, None),
        )
    clear_dataflow_cache()


@pytest.mark.parametrize(
    ("node_budget", "detach_with"),
    [
        pytest.param(100000, False, id="normal_ast_live_siblings"),
        pytest.param(100000, True, id="normal_ast_detached_with"),
        pytest.param(20, False, id="oversized_ast_live_siblings"),
        pytest.param(20, True, id="oversized_ast_detached_with"),
    ],
)
def test_evict_tree_scopes_interior_stmt_evicts_enclosing_sibling_scopes_and_stmts(
    monkeypatch: pytest.MonkeyPatch,
    node_budget: int,
    detach_with: bool,
) -> None:
    """Verifies evict_tree_scopes(stmt) evicts enclosing sibling scopes/with-statements without flushing unrelated functions."""
    from pydoppelgangerhunt.fixer import _df_cache as df_cache_mod
    from pydoppelgangerhunt.fixer._df_cell_carriers import (
        _extract_cached_scope_free_reads,
    )
    from pydoppelgangerhunt.fixer._df_closures import _collect_scope_closures

    clear_dataflow_cache()
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", node_budget)
    filler = "\n".join(f"    pad_{idx} = {idx}" for idx in range(10))
    mod_src = (
        "def target_outer(ctx, item):\n"
        "    mut_stmt = item\n"
        f"{filler}\n"
        "    with ctx() as val:\n"
        "        target_res = val + mut_stmt\n"
        "    def sibling_inner():\n"
        "        return target_res\n"
        "    return sibling_inner()\n"
        "\n"
        "def unrelated_outer(ctx, other):\n"
        "    with ctx() as uval:\n"
        "        other_res = uval + other\n"
        "    def unrelated_inner():\n"
        "        return other_res\n"
        "    return unrelated_inner()\n"
    )
    mod_tree = df_cache_mod._get_or_parse_tree(mod_src)
    assert isinstance(mod_tree, ast.Module)
    target_outer, unrelated_outer = mod_tree.body
    assert isinstance(target_outer, ast.FunctionDef)
    assert isinstance(unrelated_outer, ast.FunctionDef)
    mut_stmt = target_outer.body[0]
    target_with = next(s for s in target_outer.body if isinstance(s, ast.With))
    sibling_inner = next(s for s in target_outer.body if isinstance(s, ast.FunctionDef))
    unrelated_with = next(s for s in unrelated_outer.body if isinstance(s, ast.With))
    unrelated_inner = next(
        s for s in unrelated_outer.body if isinstance(s, ast.FunctionDef)
    )

    lines = mod_src.splitlines()
    target_sib_line = next(
        idx for idx, ln in enumerate(lines, start=1) if "    return sibling_inner()" in ln
    )
    unrelated_sib_line = next(
        idx for idx, ln in enumerate(lines, start=1) if "    return unrelated_inner()" in ln
    )
    inspect_single_unit_scope(
        {
            "name": "target_outer",
            "start": target_sib_line,
            "end": target_sib_line,
            "kind": "basic_block",
            "source_text": mod_src,
            "closure_strictness": "pedantic",
        },
        tree=mod_tree,
    )
    inspect_single_unit_scope(
        {
            "name": "unrelated_outer",
            "start": unrelated_sib_line,
            "end": unrelated_sib_line,
            "kind": "basic_block",
            "source_text": mod_src,
            "closure_strictness": "pedantic",
        },
        tree=mod_tree,
    )
    _collect_scope_closures(sibling_inner, target_outer)
    _collect_scope_closures(unrelated_inner, unrelated_outer)
    _extract_cached_scope_free_reads(sibling_inner)
    _extract_cached_scope_free_reads(unrelated_inner)

    assert target_with in df_cache_mod._stmt_with_summary_cache
    assert unrelated_with in df_cache_mod._stmt_with_summary_cache
    assert any(
        df_cache_mod._key_references_scope_nodes(k, {sibling_inner})
        for k in df_cache_mod._scope_closures_cache
    )
    assert any(
        df_cache_mod._key_references_scope_nodes(k, {sibling_inner})
        for k in df_cache_mod._scope_cell_info_cache
    )

    if detach_with:
        target_outer.body = [s for s in target_outer.body if s is not target_with]
    df_cache_mod.evict_tree_scopes(mut_stmt)

    assert target_with not in df_cache_mod._stmt_with_summary_cache
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, {target_outer, sibling_inner, mut_stmt})
        for k in df_cache_mod._scope_closures_cache
    )
    assert not any(
        df_cache_mod._key_references_scope_nodes(k, {target_outer, sibling_inner, mut_stmt})
        for k in df_cache_mod._scope_cell_info_cache
    )
    assert unrelated_with in df_cache_mod._stmt_with_summary_cache
    assert any(
        df_cache_mod._key_references_scope_nodes(k, {unrelated_inner})
        for k in df_cache_mod._scope_closures_cache
    )
    assert any(
        df_cache_mod._key_references_scope_nodes(k, {unrelated_inner})
        for k in df_cache_mod._scope_cell_info_cache
    )
    clear_dataflow_cache()


@pytest.mark.parametrize(
    ("debug_mode", "mutate_ast"),
    [
        pytest.param(False, False, id="establishes_baseline_and_reuses_parent_map"),
        pytest.param(False, True, id="detects_in_place_mutation_and_rebuilds_in_prod"),
        pytest.param(True, True, id="detects_in_place_mutation_and_raises_in_debug"),
    ],
)
def test_get_scope_parent_map_records_baseline_fingerprint_for_chosen_root(
    debug_mode: bool,
    mutate_ast: bool,
) -> None:
    """Verifies _get_scope_parent_map establishes a baseline fingerprint when chosen_root lacks one."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    mod_tree = ast.parse("def target(x):\n    y = x + 1\n    return y\n")
    fn_node = mod_tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)

    prev_debug = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = debug_mode
    try:
        df_cache_mod._ensure_scope_owner_parent_map(mod_tree)
        assert mod_tree in df_cache_mod._scope_parent_maps
        assert mod_tree in df_cache_mod._cached_scope_fingerprints
        # Simulate fingerprint-table LRU eviction while chosen_root remains in _scope_parent_maps
        df_cache_mod._cached_scope_fingerprints.pop(mod_tree, None)
        assert mod_tree not in df_cache_mod._cached_scope_fingerprints

        pmap_first = df_cache_mod._get_scope_parent_map(fn_node)
        assert mod_tree in df_cache_mod._cached_scope_fingerprints

        if not mutate_ast:
            pmap_second = df_cache_mod._get_scope_parent_map(fn_node)
            assert pmap_second is pmap_first
        else:
            fn_node.body[0] = ast.Assign(
                targets=[ast.Name(id="y", ctx=ast.Store(), lineno=2, col_offset=4)],
                value=ast.Constant(value=99, lineno=2, col_offset=8),
                lineno=2,
                col_offset=4,
                end_lineno=2,
                end_col_offset=10,
            )
            if debug_mode:
                with pytest.raises(RuntimeError, match="mutated in-place"):
                    df_cache_mod._get_scope_parent_map(fn_node)
            else:
                pmap_rebuilt = df_cache_mod._get_scope_parent_map(fn_node)
                assert pmap_rebuilt is not pmap_first
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_debug
        clear_dataflow_cache()


def test_scope_cache_key_tuple_subclass_and_dict_subclass_preservation() -> None:
    """Verifies _canonical_param_type_key distinguishes tuple subclasses and set_entry_locked preserves dict subclasses."""
    from typing import NamedTuple
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    class _Pair(NamedTuple):
        left: int
        right: int

    class _TrackedCustomDict(dict):
        def cached_ast_nodes(self) -> tuple[ast.AST, ...]:
            return ()

    clear_dataflow_cache()
    owner = ast.parse("def f():\n    return 1\n").body[0]

    direct_sub_key = df_cache_mod._make_trusted_scope_cache_key(
        owner, "test_pair", (_Pair(1, 2),)
    )
    direct_plain_key = df_cache_mod._make_trusted_scope_cache_key(
        owner, "test_pair", ((1, 2),)
    )
    assert direct_sub_key != direct_plain_key
    assert len({direct_sub_key, direct_plain_key}) == 2
    assert df_cache_mod._params_type_hash((_Pair(1, 2),)) != df_cache_mod._params_type_hash(((1, 2),))

    fs_sub_key = df_cache_mod._make_trusted_scope_cache_key(
        owner, "test_pair_fs", (frozenset({_Pair(1, 2)}),)
    )
    fs_plain_key = df_cache_mod._make_trusted_scope_cache_key(
        owner, "test_pair_fs", (frozenset({(1, 2)}),)
    )
    assert fs_sub_key != fs_plain_key
    assert len({fs_sub_key, fs_plain_key}) == 2
    assert {fs_sub_key: "sub", fs_plain_key: "plain"}[fs_plain_key] == "plain"

    custom_val = _TrackedCustomDict({"a": 1})
    plain_val = {"a": 1}
    with df_cache_mod._downstream_cache_lock:
        stored_custom, ok_custom = df_cache_mod._scope_cell_info_cache.set_entry_locked(
            direct_sub_key, custom_val
        )
        stored_plain, ok_plain = df_cache_mod._scope_cell_info_cache.set_entry_locked(
            direct_plain_key, plain_val
        )
    assert ok_custom is True and ok_plain is True
    assert type(stored_custom) is _TrackedCustomDict
    assert isinstance(stored_plain, df_cache_mod._TrackedCachedDict)
    clear_dataflow_cache()


def test_tracked_cached_dict_or_and_lazy_seed_span_caches_thread_safety() -> None:
    """Verifies concurrent safety of _TrackedCachedDict.__or__, _SpanLineIndex.get_dump, and _LazyModuleStmtSeedMap."""
    import concurrent.futures
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    import pydoppelgangerhunt.fixer._df_span_index as span_idx_mod
    from pydoppelgangerhunt.fixer._df_with_provenance import _build_module_suppress_seed

    clear_dataflow_cache()
    tracked = df_cache_mod._TrackedCachedDict({"seed": 0})
    mod_src = "import contextlib\n" + "\n".join(
        f"cm_{i} = contextlib.nullcontext" for i in range(64)
    ) + "\n"
    mod_tree = ast.parse(mod_src)
    span_idx = span_idx_mod._SpanLineIndex(mod_tree)
    stmt_nodes = list(mod_tree.body)
    seed_map = _build_module_suppress_seed(mod_tree).by_top_stmt_id

    def _mutator_worker(worker_id: int) -> int:
        nonlocal tracked
        for step in range(40):
            key = f"k_{worker_id}_{step % 8}"
            if step % 2 == 0:
                tracked |= {key: step}
            else:
                tracked.pop(key, None)
        return worker_id

    def _reader_worker(worker_id: int) -> int:
        for step in range(40):
            merged = tracked | {"probe": step}
            assert "seed" in merged
            stmt = stmt_nodes[(worker_id + step) % len(stmt_nodes)]
            assert isinstance(span_idx.get_dump(stmt), str)
            assert seed_map[id(stmt)] is not None
        return worker_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(_mutator_worker if i < 2 else _reader_worker, i)
            for i in range(6)
        ]
        for fut in futures:
            fut.result()
    clear_dataflow_cache()


def test_large_module_fingerprint_suite_stride_and_session_dedup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies suite sampling bounds large fingerprints and ContextVar sessions deduplicate checks."""
    import concurrent.futures
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    n_stmts = 2500
    large_src = "\n".join(f"def f_{i}():\n    return {i}" for i in range(n_stmts)) + "\n"
    mod_tree = ast.parse(large_src)

    # Stride preserves head/tail/length detection in production mode and inspects bounded items
    fp_base = df_cache_mod._compute_scope_fingerprint(mod_tree)
    orig_first = mod_tree.body[0]
    mod_tree.body[0] = ast.FunctionDef(
        name="f_0_mutated",
        args=orig_first.args,  # type: ignore[attr-defined]
        body=orig_first.body,  # type: ignore[attr-defined]
        decorator_list=[],
    )
    assert df_cache_mod._compute_scope_fingerprint(mod_tree) != fp_base
    mod_tree.body[0] = orig_first
    assert df_cache_mod._compute_scope_fingerprint(mod_tree) == fp_base

    orig_fn = df_cache_mod._compute_scope_fingerprint
    fp_calls = [0]

    def _counting_fp(node: ast.AST) -> tuple[Any, ...]:
        fp_calls[0] += 1
        return orig_fn(node)

    monkeypatch.setattr(df_cache_mod, "_compute_scope_fingerprint", _counting_fp)
    ck = df_cache_mod._make_trusted_scope_cache_key(owner=mod_tree, kind="span_line_index")
    df_cache_mod._get_or_compute_scope_cached(
        df_cache_mod._scope_cell_info_cache, ck, lambda: ("seed_val",)
    )
    fp_calls[0] = 0

    with df_cache_mod.scope_fingerprint_verification_session():
        for _ in range(100):
            res = df_cache_mod._get_or_compute_scope_cached(
                df_cache_mod._scope_cell_info_cache, ck, lambda: ("unexpected",)
            )
            assert res == ("seed_val",)
    assert fp_calls[0] == 1

    # Epoch bump via evict_tree_scopes invalidates active session verification cleanly
    with df_cache_mod.scope_fingerprint_verification_session():
        df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, ck, lambda: ("reseeded",)
        )
        before_evict = fp_calls[0]
        df_cache_mod.evict_tree_scopes(mod_tree)
        df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, ck, lambda: ("rebuilt",)
        )
        assert fp_calls[0] > before_evict

    # Debug immutability mode bypasses session deduplication
    prev_debug = df_cache_mod._debug_ast_immutability[0]
    try:
        df_cache_mod._debug_ast_immutability[0] = True
        fp_calls[0] = 0
        with df_cache_mod.scope_fingerprint_verification_session():
            for _ in range(5):
                df_cache_mod._get_or_compute_scope_cached(
                    df_cache_mod._scope_cell_info_cache, ck, lambda: ("rebuilt",)
                )
        assert fp_calls[0] == 5
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_debug

    # Worker threads have independent ContextVar sessions and raise cleanly via fut.result()
    def _worker_check(_: int) -> int:
        with df_cache_mod.scope_fingerprint_verification_session():
            return len(
                df_cache_mod._get_or_compute_scope_cached(
                    df_cache_mod._scope_cell_info_cache, ck, lambda: ("w",)
                )
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for fut in [pool.submit(_worker_check, i) for i in range(4)]:
            assert fut.result() == 1
    clear_dataflow_cache()


@pytest.mark.parametrize(
    "bad_source_lines",
    [
        "x = 1\n",
        b"x = 1\n",
        bytearray(b"x = 1\n"),
        {"x = 1"},
        123,
    ],
)
def test_register_tree_source_lines_rejects_strings_bytes_and_non_sequences(
    bad_source_lines: Any,
) -> None:
    """Verifies _register_tree_source_lines rejects str, bytes, bytearray, sets, and scalars."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    tree = ast.parse("x = 1\n")
    with pytest.raises(TypeError):
        df_cache_mod._register_tree_source_lines(tree, bad_source_lines)
    assert df_cache_mod._register_tree_source_lines(tree, ["x = 1"]) == ("x = 1",)
    clear_dataflow_cache()


def test_tracked_cached_dict_atomic_rollback_on_value_error() -> None:
    """Verifies _TrackedCachedDict leaves existing values and refcounts intact on extraction errors."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    owner = ast.parse("def f():\n    return x\n")
    n1 = ast.Name(id="a", ctx=ast.Load())
    n2 = ast.Name(id="b", ctx=ast.Load())
    tracked = df_cache_mod._TrackedCachedDict({"k1": n1})
    ck = df_cache_mod._make_trusted_scope_cache_key(owner=owner, kind="fn_scope_ctx")
    df_cache_mod._scope_cell_info_cache[ck] = tracked

    class _UnregisteredObj:
        pass

    bad = _UnregisteredObj()
    with pytest.raises(df_cache_mod._MissingCachedAstNodesError):
        tracked["k1"] = bad
    assert tracked["k1"] is n1
    assert n1 in df_cache_mod._scope_owner_keys_index

    with pytest.raises(df_cache_mod._MissingCachedAstNodesError):
        tracked.setdefault("k2", bad)
    assert "k2" not in tracked

    with pytest.raises(df_cache_mod._MissingCachedAstNodesError):
        tracked.update({"k2": n2, "k3": bad})
    assert "k2" not in tracked and "k3" not in tracked
    assert n2 not in df_cache_mod._scope_owner_keys_index
    clear_dataflow_cache()


def test_tracked_scope_cache_update_setdefault_and_ior_sync_indexes() -> None:
    """Verifies _TrackedScopeCache update, setdefault, and |= keep kind rings and owner indexes synchronized."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    cache = df_cache_mod._scope_cell_info_cache
    owners = [ast.parse(f"def f_{i}():\n    return v_{i}\n") for i in range(4)]
    names = [ast.Name(id=f"n_{i}", ctx=ast.Load()) for i in range(4)]
    keys = [
        df_cache_mod._make_trusted_scope_cache_key(owner=owners[i], kind="stmt_deltas")
        for i in range(4)
    ]

    cache.update({keys[0]: (names[0],)})
    assert cache.setdefault(keys[1], (names[1],)) == (names[1],)
    assert cache.setdefault(keys[1], (names[2],)) == (names[1],)
    cache |= {keys[2]: (names[2],), keys[3]: (names[3],)}

    kind_ring = cache.ensure_kind_keys().get("stmt_deltas")
    assert kind_ring is not None
    for idx in range(4):
        assert keys[idx] in kind_ring
        assert names[idx] in df_cache_mod._scope_owner_keys_index

    df_cache_mod.evict_tree_scopes(owners[2])
    assert keys[2] not in cache
    assert keys[2] not in kind_ring
    assert names[2] not in df_cache_mod._scope_owner_keys_index
    clear_dataflow_cache()


class _TaggedFrozenset(frozenset):  # type: ignore[type-arg]
    """Frozenset subclass used to verify exact-type cache key discrimination."""


class _TaggedFloat(float):
    """Float subclass used to verify exact-type cache key discrimination."""


@pytest.mark.parametrize(
    ("val_a", "val_b", "expect_equal"),
    [
        pytest.param(
            frozenset({frozenset({"a"})}),
            frozenset({_TaggedFrozenset({"a"})}),
            False,
            id="frozenset_subclass_inside_frozenset_differs_from_base",
        ),
        pytest.param(
            frozenset({1.5}),
            frozenset({_TaggedFloat(1.5)}),
            False,
            id="float_subclass_inside_frozenset_differs_from_base",
        ),
        pytest.param(
            frozenset({0.0}),
            frozenset({_TaggedFloat(0.0)}),
            False,
            id="pos_zero_float_subclass_inside_frozenset_differs_from_base",
        ),
        pytest.param(
            frozenset({-0.0}),
            frozenset({_TaggedFloat(-0.0)}),
            False,
            id="neg_zero_float_subclass_inside_frozenset_differs_from_base",
        ),
        pytest.param(
            frozenset({float("nan")}),
            frozenset({_TaggedFloat("nan")}),
            False,
            id="nan_float_subclass_inside_frozenset_differs_from_base",
        ),
        pytest.param(
            frozenset({_TaggedFrozenset({"a"})}),
            frozenset({_TaggedFrozenset({"a"})}),
            True,
            id="control_same_frozenset_subclass_matches",
        ),
        pytest.param(
            frozenset({_TaggedFloat(1.5)}),
            frozenset({_TaggedFloat(1.5)}),
            True,
            id="control_same_float_subclass_matches",
        ),
    ],
)
def test_scope_cache_key_preserves_frozenset_and_float_subclass_identity(
    val_a: object,
    val_b: object,
    expect_equal: bool,
) -> None:
    """Verifies _canonical_param_type_key and _ScopeCacheKey preserve frozenset and float subclass types."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    owner = ast.parse("x = 1\n")
    key_a = df_cache_mod._ScopeCacheKey(owner, "subclass_type_test", (val_a,))
    key_b = df_cache_mod._ScopeCacheKey(owner, "subclass_type_test", (val_b,))
    assert (key_a == key_b) is expect_equal
    assert (
        df_cache_mod._canonical_param_type_key(val_a)
        == df_cache_mod._canonical_param_type_key(val_b)
    ) is expect_equal
    assert df_cache_mod._params_exact_types_match(val_a, val_b) is expect_equal
    if expect_equal:
        assert hash(key_a) == hash(key_b)


def test_weak_parent_map_bool_avoids_len_and_oversized_entry_builds_outside_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _WeakParentMap.__bool__ avoids O(N) __len__, handles non-weakref roots, and builds oversized entries lock-free."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    tree = ast.parse("def f(x):\n    return x + 1\n")
    pmap = df_cache_mod._build_parent_map_unlocked(tree)
    wmap = df_cache_mod._WeakParentMap(pmap)
    empty_wmap = df_cache_mod._WeakParentMap({})

    len_calls = 0

    def _fail_len(self: Any) -> int:
        nonlocal len_calls
        len_calls += 1
        raise AssertionError("__bool__ must not delegate to __len__")

    monkeypatch.setattr(df_cache_mod._WeakParentMap, "__len__", _fail_len)
    assert bool(wmap) is True
    assert bool(empty_wmap) is False
    assert len_calls == 0

    # Non-weakrefable custom AST root node marks has_non_weakref and falls back to dict mapping
    class _NoWeakrefRoot(ast.AST):
        __slots__ = ()

    custom_root = _NoWeakrefRoot()
    child_node = ast.Name(id="v", ctx=ast.Load())
    oversized_custom = df_cache_mod._build_oversized_parent_entry_unlocked(
        custom_root, {child_node: custom_root}
    )
    assert oversized_custom.has_non_weakref is True
    assert oversized_custom.weak_parent_map.get(child_node) is custom_root

    # Oversized entry construction runs outside _downstream_cache_lock
    monkeypatch.setattr(df_cache_mod, "_MAX_PARENT_MAP_TOTAL_NODES", 10)
    orig_builder = df_cache_mod._build_oversized_parent_entry_unlocked
    lock_free_observations: list[bool] = []
    worker_errors: list[BaseException] = []

    def _spy_oversized_builder(*args: Any, **kwargs: Any) -> Any:
        res: list[bool] = []

        def _probe() -> None:
            try:
                acquired = df_cache_mod._downstream_cache_lock.acquire(blocking=False)
                if acquired:
                    df_cache_mod._downstream_cache_lock.release()
                res.append(acquired)
            except BaseException as exc:  # pragma: no cover - captured for re-raise
                worker_errors.append(exc)

        t = threading.Thread(target=_probe)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive()
        lock_free_observations.extend(res)
        return orig_builder(*args, **kwargs)

    monkeypatch.setattr(
        df_cache_mod, "_build_oversized_parent_entry_unlocked", _spy_oversized_builder
    )
    big_tree = ast.parse(
        "def big(x):\n"
        + "".join(f"    a_{i} = x + {i}\n" for i in range(12))
        + "    return a_0\n"
    )
    anc_map = df_cache_mod._get_scope_ancestor_map(big_tree)
    assert isinstance(anc_map, dict)
    if worker_errors:
        raise worker_errors[0]
    assert lock_free_observations and all(lock_free_observations)
    clear_dataflow_cache()


def test_tracked_cached_dict_unregister_pops_empty_key_nodes_bucket() -> None:
    """Verifies deleting the last AST node from a _TrackedCachedDict under a non-AST key removes empty _scope_key_nodes_index buckets."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    n1 = ast.Name(id="first", ctx=ast.Load())
    n2 = ast.Name(id="second", ctx=ast.Load())
    non_ast_key = ("non_ast_tuple_key", 42)
    tracked = df_cache_mod._TrackedCachedDict({"k1": n1, "k2": n2})
    df_cache_mod._scope_cell_info_cache[non_ast_key] = tracked

    assert non_ast_key in df_cache_mod._scope_key_nodes_index
    assert df_cache_mod._scope_key_nodes_index[non_ast_key] == {n1, n2}

    # Control: removing one entry keeps the bucket with the remaining node
    del tracked["k1"]
    assert non_ast_key in df_cache_mod._scope_key_nodes_index
    assert df_cache_mod._scope_key_nodes_index[non_ast_key] == {n2}
    assert n1 not in df_cache_mod._scope_owner_keys_index

    # Removing the final AST node must pop the now-empty bucket from _scope_key_nodes_index
    del tracked["k2"]
    assert non_ast_key not in df_cache_mod._scope_key_nodes_index
    assert n2 not in df_cache_mod._scope_owner_keys_index
    clear_dataflow_cache()


@pytest.mark.parametrize("debug_mode", [True, False], ids=["debug_raises", "prod_evicts_and_recomputes"])
def test_get_or_compute_scope_cached_fingerprints_aux_ast_nodes(
    debug_mode: bool,
) -> None:
    """Verifies _get_or_compute_scope_cached checks aux_ast_nodes fingerprints for in-place mutations outside owner."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    prev_debug = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = debug_mode
    try:
        mod_tree = ast.parse("def target(x):\n    return x + 1\n")
        fn_node = mod_tree.body[0]
        key = df_cache_mod._make_trusted_scope_cache_key(
            owner=fn_node,
            kind="fn_scope_ctx",
            aux_ast_nodes=(mod_tree,),
        )
        compute_calls = 0

        def _compute() -> int:
            nonlocal compute_calls
            compute_calls += 1
            return len(mod_tree.body)

        first = df_cache_mod._get_or_compute_scope_cached(
            df_cache_mod._scope_cell_info_cache, key, _compute
        )
        assert first == 1
        assert compute_calls == 1

        # Negative control: repeated lookup without mutation hits cache in O(1)
        with df_cache_mod.scope_fingerprint_verification_session():
            cached = df_cache_mod._get_or_compute_scope_cached(
                df_cache_mod._scope_cell_info_cache, key, _compute
            )
            cached_again = df_cache_mod._get_or_compute_scope_cached(
                df_cache_mod._scope_cell_info_cache, key, _compute
            )
        assert cached == 1
        assert cached_again == 1
        assert compute_calls == 1

        # Mutate aux_ast_nodes (mod_tree) outside fn_node without calling evict_tree_scopes
        mod_tree.body.append(ast.parse("from evil import *\n").body[0])
        if debug_mode:
            with pytest.raises(RuntimeError, match="mutated in-place"):
                df_cache_mod._get_or_compute_scope_cached(
                    df_cache_mod._scope_cell_info_cache, key, _compute
                )
        else:
            refreshed = df_cache_mod._get_or_compute_scope_cached(
                df_cache_mod._scope_cell_info_cache, key, _compute
            )
            assert refreshed == 2
            assert compute_calls == 2
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_debug
        clear_dataflow_cache()


@pytest.mark.parametrize(
    ("use_weak_map", "expect_stored_parent_map"),
    [
        pytest.param(True, False, id="weak_parent_map_skips_len_and_drops_stored_map"),
        pytest.param(False, True, id="control_plain_small_dict_retains_stored_map"),
    ],
)
def test_make_nested_fn_scope_context_skips_len_on_weak_parent_map(
    monkeypatch: pytest.MonkeyPatch,
    use_weak_map: bool,
    expect_stored_parent_map: bool,
) -> None:
    """Verifies _make_nested_fn_scope_context checks _WeakParentMap before calling len() to avoid O(N) scans."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod
    import pydoppelgangerhunt.fixer._df_cell_carriers as carriers_mod

    class _NoLenWeakParentMap(df_cache_mod._WeakParentMap):
        def __len__(self) -> int:
            raise AssertionError("__len__ must not be called on _WeakParentMap")

    mod_tree = ast.parse("def nested(x):\n    return x + 1\n")
    fn_node = mod_tree.body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    raw_dict: dict[ast.AST, ast.AST] = {fn_node.body[0]: fn_node}
    returned_map: object = (
        _NoLenWeakParentMap(raw_dict) if use_weak_map else raw_dict
    )
    monkeypatch.setattr(
        carriers_mod,
        "_get_scope_parent_map",
        lambda _node, *, use_cache=True: returned_map,
    )
    ctx = carriers_mod._make_nested_fn_scope_context(
        fn_node, frozenset({"x"}), scope_ctx=None, use_cache=True
    )
    assert ctx.has_nonempty_parent_map is True
    assert (ctx._stored_parent_map is not None) is expect_stored_parent_map


def test_node_fingerprint_scalar_detects_comp_async_and_match_pattern_mutations() -> None:
    """Verifies _compute_scope_fingerprint detects in-place mutations to comprehension.is_async and match patterns."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    comp_mod = ast.parse("async def f(xs):\n    return [x for x in xs]\n")
    comp_node = next(n for n in ast.walk(comp_mod) if isinstance(n, ast.comprehension))
    fp_sync = df_cache_mod._compute_scope_fingerprint(comp_mod)
    comp_node.is_async = 1
    fp_async = df_cache_mod._compute_scope_fingerprint(comp_mod)
    assert fp_async != fp_sync

    match_mod = ast.parse(
        "def m(v):\n"
        "    match v:\n"
        "        case [first, *rest_seq]:\n"
        "            return first, rest_seq\n"
        "        case {'k': True, **rest_map}:\n"
        "            return rest_map\n"
    )
    fp_base = df_cache_mod._compute_scope_fingerprint(match_mod)

    match_as = next(n for n in ast.walk(match_mod) if isinstance(n, ast.MatchAs) and n.name == "first")
    match_as.name = "renamed_first"
    fp_as = df_cache_mod._compute_scope_fingerprint(match_mod)
    assert fp_as != fp_base

    match_star = next(n for n in ast.walk(match_mod) if isinstance(n, ast.MatchStar))
    match_star.name = "renamed_star"
    fp_star = df_cache_mod._compute_scope_fingerprint(match_mod)
    assert fp_star != fp_as

    match_map = next(n for n in ast.walk(match_mod) if isinstance(n, ast.MatchMapping))
    match_map.rest = "renamed_map"
    fp_map = df_cache_mod._compute_scope_fingerprint(match_mod)
    assert fp_map != fp_star

    match_sing = next(n for n in ast.walk(match_mod) if isinstance(n, ast.MatchSingleton))
    match_sing.value = False
    fp_sing = df_cache_mod._compute_scope_fingerprint(match_mod)
    assert fp_sing != fp_map


@pytest.mark.parametrize(
    "index_kind",
    [
        pytest.param("ensure_owner", id="ensure_scope_owner_parent_map"),
        pytest.param("tree_contains", id="tree_contains_scope_node"),
        pytest.param("ancestor_map", id="get_scope_ancestor_map"),
    ],
)
@pytest.mark.parametrize("debug_mode", [True, False], ids=["debug", "prod"])
def test_parent_map_root_indexing_records_baseline_fingerprint_for_chosen_root_lookup(
    index_kind: str,
    debug_mode: bool,
) -> None:
    """Verifies trees indexed via parent-map helpers detect in-place AST mutations on _get_scope_parent_map chosen_root hits."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    prev_debug = df_cache_mod._debug_ast_immutability[0]
    df_cache_mod._debug_ast_immutability[0] = debug_mode
    try:
        clear_dataflow_cache()
        mod_tree = ast.parse("def work(x):\n    y = x + 1\n    return y\n")
        fn_node = mod_tree.body[0]
        assert isinstance(fn_node, ast.FunctionDef)
        if index_kind == "ensure_owner":
            df_cache_mod._ensure_scope_owner_parent_map(mod_tree)
        elif index_kind == "tree_contains":
            assert df_cache_mod._tree_contains_scope_node(mod_tree, fn_node) is True
        else:
            assert df_cache_mod._get_scope_ancestor_map(mod_tree)[fn_node] is mod_tree

        assert mod_tree in df_cache_mod._cached_scope_fingerprints
        # Positive control: unmutated chosen_root fast path returns cached parent map
        pmap_before = df_cache_mod._get_scope_parent_map(fn_node)
        assert pmap_before[fn_node] is mod_tree

        # Mutate mod_tree in place without calling evict_tree_scopes(mod_tree)
        injected_stmt = ast.parse("z = 99\n").body[0]
        fn_node.body.insert(0, injected_stmt)
        if debug_mode:
            with pytest.raises(
                RuntimeError,
                match="mutated in-place without calling evict_tree_scopes",
            ):
                df_cache_mod._get_scope_parent_map(fn_node)
        else:
            pmap_after = df_cache_mod._get_scope_parent_map(fn_node)
            assert pmap_after.get(injected_stmt) is fn_node
    finally:
        df_cache_mod._debug_ast_immutability[0] = prev_debug
        clear_dataflow_cache()


def test_evict_tree_scopes_unregisters_cached_keys_once_and_prunes_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies _evict_tree_scopes does not redundantly invoke _unregister_key_if_absent after TrackedScopeCache.pop."""
    import pydoppelgangerhunt.fixer._df_cache as df_cache_mod

    clear_dataflow_cache()
    mod_tree = ast.parse("def work(x):\n    return x + 1\n")
    ck = df_cache_mod._make_trusted_scope_cache_key(owner=mod_tree, kind="span_line_index")
    df_cache_mod._get_or_compute_scope_cached(
        df_cache_mod._scope_cell_info_cache, ck, lambda: ("val",)
    )

    unreg_calls: list[Any] = []
    orig_unreg = df_cache_mod._unregister_key_if_absent

    def _spy_unreg(key: Any) -> None:
        unreg_calls.append(key)
        orig_unreg(key)

    monkeypatch.setattr(df_cache_mod, "_unregister_key_if_absent", _spy_unreg)
    df_cache_mod.evict_tree_scopes(mod_tree)
    assert unreg_calls == [ck]
    assert ck not in df_cache_mod._scope_key_nodes_index

    # Orphan reverse-index key (not in either cache) is still pruned during eviction
    orphan_tree = ast.parse("def orphan():\n    return 0\n")
    orphan_key = df_cache_mod._make_trusted_scope_cache_key(
        owner=orphan_tree, kind="span_line_index"
    )
    df_cache_mod._register_key_ast_nodes(orphan_key, (orphan_tree,))
    unreg_calls.clear()
    df_cache_mod.evict_tree_scopes(orphan_tree)
    assert unreg_calls == [orphan_key]
    assert orphan_key not in df_cache_mod._scope_key_nodes_index





