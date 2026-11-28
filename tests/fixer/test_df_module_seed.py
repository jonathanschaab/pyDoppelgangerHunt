"""Tests for module-level suppression seeding, enclosing compound clauses, star/relative imports, and escaping defs."""

from __future__ import annotations

import ast
import sys

import pytest

from pydoppelgangerhunt.fixer import (
    clear_dataflow_cache,
    inspect_single_unit_scope,
)
from tests.fixer._prov_test_helpers import (
    _late_bound_capture_safety_by_mode,
)

def test_local_class_allowlist_and_nested_function_seeding() -> None:
    """Verifies local/relative allowlist matching, __exit__ proof, and nested function seeding."""
    from pydoppelgangerhunt.fixer._df_block_flow import _analyze_block_assignment
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    local_and_rel_src = (
        "from .tx import RelativeTx\n"
        "class SafeTransaction:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n"
        "class DynamicTransaction(BaseTx):\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        self._cleanup(exc_type)\n"
        "def run_tx(compute):\n"
        "    with SafeTransaction():\n"
        "        auto_proven = compute()\n"
        "    with DynamicTransaction(), RelativeTx():\n"
        "        allowlisted = compute()\n"
        "    return auto_proven, allowlisted\n"
    )
    local_rel_tree = ast.parse(local_and_rel_src)
    tx_fn = local_rel_tree.body[3]
    assert isinstance(tx_fn, ast.FunctionDef)
    prov_tx = _seed_scope_suppress_aliases(tx_fn, mod_tree=local_rel_tree)
    assert "SafeTransaction" in prov_tx.non_suppressors
    assert "SafeTransaction" not in prov_tx.shadowed_names
    assert "DynamicTransaction" not in prov_tx.shadowed_names
    assert "RelativeTx" not in prov_tx.shadowed_names

    defs_no_allow, conds_no_allow = _analyze_block_assignment(
        tx_fn.body[:-1],
        suppress_aliases=prov_tx,
        strict_with_suppression=True,
    )
    assert "auto_proven" in defs_no_allow
    assert "allowlisted" not in defs_no_allow
    assert "allowlisted" in conds_no_allow

    defs_with_allow, conds_with_allow = _analyze_block_assignment(
        tx_fn.body[:-1],
        suppress_aliases=prov_tx,
        extra_non_suppressing_ctx_managers=frozenset(
            {"DynamicTransaction", ".RelativeTx"}
        ),
        strict_with_suppression=True,
    )
    assert {"auto_proven", "allowlisted"} <= defs_with_allow
    assert "allowlisted" not in conds_with_allow

    nested_seed_src = (
        "from contextlib import nullcontext as nc\n"
        "def outer():\n"
        "    def inner(compute):\n"
        "        with nc():\n"
        "            x = compute()\n"
        "        return x\n"
        "    return inner\n"
        "nc = unknown_rebind()\n"
    )
    nested_seed_tree = ast.parse(nested_seed_src)
    outer_seed_fn = nested_seed_tree.body[1]
    assert isinstance(outer_seed_fn, ast.FunctionDef)
    inner_seed_fn = outer_seed_fn.body[0]
    assert isinstance(inner_seed_fn, ast.FunctionDef)
    prov_outer = _seed_scope_suppress_aliases(outer_seed_fn, mod_tree=nested_seed_tree)
    prov_inner = _seed_scope_suppress_aliases(inner_seed_fn, mod_tree=nested_seed_tree)
    assert prov_inner.non_suppressors == prov_outer.non_suppressors
    assert "nc" not in prov_inner.non_suppressors

def test_module_compound_statement_suppressor_alias_propagation_and_linear_seed() -> (
    None
):
    """Verifies suppressor imports inside module-level compound blocks propagate into nested functions."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import (
        _ModuleSuppressSeed,
        _build_module_suppress_seed,
        _seed_scope_suppress_aliases,
    )

    nested_mod_src = (
        "if True:\n"
        "    from contextlib import suppress as sup\n"
        "    def target_fn(d, items):\n"
        "        with sup(KeyError):\n"
        "            val = d['k']\n"
        "        return [val for _ in items]\n"
    )
    mod_tree = ast.parse(nested_mod_src)
    if_stmt = mod_tree.body[0]
    assert isinstance(if_stmt, ast.If)
    target_fn = if_stmt.body[1]
    assert isinstance(target_fn, ast.FunctionDef)

    seeded_prov = _seed_scope_suppress_aliases(
        target_fn, mod_tree, u_start=target_fn.lineno
    )
    assert "sup" in seeded_prov.suppressors

    bound, _ = _extract_block_definite_bindings_and_term(
        target_fn.body[:1],
        None,
        strict_with_suppression=False,
        suppress_aliases=seeded_prov,
    )
    assert "val" not in bound

    many_mod_src = "".join(
        [
            "from contextlib import suppress as s0\n",
            *[f"s{i} = s{i - 1}\n" for i in range(1, 50)],
            "def f(x):\n    return x\n",
            "from contextlib import suppress as post_sup\n",
        ]
    )
    many_tree = ast.parse(many_mod_src)
    seed = _build_module_suppress_seed(many_tree)
    f_stmt = many_tree.body[-2]
    assert len(seed.by_top_stmt_id) == len(many_tree.body)
    assert "s49" in seed.prefix_by_top_stmt_id[id(f_stmt)].suppressors
    assert "post_sup" in seed.suffix_delta_by_top_stmt_id[id(f_stmt)].suppressors
    assert {"s49", "post_sup"} <= seed.by_top_stmt_id[id(f_stmt)].suppressors

    from types import MappingProxyType

    default_seed_a = _ModuleSuppressSeed(seeded_prov, MappingProxyType({}), False)
    default_seed_b = _ModuleSuppressSeed(seeded_prov, MappingProxyType({}), False)
    assert default_seed_a.prefix_by_top_stmt_id == {}
    assert default_seed_b.suffix_delta_by_top_stmt_id == {}
    with pytest.raises(TypeError):
        default_seed_a.prefix_by_top_stmt_id[1] = seeded_prov  # type: ignore[index]
    with pytest.raises(TypeError):
        default_seed_b.suffix_delta_by_top_stmt_id[1] = seeded_prov  # type: ignore[index]

def test_seed_scope_suppress_aliases_clears_function_locals_at_entry() -> None:
    """Verifies function-local assignments shadow module-level non-suppressor imports even when u_start=0."""
    from pydoppelgangerhunt.fixer._df_block_flow import (
        _extract_block_definite_bindings_and_term,
    )
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    src = (
        "from contextlib import nullcontext\n"
        "def run(custom_cm, d):\n"
        "    nullcontext = custom_cm\n"
        "    with nullcontext():\n"
        "        val = d['k']\n"
        "    from contextlib import suppress as late_sup\n"
        "    return val, late_sup\n"
    )
    mod_tree = ast.parse(src)
    fn_node = mod_tree.body[1]
    assert isinstance(fn_node, ast.FunctionDef)

    prov_zero = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=0)
    assert "nullcontext" not in prov_zero.non_suppressors
    assert "nullcontext" in prov_zero.shadowed_names

    prov_mid = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=3)
    assert "nullcontext" not in prov_mid.non_suppressors
    assert "late_sup" not in prov_mid.suppressors

    prov_after_sup = _seed_scope_suppress_aliases(fn_node, mod_tree, u_start=7)
    assert "nullcontext" not in prov_after_sup.non_suppressors
    assert "late_sup" in prov_after_sup.suppressors

    bound, _ = _extract_block_definite_bindings_and_term(
        fn_node.body[:2],
        None,
        frozenset({"custom_cm", "d"}),
        strict_with_suppression=True,
        suppress_aliases=prov_zero,
    )
    assert "val" not in bound

def test_seed_scope_suppress_aliases_stops_before_u_start_and_post_unit() -> None:
    """Verifies pre-unit provenance seeding stops before u_start in linear/if blocks while remaining sound across loops."""
    from pydoppelgangerhunt.fixer._df_module_seed import _seed_scope_suppress_aliases

    # 1. Loop body can re-enter after post-unit rebind, so loop body invalidates cm conservatively
    loop_src = (
        "import contextlib\n"
        "def run_loop(compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    for _ in range(1):\n"
        "        with cm():\n"
        "            val = compute()\n"
        "        cm = custom_unknown\n"
        "    return val\n"
    )
    loop_mod = ast.parse(loop_src)
    loop_fn = loop_mod.body[1]
    assert isinstance(loop_fn, ast.FunctionDef)
    prov_loop = _seed_scope_suppress_aliases(loop_fn, loop_mod, u_start=5)
    assert "cm" not in prov_loop.non_suppressors

    # 2. Single-pass `if` branch: post-unit `cm = custom_unknown` does not pre-invalidate u_start=5
    if_src = (
        "import contextlib\n"
        "def run_if(cond, compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    if cond:\n"
        "        with cm() as cm:\n"
        "            val = compute()\n"
        "        cm = custom_unknown\n"
        "    return val\n"
    )
    if_mod = ast.parse(if_src)
    if_fn = if_mod.body[1]
    assert isinstance(if_fn, ast.FunctionDef)
    prov_if_before = _seed_scope_suppress_aliases(if_fn, if_mod, u_start=5)
    assert "cm" in prov_if_before.non_suppressors
    prov_if_after = _seed_scope_suppress_aliases(if_fn, if_mod, u_start=8)
    assert "cm" not in prov_if_after.non_suppressors

    # 3. Direct sequential statements: `with cm() as cm:` and post-unit rebind must not pre-invalidate u_start=4
    direct_src = (
        "import contextlib\n"
        "def run_direct(compute, custom_unknown):\n"
        "    cm = contextlib.nullcontext\n"
        "    with cm() as cm:\n"
        "        val = compute()\n"
        "    cm = custom_unknown\n"
        "    return val\n"
    )
    direct_mod = ast.parse(direct_src)
    direct_fn = direct_mod.body[1]
    assert isinstance(direct_fn, ast.FunctionDef)
    prov_direct = _seed_scope_suppress_aliases(direct_fn, direct_mod, u_start=4)
    assert "cm" in prov_direct.non_suppressors
    prov_direct_late = _seed_scope_suppress_aliases(direct_fn, direct_mod, u_start=7)
    assert "cm" not in prov_direct_late.non_suppressors

@pytest.mark.parametrize(
    ("module_src", "expect_safe"),
    [
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_from_import_nullcontext",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with contextlib.nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_stdlib_module_alias",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "cm = contextlib.nullcontext()\n"
                "from evil import *\n"
                "def run(compute):\n"
                "    with cm:\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="star_import_invalidates_prior_instance_alias",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
                "from evil import *\n"
            ),
            False,
            id="post_def_star_import_invalidates_prior_nullcontext",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "if flag:\n"
                "    from evil import *\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            False,
            id="conditional_star_import_invalidates_prior_nullcontext",
        ),
        pytest.param(
            (
                "from evil import *\n"
                "from contextlib import nullcontext\n"
                "def run(compute):\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_explicit_import_after_star_import_remains_proven",
        ),
        pytest.param(
            (
                "from evil import *\n"
                "def run(compute):\n"
                "    from contextlib import nullcontext\n"
                "    with nullcontext():\n"
                "        v = compute()\n"
                "    cb = lambda: v\n"
                "    return cb()\n"
            ),
            True,
            id="control_function_local_import_after_module_star_import_remains_proven",
        ),
    ],
)
def test_wildcard_star_import_invalidates_prior_non_suppression_proofs(
    module_src: str,
    expect_safe: bool,
) -> None:
    """Verifies wildcard imports withdraw prior positive non-suppression proofs in strict and pedantic modes while preserving later explicit imports."""
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == {"strict": expect_safe, "pedantic": expect_safe}

@pytest.mark.parametrize(
    ("module_src", "expect_pre_bound"),
    [
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        if i > 0:\n"
                "            with cm(ValueError):\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            False,
            id="for_enclosing_loop_rebinding_after_unit_clears_pre_bound",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(conds, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    while conds:\n"
                "        if conds.pop():\n"
                "            with cm(ValueError):\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            False,
            id="while_enclosing_loop_rebinding_after_unit_clears_pre_bound",
        ),
        pytest.param(
            (
                "import contextlib\n"
                "def run(items, compute):\n"
                "    cm = contextlib.nullcontext\n"
                "    for i in items:\n"
                "        cm = contextlib.nullcontext\n"
                "        if i > 0:\n"
                "            with cm():\n"
                "                v = compute()\n"
                "            res = v + 1\n"
                "            return res\n"
                "        cm = contextlib.suppress\n"
            ),
            True,
            id="control_for_enclosing_loop_restores_cm_before_with",
        ),
    ],
)
def test_enclosing_loop_carried_ctx_manager_rebinding_invalidates_pre_unit_with(
    module_src: str,
    expect_pre_bound: bool,
) -> None:
    """Verifies cell traversal propagates loop-body context-manager rebinding into pre-unit with statements."""
    from pydoppelgangerhunt.fixer._df_cell_traversal import _analyze_enclosing_fn_cells

    tree = ast.parse(module_src)
    fn_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    unit_line = next(
        idx for idx, ln in enumerate(module_src.splitlines(), start=1) if "res = v + 1" in ln
    )
    for strict_mode in (False, True):
        clear_dataflow_cache()
        info = _analyze_enclosing_fn_cells(
            fn_node,
            (unit_line, unit_line, None, None),
            mod_tree=tree,
            strict_with_suppression=strict_mode,
        )
        assert ("v" in info.pre_bound) is expect_pre_bound

@pytest.mark.parametrize(
    ("fn_src", "expect_strict_safe", "expect_pedantic_safe"),
    [
        pytest.param(
            (
                "from contextlib import suppress\n"
                "def outer(items, compute):\n"
                "    cm = suppress\n"
                "    for cm in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm(ValueError):\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            False,
            False,
            id="enclosing_for_else_zero_trip_preserves_outer_suppressor",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, compute):\n"
                "    cm = nullcontext\n"
                "    for cm in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm():\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            True,
            False,
            id="enclosing_for_else_clears_nullcontext_proof_in_pedantic",
        ),
        pytest.param(
            (
                "from contextlib import nullcontext\n"
                "def outer(items, compute):\n"
                "    cm = nullcontext\n"
                "    for other in items:\n"
                "        pass\n"
                "    else:\n"
                "        with cm():\n"
                "            val = compute()\n"
                "        cb = lambda: val\n"
                "    return cb()\n"
            ),
            True,
            True,
            id="control_enclosing_for_else_unrelated_target_keeps_nullcontext_proof",
        ),
    ],
)
def test_enclosing_for_else_zero_trip_preserves_outer_suppressor_in_cell_traversal(
    fn_src: str,
    expect_strict_safe: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies enclosing for/else traversal preserves outer suppressor provenance on the zero-trip path into else."""
    lines = fn_src.splitlines()
    unit_line = next(
        idx for idx, ln in enumerate(lines, start=1) if ln.strip() == "cb = lambda: val"
    )
    tree = ast.parse(fn_src)
    for mode, expect_safe in (
        ("strict", expect_strict_safe),
        ("pedantic", expect_pedantic_safe),
    ):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": unit_line,
                "end": unit_line,
                "kind": "basic_block",
                "source_text": fn_src,
                "source_code": lines[unit_line - 1] + "\n",
                "closure_strictness": mode,
            },
            tree=tree,
        )
        assert bool(sc["is_control_flow_safe"]) is expect_safe, (mode, sc)

@pytest.mark.parametrize(
    ("body_src", "expect_safe_prefix_len"),
    [
        pytest.param(
            "w = True\nx: (w := custom_obj) = 1\nif w:\n    y = 1\n",
            2,
            id="annassign_with_value_annotation_walrus_clears_cur_bool",
        ),
        pytest.param(
            "w = True\nx: (w := custom_obj)\nif w:\n    y = 1\n",
            2,
            id="annassign_without_value_annotation_walrus_clears_cur_bool",
        ),
        pytest.param(
            "w = True\nw: (w := custom_obj) = True\nif w:\n    y = 1\n",
            2,
            id="annassign_self_target_annotation_walrus_clears_cur_bool",
        ),
        pytest.param(
            "x: (w := 1) = w\n",
            0,
            id="control_annassign_annotation_walrus_does_not_seed_cur_safe",
        ),
        pytest.param(
            "w = True\nx: int = 1\nif w:\n    y = 1\n",
            3,
            id="control_plain_annassign_preserves_cur_bool",
        ),
    ],
)
def test_annassign_annotation_walrus_clears_safe_bool_without_seeding_safe_names(
    body_src: str,
    expect_safe_prefix_len: int,
) -> None:
    """Verifies AnnAssign.annotation walrus targets discard cur_bool conservatively without seeding cur_safe."""
    from pydoppelgangerhunt.fixer._df_provenance_walk import (  # pylint: disable=import-outside-toplevel
        _split_suppress_with_body,
    )

    stmts = ast.parse(body_src).body
    safe_prefix, _risky_suffix = _split_suppress_with_body(
        stmts,
        safe_names=frozenset({"custom_obj", "int"}),
        strict_names=True,
    )
    assert len(safe_prefix) == expect_safe_prefix_len

@pytest.mark.parametrize(
    ("src", "marker", "expect_safe", "min_py"),
    [
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except ValueError:\n"
            "        cm = suppress\n"
            "    except KeyError:\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            True,
            (3, 9),
            id="try_mutually_exclusive_except_handlers_isolated",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except ValueError:\n"
            "        cm = suppress\n"
            "    else:\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            True,
            (3, 9),
            id="try_orelse_isolated_from_except_handler",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except ValueError:\n"
            "        cm = suppress\n"
            "    finally:\n"
            "        cm = nullcontext\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            True,
            (3, 9),
            id="try_finally_rebind_overrides_prior_except_mutation",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except ValueError:\n"
            "        cm = suppress\n"
            "    finally:\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            False,
            (3, 9),
            id="adversarial_try_finally_inherits_except_suppressor",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except (cm := suppress):\n"
            "        pass\n"
            "    except KeyError:\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            False,
            (3, 9),
            id="adversarial_prior_except_header_walrus_reaches_later_handler",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "cm = nullcontext\n"
            "try:\n"
            "    pass\n"
            "except KeyError:\n"
            "    def run(compute):\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n"
            "finally:\n"
            "    cm = suppress\n",
            "        cb = lambda: v",
            False,
            (3, 9),
            id="adversarial_def_in_except_observes_subsequent_finally_rebind",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    for _ in range(2):\n"
            "        try:\n"
            "            risky()\n"
            "        except ValueError:\n"
            "            cm = suppress\n"
            "        except KeyError:\n"
            "            with cm():\n"
            "                v = compute()\n"
            "            cb = lambda: v\n"
            "            return cb()\n",
            "            cb = lambda: v",
            False,
            (3, 9),
            id="adversarial_loop_enclosing_try_propagates_sibling_except_rebind",
        ),
        pytest.param(
            "from contextlib import nullcontext, suppress\n"
            "def run(risky, compute):\n"
            "    cm = nullcontext\n"
            "    try:\n"
            "        risky()\n"
            "    except* ValueError:\n"
            "        cm = suppress\n"
            "    except* KeyError:\n"
            "        with cm():\n"
            "            v = compute()\n"
            "        cb = lambda: v\n"
            "        return cb()\n",
            "        cb = lambda: v",
            False,
            (3, 11),
            id="adversarial_trystar_prior_handler_reaches_subsequent_handler",
        ),
    ],
)
def test_enclosing_try_clause_isolation_and_trystar_soundness(
    src: str,
    marker: str,
    expect_safe: bool,
    min_py: tuple[int, int],
) -> None:
    """Verifies Try isolates mutually exclusive except/orelse clauses while preserving finally, header, loop, and TryStar hazards."""
    if sys.version_info < min_py:
        pytest.skip(f"Requires Python {min_py[0]}.{min_py[1]}+")
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(src, marker)
    assert safety["strict"] is expect_safe, safety
    assert safety["pedantic"] is expect_safe, safety

@pytest.mark.parametrize(
    ("header_src", "with_stmt", "expect_strict_safe", "expect_pedantic_safe"),
    [
        pytest.param(
            "import builtins\n",
            "    with builtins.open(path):\n",
            True,
            True,
            id="module_import_builtins_open_trusted_in_function",
        ),
        pytest.param(
            "from io import open\n",
            "    with open(path):\n",
            True,
            True,
            id="module_from_io_import_open_trusted_in_function",
        ),
        pytest.param(
            "import builtins as b\n",
            "    with b.open(path):\n",
            True,
            True,
            id="module_import_builtins_alias_open_trusted_in_function",
        ),
        pytest.param(
            "def custom_open(p): return p\nopen = custom_open\n",
            "    with open(path):\n",
            False,
            False,
            id="neg_module_shadowed_open_rejected_in_both_modes",
        ),
        pytest.param(
            "from contextlib import suppress\n",
            "    open = suppress\n    with open(ValueError):\n",
            False,
            False,
            id="neg_function_local_shadowed_open_rejected_in_both_modes",
        ),
    ],
)
def test_module_level_builtins_and_io_open_imports_not_self_shadowed_in_functions(
    header_src: str,
    with_stmt: str,
    expect_strict_safe: bool,
    expect_pedantic_safe: bool,
) -> None:
    """Verifies module-level import builtins / from io import open are not treated as function-local shadows."""
    src = (
        header_src
        + "def run(path, compute):\n"
        + with_stmt
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    safety = _late_bound_capture_safety_by_mode(src, "    cb = lambda: v")
    assert safety["strict"] is expect_strict_safe, safety
    assert safety["pedantic"] is expect_pedantic_safe, safety

def test_module_suppress_seed_checkpoint_boundaries_and_sublinear_storage() -> None:
    """Verifies checkpointed _LazyModuleStmtSeedMap matches prefix/suffix/combined provenance across stride boundaries."""
    from pydoppelgangerhunt.fixer._df_module_seed import (  # pylint: disable=import-outside-toplevel
        _LazyModuleStmtSeedMap,
        _build_module_suppress_seed,
    )

    lines = [
        "import contextlib",
        "from contextlib import nullcontext",
        "cm_pre = nullcontext",
    ]
    for idx in range(70):
        lines.append(f"def helper_a_{idx}(): return {idx}")
    lines.append("def fn_early(step):\n    with cm_pre():\n        v = step()\n    return v")
    lines.append("cm_mid = nullcontext")
    for idx in range(70):
        lines.append(f"val_b_{idx} = {idx}")
    lines.append("def fn_mid(step):\n    with cm_mid():\n        v = step()\n    return v")
    lines.append("cm_pre = contextlib.suppress")
    for idx in range(70):
        lines.append(f"def helper_c_{idx}(): return {idx}")
    lines.append("def fn_late(step):\n    with cm_mid():\n        v = step()\n    return v")

    mod = ast.parse("\n".join(lines) + "\n")
    seed = _build_module_suppress_seed(mod)
    combined_map = seed.by_top_stmt_id
    prefix_map = seed.prefix_by_top_stmt_id
    suffix_map = seed.suffix_delta_by_top_stmt_id
    assert isinstance(combined_map, _LazyModuleStmtSeedMap)
    assert isinstance(prefix_map, _LazyModuleStmtSeedMap)
    assert isinstance(suffix_map, _LazyModuleStmtSeedMap)

    by_name = {
        stmt.name: stmt
        for stmt in mod.body
        if isinstance(stmt, ast.FunctionDef)
    }
    fn_early = by_name["fn_early"]
    fn_mid = by_name["fn_mid"]
    fn_late = by_name["fn_late"]

    early_pref = prefix_map[id(fn_early)]
    early_suf = suffix_map[id(fn_early)]
    early_comb = combined_map[id(fn_early)]
    assert "cm_pre" in early_pref.non_suppressors
    assert "cm_pre" in early_suf.suppressors
    assert "cm_pre" in early_comb.suppressors
    assert "cm_pre" not in early_comb.non_suppressors

    mid_comb = combined_map[id(fn_mid)]
    assert "cm_mid" in mid_comb.non_suppressors
    assert "cm_pre" in mid_comb.suppressors

    late_pref = prefix_map[id(fn_late)]
    late_comb = combined_map[id(fn_late)]
    assert "cm_mid" in late_pref.non_suppressors
    assert "cm_mid" in late_comb.non_suppressors
    assert "cm_pre" in late_pref.suppressors

    # Probe exact stride boundaries (0, 63, 64, 65, 127, 128, 129, last) without error
    for probe_idx in (0, 63, 64, 65, 127, 128, 129, len(mod.body) - 1):
        prov = combined_map[id(mod.body[probe_idx])]
        assert (
            prov.suppressors
            & (prov.non_suppressors | prov.non_suppressor_instances)
            == frozenset()
        )
    assert combined_map._prefix_checkpoints is not None
    assert len(combined_map._prefix_checkpoints) < len(mod.body) // 10

@pytest.mark.parametrize(
    ("helper_import", "run_import", "allow_entry", "expected_rebound_key", "expect_safe"),
    [
        pytest.param(
            "from . import tx as t\n    t.atomic = swallow\n",
            "from .tx import atomic\n",
            ".tx.atomic",
            (".tx", "atomic"),
            {"strict": False, "pedantic": False},
            id="single_dot_relative_import_rebind_revokes_allowlist",
        ),
        pytest.param(
            "from .. import tx as t\n    t.atomic = swallow\n",
            "from ..tx import atomic\n",
            "..tx.atomic",
            ("..tx", "atomic"),
            {"strict": False, "pedantic": False},
            id="double_dot_relative_import_rebind_revokes_allowlist",
        ),
        pytest.param(
            "from .pkg import tx as t\n    t.atomic = swallow\n",
            "from .pkg.tx import atomic\n",
            ".pkg.tx.atomic",
            (".pkg.tx", "atomic"),
            {"strict": False, "pedantic": False},
            id="nested_relative_import_rebind_revokes_allowlist",
        ),
        pytest.param(
            "from . import other as o\n    o.atomic = swallow\n",
            "from .tx import atomic\n",
            ".tx.atomic",
            None,
            {"strict": True, "pedantic": True},
            id="control_unrelated_relative_import_rebind_keeps_allowlist",
        ),
    ],
)
def test_scoped_relative_import_rebind_revokes_allowlisted_factory(
    helper_import: str,
    run_import: str,
    allow_entry: str,
    expected_rebound_key: tuple[str, str] | None,
    expect_safe: dict[str, bool],
) -> None:
    """Verifies scoped relative imports preserve dot-qualified names during exit-protocol factory rebind scanning."""
    from pydoppelgangerhunt.fixer._df_exit_protocol import (  # pylint: disable=import-outside-toplevel
        scan_exit_protocol_patches,
    )

    module_src = (
        run_import
        + "def helper(swallow):\n"
        + f"    {helper_import}"
        + "def run(compute):\n"
        + "    with atomic():\n"
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    clear_dataflow_cache()
    tree = ast.parse(module_src)
    scan = scan_exit_protocol_patches(tree)
    if expected_rebound_key is not None:
        assert expected_rebound_key in scan.rebound_attrs
    else:
        assert (".tx", "atomic") not in scan.rebound_attrs

    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == "    cb = lambda: v"
    )
    got: dict[str, bool] = {}
    for mode in ("strict", "pedantic"):
        clear_dataflow_cache()
        sc = inspect_single_unit_scope(
            {
                "start": line_num,
                "end": line_num,
                "kind": "basic_block",
                "source_text": module_src,
                "source_code": "    cb = lambda: v\n",
                "closure_strictness": mode,
                "non_suppressing_ctx_managers": {allow_entry},
            },
            tree=tree,
        )
        got[mode] = bool(sc["is_control_flow_safe"])
    assert got == expect_safe

@pytest.mark.parametrize(
    ("post_def_src", "expect_safe"),
    [
        pytest.param(
            "from pkg import SwallowErrors as alias_cm\ncm = alias_cm()\n",
            {"strict": False, "pedantic": False},
            id="post_def_top_level_suspicious_import_alias_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "if flag:\n"
                "    from pkg import SuppressErrors as alias_cm\n"
                "    cm = alias_cm()\n"
            ),
            {"strict": False, "pedantic": False},
            id="post_def_conditional_suspicious_import_alias_flags_strict_and_pedantic",
        ),
        pytest.param(
            "from pkg import BenignHelper as alias_cm\ncm = alias_cm()\n",
            {"strict": True, "pedantic": False},
            id="control_post_def_benign_import_alias_stays_heuristic_safe_in_strict",
        ),
    ],
)
def test_post_def_suspicious_import_alias_flags_earlier_function(
    post_def_src: str,
    expect_safe: dict[str, bool],
) -> None:
    """Verifies suspicious imported aliases defined after a function still flag earlier with-blocks in strict mode."""
    module_src = (
        "flag = False\n"
        "def run(step):\n"
        "    with cm():\n"
        "        v = step()\n"
        "    cb = lambda: v\n"
        "    return cb()\n"
        + post_def_src
    )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == expect_safe

@pytest.mark.parametrize(
    ("mutator_body", "use_nonlocal", "expect_safe"),
    [
        pytest.param(
            "    from contextlib import suppress as cm\n",
            False,
            {"strict": False, "pedantic": False},
            id="global_importfrom_suppress_as_cm_flags_strict_and_pedantic",
        ),
        pytest.param(
            "    from pkg import SwallowErrors as cm\n",
            False,
            {"strict": False, "pedantic": False},
            id="global_importfrom_suspicious_alias_as_cm_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "    class cm:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return True\n"
            ),
            True,
            {"strict": False, "pedantic": False},
            id="nonlocal_class_def_suppressing_exit_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "    @contextlib.contextmanager\n"
                "    def cm():\n"
                "        try:\n"
                "            yield\n"
                "        except Exception:\n"
                "            pass\n"
            ),
            False,
            {"strict": False, "pedantic": False},
            id="global_contextmanager_def_suppressor_flags_strict_and_pedantic",
        ),
        pytest.param(
            (
                "    class cm:\n"
                "        def __enter__(self):\n"
                "            return self\n"
                "        def __exit__(self, *a):\n"
                "            return False\n"
            ),
            True,
            {"strict": True, "pedantic": False},
            id="control_nonlocal_non_suppressing_class_stays_safe_in_strict",
        ),
    ],
)
def test_cross_scope_escaping_import_class_and_def_suppressors(
    mutator_body: str,
    use_nonlocal: bool,
    expect_safe: dict[str, bool],
) -> None:
    """Verifies global/nonlocal escaping ImportFrom, ClassDef, and def suppressors propagate into strict mode."""
    if use_nonlocal:
        module_src = (
            "import contextlib\n"
            "def outer(step):\n"
            "    cm = contextlib.nullcontext\n"
            "    def mutate():\n"
            "        nonlocal cm\n"
            + "".join(f"    {ln}\n" for ln in mutator_body.splitlines())
            + "    mutate()\n"
            + "    with cm():\n"
            + "        v = step()\n"
            + "    cb = lambda: v\n"
            + "    return cb()\n"
        )
    else:
        module_src = (
            "import contextlib\n"
            "cm = contextlib.nullcontext\n"
            "def mutate():\n"
            "    global cm\n"
            + mutator_body
            + "def run(step):\n"
            + "    mutate()\n"
            + "    with cm():\n"
            + "        v = step()\n"
            + "    cb = lambda: v\n"
            + "    return cb()\n"
        )
    safety = _late_bound_capture_safety_by_mode(module_src, "    cb = lambda: v")
    assert safety == expect_safe

@pytest.mark.parametrize(
    ("import_src", "with_src", "allowlist", "expect_pedantic_safe"),
    [
        pytest.param(
            "from ..pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"pkg.tx.atomic"}),
            False,
            id="absolute_exact_allowlist_does_not_match_double_dot_relative_import",
        ),
        pytest.param(
            "from ..pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"pkg.*.atomic"}),
            False,
            id="absolute_prefix_wildcard_does_not_match_double_dot_relative_import",
        ),
        pytest.param(
            "from .. import pkg\n",
            "    with pkg.tx.atomic():\n",
            frozenset({"pkg.tx.atomic"}),
            False,
            id="absolute_exact_allowlist_does_not_match_relative_module_attr_chain",
        ),
        pytest.param(
            "from .. import pkg\n",
            "    with pkg.tx.atomic():\n",
            frozenset({"pkg.*.atomic"}),
            False,
            id="absolute_prefix_wildcard_does_not_match_relative_module_attr_chain",
        ),
        pytest.param(
            "from .pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"pkg.tx.atomic"}),
            False,
            id="absolute_exact_allowlist_does_not_match_single_dot_relative_import",
        ),
        pytest.param(
            "from pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"pkg.tx.atomic"}),
            True,
            id="control_absolute_exact_allowlist_matches_absolute_import",
        ),
        pytest.param(
            "from pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"pkg.*.atomic"}),
            True,
            id="control_absolute_prefix_wildcard_matches_absolute_import",
        ),
        pytest.param(
            "from ..pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"..pkg.tx.atomic"}),
            True,
            id="control_relative_exact_allowlist_matches_same_dot_depth",
        ),
        pytest.param(
            "from ..pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({"..pkg.*.atomic"}),
            True,
            id="control_relative_prefix_wildcard_matches_same_dot_depth",
        ),
        pytest.param(
            "from ..pkg.tx import atomic\n",
            "    with atomic():\n",
            frozenset({".atomic"}),
            True,
            id="control_single_dot_suffix_allowlist_matches_relative_import",
        ),
        pytest.param(
            "from . import atomic as a\n",
            "    with a():\n",
            frozenset({"atomic"}),
            True,
            id="control_bare_name_allowlist_matches_aliased_relative_import",
        ),
    ],
)
def test_absolute_allowlist_does_not_match_relative_imports(
    import_src: str,
    with_src: str,
    allowlist: frozenset[str],
    expect_pedantic_safe: bool,
) -> None:
    """Verifies absolute allowlist paths and prefix wildcards never match relative imports while preserving explicit relative syntax."""
    module_src = (
        import_src
        + "def run(compute):\n"
        + with_src
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    line_num = next(
        idx
        for idx, ln in enumerate(module_src.splitlines(), start=1)
        if ln == "    cb = lambda: v"
    )
    clear_dataflow_cache()
    sc = inspect_single_unit_scope(
        {
            "start": line_num,
            "end": line_num,
            "kind": "basic_block",
            "source_text": module_src,
            "source_code": "    cb = lambda: v\n",
            "closure_strictness": "pedantic",
            "non_suppressing_ctx_managers": set(allowlist),
        }
    )
    assert bool(sc["is_control_flow_safe"]) is expect_pedantic_safe

@pytest.mark.parametrize(
    ("setup_src", "with_src", "allowlist", "expect_safe"),
    [
        pytest.param(
            "import myapp\nimport io\nmyapp.io.open = custom\n",
            "    with io.open('p'):\n",
            frozenset(),
            True,
            id="myapp_io_open_rebind_does_not_invalidate_stdlib_io_open",
        ),
        pytest.param(
            "import myapp\nfrom otherpkg import tx\nmyapp.tx.atomic = custom\n",
            "    with tx.atomic():\n",
            frozenset({"otherpkg.tx.atomic"}),
            True,
            id="myapp_tx_atomic_rebind_does_not_invalidate_otherpkg_tx_atomic",
        ),
        pytest.param(
            "import myapp\nfrom myapp import tx\nmyapp.tx.atomic = custom\n",
            "    with tx.atomic():\n",
            frozenset({"myapp.tx.atomic"}),
            False,
            id="myapp_tx_atomic_rebind_still_invalidates_same_pkg_tx",
        ),
        pytest.param(
            "import myapp\nfrom myapp.tx import atomic\nmyapp.tx.atomic = custom\n",
            "    with atomic():\n",
            frozenset({"myapp.tx.atomic"}),
            False,
            id="myapp_tx_atomic_rebind_still_invalidates_direct_from_import",
        ),
    ],
)
def test_non_stdlib_dotted_attribute_rebind_does_not_falsely_invalidate_other_modules(
    setup_src: str,
    with_src: str,
    allowlist: frozenset[str],
    expect_safe: bool,
) -> None:
    """Verifies dotted non-stdlib rebinds keep qualified targets and avoid cross-module false invalidation."""
    src = (
        setup_src
        + "def run(compute, custom):\n"
        + with_src
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(
        src,
        "    cb = lambda: v",
        non_suppressing_ctx_managers=allowlist,
    )
    assert safety["strict"] is expect_safe, safety
    assert safety["pedantic"] is expect_safe, safety

@pytest.mark.parametrize(
    ("setup_src", "with_src", "allowlist", "expect_pedantic_safe"),
    [
        pytest.param(
            "from evil import *\nfrom mypkg import MyTx\n",
            "    with MyTx():\n",
            frozenset({"mypkg.MyTx"}),
            True,
            id="explicit_from_import_after_star_import_trusted_by_allowlist",
        ),
        pytest.param(
            "from evil import *\nimport mypkg\n",
            "    with mypkg.MyTx():\n",
            frozenset({"mypkg.MyTx"}),
            True,
            id="explicit_module_import_after_star_import_trusted_by_allowlist",
        ),
        pytest.param(
            "from evil import *\n",
            "    with MyTx():\n",
            frozenset({"MyTx"}),
            False,
            id="neg_unqualified_symbol_after_star_import_fails_closed",
        ),
        pytest.param(
            "from mypkg import MyTx\nfrom evil import *\n",
            "    with MyTx():\n",
            frozenset({"mypkg.MyTx"}),
            False,
            id="neg_import_before_star_import_fails_closed",
        ),
        pytest.param(
            "from evil import *\nif cond:\n    from mypkg import MyTx\n",
            "    with MyTx():\n",
            frozenset({"mypkg.MyTx"}),
            False,
            id="neg_conditional_import_after_star_import_fails_closed",
        ),
        pytest.param(
            "from evil import *\n",
            "    with db.atomic():\n",
            frozenset({"*.atomic"}),
            False,
            id="neg_unimported_wildcard_receiver_after_star_import_fails_closed",
        ),
    ],
)
def test_explicit_import_after_star_import_matches_allowlist_while_prior_and_unqualified_fail_closed(
    setup_src: str,
    with_src: str,
    allowlist: frozenset[str],
    expect_pedantic_safe: bool,
) -> None:
    """Verifies explicit imports executed after a star import match allowlists while earlier or unqualified names fail closed."""
    src = (
        setup_src
        + "def run(compute, cond, db):\n"
        + with_src
        + "        v = compute()\n"
        + "    cb = lambda: v\n"
        + "    return cb()\n"
    )
    safety = _late_bound_capture_safety_by_mode(
        src,
        "    cb = lambda: v",
        non_suppressing_ctx_managers=allowlist,
    )
    assert safety["pedantic"] is expect_pedantic_safe, safety
