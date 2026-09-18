"""Unit tests for fixer variable scoping, control flow hazards, walrus, and pattern matching."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from pydoppelgangerhunt import (
    analyze_unit_variable_scope,
    generate_refactoring_patch,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _analyze_block_assignment,
    _block_terminates,
    _extract_deleted_names,
    _inspect_unit_scope,
    _normalize_receiver_attr_name,
    _normalize_receiver_attrs,
    _unfold_receiver_attribute,
    _walrus_assignment_in_expr,
)
from pydoppelgangerhunt.parser import harvest_file_units


def test_fixer_control_flow_and_side_effect_safety(tmp_path: Path) -> None:
    """Verifies control flow hazard detection, multi-value returns, and import propagation."""
    # 1. Test naked break and continue detection in compound block
    file_ctrl = tmp_path / "ctrl_hazards.py"
    file_ctrl.write_text(
        "def outer_loop(items):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            break\n"
        "        if x == 0:\n"
        "            continue\n",
        encoding="utf-8",
    )
    # Unit representing the if block with break
    u_break = {
        "file": str(file_ctrl),
        "start": 3,
        "end": 4,
        "name": "outer_loop:If",
        "kind": "compound_block",
    }
    scope_break = analyze_unit_variable_scope(u_break)
    assert "naked_break" in scope_break["control_flow_hazards"]
    assert scope_break["is_control_flow_safe"] is False

    helper_break = synthesize_shared_helper_code(u_break, u_break)
    assert "WARNING: Non-local control flow hazard detected" in helper_break
    assert "naked_break" in helper_break

    # Unit representing the for loop with internal break
    u_loop = {
        "file": str(file_ctrl),
        "start": 2,
        "end": 6,
        "name": "outer_loop:For",
        "kind": "compound_block",
    }
    scope_loop = analyze_unit_variable_scope(u_loop)
    # Inside loop, break and continue are enclosed, so not naked
    assert "naked_break" not in scope_loop["control_flow_hazards"]
    assert "naked_continue" not in scope_loop["control_flow_hazards"]

    # 2. Test embedded return in compound block
    file_ret = tmp_path / "embedded_ret.py"
    file_ret.write_text(
        "def compute(val: int) -> int:\n"
        "    if val < 0:\n"
        "        return -1\n"
        "    return val * 2\n",
        encoding="utf-8",
    )
    u_emb_ret = {
        "file": str(file_ret),
        "start": 2,
        "end": 3,
        "name": "compute:If",
        "kind": "compound_block",
    }
    scope_emb_ret = analyze_unit_variable_scope(u_emb_ret)
    assert "embedded_return" in scope_emb_ret["control_flow_hazards"]
    assert scope_emb_ret["is_control_flow_safe"] is False

    helper_emb_ret = synthesize_shared_helper_code(u_emb_ret, u_emb_ret)
    assert "embedded_return" in helper_emb_ret

    # 3. Test generator yield detection
    file_gen = tmp_path / "gen_func.py"
    file_gen.write_text(
        "def number_stream(n: int):\n"
        "    for i in range(n):\n"
        "        yield i\n",
        encoding="utf-8",
    )
    u_gen = {"file": str(file_gen), "start": 1, "end": 3, "name": "number_stream", "kind": "function"}
    scope_gen = analyze_unit_variable_scope(u_gen)
    assert scope_gen["has_yield"] is True
    assert "generator_yield" in scope_gen["control_flow_hazards"]
    helper_gen = synthesize_shared_helper_code(u_gen, u_gen)
    assert "-> Iterator[Any]:" in helper_gen

    # 4. Test multiple local reassignments and tuple return synthesis
    file_mut = tmp_path / "mut_locals.py"
    file_mut.write_text(
        "def process_batch(items, delta: int):\n"
        "    total = 0\n"
        "    count = 0\n"
        "    total += delta\n"
        "    count += 1\n"
        "    return total, count\n",
        encoding="utf-8",
    )
    # Unit representing the two reassignments in lines 4-5
    u_mut = {
        "file": str(file_mut),
        "start": 4,
        "end": 5,
        "name": "process_batch:stmts_4-5",
        "kind": "sliding_window",
    }
    scope_mut = analyze_unit_variable_scope(u_mut)
    assert "total" in scope_mut["outputs"]
    assert "count" in scope_mut["outputs"]
    assert len(scope_mut["outputs"]) >= 2

    helper_mut = synthesize_shared_helper_code(u_mut, u_mut)
    assert "Tuple[" in helper_mut
    assert "return total, count" in helper_mut
    assert "Call site:" in helper_mut
    assert "total, count = _shared_process_batch(...)" in helper_mut

    # 5. Test multi-value return in whole function (return a, b)
    u_fn_ret = {
        "file": str(file_mut),
        "start": 1,
        "end": 6,
        "name": "process_batch",
        "kind": "function",
    }
    scope_fn_ret = analyze_unit_variable_scope(u_fn_ret)
    assert "total" in scope_fn_ret["outputs"]
    assert "count" in scope_fn_ret["outputs"]

    # 6. Test local import tracking and include_imports
    file_imp = tmp_path / "local_imports.py"
    file_imp.write_text(
        "def parse_config(path: str) -> dict:\n"
        "    import json\n"
        "    from collections import deque\n"
        "    with open(path) as fh:\n"
        "        return json.load(fh)\n",
        encoding="utf-8",
    )
    u_imp = {"file": str(file_imp), "start": 1, "end": 5, "name": "parse_config", "kind": "function"}
    scope_imp = analyze_unit_variable_scope(u_imp)
    assert "import json" in scope_imp["local_imports"]
    assert "from collections import deque" in scope_imp["local_imports"]

    helper_with_imports = synthesize_shared_helper_code(u_mut, u_mut, include_imports=True)
    assert "from typing import" in helper_with_imports
    assert "Tuple" in helper_with_imports

    # 7. Test missing import propagation in generate_refactoring_patch
    file_target = tmp_path / "patch_target.py"
    file_target.write_text(
        '"""Target module docstring."""\n\n'
        'from __future__ import annotations\n\n'
        'def run_step(a: int, b: int) -> int:\n'
        '    return a + b\n',
        encoding="utf-8",
    )
    u_patch1 = {"file": str(file_target), "start": 5, "end": 6, "name": "run_step"}
    patch = generate_refactoring_patch([(0.95, u_mut, u_patch1)], repo_root=str(tmp_path))
    assert "+from typing import" in patch
    assert "Tuple" in patch

def test_conditional_variable_escape_prevention(tmp_path: Path) -> None:
    """Test that conditionally assigned variables are safely initialized to avoid runtime UnboundLocalError."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "def compute_summary(flag: bool):\n"
        "    if flag:\n"
        "        metric = 42\n"
        "    return flag\n"
    )
    src_file = tmp_path / "cond_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_block = {"file": str(src_file), "start": 2, "end": 3, "name": "compute_summary:If", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_block)
    assert "metric" in scope["conditional_outputs"]
    assert "flag" in scope["inputs"]

    helper_code = synthesize_shared_helper_code(u_block, u_block)
    assert "metric = None" in helper_code
    assert "Optional" in helper_code

    # Execute synthesized helper dynamically in namespace to verify no UnboundLocalError occurs when flag=False
    namespace: Dict[str, Any] = {}
    exec(helper_code, namespace)  # nosec
    helper_fn = namespace["_shared_compute_summary"]
    assert helper_fn(False) is None
    assert helper_fn(True) == 42

def test_async_coroutine_context_synthesis(tmp_path: Path) -> None:
    """Test that extracted blocks with await or async context synthesize async def and await call sites."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "async def process_item(item_id: int):\n"
        "    async with acquire_lock(item_id):\n"
        "        data = await fetch_remote(item_id)\n"
        "        return data\n"
    )
    src_file = tmp_path / "async_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_async = {"file": str(src_file), "start": 2, "end": 3, "name": "process_item:AsyncWith", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_async)
    assert scope["is_async"] is True
    assert "item_id" in scope["inputs"]

    helper_code = synthesize_shared_helper_code(u_async, u_async)
    assert helper_code.startswith("async def _shared_process_item")
    assert "await _shared_process_item" in helper_code
    # Must compile cleanly as an async function without SyntaxError
    compiled = compile(helper_code, "<test_async_helper>", "exec")
    assert compiled is not None

def test_global_and_nonlocal_scope_preservation(tmp_path: Path) -> None:
    """Test that variables altered via global/nonlocal preserve scope modifiers and avoid return-tuple leakage."""
    from pydoppelgangerhunt.fixer import analyze_unit_variable_scope, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    code = (
        "def outer():\n"
        "    global global_metric\n"
        "    nonlocal captured_scale\n"
        "    global_metric += 1\n"
        "    captured_scale *= 2\n"
    )
    src_file = tmp_path / "scope_mod_src.py"
    src_file.write_text(code, encoding="utf-8")

    u_scope = {"file": str(src_file), "start": 2, "end": 5, "name": "outer:compound", "kind": "compound_block"}
    scope = analyze_unit_variable_scope(u_scope)
    assert "global_metric" in scope["globals"]
    assert "captured_scale" in scope["nonlocals"]

    helper_code = synthesize_shared_helper_code(u_scope, u_scope)
    # Synthesis must be declined for nonlocal-dependent units to avoid compile-time SyntaxError
    assert helper_code == ""

    # Global-only units must preserve global modifiers and avoid return-tuple leakage
    code_glob = (
        "def outer():\n"
        "    global global_metric\n"
        "    global_metric += 1\n"
    )
    src_glob = tmp_path / "scope_glob_src.py"
    src_glob.write_text(code_glob, encoding="utf-8")
    u_glob = {"file": str(src_glob), "start": 2, "end": 3, "name": "outer:compound", "kind": "compound_block"}
    helper_glob = synthesize_shared_helper_code(u_glob, u_glob)
    assert "global global_metric" in helper_glob
    assert "return" not in helper_glob

def test_synthesize_helper_with_pragma_and_import_preservation(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code and local import capture preserve pragmas."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
        synthesize_shared_helper_code,
    )

    f1 = tmp_path / "mod_a.py"
    f2 = tmp_path / "mod_b.py"

    code_a = (
        "def process_a(val):\n"
        "    from typing_extensions import Buffer  # type: ignore\n"
        "    # Step 1: calculate\n"
        "    result = val * 10  # type: ignore[operator]\n"
        "    return result\n"
    )
    code_b = (
        "def process_b(val):\n"
        "    # Step 1: calculate\n"
        "    result = val * 10  # type: ignore[operator]\n"
        "    return result\n"
    )
    f1.write_text(code_a, encoding="utf-8")
    f2.write_text(code_b, encoding="utf-8")

    u1 = {"file": str(f1), "start": 1, "end": 5, "name": "process_a", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 5, "name": "process_b", "kind": "function"}

    # Scope analysis captures local import with # type: ignore intact
    scope = analyze_unit_variable_scope(u1)
    assert any("# type: ignore" in imp for imp in scope.get("local_imports", []))

    # Shared helper synthesis preserves comments and # type: ignore pragma
    helper = synthesize_shared_helper_code(u1, u2, preserve_pragmas=True)
    assert "# type: ignore[operator]" in helper

def test_mixed_binding_preserves_cls_argument(tmp_path: Path) -> None:
    """Verifies that mixed instance and class binding retains cls in helper call arguments."""
    f = tmp_path / "factory.py"
    code = (
        "class WidgetFactory:\n"
        "    def build_item(self, cls: type, item_id: int) -> object:\n"
        "        self.count += 1\n"
        "        return cls(self.prefix, item_id)\n"
    )
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 4, "name": "build_item", "kind": "function", "enclosing_class": "WidgetFactory"}
    scope = analyze_unit_variable_scope(u)
    assert scope.get("binding_kind") == "mixed"
    assert scope.get("has_instance_binding") is True
    assert scope.get("has_class_binding") is True

def test_closure_returns_not_classified_as_control_flow_hazards(tmp_path: Path) -> None:
    """Verifies that whole closure units with return statements are not flagged as embedded_return hazards."""
    code = (
        "def factory_calc(multiplier: int):\n"
        "    def step_fn(val: int) -> int:\n"
        "        offset = 10\n"
        "        return val * multiplier + offset\n"
        "    return step_fn\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    u = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "factory_calc:step_fn",
        "kind": "closure",
    }
    info = _inspect_unit_scope(u, repo_root=str(tmp_path))
    assert info["is_control_flow_safe"] is True
    assert "embedded_return" not in info["control_flow_hazards"]

    helper = synthesize_shared_helper_code(u, u, repo_root=str(tmp_path))
    assert "WARNING: Non-local control flow hazard" not in helper

def test_nested_closure_control_flow_and_returns_do_not_leak_to_outer(tmp_path: Path) -> None:
    """Verifies that returns, yields, and awaits in nested closures do not leak into outer function scope."""
    code = (
        "def outer_factory(base: int):\n"
        "    multiplier = 2\n"
        "    async def inner_worker(val: int) -> int:\n"
        "        await do_async_op()\n"
        "        if val < 0:\n"
        "            return 0\n"
        "        return val * multiplier\n"
        "    return inner_worker\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1, harvest_closures=True)
    outer_u = next(u for u in units if u["name"] == "outer_factory")
    scope = analyze_unit_variable_scope(outer_u, repo_root=str(tmp_path))

    # Outer factory must be synchronous, non-generator, and only return inner_worker
    assert scope["is_async"] is False
    assert scope["has_yield"] is False
    assert scope["outputs"] == ["inner_worker"]

def test_lambda_parameter_scoping_no_leakage(tmp_path: Path) -> None:
    """Verifies that parameters of lambda expressions do not leak into outer inputs."""
    code = (
        "def transform_items(items: list[int]) -> list[int]:\n"
        "    return sorted(items, key=lambda x: x.val)\n"
    )
    f = tmp_path / "lambda_mod.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 2, "name": "transform_items"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "items" in scope["inputs"]
    assert "x" not in scope["inputs"]

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_pattern_bound_variables_not_treated_as_inputs(tmp_path: Path) -> None:
    """Verifies that pattern-bound variables in match statements are recognized as stores, not inputs."""
    code = (
        "def process_data(data: list) -> int:\n"
        "    match data:\n"
        "        case [head, *tail]:\n"
        "            total = head + len(tail)\n"
        "            return total\n"
        "        case _:\n"
        "            return 0\n"
    )
    f = tmp_path / "matcher_unit.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 7, "name": "process_data", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "data" in scope["inputs"]
    assert "head" not in scope["inputs"]
    assert "tail" not in scope["inputs"]
    assert "total" in scope["outputs"]

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_definite_assignment_requires_irrefutable_default() -> None:
    """Verifies that match block assignments require an irrefutable pattern to be definite."""
    code_no_default = (
        "match x:\n"
        "    case 1:\n"
        "        val = 10\n"
        "    case 2:\n"
        "        val = 20\n"
    )
    tree1 = ast.parse(code_no_default)
    def1, cond1 = _analyze_block_assignment(tree1.body)
    assert "val" not in def1
    assert "val" in cond1

    code_with_default = (
        "match x:\n"
        "    case 1:\n"
        "        val = 10\n"
        "    case _:\n"
        "        val = 20\n"
    )
    tree2 = ast.parse(code_with_default)
    def2, cond2 = _analyze_block_assignment(tree2.body)
    assert "val" in def2
    assert "val" not in cond2

def test_inner_function_decorators_and_defaults_lexical_scoping(tmp_path: Path) -> None:
    """Verifies that variables in inner function decorators and defaults are captured into outer inputs."""
    code = (
        "def make_pipeline(outer_mult: int, threshold: int) -> object:\n"
        "    @transform_decorator\n"
        "    def step(x: int = threshold) -> int:\n"
        "        return x * outer_mult\n"
        "    return step\n"
    )
    f = tmp_path / "pipeline.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 5, "name": "make_pipeline", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "outer_mult" in scope["inputs"]
    assert "threshold" in scope["inputs"]
    assert "transform_decorator" in scope["inputs"]
    assert "x" not in scope["inputs"]

def test_aug_assign_compound_block_inputs_and_outputs(tmp_path: Path) -> None:
    """Verifies that AugAssign targets without prior assignment in the block are tracked as both inputs and outputs."""
    code = (
        "def compute_a(items: list) -> int:\n"
        "    total = 0\n"
        "    total += len(items)\n"
        "    return total\n\n"
        "def compute_b(items: list) -> int:\n"
        "    total = 0\n"
        "    total += len(items)\n"
        "    return total\n"
    )
    f = tmp_path / "calc.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": str(f), "start": 3, "end": 3, "name": "aug_a", "kind": "compound_block"}
    u2 = {"file": str(f), "start": 8, "end": 8, "name": "aug_b", "kind": "compound_block"}

    scope = analyze_unit_variable_scope(u1, repo_root=str(tmp_path))
    assert "total" in scope["inputs"]
    assert "total" in scope["outputs"]
    assert "items" in scope["inputs"]

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "total: Any" in helper
    assert "return total" in helper

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "total = _shared_aug_a" in patch
    assert "total," in patch or "(total," in patch or "(items, total)" in patch or "(total)" in patch

def test_comprehension_token_span_and_bare_call_replacement(tmp_path: Path) -> None:
    """Verifies that comprehension units return the evaluated expression and replace exact token spans."""
    code = (
        "def make_lists(items: list):\n"
        "    a_list = [k * 2 for k in items if k > 0]\n"
        "    b_list = [k * 2 for k in items if k > 0]\n"
        "    return a_list, b_list\n"
    )
    f = tmp_path / "comps.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=1, min_tokens=5, comprehensions=True)
    comps = [u for u in units if u["kind"] == "comprehension"]
    assert len(comps) == 2
    c1, c2 = comps[0], comps[1]

    scope = analyze_unit_variable_scope(c1, repo_root=str(tmp_path))
    assert "items" in scope["inputs"]
    assert "k" not in scope["inputs"]
    assert "k" not in scope["outputs"]

    helper = synthesize_shared_helper_code(c1, c2, repo_root=str(tmp_path))
    assert "return [k * 2 for k in items if k > 0]" in helper
    assert "return k" not in helper

    patch = generate_refactoring_patch([(1.0, c1, c2)], repo_root=str(tmp_path), replace_clones=True)
    assert "a_list = _shared_" in patch
    assert "b_list = _shared_" in patch

def test_annassign_without_value_not_treated_as_definite_assignment(tmp_path: Path) -> None:
    """Verifies that type annotations without values do not prevent conditional output initialization."""
    from pydoppelgangerhunt.fixer import _analyze_block_assignment, synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

    tree = ast.parse("res: int\nif flag:\n    res = 42\n")
    definite, conditional = _analyze_block_assignment(tree.body)
    assert "res" not in definite
    assert "res" in conditional

    code = "res: int\nif flag:\n    res = 42\n"
    f = tmp_path / "ann.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": "ann.py", "start": 1, "end": 3, "name": "block", "kind": "compound_block"}
    helper = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    assert "res = None" in helper

def test_comprehension_multi_generator_scoping(tmp_path: Path) -> None:
    """Verifies that multi-generator comprehensions do not leak inner loop variables into inputs."""
    code = "matrix = [[1, 2], [3, 4]]\nflat = [x for row in matrix for x in row]\n"
    f = tmp_path / "comp_matrix.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "comp_matrix.py",
        "start": 2,
        "end": 2,
        "start_col": 7,
        "end_col": 42,
        "name": "flat_comp",
        "kind": "comprehension",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "matrix" in scope["inputs"]
    assert "row" not in scope["inputs"]
    assert "x" not in scope["inputs"]

def test_walrus_operator_comprehension_enclosing_scope(tmp_path: Path) -> None:
    """Verifies that walrus expressions in comprehensions register as stores in the enclosing scope."""
    code = "def process_data(items):\n    squared = [val := x * 2 for x in items]\n    return squared, val\n"
    f = tmp_path / "walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {"file": "walrus.py", "start": 1, "end": 3, "name": "process_data", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "val" in scope["locals"]
    assert "val" in scope["outputs"]
    assert "items" in scope["inputs"]
    assert "val" not in scope["inputs"]

def test_async_comprehension_detected_as_async(tmp_path: Path) -> None:
    """Verifies that an async comprehension correctly tags its unit scope as async."""
    code = "async def fetch():\n    results = [x async for x in aiter]\n    return results\n"
    f = tmp_path / "async_comp.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "async_comp.py",
        "start": 2,
        "end": 2,
        "start_col": 14,
        "end_col": 38,
        "name": "async_listcomp",
        "kind": "comprehension",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert scope["is_async"] is True

def test_except_handler_as_err_variable_not_treated_as_input_or_output(tmp_path: Path) -> None:
    """Verifies that 'except Exception as err:' does not leak 'err' into inputs or subroutine outputs."""
    code = (
        "def run_task():\n"
        "    try:\n"
        "        action()\n"
        "    except Exception as err:\n"
        "        logger.error(err)\n"
    )
    f = tmp_path / "except_scope.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "except_scope.py",
        "start": 2,
        "end": 5,
        "name": "run_task:try_block",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "err" not in scope["inputs"]
    assert "err" not in scope["outputs"]
    assert "logger" in scope["inputs"]

def test_nested_closure_comprehension_owner_attribution(tmp_path: Path) -> None:
    """Verifies that comprehensions inside nested closures are accurately attributed to their lexical parent path."""
    code = (
        "def outer():\n"
        "    def inner():\n"
        "        return [x for x in [1, 2, 3]]\n"
        "    return inner\n"
    )
    f = tmp_path / "nested_comp.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(
        str(f),
        repo_root=str(tmp_path),
        min_lines=1,
        min_tokens=1,
        harvest_closures=True,
        comprehensions=True,
    )
    comp_units = [u for u in units if u.get("kind") == "comprehension"]
    assert len(comp_units) == 1
    assert comp_units[0]["name"].startswith("outer:inner:listcomp_L")

def test_attribute_del_tracked_as_attribute_written(tmp_path: Path) -> None:
    """Verifies that 'del self.attr' is tracked in attrs_written and triggers instance binding."""
    code = (
        "class CacheManager:\n"
        "    def clear_cache(self):\n"
        "        del self._cache\n"
    )
    f = tmp_path / "del_attr.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "del_attr.py",
        "start": 2,
        "end": 3,
        "name": "CacheManager:clear_cache",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "self._cache" in scope["attrs_written"]
    assert scope["has_instance_binding"] is True
    assert scope["binding_kind"] == "instance"

def test_deleted_variable_not_treated_as_subroutine_output(tmp_path: Path) -> None:
    """Verifies that a locally deleted variable ('del x') is not returned in subroutine outputs."""
    code = (
        "def do_calc():\n"
        "    x = 10\n"
        "    del x\n"
        "    y = 20\n"
    )
    f = tmp_path / "del_var.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "del_var.py",
        "start": 2,
        "end": 4,
        "name": "do_calc:stmts",
        "kind": "sliding_window",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "x" not in scope["outputs"]
    assert "y" in scope["outputs"]

def test_local_import_not_treated_as_free_var_input_or_subroutine_output(tmp_path: Path) -> None:
    """Verifies that locally imported modules and functions are not treated as inputs or subroutine outputs."""
    code = (
        "def compute(val):\n"
        "    import math\n"
        "    from os.path import join\n"
        "    res = math.sqrt(val)\n"
        "    path = join('dir', str(res))\n"
    )
    f = tmp_path / "local_import.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "local_import.py",
        "start": 2,
        "end": 5,
        "name": "compute:stmts",
        "kind": "sliding_window",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "math" not in scope["inputs"]
    assert "join" not in scope["inputs"]
    assert "val" in scope["inputs"]
    assert "math" not in scope["outputs"]
    assert "join" not in scope["outputs"]
    assert "res" in scope["outputs"]
    assert "path" in scope["outputs"]

def test_for_loop_iter_evaluated_before_target_binding(tmp_path: Path) -> None:
    """Verifies that variables referenced in loop iterators are evaluated before target assignment."""
    code = (
        "def process(items):\n"
        "    for item in transform(items, item):\n"
        "        print(item)\n"
    )
    f = tmp_path / "for_order.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "for_order.py",
        "start": 2,
        "end": 3,
        "name": "process:for_loop",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "item" in scope["inputs"]
    assert "transform" in scope["inputs"]
    assert "items" in scope["inputs"]

def test_annassign_type_annotation_does_not_leak_into_inputs(tmp_path: Path) -> None:
    """Verifies that type annotations on local variables are not treated as runtime inputs."""
    code = (
        "def run():\n"
        "    data: CustomType = fetch_data()\n"
        "    entries: List[str] = parse(data)\n"
        "    return entries\n"
    )
    f = tmp_path / "ann_leak.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "ann_leak.py",
        "start": 1,
        "end": 4,
        "name": "run",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "CustomType" not in scope["inputs"]
    assert "List" not in scope["inputs"]
    assert "fetch_data" in scope["inputs"]
    assert "parse" in scope["inputs"]
    assert "entries" in scope["outputs"]

def test_while_test_walrus_treated_as_definite_assignment(tmp_path: Path) -> None:
    """Verifies that walrus expressions in while loop tests are marked as definite assignments."""
    code = (
        "def drain_stream(stream):\n"
        "    while (chunk := stream.read(1024)):\n"
        "        process(chunk)\n"
        "    return chunk\n"
    )
    f = tmp_path / "while_walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "while_walrus.py",
        "start": 2,
        "end": 3,
        "name": "drain_stream:while_loop",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "chunk" in scope["definite_stores"]

def test_nested_function_walrus_does_not_leak_to_outer_scope(tmp_path: Path) -> None:
    """Verifies that walrus expressions inside nested functions do not register in outer unit stores."""
    code = (
        "def outer(data):\n"
        "    def inner():\n"
        "        if (inner_val := process(data)):\n"
        "            return inner_val\n"
        "    return inner()\n"
    )
    f = tmp_path / "nested_walrus.py"
    f.write_text(code, encoding="utf-8")

    u = {
        "file": "nested_walrus.py",
        "start": 1,
        "end": 5,
        "name": "outer",
        "kind": "function",
    }
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "inner_val" not in scope["locals"]
    assert "inner_val" not in scope["outputs"]

def test_assign_rhs_evaluation_order_captures_inputs(tmp_path: Path) -> None:
    """Verifies that Assign statements visit RHS values before LHS targets, capturing free variables."""
    code = (
        "def compute(item: int):\n"
        "    total = total + item\n"
        "    a, b = b, a\n"
        "    return total, a, b\n"
    )
    f = tmp_path / "assign_order.py"
    f.write_text(code, encoding="utf-8")

    u_sub = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "compute:sub",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u_sub, repo_root=str(tmp_path))
    assert "total" in scope["inputs"]
    assert "b" in scope["inputs"]
    assert "a" in scope["inputs"]
    assert "item" in scope["inputs"]

def test_scope_hierarchy_visitor_decorator_and_default_isolation(tmp_path: Path) -> None:
    """Verifies that decorator expressions and default arguments are scoped in enclosing outer frames."""
    code = (
        "@wrap([x for x in outer_list])\n"
        "class Service:\n"
        "    @route([y for y in class_list])\n"
        "    def handle(self, val=[z for z in def_list]):\n"
        "        pass\n"
    )
    f = tmp_path / "hierarchy.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=1, min_tokens=1, comprehensions=True)
    comps = {u["name"]: u for u in units if u["kind"] == "comprehension"}

    outer_c = comps.get("module:listcomp_L1")
    assert outer_c is not None
    assert outer_c.get("enclosing_class") is None

    class_c = comps.get("Service:listcomp_L3")
    assert class_c is not None
    assert class_c.get("enclosing_class") == "Service"
    assert "handle" not in class_c["name"]

    def_c = comps.get("Service:listcomp_L4")
    assert def_c is not None
    assert def_c.get("enclosing_class") == "Service"
    assert "handle" not in def_c["name"]

def test_analyze_block_assignment_short_circuit_walrus() -> None:
    """Verifies that short-circuit walrus expressions are correctly identified as conditional."""
    code_and = "if cond and (x := 1):\n    pass\n"
    d1, c1 = _analyze_block_assignment(ast.parse(code_and).body)
    assert "x" in c1
    assert "x" not in d1

    code_or = "if cond or (x := 1):\n    pass\n"
    d2, c2 = _analyze_block_assignment(ast.parse(code_or).body)
    assert "x" in c2
    assert "x" not in d2

    code_ifexp = "val = (x := 1) if cond else 0\n"
    d3, c3 = _analyze_block_assignment(ast.parse(code_ifexp).body)
    assert "val" in d3
    assert "x" in c3
    assert "x" not in d3

def test_analyze_block_assignment_unconditional_walrus() -> None:
    """Verifies that unconditionally evaluated walrus expressions are identified as definite."""
    code_assign = "val = (x := 1)\n"
    d1, c1 = _analyze_block_assignment(ast.parse(code_assign).body)
    assert "val" in d1
    assert "x" in d1
    assert "x" not in c1

    code_head = "if (x := 1) and cond:\n    pass\n"
    d2, c2 = _analyze_block_assignment(ast.parse(code_head).body)
    assert "x" in d2
    assert "x" not in c2

def test_analyze_block_assignment_with_target_variables() -> None:
    """Verifies that with/async with optional_vars targets are captured as definite assignments."""
    code_with = "with open('file') as fh:\n    data = fh.read()\n"
    d, _ = _analyze_block_assignment(ast.parse(code_with).body)
    assert "fh" in d

def test_analyze_block_assignment_try_finally_definite() -> None:
    """Verifies that statements in finally: blocks are recognized as definite assignments."""
    code_try = "try:\n    risky()\nfinally:\n    cleaned = True\n"
    d, _ = _analyze_block_assignment(ast.parse(code_try).body)
    assert "cleaned" in d

def test_nested_closure_shadowed_self_param_isolation(tmp_path: Path) -> None:
    """Verifies that local functions taking a parameter named 'self' do not leak receiver attributes."""
    code = (
        "class Service:\n"
        "    def process(self, data: int) -> int:\n"
        "        def inner(self, v: int) -> int:\n"
        "            return self.shadowed + v\n"
        "        return inner(self, data)\n"
    )
    f = tmp_path / "shadowed_self.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 5, "name": "process", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.shadowed" not in scope["attrs_read"]
    assert "self.shadowed" not in scope["instance_attrs"]

def test_nonlocal_variable_mutation_recorded_in_stores(tmp_path: Path) -> None:
    """Verifies that nonlocal mutations within inner closures are recorded in unit stores and outputs."""
    code = (
        "def outer(limit: int) -> int:\n"
        "    total = 0\n"
        "    def increment(step: int) -> None:\n"
        "        nonlocal total\n"
        "        total += step\n"
        "    increment(limit)\n"
        "    return total\n"
    )
    f = tmp_path / "nonlocal_stores.py"
    f.write_text(code, encoding="utf-8")
    u_block = {
        "file": str(f),
        "start": 3,
        "end": 6,
        "name": "outer:sub",
        "kind": "compound_block",
    }
    raw_info = _inspect_unit_scope(u_block, repo_root=str(tmp_path))
    assert "total" in raw_info["stores"]
    assert "total" in raw_info["outputs"]

    scope = analyze_unit_variable_scope(u_block, repo_root=str(tmp_path))
    assert "total" in scope["outputs"]
    assert "total" in scope["nonlocals"]

def test_ast_delete_in_block_assignment() -> None:
    """Verifies that ast.Delete discards variables from definite stores unless reassigned."""
    tree1 = ast.parse("x = 1\ny = 2\ndel x\n")
    d1, _ = _analyze_block_assignment(tree1.body)
    assert "x" not in d1
    assert "y" in d1

    tree2 = ast.parse("x = 1\ndel x\nx = 3\n")
    d2, _ = _analyze_block_assignment(tree2.body)
    assert "x" in d2

    tree_del_tuple = ast.parse("a = 1\nb = 2\ndel a, b\n")
    d3, _ = _analyze_block_assignment(tree_del_tuple.body)
    assert "a" not in d3
    assert "b" not in d3

def test_inner_closure_delete_isolation(tmp_path: Path) -> None:
    """Verifies that del on an inner closure local variable does not delete outer variables of the same name."""
    code = (
        "def compute() -> int:\n"
        "    x = 10\n"
        "    def inner_cleanup() -> None:\n"
        "        x = 20\n"
        "        del x\n"
        "    inner_cleanup()\n"
        "    return x\n"
    )
    f = tmp_path / "inner_del.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 7, "name": "compute", "kind": "function"}
    raw_info = _inspect_unit_scope(u, repo_root=str(tmp_path))
    assert "x" not in raw_info.get("deleted_names", set())

    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "x" in scope["outputs"]

def test_if_else_terminating_branch_definite_assignment(tmp_path: Path) -> None:
    """Verifies that if-else with a terminating branch correctly tracks definite assignment."""
    code = (
        "def parse_or_raise(flag: bool, num: int) -> int:\n"
        "    if flag:\n"
        "        val = num * 2\n"
        "    else:\n"
        "        raise ValueError('invalid flag')\n"
        "    return val\n"
    )
    f = tmp_path / "terminating_else.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 6, "name": "parse_or_raise", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "val" in scope["definite_stores"]
    assert "val" not in scope["conditional_outputs"]

    helper = synthesize_shared_helper_code(u, u, repo_root=str(tmp_path))
    assert "Optional[" not in helper
    assert "val = None" not in helper

def test_if_terminating_no_else_conditional_isolation(tmp_path: Path) -> None:
    """Verifies that if-body assignments in a terminating if block do not leak into conditional outputs."""
    code = (
        "def guard_step(error: bool, data: str) -> str:\n"
        "    if error:\n"
        "        err_msg = f'Failed on {data}'\n"
        "        raise RuntimeError(err_msg)\n"
        "    res = data.upper()\n"
        "    return res\n"
    )
    f = tmp_path / "terminating_guard.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 6, "name": "guard_step", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "err_msg" not in scope["conditional_outputs"]
    assert "res" in scope["definite_stores"]
    assert "res" not in scope["conditional_outputs"]

def test_with_block_definite_assignment(tmp_path: Path) -> None:
    """Verifies that sequential assignments inside with blocks are recorded as definite stores."""
    code = (
        "def read_record(path: str) -> str:\n"
        "    with open(path, 'r', encoding='utf-8') as fh:\n"
        "        content = fh.read()\n"
        "    return content\n"
    )
    f = tmp_path / "with_block.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 4, "name": "read_record", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert "content" in scope["definite_stores"]
    assert "content" not in scope["conditional_outputs"]

def test_try_terminating_handlers_definite_assignment(tmp_path: Path) -> None:
    """Verifies that try-except with terminating handlers or dual assignments tracks definite assignment."""
    code1 = (
        "def safe_convert(s: str) -> int:\n"
        "    try:\n"
        "        val = int(s)\n"
        "    except ValueError:\n"
        "        return 0\n"
        "    return val\n"
    )
    f1 = tmp_path / "try_term.py"
    f1.write_text(code1, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 6, "name": "safe_convert", "kind": "function"}
    scope1 = analyze_unit_variable_scope(u1, repo_root=str(tmp_path))
    assert "val" in scope1["definite_stores"]
    assert "val" not in scope1["conditional_outputs"]

    code2 = (
        "def fallback_convert(s: str) -> int:\n"
        "    try:\n"
        "        val = int(s)\n"
        "    except ValueError:\n"
        "        val = -1\n"
        "    return val\n"
    )
    f2 = tmp_path / "try_fallback.py"
    f2.write_text(code2, encoding="utf-8")
    u2 = {"file": str(f2), "start": 1, "end": 6, "name": "fallback_convert", "kind": "function"}
    scope2 = analyze_unit_variable_scope(u2, repo_root=str(tmp_path))
    assert "val" in scope2["definite_stores"]
    assert "val" not in scope2["conditional_outputs"]

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_terminating_case_definite_assignment() -> None:
    """Verifies that Match with a terminating case preserves definite assignment across non-terminating branches."""
    code = (
        "match cmd:\n"
        "    case 'start':\n"
        "        res = 1\n"
        "    case 'stop':\n"
        "        res = 2\n"
        "    case _:\n"
        "        raise ValueError('unknown')\n"
    )
    tree = ast.parse(code)
    def_assigned, _ = _analyze_block_assignment(tree.body)
    assert "res" in def_assigned

def test_extract_deleted_names_inside_except_handler() -> None:
    """Verifies that _extract_deleted_names extracts deletions inside except handlers and preserves inner closures."""
    code = (
        "try:\n"
        "    val = 10\n"
        "except ValueError:\n"
        "    del err_val\n"
        "    def inner():\n"
        "        del inner_scoped\n"
    )
    tree = ast.parse(code)
    dels = _extract_deleted_names(tree.body)
    assert "err_val" in dels
    assert "inner_scoped" not in dels

def test_try_finally_deletion_discards_from_definite() -> None:
    """Verifies that _analyze_block_assignment discards variables deleted in finally blocks."""
    code = (
        "x = 10\n"
        "y = 20\n"
        "try:\n"
        "    z = 30\n"
        "finally:\n"
        "    del x\n"
    )
    tree = ast.parse(code)
    definite, _ = _analyze_block_assignment(tree.body)
    assert "x" not in definite
    assert "y" in definite

def test_chained_comparison_walrus_short_circuit() -> None:
    """Verifies that _walrus_assignment_in_expr treats chained comparison tails as conditional."""
    code = "res = (a < (b := 1) < (c := 2))"
    tree = ast.parse(code)
    assign_stmt = tree.body[0]
    assert isinstance(assign_stmt, ast.Assign)
    definite, conditional = _walrus_assignment_in_expr(assign_stmt.value)
    assert "b" in definite
    assert "c" in conditional

def test_subroutine_with_local_function_scope_isolation(tmp_path: Path) -> None:
    """Verifies that compound blocks with local helper functions isolate arguments and returns."""
    code = (
        "def outer(val):\n"
        "    def helper(a):\n"
        "        return a * 2\n"
        "    result = helper(val)\n"
        "    return result\n"
    )
    f = tmp_path / "sub_fn.py"
    f.write_text(code, encoding="utf-8")
    u_block = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "outer:2-4",
        "kind": "compound_block",
    }
    scope = analyze_unit_variable_scope(u_block, repo_root=str(tmp_path))
    assert "val" in scope["inputs"]
    assert "a" not in scope["inputs"]
    assert "helper" not in scope["inputs"]
    assert "result" in scope["outputs"]
    assert "helper" in scope["outputs"]
    assert scope["is_control_flow_safe"] is True
    assert "embedded_return" not in scope["control_flow_hazards"]

def test_with_block_unconditional_termination_and_deletion() -> None:
    """Verifies that terminating with blocks are recognized and deleted names in with blocks are discarded."""
    term_code = "with lock:\n    return val\nx = 1\n"
    tree_term = ast.parse(term_code)
    assert _block_terminates(tree_term.body) is True

    del_code = "x = 1\nwith lock:\n    del x\n"
    tree_del = ast.parse(del_code)
    definite, conditional = _analyze_block_assignment(tree_del.body)
    assert "x" not in definite
    assert "x" not in conditional

def test_loop_block_deletion_promoted_to_conditional() -> None:
    """Verifies that variable deletions inside for and while loops are discarded from definite."""
    for_code = "x = 1\nfor i in items:\n    del x\n"
    tree_for = ast.parse(for_code)
    def_for, cond_for = _analyze_block_assignment(tree_for.body)
    assert "x" not in def_for
    assert "x" in cond_for

    while_code = "x = 1\nwhile cond:\n    del x\n"
    tree_while = ast.parse(while_code)
    def_while, cond_while = _analyze_block_assignment(tree_while.body)
    assert "x" not in def_while
    assert "x" in cond_while

def test_walrus_assignment_in_call_keywords_and_targets() -> None:
    """Verifies walrus expressions inside function call keywords and assignment/augassign targets."""
    call_stmt = ast.parse("fn(x=(y := 1), **(z := {}))").body[0]
    assert isinstance(call_stmt, ast.Expr)
    expr_call = call_stmt.value
    def_call, _ = _walrus_assignment_in_expr(expr_call)
    assert "y" in def_call
    assert "z" in def_call

    assign_code = "data[(k := 1)] = val\n"
    tree_assign = ast.parse(assign_code)
    def_assign, _ = _analyze_block_assignment(tree_assign.body)
    assert "k" in def_assign

    augassign_code = "data[(m := 2)] += val\n"
    tree_augassign = ast.parse(augassign_code)
    def_augassign, _ = _analyze_block_assignment(tree_augassign.body)
    assert "m" in def_augassign

def test_if_else_single_and_dual_branch_deletion() -> None:
    """Verifies that if/else deletions in only one branch demote to conditional, while both branches discard."""
    code_single = "x = 1\nif cond:\n    del x\nelse:\n    y = 2\n"
    d_single, c_single = _analyze_block_assignment(ast.parse(code_single).body)
    assert "x" not in d_single
    assert "x" in c_single
    assert "y" in c_single

    code_both = "x = 1\nif cond:\n    del x\nelse:\n    del x\n"
    d_both, c_both = _analyze_block_assignment(ast.parse(code_both).body)
    assert "x" not in d_both
    assert "x" not in c_both

def test_with_and_finally_conditional_deletion_demotion() -> None:
    """Verifies that deletions under conditional sub-blocks within with and finally demote to conditional."""
    code_with = "x = 1\nwith lock:\n    if cond:\n        del x\n"
    d_with, c_with = _analyze_block_assignment(ast.parse(code_with).body)
    assert "x" not in d_with
    assert "x" in c_with

    code_fin_cond = "x = 1\ntry:\n    pass\nfinally:\n    if cond:\n        del x\n"
    d_fin, c_fin = _analyze_block_assignment(ast.parse(code_fin_cond).body)
    assert "x" not in d_fin
    assert "x" in c_fin

    code_fin_uncond = "x = 1\ntry:\n    pass\nfinally:\n    del x\n"
    d_fin_u, c_fin_u = _analyze_block_assignment(ast.parse(code_fin_uncond).body)
    assert "x" not in d_fin_u
    assert "x" not in c_fin_u

def test_undefined_variable_deletion_does_not_create_conditional() -> None:
    """Verifies that deleting a variable that was never defined does not add it to conditional stores."""
    code = "if cond:\n    del z\n"
    d, c = _analyze_block_assignment(ast.parse(code).body)
    assert "z" not in d
    assert "z" not in c

def test_conditional_output_which_is_also_input_not_overwritten_with_none(tmp_path: Path) -> None:
    """Verifies that when a conditional output is also an input parameter, it is not initialized to None."""
    code1 = (
        "def compute(flag, total):\n"
        "    if flag:\n"
        "        total = total + 10\n"
        "    return total\n"
    )
    f1 = tmp_path / "mod1.py"
    f1.write_text(code1, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 3, "name": "compute:If", "kind": "compound_block"}
    helper = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    assert "total = None" not in helper
    assert "total: Any" in helper or "total" in helper

def test_subroutine_nonterminal_embedded_return_hazard_rejected(tmp_path: Path) -> None:
    """Verifies that subroutine compound blocks with non-terminal returns are rejected as hazards."""
    src = (
        "def func_a(x: int) -> int:\n"
        "    if x < 0:\n"
        "        return 0\n"
        "    y = x * 2\n"
        "    return y\n"
        "\n"
        "def func_b(x: int) -> int:\n"
        "    if x < 0:\n"
        "        return 0\n"
        "    y = x * 2\n"
        "    return y\n"
    )
    f = tmp_path / "hazard.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "func_a:blk", "file": "hazard.py", "start": 2, "end": 4, "kind": "compound_block"}
    u2 = {"name": "func_b:blk", "file": "hazard.py", "start": 8, "end": 10, "kind": "compound_block"}

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_type2_arity_mismatch_skips_delegation(tmp_path: Path) -> None:
    """Verifies that clone pairs with differing input or output variable arities skip replacement."""
    src = (
        "def compute_first(a: int, b: int) -> int:\n"
        "    res = a * b + 1\n"
        "    return res\n"
        "\n"
        "def compute_second(a: int) -> int:\n"
        "    res = a * 10 + 1\n"
        "    return res\n"
    )
    f = tmp_path / "arity.py"
    f.write_text(src, encoding="utf-8")

    u1 = {"name": "compute_first", "file": "arity.py", "start": 1, "end": 3, "kind": "function"}
    u2 = {"name": "compute_second", "file": "arity.py", "start": 5, "end": 7, "kind": "function"}

    real_scope = analyze_unit_variable_scope

    def fake_scope(unit: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        res = real_scope(unit, *args, **kwargs)
        if unit.get("name") == "compute_second":
            res["inputs"] = ["a"]
        return res

    with mock.patch("pydoppelgangerhunt.fixer.analyze_unit_variable_scope", side_effect=fake_scope):
        patch = generate_refactoring_patch(
            [(0.90, u1, u2)],
            repo_root=str(tmp_path),
            replace_clones=True,
        )
        assert patch == ""

def test_partially_overlapping_inputs_preserves_full_signature(tmp_path: Path) -> None:
    """Verifies that Type-2 clone pairs with partially overlapping parameter names preserve all inputs."""
    code1 = (
        "def compute_delta(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def compute_delta_v2(a: int, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "calc1.py"
    f2 = tmp_path / "calc2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "calc1.py", "start": 1, "end": 2, "name": "compute_delta", "kind": "function"}
    u2 = {"file": "calc2.py", "start": 1, "end": 2, "name": "compute_delta_v2", "kind": "function"}

    scope = analyze_unit_variable_scope(u1, u2, repo_root=str(tmp_path))
    assert scope["inputs"] == ["x", "y", "timeout"]

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_compute_delta_compute_delta_v2(x: int, y: int, timeout: float = 1.0) -> int:" in helper
    assert "return x * 2 + y" in helper

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_compute_delta_compute_delta_v2(x, y, timeout)" in patch
    assert "return _shared_compute_delta_compute_delta_v2(a, b, timeout)" in patch

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_is_irrefutable_pattern_and_case() -> None:
    """Verifies that _is_irrefutable_case identifies wildcard, as, and alternative irrefutable patterns."""
    from pydoppelgangerhunt.fixer import _is_irrefutable_case  # pylint: disable=import-outside-toplevel

    code = (
        "match x:\n"
        "    case _:\n"
        "        pass\n"
        "    case val:\n"
        "        pass\n"
        "    case _ as y:\n"
        "        pass\n"
        "    case 1 | _:\n"
        "        pass\n"
        "    case (1 | _) as y:\n"
        "        pass\n"
        "    case 1 as y:\n"
        "        pass\n"
        "    case 1 | 2:\n"
        "        pass\n"
        "    case _ if False:\n"
        "        pass\n"
        "    case [a, b]:\n"
        "        pass\n"
    )
    tree = ast.parse(code)
    cases = tree.body[0].cases  # type: ignore[attr-defined]

    # case _:
    assert _is_irrefutable_case(cases[0]) is True
    # case val:
    assert _is_irrefutable_case(cases[1]) is True
    # case _ as y:
    assert _is_irrefutable_case(cases[2]) is True
    # case 1 | _:
    assert _is_irrefutable_case(cases[3]) is True
    # case (1 | _) as y:
    assert _is_irrefutable_case(cases[4]) is True
    # case 1 as y:
    assert _is_irrefutable_case(cases[5]) is False
    # case 1 | 2:
    assert _is_irrefutable_case(cases[6]) is False
    # case _ if False:
    assert _is_irrefutable_case(cases[7]) is False
    # case [a, b]:
    assert _is_irrefutable_case(cases[8]) is False

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_pattern_stores_deletion_and_as_pattern_definite_assignment() -> None:
    """Verifies that variable deletions inside match bodies discard pattern-bound stores, and as-patterns are definite."""
    code = (
        "match x:\n"
        "    case (a, b):\n"
        "        del a\n"
        "        res = 1\n"
        "    case _ as fallback:\n"
        "        res = 2\n"
    )
    tree = ast.parse(code)
    definite, conditional = _analyze_block_assignment(tree.body)

    # res is assigned in all non-terminating branches of an irrefutable match
    assert "res" in definite
    assert "res" not in conditional

    # a was bound by pattern but deleted in case 1 body
    assert "a" not in definite
    assert "a" not in conditional

    # b and fallback were conditionally bound in respective branches
    assert "b" in conditional
    assert "b" not in definite
    assert "fallback" in conditional
    assert "fallback" not in definite

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_match_guard_walrus_definite_in_case_body() -> None:
    """Verifies that walrus variables defined in match guards are recognized as definite within the case body."""
    code = (
        "match x:\n"
        "    case items if (n := len(items)) > 0:\n"
        "        del n\n"
        "        res = 1\n"
        "    case _:\n"
        "        res = 2\n"
    )
    tree = ast.parse(code)
    definite, conditional = _analyze_block_assignment(tree.body)

    # res is definitely assigned across the whole match
    assert "res" in definite
    # n was walrus-assigned in the guard and deleted inside the case body, so it should not be definite or conditional
    assert "n" not in definite
    assert "n" not in conditional


def test_unfold_receiver_attribute_custom_receivers() -> None:
    """Verifies that _unfold_receiver_attribute supports custom receiver names."""
    tree_this = ast.parse("this.config.timeout")
    expr_this = getattr(tree_this.body[0], "value")
    assert _unfold_receiver_attribute(expr_this, receiver_names=("this",)) == "this.config.timeout"
    assert _unfold_receiver_attribute(expr_this) is None  # Defaults to self/cls

    tree_klass = ast.parse("klass.cache.enabled")
    expr_klass = getattr(tree_klass.body[0], "value")
    assert _unfold_receiver_attribute(expr_klass, receiver_names=("klass",)) == "klass.cache.enabled"
    assert _unfold_receiver_attribute(expr_klass) is None


def test_normalize_receiver_attr_name_and_set() -> None:
    """Verifies that receiver attribute normalization maps different prefixes to canonical <rec>."""
    assert _normalize_receiver_attr_name("self.total") == "<rec>.total"
    assert _normalize_receiver_attr_name("cls.count") == "<rec>.count"
    assert _normalize_receiver_attr_name("this.total", receiver_param="this") == "<rec>.total"
    assert _normalize_receiver_attr_name("klass.count", receiver_param="klass") == "<rec>.count"
    assert _normalize_receiver_attr_name("other.total", receiver_param="this") == "other.total"

    attrs1 = ["self.x", "self.y.z"]
    attrs2 = ["this.x", "this.y.z"]
    assert _normalize_receiver_attrs(attrs1) == {"<rec>.x", "<rec>.y.z"}
    assert _normalize_receiver_attrs(attrs2, receiver_param="this") == {"<rec>.x", "<rec>.y.z"}


def test_scope_inspection_custom_receiver_attributes(tmp_path: Path) -> None:
    """Verifies that scope inspection populates attrs_read, instance_attrs, and class_attrs for custom receivers."""
    code = (
        "class Worker:\n"
        "    def run(this, val: int) -> int:\n"
        "        this.total = this.multiplier * val\n"
        "        return this.total\n"
        "\n"
        "    @classmethod\n"
        "    def tally(klass, delta: int) -> int:\n"
        "        klass.count += delta\n"
        "        return klass.count\n"
    )
    f = tmp_path / "worker.py"
    f.write_text(code, encoding="utf-8")

    u_inst = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "run",
        "kind": "function",
        "receiver_param": "this",
        "receiver_kind": "instance",
    }
    s_inst = analyze_unit_variable_scope(u_inst, repo_root=str(tmp_path))
    assert s_inst["has_instance_binding"] is True
    assert s_inst["has_class_binding"] is False
    assert s_inst["has_receiver_access"] is True
    assert "this.multiplier" in s_inst["attrs_read"]
    assert "this.total" in s_inst["attrs_written"]
    assert "this.multiplier" in s_inst["instance_attrs"]
    assert s_inst["inputs"][0] == "this"

    u_cls = {
        "file": str(f),
        "start": 7,
        "end": 9,
        "name": "tally",
        "kind": "function",
        "receiver_param": "klass",
        "receiver_kind": "class",
    }
    s_cls = analyze_unit_variable_scope(u_cls, repo_root=str(tmp_path))
    assert s_cls["has_class_binding"] is True
    assert s_cls["has_instance_binding"] is False
    assert s_cls["has_receiver_access"] is True
    assert "klass.count" in s_cls["attrs_read"]
    assert "klass.count" in s_cls["class_attrs"]
    assert s_cls["inputs"][0] == "klass"

