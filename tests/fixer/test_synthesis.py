"""Unit tests for fixer shared helper code synthesis, parameter ranks, and type inference."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from pydoppelgangerhunt import (
    check_units_overlap,
    extract_unit_source_code,
    filter_overlapping_clone_units,
    generate_clone_diff,
    generate_refactoring_patch,
    synthesize_refactoring_suggestion,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _base_unit_name,
    _extract_required_typing_imports,
    _format_call_arguments,
    _infer_helper_return_type,
    _is_same_file_path,
    _normalize_file_path,
    analyze_unit_variable_scope,
    is_subroutine_unit,
)
from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=protected-access
    _extract_nested_scope_free_reads,
    collect_downstream_read_names,
)
from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=protected-access
    _inspect_unit_scope,
    inspect_single_unit_scope,
)
from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=protected-access
    _infer_outputs_return_type,
    _normalize_pipe_unions,
    _split_type_args,
)
from pydoppelgangerhunt.parser import harvest_file_units


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

    lines1 = extract_unit_source_code(u1, repo_root=str(tmp_path))
    assert len(lines1) == 4
    assert "process_order_v1" in lines1[0]

    missing_lines = extract_unit_source_code(
        {"file": "non_existent.py", "start": 1, "end": 5, "name": "mock"},
        repo_root=str(tmp_path),
    )
    assert "Source for mock" in missing_lines[0]

    diff = generate_clone_diff(u1, u2, repo_root=str(tmp_path))

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

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_process_alpha" in helper
    assert "v = x + y" in helper

    patch = generate_refactoring_patch([(0.85, u1, u2)], repo_root=str(tmp_path))
    assert "--- a/" in patch
    assert "+++ b/" in patch
    assert "def _shared_process_alpha" in patch


def test_synthesize_shared_helper_code_same_class_and_class_binding(tmp_path: Path) -> None:
    """Verifies synthesizing private instance and class methods with appropriate indentation."""
    f = tmp_path / "calc.py"
    code = (
        "class Calculator:\n"
        "    def add_tax_a(self, amount: float) -> float:\n"
        "        rate = 0.05\n"
        "        return amount * (1.0 + rate)\n"
        "\n"
        "    def add_tax_b(self, amount: float) -> float:\n"
        "        rate = 0.05\n"
        "        return amount * (1.0 + rate)\n"
        "\n"
        "    @classmethod\n"
        "    def create_a(cls, val: int) -> int:\n"
        "        cls.total += val\n"
        "        return cls.total\n"
        "\n"
        "    @classmethod\n"
        "    def create_b(cls, val: int) -> int:\n"
        "        cls.total += val\n"
        "        return cls.total\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "add_tax_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "add_tax_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }

    # Auto mode detects same class in same file -> method mode
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto", repo_root=str(tmp_path))
    assert "    def _shared_add_tax_a_add_tax_b(self, amount: float) -> float:" in helper
    assert "self: Any" not in helper  # self must not have : Any annotation
    assert "Call site:\n            self._shared_add_tax_a_add_tax_b(...)" in helper

    # Class method binding
    u_c1 = {
        "file": str(f),
        "start": 10,
        "end": 13,
        "name": "create_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u_c2 = {
        "file": str(f),
        "start": 15,
        "end": 18,
        "name": "create_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    cls_helper = synthesize_shared_helper_code(u_c1, u_c2, method_binding="method", repo_root=str(tmp_path))
    assert "    @classmethod\n    def _shared_create_a_create_b(cls, val: int) -> int:" in cls_helper
    assert "Call site:\n            cls._shared_create_a_create_b(...)" in cls_helper

    # Explicit module mode falls back to module-level helper with self: Any
    mod_helper = synthesize_shared_helper_code(u1, u2, method_binding="module", repo_root=str(tmp_path))
    assert "def _shared_add_tax_a_add_tax_b(self: Any, amount: float) -> float:" in mod_helper


def test_same_named_whole_method_helper_synthesis(tmp_path: Path) -> None:
    """Verifies that whole-method clones with identical names do not nest def inside helper."""
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    c1 = "class ServiceA:\n    def compute(self, n: int) -> int:\n        \"\"\"Docs.\"\"\"\n        ans = n * 10\n        return ans\n"
    c2 = "class ServiceB:\n    def compute(self, n: int) -> int:\n        \"\"\"Docs.\"\"\"\n        ans = n * 10\n        return ans\n"
    f1.write_text(c1, encoding="utf-8")
    f2.write_text(c2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 2, "end": 5, "name": "compute", "kind": "function", "enclosing_class": "ServiceA"}
    u2 = {"file": str(f2), "start": 2, "end": 5, "name": "compute", "kind": "function", "enclosing_class": "ServiceB"}
    code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def compute(" not in code
    assert "ans = n * 10" in code
    assert "return ans" in code


def test_module_helper_placed_after_docstring_and_future_imports(tmp_path: Path) -> None:
    """Verifies that cross-class module helpers are inserted after docstrings and future imports."""
    f = tmp_path / "order_proc.py"
    content = (
        '"""Module docstring."""\n'
        'from __future__ import annotations\n'
        '\n'
        'class Handler1:\n'
        '    def handle(self, num: int) -> int:\n'
        '        return num + 42\n'
        '\n'
        'class Handler2:\n'
        '    def handle(self, num: int) -> int:\n'
        '        return num + 42\n'
    )
    f.write_text(content, encoding="utf-8")
    u1 = {"file": str(f), "start": 5, "end": 6, "name": "handle", "kind": "function", "enclosing_class": "Handler1"}
    u2 = {"file": str(f), "start": 9, "end": 10, "name": "handle", "kind": "function", "enclosing_class": "Handler2"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert '"""Module docstring."""' in patch
    assert 'from __future__ import annotations' in patch
    assert "+def _shared_handle(self: Any, num: int) -> int:" in patch


def test_static_methods_avoid_self_injection(tmp_path: Path) -> None:
    """Verifies static method clones don't inject self and delegate cleanly."""
    f = tmp_path / "math_util.py"
    code = (
        "class MathUtils:\n"
        "    @staticmethod\n"
        "    def add_sq1(x: int, y: int) -> int:\n"
        "        return (x + y) ** 2\n"
        "\n"
        "    @staticmethod\n"
        "    def add_sq2(x: int, y: int) -> int:\n"
        "        return (x + y) ** 2\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 3, "end": 4, "name": "add_sq1", "kind": "function", "enclosing_class": "MathUtils"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "add_sq2", "kind": "function", "enclosing_class": "MathUtils"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "+def _shared_add_sq1_add_sq2(x: int, y: int) -> int:" in patch
    assert "+        return _shared_add_sq1_add_sq2(x, y)" in patch
    assert "self." not in patch


def test_parameter_kinds_forwarding_in_calls(tmp_path: Path) -> None:
    """Verifies that keyword-only, vararg, and kwarg parameters are forwarded with correct syntax."""
    f = tmp_path / "dispatcher.py"
    code = (
        "class EventDispatcher:\n"
        "    def dispatch_a(self, event: str, *args: object, retries: int = 3, **kwargs: object) -> bool:\n"
        "        self.sent.append(event)\n"
        "        return len(self.sent) > 0\n"
        "\n"
        "    def dispatch_b(self, event: str, *args: object, retries: int = 3, **kwargs: object) -> bool:\n"
        "        self.sent.append(event)\n"
        "        return len(self.sent) > 0\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 4, "name": "dispatch_a", "kind": "function", "enclosing_class": "EventDispatcher"}
    u2 = {"file": str(f), "start": 6, "end": 8, "name": "dispatch_b", "kind": "function", "enclosing_class": "EventDispatcher"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "retries=retries" in patch
    assert "*args" in patch
    assert "**kwargs" in patch


def test_helper_inserted_before_decorators(tmp_path: Path) -> None:
    """Verifies helper is inserted before decorators on earliest method."""
    f = tmp_path / "repo.py"
    code = (
        "class Repository:\n"
        "    @classmethod\n"
        "    def get_first(cls, name: str) -> str:\n"
        "        return f'item:{name}'\n"
        "\n"
        "    @classmethod\n"
        "    def get_second(cls, name: str) -> str:\n"
        "        return f'item:{name}'\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 3, "end": 4, "name": "get_first", "kind": "function", "enclosing_class": "Repository"}
    u2 = {"file": str(f), "start": 7, "end": 8, "name": "get_second", "kind": "function", "enclosing_class": "Repository"}
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "_shared_get_first_get_second(cls, name: str) -> str:" in patch
    assert "return cls._shared_get_first_get_second(name)" in patch


def test_static_methods_propagation_in_synthesis_and_patch(tmp_path: Path) -> None:
    """Verifies is_static propagates into synthesize_shared_helper_code and method_binding='method'."""
    u1: Dict[str, Any] = {
        "file": "calc.py",
        "start": 3,
        "end": 4,
        "name": "calc_a",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    u2: Dict[str, Any] = {
        "file": "calc.py",
        "start": 7,
        "end": 8,
        "name": "calc_b",
        "kind": "function",
        "enclosing_class": "Calculator",
    }
    code = synthesize_shared_helper_code(
        u1,
        u2,
        method_binding="method",
        is_static=True,
    )
    assert "@staticmethod" in code
    assert "(self" not in code
    assert "(cls" not in code
    assert "__class__._shared_calc_a_calc_b" in code

    f = tmp_path / "calc.py"
    calc_code = (
        "class Calculator:\n"
        "    @staticmethod\n"
        "    def add_a(x: int, y: int) -> int:\n"
        "        return x + y\n"
        "\n"
        "    @staticmethod\n"
        "    def add_b(x: int, y: int) -> int:\n"
        "        return x + y\n"
    )
    f.write_text(calc_code, encoding="utf-8")
    u1["file"] = str(f)
    u2["file"] = str(f)
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@staticmethod" in patch
    assert "__class__._shared" in patch


def test_classmethod_compound_block_injects_cls(tmp_path: Path) -> None:
    """Verifies that compound blocks inside @classmethod that don't reference cls still inject cls and @classmethod."""
    f = tmp_path / "factory.py"
    code = (
        "class Factory:\n"
        "    @classmethod\n"
        "    def make_a(cls, x: int, y: int) -> int:\n"
        "        val = (x + y) * 2\n"
        "        return val\n"
        "\n"
        "    @classmethod\n"
        "    def make_b(cls, x: int, y: int) -> int:\n"
        "        val = (x + y) * 2\n"
        "        return val\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "make_a:block",
        "kind": "compound_block",
        "enclosing_class": "Factory",
    }
    u2 = {
        "file": str(f),
        "start": 9,
        "end": 10,
        "name": "make_b:block",
        "kind": "compound_block",
        "enclosing_class": "Factory",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@classmethod" in patch
    assert "_shared_make_a_make_b(cls, " in patch
    assert "cls._shared_make_a_make_b" in patch
    assert "self." not in patch


def test_synthesize_shared_helper_harvested_units_declines_mixed_receiver(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code on real harvested units declines receiver-bound mixed-kind pairs."""
    code = (
        "class Handler:\n"
        "    def inst_worker(self, x: int) -> int:\n"
        "        y = x\n"
        "        return self.value + y\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return cls.value + y\n"
    )
    f = tmp_path / "mixed_harvest.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1)
    by_name = {u["name"]: u for u in units}

    # Direct call to synthesize_shared_helper_code without pre-configured receiver_kind
    helper = synthesize_shared_helper_code(
        by_name["inst_worker"], by_name["cls_worker"], repo_root=str(tmp_path)
    )
    assert helper == ""


def test_closure_whole_function_refactoring_and_body_extraction(tmp_path: Path) -> None:
    """Verifies that whole closure units strip their header in synthesis and delegate properly in patches."""
    code = (
        "def factory_a():\n"
        "    def inner_a(x: int) -> int:\n"
        "        step_val = 1\n"
        "        return x + step_val\n"
        "    return inner_a\n"
        "\n"
        "def factory_b():\n"
        "    def inner_b(x: int) -> int:\n"
        "        step_val = 1\n"
        "        return x + step_val\n"
        "    return inner_b\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "factory_a:inner_a",
        "kind": "closure",
    }
    u2 = {
        "file": str(f),
        "start": 8,
        "end": 10,
        "name": "factory_b:inner_b",
        "kind": "closure",
    }
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper
    # def inner_a must NOT be nested inside the helper body
    assert "def inner_a" not in helper
    assert "def inner_b" not in helper

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch
    # Closure definitions must be preserved and delegate to helper
    assert "def inner_a(x: int) -> int:" in patch
    assert "return _shared_inner_a_inner_b(x)" in patch


def test_synthesize_shared_helper_independent_receiver_kinds(tmp_path: Path) -> None:
    """Verifies that units with differing is_static flags are not cross-contaminated into identical receiver kinds."""
    code_stat = (
        "def stat_fn(x: int) -> int:\n"
        "    return x + 1\n"
    )
    code_inst = (
        "class Worker:\n"
        "    def inst_fn(self, x: int) -> int:\n"
        "        return self.value + x\n"
    )
    f1 = tmp_path / "mod_stat.py"
    f2 = tmp_path / "mod_inst.py"
    f1.write_text(code_stat, encoding="utf-8")
    f2.write_text(code_inst, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 2, "name": "stat_fn", "kind": "function", "is_static": True}
    u2 = {"file": str(f2), "start": 2, "end": 3, "name": "inst_fn", "kind": "function", "is_static": False}

    # Should detect differing receiver kinds and decline helper replacement due to self.value reference
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""


def test_mixed_async_and_sync_pair_declined(tmp_path: Path) -> None:
    """Verifies that clone pairs with mixed async and sync execution models are declined."""
    code = (
        "async def async_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
        "\n"
        "def sync_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
    )
    f = tmp_path / "mixed_async.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 3, "name": "async_worker", "kind": "function"}
    u2 = {"file": str(f), "start": 5, "end": 7, "name": "sync_worker", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_mixed_generator_and_function_pair_declined(tmp_path: Path) -> None:
    """Verifies that clone pairs with mixed generator (yield) and normal function models are declined."""
    code = (
        "def gen_worker(x: int):\n"
        "    y = x * 2\n"
        "    yield y + 1\n"
        "\n"
        "def fn_worker(x: int) -> int:\n"
        "    y = x * 2\n"
        "    return y + 1\n"
    )
    f = tmp_path / "mixed_gen.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 3, "name": "gen_worker", "kind": "function"}
    u2 = {"file": str(f), "start": 5, "end": 7, "name": "fn_worker", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_closures_in_classes_synthesize_module_helper_auto_mode(tmp_path: Path) -> None:
    """Verifies that closures inside classes synthesize module-level helpers in auto mode."""
    code = (
        "class Service:\n"
        "    def run_a(self, items: list) -> list:\n"
        "        def transform(val: int) -> int:\n"
        "            x = val * 2\n"
        "            y = x + 3\n"
        "            return y\n"
        "        return [transform(i) for i in items]\n"
        "    def run_b(self, items: list) -> list:\n"
        "        def transform(val: int) -> int:\n"
        "            x = val * 2\n"
        "            y = x + 3\n"
        "            return y\n"
        "        return [transform(i) for i in items]\n"
    )
    f = tmp_path / "service.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=2, min_tokens=5, harvest_closures=True)
    closures = [u for u in units if u["kind"] == "closure"]
    assert len(closures) == 2
    c1, c2 = closures[0], closures[1]

    helper = synthesize_shared_helper_code(c1, c2, method_binding="auto", repo_root=str(tmp_path))
    # Helper must be module-level (no indentation, no self receiver)
    assert helper.startswith("def _shared_transform(")
    assert "self" not in helper


def test_base_unit_name_fallback_for_underscore_or_empty_name() -> None:
    """Verifies that _base_unit_name falls back to 'helper' when unit name has only underscores."""
    assert _base_unit_name({"name": "_", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "___", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "", "kind": "function"}) == "helper"
    assert _base_unit_name({"name": "outer:_", "kind": "closure"}) == "helper"
    assert _base_unit_name({"name": "calculate", "kind": "function"}) == "calculate"


def test_format_call_arguments_vararg_kwarg_ordering() -> None:
    """Verifies that _format_call_arguments orders positional arguments and free variables before varargs/kwargs."""
    inputs = ["self", "*args", "**kwargs", "extra_pos"]
    param_details = [
        {"name": "self", "kind": "pos"},
        {"name": "*args", "kind": "vararg"},
        {"name": "**kwargs", "kind": "kwarg"},
        {"name": "extra_pos", "kind": "pos"},
    ]
    args_str = _format_call_arguments(inputs, param_details, receiver_to_omit="self")
    assert args_str == "extra_pos, *args, **kwargs"
    # Ensure it parses cleanly without SyntaxError: positional argument follows keyword argument unpacking
    ast.parse(f"call({args_str})")


def test_synthesize_shared_helper_code_body_typing_imports(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code captures typing constructs inside the helper body."""
    code = (
        "def transform(data: list) -> list:\n"
        "    res: List[Dict[str, Any]] = []\n"
        "    for x in data:\n"
        "        res.append({'val': x})\n"
        "    return res\n"
    )
    f = tmp_path / "typing_body.py"
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 1, "end": 5, "name": "transform", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u1, include_imports=True, repo_root=str(tmp_path))
    assert "from typing import" in helper
    assert "Dict" in helper
    assert "List" in helper
    assert "Any" in helper


def test_defensive_unit_file_and_name_none_handling() -> None:
    """Verifies that units with None or missing file/name attributes are handled without exceptions."""
    u_none_file: Dict[str, Any] = {"file": None, "start": 1, "end": 5, "name": "test"}
    assert filter_overlapping_clone_units([u_none_file]) == [u_none_file]
    assert not check_units_overlap(u_none_file, {"file": "a.py", "start": 1, "end": 5})

    u_none_name: Dict[str, Any] = {"file": "a.py", "start": 1, "end": 5, "name": None}
    suggestion = synthesize_refactoring_suggestion(u_none_name, u_none_name)
    assert "[REFACTOR SUGGESTION]" in suggestion


def test_batch_36_path_resolution_and_same_file_matching(tmp_path: Any) -> None:
    """Tests Batch 36: robust path normalization and resolution across duplicate units."""

    # 1. Base string equivalence and anchor handling
    assert not _is_same_file_path("", "foo.py")
    assert not _is_same_file_path("foo.py", "")
    assert _is_same_file_path("foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo/bar.ipynb#cell_1", "foo/bar.ipynb#cell_2")
    assert not _is_same_file_path("worker.py#v1", "worker.py#v2")
    assert _is_same_file_path("./foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo\\bar.py", "foo/bar.py")
    assert _normalize_file_path("") == ""
    assert _normalize_file_path("./mod.ipynb#cell_1").endswith("mod.ipynb")
    assert _normalize_file_path("worker.py#cell_data.py").endswith("worker.py#cell_data.py")

    # 2. Filesystem-based relative vs absolute path equivalence
    target_file = tmp_path / "sample.py"
    target_file.write_text(
        "class Worker:\n"
        "    def task_a(self, x):\n"
        "        val = x * 2 + 1\n"
        "        return val\n"
        "    def task_b(self, x):\n"
        "        val = x * 2 + 1\n"
        "        return val\n",
        encoding="utf-8",
    )

    abs_path_str = str(target_file)
    rel_path_dot = "./sample.py"
    rel_path_plain = "sample.py"

    assert _is_same_file_path(abs_path_str, rel_path_dot, repo_root=str(tmp_path))
    assert _is_same_file_path(rel_path_plain, rel_path_dot, repo_root=str(tmp_path))

    symlink_file = tmp_path / "sym_sample.py"
    try:
        symlink_file.symlink_to(target_file)
    except (OSError, NotImplementedError):
        pass
    if symlink_file.is_symlink():
        assert not _is_same_file_path(abs_path_str, str(symlink_file), repo_root=str(tmp_path))
        assert not _is_same_file_path(str(symlink_file), abs_path_str, repo_root=str(tmp_path))

    # 3. check_units_overlap with differing path formats
    u_base = {"file": abs_path_str, "start": 2, "end": 4}
    u_overlap = {"file": rel_path_dot, "start": 3, "end": 5}
    u_disjoint = {"file": rel_path_dot, "start": 5, "end": 7}
    assert check_units_overlap(u_base, u_overlap)
    assert not check_units_overlap(u_base, u_disjoint)

    # 4. filter_overlapping_clone_units groups identical files despite leading dot-slash
    filtered = filter_overlapping_clone_units([u_base, u_overlap])
    assert len(filtered) == 1

    # 5. synthesize_shared_helper_code recognizes same-class across relative and absolute paths
    u1 = {
        "file": abs_path_str,
        "name": "task_a",
        "type": "function",
        "kind": "function",
        "start": 2,
        "end": 4,
        "params": ["self", "x"],
        "enclosing_class": "Worker",
        "enclosing_class_start": 1,
        "receiver_kind": "method",
    }
    u2 = {
        "file": rel_path_dot,
        "name": "task_b",
        "type": "function",
        "kind": "function",
        "start": 5,
        "end": 7,
        "params": ["self", "x"],
        "enclosing_class": "Worker",
        "enclosing_class_start": 1,
        "receiver_kind": "method",
    }
    helper_code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "def _shared_task_a_task_b(self, x: Any) -> Any:" in helper_code

    # 6. generate_refactoring_patch succeeds with mixed path formats
    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "def _shared_task_a_task_b(self, x: Any) -> Any:" in patch
    assert "self._shared_task_a_task_b(x)" in patch


def test_batch_58_parenthesized_return_and_relative_path_resolution(tmp_path: Path) -> None:
    """Batch 58: Test parenthesized/tabbed return detection and relative path resolution in patch/harvest."""
    # 1. Test parenthesized return 'return(res)' in synthesize_shared_helper_code
    code_paren = (
        "def compute_paren(val: int) -> int:\n"
        "    res = val * 2\n"
        "    return(res)\n"
    )
    f1 = tmp_path / "mod_paren.py"
    f1.write_text(code_paren, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 3, "name": "compute_paren", "kind": "function"}

    helper_paren = synthesize_shared_helper_code(u1, u1, repo_root=str(tmp_path))
    # Must contain return(res) and not append a duplicate 'return res'
    assert "return(res)" in helper_paren
    assert "return res" not in helper_paren

    # Test tabbed return 'return\tres'
    code_tab = (
        "def compute_tab(val: int) -> int:\n"
        "    res = val * 2\n"
        "    return\tres\n"
    )
    f2 = tmp_path / "mod_tab.py"
    f2.write_text(code_tab, encoding="utf-8")
    u2 = {"file": str(f2), "start": 1, "end": 3, "name": "compute_tab", "kind": "function"}

    helper_tab = synthesize_shared_helper_code(u2, u2, repo_root=str(tmp_path))
    assert "return\tres" in helper_tab
    assert "return res" not in helper_tab

    # 2. Test generate_refactoring_patch with subpackage path and relative repo_root
    subpkg = tmp_path / "nested_subpkg"
    subpkg.mkdir(parents=True, exist_ok=True)
    sub_file = subpkg / "logic.py"
    sub_code = (
        "def process(a: int, b: int) -> int:\n"
        "    c = a + b\n"
        "    return c\n"
    )
    sub_file.write_text(sub_code, encoding="utf-8")
    u_sub = {
        "file": str(sub_file),
        "start": 1,
        "end": 3,
        "name": "process",
        "kind": "function",
    }
    patch = generate_refactoring_patch([(1.0, u_sub, u_sub)], repo_root=str(tmp_path))
    assert "--- a/nested_subpkg/logic.py" in patch
    assert "+++ b/nested_subpkg/logic.py" in patch

    # 3. Test harvest_file_units with absolute file_path and relative repo_root
    units = harvest_file_units(
        str(sub_file.resolve()),
        repo_root=str(tmp_path.resolve()),
        min_lines=1,
        min_tokens=1,
    )
    assert len(units) >= 1
    assert units[0]["file"] == "nested_subpkg/logic.py"


def test_batch_60_generator_docstring_call_site_and_overlap(tmp_path: Path) -> None:
    """Test generator call site docstring synthesis and check_units_overlap column boundary validation."""
    # 1. Sync generator without return value
    gen_file = tmp_path / "gen_logic.py"
    gen_file.write_text(
        "def produce_stream(limit: int):\n"
        "    for val in range(limit):\n"
        "        yield val * 2\n",
        encoding="utf-8",
    )
    u_gen = {"file": str(gen_file), "start": 1, "end": 3, "name": "produce_stream", "kind": "function"}
    helper_gen = synthesize_shared_helper_code(u_gen, u_gen, repo_root=str(tmp_path))
    assert "yield from _shared_produce_stream(...)" in helper_gen

    # 2. Sync generator with return value (StopIteration.value)
    gen_ret_file = tmp_path / "gen_ret.py"
    gen_ret_file.write_text(
        "def accumulate_stream(limit: int):\n"
        "    accum = 0\n"
        "    for val in range(limit):\n"
        "        accum += val\n"
        "        yield val\n"
        "    return accum\n",
        encoding="utf-8",
    )
    u_gen_ret = {"file": str(gen_ret_file), "start": 1, "end": 6, "name": "accumulate_stream", "kind": "function"}
    helper_gen_ret = synthesize_shared_helper_code(u_gen_ret, u_gen_ret, repo_root=str(tmp_path))
    assert "(yield from _shared_accumulate_stream(...))" in helper_gen_ret

    # 3. Async generator
    agen_file = tmp_path / "agen_logic.py"
    agen_file.write_text(
        "async def produce_async(limit: int):\n"
        "    for val in range(limit):\n"
        "        yield val * 3\n",
        encoding="utf-8",
    )
    u_agen = {"file": str(agen_file), "start": 1, "end": 3, "name": "produce_async", "kind": "function"}
    helper_agen = synthesize_shared_helper_code(u_agen, u_agen, repo_root=str(tmp_path))
    assert "async for _item in _shared_produce_async(...):" in helper_agen
    assert "yield _item" in helper_agen

    # 4. check_units_overlap column boundary validation
    u_base = {"file": "core.py", "start": 10, "end": 10}
    # Overlapping columns (0..20 and 15..30)
    assert check_units_overlap(
        dict(u_base, start_col=0, end_col=20),
        dict(u_base, start_col=15, end_col=30),
    ) is True
    # Disjoint columns (0..10 and 15..30)
    assert check_units_overlap(
        dict(u_base, start_col=0, end_col=10),
        dict(u_base, start_col=15, end_col=30),
    ) is False
    # Malformed / inverted column bounds (e.g. start_col > end_col)
    assert check_units_overlap(
        dict(u_base, start_col=25, end_col=10),
        dict(u_base, start_col=15, end_col=30),
    ) is False


def test_batch_71_type2_clone_parameter_renaming_and_body_retention(tmp_path: Path) -> None:
    """Verifies that Type-2 clones with renamed parameters preserve full helper body and map arguments at call sites."""
    f = tmp_path / "calc.py"
    code = (
        "def calc_alpha(x: int) -> int:\n"
        "    a = x * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    d = c + 4\n"
        "    return d\n"
        "\n"
        "def calc_beta(y: int) -> int:\n"
        "    a = y * 2\n"
        "    b = a + 1\n"
        "    c = b * 3\n"
        "    d = c + 4\n"
        "    return d\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {"file": str(f), "start": 1, "end": 6, "name": "calc_alpha", "kind": "function"}
    u2 = {"file": str(f), "start": 8, "end": 13, "name": "calc_beta", "kind": "function"}

    # 1. Helper synthesis must retain all statements despite variable renaming on line 1
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "a = x * 2" in helper
    assert "b = a + 1" in helper
    assert "c = b * 3" in helper
    assert "d = c + 4" in helper
    assert "return d" in helper

    # 2. Refactoring patch must delegate with x in calc_alpha and y in calc_beta
    patch = generate_refactoring_patch([(0.90, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_calc_alpha_calc_beta(x)" in patch
    assert "return _shared_calc_alpha_calc_beta(y)" in patch
    # Ensure calc_beta does not reference x
    beta_section = patch.split("def calc_beta(y: int) -> int:")[1]
    assert "(x)" not in beta_section
    assert "(y)" in beta_section

    # 3. Compound blocks with renamed outputs
    f_block = tmp_path / "block.py"
    b_code = (
        "def proc1(p: int) -> int:\n"
        "    v = p * 2\n"
        "    out1 = v + 10\n"
        "    return out1\n"
        "\n"
        "def proc2(q: int) -> int:\n"
        "    v = q * 2\n"
        "    out2 = v + 10\n"
        "    return out2\n"
    )
    f_block.write_text(b_code, encoding="utf-8")
    u_b1 = {"file": str(f_block), "start": 2, "end": 3, "name": "b1:1", "kind": "compound_block"}
    u_b2 = {"file": str(f_block), "start": 7, "end": 8, "name": "b2:1", "kind": "compound_block"}
    patch_block = generate_refactoring_patch([(0.90, u_b1, u_b2)], repo_root=str(tmp_path), replace_clones=True)
    assert "v, out1 = _shared_b1_b2(p)" in patch_block
    assert "v, out2 = _shared_b1_b2(q)" in patch_block


def test_multiline_comprehension_helper_synthesis_and_apply(tmp_path: Path) -> None:
    """Verifies that multiline comprehensions with comments are wrapped in return (...) and run cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "def parse_items(items: list) -> list:\n"
        "    return [\n"
        "        # double item value\n"
        "        x * 2\n"
        "        for x in items\n"
        "        if x > 0\n"
        "    ]\n"
        "\n"
        "def process_items(elements: list) -> list:\n"
        "    return [\n"
        "        # double item value\n"
        "        e * 2\n"
        "        for e in elements\n"
        "        if e > 0\n"
        "    ]\n"
    )
    f = tmp_path / "comp.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "parse_items:listcomp",
        "file": "comp.py",
        "start": 2,
        "end": 7,
        "start_col": 11,
        "end_col": 5,
        "kind": "comprehension",
    }
    u2 = {
        "name": "process_items:listcomp",
        "file": "comp.py",
        "start": 10,
        "end": 15,
        "start_col": 11,
        "end_col": 5,
        "kind": "comprehension",
    }

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert "return (" in patch

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
            "from comp import parse_items, process_items; "
            "print(parse_items([1, -1, 3]), process_items([2, -5, 4]))",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0
    assert run_proc.stdout.strip() == "[2, 6] [4, 8]"


def test_type_merge_positional_alignment_renamed_parameters(tmp_path: Path) -> None:
    """Verifies that type annotations merge positionally across renamed parameter names in Type-2 clones."""
    code1 = (
        "def process_val(x: int, y: int, timeout: float = 1.0) -> int:\n"
        "    return x * 2 + y\n"
    )
    code2 = (
        "def process_val_v2(a: str, b: int, timeout: float = 1.0) -> int:\n"
        "    return a * 2 + b\n"
    )
    f1 = tmp_path / "proc1.py"
    f2 = tmp_path / "proc2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "proc1.py", "start": 1, "end": 2, "name": "process_val", "kind": "function"}
    u2 = {"file": "proc2.py", "start": 1, "end": 2, "name": "process_val_v2", "kind": "function"}

    helper = synthesize_shared_helper_code(
        u1, u2, repo_root=str(tmp_path), type_merge_strategy="union"
    )
    assert "def _shared_process_val_process_val_v2(x: Union[int, str], y: int, timeout: float = 1.0) -> int:" in helper


def test_infer_helper_return_type_generator_literal_and_explicit_types() -> None:
    """Verifies that _infer_helper_return_type infers literal yield types and preserves explicit annotations."""
    # 1. Sync generator with literal integer yield
    scope_int = {"has_yield": True, "yield_expr_names": [("yield", ":literal:int")]}
    res_int = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope_int,
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_int == "Iterator[int]"

    # 2. Async generator with literal string yield
    scope_str = {"has_yield": True, "yield_expr_names": [("yield", ":literal:str")]}
    res_str = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope_str,
        meta1={},
        meta2={},
        is_async=True,
    )
    assert res_str == "AsyncIterator[str]"

    # 3. Explicit return type preserved for async generator
    res_explicit = _infer_helper_return_type(
        resolved_ret="AsyncIterator[float]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": []},
        meta1={},
        meta2={},
        is_async=True,
    )
    assert res_explicit == "AsyncIterator[float]"

    # 4. Async generator with yield assignment infers AsyncGenerator[YieldT, SendT]
    res_async_gen = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": True,
            "yield_expr_names": [("yield", "x")],
        },
        meta1={"x": {"type": "int"}},
        meta2={},
        is_async=True,
    )
    assert res_async_gen == "AsyncGenerator[int, Any]"

    # 5. Async generator with explicit AsyncGenerator return annotation
    res_async_gen_explicit = _infer_helper_return_type(
        "AsyncGenerator[int, str]",
        [],
        set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": True,
            "yield_expr_names": [],
        },
        meta1={},
        meta2={},
        is_async=True,
    )
    assert res_async_gen_explicit == "AsyncGenerator[int, str]"


def test_generator_helper_synthesis_literal_yields(tmp_path: Path) -> None:
    """Verifies that synthesizing helpers from generator units infers Iterator types and executes cleanly."""
    code1 = (
        "def num_gen1(limit: int):\n"
        "    for i in range(limit):\n"
        "        yield 42\n"
    )
    code2 = (
        "def num_gen2(limit: int):\n"
        "    for i in range(limit):\n"
        "        yield 42\n"
    )
    f1 = tmp_path / "g1.py"
    f2 = tmp_path / "g2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "g1.py", "start": 1, "end": 3, "name": "num_gen1", "kind": "function"}
    u2 = {"file": "g2.py", "start": 1, "end": 3, "name": "num_gen2", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "-> Iterator[int]:" in helper


def test_batch_82_generator_container_literal_and_pep585_return_type_inference(tmp_path: Path) -> None:
    """Verifies that container literals and PEP 585 lowercase types in yield from are properly inferred."""
    # 1. PEP 585 lowercase list[int] and tuple[str, ...]
    assert _infer_helper_return_type(
        "Any", [], set(),
        scope={"has_yield": True, "yield_expr_names": [("yield_from", "items")]},
        meta1={"items": {"type": "list[int]"}},
        meta2={},
    ) == "Iterator[int]"

    assert _infer_helper_return_type(
        "Any", [], set(),
        scope={"has_yield": True, "yield_expr_names": [("yield_from", "items")]},
        meta1={"items": {"type": "tuple[str, ...]"}},
        meta2={},
    ) == "Iterator[str]"

    assert _infer_helper_return_type(
        "Any", [], set(),
        scope={"has_yield": True, "yield_expr_names": [("yield_from", "items")]},
        meta1={"items": {"type": "set[float]"}},
        meta2={},
        is_async=True,
    ) == "AsyncIterator[float]"

    # 2. Container literal in code unit: yield from ["hello", "world"]
    c1 = (
        "def str_producer():\n"
        "    yield from ['apple', 'banana', 'cherry']\n"
    )
    c2 = (
        "def str_producer2():\n"
        "    yield from ['apple', 'banana', 'cherry']\n"
    )
    f1 = tmp_path / "str1.py"
    f2 = tmp_path / "str2.py"
    f1.write_text(c1, encoding="utf-8")
    f2.write_text(c2, encoding="utf-8")
    u1 = {"file": "str1.py", "start": 1, "end": 2, "name": "str_producer", "kind": "function"}
    u2 = {"file": "str2.py", "start": 1, "end": 2, "name": "str_producer2", "kind": "function"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "-> Iterator[str]:" in helper


def test_format_call_arguments_order_preservation_and_custom_receivers() -> None:
    """Verifies that call argument formatting preserves name identity across load permutations and omits custom receivers."""
    # 1. Identical input names in permuted target_inputs must not be swapped
    inputs = ["alpha", "beta", "gamma"]
    target_inputs = ["gamma", "alpha", "beta"]
    param_details = [{"name": "alpha"}, {"name": "beta"}, {"name": "gamma"}]
    args = _format_call_arguments(inputs, param_details, target_inputs=target_inputs)
    assert args == "alpha, beta, gamma"

    # 2. Custom instance receiver "this" omitted when receiver_to_omit="self"
    args_inst = _format_call_arguments(
        ["this", "value"],
        [{"name": "this"}, {"name": "value"}],
        receiver_to_omit="self",
    )
    assert args_inst == "value"

    # 3. Custom class receiver "klass" omitted when receiver_to_omit="cls"
    args_cls = _format_call_arguments(
        ["klass", "param"],
        [{"name": "klass"}, {"name": "param"}],
        receiver_to_omit="cls",
    )
    assert args_cls == "param"

    # 4. Custom receivers omitted when receiver_to_omit="receivers"
    args_rec = _format_call_arguments(
        ["this", "klass", "data"],
        [{"name": "this"}, {"name": "klass"}, {"name": "data"}],
        receiver_to_omit="receivers",
    )
    assert args_rec == "data"


def test_sync_generator_helper_synthesis_with_outputs_emits_return_and_generator_type(tmp_path: Path) -> None:
    """Verifies that synthesizing helpers from sync generators with outputs emits return and Generator type."""
    # 1. With downstream return: helper returns count, prunes loop variable x, and has Generator type
    code1 = (
        "def count_and_yield1(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += 1\n"
        "        yield x\n"
        "    return count\n"
    )
    code2 = (
        "def count_and_yield2(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += 1\n"
        "        yield x\n"
        "    return count\n"
    )
    f1 = tmp_path / "cy1.py"
    f2 = tmp_path / "cy2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": "cy1.py", "start": 2, "end": 5, "name": "count_and_yield1:for", "kind": "compound_block"}
    u2 = {"file": "cy2.py", "start": 2, "end": 5, "name": "count_and_yield2:for", "kind": "compound_block"}

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "-> Generator[" in helper
    assert "return count" in helper
    assert "return count, x" not in helper

    # 2. Without downstream return: loop variable is not needed downstream, so return is omitted and Iterator type used
    code_no_ret1 = (
        "def pure_yield1(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += 1\n"
        "        yield x\n"
    )
    code_no_ret2 = (
        "def pure_yield2(items: list[int]):\n"
        "    count = 0\n"
        "    for x in items:\n"
        "        count += 1\n"
        "        yield x\n"
    )
    f3 = tmp_path / "py1.py"
    f4 = tmp_path / "py2.py"
    f3.write_text(code_no_ret1, encoding="utf-8")
    f4.write_text(code_no_ret2, encoding="utf-8")

    u3 = {"file": "py1.py", "start": 2, "end": 5, "name": "pure_yield1:for", "kind": "compound_block"}
    u4 = {"file": "py2.py", "start": 2, "end": 5, "name": "pure_yield2:for", "kind": "compound_block"}

    helper_no_ret = synthesize_shared_helper_code(u3, u4, repo_root=str(tmp_path))
    assert "-> Iterator[" in helper_no_ret
    assert "return " not in helper_no_ret


def test_infer_helper_return_type_sync_generator_with_outputs_and_return() -> None:
    """Verifies return type inference for sync generators with single/multiple outputs and return statements."""
    # 1. Single output
    scope_single = {"has_yield": True, "yield_expr_names": [("yield", ":literal:int")]}
    res_single = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=["total"],
        conditional_outs=set(),
        scope=scope_single,
        meta1={"total": {"type": "int"}},
        meta2={},
        is_async=False,
    )
    assert res_single == "Generator[int, None, int]"

    # 2. Multiple outputs (Tuple return)
    scope_multi = {"has_yield": True, "yield_expr_names": [("yield", ":literal:str")]}
    res_multi = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=["total", "status"],
        conditional_outs=set(),
        scope=scope_multi,
        meta1={"total": {"type": "int"}, "status": {"type": "str"}},
        meta2={},
        is_async=False,
    )
    assert res_multi == "Generator[str, None, Tuple[int, str]]"

    # 3. Explicit generator return preserved
    res_explicit = _infer_helper_return_type(
        resolved_ret="Generator[int, None, float]",
        helper_outputs=["total"],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": []},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_explicit == "Generator[int, None, float]"

    # 4. Explicit return type with has_return_value
    res_explicit_int = _infer_helper_return_type(
        resolved_ret="int",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": [], "has_return_value": True},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_explicit_int == "Generator[Any, None, int]"

    # 5. Conditional outputs
    res_cond = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=["total"],
        conditional_outs={"total"},
        scope={"has_yield": True, "yield_expr_names": [("yield", ":literal:int")]},
        meta1={"total": {"type": "int"}},
        meta2={},
        is_async=False,
    )
    assert res_cond == "Generator[int, None, Optional[int]]"

    # 6. Bare return with no value -> Iterator
    res_bare = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": [], "has_return": True, "has_return_value": False},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_bare == "Iterator[Any]"

    # 7. No return at all -> Iterator
    res_no_ret = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": [], "has_return": False, "has_return_value": False},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_no_ret == "Iterator[Any]"


def test_async_generator_with_return_rejected_in_synthesis(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code rejects async generators with return value but accepts bare return."""
    # 1. Async generator with return <value> must be rejected
    code_val = (
        "async def agen_returns(items: list[int]):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    return 42\n"
    )
    f1 = tmp_path / "ao1.py"
    f2 = tmp_path / "ao2.py"
    f1.write_text(code_val, encoding="utf-8")
    f2.write_text(code_val, encoding="utf-8")

    u1 = {"file": "ao1.py", "start": 1, "end": 4, "name": "agen_returns", "kind": "function", "is_async": True}
    u2 = {"file": "ao2.py", "start": 1, "end": 4, "name": "agen_returns", "kind": "function", "is_async": True}

    helper_val = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper_val == ""

    # 2. Async generator with bare return (no value) is valid Python (PEP 525) and should be synthesized
    code_bare = (
        "async def agen_bare(items: list[int]):\n"
        "    for x in items:\n"
        "        if x < 0:\n"
        "            return\n"
        "        yield x\n"
    )
    f3 = tmp_path / "ab1.py"
    f4 = tmp_path / "ab2.py"
    f3.write_text(code_bare, encoding="utf-8")
    f4.write_text(code_bare, encoding="utf-8")

    u3 = {"file": "ab1.py", "start": 1, "end": 5, "name": "agen_bare", "kind": "function", "is_async": True}
    u4 = {"file": "ab2.py", "start": 1, "end": 5, "name": "agen_bare", "kind": "function", "is_async": True}

    helper_bare = synthesize_shared_helper_code(u3, u4, repo_root=str(tmp_path))
    assert helper_bare != ""
    assert "async def" in helper_bare
    assert "-> AsyncIterator[" in helper_bare

    # 3. Whole-function async generator with try/finally or with block is rejected
    # because delegation via async for ...: yield loses athrow()/aclose() propagation.
    code_try = (
        "async def agen_try(items: list[int]):\n"
        "    try:\n"
        "        for x in items:\n"
        "            yield x\n"
        "    finally:\n"
        "        pass\n"
    )
    f5 = tmp_path / "at1.py"
    f6 = tmp_path / "at2.py"
    f5.write_text(code_try, encoding="utf-8")
    f6.write_text(code_try, encoding="utf-8")
    u5 = {
        "file": "at1.py", "start": 1, "end": 6,
        "name": "agen_try", "kind": "function", "is_async": True,
    }
    u6 = {
        "file": "at2.py", "start": 1, "end": 6,
        "name": "agen_try", "kind": "function", "is_async": True,
    }
    assert synthesize_shared_helper_code(u5, u6, repo_root=str(tmp_path)) == ""


def test_patch_subroutine_is_async_propagation(tmp_path: Path) -> None:
    """Verifies that is_async from enclosing functions propagates to s1/s2 and rejects illegal returns."""
    code1 = (
        "async def process_data(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    code2 = (
        "async def process_data_alt(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    f1 = tmp_path / "proc1.py"
    f2 = tmp_path / "proc2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    # compound_block without is_async explicitly set in unit dictionary
    u1 = {"file": str(f1), "start": 3, "end": 5, "name": "process_data", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 3, "end": 5, "name": "process_data_alt", "kind": "compound_block"}

    # Because total is needed downstream and the enclosing function is async def (async generator),
    # returning total via a generator subroutine is illegal under PEP 525 and must be rejected.
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""


def test_infer_outputs_return_type_counterpart_lookup() -> None:
    """Verifies that counterpart variable types from outputs2 take precedence over name collisions."""
    meta1 = {"a": {"type": "int"}, "c": {"type": "float"}}
    meta2 = {
        "a": {"type": "str"},  # Unrelated variable in clone 2 scope
        "b": {"type": "int"},  # True counterpart to 'a'
        "c": {"type": "bool"},  # Unrelated variable in clone 2 scope
        "d": {"type": "float"},  # True counterpart to 'c'
    }

    # Multiple outputs
    ret_multi = _infer_outputs_return_type(
        ["a", "c"],
        set(),
        meta1,
        meta2,
        outputs2=["b", "d"],
    )
    assert ret_multi == "Tuple[int, float]"

    # Single output
    ret_single = _infer_outputs_return_type(
        ["a"],
        set(),
        meta1,
        meta2,
        outputs2=["b"],
    )
    assert ret_single == "int"


def test_caller_unit_dicts_immutable_during_patch(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not mutate caller unit dictionaries in-place."""
    code1 = (
        "async def fn1():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    code2 = (
        "async def fn2():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {"file": str(f1), "start": 2, "end": 3, "name": "fn1", "kind": "compound_block"}
    u2 = {"file": str(f2), "start": 2, "end": 3, "name": "fn2", "kind": "compound_block"}

    u1_copy = dict(u1)
    u2_copy = dict(u2)

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""

    assert u1 == u1_copy
    assert u2 == u2_copy
    assert "is_async" not in u1
    assert "is_async" not in u2
    assert "receiver_param" not in u1
    assert "receiver_param" not in u2


def test_caller_unit_dicts_immutable_during_synthesis(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code does not mutate caller dicts in-place."""
    code = (
        "class Worker:\n"
        "    def run1(self):\n"
        "        x = 1\n"
        "        return x\n"
        "    def run2(self):\n"
        "        x = 1\n"
        "        return x\n"
    )
    src_f = tmp_path / "worker.py"
    src_f.write_text(code, encoding="utf-8")

    u1 = {"file": str(src_f), "start": 2, "end": 4, "name": "run1", "kind": "function"}
    u2 = {"file": str(src_f), "start": 5, "end": 7, "name": "run2", "kind": "function"}
    u1_copy = dict(u1)
    u2_copy = dict(u2)

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper != ""
    assert u1 == u1_copy
    assert u2 == u2_copy
    assert "is_async" not in u1
    assert "receiver_kind" not in u1
    assert "is_async" not in u2
    assert "receiver_kind" not in u2


def test_synthesize_shared_helper_code_symmetric_global_filtering(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code filters globals/nonlocals from both u1 and u2 outputs."""
    code = (
        "global_var = 0\n"
        "def fn1():\n"
        "    global global_var\n"
        "    global_var = 100\n"
        "    common_calc = 10\n"
        "    local_var = common_calc + 1\n"
        "def fn2():\n"
        "    global global_var\n"
        "    global_var = 200\n"
        "    common_calc = 10\n"
        "    local_var = common_calc + 1\n"
    )
    f = tmp_path / "mod.py"
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 6,
        "name": "fn1",
        "kind": "function",
        "outputs": ["local_var", "global_var"],
    }
    u2 = {
        "file": str(f),
        "start": 7,
        "end": 11,
        "name": "fn2",
        "kind": "function",
        "outputs": ["local_var", "global_var"],
    }

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        helper_name="_shared_helper",
    )
    assert "return local_var" in helper
    assert "return local_var, global_var" not in helper


def test_infer_helper_return_type_untyped_generator_return_value() -> None:
    """Verifies that a generator with explicit return value and untyped return emits Generator[yield_t, None, Any]."""
    scope = {
        "has_yield": True,
        "is_async": False,
        "has_return_value": True,
        "yield_expr_names": [("yield", ":literal:int")],
        "return_type": None,
    }
    ret_t = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope,
        meta1={},
        meta2={},
    )
    assert ret_t == "Generator[int, None, Any]"


@pytest.mark.skipif(sys.version_info < (3, 10), reason="Pattern matching requires Python 3.10+")
def test_pattern_match_variable_bindings_scope() -> None:
    """Verifies that Python 3.10+ pattern match bindings are recognized as local stores, not escaping reads."""
    code = (
        "def process(val):\n"
        "    match val:\n"
        "        case int(x):\n"
        "            return x\n"
        "        case [first, *rest]:\n"
        "            return (first, rest)\n"
        "        case {'data': item, **extra}:\n"
        "            return (item, extra)\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    escaped = _extract_nested_scope_free_reads(fn_node)  # type: ignore[arg-type]
    # val is a param; x, first, rest, item, extra are bound by pattern matching and should NOT escape
    assert "x" not in escaped
    assert "first" not in escaped
    assert "rest" not in escaped
    assert "item" not in escaped
    assert "extra" not in escaped

    # In downstream read visitor, pattern match binding kills prior reaching definition
    downstream_code = (
        "def worker():\n"
        "    x = 1\n"
        "    match val:\n"
        "        case int(x):\n"
        "            print(x)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(downstream_code, unit, candidates={"x"})
    assert reads is not None
    assert "x" not in reads


def test_infer_outputs_return_type_single_output_precedence() -> None:
    """Verifies that explicit single-output type takes precedence over enclosing resolved_ret unless Any."""
    # Known output type "str" should override enclosing resolved_ret "int"
    ret = _infer_outputs_return_type(
        helper_outputs=["res"],
        conditional_outs=set(),
        meta1={"res": {"type": "str"}},
        meta2={"res": {"type": "str"}},
        type_merge_strategy="prefer_first",
        resolved_ret="int",
    )
    assert ret == "str"

    # When output type is "Any", fall back to enclosing resolved_ret "int"
    ret_fallback = _infer_outputs_return_type(
        helper_outputs=["res"],
        conditional_outs=set(),
        meta1={"res": {"type": "Any"}},
        meta2={"res": {"type": "Any"}},
        type_merge_strategy="prefer_first",
        resolved_ret="int",
    )
    assert ret_fallback == "int"


def test_is_subroutine_unit_classification() -> None:
    """Verifies that is_subroutine_unit classifies subroutines vs functions and expressions."""
    # True for compound blocks, sliding windows, and clause branches
    assert is_subroutine_unit({"kind": "compound_block"}) is True
    assert is_subroutine_unit({"kind": "sliding_window"}) is True
    assert is_subroutine_unit({"kind": "clause_branch"}) is True

    # False for top-level functions, closures, methods, comprehensions, and expressions
    assert is_subroutine_unit({"kind": "function"}) is False
    assert is_subroutine_unit({"kind": "method"}) is False
    assert is_subroutine_unit({"kind": "closure"}) is False
    assert is_subroutine_unit({"kind": "comprehension"}) is False
    assert is_subroutine_unit({"kind": "complex_expr"}) is False

    # Explicit is_subroutine flag takes precedence
    assert is_subroutine_unit({"is_subroutine": True}) is True
    assert is_subroutine_unit({"is_subroutine": False, "kind": "compound_block"}) is False

    # Name-based heuristic is eliminated; returns False for unknown/unspecified kinds
    assert is_subroutine_unit({"name": "process_module:10-25"}) is False
    assert is_subroutine_unit({"name": "plain_function"}) is False
    assert is_subroutine_unit({}) is False

    # Defensive handling of non-dict types (public API guard)
    assert is_subroutine_unit(None) is False
    assert is_subroutine_unit("invalid") is False
    assert is_subroutine_unit(123) is False
    assert is_subroutine_unit([]) is False


def test_harvested_subroutine_inherits_async_status_and_rejection(
    tmp_path: Path,
) -> None:
    """Verifies harvested compound block in async def inherits is_async and rejects return."""
    source = (
        "async def process_stream(data):\n"
        "    total = 0\n"
        "    for item in data:\n"
        "        total += item\n"
        "        yield total\n"
        "    print(total)\n"
    )
    f = tmp_path / "stream.py"
    f.write_text(source, encoding="utf-8")

    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=5)
    compound_units = [u for u in units if u.get("kind") == "compound_block"]
    assert len(compound_units) >= 1
    assert compound_units[0].get("is_async") is True

    # Scope analysis inherits async status
    scope_info = analyze_unit_variable_scope(compound_units[0], repo_root=str(tmp_path))
    assert scope_info.get("is_async") is True

    # Refactoring patch rejects async generator subroutine with outputs
    u2 = dict(compound_units[0])
    u2["start"] = 3
    u2["end"] = 5
    patch = generate_refactoring_patch(
        [(0.95, compound_units[0], u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch == ""


def test_infer_helper_return_type_generator_with_outputs_and_iterator() -> None:
    """Verifies Iterator annotations are converted to Generator when outputs are returned."""
    # 1. Iterator[int] with helper output of type str becomes Generator[int, None, str]
    res_replaced = _infer_helper_return_type(
        resolved_ret="Iterator[int]",
        helper_outputs=["total"],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": []},
        meta1={"total": {"type": "str"}},
        meta2={},
        is_async=False,
    )
    assert res_replaced == "Generator[int, None, str]"

    # 2. Iterable[int] with explicit return value becomes Generator[int, None, Any]
    res_ret_val = _infer_helper_return_type(
        resolved_ret="Iterable[int]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": [], "has_return_value": True},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_ret_val == "Generator[int, None, Any]"

    # 3. Iterator[float] without return values remains Iterator[float]
    res_kept = _infer_helper_return_type(
        resolved_ret="Iterator[float]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={"has_yield": True, "yield_expr_names": []},
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_kept == "Iterator[float]"


def test_infer_helper_return_type_generator_with_yield_assignment() -> None:
    """Verifies that yield expressions in assignment context infer Any send type instead of None."""
    # When scope has_yield_assignment is True, send type defaults to Any
    res = _infer_helper_return_type(
        resolved_ret="Iterator[int]",
        helper_outputs=["total"],
        conditional_outs=set(),
        scope={"has_yield": True, "has_yield_assignment": True, "yield_expr_names": []},
        meta1={"total": {"type": "str"}},
        meta2={},
        is_async=False,
    )
    assert res == "Generator[int, Any, str]"

    # Existing explicit send type in resolved_ret is preserved
    res_explicit = _infer_helper_return_type(
        resolved_ret="Generator[int, str, None]",
        helper_outputs=["total"],
        conditional_outs=set(),
        scope={"has_yield": True, "has_yield_assignment": True, "yield_expr_names": []},
        meta1={"total": {"type": "str"}},
        meta2={},
        is_async=False,
    )
    assert res_explicit == "Generator[int, str, str]"


def test_split_type_args_nested_bracket_depth() -> None:
    """Verifies bracket-depth aware splitting of nested generic type arguments."""
    # Empty or non-generic strings
    assert not _split_type_args("int")
    assert not _split_type_args("")

    # Multi-argument nested generic
    t1 = "Generator[Tuple[int, str], None, Dict[str, Any]]"
    assert _split_type_args(t1) == ["Tuple[int, str]", "None", "Dict[str, Any]"]

    # Stray leading and trailing whitespace immunity
    t1_ws = "  Generator[Tuple[int, str], None, Dict[str, Any]] \n "
    assert _split_type_args(t1_ws) == ["Tuple[int, str]", "None", "Dict[str, Any]"]

    # Deeply nested generics
    t2 = "Union[Dict[str, List[int]], Optional[Tuple[float, bool]]]"
    assert _split_type_args(t2) == [
        "Dict[str, List[int]]",
        "Optional[Tuple[float, bool]]",
    ]

    # Forward references and string literals containing commas
    t3 = "Tuple['Literal, Value', int]"
    assert _split_type_args(t3) == ["'Literal, Value'", "int"]

    t4 = 'Union[Literal["a, b", "c"], Dict[str, "X, Y"]]'
    assert _split_type_args(t4) == [
        'Literal["a, b", "c"]',
        'Dict[str, "X, Y"]',
    ]

    t5 = r'Tuple["a\"b, c", int]'
    assert _split_type_args(t5) == [r'"a\"b, c"', "int"]

    # Mismatched bracket depth and unclosed quotes fail closed to empty list
    assert not _split_type_args("Generator[int, Tuple[str, int]")
    assert not _split_type_args("Dict[str, List[int]")
    assert not _split_type_args("Tuple['unclosed, int]")
    assert not _split_type_args('Tuple["unclosed, int]')

    # Inference in _infer_helper_return_type preserving nested tuple yield type
    scope = {
        "has_yield": True,
        "yield_expr_names": [("yield_from", "stream")],
        "has_return_value": False,
    }
    res = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope,
        meta1={"stream": {"type": "Iterator[Tuple[int, str]]"}},
        meta2={},
        is_async=False,
    )
    assert res == "Iterator[Tuple[int, str]]"


def test_analyze_unit_variable_scope_does_not_mutate_caller_unit() -> None:
    """Verifies analyze_unit_variable_scope does not mutate the passed unit dictionary in-place."""
    unit = {
        "start": 2,
        "end": 3,
        "source_lines": ["def sample():\n", "    x = 1\n", "    return x\n"],
    }
    original_keys = set(unit.keys())
    res = analyze_unit_variable_scope(unit)
    assert "is_async" in res
    assert set(unit.keys()) == original_keys
    assert "is_async" not in unit


def test_inspect_unit_scope_prioritizes_full_source_text_over_sliced_lines() -> None:
    """Verifies full source text is prioritized over sliced unit lines to resolve async status."""
    full_text = (
        "async def async_worker(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item\n"
        "    return total\n"
    )
    unit = {
        "start": 3,
        "end": 4,
        "source_lines": ["    for item in items:\n", "        total += item\n"],
        "source_text": full_text,
    }
    scope = analyze_unit_variable_scope(unit)
    assert scope["is_async"] is True


def test_infer_helper_return_type_yield_from_tuples() -> None:
    """Verifies that yield_from on Tuple types infers Union or homogeneous element types."""
    scope = {
        "has_yield": True,
        "yield_expr_names": [("yield_from", "stream")],
        "has_return_value": False,
    }
    # Heterogeneous tuple: Tuple[int, str] yields Union[int, str]
    res_hetero = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope,
        meta1={"stream": {"type": "Tuple[int, str]"}},
        meta2={},
        is_async=False,
    )
    assert res_hetero == "Iterator[Union[int, str]]"

    # Homogeneous variadic tuple: Tuple[int, ...] yields int
    res_homo = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope,
        meta1={"stream": {"type": "Tuple[int, ...]"}},
        meta2={},
        is_async=False,
    )
    assert res_homo == "Iterator[int]"


def test_return_type_precedence_whole_function_vs_subroutine() -> None:
    """Verifies that whole-function units preserve declared return type over inferred
    local store."""
    scope = {
        "has_yield": False,
        "has_return": True,
        "has_return_value": True,
    }
    meta1 = {"val": {"type": "int"}}
    meta2 = {"val": {"type": "int"}}

    # Whole-function unit (unit_kind="function"): declared -> Optional[int] preserved over int
    res_func = _infer_helper_return_type(
        resolved_ret="Optional[int]",
        helper_outputs=["val"],
        conditional_outs=set(),
        scope=scope,
        meta1=meta1,
        meta2=meta2,
        unit_kind="function",
    )
    assert res_func == "Optional[int]"

    # Subroutine unit (unit_kind="compound_block"): outputs_ret takes precedence over enclosing ret
    res_sub = _infer_helper_return_type(
        resolved_ret="Optional[int]",
        helper_outputs=["val"],
        conditional_outs=set(),
        scope=scope,
        meta1=meta1,
        meta2=meta2,
        unit_kind="compound_block",
    )
    assert res_sub == "int"


def test_synthesize_shared_helper_union_yield_type_imports() -> None:
    """Verifies that Union is injected into typing imports when a helper has a union yield type."""
    helper_code = (
        "def _shared_helper() -> Iterator[Union[int, str]]:\n"
        "    yield 1\n"
    )
    needed = _extract_required_typing_imports(helper_code)
    assert "Iterator" in needed
    assert "Union" in needed


def test_synthesize_shared_helper_code_comprehension_unit_return_type_any(
    tmp_path: Path,
) -> None:
    """Verifies that comprehension units with no explicit return type synthesize -> Any."""
    src = (
        "def f(items):\n"
        "    return [x * 2 for x in items]\n"
        "\n"
        "def g(items):\n"
        "    return [x * 2 for x in items]\n"
    )
    f_path = tmp_path / "comp_mod.py"
    f_path.write_text(src, encoding="utf-8")
    u1 = {
        "name": "f:listcomp",
        "file": str(f_path),
        "start": 2,
        "end": 2,
        "kind": "comprehension",
    }
    u2 = {
        "name": "g:listcomp",
        "file": str(f_path),
        "start": 5,
        "end": 5,
        "kind": "comprehension",
    }
    code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "-> Any:" in code
    assert "-> None:" not in code


def test_infer_helper_return_type_yield_from_dict_keys() -> None:
    """Verifies that yield from on a Dict/dict mapping type infers Iterator[KeyType]."""
    scope = {"has_yield": True, "yield_expr_names": [("yield_from", "mapping")]}
    meta1 = {"mapping": {"type": "Dict[str, int]"}}
    meta2 = {"mapping": {"type": "dict[str, int]"}}
    ret = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope,
        meta1,
        meta2,
    )
    assert ret == "Iterator[str]"


def test_infer_helper_return_type_yield_from_mapping_keys() -> None:
    """Verifies that yield from on a Mapping/mapping type infers Iterator[KeyType]."""
    scope = {"has_yield": True, "yield_expr_names": [("yield_from", "mapping")]}
    meta1 = {"mapping": {"type": "mapping[str, int]"}}
    meta2 = {"mapping": {"type": "mapping[str, int]"}}
    ret = _infer_helper_return_type(
        "Any",
        [],
        set(),
        scope,
        meta1,
        meta2,
    )
    assert ret == "Iterator[str]"


def test_synthesize_shared_helper_code_unpaired_output_fallback_aligned(
    tmp_path: Path,
) -> None:
    """Verifies that when clone output name sets match but positional orderings conflict,
    _pair_clone_outputs fails closed, safely falling back to positional type inference
    (Tuple[Any, Any]) rather than silently miscompiling or swapping return semantics.
    """
    src1 = (
        "def f(a: int, b: str):\n"
        "    return a, b\n"
    )
    src2 = (
        "def g(b: str, a: int):\n"
        "    return b, a\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 2, "name": "f", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 2, "name": "g", "kind": "function"}
    code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert code != ""
    assert "Tuple[Any, Any]" in code
    assert "return a, b" in code


def test_synthesize_shared_helper_code_unpaired_output_positional_aligned(
    tmp_path: Path,
) -> None:
    """Verifies that when clone output names differ across units, positional pairing
    correctly aligns counterpart types."""
    src1 = (
        "def f(x: int, y: str):\n"
        "    return x, y\n"
    )
    src2 = (
        "def g(a: int, b: str):\n"
        "    return a, b\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")
    u1 = {"file": str(f1), "start": 1, "end": 2, "name": "f", "kind": "function"}
    u2 = {"file": str(f2), "start": 1, "end": 2, "name": "g", "kind": "function"}
    code = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert code != ""
    assert "Tuple[int, str]" in code
    assert "return x, y" in code


def test_patch_subroutine_effective_units_with_precomputed_outputs(
    tmp_path: Path,
) -> None:
    """Verifies that generate_refactoring_patch uses effective units with precomputed outputs
    when resolving generator subroutine outputs."""
    src1 = (
        "def process1(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "    return total\n"
    )
    src2 = (
        "def process2(items):\n"
        "    count = 0\n"
        "    for y in items:\n"
        "        count += y\n"
        "        yield y\n"
        "    return count\n"
    )
    f1 = tmp_path / "p1.py"
    f2 = tmp_path / "p2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    u1 = {
        "file": str(f1),
        "start": 3,
        "end": 5,
        "name": "process1:for",
        "kind": "compound_block",
        "outputs": ["total", "x"],
    }
    u2 = {
        "file": str(f2),
        "start": 3,
        "end": 5,
        "name": "process2:for",
        "kind": "compound_block",
        "outputs": ["count", "y"],
    }
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True
    )
    assert patch != ""
    assert "yield from _shared" in patch
    assert "total = (yield from _shared" in patch
    assert "count = (yield from _shared" in patch
    assert "total, x" not in patch
    assert "count, y" not in patch


def test_file_patch_plan_content_digest_caching() -> None:
    """Verifies that _FilePatchPlan lazily computes and caches content_digest."""
    from pydoppelgangerhunt.fixer.patch import _FilePatchPlan

    plan = _FilePatchPlan(
        file_path=Path("test.py"),
        rel_path="test.py",
        orig_text="a = 1\nb = 2\n",
        is_new_file=False,
    )
    digest1 = plan.content_digest
    assert len(digest1) == 64
    assert digest1 == hashlib.sha256("a = 1\nb = 2\n".encode("utf-8")).hexdigest()
    digest2 = plan.content_digest
    assert digest1 == digest2
    assert plan._content_digest is not None


@pytest.mark.parametrize(
    ("helper_outs", "outs2", "cond_outs", "meta1", "meta2", "expected"),
    [
        (
            ["total"],
            ["count"],
            {"count"},
            {"total": {"type": "int"}},
            {"count": {"type": "int"}},
            "Optional[int]",
        ),
        (
            ["total"],
            ["count"],
            {"total"},
            {"total": {"type": "int"}},
            {"count": {"type": "int"}},
            "Optional[int]",
        ),
        (
            ["total"],
            ["count"],
            set(),
            {"total": {"type": "int"}},
            {"count": {"type": "int"}},
            "int",
        ),
        (
            ["a", "b"],
            ["x", "y"],
            {"y"},
            {"a": {"type": "int"}, "b": {"type": "str"}},
            {"x": {"type": "int"}, "y": {"type": "str"}},
            "Tuple[int, Optional[str]]",
        ),
        (
            ["a", "b"],
            ["x", "y"],
            {"a"},
            {"a": {"type": "int"}, "b": {"type": "str"}},
            {"x": {"type": "int"}, "y": {"type": "str"}},
            "Tuple[Optional[int], str]",
        ),
    ],
)
def test_infer_outputs_return_type_symmetric_conditional_outs(
    helper_outs: List[str],
    outs2: List[str],
    cond_outs: Set[str],
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    expected: str,
) -> None:
    """Verifies that conditional_outs checks both primary and paired clone variables."""
    res = _infer_outputs_return_type(
        helper_outputs=helper_outs,
        conditional_outs=cond_outs,
        meta1=meta1,
        meta2=meta2,
        type_merge_strategy="fallback_any",
        outputs2=outs2,
    )
    assert res == expected


def test_synthesize_shared_helper_code_strictness_propagation(tmp_path: Path) -> None:
    """Verifies that closure strictness parameters configure generator output synthesis."""
    code = (
        "def outer(items):\n"
        "    cb = lambda: total\n"
        "    for x in items:\n"
        "        total = x\n"
        "        yield x\n"
        "    return 0\n"
    )
    f = tmp_path / "strictness_mod.py"
    f.write_text(code, encoding="utf-8")
    u = {
        "file": str(f),
        "start": 3,
        "end": 5,
        "kind": "sliding_window",
        "name": "outer:stmts",
    }
    # Strict mode: fails closed due to pre-unit closure capturing 'total'
    res_strict = synthesize_shared_helper_code(
        u, u, repo_root=str(tmp_path), closure_strictness="strict"
    )
    assert res_strict == ""

    # Lenient mode: bypasses pre-unit closure check and synthesizes helper
    res_lenient = synthesize_shared_helper_code(
        u, u, repo_root=str(tmp_path), closure_strictness="lenient"
    )
    assert "def _shared_outer" in res_lenient
    assert "yield" in res_lenient

    # Direct skip_pre_unit_closures=True flag behaves identically
    res_skip = synthesize_shared_helper_code(
        u, u, repo_root=str(tmp_path), skip_pre_unit_closures=True
    )
    assert "def _shared_outer" in res_skip


def test_inspect_unit_scope_synthetic_wrapper_explicit_return_and_hazards(
    tmp_path: Path,
) -> None:
    """Verifies that explicit returns and hazards inside wrapped subroutines are detected."""
    # A subroutine unit with an indented yield and return inside a compound block
    code = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "        return x\n"
    )
    mod_file = tmp_path / "sub_wrapped.py"
    mod_file.write_text(code, encoding="utf-8")
    u = {
        "file": str(mod_file),
        "start": 3,
        "end": 4,
        "kind": "compound_block",
        "name": "outer:for",
    }
    scope = _inspect_unit_scope(u, repo_root=str(tmp_path))
    assert scope["has_yield"] is True
    assert scope["has_return"] is True
    assert scope["has_return_value"] is True
    assert "x" in scope["outputs"]
    # Embedded return hazard is correctly identified on the subroutine block
    assert "embedded_return" in scope["control_flow_hazards"]
    assert scope["is_control_flow_safe"] is False

    # Nested helper function inside subroutine should not leak return out to the subroutine
    code_nested = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        def helper():\n"
        "            return 99\n"
        "        yield helper()\n"
    )
    mod_file2 = tmp_path / "sub_nested.py"
    mod_file2.write_text(code_nested, encoding="utf-8")
    u2 = {
        "file": str(mod_file2),
        "start": 3,
        "end": 5,
        "kind": "compound_block",
        "name": "outer:for",
    }
    scope2 = _inspect_unit_scope(u2, repo_root=str(tmp_path))
    assert scope2["has_yield"] is True
    assert scope2["has_return"] is False
    assert scope2["has_return_value"] is False
    assert "embedded_return" not in scope2["control_flow_hazards"]


def test_process_func_does_not_flatten_user_inner_wrapper(tmp_path: Path) -> None:
    """Verifies that an inner function named _wrapper in a whole-function unit is not flattened."""
    code = (
        "def my_decorator(fn):\n"
        "    inner_var = 1\n"
        "    def _wrapper(*args, **kwargs):\n"
        "        wrapper_local = 2\n"
        "        return fn(*args, **kwargs)\n"
        "    return _wrapper\n"
    )
    f = tmp_path / "dec.py"
    f.write_text(code, encoding="utf-8")
    unit = {
        "file": str(f),
        "start": 1,
        "end": 6,
        "name": "my_decorator",
        "kind": "function",
    }
    scope_info = analyze_unit_variable_scope(unit, repo_root=str(tmp_path))
    assert "wrapper_local" not in scope_info.get("stores", set())
    assert "wrapper_local" not in scope_info.get("outputs", [])
    assert "_wrapper" in scope_info.get("outputs", [])


def test_infer_helper_return_type_subroutine_no_outputs_ignores_resolved_ret() -> None:
    """Verifies subroutine with no outputs and no return returns None, ignoring enclosing return."""
    # pylint: disable=protected-access
    ret = _infer_helper_return_type(
        helper_outputs=[],
        conditional_outs=set(),
        meta1={},
        meta2={},
        scope={"has_return": False, "has_return_value": False},
        resolved_ret="int",
        is_subroutine=True,
    )
    assert ret == "None"


def test_inspect_single_unit_scope_interface() -> None:
    """Verifies that inspect_single_unit_scope analyzes an individual unit with a single tree."""
    code = "def f():\n    a = 1\n    return a + 2\n"
    tree = ast.parse(code)
    unit = {"file": "mod.py", "start": 2, "end": 3, "source_text": code}
    info = inspect_single_unit_scope(unit, tree=tree)
    assert "inputs" in info
    assert "outputs" in info
    assert "definite_stores" in info


def test_yield_send_type_non_expr_contexts() -> None:
    """Verifies that yield expressions in non-ast.Expr parent contexts (return, call, if, assign)
    set has_yield_assignment=True to preserve send-type analysis."""
    # Context 1: return (yield x)
    code_ret = "def g():\n    return (yield 42)\n"
    u_ret = {"file": "m.py", "start": 2, "end": 2, "source_text": code_ret}
    info_ret = inspect_single_unit_scope(u_ret)
    assert info_ret.get("has_yield_assignment") is True

    # Context 2: f((yield x))
    code_call = "def g():\n    func((yield 1))\n"
    u_call = {"file": "m.py", "start": 2, "end": 2, "source_text": code_call}
    info_call = inspect_single_unit_scope(u_call)
    assert info_call.get("has_yield_assignment") is True

    # Context 3: if (yield x):
    code_if = "def g():\n    if (yield 1):\n        pass\n"
    u_if = {"file": "m.py", "start": 2, "end": 3, "source_text": code_if}
    info_if = inspect_single_unit_scope(u_if)
    assert info_if.get("has_yield_assignment") is True

    # Context 4: bare statement yield (parent is ast.Expr)
    code_expr = "def g():\n    yield 1\n"
    u_expr = {"file": "m.py", "start": 2, "end": 2, "source_text": code_expr}
    info_expr = inspect_single_unit_scope(u_expr)
    assert info_expr.get("has_yield_assignment") is False


def test_subroutine_async_preservation_in_async_func(tmp_path: Path) -> None:
    """Verifies that a pure synchronous subroutine block inside an async def does not
    inherit is_async=True and can be successfully refactored with a sync clone."""
    file1 = tmp_path / "mod1.py"
    file2 = tmp_path / "mod2.py"
    file1.write_text(
        "async def handle_request(req):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    res = a + b\n"
        "    await req.send(res)\n",
        encoding="utf-8",
    )
    file2.write_text(
        "def process_data():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    res = a + b\n"
        "    return res\n",
        encoding="utf-8",
    )
    u1 = {"file": str(file1), "start": 2, "end": 4, "is_subroutine": True}
    scope1 = inspect_single_unit_scope(u1, repo_root=str(tmp_path))
    assert scope1.get("is_async") is False
    assert scope1.get("has_yield") is False

    file_gen = tmp_path / "gen.py"
    file_gen.write_text(
        "async def gen_items():\n"
        "    yield 1\n",
        encoding="utf-8",
    )
    u_gen = {"file": str(file_gen), "start": 2, "end": 2, "is_subroutine": True}
    scope_gen = inspect_single_unit_scope(u_gen, repo_root=str(tmp_path))
    assert scope_gen.get("has_yield") is True


def test_generator_send_type_preserved_when_return_type_none() -> None:
    """Verifies that a generator with yield assignment but no outputs or return value
    preserves send type as Generator[yield_t, send_t, None] rather than
    falling back to Iterator."""
    res = _infer_helper_return_type(
        resolved_ret="Iterator[int]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": True,
            "yield_expr_names": [],
            "has_return_value": False,
        },
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res == "Generator[int, Any, None]"

    res_inferred = _infer_helper_return_type(
        resolved_ret="Generator[str, bytes, None]",
        helper_outputs=[],
        conditional_outs=set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": True,
            "yield_expr_names": [],
            "has_return_value": False,
        },
        meta1={},
        meta2={},
        is_async=False,
    )
    assert res_inferred == "Generator[str, bytes, None]"


def test_normalize_pipe_unions_variants() -> None:
    """Verifies PEP 604 pipe union normalization across simple, nested, and quoted types."""
    assert _normalize_pipe_unions("int | str") == "Union[int, str]"
    assert _normalize_pipe_unions("int | str | float") == "Union[int, str, float]"
    assert _normalize_pipe_unions("int | int") == "int"
    assert _normalize_pipe_unions("list[int | str]") == "list[Union[int, str]]"
    assert _normalize_pipe_unions("int | None") == "Optional[int]"
    assert _normalize_pipe_unions("None | int") == "Optional[int]"
    assert _normalize_pipe_unions("Union[int, None]") == "Optional[int]"
    assert _normalize_pipe_unions("Iterator[int | None]") == "Iterator[Optional[int]]"
    assert _normalize_pipe_unions("tuple[int | str, ...]") == "tuple[Union[int, str], ...]"
    assert _normalize_pipe_unions("Tuple[int | str, float]") == "Tuple[Union[int, str], float]"
    assert _normalize_pipe_unions("Union[int | str, float]") == "Union[int, str, float]"
    assert _normalize_pipe_unions("typing.Union[int | str, float]") == "Union[int, str, float]"
    assert _normalize_pipe_unions("int | typing.Union[str, float]") == "Union[int, str, float]"
    assert _normalize_pipe_unions("typing.Union[int]") == "int"
    assert (
        _normalize_pipe_unions("list[typing.Union[int | str, float]]")
        == "list[Union[int, str, float]]"
    )
    assert (
        _normalize_pipe_unions("Dict[str, list[int | None]]")
        == "Dict[str, list[Optional[int]]]"
    )
    assert _normalize_pipe_unions('Annotated[int, "a | b"]') == 'Annotated[int, "a | b"]'
    assert (
        _normalize_pipe_unions('Annotated[int | None, "a | b"]')
        == 'Annotated[Optional[int], "a | b"]'
    )
    assert _normalize_pipe_unions("'int | str'") == "'Union[int, str]'"
    assert _normalize_pipe_unions("int") == "int"
    assert _normalize_pipe_unions("") == ""
    assert _normalize_pipe_unions(None) == ""
    assert _normalize_pipe_unions(123) == "123"  # type: ignore[arg-type]


def test_infer_helper_return_type_pep604_pipe_union_in_container_yields() -> None:
    """Verifies that _infer_helper_return_type normalizes pipe unions in container yields."""
    # Container yield_from with list[int | str]
    res_list = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": False,
            "yield_expr_names": [("yield_from", "items")],
            "has_return_value": False,
        },
        meta1={"items": {"type": "list[int | str]"}},
        meta2={},
        is_async=False,
    )
    assert res_list == "Iterator[Union[int, str]]"

    # Container yield_from with tuple[int | str, ...]
    res_tuple = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": False,
            "yield_expr_names": [("yield_from", "items")],
            "has_return_value": False,
        },
        meta1={"items": {"type": "tuple[int | str, ...]"}},
        meta2={},
        is_async=False,
    )
    assert res_tuple == "Iterator[Union[int, str]]"

    # Direct yield with int | None
    res_yield = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope={
            "has_yield": True,
            "has_yield_assignment": False,
            "yield_expr_names": [("yield", "item")],
            "has_return_value": False,
        },
        meta1={"item": {"type": "int | None"}},
        meta2={},
        is_async=False,
    )
    assert res_yield == "Iterator[Optional[int]]"


def test_split_type_args_trailing_comma() -> None:
    """Verifies that _split_type_args ignores trailing commas and trailing whitespace."""
    assert _split_type_args("Tuple[int,]") == ["int"]
    assert _split_type_args("Tuple[int, ]") == ["int"]
    assert _split_type_args("Tuple[int, str, ]") == ["int", "str"]
    assert _split_type_args("Tuple[]") == []
    assert _normalize_pipe_unions("Tuple[int | str, ]") == "Tuple[Union[int, str]]"


def test_find_innermost_enclosing_node_none_end_lineno() -> None:
    """Verifies _find_innermost_enclosing_node handles AST nodes with end_lineno=None."""
    from pydoppelgangerhunt.fixer.source import _find_innermost_enclosing_node

    fn = ast.FunctionDef(
        name="foo",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[ast.Pass(lineno=2, end_lineno=None)],
        decorator_list=[],
        lineno=1,
    )
    fn.end_lineno = None  # Explicitly None
    mod = ast.Module(body=[fn], type_ignores=[])

    matched_res = _find_innermost_enclosing_node(
        source_text="",
        unit={"start": 1, "end": 1},
        node_types=(ast.FunctionDef,),
        tree=mod,
    )
    assert matched_res is not None
    matched, start, end = matched_res
    assert matched is fn
    assert start == 1
    assert end == 1


def test_node_is_effectively_async_scope_pruning() -> None:
    """Verifies _node_is_effectively_async prunes nested callable and class definitions."""
    from pydoppelgangerhunt.parser import _node_is_effectively_async

    # Inner sync generator inside an async def should not make a sync block effectively async
    code_sync_gen = (
        "if cond:\n"
        "    def sync_helper():\n"
        "        yield 1\n"
    )
    tree_sync = ast.parse(code_sync_gen)
    assert not _node_is_effectively_async(tree_sync.body[0], enclosing_is_async=True)

    # Inner async function inside a sync block should not make the outer sync block async
    code_async_func = (
        "if cond:\n"
        "    async def async_helper():\n"
        "        await something()\n"
    )
    tree_async = ast.parse(code_async_func)
    assert not _node_is_effectively_async(tree_async.body[0], enclosing_is_async=False)

    # Direct await inside the block IS async
    code_direct_await = (
        "if cond:\n"
        "    await something()\n"
    )
    tree_direct = ast.parse(code_direct_await)
    assert _node_is_effectively_async(tree_direct.body[0], enclosing_is_async=False)

    # Async list comprehension without explicit await is effectively async
    code_async_comp = (
        "if cond:\n"
        "    res = [x async for x in items]\n"
    )
    tree_comp = ast.parse(code_async_comp)
    assert _node_is_effectively_async(tree_comp.body[0], enclosing_is_async=False)

    # Async generator comprehension is effectively async
    code_async_gen = (
        "if cond:\n"
        "    res = (x async for x in items)\n"
    )
    tree_gen = ast.parse(code_async_gen)
    assert _node_is_effectively_async(tree_gen.body[0], enclosing_is_async=False)


def test_parse_candidates_def_precedes_async_def() -> None:
    """Verifies that synchronous def wrapper is attempted before async def wrapper in scope."""
    from pydoppelgangerhunt.fixer.scope import _inspect_unit_scope

    unit = {
        "source_text": "return 42\n",
        "file": "test.py",
        "start": 1,
        "end": 1,
    }
    res = _inspect_unit_scope(unit)
    assert res is not None
    assert res["is_async"] is False
    assert res["has_return_value"] is True


def test_unwrap_iterable_item_type_unions_and_iterables() -> None:
    """Verifies _unwrap_iterable_item_type handles top-level unions and iterables."""
    from pydoppelgangerhunt.fixer.synthesis import _unwrap_iterable_item_type

    assert _unwrap_iterable_item_type("List[int] | List[str]") == "Union[int, str]"
    assert _unwrap_iterable_item_type("Union[List[int], List[str]]") == "Union[int, str]"
    assert _unwrap_iterable_item_type("typing.List[int] | typing.Set[str]") == "Union[int, str]"
    assert _unwrap_iterable_item_type("Tuple[int, ...] | List[int]") == "int"
    assert _unwrap_iterable_item_type("Tuple[int, str]") == "Union[int, str]"
    assert _unwrap_iterable_item_type("Dict[str, int]") == "str"
    assert _unwrap_iterable_item_type("Generator[int, None, None]") == "int"
    assert _unwrap_iterable_item_type("mapping[str, int]") == "str"
    assert _unwrap_iterable_item_type("mutablemapping[str, int]") == "str"
    assert _unwrap_iterable_item_type("sequence[int]") == "int"
    assert _unwrap_iterable_item_type("iterable[str]") == "str"
    assert _unwrap_iterable_item_type("iterator[float]") == "float"
    assert _unwrap_iterable_item_type("collection[bytes]") == "bytes"
    assert _unwrap_iterable_item_type("int") is None
    assert _unwrap_iterable_item_type("Union[List[int], int]") is None


def test_infer_helper_return_type_yield_from_unsplit_pipe_union() -> None:
    """Verifies _infer_helper_return_type unwraps pipe union iterables in yield from."""
    from pydoppelgangerhunt.fixer.synthesis import _infer_helper_return_type

    scope: Dict[str, Any] = {
        "has_yield": True,
        "yield_expr_names": [("yield_from", "items")],
    }
    meta1: Dict[str, Any] = {"items": {"type": "List[int] | List[str]"}}
    meta2: Dict[str, Any] = {}
    res = _infer_helper_return_type(
        resolved_ret="Any",
        helper_outputs=[],
        conditional_outs=set(),
        scope=scope,
        meta1=meta1,
        meta2=meta2,
    )
    assert res == "Iterator[Union[int, str]]"


def test_subroutine_with_inner_wrapper_func_namespace_collision(tmp_path: Path) -> None:
    """Verifies subroutine containing inner def _wrapper() is not flattened."""
    code = (
        "def _wrapper():\n"
        "    hidden_var = 10\n"
        "    return hidden_var\n"
        "res = _wrapper()\n"
    )
    f = tmp_path / "sub.py"
    f.write_text(code, encoding="utf-8")
    unit = {
        "file": str(f),
        "start": 1,
        "end": 4,
        "name": "compound_block",
        "kind": "compound_block",
    }
    scope_info = analyze_unit_variable_scope(unit, repo_root=str(tmp_path))
    # hidden_var inside _wrapper must NOT be flattened into the subroutine's outer scope
    assert "hidden_var" not in scope_info.get("stores", set())
    assert "hidden_var" not in scope_info.get("outputs", [])
    stores = scope_info.get("stores", set())
    outputs = scope_info.get("outputs", [])
    assert "_wrapper" in stores or "_wrapper" in outputs


def test_normalize_pipe_unions_preserves_string_literal_pipes() -> None:
    """Verifies that _normalize_pipe_unions does not transform pipe characters inside
    Literal string arguments into Union types."""
    assert _normalize_pipe_unions('Literal["r | w"]') == 'Literal["r | w"]'
    assert _normalize_pipe_unions("typing.Literal['a | b']") == "typing.Literal['a | b']"
    assert (
        _normalize_pipe_unions('Literal["r | w", "x | y"]')
        == 'Literal["r | w", "x | y"]'
    )
    assert (
        _normalize_pipe_unions('int | Literal["r | w"]')
        == 'Union[int, Literal["r | w"]]'
    )
    assert (
        _normalize_pipe_unions("typing_extensions.Literal['a | b']")
        == "typing_extensions.Literal['a | b']"
    )
    assert (
        _normalize_pipe_unions("typing_extensions.Union[int | str, float]")
        == "Union[int, str, float]"
    )


def test_split_delimited_type_string_unmatched_closing_bracket() -> None:
    """Verifies that _split_delimited_type_string handles unmatched closing brackets safely."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _split_delimited_type_string,
        _split_pipe_union_args,
    )

    # With require_balanced=True, unmatched closing bracket returns []
    assert _split_delimited_type_string("int] | str", "|", require_balanced=True) == []
    assert _split_delimited_type_string("int) | str", "|", require_balanced=True) == []
    assert _split_delimited_type_string("int} | str", "|", require_balanced=True) == []

    # With require_balanced=False, fallback to unsplit string rather than partial split
    assert (
        _split_delimited_type_string("int] | str", "|", require_balanced=False)
        == ["int] | str"]
    )
    assert _split_pipe_union_args("int] | str") == ["int] | str"]


def test_split_type_args_and_delimited_string_malformed_inputs() -> None:
    """Verifies that _split_type_args and _split_delimited_type_string strictly reject
    malformed, unbalanced, or compound disjoint bracket inputs."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _split_delimited_type_string,
        _split_type_args,
    )

    # 1. Excess closing bracket at the end
    assert _split_type_args("Tuple[int]]") == []

    # 2. Compound types with disjoint brackets
    assert _split_type_args("Tuple[int] | List[str]") == []

    # 3. Delimited type string with premature closing bracket
    assert _split_delimited_type_string("a | b]", "|") == []

    # 4. Missing opening bracket or unclosed bracket
    assert _split_type_args("int]") == []
    assert _split_type_args("[int") == []
    assert _split_type_args("Tuple[[int]") == []

    # 5. String literal containing brackets inside generic
    assert _split_type_args('Literal["]"]') == ['"]"']
    assert _split_type_args('Literal["["]') == ['"["']


def test_flatten_union_args_typing_extensions() -> None:
    """Verifies that _flatten_union_args unnests typing_extensions.Union parameters."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _flatten_union_args,
        _normalize_pipe_unions,
    )

    assert _flatten_union_args(["typing_extensions.Union[int, str]"]) == ["int", "str"]
    assert _flatten_union_args(["Union[int, str]"]) == ["int", "str"]
    assert _flatten_union_args(["typing.Union[int, str]"]) == ["int", "str"]
    assert (
        _normalize_pipe_unions("int | typing_extensions.Union[str, float]")
        == "Union[int, str, float]"
    )
