"""Unit tests for fixer shared helper code synthesis, parameter ranks, and type inference."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from pydoppelgangerhunt import (
    check_units_overlap,
    extract_unit_source_code,
    generate_clone_diff,
    generate_refactoring_patch,
    synthesize_refactoring_suggestion,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _format_call_arguments,
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
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto")
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
    cls_helper = synthesize_shared_helper_code(u_c1, u_c2, method_binding="method")
    assert "    @classmethod\n    def _shared_create_a_create_b(cls, val: int) -> int:" in cls_helper
    assert "Call site:\n            cls._shared_create_a_create_b(...)" in cls_helper

    # Explicit module mode falls back to module-level helper with self: Any
    mod_helper = synthesize_shared_helper_code(u1, u2, method_binding="module")
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
    code = synthesize_shared_helper_code(u1, u2)
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
    helper = synthesize_shared_helper_code(u1, u2)
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
    from pydoppelgangerhunt.fixer import _base_unit_name  # pylint: disable=protected-access

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
    from pydoppelgangerhunt.fixer import filter_overlapping_clone_units  # pylint: disable=import-outside-toplevel

    u_none_file: Dict[str, Any] = {"file": None, "start": 1, "end": 5, "name": "test"}
    assert filter_overlapping_clone_units([u_none_file]) == [u_none_file]
    assert not check_units_overlap(u_none_file, {"file": "a.py", "start": 1, "end": 5})

    u_none_name: Dict[str, Any] = {"file": "a.py", "start": 1, "end": 5, "name": None}
    suggestion = synthesize_refactoring_suggestion(u_none_name, u_none_name)
    assert "[REFACTOR SUGGESTION]" in suggestion

def test_batch_36_path_resolution_and_same_file_matching(tmp_path: Any) -> None:
    """Tests Batch 36: robust path normalization and resolution across duplicate units."""
    from pydoppelgangerhunt.fixer import (
        _is_same_file_path,
        _normalize_file_path,
        check_units_overlap,
        filter_overlapping_clone_units,
        generate_refactoring_patch,
        synthesize_shared_helper_code,
    )

    # 1. Base string equivalence and anchor handling
    assert not _is_same_file_path("", "foo.py")
    assert not _is_same_file_path("foo.py", "")
    assert _is_same_file_path("foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo/bar.py#hash1", "foo/bar.py#hash2")
    assert _is_same_file_path("./foo/bar.py", "foo/bar.py")
    assert _is_same_file_path("foo\\bar.py", "foo/bar.py")
    assert _normalize_file_path("") == ""
    assert _normalize_file_path("./mod.py#h").endswith("mod.py")

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
    # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import generate_refactoring_patch, synthesize_shared_helper_code
    from pydoppelgangerhunt.parser import harvest_file_units

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
    from pydoppelgangerhunt.fixer import check_units_overlap, synthesize_shared_helper_code

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
    from pydoppelgangerhunt.fixer import _infer_helper_return_type  # pylint: disable=import-outside-toplevel

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
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        _infer_helper_return_type,
        synthesize_shared_helper_code,
    )

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
