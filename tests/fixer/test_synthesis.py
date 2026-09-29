"""Unit tests for fixer shared helper code synthesis, parameter ranks, and type inference."""

from __future__ import annotations

import ast
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
    _infer_helper_return_type,
    analyze_unit_variable_scope,
    is_subroutine_unit,
    resolve_clone_generator_subroutine_outputs,
)
from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=protected-access
    _pair_clone_outputs,
    _resolve_unit_ast_end_col,
    collect_downstream_read_names,
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


def test_format_call_arguments_order_preservation_and_custom_receivers() -> None:
    """Verifies that call argument formatting preserves name identity across load permutations and omits custom receivers."""
    from pydoppelgangerhunt.fixer.synthesis import _format_call_arguments  # pylint: disable=import-outside-toplevel

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
    from pydoppelgangerhunt.fixer import _infer_helper_return_type  # pylint: disable=import-outside-toplevel

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


def test_collect_downstream_read_names_scope_and_closure_capture() -> None:
    """Verifies that collect_downstream_read_names captures free variables and immediate class body reads, while respecting shadowed locals."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    # Case 1: Immediately executed class body (class C: value = x) and closure capture (def inner(): return x)
    code = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        return x\n"
        "\n"
        "    class C:\n"
        "        value = x\n"
        "\n"
        "    print(total)\n"
    )
    unit = {"start": 3, "end": 5}

    # With candidates: both 'total' and 'x' are captured downstream
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads == {"total", "x"}

    # Without candidates: all loaded names in outer scope and escaping nested scopes are found
    all_reads = collect_downstream_read_names(code, unit)
    assert all_reads == {"total", "x", "print"}

    # Case 2: Shadowed variables in nested functions/classes do NOT escape to outer scope
    code_shadowed = (
        "def outer(items):\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        yield x\n"
        "\n"
        "    def inner_param(x):\n"
        "        return x\n"
        "\n"
        "    def inner_local():\n"
        "        x = 10\n"
        "        return x\n"
        "\n"
        "    class ClassShadowed:\n"
        "        x = 10\n"
        "        value = x\n"
        "\n"
        "    print(total)\n"
    )
    reads_shadowed = collect_downstream_read_names(code_shadowed, unit, candidates={"total", "x"})
    assert reads_shadowed == {"total"}


def test_collect_downstream_read_names_positional_and_expression_boundaries() -> None:
    """Verifies same-line boundaries, same-expression, enclosing-expression, and multiline expression reads."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    # 1. Same-line unit boundary: statement after semicolon
    code_sameline = (
        "def f(items):\n"
        "    total = 0\n"
        "    for x in items: yield x; print(total)\n"
    )
    u_sameline = {"start": 3, "end": 3, "end_col": 27}
    reads_sameline = collect_downstream_read_names(code_sameline, u_sameline, candidates={"total", "x"})
    assert reads_sameline == {"total"}

    # 2. Output used later in the same expression: (yield x) + x
    code_same_expr = (
        "def f(x):\n"
        "    res = (yield x) + x\n"
        "    return res\n"
    )
    u_same_expr = {"start": 2, "end": 2, "end_col": 19}
    reads_same_expr = collect_downstream_read_names(code_same_expr, u_same_expr, candidates={"x"})
    assert reads_same_expr == {"x"}

    # 3. Output used in an enclosing call expression: func((yield x), x)
    code_call_expr = (
        "def f(x):\n"
        "    res = func((yield x), x)\n"
        "    return res\n"
    )
    u_call_expr = {"start": 2, "end": 2, "end_col": 25}
    reads_call_expr = collect_downstream_read_names(code_call_expr, u_call_expr, candidates={"x"})
    assert reads_call_expr == {"x"}

    # 4. Multiline expression: argument on subsequent line
    code_multiline = (
        "def f(x):\n"
        "    res = func(\n"
        "        (yield x),\n"
        "        x,\n"
        "    )\n"
        "    return res\n"
    )
    u_multiline = {"start": 3, "end": 3, "end_col": 17}
    reads_multiline = collect_downstream_read_names(code_multiline, u_multiline, candidates={"x"})
    assert reads_multiline == {"x"}

    # 5. Output used BEFORE the unit on the same line: not downstream
    code_prior = (
        "def f(x):\n"
        "    res = x + (yield x)\n"
        "    return res\n"
    )
    u_prior = {"start": 2, "end": 2, "end_col": 23}
    reads_prior = collect_downstream_read_names(code_prior, u_prior, candidates={"x"})
    assert reads_prior == set()


def test_resolve_generator_subroutine_outputs_preserves_per_side_necessity_and_safe_fallback(
) -> None:
    """Verifies that resolve_generator_subroutine_outputs pairs renamed outputs and guards
    against partial knowledge."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        GeneratorCloneSideData,
        resolve_generator_subroutine_outputs,
    )

    u1_outs = ["total", "x"]
    u2_outs = ["count", "x"]

    # 1. Both sides known: only the mapped total/count slot is needed, x is discarded
    res1 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}, definite={"total"}),
        GeneratorCloneSideData(u2_outs, downstream={"count"}, definite={"count"}),
    )
    assert res1 is not None
    out1, out2 = res1
    assert out1 == ["total"]
    assert out2 == ["count"]

    # 2. Unknown fallback with definite sets: prunes conditionally assigned loop variable x
    res2 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}, definite={"total"}),
        GeneratorCloneSideData(u2_outs, downstream=None, definite={"count"}),
    )
    assert res2 is not None
    fb_def1, fb_def2 = res2
    assert fb_def1 == ["total"]
    assert fb_def2 == ["count"]

    # 3. Unknown fallback without definite sets: unconstrained fallback preserves all paired outputs
    res3 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream={"total"}),
        GeneratorCloneSideData(u2_outs, downstream=None),
    )
    assert res3 is not None
    fb_out1, fb_out2 = res3
    assert fb_out1 == ["total", "x"]
    assert fb_out2 == ["count", "x"]

    # 4. Neither side needs outputs
    res4 = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(u1_outs, downstream=set()),
        GeneratorCloneSideData(u2_outs, downstream=set()),
    )
    assert res4 is not None
    none_out1, none_out2 = res4
    assert none_out1 == []
    assert none_out2 == []

    # 5. Unequal output counts where needed output has no counterpart: fails closed (returns None)
    res_unequal = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(["total"], downstream={"total"}),
        GeneratorCloneSideData(["count", "status"], downstream={"count"}),
    )
    assert res_unequal is None

    # 6. Downstream explicitly requires output assigned in loop: fails closed if not definite
    res_needed = resolve_generator_subroutine_outputs(
        GeneratorCloneSideData(["x"], downstream={"x"}, definite=set()),
        GeneratorCloneSideData(["x"], downstream={"x"}, definite=set()),
    )
    assert res_needed is None


def test_collect_downstream_read_names_comprehensions_and_walrus() -> None:
    """Verifies that comprehension targets do not pollute enclosing scope and walrus bindings are scoped correctly."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    # 1. Comprehension inside downstream closure: target variable 'x' is comprehension-local,
    # so 'return x' correctly reads outer 'x' as a free variable
    code_closure_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        _ = [x for x in items]\n"
        "        return x\n"
        "\n"
        "    return inner\n"
    )
    u_closure_comp = {"start": 2, "end": 3}
    reads_closure = collect_downstream_read_names(code_closure_comp, u_closure_comp, candidates={"x"})
    assert reads_closure == {"x"}

    # 2. Walrus operator (:=) inside comprehension: binds to enclosing function scope per PEP 572
    code_walrus = (
        "def outer(items):\n"
        "    for w in items:\n"
        "        yield w\n"
        "\n"
        "    def inner():\n"
        "        _ = [(w := y) for y in items]\n"
        "        return w\n"
        "\n"
        "    return inner\n"
    )
    u_walrus = {"start": 2, "end": 3}
    reads_walrus = collect_downstream_read_names(code_walrus, u_walrus, candidates={"w"})
    # 'w' is bound locally inside 'inner' by the walrus expression, not read from outer
    assert reads_walrus == set()

    # 3. Direct comprehension downstream: comprehension target is not a read of generator variable
    code_direct_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    res = [x for x in items]\n"
        "    return res\n"
    )
    u_direct = {"start": 2, "end": 3}
    reads_direct = collect_downstream_read_names(code_direct_comp, u_direct, candidates={"x"})
    assert reads_direct == set()

    # 4. Direct comprehension referencing outer variable in if-filter
    code_filter_comp = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    res = [y for y in items if y == x]\n"
        "    return res\n"
    )
    reads_filter = collect_downstream_read_names(code_filter_comp, u_direct, candidates={"x"})
    assert reads_filter == {"x"}

    # 5. Comprehension inside downstream ClassDef body: target is not class attribute, iter is free read
    code_class_comp = (
        "def outer(data):\n"
        "    for x in data:\n"
        "        yield x\n"
        "    class C:\n"
        "        items = [x for x in data]\n"
        "    return C\n"
    )
    reads_class = collect_downstream_read_names(code_class_comp, u_direct, candidates={"x", "data"})
    assert reads_class == {"data"}


def test_generator_clone_with_nested_genexpr(tmp_path: Path) -> None:
    """Verifies that nested generator expressions inside generator units do not leak comprehension targets into scope."""
    code1 = (
        "def process_matrix(matrix: list[list[int]]):\n"
        "    grand_total = 0\n"
        "    for row in matrix:\n"
        "        row_sum = sum(x * 2 for x in row)\n"
        "        grand_total += row_sum\n"
        "        yield row_sum\n"
        "    return grand_total\n"
    )
    code2 = (
        "def process_matrix_alt(matrix: list[list[int]]):\n"
        "    grand_total = 0\n"
        "    for row in matrix:\n"
        "        row_sum = sum(y * 2 for y in row)\n"
        "        grand_total += row_sum\n"
        "        yield row_sum\n"
        "    return grand_total\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "mod1.py",
        "start": 3,
        "end": 6,
        "name": "process_matrix",
        "kind": "compound_block",
    }
    u2 = {
        "file": "mod2.py",
        "start": 3,
        "end": 6,
        "name": "process_matrix_alt",
        "kind": "compound_block",
    }

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        source_text1=code1,
        source_text2=code2,
    )
    assert helper != ""
    assert "def _shared" in helper
    assert "yield row_sum" in helper
    assert "return grand_total" in helper
    # Comprehension targets x and y must not be parameters of the synthesized helper
    sig_line = helper.splitlines()[0]
    params_part = sig_line[sig_line.index("(") + 1 : sig_line.rindex(")")]
    param_names = [p.split(":")[0].strip() for p in params_part.split(",") if p.strip()]
    assert "x" not in param_names and "y" not in param_names
    assert "grand_total" in param_names

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "_shared" in patch


def test_generator_clone_with_yield_from_and_returns(tmp_path: Path) -> None:
    """Verifies that generator clones containing yield from delegate return values and propagate outputs correctly."""
    code1 = (
        "def stream_blocks(blocks: list[list[int]]):\n"
        "    total_count = 0\n"
        "    for b in blocks:\n"
        "        yield from b\n"
        "        total_count += len(b)\n"
        "    return total_count\n"
    )
    code2 = (
        "def stream_blocks_alt(blocks: list[list[int]]):\n"
        "    total_count = 0\n"
        "    for b in blocks:\n"
        "        yield from b\n"
        "        total_count += len(b)\n"
        "    return total_count\n"
    )
    f1 = tmp_path / "stream1.py"
    f2 = tmp_path / "stream2.py"
    f1.write_text(code1, encoding="utf-8")
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "stream1.py",
        "start": 3,
        "end": 5,
        "name": "stream_blocks",
        "kind": "compound_block",
    }
    u2 = {
        "file": "stream2.py",
        "start": 3,
        "end": 5,
        "name": "stream_blocks_alt",
        "kind": "compound_block",
    }

    helper = synthesize_shared_helper_code(
        u1,
        u2,
        repo_root=str(tmp_path),
        source_text1=code1,
        source_text2=code2,
    )
    assert helper != ""
    assert "yield from b" in helper
    assert "return total_count" in helper

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch != ""
    assert "(yield from _shared" in patch


def test_collect_downstream_read_names_nonlocal_and_global_declarations() -> None:
    """Verifies that downstream nonlocal declarations are captured as escaping free reads, while globals are excluded."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code_nonlocal = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        nonlocal x\n"
        "        return x\n"
        "    return inner\n"
    )
    u = {"start": 2, "end": 3}
    reads_nonlocal = collect_downstream_read_names(code_nonlocal, u, candidates={"x"})
    assert reads_nonlocal == {"x"}

    code_global = (
        "def outer(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "\n"
        "    def inner():\n"
        "        global x\n"
        "        return x\n"
        "    return inner\n"
    )
    reads_global = collect_downstream_read_names(code_global, u, candidates={"x"})
    # 'global x' accesses module global x, not outer's local x
    assert reads_global == set()

    # Pre-parsed tree parameter equivalence
    tree = ast.parse(code_nonlocal)
    reads_with_tree = collect_downstream_read_names(code_nonlocal, u, candidates={"x"}, tree=tree)
    assert reads_with_tree == {"x"}


def test_generator_clone_side_data_interface() -> None:
    """Verifies that GeneratorCloneSideData cleanly encapsulates per-side clone context."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        GeneratorCloneSideData,
        resolve_generator_subroutine_outputs,
    )

    side1 = GeneratorCloneSideData(
        outputs=["total", "x"],
        downstream={"total"},
        definite={"total"},
    )
    side2 = GeneratorCloneSideData(
        outputs=["count", "x"],
        downstream={"count"},
        definite={"count"},
    )

    resolved = resolve_generator_subroutine_outputs(side1, side2)
    assert resolved == (["total"], ["count"])

    # Fail closed on unequal output counts when counterpart is missing
    side_unpaired = GeneratorCloneSideData(
        outputs=["count", "extra", "status"],
        downstream={"count", "extra"},
        definite={"count", "extra"},
    )
    assert resolve_generator_subroutine_outputs(side1, side_unpaired) is None


def test_is_async_generator_with_return_value_policy() -> None:
    """Verifies that is_async_generator_with_return_value strictly enforces PEP 525 constraints."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        is_async_generator_with_return_value,
    )

    # Clean sync generator with return value (allowed in Python 3.3+)
    sync_gen = {"has_yield": True, "is_async": False, "has_return_value": True}
    assert not is_async_generator_with_return_value(sync_gen)

    # Clean async function with return value (allowed)
    async_fn = {"has_yield": False, "is_async": True, "has_return_value": True}
    assert not is_async_generator_with_return_value(async_fn)

    # Async generator without return value (allowed)
    async_gen_no_ret = {"has_yield": True, "is_async": True, "has_return_value": False}
    assert not is_async_generator_with_return_value(async_gen_no_ret)

    # Async generator with explicit return value (illegal under PEP 525)
    async_gen_with_ret = {"has_yield": True, "is_async": True, "has_return_value": True}
    assert is_async_generator_with_return_value(async_gen_with_ret)

    # Multi-scope evaluation: if the merged helper combines yield, async, and return value, it is rejected
    assert is_async_generator_with_return_value(async_gen_no_ret, async_gen_with_ret)
    assert is_async_generator_with_return_value(async_gen_no_ret, sync_gen)

    # When no scope contains return values and has_outputs is False, not rejected
    clean_async = {"has_yield": False, "is_async": True, "has_return_value": False}
    sync_gen_no_ret = {"has_yield": True, "is_async": False, "has_return_value": False}
    assert not is_async_generator_with_return_value(clean_async, sync_gen_no_ret)

    # Async generator attempting to propagate downstream outputs (illegal under PEP 525)
    assert is_async_generator_with_return_value(async_gen_no_ret, has_outputs=True)
    assert not is_async_generator_with_return_value(sync_gen, has_outputs=True)


def test_downstream_read_visitor_aug_assign() -> None:
    """Verifies that augmented assignments downstream of a unit are captured as reads."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def compute_running_total(items):\n"
        "    for x in items:\n"
        "        yield x\n"
        "    total += x\n"
        "    counts[0] += 1\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads == {"total", "x"}


def test_pair_clone_outputs_duplicate_names() -> None:
    """Verifies that _pair_clone_outputs does not raise KeyError on duplicate output names."""
    from pydoppelgangerhunt.fixer.binding import _pair_clone_outputs  # pylint: disable=import-outside-toplevel

    # Equal lengths with duplicates on one side (len(dedup) differs -> common names only)
    pairs1 = _pair_clone_outputs(["a", "b"], ["a", "a"])
    assert pairs1 == [("a", "a")]

    # Equal raw lengths with duplicate on one side and disjoint names (returns empty without KeyError)
    pairs2 = _pair_clone_outputs(["x", "y"], ["z", "z"])
    assert pairs2 == []

    # Equal deduplicated lengths with duplicates on both sides
    pairs3 = _pair_clone_outputs(["a", "b", "a"], ["c", "d", "c"])
    assert pairs3 == [("a", "c"), ("b", "d")]

    # Unequal raw lengths with duplicates
    pairs4 = _pair_clone_outputs(["a", "b", "a"], ["a", "a"])
    assert pairs4 == [("a", "a")]


def test_extract_nested_scope_free_reads_type_annotations() -> None:
    """Verifies that type annotations on parameters and return types in nested functions are captured as free reads."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def outer():\n"
        "    class MyType:\n"
        "        pass\n"
        "    class ReturnType:\n"
        "        pass\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "    def nested(param: MyType) -> ReturnType:\n"
        "        return None\n"
    )
    unit = {"start": 6, "end": 7}
    reads = collect_downstream_read_names(code, unit, candidates={"MyType", "ReturnType", "x"})
    assert reads is not None
    assert "MyType" in reads
    assert "ReturnType" in reads


def test_patch_subroutine_is_async_propagation(tmp_path: Path) -> None:
    """Verifies that is_async from enclosing functions propagates to s1/s2 and rejects illegal returns."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch  # pylint: disable=import-outside-toplevel

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


def test_pair_clone_outputs_positional_alignment() -> None:
    """Verifies that _pair_clone_outputs preserves 1-to-1 positional order even when variable names collide."""
    from pydoppelgangerhunt.fixer.binding import _pair_clone_outputs  # pylint: disable=import-outside-toplevel

    # Colliding variable names across different semantic positions
    pairs = _pair_clone_outputs(["a", "b"], ["b", "c"])
    assert pairs == [("a", "c"), ("b", "b")]

    # Swapped variable names with identical name set preserve canonical identity mapping
    pairs_swapped = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_swapped == [("x", "x"), ("y", "y")]


def test_extract_nested_scope_free_reads_outer_scope_shadowing() -> None:
    """Verifies that defaults, decorators, and annotations in nested functions are not wiped by inner local stores."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def outer():\n"
        "    for x in range(10):\n"
        "        yield x\n"
        "    def inner_default(a=x):\n"
        "        x = 1\n"
        "        return a + x\n"
        "    def dec(fn):\n"
        "        return fn\n"
        "    @dec(x)\n"
        "    def inner_dec():\n"
        "        x = 2\n"
        "        return x\n"
        "    def inner_ann() -> x:\n"
        "        x = 3\n"
        "        return x\n"
        "    class InnerClass(x):\n"
        "        x = 4\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"x", "dec"})
    assert reads is not None
    assert "x" in reads


def test_infer_outputs_return_type_counterpart_lookup() -> None:
    """Verifies that counterpart variable types from outputs2 take precedence over name collisions."""
    from pydoppelgangerhunt.fixer.synthesis import _infer_outputs_return_type  # pylint: disable=import-outside-toplevel

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


def test_downstream_reads_in_enclosing_class() -> None:
    """Verifies that collect_downstream_read_names traverses statements and methods in an enclosing class."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "class ProcessingEngine:\n"
        "    x = 10\n"
        "    total = x * 2\n"
        "    downstream_attr = total + 5\n"
        "    def get_total(self):\n"
        "        return total\n"
    )
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    assert "total" in reads


def test_caller_unit_dicts_immutable_during_patch(tmp_path: Path) -> None:
    """Verifies that generate_refactoring_patch does not mutate caller unit dictionaries in-place."""
    from pydoppelgangerhunt.fixer.patch import generate_refactoring_patch  # pylint: disable=import-outside-toplevel

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

    u1 = {"file": str(f1), "start": 2, "end": 3, "name": "fn1", "kind": "block"}
    u2 = {"file": str(f2), "start": 2, "end": 3, "name": "fn2", "kind": "block"}

    u1_copy = dict(u1)
    u2_copy = dict(u2)

    generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=False)

    assert u1 == u1_copy
    assert u2 == u2_copy
    assert "is_async" not in u1
    assert "is_async" not in u2
    assert "receiver_param" not in u1
    assert "receiver_param" not in u2


def test_synthesize_shared_helper_code_symmetric_global_filtering(tmp_path: Path) -> None:
    """Verifies that synthesize_shared_helper_code filters globals/nonlocals from both u1 and u2 outputs."""
    from pydoppelgangerhunt.fixer.synthesis import synthesize_shared_helper_code  # pylint: disable=import-outside-toplevel

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


def test_downstream_read_visitor_reaching_definitions_reassigned_variable() -> None:
    """Verifies that reaching definitions clear variables killed by unconditional assignments before reads."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    # Case 1: Unconditional reassignment kills reaching definition
    code1 = (
        "def worker():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    x = 10\n"
        "    print(x)\n"
    )
    unit1 = {"start": 2, "end": 3}
    reads1 = collect_downstream_read_names(code1, unit1, candidates={"x", "y"})
    assert reads1 is not None
    assert "x" not in reads1
    assert "y" not in reads1

    # Case 2: Read in RHS before kill retains the read
    code2 = (
        "def worker():\n"
        "    x = 1\n"
        "    x = x + 10\n"
    )
    unit2 = {"start": 2, "end": 2}
    reads2 = collect_downstream_read_names(code2, unit2, candidates={"x"})
    assert reads2 is not None
    assert "x" in reads2

    # Case 3: Conditional If without else does NOT kill reaching definition
    code3 = (
        "def worker():\n"
        "    x = 1\n"
        "    if condition:\n"
        "        x = 10\n"
        "    print(x)\n"
    )
    unit3 = {"start": 2, "end": 2}
    reads3 = collect_downstream_read_names(code3, unit3, candidates={"x"})
    assert reads3 is not None
    assert "x" in reads3

    # Case 4: Conditional If with else in both branches kills reaching definition
    code4 = (
        "def worker():\n"
        "    x = 1\n"
        "    if condition:\n"
        "        x = 10\n"
        "    else:\n"
        "        x = 20\n"
        "    print(x)\n"
    )
    unit4 = {"start": 2, "end": 2}
    reads4 = collect_downstream_read_names(code4, unit4, candidates={"x"})
    assert reads4 is not None
    assert "x" not in reads4

    # Case 5: AnnAssign with value kills, but without value does not kill
    code5_with_val = (
        "def worker():\n"
        "    x = 1\n"
        "    x: int = 10\n"
        "    print(x)\n"
    )
    assert "x" not in collect_downstream_read_names(code5_with_val, {"start": 2, "end": 2}, candidates={"x"})  # type: ignore[operator]

    code5_no_val = (
        "def worker():\n"
        "    x = 1\n"
        "    x: int\n"
        "    print(x)\n"
    )
    assert "x" in collect_downstream_read_names(code5_no_val, {"start": 2, "end": 2}, candidates={"x"})  # type: ignore[operator]


def test_infer_helper_return_type_untyped_generator_return_value() -> None:
    """Verifies that a generator with explicit return value and untyped return emits Generator[yield_t, None, Any]."""
    from pydoppelgangerhunt.fixer.synthesis import _infer_helper_return_type  # pylint: disable=protected-access

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
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        _extract_nested_scope_free_reads,
        collect_downstream_read_names,
    )

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


def test_same_line_unit_boundaries_semicolon_downstream_resolution() -> None:
    """Verifies that same-line units without column offsets correctly resolve downstream statements after semicolons."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def runner():\n"
        "    yield x; print(total)\n"
    )
    # Unit on line 2 with NO start_col or end_col
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "x"})
    assert reads is not None
    # total is downstream of yield x on the same line
    assert "total" in reads
    # x is inside the unit, so it should NOT be reported as a downstream read
    assert "x" not in reads


def test_downstream_read_visitor_try_except_else_kills() -> None:
    """Verifies that exception handler kills do not leak into node.orelse in visit_Try."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as total:\n"
        "        pass\n"
        "    else:\n"
        "        print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_resolve_unit_ast_end_col_multiline_unit_statements() -> None:
    """Verifies that multi-line units ending on a line with multiple statements include all statements up through the line end."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def worker():\n"
        "    a = 1\n"
        "    total = 10; count = 20\n"
        "    print(res)\n"
    )
    # Multi-line unit covering lines 2 to 3
    unit = {"start": 2, "end": 3}
    reads = collect_downstream_read_names(code, unit, candidates={"total", "count", "res"})
    assert reads is not None
    # total and count are on line 3 (end line of multi-line unit), so they are NOT downstream reads
    assert "total" not in reads
    assert "count" not in reads


def test_downstream_read_visitor_try_finally_unconditional_kills() -> None:
    """Verifies that unconditional assignments in finally: kill reaching definitions downstream."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        do_something()\n"
        "    finally:\n"
        "        total = 99\n"
        "    print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code_killed, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads

    code_retained = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        do_something()\n"
        "    finally:\n"
        "        pass\n"
        "    print(total)\n"
    )
    reads_retained = collect_downstream_read_names(code_retained, unit, candidates={"total"})
    assert reads_retained is not None
    assert "total" in reads_retained


def test_downstream_read_visitor_try_except_else_joint_kills() -> None:
    """Verifies that reaching definitions are killed only when all try/except/else branches assign."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    # All branches assign total -> killed downstream
    code_all_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    except ValueError:\n"
        "        total = 2\n"
        "    except TypeError:\n"
        "        total = 3\n"
        "    print(total)\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code_all_killed, unit, candidates={"total"})
    assert reads is not None
    assert "total" not in reads

    # One except branch does not assign total -> total reaches downstream
    code_partial = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    except ValueError:\n"
        "        pass\n"
        "    except TypeError:\n"
        "        total = 3\n"
        "    print(total)\n"
    )
    reads_partial = collect_downstream_read_names(code_partial, unit, candidates={"total"})
    assert reads_partial is not None
    assert "total" in reads_partial

    # Try body does not assign, but else and all except assign -> killed downstream
    code_else_killed = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"
        "        total = 2\n"
        "    else:\n"
        "        total = 1\n"
        "    print(total)\n"
    )
    reads_else = collect_downstream_read_names(code_else_killed, unit, candidates={"total"})
    assert reads_else is not None
    assert "total" not in reads_else

    # Exception name is deleted at except block exit in Python 3 -> reaches downstream
    code_as_name = (
        "def worker():\n"
        "    total = 10\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as total:\n"
        "        pass\n"
        "    print(total)\n"
    )
    reads_as_name = collect_downstream_read_names(code_as_name, unit, candidates={"total"})
    assert reads_as_name is not None
    assert "total" in reads_as_name


def test_downstream_read_visitor_chained_with_context_managers() -> None:
    """Verifies that earlier context manager targets kill prior reaching definitions for later items."""
    from pydoppelgangerhunt.fixer.binding import collect_downstream_read_names  # pylint: disable=import-outside-toplevel

    code = (
        "def worker():\n"
        "    x = 10\n"
        "    with open_mgr() as x, use_mgr(x) as y:\n"
        "        pass\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"x"})
    assert reads is not None
    # x was killed by open_mgr() as x before use_mgr(x) was evaluated
    assert "x" not in reads

    # Also verify intra-item evaluation: open_mgr(x) evaluates BEFORE as x kills it
    code_intra = (
        "def worker():\n"
        "    x = 10\n"
        "    with open_mgr(x) as x:\n"
        "        pass\n"
    )
    reads_intra = collect_downstream_read_names(code_intra, unit, candidates={"x"})
    assert reads_intra is not None
    assert "x" in reads_intra


def test_class_scope_visitor_global_and_nonlocal() -> None:
    """Verifies that global declarations in class bodies are excluded from free reads and class stores."""
    from pydoppelgangerhunt.fixer.binding import _extract_nested_scope_free_reads  # pylint: disable=import-outside-toplevel

    code_global = (
        "class MyClass:\n"
        "    global g_val\n"
        "    g_val = 100\n"
        "    x = g_val\n"
        "    y = outer_read\n"
    )
    tree_g = ast.parse(code_global)
    class_node = tree_g.body[0]
    escaped_g = _extract_nested_scope_free_reads(class_node)  # type: ignore[arg-type]
    assert "g_val" not in escaped_g
    assert "outer_read" in escaped_g

    code_nonlocal = (
        "def outer():\n"
        "    n_val = 1\n"
        "    class Inner:\n"
        "        nonlocal n_val\n"
        "        n_val = 2\n"
        "        z = n_val\n"
    )
    tree_nl = ast.parse(code_nonlocal)
    func_node = tree_nl.body[0]
    inner_class_node = func_node.body[1]  # type: ignore[attr-defined]
    escaped_nl = _extract_nested_scope_free_reads(inner_class_node)  # type: ignore[arg-type]
    # n_val is nonlocal to Inner, so it escapes Inner as a free read referencing outer scope
    assert "n_val" in escaped_nl


def test_infer_outputs_return_type_single_output_precedence() -> None:
    """Verifies that explicit single-output type takes precedence over enclosing resolved_ret unless Any."""
    from pydoppelgangerhunt.fixer.synthesis import _infer_outputs_return_type  # pylint: disable=import-outside-toplevel

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


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 type_params syntax requires Python 3.12+")
def test_extract_nested_scope_free_reads_pep695_type_params() -> None:
    """Verifies that PEP 695 type parameter scopes do not treat type vars as free reads."""
    from pydoppelgangerhunt.fixer.binding import _extract_nested_scope_free_reads  # pylint: disable=import-outside-toplevel

    code = (
        "def inner[T: BoundType](val: T) -> T:\n"
        "    return val\n"
    )
    tree = ast.parse(code)
    fn_node = tree.body[0]
    escaped = _extract_nested_scope_free_reads(fn_node)  # type: ignore[arg-type]
    assert "T" not in escaped
    assert "BoundType" in escaped


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

    # Name-based fallback when kind is unspecified
    assert is_subroutine_unit({"name": "process_module:10-25"}) is True
    assert is_subroutine_unit({"name": "plain_function"}) is False
    assert is_subroutine_unit({}) is False


def test_resolve_unit_ast_end_col_bounds_validation() -> None:
    """Verifies bounds validation in _resolve_unit_ast_end_col."""
    tree = ast.parse("x = 10\ny = 20\n")
    # Inverted start and end coordinates
    assert _resolve_unit_ast_end_col(tree, {"start": 5, "end": 2}) is None
    # Zero or negative start
    assert _resolve_unit_ast_end_col(tree, {"start": 0, "end": 2}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": -1, "end": 2}) is None
    # Zero or negative end
    assert _resolve_unit_ast_end_col(tree, {"start": 1, "end": 0}) is None
    assert _resolve_unit_ast_end_col(tree, {"start": 1, "end": -1}) is None


def test_resolve_clone_generator_subroutine_outputs_precomputed_and_fallback(
    tmp_path: Path,
) -> None:
    """Verifies generator subroutine output resolution with precomputed outputs and fallback."""
    # Case 1: Precomputed outputs present on u1 and u2
    u1_pre = {"outputs": ["a", "b"]}
    u2_pre = {"outputs": ["x", "y"]}
    s1 = {"globals": ["b"], "nonlocals": []}
    s2 = {"globals": [], "nonlocals": ["y"]}
    res = resolve_clone_generator_subroutine_outputs(u1_pre, u2_pre, s1, s2)
    assert res is not None
    outs1, outs2 = res
    assert outs1 == ["a"]
    assert outs2 == ["x"]

    # Case 2: On-disk file resolution and downstream reads
    src1 = (
        "def gen1():\n"
        "    total = 0\n"
        "    yield total\n"
        "    print(total)\n"
    )
    src2 = (
        "def gen2():\n"
        "    total = 0\n"
        "    yield total\n"
        "    print(total)\n"
    )
    f1 = tmp_path / "mod1.py"
    f2 = tmp_path / "mod2.py"
    f1.write_text(src1, encoding="utf-8")
    f2.write_text(src2, encoding="utf-8")

    unit1 = {"file": str(f1), "start": 2, "end": 3}
    unit2 = {"file": str(f2), "start": 2, "end": 3}
    scope1 = {"outputs": ["total"], "definite_stores": ["total"], "inputs": []}
    scope2 = {"outputs": ["total"], "definite_stores": ["total"], "inputs": []}
    res_disk = resolve_clone_generator_subroutine_outputs(
        unit1, unit2, scope1, scope2, repo_root=str(tmp_path)
    )
    assert res_disk is not None
    assert res_disk == (["total"], ["total"])

    # Case 3: Fail-closed counterpart mismatch on unequal arity with needed downstream output
    src3_extra = (
        "def gen1():\n"
        "    total = 0\n"
        "    extra = 1\n"
        "    yield total\n"
        "    print(total, extra)\n"
    )
    scope1_extra = {
        "outputs": ["total", "extra"],
        "definite_stores": ["total", "extra"],
        "inputs": [],
    }
    scope2_single = {"outputs": ["other"], "definite_stores": ["other"], "inputs": []}
    res_mismatch = resolve_clone_generator_subroutine_outputs(
        unit1, unit2, scope1_extra, scope2_single, source_text1=src3_extra, source_text2=src2
    )
    assert res_mismatch is None


def test_downstream_read_visitor_record_downstream_read_and_comprehensions() -> None:
    """Verifies that _record_downstream_read respects comprehension scoping and killed variables."""
    code = (
        "def runner():\n"
        "    data = [1, 2, 3]\n"
        "    res = [item for item in data]\n"
        "    def nested():\n"
        "        return [val for val in data]\n"
        "    class Helper:\n"
        "        stored = [elem for elem in data]\n"
        "    total = 0\n"
        "    total += 5\n"
        "    return total\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(
        code, unit, candidates={"data", "item", "val", "elem", "total"}
    )
    assert reads is not None
    assert "data" in reads
    assert "item" not in reads
    assert "val" not in reads
    assert "elem" not in reads
    assert "total" not in reads


def test_class_body_comprehension_free_reads() -> None:
    """Verifies that class-body comprehensions do not resolve class attributes (PEP 227)."""
    code = (
        "def outer():\n"
        "    x = 10\n"
        "    class C:\n"
        "        x = 100\n"
        "        values = [x for _ in range(1)]\n"
        "    return C\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"x"})
    assert reads is not None
    assert "x" in reads


def test_conditional_named_expr_does_not_kill_variables() -> None:
    """Verifies that conditionally evaluated named expressions do not kill reaching definitions."""
    code = (
        "def compute(flag, items):\n"
        "    total = 10\n"
        "    if flag and (total := 20):\n"
        "        pass\n"
        "    return total\n"
    )
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_pair_clone_outputs_swapped_names_and_positional_roles() -> None:
    """Verifies positional pairing across renamed variable roles and identity on permutations."""
    # Equal arity with different names: preserves positional sequence
    pairs_pos = _pair_clone_outputs(["x", "y"], ["b", "a"])
    assert pairs_pos == [("x", "b"), ("y", "a")]

    # Identical name sets preserve identity mapping to avoid swapping variables on reordering
    pairs_ident = _pair_clone_outputs(["x", "y"], ["y", "x"])
    assert pairs_ident == [("x", "x"), ("y", "y")]

    # Mismatched arity: only common names paired
    pairs_mismatched = _pair_clone_outputs(["a", "b"], ["b"])
    assert pairs_mismatched == [("b", "b")]


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


def test_split_type_args_nested_bracket_depth() -> None:
    """Verifies bracket-depth aware splitting of nested generic type arguments."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _split_type_args,
        _infer_helper_return_type,
    )

    # Empty or non-generic strings
    assert _split_type_args("int") == []
    assert _split_type_args("") == []

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


def test_except_handler_name_not_treated_as_escaping_free_read() -> None:
    """Verifies ast.ExceptHandler.name is recorded as local store and not an escaping read."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        _extract_nested_scope_free_reads,
    )
    import ast  # pylint: disable=import-outside-toplevel

    # Nested function with except handler
    func_src = (
        "def nested():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as err:\n"
        "        return str(err)\n"
    )
    node = ast.parse(func_src).body[0]
    assert isinstance(node, ast.FunctionDef)
    free = _extract_nested_scope_free_reads(node)
    assert "err" not in free

    # Nested class with except handler in body
    cls_src = (
        "class NestedClass:\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as err:\n"
        "        err_msg = str(err)\n"
    )
    cls_node = ast.parse(cls_src).body[0]
    assert isinstance(cls_node, ast.ClassDef)
    cls_free = _extract_nested_scope_free_reads(cls_node)
    assert "err" not in cls_free


def test_analyze_unit_variable_scope_does_not_mutate_caller_unit() -> None:
    """Verifies analyze_unit_variable_scope does not mutate the passed unit dictionary in-place."""
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
    )

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
    from pydoppelgangerhunt.fixer.scope import (  # pylint: disable=import-outside-toplevel
        analyze_unit_variable_scope,
    )

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


def test_resolve_clone_generator_subroutine_outputs_requires_both_outputs() -> None:
    """Verifies resolve_clone_generator_subroutine_outputs requires both sides to have outputs."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        resolve_clone_generator_subroutine_outputs,
    )

    u1 = {"outputs": ["a"], "start": 2, "end": 2}
    u2 = {"start": 2, "end": 2}  # No precomputed outputs
    scope1 = {"outputs": ["a"], "definite_stores": ["a"], "inputs": []}
    scope2 = {"outputs": ["b"], "definite_stores": ["b"], "inputs": []}

    src1 = "def f1():\n    a = 1\n    return a\n"
    src2 = "def f2():\n    b = 2\n    return b\n"
    res = resolve_clone_generator_subroutine_outputs(
        u1, u2, scope1, scope2, source_text1=src1, source_text2=src2
    )
    assert res is not None
    outs1, outs2 = res
    assert outs1 == ["a"]
    assert outs2 == ["b"]


def test_downstream_reads_preserve_exception_alias_reassigned_in_same_handler() -> None:
    """Verifies reassigning an exception alias in handler preserves downstream reads."""
    from pydoppelgangerhunt.fixer.binding import (  # pylint: disable=import-outside-toplevel
        collect_downstream_read_names,
    )

    code = (
        "def process():\n"
        "    try:\n"
        "        do_work()\n"
        "    except ValueError as err:\n"
        "        err = 'custom: ' + str(err)\n"
        "        log(err)\n"
    )
    # Unit is line 5: "err = 'custom: ' + str(err)"
    unit = {"start": 5, "end": 5}
    reads = collect_downstream_read_names(code, unit, candidates={"err"})
    assert reads is not None
    assert "err" in reads

    # Conversely, if unit is inside try body, the subsequent except handler DOES kill err
    code_try = (
        "def process_try():\n"
        "    try:\n"
        "        err = 'initial'\n"
        "    except ValueError as err:\n"
        "        log(err)\n"
    )
    unit_try = {"start": 3, "end": 3}
    reads_try = collect_downstream_read_names(code_try, unit_try, candidates={"err"})
    assert reads_try is not None
    assert "err" not in reads_try


def test_infer_helper_return_type_yield_from_tuples() -> None:
    """Verifies that yield_from on Tuple types infers Union or homogeneous element types."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _infer_helper_return_type,
    )

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


def test_collect_downstream_read_names_loop_carried_dependence() -> None:
    """Verifies that reads earlier in an enclosing loop body are captured as downstream reads."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        collect_downstream_read_names,
    )

    code = (
        "def process_batches(batches):\n"
        "    total = 0\n"
        "    for batch in batches:\n"
        "        log(total)\n"
        "        for x in batch:\n"
        "            total += x\n"
        "            yield x\n"
    )
    # Unit is lines 5 to 7: inner loop where total is updated
    unit = {"start": 5, "end": 7}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    assert "total" in reads


def test_collect_downstream_read_names_try_finally_exceptional_path() -> None:
    """Verifies that try body stores do not kill variables in finally block across
    exceptional paths."""
    from pydoppelgangerhunt.fixer.dataflow import (  # pylint: disable=import-outside-toplevel
        collect_downstream_read_names,
    )

    code = (
        "def run_transaction():\n"
        "    total = 10\n"
        "    try:\n"
        "        total = 1\n"
        "    finally:\n"
        "        print(total)\n"
    )
    # Unit is line 2: "total = 10"
    unit = {"start": 2, "end": 2}
    reads = collect_downstream_read_names(code, unit, candidates={"total"})
    assert reads is not None
    # print(total) in finally could run after an exception in try before total = 1,
    # so line 2's total must be considered read downstream.
    assert "total" in reads


def test_return_type_precedence_whole_function_vs_subroutine() -> None:
    """Verifies that whole-function units preserve declared return type over inferred
    local store."""
    from pydoppelgangerhunt.fixer.synthesis import (  # pylint: disable=import-outside-toplevel
        _infer_helper_return_type,
    )

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





