"""Unit tests for fixer enclosing scope inspection, receiver binding, and method kind determination."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pydoppelgangerhunt import (
    analyze_unit_variable_scope,
    find_enclosing_class,
    find_enclosing_function,
    generate_refactoring_patch,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _is_method_of_class,
    _populate_unit_receiver_metadata,
)
from pydoppelgangerhunt.parser import harvest_file_units


def test_scope_binding_instance_and_class_detection(tmp_path: Path) -> None:
    """Verifies that scope analysis identifies instance and class method bindings."""
    code = (
        "class Service:\n"
        "    def run_instance(self, delta: int) -> int:\n"
        "        self.total = self.base_value + delta\n"
        "        return self.total\n"
        "\n"
        "    @classmethod\n"
        "    def run_class(cls, factor: int) -> int:\n"
        "        cls.count += factor\n"
        "        return cls.count\n"
        "\n"
        "    def run_plain(x: int) -> int:\n"
        "        return x * 2\n"
    )
    src = tmp_path / "service.py"
    src.write_text(code, encoding="utf-8")

    u_inst = {"file": str(src), "start": 2, "end": 4, "name": "run_instance", "kind": "function"}
    scope_inst = analyze_unit_variable_scope(u_inst)
    assert scope_inst["has_instance_binding"] is True
    assert scope_inst["has_class_binding"] is False
    assert scope_inst["binding_kind"] == "instance"
    assert scope_inst["inputs"][0] == "self"
    assert "self.base_value" in scope_inst["instance_attrs"] or "self.total" in scope_inst["instance_attrs"]

    u_cls = {"file": str(src), "start": 6, "end": 9, "name": "run_class", "kind": "function"}
    scope_cls = analyze_unit_variable_scope(u_cls)
    assert scope_cls["has_class_binding"] is True
    assert scope_cls["has_instance_binding"] is False
    assert scope_cls["binding_kind"] == "class"
    assert scope_cls["inputs"][0] == "cls"
    assert "cls.count" in scope_cls["class_attrs"]

    u_plain = {"file": str(src), "start": 11, "end": 12, "name": "run_plain", "kind": "function"}
    scope_plain = analyze_unit_variable_scope(u_plain)
    assert scope_plain["has_instance_binding"] is False
    assert scope_plain["has_class_binding"] is False
    assert scope_plain["binding_kind"] is None

def test_find_enclosing_class_and_function(tmp_path: Path) -> None:
    """Verifies locating enclosing class definitions and functions including nested scopes."""
    code = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def compute(self, x: int) -> int:\n"
        "            total = x + 1\n"
        "            return total\n"
        "\n"
        "def standalone(a: int) -> int:\n"
        "    return a * 2\n"
    )
    src = tmp_path / "nested.py"
    src.write_text(code, encoding="utf-8")

    # Inside Inner method
    u_inner = {"file": str(src), "start": 3, "end": 5, "name": "compute", "kind": "function"}
    enc_class = find_enclosing_class(code, u_inner)
    assert enc_class is not None
    assert enc_class["name"] == "Inner"
    assert enc_class["start"] == 2
    assert enc_class["indent"] == "    "
    assert enc_class["method_indent"] == "        "

    enc_fn = find_enclosing_function(code, u_inner)
    assert enc_fn is not None
    assert enc_fn["name"] == "compute"
    assert enc_fn["start"] == 3

    # Compound block inside compute
    u_block = {"file": str(src), "start": 4, "end": 4, "name": "compute:Assign", "kind": "compound_block"}
    enc_block_cls = find_enclosing_class(code, u_block)
    assert enc_block_cls is not None
    assert enc_block_cls["name"] == "Inner"
    enc_block_fn = find_enclosing_function(code, u_block)
    assert enc_block_fn is not None
    assert enc_block_fn["name"] == "compute"

    # Top-level standalone function has no enclosing class
    u_standalone = {"file": str(src), "start": 7, "end": 8, "name": "standalone", "kind": "function"}
    assert find_enclosing_class(code, u_standalone) is None
    standalone_fn = find_enclosing_function(code, u_standalone)
    assert standalone_fn is not None
    assert standalone_fn["name"] == "standalone"

    # Edge cases: empty text, syntax error, zero start line
    assert find_enclosing_class("", u_inner) is None
    assert find_enclosing_class("def broken(: pass", u_inner) is None
    assert find_enclosing_class(code, {"start": 0, "end": 0}) is None
    assert find_enclosing_function("", u_inner) is None
    assert find_enclosing_function("def broken(: pass", u_inner) is None
    assert find_enclosing_function(code, {"start": 0, "end": 0}) is None

def test_differing_receiver_kinds_in_same_class(tmp_path: Path) -> None:
    """Verifies that clones with different receiver kinds fall back to module helper."""
    f = tmp_path / "mixed_receivers.py"
    code = (
        "class Handler:\n"
        "    def inst_worker(self, key: str) -> str:\n"
        "        res = f'key:{key}'\n"
        "        return res\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, key: str) -> str:\n"
        "        res = f'key:{key}'\n"
        "        return res\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "inst_worker",
        "kind": "function",
        "enclosing_class": "Handler",
    }
    u2 = {
        "file": str(f),
        "start": 7,
        "end": 9,
        "name": "cls_worker",
        "kind": "function",
        "enclosing_class": "Handler",
    }

    # In auto mode, differing receivers must fall back to module helper
    patch_auto = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch_auto
    assert "+def _shared_inst_worker_cls_worker(key: str) -> str:" in patch_auto
    assert "self." not in patch_auto.split("def cls_worker")[1]

    # Explicit method binding must also fall back to module helper because differing receivers cannot share one descriptor
    patch_method = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch_method
    assert "+def _shared_inst_worker_cls_worker(key: str) -> str:" in patch_method
    assert "return _shared_inst_worker_cls_worker(key)" in patch_method

def test_same_file_cross_class_method_binding_fallback(tmp_path: Path) -> None:
    """Verifies that clones across different classes fall back to module helper even with method_binding='method'."""
    f = tmp_path / "cross_class.py"
    code = (
        "class Alpha:\n"
        "    def work(self, v: int) -> int:\n"
        "        return v * 10\n"
        "\n"
        "class Beta:\n"
        "    def work(self, v: int) -> int:\n"
        "        return v * 10\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {"file": str(f), "start": 2, "end": 3, "name": "work", "kind": "function", "enclosing_class": "Alpha"}
    u2 = {"file": str(f), "start": 6, "end": 7, "name": "work", "kind": "function", "enclosing_class": "Beta"}

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "+def _shared_work(" in patch
    # Helper must be at module level, not inside Beta
    assert "class Beta" in patch

def test_class_body_comprehension_clones_use_module_binding(tmp_path: Path) -> None:
    """Verifies that clones directly in class body fall back to module helper and avoid self._shared in class body."""
    f = tmp_path / "table.py"
    code = (
        "class ConfigTable:\n"
        "    A = [k.upper() for k in ('x', 'y')]\n"
        "    B = [k.upper() for k in ('x', 'y')]\n"
    )
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 2,
        "name": "ConfigTable:listcomp_L2",
        "kind": "comprehension",
        "enclosing_class": "ConfigTable",
    }
    u2 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "ConfigTable:listcomp_L3",
        "kind": "comprehension",
        "enclosing_class": "ConfigTable",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "self." not in patch
    assert "+def _shared" in patch

def test_class_suite_tab_and_two_space_indentation(tmp_path: Path) -> None:
    """Verifies that suite indentation is dynamically derived for tab and 2-space classes."""
    tab_code = (
        "class TabbedClass:\n"
        "\tdef m1(self, x: int) -> int:\n"
        "\t\treturn x + 1\n"
        "\n"
        "\tdef m2(self, x: int) -> int:\n"
        "\t\treturn x + 1\n"
    )
    f_tab = tmp_path / "tabbed.py"
    f_tab.write_text(tab_code, encoding="utf-8")
    u_tab = {"file": str(f_tab), "start": 2, "end": 3, "name": "m1", "kind": "function"}
    meta_tab = find_enclosing_class(tab_code, u_tab)
    assert meta_tab is not None
    assert meta_tab["method_indent"] == "\t"

    u_tab2 = {"file": str(f_tab), "start": 5, "end": 6, "name": "m2", "kind": "function"}
    patch_tab = generate_refactoring_patch(
        [(0.95, u_tab, u_tab2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch_tab
    assert "+\tdef _shared_m1_m2(self" in patch_tab
    assert "+\t\treturn " in patch_tab

    two_space_code = (
        "class TwoSpaceClass:\n"
        "  def m1(self, x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "  def m2(self, x: int) -> int:\n"
        "    return x + 1\n"
    )
    f_two = tmp_path / "two.py"
    f_two.write_text(two_space_code, encoding="utf-8")
    u_two = {"file": str(f_two), "start": 2, "end": 3, "name": "m1", "kind": "function"}
    meta_two = find_enclosing_class(two_space_code, u_two)
    assert meta_two is not None
    assert meta_two["method_indent"] == "  "

def test_receiver_bound_mixed_kind_declines_replacement(tmp_path: Path) -> None:
    """Verifies that differing receiver kinds with referenced receivers decline replacement."""
    code = (
        "class Handler:\n"
        "    def inst_worker(self, x: int) -> int:\n"
        "        return self.value + x\n"
        "\n"
        "    @classmethod\n"
        "    def cls_worker(cls, x: int) -> int:\n"
        "        return cls.value + x\n"
    )
    f = tmp_path / "mixed_referenced.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "inst_worker",
        "kind": "function",
        "enclosing_class": "Handler",
        "receiver_kind": "instance",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 7,
        "name": "cls_worker",
        "kind": "function",
        "enclosing_class": "Handler",
        "receiver_kind": "class",
    }

    # Synthesis must return empty string
    helper = synthesize_shared_helper_code(u1, u2)
    assert helper == ""

    # Patch generation must skip / return empty patch
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch == ""

def test_class_body_in_factory_function_uses_module_binding(tmp_path: Path) -> None:
    """Verifies that class body units in a local class defined inside a factory function use module binding."""
    code = (
        "def make_class():\n"
        "    class LocalClass:\n"
        "        vals1 = [x * 2 for x in range(10)]\n"
        "        vals2 = [x * 2 for x in range(10)]\n"
        "    return LocalClass\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "vals1",
        "kind": "comprehension",
        "enclosing_class": "LocalClass",
    }
    u2 = {
        "file": str(f),
        "start": 4,
        "end": 4,
        "name": "vals2",
        "kind": "comprehension",
        "enclosing_class": "LocalClass",
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch
    # Must NOT emit self. inside class creation body
    assert "self._shared" not in patch
    # Must synthesize module helper
    assert "+def _shared" in patch

def test_nested_class_static_method_uses_dunder_class(tmp_path: Path) -> None:
    """Verifies that nested static methods in Outer.Inner delegate via __class__._shared."""
    code = (
        "class Outer:\n"
        "    class Inner:\n"
        "        @staticmethod\n"
        "        def add1(a: int, b: int) -> int:\n"
        "            return a + b\n"
        "\n"
        "        @staticmethod\n"
        "        def add2(a: int, b: int) -> int:\n"
        "            return a + b\n"
    )
    f = tmp_path / "nested_static.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 4,
        "end": 5,
        "name": "add1",
        "kind": "function",
        "enclosing_class": "Inner",
        "receiver_kind": "static",
        "is_static": True,
    }
    u2 = {
        "file": str(f),
        "start": 8,
        "end": 9,
        "name": "add2",
        "kind": "function",
        "enclosing_class": "Inner",
        "receiver_kind": "static",
        "is_static": True,
    }

    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch
    assert "@staticmethod" in patch
    assert "__class__._shared" in patch
    assert "Inner._shared" not in patch

def test_comprehension_inside_method_uses_module_binding(tmp_path: Path) -> None:
    """Verifies that duplicate comprehensions inside methods use module binding without self._shared."""
    code = (
        "class Worker:\n"
        "    def run_a(self, data: list) -> list:\n"
        "        return [x * 2 for x in data]\n"
        "\n"
        "    def run_b(self, data: list) -> list:\n"
        "        return [x * 2 for x in data]\n"
    )
    f = tmp_path / "worker_comp.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 3,
        "end": 3,
        "name": "run_a:listcomp_L3",
        "kind": "comprehension",
        "enclosing_class": "Worker",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 6,
        "name": "run_b:listcomp_L6",
        "kind": "comprehension",
        "enclosing_class": "Worker",
    }
    patch = generate_refactoring_patch(
        [(0.95, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="auto",
        replace_clones=True,
    )
    assert patch
    # Must synthesize module helper, not method helper inside Worker
    assert "+def _shared_run_a_run_b(" in patch
    # Must not invoke self._shared inside run_a or run_b
    assert "self._shared_run_a_run_b" not in patch
    assert "_shared_run_a_run_b(" in patch

def test_parentheses_on_staticmethod_and_classmethod_decorators(tmp_path: Path) -> None:
    """Verifies that @staticmethod() and @classmethod() call decorators are recognized properly."""
    code = (
        "class CallDec:\n"
        "    @staticmethod()\n"
        "    def s_fn(x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
        "\n"
        "    @classmethod()\n"
        "    def c_fn(cls, x: int) -> int:\n"
        "        y = x\n"
        "        return y + 1\n"
    )
    f = tmp_path / "calldec.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1)
    by_name = {u["name"]: u for u in units}

    assert by_name["s_fn"]["is_static"] is True
    assert by_name["s_fn"]["receiver_kind"] == "static"

    assert by_name["c_fn"]["is_static"] is False
    assert by_name["c_fn"]["receiver_kind"] == "class"

    enc_s = find_enclosing_function(code, by_name["s_fn"])
    assert enc_s is not None
    assert enc_s["is_static"] is True

    enc_c = find_enclosing_function(code, by_name["c_fn"])
    assert enc_c is not None
    assert enc_c["is_class_method"] is True

def test_local_class_methods_not_scoped_as_closures_of_factory_function(tmp_path: Path) -> None:
    """Verifies that methods of classes defined in functions are scoped to the class rather than as closures."""
    code = (
        "def make_handler():\n"
        "    class LocalHandler:\n"
        "        def handle(self, item: str) -> str:\n"
        "            def nested_sub():\n"
        "                item_clean = item.strip()\n"
        "                temp = item_clean.lower()\n"
        "                return temp\n"
        "            return nested_sub()\n"
        "    return LocalHandler\n"
    )
    f = tmp_path / "local_cls.py"
    f.write_text(code, encoding="utf-8")
    units = harvest_file_units(str(f), str(tmp_path), min_lines=2, min_tokens=1, harvest_closures=True)
    by_name = {u["name"]: u for u in units}

    # handle is a method of LocalHandler, not a closure of make_handler
    assert "handle" in by_name
    handle_u = by_name["handle"]
    assert handle_u["kind"] == "function"
    assert handle_u["enclosing_class"] == "LocalHandler"
    assert handle_u["receiver_kind"] == "instance"

    # nested_sub is a closure of handle
    assert "handle:nested_sub" in by_name
    sub_u = by_name["handle:nested_sub"]
    assert sub_u["kind"] == "closure"
    assert sub_u["enclosing_class"] == "LocalHandler"

def test_instance_method_paired_with_module_function_declines_replacement_when_receiver_accessed(tmp_path: Path) -> None:
    """Verifies that pairing an instance method accessing self.val with a module function declines replacement."""
    code1 = (
        "class Service:\n"
        "    def compute(self, x: int) -> int:\n"
        "        val = self.offset\n"
        "        return val + x * 2\n"
    )
    code2 = (
        "def compute(x: int) -> int:\n"
        "    val = 10\n"
        "    return val + x * 2\n"
    )
    f1 = tmp_path / "srv.py"
    f1.write_text(code1, encoding="utf-8")
    f2 = tmp_path / "util.py"
    f2.write_text(code2, encoding="utf-8")

    u1 = {
        "file": "srv.py", "start": 2, "end": 4, "name": "Service:compute",
        "kind": "function", "enclosing_class": "Service", "receiver_kind": "instance"
    }
    u2 = {
        "file": "util.py", "start": 1, "end": 3, "name": "compute",
        "kind": "function", "enclosing_class": None, "receiver_kind": None
    }

    # Synthesis must return "" because receiver kinds differ and u1 accesses self
    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    # Patch generation must skip this clone pair
    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch == ""

def test_same_name_classes_in_different_factories_not_same_class(tmp_path: Path) -> None:
    """Verifies that two identically named local classes in different factory functions are not treated as same class."""
    code = (
        "def factory_one():\n"
        "    class Config:\n"
        "        def process(self, x: int) -> int:\n"
        "            y = x * 2\n"
        "            return y + 1\n"
        "    return Config\n"
        "\n"
        "def factory_two():\n"
        "    class Config:\n"
        "        def process(self, x: int) -> int:\n"
        "            y = x * 2\n"
        "            return y + 1\n"
        "    return Config\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")

    u1 = {"file": "factories.py", "start": 3, "end": 5, "name": "process", "kind": "function"}
    u2 = {"file": "factories.py", "start": 10, "end": 12, "name": "process", "kind": "function"}

    # In auto mode, because start lines differ (line 2 vs line 9), they cannot share a private class method
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto", repo_root=str(tmp_path))
    # Helper should fall back to module-level helper (no self parameter omission or @classmethod/method indent)
    assert helper.startswith("def _shared_process(self: Any, x: int) -> int:")

def test_instance_method_paired_with_module_function_omits_receiver_parameter(tmp_path: Path) -> None:
    """Verifies that an instance method paired with a module function without receiver references omits self."""
    code = (
        "class Service:\n"
        "    def foo(self, x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
        "\n"
        "def bar(x: int) -> int:\n"
        "        y = x * 2\n"
        "        return y + 1\n"
    )
    f = tmp_path / "service_mod.py"
    f.write_text(code, encoding="utf-8")
    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "foo",
        "kind": "function",
        "enclosing_class": "Service",
        "receiver_kind": "instance",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "bar",
        "kind": "function",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper.startswith("def _shared_foo_bar(x: int) -> int:")
    assert "self" not in helper.splitlines()[0]

    patch = generate_refactoring_patch([(1.0, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert "return _shared_foo_bar(x)" in patch
    assert "_shared_foo_bar(self" not in patch

def test_direct_method_verification_distinguishes_closures(tmp_path: Path) -> None:
    """Verifies that direct method verification distinguishes class methods from nested closures."""
    code = (
        "class Worker:\n"
        "    def outer_method(self, a: int) -> int:\n"
        "        def inner_closure(b: int) -> int:\n"
        "            y = b * 2\n"
        "            z = y + 1\n"
        "            w = z * 3\n"
        "            return w\n"
        "        return inner_closure(a)\n"
    )
    f = tmp_path / "worker.py"
    f.write_text(code, encoding="utf-8")

    # Harvest units to check parser hierarchy scoping
    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=2, min_tokens=5, harvest_closures=True)
    by_name = {u["name"]: u for u in units}
    assert by_name["outer_method"]["enclosing_class"] == "Worker"
    assert by_name["outer_method"]["receiver_kind"] == "instance"
    assert by_name["outer_method:inner_closure"]["enclosing_class"] == "Worker"
    assert by_name["outer_method:inner_closure"]["receiver_kind"] is None

    # Verify find_enclosing_class and _is_method_of_class
    cls_meta = find_enclosing_class(code, by_name["outer_method"])
    fn_outer = find_enclosing_function(code, by_name["outer_method"])
    fn_inner = find_enclosing_function(code, by_name["outer_method:inner_closure"])

    assert _is_method_of_class(fn_outer, cls_meta) is True
    assert _is_method_of_class(fn_inner, cls_meta) is False

def test_separate_scopes_identical_class_name_enclosing_class_start(tmp_path: Path) -> None:
    """Verifies that identically named classes in different scopes have distinct enclosing_class_start."""
    code = (
        "def factory_one():\n"
        "    class Config:\n"
        "        def process(self, a: int, b: int) -> int:\n"
        "            x = a * 10\n"
        "            y = b * 20\n"
        "            z = x + y\n"
        "            return z\n"
        "    return Config\n\n"
        "def factory_two():\n"
        "    class Config:\n"
        "        def process(self, a: int, b: int) -> int:\n"
        "            x = a * 10\n"
        "            y = b * 20\n"
        "            z = x + y\n"
        "            return z\n"
        "    return Config\n"
    )
    f = tmp_path / "factories.py"
    f.write_text(code, encoding="utf-8")

    units = harvest_file_units(str(f), repo_root=str(tmp_path), min_lines=3, min_tokens=5)
    processes = [u for u in units if u["name"] == "process"]
    assert len(processes) == 2
    u1, u2 = processes[0], processes[1]
    assert u1["enclosing_class"] == "Config"
    assert u2["enclosing_class"] == "Config"
    assert u1["enclosing_class_start"] is not None
    assert u2["enclosing_class_start"] is not None
    assert u1["enclosing_class_start"] != u2["enclosing_class_start"]

    # In auto mode, separate classes must NOT synthesize a private instance method
    helper = synthesize_shared_helper_code(u1, u2, method_binding="auto", repo_root=str(tmp_path))
    assert not helper.startswith("    def _shared_process(self")
    assert "def _shared_process(self" in helper or "def _shared_process(" in helper

def test_receiver_isolation_in_factory_function(tmp_path: Path) -> None:
    """Verifies that nested class/method receiver accesses do not pollute the outer function's scope."""
    code = (
        "def build_service(mult: int):\n"
        "    class Inner:\n"
        "        def calc(self, v: int) -> int:\n"
        "            return self.mult * v\n"
        "    return Inner()\n"
    )
    f = tmp_path / "factory.py"
    f.write_text(code, encoding="utf-8")

    u = {"file": str(f), "start": 1, "end": 5, "name": "build_service", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))
    assert scope["has_instance_binding"] is False
    assert scope["has_class_binding"] is False
    assert "self" not in scope["inputs"]
    assert "mult" in scope["inputs"]

def test_populate_unit_receiver_metadata_enclosing_class_population(tmp_path: Path) -> None:
    """Verifies that _populate_unit_receiver_metadata populates missing enclosing_class."""
    code = (
        "class Model:\n"
        "    def predict(self, val: int) -> int:\n"
        "        return val * 10\n"
    )
    f = tmp_path / "model.py"
    f.write_text(code, encoding="utf-8")

    unit = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "name": "predict",
        "kind": "function",
    }
    _populate_unit_receiver_metadata(unit, repo_root=str(tmp_path))
    assert unit.get("enclosing_class") == "Model"
    assert unit.get("enclosing_class_start") == 1
    assert unit.get("receiver_kind") == "instance"

def test_nested_closure_receiver_attribute_tracking(tmp_path: Path) -> None:
    """Verifies that attributes accessed or mutated inside nested closures are attributed to the outer receiver."""
    code = (
        "class Pipeline:\n"
        "    def run(self, items: list) -> list:\n"
        "        def transform(x: int) -> int:\n"
        "            self.count += 1\n"
        "            return x + self.offset\n"
        "        return [transform(i) for i in items]\n"
    )
    f = tmp_path / "closure_receiver.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 6, "name": "run", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.count" in scope["attrs_read"]
    assert "self.count" in scope["attrs_written"]
    assert "self.offset" in scope["attrs_read"]
    assert "self.count" in scope["instance_attrs"]
    assert "self.offset" in scope["instance_attrs"]
    assert scope["has_receiver_access"] is True

def test_nested_class_receiver_attribute_isolation(tmp_path: Path) -> None:
    """Verifies that nested class methods do not leak their own self attributes to the enclosing method."""
    code = (
        "class Outer:\n"
        "    def execute(self, val: int) -> int:\n"
        "        class Inner:\n"
        "            def helper(self, x: int) -> int:\n"
        "                self.inner_data = x * 2\n"
        "                return self.inner_data\n"
        "        return Inner().helper(val)\n"
    )
    f = tmp_path / "nested_class_receiver.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 2, "end": 7, "name": "execute", "kind": "function"}
    scope = analyze_unit_variable_scope(u, repo_root=str(tmp_path))

    assert "self.inner_data" not in scope["attrs_read"]
    assert "self.inner_data" not in scope["attrs_written"]
    assert "self.inner_data" not in scope["instance_attrs"]

def test_find_enclosing_function_defensive(tmp_path: Path) -> None:
    """Verifies that find_enclosing_function handles functions without decorators or metadata safely."""
    code = "def sample():\n    pass\n"
    f = tmp_path / "sample_mod.py"
    f.write_text(code, encoding="utf-8")
    u = {"file": str(f), "start": 1, "end": 2, "name": "sample", "kind": "function"}
    meta = find_enclosing_function(code, u)
    assert meta is not None
    assert meta["name"] == "sample"
    assert not meta["is_static"]
    assert not meta["is_class_method"]

def test_class_level_comprehension_receiver_kind_is_none(tmp_path: Path) -> None:
    """Verifies that a comprehension in a class scope has receiver kind 'none'."""
    code = (
        "class Container:\n"
        "    items = [x * 2 for x in range(5)]\n"
    )
    f = tmp_path / "cls_comp.py"
    f.write_text(code, encoding="utf-8")
    u_comp = {
        "file": str(f),
        "start": 2,
        "end": 2,
        "name": "Container:listcomp_L2",
        "kind": "comprehension",
        "enclosing_class": "Container",
        "enclosing_class_start": 1,
    }
    helper = synthesize_shared_helper_code(u_comp, u_comp, repo_root=str(tmp_path))
    assert "self: Any" not in helper
    assert "def _shared" in helper

def test_batch_54_fixer_path_resolution_and_receiver_metadata(tmp_path: Path) -> None:
    """Batch 54: Test _normalize_file_path and _populate_unit_receiver_metadata edge cases."""
    # pylint: disable=import-outside-toplevel
    from pydoppelgangerhunt.fixer import (
        _normalize_file_path,
        _populate_unit_receiver_metadata,
        find_enclosing_class,
        find_enclosing_function,
    )

    # 1. Test _normalize_file_path with existing cwd-relative file and repo_root
    pyproject = "pyproject.toml"
    assert _normalize_file_path(pyproject, repo_root=str(tmp_path)).endswith("pyproject.toml")
    assert _normalize_file_path("") == ""

    # 2. Test _populate_unit_receiver_metadata with file existing relative to tmp_path
    mod_code = (
        "class Service:\n"
        "    @classmethod\n"
        "    def create(cls, data):\n"
        "        return cls(data)\n"
    )
    mod_file = tmp_path / "service.py"
    mod_file.write_text(mod_code, encoding="utf-8")

    unit_cls = {"file": "service.py", "start": 3, "end": 4, "name": "create"}
    _populate_unit_receiver_metadata(unit_cls, repo_root=str(tmp_path))
    assert unit_cls.get("enclosing_class") == "Service"
    assert unit_cls.get("receiver_kind") == "class"

    # 3. Test find_enclosing_class and find_enclosing_function edge cases
    assert find_enclosing_class("", unit_cls) is None
    assert find_enclosing_function("", unit_cls) is None

def test_receiver_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing different receiver attributes are rejected."""
    src = (
        "class StateManager:\n"
        "    def update_alpha(self, val: int) -> int:\n"
        "        self.alpha = val * 2\n"
        "        return self.alpha\n"
        "\n"
        "    def update_beta(self, val: int) -> int:\n"
        "        self.beta = val * 2\n"
        "        return self.beta\n"
    )
    f = tmp_path / "state_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "update_alpha",
        "file": "state_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "StateManager",
    }
    u2 = {
        "name": "update_beta",
        "file": "state_mismatch.py",
        "start": 6,
        "end": 8,
        "kind": "function",
        "enclosing_class": "StateManager",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_receiver_chained_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing different chained receiver attributes are rejected."""
    src = (
        "class ServiceDriver:\n"
        "    def configure_timeout(self, setting: int) -> int:\n"
        "        self.config.timeout = setting\n"
        "        return self.config.timeout\n"
        "\n"
        "    def configure_retries(self, setting: int) -> int:\n"
        "        self.config.retries = setting\n"
        "        return self.config.retries\n"
    )
    f = tmp_path / "service_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "configure_timeout",
        "file": "service_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ServiceDriver",
    }
    u2 = {
        "name": "configure_retries",
        "file": "service_mismatch.py",
        "start": 6,
        "end": 8,
        "kind": "function",
        "enclosing_class": "ServiceDriver",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_receiver_cross_class_attribute_mismatch_rejected(tmp_path: Path) -> None:
    """Verifies that cross-class method clones accessing different attributes are rejected."""
    src = (
        "class ModelA:\n"
        "    def fetch(self) -> int:\n"
        "        res = self.primary_data + 1\n"
        "        return res\n"
        "\n"
        "class ModelB:\n"
        "    def fetch(self) -> int:\n"
        "        res = self.secondary_data + 1\n"
        "        return res\n"
    )
    f = tmp_path / "models_mismatch.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "fetch",
        "file": "models_mismatch.py",
        "start": 2,
        "end": 4,
        "kind": "function",
        "enclosing_class": "ModelA",
    }
    u2 = {
        "name": "fetch",
        "file": "models_mismatch.py",
        "start": 7,
        "end": 9,
        "kind": "function",
        "enclosing_class": "ModelB",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_receiver_identical_attributes_accepted_and_executed(tmp_path: Path) -> None:
    """Verifies that methods accessing identical receiver attributes are refactored cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class Ledger:\n"
        "    def __init__(self, initial: int) -> None:\n"
        "        self.balance: int = initial\n"
        "\n"
        "    def credit_direct(self, amount: int) -> int:\n"
        "        self.balance += amount\n"
        "        return self.balance\n"
        "\n"
        "    def credit_wire(self, amount: int) -> int:\n"
        "        self.balance += amount\n"
        "        return self.balance\n"
    )
    f = tmp_path / "ledger.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "credit_direct",
        "file": "ledger.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "Ledger",
    }
    u2 = {
        "name": "credit_wire",
        "file": "ledger.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "Ledger",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert "_shared_credit_direct_credit_wire(self, amount: int) -> int:" in helper
    assert "self.balance += amount" in helper

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "return self._shared_credit_direct_credit_wire(amount)" in patch

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
            "from ledger import Ledger; "
            "led = Ledger(100); "
            "r1 = led.credit_direct(50); "
            "r2 = led.credit_wire(25); "
            "print(r1, r2, led.balance)",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "150 175 175"

def test_mangled_private_attribute_module_binding_rejected(tmp_path: Path) -> None:
    """Verifies that methods accessing mangled private attributes reject module-level binding."""
    src = (
        "class Vault:\n"
        "    def __init__(self, key: str) -> None:\n"
        "        self.__key = key\n"
        "\n"
        "    def access_alpha(self) -> str:\n"
        "        tag = 'v1_'\n"
        "        return tag + self.__key\n"
        "\n"
        "    def access_beta(self) -> str:\n"
        "        tag = 'v1_'\n"
        "        return tag + self.__key\n"
    )
    f = tmp_path / "vault_mod.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "access_alpha",
        "file": "vault_mod.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "Vault",
    }
    u2 = {
        "name": "access_beta",
        "file": "vault_mod.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "Vault",
    }

    helper = synthesize_shared_helper_code(u1, u2, method_binding="module", repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="module",
        replace_clones=True,
    )
    assert patch == ""

def test_mangled_private_attribute_cross_class_rejected(tmp_path: Path) -> None:
    """Verifies that cross-class clones accessing mangled private attributes are rejected."""
    src = (
        "class NodeAlpha:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.__val = val\n"
        "    def get_metric(self) -> int:\n"
        "        scale = 10\n"
        "        return self.__val * scale\n"
        "\n"
        "class NodeBeta:\n"
        "    def __init__(self, val: int) -> None:\n"
        "        self.__val = val\n"
        "    def get_metric(self) -> int:\n"
        "        scale = 10\n"
        "        return self.__val * scale\n"
    )
    f = tmp_path / "cross_vault.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "get_metric",
        "file": "cross_vault.py",
        "start": 4,
        "end": 6,
        "kind": "function",
        "enclosing_class": "NodeAlpha",
    }
    u2 = {
        "name": "get_metric",
        "file": "cross_vault.py",
        "start": 11,
        "end": 13,
        "kind": "function",
        "enclosing_class": "NodeBeta",
    }

    helper = synthesize_shared_helper_code(u1, u2, repo_root=str(tmp_path))
    assert helper == ""

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        replace_clones=True,
    )
    assert patch == ""

def test_mangled_private_attribute_same_class_method_binding_accepted_and_executed(
    tmp_path: Path,
) -> None:
    """Verifies that methods accessing mangled attributes within the same class refactor and execute cleanly."""
    import subprocess  # pylint: disable=import-outside-toplevel

    src = (
        "class SecretHolder:\n"
        "    def __init__(self, token: str) -> None:\n"
        "        self.__token = token\n"
        "\n"
        "    def reveal_first(self) -> str:\n"
        "        prefix = 'token:'\n"
        "        return prefix + self.__token\n"
        "\n"
        "    def reveal_second(self) -> str:\n"
        "        prefix = 'token:'\n"
        "        return prefix + self.__token\n"
    )
    f = tmp_path / "secret_holder.py"
    f.write_text(src, encoding="utf-8")

    u1 = {
        "name": "reveal_first",
        "file": "secret_holder.py",
        "start": 5,
        "end": 7,
        "kind": "function",
        "enclosing_class": "SecretHolder",
    }
    u2 = {
        "name": "reveal_second",
        "file": "secret_holder.py",
        "start": 9,
        "end": 11,
        "kind": "function",
        "enclosing_class": "SecretHolder",
    }

    helper = synthesize_shared_helper_code(u1, u2, method_binding="method", repo_root=str(tmp_path))
    assert "def _shared_reveal_first_reveal_second(self) -> str:" in helper
    assert "return prefix + self.__token" in helper

    patch = generate_refactoring_patch(
        [(1.0, u1, u2)],
        repo_root=str(tmp_path),
        method_binding="method",
        replace_clones=True,
    )
    assert patch != ""
    assert "return self._shared_reveal_first_reveal_second()" in patch

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
            "from secret_holder import SecretHolder; "
            "h = SecretHolder('super_secret_xyz'); "
            "print(h.reveal_first(), h.reveal_second())",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_proc.returncode == 0, f"run failed: {run_proc.stderr}"
    assert run_proc.stdout.strip() == "token:super_secret_xyz token:super_secret_xyz"

def test_batch_83_custom_receiver_name_delegation(tmp_path: Path) -> None:
    """Verifies that custom receiver parameter names (e.g. this, klass) are respected during delegation and helper generation."""
    f = tmp_path / "custom_rec.py"
    code = (
        "class Worker:\n"
        "    def task_a(this, amount: int) -> int:\n"
        "        val = amount * 2\n"
        "        return val\n"
        "\n"
        "    def task_b(this, amount: int) -> int:\n"
        "        val = amount * 2\n"
        "        return val\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "task_a",
        "kind": "function",
        "enclosing_class": "Worker",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "task_b",
        "kind": "function",
        "enclosing_class": "Worker",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "+    def _shared_task_a_task_b(this, amount: int) -> int:" in patch
    assert "+        return this._shared_task_a_task_b(amount)" in patch

def test_batch_83_method_without_positional_params_falls_back_to_module(tmp_path: Path) -> None:
    """Verifies that methods without positional receiver parameters fall back defensively to module binding."""
    f = tmp_path / "no_receiver.py"
    code = (
        "class Utility:\n"
        "    def no_rec_a():\n"
        "        base = 10\n"
        "        return base + 1\n"
        "\n"
        "    def no_rec_b():\n"
        "        base = 10\n"
        "        return base + 1\n"
    )
    f.write_text(code, encoding="utf-8")

    u1 = {
        "file": str(f),
        "start": 2,
        "end": 4,
        "name": "no_rec_a",
        "kind": "function",
        "enclosing_class": "Utility",
    }
    u2 = {
        "file": str(f),
        "start": 6,
        "end": 8,
        "name": "no_rec_b",
        "kind": "function",
        "enclosing_class": "Utility",
    }

    patch = generate_refactoring_patch([(0.95, u1, u2)], repo_root=str(tmp_path), replace_clones=True)
    assert patch
    assert "+def _shared_no_rec_a_no_rec_b(" in patch
    assert "+        return _shared_no_rec_a_no_rec_b()" in patch

def test_batch_83_classmethod_receiver_ordering_cls_before_self() -> None:
    """Verifies that _format_helper_parameters sorts 'cls' before 'self' when primary_receiver is 'cls'."""
    from pydoppelgangerhunt.fixer.synthesis import _format_helper_parameters  # pylint: disable=import-outside-toplevel

    inputs = ["self", "cls", "data"]
    meta = {
        "self": {"type": "Any", "kind": "pos"},
        "cls": {"type": "Any", "kind": "pos"},
        "data": {"type": "dict", "kind": "pos"},
    }

    params_cls = _format_helper_parameters(
        inputs,
        meta,
        meta,
        effective_binding="method",
        is_static_clone=False,
        is_class_receiver=True,
        type_merge_strategy="union",
        receiver_param="cls",
    )
    assert params_cls == ["cls", "self: Any", "data: dict"]

    params_self = _format_helper_parameters(
        inputs,
        meta,
        meta,
        effective_binding="method",
        is_static_clone=False,
        is_class_receiver=False,
        type_merge_strategy="union",
        receiver_param="self",
    )
    assert params_self == ["self", "cls: Any", "data: dict"]

def test_batch_83_populate_receiver_param_when_kind_present(tmp_path: Path) -> None:
    """Verifies that _populate_unit_receiver_metadata populates receiver_param even if receiver_kind was already present."""
    from pydoppelgangerhunt.fixer.binding import _populate_unit_receiver_metadata  # pylint: disable=import-outside-toplevel

    f = tmp_path / "mod.py"
    f.write_text(
        "class Worker:\n"
        "    def run(this, val: int) -> int:\n"
        "        return val * 2\n",
        encoding="utf-8",
    )
    unit = {
        "file": str(f),
        "start": 2,
        "end": 3,
        "enclosing_class": "Worker",
        "enclosing_class_start": 1,
        "receiver_kind": "instance",
    }
    _populate_unit_receiver_metadata(unit, repo_root=str(tmp_path))
    assert unit.get("receiver_param") == "this"

def test_batch_83_normalize_receiver_order_custom_receiver() -> None:
    """Verifies that _normalize_receiver_order properly orders custom receiver names (e.g. this, klass)."""
    from pydoppelgangerhunt.fixer.scope import _normalize_receiver_order  # pylint: disable=import-outside-toplevel

    inputs_inst = ["x", "y", "this"]
    _normalize_receiver_order(inputs_inst, has_instance_binding=True, has_class_binding=False, primary_receiver="this")
    assert inputs_inst == ["this", "x", "y"]

    inputs_cls = ["a", "b", "klass"]
    _normalize_receiver_order(inputs_cls, has_instance_binding=False, has_class_binding=True, primary_receiver="klass")
    assert inputs_cls == ["klass", "a", "b"]
