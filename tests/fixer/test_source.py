"""Unit tests for fixer source token slicing, comments/pragmas, and import manipulation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pydoppelgangerhunt import (
    analyze_unit_variable_scope,
    extract_unit_source_code,
    replace_unit_in_source,
    synthesize_shared_helper_code,
)
from pydoppelgangerhunt.fixer import (  # pylint: disable=protected-access
    _detect_indent_step,
    _extract_required_typing_imports,
    _extract_unit_body_lines,
    _find_module_helper_insertion_index,
    _get_module_imported_names,
    _insert_imports_into_module,
)


def test_fixer_feedback_and_advanced_robustness(tmp_path: Path) -> None:
    """Verifies nested nonlocals, typed generator inference, import dedup, and AST typing extraction."""
    # 1. Nested function scoping with nonlocals (mutated vs read-only)
    file_nested = tmp_path / "nested_nonlocal.py"
    file_nested.write_text(
        "def outer():\n"
        "    count = 0\n"
        "    def inner():\n"
        "        nonlocal count\n"
        "        count += 1\n"
        "    return inner\n",
        encoding="utf-8",
    )
    u_inner = {
        "file": str(file_nested),
        "start": 3,
        "end": 5,
        "name": "outer:inner",
        "kind": "function",
    }
    scope_inner = analyze_unit_variable_scope(u_inner)
    assert "count" in scope_inner["outputs"]

    file_ro = tmp_path / "nested_nonlocal_ro.py"
    file_ro.write_text(
        "def outer():\n"
        "    base = 10\n"
        "    def inner(x):\n"
        "        nonlocal base\n"
        "        return x + base\n"
        "    return inner\n",
        encoding="utf-8",
    )
    u_ro = {
        "file": str(file_ro),
        "start": 3,
        "end": 5,
        "name": "outer:inner",
        "kind": "function",
    }
    scope_ro = analyze_unit_variable_scope(u_ro)
    assert "base" in scope_ro["inputs"]
    assert "base" not in scope_ro["outputs"]

    # 2. Generator return type inference for yield from and typed yield
    file_gen_from = tmp_path / "gen_yield_from.py"
    file_gen_from.write_text(
        "def stream_items(src: List[str]):\n"
        "    yield from src\n",
        encoding="utf-8",
    )
    u_gen_from = {
        "file": str(file_gen_from),
        "start": 1,
        "end": 2,
        "name": "stream_items",
        "kind": "function",
    }
    helper_gen_from = synthesize_shared_helper_code(u_gen_from, u_gen_from, include_imports=True)
    assert "-> Iterator[str]:" in helper_gen_from
    assert "from typing import Iterator, List" in helper_gen_from

    file_gen_val = tmp_path / "gen_yield_val.py"
    file_gen_val.write_text(
        "def yield_single(val: int):\n"
        "    yield val\n",
        encoding="utf-8",
    )
    u_gen_val = {
        "file": str(file_gen_val),
        "start": 1,
        "end": 2,
        "name": "yield_single",
        "kind": "function",
    }
    helper_gen_val = synthesize_shared_helper_code(u_gen_val, u_gen_val)
    assert "-> Iterator[int]:" in helper_gen_val

    # 3. Import deduplication in _insert_imports_into_module
    orig_lines = ["from typing import Tuple\n", "import os\n"]
    res_dedup = _insert_imports_into_module(orig_lines, ["from typing import Tuple"])
    assert res_dedup == orig_lines

    new_lines = _insert_imports_into_module(orig_lines, ["from typing import List"])
    assert "from typing import List\n" in new_lines

    # 4. AST-based typing symbol extraction with shadowing parameter immunity
    shadow_func = "def helper(Tuple: int, List: str) -> None:\n    pass\n"
    extracted_shadow = _extract_required_typing_imports(shadow_func)
    assert "Tuple" not in extracted_shadow
    assert "List" not in extracted_shadow

    real_func = "def helper(items: List[str]) -> Tuple[int, Optional[str]]:\n    pass\n"
    extracted_real = _extract_required_typing_imports(real_func)
    assert extracted_real == ["List", "Optional", "Tuple"]

    # Raw signature without wrapper
    sig_raw = "x: Dict[str, Any]"
    extracted_sig = _extract_required_typing_imports(sig_raw)
    assert "Dict" in extracted_sig
    assert "Any" in extracted_sig

def test_slice_source_by_token_range() -> None:
    """Tests precise character/token-range slicing across lines and columns."""
    from pydoppelgangerhunt import slice_source_by_token_range  # pylint: disable=import-outside-toplevel

    code = (
        "def example(alpha: int, beta: str) -> None:\n"
        "    first_val = 10; second_val = 20  # inline\n"
        "    return None\n"
    )

    # 1. Single-line slice: "first_val = 10" is columns 4 to 18 on line 2
    sl1 = slice_source_by_token_range(code, 2, 4, 2, 18)
    assert sl1 == "first_val = 10"

    # 2. Multi-line slice
    sl2 = slice_source_by_token_range(code, 1, 4, 2, 18)
    assert sl2.startswith("example(alpha: int, beta: str) -> None:\n    first_val = 10")

    # 3. None end_col includes through end of line
    sl3 = slice_source_by_token_range(code, 3, 4, 3, None)
    assert sl3 == "return None\n"

    # 4. Out of bounds and edge cases
    assert slice_source_by_token_range("", 1, 0, 1, 5) == ""
    assert slice_source_by_token_range(code, 0, 0, 1, 5) == ""
    assert slice_source_by_token_range(code, 5, 0, 10, 5) == ""
    assert slice_source_by_token_range(code, 3, 0, 2, 5) == ""

def test_extract_unit_comments_and_pragmas() -> None:
    """Tests extraction and classification of comments, # type: ignore, and # noqa pragmas."""
    from pydoppelgangerhunt import extract_unit_comments_and_pragmas  # pylint: disable=import-outside-toplevel

    code = (
        "# Top comment\n"
        "# Another header\n"
        "def compute():\n"
        "    x = 1  # inline note\n"
        "    y = 2  # type: ignore[assignment]\n"
        "    z = 3  # noqa: E501\n"
        "    w = 4  # pylint: disable=unused-variable\n"
    )

    # Harvest lines 3 through 7
    comments = extract_unit_comments_and_pragmas(code, 3, 7)
    assert len(comments) == 4
    texts = [c["text"] for c in comments]
    assert "# inline note" in texts
    assert "# type: ignore[assignment]" in texts
    assert "# noqa: E501" in texts
    assert "# pylint: disable=unused-variable" in texts

    # Verify classification
    pragmas = [c for c in comments if c["is_pragma"]]
    assert len(pragmas) == 3
    kinds = {c["pragma_kind"] for c in pragmas}
    assert kinds == {"type_ignore", "noqa", "pylint"}

    # Leading comments harvesting
    leading = extract_unit_comments_and_pragmas(code, 3, 4, include_leading=True)
    leading_texts = [c["text"] for c in leading]
    assert "# Top comment" in leading_texts
    assert "# Another header" in leading_texts

    # Empty code
    assert extract_unit_comments_and_pragmas("", 1, 5) == []

def test_find_module_helper_insertion_index_shebang_and_encoding() -> None:
    """Verifies that module helper insertion respects shebang and encoding cookies."""
    lines_both = [
        "#!/usr/bin/env python3\n",
        "# -*- coding: utf-8 -*-\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_both) == 2

    lines_shebang = [
        "#!/usr/bin/env python3\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_shebang) == 1

    lines_encoding = [
        "# coding=utf-8\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_encoding) == 1

    lines_comment_then_encoding = [
        "# Ordinary comment\n",
        "# -*- coding: utf-8 -*-\n",
        "x = 1\n",
    ]
    assert _find_module_helper_insertion_index(lines_comment_then_encoding) == 2

    lines_syntax_error = [
        "#!/usr/bin/env python3\n",
        "# coding=utf-8\n",
        "invalid ? ? ?\n",
    ]
    assert _find_module_helper_insertion_index(lines_syntax_error) == 2

def test_detect_indent_step_multi_level_two_spaces() -> None:
    """Verifies that _detect_indent_step correctly handles 2-space indentation at various nesting levels."""
    assert _detect_indent_step("  ") == "  "
    assert _detect_indent_step("    ") == "    "
    assert _detect_indent_step("      ") == "  "
    assert _detect_indent_step("        ") == "    "
    assert _detect_indent_step("          ") == "  "
    assert _detect_indent_step("\t\t") == "\t"

def test_insert_imports_raw_and_unicode_docstrings() -> None:
    """Verifies that raw (r\"\"\") and unicode (u\"\"\") docstrings are preserved ahead of inserted imports."""
    raw_lines = [
        'r"""Raw module docstring with \\s+ escapes."""\n',
        "x = 1\n",
    ]
    res_raw = _insert_imports_into_module(raw_lines, ["from typing import Any"])
    assert res_raw[0].startswith('r"""')
    assert "from typing import Any\n" in res_raw
    assert res_raw.index("from typing import Any\n") > 0

    uni_lines = [
        'u"""Unicode module docstring with unicode text."""\n',
        "x = 1\n",
    ]
    res_uni = _insert_imports_into_module(uni_lines, ["from typing import Any"])
    assert res_uni[0].startswith('u"""')
    assert res_uni.index("from typing import Any\n") > 0

def test_extract_required_typing_imports_comprehensive() -> None:
    """Verifies that standard library typing symbols such as Mapping and Literal are extracted."""
    sig = "(config: Mapping[str, Any], mode: Literal['fast', 'slow']) -> Optional[Tuple[int, ...]]"
    needed = _extract_required_typing_imports(sig)
    for expected in ["Any", "Literal", "Mapping", "Optional", "Tuple"]:
        assert expected in needed

def test_module_helper_insertion_index_stops_at_first_non_import() -> None:
    """Verifies that module helper insertion index terminates at the initial import block boundary."""
    code = (
        '"""Module docstring."""\n'
        "import sys\n"
        "import os\n\n"
        "class Config:\n"
        "    val = 42\n\n"
        "import math\n"
    )
    lines = code.splitlines(keepends=True)
    idx = _find_module_helper_insertion_index(lines)
    # Must be after import os (line 3), before class Config (line 5), not after import math (line 8)
    assert idx == 3

def test_extract_unit_source_code_empty_and_directory_path(tmp_path: Path) -> None:
    """Verifies that extract_unit_source_code safely handles empty paths and directory paths."""
    u_empty = {"file": "", "name": "dummy", "start": 1, "end": 5}
    assert extract_unit_source_code(u_empty) == ["# Source for dummy lines 1-5\n"]

    u_dir = {"file": str(tmp_path), "name": "dir_unit", "start": 1, "end": 3}
    assert extract_unit_source_code(u_dir) == ["# Source for dir_unit lines 1-3\n"]

def test_expression_unit_midline_pragma_placement() -> None:
    """Verifies that boundary pragmas are appended to the line end when trailing code is present."""
    source = "call([x for x in data], extra_arg)  # type: ignore\n"
    unit = {
        "kind": "comprehension",
        "start": 1,
        "end": 1,
        "start_col": 5,
        "end_col": 22,
    }
    rep = "_shared(data)"
    result = replace_unit_in_source(source, unit, rep, preserve_boundary_pragmas=True)
    assert "extra_arg" in result
    assert result.endswith("  # type: ignore\n")
    assert result == "call(_shared(data), extra_arg)  # type: ignore\n"

def test_extract_unit_body_lines_single_line_function() -> None:
    """Verifies that _extract_unit_body_lines strips single-line function headers."""
    unit = {"kind": "function", "start": 1, "end": 1, "name": "foo"}
    raw_lines = ["def foo(x: int) -> int: return x * 2\n"]
    extracted = _extract_unit_body_lines(unit, raw_lines)
    assert extracted == ["return x * 2"]

def test_insert_imports_into_module_deduplicates_within_import_lines() -> None:
    """Verifies that _insert_imports_into_module eliminates duplicate imports present in import_lines."""
    from pydoppelgangerhunt.fixer import _insert_imports_into_module  # pylint: disable=import-outside-toplevel

    orig = ["def foo(): pass\n"]
    imports = ["from typing import Any", "from typing import Any", "from typing import Tuple"]
    res = _insert_imports_into_module(orig, imports)
    assert res.count("from typing import Any\n") == 1
    assert res.count("from typing import Tuple\n") == 1

def test_extract_unit_body_lines_and_scope_method_kind(tmp_path: Path) -> None:
    """Verifies that units with kind='method' strip headers/docstrings and preserve method parameters."""
    from pydoppelgangerhunt.fixer import _extract_unit_body_lines  # pylint: disable=import-outside-toplevel

    code = (
        "class Calculator:\n"
        "    def compute(self, a: int, b: int) -> int:\n"
        "        '''Compute sum.'''\n"
        "        total = a + b\n"
        "        return total\n"
    )
    f = tmp_path / "calc.py"
    f.write_text(code, encoding="utf-8")

    unit = {
        "file": str(f),
        "start": 2,
        "end": 5,
        "name": "Calculator:compute",
        "kind": "method",
    }
    raw_lines = code.splitlines()[1:]
    body_lines = _extract_unit_body_lines(unit, raw_lines)
    assert body_lines == ["total = a + b", "return total"]

    scope = analyze_unit_variable_scope(unit, repo_root=str(tmp_path))
    assert "self" in scope["inputs"]
    assert "a" in scope["inputs"]
    assert "b" in scope["inputs"]
    assert "total" in scope["outputs"]
    assert scope["return_type"] == "int"

def test_find_module_helper_insertion_index_with_try_except_and_conditional_imports() -> None:
    """Verifies that _find_module_helper_insertion_index places helpers after try/except and conditional imports."""
    from pydoppelgangerhunt.fixer import _find_module_helper_insertion_index  # pylint: disable=import-outside-toplevel

    lines = [
        '"""Module docstring."""\n',
        "import os\n",
        "try:\n",
        "    import tomllib\n",
        "except ImportError:\n",
        "    import tomli as tomllib\n",
        "import sys\n",
        "\n",
        "def existing_fn():\n",
        "    pass\n",
    ]
    idx = _find_module_helper_insertion_index(lines)
    # The last import is 'import sys' at line 7 (1-indexed). The helper should be inserted at index 7 (line 7), before line 9.
    assert idx == 7

def test_is_docstring_node_and_extract_docstring_end_line() -> None:
    """Verifies that _is_docstring_node and _extract_docstring_end_line accurately detect docstrings."""
    from pydoppelgangerhunt.fixer import (  # pylint: disable=import-outside-toplevel
        _extract_docstring_end_line,
        _is_docstring_node,
    )

    # 1. Non-docstrings: None, Call, Constant int
    assert not _is_docstring_node(None)
    assert not _is_docstring_node(ast.parse("pass").body[0])
    assert not _is_docstring_node(ast.parse("42").body[0])
    assert not _is_docstring_node(ast.parse("print('hi')").body[0])

    # 2. String constant docstring node
    doc_node = ast.parse("'''docstring'''").body[0]
    assert _is_docstring_node(doc_node)

    # 3. _extract_docstring_end_line on module and function with multiline docstring
    tree_multiline = ast.parse('"""Line 1\nLine 2\nLine 3"""\nx = 1\n')
    assert _extract_docstring_end_line(tree_multiline) == 3

    # 4. _extract_docstring_end_line with no docstring
    tree_no_doc = ast.parse("x = 1\ny = 2\n")
    assert _extract_docstring_end_line(tree_no_doc) == 0

    # 5. _extract_docstring_end_line on a function node
    fn_tree = ast.parse("def f():\n    '''Function docstring.'''\n    return 10\n")
    assert _extract_docstring_end_line(fn_tree.body[0]) == 2

def test_batch_82_get_module_imported_names_guarded_imports() -> None:
    """Verifies that _get_module_imported_names inspects top-level If and Try import blocks."""
    from pydoppelgangerhunt.fixer import _get_module_imported_names  # pylint: disable=import-outside-toplevel

    code = (
        "import sys\n"
        "from os import path\n"
        "if TYPE_CHECKING:\n"
        "    from typing import Optional, List\n"
        "try:\n"
        "    from collections.abc import Sequence\n"
        "except ImportError:\n"
        "    from typing import Sequence\n"
    )
    imported = _get_module_imported_names(code)
    assert "sys" in imported
    assert "path" in imported
    assert "Optional" in imported
    assert "List" in imported
    assert "Sequence" in imported

    unconditional = _get_module_imported_names(code, include_conditional=False)
    assert "sys" in unconditional
    assert "path" in unconditional
    assert "Optional" not in unconditional
    assert "List" not in unconditional
    assert "Sequence" not in unconditional

def test_batch_82_slice_unit_token_lines_single_line_bounds() -> None:
    """Verifies defensive single-line column bounds check in _slice_unit_token_lines."""
    from pydoppelgangerhunt.fixer import _slice_unit_token_lines  # pylint: disable=import-outside-toplevel

    line = ["    total = sum(x for x in data)"]
    # Normal column slice
    res_normal = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 32},
        list(line),
    )
    assert res_normal == ["sum(x for x in data)"]

    # Inverted column bounds (e_col <= s_col) must not collapse the line to empty
    res_inverted = _slice_unit_token_lines(
        {"kind": "comprehension", "start_col": 12, "end_col": 5},
        list(line),
    )
    assert res_inverted == ["sum(x for x in data)"]


def test_insert_imports_orders_future_annotations_first() -> None:
    """Verifies _insert_imports_into_module ensures from __future__ is placed before other imports."""
    orig_lines = [
        "\"\"\"Docstring.\"\"\"\n",
        "\n",
        "import os\n",
    ]
    imports_to_add = [
        "import sys",
        "from __future__ import annotations",
        "from typing import List",
    ]
    result = _insert_imports_into_module(orig_lines, imports_to_add)
    result_text = "".join(result)
    fut_pos = result_text.find("from __future__ import annotations")
    sys_pos = result_text.find("import sys")
    typing_pos = result_text.find("from typing import List")
    assert fut_pos != -1
    assert sys_pos != -1
    assert typing_pos != -1
    assert fut_pos < sys_pos
    assert fut_pos < typing_pos


def test_get_module_imported_names_dotted_imports() -> None:
    """Verifies that dotted imports record both the root bound identifier and full module name."""
    src = (
        "import os.path\n"
        "import xml.etree.ElementTree as ET\n"
        "from urllib.parse import urlparse\n"
        "if True:\n"
        "    import posixpath.constants\n"
    )
    names = _get_module_imported_names(src, include_conditional=True)
    assert "os" in names
    assert "os.path" in names
    assert "ET" in names
    assert "urlparse" in names
    assert "posixpath" in names
    assert "posixpath.constants" in names


def test_insert_imports_into_module_corrupt_source_fallback() -> None:
    """Verifies that if source code causes ast.parse to raise ValueError, manual parser falls back gracefully."""
    corrupt_lines = [
        "\"\"\"Docstring with null\x00byte.\"\"\"\n",
        "def compute():\n",
        "    return 42\n",
    ]
    imports = ["import math"]
    res = _insert_imports_into_module(corrupt_lines, imports)
    res_text = "".join(res)
    assert "import math" in res_text


